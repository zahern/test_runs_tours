"""
15_nearmiss_ecf_od_skim.py
Build a genuinely OD-pair-specific near-miss / Expected Crash Frequency (ECF)
safety skim, replacing the flat nearmiss_skim_507.omx (which was just a
per-zone count vector broadcast across rows/columns via np.tile — see
13_nearmiss_skim_matrix.py — so every OD pair sharing an origin or
destination zone got an identical value, regardless of which OD pair it was).

Method: build a real road network graph from the 10,302 LineString segments
in SEQ_crashes_nearmiss_embedded.gpkg (the same per-segment ExpCrashes /
ObsCrashes / nm_total / nm_per_km / safety_index data documented in
near_miss_ecf.csv, but with segment geometry attached), run shortest-path
routing (by segment length) from every one of the 507 SA2 zone centroids,
and accumulate each segment's safety exposure along the actual shortest path
to every other zone — so OD pairs along genuinely riskier corridors get a
higher exposure value, and OD pairs that don't share any road get a
different value than OD pairs that do.

Outputs:
    nearmiss_ecf_skim_507.omx — OD-specific matrices:
        ecf_total         : sum of ExpCrashes along the route (the genuine
                             "ECF" — Expected Crash Frequency — safety
                             component requested)
        nm_total           : sum of near-miss event counts along the route
        nm_{am,op,pm,ev}   : time-period-specific near-miss counts along route
        safety_index_total : sum of per-segment safety_index along route
        route_length_km    : total routed distance (for normalising to a rate)
        ecf_per_km          : ecf_total / route_length_km
        nm_per_km_total     : nm_total / route_length_km
    nearmiss_ecf_od_507_summary.csv — coverage / top-OD-pair diagnostics
"""
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEMAND_DIR = r"Z:\test_runs_tours\demand_"
GPKG_PATH = os.path.join(DEMAND_DIR, "SEQ_crashes_nearmiss_embedded.gpkg")
ZONE_LIST_507 = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
CROSSWALK_507 = os.path.join(DEMAND_DIR, "sa2_crosswalk_507.csv")
OUT_OMX = os.path.join(DEMAND_DIR, "nearmiss_ecf_skim_507.omx")
OUT_SUMMARY = os.path.join(DEMAND_DIR, "nearmiss_ecf_od_507_summary.csv")

NODE_SNAP_TOL_M = 100.0  # merge segment endpoints within this distance into one node.
# The raw segment endpoints don't always coincide exactly at real intersections
# (off by tens-to-hundreds of metres between adjacent ~1km-diced segments) —
# a tight tolerance fragments the network into many small disconnected
# components and silently zeroes out most OD pairs. 100m balances connecting
# genuine intersections against false-merging distinct ones (checked via
# connected-component analysis: 2m tolerance -> 68.7% of nodes in the largest
# component, 100m -> ~90%+).
ZONE_SNAP_MAX_M = 5000  # max distance to snap a zone centroid onto the network


def main():
    with open(ZONE_LIST_507) as f:
        zone_ids_507 = sorted(int(x) for x in re.split(r"[\s,]+", f.read().strip()) if x)
    n_zones = len(zone_ids_507)
    print(f"Target SA2 zones: {n_zones}")

    print("Loading road network...")
    gdf = gpd.read_file(GPKG_PATH, layer="roads_with_nearmiss")
    print(f"  {len(gdf)} segments, CRS={gdf.crs}")

    # ── 1. Build node set from segment endpoints (snap within NODE_SNAP_TOL_M) ──
    starts = np.array([(g.coords[0][0], g.coords[0][1]) for g in gdf.geometry])
    ends = np.array([(g.coords[-1][0], g.coords[-1][1]) for g in gdf.geometry])
    all_pts = np.vstack([starts, ends])

    print("Snapping segment endpoints into shared nodes...")
    tree = cKDTree(all_pts)
    pairs = tree.query_pairs(r=NODE_SNAP_TOL_M)
    # Union-find to merge close points into single node ids
    parent = list(range(len(all_pts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)
    raw_node_id = np.array([find(i) for i in range(len(all_pts))])
    unique_raw, node_id = np.unique(raw_node_id, return_inverse=True)
    n_nodes = len(unique_raw)
    node_coords = np.array([all_pts[raw_node_id == r][0] for r in unique_raw])
    print(f"  {n_nodes} unique nodes from {len(gdf)*2} endpoints")

    start_node = node_id[: len(gdf)]
    end_node = node_id[len(gdf):]

    # ── 2. Per-segment exposure attributes ──────────────────────────────────
    length_km = (gdf["segment_length_m"].fillna(0).astype(float) / 1000.0).values
    exp_crashes = gdf["ExpCrashes"].fillna(0).astype(float).values
    nm_total = gdf["nm_total"].fillna(0).astype(float).values
    nm_am = gdf["nm_am"].fillna(0).astype(float).values if "nm_am" in gdf.columns else np.zeros(len(gdf))
    nm_op = gdf["nm_op"].fillna(0).astype(float).values if "nm_op" in gdf.columns else np.zeros(len(gdf))
    nm_pm = gdf["nm_pm"].fillna(0).astype(float).values if "nm_pm" in gdf.columns else np.zeros(len(gdf))
    nm_ev = gdf["nm_ev"].fillna(0).astype(float).values if "nm_ev" in gdf.columns else np.zeros(len(gdf))
    safety_index = gdf["safety_index"].fillna(0).astype(float).values

    # ── 3. Build routing graph (weight = length_km) + edge exposure lookup ──
    print("Building routing graph...")
    row = np.concatenate([start_node, end_node])
    col = np.concatenate([end_node, start_node])
    weight = np.concatenate([length_km, length_km])
    weight = np.where(weight <= 0, 0.001, weight)  # avoid zero-weight edges
    graph = csr_matrix((weight, (row, col)), shape=(n_nodes, n_nodes))

    # edge -> exposure lookup (undirected, both directions point to same segment)
    edge_exposure = {}
    for i in range(len(gdf)):
        u, v = start_node[i], end_node[i]
        vals = (exp_crashes[i], nm_total[i], nm_am[i], nm_op[i], nm_pm[i], nm_ev[i],
                safety_index[i], length_km[i])
        edge_exposure[(u, v)] = vals
        edge_exposure[(v, u)] = vals

    # ── 4. Snap zone centroids to nearest network node ──────────────────────
    print("Snapping zone centroids to network...")
    crosswalk = pd.read_csv(CROSSWALK_507)
    crosswalk["sa2_code"] = crosswalk["sa2_code"].astype(int)
    crosswalk = crosswalk.set_index("sa2_code").reindex(zone_ids_507)

    zone_gdf = gpd.GeoDataFrame(
        crosswalk.reset_index(),
        geometry=gpd.points_from_xy(crosswalk["sa2_lon"], crosswalk["sa2_lat"]),
        crs=4326,
    ).to_crs(gdf.crs)
    zone_xy = np.array([(p.x, p.y) for p in zone_gdf.geometry])

    node_tree = cKDTree(node_coords)
    zone_dist, zone_node = node_tree.query(zone_xy)
    n_far = int((zone_dist > ZONE_SNAP_MAX_M).sum())
    if n_far:
        print(f"  WARNING: {n_far} zones snap >{ZONE_SNAP_MAX_M}m from the network "
              f"(missing/disconnected geometry) — their OD rows will be NaN")
    zone_node = np.where(zone_dist > ZONE_SNAP_MAX_M, -1, zone_node)

    # ── 5. Multi-source Dijkstra by length, with predecessors ───────────────
    print(f"Running Dijkstra from {n_zones} zone sources over {n_nodes} nodes...")
    valid_sources = [n for n in zone_node if n >= 0]
    dist_matrix, predecessors = dijkstra(
        graph, directed=False, indices=valid_sources, return_predecessors=True
    )

    src_to_row = {n: i for i, n in enumerate(valid_sources)}

    # ── 6. Accumulate exposure along the shortest-path tree per source ──────
    # For each source, process nodes in increasing distance order so a node's
    # cumulative exposure = predecessor's cumulative exposure + this edge's
    # exposure (avoids re-walking full paths for every OD pair).
    print("Accumulating exposure along shortest-path trees...")
    n_metrics = 8  # exp_crashes, nm_total, nm_am, nm_op, nm_pm, nm_ev, safety_index, length_km
    zone_to_zone = {m: np.full((n_zones, n_zones), np.nan, dtype=np.float64) for m in range(n_metrics)}

    for zi, src_node in enumerate(zone_node):
        if src_node < 0:
            continue
        row_i = src_to_row[src_node]
        dist_row = dist_matrix[row_i]
        pred_row = predecessors[row_i]

        order = np.argsort(dist_row)
        cum = np.zeros((n_nodes, n_metrics), dtype=np.float64)
        reached = np.zeros(n_nodes, dtype=bool)
        reached[src_node] = True
        for node in order:
            if node == src_node or not np.isfinite(dist_row[node]):
                continue
            p = pred_row[node]
            if p < 0:
                continue
            vals = edge_exposure.get((p, node))
            if vals is None:
                continue
            cum[node] = cum[p] + np.array(vals)
            reached[node] = True

        for zj, dst_node in enumerate(zone_node):
            if dst_node < 0 or zi == zj:
                continue
            if not reached[dst_node]:
                continue
            for m in range(n_metrics):
                zone_to_zone[m][zi, zj] = cum[dst_node, m]

        if zi % 50 == 0:
            print(f"  source zone {zi + 1}/{n_zones} done")

    metric_names = ["ecf_total", "nm_total", "nm_am", "nm_op", "nm_pm", "nm_ev",
                     "safety_index_total", "route_length_km"]
    matrices = {name: zone_to_zone[m].astype(np.float32) for m, name in enumerate(metric_names)}

    with np.errstate(invalid="ignore", divide="ignore"):
        ecf_per_km = np.where(matrices["route_length_km"] > 0,
                              matrices["ecf_total"] / matrices["route_length_km"], 0.0)
        nm_per_km_total = np.where(matrices["route_length_km"] > 0,
                                   matrices["nm_total"] / matrices["route_length_km"], 0.0)
    matrices["ecf_per_km"] = np.nan_to_num(ecf_per_km).astype(np.float32)
    matrices["nm_per_km_total"] = np.nan_to_num(nm_per_km_total).astype(np.float32)
    for name in metric_names:
        matrices[name] = np.nan_to_num(matrices[name], nan=0.0).astype(np.float32)

    n_od_covered = int(np.count_nonzero(~np.isnan(zone_to_zone[0])))
    print(f"\nOD pairs with a valid route: {n_od_covered}/{n_zones * (n_zones - 1)}")

    import sys
    sys.path.insert(0, DEMAND_DIR)
    from omx_utils import write_omx
    zone_mapping = {str(z): i for i, z in enumerate(zone_ids_507)}
    write_omx(OUT_OMX, matrices, zone_mapping=zone_mapping, shape=(n_zones, n_zones))
    print(f"Written: {OUT_OMX}")

    # ── Summary diagnostics ──────────────────────────────────────────────────
    od_rows = []
    ecf_mat = matrices["ecf_total"]
    for zi, oz in enumerate(zone_ids_507):
        for zj, dz in enumerate(zone_ids_507):
            if zi != zj and ecf_mat[zi, zj] > 0:
                od_rows.append((oz, dz, float(ecf_mat[zi, zj]), float(matrices["nm_total"][zi, zj]),
                                float(matrices["route_length_km"][zi, zj])))
    od_df = pd.DataFrame(od_rows, columns=["origin_sa2", "dest_sa2", "ecf_total", "nm_total", "route_length_km"])
    od_df.to_csv(OUT_SUMMARY, index=False)
    print(f"Written: {OUT_SUMMARY} ({len(od_df):,} OD pairs)")
    print("\nTop 10 OD pairs by ecf_total:")
    print(od_df.nlargest(10, "ecf_total").to_string(index=False))


if __name__ == "__main__":
    main()
