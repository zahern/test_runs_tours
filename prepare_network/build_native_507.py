"""
build_native_507.py

Native rebuild of the DynusT model's zone.dat, network.dat, origin.dat,
destination.dat and demand.dat to literally use 507 zones (SA2-based),
replacing the prior SuperZone.dat aggregation-only approach.

demand_truck.dat / demand_HOV.dat are written empty (0 O-D matrices):
scenario.dat only loads 1 vehicle type (PC) from demand.dat, so these
files are not functionally read regardless -- confirmed by reading the
live model's scenario.dat directly.

Reuses the crosswalk (old_zone_id -> sa2_code -> super_zone_id 1..507)
already built and validated by build_superzone_507.py. Writes everything
to OUT_DIR; never touches the live model directory.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
from scipy.spatial import cKDTree

MODEL_DIR = r"C:\Users\ahernz\SEQ_DTA_2025"
# IMPORTANT: origin.dat/destination.dat/network.dat in MODEL_DIR are now the
# already-converted 507-zone outputs from a prior run of this script (the
# live model directory was already swapped). Source the original
# (pre-conversion, 4654-zone) versions from the backup made just before that
# swap -- NOT from the live directory, which would self-reference its own
# already-remapped output on any re-run.
BACKUP_DIR = os.path.join(MODEL_DIR, "pre_native_507_backup")
# pre_507_conversion_backup is the very first, fully-pristine backup (agg
# flag still 0); pre_native_507_backup only has origin.dat/destination.dat
# since those were never touched by the earlier SuperZone-only pass.
BACKUP_NETWORK = os.path.join(MODEL_DIR, "pre_507_conversion_backup", "network.dat")
ORIGIN_DAT = os.path.join(BACKUP_DIR, "origin.dat")
DESTINATION_DAT = os.path.join(BACKUP_DIR, "destination.dat")
CROSSWALK_CSV = os.path.join(MODEL_DIR, "zone_crosswalk_507.csv")

SA2_SHP = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
DEMAND_OMX = r"Z:\test_runs_tours\demand_\demand_integrated_sa2.omx"
ZONE_IDS_507_TXT = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
XY_DAT = os.path.join(MODEL_DIR, "xy.dat")  # never touched by any prior step, still original

OUT_DIR = r"Z:\test_runs_tours\prepare_network\new\507_native"
TARGET_CRS = 7856
N_SUPER_ZONES = 507
PEAK_MULT_FACTOR = 0.10  # applied per interval; 2 intervals -> 20% of the
                          # daily total loaded across the 2-hour window
START_TIMES = [0.0, 60.0, 120.0]

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# Crosswalk
# -----------------------------------------------------------------------
def load_crosswalk():
    cw = pd.read_csv(CROSSWALK_CSV)
    old_to_super = dict(zip(cw["old_zone_id"], cw["super_zone_id"]))
    assert set(range(1, N_SUPER_ZONES + 1)) >= set(old_to_super.values())
    return cw, old_to_super


def load_full_507_sa2_table(crosswalk_df):
    # The crosswalk CSV only has rows for old (4654-system) zones, so it only
    # references the 304 SA2 zones that actually overlap this SEQ network --
    # the other 203 statewide SA2 zones never appear as a value in it. zone.dat
    # and demand.dat need geometry/demand for all 507, so rebuild the full
    # sa2_code -> super_zone_id table the same way build_superzone_507.py did
    # (sorted ascending SA2 code -> index+1) and cross-check it agrees with
    # the crosswalk for the 304 zones that are in both.
    with open(ZONE_IDS_507_TXT) as f:
        zone_ids_507 = sorted(int(x) for x in f.read().split() if x.strip())
    assert len(zone_ids_507) == N_SUPER_ZONES
    full = pd.DataFrame({"sa2_code": zone_ids_507})
    full["super_zone_id"] = range(1, N_SUPER_ZONES + 1)

    cw_unique = crosswalk_df.drop_duplicates("sa2_code")[["sa2_code", "super_zone_id"]]
    merged = cw_unique.merge(full, on="sa2_code", suffixes=("_cw", "_full"))
    mismatches = merged[merged["super_zone_id_cw"] != merged["super_zone_id_full"]]
    assert mismatches.empty, f"super_zone_id numbering mismatch:\n{mismatches}"

    return full


# -----------------------------------------------------------------------
# network.dat: parse original (pre-SuperZone) version, remap, rewrite
# -----------------------------------------------------------------------
def parse_network_for_remap(path):
    with open(path) as f:
        lines = f.readlines()
    header = lines[0].split()
    n_zones, n_nodes, n_links, k_paths, agg_flag = (int(x) for x in header)
    node_lines = lines[1:1 + n_nodes]
    link_lines = lines[1 + n_nodes:1 + n_nodes + n_links]
    assert len(link_lines) == n_links, f"expected {n_links} link lines, got {len(link_lines)}"

    node_ids_ordered = []
    node_old_zone = {}
    for line in node_lines:
        nid, zid = (int(x) for x in line.split())
        node_ids_ordered.append(nid)
        node_old_zone[nid] = zid

    return {
        "n_zones_orig": n_zones,
        "n_nodes": n_nodes,
        "n_links": n_links,
        "k_paths": k_paths,
        "node_ids_ordered": node_ids_ordered,
        "node_old_zone": node_old_zone,
        "link_lines": link_lines,
    }


def write_network_dat(path, net, old_to_super):
    with open(path, "w") as f:
        f.write(f"{N_SUPER_ZONES}  {net['n_nodes']}  {net['n_links']}      "
                f"{net['k_paths']}      0\n")
        for nid in net["node_ids_ordered"]:
            old_zone = net["node_old_zone"][nid]
            new_zone = old_to_super[old_zone]
            f.write(f"{nid} {new_zone}\n")
        f.writelines(net["link_lines"])


# -----------------------------------------------------------------------
# origin.dat / destination.dat: parse original, union-merge per new zone
# -----------------------------------------------------------------------
def parse_origin(path):
    with open(path) as f:
        lines = f.readlines()
    zone_links = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        zone_id, count, mode = int(parts[0]), int(parts[1]), int(parts[2])
        links = []
        for j in range(count):
            up, down, w = lines[i + 1 + j].split()
            links.append((int(up), int(down), float(w)))
        zone_links[zone_id] = links
        i += 1 + count
    return zone_links


def parse_destination(path):
    zone_nodes = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            zone_id, count = int(parts[0]), int(parts[1])
            nodes = [int(x) for x in parts[2:2 + count]]
            zone_nodes[zone_id] = nodes
    return zone_nodes


def build_new_origin(old_origin, old_to_super):
    new_links = {z: [] for z in range(1, N_SUPER_ZONES + 1)}
    seen = {z: set() for z in range(1, N_SUPER_ZONES + 1)}
    for old_zone, links in old_origin.items():
        new_zone = old_to_super[old_zone]
        for (up, down, w) in links:
            key = (up, down)
            if key not in seen[new_zone]:
                seen[new_zone].add(key)
                new_links[new_zone].append((up, down, w))
    return new_links


def build_new_destination(old_dest, old_to_super):
    # The manual caps a destination node at 2 owning zones. The live model's
    # *original* destination.dat already violates this for 2 nodes (a
    # pre-existing data quality issue, confirmed by checking how many old
    # zones reference them -- 3 each, not introduced by this remap). Cap
    # node ownership at 2 new zones here so the output is fully compliant
    # rather than just preserving/relocating the pre-existing violation.
    new_nodes = {z: [] for z in range(1, N_SUPER_ZONES + 1)}
    seen = {z: set() for z in range(1, N_SUPER_ZONES + 1)}
    node_owner_count = {}
    dropped = []
    for old_zone, nodes in old_dest.items():
        new_zone = old_to_super[old_zone]
        for n in nodes:
            if n in seen[new_zone]:
                continue
            owners = node_owner_count.get(n, 0)
            if owners >= 2:
                dropped.append((n, old_zone, new_zone))
                continue
            seen[new_zone].add(n)
            new_nodes[new_zone].append(n)
            node_owner_count[n] = owners + 1
    return new_nodes, dropped, node_owner_count


def validate_destination_sharing(new_nodes):
    node_zone_count = {}
    for zone, nodes in new_nodes.items():
        for n in nodes:
            node_zone_count[n] = node_zone_count.get(n, 0) + 1
    return {n: c for n, c in node_zone_count.items() if c > 2}


# -----------------------------------------------------------------------
# Gateway-node loading for dormant zones (no nodes in this SEQ network).
# DynusT hard-requires every declared zone to have >=1 destination
# (confirmed via the live model's ErrorLog.dat: "Each zone needs to have
# at least one dest" x203, one per dormant zone). These zones carry real
# demand (714K trips/day, 8.3% of the total -- mostly regional QLD trips
# heading into SEQ), so rather than zero them out, snap each to its
# nearest real network node -- standard practice for external/gateway
# zones representing trips from outside the detailed study area.
# -----------------------------------------------------------------------
def parse_xy(path):
    df = pd.read_csv(path, header=None, names=["node_id", "x", "y"], sep=r"\s+")
    return df


def build_node_incident_links(link_lines):
    node_links = {}
    for line in link_lines:
        parts = line.split()
        up, down = int(parts[0]), int(parts[1])
        node_links.setdefault(up, []).append((up, down))
        node_links.setdefault(down, []).append((up, down))
    return node_links


def assign_gateway_nodes(dormant_zone_ids, gdf_507_m, xy_df, node_links,
                          new_origin, new_dest, node_owner_count):
    node_ids = xy_df["node_id"].to_numpy()
    coords = xy_df[["x", "y"]].to_numpy()
    tree = cKDTree(coords)

    zone_centroid = {
        int(row["super_zone_id"]): (row.geometry.centroid.x, row.geometry.centroid.y)
        for _, row in gdf_507_m.iterrows()
    }

    assigned = []
    for zone_id in dormant_zone_ids:
        cx, cy = zone_centroid[zone_id]
        # query enough neighbors to find one not already at the 2-owner cap
        k = 10
        chosen_node = None
        while chosen_node is None:
            dists, idxs = tree.query([cx, cy], k=min(k, len(node_ids)))
            for idx in np.atleast_1d(idxs):
                cand = int(node_ids[idx])
                if node_owner_count.get(cand, 0) < 2:
                    chosen_node = cand
                    break
            if chosen_node is None:
                if k >= len(node_ids):
                    raise RuntimeError(f"zone {zone_id}: no available node found "
                                        f"within all {len(node_ids)} nodes")
                k *= 4

        new_dest[zone_id] = [chosen_node]
        node_owner_count[chosen_node] = node_owner_count.get(chosen_node, 0) + 1

        links = node_links.get(chosen_node)
        if not links:
            raise RuntimeError(f"zone {zone_id}: node {chosen_node} has no incident links")
        new_origin[zone_id] = [links[0] + (0.0,)]

        assigned.append((zone_id, chosen_node))

    return assigned


def write_origin_dat(path, new_links):
    with open(path, "w") as f:
        for z in range(1, N_SUPER_ZONES + 1):
            links = new_links[z]
            f.write(f"{z} {len(links)} 0\n")
            for (up, down, w) in links:
                f.write(f"{up} {down} {w:.3f}\n")


def write_destination_dat(path, new_nodes):
    with open(path, "w") as f:
        for z in range(1, N_SUPER_ZONES + 1):
            nodes = new_nodes[z]
            f.write(f"{z} {len(nodes)} " + " ".join(str(n) for n in nodes) + "\n")


# -----------------------------------------------------------------------
# zone.dat: 507 SA2 boundary polygons, numbered per the crosswalk
# -----------------------------------------------------------------------
def load_507_polygons_with_crosswalk_numbering(full_507_df):
    gdf = gpd.read_file(SA2_SHP)
    gdf["SA2_CODE21"] = gdf["SA2_CODE21"].astype(str).str.strip()

    full = full_507_df.copy()
    full["sa2_code_str"] = full["sa2_code"].astype(int).astype(str)
    assert len(full) == N_SUPER_ZONES

    gdf_507 = gdf[gdf["SA2_CODE21"].isin(full["sa2_code_str"])].copy()
    if gdf_507.crs is None:
        gdf_507.set_crs(7844, inplace=True)
    gdf_507_m = gdf_507.to_crs(TARGET_CRS)
    gdf_507_m = gdf_507_m.merge(
        full, left_on="SA2_CODE21", right_on="sa2_code_str", how="inner"
    )
    assert len(gdf_507_m) == N_SUPER_ZONES, \
        f"expected {N_SUPER_ZONES} matched polygons, got {len(gdf_507_m)}"
    return gdf_507_m


def polygon_to_single_ring(geom):
    # zone.dat only supports one boundary ring per zone (no holes). 11 of the
    # 507 SA2s are "<Town> Surrounds"-style donuts with a smaller town SA2
    # sitting in a hole inside them (e.g. "Roma" inside "Roma Surrounds") --
    # using exterior.coords alone would silently fill that hole in, making
    # the Surrounds zone's outline 100% overlap the town zone it encloses.
    # Fix: cut a zero-width slit from the exterior ring to each hole so the
    # whole thing traces as a single connected ring with the hole excluded,
    # same trick used for "island" polygons in formats with no native
    # multi-ring support. Nearest exterior/hole point pair found via KDTree
    # since these rings can have thousands of points (brute force is too slow).
    ext_coords = list(geom.exterior.coords)[:-1]
    if not geom.interiors:
        return ext_coords + [ext_coords[0]]

    ring = list(ext_coords)
    for interior in geom.interiors:
        hole_coords = list(interior.coords)[:-1]
        ext_arr = np.array(ring)
        tree = cKDTree(ext_arr)
        hole_arr = np.array(hole_coords)
        dists, idxs = tree.query(hole_arr)
        hj = int(np.argmin(dists))
        ei = int(idxs[hj])
        hole_reordered = hole_coords[hj:] + hole_coords[:hj] + [hole_coords[hj]]
        ring = ring[:ei + 1] + [ring[ei]] + hole_reordered + [ring[ei]] + ring[ei + 1:]
    return ring + [ring[0]]


def write_zone_dat(path, gdf_507_m):
    # NOTE: convex_hull was tried first and is wrong here -- it bulges
    # outward over any concave/irregular SA2 shape, so adjacent zones'
    # hulls overlap each other instead of tiling cleanly. Use each zone's
    # actual exterior ring instead, which exactly matches the source ABS
    # SA2 boundaries (these already form a clean, non-overlapping partition,
    # and adjacent zones share identical vertex coordinates at their common
    # border, so using them directly keeps shared edges exact -- no overlap,
    # no gaps). zone.dat only supports one boundary ring per zone (no holes,
    # no multi-part), so MultiPolygon zones (52 of 507 -- islands, bay
    # areas, etc.) use their largest part by area; smaller disjoint parts
    # are dropped from the *outline* only (this is a display file per the
    # manual -- it doesn't affect node/zone assignment or routing).
    point_to_idx = {}
    idx_counter = 1
    zone_to_point_idx = {}
    multipart_zones = []
    holed_zones = []

    gdf_sorted = gdf_507_m.sort_values("super_zone_id")
    for _, row in gdf_sorted.iterrows():
        zone_id = int(row["super_zone_id"])
        geom = row.geometry
        if geom.geom_type == "MultiPolygon":
            multipart_zones.append((zone_id, len(geom.geoms)))
            geom = max(geom.geoms, key=lambda g: g.area)
        if geom.interiors:
            holed_zones.append((zone_id, len(geom.interiors)))
        pts = polygon_to_single_ring(geom)
        ids = []
        for pt in pts:
            if pt not in point_to_idx:
                point_to_idx[pt] = idx_counter
                idx_counter += 1
            ids.append(point_to_idx[pt])
        zone_to_point_idx[zone_id] = ids

    if multipart_zones:
        print(f"  {len(multipart_zones)} zones are MultiPolygon -- used largest "
              f"part only (e.g. {multipart_zones[:5]}{'...' if len(multipart_zones) > 5 else ''})")
    if holed_zones:
        print(f"  {len(holed_zones)} zones have an interior hole (smaller zone "
              f"nested inside, e.g. a town SA2 inside its '... Surrounds' SA2) -- "
              f"cut a slit to the hole instead of filling it in: {holed_zones}")

    unique_points_list = list(point_to_idx.keys())

    with open(path, "w") as f:
        f.write(" This file defines zone regions\n")
        f.write(" number of feature points, number of zones\n")
        f.write(f"     {len(unique_points_list)}      {len(zone_to_point_idx)}\n")
        f.write("node #, x-coordinate, y-coordinate\n")
        for i, (x, y) in enumerate(unique_points_list, start=1):
            f.write(f"      {i}     {x:.3f},    {y:.3f}\n")
        f.write("zone #, number of nodes, node #'s\n")
        for zone_id in sorted(zone_to_point_idx.keys()):
            ids = zone_to_point_idx[zone_id]
            id_str = ", ".join(str(pid) for pid in ids)
            f.write(f"   {zone_id}     {len(ids)}: {id_str},\n")


# -----------------------------------------------------------------------
# demand.dat: 507x507 daily car demand from demand_integrated_sa2.omx,
# reindexed into super_zone_id order, peak-hour-factored across 2 intervals
# -----------------------------------------------------------------------
def load_demand_combined_car_reindexed(full_507_df):
    with h5py.File(DEMAND_OMX, "r") as f:
        mat = f["data"]["demand_combined_car"][:]
        sa2_codes_omx = [s.decode() if isinstance(s, bytes) else s
                          for s in f["lookup"]["SA2_CODE"][:]]

    omx_pos = {code: i for i, code in enumerate(sa2_codes_omx)}

    full = full_507_df.sort_values("super_zone_id")
    assert list(full["super_zone_id"]) == list(range(1, N_SUPER_ZONES + 1))

    new_order = []
    for code in full["sa2_code"]:
        code_str = str(int(code))
        assert code_str in omx_pos, f"SA2 code {code_str} not found in OMX lookup"
        new_order.append(omx_pos[code_str])
    new_order = np.array(new_order)

    reindexed = mat[np.ix_(new_order, new_order)]
    return reindexed, mat, omx_pos, full


def write_demand_dat(path, matrix, n_intervals, mult_factor, start_times):
    # DynusT requires strict fixed-width Fortran format (confirmed by "Error when
    # reading demand.dat" when using free-form whitespace-separated output):
    #   header:      I5, F6.3  (e.g. "    2 0.100")
    #   start times: n+1 × F8.1 (e.g. "     0.0    60.0   120.0")
    #   OD values:   6 × F10.4 per line (e.g. "    0.0000    9.6143")
    flat = matrix.flatten()
    with open(path, "w") as f:
        f.write(f"{n_intervals:5d} {mult_factor:.3f}\n")
        f.write("".join(f"{t:8.1f}" for t in start_times) + "\n")
        for k in range(n_intervals):
            f.write(f"Start Time =    {start_times[k]:.1f}\n")
            for i in range(0, len(flat), 6):
                chunk = flat[i:i + 6]
                f.write("".join(f"{v:10.4f}" for v in chunk) + "\n")


def write_empty_demand(path):
    with open(path, "w") as f:
        f.write("0  1.0\n")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1: Loading crosswalk")
    crosswalk_df, old_to_super = load_crosswalk()
    print(f"  {len(crosswalk_df)} old-zone rows, "
          f"{crosswalk_df['sa2_code'].nunique()} distinct SA2 codes")

    full_507 = load_full_507_sa2_table(crosswalk_df)
    print(f"  full 507-zone SA2 table loaded and cross-checked against crosswalk")

    print("\nSTEP 2: Rebuilding zone.dat (507 SA2 polygons)")
    gdf_507_m = load_507_polygons_with_crosswalk_numbering(full_507)
    zone_dat_path = os.path.join(OUT_DIR, "zone.dat")
    write_zone_dat(zone_dat_path, gdf_507_m)
    print(f"  wrote {zone_dat_path}")

    print("\nSTEP 3: Rebuilding network.dat")
    net = parse_network_for_remap(BACKUP_NETWORK)
    print(f"  parsed: n_nodes={net['n_nodes']} n_links={net['n_links']} "
          f"k_paths={net['k_paths']}")
    unmapped_old_zones = set(net["node_old_zone"].values()) - set(old_to_super.keys())
    assert not unmapped_old_zones, f"old zones with no crosswalk entry: {unmapped_old_zones}"
    network_dat_path = os.path.join(OUT_DIR, "network.dat")
    write_network_dat(network_dat_path, net, old_to_super)
    print(f"  wrote {network_dat_path}")

    print("\nSTEP 4: Rebuilding origin.dat (generation links)")
    old_origin = parse_origin(ORIGIN_DAT)
    assert set(old_origin.keys()) == set(old_to_super.keys()), "origin.dat zone coverage mismatch"
    new_origin = build_new_origin(old_origin, old_to_super)
    n_old_links = sum(len(v) for v in old_origin.values())
    n_new_links = sum(len(v) for v in new_origin.values())
    dormant_zone_ids = sorted(z for z, v in new_origin.items() if len(v) == 0)
    print(f"  old generation links: {n_old_links}, new (deduped): {n_new_links}")
    print(f"  new zones with 0 generation links (dormant, no nodes in this "
          f"SEQ network): {len(dormant_zone_ids)}")

    print("\nSTEP 5: Rebuilding destination.dat (destination nodes)")
    old_dest = parse_destination(DESTINATION_DAT)
    assert set(old_dest.keys()) == set(old_to_super.keys()), "destination.dat zone coverage mismatch"
    new_dest, dropped, node_owner_count = build_new_destination(old_dest, old_to_super)
    n_old_dest = sum(len(v) for v in old_dest.values())
    n_new_dest = sum(len(v) for v in new_dest.values())
    print(f"  old destination nodes: {n_old_dest}, new (deduped): {n_new_dest}")
    if dropped:
        print(f"  capped {len(dropped)} (node, zone) pairs to enforce the manual's "
              f"max-2-owning-zones rule (pre-existing in the original data, not "
              f"introduced by this remap): {dropped}")

    print("\nSTEP 5b: Gateway-node loading for dormant zones")
    print(f"  ErrorLog.dat from the live run confirmed DynusT hard-requires "
          f">=1 destination per zone -- snapping {len(dormant_zone_ids)} dormant "
          f"zones to their nearest real network node instead of dropping them "
          f"(per user decision: preserve their real demand, ~714K trips/day)")
    xy_df = parse_xy(XY_DAT)
    node_links = build_node_incident_links(net["link_lines"])
    assigned = assign_gateway_nodes(dormant_zone_ids, gdf_507_m, xy_df, node_links,
                                     new_origin, new_dest, node_owner_count)
    print(f"  assigned gateway nodes for {len(assigned)} zones "
          f"(e.g. {assigned[:5]}{'...' if len(assigned) > 5 else ''})")

    n_still_empty_origin = sum(1 for v in new_origin.values() if len(v) == 0)
    n_still_empty_dest = sum(1 for v in new_dest.values() if len(v) == 0)
    assert n_still_empty_origin == 0, f"{n_still_empty_origin} zones still have 0 generation links"
    assert n_still_empty_dest == 0, f"{n_still_empty_dest} zones still have 0 destinations"
    print(f"  OK: all {N_SUPER_ZONES} zones now have >=1 generation link and >=1 destination")

    violations = validate_destination_sharing(new_dest)
    assert not violations, f"destination node >2-zone cap enforcement failed: {violations}"
    print("  OK: no destination node referenced by more than 2 zones")

    origin_dat_path = os.path.join(OUT_DIR, "origin.dat")
    write_origin_dat(origin_dat_path, new_origin)
    print(f"  wrote {origin_dat_path}")
    destination_dat_path = os.path.join(OUT_DIR, "destination.dat")
    write_destination_dat(destination_dat_path, new_dest)
    print(f"  wrote {destination_dat_path}")

    print("\nSTEP 6: Rebuilding demand.dat from demand_combined_car")
    reindexed, orig_mat, omx_pos, full_507_sorted = load_demand_combined_car_reindexed(full_507)
    print(f"  reindexed matrix shape: {reindexed.shape}, "
          f"sum={reindexed.sum():.1f} (matches original sum={orig_mat.sum():.1f}: "
          f"{np.isclose(reindexed.sum(), orig_mat.sum())})")

    # Spot check: zone 301011001 (Alexandra Hills) row sum pre/post reindex
    spot_sa2 = "301011001"
    if spot_sa2 in omx_pos:
        orig_row_sum = orig_mat[omx_pos[spot_sa2]].sum()
        new_idx = list(full_507_sorted["sa2_code"].astype(int).astype(str)).index(spot_sa2)
        new_row_sum = reindexed[new_idx].sum()
        print(f"  spot check SA2 {spot_sa2}: orig row sum={orig_row_sum:.1f}, "
              f"reindexed row sum={new_row_sum:.1f}, match={np.isclose(orig_row_sum, new_row_sum)}")

    demand_dat_path = os.path.join(OUT_DIR, "demand.dat")
    write_demand_dat(demand_dat_path, reindexed, n_intervals=2,
                      mult_factor=PEAK_MULT_FACTOR, start_times=START_TIMES)
    print(f"  wrote {demand_dat_path} "
          f"(2 intervals x {reindexed.size} entries, mult_factor={PEAK_MULT_FACTOR} "
          f"-> {2*PEAK_MULT_FACTOR*100:.0f}% of daily total loaded across 0-120min)")

    print("\nSTEP 7: Writing empty demand_truck.dat / demand_HOV.dat")
    truck_path = os.path.join(OUT_DIR, "demand_truck.dat")
    hov_path = os.path.join(OUT_DIR, "demand_HOV.dat")
    write_empty_demand(truck_path)
    write_empty_demand(hov_path)
    print(f"  wrote {truck_path}")
    print(f"  wrote {hov_path}")

    print("\n" + "=" * 70)
    print("DONE. Live model directory was NOT modified.")
    print(f"Review outputs in {OUT_DIR}, then swap into the model directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
