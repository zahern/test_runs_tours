"""
build_superzone_507.py

Converts the live DynusT model's 4,654-zone system to the 507 SA2-zone
system used by the population/demand pipeline, using DynusT's built-in
SuperZone.dat zone-aggregation mechanism (manual section 4.22).

Does NOT touch zone.dat, origin.dat, destination.dat, or any demand*.dat
file -- those stay valid at the original 4,654-zone resolution per the
manual (demand.dat footnote 5, p.69). Only produces new files; never
overwrites the live model directory.

Outputs (written to OUT_DIR):
  - SuperZone.dat            DynusT zone-aggregation input file
  - network_aggflag1.dat     network.dat with header field 5 (aggregation
                              flag) flipped 0 -> 1, otherwise byte-identical
  - zone_crosswalk_507.csv   old_zone_id, sa2_code, super_zone_id,
                              centroid_x, centroid_y, match_type
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

MODEL_DIR = r"C:\Users\ahernz\SEQ_DTA_2025"
NETWORK_DAT = os.path.join(MODEL_DIR, "network.dat")
XY_DAT = os.path.join(MODEL_DIR, "xy.dat")
ORIGIN_DAT = os.path.join(MODEL_DIR, "origin.dat")
DESTINATION_DAT = os.path.join(MODEL_DIR, "destination.dat")

ZONE_IDS_507_TXT = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
SA2_SHP = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"

OUT_DIR = r"Z:\test_runs_tours\prepare_network\new\507_conversion"
TARGET_CRS = 7856  # GDA2020 / MGA Zone 56 -- matches xy.dat / zone.dat

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------
# 1. Parse network.dat: header + node -> old_zone_id
# -----------------------------------------------------------------------
def parse_network(path):
    with open(path) as f:
        lines = f.readlines()

    header = lines[0].split()
    n_zones, n_nodes, n_links, k_paths, agg_flag = (int(x) for x in header)

    node_zone = {}
    for line in lines[1:1 + n_nodes]:
        node_id, zone_id = (int(x) for x in line.split())
        node_zone[node_id] = zone_id

    assert len(node_zone) == n_nodes, \
        f"Expected {n_nodes} node records, parsed {len(node_zone)}"

    return {
        "lines": lines,
        "n_zones": n_zones,
        "n_nodes": n_nodes,
        "n_links": n_links,
        "k_paths": k_paths,
        "agg_flag": agg_flag,
        "node_zone": node_zone,
    }


# -----------------------------------------------------------------------
# 2. Parse xy.dat: node -> (x, y)
# -----------------------------------------------------------------------
def parse_xy(path):
    df = pd.read_csv(path, header=None, names=["node_id", "x", "y"], sep=r"\s+")
    return dict(zip(df["node_id"], zip(df["x"], df["y"])))


# -----------------------------------------------------------------------
# 3. Parse origin.dat / destination.dat: zone -> [connector node ids]
#    Format (manual 4.20/4.21): "zone  count  [mode]  node node node..."
#    destination.dat has no mode field; origin.dat does. We only need the
#    node ids as a centroid fallback, so just grab every integer after the
#    count that looks like a node id (i.e. skip the optional loading-mode
#    flag in origin.dat by taking the *last* `count` tokens on the line
#    isn't reliable since origin.dat is multi-line per zone). We therefore
#    only use destination.dat (single-line per zone, simpler and
#    sufficient for the empty-zone fallback).
# -----------------------------------------------------------------------
def parse_destination(path):
    zone_nodes = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            zone_id, count = int(parts[0]), int(parts[1])
            node_ids = [int(x) for x in parts[2:2 + count]]
            zone_nodes[zone_id] = node_ids
    return zone_nodes


# -----------------------------------------------------------------------
# 4. Compute centroid (x, y) for every old zone 1..n_zones
# -----------------------------------------------------------------------
def compute_zone_centroids(n_zones, node_zone, node_xy, dest_zone_nodes):
    zone_node_ids = {z: [] for z in range(1, n_zones + 1)}
    for node_id, zone_id in node_zone.items():
        zone_node_ids[zone_id].append(node_id)

    centroids = {}
    fallback_zones = []
    for zone_id in range(1, n_zones + 1):
        node_ids = zone_node_ids[zone_id]
        coords = [node_xy[n] for n in node_ids if n in node_xy]
        if coords:
            xs, ys = zip(*coords)
            centroids[zone_id] = (float(np.mean(xs)), float(np.mean(ys)))
        else:
            # Fallback: use this zone's destination.dat connector node(s)
            fallback_nodes = dest_zone_nodes.get(zone_id, [])
            fallback_coords = [node_xy[n] for n in fallback_nodes if n in node_xy]
            if not fallback_coords:
                raise ValueError(
                    f"Zone {zone_id} has no member nodes AND no usable "
                    f"destination.dat connector node -- cannot place it."
                )
            xs, ys = zip(*fallback_coords)
            centroids[zone_id] = (float(np.mean(xs)), float(np.mean(ys)))
            fallback_zones.append(zone_id)

    return centroids, fallback_zones


# -----------------------------------------------------------------------
# 5. Load the 507 SA2 polygons, reprojected to match node coordinates
# -----------------------------------------------------------------------
def load_507_sa2_polygons():
    with open(ZONE_IDS_507_TXT) as f:
        content = f.read()
    zone_ids_507 = sorted(int(x) for x in content.split() if x.strip())
    assert len(zone_ids_507) == 507, \
        f"Expected 507 SA2 zone ids, found {len(zone_ids_507)}"

    gdf = gpd.read_file(SA2_SHP)
    gdf["SA2_CODE21"] = gdf["SA2_CODE21"].astype(str).str.strip()

    zone_ids_str = [str(z) for z in zone_ids_507]
    gdf_507 = gdf[gdf["SA2_CODE21"].isin(zone_ids_str)].copy()

    found = set(gdf_507["SA2_CODE21"].astype(int))
    missing = sorted(set(zone_ids_507) - found)
    if missing:
        raise ValueError(
            f"{len(missing)} of the 507 SA2 codes were not found in the "
            f"shapefile: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    if gdf_507.crs is None:
        gdf_507.set_crs(7844, inplace=True)  # GDA2020 geographic, ABS default
    gdf_507_m = gdf_507.to_crs(TARGET_CRS)

    # Sorted super-zone numbering: 1..507 over ascending SA2 code
    gdf_507_m = gdf_507_m.sort_values("SA2_CODE21").reset_index(drop=True)
    gdf_507_m["super_zone_id"] = gdf_507_m.index + 1

    return gdf_507_m[["SA2_CODE21", "super_zone_id", "geometry"]]


# -----------------------------------------------------------------------
# 6. Spatial join: old zone centroids -> 507 SA2 super zones
#    Point-in-polygon first; nearest-polygon fallback for centroids that
#    fall outside all 507 zones (gateway/external zones).
# -----------------------------------------------------------------------
def assign_super_zones(centroids, gdf_507):
    # The 507 SA2 zones span all of Queensland (confirmed: 284 "Rest of Qld"
    # vs 223 "Greater Brisbane" by GCC_NAME21), while MGA Zone 56 (TARGET_CRS)
    # is only locally accurate near its central meridian (~153E, i.e. SEQ).
    # Point-in-polygon containment is CRS-invariant so it's fine in MGA56,
    # but nearest-neighbor DISTANCE ranking across the whole state would be
    # skewed by that projection's distortion far from SEQ. So: do the
    # containment test in TARGET_CRS (consistent with the rest of the model),
    # but do the nearest-polygon fallback in geographic coordinates (EPSG:4326)
    # so distance ranking stays roughly proportional to real-world distance
    # statewide.
    zone_ids = sorted(centroids.keys())
    xs = [centroids[z][0] for z in zone_ids]
    ys = [centroids[z][1] for z in zone_ids]
    points = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=TARGET_CRS)
    gdf_pts = gpd.GeoDataFrame({"old_zone_id": zone_ids, "x": xs, "y": ys},
                                geometry=points, crs=TARGET_CRS)

    contained = gpd.sjoin(gdf_pts, gdf_507, how="left", predicate="within")
    # sjoin can produce duplicate rows if a point is exactly on a shared
    # boundary between polygons; keep the first match per zone.
    contained = contained.drop_duplicates(subset="old_zone_id", keep="first")
    contained = contained.set_index("old_zone_id")

    matched_mask = contained["super_zone_id"].notna()
    direct_ids = contained.index[matched_mask].tolist()
    unmatched_ids = contained.index[~matched_mask].tolist()

    result = {}
    for zone_id in direct_ids:
        result[zone_id] = (
            int(contained.loc[zone_id, "super_zone_id"]),
            contained.loc[zone_id, "SA2_CODE21"],
            "direct",
        )

    if unmatched_ids:
        gdf_pts_4326 = gdf_pts.to_crs(4326)
        gdf_507_4326 = gdf_507.to_crs(4326)
        gdf_unmatched = gdf_pts_4326[gdf_pts_4326["old_zone_id"].isin(unmatched_ids)]
        nearest = gpd.sjoin_nearest(
            gdf_unmatched, gdf_507_4326, how="left", distance_col="dist_deg"
        )
        nearest = nearest.drop_duplicates(subset="old_zone_id", keep="first")
        nearest = nearest.set_index("old_zone_id")
        for zone_id in unmatched_ids:
            result[zone_id] = (
                int(nearest.loc[zone_id, "super_zone_id"]),
                nearest.loc[zone_id, "SA2_CODE21"],
                "nearest_fallback",
            )

        dist_km = nearest["dist_deg"] * 111.0  # rough deg->km at this latitude
        print(f"    fallback distance to nearest SA2 (km, approx): "
              f"min={dist_km.min():.1f} median={dist_km.median():.1f} "
              f"max={dist_km.max():.1f}")
        far = (dist_km > 20).sum()
        if far:
            print(f"    WARNING: {far} fallback zones are >20km from their "
                  f"nearest 507-zone SA2 -- review these in the crosswalk CSV")

    return result, direct_ids, unmatched_ids


# -----------------------------------------------------------------------
# 7. Write SuperZone.dat (manual section 4.22 / Figure 4-25 format)
# -----------------------------------------------------------------------
def write_superzone_dat(path, n_super_zones, old_zone_ids, super_zone_of, per_line=15):
    def chunked(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    with open(path, "w") as f:
        f.write(f"{n_super_zones}\n")
        f.write("Original Zones\n")
        for chunk in chunked(old_zone_ids, per_line):
            f.write(" ".join(str(z) for z in chunk) + "\n")
        f.write("Mapping of Original Zones to Aggregated zones\n")
        for chunk in chunked(old_zone_ids, per_line):
            f.write(" ".join(str(super_zone_of[z]) for z in chunk) + "\n")


# -----------------------------------------------------------------------
# 8. Write network_aggflag1.dat: header field 5 flipped 0 -> 1
# -----------------------------------------------------------------------
def write_network_aggflag1(path, lines, n_zones, n_nodes, n_links, k_paths):
    new_lines = list(lines)
    new_lines[0] = f"{n_zones}  {n_nodes}  {n_links}      {k_paths}      1\n"
    with open(path, "w") as f:
        f.writelines(new_lines)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1: Parsing network.dat")
    net = parse_network(NETWORK_DAT)
    print(f"  n_zones={net['n_zones']} n_nodes={net['n_nodes']} "
          f"n_links={net['n_links']} k_paths={net['k_paths']} "
          f"agg_flag={net['agg_flag']}")

    print("\nSTEP 2: Parsing xy.dat")
    node_xy = parse_xy(XY_DAT)
    print(f"  {len(node_xy)} node coordinates loaded")

    print("\nSTEP 3: Parsing destination.dat (fallback source)")
    dest_zone_nodes = parse_destination(DESTINATION_DAT)
    print(f"  {len(dest_zone_nodes)} zones with destination connectors")

    print("\nSTEP 4: Computing old-zone centroids")
    centroids, fallback_zones = compute_zone_centroids(
        net["n_zones"], net["node_zone"], node_xy, dest_zone_nodes
    )
    print(f"  {len(centroids)} centroids computed "
          f"({len(fallback_zones)} via destination.dat fallback: {fallback_zones})")

    print("\nSTEP 5: Loading 507 SA2 polygons")
    gdf_507 = load_507_sa2_polygons()
    print(f"  {len(gdf_507)} SA2 polygons loaded, super_zone_id 1..{len(gdf_507)}")

    print("\nSTEP 6: Spatial join (old zone centroid -> super zone)")
    assignment, direct_ids, fallback_ids = assign_super_zones(centroids, gdf_507)
    print(f"  direct point-in-polygon matches: {len(direct_ids)}")
    print(f"  nearest-fallback matches (external/gateway zones): {len(fallback_ids)}")

    # --- Validation ---
    print("\nSTEP 7: Validating coverage")
    old_zone_ids = list(range(1, net["n_zones"] + 1))
    assert set(assignment.keys()) == set(old_zone_ids), "Zone coverage mismatch!"
    super_zone_of = {z: assignment[z][0] for z in old_zone_ids}
    sa2_of = {z: assignment[z][1] for z in old_zone_ids}
    match_type_of = {z: assignment[z][2] for z in old_zone_ids}

    used_super_zones = sorted(set(super_zone_of.values()))
    unused_super_zones = sorted(set(range(1, len(gdf_507) + 1)) - set(used_super_zones))
    print(f"  {len(used_super_zones)} of {len(gdf_507)} super zones received "
          f"at least one old zone")
    if unused_super_zones:
        unused_sa2 = gdf_507.set_index("super_zone_id").loc[unused_super_zones, "SA2_CODE21"]
        print(f"  {len(unused_super_zones)} super zones got ZERO old zones "
              f"(SA2 areas not covered by the live network's zone footprint):")
        print(f"    super_zone_ids: {unused_super_zones}")
        print(f"    sa2_codes: {unused_sa2.tolist()}")

    # Diagnostic: compare spatial extent of the live network vs the 507 SA2 set
    xs_all = [c[0] for c in centroids.values()]
    ys_all = [c[1] for c in centroids.values()]
    net_bounds = (min(xs_all), min(ys_all), max(xs_all), max(ys_all))
    sa2_bounds = gdf_507.total_bounds
    print(f"  live-network centroid extent (E,N): "
          f"{net_bounds[0]:.0f}-{net_bounds[2]:.0f}, {net_bounds[1]:.0f}-{net_bounds[3]:.0f}")
    print(f"  507-SA2-zone extent           (E,N): "
          f"{sa2_bounds[0]:.0f}-{sa2_bounds[2]:.0f}, {sa2_bounds[1]:.0f}-{sa2_bounds[3]:.0f}")

    counts_per_super = pd.Series(list(super_zone_of.values())).value_counts()
    print(f"  old-zones-per-super-zone: min={counts_per_super.min()} "
          f"max={counts_per_super.max()} mean={counts_per_super.mean():.1f}")

    # --- Write outputs ---
    print("\nSTEP 8: Writing outputs to", OUT_DIR)
    superzone_path = os.path.join(OUT_DIR, "SuperZone.dat")
    write_superzone_dat(superzone_path, len(gdf_507), old_zone_ids, super_zone_of)
    print(f"  wrote {superzone_path}")

    network_path = os.path.join(OUT_DIR, "network_aggflag1.dat")
    write_network_aggflag1(
        network_path, net["lines"], net["n_zones"], net["n_nodes"],
        net["n_links"], net["k_paths"]
    )
    print(f"  wrote {network_path}")

    crosswalk_path = os.path.join(OUT_DIR, "zone_crosswalk_507.csv")
    crosswalk_df = pd.DataFrame([
        {
            "old_zone_id": z,
            "sa2_code": sa2_of[z],
            "super_zone_id": super_zone_of[z],
            "centroid_x": centroids[z][0],
            "centroid_y": centroids[z][1],
            "match_type": match_type_of[z],
        }
        for z in old_zone_ids
    ])
    crosswalk_df.to_csv(crosswalk_path, index=False)
    print(f"  wrote {crosswalk_path} ({len(crosswalk_df)} rows)")

    print("\n" + "=" * 70)
    print("DONE. Live model directory was NOT modified.")
    print("Review zone_crosswalk_507.csv, then manually copy SuperZone.dat")
    print("and network_aggflag1.dat (-> network.dat) into the model directory")
    print("when ready to activate zone aggregation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
