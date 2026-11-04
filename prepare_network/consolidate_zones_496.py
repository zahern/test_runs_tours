"""
consolidate_zones_496.py

Eliminates the 11 "zone-within-zone" pairs in the 507 SA2 system (small
town SA2s nested inside their surrounding rural SA2's hole, e.g. Roma
inside Roma Surrounds). Each inner zone is absorbed into its outer zone,
producing a clean 496-zone system with no overlapping or nested zones.

What changes:
  DynusT (C:\\Users\\ahernz\\SEQ_DTA_2025):
    zone.dat     -- 496 polygons, outer zones now use union(outer, inner)
                    geometry so the hole is filled and the outline is clean
    network.dat  -- header declares 496 zones; node-zone assignments updated
    origin.dat   -- generation links merged for consolidated zones
    destination.dat -- same
    demand.dat   -- 496x496 OD matrix, F10.4 fixed-width Fortran format

  demand_ pipeline (Z:\\test_runs_tours\\demand_):
    All *.omx files: inner zone row/col summed into outer zone, then
    removed. demand_* matrices: sum. skim_* matrices: demand-weighted
    average (fallback to outer value when demand is zero). All other
    matrices (crash_*, nearmiss_*): sum.

Nothing is overwritten in place: DynusT outputs go to
  Z:\\test_runs_tours\\prepare_network\\new\\496_native\\
OMX outputs go to
  Z:\\test_runs_tours\\demand_\\consolidated_496\\
Swap into the live directories after review.
"""
import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
MODEL_DIR        = r"C:\Users\ahernz\SEQ_DTA_2025"
BACKUP_DIR       = os.path.join(MODEL_DIR, "pre_native_507_backup")
BACKUP_NETWORK   = os.path.join(MODEL_DIR, "pre_507_conversion_backup", "network.dat")
ORIGIN_DAT       = os.path.join(BACKUP_DIR, "origin.dat")
DESTINATION_DAT  = os.path.join(BACKUP_DIR, "destination.dat")
CROSSWALK_CSV    = os.path.join(MODEL_DIR, "zone_crosswalk_507.csv")
XY_DAT           = os.path.join(MODEL_DIR, "xy.dat")
SA2_SHP          = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
ZONE_IDS_507_TXT = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
DEMAND_OMX       = r"Z:\test_runs_tours\demand_\demand_integrated_sa2.omx"
DEMAND_DIR       = r"Z:\test_runs_tours\demand_"

OUT_DYNUS        = r"Z:\test_runs_tours\prepare_network\new\496_native"
OUT_OMX          = r"Z:\test_runs_tours\demand_\consolidated_496"
TARGET_CRS = 7856
PEAK_MULT_FACTOR = 0.10
START_TIMES = [0.0, 60.0, 120.0]

os.makedirs(OUT_DYNUS, exist_ok=True)
os.makedirs(OUT_OMX, exist_ok=True)

# 11 (inner_super_zone_id, outer_super_zone_id) pairs — inner absorbed into outer
DONUT_PAIRS = [
    (174, 175),   # Roma -> Roma Surrounds
    (185, 186),   # Stanthorpe -> Stanthorpe Surrounds
    (190, 189),   # Emerald -> Central Highlands - West
    (211, 210),   # Biloela -> Banana
    (387, 388),   # Mount Isa -> Mount Isa Surrounds
    (442, 443),   # Ayr -> Burdekin
    (444, 445),   # Charters Towers -> Dalrymple
    (446, 447),   # Ingham -> Ingham Surrounds
    (486, 488),   # Kingaroy -> Kingaroy Surrounds - South
    (493, 495),   # Gympie - North -> Gympie Surrounds
    (494, 495),   # Gympie - South -> Gympie Surrounds
]
INNER_TO_OUTER = {inner: outer for inner, outer in DONUT_PAIRS}

# -----------------------------------------------------------------------
# 1. Build zone mappings
# -----------------------------------------------------------------------
def build_zone_mappings():
    # Load full 507-zone table (sorted SA2 code -> super_zone_id 1..507)
    with open(ZONE_IDS_507_TXT) as f:
        zone_ids_507 = sorted(int(x) for x in f.read().split() if x.strip())
    assert len(zone_ids_507) == 507
    full507 = pd.DataFrame({"sa2_code": zone_ids_507})
    full507["super_zone_id"] = range(1, 508)

    # Consolidation: for each of the 507 zones, what is its "effective" zone
    # after absorbing inner zones into outer zones?
    effective = {z: INNER_TO_OUTER.get(z, z) for z in range(1, 508)}
    # Renumber remaining 496 effective zones 1..496 in sorted order
    distinct_effective = sorted(set(effective.values()))
    assert len(distinct_effective) == 496, len(distinct_effective)
    new_id = {old: new for new, old in enumerate(distinct_effective, start=1)}

    # final 507->496 map
    zone507_to_496 = {z: new_id[effective[z]] for z in range(1, 508)}
    inner_zones_507 = sorted(INNER_TO_OUTER.keys())

    # Build the 496-zone SA2 table (outer zones, no inner)
    full496 = full507[~full507["super_zone_id"].isin(inner_zones_507)].copy()
    full496["zone_496"] = full496["super_zone_id"].map(zone507_to_496)
    full496 = full496.sort_values("zone_496").reset_index(drop=True)

    # Load 507-crosswalk and extend to 496
    cw = pd.read_csv(CROSSWALK_CSV)
    old_to_496 = {
        int(r.old_zone_id): zone507_to_496[int(r.super_zone_id)]
        for _, r in cw.iterrows()
    }

    return full507, full496, zone507_to_496, old_to_496, inner_zones_507, new_id

# -----------------------------------------------------------------------
# 2. Load SA2 polygons — merged for consolidated pairs
# -----------------------------------------------------------------------
def load_496_polygons(full496, inner_to_outer, zone507_to_496):
    gdf = gpd.read_file(SA2_SHP)
    gdf["SA2_CODE21"] = gdf["SA2_CODE21"].astype(str).str.strip()
    all_507_codes = full496["sa2_code"].astype(int).astype(str).tolist()
    # Also need inner zone codes to union into outer
    full507_all_codes = set(all_507_codes)
    inner_codes = {}  # outer_super_zone -> [inner geometries]

    # We need all 507 geometries (inner + outer) to do the union
    from io import StringIO
    with open(ZONE_IDS_507_TXT) as f:
        zone_ids_507 = sorted(int(x) for x in f.read().split() if x.strip())
    all_codes_507 = [str(z) for z in zone_ids_507]
    gdf_all = gdf[gdf["SA2_CODE21"].isin(all_codes_507)].copy()
    if gdf_all.crs is None:
        gdf_all.set_crs(7844, inplace=True)
    gdf_all = gdf_all.to_crs(TARGET_CRS)

    # Build super_zone_id lookup
    code_to_super = dict(zip(
        [str(int(z)) for z in zone_ids_507],
        range(1, 508)
    ))
    gdf_all["super_zone_id"] = gdf_all["SA2_CODE21"].map(code_to_super)

    # For each outer (surviving) zone, union its geometry with all inner zones
    # that were absorbed into it
    outer_to_inner_geoms = {}
    for inner_sid, outer_sid in inner_to_outer.items():
        rows = gdf_all[gdf_all["super_zone_id"] == inner_sid]
        for _, r in rows.iterrows():
            outer_to_inner_geoms.setdefault(outer_sid, []).append(r.geometry)

    geoms_for_496 = {}
    for _, row in gdf_all.iterrows():
        sid = int(row["super_zone_id"])
        if sid in inner_to_outer:
            continue  # skip inner zones
        geom = row.geometry
        extras = outer_to_inner_geoms.get(sid, [])
        if extras:
            from shapely.ops import unary_union
            geom = unary_union([geom] + extras)
        zone_496 = zone507_to_496[sid]
        geoms_for_496[zone_496] = (row["SA2_CODE21"], geom)

    assert len(geoms_for_496) == 496, f"expected 496, got {len(geoms_for_496)}"
    return geoms_for_496  # zone_496 -> (sa2_code_str, geom)

# -----------------------------------------------------------------------
# 3. zone.dat
# -----------------------------------------------------------------------
def polygon_to_single_ring(geom):
    from shapely.geometry import Polygon
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    ext_coords = list(geom.exterior.coords)[:-1]
    if not list(geom.interiors):
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


def write_zone_dat(path, geoms_496):
    point_to_idx = {}
    idx_counter = 1
    zone_to_point_idx = {}
    for zone_id in sorted(geoms_496.keys()):
        _, geom = geoms_496[zone_id]
        pts = polygon_to_single_ring(geom)
        ids = []
        for pt in pts:
            if pt not in point_to_idx:
                point_to_idx[pt] = idx_counter
                idx_counter += 1
            ids.append(point_to_idx[pt])
        zone_to_point_idx[zone_id] = ids
    unique_points = list(point_to_idx.keys())
    with open(path, "w") as f:
        f.write(" This file defines zone regions\n")
        f.write(" number of feature points, number of zones\n")
        f.write(f"     {len(unique_points)}      {len(zone_to_point_idx)}\n")
        f.write("node #, x-coordinate, y-coordinate\n")
        for i, (x, y) in enumerate(unique_points, start=1):
            f.write(f"      {i}     {x:.3f},    {y:.3f}\n")
        f.write("zone #, number of nodes, node #'s\n")
        for zone_id in sorted(zone_to_point_idx.keys()):
            ids = zone_to_point_idx[zone_id]
            f.write(f"   {zone_id}     {len(ids)}: " +
                    ", ".join(str(p) for p in ids) + ",\n")

# -----------------------------------------------------------------------
# 4. network.dat
# -----------------------------------------------------------------------
def parse_network(path):
    with open(path) as f:
        lines = f.readlines()
    h = lines[0].split()
    n_zones, n_nodes, n_links, k, agg = (int(x) for x in h)
    node_lines = lines[1:1 + n_nodes]
    link_lines = lines[1 + n_nodes:1 + n_nodes + n_links]
    node_ids = []
    node_old_zone = {}
    for l in node_lines:
        nid, zid = (int(x) for x in l.split())
        node_ids.append(nid)
        node_old_zone[nid] = zid
    return {"n_nodes": n_nodes, "n_links": n_links, "k": k,
            "node_ids": node_ids, "node_old_zone": node_old_zone, "link_lines": link_lines}


def write_network_dat(path, net, old_to_496, n_zones):
    with open(path, "w") as f:
        f.write(f"{n_zones}  {net['n_nodes']}  {net['n_links']}      {net['k']}      0\n")
        for nid in net["node_ids"]:
            f.write(f"{nid} {old_to_496[net['node_old_zone'][nid]]}\n")
        f.writelines(net["link_lines"])

# -----------------------------------------------------------------------
# 5. origin.dat / destination.dat
# -----------------------------------------------------------------------
def parse_origin(path):
    with open(path) as f:
        lines = f.readlines()
    result = {}
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if not p:
            i += 1; continue
        zid, cnt, mode = int(p[0]), int(p[1]), int(p[2])
        links = []
        for j in range(cnt):
            up, dn, w = lines[i + 1 + j].split()
            links.append((int(up), int(dn), float(w)))
        result[zid] = links
        i += 1 + cnt
    return result


def parse_destination(path):
    result = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p: continue
            zid, cnt = int(p[0]), int(p[1])
            result[zid] = [int(x) for x in p[2:2 + cnt]]
    return result


def build_connectivity_496(old_origin, old_dest, old_to_496, n_zones):
    new_origin = {z: [] for z in range(1, n_zones + 1)}
    new_dest   = {z: [] for z in range(1, n_zones + 1)}
    seen_orig  = {z: set() for z in range(1, n_zones + 1)}
    seen_dest  = {z: set() for z in range(1, n_zones + 1)}
    node_owner = {}

    for old_z, links in old_origin.items():
        new_z = old_to_496[old_z]
        for (up, dn, w) in links:
            key = (up, dn)
            if key not in seen_orig[new_z]:
                seen_orig[new_z].add(key)
                new_origin[new_z].append((up, dn, w))

    for old_z, nodes in old_dest.items():
        new_z = old_to_496[old_z]
        for n in nodes:
            if n not in seen_dest[new_z]:
                cnt = node_owner.get(n, 0)
                if cnt >= 2:
                    continue
                seen_dest[new_z].add(n)
                new_dest[new_z].append(n)
                node_owner[n] = cnt + 1

    return new_origin, new_dest, node_owner


def assign_gateway_nodes(empty_zones, geoms_496, xy_df, link_lines,
                          new_origin, new_dest, node_owner):
    node_ids = xy_df["node_id"].to_numpy()
    coords = xy_df[["x", "y"]].to_numpy()
    tree = cKDTree(coords)
    node_links = {}
    for line in link_lines:
        p = line.split()
        up, dn = int(p[0]), int(p[1])
        node_links.setdefault(up, []).append((up, dn))
        node_links.setdefault(dn, []).append((up, dn))

    for z in empty_zones:
        geom = geoms_496[z][1]
        cx, cy = geom.centroid.x, geom.centroid.y
        k = 10
        chosen = None
        while chosen is None:
            dists, idxs = tree.query([cx, cy], k=min(k, len(node_ids)))
            for idx in np.atleast_1d(idxs):
                cand = int(node_ids[idx])
                if node_owner.get(cand, 0) < 2:
                    chosen = cand
                    break
            if chosen is None:
                k *= 4
        new_dest[z] = [chosen]
        node_owner[chosen] = node_owner.get(chosen, 0) + 1
        inc = node_links.get(chosen)
        if inc:
            new_origin[z] = [inc[0] + (0.0,)]


def write_origin_dat(path, new_origin):
    with open(path, "w") as f:
        for z in sorted(new_origin.keys()):
            links = new_origin[z]
            f.write(f"{z} {len(links)} 0\n")
            for (up, dn, w) in links:
                f.write(f"{up} {dn} {w:.3f}\n")


def write_destination_dat(path, new_dest):
    with open(path, "w") as f:
        for z in sorted(new_dest.keys()):
            nodes = new_dest[z]
            f.write(f"{z} {len(nodes)} " + " ".join(str(n) for n in nodes) + "\n")

# -----------------------------------------------------------------------
# 6. demand.dat (496x496, fixed-width Fortran F10.4)
# -----------------------------------------------------------------------
def build_demand_496(zone507_to_496, n_zones):
    with h5py.File(DEMAND_OMX, "r") as f:
        mat = f["data"]["demand_combined_car"][:]
        sa2_codes_omx = [s.decode() for s in f["lookup"]["SA2_CODE"][:]]
    with open(ZONE_IDS_507_TXT) as f:
        zone_ids_507 = sorted(int(x) for x in f.read().split() if x.strip())

    omx_pos = {code: i for i, code in enumerate(sa2_codes_omx)}
    # Build 496-length index list (same row order as zone_id 1..496)
    # zone507_to_496 maps 507 super_zone_ids to 496 new ids; we need to aggregate
    # which omx rows/cols belong to each 496 zone
    from collections import defaultdict
    zone496_to_omx = defaultdict(list)
    for i, sa2_code in enumerate(zone_ids_507):
        sid = i + 1
        z496 = zone507_to_496[sid]
        omx_idx = omx_pos.get(str(sa2_code))
        if omx_idx is not None:
            zone496_to_omx[z496].append(omx_idx)

    # Build 496x496 demand matrix by summing rows/cols of merged zones
    demand_496 = np.zeros((n_zones, n_zones), dtype=np.float64)
    for zi in range(1, n_zones + 1):
        for zj in range(1, n_zones + 1):
            ii = zone496_to_omx.get(zi, [])
            jj = zone496_to_omx.get(zj, [])
            if ii and jj:
                demand_496[zi - 1, zj - 1] = mat[np.ix_(ii, jj)].sum()
    return demand_496


def write_demand_dat(path, matrix, n_intervals, mult_factor, start_times):
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
# 7. Consolidate all OMX files in demand_
# -----------------------------------------------------------------------
def consolidate_omx_file(src_path, dst_path, zone507_to_496, n_zones):
    with h5py.File(src_path, "r") as src:
        sa2_codes_omx = [s.decode() for s in src["lookup"]["SA2_CODE"][:]]
        omx_pos = {code: i for i, code in enumerate(sa2_codes_omx)}

        with open(ZONE_IDS_507_TXT) as f:
            zone_ids_507 = sorted(int(x) for x in f.read().split() if x.strip())

        from collections import defaultdict
        zone496_to_omx = defaultdict(list)
        for i, sa2_code in enumerate(zone_ids_507):
            sid = i + 1
            z496 = zone507_to_496[sid]
            omx_idx = omx_pos.get(str(sa2_code))
            if omx_idx is not None:
                zone496_to_omx[z496].append(omx_idx)

        # Build new SA2_CODE lookup (outer zone codes only, in zone_496 order)
        with open(ZONE_IDS_507_TXT) as f:
            zone_ids_507 = sorted(int(x) for x in f.read().split() if x.strip())
        full507 = pd.DataFrame({"sa2_code": zone_ids_507, "super_zone_id": range(1, 508)})
        inner_sids = set(INNER_TO_OUTER.keys())
        full496 = full507[~full507["super_zone_id"].isin(inner_sids)].copy()
        full496["zone_496"] = full496["super_zone_id"].map(zone507_to_496)
        full496 = full496.sort_values("zone_496")
        new_sa2_codes = np.array([str(int(c)).encode() for c in full496["sa2_code"]])

        # Get demand matrix for weighting skims
        demand_mat = None
        if "demand_combined_car" in src["data"]:
            demand_mat = src["data"]["demand_combined_car"][:]

        with h5py.File(dst_path, "w") as dst:
            lk = dst.create_group("lookup")
            lk.create_dataset("SA2_CODE", data=new_sa2_codes)
            data = dst.create_group("data")
            for mname in src["data"].keys():
                src_mat = src["data"][mname][:]
                new_mat = np.zeros((n_zones, n_zones), dtype=src_mat.dtype)
                is_skim = mname.startswith("skim_") or "avg" in mname
                for zi in range(1, n_zones + 1):
                    for zj in range(1, n_zones + 1):
                        ii = zone496_to_omx.get(zi, [])
                        jj = zone496_to_omx.get(zj, [])
                        if not ii or not jj:
                            continue
                        block = src_mat[np.ix_(ii, jj)]
                        if is_skim and demand_mat is not None:
                            d_block = demand_mat[np.ix_(ii, jj)]
                            total_d = d_block.sum()
                            if total_d > 0:
                                new_mat[zi - 1, zj - 1] = (block * d_block).sum() / total_d
                            else:
                                new_mat[zi - 1, zj - 1] = block[0, 0]
                        else:
                            new_mat[zi - 1, zj - 1] = block.sum()
                data.create_dataset(mname, data=new_mat, compression="gzip")
    return new_mat.shape


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1: Building zone mappings (507 -> 496, absorbing 11 donut pairs)")
    full507, full496, zone507_to_496, old_to_496, inner_zones_507, _ = build_zone_mappings()
    n_zones = 496
    print(f"  inner zones removed: {sorted(INNER_TO_OUTER.keys())}")
    print(f"  resulting zone count: {n_zones}")

    print("\nSTEP 2: Loading 496 zone polygons (outer zones unioned with inner)")
    geoms_496 = load_496_polygons(full496, INNER_TO_OUTER, zone507_to_496)
    print(f"  {len(geoms_496)} zones loaded")

    print("\nSTEP 3: Writing zone.dat")
    write_zone_dat(os.path.join(OUT_DYNUS, "zone.dat"), geoms_496)
    print(f"  wrote {OUT_DYNUS}\\zone.dat")

    print("\nSTEP 4: Writing network.dat")
    net = parse_network(BACKUP_NETWORK)
    write_network_dat(os.path.join(OUT_DYNUS, "network.dat"), net, old_to_496, n_zones)
    print(f"  wrote {OUT_DYNUS}\\network.dat")

    print("\nSTEP 5: Writing origin.dat + destination.dat")
    old_origin = parse_origin(ORIGIN_DAT)
    old_dest   = parse_destination(DESTINATION_DAT)
    xy_df = pd.read_csv(XY_DAT, header=None, names=["node_id", "x", "y"], sep=r"\s+")
    new_origin, new_dest, node_owner = build_connectivity_496(
        old_origin, old_dest, old_to_496, n_zones)
    empty_origin = [z for z, v in new_origin.items() if not v]
    empty_dest   = [z for z, v in new_dest.items() if not v]
    empty_zones  = sorted(set(empty_origin) | set(empty_dest))
    print(f"  zones needing gateway-node assignment: {len(empty_zones)}")
    assign_gateway_nodes(empty_zones, geoms_496, xy_df, net["link_lines"],
                          new_origin, new_dest, node_owner)
    empty_check_o = sum(1 for v in new_origin.values() if not v)
    empty_check_d = sum(1 for v in new_dest.values() if not v)
    assert empty_check_o == 0 and empty_check_d == 0, \
        f"still {empty_check_o} empty origin / {empty_check_d} empty dest"
    node_viol = {n: c for n, c in node_owner.items() if c > 2}
    assert not node_viol, f"dest node >2 zone cap violated: {node_viol}"
    print("  OK: all 496 zones have >=1 generation link and >=1 destination")
    write_origin_dat(os.path.join(OUT_DYNUS, "origin.dat"), new_origin)
    write_destination_dat(os.path.join(OUT_DYNUS, "destination.dat"), new_dest)
    print(f"  wrote origin.dat + destination.dat")

    print("\nSTEP 6: Writing demand.dat (496x496, F10.4 format)")
    demand_496 = build_demand_496(zone507_to_496, n_zones)
    print(f"  matrix sum: {demand_496.sum():.1f} trips/day")
    write_demand_dat(os.path.join(OUT_DYNUS, "demand.dat"), demand_496,
                      n_intervals=2, mult_factor=PEAK_MULT_FACTOR,
                      start_times=START_TIMES)
    write_empty_demand(os.path.join(OUT_DYNUS, "demand_truck.dat"))
    write_empty_demand(os.path.join(OUT_DYNUS, "demand_HOV.dat"))
    print(f"  wrote demand.dat + empty demand_truck.dat / demand_HOV.dat")

    print("\nSTEP 7: Consolidating OMX files in demand_")
    omx_files = [f for f in os.listdir(DEMAND_DIR) if f.endswith(".omx")]
    for fname in omx_files:
        src = os.path.join(DEMAND_DIR, fname)
        dst = os.path.join(OUT_OMX, fname)
        try:
            shape = consolidate_omx_file(src, dst, zone507_to_496, n_zones)
            print(f"  {fname}: {shape} -> {dst}")
        except Exception as e:
            print(f"  {fname}: SKIPPED ({e})")

    print("\n" + "=" * 70)
    print("DONE. Review outputs:")
    print(f"  DynusT: {OUT_DYNUS}")
    print(f"  OMX:    {OUT_OMX}")
    print("Then swap into their respective live directories.")
    print("=" * 70)


if __name__ == "__main__":
    main()
