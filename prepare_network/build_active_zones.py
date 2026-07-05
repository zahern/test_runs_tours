"""
build_active_zones.py

Builds a clean DynusT model using ONLY the zones that have genuine SEQ
network connections -- dropping the ~194 dormant SA2 zones (regional QLD:
Cairns, Townsville, Outback, etc.) that fall entirely outside the SEQ
network footprint and were previously kept as fake gateway-node placeholders.

Result: ~302 zones, all with real centroid connectors, no workarounds.

Outputs:
  DynusT files  ->  Z:\\test_runs_tours\\prepare_network\\new\\active_only\\
  Not-modelled outline  ->  Z:\\test_runs_tours\\prepare_network\\new\\not_modelled.shp
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
from scipy.spatial import cKDTree
from collections import defaultdict

MODEL_DIR       = r"C:\Users\ahernz\SEQ_DTA_2025"
BACKUP_NETWORK  = os.path.join(MODEL_DIR, "pre_507_conversion_backup", "network.dat")
BACKUP_ORIGIN   = os.path.join(MODEL_DIR, "pre_native_507_backup", "origin.dat")
BACKUP_DEST     = os.path.join(MODEL_DIR, "pre_native_507_backup", "destination.dat")
CROSSWALK_CSV   = os.path.join(MODEL_DIR, "zone_crosswalk_507.csv")
SA2_SHP         = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
DEMAND_OMX      = r"Z:\test_runs_tours\demand_\demand_integrated_sa2.omx"
OUT_DIR         = r"Z:\test_runs_tours\prepare_network\new\active_only"
OUT_NOT_MOD     = r"Z:\test_runs_tours\prepare_network\new\not_modelled.shp"
TARGET_CRS      = 7856
PEAK_MULT       = 0.10
START_TIMES     = [0.0, 60.0, 120.0]

# 11 donut pairs (inner absorbed into outer) -- same as consolidate_zones_496.py
INNER_TO_OUTER = {174:175,185:186,190:189,211:210,387:388,
                  442:443,444:445,446:447,486:488,493:495,494:495}

os.makedirs(OUT_DIR, exist_ok=True)


def load_crosswalk():
    cw = pd.read_csv(CROSSWALK_CSV)
    return dict(zip(cw["old_zone_id"], cw["super_zone_id"]))  # old -> 507-zone id


def build_507_to_496():
    effective = {z: INNER_TO_OUTER.get(z, z) for z in range(1, 508)}
    distinct  = sorted(set(effective.values()))
    assert len(distinct) == 496
    new_id = {old: new for new, old in enumerate(distinct, start=1)}
    return {z: new_id[effective[z]] for z in range(1, 508)}  # 507 -> 496


def parse_network(path):
    with open(path) as f:
        lines = f.readlines()
    h = lines[0].split()
    n_zones, n_nodes, n_links, k = int(h[0]), int(h[1]), int(h[2]), int(h[3])
    node_lines = lines[1:1+n_nodes]
    link_lines = lines[1+n_nodes:1+n_nodes+n_links]
    node_ids, node_old_zone = [], {}
    for l in node_lines:
        nid, zid = (int(x) for x in l.split())
        node_ids.append(nid)
        node_old_zone[nid] = zid
    return dict(n_nodes=n_nodes, n_links=n_links, k=k,
                node_ids=node_ids, node_old_zone=node_old_zone,
                link_lines=link_lines)


def parse_origin(path):
    result = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if not p: i += 1; continue
        z, cnt, mode = int(p[0]), int(p[1]), int(p[2])
        links = []
        for j in range(cnt):
            up, dn, w = lines[i+1+j].split()
            links.append((int(up), int(dn), float(w)))
        result[z] = links
        i += 1 + cnt
    return result


def parse_destination(path):
    result = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p: continue
            z, cnt = int(p[0]), int(p[1])
            result[z] = [int(x) for x in p[2:2+cnt]]
    return result


def main():
    print("="*70)
    print("STEP 1: Building zone mappings")
    old_to_507 = load_crosswalk()
    z507_to_496 = build_507_to_496()
    old_to_496  = {z: z507_to_496[sid] for z, sid in old_to_507.items()}

    # Parse original network to get node -> 496-zone assignments
    net = parse_network(BACKUP_NETWORK)
    node_to_496 = {nid: old_to_496[net["node_old_zone"][nid]]
                   for nid in net["node_ids"]}

    # Active 496-zones = those that have at least one node
    active_496 = sorted(set(node_to_496.values()))
    n_active = len(active_496)
    dormant_496 = sorted(set(range(1, 497)) - set(active_496))
    print(f"  Active zones (have real nodes): {n_active}")
    print(f"  Dormant zones (no nodes, removed): {len(dormant_496)}")

    # Renumber active zones 1..n_active
    remap = {old496: new for new, old496 in enumerate(active_496, start=1)}
    old_to_new = {nid: remap[node_to_496[nid]] for nid in net["node_ids"]}

    # Also build: 507-zone id -> new zone id (for demand matrix)
    # 507-zone is active only if its 496 mapping is in active_496
    z507_to_new = {}
    for z507, z496 in z507_to_496.items():
        if z496 in active_496:
            z507_to_new[z507] = remap[z496]
        # else: dormant, no mapping

    print(f"\nSTEP 2: Building origin/destination for {n_active} active zones")
    old_origin = parse_origin(BACKUP_ORIGIN)
    old_dest   = parse_destination(BACKUP_DEST)

    new_origin  = {z: [] for z in range(1, n_active+1)}
    seen_orig   = {z: set() for z in range(1, n_active+1)}
    for old_z, links in old_origin.items():
        old_super = old_to_507[old_z]
        z496      = z507_to_496[old_super]
        if z496 not in active_496:
            continue  # skip dormant old zones
        new_z = remap[z496]
        for (up, dn, w) in links:
            key = (up, dn)
            if key not in seen_orig[new_z]:
                seen_orig[new_z].add(key)
                new_origin[new_z].append((up, dn, w))

    new_dest    = {z: [] for z in range(1, n_active+1)}
    seen_dest   = {z: set() for z in range(1, n_active+1)}
    node_owner  = {}
    for old_z, nodes in old_dest.items():
        old_super = old_to_507[old_z]
        z496      = z507_to_496[old_super]
        if z496 not in active_496:
            continue
        new_z = remap[z496]
        for n in nodes:
            if n in seen_dest[new_z]: continue
            if node_owner.get(n, 0) >= 2: continue
            seen_dest[new_z].add(n)
            new_dest[new_z].append(n)
            node_owner[n] = node_owner.get(n, 0) + 1

    empty_o = [z for z,v in new_origin.items() if not v]
    empty_d = [z for z,v in new_dest.items() if not v]
    print(f"  Zones still needing connections: origin={len(empty_o)} dest={len(empty_d)}")
    assert not empty_o and not empty_d, "Some active zones have no connectors -- unexpected"

    viol = {n: c for n, c in node_owner.items() if c > 2}
    assert not viol, f"dest node >2 zones: {viol}"
    print(f"  OK: all {n_active} zones have >=1 connector, no destination-node violations")

    print(f"\nSTEP 3: Writing network.dat")
    with open(os.path.join(OUT_DIR, "network.dat"), "w") as f:
        f.write(f"{n_active}  {net['n_nodes']}  {net['n_links']}      {net['k']}      0\n")
        for nid in net["node_ids"]:
            f.write(f"{nid} {old_to_new[nid]}\n")
        f.writelines(net["link_lines"])
    print(f"  wrote network.dat  ({n_active} zones)")

    print(f"\nSTEP 4: Writing origin.dat / destination.dat")
    with open(os.path.join(OUT_DIR, "origin.dat"), "w") as f:
        for z in range(1, n_active+1):
            links = new_origin[z]
            f.write(f"{z} {len(links)} 0\n")
            for (up, dn, w) in links:
                f.write(f"{up} {dn} {w:.3f}\n")
    with open(os.path.join(OUT_DIR, "destination.dat"), "w") as f:
        for z in range(1, n_active+1):
            nodes = new_dest[z]
            f.write(f"{z} {len(nodes)} " + " ".join(str(n) for n in nodes) + "\n")
    print(f"  wrote origin.dat + destination.dat")

    print(f"\nSTEP 5: Loading SA2 polygons and writing zone.dat")
    gdf = gpd.read_file(SA2_SHP)
    gdf["SA2_CODE21"] = gdf["SA2_CODE21"].astype(str).str.strip()

    # Get OMX SA2 codes (authoritative list of 507 codes in sorted order)
    with h5py.File(DEMAND_OMX, "r") as fh:
        sa2_codes_507 = [s.decode() for s in fh["lookup"]["SA2_CODE"][:]]
    assert len(sa2_codes_507) == 507

    code_to_super507 = {code: i+1 for i, code in enumerate(sa2_codes_507)}

    # For each active 507-zone, get its polygon (union with absorbed inner if needed)
    inner_geoms = defaultdict(list)
    for inner, outer in INNER_TO_OUTER.items():
        for code, s507 in code_to_super507.items():
            if s507 == inner:
                rows = gdf[gdf["SA2_CODE21"] == code]
                if not rows.empty:
                    g = rows.iloc[0].geometry
                    if g.geom_type == "MultiPolygon":
                        g = max(g.geoms, key=lambda x: x.area)
                    inner_geoms[outer].append(g.to_crs(TARGET_CRS) if hasattr(g, "to_crs") else g)

    from shapely.ops import unary_union

    zone_geoms = {}   # new_zone_id -> polygon
    not_modelled_geoms = []

    for code, s507 in code_to_super507.items():
        z496 = z507_to_496[s507]
        rows = gdf[gdf["SA2_CODE21"] == code]
        if rows.empty:
            continue
        if rows.crs is None:
            rows = rows.set_crs(7844)
        rows_m = rows.to_crs(TARGET_CRS)
        geom = rows_m.iloc[0].geometry
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda x: x.area)

        if z496 in active_496:
            new_z = remap[z496]
            extras = inner_geoms.get(s507, [])
            if extras:
                geom = unary_union([geom] + extras)
            zone_geoms[new_z] = geom
        else:
            # dormant zone -- add to not-modelled outline
            not_modelled_geoms.append(geom)

    assert len(zone_geoms) == n_active, f"zone_geoms count mismatch: {len(zone_geoms)} vs {n_active}"

    # Write zone.dat
    from scipy.spatial import cKDTree as _KDT
    def poly_to_ring(geom):
        ext = list(geom.exterior.coords)[:-1]
        if not list(geom.interiors):
            return ext + [ext[0]]
        ring = list(ext)
        for interior in geom.interiors:
            hole = list(interior.coords)[:-1]
            arr = np.array(ring); tree = _KDT(arr)
            dists, idxs = tree.query(np.array(hole))
            hj = int(np.argmin(dists)); ei = int(idxs[hj])
            hr = hole[hj:] + hole[:hj] + [hole[hj]]
            ring = ring[:ei+1] + [ring[ei]] + hr + [ring[ei]] + ring[ei+1:]
        return ring + [ring[0]]

    pt_idx = {}; ctr = 1; zone_pts = {}
    for z in range(1, n_active+1):
        geom = zone_geoms[z]
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda x: x.area)
        pts = poly_to_ring(geom)
        ids = []
        for p in pts:
            if p not in pt_idx: pt_idx[p] = ctr; ctr += 1
            ids.append(pt_idx[p])
        zone_pts[z] = ids

    unique_pts = list(pt_idx.keys())
    with open(os.path.join(OUT_DIR, "zone.dat"), "w") as f:
        f.write(" This file defines zone regions\n")
        f.write(" number of feature points, number of zones\n")
        f.write(f"     {len(unique_pts)}      {n_active}\n")
        f.write("node #, x-coordinate, y-coordinate\n")
        for i, (x, y) in enumerate(unique_pts, start=1):
            f.write(f"      {i}     {x:.3f},    {y:.3f}\n")
        f.write("zone #, number of nodes, node #'s\n")
        for z in range(1, n_active+1):
            ids = zone_pts[z]
            f.write(f"   {z}     {len(ids)}: " + ", ".join(str(p) for p in ids) + ",\n")
    print(f"  wrote zone.dat  ({len(unique_pts)} feature points, {n_active} zones)")

    print(f"\nSTEP 6: Writing demand.dat ({n_active}x{n_active})")
    with h5py.File(DEMAND_OMX, "r") as fh:
        mat507 = fh["data"]["demand_combined_car"][:]
        sa2_codes_omx = [s.decode() for s in fh["lookup"]["SA2_CODE"][:]]

    omx_pos = {code: i for i, code in enumerate(sa2_codes_omx)}

    # Map OMX rows to active new zone IDs
    zone_new_to_omx = defaultdict(list)
    for code, s507 in code_to_super507.items():
        new_z = z507_to_new.get(s507)
        if new_z is None: continue
        idx = omx_pos.get(code)
        if idx is not None:
            zone_new_to_omx[new_z].append(idx)

    demand = np.zeros((n_active, n_active), dtype=np.float64)
    for zi in range(1, n_active+1):
        for zj in range(1, n_active+1):
            ii = zone_new_to_omx.get(zi, [])
            jj = zone_new_to_omx.get(zj, [])
            if ii and jj:
                demand[zi-1, zj-1] = mat507[np.ix_(ii, jj)].sum()

    print(f"  demand matrix: sum={demand.sum():.1f}  max={demand.max():.4f}  "
          f"values>=10000: {(demand>=10000).sum()}")

    flat = demand.flatten()
    with open(os.path.join(OUT_DIR, "demand.dat"), "w") as f:
        f.write(f"{2:5d} {PEAK_MULT:.3f}\n")
        f.write("".join(f"{t:8.1f}" for t in START_TIMES) + "\n")
        for k in range(2):
            f.write(f"Start Time =    {START_TIMES[k]:.1f}\n")
            for i in range(0, len(flat), 6):
                chunk = flat[i:i+6]
                f.write("".join(f" {v:.4f}" for v in chunk) + "\n")

    with open(os.path.join(OUT_DIR, "demand_truck.dat"), "w") as f: f.write("0  1.0\n")
    with open(os.path.join(OUT_DIR, "demand_HOV.dat"), "w") as f:   f.write("0  1.0\n")
    print(f"  wrote demand.dat + empty truck/HOV files")

    print(f"\nSTEP 7: Writing not_modelled.shp ({len(not_modelled_geoms)} zones)")
    if not_modelled_geoms:
        gdf_nm = gpd.GeoDataFrame(
            {"zone": list(range(1, len(not_modelled_geoms)+1))},
            geometry=not_modelled_geoms, crs=TARGET_CRS
        ).to_crs(4326)
        gdf_nm.to_file(OUT_NOT_MOD)
        print(f"  wrote {OUT_NOT_MOD}")
        print(f"  (load this in any GIS tool to see which SA2 areas fall outside the DTA network)")

    print("\n" + "="*70)
    print(f"DONE. {n_active} active zones, {len(dormant_496)} dormant zones removed.")
    print(f"Review outputs in {OUT_DIR}, then swap into the live model directory.")
    print("="*70)


if __name__ == "__main__":
    main()
