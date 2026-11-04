"""
rebuild_dta_349_zones_gpkg.py
=============================
Regenerate demand_/dta_349_zones.gpkg with TRUE EPSG:4326 coordinates.

The existing gpkg was built from the DynusT local grid through a broken CRS
conversion - its geometries claim EPSG:4326 but sit at latitude ~-89 (the
South Pole), so the ABM->SEQ_Running_UE zone crosswalk matched 0/330 zones.

Source geometry: DTA/SEQ_Running_UE/zone.dat
    line 3:            "<n_feature_points> <n_zones>"   (142152  349)
    coordinate block:  "<node#>  <x>,  <y>"             (DynusT local grid)
    zone block:        "<zone#>  <n>:  id, id, id, ..." (boundary ring,
                       wrapping across lines until n ids are consumed)

Transform: the exact 3-control-point affine from prepare_network/fix_xy.py
(itself a copy of dynus.py get_affine) - DynusT local grid -> GDA lat/lon.
zone.dat's coordinates are in the same grid (one control point, 563910/135885,
sits inside its value range).

Usage (HPC, zigenv - needs geopandas/shapely):
    cd ~/test_runs_tours/prepare_network
    python rebuild_dta_349_zones_gpkg.py
Then rebuild the crosswalk:
    cd ../code
    rm -f saved_data/abm_to_seq_running_ue_crosswalk.csv
    NEW_POP=1 python predict_and_export_dta_tested.py
"""
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZONE_DAT = os.path.normpath(os.path.join(HERE, '..', 'DTA', 'SEQ_Running_UE', 'zone.dat'))
OUT_GPKG = os.path.normpath(os.path.join(HERE, '..', 'demand_', 'dta_349_zones.gpkg'))


# ── affine (exact copy of fix_xy.py / dynus.py get_affine) ──────────────────
def get_affine():
    lat_lon = np.array([
        [-27.57639,   152.699837],
        [-27.263422,  153.079553],
        [-28.1682445, 153.543538],
    ])
    x_y = np.array([
        [264800, 372600],
        [399690, 498300],
        [563910, 135885],
    ])
    lat, lon = lat_lon[:, 0], lat_lon[:, 1]
    A = np.zeros((6, 6))
    b = np.zeros(6)
    for i in range(3):
        A[i,   0] = lon[i]; A[i,   1] = lat[i]; A[i,   2] = 1; b[i]   = x_y[i, 0]
        A[i+3, 3] = lon[i]; A[i+3, 4] = lat[i]; A[i+3, 5] = 1; b[i+3] = x_y[i, 1]
    return np.linalg.solve(A, b)


def xy_to_lonlat(x, y, params):
    a, b_, c, d, e, f = params
    det = a * e - b_ * d
    lon = (e * (x - c) - b_ * (y - f)) / det
    lat = (a * (y - f) - d * (x - c)) / det
    return lon, lat


def parse_zone_dat(path):
    """Return (coords: dict node_id -> (x, y), zones: dict zone_id -> [node ids])."""
    with open(path) as f:
        lines = f.read().splitlines()

    n_points, n_zones = (int(t) for t in lines[2].split())
    coords = {}
    i = 4  # skip 2 header lines, counts line, coord header line
    while i < len(lines) and len(coords) < n_points:
        toks = lines[i].replace(',', ' ').split()
        if len(toks) >= 3:
            coords[int(toks[0])] = (float(toks[1]), float(toks[2]))
        i += 1

    # skip the "zone #, number of nodes, node #'s" header
    while i < len(lines) and 'zone' in lines[i].lower():
        i += 1

    zones = {}
    cur_zone, need, acc = None, 0, []
    zone_start = re.compile(r'^\s*(\d+)\s+(\d+)\s*:\s*(.*)$')
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        m = zone_start.match(line)
        if m and cur_zone is None:
            cur_zone, need = int(m.group(1)), int(m.group(2))
            acc = [int(t) for t in m.group(3).replace(',', ' ').split() if t.strip()]
        elif cur_zone is not None:
            acc.extend(int(t) for t in line.replace(',', ' ').split() if t.strip())
        if cur_zone is not None and len(acc) >= need:
            zones[cur_zone] = acc[:need]
            cur_zone, need, acc = None, 0, []

    assert len(zones) == n_zones, f'parsed {len(zones)} zones, expected {n_zones}'
    return coords, zones


def main():
    import geopandas as gpd
    from shapely.geometry import Polygon

    print(f'Parsing {ZONE_DAT} ...')
    coords, zones = parse_zone_dat(ZONE_DAT)
    print(f'  {len(coords):,} feature points, {len(zones)} zones')

    params = get_affine()
    geoms, zone_ids, bad = [], [], []
    for zid in sorted(zones):
        pts = [coords[n] for n in zones[zid] if n in coords]
        if len(pts) < 3:
            bad.append(zid)
            continue
        ring = [xy_to_lonlat(x, y, params) for x, y in pts]
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)   # standard self-intersection repair
        geoms.append(poly)
        zone_ids.append(zid)
    if bad:
        print(f'  WARNING: {len(bad)} zones had <3 points and were skipped: {bad}')

    gdf = gpd.GeoDataFrame({'zone_id': zone_ids}, geometry=geoms, crs='EPSG:4326')

    # ── sanity: everything must land in SEQ, not at the South Pole ────────
    cent = gdf.geometry.centroid
    ok = cent.x.between(150, 155) & cent.y.between(-30, -25)
    print(f'  centroids inside SEQ bounds: {int(ok.sum())}/{len(gdf)} '
          f'(lon {cent.x.min():.3f}..{cent.x.max():.3f}, '
          f'lat {cent.y.min():.3f}..{cent.y.max():.3f})')
    if ok.mean() < 0.95:
        raise SystemExit('ABORT: <95% of zones inside SEQ bounds - '
                         'affine assumption is wrong for zone.dat, not writing.')

    if os.path.exists(OUT_GPKG):
        bak = OUT_GPKG + '.broken_south_pole.bak'
        os.replace(OUT_GPKG, bak)
        print(f'  archived broken gpkg -> {bak}')
    gdf.to_file(OUT_GPKG, layer='dta_349_zones', driver='GPKG')
    print(f'  wrote {len(gdf)} zones -> {OUT_GPKG}')


if __name__ == '__main__':
    main()
