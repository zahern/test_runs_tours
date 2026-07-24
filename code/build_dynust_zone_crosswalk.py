"""
build_dynust_zone_crosswalk.py
================================
Build a spatial nearest-centroid crosswalk between the ABM's zone system
(SA2 / ORDER, from seq_zone_ids_.csv) and a DynusT network's own zone
numbering (1..N, the "izone" field used in vehicle.dat).

WHY THIS EXISTS
---------------
write_dynus_vehicle_dat() (stage6_plan_finalisation.py) writes the ABM's
OTAZ/DTAZ straight into vehicle.dat's zone fields, which is only correct
if the ABM's zone numbers already match the target DynusT network's zone
numbering 1:1. That does NOT hold for DTA_Brisbane_tested:
  - ABM zones (seq_zone_ids_.csv):    330 zones, numbered 1-330 (ORDER)
  - DTA_Brisbane_tested zones:        1515 zones, numbered 1-1515
  - TAZ_mapping.dat's "TAZ" column is a third, unrelated legacy zone id
    (values up to ~2710) - not usable as a crosswalk to either ABM scheme.
There is no existing crosswalk file for this network anywhere in the repo
(confirmed: `origin_link_map`, the one hook meant for this, is never
populated anywhere). This script builds one from first principles using
actual georeferenced data:
  1. DynusT zone centroids: origin.dat gives each zone's generation-link
     node ids; GIS_Network.xls's NODE sheet gives those node ids' real
     coordinates (mislabeled "Longitude"/"Latitude" but are actually
     GDA94 / MGA Zone 56 easting/northing, EPSG:28356 - confirmed by
     round-tripping to WGS84 and checking the result lands in Brisbane).
     Centroid = mean coordinate of a zone's connector nodes.
  2. ABM zone centroids: seq_zone_ids_.csv's longitude/latitude columns,
     keyed by ORDER.
  3. Nearest-neighbour match (in projected metres) from each ABM zone to
     its closest DynusT zone, recording the distance.

The distance column is what lets a downstream step decide which ABM zones
are actually "in" this test network's coverage area. Empirically there is
a sharp cliff in the data: ~228 zones match within 8 km (greater Brisbane),
then the next-nearest zones jump straight to 59-114 km away (Sunshine
Coast, Gold Coast, Gympie, even northern NSW) - i.e. ABM zones that simply
aren't part of this smaller test network at all. That confirms the
"smaller region of Brisbane" framing: don't force those distant zones'
trips down to link-level resolution, drop them.

Usage
-----
    python build_dynust_zone_crosswalk.py \\
        --dta-dir "Z:\\test_runs_tours\\DTA\\DTA_Brisbane_tested" \\
        --out saved_data/abm_to_dynust_zone_crosswalk.csv

Output columns: ORDER, SA2_NAME16, dynust_zone, dist_m, in_network
(in_network is a convenience bool using --max-dist-m, default 8000).
"""
import argparse
import os

import numpy as np
import pandas as pd
from pyproj import Transformer

NETWORK_CRS = 'EPSG:28356'   # GDA94 / MGA Zone 56 - confirmed against Brisbane CBD coords
WGS84 = 'EPSG:4326'


def parse_origin_dat(path: str) -> dict[int, set[int]]:
    """Parse DynusT origin.dat into {zone_id: {connector node ids}}.

    Format: repeating blocks of
        <zone_id> <n_connectors> <flag>
        <n_connectors> lines of "usec dsec length"
    """
    zone_nodes: dict[int, set[int]] = {}
    with open(path) as f:
        lines = f.read().split('\n')
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        zid, n_conn = int(parts[0]), int(parts[1])
        nodes: set[int] = set()
        for k in range(1, n_conn + 1):
            if i + k >= n:
                break
            toks = lines[i + k].split()
            if len(toks) >= 2:
                nodes.add(int(toks[0]))
                nodes.add(int(toks[1]))
        zone_nodes[zid] = nodes
        i += n_conn + 1
    return zone_nodes


def build_dynust_zone_centroids(dta_dir: str) -> pd.DataFrame:
    """Return DataFrame[zone, utm_e, utm_n, lon, lat, n_nodes] for every
    DynusT zone in the network at dta_dir, using origin.dat connector
    nodes + GIS_Network.xls node coordinates."""
    zone_nodes = parse_origin_dat(os.path.join(dta_dir, 'origin.dat'))

    xls = pd.ExcelFile(os.path.join(dta_dir, 'GIS_Network.xls'))
    nodes = xls.parse('NODE').set_index('ID')[['Longitude', 'Latitude']]

    rows = []
    for zid, nids in zone_nodes.items():
        coords = nodes.loc[nodes.index.intersection(nids)]
        if len(coords) == 0:
            continue
        rows.append({
            'zone':    zid,
            'utm_e':   coords['Longitude'].mean(),
            'utm_n':   coords['Latitude'].mean(),
            'n_nodes': len(coords),
        })
    zc = pd.DataFrame(rows)

    t = Transformer.from_crs(NETWORK_CRS, WGS84, always_xy=True)
    zc['lon'], zc['lat'] = t.transform(zc['utm_e'].values, zc['utm_n'].values)
    return zc


def build_zone_centroids_from_gpkg(zone_gpkg: str) -> pd.DataFrame:
    """Zone centroids from a georeferenced zone-polygon GeoPackage.

    Path for networks that ship no GIS_Network.xls — e.g. SEQ_Running_UE,
    whose xy.dat is in an unidentified local grid, but whose 349 zones are
    already georeferenced in demand_/dta_349_zones.gpkg (EPSG:4326,
    column zone_id, count matches network.dat's header exactly).
    Returns the same frame shape as build_dynust_zone_centroids.
    """
    import geopandas as gpd
    gdf = gpd.read_file(zone_gpkg)
    zid_col = 'zone_id' if 'zone_id' in gdf.columns else gdf.columns[0]
    proj = gdf.to_crs(NETWORK_CRS)
    cent = proj.geometry.centroid
    zc = pd.DataFrame({
        'zone':    pd.to_numeric(gdf[zid_col], errors='coerce').astype(int),
        'utm_e':   cent.x.values,
        'utm_n':   cent.y.values,
        'n_nodes': 0,
    })
    t = Transformer.from_crs(NETWORK_CRS, WGS84, always_xy=True)
    zc['lon'], zc['lat'] = t.transform(zc['utm_e'].values, zc['utm_n'].values)
    return zc


def build_crosswalk(dta_dir: str, abm_zone_csv: str, max_dist_m: float = 8000.0,
                    zone_gpkg: str = None) -> pd.DataFrame:
    zc = (build_zone_centroids_from_gpkg(zone_gpkg) if zone_gpkg
          else build_dynust_zone_centroids(dta_dir))

    _raw = pd.read_csv(abm_zone_csv)
    if 'sa2_code' in _raw.columns:
        # Full-SEQ reference schema (sa2_zones_reference_357.csv, 371 zones):
        # the ABM zone id IS the SA2 code (NEW_POP=1 scope). Output column
        # stays named ORDER because apply_dynust_zone_crosswalk matches its
        # values against whatever OTAZ/DTAZ carry - SA2 codes in this scope.
        abm = pd.DataFrame({
            'ORDER':      pd.to_numeric(_raw['sa2_code'], errors='coerce').astype(int),
            'SA2_NAME16': _raw.get('sa2_name', ''),
            'longitude':  pd.to_numeric(_raw['centroid_lon'], errors='coerce'),
            'latitude':   pd.to_numeric(_raw['centroid_lat'], errors='coerce'),
        }).dropna(subset=['longitude', 'latitude'])
    else:
        abm = _raw[['SA2_MAIN16', 'ORDER', 'SA2_NAME16', 'longitude', 'latitude']]
    t = Transformer.from_crs(WGS84, NETWORK_CRS, always_xy=True)
    abm['utm_e'], abm['utm_n'] = t.transform(abm['longitude'].values, abm['latitude'].values)

    dz = zc[['zone', 'utm_e', 'utm_n']].to_numpy()
    best_zone = np.empty(len(abm), dtype=int)
    best_dist = np.empty(len(abm), dtype=float)
    for i, (e, n) in enumerate(zip(abm['utm_e'].to_numpy(), abm['utm_n'].to_numpy())):
        d2 = (dz[:, 1] - e) ** 2 + (dz[:, 2] - n) ** 2
        j = np.argmin(d2)
        best_zone[i] = int(dz[j, 0])
        best_dist[i] = np.sqrt(d2[j])

    abm['dynust_zone'] = best_zone
    abm['dist_m'] = best_dist
    abm['in_network'] = abm['dist_m'] <= max_dist_m
    return abm[['ORDER', 'SA2_NAME16', 'dynust_zone', 'dist_m', 'in_network']]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dta-dir', default=r'Z:\test_runs_tours\DTA\DTA_Brisbane_tested')
    ap.add_argument('--zone-gpkg', default=None,
                    help='Zone-polygon GeoPackage with a zone_id column - use '
                         'instead of origin.dat+GIS_Network.xls for networks '
                         'without the xls (SEQ_Running_UE: '
                         '../demand_/dta_349_zones.gpkg).')
    ap.add_argument('--abm-zones', default='seq_zone_ids_.csv')
    ap.add_argument('--max-dist-m', type=float, default=8000.0,
                     help='ABM zones farther than this from any DynusT zone centroid '
                          'are flagged in_network=False (default 8km — see module '
                          'docstring for the empirical cliff this is based on).')
    ap.add_argument('--out', default='saved_data/abm_to_dynust_zone_crosswalk.csv')
    args = ap.parse_args()

    cw = build_crosswalk(args.dta_dir, args.abm_zones, args.max_dist_m,
                         zone_gpkg=args.zone_gpkg)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cw.to_csv(args.out, index=False)
    n_in = int(cw['in_network'].sum())
    print(f"Crosswalk saved -> {args.out}")
    print(f"  {n_in}/{len(cw)} ABM zones matched within {args.max_dist_m/1000:.1f} km "
          f"of a DynusT zone centroid (kept)")
    print(f"  {len(cw) - n_in}/{len(cw)} ABM zones are outside this network's coverage "
          f"(dropped — nearest match is {cw.loc[~cw['in_network'],'dist_m'].min()/1000:.1f}"
          f"-{cw['dist_m'].max()/1000:.1f} km away)")
