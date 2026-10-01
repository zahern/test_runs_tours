#!/usr/bin/env python
"""
build_sa2_371_geojson.py — the 371 modelled SA2 polygons as GeoJSON.
================================================================================
Filters the full ABS SA2 shapefile to the 371 modelled SA2 codes and writes
saved_data/sa2_371.geojson (EPSG:4326) carrying SA2_MAIN16 + ORDER. This replaces
the old 329-zone sa2_seq.geojson so WorldPop validation and Overture building
aggregation cover ALL model zones (no coverage-artifact gap).

Run (env with geopandas): python build_sa2_371_geojson.py
Env: ZONE_IDS_371, SA2_SHP, SA2_371_OUT.
"""
import os
import logging

import pandas as pd
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('sa2_371')

_HERE = os.path.dirname(os.path.abspath(__file__))
ZONE_IDS_371 = os.environ.get('ZONE_IDS_371', os.path.join(_HERE, 'saved_data', 'seq_zone_ids_371.csv'))
# Two polygon sources, unioned for full 371 coverage: the demand-model polygons
# (include the custom SPLIT zones the migration added) + the ABS SA2 2016
# shapefile (standard SA2s). Neither alone covers all 371; together they do.
DEMAND_GPKG  = os.environ.get('DEMAND_GPKG', os.path.join(_HERE, '..', 'demand_', 'zone_demand_polygons.gpkg'))
SA2_SHP      = os.environ.get('SA2_SHP', os.path.join(_HERE, 'saved_data', 'SA2_2021_AUST_SHP_GDA2020', 'SA2_2016_AUST.shp'))
OUT          = os.environ.get('SA2_371_OUT', os.path.join(_HERE, 'saved_data', 'sa2_371.geojson'))


def _load_source(path, code_col):
    if not os.path.exists(path):
        return None
    g = gpd.read_file(path)
    if code_col not in g.columns:
        return None
    g = g[[code_col, 'geometry']].copy()
    g['SA2_MAIN16'] = pd.to_numeric(g[code_col], errors='coerce').astype('Int64')
    g = g[g['SA2_MAIN16'].notna()].to_crs(4326)
    return g[['SA2_MAIN16', 'geometry']]


def main():
    if not os.path.exists(ZONE_IDS_371):
        raise SystemExit(f'FATAL: missing {ZONE_IDS_371}')
    ids = pd.read_csv(ZONE_IDS_371)
    _idcol = next((c for c in ('SA2_MAIN16', 'sa2_code', 'SA2_CODE') if c in ids.columns), None)
    if _idcol is None:
        raise SystemExit(f'FATAL: no SA2 code column in {ZONE_IDS_371} (cols {list(ids.columns)})')
    ids['SA2_MAIN16'] = pd.to_numeric(ids[_idcol], errors='coerce').astype('Int64')
    codes = set(ids['SA2_MAIN16'].dropna().astype(int))
    log.info('target modelled zones: %d SA2 codes', len(codes))

    # PRIMARY: demand-model polygons (has the split zones); FILL: ABS shapefile.
    parts, have = [], set()
    for label, path, col in (('demand polygons', DEMAND_GPKG, 'sa2_code'),
                             ('ABS SA2 2016', SA2_SHP, 'SA2_MAIN16')):
        g = _load_source(path, col)
        if g is None:
            log.warning('  source unavailable/skipped: %s (%s)', label, path)
            continue
        g = g[g['SA2_MAIN16'].astype(int).isin(codes) & ~g['SA2_MAIN16'].astype(int).isin(have)]
        if len(g):
            parts.append(g)
            have |= set(g['SA2_MAIN16'].astype(int))
            log.info('  +%d polygons from %s (cumulative %d/%d)', len(g), label, len(have), len(codes))
    if not parts:
        raise SystemExit('FATAL: no polygon sources available')
    gf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry='geometry', crs=4326)
    # carry ORDER from the model reference
    if 'ORDER' in ids.columns:
        gf = gf.merge(ids[['SA2_MAIN16', 'ORDER']].drop_duplicates('SA2_MAIN16'),
                      on='SA2_MAIN16', how='left')

    missing = codes - have
    log.info('total %d/%d SA2 polygons (union of sources)', len(have), len(codes))
    if missing:
        log.warning('  %d codes still missing (e.g. %s)', len(missing), sorted(missing)[:8])

    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    gf.to_file(OUT, driver='GeoJSON')
    log.info('wrote %d SA2 polygons → %s', len(gf), OUT)


if __name__ == '__main__':
    main()
