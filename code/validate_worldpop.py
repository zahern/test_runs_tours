#!/usr/bin/env python
"""
validate_worldpop.py — validate zone population against WorldPop.
================================================================================
Cross-checks the model's zone population (zone_landuse TOTPOP and/or the synthetic
population counts) against WorldPop's gridded population estimate for SEQ, by
zonal-summing the WorldPop raster within each SA2 polygon. Reports per-zone ratios,
correlation, and a suggested rescale, so the land use / synthetic population can be
validated (and optionally re-weighted) to an independent world-population map.

WorldPop raster (Australia, ~100 m). Download once (needs internet):
  https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/AUS/aus_ppp_2020_constrained.tif
  → saved_data/worldpop/aus_ppp_2020_constrained.tif   (set WORLDPOP_TIF)

Run (env with rasterio + geopandas, e.g. geotfenv):
  python validate_worldpop.py
Env: WORLDPOP_TIF, SA2_GEOJSON, ZONE_LANDUSE, SYNTH_POP, POP_COL, WORLDPOP_OUT.
"""
import os
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('worldpop')

_HERE = os.path.dirname(os.path.abspath(__file__))
WORLDPOP_TIF = os.environ.get('WORLDPOP_TIF', os.path.join(_HERE, 'saved_data', 'worldpop', 'aus_ppp_2020_constrained.tif'))
SA2_GEOJSON  = os.environ.get('SA2_GEOJSON',
    (os.path.join(_HERE, 'saved_data', 'sa2_371.geojson')
     if os.path.exists(os.path.join(_HERE, 'saved_data', 'sa2_371.geojson'))
     else os.path.join(_HERE, 'saved_data', 'sa2_seq.geojson')))
ZONE_LANDUSE = os.environ.get('ZONE_LANDUSE', os.path.join(_HERE, '..', 'new_pop', 'saved_data', 'zone_landuse_357.csv'))
SYNTH_POP    = os.environ.get('SYNTH_POP',    os.path.join(_HERE, '..', 'new_pop', 'saved_data', 'synth_population.csv'))
POP_COL      = os.environ.get('POP_COL', 'TOTPOP')
OUT          = os.environ.get('WORLDPOP_OUT', os.path.join(_HERE, 'saved_data', 'worldpop_validation.csv'))


def _zonal_pop(sa2, zcol):
    """WorldPop population summed within each SA2 polygon → Series indexed by zone."""
    try:
        from rasterstats import zonal_stats
        log.info('zonal sum via rasterstats …')
        stats = zonal_stats(sa2.to_crs(4326), WORLDPOP_TIF, stats='sum', nodata=-99999,
                            all_touched=False)
        return pd.Series([(s['sum'] or 0.0) for s in stats], index=sa2[zcol].values)
    except Exception as e:
        log.warning('rasterstats unavailable (%s) — falling back to rasterio.mask', e)
    import rasterio
    from rasterio.mask import mask as rio_mask
    out = {}
    with rasterio.open(WORLDPOP_TIF) as src:
        _sa2 = sa2.to_crs(src.crs)
        nod = src.nodata if src.nodata is not None else -99999
        for _, row in _sa2.iterrows():
            try:
                arr, _ = rio_mask(src, [row.geometry.__geo_interface__], crop=True, filled=True, nodata=nod)
                a = arr[0].astype(float)
                out[row[zcol]] = float(np.nansum(np.where(a == nod, 0.0, np.clip(a, 0, None))))
            except Exception:
                out[row[zcol]] = 0.0
    return pd.Series(out)


def main():
    for _p in (WORLDPOP_TIF, SA2_GEOJSON):
        if not os.path.exists(_p):
            raise SystemExit(f'FATAL: missing {_p} (download WorldPop raster / check SA2 geojson)')
    import geopandas as gpd
    sa2 = gpd.read_file(SA2_GEOJSON)
    zcol = next((c for c in ('SA2_MAIN16', 'SA2_MAIN21', 'ORDER', 'zone', 'zone_id')
                 if c in sa2.columns), sa2.columns[0])
    sa2[zcol] = pd.to_numeric(sa2[zcol], errors='coerce')
    sa2 = sa2[sa2[zcol].notna()].copy()
    sa2[zcol] = sa2[zcol].astype('int64')

    wp = _zonal_pop(sa2, zcol).rename('worldpop').astype(float)
    wp.index = pd.to_numeric(pd.Series(wp.index), errors='coerce').astype('Int64').values
    df = wp.reset_index().rename(columns={'index': 'zone'})

    # model population: zone_landuse POP_COL, and/or synth pop headcount per zone
    model = None
    if os.path.exists(ZONE_LANDUSE):
        lu = pd.read_csv(ZONE_LANDUSE)
        _z = next((c for c in ('SA2_MAIN16', 'ORDER', 'zone', 'zone_id') if c in lu.columns), lu.columns[0])
        if POP_COL in lu.columns:
            model = lu[[_z, POP_COL]].rename(columns={_z: 'zone', POP_COL: 'model_pop'})
            model['zone'] = pd.to_numeric(model['zone'], errors='coerce').astype('Int64')
    if model is None and os.path.exists(SYNTH_POP):
        sp = pd.read_csv(SYNTH_POP, usecols=lambda c: c in ('home_zone', 'SA2_MAIN16', 'zone', 'ORDER'))
        _z = next((c for c in ('home_zone', 'SA2_MAIN16', 'zone', 'ORDER') if c in sp.columns), None)
        if _z:
            model = sp.groupby(_z).size().rename('model_pop').reset_index().rename(columns={_z: 'zone'})
            model['zone'] = pd.to_numeric(model['zone'], errors='coerce').astype('Int64')

    if model is not None:
        df = df.merge(model, on='zone', how='outer')
    df['worldpop'] = df['worldpop'].fillna(0.0)
    if 'model_pop' in df.columns:
        df['model_pop'] = df['model_pop'].fillna(0.0)
        df['ratio'] = df['model_pop'] / df['worldpop'].replace(0, np.nan)
        _tot_wp, _tot_m = df['worldpop'].sum(), df['model_pop'].sum()
        _valid = df['worldpop'] > 10
        _corr = df.loc[_valid, ['worldpop', 'model_pop']].corr().iloc[0, 1] if _valid.sum() > 2 else np.nan
        df['flag'] = np.where(_valid & ((df['ratio'] > 1.5) | (df['ratio'] < 0.67)), 'CHECK', '')
        log.info('WorldPop total=%.0f  model total=%.0f  overall scale (model/WP)=%.3f',
                 _tot_wp, _tot_m, _tot_m / max(_tot_wp, 1))
        log.info('per-zone corr(model, WorldPop)=%.3f  |  %d zones flagged (ratio outside 0.67–1.5)',
                 _corr, int((df['flag'] == 'CHECK').sum()))
        df['worldpop_rescale'] = (df['worldpop'] / max(_tot_wp, 1)) * _tot_m  # WP shares at model total
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    df.to_csv(OUT, index=False)
    log.info('wrote WorldPop validation (%d zones) → %s', len(df), OUT)


if __name__ == '__main__':
    main()
