"""
20b_build_sa2_2016_2021_concordance.py
=======================================
Build a 2016→2021 SA2 boundary concordance using spatial area-overlap.

The 329-zone ORDER system (seq_zone_ids_.csv) uses SA2 2016 codes (SA2_MAIN16).
22 of those codes were split or renamed between the 2016 and 2021 ASGS editions,
so they have no direct match in the 507-zone new_pop (which uses 2021 codes).

This script spatially intersects the 2016 and 2021 SA2 shapefiles and assigns
each 2016 SA2 to the 2021 SA2 with the greatest area overlap.

Outputs
-------
demand_/sa2_2016_to_2021_concordance.csv
    sa2_main16  : 2016 SA2 code
    sa2_name16  : 2016 SA2 name
    sa2_code21  : best-matching 2021 SA2 code  (largest area overlap)
    sa2_name21  : 2021 SA2 name
    overlap_km2 : area of intersection (km²)
    overlap_frac: overlap / 2016 area

Usage
-----
    cd Z:/test_runs_tours/demand_
    python 20b_build_sa2_2016_2021_concordance.py

Requirements: geopandas, pyproj  (both available in zigenv)

Run this BEFORE 20_build_sa2_to_order_crosswalk.py.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

DEMAND_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR   = os.path.join(DEMAND_DIR, '..', 'code')
SA1_DIR    = os.path.join(DEMAND_DIR, '..', 'SA1_SHP')

SHP_2016 = os.path.join(SA1_DIR, 'SA2_2016_AUST.shp')
SHP_2021 = os.path.join(DEMAND_DIR, '..', 'SA2_2021_AUST_SHP_GDA2020',
                         'SA2_2021_AUST_GDA2020.shp')
ORDER_CSV = os.path.join(CODE_DIR, 'seq_zone_ids_.csv')
SA2_XWALK = os.path.join(DEMAND_DIR, 'sa2_crosswalk_507.csv')
OUT_CSV   = os.path.join(DEMAND_DIR, 'sa2_2016_to_2021_concordance.csv')

# Area-of-interest: QLD + Richmond-Tweed NSW (SA4 = 112)
_QLD_STE_2016 = '3'
_QLD_STE_2021 = '3'
_TWEED_SA4_2021 = '112'


def main():
    # ── Load 329-zone ORDER system ─────────────────────────────────────────
    order_df = pd.read_csv(ORDER_CSV)
    order_df['SA2_MAIN16'] = pd.to_numeric(order_df['SA2_MAIN16'], errors='coerce')
    order_df = order_df[order_df['SA2_MAIN16'] > 0]
    order_sa2s = set(order_df['SA2_MAIN16'].astype(int))
    print(f"ORDER zones with valid SA2_MAIN16: {len(order_sa2s)}")

    # ── Load 2021 SA2 codes already in 507-zone set ────────────────────────
    xwalk_507 = pd.read_csv(SA2_XWALK, usecols=['sa2_code', 'sa2_name'])
    sa2_codes_2021 = set(xwalk_507['sa2_code'].astype(int))
    print(f"SA2 2021 codes in 507-zone set: {len(sa2_codes_2021)}")

    # ── Which 2016 ORDER codes have no direct 2021 match? ─────────────────
    unmatched_2016 = order_sa2s - sa2_codes_2021
    print(f"2016 ORDER codes needing concordance: {len(unmatched_2016)}")
    if not unmatched_2016:
        print("All ORDER zones already have direct 2021 matches — no concordance needed.")
        pd.DataFrame(columns=['sa2_main16','sa2_name16','sa2_code21','sa2_name21',
                               'overlap_km2','overlap_frac']).to_csv(OUT_CSV, index=False)
        return

    # ── Load SA2 2016 shapefile ────────────────────────────────────────────
    print(f"\nLoading SA2 2016 shapefile from {SHP_2016} ...")
    gdf16 = gpd.read_file(SHP_2016)
    print(f"  Columns: {list(gdf16.columns)}")

    # Identify 2016 code column
    col16 = next((c for c in ['SA2_MAIN16', 'SA2_CODE16', 'SA2_CODE', 'SA2MAIN16']
                  if c in gdf16.columns), None)
    name16 = next((c for c in ['SA2_NAME16', 'SA2_NAME'] if c in gdf16.columns), None)
    ste16  = next((c for c in ['STE_CODE16', 'STE_CODE', 'STATE_CODE'] if c in gdf16.columns), None)

    if col16 is None:
        raise KeyError(f"Cannot find SA2 2016 code column. Available: {list(gdf16.columns)}")
    print(f"  Using 2016 code column: {col16}")

    gdf16[col16] = pd.to_numeric(gdf16[col16], errors='coerce')
    gdf16 = gdf16.dropna(subset=[col16])
    gdf16[col16] = gdf16[col16].astype(int)

    # Filter to SEQ study area (QLD + Tweed) where needed
    if ste16:
        gdf16_seq = gdf16[
            (gdf16[ste16].astype(str) == _QLD_STE_2016) |
            (gdf16[col16].astype(str).str.startswith('112'))
        ].copy()
    else:
        gdf16_seq = gdf16.copy()
    print(f"  2016 SA2 zones in study area: {len(gdf16_seq)}")

    # Only keep zones we need concordance for
    gdf16_unmatched = gdf16_seq[gdf16_seq[col16].isin(unmatched_2016)].copy()
    print(f"  2016 zones to resolve: {len(gdf16_unmatched)}")

    # ── Load SA2 2021 shapefile ────────────────────────────────────────────
    print(f"\nLoading SA2 2021 shapefile from {SHP_2021} ...")
    gdf21 = gpd.read_file(SHP_2021)

    col21  = next((c for c in ['SA2_CODE21', 'SA2_MAIN21', 'SA2_CODE', 'SA2CODE21']
                   if c in gdf21.columns), None)
    name21 = next((c for c in ['SA2_NAME21', 'SA2_NAME'] if c in gdf21.columns), None)
    ste21  = next((c for c in ['STE_CODE21', 'STE_CODE', 'STATE_CODE'] if c in gdf21.columns), None)
    sa4_21 = next((c for c in ['SA4_CODE21', 'SA4_CODE'] if c in gdf21.columns), None)

    if col21 is None:
        raise KeyError(f"Cannot find SA2 2021 code column. Available: {list(gdf21.columns)}")
    print(f"  Using 2021 code column: {col21}")

    gdf21[col21] = pd.to_numeric(gdf21[col21], errors='coerce')
    gdf21 = gdf21.dropna(subset=[col21])
    gdf21[col21] = gdf21[col21].astype(int)

    # Filter to SEQ study area (QLD + Tweed)
    if ste21 and sa4_21:
        gdf21_seq = gdf21[
            (gdf21[ste21].astype(str) == _QLD_STE_2021) |
            (gdf21[sa4_21].astype(str) == _TWEED_SA4_2021)
        ].copy()
    elif ste21:
        gdf21_seq = gdf21[gdf21[ste21].astype(str) == _QLD_STE_2021].copy()
    else:
        gdf21_seq = gdf21.copy()
    print(f"  2021 SA2 zones in study area: {len(gdf21_seq)}")

    # ── Reproject both to GDA2020 projected for area calculations ─────────
    # EPSG:7855 = GDA2020 / MGA Zone 55 (covers SEQ)
    _PROJ = 'EPSG:7855'
    print(f"\nReprojecting to {_PROJ} for area calculations ...")
    if gdf16_unmatched.crs is None:
        gdf16_unmatched = gdf16_unmatched.set_crs('EPSG:4283')   # GDA94 (2016 shp default)
    gdf16_p = gdf16_unmatched.to_crs(_PROJ)

    if gdf21_seq.crs is None:
        gdf21_seq = gdf21_seq.set_crs('EPSG:7844')               # GDA2020 (2021 shp default)
    gdf21_p = gdf21_seq.to_crs(_PROJ)

    # ── Spatial intersection: 2016 unmatched × 2021 SEQ ───────────────────
    print("Computing spatial intersection (may take ~30s) ...")
    gdf16_p['_area16_m2'] = gdf16_p.geometry.area

    joined = gpd.overlay(
        gdf16_p[[col16, name16 or col16, '_area16_m2', 'geometry']].rename(
            columns={col16: 'sa2_main16', name16 or col16: 'sa2_name16'}),
        gdf21_p[[col21, name21 or col21, 'geometry']].rename(
            columns={col21: 'sa2_code21', name21 or col21: 'sa2_name21'}),
        how='intersection',
        keep_geom_type=False,
    )
    joined['_overlap_m2'] = joined.geometry.area
    print(f"  Intersection rows: {len(joined)}")

    # ── Pick best 2021 match (largest overlap) per 2016 zone ──────────────
    best = (
        joined
        .sort_values('_overlap_m2', ascending=False)
        .drop_duplicates(subset=['sa2_main16'])
        [['sa2_main16', 'sa2_name16', 'sa2_code21', 'sa2_name21',
          '_overlap_m2', '_area16_m2']]
        .copy()
    )
    best['overlap_km2']  = best['_overlap_m2'] / 1e6
    best['overlap_frac'] = np.clip(best['_overlap_m2'] / best['_area16_m2'], 0, 1)
    best = best.drop(columns=['_overlap_m2', '_area16_m2'])
    best = best.sort_values('sa2_main16').reset_index(drop=True)

    # ── Report and save ────────────────────────────────────────────────────
    print(f"\n=== 2016→2021 SA2 Concordance ===")
    print(f"  Resolved {len(best)} of {len(unmatched_2016)} unmatched 2016 zones")

    unresolved = unmatched_2016 - set(best['sa2_main16'])
    if unresolved:
        print(f"  WARNING: {len(unresolved)} zones could not be resolved spatially "
              f"(likely outside shapefile extent): {sorted(unresolved)}")

    print(best[['sa2_main16', 'sa2_name16', 'sa2_code21', 'sa2_name21',
                'overlap_km2', 'overlap_frac']].to_string(index=False))

    best.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == '__main__':
    main()
