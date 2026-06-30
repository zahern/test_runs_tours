"""
07_zone_polygons_crash.py
Update GeoPackage with crash risk and 507-zone coverage.
"""
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd

OUT_DIR = r"Z:\test_runs_tours\demand_"

# Load zone IDs
with open(r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt") as f:
    content = f.read()
zone_ids_507 = sorted([int(x) for x in re.split(r'[\s,]+', content.strip()) if x])
zone_ids_set = set(zone_ids_507)
print(f"Loaded {len(zone_ids_507)} SA2 zones")

# --- a) Load SA2 shapefile ---
shp_path = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
gdf_all = gpd.read_file(shp_path)
print(f"Total shapefile rows: {len(gdf_all)}")

# Filter to Queensland
gdf_qld = gdf_all[gdf_all['STE_CODE21'] == '3'].copy()
print(f"Queensland SA2 count: {len(gdf_qld)}")

# Filter to 507 zones
gdf_qld['SA2_CODE21_int'] = pd.to_numeric(gdf_qld['SA2_CODE21'], errors='coerce').astype('Int64')
gdf_507 = gdf_qld[gdf_qld['SA2_CODE21_int'].isin(zone_ids_507)].copy()
print(f"507-zone subset: {len(gdf_507)} polygons")

# Reproject to WGS84
gdf_507 = gdf_507.to_crs(epsg=4326)
gdf_507['sa2_lat'] = gdf_507.geometry.centroid.y
gdf_507['sa2_lon'] = gdf_507.geometry.centroid.x

# Use original CRS for area calculation
gdf_area = gdf_qld[gdf_qld['SA2_CODE21_int'].isin(zone_ids_507)].copy()
gdf_area = gdf_area.to_crs(epsg=32756)  # GDA2020 / MGA zone 56 (metres)
gdf_507['area_sqkm'] = gdf_area.geometry.area / 1e6

gdf_507 = gdf_507.rename(columns={
    'SA2_CODE21': 'sa2_code_str',
    'SA2_NAME21': 'sa2_name',
    'AREASQKM21': 'area_sqkm_official'
})
gdf_507['sa2_code'] = gdf_507['SA2_CODE21_int'].astype(int)

# --- b) Load crosswalk ---
crosswalk = pd.read_csv(os.path.join(OUT_DIR, 'sa2_crosswalk_507.csv'))
crosswalk['sa2_code'] = crosswalk['sa2_code'].astype(int)
gdf_507 = gdf_507.merge(
    crosswalk[['sa2_code', 'n_emme_zones', 'match_type']],
    on='sa2_code', how='left'
)

# --- c) Load crash data and aggregate to zone level ---
crash_od = pd.read_csv(os.path.join(OUT_DIR, 'crash_od_507.csv'))
print(f"Crash OD: {len(crash_od):,} rows")

# Inbound: sum all OD pairs arriving at each dest SA2
crash_inbound = (
    crash_od
    .groupby('dest_sa2')
    .agg(
        crash_inbound_total=('crash_total', 'sum'),
        crash_inbound_runoff=('crash_runoff', 'sum'),
        crash_inbound_headon=('crash_headon', 'sum'),
        crash_inbound_rearend=('crash_rearend', 'sum'),
        crash_inbound_intersection=('crash_intersection', 'sum'),
    )
    .reset_index()
    .rename(columns={'dest_sa2': 'sa2_code'})
)

# Outbound: sum all OD pairs originating from each origin SA2
crash_outbound = (
    crash_od
    .groupby('origin_sa2')
    .agg(
        crash_outbound_total=('crash_total', 'sum'),
    )
    .reset_index()
    .rename(columns={'origin_sa2': 'sa2_code'})
)

gdf_507 = gdf_507.merge(crash_inbound, on='sa2_code', how='left')
gdf_507 = gdf_507.merge(crash_outbound, on='sa2_code', how='left')

# Crash density
gdf_507['crash_density'] = gdf_507['crash_inbound_total'] / gdf_507['area_sqkm'].replace(0, np.nan)

# --- d) Load synth pop OD origin totals ---
synth_od = pd.read_csv(os.path.join(OUT_DIR, 'synth_pop_od_507.csv'))
synth_origin = synth_od.groupby(['origin_sa2', 'mode'])['demand'].sum().unstack(fill_value=0).reset_index()
synth_origin.columns = ['sa2_code'] + [f'synth_demand_{m}' for m in synth_origin.columns[1:]]
synth_origin['sa2_code'] = synth_origin['sa2_code'].astype(int)
gdf_507 = gdf_507.merge(synth_origin, on='sa2_code', how='left')

# --- e) Load HHTS OD origin totals ---
hhts_od = pd.read_csv(os.path.join(OUT_DIR, 'hhts_od_by_mode_507.csv'))
hhts_origin = hhts_od.groupby(['origin_sa2', 'mode'])['weighted_trips'].sum().unstack(fill_value=0).reset_index()
hhts_origin.columns = ['sa2_code'] + [f'hhts_demand_{m}' for m in hhts_origin.columns[1:]]
hhts_origin['sa2_code'] = hhts_origin['sa2_code'].astype(int)
gdf_507 = gdf_507.merge(hhts_origin, on='sa2_code', how='left')

# Fill NaN numerics with 0
numeric_cols = [c for c in gdf_507.columns if c not in ['geometry', 'sa2_code_str', 'sa2_name', 'match_type', 'CHG_FLAG21', 'CHG_LBL21']]
for col in numeric_cols:
    if gdf_507[col].dtype in [float, 'float64', 'float32']:
        gdf_507[col] = gdf_507[col].fillna(0)

# Drop duplicate/unused columns
drop_cols = [c for c in gdf_507.columns if c in ['SA2_CODE21_int', 'SA3_CODE21', 'SA3_NAME21', 'SA4_CODE21',
             'SA4_NAME21', 'GCC_CODE21', 'GCC_NAME21', 'STE_CODE21', 'STE_NAME21',
             'AUS_CODE21', 'AUS_NAME21', 'LOCI_URI21', 'CHG_FLAG21', 'CHG_LBL21']]
gdf_507 = gdf_507.drop(columns=drop_cols, errors='ignore')

# --- g) Save GeoPackage ---
out_gpkg = os.path.join(OUT_DIR, 'zone_demand_crash_507.gpkg')
if os.path.exists(out_gpkg):
    os.remove(out_gpkg)
gdf_507.to_file(out_gpkg, layer='sa2_demand_crash', driver='GPKG')
print(f"Saved {out_gpkg} ({len(gdf_507)} features)")

# --- h) Save zone_summary_507.csv ---
summary_cols = [c for c in gdf_507.columns if c != 'geometry']
summary_df = gdf_507[summary_cols].copy()
out_csv = os.path.join(OUT_DIR, 'zone_summary_507.csv')
summary_df.to_csv(out_csv, index=False)
print(f"Saved {out_csv} ({len(summary_df)} rows)")

print("\n=== ZONE POLYGONS SUMMARY ===")
print(f"Total features in GeoPackage: {len(gdf_507)}")
print(f"SA2 zones with EMME coverage: {(gdf_507['n_emme_zones'] > 0).sum()}")
print(f"Total crash inbound (sum): {gdf_507['crash_inbound_total'].sum():.1f}")
print(f"Top 5 zones by crash inbound:")
print(gdf_507.nlargest(5, 'crash_inbound_total')[['sa2_code', 'sa2_name', 'crash_inbound_total', 'crash_density']].to_string(index=False))
