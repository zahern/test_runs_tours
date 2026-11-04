"""
Script 05: Create zone demand GeoPackage with SA2 polygon geometry and demand attributes.
"""
import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd

# ── Paths ──────────────────────────────────────────────────────────────────────
SA2_SHP       = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
OUTPUT_DIR    = r"Z:\test_runs_tours\demand_"
SA2_REF_CSV   = os.path.join(OUTPUT_DIR, 'sa2_zones_reference.csv')
MAPPING_CSV   = os.path.join(OUTPUT_DIR, 'sa2_to_emme_mapping.csv')
SYNTH_OD_CSV  = os.path.join(OUTPUT_DIR, 'synth_pop_od_sa2.csv')
HHTS_OD_CSV   = os.path.join(OUTPUT_DIR, 'hhts_od_by_sa2_mode.csv')

POST_DEMAND_CSV = r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM\#Model_Masters\scripts\post_demand\post_demand.csv"

OUT_GPKG = os.path.join(OUTPUT_DIR, 'zone_demand_polygons.gpkg')
OUT_CSV  = os.path.join(OUTPUT_DIR, 'zone_demand_sa2.csv')

print("=" * 60)
print("Script 05: Build Zone Demand GeoPackage")
print("=" * 60)

# ── 1. Load SA2 shapefile ─────────────────────────────────────────────────────
print("\n[1] Loading SA2 shapefile ...")
sa2_shp = gpd.read_file(SA2_SHP)
print(f"    Total SA2 polygons: {len(sa2_shp)}")

# Reproject to WGS84 for consistency
if sa2_shp.crs and sa2_shp.crs.to_epsg() != 4326:
    sa2_shp = sa2_shp.to_crs(epsg=4326)
    print("    Reprojected to EPSG:4326 (WGS84)")

# Keep only QLD (and possibly border zones that appear in our data)
sa2_qld = sa2_shp[sa2_shp['STE_NAME21'] == 'Queensland'].copy()
print(f"    QLD SA2 polygons: {len(sa2_qld)}")
sa2_qld = sa2_qld.rename(columns={'SA2_CODE21': 'sa2_code', 'SA2_NAME21': 'sa2_name_abs'})

# ── 2. Load SA2 zones reference ───────────────────────────────────────────────
print("\n[2] Loading SA2 zones reference ...")
sa2_ref = pd.read_csv(SA2_REF_CSV)
# Normalise sa2_code to integer string (handles float like 209041530.0)
sa2_ref['sa2_code'] = sa2_ref['sa2_code'].apply(
    lambda x: str(int(float(x))) if pd.notna(x) else None
)
print(f"    SA2 reference rows: {len(sa2_ref)}")

# ── 3. Aggregate synth pop demand ─────────────────────────────────────────────
print("\n[3] Aggregating synth pop demand by origin SA2 ...")
synth_od = pd.read_csv(SYNTH_OD_CSV)
synth_od['origin_sa2'] = synth_od['origin_sa2'].astype(str)

synth_origin = synth_od.pivot_table(
    index='origin_sa2', columns='mode', values='demand', aggfunc='sum', fill_value=0
).reset_index()
synth_origin.columns.name = None
synth_origin.rename(columns={'origin_sa2': 'sa2_code'}, inplace=True)

# Rename columns
for mode in ['car', 'pt', 'walk', 'bike', 'other']:
    if mode in synth_origin.columns:
        synth_origin.rename(columns={mode: f'synth_demand_{mode}'}, inplace=True)
    else:
        synth_origin[f'synth_demand_{mode}'] = 0.0

synth_origin['synth_demand_total'] = (
    synth_origin.get('synth_demand_car', 0) +
    synth_origin.get('synth_demand_pt', 0) +
    synth_origin.get('synth_demand_walk', 0) +
    synth_origin.get('synth_demand_bike', 0) +
    synth_origin.get('synth_demand_other', 0)
)
print(f"    Synth origin SA2s: {len(synth_origin)}")
print(f"    Synth total demand: {synth_origin['synth_demand_total'].sum():,.0f}")

# ── 4. Aggregate HHTS demand ──────────────────────────────────────────────────
print("\n[4] Aggregating HHTS demand by origin SA2 ...")
hhts_od = pd.read_csv(HHTS_OD_CSV)
hhts_od['origin_sa2'] = hhts_od['origin_sa2'].astype(str)

hhts_origin = hhts_od.pivot_table(
    index='origin_sa2', columns='mode', values='total_weighted_trips', aggfunc='sum', fill_value=0
).reset_index()
hhts_origin.columns.name = None
hhts_origin.rename(columns={'origin_sa2': 'sa2_code'}, inplace=True)

for mode in ['car', 'pt', 'walk', 'bike', 'other']:
    if mode in hhts_origin.columns:
        hhts_origin.rename(columns={mode: f'hhts_demand_{mode}'}, inplace=True)
    else:
        hhts_origin[f'hhts_demand_{mode}'] = 0.0

hhts_origin['hhts_demand_total'] = (
    hhts_origin.get('hhts_demand_car', 0) +
    hhts_origin.get('hhts_demand_pt', 0) +
    hhts_origin.get('hhts_demand_walk', 0) +
    hhts_origin.get('hhts_demand_bike', 0) +
    hhts_origin.get('hhts_demand_other', 0)
)
print(f"    HHTS origin SA2s: {len(hhts_origin)}")
print(f"    HHTS total demand: {hhts_origin['hhts_demand_total'].sum():,.0f}")

# ── 5. Aggregate SEQSTM demand ────────────────────────────────────────────────
print("\n[5] Aggregating SEQSTM post-demand (chunked) ...")
mapping = pd.read_csv(MAPPING_CSV)
mapping['emme_zone_id'] = mapping['emme_zone_id'].astype(int)
# sa2_code is stored as float (e.g. 209041530.0); convert to clean integer string
mapping['sa2_code'] = mapping['sa2_code'].dropna().apply(
    lambda x: str(int(float(x))) if pd.notna(x) else None
)
emme_to_sa2 = mapping.dropna(subset=['sa2_code']).set_index('emme_zone_id')['sa2_code'].to_dict()

# Read post_demand.csv in chunks; aggregate by origin zone, time period
seqstm_agg = {}  # sa2_code -> {period -> demand_total}

chunk_count = 0
CHUNKSIZE = 500_000
total_rows_read = 0

try:
    for chunk in pd.read_csv(POST_DEMAND_CSV, chunksize=CHUNKSIZE, low_memory=False):
        chunk_count += 1
        total_rows_read += len(chunk)
        if chunk_count % 10 == 0:
            print(f"    Chunk {chunk_count}: {total_rows_read:,} rows read ...")

        # Map EMME origin zone to SA2
        chunk['sa2_code'] = chunk['origin_zone'].map(emme_to_sa2)
        valid = chunk[chunk['sa2_code'].notna()].copy()

        # Group by sa2_code and time_period
        if 'time_period' in valid.columns and 'demand' in valid.columns:
            grp = valid.groupby(['sa2_code', 'time_period'])['demand'].sum()
            for (sa2, period), demand in grp.items():
                if sa2 not in seqstm_agg:
                    seqstm_agg[sa2] = {}
                seqstm_agg[sa2][period] = seqstm_agg[sa2].get(period, 0) + demand
        elif 'period_label' in valid.columns and 'demand' in valid.columns:
            grp = valid.groupby(['sa2_code', 'period_label'])['demand'].sum()
            for (sa2, period), demand in grp.items():
                if sa2 not in seqstm_agg:
                    seqstm_agg[sa2] = {}
                seqstm_agg[sa2][period] = seqstm_agg[sa2].get(period, 0) + demand

    print(f"    Total rows read: {total_rows_read:,}")
    print(f"    SA2 zones with SEQSTM demand: {len(seqstm_agg)}")
except Exception as e:
    print(f"    WARNING: Error reading post_demand.csv: {e}")
    seqstm_agg = {}

# Convert to dataframe
if seqstm_agg:
    seqstm_rows = []
    for sa2, periods in seqstm_agg.items():
        row = {'sa2_code': sa2}
        row.update({f'seqstm_demand_{k.lower()}': v for k, v in periods.items()})
        seqstm_rows.append(row)
    seqstm_df = pd.DataFrame(seqstm_rows)

    # Ensure columns exist
    for period in ['am', 'pm', 'md', 'ea', 'ev', 'op']:
        col = f'seqstm_demand_{period}'
        if col not in seqstm_df.columns:
            seqstm_df[col] = 0.0

    # Total
    seqstm_df['seqstm_demand_total'] = seqstm_df.filter(like='seqstm_demand').sum(axis=1)
    print(f"    SEQSTM demand rows: {len(seqstm_df)}")
else:
    seqstm_df = pd.DataFrame(columns=['sa2_code', 'seqstm_demand_total'])
    print("    WARNING: No SEQSTM demand aggregated")

# ── 6. Merge all attributes onto shapefile ────────────────────────────────────
print("\n[6] Merging all attributes onto SA2 shapefile ...")

gdf = sa2_qld.copy()
gdf['sa2_code'] = gdf['sa2_code'].astype(str)

# Add zones reference info
gdf = gdf.merge(
    sa2_ref[['sa2_code', 'n_emme_zones', 'area_sqkm', 'centroid_lon', 'centroid_lat']],
    on='sa2_code', how='left'
)

# Add synth demand
gdf = gdf.merge(synth_origin, on='sa2_code', how='left')

# Add HHTS demand
gdf = gdf.merge(hhts_origin, on='sa2_code', how='left')

# Add SEQSTM demand
if len(seqstm_df) > 0:
    gdf = gdf.merge(seqstm_df, on='sa2_code', how='left')

# Fill NaN demands with 0
demand_cols = [c for c in gdf.columns if 'demand' in c.lower()]
for col in demand_cols:
    gdf[col] = gdf[col].fillna(0.0)

print(f"    Final GeoDataFrame rows: {len(gdf)}")
print(f"    Columns: {[c for c in gdf.columns if c != 'geometry']}")

# ── 7. Save GeoPackage ────────────────────────────────────────────────────────
print(f"\n[7] Saving GeoPackage: {OUT_GPKG} ...")
gdf.to_file(OUT_GPKG, layer='sa2_demand', driver='GPKG')
fsize = os.path.getsize(OUT_GPKG) / (1024 ** 2)
print(f"    Saved: {fsize:.1f} MB, {len(gdf)} features")

# ── 8. Save CSV (without geometry) ────────────────────────────────────────────
print(f"\n[8] Saving CSV: {OUT_CSV} ...")
gdf_csv = gdf.drop(columns=['geometry'])
gdf_csv.to_csv(OUT_CSV, index=False)
print(f"    Saved: {len(gdf_csv)} rows")

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCRIPT 05 COMPLETE")
print(f"  GeoPackage: {OUT_GPKG} ({fsize:.1f} MB)")
print(f"  CSV: {OUT_CSV}")
print(f"  Features: {len(gdf)}")
print(f"  Zones with synth demand: {(gdf['synth_demand_total'] > 0).sum() if 'synth_demand_total' in gdf.columns else 'n/a'}")
print(f"  Zones with HHTS demand:  {(gdf['hhts_demand_total'] > 0).sum() if 'hhts_demand_total' in gdf.columns else 'n/a'}")
if 'seqstm_demand_total' in gdf.columns:
    print(f"  Zones with SEQSTM demand: {(gdf['seqstm_demand_total'] > 0).sum()}")
print("=" * 60)
