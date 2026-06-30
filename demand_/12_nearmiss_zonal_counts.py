"""
12_nearmiss_zonal_counts.py
Spatially join near-miss events to SA2 zones and produce time-period x zone x type counts.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import glob
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = r"Z:\test_runs_tours\demand_"
NEAR_MISS_DIR = r"Z:\test_runs_tours\crash_info_zonal"
SHP_PATH = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"

# ---------- Step a: Load SA2 shapefile ----------
print("Step a: Loading SA2 shapefile...")
sa2_gdf = gpd.read_file(SHP_PATH)
print(f"  Columns: {list(sa2_gdf.columns)}")
print(f"  CRS: {sa2_gdf.crs}")
print(f"  Total SA2s (national): {len(sa2_gdf)}")
print(f"  Sample rows:")
print(sa2_gdf[['SA2_CODE21','SA2_NAME21','STE_NAME21']].head(3).to_string())

# Reproject to WGS84 for lat/lng matching
sa2_gdf = sa2_gdf.to_crs('EPSG:4326')

# Filter to Queensland only
qld_mask = sa2_gdf['STE_NAME21'] == 'Queensland'
sa2_qld = sa2_gdf[qld_mask].copy()
print(f"  Queensland SA2s: {len(sa2_qld)}")
# Use full national set for sjoin (some SEQ zones may be on state boundary)
# but rename for clarity
sa2_for_join = sa2_gdf[['SA2_CODE21', 'SA2_NAME21', 'STE_NAME21', 'geometry']].copy()

# ---------- Step b: Load and stack all near-miss CSVs ----------
print("\nStep b: Loading near-miss CSV files...")
csv_files = [f for f in os.listdir(NEAR_MISS_DIR)
             if f.endswith('.csv') and f != 'SEQ_SA2_OD_Matrix_Length.csv']
print(f"  Found {len(csv_files)} near-miss files: {csv_files}")

all_dfs = []
file_counts = {}

for fname in sorted(csv_files):
    fpath = os.path.join(NEAR_MISS_DIR, fname)
    df_f = pd.read_csv(fpath)
    n = len(df_f)
    file_counts[fname] = n
    print(f"  {fname}: {n:,} rows, cols={list(df_f.columns)}")

    # Use roadMatchedPoint if available, fallback to rawPoint
    lat_col = 'roadMatchedPoint.lat'
    lng_col = 'roadMatchedPoint.lng'
    if lat_col not in df_f.columns or df_f[lat_col].isna().all():
        lat_col = 'rawPoint.lat'
        lng_col = 'rawPoint.lng'

    # Standardise lat/lng to common column names, coerce to numeric (handles 'undefined' strings)
    df_f = df_f.copy()
    matched_lat = pd.to_numeric(df_f[lat_col], errors='coerce')
    matched_lng = pd.to_numeric(df_f[lng_col], errors='coerce')
    raw_lat = pd.to_numeric(df_f['rawPoint.lat'], errors='coerce')
    raw_lng = pd.to_numeric(df_f['rawPoint.lng'], errors='coerce')
    # Use road-matched coords; fallback to raw if road-matched is NaN
    df_f['_lat'] = matched_lat.where(matched_lat.notna(), raw_lat)
    df_f['_lng'] = matched_lng.where(matched_lng.notna(), raw_lng)
    df_f['source_file'] = fname
    all_dfs.append(df_f)

df = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Total stacked rows: {len(df):,}")
print(f"  Sample timestamp values: {df['nearMiss.timestamp'].head(3).tolist()}")
print(f"  Sample unix ts values: {df['nearMiss.timestamp.seconds'].head(3).tolist()}")

# ---------- Step c: Parse timestamp -> hour ----------
print("\nStep c: Parsing timestamps...")
try:
    ts_sec = pd.to_numeric(df['nearMiss.timestamp.seconds'], errors='coerce')
    valid_ts = ts_sec.notna() & (ts_sec > 0)
    hour = pd.Series(np.nan, index=df.index)
    if valid_ts.sum() > 0:
        dt_utc = pd.to_datetime(ts_sec[valid_ts], unit='s', utc=True)
        dt_bne = dt_utc.dt.tz_convert('Australia/Brisbane')
        hour[valid_ts] = dt_bne.dt.hour.values

    # Fallback: parse string timestamp for remaining rows
    missing_hour = hour.isna()
    if missing_hour.sum() > 0:
        print(f"  Parsing string timestamp for {missing_hour.sum()} rows...")
        dt_str = pd.to_datetime(df.loc[missing_hour, 'nearMiss.timestamp'], errors='coerce')
        hour[missing_hour] = dt_str.dt.hour

    df['hour'] = hour.astype('Int64')
    print(f"  Hours parsed: {df['hour'].notna().sum():,} / {len(df):,}")
    print(f"  Hour distribution (sample): {df['hour'].value_counts().sort_index().head(10).to_dict()}")
except Exception as e:
    print(f"  Warning: timestamp parse error: {e}")
    df['hour'] = np.nan

# ---------- Step d: Map hour -> time period ----------
print("\nStep d: Assigning time periods...")

def assign_period(hour):
    if pd.isna(hour):
        return 'ev'
    h = int(hour)
    if 7 <= h <= 8:
        return 'am'
    elif 9 <= h <= 15:
        return 'op'
    elif 16 <= h <= 17:
        return 'pm'
    else:
        return 'ev'

df['time_period'] = df['hour'].apply(assign_period)
print(f"  Period counts: {df['time_period'].value_counts().to_dict()}")

# ---------- Step e: Spatial join events to SA2 zones ----------
print("\nStep e: Spatial join to SA2 zones...")
# Drop rows with no valid coordinates
coord_mask = df['_lat'].notna() & df['_lng'].notna()
print(f"  Rows with valid coords: {coord_mask.sum():,} / {len(df):,}")
df_geo = df[coord_mask].copy()

# Create GeoDataFrame
gdf_events = gpd.GeoDataFrame(
    df_geo,
    geometry=gpd.points_from_xy(df_geo['_lng'], df_geo['_lat']),
    crs='EPSG:4326'
)
print(f"  Created GeoDataFrame with {len(gdf_events):,} points")
print(f"  Lat range: {df_geo['_lat'].min():.4f}..{df_geo['_lat'].max():.4f}")
print(f"  Lng range: {df_geo['_lng'].min():.4f}..{df_geo['_lng'].max():.4f}")

# Spatial join
print("  Running sjoin (this may take a moment)...")
joined = gpd.sjoin(gdf_events, sa2_for_join, how='left', predicate='within')
print(f"  Joined rows: {len(joined):,}")

# Handle multiple matches (take first)
joined = joined.loc[~joined.index.duplicated(keep='first')]

# Copy SA2 columns back to original df index
df.loc[df_geo.index, 'SA2_CODE'] = joined['SA2_CODE21'].values
df.loc[df_geo.index, 'SA2_NAME'] = joined['SA2_NAME21'].values
df.loc[df_geo.index, 'STE_NAME'] = joined['STE_NAME21'].values

# Unmatched events
df['SA2_CODE'] = df['SA2_CODE'].fillna('UNMATCHED')
df['SA2_NAME'] = df['SA2_NAME'].fillna('UNMATCHED')

matched = (df['SA2_CODE'] != 'UNMATCHED').sum()
pct_matched = 100.0 * matched / len(df)
print(f"  Matched to SA2: {matched:,} / {len(df):,} ({pct_matched:.1f}%)")
print(f"  Unmatched: {(df['SA2_CODE'] == 'UNMATCHED').sum():,}")

# ---------- Step f: Compute counts table ----------
print("\nStep f: Computing counts table...")

# Standardise column names for safety
type_col = 'nearMiss.type'
veh_col = 'vehicleType'
speed_col = 'speed (km/h)'
accel_col = 'acceleration.total (G)'
weight_col = 'totalCount'

# Force numeric types on measurement columns (some files may have string values)
df[speed_col] = pd.to_numeric(df[speed_col], errors='coerce')
df[accel_col] = pd.to_numeric(df[accel_col], errors='coerce')
df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce').fillna(1)

group_cols = ['SA2_CODE', 'SA2_NAME', 'time_period', type_col, veh_col]
agg_df = df.groupby(group_cols, dropna=False).agg(
    event_count=('source_file', 'count'),
    weighted_count=(weight_col, 'sum'),
    avg_speed=(speed_col, 'mean'),
    avg_accel=(accel_col, 'mean'),
).reset_index()

counts_path = os.path.join(OUT_DIR, 'nearmiss_zonal_counts.csv')
agg_df.to_csv(counts_path, index=False)
print(f"  Written: {counts_path} ({len(agg_df):,} rows)")

# ---------- Step g: Pivot to summary table ----------
print("\nStep g: Building summary pivot table...")

# Helper: map nearMiss.type to short name
type_map = {
    'NEAR_MISS_TYPE_STEERING': 'steering',
    'NEAR_MISS_TYPE_BRAKING': 'braking',
    'NEAR_MISS_TYPE_COMBINED': 'combined',
}
df['nm_type_short'] = df[type_col].map(type_map).fillna('other')
df['veh_short'] = df[veh_col].str.lower().fillna('unknown')

# For each SA2, compute summary columns
sa2_codes = df[df['SA2_CODE'] != 'UNMATCHED']['SA2_CODE'].unique()
summary_rows = []

periods = ['am', 'op', 'pm', 'ev']
nm_types = ['steering', 'braking', 'combined']
veh_types = ['car', 'hcv']

for sa2 in sa2_codes:
    sub = df[df['SA2_CODE'] == sa2]
    sa2_name = sub['SA2_NAME'].iloc[0] if len(sub) > 0 else ''

    row = {'sa2_code': sa2, 'sa2_name': sa2_name}

    # Period totals
    for p in periods:
        psub = sub[sub['time_period'] == p]
        row[f'{p}_total'] = len(psub)

        # By type
        for t in nm_types:
            row[f'{p}_{t}'] = len(psub[psub['nm_type_short'] == t])

        # By vehicle
        for v in veh_types:
            row[f'{p}_{v}'] = len(psub[psub['veh_short'] == v])

        # Avg speed / accel
        row[f'avg_speed_{p}'] = round(psub[speed_col].mean(), 2) if len(psub) > 0 else np.nan
        row[f'avg_accel_{p}'] = round(psub[accel_col].mean(), 4) if len(psub) > 0 else np.nan

    row['total_events'] = len(sub)
    row['total_car'] = len(sub[sub['veh_short'] == 'car'])
    row['total_hcv'] = len(sub[sub['veh_short'] == 'hcv'])

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
# Sort by total_events desc
summary_df = summary_df.sort_values('total_events', ascending=False).reset_index(drop=True)
summary_path = os.path.join(OUT_DIR, 'nearmiss_zonal_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"  Written: {summary_path} ({len(summary_df):,} SA2 zones with events)")

# ---------- Step h: Temporal summary ----------
print("\nStep h: Building temporal summary...")
total_events = len(df)

temporal_rows = []
for p in periods:
    psub = df[df['time_period'] == p]
    n = len(psub)
    temporal_rows.append({
        'time_period': p,
        'total_events': n,
        'steering_events': len(psub[psub['nm_type_short'] == 'steering']),
        'braking_events': len(psub[psub['nm_type_short'] == 'braking']),
        'combined_events': len(psub[psub['nm_type_short'] == 'combined']),
        'car_events': len(psub[psub['veh_short'] == 'car']),
        'hcv_events': len(psub[psub['veh_short'] == 'hcv']),
        'avg_speed': round(psub[speed_col].mean(), 2) if n > 0 else np.nan,
        'avg_accel': round(psub[accel_col].mean(), 4) if n > 0 else np.nan,
        'pct_of_daily_total': round(100.0 * n / total_events, 2) if total_events > 0 else 0.0,
    })

temporal_df = pd.DataFrame(temporal_rows)
temporal_path = os.path.join(OUT_DIR, 'nearmiss_temporal_summary.csv')
temporal_df.to_csv(temporal_path, index=False)
print(f"  Written: {temporal_path}")
print(temporal_df.to_string(index=False))

# Hourly profile
print("\nBuilding hourly profile...")
hourly_rows = []
for h in range(24):
    hsub = df[df['hour'] == h]
    n = len(hsub)
    hourly_rows.append({
        'hour': h,
        'time_period': assign_period(h),
        'total_events': n,
        'steering': len(hsub[hsub['nm_type_short'] == 'steering']),
        'braking': len(hsub[hsub['nm_type_short'] == 'braking']),
        'combined': len(hsub[hsub['nm_type_short'] == 'combined']),
    })

hourly_df = pd.DataFrame(hourly_rows)
hourly_path = os.path.join(OUT_DIR, 'nearmiss_hourly_profile.csv')
hourly_df.to_csv(hourly_path, index=False)
print(f"  Written: {hourly_path}")

# ---------- Step i: Print report ----------
print()
print("=== SCRIPT 12 SUMMARY ===")
print(f"  Total events loaded: {total_events:,}")
print()
print("  Events per file:")
for fname, cnt in sorted(file_counts.items()):
    print(f"    {fname}: {cnt:,}")
print()
print("  Events by time period:")
for _, r in temporal_df.iterrows():
    print(f"    {r['time_period']}: {int(r['total_events']):,} ({r['pct_of_daily_total']:.1f}%)")
print()
print("  Events by type:")
for t in [type_map[k] for k in type_map]:
    n = len(df[df['nm_type_short'] == t])
    print(f"    {t}: {n:,}")
print()
print("  Events by vehicle type:")
for v in ['car', 'hcv']:
    n = len(df[df['veh_short'] == v])
    print(f"    {v.upper()}: {n:,}")
print()
print("  Top 10 SA2 zones by near-miss count:")
top10 = summary_df.head(10)[['sa2_code', 'sa2_name', 'total_events', 'am_total', 'op_total', 'pm_total', 'ev_total']]
print(top10.to_string(index=False))
print()
print(f"  % events matched to SA2: {pct_matched:.1f}%")

# Date range
try:
    ts_valid = pd.to_numeric(df['nearMiss.timestamp.seconds'], errors='coerce').dropna()
    dt_min = pd.to_datetime(ts_valid.min(), unit='s', utc=True).tz_convert('Australia/Brisbane')
    dt_max = pd.to_datetime(ts_valid.max(), unit='s', utc=True).tz_convert('Australia/Brisbane')
    print(f"  Date range: {dt_min.date()} to {dt_max.date()}")
except Exception as e:
    print(f"  Date range: (error: {e})")

print("Script 12 complete.")
