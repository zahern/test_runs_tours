"""
14_nearmiss_ecf_road.py
Road-segment-level near-miss + expected crash frequency dataset.
Joins telematics near-miss events to the SEQ road network GeoPackage.

Outputs:
  Z:\test_runs_tours\demand_\SEQ_crashes_nearmiss_embedded.gpkg
  Z:\test_runs_tours\demand_\near_miss_ecf.csv
"""

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"Z:\test_runs_tours\crash_info_zonal")
OUTPUT_DIR = Path(r"Z:\test_runs_tours\demand_")
GPKG_PATH  = DATA_DIR / "SEQ_Expected_Crashes_1000.gpkg"
OUT_GPKG   = OUTPUT_DIR / "SEQ_crashes_nearmiss_embedded.gpkg"
OUT_CSV    = OUTPUT_DIR / "near_miss_ecf.csv"

CSV_FILES = {
    "aug2025":       "aug2025-08-01_2025-08-31.csv",
    "sep_nov2025":   "sep-nov2025-09-01_2025-11-30.csv",
    "dec_feb2026":   "dec-feb2025-12-01_2026-02-28.csv",
    "mar_apr2026":   "marchapril2026-03-01_2026-04-30.csv",
    "may2026":       "may2026-05-01_2026-05-31.csv",
    "jun2026":       "june_near_miss2026-06-01_2026-06-29.csv",
}

PERIODS = {
    "am": (7, 8),
    "op": (9, 15),
    "pm": (16, 17),
    "ev": None,          # everything else
}

print("=" * 70)
print("14_nearmiss_ecf_road.py  —  Near-miss × Road Segment Analysis")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STEP A — Load and stack all near-miss CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP A  Load & stack near-miss CSVs ──────────────────────────────")

frames = []
for month_label, fname in CSV_FILES.items():
    fpath = DATA_DIR / fname
    df = pd.read_csv(fpath, dtype=str, low_memory=False)
    df["month_period"] = month_label
    df["source_file"]  = fname
    frames.append(df)
    print(f"  {fname:55s}  {len(df):>7,} rows")

events = pd.concat(frames, ignore_index=True)
print(f"\n  Combined total: {len(events):,} events")

# ── b) osmWayId sample ────────────────────────────────────────────────────────
print("\n  osmWayId sample (first file):")
print(" ", events[events["source_file"] == CSV_FILES["aug2025"]]["osmWayId"].dropna().head(10).tolist())

# ── c) Coordinates ────────────────────────────────────────────────────────────
for col in ["roadMatchedPoint.lat", "roadMatchedPoint.lng",
            "rawPoint.lat", "rawPoint.lng"]:
    events[col] = pd.to_numeric(events[col].replace("undefined", np.nan), errors="coerce")

events["lat"] = events["roadMatchedPoint.lat"].where(
    events["roadMatchedPoint.lat"].notna(), events["rawPoint.lat"]
)
events["lon"] = events["roadMatchedPoint.lng"].where(
    events["roadMatchedPoint.lng"].notna(), events["rawPoint.lng"]
)

# ── d) Parse hour ─────────────────────────────────────────────────────────────
events["ts_sec"] = pd.to_numeric(events["nearMiss.timestamp.seconds"], errors="coerce")
has_ts = events["ts_sec"].notna()

events.loc[has_ts, "hour"] = (
    pd.to_datetime(events.loc[has_ts, "ts_sec"], unit="s", utc=True)
    .dt.tz_convert("Australia/Brisbane")
    .dt.hour
)

# Fallback: parse timestamp string
need_fallback = ~has_ts
if need_fallback.any():
    ts_str = pd.to_datetime(events.loc[need_fallback, "nearMiss.timestamp"],
                             errors="coerce", dayfirst=False)
    events.loc[need_fallback, "hour"] = ts_str.dt.hour

events["hour"] = pd.to_numeric(events["hour"], errors="coerce").astype("Int64")

# ── e) Time period ─────────────────────────────────────────────────────────────
def assign_period(h):
    if pd.isna(h):
        return "ev"
    h = int(h)
    if 7 <= h <= 8:
        return "am"
    if 9 <= h <= 15:
        return "op"
    if 16 <= h <= 17:
        return "pm"
    return "ev"

events["time_period"] = events["hour"].apply(assign_period)

# ── f/g) Print summaries ───────────────────────────────────────────────────────
print(f"\n  Per-file counts:")
print(events.groupby("source_file")["source_file"].count().to_string())
print(f"\n  Per time-period counts:")
print(events["time_period"].value_counts().to_string())
print(f"\n  Per near-miss type counts:")
print(events["nearMiss.type"].value_counts().to_string())
print(f"\n  Vehicle type counts:")
print(events["vehicleType"].value_counts().to_string())

# ── Numeric conversion of key fields ─────────────────────────────────────────
for col in ["speed (km/h)", "acceleration.total (G)",
            "acceleration.x (G)", "acceleration.y (G)", "acceleration.z (G)", "iri", "totalCount"]:
    events[col] = pd.to_numeric(events[col], errors="coerce")

# ══════════════════════════════════════════════════════════════════════════════
# STEP B — ID matching + spatial fallback
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP B  ID matching ──────────────────────────────────────────────")

gdf_proj = gpd.read_file(str(GPKG_PATH), layer="roads")   # EPSG:28355
gdf_wgs  = gdf_proj.to_crs(4326)

print(f"  GeoPackage loaded: {len(gdf_proj):,} road segments  CRS={gdf_proj.crs}")

# ── i) Direct match: osmWayId → segment_id ────────────────────────────────────
seg_ids = set(gdf_proj["segment_id"].astype(str))
events["osmWayId_str"] = events["osmWayId"].astype(str).str.strip()

direct_mask = events["osmWayId_str"].isin(seg_ids)
print(f"\n  Direct osmWayId → segment_id match: {direct_mask.sum():,} / {len(events):,} "
      f"({direct_mask.mean()*100:.1f}%)")

# ── j) Alternative matching strategies ───────────────────────────────────────
# Check if osmWayId is embedded in segment_id  (segment_id contains osmWayId)
# segment_id example: "16015103990000032888"  osmWayId: "172734548"
# Build lookup: {osmWayId_as_str: segment_id} by checking substring
print("\n  Investigating segment_id format vs osmWayId format …")
sample_seg = gdf_proj["segment_id"].head(20).tolist()
sample_way = events["osmWayId_str"].dropna().unique()[:20].tolist()
print(f"  segment_id examples: {sample_seg[:5]}")
print(f"  osmWayId   examples: {sample_way[:5]}")

# Strategy 1: strip leading zeros
seg_stripped = {s.lstrip("0"): s for s in seg_ids}
way_stripped  = events["osmWayId_str"].str.lstrip("0")
strip_mask    = way_stripped.isin(seg_stripped)
print(f"\n  Strip-leading-zeros match: {strip_mask.sum():,} ({strip_mask.mean()*100:.1f}%)")

# Strategy 2: osmWayId embedded in segment_id
# Build mapping: osmWayId → segment_id where segment_id ends with osmWayId
def build_suffix_map(seg_id_series, way_ids):
    """Map osmWayId → segment_id where segment_id ends with osmWayId."""
    seg_list = list(seg_id_series.astype(str))
    mapping = {}
    for wid in way_ids:
        wid_s = str(wid).strip()
        if not wid_s or wid_s == "nan":
            continue
        for sid in seg_list:
            if sid.endswith(wid_s):
                mapping[wid_s] = sid
                break
    return mapping

print("  Building suffix-match map (osmWayId tail in segment_id) …")
t0 = time.time()
unique_ways = events["osmWayId_str"].dropna().unique()
suffix_map  = build_suffix_map(gdf_proj["segment_id"], unique_ways)
suffix_mask = events["osmWayId_str"].isin(suffix_map)
print(f"  Suffix-match: {suffix_mask.sum():,} ({suffix_mask.mean()*100:.1f}%)  "
      f"[{time.time()-t0:.1f}s]")

# Strategy 3: segment_id contains osmWayId as substring anywhere
# (only check if suffix doesn't already dominate)
# Check first 1000 unique ways for contains
print("  Building contains-match map …")
t0 = time.time()
seg_list = gdf_proj["segment_id"].astype(str).tolist()
contains_map = {}
for wid in unique_ways[:5000]:  # sample first 5000 for speed check
    wid_s = str(wid).strip()
    if not wid_s or wid_s == "nan":
        continue
    for sid in seg_list:
        if wid_s in sid:
            contains_map[wid_s] = sid
            break
contains_sample_mask = events["osmWayId_str"].isin(contains_map)
# Scale: how many unique ways hit?
hit_rate_sample = len(contains_map) / min(5000, len(unique_ways))
print(f"  Contains-match (sample 5000 ways): {contains_sample_mask.sum():,} events  "
      f"hit-rate={hit_rate_sample*100:.1f}%  [{time.time()-t0:.1f}s]")

# Decide best approach
best_rate = max(direct_mask.mean(), strip_mask.mean(), suffix_mask.mean())
print(f"\n  Best ID-match rate: {best_rate*100:.1f}%")

# Build final id_match series: segment_id per event row (or NaN)
events["segment_id"] = np.nan
events["match_type"] = "unmatched"

# Apply best strategy: suffix_map covers "embedded" case
# Direct match first, then suffix
direct_rows = events[direct_mask].index
events.loc[direct_rows, "segment_id"] = events.loc[direct_rows, "osmWayId_str"]
events.loc[direct_rows, "match_type"] = "direct_id"

# Suffix match for remaining unmatched
unmatched_mask = events["match_type"] == "unmatched"
can_suffix = unmatched_mask & events["osmWayId_str"].isin(suffix_map)
events.loc[can_suffix, "segment_id"] = events.loc[can_suffix, "osmWayId_str"].map(suffix_map)
events.loc[can_suffix, "match_type"] = "suffix_id"

print(f"\n  After ID matching:")
print(events["match_type"].value_counts().to_string())

# ── l) Spatial nearest-segment matching for remaining unmatched ────────────────
print("\n── STEP B-l  Spatial matching for unmatched events ─────────────────")

unmatched_mask = events["match_type"] == "unmatched"
unmatched_df   = events[unmatched_mask & events["lat"].notna() & events["lon"].notna()].copy()
print(f"  Unmatched events with valid coords: {len(unmatched_df):,}")

if len(unmatched_df) > 0:
    t0 = time.time()
    unmatched_gdf = gpd.GeoDataFrame(
        unmatched_df,
        geometry=gpd.points_from_xy(unmatched_df["lon"], unmatched_df["lat"]),
        crs=4326
    ).to_crs(28355)

    # Use sjoin_nearest (GeoPandas ≥ 0.10)
    road_geom = gdf_proj[["segment_id", "geometry"]].copy()
    joined = gpd.sjoin_nearest(
        unmatched_gdf[["geometry"]],
        road_geom,
        how="left",
        max_distance=500,
        distance_col="snap_dist"
    )
    # joined index aligns with unmatched_gdf index
    # handle potential duplicates from sjoin (take first match per point)
    joined = joined[~joined.index.duplicated(keep="first")]

    spatial_seg    = joined["segment_id"]
    has_spatial    = spatial_seg.notna()

    events.loc[unmatched_df.index[has_spatial], "segment_id"] = spatial_seg[has_spatial].values
    events.loc[unmatched_df.index[has_spatial], "match_type"] = "spatial"

    print(f"  Spatial matched: {has_spatial.sum():,} / {len(unmatched_df):,}  [{time.time()-t0:.1f}s]")

# ── n) Match summary ──────────────────────────────────────────────────────────
print(f"\n  Final match counts:")
print(events["match_type"].value_counts().to_string())
matched_events = events[events["segment_id"].notna()].copy()
print(f"\n  Total matched events: {len(matched_events):,} / {len(events):,} "
      f"({len(matched_events)/len(events)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP C — Aggregate near-miss events per road segment
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP C  Aggregate per road segment ───────────────────────────────")

me = matched_events.copy()
me["segment_id"] = me["segment_id"].astype(str)

# Helper: count with condition
def count_cond(series, condition):
    return condition.astype(int).groupby(me["segment_id"]).sum()

# ── Per-segment totals ────────────────────────────────────────────────────────
grp = me.groupby("segment_id")

agg_total = pd.DataFrame({
    "nm_total":          grp.size(),
    "nm_steering":       grp.apply(lambda x: (x["nearMiss.type"] == "NEAR_MISS_TYPE_STEERING").sum()),
    "nm_braking":        grp.apply(lambda x: (x["nearMiss.type"] == "NEAR_MISS_TYPE_BRAKING").sum()),
    "nm_combined":       grp.apply(lambda x: (x["nearMiss.type"] == "NEAR_MISS_TYPE_COMBINED").sum()),
    "nm_car":            grp.apply(lambda x: (x["vehicleType"] == "CAR").sum()),
    "nm_hcv":            grp.apply(lambda x: (x["vehicleType"] == "HCV").sum()),
    "nm_totalcount_sum": grp["totalCount"].sum(),
    "avg_speed_all":     grp["speed (km/h)"].mean(),
    "avg_accel_all":     grp["acceleration.total (G)"].mean(),
    "avg_iri_all":       grp["iri"].mean(),
    "max_speed":         grp["speed (km/h)"].max(),
})

# ── Per time-period aggregations ──────────────────────────────────────────────
period_frames = {}
for period in ["am", "op", "pm", "ev"]:
    pm_events = me[me["time_period"] == period]
    if len(pm_events) == 0:
        period_frames[period] = pd.DataFrame()
        continue
    pg = pm_events.groupby("segment_id")
    pf = pd.DataFrame({
        f"nm_{period}":          pg.size(),
        f"nm_{period}_steering": pg.apply(lambda x: (x["nearMiss.type"] == "NEAR_MISS_TYPE_STEERING").sum()),
        f"nm_{period}_braking":  pg.apply(lambda x: (x["nearMiss.type"] == "NEAR_MISS_TYPE_BRAKING").sum()),
        f"nm_{period}_combined": pg.apply(lambda x: (x["nearMiss.type"] == "NEAR_MISS_TYPE_COMBINED").sum()),
        f"nm_{period}_car":      pg.apply(lambda x: (x["vehicleType"] == "CAR").sum()),
        f"nm_{period}_hcv":      pg.apply(lambda x: (x["vehicleType"] == "HCV").sum()),
        f"avg_speed_{period}":   pg["speed (km/h)"].mean(),
        f"avg_accel_{period}":   pg["acceleration.total (G)"].mean(),
        f"avg_iri_{period}":     pg["iri"].mean(),
    })
    period_frames[period] = pf

# ── Per monthly period ────────────────────────────────────────────────────────
month_frames = {}
for month in ["aug2025", "sep_nov2025", "dec_feb2026", "mar_apr2026", "may2026", "jun2026"]:
    mo = me[me["month_period"] == month]
    if len(mo) == 0:
        month_frames[month] = pd.DataFrame()
        continue
    mf = mo.groupby("segment_id").size().rename(f"nm_{month}")
    month_frames[month] = mf.to_frame()

# ── Combine all aggregations ──────────────────────────────────────────────────
agg_nm = agg_total.copy()
for period in ["am", "op", "pm", "ev"]:
    if not period_frames[period].empty:
        agg_nm = agg_nm.join(period_frames[period], how="left")
for month in ["aug2025", "sep_nov2025", "dec_feb2026", "mar_apr2026", "may2026", "jun2026"]:
    if not month_frames[month].empty:
        agg_nm = agg_nm.join(month_frames[month], how="left")

# Fill count columns with 0
count_cols = [c for c in agg_nm.columns if c.startswith("nm_") and "avg" not in c and "per_km" not in c]
agg_nm[count_cols] = agg_nm[count_cols].fillna(0).astype(int)

print(f"  Aggregated rows (segments with ≥1 near-miss): {len(agg_nm):,}")
print(f"  Aggregation columns: {len(agg_nm.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP D — Join to GeoPackage attributes
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP D  Join to road network ─────────────────────────────────────")

# Compute segment length in metres (projected CRS = EPSG:28355)
gdf_proj = gdf_proj.copy()
gdf_proj["segment_length_m"] = gdf_proj.geometry.length

# Add centroid lat/lon (WGS84)
gdf_wgs["centroid_lon"] = gdf_wgs.geometry.centroid.x
gdf_wgs["centroid_lat"] = gdf_wgs.geometry.centroid.y
gdf_proj["centroid_lon"] = gdf_wgs["centroid_lon"].values
gdf_proj["centroid_lat"] = gdf_wgs["centroid_lat"].values

# Merge aggregated near-miss onto gdf
agg_nm_reset = agg_nm.reset_index()   # segment_id becomes a column
merged = gdf_proj.merge(agg_nm_reset, on="segment_id", how="left")

# Fill count columns with 0 for segments with no near-miss events
count_cols_all = [c for c in agg_nm.columns if c.startswith("nm_") and "avg" not in c and "per_km" not in c]
for c in count_cols_all:
    if c in merged.columns:
        merged[c] = merged[c].fillna(0)

# Ensure month columns exist (may be missing if no events in that month)
for month in ["aug2025", "sep_nov2025", "dec_feb2026", "mar_apr2026", "may2026", "jun2026"]:
    col = f"nm_{month}"
    if col not in merged.columns:
        merged[col] = 0

# Ensure all period columns exist
for period in ["am", "op", "pm", "ev"]:
    for suffix in ["", "_steering", "_braking", "_combined", "_car", "_hcv"]:
        col = f"nm_{period}{suffix}"
        if col not in merged.columns:
            merged[col] = 0

# Cast count cols to integer
nm_count_cols = [c for c in merged.columns
                 if c.startswith("nm_") and not c.startswith("nm_per") and c != "nm_totalcount_sum"]
merged[nm_count_cols] = merged[nm_count_cols].fillna(0).astype(int)

# ── s) Derived safety metrics ─────────────────────────────────────────────────
len_km = merged["segment_length_m"] / 1000.0
len_km_safe = len_km.replace(0, np.nan)

merged["nm_per_km"]    = merged["nm_total"] / len_km_safe
merged["nm_per_km_am"] = merged.get("nm_am", 0) / len_km_safe
merged["nm_per_km_op"] = merged.get("nm_op", 0) / len_km_safe
merged["nm_per_km_pm"] = merged.get("nm_pm", 0) / len_km_safe
merged["nm_per_km_ev"] = merged.get("nm_ev", 0) / len_km_safe

merged["exp_crashes_per_km"] = merged["ExpCrashes"] / len_km_safe
merged["obs_crashes_per_km"] = merged["ObsCrashes"] / len_km_safe

merged["crash_nm_ratio"] = np.where(
    merged["nm_total"] > 0,
    merged["ObsCrashes"] / merged["nm_total"],
    np.nan
)

# Composite safety index (normalise each component to [0,1])
def minmax(s):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0.0
    return (s - mn) / (mx - mn)

merged["safety_index"] = (
    minmax(merged["ExpCrashes"].fillna(0))  * 0.4 +
    minmax(merged["nm_per_km"].fillna(0))   * 0.3 +
    minmax(merged["avg_accel_all"].fillna(0)) * 0.3
)

merged["has_nearmiss"] = (merged["nm_total"] > 0).astype(int)
merged["has_crashes"]  = (merged["ObsCrashes"] > 0).astype(int)

print(f"  Merged GeoDataFrame: {len(merged):,} rows, {len(merged.columns)} columns")
print(f"  Segments with near-miss events: {merged['has_nearmiss'].sum():,}")
print(f"  Segments with observed crashes: {merged['has_crashes'].sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP E — Write GeoPackage
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP E  Write GeoPackage ─────────────────────────────────────────")

merged_gdf = merged.copy()
merged_gdf.to_file(str(OUT_GPKG), layer="roads_with_nearmiss", driver="GPKG")
gpkg_size = OUT_GPKG.stat().st_size / 1e6
print(f"  Written: {OUT_GPKG}")
print(f"  File size: {gpkg_size:.1f} MB  |  rows: {len(merged_gdf):,}  |  cols: {len(merged_gdf.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP F — Export near_miss_ecf.csv
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP F  Export CSV ───────────────────────────────────────────────")

# Define column order
csv_cols = [
    # SEGMENT IDENTIFICATION
    "segment_id", "Hom_Sec_ID", "ID", "Road_Name", "Road_ID",
    # LOCATION
    "locality_1", "locality_2", "lga_1", "lga_2",
    "locality_left", "locality_right", "lga_name_left", "lga_name_right",
    "centroid_lat", "centroid_lon", "segment_length_m",
    # ROAD CHARACTERISTICS
    "class", "Road_Stereotype", "surface_type",
    # OBSERVED CRASHES
    "Obs_C_Run_off_Road", "Obs_C_Head_on", "Obs_C_Rear_end",
    "Obs_C_Intersection", "Obs_C_Other", "ObsCrashes", "obs_crashes_per_km",
    # EXPECTED CRASHES
    "Exp_C_Run_off_Road", "Exp_C_Head_on", "Exp_C_Rear_end",
    "Exp_C_Intersection", "Exp_C_Other", "ExpCrashes", "exp_crashes_per_km",
    # NEAR-MISS TOTALS
    "nm_total", "nm_steering", "nm_braking", "nm_combined",
    "nm_car", "nm_hcv", "nm_totalcount_sum", "nm_per_km",
    # NEAR-MISS BY TIME PERIOD
    "nm_am", "nm_am_steering", "nm_am_braking", "nm_am_combined", "nm_am_car", "nm_am_hcv", "nm_per_km_am",
    "nm_op", "nm_op_steering", "nm_op_braking", "nm_op_combined", "nm_op_car", "nm_op_hcv", "nm_per_km_op",
    "nm_pm", "nm_pm_steering", "nm_pm_braking", "nm_pm_combined", "nm_pm_car", "nm_pm_hcv", "nm_per_km_pm",
    "nm_ev", "nm_ev_steering", "nm_ev_braking", "nm_ev_combined", "nm_ev_car", "nm_ev_hcv", "nm_per_km_ev",
    # NEAR-MISS BY MONTH
    "nm_aug2025", "nm_sep_nov2025", "nm_dec_feb2026", "nm_mar_apr2026", "nm_may2026", "nm_jun2026",
    # SPEED & ACCEL
    "avg_speed_all", "avg_speed_am", "avg_speed_op", "avg_speed_pm", "avg_speed_ev", "max_speed",
    "avg_accel_all", "avg_accel_am", "avg_accel_op", "avg_accel_pm", "avg_accel_ev",
    "avg_iri_all", "avg_iri_am", "avg_iri_op", "avg_iri_pm", "avg_iri_ev",
    # COMPOSITE SAFETY METRICS
    "crash_nm_ratio", "exp_crashes_per_km", "nm_per_km", "safety_index",
    "has_nearmiss", "has_crashes",
]

# Some columns may be missing (e.g. avg_speed_am if no am-period data) — add them as NaN
for c in csv_cols:
    if c not in merged.columns:
        merged[c] = np.nan

# Drop geometry for CSV
csv_df = merged[csv_cols].copy()
csv_df.to_csv(str(OUT_CSV), index=False)

csv_size = OUT_CSV.stat().st_size / 1e6
pct_nm   = merged["has_nearmiss"].mean() * 100
print(f"  Written: {OUT_CSV}")
print(f"  File size: {csv_size:.2f} MB  |  rows: {len(csv_df):,}  |  cols: {len(csv_df.columns)}")
print(f"  % segments with near-miss events: {pct_nm:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

both_mask = (merged["has_nearmiss"] == 1) & (merged["has_crashes"] == 1)
total_nm  = events["match_type"].value_counts().get("unmatched", 0)

print(f"  Total road segments:                {len(merged):>8,}")
print(f"  Segments with near-miss events:     {merged['has_nearmiss'].sum():>8,}  "
      f"({merged['has_nearmiss'].mean()*100:.1f}%)")
print(f"  Segments with observed crashes:     {merged['has_crashes'].sum():>8,}  "
      f"({merged['has_crashes'].mean()*100:.1f}%)")
print(f"  Segments with BOTH:                 {both_mask.sum():>8,}")
print(f"  Total events loaded:                {len(events):>8,}")
print(f"  Total near-miss events matched:     {len(matched_events):>8,}  "
      f"({len(matched_events)/len(events)*100:.1f}%)")

print(f"\n  Near-miss breakdown by time period:")
print(f"    AM (07-08): {(events['time_period']=='am').sum():>8,}")
print(f"    OP (09-15): {(events['time_period']=='op').sum():>8,}")
print(f"    PM (16-17): {(events['time_period']=='pm').sum():>8,}")
print(f"    EV (18-06): {(events['time_period']=='ev').sum():>8,}")

print(f"\n  Top 5 segments by nm_total:")
top5 = merged.nlargest(5, "nm_total")[
    ["segment_id", "Road_Name", "class", "nm_total", "ExpCrashes", "ObsCrashes"]
]
print(top5.to_string(index=False))

print(f"\n  Top 10 segments by safety_index (combined crash + near-miss metric):")
top10 = merged.nlargest(10, "safety_index")[
    ["segment_id", "Road_Name", "class", "Road_Stereotype",
     "nm_total", "nm_per_km", "ExpCrashes", "ObsCrashes", "safety_index"]
]
print(top10.to_string(index=False))

print(f"\n  CSV columns ({len(csv_cols)} total):")
for i, c in enumerate(csv_cols):
    print(f"    {i+1:>3}. {c}")

print("\n  DONE")
print("=" * 70)
