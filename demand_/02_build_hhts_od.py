"""
Script 02: Extract HHTS trip OD at SA2 level by mode.
SA1 -> SA2 mapping via first 9 chars of SA1_2021 code.
"""
import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
QTS_TRIPS  = r"Z:\test_runs_tours\5_QTS_TRIPS.xlsx"
OUTPUT_DIR = r"Z:\test_runs_tours\demand_"
MAPPING_CSV = os.path.join(OUTPUT_DIR, 'sa2_to_emme_mapping.csv')

print("=" * 60)
print("Script 02: Build HHTS OD at SA2 Level")
print("=" * 60)

# ── 1. Load QTS trips ─────────────────────────────────────────────────────────
print("\n[1] Loading QTS_TRIPS.xlsx ...")
trips = pd.read_excel(QTS_TRIPS)
print(f"    Loaded {len(trips)} trips")
print(f"    Columns: {list(trips.columns)}")

# ── 2. SA1 -> SA2 mapping ─────────────────────────────────────────────────────
print("\n[2] Deriving SA2 codes from SA1 codes (first 9 chars) ...")

def sa1_to_sa2(sa1_code):
    """Convert SA1 code (11-digit string) to SA2 code (9-digit prefix)."""
    s = str(sa1_code).strip()
    # SA1 2021 codes are 11 digits; SA2 = first 9
    # Remove any decimal suffix if read as float
    if '.' in s:
        s = s.split('.')[0]
    s = s.zfill(11)
    return s[:9]

trips['ORIGSA1_str'] = trips['ORIGSA1_2021'].apply(
    lambda x: str(int(x)) if pd.notna(x) and str(x) != 'nan' else None
)
trips['DESTSA1_str'] = trips['DESTSA1_2021'].apply(
    lambda x: str(int(x)) if pd.notna(x) and str(x) != 'nan' else None
)

trips['origin_sa2'] = trips['ORIGSA1_str'].apply(
    lambda x: sa1_to_sa2(x) if x else None
)
trips['dest_sa2'] = trips['DESTSA1_str'].apply(
    lambda x: sa1_to_sa2(x) if x else None
)

# Load known SA2 codes from mapping for cross-check
if os.path.exists(MAPPING_CSV):
    mapping = pd.read_csv(MAPPING_CSV)
    known_sa2 = set(mapping['sa2_code'].dropna().astype(str))
    trips['origin_sa2_in_seqstm'] = trips['origin_sa2'].isin(known_sa2)
    trips['dest_sa2_in_seqstm'] = trips['dest_sa2'].isin(known_sa2)
    print(f"    Origin in SEQSTM SA2: {trips['origin_sa2_in_seqstm'].sum()} / {len(trips)}")
    print(f"    Dest in SEQSTM SA2:   {trips['dest_sa2_in_seqstm'].sum()} / {len(trips)}")
else:
    print("    WARNING: sa2_to_emme_mapping.csv not found; skipping cross-check")
    known_sa2 = set()

print(f"    Unique origin SA2: {trips['origin_sa2'].nunique()}")
print(f"    Unique dest SA2:   {trips['dest_sa2'].nunique()}")

# ── 3. Map MAINMODE to standard categories ────────────────────────────────────
print("\n[3] Mapping MAINMODE to standard categories ...")
print("    Raw MAINMODE values:")
for mode, cnt in trips['MAINMODE'].value_counts().items():
    print(f"      {mode}: {cnt}")

def map_mode(m):
    if pd.isna(m):
        return 'other'
    m_lower = str(m).lower()
    if any(k in m_lower for k in ['car driver', 'truck driver', 'taxi', 'ride share', 'uber',
                                    'motorcycle', 'truck passenger']):
        return 'car'
    if 'car passenger' in m_lower:
        return 'car'
    if any(k in m_lower for k in ['train', 'bus', 'ferry', 'tram', 'light rail']):
        return 'pt'
    if any(k in m_lower for k in ['walk', 'walking']):
        return 'walk'
    if any(k in m_lower for k in ['bicycle', 'bike', 'e-bike', 'e-scooter', 'ebike', 'escooter']):
        return 'bike'
    return 'other'

trips['mode_category'] = trips['MAINMODE'].apply(map_mode)
print("\n    Mode category totals:")
print(trips['mode_category'].value_counts().to_string())

# ── 4. Group by OD pair and mode ──────────────────────────────────────────────
print("\n[4] Aggregating to SA2 OD by mode ...")

# Filter to rows with valid SA2 codes and weight
valid = trips[
    trips['origin_sa2'].notna() &
    trips['dest_sa2'].notna() &
    trips['TRIPWGT23'].notna()
].copy()
print(f"    Valid trips (both SA2 + weight): {len(valid)}")

od_agg = valid.groupby(
    ['origin_sa2', 'dest_sa2', 'mode_category']
)['TRIPWGT23'].sum().reset_index()
od_agg.columns = ['origin_sa2', 'dest_sa2', 'mode', 'total_weighted_trips']

print(f"    OD pairs (long format): {len(od_agg)}")

# ── 5. Save outputs ───────────────────────────────────────────────────────────
print("\n[5] Saving outputs ...")

out_long = os.path.join(OUTPUT_DIR, 'hhts_od_by_sa2_mode.csv')
od_agg.to_csv(out_long, index=False)
print(f"    Saved: {out_long}")

# Pivot table: rows = (origin_sa2, dest_sa2), cols = modes
od_pivot = od_agg.pivot_table(
    index=['origin_sa2', 'dest_sa2'],
    columns='mode',
    values='total_weighted_trips',
    aggfunc='sum',
    fill_value=0
).reset_index()
od_pivot.columns.name = None

out_pivot = os.path.join(OUTPUT_DIR, 'hhts_od_sa2_pivot.csv')
od_pivot.to_csv(out_pivot, index=False)
print(f"    Saved: {out_pivot}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCRIPT 02 COMPLETE")
print(f"  Total input trips: {len(trips)}")
print(f"  Valid OD trips used: {len(valid)}")
print(f"  Total weighted trips: {valid['TRIPWGT23'].sum():,.0f}")
print(f"  OD long rows: {len(od_agg)}")
print(f"  OD pivot rows: {len(od_pivot)}")
print("\n  Trips per mode:")
for mode, grp in od_agg.groupby('mode'):
    print(f"    {mode}: {grp['total_weighted_trips'].sum():,.0f} weighted trips, "
          f"{len(grp)} OD pairs")
print("=" * 60)
