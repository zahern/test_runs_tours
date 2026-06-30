"""
03_hhts_od_507.py
Extract HHTS OD for all trip purposes at SA2 level.
"""
import os
import re
import pandas as pd

OUT_DIR = r"Z:\test_runs_tours\demand_"

# --- a) Load HHTS trips ---
print("Loading HHTS trips...")
hhts = pd.read_excel(r"Z:\test_runs_tours\5_QTS_TRIPS.xlsx")
print(f"HHTS shape: {hhts.shape}")
total_trips = len(hhts)

# --- b) Convert SA1 codes to SA2 ---
hhts['orig_sa2'] = hhts['ORIGSA1_2021'].astype(str).str.strip().str[:9]
hhts['dest_sa2'] = hhts['DESTSA1_2021'].astype(str).str.strip().str[:9]

# Keep only 9-digit numeric codes
hhts = hhts[hhts['orig_sa2'].str.match(r'^\d{9}$', na=False)]
hhts = hhts[hhts['dest_sa2'].str.match(r'^\d{9}$', na=False)]
print(f"After SA2 extraction (9-digit only): {len(hhts):,}")

# --- c) Filter to 507 zones ---
with open(r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt") as f:
    content = f.read()
zone_ids_507 = [int(x) for x in re.split(r'[\s,]+', content.strip()) if x]
zone_ids_str = set(str(z) for z in zone_ids_507)

hhts_507 = hhts[
    hhts['orig_sa2'].isin(zone_ids_str) &
    hhts['dest_sa2'].isin(zone_ids_str)
].copy()
n_in_507 = len(hhts_507)
print(f"Trips with both SA2s in 507 zones: {n_in_507:,} ({100*n_in_507/total_trips:.1f}%)")

# --- d) Map MAINMODE -> mode category ---
def map_mainmode(mode):
    if pd.isna(mode):
        return 'other'
    mode = str(mode).lower().strip()
    if any(x in mode for x in ['train', 'bus', 'ferry', 'tram', 'transit', 'public transport', 'light rail']):
        return 'pt'
    elif any(x in mode for x in ['car', 'driver', 'passenger', 'truck', 'motor', 'taxi', 'rideshare', 'vehicle']):
        return 'car'
    elif any(x in mode for x in ['bike', 'cycl', 'bicycle']):
        return 'bike'
    elif any(x in mode for x in ['walk', 'foot', 'pedestrian']):
        return 'walk'
    else:
        return 'other'

print(f"\nMAINMODE values in 507 subset:")
print(hhts_507['MAINMODE'].value_counts().head(20))

hhts_507['mode'] = hhts_507['MAINMODE'].apply(map_mainmode)

# --- e) Map OVERALL_PURPOSE -> categories ---
def map_purpose(purp):
    if pd.isna(purp):
        return 'other'
    purp = str(purp).lower().strip()
    if 'work' in purp:
        return 'work'
    elif any(x in purp for x in ['school', 'educat', 'uni', 'tafe', 'college']):
        return 'education'
    elif any(x in purp for x in ['shop', 'retail', 'grocery']):
        return 'shopping'
    elif any(x in purp for x in ['recreat', 'sport', 'leisure', 'social', 'entertain']):
        return 'recreation'
    elif any(x in purp for x in ['home', 'return']):
        return 'other_home'
    else:
        return 'other'

print(f"\nOVERALL_PURPOSE values in 507 subset:")
print(hhts_507['OVERALL_PURPOSE'].value_counts().head(20))

hhts_507['purpose'] = hhts_507['OVERALL_PURPOSE'].apply(map_purpose)

# --- f) Group by OD-mode-purpose ---
od_grouped = (
    hhts_507
    .groupby(['orig_sa2', 'dest_sa2', 'mode', 'purpose'])['TRIPWGT23']
    .sum()
    .reset_index()
    .rename(columns={'orig_sa2': 'origin_sa2', 'dest_sa2': 'dest_sa2', 'TRIPWGT23': 'weighted_trips'})
)

# Convert to int SA2 codes
od_grouped['origin_sa2'] = od_grouped['origin_sa2'].astype(int)
od_grouped['dest_sa2'] = od_grouped['dest_sa2'].astype(int)

# --- g) Save hhts_od_507.csv ---
out_od = os.path.join(OUT_DIR, 'hhts_od_507.csv')
od_grouped.to_csv(out_od, index=False)
print(f"\nSaved {out_od} ({len(od_grouped):,} rows)")

# --- h) Save hhts_od_by_mode_507.csv ---
od_by_mode = (
    od_grouped
    .groupby(['origin_sa2', 'dest_sa2', 'mode'])['weighted_trips']
    .sum()
    .reset_index()
)
out_mode = os.path.join(OUT_DIR, 'hhts_od_by_mode_507.csv')
od_by_mode.to_csv(out_mode, index=False)
print(f"Saved {out_mode} ({len(od_by_mode):,} rows)")

# --- i) Print summary ---
print("\n=== HHTS OD SUMMARY ===")
print(f"Total trips in raw HHTS: {total_trips:,}")
print(f"Trips with both SA2s in 507: {n_in_507:,} ({100*n_in_507/total_trips:.1f}%)")
print(f"\nWeighted trips by mode:")
print(od_by_mode.groupby('mode')['weighted_trips'].sum().to_string())
print(f"\nWeighted trips by purpose:")
print(od_grouped.groupby('purpose')['weighted_trips'].sum().to_string())
print(f"\nOD pairs: {len(od_by_mode):,}")
print(f"Origin zones with trips: {od_by_mode['origin_sa2'].nunique()}")
print(f"Dest zones with trips: {od_by_mode['dest_sa2'].nunique()}")
