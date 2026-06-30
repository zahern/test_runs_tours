"""
02_synth_pop_demand.py
Load all 507 synth pop zones and build OD demand matrices.
"""
import os
import re
import numpy as np
import pandas as pd

OUT_DIR = r"Z:\test_runs_tours\demand_"
SYNTH_POP_DIR = r"Z:\Paper_Plan\synth_pop\outputs\sa_gan_results_507"

# --- a) Load zone IDs and crosswalk ---
with open(r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt") as f:
    content = f.read()
zone_ids_507 = [int(x) for x in re.split(r'[\s,]+', content.strip()) if x]
zone_set = set(zone_ids_507)
print(f"Loaded {len(zone_ids_507)} zone IDs")

crosswalk = pd.read_csv(os.path.join(OUT_DIR, 'sa2_crosswalk_507.csv'))
print(f"Crosswalk: {len(crosswalk)} rows")

# --- b) Load HHTS trips for mode calibration ---
print("Loading HHTS trips...")
hhts = pd.read_excel(r"Z:\test_runs_tours\5_QTS_TRIPS.xlsx")
print(f"HHTS shape: {hhts.shape}")

# --- c) Load all 507 synth pop zones (skip empty files) ---
print("\nLoading synth pop zones...")
all_dfs = []
missing = []
skipped_empty = []
first_zone = True

for zone_id in zone_ids_507:
    fpath = os.path.join(SYNTH_POP_DIR, f"zone_{zone_id}_synthetic_population.csv")
    if os.path.exists(fpath):
        fsize = os.path.getsize(fpath)
        if fsize < 10:
            skipped_empty.append(zone_id)
            continue
        try:
            df = pd.read_csv(fpath)
            if len(df) == 0:
                skipped_empty.append(zone_id)
                continue
            df['home_sa2'] = zone_id
            if first_zone:
                print(f"\nFirst zone ({zone_id}) mtwp value_counts:")
                print(df['mtwp'].value_counts().sort_index())
                print(f"empp range: {df['empp'].min()} - {df['empp'].max()}")
                first_zone = False
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error reading zone {zone_id}: {e}")
            skipped_empty.append(zone_id)
    else:
        missing.append(zone_id)

print(f"\nLoaded {len(all_dfs)} zones, missing {len(missing)}, skipped empty: {skipped_empty}")

pop = pd.concat(all_dfs, ignore_index=True)
print(f"Total persons loaded: {len(pop):,}")

# --- d) Map MTWP -> mode category ---
# ABS Census 2021 MTWP codes (confirmed from data inspection):
# 1=Train, 2=Bus, 3=Ferry, 4=Tram/light rail, 5=Taxi/rideshare -> pt
# 6=Car driver, 7=Car passenger, 8=Truck, 9=Motorbike -> car
# 10=Bicycle -> bike
# 11=Walk -> walk
# 12=Work at home, 13=Other, 14=Did not go to work, 15=Not applicable, 16=Not stated -> other
print("\nActual mtwp codes found:")
print(pop['mtwp'].value_counts().sort_index())

def map_mtwp(code):
    if code in [1, 2, 3, 4, 5]:
        return 'pt'
    elif code in [6, 7, 8, 9]:
        return 'car'
    elif code == 10:
        return 'bike'
    elif code == 11:
        return 'walk'
    else:
        return 'other'  # 12=work at home, 13=other, 14=didn't go, 15=N/A, 16=not stated

pop['mode'] = pop['mtwp'].apply(map_mtwp)
print("\nMode distribution:")
print(pop['mode'].value_counts())

# --- e) Count home-based work productions by (home_sa2, mode) ---
# empp 1-5 = employed (full time, part time, away from work, employer, self-employed)
workers = pop[
    (pop['empp'].isin([1, 2, 3, 4, 5])) &
    (pop['mode'] != 'other')
].copy()
print(f"\nTotal employed travellers: {len(workers):,}")

productions = workers.groupby(['home_sa2', 'mode']).size().reset_index(name='count')
print(f"Productions by mode:")
print(productions.groupby('mode')['count'].sum())

# --- f) Build OD matrix using HHTS destination proportions ---
hhts['orig_sa2'] = hhts['ORIGSA1_2021'].astype(str).str.strip().str[:9]
hhts['dest_sa2'] = hhts['DESTSA1_2021'].astype(str).str.strip().str[:9]
hhts = hhts[hhts['orig_sa2'].str.match(r'^\d{9}$', na=False)]
hhts = hhts[hhts['dest_sa2'].str.match(r'^\d{9}$', na=False)]
print(f"\nHHTS after SA2 extraction: {len(hhts):,} trips")
print(f"HHTS MAINMODE values:")
print(hhts['MAINMODE'].value_counts().head(20))

def map_mainmode(mode):
    if pd.isna(mode):
        return 'other'
    mode = str(mode).lower().strip()
    if any(x in mode for x in ['train', 'bus', 'ferry', 'tram', 'transit', 'public', 'light rail']):
        return 'pt'
    elif any(x in mode for x in ['car', 'driver', 'passenger', 'truck', 'motor', 'taxi', 'rideshare', 'vehicle', 'auto']):
        return 'car'
    elif any(x in mode for x in ['bike', 'cycl', 'bicycle']):
        return 'bike'
    elif any(x in mode for x in ['walk', 'foot', 'pedestrian']):
        return 'walk'
    else:
        return 'other'

hhts['mode'] = hhts['MAINMODE'].apply(map_mainmode)
print(f"\nHHTS mode distribution:")
print(hhts['mode'].value_counts())

# Work trips
hhts['purpose_lower'] = hhts['OVERALL_PURPOSE'].astype(str).str.lower()
work_trips = hhts[
    hhts['purpose_lower'].str.contains('work', na=False) |
    hhts['DESTPURP'].astype(str).str.lower().str.contains('work', na=False)
].copy()
print(f"\nHHTS work trips: {len(work_trips):,}")
print(f"OVERALL_PURPOSE sample:")
print(hhts['OVERALL_PURPOSE'].value_counts().head(10))

zone_ids_str = set(str(z) for z in zone_ids_507)
work_trips_507 = work_trips[
    work_trips['orig_sa2'].isin(zone_ids_str) &
    work_trips['dest_sa2'].isin(zone_ids_str)
].copy()
print(f"HHTS work trips with both SA2s in 507: {len(work_trips_507):,}")

if len(work_trips_507) > 200:
    trips_for_props = work_trips_507
    print("Using 507-zone work trips for destination proportions")
else:
    trips_for_props = work_trips
    print(f"Falling back to all work trips ({len(work_trips)} total)")

trips_for_props = trips_for_props[trips_for_props['mode'] != 'other'].copy()

dest_props = (
    trips_for_props
    .groupby(['orig_sa2', 'dest_sa2', 'mode'])['TRIPWGT23']
    .sum()
    .reset_index()
)
dest_props['total_by_orig_mode'] = dest_props.groupby(['orig_sa2', 'mode'])['TRIPWGT23'].transform('sum')
dest_props['prop'] = dest_props['TRIPWGT23'] / dest_props['total_by_orig_mode']
print(f"\nDestination proportions: {len(dest_props):,} OD-mode combos")

# Global fallback proportions by mode
mode_dest_props = {}
for mode in ['car', 'pt', 'bike', 'walk']:
    mode_df = trips_for_props[trips_for_props['mode'] == mode]
    if len(mode_df) > 0:
        dest_counts = mode_df.groupby('dest_sa2')['TRIPWGT23'].sum()
        dest_total = dest_counts.sum()
        if dest_total > 0:
            mode_dest_props[mode] = (dest_counts / dest_total).to_dict()
    print(f"  {mode} fallback destinations: {len(mode_dest_props.get(mode, {}))}")

prod_lookup = productions.set_index(['home_sa2', 'mode'])['count'].to_dict()
od_rows = []
n_origin_specific = 0
n_fallback = 0

print("\nBuilding OD matrix...")
for (home_sa2, mode), production in prod_lookup.items():
    home_sa2_str = str(home_sa2)
    origin_props = dest_props[
        (dest_props['orig_sa2'] == home_sa2_str) &
        (dest_props['mode'] == mode)
    ]
    if len(origin_props) > 0:
        n_origin_specific += 1
        for _, prop_row in origin_props.iterrows():
            dest = prop_row['dest_sa2']
            if dest in zone_ids_str:
                demand = production * prop_row['prop']
                if demand > 0:
                    od_rows.append({'origin_sa2': home_sa2, 'dest_sa2': int(dest), 'mode': mode, 'demand': demand})
    elif mode in mode_dest_props:
        n_fallback += 1
        for dest, prop in mode_dest_props[mode].items():
            if dest in zone_ids_str:
                demand = production * prop
                if demand > 0:
                    od_rows.append({'origin_sa2': home_sa2, 'dest_sa2': int(dest), 'mode': mode, 'demand': demand})

print(f"Origin-specific proportions: {n_origin_specific}, Fallback: {n_fallback}")
od_df = pd.DataFrame(od_rows)
print(f"OD pairs generated: {len(od_df):,}")

# --- g) Scale to match HHTS total weighted trips per mode ---
hhts_mode_totals = work_trips[work_trips['mode'] != 'other'].groupby('mode')['TRIPWGT23'].sum()
synth_mode_totals = od_df.groupby('mode')['demand'].sum()

print("\nScaling demand to HHTS totals:")
for mode in ['car', 'pt', 'bike', 'walk']:
    hhts_total = hhts_mode_totals.get(mode, 0)
    synth_total = synth_mode_totals.get(mode, 0)
    if synth_total > 0 and hhts_total > 0:
        scale = hhts_total / synth_total
        od_df.loc[od_df['mode'] == mode, 'demand'] *= scale
        print(f"  {mode}: synth={synth_total:.0f}, hhts={hhts_total:.0f}, scale={scale:.3f}")
    else:
        print(f"  {mode}: synth={synth_total:.0f}, hhts={hhts_total:.0f} - skipped")

# --- h) Save outputs ---
out_od = os.path.join(OUT_DIR, 'synth_pop_od_507.csv')
od_df.to_csv(out_od, index=False)
print(f"\nSaved {out_od} ({len(od_df):,} rows)")

summary = od_df.groupby('mode')['demand'].sum().reset_index()
summary.columns = ['mode', 'total_demand']
summary['n_zones_loaded'] = len(all_dfs)
out_sum = os.path.join(OUT_DIR, 'synth_pop_summary_507.csv')
summary.to_csv(out_sum, index=False)
print(f"Saved {out_sum}")

print("\n=== SYNTH POP SUMMARY ===")
print(f"Zones loaded: {len(all_dfs)}/507")
print(f"Skipped (empty): {skipped_empty}")
print(f"Total persons: {len(pop):,}")
print(f"Employed travellers: {len(workers):,}")
print(f"Total demand by mode (after scaling):")
print(od_df.groupby('mode')['demand'].sum().to_string())
print(f"Total OD pairs: {len(od_df):,}")
