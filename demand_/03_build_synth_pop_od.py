"""
Script 03: Build OD matrices from synthetic population, balanced against HHTS.
"""
import os
import sys
# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import glob
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SYNTH_BASE     = r"Z:\Paper_Plan\synth_pop"
SYNTH_ZONES_DIR = SYNTH_BASE + r"\zones_sa_5_main"
SYNTH_ZONE_IDS  = SYNTH_BASE + r"\zone_ids_sa_only_507.txt"
OUTPUT_DIR      = r"Z:\test_runs_tours\demand_"
HHTS_OD_CSV     = os.path.join(OUTPUT_DIR, 'hhts_od_by_sa2_mode.csv')
MAPPING_CSV     = os.path.join(OUTPUT_DIR, 'sa2_to_emme_mapping.csv')

print("=" * 60)
print("Script 03: Build Synthetic Population OD Matrices")
print("=" * 60)

# ── 1. Read zone IDs ──────────────────────────────────────────────────────────
print("\n[1] Reading zone IDs ...")
with open(SYNTH_ZONE_IDS) as f:
    zone_ids_from_txt = [l.strip() for l in f if l.strip()]
print(f"    Zone IDs in txt: {len(zone_ids_from_txt)}")

# Find all actual synth pop files in the flat directory
all_files = glob.glob(os.path.join(SYNTH_ZONES_DIR, '*_synthetic_population.csv'))
zone_ids_from_files = [os.path.basename(fp).replace('_synthetic_population.csv', '')
                       for fp in all_files]
print(f"    Synth pop files found: {len(zone_ids_from_files)}")

# Use intersection (files that exist AND are in our zone list)
zone_ids_set = set(zone_ids_from_txt)
valid_zone_files = [(zid, fp) for zid, fp in zip(zone_ids_from_files, all_files)
                    if zid in zone_ids_set]
# Also add files from zone list that aren't in files list
txt_ids_with_files = set(zone_ids_from_files)
missing = [zid for zid in zone_ids_from_txt if zid not in txt_ids_with_files]
print(f"    Zones from txt with matching files: {len(valid_zone_files)}")
if missing:
    print(f"    WARNING: {len(missing)} zone IDs in txt have no matching file: {missing[:5]}")

# If a file exists but not in txt, include it anyway (it's valid synth pop data)
extra_files = [(zid, fp) for zid, fp in zip(zone_ids_from_files, all_files)
               if zid not in zone_ids_set]
if extra_files:
    print(f"    Extra files not in txt (will be included): {len(extra_files)}")
    valid_zone_files.extend(extra_files)

print(f"    Total zone files to process: {len(valid_zone_files)}")

# ── 2. Load all synth pop data ────────────────────────────────────────────────
print("\n[2] Loading synthetic population files ...")
dfs = []
failed = []
for zid, fp in valid_zone_files:
    try:
        df = pd.read_csv(fp, usecols=['abspid', 'mtwp', 'empp', 'agep', 'vehrd'])
        df['zone_id'] = zid
        dfs.append(df)
    except Exception as e:
        failed.append((zid, str(e)))

if failed:
    print(f"    WARNING: Failed to load {len(failed)} files:")
    for zid, err in failed[:5]:
        print(f"      {zid}: {err}")

synth = pd.concat(dfs, ignore_index=True)
print(f"    Total persons loaded: {len(synth):,}")
print(f"    Zones loaded: {synth['zone_id'].nunique()}")

# ── 3. Check MTWP codes ───────────────────────────────────────────────────────
print("\n[3] MTWP value distribution (actual values in data):")
print(synth['mtwp'].value_counts().sort_index().to_string())

# ABS MTWP -> mode_category mapping
# Observed codes from data: 1=Train, 2=Bus, 3=Ferry, 4=Tram, 5=Taxi, 6=?, 7=?,
# 8=?, 9=?, 10=Car driver, 11=Car passenger, 12=Truck, 13=Motorbike,
# 14=Bicycle, 15=Walk only, 16=Worked at home, 17=Did not go to work
# Also seen: 6,7,8,9 which may be variants
MTWP_MAP = {
    1:  'pt',    # Train
    2:  'pt',    # Bus
    3:  'pt',    # Ferry
    4:  'pt',    # Tram/light rail
    5:  'car',   # Taxi/ride-share
    6:  'pt',    # Could be other PT / bus variant
    7:  'car',   # Could be car variant
    8:  'other', # Other
    9:  'other', # Other
    10: 'car',   # Car as driver
    11: 'car',   # Car as passenger
    12: 'car',   # Truck
    13: 'car',   # Motorbike
    14: 'bike',  # Bicycle
    15: 'walk',  # Walked only
    16: 'other', # Worked at home
    17: 'other', # Did not go to work
    8888: 'other',
    9999: 'other',
}

synth['mode_category'] = synth['mtwp'].map(MTWP_MAP).fillna('other')
print("\n    Mode distribution from synth pop:")
print(synth['mode_category'].value_counts().to_string())

# ── 4. Count work trip productions by zone and mode ──────────────────────────
print("\n[4] Computing work trip productions by zone and mode ...")

# Employed persons who travel to work (empp=1..4: employed; mtwp not 16/17/9999)
EMPLOYED_CODES  = [1, 2, 3, 4]
NO_TRAVEL_CODES = [16, 17, 9999]

workers = synth[
    synth['empp'].isin(EMPLOYED_CODES) &
    ~synth['mtwp'].isin(NO_TRAVEL_CODES)
].copy()
print(f"    Workers (employed, travelling): {len(workers):,}")
print(f"    Home zones: {workers['zone_id'].nunique()}")

# Productions: count of workers per home zone and mode
productions = workers.groupby(['zone_id', 'mode_category']).size().reset_index(name='count')
print(f"    Production rows: {len(productions)}")

# Total by mode
print("\n    Work trips by mode (synth pop):")
mode_totals = productions.groupby('mode_category')['count'].sum()
print(mode_totals.to_string())

# ── 5. Build destination proportions from HHTS ───────────────────────────────
print("\n[5] Building destination proportions from HHTS ...")

# Load HHTS OD
if not os.path.exists(HHTS_OD_CSV):
    print("    ERROR: HHTS OD file not found. Run script 02 first.")
    sys.exit(1)

hhts_od = pd.read_csv(HHTS_OD_CSV)
print(f"    HHTS OD rows: {len(hhts_od)}")

# For each mode, compute destination proportions
# dest_prop[mode][dest_sa2] = fraction of trips going to that dest
dest_props = {}
for mode in ['car', 'pt', 'walk', 'bike', 'other']:
    mode_od = hhts_od[hhts_od['mode'] == mode].copy()
    if len(mode_od) == 0:
        dest_props[mode] = pd.Series(dtype=float)
        continue
    dest_total = mode_od.groupby('dest_sa2')['total_weighted_trips'].sum()
    total = dest_total.sum()
    if total > 0:
        dest_props[mode] = dest_total / total
    else:
        dest_props[mode] = pd.Series(dtype=float)
    print(f"    Mode '{mode}': {len(dest_props[mode])} destination SA2s in HHTS")

# ── 6. Get all unique SA2 zones ───────────────────────────────────────────────
# Use the union of synth pop zones and HHTS zones
synth_zones = set(synth['zone_id'].astype(str).unique())
hhts_zones  = set(
    hhts_od['origin_sa2'].astype(str).unique().tolist() +
    hhts_od['dest_sa2'].astype(str).unique().tolist()
)
all_zones = sorted(synth_zones | hhts_zones)
print(f"\n[6] Total unique SA2 zones: {len(all_zones)}")

# ── 7. Build OD matrix ────────────────────────────────────────────────────────
print("\n[7] Building OD matrices per mode (productions x destination_fractions) ...")

od_rows = []
modes = ['car', 'pt', 'walk', 'bike', 'other']

for mode in modes:
    props = dest_props[mode]
    if len(props) == 0:
        print(f"    Mode '{mode}': no HHTS destinations, using uniform distribution")
        # Uniform distribution over all zones
        dest_sa2s = pd.Series(1.0 / len(all_zones), index=all_zones)
    else:
        dest_sa2s = props
        # Fill missing zones with uniform small weight
        missing_zones = [z for z in all_zones if z not in dest_sa2s.index]
        if missing_zones:
            # Give tiny weight to missing destinations
            min_prop = dest_sa2s.min() * 0.1
            dest_sa2s = pd.concat([
                dest_sa2s,
                pd.Series(min_prop, index=missing_zones)
            ])
        # Re-normalise
        dest_sa2s = dest_sa2s / dest_sa2s.sum()

    # Productions for this mode
    prod_mode = productions[productions['mode_category'] == mode].set_index('zone_id')['count']

    # OD = productions[o] * dest_proportion[d]
    mode_rows = 0
    for orig_zone, prod_count in prod_mode.items():
        if prod_count <= 0:
            continue
        for dest_zone, prop in dest_sa2s.items():
            demand = prod_count * prop
            if demand > 0.001:  # threshold to keep sparse
                od_rows.append({
                    'origin_sa2': str(orig_zone),
                    'dest_sa2': str(dest_zone),
                    'mode': mode,
                    'demand': demand,
                })
        mode_rows += 1

    print(f"    Mode '{mode}': {prod_mode.sum():.0f} productions from "
          f"{len(prod_mode)} zones -> {mode_rows} origin zones processed")

synth_od = pd.DataFrame(od_rows)
print(f"\n    Total OD rows before balancing: {len(synth_od):,}")

# ── 8. Balance synth pop OD to HHTS totals ───────────────────────────────────
print("\n[8] Balancing synth pop OD to HHTS totals ...")

hhts_mode_totals = hhts_od.groupby('mode')['total_weighted_trips'].sum()
synth_mode_totals = synth_od.groupby('mode')['demand'].sum()

print("    Mode totals comparison:")
print(f"    {'Mode':<10} {'HHTS':>15} {'Synth':>15} {'Scale factor':>15}")
for mode in modes:
    hhts_total = hhts_mode_totals.get(mode, 0)
    synth_total = synth_mode_totals.get(mode, 0)
    scale = hhts_total / synth_total if synth_total > 0 else 1.0
    print(f"    {mode:<10} {hhts_total:>15,.0f} {synth_total:>15,.0f} {scale:>15.4f}")
    if synth_total > 0 and hhts_total > 0:
        synth_od.loc[synth_od['mode'] == mode, 'demand'] *= scale
    elif synth_total > 0:
        # No HHTS data for this mode; keep synth as-is (already normalised)
        pass

balanced_totals = synth_od.groupby('mode')['demand'].sum()
print("\n    After balancing:")
print(balanced_totals.to_string())

# ── 9. Save outputs ───────────────────────────────────────────────────────────
print("\n[9] Saving outputs ...")

out_od = os.path.join(OUTPUT_DIR, 'synth_pop_od_sa2.csv')
synth_od.to_csv(out_od, index=False)
print(f"    Saved: {out_od}")

out_summary = os.path.join(OUTPUT_DIR, 'synth_pop_od_summary.csv')
summary = synth_od.groupby('mode').agg(
    total_demand=('demand', 'sum'),
    od_pairs=('demand', 'count'),
    n_origins=('origin_sa2', 'nunique'),
    n_dests=('dest_sa2', 'nunique'),
).reset_index()
summary.to_csv(out_summary, index=False)
print(f"    Saved: {out_summary}")

# ── 10. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCRIPT 03 COMPLETE")
print(f"  Total persons in synth pop: {len(synth):,}")
print(f"  Workers (travelling): {len(workers):,}")
print(f"  OD rows (long format): {len(synth_od):,}")
print(f"  Total demand (all modes): {synth_od['demand'].sum():,.0f}")
print("=" * 60)
