"""
06_build_integrated_omx.py
Combine all outputs into a single integrated OMX for ABM use.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, r"Z:\test_runs_tours\demand_")
from omx_utils import write_omx, read_omx_matrix, list_omx_matrices

OUT_DIR = r"Z:\test_runs_tours\demand_"

# Load zone list
with open(r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt") as f:
    content = f.read()
zone_ids_507 = sorted([int(x) for x in re.split(r'[\s,]+', content.strip()) if x])
n_sa2 = len(zone_ids_507)
sa2_to_idx = {code: i for i, code in enumerate(zone_ids_507)}
print(f"Zone count: {n_sa2}")

def make_od_matrix(df, origin_col, dest_col, value_col):
    """Pivot OD DataFrame to n_sa2 x n_sa2 numpy array."""
    mat = np.zeros((n_sa2, n_sa2), dtype=np.float32)
    for _, row in df.iterrows():
        try:
            o = sa2_to_idx.get(int(row[origin_col]), -1)
            d = sa2_to_idx.get(int(row[dest_col]), -1)
            if o >= 0 and d >= 0:
                v = row[value_col]
                if pd.notna(v):
                    mat[o, d] += float(v)
        except (ValueError, TypeError):
            pass
    return mat

# --- a) Synth pop OD ---
print("Loading synth pop OD...")
synth_od = pd.read_csv(os.path.join(OUT_DIR, 'synth_pop_od_507.csv'))
print(f"Synth OD: {len(synth_od):,} rows")

synth_matrices = {}
for mode in ['car', 'pt', 'walk', 'bike']:
    sub = synth_od[synth_od['mode'] == mode]
    synth_matrices[f'demand_synth_{mode}'] = make_od_matrix(sub, 'origin_sa2', 'dest_sa2', 'demand')
    print(f"  demand_synth_{mode}: sum={synth_matrices[f'demand_synth_{mode}'].sum():.0f}")

# --- b) HHTS OD ---
print("\nLoading HHTS OD...")
hhts_od = pd.read_csv(os.path.join(OUT_DIR, 'hhts_od_by_mode_507.csv'))
print(f"HHTS OD: {len(hhts_od):,} rows")

hhts_matrices = {}
for mode in ['car', 'pt', 'walk', 'bike']:
    sub = hhts_od[hhts_od['mode'] == mode]
    hhts_matrices[f'demand_hhts_{mode}'] = make_od_matrix(sub, 'origin_sa2', 'dest_sa2', 'weighted_trips')
    print(f"  demand_hhts_{mode}: sum={hhts_matrices[f'demand_hhts_{mode}'].sum():.0f}")

# Combined
combined_matrices = {}
for mode in ['car', 'pt', 'walk', 'bike']:
    combined = (synth_matrices[f'demand_synth_{mode}'] +
                hhts_matrices[f'demand_hhts_{mode}'])
    combined_matrices[f'demand_combined_{mode}'] = combined.astype(np.float32)
total_demand = sum(combined_matrices.values())
combined_matrices['demand_total'] = total_demand.astype(np.float32)

# --- c) Skim matrices ---
print("\nLoading skim matrices...")
skim_matrices = {}

SKIM_NAME_MAP = {
    'skim_car_time_am': 'SOVTOLL_TIME__AM',
    'skim_car_time_pm': 'SOVTOLL_TIME__PM',
    'skim_car_time_md': 'SOVTOLL_TIME__MD',
    'skim_car_time_ea': 'SOVTOLL_TIME__EA',
    'skim_car_time_ev': 'SOVTOLL_TIME__EV',
    'skim_car_dist_am': 'SOVTOLL_DIST__AM',
    'skim_car_dist_pm': 'SOVTOLL_DIST__PM',
    'skim_car_dist_md': 'SOVTOLL_DIST__MD',
    'skim_car_toll_am': 'SOVTOLL_TOLL__AM',
    'skim_car_toll_pm': 'SOVTOLL_TOLL__PM',
    'skim_transit_ivt_am': 'DRV_LOC_WLK_TOTIVT__AM',
    'skim_transit_ivt_pm': 'DRV_LOC_WLK_TOTIVT__PM',
    'skim_transit_fare_am': 'DRV_LOC_WLK_FAR__AM',
    'skim_transit_fare_pm': 'DRV_LOC_WLK_FAR__PM',
    'skim_dist_car': 'DIST',
    'skim_dist_bike': 'DISTBIKE',
    'skim_dist_walk': 'DISTWALK',
}

skims_omx_path = os.path.join(OUT_DIR, 'sa2_skims_507.omx')
if os.path.exists(skims_omx_path):
    available_skims = list_omx_matrices(skims_omx_path)
    print(f"Available skims: {available_skims}")
    for out_name, src_name in SKIM_NAME_MAP.items():
        if src_name in available_skims:
            skim_matrices[out_name] = read_omx_matrix(skims_omx_path, src_name)
        else:
            print(f"  WARNING: {src_name} not found, filling zeros")
            skim_matrices[out_name] = np.zeros((n_sa2, n_sa2), dtype=np.float32)
else:
    print("WARNING: sa2_skims_507.omx not found, filling zeros")
    for out_name in SKIM_NAME_MAP:
        skim_matrices[out_name] = np.zeros((n_sa2, n_sa2), dtype=np.float32)

# --- d) Crash matrices ---
print("\nLoading crash matrices...")
crash_matrices = {}
CRASH_NAME_MAP = {
    'crash_total': 'crash_total',
    'crash_runoff': 'crash_runoff',
    'crash_headon': 'crash_headon',
    'crash_rearend': 'crash_rearend',
    'crash_intersection': 'crash_intersection',
    'crash_other': 'crash_other',
    'crash_rate': 'crash_rate_total',
}

crash_omx_path = os.path.join(OUT_DIR, 'crash_matrices_507.omx')
if os.path.exists(crash_omx_path):
    available_crash = list_omx_matrices(crash_omx_path)
    for out_name, src_name in CRASH_NAME_MAP.items():
        if src_name in available_crash:
            crash_matrices[out_name] = read_omx_matrix(crash_omx_path, src_name)
        else:
            print(f"  WARNING: {src_name} not found, filling zeros")
            crash_matrices[out_name] = np.zeros((n_sa2, n_sa2), dtype=np.float32)
else:
    print("WARNING: crash_matrices_507.omx not found, filling zeros")
    for out_name in CRASH_NAME_MAP:
        crash_matrices[out_name] = np.zeros((n_sa2, n_sa2), dtype=np.float32)

# HHTS calibration skims (using AM skim as proxy)
hhts_skim_matrices = {
    'hhts_avg_travtime_car': skim_matrices.get(
        'skim_car_time_am', np.zeros((n_sa2, n_sa2), dtype=np.float32)),
    'hhts_avg_dist_car': skim_matrices.get(
        'skim_car_dist_am', np.zeros((n_sa2, n_sa2), dtype=np.float32)),
}

# --- e/f) Merge and write ---
all_matrices = {}
all_matrices.update(synth_matrices)
all_matrices.update(hhts_matrices)
all_matrices.update(combined_matrices)
all_matrices.update(skim_matrices)
all_matrices.update(crash_matrices)
all_matrices.update(hhts_skim_matrices)

out_omx = os.path.join(OUT_DIR, 'demand_integrated_sa2.omx')
if os.path.exists(out_omx):
    os.remove(out_omx)

zone_mapping = {str(code): i for i, code in enumerate(zone_ids_507)}
print(f"\nWriting integrated OMX ({len(all_matrices)} matrices)...")
write_omx(out_omx, all_matrices, zone_mapping=zone_mapping, shape=(n_sa2, n_sa2))

fsize = os.path.getsize(out_omx) / (1024**3)
print(f"Saved {out_omx} ({fsize:.2f} GB)")

print(f"\n=== INTEGRATED OMX SUMMARY ===")
print(f"Zone count: {n_sa2}")
print(f"Total matrices: {len(all_matrices)}")
print(f"Matrix names and statistics:")
for name in sorted(all_matrices.keys()):
    m = all_matrices[name]
    nz = int(np.count_nonzero(m))
    print(f"  {name}: non-zero={nz:,}, sum={float(m.sum()):.1f}")
