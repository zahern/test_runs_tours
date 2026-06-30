"""
10_crash_length_skim_507.py
Build length + crash skim OMX for the 507 SA2 zones.
"""
import numpy as np
import pandas as pd
import h5py
import sys
import os

OUT_DIR = r"Z:\test_runs_tours\demand_"
ZONE_IDS_FILE = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
OD_MATRIX_FILE = r"Z:\test_runs_tours\crash_info_zonal\SEQ_SA2_OD_Matrix_Length.csv"

# ---------- helpers ----------

def write_omx(path, zone_ids, matrices_dict):
    with h5py.File(path, 'w') as f:
        f.attrs['OMX_VERSION'] = '0.2'
        f.attrs['SHAPE'] = np.array([len(zone_ids), len(zone_ids)], dtype=np.int32)
        data_grp = f.create_group('data')
        for name, mat in matrices_dict.items():
            ds = data_grp.create_dataset(name, data=mat.astype(np.float32), compression='gzip')
            ds.attrs['_FillValue'] = np.float32(0.0)
        lkp_grp = f.create_group('lookup')
        lkp_grp.create_dataset('SA2_CODE', data=np.array([str(z) for z in zone_ids], dtype='S12'))
    print(f"  Written: {path} ({len(zone_ids)} zones, {len(matrices_dict)} matrices)")

# ---------- Step a: load 507 zone IDs ----------
print("Step a: Loading 507 zone IDs...")
with open(ZONE_IDS_FILE) as fh:
    zone_ids_507 = sorted([int(line.strip()) for line in fh if line.strip()])
print(f"  {len(zone_ids_507)} zones loaded, range {zone_ids_507[0]}..{zone_ids_507[-1]}")

zone_set = set(zone_ids_507)
zone_index = {z: i for i, z in enumerate(zone_ids_507)}
N = len(zone_ids_507)

# ---------- Step b: load OD matrix ----------
print("Step b: Loading OD matrix...")
df = pd.read_csv(OD_MATRIX_FILE)
print(f"  Loaded {len(df):,} rows. Columns: {list(df.columns)}")

# ---------- Step c: filter to 507 zones ----------
print("Step c: Filtering to 507-zone pairs...")
mask = df['Origin_SA2'].isin(zone_set) & df['Destination_SA2'].isin(zone_set)
df = df[mask].copy()
print(f"  {len(df):,} rows after filter")

# ---------- Step d: pivot to matrices ----------
print("Step d: Building matrices...")

col_map = {
    'Length_km': 'length_km',
    'Exp_C_Run_off_Road': 'crash_runoff',
    'Exp_C_Head_on': 'crash_headon',
    'Exp_C_Rear_end': 'crash_rearend',
    'Exp_C_Intersection': 'crash_intersection',
    'Exp_C_Other': 'crash_other',
    'ExpCrashes': 'crash_total',
    'ExpCrashes_per_km': 'crash_per_km',
}

matrices = {}
for src_col, mat_name in col_map.items():
    piv = df.pivot_table(index='Origin_SA2', columns='Destination_SA2',
                         values=src_col, aggfunc='sum')
    piv = piv.reindex(index=zone_ids_507, columns=zone_ids_507, fill_value=0).fillna(0)
    matrices[mat_name] = piv.values.astype(np.float32)
    print(f"  {mat_name}: sum={matrices[mat_name].sum():.2f}, nonzero={np.count_nonzero(matrices[mat_name])}")

# ---------- Step e: write OMX ----------
print("Step e: Writing crash_length_skim_507.omx...")
omx_path = os.path.join(OUT_DIR, 'crash_length_skim_507.omx')
write_omx(omx_path, zone_ids_507, matrices)

# ---------- Step f: summary CSV ----------
print("Step f: Building summary CSV...")
length_mat = matrices['length_km']
crash_mat = matrices['crash_total']
crash_per_km = matrices['crash_per_km']

# Per-zone stats
rows = []
for i, sa2 in enumerate(zone_ids_507):
    # Outbound = row i
    row_length = length_mat[i, :]
    row_crash = crash_mat[i, :]
    row_cpk = crash_per_km[i, :]

    nonzero_dest = np.sum(row_length > 0)
    avg_length = row_length[row_length > 0].mean() if nonzero_dest > 0 else 0.0
    total_crash = row_crash.sum()
    avg_cpk = row_cpk[row_cpk > 0].mean() if np.any(row_cpk > 0) else 0.0
    max_crash_idx = np.argmax(row_crash)
    max_crash_dest = zone_ids_507[max_crash_idx]
    max_crash_val = row_crash[max_crash_idx]

    rows.append({
        'sa2_code': sa2,
        'avg_length_to_all_dest': round(avg_length, 4),
        'total_expected_crashes_outbound': round(float(total_crash), 4),
        'crash_per_km_outbound': round(float(avg_cpk), 6),
        'max_crash_dest': max_crash_dest,
        'max_crash_value': round(float(max_crash_val), 4),
    })

summary_df = pd.DataFrame(rows)
summary_path = os.path.join(OUT_DIR, 'crash_length_skim_507_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"  Written: {summary_path} ({len(summary_df)} rows)")

# ---------- Step g: print stats ----------
total_pairs = len(df)
max_pairs = N * N
pct_coverage = 100.0 * total_pairs / max_pairs
total_crashes = float(matrices['crash_total'].sum())

print()
print("=== SCRIPT 10 SUMMARY ===")
print(f"  Total OD pairs in data: {total_pairs:,}")
print(f"  507×507 space:          {max_pairs:,}")
print(f"  Coverage:               {pct_coverage:.1f}%")
print(f"  Total expected crashes: {total_crashes:,.2f}")
print(f"  OMX file:               {omx_path}")
print(f"  Summary CSV:            {summary_path}")
print("Script 10 complete.")
