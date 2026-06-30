"""
13_nearmiss_skim_matrix.py
Build near-miss density OMX matrices to add to the integrated demand OMX.
"""
import numpy as np
import pandas as pd
import h5py
import os

OUT_DIR = r"Z:\test_runs_tours\demand_"
ZONE_IDS_FILE = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
COUNTS_CSV = os.path.join(OUT_DIR, 'nearmiss_zonal_counts.csv')
NEARMISS_OMX = os.path.join(OUT_DIR, 'nearmiss_skim_507.omx')
INTEGRATED_OMX = os.path.join(OUT_DIR, 'demand_integrated_sa2.omx')

# ---------- Step a: load zone IDs ----------
print("Step a: Loading 507 zone IDs...")
with open(ZONE_IDS_FILE) as fh:
    zone_ids_507 = sorted([int(line.strip()) for line in fh if line.strip()])
N = len(zone_ids_507)
zone_index = {z: i for i, z in enumerate(zone_ids_507)}
print(f"  {N} zones")

# ---------- Step a: load nearmiss_zonal_counts.csv ----------
print("\nStep a: Loading nearmiss_zonal_counts.csv...")
counts_df = pd.read_csv(COUNTS_CSV)
print(f"  {len(counts_df):,} rows, columns: {list(counts_df.columns)}")

# ---------- Step b: build per-zone, per-period event count vectors ----------
print("\nStep b: Building zonal count vectors per period...")
periods = ['am', 'op', 'pm', 'ev']

# Aggregate event_count by SA2_CODE x time_period
# Only include rows with valid SA2 codes (not UNMATCHED)
valid_counts = counts_df[counts_df['SA2_CODE'] != 'UNMATCHED'].copy()

# Convert SA2_CODE to int where possible
valid_counts['SA2_CODE_INT'] = pd.to_numeric(valid_counts['SA2_CODE'], errors='coerce')
valid_counts = valid_counts.dropna(subset=['SA2_CODE_INT'])
valid_counts['SA2_CODE_INT'] = valid_counts['SA2_CODE_INT'].astype(int)

# Sum event_count per SA2 x period
zone_period = valid_counts.groupby(['SA2_CODE_INT', 'time_period'])['event_count'].sum().reset_index()

# Build count vector per period: array of N values (index = zone position)
count_vectors = {}
for p in periods:
    vec = np.zeros(N, dtype=np.float32)
    sub = zone_period[zone_period['time_period'] == p]
    for _, row in sub.iterrows():
        sa2 = int(row['SA2_CODE_INT'])
        if sa2 in zone_index:
            vec[zone_index[sa2]] = row['event_count']
    count_vectors[p] = vec
    nonzero = np.count_nonzero(vec)
    print(f"  {p}: total={vec.sum():.0f}, zones_with_events={nonzero}")

# ---------- Step c: build matrices ----------
print("\nStep c: Building destination-risk, origin-risk and combined matrices...")
matrices = {}

for p in periods:
    vec = count_vectors[p]

    # Destination risk: Matrix[i,j] = count in dest zone j
    dest_mat = np.tile(vec, (N, 1)).astype(np.float32)  # shape (N, N), each row = vec

    # Origin risk: Matrix[i,j] = count in origin zone i
    orig_mat = np.tile(vec[:, np.newaxis], (1, N)).astype(np.float32)  # each col = vec

    # Combined: average of origin and destination risk
    combined_mat = ((orig_mat + dest_mat) / 2.0).astype(np.float32)

    matrices[f'nearmiss_dest_{p}'] = dest_mat
    matrices[f'nearmiss_origin_{p}'] = orig_mat
    matrices[f'nearmiss_total_{p}'] = combined_mat

    print(f"  {p}: dest mean={dest_mat.mean():.4f}, max={dest_mat.max():.1f}, "
          f"origin mean={orig_mat.mean():.4f}, combined mean={combined_mat.mean():.4f}")

# ---------- Step c: write nearmiss_skim_507.omx ----------
print("\nStep c: Writing nearmiss_skim_507.omx...")
with h5py.File(NEARMISS_OMX, 'w') as f:
    f.attrs['OMX_VERSION'] = '0.2'
    f.attrs['SHAPE'] = np.array([N, N], dtype=np.int32)
    data_grp = f.create_group('data')
    for name, mat in matrices.items():
        ds = data_grp.create_dataset(name, data=mat, compression='gzip')
        ds.attrs['_FillValue'] = np.float32(0.0)
    lkp_grp = f.create_group('lookup')
    lkp_grp.create_dataset('SA2_CODE', data=np.array([str(z) for z in zone_ids_507], dtype='S12'))

print(f"  Written: {NEARMISS_OMX} ({N} zones, {len(matrices)} matrices)")

# ---------- Step d: append to demand_integrated_sa2.omx ----------
print("\nStep d: Appending matrices to demand_integrated_sa2.omx...")
if os.path.exists(INTEGRATED_OMX):
    with h5py.File(INTEGRATED_OMX, 'a') as f:
        data_grp = f['data'] if 'data' in f else f.create_group('data')
        existing = list(data_grp.keys())
        print(f"  Existing matrices: {existing}")
        added = []
        skipped = []
        for name, mat in matrices.items():
            if name in data_grp:
                del data_grp[name]
                print(f"  Replacing existing matrix: {name}")
            ds = data_grp.create_dataset(name, data=mat, compression='gzip')
            ds.attrs['_FillValue'] = np.float32(0.0)
            added.append(name)
        print(f"  Added/replaced {len(added)} matrices to {INTEGRATED_OMX}")
else:
    print(f"  WARNING: {INTEGRATED_OMX} not found, skipping append step.")

# ---------- Step e: print stats ----------
print()
print("=== SCRIPT 13 SUMMARY ===")
print(f"  Zone count (N): {N}")
print(f"  Matrices built: {len(matrices)}")
print()
print("  Matrix stats per period:")
for p in periods:
    dest = matrices[f'nearmiss_dest_{p}']
    orig = matrices[f'nearmiss_origin_{p}']
    comb = matrices[f'nearmiss_total_{p}']
    vec = count_vectors[p]
    print(f"  {p.upper()}: total_events={vec.sum():.0f}, "
          f"dest_mean={dest.mean():.4f}, dest_max={dest.max():.1f}, "
          f"combined_mean={comb.mean():.4f}")
print()
print(f"  nearmiss_skim_507.omx: {os.path.getsize(NEARMISS_OMX)/1024:.1f} KB")
print("Script 13 complete.")
