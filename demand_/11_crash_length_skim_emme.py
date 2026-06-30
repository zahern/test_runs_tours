"""
11_crash_length_skim_emme.py
Build length + crash skim OMX for the 2062 SEQSTM EMME post-demand zones.
"""
import numpy as np
import pandas as pd
import h5py
import sys
import os

OUT_DIR = r"Z:\test_runs_tours\demand_"
CROSSWALK_EXP = os.path.join(OUT_DIR, 'sa2_crosswalk_507_expanded.csv')
ZONES_REF = r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM\#Model_Masters\scripts\post_demand\zones_reference.csv"
SA2_OMX = os.path.join(OUT_DIR, 'crash_length_skim_507.omx')
ZONE_IDS_FILE = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"

N_EMME = 2062

def write_omx(path, zone_ids, matrices_dict):
    with h5py.File(path, 'w') as f:
        f.attrs['OMX_VERSION'] = '0.2'
        f.attrs['SHAPE'] = np.array([len(zone_ids), len(zone_ids)], dtype=np.int32)
        data_grp = f.create_group('data')
        for name, mat in matrices_dict.items():
            ds = data_grp.create_dataset(name, data=mat.astype(np.float32), compression='gzip')
            ds.attrs['_FillValue'] = np.float32(0.0)
        lkp_grp = f.create_group('lookup')
        lkp_grp.create_dataset('EMME_ZONE', data=np.array([str(z) for z in zone_ids], dtype='S12'))
    print(f"  Written: {path} ({len(zone_ids)} zones, {len(matrices_dict)} matrices)")

# ---------- Step a: load crosswalk expanded ----------
print("Step a: Loading sa2_crosswalk_507_expanded.csv...")
xwalk = pd.read_csv(CROSSWALK_EXP)
print(f"  Columns: {list(xwalk.columns)}")
print(f"  Shape: {xwalk.shape}")

# Build emme_zone_id -> sa2_code dict
# emme_zone_id is an int, sa2_code is a 9-digit int
emme_sa2_map = {}
for _, row in xwalk.iterrows():
    emme_sa2_map[int(row['emme_zone_id'])] = int(row['sa2_code'])

print(f"  EMME->SA2 entries: {len(emme_sa2_map)}")

# ---------- Step b: load zones reference for 2062 zone IDs ----------
print("Step b: Loading zones_reference.csv for EMME zone list...")
zref = pd.read_csv(ZONES_REF)
print(f"  Columns: {list(zref.columns)}")
emme_zone_ids = sorted(zref['emme_zone_id'].astype(int).tolist())
print(f"  {len(emme_zone_ids)} EMME zones, range {emme_zone_ids[0]}..{emme_zone_ids[-1]}")

# ---------- Step c: load SA2 507 OMX ----------
print("Step c: Loading crash_length_skim_507.omx...")
mat_names = ['length_km', 'crash_runoff', 'crash_headon', 'crash_rearend',
             'crash_intersection', 'crash_other', 'crash_total', 'crash_per_km']

sa2_matrices = {}
with h5py.File(SA2_OMX, 'r') as f:
    for name in mat_names:
        sa2_matrices[name] = np.array(f['data'][name], dtype=np.float32)
        print(f"  Loaded {name}: shape={sa2_matrices[name].shape}, sum={sa2_matrices[name].sum():.2f}")

# ---------- Step d: load sorted 507 SA2 codes for index mapping ----------
print("Step d: Building SA2 index map...")
with open(ZONE_IDS_FILE) as fh:
    zone_ids_507 = sorted([int(line.strip()) for line in fh if line.strip()])

sa2_index_map = {z: i for i, z in enumerate(zone_ids_507)}
print(f"  SA2 index map: {len(sa2_index_map)} entries")

# ---------- Step e: map EMME zones to SA2 ----------
print("Step e: Mapping 2062 EMME zones to SA2...")
# For each EMME zone 1..2062 (0-based index z, zone_id = z+1)
# find its SA2 code via emme_sa2_map, then its SA2 index via sa2_index_map
sa2_idx = np.full(N_EMME, -1, dtype=np.int32)
for z in range(N_EMME):
    zone_id = z + 1
    sa2_code = emme_sa2_map.get(zone_id)
    if sa2_code is not None:
        idx = sa2_index_map.get(sa2_code, -1)
        sa2_idx[z] = idx

mapped_count = np.sum(sa2_idx >= 0)
print(f"  Mapped: {mapped_count}/{N_EMME} EMME zones have valid SA2 index")
print(f"  Unmapped: {N_EMME - mapped_count} EMME zones")

# ---------- Step f: build 2062×2062 matrices ----------
print("Step f: Building 2062x2062 matrices (vectorised)...")
valid = sa2_idx >= 0
v_idx = sa2_idx[valid]       # SA2 indices for valid EMME zones
e_idx = np.where(valid)[0]   # EMME 0-based indices for valid zones

emme_matrices = {}
for mat_name, sa2_mat in sa2_matrices.items():
    emme_mat = np.zeros((N_EMME, N_EMME), dtype=np.float32)
    emme_mat[np.ix_(e_idx, e_idx)] = sa2_mat[np.ix_(v_idx, v_idx)]
    emme_matrices[mat_name] = emme_mat
    nz = np.count_nonzero(emme_mat)
    print(f"  {mat_name}: nonzero={nz:,}, sum={emme_mat.sum():.2f}")

# ---------- Step g: write OMX ----------
print("Step g: Writing crash_length_skim_emme2062.omx...")
omx_path = os.path.join(OUT_DIR, 'crash_length_skim_emme2062.omx')

# Use actual emme_zone_ids list (1..2062)
write_omx(omx_path, emme_zone_ids, emme_matrices)

file_size_mb = os.path.getsize(omx_path) / (1024**2)

# ---------- Step h: print coverage stats ----------
length_mat = emme_matrices['length_km']
nonzero_pairs = np.count_nonzero(length_mat)
total_pairs = N_EMME * N_EMME
pct_coverage = 100.0 * nonzero_pairs / total_pairs

print()
print("=== SCRIPT 11 SUMMARY ===")
print(f"  EMME zones:        {N_EMME}")
print(f"  Mapped EMME zones: {mapped_count}")
print(f"  Nonzero OD pairs:  {nonzero_pairs:,} / {total_pairs:,} ({pct_coverage:.1f}%)")
print(f"  File size:         {file_size_mb:.1f} MB")
print(f"  OMX file:          {omx_path}")
print("Script 11 complete.")
