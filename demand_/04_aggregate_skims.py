"""
04_aggregate_skims.py
Aggregate SEQSTM skims (2062x2062) to SA2 level (507x507).
Uses h5py directly (avoids openmatrix/tables binary incompatibility).
"""
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, r"Z:\test_runs_tours\demand_")
from omx_utils import write_omx

OUT_DIR = r"Z:\test_runs_tours\demand_"
SKIMS_PATH = r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM\#Model_Masters\ActivitySim\asim_model_inputs\seq_skims_base.omx"

# --- Load zone list ---
with open(r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt") as f:
    content = f.read()
zone_ids_507 = sorted([int(x) for x in re.split(r'[\s,]+', content.strip()) if x])
n_sa2 = len(zone_ids_507)
sa2_to_idx = {code: i for i, code in enumerate(zone_ids_507)}
print(f"Loaded {n_sa2} SA2 zones")

# --- a) Load expanded crosswalk: emme_zone_id -> sa2_code ---
expanded = pd.read_csv(os.path.join(OUT_DIR, 'sa2_crosswalk_507_expanded.csv'))
expanded['sa2_code'] = expanded['sa2_code'].astype(int)
expanded['emme_zone_id'] = expanded['emme_zone_id'].astype(int)
print(f"Expanded crosswalk: {len(expanded)} rows")
print(f"Unique EMME zones: {expanded['emme_zone_id'].nunique()}")

# --- b/c) Build aggregation index array ---
n_emme = 2062
agg_idx = np.full(n_emme, -1, dtype=np.int32)
emme_to_sa2 = dict(zip(expanded['emme_zone_id'].astype(int), expanded['sa2_code'].astype(int)))

for emme_id, sa2_code in emme_to_sa2.items():
    if 1 <= emme_id <= n_emme:
        sa2_i = sa2_to_idx.get(sa2_code, -1)
        agg_idx[emme_id - 1] = sa2_i  # 1-based emme_id -> 0-based array index

mapped_emme = np.where(agg_idx >= 0)[0]
n_mapped = len(mapped_emme)
print(f"Mapped EMME zones: {n_mapped}/{n_emme}")
print(f"SA2 zones with EMME coverage: {len(set(agg_idx[mapped_emme]))}/{n_sa2}")

# --- d) Priority matrices to process ---
PRIORITY_MATRICES = [
    'SOVTOLL_TIME__AM', 'SOVTOLL_TIME__MD', 'SOVTOLL_TIME__PM', 'SOVTOLL_TIME__EA', 'SOVTOLL_TIME__EV',
    'SOVTOLL_DIST__AM', 'SOVTOLL_DIST__MD', 'SOVTOLL_DIST__PM',
    'SOVTOLL_TOLL__AM', 'SOVTOLL_TOLL__PM',
    'HOV2TOLL_TIME__AM', 'HOV2TOLL_TIME__PM',
    'DRV_LOC_WLK_TOTIVT__AM', 'DRV_LOC_WLK_TOTIVT__PM',
    'DRV_LOC_WLK_FAR__AM', 'DRV_LOC_WLK_FAR__PM',
    'WLK_LOC_DRV_TOTIVT__AM', 'WLK_LOC_DRV_TOTIVT__PM',
    'DIST', 'DISTBIKE', 'DISTWALK',
]

with h5py.File(SKIMS_PATH, 'r') as f:
    available_set = set(f['data'].keys())
    print(f"Total matrices in OMX: {len(available_set)}")

matrices_to_process = [m for m in PRIORITY_MATRICES if m in available_set]
missing_mats = [m for m in PRIORITY_MATRICES if m not in available_set]
print(f"Priority matrices found: {len(matrices_to_process)}/{len(PRIORITY_MATRICES)}")
if missing_mats:
    print(f"Missing: {missing_mats}")

# --- Nearest-neighbor imputation setup ---
crosswalk = pd.read_csv(os.path.join(OUT_DIR, 'sa2_crosswalk_507.csv'))
crosswalk = crosswalk.sort_values('sa2_code').reset_index(drop=True)
has_emme = np.array([crosswalk.iloc[i]['n_emme_zones'] > 0 for i in range(n_sa2)])
print(f"SA2 zones with direct EMME coverage: {has_emme.sum()}/{n_sa2}")

coords = crosswalk[['sa2_lat', 'sa2_lon']].values.astype(float)
unmapped_sa2 = np.where(~has_emme)[0]
mapped_sa2_zones = np.where(has_emme)[0]

nearest_for_unmapped = {}
if len(unmapped_sa2) > 0 and len(mapped_sa2_zones) > 0:
    for u_idx in unmapped_sa2:
        if np.isnan(coords[u_idx, 0]):
            nearest_for_unmapped[u_idx] = mapped_sa2_zones[0]
            continue
        dists = np.sqrt(
            (coords[mapped_sa2_zones, 0] - coords[u_idx, 0])**2 +
            (coords[mapped_sa2_zones, 1] - coords[u_idx, 1])**2
        )
        nearest_for_unmapped[u_idx] = mapped_sa2_zones[np.argmin(dists)]

print(f"Unmapped SA2 zones (will use nearest neighbor): {len(unmapped_sa2)}")

# --- e) Aggregate matrices ---
print(f"\nAggregating {len(matrices_to_process)} matrices...")
aggregated = {}
src_idx = agg_idx[mapped_emme]  # SA2 indices for mapped EMME rows/cols (shape: n_mapped)

# Pre-build outer index arrays for vectorised add.at
row_idx = src_idx[:, None]  # (n_mapped, 1)
col_idx = src_idx[None, :]  # (1, n_mapped)

coverage_pct = None

with h5py.File(SKIMS_PATH, 'r') as f:
    for mat_name in matrices_to_process:
        t0 = time.time()
        mat = np.array(f['data'][mat_name], dtype=np.float32)

        if mat.shape != (n_emme, n_emme):
            print(f"  WARNING: {mat_name} shape {mat.shape} unexpected, skipping")
            continue

        sa2_mat = np.zeros((n_sa2, n_sa2), dtype=np.float64)
        counts   = np.zeros((n_sa2, n_sa2), dtype=np.int32)

        sub_mat = mat[np.ix_(mapped_emme, mapped_emme)]

        np.add.at(sa2_mat, (row_idx, col_idx), sub_mat)
        np.add.at(counts,  (row_idx, col_idx), 1)

        # Compute mean
        with np.errstate(invalid='ignore', divide='ignore'):
            sa2_mat = np.where(counts > 0, sa2_mat / counts, np.nan)

        # Coverage for first matrix
        if coverage_pct is None:
            n_valid = int(np.sum(counts > 0))
            coverage_pct = 100.0 * n_valid / (n_sa2 * n_sa2)

        # Nearest-neighbor imputation for unmapped SA2 zones
        for u_idx, nn_idx in nearest_for_unmapped.items():
            sa2_mat[u_idx, :] = sa2_mat[nn_idx, :]
            sa2_mat[:, u_idx] = sa2_mat[:, nn_idx]
            sa2_mat[u_idx, u_idx] = sa2_mat[nn_idx, nn_idx]

        sa2_mat = np.where(np.isnan(sa2_mat), 0.0, sa2_mat).astype(np.float32)
        aggregated[mat_name] = sa2_mat

        elapsed = time.time() - t0
        nz = int(np.count_nonzero(sa2_mat))
        print(f"  {mat_name}: {elapsed:.1f}s, non-zero={nz:,}, max={sa2_mat.max():.2f}")

# --- f) Save OMX ---
out_omx = os.path.join(OUT_DIR, 'sa2_skims_507.omx')
if os.path.exists(out_omx):
    os.remove(out_omx)

zone_mapping = {str(code): i for i, code in enumerate(zone_ids_507)}
print(f"\nWriting {out_omx}...")
write_omx(out_omx, aggregated, zone_mapping=zone_mapping, shape=(n_sa2, n_sa2))
fsize = os.path.getsize(out_omx) / (1024**2)
print(f"Saved ({fsize:.1f} MB)")

# --- g) Print summary ---
print(f"\n=== SKIM AGGREGATION SUMMARY ===")
print(f"Matrices processed: {len(aggregated)}")
print(f"Zone dimensions: {n_sa2} x {n_sa2}")
print(f"EMME zones mapped: {n_mapped}/{n_emme} ({100*n_mapped/n_emme:.1f}%)")
print(f"SA2 zones with EMME coverage: {has_emme.sum()}/{n_sa2} ({100*has_emme.sum()/n_sa2:.1f}%)")
print(f"SA2 OD pair coverage (first matrix): {coverage_pct:.1f}%")
print(f"Matrices saved: {list(aggregated.keys())}")
