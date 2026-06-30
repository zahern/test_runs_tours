"""
Script 04: Create OMX demand file with SA2 OD matrices and aggregated SEQSTM skims.
Uses h5py directly (openmatrix has a PyTables binary incompatibility).
"""
import os
import sys
import numpy as np
import pandas as pd
import h5py
import json

# ── Paths ──────────────────────────────────────────────────────────────────────
SEQSTM_SKIMS  = r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM\#Model_Masters\ActivitySim\asim_model_inputs\seq_skims_base.omx"
OUTPUT_DIR    = r"Z:\test_runs_tours\demand_"
SYNTH_OD_CSV  = os.path.join(OUTPUT_DIR, 'synth_pop_od_sa2.csv')
MAPPING_CSV   = os.path.join(OUTPUT_DIR, 'sa2_to_emme_mapping.csv')
SA2_REF_CSV   = os.path.join(OUTPUT_DIR, 'sa2_zones_reference.csv')
OUTPUT_OMX    = os.path.join(OUTPUT_DIR, 'synth_pop_demand.omx')

# Key skim matrices to aggregate (at most 20)
TARGET_SKIMS = [
    'DIST', 'DISTBIKE', 'DISTWALK',
    'SOVTOLL_TIME__AM', 'SOVTOLL_TIME__PM', 'SOVTOLL_TIME__MD',
    'SOVTOLL_DIST__AM', 'SOVTOLL_DIST__PM',
    'SOVTOLL_TOLL__AM', 'SOVTOLL_TOLL__PM',
    'WLK_LOC_WLK_TOTIVT__AM', 'WLK_LOC_WLK_TOTIVT__PM', 'WLK_LOC_WLK_TOTIVT__MD',
    'WLK_LOC_WLK_FAR__AM', 'WLK_LOC_WLK_FAR__PM',
    'WLK_LOC_WLK_WAIT__AM', 'WLK_LOC_WLK_WAIT__PM',
    'HOV2TOLL_TIME__AM', 'HOV2TOLL_TIME__PM',
    'HOV2TOLL_DIST__AM',
]

print("=" * 60)
print("Script 04: Build Demand OMX")
print("=" * 60)

# ── 1. Load inputs ────────────────────────────────────────────────────────────
print("\n[1] Loading synth pop OD ...")
synth_od = pd.read_csv(SYNTH_OD_CSV)
print(f"    OD rows: {len(synth_od):,}")

print("\n[2] Loading SA2 -> EMME mapping ...")
mapping = pd.read_csv(MAPPING_CSV)
sa2_ref = pd.read_csv(SA2_REF_CSV)
print(f"    Mapping rows: {len(mapping)}")
print(f"    SA2 reference rows: {len(sa2_ref)}")

# ── 3. Build SA2 zone index ───────────────────────────────────────────────────
print("\n[3] Building SA2 zone index ...")

# Unique SA2 codes from the synth OD (these are our output zones)
sa2_codes_od = sorted(set(
    synth_od['origin_sa2'].astype(str).unique().tolist() +
    synth_od['dest_sa2'].astype(str).unique().tolist()
))
print(f"    SA2 zones from OD: {len(sa2_codes_od)}")

# Also include mapped SA2s
sa2_codes_mapped = sorted(mapping['sa2_code'].dropna().astype(str).unique().tolist())
print(f"    SA2 zones from mapping: {len(sa2_codes_mapped)}")

# Use union, sorted
all_sa2_codes = sorted(set(sa2_codes_od) | set(sa2_codes_mapped))
print(f"    Total SA2 zones in OMX: {len(all_sa2_codes)}")

N = len(all_sa2_codes)
sa2_idx = {code: i for i, code in enumerate(all_sa2_codes)}

# ── 4. Build demand matrices ──────────────────────────────────────────────────
print("\n[4] Building demand matrices ...")
modes = ['car', 'pt', 'walk', 'bike', 'other']
demand_mats = {}

for mode in modes:
    mat = np.zeros((N, N), dtype=np.float32)
    mode_od = synth_od[synth_od['mode'] == mode]
    placed = 0
    skipped = 0
    for _, row in mode_od.iterrows():
        o = str(row['origin_sa2'])
        d = str(row['dest_sa2'])
        if o in sa2_idx and d in sa2_idx:
            mat[sa2_idx[o], sa2_idx[d]] += float(row['demand'])
            placed += 1
        else:
            skipped += 1
    demand_mats[mode] = mat
    total = mat.sum()
    print(f"    Mode '{mode}': {total:,.0f} trips placed ({placed} rows), {skipped} skipped")

# Total demand matrix
total_mat = sum(demand_mats.values())
print(f"    Total demand matrix sum: {total_mat.sum():,.0f}")

# ── 5. Build EMME zone -> SA2 index mapping ───────────────────────────────────
print("\n[5] Building EMME zone -> SA2 index lookup for skim aggregation ...")

# emme_zone_id -> sa2_code
emme_to_sa2 = mapping[mapping['sa2_code'].notna()].set_index('emme_zone_id')['sa2_code'].astype(str).to_dict()
N_EMME = 2062  # shape of SEQSTM skims

# For each EMME zone, get its SA2 index (or -1 if not mapped)
emme_sa2_idx = np.full(N_EMME + 1, -1, dtype=np.int32)  # 1-indexed EMME zones
for emme_z, sa2_code in emme_to_sa2.items():
    if sa2_code in sa2_idx and 1 <= int(emme_z) <= N_EMME:
        emme_sa2_idx[int(emme_z)] = sa2_idx[sa2_code]

mapped_emme = (emme_sa2_idx[1:N_EMME+1] >= 0).sum()
print(f"    EMME zones mapped to SA2: {mapped_emme} / {N_EMME}")

# ── 6. Aggregate SEQSTM skims to SA2 ─────────────────────────────────────────
print("\n[6] Aggregating SEQSTM skims to SA2 level ...")
print(f"    Source skims file: {SEQSTM_SKIMS}")

agg_skims = {}

with h5py.File(SEQSTM_SKIMS, 'r') as f:
    available_skims = list(f['data'].keys())
    print(f"    Available skim matrices: {len(available_skims)}")

    # Filter to target skims that actually exist
    to_process = [s for s in TARGET_SKIMS if s in available_skims]
    print(f"    Target skims to aggregate: {len(to_process)}")
    if len(to_process) < len(TARGET_SKIMS):
        missing_skims = [s for s in TARGET_SKIMS if s not in available_skims]
        print(f"    WARNING: Not found in OMX: {missing_skims}")

    # Precompute: which EMME zones (0-indexed) map to which SA2 index
    # emme_sa2_idx is 1-indexed; convert to 0-indexed arrays
    mapped_emme_0idx = np.array([z - 1 for z in range(1, N_EMME + 1) if emme_sa2_idx[z] >= 0])
    mapped_sa2_of_emme = np.array([emme_sa2_idx[z] for z in range(1, N_EMME + 1) if emme_sa2_idx[z] >= 0])
    n_mapped = len(mapped_emme_0idx)
    print(f"    Using vectorised aggregation: {n_mapped} mapped EMME zones")

    for i, skim_name in enumerate(to_process):
        print(f"    [{i+1}/{len(to_process)}] Aggregating {skim_name} ...", flush=True)
        raw = f['data'][skim_name][:]  # shape: (2062, 2062), float32/float64

        # Extract only the rows/cols corresponding to mapped EMME zones
        sub = raw[np.ix_(mapped_emme_0idx, mapped_emme_0idx)]  # (n_mapped, n_mapped)

        # For each pair (sa2_i, sa2_j), sum over sub-block and count
        agg_mat   = np.zeros((N, N), dtype=np.float64)
        count_mat = np.zeros((N, N), dtype=np.int32)

        # Vectorised scatter-add using numpy
        # mapped_sa2_of_emme[r] -> SA2 row index
        # mapped_sa2_of_emme[c] -> SA2 col index
        for r in range(n_mapped):
            sa2_r = mapped_sa2_of_emme[r]
            row_vals = sub[r, :]  # (n_mapped,)
            # Add to each (sa2_r, sa2_c) for all c
            np.add.at(agg_mat[sa2_r],   mapped_sa2_of_emme, row_vals)
            np.add.at(count_mat[sa2_r], mapped_sa2_of_emme, (row_vals != 0).astype(np.int32))

        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            agg_mat = np.where(count_mat > 0, agg_mat / count_mat, 0.0)

        agg_skims[f'AGG_{skim_name}'] = agg_mat.astype(np.float32)
        print(f"      Non-zero cells: {(agg_mat > 0).sum()}, max value: {agg_mat.max():.2f}")

print(f"    Aggregated {len(agg_skims)} skim matrices")

# ── 7. Write output OMX ───────────────────────────────────────────────────────
print(f"\n[7] Writing output OMX: {OUTPUT_OMX}")

# Write as HDF5 with OMX layout
with h5py.File(OUTPUT_OMX, 'w') as f:
    # OMX metadata
    f.attrs['OMX_VERSION'] = '0.2'
    f.attrs['SHAPE'] = np.array([N, N])

    # Data group
    data_grp = f.create_group('data')

    # Demand matrices
    for mode in modes:
        mat = demand_mats[mode]
        ds = data_grp.create_dataset(
            f'demand_{mode}',
            data=mat,
            compression='gzip',
            compression_opts=4
        )
        ds.attrs['_FillValue'] = 0.0
        print(f"    Written demand_{mode}: shape={mat.shape}, sum={mat.sum():.0f}")

    # Total demand
    ds = data_grp.create_dataset(
        'demand_total',
        data=total_mat.astype(np.float32),
        compression='gzip',
        compression_opts=4
    )
    ds.attrs['_FillValue'] = 0.0
    print(f"    Written demand_total: sum={total_mat.sum():.0f}")

    # Aggregated skim matrices
    for name, mat in agg_skims.items():
        ds = data_grp.create_dataset(
            name,
            data=mat,
            compression='gzip',
            compression_opts=4
        )
        ds.attrs['_FillValue'] = 0.0
        print(f"    Written {name}")

    # Lookup group (zone mapping)
    lookup_grp = f.create_group('lookup')
    zone_arr = np.array([c.encode('utf-8') for c in all_sa2_codes])
    lookup_grp.create_dataset('SA2_CODE', data=zone_arr)

print(f"\n    OMX written successfully.")
fsize = os.path.getsize(OUTPUT_OMX) / (1024 ** 2)
print(f"    File size: {fsize:.1f} MB")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCRIPT 04 COMPLETE")
print(f"  Output: {OUTPUT_OMX}")
print(f"  File size: {fsize:.1f} MB")
print(f"  Zones: {N} SA2 zones ({N}x{N} = {N*N:,} cells)")
print(f"  Demand matrices: demand_car, demand_pt, demand_walk, demand_bike, demand_other, demand_total")
print(f"  Skim matrices: {len(agg_skims)} aggregated from SEQSTM")
print("=" * 60)
