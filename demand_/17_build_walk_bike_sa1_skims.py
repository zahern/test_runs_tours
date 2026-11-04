"""
17_build_walk_bike_sa1_skims.py
Build SA1-origin × SA2-destination walk and bike distance skims, providing
fine-grained active-mode LOS without the 175M-cell cost of a full SA1×SA1
matrix. Size: 13,241 SA1 rows × 529 SA2 cols ≈ 7M OD pairs per matrix.

Method:
  For each SA1 in the study area:
    1. Find the SA1's mapped EMME zone (from sa1_to_emme_crosswalk_seqtweed.csv)
    2. Extract that EMME zone's row from the 2,062×2,062 DISTWALK/DISTBIKE matrix
    3. Map each destination EMME zone → destination SA2 (mean across EMME zones
       in the same SA2) using sa2_crosswalk_507_expanded.csv
    4. Result: one DISTWALK and DISTBIKE value per (SA1, dest_SA2) pair

Why this architecture:
  - Walk trips are typically <1.5 km: a within-SA2 origin error of 1-3 km from
    SA2-centroid averaging often flips walk from viable to impossible.
  - Bike trips: <8 km; same issue but at larger scale.
  - Auto/transit: SA2→SA2 is fine (relative error small vs trip length).

Output: walk_bike_sa1_to_sa2_skims.h5  (HDF5, faster to load than CSV)
  Dataset 'DISTWALK':  float32 array (n_sa1 × n_sa2)
  Dataset 'DISTBIKE':  float32 array (n_sa1 × n_sa2)
  Dataset 'sa1_codes': int64 array (n_sa1)   — index into rows
  Dataset 'sa2_codes': int64 array (n_sa2)   — index into cols

Typical file size: ~55 MB (2 matrices × 7M float32 cells × 4 bytes).
"""
import os
import re
import numpy as np
import pandas as pd
import h5py

DEMAND_DIR = r"Z:\test_runs_tours\demand_"
SA1_EMME_XWALK = os.path.join(DEMAND_DIR, "sa1_to_emme_crosswalk_seqtweed.csv")
SA2_EMME_XWALK = os.path.join(DEMAND_DIR, "sa2_crosswalk_507_expanded.csv")
SOURCE_SKIMS   = (r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM"
                   r"\#Model_Masters\ActivitySim\asim_model_inputs\seq_skims_base.omx")
ZONE_LIST      = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
OUT_H5         = os.path.join(DEMAND_DIR, "walk_bike_sa1_to_sa2_skims.h5")

MATRICES = ["DISTWALK", "DISTBIKE"]

# EMME zones are 1-indexed (numpy_index_plus_one), so EMME zone ID N is row N-1
N_EMME = 2062


def main():
    # ── Load zone lists ─────────────────────────────────────────────────────
    with open(ZONE_LIST) as f:
        sa2_codes = sorted(int(x) for x in re.split(r"[\s,]+", f.read().strip()) if x)
    sa2_to_idx = {code: i for i, code in enumerate(sa2_codes)}
    n_sa2 = len(sa2_codes)
    print(f"Destination SA2 zones: {n_sa2}")

    # ── SA1→EMME mapping ────────────────────────────────────────────────────
    sa1_xwalk = pd.read_csv(SA1_EMME_XWALK, dtype={"SA1_CODE21": str, "SA2_CODE21": str})
    sa1_xwalk["SA1_CODE21"] = sa1_xwalk["SA1_CODE21"].str.strip()
    sa1_xwalk = sa1_xwalk.dropna(subset=["emme_zone_id"]).copy()
    sa1_xwalk["emme_zone_id"] = sa1_xwalk["emme_zone_id"].astype(int)
    sa1_codes = sorted(sa1_xwalk["SA1_CODE21"].astype(np.int64).unique().tolist())
    sa1_to_emme = dict(zip(sa1_xwalk["SA1_CODE21"].astype(np.int64), sa1_xwalk["emme_zone_id"].astype(int)))
    n_sa1 = len(sa1_codes)
    sa1_to_row = {int(code): i for i, code in enumerate(sa1_codes)}
    print(f"SA1 zones to process: {n_sa1}")

    # ── EMME→SA2 aggregation index ──────────────────────────────────────────
    sa2_xwalk = pd.read_csv(SA2_EMME_XWALK)
    sa2_xwalk["sa2_code"] = sa2_xwalk["sa2_code"].astype(int)
    sa2_xwalk["emme_zone_id"] = sa2_xwalk["emme_zone_id"].astype(int)
    emme_to_sa2 = dict(zip(sa2_xwalk["emme_zone_id"], sa2_xwalk["sa2_code"]))

    # Map each EMME zone (1..2062) → SA2 output column index (-1 = not in study area)
    emme_col = np.full(N_EMME + 1, -1, dtype=np.int32)  # index 0 unused (EMME is 1-based)
    for emme_id, sa2_code in emme_to_sa2.items():
        if 1 <= emme_id <= N_EMME:
            col = sa2_to_idx.get(sa2_code, -1)
            emme_col[emme_id] = col

    dest_cols = np.arange(1, N_EMME + 1)[emme_col[1:] >= 0]   # EMME zones that land in study area
    dest_sa2s = emme_col[1:][emme_col[1:] >= 0]                 # their SA2 output column indices
    print(f"EMME zones mapping to study-area SA2: {len(dest_cols)}/{N_EMME}")

    # ── Read source matrices ─────────────────────────────────────────────────
    print(f"Reading source skims from {SOURCE_SKIMS}...")
    raw = {}
    with h5py.File(SOURCE_SKIMS, "r") as f:
        for name in MATRICES:
            raw[name] = np.array(f["data"][name], dtype=np.float32)   # (2062, 2062)
            print(f"  {name}: shape={raw[name].shape}  max={raw[name].max():.2f}  "
                  f"nonzero={np.count_nonzero(raw[name]):,}")

    # ── Build SA1→SA2 matrices ───────────────────────────────────────────────
    # For each SA1: pick its EMME zone's row; average-aggregate columns to SA2.
    # To average: for each SA2 destination col, sum contributions from all
    # EMME source cols that map to that SA2, divide by count.
    print(f"\nBuilding {n_sa1}x{n_sa2} SA1->SA2 walk/bike matrices...")
    sa1_skims = {name: np.zeros((n_sa1, n_sa2), dtype=np.float32) for name in MATRICES}
    sa2_counts = np.zeros(n_sa2, dtype=np.int32)
    np.add.at(sa2_counts, dest_sa2s, 1)

    for sa1_code in sa1_codes:
        row_i = sa1_to_row[int(sa1_code)]
        emme_id = sa1_to_emme.get(int(sa1_code))
        if emme_id is None or emme_id < 1 or emme_id > N_EMME:
            continue
        emme_row_idx = emme_id - 1   # 0-based

        for name in MATRICES:
            emme_row = raw[name][emme_row_idx]   # (2062,) distances from this EMME zone
            # Accumulate into SA2 destination columns
            row_out = np.zeros(n_sa2, dtype=np.float64)
            np.add.at(row_out, dest_sa2s, emme_row[dest_cols - 1])
            # Average (divide by number of EMME source zones per dest SA2)
            with np.errstate(invalid="ignore", divide="ignore"):
                row_out = np.where(sa2_counts > 0, row_out / sa2_counts, 0.0)
            sa1_skims[name][row_i] = row_out.astype(np.float32)

        if row_i % 1000 == 0:
            print(f"  SA1 {row_i}/{n_sa1} done")

    # ── Save ────────────────────────────────────────────────────────────────
    with h5py.File(OUT_H5, "w") as f:
        f.create_dataset("sa1_codes", data=np.array(sa1_codes, dtype=np.int64))  # 11-digit SA1 codes
        f.create_dataset("sa2_codes", data=np.array(sa2_codes, dtype=np.int64))
        for name in MATRICES:
            ds = f.create_dataset(name, data=sa1_skims[name], compression="lzf")
            ds.attrs["units"] = "km"
            ds.attrs["description"] = f"{name} from SA1 origin to SA2 destination, averaged over EMME zones"

    fsize = os.path.getsize(OUT_H5) / (1024**2)
    print(f"\nWritten: {OUT_H5}  ({fsize:.1f} MB, {n_sa1} SA1 × {n_sa2} SA2)")
    print(f"  DISTWALK: max={sa1_skims['DISTWALK'].max():.2f} km  "
          f"mean={sa1_skims['DISTWALK'][sa1_skims['DISTWALK']>0].mean():.2f} km")
    print(f"  DISTBIKE: max={sa1_skims['DISTBIKE'].max():.2f} km  "
          f"mean={sa1_skims['DISTBIKE'][sa1_skims['DISTBIKE']>0].mean():.2f} km")


if __name__ == "__main__":
    main()
