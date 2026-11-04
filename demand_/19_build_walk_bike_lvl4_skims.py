"""
19_build_walk_bike_lvl4_skims.py
Build Level4-origin x SA2-destination walk/bike distance skims using the
SEQ-STM model's native Level 4 micro-zone system (~7,000 zones) as the
fine-grain active-mode origin zone. This replaces the earlier SA1-based
approach (17_build_walk_bike_sa1_skims.py) with the SEQSTM-native zone system:

  Level4 zone (7,022 retained) -- EMME zone (1,738 EMME, via crosswalk)
  -> DISTWALK/DISTBIKE from seq_skims_base.omx (2,062 EMME x 2,062 EMME)
  -> aggregate to Level4 x 324 SA2 output

Size: 7,022 Level4 x 324 SA2 = 2.27M OD pairs x 2 matrices = ~18 MB compressed.

Output: demand_/walk_bike_lvl4_to_sa2_skims.h5
  sa2_codes : int64 (324) -- column SA2 codes
  lvl4_ids  : int64 (7022) -- row Level4 IDs
  DISTWALK  : float32 (7022, 324) -- km, Level4 origin to SA2 dest (mean of EMME zones in dest SA2)
  DISTBIKE  : float32 (7022, 324) -- km, same
"""
import os, re
import numpy as np
import pandas as pd
import h5py

DEMAND_DIR = r"Z:\test_runs_tours\demand_"
LVL4_XWALK = os.path.join(DEMAND_DIR, "lvl4_to_emme_crosswalk.csv")
SA2_EMME_XWALK = os.path.join(DEMAND_DIR, "sa2_crosswalk_507_expanded.csv")
SOURCE_SKIMS = (r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM"
                r"\#Model_Masters\ActivitySim\asim_model_inputs\seq_skims_base.omx")
ZONE_LIST = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
OUT_H5 = os.path.join(DEMAND_DIR, "walk_bike_lvl4_to_sa2_skims.h5")

MATRICES = ["DISTWALK", "DISTBIKE"]
N_EMME = 2062


def main():
    with open(ZONE_LIST) as f:
        sa2_codes = sorted(int(x) for x in re.split(r"[\s,]+", f.read().strip()) if x)
    sa2_to_col = {code: i for i, code in enumerate(sa2_codes)}
    n_sa2 = len(sa2_codes)

    xwalk = pd.read_csv(LVL4_XWALK)
    xwalk["emme_zone_id"] = xwalk["emme_zone_id"].astype(int)
    lvl4_ids = sorted(xwalk["LvL4_ID"].unique().tolist())
    lvl4_to_emme = dict(zip(xwalk["LvL4_ID"].astype(int), xwalk["emme_zone_id"].astype(int)))
    lvl4_to_row = {int(lid): i for i, lid in enumerate(lvl4_ids)}
    n_lvl4 = len(lvl4_ids)
    print(f"Level4 zones: {n_lvl4}  SA2 destinations: {n_sa2}")

    # EMME -> SA2 aggregation: each EMME zone -> which SA2 column index
    sa2_xwalk = pd.read_csv(SA2_EMME_XWALK)
    emme_to_sa2 = dict(zip(sa2_xwalk["emme_zone_id"].astype(int), sa2_xwalk["sa2_code"].astype(int)))
    emme_col = np.full(N_EMME + 1, -1, dtype=np.int32)
    for emme_id, sa2_code in emme_to_sa2.items():
        if 1 <= emme_id <= N_EMME:
            emme_col[emme_id] = sa2_to_col.get(sa2_code, -1)

    dest_emme = np.where(emme_col[1:] >= 0)[0]         # 0-based EMME row indices
    dest_sa2_cols = emme_col[1:][emme_col[1:] >= 0]     # corresponding SA2 col indices
    print(f"EMME zones mapping to study-area SA2: {len(dest_emme)}/{N_EMME}")

    # Per-destination-SA2: count of contributing EMME zones (for averaging)
    sa2_counts = np.zeros(n_sa2, dtype=np.int32)
    np.add.at(sa2_counts, dest_sa2_cols, 1)

    print(f"Reading source skims from {SOURCE_SKIMS}...")
    raw = {}
    with h5py.File(SOURCE_SKIMS, "r") as f:
        for name in MATRICES:
            raw[name] = np.array(f["data"][name], dtype=np.float32)
            nnz = np.count_nonzero(raw[name])
            print(f"  {name}: {raw[name].shape}  max={raw[name].max():.2f}  nnz={nnz:,}")

    print(f"\nBuilding {n_lvl4}x{n_sa2} Level4->SA2 walk/bike matrices...")
    lvl4_skims = {name: np.zeros((n_lvl4, n_sa2), dtype=np.float32) for name in MATRICES}

    for lvl4_id in lvl4_ids:
        row_i = lvl4_to_row[int(lvl4_id)]
        emme_id = lvl4_to_emme.get(int(lvl4_id))
        if emme_id is None or emme_id < 1 or emme_id > N_EMME:
            continue
        emme_row_0based = emme_id - 1

        for name in MATRICES:
            emme_row = raw[name][emme_row_0based]
            row_out = np.zeros(n_sa2, dtype=np.float64)
            np.add.at(row_out, dest_sa2_cols, emme_row[dest_emme])
            with np.errstate(invalid="ignore", divide="ignore"):
                row_out = np.where(sa2_counts > 0, row_out / sa2_counts, 0.0)
            lvl4_skims[name][row_i] = row_out.astype(np.float32)

        if row_i % 1000 == 0:
            print(f"  Level4 {row_i}/{n_lvl4} done")

    with h5py.File(OUT_H5, "w") as f:
        f.create_dataset("lvl4_ids", data=np.array(lvl4_ids, dtype=np.int64))
        f.create_dataset("sa2_codes", data=np.array(sa2_codes, dtype=np.int64))
        for name in MATRICES:
            ds = f.create_dataset(name, data=lvl4_skims[name], compression="lzf")
            ds.attrs["units"] = "km"
    fsize = os.path.getsize(OUT_H5) / (1024**2)
    print(f"\nWritten: {OUT_H5}  ({fsize:.1f} MB, {n_lvl4} Level4 x {n_sa2} SA2)")
    print(f"  DISTWALK: max={lvl4_skims['DISTWALK'].max():.2f} km  "
          f"mean={lvl4_skims['DISTWALK'][lvl4_skims['DISTWALK']>0].mean():.2f} km")


if __name__ == "__main__":
    main()
