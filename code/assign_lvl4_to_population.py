"""
assign_lvl4_to_population.py
Add HOME_LVL4 column to every zone_<sa2>.csv in new_pop/saved_data/zone_ids_sa2/.

Each person has an SA2 home zone (OTAZ = SA2 code). This assigns a Level 4
micro-zone (the SEQ-STM model's ~7,000-zone system) as a sub-zone, enabling
fine-grained walk/bike distance lookup from walk_bike_lvl4_to_sa2_skims.h5.

Sampling: uniform within the Level 4 zones that map to the person's home SA2.

Usage:
    python assign_lvl4_to_population.py
    NEW_POP_DIR=<path> python assign_lvl4_to_population.py
"""
import os
import re
import glob
import numpy as np
import pandas as pd

NEW_POP_DIR = os.environ.get("NEW_POP_DIR", r"Z:\test_runs_tours\new_pop")
ZONE_DIR    = os.path.join(NEW_POP_DIR, "saved_data", "zone_ids_sa2")
LVL4_XWALK = r"Z:\test_runs_tours\demand_\lvl4_to_emme_crosswalk.csv"
SA2_XWALK   = r"Z:\test_runs_tours\demand_\sa2_crosswalk_507.csv"


def main():
    xwalk = pd.read_csv(LVL4_XWALK)
    xwalk["sa2_code"] = pd.to_numeric(xwalk["sa2_code"], errors="coerce")
    xwalk = xwalk.dropna(subset=["sa2_code"])
    xwalk["sa2_code"] = xwalk["sa2_code"].astype(int)

    sa2_to_lvl4s = xwalk.groupby("sa2_code")["LvL4_ID"].apply(list).to_dict()
    print(f"SA2 zones with Level4 mapping: {len(sa2_to_lvl4s)}")

    zone_files = sorted(glob.glob(os.path.join(ZONE_DIR, "zone_*.csv")))
    print(f"Zone files to process: {len(zone_files)}")

    updated = skipped = 0
    for fp in zone_files:
        m = re.search(r"zone_(\d+)", os.path.basename(fp))
        if not m:
            continue
        sa2_code = int(m.group(1))
        lvl4_options = sa2_to_lvl4s.get(sa2_code)
        if not lvl4_options:
            skipped += 1
            continue

        df = pd.read_csv(fp, low_memory=False)
        if len(df) == 0:
            skipped += 1
            continue

        rng = np.random.default_rng(seed=sa2_code % (2**32))
        df["HOME_LVL4"] = rng.choice(
            lvl4_options, size=len(df), replace=True
        ).astype(np.int64)

        df.to_csv(fp, index=False)
        updated += 1

    print(f"Done. Updated {updated} files, skipped {skipped} (no Level4 mapping).")
    print("Re-run build_synth_population --new-pop to rebuild synth_population.csv.")


if __name__ == "__main__":
    main()
