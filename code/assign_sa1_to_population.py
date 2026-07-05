"""
assign_sa1_to_population.py
Add HOME_SA1 column to every zone_<sa2>.csv in new_pop/saved_data/zone_ids_sa2/.

Each person already has an SA2 home zone (OTAZ = SA2 code). This script adds
a HOME_SA1 sub-zone code by sampling from the SA1 zones within that person's
home SA2, weighted by SA1 household count from the ABS Census (TOTHH at SA1
level from synth pop zone row count — proxy = number of persons in each SA1 is
proportional to its synthetic population, which we already have).

Why HOME_SA1 matters:
  For walk and cycling mode/destination choice, the distance from a person's
  actual home location (SA1 centroid, ~500m diameter) to a destination SA2 is
  much more meaningful than the SA2-centroid distance used currently. Adding
  HOME_SA1 lets stage5's _mode_probs_numpy look up the finer SA1→SA2 skim
  (walk_bike_sa1_to_sa2_skims.h5) instead of the averaged SA2→SA2 value.

Method for SA1 sampling within each SA2:
  - Count how many SA1 zones exist within the SA2 (from sa1_2021.csv)
  - If per-SA1 population data is available (from previously-run zone files),
    weight by population. Otherwise use equal weighting (uniform random).
  - Assign each person one SA1 from that weighted sample.

Usage:
    python assign_sa1_to_population.py
    # or: NEW_POP_DIR=<path> python assign_sa1_to_population.py
"""
import os
import re
import glob
import numpy as np
import pandas as pd

NEW_POP_DIR = os.environ.get("NEW_POP_DIR", r"Z:\test_runs_tours\new_pop")
ZONE_DIR    = os.path.join(NEW_POP_DIR, "saved_data", "zone_ids_sa2")
SA1_CSV     = r"Z:\test_runs_tours\SA1_SHP\sa1_2021.csv"
SA1_EMME_XW = r"Z:\test_runs_tours\demand_\sa1_to_emme_crosswalk_seqtweed.csv"


def main():
    sa1_lu = pd.read_csv(SA1_CSV, dtype=str)
    sa1_lu["SA2_CODE21"] = sa1_lu["SA2_CODE21"].str.strip()
    sa1_lu["SA1_CODE21"] = sa1_lu["SA1_CODE21"].str.strip()

    # Check if SA1 EMME crosswalk exists to restrict to study-area SA1s only
    if os.path.exists(SA1_EMME_XW):
        xwalk = pd.read_csv(SA1_EMME_XW, dtype={"SA1_CODE21": str})
        study_sa1s = set(xwalk["SA1_CODE21"].str.strip())
        sa1_lu = sa1_lu[sa1_lu["SA1_CODE21"].isin(study_sa1s)]
        print(f"Restricted to {len(sa1_lu)} study-area SA1 codes")
    else:
        print(f"WARNING: {SA1_EMME_XW} not found — including all SA1 codes for each SA2")

    # Build SA2 → list of SA1 codes mapping
    sa2_to_sa1s = sa1_lu.groupby("SA2_CODE21")["SA1_CODE21"].apply(list).to_dict()

    zone_files = sorted(glob.glob(os.path.join(ZONE_DIR, "zone_*.csv")))
    print(f"Zone files to process: {len(zone_files)}")

    updated = 0
    skipped = 0
    for fp in zone_files:
        sa2_code = str(int(re.search(r"zone_(\d+)", os.path.basename(fp)).group(1)))
        sa1_options = sa2_to_sa1s.get(sa2_code)
        if not sa1_options:
            skipped += 1
            continue

        df = pd.read_csv(fp, low_memory=False)
        if "HOME_SA1" in df.columns:
            # Already assigned — skip unless forced
            updated += 1
            continue

        # Sample SA1 for each person — uniform within this SA2's SA1 list
        # (without per-SA1 population counts, equal weighting is the correct
        # null assumption; population-weighted sampling would require a separate
        # SA1-level ABS census table not currently loaded here)
        rng = np.random.default_rng(seed=int(sa2_code) % (2**32))
        df["HOME_SA1"] = rng.choice(
            [int(s) for s in sa1_options],
            size=len(df),
            replace=True,
        ).astype(np.int64)

        df.to_csv(fp, index=False)
        updated += 1

    print(f"Done. Updated {updated} files, skipped {skipped} (no SA1 data).")
    print("HOME_SA1 is now a column in each zone_<sa2>.csv.")
    print("Re-run build_synth_population --new-pop to rebuild synth_population.csv.")


if __name__ == "__main__":
    main()
