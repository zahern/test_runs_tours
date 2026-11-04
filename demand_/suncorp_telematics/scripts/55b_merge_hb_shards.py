"""
55b_merge_hb_shards.py
========================================================================
Merge per-shard outputs of 55_rebuild_hb_full.py into the single
full-period file data/hard_braking_events_full.csv, then validate it.

Validation checks (fail loudly if violated):
  * every EventStart within the raw window (<= 2026-04-02, allowing a
    trip that starts 2026-03-31 to run past midnight -- cf.
    results/event_points_full.csv.gz max dt 2026-04-01 00:12:53)
  * Category values are exactly the 3 expected labels
  * PeakDecel_g respects the [0.20/0.50/0.75) category bounds
  * monthly counts are roughly uniform (flags a March-only regression:
    no single month may hold > 40% of rows)

The legacy data/hard_braking_events.csv is left untouched; downstream
(18_prepare_eb_dataset.py) prefers the _full file when present.

Usage:
  python scripts/55b_merge_hb_shards.py \
      --pattern "results/hb_full.shard*.csv.gz" \
      --out data/hard_braking_events_full.csv
"""
import argparse
import glob
import os

import pandas as pd

BOUNDS = [("Hard-Braking Candidate", 0.20, 0.50),
          ("Critical-Likely", 0.50, 0.75),
          ("Extreme", 0.75, 1e9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/hb_full.shard*.csv.gz")
    ap.add_argument("--out", default="data/hard_braking_events_full.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"[55b] FATAL: no shards match '{args.pattern}'. "
                         f"Run pbs_55_rebuild_hb.pbs first.")
    print(f"[55b] merging {len(files)} shards ...", flush=True)
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f))
        except EOFError:
            raise SystemExit(
                f"[55b] FATAL: truncated gzip shard: {f}. That shard's "
                f"worker died mid-write (e.g. segfault exit 139). Delete its "
                f"'{f}.done_files' marker and re-run that shard, then "
                f"re-submit the merge.")
    df = pd.concat(parts, ignore_index=True)
    print(f"[55b] rows before dedup: {len(df):,}", flush=True)
    df = df.drop_duplicates()
    print(f"[55b] rows after  dedup: {len(df):,}", flush=True)
    df = df.sort_values("EventStart").reset_index(drop=True)

    # ---- validation ----
    cats = set(df["Category"].unique())
    assert cats <= {"Hard-Braking Candidate", "Critical-Likely",
                    "Extreme"}, f"unexpected categories: {cats}"
    for cat, lo, hi in BOUNDS:
        sub = df.loc[df["Category"] == cat, "PeakDecel_g"]
        assert (sub >= lo - 1e-9).all() and (sub < hi).all(), \
            f"PeakDecel out of bounds for {cat}"
    mx = df["EventStart"].max()
    assert mx <= "2026-04-02", f"events beyond raw window: max {mx}"
    ym = df["EventStart"].str.slice(0, 7).value_counts(normalize=True)
    assert ym.max() <= 0.40, \
        f"temporal skew: month {ym.idxmax()} holds {ym.max():.1%}"
    print("[55b] validation OK", flush=True)
    print("[55b] monthly share:\n" +
          df["EventStart"].str.slice(0, 7).value_counts()
          .sort_index().to_string(), flush=True)
    print("[55b] by category:\n" +
          df["Category"].value_counts().to_string(), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[55b] wrote {args.out} ({len(df):,} rows)", flush=True)


if __name__ == "__main__":
    main()
