"""Merge event_points_full.shard*.csv.gz shards (from the PBS array job)
into a single results/event_points_full.csv.gz."""
import argparse, glob, gzip, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/event_points_full.shard*.csv.gz")
    ap.add_argument("--out", default="results/event_points_full.csv.gz")
    args = ap.parse_args()
    shards = sorted(glob.glob(args.pattern))
    if not shards:
        raise SystemExit(f"no shards matching {args.pattern}")
    print(f"merging {len(shards)} shards -> {args.out}")
    n = 0
    with gzip.open(args.out, "wt", newline="") as fout:
        for i, s in enumerate(shards):
            with gzip.open(s, "rt") as fin:
                header = fin.readline()
                if i == 0:
                    fout.write(header)
                for line in fin:
                    fout.write(line); n += 1
    print(f"done: {n:,} event rows")


if __name__ == "__main__":
    main()
