"""
50_extract_raw_events.py
========================================================================
Re-extract per-event telematics from the RAW trip files for the FULL
2025-01 -> 2026-03 period (not just the March-2026 window the original
02_master_scan.py hard-coded).  This is what unlocks every real Suncorp
fixture instead of only the 6 March-2026 games.

Each raw file (data/raw_near_miss_2026/trips-*.csv.gz) holds one row per
trip with delimited trajectory paths.  We explode SnappedPath /
TimestampPath / SpeedPath_kph / XAccPath / YAccPath into per-point rows,
flag kinematic events by the same thresholds as the original pipeline,
keep only points inside the Suncorp study bbox, and stream the event rows
to a single compressed output.

Event thresholds (accelerometer in g), matching 02_master_scan.py:
    brake   : XAcc <= -0.30
    accel   : XAcc >= +0.30
    corner  : |YAcc| >= 0.35
    extreme : Speed >= SPEED_EXTREME (kph)
    mod     : XAcc <= -0.20 or XAcc >= 0.20 or |YAcc| >= 0.30  (moderate)

Output: results/event_points_full.csv.gz
    columns: dt, SnappedLongitude, SnappedLatitude, Speed, XAcc, YAcc,
             brake, accel, corner, extreme, mod, VehicleType

Checkpointed: writes a progress marker every CKPT files so a re-launched
PBS job resumes instead of restarting.  Designed for HPC (streams, low mem).
"""
import argparse, glob, gzip, os, sys, time
import numpy as np
import pandas as pd

BBOX = dict(lon_min=152.98, lon_max=153.05, lat_min=-27.49, lat_max=-27.44)
TH_BRAKE, TH_ACCEL, TH_CORNER = -0.30, 0.30, 0.35
TH_BRAKE_M, TH_ACCEL_M, TH_CORNER_M = -0.20, 0.20, 0.30
SPEED_EXTREME = 110.0  # kph

OUT_COLS = ["dt", "SnappedLongitude", "SnappedLatitude", "Speed", "XAcc", "YAcc",
            "brake", "accel", "corner", "extreme", "mod", "VehicleType"]


def parse_trip(sp, tp, spd, xa, ya, vtype):
    """Explode one trip's aligned paths into arrays; returns per-point lists
    filtered to the study bbox, or None."""
    if not isinstance(sp, str) or not isinstance(tp, str):
        return None
    pts = sp.split(","); ts = tp.split(",")
    xs = xa.split(",") if isinstance(xa, str) else []
    ys = ya.split(",") if isinstance(ya, str) else []
    ss = spd.split(",") if isinstance(spd, str) else []
    n = min(len(pts), len(ts))
    out = []
    for i in range(n):
        ll = pts[i].split()
        if len(ll) != 2:
            continue
        try:
            lon = float(ll[0]); lat = float(ll[1])
        except ValueError:
            continue
        if not (BBOX["lon_min"] <= lon <= BBOX["lon_max"] and
                BBOX["lat_min"] <= lat <= BBOX["lat_max"]):
            continue

        def g(arr):
            if i < len(arr) and arr[i] not in ("null", "", "None"):
                try:
                    return float(arr[i])
                except ValueError:
                    return np.nan
            return np.nan
        x = g(xs); y = g(ys); v = g(ss)
        out.append((ts[i], lon, lat, v, x, y, vtype))
    return out or None


def flag_events(df):
    xa = df["XAcc"].fillna(0.0); ya = df["YAcc"].fillna(0.0); sp = df["Speed"].fillna(0.0)
    df["brake"] = (xa <= TH_BRAKE).astype(int)
    df["accel"] = (xa >= TH_ACCEL).astype(int)
    df["corner"] = (ya.abs() >= TH_CORNER).astype(int)
    df["extreme"] = (sp >= SPEED_EXTREME).astype(int)
    df["mod"] = ((xa <= TH_BRAKE_M) | (xa >= TH_ACCEL_M) | (ya.abs() >= TH_CORNER_M)).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.environ.get("RAW_DIR",
                    "data/raw_near_miss_2026"),
                    help="directory of trips-*.csv.gz")
    ap.add_argument("--out", default="results/event_points_full.csv.gz")
    ap.add_argument("--date-min", default="2025-01-01")
    ap.add_argument("--date-max", default="2026-03-31")
    ap.add_argument("--ckpt", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="process only N files (0=all)")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (PBS array id)")
    ap.add_argument("--nshards", type=int, default=1, help="total shards for array job")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "trips-*.csv.gz")))
    if not files:
        raise SystemExit(
            f"[50] FATAL: no trips-*.csv.gz found in '{args.raw_dir}'. "
            f"The 88 GB raw trip files must be present (copy raw_near_miss_2026/ to "
            f"this project, or set --raw-dir / $RAW_DIR to their location). Refusing "
            f"to write an empty event file.")
    if args.limit:
        files = files[:args.limit]
    if args.nshards > 1:
        files = files[args.shard::args.nshards]
        base, ext = args.out, ""
        if args.out.endswith(".csv.gz"):
            base, ext = args.out[:-7], ".csv.gz"
        args.out = f"{base}.shard{args.shard:03d}{ext}"
        print(f"[50] shard {args.shard}/{args.nshards}: {len(files)} files -> {args.out}")
    print(f"[50] {len(files)} raw trip files in {args.raw_dir}")
    print(f"[50] extracting study-area events {args.date_min}..{args.date_max}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done_marker = args.out + ".done_files"
    done = set()
    if os.path.exists(done_marker):
        done = set(open(done_marker).read().split())
        print(f"[50] resuming: {len(done)} files already processed")
    mode = "at" if done else "wt"
    first_write = not done

    usecols = ["StartDate", "SnappedPath", "TimestampPath", "SpeedPath_kph",
               "XAccPath", "YAccPath", "VehicleType"]
    t0 = time.time(); n_events = 0
    with gzip.open(args.out, mode, newline="") as fout:
        if first_write:
            fout.write(",".join(OUT_COLS) + "\n")
        for k, f in enumerate(files):
            if os.path.basename(f) in done:
                continue
            try:
                df = pd.read_csv(f, usecols=usecols)
            except Exception as e:
                print(f"  !! {os.path.basename(f)}: {e} (SKIPPING)", flush=True)
                with open(done_marker, "a") as dm:
                    dm.write(os.path.basename(f) + "\n")
                continue
            df = df[(df.StartDate >= args.date_min) & (df.StartDate <= args.date_max)]
            rows = []
            for t in df.itertuples(index=False):
                pr = parse_trip(t.SnappedPath, t.TimestampPath, t.SpeedPath_kph,
                                t.XAccPath, t.YAccPath, t.VehicleType)
                if pr:
                    rows.extend(pr)
            if rows:
                pdf = pd.DataFrame(rows, columns=["dt", "SnappedLongitude", "SnappedLatitude",
                                                  "Speed", "XAcc", "YAcc", "VehicleType"])
                pdf = flag_events(pdf)
                mask = pdf[["brake", "accel", "corner", "extreme", "mod"]].sum(axis=1) > 0
                ev = pdf.loc[mask, OUT_COLS]
                ev.to_csv(fout, header=False, index=False)
                n_events += len(ev)
            with open(done_marker, "a") as dm:
                dm.write(os.path.basename(f) + "\n")
            if (k + 1) % args.ckpt == 0:
                el = time.time() - t0
                print(f"  [{k+1}/{len(files)}] {n_events:,} events | "
                      f"{el:.0f}s | {el/(k+1):.2f}s/file", flush=True)
    print(f"[50] DONE: {n_events:,} study-area events -> {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
