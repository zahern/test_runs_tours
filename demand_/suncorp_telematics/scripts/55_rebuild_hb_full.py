"""
55_rebuild_hb_full.py
========================================================================
Rebuild hard-braking events from the RAW trip files for the FULL
2025-01-01 -> 2026-03-31 period.

Why this exists
---------------
data/hard_braking_events.csv (Aug 2026, external deliverable) claims
EventStart dates out to Dec 2026, but the raw trips in
data/raw_near_miss_2026/ end 2026-03-31 -- ~400k rows in that file
cannot have come from these trips. It is also March-dominated
(~66% of 2.5M rows fall in 2026-03). Downstream products built from it
(data/near_miss_hb_ecf.csv, results/eb_*) therefore inherit an
inconsistent temporal footprint.

This script re-derives HB events directly from the 5000 raw
trips-*.csv.gz files so every row is traceable to a real trip in the
full-year window. Output schema is byte-compatible with the provider
file (same 14 columns, same order) so all existing consumers
(18_prepare_eb_dataset.py, 29_deployment_plan.py, build_events_cache.py,
build_surfaces.py) work unchanged:

    VehicleID,VehicleType,EventStart,EventEnd,PeakDecel_g,PeakJerk_g_s,
    SpeedLoss_kmh,Duration_s,Category,Confidence,
    SnappedLongitude,SnappedLatitude,RawLongitude,RawLatitude

Detection logic (calibrated to the provider file's observed bounds)
-------------------------------------------------------------------
Provider PeakDecel_g ranges observed in data/hard_braking_events.csv:
    Candidate         [0.20, 0.50)
    Critical-Likely   [0.50, 0.75)
    Extreme           [0.75,  +inf)
(SpeedLoss/Duration/PeakJerk are ~always 0 there -- single-sample
exceedances. We compute honest values here; downstream only uses
Category + coordinates + timestamps.)

Per trip (paths are comma-separated, 1:1 sample-aligned):
  * per-sample decel signal d (g):
      d = -XAcc            if XAcc valid
      d = speed-derived    otherwise, i.e. (v[i-1]-v[i])/3.6/9.80665/dt
                           with dt from TimestampPath. The fallback is
                           skipped unless 1 s <= dt <= 300 s: raw timestamps
                           resolve to milliseconds, so sub-second dt turns
                           speed quantisation into hundreds of g.
  * flag samples with d >= 0.20; strictly contiguous flagged runs form
    one event each (no gap-merging, mirroring the provider's
    single-sample-style events).
  * PeakDecel = max d in run; Category by the bounds above;
    Confidence = Low (provider file is 99.999% Low: 20 High / 2.5M).
  * EventStart/End = TimestampPath at run first/last sample.
  * Coords = SnappedPath/RawPath "lon lat" pair at the peak sample
    (empty string when missing, as in the provider file).
  * SpeedLoss_kmh = max(0, v_first - v_last); Duration_s from timestamps;
    PeakJerk_g_s = max |dXAcc|/dt inside the run (0 if XAcc missing).

Trips are pre-filtered on StartDate in [date_min, date_max]
(default 2025-01-01..2026-03-31, same as 50_extract_raw_events.py).

HPC: 20-way PBS array like pbs_50_extract.pbs --
    --shard $PBS_ARRAY_INDEX --nshards 20
Resume: per-shard "<out>.done_files" marker (same convention as
script 50). Merge shards afterwards with 55b_merge_hb_shards.py.

Usage:
  # smoke: 2 files, shard 0 of 20
  python scripts/55_rebuild_hb_full.py --limit 2 --shard 0 --nshards 20 \
      --out results/hb_full.smoke.csv.gz
  # full (via PBS): see pbs_55_rebuild_hb.pbs
"""
import argparse
import csv
import glob
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd

csv.field_size_limit(10 ** 8)

G = 9.80665
TH_HB = 0.20
TH_CRIT = 0.50
TH_EXTREME = 0.75
# Speed-derived decel needs a sane time base: raw TimestampPath resolution
# runs to milliseconds, so sub-second dt amplifies speed quantisation noise
# into hundreds of g (observed max 381g without this floor). 1 Hz+ resampled
# provider-style signals never see dt < 1 s.
DT_MIN_S = 1.0
DT_MAX_S = 300.0

OUT_COLS = ["VehicleID", "VehicleType", "EventStart", "EventEnd",
            "PeakDecel_g", "PeakJerk_g_s", "SpeedLoss_kmh", "Duration_s",
            "Category", "Confidence",
            "SnappedLongitude", "SnappedLatitude",
            "RawLongitude", "RawLatitude"]


def _floats(toks, n):
    """Parse first n tokens to float array; 'null'/''/bad -> NaN."""
    out = np.full(n, np.nan)
    for i in range(min(n, len(toks))):
        s = toks[i].strip()
        if not s or s in ("null", "NULL", "None", "nan", "NaN"):
            continue
        try:
            out[i] = float(s)
        except ValueError:
            continue
    return out


def _lonlat(toks, n):
    """Parse 'lon lat' tokens to two float arrays; bad -> NaN."""
    lon = np.full(n, np.nan)
    lat = np.full(n, np.nan)
    for i in range(min(n, len(toks))):
        sp = toks[i].strip().split()
        if len(sp) != 2:
            continue
        try:
            lon[i] = float(sp[0])
            lat[i] = float(sp[1])
        except ValueError:
            continue
    return lon, lat


def _fmt_ts(t):
    if t is pd.NaT or t is None:
        return ""
    try:
        return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def detect_trip_events(row):
    """Return list of event dicts for one raw trip row. Empty list if none."""
    sp = row.get("SnappedPath")
    tp = row.get("TimestampPath")
    if not isinstance(sp, str) or not isinstance(tp, str) or not sp or not tp:
        return []
    pts = sp.split(",")
    tst = tp.split(",")
    n = min(len(pts), len(tst))
    if n < 2:
        return []

    spd = (row.get("SpeedPath_kph") or "").split(",")
    xa = (row.get("XAccPath") or "").split(",")
    raw = (row.get("RawPath") or "").split(",")

    v = _floats(spd, n)
    x = _floats(xa, n)
    slon, slat = _lonlat(pts, n)
    rlon, rlat = _lonlat(raw, n)
    try:
        t = pd.to_datetime(tst[:n], errors="coerce")
    except Exception:
        return []

    # per-sample decel signal (g): accelerometer first, speed-derived fallback.
    # The fallback is only valid on a sane time base (see DT_MIN_S note).
    dt = t.to_series().diff().dt.total_seconds().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        dv = np.empty(n)
        dv[0] = np.nan
        dv[1:] = v[:-1] - v[1:]
        speed_d = dv / 3.6 / G / dt
    speed_d[(dt < DT_MIN_S) | (dt > DT_MAX_S) | ~np.isfinite(dt)] = np.nan
    x_valid = np.isfinite(x)
    d = np.where(x_valid, -x, speed_d)

    flagged = np.isfinite(d) & (d >= TH_HB)
    if not flagged.any():
        return []

    idx = np.flatnonzero(flagged)
    # strictly contiguous runs
    breaks = np.flatnonzero(np.diff(idx) > 1)
    runs = np.split(idx, breaks + 1) if len(breaks) else [idx]

    vid = row.get("VehicleID", "")
    vtype = row.get("VehicleType", "")
    fallback_ts = row.get("StartTimestamp_Local", "")
    events = []
    for run in runs:
        a, b = int(run[0]), int(run[-1])
        seg = d[run]
        peak_rel = int(np.argmax(seg))
        p = int(run[peak_rel])
        peak = float(seg[peak_rel])
        if peak >= TH_EXTREME:
            cat = "Extreme"
        elif peak >= TH_CRIT:
            cat = "Critical-Likely"
        else:
            cat = "Hard-Braking Candidate"
        t0, t1 = t[a], t[b]
        s0, s1 = _fmt_ts(t0), _fmt_ts(t1)
        if not s0:
            s0 = fallback_ts
        if not s1:
            s1 = s0
        dur = 0.0
        try:
            if pd.notna(t0) and pd.notna(t1):
                dur = max(0.0, (t1 - t0).total_seconds())
        except Exception:
            pass
        vloss = 0.0
        if np.isfinite(v[a]) and np.isfinite(v[b]):
            vloss = max(0.0, float(v[a] - v[b]))
        jerk = 0.0
        if len(run) > 1 and x_valid[run].sum() >= 2:
            xs = x[run]
            ok = np.isfinite(xs)
            if ok.sum() >= 2:
                dts = dt[run][1:]
                dx = np.abs(np.diff(xs))
                with np.errstate(divide="ignore", invalid="ignore"):
                    j = dx / dts
                j = j[np.isfinite(j) & (dts > 0)]
                if len(j):
                    jerk = float(j.max())

        def num(z):
            return "" if not np.isfinite(z) else float(z)

        events.append({
            "VehicleID": vid, "VehicleType": vtype,
            "EventStart": s0, "EventEnd": s1,
            "PeakDecel_g": round(peak, 3),
            "PeakJerk_g_s": round(jerk, 3),
            "SpeedLoss_kmh": round(vloss, 1),
            "Duration_s": round(dur, 1),
            "Category": cat, "Confidence": "Low",
            "SnappedLongitude": num(slon[p]), "SnappedLatitude": num(slat[p]),
            "RawLongitude": num(rlon[p]), "RawLatitude": num(rlat[p]),
        })
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.environ.get(
        "RAW_DIR", "data/raw_near_miss_2026"))
    ap.add_argument("--out", default="results/hb_full.csv.gz")
    ap.add_argument("--date-min", default="2025-01-01")
    ap.add_argument("--date-max", default="2026-03-31")
    ap.add_argument("--ckpt", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "trips-*.csv.gz")))
    if not files:
        raise SystemExit(
            f"[55] FATAL: no trips-*.csv.gz in '{args.raw_dir}'. "
            f"Set --raw-dir / $RAW_DIR to data/raw_near_miss_2026.")
    if args.limit:
        files = files[:args.limit]
    if args.nshards > 1:
        files = files[args.shard::args.nshards]
        base = args.out[:-7] if args.out.endswith(".csv.gz") else args.out
        args.out = f"{base}.shard{args.shard:03d}.csv.gz"
        print(f"[55] shard {args.shard}/{args.nshards}: "
              f"{len(files)} files -> {args.out}", flush=True)
    print(f"[55] {len(files)} raw trip files; window "
          f"{args.date_min}..{args.date_max}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done_marker = args.out + ".done_files"
    done = set()
    if os.path.exists(done_marker):
        done = set(open(done_marker).read().split())
        print(f"[55] resuming: {len(done)} files already processed", flush=True)
    first_write = not done

    usecols = ["VehicleID", "VehicleType", "StartDate",
               "StartTimestamp_Local", "TimestampPath", "SnappedPath",
               "RawPath", "SpeedPath_kph", "XAccPath"]
    t0 = time.time()
    n_events = 0
    cat_counts = {"Hard-Braking Candidate": 0, "Critical-Likely": 0,
                  "Extreme": 0}
    with gzip.open(args.out, "at" if done else "wt", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=OUT_COLS)
        if first_write:
            w.writeheader()
        for k, f in enumerate(files):
            if os.path.basename(f) in done:
                continue
            # Pre-check: some raw files are corrupted (just \x00 bytes)
            try:
                with open(f, "rb") as raw:
                    if raw.read(2) != b"\x1f\x8b":
                        print(f"  !! {os.path.basename(f)}: not a valid gzip (SKIPPING)", flush=True)
                        with open(done_marker, "a") as dm:
                            dm.write(os.path.basename(f) + "\n")
                        continue
            except Exception as e:
                print(f"  !! {os.path.basename(f)}: gzip check failed: {e} (SKIPPING)", flush=True)
                with open(done_marker, "a") as dm:
                    dm.write(os.path.basename(f) + "\n")
                continue
            try:
                df = pd.read_csv(f, usecols=usecols)
            except Exception as e:
                print(f"  !! {os.path.basename(f)}: {e} (SKIPPING)", flush=True)
                with open(done_marker, "a") as dm:
                    dm.write(os.path.basename(f) + "\n")
                continue
            df = df[(df.StartDate >= args.date_min) &
                    (df.StartDate <= args.date_max)]
            for t in df.itertuples(index=False):
                row = t._asdict()
                for ev in detect_trip_events(row):
                    w.writerow(ev)
                    n_events += 1
                    cat_counts[ev["Category"]] += 1
            with open(done_marker, "a") as dm:
                dm.write(os.path.basename(f) + "\n")
            if (k + 1) % args.ckpt == 0:
                el = time.time() - t0
                print(f"  [{k+1}/{len(files)}] {n_events:,} events "
                      f"(cand={cat_counts['Hard-Braking Candidate']:,}, "
                      f"crit={cat_counts['Critical-Likely']:,}, "
                      f"ext={cat_counts['Extreme']:,}) | {el:.0f}s | "
                      f"{el/(k+1):.2f}s/file", flush=True)
    print(f"[55] DONE: {n_events:,} HB events -> {args.out} "
          f"in {time.time()-t0:.0f}s", flush=True)
    print(f"[55] by category: {cat_counts}", flush=True)


if __name__ == "__main__":
    main()
