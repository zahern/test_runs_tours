#!/usr/bin/env python3
"""Audit the Compass IoT hard-braking feed and the segment table it feeds.

Stage A  segment-level table (seconds).
Stage B  raw event feed (5.2 GB): rows, vehicles, window, tiers, confidence,
         physics, per-month counts.

Stage B is resumable.  The feed is read in byte-range slices and every finished
slice writes its own counters to _texbuild/audit_ckpt/, so a dropped connection
costs at most the slice in flight instead of the whole pass.  Re-running the
script continues at the first unfinished slice; the report is written only once
the last slice lands.

Usage
    python _texbuild/compass_audit.py                  # resume, run to the end
    python _texbuild/compass_audit.py --slice-mb 64     # smaller slices
    python _texbuild/compass_audit.py --max-slices 2    # bounded run, then resume later
    python _texbuild/compass_audit.py --restart         # discard checkpoints
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

FEED = "hard_braking_events_year.csv"
OUT = "_texbuild/compass_audit.txt"
CKPT = "_texbuild/audit_ckpt"
STATE = os.path.join(CKPT, "state.json")
USECOLS = ["VehicleID", "VehicleType", "EventStart", "Category",
           "Confidence", "PeakDecel_g", "SpeedLoss_kmh", "Duration_s"]

L = []


def say(msg):
    print(msg, file=sys.stderr, flush=True)


def add(msg=""):
    L.append(msg)
    say(msg)


# ------------------------------------------------------------------ Stage A
def stage_a():
    add("=" * 78)
    add("STAGE A -- segment analysis table (near_miss_hb_ecf_year.csv)")
    add("=" * 78)
    cols = ["segment_id", "ObsCrashes", "hb_candidate", "hb_critical",
            "hb_extreme", "nm_total", "seg_len_km", "AADT", "class", "lga",
            "surface_type", "Road_Stereotype"]
    df = pd.read_csv("near_miss_hb_ecf_year.csv", usecols=cols, low_memory=False)
    n = len(df)
    add(f"rows {n:,}  columns {df.shape[1]} (12 of 75 read)")
    add(f"total km {df.seg_len_km.sum():,.1f}")
    hb = df.nm_total
    add(f"HB events {hb.sum():,.0f}  tier split cand/crit/ext "
        f"{df.hb_candidate.sum():,.0f}/{df.hb_critical.sum():,.0f}/{df.hb_extreme.sum():,.0f}")
    add(f"segments with HB>0 {int((hb > 0).sum()):,} ({(hb > 0).mean():.1%})")
    add(f"crashes {df.ObsCrashes.sum():,.0f}  segments crash>0 "
        f"{int((df.ObsCrashes > 0).sum()):,} ({(df.ObsCrashes > 0).mean():.1%})")

    add("")
    add("-- HB coverage by road class --")
    g = df.groupby("class").agg(segments=("segment_id", "size"),
                                km=("seg_len_km", "sum"),
                                hb=("nm_total", "sum"),
                                crashes=("ObsCrashes", "sum"))
    g["seg_with_hb_pct"] = df.assign(has=hb > 0).groupby("class")["has"].mean() * 100
    g["hb_per_km"] = g.hb / g.km
    g["crashes_per_km"] = g.crashes / g.km
    add(g.sort_values("hb", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))

    add("")
    add("-- HB coverage by road stereotype --")
    g2 = df.groupby("Road_Stereotype").agg(segments=("segment_id", "size"),
                                           km=("seg_len_km", "sum"),
                                           hb=("nm_total", "sum"))
    g2["hb_per_km"] = g2.hb / g2.km
    g2["seg_with_hb_pct"] = df.assign(has=hb > 0).groupby("Road_Stereotype")["has"].mean() * 100
    add(g2.sort_values("hb", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))

    add("")
    add("-- concentration of HB events across segments --")
    s = hb.sort_values(ascending=False).values
    tot = s.sum()
    for k in (10, 100, 500, 1000):
        add(f"  top {k:>5} segments = {s[:k].sum() / tot:.1%} of all HB events")
    nz = s[s > 0]
    add(f"  median HB among segments with any event: {np.median(nz):,.0f}")
    add(f"  max HB on one segment: {s[0]:,.0f}; p99 among >0: {np.percentile(nz, 99):,.0f}")

    add("")
    add("-- overlap of the two responses --")
    both = (hb > 0) & (df.ObsCrashes > 0)
    add(f"  HB>0 & crashes>0 : {int(both.sum()):,}")
    add(f"  HB>0 & crashes=0 : {int(((hb > 0) & (df.ObsCrashes == 0)).sum()):,}")
    add(f"  HB=0 & crashes>0 : {int(((hb == 0) & (df.ObsCrashes > 0)).sum()):,} "
        f"({int(df.loc[(hb == 0) & (df.ObsCrashes > 0), 'ObsCrashes'].sum()):,} crashes)")
    add(f"  HB=0 & crashes=0 : {int(((hb == 0) & (df.ObsCrashes == 0)).sum()):,}")

    add("")
    add("-- AADT / per-km sanity --")
    add(f"  AADT missing: {int(df.AADT.isna().sum()):,}")
    add(f"  HB per km (all segments): {hb.sum() / df.seg_len_km.sum():,.1f}")
    urb = df.Road_Stereotype.str.contains("Urban", na=False)
    add(f"  HB per km urban {hb[urb].sum() / df.loc[urb, 'seg_len_km'].sum():,.1f} | "
        f"rural {hb[~urb].sum() / df.loc[~urb, 'seg_len_km'].sum():,.1f}")
    sub = df[hb > 0].copy()
    sub["hb_per_km"] = sub.nm_total / sub.seg_len_km
    add(f"  segments with HB: median HB/km {sub.hb_per_km.median():,.1f}, "
        f"p95 {sub.hb_per_km.quantile(.95):,.1f}, max {sub.hb_per_km.max():,.1f}")


# ------------------------------------------------------- Stage B plumbing
class RangedFile:
    """Binary read-only view of [start, end) so one slice is one small read."""

    def __init__(self, path, start, end):
        self._f = open(path, "rb")
        self._f.seek(start)
        self._pos = start
        self._end = end

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._end - self._pos
        n = min(n, self._end - self._pos)
        if n <= 0:
            return b""
        buf = self._f.read(n)
        self._pos += len(buf)
        return buf

    def readable(self):
        return True

    def close(self):
        self._f.close()


def feed_header():
    """Column names of the full feed and the byte offset where data starts."""
    with open(FEED, "rb") as f:
        line = f.readline()
    names = next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")]))
    return names, len(line)


def next_boundary(offset, size):
    """First newline boundary at or after `offset` (a slice must start on a row)."""
    if offset >= size:
        return size
    with open(FEED, "rb") as f:
        f.seek(offset)
        while True:
            blk = f.read(1 << 16)
            if not blk:
                return size
            k = blk.find(b"\n")
            if k >= 0:
                return min(offset + k + 1, size)
            offset += len(blk)


def scan_slice(start, end, names):
    """Counters for the rows in bytes [start, end)."""
    f = RangedFile(FEED, start, end)
    try:
        df = pd.read_csv(f, header=None, names=names, usecols=USECOLS,
                         low_memory=False)
    finally:
        f.close()

    rows = int(len(df))
    d = pd.to_datetime(df.EventStart, errors="coerce", format="mixed")
    dec = pd.to_numeric(df.PeakDecel_g, errors="coerce")
    loss = pd.to_numeric(df.SpeedLoss_kmh, errors="coerce")
    dur = pd.to_numeric(df.Duration_s, errors="coerce")
    conf = df.Confidence.astype(str).str.lower()

    mn, mx = d.min(), d.max()
    return {
        "start": int(start),
        "end": int(end),
        "rows": rows,
        "per_veh": {str(k): int(v) for k, v in
                    df.VehicleID.value_counts().to_dict().items()},
        "cat": {str(k): int(v) for k, v in
                df.Category.value_counts().to_dict().items()},
        "veh": {str(k): int(v) for k, v in
                df.VehicleType.value_counts().to_dict().items()},
        "conf": {str(k): int(v) for k, v in
                 df.Confidence.value_counts().to_dict().items()},
        "ym": {str(k): int(v) for k, v in
               d.dt.strftime("%Y-%m").value_counts().to_dict().items()},
        "stubs": int(((dur.fillna(0) == 0) & (loss.fillna(0) == 0)).sum()),
        "ge05": int((dec >= 0.5).sum()),
        "ge075": int((dec >= 0.75).sum()),
        "high": int((conf == "high").sum()),
        "first": None if pd.isna(mn) else mn.isoformat(),
        "last": None if pd.isna(mx) else mx.isoformat(),
    }


def slice_path(k):
    return os.path.join(CKPT, f"slice_{k:03d}.json")


def load_state(names_end):
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"next_start": names_end, "done": []}


def save_state(state):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, STATE)


def merge_parts(parts):
    agg = collections.defaultdict(collections.Counter)
    agg["rows"] = 0
    agg["stubs"] = agg["ge05"] = agg["ge075"] = agg["high"] = 0
    agg["first"] = agg["last"] = None
    for p in sorted(parts, key=lambda x: x["start"]):
        agg["rows"] += p["rows"]
        for key in ("per_veh", "cat", "veh", "conf", "ym"):
            agg[key].update(p[key])
        for key in ("stubs", "ge05", "ge075", "high"):
            agg[key] += p[key]
        if p["first"] and (agg["first"] is None or p["first"] < agg["first"]):
            agg["first"] = p["first"]
        if p["last"] and (agg["last"] is None or p["last"] > agg["last"]):
            agg["last"] = p["last"]
    return agg


def report_b(agg):
    rows = agg["rows"]
    per_veh = agg["per_veh"]
    nveh = len(per_veh)
    add("")
    add("=" * 78)
    add("STAGE B -- raw Compass IoT feed (hard_braking_events_year.csv, 5.17 GB)")
    add("=" * 78)
    add(f"rows read            {rows:,}")
    add(f"distinct vehicles    {nveh:,}")
    add(f"event window         {agg['first']} .. {agg['last']}")
    add(f"months with events   {len(agg['ym'])}")
    if nveh:
        vals = list(per_veh.values())
        add(f"rows/vehicle  mean {rows / nveh:,.0f}  median "
            f"{np.median(vals):,.0f}  p95 {np.percentile(vals, 95):,.0f}  max "
            f"{max(vals):,}")
    add("")
    add("-- vehicle types --")
    for k, v in agg["veh"].most_common():
        add(f"  {k:<12} {v:>12,}  {v / rows:6.2%}")
    add("")
    add("-- severity tiers (vendor Category) --")
    for k, v in agg["cat"].most_common():
        add(f"  {k:<28} {v:>12,}  {v / rows:6.2%}")
    add("")
    add("-- event confidence --")
    for k, v in agg["conf"].most_common():
        add(f"  {k:<12} {v:>12,}  {v / rows:6.2%}")
    add("")
    add("-- per-event physics --")
    add(f"  peak decel >= 0.50 g : {agg['ge05']:,} ({agg['ge05'] / rows:.2%})")
    add(f"  peak decel >= 0.75 g : {agg['ge075']:,} ({agg['ge075'] / rows:.2%})")
    add(f"  zero-duration & zero-speed-loss stubs: {agg['stubs']:,} "
        f"({agg['stubs'] / rows:.2%})")
    add(f"  Confidence == High  : {agg['high']:,} ({agg['high'] / rows:.2%})")
    add("")
    add("-- events per month --")
    for k in sorted(agg["ym"]):
        add(f"  {k}  {agg['ym'][k]:>10,}")


# ------------------------------------------------------------------ Stage B
def stage_b(slice_mb, max_slices, restart):
    names, offset = feed_header()
    missing = [c for c in USECOLS if c not in names]
    if missing:
        raise SystemExit(f"feed is missing columns {missing}; has {names}")

    if restart and os.path.isdir(CKPT):
        for fn in os.listdir(CKPT):
            if fn.endswith(".json"):
                os.remove(os.path.join(CKPT, fn))
    os.makedirs(CKPT, exist_ok=True)

    size = os.path.getsize(FEED)
    chunk = int(slice_mb * 1024 * 1024)
    state = load_state(offset)
    say(f"feed {size / 1e9:.2f} GB in {slice_mb} MB slices; resuming at byte "
        f"{state['next_start']:,} ({state['next_start'] / size:.1%}), "
        f"{len(state['done'])} slice(s) already checkpointed")

    t0 = time.time()
    ran = 0
    while state["next_start"] < size:
        if max_slices and ran >= max_slices:
            break
        start = state["next_start"]
        end = next_boundary(min(start + chunk, size), size)
        k = len(state["done"])
        t1 = time.time()
        part = scan_slice(start, end, names)
        part["seconds"] = round(time.time() - t1, 1)
        tmp = slice_path(k) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(part, f)
        os.replace(tmp, slice_path(k))
        state["done"].append(k)
        state["next_start"] = end
        save_state(state)
        ran += 1
        say(f"  slice {k + 1}: {part['rows']:,} rows, {part['seconds']:.0f}s, "
            f"{end / size:.1%} of feed, elapsed {time.time() - t0:.0f}s")

    parts = []
    for k in state["done"]:
        with open(slice_path(k), encoding="utf-8") as f:
            parts.append(json.load(f))
    agg = merge_parts(parts)

    if state["next_start"] < size:
        say(f"\nstopped at {state['next_start'] / size:.1%} of the feed; re-run to "
            "continue (checkpoints kept)")
        report_b(agg)
        return False

    report_b(agg)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slice-mb", type=int, default=128,
                    help="bytes per checkpointed slice (default 128)")
    ap.add_argument("--max-slices", type=int, default=0,
                    help="stop after this many new slices (0 = all)")
    ap.add_argument("--restart", action="store_true",
                    help="discard existing slice checkpoints")
    args = ap.parse_args()

    stage_a()
    add("")
    complete = stage_b(args.slice_mb, args.max_slices, args.restart)
    if complete:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        say(f"\nwrote {OUT}")
