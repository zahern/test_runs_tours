#!/usr/bin/env python
"""
Why does road_osmid 530172035 get the TOP predicted speeding in all six
games when it has ZERO observed speeding every time?

Candidates:
  (a) the segment genuinely has speeding history and only the game-day
      windows missed it  -> a real (if fixable) modelling issue
  (b) it never has speeding at all -> a data/join artefact, and the single
      biggest predicted segment in the whole study is spurious
  (c) its exposure is enormous, so a rate-based offset predicts large counts
      regardless of outcome  -> the offset, not the signal, is driving it
"""
import glob
import os

import numpy as np
import pandas as pd

TARGETS = [530172035, 1433306178, 29702566]
panels = sorted(glob.glob("results/*15min*.parquet"))
if not panels:
    panels = sorted(glob.glob("**/*15min*panel*.parquet", recursive=True))
print("panel files found: %d" % len(panels))
for p in panels[:20]:
    print("   ", p)

if not panels:
    raise SystemExit("no panel files to inspect")

# Find the speeding column name from the first panel.
probe = pd.read_parquet(panels[0])
print()
print("panel columns:", list(probe.columns))
cand = [c for c in probe.columns
        if "judged" in c.lower() or "speeding" in c.lower()
        or c.lower().startswith("obs")]
print("candidate outcome columns:", cand)

rows = []
for p in panels:
    df = pd.read_parquet(p)
    cols = [c for c in df.columns if c == "road_osmid"] + \
           [c for c in df.columns
            if "judged" in c.lower() or "speeding" in c.lower()]
    d = df[df["road_osmid"].isin(TARGETS)][cols]
    if len(d):
        d = d.assign(panel=os.path.basename(p))
        rows.append(d)
allp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

print()
print("=== rows for the three dominant predicted segments ===")
print("total rows across panels: %d" % len(allp))
if len(allp):
    agg = allp.groupby("road_osmid").agg(
        rows=("panel", "size"), panels=("panel", "nunique"))
    for c in [c for c in allp.columns
              if c not in ("road_osmid", "panel")]:
        agg["sum_" + c] = allp.groupby("road_osmid")[c].sum()
        agg["mean_" + c] = allp.groupby("road_osmid")[c].mean()
    print(agg.to_string())

print()
print("=== how do these compare to the network? ===")
df = pd.read_parquet(panels[0])
spd = [c for c in df.columns if "judged" in c.lower() or "speeding" in c.lower()]
if spd:
    c = spd[0]
    tot = pd.to_numeric(df[c], errors="coerce")
    print("outcome column:", c, " network mean %.3f median %.3f" %
          (tot.mean(), tot.median()))
    for t in TARGETS:
        v = pd.to_numeric(df.loc[df["road_osmid"] == t, c], errors="coerce")
        print("   %-12s n=%4d  mean=%s  sum=%s"
              % (t, len(v), ("%.3f" % v.mean()) if len(v) else "n/a",
                 ("%.1f" % v.sum()) if len(v) else "n/a"))

print()
print("=== is 530172035 duplicated / mis-joined in the panel? ===")
df = pd.read_parquet(panels[0])
d = df[df["road_osmid"] == 530172035]
print("rows for 530172035 in panel[0]: %d" % len(d))
if len(d):
    keys = [c for c in ("panel_id", "bin", "week", "month") if c in d.columns]
    print("distinct %s: %s" % (keys, {k: d[k].nunique() for k in keys}))
    print(d.head(8).to_string()[:1500])
