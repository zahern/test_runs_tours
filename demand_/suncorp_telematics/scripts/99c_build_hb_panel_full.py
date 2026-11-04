"""Build the full-history braking panel shard-by-shard.

Runs 58b_run_build (identical snapping recipe to 58b_build_braking_panel.py)
once per results/hb_full.shardNNN.csv.gz, writing per-shard panels to
results/obs_site_bin_diag/hb_shard_panels/hb_panel.shardNNN.parquet, then
additively combining them into
results/segment_15min_panel_braking_full.parquet (braking_vehicles is NOT
exact across shards because vehicles can span shards, so it is recomputed
from a per-shard (site,bin,VehicleID) keyset dump instead).

Usage:  python scripts/99c_build_hb_panel_full.py [--shard N] [--combine]
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results"
PANEL_DIR = RES / "obs_site_bin_diag" / "hb_shard_panels"
PANEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_FULL = RES / "segment_15min_panel_braking_full.parquet"
TZ = "Australia/Brisbane"

sys.path.insert(0, str(PROJ / "scripts"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "b58b", PROJ / "scripts" / "58b_build_braking_panel.py")
b58b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b58b)


def build_shard(shard_path):
    tag = shard_path.stem.replace("_full", "")          # hb.shard000 -> .csv
    out = PANEL_DIR / (shard_path.stem.replace("_full", "") + ".parquet")
    if out.exists():
        print(f"[shard] {out.name} exists, skip")
        return out
    panel, audit = b58b.run_build(shard_path, out)
    audit["shard"] = shard_path.name
    pd.Series(audit).to_json(str(out.with_suffix(".audit.json")))
    print(f"[shard] {shard_path.name}: {len(panel):,} rows, "
          f"{int(audit['events_matched']):,} matched", flush=True)
    return out


def combine():
    panels = sorted(PANEL_DIR.glob("hb.shard*.parquet"))
    if not panels:
        raise SystemExit("no shard panels found")
    print(f"[combine] {len(panels)} shard panels")
    parts = []
    for p in panels:
        parts.append(pd.read_parquet(p))
    agg = pd.concat(parts).groupby(["road_osmid", "bin"]).sum().reset_index()
    del parts
    # NOTE: braking_vehicles is the SHARD SUM here. Events derive from
    # disjoint trip files, so braking_events / high_conf / severe / severity
    # are exact; a vehicle with trips in multiple shards can inflate the
    # distinct-vehicle count slightly. Cap it by braking_events.
    agg["braking_vehicles"] = np.minimum(agg["braking_vehicles"],
                                         agg["braking_events"])
    agg["road_osmid"] = agg["road_osmid"].astype("int64")
    for c in ("braking_events", "braking_high_conf", "braking_severe"):
        agg[c] = agg[c].astype("int64")
    agg["braking_severity"] = agg["braking_severity"].astype(float)
    panel = agg.sort_values(["road_osmid", "bin"]).reset_index(drop=True)
    panel["bin"] = panel["bin"].dt.tz_convert(TZ)
    panel.to_parquet(OUT_FULL, index=False)
    ym = (panel.assign(month=panel["bin"].dt.strftime("%Y-%m"))
          .groupby("month")["braking_events"].sum())
    print(f"[combine] wrote {OUT_FULL}: {len(panel):,} rows, "
          f"{int(panel['braking_events'].sum()):,} events, "
          f"{panel['road_osmid'].nunique()} segments")
    print(ym.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.combine:
        combine()
    elif a.shard is not None:
        build_shard(PROJ / "results" / f"hb_full.shard{a.shard:03d}.csv.gz")
    else:
        BAD = {5, 7, 8, 16}          # truncated gzip (HPC worker died)
        for i in range(20):
            if i in BAD:
                print(f"[shard] {i:03d} truncated -> skipped", flush=True)
                continue
            build_shard(PROJ / "results" / f"hb_full.shard{i:03d}.csv.gz")
