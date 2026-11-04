#!/usr/bin/env python3
"""Diagnose why panel observed mass and the 59 holdout cells observed mass
disagree, before either is quoted against a plan.

Prints, for the first fixture: panel window total, cells total, per-plan-site
panel vs cells observed, and duplicate-key checks in the cells file,
prediction parquet and panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TZ = "Australia/Brisbane"
FX = "NRL Broncos-Panthers @ 2026-03-06 19:00"
PLAN = [1376779235, 922464068, 28681153, 31977538, 1119727324]

panel = pd.read_parquet(ROOT / "results" / "segment_15min_panel_2026-03.parquet",
                        columns=["road_osmid", "bin", "speeding_trips"])
panel["bin"] = pd.to_datetime(panel["bin"], utc=True).dt.tz_convert(TZ)
start = pd.Timestamp("2026-03-06 16:00", tz=TZ)
end = start + pd.Timedelta(minutes=15 * 28)
w = panel[(panel["bin"] >= start) & (panel["bin"] < end)].copy()
print(f"panel window rows={len(w):,} unique(seg,bin)="
      f"{len(w.drop_duplicates(['road_osmid','bin'])):,} "
      f"total={w['speeding_trips'].sum():.0f}")
print("panel duplicate keys:",
      int(w.duplicated(["road_osmid", "bin"]).sum()))
site = w[w["road_osmid"].isin(PLAN)]
print("panel plan-site rows:", len(site),
      "mass:", site["speeding_trips"].sum())
print(site.groupby("road_osmid")["speeding_trips"].sum().to_string())

cells = pd.read_csv(ROOT / "results" / "game_day_pred_vs_actual_cells.csv.gz")
cells["bin"] = pd.to_datetime(cells["bin"], utc=True)
c = cells[cells["fixture"] == FX].copy()
print(f"\ncells rows={len(c):,} unique(fx,seg,bin)="
      f"{len(c.drop_duplicates(['fixture','road_osmid','bin'])):,} "
      f"total={c['observed'].sum():.0f}")
print("cells duplicate keys:",
      int(c.duplicated(["fixture", "road_osmid", "bin"]).sum()))
cs = c[c["road_osmid"].isin(PLAN)]
print("cells plan-site rows:", len(cs), "mass:", cs["observed"].sum())
print(cs.groupby("road_osmid")["observed"].sum().to_string())
print("\ncells rows per bin range:",
      c["bin"].min(), "->", c["bin"].max(), "n_bins:", c["bin"].nunique())
print("cells segments:", c["road_osmid"].nunique())

pred = pd.read_parquet(ROOT / "results" / "game_day_pred_15min.parquet",
                       columns=["fixture", "road_osmid", "bin",
                                "pred_speeding"])
p = pred[pred["fixture"] == FX]
print(f"\npred rows={len(p):,} unique="
      f"{len(p.drop_duplicates(['fixture','road_osmid','bin'])):,} "
      f"segments={p['road_osmid'].nunique()} bins={p['bin'].nunique()} "
      f"total={p['pred_speeding'].sum():.2f}")
print("pred duplicate keys:",
      int(p.duplicated(["fixture", "road_osmid", "bin"]).sum()))

# Restrict panel to the cell keys present in the cells file and re-sum.
keys = c[["road_osmid", "bin"]].copy()
keys["bin"] = keys["bin"].dt.tz_convert(TZ).dt.tz_localize(None)
w2 = w.copy()
w2["bin_local"] = w2["bin"].dt.tz_localize(None)
j = w2.merge(keys.drop_duplicates(), left_on=["road_osmid", "bin_local"],
             right_on=["road_osmid", "bin"], how="inner")
print(f"\npanel restricted to cells-file keys: rows={len(j):,} "
      f"mass={j['speeding_trips'].sum():.0f}")
print("panel mass on cells-file keys by site:")
print(j[j["road_osmid"].isin(PLAN)].groupby("road_osmid")["speeding_trips"]
      .sum().to_string())
