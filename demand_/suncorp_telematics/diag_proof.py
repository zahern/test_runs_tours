#!/usr/bin/env python
"""
Why is r negative? The decisive question: are the top-20 predicted segments
ZERO observed because there was no speeding, or because they were never
OBSERVED in that game's window?

If a segment has no observed rows at all, "0 observed speeding" is missing
data, and the proof is measuring data coverage rather than model skill.
That distinction changes the conclusion entirely.
"""
import numpy as np
import pandas as pd

ARM = "results_tests/t4_armB_calib"
top = pd.read_csv(ARM + "/game_day_top20_per_game.csv")
pred = pd.read_parquet(ARM + "/game_day_pred_15min.parquet",
                       columns=["fixture", "road_osmid", "pred_speeding"])

print("=== observed-row coverage ===")
try:
    top["obs_speeding"] = pd.to_numeric(top["obs_speeding"], errors="coerce")
except KeyError:
    pass

print("columns in top20 csv:", list(top.columns))
print()

for fx, g in top.groupby("fixture"):
    g = g.copy()
    if "obs_speeding" not in g.columns:
        print(fx, "-> no obs_speeding column")
        continue
    obs = g["obs_speeding"]
    n_rows = len(g)
    n_nan = int(obs.isna().sum())
    n_zero = int((obs.fillna(-1) == 0).sum())
    n_pos = int((obs.fillna(-1) > 0).sum())
    big = g.nlargest(3, "pred_speeding")[["road_osmid", "pred_speeding",
                                          "obs_speeding"]]
    print("%-44s n=%d  obs_missing=%d  obs_zero=%d  obs_positive=%d"
          % (str(fx)[:44], n_rows, n_nan, n_zero, n_pos))
    print("    top-3 predicted segments:")
    for _, r in big.iterrows():
        print("      osm=%-12s pred=%8.1f  obs=%s"
              % (int(r["road_osmid"]), r["pred_speeding"], r["obs_speeding"]))

print()
print("=== what share of predicted speeding sits on unobserved segments? ===")
tot_pred = tot_unobs = 0.0
for fx, g in top.groupby("fixture"):
    g = g.copy()
    g["obs_speeding"] = pd.to_numeric(g["obs_speeding"], errors="coerce")
    unobs = g["obs_speeding"].isna() | (g["obs_speeding"].fillna(0) <= 0)
    p = g["pred_speeding"].sum()
    pu = g.loc[unobs, "pred_speeding"].sum()
    tot_pred += p
    tot_unobs += pu
    print("%-44s pred_total=%8.1f  on unobserved=%8.1f (%5.1f%%)"
          % (str(fx)[:44], p, pu, 100.0 * pu / max(p, 1e-9)))
print("-" * 78)
print("OVERALL: %.1f%% of top-20 predicted speeding is on segments with no "
      "observed speeding" % (100.0 * tot_unobs / max(tot_pred, 1e-9)))

print()
print("=== is the damage driven by a few outlier segments? ===")
allp = top.copy()
allp["obs_speeding"] = pd.to_numeric(allp["obs_speeding"], errors="coerce")
allp = allp.sort_values("pred_speeding", ascending=False)
s = allp["pred_speeding"].cumsum() / allp["pred_speeding"].sum()
for frac in (0.01, 0.05, 0.10, 0.25):
    n = max(1, int(round(frac * len(allp))))
    print("  top %5.1f%% of rows (%4d rows) carry %5.1f%% of predicted speeding"
          % (100 * frac, n, 100 * s.iloc[n - 1]))
