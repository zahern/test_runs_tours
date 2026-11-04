"""Compare default vs calibrated-SD surfaces: do rankings vary by game?"""
import json

import numpy as np
import pandas as pd

RES = ("/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/"
       "suncorp_telematics/results")

pred = pd.read_parquet(f"{RES}/game_day_pred_15min.parquet")
pred["bin"] = pd.to_datetime(pred["bin"], utc=True)
pred["road_osmid"] = pd.to_numeric(pred["road_osmid"], errors="coerce")

m = json.load(open(f"{RES}/game_day_holdout_metrics.json"))
aud = m["audit"]
pp = aud.get("two_stage", {}).get("pruned_propensity", {})

print("=== shipped propensity design (section 3 fix) ===")
terms = pp.get("terms") or {}
pv = pp.get("p_values") or {}
print(f"n_terms={len(terms)}")
for t, b in sorted(terms.items(), key=lambda kv: abs(kv[1][0]), reverse=True):
    p = pv.get(t, float("nan"))
    print(f"  {t:18s} {b[0]:+9.4f} se={b[1]:.4f} p={p:.4f}")

for col in ("pred_speeding", "pred_sd_game"):
    if col not in pred.columns:
        print(f"\n=== {col} ===\n  absent from this parquet (no SD columns); "
              "skipping")
        continue
    tot = pred.groupby(["fixture", "road_osmid"])[col].sum().reset_index()
    W = tot.pivot(index="road_osmid", columns="fixture", values=col).fillna(0.0)
    A = W.to_numpy()
    g = A.mean()
    ss_tot = ((A - g) ** 2).sum()
    ss_seg = ((A.mean(axis=1)[:, None] - g) ** 2).sum()
    ss_fx = ss_tot - ss_seg
    pool = W.sum(axis=1).sort_values(ascending=False).head(250).index
    Wp = W.loc[pool].to_numpy()
    rk = pd.DataFrame(Wp).rank(axis=0, ascending=False).to_numpy()
    t20 = rk <= 20
    overlaps = []
    for i in range(1, Wp.shape[1]):
        a, b = set(np.where(t20[:, 0])[0]), set(np.where(t20[:, i])[0])
        overlaps.append(len(a & b) / 20)
    cv = Wp.std(axis=1) / np.maximum(Wp.mean(axis=1), 1e-9)
    print(f"\n=== {col} ===")
    print(f"  between-segment {100*ss_seg/ss_tot:5.1f}%  "
          f"between-fixture {100*ss_fx/ss_tot:5.1f}%")
    print(f"  game totals: " + ", ".join(f"{v:,.0f}" for v in A.sum(axis=0)))
    print(f"  spread of game totals: {A.sum(axis=0).max()/max(A.sum(axis=0).min(),1):.2f}x")
    print(f"  top-20 mean rank change: {np.abs(rk-rk.mean(axis=1)[:,None]).mean():.0f} pos")
    print(f"  segments changing top-20 membership: {(t20.std(axis=1)>0).sum()}/250")
    print(f"  top-20 Jaccard vs game1: " +
          ", ".join(f"{x:.2f}" for x in overlaps))
    print(f"  pool CV across games: median {np.median(cv):.2f} "
          f"p90 {np.percentile(cv,90):.2f}")
    print(f"  distinct hottest game: {pd.DataFrame(Wp).idxmax(axis=1).nunique()}/"
          f"{Wp.shape[1]}")
