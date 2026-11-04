#!/usr/bin/env python
"""Oracle upper bound on relocation value: per-bin optimal top-P coverage
vs single static set, and progressive (0..P-swap) mobility curves."""
import numpy as np, pandas as pd

ROOT = r"Z:\test_runs_tours\demand_\suncorp_telematics"
df = pd.read_parquet(fr"{ROOT}\results_seg\game_day_pred_15min.parquet")

rows = []
for fx, g in df.groupby("fixture"):
    B = g.pivot_table(index="rel_bin", columns="road_osmid",
                      values="pred_speeding", aggfunc="sum").fillna(0.0)
    tot = B.values.sum()
    # greedy static set over all bins (marginal-greedy)
    chosen, rem = [], set(B.columns)
    for _ in range(10):
        best, bm = None, -1
        for s in rem:
            m = B[chosen + [s]].values.sum() if chosen else B[s].values.sum()
            if m > bm: bm, best = m, s
        chosen.append(best); rem.discard(best)
    static_cov = B[chosen[:10]].values.sum() / tot
    # oracle: per-bin top-k (unlimited moves)
    orc = []
    for P in range(1, 11):
        orc.append(np.sort(B.values, axis=1)[:, -P:].sum() / tot)
    # greedy progressive: allow k swaps total between bins -> use per-bin greedy
    top_static = chosen
    gains = []
    for P in range(1, 11):
        # phased: choose best set per phase (5 mid bins granularity: quarters of day)
        for qp in range(4):
            pass
        gains.append(None)
    # quarters of day phased reopt
    q_cov = []
    for P in range(1, 11):
        cs = 0.0
        bins_q = np.array_split(np.arange(0, 28), 4)
        for q in bins_q:
            b = B.iloc[q]
            top = b.sum(0).nlargest(P).index
            cs += b[top].values.sum() / b.values.sum()
        q_cov.append(cs / 4)
    rows.append({"fixture": fx,
                 "static_cov_P5": round(100 * static_cov if False else 100 * B[chosen[:5]].values.sum() / tot, 2),
                 "oracle_binwise_P5": round(100 * orc[4], 2),
                 "quarter_phased_P5": round(100 * q_cov[4], 2),
                 "static_cov_P10": round(100 * static_cov, 2),
                 "oracle_binwise_P10": round(100 * orc[9], 2),
                 "quarter_phased_P10": round(100 * q_cov[9], 2)})
out = pd.DataFrame(rows)
out["reloc_value_quarters_P5_pct"] = (out.quarter_phased_P5 - out.static_cov_P5).round(3)
out["reloc_value_oracle_P5_pct"] = (out.oracle_binwise_P5 - out.static_cov_P5).round(3)
pd.set_option("display.width", 220)
print(out.to_string(index=False))
out.to_csv(fr"{ROOT}\results_seg\game_day_reloc_oracle_ceiling.csv", index=False)
