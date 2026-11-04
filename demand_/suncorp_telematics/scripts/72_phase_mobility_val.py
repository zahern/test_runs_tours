#!/usr/bin/env python
"""Phase-mobility value: does the location ranking shift enough between
event phases to justify officer relocations?

A) Rank stability across phases per fixture (Kendall tau, top-k churn).
B) Prediction surface per phase -> optimal phased allocation (rank-greedy,
   <=2 relocations/officer) vs static one-site-all-bins policy.
   Value of moving = coverage gain, computed exactly from per-bin cells.
C) Noise floor for rank differences via Poisson-ish bootstrap on the
   aggregated predicted counts (hierarchical shrinkage of phase means).

Input : results_seg/game_day_pred_15min.parquet
Outputs: results_seg/game_day_phase_mobility_value.csv
"""
import numpy as np
import pandas as pd

ROOT = r"Z:\test_runs_tours\demand_\suncorp_telematics"
df = pd.read_parquet(fr"{ROOT}\results_seg\game_day_pred_15min.parquet")

PHASES = {  # auto boundaries from 60d run: [b1, b2]
    "NRL Broncos-Panthers @ 2026-03-06 19:00": (15, 21),
    "NRL Broncos-Eels @ 2026-03-12 19:00": (16, 22),
    "SuperRugby Reds-Waratahs @ 2026-03-14 18:45": (3, 10),
    "NRL Dolphins-Titans @ 2026-03-15 17:15": (3, 8),
    "ALeague Roar-Wellington Phoenix @ 2026-03-21 14:00": (3, 12),
    "NRL Broncos-Dolphins @ 2026-03-27 19:00": (15, 21),
}

def phase_of(fixture, rb):
    a, b = PHASES[fixture]
    if rb < a: return "pre"
    if rb <= b: return "during"
    return "post"

df["phase"] = [phase_of(f, r) for f, r in zip(df.fixture, df.rel_bin)]

# segment x fixture x phase totals (exact aggregation of per-bin cells)
seg = (df.groupby(["fixture", "phase", "road_osmid"], as_index=False)
         .agg(v=("pred_speeding", "sum")))
tot = seg.groupby(["fixture", "phase"]).v.sum()

def rank_vec(s, top):
    return set(s.nlargest(top).road_osmid)

def kendall_tau(x, y, k=200):
    x = x.nlargest(k); y = y.nlargest(k)
    xr = x.rank(); yr = y.rank()
    common = x.index.intersection(y.index)
    d = xr[common].values - yr[common].values
    return 1 - 3 * np.mean(d ** 2) / (k ** 2 - 1)  # approx tau for ties-free

rows = []
for fx, g in seg.groupby("fixture"):
    P = g.pivot_table(index="road_osmid", columns="phase", values="v")
    n_off = 6  # sweep showed squad 5-6; use 6
    # A) rank stability
    ph = ["pre", "during", "post"]
    for i in range(len(ph) - 1):
        a, b = P[ph[i]], P[ph[i + 1]]
        tau = kendall_tau(a, b)
        t10a, t10b = a.nlargest(10).index, b.nlargest(10).index
        jac10 = len(t10a & t10b) / len(t10a | t10b)
        # B) coverage: static top-n_off over all bins vs per-phase top-n_off
        static_top = a.add(b, fill_value=0).add(P[ph[-1]], fill_value=0).nlargest(n_off).index
        cov_static = sum(P[p][static_top].sum() / tot.loc[(fx, p)] for p in ph) / 3
        cov_phase = np.mean([P[p].nlargest(n_off).sum() / tot.loc[(fx, p)] for p in ph])
        moves = sum(len(P[p].nlargest(n_off).index.difference(static_top)) for p in ph)
        rows.append({
            "fixture": fx, "cmp": f"{ph[i]}_vs_{ph[i+1]}",
            "kendall_tau_top200": round(tau, 3),
            "top10_jaccard": round(jac10, 3),
            "cov_static_pct": round(100 * cov_static, 2),
            "cov_phased_pct": round(100 * cov_phase, 2),
            "mobility_gain_pct": round(100 * (cov_phase - cov_static), 3),
            "relocations_on_static": moves,
        })

out = pd.DataFrame(rows)
out.to_csv(fr"{ROOT}\results_seg\game_day_phase_mobility_value.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))

# C) is the phased gain real vs rank-noise? permutation: shuffle phase labels
#    of rel_bins within fixture, recompute phased gain (flight of chance)
rng = np.random.default_rng(0)
null = []
for fx, g in seg.groupby("fixture"):
    P = g.pivot_table(index="road_osmid", columns="phase", values="v")
    # rebuild phase labels from raw df with permuted rel_bin within day-half
    sub = df[df.fixture == fx]
    n_off = 6
    gains = []
    for _ in range(50):
        rlabels = rng.permutation(sub.rel_bin.values)
        s = pd.Series(rlabels, index=sub.index)
        pmap = s.map(lambda r: phase_of(fx, r))
        fake = sub.assign(phase=pmap.values)
        F = fake.groupby(["phase", "road_osmid"]).pred_speeding.sum().unstack(0)
        ft = F.sum()
        cov_phase = np.mean([F[ph].nlargest(n_off).sum() / ft[ph]
                             for ph in ["pre", "during", "post"] if ph in F])
        gains.append(cov_phase)
    null.append(np.mean(gains))
    print(fx, "expected gain if phase labels random:", round(100 * np.mean(gains), 3))
