#!/usr/bin/env python
"""Propensity-slope hierarchical model + mobility value via ranking.

 replaces the scaffold: EMPIRICAL-BAYES gamma-poisson hierarchy on the exact
 per-bin cells (equivalent to a Poisson GLMM with segment random intercepts
 and phase-specific slopes, with the Cauchy/probit part handled by pred_propensity).

 Model:
   segment x phase x bin cells: counts ~ Poisson(sum over bin pred_speeding),
   exposure T = sum exp.
   Level 1 (phase-specific): rate r_jp | prior -> Gamma(alpha_p, beta_p),
       alpha_p, beta_p fit by method-of-moments on r_jp across segments
       (phase-specific rank divergence retained).
   Level 2 (segment pooling): posterior mean shrunk toward the segment's
       across-phase mean with weight 1/(1+T beta_p / kappa ... ) via a
       segment-level Gamma prior fit on grand-mean rates.

 Outputs:
   results_seg/game_day_propensity_hier_phase_rates.csv
   results_seg/game_day_propensity_hier_rank_tau.csv
   results_seg/game_day_propensity_mobility_value.csv
"""
import numpy as np
import pandas as pd
from scipy import optimize

ROOT = r"Z:\test_runs_tours\demand_\suncorp_telematics"
df = pd.read_parquet(fr"{ROOT}\results_seg\game_day_pred_15min.parquet")

PHASES = {
    "NRL Broncos-Panthers @ 2026-03-06 19:00": (15, 21),
    "NRL Broncos-Eels @ 2026-03-12 19:00": (16, 22),
    "SuperRugby Reds-Waratahs @ 2026-03-14 18:45": (3, 10),
    "NRL Dolphins-Titans @ 2026-03-15 17:15": (3, 8),
    "ALeague Roar-Wellington Phoenix @ 2026-03-21 14:00": (3, 12),
    "NRL Broncos-Dolphins @ 2026-03-27 19:00": (15, 21),
}
def phase_of(f, rb):
    a, b = PHASES[f]
    return "pre" if rb < a else ("during" if rb <= b else "post")

df["phase"] = [phase_of(f, r) for f, r in zip(df.fixture, df.rel_bin)]
PH = ["pre", "during", "post"]

# ---- exact aggregation: segment x fixture x phase ----
tot_bin = (df.groupby(["fixture", "phase", "road_osmid"], as_index=False)
             .agg(E=("pred_speeding", "sum"), Texp=("exp", "sum")))

def fit_gamma_mom(x):
    """method-of-moments gamma for positive rates."""
    m, v = x.mean(), x.var(ddof=1)
    if v <= m:  # under-dispersed -> weak prior
        beta = 1.0 / max(m, 1e-12)
        return max(m, 1e-12), beta
    beta = m / v
    return m * beta, beta

# ---- level 1: phase-specific gamma priors ----
priors = {}
for (fx, ph), g in tot_bin.groupby(["fixture", "phase"]):
    r = g.E.values / np.maximum(g.Texp.values, 1e-12)
    r = r[r > 0]
    a, b = fit_gamma_mom(r)
    priors[(fx, ph)] = (a, b)

mean_rate = pd.DataFrame(
    [{"fixture": fx, "phase": ph, "alpha": a, "beta": b,
      "prior_mean": a / b}
     for (fx, ph), (a, b) in priors.items()])
mean_rate.to_csv(fr"{ROOT}\results_seg\game_day_propensity_hier_phase_rates.csv", index=False)

# ---- level 2: shrinkage toward segment grand mean (random intercept) ----
seg_grand = (tot_bin.groupby(["fixture", "road_osmid"], as_index=False)
                    .agg(E_seg=("E", "sum"), T_seg=("Texp", "sum")))
seg_grand["rate_seg"] = seg_grand.E_seg / seg_grand.T_seg.clip(lower=1e-12)
# segment-level prior
out_rows = []
for fx, gseg in seg_grand.groupby("fixture"):
    r = gseg.rate_seg.values[gseg.rate_seg.values > 0]
    a_seg, b_seg = fit_gamma_mom(r)
    for ph in PH:
        a, b = priors[(fx, ph)]
        g = tot_bin[(tot_bin.fixture == fx) & (tot_bin.phase == ph)][
            ["road_osmid", "E", "Texp"]]
        # posterior: two-stage precision-weighted blend of
        #  phase posterior mean (E+a)/(T+b)  and  segment shrunk mean
        post1 = (g.E.values + a) / (g.Texp.values + b)
        # segment shrunk toward its own posterior under segment prior
        segmap = (gseg.road_osmid.values, gseg.T_seg.values, gseg.E_seg.values)
        sm = pd.Series(segmap[2], index=segmap[0])
        sT = pd.Series(segmap[1], index=segmap[0])
        shrunk_seg = (sm.loc[g.road_osmid].values + a_seg) / (sT.loc[g.road_osmid].values + b_seg)
        blend_w = 1.0 / (1.0 + b * g.Texp.values)   # more exposure -> keep phase-specific
        rate = blend_w * post1 + (1 - blend_w) * shrunk_seg
        out_rows.append(pd.DataFrame({
            "fixture": fx, "phase": ph, "road_osmid": g.road_osmid.values,
            "rate_raw": g.E.values / g.Texp.values, "rate_seg_shrunk": shrunk_seg,
            "rate_phase_post": post1, "rate_hier": rate,
            "raw_E": g.E.values, "Texp": g.Texp.values}))
hier = pd.concat(out_rows, ignore_index=True)
hier.to_csv(fr"{ROOT}\results_seg\game_day_propensity_hier_segment_rates.csv", index=False)

# ---- rank concordance across phases under hierarchical rates ----
rows = []
def ktau_pairs(x, y, k=200):
    xs = x.nlargest(k).rank(); ys = y.nlargest(k).rank()
    common = xs.index.intersection(ys.index)
    d = xs[common].values - ys[common].values
    return 1 - 3 * np.mean(d ** 2) / (k ** 2 - 1)

for fx, g in hier.groupby("fixture"):
    P = g.pivot_table(index="road_osmid", columns="phase", values="rate_hier")
    for i in range(2):
        rows.append({"fixture": fx, "cmp": f"{PH[i]}_vs_{PH[i+1]}",
                     "tau_hier_rate_top200": round(ktau_pairs(P[PH[i]], P[PH[i+1]]), 3),
                     "top10_jaccard_hier": round(len(P[PH[i]].nlargest(10).index &
                                                  P[PH[i+1]].nlargest(10).index) / 10, 3),
                     "tau_raw_top200": round(ktau_pairs(
                         tot_bin[(tot_bin.fixture == fx) & (tot_bin.phase == PH[i])]
                         .set_index("road_osmid").E.clip(lower=1e-6),
                         tot_bin[(tot_bin.fixture == fx) & (tot_bin.phase == PH[i+1])]
                         .set_index("road_osmid").E.clip(lower=1e-6)), 3)})
tau_df = pd.DataFrame(rows)
tau_df.to_csv(fr"{ROOT}\results_seg\game_day_propensity_hier_rank_tau.csv", index=False)
print(tau_df.to_string(index=False))

# ---- mobility value: phased ranking (hier rate per phase) ----
# selection strategy A: static sites from day-aggregate volume ranking
# strategy B: per-phase ranking on rate_hier, top-P sites per phase
strat = []
for fx, g in df.groupby("fixture"):
    B = g.pivot_table(index="rel_bin", columns="road_osmid",
                      values="pred_speeding", aggfunc="sum").fillna(0.0)
    totV = B.values.sum()
    day_tot = B.sum(0)   # volume ranking basis
    H = hier[hier.fixture == fx].pivot_table(
        index="road_osmid", columns="phase", values="rate_hier")
    static_top = day_tot.nlargest(6).index
    cov_static = B[static_top].values.sum() / totV
    cov_phase = []
    moves = 0
    for ph, q in zip(PH, np.array_split(np.arange(0, 28), 3)):
        qb = B.iloc[q]
        topH = H[ph].nlargest(6).index
        cov_phase.append(qb[topH].values.sum() / qb.values.sum())
        moves += len(set(topH) - set(static_top))
    strat.append({"fixture": fx,
                  "cov_static_volume_pct": round(100 * cov_static, 2),
                  "cov_phased_hier_rate_pct": round(100 * np.mean(cov_phase), 2),
                  "gain_pp": round(100 * (np.mean(cov_phase) - cov_static), 3),
                  "relocations": moves})
mv = pd.DataFrame(strat)
mv.to_csv(fr"{ROOT}\results_seg\game_day_propensity_mobility_value.csv", index=False)
pd.set_option("display.width", 200)
print()
print(mv.to_string(index=False))
