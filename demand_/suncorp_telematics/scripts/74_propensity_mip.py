#!/usr/bin/env python
"""Re-optimised game-day deployment MIP with the hierarchical phase-specific
propensity surface as the objective (vs the day-aggregate volume objective).

Per fixture (scipy.optimize.milp), 300 m-separated candidate sites from the
60d top-20 union:
  x[t,p,s] binary: officer t stationed at site s in phase p (<=1 site/phase,
                   uniqueness 1 officer per site/phase)
  y[p,s]   = site covered
  m[t,j]   aux for officer moves, <= 2 moves total per officer per game
Strategies (objective V):
  volume_static : V = day-aggregate pred counts, officers pinned to same
                  site all phases (site-set equality across phases)
  volume_phased : V = per-phase pred counts
  hier_prop     : V = per-phase Texp * rate_hier (propensity hierarchy)
All strategies evaluated on BOTH realized pred counts and propensity units.

Output: results_seg/game_day_propensity_mip_reschedule.csv + sites file.
"""
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = r"Z:\test_runs_tours\demand_\suncorp_telematics"
df = pd.read_parquet(fr"{ROOT}\results_seg\game_day_pred_15min.parquet")
sites75 = pd.read_csv(fr"{ROOT}\results_seg\game_day_top20_per_game.csv")
hier = pd.read_csv(fr"{ROOT}\results_seg\game_day_propensity_hier_segment_rates.csv")

PHASES = {
    "NRL Broncos-Panthers @ 2026-03-06 19:00": (15, 21),
    "NRL Broncos-Eels @ 2026-03-12 19:00": (16, 22),
    "SuperRugby Reds-Waratahs @ 2026-03-14 18:45": (3, 10),
    "NRL Dolphins-Titans @ 2026-03-15 17:15": (3, 8),
    "ALeague Roar-Wellington Phoenix @ 2026-03-21 14:00": (3, 12),
    "NRL Broncos-Dolphins @ 2026-03-27 19:00": (15, 21),
}
PH = ["pre", "during", "post"]
def phase_of(f, r):
    a, b = PHASES[f]
    return "pre" if r < a else ("during" if r <= b else "post")
df["phase"] = [phase_of(f, r) for f, r in zip(df.fixture, df.rel_bin)]

def haversine_m(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2 * R * math.asin(math.sqrt(h))

# ---- candidate pool: union of fixture top-20 (has coords), volume-ranked, 300 m apart
cand = (sites75.groupby(["road_osmid", "road_name", "mid_lat", "mid_lon"], as_index=False)
               .agg(volume=("pred_speeding", "max"))   # max across fixtures as quality key
               .sort_values("volume", ascending=False)
               .reset_index(drop=True))
chosen_idx = []
for i, row in cand.iterrows():
    pt = (row.mid_lat, row.mid_lon)
    if all(haversine_m(pt, (cand.mid_lat.iloc[k], cand.mid_lon.iloc[k])) > 300.0
           for k in chosen_idx):
        chosen_idx.append(i)
CAND = cand.loc[chosen_idx].reset_index(drop=True)
S = len(CAND); OSM = CAND.road_osmid.values
NS = S
print(f"candidate sites after volume-ranked 300 m separation: {S}")

N_OFF, NP = 5, 3
BINQ = [np.array_split(np.arange(0, 28), 3)]

def build_and_solve(V, phased):
    """maximize sum over phases of V[p,s] * y[p,s]."""
    nx = N_OFF * NP * S; ny = NP * S; nm = N_OFF * (NP - 1)
    n = nx + ny + nm
    ix = np.arange(nx).reshape(N_OFF, NP, S)
    iy = nx + np.arange(ny).reshape(NP, S)
    im = nx + ny + np.arange(nm).reshape(N_OFF, NP - 1)
    rows_, lb_, ub_ = [], [], []
    # each officer exactly one site per phase
    for t in range(N_OFF):
        for p in range(NP):
            r = np.zeros(n); r[ix[t, p, :]] = 1
            rows_.append(r); lb_.append(1); ub_.append(1)
    # one officer per site per phase
    for p in range(NP):
        for s in range(S):
            r = np.zeros(n); r[ix[:, p, s]] = 1
            rows_.append(r); lb_.append(0); ub_.append(1)
    # coverage activation (both directions: y>=x and y<=sum x)
    for p in range(NP):
        for s in range(S):
            for t in range(N_OFF):
                r = np.zeros(n); r[iy[p, s]] = 1; r[ix[t, p, s]] = -1
                rows_.append(r); lb_.append(0); ub_.append(np.inf)
            r = np.zeros(n); r[iy[p, s]] = 1
            r[ix[:, p, s]] = -1
            rows_.append(r); lb_.append(-np.inf); ub_.append(0)
    # move aux: m[t,j] >= |x[t,j+1,s] - x[t,j,s]| per s; total moves <= 2
    for t in range(N_OFF):
        for j in range(NP - 1):
            for s in range(S):
                r1 = np.zeros(n); r1[im[t, j]] = 1; r1[ix[t, j, s]] = -1; r1[ix[t, j + 1, s]] = 1
                rows_.append(r1); lb_.append(0); ub_.append(np.inf)  # m >= x_{p+1}-x_p
                r2 = np.zeros(n); r2[im[t, j]] = 1; r2[ix[t, j, s]] = 1; r2[ix[t, j + 1, s]] = -1
                rows_.append(r2); lb_.append(0); ub_.append(np.inf)  # m >= x_p-x_{p+1}
        rT = np.zeros(n); rT[im[t, :]] = 1
        rows_.append(rT); lb_.append(0); ub_.append(2)
        if not phased:  # static: officer t same site in every phase
            for s in range(S):
                for p in range(1, NP):
                    r = np.zeros(n); r[ix[t, p, s]] = 1; r[ix[t, 0, s]] = -1
                    rows_.append(r); lb_.append(0); ub_.append(0)
    c = np.zeros(n)
    if V.ndim == 2:
        for p in range(NP):
            c[iy[p, :]] = V[p]
    else:
        for p in range(NP):
            c[iy[p, :]] = V
    A = sparse.csr_matrix(np.array(rows_))
    lc = LinearConstraint(A, lb_, ub_)
    res = milp(c=-c, constraints=lc,
               integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
               options={"time_limit": 120, "mip_rel_gap": 1e-4})
    print("   milp:", res.status, res.message, "obj=", res.fun)
    solsites = []
    if res.x is not None:
        x = np.round(res.x).astype(int)
        for p in range(NP):
            sset = [OSM[s] for t in range(N_OFF) for s in range(S)
                    if x[ix[t, p, s]]]
            solsites.append(sset)
    return solsites, (0 if res.x is None else int(np.round(res.fun * 0)) or None)

rows = []
binq_split = np.array_split(np.arange(0, 28), 3)
for fx in df.fixture.unique():
    print("solving", fx)
    g = df[df.fixture == fx]
    B = g.pivot_table(index="rel_bin", columns="road_osmid",
                      values="pred_speeding", aggfunc="sum").reindex(columns=OSM).fillna(0.0)
    E = g.pivot_table(index="rel_bin", columns="road_osmid",
                      values="exp", aggfunc="sum").reindex(columns=OSM).fillna(0.0)
    Vcnt = np.array([B.iloc[q].values.sum(0) for q in binq_split])
    hfx = hier[hier.fixture == fx].pivot_table(
        index="road_osmid", columns="phase", values="rate_hier").reindex(OSM)
    Vprop = np.array([(E.iloc[q].values * hfx[ph].fillna(0).values).sum(0)
                      for ph, q in zip(PH, binq_split)])
    day_tot = Vcnt.sum(0)

    strats = {"volume_static": np.tile(day_tot, (NP, 1)),
              "volume_phased": Vcnt,
              "hier_propensity": Vprop}
    for name, V in strats.items():
        phased = not name.endswith("static")
        solsites, _ = build_and_solve(V, phased)
        cov_cnt = np.mean([
            B.iloc[q][solsites[p]].values.sum() / B.iloc[q].values.sum()
            for p, q in enumerate(binq_split)])
        rp = []
        for p, (ph, q) in enumerate(zip(PH, binq_split)):
            val = E.iloc[q].values * hfx[ph].fillna(0).values
            idx = [int(np.where(OSM == s)[0][0]) for s in solsites[p]]
            rp.append(val[:, idx].sum() / val.sum())
        cov_prop = np.mean(rp)
        rows.append({"fixture": fx, "strategy": name,
                     "cov_count_pct": round(100 * cov_cnt, 2),
                     "cov_prop_pct": round(100 * cov_prop, 3),
                     "n_moves": sum(len(set(solsites[p+1]) - set(solsites[p])) for p in range(NP-1)),
                     "sites_pre": "|".join(map(str, solsites[0])),
                     "sites_during": "|".join(map(str, solsites[1])),
                     "sites_post": "|".join(map(str, solsites[2]))})

out = pd.DataFrame(rows)
out.to_csv(fr"{ROOT}\results_seg\game_day_propensity_mip_reschedule.csv", index=False)
pd.set_option("display.width", 240)
print(out[["fixture", "strategy", "cov_count_pct", "cov_prop_pct", "n_moves"]].to_string(index=False))
