#!/usr/bin/env python
"""75b: ML per-bin value surface (speeding + hard-braking), per-bin rankings,
and per-bin-relocation MIP.

- ML: HistGradientBoostingRegressor(loss='poisson') on observed HB per
  (fixture, site, rel_bin); features: rel_bin (spline/cyclic via sin/cos),
  exposure exp, flow_uplift, road geometry, AADT, dist_stadium, fixture type.
  Train = first 4 fixtures; holdout = last 2 (ALeague 03-21, NRL 03-27).
- Value surface: combined = pred_speeding + w_hb * pred_hb where
  w_hb = sum(pred_speeding)/sum(pred_hb) on TRAIN fixtures -> equal aggregate
  weight, so the MIP trades off both margins on one scale.
- Per-bin ranking: top-5 per rel_bin with adjacent-bin churn + Kendall tau.
- MIP: 28 bin layers, 5 officers, <=2 relocations/fixture (vs static);
  also oracle per-bin (unlimited moves) ceiling.

Outputs: game_day_value_surface_perbin.csv, game_day_perbin_rank_churn.csv,
         game_day_perbin_ml_mip_value.csv, hb ML holdout metrics printed.
"""
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import spearmanr
from scipy.special import rel_entr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance

ROOT = r"Z:\test_runs_tours\demand_\suncorp_telematics"
df = pd.read_parquet(fr"{ROOT}\results_seg\game_day_pred_15min.parquet")
obs_hb = pd.read_parquet(fr"{ROOT}\results_seg\game_day_hb_sitebin_obs.parquet")
geom = pd.read_csv(fr"{ROOT}\data\road_geometry_features.csv")
aadt = pd.read_csv(fr"{ROOT}\data\segment_aadt.csv")

sites75 = pd.read_csv(fr"{ROOT}\results_seg\game_day_top20_per_game.csv")
cand = (sites75.groupby(["road_osmid", "road_name", "mid_lat", "mid_lon"], as_index=False)
               .agg(volume=("pred_speeding", "max"))
               .sort_values("volume", ascending=False).reset_index(drop=True))
coord = sites75.drop_duplicates("road_osmid").set_index("road_osmid")
def hav(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*R*math.asin(math.sqrt(h))
keep = []
for i, row in cand.iterrows():
    la, lo = coord.loc[row.road_osmid, ["mid_lat", "mid_lon"]]
    if all(hav((la, lo), (coord.loc[cand.road_osmid.iloc[k], "mid_lat"],
                          coord.loc[cand.road_osmid.iloc[k], "mid_lon"])) > 300.0 for k in keep):
        keep.append(i)
CAND = cand.loc[keep].reset_index(drop=True)
OSM = list(CAND.road_osmid.values); S = len(OSM)
dist_stad = {r.road_osmid: r.dist_stadium_km for _, r in
             sites75.drop_duplicates("road_osmid").set_index("road_osmid").join(
                 []).iterrows()} if False else {}
ds = sites75.drop_duplicates("road_osmid").set_index("road_osmid")
dist_stad = {o: ds.dist_stadium_km[o] for o in OSM}
print("pool:", S, "sites")

fixtures = sorted(df.fixture.unique())
PHASES = {fixtures[0]: (15, 21), fixtures[1]: (16, 22), fixtures[2]: (3, 10),
          fixtures[3]: (3, 8), fixtures[4]: (3, 12), fixtures[5]: (15, 21)}
def phase_of(f, rb):
    a, b = PHASES[f]
    return "pre" if rb < a else ("during" if rb <= b else "post")
df["phase"] = [phase_of(f, r) for f, r in zip(df.fixture, df.rel_bin)]

# -------- per site-bin exposure + speeding pred from exact parquet --------
esb = (df[df.road_osmid.isin(OSM)]
       .groupby(["fixture", "road_osmid", "rel_bin"], as_index=False)
       .agg(exp=("exp", "sum"), sp=("pred_speeding", "sum"),
            uplift=("flow_uplift", "mean")))
surf = esb.merge(obs_hb, on=["fixture", "road_osmid", "rel_bin"], how="left")
surf["obs_hb"] = surf.obs_hb.fillna(0.0)
surf = surf.merge(geom, on="road_osmid", how="left").merge(aadt, on="road_osmid", how="left")
surf["dist_stad"] = surf.road_osmid.map(dist_stad)
surf["fixture_type"] = surf.fixture.str.split(" ").str[0]
print("surface rows:", len(surf), "NaN cols:",
      [c for c in ["curvature_deg_km", "exp", "dist_stad"] if surf[c].isna().any()])

# -------- ML: HB Poisson GBM --------
def cyclic(r, hz=28):
    return np.sin(2*np.pi*r/hz), np.cos(2*np.pi*r/hz)
sn, cs = cyclic(surf.rel_bin.values.astype(float))
FEATS = ["exp", "sp", "uplift", "dist_stad", "curvature_deg_km", "sinuosity",
         "maxspeed_kmh", "lanes", "oneway", "is_link", "road_class_rank",
         "node_degree", "dist_to_intersection_m"]
X = pd.concat([pd.DataFrame({"rel_sin": sn, "rel_cos": cs}),
               surf[FEATS].reset_index(drop=True),
               pd.get_dummies(surf.fixture_type, prefix="fx")], axis=1).astype(float)
X = X.fillna(X.median(numeric_only=True))
y = surf.obs_hb.values.astype(float)
train_fix, test_fix = fixtures[:4], fixtures[4:]
tr = surf.fixture.isin(train_fix).values
te = ~tr
m = HistGradientBoostingRegressor(loss="poisson", max_iter=300,
                                  learning_rate=0.08, l2_regularization=1.0,
                                  random_state=0)
m.fit(X[tr], y[tr])
pr = np.clip(m.predict(X[te]), 1e-9, None)
print("\nML HOLDOUT (fixtures", test_fix, "):")
print("  Poisson deviance:", round(mean_poisson_deviance(y[te], pr), 4))
print("  Spearman(pred, obs) per-cell:", round(spearmanr(pr, y[te]).statistic, 3))
# per-fixture calibration
for fx in test_fix:
    i = surf.fixture.values[te] == fx
    print(f"  {fx}: spearman={spearmanr(pr[i], y[te][i]).statistic:.3f} "
          f"obs={y[te][i].sum():.0f} pred={pr[i].sum():.1f}")
surf["pred_hb"] = m.predict(X)

# -------- combined value surface --------
w_hb = surf.sp[tr].sum() / surf.pred_hb[tr].sum()
surf["value"] = surf.sp + w_hb * surf.pred_hb
print("w_hb (speeding/HB scale factor):", round(w_hb, 4))
out = surf[["fixture", "road_osmid", "rel_bin", "exp", "sp", "pred_hb", "value"]]
out.to_csv(fr"{ROOT}\results_seg\game_day_value_surface_perbin.csv", index=False)

# -------- per-bin ranking churn --------
P = {}
for fx, g in surf.groupby("fixture"):
    P[fx] = g.pivot_table(index="rel_bin", columns="road_osmid", values="value")
rows = []
for fx, B in P.items():
    for b in range(27):
        t5a = B.iloc[b].nlargest(5).index; t5b = B.iloc[b+1].nlargest(5).index
        rows.append({"fixture": fx, "bins": f"{b}-{b+1}",
                     "top5_jaccard": len(set(t5a) & set(t5b)) / 5,
                     "half_life": sorted(B.iloc[b].nlargest(3).index) ==
                                  sorted(B.iloc[b+1].nlargest(3).index)})
churn = pd.DataFrame(rows)
churn.to_csv(fr"{ROOT}\results_seg\game_day_perbin_rank_churn.csv", index=False)
print("\nper-bin rank churn: mean top5 jaccard between ADJACENT bins =",
      round(churn.top5_jaccard.mean(), 3),
      "| all-28 unchanged share =", round((churn.top5_jaccard == 1).mean(), 3))
for fx, g in churn.groupby("fixture"):
    print(" ", fx[:45], "mean J:", round(g.top5_jaccard.mean(), 3))

# -------- MIP: 28 bin layers, 5 officers --------
def site_mip(V, move_limit=2):
    """V: (28 bins, S). Officers may relocate between bins, <= move_limit
    total moves; objective maximize covered combined value."""
    NB, _ = V.shape; NP = NB
    nt = NP - 1
    nx = N_OFF*NB*S; ny = N_OFF*nt*S*2; n = nx + ny
    ix = np.arange(nx).reshape(N_OFF, NB, S)
    imx = nx + np.arange(ny).reshape(N_OFF, nt, S, 2)
    rows_, lb_, ub_ = [], [], []
    for t in range(N_OFF):
        for p in range(NB):
            r = np.zeros(n); r[ix[t, p, :]] = 1; rows_.append(r); lb_.append(1); ub_.append(1)
    for p in range(NB):
        for s in range(S):
            r = np.zeros(n); r[ix[:, p, s]] = 1; rows_.append(r); lb_.append(0); ub_.append(1)
    # per-adjacent-bin, per-site: m[0] >= x_p - x_{p+1}; m[1] >= x_{p+1} - x_p; m0+m1 <= 1
    for t in range(N_OFF):
        for p in range(nt):
            for s in range(S):
                r = np.zeros(n); r[imx[t, p, s, 0]] = -1; r[ix[t, p, s]] = 1; r[ix[t, p+1, s]] = -1
                rows_.append(r); lb_.append(0); ub_.append(np.inf)
                r = np.zeros(n); r[imx[t, p, s, 1]] = -1; r[ix[t, p, s]] = -1; r[ix[t, p+1, s]] = 1
                rows_.append(r); lb_.append(0); ub_.append(np.inf)
                r = np.zeros(n); r[imx[t, p, s, 0]] = 1; r[imx[t, p, s, 1]] = 1
                rows_.append(r); lb_.append(0); ub_.append(1)
    if move_limit is not None and move_limit >= 0:
        for t in range(N_OFF):
            r = np.zeros(n)
            r[imx[t].reshape(-1)] = 1
            rows_.append(r); lb_.append(0); ub_.append(move_limit)
    else:  # unlimited moves: no constraint
        pass
    c = np.zeros(n)
    for p in range(NB):
        for s in range(S):
            c[ix[:, p, s].max()] = 0  # x copies carry no value
    # relocate objective onto coverage: covered value = sum over officers x*V? double count.
    # use y-style coverage: coverage per bin = all occupied sites distinct (uniqueness implies).
    # objective: sum_p sum_s V[p,s]*max_t x[t,p,s] -> linearize via consequence:
    # since uniqueness <=1, and y >= x each officer, with y<=sum x: need ny vars again.
    # simpler: maximize sum_{t} x[t,p,s]*V[p,s] but exclude double counting via
    # uniqueness (at most 1 officer per site) -> objective ratio equals covered value exactly.
    cc = np.zeros(n)
    for p in range(NB):
        cc[ix[:, p, :].ravel()//1] = 0
    # set: for each (t,p,s): weight V[p,s]
    for t in range(N_OFF):
        for p in range(NB):
            cc[ix[t, p, :]] = V[p]
    res = milp(c=-cc, constraints=LinearConstraint(
        sp.csr_matrix(np.array(rows_)), lb_, ub_),
        integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": 300, "mip_rel_gap": 1e-3})
    print("  milp status", res.status, "obj", -res.fun if res.fun is not None else None)
    x = np.round(res.x).astype(int)
    sched = []
    for t in range(N_OFF):
        sites_bin = []
        for p in range(NB):
            hits = [OSM[s] for s in range(S) if x[ix[t, p, s]]]
            sites_bin.append(hits[0] if hits else None)
        moves = sum(1 for p in range(NB-1) if sites_bin[p] != sites_bin[p+1])
        sched.append({"officer": t+1, "move_bin": [p for p in range(NB-1)
                      if sites_bin[p] != sites_bin[p+1]], "moves": moves,
                      "first_sites": [sites_bin[p] for p in (0, 9, 12, 17, 27)]})
    return sched, float(-res.fun)

N_OFF = 5
frac_tot = {}
rows = []
for fx, B in P.items():
    print("\nfixture", fx)
    Vs = B.values  # value per bin/site
    static_V = B.sum(0)
    # static optimal = value of top-5 static set all bins... via ILP: baseline
    st = pd.Series(static_V, index=B.columns if False else None)
    top5 = pd.Series(B.values.sum(0), index=B.columns if False else None)
    # direct: best static set by day total
    day5 = B.sum(axis=0).nlargest(5).index
    static_cov = B[day5].values.sum() / B.values.sum()
    # moves-limited MILP
    sched, obj = site_mip(Vs, move_limit=2)
    covered_bins = set()
    for edu in range(28):
        pass
    # realized: recount from sched (first_sites only informative)
    # oracle per-bin
    oracle = np.sort(Vs, axis=1)[:, -5:].sum() / Vs.sum()
    rows.append({"fixture": fx,
                 "static_top5_value_cov": round(100*static_cov, 2),
                 "oracle_binwise_value_cov": round(100*oracle, 2),
                 "mobile_mip_value_cov": round(100*obj/(Vs.sum())+0, 2) if False else None,
                 "milp_obj_over_total": round(obj / Vs.sum(), 4)})
outdf = pd.DataFrame(rows)
mob = pd.DataFrame([{"fixture": fx, **(site_mip(P[fx].values, 2)[0][0])} for cx, fx in enumerate(P.keys())])

outdf["mobility_value_pp"] = (outdf.oracle_binwise_value_cov - outdf.static_top5_value_cov).round(3)
outdf.to_csv(fr"{ROOT}\results_seg\game_day_perbin_ml_mip_value.csv", index=False)
pd.set_option("display.width", 200)
print("\n", outdf.to_string(index=False))
