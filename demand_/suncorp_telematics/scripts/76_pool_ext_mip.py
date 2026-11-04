#!/usr/bin/env python
"""76: EXPANDED candidate pool (top-56 by combined HB+speeding value per phase)
+ per-bin rankings + per-bin relocation MIP on the widened pool.

Steps:
1) Superset: top-400 roads by day-aggregate speeding prediction (parquet).
2) Snap observed HB events (fixture windows only) to all superset polylines.
   Combined per-phase value = pred_speeding + w_hb0 * snapped HB counts
   (w_hb0 = ratio of train-fixture totals, computed on the overlap).
3) Pool = per-phase top-56 combined, deduplicated + 300 m-separated
   (polyline centroids).
4) ML HB model retrained on the expanded pool (train 4 fixtures, holdout 2);
   value surface = pred_speeding + w_hb * pred_hb per (site, bin).
5) Per-bin ranking churn (adjacent-bin top-5 Jaccard + Kendall tau).
6) MIP = 28 bin layers x 5 officers x pool, relocation budget <=2/officer,
   vs static; plus oracle (binwise top-5).

Outputs: game_day_hb_sitebin_obs_ext.parquet, game_day_pool_ext.csv,
   game_day_value_surface_ext.csv, game_day_perbin_churn_ext.csv,
   game_day_perbin_mip_ext.csv
"""
import json, math
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import shapely

ROOT = r"Z:\test_runs_tours\demand_\suncorp_telematics"
df = pd.read_parquet(fr"{ROOT}\results_seg\game_day_pred_15min.parquet")
fixtures = sorted(df.fixture.unique())
PHASES = {fixtures[0]: (15, 21), fixtures[1]: (16, 22), fixtures[2]: (3, 10),
          fixtures[3]: (3, 8), fixtures[4]: (3, 12), fixtures[5]: (15, 21)}
PH = ["pre", "during", "post"]
def phase_of(f, r):
    a, b = PHASES[f]
    return "pre" if r < a else ("during" if r <= b else "post")
df["phase"] = [phase_of(f, r) for f, r in zip(df.fixture, df.rel_bin)]
ko = {fx: pd.to_datetime(fx.split(" @ ")[-1], format="%Y-%m-%d %H:%M") for fx in fixtures}

def hav(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*R*math.asin(math.sqrt(h))

gj = json.load(open(fr"{ROOT}\data\osm_roads_suncorp.geojson"))
pinfo = {f["properties"]["osmid"]: {
            "pl": LineString(f["geometry"]["coordinates"]),
            "ll": LineString(f["geometry"]["coordinates"]).interpolate(0.5, normalized=True),
            "name": f["properties"].get("name")}
         for f in gj["features"]}
print("geojson features:", len(pinfo))

# ---- 1) superset: top-400 by day speeding ----
day = df.groupby("road_osmid").pred_speeding.sum().nlargest(400)
SUPER = list(day.index)
super_ll = {o: (pinfo[o]["ll"].y, pinfo[o]["ll"].x) for o in SUPER if o in pinfo}
SUPER = list(super_ll)
print("superset with coords:", len(SUPER))
# 300 m separated superset to keep snap work down? keep all for ranking

# ---- 2) snap HB events within windows ----
hbv = pd.read_csv(fr"{ROOT}\data\hard_braking_events.csv",
                  usecols=["EventStart", "SnappedLatitude", "SnappedLongitude"],
                  parse_dates=["EventStart"])
rows = []
for fx, k in ko.items():
    t0, t1 = k - pd.Timedelta(hours=3), k + pd.Timedelta(hours=4)
    idx = np.where((hbv.EventStart.values >= t0.to_datetime64()) &
                   (hbv.EventStart.values < t1.to_datetime64()))[0]
    pts = np.column_stack([hbv.SnappedLongitude.values[idx],
                           hbv.SnappedLatitude.values[idx]])
    pw = shapely.points(pts)
    rb = ((hbv.EventStart.values[idx] - t0.to_datetime64())
          / np.timedelta64(15, "m")).astype(int)
    inwin = (rb >= 0) & (rb <= 27)
    pw, rb = pw[inwin], rb[inwin]
    print(fx, "window points:", len(pw))
    for o in SUPER:
        d = shapely.distance(pw, pinfo[o]["pl"])
        m = d < 75.0
        if m.sum():
            s = pd.DataFrame({"rel_bin": rb[m]})
            s["fixture"] = fx; s["road_osmid"] = o
            rows.append(s.groupby(["fixture", "road_osmid", "rel_bin"]).size()
                        .rename("obs_hb").reset_index())
obs = pd.concat(rows, ignore_index=True)
obs.to_parquet(fr"{ROOT}\results_seg\game_day_hb_sitebin_obs_ext.parquet")
print("snapped cells:", len(obs), "events:", obs.obs_hb.sum())

# ---- 3) per-phase combined rank -> pool of 56 ----
esb = (df[df.road_osmid.isin(SUPER)]
       .groupby(["fixture", "phase", "road_osmid"], as_index=False)
       .agg(exp=("exp", "sum"), sp=("pred_speeding", "sum")))
hbph = (obs.merge(df.drop_duplicates("fixture").assign(ko_dummy=1)[["fixture"]], on="fixture", how="left")
           .groupby(["fixture", "road_osmid"], as_index=False).obs_hb.sum())
obs_ph = []
for fx, k in ko.items():
    w = obs[obs.fixture == fx].copy()
    w["phase"] = [phase_of(fx, r) for r in w.rel_bin]
    obs_ph.append(w.groupby(["fixture", "phase", "road_osmid"], as_index=False).obs_hb.sum())
obs_ph = pd.concat(obs_ph)
sp_ph = (df[df.road_osmid.isin(SUPER)]
         .groupby(["fixture", "phase", "road_osmid"], as_index=False)
         .pred_speeding.sum())
tot_sp_tr = 0.0; tot_hb_tr = 0.0
for fx in fixtures[:4]:
    tot_sp_tr += df[df.fixture == fx].pred_speeding.sum()
    tot_sp_tr += 0
# train scale: use snapped HB totals on train fixtures (only those with window obs)
tot_hb_tr = obs[obs.fixture.isin(fixtures[:4])].obs_hb.sum()
tot_sp_tr = df[df.fixture.isin(fixtures[:4])].pred_speeding.sum()
w_hb0 = tot_sp_tr / max(tot_hb_tr, 1)
# NOTE: w_hb is vs counts on FULL network for speeding; HB counts here are pool-local,
# so the ratio is done at comparable pool levels below instead (w set to 1 then rebalanced
# inside the value computation). We rank by (sp_z + value_hb) per phase with equal scales:
comb = sp_ph.merge(obs_ph, on=["fixture", "phase", "road_osmid"], how="left").fillna({"obs_hb": 0})
# z-normalise each margin per phase across the superset
for col in ["pred_speeding", "obs_hb"]:
    comb[col + "_z"] = comb.groupby(["fixture", "phase"])[col].transform(
        lambda v: (v - v.mean()) / (v.std() + 1e-12))
comb["value"] = comb.pred_speeding_z + comb.obs_hb_z
rank = (comb.groupby(["phase", "road_osmid"], as_index=False)
            .value.mean()     # mean across fixtures
            .sort_values("value", ascending=False))
pool, taken = [], []
for _, r in rank.iterrows():
    o = r.road_osmid
    if o not in pinfo:
        continue
    ll = (pinfo[o]["ll"].y, pinfo[o]["ll"].x)
    if all(hav(ll, token[1]) > 300.0 for token in taken):
        pool.append((o, ll)); taken.append((o, ll))
    if len(pool) >= 56:
        break
pool = pd.DataFrame(pool, columns=["road_osmid", "ll"])
pool["mid_lat"] = [l[0] for l in pool.ll]
pool["mid_lon"] = [l[1] for l in pool.ll]
pool["road_name"] = [pinfo[o]["name"] for o in pool.road_osmid]
POOL = list(pool.road_osmid.values)
pool.drop(columns="ll").to_csv(fr"{ROOT}\results_seg\game_day_pool_ext.csv", index=False)
print("expanded pool:", len(pool), "sites")

# ---- 4) ML on expanded pool ----
geom = pd.read_csv(fr"{ROOT}\data\road_geometry_features.csv")
aadt = pd.read_csv(fr"{ROOT}\data\segment_aadt.csv")
esb2 = (df[df.road_osmid.isin(POOL)]
        .groupby(["fixture", "road_osmid", "rel_bin"], as_index=False)
        .agg(exp=("exp", "sum"), sp=("pred_speeding", "sum"), uplift=("flow_uplift", "mean")))
surf = esb2.merge(obs, on=["fixture", "road_osmid", "rel_bin"], how="left")
surf["obs_hb"] = surf.obs_hb.fillna(0.0)
surf = surf.merge(geom, on="road_osmid", how="left").merge(aadt, on="road_osmid", how="left")
# stadium coordinates (Suncorp Stadium, Milton)
STAD = (-27.4989, 152.9919)
pool_ll = pool.set_index("road_osmid")[["mid_lat", "mid_lon"]]
surf["dist_stad"] = surf.road_osmid.map(
    pool_ll.apply(lambda r: hav((r.mid_lat, r.mid_lon), STAD) / 1000.0, axis=1))
surf["fixture_type"] = surf.fixture.str.split(" ").str[0]
sn_, cs_ = np.sin(2*np.pi*surf.rel_bin.values/28), np.cos(2*np.pi*surf.rel_bin.values/28)
FEATS = ["exp", "sp", "uplift", "dist_stad", "curvature_deg_km", "sinuosity",
         "maxspeed_kmh", "lanes", "oneway", "is_link", "road_class_rank",
         "node_degree", "dist_to_intersection_m"]
X = pd.concat([pd.DataFrame({"rel_sin": sn_, "rel_cos": cs_}),
               surf[FEATS].reset_index(drop=True),
               pd.get_dummies(surf.fixture_type, prefix="fx")], axis=1).astype(float)
X = X.fillna(X.median(numeric_only=True))
y = surf.obs_hb.values.astype(float)
tr = surf.fixture.isin(fixtures[:4]).values; te = ~tr
m = HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.08,
                                  l2_regularization=1.0, random_state=0)
m.fit(X[tr], y[tr])
prh = np.clip(m.predict(X[te]), 1e-9, None)
print("\nML HOLDOUT (", fixtures[4:], "):")
print("  Poisson deviance:", round(mean_poisson_deviance(y[te], prh), 4),
      "Spearman:", round(spearmanr(prh, y[te]).statistic, 3))
for fx in fixtures[4:]:
    i = surf.fixture.values[te] == fx
    print(f"  {fx}: spearman={spearmanr(prh[i], y[te][i]).statistic:.3f} "
          f"obs={y[te][i].sum():.0f} pred={prh[i].sum():.1f}")
surf["pred_hb"] = m.predict(X)

w_hb = surf.sp[tr].sum() / surf.pred_hb[tr].sum()
surf["value"] = surf.sp + w_hb * surf.pred_hb
print("w_hb:", round(w_hb, 5))
surf[["fixture", "road_osmid", "rel_bin", "exp", "sp", "pred_hb", "value"]].to_csv(
    fr"{ROOT}\results_seg\game_day_value_surface_ext.csv", index=False)

# ---- 5) per-bin churn ----
P = {fx: g.pivot_table(index="rel_bin", columns="road_osmid", values="value")
     for fx, g in surf.groupby("fixture")}
rows = []
for fx, B in P.items():
    for b in range(27):
        t5a = set(B.iloc[b].nlargest(5).index); t5b = set(B.iloc[b+1].nlargest(5).index)
        t10a = set(B.iloc[b].nlargest(10).index); t10b = set(B.iloc[b+1].nlargest(10).index)
        rows.append({"fixture": fx, "bin": b,
                     "top5_j": len(t5a & t5b)/5, "top10_j": len(t10a & t10b)/10})
churn = pd.DataFrame(rows)
churn.to_csv(fr"{ROOT}\results_seg\game_day_perbin_churn_ext.csv", index=False)
print("\nadjacent-bin churn (expanded pool): top5 J =", round(churn.top5_j.mean(), 3),
      "top10 J =", round(churn.top10_j.mean(), 3),
      "fraction bins with any change (top5):", round((churn.top5_j < 1).mean(), 3))

# ---- 6) per-bin MIP ----
def solve_bins(V, move_limit):
    NB, SS = V.shape; nt = NB - 1
    nx = N_OFF*NB*SS; ny = N_OFF*nt*SS*2; n = nx + ny
    ix = np.arange(nx).reshape(N_OFF, NB, SS)
    imx = nx + np.arange(ny).reshape(N_OFF, nt, SS, 2)
    rows_, lb_, ub_ = [], [], []
    for t in range(N_OFF):
        for p in range(NB):
            r = np.zeros(n); r[ix[t, p, :]] = 1; rows_.append(r); lb_.append(1); ub_.append(1)
    for p in range(NB):
        for s in range(SS):
            r = np.zeros(n); r[ix[:, p, s]] = 1; rows_.append(r); lb_.append(0); ub_.append(1)
    for t in range(N_OFF):
        for p in range(nt):
            for s in range(SS):
                r = np.zeros(n); r[imx[t, p, s, 0]] = 1; r[ix[t, p, s]] = -1; r[ix[t, p+1, s]] = 1
                rows_.append(r); lb_.append(0); ub_.append(np.inf)
                r = np.zeros(n); r[imx[t, p, s, 1]] = 1; r[ix[t, p, s]] = 1; r[ix[t, p+1, s]] = -1
                rows_.append(r); lb_.append(0); ub_.append(np.inf)
                r = np.zeros(n); r[imx[t, p, s, 0]] = 1; r[imx[t, p, s, 1]] = 1
                rows_.append(r); lb_.append(0); ub_.append(1)
        if move_limit is not None:
            r = np.zeros(n); r[imx[t].reshape(-1)] = 1
            rows_.append(r); lb_.append(0); ub_.append(move_limit)
    c = np.zeros(n)
    for t in range(N_OFF):
        for p in range(NB):
            c[ix[t, p, :]] = V[p]
    res = milp(c=-c, constraints=LinearConstraint(
        sp.csr_matrix(np.array(rows_)), lb_, ub_),
        integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": 300, "mip_rel_gap": 1e-3})
    x = np.round(res.x).astype(int)
    sched = []
    for t in range(N_OFF):
        sb = [[POOL[s] for s in range(SS) if x[ix[t, p, s]]] for p in range(NB)]
        seq = [s_[0] if s_ else None for s_ in sb]
        moves = sum(1 for p in range(NB-1) if seq[p] != seq[p+1])
        sched.append({"officer": t+1, "moves": moves,
                      "move_bins": [p for p in range(NB-1) if seq[p] != seq[p+1]],
                      "seq_sample": seq[0::4]})
    return sched, float(res.fun), x

N_OFF = 5
rows = []
for fx, B in P.items():
    V = B.values
    Vs = V / V.sum()
    day5 = B.sum(axis=0).nlargest(5).index
    static = B[day5].values.sum() / V.sum()
    # static MILP equivalent: maximize with no moves allowed (move_limit=0)
    _, obj_st, _ = solve_bins(V, move_limit=0)
    _, obj_mb, _ = solve_bins(V, move_limit=2)
    oracle = np.sort(V, axis=1)[:, -5:].sum() / V.sum()
    st_pct = 100 * obj_st / V.sum()
    mb_pct = 100 * obj_mb / V.sum()
    rows.append({"fixture": fx, "static_cov_pct": round(abs(st_pct), 2),
                 "mobile2_cov_pct": round(abs(mb_pct), 2),
                 "oracle_cov_pct": round(100*oracle, 2),
                 "mobility_pp": round(abs(mb_pct - st_pct), 3),
                 "oracle_headroom_pp": round(100*oracle - abs(mb_pct), 3)})
    print(fx, "-> static", rows[-1]["static_cov_pct"],
          "| <=2 moves", rows[-1]["mobile2_cov_pct"],
          "| oracle", rows[-1]["oracle_cov_pct"])
out = pd.DataFrame(rows)
out.to_csv(fr"{ROOT}\results_seg\game_day_perbin_mip_ext.csv", index=False)
pd.set_option("display.width", 200)
print("\n", out.to_string(index=False))
