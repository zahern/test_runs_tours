"""
43_finable_speeding.py - finable-speeding estimand for enforcement deployment.

Why: the inherited `speeding` column has NO defined threshold anywhere in
the repo (no script creates segment_roads_part0_5000.csv.gz; the 54
extractor does not even retain pure-speeding points). For a paper about
police maximising finable detections, the estimand must be offences, not
an unknown flag. This rebuilds it from first principles:

    exceedance = GPS Speed_kph - posted limit (OSM maxspeed, 50 default)
    finable    = MARGIN <= exceedance <= MAX_EXCEEDANCE

Margin logic: fleet GPS noise is ~2-5 kph on snapped points, so 10 clears
false positives. The upper bound excludes observations likely to reflect a
bad posted-limit match, GPS artefact, or non-comparable extreme event rather
than the bounded enforcement-occurrence estimand. Sensitivity at 5/15/20
km/h is reported within the same upper bound; excluded tails are preserved
for QA and are not silently treated as zero.

Then: convergent NB SPF + EB per segment, corridor ranking, assignment to
the P4 police posts, staffed-vs-random-vs-none detection coverage, and a
fixture-phase split (43 real fixtures) for roster timing.

Inputs: results/event_points_full.csv.gz (Speed + coords + dt),
        data/osm_roads_suncorp.geojson, data/road_geometry_features.csv,
        data/near_miss_hb_ecf.csv (exposure + highway + geometry),
        data/fixture_calendar.csv, results/post_selection_mip.csv
Outputs: results/finable_speeding_segments.csv, results/finable_speeding_corridors.csv,
         results/finable_speeding_coverage.csv, results/finable_speeding_phase.csv,
         results/finable_speeding.tex, results/finable_speeding_audit.json,
         figures/finable_corridors.png, figures/finable_coverage.png

Usage: python 43_finable_speeding.py [--margin 10] [--max-exceedance 30]
                                      [--staffed-p 5]
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    try:
        from statsmodels.tools.sm_exceptions import HessianInversionWarning
    except ImportError:
        HessianInversionWarning = Warning
    _HAS_STATSMODELS = True
    _STATSMODELS_ERROR = None
except Exception as e:  # noqa: BLE001 - SciPy/statsmodels may mismatch
    smf = None
    ConvergenceWarning = Warning
    HessianInversionWarning = Warning
    _HAS_STATSMODELS = False
    _STATSMODELS_ERROR = e

PROJ = os.environ.get(
    "SUNCORP_PROJ",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

EVENTS = os.path.join(RES, "event_points_full.csv.gz")
GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
FEATS = os.path.join(DATA, "road_geometry_features.csv")
SCREEN = os.path.join(DATA, "near_miss_hb_ecf.csv")
CAL = os.path.join(DATA, "fixture_calendar.csv")
POSTS = os.path.join(RES, "post_selection_mip.csv")
OUTLIER_EXCEEDANCES = os.path.join(RES, "finable_exceedances_outliers.csv.gz")

SNAP_M = 50.0
QLD_DEFAULT_LIMIT = 50.0  # urban default where OSM has no maxspeed tag
MAX_EXCEEDANCE_DEFAULT = 30.0

sns.set_style("whitegrid")


def standardise(df, cols):
    for c in cols:
        z = c + "_z"
        if c in df.columns and z not in df.columns:
            sd = float(df[c].std())
            df[z] = (df[c] - float(df[c].mean())) / (sd if sd and sd > 0 else 1.0)
    return df


class _PoissonSPFFallback:
    def __init__(self, columns, params, dummy_levels):
        self.columns = columns
        self.params = np.asarray(params, dtype=float)
        self.dummy_levels = dummy_levels

    def _design(self, data):
        x = pd.DataFrame({"const": 1.0}, index=data.index)
        x["log_exposure"] = np.log(
            pd.to_numeric(data["exposure_trips"], errors="coerce")
            .fillna(0.0) + 1.0)
        if self.dummy_levels:
            highway = data["highway"].fillna("unknown").astype(str)
            for level in self.dummy_levels:
                x[f"highway_{level}"] = (highway == level).astype(float)
        for name in self.columns:
            if name in ("const", "log_exposure") or name.startswith("highway_"):
                continue
            x[name] = pd.to_numeric(data[name], errors="coerce").fillna(0.0)
        return x.reindex(columns=self.columns, fill_value=0.0).to_numpy(float)

    def predict(self, data):
        x = self._design(data)
        return np.exp(np.clip(x @ self.params, -30.0, 30.0))


def _fit_poisson_spf_fallback(df, formula):
    from scipy.optimize import minimize
    from scipy.special import gammaln

    columns = ["const", "log_exposure"]
    dummy_levels = []
    if "C(highway)" in formula:
        levels = sorted(df["highway"].fillna("unknown").astype(str).unique())
        dummy_levels = levels[1:]
        columns.extend(f"highway_{level}" for level in dummy_levels)
    for name in ("maxspeed_kmh_z", "lanes_z", "curvature_deg_km_z",
                 "dist_to_intersection_m_z"):
        if name in formula:
            columns.append(name)
    model = _PoissonSPFFallback(columns, np.zeros(len(columns)), dummy_levels)
    x = model._design(df)
    y = pd.to_numeric(df["finable"], errors="coerce").fillna(0.0).to_numpy()
    beta0 = np.linalg.lstsq(x, np.log1p(y), rcond=None)[0]

    def objective(beta):
        eta = np.clip(x @ beta, -30.0, 30.0)
        return float(np.sum(np.exp(eta) - y * eta))

    def gradient(beta):
        eta = np.clip(x @ beta, -30.0, 30.0)
        return x.T @ (np.exp(eta) - y)

    fit = minimize(objective, beta0, jac=gradient, method="L-BFGS-B",
                   options={"maxiter": 2000})
    beta = np.asarray(fit.x if fit.success else beta0, dtype=float)
    model.params = beta
    lam = model.predict(df)
    loglik = float(np.sum(y * np.log(np.maximum(lam, 1e-12)) - lam
                         - gammaln(y + 1.0)))
    aic = 2.0 * len(beta) - 2.0 * loglik
    pearson = float(np.sum(((y - lam) ** 2) / np.maximum(lam, 1e-12))
                    / max(len(y) - len(beta), 1))
    alpha = float(max((pearson - 1.0) / max(float(np.mean(lam)), 1e-9),
                      1e-3))
    return model, alpha, formula + " [Poisson-fallback]", True, aic


def fit_convergent_spf(df, full, lean):
    if not _HAS_STATSMODELS:
        print(f"   statsmodels unavailable ({_STATSMODELS_ERROR}); "
              "using local Poisson fallback")
        return _fit_poisson_spf_fallback(df, full or lean)
    import statsmodels.api as sm
    for formula in (full, lean):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                warnings.simplefilter("error", HessianInversionWarning)
                res = smf.negativebinomial(formula, data=df).fit(
                    disp=False, method="lbfgs", maxiter=2000)
            converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
            alpha = float(res.params["alpha"]) if "alpha" in res.params else float(res.scale)
            if converged and np.isfinite(res.aic) and 0 < alpha <= 100:
                return res, alpha, formula, True, float(res.aic)
            print(f"   degenerate fit rejected (alpha={alpha:.3g}): {formula}")
        except Exception as e:  # noqa: BLE001
            print(f"   fit failed ({type(e).__name__}): {formula}")
            continue
    fam = sm.families.Poisson()
    res = smf.glm(lean, data=df, family=fam).fit(disp=False)
    pearson = float(((res.resid_pearson) ** 2).sum() / max(res.df_resid, 1))
    lam = np.asarray(res.fittedvalues, dtype=float)
    alpha = float(max((pearson - 1.0) / max(np.mean(lam), 1e-9), 1e-3))
    return res, alpha, lean + " [Poisson-GLM fallback]", True, float(res.aic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=10.0)
    ap.add_argument("--max-exceedance", type=float,
                    default=MAX_EXCEEDANCE_DEFAULT,
                    help="upper bound on km/h over posted limit; higher "
                         "values are retained in the QA tail only")
    ap.add_argument("--staffed-p", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    M, X, P = (float(args.margin), float(args.max_exceedance),
               int(args.staffed_p))
    if X < M:
        raise SystemExit("FATAL: --max-exceedance must be >= --margin")
    rng = np.random.default_rng(args.seed)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    audit = {
        "margin_kph": M,
        "max_exceedance_kph": X,
        "definition": "margin <= Speed_kph - OSM posted limit <= max_exceedance",
    }
    print("=" * 80)
    print(f"43_finable_speeding.py - bounded finable speeding "
          f"({M:.0f}-{X:.0f} km/h over limit)")
    print("=" * 80)

    # 1. Segments: geometry + limits
    print("\n[1/6] Loading segments + limits...")
    import geopandas as gpd
    from scipy.spatial import cKDTree
    g = gpd.read_file(GEOJSON)
    g["osmid"] = pd.to_numeric(g["osmid"], errors="coerce")
    g = g.dropna(subset=["osmid"])
    gu = g.to_crs(28356)
    mid = gu.geometry.centroid
    seg_xy = np.column_stack([mid.x.values, mid.y.values])
    seg_ids = g["osmid"].values.astype(np.int64)
    feat = pd.read_csv(FEATS)
    lim = dict(zip(feat["road_osmid"].values, feat["maxspeed_kmh"].values))
    seg_lim = np.array([lim.get(int(o), np.nan) for o in seg_ids], dtype=float)
    n_def = int(np.isfinite(seg_lim).sum())
    seg_lim[~np.isfinite(seg_lim)] = QLD_DEFAULT_LIMIT
    print(f"   segments={len(seg_ids):,} with OSM limit={n_def:,} default-50={len(seg_ids)-n_def:,}")
    audit.update({"n_seg": int(len(seg_ids)), "n_limit_default": int(len(seg_ids) - n_def)})
    tree = cKDTree(seg_xy)

    # 2. Fixture windows
    cal = pd.read_csv(CAL)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"])
    wins = [(r["label"], r["ko"] - pd.Timedelta(hours=3), r["ko"] - pd.Timedelta(minutes=30),
             r["ko"] + pd.Timedelta(hours=1.5), r["ko"] + pd.Timedelta(hours=4))
            for _, r in cal.iterrows()]
    print(f"   fixtures={len(wins)}")

    # 3. Stream events: snap, exceedance, flag, phase
    print("\n[2/6] Streaming events (snap + exceedance + phase)...")
    tot = {5: 0, 10: 0, 15: 0, 20: 0}
    seg_fin = {}
    seg_tot = {}
    phase_rows = []
    rec_list = []  # per-event exceedance records (POT severity + survival job)
    n_ev = n_snap = n_judge = n_high_tail = 0
    n_chunk = 0
    tail_list = []
    for ch in pd.read_csv(EVENTS, usecols=["dt", "SnappedLongitude", "SnappedLatitude", "Speed"],
                          parse_dates=["dt"], chunksize=500_000):
        n_chunk += 1
        print(f"   chunk {n_chunk}: rows={len(ch):,} judged_so_far={n_judge:,}", flush=True)
        ch["dt"] = pd.to_datetime(ch["dt"], errors="coerce")
        ch = ch.dropna(subset=["dt", "SnappedLongitude", "SnappedLatitude", "Speed"])
        if ch.empty:
            continue
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
            ch["SnappedLongitude"].values, ch["SnappedLatitude"].values), crs=4326).to_crs(28356)
        xy = np.column_stack([pts.geometry.x.values, pts.geometry.y.values])
        dd, ii = tree.query(xy, distance_upper_bound=SNAP_M)
        ok = np.isfinite(dd)
        ch = ch[ok].reset_index(drop=True)
        if ch.empty:
            continue
        lims = seg_lim[ii[ok]]
        exc = ch["Speed"].values - lims
        n_ev += len(ch)
        n_snap += len(ch)
        n_judge += len(ch)
        high_tail = exc > X
        n_high_tail += int(high_tail.sum())
        if high_tail.any():
            tail_list.append(pd.DataFrame({
                "road_osmid": seg_ids[ii[ok]][high_tail].astype(np.int64),
                "dt": ch.loc[high_tail, "dt"].values,
                "exceedance": exc[high_tail],
                "reason": "exceedance_above_max",
            }))
        for t in tot:
            tot[t] += int(((exc >= t) & (exc <= X)).sum())
        fin = (exc >= M) & (exc <= X)
        sids = seg_ids[ii[ok]][fin]
        for s in sids:
            seg_fin[int(s)] = seg_fin.get(int(s), 0) + 1
        for s in seg_ids[ii[ok]]:
            seg_tot[int(s)] = seg_tot.get(int(s), 0) + 1
        # fixture-phase split for finable events only
        fdt = ch.loc[fin, "dt"]
        fsid = seg_ids[ii[ok]][fin]
        ex_fin = exc[fin]
        flab = np.array([""] * len(fsid), dtype=object)
        fph = np.array([""] * len(fsid), dtype=object)
        for label, a, b, c, d in wins:
            m_pre = (fdt >= a) & (fdt < b)
            m_dur = (fdt >= b) & (fdt < c)
            m_pst = (fdt >= c) & (fdt < d)
            for m, ph in ((m_pre, "pre"), (m_dur, "during"), (m_pst, "post")):
                hit = m.values
                if hit.any():
                    flab[hit] = label
                    fph[hit] = ph
                    vv = pd.Series(fsid[hit]).value_counts()
                    for s, n in vv.items():
                        phase_rows.append((label, ph, int(s), int(n)))
        rec_list.append(pd.DataFrame({"road_osmid": fsid, "dt": fdt.values,
                                      "exceedance": ex_fin, "fixture": flab,
                                      "phase": fph}))
        print(f"   events snapped+judged={n_judge:,} bounded margin sensitivity: " +
            ", ".join(f">={t}kph:{n:,}" for t, n in tot.items()))
        audit.update({"n_events_judged": int(n_judge),
                "sensitivity": {f"ge_{t}kph": int(n) for t, n in tot.items()},
                "n_exceedance_above_max": int(n_high_tail),
                "share_exceedance_above_max": float(n_high_tail / max(n_judge, 1))})

    # Persist per-event exceedances (severity POT + survival job 44).
    exdf = (pd.concat(rec_list, ignore_index=True) if rec_list
            else pd.DataFrame(columns=["road_osmid", "dt", "exceedance", "fixture", "phase"]))
    exdf.to_csv(os.path.join(RES, "finable_exceedances.csv.gz"), index=False)
    print(f"   persisted {len(exdf):,} finable exceedance records")
    audit["n_exceedance_records"] = int(len(exdf))
    audit["share_exceedance_above_max_of_exceedances"] = float(
        n_high_tail / max(n_high_tail + len(exdf), 1))
    taildf = (pd.concat(tail_list, ignore_index=True) if tail_list
              else pd.DataFrame(columns=["road_osmid", "dt", "exceedance",
                                         "reason"]))
    taildf.to_csv(OUTLIER_EXCEEDANCES, index=False)
    print(f"   persisted {len(taildf):,} excluded high-tail records -> "
          f"{os.path.basename(OUTLIER_EXCEEDANCES)}")

    # POT-GPD severity: P(exceed >= m | exceed >= M) tail model.
    print("\n[POT] GPD severity fit...")
    from scipy.stats import genpareto
    xa = pd.to_numeric(exdf["exceedance"], errors="coerce").to_numpy(dtype=float)
    xa = xa[np.isfinite(xa) & (xa >= M)]
    pot = {"threshold_kph": M, "n_exceed": int(len(xa))}
    try:
        if len(xa) < 200:
            raise RuntimeError("too few exceedances for GPD")
        cc, _, ss = genpareto.fit(xa - M, floc=0)
        pot.update({"xi": float(cc), "sigma": float(ss), "fitted": True})
        print(f"   GPD xi={cc:.3f} sigma={ss:.2f} on n={len(xa):,}")
    except Exception as e:  # noqa: BLE001 - fall back to empirical tail
        pot.update({"fitted": False, "error": str(e)[:200]})
        print(f"   GPD fit failed ({e}); empirical tail only")
    audit["pot"] = pot

    # 4. SPF + EB on finable totals (exposure/geometry from screening table)
    print("\n[3/6] SPF + EB on finable speeding...")
    sc = pd.read_csv(SCREEN, usecols=["road_osmid", "road_name", "highway", "exposure_trips",
                                      "maxspeed_kmh", "lanes", "curvature_deg_km",
                                      "dist_to_intersection_m"])
    sc["finable"] = sc["road_osmid"].map(seg_fin).fillna(0.0)
    sc = sc[sc["exposure_trips"] > 0].copy()
    for c in ["maxspeed_kmh", "lanes", "curvature_deg_km", "dist_to_intersection_m"]:
        sc[c] = sc[c].fillna(float(sc[c].median()))
    sc = standardise(sc, ["maxspeed_kmh", "lanes", "curvature_deg_km",
                           "dist_to_intersection_m"])
    sc["finable"] = sc["road_osmid"].map(seg_fin).fillna(0.0)
    sc = sc[sc["exposure_trips"] > 0].copy()
    for c in ["maxspeed_kmh", "lanes"]:
        sc[c] = sc[c].fillna(float(sc[c].median()))
    sc = standardise(sc, ["maxspeed_kmh", "lanes"])
    sc["highway"] = sc["highway"].fillna("unknown").astype(str)
    full = ("finable ~ np.log(exposure_trips + 1) + C(highway) + maxspeed_kmh_z"
            " + lanes_z + curvature_deg_km_z + dist_to_intersection_m_z")
    lean = "finable ~ np.log(exposure_trips + 1) + lanes_z + dist_to_intersection_m_z"
    have = all(c in sc.columns for c in ["exposure_trips", "lanes_z",
                                         "dist_to_intersection_m_z", "maxspeed_kmh_z",
                                         "curvature_deg_km_z", "highway", "finable"])
    res, alpha, fu, conv, aic = fit_convergent_spf(sc, full if have else lean, lean)
    print(f"   {fu} converged={conv} AIC={aic:.0f} alpha={alpha:.3f}")
    if not conv:
        raise SystemExit("FATAL: finable SPF did not converge.")
    lam = np.asarray(res.predict(sc), dtype=float)
    lam = np.where(np.isfinite(lam), lam, 0.0)
    w = 1.0 / (1.0 + alpha * lam)
    y = sc["finable"].values
    sc["eb_finable"] = w * lam + (1.0 - w) * y
    obs, ebtot = float(y.sum()), float(sc["eb_finable"].sum())
    print(f"   observed finable(>={M:.0f})={obs:,.0f} EB={ebtot:,.0f} mean w={w.mean():.3f}")
    audit.update({"spf": fu, "aic": aic, "alpha": alpha, "obs_finable": obs,
                  "eb_finable": ebtot, "mean_w": float(w.mean())})

    # 5. Corridors + post assignment + coverage
    print("\n[4/6] Corridors + max-detection coverage...")
    if os.path.exists(META_CAND := os.path.join(DATA, "road_meta.csv")):
        meta = pd.read_csv(META_CAND)
        nc = "name" if "name" in meta.columns else "road_name"
        sc = sc.merge(meta[["osmid", nc]].rename(
            columns={"osmid": "road_osmid", nc: "meta_name"}), on="road_osmid", how="left")
        sc["road_name"] = sc["meta_name"].fillna(sc.get("road_name", "")).fillna("")
    sc["road_name"] = sc.get("road_name", "").fillna("").astype(str)
    named = sc[sc["road_name"].str.strip() != ""]
    corr = named.groupby("road_name").agg(
        n_seg=("road_osmid", "size"), eb=("eb_finable", "sum"),
        obs=("finable", "sum")).reset_index().sort_values("eb", ascending=False)
    corr.to_csv(os.path.join(RES, "finable_speeding_corridors.csv"), index=False)
    print(corr.head(12).to_string(index=False))
    posts = pd.read_csv(POSTS)
    plat, plon = posts["lat"].to_numpy(), posts["lon"].to_numpy()
    import geopandas as gpd2  # noqa: F401 (reuse import)
    pc = gpd.GeoDataFrame(geometry=gpd.points_from_xy(plon, plat), crs=4326).to_crs(28356)
    pxy = np.column_stack([pc.geometry.x.values, pc.geometry.y.values])
    sc = sc.merge(pd.DataFrame({"road_osmid": seg_ids,
                                "segx": seg_xy[:, 0], "segy": seg_xy[:, 1]}),
                  on="road_osmid", how="left")
    from scipy.spatial import cKDTree as KD2
    pt = KD2(pxy)
    dd2, ii2 = pt.query(np.column_stack([sc["segx"].fillna(0).values,
                                         sc["segy"].fillna(0).values]))
    sc["assigned_post"] = np.where(sc["segx"].notna(), ii2, -1)
    ben = sc[sc["assigned_post"] >= 0].groupby("assigned_post")["eb_finable"].sum()
    P = int(min(int(args.staffed_p), len(posts)))
    staffed = set(ben.sort_values(ascending=False).head(P).index.astype(int).tolist())
    sc["covered"] = sc["assigned_post"].isin(staffed)
    total = float(sc["eb_finable"].sum())
    cov = float(sc.loc[sc["covered"], "eb_finable"].sum() / max(total, 1))
    det = float(sc.loc[sc["covered"], "eb_finable"].sum())
    a = sc["assigned_post"].to_numpy()
    rand = []
    for _ in range(200):
        rnd = set(rng.choice(len(posts), size=P, replace=False).tolist())
        m = np.isin(a, list(rnd))
        rand.append(float(sc.loc[m, "eb_finable"].sum() / max(total, 1)))
    pd.DataFrame([
        {"scenario": "staffed_posts", "coverage": cov, "detections": det,
         "note": f"margin {M:.0f}kph, P={P} of K={len(posts)}"},
        {"scenario": "no_staffing", "coverage": 0.0, "detections": 0.0, "note": "counterfactual"},
        {"scenario": "random_siting_mean", "coverage": float(np.mean(rand)),
         "detections": float(np.mean(rand)) * total, "note": "200 draws"}]
    ).to_csv(os.path.join(RES, "finable_speeding_coverage.csv"), index=False)
    print(f"   coverage staffed={100 * cov:.1f}% random={100 * float(np.mean(rand)):.1f}% detections={det:,.0f}")
    audit.update({"coverage": cov, "detections": det, "coverage_random": float(np.mean(rand))})
    sc[["road_osmid", "road_name", "exposure_trips", "finable",
        "eb_finable", "assigned_post", "covered"]].to_csv(
        os.path.join(RES, "finable_speeding_segments.csv"), index=False)

    # phase split
    if phase_rows:
        ph = pd.DataFrame(phase_rows, columns=["fixture", "phase", "road_osmid", "n"])
        phg = ph.groupby(["phase", "road_osmid"])["n"].sum().unstack(fill_value=0)
        phg.to_csv(os.path.join(RES, "finable_speeding_phase.csv"))
        sh = ph.groupby("phase")["n"].sum()
        print("   finable per phase: " + ", ".join(f"{p}={int(sh[p]):,}" for p in ["pre", "during", "post"] if p in sh))
        audit["phase_totals"] = {p: int(sh[p]) for p in sh.index}

    # 6. Figures + tex
    print("\n[5/6] Figures + LaTeX...")
    t15 = corr.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(t15["road_name"], t15["eb"], color="#0B6E4F", alpha=0.9)
    ax.set_xlabel(f"EB finable speeding events (>={M:.0f} km/h over limit)")
    ax.set_title("Top finable-speeding corridors")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "finable_corridors.png"), dpi=150)
    plt.close(fig)
    cdf = pd.read_csv(os.path.join(RES, "finable_speeding_coverage.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(cdf["scenario"], cdf["coverage"], color=["#0B6E4F", "#888888", "#5b8fb0"])
    ax.set_ylabel("Share of finable speeding detected")
    ax.set_title(f"Maximising finable detections (margin {M:.0f} kph, P={P})")
    for i, v in enumerate(cdf["coverage"]):
        ax.text(i, v + 0.01, f"{100 * v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "finable_coverage.png"), dpi=150)
    plt.close(fig)
    L = []
    L.append("% finable_speeding.tex - auto-generated by 43_finable_speeding.py")
    L.append(f"% Estimand: GPS speed >= posted limit + {M:.0f} km/h (offences, not flags).")
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append("\\caption{\\textcolor{Highlight}{Top finable-speeding corridors}"
             f" (exceedance \\(\\ge{M:.0f}\\) km/h; $P={P}$ of $K={len(posts)}$ posts).}}")
    L.append("\\label{tab:finable}")
    L.append("\\begin{tabular}{lrrr}")
    L.append("\\hline")
    L.append("Corridor & Segments & EB finable & Observed \\\\")
    L.append("\\hline")
    for _, r in corr.head(15).iterrows():
        L.append("{0} & {1} & {2:,.0f} & {3:,.0f} \\\\".format(
            str(r["road_name"])[:32], int(r["n_seg"]), r["eb"], r["obs"]))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    L.append(f"At a $\\ge{M:.0f}$ km/h fining margin, staffing $P={P}$ posts detects"
             f" {100 * cov:.1f}\\% of finable speeding"
             f" (random {100 * float(np.mean(rand)):.1f}\\%, none $0\\%$;"
             f" {det:,.0f} detections from {obs:,.0f} finable events;"
             f" sensitivity: " + ", ".join(f"\\(\\ge{t}\\)kph {n:,}" for t, n in tot.items()) + ")."
             f" SPF AIC {aic:.0f}, $\\alpha={alpha:.2f}$, converged.")
    with open(os.path.join(RES, "finable_speeding.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    # POT severity table: observed vs GPD-fitted exceedance survival.
    P = []
    P.append("% finable_pot.tex - auto-generated by 43_finable_speeding.py")
    P.append("%% Peaks-over-threshold severity: P(exceed>=m | exceed>=%.0f kph)." % M)
    P.append("\\begin{table}[htbp]")
    P.append("\\centering")
    P.append("\\caption{\\textcolor{Highlight}{Exceedance severity (POT-GPD)}"
             f" above $\\ge{M:.0f}$ kph: observed vs fitted tail.}}")
    P.append("\\label{tab:pot}")
    P.append("\\begin{tabular}{rrr}")
    P.append("\\hline")
    P.append("Margin $m$ (kph) & Observed $P(\\ge m)$ & GPD fitted \\\\")
    P.append("\\hline")
    for m in [10, 15, 20, 25, 30, 40]:
        obs_s = float((xa >= m).mean()) if len(xa) else float("nan")
        if pot.get("fitted") and m >= M:
            xi, sg = pot["xi"], pot["sigma"]
            if abs(xi) < 1e-6:
                fit_s = float(np.exp(-(m - M) / sg)) if sg > 0 else float("nan")
            else:
                base = 1.0 + xi * (m - M) / sg if sg > 0 else -1.0
                fit_s = float(base ** (-1.0 / xi)) if base > 0 else 0.0
        elif m == int(M):
            fit_s = 1.0
        else:
            fit_s = float("nan")
        P.append("{0} & {1:.4f} & {2:.4f} \\\\".format(
            m, obs_s, fit_s if np.isfinite(fit_s) else -1).replace("-1.0000", "--"))
    P.append("\\hline")
    P.append("\\end{tabular}")
    P.append("\\end{table}")
    if pot.get("fitted"):
        P.append("")
        P.append(f"GPD above ${M:.0f}$ kph: $\\xi={pot['xi']:.3f}$, "
                 f"$\\sigma={pot['sigma']:.2f}$ kph on $n={pot['n_exceed']:,}$ exceedances.")
    with open(os.path.join(RES, "finable_pot.tex"), "w") as f:
        f.write("\n".join(P) + "\n")
    print("   Wrote finable_pot.tex")
    with open(os.path.join(RES, "finable_speeding_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print("DONE! Finable speeding complete.")


if __name__ == "__main__":
    main()
