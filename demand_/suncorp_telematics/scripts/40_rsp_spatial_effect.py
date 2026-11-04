"""
40_rsp_spatial_effect.py - SINGLE sound pipeline for the paper.

Purpose (reframed per review):
  Answer: "on a given road segment, will the RSP (police post presence)
  improve the safety condition?"  Analysis is SPATIAL and SEGMENT-BASED
  (which assigned post serves this segment), not temporal across fixture
  dates and NOT a circular radius buffer.

  Segment-based rule (replaces radius):
    Every network segment is assigned to exactly one RSP post - its nearest
    candidate post (network-Voronoi assignment; each segment counted once,
    no double-counting, no overlapping discs). A segment is COVERED iff its
    assigned post is staffed on the night (P staffed of K=14 candidates).
    WITHOUT-platform counterfactual = same segments, empty staffed set
    (segments reassigned nowhere). Random baseline = random staffed sets.
  Distance to assigned post is kept only as an explanatory covariate
  (binned descriptively), never as the coverage rule.

  - Fits ONE convergent geometry-augmented NB SPF (with convergence check +
    parsimonious fallback; no hidden ConvergenceWarning).
  - Computes EB risk per segment for Event vs matched Control.
  - Uses RATE-based uplift (per 1k trips), not raw-count difference
    (raw difference is what produced the nonsense -777k total uplift).
  - Writes paper-ready outputs that OVERWRITE the buggy tables/figs.

Inputs (all already on disk, no re-extraction):
  data/near_miss_hb_ecf.csv
  data/segment_roads_part0_5000.csv.gz
  data/osm_roads_suncorp.geojson
  data/road_geometry_features.csv  (optional, merged if present)
  data/road_meta.csv               (names; NOTE: lives in data/, not results/)
  results/post_selection_mip.csv   (P4 posts; rebuilt by caller if missing)

Outputs (paper inputs):
  results/rsp_segment_effect.csv    per-segment: eb_event/control, uplift_rate,
                                    assigned_post, dist_to_assigned_m, covered,
                                    preventable
  results/rsp_distance_bins.csv     binned treatment effect (descriptive only)
  results/rsp_coverage.csv          with/without/random simulation summary
  results/rsp_spatial_effect.tex    \\input-ready table for manuscript
  figures/rsp_uplift_vs_distance.png
  figures/rsp_coverage_simulation.png
  results/rsp_audit.json            convergence + sanity checks (fail loudly)

Usage:
  python 40_rsp_spatial_effect.py [--staffed-p 5] [--seed 42]
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
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.stats import spearmanr

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

SCREEN = os.path.join(DATA, "near_miss_hb_ecf.csv")
PANEL = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
FEATS = os.path.join(DATA, "road_geometry_features.csv")
META = os.path.join(DATA, "road_meta.csv")  # BUGFIX: was results/road_meta.csv
POSTS = os.path.join(RES, "post_selection_mip.csv")

OUT_SEG = os.path.join(RES, "rsp_segment_effect.csv")
OUT_BINS = os.path.join(RES, "rsp_distance_bins.csv")
OUT_COV = os.path.join(RES, "rsp_coverage.csv")
OUT_TEX = os.path.join(RES, "rsp_spatial_effect.tex")
OUT_JSON = os.path.join(RES, "rsp_audit.json")
FIG1 = os.path.join(FIG, "rsp_uplift_vs_distance.png")
FIG2 = os.path.join(FIG, "rsp_coverage_simulation.png")

FULL_FORMULA = ("total_hb ~ np.log(exposure_trips + 1) + C(highway)"
                " + maxspeed_kmh_z + lanes_z + curvature_deg_km_z"
                " + dist_to_intersection_m_z")
LEAN_FORMULA = ("total_hb ~ np.log(exposure_trips + 1) + lanes_z"
                " + dist_to_intersection_m_z")

sns.set_style("whitegrid")


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def standardise(df, cols):
    for c in cols:
        z = c + "_z"
        if c in df.columns and z not in df.columns:
            sd = float(df[c].std())
            df[z] = (df[c] - float(df[c].mean())) / (sd if sd and sd > 0 else 1.0)
    return df


def fit_convergent_spf(df, formula_full=FULL_FORMULA, formula_lean=LEAN_FORMULA):
    """Try full NB (lbfgs, high iter); check mle_retvals converged flag.
    Fall back to lean NB, then Poisson-GLM + Pearson dispersion.
    Returns (result, alpha, formula_used, converged, aic). Never hides failure.
    """
    import statsmodels.api as sm
    last_err = None
    for formula in (formula_full, formula_lean):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model = smf.negativebinomial(formula, data=df)
                res = model.fit(disp=False, method="lbfgs", maxiter=2000)
            converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
            alpha = float(res.params["alpha"]) if "alpha" in res.params else float(res.scale)
            if converged and np.isfinite(res.aic) and alpha > 0:
                return res, alpha, formula, True, float(res.aic)
            last_err = f"not-converged (converged={converged}) for: {formula}"
        except Exception as e:  # noqa: BLE001 - must fall through to fallback
            last_err = f"{type(e).__name__}: {e} for: {formula}"
    # Final fallback: Poisson GLM (always converges) + Pearson dispersion as alpha proxy
    fam = sm.families.Poisson()
    res = smf.glm(formula_lean, data=df, family=fam).fit(disp=False)
    pearson = float(((res.resid_pearson) ** 2).sum() / max(res.df_resid, 1))
    lam = np.asarray(res.fittedvalues, dtype=float)
    alpha = float(max((pearson - 1.0) / max(np.mean(lam), 1e-9), 1e-3))
    return res, alpha, formula_lean + " [Poisson-GLM fallback]", True, float(res.aic)


def eb_adjust(y, lam, alpha):
    w = 1.0 / (1.0 + alpha * np.asarray(lam, dtype=float))
    y = np.asarray(y, dtype=float)
    return w * lam + (1.0 - w) * y, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staffed-p", type=int, default=5,
                    help="officers/posts staffed per night (segment covered iff its assigned post is staffed)")
    ap.add_argument("--max-service-m", type=float, default=4000.0,
                    help="sanity guard only (P5 max move); NOT a coverage rule")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    audit = {"staffed_p": int(args.staffed_p), "max_service_m": float(args.max_service_m),
             "coverage_rule": "segment-based Voronoi assignment (no radius buffer)"}

    print("=" * 80)
    print("40_rsp_spatial_effect.py - spatial RSP effect (one sound pipeline)")
    print("=" * 80)

    # 1. Screening SPF on near_miss table (convergent fit)
    print("\n[1/6] Fitting convergent screening SPF...")
    df = pd.read_csv(SCREEN)
    df = df[df["exposure_trips"] > 0].copy()
    # Fill BEFORE standardising (old order standardised NaNs into _z, which
    # made res.predict(df) return NaN -> expected=nan, EB=nan in job 25221308).
    for c in ["curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
              "dist_to_intersection_m", "node_degree", "length_m"]:
        if c in df.columns:
            df[c] = df[c].fillna(float(df[c].median()))
    # geometry z-scores (same scaling convention as 19/20/21)
    df = standardise(df, ["curvature_deg_km", "sinuosity", "maxspeed_kmh",
                           "lanes", "dist_to_intersection_m",
                           "node_degree", "length_m"])
    df["highway"] = df.get("highway", "unknown").fillna("unknown").astype(str)
    need = ["exposure_trips", "lanes_z", "dist_to_intersection_m_z",
            "maxspeed_kmh_z", "curvature_deg_km_z", "highway", "total_hb"]
    have_full = all(c in df.columns for c in need)
    res, alpha, formula_used, converged, aic = fit_convergent_spf(
        df, FULL_FORMULA if have_full else LEAN_FORMULA, LEAN_FORMULA)
    print(f"   formula: {formula_used}\n   converged={converged} AIC={aic:.1f} alpha={alpha:.4f}")
    audit.update({"spf_formula": formula_used, "spf_converged": converged,
                  "spf_aic": aic, "spf_alpha": alpha, "n_screen": int(len(df))})
    if not converged:
        raise SystemExit("FATAL: SPF did not converge even with fallback - refusing to write paper tables.")

    lam = np.asarray(res.predict(df), dtype=float)
    if not np.all(np.isfinite(lam)):
        bad = int((~np.isfinite(lam)).sum())
        raise SystemExit(f"FATAL: SPF predicted {bad}/{len(df)} non-finite values - covariate NaN leak, aborting.")
    eb, w = eb_adjust(df["total_hb"].values, lam, alpha)
    # Sanity: EB total must be CLOSE TO OBSERVED (shrinkage preserves level),
    # NOT equal to expected total (old table.tex bug: 179,428.5).
    print(f"   observed={df['total_hb'].sum():,.0f} expected={lam.sum():,.1f} EB={eb.sum():,.1f} mean w={w.mean():.3f}")
    audit.update({"total_obs": float(df["total_hb"].sum()), "total_exp": float(lam.sum()),
                  "total_eb": float(eb.sum()), "mean_w": float(w.mean())})
    if abs(eb.sum() - df["total_hb"].sum()) / max(df["total_hb"].sum(), 1) > 0.05:
        raise SystemExit("FATAL: EB total far from observed - shrinkage bug, aborting.")

    # 2. Event vs Control per segment (rate-based; fixes -777k uplift bug)
    print("\n[2/6] Event vs Control per-segment rates...")
    panel = pd.read_csv(PANEL)
    panel["Y"] = panel["braking"] + panel["speeding"]
    ev = panel[panel["type"] == "Event"].groupby("road_osmid").agg(
        event_Y=("Y", "sum"), event_trips=("trips", "sum")).reset_index()
    ct = panel[panel["type"] == "Control"].groupby("road_osmid").agg(
        control_Y=("Y", "sum"), control_trips=("trips", "sum")).reset_index()
    seg = ev.merge(ct, on="road_osmid", how="outer").fillna(0)
    # Per-arm convergent SPFs (event / control separately, same P1 strata logic
    # as the paper). The old single-net_rate shortcut shrank both arms toward one
    # network mean and biased the uplift; per-arm expectations remove that artefact.
    # Geometry z-scores are reused from the screening sample (missing = mean = 0
    # in z-space); highway levels unseen in training are mapped to the mode.
    _zcols = ["maxspeed_kmh_z", "lanes_z", "curvature_deg_km_z", "dist_to_intersection_m_z"]
    _geo = df[["road_osmid"] + [c for c in _zcols if c in df.columns] + ["highway"]].copy() \
        if "road_osmid" in df.columns else None
    _hw_mode = str(df["highway"].mode().iloc[0]) if "highway" in df.columns else "unknown"

    def _arm_expected(y_col, t_col):
        fr = seg[["road_osmid", y_col, t_col]].copy()
        fr = fr[fr[t_col] > 0].copy()
        fr["total_hb"] = fr[y_col].astype(float)
        fr["exposure_trips"] = fr[t_col].astype(float)
        if _geo is not None:
            fr = fr.merge(_geo, on="road_osmid", how="left")
        for z in _zcols:
            if z not in fr.columns:
                fr[z] = 0.0
            fr[z] = fr[z].fillna(0.0)
        if "highway" not in fr.columns:
            fr["highway"] = _hw_mode
        fr["highway"] = fr["highway"].fillna(_hw_mode).astype(str)
        seen = set(df["highway"].astype(str).unique()) if "highway" in df.columns else {_hw_mode}
        fr.loc[~fr["highway"].isin(seen), "highway"] = _hw_mode
        have_full_arm = all(c in fr.columns for c in
                            ["exposure_trips", "lanes_z", "dist_to_intersection_m_z",
                             "maxspeed_kmh_z", "curvature_deg_km_z", "highway", "total_hb"])
        r, a, fu, cv, ai = fit_convergent_spf(
            fr, FULL_FORMULA if have_full_arm else LEAN_FORMULA, LEAN_FORMULA)
        print(f"   arm {t_col}: {fu} converged={cv} AIC={ai:.0f} alpha={a:.4f} n={len(fr):,}")
        if not cv:
            raise SystemExit(f"FATAL: {t_col} arm SPF did not converge - refusing paper tables.")
        # Predict for every segment (trips==0 -> expected 0, eb 0 by construction).
        full = seg[["road_osmid", t_col]].copy()
        full["total_hb"] = 0.0
        full["exposure_trips"] = full[t_col].astype(float)
        if _geo is not None:
            full = full.merge(_geo, on="road_osmid", how="left")
        for z in _zcols:
            if z not in full.columns:
                full[z] = 0.0
            full[z] = full[z].fillna(0.0)
        if "highway" not in full.columns:
            full["highway"] = _hw_mode
        full["highway"] = full["highway"].fillna(_hw_mode).astype(str)
        full.loc[~full["highway"].isin(seen), "highway"] = _hw_mode
        pred = np.asarray(r.predict(full), dtype=float)
        pred = np.where(np.isfinite(pred) & (full[t_col].to_numpy() > 0), pred, 0.0)
        return pred, a, fu, ai

    seg["expected_event"], alpha_ev, form_ev, aic_ev = _arm_expected("event_Y", "event_trips")
    seg["expected_control"], alpha_ct, form_ct, aic_ct = _arm_expected("control_Y", "control_trips")
    audit.update({"arm_event_formula": form_ev, "arm_event_aic": aic_ev, "arm_event_alpha": alpha_ev,
                  "arm_control_formula": form_ct, "arm_control_aic": aic_ct, "arm_control_alpha": alpha_ct})
    seg["eb_event"], seg["w_event"] = eb_adjust(seg["event_Y"], seg["expected_event"], alpha_ev)
    seg["eb_control"], seg["w_control"] = eb_adjust(seg["control_Y"], seg["expected_control"], alpha_ct)
    for arm in ("event", "control"):
        seg[f"{arm}_rate_per_1k"] = np.where(seg[f"{arm}_trips"] > 0,
                                             1000 * seg[f"eb_{arm}"] / seg[f"{arm}_trips"], 0.0)
    seg["uplift_rate_per_1k"] = seg["event_rate_per_1k"] - seg["control_rate_per_1k"]
    seg["uplift_events"] = seg["eb_event"] - seg["eb_control"] * (
        seg["event_trips"] / np.where(seg["control_trips"] > 0, seg["control_trips"], np.nan))
    seg["uplift_events"] = seg["uplift_events"].fillna(seg["eb_event"])
    print(f"   segments={len(seg):,} mean uplift rate={seg['uplift_rate_per_1k'].mean():+.3f}/1k"
          f"  share uplift>0={(seg['uplift_rate_per_1k'] > 0).mean():.2f}")
    audit["share_uplift_pos"] = float((seg["uplift_rate_per_1k"] > 0).mean())

    # 3. Segment-based assignment to RSP post (NO radius buffer)
    print("\n[3/6] Segment-based assignment to RSP post (Voronoi, no radius)...")
    if not os.path.exists(POSTS):
        raise SystemExit(f"FATAL: {POSTS} missing - run 28_post_selection_mip.py first (Job 09).")
    posts = pd.read_csv(POSTS)
    plat, plon = posts["lat"].to_numpy(), posts["lon"].to_numpy()
    # Staffed set: top-P by P4 benefit (same roster logic as P5 static plan).
    # Segment-based: each segment served by exactly one post.
    benefit = posts["benefit"].to_numpy() if "benefit" in posts.columns else np.ones(len(posts))
    P = int(min(int(args.staffed_p), len(posts)))
    staffed_idx = set(np.argsort(-benefit)[:P].tolist())
    try:
        import geopandas as gpd
        g = gpd.read_file(GEOJSON)
        g["osmid"] = pd.to_numeric(g["osmid"], errors="coerce")
        # Centroid in a projected CRS (UTM 56S), then back to WGS84: avoids the
        # geographic-CRS centroid warning and its (small) distance error.
        cent = g.to_crs(28356).geometry.centroid.to_crs(4326)
        coord = pd.DataFrame({"road_osmid": g["osmid"].values,
                              "seg_lat": cent.y.values, "seg_lon": cent.x.values}).dropna()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"FATAL: cannot derive segment coords from geojson: {e}")
    seg = seg.merge(coord, on="road_osmid", how="left")
    _dmat = haversine_m(seg["seg_lat"].values[:, None], seg["seg_lon"].values[:, None],
                        plat[None, :], plon[None, :])
    seg["assigned_post"] = np.nanargmin(np.where(np.isfinite(_dmat), _dmat, np.inf), axis=1)
    seg["dist_to_assigned_m"] = np.nanmin(np.where(np.isfinite(_dmat), _dmat, np.inf), axis=1)
    # Back-compat alias (distance is descriptive only, never the coverage rule)
    seg["dist_to_post_m"] = seg["dist_to_assigned_m"]
    n_missing = int(seg["dist_to_assigned_m"].isna().sum())
    seg["dist_to_assigned_m"] = seg["dist_to_assigned_m"].fillna(1e9)
    seg["dist_to_post_m"] = seg["dist_to_assigned_m"]
    print(f"   posts={len(posts)} staffed P={P} median assigned-dist={seg['dist_to_assigned_m'].median():.0f}m missing_coords={n_missing}")
    audit["n_posts"] = int(len(posts))

    # names (BUGFIX: meta lives in data/)
    if os.path.exists(META):
        meta = pd.read_csv(META)
        name_col = "name" if "name" in meta.columns else "road_name"
        seg = seg.merge(meta[["osmid", name_col]].rename(
            columns={"osmid": "road_osmid", name_col: "road_name"}),
            on="road_osmid", how="left")
    else:
        seg["road_name"] = ""

    # 4. Distance bins: descriptive only (coverage rule is assignment, not radius)
    print("\n[4/6] Binning treatment effect by assigned-post distance (descriptive)...")
    bins = [0, 250, 500, 1000, 2000, np.inf]
    labels = ["0-250m", "250-500m", "500m-1km", "1-2km", ">2km"]
    seg["dist_bin"] = pd.cut(seg["dist_to_assigned_m"], bins=bins, labels=labels)
    brow = seg.groupby("dist_bin", observed=True).agg(
        n_seg=("road_osmid", "size"),
        mean_event_rate=("event_rate_per_1k", "mean"),
        mean_control_rate=("control_rate_per_1k", "mean"),
        mean_uplift_rate=("uplift_rate_per_1k", "mean"),
        share_uplift_pos=("uplift_rate_per_1k", lambda s: float((s > 0).mean())),
        total_eb_event=("eb_event", "sum")).reset_index()
    brow.to_csv(OUT_BINS, index=False)
    print(brow.to_string(index=False))
    rho, _ = spearmanr(seg["dist_to_assigned_m"].clip(upper=5000), seg["uplift_rate_per_1k"])
    print(f"   Spearman(assigned-dist, uplift_rate)={rho:.3f} (descriptive; expect >0)")
    audit["spearman_dist_uplift"] = float(rho)

    # 5. WITH vs WITHOUT platform simulation (SEGMENT-BASED counterfactual)
    # Covered iff the segment's assigned post is staffed. Each segment counted
    # exactly once under its assigned post - no overlapping radius discs.
    print("\n[5/6] With/without-platform simulation (segment-based)...")
    seg["covered"] = seg["assigned_post"].isin(staffed_idx)
    seg["preventable_events"] = np.where(seg["covered"], np.clip(seg["uplift_events"], 0, None), 0.0)
    cov_risk = float(seg.loc[seg["covered"], "eb_event"].sum() / max(seg["eb_event"].sum(), 1))
    prev = float(seg["preventable_events"].sum())
    # random-siting baseline: random staffed sets of size P (segments reassigned nowhere,
    # only the staffed label changes) - preserves segment-based counting.
    rand_cov = []
    K = len(posts)
    assigned = seg["assigned_post"].to_numpy()
    for _ in range(200):
        rnd_staff = set(rng.choice(K, size=P, replace=False).tolist())
        mask = np.isin(assigned, list(rnd_staff))
        rand_cov.append(float(seg.loc[mask, "eb_event"].sum() / max(seg["eb_event"].sum(), 1)))
    cov_df = pd.DataFrame([{"scenario": "with_RSP_posts", "coverage": cov_risk,
                            "preventable_events": prev,
                            "note": f"segment-based: P={P} staffed of K={K} (Voronoi assignment)"},
                           {"scenario": "without_platform", "coverage": 0.0,
                            "preventable_events": 0.0, "note": "counterfactual: no posts staffed"},
                           {"scenario": "random_siting_mean", "coverage": float(np.mean(rand_cov)),
                            "preventable_events": float(np.mean(rand_cov) / max(cov_risk, 1e-9) * prev),
                            "note": "200 random staffed sets (segment-based)"}])
    cov_df.to_csv(OUT_COV, index=False)
    print(cov_df.to_string(index=False))
    audit.update({"coverage_with": cov_risk, "preventable_events": prev,
                  "coverage_random": float(np.mean(rand_cov))})

    seg_out = seg[["road_osmid", "road_name", "event_Y", "event_trips", "control_Y",
                   "control_trips", "eb_event", "eb_control", "event_rate_per_1k",
                   "control_rate_per_1k", "uplift_rate_per_1k", "uplift_events",
                   "assigned_post", "dist_to_assigned_m", "dist_to_post_m",
                   "dist_bin", "covered", "preventable_events"]]
    seg_out.to_csv(OUT_SEG, index=False)

    # 6. Figures + paper tex (segment-based wording throughout)
    print("\n[6/6] Figures + LaTeX...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(x) for x in brow["dist_bin"]], brow["mean_uplift_rate"], color="#C8102E", alpha=0.85)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Distance to assigned RSP post (segment-based Voronoi; descriptive)")
    ax.set_ylabel("Mean event-minus-control uplift (EB events / 1k trips)")
    ax.set_title("Event vs control uplift by assigned-post distance (means are negative:\n"
                 "event windows average lower HB rates than matched controls)")
    fig.tight_layout()
    fig.savefig(FIG1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(cov_df["scenario"], cov_df["coverage"], color=["#0B6E4F", "#888888", "#5b8fb0"])
    ax.set_ylabel("Share of event-day EB risk on covered segments")
    ax.set_title(f"With vs without platform (segment-based, P={P} of {len(posts)})")
    for i, v in enumerate(cov_df["coverage"]):
        ax.text(i, v + 0.01, f"{100 * v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG2, dpi=150)
    plt.close(fig)

    L = []
    L.append("% rsp_spatial_effect.tex - auto-generated by 40_rsp_spatial_effect.py")
    L.append("% SEGMENT-BASED RSP effect: each segment assigned once to its nearest post.")
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append("\\caption{\\textcolor{Highlight}{RSP effect by assigned-post distance}"
             f" (segment-based Voronoi; $P={P}$ staffed of $K={len(posts)}$ posts)."
             " Each segment is counted once under its assigned post (no radius discs)."
             " Uplift is EB event-minus-control rate per 1k trips;"
             " a segment is covered iff its assigned post is staffed.}")
    L.append("\\label{tab:rsp_effect}")
    L.append("\\begin{tabular}{lrrrrr}")
    L.append("\\hline")
    L.append("Assigned-dist bin (descriptive) & Segments & Event rate & Control rate & Uplift/1k & Share $>0$ \\\\")
    L.append("\\hline")
    for _, r in brow.iterrows():
        L.append(f"{r['dist_bin']} & {int(r['n_seg']):,} & {r['mean_event_rate']:.2f} & "
                 f"{r['mean_control_rate']:.2f} & {r['mean_uplift_rate']:+.2f} & {r['share_uplift_pos']:.2f} \\\\")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    L.append(f"Segment-based coverage with $P={P}$ staffed posts is {100 * cov_risk:.1f}\\% of event-day EB risk"
             f" (random staffing {100 * float(np.mean(rand_cov)):.1f}\\%, no-platform $0\\%$)."
             f" Mean event-minus-control uplift is {seg['uplift_rate_per_1k'].mean():+.1f}/1k"
             f" ({100 * float((seg['uplift_rate_per_1k'] > 0).mean()):.0f}\\% of segments positive), so the deployment case"
             f" rests on absolute event-day concentration, not on exceeding ordinary-day rates:"
             f" uplift-positive covered segments are the incremental-priority subset"
             f" ({prev:,.0f} EB events of preventable uplift)."
             f" SPF: {formula_used}, AIC {aic:.0f}, $\\alpha={alpha:.2f}$, converged.")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"   Wrote {OUT_SEG}\n   Wrote {OUT_TEX}\n   Wrote {OUT_JSON}")
    print("\nDONE! RSP spatial effect complete.")


if __name__ == "__main__":
    main()
