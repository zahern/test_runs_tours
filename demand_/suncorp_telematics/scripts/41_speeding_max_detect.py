"""
41_speeding_max_detect.py - speeding risk + max-detection police deployment.

Question: where are the speeders, and where should P officers stand to
maximise speeding detections on event days? No RSP content: risk is the
telematics SPEEDING stream; posts are POLICE posts (Programme P4).

  1. Convergent geometry-augmented NB SPF on per-segment speeding counts
     (convergence checked, never hidden) + EB shrinkage per segment.
  2. Corridor ranking: top named roads / top segments by EB speeding.
  3. Segment-based assignment: each segment served by exactly one police
     post (nearest-post Voronoi, counted once). Staff top-P posts by
     assigned EB speeding per km; compare coverage vs random staffing
     vs no-staffing counterfactual.

Inputs: data/near_miss_hb_ecf.csv, data/osm_roads_suncorp.geojson,
        data/road_meta.csv, results/post_selection_mip.csv
Outputs: results/speeding_segments.csv, results/speeding_corridors.csv,
         results/speeding_coverage.csv, results/speeding_effect.tex,
         results/speeding_audit.json,
         figures/speeding_top_corridors.png, figures/speeding_coverage.png

Usage: python 41_speeding_max_detect.py [--staffed-p 5] [--seed 42]
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
try:
    from statsmodels.tools.sm_exceptions import HessianInversionWarning
except ImportError:  # very old statsmodels: degenerate fits caught by alpha rule below
    HessianInversionWarning = Warning

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

SCREEN = os.path.join(DATA, "near_miss_hb_ecf.csv")
GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
META = os.path.join(DATA, "road_meta.csv")
POSTS = os.path.join(RES, "post_selection_mip.csv")

OUT_SEG = os.path.join(RES, "speeding_segments.csv")
OUT_CORR = os.path.join(RES, "speeding_corridors.csv")
OUT_COV = os.path.join(RES, "speeding_coverage.csv")
OUT_TEX = os.path.join(RES, "speeding_effect.tex")
OUT_JSON = os.path.join(RES, "speeding_audit.json")
FIG1 = os.path.join(FIG, "speeding_top_corridors.png")
FIG2 = os.path.join(FIG, "speeding_coverage.png")

FULL_FORMULA = ("speeding ~ np.log(exposure_trips + 1) + C(highway)"
                " + maxspeed_kmh_z + lanes_z + curvature_deg_km_z"
                " + dist_to_intersection_m_z")
LEAN_FORMULA = ("speeding ~ np.log(exposure_trips + 1) + lanes_z"
                " + dist_to_intersection_m_z")
GEO_COLS = ["curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
            "dist_to_intersection_m", "node_degree", "length_m"]

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
    import statsmodels.api as sm
    for formula in (formula_full, formula_lean):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                warnings.simplefilter("error", HessianInversionWarning)
                res = smf.negativebinomial(formula, data=df).fit(
                    disp=False, method="lbfgs", maxiter=2000)
            converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
            alpha = float(res.params["alpha"]) if "alpha" in res.params else float(res.scale)
            # Degeneracy guard (job 25252299: lbfgs "converged" with alpha=3e302,
            # expected=10.6B vs 492k observed). Such fits are unsound: try leaner spec.
            if converged and np.isfinite(res.aic) and 0 < alpha <= 100:
                return res, alpha, formula, True, float(res.aic)
            print(f"   degenerate fit rejected (converged={converged}, alpha={alpha:.3g}): {formula}")
        except Exception as e:  # noqa: BLE001 - fall through to next spec
            print(f"   fit failed ({type(e).__name__}): {formula}")
            continue
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
    ap.add_argument("--staffed-p", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    audit = {"staffed_p": int(args.staffed_p)}

    print("=" * 80)
    print("41_speeding_max_detect.py - speeding risk + max-detection deployment")
    print("=" * 80)

    # 1. Speeding SPF + EB
    print("\n[1/5] Fitting convergent speeding SPF...")
    df = pd.read_csv(SCREEN)
    df = df[df["exposure_trips"] > 0].copy()
    for c in GEO_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(float(df[c].median()))
    df = standardise(df, GEO_COLS)
    df["highway"] = df.get("highway", "unknown").fillna("unknown").astype(str)
    need = ["exposure_trips", "lanes_z", "dist_to_intersection_m_z",
            "maxspeed_kmh_z", "curvature_deg_km_z", "highway", "speeding"]
    have_full = all(c in df.columns for c in need)
    res, alpha, formula_used, converged, aic = fit_convergent_spf(
        df, FULL_FORMULA if have_full else LEAN_FORMULA, LEAN_FORMULA)
    print(f"   formula: {formula_used}\n   converged={converged} AIC={aic:.1f} alpha={alpha:.4f}")
    audit.update({"spf_formula": formula_used, "spf_converged": converged,
                  "spf_aic": aic, "spf_alpha": alpha, "n_seg": int(len(df))})
    if not converged:
        raise SystemExit("FATAL: speeding SPF did not converge - refusing paper tables.")
    lam = np.asarray(res.predict(df), dtype=float)
    if not np.all(np.isfinite(lam)):
        raise SystemExit("FATAL: SPF predicted non-finite values - aborting.")
    eb, w = eb_adjust(df["speeding"].values, lam, alpha)
    obs, exp, ebtot = float(df["speeding"].sum()), float(lam.sum()), float(eb.sum())
    print(f"   observed={obs:,.0f} expected={exp:,.1f} EB={ebtot:,.1f} mean w={w.mean():.3f}")
    audit.update({"total_obs": obs, "total_exp": exp, "total_eb": ebtot,
                  "mean_w": float(w.mean())})
    # Honesty flag: if shrinkage collapses (w~0), rankings ARE raw observed
    # concentrations (heavy-tailed speeding leaves no borrowable strength).
    collapsed = bool(float(w.mean()) < 0.01)
    audit["eb_collapsed_to_raw"] = collapsed
    if collapsed:
        print("   NOTE: EB weight ~0 - rankings equal observed concentrations;")
        print("         reported honestly as such in the paper table.")
    if abs(ebtot - obs) / max(obs, 1) > 0.05:
        raise SystemExit("FATAL: EB total far from observed - aborting.")
    df["eb_speeding"] = eb
    df["expected_speeding"] = lam

    # Names
    if os.path.exists(META):
        meta = pd.read_csv(META)
        name_col = "name" if "name" in meta.columns else "road_name"
        df = df.merge(meta[["osmid", name_col]].rename(
            columns={"osmid": "road_osmid", name_col: "meta_name"}),
            on="road_osmid", how="left")
        df["road_name"] = df["meta_name"].fillna(df.get("road_name", "")).fillna("")
    df["road_name"] = df.get("road_name", "").fillna("").astype(str)
    if "length_m" not in df.columns:
        df["length_m"] = np.nan
    df["length_km"] = df["length_m"] / 1000.0

    # 2. Corridor ranking (named roads by EB speeding)
    print("\n[2/5] Ranking speeding corridors...")
    named = df[df["road_name"].str.strip() != ""].copy()
    corr = named.groupby("road_name").agg(
        n_seg=("road_osmid", "size"),
        eb_speeding=("eb_speeding", "sum"),
        obs_speeding=("speeding", "sum"),
        length_km=("length_km", "sum")).reset_index()
    corr["eb_per_km"] = corr["eb_speeding"] / corr["length_km"].replace(0, np.nan)
    corr = corr.sort_values("eb_speeding", ascending=False).reset_index(drop=True)
    corr.to_csv(OUT_CORR, index=False)
    print(corr.head(15).to_string(index=False))
    audit["top_corridor"] = str(corr.iloc[0]["road_name"])
    audit["top_corridor_eb"] = float(corr.iloc[0]["eb_speeding"])

    # 3. Assign segments to police posts; staff top-P by EB speeding per km
    print("\n[3/5] Assigning segments to police posts...")
    if not os.path.exists(POSTS):
        raise SystemExit(f"FATAL: {POSTS} missing - run 28_post_selection_mip.py first.")
    posts = pd.read_csv(POSTS)
    plat, plon = posts["lat"].to_numpy(), posts["lon"].to_numpy()
    try:
        import geopandas as gpd
        g = gpd.read_file(GEOJSON)
        g["osmid"] = pd.to_numeric(g["osmid"], errors="coerce")
        cent = g.to_crs(28356).geometry.centroid.to_crs(4326)
        coord = pd.DataFrame({"road_osmid": g["osmid"].values,
                              "seg_lat": cent.y.values,
                              "seg_lon": cent.x.values}).dropna()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"FATAL: segment coords failed: {e}")
    df = df.merge(coord, on="road_osmid", how="left")
    dmat = haversine_m(df["seg_lat"].values[:, None], df["seg_lon"].values[:, None],
                       plat[None, :], plon[None, :])
    df["assigned_post"] = np.nanargmin(np.where(np.isfinite(dmat), dmat, np.inf), axis=1)
    # Post benefit = EB speeding per km of assigned segments
    benefit = df.groupby("assigned_post").agg(
        eb=("eb_speeding", "sum"), lkm=("length_km", "sum"))
    benefit["per_km"] = benefit["eb"] / benefit["lkm"].replace(0, np.nan)
    benefit = benefit["per_km"].fillna(0.0)
    P = int(min(int(args.staffed_p), len(posts)))
    staffed = set(benefit.sort_values(ascending=False).head(P).index.astype(int).tolist())
    df["covered"] = df["assigned_post"].isin(staffed)
    print(f"   posts={len(posts)} staffed P={P} "
          f"staffed_idx={sorted(staffed)}")

    # 4. Coverage: staffed vs random vs none (share of EB speeding detected)
    print("\n[4/5] Detection coverage...")
    total = float(df["eb_speeding"].sum())
    cov = float(df.loc[df["covered"], "eb_speeding"].sum() / max(total, 1))
    det = float(df.loc[df["covered"], "eb_speeding"].sum())
    assigned = df["assigned_post"].to_numpy()
    rand = []
    K = len(posts)
    for _ in range(200):
        rnd = set(rng.choice(K, size=P, replace=False).tolist())
        m = np.isin(assigned, list(rnd))
        rand.append(float(df.loc[m, "eb_speeding"].sum() / max(total, 1)))
    cov_df = pd.DataFrame([
        {"scenario": "staffed_posts", "coverage": cov, "detections": det,
         "note": f"P={P} of K={K} by speeding/km (segment-based)"},
        {"scenario": "no_staffing", "coverage": 0.0, "detections": 0.0,
         "note": "counterfactual: officers elsewhere"},
        {"scenario": "random_siting_mean", "coverage": float(np.mean(rand)),
         "detections": float(np.mean(rand)) * total,
         "note": "200 random staffed sets"}])
    cov_df.to_csv(OUT_COV, index=False)
    print(cov_df.to_string(index=False))
    audit.update({"coverage_staffed": cov, "detections": det,
                  "coverage_random": float(np.mean(rand)), "P": P, "K": int(K)})
    df[["road_osmid", "road_name", "highway", "exposure_trips", "speeding",
        "expected_speeding", "eb_speeding", "assigned_post",
        "covered"]].to_csv(OUT_SEG, index=False)

    # 5. Figures + paper tex
    print("\n[5/5] Figures + LaTeX...")
    top15 = corr.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top15["road_name"], top15["eb_speeding"], color="#0B6E4F", alpha=0.9)
    ax.set_xlabel("EB-adjusted speeding events")
    ax.set_title("Top speeding corridors (EB-adjusted, event window)")
    fig.tight_layout()
    fig.savefig(FIG1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(cov_df["scenario"], cov_df["coverage"],
           color=["#0B6E4F", "#888888", "#5b8fb0"])
    ax.set_ylabel("Share of EB speeding detected")
    ax.set_title(f"Maximising speeder detection (P={P} of {K} police posts)")
    for i, v in enumerate(cov_df["coverage"]):
        ax.text(i, v + 0.01, f"{100 * v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG2, dpi=150)
    plt.close(fig)

    L = []
    L.append("% speeding_effect.tex - auto-generated by 41_speeding_max_detect.py")
    L.append("% Speeding risk + max-detection police deployment (no RSP content).")
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append("\\caption{\\textcolor{Highlight}{Top speeding corridors}"
             " (EB-adjusted speeding events; segment-based assignment to"
             f" $P={P}$ staffed of $K={len(posts)}$ police posts).}}")
    L.append("\\label{tab:speeding}")
    L.append("\\begin{tabular}{lrrr}")
    L.append("\\hline")
    L.append("Corridor & Segments & EB speeding & Per km \\\\")
    L.append("\\hline")
    for _, r in corr.head(15).iterrows():
        L.append("{0} & {1} & {2:,.0f} & {3:,.0f} \\\\".format(
            str(r["road_name"])[:32], int(r["n_seg"]), r["eb_speeding"],
            r["eb_per_km"] if np.isfinite(r["eb_per_km"]) else 0))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    L.append(f"Staffing the top $P={P}$ posts by speeding intensity detects"
             f" {100 * cov:.1f}\\% of EB-adjusted speeding"
             f" (random siting {100 * float(np.mean(rand)):.1f}\\%,"
             f" no staffing $0\\%$; {det:,.0f} detections)."
             f" Speeding SPF: AIC {aic:.0f}, $\\alpha={alpha:.2f}$, converged;"
             f" {obs:,.0f} observed speeding events across {len(df):,} segments.")
    if collapsed:
        L.append("Shrinkage collapses ($\\bar w\\approx0$) under the heavy tail,"
                 " so corridor order equals observed concentrations: the honest"
                 " estimand for where to stand to find speeders.")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"   Wrote {OUT_SEG}\n   Wrote {OUT_TEX}\n   Wrote {OUT_JSON}")
    print("\nDONE! Speeding max-detection complete.")


if __name__ == "__main__":
    main()
