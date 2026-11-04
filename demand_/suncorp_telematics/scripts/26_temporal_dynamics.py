"""
26_temporal_dynamics.py
Proper time-series / random-walk modelling of how risky-driving risk evolves
over the study year and around Suncorp events.

What was missing
----------------
Jobs 19-24 treat every window as independent: SPFs are static strata
(pre/during/post), the hourly profile is averaged across fixtures, and the
conceptual model is a cross-sectional OLS. Nothing models *temporal
dependence* -- whether risk drifts over the season, carries momentum between
fixtures, or decays after an event.

What this adds
--------------
A. Network-level local-level state-space model (Kalman / Bayesian structural
   time series). Latent risk follows a random walk theta_t = theta_{t-1}+eta_t;
   observations are noisy log-rates. Fitted with statsmodels
   UnobservedComponents MLE per type (Event / Control). Reports the RW
   innovation variance (is there real drift?), smoothed level trajectories
   with credible bands, next-window forecasts, and an AIC comparison against
   a deterministic-constant (no-dynamics) null.

B. Per-segment dynamic Empirical Bayes with a random-walk prior. Sequential
   filtering across fixtures in chronological order:
       prior_{d}   = g*m_{d-1} + (1-g)*lambda_d      (RW carry-over)
       w           = 1 / (1 + alpha*prior_d)         (NB EB weight)
       m_d         = w*y_d + (1-w)*prior_d
   where lambda_d is that window's SPF expectation (that window's own trips),
   alpha the NB dispersion, and g a momentum coefficient tuned by
   one-step-ahead validation on training fixtures (g=0 recovers static EB).
   Final filtered state = current risk estimate per segment.

C. Momentum test: lag-1 autocorrelation of demeaned segment log-rates across
   consecutive fixtures (Event vs Control). Positive AC validates the
   random-walk premise; control windows share ref_dt with their matched
   fixture so cross-day decay is not identifiable in this design.

D. Out-of-sample holdout (2026 fixtures 14-16): static EB vs dynamic-RW EB,
   same rank metrics as 21 (row RMSE, segment Spearman, Top-N overlap).

Outputs
    results/temporal_level.csv          smoothed latent levels per type/day
    results/temporal_summary.csv        model params, AICs, tuned g
    results/temporal_autocorr.csv       lag-1 autocorrelation (momentum)
    results/temporal_holdout.csv        static vs dynamic EB holdout metrics
    results/temporal_dynamic_event.csv  per-segment current dynamic EB state
    results/temporal_dynamics.tex       LaTeX fragments
    figures/temporal_level_curve.png    Kalman-smoothed level +-95% band
    figures/temporal_momentum.png       lag-plot of consecutive log-rates

Usage
    python 26_temporal_dynamics.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJ = os.environ.get(
    "PROJ",
    "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics",
)
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
os.makedirs(FIG, exist_ok=True)

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
FEATURES = os.path.join(DATA, "road_geometry_features.csv")

OUT_LEVEL = os.path.join(RES, "temporal_level.csv")
OUT_SUMMARY = os.path.join(RES, "temporal_summary.csv")
OUT_AUTOCORR = os.path.join(RES, "temporal_autocorr.csv")
OUT_HOLDOUT = os.path.join(RES, "temporal_holdout.csv")
OUT_DYNSEG = os.path.join(RES, "temporal_dynamic_event.csv")
OUT_TEX = os.path.join(RES, "temporal_dynamics.tex")
FIG_LEVEL = os.path.join(FIG, "temporal_level_curve.png")
FIG_AUTOCORR = os.path.join(FIG, "temporal_momentum.png")

HOLD_OUT_EVENTS = [14, 15, 16]
FORMULA_AUG = ("Y ~ np.log(trips + 1) + C(highway) + maxspeed_kmh_z "
               "+ lanes_z + curvature_deg_km_z + dist_to_intersection_m_z")
FORMULA_BASE = "Y ~ np.log(trips + 1)"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_panel():
    """Load segment-window table (+geometry), return df and chronological
    day-level aggregates."""
    df = pd.read_csv(SEGMENT_ROADS)
    df["Y"] = df["braking"] + df["speeding"]
    df["ref_dt"] = pd.to_datetime(df["ref_dt"])

    if "curvature_deg_km" not in df.columns and os.path.exists(FEATURES):
        feat = pd.read_csv(FEATURES).drop(columns=["highway"], errors="ignore")
        df = df.merge(feat, on="road_osmid", how="left")
    for c in ["curvature_deg_km", "sinuosity", "is_link", "oneway",
              "road_class_rank", "node_degree", "length_m"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in ["maxspeed_kmh", "lanes", "dist_to_intersection_m"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in ["curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
              "dist_to_intersection_m", "node_degree", "length_m"]:
        if c in df.columns:
            sd = df[c].std()
            df[c + "_z"] = (df[c] - df[c].mean()) / (sd if sd and sd > 0 else 1.0)
    if "highway" in df.columns:
        df["highway"] = df["highway"].fillna("unknown").astype(str)

    # day-level aggregation: one row per (type, ref_id)
    day = (df.groupby(["type", "ref_id"], as_index=False)
             .agg(Y=("Y", "sum"), trips=("trips", "sum"),
                  ref_dt=("ref_dt", "first")))
    day = day[day["trips"] > 0].sort_values(["type", "ref_dt"]).reset_index(drop=True)
    day["rate_1k"] = 1000.0 * day["Y"] / day["trips"]
    return df, day


def fit_local_level(day, typ):
    """Fit local-level (random walk) state-space model on log daily rate."""
    s = day[day["type"] == typ]
    y = np.log(s["rate_1k"].values)

    out = {"type": typ, "n_days": len(y)}
    # null: deterministic constant (no dynamics)
    try:
        null = UnobservedComponents(y, level="deterministic constant").fit(disp=False)
        out["aic_null"] = float(null.aic)
    except Exception:
        out["aic_null"] = np.nan

    rw = UnobservedComponents(y, level="local level").fit(disp=False)
    out["aic_rw"] = float(rw.aic)
    p = dict(zip(rw.model.param_names, np.asarray(rw.params)))
    out["sigma2_irregular"] = float(p.get("sigma2.irregular", np.nan))
    out["sigma2_rw"] = float(p.get("sigma2.level", np.nan))
    out["rw_share"] = out["sigma2_rw"] / (out["sigma2_rw"] + out["sigma2_irregular"])

    lvl = rw.smoothed_state[0]
    se = np.sqrt(np.maximum(rw.smoothed_state_cov[0, 0], 0))
    lev = pd.DataFrame({
        "type": typ,
        "date": s["ref_dt"].dt.date.values,
        "log_rate": y,
        "level": lvl,
        "lo": lvl - 1.96 * se,
        "hi": lvl + 1.96 * se,
    })
    fc_log = float(np.asarray(rw.forecast(1)[0]))
    out["forecast_next_rate_1k"] = float(np.exp(fc_log) )
    return out, lev, rw


def fit_spf_alpha(train_df):
    """NB SPF on training rows; returns (result, alpha, formula_used)."""
    has_geo = "curvature_deg_km_z" in train_df.columns
    formula = FORMULA_AUG if has_geo else FORMULA_BASE
    res = smf.negativebinomial(formula, data=train_df).fit(disp=False)
    alpha = float(res.params["alpha"]) if "alpha" in res.params else 1.0
    return res, alpha, formula


def tune_gamma(Ymat, LAMmat, alpha, train_idx):
    """Grid-search momentum g by one-step-ahead SSE on training fixtures."""
    best_g, best_sse = 0.0, np.inf
    T = Ymat.shape[1]
    tr = [t for t in range(T) if t in set(train_idx)]
    for g in np.arange(0.0, 1.0, 0.05):
        m = LAMmat[:, tr[0]].copy()
        sse = 0.0
        for t in tr[1:]:
            prior = g * m + (1 - g) * LAMmat[:, t]
            sse += np.sum((Ymat[:, t] - prior) ** 2)
            w = 1.0 / (1.0 + alpha * prior)
            m = w * Ymat[:, t] + (1 - w) * prior
        if sse < best_sse:
            best_sse, best_g = sse, float(g)
    return best_g


def dynamic_filter(Ymat, LAMmat, alpha, g):
    """Run sequential RW-prior EB filter over all columns; return matrix m."""
    m = np.empty_like(LAMmat)
    m[:, 0] = LAMmat[:, 0]
    priors = np.empty_like(LAMmat)
    priors[:, 0] = np.nan
    for t in range(1, LAMmat.shape[1]):
        prior = g * m[:, t - 1] + (1 - g) * LAMmat[:, t]
        priors[:, t] = prior
        w = 1.0 / (1.0 + alpha * prior)
        m[:, t] = w * Ymat[:, t] + (1 - w) * prior
    return m, priors


def build_matrices(df, typ, res_spf):
    """Pivot per-segment x fixture matrices of Y and SPF expectation."""
    sub = df[df["type"] == typ].copy()
    sub = sub[sub["trips"] > 0]
    sub["expected"] = res_spf.predict(sub)
    ids = sorted(sub["road_osmid"].unique())
    ref_order = (sub.groupby("ref_id")["ref_dt"].first()
                   .sort_values().index.tolist())
    Yp = (sub.pivot_table(index="road_osmid", columns="ref_id",
                          values="Y", aggfunc="sum")
             .reindex(index=ids, columns=ref_order).fillna(0.0))
    Lp = (sub.pivot_table(index="road_osmid", columns="ref_id",
                          values="expected", aggfunc="sum")
             .reindex(index=ids, columns=ref_order).fillna(0.0))
    return ids, ref_order, Yp.values, Lp.values


def rank_metrics(score, obs, ids, ns=(20, 50)):
    """Spearman + top-N overlap of segment scores vs realised counts."""
    seg = pd.DataFrame({"road_osmid": ids, "score": score, "obs": obs})
    rho = spearmanr(seg["score"], seg["obs"]).statistic
    out = {"spearman": rho}
    for n in ns:
        top_obs = set(seg.nlargest(n, "obs")["road_osmid"])
        top_sc = set(seg.nlargest(n, "score")["road_osmid"])
        out[f"top{n}_overlap"] = len(top_obs & top_sc) / float(n)
    return out


# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("26_temporal_dynamics.py - random-walk / time-series risk modelling")
    print("=" * 80)

    print("\n1. Loading panel data...")
    df, day = load_panel()
    print(f"   {len(df):,} segment-windows | {df['ref_id'].nunique()} fixtures | "
          f"{df['road_osmid'].nunique():,} segments")

    # ---- A. network-level local-level (random walk) models -----------------
    print("\n2. Local-level (random walk) state-space models...")
    levels, summary_rows = [], []
    for typ in ["Event", "Control"]:
        summ, lev, _ = fit_local_level(day, typ)
        summary_rows.append(summ)
        levels.append(lev)
        print(f"   {typ:8s}: AIC null={summ['aic_null']:.1f} "
              f"AIC RW={summ['aic_rw']:.1f}  sigma2_rw={summ['sigma2_rw']:.5f} "
              f"(share {summ['rw_share']:.2f})  next={summ['forecast_next_rate_1k']:.2f}/1k")
    lev_df = pd.concat(levels, ignore_index=True)
    lev_df.to_csv(OUT_LEVEL, index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, typ in zip(axes, ["Event", "Control"]):
        d = lev_df[lev_df["type"] == typ]
        ax.plot(d["date"], d["level"], "-o", ms=3, lw=1.5, color="firebrick",
                label="latent level (RW)")
        ax.fill_between(d["date"], d["lo"], d["hi"], alpha=0.2,
                        color="firebrick", label="95% band")
        ax.plot(d["date"], d["log_rate"], "x", ms=4, color="0.4",
                label="observed log-rate")
        ax.set_title(f"{typ} windows"); ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Local-level (random walk) filtering of network risk")
    fig.tight_layout()
    fig.savefig(FIG_LEVEL, dpi=150)
    plt.close(fig)
    print(f"   Saved {FIG_LEVEL}")

    # ---- B/C/D. per-segment dynamic EB -------------------------------------
    print("\n3. Fitting SPF (training fixtures only for honest holdout)...")
    train_df = df[~df["ref_id"].isin(HOLD_OUT_EVENTS) & (df["trips"] > 0)].copy()
    spf_res, alpha, formula = fit_spf_alpha(train_df)
    print(f"   formula: {formula}")
    print(f"   AIC={spf_res.aic:.0f}  alpha={alpha:.4f}")

    print("\n4. Building per-segment matrices and tuning momentum g...")
    results_rows = []
    dyn_store = {}
    for typ in ["Event", "Control"]:
        ids, ref_order, Ym, Lm = build_matrices(df, typ, spf_res)
        tr_idx = [ref_order.index(r) for r in ref_order
                  if int(r) not in HOLD_OUT_EVENTS]
        te_idx = [ref_order.index(r) for r in ref_order
                  if int(r) in HOLD_OUT_EVENTS]

        g_star = tune_gamma(Ym, Lm, alpha, tr_idx)
        for r in summary_rows:
            if r["type"] == typ:
                r["gamma_star"] = g_star
        print(f"   {typ}: tuned momentum g* = {g_star:.2f} "
              f"(one-step-ahead SSE minimisation)")

        # full-history filter (all fixtures incl. holdout, for current state)
        Mall, Pall = dynamic_filter(Ym, Lm, alpha, g_star)

        # holdout: filter on training cols only, then step through holdouts
        Mtr, _ = dynamic_filter(Ym[:, tr_idx], Lm[:, tr_idx], alpha, g_star)
        m_carry = Mtr[:, -1].copy()
        eb_dyn_te = np.zeros_like(Ym[:, te_idx], dtype=float)
        eb_sta_te = np.zeros_like(Ym[:, te_idx], dtype=float)
        for k, t in enumerate(te_idx):
            prior_dyn = g_star * m_carry + (1 - g_star) * Lm[:, t]
            w_dyn = 1.0 / (1.0 + alpha * prior_dyn)
            eb_dyn_te[:, k] = w_dyn * Ym[:, t] + (1 - w_dyn) * prior_dyn
            m_carry = eb_dyn_te[:, k]

            w_sta = 1.0 / (1.0 + alpha * Lm[:, t])
            eb_sta_te[:, k] = w_sta * Ym[:, t] + (1 - w_sta) * Lm[:, t]

        y_te = Ym[:, te_idx]
        lam_te = Lm[:, te_idx]
        rmse = lambda a: float(np.sqrt(((a - y_te) ** 2).mean()))
        seg_dyn = eb_dyn_te.sum(axis=1)
        seg_sta = eb_sta_te.sum(axis=1)
        seg_obs = y_te.sum(axis=1)
        seg_lam = lam_te.sum(axis=1)

        mdyn = rank_metrics(seg_dyn, seg_obs, ids)
        msta = rank_metrics(seg_sta, seg_obs, ids)
        mlam = rank_metrics(seg_lam, seg_obs, ids)

        results_rows.append({
            "type": typ,
            "rmse_static_eb": rmse(eb_sta_te),
            "rmse_dynamic_eb": rmse(eb_dyn_te),
            "rmse_expected_only": rmse(lam_te),
            "spearman_static_eb": msta["spearman"],
            "spearman_dynamic_eb": mdyn["spearman"],
            "spearman_expected_only": mlam["spearman"],
            "top20_overlap_static": msta["top20_overlap"],
            "top20_overlap_dynamic": mdyn["top20_overlap"],
            "top50_overlap_static": msta["top50_overlap"],
            "top50_overlap_dynamic": mdyn["top50_overlap"],
        })
        dyn_store[typ] = pd.DataFrame({
            "road_osmid": ids,
            f"current_eb_{typ.lower()}": Mall[:, -1],
            f"weight_last_{typ.lower()}": np.where(
                Pall[:, -1] > 0, 1.0 / (1.0 + alpha * Pall[:, -1]), np.nan),
        })

    hold = pd.DataFrame(results_rows)
    hold.to_csv(OUT_HOLDOUT, index=False)
    print(hold.round(3).to_string(index=False))

    # merge per-segment dynamic states (Event + Control side by side)
    dynseg = dyn_store["Event"].merge(
        dyn_store["Control"], on="road_osmid", how="outer")
    dynseg.to_csv(OUT_DYNSEG, index=False)

    # ---- C. momentum: lag-1 autocorrelation of segment log-rates ----------
    # Control windows share ref_dt with their matched fixture, so cross-day
    # decay is not identifiable; instead test the RW premise directly: do
    # segment residuals carry over between consecutive fixtures? Positive
    # lag-1 autocorrelation of demeaned log-rates => genuine temporal
    # dependence that the dynamic filter exploits (static EB assumes AC=0).
    print("\n5. Momentum: lag-1 autocorrelation of segment log-rates...")
    K = 0.5  # regularisation for log-rate

    def ac1_for(typ):
        sub = df[(df["type"] == typ) & (df["trips"] > 0)].copy()
        sub["lrt"] = np.log((sub["Y"] + K) / (sub["trips"] + K))
        piv = sub.pivot_table(index="road_osmid", columns="ref_id",
                              values="lrt", aggfunc="mean")
        piv = piv[piv.notna().sum(axis=1) >= 4]
        dm = piv.sub(piv.mean(axis=1), axis=0)
        cols = dm.columns.tolist()
        num, den, npair = [], [], 0
        for a, b in zip(cols[:-1], cols[1:]):
            m = dm[[a, b]].dropna()
            num.append(float((m[a] * m[b]).sum()))
            den.append(float(((m[b] - m[a]) ** 2).sum() / 2.0))
            npair += len(m)
        ac1 = float(np.sum(num) / max(np.sum(den), 1e-12))
        return ac1, npair

    ac_rows = []
    for typ in ["Event", "Control"]:
        ac1, npair = ac1_for(typ)
        ac_rows.append({"type": typ, "ac1_log_rate": ac1,
                        "segment_pairs": npair})
        print(f"   {typ}: lag-1 autocorr = {ac1:.3f} ({npair:,} pairs)")
    pd.DataFrame(ac_rows).to_csv(OUT_AUTOCORR, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, typ in zip(axes, ["Event", "Control"]):
        sub = df[(df["type"] == typ) & (df["trips"] > 0)].copy()
        sub["lrt"] = np.log((sub["Y"] + K) / (sub["trips"] + K))
        piv = sub.pivot_table(index="road_osmid", columns="ref_id",
                              values="lrt", aggfunc="mean")
        piv = piv[piv.notna().sum(axis=1) >= 4]
        dm = piv.sub(piv.mean(axis=1), axis=0)
        xs, ys = [], []
        for a, b in zip(dm.columns[:-1], dm.columns[1:]):
            m2 = dm[[a, b]].dropna()
            xs.append(m2[a].values); ys.append(m2[b].values)
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        ax.plot(xs, ys, ".", ms=1.5, alpha=0.15, color="steelblue")
        b1 = float(np.polyfit(xs, ys, 1)[0])
        lim = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
        ax.plot(lim, lim, "--", color="0.6", lw=1)
        sl = float(np.polyfit(xs, ys, 1)[0])
        ax.plot(lim, [sl * l for l in lim], "-", color="firebrick", lw=1.5,
                label=f"slope={sl:.2f}")
        ax.set_title(f"{typ}"); ax.set_xlabel("log-rate at fixture d")
        ax.set_ylabel("log-rate at fixture d+1"); ax.legend(fontsize=8)
    fig.suptitle("Momentum: demeaned log-rates, consecutive fixtures")
    fig.tight_layout(); fig.savefig(FIG_AUTOCORR, dpi=150); plt.close(fig)
    print(f"   Saved {FIG_AUTOCORR}")

    # ---- LaTeX ---------------------------------------------------------------
    L = []
    L.append("% temporal_dynamics.tex - auto-generated by 26_temporal_dynamics.py")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{Random-walk state-space fit of network risk "
             "(local level vs no-dynamics null) and tuned momentum $g^*$.}")
    L.append("\\label{tab:temporal_rw}")
    L.append("\\begin{tabular}{lrrrrr}")
    L.append("\\hline")
    L.append("Type & AIC null & AIC RW & $\\sigma^2_{\\text{rw}}$ & RW share & "
             "$g^*$ \\\\")
    L.append("\\hline")
    for r in summary_rows:
        L.append("{0} & {1:.1f} & {2:.1f} & {3:.5f} & {4:.2f} & {5:.2f} \\\\".format(
            r["type"], r["aic_null"], r["aic_rw"], r["sigma2_rw"],
            r["rw_share"], r.get("gamma_star", np.nan)))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")

    L.append("")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{Static vs random-walk dynamic EB on the 2026 "
             "hold-out fixtures (segments ranked by EB score vs realised).}")
    L.append("\\label{tab:temporal_holdout}")
    L.append("\\begin{tabular}{lrrrrr}")
    L.append("\\hline")
    L.append("Type & Metric & RMSE & Spearman & Top-20 & Top-50 \\\\")
    L.append("\\hline")
    for _, r in hold.iterrows():
        L.append("{0} & static EB & {1:.2f} & {2:.3f} & {3:.2f} & {4:.2f} \\\\".format(
            r["type"], r["rmse_static_eb"], r["spearman_static_eb"],
            r["top20_overlap_static"], r["top50_overlap_static"]))
        L.append("{0} & dynamic RW EB & {1:.2f} & {2:.3f} & {3:.2f} & {4:.2f} \\\\".format(
            "", r["rmse_dynamic_eb"], r["spearman_dynamic_eb"],
            r["top20_overlap_dynamic"], r["top50_overlap_dynamic"]))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")

    if ac_rows:
        ev = next(r for r in ac_rows if r["type"] == "Event")
        ct2 = next(r for r in ac_rows if r["type"] == "Control")
        L.append("")
        L.append("Segment log-rates carry momentum between consecutive "
                 "fixtures: lag-1 autocorrelation {0:.3f} on event windows "
                 "({1:.3f} control), supporting the random-walk evolution "
                 "assumed by the dynamic filter.".format(
                     ev["ac1_log_rate"], ct2["ac1_log_rate"]))
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nWrote {OUT_TEX}")

    print("\n" + "=" * 80)
    print("DONE! Temporal dynamics complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
