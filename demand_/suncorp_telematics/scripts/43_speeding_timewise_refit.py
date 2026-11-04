"""
43_speeding_timewise_refit.py - speeding-only Fig5 refit WITHOUT zero-inflation.

Context
-------
42_speeding_timewise_fix.py ran green but its NB random-parameters SPFs
collapsed (alpha 0.001-0.079, mean EB weight ~1.0) and underpredicted
observed speeding 3-5x per stratum (e.g. Event/pre expected 17.9k vs
observed 91.6k). Those EB totals must NOT enter the paper.

Modelling constraint (analyst-imposed, non-negotiable)
------------------------------------------------------
No zero-inflated or hurdle components. Every modelled row has trips > 0,
i.e. the segment was exposed in that window, so a zero is a sampling zero
(no exceedance in a short window), NOT a structural "safe state". A ZI /
hurdle mixing weight would assert segments where speeding is impossible,
which is false on an exposed network.

What this script does instead (all within NB2 / Poisson-GLM family)
-------------------------------------------------------------------
Per (type x phase) stratum, try fixed-effects NB specs via statsmodels
(the same estimator that convergently fits the screening speeding SPF in
41_speeding_max_detect.py: AIC 30,827, alpha=6.47), richest first:
  (a) full:  logexp + highway class + maxspeed + lanes + curvature + dist_int
  (b) lean:  logexp + lanes + dist_to_intersection   (script-41 spec)
  (c) base:  logexp only
A spec is accepted only if ALL hold (convergence AND fidelity):
  - optimizer reports converged, AIC finite, 0 < alpha <= 100
    (degeneracy guard from script 41: rejects lbfgs "converged" fits
    with absurd alpha / exploded expectations);
  - mean fidelity: |sum(predicted) - sum(observed)| / sum(observed) < 10%.
    A count model whose mean misses the observed total 5x is rejected no
    matter its AIC.
If no NB spec passes, fall back to Poisson-GLM (lean) + Pearson
moment alpha = max((pearson-1)/mean(lam), 1e-3) (script-41 fallback).
With an intercept, Poisson-GLM fitted totals equal observed totals by
construction, so aggregate fidelity is guaranteed without any
zero-process machinery.
Select the accepted spec with the lowest AIC per stratum.

EB + guards
-----------
w = 1/(1 + alpha*lam); EB = w*lam + (1-w)*y. ABORT (SystemExit, no
artefacts) if any stratum's EB total differs from observed by > 5%:
a paper EB surface that does not preserve aggregates is unsound.

Hourly plot fix #2
------------------
In addition to NaN (not 0) for unobserved hours (job 12 fix), the line
is BROKEN at NaN gaps (contiguous observed blocks plotted separately),
so no straight segment bridges the unobserved 01-10 range.

Outputs
-------
  results/speeding_timewise_refit_segment_phase.csv
  results/speeding_timewise_refit_hourly.csv        (NaN = unobserved hour)
  results/speeding_timewise_refit_summary.csv
  results/speeding_timewise_refit.tex
  results/speeding_timewise_refit_audit.json
  figures/speeding_timewise_refit_phase_curve.png
  figures/speeding_timewise_refit_hourly.png

Usage:
  python 43_speeding_timewise_refit.py
"""
import json
import os
import sys
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
except ImportError:
    HessianInversionWarning = Warning

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
FEATURES = os.path.join(DATA, "road_geometry_features.csv")
OUT_SEG_PHASE = os.path.join(RES, "speeding_timewise_refit_segment_phase.csv")
OUT_HOURLY = os.path.join(RES, "speeding_timewise_refit_hourly.csv")
OUT_SUMMARY = os.path.join(RES, "speeding_timewise_refit_summary.csv")
OUT_TEX = os.path.join(RES, "speeding_timewise_refit.tex")
OUT_JSON = os.path.join(RES, "speeding_timewise_refit_audit.json")
FIG_PHASE = os.path.join(FIG, "speeding_timewise_refit_phase_curve.png")
FIG_HOURLY = os.path.join(FIG, "speeding_timewise_refit_hourly.png")

PHASES = ["pre", "during", "post"]
TYPES = ["Event", "Control"]
GEO_COLS = ["curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
            "dist_to_intersection_m", "node_degree", "length_m"]
MEAN_FID_TOL = 0.10   # spec accepted only if fitted total within 10% of observed
EB_TOL = 0.05         # abort if any stratum EB total differs from observed by > 5%

FULL_FORMULA = ("Y ~ np.log(trips + 1) + C(highway)"
                " + maxspeed_kmh_z + lanes_z + curvature_deg_km_z"
                " + dist_to_intersection_m_z")
LEAN_FORMULA = "Y ~ np.log(trips + 1) + lanes_z + dist_to_intersection_m_z"
BASE_FORMULA = "Y ~ np.log(trips + 1)"

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def standardise(df, cols):
    for c in cols:
        z = c + "_z"
        if c in df.columns and z not in df.columns:
            sd = float(df[c].std())
            df[z] = (df[c] - float(df[c].mean())) / (sd if sd and sd > 0 else 1.0)
    return df


def try_nb(df, formula):
    """Fit fixed-effects NB; return (res, alpha, aic) or raise."""
    import statsmodels.api as sm  # noqa: F401 (kept for parity with script 41)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", HessianInversionWarning)
        res = smf.negativebinomial(formula, data=df).fit(
            disp=False, method="lbfgs", maxiter=2000)
    converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
    alpha = float(res.params["alpha"]) if "alpha" in res.params else float(res.scale)
    if not (converged and np.isfinite(res.aic) and 0 < alpha <= 100):
        raise RuntimeError(
            f"degenerate NB (converged={converged}, alpha={alpha:.3g}, AIC={res.aic})")
    lam = np.asarray(res.predict(df), dtype=float)
    if not np.all(np.isfinite(lam)):
        raise RuntimeError("non-finite NB predictions")
    obs, exp = float(df["Y"].sum()), float(lam.sum())
    if abs(exp - obs) / max(obs, 1) > MEAN_FID_TOL:
        raise RuntimeError(
            f"mean-fidelity fail: expected={exp:,.0f} vs observed={obs:,.0f}")
    return res, alpha, float(res.aic), lam


def poisson_moment_fallback(df):
    """Poisson-GLM (lean) + Pearson moment alpha. Totals match by
    construction (intercept score equation); no zero-process terms."""
    import statsmodels.api as sm
    res = smf.glm(LEAN_FORMULA, data=df,
                  family=sm.families.Poisson()).fit(disp=False)
    lam = np.asarray(res.fittedvalues, dtype=float)
    pearson = float(((res.resid_pearson) ** 2).sum() / max(res.df_resid, 1))
    alpha = float(max((pearson - 1.0) / max(np.mean(lam), 1e-9), 1e-3))
    return res, alpha, float(res.aic), lam, "POISSON-GLM-MOMENT-FALLBACK"


def fit_stratum(df):
    """Richest-first NB with convergence+fidelity guards; moment fallback."""
    tried = []
    for tag, f in (("full", FULL_FORMULA), ("lean", LEAN_FORMULA),
                   ("base", BASE_FORMULA)):
        try:
            res, alpha, aic, lam = try_nb(df, f)
            return {"tag": tag, "formula": f, "alpha": alpha,
                    "aic": aic, "lam": lam, "fallback": False,
                    "tried": tried + [tag + ":OK"]}
        except Exception as e:  # noqa: BLE001 - record and try leaner spec
            tried.append(f"{tag}:FAIL({e})")
    res, alpha, aic, lam, fb = poisson_moment_fallback(df)
    return {"tag": "poisson-moment", "formula": LEAN_FORMULA + " [" + fb + "]",
            "alpha": alpha, "aic": aic, "lam": lam, "fallback": True,
            "tried": tried + ["poisson-moment:OK"]}


def eb_adjust(y, lam, alpha):
    w = 1.0 / (1.0 + alpha * np.asarray(lam, dtype=float))
    return w * np.asarray(lam, dtype=float) + (1.0 - w) * np.asarray(y, dtype=float), w


def phase_clock_window(kickoff_h, phase):
    if phase == "pre":
        return (kickoff_h - 3.0, kickoff_h - 0.5)
    if phase == "during":
        return (kickoff_h - 0.5, kickoff_h + 1.5)
    return (kickoff_h + 1.5, kickoff_h + 4.0)


def plot_broken(ax, x_all, y_all, observed, **kw):
    """Plot contiguous observed blocks separately: no line bridges NaN gaps."""
    x_all = np.asarray(x_all)
    blocks, cur = [], []
    for h in range(24):
        if observed[h]:
            cur.append(h)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    for b in blocks:
        ax.plot(x_all[b], y_all[b], **kw)


def main():
    audit = {"estimand": "speeding-only threshold exceedances per segment x window",
             "no_zero_inflation": "ZI/hurdle excluded by design: all rows trips>0, "
             "zeros are at-risk sampling zeros, not structural safe states"}
    print("=" * 80)
    print("43_speeding_timewise_refit.py - speeding-only refit, no zero-inflation")
    print("=" * 80)

    df = pd.read_csv(SEGMENT_ROADS)
    # At-risk filter made explicit: every modelled row carries exposure, so a
    # zero can never be a structural "safe state" (the ZI identifying claim).
    df = df[df["trips"] > 0].copy()
    assert "speeding" in df.columns
    df["Y"] = df["speeding"].astype(float)
    audit["n_rows_at_risk"] = int(len(df))
    audit["total_speeding"] = float(df["Y"].sum())
    print(f"   at-risk rows={len(df):,} speeding={audit['total_speeding']:,.0f}")

    if os.path.exists(FEATURES):
        feat = pd.read_csv(FEATURES).drop(columns=["highway"], errors="ignore")
        df = df.merge(feat, on="road_osmid", how="left")
        for c in ["maxspeed_kmh", "lanes", "dist_to_intersection_m"]:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].median())
        df = standardise(df, GEO_COLS)
    df["highway"] = df.get("highway", "unknown").fillna("unknown").astype(str)
    df["ref_dt"] = pd.to_datetime(df["ref_dt"])
    df["kickoff_h"] = df["ref_dt"].dt.hour + df["ref_dt"].dt.minute / 60.0

    # ---- stratum fits (richest-first, guarded) ----
    fits, rows = {}, []
    for t in TYPES:
        for ph in PHASES:
            sub = df[(df["type"] == t) & (df["phase"] == ph)].copy()
            if len(sub) < 50:
                raise SystemExit(f"FATAL: stratum {t}/{ph} n={len(sub)} too small")
            fit = fit_stratum(sub)
            fits[(t, ph)] = fit
            sub["expected"] = fit["lam"]
            sub["eb"], sub["w"] = eb_adjust(sub["Y"].values, fit["lam"], fit["alpha"])
            obs, exp = float(sub["Y"].sum()), float(sub["expected"].sum())
            ebt = float(sub["eb"].sum())
            if abs(ebt - obs) / max(obs, 1) > EB_TOL:
                raise SystemExit(
                    f"FATAL: stratum {t}/{ph} EB total {ebt:,.0f} vs observed "
                    f"{obs:,.0f} exceeds {100 * EB_TOL:.0f}% - refusing artefacts.")
            print(f"   {t:7s}/{ph:6s}: n={len(sub):>6,} spec={fit['tag']:>15s} "
                  f"AIC={fit['aic']:>9.0f} alpha={fit['alpha']:.3f} "
                  f"obs={obs:>8,.0f} exp={exp:>8,.0f} EB={ebt:>8,.0f} "
                  f"wbar={float(sub['w'].mean()):.3f}"
                  f"{' [FALLBACK]' if fit['fallback'] else ''}")
            audit[f"{t}/{ph}"] = {
                "spec": fit["tag"], "aic": fit["aic"], "alpha": fit["alpha"],
                "obs": obs, "exp": exp, "eb": ebt,
                "wbar": float(sub["w"].mean()), "fallback": fit["fallback"]}
            g = sub.groupby("road_osmid").agg(
                Y=("Y", "sum"), exposure=("trips", "sum"),
                expected=("expected", "sum"), eb=("eb", "sum"),
                w=("w", "mean")).reset_index()
            g["type"] = t
            g["phase"] = ph
            rows.append(g)
    seg_phase = pd.concat(rows, ignore_index=True)
    seg_phase.to_csv(OUT_SEG_PHASE, index=False)

    # ---- hourly (NaN where unobserved; line broken at gaps) ----
    def build_hourly(sub_df, eb_phase_df):
        sub = sub_df.merge(eb_phase_df[["road_osmid", "phase", "eb"]],
                           on=["road_osmid", "phase"], how="left").dropna(subset=["eb"])
        hourly = {r: np.zeros(24) for r in sub["road_osmid"].unique()}
        counts = np.zeros(24)
        for _, r in sub.iterrows():
            a, b = phase_clock_window(r["kickoff_h"], r["phase"])
            hrs = np.arange(int(np.ceil(a)), int(np.floor(b)) + 1)
            if len(hrs) == 0:
                continue
            share = r["eb"] / len(hrs)
            for h in hrs:
                hourly[r["road_osmid"]][h % 24] += share
                counts[h % 24] += 1
        out = pd.DataFrame(
            [{"road_osmid": s, **{f"h{h:02d}": hourly[s][h] for h in range(24)}}
             for s in hourly])
        observed = counts > 0
        for h in range(24):
            out[f"h{h:02d}"] = out[f"h{h:02d}"] / counts[h] if observed[h] else np.nan
        return out, observed

    hourly_df, observed = build_hourly(
        df[df["type"] == "Event"], seg_phase[seg_phase["type"] == "Event"])
    hourly_df.to_csv(OUT_HOURLY, index=False)
    audit["unobserved_hours"] = [int(h) for h in range(24) if not observed[h]]
    audit["observed_hours"] = [int(h) for h in range(24) if observed[h]]
    print(f"   observed hours: {audit['observed_hours']}")
    print(f"   unobserved (NaN, gap in line): {audit['unobserved_hours']}")

    summary = seg_phase.groupby(["type", "phase"]).agg(
        segments=("road_osmid", "nunique"),
        total_speeding=("Y", "sum"),
        total_expected=("expected", "sum"),
        total_eb=("eb", "sum"),
        mean_weight=("w", "mean")).reset_index()
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))

    # ---- replacement Fig5 panels ----
    ev = summary[summary["type"] == "Event"].set_index("phase").reindex(PHASES)
    ct = summary[summary["type"] == "Control"].set_index("phase").reindex(PHASES)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(PHASES))
    ax.plot(x, ev["total_eb"].values, "o-", label="Event", lw=2)
    ax.plot(x, ct["total_eb"].values, "s--", label="Control", lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PHASES])
    ax.set_xlabel("Event window phase (pre: KO-3h..KO-0.5h; during: KO-0.5h..KO+1.5h; post: KO+1.5h..KO+4h)")
    ax.set_ylabel("Total EB-adjusted SPEEDING events (threshold exceedances)")
    ax.set_title("Phase-resolved speeding: Event vs Control (speeding-only EB)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PHASE, dpi=150)
    plt.close(fig)

    mean_hour = hourly_df[[f"h{h:02d}" for h in range(24)]].mean().values
    xx = np.arange(24)
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_broken(ax, xx, mean_hour, observed, marker="o", lw=2, color="firebrick")
    ax.set_xlabel("Clock hour (only hours covered by a fixture window; gaps = no observation, never zero)")
    ax.set_ylabel("Mean EB speeding events per segment per contributing window")
    ax.set_title("Hourly speeding profile (speeding-only; averaged over contributing event windows)")
    ax.set_xticks(audit["observed_hours"])
    ax.set_xticklabels([str(h) for h in audit["observed_hours"]])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_HOURLY, dpi=150)
    plt.close(fig)

    with open(OUT_TEX, "w") as f:
        f.write("% speeding_timewise_refit.tex - auto-generated by 43_speeding_timewise_refit.py\n")
        f.write("% Speeding-only EB phase totals (no zero-inflation; EB within 5% of observed).\n")
        for _, r in summary.iterrows():
            f.write(f"% {r['type']}/{r['phase']}: speeding={r['total_speeding']:,.0f} "
                    f"EB={r['total_eb']:,.0f} w={r['mean_weight']:.3f}\n")
    audit["outputs"] = [OUT_SEG_PHASE, OUT_HOURLY, OUT_SUMMARY, OUT_TEX,
                        FIG_PHASE, FIG_HOURLY]
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print("DONE! Speeding-only refit complete (no zero-inflation).")


if __name__ == "__main__":
    main()
