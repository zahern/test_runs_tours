"""Temporal instability tests for the telematics panel (Jan 2025 - Mar 2026).

Grid cells (0.01 deg) x calendar quarter -> HB counts. NB intercept-only MLE:
  (a) pooled (const, alpha) vs per-quarter separate: Chow-type LR, df=8
  (b) pooled + quarter dummies (common alpha) vs pooled: mean-shift LR, df=4
Crash counts are 5-year averages without timestamps, so instability testing
applies to the telematics panel only.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_min
from scipy import stats as scipy_stats

HERE = os.path.dirname(os.path.abspath(__file__))
_TELE_CANDS = [
    os.path.join(HERE, "hard_braking_events_year.csv"),
    "/mnt/hpccs01/home/ahernz/OneDrive - Queensland University of Technology/Desktop/telematics_data/hard_braking_events_year.csv",
]
TELE = next((p for p in _TELE_CANDS if os.path.exists(p)), _TELE_CANDS[-1])
OUT_TEX = os.path.join(HERE, "temporal_stability.tex")
OUT_CSV = os.path.join(HERE, "temporal_stability.csv")


def nb_ll(params, y):
    const, ln_a = params
    mu = np.exp(const)
    a = np.exp(ln_a)
    r = 1.0 / a
    p = 1.0 / (1.0 + a * mu)
    return float(np.sum(scipy_stats.nbinom.logpmf(y.astype(int), r, p)))


def fit_nb(y):
    r = scipy_min(lambda p: -nb_ll(p, y), [np.log(max(y.mean(), 0.1)), 0.0],
                  method="L-BFGS-B", options={"maxiter": 2000})
    const, ln_a = (float(x) for x in r.x)
    return {"mu": float(np.exp(const)), "alpha": float(np.exp(ln_a)),
            "ll": float(-r.fun), "k": 2, "n": len(y), "ok": bool(r.success)}


def main():
    print("Loading year events ...", flush=True)
    df = pd.read_csv(TELE, usecols=["EventStart", "SnappedLongitude",
                                    "SnappedLatitude", "VehicleID"],
                     low_memory=False)
    df["EventStart"] = pd.to_datetime(df["EventStart"], errors="coerce")
    df = df.dropna(subset=["EventStart", "SnappedLongitude", "SnappedLatitude"])
    df = df[(df["EventStart"] >= "2025-01-01") & (df["EventStart"] < "2026-04-01")]
    df["quarter"] = df["EventStart"].dt.to_period("Q").astype(str)
    print(f"  {len(df):,} events, quarters: {sorted(df['quarter'].unique())}", flush=True)

    df["gx"] = (df["SnappedLongitude"] / 0.01).round().astype(int)
    df["gy"] = (df["SnappedLatitude"] / 0.01).round().astype(int)
    df["grid"] = df["gx"].astype(str) + "_" + df["gy"].astype(str)
    panel = df.groupby(["grid", "quarter"]).size().reset_index(name="cnt")
    print(f"  panel: {panel['grid'].nunique()} cells, {len(panel):,} cell-quarters", flush=True)

    qs = sorted(panel["quarter"].unique())
    per_q = {}
    for q in qs:
        y = panel.loc[panel["quarter"] == q, "cnt"].values
        per_q[q] = fit_nb(y)
        print(f"  {q}: n={len(y)} mu={per_q[q]['mu']:.2f} alpha={per_q[q]['alpha']:.3f} LL={per_q[q]['ll']:.1f}", flush=True)

    y_all = panel["cnt"].values
    pooled = fit_nb(y_all)
    ll_sep = sum(per_q[q]["ll"] for q in qs)
    lr_a = 2 * (ll_sep - pooled["ll"])
    df_a = 2 * len(qs) - 2
    p_a = float(1 - scipy_stats.chi2.cdf(lr_a, df_a))

    # quarter-dummy pooled model (common alpha): design [1, dummies]
    D = pd.get_dummies(panel["quarter"], drop_first=True).values.astype(float)
    X = np.column_stack([np.ones(len(panel)), D])

    def nb_ll_d(p, y, X):
        mu = np.exp(X @ p[:-1])
        a = np.exp(p[-1])
        r = 1.0 / a
        pr = 1.0 / (1.0 + a * mu)
        return float(np.sum(scipy_stats.nbinom.logpmf(y.astype(int), r, pr)))

    y = panel["cnt"].values
    r = scipy_min(lambda p: -nb_ll_d(p, y, X),
                  np.zeros(X.shape[1] + 1), method="L-BFGS-B",
                  options={"maxiter": 2000})
    ll_d = float(-r.fun)
    lr_b = 2 * (ll_d - pooled["ll"])
    df_b = X.shape[1] - 1
    p_b = float(1 - scipy_stats.chi2.cdf(lr_b, df_b))
    print(f"Pooled: LL={pooled['ll']:.1f} alpha={pooled['alpha']:.3f}", flush=True)
    print(f"LR-a (separate vs pooled): {lr_a:.1f} df={df_a} p={p_a:.4g}", flush=True)
    print(f"LR-b (quarter dummies vs pooled): {lr_b:.1f} df={df_b} p={p_b:.4g}", flush=True)

    rows = [{"quarter": q, "cells": int((panel['quarter'] == q).sum()),
             "mean_cnt": per_q[q]["mu"], "alpha": per_q[q]["alpha"],
             "ll": per_q[q]["ll"]} for q in qs]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[!ht]\n")
        f.write("  \\caption{Temporal instability: grid-cell HB counts by calendar quarter, Jan 2025--Mar 2026 (NB intercept-only per quarter)}\\label{tab:temporal_stability}\n")
        f.write("  \\begin{center}\n    \\begin{tabular}{lrrrr}\n      \\toprule\n")
        f.write("      Quarter & Cells & Mean count & $\\hat\\alpha$ & $\\log L$ \\\\\n      \\midrule\n")
        for r_ in rows:
            f.write(f"      {r_['quarter']} & {r_['cells']:,} & {r_['mean_cnt']:.2f} & {r_['alpha']:.3f} & {r_['ll']:,.1f} \\\\\n")
        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
        f.write(f"  \\emph{{Note:}} Pooled NB: $\\log L={pooled['ll']:,.1f}$, $\\hat\\alpha={pooled['alpha']:.3f}$. "
                f"Chow-type LR (separate vs.\\ pooled): ${lr_a:,.1f}$ ($df={df_a}$, $p={p_a:.3g}$); "
                f"quarter-dummy LR: ${lr_b:,.1f}$ ($df={df_b}$, $p={p_b:.3g}$). "
                f"Crash counts are 5-year averages without timestamps, so instability testing applies to the telematics panel only.\n")
        f.write("\\end{table}\n")
    print("Wrote temporal_stability.tex / .csv", flush=True)


if __name__ == "__main__":
    main()
