"""Rectangle-identity check for the discrete NB copula likelihood.

Mathematical identity under test: with the independence copula C(u,v)=uv,
the rectangle pmf factorises EXACTLY into marginal pmfs,

    P(Y1=y1, Y2=y2) = P1(y1) * P2(y2),

so on any fixed sample the independence-rectangle log-likelihood MUST equal
the sum of the two marginal NB log-likelihoods (up to ~1e-6).

Observed violation (jobs 25256286--25256289, N=10,780 DE-spec margins):
    rectangle-independence LL = -81,231.2  (copula_classic.csv, ll_indep)
    marginal-LL sum           = -65,517.7  (HB -48,811.4 + Crash -16,706.3)
    gap ~= 15,713 log-points -- impossible for a correct implementation.

This script decomposes the gap on the current sample:
  A = marginal LL sum via exact NB2 gammaln log-pmf (reference; no scipy)
  B = independence-rectangle LL via _nb_cdf + _joint_pmf_vec (as reported)
  C = marginal-pmf-product LL via _nb_pmf (scipy; isolates scipy error)
per margin (HB alpha~69 / Crash alpha~17) and worst observations, plus the
three copula-rectangle LLs for context. If B != A but C == A, the CDF path
(catastrophic cancellation in cdf(y)-cdf(y-1) at tiny r=1/alpha) is at
fault; if C != A too, scipy nbinom itself is inaccurate at r << 1 and every
rectangle LL (Joe table included) is suspect.

Usage (HPC): python -u copula_identity_check.py [--data ...] [--out-dir ...]
Writes copula_identity.csv + prints verdict. CPU-only, no SA/DE search.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nearmiss_hb_analysis import (
    _joint_pmf_vec,
    _nb_cdf,
    _nb_pmf,
    _rebuild_mu,
    fit_model,
)
from refit_sparsea_agds import load_like_main

HERE = os.path.dirname(os.path.abspath(__file__))
HB_FEATS = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
            "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5", "RSDSO", "RSDSDL5"]
CRASH_FEATS = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]
EPS = 1e-10


def nb2_logpmf(y, mu, alpha):
    """Exact NB2 log-pmf in log space (reference; Var = mu + alpha*mu^2)."""
    r = 1.0 / np.asarray(alpha, dtype=float)
    p = 1.0 / (1.0 + np.asarray(alpha, dtype=float) * np.asarray(mu, dtype=float))
    y = np.asarray(y, dtype=float)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(p) + y * np.log1p(-p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "near_miss_hb_ecf.csv"))
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    df, _ = load_like_main(args.data)
    nm_col = "nm_total" if "nm_total" in df.columns else "HB_Total"
    for c in sorted(set(HB_FEATS) | set(CRASH_FEATS)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df)} rows; HB col={nm_col}", flush=True)

    m_hb = fit_model(df, HB_FEATS, nm_col, "log_len", label="ID-HB")
    m_cr = fit_model(df, CRASH_FEATS, "ObsCrashes", "log_len", label="ID-CR")

    feats = sorted(set(HB_FEATS) | set(CRASH_FEATS))
    df_c = df.dropna(subset=feats + [nm_col, "ObsCrashes", "log_len"]).reset_index(drop=True)
    mu1, a1 = _rebuild_mu(df_c, m_hb)
    mu2, a2 = _rebuild_mu(df_c, m_cr)
    y1 = df_c[nm_col].values.astype(int)
    y2 = df_c["ObsCrashes"].values.astype(int)
    n = len(df_c)
    print(f"joint sample n={n}; alpha_HB={a1:.3f} alpha_CR={a2:.3f}", flush=True)

    # A: exact reference marginal LLs on the JOINT sample
    A1 = float(nb2_logpmf(y1, mu1, a1).sum())
    A2 = float(nb2_logpmf(y2, mu2, a2).sum())
    A = A1 + A2

    # C: scipy-pmf-product marginal LLs (isolates scipy nbinom error)
    C1 = float(np.log(np.maximum(_nb_pmf(y1, mu1, a1), 1e-300)).sum())
    C2 = float(np.log(np.maximum(_nb_pmf(y2, mu2, a2), 1e-300)).sum())
    C = C1 + C2

    # B: independence-rectangle LL via the reported CDF path
    u1c = np.clip(_nb_cdf(y1, mu1, a1), EPS, 1 - EPS)
    u1p = np.where(y1 > 0, np.clip(_nb_cdf(y1 - 1, mu1, a1), 0.0, 1 - EPS), 0.0)
    u2c = np.clip(_nb_cdf(y2, mu2, a2), EPS, 1 - EPS)
    u2p = np.where(y2 > 0, np.clip(_nb_cdf(y2 - 1, mu2, a2), 0.0, 1 - EPS), 0.0)
    rect = _joint_pmf_vec(u1c, u1p, u2c, u2p, lambda u, v, t: u * v, 0.0)
    B = float(np.log(np.maximum(rect, EPS)).sum())

    # per-margin rectangle check: P1 rect pmf = u1c-u1p vs exact pmf
    r1 = np.maximum(u1c - u1p, 1e-300)
    r2 = np.maximum(u2c - u2p, 1e-300)
    B1 = float(np.log(np.maximum(r1, EPS)).sum())
    B2 = float(np.log(np.maximum(r2, EPS)).sum())

    print(f"A  exact marginal sum      = {A:,.1f}  (HB {A1:,.1f} + CR {A2:,.1f})", flush=True)
    print(f"C  scipy pmf-product sum   = {C:,.1f}  (HB {C1:,.1f} + CR {C2:,.1f})", flush=True)
    print(f"B  indep-rectangle (CDF)   = {B:,.1f}  (HB {B1:,.1f} + CR {B2:,.1f})", flush=True)
    print(f"gap B-A = {B - A:+,.1f} | C-A = {C - A:+,.1f} "
          f"| HB part {B1 - A1:+,.1f} | CR part {B2 - A2:+,.1f}", flush=True)

    worst = np.argsort(np.log(np.maximum(rect, EPS)))[:10]
    aud = pd.DataFrame({
        "y_hb": y1[worst], "y_crash": y2[worst],
        "mu_hb": np.round(mu1[worst], 2), "mu_crash": np.round(mu2[worst], 3),
        "logpmf_exact": np.round(nb2_logpmf(y1[worst], mu1[worst], a1)
                                 + nb2_logpmf(y2[worst], mu2[worst], a2), 2),
        "logpmf_rect": np.round(np.log(np.maximum(rect[worst], EPS)), 2)})
    print(aud.to_string(index=False), flush=True)
    aud.to_csv(os.path.join(args.out_dir, "copula_identity.csv"), index=False)

    if abs(B - A) > 1.0:
        print(f"VERDICT: IDENTITY VIOLATED (gap {B - A:+,.1f} >> 1e-6 tolerance). "
              "Rectangle LLs (Joe table included) are unreliable until fixed.", flush=True)
    else:
        print("VERDICT: identity holds; rectangle path is sound.", flush=True)


if __name__ == "__main__":
    main()
