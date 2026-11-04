"""Fix for the classic copula IFM anomaly (negative LRs vs nested independence).

Re-estimates Frank / Gaussian / Kimeldorf-Sampson by IFM on the DE-spec
margins (the same margins behind tab:copula_joe, where evaluation is
numerically well-behaved), with multi-start optimization and a clip audit:
counts observations where the rectangle pmf hits the 1e-300 floor, which is
the underflow signature behind the old failures.
Regenerates copula_joe.tex with classic + Joe-family rows in one table.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_min
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nearmiss_hb_analysis import (
    fit_model, _nb_cdf, _joint_pmf_vec, _rebuild_mu,
    _COPULA_FNS, _COPULA_BOUNDS,
)
from copula_extensions import _joe_cop

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "near_miss_hb_ecf.csv")
OUT_TEX = os.path.join(HERE, "copula_joe.tex")
OUT_CSV = os.path.join(HERE, "copula_joe.csv")

HB_FEATS = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
            "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5", "RSDSO", "RSDSDL5"]
CRASH_FEATS = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]
EPS = 1e-10
FLOOR = 1e-300


def _margins(df, nm_model, crash_model, nm_col="HB_Total"):
    all_feats = list(set(nm_model["features"]) | set(crash_model["features"]))
    df_c = df.dropna(subset=all_feats + [nm_col, "ObsCrashes", "log_len"]).reset_index(drop=True)
    mu1, alpha1 = _rebuild_mu(df_c, nm_model)
    mu2, alpha2 = _rebuild_mu(df_c, crash_model)
    y1 = df_c[nm_col].values.astype(int)
    y2 = df_c["ObsCrashes"].values.astype(int)
    u1c = np.clip(_nb_cdf(y1, mu1, alpha1), EPS, 1 - EPS)
    u1p = np.where(y1 > 0, np.clip(_nb_cdf(y1 - 1, mu1, alpha1), 0.0, 1 - EPS), 0.0)
    u2c = np.clip(_nb_cdf(y2, mu2, alpha2), EPS, 1 - EPS)
    u2p = np.where(y2 > 0, np.clip(_nb_cdf(y2 - 1, mu2, alpha2), 0.0, 1 - EPS), 0.0)
    return dict(u1c=u1c, u1p=u1p, u2c=u2c, u2p=u2p, n=len(df_c),
                y1=y1, y2=y2, mu1=mu1, mu2=mu2)


def fit_ifm(M, cop_fn, bounds, starts, ndep, k_indep):
    u1c, u1p, u2c, u2p = M["u1c"], M["u1p"], M["u2c"], M["u2p"]
    indep = _joint_pmf_vec(u1c, u1p, u2c, u2p, lambda u, v, t: u * v, 0.0)
    ll_indep = float(np.sum(np.log(np.maximum(indep, EPS))))
    n = M["n"]

    def neg_ll(t):
        pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, float(t[0]))
        return -float(np.sum(np.log(pmf)))

    best = (np.inf, None)
    for s in starts:
        r = scipy_min(neg_ll, [s], method="L-BFGS-B", bounds=[bounds],
                      options={"maxiter": 500})
        if float(r.fun) < best[0]:
            best = (float(r.fun), float(r.x[0]))
    th, ll = best[1], -best[0]
    pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, th)
    nclip = int((pmf <= FLOOR * 1.000001).sum())
    k = k_indep + ndep
    bic = k * np.log(n) - 2 * ll
    bic0 = k_indep * np.log(n) - 2 * ll_indep
    lr = 2 * (ll - ll_indep)
    p = float(1 - scipy_stats.chi2.cdf(max(lr, 0.0), df=ndep))
    return {"theta": th, "ll": ll, "bic": bic, "dBIC": bic - bic0,
            "lr": lr, "lr_p": p, "nclip": nclip, "ll_indep": ll_indep, "n": n}


def main():
    df = pd.read_csv(DATA, low_memory=False)
    for c in sorted(set(HB_FEATS) | set(CRASH_FEATS)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df)} rows", flush=True)

    print("Fitting fixed margins ...", flush=True)
    m_hb = fit_model(df, HB_FEATS, "HB_Total", "log_len", label="HBfx")
    m_cr = fit_model(df, CRASH_FEATS, "ObsCrashes", "log_len", label="CRfx")
    M = _margins(df, m_hb, m_cr)
    k_indep = m_hb["k"] + m_cr["k"]
    print(f"margin sample n={M['n']}, k_indep={k_indep}", flush=True)

    rows = []
    starts = {"frank": [0.5, 1.0, 3.0], "normal": [-0.3, 0.3, 0.6],
              "kimeldorf": [0.2, 0.5, 2.0]}
    for name in ["frank", "normal", "kimeldorf"]:
        lo, hi = _COPULA_BOUNDS[name]
        r = fit_ifm(M, _COPULA_FNS[name], (lo, hi), starts[name], 1, k_indep)
        r["model"] = {"frank": "Frank", "normal": "Gaussian",
                      "kimeldorf": "Kimeldorf--Sampson"}[name]
        rows.append(r)
        print(f"{r['model']}: theta={r['theta']:.3f} LL={r['ll']:.1f} "
              f"dBIC={r['dBIC']:+.1f} LR={r['lr']:.1f} p={r['lr_p']:.4g} "
              f"nclip={r['nclip']}", flush=True)

    # audit: worst observations under independence (tail drivers)
    indep = _joint_pmf_vec(M["u1c"], M["u1p"], M["u2c"], M["u2p"],
                           lambda u, v, t: u * v, 0.0)
    ll_i = np.log(np.maximum(indep, EPS))
    worst = np.argsort(ll_i)[:10]
    aud = pd.DataFrame({"idx": worst, "y_hb": M["y1"][worst],
                        "y_crash": M["y2"][worst],
                        "mu_hb": np.round(M["mu1"][worst], 2),
                        "mu_crash": np.round(M["mu2"][worst], 3),
                        "ll_indep": np.round(ll_i[worst], 2)})
    aud.to_csv(os.path.join(HERE, "copula_clip_audit.csv"), index=False)
    print("Worst-10 tail observations:", flush=True)
    print(aud.to_string(index=False), flush=True)

    # Joe on same margins (single value for the combined table)
    rj = fit_ifm(M, _joe_cop, (1.001, 20.0), [1.5, 2.0, 4.0], 1, k_indep)
    print(f"Joe check: theta={rj['theta']:.3f} LL={rj['ll']:.1f} dBIC={rj['dBIC']:+.1f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "copula_classic.csv"), index=False)

    # rebuild combined table: classics + joe family (read joe rows)
    joe = pd.read_csv(os.path.join(HERE, "copula_joe.csv"))
    joe = joe[~joe["model"].isin(res["model"])]
    full = pd.concat([res[["model", "ll", "bic", "dBIC", "lr", "lr_p"]],
                      joe[["model", "ll", "bic", "dBIC", "lr", "lr_p"]]],
                     ignore_index=True)
    full.to_csv(os.path.join(HERE, "copula_all.csv"), index=False)

    P = pd.read_csv(os.path.join(HERE, "copula_joe_params.csv")).iloc[0]
    g = eval(str(P["pjoe_g"]))
    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[!ht]\n")
        f.write("  \\caption{Copula dependence (IFM, DE-spec margins, $N=10{,}780$): classic families with stabilized evaluation plus Joe-family extensions}\\label{tab:copula_joe}\n")
        f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrr}\n      \\toprule\n")
        f.write("      Model & $\\log L$ & BIC & $\\Delta$BIC vs.\\ indep. & LR & $p$ \\\\\n      \\midrule\n")
        for _, r_ in full.iterrows():
            f.write(f"      {r_['model']} & {r_['ll']:,.1f} & {r_['bic']:,.1f} & {r_['dBIC']:+,.1f} & {r_['lr']:,.1f} & {r_['lr_p']:.3f} \\\\\n")
        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
        f.write("  \\emph{Note:} Classics re-estimated with multi-start optimization; floored rectangle evaluations: "
                + ", ".join(f"{r_['model']}={int(r_['nclip'])}" for _, r_ in res.iterrows())
                + f". Mixture: weight $w={P['mix_w']:.2f}$ on Gaussian ($\\rho={P['mix_rho']:.2f}$), $1-w$ on Joe ($\\theta={P['mix_thetaJ']:.2f}$). "
                + f"Parameterized Joe: $\\theta_i=1+\\exp({g[0]:.2f}{g[1]:+.2f}\\cdot\\mathrm{{AADT}}_i{g[2]:+.2f}\\cdot\\mathrm{{SL}}_i)$. "
                + f"RP margins: mixture $w={P['rpmix_w']:.2f}$, $\\rho={P['rpmix_rho']:.2f}$, $\\theta={P['rpmix_thetaJ']:.2f}$.\n")
        f.write("\\end{table}\n")
    print("Wrote copula_joe.tex (combined) / copula_all.csv", flush=True)


if __name__ == "__main__":
    main()
