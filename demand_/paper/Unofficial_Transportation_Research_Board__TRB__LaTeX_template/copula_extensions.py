"""Extended copula variants (IFM) with fixed NB margins.

Adds to the Frank / Gaussian / Kimeldorf baseline in nearmiss_hb_analysis:
  1. Joe copula (upper-tail dependence for joint extremes)
  2. Gaussian-Joe mixture (weight + both dependence params estimated)
  3. Parameterized Joe: theta_i = 1 + exp(g0 + g1*AADT + g2*SL)
  4. Random-parameter margins + Gaussian-Joe mixture
All fits use the discrete rectangle likelihood; LR tests vs independence.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_min
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nearmiss_hb_analysis import (
    fit_model, _nb_cdf, _joint_pmf_vec, _rebuild_mu, _normal_cop,
)
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "near_miss_hb_ecf.csv")
OUT_TEX = os.path.join(HERE, "copula_joe.tex")
OUT_CSV = os.path.join(HERE, "copula_joe.csv")

HB_FEATS = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
            "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5", "RSDSO", "RSDSDL5"]
CRASH_FEATS = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]
HB_RDM = ["IC:normal", "IV_M:normal"]
HB_RDM_COR = ["PA_COM1:normal", "RSDSO:normal", "RSDSDL5:normal"]
CR_RDM = ["PSDS1T24:normal", "RSDSDG10:normal"]
CR_RDM_COR = ["SL:normal", "RSDSDL5:normal"]

EPS = 1e-10


def _joe_cop(u, v, theta):
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    if float(np.min(theta) if np.ndim(theta) else theta) < 1.0 + 1e-8:
        return u * v
    th = theta
    a = (1.0 - u) ** th + (1.0 - v) ** th - ((1.0 - u) * (1.0 - v)) ** th
    return 1.0 - np.maximum(a, 1e-300) ** (1.0 / th)


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
    n = len(df_c)
    indep = _joint_pmf_vec(u1c, u1p, u2c, u2p, lambda u, v, t: u * v, 0.0)
    ll_indep = float(np.sum(np.log(np.maximum(indep, EPS))))
    return dict(u1c=u1c, u1p=u1p, u2c=u2c, u2p=u2p, n=n, ll_indep=ll_indep,
                k_indep=nm_model["k"] + crash_model["k"])


def _fit_report(name, ll_joint, m, ndep):
    n = m["n"]
    k = m["k_indep"] + ndep
    bic = k * np.log(n) - 2 * ll_joint
    bic0 = m["k_indep"] * np.log(n) - 2 * m["ll_indep"]
    lr = 2 * (ll_joint - m["ll_indep"])
    p = float(1 - scipy_stats.chi2.cdf(max(lr, 0.0), df=ndep))
    return {"model": name, "ll": ll_joint, "bic": bic, "k": k,
            "dBIC": bic - bic0, "lr": lr, "lr_p": p, "n": n}


def main():
    df = pd.read_csv(DATA, low_memory=False)
    for c in sorted(set(HB_FEATS) | set(CRASH_FEATS)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df)} rows", flush=True)

    print("\nFitting fixed margins ...", flush=True)
    m_hb = fit_model(df, HB_FEATS, "HB_Total", "log_len", label="HBfx")
    m_cr = fit_model(df, CRASH_FEATS, "ObsCrashes", "log_len", label="CRfx")
    print("Fitting random margins ...", flush=True)
    m_hb_r = fit_model(df, HB_FEATS, "HB_Total", "log_len", label="HBr",
                       rdm_terms=HB_RDM, rdm_cor_terms=HB_RDM_COR, draws_R=200)
    m_cr_r = fit_model(df, CRASH_FEATS, "ObsCrashes", "log_len", label="CRr",
                       rdm_terms=CR_RDM, rdm_cor_terms=CR_RDM_COR, draws_R=200)

    M = _margins(df, m_hb, m_cr)
    Mr = _margins(df, m_hb_r, m_cr_r)
    u1c, u1p, u2c, u2p = M["u1c"], M["u1p"], M["u2c"], M["u2p"]
    rows = []

    # 1. Joe (1-D)
    def neg_joe(t):
        pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, _joe_cop, float(t[0]))
        return -float(np.sum(np.log(pmf)))
    r = scipy_min(neg_joe, [2.0], method="L-BFGS-B", bounds=[(1.001, 20.0)],
                  options={"maxiter": 500})
    th_j = float(r.x[0])
    ll_j = -float(r.fun)
    rows.append(_fit_report("Joe", ll_j, M, 1))
    print(f"Joe: theta={th_j:.3f} LL={ll_j:.1f}", flush=True)

    # 2. Gaussian-Joe mixture (w, rho, thetaJ)
    def neg_mix(t):
        w, rho, thj = float(t[0]), float(t[1]), float(t[2])
        pg = _joint_pmf_vec(u1c, u1p, u2c, u2p, _normal_cop, rho)
        pj = _joint_pmf_vec(u1c, u1p, u2c, u2p, _joe_cop, thj)
        pmf = w * pg + (1 - w) * pj
        return -float(np.sum(np.log(np.maximum(pmf, EPS))))
    r = scipy_min(neg_mix, [0.5, 0.3, 2.0], method="L-BFGS-B",
                  bounds=[(0.01, 0.99), (-0.9, 0.9), (1.001, 20.0)],
                  options={"maxiter": 300})
    w_m, rho_m, thj_m = (float(x) for x in r.x)
    ll_m = -float(r.fun)
    rows.append(_fit_report("Gaussian-Joe mixture", ll_m, M, 3))
    print(f"Mix: w={w_m:.3f} rho={rho_m:.3f} thetaJ={thj_m:.3f} LL={ll_m:.1f}", flush=True)

    # 3. Parameterized Joe: theta_i = 1 + exp(g0 + g1*AADT_s + g2*SL_s)
    sc = StandardScaler()
    Z = sc.fit_transform(df[["AADT", "SL"]].values.astype(float))
    # align Z rows with margin sample: rebuild index via same dropna
    all_feats = list(set(HB_FEATS) | set(CRASH_FEATS))
    df_c = df.dropna(subset=all_feats + ["HB_Total", "ObsCrashes", "log_len"]).reset_index(drop=True)
    Zc = StandardScaler().fit_transform(df_c[["AADT", "SL"]].values.astype(float))

    def neg_pjoe(g):
        th = 1.0 + np.exp(g[0] + g[1] * Zc[:, 0] + g[2] * Zc[:, 1])
        c11 = _joe_cop(u1c, u2c, th); c10 = _joe_cop(u1c, u2p, th)
        c01 = _joe_cop(u1p, u2c, th); c00 = _joe_cop(u1p, u2p, th)
        pmf = np.maximum(c11 - c10 - c01 + c00, EPS)
        return -float(np.sum(np.log(pmf)))
    r = scipy_min(neg_pjoe, [0.5, 0.0, 0.0], method="L-BFGS-B",
                  options={"maxiter": 300})
    g = [float(x) for x in r.x]
    ll_p = -float(r.fun)
    rows.append(_fit_report("Joe, parameterized dependence", ll_p, M, 3))
    print(f"PJoe: g={g} LL={ll_p:.1f}", flush=True)

    # 4. Random margins + Gaussian-Joe mixture
    v1c, v1p, v2c, v2p = Mr["u1c"], Mr["u1p"], Mr["u2c"], Mr["u2p"]

    def neg_mixr(t):
        w, rho, thj = float(t[0]), float(t[1]), float(t[2])
        pg = _joint_pmf_vec(v1c, v1p, v2c, v2p, _normal_cop, rho)
        pj = _joint_pmf_vec(v1c, v1p, v2c, v2p, _joe_cop, thj)
        pmf = w * pg + (1 - w) * pj
        return -float(np.sum(np.log(np.maximum(pmf, EPS))))
    r = scipy_min(neg_mixr, [0.5, 0.3, 2.0], method="L-BFGS-B",
                  bounds=[(0.01, 0.99), (-0.9, 0.9), (1.001, 20.0)],
                  options={"maxiter": 300})
    w_r, rho_r, thj_r = (float(x) for x in r.x)
    ll_mr = -float(r.fun)
    rows.append(_fit_report("RP margins + Gaussian-Joe mixture", ll_mr, Mr, 3))
    print(f"RP-Mix: w={w_r:.3f} rho={rho_r:.3f} thetaJ={thj_r:.3f} LL={ll_mr:.1f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    params = {"joe_theta": th_j, "mix_w": w_m, "mix_rho": rho_m,
              "mix_thetaJ": thj_m, "pjoe_g": g, "rpmix_w": w_r,
              "rpmix_rho": rho_r, "rpmix_thetaJ": thj_r}
    pd.DataFrame([params]).to_csv(os.path.join(HERE, "copula_joe_params.csv"), index=False)

    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[!ht]\n")
        f.write("  \\caption{Extended copula variants: Joe, Gaussian--Joe mixture, parameterized dependence, and random-parameter margins (IFM, $N=10{,}780$)}\\label{tab:copula_joe}\n")
        f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrr}\n      \\toprule\n")
        f.write("      Model & $\\log L$ & BIC & $\\Delta$BIC vs.\\ indep. & LR & $p$ \\\\\n      \\midrule\n")
        for _, r_ in res.iterrows():
            f.write(f"      {r_['model']} & {r_['ll']:,.1f} & {r_['bic']:,.1f} & {r_['dBIC']:+,.1f} & {r_['lr']:,.1f} & {r_['lr_p']:.3f} \\\\\n")
        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
        f.write(f"  \\emph{{Note:}} Mixture: weight $w={w_m:.2f}$ on Gaussian ($\\rho={rho_m:.2f}$), $1-w$ on Joe ($\\theta={thj_m:.2f}$). Parameterized Joe: $\\theta_i=1+\\exp({g[0]:.2f}{g[1]:+.2f}\\cdot\\mathrm{{AADT}}_i{g[2]:+.2f}\\cdot\\mathrm{{SL}}_i)$. RP margins: random-parameter NB margins (\\Cref{{tab:rp_check}}) with mixture $w={w_r:.2f}$, $\\rho={rho_r:.2f}$, $\\theta={thj_r:.2f}$.\n")
        f.write("\\end{table}\n")
    print("Wrote copula_joe.tex / copula_joe.csv", flush=True)


if __name__ == "__main__":
    main()
