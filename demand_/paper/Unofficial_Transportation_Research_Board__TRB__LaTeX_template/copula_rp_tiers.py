"""Copula coverage for the remaining parts.

A. Random-parameter margins x {Frank, Gaussian, Clayton, Joe} (+ mixture
   already done in copula_extensions.py; re-reported here for one table).
B. Severity-tier copulas (candidate / critical / extreme vs crashes) x
   {Frank, Gaussian, Clayton, Joe, Gaussian-Joe mixture} with fixed NB
   tier margins (DE HB spec) and DE crash margins.
All IFM with multi-start; LR vs independence; floored-evaluation audit.
"""
import os, sys

# Harden for CPU-only HPC nodes. Must execute before the jax import inside
# nearmiss_hb_analysis (imported below). The .pbs files also export these;
# this is the fallback when run without them -- the likely cause of the
# 09-Sep copula_rpt death right after "Fitting aggregate margins ..."
# (JAX attempted cuInit on an ngpus=0 node, then the draws_R=200 RP fits
# ran without the XLA memory caps).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_min
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _metacount_source import METACOUNT_SOURCE
from nearmiss_hb_analysis import (
    fit_model, _nb_cdf, _joint_pmf_vec, _rebuild_mu,
    _COPULA_FNS, _COPULA_BOUNDS, _normal_cop,
)
from copula_extensions import _joe_cop
from exposure import OFF, add_exposure_offsets

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "near_miss_hb_ecf.csv")

MCR_MIN_VERSION = (1, 0, 151)


def check_mcr_version():
    """Gate on metacountregressor>=1.0.151 (the estimator behind fit_model).

    The .pbs files pip-install the pin before launching; this fails loudly
    instead of silently running stale estimators if the install did not take.
    """
    import metacountregressor as mcr
    if METACOUNT_SOURCE:
        ver = getattr(mcr, "__version__", "0.0.0")
    else:
        try:
            from importlib.metadata import version as _pkg_version
            ver = _pkg_version("metacountregressor")
        except Exception:
            ver = getattr(mcr, "__version__", "0.0.0")
    try:
        tup = tuple(int(x) for x in ver.split(".")[:3])
    except ValueError:
        tup = (0, 0, 0)
        print(f"metacountregressor {ver} from {mcr.__file__} (required >= 1.0.151)",
            flush=True)
    if tup < MCR_MIN_VERSION:
        print(f"FATAL: metacountregressor {ver} < 1.0.151 -- refusing to run",
              flush=True)
        sys.exit(2)
    return ver

HB_FEATS = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
            "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5", "RSDSO", "RSDSDL5"]
CRASH_FEATS = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]
HB_RDM = ["IC:normal", "IV_M:normal"]
HB_RDM_COR = ["PA_COM1:normal", "RSDSO:normal", "RSDSDL5:normal"]
CR_RDM = ["PSDS1T24:normal", "RSDSDG10:normal"]
CR_RDM_COR = ["SL:normal", "RSDSDL5:normal"]
TIERS = [("hb_candidate", "Candidate"), ("hb_critical", "Critical-Likely"),
         ("hb_extreme", "Extreme")]

EPS = 1e-10
FLOOR = 1e-300
STARTS = {"frank": [0.5, 1.0, 3.0], "normal": [-0.3, 0.3, 0.6],
          "kimeldorf": [0.2, 0.5, 2.0], "joe": [1.5, 2.0, 4.0]}
BOUNDS = dict(_COPULA_BOUNDS)
BOUNDS["joe"] = (1.001, 20.0)
FNS = dict(_COPULA_FNS)
FNS["joe"] = _joe_cop
NAMES = {"frank": "Frank", "normal": "Gaussian",
         "kimeldorf": "Kimeldorf--Sampson", "joe": "Joe"}


def _margins(df, nm_model, crash_model, nm_col):
    all_feats = list(set(nm_model["features"]) | set(crash_model["features"]))
    df_c = df.dropna(subset=all_feats + [nm_col, "ObsCrashes", OFF]).reset_index(drop=True)
    mu1, alpha1 = _rebuild_mu(df_c, nm_model)
    mu2, alpha2 = _rebuild_mu(df_c, crash_model)
    y1 = df_c[nm_col].values.astype(int)
    y2 = df_c["ObsCrashes"].values.astype(int)
    u1c = np.clip(_nb_cdf(y1, mu1, alpha1), EPS, 1 - EPS)
    u1p = np.where(y1 > 0, np.clip(_nb_cdf(y1 - 1, mu1, alpha1), 0.0, 1 - EPS), 0.0)
    u2c = np.clip(_nb_cdf(y2, mu2, alpha2), EPS, 1 - EPS)
    u2p = np.where(y2 > 0, np.clip(_nb_cdf(y2 - 1, mu2, alpha2), 0.0, 1 - EPS), 0.0)
    indep = _joint_pmf_vec(u1c, u1p, u2c, u2p, lambda u, v, t: u * v, 0.0)
    return dict(u1c=u1c, u1p=u1p, u2c=u2c, u2p=u2p, n=len(df_c),
                ll_indep=float(np.sum(np.log(np.maximum(indep, EPS)))),
                k_indep=nm_model["k"] + crash_model["k"])


def fit_ifm(M, key, k_indep=None):
    u1c, u1p, u2c, u2p = M["u1c"], M["u1p"], M["u2c"], M["u2p"]
    fn = FNS[key]
    lo, hi = BOUNDS[key]
    n = M["n"]
    ki = k_indep if k_indep is not None else M["k_indep"]

    def neg_ll(t):
        pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, fn, float(t[0]))
        return -float(np.sum(np.log(pmf)))

    best = (np.inf, None)
    for s in STARTS[key]:
        r = scipy_min(neg_ll, [s], method="L-BFGS-B", bounds=[(lo, hi)],
                      options={"maxiter": 500})
        if float(r.fun) < best[0]:
            best = (float(r.fun), float(r.x[0]))
    th, ll = best[1], -best[0]
    pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, fn, th)
    nclip = int((pmf <= FLOOR * 1.000001).sum())
    k = ki + 1
    bic = k * np.log(n) - 2 * ll
    bic0 = ki * np.log(n) - 2 * M["ll_indep"]
    lr = 2 * (ll - M["ll_indep"])
    p = float(1 - scipy_stats.chi2.cdf(max(lr, 0.0), df=1))
    return {"theta": th, "ll": ll, "bic": bic, "dBIC": bic - bic0,
            "lr": lr, "lr_p": p, "nclip": nclip, "n": n}


def fit_mixture(M, k_indep=None):
    u1c, u1p, u2c, u2p = M["u1c"], M["u1p"], M["u2c"], M["u2p"]
    n = M["n"]
    ki = k_indep if k_indep is not None else M["k_indep"]

    def neg_mix(t):
        w, rho, thj = float(t[0]), float(t[1]), float(t[2])
        pg = _joint_pmf_vec(u1c, u1p, u2c, u2p, _normal_cop, rho)
        pj = _joint_pmf_vec(u1c, u1p, u2c, u2p, _joe_cop, thj)
        pmf = w * pg + (1 - w) * pj
        return -float(np.sum(np.log(np.maximum(pmf, EPS))))
    r = scipy_min(neg_mix, [0.5, 0.3, 2.0], method="L-BFGS-B",
                  bounds=[(0.01, 0.99), (-0.9, 0.9), (1.001, 20.0)],
                  options={"maxiter": 300})
    w, rho, thj = (float(x) for x in r.x)
    ll = -float(r.fun)
    k = ki + 3
    bic = k * np.log(n) - 2 * ll
    bic0 = ki * np.log(n) - 2 * M["ll_indep"]
    lr = 2 * (ll - M["ll_indep"])
    p = float(1 - scipy_stats.chi2.cdf(max(lr, 0.0), df=3))
    return {"theta": f"w={w:.2f},rho={rho:.2f},thJ={thj:.2f}", "ll": ll,
            "bic": bic, "dBIC": bic - bic0, "lr": lr, "lr_p": p, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--draws", type=int, default=200,
                    help="Halton draws for the RP-margin fits (fallback: 100)")
    ap.add_argument("--skip-rp", action="store_true",
                    help="Skip block A (RP margins); run tier block only")
    ap.add_argument("--skip-tiers", action="store_true",
                    help="Skip block B (tiers); run RP block only")
    ap.add_argument("--resume", action="store_true",
                    help="Load fitted RP margins from margins_rp_ckpt.pkl if present")
    a = ap.parse_args()
    OUT = a.out_dir
    check_mcr_version()

    df = pd.read_csv(a.data, low_memory=False)
    df = add_exposure_offsets(df)
    for c in sorted(set(HB_FEATS) | set(CRASH_FEATS)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df)} rows", flush=True)

    print("Fitting fixed aggregate margins ...", flush=True)
    m_hb = fit_model(df, HB_FEATS, "HB_Total", OFF, label="HBfx")
    m_cr = fit_model(df, CRASH_FEATS, "ObsCrashes", OFF, label="CRfx")
    print("Fixed margins done.", flush=True)

    # ---- B first: tier copulas use only fixed margins (cheap). Written
    # immediately so a later RP-margin failure cannot take them down too. ----
    if a.skip_tiers:
        print("Skipping tier block (--skip-tiers).", flush=True)
        tier_rows = []
    else:
        tier_rows = []
        for col, lab in TIERS:
            print(f"\nTier {lab} ...", flush=True)
            try:
                m_t = fit_model(df, HB_FEATS, col, OFF, label=f"Tier-{lab}")
            except Exception as e:
                print(f"  Tier {lab} margin FAILED: {e}", flush=True)
                continue
            Mt = _margins(df, m_t, m_cr, col)
            for key in ["frank", "normal", "kimeldorf", "joe"]:
                try:
                    r = fit_ifm(Mt, key)
                except Exception as e:
                    print(f"  {NAMES[key]} FAILED: {e}", flush=True)
                    continue
                r["model"] = f"{lab}: {NAMES[key]}"
                r["tier"] = lab
                tier_rows.append(r)
                print(f"  {NAMES[key]}: dBIC={r['dBIC']:+.1f} LR={r['lr']:.1f} p={r['lr_p']:.4g} nclip={r['nclip']}", flush=True)
            try:
                rm = fit_mixture(Mt)
            except Exception as e:
                print(f"  Mixture FAILED: {e}", flush=True)
                continue
            rm["model"] = f"{lab}: Gaussian-Joe mixture"
            rm["tier"] = lab
            tier_rows.append(rm)
            print(f"  Mixture: dBIC={rm['dBIC']:+.1f} LR={rm['lr']:.1f}", flush=True)
    if tier_rows:
        tr = pd.DataFrame(tier_rows)
        tr.to_csv(os.path.join(OUT, "copula_tiers.csv"), index=False)
        with open(os.path.join(OUT, "copula_tiers.tex"), "w") as f:
            f.write("\\begin{table}[!ht]\n")
            f.write("  \\caption{Severity-tier copulas vs crashes (IFM, fixed NB tier margins, $N=10{,}780$)}\\label{tab:copula_tiers}\n")
            f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrr}\n      \\toprule\n")
            f.write("      Tier: model & $\\log L$ & BIC & $\\Delta$BIC vs.\\ indep. & LR & $p$ \\\\\n      \\midrule\n")
            for _, r_ in tr.iterrows():
                f.write(f"      {r_['model']} & {r_['ll']:,.1f} & {r_['bic']:,.1f} & {r_['dBIC']:+,.1f} & {r_['lr']:,.1f} & {r_['lr_p']:.3f} \\\\\n")
            f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
            f.write("  \\emph{Note:} Tier margins use the DE hard-braking specification with tier counts as response; crash margin is the DE crash specification.\n")
            f.write("\\end{table}\n")
        print(f"Wrote copula_tiers (.tex/.csv): {len(tr)} rows", flush=True)
    else:
        print("WARNING: no tier rows -- copula_tiers NOT written", flush=True)

    # ---- A. RP margins x all families (heavy: draws_R sim fits). Guarded so
    # a failure here cannot retract the tier outputs written above. ----
    if not a.skip_rp:
        import pickle
        ckpt = os.path.join(OUT, "margins_rp_ckpt.pkl")
        m_hb_r = m_cr_r = None
        if a.resume and os.path.isfile(ckpt):
            try:
                with open(ckpt, "rb") as pf:
                    saved = pickle.load(pf)
                if saved.get("draws") == a.draws:
                    m_hb_r, m_cr_r = saved["hb"], saved["cr"]
                    print(f"Resumed RP margins from {ckpt} (draws_R={a.draws}).",
                          flush=True)
                else:
                    print(f"Checkpoint draws {saved.get('draws')} != {a.draws}; refitting.",
                          flush=True)
            except Exception as e:
                print(f"Checkpoint load FAILED ({e}); refitting.", flush=True)
                m_hb_r = m_cr_r = None
        if m_hb_r is None:
            print(f"\nFitting RP margins (draws_R={a.draws}) ...", flush=True)
            for draws in ([a.draws] if a.draws <= 100 else [a.draws, 100]):
                try:
                    print(f"  HB random (draws_R={draws}) ...", flush=True)
                    m_hb_r = fit_model(df, HB_FEATS, "HB_Total", OFF, label="HBr",
                                       rdm_terms=HB_RDM, rdm_cor_terms=HB_RDM_COR, draws_R=draws)
                    print(f"  HB random done: LL={m_hb_r['ll']:.1f} k={m_hb_r['k']}.", flush=True)
                    print(f"  Crash random (draws_R={draws}) ...", flush=True)
                    m_cr_r = fit_model(df, CRASH_FEATS, "ObsCrashes", OFF, label="CRr",
                                       rdm_terms=CR_RDM, rdm_cor_terms=CR_RDM_COR, draws_R=draws)
                    print(f"  Crash random done: LL={m_cr_r['ll']:.1f} k={m_cr_r['k']}.", flush=True)
                    print(f"RP margins done (draws_R={draws}).", flush=True)
                    try:
                        with open(ckpt, "wb") as pf:
                            pickle.dump({"hb": m_hb_r, "cr": m_cr_r,
                                         "draws": draws}, pf)
                        print(f"Checkpointed RP margins to {ckpt}.", flush=True)
                    except Exception as e:
                        print(f"Checkpoint save FAILED ({e}); continuing.", flush=True)
                    break
                except Exception as e:
                    print(f"RP margins draws_R={draws} FAILED: {e}", flush=True)
                    m_hb_r = m_cr_r = None
        if m_hb_r is None:
            print("WARNING: RP margins failed at all draws -- skipping RP block", flush=True)
        else:
            Mr = _margins(df, m_hb_r, m_cr_r, "HB_Total")
            rp_rows = []
            for key in ["frank", "normal", "kimeldorf", "joe"]:
                try:
                    r = fit_ifm(Mr, key)
                except Exception as e:
                    print(f"RP {NAMES[key]} FAILED: {e}", flush=True)
                    continue
                r["model"] = NAMES[key] + " (RP margins)"
                rp_rows.append(r)
                print(f"RP {r['model']}: th={r['theta'] if isinstance(r['theta'], str) else round(r['theta'], 3)} "
                      f"dBIC={r['dBIC']:+.1f} LR={r['lr']:.1f} nclip={r['nclip']}", flush=True)
            if rp_rows:
                rp = pd.DataFrame(rp_rows)
                rp.to_csv(os.path.join(OUT, "copula_rp_all.csv"), index=False)
                with open(os.path.join(OUT, "copula_rp_all.tex"), "w") as f:
                    f.write("\\begin{table}[!ht]\n")
                    f.write("  \\caption{Copula dependence with random-parameter NB margins (IFM, $N=10{,}780$)}\\label{tab:copula_rp}\n")
                    f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrr}\n      \\toprule\n")
                    f.write("      Model & $\\log L$ & BIC & $\\Delta$BIC vs.\\ indep. & LR & $p$ \\\\\n      \\midrule\n")
                    for _, r_ in rp.iterrows():
                        f.write(f"      {r_['model']} & {r_['ll']:,.1f} & {r_['bic']:,.1f} & {r_['dBIC']:+,.1f} & {r_['lr']:,.1f} & {r_['lr_p']:.3f} \\\\\n")
                    f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
                    f.write("  \\emph{Note:} Margins are the random-parameter NB fits of \\Cref{tab:rp_check}. Floored evaluations: "
                            + ", ".join(f"{r_['model'].split(' (')[0]}={int(r_['nclip'])}" for _, r_ in rp.iterrows()) + ".\n")
                    f.write("\\end{table}\n")
                print(f"Wrote copula_rp_all (.tex/.csv): {len(rp)} rows", flush=True)
            else:
                print("WARNING: no RP rows -- copula_rp_all NOT written", flush=True)
    else:
        print("Skipping RP block (--skip-rp).", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
