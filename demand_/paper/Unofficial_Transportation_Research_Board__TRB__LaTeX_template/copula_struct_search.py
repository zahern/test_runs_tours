"""Extensive joint structure search scored on Joe-copula joint BIC (SP4-style).

Greedy best-improvement toggle search over (HB mask, crash mask) from the
30 geometry candidates, warm-started at the DE benchmark masks. Each
candidate is scored by refitting both fixed NB margins and optimizing the
Joe dependence (IFM); score = joint BIC with k = k1 + k2 + 1.
Min 2 variables per response; checkpointed; capped evaluations.
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_min
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nearmiss_hb_analysis import fit_model, _nb_cdf, _joint_pmf_vec, _rebuild_mu
from copula_extensions import _joe_cop
from exposure import OFF, add_exposure_offsets

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "near_miss_hb_ecf.csv")

CANDS = ["AADT", "SL", "NL", "SKD_SP", "CU_MOD", "CU_SHA", "CU_VSHA",
         "DEL_P", "IC", "IP", "MEDPM", "MEDSFBar", "PA_COM1",
         "PSPS1T24", "PSPSNIL", "PSDSG24", "PSDS1T24", "PSDSNIL",
         "IV_VL", "IV_L", "IV_M", "IV_100",
         "RSPSDL5", "RSDSDL1", "RSDSDG10", "RSDSO",
         "RSPSDL1", "RSDSDL5", "RSPSDL10", "RSDSDL10", "RSPSDG10"]
# note: CU_ST (reference) and IV_100 (near-zero variance) excluded -> J=30
CANDS = [c for c in CANDS if c not in ("IV_100",)]
assert len(CANDS) == 30, len(CANDS)

HB0 = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
       "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5", "RSDSO", "RSDSDL5"]
CR0 = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
       "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]

EPS = 1e-10
MAX_EVALS = 100
CKPT = os.path.join(HERE, "copula_struct_ckpt.csv")
ALPHA = 0.10  # significance gate (paper rule)
PENALTY = 10.0  # per insignificant margin coefficient

_cache = {}


def _margin_insig(m):
    """Count insignificant fixed coefficients (ex-intercept/dispersion)."""
    try:
        ct = m["coef_table"]
        main = ct[~ct["Variable"].str.startswith(("RND", "chol(", "param", "ln("))
                   & (ct["Variable"] != "(Intercept)")]
        return int(((main["p"] > ALPHA) | main["p"].isna()).sum())
    except Exception:
        return 0


def joint_bic(df, mh, mc):
    """Refit both margins on masks, Joe-IFM, return joint BIC + details."""
    key = (tuple(sorted(mh)), tuple(sorted(mc)))
    if key in _cache:
        return _cache[key]
    m1 = fit_model(df, list(mh), "HB_Total", OFF, label="J-HB")
    m2 = fit_model(df, list(mc), "ObsCrashes", OFF, label="J-CR")
    allf = list(set(mh) | set(mc))
    df_c = df.dropna(subset=allf + ["HB_Total", "ObsCrashes", OFF]).reset_index(drop=True)
    mu1, a1 = _rebuild_mu(df_c, m1)
    mu2, a2 = _rebuild_mu(df_c, m2)
    y1 = df_c["HB_Total"].values.astype(int)
    y2 = df_c["ObsCrashes"].values.astype(int)
    u1c = np.clip(_nb_cdf(y1, mu1, a1), EPS, 1 - EPS)
    u1p = np.where(y1 > 0, np.clip(_nb_cdf(y1 - 1, mu1, a1), 0.0, 1 - EPS), 0.0)
    u2c = np.clip(_nb_cdf(y2, mu2, a2), EPS, 1 - EPS)
    u2p = np.where(y2 > 0, np.clip(_nb_cdf(y2 - 1, mu2, a2), 0.0, 1 - EPS), 0.0)

    def neg_ll(t):
        pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, _joe_cop, float(t[0]))
        return -float(np.sum(np.log(pmf)))
    r = scipy_min(neg_ll, [2.0], method="L-BFGS-B", bounds=[(1.001, 20.0)],
                  options={"maxiter": 300})
    th, ll = float(r.x[0]), -float(r.fun)
    n = len(df_c)
    k = m1["k"] + m2["k"] + 1
    bic = k * np.log(n) - 2 * ll
    n_insig = _margin_insig(m1) + _margin_insig(m2)
    pbic = bic + PENALTY * n_insig
    out = {"bic": bic, "pbic": pbic, "n_insig": n_insig,
           "ll": ll, "theta": th, "k": k, "n": n,
           "ll1": m1["ll"], "ll2": m2["ll"]}
    _cache[key] = out
    return out


def main():
    df = pd.read_csv(DATA, low_memory=False)
    df = add_exposure_offsets(df)
    for c in CANDS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df)} rows", flush=True)

    mh, mc = list(HB0), list(CR0)
    cur = joint_bic(df, mh, mc)
    print(f"START HB({len(mh)}) CR({len(mc)}) jointBIC={cur['bic']:.1f} pBIC={cur['pbic']:.1f} theta={cur['theta']:.3f}", flush=True)
    hist = [{"eval": 0, "bic": cur["bic"], "pbic": cur["pbic"], "theta": cur["theta"],
             "n_hb": len(mh), "n_cr": len(mc)}]
    neval = 0
    for p in range(3):
        improved = False
        cands = []
        for v in CANDS:
            if v in mh and len(mh) > 2:
                cands.append((0, v, [x for x in mh if x != v], mc))
            elif v not in mh:
                cands.append((0, v, mh + [v], mc))
            if v in mc and len(mc) > 2:
                cands.append((1, v, mh, [x for x in mc if x != v]))
            elif v not in mc:
                cands.append((1, v, mh, mc + [v]))
        scored = []
        for side, v, th, tc in cands:
            if neval >= MAX_EVALS:
                break
            try:
                s = joint_bic(df, th, tc)
            except Exception as e:
                print(f"  eval failed {v}: {e}", flush=True)
                continue
            neval += 1
            scored.append((s["pbic"], side, v, th, tc, s))
            if neval % 10 == 0:
                print(f"  pass {p}: {neval} evals, best-so-far {min(s['pbic'] for s in [cur] + [x[5] for x in scored]):.1f}", flush=True)
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < cur["pbic"] - 1e-6:
            _, side, v, mh, mc, cur = scored[0]
            improved = True
            print(f"PASS {p}: flip {v} -> jointBIC={cur['bic']:.1f} pBIC={cur['pbic']:.1f} theta={cur['theta']:.3f} "
                  f"HB({len(mh)}) CR({len(mc)})", flush=True)
        hist.append({"eval": neval, "bic": cur["bic"], "pbic": cur["pbic"], "theta": cur["theta"],
                     "n_hb": len(mh), "n_cr": len(mc),
                     "mh": "+".join(sorted(mh)), "mc": "+".join(sorted(mc))})
        pd.DataFrame(hist).to_csv(CKPT, index=False)
        if not improved or neval >= MAX_EVALS:
            print(f"STOP pass {p} improved={improved} neval={neval}", flush=True)
            break

    print(f"\nFINAL jointBIC={cur['bic']:.1f} pBIC={cur['pbic']:.1f} theta={cur['theta']:.3f}", flush=True)
    print(f"HB({len(mh)}): {sorted(mh)}", flush=True)
    print(f"CR({len(mc)}): {sorted(mc)}", flush=True)
    with open(os.path.join(HERE, "copula_struct.tex"), "w") as f:
        f.write("\\begin{table}[!ht]\n")
        f.write("  \\caption{Joint structure search scored on penalised Joe-copula BIC (warm-started at DE masks)}\\label{tab:copula_struct}\n")
        f.write("  \\begin{center}\n    \\begin{tabular}{lrrr}\n      \\toprule\n")
        f.write("      Specification & Joint BIC & pBIC & $\\hat\\theta_{Joe}$ \\\\\n      \\midrule\n")
        start = hist[0]
        f.write(f"      DE masks (start) & {start['bic']:,.1f} & {start['pbic']:,.1f} & {start['theta']:.2f} \\\\\n")
        f.write(f"      Searched masks & {cur['bic']:,.1f} & {cur['pbic']:,.1f} & {cur['theta']:.2f} \\\\\n")
        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
        f.write(f"  \\emph{{Note:}} Greedy best-improvement toggles over $J=30$ candidates per response ({neval} joint refits) on penalised joint BIC (BIC$+10\\times$#insig across both margins). "
                f"Searched HB ({len(mh)}): {', '.join(sorted(mh))}. Searched crash ({len(mc)}): {', '.join(sorted(mc))}.\n")
        f.write("\\end{table}\n")
    pd.DataFrame(hist).to_csv(os.path.join(HERE, "copula_struct_hist.csv"), index=False)
    print("Wrote copula_struct.tex", flush=True)


if __name__ == "__main__":
    main()
