"""Significance-constrained refits for the reported (fixed-effect) models.

Produces fully-significant (p < 0.10, i.e. |z| > 1.645) replacements for every
model flagged as non-significant in the paper audit:

  1. Final condensed HB  (11 searched vars -> backward elimination -> tex)
  2. Final condensed Crash (8 searched vars -> backward elimination -> tex)
  3. Extreme-tier HB category (10 vars, N=384 events -> prune to 3-4 vars)
  4. Critical-tier HB category (drop borderline Intersection Present if p>0.05)
  5. Significance-constrained random-parameter re-search (single-start GA over
     roles; any candidate with a retained SD p>0.10 or |mean|>5 rejected)

All estimation uses metacountregressor.fit_model (same estimator as the paper's
DE benchmarks), full N=10,780 sample, VMT-years exposure offset
(log_len + log AADT + log obs_years; see exposure.py).
Outputs overwrite the supplement tables and write NEW fully-significant
fixed-effect tables hb_final_spec_fixed_{nm,crash}.tex plus
hb_category_hb_extreme_refit.tex.

Usage (HPC): python -u refit_significant.py [--data near_miss_hb_ecf.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nearmiss_hb_analysis import fit_model, VLABEL  # same estimator as DE benchmarks
from exposure import OFF, add_exposure_offsets

HERE = os.path.dirname(os.path.abspath(__file__))

MCR_MIN_VERSION = (1, 0, 151)


def _V(c):
    return VLABEL.get(c, c)


def check_mcr_version():
    """Fail fast on stale metacountregressor (cf. copula_rp_tiers.py gate).

    Job 25256288 died downstream with an empty-steps IndexError because the
    node still had 1.0.143; the PBS wrapper now asserts >=1.0.151, but the
    script must also refuse to run stale so local runs fail loudly too.
    """
    try:
        import metacountregressor as mcr
        # Prefer distribution metadata over the in-code string: the 1.0.151
        # wheel ships a stale __version__ ("1.0.143") while its dist-info
        # correctly says 1.0.151 (proven by clean force-reinstall).
        try:
            from importlib.metadata import version as _pkg_version
            ver = _pkg_version("metacountregressor")
        except Exception:
            ver = getattr(mcr, "__version__", "0.0.0")
    except Exception:
        print("FATAL: metacountregressor not importable -- refusing to run",
              flush=True)
        sys.exit(2)
    try:
        tup = tuple(int(x) for x in ver.split(".")[:3])
    except ValueError:
        tup = (0, 0, 0)
    print(f"metacountregressor {ver} (required >= 1.0.151)", flush=True)
    if tup < MCR_MIN_VERSION:
        print(f"FATAL: metacountregressor {ver} < 1.0.151 -- refusing to run",
              flush=True)
        sys.exit(2)
    return ver

# Searched variable sets (union of fixed + random roles from Tab:mcr;
# counts match pruned_specs: HB 11 vars, Crash 8 vars).
HB_SEARCHED = ["AADT", "SL", "SKD_SP", "IP", "PSDSG24", "IV_VL",
               "IC", "IV_M", "PA_COM1", "RSDSO", "RSDSDL5"]
CRASH_SEARCHED = ["AADT", "IP", "MEDSFBar", "PSDSNIL",
                  "PSDS1T24", "RSDSDG10", "SL", "RSDSDL5"]
# Category-tier starting set (same 10 covariates as the Candidate margin).
CAT_START = ["AADT", "IC", "IP", "IV_M", "PA_COM1",
             "PSDSG24", "RSDSDL5", "RSDSO", "SKD_SP", "SL"]

ALPHA = 0.10  # significance gate (matches penalised-BIC penalty rule)


def _coef_p(m, var):
    # coef_table stores human-readable labels (fit_model applies VLABEL),
    # so accept either the raw code or its verbose form.
    ct = m["coef_table"]
    for cand in (var, _V(var)):
        row = ct[ct["Variable"] == cand]
        if len(row):
            return float(row["p"].iloc[0])
    return 1.0


def backward_eliminate(df, feats, y_col, label, alpha=ALPHA):
    """Fit NB, drop worst p>alpha covariate, repeat until fully significant.

    NOTE (job 25256288 root cause): fit_model labels coef_table rows with
    human-readable VLABEL names (e.g. "Annual Avg Daily Traffic (AADT)"),
    not raw codes ("AADT"). Filtering with raw codes yields an empty
    candidate set and steps==[] -> IndexError at steps[-1]. Always map
    through _V() for table lookups and map the dropped verbose label back
    to its raw code for the next fit.
    """
    feats = list(feats)
    # Fail loudly on absent columns: main() only coerces columns that EXIST,
    # so a renamed/missing covariate would silently fit without it and die
    # downstream (job 25271020: empty steps -> IndexError at steps[-1]).
    _missing = [c for c in feats if c not in df.columns]
    if _missing:
        raise RuntimeError(f"[{label}] feats absent from data columns: {_missing}")
    steps = []
    while True:
        m = fit_model(df, feats, y_col, OFF, label=label)
        ct = m["coef_table"]
        # Only gate real covariates (skip intercept / dispersion / random rows).
        verbose = [_V(f) for f in feats]
        inv = {_V(f): f for f in feats}
        cand = ct[ct["Variable"].isin(verbose)]
        if len(cand) == 0:
            break
        worst = cand.loc[cand["p"].idxmax()]
        if float(worst["p"]) <= alpha:
            steps.append((list(feats), m))
            break
        steps.append((list(feats), m))
        worst_raw = inv.get(worst["Variable"], worst["Variable"])
        feats = [f for f in feats if f != worst_raw]
        print(f"[{label}] drop {worst['Variable']} ({worst_raw}) "
              f"p={worst['p']:.4f} -> {len(feats)} vars", flush=True)
        if len(feats) < 2:
            m = fit_model(df, feats, y_col, OFF, label=label)
            steps.append((list(feats), m))
            break
    if not steps:
        try:
            from importlib.metadata import version as _pkg_version
            _mcr = _pkg_version("metacountregressor")
        except Exception:
            _mcr = "unknown"
        raise RuntimeError(
            f"[{label}] first fit returned none of the searched covariates "
            f"(coef vars={list(ct['Variable'])}, feats={feats}, "
            f"metacountregressor={_mcr}) — check estimator version/data columns")
    return feats, steps[-1][1], steps


def write_coef_tex(m, feats, path, caption, label):
    ct = m["coef_table"]
    allowed = set([_V(f) for f in feats] + ["(Intercept)", "ln(overdispersion)"])
    with open(path, "w") as f:
        f.write("% Auto-generated by refit_significant.py (fully-significant, "
                f"p<{ALPHA})\n")
        f.write(f"% Model: {label}: LL={m['ll']:.2f}, BIC={m['bic']:.2f}, "
                f"k={m['k']}, n={m['n']}\n")
        f.write("\\begin{table}[!ht]\n")
        f.write(f"  \\caption{{{caption}}}\\label{{{label}}}\n")
        f.write("  \\begin{center}\n")
        f.write("      \\begin{tabular}{lrrrrc}\n")
        f.write("        \\toprule\n")
        f.write("        Variable & Estimate & IRR & z & p & Sig \\\\\n")
        f.write("        \\midrule\n")
        for _, r in ct.iterrows():
            v = r["Variable"]
            if v.startswith(("RND", "chol", "ln(")) or v not in allowed:
                if v not in ("(Intercept)", "ln(overdispersion)"):
                    continue
            sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05
                  else ("*" if r["p"] < 0.10 else ""))
            irr = f"{np.exp(r['Estimate']):.4f}" if "RND" not in v and "chol" not in v and "ln(" not in v else "--"
            f.write(f"    {v} & {r['Estimate']:+.4f} & {irr} & "
                    f"{r['z']:+.2f} & {r['p']:.4f} & {sig} \\\\\n")
        f.write("        \\bottomrule\n")
        f.write("      \\end{tabular}\n")
        f.write("  \\end{center}\n")
        f.write(f"  \\footnotesize{{Note: $N$ = {m['n']:,} segments. Negative "
                "Binomial (NB) regression. All predictors standardized. "
                f"Backward elimination to $p<{ALPHA}$; zero insignificant "
                "coefficients retained.}\n")
        f.write("\\end{table}\n")
    print(f"Wrote {path} (k={m['k']}, BIC={m['bic']:.1f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "near_miss_hb_ecf.csv"))
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    check_mcr_version()

    df = pd.read_csv(args.data, low_memory=False)
    df = add_exposure_offsets(df)
    for c in sorted(set(HB_SEARCHED) | set(CRASH_SEARCHED) | set(CAT_START)):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df)} rows", flush=True)

    # 1-2. Final condensed fixed-effect refits (must need zero/few drops).
    for feats, y, tag, cap, lab in [
        (HB_SEARCHED, "HB_Total", "HB-final-fixed",
         "Final Condensed Model (fixed-effect, fully significant): Hard-Braking Response",
         "tab:hb_final_fixed_nm"),
        (CRASH_SEARCHED, "ObsCrashes", "Crash-final-fixed",
         "Final Condensed Model (fixed-effect, fully significant): Crash Response",
         "tab:hb_final_fixed_crash"),
    ]:
        kept, m, _ = backward_eliminate(df, feats, y, tag)
        assert all(_coef_p(m, v) <= ALPHA for v in kept), f"{tag} still has p>0.10"
        assert float(m["coef_table"].set_index("Variable").loc[
            "ln(overdispersion)", "p"]) < ALPHA, f"{tag} dispersion n.s."
        write_coef_tex(m, kept,
                       os.path.join(args.out_dir,
                                    f"hb_final_spec_fixed_{'nm' if 'HB' in tag else 'crash'}.tex"),
                       cap, lab)

    # 3. Extreme tier (expect heavy pruning: 10 -> ~3-4 vars).
    kept, m, _ = backward_eliminate(df, CAT_START, "hb_extreme", "Extreme-refit")
    write_coef_tex(m, kept,
                   os.path.join(args.out_dir, "hb_category_hb_extreme_refit.tex"),
                   "HB Category Model (significance-pruned, $p<0.10$): Extreme Tier",
                   "tab:cat_hb_extreme_refit")

    # 4. Critical tier (strict 0.05 gate for the borderline IP term).
    kept, m, _ = backward_eliminate(df, CAT_START, "hb_critical",
                                    "Critical-refit", alpha=0.05)
    write_coef_tex(m, kept,
                   os.path.join(args.out_dir, "hb_category_hb_critical_refit.tex"),
                   "HB Category Model (significance-pruned, $p<0.05$): Critical-Likely Tier",
                   "tab:cat_hb_critical_refit")

    # 5. Significance-constrained random-parameter check on DE-benchmark specs:
    #    accept a random block only if EVERY retained SD has p<0.10 and every
    #    |mean|<5 (degenerate-mean guard that rejects the retired tables).
    from run_rp_check import (HB_FEATS, CRASH_FEATS, HB_RDM, HB_RDM_COR,
                              CR_RDM, CR_RDM_COR)
    rows = []
    for label, feats, y, rdm, rdm_cor in [
        ("HB-RP-constrained", HB_FEATS, "HB_Total", HB_RDM, HB_RDM_COR),
        ("Crash-RP-constrained", CRASH_FEATS, "ObsCrashes", CR_RDM, CR_RDM_COR),
    ]:
        m = fit_model(df, feats, y, OFF, label=label,
                      rdm_terms=rdm, rdm_cor_terms=rdm_cor, draws_R=200)
        ct = m["coef_table"]
        sd = ct[ct["Variable"].str.startswith("RND sd(")]
        means = ct[ct["Variable"].str.startswith("RND mean(")]
        ok_sd = bool((sd["p"] < ALPHA).all()) if len(sd) else True
        ok_mean = bool((means["Estimate"].abs() < 5).all()) if len(means) else True
        disp_p = float(ct.set_index("Variable").loc["ln(overdispersion)", "p"])
        verdict = "ACCEPT" if (ok_sd and ok_mean and disp_p < ALPHA) else "REJECT"
        print(f"[{label}] SD p<0.10: {ok_sd}, |mean|<5: {ok_mean}, "
              f"disp p={disp_p:.4f} -> {verdict}", flush=True)
        rows.append({"model": label, "ll": m["ll"], "bic": m["bic"],
                     "k": m["k"], "verdict": verdict})
        m["coef_table"].to_csv(os.path.join(
            args.out_dir, f"rp_constrained_{label.split('-')[0]}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir,
                                           "rp_constrained_verdict.csv"), index=False)
    print("All refits complete. Fully-significant tables written.", flush=True)


if __name__ == "__main__":
    main()
