"""Bayesian compilation of the best HB / crash specs (package-native).

Takes the SEARCH winners (best_models_master.csv: HB 14 / crash 10) and
compiles each into a PyMC model via
metacountregressor.bayesian_model.build_bayesian_model, on the VMT-years
exposure offset (exposure.py). This is the package's designed flow: the
search chooses structure, Bayes quantifies uncertainty -- MLE coefficients
are NOT treated as fixed values.

Outputs (out-dir): bayes_hb.csv/.tex (tab:bayes_hb),
bayes_crash.csv/.tex (tab:bayes_crash): posterior mean/SD/94% HDI per
coefficient plus dispersion.

Needs the bayesian extra: pip install "metacountregressor[bayesian]==1.0.151"
(run_bayes_hb_crash.pbs does this).

Usage (HPC): python -u bayes_hb_crash.py [--draws 500 --tune 500 --chains 2]
"""
import os, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exposure import OFF, add_exposure_offsets

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "near_miss_hb_ecf.csv")
SPEC_CSV = os.path.join(HERE, "best_models_master.csv")

MCR_MIN_VERSION = (1, 0, 151)

HB_FALLBACK = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
               "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5",
               "RSDSO", "RSDSDL5"]
CR_FALLBACK = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]


def check_mcr_version():
    try:
        from importlib.metadata import version as _pkg_version
        ver = _pkg_version("metacountregressor")
    except Exception:
        import metacountregressor as mcr
        ver = getattr(mcr, "__version__", "0.0.0")
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


def load_best_specs(path):
    try:
        spec = pd.read_csv(path)
        hb = spec.loc[spec["response"] == "HB", "features"].iloc[0].split(";")
        cr = spec.loc[spec["response"] == "CRASH", "features"].iloc[0].split(";")
        print(f"Best specs from {os.path.basename(path)}: "
              f"HB({len(hb)}) crash({len(cr)})", flush=True)
        return [h.strip() for h in hb], [c.strip() for c in cr]
    except Exception as e:
        print(f"WARNING: spec CSV unreadable ({e}); using DE fallback.",
              flush=True)
        return list(HB_FALLBACK), list(CR_FALLBACK)


def run_one(df, feats, y_col, tag, label, a):
    from metacountregressor.bayesian_model import build_bayesian_model
    # Coerce FIRST (roadside-severity dummies are NaN where absent), then
    # drop only on the response: job 25344642 dropped to n=169 by dropna-ing
    # raw features before coercion.
    df = df.copy()
    for c in feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[OFF] = pd.to_numeric(df[OFF], errors="coerce")
    dsub = df.dropna(subset=[y_col, OFF]).reset_index(drop=True)
    if len(dsub) < 0.9 * len(df):
        print(f"FATAL [{tag}]: estimation sample n={len(dsub)} << N={len(df)} "
              f"-- refusing to sample a degenerate subset", flush=True)
        sys.exit(2)
    print(f"[{tag}] estimation sample n={len(dsub)} (of N={len(df)})",
          flush=True)
    # Standardize like fit_model: raw scales (AADT in thousands vs 0/1
    # dummies) give NUTS a funnel (job 25363155: hundreds of divergences,
    # treedepth + Rhat failures). Same standardized scale as the MLE tables.
    from sklearn.preprocessing import StandardScaler
    dsub[feats] = StandardScaler().fit_transform(
        dsub[feats].values.astype(float))
    print(f"[{tag}] standardized {len(feats)} covariates.", flush=True)
    print(f"[{tag}] n={len(dsub)} sampling "
          f"(tune={a.tune} draws={a.draws} chains={a.chains}) ...", flush=True)
    bm = build_bayesian_model(
        {"model_spec": {"family": "count", "fixed_terms": list(feats)}},
        df=dsub, y_col=y_col, offset_col=OFF)
    idata = bm.sample(draws=a.draws, tune=a.tune, chains=a.chains,
                      target_accept=a.target_accept,
                      max_treedepth=a.max_treedepth, progressbar=False)
    import arviz as az
    # Keep coefficient-scale terms only: the compiled model also carries a
    # per-observation node (~N dims; job 25363155 papered 10,791 "terms").
    post = idata.posterior
    coef_vars, dropped = [], []
    cap = max(100, 5 * (len(feats) + 2))
    for v in post.data_vars:
        size = int(post[v].size / (a.chains * a.draws)) if a.chains * a.draws else 0
        (coef_vars if size <= cap else dropped).append((v, size))
    if dropped:
        print(f"[{tag}] dropping per-observation nodes from table: "
              + ", ".join(f"{v}(dim={s})" for v, s in dropped), flush=True)
    summ = az.summary(idata, var_names=[v for v, _ in coef_vars],
                      hdi_prob=0.94)
    summ = summ.reset_index().rename(columns={"index": "Variable"})
    keep = [c for c in ["Variable", "mean", "sd", "hdi_3%", "hdi_97%",
                        "ess_bulk", "r_hat"] if c in summ.columns]
    summ = summ[keep]
    summ.to_csv(os.path.join(a.out_dir, f"bayes_{tag}.csv"), index=False)
    with open(os.path.join(a.out_dir, f"bayes_{tag}.tex"), "w") as f:
        f.write(f"% Auto-generated by bayes_hb_crash.py ({tag}: "
                f"tune={a.tune} draws={a.draws} chains={a.chains})\n")
        f.write("\\begin{table}[!ht]\n")
        f.write(f"  \\caption{{Bayesian posterior for the best {label} spec ($N={len(dsub):,}$)}}\\label{{tab:bayes_{tag}}}\n")
        f.write("  \\begin{center}\n    \\begin{tabular}{lrrrr}\n      \\toprule\n")
        f.write("      Term & Mean & SD & HDI 3\\% & HDI 97\\% \\\\\n      \\midrule\n")
        for _, r_ in summ.iterrows():
            f.write(f"      {r_['Variable']} & {r_['mean']:.4f} & {r_['sd']:.4f} & {r_['hdi_3%']:.4f} & {r_['hdi_97%']:.4f} \\\\\n")
        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
        f.write("  \\emph{Note:} Compiled from the searched best spec via package-native Bayesian compilation (VMT-years offset); MLE values not fixed.\n")
        f.write("\\end{table}\n")
    print(f"[{tag}] wrote bayes_{tag} (.tex/.csv): {len(summ)} terms.",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--spec-csv", default=SPEC_CSV)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--max-treedepth", type=int, default=10)
    ap.add_argument("--only", choices=["hb", "crash", "both"], default="both")
    a = ap.parse_args()
    check_mcr_version()

    try:
        import pymc  # noqa: F401
        import arviz  # noqa: F401
    except ImportError:
        print('FATAL: Bayesian extra missing -- run: pip install '
              '"metacountregressor[bayesian]==1.0.151"', flush=True)
        sys.exit(2)

    hb_feats, cr_feats = load_best_specs(a.spec_csv)
    df = pd.read_csv(a.data, low_memory=False)
    df = add_exposure_offsets(df)
    print(f"Loaded {len(df)} rows; offset={OFF}", flush=True)

    if a.only in ("hb", "both"):
        run_one(df, hb_feats, "HB_Total", "hb", "hard-braking", a)
    if a.only in ("crash", "both"):
        run_one(df, cr_feats, "ObsCrashes", "crash", "crash", a)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
