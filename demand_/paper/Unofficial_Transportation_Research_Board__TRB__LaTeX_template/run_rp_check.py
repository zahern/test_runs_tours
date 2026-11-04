"""Random-parameter check for the DE benchmark specs (Tables 8/9).

Fits, for each response (HB_Total, ObsCrashes):
  (a) fixed-only NB (replicates Tables 8/9)
  (b) random-parameter NB with MCR-style roles
and writes a fixed-vs-random comparison (LL/AIC/BIC, LR test, SD significance).
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nearmiss_hb_analysis import fit_model

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "near_miss_hb_ecf.csv")
OUT_TEX = os.path.join(HERE, "rp_check.tex")
OUT_CSV = os.path.join(HERE, "rp_check.csv")

HB_FEATS = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
            "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5", "RSDSO", "RSDSDL5"]
CRASH_FEATS = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]

HB_RDM = ["IC:normal", "IV_M:normal"]
HB_RDM_COR = ["PA_COM1:normal", "RSDSO:normal", "RSDSDL5:normal"]
CR_RDM = ["PSDS1T24:normal", "RSDSDG10:normal"]
CR_RDM_COR = ["SL:normal", "RSDSDL5:normal"]


def main():
    df = pd.read_csv(DATA, low_memory=False)
    # Same convention as nearmiss_hb_analysis.load_segments: roadside-severity
    # dummies are NaN where the condition is absent -> fill with 0 so the
    # estimation sample stays at the full N=10,780.
    feat_union = sorted(set(HB_FEATS) | set(CRASH_FEATS))
    for c in feat_union:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df)} rows", flush=True)
    rows = []

    for label, feats, y, rdm, rdm_cor in [
        ("HB fixed", HB_FEATS, "HB_Total", None, None),
        ("HB random", HB_FEATS, "HB_Total", HB_RDM, HB_RDM_COR),
        ("Crash fixed", CRASH_FEATS, "ObsCrashes", None, None),
        ("Crash random", CRASH_FEATS, "ObsCrashes", CR_RDM, CR_RDM_COR),
    ]:
        print(f"\n=== Fitting {label} ===", flush=True)
        m = fit_model(df, feats, y, "log_len", label=label,
                      rdm_terms=rdm, rdm_cor_terms=rdm_cor, draws_R=200)
        print(f"  LL={m['ll']:.1f} AIC={m['aic']:.1f} BIC={m['bic']:.1f} "
              f"k={m['k']} converged={m['converged']}", flush=True)
        ct = m["coef_table"]
        sd_rows = ct[ct["Variable"].str.startswith("RND sd(")]
        n_sd_sig = int(((sd_rows["p"] < 0.05).sum())) if len(sd_rows) else 0
        rows.append({"model": label, "ll": m["ll"], "aic": m["aic"],
                     "bic": m["bic"], "k": m["k"], "n": m["n"],
                     "converged": m["converged"],
                     "n_sd": len(sd_rows), "n_sd_sig05": n_sd_sig})
        m["coef_table"].to_csv(os.path.join(HERE, f"rp_check_{label.replace(' ', '_')}.csv"), index=False)

    res = pd.DataFrame(rows)
    # LR tests: random vs fixed per response
    for resp, fx, rn in [("HB", "HB fixed", "HB random"), ("Crash", "Crash fixed", "Crash random")]:
        a = res.set_index("model")
        lr = 2 * (a.loc[rn, "ll"] - a.loc[fx, "ll"])
        dfree = int(a.loc[rn, "k"] - a.loc[fx, "k"])
        from scipy import stats as st
        p = float(1 - st.chi2.cdf(lr, dfree))
        print(f"{resp}: LR={lr:.1f} df={dfree} p={p:.4g}", flush=True)
        res.loc[res["model"] == rn, "lr_vs_fixed"] = lr
        res.loc[res["model"] == rn, "lr_df"] = dfree
        res.loc[res["model"] == rn, "lr_p"] = p
    res.to_csv(OUT_CSV, index=False)

    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[!ht]\n")
        f.write("  \\caption{Fixed vs.\ random-parameter NB on the DE benchmark specifications ($N=10{,}780$, $R=200$ Halton draws)}\\label{tab:rp_check}\n")
        f.write("  \\begin{center}\n    \\begin{tabular}{lrrrrr}\n      \\toprule\n")
        f.write("      Model & LL & AIC & BIC & $k$ & $\\Delta$BIC vs.\ fixed \\\\\n      \\midrule\n")
        for _, r in res.iterrows():
            base = res[(res["model"].str.split().str[0] == r["model"].split()[0]) & (res["model"].str.endswith("fixed"))]
            d = r["bic"] - float(base["bic"].iloc[0]) if r["model"].endswith("random") else 0.0
            f.write(f"      {r['model']} & {r['ll']:,.1f} & {r['aic']:,.1f} & {r['bic']:,.1f} & {int(r['k'])} & {d:+,.1f} \\\\\n")
        f.write("      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n")
        f.write("  \\emph{Note:} Random blocks follow the MCR role search: HB, random-independent IC/IV\\_M plus random-correlated PA\\_COM1/RSDSO/RSDSDL5; crashes, random-independent PSDS1T24/RSDSDG10 plus random-correlated SL/RSDSDL5. LR tests and SD significance are reported in \\texttt{rp\\_check.csv}.\n")
        f.write("\\end{table}\n")
    print("Wrote rp_check.tex / rp_check.csv", flush=True)


if __name__ == "__main__":
    main()
