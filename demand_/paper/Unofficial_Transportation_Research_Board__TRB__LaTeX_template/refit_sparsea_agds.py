"""Variable-selection refit with the SparseEA-AGDS-style solver.

Re-runs the paper's Stage-1 (SP1) penalised-BIC search over the 30 candidate
geometry variables for both responses (HB_Total, ObsCrashes) using
solver="sparsea-agds" (see sparsea_agds.py) instead of scipy DE -- same
fitness, same penalty (p-threshold 0.05, 3.0/insig), same seeds (42/43),
same full N=10,780 sample, single run, no restarts.

Outputs (out-dir):
  hb_sparseaagds_best_nm.tex / hb_sparseaagds_best_crash.tex
  hb_sparseaagds_features_{nm,crash}.csv
  hb_sparseaagds_convergence_{nm,crash}.png
  sparseaagds_vs_de.csv   (penalised-BIC comparison vs the DE benchmarks)

Usage (HPC): python -u refit_sparsea_agds.py [--data ...] [--out-dir ...]
             [--maxiter 60] [--popsize 12]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nearmiss_hb_analysis import (
    differential_evolution_feature_search,
    fit_model,
    model_to_latex,
)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Same 30-candidate set as load_data() in nearmiss_hb_analysis.py
GEO_CANDIDATES = [
    "AADT", "SL", "NL", "SKD_SP", "CU_MOD", "CU_SHA", "CU_VSHA",
    "DEL_P", "IC", "IP", "MEDPM", "MEDSFBar", "PA_COM1",
    "PSPS1T24", "PSPSNIL", "PSDSG24", "PSDS1T24", "PSDSNIL",
    "IV_VL", "IV_L", "IV_M", "IV_100",
    "RSPSDL5", "RSDSDL1", "RSDSDG10", "RSDSO",
    "RSPSDL1", "RSDSDL5", "RSPSDL10", "RSDSDL10", "RSPSDG10",
]


def load_like_main(data_path):
    """Mirror load_data(): fixed GEO list, NaN->0, log_len offset."""
    df = pd.read_csv(data_path, low_memory=False)
    GEO = [c for c in GEO_CANDIDATES
           if c in df.columns and df[c].std() > 1e-10]
    df = df.dropna(subset=["ObsCrashes"])
    df = df[df["ObsCrashes"] >= 0]
    for c in GEO:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)
    for c in ("nm_total", "HB_Total"):
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(0)
    df["seg_len_km"] = (df.get("segment_length_m", 1000) / 1000).clip(lower=0.001)
    df["log_len"] = np.log(df["seg_len_km"])
    return df, GEO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "near_miss_hb_ecf.csv"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=12)
    args = ap.parse_args()

    df, GEO = load_like_main(args.data)
    nm_col = "nm_total" if "nm_total" in df.columns else "HB_Total"
    print(f"Segments: {len(df)} | Candidates ({len(GEO)}): {GEO} | HB col: {nm_col}",
          flush=True)

    rows = []
    for y_col, tag, seed, mi, ps in [
        (nm_col, "nm", 42, args.maxiter, args.popsize),
        ("ObsCrashes", "crash", 43, max(args.maxiter // 2, 30), max(args.popsize - 4, 8)),
    ]:
        model, feats, hist = differential_evolution_feature_search(
            df, GEO, y_col, "log_len",
            label=f"SparseAGDS_{tag}", p_threshold=0.05,
            penalty_per_insig=3.0, maxiter=mi, popsize=ps, seed=seed,
            solver="sparsea-agds",
            plot_path=os.path.join(args.out_dir,
                                   f"hb_sparseaagds_convergence_{tag}.png"),
        )
        pd.DataFrame({"feature": feats, "response": tag,
                      "step": "sparsea-agds"}).to_csv(
            os.path.join(args.out_dir,
                         f"hb_sparseaagds_features_{tag}.csv"), index=False)
        model_to_latex(
            model, os.path.join(args.out_dir,
                                f"hb_sparseaagds_best_{tag}.tex"),
            f"SparseEA-AGDS Best Model: {tag}", f"tab:sparseaagds_{tag}")
        # head-to-head vs the DE benchmark refit on the same feature set
        rows.append({"response": tag, "n_feat": len(feats),
                     "ll": model["ll"], "bic": model["bic"],
                     "k": model["k"], "n": model["n"],
                     "penalised_bic": float(min(hist)) if len(hist) else float("nan"),
                     "features": ";".join(feats)})
        print(f"[{tag}] done: {len(feats)} feats, BIC={model['bic']:.1f}",
              flush=True)

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "sparseaagds_vs_de.csv"), index=False)
    print("Wrote sparseaagds_vs_de.csv", flush=True)


if __name__ == "__main__":
    main()
