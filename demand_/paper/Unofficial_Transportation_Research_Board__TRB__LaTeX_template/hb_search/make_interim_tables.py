#!/usr/bin/env python3
"""
Generate compile-safe INTERIM paper tables on the full-year data.
=================================================================
Fits the paper's established DE feature sets (no search) on
near_miss_hb_ecf_year.csv, then runs the same held-out / transfer / copula /
master-table machinery as the full pipeline. Produces the .tex includes the
restructured paper references, so the manuscript compiles NOW. The HPC job
(run_full_search.pbs) overwrites these with freshly-searched, full-sample
numbers.

  python make_interim_tables.py            # -> writes .tex/.csv into paper dir
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hb_vs_crash_prediction_search as P
import finalize

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Established DE-benchmark feature sets (hb_de_features_*.csv)
HB_FEATS = ["AADT", "SL", "NL", "SKD_SP", "IC", "IP", "PA_COM1",
            "PSDSG24", "PSDS1T24", "IV_VL", "IV_M", "RSPSDL5", "RSDSO", "RSDSDL5"]
CRASH_FEATS = ["AADT", "SL", "IP", "MEDSFBar", "PSDS1T24", "PSDSNIL",
               "IV_VL", "RSDSDG10", "RSDSO", "RSDSDL5"]


def main():
    out_dir = PAPER_DIR
    # point the shared loader at the full-year table
    P.DATA_PATH = os.path.join(PAPER_DIR, "near_miss_hb_ecf_year.csv")
    df, geo = P.load_data()
    hb_feats = [f for f in HB_FEATS if f in geo]
    cr_feats = [f for f in CRASH_FEATS if f in geo]
    print(f"  Interim: HB {len(hb_feats)} feats, Crash {len(cr_feats)} feats "
          f"on {len(df):,} segments (full year)")

    panel_path = os.path.join(PAPER_DIR, "near_miss_hb_panel_year.csv")
    panel = pd.read_csv(panel_path, dtype={"segment_id": str}) \
        if os.path.exists(panel_path) else None

    hb_de = P.fit_nb_model(df, hb_feats, "nm_total", "log_len", label="HB-DE")
    cr_de = P.fit_nb_model(df, cr_feats, "ObsCrashes", "log_len", label="Crash-DE")

    st = P.specification_transfer(df, hb_feats, cr_feats, out_dir)
    encomp = P.encompassing_test(df, hb_feats, cr_feats, out_dir)
    copula = P.bivariate_copula_analysis(df, hb_de, cr_de, out_dir)

    holdout = finalize.run_holdout(df, panel, hb_feats, cr_feats, out_dir, kfolds=5)
    finalize.build_master_table(df, hb_de, cr_de, hb_feats, cr_feats, st, encomp,
                                copula, holdout, None, None, out_dir)
    P.transferability_summary(st, encomp, copula, out_dir)
    print("\n  Interim tables written to paper dir (overwritten by HPC run).")


if __name__ == "__main__":
    main()
