"""
24_holdout_validation.py
Event-based hold-out validation protocol for the Suncorp EB risk model.

This operationalises the "leave-one-fixture-out prospective validation" that
the paper flags as future work. Two protocols are implemented:

  (A) Leave-one-event-out (LOEO): each of the 16 event windows is held out in
      turn; the SPF / EB model is trained on the other 15 and used to predict
      segment risk in the held-out event. Metrics are aggregated across all 16.

  (B) Prospective hold-out: train on the 2025 event year (ref_id 1..13) and
      test on the March-2026 Suncorp fixtures (ref_id 14,15,16) -- a true
      out-of-time, out-of-sample test of the forward-prediction capability.

For every held-out event we report:
    - RMSE and MAE of predicted vs observed segment risk (raw SPF and EB)
    - Top-20 / Top-50 segment rank overlap (Jaccard) between predicted-EB
      ranking and observed-count ranking
    - Spearman rank correlation of the two rankings

Outputs
    results/holdout_validation.csv      per-held-out-event metrics
    results/holdout_validation.tex      protocol description + results table (paper)
    results/holdout_validation.md       human-readable summary

Usage
    python 24_holdout_validation.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import prepare_metacount_frame, fit_metacount_nb, predict_metacount

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
os.makedirs(RES, exist_ok=True)

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
PROSPECTIVE_TEST = [14, 15, 16]
OUT_CSV = os.path.join(RES, "holdout_validation.csv")
OUT_TEX = os.path.join(RES, "holdout_validation.tex")
OUT_TABLE = os.path.join(RES, "holdout_table.tex")
OUT_MD = os.path.join(RES, "holdout_validation.md")


def fit_spf(df):
    """Exposure-only NB via metacountregressor (keeps holdout honest + fast)."""
    mc, _ = prepare_metacount_frame(df, exposure_col="trips",
                                    highway_col="__none__")
    info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                            fixed_terms=["logexp"], rdm_terms=[],
                            model="nb", R=100, de_maxiter=6,
                            de_popsize=6, verbose=False)
    return info, float(info["alpha"])


def eb(y, expected, alpha):
    w = 1.0 / (1.0 + alpha * expected)
    return w * expected + (1.0 - w) * y


def topk_overlap(pred_rank, obs_rank, k):
    p = set(pred_rank.head(k).index); o = set(obs_rank.head(k).index)
    return len(p & o) / len(p | o) if (p | o) else 0.0


def evaluate(train_df, test_df):
    info, alpha = fit_spf(train_df)
    test = test_df.copy()
    mc_te, _ = prepare_metacount_frame(test, exposure_col="trips",
                                       highway_col="__none__")
    test["expected"] = predict_metacount(mc_te, info)
    test["eb"] = eb(test["Y"].values, test["expected"].values, alpha)
    rmse_raw = np.sqrt(((test["Y"] - test["expected"]) ** 2).mean())
    rmse_eb = np.sqrt(((test["Y"] - test["eb"]) ** 2).mean())
    mae_eb = (test["Y"] - test["eb"]).abs().mean()
    pred_rank = test.sort_values("eb", ascending=False)["road_osmid"]
    obs_rank = test.sort_values("Y", ascending=False)["road_osmid"]
    rho = spearmanr(test["eb"], test["Y"]).correlation
    return {
        "rmse_raw": rmse_raw, "rmse_eb": rmse_eb, "mae_eb": mae_eb,
        "spearman": rho,
        "top20": topk_overlap(pred_rank, obs_rank, 20),
        "top50": topk_overlap(pred_rank, obs_rank, 50),
        "n_seg": len(test),
    }


def main():
    print("=" * 80)
    print("24_holdout_validation.py - event-based hold-out validation")
    print("=" * 80)

    print("\n1. Loading segment table...")
    df = pd.read_csv(SEGMENT_ROADS)
    df["Y"] = df["braking"] + df["speeding"]
    ev = df[df["type"] == "Event"].copy()
    event_ids = sorted(ev["ref_id"].unique())
    print(f"   {len(event_ids)} event windows: {event_ids}")

    # ---- (A) Leave-one-event-out --------------------------------------------
    print("\n2. Protocol A: leave-one-event-out across 16 events...")
    loeo = []
    for e in event_ids:
        test = ev[ev["ref_id"] == e]
        train = ev[ev["ref_id"] != e]
        m = evaluate(train, test)
        m["held_out"] = int(e)
        m["protocol"] = "LOEO"
        loeo.append(m)
        print(f"   hold-out {e:>2}: RMSE(EB)={m['rmse_eb']:.3f}  "
              f"top20={m['top20']:.2f}  rho={m['spearman']:.3f}")

    # ---- (B) Prospective hold-out -------------------------------------------
    print("\n3. Protocol B: prospective (train 2025, test 2026)...")
    train = ev[~ev["ref_id"].isin(PROSPECTIVE_TEST)]
    test = ev[ev["ref_id"].isin(PROSPECTIVE_TEST)]
    m = evaluate(train, test)
    m["held_out"] = "14,15,16"; m["protocol"] = "Prospective"
    pros = [m]
    print(f"   test {PROSPECTIVE_TEST}: RMSE(EB)={m['rmse_eb']:.3f}  "
          f"top20={m['top20']:.2f}  rho={m['spearman']:.3f}")

    all_rows = pd.DataFrame(loeo + pros)
    all_rows.to_csv(OUT_CSV, index=False)
    print(f"   Wrote {OUT_CSV}")

    # aggregate LOEO
    agg = all_rows[all_rows.protocol == "LOEO"][
        ["rmse_raw", "rmse_eb", "mae_eb", "spearman", "top20", "top50"]
    ].mean()
    pros_row = all_rows[all_rows.protocol == "Prospective"].iloc[0]

    # ---- LaTeX for the paper ------------------------------------------------
    L = []
    L.append("% holdout_validation.tex - auto-generated by 24_holdout_validation.py")
    L.append("\\subsection{Hold-out validation protocol}\\label{sec:holdout}")
    L.append("To demonstrate that the Empirical-Bayes street-risk scores are not")
    L.append("over-fitted to the events used to build them, two out-of-sample")
    L.append("protocols are applied. Both train the NB count model and EB shrinkage on a set of")
    L.append("events and then predict segment-level risk (raw model and EB-adjusted) on")
    L.append("events the model has not seen.")
    L.append("")
    L.append("\\paragraph{Protocol A -- leave-one-event-out (LOEO).} Each of the 16")
    L.append("event windows is held out in turn; the negative-binomial count model is")
    L.append("calibrated on the remaining 15 events and used to predict the held-out")
    L.append("event's segment counts. Reported metrics are averaged over all 16")
    L.append("folds.")
    L.append("")
    L.append("\\paragraph{Protocol B -- prospective hold-out.} The model is trained on")
    L.append("the 2025 event year (event windows 1--13) and tested on the three")
    L.append("March-2026 Suncorp fixtures (windows 14--16), a true out-of-time,")
    L.append("out-of-sample check of the forward-prediction capability described in")
    L.append("Sec.\\,\\ref{sec:concept}.")
    L.append("")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{Hold-out validation of the EB street-risk model.}")
    L.append("\\label{tab:holdout}")
    L.append("\\begin{tabular}{lrrrrrr}")
    L.append("\\hline")
    L.append("Protocol & RMSE$_{\\text{raw}}$ & RMSE$_{\\text{EB}}$ & MAE$_{\\text{EB}}$ "
             "& $\\rho$ & Top-20 & Top-50 \\\\")
    L.append("\\hline")
    L.append("LOEO (16 folds) & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.2f} & {5:.2f} \\\\".format(
        agg["rmse_raw"], agg["rmse_eb"], agg["mae_eb"], agg["spearman"],
        agg["top20"], agg["top50"]))
    L.append("Prospective (2026) & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.2f} & {5:.2f} \\\\".format(
        pros_row["rmse_raw"], pros_row["rmse_eb"], pros_row["mae_eb"],
        pros_row["spearman"], pros_row["top20"], pros_row["top50"]))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("RMSE and MAE are computed on per-segment risky-event counts; "
             "$\\rho$ is the Spearman rank correlation between EB-predicted and "
             "observed segment risk; Top-$K$ is the Jaccard overlap of the "
             "predicted-EB and observed-count top-$K$ segment sets.")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    print("   Wrote " + OUT_TEX)

    # Table-only fragment for \input from the paper (no duplicate prose)
    T = []
    T.append("\\begin{table}[h]")
    T.append("\\centering")
    T.append("\\caption{Hold-out validation of the EB street-risk model.}")
    T.append("\\label{tab:holdout}")
    T.append("\\begin{tabular}{lrrrrrr}")
    T.append("\\hline")
    T.append("Protocol & RMSE$_{\\text{raw}}$ & RMSE$_{\\text{EB}}$ & "
             "MAE$_{\\text{EB}}$ & $\\rho$ & Top-20 & Top-50 \\\\")
    T.append("\\hline")
    T.append("LOEO (16 folds) & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.2f} & {5:.2f} \\\\".format(
        agg["rmse_raw"], agg["rmse_eb"], agg["mae_eb"], agg["spearman"],
        agg["top20"], agg["top50"]))
    T.append("Prospective (2026) & {0:.3f} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.2f} & {5:.2f} \\\\".format(
        pros_row["rmse_raw"], pros_row["rmse_eb"], pros_row["mae_eb"],
        pros_row["spearman"], pros_row["top20"], pros_row["top50"]))
    T.append("\\hline")
    T.append("\\end{tabular}")
    T.append("\\end{table}")
    with open(OUT_TABLE, "w") as f:
        f.write("\n".join(T) + "\n")
    print("   Wrote " + OUT_TABLE)

    md = (
        "# Hold-out Validation Summary\n\n"
        "- **LOEO (16 folds):** RMSE(EB)={0:.3f}, MAE(EB)={1:.3f}, "
        "Spearman={2:.3f}, Top20={3:.2f}, Top50={4:.2f}\n"
        "- **Prospective (2026 test):** RMSE(EB)={5:.3f}, MAE(EB)={6:.3f}, "
        "Spearman={7:.3f}, Top20={8:.2f}\n"
    ).format(agg["rmse_eb"], agg["mae_eb"], agg["spearman"], agg["top20"],
             agg["top50"], pros_row["rmse_eb"], pros_row["mae_eb"],
             pros_row["spearman"], pros_row["top20"])
    with open(OUT_MD, "w") as f:
        f.write(md)
    print("   Wrote " + OUT_MD)
    print("\nDONE! Hold-out validation complete.")


if __name__ == "__main__":
    main()
