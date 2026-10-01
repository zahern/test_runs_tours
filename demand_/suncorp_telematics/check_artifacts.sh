#!/bin/bash
# Did the killed run (26095208) leave usable predictions + holdout metrics?
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1
module purge 2>/dev/null
module load Anaconda3/2024.02-1 2>/dev/null
conda activate zigenv 2>/dev/null
export JAX_PLATFORMS=cpu

echo "=== shared prediction file (what stage 60 consumes) ==="
ls -la --time-style=+%d_%H:%M results/game_day_pred_15min.parquet 2>/dev/null || echo MISSING
echo
echo "=== arm A / arm B metrics ==="
for d in results_tests/t4_armA_base results_tests/t4_armB_calib; do
  echo "--- $d ---"
  ls -la --time-style=+%d_%H:%M "$d" 2>/dev/null | head -9
done
echo
echo "=== how many games are actually in the predictions? ==="
python - <<'PY' 2>&1 | tail -20
import pandas as pd
df = pd.read_parquet("results/game_day_pred_15min.parquet",
                     columns=["fixture", "site", "rel_bin", "pred_speeding"])
print("rows      :", len(df))
print("fixtures  :", df["fixture"].nunique())
print("sites     :", df["site"].nunique())
print("per fixture rows:")
print(df.groupby("fixture").size().to_string())
print("pred_speeding total:", float(df["pred_speeding"].sum()))
PY
echo
echo "=== holdout metrics: exposure calibration flag + binomial ==="
python - <<'PY' 2>&1 | tail -25
import json
for nm, d in (("A", "results_tests/t4_armA_base"),
              ("B", "results_tests/t4_armB_calib")):
    try:
        j = json.load(open(d + "/game_day_holdout_metrics.json"))
    except Exception as exc:
        print(nm, "unreadable:", exc); continue
    au = j.get("audit", {}) or {}
    bi = au.get("binomial") or {}
    m = j.get("metrics", {}) or {}
    rep = au.get("exposure_coef_report") or {}
    print(nm, "calibration=", au.get("exposure_calibration"),
          "bin_spearman=", m.get("bin_spearman"),
          "chi2=", m.get("chi2"))
    print("   binomial available:", bi.get("available"),
          "n_trials=", bi.get("n_trials"),
          "logexp_coef=", bi.get("logexp_coef"))
    print("   coef report: beta=", rep.get("beta"),
          "verdict=", rep.get("verdict"))
PY
echo
echo "=== did the MIP write anything before it was killed? ==="
ls -la --time-style=+%d_%H:%M results_tests/t4_mip/ 2>/dev/null | head -10 || echo "no MIP output"
echo
echo "=== proof figures from the arms ==="
ls -la --time-style=+%d_%H:%M figures_tests/t4_armA_base/ 2>/dev/null | head -8
ls -la --time-style=+%d_%H:%M figures_tests/t4_armB_calib/ 2>/dev/null | head -8
