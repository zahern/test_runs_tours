#!/bin/bash
# Two things to explain before resubmitting:
#  (1) results/game_day_pred_15min.parquet is dated 29 Aug, yet both arms
#      exited 0 and wrote their own per-arm copy -- so does stage 59 update
#      the shared file? If not, the MIP has been reading a STALE surface and
#      my "run the arms first" reorder achieved nothing.
#  (2) figures_tests/t4_armA_base has calibration.png from the killed run
#      but surfaces.png/coefs.png from 10:12, and NO game_day_pred_vs_actual.png
#      anywhere. Why didn't the proof figure get written?
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1
L=t4mip.o26095208

echo "=== (1) does stage 59 write the SHARED results/ file? ==="
grep -nE 'results/game_day_pred|game_day_pred_15min|out_dir /|OUT_PRED|pred_15min' \
    scripts/59_game_day_search_predict.py | head -20
echo
echo "=== (2) figure writes in stage 59 ==="
grep -nE 'savefig|plot_pred_vs_actual|pred_vs_actual|surfaces' \
    scripts/59_game_day_search_predict.py | head -25
echo
echo "=== (3) what the killed run's log says about figures ==="
grep -nE 'saved|figure|png|surfaces|pred_vs_actual|plot' "$L" | tail -30
echo
echo "=== (4) all pngs the run mentions, with their target paths ==="
grep -oE '[A-Za-z0-9_/.-]*\.png' "$L" | sort -u | head -20
echo
echo "=== (5) arm B per-arm parquet: does it have 6 fixtures? ==="
module purge 2>/dev/null
module load Anaconda3/2024.02-1 2>/dev/null
conda activate zigenv 2>/dev/null
export JAX_PLATFORMS=cpu
python - <<'PY' 2>&1 | tail -14
import pandas as pd
for tag, p in (("shared", "results/game_day_pred_15min.parquet"),
               ("armB", "results_tests/t4_armB_calib/game_day_pred_15min.parquet")):
    df = pd.read_parquet(p, columns=["fixture", "rel_bin", "pred_speeding"])
    print("%-7s fixtures=%d rows=%d total_pred=%.0f"
          % (tag, df["fixture"].nunique(), len(df), df["pred_speeding"].sum()))
    print("        ", sorted(df["fixture"].dropna().unique().tolist())[:8])
PY
