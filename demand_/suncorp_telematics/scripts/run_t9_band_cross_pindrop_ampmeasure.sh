#!/bin/bash
# run_t9_band_cross_pindrop_ampmeasure.sh - in-sample calibration of the
# band+cross+pin-drop spec, MEASURED but not applied.
#
# Byte-identical to run_t9_band_cross_pindrop.sh (same pinned spec, seed, split,
# grid, crosses, Poisson refit, --exposure-pin-drop on) with one flag:
# --surface-amp-cal measure.
#
# WHY: the "the surface ships at 0.55x of its own training rate" claim and the
# "in-sample slope 2.03" number came from DIFFERENT specs. 2.03 is the amp
# arm's train-row slope (band+cross, volume covariates still free); the
# pin-drop spec was never asked for it, so the two were being reported
# together without ever being measured together. This run measures both on
# the spec that actually ships:
#
#   in_sample_slope        compression of the response gradient (scale-free)
#   in_sample_level_ratio  sum(pred)/sum(Y) on train game rows (the level)
#   in_sample_rate_ratio   pred/E over Y/E, the analogue of the 0.55x figure
#
# The slope CANNOT answer the level question on its own -- a surface that is
# 0.5x everywhere has slope 1.0 exactly -- which is why the level ratio is
# recorded too. `measure` leaves every rate column untouched, so the run
# answers the question without shipping a different model; it is not an
# amplitude-calibration run.
#
# The in-sample number is available right after the refit
# (results_tests/t9_ab_band_cross_pindrop_ampmeasure/surface_amp_in_sample.json),
# so the run can be stopped there instead of waiting ~40 min for prediction.
#
# Usage: bash scripts/run_t9_band_cross_pindrop_ampmeasure.sh [pinned_spec.json]
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT"

PIN="${1:-results_tests/t8_ar_fix/game_day_search_best.json}"
[ -s "$PIN" ] || { echo "FATAL: missing pinned spec $PIN" >&2; exit 1; }
OUT="results_tests/t9_ab_band_cross_pindrop_ampmeasure"
FIG="figures_tests/t9_ab_band_cross_pindrop_ampmeasure"
mkdir -p logs

COMMON=(
  --search-iters 0 --search-chunk 1 --n-starts 1 --search-R 5 --refit-R 10
  --search-vars slim --target trips_mean_gt_10 --group-by event --seed 7
  --force-grouped-game on --two-stage off
  --exposure offset --exposure-calibration on
  --temporal-mode ar1 --temporal-refit ar2
  --n-future 4 --resume --group-temporal off
)
TERMS="log_att,in_event,rel_sin,rel_cos,ko_tod_sin,ko_tod_cos,big_game"
STAMP=$(date +%Y%m%d_%H%M%S)

rm -rf "$OUT" "$FIG"
mkdir -p "$OUT" "$FIG"
python - "$PIN" "$OUT/game_day_search_best.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
j = json.load(open(src))
j["group_temporal"] = False
j["pinned_for"] = "t9 band+cross+pin-drop, in-sample calibration measured not applied"
json.dump(j, open(dst, "w"), indent=1)
print(f"   pinned spec: {dst} (BIC={j.get('best_bic')}, "
      f"iters={j.get('iters_completed')})")
PY

echo "== 59 grain=cell bands=band + ko_x_flow + pin-drop + amp MEASURE -> $OUT =="
python -u scripts/59_game_day_search_predict.py \
  --out-dir "$OUT" --fig-dir "$FIG" \
  --grouped-game-grain cell --grouped-game-bands band \
  --grouped-game-terms "$TERMS" \
  --exposure-pin-drop on \
  --surface-amp-cal measure \
  "${COMMON[@]}" 2>&1 | tee "logs/t9_band_cross_pindrop_ampmeasure_$STAMP.log"
RC=${PIPESTATUS[0]}
echo "== band+cross+pin-drop+amp-measure exit: $RC =="
echo "$RC" > "$OUT/exit_code"
exit "$RC"
