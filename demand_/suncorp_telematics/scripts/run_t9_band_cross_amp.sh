#!/bin/bash
# run_t9_band_cross_amp.sh - band grain + kickoff x flow crosses, with the
# surface amplitude calibrated on TRAIN rows.
#
# Identical to run_t9_band_cross.sh (same pinned spec, same seed, same split,
# same grid) plus one flag: --surface-amp-cal on. That flag fits the exponent
# k of p -> m(p/m)^k from the in-sample calibration slope of the FINAL count
# fit on train game rows, then applies it to the future rate columns.
#
# Why this and not a shrink of the crossed coefficients: the shrink was
# measured as impossible (scripts/reorder_shrink_curve.py) -- calibration
# needs the crossed coefficients at s <= 0.65 while the reordering criterion
# needs s >= 0.98, so the two admissible sets are disjoint. Amplitude is the
# one channel that moves calibration without moving any rank, so if the gate
# can be satisfied at all without giving up the reordering, it is here.
#
# Writes results_tests/t9_ab_band_cross_amp so both earlier arms stay intact:
#   python scripts/compare_t9_ab_grain.py band_cross band_cross_amp
#
# Usage: bash scripts/run_t9_band_cross_amp.sh [pinned_spec.json]
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT"

PIN="${1:-results_tests/t8_ar_fix/game_day_search_best.json}"
[ -s "$PIN" ] || { echo "FATAL: missing pinned spec $PIN" >&2; exit 1; }
OUT="results_tests/t9_ab_band_cross_amp"
FIG="figures_tests/t9_ab_band_cross_amp"
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
j["pinned_for"] = "t9 band grain + crosses + train-fitted amplitude cal"
json.dump(j, open(dst, "w"), indent=1)
print(f"   pinned spec: {dst} (BIC={j.get('best_bic')}, "
      f"iters={j.get('iters_completed')})")
PY

echo "== 59 grain=band + ko_x_flow + surface-amp-cal -> $OUT =="
python -u scripts/59_game_day_search_predict.py \
  --out-dir "$OUT" --fig-dir "$FIG" \
  --grouped-game-grain cell --grouped-game-bands band \
  --grouped-game-terms "$TERMS" \
  --surface-amp-cal on \
  "${COMMON[@]}" 2>&1 | tee "logs/t9_band_cross_amp_$STAMP.log"
RC=${PIPESTATUS[0]}
echo "== band+cross+amp exit: $RC =="
echo "$RC" > "$OUT/exit_code"
