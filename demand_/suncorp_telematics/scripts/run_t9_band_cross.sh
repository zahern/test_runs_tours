#!/bin/bash
# run_t9_band_cross.sh - the band grain WITH the kickoff x flow crosses.
#
# Identical to run_t9_band_grain.sh except that 59 now carries two forced
# terms, ko_sin_x_flow and ko_cos_x_flow (kickoff time-of-day x segment
# background flow). They exist because every previous crossed term multiplies
# a segment covariate by the CROWD, and one crowd scalar cannot reorder:
# for attendance A the multiplier is exp(log(A) * g(x)) for a single segment
# score g(x), which is monotone in g for every A. That is why pairwise surface
# Spearman sat at 0.997 across all three grains. Kickoff is the only fixture
# attribute that varies independently of the crowd, so sin/cos(kickoff) x flow
# adds a second scoring direction and the ordering CAN move between an
# afternoon and an evening game.
#
# Writes results_tests/t9_ab_band_cross so the plain band arm is preserved:
#   python scripts/compare_t9_ab_grain.py band band_cross
#
# Usage: bash scripts/run_t9_band_cross.sh [pinned_spec.json]
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT"

PIN="${1:-results_tests/t8_ar_fix/game_day_search_best.json}"
[ -s "$PIN" ] || { echo "FATAL: missing pinned spec $PIN" >&2; exit 1; }
OUT="results_tests/t9_ab_band_cross"
FIG="figures_tests/t9_ab_band_cross"
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
j["pinned_for"] = "t9 band grain + kickoff x flow crosses (search skipped)"
json.dump(j, open(dst, "w"), indent=1)
print(f"   pinned spec: {dst} (BIC={j.get('best_bic')}, "
      f"iters={j.get('iters_completed')})")
PY

echo "== 59 grain=cell bands=band + ko_x_flow -> $OUT =="
python -u scripts/59_game_day_search_predict.py \
  --out-dir "$OUT" --fig-dir "$FIG" \
  --grouped-game-grain cell --grouped-game-bands band \
  --grouped-game-terms "$TERMS" \
  "${COMMON[@]}" 2>&1 | tee "logs/t9_band_cross_$STAMP.log"
RC=${PIPESTATUS[0]}
echo "== band+cross exit: $RC =="
echo "$RC" > "$OUT/exit_code"
