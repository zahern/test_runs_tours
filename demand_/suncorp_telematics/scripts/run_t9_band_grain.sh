#!/bin/bash
# run_t9_band_grain.sh - third arm of the grain comparison: three distance
# bands, road class collapsed away (--grouped-game-bands band).
#
# Same pinned spec, same seed, same split, same flags as run_t9_ab_grain.sh,
# so the only difference from the `cell` arm is the resolution of the grouping
# key:
#   cell : road class x distance band -> 12 groups (9 populated, thinnest 16
#          events against 7 grouped coefficients)
#   band : distance band only           -> 3 groups (~145 / ~2.1k / ~1k events
#          against 21 coefficients)
#
# Writes results_tests/t9_ab_band so compare_t9_ab_grain.py reads it next to
# the cell and fixture arms:
#   python scripts/compare_t9_ab_grain.py cell band
#
# Usage: bash scripts/run_t9_band_grain.sh [pinned_spec.json]
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT"

PIN="${1:-results_tests/t8_ar_fix/game_day_search_best.json}"
[ -s "$PIN" ] || { echo "FATAL: missing pinned spec $PIN" >&2; exit 1; }
OUT="results_tests/t9_ab_band"
FIG="figures_tests/t9_ab_band"
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
j["pinned_for"] = "t9 band grain (3 distance bands, search skipped)"
json.dump(j, open(dst, "w"), indent=1)
print(f"   pinned spec: {dst} (BIC={j.get('best_bic')}, "
      f"iters={j.get('iters_completed')})")
PY

echo "== 59 grain=cell bands=band -> $OUT =="
python -u scripts/59_game_day_search_predict.py \
  --out-dir "$OUT" --fig-dir "$FIG" \
  --grouped-game-grain cell --grouped-game-bands band \
  --grouped-game-terms "$TERMS" \
  "${COMMON[@]}" 2>&1 | tee "logs/t9_band_$STAMP.log"
RC=${PIPESTATUS[0]}
echo "== grain=band exit: $RC =="
echo "$RC" > "$OUT/exit_code"
