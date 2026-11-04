#!/bin/bash
# run_t9_cell_response.sh - re-run the cell-grain arm once to capture the
# per-cell game-day response table (game_day_cell_response.csv).
#
# The A/B arms already on disk (results_tests/t9_ab_*) saved only the
# coefficient table, whose grouped rows are on the design's standardised
# scale. The per-cell multipliers therefore have to be read off the fitted
# model, which only exists while 59 is running: this rerun is the same pinned
# spec, same seed and same flags as the cell arm, writing to its own directory
# so the A/B artifacts stay untouched. The probe is dumped right after the
# refit, so the run can be stopped once the CSV appears.
#
# Usage: bash scripts/run_t9_cell_response.sh [pinned_spec.json]
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT"

PIN="${1:-results_tests/t8_ar_fix/game_day_search_best.json}"
[ -s "$PIN" ] || { echo "FATAL: missing pinned spec $PIN" >&2; exit 1; }
OUT="results_tests/t9_cell_response"
FIG="figures_tests/t9_cell_response"
mkdir -p "$OUT" "$FIG" logs
rm -f "$OUT"/game_day_cell_response.csv

python - "$PIN" "$OUT/game_day_search_best.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
j = json.load(open(src))
j["group_temporal"] = False
j["pinned_for"] = "t9 cell response probe (search skipped)"
json.dump(j, open(dst, "w"), indent=1)
print(f"   pinned spec: {dst} (BIC={j.get('best_bic')}, "
      f"iters={j.get('iters_completed')})")
PY

STAMP=$(date +%Y%m%d_%H%M%S)
TERMS="log_att,in_event,rel_sin,rel_cos,ko_tod_sin,ko_tod_cos,big_game"
echo "== 59 grain=cell -> $OUT (cell response probe) =="
python -u scripts/59_game_day_search_predict.py \
  --out-dir "$OUT" --fig-dir "$FIG" \
  --grouped-game-grain cell --grouped-game-terms "$TERMS" \
  --search-iters 0 --search-chunk 1 --n-starts 1 --search-R 5 --refit-R 10 \
  --search-vars slim --target trips_mean_gt_10 --group-by event --seed 7 \
  --force-grouped-game on --two-stage off \
  --exposure offset --exposure-calibration on \
  --temporal-mode ar1 --temporal-refit ar2 \
  --n-future 4 --resume --group-temporal off \
  2>&1 | tee "logs/t9_cell_response_$STAMP.log"
echo "== exit: ${PIPESTATUS[0]} =="
