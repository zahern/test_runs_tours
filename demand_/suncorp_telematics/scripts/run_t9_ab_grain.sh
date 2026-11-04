#!/bin/bash
# run_t9_ab_grain.sh - A/B the game-day surface at two grouping grains.
#
# Runs 59 twice on the SAME split and the SAME searched spec, changing only
# --grouped-game-grain:
#   cell    : 12 groups (road class x distance band), each with one
#             coefficient per game-day attribute  -> 12 x k coefficients
#   fixture : 39 groups, each with one coefficient per segment attribute
#             -> 39 x k coefficients (the T9 parameterisation)
#
# The search is SKIPPED for both arms (--search-iters 0 --resume against a
# pinned checkpoint), so the only difference between the two runs is the
# grain. Comparing search output against refit output would otherwise
# attribute the spec's effect to the grain.
#
# Usage: bash scripts/run_t9_ab_grain.sh [pinned_spec.json]
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT"

PIN="${1:-results_tests/t8_ar_fix/game_day_search_best.json}"
[ -s "$PIN" ] || { echo "FATAL: missing pinned spec $PIN" >&2; exit 1; }
mkdir -p logs

# The pinned spec must use families the LOCAL package knows. The t9 checkpoint
# was searched on the cluster (metacountregressor >= 1.0.188) and carries
# 'neglognormal' for maxspeed_z/aadt_z, which 1.0.161 does not implement: the
# refit then dies at nb-full with "'neglognormal'" and walks the ladder to the
# fixed-only fallback for a reason that has nothing to do with the grain.
# t8's spec (the SRC the t9 pbs copies anyway) uses only
# normal/lognormal/triangular/uniform, so both arms get the same real fit.

# Identical for both arms; --group-temporal off matches the pinned
# checkpoint (a mismatch makes 59 discard it and re-search).
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

for GRAIN in cell fixture; do
  OUT="results_tests/t9_ab_$GRAIN"
  FIG="figures_tests/t9_ab_$GRAIN"
  rm -rf "$OUT" "$FIG"
  mkdir -p "$OUT" "$FIG"
  # Copy the pin with group_temporal cleared: the flag only records how the
  # SEARCH space was parameterised, and 59 discards a checkpoint whose flag
  # disagrees with --group-temporal. The spec's terms are copied untouched.
  python - "$PIN" "$OUT/game_day_search_best.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
j = json.load(open(src))
j["group_temporal"] = False
j["pinned_for"] = "t9 grain A/B (search skipped)"
json.dump(j, open(dst, "w"), indent=1)
print(f"   pinned spec: {dst} (BIC={j.get('best_bic')}, "
      f"iters={j.get('iters_completed')})")
PY
  echo "== 59 grain=$GRAIN -> $OUT =="
  python -u scripts/59_game_day_search_predict.py \
    --out-dir "$OUT" --fig-dir "$FIG" \
    --grouped-game-grain "$GRAIN" --grouped-game-terms "$TERMS" \
    "${COMMON[@]}" 2>&1 | tee "logs/t9_ab_${GRAIN}_$STAMP.log"
  RC=${PIPESTATUS[0]}
  echo "== grain=$GRAIN exit: $RC =="
  echo "$RC" > "$OUT/exit_code"
done

python scripts/compare_t9_ab_grain.py 2>&1 | tee "logs/t9_ab_compare_$STAMP.log"
