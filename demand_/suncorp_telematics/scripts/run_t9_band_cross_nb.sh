#!/bin/bash
# run_t9_band_cross_nb.sh - the band+cross arm refitted as NB2 instead of Poisson.
#
# The band+cross refit came back as "poisson-full ... alpha=0.0000": the
# search's dispersion gene chose Poisson (0) for this panel and the refit
# honoured the spec. A Poisson mean surface fitted to counts that are 98.5%
# zero and mean 0.03 is systematically under-dispersed, and an under-dispersed
# surface shows up as exactly what the handoff gate rejects -- a calibration
# slope of 1.4613 instead of 1.0.
#
# This arm is byte-identical to run_t9_band_cross.sh (same pinned spec, seed,
# split, grid, forced crosses) with ONE change: --refit-model nb, which forces
# the NB2 family instead of following the spec's dispersion gene. Everything
# structural (fixed/random/grouped/hetero terms) is untouched, so the
# comparison isolates the dispersion.
#
# Two outcomes, both informative:
#   nb-full with a finite alpha -> the amplitude question is answered by the
#     dispersion and the calibration slope should move toward 1 on its own;
#   the ladder degrading -> NB2 has no finite MLE on a 1.5%-positive panel
#     (alpha runs to the package clip), the run FAILS the fit-fidelity gate,
#     and the conclusion is that the Poisson collapse is the data, not a
#     search bug.
#
# Usage: bash scripts/run_t9_band_cross_nb.sh [pinned_spec.json]
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT"

PIN="${1:-results_tests/t8_ar_fix/game_day_search_best.json}"
[ -s "$PIN" ] || { echo "FATAL: missing pinned spec $PIN" >&2; exit 1; }
OUT="results_tests/t9_ab_band_cross_nb"
FIG="figures_tests/t9_ab_band_cross_nb"
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
j["pinned_for"] = "t9 band grain + crosses, NB2 likelihood (dispersion forced)"
json.dump(j, open(dst, "w"), indent=1)
print(f"   pinned spec: {dst} (BIC={j.get('best_bic')}, "
      f"iters={j.get('iters_completed')}, dispersion gene={j.get('dispersion')})")
PY

echo "== 59 grain=band + ko_x_flow + --refit-model nb -> $OUT =="
python -u scripts/59_game_day_search_predict.py \
  --out-dir "$OUT" --fig-dir "$FIG" \
  --grouped-game-grain cell --grouped-game-bands band \
  --grouped-game-terms "$TERMS" \
  --refit-model nb \
  "${COMMON[@]}" 2>&1 | tee "logs/t9_band_cross_nb_$STAMP.log"
RC=${PIPESTATUS[0]}
echo "== band+cross+NB2 exit: $RC =="
echo "$RC" > "$OUT/exit_code"
