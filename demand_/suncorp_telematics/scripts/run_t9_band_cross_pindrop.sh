#!/bin/bash
# run_t9_band_cross_pindrop.sh - the band+cross arm with the exposure pin made real.
#
# Byte-identical to run_t9_band_cross.sh (same pinned spec, seed, split, grid,
# forced kickoff x flow crosses, Poisson refit) with ONE change:
# --exposure-pin-drop on, which removes the free volume covariates
# (log_flow_z fixed, aadt_z random slope) from the refit design.
#
# WHY: --exposure offset pins the logexp coefficient at 1.0, but log_flow_z is
# built from the same seg_flow as the offset and was fitted free at c=-0.265
# with slope b=0.512 on logexp, so the NET elasticity was 1 + c/b = 0.48. The
# shipped surface therefore responds to flow at half strength: it is compressed
# by 2.07x, which is the in-sample calibration slope of 2.03 the amplitude
# experiments chased, and it under-predicts the holdout by 3.95x with an
# exposure gradient 8.7x too steep (1.08x deficit in the bottom octile, 9.38x
# in the top). Measured by scripts/diagnose_exposure_transfer.py.
#
# Three outcomes, all informative:
#   slope -> ~1 and the octile gradient flattens -> the pin was the defect;
#   slope unchanged -> the compression is not the elasticity, and the volume
#     covariates were doing other work;
#   the level stays 3-4x short -> the residual deficit is the game-day channel
#     (crowd lift) that this parameterisation cannot deliver, not exposure.
#
# Usage: bash scripts/run_t9_band_cross_pindrop.sh [pinned_spec.json]
set -u
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT"

PIN="${1:-results_tests/t8_ar_fix/game_day_search_best.json}"
[ -s "$PIN" ] || { echo "FATAL: missing pinned spec $PIN" >&2; exit 1; }
OUT="results_tests/t9_ab_band_cross_pindrop"
FIG="figures_tests/t9_ab_band_cross_pindrop"
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
j["pinned_for"] = "t9 band+cross with the exposure pin made real (search skipped)"
json.dump(j, open(dst, "w"), indent=1)
print(f"   pinned spec: {dst} (BIC={j.get('best_bic')}, "
      f"iters={j.get('iters_completed')})")
PY

echo "== 59 grain=cell bands=band + ko_x_flow + --exposure-pin-drop on -> $OUT =="
python -u scripts/59_game_day_search_predict.py \
  --out-dir "$OUT" --fig-dir "$FIG" \
  --grouped-game-grain cell --grouped-game-bands band \
  --grouped-game-terms "$TERMS" \
  --exposure-pin-drop on \
  "${COMMON[@]}" 2>&1 | tee "logs/t9_band_cross_pindrop_$STAMP.log"
RC=${PIPESTATUS[0]}
echo "== band+cross+pin-drop exit: $RC =="
echo "$RC" > "$OUT/exit_code"
echo "== exposure gradient on this arm =="
python scripts/diagnose_exposure_transfer.py "$OUT" 2>&1 \
  | tee "$OUT/exposure_transfer.log" || true
exit "$RC"
