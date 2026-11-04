#!/bin/bash
# run_game_day_chain.sh - local sequential chain after 58's full pass:
#   59 (search + future-game predictions) -> 60 (site-selection MIP + plots)
# Assumes results/segment_15min_panel.parquet exists (58 --merge done).
# 59's handoff gate is enforced here: this chain used to print "59 exit: N"
# and run 60 anyway, so a surface that could not reorder by game still got a
# deployment plan built from it.
cd "$(dirname "$0")"
set -u
LOG_DIR=logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

echo "== 59 full search + predict =="
# Chunked, resumable search (R=40, <=100k-row sampled train frame). Re-running
# continues from results/game_day_search_best.json until --search-iters is met.
RESUME_FLAG=""
[ -s results/game_day_search_best.json ] && RESUME_FLAG="--resume"
python -u scripts/59_game_day_search_predict.py \
  --search-iters 120 --search-chunk 25 --chunk-time 3600 \
  --wall-budget 18000 --n-starts 2 \
  --search-R 40 --refit-R 200 --search-vars slim $RESUME_FLAG \
  > "$LOG_DIR/59_full_$STAMP.log" 2> "$LOG_DIR/59_full_$STAMP.log.err"
RC59=$?
echo "59 exit: $RC59"
if [ "$RC59" -ne 0 ]; then
  echo "chain ABORTED: 59 failed (see $LOG_DIR/59_full_$STAMP.log)."
  grep -h "handoff FAIL\|GATE FAIL" "$LOG_DIR/59_full_$STAMP.log" 2>/dev/null | head -5
  echo "  60 is not run: a surface that failed the handoff gate must not be"
  echo "  turned into a deployment plan. Override with ALLOW_WEAK_SURFACE=1"
  echo "  only for a deliberate exploration run."
  exit "$RC59"
fi

echo "== 60 MIP + plots =="
python -u scripts/60_game_day_mip.py --min-marg 0.5 \
  > "$LOG_DIR/60_full_$STAMP.log" 2> "$LOG_DIR/60_full_$STAMP.log.err"
echo "60 exit: $?"
echo "chain done: $STAMP"
