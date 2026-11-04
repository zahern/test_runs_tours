#!/usr/bin/env bash
# Wait for the Stage 3 prediction to swap its output in, then re-run Stage 4
# (schedule placement) over the new durations, in person-aligned chunks.
#
# Started detached from the tool call so it survives the session; everything
# it does is appended to _tmp_s4_predict.log.
set -u
cd "$(dirname "$0")" || exit 1

S3_LOG=_tmp_s3_predict_fast.log
RUN=runs/agds_default_staged__paper_predict_20261024

echo "[chain] $(date +%H:%M:%S) waiting for Stage 3 to finish"
until grep -q "done: stage3_output.csv replaced" "$S3_LOG" 2>/dev/null; do
    sleep 60
done
echo "[chain] $(date +%H:%M:%S) Stage 3 finished; verifying output"

if [ ! -s "$RUN/stage3_output.csv" ]; then
    echo "[chain] stage3_output.csv missing - aborting" >&2
    exit 1
fi
if [ -f "$RUN/_stage3_predict_progress.json" ]; then
    echo "[chain] progress file still present - aborting" >&2
    exit 1
fi

echo "[chain] $(date +%H:%M:%S) starting Stage 4 prediction"
python _rerun_stage4_paper.py --predict --chunk 50000 >> _tmp_s4_predict.log 2>&1
rc=$?
echo "[chain] $(date +%H:%M:%S) Stage 4 exit=$rc"
exit $rc
