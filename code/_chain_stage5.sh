#!/usr/bin/env bash
# Wait for the Stage 4 re-run to swap its output in, then rebuild the Stage 5
# trip chains from the new schedule, assemble them, and finalise the plan.
#
# Everything runs in ONE python process: stage5_models.pkl cannot be read back
# in this environment (Larch unpickling raises on state the installed build no
# longer restores), so the models have to stay in memory from estimation
# through finalisation.
set -u
cd "$(dirname "$0")" || exit 1

RUN=runs/agds_default_staged__paper_predict_20261024

echo "[chain5] $(date +%H:%M:%S) waiting for Stage 4 to finish"
until grep -q "done: stage4_output.csv replaced" _tmp_s4_predict.log 2>/dev/null; do
    sleep 60
done
echo "[chain5] $(date +%H:%M:%S) Stage 4 finished"

if [ ! -s "$RUN/stage4_output.csv" ]; then
    echo "[chain5] stage4_output.csv missing - aborting" >&2
    exit 1
fi

echo "[chain5] $(date +%H:%M:%S) Stage 5 estimate + chains + assemble + finalize"
python _rerun_stage5_paper.py --estimate --predict --assemble --finalize \
       --chunk 30000 >> _tmp_s5_run.log 2>&1
rc=$?
echo "[chain5] $(date +%H:%M:%S) Stage 5 exit=$rc"
exit $rc
