#!/usr/bin/env bash
# Stage 5 after the Stage 3/4 re-run: build the trip chains from the new
# schedule (chunked, checkpointed), assemble the plan CSV, then finalise the
# plan in a separate process so the assembly's memory is released first.
set -u
cd "$(dirname "$0")" || exit 1

echo "[s5b] $(date +%H:%M:%S) chains + assemble"
python _rerun_stage5_paper.py --predict --assemble --chunk 20000 >> _tmp_s5_run3.log 2>&1
rc=$?
echo "[s5b] $(date +%H:%M:%S) predict+assemble exit=$rc"
[ $rc -ne 0 ] && exit $rc

echo "[s5b] $(date +%H:%M:%S) finalize"
python _rerun_stage5_paper.py --finalize >> _tmp_s5_run3.log 2>&1
rc=$?
echo "[s5b] $(date +%H:%M:%S) finalize exit=$rc"
exit $rc
