#!/bin/bash
# resubmit_paper_finish.sh
#
# Submits the outstanding compute passes needed to finish the paper, after the
# 22-23 Oct failures:
#
#   1. job_predict_from_merged.pbs   held-out predict/validation + calibrated
#                                    mode shares + paper Stage-B refresh.
#                                    Previously OOM (exit 137) at 192GB; now
#                                    requests 400GB.
#   2. job_ps_iv_nonmand.pbs         re-runs PS-IV/nonmandatory, whose rho=0.9
#                                    replication was OOM-killed in 27013818[8].
#   3. job_ps_merge.pbs              re-merges ps_diagnostics.csv/auto_ps.tex
#                                    after (2) (chained afterok).
#
# The safety chain (27016405-27016408) is already queued/held; this script does
# not touch it.  Run from the code dir on an aqua login node:
#
#   bash resubmit_paper_finish.sh              # submit everything
#   bash resubmit_paper_finish.sh --predict    # predict pass only
#   bash resubmit_paper_finish.sh --ps         # PS-IV re-run + merge only
#
set -euo pipefail
cd "$(dirname "$0")"

RUN_TAG=${RUN_TAG:-paper_missing_20260929}
RUN_ROOT=${RUN_ROOT:-runs/ga_parallel}
PS_ROOT=${PS_ROOT:-runs/ps_inference}
PS_B=${PS_B:-20}
GA_BUDGET=${GA_BUDGET:-smoke}
PROJECT=${PROJECT:-FPAXJW8067}

DO_PREDICT=1
DO_PS=1
case "${1:-}" in
  --predict) DO_PS=0 ;;
  --ps)      DO_PREDICT=0 ;;
  "" ) ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

echo "Run tag : ${RUN_TAG}"
echo "Project : ${PROJECT}"
echo

if [ "${DO_PREDICT}" = "1" ]; then
  echo "=== [1/3] predict + validate + publish (400GB) ==="
  J_PRED=$(qsub -P "${PROJECT}" -j oe \
      -v "RUN_TAG=${RUN_TAG},RUN_ROOT=${RUN_ROOT},PUBLISH_PAPER=1" \
      job_predict_from_merged.pbs | awk '{print $1}')
  echo "  predict -> ${J_PRED}"
  echo
fi

if [ "${DO_PS}" = "1" ]; then
  echo "=== [2/3] PS-IV/nonmandatory re-run (160GB) ==="
  J_PS=$(qsub -P "${PROJECT}" -j oe \
      -v "RUN_TAG=${RUN_TAG},RUN_ROOT=${RUN_ROOT},PS_ROOT=${PS_ROOT},PS_B=${PS_B},GA_BUDGET=${GA_BUDGET}" \
      job_ps_iv_nonmand.pbs | awk '{print $1}')
  echo "  ps_iv_nonmand -> ${J_PS}"

  echo "=== [3/3] re-merge PS tables (afterok ${J_PS}) ==="
  J_MERGE=$(qsub -P "${PROJECT}" -j oe -W "depend=afterok:${J_PS}" \
      -v "RUN_TAG=${RUN_TAG},RUN_ROOT=${RUN_ROOT},PS_ROOT=${PS_ROOT}" \
      job_ps_merge.pbs | awk '{print $1}')
  echo "  ps_merge -> ${J_MERGE}  (after ${J_PS})"
  echo
fi

echo "--- current queue ---"
qstat -u "$(whoami)" || true
echo
echo "Logs: ga_predict_merged.o<id>, ps_iv_nonmand.o<id>, ps_merge.o<id>"
