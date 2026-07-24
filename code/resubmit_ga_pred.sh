#!/bin/bash
# Resubmit the two ga_pred jobs killed by the 24h walltime (23540863, 23559237).
#
# Both died in Stage 6's per-activity destination repair loop (see the
# post-mortem comments in stage6_plan_finalisation.py). The code fixes are in
# place; this rerun:
#   - drops the -W depend (the merge jobs completed successfully),
#   - reuses each dead job's PIPELINE_RUN_TAG so Stage-5 prediction resumes
#     from the 7 batch checkpoints in the existing run dir (~5h saved;
#     estimation itself re-runs, ~1.5h, because startup clears stale model
#     pickles).
#
# Usage:
#   bash resubmit_ga_pred.sh            # submit both chains
#   bash resubmit_ga_pred.sh chain1     # standard-pop chain only (was 23540863)
#   bash resubmit_ga_pred.sh chain2     # new_pop-mapped chain only (was 23559237)

set -euo pipefail
cd "$(dirname "$0")"

COMMON_VARS="MODE=ga_staged,CONDA_ENV=zigenv,RUN_ROOT=runs/ga_parallel,THREADS=1,S5_WORKERS=8"
COMMON_VARS+=",GA_ESTIMATOR=sa_bandit,GA_USE_SA=1,GA_USE_BANDIT_SA=1,GA_BUDGET=fast"
COMMON_VARS+=",GA_STAGED_REUSE_ESTIMATOR=1,INSTALL_ESTIMATORS=1,PIPELINE_ACCELERATOR=cpu"
COMMON_VARS+=",GA_N_RESTARTS=0,GA_SIG_PENALTY_WEIGHT=1.50,GA_SIG_PVAL_THRESH=0.05"
COMMON_VARS+=",NEW_POP=0,NEW_POP_DIR=../new_pop,PIPELINE_SKIMS_PATH="

submit_chain1() {
    qsub -N ga_pred_rerun_23540772 \
        -l select=1:ncpus=8:mem=250GB -l walltime=24:00:00 \
        -v "${COMMON_VARS},NEW_POP_MAPPED=0,GA_STAGED_GA_DIR=runs/ga_parallel/ga_parallel_23540772.aqua/merged,PIPELINE_RUN_TAG=23540863.aqua" \
        pbs_pipeline_mode.pbs
}

submit_chain2() {
    qsub -N ga_pred_rerun_23559212 \
        -l select=1:ncpus=8:mem=250GB -l walltime=24:00:00 \
        -v "${COMMON_VARS},NEW_POP_MAPPED=1,GA_STAGED_GA_DIR=runs/ga_parallel/ga_parallel_23559212.aqua/merged,PIPELINE_RUN_TAG=23559237.aqua" \
        pbs_pipeline_mode.pbs
}

case "${1:-both}" in
    chain1) echo "Submitting chain 1 (standard pop)...";  submit_chain1 ;;
    chain2) echo "Submitting chain 2 (new_pop mapped)..."; submit_chain2 ;;
    both)   echo "Submitting chain 1 (standard pop)...";  submit_chain1
            echo "Submitting chain 2 (new_pop mapped)..."; submit_chain2 ;;
    *)      echo "Usage: $0 [chain1|chain2|both]" >&2; exit 1 ;;
esac

echo
qstat -u "$(whoami)" || true
