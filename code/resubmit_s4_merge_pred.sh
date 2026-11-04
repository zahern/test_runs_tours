#!/bin/bash
# Resubmit Stage 4 + merge + predict for gui_enc3 and gui_enc5.
#
# Background
# ----------
# Both chains (gui_enc3, gui_enc5) had Stage 4 jobs (24067531 / 24067608)
# die on the cluster without producing any PBS output files. Stages 0-3 and 5
# completed fine; the merge/predict jobs (depend=afterok:all_stages) therefore
# never ran.
#
# Stage 5 error file ga_s5_gui_enc5.e24067609 contains only non-fatal
# WARNINGs (parameter-at-bound, collinear CA columns, Larch datatree API
# unavailable). Stage 5 itself finished with exit_code=0.
#
# This script:
#   1. Resubmits stage 4 for each run tag.
#   2. Chains a merge job (afterok:new_s4).
#   3. Chains a predict job (afterok:merge).
#
# Stages 0-3 and 5 are already complete; the merge script (STRICT_MERGE=1)
# will pick up all six stage directories (0-5) once stage 4 finishes.
#
# Usage:
#   bash resubmit_s4_merge_pred.sh            # both chains
#   bash resubmit_s4_merge_pred.sh enc3       # gui_enc3 only
#   bash resubmit_s4_merge_pred.sh enc5       # gui_enc5 only

set -euo pipefail
cd "$(dirname "$0")"

RUN_ROOT="runs/ga_parallel"
CONDA_ENV="zigenv"

# ── Resource shapes (match original submission) ───────────────────────────
STAGE_SELECT="1:ncpus=8:mem=128GB"
STAGE_WALLTIME="36:00:00"
MERGE_SELECT="1:ncpus=1:mem=8GB"
MERGE_WALLTIME="01:00:00"
PRED_SELECT="1:ncpus=8:mem=250GB"
PRED_WALLTIME="24:00:00"

# ── Common stage-4 variables (shared by both runs) ────────────────────────
COMMON_STAGE_VARS="RUN_ROOT=${RUN_ROOT},CONDA_ENV=${CONDA_ENV}"
COMMON_STAGE_VARS+=",GA_BUDGET=standard,GA_ESTIMATOR=default"
COMMON_STAGE_VARS+=",GA_USE_SA=1,GA_USE_BANDIT_SA=1,GA_N_RESTARTS=0"
COMMON_STAGE_VARS+=",GA_PARALLEL_SEGMENTS=4,PIPELINE_ACCELERATOR=cpu"
COMMON_STAGE_VARS+=",STAGE23_JOINT=0,GA_SIG_PENALTY_WEIGHT=1.50,GA_SIG_PVAL_THRESH=0.05"
COMMON_STAGE_VARS+=",GA_EARLY_STOP=1,GA_S1_RIDGE=2.0,GA_IMPUTE_INCOME=1,GA_RESOLVE_CONFLICTS=1"
COMMON_STAGE_VARS+=",NEW_POP=0,NEW_POP_MAPPED=1,NEW_POP_DIR=../new_pop,PIPELINE_SKIMS_PATH="

# ── Common predict variables (shared by both runs) ────────────────────────
COMMON_PRED_VARS="MODE=ga_staged,CONDA_ENV=${CONDA_ENV},RUN_ROOT=${RUN_ROOT}"
COMMON_PRED_VARS+=",THREADS=1,S5_WORKERS=8,PIPELINE_ACCELERATOR=cpu"
COMMON_PRED_VARS+=",GA_ESTIMATOR=default,GA_USE_SA=1,GA_USE_BANDIT_SA=1"
COMMON_PRED_VARS+=",GA_BUDGET=standard,GA_STAGED_REUSE_ESTIMATOR=1,INSTALL_ESTIMATORS=1"
COMMON_PRED_VARS+=",GA_N_RESTARTS=0,GA_SIG_PENALTY_WEIGHT=1.50,GA_SIG_PVAL_THRESH=0.05"
COMMON_PRED_VARS+=",NEW_POP=0,NEW_POP_MAPPED=1,NEW_POP_DIR=../new_pop,PIPELINE_SKIMS_PATH="

# ─────────────────────────────────────────────────────────────────────────
submit_enc3() {
    local RUN_TAG="gui_enc3"
    local MERGED_DIR="${RUN_ROOT}/${RUN_TAG}/merged"

    echo ""
    echo "=== Submitting gui_enc3 stage 4 → merge → predict ==="

    # Stage 4 (enc3 did NOT use STAGE5_ENABLE_MXL)
    local jid4
    jid4=$(qsub \
        -N "ga_s4_${RUN_TAG}_rerun" \
        -l "select=${STAGE_SELECT}" \
        -l "walltime=${STAGE_WALLTIME}" \
        -v "STAGE_ID=4,RUN_TAG=${RUN_TAG},${COMMON_STAGE_VARS},STAGE5_ENABLE_MXL=0" \
        job_ga_stage_single.pbs | awk '{print $1}')
    echo "  Stage 4 submitted: ${jid4}"

    # Merge (picks up stage dirs 0-5; stage 5 already present)
    local jid_merge
    jid_merge=$(qsub \
        -N "ga_merge_${RUN_TAG}_rerun" \
        -W "depend=afterok:${jid4}" \
        -l "select=${MERGE_SELECT}" \
        -l "walltime=${MERGE_WALLTIME}" \
        -v "RUN_ROOT=${RUN_ROOT},RUN_TAG=${RUN_TAG},CONDA_ENV=${CONDA_ENV},STAGE_IDS=0_1_2_3_4_5,STRICT_MERGE=1" \
        job_merge_ga_results.pbs | awk '{print $1}')
    echo "  Merge submitted:   ${jid_merge}"

    # Predict (full ga_staged pipeline run)
    local jid_pred
    jid_pred=$(qsub \
        -N "ga_pred_${RUN_TAG}_rerun" \
        -W "depend=afterok:${jid_merge}" \
        -l "select=${PRED_SELECT}" \
        -l "walltime=${PRED_WALLTIME}" \
        -v "${COMMON_PRED_VARS},STAGE5_ENABLE_MXL=0,GA_STAGED_GA_DIR=${MERGED_DIR}" \
        pbs_pipeline_mode.pbs | awk '{print $1}')
    echo "  Predict submitted: ${jid_pred}"

    echo "  Chain: ${jid4} → ${jid_merge} → ${jid_pred}"
}

# ─────────────────────────────────────────────────────────────────────────
submit_enc5() {
    local RUN_TAG="gui_enc5"
    local MERGED_DIR="${RUN_ROOT}/${RUN_TAG}/merged"

    echo ""
    echo "=== Submitting gui_enc5 stage 4 → merge → predict ==="

    # Stage 4 (enc5 used STAGE5_ENABLE_MXL=1)
    local jid4
    jid4=$(qsub \
        -N "ga_s4_${RUN_TAG}_rerun" \
        -l "select=${STAGE_SELECT}" \
        -l "walltime=${STAGE_WALLTIME}" \
        -v "STAGE_ID=4,RUN_TAG=${RUN_TAG},${COMMON_STAGE_VARS},STAGE5_ENABLE_MXL=1" \
        job_ga_stage_single.pbs | awk '{print $1}')
    echo "  Stage 4 submitted: ${jid4}"

    # Merge
    local jid_merge
    jid_merge=$(qsub \
        -N "ga_merge_${RUN_TAG}_rerun" \
        -W "depend=afterok:${jid4}" \
        -l "select=${MERGE_SELECT}" \
        -l "walltime=${MERGE_WALLTIME}" \
        -v "RUN_ROOT=${RUN_ROOT},RUN_TAG=${RUN_TAG},CONDA_ENV=${CONDA_ENV},STAGE_IDS=0_1_2_3_4_5,STRICT_MERGE=1" \
        job_merge_ga_results.pbs | awk '{print $1}')
    echo "  Merge submitted:   ${jid_merge}"

    # Predict
    local jid_pred
    jid_pred=$(qsub \
        -N "ga_pred_${RUN_TAG}_rerun" \
        -W "depend=afterok:${jid_merge}" \
        -l "select=${PRED_SELECT}" \
        -l "walltime=${PRED_WALLTIME}" \
        -v "${COMMON_PRED_VARS},STAGE5_ENABLE_MXL=1,GA_STAGED_GA_DIR=${MERGED_DIR}" \
        pbs_pipeline_mode.pbs | awk '{print $1}')
    echo "  Predict submitted: ${jid_pred}"

    echo "  Chain: ${jid4} → ${jid_merge} → ${jid_pred}"
}

# ── Dispatch ──────────────────────────────────────────────────────────────
case "${1:-both}" in
    enc3)  submit_enc3 ;;
    enc5)  submit_enc5 ;;
    both)  submit_enc3; submit_enc5 ;;
    *)     echo "Usage: $0 [enc3|enc5|both]" >&2; exit 1 ;;
esac

echo ""
echo "Done. Current queue:"
qstat -u "$(whoami)" || true
