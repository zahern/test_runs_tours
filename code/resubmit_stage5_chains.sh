#!/bin/bash
# Resubmit stage5 -> merge -> pred for logsumfix, destfix, tddsched, accessfix.
#
# Background: stage5 destination-choice models were unidentifiable because the
# importance-sampling log_correction is collinear with the locked size_term
# (|r|=1.0). stage5_destination_choice.py now orthogonalizes log_correction
# (STAGE5_ORTHO_CORRECTION, default 1). Re-running stage5 with the fixed code
# re-estimates the destination models; stage0-4 ga_results.pkl are reused
# unchanged. The logsumfix pred also gets 36h walltime (was killed at 24h).
#
# Usage:
#   bash resubmit_stage5_chains.sh            # all four chains
#   bash resubmit_stage5_chains.sh logsumfix  # one chain only

set -euo pipefail
cd "$(dirname "$0")"

RUN_ROOT="runs/ga_parallel"
CONDA_ENV="zigenv"

S5_SELECT="1:ncpus=8:mem=128GB"
S5_WALLTIME="36:00:00"
MERGE_SELECT="1:ncpus=1:mem=8GB"
MERGE_WALLTIME="01:00:00"
PRED_SELECT="1:ncpus=8:mem=250GB"

# ── Common stage-5 variables (replicate originals exactly) ───────────────
COMMON_STAGE_VARS="RUN_ROOT=${RUN_ROOT},RUN_TAG=%TAG%,CONDA_ENV=${CONDA_ENV}"
COMMON_STAGE_VARS+=",GA_BUDGET=standard,GA_ESTIMATOR=metacount_mv,GA_ENGINE=portfolio"
COMMON_STAGE_VARS+=",GA_USE_SA=0,GA_USE_BANDIT_SA=0,GA_N_RESTARTS=0,GA_PARALLEL_SEGMENTS=4"
COMMON_STAGE_VARS+=",PIPELINE_ACCELERATOR=cpu,STAGE23_JOINT=0,GA_SIG_PENALTY_WEIGHT=1.50"
COMMON_STAGE_VARS+=",GA_SIG_PVAL_THRESH=0.05,GA_EARLY_STOP=1,GA_S1_RIDGE=2.0,GA_IMPUTE_INCOME=1"
COMMON_STAGE_VARS+=",GA_RESOLVE_CONFLICTS=1,STAGE5_ENABLE_MXL=0,STAGE5_ENABLE_GSE=1"
COMMON_STAGE_VARS+=",STAGE5_CRASH_ZERO_TO_MAX=1,STAGE5_LOGSUM_SMOOTH=1"
COMMON_STAGE_VARS+=",NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop"
COMMON_STAGE_VARS+=",PIPELINE_SKIMS_PATH=../demand_/skims_371.omx,GA_PROFILE=0"
COMMON_STAGE_VARS+=",GA_COMPLEXITY_PENALTY=0.4,GA_REHEAT_FRACTION=0.3,GA_STAGE123_GENS_MULT=2.0"
COMMON_STAGE_VARS+=",GA_MUTATION_MIN=0.05,PIPELINE_PERSON_LATENT=0,PIPELINE_WFH=0"
COMMON_STAGE_VARS+=",PIPELINE_MOBILITY=0,PIPELINE_TOUR_SEQ=0,GA_MEM_PER_WORKER_GB="
COMMON_STAGE_VARS+=",DEST_EST_SAMPLE_SIZE=30,STAGE5_DEST_SAMPLE_SIZE=30"
COMMON_STAGE_VARS+=",GA_MODE_DEGEN_PENALTY=20,ESTIMATORS_SKIP_UPGRADE=1"

# ── Common predict variables ─────────────────────────────────────────────
COMMON_PRED_VARS="MODE=ga_staged,CONDA_ENV=${CONDA_ENV},RUN_ROOT=${RUN_ROOT}"
COMMON_PRED_VARS+=",THREADS=1,S5_WORKERS=8,PIPELINE_ACCELERATOR=cpu"
COMMON_PRED_VARS+=",GA_ESTIMATOR=metacount_mv,GA_ENGINE=portfolio,GA_USE_SA=0,GA_USE_BANDIT_SA=0"
COMMON_PRED_VARS+=",STAGE5_CRASH_ZERO_TO_MAX=1,STAGE5_LOGSUM_SMOOTH=1"
COMMON_PRED_VARS+=",GA_BUDGET=standard,GA_STAGED_REUSE_ESTIMATOR=1,GA_STAGED_GA_DIR=%TAG%"
COMMON_PRED_VARS+=",INSTALL_ESTIMATORS=1,GA_N_RESTARTS=0,GA_SIG_PENALTY_WEIGHT=1.50"
COMMON_PRED_VARS+=",GA_SIG_PVAL_THRESH=0.05,NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop"
COMMON_PRED_VARS+=",PIPELINE_SKIMS_PATH=../demand_/skims_371.omx,GA_PROFILE=0"
COMMON_PRED_VARS+=",GA_COMPLEXITY_PENALTY=0.4,GA_REHEAT_FRACTION=0.3"
COMMON_PRED_VARS+=",PIPELINE_PERSON_LATENT=0,PIPELINE_WFH=0,PIPELINE_MOBILITY=0"
COMMON_PRED_VARS+=",PIPELINE_TOUR_SEQ=0,GA_MEM_PER_WORKER_GB=,GA_SEARCH_PREDAY=1"
COMMON_PRED_VARS+=",STAGE6_TOUR_MODE_CONSISTENCY=1,DEST_EST_SAMPLE_SIZE=30"
COMMON_PRED_VARS+=",STAGE5_DEST_SAMPLE_SIZE=30,GA_MODE_DEGEN_PENALTY=20"
COMMON_PRED_VARS+=",ESTIMATORS_SKIP_UPGRADE=1"

# ── Per-chain extras (replicate each chain's original config) ────────────
#   logsumfix: plain
#   destfix:   no STAGE5_LOGSUM_ORTHOG (matches original)
#   tddsched:  LOGSUM_ORTHOG + tdd stage4 + joint tours
#   accessfix: LOGSUM_ORTHOG, CDAP_HOUSEHOLD_COORD=0
STAGE_EXTRA_LOGSUMFIX="STAGE5_LOGSUM_ORTHOG=1,CDAP_HOUSEHOLD_COORD=0,JOINT_TOURS=0"
STAGE_EXTRA_DESTFIX="CDAP_HOUSEHOLD_COORD=0,JOINT_TOURS=0"
STAGE_EXTRA_TDDSCHED="STAGE5_LOGSUM_ORTHOG=1,STAGE5_DEST_QUANTITY=0,STAGE5_DEST_HIST=1,STAGE4_MANDATORY_LOS=1,STAGE4_TIMING_MODEL=tdd,CDAP_HOUSEHOLD_COORD=1,CDAP_HH_ESTIMATE=1,JOINT_TOURS=1"
STAGE_EXTRA_ACCESSFIX="STAGE5_LOGSUM_ORTHOG=1,CDAP_HOUSEHOLD_COORD=0,JOINT_TOURS=0"

PRED_EXTRA_LOGSUMFIX="STAGE5_LOGSUM_ORTHOG=1,CDAP_HOUSEHOLD_COORD=0,JOINT_TOURS=0"
PRED_EXTRA_DESTFIX="CDAP_HOUSEHOLD_COORD=0,JOINT_TOURS=0"
PRED_EXTRA_TDDSCHED="STAGE5_LOGSUM_ORTHOG=1,STAGE5_DEST_QUANTITY=0,STAGE5_DEST_HIST=1,STAGE4_MANDATORY_LOS=1,STAGE4_TIMING_MODEL=tdd,CDAP_HOUSEHOLD_COORD=1,CDAP_HH_ESTIMATE=1,JOINT_TOURS=1"
PRED_EXTRA_ACCESSFIX="STAGE5_LOGSUM_ORTHOG=1,CDAP_HOUSEHOLD_COORD=0,JOINT_TOURS=0"

PRED_WALLTIME_LOGSUMFIX="36:00:00"
PRED_WALLTIME_DEFAULT="24:00:00"

# ─────────────────────────────────────────────────────────────────────────
submit_chain() {
    local TAG="$1"
    local STAGE_EXTRA="$2"
    local PRED_EXTRA="$3"
    local PRED_WT="$4"

    echo ""
    echo "=== Submitting ${TAG} stage 5 -> merge -> predict ==="

    local jid5
    jid5=$(qsub \
        -N "ga_s5_${TAG}_rerun" \
        -l "select=${S5_SELECT}" \
        -l "walltime=${S5_WALLTIME}" \
        -v "STAGE_ID=5,${COMMON_STAGE_VARS//%TAG%/$TAG},${STAGE_EXTRA}" \
        job_ga_stage_single.pbs | awk '{print $1}')
    echo "  Stage 5 submitted: ${jid5}"

    local jid_merge
    jid_merge=$(qsub \
        -N "ga_merge_${TAG}_rerun" \
        -W "depend=afterok:${jid5}" \
        -l "select=${MERGE_SELECT}" \
        -l "walltime=${MERGE_WALLTIME}" \
        -v "RUN_ROOT=${RUN_ROOT},RUN_TAG=${TAG},CONDA_ENV=${CONDA_ENV},STAGE_IDS=0_1_2_3_4_5,STRICT_MERGE=1" \
        job_merge_ga_results.pbs | awk '{print $1}')
    echo "  Merge submitted:   ${jid_merge}"

    local jid_pred
    jid_pred=$(qsub \
        -N "ga_pred_${TAG}_rerun" \
        -W "depend=afterok:${jid_merge}" \
        -l "select=${PRED_SELECT}" \
        -l "walltime=${PRED_WT}" \
        -v "${COMMON_PRED_VARS//%TAG%/${RUN_ROOT}/${TAG}/merged},${PRED_EXTRA}" \
        pbs_pipeline_mode.pbs | awk '{print $1}')
    echo "  Predict submitted: ${jid_pred} (walltime ${PRED_WT})"

    echo "  Chain: ${jid5} -> ${jid_merge} -> ${jid_pred}"
}

case "${1:-all}" in
    logsumfix) submit_chain logsumfix "$STAGE_EXTRA_LOGSUMFIX" "$PRED_EXTRA_LOGSUMFIX" "$PRED_WALLTIME_LOGSUMFIX" ;;
    destfix)   submit_chain destfix   "$STAGE_EXTRA_DESTFIX"   "$PRED_EXTRA_DESTFIX"   "$PRED_WALLTIME_DEFAULT" ;;
    tddsched)  submit_chain tddsched  "$STAGE_EXTRA_TDDSCHED"  "$PRED_EXTRA_TDDSCHED"  "$PRED_WALLTIME_DEFAULT" ;;
    accessfix) submit_chain accessfix "$STAGE_EXTRA_ACCESSFIX" "$PRED_EXTRA_ACCESSFIX" "$PRED_WALLTIME_DEFAULT" ;;
    all)
        submit_chain logsumfix "$STAGE_EXTRA_LOGSUMFIX" "$PRED_EXTRA_LOGSUMFIX" "$PRED_WALLTIME_LOGSUMFIX"
        submit_chain destfix   "$STAGE_EXTRA_DESTFIX"   "$PRED_EXTRA_DESTFIX"   "$PRED_WALLTIME_DEFAULT"
        submit_chain tddsched  "$STAGE_EXTRA_TDDSCHED"  "$PRED_EXTRA_TDDSCHED"  "$PRED_WALLTIME_DEFAULT"
        submit_chain accessfix "$STAGE_EXTRA_ACCESSFIX" "$PRED_EXTRA_ACCESSFIX" "$PRED_WALLTIME_DEFAULT"
        ;;
    *) echo "Usage: $0 [logsumfix|destfix|tddsched|accessfix|all]" >&2; exit 1 ;;
esac

echo ""
echo "Done. Current queue:"
qstat -u "$(whoami)" || true