#!/bin/bash -l
# Re-run ONLY the qld_conv1 stage-1 nonmand shard so it produces a real
# ga_results.pkl (the 12/10 attempt was SIGKILLed at the 12 h walltime mid-GA,
# leaving only best_Stage1___Non_Mandatory__combined_.npz).
#
# Run on an aqua LOGIN node from ~/test_runs_tours/code:
#   DRY_RUN=1 bash resubmit_stage1_nonmand_only.sh   # print the qsub line
#   bash resubmit_stage1_nonmand_only.sh             # submit
#
# Why the budget knobs: the killed job used GA_BUDGET=standard (pop=60,
# gens=263) and GA_STAGE123_GENS_MULT defaulted to 2.0 -> 526 generations at
# ~14 min/gen (~130 h of work) inside a 12 h limit. It had already found its
# best (BIC/n=2.7598, 23 features) by generation 9 and then stagnated for 38
# generations. The re-run keeps pop=60 and GA_SEED=42 but caps the search and
# stops on stagnation:
#   GA_STAGE123_GENS_MULT=0.12  -> ~31 main generations (~7.4 h worst case)
#   GA_EARLY_STOP_PATIENCE=20   -> stop ~gen 29 if still at 2.7598 (~6.9 h)
#   GA_EARLY_STOP_MIN_GENS=16   -> keeps the exploration floor
# Worst case stays well inside the 12 h walltime, and the winning feature set
# (found by generation 9 in the killed run) is not at risk.
#
# Everything else is copied verbatim from the killed job's Variable_List
# (26674786.aqua) so the run continues the identical conv1 configuration.

set -euo pipefail

cd "$(dirname "$0")"

RUN_ROOT=${RUN_ROOT:-runs/ga_parallel}
RUN_TAG=${RUN_TAG:-qld_conv1}
CONDA_ENV=${CONDA_ENV:-zigenv}
PROJECT_RPID=${PROJECT_RPID:-FPAXJW8067}
DRY_RUN=${DRY_RUN:-0}
STAGE_ID=1
SEG=nonmand
NCPUS=${NCPUS:-2}
MEM=${MEM:-32GB}
WALLTIME=${WALLTIME:-12:00:00}

GA_STAGE123_GENS_MULT=${GA_STAGE123_GENS_MULT:-0.12}
GA_EARLY_STOP=${GA_EARLY_STOP:-1}
GA_EARLY_STOP_PATIENCE=${GA_EARLY_STOP_PATIENCE:-20}
GA_EARLY_STOP_MIN_GENS=${GA_EARLY_STOP_MIN_GENS:-16}

# -- preflight ---------------------------------------------------------------
if qstat -u "$USER" 2>/dev/null | grep -qE "ga_s${STAGE_ID}${SEG}_${RUN_TAG}|${RUN_TAG}"; then
  echo "ABORT: a '${RUN_TAG}' job is still queued/running:"
  qstat -u "$USER" 2>/dev/null | grep -E "ga_s${STAGE_ID}${SEG}_${RUN_TAG}|${RUN_TAG}"
  echo "Wait for it to finish (or qdel it) before resubmitting."
  exit 1
fi
if [ -f "${RUN_ROOT}/${RUN_TAG}/stage${STAGE_ID}/shards/${SEG}/ga_results.pkl" ]; then
  echo "ABORT: ${RUN_ROOT}/${RUN_TAG}/stage${STAGE_ID}/shards/${SEG}/ga_results.pkl"
  echo "already exists - nothing to re-run (delete it first to force a rerun)."
  exit 1
fi

# -- -v payload (verbatim from 26674786.aqua, plus the early-stop knobs) ------
COMMON_VARS="RUN_ROOT=${RUN_ROOT},RUN_TAG=${RUN_TAG},CONDA_ENV=${CONDA_ENV}"
COMMON_VARS="${COMMON_VARS},GA_BUDGET=standard,GA_ESTIMATOR=metacount_mv,GA_ENGINE=agds"
COMMON_VARS="${COMMON_VARS},GA_N_RESTARTS=0,GA_PARALLEL_SEGMENTS=1,GA_SHARD_VARIANTS=1"
COMMON_VARS="${COMMON_VARS},GA_S4_VARIANTS=rrm;tdd,GA_WALLTIME_MARGIN_SECONDS=300"
COMMON_VARS="${COMMON_VARS},GA_L1_L2_ALPHA=1e-3,GA_SEED=42"
COMMON_VARS="${COMMON_VARS},NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop"
COMMON_VARS="${COMMON_VARS},PIPELINE_SKIMS_PATH=../demand_/skims_371.omx"
COMMON_VARS="${COMMON_VARS},STAGE5B_CA_FULLGRID_CASES=1500"
COMMON_VARS="${COMMON_VARS},CDAP_MXL=0,CDAP_MXL_DRAWS=50,CDAP_MXL_MAX_N=2000"
COMMON_VARS="${COMMON_VARS},CDAP_MH_BURN=1000,CDAP_MH_DRAWS=2000,CDAP_MH_THIN=5"
COMMON_VARS="${COMMON_VARS},CDAP_MH_SEED=42,CDAP_MH_PRIOR_SD=5.0"
COMMON_VARS="${COMMON_VARS},GA_STAGE5_GENS=40,GA_STAGE5_RESTARTS=1,GA_STAGE5_STAG=20"
COMMON_VARS="${COMMON_VARS},STAGE5_LC_GENES=1,STAGE5_DEST_SAMPLE_SIZE=20"
COMMON_VARS="${COMMON_VARS},GA_S5_MODE_VARIANTS=nl;mnl;lccm"
COMMON_VARS="${COMMON_VARS},GA_S5_DEST_VARIANTS=mnl;nl_dist"
COMMON_VARS="${COMMON_VARS},GA_STAGE5_RUN_MODE=1,GA_STAGE5_RUN_DEST=1"
COMMON_VARS="${COMMON_VARS},STAGE5_LCCM_MAXITER=400,STAGE5_LCCM_METHOD=slsqp"
COMMON_VARS="${COMMON_VARS},STAGE5_LCCM_CAP=25"
COMMON_VARS="${COMMON_VARS},GA_STAGE123_GENS_MULT=${GA_STAGE123_GENS_MULT}"
COMMON_VARS="${COMMON_VARS},GA_EARLY_STOP=${GA_EARLY_STOP}"
COMMON_VARS="${COMMON_VARS},GA_EARLY_STOP_PATIENCE=${GA_EARLY_STOP_PATIENCE}"
COMMON_VARS="${COMMON_VARS},GA_EARLY_STOP_MIN_GENS=${GA_EARLY_STOP_MIN_GENS}"
COMMON_VARS="${COMMON_VARS},STAGE_ID=${STAGE_ID},SHARD_SEGMENT=${SEG}"
COMMON_VARS="${COMMON_VARS},GA_WALLTIME_SECONDS=$(( ${WALLTIME%%:*} * 3600 ))"
COMMON_VARS="${COMMON_VARS},GA_JAX_CACHE_MEM_LIMIT_GB=32,GA_JAX_CACHE_CHECK_EVERY=10"

CMD=(qsub -P "${PROJECT_RPID}" -N "ga_s${STAGE_ID}${SEG}_${RUN_TAG}"
     -l "select=1:ncpus=${NCPUS}:mem=${MEM}" -l "walltime=${WALLTIME}"
     -v "${COMMON_VARS}" job_ga_shard_single.pbs)

echo "stage ${STAGE_ID}/${SEG}  pop=60 gens~$(( 263 * ${GA_STAGE123_GENS_MULT%.*} ))  " \
     "early_stop=patience ${GA_EARLY_STOP_PATIENCE}/min ${GA_EARLY_STOP_MIN_GENS}"
echo "ga_results.pkl -> ${RUN_ROOT}/${RUN_TAG}/stage${STAGE_ID}/shards/${SEG}/ga_results.pkl"

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY: ${CMD[*]}"
  exit 0
fi

JID=$("${CMD[@]}")
echo "submitted: ${JID}"
sleep 5
echo "--- verify env landed ---"
qstat -f "${JID%%.*}" 2>/dev/null \
  | sed -n '/Variable_List/,/^\s*[A-Za-z_]* =/p' | tr -d '\t' | tr ',' '\n' \
  | grep -E "GA_BUDGET|GA_STAGE123_GENS_MULT|GA_EARLY_STOP|GA_SEED|STAGE_ID|SHARD_SEGMENT" \
  | sed 's/^ *//' || true
