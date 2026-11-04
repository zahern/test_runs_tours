#!/bin/bash -l
# Resubmit the missing/killed qld_conv1 shards + final merge.
#
# Run on an aqua LOGIN node from ~/test_runs_tours/code:
#   DRY_RUN=1 bash resubmit_qld_conv1_missing.sh   # print plan only
#   bash resubmit_qld_conv1_missing.sh             # submit for real
#
# Queues 20 shards (segment-precise; completed stage2 x3 + stage4/edu are NOT
# resubmitted):
#   new      : 1/work,edu,nonmand  3/work,edu,nonmand   (never submitted)
#   redo     : 0/all  4/work,nonmand  5/work,edu,nonmand  51/x8  (SIGTERM-killed)
# Resources follow the shard table in job_submit_ga_sharded.pbs, with stage-5
# at 24h (the killed attempt ran 12h against a ~15h 40-gen budget).
# -v values are copied verbatim from the killed s5work job (26665306) so the
# redo continues the identical configuration; the new kill-save handler in
# _run_ga_shard.py now finalises on any future SIGTERM.

set -euo pipefail

RUN_ROOT=${RUN_ROOT:-runs/ga_parallel}
RUN_TAG=${RUN_TAG:-qld_conv1}
CONDA_ENV=${CONDA_ENV:-zigenv}
PROJECT_RPID=${PROJECT_RPID:-FPAXJW8067}
DRY_RUN=${DRY_RUN:-0}
STRICT_MERGE=${STRICT_MERGE:-1}

_P_FLAG=()
if [ -n "${PROJECT_RPID}" ]; then
  _P_FLAG=(-P "${PROJECT_RPID}")
fi

# -- preflight: never double-submit under still-active qld_conv1 jobs ---------
_active=$(qstat -u "$USER" 2>/dev/null | grep -F "${RUN_TAG}" || true)
if [ -n "${_active}" ]; then
  echo "ABORT: active '${RUN_TAG}' jobs still in the system:"
  echo "${_active}"
  echo "Wait for them to finish (or qdel them) before resubmitting."
  exit 1
fi

# -- -v payload (from killed job 26665306; GA_BUDGET falls back to default) ----
GA_BUDGET=${GA_BUDGET:-standard}
COMMON_VARS="RUN_ROOT=${RUN_ROOT},RUN_TAG=${RUN_TAG},CONDA_ENV=${CONDA_ENV},GA_BUDGET=${GA_BUDGET},GA_ESTIMATOR=metacount_mv,GA_ENGINE=agds,GA_N_RESTARTS=0,GA_PARALLEL_SEGMENTS=1,GA_SHARD_VARIANTS=1,GA_S4_VARIANTS=rrm;tdd,GA_WALLTIME_MARGIN_SECONDS=300,GA_L1_L2_ALPHA=1e-3,GA_SEED=42,NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,PIPELINE_SKIMS_PATH=../demand_/skims_371.omx,STAGE5B_CA_FULLGRID_CASES=1500,CDAP_MXL=0,CDAP_MXL_DRAWS=50,CDAP_MXL_MAX_N=2000,CDAP_MH_BURN=1000,CDAP_MH_DRAWS=2000,CDAP_MH_THIN=5,CDAP_MH_SEED=42,CDAP_MH_PRIOR_SD=5.0,GA_STAGE5_GENS=40,GA_STAGE5_RESTARTS=1,GA_STAGE5_STAG=20,STAGE5_LC_GENES=1,STAGE5_DEST_SAMPLE_SIZE=20,GA_S5_MODE_VARIANTS=nl;mnl;lccm,GA_S5_DEST_VARIANTS=mnl;nl_dist,GA_STAGE5_RUN_MODE=1,GA_STAGE5_RUN_DEST=1,STAGE5_LCCM_MAXITER=400,STAGE5_LCCM_METHOD=slsqp,STAGE5_LCCM_CAP=25"

# -- shard table: stage:segment:ncpus:mem:walltime -----------------------------
SHARDS="
0:all:4:48GB:12:00:00
1:work:2:32GB:12:00:00
1:edu:2:32GB:12:00:00
1:nonmand:2:32GB:12:00:00
3:work:2:16GB:12:00:00
3:edu:2:16GB:12:00:00
3:nonmand:2:16GB:12:00:00
4:work:2:8GB:12:00:00
4:nonmand:2:16GB:12:00:00
5:work:4:96GB:24:00:00
5:edu:4:96GB:24:00:00
5:nonmand:4:128GB:24:00:00
51:work:4:96GB:12:00:00
51:edu:4:96GB:12:00:00
51:escort:4:96GB:12:00:00
51:shopping:4:96GB:12:00:00
51:othmaint:4:96GB:12:00:00
51:othdiscr:4:96GB:12:00:00
51:eat:4:96GB:12:00:00
51:social:4:96GB:12:00:00
"

submit_shard() {
  local sid="$1" seg="$2" ncpus="$3" mem="$4" wt="$5"
  local wt_secs mem_gb
  wt_secs=$(echo "${wt}" | awk -F: '{print ($1*3600)+($2*60)+$3}')
  mem_gb=$(echo "${mem}" | grep -oE '[0-9]+' | head -1)
  local cmd=(qsub "${_P_FLAG[@]}" -N "ga_s${sid}${seg}_${RUN_TAG}"
    -l "select=1:ncpus=${ncpus}:mem=${mem}" -l "walltime=${wt}"
    -v "${COMMON_VARS},STAGE_ID=${sid},SHARD_SEGMENT=${seg},GA_WALLTIME_SECONDS=${wt_secs},GA_JAX_CACHE_MEM_LIMIT_GB=${mem_gb},GA_JAX_CACHE_CHECK_EVERY=10,STAGE5_DEST_SAMPLE_SIZE=20"
    job_ga_shard_single.pbs)
  if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY: ${cmd[*]}" >&2
    echo "dry_${sid}_${seg}"
  else
    "${cmd[@]}" | awk '{print $1}'
  fi
}

DEPS=""
while read -r line; do
  [ -z "${line}" ] && continue
  sid=$(echo "$line" | cut -d: -f1)
  seg=$(echo "$line" | cut -d: -f2)
  ncpus=$(echo "$line" | cut -d: -f3)
  mem=$(echo "$line" | cut -d: -f4)
  wt=$(echo "$line" | cut -d: -f5-)
  jid=""
  if jid=$(submit_shard "$sid" "$seg" "$ncpus" "$mem" "$wt"); then
    echo "shard s${sid}/${seg}: ${jid}"
    DEPS="${DEPS}:${jid}"
  else
    echo "shard s${sid}/${seg}: SUBMIT FAILED -- continuing"
  fi
done <<< "${SHARDS}"
DEPS=$(echo "${DEPS}" | sed 's/^://')

if [ -z "${DEPS}" ]; then
  echo "No shards submitted. Nothing to merge."
  exit 0
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY: merge would depend on: ${DEPS}"
  exit 0
fi

jid_merge=$(qsub "${_P_FLAG[@]}" -N "ga_shardmerge_${RUN_TAG}" \
  -W "depend=afterany:${DEPS}" \
  -l "select=1:ncpus=1:mem=8GB" -l "walltime=01:00:00" \
  -v "RUN_ROOT=${RUN_ROOT},RUN_TAG=${RUN_TAG},CONDA_ENV=${CONDA_ENV},STAGE_IDS=0_1_2_3_4_5_51,STRICT_MERGE=${STRICT_MERGE},PUBLISH_PAPER=0" \
  job_merge_ga_shards.pbs | awk '{print $1}')
echo "Submitted merge: ${jid_merge} (afterany ${DEPS})"
echo "Merged artifact will be: ${RUN_ROOT}/${RUN_TAG}/merged/ga_results.pkl"
