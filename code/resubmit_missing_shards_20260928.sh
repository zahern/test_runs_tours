#!/bin/bash -l
# =============================================================================
# Resubmit the 10 missing shards of RUN_TAG=cdap_s5_refit_20260928, then merge.
#
# Diagnosis (see .o/.e logs of 28-29/09):
#   ga_s0all    (Stage 0, all)      -> PBS walltime kill (43204s > 43200s limit).
#                                      NOT a crash: needs 24h, not 12h.
#   ga_s1nonmand (Stage 1, nonmand)  -> same walltime kill, mid-search.
#   ga_s51* x8  (Stage 51, all puropose segments) -> cgroup OOM-killed.
#                                      Cause: GA_JAX_CACHE_MEM_LIMIT_GB=32 on a
#                                      32GB node leaves nothing for data.
#                                      Fix: 64GB node + 16GB cache cap.
#                                      (If 64GB is unavailable, use mem=48GB
#                                      with GA_JAX_CACHE_MEM_LIMIT_GB=12.)
#
# Consistency: same RUN_TAG reuses the 20 completed shards. USE_OVERTURE_LANDUSE=0
# is pinned via EXTRA_ENV because the tag's other shards ran with the flag OFF
# (the chain default flipped to 1 AFTER this tag ran) — mixing would silently
# corrupt the merge. The Overture arm belongs in a NEW tag, not this one.
#
# Usage (from test_runs_tours/code on the submit host):
#   bash resubmit_missing_shards_20260928.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

RUN_TAG=cdap_s5_refit_20260928
CONDA_ENV=zigenv

# Base -v payload, verbatim from the 28/09 submissions, plus the overture pin.
# GA_WALLTIME_SECONDS is set per job below (must stay UNDER the PBS walltime
# so the search exits gracefully and checkpoints instead of being killed).
BASE_V="RUN_ROOT=runs/ga_parallel,RUN_TAG=${RUN_TAG},CONDA_ENV=${CONDA_ENV},GA_BUDGET=standard,GA_ESTIMATOR=metacount_mv,GA_ENGINE=agds,GA_N_RESTARTS=0,GA_PARALLEL_SEGMENTS=1,GA_SHARD_VARIANTS=1,GA_S4_VARIANTS=rrm;tdd,GA_STAGE4_MXL_SEARCH=0,GA_STAGE4_GENS=,GA_STAGE4_RESTARTS=,GA_STAGE4_EARLY_STOP=1,GA_STAGE4_EARLY_STOP_PATIENCE=120,GA_STAGE4_EARLY_STOP_MIN_GENS=120,GA_SA2_GENS=,GA_SA2_RESTARTS=,STAGE2_MH=0,STAGE2_MH_BURN=1000,STAGE2_MH_DRAWS=2000,STAGE2_MH_THIN=2,STAGE2_MH_SEED=42,STAGE2_MH_PROPOSAL_SCALE=1.0,GA_S5_MODE_VARIANTS=all,GA_S5_DEST_VARIANTS=all,GA_STAGE5_GENS=2,GA_STAGE5_RESTARTS=0,GA_STAGE5_STAG=,GA_WALLTIME_MARGIN_SECONDS=900,GA_STAGE5_RUN_MODE=1,GA_STAGE5_RUN_DEST=1,GA_SEED=20260928,GA_L1_L2_ALPHA=1e-3,STAGE5B_CA_FULLGRID_CASES=1500,CDAP_MXL=mh,CDAP_MXL_DRAWS=50,CDAP_MXL_MAX_N=2000,CDAP_MH_BURN=1000,CDAP_MH_DRAWS=2000,CDAP_MH_THIN=5,CDAP_MH_SEED=42,CDAP_MH_PRIOR_SD=5.0,NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,PIPELINE_SKIMS_PATH=../demand_/skims_371.omx,EXTRA_ENV=STAGE5_ENABLE_MXL=1;STAGE5_ENABLE_GSE=1;GA_SEARCH_GSE=1;CDAP_HOUSEHOLD_COORD=1;CDAP_HH_ESTIMATE=1;JOINT_TOURS=1;USE_OVERTURE_LANDUSE=0,STAGE5_DEST_SAMPLE_SIZE=20"

IDS=""
submit_shard() {  # $1=name $2=stage $3=segment $4=resources $5=pbs_walltime $6=ga_walltime $7=jaxcache
  _id=$(qsub -P FPAXJW8067 -N "$1" -l "$4" -l "walltime=$5" \
    -v "$BASE_V,STAGE_ID=$2,SHARD_SEGMENT=$3,GA_WALLTIME_SECONDS=$6,GA_JAX_CACHE_MEM_LIMIT_GB=$7,GA_JAX_CACHE_CHECK_EVERY=10" \
    job_ga_shard_single.pbs)
  echo "submitted $1 -> ${_id} (stage=$2 seg=$3)"
  IDS="${IDS}:${_id%%.*}"
}

# --- walltime victims: same footprint, 24h walltime, 23h graceful budget ---
submit_shard ga_s0all_retry    0  all     "select=1:ncpus=4:mem=48GB" 24:00:00 82800 48
submit_shard ga_s1nonmand_retry 1 nonmand "select=1:ncpus=2:mem=32GB" 24:00:00 82800 32

# --- OOM victims: 64GB node, cache capped at 16GB, 12h walltime ---
for seg in eat edu escort othdiscr othmaint shopping social work; do
  submit_shard "ga_s51${seg}_retry" 51 "$seg" \
    "select=1:ncpus=4:mem=64GB" 12:00:00 42300 16
done

# --- re-merge everything (afterany: runs even if a retry fails again) + paper ---
_MERGE_V="RUN_ROOT=runs/ga_parallel,RUN_TAG=${RUN_TAG},CONDA_ENV=${CONDA_ENV},STAGE_IDS=0_1_2_3_4_5_51,STRICT_MERGE=0,PUBLISH_PAPER=1"
_MERGE_ID=$(qsub -P FPAXJW8067 -N "ga_shardmerge_${RUN_TAG}_retry" \
  -W "depend=afterany${IDS}" -l select=1:ncpus=1:mem=8GB -l walltime=01:00:00 \
  -v "${_MERGE_V}" job_merge_ga_shards.pbs)
echo "submitted merge -> ${_MERGE_ID} (depends on${IDS})"
echo "DONE. Paper artifacts refresh via publish_paper_artifacts.py in the merge (PUBLISH_PAPER=1)."
