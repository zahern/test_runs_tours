#!/bin/bash
# Fix qld_shard2 missing segments:
#   s4 escort/shopping/social -> walltime killed (12h, Exit -29, Gen29 done but killed before write)
#   s5 work/nonmand          -> OOM killed (Exit 137, cgroup/OOM)
# Run from login node: bash resubmit_qld_shard2_fix.sh
set -euo pipefail
cd "$(dirname "$0")"

COMMON="RUN_ROOT=runs/ga_parallel,RUN_TAG=qld_shard2,CONDA_ENV=zigenv,GA_BUDGET=ultrafast,GA_ESTIMATOR=metacount_mv,GA_ENGINE=agds,GA_N_RESTARTS=0,GA_PARALLEL_SEGMENTS=1,GA_SHARD_VARIANTS=1,GA_S4_VARIANTS=rrm;tdd,GA_S5_MODE_VARIANTS=nl;mnl,GA_S5_DEST_VARIANTS=mnl;nl_dist,NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,PIPELINE_SKIMS_PATH=../demand_/skims_371.omx"

echo "=== resubmitting 3x s4 with 24h walltime ==="
J_S4ESC=$(qsub -P FPAXJW8067 -N ga_s4escort_qld_shard2_fix -l select=1:ncpus=2:mem=16GB -l place=excl -l walltime=24:00:00 -v "${COMMON},STAGE_ID=4,SHARD_SEGMENT=escort,GA_WALLTIME_SECONDS=82800" job_ga_shard_single.pbs | awk '{print $1}')
echo "escort: $J_S4ESC"
J_S4SHO=$(qsub -P FPAXJW8067 -N ga_s4shopping_qld_shard2_fix -l select=1:ncpus=2:mem=16GB -l place=excl -l walltime=24:00:00 -v "${COMMON},STAGE_ID=4,SHARD_SEGMENT=shopping,GA_WALLTIME_SECONDS=82800" job_ga_shard_single.pbs | awk '{print $1}')
echo "shopping: $J_S4SHO"
J_S4SOC=$(qsub -P FPAXJW8067 -N ga_s4social_qld_shard2_fix -l select=1:ncpus=2:mem=16GB -l place=excl -l walltime=24:00:00 -v "${COMMON},STAGE_ID=4,SHARD_SEGMENT=social,GA_WALLTIME_SECONDS=82800" job_ga_shard_single.pbs | awk '{print $1}')
echo "social: $J_S4SOC"

echo "=== resubmitting 2x s5 with bigger mem ==="
J_S5W=$(qsub -P FPAXJW8067 -N ga_s5work_qld_shard2_fix -l select=1:ncpus=4:mem=128GB -l place=excl -l walltime=12:00:00 -v "${COMMON},STAGE_ID=5,SHARD_SEGMENT=work,GA_WALLTIME_SECONDS=39600" job_ga_shard_single.pbs | awk '{print $1}')
echo "s5/work: $J_S5W"
J_S5N=$(qsub -P FPAXJW8067 -N ga_s5nonmand_qld_shard2_fix -l select=1:ncpus=4:mem=160GB -l place=excl -l walltime=12:00:00 -v "${COMMON},STAGE_ID=5,SHARD_SEGMENT=nonmand,GA_WALLTIME_SECONDS=39600" job_ga_shard_single.pbs | awk '{print $1}')
echo "s5/nonmand: $J_S5N"

DEPS="${J_S4ESC}:${J_S4SHO}:${J_S4SOC}:${J_S5W}:${J_S5N}"
echo "=== merge after fix (afterok ${DEPS}) ==="
J_MERGE=$(qsub -P FPAXJW8067 -N ga_shardmerge_qld_shard2_fix -W depend=afterok:${DEPS} -l select=1:ncpus=1:mem=8GB -l walltime=01:00:00 -v "RUN_ROOT=runs/ga_parallel,RUN_TAG=qld_shard2,CONDA_ENV=zigenv,STAGE_IDS=0_1_2_3_4_5,STRICT_MERGE=0,PUBLISH_PAPER=1" job_merge_ga_shards.pbs | awk '{print $1}')
echo "merge: $J_MERGE"
echo "Chain: ${DEPS} -> ${J_MERGE}"
qstat -u "$(whoami)" || true
