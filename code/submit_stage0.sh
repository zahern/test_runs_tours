#!/usr/bin/env bash
# Stage 0 must be RE-RUN, not just re-merged: its ga_results.pkl is an empty
# AllStagesResult (all fields None), and merge_ga_shards.py reads only that
# file - hence "merged 0 segment(s)" + the -1e9 sentinel row.
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
TAG=qld_conv1
P=FPAXJW8067

BASE="RUN_ROOT=runs/ga_parallel,RUN_TAG=${TAG},CONDA_ENV=zigenv,\
GA_ESTIMATOR=metacount_mv,GA_ENGINE=agds,GA_PARALLEL_SEGMENTS=1,GA_SHARD_VARIANTS=1,\
GA_WALLTIME_MARGIN_SECONDS=300,GA_L1_L2_ALPHA=1e-3,GA_SEED=42,\
NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,\
PIPELINE_SKIMS_PATH=../demand_/skims_371.omx,CDAP_MXL=0,STAGE2_MH=0"

# 72 gens matched the prior good stage-0 shard's converged trace.
S0="$BASE,GA_STAGE0_GENS=72,GA_STAGE0_RESTARTS=0"
qsub -P $P -N ga_s0all_${TAG} -l select=1:ncpus=4:mem=64GB -l walltime=12:00:00 \
     -v "${S0},STAGE_ID=0,SHARD_SEGMENT=all" job_ga_shard_single.pbs

echo "--- submitted stage 0; state: ---"
qstat -u ahernz 2>/dev/null | grep -E "ga_s0all|Job ID|----"