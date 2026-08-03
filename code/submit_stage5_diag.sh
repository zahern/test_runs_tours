#!/bin/bash
# submit_stage5_diag.sh — quick stage-5-ONLY run to surface the dest flat-flow
# [DEST DIAG] line (which CA columns are 1-D / non-float64) and whether the Larch
# dest model now compiles after the float64 coercion, without running the whole
# 6-stage chain. ~15-40 min.
#
# Usage:   bash submit_stage5_diag.sh
# Then:    grep -a "\[DEST DIAG\]\|Larch dest estimation failed\|chosen-in-set" ga_s5diag.o*
set -euo pipefail
cd "$(dirname "$0")"

VARS="STAGE_ID=5,RUN_ROOT=runs/ga_parallel,RUN_TAG=s5diag,CONDA_ENV=zigenv,GA_BUDGET=smoke,GA_ESTIMATOR=metacount_mv,GA_USE_SA=1,GA_USE_BANDIT_SA=0,GA_N_RESTARTS=0,GA_PARALLEL_SEGMENTS=1,PIPELINE_ACCELERATOR=cpu,STAGE5_ENABLE_GSE=1,GA_COMPLEXITY_PENALTY=0.4,GA_SIG_PENALTY_WEIGHT=1.5,GA_SIG_PVAL_THRESH=0.05,NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,PIPELINE_SKIMS_PATH=../demand_/sa2_skims_507_omxspec.omx,STAGE5_DEST_DIAG=1"

qsub -N ga_s5diag -l select=1:ncpus=4:mem=96GB -l walltime=03:00:00 -v "$VARS" job_ga_stage_single.pbs

echo "Submitted ga_s5diag. When it finishes, inspect:"
echo "  grep -a '\[DEST DIAG\]' ga_s5diag.o*                   # 1-D / non-float64 CA columns"
echo "  grep -a 'Larch dest estimation failed' ga_s5diag.o*   # gone = full dest compiles"
echo "  grep -a 'chosen-in-set' ga_s5diag.o*                  # importance-sample invariant"
