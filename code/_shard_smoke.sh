#!/usr/bin/env bash
cd ~/test_runs_tours/code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zigenv
export PYTHONPATH="$PWD/compat:${PYTHONPATH:-}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export GA_BUDGET=smoke GA_ESTIMATOR=metacount_mv GA_ENGINE=agds GA_N_RESTARTS=0
export GA_PARALLEL_SEGMENTS=1 GA_SHARD_VARIANTS=1
export GA_S4_VARIANTS="rrm;tdd" GA_S5_MODE_VARIANTS="nl;mnl" GA_S5_DEST_VARIANTS="mnl;nl_dist"
export NEW_POP=1 NEW_POP_MAPPED=0 NEW_POP_DIR=../new_pop PIPELINE_SKIMS_PATH=../demand_/skims_371.omx
export JAX_PLATFORMS=cpu PIPELINE_ACCELERATOR=cpu OMP_NUM_THREADS=1 GA_METACOUNT_FORCE_CPU=1 TF_CPP_MIN_LOG_LEVEL=3
python -u -X utf8 _run_ga_shard.py --stage-id 1 --segment work --run-root runs/ga_smoketest --run-tag qld_smoketest
echo "SHARD_SMOKE_EXIT=$?"
