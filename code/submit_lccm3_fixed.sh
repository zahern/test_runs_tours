#!/usr/bin/env bash
# submit_lccm3_fixed.sh
# Submit LC sweep with the monkey patch fix applied
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code

TAG=lccm3_fixed
COMMON="RUN_ROOT=runs/ga_parallel,RUN_TAG=${TAG},CONDA_ENV=zigenv,\
GA_BUDGET=smoke,GA_ESTIMATOR=metacount_mv,GA_ENGINE=agds,GA_N_RESTARTS=0,\
GA_PARALLEL_SEGMENTS=1,GA_SHARD_VARIANTS=1,GA_S4_VARIANTS=rrm;tdd,\
GA_STAGE4_MXL_SEARCH=0,GA_STAGE4_EARLY_STOP=1,GA_STAGE4_EARLY_STOP_PATIENCE=120,\
GA_STAGE4_EARLY_STOP_MIN_GENS=120,STAGE2_MH=0,STAGE2_MH_BURN=1000,\
STAGE2_MH_DRAWS=2000,STAGE2_MH_THIN=2,STAGE2_MH_SEED=42,STAGE2_MH_PROPOSAL_SCALE=1.0,\
GA_S5_MODE_VARIANTS=nl;mnl;lccm,GA_S5_DEST_VARIANTS=mnl;nl_dist,\
GA_WALLTIME_MARGIN_SECONDS=300,GA_STAGE5_RUN_MODE=1,GA_STAGE5_RUN_DEST=1,\
GA_L1_L2_ALPHA=1e-3,STAGE5B_CA_FULLGRID_CASES=1500,CDAP_MXL=0,CDAP_MXL_DRAWS=50,\
CDAP_MXL_MAX_N=2000,CDAP_MH_BURN=1000,CDAP_MH_DRAWS=2000,CDAP_MH_THIN=5,\
CDAP_MH_SEED=42,CDAP_MH_PRIOR_SD=5.0,NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,\
PIPELINE_SKIMS_PATH=../demand_/skims_371.omx,STAGE_ID=5,GA_WALLTIME_SECONDS=43200,\
GA_JAX_CACHE_MEM_LIMIT_GB=96,GA_JAX_CACHE_CHECK_EVERY=10,STAGE5_DEST_SAMPLE_SIZE=20,\
STAGE5_LCCM_MAXITER=400,STAGE5_LCCM_METHOD=slsqp,STAGE5_LCCM_CAP=25"

echo "== $(date) submitting LC sweep with fix (RUN_TAG=${TAG}) =="
for SEG in work nonmand edu; do
  qsub -P FPAXJW8067 -N ga_s5${SEG}_${TAG} \
       -l select=1:ncpus=4:mem=96GB -l walltime=12:00:00 \
       -v "${COMMON},SHARD_SEGMENT=${SEG}" job_ga_shard_single.pbs
done

echo "== merge =="
qsub -P FPAXJW8067 -N ga_shardmerge_${TAG} -l select=1:ncpus=1:mem=8GB \
     -l walltime=02:00:00 -v "RUN_ROOT=runs/ga_parallel,RUN_TAG=${TAG}" \
     job_merge_ga_shards.pbs

echo "== queue =="
qstat -u ahernz | grep -E 'lccm3|Job ID|----' || true