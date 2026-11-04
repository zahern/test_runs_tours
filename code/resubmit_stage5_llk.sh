#!/usr/bin/env bash
# Stage-5 shards were started BEFORE the ll/k fix landed, so they would write
# leaderboards with blank ll/k (the lccm3_fixed staleness trap). They are only
# minutes into a 12h budget, so cancel + resubmit so all three segments run
# identical code. Stage 2/4/51 are untouched by this change (s4 has its own
# refit path; 51 uses stage5b), so they keep running.
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
set -u
TAG=qld_conv1
P=FPAXJW8067

echo "=== compile check ==="
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
python -m py_compile ga_hypothesis_testing.py ga_variant_refits.py ga_shard.py \
  && echo "COMPILE OK" || { echo "COMPILE FAILED - aborting"; exit 1; }

echo "=== cancel stale stage5 shards ==="
for j in 26665306 26665307 26665309; do qdel $j 2>&1 || true; done
sleep 8

echo "=== clear any partial stage5 artifacts for this tag ==="
rm -rf runs/ga_parallel/${TAG}/stage5/shards/*/variants 2>/dev/null || true

echo "=== resubmit stage5 with the ll/k fix ==="
S5="RUN_ROOT=runs/ga_parallel,RUN_TAG=${TAG},CONDA_ENV=zigenv,\
GA_ESTIMATOR=metacount_mv,GA_ENGINE=agds,GA_PARALLEL_SEGMENTS=1,GA_SHARD_VARIANTS=1,\
GA_WALLTIME_MARGIN_SECONDS=300,GA_L1_L2_ALPHA=1e-3,GA_SEED=42,\
NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,\
PIPELINE_SKIMS_PATH=../demand_/skims_371.omx,\
STAGE5B_CA_FULLGRID_CASES=1500,CDAP_MXL_DRAWS=50,CDAP_MXL_MAX_N=2000,\
CDAP_MH_BURN=1000,CDAP_MH_DRAWS=2000,CDAP_MH_THIN=5,CDAP_MH_SEED=42,CDAP_MH_PRIOR_SD=5.0,\
GA_STAGE5_GENS=40,GA_STAGE5_RESTARTS=1,GA_STAGE5_STAG=20,STAGE5_LC_GENES=1,\
STAGE5_DEST_SAMPLE_SIZE=20,GA_S5_MODE_VARIANTS=nl;mnl;lccm,GA_S5_DEST_VARIANTS=mnl;nl_dist,\
GA_STAGE5_RUN_MODE=1,GA_STAGE5_RUN_DEST=1,\
STAGE5_LCCM_MAXITER=400,STAGE5_LCCM_METHOD=slsqp,STAGE5_LCCM_CAP=25"
for SEG in work nonmand edu; do
  qsub -P $P -N ga_s5${SEG}_${TAG} -l select=1:ncpus=4:mem=96GB \
       -l walltime=12:00:00 -v "${S5},STAGE_ID=5,SHARD_SEGMENT=${SEG}" \
       job_ga_shard_single.pbs
done
echo
qstat -u ahernz 2>/dev/null | grep -E "ga_s5|Job ID|----"