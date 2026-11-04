#!/usr/bin/env bash
# Submit the full convergence-evidence rerun set.
#   A. stage 5  (3 segs) - real search budget + LC genes enabled + lccm winner-capable + nl_dist fixed
#   B. stage 51 (8 segs) - real search budget (had generations=1, no leaderboard)
#   C. stage 2/4        - STAGE2_MH=1 + GA_STAGE4_MXL_SEARCH=1 (never run in any prior job)
#   D. stage 0 merge    - re-merge the good shard into merged/ (was empty/sentinel)
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
set -u
TAG=qld_conv1
P=FPAXJW8067

BASE="RUN_ROOT=runs/ga_parallel,RUN_TAG=${TAG},CONDA_ENV=zigenv,\
GA_ESTIMATOR=metacount_mv,GA_ENGINE=agds,GA_PARALLEL_SEGMENTS=1,GA_SHARD_VARIANTS=1,\
GA_WALLTIME_MARGIN_SECONDS=300,GA_L1_L2_ALPHA=1e-3,GA_SEED=42,\
NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,\
PIPELINE_SKIMS_PATH=../demand_/skims_371.omx,\
STAGE5B_CA_FULLGRID_CASES=1500,CDAP_MXL_DRAWS=50,CDAP_MXL_MAX_N=2000,\
CDAP_MH_BURN=1000,CDAP_MH_DRAWS=2000,CDAP_MH_THIN=5,CDAP_MH_SEED=42,CDAP_MH_PRIOR_SD=5.0"

echo "=============================================="
echo "A. STAGE 5  - gens=40 restarts=1 stag=20 LC_GENES=1"
echo "=============================================="
S5="$BASE,GA_STAGE5_GENS=40,GA_STAGE5_RESTARTS=1,GA_STAGE5_STAG=20,\
STAGE5_LC_GENES=1,STAGE5_DEST_SAMPLE_SIZE=20,\
GA_S5_MODE_VARIANTS=nl;mnl;lccm,GA_S5_DEST_VARIANTS=mnl;nl_dist,\
GA_STAGE5_RUN_MODE=1,GA_STAGE5_RUN_DEST=1,\
STAGE5_LCCM_MAXITER=400,STAGE5_LCCM_METHOD=slsqp,STAGE5_LCCM_CAP=25"
for SEG in work nonmand edu; do
  qsub -P $P -N ga_s5${SEG}_${TAG} -l select=1:ncpus=4:mem=96GB \
       -l walltime=12:00:00 -v "${S5},STAGE_ID=5,SHARD_SEGMENT=${SEG}" \
       job_ga_shard_single.pbs
done

echo
echo "=============================================="
echo "B. STAGE 51 - gens=40 restarts=1  (8 purposes)"
echo "=============================================="
S51="$BASE,GA_STAGE51_GENS=40,GA_STAGE51_RESTARTS=1,CDAP_MXL=0"
for SEG in work edu escort shopping othmaint othdiscr eat social; do
  qsub -P $P -N ga_s51${SEG}_${TAG} -l select=1:ncpus=4:mem=96GB \
       -l walltime=12:00:00 -v "${S51},STAGE_ID=51,SHARD_SEGMENT=${SEG}" \
       job_ga_shard_single.pbs
done

echo
echo "=============================================="
echo "C. STAGE 2 + STAGE 4 - samplers ON"
echo "=============================================="
S24="$BASE,CDAP_MXL=0,GA_STAGE4_EARLY_STOP=1,GA_STAGE4_EARLY_STOP_PATIENCE=120,\
GA_STAGE4_EARLY_STOP_MIN_GENS=120,GA_S4_VARIANTS=rrm;tdd"
# stage 2: STAGE2_MH=1 (never run in any prior job)
S2="$S24,STAGE2_MH=1,STAGE2_MH_BURN=1000,STAGE2_MH_DRAWS=2000,STAGE2_MH_THIN=2,\
STAGE2_MH_SEED=42,STAGE2_MH_PROPOSAL_SCALE=1.0"
for SEG in start end budget; do
  qsub -P $P -N ga_s2${SEG}_${TAG} -l select=1:ncpus=4:mem=32GB \
       -l walltime=12:00:00 -v "${S2},STAGE_ID=2,SHARD_SEGMENT=${SEG}" \
       job_ga_shard_single.pbs
done
# stage 4: GA_STAGE4_MXL_SEARCH=1 (never run in any prior job)
S4="$S24,GA_STAGE4_MXL_SEARCH=1,GA_STAGE4_GENS=196,GA_STAGE4_RESTARTS=0,STAGE2_MH=0"
for SEG in work edu nonmand; do
  qsub -P $P -N ga_s4${SEG}_${TAG} -l select=1:ncpus=4:mem=64GB \
       -l walltime=12:00:00 -v "${S4},STAGE_ID=4,SHARD_SEGMENT=${SEG}" \
       job_ga_shard_single.pbs
done

echo
echo "=============================================="
echo "D. STAGE 0 merge - re-merge the good shard"
echo "=============================================="
qsub -P $P -N ga_merge_s0 -l select=1:ncpus=1:mem=8GB -l walltime=02:00:00 \
     -v "RUN_ROOT=runs/ga_parallel,RUN_TAG=cdap_s5_refit_20260928,STAGE_IDS=0" \
     job_merge_ga_shards.pbs

echo
echo "=============================================="
qstat -u ahernz 2>/dev/null | tail -25