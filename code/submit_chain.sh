#!/bin/bash -l
# submit_chain.sh — submit the full GA→prediction→MATSim pipeline on the HPC.
#
# Usage:  cd /hpc-fs.qut.edu.au/ahernz/test_runs_tours/code
#         ./submit_chain.sh [RUN_TAG] [full|smoke]
#
# RUN_TAG defaults to agds_full_<date>.  All other knobs are set below and can
# be overridden via -v on the qsub line by editing this file.
# "smoke" = 4h walltimes + fast budget so jobs backfill and start quickly
# (just to confirm the chain runs; resume driver chains stage 5 forward).
#
# What this wires up (see job_submit_ga_parallel_chain.pbs):
#   * GA_CHECKPOINT_DIR   -> runs/ga_parallel/<TAG>/stage5  (resume on walltime kill)
#   * GA_WALLTIME_SECONDS -> derived from STAGE_WALLTIME (auto, 600s buffer)
#   * Stage-5 auto-resume driver (ga_s5resume_<TAG>) re-submits stage 5 until all
#     3 segments complete, then merge/prediction/MATSim proceed.
#
# IMPORTANT: the chain submits sibling .pbs files by relative name, so run this
# from the code directory.

set -euo pipefail

# cd to wherever this script lives (the code dir), so it works under any mount
# path (e.g. /mnt/hpccs01/home/ahernz/test_runs_tours/code on the HPC, or the
# /hpc-fs.qut.edu.au/... path from Windows). The chain submits sibling .pbs files
# by relative name, so the working dir MUST be the code directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ ! -f job_submit_ga_parallel_chain.pbs ]; then
  echo "ERROR: job_submit_ga_parallel_chain.pbs not found in ${SCRIPT_DIR}" >&2
  echo "       Run this script from inside the code directory (or fix SCRIPT_DIR)." >&2
  exit 1
fi
echo "Working dir: ${SCRIPT_DIR}"

# ── Tunable knobs ─────────────────────────────────────────────────────────────
RUN_TAG="${1:-agds_full_$(date +%Y%m%d)}"
# Optional 2nd arg: "smoke" for a quick scheduling check — short walltimes
# (fit backfill windows so jobs start fast) + light GA budget. The walltime
# guard + resume driver keep chaining stage-5 slots forward afterwards.
MODE="${2:-full}"

GA_BUDGET=standard
GA_ESTIMATOR=metacount_mv
GA_ENGINE=agds

# Stage resources (one walltime slot can't fit all 3 stage-5 segments ~19h each;
# the auto-resume driver chains extra slots up to STAGE5_MAX_RESUMES).
# Sized for ~125GB nodes: the JAX trims (preallocate off, tight cache guard)
# plus dest sample 20 keep stage 5 inside this. Do NOT raise STAGE_MEM above
# what the node offers — it won't schedule.
STAGE_NCPUS=8
STAGE_MEM=128GB
STAGE_WALLTIME=48:00:00
STAGE5_PARALLEL_SEGMENTS=2

# FALLBACK if stage 5 still OOMs (watch its .o "Mem usage" line; if it climbs
# past ~110GB, uncomment the next line): samples 12 dest zones instead of 20,
# cutting another ~40% off dest-side matrices at the cost of noisier fitness.
# STAGE5_DEST_SAMPLE_SIZE=12

# Prediction resources
PRED_NCPUS=8
PRED_MEM=250GB
PRED_WALLTIME=24:00:00

# How many extra walltime slots stage 5 may use to finish all segments.
STAGE5_MAX_RESUMES=4

# Smoke mode (./submit_chain.sh <tag> smoke): short backfill-friendly walltimes
# so jobs start fast, plus a light GA budget. Applied AFTER all defaults above
# so nothing overwrites it. The walltime guard + resume driver keep chaining
# stage-5 slots forward afterwards.
if [ "${MODE}" = "smoke" ]; then
  GA_BUDGET=fast
  STAGE_WALLTIME=04:00:00
  PRED_WALLTIME=04:00:00
fi

# ── Submit ────────────────────────────────────────────────────────────────────
VARS="RUN_TAG=${RUN_TAG}"
VARS="${VARS},GA_BUDGET=${GA_BUDGET}"
VARS="${VARS},GA_ESTIMATOR=${GA_ESTIMATOR}"
VARS="${VARS},GA_ENGINE=${GA_ENGINE}"
VARS="${VARS},STAGE_NCPUS=${STAGE_NCPUS},STAGE_MEM=${STAGE_MEM},STAGE_WALLTIME=${STAGE_WALLTIME}"
VARS="${VARS},STAGE5_PARALLEL_SEGMENTS=${STAGE5_PARALLEL_SEGMENTS}"
VARS="${VARS},PRED_NCPUS=${PRED_NCPUS},PRED_MEM=${PRED_MEM},PRED_WALLTIME=${PRED_WALLTIME}"
VARS="${VARS},STAGE5_MAX_RESUMES=${STAGE5_MAX_RESUMES}"
# Fallback hook: only passed when uncommented above (low-memory dest sampling).
if [ -n "${STAGE5_DEST_SAMPLE_SIZE:-}" ]; then
  VARS="${VARS},STAGE5_DEST_SAMPLE_SIZE=${STAGE5_DEST_SAMPLE_SIZE}"
fi

echo "Submitting chain with:"
echo "  RUN_TAG            = ${RUN_TAG}"
echo "  GA_BUDGET          = ${GA_BUDGET}"
echo "  GA_ESTIMATOR       = ${GA_ESTIMATOR}"
echo "  GA_ENGINE          = ${GA_ENGINE}"
echo "  STAGE (ncpu/mem/wt)= ${STAGE_NCPUS}/${STAGE_MEM}/${STAGE_WALLTIME}"
echo "  STAGE5_PARSEG      = ${STAGE5_PARALLEL_SEGMENTS}"
echo "  PRED  (ncpu/mem/wt)= ${PRED_NCPUS}/${PRED_MEM}/${PRED_WALLTIME}"
echo "  STAGE5_MAX_RESUMES= ${STAGE5_MAX_RESUMES}"
echo "  DEST_SAMPLE      = ${STAGE5_DEST_SAMPLE_SIZE:-20 (default)}"
echo "  (GA_CHECKPOINT_DIR and GA_WALLTIME_SECONDS are derived automatically)"
echo

JOBID=$(qsub -v "${VARS}" job_submit_ga_parallel_chain.pbs | awk '{print $1}')

echo "Submitted orchestrator job: ${JOBID}"
echo "Orchestrator log : ga_s0_${RUN_TAG}.o${JOBID} (and siblings)"
echo "Stage-5 resume  : ga_s5resume_${RUN_TAG}.o*"
echo "Track progress  : qstat -f ${JOBID}"
