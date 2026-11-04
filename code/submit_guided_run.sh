#!/bin/bash
# =============================================================================
# submit_guided_run.sh — next guided verification run (2 chained PBS jobs)
# =============================================================================
# Purpose: exercise EVERY fix from the 2026-07 stability work in one traceable
# chain, with a fresh constraint-guided SA-bandit search (not reused results):
#
#   Job A  (search, 64 GB)   MODE=ga         → runs/ga_parallel/sa_bandit__<TAG>
#   Job B  (apply, 250 GB)   MODE=ga_staged  → runs/ga_parallel/sa_bandit_staged__<TAG>
#          depends afterok on Job A; reuses A's ga_results.pkl + stage5b_models.pkl
#
# What this chain verifies (grep the .o files for these):
#   1. Constraint-guided search        → "[constraints:stage" lines from
#      ga_constraints.searcher_constraint_kwargs (fixed/banned/mutex/require-one)
#   2. Estimators from PyPI latest     → "[pkg] MetaCount + SearchLibrium
#      upgraded (PyPI latest)" (ESTIMATOR_SOURCE=pypi; metacountregressor
#      >=1.0.107, SearchLibrium >=0.0.154 — no longer the stale 0.0.138 editable)
#   3. Random params as a decision     → "greedy random-param search:" lines
#      (GA_RAND_PARAM_SEARCH=1) and stage-5 MXL policy lines
#   4. Zero-gradient guard, both ways  → "_estimate ... retry done: LL=..."
#      should appear where needed and LL must improve past the initial LL;
#      the nonmand mode NL (numba-stuck last run) now retries on jax
#   5. MXL enabled + all-stuck→NL guard→ STAGE5_ENABLE_MXL=1; look for either a
#      genuine MXL fit or "MXL: ALL ... params unmoved ... falling back to NL"
#   6. Full-zone calibration           → "rebuilt full-zone calibration tree"
#      and shadow prices with remaining_fail_zones in single digits
#   7. Stage 5b end-to-end             → "reusing fitted models ... predict-only"
#      with NO KeyError; 'home' return legs excluded from stop trips
#   8. Stage 6 bounded repair + export → "Repair loop ... ABORTED" (by design,
#      ~55 s/chunk) then "Saved: vehicle.dat"
#   9. SA1 walk/bike skims             → "[pkg] h5py installed" and NO
#      "No module named 'h5py'" warnings
#  10. Slide auto-refresh              → "[sa2-slide] deck updated" at job end
#
# Usage (from ~/test_runs_tours/code on an aqua login node):
#   bash submit_guided_run.sh              # TAG defaults to guided01
#   TAG=guided02 bash submit_guided_run.sh # later iterations
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-guided01}"
GA_DIR="runs/ga_parallel/sa_bandit__${TAG}"

# Environment shared by both jobs. Differences from the July reruns:
#   GA_RAND_PARAM_SEARCH=1  greedy random-vs-fixed BIC toggle (stage 4 MXL)
#   STAGE5_ENABLE_MXL=1     exercise the stage-5 mixed logit + its
#                           all-stuck → NL fallback guard
COMMON="CONDA_ENV=zigenv,RUN_ROOT=runs/ga_parallel,THREADS=1,S5_WORKERS=8"
# GA_BUDGET fast -> standard (100 -> 350 generation-equivalents): the candidate
# pools were widened 2026-07-16 (origin accessibility + micromobility for mode;
# full 8-measure accessibility + extra OSM landuse columns for destination) so
# the search needs a longer run to explore the bigger space. ~2-3x the fast
# walltime; 24h walltime still comfortably covers it.
COMMON+=",GA_ESTIMATOR=sa_bandit,GA_USE_SA=1,GA_USE_BANDIT_SA=1,GA_BUDGET=standard"
# Always take estimators from PyPI latest (metacountregressor>=1.0.107,
# SearchLibrium>=0.0.154) — explicit ESTIMATOR_SOURCE=pypi overrides the
# bandit-SA editable default. If a run ever needs the local dev trees instead,
# change this to editable.
COMMON+=",ESTIMATOR_SOURCE=pypi"
COMMON+=",INSTALL_ESTIMATORS=1,PIPELINE_ACCELERATOR=cpu,GA_N_RESTARTS=0"
COMMON+=",GA_SIG_PENALTY_WEIGHT=1.50,GA_SIG_PVAL_THRESH=0.05"
# Search-quality defaults (aligned with job_submit_ga_parallel_chain.pbs):
# looser complexity penalty keeps behavioural terms, more gens for the cheap
# stages 1-3, GSE gradient loadings on the stage-5 mode model.
COMMON+=",GA_COMPLEXITY_PENALTY=0.4,GA_STAGE123_GENS_MULT=2.0,STAGE5_ENABLE_GSE=1"
# 8 → 4 workers + per-worker JAX cache budget. Job 23926202 OOM'd at 202 GB /
# 194 GB in Stage 1: _maybe_clear_jax_caches checks EACH worker's own RSS
# against 80% of GA_JAX_CACHE_MEM_LIMIT_GB (which the PBS script defaulted to
# 200 GB), so 8 workers could collectively hit the cgroup limit while every
# one of them individually stayed "under budget". 4 workers × 38 GB (+ main
# process) fits 194 GB with headroom; Stage 1 only has 3 segments anyway, so
# the concurrency loss is small.
COMMON+=",GA_PARALLEL_SEGMENTS=4"
COMMON+=",GA_JAX_CACHE_MEM_LIMIT_GB=38,GA_JAX_CACHE_CHECK_EVERY=50"
COMMON+=",GA_RAND_PARAM_SEARCH=1,STAGE5_ENABLE_MXL=1"
# NEW_POP=1: full-SEQ SA2-direct scope (371 zones incl. Tweed/NSW/Toowoomba,
# skims skims_371.omx with CRASH, landuse zone_landuse_507.csv).
# The legacy 330-zone ORDER scope (NEW_POP=0) predates the 2026-07 zone work.
COMMON+=",NEW_POP=1,NEW_POP_MAPPED=0,NEW_POP_DIR=../new_pop,PIPELINE_SKIMS_PATH="
COMMON+=",PIPELINE_RUN_TAG=${TAG}"

# ── Job A: fresh constraint-guided SA-bandit search (HHTS only) ────────────
# MODE=ga estimates on HHTS and does NOT load the 3.7M synthetic population.
# mem=64GB was tried and OOM-killed one of the 8 GA_PARALLEL_SEGMENTS workers
# mid-Stage-1 (job 23762949: mem usage hit 66569884kb against the 64GB cap,
# BrokenProcessPool). 8 concurrent bandit-SA workers each hold a growing
# eval cache; job_sa_bandit.pbs/job_sa_bandit_507.pbs budget 194GB for just 4
# such workers, so match that headroom here for 8. GA checkpoints roll into
# ga_results.pkl as it goes (stage-level resume already skips completed
# Stage 0 via ga_results_checkpoint.pkl), so a walltime/OOM kill loses at
# most the in-flight stage.
# RESUME_AFTER=<jobid>: chain this Job A behind an already-running search job
# with afterany (fires on ANY exit, incl. walltime kill), so the resume starts
# the moment the current attempt dies. Stage-level resume via
# ga_results_checkpoint.pkl skips completed stages, so nothing is recomputed.
#   CUR=$(qstat -u $(whoami) | awk '/ga_search_guided03/{print $1}')
#   TAG=guided03 RESUME_AFTER=$CUR bash submit_guided_run.sh
_DEPEND_A=()
if [ -n "${RESUME_AFTER:-}" ]; then
    _DEPEND_A=(-W "depend=afterany:${RESUME_AFTER}")
    echo "Job A will start after job ${RESUME_AFTER} ends (afterany)"
fi
JOB_A=$(qsub -N ga_search_${TAG} \
    "${_DEPEND_A[@]}" \
    -l select=1:ncpus=8:mem=194GB -l walltime=24:00:00 \
    -v "MODE=ga,${COMMON}" \
    pbs_pipeline_mode.pbs)
echo "Job A (search): ${JOB_A}  → ${GA_DIR}"

# ── Job B: staged full-population application ───────────────────────────────
# Reuses Job A's search results and stage-5b models (predict-only 5b — the
# path that OOM-killed job 23661073 is closed off three ways now).
JOB_B=$(qsub -N ga_apply_${TAG} \
    -W depend=afterok:${JOB_A} \
    -l select=1:ncpus=8:mem=250GB -l walltime=24:00:00 \
    -v "MODE=ga_staged,${COMMON},GA_STAGED_REUSE_ESTIMATOR=1,GA_STAGED_GA_DIR=${GA_DIR}" \
    pbs_pipeline_mode.pbs)
echo "Job B (apply):  ${JOB_B}  → runs/ga_parallel/sa_bandit_staged__${TAG}  (afterok:${JOB_A})"

# ── Jobs C-E: safety vs no-safety crash-skim showcase ───────────────────────
# After the search+apply chain completes, run both safety-experiment arms and
# the comparison so every guided run ends with the crash-skim outcome
# quantified: C predicts with the crash-risk skims active, D re-predicts with
# them zeroed (counterfactual), E compares the two plan sets and writes
# safety_report.html + safety_comparison.csv + mode-shift plots
# (runs/ga_parallel/safety_analysis__<TAG>).
JOB_C=$(qsub -N safety_base_${TAG} \
    -W depend=afterok:${JOB_B} \
    -l select=1:ncpus=8:mem=250GB -l walltime=24:00:00 \
    -v "MODE=safety_baseline,${COMMON}" \
    pbs_pipeline_mode.pbs)
echo "Job C (safety baseline):   ${JOB_C}  (afterok:${JOB_B})"

JOB_D=$(qsub -N safety_nosa_${TAG} \
    -W depend=afterok:${JOB_C} \
    -l select=1:ncpus=8:mem=250GB -l walltime=24:00:00 \
    -v "MODE=safety_nosafety,${COMMON}" \
    pbs_pipeline_mode.pbs)
echo "Job D (safety counterfactual): ${JOB_D}  (afterok:${JOB_C})"

JOB_E=$(qsub -N safety_anal_${TAG} \
    -W depend=afterok:${JOB_D} \
    -l select=1:ncpus=4:mem=32GB -l walltime=02:00:00 \
    -v "MODE=safety_analysis,CONDA_ENV=zigenv,RUN_ROOT=runs/ga_parallel,THREADS=1,S5_WORKERS=1,PIPELINE_ACCELERATOR=cpu,PIPELINE_RUN_TAG=${TAG}" \
    pbs_pipeline_mode.pbs)
echo "Job E (safety showcase):   ${JOB_E}  (afterok:${JOB_D})"

echo
echo "Monitor:   qstat -u \$(whoami)"
echo "Then grep the .o files for the 10 checks listed at the top of this script, e.g.:"
echo "  grep -E 'constraints:stage|retry done|falling back to NL|remaining_fail_zones|greedy random-param' ga_search_${TAG}.o*"
echo "  grep -E 'retry done|rebuilt full-zone|Stage 5b|Repair loop|vehicle.dat|sa2-slide' ga_apply_${TAG}.o*"
echo "Safety showcase outputs (after Job E):"
echo "  runs/ga_parallel/safety_analysis__${TAG}/safety_report.html"
echo "  runs/ga_parallel/safety_analysis__${TAG}/safety_comparison.csv + plots/"
