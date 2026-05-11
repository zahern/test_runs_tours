#!/bin/bash
# Submit one or more pipeline modes to PBS with cleaner resource control.
#
# Default resources (as of April 2026):
#   --gpu 1 --accelerator gpu --cpus 8 --workers 8 --mem 250GB --walltime 23:00:00
#
# Quick start (all 10 jobs — main chain + joint model + estimator variants + safety):
#   ./submit_pipeline_runs.sh --all

set -euo pipefail

MODES=(main ga ga_staged)
CPUS=8
THREADS=1
S5_WORKERS=8
MEM=250GB
WALLTIME=23:00:00
CONDA_ENV=zigenv
ZONE_ARG=
SUBMIT_MODE=parallel
GPU_COUNT=1
RUN_ROOT=runs
PBS_SCRIPT=pbs_pipeline_mode.pbs
ACCELERATOR=gpu
INCLUDE_SAFETY=0
INCLUDE_JOINT=0
INCLUDE_ESTIMATOR_VARIANTS=0   # submit main+metacount, main+searchlibrium, main+both
INCLUDE_GA_ESTIMATOR_VARIANTS=0 # submit ga+ga_staged for metacount and searchlibrium
PAPER_EXPERIMENT=0              # ga_staged + safety chain with conservative defaults
STAGE5_SMOKE_PRESET=0           # ga_stage5_smoke only (quick wiring check)
GA_ESTIMATOR=default            # GA_ESTIMATOR for the primary MODES above
INSTALL_ESTIMATORS=1            # default: install latest estimator packages at job start
CLEAN_RUN_DIRS=0                # 1 = clear previous run directories before submit (destructive)
# Lighter resources for analysis/joint jobs (no GPU, less RAM)
JOINT_MEM=32GB
JOINT_WALLTIME=04:00:00
ANALYSIS_MEM=32GB
ANALYSIS_WALLTIME=02:00:00
# MCR ExperimentBuilder structure-search jobs (metacountregressor 1.0.49+)
# Each family (count, duration, linear, cmf) runs an independent PBS job.
# --mcr-algo all submits SA+DE+HS variants per family (3× more jobs).
INCLUDE_MCR_SEARCH=0
MCR_FAMILIES="count duration linear"  # space-separated; add 'cmf' if data is CMF-compatible
MCR_ALGO=sa                           # sa | de | hs | all
MCR_MAX_ITER=2000                     # metaheuristic iterations per job
MCR_R=200                             # Halton draws for simulation
MCR_MEM=64GB
MCR_WALLTIME=08:00:00
MCR_CPUS=8
WALLTIME_SET=0
ANALYSIS_WALLTIME_SET=0

usage() {
cat <<'EOF'
Usage:
    ./submit_pipeline_runs.sh [options]

Options:
    --modes "main ga ga_staged"   Space-separated modes to submit (default: all three).
    --cpus N                       PBS ncpus per heavy job.         Default: 8
    --threads N                    OMP/MKL/OpenBLAS threads/job.   Default: 1
    --workers N                    Stage 5 zone workers.            Default: 8
                                   Set equal to --cpus for full parallelism.
                                   e.g. --cpus 8 --workers 8 cuts ~14h → ~2h.
    --mem SIZE                     PBS memory per heavy job.        Default: 250GB
    --walltime HH:MM:SS            PBS walltime per heavy job.      Default: 23:00:00
    --joint-mem SIZE               RAM for joint-model job.         Default: 32GB
    --joint-walltime HH:MM:SS      Walltime for joint-model job.    Default: 04:00:00
    --analysis-mem SIZE            RAM for safety_analysis job.     Default: 32GB
    --analysis-walltime HH:MM:SS   Walltime for safety_analysis.    Default: 02:00:00
    --zone ARG                     Optional pipeline_logger zone/file argument.
    --env NAME                     Conda environment.               Default: zigenv
    --run-root PATH                Base output directory.           Default: runs
    --parallel                     Submit all jobs independently (default).
    --sequential                   Submit jobs with afterok dependencies (main→ga→ga_staged).
     --include-safety               Also submit safety_baseline → safety_nosafety
                                    → safety_analysis chain.
     --include-joint                Also submit joint_model job (Tobit-I + Heckman
                                    participation + duration comparison) after main.
     --include-estimator-variants   Also submit three extra main-mode jobs, one for
                                    each alternative estimator:
                                      main (GA_ESTIMATOR=metacount)
                                      main (GA_ESTIMATOR=searchlibrium)
                                      main (GA_ESTIMATOR=both)
                                    Outputs land in runs/main_metacount__<JOBID>/,
                                    runs/main_searchlibrium__<JOBID>/, etc.
     --include-ga-estimator-variants
                                    Also submit ga → ga_staged chains for both
                                    metacount and searchlibrium estimators
                                    (independent of the primary MODES chain).
                                    Each ga job is followed by ga_staged (afterok).
                                    Outputs land in runs/ga_metacount__<JOBID>/,
                                    runs/ga_staged_metacount__<JOBID>/, etc.
     --ga-estimator VAL             Set GA_ESTIMATOR for the primary MODES.
                                    VAL: default | metacount | searchlibrium | both
                                    Default: default
    --force-install                Set INSTALL_ESTIMATORS=1 (default): ensure latest
                                    MetaCount + SearchLibrium from PyPI at job start.
    --skip-estimator-install       Set INSTALL_ESTIMATORS=0 (advanced): skip
                                    automatic estimator package refresh.
    --clean-runs                   Also clear previous run directories under
                             --run-root before submit.
                             WARNING: destructive. Leave off unless needed.
     --include-mcr-search           Submit MCR ExperimentBuilder structure-search jobs
                                    (metacountregressor 1.0.49+ required).
                                    One PBS job per family (default: count, duration, linear).
                                    Use --mcr-algo all to also run DE and HS variants.
     --mcr-algo sa|de|hs|all        Search algorithm for MCR jobs.   Default: sa
                                    'all' submits 3 variants per family (SA + DE + HS).
     --mcr-families "f1 f2 ..."     Space-separated MCR families.   Default: "count duration linear"
                                    Add 'cmf' if your data has aadt_col, baseline_vars, local_vars.
     --mcr-max-iter N               Metaheuristic iterations per MCR job. Default: 2000
     --mcr-r N                      Halton simulation draws per MCR job.  Default: 200
     --mcr-mem SIZE                 RAM per MCR search job.          Default: 64GB
     --mcr-walltime HH:MM:SS        Walltime per MCR search job.     Default: 08:00:00
     --all                          Shorthand for --include-safety --include-joint
                                    --include-estimator-variants --include-mcr-search.
                                    Submits all 13 jobs (SA default):
                                      main → ga → ga_staged
                                      main → joint_model
                                      main (metacount) | main (searchlibrium) | main (both)
                                      safety_baseline → safety_nosafety → safety_analysis
                                      mcr_search_count | mcr_search_duration | mcr_search_linear
                                    With --mcr-algo all: 22 jobs (adds DE+HS per family).
                                    With --mcr-families "count duration linear cmf": 16 jobs.
     --accelerator cpu|gpu          JAX/Larch platform.              Default: gpu
     --gpu N                        GPU count per heavy job.         Default: 1
    --paper-experiment             Submit GA-staged + safety chain as one
                             sequential paper experiment:
                             ga_staged -> safety_baseline -> safety_nosafety -> safety_analysis
                             Uses walltime=23:59:00 for heavy jobs unless --walltime
                             was explicitly provided, and analysis walltime=04:00:00
                             unless --analysis-walltime was explicitly provided.
    --stage5-smoke                Submit Stage 5-only GA smoke test as one job
                                  (mode: ga_stage5_smoke).
                                  Defaults to CPU, no GPU, workers=1, walltime=02:00:00
                                  unless explicitly overridden.
     --help                         Show this message.

Examples:
    # All 13 jobs: main chain, safety, estimator variants + MCR search SA (recommended):
    ./submit_pipeline_runs.sh --all

    # Same but override walltime only:
    ./submit_pipeline_runs.sh --all --walltime 20:00:00

    # All 22 jobs: main chain + safety + variants + MCR SA+DE+HS × all families:
    ./submit_pipeline_runs.sh --all --mcr-algo all \
        --mcr-families "count duration linear cmf"

    # MCR search only (SA, 3 families = 3 jobs), no pipeline:
    ./submit_pipeline_runs.sh --modes "" --include-mcr-search

    # MCR search with HS algorithm, more iterations:
    ./submit_pipeline_runs.sh --modes "" --include-mcr-search \
        --mcr-algo hs --mcr-max-iter 5000 --mcr-r 500

    # Main chain only, GPU + 8 workers:
    ./submit_pipeline_runs.sh --gpu 1 --accelerator gpu \
        --cpus 8 --workers 8 --mem 250GB --walltime 23:00:00

    # CPU-only (no GPU node):
    ./submit_pipeline_runs.sh --gpu 0 --accelerator cpu \
        --cpus 8 --workers 8 --mem 250GB

    # Sequential main→ga→ga_staged + joint model comparison:
    ./submit_pipeline_runs.sh --sequential --include-joint

    # Safety experiment only:
    ./submit_pipeline_runs.sh --modes "" --include-safety

    # Run main with MetaCount estimator only (single job):
    ./submit_pipeline_runs.sh --modes "main" --ga-estimator metacount

    # Run all three estimator variants of main (no GA chain):
    ./submit_pipeline_runs.sh --modes "" --include-estimator-variants

    # Force-install MetaCount+SearchLibrium on the HPC at job start:
    ./submit_pipeline_runs.sh --all --force-install

    # Paper experiment chain (GA staged + safety ablation + analysis):
    ./submit_pipeline_runs.sh --paper-experiment

    # Stage 5-only GA smoke test (very fast wiring check):
    ./submit_pipeline_runs.sh --stage5-smoke

Job graph (--all, MCR_ALGO=sa, MCR_FAMILIES="count duration linear"):
    J1: main  ──► J2: ga  ──► J3: ga_staged
         │
         └──► J4: joint_model        (Tobit-I + Heckman comparison, light)
    J5: main (GA_ESTIMATOR=metacount)
    J6: main (GA_ESTIMATOR=searchlibrium)
    J7: main (GA_ESTIMATOR=both)
    J8: safety_baseline ──► J9: safety_nosafety ──► J10: safety_analysis
    J11: mcr_search_count_sa    (ExperimentBuilder, SA, count model family)
    J12: mcr_search_duration_sa (ExperimentBuilder, SA, duration model family)
    J13: mcr_search_linear_sa   (ExperimentBuilder, SA, linear model family)
    [--mcr-algo all: +6 jobs DE+HS × 3 families = 19 total]
    [+cmf family:  +3 jobs (SA) or +9 (all algos) = 16 or 22 total]

Notes:
    - GPU (--accelerator gpu) speeds up model estimation via JAX/Larch.
      Prediction workers use CPU only — no GPU contention.
    - --workers N runs Stage 5 zone prediction in parallel (N fork processes).
      Rule of thumb: --mem = 100GB base + (workers × 8GB).
      With 8 workers the ~14h Stage 5 drops to ~2h wall-clock.
    - Joint model uses lighter resources (no GPU, 32 GB) and runs after main.
    - safety_analysis runs safety_experiment_analysis.py, not pipeline_logger.py.
    - --include-estimator-variants submits three independent main-mode jobs, one per
      alternative estimator. Each writes to its own run directory (named with the
      estimator suffix) so results can be compared side-by-side.
        - MetaCount and SearchLibrium are refreshed from PyPI by default at job start
            (latest metacountregressor and SearchLibrium).
            Use --skip-estimator-install only if you intentionally want to reuse the
            currently installed environment state.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --modes)
            read -r -a MODES <<< "$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --workers)
            S5_WORKERS="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --walltime)
            WALLTIME="$2"
            WALLTIME_SET=1
            shift 2
            ;;
        --zone)
            ZONE_ARG="$2"
            shift 2
            ;;
        --env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --run-root)
            RUN_ROOT="$2"
            shift 2
            ;;
        --parallel)
            SUBMIT_MODE=parallel
            shift
            ;;
        --sequential)
            SUBMIT_MODE=sequential
            shift
            ;;
        --include-safety)
            INCLUDE_SAFETY=1
            shift
            ;;
        --include-joint)
            INCLUDE_JOINT=1
            shift
            ;;
        --include-estimator-variants)
            INCLUDE_ESTIMATOR_VARIANTS=1
            shift
            ;;
        --include-ga-estimator-variants)
            INCLUDE_GA_ESTIMATOR_VARIANTS=1
            shift
            ;;
        --ga-estimator)
            GA_ESTIMATOR="$2"
            shift 2
            ;;
        --force-install)
            INSTALL_ESTIMATORS=1
            shift
            ;;
        --skip-estimator-install)
            INSTALL_ESTIMATORS=0
            shift
            ;;
        --clean-runs)
            CLEAN_RUN_DIRS=1
            shift
            ;;
        --include-mcr-search)
            INCLUDE_MCR_SEARCH=1
            shift
            ;;
        --mcr-algo)
            MCR_ALGO="$2"
            shift 2
            ;;
        --mcr-families)
            MCR_FAMILIES="$2"
            shift 2
            ;;
        --mcr-max-iter)
            MCR_MAX_ITER="$2"
            shift 2
            ;;
        --mcr-r)
            MCR_R="$2"
            shift 2
            ;;
        --mcr-mem)
            MCR_MEM="$2"
            shift 2
            ;;
        --mcr-walltime)
            MCR_WALLTIME="$2"
            shift 2
            ;;
        --all)
            INCLUDE_SAFETY=1
            INCLUDE_JOINT=1
            INCLUDE_ESTIMATOR_VARIANTS=1
            INCLUDE_MCR_SEARCH=1
            shift
            ;;
        --joint-mem)
            JOINT_MEM="$2"
            shift 2
            ;;
        --joint-walltime)
            JOINT_WALLTIME="$2"
            shift 2
            ;;
        --analysis-mem)
            ANALYSIS_MEM="$2"
            shift 2
            ;;
        --analysis-walltime)
            ANALYSIS_WALLTIME="$2"
            ANALYSIS_WALLTIME_SET=1
            shift 2
            ;;
        --paper-experiment)
            PAPER_EXPERIMENT=1
            shift
            ;;
        --stage5-smoke)
            STAGE5_SMOKE_PRESET=1
            shift
            ;;
        --accelerator)
            ACCELERATOR="$2"
            shift 2
            ;;
        --gpu)
            GPU_COUNT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ "$PAPER_EXPERIMENT" -eq 1 ]; then
    MODES=(ga_staged)
    INCLUDE_SAFETY=1
    SUBMIT_MODE=sequential
    if [ "$WALLTIME_SET" -eq 0 ]; then
        WALLTIME=23:59:00
    fi
    if [ "$ANALYSIS_WALLTIME_SET" -eq 0 ]; then
        ANALYSIS_WALLTIME=04:00:00
    fi
    echo "Paper experiment preset enabled: ga_staged + safety chain (sequential)."
fi

if [ "$PAPER_EXPERIMENT" -eq 1 ] && [ "$STAGE5_SMOKE_PRESET" -eq 1 ]; then
    echo "--paper-experiment and --stage5-smoke cannot be used together" >&2
    exit 1
fi

if [ "$STAGE5_SMOKE_PRESET" -eq 1 ]; then
    MODES=(ga_stage5_smoke)
    SUBMIT_MODE=parallel
    # Default to CPU. Pass --gpu 1 --accelerator gpu after --stage5-smoke to use a GPU node.
    # GPU enables JAX/XLA (CUDA) for the final model fit which is the slow part.
    if [ "$ACCELERATOR" != "gpu" ]; then
        ACCELERATOR=cpu
        GPU_COUNT=0
    fi
    S5_WORKERS=1
    if [ "$WALLTIME_SET" -eq 0 ]; then
        WALLTIME=02:00:00
    fi
    echo "Stage 5 smoke preset enabled: ga_stage5_smoke (accelerator=${ACCELERATOR}, workers=1)."
fi

if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub not found in PATH" >&2
    exit 1
fi

if [ ! -f "$PBS_SCRIPT" ]; then
    echo "PBS script not found: $PBS_SCRIPT" >&2
    exit 1
fi

assert_lf_only() {
    local f="$1"
    if LC_ALL=C grep -q $'\r' "$f"; then
        echo "CRLF detected in $f (contains \\r)." >&2
        echo "Convert to LF before submitting, e.g.:" >&2
        echo "  sed -i 's/\r$//' $f" >&2
        exit 1
    fi
}

assert_lf_only "$PBS_SCRIPT"

if [ "$THREADS" -gt "$CPUS" ]; then
    echo "--threads cannot exceed --cpus" >&2
    exit 1
fi

if [ "$S5_WORKERS" -gt "$CPUS" ]; then
    echo "Warning: --workers ($S5_WORKERS) exceeds --cpus ($CPUS) — reducing workers to $CPUS" >&2
    S5_WORKERS="$CPUS"
fi

if [ "$ACCELERATOR" = "gpu" ] && [ "$GPU_COUNT" -le 0 ]; then
    echo "--accelerator gpu requires --gpu > 0" >&2
    exit 1
fi

if [ "$ACCELERATOR" != "cpu" ] && [ "$ACCELERATOR" != "gpu" ]; then
    echo "--accelerator must be cpu or gpu" >&2
    exit 1
fi

SELECT="select=1:ncpus=${CPUS}:mem=${MEM}"
if [ "$GPU_COUNT" -gt 0 ]; then
    SELECT="${SELECT}:ngpus=${GPU_COUNT}"
fi
# Light select strings (no GPU) for joint_model and safety_analysis
SELECT_JOINT="select=1:ncpus=4:mem=${JOINT_MEM}"
SELECT_ANALYSIS="select=1:ncpus=4:mem=${ANALYSIS_MEM}"

submit_one() {
    local mode="$1"
    local depend="${2:-}"
    local estimator="${3:-${GA_ESTIMATOR}}"
    local job_name="saw_${mode}"
    # Append estimator suffix to job name so qstat clearly shows which variant
    if [ -n "$estimator" ] && [ "$estimator" != "default" ]; then
        job_name="saw_${mode}_${estimator}"
    fi
    local vars out_f err_f
    vars="MODE=${mode},ZONE_ARG=${ZONE_ARG},CONDA_ENV=${CONDA_ENV},THREADS=${THREADS},S5_WORKERS=${S5_WORKERS},RUN_ROOT=${RUN_ROOT},PIPELINE_ACCELERATOR=${ACCELERATOR},GA_ESTIMATOR=${estimator},INSTALL_ESTIMATORS=${INSTALL_ESTIMATORS}"
    out_f="$(pwd)/${job_name}.out"
    err_f="$(pwd)/${job_name}.err"

    if [ -n "$depend" ]; then
        qsub -N "$job_name" -o "$out_f" -e "$err_f" -l walltime="$WALLTIME" -l "$SELECT" -W "depend=afterok:${depend}" -v "$vars" "$PBS_SCRIPT"
    else
        qsub -N "$job_name" -o "$out_f" -e "$err_f" -l walltime="$WALLTIME" -l "$SELECT" -v "$vars" "$PBS_SCRIPT"
    fi
}

echo "Submitting modes: ${MODES[*]:-<none>}"
echo "  submit_mode        = $SUBMIT_MODE"
echo "  select             = $SELECT"
echo "  walltime           = $WALLTIME"
echo "  threads            = $THREADS"
echo "  s5_workers         = $S5_WORKERS"
echo "  accelerator        = $ACCELERATOR"
echo "  ga_estimator       = $GA_ESTIMATOR"
echo "  install_estimators = $INSTALL_ESTIMATORS"
echo "  zone_arg           = ${ZONE_ARG:-<all>}"
echo "  conda_env          = $CONDA_ENV"
echo "  include_safety     = $INCLUDE_SAFETY"
echo "  include_joint      = $INCLUDE_JOINT"
echo "  include_est_var    = $INCLUDE_ESTIMATOR_VARIANTS"
echo "  include_ga_est_var = $INCLUDE_GA_ESTIMATOR_VARIANTS"
echo "  clean_run_dirs     = $CLEAN_RUN_DIRS"
echo "  joint select       = $SELECT_JOINT  walltime=$JOINT_WALLTIME"
echo "  analysis sel.      = $SELECT_ANALYSIS  walltime=$ANALYSIS_WALLTIME"

# ── submit_safety_chain ────────────────────────────────────────────────────
# Submits safety_baseline → safety_nosafety → safety_analysis (afterok chain).
# safety_baseline and safety_nosafety use full heavy resources (GPU).
# safety_analysis uses light resources (no GPU, ANALYSIS_MEM/WALLTIME).
# Pass a job-id as first argument to make safety_baseline depend on it.
submit_safety_chain() {
    local base_depend="${1:-}"
    local jid_base jid_nosafety jid_analysis
    local vars_base vars_nosa vars_anal

    vars_base="MODE=safety_baseline,ZONE_ARG=${ZONE_ARG},CONDA_ENV=${CONDA_ENV},THREADS=${THREADS},S5_WORKERS=${S5_WORKERS},RUN_ROOT=${RUN_ROOT},PIPELINE_ACCELERATOR=${ACCELERATOR}"
    vars_nosa="MODE=safety_nosafety,ZONE_ARG=${ZONE_ARG},CONDA_ENV=${CONDA_ENV},THREADS=${THREADS},S5_WORKERS=${S5_WORKERS},RUN_ROOT=${RUN_ROOT},PIPELINE_ACCELERATOR=${ACCELERATOR}"
    vars_anal="MODE=safety_analysis,ZONE_ARG=,CONDA_ENV=${CONDA_ENV},THREADS=1,S5_WORKERS=1,RUN_ROOT=${RUN_ROOT},PIPELINE_ACCELERATOR=cpu"

    if [ -n "$base_depend" ]; then
        jid_base=$(qsub -N saw_safety_base \
            -o "$(pwd)/saw_safety_base.out" -e "$(pwd)/saw_safety_base.err" \
            -l walltime="$WALLTIME" -l "$SELECT" \
            -W "depend=afterok:${base_depend}" -v "$vars_base" "$PBS_SCRIPT")
        echo "  safety_baseline -> $jid_base (afterok:$base_depend)"
    else
        jid_base=$(qsub -N saw_safety_base \
            -o "$(pwd)/saw_safety_base.out" -e "$(pwd)/saw_safety_base.err" \
            -l walltime="$WALLTIME" -l "$SELECT" \
            -v "$vars_base" "$PBS_SCRIPT")
        echo "  safety_baseline -> $jid_base"
    fi

    jid_nosafety=$(qsub -N saw_safety_nosa \
        -o "$(pwd)/saw_safety_nosa.out" -e "$(pwd)/saw_safety_nosa.err" \
        -l walltime="$WALLTIME" -l "$SELECT" \
        -W "depend=afterok:${jid_base}" -v "$vars_nosa" "$PBS_SCRIPT")
    echo "  safety_nosafety -> $jid_nosafety (afterok:$jid_base)"

    jid_analysis=$(qsub -N saw_safety_anal \
        -o "$(pwd)/saw_safety_anal.out" -e "$(pwd)/saw_safety_anal.err" \
        -l walltime="$ANALYSIS_WALLTIME" -l "$SELECT_ANALYSIS" \
        -W "depend=afterok:${jid_nosafety}" -v "$vars_anal" "$PBS_SCRIPT")
    echo "  safety_analysis -> $jid_analysis (afterok:$jid_nosafety)"

    prev_jid="$jid_analysis"
}

# ── submit_joint ────────────────────────────────────────────────────────────
# Submits the Tobit-I + Heckman joint participation+duration model job.
# Runs after main (needs stage2_models.pkl); uses light resources (no GPU).
# Writes stage2_joint_model_comparison.html + comparison plots into the main
# run directory.
submit_joint() {
    local depend="${1:-}"
    local jid_joint
    local vars_joint
    vars_joint="MODE=joint_model,ZONE_ARG=,CONDA_ENV=${CONDA_ENV},THREADS=1,S5_WORKERS=1,RUN_ROOT=${RUN_ROOT},PIPELINE_ACCELERATOR=cpu"

    if [ -n "$depend" ]; then
        jid_joint=$(qsub -N saw_joint \
            -o "$(pwd)/saw_joint.out" -e "$(pwd)/saw_joint.err" \
            -l walltime="$JOINT_WALLTIME" -l "$SELECT_JOINT" \
            -W "depend=afterok:${depend}" -v "$vars_joint" "$PBS_SCRIPT")
        echo "  joint_model     -> $jid_joint (afterok:$depend)"
    else
        jid_joint=$(qsub -N saw_joint \
            -o "$(pwd)/saw_joint.out" -e "$(pwd)/saw_joint.err" \
            -l walltime="$JOINT_WALLTIME" -l "$SELECT_JOINT" \
            -v "$vars_joint" "$PBS_SCRIPT")
        echo "  joint_model     -> $jid_joint"
    fi
    prev_jid="$jid_joint"
}

# ── submit_mcr_search ────────────────────────────────────────────────────────
# Submits one PBS job per (family, algorithm) combination using MCR
# ExperimentBuilder. Each job runs pipeline_logger.py in 'mcr_search' mode
# with MCR_FAMILY and MCR_ALGO passed as PBS variables.
# Light resources: no GPU needed (JAX CPU solver).
#
# Job naming: saw_mcr_<family>_<algo>  e.g. saw_mcr_count_sa
# Output dir : runs/mcr_<family>_<algo>__<JOBID>/
submit_mcr_search() {
    local depend="${1:-}"
    local _select="select=1:ncpus=${MCR_CPUS}:mem=${MCR_MEM}"
    # Resolve algorithm list
    local algos
    if [ "$MCR_ALGO" = "all" ]; then
        algos="sa de hs"
    else
        algos="$MCR_ALGO"
    fi
    # Read families into array
    read -r -a _fams <<< "$MCR_FAMILIES"
    echo ""
    echo "── MCR structure-search jobs [families=${MCR_FAMILIES}  algos=${algos}] ──"
    local _job_count=0
    for fam in "${_fams[@]}"; do
        for alg in $algos; do
            local job_name="saw_mcr_${fam}_${alg}"
            local vars_mcr
            vars_mcr="MODE=mcr_search,MCR_FAMILY=${fam},MCR_ALGO=${alg},MCR_MAX_ITER=${MCR_MAX_ITER},MCR_R=${MCR_R},ZONE_ARG=,CONDA_ENV=${CONDA_ENV},THREADS=1,S5_WORKERS=1,RUN_ROOT=${RUN_ROOT},PIPELINE_ACCELERATOR=cpu,INSTALL_ESTIMATORS=${INSTALL_ESTIMATORS}"
            local jid_mcr
            if [ -n "$depend" ]; then
                jid_mcr=$(qsub -N "$job_name" \
                    -o "$(pwd)/${job_name}.out" -e "$(pwd)/${job_name}.err" \
                    -l walltime="$MCR_WALLTIME" -l "$_select" \
                    -W "depend=afterok:${depend}" -v "$vars_mcr" "$PBS_SCRIPT")
                echo "  mcr_${fam}_${alg} -> $jid_mcr (afterok:$depend)"
            else
                jid_mcr=$(qsub -N "$job_name" \
                    -o "$(pwd)/${job_name}.out" -e "$(pwd)/${job_name}.err" \
                    -l walltime="$MCR_WALLTIME" -l "$_select" \
                    -v "$vars_mcr" "$PBS_SCRIPT")
                echo "  mcr_${fam}_${alg} -> $jid_mcr"
            fi
            echo "    log: ${job_name}.out"
            echo "    dir: ${RUN_ROOT}/mcr_${fam}_${alg}__<JOBID>/"
            _job_count=$((_job_count + 1))
        done
    done
    echo "  MCR search submitted: ${_job_count} job(s)  [families=${#_fams[@]}  algos=$(echo $algos | wc -w | tr -d ' ')]"
    echo "────────────────────────────────────────────────────────────────────"
}

# ── submit_ga_estimator_chain ────────────────────────────────────────────────
# Submits ga → ga_staged (afterok) for a given estimator.
# The ga_staged job always waits for ga so results are consistent.
# Optional second argument: a job-id that ga itself should depend on.
submit_ga_estimator_chain() {
    local est="$1"
    local depend="${2:-}"
    local jid_ga jid_ga_staged

    echo ""
    echo "── GA estimator chain [GA_ESTIMATOR=${est}] ──────────────────────────"
    jid_ga=$(submit_one "ga" "$depend" "$est")
    echo "  ga [GA_ESTIMATOR=${est}]         -> $jid_ga${depend:+ (afterok:$depend)}"

    jid_ga_staged=$(submit_one "ga_staged" "$jid_ga" "$est")
    echo "  ga_staged [GA_ESTIMATOR=${est}]  -> $jid_ga_staged (afterok:$jid_ga)"
    echo "    logs : saw_ga_${est}.out  saw_ga_staged_${est}.out"
    echo "    dirs : ${RUN_ROOT}/ga_${est}__<JOBID>/  ${RUN_ROOT}/ga_staged_${est}__<JOBID>/"
    echo "────────────────────────────────────────────────────────────────────"
}

# ── clean_previous_outputs ─────────────────────────────────────────────────
# Remove .err/.out files ONLY for the modes being submitted in this call so
# outputs from other running/completed jobs are never accidentally deleted.
# Also clear run-directory contents when --clean-runs is given.
clean_previous_outputs() {
    echo ""
    echo "── Cleaning PBS log files for submitted modes only ─────────────────"

    # Build the list of job-name stems that will be submitted in this call.
    local stems=()
    for mode in "${MODES[@]}"; do
        local stem="saw_${mode}"
        local est="${GA_ESTIMATOR}"
        [ -n "$est" ] && [ "$est" != "default" ] && stem="saw_${mode}_${est}"
        stems+=("$stem")
        # ga always auto-adds ga_staged
        if [ "$mode" = "ga" ]; then
            local gs_stem="saw_ga_staged"
            [ -n "$est" ] && [ "$est" != "default" ] && gs_stem="saw_ga_staged_${est}"
            stems+=("$gs_stem")
        fi
    done
    [ "$INCLUDE_JOINT" -eq 1 ]  && stems+=("saw_joint")
    if [ "$INCLUDE_SAFETY" -eq 1 ]; then
        stems+=("saw_safety_base" "saw_safety_nosa" "saw_safety_anal")
    fi
    if [ "$INCLUDE_ESTIMATOR_VARIANTS" -eq 1 ]; then
        for est in metacount searchlibrium both; do
            stems+=("saw_main_${est}")
        done
    fi
    if [ "$INCLUDE_GA_ESTIMATOR_VARIANTS" -eq 1 ]; then
        for est in metacount searchlibrium; do
            stems+=("saw_ga_${est}" "saw_ga_staged_${est}")
        done
    fi
    if [ "$INCLUDE_MCR_SEARCH" -eq 1 ]; then
        local _algos; _algos=$( [ "$MCR_ALGO" = "all" ] && echo "sa de hs" || echo "$MCR_ALGO" )
        read -r -a _fams_arr <<< "$MCR_FAMILIES"
        for fam in "${_fams_arr[@]}"; do
            for alg in $_algos; do
                stems+=("saw_mcr_${fam}_${alg}")
            done
        done
    fi

    local deleted=0
    for stem in "${stems[@]}"; do
        for pat in "${stem}.out" "${stem}.err" "${stem}.e[0-9]*" "${stem}.o[0-9]*"; do
            # shellcheck disable=SC2086
            local matches; matches=$(ls $pat 2>/dev/null | wc -l) || true
            if [ "$matches" -gt 0 ]; then
                # shellcheck disable=SC2086
                rm -f $pat
                deleted=$((deleted + matches))
            fi
        done
    done
    echo "  Removed ${deleted} log file(s) for: ${stems[*]}"
    echo "  (Logs from other modes/jobs were left untouched.)"

    # Optional destructive cleanup of run directories.
    if [ "$CLEAN_RUN_DIRS" -eq 1 ]; then
        if [ -d "${RUN_ROOT}" ]; then
            local cleared=0
            for mode_pat in "${MODES[@]}" ; do
                for run_dir in "${RUN_ROOT}/${mode_pat}__"*/; do
                    if [ -d "$run_dir" ]; then
                        find "$run_dir" -mindepth 1 -delete 2>/dev/null && cleared=$((cleared + 1)) || true
                    fi
                done
            done
            if [ "$INCLUDE_JOINT" -eq 1 ]; then
                for run_dir in "${RUN_ROOT}/joint_model__"*/; do
                    if [ -d "$run_dir" ]; then
                        find "$run_dir" -mindepth 1 -delete 2>/dev/null && cleared=$((cleared + 1)) || true
                    fi
                done
            fi
            if [ "$INCLUDE_SAFETY" -eq 1 ]; then
                for sm in safety_baseline safety_nosafety safety_analysis; do
                    for run_dir in "${RUN_ROOT}/${sm}__"*/; do
                        if [ -d "$run_dir" ]; then
                            find "$run_dir" -mindepth 1 -delete 2>/dev/null && cleared=$((cleared + 1)) || true
                        fi
                    done
                done
            fi
            if [ "$INCLUDE_ESTIMATOR_VARIANTS" -eq 1 ]; then
                for est_suffix in metacount searchlibrium both; do
                    for run_dir in "${RUN_ROOT}/main_${est_suffix}__"*/; do
                        if [ -d "$run_dir" ]; then
                            find "$run_dir" -mindepth 1 -delete 2>/dev/null && cleared=$((cleared + 1)) || true
                        fi
                    done
                done
            fi
            if [ "$INCLUDE_GA_ESTIMATOR_VARIANTS" -eq 1 ]; then
                for est_suffix in metacount searchlibrium; do
                    for run_dir in "${RUN_ROOT}/ga_${est_suffix}__"*/ "${RUN_ROOT}/ga_staged_${est_suffix}__"*/; do
                        if [ -d "$run_dir" ]; then
                            find "$run_dir" -mindepth 1 -delete 2>/dev/null && cleared=$((cleared + 1)) || true
                        fi
                    done
                done
            fi
            echo "  Cleared ${cleared} run output director(ies) in ${RUN_ROOT}/."
        fi
    else
        echo "  Run directory cleanup skipped (default). Use --clean-runs to enable."
    fi
    echo "────────────────────────────────────────────────────────────────────"
    echo ""
}

clean_previous_outputs

prev_jid=
jid_main=
jid_ga=
for mode in "${MODES[@]}"; do
    # ga_staged ALWAYS depends on ga (afterok), regardless of --parallel/--sequential.
    # This ensures ga results are available before ga_staged starts.
    if [ "$mode" = "ga_staged" ] && [ -n "$jid_ga" ]; then
        jid=$(submit_one "$mode" "$jid_ga")
        echo "  $mode -> $jid (afterok:$jid_ga)"
    elif [ "$SUBMIT_MODE" = sequential ] && [ -n "$prev_jid" ]; then
        jid=$(submit_one "$mode" "$prev_jid")
        echo "  $mode -> $jid (afterok:$prev_jid)"
    else
        jid=$(submit_one "$mode")
        echo "  $mode -> $jid"
    fi
    # track job IDs for downstream dependencies
    if [ "$mode" = "main" ]; then
        jid_main="$jid"
    fi
    if [ "$mode" = "ga" ]; then
        jid_ga="$jid"
    fi
    prev_jid="$jid"
done

# If ga was in MODES but ga_staged was NOT, auto-submit ga_staged after ga.
# Rule: when you run ga, ga_staged always follows it.
if [ -n "$jid_ga" ]; then
    _ga_staged_in_modes=0
    for _m in "${MODES[@]}"; do
        [ "$_m" = "ga_staged" ] && _ga_staged_in_modes=1
    done
    if [ "$_ga_staged_in_modes" -eq 0 ]; then
        echo "  (auto) ga_staged always follows ga — adding ga_staged after ga"
        jid=$(submit_one "ga_staged" "$jid_ga")
        echo "  ga_staged -> $jid (afterok:$jid_ga)"
        prev_jid="$jid"
    fi
fi

# joint_model: always depends on main (needs stage2_models.pkl)
if [ "$INCLUDE_JOINT" -eq 1 ]; then
    if [ -n "$jid_main" ]; then
        submit_joint "$jid_main"
    else
        submit_joint
    fi
fi

if [ "$INCLUDE_SAFETY" -eq 1 ]; then
    # Safety chain is independent of the main model chain — submit in parallel
    # unless --sequential was given (in which case chain after last main job).
    if [ "$SUBMIT_MODE" = sequential ] && [ -n "$prev_jid" ]; then
        submit_safety_chain "$prev_jid"
    else
        submit_safety_chain
    fi
fi

# ── estimator variants (main only, three parallel jobs) ─────────────────────
# Each variant runs pipeline_logger.py main mode with a different GA_ESTIMATOR.
# Outputs land in separate run directories for side-by-side comparison.
if [ "$INCLUDE_ESTIMATOR_VARIANTS" -eq 1 ]; then
    echo ""
    echo "── Estimator variant jobs ──────────────────────────────────────────"
    for est in metacount searchlibrium both; do
        jid_est=$(submit_one "main" "" "$est")
        echo "  main [GA_ESTIMATOR=$est] -> $jid_est"
        echo "    log: saw_main_${est}.out"
        echo "    dir: ${RUN_ROOT}/main_${est}__<JOBID>/"
    done
    echo "────────────────────────────────────────────────────────────────────"
fi

# ── GA estimator variant chains (ga → ga_staged per estimator) ───────────────
# For each estimator (metacount, searchlibrium) submit ga then ga_staged (afterok).
# Runs independently of the primary MODES chain.
if [ "$INCLUDE_GA_ESTIMATOR_VARIANTS" -eq 1 ]; then
    for est in metacount searchlibrium; do
        submit_ga_estimator_chain "$est"
    done
fi

# ── MCR ExperimentBuilder structure-search jobs ───────────────────────────────
# One PBS job per (family, algo) combination.  All run independently (parallel).
# Families: count, duration, linear  [+ cmf if --mcr-families includes it]
# Algorithms: sa | de | hs | all
# metacountregressor 1.0.49+ required; jaxopt must be in the environment.
if [ "$INCLUDE_MCR_SEARCH" -eq 1 ]; then
    # MCR search jobs are independent of the main pipeline chain.
    # Pass an optional dependency on main if --sequential was given.
    if [ "$SUBMIT_MODE" = sequential ] && [ -n "$jid_main" ]; then
        submit_mcr_search "$jid_main"
    else
        submit_mcr_search
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Resources used"
echo "═══════════════════════════════════════════════════════════════"
echo "  Heavy jobs  : $SELECT  walltime=$WALLTIME  accelerator=$ACCELERATOR"
echo "  Joint model : $SELECT_JOINT  walltime=$JOINT_WALLTIME  (no GPU)"
echo "  Analysis    : $SELECT_ANALYSIS  walltime=$ANALYSIS_WALLTIME  (no GPU)"
echo "  MCR search  : select=1:ncpus=${MCR_CPUS}:mem=${MCR_MEM}  walltime=$MCR_WALLTIME  algo=$MCR_ALGO  max_iter=$MCR_MAX_ITER  R=$MCR_R"
echo "  MCR families: $MCR_FAMILIES"
echo ""
echo "  Job graph"
echo "  ---------"
if [ "${#MODES[@]}" -gt 0 ]; then
    echo "  ${MODES[*]}  (${SUBMIT_MODE}, GA_ESTIMATOR=${GA_ESTIMATOR})"
    if [ -n "$jid_ga" ]; then
        echo "  NOTE: ga_staged always runs afterok:ga"
    fi
fi
if [ "$INCLUDE_JOINT" -eq 1 ]; then
    echo "  main ──► joint_model  (Tobit-I + Heckman comparison)"
fi
if [ "$INCLUDE_ESTIMATOR_VARIANTS" -eq 1 ]; then
    echo "  main [GA_ESTIMATOR=metacount]     (independent)"
    echo "  main [GA_ESTIMATOR=searchlibrium] (independent)"
    echo "  main [GA_ESTIMATOR=both]          (independent)"
fi
if [ "$INCLUDE_GA_ESTIMATOR_VARIANTS" -eq 1 ]; then
    echo "  ga [metacount]  ──► ga_staged [metacount]     (afterok, independent chain)"
    echo "  ga [searchlibrium] ──► ga_staged [searchlibrium] (afterok, independent chain)"
fi
if [ "$INCLUDE_SAFETY" -eq 1 ]; then
    echo "  safety_baseline ──► safety_nosafety ──► safety_analysis"
fi
if [ "$INCLUDE_MCR_SEARCH" -eq 1 ]; then
    read -r -a _mcr_fams_arr <<< "$MCR_FAMILIES"
    _mcr_alg_list=$( [ "$MCR_ALGO" = "all" ] && echo "sa de hs" || echo "$MCR_ALGO" )
    _n_mcr_fams=${#_mcr_fams_arr[@]}
    _n_mcr_algs=$(echo $_mcr_alg_list | wc -w | tr -d ' ')
    _n_mcr_jobs=$(( _n_mcr_fams * _n_mcr_algs ))
    echo "  MCR search: families=[${MCR_FAMILIES}]  algos=[${_mcr_alg_list}]"
    for fam in "${_mcr_fams_arr[@]}"; do
        for alg in $_mcr_alg_list; do
            echo "    mcr_${fam}_${alg}  (independent, ${MCR_MEM}, ${MCR_WALLTIME}, ${MCR_MAX_ITER} iter, R=${MCR_R})"
        done
    done
    echo "  MCR total: ${_n_mcr_jobs} job(s)"
fi
# ── Total job count ───────────────────────────────────────────────────────────
_total=0
_total=$(( _total + ${#MODES[@]} ))
[ -n "$jid_ga" ] && _total=$(( _total + 1 ))   # auto ga_staged if ga in modes but ga_staged not
_gst_in=0; for _m in "${MODES[@]}"; do [ "$_m" = "ga_staged" ] && _gst_in=1; done
[ "$_gst_in" -eq 1 ] && _total=$(( _total - 1 ))  # already counted above
[ "$INCLUDE_JOINT" -eq 1 ]                  && _total=$(( _total + 1 ))
[ "$INCLUDE_ESTIMATOR_VARIANTS" -eq 1 ]     && _total=$(( _total + 3 ))
if [ "$INCLUDE_GA_ESTIMATOR_VARIANTS" -eq 1 ]; then _total=$(( _total + 4 )); fi
[ "$INCLUDE_SAFETY" -eq 1 ]                 && _total=$(( _total + 3 ))
if [ "$INCLUDE_MCR_SEARCH" -eq 1 ]; then
    read -r -a _c_fams <<< "$MCR_FAMILIES"
    _c_alg=$( [ "$MCR_ALGO" = "all" ] && echo "sa de hs" || echo "$MCR_ALGO" )
    _c_alg_n=$(echo $_c_alg | wc -w | tr -d ' ')
    _total=$(( _total + ${#_c_fams[@]} * _c_alg_n ))
fi
echo ""
echo "  TOTAL JOBS SUBMITTED: ${_total}"
echo ""
echo "  Key outputs"
echo "  -----------"
for mode in "${MODES[@]}"; do
    echo "  runs/${mode}__<JOBID>/   stage models, HTML reports, plots"
done
if [ "$INCLUDE_JOINT" -eq 1 ]; then
    echo "  runs/main__<JOBID>/      stage2_joint_model_comparison.html"
    echo "                           plots/joint_participation_comparison.png"
    echo "                           plots/joint_budget_scatter.png"
    echo "                           plots/joint_coefficient_imr.png"
fi
if [ "$INCLUDE_ESTIMATOR_VARIANTS" -eq 1 ]; then
    for est in metacount searchlibrium both; do
        echo "  runs/main_${est}__<JOBID>/        full pipeline run with GA_ESTIMATOR=${est}"
    done
fi
if [ "$INCLUDE_GA_ESTIMATOR_VARIANTS" -eq 1 ]; then
    for est in metacount searchlibrium; do
        echo "  runs/ga_${est}__<JOBID>/          GA run with GA_ESTIMATOR=${est}"
        echo "  runs/ga_staged_${est}__<JOBID>/   staged GA run with GA_ESTIMATOR=${est}"
    done
fi
if [ "$INCLUDE_SAFETY" -eq 1 ]; then
    echo "  runs/safety_baseline__<JOBID>/  full plan with safety skims"
    echo "  runs/safety_nosafety__<JOBID>/  full plan without safety skims"
    echo "  runs/safety_analysis__<JOBID>/  safety_experiment_report.html"
    echo "                                  plots/safety_mode_share.png"
    echo "                                  plots/safety_activity_levels.png"
    echo "                                  plots/safety_zone_derivation.png"
    echo "                                  safety_zone_ranking.csv"
fi
if [ "$INCLUDE_MCR_SEARCH" -eq 1 ]; then
    read -r -a _kout_fams <<< "$MCR_FAMILIES"
    _kout_algs=$( [ "$MCR_ALGO" = "all" ] && echo "sa de hs" || echo "$MCR_ALGO" )
    for fam in "${_kout_fams[@]}"; do
        for alg in $_kout_algs; do
            echo "  runs/mcr_${fam}_${alg}__<JOBID>/  MCR ExperimentBuilder search (${fam}, ${alg})"
            echo "                                  mcr_search_result.json  best_spec.json"
            echo "                                  mcr_search_summary.html"
        done
    done
fi
echo ""
echo "  Monitor (real-time — available immediately once job starts)"
echo "  -------"
echo "  qstat -u \$USER"
for mode in "${MODES[@]}"; do
    echo "  tail -f saw_${mode}.out                                 # PBS stdout"
    echo "  tail -f ${RUN_ROOT}/${mode}__<JOBID>/pipeline_run.log  # Python log"
done
if [ "$INCLUDE_ESTIMATOR_VARIANTS" -eq 1 ]; then
    for est in metacount searchlibrium both; do
        echo "  tail -f saw_main_${est}.out"
        echo "  tail -f ${RUN_ROOT}/main_${est}__<JOBID>/pipeline_run.log"
    done
fi
if [ "$INCLUDE_GA_ESTIMATOR_VARIANTS" -eq 1 ]; then
    for est in metacount searchlibrium; do
        echo "  tail -f saw_ga_${est}.out"
        echo "  tail -f saw_ga_staged_${est}.out"
        echo "  tail -f ${RUN_ROOT}/ga_${est}__<JOBID>/pipeline_run.log"
        echo "  tail -f ${RUN_ROOT}/ga_staged_${est}__<JOBID>/pipeline_run.log"
    done
fi
if [ "$INCLUDE_JOINT" -eq 1 ]; then
    echo "  tail -f saw_joint.out"
fi
if [ "$INCLUDE_SAFETY" -eq 1 ]; then
    echo "  tail -f saw_safety_base.out   saw_safety_base.err"
    echo "  tail -f saw_safety_nosa.out   saw_safety_nosa.err"
    echo "  tail -f saw_safety_anal.out   saw_safety_anal.err"
fi
echo "═══════════════════════════════════════════════════════════════"
