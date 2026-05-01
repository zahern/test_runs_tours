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
GA_ESTIMATOR=default            # GA_ESTIMATOR for the primary MODES above
INSTALL_ESTIMATORS=0            # 1 = force reinstall MetaCount+SearchLibrium at job start
# Lighter resources for analysis/joint jobs (no GPU, less RAM)
JOINT_MEM=32GB
JOINT_WALLTIME=04:00:00
ANALYSIS_MEM=32GB
ANALYSIS_WALLTIME=02:00:00

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
     --ga-estimator VAL             Set GA_ESTIMATOR for the primary MODES.
                                    VAL: default | metacount | searchlibrium | both
                                    Default: default
     --force-install                Set INSTALL_ESTIMATORS=1: unconditionally
                                    (re-)install MetaCount + SearchLibrium at job
                                    start via pip.  Useful on first use.
     --all                          Shorthand for --include-safety --include-joint
                                    --include-estimator-variants.
                                    Submits all 10 jobs:
                                      main → ga → ga_staged
                                      main → joint_model
                                      main (metacount) | main (searchlibrium) | main (both)
                                      safety_baseline → safety_nosafety → safety_analysis
     --accelerator cpu|gpu          JAX/Larch platform.              Default: gpu
     --gpu N                        GPU count per heavy job.         Default: 1
     --help                         Show this message.

Examples:
    # All 10 jobs with standard resources (recommended):
    ./submit_pipeline_runs.sh --all

    # Same but override walltime only:
    ./submit_pipeline_runs.sh --all --walltime 20:00:00

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

Job graph (--all):
    J1: main  ──► J2: ga  ──► J3: ga_staged
         │
         └──► J4: joint_model        (Tobit-I + Heckman comparison, light)
    J5: main (GA_ESTIMATOR=metacount)
    J6: main (GA_ESTIMATOR=searchlibrium)
    J7: main (GA_ESTIMATOR=both)
    J8: safety_baseline ──► J9: safety_nosafety ──► J10: safety_analysis

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
    - MetaCount and SearchLibrium are installed automatically inside each PBS job
      when GA_ESTIMATOR requires them. Pass --force-install to reinstall regardless.
      Source paths checked: /mnt/hpccs01/home/ahernz/source/{MetaCount,SearchLibrium}
      Fallback: PyPI (pip install metacountregressor SearchLibrium).
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
        --ga-estimator)
            GA_ESTIMATOR="$2"
            shift 2
            ;;
        --force-install)
            INSTALL_ESTIMATORS=1
            shift
            ;;
        --all)
            INCLUDE_SAFETY=1
            INCLUDE_JOINT=1
            INCLUDE_ESTIMATOR_VARIANTS=1
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
            shift 2
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

# ── clean_previous_outputs ─────────────────────────────────────────────────
# Remove .err/.out files produced by PBS from the previous submission so they
# do not accumulate between runs.  Also clear the contents of run directories
# that belong to the modes being re-submitted (preserves the directory tree
# itself so any pre-staged inputs are not lost).
clean_previous_outputs() {
    echo ""
    echo "── Cleaning previous PBS output files ──────────────────────────────"
    # PBS stderr/stdout files in the code directory — both old-style saw_<mode>.e<jobid>
    # and new fixed-name saw_<mode>.out / saw_<mode>.err
    local patterns=(
        "saw_main.e[0-9]*"              "saw_main.o[0-9]*"
        "saw_ga.e[0-9]*"               "saw_ga.o[0-9]*"
        "saw_ga_staged.e[0-9]*"        "saw_ga_staged.o[0-9]*"
        "saw_joint.e[0-9]*"            "saw_joint.o[0-9]*"
        "saw_safety*.e[0-9]*"          "saw_safety*.o[0-9]*"
        "saw_main_metacount.e[0-9]*"   "saw_main_metacount.o[0-9]*"
        "saw_main_searchlibrium.e[0-9]*" "saw_main_searchlibrium.o[0-9]*"
        "saw_main_both.e[0-9]*"        "saw_main_both.o[0-9]*"
        "saw_main.out"                 "saw_main.err"
        "saw_ga.out"                   "saw_ga.err"
        "saw_ga_staged.out"            "saw_ga_staged.err"
        "saw_joint.out"                "saw_joint.err"
        "saw_safety_base.out"          "saw_safety_base.err"
        "saw_safety_nosa.out"          "saw_safety_nosa.err"
        "saw_safety_anal.out"          "saw_safety_anal.err"
        "saw_main_metacount.out"       "saw_main_metacount.err"
        "saw_main_searchlibrium.out"   "saw_main_searchlibrium.err"
        "saw_main_both.out"            "saw_main_both.err"
        "log_main.out"                 "log_main.err"
        "log_ga.out"                   "log_ga.err"
        "log_ga_staged.out"            "log_ga_staged.err"
        "log_safety*.out"              "log_safety*.err"
    )
    local deleted=0
    for pat in "${patterns[@]}"; do
        # shellcheck disable=SC2086
        local matches
        matches=$(ls $pat 2>/dev/null | wc -l) || true
        if [ "$matches" -gt 0 ]; then
            # shellcheck disable=SC2086
            rm -f $pat
            deleted=$((deleted + matches))
        fi
    done
    echo "  Removed ${deleted} PBS log file(s) from code directory."

    # Clear the contents of run output directories for modes being submitted.
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
        echo "  Cleared ${cleared} run output director(ies) in ${RUN_ROOT}/."
    fi
    echo "────────────────────────────────────────────────────────────────────"
    echo ""
}

clean_previous_outputs

prev_jid=
jid_main=
for mode in "${MODES[@]}"; do
    if [ "$SUBMIT_MODE" = sequential ] && [ -n "$prev_jid" ]; then
        jid=$(submit_one "$mode" "$prev_jid")
        echo "  $mode -> $jid (afterok:$prev_jid)"
    else
        jid=$(submit_one "$mode")
        echo "  $mode -> $jid"
    fi
    # track the main job id so joint/safety can depend on it when parallel
    if [ "$mode" = "main" ]; then
        jid_main="$jid"
    fi
    prev_jid="$jid"
done

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

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Resources used"
echo "═══════════════════════════════════════════════════════════════"
echo "  Heavy jobs  : $SELECT  walltime=$WALLTIME  accelerator=$ACCELERATOR"
echo "  Joint model : $SELECT_JOINT  walltime=$JOINT_WALLTIME  (no GPU)"
echo "  Analysis    : $SELECT_ANALYSIS  walltime=$ANALYSIS_WALLTIME  (no GPU)"
echo ""
echo "  Job graph"
echo "  ---------"
if [ "${#MODES[@]}" -gt 0 ]; then
    echo "  ${MODES[*]}  (${SUBMIT_MODE}, GA_ESTIMATOR=${GA_ESTIMATOR})"
fi
if [ "$INCLUDE_JOINT" -eq 1 ]; then
    echo "  main ──► joint_model  (Tobit-I + Heckman comparison)"
fi
if [ "$INCLUDE_ESTIMATOR_VARIANTS" -eq 1 ]; then
    echo "  main [GA_ESTIMATOR=metacount]     (independent)"
    echo "  main [GA_ESTIMATOR=searchlibrium] (independent)"
    echo "  main [GA_ESTIMATOR=both]          (independent)"
fi
if [ "$INCLUDE_SAFETY" -eq 1 ]; then
    echo "  safety_baseline ──► safety_nosafety ──► safety_analysis"
fi
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
        echo "  runs/main_${est}__<JOBID>/  full pipeline run with GA_ESTIMATOR=${est}"
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
if [ "$INCLUDE_JOINT" -eq 1 ]; then
    echo "  tail -f saw_joint.out"
fi
if [ "$INCLUDE_SAFETY" -eq 1 ]; then
    echo "  tail -f saw_safety_base.out   saw_safety_base.err"
    echo "  tail -f saw_safety_nosa.out   saw_safety_nosa.err"
    echo "  tail -f saw_safety_anal.out   saw_safety_anal.err"
fi
echo "═══════════════════════════════════════════════════════════════"
