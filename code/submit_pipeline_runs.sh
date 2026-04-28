#!/bin/bash
# Submit one or more pipeline modes to PBS with cleaner resource control.

set -euo pipefail

MODES=(main ga ga_staged)
CPUS=2
THREADS=1
S5_WORKERS=1
MEM=194GB
WALLTIME=20:59:00
CONDA_ENV=zigenv
ZONE_ARG=
SUBMIT_MODE=parallel
GPU_COUNT=0
RUN_ROOT=runs
PBS_SCRIPT=pbs_pipeline_mode.pbs
ACCELERATOR=cpu
INCLUDE_SAFETY=0

usage() {
cat <<'EOF'
Usage:
    ./submit_pipeline_runs.sh [options]

Options:
    --modes "main ga ga_staged"   Space-separated modes to submit.
    --cpus N                       PBS ncpus request per job.
    --threads N                    OMP/MKL/OpenBLAS thread count inside job.
    --workers N                    Stage 5 zone workers (fork processes). Default 1.
                                   Set equal to --cpus for full parallelism.
                                   e.g. --cpus 8 --workers 8 cuts ~14h → ~2h.
    --mem SIZE                     PBS memory request per job.
    --walltime HH:MM:SS            PBS walltime per job.
    --zone ARG                     Optional pipeline_logger zone/file argument.
    --env NAME                     Conda environment to activate on the cluster.
    --run-root PATH                Base output directory.
    --parallel                     Submit all jobs independently.
    --sequential                   Submit jobs with afterok dependencies.
    --include-safety               Also submit safety_baseline, safety_nosafety,
                                                                 and safety_analysis.
    --accelerator cpu|gpu          JAX platform hint passed into the job.
    --gpu N                        Request GPUs. Default is 0.
    --help                         Show this message.

Examples:
    # Basic CPU-only run (default):
    ./submit_pipeline_runs.sh

    # GPU estimation + 8 CPU prediction workers (~3-4h total vs ~15h serial):
    ./submit_pipeline_runs.sh --gpu 1 --accelerator gpu --cpus 8 --workers 8 \
        --mem 250GB --walltime 05:00:00

    # CPU-only with prediction parallelism (no GPU node required):
    ./submit_pipeline_runs.sh --cpus 8 --workers 8 --mem 250GB --walltime 05:00:00

    # Sequential modes sharing a GPU:
    ./submit_pipeline_runs.sh --modes "main ga" --sequential \
        --gpu 1 --accelerator gpu --cpus 8 --workers 8 --mem 250GB

Notes:
    - GPU (--accelerator gpu) speeds up model estimation via JAX/Larch.
      Prediction workers automatically use CPU only — no GPU contention.
    - Use --workers N to run Stage 5 zone prediction in parallel (N fork processes).
      Set --cpus = --workers; each worker needs ~6-8 GB RAM.
      Rule of thumb: --mem = 100GB base + (workers × 8GB).
      With 8 workers the ~14h Stage 5 drops to roughly 14h/8 ≈ 2h wall-clock.
    - --threads controls BLAS/numba threading inside each process.
      Leave at 1 unless profiling shows BLAS is a bottleneck (it usually isn't).
    - safety_analysis is a post-processing script, not a pipeline_logger mode.
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

submit_one() {
    local mode="$1"
    local depend="${2:-}"
    local job_name="saw_${mode}"
    local vars
    vars="MODE=${mode},ZONE_ARG=${ZONE_ARG},CONDA_ENV=${CONDA_ENV},THREADS=${THREADS},S5_WORKERS=${S5_WORKERS},RUN_ROOT=${RUN_ROOT},PIPELINE_ACCELERATOR=${ACCELERATOR}"

    if [ -n "$depend" ]; then
        qsub -N "$job_name" -l walltime="$WALLTIME" -l "$SELECT" -W "depend=afterok:${depend}" -v "$vars" "$PBS_SCRIPT"
    else
        qsub -N "$job_name" -l walltime="$WALLTIME" -l "$SELECT" -v "$vars" "$PBS_SCRIPT"
    fi
}

echo "Submitting modes: ${MODES[*]}"
echo "  submit_mode = $SUBMIT_MODE"
echo "  select      = $SELECT"
echo "  walltime    = $WALLTIME"
echo "  threads     = $THREADS"
echo "  s5_workers  = $S5_WORKERS"
echo "  accelerator = $ACCELERATOR"
echo "  zone_arg    = ${ZONE_ARG:-<all>}"
echo "  conda_env   = $CONDA_ENV"
echo "  include_safety = $INCLUDE_SAFETY"

submit_safety_chain() {
    local base_depend="${1:-}"
    local jid_base
    local jid_nosafety
    local jid_analysis

    if [ -n "$base_depend" ]; then
        jid_base=$(submit_one safety_baseline "$base_depend")
        echo "  safety_baseline -> $jid_base (afterok:$base_depend)"
    else
        jid_base=$(submit_one safety_baseline)
        echo "  safety_baseline -> $jid_base"
    fi

    jid_nosafety=$(submit_one safety_nosafety "$jid_base")
    echo "  safety_nosafety -> $jid_nosafety (afterok:$jid_base)"

    jid_analysis=$(submit_one safety_analysis "$jid_nosafety")
    echo "  safety_analysis -> $jid_analysis (afterok:$jid_nosafety)"

    prev_jid="$jid_analysis"
}

prev_jid=
for mode in "${MODES[@]}"; do
    if [ "$SUBMIT_MODE" = sequential ] && [ -n "$prev_jid" ]; then
        jid=$(submit_one "$mode" "$prev_jid")
        echo "  $mode -> $jid (afterok:$prev_jid)"
    else
        jid=$(submit_one "$mode")
        echo "  $mode -> $jid"
    fi
    prev_jid="$jid"
done

if [ "$INCLUDE_SAFETY" -eq 1 ]; then
    if [ "$SUBMIT_MODE" = sequential ]; then
        submit_safety_chain "$prev_jid"
    else
        submit_safety_chain
    fi
fi

echo ""
echo "Monitor with: qstat -u \$USER"
echo "Run outputs will be under ${RUN_ROOT}/<mode>/"
echo ""
echo "To follow live output in real time (once job starts running):"
for mode in "${MODES[@]}"; do
    echo "  tail -f ${RUN_ROOT}/${mode}__<JOBID>/pipeline_run.log"
done
if [ "$INCLUDE_SAFETY" -eq 1 ]; then
    for safety_mode in safety_baseline safety_nosafety safety_analysis; do
        echo "  tail -f ${RUN_ROOT}/${safety_mode}__<JOBID>/pipeline_run.log"
    done
fi
echo ""
echo "PBS output (.e/.o) files appear in this directory as saw_<mode>.e<JOBID>"
echo "and contain merged stdout+stderr — same content as the pipeline_run.log above."