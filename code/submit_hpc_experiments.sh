#!/bin/bash
# =============================================================================
# HPC Job Submission Script for PYRAM Framework Experiments
# =============================================================================
# Submits multiple experiments with different model configurations to test
# the framework with hazard-based activity generation and various model setups.
#
# Usage:
#   bash submit_hpc_experiments.sh [EXPERIMENT_NAME] [NUM_RUNS]
#
# Examples:
#   bash submit_hpc_experiments.sh baseline 1
#   bash submit_hpc_experiments.sh hazard_nl_mxl 3
#   bash submit_hpc_experiments.sh full_sweep 5

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENT_NAME="${1:-baseline}"
NUM_RUNS="${2:-1}"
PROJECT_ROOT="/mnt/hpccs01/home/ahernz/test_runs_tours"
CODE_DIR="${PROJECT_ROOT}/code"
PYRAM_DIR="${PROJECT_ROOT}/pyram"
RUN_ROOT="${PROJECT_ROOT}/runs"
CONDA_ENV="zigenv"

# Experiment configurations
declare -A EXPERIMENTS=(
    # Baseline: MNL destination, NL mode, no DTA
    ["baseline"]="PYRAM_DESTINATION__MODEL_TYPE=mnl PYRAM_MODE__MODEL_TYPE=nl PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=cross_classification"
    
    # Hazard baseline: Competing hazard activity generation
    ["hazard_baseline"]="PYRAM_DESTINATION__MODEL_TYPE=mnl PYRAM_MODE__MODEL_TYPE=nl PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true"
    
    # Hazard + NL Distance destination
    ["hazard_nl_dist"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=nl PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1"
    
    # Hazard + Mixed Logit mode
    ["hazard_mxl"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1"
    
    # Hazard + GSE Mixed Logit
    ["hazard_gse"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1"
    
    # Full hazard + DTA feedback
    ["hazard_full_dta"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=true PYRAM_DTA__ITERATIONS=3 PYRAM_DTA__ENGINE=matsim PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1 DTA_FEEDBACK_ITERS=3 DTA_CONVERGENCE_THRESHOLD=0.02"
    
    # Hazard with Weibull hazard
    ["hazard_weibull"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=weibull PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1"
    
    # Hazard with Exponential hazard
    ["hazard_exp"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=exponential PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1"
    
    # Hazard + Trip-level destination enhancement
    ["hazard_trip_dest"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1 STAGE5_TRIP_DEST_ENHANCE=1"
    
    # Hazard + Parking + DTA
    ["hazard_parking_dta"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=true PYRAM_DTA__ITERATIONS=3 PYRAM_DTA__ENGINE=matsim PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1 STAGE7_PARKING=1 DTA_FEEDBACK_ITERS=3 DTA_CONVERGENCE_THRESHOLD=0.02 STAGE7_PARKING=1"
    
    # Sensitivity: Different time bins
    ["hazard_48bins"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=48 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1"
    
    ["hazard_192bins"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=192 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1"
    
    # Sensitivity: Different time spans
    ["hazard_12h"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__MAX_TIME_HOURS=12 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1"
    
    ["hazard_48h"]="PYRAM_DESTINATION__MODEL_TYPE=nl_distance PYRAM_MODE__MODEL_TYPE=mxl_gse PYRAM_MODE__MIXED_LOGIT=true PYRAM_DTA__ENABLE=false PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant PYRAM_ACTIVITY_GENERATION__TIME_BINS=96 PYRAM_ACTIVITY_GENERATION__MAX_TIME_HOURS=48 PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true STAGE5_DEST_NL_DISTANCE=1 STAGE5_ENABLE_MXL=1 STAGE5_ENABLE_GSE=1"
)

# Resource configurations
declare -A RESOURCES=(
    ["baseline"]="STAGE_NCPUS=4 STAGE_MEM=128GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=128GB PRED_WALLTIME=24:00:00"
    ["hazard_baseline"]="STAGE_NCPUS=4 STAGE_MEM=128GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=128GB PRED_WALLTIME=24:00:00"
    ["hazard_nl_dist"]="STAGE_NCPUS=4 STAGE_MEM=128GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=128GB PRED_WALLTIME=24:00:00"
    ["hazard_mxl"]="STAGE_NCPUS=8 STAGE_MEM=256GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=256GB PRED_WALLTIME=24:00:00"
    ["hazard_gse"]="STAGE_NCPUS=8 STAGE_MEM=256GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=256GB PRED_WALLTIME=24:00:00"
    ["hazard_full_dta"]="STAGE_NCPUS=16 STAGE_MEM=512GB STAGE_WALLTIME=48:00:00 PRED_NCPUS=16 PRED_MEM=512GB PRED_WALLTIME=48:00:00 DTA_FEEDBACK_WALLTIME=48:00:00"
    ["hazard_weibull"]="STAGE_NCPUS=8 STAGE_MEM=256GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=256GB PRED_WALLTIME=24:00:00"
    ["hazard_exp"]="STAGE_NCPUS=8 STAGE_MEM=256GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=256GB PRED_WALLTIME=24:00:00"
    ["hazard_trip_dest"]="STAGE_NCPUS=8 STAGE_MEM=256GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=256GB PRED_WALLTIME=24:00:00"
    ["hazard_parking_dta"]="STAGE_NCPUS=16 STAGE_MEM=512GB STAGE_WALLTIME=48:00:00 PRED_NCPUS=16 PRED_MEM=512GB PRED_WALLTIME=48:00:00 DTA_FEEDBACK_WALLTIME=48:00:00"
    ["hazard_48bins"]="STAGE_NCPUS=4 STAGE_MEM=128GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=128GB PRED_WALLTIME=24:00:00"
    ["hazard_192bins"]="STAGE_NCPUS=8 STAGE_MEM=256GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=256GB PRED_WALLTIME=24:00:00"
    ["hazard_12h"]="STAGE_NCPUS=4 STAGE_MEM=128GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=128GB PRED_WALLTIME=24:00:00"
    ["hazard_48h"]="STAGE_NCPUS=8 STAGE_MEM=256GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=256GB PRED_WALLTIME=24:00:00"
)

# Default resources
DEFAULT_RESOURCES="STAGE_NCPUS=4 STAGE_MEM=128GB STAGE_WALLTIME=24:00:00 PRED_NCPUS=8 PRED_MEM=128GB PRED_WALLTIME=24:00:00"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

submit_job() {
    local exp_name="$1"
    local run_id="$2"
    local env_vars="$3"
    local resources="$4"
    
    local run_tag="${exp_name}_run${run_id}_$(date +%Y%m%d_%H%M%S)"
    local run_dir="${RUN_ROOT}/${run_tag}"
    
    log "Submitting experiment: ${exp_name}, run: ${run_id}"
    log "Run tag: ${run_tag}"
    log "Run dir: ${run_dir}"
    
    # Create run directory
    mkdir -p "${run_dir}"
    
    # Write experiment config
    cat > "${run_dir}/experiment_config.yaml" <<EOF
experiment: ${exp_name}
run_id: ${run_id}
run_tag: ${run_tag}
timestamp: $(date -Iseconds)
resources: ${resources}
env_vars: ${env_vars}
EOF
    
    # Build qsub command
    local qsub_cmd="qsub -N \"${exp_name}_${run_id}\""
    
    # Add environment variables
    local all_env="${env_vars}"
    if [[ -n "${resources}" ]]; then
        all_env="${all_env} ${resources}"
    fi
    all_env="${all_env} RUN_TAG=${run_tag} RUN_ROOT=${RUN_ROOT} CONDA_ENV=zigenv PYTHONPATH=${PYRAM_DIR}:${CODE_DIR}"
    
    qsub_cmd="${qsub_cmd} -v \"${all_env}\""
    
    # Add resource limits
    qsub_cmd="${qsub_cmd} -l \"select=1:ncpus=1:mem=2GB\" -l \"walltime=00:30:00\""
    
    qsub_cmd="${qsub_cmd} ${CODE_DIR}/job_submit_ga_parallel_chain.pbs"
    
    log "qsub command: ${qsub_cmd}"
    
    # Submit
    local job_id
    job_id=$(eval "${qsub_cmd}" | awk '{print $1}')
    
    if [[ -n "${job_id}" ]]; then
        log "Submitted job: ${job_id}"
        echo "${job_id}" > "${run_dir}/job_id.txt"
        echo "${exp_name}" > "${run_dir}/experiment_name.txt"
        echo "${run_id}" > "${run_dir}/run_id.txt"
    else
        log "ERROR: Failed to submit job"
        return 1
    fi
}

wait_for_jobs() {
    local job_ids=("$@")
    log "Waiting for ${#job_ids[@]} jobs to complete..."
    
    for job_id in "${job_ids[@]}"; do
        log "Waiting for job ${job_id}..."
        while qstat "${job_id}" >/dev/null 2>&1; do
            sleep 60
        done
        log "Job ${job_id} completed"
    done
}

check_results() {
    local exp_name="$1"
    local run_id="$2"
    local run_tag="${exp_name}_run${run_id}_*"
    
    local run_dirs=($(find "${RUN_ROOT}" -maxdepth 1 -type d -name "${run_tag}" 2>/dev/null))
    
    if [[ ${#run_dirs[@]} -eq 0 ]]; then
        log "ERROR: No run directory found for ${exp_name} run ${run_id}"
        return 1
    fi
    
    local run_dir="${run_dirs[0]}"
    
    # Check for required outputs
    local required_files=(
        "stage5_models.pkl"
        "stage5_models.html"
        "stage6_output.csv"
        "stage6_output.html"
        "merged/ga_results.pkl"
        "merged/ga_results.html"
    )
    
    log "Checking results in ${run_dir}..."
    
    for file in "${required_files[@]}"; do
        if [[ -f "${run_dir}/${file}" ]]; then
            log "  ✓ ${file}"
        else
            log "  ✗ ${file} (MISSING)"
        fi
    done
    
    # Check for stage-specific model files
    for stage in 1 2 3 4 5 6 7; do
        if [[ -f "${run_dir}/stage${stage}_models.pkl" ]]; then
            log "  ✓ stage${stage}_models.pkl"
        fi
        if [[ -f "${run_dir}/stage${stage}_models.html" ]]; then
            log "  ✓ stage${stage}_models.html"
        fi
    done
    
    # Check for activity generation outputs
    if [[ -f "${run_dir}/stage7_output.csv" ]]; then
        log "  ✓ stage7_output.csv (with parking)"
    fi
    if [[ -f "${run_dir}/activity_generation_report.html" ]]; then
        log "  ✓ activity_generation_report.html"
    fi
    if [[ -f "${run_dir}/activity_generation_model.pkl" ]]; then
        log "  ✓ activity_generation_model.pkl"
    fi
    
    # Check for DTA feedback outputs
    if [[ -d "${run_dir}/dta_feedback" ]]; then
        log "  ✓ dta_feedback directory"
        if [[ -f "${run_dir}/dta_feedback/convergence.json" ]]; then
            log "  ✓ dta_feedback/convergence.json"
        fi
        if [[ -f "${run_dir}/dta_feedback/convergence.html" ]]; then
            log "  ✓ dta_feedback/convergence.html"
        fi
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    local exp_name="${1:-baseline}"
    local num_runs="${2:-${NUM_RUNS}}"
    
    if [[ ! "${EXPERIMENTS[${exp_name}]+_}" ]]; then
        log "ERROR: Unknown experiment: ${exp_name}"
        log "Available experiments: ${!EXPERIMENTS[*]}"
        exit 1
    fi
    
    local env_vars="${EXPERIMENTS[${exp_name}]}"
    local resources="${RESOURCES[${exp_name}]:-${DEFAULT_RESOURCES}}"
    
    log "========================================="
    log "Starting experiment: ${exp_name}"
    log "Number of runs: ${num_runs}"
    log "Environment: ${env_vars}"
    log "Resources: ${resources}"
    log "========================================="
    
    # Change to code directory
    cd "${CODE_DIR}"
    
    # Load conda
    source ~/.bashrc
    conda activate "${CONDA_ENV}"
    
    # Verify environment
    python -c "import jax; print('JAX:', jax.__version__); import numpy; print('NumPy:', numpy.__version__)"
    
    # Submit jobs
    local job_ids=()
    for ((run=1; run<=num_runs; run++)); do
        submit_job "${exp_name}" "${run}" "${env_vars}" "${resources}"
        local job_id=$(cat "${RUN_ROOT}/${exp_name}_run${run}_*/job_id.txt" 2>/dev/null || echo "")
        if [[ -n "${job_id}" ]]; then
            job_ids+=("${job_id}")
        fi
        sleep 5  # Small delay between submissions
    done
    
    log "All jobs submitted. Job IDs: ${job_ids[*]}"
    
    # Wait for completion
    wait_for_jobs "${job_ids[@]}"
    
    # Check results
    log "========================================="
    log "All jobs completed. Checking results..."
    log "========================================="
    
    for ((run=1; run<=num_runs; run++)); do
        check_results "${exp_name}" "${run}"
    done
    
    log "========================================="
    log "Experiment ${exp_name} completed!"
    log "========================================="
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi