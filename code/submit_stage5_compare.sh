#!/bin/bash
# Submit isolated Stage 5 Larch and SearchLibrium comparison arms.
# Usage: ./submit_stage5_compare.sh [run_tag] [mh|gibbs|sgd]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_TAG="${1:-stage5_compare_$(date +%Y%m%d_%H%M%S)}"
SEARCHLIBRIUM_METHOD="${2:-mh}"
case "${SEARCHLIBRIUM_METHOD}" in
    mh|gibbs|sgd) ;;
    *) echo "ERROR: method must be mh, gibbs, or sgd" >&2; exit 2 ;;
esac

RUN_ROOT="${RUN_ROOT:-runs/stage5_compare}"
BASE_DIR="${RUN_ROOT}/${RUN_TAG}"
CONDA_ENV="${CONDA_ENV:-zigenv}"
COMMON="CONDA_ENV=${CONDA_ENV},RUN_ROOT=${RUN_ROOT},STAGE5_DEST_SAMPLE_SIZE=${STAGE5_DEST_SAMPLE_SIZE:-20},STAGE5_CONVERGENCE_MIN_ESS=${STAGE5_CONVERGENCE_MIN_ESS:-20}"

LARCH_ID="$(qsub -N s5_larch \
    -v "${COMMON},STAGE5_ARM=larch,STAGE5_DEST_ESTIMATOR=larch,STAGE5_COMPARE_DIR=${BASE_DIR}/larch" \
    job_stage5_compare.pbs | awk '{print $1}')"
SEARCH_ID="$(qsub -N s5_sl_${SEARCHLIBRIUM_METHOD} \
    -v "${COMMON},STAGE5_ARM=searchlibrium_${SEARCHLIBRIUM_METHOD},STAGE5_DEST_ESTIMATOR=${SEARCHLIBRIUM_METHOD},STAGE5_COMPARE_DIR=${BASE_DIR}/searchlibrium_${SEARCHLIBRIUM_METHOD}" \
    job_stage5_compare.pbs | awk '{print $1}')"

echo "Submitted Stage 5 comparison arms"
echo "  Larch                    : ${LARCH_ID} -> ${BASE_DIR}/larch"
echo "  SearchLibrium ${SEARCHLIBRIUM_METHOD} : ${SEARCH_ID} -> ${BASE_DIR}/searchlibrium_${SEARCHLIBRIUM_METHOD}"
echo "  Convergence reports      : ${BASE_DIR}/<arm>/stage5_convergence_report.json"
echo "  Final model bundles      : ${BASE_DIR}/<arm>/stage5_models.pkl"
