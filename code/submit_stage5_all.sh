#!/bin/bash
# submit_stage5_all.sh
# Submit the full QLD pipeline plus the Stage-5 multi-method comparison jobs.
# Run from the code dir on a login node:
#   bash submit_stage5_all.sh
# or from a working session:
#   ssh aquarius 'cd /home/ahernz/test_runs_tours/code && bash submit_stage5_all.sh'
set -euo pipefail
cd "$(dirname "$0")"

echo "== $(date) submitting full QLD pipeline (stages 0-5, 371 zones) =="
qsub job_qld_stages05.pbs

echo "== submitting Stage-5 four-arm method comparison =="
echo "   (larch MNL / SearchLibrium MNL / larch latent class / SearchLibrium latent class)"
qsub -v PAPER_N_CASES=0,PAPER_N_PERT=3,PAPER_LC_MAXITER=300 job_stage5_paper_arms.pbs

echo "== submitting Stage-5 larch vs SearchLibrium full-spec comparison =="
qsub -v STAGE5_ARM=searchlibrium_mh_full,STAGE5_DEST_ESTIMATOR=mh,STAGE5_DEST_FULL_SPEC=1,STAGE5_DEST_MH_DRAWS=32000,STAGE5_DEST_MH_BURN_IN=6400,STAGE5_REQUIRE_LOGSUM_VARIANCE=1 job_stage5_compare.pbs

# Optional crash-rate arms (uncomment to include):
# qsub -v ARM=crash_rate,ESTIMATOR=larch job_stage5_larch.pbs
# qsub -v ARM=crash_rate,ESTIMATOR=mh    job_stage5_larch.pbs

echo "== queue =="
qstat -u "$USER" || true
