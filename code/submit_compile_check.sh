#!/bin/bash
# Submit the syntax gate for the 2026-10-13 latent-class / logsum edits and
# wait for it. Run from the cluster (login node is fine -- this only calls qsub;
# the Python runs inside the batch job).
#
#   bash submit_compile_check.sh
#
set -uo pipefail
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code || exit 1

PBS_OUT=$(qsub job_compile_check.pbs)
echo "submitted: $PBS_OUT"
JOBID="${PBS_OUT##* }"

# PBS may return "<id>.aqua" or "<id>.<server>"
JOBID="${JOBID%%.*}"
echo "jobid: $JOBID"
echo

# Poll up to ~9 minutes (job walltime is 10 min).
for i in $(seq 1 54); do
  if ! qstat "$JOBID" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

echo
echo "-------- output --------"
cat "compile_check.out" 2>/dev/null || {
  echo "no compile_check.out yet -- job still queued?"
  qstat "$JOBID" 2>/dev/null || true
}
echo "------------------------"