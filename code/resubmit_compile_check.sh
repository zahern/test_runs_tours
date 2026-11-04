#!/bin/bash
# Resubmit the compile gate with a short walltime (better backfill position),
# cancelling the earlier queued copy.
set -uo pipefail
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code || exit 1
OLD="${1:-26739173}"

if [ -n "$OLD" ]; then
  qdel "$OLD" 2>&1 | head -2
  echo "cancelled $OLD"
fi

sed -i 's/^#PBS -l walltime=.*/#PBS -l walltime=00:03:00/' job_compile_check.pbs
grep '^#PBS' job_compile_check.pbs

NEW=$(qsub job_compile_check.pbs)
echo "submitted: $NEW"
echo "${NEW%%.*}" > .compile_check_jobid
cat .compile_check_jobid