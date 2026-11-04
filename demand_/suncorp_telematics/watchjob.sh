#!/bin/bash
# Usage: bash watchjob.sh <JOBID>[.aqua] [log-prefix]
JOB=$1
NAME=${2:-suncorp_postselp}
LOG=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics/${NAME}.o${JOB%%.*}
for i in $(seq 1 120); do
  S=$(qstat -f "$JOB" 2>/dev/null | grep 'job_state =' | awk '{print $3}')
  if [ -z "$S" ]; then echo "FINAL state=GONE (completed or purged)"; break; fi
  if [ "$S" != "Q" ] && [ "$S" != "R" ] && [ "$S" != "H" ]; then echo "FINAL state=$S"; break; fi
  sleep 20
done
echo "--- qstat ---"
qstat -f "$JOB" 2>/dev/null | grep -E 'job_state|Exit_status|resources_used' || echo "(no qstat entry)"
echo "--- output tail ($NAME) ---"
tail -n 30 "$LOG" 2>/dev/null || echo "(no output file yet)"
