#!/bin/bash
# Poll one PBS job until it leaves the queue, then dump its output.
#   bash poll_one_job.sh <jobid> [max_seconds]
set -uo pipefail
JOB="${1:?jobid}"
MAX="${2:-420}"
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
cd "$CODE" || exit 1

state() { qstat -f "$JOB" 2>/dev/null | sed -n 's/^ *job_state = //p'; }

echo "polling $JOB (max ${MAX}s)"
start=$(date +%s)
last=""
while :; do
  now=$(date +%s)
  el=$(( now - start ))
  st=$(state)
  if [ -z "$st" ]; then
    echo "[${el}s] job no longer in queue -> finished"
    break
  fi
  if [ "$st" != "$last" ]; then
    echo "[${el}s] job_state=$st"
    last="$st"
  fi
  if [ "$st" = "F" ]; then
    echo "[${el}s] completed (F)"
    break
  fi
  if [ "$el" -ge "$MAX" ]; then
    echo "[${el}s] still $st - giving up this round (job keeps running)"
    break
  fi
  sleep 15
done

echo
echo "======== compile_check.out ========"
cat compile_check.out 2>/dev/null || echo "(no output file yet)"
echo "===================================="