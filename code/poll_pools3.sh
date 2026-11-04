#!/bin/bash
BASE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
for i in $(seq 1 10); do
  f="$BASE/ga_s5_destfix_poolfix.o24998793"
  if [ -f "$f" ] && [ -s "$f" ]; then
    echo "=== stdout appeared (iter $i) ==="
    grep -E 'Candidate pools|mode_cands|dest_cands|valid segments|Preparing HHTS|running .*segment|pool=' "$f" | head -30
    echo "=== tail ==="
    tail -5 "$f"
    break
  fi
  echo "poll $i: no stdout yet (elapsed $((i*180))s)"
  sleep 180
done
echo "=== timings ==="
cat "$BASE/runs/ga_parallel/destfix_poolfix/stage5/timings.csv" 2>/dev/null
echo "=== qstat ==="
qstat -u ahernz | grep 24998793