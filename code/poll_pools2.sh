#!/bin/bash
BASE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
for i in 1 2 3 4 5 6 7 8; do
  f="$BASE/ga_s5_destfix_poolfix.o24998793"
  if [ -f "$f" ]; then
    echo "=== stdout exists, pool lines: ==="
    grep -E 'Candidate pools|mode_cands|dest_cands|valid segments|Preparing HHTS|running .*segment|pool=' "$f" | head -30
    echo "=== tail of stdout ==="
    tail -5 "$f"
    break
  fi
  echo "poll $i: no stdout file yet (elapsed $((i*120))s)"
  sleep 120
done
echo "=== timings ==="
cat "$BASE/runs/ga_parallel/destfix_poolfix/stage5/timings.csv" 2>/dev/null
echo "=== ga_reports ==="
ls -la "$BASE/runs/ga_parallel/destfix_poolfix/stage5/ga_reports/" 2>/dev/null