#!/bin/bash
BASE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
JID=24998793
for i in $(seq 1 12); do
  st=$(qstat -u ahernz | grep "$JID" | awk '{print $10}')
  echo "poll $i: state=${st:-GONE} at $(date +%H:%M)"
  if [ "$st" != "R" ]; then
    echo "=== job finished ==="
    f="$BASE/ga_s5_destfix_poolfix.o$JID"
    ls -la "$f" 2>/dev/null
    grep -E 'Candidate pools|HHTS tours|mode_cands|dest_cands' "$f" 2>/dev/null | head -20
    echo "=== ga_reports ==="
    ls -la "$BASE/runs/ga_parallel/destfix_poolfix/stage5/ga_reports/" 2>/dev/null
    echo "=== timings ==="
    cat "$BASE/runs/ga_parallel/destfix_poolfix/stage5/timings.csv" 2>/dev/null
    break
  fi
  sleep 900
done