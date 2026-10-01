#!/bin/bash
for i in 1 2 3 4 5 6; do
  f=$(ls /mnt/hpccs01/home/ahernz/test_runs_tours/code/ga_s5_destfix_poolfix.o* 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    echo "logfile: $f"
    grep -E 'Candidate pools|mode_cands|dest_cands|valid segments|Stage 5: running' "$f" | head -20
    break
  fi
  echo "poll $i: no stdout yet"
  sleep 180
done
ls /mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/destfix_poolfix/stage5/ga_reports 2>/dev/null