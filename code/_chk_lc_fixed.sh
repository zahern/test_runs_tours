#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== NEW LC SWEEP (lccm3_fixed) ==="
qstat -u ahernz 2>/dev/null | grep -E '2663730[34567]|Job ID|----' || echo "none in queue"
echo
for j in 26637303 26637305 26637306 26637307; do
  o=$(ls -t ga_s5*fixed.o$j ga_shardmerge_fixed.o$j 2>/dev/null | head -1)
  e=$(ls -t ga_s5*fixed.e$j ga_shardmerge_fixed.e$j 2>/dev/null | head -1)
  echo "--- job $j ---"
  if [ -n "$o" ]; then
    echo "  stdout: $o ($(wc -l < $o) lines)"
    grep -E 'Exit_status|Finished:' $o 2>/dev/null | tail -2 || echo "  (still running)"
  else
    echo "  stdout: (not started)"
  fi
  if [ -n "$e" ] && [ -s "$e" ]; then
    echo "  stderr tail:"
    tail -12 $e | sed 's/^/    /'
  fi
  echo
done