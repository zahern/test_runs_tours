#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== queue ==="
qstat -u ahernz 2>/dev/null | grep -E '26624923|2662035[234]|Job ID|----' || echo 'none'
echo
echo "=== lccm3 convergence ==="
for s in work nonmand edu; do
  t=runs/ga_parallel/qld_lccm3/stage5/shards/\$s/convergence_trace.csv
  echo "--- \$s ---"
  if [ -f "\$t" ]; then tail -2 "\$t"; else echo "(none)"; fi
done
echo
echo "=== LC refit diagnostics ==="
grep -hE 'stage5-refit' ga_s5*_qld_lccm3.e* 2>/dev/null | tail -8 || echo "(none)"