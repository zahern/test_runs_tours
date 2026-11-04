#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== vmap verify 26624923 final ==="
grep -E 'Exit_status|Finished:' job_vmap_verify.pbs.o26624923 2>/dev/null | tail -2
echo
echo "=== old lccm3 jobs status ==="
qstat -u ahernz 2>/dev/null | grep -E '2662035[234]|26620355' || echo 'completed/failed'
echo
echo "=== lccm3 stderr tails ==="
for s in work nonmand edu; do
  f=$(ls -t ga_s5${s}_qld_lccm3.e* 2>/dev/null | head -1)
  echo "--- $s ($f) ---"
  if [ -n "$f" ] && [ -f "$f" ]; then tail -5 "$f"; else echo "(no file)"; fi
done