#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== merge 26620355 (premature, expected failure) ==="
cat ga_shardmerge_qld_lccm3.e26620355 2>/dev/null | tail -12
echo
echo "=== lccm3 shard convergence traces ==="
for s in work nonmand edu; do
  t="runs/ga_parallel/qld_lccm3/stage5/shards/${s}/convergence_trace.csv"
  echo "--- ${s} ---"
  if [ -f "$t" ]; then tail -3 "$t"; else echo "(none yet)"; fi
done
echo
echo "=== vmap verify #2 (26624077) ==="
v=job_vmap_verify.pbs.e26624077
o=job_vmap_verify.pbs.o26624077
if [ -f "$v" ]; then
  grep -E '_logsums_vmap surface|census|NaN census|maxdiff|breach|self-verify' "$v" | tail -12
  echo "-- exit --"; grep -E 'Exit_status' "$o" 2>/dev/null || echo "(running)"
else
  echo "(no stderr yet)"
  qstat -u ahernz 2>/dev/null | grep 26624077 || echo "26624077 left queue"
fi