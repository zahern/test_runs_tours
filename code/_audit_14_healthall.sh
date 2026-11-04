#!/bin/bash
# READ-ONLY. (a) still-running jobs; (b) health-indicator occurrence counts
# across ALL recent GA stdout/stderr, attributed to files.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code

echo "=== qstat (currently running/queued) ==="
qstat -u ahernz 2>/dev/null | head -30 || echo "(qstat unavailable)"

echo
echo "=== staged shardmerge_lccm3fix / lccm3b env ==="
for b in ga_shardmerge_lccm3fix.o26639468 ga_shardmerge_lccm3b.o26639465; do
  echo "--- $b"
  grep -h -oE 'Exit_status +: +[0-9]+|RUN_TAG=[^ ,]*|Submit_arguments +: .*' "$CODE/$b" 2>/dev/null | head -4
  tail -20 "$CODE/$b" 2>/dev/null | grep -v -E '^\s+[a-z_]+ +:' | head -20
done

echo
echo "=== health indicator counts across ALL .o/.e newer than 2026-09-25 ==="
for f in $(ls -1t "$CODE"/*.o[0-9]* "$CODE"/*.e[0-9]* 2>/dev/null); do
  base=$(basename "$f")
  d=$(stat -c '%y' "$f" | cut -d' ' -f1)
  case "$d" in 2026-09-2[5-9]|2026-09-30|2026-10-*) ;; *) continue;; esac
  nm=$(grep -c 'NEVER MOVED' "$f" 2>/dev/null)
  ab=$(grep -c -i 'at a bound' "$f" 2>/dev/null)
  zg=$(grep -c -i 'zero gradient' "$f" 2>/dev/null)
  ni=$(grep -c -i 'not identified' "$f" 2>/dev/null)
  sd=$(grep -c -i 'stuck near-zero' "$f" 2>/dev/null)
  rv=$(grep -c -i 'reverting' "$f" 2>/dev/null)
  sn=$(grep -c -E 'sentinel|1000000000' "$f" 2>/dev/null)
  tb=$(grep -c 'Traceback' "$f" 2>/dev/null)
  tot=$((nm+ab+zg+ni+sd+rv+sn+tb))
  [ "$tot" -gt 0 ] && printf '%-46s NEVERMOVED=%s ATBOUND=%s ZEROGRAD=%s NOTIDENT=%s STUCK=%s REVERT=%s SENTINEL=%s TB=%s\n' "$base" "$nm" "$ab" "$zg" "$ni" "$sd" "$rv" "$sn" "$tb"
done