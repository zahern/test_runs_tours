#!/bin/bash
# Report accurate state for the compile-check jobs.
#   bash compile_status.sh 26740325 26740327 26739929
set -uo pipefail
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code || exit 1

for j in "$@"; do
  line=$(qstat -f "$j" 2>/dev/null | grep 'job_state' | head -1)
  if [ -z "$line" ]; then
    echo "$j : not in queue (finished or unknown)"
  else
    st=${line##*= }
    cmt=$(qstat -f "$j" 2>/dev/null | grep -m1 'comment =' | cut -c1-70)
    echo "$j : state=$st  $cmt"
  fi
done

echo
echo "-------- output files --------"
for f in compile_check.out compile_check.o26740325 compile_check.o26740327 compile_check.o26739929; do
  if [ -f "$f" ]; then
    echo "== $f =="
    cat "$f"
    echo
  fi
done
ls -1 compile_check.* 2>/dev/null