#!/bin/bash
# READ-ONLY audit. Lists run tags, mtimes, and PBS job output files.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
RUNS=$CODE/runs/ga_parallel

echo "=== TOP-LEVEL OF runs/ga_parallel (by mtime, oldest last) ==="
ls -1t --time-style=long-iso "$RUNS" 2>/dev/null | head -60

echo
echo "=== RUN TAGS with mtime and dir size ==="
for d in $(ls -1 "$RUNS" 2>/dev/null); do
  if [ -d "$RUNS/$d" ]; then
    sz=$(du -sh "$RUNS/$d" 2>/dev/null | cut -f1)
    mt=$(stat -c '%y' "$RUNS/$d" 2>/dev/null | cut -d. -f1)
    nf=$(find "$RUNS/$d" -type f 2>/dev/null | wc -l)
    printf '%s | mtime=%s | size=%s | files=%s\n' "$d" "$mt" "$sz" "$nf"
  fi
done

echo
echo "=== PBS OUTPUT FILES (.o*) in code dir ==="
ls -1t --time-style=long-iso "$CODE" 2>/dev/null | grep -E '\.o[0-9]+$' | head -60

echo
echo "=== PBS SCRIPTS in code dir ==="
ls -1t --time-style=long-iso "$CODE" 2>/dev/null | grep -E '\.pbs$|\.sh$' | head -60

echo
echo "=== TOP-LEVEL PY in code dir ==="
ls -1t --time-style=long-iso "$CODE" 2>/dev/null | grep -E '\.py$' | head -80