#!/bin/bash
# READ-ONLY audit. Extract Exit_status / Finished lines from PBS .o* files.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code

echo "=== ALL .o* FILES: mtime + Exit_status + Finished ==="
for f in $(ls -1t "$CODE"/*.o* 2>/dev/null); do
  base=$(basename "$f")
  mt=$(stat -c '%y' "$f" 2>/dev/null | cut -d. -f1)
  # only the .o files (job stdout), not .e
  case "$base" in *.e[0-9]*) continue;; esac
  ex=$(grep -h -E '^\s*Exit_status' "$f" 2>/dev/null | head -2 | tr '\n' ' ')
  fin=$(grep -h -E 'Finished:' "$f" 2>/dev/null | head -2 | tr '\n' ' ')
  printf '%s | mtime=%s | %s | %s\n' "$base" "$mt" "$ex" "$fin"
done

echo
echo "=== .e FILES with content (stderr) ==="
for f in $(ls -1t "$CODE"/*.e[0-9]* 2>/dev/null); do
  base=$(basename "$f")
  sz=$(stat -c '%s' "$f" 2>/dev/null)
  mt=$(stat -c '%y' "$f" 2>/dev/null | cut -d. -f1)
  if [ "$sz" -gt 0 ]; then
    printf '--- %s | mtime=%s | bytes=%s\n' "$base" "$mt" "$sz"
    tail -20 "$f"
  fi
done