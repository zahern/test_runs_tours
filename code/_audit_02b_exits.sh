#!/bin/bash
# READ-ONLY audit. Clean exit-code table for PBS .o* files. No stderr dumps.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code

printf '%s | %s | %s | %s\n' FILE MTIME EXIT FINISHED
for f in $(ls -1t "$CODE"/*.o[0-9]* 2>/dev/null); do
  base=$(basename "$f")
  case "$base" in *.e[0-9]*) continue;; esac
  mt=$(stat -c '%y' "$f" 2>/dev/null | cut -d. -f1 | cut -c1-16)
  ex=$(grep -h -oE 'Exit_status:[ ]*[0-9]+' "$f" 2>/dev/null | tail -1 | tr -d ' ')
  fin=$(grep -h -oE 'Finished: [^ ]+ [^ ]+ [^ ]+' "$f" 2>/dev/null | tail -1)
  printf '%s | %s | %s | %s\n' "$base" "$mt" "$ex" "$fin"
done