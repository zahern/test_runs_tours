#!/bin/bash
# READ-ONLY. Correct exit table + env-var mapping for GA jobs.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code

printf '%-42s %-17s %-6s %-5s %-22s %-8s %-9s %s\n' FILE MTIME EXIT STAGE SEGMENT RUNTAG BUDGET LINES
for f in $(ls -1t "$CODE"/*.o[0-9]* 2>/dev/null | head -110); do
  base=$(basename "$f")
  case "$base" in *.e[0-9]*) continue;; esac
  mt=$(stat -c '%y' "$f" 2>/dev/null | cut -d. -f1 | cut -c1-16)
  ex=$(grep -h -oE 'Exit_status +: +[0-9]+' "$f" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  [ -z "$ex" ] && ex="RUNNING"
  sub=$(grep -h -m1 -oE 'Submit_arguments +: .*' "$f" 2>/dev/null)
  st=$(printf '%s' "$sub" | grep -oE 'STAGE_ID=[^ ,]*' | cut -d= -f2)
  [ -z "$st" ] && st="-"
  seg=$(printf '%s' "$sub" | grep -oE 'SHARD_SEGMENT=[^ ,]*' | cut -d= -f2)
  [ -z "$seg" ] && seg="-"
  rt=$(printf '%s' "$sub" | grep -oE 'RUN_TAG=[^ ,]*' | cut -d= -f2)
  [ -z "$rt" ] && rt="-"
  bg=$(printf '%s' "$sub" | grep -oE 'GA_BUDGET=[^ ,]*' | cut -d= -f2)
  [ -z "$bg" ] && bg="-"
  nl=$(wc -l < "$f" 2>/dev/null)
  printf '%-42s %-17s %-6s %-5s %-22s %-8s %-9s %s\n' "$base" "$mt" "$ex" "$st" "$seg" "$rt" "$bg" "$nl"
done