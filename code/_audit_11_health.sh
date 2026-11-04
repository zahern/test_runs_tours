#!/bin/bash
# READ-ONLY. Estimation-health indicator grep in GA job stdout/stderr for the
# lccm / full40 / vmap era jobs. Counts only, per file.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
PAT='NEVER MOVED|never moved|AT A BOUND|at a bound|zero gradient|not identified|stuck near-zero|reverting|sentinel|-1e\+09|-1000000000|non-finite|Traceback|LC refit|lc_refit|lccm|LCCM|MXL|mxl|MH |RWMH|metacount|converged'

for f in $(ls -1t "$CODE"/ga_s5*lccm3*.o* "$CODE"/ga_s5*full40*.o* "$CODE"/ga_s5*vmap*.o* "$CODE"/ga_shardmerge_lccm3*.o* 2>/dev/null); do
  base=$(basename "$f")
  case "$base" in *.e[0-9]*) tag="ERR";; *) tag="OUT";; esac
  [ "$(stat -c '%s' "$f")" -eq 0 ] && continue
  echo "##### $base ($tag, $(stat -c '%s' "$f") bytes)"
  grep -oE "$PAT" "$f" 2>/dev/null | sort | uniq -c | sort -rn | head -25
done