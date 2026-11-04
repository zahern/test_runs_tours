#!/bin/bash
# READ-ONLY. Artifact inventory for the QLD-era run tags.
RUNS=/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel

for tag in qld_lccm qld_lccm2 qld_lccm3 qld_full40 lccm3_fixed vmap_e1_pilot qld_thorough qld_refit_20260928 cdap_s5_refit_20260928 paper_missing_20260929; do
  d="$RUNS/$tag"
  echo "##################### $tag"
  [ -d "$d" ] || { echo "(missing)"; continue; }
  echo "-- stage/segment dirs:"
  find "$d" -maxdepth 2 -type d 2>/dev/null | sed "s|$RUNS/||" | sort | head -40
  echo "-- key artifact filenames (basename | relpath | bytes | mtime):"
  find "$d" -type f \( -name '*leaderboard*' -o -name '*convergence*' -o -name '*summary*' -o -name 'winner*' -o -name '*_fit.json' -o -name '*result*.json' -o -name '*health*' -o -name '*diagnos*' \) 2>/dev/null \
    | while read -r p; do
        printf '%s | %s | %s | %s\n' "$(basename "$p")" "${p#$d/}" "$(stat -c '%s' "$p")" "$(stat -c '%y' "$p" | cut -d. -f1)"
      done | sort | head -70
  echo
done