#!/bin/bash
# READ-ONLY. Dump convergence evidence for stage5-era run tags.
RUNS=/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel

for tag in lccm3_fixed qld_lccm3 qld_lccm2 qld_lccm qld_full40 vmap_e1_pilot; do
  d="$RUNS/$tag"
  [ -d "$d" ] || continue
  echo "##################### $tag"
  for seg in work nonmand edu; do
    for f in convergence_trace.csv ga_reports/ga_summary.csv variants/leaderboard.csv; do
      p="$d/stage5/shards/$seg/$f"
      if [ -f "$p" ]; then
        echo "----- $seg/$f"
        cat "$p"
        echo
      fi
    done
    for f in variants/winner.json variants/winner_spec_mode.json variants/winner_spec_dest.json; do
      p="$d/stage5/shards/$seg/$f"
      [ -f "$p" ] && { echo "----- $seg/$f"; cat "$p"; echo; }
    done
  done
  for f in merged/ga_reports/ga_summary.csv merged/variant_board/leaderboard.csv merged/variant_board/winner.json merged/shard_leaderboard.csv; do
    p="$d/$f"
    [ -f "$p" ] && { echo "----- MERGED $f"; cat "$p"; echo; }
  done
  echo
done