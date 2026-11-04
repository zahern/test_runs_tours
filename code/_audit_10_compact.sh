#!/bin/bash
# READ-ONLY. Compact health summary per run tag / segment.
RUNS=/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel

for tag in lccm3_fixed qld_lccm3 qld_lccm2 qld_lccm qld_full40 vmap_e1_pilot; do
  d="$RUNS/$tag"; [ -d "$d" ] || continue
  echo "===== $tag"
  for stage in stage5 stage51 stage4 stage1 stage0 stage2 stage3; do
    [ -d "$d/$stage/shards" ] || continue
    for sd in "$d/$stage/shards"/*; do
      seg=$(basename "$sd")
      gs="$sd/ga_reports/ga_summary.csv"
      lb="$sd/variants/leaderboard.csv"
      line="$stage/$seg"
      if [ -f "$gs" ]; then
        g=$(tail -1 "$gs" | awk -F, '{print "gens="$5" restarts="$6" cache="$7" nfeat="$8" best="$3}')
        line="$line [$g]"
      else line="$line [NO ga_summary]"; fi
      if [ -f "$lb" ]; then
        conv=$(tail -n +2 "$lb" | awk -F, '$7==1' | wc -l)
        tot=$(tail -n +2 "$lb" | wc -l)
        fails=$(tail -n +2 "$lb" | awk -F, '$7!=1 {print $4"/"$3"="$13}' | tr '\n' ' ')
        llmiss=$(tail -n +2 "$lb" | awk -F, '$8=="" {c++} END{print c+0}')
        kmiss=$(tail -n +2 "$lb" | awk -F, '$9=="" {c++} END{print c+0}')
        line="$line lb:$conv/$tot converged, ll_missing=$llmiss k_missing=$kmiss FAILS[$fails]"
      else line="$line [NO leaderboard]"; fi
      [ -f "$sd/convergence_trace.csv" ] && line="$line trace_rows=$(tail -n +2 "$sd/convergence_trace.csv" | wc -l)"
      echo "  $line"
    done
  done
done