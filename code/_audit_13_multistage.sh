#!/bin/bash
# READ-ONLY. Multi-stage run cdap_s5_refit_20260928 + paper_missing_20260929:
# per-stage/segment convergence + leaderboards.
RUNS=/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel

for tag in cdap_s5_refit_20260928 paper_missing_20260929 qld_refit_20260928 qld_thorough; do
  d="$RUNS/$tag"; [ -d "$d" ] || continue
  echo "################ $tag"
  for stage in stage0 stage1 stage2 stage3 stage4 stage5 stage51; do
    [ -d "$d/$stage/shards" ] || continue
    for sd in "$d/$stage/shards"/*; do
      seg=$(basename "$sd")
      gs="$sd/ga_reports/ga_summary.csv"
      lb="$sd/variants/leaderboard.csv"
      tr="$sd/convergence_trace.csv"
      line="  $stage/$seg"
      if [ -f "$gs" ]; then
        # pull best_fitness, final, generations, restarts, cache, nfeat
        line="$line $(tail -1 "$gs" | awk -F, '{print "best="$3" gens="$5" restarts="$6" cache="$7" nfeat="$8}')"
      else line="$line [NO ga_summary]"; fi
      if [ -f "$lb" ]; then
        line="$line | lb converged=$(tail -n +2 "$lb" | awk -F, '$7==1' | wc -l)/$(tail -n +2 "$lb" | wc -l)"
        line="$line llmissing=$(tail -n +2 "$lb" | awk -F, '$8==""{c++}END{print c+0}')"
        line="$line | FAILS: $(tail -n +2 "$lb" | awk -F, '$7!=1 {print $4"("$2"/"$3")="$13}' | tr '\n' ' ')"
      else line="$line | [NO leaderboard]"; fi
      [ -f "$tr" ] && line="$line | trace=$(tail -n +2 "$tr" | wc -l)rows"
      echo "$line"
    done
  done
  for f in merged/ga_reports/ga_summary.csv merged/variant_board/leaderboard.csv; do
    p="$d/$f"
    if [ -f "$p" ]; then
      echo "  ---- MERGED $f ($(stat -c '%y' "$p" | cut -d. -f1))"
      head -1 "$p"
      tail -n +2 "$p" | awk -F, '{print "    "$1","$2","$3","$4",conv="$7",bic="$11",note="$13}' | head -40
    fi
  done
  echo
done

echo "################ lccm3_fixed vs qld_lccm3 merged leaderboard sizes"
for t in lccm3_fixed qld_lccm3; do
  echo "--- $t"
  ls -1 --time-style=long-iso -l "$RUNS/$t/merged/variant_board/leaderboard.csv" 2>/dev/null
  awk -F, 'NR>1 {print "   "$2","$3","$4,"conv="$7","bic="$11}' "$RUNS/$t/merged/variant_board/leaderboard.csv" 2>/dev/null
done