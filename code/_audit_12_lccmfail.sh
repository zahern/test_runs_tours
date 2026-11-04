#!/bin/bash
# READ-ONLY. Why does the lccm variant refit fail? + MXL/MH wiring + per-run
# estimation-health signal counts in stage-5 stdout.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code

for f in ga_s5work_lccm3_fixed.o26637303 ga_s5nonmand_lccm3_fixed.o26637305 ga_s5work_vmap_e1.o26632851 ga_shardmerge_lccm3_fixed.o26637307; do
  echo "##################### $f"
  echo "--- lccm / mxl / gse context lines:"
  grep -n -i -E 'lccm|mxl|gse|latent class' "$CODE/$f" 2>/dev/null | grep -v -E 'Submit_arguments|Variable_List|Output_Path|Error_Path' | head -30
  echo "--- sentinel / fitness failure lines:"
  grep -n -E 'sentinel|-1e\+09|-1000000000|failed|FAILED|Traceback|Error' "$CODE/$f" 2>/dev/null | grep -v -E 'Submit_arguments|Variable_List' | head -20
  echo
done

echo "##################### CONVERGENCE TRACE (lccm3_fixed)"
for tag in lccm3_fixed qld_lccm3 vmap_e1_pilot; do
  for seg in work edu nonmand; do
    p="/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/$tag/stage5/shards/$seg/convergence_trace.csv"
    [ -f "$p" ] && { echo "--- $tag/$seg"; cat "$p"; }
  done
done