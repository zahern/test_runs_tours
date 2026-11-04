#!/bin/bash
# READ-ONLY. Exact parameter-level health lines for the latest stage5 shards.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
for b in ga_s5work_lccm3_fixed.e26637303 ga_s5nonmand_lccm3_fixed.e26637305 ga_s5edu_lccm3_fixed.e26637306 ga_s5work_vmap_e1.e26632851; do
  echo "##################### $b"
  echo "--- NEVER MOVED lines:"
  grep -n 'NEVER MOVED' "$CODE/$b" 2>/dev/null | cut -c1-400
  echo "--- AT A BOUND lines:"
  grep -n -i 'at a bound' "$CODE/$b" 2>/dev/null | cut -c1-400
  echo "--- stuck-drop / revert lines:"
  grep -n -E 'stuck near-zero|reverting' "$CODE/$b" 2>/dev/null | cut -c1-500
  echo "--- Traceback context (first 12 lines of each TB):"
  grep -n -A6 'Traceback' "$CODE/$b" 2>/dev/null | cut -c1-220 | head -40
  echo
done