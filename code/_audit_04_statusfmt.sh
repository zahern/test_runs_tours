#!/bin/bash
# READ-ONLY. Inspect raw tail of a few .o files to learn status block format.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
for b in ga_s5work_lccm3_fixed.o26637303 ga_s5nonmand_lccm3_fixed.o26637305 ga_shardmerge_lccm3b.o26639465 ga_s5work_qld_lccm3.o26620352 ga_s5nonmand_qld_lccm3.o26620353 ga_s5edu_lccm3_fixed.o26637306; do
  echo "########## $b (last 25 lines)"
  tail -25 "$CODE/$b" 2>/dev/null
  echo
done

echo "########## _final_status.sh contents (pre-existing helper)"
cat "$CODE/_final_status.sh" 2>/dev/null | head -60

echo
echo "########## any PBS status blocks anywhere: 'Exit_status' literal search"
grep -l "Exit_status" "$CODE"/*.o[0-9]* 2>/dev/null | head -20
echo "--- count of .o files:"
ls -1 "$CODE"/*.o[0-9]* 2>/dev/null | wc -l