#!/bin/bash
# READ-ONLY. Exact exception text for the latent-class crash + remaining detail.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code

echo "=== full traceback context, work lccm3_fixed (lines 130-200 of stderr) ==="
sed -n '130,200p' "$CODE/ga_s5work_lccm3_fixed.e26637303" 2>/dev/null | cut -c1-230

echo
echo "=== edu lccm3_fixed: NEVER MOVED / AT BOUND / stuck-drop / traceback ==="
grep -n -E 'NEVER MOVED|AT A BOUND|stuck near-zero|reverting|Traceback|Error|error' "$CODE/ga_s5edu_lccm3_fixed.e26637306" 2>/dev/null | cut -c1-300 | head -40

echo
echo "=== vmap_e1 stderr: NEVER MOVED / AT BOUND / stuck / traceback ==="
grep -n -E 'NEVER MOVED|AT A BOUND|stuck near-zero|reverting|Traceback' "$CODE/ga_s5work_vmap_e1.e26632851" 2>/dev/null | cut -c1-300 | head -30

echo
echo "=== MXL / MH env actually used, latest jobs ==="
for b in ga_s5work_lccm3_fixed.o26637303 ga_s5nonmand_qld_full40.o26552451 ga_s51othdiscr_retry.o26556257 ga_s1nonmand_retry.o26530730 ga_s0all_retry.o26530729; do
  echo "--- $b"
  grep -h -m1 -oE 'CDAP_MXL=[0-9a-z]*|STAGE2_MH=[0-9]*|GA_STAGE4_MXL_SEARCH=[0-9]*|STAGE5_LC_GENES=[0-9]*|GA_ENGINE=[a-z]*|GA_ESTIMATOR=[a-z_]*' "$CODE/$b" 2>/dev/null | tr '\n' ' '
  echo
done

echo
echo "=== cdap_s5_refit stage0 shard detail ==="
ls -1 --time-style=long-iso -l "$CODE/runs/ga_parallel/cdap_s5_refit_20260928/stage0/shards/all/" 2>/dev/null
echo "--- ga_s0all_retry stdout (sentinel evidence):"
grep -n -E 'sentinel|-1e\+09|-1000000000|converged|best_fitness|GA done|Stage 0' "$CODE/ga_s0all_retry.o26530729" 2>/dev/null | cut -c1-250 | head -25