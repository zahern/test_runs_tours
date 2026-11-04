#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== stage-0 shard vs merged ga_summary.csv ==="
R=runs/ga_parallel/cdap_s5_refit_20260928
echo "--- SHARD ---"
cat $R/stage0/shards/all/ga_reports/ga_summary.csv 2>/dev/null | head -3
echo "--- MERGED ---"
cat $R/stage0/merged/ga_reports/ga_summary.csv 2>/dev/null | head -3
echo
echo "=== merge script: how does it read best_fitness? ==="
ls -la merge_ga_shards.py job_merge_ga_shards.pbs 2>&1 | sed 's/^/  /'
echo
echo "=== sentinel default in merge script ==="
grep -n "1e9\|-1000000000\|best_fitness\|def .*summary\|stage.*0" merge_ga_shards.py 2>/dev/null | head -30