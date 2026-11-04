#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
R=runs/ga_parallel/cdap_s5_refit_20260928
echo "=== merged dir tree ==="
find $R/stage0/merged -type f 2>/dev/null | head -20
echo
echo "=== merged ga_summary.csv (raw, cat -A first 2 lines) ==="
cat $R/stage0/merged/ga_reports/ga_summary.csv 2>&1 | head -3
echo
echo "=== merge_ga_shards.py lines 60-100 (sentinel gate) ==="
sed -n '60,100p' merge_ga_shards.py
echo
echo "=== merge_ga_shards.py lines 125,150 (conflict/segment walk) ==="
sed -n '125,150p' merge_ga_shards.py