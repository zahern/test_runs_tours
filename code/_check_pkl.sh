#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
R=runs/ga_parallel/cdap_s5_refit_20260928
echo "=== stage0 shard dir listing ==="
ls -la $R/stage0/shards/all/ 2>&1 | head -20
echo
echo "=== ga_results.pkl present anywhere in stage0? ==="
find $R/stage0 -name "ga_results.pkl" 2>/dev/null
echo "(end find)"
echo
echo "=== what other stages have ga_results.pkl ==="
for s in 1 2 3 4 5 51; do
  n=$(find $R/stage$s -name "ga_results.pkl" 2>/dev/null | wc -l)
  echo "  stage$s: $n ga_results.pkl"
done