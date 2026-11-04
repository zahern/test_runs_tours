#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== 1. merged artifact at correct path ==="
cat runs/ga_parallel/cdap_s5_refit_20260928/merged/ga_reports/ga_summary.csv 2>/dev/null | head -4 || echo "(none)"
echo
echo "=== 2. did the conv1 jobs run? check accounting ==="
tracejob -a -q cpu_batch_exec -r 26665306 2>/dev/null | head -5 || echo "(tracejob n/a)"
echo "--- recent completed via qstat -x ---"
qstat -x -u ahernz 2>/dev/null | grep -E "conv1|2666530|2666531|2666532" | head -20 || echo "(qstat -x n/a / nothing)"
echo
echo "=== 3. any conv1 output at all, anywhere ==="
ls -lat *.o266653* *.e266653* 2>/dev/null | head -20 || echo "(NONE)"
echo
echo "=== 4. full qstat -u (all jobs) ==="
qstat -u ahernz 2>/dev/null | head -20