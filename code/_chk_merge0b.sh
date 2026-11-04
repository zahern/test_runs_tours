#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== all my jobs now (any state) ==="
qstat -u ahernz 2>/dev/null | grep -E "conv1|merge_s0|Job ID|----"
echo
echo "=== merge output files (any name) ==="
ls -la *merge*26665329* 2>/dev/null || echo "(none)"
ls -la ga_merge_s0.* 2>/dev/null || echo "(no ga_merge_s0 files)"
echo
echo "=== any recent .o/.e for merge ==="
ls -t *.o* *.e* 2>/dev/null | head -8