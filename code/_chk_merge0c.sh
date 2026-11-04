#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== merge stdout ==="
cat ga_merge_s0.o26665329
echo
echo "=== merge stderr ==="
cat ga_merge_s0.e26665329
echo
echo "=== output files for the conv1 jobs ==="
ls -la ga_s5*conv1.* ga_s2*conv1.* ga_s4*conv1.* ga_s51*conv1.* 2>/dev/null | head -30 || echo "(NONE - jobs never started)"