#!/bin/bash
echo "=== files in run dir ==="
find /mnt/hpccs01/home/ahernz/test_runs_tours/runs/ga_parallel/destfix_poolfix -type f 2>/dev/null
echo "=== stdout/err for 24998793 ==="
ls -la /mnt/hpccs01/home/ahernz/test_runs_tours/code/*24998793* 2>/dev/null
echo "=== node processes ==="
ssh cpu1n042 "ps aux | grep -E 'python|python3' | grep ahernz | head -10" 2>/dev/null || echo "cannot ssh to node"
echo "=== qstat ==="
qstat -u ahernz | grep 24998793