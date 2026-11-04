#!/bin/bash
echo "=== job CPU info ==="
qstat -f 24998793 2>/dev/null | grep -iE 'cpu|resources_used|cput|walltime'
echo "=== recent modified files in code dir (any write activity = alive) ==="
find /mnt/hpccs01/home/ahernz/test_runs_tours/code -newermt '5 minutes ago' -type f 2>/dev/null | grep -v __pycache__ | head
echo "=== try reading node's /tmp or home spool ==="
ssh -o ConnectTimeout=10 cpu1n042 "ls -la /tmp/*24998793* /var/tmp/*24998793* 2>/dev/null; find /tmp -name '*24998793*' 2>/dev/null | head" 2>&1 | head -10