#!/bin/bash
echo "=== latest poolfix artifacts ==="
find /mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/destfix_poolfix -type f -newermt '30 minutes ago' 2>/dev/null
echo "=== stage5 timeline check: is stdout file present yet ==="
ls -la /mnt/hpccs01/home/ahernz/test_runs_tours/code/ga_s5_destfix_poolfix.o* 2>/dev/null || echo "still absent"
echo "=== job elapsed ==="
qstat -f 24998793 2>/dev/null | grep -iE 'stime|etime'
echo "=== check node cpu via pbsmom ==="
qstat -f 24998793 2>/dev/null | grep -iE 'comment|exec_host'