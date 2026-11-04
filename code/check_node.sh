#!/bin/bash
echo "=== try node ssh ==="
ssh -o ConnectTimeout=10 cpu1n042 "ps -o pid,etime,pcpu,pmem,rss,args -u ahernz | grep -E 'python' | grep -v grep | head -10" 2>&1 | head -15
echo "=== run dir depth ==="
find /mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/destfix_poolfix -type f 2>/dev/null
echo "=== pipeline logs anywhere recent ==="
find /mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel -name 'pipeline_run.log' -newermt '2 hours ago' 2>/dev/null | head
echo "=== ckpt dir ==="
ls -la /mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/destfix_poolfix/stage5/ga_reports/ 2>/dev/null
ls -la /mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/destfix_poolfix/stage5/plots/ 2>/dev/null