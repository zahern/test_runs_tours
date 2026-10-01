#!/bin/bash
echo "=== gatefix run dir structure (reference) ==="
find /mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/destfix_gatefix -maxdepth 3 -type f 2>/dev/null | head -40
echo "=== try spooled stdout on nodes ==="
for node in cpu1n042 cpu1n015; do
  echo "--- $node ---"
  ssh -o ConnectTimeout=10 $node "ls -la /var/spool/pbs/spool/*24998793* 2>/dev/null; ls -la /var/spool/torque/spool/*24998793* 2>/dev/null; ps -o pid,etime,pcpu,pmem,rss,args -u ahernz | grep -E 'python3? |pip' | grep -v grep | head -5" 2>&1 | head -20
done