#!/bin/bash
BASE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
JID=24998793
for i in $(seq 1 8); do
  echo "=== poll $i ($(date +%H:%M)) ==="
  qstat -u ahernz | grep "$JID" | awk '{print "state:", $10}'
  echo "--- cput ---"
  qstat -f $JID 2>/dev/null | grep -iE 'cput|walltime =|mem ='
  echo "--- poolfix artifacts ---"
  find "$BASE/runs/ga_parallel/destfix_poolfix" -type f -newermt '10 minutes ago' 2>/dev/null | head
  st=$(qstat -u ahernz | grep "$JID" | awk '{print $10}')
  if [ "$st" != "R" ]; then break; fi
  sleep 600
done