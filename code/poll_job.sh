#!/bin/bash
JID="24998793"
for i in 1 2 3 4 5 6; do
  st=$(qstat -u ahernz | grep "$JID" | awk '{print $10}')
  echo "poll $i: ${st:-GONE}"
  if [ "$st" = "R" ]; then break; fi
  sleep 120
done
qstat -u ahernz | grep "$JID" || echo "job no longer in qstat"