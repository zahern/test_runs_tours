#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== tail of larch_patches.py (does it auto-apply?) ==="
tail -25 larch_patches.py
echo "=================================================="
echo "=== targeted import search in known entrypoints ==="
for f in ga_shard.py ga_hypothesis_testing.py ga_variant_refits.py stage5_destination_choice.py stage5b_trip_destination.py job_ga_shard_single.pbs run_ga_shard.py; do
  if [ -f "$f" ]; then
    hit=$(grep -n "larch_patches" "$f" 2>/dev/null)
    if [ -n "$hit" ]; then echo "[$f] $hit"; else echo "[$f] NO larch_patches reference"; fi
  fi
done
echo "=================================================="
echo "=== which python does the pbs invoke? ==="
grep -n "python\|^export\|^  " job_ga_shard_single.pbs 2>/dev/null | grep -i python | head