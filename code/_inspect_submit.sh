#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== STAGE_SEGMENTS ==="
sed -n '/^STAGE_SEGMENTS/,/^}/p' ga_shard.py
echo
echo "=== SHARD_SEGMENTS ==="
sed -n '/^SHARD_SEGMENTS/,/^}/p' ga_shard.py
echo
echo "=== job_ga_shard_single.pbs: STAGE_ID / SHARD_SEGMENT handling ==="
grep -n "STAGE_ID\|SHARD_SEGMENT\|STAGE_IDS" job_ga_shard_single.pbs | head -15
echo
echo "=== job_merge_ga_shards.pbs vars ==="
grep -n "STAGE_IDS\|RUN_TAG\|qsub\|merge_ga_shards" job_merge_ga_shards.pbs | head -15
echo
echo "=== _run_ga_shard.py: stage5 gens wiring ==="
grep -n "GA_STAGE5_GENS\|GA_STAGE5_RESTARTS\|GA_STAGE5_STAG\|GA_STAGE51_GENS\|s5_stag" ga_hypothesis_testing.py | head -15