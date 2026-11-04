#!/bin/bash
# READ-ONLY. Confirm stage dispatch + segment lists + stage->file mapping.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code

echo "=== run_ga_stage dispatch (ga_hypothesis_testing.py 9930-9990) ==="
sed -n '9930,9990p' "$CODE/ga_hypothesis_testing.py"

echo
echo "=== ga_shard.py: segment lists / stage names ==="
grep -n -E "^STAGE|^SEG|^_SEG|SEGMENTS|def segments|def s5_|PRIMARY_GROUP|def variant_group|def pick_winners|def shard_keep|SHARD_SEGMENT" "$CODE/ga_shard.py" | head -60

echo
echo "=== job_ga_shard_single.pbs ==="
cat "$CODE/job_ga_shard_single.pbs"

echo
echo "=== _run_ga_single_stage.py: stage id -> function ==="
grep -n -E "stage|STAGE" "$CODE/_run_ga_single_stage.py" | head -60