#!/bin/bash
# READ-ONLY. STAGE_SEGMENTS map + primary/variant groups.
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== ga_shard.py 60-250 ==="
sed -n '60,250p' "$CODE/ga_shard.py"