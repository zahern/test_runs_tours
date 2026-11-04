#!/bin/bash
CODE=/mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== ga_shard.py STAGE_SEGMENTS (68-80) ==="
sed -n '68,80p' "$CODE/ga_shard.py"
echo
echo "=== STAGE5_LC_GENES in any recent job env? ==="
for f in $(ls -1t "$CODE"/*.o[0-9]* 2>/dev/null | head -40); do
  v=$(grep -h -m1 -oE 'STAGE5_LC_GENES=[0-9]*' "$f" 2>/dev/null)
  [ -n "$v" ] && echo "$(basename $f): $v"
done
echo "(none listed above = STAGE5_LC_GENES absent, so default 0)"
echo
echo "=== lccm3_fixed vs qld_lccm3 shard leaderboard diff ==="
diff "$CODE/runs/ga_parallel/lccm3_fixed/stage5/shards/work/variants/leaderboard.csv" \
     "$CODE/runs/ga_parallel/qld_lccm3/stage5/shards/work/variants/leaderboard.csv" && echo "IDENTICAL work leaderboard"
diff "$CODE/runs/ga_parallel/lccm3_fixed/stage5/shards/nonmand/variants/leaderboard.csv" \
     "$CODE/runs/ga_parallel/qld_lccm3/stage5/shards/nonmand/variants/leaderboard.csv" && echo "IDENTICAL nonmand leaderboard"
diff "$CODE/runs/ga_parallel/lccm3_fixed/stage5/shards/edu/variants/leaderboard.csv" \
     "$CODE/runs/ga_parallel/qld_lccm3/stage5/shards/edu/variants/leaderboard.csv" && echo "IDENTICAL edu leaderboard"
echo
echo "=== lccm3_fixed merged leaderboard content (stale?) ==="
awk -F, 'NR>1{print "  "$2","$3","$4,"conv="$7}' "$CODE/runs/ga_parallel/lccm3_fixed/merged/variant_board/leaderboard.csv"