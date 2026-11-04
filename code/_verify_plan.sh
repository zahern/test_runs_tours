#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== 1. Is 'lccm' in VARIANT_GROUPS? ==="
sed -n '160,205p' ga_shard.py
echo
echo "=== 2. PRIMARY_GROUP for s5mode ==="
grep -n "PRIMARY_GROUP" ga_shard.py | head -5
sed -n '/^PRIMARY_GROUP/,/^}/p' ga_shard.py | head -20
echo
echo "=== 3. STAGE5_LC_GENES references ==="
grep -rn "STAGE5_LC_GENES" *.pbs *.py 2>/dev/null | head -10
echo
echo "=== 4. GA_STAGE5_GENS / generations default ==="
grep -rn "GA_STAGE5_GENS\|GA_STAGE5_RESTARTS\|GA_STAGE5_STAG" *.pbs *.py 2>/dev/null | head -12
echo
echo "=== 5. leaderboard _row() ll/k handling ==="
sed -n '25,60p' ga_variant_refits.py