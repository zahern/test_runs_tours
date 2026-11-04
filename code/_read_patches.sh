#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== larch_patches.py head (1-45) ==="
sed -n '1,45p' larch_patches.py
echo
echo "=== larch_patches.py apply() (79-END) ==="
sed -n '79,400p' larch_patches.py
echo
echo "=== who calls larch_patches.apply ? ==="
grep -rn "larch_patches" --include=*.py --include=*.pbs --include=*.sh . 2>/dev/null | grep -v "^./larch_patches.py" | head -20