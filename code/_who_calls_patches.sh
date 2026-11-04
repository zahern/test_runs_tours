#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== does anything import/call larch_patches? ==="
grep -rn "larch_patches" --include=*.py --include=*.pbs --include=*.sh . 2>/dev/null | grep -v "^\./larch_patches.py" | head -30
echo "--- end ---"
echo
echo "=== tail of apply(): is apply() auto-invoked at module import? ==="
tail -30 larch_patches.py
echo
echo "=== search for 'apply()' call sites ==="
grep -rn "larch_patches\.apply\|patches\.apply\|import larch_patches" . 2>/dev/null | head -20
echo "--- end ---"
echo
echo "=== pbs entrypoint: what does the shard script run? ==="
grep -n "python\|python3\|python3 -u\|RUN_CMD\|\$CMD" job_ga_shard_single.pbs | head -25