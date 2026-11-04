#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
echo "=== 1. py_compile the three edited files ==="
python -m py_compile ga_shard.py ga_variant_refits.py ga_hypothesis_testing.py stage5_destination_choice.py && echo "COMPILE OK" || echo "COMPILE FAILED"
echo
echo "=== 2. lccm now resolves into a real group ==="
python - <<'PY'
import ga_shard as s
for slot, var in [("s5mode","nl"),("s5mode","mnl"),("s5mode","lccm"),
                  ("s5mode","mxl"),("s5dest","mnl"),("s5dest","nl_dist")]:
    g = s.variant_group(slot, var)
    prim = s.PRIMARY_GROUP.get(slot)
    ok = "WINNER-CAPABLE" if g == prim else "diagnostic-only"
    print(f"  {slot}/{var:8s} -> group={g:12s} primary={prim:8s} {ok}")
print("  lccm description:", s.VARIANT_DESCRIPTIONS.get(("s5mode","lccm")))
PY
echo
echo "=== 3. confirm _stage5_dest_only_fitness honours the flag ==="
grep -n "STAGE5_DEST_NL_DISTANCE" ga_hypothesis_testing.py | head
echo
echo "=== 4. confirm _build_dest_model_nl_distance is importable ==="
python -c "
import stage5_destination_choice as s5
f=getattr(s5,'_build_dest_model_nl_distance',None)
print('  _build_dest_model_nl_distance present:', callable(f))
" 2>&1 | tail -3