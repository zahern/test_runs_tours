#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("s5", "stage5_destination_choice.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("import OK")
import inspect
print(inspect.signature(m._predict_transit_access))
PY