#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("s5", "stage5_destination_choice.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)   # raises on any NameError/import-time bug
print("module imports OK")
print("TRANSIT_ACCESS_MAP   :", m.TRANSIT_ACCESS_MAP)
print("TRANSIT_ACCESS_NAMES :", m.TRANSIT_ACCESS_NAMES)
print("PT_ACCESS_SKIM       :", m.PT_ACCESS_SKIM)
print("PT_BIKE_FROM_WALK    :", round(m.PT_BIKE_FROM_WALK, 4))
print("_pt_composite_skim   :", callable(m._pt_composite_skim))
print("_fill_transit_gaps   :", callable(m._fill_transit_gaps))
# access-code normalisation round-trips
print("_transit_access_code('Bike') =", m._transit_access_code('Bike'))
print("_transit_access_code(3)      =", m._transit_access_code(3))
print("_transit_mode_name(1,3)      =", m._transit_mode_name(1, 3))
# the chooser is importable and takes the same signature the call sites use
import inspect
print("_predict_transit_access sig  :", inspect.signature(m._predict_transit_access))
PY