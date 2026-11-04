#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PIPELINE_SKIMS_PATH=${PIPELINE_SKIMS_PATH:-../demand_/skims_371.omx}
python - <<'PY'
import os, numpy as np, openmatrix as omx
sk = omx.open_file(os.path.abspath(os.environ["PIPELINE_SKIMS_PATH"]))
names = list(sk.listMatrices())
print("=== ALL access-to-transit skims (2x2 walk/drive x walk/PT-connectivity) ===")
for m in sorted(names):
    if ("LOC" in m) and ("TOTIVT" in m) and m.endswith("__AM"):
        a = np.asarray(sk[m], dtype=np.float64)
        o = a[~np.isnan(a) & (a > 0)]
        print(f"  {m:<28} NaN={np.isnan(a).mean():5.1%}  median={np.median(o):6.1f}")
PY
echo
echo "=== HHTS tour fields: is transit ACCESS observed? ==="
grep -rnE "transit_access|pt_access|access_mode|TransitAccess|TRANSIT_ACCESS" data_connect.py pipeline_logger.py cdap_households.py 2>/dev/null | head -12
echo
echo "=== raw survey transit columns seen in the pipeline ==="
grep -rnE "'.{0,4}(Transit|Transit_|PT_|Bus|Train).{0,14}'" cdap_households_estimation.py 2>/dev/null | head -12