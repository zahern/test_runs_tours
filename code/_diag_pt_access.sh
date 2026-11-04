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
print("=== access-to-PT skim availability ===")
for key in ("WLK_LOC_", "DRV_LOC_", "BIKE_", "CYC_", "PT_"):
    hit = [m for m in names if m.startswith(key) and "AM" in m]
    print(f"{key:<12} ({len(hit)}): {sorted(hit)[:8]}")
print()
print("=== NaN + value range for the three candidate access modes ===")
for m in ("WLK_LOC_WLK_TOTIVT__AM", "DRV_LOC_WLK_TOTIVT__AM"):
    if m in names:
        a = np.asarray(sk[m], dtype=np.float64)
        o = a[~np.isnan(a) & (a > 0)]
        print(f"{m:<26} NaN={np.isnan(a).mean():5.1%}  median={np.median(o):6.1f}  "
              f"p95={np.percentile(o,95):6.1f} min={o.min():.1f}")
    else:
        print(f"{m:<26} ABSENT")
print()
print("=== PT sub-mode detail (wait/board/far) for walk-access ===")
for m in ("WLK_LOC_WLK_WAIT__AM","WLK_LOC_WLK_BOARDS__AM","WLK_LOC_WLK_FAR__AM","WLK_LOC_WLK_WACE__AM"):
    if m in names:
        a = np.asarray(sk[m], dtype=np.float64)
        o = a[~np.isnan(a)]
        print(f"  {m:<26} NaN={np.isnan(a).mean():5.1%} median={np.median(o):8.2f}")
PY
echo
echo "=== nest structure in _build_mode_model ==="
grep -n 'nest\|_nest\|mu_\|Mu_Motor\|Mu_NonMotor' stage5_destination_choice.py | grep -iE 'nest|mu' | head -20