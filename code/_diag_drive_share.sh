#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PIPELINE_SKIMS_PATH=${PIPELINE_SKIMS_PATH:-../demand_/skims_371.omx}
python - <<'PY'
import os, numpy as np, openmatrix as omx
sk = omx.open_file(os.path.abspath(os.environ["PIPELINE_SKIMS_PATH"]))
names = set(sk.listMatrices())
walk = np.asarray(sk["WLK_LOC_WLK_TOTIVT__AM"], dtype=np.float64)   # walk->PT
pr   = np.asarray(sk["WLK_LOC_DRV_TOTIVT__AM"], dtype=np.float64)   # walk->PnR, then ride
print("=== the two access skims on the SAME OD pairs (where walk is observed) ===")
m = np.isfinite(walk) & (walk > 0) & np.isfinite(pr) & (pr > 0)
print(f"comparable cells: {m.sum()}")
d = (walk[m] - pr[m]) / 60.0
print(f"walk_h - pr_hours: mean={d.mean():.3f}  median={np.median(d):.3f}  "
      f"P(walk slower)={np.mean(d>0):.1%}")
print(f"  walk_h  median={np.median(walk[m])/60:.2f} h   p95={np.percentile(walk[m],95)/60:.2f} h")
print(f"  pr_h    median={np.median(pr[m])/60:.2f} h   p95={np.percentile(pr[m],95)/60:.2f} h")
print()
print("KEY: if pr_h is mostly SHORTER than walk_h, a utility that only")
print("     penalises composite time will over-select drive access.")
print()
print("=== how often is PnR actually the faster option? ===")
print(f"  P(walk_h > pr_h)  = {np.mean(walk[m]/60 > pr[m]/60):.1%}")
print(f"  P(walk_h < pr_h)  = {np.mean(walk[m]/60 < pr[m]/60):.1%}")
print()
print("=== observed survey mode split for reference ===")
import pandas as pd
d0 = pd.read_csv("initial_trips_hhts.csv", low_memory=False)
vc = d0["MAINMODE"].value_counts(normalize=True)
for k in ("DA", "Walk", "Transit", "Bike", "Other"):
    if k in vc:
        print(f"  {k:<8} {vc[k]:.2%}")
tr = d0[d0["MAINMODE"] == "Transit"]
print(f"\n  Transit trips: {len(tr)}")
print(f"  SEQ car-ownership proxy: share of transit trips with a car is NOT in MAINMODE,")
print(f"  so 'drive to PT' share must come from the simulation rule, not the survey.")
PY