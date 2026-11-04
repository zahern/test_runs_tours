#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import pandas as pd
d = pd.read_csv("initial_trips_hhts.csv", low_memory=False)
print("rows:", len(d))
print("\n=== MAINMODE value counts (raw survey) ===")
vc = d["MAINMODE"].value_counts(dropna=False)
for k, v in vc.head(30).items():
    print(f"  {str(k)[:40]:<40} {v:>9}  {v/len(d):6.2%}")
print("\n=== unique MAINMODE count:", d["MAINMODE"].nunique())
print("\n=== any code whose TEXT mentions transit/park/bus/train/ferry? ===")
s = d["MAINMODE"].astype(str)
hit = s[s.str.contains("transit|park|bus|train|ferry|rail|PT", case=False, na=False)]
print("  matched rows:", len(hit), " unique codes:", hit.unique()[:20])
PY