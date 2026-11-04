#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import pandas as pd, numpy as np
d = pd.read_csv("initial_trips_hhts.csv", low_memory=False)
print("rows:", len(d), " cols:", len(d.columns))
print("\n=== ALL columns ===")
for i, c in enumerate(d.columns):
    print(f"  {i:2d} {c}")
print("\n=== tour_id structure ===")
print("n tours:", d['tour_id'].nunique(), " n trip ids:", d['TRIPID'].nunique())
print("trips per tour:", d.groupby('tour_id').size().describe()[['mean','50%','max']].to_dict())
print("\n=== does MAINMODE vary WITHIN a tour? ===")
vmode = d.groupby('tour_id')['MAINMODE'].nunique()
print(vmode.value_counts().head(10).to_dict())
multi = vmode[vmode > 1]
print(f"tours with >1 distinct trip mode: {len(multi)} of {len(vmode)} ({len(multi)/len(vmode):.1%})")
print("\n=== example multi-mode tour ===")
if len(multi):
    t = multi.index[0]
    ex = d[d['tour_id'] == t].sort_values('purpose_rank')
    print(ex[['TRIPID','STARTSTOP','ENDSTOP','ORIGPURP','DESTPURP','MAINMODE','purpose_rank',
              'tour_leg','trip_direction','OVERALL_PURPOSE']].to_string(index=False))
print("\n=== tour_leg / trip_direction semantics ===")
for c in ('tour_leg','trip_direction','purpose_rank'):
    if c in d.columns:
        print(f"  {c}: {d[c].value_counts(dropna=False).head(8).to_dict()}")
print("\n=== which trip is the 'main' one? ===")
print("main_trip_id unique:", d['main_trip_id'].nunique() if 'main_trip_id' in d else 'n/a')
print("OVERALL_PURPOSE values:", d['OVERALL_PURPOSE'].value_counts(dropna=False).head(10).to_dict())
PY