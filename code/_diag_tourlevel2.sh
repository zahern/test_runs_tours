#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import pandas as pd, numpy as np
d = pd.read_csv("initial_trips_hhts.csv", low_memory=False)
print("rows:", len(d), "tours:", d['tour_id'].nunique())
print("\n=== MAINMODE varies WITHIN a tour? ===")
v = d.groupby('tour_id')['MAINMODE'].nunique()
print(v.value_counts().sort_index().to_dict())
print(f"tours with >1 distinct trip mode: {(v>1).sum()} ({(v>1).mean():.1%})")
print("\n=== trips per tour ===")
print(d.groupby('tour_id').size().value_counts().sort_index().head(8).to_dict())
print("\n=== a TRANSIT tour with multiple trip modes ===")
tr = d[d['MAINMODE']=='Transit']
vt = tr.groupby('tour_id')['MAINMODE'].nunique()
cand = vt[vt>1].index[:3]
if len(cand):
    for t in cand:
        ex = d[d['tour_id']==t].sort_values('purpose_rank')
        print(f"\n--- tour {t} ---")
        print(ex[['STARTSTOP','ENDSTOP','ORIGPURP','DESTPURP','MAINMODE',
                  'purpose_rank','tour_leg','trip_direction']].to_string(index=False))
print("\n=== mode SEQUENCE within tours (top 15) ===")
seq = (d.sort_values('purpose_rank').groupby('tour_id')['MAINMODE']
       .apply(lambda s: '>'.join(s.astype(str))))
print(seq.value_counts().head(15).to_dict())
print("\n=== for tours whose MAXIMUM-rank trip is Transit, what else appears? ===")
top = d.sort_values('purpose_rank').groupby('tour_id').tail(1)
tt = top[top['MAINMODE']=='Transit']
print("tours with a Transit top-leg trip:", len(tt))
# co-occurring modes within those tours
co = d[d['tour_id'].isin(tt['tour_id'])].groupby('tour_id')['MAINMODE'].apply(
    lambda s: tuple(sorted(set(s))))
print("co-occurring mode sets in those tours:")
for k, c in co.value_counts().head(8).items():
    print(f"   {c:>5}  {k}")
PY