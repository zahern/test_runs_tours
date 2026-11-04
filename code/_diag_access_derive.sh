#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import pandas as pd, numpy as np
d = pd.read_csv("initial_trips_hhts.csv", low_memory=False)
d['tour_id'] = d['tour_id'].astype(str)
print("rows:", len(d))

print("\n=== tour_leg x trip_direction ===")
print(pd.crosstab(d['tour_leg'], d['trip_direction']))

# A tour = the trips sharing (PERSID, date, tour_id). Use main_trip_id as the
# anchor key; outbound legs hang off it.
print("\n=== is main_trip_id unique per tour group? ===")
grp = d.groupby(['PERSID','date','tour_id'])['main_trip_id'].nunique()
print("groups:", len(grp), " with >1 distinct main_trip_id:", int((grp>1).sum()))

main = d[d['tour_leg'] == 'main'].copy()
print("\nMAIN trips:", len(main), " unique main_trip_id:", main['main_trip_id'].nunique())
print("MAIN MODE distribution (this IS the observed tour mode):")
vc = main['MAINMODE'].value_counts()
for k, v in vc.items():
    print(f"  {str(k):<10} {v:>7}  {v/len(main):6.2%}")

print("\n=== THE KEY TEST: for MAIN=Transit tours, what are the outbound legs? ===")
mt = main[main['MAINMODE'] == 'Transit']
print("transit-anchor trips:", len(mt))
keys = set(zip(mt['PERSID'], mt['date'], mt['tour_id']))
ob = d[d['tour_leg'] != 'main'].copy()
ob['_k'] = list(zip(ob['PERSID'], ob['date'], ob['tour_id']))
ob = ob[ob['_k'].isin(keys)]
print("outbound legs attached to transit tours:", len(ob))
if len(ob):
    print("\noutbound-leg mode distribution for TRANSIT tours:")
    for k, v in ob['MAINMODE'].value_counts().items():
        print(f"  {str(k):<10} {v:>7}  {v/len(ob):6.2%}")
    print("\nACCESS-MODE SIGNAL:")
    acc = ob[ob['MAINMODE'].isin(['Walk','Bike','DA'])]
    print(f"  outbound legs that are Walk/Bike/DA: {len(acc)} "
          f"({len(acc)/len(ob):.1%} of transit-tour outbound legs)")
    for k, v in acc['MAINMODE'].value_counts().items():
        print(f"    {str(k):<8} {v:>6}")

print("\n=== contrast: MAIN=DA tours -> what are their outbound legs? ===")
md = main[main['MAINMODE'] == 'DA']
keys2 = set(zip(md['PERSID'], md['date'], md['tour_id']))
ob2 = d[d['tour_leg'] != 'main'].copy()
ob2['_k'] = list(zip(ob2['PERSID'], ob2['date'], ob2['tour_id']))
ob2 = ob2[ob2['_k'].isin(keys2)]
print("outbound legs on DA tours:", len(ob2))
if len(ob2):
    for k, v in ob2['MAINMODE'].value_counts().head(5).items():
        print(f"  {str(k):<10} {v:>7}  {v/len(ob2):6.2%}")

print("\n=== MAIN=Walk tours -> outbound legs (walk-access chaining check) ===")
mw = main[main['MAINMODE'] == 'Walk']
keys3 = set(zip(mw['PERSID'], mw['date'], mw['tour_id']))
ob3 = d[d['tour_leg'] != 'main'].copy()
ob3['_k'] = list(zip(ob3['PERSID'], ob3['date'], ob3['tour_id']))
ob3 = ob3[ob3['_k'].isin(keys3)]
print("outbound legs on Walk tours:", len(ob3))
if len(ob3):
    for k, v in ob3['MAINMODE'].value_counts().head(5).items():
        print(f"  {str(k):<10} {v:>7}  {v/len(ob3):6.2%}")
PY