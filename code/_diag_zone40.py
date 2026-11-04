import pandas as pd, numpy as np, sys, os

lu = pd.read_csv('zone_order_and_landuse.csv')
zcol = 'ORDER'
emp_cols = ['FPSEMPN','HEREMPN','RETEMPN','MWTEMPN','OTHEMP','AGREMPN']
lu['TOTEMP'] = sum(lu[c].fillna(0) for c in emp_cols if c in lu.columns)

# ---- 1. WORK SIZE TERM (current coefficients) ----
def work_size_raw(r):
    TOTEMP     = max(0, float(r.get('TOTEMP', 0) or 0))
    FPSEMPN    = max(0, float(r.get('FPSEMPN', 0) or 0))
    HEREMPN    = max(0, float(r.get('HEREMPN', 0) or 0))
    RETEMPN    = max(0, float(r.get('RETEMPN', 0) or 0))
    commercial = max(0, float(r.get('commercial', 0) or 0))
    return (TOTEMP**1.0
            + FPSEMPN**0.75
            + HEREMPN**0.35
            + RETEMPN**0.25
            + commercial**0.65)

lu['ws_raw'] = lu.apply(lambda r: work_size_raw(dict(r)), axis=1)
lu['ws_log'] = np.log(lu['ws_raw'].clip(lower=1e-6))
lu['ws_dm']  = lu['ws_log'] - lu['ws_log'].mean()

print("=== WORK SIZE TERM (current code) ===")
cols = [zcol,'TOTEMP','FPSEMPN','commercial','ws_raw','ws_dm']
top = lu.nlargest(15,'ws_dm')[cols].rename(columns={zcol:'zone'})
print(top.to_string(index=False))

z40r = lu[lu[zcol]==40].iloc[0]
rank40 = int((lu['ws_dm'] > z40r['ws_dm']).sum()) + 1
print(f"\nZone 40 rank: {rank40}/{len(lu)}  dm={z40r['ws_dm']:+.3f}")
print(f"  TOTEMP={z40r['TOTEMP']:.0f}  FPSEMPN={z40r['FPSEMPN']:.0f}  commercial={z40r['commercial']:.0f}")

# ---- 2. EDU SIZE TERM ----
edu_cols_map = {'Primary':1.0,'Secondary':1.0,'Education':0.9,'HSENROLL':0.85}
avail_edu = {c:e for c,e in edu_cols_map.items() if c in lu.columns}
print(f"\n=== EDU SIZE TERM cols available: {list(avail_edu.keys())} ===")
if avail_edu:
    lu['edu_raw'] = sum(
        lu[c].fillna(0).clip(lower=0)**e for c,e in avail_edu.items()
    )
    lu['edu_log'] = np.log(lu['edu_raw'].clip(lower=1e-6))
    lu['edu_dm']  = lu['edu_log'] - lu['edu_log'].mean()
    top_edu = lu.nlargest(15,'edu_dm')[[zcol,'edu_raw','edu_dm'] + list(avail_edu.keys())].rename(columns={zcol:'zone'})
    print(top_edu.to_string(index=False))
    z40e = lu[lu[zcol]==40].iloc[0]
    rank40e = int((lu['edu_dm'] > z40e['edu_dm']).sum()) + 1
    print(f"\nZone 40 edu rank: {rank40e}/{len(lu)}  dm={z40e['edu_dm']:+.3f}")
    for c in avail_edu:
        print(f"  {c}: {z40e[c]:.1f}")
else:
    print("  No edu cols found in landuse!")

# ---- 3. DISTANCE from zone 40 ----
try:
    import openmatrix as omx
    # 371-only: the retired 330-system downcasted_matrices_base.omx is never used.
    f = omx.open_file('../demand_/skims_371.omx','r')
    maps = f.listMappings()
    mapping = dict(f.mapping(maps[0])) if maps else {i+1:i for i in range(371)}
    dist_mat = np.array(f['DIST'])
    f.close()
    # mapping: zone_number -> row_index
    z40_idx = mapping.get(40)
    if z40_idx is not None:
        col = dist_mat[:, z40_idx]  # dist from each zone TO zone 40
        print(f"\n=== DIST to zone 40 ===")
        print(f"  mean={col.mean():.1f}  median={np.median(col):.1f}  p10={np.percentile(col,10):.1f}  p90={np.percentile(col,90):.1f}")
        print(f"  n_zero={int((col==0).sum())}  n_lt5={int((col<5).sum())}  n_gt50={int((col>50).sum())}")
        # Compare with zone 182 (the next big work zone)
        z182_idx = mapping.get(182)
        if z182_idx is not None:
            col182 = dist_mat[:, z182_idx]
            print(f"\n=== DIST to zone 182 (comparison) ===")
            print(f"  mean={col182.mean():.1f}  median={np.median(col182):.1f}  p10={np.percentile(col182,10):.1f}  p90={np.percentile(col182,90):.1f}")
    else:
        print(f"\nZone 40 not in OMX mapping. Keys: {list(mapping.keys())[:10]}")
except Exception as exc:
    print(f"\nOMX read error: {exc}")

# ---- 4. Validation vs new size term ----
val_path = 'runs/main/location_choice/work_destination_validation.csv'
if os.path.exists(val_path):
    val = pd.read_csv(val_path)
    val = val.merge(lu[[zcol,'ws_dm']].rename(columns={zcol:'zone'}), on='zone', how='left')
    val['pct_err'] = 100*(val['obs_workers'] - val['pred_workers_scaled']) / val['obs_workers'].clip(lower=1)
    print("\n=== Old validation vs new size term (zone 40 neighborhood) ===")
    sub = val[val['zone'].isin([38,39,40,41,42,43,182,208,64])][
        ['zone','obs_workers','pred_workers_scaled','pct_err','ws_dm']
    ].sort_values('zone')
    print(sub.to_string(index=False))

print("\nDone.")
