import pandas as pd, numpy as np

lu = pd.read_csv('zone_order_and_landuse.csv')
emp_cols = ['FPSEMPN','HEREMPN','RETEMPN','MWTEMPN','OTHEMP','AGREMPN']
lu['TOTEMP'] = sum(lu[c].fillna(0) for c in emp_cols if c in lu.columns)

# Check university column
if 'university' in lu.columns:
    univ = lu.groupby('ORDER')['university'].sum() if False else lu.set_index('ORDER')['university']
    print('Top 20 zones by university POI count:')
    print(univ.sort_values(ascending=False).head(20).to_string())
    print()
    print('Zone 40 university =', lu[lu['ORDER']==40]['university'].values[0])
else:
    print('university column NOT found in zone_order_and_landuse.csv')

# Work size without commercial: TOTEMP^1 + FPSEMPN^0.85
lu['ws_nocomm'] = lu['TOTEMP']**1.0 + lu['FPSEMPN'].fillna(0)**0.85
lu['ws_nocomm_log'] = np.log(lu['ws_nocomm'].clip(lower=1e-6))
lu['ws_nocomm_dm'] = lu['ws_nocomm_log'] - lu['ws_nocomm_log'].mean()
print()
print('Work size WITHOUT commercial (TOTEMP^1 + FPSEMPN^0.85), top 15:')
top = lu.nlargest(15,'ws_nocomm_dm')[['ORDER','TOTEMP','FPSEMPN','ws_nocomm_dm']].rename(columns={'ORDER':'zone'})
print(top.to_string(index=False))
z40 = lu[lu['ORDER']==40].iloc[0]
rank = int((lu['ws_nocomm_dm'] > z40['ws_nocomm_dm']).sum()) + 1
print('Zone 40 rank (no commercial): %d/%d  dm=%.3f' % (rank, len(lu), z40['ws_nocomm_dm']))
print('Zone 158 dm=%.3f' % lu[lu['ORDER']==158]['ws_nocomm_dm'].values[0])
