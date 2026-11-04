import pandas as pd

p = pd.read_csv(r'Z:\test_runs_tours\demand_\suncorp_telematics\results\mip_free\game_day_mip_plans.csv')
print('Fixtures:', p['fixture'].unique())
print()
for fx in p['fixture'].unique():
    sub = p[p['fixture']==fx]
    print(f'{fx}:')
    print('  officers:', len(sub))
    print('  moves:', p[p['fixture']==fx]['n_moves'].tolist())
    for _, r in sub.iterrows():
        if r['n_moves'] > 0:
            print('  officer', int(r['officer']), ':', r['legs'])