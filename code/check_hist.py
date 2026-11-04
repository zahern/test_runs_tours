import pandas as pd
base = '/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/destfix/stage5/ga_reports'
df = pd.read_csv(f'{base}/ga_stage5_history.csv')
print("stage5 overall history rows:", len(df))
print("gen range:", df['generation'].min(), '-', df['generation'].max())
print("best_so_far first/last:", df['best_so_far'].iloc[0], '/', df['best_so_far'].iloc[-1])
print("improvements (best_so_far changes):", df['best_so_far'].nunique())
imp = df[df['best_so_far'].diff() < 0]
print("num improvement steps:", len(imp))
print("best_fitness min:", df['best_fitness'].min(), "at gen:", df.loc[df['best_fitness'].idxmin(), 'generation'])

sd = pd.read_csv(f'{base}/ga_stage5_segment_history.csv')
print("\nsegments:", sd['segment'].unique())
for seg, g in sd.groupby('segment'):
    print(f"  {seg}: gens={g['generation'].min()}-{g['generation'].max()} best_so_far {g['best_so_far'].iloc[0]:.4f} -> {g['best_so_far'].iloc[-1]:.4f} improvements={g['best_so_far'].diff().lt(0).sum()}")