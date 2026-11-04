import pandas as pd
import numpy as np

base = '/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel/destfix'
df = pd.read_csv(f'{base}/stage5/stage5_models_coefficients.csv')
se = pd.to_numeric(df['std_err'], errors='coerce')
print(f"stage5 new rows: {len(df)}  NaN-stderr: {se.isna().sum()}  zero: {(se==0).sum()}")
df2 = pd.read_csv(f'{base}/stage5/stage5_coefficients.csv')
print(f"stage5 GA-selected rows: {len(df2)}")
# check dest model values are no longer pinned at 0 (ortho fix)
dest = df[df['model_type']=='dest']
zero_val = (pd.to_numeric(dest['value'], errors='coerce')==0).sum()
print(f"dest rows: {len(dest)}  value==0: {zero_val}")
print("dest value range:", pd.to_numeric(dest['value'], errors='coerce').describe()[['min','max']].to_dict())