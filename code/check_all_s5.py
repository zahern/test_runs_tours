import pandas as pd
base = '/mnt/hpccs01/home/ahernz/test_runs_tours/code/runs/ga_parallel'
for tag in ['logsumfix', 'destfix', 'tddsched', 'accessfix']:
    try:
        df = pd.read_csv(f'{base}/{tag}/stage5/stage5_models_coefficients.csv')
        se = pd.to_numeric(df['std_err'], errors='coerce')
        dest = df[df['model_type']=='dest']
        dval = pd.to_numeric(dest['value'], errors='coerce')
        print(f"{tag}: stage5 rows={len(df)} NaN={se.isna().sum()} | dest rows={len(dest)} val==0:{(dval==0).sum()} range=[{dval.min():.3f},{dval.max():.3f}]")
    except Exception as e:
        print(f"{tag}: ERR {e}")
