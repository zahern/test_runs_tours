import pandas as pd

coef = pd.read_csv('runs/stage5_smoke_models_coefficients.csv')
coef = coef[coef['segment'] == 'work'].copy()

lines = ['STAGE5_SMOKE_SUMMARY']
for mtype in ['mode', 'dest']:
    part = coef[coef['model_type'] == mtype].copy()
    # Treat std_err==0 as fixed/holdfast terms and exclude from significance ratio.
    free = part[pd.to_numeric(part['std_err'], errors='coerce') > 0].copy()
    t = pd.to_numeric(free['tstat'], errors='coerce')
    tf = t[t.notna()]
    sig = int((tf.abs() > 1.96).sum())
    tot = int(tf.shape[0])
    ratio = (sig / tot) if tot > 0 else float('nan')
    lines.append(f'{mtype}: sig={sig}/{tot} ratio={ratio:.3f}')

with open('runs/stage5_smoke_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('WROTE runs/stage5_smoke_report.txt')
