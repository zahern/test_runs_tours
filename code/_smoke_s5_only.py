import numpy as np
import pandas as pd
from data_connect import Synth_Connect
from stage5_destination_choice import run_stage5


def sig_ratio_from_pf(m):
    pf = m._final_pf if hasattr(m, '_final_pf') and isinstance(getattr(m, '_final_pf'), pd.DataFrame) else m.pf
    tcol = 't_stat' if 't_stat' in pf.columns else ('tstat' if 'tstat' in pf.columns else None)
    if tcol is None:
        return (np.nan, 0, 0)
    t = pd.to_numeric(pf[tcol], errors='coerce')
    free = (~pf['holdfast'].fillna(False)) if 'holdfast' in pf.columns else pd.Series(True, index=pf.index)
    tf = t[free & t.notna()]
    sig = int((tf.abs() > 1.96).sum())
    tot = int(tf.shape[0])
    return ((sig / tot) if tot > 0 else np.nan, sig, tot)


sc = Synth_Connect()
out = sc.process_survey(save=False)
land = sc.get_land()
synth_s4 = out['tours'].sample(n=min(2500, len(out['tours'])), random_state=42).copy()
if len(synth_s4) > 1200:
    synth_s4 = synth_s4.sample(n=1200, random_state=42).copy()
plan, models = run_stage5(
    out['trips'],
    out['persons'],
    synth_s4,
    'downcasted_matrices_base.omx',
    land,
    tours=out['tours'],
    seed=42,
    save_path='runs/stage5_smoke_models.pkl',
    segments=['work'],
    verbose=False,
)

lines = ['STAGE5_SMOKE_SUMMARY']
if 'work' not in models:
    lines.append('work segment missing in stage5 models')
else:
    work = models['work']
    for mtype, key in [('mode', 'mode_model'), ('dest', 'dest_model')]:
        mdl = work.get(key)
        if mdl is None:
            lines.append(f'{mtype}: missing model')
            continue
        ratio, sig, tot = sig_ratio_from_pf(mdl)
        lines.append(f'{mtype}: sig={sig}/{tot} ratio={ratio:.3f}')

with open('runs/stage5_smoke_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('WROTE runs/stage5_smoke_report.txt')
