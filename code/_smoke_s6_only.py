import numpy as np
import pandas as pd
from data_connect import Synth_Connect
from stage6_mode_choice import estimate_mode_models


def sig_ratio_from_model(model):
    t = np.asarray(model.tstat_ if model.tstat_ is not None else [])
    tf = t[np.isfinite(t)]
    if tf.size > 0:
        sig = int((np.abs(tf) > 1.96).sum())
        tot = int(tf.size)
        return (sig / tot, sig, tot)

    pf = None
    if hasattr(model, '_final_pf') and isinstance(getattr(model, '_final_pf'), pd.DataFrame):
        pf = model._final_pf
    elif hasattr(model, 'pf') and isinstance(getattr(model, 'pf'), pd.DataFrame):
        pf = model.pf
    if pf is None:
        return (np.nan, 0, 0)

    tcol = 't_stat' if 't_stat' in pf.columns else ('tstat' if 'tstat' in pf.columns else None)
    if tcol is None:
        return (np.nan, 0, 0)

    ts = pd.to_numeric(pf[tcol], errors='coerce')
    if 'holdfast' in pf.columns:
        free = ~pf['holdfast'].fillna(False)
    else:
        free = pd.Series(True, index=pf.index)
    tf = ts[free & ts.notna()]
    sig = int((tf.abs() > 1.96).sum())
    tot = int(tf.shape[0])
    return ((sig / tot) if tot > 0 else np.nan, sig, tot)

sc = Synth_Connect()
out = sc.process_survey(save=False)
land = sc.get_land()
trips = out['trips']
if len(trips) > 12000:
    trips = trips.sample(n=12000, random_state=42).copy()
models = estimate_mode_models(trips, out['persons'], landuse=land, min_sample=80)
rows = []
for purpose, model in models.items():
    ratio, sig, tot = sig_ratio_from_model(model)
    rows.append({
        'purpose': purpose,
        'n_obs': int(model.n_obs_),
        'rho2': float(model.rho2_) if model.rho2_ is not None else np.nan,
        'sig_ratio': ratio,
        'sig_count': sig,
        'tot_count': tot,
        'engine': model._estimation_engine,
    })

df = pd.DataFrame(rows).sort_values('purpose')
with open('runs/stage6_smoke_report.txt', 'w', encoding='utf-8') as f:
    f.write('STAGE6_SMOKE_SUMMARY\n')
    f.write(df.to_string(index=False))
print('WROTE runs/stage6_smoke_report.txt')
