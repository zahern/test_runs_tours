"""Scratch: refit Stage 3 with the all-ptype spec, then run the validation
panel on the refitted models and print the before/after duration comparison.
"""
import logging
import os
import pickle

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s  %(message)s')
for _noisy in ('matplotlib', 'PIL', 'searchlibrium'):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from data_connect import Synth_Connect                                  # noqa: E402
from stage3_activity_duration import estimate_duration_models           # noqa: E402
from validate_hhts import validate_stage3                               # noqa: E402

RUN = 'runs/agds_default_staged__paper_predict_20261024'
OUT = '_tmp_val_stage3_refit'
PURPS = ['work', 'edu', 'social', 'escort', 'shopping', 'othmaint',
         'othdiscr', 'eat']


def fit_table(models, tag):
    rows = []
    for p in PURPS:
        e = models.get(p)
        if not isinstance(e, dict):
            continue
        rows.append({'purpose': p, 'n_obs': e.get('n_obs'),
                     'sigma': round(float(e.get('sigma') or 0), 3),
                     'fit_obs': round(float(e.get('mean_obs') or 0), 1),
                     'fit_pred': round(float(e.get('mean_pred_uncal') or 0), 1),
                     'calib': round(float(e.get('calibration_factor') or 0), 3),
                     'nfeat': len(e.get('features') or [])})
    print(f'\n{tag}', flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def panel_table(csv_path):
    df = pd.read_csv(csv_path)
    print('\nvalidation panel:', flush=True)
    print(df[['purpose', 'n_obs', 'obs_mean_min', 'raw_mean_min',
              'pred_mean_min', 'raw_bias_pct', 'mean_bias_pct']]
          .round(1).to_string(index=False), flush=True)


def main():
    out = Synth_Connect().process_survey(save=False)
    print('survey frames:',
          {k: (v.shape if hasattr(v, 'shape') else type(v))
           for k, v in out.items()}, flush=True)

    with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as fh:
        old = pickle.load(fh)

    print('\nrefitting Stage 3 (all person types) ...', flush=True)
    new = estimate_duration_models(out['tours'], out['persons'],
                                   trips=out.get('trips'))
    fit_table(new, 'refitted models')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'stage3_models_refit.pkl'), 'wb') as fh:
        pickle.dump(new, fh)

    print('\nvalidating OLD models (zero-fill era) ...', flush=True)
    res_old = validate_stage3(out, old, OUT)
    panel_table(os.path.join(OUT, 'val_stage3_duration_summary.csv'))
    print('old imputed:', res_old['metrics'].get('imputed_features'),
          flush=True)

    print('\nvalidating NEW models ...', flush=True)
    res_new = validate_stage3(out, new, OUT + '/refit')
    panel_table(os.path.join(OUT, 'refit', 'val_stage3_duration_summary.csv'))
    m = res_new['metrics']
    print('new imputed:', m.get('imputed_features'), flush=True)
    print(f"\nmean |bias|  old raw={res_old['metrics'].get('mean_abs_bias_raw_pct'):.1f}%"
          f"  new raw={m.get('mean_abs_bias_raw_pct'):.1f}%"
          f"  new pipeline={m.get('mean_abs_bias_pct'):.1f}%", flush=True)


if __name__ == '__main__':
    main()
