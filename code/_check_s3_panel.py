"""Scratch: validate the refitted Stage 3 models, and the old ones, with the
current panel code so the before/after table is measured the same way.
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
from validate_hhts import validate_stage3                               # noqa: E402

RUN = 'runs/agds_default_staged__paper_predict_20261024'
OUT = '_tmp_val_stage3_refit'
COLS = ['purpose', 'n_obs', 'obs_mean_min', 'obs_mean_capped_min',
        'raw_mean_min', 'pred_mean_min', 'raw_bias_pct', 'mean_bias_pct',
        'mean_bias_vs_unclipped_pct']


def run(out, models, tag):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    res = validate_stage3(out, models, d)
    df = pd.read_csv(os.path.join(d, 'val_stage3_duration_summary.csv'))
    print(f'\n=== {tag}', flush=True)
    print(df[COLS].round(1).to_string(index=False), flush=True)
    m = res['metrics']
    print('mean |bias|  raw=%.1f%%  pipeline=%.1f%%  (vs unclipped obs %.1f%%)'
          % (m.get('mean_abs_bias_raw_pct'), m.get('mean_abs_bias_pct'),
             m.get('mean_abs_bias_vs_unclipped_pct')), flush=True)
    print('imputed:', m.get('imputed_features'), flush=True)
    return df


def main():
    out = Synth_Connect().process_survey(save=False)
    with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as fh:
        old = pickle.load(fh)
    with open(os.path.join(OUT, 'stage3_models_refit.pkl'), 'rb') as fh:
        new = pickle.load(fh)

    old_df = run(out, old, 'old')
    new_df = run(out, new, 'new')

    merged = old_df[['purpose', 'obs_mean_capped_min', 'raw_mean_min',
                     'pred_mean_min']].merge(
        new_df[['purpose', 'raw_mean_min', 'pred_mean_min',
                'mean_bias_pct']],
        on='purpose', suffixes=('_old', '_new'))
    merged['raw_bias_old_pct'] = (
        (merged['raw_mean_min_old'] - merged['obs_mean_capped_min'])
        / merged['obs_mean_capped_min'] * 100)
    merged['raw_bias_new_pct'] = (
        (merged['raw_mean_min_new'] - merged['obs_mean_capped_min'])
        / merged['obs_mean_capped_min'] * 100)
    print('\n=== before / after (capped observed comparator)', flush=True)
    print(merged[['purpose', 'obs_mean_capped_min', 'raw_mean_min_old',
                  'raw_mean_min_new', 'raw_bias_old_pct', 'raw_bias_new_pct',
                  'mean_bias_pct']].round(1).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
