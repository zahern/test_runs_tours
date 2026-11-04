"""Scratch: run the production Stage 3 prediction path with the refitted
models on real Stage 2 output, and check the emitted durations per purpose.
"""
import logging
import os
import pickle

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
for _noisy in ('matplotlib', 'PIL', 'searchlibrium'):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from data_connect import Synth_Connect                                  # noqa: E402
from stage2_day_duration import aggregate_person_day, predict_day_budget  # noqa: E402
from stage3_activity_duration import (                                   # noqa: E402
    predict_activity_durations, DURATION_BOUNDS,
)

RUN = 'runs/agds_default_staged__paper_predict_20261024'
OUT = '_tmp_val_stage3_refit'
PURPS = ['work', 'edu', 'social', 'escort', 'shopping', 'othmaint',
         'othdiscr', 'eat']
N_PERSONS = 5000


def main():
    out = Synth_Connect().process_survey(save=False)
    with open(os.path.join(RUN, 'stage2_models.pkl'), 'rb') as fh:
        s2 = pickle.load(fh)
    with open(os.path.join(OUT, 'stage3_models_refit.pkl'), 'rb') as fh:
        s3 = pickle.load(fh)

    day = aggregate_person_day(out['tours'], out['persons'])
    s2_pred = predict_day_budget(day.copy(), s2, sample=False, seed=42)
    sample = s2_pred.sample(n=min(N_PERSONS, len(s2_pred)), random_state=7)

    acts = predict_activity_durations(sample, s3, sample=True, seed=42)
    print('\nimputed features:', acts.attrs.get('imputed_features'), flush=True)

    # Distributions only: the observed comparison for these models is the
    # Stage 3 validation panel (validate_hhts.validate_stage3), which masks the
    # survey by purpose and caps to the bounds; the trips table here is
    # unscoped, so a per-purpose bias computed from it would be meaningless.
    rows = []
    for p in PURPS:
        m = acts['purpose'] == p
        if not m.any():
            continue
        vals = acts.loc[m, 'predicted_duration_min']
        lo, hi = DURATION_BOUNDS[p]
        rows.append({'purpose': p, 'n': int(m.sum()),
                     'pred_mean': round(float(vals.mean()), 1),
                     'p50': round(float(vals.median()), 1),
                     'p90': round(float(vals.quantile(0.9)), 1),
                     'bounds': f'{lo:.0f}-{hi:.0f}'})
    df = pd.DataFrame(rows)
    print('\nproduction Stage 3 output:', flush=True)
    print(df.to_string(index=False), flush=True)

    n_over = int((acts['predicted_duration_min'] > 960).sum())
    print(f'\nrows: {len(acts)}  persons: {acts["person_id"].nunique()}  '
          f'over 16h: {n_over}', flush=True)


if __name__ == '__main__':
    main()
