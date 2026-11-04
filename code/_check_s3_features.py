"""Scratch: why do the Stage 3 draws undershoot for shopping?

Compares, per purpose, the in-sample predicted mean when a fitted feature is
missing from the frame and gets zero-filled (current behaviour) versus filled
with the model's training mean (neutral, z=0), and after rebuilding the GA
columns the way estimation does.
"""
import logging
import os
import pickle

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format='%(levelname)s  %(message)s')

from data_connect import Synth_Connect                                  # noqa: E402
from stage3_activity_duration import (                                   # noqa: E402
    prepare_duration_data, _augment_duration_features, DURATION_BOUNDS,
)

RUN = 'runs/agds_default_staged__paper_predict_20261024'
PURPS = ['work', 'edu', 'social', 'escort', 'shopping', 'othmaint',
         'othdiscr', 'eat']


def matrix(subset, feats, model, fill):
    X = np.zeros((len(subset), len(feats)), dtype=float)
    for j, f in enumerate(feats):
        if f in subset.columns:
            X[:, j] = pd.to_numeric(subset[f], errors='coerce').fillna(0).values
        elif fill == 'mean':
            X[:, j] = model.mu_[j]
    return X


def mean_pred(model, X):
    eta = model._eta(X)
    return float(np.mean(np.exp(eta + 0.5 * model.sigma_ ** 2)))


def main():
    out = Synth_Connect().process_survey(save=False)
    data = _augment_duration_features(
        prepare_duration_data(out['tours'], out['persons'], trips=out.get('trips'))
    )
    with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as fh:
        models = pickle.load(fh)
    pcol = 'main_purpose'
    print(f'\nframes: {len(data)} activity rows, purpose col {pcol}', flush=True)

    purp_feats = {p: models[p]['features'] for p in PURPS if p in models}
    missing_all = {p: [f for f in fs if f not in data.columns]
                   for p, fs in purp_feats.items()}
    print('\nmissing at validation time:')
    for p, miss in missing_all.items():
        print(f'  {p:9s} {len(miss)}/{len(purp_feats[p])} -> {miss}', flush=True)

    # rebuild the GA columns the way estimation does
    enriched = data
    try:
        from ga_hypothesis_testing import ensure_ga_search_columns
        _want_pt = {str(f) for fs in purp_feats.values() for f in fs
                    if str(f).startswith('ptype_')}
        _attrs = dict(data.attrs)
        enriched = ensure_ga_search_columns(data, out['persons'],
                                            ptype_subset=_want_pt,
                                            stage_id=3, zone_col='DESTSA2_2021')
        enriched.attrs.update(_attrs)
        print('\nafter ensure_ga_search_columns:', flush=True)
        for p, fs in purp_feats.items():
            miss = [f for f in fs if f not in enriched.columns]
            print(f'  {p:9s} still missing {miss}', flush=True)
    except Exception as exc:
        print('enrichment failed:', exc, flush=True)

    rows = []
    for p in PURPS:
        if p not in models:
            continue
        entry = models[p]
        m, feats = entry['model'], entry['features']
        mask = (data[pcol] == p).to_numpy()
        if not mask.any():
            continue
        obs = float(pd.to_numeric(data.loc[mask, 'duration'], errors='coerce').mean())
        e_obs = float(pd.to_numeric(enriched.loc[mask, 'duration'], errors='coerce').mean())
        zi = mean_pred(m, matrix(data[mask], feats, m, 'zero'))
        mi = mean_pred(m, matrix(data[mask], feats, m, 'mean'))
        en = mean_pred(m, matrix(enriched[mask], feats, m, 'mean'))
        rows.append({'purpose': p, 'n': int(mask.sum()), 'obs': obs,
                     'obs_enriched': e_obs, 'zero_fill': zi,
                     'mean_fill': mi, 'enriched': en,
                     'fit_mean_obs': entry['mean_obs'],
                     'fit_mean_pred': entry['mean_pred_uncal'],
                     'sigma': round(float(m.sigma_), 3)})
    df = pd.DataFrame(rows)
    df['bias_zero'] = (df.zero_fill - df.obs) / df.obs * 100
    df['bias_mean'] = (df.mean_fill - df.obs) / df.obs * 100
    df['bias_enr'] = (df.enriched - df.obs) / df.obs * 100
    print('\n' + df.round(1).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
