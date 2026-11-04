"""Scratch check: Stage 4 period re-draw vs the old boundary clamp.

Runs predict_schedule on the first chunk of a run's Stage 3 output and compares
the result with that run's existing Stage 4 output (built by the clamped code).
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage4_schedule_placement import (          # noqa: E402
    predict_schedule, PERIOD_STARTS, PERIOD_ENDS, PERIOD_LABELS,
)

RUN = 'runs/agds_default_staged__paper_predict_20261024'
NROWS = 200_000

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')


def period_consistency(df: pd.DataFrame) -> dict:
    start = df['STARTIME'].to_numpy(dtype=float)
    labels = df['time_period'].astype(str).to_numpy()
    lo = np.array([PERIOD_STARTS[PERIOD_LABELS.index(l)] if l in PERIOD_LABELS else np.nan
                   for l in np.unique(labels)])
    idx = {l: i for i, l in enumerate(np.unique(labels))}
    lo_all = np.array([lo[idx[l]] for l in labels])
    hi_all = np.array([PERIOD_ENDS[PERIOD_LABELS.index(l)] if l in PERIOD_LABELS else np.nan
                       for l in labels])
    inside = (start >= lo_all) & (start < hi_all)
    overrun = (df['ENDTIME'].to_numpy(dtype=float) > df['end_of_day'].to_numpy(dtype=float))
    raw = np.asarray(start, dtype=float)
    hourly = np.bincount(np.clip((raw // 60).astype(int), 0, 23), minlength=24)
    return {
        'n': len(df),
        'start_inside_label': float(inside.mean()),
        'end_after_eod': float(overrun.mean()),
        'mean_start_hr': float(np.nanmean(raw) / 60.0),
        'mean_end_hr': float(df['ENDTIME'].mean() / 60.0),
        'hourly': hourly,
        'period_mix': df['time_period'].value_counts(normalize=True),
    }


def order_checks(df: pd.DataFrame) -> dict:
    """Within person: does the chain stay ordered, and inside the day window?"""
    d = df.sort_values(['person_id', 'schedule_seq'], kind='stable')
    pid = d['person_id'].to_numpy()
    s = d['STARTIME'].to_numpy(dtype=float)
    e = d['ENDTIME'].to_numpy(dtype=float)
    eod = d['end_of_day'].to_numpy(dtype=float)
    prev_e = np.empty(len(d), dtype=float)
    prev_e[0] = -np.inf
    prev_e[1:] = e[:-1]
    same = np.empty(len(d), dtype=bool)
    same[0] = False
    same[1:] = pid[1:] == pid[:-1]
    back = same & (s < prev_e - 1)
    overlap = same & (s < prev_e)
    return {
        'chain_starts_before_prev_end': float(back.mean()),
        'chain_overlaps_prev': float(overlap.mean()),
        'end_after_eod': float((e > eod + 1).mean()),
        'end_after_24h': float((e > 1440).mean()),
    }


def main() -> None:
    acts = pd.read_csv(os.path.join(RUN, 'stage3_output.csv'), nrows=NROWS)
    print(f'stage3 chunk: {acts.shape}')

    from stage4_schedule_placement import load_models
    models = load_models(os.path.join(RUN, 'stage4_models.pkl'))

    import time
    t0 = time.time()
    new = predict_schedule(acts.copy(), models, sample=True, seed=42)
    print(f'predict_schedule: {time.time() - t0:.1f}s for {len(acts):,} activities')
    old = pd.read_csv(os.path.join(RUN, 'stage4_output.csv'),
                      nrows=min(len(acts), 200_000))

    a, b = period_consistency(new), period_consistency(old)
    print('\n=== chain order ===')
    print('  new:', {k: round(v, 4) for k, v in order_checks(new).items()})
    print('  old:', {k: round(v, 4) for k, v in order_checks(old).items()})
    print('\n=== new (re-draw) ===')
    for k, v in a.items():
        print(f'  {k}: {v}' if not isinstance(v, (np.ndarray, pd.Series))
              else f'  {k}:\n{v.to_string() if isinstance(v, pd.Series) else v}')
    print('\n=== old (clamp) ===')
    for k, v in b.items():
        print(f'  {k}: {v}' if not isinstance(v, (np.ndarray, pd.Series))
              else f'  {k}:\n{v.to_string() if isinstance(v, pd.Series) else v}')

    overrun = (new['ENDTIME'].to_numpy(dtype=float)
               - new['end_of_day'].to_numpy(dtype=float))
    overrun = overrun[overrun > 1]
    if overrun.size:
        print('\nend-of-day overrun minutes (new):',
              np.round(np.percentile(overrun, [50, 75, 90, 99]), 1).tolist(),
              f'max={overrun.max():.0f}')

    print('\nhourly departures  hour   old     new')
    for h in range(24):
        print(f'  {h:02d}:00  {b["hourly"][h]:7d} {a["hourly"][h]:7d}')

    # ── why did activities run out of room? ────────────────────────────
    n_new = new.reset_index(drop=True)
    at_eod = np.isclose(n_new['ENDTIME'].to_numpy(dtype=float),
                        np.round(n_new['end_of_day'].to_numpy(dtype=float)), atol=1.0)
    print(f'\nno-room fallback rows (ENDTIME == end_of_day): {int(at_eod.sum()):,}')
    sub = n_new[at_eod]
    if len(sub):
        print('  duration_min: ', np.round(np.nanpercentile(sub['predicted_duration_min'],
                                                           [5, 25, 50, 75, 95]), 1))
        print('  end_of_day  : ', np.round(np.nanpercentile(sub['end_of_day'],
                                                           [5, 25, 50, 75, 95]), 1))
        print('  start_of_day: ', np.round(np.nanpercentile(sub['start_of_day'],
                                                           [5, 25, 50, 75, 95]), 1))
        print('  purposes    : ', sub['purpose'].value_counts(normalize=True).round(3).to_dict())
        if 'schedule_seq' in sub.columns:
            print('  share that are the person\u2019s first activity:',
                  round(float((sub['schedule_seq'] == 0).mean()), 3))
        print('  activities per person (all rows):',
              round(float(n_new['person_id'].map(
                  n_new['person_id'].value_counts()).mean()), 2))


if __name__ == '__main__':
    main()
