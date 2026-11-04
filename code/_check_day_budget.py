"""Scratch check: does the Stage 3 day-window reconciliation remove the
end-of-day overrun that Stage 4 had to absorb?

Applies reconcile_day_budget() to a run's Stage 3 durations and re-places the
same activities, so the only difference is the duration target.
"""
import os
import sys
import time
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage3_activity_duration import reconcile_day_budget        # noqa: E402
from stage4_schedule_placement import predict_schedule, load_models  # noqa: E402
from _check_s4_redraw import period_consistency, order_checks    # noqa: E402

RUN = 'runs/agds_default_staged__paper_predict_20261024'
NROWS = 200_000
SEED = 42

logging.basicConfig(level=logging.WARNING, format='%(levelname)s  %(message)s')


def reconcile(acts: pd.DataFrame) -> tuple:
    """Apply the Stage 3 rule per person, as predict_activity_durations does."""
    out = acts.copy()
    info_rows = []
    for pid, grp in out.groupby('person_id', sort=False):
        B = float(grp['B'].iloc[0])
        B_min = B * 60.0 if B <= 24 else B
        window = float(grp['end_of_day'].iloc[0]) - float(grp['start_of_day'].iloc[0])
        arr, info = reconcile_day_budget(
            grp['predicted_duration_min'].to_numpy(dtype=float),
            grp['purpose'].to_numpy(), B_min, window_min=window,
        )
        out.loc[grp.index, 'predicted_duration_min'] = arr
        info_rows.append(info)
    return out, pd.DataFrame(info_rows)


def report(name: str, df: pd.DataFrame) -> dict:
    eod = df['end_of_day'].to_numpy(dtype=float)
    end = df['ENDTIME'].to_numpy(dtype=float)
    start = df['STARTIME'].to_numpy(dtype=float)
    pc = period_consistency(df)
    return {
        'name': name,
        'end_after_eod_%': round(100 * float((end > eod + 1).mean()), 2),
        'mean_dur': round(float(df['predicted_duration_min'].mean()), 1),
        'mean_start_hr': round(float(np.nanmean(start) / 60.0), 2),
        'start_inside_period_%': round(100 * pc['start_inside_label'], 2),
        'hourly': pc['hourly'],
        'chain': order_checks(df),
        # rows the builder flagged as having no room at all
        'no_room_rows': int(np.sum(end > eod + 1)),
    }


def main() -> None:
    acts = pd.read_csv(os.path.join(RUN, 'stage3_output.csv'), nrows=NROWS)
    models = load_models(os.path.join(RUN, 'stage4_models.pkl'))
    print(f'stage3 chunk: {acts.shape}  persons={acts["person_id"].nunique():,}')

    # person-level feasibility of the chain (what reconciliation targets)
    g = acts.groupby('person_id').agg(n=('purpose', 'size'),
                                      dur=('predicted_duration_min', 'sum'),
                                      sod=('start_of_day', 'first'),
                                      eod=('end_of_day', 'first'),
                                      B=('B', 'first'))
    W = g['eod'] - g['sod']
    room = W - 10.0 * g['n']
    need = g['dur'] - room
    print(f'\nbaseline: persons whose durations + gaps exceed the window: '
          f'{int((need > 1e-6).sum()):,}/{len(g):,} ({100*(need > 1e-6).mean():.1f}%), '
          f'median excess {np.median(need[need > 0]):.0f} min' if (need > 0).any() else '')

    rec, info = reconcile(acts)
    changed = (rec['predicted_duration_min'] - acts['predicted_duration_min']).abs() > 1e-9
    print(f'reconciled: {int(changed.sum()):,}/{len(acts):,} activities changed '
          f'({100*changed.mean():.1f}%), window-bound persons '
          f'{100*info["window_bound"].mean():.1f}%, floor-bound '
          f'{100*info["floor_bound"].mean():.1f}%')
    print('  mean duration by purpose (before -> after):')
    for p in sorted(acts['purpose'].unique()):
        m = acts['purpose'] == p
        print(f'    {p:10s} {acts.loc[m, "predicted_duration_min"].mean():6.1f} -> '
              f'{rec.loc[m, "predicted_duration_min"].mean():6.1f}')

    rows = []
    outs = {}
    for name, frame in (('before', acts), ('after', rec)):
        t0 = time.time()
        out = predict_schedule(frame.copy(), models, sample=True, seed=SEED)
        rows.append({**report(name, out), 'sec': round(time.time() - t0, 1)})
        outs[name] = out

    print('\n=== placement ===')
    for r in rows:
        print(f"  {r['name']:6s} end>eod {r['end_after_eod_%']:6.2f}%  "
              f"mean_dur {r['mean_dur']:6.1f}  mean_start {r['mean_start_hr']:.2f}h  "
              f"start_in_period {r['start_inside_period_%']:5.1f}%  "
              f"chain_before_prev_end {r['chain']['chain_starts_before_prev_end']:.3f}  "
              f"end>24h {r['chain']['end_after_24h']:.4f}  ({r['sec']}s)")

    h0, h1 = rows[0]['hourly'].astype(float), rows[1]['hourly'].astype(float)
    p0, p1 = h0 / h0.sum(), h1 / h1.sum()
    print('\n=== hourly departures: before vs after ===')
    print(f"  r = {np.corrcoef(p0, p1)[0, 1]:.4f}  "
          f"max |diff| = {100 * np.abs(p1 - p0).max():.2f} pp")

    # vs the shipped run's own Stage 4 output (old clamped builder)
    try:
        shipped = pd.read_csv(os.path.join(RUN, 'stage4_output.csv'),
                              nrows=len(acts), usecols=['STARTIME'])
        s = shipped['STARTIME'].to_numpy(dtype=float)
        hs = np.bincount(np.clip((s // 60).astype(int), 0, 23), minlength=24).astype(float)
        ps = hs / hs.sum()
        print('\n=== hourly departures: shipped output vs final ===')
        print(f"  r = {np.corrcoef(ps, p1)[0, 1]:.4f}  "
              f"max |diff| = {100 * np.abs(p1 - ps).max():.2f} pp  "
              f"mean start {s.mean()/60:.2f}h -> "
              f"{np.nanmean(outs['after']['STARTIME'])/60:.2f}h")
    except Exception as exc:            # noqa: BLE001
        print(f'  (shipped comparison skipped: {exc})')
    for h in range(24):
        print(f"    {h:02d}:00  {100*p0[h]:5.2f}%  {100*p1[h]:5.2f}%  {100*(p1[h]-p0[h]):+5.2f}")

    # who still runs out of day? should be the chains that do not fit at all
    out = outs['after'].reset_index(drop=True)
    over = out['ENDTIME'].to_numpy(dtype=float) > out['end_of_day'].to_numpy(dtype=float) + 1
    ids = out.loc[over, 'person_id'].unique()
    g3 = rec.groupby('person_id').agg(n=('purpose', 'size'),
                                      dur=('predicted_duration_min', 'sum'),
                                      sod=('start_of_day', 'first'),
                                      eod=('end_of_day', 'first'))
    infeasible = set(g3.index[(g3['dur'] + 10.0 * g3['n']) > (g3['eod'] - g3['sod']) + 1e-6])
    print(f"\npersons with a row past end_of_day: {len(ids):,} "
          f"({100*len(ids)/len(g3):.2f}% of persons); of those, "
          f"{len(set(ids) - infeasible):,} had a chain that does fit -> leaked")
    print(f"persons whose chain cannot fit (durations + gaps > window): "
          f"{len(infeasible):,} ({100*len(infeasible)/len(g3):.2f}%), "
          f"mean activities {g3.loc[list(infeasible), 'n'].mean():.1f}")

    # per-activity feasibility after reconciliation, as Stage 4 sees it
    g2 = rec.groupby('person_id').agg(n=('purpose', 'size'),
                                      dur=('predicted_duration_min', 'sum'),
                                      sod=('start_of_day', 'first'),
                                      eod=('end_of_day', 'first'))
    need2 = g2['dur'] + 10.0 * g2['n'] - (g2['eod'] - g2['sod'])
    print(f'\nafter: persons whose durations + gaps still exceed the window: '
          f'{int((need2 > 1e-6).sum()):,}/{len(g2):,} ({100*(need2 > 1e-6).mean():.2f}%)')


if __name__ == '__main__':
    main()
