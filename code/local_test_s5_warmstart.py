#!/usr/bin/env python
"""Local test of the Stage-5 validated warm-start (fail-fast + minimal fallback).

Runs the REAL _ga_stage5_segment_worker in-process with MOCKED fitness fns on
the tiny smoke dataset, so no HPC data/skims walltime is needed:
  Test A: every fit fails  -> worker must raise RuntimeError in seconds
          (fail fast), NOT burn walltime on -1e9 evals.
  Test B: all-ones baseline fails, minimal spec succeeds -> worker must swap
          the seed to the minimal spec and complete the segment (no raise).

Run from the code dir:  python local_test_s5_warmstart.py
"""
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

os.environ['PANDAS_USE_NUMEXPR'] = '0'
os.environ['STAGE5_BATCH_SIZE'] = '1'
os.environ['GA_AGDS_MXL_SEARCH'] = '0'
os.environ.pop('GA_WALLTIME_SECONDS', None)  # guard inactive unless set

import numpy as np

from data_connect import Synth_Connect
import ga_hypothesis_testing as G


def _load():
    sc = Synth_Connect()
    out = sc.process_survey(save=False)
    land = sc.get_land()
    tours = out['trips'] if 'trips' in out else out['tours']
    persons = out['persons']
    if 'purpose' in tours.columns and (tours['purpose'] == 'work').any():
        seg_tours = tours[tours['purpose'] == 'work'].head(60).copy()
    else:
        seg_tours = tours.head(60).copy()
    return tours, persons, land, seg_tours


def _payload(tours, persons, land, seg_tours, skim_df):
    return {
        'seg': 'work',
        'seg_label': 'Work',
        'tours': tours,
        'persons': persons,
        'landuse': land,
        'skim_df': skim_df,
        'seg_tours': seg_tours,
        'seg_mode_cands': ['CRASH', 'emp_density', 'hh_couple'],
        'seg_dest_cands': ['c_cbd_dest', 'c_dist_sq', 'is_cbd'],
        'ga_kwargs': dict(pop_size=8, n_generations=4, min_features=1,
                           stagnation_limit=2, mutation_rate=0.2),
        'n_restarts': 0,
        'skims_path': os.path.join(CODE, 'downcasted_matrices_base.omx'),
    }


def _skim_df():
    import openmatrix as omx
    from stage5_destination_choice import _skim_flat
    h = omx.open_file(os.path.join(CODE, 'downcasted_matrices_base.omx'), 'r')
    try:
        return _skim_flat(h)
    finally:
        h.close()


def test_a_fail_fast(tours, persons, land, seg_tours, skim_df):
    print('== Test A: all fits fail -> expect RuntimeError, fast ==', flush=True)
    G._stage5_mode_only_fitness = lambda *a, **k: -1e9
    G._stage5_dest_only_fitness = lambda *a, **k: -1e9
    import time
    t0 = time.time()
    try:
        G._ga_stage5_segment_worker(
            _payload(tours, persons, land, seg_tours, skim_df))
    except RuntimeError as e:
        dt = time.time() - t0
        msg = str(e)
        ok = ('warm-start FAILED' in msg) and (dt < 600)
        print(f'  RuntimeError in {dt:.1f}s: {msg[:150]}', flush=True)
        print(f'  -> A {"PASS" if ok else "FAIL"}', flush=True)
        return ok
    print('  -> A FAIL (no RuntimeError raised!)', flush=True)
    return False


def test_b_fallback(tours, persons, land, seg_tours, skim_df):
    print('== Test B: all-ones fails, minimal succeeds -> expect completion ==',
          flush=True)
    importlib_reload = __import__('importlib').reload  # noqa
    del importlib_reload

    def fake_mode(genes, *a, **k):
        g = np.asarray(genes, dtype=int)
        if g.sum() == len(g):
            return -1e9  # all-ones baseline fails
        return -2.0 - 0.01 * float(g.sum())  # valid, sparser is better

    def fake_dest(genes, *a, **k):
        g = np.asarray(genes, dtype=int)
        if g.sum() == len(g):
            return -1e9
        return -1.5 - 0.01 * float(g.sum())

    G._stage5_mode_only_fitness = fake_mode
    G._stage5_dest_only_fitness = fake_dest
    try:
        res = G._ga_stage5_segment_worker(
            _payload(tours, persons, land, seg_tours, skim_df))
    except RuntimeError as e:
        print(f'  -> B FAIL (unexpected RuntimeError: {str(e)[:150]})', flush=True)
        return False
    ok = (res.get('completed', False) is True
          and res.get('walltime_hit', True) is False
          and len(res.get('best_mode_vars', [])) >= 1)
    print(f"  completed={res.get('completed')} walltime_hit={res.get('walltime_hit')} "
          f"mode_vars={res.get('best_mode_vars')} dest_vars={res.get('best_dest_vars')}",
          flush=True)
    print(f'  -> B {"PASS" if ok else "FAIL"}', flush=True)
    return ok


def main() -> int:
    print('loading smoke data ...', flush=True)
    tours, persons, land, seg_tours = _load()
    print(f'data: tours={len(tours)} persons={len(persons)} seg_tours={len(seg_tours)}',
          flush=True)
    skim_df = _skim_df()
    print(f'skim_df: {skim_df.shape}', flush=True)
    okA = test_a_fail_fast(tours, persons, land, seg_tours, skim_df)
    okB = test_b_fallback(tours, persons, land, seg_tours, skim_df)
    print(f'\nRESULT: {"ALL PASS" if (okA and okB) else "FAIL"}', flush=True)
    return 0 if (okA and okB) else 1


if __name__ == '__main__':
    sys.exit(main())
