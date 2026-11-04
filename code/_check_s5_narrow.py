"""Prove the narrow Stage 5 read reproduces the wide read exactly.

Reads the same Stage 4 chunks twice — once parsing all 713 columns, once parsing
only ``S5_NARROW_COLS`` — builds a trip chain from each, and compares the two
plans row for row on every column the narrow read keeps.

Equality here is exact (no tolerance): the builder's only dependence on the
input schema is via ``if c in df.columns`` tests and the ``attr_cols``
carry-through, so if the columns it reads hold the same values the plans must be
identical.  Any difference means a column needed by the builder was left out of
the narrow list.

Usage
-----
    python _check_s5_narrow.py                       # 12k rows, one chunk
    python _check_s5_narrow.py --chunk 20000 --chunks 2 --persons 6000

Env: STAGE5_RERUN_RUN, STAGE5_CHECK_MODELS override the run dir / bundle.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

import _rerun_stage5_paper as drv           # sets the pipeline env vars

MODELS = os.environ.get(
    'STAGE5_CHECK_MODELS',
    os.path.join(drv.RUN, 'stage5_models_prefit.pkl'))

# Columns Stage 6 / Stage 6b read off the plan; the narrow plan must carry them
# even though Stage 5 never looks at them.
PLAN_CONTRACT = (
    'person_id', 'household_id', 'tour_id', 'row_type', 'seq', 'purpose',
    'OTAZ', 'DTAZ', 'STARTIME', 'ENDTIME', 'mode', 'mode_name', 'travtime',
    'predicted_duration_min', 'incp', 'indp', 'vehrd', 'nprd', 'sexp', 'age5p',
    'WORKSTAT', 'STUDYING', 'CARLICENCE', 'WORK_TAZ', 'SCHOOL_TAZ',
    'HOME_SA1', 'HOME_LVL4', 'dist', 'log_dist', 'tt_avg',
)


def _rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float('nan')


def build(usecols, chunk_size, n_chunks, models, landuse, skims, skim_df,
          skim_lookup, label):
    from stage5_destination_choice import _attach_skims, _build_trip_chain

    plans, persons = [], 0
    read_s, build_s = 0.0, 0.0
    rss_before = _rss()
    peak = rss_before
    cols_read = None

    t_all = time.time()
    # Time each generator step: person_chunks parses the CSV lazily, so the read
    # cost lands in next(), not in the loop body.
    it = drv.person_chunks(drv.S4_CSV, chunk_size, usecols=usecols)
    for i in range(n_chunks):
        t_read0 = time.time()
        try:
            chunk = next(it)
        except StopIteration:
            break
        t_read1 = time.time()
        read_s += t_read1 - t_read0
        peak = max(peak, _rss())
        pp = drv.chunk_persons(chunk)
        plan = _build_trip_chain(chunk, models, pp, landuse, skims,
                                skim_lookup, seed=42)
        plan = _attach_skims(plan, skim_df)
        t_build1 = time.time()
        build_s += t_build1 - t_read1
        plans.append(plan)
        persons += len(pp)
        cols_read = len(chunk.columns)
        print(f'  [{label}] chunk {i}: {len(chunk):,} rows '
              f'x {len(chunk.columns)} cols -> {len(plan):,} plan rows '
              f'({t_read1 - t_read0:.1f}s read, {t_build1 - t_read1:.0f}s '
              f'build, RSS {_rss():,.0f} MB)', flush=True)
        del chunk, pp, plan
        gc.collect()
        peak = max(peak, _rss())

    total = time.time() - t_all
    print(f'  [{label}] {persons:,} persons, {cols_read} columns parsed, '
          f'{total:.0f}s total, peak RSS {peak:,.0f} MB '
          f'(+{peak - rss_before:,.0f} MB)', flush=True)
    return (pd.concat(plans, ignore_index=True) if plans else pd.DataFrame(),
            {'persons': persons, 'cols': cols_read, 'secs': total,
             'read_s': read_s, 'build_s': build_s, 'peak_mb': peak,
             'delta_mb': peak - rss_before})


def _read_only(args, narrow, header) -> int:
    """Parse the same chunks twice and report only what parsing costs."""
    out = {}
    for label, usecols in (('wide', None), ('narrow', narrow)):
        gc.collect()
        base = _rss()
        peak, secs, rows, ncols, n_chunks = base, 0.0, 0, 0, 0
        it = drv.person_chunks(drv.S4_CSV, args.chunk, usecols=usecols)
        for i in range(args.chunks):
            t0 = time.time()
            try:
                chunk = next(it)
            except StopIteration:
                break
            secs += time.time() - t0
            peak = max(peak, _rss())
            rows += len(chunk)
            ncols = len(chunk.columns)
            n_chunks += 1
            del chunk
            gc.collect()
            peak = max(peak, _rss())
        out[label] = {'secs': secs, 'rows': rows, 'cols': ncols,
                      'peak_mb': peak, 'delta_mb': peak - base,
                      'chunks': n_chunks}
        s = out[label]
        print(f'  [{label}] {s["chunks"]} chunks, {s["rows"]:,} rows x '
              f'{s["cols"]} cols in {s["secs"]:.1f}s '
              f'({s["rows"] / max(s["secs"], 1e-9):,.0f} rows/s), '
              f'peak RSS {s["peak_mb"]:,.0f} MB (+{s["delta_mb"]:,.0f} MB)',
              flush=True)

    w, n = out['wide'], out['narrow']
    print('\n=== parse cost')
    print(f'  columns      {w["cols"]:5d} -> {n["cols"]:5d}')
    print(f'  wall         {w["secs"]:6.1f}s -> {n["secs"]:6.1f}s  '
          f'({w["secs"] / max(n["secs"], 1e-9):.1f}x faster)')
    print(f'  rows/s       {w["rows"] / max(w["secs"], 1e-9):8,.0f} -> '
          f'{n["rows"] / max(n["secs"], 1e-9):8,.0f}')
    print(f'  RSS delta    {w["delta_mb"]:7,.0f} MB -> {n["delta_mb"]:7,.0f} MB')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=12000)
    ap.add_argument('--chunks', type=int, default=1)
    ap.add_argument('--persons', type=int, default=None,
                    help='stop after this many persons (rounded up to a chunk)')
    ap.add_argument('--models', default=MODELS)
    ap.add_argument('--read-only', action='store_true',
                    help='measure parsing cost only (no models, skims or chains)')
    args = ap.parse_args(argv)

    header = pd.read_csv(drv.S4_CSV, nrows=0).columns.tolist()
    narrow = [c for c in drv.S5_NARROW_COLS if c in header]
    missing = [c for c in drv.S5_NARROW_COLS if c not in header]
    print(f'Stage 4 frame: {len(header)} columns')
    print(f'narrow read : {len(narrow)} columns'
          + (f'  (absent from frame, skipped: {missing})' if missing else ''))
    print(f'not read    : {len(header) - len(narrow)} columns\n', flush=True)

    if args.read_only:
        return _read_only(args, narrow, header)

    import pipeline_logger as pl
    from stage5_destination_choice import (_open_skims, _skim_flat,
                                           _skim_lookup)

    print(f'loading Stage 5 bundle {os.path.basename(args.models)}')
    models = drv.load_s5_models(args.models)
    print(f'segments: {list(models)}', flush=True)

    print('loading pipeline data (HHTS + landed use)')
    _, _out, landuse = pl.load_data()
    del _out
    gc.collect()

    skims, opened = _open_skims(pl.PIPELINE_SKIMS_PATH)
    skim_df = _skim_flat(skims)
    skim_lookup = _skim_lookup(skim_df)
    print(f'skim table: {len(skim_df):,} OD pairs\n', flush=True)

    n_chunks = args.chunks
    if args.persons:
        approx = max(1, -(-args.persons // max(1, args.chunk // 3)))
        n_chunks = max(n_chunks, approx)

    try:
        print('=== wide read (all columns)')
        wide, wstat = build(None, args.chunk, n_chunks, models, landuse, skims,
                            skim_df, skim_lookup, 'wide')
        gc.collect()
        print('\n=== narrow read (S5_NARROW_COLS)')
        narrowdf, nstat = build(narrow, args.chunk, n_chunks, models, landuse,
                                skims, skim_df, skim_lookup, 'narrow')
    finally:
        if opened:
            skims.close()

    print('\n=== comparison')
    ok = True
    if len(wide) != len(narrowdf):
        print(f'FAIL row count: wide {len(wide):,} vs narrow {len(narrowdf):,}')
        return 1
    print(f'row count: {len(wide):,} both')

    extra = [c for c in narrowdf.columns if c not in wide.columns]
    if extra:
        print(f'FAIL narrow produced columns the wide plan lacks: {extra}')
        ok = False

    key = ['person_id', 'seq']
    ws = wide.sort_values(key).reset_index(drop=True)
    ns = narrowdf.sort_values(key).reset_index(drop=True)

    diffs = []
    for c in narrowdf.columns:
        a, b = ws[c], ns[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.array_equal(a.to_numpy(dtype='float64', na_value=np.nan),
                                  b.to_numpy(dtype='float64', na_value=np.nan),
                                  equal_nan=True)
        else:
            same = (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all()
        if not same:
            n_bad = int((a.astype(str).to_numpy()
                         != b.astype(str).to_numpy()).sum())
            diffs.append((c, n_bad))
    if diffs:
        print('FAIL columns differ (n rows differing):')
        for c, n_bad in diffs[:20]:
            print(f'   {c}: {n_bad:,}')
        ok = False
    else:
        print(f'all {len(narrowdf.columns)} narrow plan columns identical')

    dropped = [c for c in wide.columns if c not in narrowdf.columns]
    print(f'columns dropped from the plan: {len(dropped)} '
          f'(carry-through only, e.g. {dropped[:8]})')

    absent = [c for c in PLAN_CONTRACT if c not in narrowdf.columns]
    print('downstream contract: '
          + ('complete' if not absent else f'MISSING {absent}'))
    if absent:
        ok = False

    print('\n=== cost')
    print(f'  parsed columns   {wstat["cols"]:5d} -> {nstat["cols"]:5d}')
    print(f'  chunk read       {wstat["read_s"]:6.1f}s -> '
          f'{nstat["read_s"]:6.1f}s  '
          f'({wstat["read_s"] / max(nstat["read_s"], 1e-9):.1f}x faster)')
    print(f'  chain build      {wstat["build_s"]:6.1f}s -> '
          f'{nstat["build_s"]:6.1f}s')
    print(f'  peak RSS delta   {wstat["delta_mb"]:7,.0f} MB -> '
          f'{nstat["delta_mb"]:7,.0f} MB')

    print('\n' + ('PASS: narrow read is equivalent to the wide read'
                  if ok else 'FAIL: narrow read diverges from the wide read'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
