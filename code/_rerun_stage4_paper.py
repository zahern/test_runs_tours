"""Re-run Stage 4 (schedule placement) for the paper run tag, in person chunks.

The Stage 3 fix changed the durations the run emits, so the placement pass has
to be repeated.  The timing models do not: ``stage4_models.pkl`` is fitted on
HHTS observed trips and never sees a synthetic duration, so it is reused as is
and this driver only repeats the prediction pass.

The whole pass does not fit on a 16 GB desktop — ``stage3_output.csv`` is 14 GB
and ``predict_schedule`` copies the frame it is handed — so the input is
streamed in person-sized chunks.  Chunks are aligned to ``person_id``
boundaries: the per-person schedule arithmetic and the ``cumcount`` day
position both assume a person's rows are contiguous, and a chunk boundary
through the middle of a person would schedule their morning and afternoon as
two independent days.  Completed chunks are recorded in
``_stage4_predict_progress.json`` and the output accumulates in
``stage4_output.csv.new`` until the last chunk renames it over the old file.

Usage
-----
    python _rerun_stage4_paper.py --predict [--chunk 20000] [--persons 20000]
    python _rerun_stage4_paper.py --predict --resume
    python _rerun_stage4_paper.py --header-check          # schema vs old file

Env: STAGE4_RERUN_RUN overrides the run directory.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time

import pandas as pd

try:                                       # NumPy 2.0 shim, same as the pipeline
    import compat.numpy_shim  # noqa: F401
except Exception:
    pass

RUN = os.path.abspath(os.environ.get(
    'STAGE4_RERUN_RUN',
    os.path.join('runs', 'agds_default_staged__paper_predict_20261024')))
S3_CSV = os.path.join(RUN, 'stage3_output.csv')
S4_CSV = os.path.join(RUN, 'stage4_output.csv')
S4_NEW = S4_CSV + '.new'
S4_BAK = S4_CSV + '.bak_pre_s3fix'
PROGRESS = os.path.join(RUN, '_stage4_predict_progress.json')
MODELS = os.path.join(RUN, 'stage4_models.pkl')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('s4_rerun')


def old_header() -> list:
    with open(S4_CSV, newline='') as fh:
        return next(csv.reader(fh))


def person_chunks(path: str, chunk_size: int):
    """Read CSV chunks that never split a person across two yields.

    The trailing block of the chunk is held back and prepended to the next one;
    a single person larger than ``chunk_size`` accumulates until they end.
    """
    pending: pd.DataFrame | None = None
    for raw in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk = raw if pending is None else pd.concat([pending, raw],
                                                      ignore_index=True)
        pending = None
        if len(chunk) == 0:
            continue
        pid = chunk['person_id'].to_numpy()
        tail = pid == pid[-1]
        if tail.all():
            pending = chunk
            continue
        pending = chunk.loc[tail].reset_index(drop=True)
        yield chunk.loc[~tail].reset_index(drop=True)
    if pending is not None and len(pending):
        yield pending


def reconcile_columns(acts: pd.DataFrame, needed: list) -> pd.DataFrame:
    """Force a scheduled chunk onto the run's recorded Stage 4 schema."""
    missing = [c for c in needed if c not in acts.columns]
    if missing:
        logger.warning(f'{len(missing)} expected columns missing after '
                       f'scheduling: {missing[:8]}')
        for c in missing:
            acts[c] = float('nan')
    extra = [c for c in acts.columns if c not in needed]
    if extra:
        logger.info(f'dropping {len(extra)} columns not in the recorded schema: '
                    f'{extra[:8]}')
    return acts.reindex(columns=list(needed))


def load_models():
    import pickle
    if not os.path.isfile(MODELS):
        raise SystemExit(f'stage4 models not found: {MODELS}')
    with open(MODELS, 'rb') as fh:
        models = pickle.load(fh)
    logger.info(f'loaded {len(models)} Stage 4 timing models from {MODELS}')
    return models


def _read_progress() -> dict:
    if os.path.isfile(PROGRESS):
        try:
            with open(PROGRESS) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {'chunks_done': 0, 'persons_done': 0, 'rows_written': 0}


def _write_progress(state: dict) -> None:
    tmp = PROGRESS + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(state, fh)
    os.replace(tmp, PROGRESS)


def predict(chunk_size: int = 20000, max_persons: int | None = None,
            resume: bool = False) -> None:
    from stage4_schedule_placement import predict_schedule

    models = load_models()
    state = _read_progress() if resume else {
        'chunks_done': 0, 'persons_done': 0, 'rows_written': 0}
    done = int(state.get('chunks_done', 0))
    mode = 'a' if resume and done and os.path.isfile(S4_NEW) else 'w'
    SCHEMA = old_header()
    logger.info(f'predicting Stage 4 into {S4_NEW} (mode={mode}, resume from '
                f'chunk {done}, {chunk_size:,} persons per chunk, '
                f'{len(SCHEMA)} columns)')

    t0 = time.time()
    persons = 0
    for i, chunk in enumerate(person_chunks(S3_CSV, chunk_size)):
        if i < done:
            continue
        if max_persons is not None and persons >= max_persons:
            break
        t_chunk = time.time()
        acts = predict_schedule(chunk, models, sample=True, seed=42)
        acts = reconcile_columns(acts, SCHEMA)
        acts.to_csv(S4_NEW, mode=mode, header=(i == 0), index=False)
        mode = 'a'
        persons += len(chunk)
        state.update({
            'chunks_done': i + 1,
            'persons_done': int(state.get('persons_done', 0)) + len(chunk),
            'rows_written': int(state.get('rows_written', 0)) + len(acts),
        })
        _write_progress(state)
        elapsed = time.time() - t0
        logger.info(f'chunk {i + 1}: {len(chunk):,} rows -> {len(acts):,} '
                    f'scheduled activities in {time.time() - t_chunk:.0f}s '
                    f'| {state["persons_done"]:,} rows done, '
                    f'{persons / max(elapsed, 1e-6) * 3600:,.0f} rows/h')
        del acts, chunk
        # predict_schedule copies the 610-column frame several times; releasing
        # those copies between chunks keeps peak RSS bounded on a 16 GB box.
        gc.collect()

    if max_persons is None:
        if os.path.isfile(S4_CSV):
            if os.path.isfile(S4_BAK):
                os.remove(S4_BAK)
            os.replace(S4_CSV, S4_BAK)
        os.replace(S4_NEW, S4_CSV)
        try:
            os.remove(PROGRESS)
        except OSError:
            pass
        logger.info(f'done: stage4_output.csv replaced '
                    f'({state["rows_written"]:,} activity rows)')
    else:
        logger.info(f'smoke run stopped after {state["persons_done"]:,} persons; '
                    f'{S4_NEW} kept, progress recorded')


def header_check(chunk_size: int = 20000) -> bool:
    """Schedule the first chunk and compare the produced schema to the old file."""
    from stage4_schedule_placement import predict_schedule

    models = load_models()
    chunk = next(person_chunks(S3_CSV, chunk_size))
    acts = predict_schedule(chunk, models, sample=True, seed=42)
    old_cols = old_header()
    raw_missing = [c for c in old_cols if c not in acts.columns]
    acts = reconcile_columns(acts, old_cols)
    ok = list(acts.columns) == old_cols
    print(f'old {len(old_cols)} cols, scheduled {len(acts.columns)} cols, '
          f'schema identical: {ok}')
    print('missing before reconcile:', raw_missing[:20])
    extra = [c for c in acts.columns if c not in old_cols]
    print('extra columns:', extra[:20])
    return ok


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--predict', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--header-check', action='store_true')
    ap.add_argument('--chunk', type=int, default=20000)
    ap.add_argument('--persons', type=int, default=None)
    args = ap.parse_args(argv)

    if not os.path.isdir(RUN):
        print(f'run dir not found: {RUN}', file=sys.stderr)
        return 2
    if args.header_check:
        return 0 if header_check(args.chunk) else 1
    if args.predict:
        predict(args.chunk, args.persons, args.resume)
    return 0


if __name__ == '__main__':
    sys.exit(main())
