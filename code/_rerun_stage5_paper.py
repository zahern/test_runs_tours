"""Re-run Stage 5 (mode + destination + trip chain) for the paper run tag.

Stage 5 re-runs because the trip chain is built from the Stage 4 schedule, which
in turn is built from the Stage 3 durations that changed.

The model bundle is read with ``stage5_destination_choice.load_models``, which
uses cloudpickle.  Plain ``pickle.load`` on the same file raises
``Cannot set 'availability_any'`` and hands back a hollow object, so the stdlib
path must not be used here — but the cloudpickle path round-trips the fitted
Larch models correctly, which lets estimation, prediction and finalisation run
in separate processes on a machine that cannot hold all three at once.

``--estimate`` is only needed when the run has no ``stage5_models.pkl``; this
run's Stage 5 GA search never converged (``selected_features_summary.json``
carries the ``-1e9`` penalised sentinel), so the baseline specification is the
one the staged driver would use anyway.

The pass is streamed and resumable.  ``run_stage5`` builds chains for the whole
synthetic population at once, which needs the 17 GB Stage 4 frame plus a
home-zone dict over 4M persons; here the work is per person chunk, checkpointed
to ``stage5_batches_rerun/``, and the progress file records how many chunks are
done so an interrupted run resumes instead of recomputing.

Usage
-----
    python _rerun_stage5_paper.py --estimate --predict --assemble --finalize
    python _rerun_stage5_paper.py --predict --resume
    python _rerun_stage5_paper.py --assemble

Env: STAGE5_RERUN_RUN overrides the run directory.
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

import numpy as np
import pandas as pd

try:                                       # NumPy 2.0 shim, same as the pipeline
    import compat.numpy_shim  # noqa: F401
except Exception:
    pass

RUN = os.path.abspath(os.environ.get(
    'STAGE5_RERUN_RUN',
    os.path.join('runs', 'agds_default_staged__paper_predict_20261024')))

os.environ.setdefault('PIPELINE_OUT_DIR', RUN)
os.environ.setdefault('PIPELINE_RUN_DIR', RUN)

S4_CSV = os.path.join(RUN, 'stage4_output.csv')
S5_CSV = os.path.join(RUN, 'stage5_output.csv')
S6_CSV = os.path.join(RUN, 'stage6_output.csv')
# Overridable so a smoke run can write to a scratch directory instead of the
# real batch folder; a stray checkpoint there would make --resume skip work.
BATCH_DIR = os.environ.get('STAGE5_BATCH_DIR') or os.path.join(
    RUN, 'stage5_batches_rerun')
PROGRESS = os.environ.get('STAGE5_PROGRESS') or os.path.join(
    RUN, '_stage5_predict_progress.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('s5_rerun')

# Person attributes the chain builder reads off the persons table.  Everything
# else it needs is already carried on the Stage 4 activity rows, so a chunk's
# own person block is enough — the full 4M-row population never has to be in
# memory at once.
PERSON_COLS = ('person_id', 'OTAZ', 'HOME_TAZ', 'home_taz', 'home_zone',
               'hhid', 'HHID', 'hh_id', 'HH_ID', 'household_id', 'vehrd',
               'WORK_TAZ', 'SCHOOL_TAZ')

# ── Narrow read contract ───────────────────────────────────────────────────
# Stage 4 emits a 713-column frame that is mostly the input household/person
# survey block carried through untouched.  The chain builder reads 35 of those
# columns; the rest cost parsing time, memory and CSV I/O for every chunk.
#
# The list is the union of:
#   * the identity/schedule columns _build_trip_chain emits per plan row,
#   * the person covariates _predict_batch puts on its tour table and hands to
#     the mode model (incp, indp, vehrd, nprd, sexp, age5p, WORKSTAT, STUDYING,
#     CARLICENCE/carlicence, bikes, occp_group),
#   * the M3 schedule-feasibility block Stage 4 writes (planning_order through
#     end_of_day),
#   * the fine-grained walk/bike skim keys (HOME_SA1, HOME_LVL4),
#   * PERSON_COLS, because the chunk's own person block is built from these.
# WORKSTAT is only read by Stage 6's mode module, not by Stage 5, but it travels
# on the plan rows as a carry-through attribute, so it has to be read here.
#
# Columns the builder tests with `if c in df.columns` and that Stage 4 does not
# emit (DTAZ, mode, tour_leg, MAINMODE, tour_id) stay absent in both the wide
# and the narrow read, so the code paths are identical either way.
S5_NARROW_COLS = (
    'person_id', 'household_id', 'purpose', 'ptour_id',
    'OTAZ', 'STARTIME', 'ENDTIME', 'predicted_duration_min',
    'WORK_TAZ', 'SCHOOL_TAZ',
    'incp', 'indp', 'vehrd', 'nprd', 'sexp', 'age5p', 'WORKSTAT',
    'STUDYING', 'CARLICENCE', 'carlicence', 'bikes', 'occp_group',
    'planning_order', 'activity_seq', 'schedule_seq',
    'schedule_n_activities', 'sched_prev_end', 'sched_next_start',
    'sched_prev_taz', 'sched_next_taz', 'start_of_day', 'end_of_day',
    'HOME_SA1', 'HOME_LVL4',
    'n_shopping_out', 'weight',
)


def person_chunks(path: str, chunk_size: int, usecols=None):
    """Read CSV chunks that never split a person across two yields.

    ``usecols`` restricts the parse to the columns the chain builder reads
    (``S5_NARROW_COLS``).  Dtypes are inferred per column, so a narrow read
    yields exactly the same values and dtypes as a full read for the columns it
    keeps — the difference is only in parse cost and footprint.
    """
    pending: pd.DataFrame | None = None
    # low_memory=False makes pandas hold the whole chunk's column dtypes at
    # once, which is what turned the first read of the 713-column Stage 4 frame
    # into a pandas "C error: out of memory".
    for raw in pd.read_csv(path, chunksize=chunk_size, usecols=usecols):
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


def chunk_persons(chunk: pd.DataFrame) -> pd.DataFrame:
    """One row per person from the chunk's own activity rows."""
    cols = [c for c in PERSON_COLS if c in chunk.columns]
    return chunk[cols].drop_duplicates('person_id').reset_index(drop=True)


def _batch_path(i: int, chunk: pd.DataFrame) -> str:
    pid = chunk['person_id'].to_numpy()
    return os.path.join(BATCH_DIR, f'batch_{i:05d}_{pid[0]}_{pid[-1]}.pkl')


def _write_progress(state: dict) -> None:
    tmp = PROGRESS + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(state, fh)
    os.replace(tmp, PROGRESS)


def _read_progress() -> dict:
    if os.path.isfile(PROGRESS):
        try:
            with open(PROGRESS) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {'chunks_done': 0, 'persons_done': 0, 'rows_written': 0}


def estimate():
    """Estimate Stage 5 on HHTS and save the bundle with cloudpickle."""
    import gc

    import pipeline_logger as pl

    _, out, landuse = pl.load_data()
    logger.info(f'HHTS: {len(out["persons"]):,} persons, '
                f'{len(out["tours"]):,} tours, {len(out["trips"]):,} trips')
    t0 = time.time()
    models, skims, _sk_lookup = pl.estimate_stage5(out, landuse)
    logger.info(f'Stage 5 estimated in {(time.time() - t0) / 60:.1f} min; '
                f'segments: {list(models)}')
    try:
        skims.close()
    except Exception:
        pass
    try:
        import jax
        jax.clear_caches()
        logger.info('released JAX compilation caches')
    except Exception as exc:  # noqa: BLE001
        logger.info(f'jax cache release skipped ({exc})')
    gc.collect()
    return models


def load_s5_models(path: str | None = None):
    """Read the run's Stage 5 bundle (cloudpickle, not stdlib pickle)."""
    from stage5_destination_choice import load_models

    path = path or os.path.join(RUN, 'stage5_models.pkl')
    if not os.path.isfile(path):
        raise SystemExit(f'no Stage 5 bundle at {path}; run with --estimate, '
                         'or pass --models <bundle> to use an existing one')
    models = load_models(path)
    logger.info(f'loaded Stage 5 models: {list(models)}')
    return models


def predict(input_csv: str, chunk_size: int, max_persons: int | None,
            resume: bool, models, landuse, usecols=None) -> None:
    from stage5_destination_choice import (
        _attach_skims, _build_trip_chain, _open_skims, _skim_flat, _skim_lookup,
    )
    import pipeline_logger as pl

    os.makedirs(BATCH_DIR, exist_ok=True)
    logger.info(f'Stage 5 models: {list(models)}')
    if usecols is None:
        logger.info('reading the full Stage 4 frame (all columns)')
    else:
        logger.info(f'narrow read: {len(usecols)} of the frame\'s columns')
        logger.info(f'  {sorted(usecols)}')

    skims, close_skims = _open_skims(pl.PIPELINE_SKIMS_PATH)
    try:
        skim_df = _skim_flat(skims)
        sk_lookup = _skim_lookup(skim_df)
        if len(skim_df) == 0:
            raise SystemExit('skim table is empty - cannot build chains')
        logger.info(f'skim table: {len(skim_df):,} OD pairs')

        state = _read_progress() if resume else {
            'chunks_done': 0, 'persons_done': 0, 'rows_written': 0}
        done = int(state.get('chunks_done', 0))
        logger.info(f'building trip chains from {input_csv} (resume from chunk '
                    f'{done}, {chunk_size:,} rows per chunk)')

        t0 = time.time()
        for i, chunk in enumerate(person_chunks(input_csv, chunk_size,
                                                usecols=usecols)):
            if i < done:
                continue
            if max_persons is not None and state['persons_done'] >= max_persons:
                break
            t_chunk = time.time()
            persons = chunk_persons(chunk)
            plan = _build_trip_chain(chunk, models, persons, landuse, skims,
                                     sk_lookup, seed=42)
            plan = _attach_skims(plan, skim_df)
            plan.to_pickle(_batch_path(i, chunk))
            state.update({
                'chunks_done': i + 1,
                'persons_done': int(state['persons_done']) + len(persons),
                'rows_written': int(state['rows_written']) + len(plan),
            })
            _write_progress(state)
            elapsed = time.time() - t0
            logger.info(f'chunk {i + 1}: {len(persons):,} persons -> '
                        f'{len(plan):,} plan rows in {time.time() - t_chunk:.0f}s'
                        f' | {state["persons_done"]:,} persons done, '
                        f'{state["persons_done"] / max(elapsed, 1e-6) * 3600:,.0f}'
                        f' persons/h')
            del plan, chunk, persons
            gc.collect()
    finally:
        # _open_skims returns (handle, opened_here) — the second value is a
        # flag, not a closer.
        if close_skims:
            skims.close()


def batch_files() -> list:
    files = sorted(os.listdir(BATCH_DIR)) if os.path.isdir(BATCH_DIR) else []
    return [os.path.join(BATCH_DIR, f) for f in files if f.endswith('.pkl')]


def assemble(resume: bool = False) -> None:
    """Stream the batch checkpoints into stage5_output.csv / stage6_output.csv.

    The pipeline writes the same plan to both names, but on this share the two
    passes cost over an hour of I/O, so the second name is linked (or copied
    when the filesystem will not link) from the first.
    """
    files = batch_files()
    if not files:
        raise SystemExit(f'no batch checkpoints in {BATCH_DIR}')
    logger.info(f'assembling {len(files)} batches')

    header: list | None = None
    mode = 'w'
    if resume and os.path.isfile(S5_CSV):
        mode = 'a'
        with open(S5_CSV, newline='') as fh:
            header = next(csv.reader(fh))
        logger.info(f'appending to {S5_CSV} ({len(header)} columns)')
    rows = 0
    for path in files:
        plan = pd.read_pickle(path)
        if header is None:
            header = list(plan.columns)
            pd.DataFrame(columns=header).to_csv(S5_CSV, index=False)
        missing = [c for c in header if c not in plan.columns]
        if missing:
            logger.warning(f'{os.path.basename(path)}: {len(missing)} '
                           f'columns missing: {missing[:6]}')
            for c in missing:
                plan[c] = np.nan
        plan.reindex(columns=header).to_csv(S5_CSV, mode='a', header=False,
                                           index=False)
        rows += len(plan)
        del plan
        gc.collect()
        if len(files) > 20 and (files.index(path) + 1) % 20 == 0:
            logger.info(f'  {files.index(path) + 1}/{len(files)} batches, '
                        f'{rows:,} rows written')
    logger.info(f'{S5_CSV}: {rows:,} rows from {len(files)} batches')

    import shutil
    for target in (S6_CSV,):
        try:
            if os.path.exists(target):
                os.remove(target)
            os.link(S5_CSV, target)
            logger.info(f'{target}: hard link to {os.path.basename(S5_CSV)}')
        except Exception as exc:  # noqa: BLE001
            logger.info(f'link failed ({exc}); copying {S5_CSV}')
            shutil.copyfile(S5_CSV, target)


def finalize(models, out, landuse, plan_out: str,
             skip_5b: bool = False) -> None:
    import openmatrix as omx
    import pipeline_logger as pl
    from stage5_destination_choice import _skim_flat, _skim_lookup

    logger.info(f'reading {S5_CSV}')
    scheduled_dest = pd.read_csv(S5_CSV, low_memory=False)
    logger.info(f'  {len(scheduled_dest):,} rows, '
                f'{scheduled_dest["person_id"].nunique():,} persons')

    if not skip_5b:
        try:
            scheduled_dest = pl.stage5b(out, scheduled_dest, landuse,
                                        g_tripdest=None)
            logger.info(f'Stage 5b: {len(scheduled_dest):,} rows')
        except Exception as exc:  # noqa: BLE001
            import traceback
            logger.warning(f'Stage 5b failed (non-fatal): {exc!r}\n'
                           f'{traceback.format_exc(limit=8)}')

    skims = omx.open_file(pl.PIPELINE_SKIMS_PATH, 'r')
    try:
        sk = _skim_lookup(_skim_flat(skims))
        t0 = time.time()
        plan, _ = pl.stage6(out, scheduled_dest, landuse, stage5_models=models,
                            skims=skims, skim_lookup=sk)
        logger.info(f'Stage 6 plan: {len(plan):,} rows in '
                    f'{(time.time() - t0) / 60:.1f} min')
    finally:
        try:
            skims.close()
        except Exception:
            pass

    plan.to_csv(plan_out, index=False)
    logger.info(f'plan -> {plan_out}')
    try:
        pl.plot(plan)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f'plan plots failed (non-fatal): {exc}')
    if 'row_type' in plan.columns:
        logger.info(f'  trips {int((plan["row_type"] == "trip").sum()):,} | '
                    f'activities {int((plan["row_type"] == "activity").sum()):,}')


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimate', action='store_true')
    ap.add_argument('--predict', action='store_true')
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--finalize', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--chunk', type=int, default=8000)
    ap.add_argument('--persons', type=int, default=None)
    ap.add_argument('--input', default=S4_CSV)
    ap.add_argument('--plan-out', default=os.path.join(RUN, 'plan.csv'))
    ap.add_argument('--skip-5b', action='store_true')
    ap.add_argument('--models', default=None,
                    help='Stage 5 bundle to load (default: RUN/stage5_models.pkl)')
    ap.add_argument('--wide', action='store_true',
                    help='parse every Stage 4 column instead of S5_NARROW_COLS')
    args = ap.parse_args(argv)

    if not os.path.isdir(RUN):
        print(f'run dir not found: {RUN}', file=sys.stderr)
        return 2

    models = None
    if args.estimate:
        models = estimate()
    if args.predict:
        if models is None:
            models = load_s5_models(args.models)
        import pipeline_logger as pl
        _, _out, landuse = pl.load_data()
        del _out
        predict(args.input, args.chunk, args.persons, args.resume, models,
                landuse, usecols=None if args.wide else S5_NARROW_COLS)
        del landuse
    if args.assemble:
        assemble(args.resume)
    if args.finalize:
        if models is None:
            models = load_s5_models(args.models)
        import pipeline_logger as pl
        _, out, landuse = pl.load_data()
        finalize(models, out, landuse, args.plan_out, args.skip_5b)
    return 0


if __name__ == '__main__':
    sys.exit(main())
