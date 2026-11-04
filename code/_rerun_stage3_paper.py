"""Re-run Stage 3 (estimation + prediction) for the paper run tag.

Estimation refits the HHTS duration models with the run's own GA-selected
feature spec (``selected_features_summary.json`` -> ``stage3.features``) and
writes the model bundle and coefficient tables back into the run directory.

Prediction streams the run's ``stage2_output.csv`` in person chunks (the full
population is ~4M persons, and the whole-frame pass needs a 194 GB node), so
it runs in bounded memory and can be resumed: completed chunks are recorded in
``_stage3_predict_progress.json`` and the output accumulates in
``stage3_output.csv.new`` until the final chunk renames it over the old file.

Usage
-----
    python _rerun_stage3_paper.py --estimate
    python _rerun_stage3_paper.py --predict [--chunk 20000] [--persons 20000]
    python _rerun_stage3_paper.py --predict --resume
    python _rerun_stage3_paper.py --header-check          # schema vs old file

Env: STAGE3_RERUN_RUN overrides the run directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time

import pandas as pd

compat = None
try:                                       # NumPy 2.0 shim, same as the pipeline
    import compat.numpy_shim  # noqa: F401
except Exception:
    pass

RUN = os.path.abspath(os.environ.get(
    'STAGE3_RERUN_RUN',
    os.path.join('runs', 'agds_default_staged__paper_predict_20261024')))
S2_CSV = os.path.join(RUN, 'stage2_output.csv')
S3_CSV = os.path.join(RUN, 'stage3_output.csv')
S3_NEW = S3_CSV + '.new'
S3_BAK = S3_CSV + '.bak_pre_s3fix'
PROGRESS = os.path.join(RUN, '_stage3_predict_progress.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('s3_rerun')


def load_selected_features() -> dict:
    path = os.path.join(RUN, 'selected_features_summary.json')
    with open(path) as fh:
        return json.load(fh)['stage3']['features']


def estimate(duration_model: str | None = None) -> dict:
    from data_connect import Synth_Connect
    from stage3_activity_duration import estimate_duration_models, save_models

    feats = load_selected_features()
    logger.info(f'estimating Stage 3 on HHTS with the run\'s GA spec '
                f'({sum(len(v or []) for v in feats.values())} feature terms '
                f'across {len(feats)} purposes)')
    out = Synth_Connect().process_survey(save=False)
    logger.info(f"  HHTS: {len(out['persons']):,} persons, "
                f"{len(out['tours']):,} tours, {len(out['trips']):,} trips")

    models = estimate_duration_models(
        out['tours'], out['persons'], trips=out.get('trips'),
        feature_override=feats, duration_model=duration_model,
    )
    save_models(models, os.path.join(RUN, 'stage3_models.pkl'))
    models_dir = os.path.join(RUN, 'models')
    if os.path.isdir(models_dir):
        save_models(models, os.path.join(models_dir, 'stage3_models.pkl'))
    try:
        from ga_hypothesis_testing import _ga_save_stage_coef_csv
        _ga_save_stage_coef_csv(models, 'stage3', RUN)
        logger.info('  wrote stage3_coefficients.csv/.html')
    except Exception as exc:
        logger.warning(f'  coefficient export failed: {exc}')

    rows = []
    for purpose, entry in models.items():
        if not isinstance(entry, dict):
            continue
        rows.append({
            'purpose': purpose, 'n_obs': entry.get('n_obs'),
            'sigma': round(float(entry.get('sigma') or 0), 3),
            'obs_mean': round(float(entry.get('mean_obs') or 0), 1),
            'pred_mean': round(float(entry.get('mean_pred_uncal') or 0), 1),
            'calib': round(float(entry.get('calibration_factor') or 0), 3),
            'n_feats': len(entry.get('features') or []),
            'features': ', '.join(entry.get('features') or []),
        })
    logger.info('\n' + pd.DataFrame(rows).to_string(index=False))
    return models


def old_header() -> list:
    with open(S3_CSV, newline='') as fh:
        return next(csv.reader(fh))


def header_check(chunk_size: int = 20000) -> bool:
    """Predict the first chunk and compare the produced schema to the old file."""
    import pickle
    from stage3_activity_duration import predict_activity_durations

    with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as fh:
        models = pickle.load(fh)
    chunk = next(pd.read_csv(S2_CSV, chunksize=chunk_size))
    acts = predict_activity_durations(chunk, models, seed=42)
    old_cols = old_header()
    raw_missing = [c for c in old_cols if c not in acts.columns]
    acts = reconcile_columns(acts, old_cols)
    ok = list(acts.columns) == old_cols
    print(f'old {len(old_cols)} cols, predicted {len(acts.columns)} cols, '
          f'schema identical: {ok}')
    print('rebuilt from the recorded schema:', raw_missing[:20])
    print('imputed features:',
          {k: v for k, v in (acts.attrs.get('imputed_features') or {}).items()
           if v})
    return ok


def reconcile_columns(acts: pd.DataFrame, needed: list) -> pd.DataFrame:
    """Force a predicted chunk onto the run's recorded Stage 3 schema.

    The stage-3 taste interactions (``ix3_<segment>_x_<continuous>``) are only
    materialised when that segment varies in the frame at hand and the
    round-robin cap admits the pair, so a 20k-person chunk can produce a
    slightly different set from the whole-population pass.  Appending CSVs
    with differing columns would misalign the file, so the missing products
    are rebuilt here from their own names (``seg * cont``, exactly what the
    single pass computed) and the frame is reordered onto ``needed``.
    """
    missing = [c for c in needed if c not in acts.columns]
    for name in missing:
        if not name.startswith('ix3_'):
            continue
        body = name[len('ix3_'):]
        if '_x_' not in body:
            continue
        seg, cont = body.rsplit('_x_', 1)
        if seg in acts.columns and cont in acts.columns:
            acts[name] = (pd.to_numeric(acts[seg], errors='coerce').fillna(0.0)
                          * pd.to_numeric(acts[cont], errors='coerce').fillna(0.0))
    still_missing = [c for c in needed if c not in acts.columns]
    if still_missing:
        logger.warning(f'{len(still_missing)} expected columns could not be '
                       f'rebuilt: {still_missing[:8]}')
        for c in still_missing:
            acts[c] = float('nan')
    extra = [c for c in acts.columns if c not in needed]
    if extra:
        logger.info(f'dropping {len(extra)} columns not in the recorded schema: '
                    f'{extra[:8]}')
    return acts.reindex(columns=list(needed))


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
    import pickle
    from stage3_activity_duration import predict_activity_durations

    with open(os.path.join(RUN, 'stage3_models.pkl'), 'rb') as fh:
        models = pickle.load(fh)

    state = _read_progress() if resume else {
        'chunks_done': 0, 'persons_done': 0, 'rows_written': 0}
    done = int(state.get('chunks_done', 0))
    mode = 'a' if resume and done and os.path.isfile(S3_NEW) else 'w'
    SCHEMA = old_header()
    logger.info(f'predicting Stage 3 into {S3_NEW} (mode={mode}, '
                f'resume from chunk {done}, chunk={chunk_size:,} persons, '
                f'{len(SCHEMA)} columns)')

    t0 = time.time()
    persons = 0
    for i, chunk in enumerate(pd.read_csv(S2_CSV, chunksize=chunk_size)):
        if i < done:
            continue
        if max_persons is not None and persons >= max_persons:
            break
        t_chunk = time.time()
        acts = predict_activity_durations(chunk, models, sample=True,
                                          seed=42 + i)
        acts = reconcile_columns(acts, SCHEMA)
        acts.to_csv(S3_NEW, mode=mode, header=(i == 0), index=False)
        mode = 'a'
        persons += len(chunk)
        state.update({
            'chunks_done': i + 1,
            'persons_done': int(state.get('persons_done', 0)) + len(chunk),
            'rows_written': int(state.get('rows_written', 0)) + len(acts),
        })
        _write_progress(state)
        elapsed = time.time() - t0
        rate = persons / max(elapsed, 1e-6)
        logger.info(f'chunk {i + 1}: {len(chunk):,} persons -> {len(acts):,} '
                    f'activities in {time.time() - t_chunk:.0f}s '
                    f'| {state["persons_done"]:,} persons done, '
                    f'{rate * 3600:,.0f} persons/h')
        del acts, chunk

    done_all = max_persons is None
    if done_all:
        # keep the pre-fix output: it is the only copy of the old durations
        if os.path.isfile(S3_CSV):
            if os.path.isfile(S3_BAK):
                os.remove(S3_BAK)
            os.replace(S3_CSV, S3_BAK)
        os.replace(S3_NEW, S3_CSV)
        try:
            os.remove(PROGRESS)
        except OSError:
            pass
        logger.info(f'done: stage3_output.csv replaced '
                    f'({state["rows_written"]:,} activity rows)')
    else:
        logger.info(f'smoke run stopped after {state["persons_done"]:,} persons; '
                    f'{S3_NEW} kept, progress recorded')


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimate', action='store_true')
    ap.add_argument('--predict', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--header-check', action='store_true')
    ap.add_argument('--chunk', type=int, default=20000)
    ap.add_argument('--persons', type=int, default=None)
    ap.add_argument('--duration-model', default=None)
    args = ap.parse_args(argv)

    if not os.path.isdir(RUN):
        print(f'run dir not found: {RUN}', file=sys.stderr)
        return 2
    if args.estimate:
        estimate(args.duration_model)
    if args.header_check:
        return 0 if header_check(args.chunk) else 1
    if args.predict:
        predict(args.chunk, args.persons, args.resume)
    return 0


if __name__ == '__main__':
    sys.exit(main())
