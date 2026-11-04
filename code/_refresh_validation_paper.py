"""Refresh the HHTS re-prediction validation panels for a run tag.

Mirrors ``validate_hhts.run_validation``'s sequence, but persists the metrics
JSON after every stage so a slow or failing later stage cannot lose the panels
that already exist, and prints per-stage wall time.

Usage:
    python _refresh_validation_paper.py [run_dir]
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
import time

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s  %(message)s')
for _noisy in ('matplotlib', 'PIL', 'searchlibrium'):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from data_connect import Synth_Connect                                  # noqa: E402
from validate_hhts import (                                             # noqa: E402
    validate_stage1, validate_stage2, validate_stage3, validate_stage4,
    validate_stage5, validate_stage5_holdout, validate_profiles,
    write_metrics_json, plot_fit_summary, write_html_report,
)

RUN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    'runs', 'agds_default_staged__paper_predict_20261024'))
STAGES = ('stage1', 'stage2', 'stage3', 'stage4', 'stage5')
SKIP_STAGE5 = os.environ.get('VALIDATION_SKIP_STAGE5', '0') == '1'


def load(name: str) -> dict:
    path = os.path.join(RUN, f'{name}_models.pkl')
    if not os.path.isfile(path):
        print(f'  [validation] missing {path}', flush=True)
        return {}
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def main() -> int:
    val_dir = os.path.join(RUN, 'plots', 'validation')
    os.makedirs(val_dir, exist_ok=True)
    print(f'run: {RUN}\nplots: {val_dir}', flush=True)

    out = Synth_Connect().process_survey(save=False)
    print('survey frames:', {k: (v.shape if hasattr(v, 'shape') else type(v))
                             for k, v in out.items()}, flush=True)
    models = {k: load(k) for k in STAGES}

    results = []
    plan = [('Stage 1', validate_stage1, 1), ('Stage 2', validate_stage2, 2),
            ('Stage 3', validate_stage3, 3), ('Stage 4', validate_stage4, 4)]
    for label, fn, idx in plan:
        t = time.time()
        try:
            res = fn(out, models[f'stage{idx}'], val_dir)
            results.append(res)
            print(f'  {label}: {time.time() - t:.0f}s -> '
                  f'{len(res.get("plots") or [])} artifacts', flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f'  {label} validation failed: {exc}', flush=True)
        write_metrics_json(results, RUN)

    s5_result = None
    if not SKIP_STAGE5:
        t = time.time()
        try:
            s5_result = validate_stage5(out, models['stage5'], val_dir)
            results.append(s5_result)
            print(f'  Stage 5: {time.time() - t:.0f}s', flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f'  Stage 5 validation failed: {exc}', flush=True)
        write_metrics_json(results, RUN)
        t = time.time()
        try:
            results.append(validate_stage5_holdout(out, models['stage5'], val_dir))
            print(f'  Stage 5 hold-out: {time.time() - t:.0f}s', flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f'  Stage 5 hold-out failed: {exc}', flush=True)

    t = time.time()
    try:
        results.append(validate_profiles(
            out, models['stage4'],
            dist_rows=(s5_result or {}).get('_dist_rows'), save_dir=val_dir))
        print(f'  Profiles: {time.time() - t:.0f}s', flush=True)
    except Exception as exc:                                       # noqa: BLE001
        print(f'  Profile validation failed: {exc}', flush=True)

    try:
        summary = plot_fit_summary(results, val_dir)
    except Exception as exc:                                       # noqa: BLE001
        print(f'  summary plot failed: {exc}', flush=True)
        summary = ''
    try:
        write_html_report(results, summary, val_dir, RUN)
    except Exception as exc:                                       # noqa: BLE001
        print(f'  HTML report failed: {exc}', flush=True)
    path = write_metrics_json(results, RUN)
    print(f'done: metrics -> {path}', flush=True)

    try:
        from build_validation_dashboard import build_validation_dashboard
        build_validation_dashboard(RUN, out_dir=os.path.join(RUN, 'summarize'))
        print('dashboard refreshed', flush=True)
    except Exception as exc:                                       # noqa: BLE001
        print(f'dashboard skipped: {exc}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
