"""Estimate one isolated Stage 5 arm and write convergence artifacts."""

import argparse
import json
import os

import compat.numpy_shim  # noqa: F401


def _finite(value):
    try:
        import numpy as np
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _json_value(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if _finite(value) else None
    return value


def _larch_row(segment, model_type, result, model, ll):
    field_prefix = 'dest' if model_type == 'destination' else model_type
    diag = dict(result.get(f'{field_prefix}_fit_diagnostics', {}) or {})
    if not diag:
        diag = dict(getattr(model, '_fit_diagnostics', {}) or {})
    converged = diag.get('converged')
    if converged is None:
        converged = bool(result.get(f'{model_type}_coef')) and _finite(ll)
    return {
        'segment': segment,
        'model_type': model_type,
        'estimator': 'larch',
        'converged': bool(converged),
        'criterion': 'optimizer_success_and_finite_loglike',
        'loglike': float(ll) if _finite(ll) else None,
        'success': diag.get('success'),
        'status': diag.get('status'),
        'message': diag.get('message', ''),
        'nit': diag.get('nit'),
        'nfev': diag.get('nfev'),
        'n_params': diag.get('n_params', len(result.get(f'{field_prefix}_coef', {}) or {})),
        'n_free': diag.get('n_free'),
        'acceptance_rate': None,
        'ess_min': None,
        'stuck_params': len(diag.get('stuck_params', []) or []),
    }


def _adaptive_row(segment, result, min_ess):
    method = str(result.get('dest_estimator', '') or 'larch')
    metadata = dict(result.get('dest_estimator_metadata', {}) or {})
    diagnostics = dict(metadata.get('diagnostics', {}) or {})
    ll = result.get('dest_estimator_ll')
    acceptance = metadata.get('acceptance_rate')
    ess_min = diagnostics.get('ess_min')
    coef_ok = bool(result.get('dest_coef'))
    finite_ll = _finite(ll)

    if method in {'mh', 'gibbs'}:
        converged = (
            coef_ok and finite_ll and _finite(ess_min) and float(ess_min) >= min_ess
            and _finite(acceptance) and float(acceptance) > 0.0
        )
        criterion = f'finite_loglike_and_ess_min>={min_ess:g}_and_positive_acceptance'
    else:
        noise = diagnostics.get('gradient_noise', [])
        finite_noise = [value for value in noise if _finite(value)]
        converged = coef_ok and finite_ll and bool(finite_noise)
        criterion = 'finite_loglike_and_finite_gradient_noise'

    return {
        'segment': segment,
        'model_type': 'destination',
        'estimator': method,
        'converged': bool(converged),
        'criterion': criterion,
        'loglike': float(ll) if finite_ll else None,
        'success': None,
        'status': None,
        'message': '',
        'nit': diagnostics.get('iterations'),
        'nfev': None,
        'n_params': len(result.get('dest_coef', {}) or {}),
        'n_free': None,
        'acceptance_rate': float(acceptance) if _finite(acceptance) else None,
        'ess_min': float(ess_min) if _finite(ess_min) else None,
        'stuck_params': 0,
    }


def _missing_rows(segment):
    return [
        {'segment': segment, 'model_type': 'mode', 'estimator': 'missing',
         'converged': False, 'criterion': 'segment_result_present',
         'loglike': None, 'success': False, 'status': None,
         'message': 'missing segment', 'nit': None, 'nfev': None,
         'n_params': 0, 'n_free': None, 'acceptance_rate': None,
         'ess_min': None, 'stuck_params': 0},
        {'segment': segment, 'model_type': 'destination', 'estimator': 'missing',
         'converged': False, 'criterion': 'segment_result_present',
         'loglike': None, 'success': False, 'status': None,
         'message': 'missing segment', 'nit': None, 'nfev': None,
         'n_params': 0, 'n_free': None, 'acceptance_rate': None,
         'ess_min': None, 'stuck_params': 0},
    ]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--min-ess', type=float,
        default=float(os.environ.get('STAGE5_CONVERGENCE_MIN_ESS', '20')),
    )
    parser.add_argument(
        '--required-segments',
        default=os.environ.get('STAGE5_REQUIRED_SEGMENTS', 'work,edu,nonmand'),
    )
    return parser.parse_args()


def main() -> int:
    import pandas as pd
    import pipeline_logger as pl

    args = _parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    pl.OUT_DIR = out_dir
    # These attributes are harmless on older pipeline_logger revisions and
    # make the intended run directory explicit for revisions that use them.
    pl.PIPELINE_OUT_DIR = out_dir
    pl.PIPELINE_RUN_DIR = out_dir

    print(f"Stage 5 comparison arm: {os.environ.get('STAGE5_ARM', 'unknown')}")
    print(f"STAGE5_DEST_ESTIMATOR: {os.environ.get('STAGE5_DEST_ESTIMATOR', 'larch')}")
    print(f'Output directory: {out_dir}')

    _, out, landuse = pl.load_data()
    models, skims, _ = pl.estimate_stage5(out, landuse)
    if skims is not None:
        try:
            skims.close()
        except Exception:
            pass

    required = [value.strip() for value in args.required_segments.split(',') if value.strip()]
    rows = []
    for segment in required:
        result = models.get(segment)
        if not isinstance(result, dict):
            rows.extend(_missing_rows(segment))
            continue

        rows.append(_larch_row(
            segment, 'mode', result, result.get('mode_model'), result.get('mode_ll'),
        ))
        estimator = str(result.get('dest_estimator', 'larch') or 'larch')
        if estimator in {'mh', 'gibbs', 'sgd'} and result.get('dest_estimator_metadata'):
            rows.append(_adaptive_row(segment, result, args.min_ess))
        else:
            rows.append(_larch_row(
                segment, 'destination', result, result.get('dest_model'), result.get('dest_ll'),
            ))

    report = {
        'arm': os.environ.get('STAGE5_ARM', 'unknown'),
        'estimator': os.environ.get('STAGE5_DEST_ESTIMATOR', 'larch'),
        'output_dir': out_dir,
        'required_segments': required,
        'convergence_min_ess': args.min_ess,
        'rows': rows,
        'all_required_models_converged': all(bool(row['converged']) for row in rows),
        'artifacts': {
            'stage5_models': os.path.join(out_dir, 'stage5_models.pkl'),
            'coefficients_csv': os.path.join(out_dir, 'stage5_models_coefficients.csv'),
        },
    }
    report['artifacts_present'] = all(
        os.path.isfile(path) and os.path.getsize(path) > 0
        for path in report['artifacts'].values()
    )
    report['passed'] = bool(
        report['artifacts_present'] and report['all_required_models_converged']
    )

    report_path = os.path.join(out_dir, 'stage5_convergence_report.json')
    csv_path = os.path.join(out_dir, 'stage5_convergence_report.csv')
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(_json_value(report), handle, indent=2, sort_keys=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f'Convergence report: {report_path}')
    print(f'Convergence CSV: {csv_path}')
    for row in rows:
        status = 'CONVERGED' if row['converged'] else 'NOT_CONVERGED'
        print(
            f"  {row['segment']}/{row['model_type']}/{row['estimator']}: "
            f"{status} LL={row['loglike']} ESS={row['ess_min']}"
        )
    print(f"Final model bundle: {report['artifacts']['stage5_models']}")
    print(f"Comparison arm status: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
