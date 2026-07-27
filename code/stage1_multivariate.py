"""
stage1_multivariate.py
======================
Wire MetaCountRegressor's MULTIVARIATE count model into Stage 1 (activity
frequency). Instead of estimating an independent univariate model per purpose,
this fits the 8 per-purpose activity counts (n_work_out … n_eat_out) JOINTLY
with a copula, so the correlation between activity types (e.g. people who do
more shopping also do more eat-out) is captured.

Uses metacountregressor.multivariate_count_regressor.fit_multivariate_activity_model
(the DataFrame convenience wrapper: per-activity NB/Poisson marginals + a
gaussian / vine-frank copula). Requires a metacountregressor build that ships
the multivariate module (the one in C:/Users/ahernz/source/MetaCount) — install
it into the run env if the import below fails.

Invoked from ga_stage1 when GA_ESTIMATOR=metacount_mv; also runnable standalone
(job_stage1_multivariate.pbs). Writes:
    <out_dir>/stage1_multivariate_coefficients.csv
    <out_dir>/stage1_multivariate_report.html
    <out_dir>/stage1_multivariate_correlation.csv
and returns the MultivariateCountFit (or None on failure — never raises).
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Copula / marginal are tunable via env so the qsub can sweep them.
MV_COPULA = os.environ.get('GA_MV_COPULA', 'gaussian')      # 'gaussian' | 'vine-frank'
MV_MARGINAL = os.environ.get('GA_MV_MARGINAL', 'nb')        # 'nb' | 'poisson'
MV_MAXITER = int(os.environ.get('GA_MV_MAXITER', '1000'))


def _import_mv():
    """Import the multivariate activity model, adding the local source tree to
    sys.path as a fallback if the installed package predates the module."""
    try:
        from metacountregressor.multivariate_count_regressor import (
            fit_multivariate_activity_model)
        return fit_multivariate_activity_model
    except Exception as e:
        import sys
        for _cand in (os.environ.get('METACOUNT_SRC', ''),
                      r'C:\Users\ahernz\source\MetaCount',
                      os.path.expanduser('~/source/MetaCount'),
                      os.path.expanduser('~/MetaCount')):
            if _cand and os.path.isdir(_cand) and _cand not in sys.path:
                sys.path.insert(0, _cand)
        try:
            from metacountregressor.multivariate_count_regressor import (
                fit_multivariate_activity_model)
            return fit_multivariate_activity_model
        except Exception as e2:
            logger.error(
                "stage1_multivariate: cannot import fit_multivariate_activity_model "
                f"({e2!r}). Install the metacountregressor build that ships "
                "multivariate_count_regressor into the run env "
                "(pip install -e C:/Users/ahernz/source/MetaCount) or set "
                "METACOUNT_SRC to its path.")
            return None


def _activity_and_covariates(df):
    """(activity_cols, covariate_cols dict) present in df, from the Stage-1
    per-purpose count columns and DEFAULT_FEATURES_BY_PURPOSE."""
    from stage1_activity_frequency import PURPOSES, DEFAULT_FEATURES_BY_PURPOSE
    activity_cols, covariate_cols = [], {}
    for p in PURPOSES:
        col = f'n_{p}_out'
        if col not in df.columns:
            continue
        activity_cols.append(col)
        feats = DEFAULT_FEATURES_BY_PURPOSE.get(
            p, DEFAULT_FEATURES_BY_PURPOSE.get('nonmand', []))
        # keep present, numeric, non-constant covariates only
        keep = []
        for c in dict.fromkeys(feats):
            if c in df.columns:
                v = pd.to_numeric(df[c], errors='coerce')
                if v.notna().any() and v.nunique(dropna=True) > 1:
                    keep.append(c)
        covariate_cols[col] = keep
    return activity_cols, covariate_cols


def _write_reports(fit, activity_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for m, act in enumerate(fit.activity_names):
        beta = np.asarray(fit.coef[m]).ravel()
        se = (np.asarray(fit.se[m]).ravel() if getattr(fit, 'se', None)
              and len(fit.se) > m else np.full(len(beta), np.nan))
        names = fit.feature_names[m] if m < len(fit.feature_names) else \
            [f'b{i}' for i in range(len(beta))]
        alpha = fit.alphas[m] if getattr(fit, 'alphas', None) is not None and m < len(fit.alphas) else np.nan
        for i, nm in enumerate(names):
            b = float(beta[i]) if i < len(beta) else np.nan
            s = float(se[i]) if i < len(se) else np.nan
            t = (b / s) if (np.isfinite(b) and np.isfinite(s) and s > 0) else np.nan
            rows.append({'activity': act, 'term': nm, 'value': b, 'std_err': s,
                         't_stat': t, 'nb_alpha': alpha})
    cdf = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'stage1_multivariate_coefficients.csv')
    cdf.to_csv(csv_path, index=False)

    corr = np.asarray(getattr(fit, 'correlation', np.eye(len(activity_cols))))
    corr_df = pd.DataFrame(corr, index=fit.activity_names, columns=fit.activity_names)
    corr_path = os.path.join(out_dir, 'stage1_multivariate_correlation.csv')
    corr_df.to_csv(corr_path)

    html = ["<meta charset='utf-8'><h1>Stage 1 — Multivariate activity-count model</h1>",
            f"<p>{MV_MARGINAL.upper()} marginals, {MV_COPULA} copula. "
            f"LL={fit.loglik:.2f}  AIC={fit.aic:.1f}  BIC={fit.bic:.1f}  "
            f"converged={fit.converged}</p>",
            "<h2>Activity-type correlation (copula)</h2>",
            corr_df.round(3).to_html(),
            "<h2>Coefficients</h2>",
            cdf.round(4).to_html(index=False)]
    html_path = os.path.join(out_dir, 'stage1_multivariate_report.html')
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(html))
    logger.info(f"stage1_multivariate: wrote {csv_path}, {corr_path}, {html_path}")


def fit_stage1_multivariate(tours, persons, landuse=None, out_dir='saved_data'):
    """Fit the joint multivariate activity-count model on Stage-1 data and write
    reports. Returns the fit object, or None on any failure (never raises, so it
    can't break the pipeline stage)."""
    try:
        fit_fn = _import_mv()
        if fit_fn is None:
            return None
        from stage1_activity_frequency import prepare_frequency_data
        df = prepare_frequency_data(tours, persons, landuse=landuse)
        activity_cols, covariate_cols = _activity_and_covariates(df)
        if len(activity_cols) < 2:
            logger.warning("stage1_multivariate: <2 activity count columns present "
                           f"({activity_cols}); nothing multivariate to fit.")
            return None
        # Drop rows with any NaN in the used columns.
        used = activity_cols + sorted({c for cs in covariate_cols.values() for c in cs})
        d = df[used].apply(pd.to_numeric, errors='coerce').dropna()
        logger.info(f"stage1_multivariate: fitting {len(activity_cols)} activities "
                    f"jointly on {len(d)} persons ({MV_MARGINAL}/{MV_COPULA}) ...")
        fit = fit_fn(d, activity_cols, covariate_cols, copula=MV_COPULA,
                     marginal=MV_MARGINAL, add_intercept=True,
                     maxiter=MV_MAXITER, verbose=True)
        _write_reports(fit, activity_cols, out_dir)
        try:
            print(fit.summary())
        except Exception:
            pass
        return fit
    except Exception as e:
        logger.error(f"stage1_multivariate: fit failed ({e!r}); Stage 1 continues "
                     "with the standard estimator.")
        return None


if __name__ == '__main__':
    # Standalone: load Stage-1 inputs via the pipeline and fit the MV model.
    import argparse
    ap = argparse.ArgumentParser(description="Fit the Stage-1 multivariate activity model")
    ap.add_argument('--out-dir', default='saved_data')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    import pipeline_logger as pl
    sc, out, landuse = pl.load_data()
    fit = fit_stage1_multivariate(out['tours'], out['persons'], landuse=landuse,
                                  out_dir=args.out_dir)
    print("OK" if fit is not None else "FAILED (see log)")
