"""
stage1_multivariate.py
======================
Wire MetaCountRegressor's MULTIVARIATE count model into Stage 1 (activity
frequency). Instead of estimating an independent univariate model per purpose,
this fits the 8 per-purpose activity counts (n_work_out … n_eat_out) JOINTLY
with a copula, so the correlation between activity types (e.g. people who do
more shopping also do more eat-out) is captured.  When HHTS joint tours are
derivable, a 9th count dimension — the per-person household JOINT-tour
participation count (``n_joint_tours``, "number of joint travel") — is added
and modelled jointly as well (gracefully dropped if the fit fails).

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
import math
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Copula / marginal are tunable via env so the qsub can sweep them.
MV_COPULA = os.environ.get('GA_MV_COPULA', 'gaussian')
#   'gaussian' | 'vine-frank' | 'vine-joe' | 'vine-gumbel' | 'vine-clayton'
#   joe/gumbel capture UPPER-tail dependence (people who do a lot of one
#   activity also do a lot of another); clayton captures LOWER-tail; gaussian
#   is symmetric/tail-free. Estimation uses the family; prediction reproduces
#   the fitted implied rank-correlation (see predict_joint_counts).
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


def _activity_and_covariates(df, extra_activity_cols=('n_joint_tours',)):
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
    # Optional extra activity-count dimension(s) — e.g. per-person household
    # JOINT-tour participation count (the "number of joint travel").  Only kept
    # when present, numeric and genuinely varying.
    for c in extra_activity_cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce')
            if v.notna().any() and v.nunique(dropna=True) > 1:
                activity_cols.append(c)
                covariate_cols.setdefault(c, [])
    return activity_cols, covariate_cols


def _add_joint_tour_count(df, tours: pd.DataFrame) -> pd.DataFrame:
    """Add a per-person household JOINT-tour participation count as an extra
    Stage-1 activity dimension (``n_joint_tours``) so the multivariate copula
    also models the "number of joint travel".

    Joint tours are derived from the observed HHTS ``tours`` via
    cdap_households_estimation.derive_observed_joint_tours (uses the joint
    flag when present, else the shared-trip heuristic).  Every participant of a
    joint tour gets +1; persons with no observed joint tour get 0.

    Never raises and returns ``df`` unchanged on any failure so the standard
    8-purpose MV fit is never lost to a joint-count problem.
    """
    try:
        from cdap_households_estimation import derive_observed_joint_tours
        from cdap_households import HouseholdCdapConfig
        jt = derive_observed_joint_tours(tours, HouseholdCdapConfig())
        if jt is None or not len(jt) or 'participants' not in jt.columns:
            return df
        cnt = {}
        parts = jt['participants']
        for p_list in parts:
            if isinstance(p_list, list):
                iter_parts = p_list
            else:
                iter_parts = [p.strip()
                              for p in str(p_list).replace('|', ',').split(',')]
            for p in iter_parts:
                try:
                    pid = int(str(p).strip())
                except (ValueError, TypeError):
                    continue
                cnt[pid] = cnt.get(pid, 0) + 1
        if not cnt:
            return df
        key = 'person_id' if 'person_id' in df.columns else df.columns[0]
        js = pd.DataFrame({
            'pid_s': [str(k) for k in cnt.keys()],
            'n_joint_tours': list(cnt.values()),
        })
        out = df.copy()
        out['pid_s'] = out[key].astype(str)
        out = out.merge(js, on='pid_s', how='left')
        out['n_joint_tours'] = out['n_joint_tours'].fillna(0).astype(int)
        out = out.drop(columns=['pid_s'])
        n_positive = int((out['n_joint_tours'] > 0).sum())
        logger.info(
            f"stage1_multivariate: n_joint_tours added "
            f"({n_positive:,} persons in ≥1 joint tour, "
            f"max={int(out['n_joint_tours'].max())})")
        return out
    except Exception as e:
        logger.warning(f"stage1_multivariate: n_joint_tours skipped ({e!r})")
        return df


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
            # Wald p-value from the reported t (2026-10-09: the CSV previously
            # shipped no p column, so audits counted 0/75 significant).
            try:
                p = float(math.erfc(abs(t) / math.sqrt(2.0))) if np.isfinite(t) else np.nan
            except (TypeError, ValueError):
                p = np.nan
            rows.append({'activity': act, 'term': nm, 'value': b, 'std_err': s,
                         't_stat': t, 'p_value': p, 'nb_alpha': alpha})
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
        df = _add_joint_tour_count(df, tours)
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
        try:
            fit = fit_fn(d, activity_cols, covariate_cols, copula=MV_COPULA,
                         marginal=MV_MARGINAL, add_intercept=True,
                         maxiter=MV_MAXITER, verbose=True)
        except Exception as _mv_fit_e:
            # Graceful degradation: if the extra joint-tour dimension makes the
            # copula fit fail, retry on the standard per-purpose counts only so
            # the multivariate report is still produced.
            if 'n_joint_tours' in activity_cols:
                logger.warning(
                    f"stage1_multivariate: fit with n_joint_tours failed "
                    f"({_mv_fit_e!r}); retrying without the joint dimension")
                activity_cols = [c for c in activity_cols if c != 'n_joint_tours']
                covariate_cols = {k: v for k, v in covariate_cols.items()
                                  if k != 'n_joint_tours'}
                used = activity_cols + sorted(
                    {c for cs in covariate_cols.values() for c in cs})
                d = df[used].apply(pd.to_numeric, errors='coerce').dropna()
                fit = fit_fn(d, activity_cols, covariate_cols, copula=MV_COPULA,
                             marginal=MV_MARGINAL, add_intercept=True,
                             maxiter=MV_MAXITER, verbose=True)
            else:
                raise
        _write_reports(fit, activity_cols, out_dir)
        try:
            print(fit.summary())
        except Exception:
            pass
        # Persist the fit + its covariate recipe so prediction can draw
        # JOINT (copula-correlated) counts per person. Never fatal.
        try:
            import pickle
            bundle = {'fit': fit, 'activity_cols': activity_cols,
                      'covariate_cols': covariate_cols,
                      'marginal': MV_MARGINAL, 'copula': MV_COPULA}
            with open(os.path.join(out_dir, 'stage1_multivariate_fit.pkl'),
                      'wb') as _fh:
                pickle.dump(bundle, _fh)
            logger.info("stage1_multivariate: saved fit bundle "
                        "stage1_multivariate_fit.pkl")
        except Exception as _pk_e:
            logger.warning(f"stage1_multivariate: bundle save skipped "
                           f"({_pk_e!r}) — prediction falls back to "
                           f"separate per-purpose models")
        return fit
    except Exception as e:
        logger.error(f"stage1_multivariate: fit failed ({e!r}); Stage 1 continues "
                     "with the standard estimator.")
        return None


def predict_joint_counts(synth, bundle=None, bundle_path=None, seed=42,
                         landuse=None):
    """Draw JOINT activity counts per person from a fitted MV copula model.

    Replaces the independent per-purpose draws with copula-correlated draws:
    marginal NB/Poisson means come from the person's covariates, dependence
    between activity types from the fitted copula correlation, so simulated
    persons who shop a lot also tend to eat out (etc.) exactly as estimated.

    Parameters
    ----------
    synth       : synthetic-person DataFrame (same attributes as estimation)
    bundle      : dict from stage1_multivariate_fit.pkl (fit, activity_cols,
                  covariate_cols, marginal, copula); or pass bundle_path
    bundle_path : path to the pickle (used when bundle is None)
    seed        : RNG seed for the copula draws
    landuse     : enriched landuse for *_acc features (optional; missing
                  features are filled with 0 + logged, as in estimation)

    Returns
    -------
    DataFrame with n_{purpose}_out int columns (MAX_COUNT-capped), indexed
    like synth. Raises on any problem (caller falls back to separate models).
    """
    import pickle
    from scipy.stats import norm, nbinom, poisson
    from stage1_activity_frequency import (
        PURPOSES, MAX_COUNT, assign_ptype_column,
        _add_categorical_person_features,
    )
    if bundle is None:
        with open(bundle_path, 'rb') as _fh:
            bundle = pickle.load(_fh)
    fit = bundle['fit']
    activity_cols = list(bundle['activity_cols'])
    covariate_cols = dict(bundle['covariate_cols'])
    marginal = bundle.get('marginal', 'nb')
    copula = bundle.get('copula', 'gaussian')
    rng = np.random.default_rng(seed)

    df = synth.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})
    if 'ptype' not in df.columns:
        df = assign_ptype_column(df)
    df = _add_categorical_person_features(df)
    if landuse is not None:
        try:
            from synth_attributes import attach_person_accessibility
            df = attach_person_accessibility(df, landuse)
        except Exception as _acc_e:
            logger.warning(f"predict_joint_counts: accessibility attach "
                           f"skipped ({_acc_e!r})")

    # feature matrices in the saved per-activity order (const -> ones)
    X_list, missing = [], set()
    feat_names = list(getattr(fit, 'feature_names', []))
    for m, act in enumerate(activity_cols):
        names = list(feat_names[m]) if m < len(feat_names) else []
        cols = [c for c in names if c not in ('const', 'Intercept')]
        Xm = np.zeros((len(df), len(names)))
        for j, nm in enumerate(names):
            if nm in ('const', 'Intercept'):
                Xm[:, j] = 1.0
            elif nm in df.columns:
                Xm[:, j] = pd.to_numeric(df[nm],
                                         errors='coerce').fillna(0).values
            else:
                missing.add(nm)
        X_list.append(Xm)
    if missing:
        logger.warning(f"predict_joint_counts: {len(missing)} fit features "
                       f"absent in synth, filled 0: {sorted(missing)[:8]}")
    means = np.asarray(fit.predict(X_list), dtype=float)
    means = np.where(np.isfinite(means), np.maximum(means, 0.0), 0.0)

    # Joint uniforms. Gaussian and the vine families (frank/joe/gumbel/clayton)
    # all expose a fitted implied rank-correlation matrix (fit.correlation);
    # sample correlated uniforms from it via a Gaussian sampler. For the vine
    # copulas this reproduces the estimated PAIRWISE rank dependence (the family
    # governs the fit / model selection and the copula-parameter -> Kendall tau
    # mapping); it does not reproduce Archimedean tail-dependence asymmetry in
    # the draws. Only fall back to independent draws if no usable matrix exists.
    M = len(activity_cols)
    R = np.asarray(getattr(fit, 'correlation', np.eye(M)), dtype=float)
    if R.shape == (M, M) and np.all(np.isfinite(R)):
        R = (R + R.T) / 2.0 + 1e-8 * np.eye(M)
        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            # project to the nearest positive-definite correlation matrix
            w, V = np.linalg.eigh(R)
            R = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
            d = np.sqrt(np.clip(np.diag(R), 1e-12, None))
            R = R / np.outer(d, d)
            L = np.linalg.cholesky(R + 1e-8 * np.eye(M))
        u = norm.cdf(rng.standard_normal((len(df), M)) @ L.T)
        if str(copula).lower() != 'gaussian':
            logger.info(f"predict_joint_counts: {copula} sampled via fitted "
                        f"implied rank-correlation (Gaussian sampler)")
    else:
        logger.warning(f"predict_joint_counts: copula={copula!r} has no usable "
                       f"correlation matrix — independent draws around MV means")
        u = rng.uniform(size=(len(df), M))
    u = np.clip(u, 1e-12, 1 - 1e-12)

    out = pd.DataFrame(index=df.index)
    alphas = list(getattr(fit, 'alphas', [np.nan] * M))
    for m, act in enumerate(activity_cols):
        mu = means[:, m]
        if str(marginal).lower() == 'poisson':
            y = poisson.ppf(u[:, m], mu)
        else:
            a = float(alphas[m]) if m < len(alphas) else np.nan
            if not np.isfinite(a) or a <= 0:
                a = 1.0
            y = nbinom.ppf(u[:, m], n=1.0 / a, p=1.0 / (1.0 + a * mu))
        y = np.nan_to_num(y, nan=0.0, posinf=10.0).astype(int)
        out[act] = np.maximum(y, 0)
    # cap like the estimation counts + keep only purpose columns
    for p in PURPOSES:
        col = f'n_{p}_out'
        if col in out.columns:
            out[col] = out[col].clip(
                upper=int(MAX_COUNT.get(p, 9))).astype(int)
        else:
            out[col] = 0
    logger.info(f"predict_joint_counts: joint draws for {len(df):,} persons "
                f"({M} dims, copula={copula}, marginal={marginal})")
    return out[[f'n_{p}_out' for p in PURPOSES]]


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
