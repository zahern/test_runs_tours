'''Alternative fitness evaluators for stages 1–4 using MetaCount and SearchLibrium.

Install (editable, so local edits take effect immediately):
    pip install -e C:/Users/ahernz/source/MetaCount
    pip install -e C:/Users/ahernz/source/SearchLibrium
    pip install jax jaxlib jaxopt          # required MetaCount backend

Or from PyPI:
    pip install metacountregressor SearchLibrium

Controlled by the GA_ESTIMATOR environment variable:
    GA_ESTIMATOR=default        (existing ordered-logit / OLS / MNL — unchanged)
    GA_ESTIMATOR=metacount      (MetaCount NB2 for stage1, mixed lognormal for 2-3)
    GA_ESTIMATOR=searchlibrium  (SearchLibrium MNL for stage4, same fallback for 1-3)
    GA_ESTIMATOR=both           (MetaCount for 1-3, SearchLibrium for 4)

Fitness functions all return  –BIC / n  (higher = better, same convention as
the rest of ga_hypothesis_testing.py).  The GA is unchanged; only the model
fitted at each evaluation differs.'''
"""
"""'''
ga_external_estimators.py
Alternative fitness evaluators for stages 1–4 using MetaCount and SearchLibrium.

Install (editable, so local edits take effect immediately):

  On the HPC cluster (from the conda env):
    pip install -e /mnt/hpccs01/home/ahernz/source/MetaCount
    pip install -e /mnt/hpccs01/home/ahernz/source/SearchLibrium
    pip install jax jaxlib jaxopt          # required MetaCount backend

  Local Windows dev machine:
    pip install -e C:/Users/ahernz/source/MetaCount
    pip install -e C:/Users/ahernz/source/SearchLibrium
    pip install jax jaxlib jaxopt

Or from PyPI:
    pip install metacountregressor SearchLibrium --upgrade
''''''
Controlled by the GA_ESTIMATOR environment variable:
    GA_ESTIMATOR=default        (existing ordered-logit / OLS / MNL — unchanged)
    GA_ESTIMATOR=metacount      (MetaCount NB2 for stage1, mixed lognormal for 2-3)
    GA_ESTIMATOR=searchlibrium  (SearchLibrium MNL for stage4, same fallback for 1-3)
    GA_ESTIMATOR=both           (MetaCount for 1-3, SearchLibrium for 4)
''''''
Fitness functions all return  –BIC / n  (higher = better, same convention as
the rest of ga_hypothesis_testing.py).  The GA is unchanged; only the model
fitted at each evaluation differs.
"""
=========================
Alternative fitness evaluators for stages 1–4 using MetaCount and SearchLibrium.

Install (editable, so local edits take effect immediately):
    pip install -e C:/Users/ahernz/source/MetaCount
    pip install -e C:/Users/ahernz/source/SearchLibrium
    pip install jax jaxlib jaxopt          # required MetaCount backend

Or from PyPI:
    pip install metacountregressor SearchLibrium

Controlled by the GA_ESTIMATOR environment variable:
    GA_ESTIMATOR=default        (existing ordered-logit / OLS / MNL — unchanged)
    GA_ESTIMATOR=metacount      (MetaCount NB2 for stage1, mixed lognormal for 2-3)
    GA_ESTIMATOR=searchlibrium  (SearchLibrium MNL for stage4, same fallback for 1-3)
    GA_ESTIMATOR=both           (MetaCount for 1-3, SearchLibrium for 4)

Fitness functions all return  –BIC / n  (higher = better, same convention as
the rest of ga_hypothesis_testing.py).  The GA is unchanged; only the model
fitted at each evaluation differs.
"""
'''
from __future__ import annotations

import os
import logging
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability checks (graceful import — missing package → fall back to default)
# ---------------------------------------------------------------------------

try:
    from metacountregressor import ExperimentBuilder, ModelConstraints
    _METACOUNT_OK = True
except ImportError:
    _METACOUNT_OK = False
    logger.debug("metacountregressor not installed — MetaCount estimators unavailable")

try:
    from SearchLibrium import MultinomialLogit, MixedLogit
    _SEARCHLIB_OK = True
except ImportError:
    _SEARCHLIB_OK = False
    logger.debug("SearchLibrium not installed — SearchLibrium estimators unavailable")


def ga_estimator() -> str:
    """Read GA_ESTIMATOR env var; default to 'default'."""
    v = os.environ.get('GA_ESTIMATOR', 'default').strip().lower()
    if v not in ('default', 'metacount', 'searchlibrium', 'both'):
        logger.warning(f"Unknown GA_ESTIMATOR='{v}'; using 'default'")
        return 'default'
    return v


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bic_per_obs(log_lik: float, k: int, n: int) -> float:
    """–BIC/n  (higher = better, matches ga_hypothesis_testing convention)."""
    bic = -2.0 * log_lik + k * np.log(max(n, 2))
    return -bic / max(n, 1)


def _ensure_id(df: pd.DataFrame, id_col: str = 'obs_id') -> pd.DataFrame:
    """Add a sequential integer id column if not present."""
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))
    return df


# ---------------------------------------------------------------------------
# MetaCount helpers
# ---------------------------------------------------------------------------

def _metacount_fit(
    df: pd.DataFrame,
    id_col: str,
    y_col: str,
    selected_vars: List[str],
    family: str = 'nb2',
    offset_col: Optional[str] = None,
) -> float:
    """
    Fit a single MetaCount model (no search) and return –BIC/n.

    Parameters
    ----------
    df           : estimation data (one row per observation)
    id_col       : unique ID column name
    y_col        : response column (count or continuous)
    selected_vars: variables selected by the GA chromosome
    family       : 'nb2' | 'poisson' | 'lognormal' | 'gaussian'
    offset_col   : log-offset column (count models with exposure)

    Returns –BIC/n, or –1e9 on failure.
    """
    if not _METACOUNT_OK:
        return -1e9
    if not selected_vars:
        return -1e9

    try:
        spec = {
            'fixed_terms':      selected_vars,
            'rdm_terms':        [],
            'rdm_cor_terms':    [],
            'grouped_terms':    [],
            'hetro_in_means':   [],
            'zi_terms':         [],
            'membership_terms': [],
            'dispersion':       1 if family in ('nb2', 'negbin') else 0,
            'latent_classes':   1,
        }
        builder = ExperimentBuilder(
            df,
            id_col     = id_col,
            y_col      = y_col,
            offset_col = offset_col,
            default_model_family = (
                'count'    if family in ('nb2', 'poisson', 'negbin')
                else 'linear' if family == 'gaussian'
                else 'duration'
            ),
        )
        # fit_manual_model fits a specific spec without running the full SA search
        fitted = builder.fit_manual_model(spec, R=0)  # R=0 → fixed coefficients, no MC draws
        if fitted is None:
            return -1e9
        bic = getattr(fitted, 'bic', None) or getattr(fitted, 'BIC', None)
        ll  = getattr(fitted, 'loglik', None) or getattr(fitted, 'log_lik', None)
        n   = len(df)
        k   = len(selected_vars)
        if bic is not None and np.isfinite(bic):
            return -float(bic) / max(n, 1)
        if ll is not None and np.isfinite(ll):
            return _bic_per_obs(float(ll), k, n)
        return -1e9
    except Exception as e:
        logger.debug(f"_metacount_fit failed ({family}, {selected_vars}): {e}")
        return -1e9


def _metacount_fit_mixed(
    df: pd.DataFrame,
    id_col: str,
    y_col: str,
    selected_vars: List[str],
    random_vars: Optional[List[str]] = None,
    family: str = 'nb2',
    offset_col: Optional[str] = None,
    R: int = 200,
) -> float:
    """
    Fit a MetaCount model with random parameters (mixed model) and return –BIC/n.
    Used for the FINAL estimation after the GA has selected variables.

    random_vars : subset of selected_vars to make random (Normal distribution).
                  If None, uses all selected_vars.
    R           : Halton draws for simulation (200 = good balance; 500 for final paper).
    """
    if not _METACOUNT_OK:
        return -1e9
    if not selected_vars:
        return -1e9

    rdm = random_vars if random_vars is not None else selected_vars
    fixed = [v for v in selected_vars if v not in rdm]

    try:
        spec = {
            'fixed_terms':      fixed,
            'rdm_terms':        [f'{v}:normal' for v in rdm],
            'rdm_cor_terms':    [],
            'grouped_terms':    [],
            'hetro_in_means':   [],
            'zi_terms':         [],
            'membership_terms': [],
            'dispersion':       1 if family in ('nb2', 'negbin') else 0,
            'latent_classes':   1,
        }
        builder = ExperimentBuilder(
            df,
            id_col     = id_col,
            y_col      = y_col,
            offset_col = offset_col,
            default_model_family = (
                'count'    if family in ('nb2', 'poisson', 'negbin')
                else 'linear' if family == 'gaussian'
                else 'duration'
            ),
        )
        fitted = builder.fit_manual_model(spec, R=R)
        if fitted is None:
            return -1e9
        bic = getattr(fitted, 'bic', None) or getattr(fitted, 'BIC', None)
        ll  = getattr(fitted, 'loglik', None) or getattr(fitted, 'log_lik', None)
        n   = len(df)
        k   = len(selected_vars) + len(rdm)  # mean + std per random param
        if bic is not None and np.isfinite(bic):
            return -float(bic) / max(n, 1)
        if ll is not None and np.isfinite(ll):
            return _bic_per_obs(float(ll), k, n)
        return -1e9
    except Exception as e:
        logger.debug(f"_metacount_fit_mixed failed: {e}")
        return -1e9


# ---------------------------------------------------------------------------
# Stage 1 — activity frequency  (count model: NB2 per purpose)
# ---------------------------------------------------------------------------

def stage1_fitness_metacount(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    purposes: List[str],
) -> float:
    """
    GA fitness for Stage 1 using MetaCount NB2 count regression.

    Replaces the ordered logit.  Each activity-frequency outcome
    (n_work_out, n_edu_out, …) is a non-negative integer count —
    NB2 is the statistically correct model family here.

    Returns mean –BIC/n across purposes.
    """
    selected = [f for f, g in zip(feat_names, genes) if g == 1]
    if not selected:
        return -1e9

    df = _ensure_id(data, 'obs_id')
    scores = []
    for purpose in purposes:
        y_col = f'n_{purpose}_out'
        if y_col not in df.columns:
            continue
        sub = df[['obs_id', y_col] + [c for c in selected if c in df.columns]].dropna()
        if len(sub) < 30:
            continue
        score = _metacount_fit(sub, id_col='obs_id', y_col=y_col,
                               selected_vars=[c for c in selected if c in sub.columns],
                               family='nb2')
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if scores else -1e9


# ---------------------------------------------------------------------------
# Stage 2 — day duration  (lognormal regression per target)
# ---------------------------------------------------------------------------

def stage2_fitness_metacount(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    targets: List[str],
) -> float:
    """
    GA fitness for Stage 2 using MetaCount mixed lognormal regression.

    Targets (start_of_day, end_of_day, B) are continuous duration/time
    variables.  MetaCount's 'duration' family fits log-normal regression
    with optional random effects for taste heterogeneity.

    Returns mean –BIC/n across targets.
    """
    selected = [f for f, g in zip(feat_names, genes) if g == 1]
    if not selected:
        return -1e9

    df = _ensure_id(data, 'obs_id')
    scores = []
    for target in targets:
        if target not in df.columns:
            continue
        sub = df[['obs_id', target] + [c for c in selected if c in df.columns]].dropna()
        if len(sub) < 30:
            continue
        # Use Gaussian family for start/end times (can be negative);
        # use duration (lognormal) for B (strictly positive activity budget)
        family = 'duration' if target == 'B' else 'gaussian'
        score = _metacount_fit(sub, id_col='obs_id', y_col=target,
                               selected_vars=[c for c in selected if c in sub.columns],
                               family=family)
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if scores else -1e9


# ---------------------------------------------------------------------------
# Stage 3 — activity duration  (lognormal per purpose)
# ---------------------------------------------------------------------------

def stage3_fitness_metacount(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    purposes: List[str],
    purp_col: str,
    dur_col: str = 'duration_min',
) -> float:
    """
    GA fitness for Stage 3 using MetaCount mixed lognormal regression.

    Each tour's activity duration is a positive continuous variable.
    MetaCount 'duration' family estimates a lognormal regression with
    optional random coefficients — capturing that e.g. the time-sensitivity
    parameter varies across travellers (taste variation without requiring
    a full Larch mixed logit).

    Returns mean –BIC/n across purposes.
    """
    selected = [f for f, g in zip(feat_names, genes) if g == 1]
    if not selected:
        return -1e9
    if dur_col not in data.columns:
        return -1e9

    df = _ensure_id(data, 'obs_id')
    scores = []
    for purpose in purposes:
        sub = df[df[purp_col] == purpose]
        if len(sub) < 30:
            continue
        sub = sub[['obs_id', dur_col] + [c for c in selected if c in sub.columns]].dropna()
        if len(sub) < 30:
            continue
        score = _metacount_fit(sub, id_col='obs_id', y_col=dur_col,
                               selected_vars=[c for c in selected if c in sub.columns],
                               family='duration')
        if np.isfinite(score):
            scores.append(score)

    return float(np.mean(scores)) if scores else -1e9


# ---------------------------------------------------------------------------
# Stage 4 — schedule placement  (MNL/MXL timing choice, SearchLibrium)
# ---------------------------------------------------------------------------

def _to_long_format(
    data: pd.DataFrame,
    choice_col: str,
    feat_cols: List[str],
    n_alts: int,
    id_col: str = 'case_id',
) -> pd.DataFrame:
    """
    Expand wide timing data (one row per trip) to long format
    (one row per alternative per trip) for SearchLibrium.

    Alternatives are coded 0, 1, …, n_alts–1 (time period indices).
    Features are treated as alternative-invariant (idco) — each alternative
    gets the same feature values, with the chosen alternative flagged.
    """
    if id_col not in data.columns:
        data = data.copy()
        data[id_col] = np.arange(len(data))

    frames = []
    for alt in range(n_alts):
        chunk = data[feat_cols + [id_col, choice_col]].copy()
        chunk['alt_id'] = alt
        chunk['chosen'] = (data[choice_col] == alt).astype(int)
        frames.append(chunk)

    long = pd.concat(frames, ignore_index=True)
    return long.sort_values([id_col, 'alt_id']).reset_index(drop=True)


def stage4_fitness_searchlibrium(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    purposes: List[str],
    period_col: str = 'period',
    n_alts: int = 6,
) -> float:
    """
    GA fitness for Stage 4 using SearchLibrium MNL.

    During GA evaluation we use standard MNL (fast — no simulation draws).
    The MNL BIC is comparable to the existing MNL fitness so the GA
    convergence curves are interpretable.

    After the GA selects variables, call stage4_final_mixed_logit() to
    refit the FINAL model as a Mixed Logit with random coefficients.

    Returns mean –BIC/n across purposes.
    """
    selected = [f for f, g in zip(feat_names, genes) if g == 1]
    if not selected or not _SEARCHLIB_OK:
        return -1e9

    scores = []
    for purpose in purposes:
        sub = data[data.get('purpose', data.iloc[:, 0]) == purpose] \
              if 'purpose' in data.columns else data
        if len(sub) < 20:
            continue
        avail_feats = [f for f in selected if f in sub.columns]
        if not avail_feats:
            continue

        try:
            long = _to_long_format(
                sub.copy(), choice_col=period_col,
                feat_cols=avail_feats, n_alts=n_alts,
            )
            n_obs  = len(sub)
            alts   = np.arange(n_alts)

            mnl = MultinomialLogit()
            mnl.setup(
                X        = long[avail_feats].values.astype(float),
                y        = long['chosen'].values.astype(int),
                varnames = avail_feats,
                alts     = alts,
                ids      = long['case_id'].values.astype(int),
                base_alt = alts[0],
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = mnl.fit()

            if not result.success:
                continue
            ll = -float(result.fun)
            k  = len(avail_feats)
            scores.append(_bic_per_obs(ll, k, n_obs))
        except Exception as e:
            logger.debug(f"stage4_fitness_searchlibrium [{purpose}]: {e}")
            continue

    return float(np.mean(scores)) if scores else -1e9


def stage4_final_mixed_logit(
    data: pd.DataFrame,
    selected_vars: List[str],
    purposes: List[str],
    period_col: str = 'period',
    n_alts: int = 6,
    n_draws: int = 500,
    rand_dist: str = 'n',
) -> dict:
    """
    Refit Stage 4 as Mixed Logit (SearchLibrium) after GA variable selection.

    Random coefficients are placed on all selected variables (Normal distribution
    by default), capturing taste variation in time-of-day sensitivity across
    travellers.

    Parameters
    ----------
    data          : output of prepare_timing_data()
    selected_vars : GA-selected features
    purposes      : list of trip purposes to estimate
    period_col    : column encoding chosen time period (int 0..n_alts–1)
    n_alts        : number of time period alternatives
    n_draws       : Halton draws for simulation (500 = production quality)
    rand_dist     : distribution for random params ('n'=Normal, 'ln'=LogNormal)

    Returns dict  {purpose: {'ll': ..., 'bic': ..., 'coefs': ..., 'randvars': ...}}
    """
    if not _SEARCHLIB_OK:
        logger.warning("SearchLibrium not installed — cannot run Mixed Logit for Stage 4")
        return {}

    results = {}
    randvars = {v: rand_dist for v in selected_vars}

    for purpose in purposes:
        sub = data[data['purpose'] == purpose] if 'purpose' in data.columns else data
        if len(sub) < 30:
            continue
        avail_feats = [f for f in selected_vars if f in sub.columns]
        if not avail_feats:
            continue

        try:
            long = _to_long_format(
                sub.copy(), choice_col=period_col,
                feat_cols=avail_feats, n_alts=n_alts,
            )
            alts = np.arange(n_alts)
            rv   = {v: rand_dist for v in avail_feats}

            mxl = MixedLogit()
            mxl.setup(
                X        = long[avail_feats].values.astype(float),
                y        = long['chosen'].values.astype(int),
                varnames = avail_feats,
                alts     = alts,
                randvars = rv,
                ids      = long['case_id'].values.astype(int),
                n_draws  = n_draws,
                halton   = True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = mxl.fit()

            ll = -float(result.fun)
            k  = len(avail_feats) * 2  # mean + std per random param
            n  = len(sub)
            bic = -2 * ll + k * np.log(n)

            results[purpose] = {
                'll':      ll,
                'bic':     bic,
                'coefs':   getattr(mxl, 'coeff_est', result.x).tolist(),
                'randvars': rv,
                'converged': result.success,
                'n_obs':   n,
            }
            print(f"  [{purpose}] MixedLogit  LL={ll:.1f}  BIC={bic:.1f}  "
                  f"converged={result.success}  n={n}")
        except Exception as e:
            logger.warning(f"stage4_final_mixed_logit [{purpose}]: {e}")

    return results


# ---------------------------------------------------------------------------
# Final-estimation wrappers (post-GA, higher draw count)
# ---------------------------------------------------------------------------

def stage1_final_mixed_metacount(
    data: pd.DataFrame,
    selected_vars: List[str],
    purposes: List[str],
    R: int = 500,
) -> dict:
    """
    Refit Stage 1 as MetaCount mixed NB2 (random coefficients) after GA selection.

    Returns {purpose: –BIC/n} for reporting.
    """
    df = _ensure_id(data, 'obs_id')
    results = {}
    for purpose in purposes:
        y_col = f'n_{purpose}_out'
        if y_col not in df.columns:
            continue
        sub  = df[['obs_id', y_col] + [c for c in selected_vars if c in df.columns]].dropna()
        avail = [c for c in selected_vars if c in sub.columns]
        score = _metacount_fit_mixed(sub, 'obs_id', y_col, avail, family='nb2', R=R)
        results[purpose] = score
        print(f"  [Stage1 {purpose}] mixed NB2  –BIC/n={score:.4f}")
    return results


def stage23_final_mixed_metacount(
    data: pd.DataFrame,
    selected_vars: List[str],
    y_col: str,
    stage_label: str = 'Stage2/3',
    R: int = 500,
) -> float:
    """
    Refit Stage 2 or 3 as MetaCount mixed lognormal after GA selection.

    Returns –BIC/n.
    """
    df = _ensure_id(data, 'obs_id')
    sub = df[['obs_id', y_col] + [c for c in selected_vars if c in df.columns]].dropna()
    avail = [c for c in selected_vars if c in sub.columns]
    family = 'duration' if y_col not in ('start_of_day', 'end_of_day') else 'gaussian'
    score = _metacount_fit_mixed(sub, 'obs_id', y_col, avail, family=family, R=R)
    print(f"  [{stage_label} {y_col}] mixed lognormal  –BIC/n={score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Availability report  (call at startup to tell user what's available)
# ---------------------------------------------------------------------------

def report_availability() -> None:
    est = ga_estimator()
    print(f"  GA_ESTIMATOR = '{est}'")
    print(f"  metacountregressor : {'✓ available' if _METACOUNT_OK else '✗ not installed'}")
    print(f"  SearchLibrium      : {'✓ available' if _SEARCHLIB_OK else '✗ not installed'}")
    if not _METACOUNT_OK and est in ('metacount', 'both'):
        print("  → HPC install (editable):")
        print("      pip install -e /mnt/hpccs01/home/ahernz/source/MetaCount")
        print("      pip install jax jaxlib jaxopt")
        print("    OR PyPI: pip install metacountregressor --upgrade")
        print("    Tip: pbs_pipeline_mode.pbs installs automatically when GA_ESTIMATOR is set")
    if not _SEARCHLIB_OK and est in ('searchlibrium', 'both'):
        print("  → HPC install (editable):")
        print("      pip install -e /mnt/hpccs01/home/ahernz/source/SearchLibrium")
        print("    OR PyPI: pip install SearchLibrium --upgrade")
        print("    Tip: pbs_pipeline_mode.pbs installs automatically when GA_ESTIMATOR is set")
