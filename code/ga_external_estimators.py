"""
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

Controlled by the GA_ESTIMATOR environment variable:

    GA_ESTIMATOR=default        (existing ordered-logit / OLS / MNL — unchanged)
    GA_ESTIMATOR=metacount      (MetaCount NB2 for stage1, mixed lognormal for 2–3)
    GA_ESTIMATOR=searchlibrium  (SearchLibrium MNL for stage4, same fallback for 1–3)
    GA_ESTIMATOR=both           (MetaCount for 1–3, SearchLibrium for 4)
    GA_ESTIMATOR=hybrid         (same as 'both' with default-estimator fallback)

Fitness functions all return –BIC / n (higher = better, same convention as
the rest of ga_hypothesis_testing.py).

The GA is unchanged; only the model fitted at each evaluation differs.
"""
from __future__ import annotations

import inspect
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

_METACOUNT_IMPORT_ERROR = None
_SEARCHLIB_IMPORT_ERROR = None
_METACOUNT_FAILURES: set[tuple[str, str]] = set()
_SEARCHLIB_FAILURES: set[tuple[str, str]] = set()
_METACOUNT_START_VALUES: dict[tuple[str, str, tuple[str, ...]], np.ndarray] = {}
_SEARCHLIB_START_VALUES: dict[tuple[str, str, tuple[str, ...], int], np.ndarray] = {}


def _configure_metacount_backend() -> None:
    """
    Normalise JAX backend hints before importing MetaCount.

    MetaCount uses JAX for matrix operations.  Several CUDA kernel paths
    (cusolver_gesvdj_ffi, rocm) are unreliable or absent on the HPC nodes
    where this pipeline runs.  Force CPU so the GA fitness evaluations are
    stable.  CPU is also faster here: the GA calls MetaCount with small
    per-purpose datasets (few thousand rows), so GPU launch overhead dominates.

    Override with  GA_METACOUNT_FORCE_CPU=0  to allow GPU (not recommended).
    """
    if os.environ.get('GA_METACOUNT_FORCE_CPU', '1') == '1':
        os.environ['JAX_PLATFORMS'] = 'cpu'
        return

    # Fallback normalisation when CPU is not forced:
    platform_hint = os.environ.get('JAX_PLATFORMS', '').strip().lower()
    if platform_hint in ('rocm',):
        os.environ['JAX_PLATFORMS'] = 'cpu'
        logger.warning("JAX_PLATFORMS=rocm is unavailable; forcing cpu for MetaCount")
    elif platform_hint == 'gpu':
        os.environ['JAX_PLATFORMS'] = 'cuda'


_configure_metacount_backend()

try:
    from metacountregressor import ExperimentBuilder, ModelConstraints
    _METACOUNT_OK = True
except Exception as exc:
    _METACOUNT_OK = False
    _METACOUNT_IMPORT_ERROR = exc
    logger.debug("metacountregressor unavailable: %s", exc)

try:
    from SearchLibrium import MultinomialLogit, MixedLogit
    _SEARCHLIB_OK = True
except Exception as exc:
    _SEARCHLIB_OK = False
    _SEARCHLIB_IMPORT_ERROR = exc
    logger.debug("SearchLibrium unavailable: %s", exc)


def ga_estimator() -> str:
    """Read GA_ESTIMATOR env var; default to 'default'."""
    v = os.environ.get('GA_ESTIMATOR', 'default').strip().lower()
    if v not in ('default', 'metacount', 'searchlibrium', 'both', 'hybrid'):
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


def _signature_names(func) -> set[str]:
    try:
        return set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return set()


def _result_success(result) -> bool:
    success = getattr(result, 'success', None)
    return bool(True if success is None else success)


def _result_vector(result) -> Optional[np.ndarray]:
    for attr in ('x', 'params', 'coeff_est', 'coef_', 'betas'):
        value = getattr(result, attr, None)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            continue
        if arr.size:
            return arr
    return None


def _warn_estimator_failure(kind: str, context: str, exc: Exception) -> None:
    registry = _METACOUNT_FAILURES if kind == 'metacount' else _SEARCHLIB_FAILURES
    key = (kind, context)
    if key not in registry:
        registry.add(key)
        logger.warning("%s failure [%s]: %s", kind, context, exc)
    else:
        logger.debug("%s failure [%s]: %s", kind, context, exc)


def _metacount_start_key(family: str, y_col: str, selected_vars: List[str]) -> tuple[str, str, tuple[str, ...]]:
    return family, y_col, tuple(selected_vars)


def _searchlib_start_key(purpose: str, selected_vars: List[str], n_alts: int) -> tuple[str, str, tuple[str, ...], int]:
    return 'mnl', purpose, tuple(selected_vars), int(n_alts)


def _create_metacount_builder(
    df: pd.DataFrame,
    id_col: str,
    y_col: str,
    family: str,
    offset_col: Optional[str] = None,
):
    family_name = (
        'count' if family in ('nb2', 'poisson', 'negbin')
        else 'linear' if family == 'gaussian'
        else 'duration'
    )
    attempts: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, object], ...]] = set()
    id_sets = [
        {'id_col': id_col, 'y_col': y_col},
        {'idvar': id_col, 'yvar': y_col},
        {'id_col': id_col, 'response_col': y_col},
    ]
    family_sets = [
        {},
        {'default_model_family': family_name},
        {'model_family': family_name},
        {'family': family_name},
    ]
    offset_sets = [{}]
    if offset_col:
        offset_sets.extend([
            {'offset_col': offset_col},
            {'offset': offset_col},
            {'exposure_col': offset_col},
        ])

    for id_kwargs in id_sets:
        for family_kwargs in family_sets:
            for offset_kwargs in offset_sets:
                kwargs = dict(id_kwargs)
                kwargs.update(family_kwargs)
                kwargs.update(offset_kwargs)
                key = tuple(sorted(kwargs.items()))
                if key in seen:
                    continue
                seen.add(key)
                attempts.append(kwargs)
    attempts.append({})

    last_exc = None
    for kwargs in attempts:
        try:
            return ExperimentBuilder(df, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError('ExperimentBuilder initialisation failed')


def _fit_metacount_manual_model(builder, spec: dict, R: int, start_vector: Optional[np.ndarray] = None):
    fit_method = builder.fit_manual_model
    param_names = _signature_names(fit_method)
    kwargs: dict[str, object] = {}
    if 'R' in param_names:
        kwargs['R'] = R
    elif 'draws' in param_names:
        kwargs['draws'] = R
    elif 'n_draws' in param_names:
        kwargs['n_draws'] = R

    start_arg_name = None
    if start_vector is not None and start_vector.size:
        for candidate in ('x0', 'start_values', 'start_params', 'initial_values', 'init_params'):
            if candidate in param_names:
                start_arg_name = candidate
                break
    if start_arg_name is not None:
        kwargs[start_arg_name] = start_vector

    try:
        return fit_method(spec, **kwargs) if kwargs else fit_method(spec)
    except TypeError:
        if start_arg_name is not None:
            kwargs.pop(start_arg_name, None)
            return fit_method(spec, **kwargs) if kwargs else fit_method(spec)
        raise


def _fit_searchlib_model(model, start_vector: Optional[np.ndarray] = None):
    fit_method = model.fit
    param_names = _signature_names(fit_method)
    kwargs: dict[str, object] = {}
    start_arg_name = None
    if start_vector is not None and start_vector.size:
        for candidate in ('x0', 'start_values', 'start_params', 'initial_values', 'init_params'):
            if candidate in param_names:
                start_arg_name = candidate
                break
    if start_arg_name is not None:
        kwargs[start_arg_name] = start_vector
    try:
        return fit_method(**kwargs) if kwargs else fit_method()
    except TypeError:
        if start_arg_name is not None:
            kwargs.pop(start_arg_name, None)
            return fit_method(**kwargs) if kwargs else fit_method()
        raise


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
        cache_key = _metacount_start_key(family, y_col, selected_vars)
        builder = _create_metacount_builder(df, id_col=id_col, y_col=y_col,
                                            family=family, offset_col=offset_col)
        fitted = _fit_metacount_manual_model(
            builder, spec, R=0,
            start_vector=_METACOUNT_START_VALUES.get(cache_key),
        )
        if fitted is None:
            return -1e9
        start_vec = _result_vector(fitted)
        if start_vec is not None:
            _METACOUNT_START_VALUES[cache_key] = start_vec
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
        _warn_estimator_failure('metacount', f"fixed:{family}:{y_col}", e)
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
        cache_key = _metacount_start_key(family, y_col, selected_vars)
        seed_vector = _METACOUNT_START_VALUES.get(cache_key)
        builder = _create_metacount_builder(df, id_col=id_col, y_col=y_col,
                                            family=family, offset_col=offset_col)
        fitted = _fit_metacount_manual_model(builder, spec, R=R, start_vector=seed_vector)
        if fitted is None:
            return -1e9
        start_vec = _result_vector(fitted)
        if start_vec is not None:
            _METACOUNT_START_VALUES[cache_key] = start_vec
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
        _warn_estimator_failure('metacount', f"mixed:{family}:{y_col}", e)
        return -1e9


# ---------------------------------------------------------------------------
# Metacount coefficient extraction
# ---------------------------------------------------------------------------

def _extract_metacount_coefs(fitted) -> list:
    """Extract parameter rows from a fitted metacount model object.

    Returns a list of dicts with keys: param, value, std_err, t_stat.
    Tries several attribute naming conventions used by metacountregressor.
    """
    rows = []
    if fitted is None:
        return rows

    # Try summary DataFrame (most reliable)
    for attr in ('summary', 'results_table', 'params_table', 'result'):
        obj = getattr(fitted, attr, None)
        if obj is None:
            continue
        if callable(obj):
            try:
                obj = obj()
            except Exception:
                continue
        try:
            import pandas as pd
            df = pd.DataFrame(obj)
            name_col = next((c for c in df.columns if 'name' in c.lower() or 'param' in c.lower() or 'var' in c.lower()), None)
            val_col  = next((c for c in df.columns if c.lower() in ('coef', 'value', 'estimate', 'mean')), None)
            se_col   = next((c for c in df.columns if 'std' in c.lower() or 'se' in c.lower() or 'error' in c.lower()), None)
            t_col    = next((c for c in df.columns if 'z' == c.lower() or 't_' in c.lower() or 'z_' in c.lower() or c.lower() in ('t', 'z', 'tstat', 'zstat')), None)
            if val_col is not None:
                for _, r in df.iterrows():
                    rows.append({
                        'param':   str(r[name_col]) if name_col else str(_),
                        'value':   float(r[val_col]) if pd.notna(r[val_col]) else float('nan'),
                        'std_err': float(r[se_col])  if se_col and pd.notna(r[se_col]) else float('nan'),
                        't_stat':  float(r[t_col])   if t_col  and pd.notna(r[t_col])  else float('nan'),
                    })
                return rows
        except Exception:
            pass

    # Fallback: raw coeff arrays
    for coef_attr, se_attr, name_attr in [
        ('coeff', 'se', 'param_names'),
        ('params', 'bse', 'param_names'),
        ('coef_', 'std_err_', 'feature_names_'),
        ('estimates', 'standard_errors', 'variable_names'),
    ]:
        coef_arr = getattr(fitted, coef_attr, None)
        if coef_arr is None:
            continue
        try:
            import numpy as _np
            coef_flat = _np.asarray(coef_arr).ravel()
            se_flat   = _np.asarray(getattr(fitted, se_attr, []) or []).ravel()
            names     = list(getattr(fitted, name_attr, []) or [])
            if not names:
                names = [f'x{i}' for i in range(len(coef_flat))]
            for i, (n, v) in enumerate(zip(names, coef_flat)):
                rows.append({
                    'param':   str(n),
                    'value':   float(v),
                    'std_err': float(se_flat[i]) if i < len(se_flat) else float('nan'),
                    't_stat':  float(v / se_flat[i]) if i < len(se_flat) and se_flat[i] != 0 else float('nan'),
                })
            return rows
        except Exception:
            pass
    return rows


def _write_metacount_html(coef_map: dict, html_path: str, title: str) -> None:
    """Write an HTML coefficient report for fitted metacount models.

    coef_map : {segment_label: [{'param', 'value', 'std_err', 't_stat'}, ...]}
    """
    import html as _html
    sections = []
    for seg, rows in coef_map.items():
        if not rows:
            continue
        sections.append(f"<h2>{_html.escape(str(seg))}</h2>")
        sections.append(
            "<table><tr><th>Parameter</th><th>Value</th>"
            "<th>Std Err</th><th>t/z-stat</th></tr>"
        )
        for r in rows:
            t = r.get('t_stat', float('nan'))
            sig = ' style="font-weight:bold"' if abs(t) > 1.96 else ''
            sections.append(
                f"<tr{sig}>"
                f"<td>{_html.escape(str(r['param']))}</td>"
                f"<td style='text-align:right'>{float(r['value']):.6f}</td>"
                f"<td style='text-align:right'>{float(r['std_err']):.6f}</td>"
                f"<td style='text-align:right'>{float(t):.3f}</td>"
                "</tr>"
            )
        sections.append("</table>")

    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(title)}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:20px;font-size:13px}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}"
        "th{background:#2c3e50;color:white;padding:6px 10px;text-align:left}"
        "td{border:1px solid #ddd;padding:5px 10px}"
        "tr:nth-child(even){background:#f9f9f9}"
        "h1,h2{color:#2c3e50}</style></head><body>"
        f"<h1>{_html.escape(title)}</h1>"
        + "\n".join(sections)
        + "</body></html>"
    )
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write(doc)


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
                result = _fit_searchlib_model(mnl)

            if not _result_success(result):
                continue
            ll = -float(result.fun)
            k  = len(avail_feats)
            start_vec = _result_vector(result)
            if start_vec is not None:
                _SEARCHLIB_START_VALUES[_searchlib_start_key(purpose, avail_feats, n_alts)] = start_vec
            scores.append(_bic_per_obs(ll, k, n_obs))
        except Exception as e:
            _warn_estimator_failure('searchlibrium', f"mnl:{purpose}", e)
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
            seed_key = _searchlib_start_key(purpose, avail_feats, n_alts)
            seed_vector = _SEARCHLIB_START_VALUES.get(seed_key)

            if seed_vector is None:
                seed_mnl = MultinomialLogit()
                seed_mnl.setup(
                    X        = long[avail_feats].values.astype(float),
                    y        = long['chosen'].values.astype(int),
                    varnames = avail_feats,
                    alts     = alts,
                    ids      = long['case_id'].values.astype(int),
                    base_alt = alts[0],
                )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    seed_result = _fit_searchlib_model(seed_mnl)
                if _result_success(seed_result):
                    seed_vector = _result_vector(seed_result)
                    if seed_vector is not None:
                        _SEARCHLIB_START_VALUES[seed_key] = seed_vector
                    print(f"  [{purpose}] SearchLibrium seed MNL fitted for MixedLogit start")
            else:
                print(f"  [{purpose}] SearchLibrium seed MNL reused from cached GA fit")

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
                result = _fit_searchlib_model(mxl, start_vector=seed_vector)

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
            _warn_estimator_failure('searchlibrium', f"mixed:{purpose}", e)

    return results


# ---------------------------------------------------------------------------
# Final-estimation wrappers (post-GA, higher draw count)
# ---------------------------------------------------------------------------

def _metacount_fit_mixed_with_model(
    df: pd.DataFrame,
    id_col: str,
    y_col: str,
    selected_vars: List[str],
    family: str = 'nb2',
    R: int = 500,
):
    """Like _metacount_fit_mixed but returns (score, fitted_model) so the caller
    can extract and report coefficients."""
    if not _METACOUNT_OK or not selected_vars:
        return -1e9, None
    rdm   = selected_vars
    fixed = []
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
        cache_key = _metacount_start_key(family, y_col, selected_vars)
        seed_vector = _METACOUNT_START_VALUES.get(cache_key)
        builder = _create_metacount_builder(df, id_col=id_col, y_col=y_col,
                                            family=family, offset_col=None)
        fitted = _fit_metacount_manual_model(builder, spec, R=R, start_vector=seed_vector)
        if fitted is None:
            return -1e9, None
        start_vec = _result_vector(fitted)
        if start_vec is not None:
            _METACOUNT_START_VALUES[cache_key] = start_vec
        bic = getattr(fitted, 'bic', None) or getattr(fitted, 'BIC', None)
        ll  = getattr(fitted, 'loglik', None) or getattr(fitted, 'log_lik', None)
        n   = len(df)
        k   = len(selected_vars) * 2
        if bic is not None and np.isfinite(bic):
            return -float(bic) / max(n, 1), fitted
        if ll is not None and np.isfinite(ll):
            return _bic_per_obs(float(ll), k, n), fitted
        return -1e9, fitted
    except Exception as e:
        _warn_estimator_failure('metacount', f"final:{family}:{y_col}", e)
        return -1e9, None


def stage1_final_mixed_metacount(
    data: pd.DataFrame,
    selected_vars: List[str],
    purposes: List[str],
    R: int = 500,
) -> tuple:
    """
    Refit Stage 1 as MetaCount mixed NB2 (random coefficients) after GA selection.

    Returns (scores_dict, coef_map) where scores_dict is {purpose: –BIC/n}
    and coef_map is {purpose: [coef_rows]} for HTML reporting.
    """
    df = _ensure_id(data, 'obs_id')
    scores, coef_map = {}, {}
    for purpose in purposes:
        y_col = f'n_{purpose}_out'
        if y_col not in df.columns:
            continue
        sub   = df[['obs_id', y_col] + [c for c in selected_vars if c in df.columns]].dropna()
        avail = [c for c in selected_vars if c in sub.columns]
        score, fitted = _metacount_fit_mixed_with_model(sub, 'obs_id', y_col, avail, family='nb2', R=R)
        scores[purpose]   = score
        coef_map[purpose] = _extract_metacount_coefs(fitted)
        print(f"  [Stage1 {purpose}] mixed NB2  –BIC/n={score:.4f}")
    return scores, coef_map


def stage23_final_mixed_metacount(
    data: pd.DataFrame,
    selected_vars: List[str],
    y_col: str,
    stage_label: str = 'Stage2/3',
    R: int = 500,
) -> tuple:
    """
    Refit Stage 2 or 3 as MetaCount mixed lognormal after GA selection.

    Returns (score, coef_rows) where coef_rows is a list of coef dicts
    for HTML reporting.
    """
    df = _ensure_id(data, 'obs_id')
    sub   = df[['obs_id', y_col] + [c for c in selected_vars if c in df.columns]].dropna()
    avail = [c for c in selected_vars if c in sub.columns]
    family = 'duration' if y_col not in ('start_of_day', 'end_of_day') else 'gaussian'
    score, fitted = _metacount_fit_mixed_with_model(sub, 'obs_id', y_col, avail, family=family, R=R)
    coef_rows = _extract_metacount_coefs(fitted)
    print(f"  [{stage_label} {y_col}] mixed lognormal  –BIC/n={score:.4f}")
    return score, coef_rows


# ---------------------------------------------------------------------------
# Availability report  (call at startup to tell user what's available)
# ---------------------------------------------------------------------------

def report_availability() -> None:
    est = ga_estimator()
    print(f"  GA_ESTIMATOR = '{est}'")
    print(f"  metacountregressor : {'✓ available' if _METACOUNT_OK else '✗ not installed'}")
    print(f"  SearchLibrium      : {'✓ available' if _SEARCHLIB_OK else '✗ not installed'}")
    if not _METACOUNT_OK and est in ('metacount', 'both', 'hybrid'):
        if _METACOUNT_IMPORT_ERROR is not None:
            print(f"  Import error: {_METACOUNT_IMPORT_ERROR}")
        print("  → HPC install (editable):")
        print("      pip install -e /mnt/hpccs01/home/ahernz/source/MetaCount")
        print("      pip install jax jaxlib jaxopt")
        print("    OR PyPI: pip install metacountregressor --upgrade")
        print("    Tip: pbs_pipeline_mode.pbs installs automatically when GA_ESTIMATOR is set")
    if not _SEARCHLIB_OK and est in ('searchlibrium', 'both', 'hybrid'):
        if _SEARCHLIB_IMPORT_ERROR is not None:
            print(f"  Import error: {_SEARCHLIB_IMPORT_ERROR}")
        print("  → HPC install (editable):")
        print("      pip install -e /mnt/hpccs01/home/ahernz/source/SearchLibrium")
        print("    OR PyPI: pip install SearchLibrium --upgrade")
        print("    Tip: pbs_pipeline_mode.pbs installs automatically when GA_ESTIMATOR is set")
