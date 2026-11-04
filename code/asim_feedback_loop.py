"""
asim_feedback_loop.py
=====================
Close the loop between the estimated mode-choice model and the
ActivitySim-style logsum accessibility (compute_accessibility.
compute_logsum_accessibility).

Problem
-------
``compute_logsum_accessibility`` builds ``asim_<purpose>_acc`` from mode
utilities ``V_m = ASC_m + BETA_m * time_m`` whose coefficients are
ActivitySim defaults borrowed for SEQ (BETA_TIME_CAR=-0.035,
ASC_TRANSIT=-0.8, ...), NOT estimates. Meanwhile the fitted Stage-5/6
mode model estimates its own time coefficients and ASCs
(``c_ivt_auto``, ``ASC_Transit``, ...) in ``_logsums_numpy_fallback`` —
but those estimates never flow back into the accessibility build. The
accessibility that enters destination choice as a regressor is therefore
built from different behaviour than the model estimates.

Fix (block-coordinate ascent on the joint likelihood)
-----------------------------------------------------
1. Estimate the mode model -> coefficient dict (same namespace the
   fallback reads: ``c_ivt_auto``, ``c_ivt_transit``, ``c_walk_time`` /
   ``c_walk_dist``, ``c_bike_time`` / ``c_bike_dist``, ``ASC_SR``,
   ``ASC_Walk``, ``ASC_Bike``, ``ASC_Transit``). Any source works: a
   fitted Larch model via ``_coef_dict_from_model``, a ``coef_override``
   dict, or hand values.
2. Map those coefficients onto the 11 ``ASIM_*`` env knobs (unit
   conversions below) and rebuild ``asim_*_acc``.
3. Re-estimate with the new accessibility; repeat until the destination
   log-likelihood gain falls below ``ll_tol``.

Unit conversions (must match ``_logsums_numpy_fallback`` exactly)
------------------------------------------------------------------
Fallback utilities (stage5_destination_choice.py)::

    v_da      = c_ivt_auto    * auto_h            # auto_h in HOURS
    v_transit = c_ivt_transit * trans_h           # trans_h in HOURS
    v_walk    = c_walk_time   * log(DISTWALK*12+1)  # log-MINUTES
    v_bike    = c_bike_time   * log(DISTBIKE*4 +1)  # log-MINUTES

Asim utilities (compute_accessibility.py) are LINEAR in minutes::

    V_car = ASC_CAR + BETA_TIME_CAR * mins

Hence:

* ``BETA_TIME_CAR     = c_ivt_auto / 60``      (per-hour -> per-minute)
* ``BETA_TIME_TRANSIT = c_ivt_transit / 60``
* Walk/bike are log-form in the fallback but linear in asim. Linearise
  at the median observed trip time: d/dmins[c*log(mins+1)] = c/(mins+1),
  so ``BETA_TIME_WALK = c_walk_time / (med_walk_min + 1)`` (same bike).
  This is a first-order approximation — documented, tested, and exact
  only at the median. Callers SHOULD pass observed medians; defaults
  (15/20 min) are SEQ-plausible fallbacks.
* ASCs: DA is the fallback reference (``v_da`` carries no ASC), matching
  asim's ``ASC_CAR = 0.0`` reference. ``ASC_TRANSIT/WALK/BIKE`` pass
  through directly. SR has no asim counterpart (folded into car — the
  only information loss in the mapping; ``ASC_SR`` is reported back so
  callers can see its magnitude).

Fallback chains mirror the fallback's own ``_pv`` chains
(``c_ivt_auto`` <- ``c_ivt``; ``c_ivt_transit`` <- ``c_ivt_auto``;
``c_walk_time`` <- ``c_walk_dist``; ``c_bike_time`` <- ``c_bike_dist``).
"""

import contextlib
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mirror of compute_accessibility defaults (single source of truth for the
# "no-estimate-yet" starting point of the loop).
ASIM_ENV_DEFAULTS = {
    'ASIM_BETA_TIME_CAR':     -0.035,
    'ASIM_BETA_TIME_TRANSIT': -0.045,
    'ASIM_BETA_TIME_WALK':    -0.060,
    'ASIM_BETA_TIME_BIKE':    -0.045,
    'ASIM_ASC_CAR':            0.0,
    'ASIM_ASC_TRANSIT':       -0.8,
    'ASIM_ASC_WALK':          -0.2,
    'ASIM_ASC_BIKE':          -0.4,
    'ASIM_WALK_MAX_KM':        5.0,
    'ASIM_BIKE_MAX_KM':       15.0,
    'ASIM_SIZE_GAMMA':         0.1,
}

_ASIM_ENV_KEYS = tuple(ASIM_ENV_DEFAULTS)

# Fallback-namespace defaults (must match _logsums_numpy_fallback _pv calls).
_FB_DEFAULTS = {
    'c_ivt_auto':    -1.20,
    'c_ivt_transit': -0.90,
    'c_walk_time':   -1.50,
    'c_bike_time':   -1.00,
    'ASC_Transit':   -0.50,
    'ASC_Walk':      -1.00,
    'ASC_Bike':      -1.50,
}


def _fb(coef: dict, name: str, chain: tuple) -> float:
    """Fallback-namespace lookup with the fallback's own _pv chains."""
    for key in (name,) + tuple(chain):
        v = coef.get(key)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not np.isnan(fv):
            return fv
    return _FB_DEFAULTS[name]


def coefs_to_asim_env(coef: dict,
                      med_walk_min: float = 15.0,
                      med_bike_min: float = 20.0) -> dict:
    """Map fitted fallback-namespace coefficients onto ASIM_* env values.

    Parameters
    ----------
    coef : dict
        ``{param_name: value}`` in the ``_logsums_numpy_fallback``
        namespace (i.e. what ``_coef_dict_from_model`` /
        ``coef_override`` provide).
    med_walk_min, med_bike_min : float
        Median observed walk/bike trip minutes — linearisation point for
        the log-form walk/bike coefficients. Pass sample medians; the
        defaults are SEQ-plausible fallbacks.

    Returns
    -------
    dict
        ``{ASIM_* env name: float}`` ready for :func:`asim_env`.
        Also includes ``'_meta'`` with the intermediate mapped values
        (per-hour inputs, linearisation denominators, dropped ASC_SR)
        for auditability — strip it before exporting to ``os.environ``.
    """
    coef = dict(coef or {})
    c_auto = _fb(coef, 'c_ivt_auto', ('c_ivt',))
    c_trn = _fb(coef, 'c_ivt_transit', ('c_ivt_auto', 'c_ivt'))
    c_wlk = _fb(coef, 'c_walk_time', ('c_walk_dist',))
    c_bik = _fb(coef, 'c_bike_time', ('c_bike_dist',))
    asc_trn = _fb(coef, 'ASC_Transit', ())
    asc_wlk = _fb(coef, 'ASC_Walk', ())
    asc_bik = _fb(coef, 'ASC_Bike', ())
    asc_sr = coef.get('ASC_SR', None)
    try:
        asc_sr = float(asc_sr) if asc_sr is not None else 0.0
    except Exception:
        asc_sr = 0.0

    den_w = max(float(med_walk_min) + 1.0, 1.0)
    den_b = max(float(med_bike_min) + 1.0, 1.0)
    env = {
        'ASIM_BETA_TIME_CAR':     c_auto / 60.0,
        'ASIM_BETA_TIME_TRANSIT': c_trn / 60.0,
        'ASIM_BETA_TIME_WALK':    c_wlk / den_w,
        'ASIM_BETA_TIME_BIKE':    c_bik / den_b,
        'ASIM_ASC_CAR':           0.0,       # DA reference, both systems
        'ASIM_ASC_TRANSIT':       asc_trn,
        'ASIM_ASC_WALK':          asc_wlk,
        'ASIM_ASC_BIKE':          asc_bik,
        '_meta': {
            'c_ivt_auto_per_hr': c_auto,
            'c_ivt_transit_per_hr': c_trn,
            'c_walk_time_per_logmin': c_wlk,
            'c_bike_time_per_logmin': c_bik,
            'walk_linearisation_denom': den_w,
            'bike_linearisation_denom': den_b,
            'ASC_SR_folded_into_car': asc_sr,  # information loss, reported
        },
    }
    return env


@contextlib.contextmanager
def asim_env(env: dict):
    """Temporarily set ASIM_* env vars; restore afterwards (never leaks).

    ``'_meta'`` keys are ignored. Yields the previous values.
    """
    saved = {}
    keys = [k for k in env if k in _ASIM_ENV_KEYS]
    for k in keys:
        saved[k] = os.environ.get(k)
        os.environ[k] = str(float(env[k]))
    try:
        yield saved
    finally:
        for k in keys:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


def rebuild_asim_accessibility(landuse: pd.DataFrame, skim_df: pd.DataFrame,
                               coef: dict, zone_ids=None,
                               med_walk_min: float = 15.0,
                               med_bike_min: float = 20.0):
    """Rebuild ``asim_*_acc`` from ESTIMATED (not default) mode coefficients.

    Returns ``(acc_df, env)`` where ``env`` is the applied mapping
    (auditable). Env changes are scoped — the process environment is
    restored before return.
    """
    from compute_accessibility import compute_logsum_accessibility
    env = coefs_to_asim_env(coef, med_walk_min, med_bike_min)
    with asim_env(env):
        acc = compute_logsum_accessibility(landuse, skim_df,
                                           zone_ids=zone_ids)
    return acc, env


def compare_acc(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Per-column diagnostics between two accessibility builds.

    Returns ``{col: {corr, mean_abs_change, rel_change, cv_old, cv_new}}``
    plus ``'_max_abs_change'``. corr < 0.99 or large rel_change means the
    feedback materially moved accessibility (good — the loop is doing
    something); corr ~ 1.0 means convergence.
    """
    out = {}
    mx = 0.0
    for c in old.columns:
        if c not in new.columns:
            continue
        a = pd.to_numeric(old[c], errors='coerce').values.astype(float)
        b = pd.to_numeric(new[c], errors='coerce').values.astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        a, b = a[ok], b[ok]
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
        mac = float(np.abs(b - a).max())
        rel = float(mac / (np.abs(a).mean() + 1e-12))
        cv = lambda v: float(v.std() / (abs(v.mean()) + 1e-12))
        out[c] = {'corr': corr, 'mean_abs_change': float(np.abs(b - a).mean()),
                  'rel_change': rel, 'cv_old': cv(a), 'cv_new': cv(b)}
        mx = max(mx, mac)
    out['_max_abs_change'] = mx
    return out


def run_feedback_loop(landuse: pd.DataFrame, skim_df: pd.DataFrame,
                      estimate_fn, coef0: dict = None,
                      max_iter: int = 5, ll_tol: float = 1e-4,
                      med_walk_min: float = 15.0,
                      med_bike_min: float = 20.0,
                      verbose: bool = True) -> dict:
    """Block-coordinate ascent: acc -> estimate -> acc -> ... until LL stalls.

    Parameters
    ----------
    estimate_fn : callable
        ``estimate_fn(acc_df) -> (coef_dict, ll)``. In production this
        wraps the Stage-5/6 estimators (attach acc, estimate, return the
        fallback-namespace coef dict + destination LL). In tests a stub
        suffices (see the S9 health check).
    coef0 : dict or None
        Starting coefficients; None = ASIM defaults (first estimate uses
        default-built accessibility).
    max_iter, ll_tol : int, float
        Stop after ``max_iter`` rebuilds or when the LL gain from a
        rebuild falls below ``ll_tol`` (both non-improvement and tiny
        improvement stop — the loop must terminate).

    Returns
    -------
    dict with ``history`` (per-iter ``{ll, ll_gain, env, acc_delta}``),
    ``final_acc``, ``final_coef``, ``final_ll``, ``converged``.
    """
    hist = []
    if coef0 is None:
        coef = {k: v for k, v in _FB_DEFAULTS.items()}
        coef['c_ivt'] = -1.10
        # seed env from ActivitySim defaults so iter-0 == status quo ante
        with asim_env({k: v for k, v in ASIM_ENV_DEFAULTS.items()
                       if k in _ASIM_ENV_KEYS}):
            from compute_accessibility import compute_logsum_accessibility
            acc = compute_logsum_accessibility(landuse, skim_df)
        coef, ll = estimate_fn(acc)
        hist.append({'iter': 0, 'll': float(ll), 'll_gain': float('nan'),
                     'note': 'defaults-built acc'})
    else:
        acc, _env = rebuild_asim_accessibility(
            landuse, skim_df, coef0, med_walk_min=med_walk_min,
            med_bike_min=med_bike_min)
        coef, ll = estimate_fn(acc)
        hist.append({'iter': 0, 'll': float(ll), 'll_gain': float('nan'),
                     'note': 'coef0-built acc'})

    converged = False
    best = {'acc': acc, 'coef': coef, 'll': float(ll)}
    for it in range(1, max_iter + 1):
        new_acc, env = rebuild_asim_accessibility(
            landuse, skim_df, coef, med_walk_min=med_walk_min,
            med_bike_min=med_bike_min)
        delta = compare_acc(acc, new_acc)
        new_coef, new_ll = estimate_fn(new_acc)
        gain = float(new_ll) - float(ll)
        hist.append({'iter': it, 'll': float(new_ll), 'll_gain': gain,
                     'acc_max_abs_change': delta.get('_max_abs_change'),
                     'env_meta': env.get('_meta')})
        if verbose:
            logger.info(f"[asim-loop] iter {it}: LL {ll:.4f} -> {new_ll:.4f} "
                        f"(gain {gain:+.6f}), acc max|d| "
                        f"{delta.get('_max_abs_change', float('nan')):.4f}")
        acc, coef, ll = new_acc, new_coef, float(new_ll)
        if float(new_ll) > best['ll']:
            best = {'acc': new_acc, 'coef': new_coef, 'll': float(new_ll)}
        if gain < ll_tol:      # stalled or regressed -> stop, keep best
            converged = gain >= 0
            break
    else:
        converged = False      # hit max_iter without stalling

    return {'history': hist, 'final_acc': best['acc'],
            'final_coef': best['coef'], 'final_ll': best['ll'],
            'converged': converged, 'n_iters': len(hist) - 1}
