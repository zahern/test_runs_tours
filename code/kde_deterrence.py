"""
kde_deterrence.py
=================
Data-driven deterrence (impedance) functions for gravity accessibility.

Problem
-------
``compute_accessibility`` hard-codes ``exp(-b*tt)`` with hand-picked
``b`` (0.05 work, 0.08 shop, ...) and logit forms for walk/bike. Real
trip-length distributions are fatter-tailed than exponential
(lognormal/gamma-like) — the exponential form systematically
under-weights medium-long trips and over-weights very short ones.

Fix
---
Learn the deterrence function non-parametrically per purpose from
observed HHTS trip times. ``f_p`` is the fitted trip-time SURVIVAL
function, renormalised::

    f_p(t) = S_p(t) / S_p(0),   S_p(t) = P_p(T > t) from the KDE

i.e. "the share of this purpose's trips that exceed time t". This is a
drop-in replacement for ``_decay(tt, ...)`` output in the gravity sum
``log1p( sum_j attractor_j * f_p(tt_ij) )``.

Why survival, not the density ratio ``KDE(t)/KDE(0)``: real trip-time
densities peak AWAY from zero (mode ~8-15 min), so the density ratio
exceeds 1 over most of the range and clips to a useless flat 1.0. The
survival function is monotone decreasing, bounded in [0, 1], and
exactly 1 at t=0 — the three properties a deterrence weight needs.

Also provided:
  * ``exp_equivalent_decay`` — least-squares ``b`` such that
    ``exp(-b*t)`` best matches ``f_p``. Lets callers KEEP the exp form
    (and all downstream code) while calibrating ``b`` from data.
  * ``trip_length_ks`` — split-half KS validation that the learned
    deterrence reproduces the observed trip-length distribution.
  * ``fit_purpose_deterrence`` — batch-fit over purposes with a minimum-
    sample guard (thin purposes return None -> caller keeps exp default
    and the skip is logged, never silent).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _clean_times(times) -> np.ndarray:
    t = np.asarray(pd_to_numeric(times), dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    return t


def pd_to_numeric(values):
    """pd.to_numeric without importing pandas at module import."""
    try:
        import pandas as pd
        return pd.to_numeric(values, errors='coerce')
    except Exception:
        out = np.empty(len(values), dtype=float)
        for i, v in enumerate(values):
            try:
                out[i] = float(v)
            except Exception:
                out[i] = np.nan
        return out


def fit_deterrence(times, bandwidth: float = None):
    """Fit a normalised KDE deterrence function to observed trip times.

    Parameters
    ----------
    times : array-like
        Observed trip travel times (minutes). Non-positive / NaN dropped.
    bandwidth : float or None
        Gaussian KDE bandwidth in minutes. None = Silverman's rule
        (``0.9 * min(sd, iqr/1.34) * n^-0.2``), the standard default.

    Returns
    -------
    callable ``f(tt)`` — the renormalised survival function, with
    ``f(0) == 1`` exactly, monotone non-increasing, ``0 <= f <= 1`` for
    ``tt >= 0``; vectorised over ndarrays of any shape (ravelled
    internally — ``scipy`` KDEs interpret 2-D input as multidim points,
    so callers must never pass matrices straight through).
    Attributes: ``f.bandwidth``, ``f.n_obs``, ``f.median``,
    ``f.grid_hi`` (top of the precomputed survival grid; above it f=0).

    Raises
    ------
    ValueError if fewer than 10 usable observations.
    """
    from scipy.stats import gaussian_kde
    t = _clean_times(times)
    if t.size < 10:
        raise ValueError(f"fit_deterrence needs >=10 usable times, got {t.size}")
    if bandwidth is None:
        sd = float(t.std())
        q75, q25 = float(np.percentile(t, 75)), float(np.percentile(t, 25))
        sig = min(sd, (q75 - q25) / 1.34) if q75 > q25 else sd
        sig = max(sig, 1e-6)
        bandwidth = 0.9 * sig * t.size ** (-0.2)
    kde = gaussian_kde(t, bw_method=float(bandwidth) / max(float(t.std()), 1e-9))

    # survival grid: trapezoid CDF of the KDE, then S = 1 - CDF.
    # hi covers the observed max with margin (plus 200 so capped skim
    # matrices at 90-150 min stay on-grid); beyond hi, f = 0.
    hi = max(float(t.max()) * 1.2, 200.0)
    _n_grid = 1024
    _g = np.linspace(0.0, hi, _n_grid)
    _d = np.asarray(kde(_g), dtype=float)
    _dx = _g[1] - _g[0]
    _cum = np.concatenate([[0.0], np.cumsum((_d[:-1] + _d[1:]) / 2 * _dx)])
    _surv = 1.0 - _cum / max(_cum[-1], 1e-300)
    _s0 = max(float(_surv[0]), 1e-300)

    def f(tt):
        a = np.asarray(tt, dtype=float)
        shp = a.shape
        x = np.clip(a.ravel(), 0.0, hi)
        return (np.interp(x, _g, _surv) / _s0).reshape(shp)

    f.bandwidth = float(bandwidth)
    f.n_obs = int(t.size)
    f.median = float(np.median(t))
    f.grid_hi = float(hi)
    return f


def fit_purpose_deterrence(df, purpose_col: str = 'purpose',
                           time_col: str = 'travel_time',
                           purposes=None, min_n: int = 30,
                           bandwidth: float = None) -> dict:
    """Batch-fit one deterrence function per trip purpose.

    Purposes with fewer than ``min_n`` usable times map to None (caller
    keeps the exp default). Returns ``{purpose: f or None}``.
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if purposes is None:
        purposes = sorted(df[purpose_col].dropna().unique().tolist())
    out = {}
    for p in purposes:
        t = _clean_times(df.loc[df[purpose_col] == p, time_col].values)
        if t.size < min_n:
            logger.info(f"[kde] purpose '{p}': n={t.size} < min_n={min_n} — "
                        "keeping exp default")
            out[p] = None
            continue
        try:
            out[p] = fit_deterrence(t, bandwidth=bandwidth)
            logger.info(f"[kde] purpose '{p}': n={t.size} bw={out[p].bandwidth:.2f} "
                        f"median={out[p].median:.1f}min")
        except Exception as e:                      # noqa: BLE001
            logger.warning(f"[kde] purpose '{p}' fit failed ({e}) — exp default")
            out[p] = None
    return out


def deterrence_weights(f, tt_mat, cap: float = None) -> np.ndarray:
    """Evaluate deterrence ``f`` on a travel-time matrix (drop-in for
    ``_decay`` output). ``cap`` clips times first (mirrors cap_tt).
    Shape-preserving: ``f`` is evaluated on the ravelled matrix and
    reshaped back (never pass a matrix directly to a scipy KDE)."""
    tt = np.asarray(tt_mat, dtype=float)
    if cap is not None:
        tt = np.minimum(tt, float(cap))
    w = np.asarray(f(np.maximum(tt.ravel(), 0.0)), dtype=float)
    w = w.reshape(tt.shape)
    return np.clip(w, 0.0, 1.0).astype(np.float32)


def trip_length_ks(f, times_fit, times_holdout, n_grid: int = 200) -> dict:
    """Split-half validation: does ``f`` (fit on one half) reproduce the
    trip-length distribution of the other half?

    Compares the holdout ECDF to the fit-half KDE's CDF (the distribution
    ``f`` was derived from) via ``scipy.stats.kstest``. Returns
    ``{ks_stat, p_value, n_fit, n_holdout}``.
    """
    from scipy.stats import kstest
    t_fit = _clean_times(times_fit)
    t_ho = _clean_times(times_holdout)
    if t_fit.size < 10 or t_ho.size < 3:
        raise ValueError("need >=10 fit and >=3 holdout times")
    from scipy.stats import gaussian_kde
    _bw = getattr(f, 'bandwidth', None)
    if _bw:
        _bw_arg = float(_bw) / max(float(t_fit.std()), 1e-9)
    else:
        _bw_arg = 'silverman'
    kde = gaussian_kde(t_fit, bw_method=_bw_arg)

    def _cdf(x):
        # numerical CDF of the fit-half KDE via trapezoidal integration
        hi = max(float(t_ho.max()), 1.0) * 1.5
        g = np.linspace(0, hi, n_grid)
        d = np.asarray(kde(g), dtype=float)
        dx = g[1] - g[0]
        cum = np.concatenate([[0.0], np.cumsum((d[:-1] + d[1:]) / 2 * dx)])
        area = max(cum[-1], 1e-300)
        return np.interp(np.asarray(x, dtype=float), g, cum / area)

    ks = kstest(t_ho, _cdf)
    return {'ks_stat': float(ks.statistic), 'p_value': float(ks.pvalue),
            'n_fit': int(t_fit.size), 'n_holdout': int(t_ho.size)}


def exp_equivalent_decay(f, tt_max: float = 120.0, n_grid: int = 240) -> dict:
    """Least-squares ``b`` such that ``exp(-b*t) ~= f(t)`` on [0, tt_max].

    Lets callers keep the exp form (zero downstream code change) with a
    data-calibrated ``b``. Fit in log space (relative error), weighted
    toward short trips where most mass sits (weight ``sqrt(f)``).
    Returns ``{b, rmse, max_abs_err}``.
    """
    g = np.linspace(0, float(tt_max), int(n_grid))
    fv = np.clip(np.asarray(f(g), dtype=float), 1e-12, 1.0)
    w = np.sqrt(fv)
    y = np.log(fv)
    # weighted least squares: y = -b * g  (through origin in log space)
    b = float(-(w * g * y).sum() / max((w * g * g).sum(), 1e-300))
    b = max(b, 1e-6)
    pred = np.exp(-b * g)
    return {'b': b,
            'rmse': float(np.sqrt(((pred - fv) ** 2).mean())),
            'max_abs_err': float(np.abs(pred - fv).max())}
