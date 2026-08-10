"""
mobility_models.py — zero-inflated mobility-attribute models (POLARIS lr1-style)
================================================================================
A family of pre-day models that give the synthetic population its mobility
attributes, each estimated on HHTS and each emitting a validation report:

  person-level "activity level" (weekly trips by mode, excess-zero heavy):
    • estimate_transit_usage   — weekly transit trips
    • estimate_car_usage       — weekly car (driver+passenger) trips
    • estimate_bike_usage      — weekly bike trips
  household-level ownership (count of stock):
    • estimate_household_vehicles — vehicles owned (lr1 auto-ownership)
    • estimate_household_bikes    — bikes owned

All use ZeroInflatedCount: a PROBIT inflation gate π = Φ(Xg·γ) for the
structural-zero class ("never uses / never owns") combined with a Poisson or
Negative-Binomial count for the level among potential users. This is the
zero-inflated-probit form the classic POLARIS mobility models use — most people
never bike / never take transit, so a plain count model badly under-predicts the
zeros, while the gate captures the "structural non-user" mass explicitly.

Each fit produces a validation report (validation_report): observed-vs-predicted
zero share, mean, dispersion, the count histogram, and LL/AIC/BIC — written as
CSV + HTML (+ PNG when matplotlib is present) under <out_dir>/mobility/.

Pure numpy/scipy/pandas (+ optional matplotlib) so it estimates and validates in
the base env — no jax/larch. Wired into the pipeline as gated pre-day modules
(PIPELINE_MOBILITY) before CDAP, mirroring wfh_telecommute.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import optimize as _opt
    from scipy.special import gammaln as _gammaln
    from scipy.stats import norm as _norm
    _SCIPY_OK = True
except Exception:                                    # pragma: no cover
    _SCIPY_OK = False

logger = logging.getLogger(__name__)

_PHI = (lambda z: _norm.cdf(z)) if _SCIPY_OK else (lambda z: 0.5 * (1 + np.tanh(0.8 * z)))


def _logsumexp2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


# ─────────────────────────────────────────────────────────────────────────────
# Zero-inflated count model: probit gate + Poisson / NegBin count
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ZeroInflatedCount:
    """π = Φ(Xg·γ) structural-zero prob; count ~ Poisson/NegBin(μ=exp(Xc·β)).

    Log-likelihood (y = observed count):
        y == 0 : log( π + (1-π)·P(0) )
        y >  0 : log( (1-π)·P(y) )
    Fit by L-BFGS-B on the ridge-penalised negative LL (numeric gradient)."""
    negbin: bool = True
    l2: float = 1.0
    gamma_: Optional[np.ndarray] = None      # gate coefficients (incl. intercept)
    beta_: Optional[np.ndarray] = None       # count coefficients (incl. intercept)
    log_alpha_: float = 0.0                  # NegBin log-dispersion
    gate_names_: list = field(default_factory=list)
    count_names_: list = field(default_factory=list)
    n_: int = 0
    ll_: float = np.nan

    # -- count-part log pmf / log P(0) -------------------------------------
    def _count_logpmf(self, y, mu, alpha):
        if self.negbin:
            r = 1.0 / max(alpha, 1e-6)
            return (_gammaln(y + r) - _gammaln(r) - _gammaln(y + 1)
                    + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
        return y * np.log(mu) - mu - _gammaln(y + 1)

    def _count_logp0(self, mu, alpha):
        if self.negbin:
            r = 1.0 / max(alpha, 1e-6)
            return r * np.log(r / (r + mu))
        return -mu

    def _unpack(self, theta, kg, kc):
        gamma = theta[:kg]
        beta = theta[kg:kg + kc]
        log_alpha = theta[kg + kc] if self.negbin else 0.0
        return gamma, beta, log_alpha

    def _neg_ll(self, theta, Xg, Xc, y, kg, kc):
        gamma, beta, log_alpha = self._unpack(theta, kg, kc)
        alpha = np.exp(np.clip(log_alpha, -6, 4))
        pi = np.clip(_PHI(Xg @ gamma), 1e-9, 1 - 1e-9)
        mu = np.exp(np.clip(Xc @ beta, -30, 30))
        lp0 = self._count_logp0(mu, alpha)
        is0 = (y == 0)
        ll = np.empty_like(mu)
        # y==0: log( pi + (1-pi) e^{lp0} ) = logsumexp(log pi, log(1-pi)+lp0)
        ll[is0] = _logsumexp2(np.log(pi[is0]),
                              np.log1p(-pi[is0]) + lp0[is0])
        # y>0
        nz = ~is0
        ll[nz] = np.log1p(-pi[nz]) + self._count_logpmf(y[nz], mu[nz], alpha)
        pen = self.l2 * (np.sum(gamma[1:] ** 2) + np.sum(beta[1:] ** 2))
        return -(np.sum(ll) - pen)

    def fit(self, Xg, Xc, y):
        y = np.asarray(y, float)
        Xg = np.column_stack([np.ones(len(y)), np.asarray(Xg, float)])
        Xc = np.column_stack([np.ones(len(y)), np.asarray(Xc, float)])
        kg, kc = Xg.shape[1], Xc.shape[1]
        # warm start: gate intercept ≈ probit of observed zero share; count
        # intercept ≈ log mean of positives.
        p0 = np.clip(y.mean() == 0 and 0.5 or (y == 0).mean(), 1e-3, 1 - 1e-3)
        theta0 = np.zeros(kg + kc + (1 if self.negbin else 0))
        theta0[0] = _norm.ppf(p0) if _SCIPY_OK else 0.0
        pos = y[y > 0]
        theta0[kg] = np.log(pos.mean()) if len(pos) else 0.0
        if not _SCIPY_OK:
            self.gamma_, self.beta_ = theta0[:kg], theta0[kg:kg + kc]
            self.n_ = len(y)
            return self
        res = _opt.minimize(self._neg_ll, theta0, args=(Xg, Xc, y, kg, kc),
                            method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-9})
        g, b, la = self._unpack(res.x, kg, kc)
        self.gamma_, self.beta_, self.log_alpha_ = g, b, float(la)
        self.n_, self.ll_ = len(y), float(-res.fun)
        return self

    def _pi_mu(self, Xg, Xc):
        Xg = np.column_stack([np.ones(len(Xg)), np.asarray(Xg, float)])
        Xc = np.column_stack([np.ones(len(Xc)), np.asarray(Xc, float)])
        pi = np.clip(_PHI(Xg @ self.gamma_), 0, 1)
        mu = np.exp(np.clip(Xc @ self.beta_, -30, 30))
        return pi, mu

    def predict_expected(self, Xg, Xc):
        """E[y] = (1-π)·μ."""
        pi, mu = self._pi_mu(Xg, Xc)
        return (1 - pi) * mu

    def sample(self, Xg, Xc, rng):
        """Draw a count: structural zero w.p. π, else Poisson/NegBin(μ)."""
        pi, mu = self._pi_mu(Xg, Xc)
        struct0 = rng.random(len(pi)) < pi
        if self.negbin:
            alpha = np.exp(np.clip(self.log_alpha_, -6, 4))
            r = 1.0 / max(alpha, 1e-6)
            p = r / (r + mu)
            draw = rng.negative_binomial(r, np.clip(p, 1e-6, 1 - 1e-6))
        else:
            draw = rng.poisson(mu)
        draw = np.where(struct0, 0, draw)
        return draw.astype(np.int32)

    @property
    def k_(self):
        return (len(self.gamma_) + len(self.beta_) + (1 if self.negbin else 0)
                if self.gamma_ is not None else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Validation report
# ─────────────────────────────────────────────────────────────────────────────
def validation_report(name: str, y_true, y_pred_expected, model: ZeroInflatedCount,
                      out_dir: Optional[str] = None, y_sampled=None) -> dict:
    """Observed-vs-predicted diagnostics for one mobility model. Returns a metrics
    dict and, if out_dir given, writes <name>_validation.csv/.html (+ .png)."""
    y = np.asarray(y_true, float)
    yp = np.asarray(y_pred_expected, float)
    n = len(y)
    k = model.k_ if model is not None else np.nan
    ll = getattr(model, 'll_', np.nan)
    obs_zero = float((y == 0).mean())
    # predicted P(0) = E over rows of [π + (1-π)P(0)] is not yp; approximate the
    # predicted zero share from a Poisson/NegBin structural draw if provided, else
    # from the expected-value model's implied zero mass.
    pred_zero = float((np.asarray(y_sampled) == 0).mean()) if y_sampled is not None \
        else float(np.mean(np.exp(-np.clip(yp, 0, 30))))
    metrics = {
        'model': name, 'n': n, 'k': k,
        'loglik': ll,
        'aic': (2 * k - 2 * ll) if np.isfinite(ll) else np.nan,
        'bic': (k * np.log(n) - 2 * ll) if np.isfinite(ll) else np.nan,
        'obs_mean': float(y.mean()), 'pred_mean': float(yp.mean()),
        'obs_zero_share': obs_zero, 'pred_zero_share': pred_zero,
        'obs_var_mean_ratio': float(y.var() / y.mean()) if y.mean() > 0 else np.nan,
        'obs_max': float(y.max()), 'pred_max': float(yp.max()),
        'mean_abs_err': float(np.mean(np.abs(y - yp))),
    }
    logger.info(f"[mobility:{name}] n={n} obs_mean={metrics['obs_mean']:.2f} "
                f"pred_mean={metrics['pred_mean']:.2f} obs_zero={obs_zero:.2f} "
                f"pred_zero={pred_zero:.2f} BIC={metrics['bic']:.0f}")
    if not out_dir:
        return metrics
    try:
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(out_dir, f'{name}_validation.csv'), index=False)
        # count-distribution table (obs vs predicted-expected binned)
        bins = list(range(0, 11)) + [999]
        obs_h = pd.Series(pd.cut(y, bins=bins, right=False)).value_counts(normalize=True).sort_index()
        pred_h = pd.Series(pd.cut(yp, bins=bins, right=False)).value_counts(normalize=True).sort_index()
        dist = pd.DataFrame({'obs_share': obs_h, 'pred_share': pred_h}).fillna(0)
        dist.to_csv(os.path.join(out_dir, f'{name}_count_dist.csv'))
        _write_html_report(name, metrics, dist, out_dir, model)
        _plot_report(name, y, yp, out_dir)
    except Exception as e:
        logger.debug(f"[mobility:{name}] report write skipped ({e})")
    return metrics


def _write_html_report(name, metrics, dist, out_dir, model):
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" if isinstance(v, float)
        else f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in metrics.items())
    dist_rows = "".join(
        f"<tr><td>{idx}</td><td>{r.obs_share:.3f}</td><td>{r.pred_share:.3f}</td></tr>"
        for idx, r in dist.iterrows())
    coefs = ""
    if model is not None and model.gamma_ is not None:
        gn = ['const'] + list(model.gate_names_)
        cn = ['const'] + list(model.count_names_)
        coefs = ("<h3>Gate (probit, structural-zero)</h3><table><tr><th>var</th><th>γ</th></tr>"
                 + "".join(f"<tr><td>{g}</td><td>{v:.3f}</td></tr>"
                           for g, v in zip(gn, model.gamma_))
                 + "</table><h3>Count (log-link)</h3><table><tr><th>var</th><th>β</th></tr>"
                 + "".join(f"<tr><td>{c}</td><td>{v:.3f}</td></tr>"
                           for c, v in zip(cn, model.beta_))
                 + f"</table><p>NegBin log-α = {model.log_alpha_:.3f}</p>")
    html = f"""<html><head><meta charset='utf-8'><title>{name} validation</title>
<style>body{{font-family:system-ui,sans-serif;margin:2em;max-width:860px}}
table{{border-collapse:collapse;margin:1em 0}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}
img{{max-width:100%}}</style></head><body>
<h2>{name} — zero-inflated count validation</h2>
<table><tr><th>metric</th><th>value</th></tr>{rows}</table>
<h3>Count distribution (share by count bin)</h3>
<table><tr><th>bin</th><th>observed</th><th>predicted</th></tr>{dist_rows}</table>
{coefs}
<img src='{name}_validation.png' alt='{name} plot'/>
</body></html>"""
    with open(os.path.join(out_dir, f'{name}_validation.html'), 'w', encoding='utf-8') as fh:
        fh.write(html)


def _plot_report(name, y, yp, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    mx = int(min(10, max(y.max(), 1)))
    bins = np.arange(0, mx + 2) - 0.5
    ax[0].hist(np.clip(y, 0, mx), bins=bins, alpha=0.6, label='observed', density=True)
    ax[0].hist(np.clip(yp, 0, mx), bins=bins, alpha=0.6, label='predicted E[y]', density=True)
    ax[0].set_title(f'{name}: count distribution'); ax[0].set_xlabel('trips / count')
    ax[0].legend()
    ax[1].bar(['observed', 'predicted'],
              [(y == 0).mean(), np.mean(np.exp(-np.clip(yp, 0, 30)))],
              color=['#4c72b0', '#dd8452'])
    ax[1].set_title(f'{name}: zero share'); ax[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_validation.png'), dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Feature builders
# ─────────────────────────────────────────────────────────────────────────────
def _num(df, name, default=0.0):
    return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)),
                         errors='coerce').fillna(default).values.astype(float)


def _person_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Shared person-level drivers (present in both HHTS and synth)."""
    cols, names = [], []
    cols.append((_num(df, 'incp', 7.0) - 7.0) / 4.0); names.append('incp_z')
    cols.append((_num(df, 'vehrd', 1.0)));            names.append('hh_vehicles')
    cols.append((_num(df, 'nprd', 2.0) - 2.0) / 2.0); names.append('hh_size_z')
    cols.append((_num(df, 'age5p', 8.0) - 8.0) / 4.0); names.append('age_z')
    if 'sexp' in df.columns:
        cols.append((_num(df, 'sexp', 1.0) == 2).astype(float)); names.append('is_female')
    if 'CARLICENCE' in df.columns:
        cols.append((_num(df, 'CARLICENCE', 0.0) > 0).astype(float)); names.append('has_licence')
    if 'WORKSTAT' in df.columns:
        ws = _num(df, 'WORKSTAT', 0.0)
        cols.append((ws == 1).astype(float)); names.append('is_ftworker')
        cols.append((ws == 2).astype(float)); names.append('is_ptworker')
    if 'STUDYING' in df.columns:
        cols.append((_num(df, 'STUDYING', 0.0) > 0).astype(float)); names.append('is_student')
    return np.column_stack(cols), names


def _household_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    cols.append((_num(df, 'incp', 7.0) - 7.0) / 4.0);  names.append('hh_income_z')
    cols.append((_num(df, 'nprd', 2.0) - 2.0) / 2.0);  names.append('hh_size_z')
    cols.append(_num(df, 'n_workers', 1.0));           names.append('n_workers')
    cols.append(_num(df, 'n_adults', 2.0));            names.append('n_adults')
    if 'dwelling_density' in df.columns:
        cols.append((_num(df, 'dwelling_density', 0.0)) / 1000.0); names.append('density_k')
    return np.column_stack(cols), names


# ─────────────────────────────────────────────────────────────────────────────
# Per-person weekly trip counts by mode (from HHTS trips)
# ─────────────────────────────────────────────────────────────────────────────
_MODE_ALIASES = {
    'transit': ['transit', 'pt', 'bus', 'train', 'tram', 'ferry', 'public'],
    'car':     ['da', 'sr', 'car', 'auto', 'drive', 'passenger', 'vehicle', 'hov'],
    'bike':    ['bike', 'bicycle', 'cycle', 'ebike'],
}


def person_mode_counts(trips: pd.DataFrame, person_col: str = 'person_id') -> pd.DataFrame:
    """Count trips per person by coarse mode (transit / car / bike). Returns a
    frame indexed by person with n_transit / n_car / n_bike columns."""
    mode_col = next((c for c in ['mode', 'MAINMODE', 'TRAVMODE', 'LINKMODE', 'mode_name']
                     if c in trips.columns), None)
    if mode_col is None or person_col not in trips.columns:
        return pd.DataFrame(columns=['n_transit', 'n_car', 'n_bike'])
    m = trips[mode_col].astype(str).str.lower()
    coarse = pd.Series('other', index=trips.index)
    for cat, keys in _MODE_ALIASES.items():
        hit = m.str.contains('|'.join(keys), regex=True, na=False)
        coarse[hit & (coarse == 'other')] = cat
    g = pd.DataFrame({person_col: trips[person_col].values, 'coarse': coarse.values})
    counts = g.groupby([person_col, 'coarse']).size().unstack(fill_value=0)
    for c in ['transit', 'car', 'bike']:
        if c not in counts.columns:
            counts[c] = 0
    return counts.rename(columns={'transit': 'n_transit', 'car': 'n_car',
                                  'bike': 'n_bike'})[['n_transit', 'n_car', 'n_bike']]


def attach_observed_counts(persons: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Attach OBSERVED weekly n_transit/n_car/n_bike onto a persons frame (from its
    trips), so estimation data carries the same mobility-usage covariates that
    predict_counts assigns onto synth. Idempotent; 0-fills persons with no trips."""
    if persons is None or trips is None or not len(trips):
        return persons
    pid = next((c for c in ['person_id', 'PERSID', 'persid'] if c in persons.columns), None)
    if pid is None:
        return persons
    counts = person_mode_counts(trips, person_col=pid if pid in trips.columns else 'person_id')
    if not len(counts):
        return persons
    out = persons.drop(columns=[c for c in ['n_transit', 'n_car', 'n_bike']
                                if c in persons.columns], errors='ignore')
    out = out.merge(counts, left_on=pid, right_index=True, how='left')
    return out.fillna({'n_transit': 0, 'n_car': 0, 'n_bike': 0})


# ─────────────────────────────────────────────────────────────────────────────
# Model instantiations
# ─────────────────────────────────────────────────────────────────────────────
def _fit_zic(feats_fn, df, y, l2=1.0, negbin=True):
    X, names = feats_fn(df)
    m = ZeroInflatedCount(negbin=negbin, l2=l2)
    m.fit(X, X, np.asarray(y, float))
    m.gate_names_ = names
    m.count_names_ = names
    return m


def estimate_transit_usage(persons, y):  return _fit_zic(_person_features, persons, y)
def estimate_car_usage(persons, y):      return _fit_zic(_person_features, persons, y)
def estimate_bike_usage(persons, y):     return _fit_zic(_person_features, persons, y)
def estimate_household_vehicles(hh, y):  return _fit_zic(_household_features, hh, y)
def estimate_household_bikes(hh, y):     return _fit_zic(_household_features, hh, y)


def predict_counts(df, model: ZeroInflatedCount, feats_fn, colname: str,
                   seed: int = 42) -> np.ndarray:
    X, _ = feats_fn(df)
    return model.sample(X, X, np.random.default_rng(seed))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
# Mode-usage propensity column names: the smooth expected value (1-π)·μ, a
# function of EXOGENOUS person attributes only (not the realised trip count).
# Used as the non-circular mode-choice captivity index (person.<mode>_prop),
# computed identically on HHTS and synth.
_PROP_COL = {'n_transit': 'transit_prop', 'n_car': 'car_prop', 'n_bike': 'bike_prop'}


def fit_person_models(persons: pd.DataFrame) -> dict:
    """Fit the three person-level mode-usage models on a persons frame that
    already carries observed n_transit/n_car/n_bike (see attach_observed_counts).
    Returns {ycol: ZeroInflatedCount}. Used to attach propensities to the HHTS
    estimation frame in load_data without re-deriving counts from trips."""
    models: dict = {}
    if not _SCIPY_OK:
        return models
    for ycol, est in [('n_transit', estimate_transit_usage),
                      ('n_car', estimate_car_usage),
                      ('n_bike', estimate_bike_usage)]:
        if ycol not in persons.columns:
            continue
        try:
            y = pd.to_numeric(persons[ycol], errors='coerce').fillna(0).values
            models[ycol] = est(persons, y)
        except Exception as e:
            logger.warning(f"fit_person_models {ycol} failed: {e}")
    return models


def attach_propensities(persons: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Attach transit_prop/car_prop/bike_prop = model.predict_expected(person
    features) onto a persons frame. Smooth exogenous captivity indices — safe to
    use as mode-choice covariates without the endogeneity of the raw counts.
    Idempotent; a no-op for modes whose model is missing."""
    if persons is None or not models:
        return persons
    out = persons
    X = _person_features(out)[0]
    for ycol, prop in _PROP_COL.items():
        m = models.get(ycol)
        if m is None or m.beta_ is None:
            continue
        out = out.copy() if out is persons else out
        out[prop] = m.predict_expected(X, X)
    return out


def run_mobility_models(hhts_persons: pd.DataFrame, synth: pd.DataFrame,
                        hhts_trips: Optional[pd.DataFrame] = None,
                        households: Optional[pd.DataFrame] = None,
                        synth_households: Optional[pd.DataFrame] = None,
                        out_dir: Optional[str] = None, seed: int = 42,
                        return_models: bool = False):
    """Estimate the mobility models on HHTS, assign onto synth, write validation
    reports. Returns synth with n_transit/n_car/n_bike raw counts + transit_prop/
    car_prop/bike_prop propensities (+ hh_vehicles/hh_bikes). With
    return_models=True returns (synth, {ycol: ZeroInflatedCount}) so the caller
    can attach matching propensities to the HHTS estimation frame."""
    person_models: dict = {}
    all_models: dict = {}          # {name: ZeroInflatedCount} for persistence
    if not _SCIPY_OK:
        logger.warning("mobility_models: scipy unavailable — skipping")
        return (synth, person_models) if return_models else synth
    rep_dir = os.path.join(out_dir, 'mobility') if out_dir else None
    synth = synth.copy()

    # ── person-level usage (weekly trips by mode) ────────────────────────────
    if hhts_trips is not None and len(hhts_trips):
        counts = person_mode_counts(hhts_trips)
        pid = next((c for c in ['person_id', 'PERSID', 'persid'] if c in hhts_persons.columns), None)
        if pid is not None and len(counts):
            hp = hhts_persons.set_index(pid) if pid != hhts_persons.index.name else hhts_persons
            hp = hp.join(counts, how='left').fillna({'n_transit': 0, 'n_car': 0, 'n_bike': 0})
            specs = [('transit_usage', 'n_transit', estimate_transit_usage),
                     ('car_usage', 'n_car', estimate_car_usage),
                     ('bike_usage', 'n_bike', estimate_bike_usage)]
            Xs = _person_features(synth)[0]
            for name, ycol, est in specs:
                try:
                    m = est(hp, hp[ycol].values)
                    person_models[ycol] = m
                    all_models[name] = m
                    yp = m.predict_expected(*[_person_features(hp)[0]] * 2)
                    ys = m.sample(_person_features(hp)[0], _person_features(hp)[0],
                                  np.random.default_rng(seed))
                    validation_report(name, hp[ycol].values, yp, m, rep_dir, y_sampled=ys)
                    synth[ycol] = predict_counts(synth, m, _person_features, ycol, seed=seed)
                    synth[_PROP_COL[ycol]] = m.predict_expected(Xs, Xs)  # captivity index
                except Exception as e:
                    logger.warning(f"mobility {name} failed: {e}")
    else:
        logger.info("mobility_models: no HHTS trips — skipping person-level usage")

    # ── household-level ownership (vehicles + bikes) ─────────────────────────
    hh_src = households if households is not None else _persons_to_households(hhts_persons)
    hh_out = synth_households if synth_households is not None else _persons_to_households(synth)
    for name, ycol, est in [('household_vehicles', 'vehrd', estimate_household_vehicles),
                            ('household_bikes', 'n_bikes', estimate_household_bikes)]:
        if hh_src is None or ycol not in hh_src.columns:
            logger.info(f"mobility_models: no '{ycol}' in households — skipping {name}")
            continue
        try:
            m = est(hh_src, pd.to_numeric(hh_src[ycol], errors='coerce').fillna(0).values)
            all_models[name] = m
            Xv = _household_features(hh_src)[0]
            yp = m.predict_expected(Xv, Xv)
            ys = m.sample(Xv, Xv, np.random.default_rng(seed))
            validation_report(name, pd.to_numeric(hh_src[ycol], errors='coerce').fillna(0).values,
                              yp, m, rep_dir, y_sampled=ys)
            if hh_out is not None:
                synth_col = 'hh_vehicles_pred' if ycol == 'vehrd' else 'hh_bikes_pred'
                _assign = m.sample(_household_features(hh_out)[0],
                                   _household_features(hh_out)[0], np.random.default_rng(seed))
                hh_out[synth_col] = _assign
                # broadcast household prediction back onto persons if hh id present
                hhid = next((c for c in ['hh_id', 'HHID', 'household_id'] if c in synth.columns and c in hh_out.columns), None)
                if hhid is not None:
                    synth = synth.merge(hh_out[[hhid, synth_col]], on=hhid, how='left')
        except Exception as e:
            logger.warning(f"mobility {name} failed: {e}")

    # ── persist fitted models + a combined coefficients table ────────────────
    if rep_dir and all_models:
        save_mobility_models(all_models, rep_dir)
    return (synth, person_models) if return_models else synth


def save_mobility_models(models: dict, out_dir: str) -> None:
    """Pickle the fitted mobility models and write a combined coefficients CSV
    (gate γ + count β + NegBin dispersion per model), so the mobility models are
    inspectable/reusable like the stageN_models.pkl / cdap_model.pkl."""
    import pickle
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'mobility_models.pkl'), 'wb') as fh:
            pickle.dump(models, fh)
        rows = []
        for name, m in models.items():
            if getattr(m, 'beta_', None) is None:
                continue
            gate = ['const'] + list(m.gate_names_)
            cnt = ['const'] + list(m.count_names_)
            for nm, v in zip(gate, m.gamma_):
                rows.append({'model': name, 'part': 'gate(probit)', 'param': nm, 'value': float(v)})
            for nm, v in zip(cnt, m.beta_):
                rows.append({'model': name, 'part': 'count(log)', 'param': nm, 'value': float(v)})
            if m.negbin:
                rows.append({'model': name, 'part': 'dispersion', 'param': 'log_alpha',
                             'value': float(m.log_alpha_)})
            _bic = (m.k_ * np.log(max(m.n_, 1)) - 2 * m.ll_) if np.isfinite(m.ll_) else np.nan
            rows.append({'model': name, 'part': 'fit', 'param': 'loglik', 'value': float(m.ll_)})
            rows.append({'model': name, 'part': 'fit', 'param': 'bic', 'value': float(_bic)})
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, 'mobility_coefficients.csv'), index=False)
        logger.info(f"mobility models saved → {out_dir}/mobility_models.pkl "
                    f"(+ mobility_coefficients.csv, {len(models)} models)")
    except Exception as e:
        logger.warning(f"save_mobility_models failed: {e}")


def _persons_to_households(persons: Optional[pd.DataFrame]):
    if persons is None:
        return None
    hhid = next((c for c in ['hh_id', 'HHID', 'household_id'] if c in persons.columns), None)
    if hhid is None:
        return None
    p = persons.copy()
    if 'WORKSTAT' in p.columns:
        p['_is_worker'] = pd.to_numeric(p['WORKSTAT'], errors='coerce').fillna(0).isin([1, 2]).astype(int)
    if 'age5p' in p.columns:
        p['_is_adult'] = (pd.to_numeric(p['age5p'], errors='coerce').fillna(8) >= 4).astype(int)
    g = p.groupby(hhid)
    hh = pd.DataFrame({
        hhid: list(g.groups.keys()),
        'nprd': g.size().values,
        'incp': g['incp'].first().values if 'incp' in p.columns else 7.0,
        'vehrd': g['vehrd'].first().values if 'vehrd' in p.columns else np.nan,
        'n_workers': g['_is_worker'].sum().values if '_is_worker' in p.columns else 1.0,
        'n_adults': g['_is_adult'].sum().values if '_is_adult' in p.columns else 2.0,
    })
    return hh


if __name__ == '__main__':
    # Self-check: synthetic excess-zero trip counts recovered by the ZIP model.
    rng = np.random.default_rng(0)
    n = 4000
    persons = pd.DataFrame({
        'incp': rng.integers(1, 14, n), 'vehrd': rng.integers(0, 4, n),
        'nprd': rng.integers(1, 6, n), 'age5p': rng.integers(4, 16, n),
        'sexp': rng.integers(1, 3, n), 'CARLICENCE': rng.integers(0, 2, n),
        'WORKSTAT': rng.choice([0, 1, 2], n, p=[0.3, 0.55, 0.15]),
    })
    # true transit: structural non-user if many vehicles / has licence; else count
    X, _ = _person_features(persons)
    pi = _PHI(-0.2 + 0.8 * persons['vehrd'].values - 0.5 * persons['incp'].values / 13)
    mu = np.exp(0.5 - 0.6 * persons['vehrd'].values / 3 + 0.4 * (persons['incp'].values < 5))
    struct0 = rng.random(n) < np.clip(pi, 0, 1)
    y = np.where(struct0, 0, rng.poisson(mu))
    m = estimate_transit_usage(persons, y)
    yp = m.predict_expected(X, X)
    ys = m.sample(X, X, rng)
    rep = validation_report('transit_usage_selfcheck', y, yp, m, None, y_sampled=ys)
    print(f"obs_mean={rep['obs_mean']:.2f} pred_mean={rep['pred_mean']:.2f} "
          f"obs_zero={rep['obs_zero_share']:.2f} pred_zero={rep['pred_zero_share']:.2f} "
          f"BIC={rep['bic']:.0f}")
