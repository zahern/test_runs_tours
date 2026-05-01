"""
stage2_day_duration_budget.py
==============================
Stage 2 of the activity-based model pipeline.

Predicts three things per person:
  - start_of_day  : earliest departure time (minutes from midnight)
  - end_of_day    : latest arrival home (minutes from midnight)
  - B             : total out-of-home time budget (hours)

These constrain all downstream duration and scheduling models.

Pipeline position
-----------------
    Stage 1: activity frequency  â†’  n_{purpose}_out per person
        â†’ Stage 2: day duration budget       â† YOU ARE HERE
        â†’ Stage 3: activity duration
        â†’ Stage 4: schedule placement
        â†’ Stage 5: destination choice
        â†’ Stage 6: mode choice

Model structure
---------------
    start_of_day : OLS / lognormal regression
                   predictors: ptype, activity mix, vehicles, income
    end_of_day   : OLS regression
                   predictors: ptype, start_of_day, n_activities, activity mix
    B (budget)   : end_of_day - start_of_day, clipped to plausible range
                   (a separate lognormal is fitted for direct prediction
                    when end_of_day is unavailable)

Usage â€” estimate + predict
---------------------------
    from stage2_day_duration import run_stage2

    out        = Synth_Connect().process_survey(save=False)
    synth_s1   = pd.read_csv('saved_data/stage1_output.csv')

    synth_s2, models = run_stage2(out["tours"], out["persons"], synth_s1)
    synth_s2.to_csv(os.path.join(_OUT, 'stage2_output.csv'), index=False)
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
from typing import Optional
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Activity count columns produced by Stage 1
ACTIVITY_COLS = [
    'n_work_out', 'n_edu_out', 'n_social_out', 'n_escort_out',
    'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out',
]

# Person-level predictors
PERSON_FEATURES = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp']

# Plausible day window â€” TRAINING and PREDICTION bounds (minutes from midnight)
# After filtering bad data, the HHTS shows:
#   Workers:    start ~420-540 (7-9am),  end ~960-1080 (4-6pm)
#   Students:   start ~420-480 (7-8am),  end ~840-960  (2-4pm)
#   Non-workers:start ~480-600 (8-10am), end ~900-1080 (3-6pm)
MIN_START  = 300    #  5:00 am  â€” hard floor for prediction clip
MAX_START  = 720    # 12:00 pm  â€” hard ceiling for prediction clip
MIN_END    = 660    # 11:00 am  -- opened from noon to 11am
MAX_END    = 1440   # midnight  -- extended from 11pm
MIN_BUDGET = 0.5    # 30 minutes minimum out-of-home (hours)
MAX_BUDGET = 16.0   # 16 hours maximum (hours)

COEF_CLIP_LO = -50.0
COEF_CLIP_HI = 50.0
INTERCEPT_CLIP_LO = -1440.0   # full day in minutes
INTERCEPT_CLIP_HI =  1440.0   # was 1000.0 — was capping end_of_day at 1000 min (16:40)


def _quantile_profile(values: pd.Series, q=None) -> dict:
    """Compact empirical distribution summary for rank-based remapping."""
    if q is None:
        q = np.linspace(0.001, 0.999, 199)
    s = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {'q': np.asarray(q, dtype=float), 'v': np.zeros(len(q), dtype=float)}
    return {'q': np.asarray(q, dtype=float), 'v': np.quantile(s.values.astype(float), q)}


def _apply_quantile_profile(values: np.ndarray, profile: dict) -> np.ndarray:
    """
    Monotonic rank remapping so the synthetic marginal distribution follows
    the survey more closely without destroying person-to-person ordering.
    """
    arr = np.asarray(values, dtype=float)
    if profile is None or len(arr) == 0:
        return arr
    q = np.asarray(profile.get('q', []), dtype=float)
    v = np.asarray(profile.get('v', []), dtype=float)
    if q.size == 0 or v.size == 0:
        return arr
    order = np.argsort(arr, kind='mergesort')
    ranks = (np.arange(len(arr), dtype=float) + 0.5) / max(len(arr), 1)
    mapped_sorted = np.interp(ranks, q, v, left=v[0], right=v[-1])
    out = np.empty_like(arr, dtype=float)
    out[order] = mapped_sorted
    return out


# ---------------------------------------------------------------------------
# 0.  AGGREGATE SURVEY DATA TO PERSON-DAY
# ---------------------------------------------------------------------------

def aggregate_person_day(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per person with:
      start_of_day, end_of_day, B,
      activity counts (n_*_out),
      person attributes.

    Parameters
    ----------
    tours   : from process_survey()["tours"]
    persons : from process_survey()["persons"]

    Returns
    -------
    One-row-per-person DataFrame ready for model estimation.
    """
    df = tours.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # â”€â”€ resolve time columns to MINUTES from midnight â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # build_as_tours produces two sets of time columns:
#   start_time / end_time  : raw HHTS minutes (e.g. 480 = 8:00am)
#   STARTIME   / ENDTIME   : (start_time / 60) stored as integer hours
    # We prefer start_time/end_time (minutes). If absent, derive from
    # STARTIME/ENDTIME by multiplying by 60.

    if 'start_time' not in df.columns:
        if 'STARTIME' in df.columns:
            # STARTIME is hours (e.g. 8 means 8:00am = 480 min)
            df['start_time'] = df['STARTIME'] * 60
            logger.info('Derived start_time from STARTIME*60')
        else:
            raise KeyError('tours table has neither start_time nor STARTIME')

    if 'end_time' not in df.columns:
        if 'ENDTIME' in df.columns:
            df['end_time'] = df['ENDTIME'] * 60
            logger.info('Derived end_time from ENDTIME*60')
        else:
            raise KeyError('tours table has neither end_time nor ENDTIME')

    # final unit check â€” if median is < 25 it's still hours
    for col in ['start_time', 'end_time']:
        med = df[col].dropna().median()
        if med < 25:
            df[col] = df[col] * 60
            logger.warning(f'{col} median={med:.1f} â€” multiplied by 60 to get minutes')
        elif med > 1440:
            df[col] = df[col] / 60
            logger.warning(f'{col} median={med:.1f} â€” divided by 60 to get minutes')

    logger.info(
        f'Time columns in minutes: '
        f'start_time mean={df["start_time"].mean():.0f} '
        f'end_time mean={df["end_time"].mean():.0f}'
    )

    # duration in minutes â€” build_as_tours stores this as end_time - start_time
    if 'duration' not in df.columns:
        df['duration'] = df['end_time'] - df['start_time']
    dur_med = df['duration'].dropna().median()
    if dur_med < 5:      # stored as hours
        df['duration'] = df['duration'] * 60
    elif dur_med > 86400:  # stored as seconds
        df['duration'] = df['duration'] / 60
    df['duration_hrs'] = (df['duration'] / 60).clip(lower=0.05)

    df['start_time_hour'] = df['start_time'] / 60
    df['end_time_hour']   = df['end_time']   / 60

    # â”€â”€ remove implausible times before aggregation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Trips starting at midnight or before 4am are usually data issues
    # (overnight trips recorded on wrong day, missing data coded as 0)
    # Only keep trips with plausible departure times for training.
    TRIP_START_MIN = 240    # 4:00 am â€” earliest plausible first departure
    TRIP_END_MAX   = 1380   # 11:00 pm â€” latest plausible return home
    n_before = len(df)
    df = df[
        (df['start_time'] >= TRIP_START_MIN) &
        (df['end_time']   <= TRIP_END_MAX)   &
        (df['end_time']   >  df['start_time'])
    ].copy()
    n_after = len(df)
    if n_before != n_after:
        logger.info(
            f'Filtered {n_before - n_after} implausible tour records '
            f'(start<4am or end>11pm or end<=start). '
            f'{n_after} remain.'
        )

    # build agg dict
    agg = {
        'start_of_day': ('start_time', 'min'),   # minutes from midnight
        'end_of_day':   ('end_time',   'max'),   # minutes from midnight
        'B':            ('duration_hrs', 'sum'),  # hours
        'n_tours':      ('tour_id', 'max'),
    }
    for col in ACTIVITY_COLS:
        if col in df.columns:
            agg[col] = (col, 'sum')
        else:
            df[col] = 0
            agg[col] = (col, 'sum')

    person_day = df.groupby('person_id').agg(**agg).reset_index()

    # clip person-day times to plausible range
    # These define the TRAINING range â€” the OLS learns within these bounds
    person_day['start_of_day'] = person_day['start_of_day'].clip(
        lower=300,   # 5:00am  â€” earliest realistic first departure
        upper=720,   # 12:00pm â€” latest realistic first departure
    )
    person_day['end_of_day'] = person_day['end_of_day'].clip(
        lower=600,   # 10:00am â€” minimum realistic return (short trips)
        upper=1380,  # 11:00pm â€” maximum realistic return
    )
    # ensure end > start
    person_day['end_of_day'] = person_day[['start_of_day','end_of_day']].max(axis=1) + 60
    person_day['B'] = person_day['B'].clip(lower=MIN_BUDGET, upper=MAX_BUDGET)

    # merge person attributes
    pers = persons.copy()
    if 'person_id' not in pers.columns and 'PERSID' in pers.columns:
        pers = pers.rename(columns={'PERSID': 'person_id'})

    # import ptype assignment
    try:
        from stage1_activity_frequency import assign_ptype_column
        pers = assign_ptype_column(pers)
    except ImportError:
        pers['ptype'] = 'NonWorker'

    keep = ['person_id', 'ptype'] + [c for c in PERSON_FEATURES if c in pers.columns]
    person_day = person_day.merge(pers[keep], on='person_id', how='left')

    # fill missing person features
    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2,
                'STUDYING': 0, 'WORKSTAT': 0, 'incp': 8}
    for col, val in defaults.items():
        if col not in person_day.columns:
            person_day[col] = val
        person_day[col] = person_day[col].fillna(val)

    # total activity count
    person_day['n_total'] = person_day[
        [c for c in ACTIVITY_COLS if c in person_day.columns]
    ].sum(axis=1)

    logger.info(f"Person-day dataset: {len(person_day)} persons")
    return person_day


# ---------------------------------------------------------------------------
# 1.  MODELS  (Tobit censored regression + lognormal budget)
# ---------------------------------------------------------------------------

class TobitModel:
    """
    Tobit Type-I censored regression for bounded continuous outcomes.

    Appropriate for start_of_day (left-censored at MIN_START) and
    end_of_day (right-censored at MAX_END).
    """

    def __init__(self, lo: float = -np.inf, hi: float = np.inf,
                 log_y: bool = False, ridge: float = 0.01):
        self.lo = lo
        self.hi = hi
        self.log_y = log_y
        self.ridge = ridge   # L2 penalty on slope coefficients (not intercept)
        self.coef_ = None
        self.intercept_ = None
        self.log_sigma_ = None
        self.sigma_ = None
        self.mu_ = None
        self.sd_ = None
        self.feature_names_ = None
        self.rmse_ = None
        self.ll_ = None
        self.std_err_ = None
        self.t_stat_ = None
        self.p_value_ = None
        self.n_ = None
        self.converged_ = None

    def _clip_coefficients(self):
        if self.coef_ is not None:
            self.coef_ = np.clip(self.coef_, COEF_CLIP_LO, COEF_CLIP_HI)
        if self.intercept_ is not None:
            self.intercept_ = float(np.clip(self.intercept_, INTERCEPT_CLIP_LO, INTERCEPT_CLIP_HI))

    def _transform(self, y: np.ndarray) -> np.ndarray:
        return np.log(np.clip(y, 1e-8, None)) if self.log_y else y.copy()

    def _neg_ll(self, params: np.ndarray, Xs: np.ndarray, y: np.ndarray) -> float:
        n_feat = Xs.shape[1] - 1
        intercept = params[0]
        beta = params[1:n_feat + 1]
        log_sigma = params[n_feat + 1]
        sigma = np.exp(log_sigma)

        eta = intercept + Xs[:, 1:] @ beta
        yt = self._transform(y)
        lo_t = self._transform(np.full_like(y, self.lo)) if np.isfinite(self.lo) else None
        hi_t = self._transform(np.full_like(y, self.hi)) if np.isfinite(self.hi) else None

        ll = 0.0
        uncens = (y > self.lo) & (y < self.hi)
        if uncens.any():
            resid = yt[uncens] - eta[uncens]
            ll += (-np.log(sigma) - 0.5 * (resid / sigma) ** 2).sum()
            if self.log_y:
                ll -= yt[uncens].sum()

        if lo_t is not None:
            left = y <= self.lo
            if left.any():
                z = (lo_t[left] - eta[left]) / sigma
                ll += np.log(np.clip(norm.cdf(z), 1e-10, None)).sum()

        if hi_t is not None:
            right = y >= self.hi
            if right.any():
                z = (hi_t[right] - eta[right]) / sigma
                ll += np.log(np.clip(1.0 - norm.cdf(z), 1e-10, None)).sum()

        neg_ll = -float(ll)
        # L2 ridge penalty on slope coefficients only (not intercept or log_sigma)
        # Prevents multicollinearity from sending coefficients to ±∞.
        if self.ridge > 0:
            neg_ll += self.ridge * float(np.sum(params[1:n_feat + 1] ** 2))
        return neg_ll

    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 600):
        from scipy.optimize import minimize

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])

        yt = self._transform(y)
        ols = np.linalg.lstsq(Xs, yt, rcond=None)[0]
        resid_sigma = max(np.std(yt - Xs @ ols), 1e-3)
        init = np.concatenate([ols, [np.log(resid_sigma)]])

        n_feat = X.shape[1]

        # L-BFGS-B bounds prevent the intercept and log_sigma from running away.
        # The intercept (in original-y units, since features are standardised)
        # should stay within the plausible outcome range [lo, hi].
        # log_sigma is bounded so sigma ∈ [e^-4, e^5] ≈ [0.018, 148].
        intercept_lo = float(self.lo) if np.isfinite(self.lo) else -1440.0
        intercept_hi = float(self.hi) if np.isfinite(self.hi) else  1440.0
        opt_bounds = (
            [(intercept_lo, intercept_hi)]          # intercept
            + [(-COEF_CLIP_HI, COEF_CLIP_HI)] * n_feat   # slope coefficients
            + [(-4.0, 5.0)]                         # log_sigma
        )
        res = minimize(
            self._neg_ll, init, args=(Xs, y),
            method='L-BFGS-B',
            bounds=opt_bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10},
        )
        self.intercept_ = res.x[0]
        self.coef_ = res.x[1:n_feat + 1]
        self.log_sigma_ = res.x[n_feat + 1]
        self.sigma_ = float(np.exp(self.log_sigma_))
        self.ll_ = float(-res.fun)
        self.n_ = len(y)
        self.converged_ = bool(res.success)
        self._clip_coefficients()

        try:
            hess_inv = res.hess_inv.todense() if hasattr(res.hess_inv, 'todense') else np.asarray(res.hess_inv)
            hess_inv = np.asarray(hess_inv, dtype=float)
            params = np.concatenate([[self.intercept_], self.coef_, [self.log_sigma_]])
            self.std_err_ = np.sqrt(np.clip(np.diag(hess_inv), 0.0, None))
            self.t_stat_ = np.divide(params, self.std_err_, out=np.full_like(params, np.nan), where=self.std_err_ > 0)
            self.p_value_ = 2.0 * norm.sf(np.abs(self.t_stat_))
        except Exception:
            size = n_feat + 2
            self.std_err_ = np.full(size, np.nan)
            self.t_stat_ = np.full(size, np.nan)
            self.p_value_ = np.full(size, np.nan)

        eta = self.intercept_ + (X - self.mu_) / self.sd_ @ self.coef_
        yhat = np.exp(eta + 0.5 * self.sigma_ ** 2) if self.log_y else eta
        yhat = np.clip(yhat, self.lo if np.isfinite(self.lo) else -1e9,
                       self.hi if np.isfinite(self.hi) else 1e9)
        self.rmse_ = float(np.sqrt(np.mean((y - yhat) ** 2)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu_) / self.sd_
        eta = self.intercept_ + Xs @ self.coef_
        yhat = np.exp(eta + 0.5 * self.sigma_ ** 2) if self.log_y else eta
        return np.clip(
            yhat,
            self.lo if np.isfinite(self.lo) else -1e9,
            self.hi if np.isfinite(self.hi) else 1e9,
        )

    def predict_with_noise(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        yhat = self.predict(X)
        if self.log_y:
            noisy = np.exp(rng.normal(np.log(np.clip(yhat, 1e-8, None)), self.sigma_))
        else:
            noisy = yhat + rng.normal(0, self.sigma_, size=len(X))
        return np.clip(
            noisy,
            self.lo if np.isfinite(self.lo) else -1e9,
            self.hi if np.isfinite(self.hi) else 1e9,
        )


class LinearModel:
    """
    OLS linear regression with optional log transform on y.
    Pure numpy/scipy summary statistics only; no sklearn.
    """

    def __init__(self, log_y: bool = False):
        self.log_y = log_y
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None
        self.mu_ = None
        self.sd_ = None
        self.rmse_ = None
        self.sigma_ = None
        self.ll_ = None
        self.std_err_ = None
        self.t_stat_ = None
        self.p_value_ = None
        self.n_ = None
        self.converged_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])
        y_fit = np.log(np.clip(y, 1e-6, None)) if self.log_y else y.copy()
        try:
            params = np.linalg.lstsq(Xs, y_fit, rcond=None)[0]
        except np.linalg.LinAlgError:
            params = np.zeros(Xs.shape[1])
        self.intercept_ = params[0]
        self.coef_ = params[1:]
        resid = y_fit - Xs @ params
        self.sigma_ = float(np.sqrt(np.mean(resid ** 2)))
        self.n_ = len(y)
        self.converged_ = True
        self._clip_coefficients()

        dof = max(len(y) - Xs.shape[1], 1)
        sigma2 = float((resid @ resid) / dof)
        xtx_inv = np.linalg.pinv(Xs.T @ Xs)
        self.std_err_ = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0.0, None))
        params_full = np.concatenate([[self.intercept_], self.coef_])
        self.t_stat_ = np.divide(params_full, self.std_err_, out=np.full_like(params_full, np.nan), where=self.std_err_ > 0)
        self.p_value_ = 2.0 * norm.sf(np.abs(self.t_stat_))
        self.ll_ = float(-0.5 * len(y) * (np.log(2 * np.pi * sigma2) + 1.0))
        self.rmse_ = float(np.sqrt(np.mean((y - self.predict(X)) ** 2)))
        return self

    def _clip_coefficients(self):
        if self.coef_ is not None:
            self.coef_ = np.clip(self.coef_, COEF_CLIP_LO, COEF_CLIP_HI)
        if self.intercept_ is not None:
            self.intercept_ = float(np.clip(self.intercept_, INTERCEPT_CLIP_LO, INTERCEPT_CLIP_HI))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != len(self.mu_):
            raise ValueError(
                f"Feature mismatch: model expects {len(self.mu_)} columns "
                f"but received {X.shape[1]}. "
                f"Model features: {self.feature_names_}. "
                f"Delete saved_data/stage2_models.pkl and re-run to retrain."
            )
        Xs = (X - self.mu_) / self.sd_
        yhat = self.intercept_ + Xs @ self.coef_
        if self.log_y:
            yhat = np.exp(yhat + 0.5 * self.sigma_ ** 2)
        return yhat

    def predict_with_noise(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        yhat = self.predict(X)
        if self.log_y:
            return np.exp(rng.normal(np.log(np.clip(yhat, 1e-6, None)), self.sigma_))
        return yhat + rng.normal(0, self.sigma_, size=len(X))


# ---------------------------------------------------------------------------
# 2.  BUILD FEATURE MATRIX
# ---------------------------------------------------------------------------


def _augment_day_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    if 'n_total' not in df.columns:
        df['n_total'] = df[ACTIVITY_COLS].sum(axis=1)
    else:
        df['n_total'] = pd.to_numeric(df['n_total'], errors='coerce').fillna(0.0)

    if 'n_tours' not in df.columns:
        df['n_tours'] = df['n_total'].clip(lower=1.0)
    else:
        df['n_tours'] = pd.to_numeric(df['n_tours'], errors='coerce').fillna(df['n_total']).clip(lower=1.0)

    df['n_mandatory'] = df.get('work', 0.0) + df.get('edu', 0.0)
    df['n_discretionary'] = df['n_total'] - df['n_mandatory']
    df['mandatory_share'] = np.divide(df['n_mandatory'], df['n_total'].clip(lower=1.0))
    df['escort_share'] = np.divide(df.get('escort', 0.0), df['n_total'].clip(lower=1.0))
    df['vehs_per_person'] = np.divide(
        pd.to_numeric(df.get('vehrd', 0.0), errors='coerce').fillna(0.0),
        pd.to_numeric(df.get('nprd', 1.0), errors='coerce').fillna(1.0).clip(lower=1.0)
    )
    df['income_per_person'] = np.divide(
        pd.to_numeric(df.get('incp', 0.0), errors='coerce').fillna(0.0),
        pd.to_numeric(df.get('nprd', 1.0), errors='coerce').fillna(1.0).clip(lower=1.0)
    )
    return df

def _build_X(
    df: pd.DataFrame,
    feature_cols: list,
    ptype_dummies: Optional[list] = None,
) -> tuple:
    """
    Build feature matrix from df. Returns (X, actual_cols).
    If ptype_dummies list given, one-hot encode ptype and append.
    """
    cols = []
    arrays = []

    for col in feature_cols:
        if col in df.columns:
            arrays.append(df[col].fillna(0).values.astype(float))
            cols.append(col)

    if ptype_dummies is not None and 'ptype' in df.columns:
        for pt in ptype_dummies:
            arrays.append((df['ptype'] == pt).astype(float).values)
            cols.append(f'ptype_{pt}')

    if not arrays:
        return np.ones((len(df), 1)), ['const']

    return np.column_stack(arrays), cols


# ---------------------------------------------------------------------------
# 3.  ESTIMATE MODELS
# ---------------------------------------------------------------------------

def estimate_day_models(person_day: pd.DataFrame,
                        feature_override: list = None) -> dict:
    """
    Fit three models:
      'start'  : predict start_of_day (minutes)
      'end'    : predict end_of_day   (minutes)
      'budget' : predict B (hours) directly â€” used as fallback

    Parameters
    ----------
    person_day : from aggregate_person_day()

    Returns
    -------
    dict with keys 'start', 'end', 'budget', each containing a
    LinearModel and the feature column list used.
    """
    person_day = _augment_day_features(person_day)
    ptypes = sorted(person_day['ptype'].dropna().unique())

    # â”€â”€ diagnostic: confirm training targets are in minutes â”€â”€â”€â”€â”€â”€
    for col in ['start_of_day', 'end_of_day']:
        if col in person_day.columns:
            med = person_day[col].median()
            mn  = person_day[col].min()
            mx  = person_day[col].max()
            def _fmt(v): h=int(v//60); m=int(v%60); return f'{h}:{m:02d} ({v:.0f}m)'
            print(f'  Training {col}: min={_fmt(mn)} med={_fmt(med)} max={_fmt(mx)}')
            if med < 25:
                logger.error(
                    f'{col} median={med:.1f} â€” still in hours! '
                    f'aggregate_person_day unit fix did not apply. '
                    f'Check that tours has start_time or STARTIME columns.'
                )

    # â”€â”€ start_of_day â€” Tobit left-censored at MIN_START â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # OLS on clipped data attenuates coefficients because obs censored at
    # MIN_START are treated as if that was their true value.  Tobit models
    # the latent departure time correctly.
    _pf = feature_override if feature_override else PERSON_FEATURES
    derived_feats = ['n_mandatory', 'n_discretionary', 'mandatory_share', 'escort_share', 'vehs_per_person', 'income_per_person']
    start_feats = ACTIVITY_COLS + derived_feats + _pf
    X_start, start_cols = _build_X(person_day, start_feats, ptypes)
    y_start = person_day['start_of_day'].clip(MIN_START, MAX_START).values
    m_start = TobitModel(lo=MIN_START, hi=MAX_START, log_y=False)
    m_start.fit(X_start, y_start)
    m_start.feature_names_ = start_cols
    logger.info(f"start_of_day Tobit  RMSE={m_start.rmse_:.1f} min  LL={m_start.ll_:.1f}")

    # â”€â”€ end_of_day â€” Tobit right-censored at MAX_END â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    end_feats = ACTIVITY_COLS + derived_feats + _pf + ['n_total', 'n_tours']
    X_end, end_cols = _build_X(person_day, end_feats, ptypes)

    # add predicted start as a feature (sequential dependency)
    start_pred = m_start.predict(X_start)
    X_end = np.column_stack([X_end, start_pred])
    end_cols = end_cols + ['start_pred']

    y_end = person_day['end_of_day'].clip(MIN_END, MAX_END).values

    m_end = TobitModel(lo=MIN_END, hi=MAX_END, log_y=False, ridge=0.05)
    m_end.fit(X_end, y_end, maxiter=2000)
    m_end.feature_names_ = end_cols
    logger.info(f"end_of_day Tobit    RMSE={m_end.rmse_:.1f} min  LL={m_end.ll_:.1f}")

    # â”€â”€ budget B â€” lognormal OLS (soft boundaries, no censoring) â”€â”€â”€â”€â”€â”€
    budget_feats = ACTIVITY_COLS + derived_feats + _pf + ['n_total', 'n_tours']
    X_budget, budget_cols = _build_X(person_day, budget_feats, ptypes)
    y_budget = person_day['B'].values

    m_budget = LinearModel(log_y=True)
    m_budget.fit(X_budget, y_budget)
    m_budget.feature_names_ = budget_cols
    logger.info(f"budget lognormal    RMSE={m_budget.rmse_:.3f} hrs")

    print(
        f"Stage 2 models fitted on {len(person_day)} persons\n"
        f"  start Tobit  RMSE : {m_start.rmse_:.1f} min  LL={m_start.ll_:.1f}\n"
        f"  end   Tobit  RMSE : {m_end.rmse_:.1f} min  LL={m_end.ll_:.1f}\n"
        f"  budget lognormal RMSE: {m_budget.rmse_:.3f} hrs"
    )

    return {
        'start':  {
            'model': m_start,
            'features': start_cols,
            'ptypes': ptypes,
            'observed_profile': _quantile_profile(person_day['start_of_day']),
        },
        'end':    {
            'model': m_end,
            'features': end_cols,
            'ptypes': ptypes,
            'observed_profile': _quantile_profile(person_day['end_of_day']),
        },
        'budget': {
            'model': m_budget,
            'features': budget_cols,
            'ptypes': ptypes,
            'observed_profile': _quantile_profile(person_day['B']),
        },
    }


# ---------------------------------------------------------------------------
# 4.  PREDICT â€” SYNTHETIC POPULATION
# ---------------------------------------------------------------------------

def predict_day_budget(
    synth: pd.DataFrame,
    models: dict,
    sample: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Predict start_of_day, end_of_day, and B for a synthetic population.

    Requires synth to have:
      - n_{purpose}_out columns  (from Stage 1)
      - person attributes: sexp, vehrd, nprd, WORKSTAT, STUDYING, incp
      - ptype column (or will be derived)

    Parameters
    ----------
    synth   : synthetic population DataFrame (Stage 1 output)
    models  : output of estimate_day_models()
    sample  : add residual noise to predictions (recommended)
    seed    : random seed

    Returns
    -------
    synth copy with start_of_day, end_of_day, B columns added.
    """
    df  = _augment_day_features(synth.copy())
    rng = np.random.default_rng(seed)

    # ensure person_id
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # ptype
    if 'ptype' not in df.columns:
        try:
            from stage1_activity_frequency import assign_ptype_column
            df = assign_ptype_column(df)
        except ImportError:
            df['ptype'] = 'NonWorker'

    # add ptype dummies matching EXACTLY what the model was trained on
    # (ptype_X columns already in feature_names_ â€” pre-compute them here)
    ptypes = models['start']['ptypes']
    for pt in ptypes:
        df[f'ptype_{pt}'] = (df['ptype'] == pt).astype(float)

    # total activities
    act_cols_present = [c for c in ACTIVITY_COLS if c in df.columns]
    df['n_total'] = df[act_cols_present].fillna(0).sum(axis=1)

    # fill n_tours if absent
    if 'n_tours' not in df.columns:
        df['n_tours'] = df['n_total'].clip(lower=1)

    def _build_X_exact(df, model_obj):
        """
        Build feature matrix using EXACTLY the columns stored in
        model.feature_names_. Fills missing columns with 0.
        This guarantees the shape matches model.mu_ / model.sd_.
        """
        cols = model_obj.feature_names_
        arrays = []
        for col in cols:
            if col in df.columns:
                arrays.append(df[col].fillna(0).values.astype(float))
            else:
                arrays.append(np.zeros(len(df), dtype=float))
        return np.column_stack(arrays) if arrays else np.ones((len(df), 1))

    # â”€â”€ predict start_of_day â”€â”€
    X_start = _build_X_exact(df, models['start']['model'])

    if sample:
        start_pred = models['start']['model'].predict_with_noise(X_start, rng)
    else:
        start_pred = models['start']['model'].predict(X_start)

    start_pred = np.clip(start_pred, MIN_START, MAX_START)

    # â”€â”€ sanity check: if predictions look like hours not minutes, rescale
    if np.median(start_pred) < 25:
        logger.warning(
            f'start_of_day median={np.median(start_pred):.1f} looks like hours '
            f'â€” rescaling to minutes'
        )
        start_pred = start_pred * 60
        start_pred = np.clip(start_pred, MIN_START, MAX_START)
    start_pred = _apply_quantile_profile(
        start_pred,
        models.get('start', {}).get('observed_profile')
    )
    start_pred = np.clip(start_pred, MIN_START, MAX_START)

    # â”€â”€ predict end_of_day â”€â”€
    # end model was trained with start_pred as the last column
    # rebuild X_end using exact feature names minus 'start_pred', then append
    end_model = models['end']['model']
    end_feats_no_start = [c for c in end_model.feature_names_ if c != 'start_pred']

    # build a temp model-like object with trimmed feature_names_ for _build_X_exact
    class _TempModel:
        def __init__(self, names): self.feature_names_ = names
    X_end_base = _build_X_exact(df, _TempModel(end_feats_no_start))
    X_end = np.column_stack([X_end_base, start_pred])

    if sample:
        end_pred = models['end']['model'].predict_with_noise(X_end, rng)
    else:
        end_pred = models['end']['model'].predict(X_end)

    if np.median(end_pred) < 25:
        end_pred = end_pred * 60

    end_pred = np.clip(end_pred, MIN_END, MAX_END)
    end_pred = _apply_quantile_profile(
        end_pred,
        models.get('end', {}).get('observed_profile')
    )
    end_pred = np.clip(end_pred, MIN_END, MAX_END)

    # ensure end > start + minimum plausible out-of-home time (60 min)
    end_pred = np.maximum(end_pred, start_pred + 60)
    # and end must not exceed midnight
    end_pred = np.minimum(end_pred, MAX_END)

    # â”€â”€ budget B (hours) â”€â”€
    B_direct = (end_pred - start_pred) / 60

    X_budget = _build_X_exact(df, models['budget']['model'])
    B_model = models['budget']['model'].predict(X_budget)
    B_model = np.clip(B_model, MIN_BUDGET, MAX_BUDGET)

    # Lean primarily on the start/end implied budget, then remap to the
    # observed HHTS distribution to avoid hard-bound pileups at 16 hours.
    B_final = 0.85 * B_direct + 0.15 * B_model
    B_final = _apply_quantile_profile(
        B_final,
        models.get('budget', {}).get('observed_profile')
    )
    B_final = np.clip(B_final, MIN_BUDGET, MAX_BUDGET)

    df['start_of_day'] = np.round(start_pred).astype(int)  # minutes
    df['end_of_day']   = np.round(end_pred).astype(int)    # minutes
    df['B']            = B_final                            # hours

    # â”€â”€ readable summary â”€â”€
    def _fmt(mins):
        h = int(mins // 60); m = int(mins % 60)
        return f'{h}:{m:02d} ({mins:.0f} min)'

    print(
        f'Stage 2 predictions: {len(df)} persons\n'
        f"  Mean start_of_day : {_fmt(df['start_of_day'].mean())}\n"
        f"  Mean end_of_day   : {_fmt(df['end_of_day'].mean())}\n"
        f"  Mean budget B     : {df['B'].mean():.2f} hrs\n"
        f"  start_of_day range: "
        f"{_fmt(df['start_of_day'].min())} - {_fmt(df['start_of_day'].max())}  "
        f"(model bounds: floor={MIN_START//60}:00am, ceiling={MAX_START//60}:00pm; "
        f"{int((df['start_of_day'] <= MIN_START).sum())} persons at floor, "
        f"{int((df['start_of_day'] >= MAX_START).sum())} at ceiling)"
    )

    return df


# ---------------------------------------------------------------------------
# 5.  SAVE / LOAD
# ---------------------------------------------------------------------------


def _write_model_html(models: dict, path: str, stage_name: str, notes: str = "") -> None:
    """Write a human-readable HTML summary of fitted model coefficients."""
    import os as _os, html as _html
    _os.makedirs(_os.path.dirname(path) if _os.path.dirname(path) else '.', exist_ok=True)
    rows_html = []
    for key1, val1 in models.items():
        if isinstance(val1, dict):
            for key2, entry in val1.items():
                if isinstance(entry, dict):
                    _emit_model_entry(rows_html, entry, f"{key1}/{key2}")
                else:
                    rows_html.append(
                        f"<tr><td>{_html.escape(str(key1))}</td>"
                        f"<td>{_html.escape(str(key2))}</td>"
                        f"<td colspan=4>{_html.escape(str(entry)[:200])}</td></tr>")
    body = "\n".join(rows_html)
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{stage_name}</title>"
        "<style>body{font-family:Arial,sans-serif;font-size:13px;margin:20px}"
        "h1{color:#2c3e50}table{border-collapse:collapse;width:100%}"
        "th{background:#2c3e50;color:white;padding:6px 10px;text-align:left}"
        "td{border:1px solid #ddd;padding:5px 10px}"
        "tr:nth-child(even){background:#f9f9f9}"
        ".ok{color:green}.warn{color:orange}.note{color:#555;font-style:italic}"
        "</style></head><body>"
        f"<h1>{stage_name} â€” Estimated Model Coefficients</h1>"
        f"<p class='note'>{_html.escape(notes)}</p>"
        "<table><tr><th>Group</th><th>Sub-group</th>"
        "<th>n obs</th><th>OK?</th><th>Features</th><th>Fit stat</th></tr>"
        f"{body}</table></body></html>"
    )
    with open(path, 'w', encoding='utf-8') as _f:
        _f.write(doc)
    print(f"  HTML model report â†’ {path}")


def _emit_model_entry(rows: list, entry: dict, label: str) -> None:
    """Format one model entry into an HTML table row."""
    import html as _h
    g1, g2 = (label.split('/', 1) + [''])[:2]
    n       = entry.get('n_obs', entry.get('n', '?'))
    conv    = entry.get('converged', entry.get('converged_', None))
    feats   = entry.get('features', [])
    rmse    = entry.get('rmse', entry.get('rmse_', None))
    ll      = entry.get('ll', None)
    classes = entry.get('classes', None)
    stat = (f"RMSE={float(rmse):.3f}" if rmse is not None else
            f"LL={float(ll):.1f}"     if ll   is not None else "")
    if classes is not None:
        stat += f"  classes={list(classes)}"
    conv_s = ('<span class="ok">&#10003;</span>' if conv is True  else
              '<span class="warn">&#10007;</span>' if conv is False else '&mdash;')
    feat_s = _h.escape(', '.join(str(f) for f in feats[:12]))
    if len(feats) > 12:
        feat_s += f' &hellip; (+{len(feats)-12})'
    rows.append(
        f"<tr><td>{_h.escape(g1)}</td><td>{_h.escape(g2)}</td>"
        f"<td>{n}</td><td>{conv_s}</td>"
        f"<td style='font-size:11px'>{feat_s}</td>"
        f"<td>{_h.escape(stat)}</td></tr>"
    )
def save_models(models: dict, path: str = 'saved_data/stage2_models.pkl'):
    # Save pkl first — artifact generation is non-fatal.
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Stage 2 models saved → {path}")
    try:
        html_path = path.replace('.pkl', '_report.html')
        flat = {}
        for k, v in models.items():
            mdl = v.get('model')
            flat[k] = {'model': mdl,
                       'n_obs': getattr(mdl, 'n_', '?'),
                       'rmse':  getattr(mdl, 'rmse_', None),
                       'll':    getattr(mdl, 'll_', None),
                       'features': (['intercept', 'log_sigma'] + list(v.get('features', [])) if k in ['start', 'end'] else ['intercept'] + list(v.get('features', []))),
                       'converged': getattr(mdl, 'converged_', True)}
        _write_model_html({'stage2': flat}, html_path, 'Stage 2 - Day Duration',
                          notes=f'start/end/budget duration models. Saved: {path}')
    except Exception as _html_e:
        print(f"  Stage 2 HTML report failed (non-fatal): {_html_e}")
    try:
        from paper_model_forms import write_stage_paper_artifacts
        write_stage_paper_artifacts('stage2', os.path.dirname(path) or 'saved_data')
    except Exception as _paper_e:
        print(f"  Stage 2 paper artifacts failed (non-fatal): {_paper_e}")

def load_models(path: str = 'saved_data/stage2_models.pkl') -> dict:
    with open(path, 'rb') as f:
        models = pickle.load(f)
    print(f"Stage 2 models loaded â† {path}")
    return models


# ---------------------------------------------------------------------------
# 6.  FULL PIPELINE WRAPPER
# ---------------------------------------------------------------------------

def run_stage2(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    synth_s1: pd.DataFrame,
    sample: bool = True,
    seed: int = 42,
    save_path: str = 'saved_data/stage2_models.pkl',
    feature_override: list = None,
) -> tuple:
    """Estimate day duration models on HHTS, predict on Stage 1 output."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    print("=" * 55)
    print("STAGE 2 â€” Day Duration Budget")
    print("=" * 55)
    if feature_override:
        print(f"  Feature override: {feature_override}")

    print("\n[1/2] Building person-day dataset from HHTS...")
    person_day = aggregate_person_day(tours, persons)

    print("\n[1/2] Estimating models...")
    models = estimate_day_models(person_day, feature_override=feature_override)

    if save_path:
        save_models(models, save_path)

    print("\n[2/2] Predicting synthetic population...")
    synth_s2 = predict_day_budget(synth_s1, models, sample=sample, seed=seed)

    print("\nStage 2 complete. Pass synth_s2 to run_stage3().")
    return synth_s2, models


# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------
# Usage:
#   python stage2_day_duration.py                         # resume from saved_data/stage1_output.csv
#   python stage2_day_duration.py --fresh                 # fresh: re-run Stage 1 first
#   python stage2_day_duration.py --fresh 40              # fresh: zone 40
#   python stage2_day_duration.py --input path/to/s1.csv  # explicit Stage 1 input file
#
# Resume mode (default) reads saved_data/stage1_output.csv produced by
# stage1_activity_frequency.py (or pipeline_logger.py) and runs Stage 2 only.

if __name__ == '__main__':
    import sys
    import os
    import logging
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('stage2_main')

    _OUT   = os.environ.get('PIPELINE_OUT_DIR', 'saved_data')
    S1_CSV = os.path.join(_OUT, 'stage1_output.csv')

    args  = sys.argv[1:]
    fresh = '--fresh' in args
    args  = [a for a in args if a != '--fresh']

    # --input override
    input_csv = S1_CSV
    if '--input' in args:
        idx = args.index('--input')
        input_csv = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    zone_arg = args[0] if args else None

    # â”€â”€ load HHTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log.info("Loading HHTS survey data...")
    from data_connect import Synth_Connect
    sc  = Synth_Connect()
    out = sc.process_survey(save=False)
    log.info(f"  HHTS loaded: {len(out['persons'])} persons, {len(out['tours'])} tours")

    if fresh:
        # run Stage 1 first, then hand off
        log.info("Fresh mode: running Stage 1 first...")
        import subprocess, sys as _sys
        cmd = [_sys.executable, 'stage1_activity_frequency.py', '--fresh']
        if zone_arg:
            cmd.append(zone_arg)
        subprocess.run(cmd, check=True)

    # â”€â”€ load Stage 1 output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Stage 1 output not found at '{input_csv}'.\n"
            f"  Run stage1_activity_frequency.py first, or use --fresh to run it automatically."
        )
    log.info(f"Loading Stage 1 output from {input_csv} ...")
    synth_s1 = pd.read_csv(input_csv)
    log.info(f"  {len(synth_s1):,} persons / activities loaded")

    # â”€â”€ run Stage 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    os.makedirs(_OUT, exist_ok=True)
    synth_s2, models = run_stage2(out['tours'], out['persons'], synth_s1)
    synth_s2.to_csv('saved_data/stage2_output.csv', index=False)
    log.info(f"Stage 2 output saved â†’ {_OUT}/stage2_output.csv")
    log.info(f"Pass {_OUT}/stage2_output.csv to stage3_activity_duration.py")




