"""
stage3_activity_duration.py
============================
Stage 3 of the activity-based model pipeline.

Predicts the DURATION (minutes) of each individual activity for every
person, constrained so durations sum to the day budget B from Stage 2.

Pipeline position
-----------------
    Stage 1: n_{purpose}_out  (how many activities)
    Stage 2: start_of_day, end_of_day, B  (when and total budget)
        → Stage 3: activity duration      ← YOU ARE HERE
        → Stage 4: schedule placement
        → Stage 5: destination choice
        → Stage 6: mode choice

Model structure
---------------
    For each purpose a budget-constrained lognormal is fitted:
        log(duration_i) ~ N(X_i @ beta, sigma^2)
    subject to: sum(duration_i for person p) <= B_p

    Estimation uses the tour_duration_model.py approach (already built).
    Prediction expands each person into one row per activity, runs the
    lognormal mean, then rescales so the person's total equals B.

Outputs per person
------------------
    One row per activity instance with columns:
        person_id, activity_seq, purpose, predicted_duration_min,
        start_of_day, end_of_day, B, n_tours, OTAZ
    (ready for Stage 4 schedule placement)

Usage
-----
    from stage3_activity_duration import run_stage3

    out      = Synth_Connect().process_survey(save=False)
    synth_s2 = pd.read_csv('saved_data/stage2_output.csv')

    activities, models = run_stage3(out["tours"], out["persons"], synth_s2)
    activities.to_csv(os.path.join(_OUT, 'stage3_output.csv'), index=False)
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

PURPOSES = [
    'work', 'edu', 'social', 'escort',
    'shopping', 'othmaint', 'othdiscr', 'eat',
]

ACTIVITY_COLS = [f'n_{p}_out' for p in PURPOSES]

# Duration floors / ceilings per purpose (minutes).
# Upper bounds are SOFT — the model can predict above them (lognormal tail),
# and the budget rescaling may stretch up to 1.5× the bound before clipping.
# Work raised to 840 min (14 h) so that shift workers, overtime, and
# continuous-day workers are captured in the training distribution; the soft
# ceiling becomes 1260 min.  Observed HHTS work durations reach ~900 min.
# Edu raised to 600 min (10 h) to capture university full-day timetables,
# TAFE block-days, and after-school programs; soft ceiling = 900 min.
DURATION_BOUNDS = {
    'work':      (30,  840),   # 14 h max — covers shift workers and long days
    'edu':       (30,  600),   # 10 h max — covers full uni/school days
    'social':    (15,  300),
    'escort':    (10,  180),   # widened from (5, 120)
    'shopping':  (5,   180),
    'othmaint':  (5,   120),
    'othdiscr':  (5,   180),
    'eat':       (10,  120),
}

COEF_CLIP_LO = -20.0
COEF_CLIP_HI = 20.0
INTERCEPT_CLIP_LO = -1000.0
INTERCEPT_CLIP_HI = 1000.0

PERSON_FEATURES = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp']


def _quantile_profile(values: pd.Series, q=None) -> dict:
    """Compact empirical distribution summary for rank-based remapping."""
    if q is None:
        q = np.linspace(0.001, 0.999, 199)
    s = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {'q': np.asarray(q, dtype=float), 'v': np.zeros(len(q), dtype=float)}
    return {'q': np.asarray(q, dtype=float), 'v': np.quantile(s.values.astype(float), q)}


def _apply_quantile_profile(values: np.ndarray, profile: dict) -> np.ndarray:
    """Monotonic rank remapping that preserves ordering while matching marginals."""
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
# 0.  PREPARE ESTIMATION DATA  (one row per tour from HHTS)
# ---------------------------------------------------------------------------

def prepare_duration_data(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    trips: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build activity-level estimation dataset with purpose, duration,
    and person attributes.

    Estimation source priority:
        1. trips table (DESTPURP + ACTIVITY_DURATION) — one row per
           individual activity stop, covers ALL purposes including
           eat, othdiscr which rarely appear as tour primary purpose.
        2. tours table fallback — one row per tour (main purpose only).

    Each row = one activity instance.
    Duration = time spent at the activity destination (minutes).
    """
    # ── person attributes ──────────────────────────────────────────
    pers = persons.copy()
    if 'person_id' not in pers.columns and 'PERSID' in pers.columns:
        pers = pers.rename(columns={'PERSID': 'person_id'})
    try:
        from stage1_activity_frequency import assign_ptype_column
        pers = assign_ptype_column(pers)
    except ImportError:
        pers['ptype'] = 'NonWorker'
    keep = ['person_id', 'ptype'] + [c for c in PERSON_FEATURES if c in pers.columns]

    # ── day budget per person from tours ───────────────────────────
    t = tours.copy()
    if 'person_id' not in t.columns and 'PERSID' in t.columns:
        t = t.rename(columns={'PERSID': 'person_id'})
    t['_dur_hrs'] = (t['duration'] / 60).clip(lower=0.05)
    budget = t.groupby('person_id')['_dur_hrs'].sum().reset_index(name='B')

    # tour-level activity counts (n_work_out etc.)
    act_count_cols = [c for c in ACTIVITY_COLS if c in t.columns]
    tour_counts = t.groupby('person_id')[act_count_cols].sum().reset_index()

    # start_time for start_hr feature
    tour_start = (
        t.groupby('person_id')['start_time'].min().reset_index()
        if 'start_time' in t.columns
        else None
    )

    # ── build activity rows ────────────────────────────────────────
    if trips is not None and len(trips) > 0:
        df = trips.copy()
        if 'person_id' not in df.columns and 'PERSID' in df.columns:
            df = df.rename(columns={'PERSID': 'person_id'})

        # purpose column — prefer DESTPURP
        purp_col = next((c for c in ['DESTPURP', 'purpose'] if c in df.columns), None)
        if purp_col is None:
            logger.warning("trips has no DESTPURP or purpose column — falling back to tours")
            trips = None
        else:
            df = df.rename(columns={purp_col: 'main_purpose'}) if purp_col != 'main_purpose' else df

            # duration at activity = ACTIVITY_DURATION if available,
            # else DURATION - TRAVTIME, else DURATION
            if 'ACTIVITY_DURATION' in df.columns:
                df['duration'] = pd.to_numeric(
                    df['ACTIVITY_DURATION'], errors='coerce'
                ).fillna(0).clip(lower=0)
            elif 'TRAVTIME' in df.columns and 'DURATION' in df.columns:
                df['duration'] = (
                    pd.to_numeric(df['DURATION'], errors='coerce').fillna(0)
                    - pd.to_numeric(df['TRAVTIME'], errors='coerce').fillna(0)
                ).clip(lower=0)
            elif 'DURATION' in df.columns:
                df['duration'] = pd.to_numeric(
                    df['DURATION'], errors='coerce'
                ).fillna(0).clip(lower=0)
            else:
                df['duration'] = 30.0

            # drop home trips — we don't model time "at home"
            home_mask = df['main_purpose'].isin(['home', 'Home', None])
            df = df[~home_mask].copy()

            # OTAZ / DTAZ
            for col in ['OTAZ', 'DTAZ']:
                if col not in df.columns:
                    df[col] = 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df['log_dist'] = np.log1p(np.abs(df['DTAZ'] - df['OTAZ']))

            # start hour
            st_col = next((c for c in ['STARTIME', 'start_time', 'depart']
                           if c in df.columns), None)
            df['start_hr'] = (
                pd.to_numeric(df[st_col], errors='coerce').fillna(480) / 60
                if st_col else 8.0
            )

            # merge person attrs + budget
            df = df.merge(pers[keep], on='person_id', how='left')
            df = df.merge(budget, on='person_id', how='left')
            df = df.merge(tour_counts, on='person_id', how='left')

            logger.info(
                f"prepare_duration_data: using TRIPS table "
                f"({len(df)} activity rows across "
                f"{df['person_id'].nunique()} persons)"
            )

    if trips is None or len(trips) == 0:
        # ── fallback: tour-level ───────────────────────────────────
        logger.info("prepare_duration_data: falling back to TOURS table")
        df = t.copy()
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(60)

        purp_col = next(
            (c for c in ['main_purpose', 'primary_purpose', 'tour_primary_purpose']
             if c in df.columns), None
        )
        if purp_col and purp_col != 'main_purpose':
            df = df.rename(columns={purp_col: 'main_purpose'})
        elif purp_col is None:
            df['main_purpose'] = 'othdiscr'

        orig = df.get('tour_origin', pd.Series(0, index=df.index)).fillna(0)
        dest = df.get('tour_destination', pd.Series(0, index=df.index)).fillna(0)
        df['log_dist'] = np.log1p(np.abs(dest - orig))
        df['start_hr'] = df.get('start_time', pd.Series(480, index=df.index)) / 60

        df = df.merge(pers[keep], on='person_id', how='left')
        df = df.merge(budget, on='person_id', how='left')
        df = df.merge(tour_counts, on='person_id', how='left')

    # ── common cleanup ─────────────────────────────────────────────
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(float)

    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2,
                'WORKSTAT': 0, 'STUDYING': 0, 'incp': 8,
                'B': 8.0, 'start_hr': 8.0}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    ptypes = sorted(df['ptype'].dropna().unique()) if 'ptype' in df.columns else []
    for pt in ptypes:
        df[f'ptype_{pt}'] = (df['ptype'] == pt).astype(float)

    df.attrs['ptypes'] = ptypes
    return df


# ---------------------------------------------------------------------------
# 1.  BUDGET-CONSTRAINED LOGNORMAL  (pure numpy, no JAX required)
# ---------------------------------------------------------------------------

class BudgetLognormal:
    """
    Accelerated Failure Time (AFT) lognormal duration model.
    """

    def __init__(self, l2: float = 1e-4):
        self.l2 = l2
        self.coef_ = None
        self.intercept_ = None
        self.log_sigma_ = None
        self.sigma_ = None
        self.feature_names_ = None
        self.mu_ = None
        self.sd_ = None
        self.std_err_ = None
        self.t_stat_ = None
        self.p_value_ = None
        self.ll_ = None
        self.aic_ = None
        self.rmse_ = None
        self.n_ = None
        self.converged_ = None

    def _clip_coefficients(self):
        if self.coef_ is not None:
            self.coef_ = np.clip(self.coef_, COEF_CLIP_LO, COEF_CLIP_HI)
        if self.intercept_ is not None:
            self.intercept_ = float(np.clip(self.intercept_, INTERCEPT_CLIP_LO, INTERCEPT_CLIP_HI))

    def _neg_ll(self, params: np.ndarray, Xs: np.ndarray, log_y: np.ndarray) -> float:
        n_feat = Xs.shape[1] - 1
        intercept = params[0]
        beta = params[1:n_feat + 1]
        log_sigma = params[n_feat + 1]
        sigma = np.exp(log_sigma)
        eta = intercept + Xs[:, 1:] @ beta
        resid = log_y - eta
        N = len(log_y)
        ll = (-N * log_sigma - 0.5 * np.sum((resid / sigma) ** 2) - np.sum(log_y))
        ll -= 0.5 * self.l2 * np.dot(beta, beta)
        return -float(ll)

    def _neg_ll_grad(self, params: np.ndarray, Xs: np.ndarray, log_y: np.ndarray) -> np.ndarray:
        n_feat = Xs.shape[1] - 1
        beta = params[1:n_feat + 1]
        log_sigma = params[n_feat + 1]
        sigma = np.exp(log_sigma)
        eta = params[0] + Xs[:, 1:] @ beta
        resid = log_y - eta
        s2 = sigma ** 2
        N = len(log_y)
        d_intercept = np.sum(resid) / s2
        d_beta = Xs[:, 1:].T @ resid / s2 - self.l2 * beta
        d_logsigma = -N + np.sum(resid ** 2) / s2
        return np.concatenate([[d_intercept], d_beta, [d_logsigma]])

    def fit(self, X: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])
        log_y = np.log(np.clip(y, 1e-6, None))
        ols = np.linalg.lstsq(Xs, log_y, rcond=None)[0]
        resid_std = max(np.std(log_y - Xs @ ols), 0.05)
        init = np.concatenate([ols, [np.log(resid_std)]])

        def neg_ll_fn(p): return self._neg_ll(p, Xs, log_y)
        def neg_gr_fn(p): return -self._neg_ll_grad(p, Xs, log_y)

        res = minimize(neg_ll_fn, init, jac=neg_gr_fn, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        n_feat = X.shape[1]
        self.intercept_ = res.x[0]
        self.coef_ = res.x[1:n_feat + 1]
        self.log_sigma_ = res.x[n_feat + 1]
        self.sigma_ = float(np.exp(self.log_sigma_))
        self.ll_ = float(-res.fun)
        self.aic_ = float(2 * (n_feat + 2) - 2 * self.ll_)
        self.rmse_ = float(np.sqrt(np.mean((y - self.predict_mean(X)) ** 2)))
        self.n_ = len(y)
        self.converged_ = bool(res.success)
        self._clip_coefficients()
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
        return self

    def _eta(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2 and X.shape[1] != len(self.mu_):
            raise ValueError(
                f"Feature mismatch: model expects {len(self.mu_)} columns "
                f"but received {X.shape[1]}. "
                f"Delete saved_data/stage3_models.pkl and re-run."
            )
        Xs = (X - self.mu_) / self.sd_
        return self.intercept_ + Xs @ self.coef_

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self._eta(X) + 0.5 * self.sigma_ ** 2)

    def predict_sampled(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.exp(rng.normal(self._eta(X), self.sigma_))


def _augment_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    if 'B' not in df.columns:
        df['B'] = 8.0
    df['B'] = pd.to_numeric(df['B'], errors='coerce').fillna(8.0)
    total_acts = df[ACTIVITY_COLS].sum(axis=1).clip(lower=1.0)
    df['budget_per_activity_hr'] = df['B'] / total_acts
    if 'start_of_day' in df.columns and 'end_of_day' in df.columns:
        df['day_span_hr'] = np.clip((df['end_of_day'] - df['start_of_day']) / 60.0, 0.25, 18.0)
    else:
        df['day_span_hr'] = df['B']
    df['n_mandatory'] = df.get('n_work_out', 0.0) + df.get('n_edu_out', 0.0)
    df['n_discretionary'] = total_acts - df['n_mandatory']
    return df


def _feature_cols(df: pd.DataFrame, ptypes: list,
                  feature_override: list = None) -> list:
    person_feats = feature_override if feature_override else PERSON_FEATURES
    base = (['start_hr', 'log_dist', 'B', 'budget_per_activity_hr', 'day_span_hr',
             'n_mandatory', 'n_discretionary'] +
            ACTIVITY_COLS +
            [c for c in person_feats if c in df.columns] +
            [f'ptype_{pt}' for pt in ptypes if f'ptype_{pt}' in df.columns])
    return base


def estimate_duration_models(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    trips: Optional[pd.DataFrame] = None,
    min_sample: int = 50,
    feature_override: list = None,
) -> dict:
    """
    Fit one BudgetLognormal per activity purpose.

    Parameters
    ----------
    tours, persons    : from process_survey()
    min_sample        : skip purposes with fewer observations
    feature_override  : GA-selected feature list (replaces PERSON_FEATURES)
    """
    data    = _augment_duration_features(prepare_duration_data(tours, persons, trips=trips))
    ptypes  = data.attrs.get('ptypes', [])
    feats   = _feature_cols(data, ptypes, feature_override=feature_override)

    # ---- diagnose purpose column before fitting -------------------------
    purpose_col = None
    for candidate in ['main_purpose', 'primary_purpose', 'tour_primary_purpose']:
        if candidate in data.columns and data[candidate].notna().any():
            purpose_col = candidate
            break

    if purpose_col is None:
        logger.error("No purpose column found in tours. Checked: "                     "main_purpose, primary_purpose, tour_primary_purpose")
        return {}

    unique_vals = data[purpose_col].dropna().unique()
    logger.info(f"Using purpose column: '{purpose_col}' "                f"with {len(unique_vals)} unique values: {unique_vals[:10]}")

    # map to coded purposes if column contains raw survey strings
    raw_to_code = {
        'Work (my workplace)': 'work', 'Work (other work reason)': 'work',
        'Direct Work Commute': 'work', 'Work Related': 'work',
        'Education': 'edu', 'Childcare or Kindergarten': 'edu',
        'Childcare or kindergarten': 'edu',
        'Social Visit': 'social', 'Recreation/leisure activity': 'social',
        'Recreation': 'social', 'Social': 'social',
        'Accompanying Someone': 'social', 'Accompany Someone': 'social',
        'Pick up or drop off a passenger': 'escort',
        'Pickup/Dropoff Someone': 'escort', 'Pickup/Deliver Something': 'escort',
        'Just accompanying someone': 'escort',
        'Shopping': 'shopping',
        'Personal Business': 'othmaint', 'Personal business': 'othmaint',
        'Pickup/deliver something (not work-related)': 'othmaint',
        'Other place (please specify)': 'othdiscr', 'Other Place': 'othdiscr',
        'Refuel my vehicle': 'othdiscr', 'Other Purpose': 'othdiscr',
        'At Home': 'home', 'Go home': 'home',
        'Eat/drink': 'eat',
    }
    # if values are already codes (work, edu…) mapping is identity
    already_coded = set(unique_vals).issubset(set(PURPOSES) | {'home', None})
    if not already_coded:
        logger.info("Mapping raw survey strings → purpose codes")
        data['_purpose_coded'] = data[purpose_col].map(raw_to_code)
    else:
        data['_purpose_coded'] = data[purpose_col]

    n_unmapped = data['_purpose_coded'].isna().sum()
    if n_unmapped > 0:
        unmapped_vals = data.loc[data['_purpose_coded'].isna(), purpose_col].unique()
        logger.warning(f"{n_unmapped} tours have unmapped purpose: {unmapped_vals}")

    models = {}

    for purpose in PURPOSES:
        lo, hi = DURATION_BOUNDS[purpose]

        # filter to tours whose purpose matches
        subset = data[data['_purpose_coded'] == purpose].copy()

        if len(subset) < min_sample:
            logger.warning(f"  {purpose}: {len(subset)} obs < {min_sample} — skipping")
            continue

        y = subset['duration'].clip(lo, hi).values.astype(float)

        X_cols = [c for c in feats if c in subset.columns]
        X = subset[X_cols].fillna(0).values.astype(float)

        m = BudgetLognormal()
        m.fit(X, y)
        m.feature_names_ = X_cols
        mean_pred = float(np.nanmean(m.predict_mean(X))) if len(X) else float('nan')
        mean_obs = float(np.nanmean(y)) if len(y) else float('nan')
        calib = 1.0
        if np.isfinite(mean_pred) and mean_pred > 0 and np.isfinite(mean_obs):
            # Work allows a higher ceiling: full-time workers regularly have 8–12h
            # activities; the lognormal mean undershoots because sigma is estimated
            # on a clipped distribution.  Allow up to 3.0× correction for work
            # to capture long-duration outliers.
            _calib_hi = 3.0 if purpose == 'work' else 1.50
            calib = float(np.clip(mean_obs / mean_pred, 0.67, _calib_hi))

        models[purpose] = {
            'model':    m,
            'features': X_cols,
            'bounds':   (lo, hi),
            'n_obs':    len(subset),
            'sigma':    m.sigma_,
            'll':       m.ll_,
            'rmse':     m.rmse_,
            'calibration_factor': calib,
            'mean_obs': mean_obs,
            'mean_pred_uncal': mean_pred,
            'observed_profile': _quantile_profile(pd.Series(y)),
        }
        bias_pct = (mean_pred - mean_obs) / mean_obs * 100 if mean_obs > 0 else float('nan')
        bias_str = f"{bias_pct:+.1f}%" if np.isfinite(bias_pct) else "n/a"
        logger.info(f"  {purpose}: n={len(subset)}, "
                    f"sigma={m.sigma_:.3f}, "
                    f"obs_mean={mean_obs:.1f} min, pred_mean={mean_pred:.1f} min "
                    f"(bias={bias_str}), calib={calib:.3f}")

    print(f"Stage 3: fitted {len(models)} duration models "
          f"({', '.join(models.keys())})")
    return models


# ---------------------------------------------------------------------------
# 3.  EXPAND PERSON → ACTIVITY ROWS
# ---------------------------------------------------------------------------

def expand_to_activities(synth: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the person-level DataFrame (Stage 2 output) to one row per
    activity instance.

    Each person with n_work_out=2, n_eat_out=1 gets 3 rows.

    Returns
    -------
    DataFrame with columns:
        person_id, activity_seq, purpose,
        + all original person columns carried through
    """
    df = synth.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    rows = []
    for _, row in df.iterrows():
        seq = 0
        for purpose in PURPOSES:
            col = f'n_{purpose}_out'
            n   = int(row.get(col, 0))
            for _ in range(n):
                r             = row.to_dict()
                r['purpose']  = purpose
                r['activity_seq'] = seq
                seq += 1
                rows.append(r)

    if not rows:
        logger.warning("expand_to_activities: no activity rows produced. "
                       "Check that n_{purpose}_out columns exist and are > 0.")
        return pd.DataFrame()

    activities = pd.DataFrame(rows).reset_index(drop=True)
    logger.info(f"Expanded {len(df)} persons → {len(activities)} activities")
    return activities


# ---------------------------------------------------------------------------
# 4.  PREDICT DURATIONS WITH BUDGET RESCALING
# ---------------------------------------------------------------------------

def predict_activity_durations(
    synth_s2: pd.DataFrame,
    models: dict,
    sample: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Predict duration for every activity instance, then rescale per person
    so that sum(durations) == B.

    Parameters
    ----------
    synth_s2 : Stage 2 output — one row per person with
               n_{purpose}_out, start_of_day, end_of_day, B
    models   : output of estimate_duration_models()
    sample   : sample from lognormal (True) or use mean (False)
    seed     : random seed

    Returns
    -------
    DataFrame with one row per activity, columns:
        person_id, activity_seq, purpose,
        predicted_duration_min, predicted_duration_hrs,
        start_of_day, end_of_day, B, OTAZ, ptype, …
    """
    rng = np.random.default_rng(seed)

    # expand persons → activities
    acts = expand_to_activities(synth_s2)
    if acts.empty:
        return acts

    # ptype if missing
    if 'ptype' not in acts.columns:
        try:
            from stage1_activity_frequency import assign_ptype_column
            acts = assign_ptype_column(acts)
        except ImportError:
            acts['ptype'] = 'NonWorker'

    # derived features
    acts['start_hr'] = acts.get('start_of_day', pd.Series(480, index=acts.index)) / 60

    orig = acts.get('tour_origin',
           acts.get('OTAZ', pd.Series(0, index=acts.index))).fillna(0)
    dest = acts.get('tour_destination',
           acts.get('DTAZ', pd.Series(0, index=acts.index))).fillna(0)
    acts['log_dist'] = np.log1p(np.abs(dest - orig))

    # budget column
    if 'B' not in acts.columns:
        acts['B'] = 8.0  # default 8-hr budget

    # ptype dummies
    all_ptypes = sorted(acts['ptype'].dropna().unique())
    for pt in all_ptypes:
        acts[f'ptype_{pt}'] = (acts['ptype'] == pt).astype(float)

    # person feature defaults
    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2,
                'WORKSTAT': 0, 'STUDYING': 0, 'incp': 8}
    for col, val in defaults.items():
        if col not in acts.columns:
            acts[col] = val
        acts[col] = acts[col].fillna(val)

    acts = _augment_duration_features(acts)
    acts['predicted_duration_min'] = np.nan

    # predict per purpose
    for purpose in PURPOSES:
        mask = acts['purpose'] == purpose

        if not mask.any():
            continue

        if purpose not in models:
            # fallback: use median duration from bounds
            lo, hi = DURATION_BOUNDS[purpose]
            acts.loc[mask, 'predicted_duration_min'] = (lo + hi) / 2
            continue

        entry    = models[purpose]
        feats    = entry['features']
        lo, hi   = entry['bounds']
        model    = entry['model']

        subset = acts.loc[mask]
        X_cols = [c for c in feats if c in subset.columns]

        # pad any missing feature columns with 0
        X = np.zeros((mask.sum(), len(feats)))
        for j, col in enumerate(feats):
            if col in subset.columns:
                X[:, j] = subset[col].fillna(0).values.astype(float)

        if sample:
            # ActivitySim-style: use a minimum effective sigma so individual
            # durations vary even when the model fits tightly (small sigma_).
            # Purpose-specific floors:
            #   work: 0.45 — right-skewed (shift workers, overtime, long meetings)
            #   eat:  0.40 — wider spread (quick lunch vs long dinner)
            #   others: 0.30 default
            _sigma_floors = {'work': 0.45, 'eat': 0.40}
            _eff_sigma = max(float(model.sigma_), _sigma_floors.get(purpose, 0.30))
            eta = model._eta(X)   # log-scale mean, shape (n_mask,)

            # Person-level random intercept: each person gets a consistent
            # time-preference draw (positive = prefers longer activities today,
            # negative = prefers shorter).  This creates realistic cross-person
            # variation even within the same ptype/purpose cell.
            # Aligned to acts index so the same person gets the same draw
            # across different purpose rows.
            person_ids = subset['person_id'].values if 'person_id' in subset.columns else None
            if person_ids is not None:
                unique_pids = np.unique(person_ids)
                pid_offset  = {pid: rng.normal(0, 0.15) for pid in unique_pids}
                person_offsets = np.array([pid_offset[p] for p in person_ids], dtype=float)
            else:
                person_offsets = rng.normal(0, 0.15, size=len(eta))

            preds = np.exp(rng.normal(eta + person_offsets, _eff_sigma))
        else:
            preds = model.predict_mean(X)

        calib = float(entry.get('calibration_factor', 1.0) or 1.0)
        preds = preds * calib
        preds = _apply_quantile_profile(preds, entry.get('observed_profile'))
        preds = np.clip(preds, lo, hi)
        acts.loc[mask, 'predicted_duration_min'] = preds

    # fill any remaining NaN with purpose midpoint
    for purpose in PURPOSES:
        lo, hi = DURATION_BOUNDS[purpose]
        mask = (acts['purpose'] == purpose) & acts['predicted_duration_min'].isna()
        acts.loc[mask, 'predicted_duration_min'] = (lo + hi) / 2

    # ── budget rescaling per person ──────────────────────────────────────────
    # ActivitySim approach: treat B as a SOFT guide, not a hard constraint.
    # Only scale down when significantly over-budget (>20%), and apply a
    # proportional scale that preserves the RELATIVE duration pattern of each
    # person.  Do NOT force every person's total to equal B — that collapses
    # cross-person variation and makes all plans look identical.
    #
    # Strategy:
    #  1. Enforce per-purpose floors and ceilings.
    #  2. If total > 1.20 × B_min, scale proportionally (preserves ratios).
    #  3. If total < 0.50 × B_min, scale up discretionary durations only
    #     (mandatory activities are kept as-is to reflect real constraints).
    #  4. A ±10% tolerance band around B is accepted without any rescaling,
    #     letting the lognormal draws create natural person-to-person variation.
    acts['predicted_duration_min'] = acts['predicted_duration_min'].astype(float)
    pid_col = 'person_id'

    # ── Enforce FLOORS only before rescaling ─────────────────────────────────
    # Do NOT clip the upper bound here — hard upper-bound clipping before the
    # budget rescaling creates a spike (pile-up) at the ceiling when many
    # lognormal draws fall above the bound.  The budget rescaling will pull
    # over-long activities back proportionally, which is smoother.
    for purpose in PURPOSES:
        lo, _ = DURATION_BOUNDS[purpose]
        pmask = acts['purpose'] == purpose
        acts.loc[pmask, 'predicted_duration_min'] = (
            acts.loc[pmask, 'predicted_duration_min'].clip(lower=lo)
        )

    rescaled = []
    _mandatory = {'work', 'edu'}
    for pid, grp in acts.groupby(pid_col):
        B_hrs = float(grp['B'].iloc[0])
        # B is in hours from Stage 2 — convert to minutes
        # guard against B already being in minutes (if > 24 hrs it's minutes)
        B_min = B_hrs * 60 if B_hrs <= 24 else B_hrs
        # cap at a realistic day length for this model (up to 16 hours = 960 min)
        B_min = float(np.clip(B_min, 30, 960))

        durations = grp['predicted_duration_min'].copy().astype(float)
        purposes  = grp['purpose'].values

        # Step 1: enforce floors only (no ceiling here — prevents boundary spike)
        for i, p in enumerate(purposes):
            lo_p, _ = DURATION_BOUNDS.get(p, (5, 480))
            durations.iloc[i] = float(max(durations.iloc[i], lo_p))

        total = durations.sum()
        if total <= 0:
            rescaled.append(durations)
            continue

        ratio = total / max(B_min, 1.0)

        if ratio > 1.20:
            # Over budget by more than 20% — scale everything proportionally.
            # Preserves each activity's RELATIVE share of the day.
            scale = B_min / total
            durations = durations * scale
            # Re-enforce floors ONLY after scaling (no ceiling to avoid spike)
            for i, p in enumerate(purposes):
                lo_p, _ = DURATION_BOUNDS.get(p, (5, 480))
                durations.iloc[i] = float(max(durations.iloc[i], lo_p))

        elif ratio < 0.50:
            # Under budget by more than half — scale up DISCRETIONARY only.
            # Mandatory (work/edu) durations are respected as sampled.
            discr_idx = [i for i, p in enumerate(purposes) if p not in _mandatory]
            if discr_idx:
                mandatory_total = sum(
                    durations.iloc[i] for i, p in enumerate(purposes)
                    if p in _mandatory
                )
                remaining_budget = max(B_min - mandatory_total, 0)
                discr_total = sum(durations.iloc[i] for i in discr_idx)
                if discr_total > 0 and remaining_budget > discr_total:
                    scale_d = min(remaining_budget / discr_total, 2.0)  # cap 2×
                    for i in discr_idx:
                        p = purposes[i]
                        lo_p, _ = DURATION_BOUNDS.get(p, (5, 480))
                        durations.iloc[i] = float(max(durations.iloc[i] * scale_d, lo_p))
        # else: within ±20% / ±50% band — leave as-is (preserves variation)

        rescaled.append(durations)

    acts['predicted_duration_min'] = pd.concat(rescaled).sort_index()

    # ── Final soft ceiling: only clip extreme outliers (> 1.5× upper bound) ──
    # This prevents genuinely unrealistic values (e.g. 20-hour work activity)
    # while not creating a spike at the nominal upper bound (480, 420, etc).
    for purpose in PURPOSES:
        lo, hi = DURATION_BOUNDS[purpose]
        soft_hi = hi * 1.5   # allow up to 1.5× before hard-stopping
        pmask = acts['purpose'] == purpose
        acts.loc[pmask, 'predicted_duration_min'] = (
            acts.loc[pmask, 'predicted_duration_min'].clip(lower=lo, upper=soft_hi)
        )

    acts['predicted_duration_hrs'] = acts['predicted_duration_min'] / 60

    # ── debug: show variation achieved ───────────────────────────────────────
    logger.info(
        "Stage 3 duration variation summary:\n" +
        "\n".join(
            f"  {p}: mean={acts.loc[acts['purpose']==p,'predicted_duration_min'].mean():.0f} min  "
            f"std={acts.loc[acts['purpose']==p,'predicted_duration_min'].std():.0f} min  "
            f"[{acts.loc[acts['purpose']==p,'predicted_duration_min'].min():.0f}–"
            f"{acts.loc[acts['purpose']==p,'predicted_duration_min'].max():.0f}]"
            for p in PURPOSES if (acts['purpose'] == p).any()
        )
    )
    # Warn if any purpose has very low variation (std < 5 min)
    for p in PURPOSES:
        pmask = acts['purpose'] == p
        if pmask.sum() > 10:
            std_p = acts.loc[pmask, 'predicted_duration_min'].std()
            if std_p < 5.0:
                logger.warning(
                    f"Stage 3: {p} duration std = {std_p:.1f} min — very low. "
                    "Check that sample=True is set and sigma_ > 0 in the model."
                )

    print(
        f"Stage 3 predictions: {len(acts)} activities "
        f"across {acts[pid_col].nunique()} persons\n"
        + "\n".join(
            f"  {p}: n={int((acts['purpose']==p).sum())}, "
            f"mean={acts.loc[acts['purpose']==p,'predicted_duration_min'].mean():.1f} min"
            for p in PURPOSES if (acts['purpose'] == p).any()
        )
    )

    return acts


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
        f"<h1>{stage_name} — Estimated Model Coefficients</h1>"
        f"<p class='note'>{_html.escape(notes)}</p>"
        "<table><tr><th>Group</th><th>Sub-group</th>"
        "<th>n obs</th><th>OK?</th><th>Features</th><th>Fit stat</th></tr>"
        f"{body}</table></body></html>"
    )
    with open(path, 'w', encoding='utf-8') as _f:
        _f.write(doc)
    print(f"  HTML model report → {path}")


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
def save_models(models: dict, path: str = 'saved_data/stage3_models.pkl'):
    # Save pkl first — artifact generation is non-fatal.
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Stage 3 models saved → {path}")
    try:
        html_path = path.replace('.pkl', '_report.html')
        flat = {}
        for purpose, entry in models.items():
            flat[purpose] = {
                'model': entry.get('model'),
                'n_obs': entry.get('n_obs', '?'),
                'features': entry.get('features', []),
                'converged': getattr(entry.get('model'), 'converged_', True),
                'll': entry.get('ll'),
                'rmse': entry.get('rmse'),
                'stat': f"sigma={entry.get('sigma','?')}"
            }
        _write_model_html({'lognormal': flat}, html_path,
                          'Stage 3 — Activity Duration',
                          notes=f'Lognormal duration models. Saved: {path}')
    except Exception as _html_e:
        print(f"  Stage 3 HTML report failed (non-fatal): {_html_e}")
    try:
        from paper_model_forms import write_stage_paper_artifacts
        write_stage_paper_artifacts('stage3', os.path.dirname(path) or 'saved_data')
    except Exception as _paper_e:
        print(f"  Stage 3 paper artifacts failed (non-fatal): {_paper_e}")


def load_models(path: str = 'saved_data/stage3_models.pkl') -> dict:
    with open(path, 'rb') as f:
        models = pickle.load(f)
    print(f"Stage 3 models loaded ← {path}")
    return models


# ---------------------------------------------------------------------------
# 6.  FULL PIPELINE WRAPPER
# ---------------------------------------------------------------------------

def run_stage3(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    synth_s2: pd.DataFrame,
    trips: Optional[pd.DataFrame] = None,
    sample: bool = True,
    seed: int = 42,
    save_path: str = 'saved_data/stage3_models.pkl',
    feature_override: list = None,
) -> tuple:
    """Estimate duration models on HHTS, predict for Stage 2 output."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    print("=" * 55)
    print("STAGE 3 — Activity Duration")
    print("=" * 55)
    if feature_override:
        print(f"  Feature override: {feature_override}")

    print("\n[1/2] Estimating duration models from HHTS...")
    models = estimate_duration_models(tours, persons, trips=trips,
                                      feature_override=feature_override)

    if save_path:
        save_models(models, save_path)

    print("\n[2/2] Predicting activity durations for synthetic population...")
    activities = predict_activity_durations(synth_s2, models,
                                            sample=sample, seed=seed)

    print("\nStage 3 complete. Pass activities to run_stage4().")
    return activities, models


# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------
# Usage:
#   python stage3_activity_duration.py                        # resume from saved_data/stage2_output.csv
#   python stage3_activity_duration.py --fresh                # fresh: re-run Stages 1+2 first
#   python stage3_activity_duration.py --input path/to/s2.csv # explicit Stage 2 input
#
# Resume mode (default) reads saved_data/stage2_output.csv and runs Stage 3 only.
# Stage 3 is the slowest pipeline stage (~52 min on 3.7M persons) — resuming from
# saved CSVs avoids re-running expensive upstream stages unnecessarily.

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
    log = logging.getLogger('stage3_main')

    _OUT   = os.environ.get('PIPELINE_OUT_DIR', 'saved_data')
    S2_CSV = os.path.join(_OUT, 'stage2_output.csv')

    args  = sys.argv[1:]
    fresh = '--fresh' in args
    args  = [a for a in args if a != '--fresh']

    input_csv = S2_CSV
    if '--input' in args:
        idx = args.index('--input')
        input_csv = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    zone_arg = args[0] if args else None

    # ── load HHTS ────────────────────────────────────────────────────────
    log.info("Loading HHTS survey data...")
    from data_connect import Synth_Connect
    sc  = Synth_Connect()
    out = sc.process_survey(save=False)
    log.info(f"  HHTS loaded: {len(out['persons'])} persons | "
             f"{len(out['tours'])} tours | {len(out['trips'])} trips")

    if fresh:
        log.info("Fresh mode: running Stages 1 and 2 first...")
        import subprocess, sys as _sys
        cmd = [_sys.executable, 'stage1_activity_frequency.py', '--fresh']
        if zone_arg:
            cmd.append(zone_arg)
        subprocess.run(cmd, check=True)
        subprocess.run([_sys.executable, 'stage2_day_duration.py'], check=True)

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Stage 2 output not found at '{input_csv}'.\n"
            f"  Run stage2_day_duration.py first, or use --fresh to run all prior stages."
        )
    log.info(f"Loading Stage 2 output from {input_csv} ...")
    synth_s2 = pd.read_csv(input_csv)
    log.info(f"  {len(synth_s2):,} rows loaded")

    # ── run Stage 3 ───────────────────────────────────────────────────────
    # NOTE: Stage 3 expands persons → activities. On the full 3.7M-person
    # population this produces ~7M activity rows and takes ~52 minutes.
    os.makedirs(_OUT, exist_ok=True)
    activities, models = run_stage3(
        out['tours'], out['persons'], synth_s2, trips=out['trips']
    )
    activities.to_csv('saved_data/stage3_output.csv', index=False)
    log.info(f"Stage 3 output saved → {_OUT}/stage3_output.csv")
    log.info(f"  {len(activities):,} activities across "
             f"{activities['person_id'].nunique():,} persons")
    log.info(f"Pass {_OUT}/stage3_output.csv to stage4_schedule_placement.py")

