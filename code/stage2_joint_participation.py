"""
stage2_joint_participation.py
==============================
Joint participation + duration models for Stage 2 — optional alternative to the
sequential two-stage (Stage-1 ordered-logit + Stage-2 Tobit) approach.

Two econometric approaches are implemented and compared:

    A.  Tobit Type-I (Budget Tobit)
        ----------------------------
        Latent model:    B* = X'beta + epsilon,   epsilon ~ N(0, sigma^2)
        Observed:        B  = max(0, B*)
        Estimation sample includes non-travellers (B=0).
        Jointly identifies:
          - P(travel)         = Phi(X'beta / sigma)          [participation]
          - E[B | B>0, X]     = X'beta + sigma * phi/Phi     [conditional budget]
          - E[B | X]          = Phi(X'beta/sigma) * (X'beta + sigma*phi/Phi)
        Uses the existing TobitModel class from stage2_day_duration.py.

    B.  Heckman Selection Model (Heckit, two-step)
        --------------------------------------------
        Selection equation:  D*  = Z'gamma + u,    u ~ N(0,1)  (probit)
                             D   = 1{D* > 0}
        Outcome equation:    B   = X'beta + rho*sigma*lambda(Z'gamma) + v
                             (estimated on travellers only; lambda = IMR)
        Corrects for selection bias when the decision to travel (D) is
        correlated with the duration (B) given travel.

        Identification: ideally at least one variable in Z not in X.
        Natural candidates: day_of_week, household_size (nprd), STUDYING
        if not already in outcome equation.

        When IMR coefficient is not significant (p > 0.1), selection bias
        is small and Tobit I or the sequential two-stage is sufficient.

How to use
----------
    from stage2_joint_participation import (
        aggregate_person_day_with_nontravellers,
        estimate_joint_models,
        predict_with_joint_models,
        compare_models,
        plot_joint_comparison,
        joint_model_recommendation,
    )

    full_pd   = aggregate_person_day_with_nontravellers(tours, persons)
    joint_res = estimate_joint_models(full_pd)
    metrics   = compare_models(joint_res['tobit1'], joint_res['heckman'],
                               two_stage_models, full_pd)
    html      = joint_model_recommendation(metrics)

Model selection guidance
------------------------
    Non-traveller rate > 10 %  -> Tobit I recommended over sequential.
    IMR significant (p < 0.05) -> Selection bias present -> Heckman preferred.
    IMR not significant        -> Tobit I or sequential are equivalent.
    Very small non-traveller   -> Sequential two-stage is fine; keep simple.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
from stage2_day_duration import (
    PERSON_FEATURES,
    ACTIVITY_COLS,
    MIN_BUDGET,
    MAX_BUDGET,
    TobitModel,
    _augment_day_features,
    _build_X,
    _quantile_profile,
)

# Selection equation can use additional sociodemographic signals that may
# predict the *decision* to travel but not necessarily the *duration*.
SELECTION_EXTRA = ['nprd', 'STUDYING']   # household size, student status


# ---------------------------------------------------------------------------
# 1.  DATA PREPARATION  —  include non-travellers
# ---------------------------------------------------------------------------

def aggregate_person_day_with_nontravellers(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extend aggregate_person_day() to include persons who made NO trips
    on the survey day.  Non-travellers get B=0, n_total=0.

    The 'participated' column flags actual travellers (1) vs non-travellers (0).

    Parameters
    ----------
    tours, persons : from process_survey()

    Returns
    -------
    One-row-per-person DataFrame with a 'participated' column.
    """
    from stage2_day_duration import aggregate_person_day

    # Travellers — standard aggregate
    travellers = aggregate_person_day(tours, persons)
    traveller_ids = set(travellers['person_id'].tolist())
    travellers['participated'] = 1

    # Persons who appear in persons but NOT in tours
    if 'person_id' not in persons.columns and 'PERSID' in persons.columns:
        pers = persons.rename(columns={'PERSID': 'person_id'})
    else:
        pers = persons.copy()

    # assign ptype if absent
    if 'ptype' not in pers.columns:
        try:
            from stage1_activity_frequency import assign_ptype_column
            pers = assign_ptype_column(pers)
        except ImportError:
            pers['ptype'] = 'NonWorker'

    nontraveller_ids = pers.loc[~pers['person_id'].isin(traveller_ids), 'person_id']
    logger.info(
        f"  Travellers: {len(travellers):,}  "
        f"Non-travellers found: {len(nontraveller_ids):,}"
    )

    if len(nontraveller_ids) == 0:
        # All persons travelled — just return with participated flag
        travellers['participated'] = 1
        return travellers

    # Build non-traveller rows from persons table
    keep = ['person_id', 'ptype'] + [c for c in PERSON_FEATURES if c in pers.columns]
    nt = pers.loc[pers['person_id'].isin(nontraveller_ids), keep].copy()

    # Zero out activity counts and time window
    for col in ACTIVITY_COLS:
        nt[col] = 0
    nt['n_total']      = 0
    nt['n_tours']      = 0
    nt['B']            = 0.0
    nt['start_of_day'] = np.nan
    nt['end_of_day']   = np.nan
    nt['participated'] = 0

    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2,
                'STUDYING': 0, 'WORKSTAT': 0, 'incp': 8}
    for col, val in defaults.items():
        if col not in nt.columns:
            nt[col] = val
        nt[col] = nt[col].fillna(val)

    full = pd.concat([travellers, nt], ignore_index=True, sort=False)

    # fill any columns that non-travellers are missing
    for col in travellers.columns:
        if col not in full.columns:
            full[col] = 0
    full = full.fillna({c: 0 for c in ACTIVITY_COLS + ['n_total', 'n_tours', 'B']})

    pct_nt = 100 * len(nontraveller_ids) / max(len(full), 1)
    logger.info(
        f"  Full participation dataset: {len(full):,} persons  "
        f"({pct_nt:.1f}% non-travellers)"
    )
    return full


# ---------------------------------------------------------------------------
# 2.  HECKMAN SELECTION MODEL
# ---------------------------------------------------------------------------

class HeckmanModel:
    """
    Two-step Heckman (Heckit) selection model.

    Step 1: Probit for participation (D=1 → travelled).
    Step 2: OLS of B on (X, IMR) for travellers only.
             IMR (inverse Mills ratio) = phi(Z'gamma) / Phi(Z'gamma)

    This corrects for selection bias: if the unobservables that make someone
    *more likely to travel* also make them travel *longer*, OLS on travellers
    alone overstates the marginal effect of X on duration.

    Attributes (post-fit)
    ---------------------
    sel_intercept_, sel_coef_    : probit coefficients (selection)
    out_intercept_, out_coef_    : OLS coefficients (outcome) incl. IMR term
    imr_coef_                    : coefficient on the IMR (rho*sigma)
    sigma_                       : residual SD of outcome equation
    sel_features_, out_features_ : column lists
    sel_se_, out_se_             : standard errors
    sel_t_, out_t_               : t/z-statistics
    sel_p_, out_p_               : p-values
    ll_sel_, ll_out_             : log-likelihoods
    aic_, bic_                   : model selection criteria (combined)
    participation_rate_          : observed fraction of travellers
    """

    def __init__(self, ridge: float = 0.01):
        self.ridge = ridge
        self.sel_intercept_ = None
        self.sel_coef_      = None
        self.out_intercept_ = None
        self.out_coef_      = None
        self.imr_coef_      = None
        self.sigma_         = None
        self.sel_features_  = None
        self.out_features_  = None
        self.sel_se_  = None;  self.sel_t_  = None;  self.sel_p_  = None
        self.out_se_  = None;  self.out_t_  = None;  self.out_p_  = None
        self.sel_mu_  = None;  self.sel_sd_ = None
        self.out_mu_  = None;  self.out_sd_ = None
        self.ll_sel_  = None;  self.ll_out_ = None
        self.aic_     = None;  self.bic_    = None
        self.n_       = None;  self.n_sel_  = None
        self.participation_rate_ = None
        self.converged_ = None

    # ── Step 1: Probit ───────────────────────────────────────────────────
    @staticmethod
    def _probit_neg_ll(params, X, y, ridge=0.01):
        intercept = params[0]
        beta      = params[1:]
        xb = intercept + X @ beta
        # clip to avoid log(0)
        p = norm.cdf(xb)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        penalty = ridge * float(np.sum(beta ** 2))
        return -float(ll) + penalty

    def _fit_probit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fit probit; return (xb_hat, params, log-likelihood)."""
        n_feat = X.shape[1]
        # scale X
        self.sel_mu_ = X.mean(axis=0)
        self.sel_sd_ = X.std(axis=0)
        self.sel_sd_[self.sel_sd_ == 0] = 1.0
        Xs = (X - self.sel_mu_) / self.sel_sd_

        init = np.zeros(n_feat + 1)
        bounds = [(-10, 10)] + [(-10, 10)] * n_feat
        res = minimize(
            self._probit_neg_ll, init, args=(Xs, y.astype(float), self.ridge),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-10},
        )
        self.sel_intercept_ = float(res.x[0])
        self.sel_coef_      = res.x[1:]
        self.ll_sel_        = float(-res.fun)
        xb = self.sel_intercept_ + Xs @ self.sel_coef_

        # standard errors via Hessian inverse
        try:
            H = np.asarray(res.hess_inv.todense()
                           if hasattr(res.hess_inv, 'todense')
                           else res.hess_inv, dtype=float)
            se = np.sqrt(np.clip(np.diag(H), 0, None))
            self.sel_se_ = se
            self.sel_t_  = res.x / np.where(se > 0, se, np.nan)
            self.sel_p_  = 2 * norm.sf(np.abs(self.sel_t_))
        except Exception:
            self.sel_se_ = np.full(len(res.x), np.nan)
            self.sel_t_  = np.full(len(res.x), np.nan)
            self.sel_p_  = np.full(len(res.x), np.nan)

        return xb, res.x, self.ll_sel_

    # ── Step 2: OLS with IMR ─────────────────────────────────────────────
    def _fit_outcome(self, X: np.ndarray, B: np.ndarray, xb_sel: np.ndarray):
        """OLS of B on (X, IMR) for travellers only; xb_sel is Z'gamma_hat."""
        phi = norm.pdf(xb_sel)
        Phi = norm.cdf(xb_sel)
        imr = phi / np.clip(Phi, 1e-10, None)

        # Augment X with IMR
        self.out_mu_ = X.mean(axis=0)
        self.out_sd_ = X.std(axis=0)
        self.out_sd_[self.out_sd_ == 0] = 1.0
        Xs = (X - self.out_mu_) / self.out_sd_
        Xs_aug = np.column_stack([Xs, imr])

        ones = np.ones((len(Xs_aug), 1))
        A    = np.hstack([ones, Xs_aug])

        # OLS via lstsq
        params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        self.out_intercept_ = float(params[0])
        self.out_coef_      = params[1:X.shape[1] + 1]
        self.imr_coef_      = float(params[-1])

        yhat       = A @ params
        resid      = B - yhat
        self.sigma_ = float(np.std(resid))
        self.ll_out_ = float(
            -0.5 * len(B) * np.log(2 * np.pi * self.sigma_ ** 2)
            - 0.5 * np.sum(resid ** 2) / self.sigma_ ** 2
        )

        # Standard errors (OLS, ignoring generated-regressor correction)
        n, k = A.shape
        s2 = np.sum(resid ** 2) / max(n - k, 1)
        try:
            cov = s2 * np.linalg.inv(A.T @ A)
            se  = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)

        self.out_se_ = se
        self.out_t_  = params / np.where(se > 0, se, np.nan)
        self.out_p_  = 2 * norm.sf(np.abs(self.out_t_))

    # ── Public fit method ────────────────────────────────────────────────
    def fit(
        self,
        X_sel: np.ndarray,
        X_out: np.ndarray,
        B: np.ndarray,
        participated: np.ndarray,
        sel_features: list = None,
        out_features: list = None,
    ) -> 'HeckmanModel':
        """
        Fit the two-step Heckman model.

        Parameters
        ----------
        X_sel        : (N, p) features for selection equation (all persons)
        X_out        : (N, q) features for outcome equation (all persons)
        B            : (N,)   budget hours (0 for non-travellers)
        participated : (N,)   binary 0/1
        sel_features : column names for sel equation (for reporting)
        out_features : column names for outcome equation (for reporting)
        """
        participated = np.asarray(participated, dtype=float)
        B            = np.asarray(B, dtype=float)
        self.n_      = len(B)
        self.n_sel_  = int(participated.sum())
        self.participation_rate_ = float(participated.mean())
        self.sel_features_ = sel_features or [f'x{i}' for i in range(X_sel.shape[1])]
        self.out_features_ = out_features or [f'z{i}' for i in range(X_out.shape[1])]

        # Step 1: probit on full sample
        xb_all, _, _ = self._fit_probit(X_sel, participated)

        # Step 2: OLS with IMR on travellers only
        mask = participated == 1
        xb_travellers = xb_all[mask]
        self._fit_outcome(X_out[mask], B[mask], xb_travellers)

        # AIC/BIC (combined)
        n_params = len(self.sel_coef_) + 1 + len(self.out_coef_) + 2  # +intercepts+sigma
        total_ll = self.ll_sel_ + self.ll_out_
        self.aic_ = float(2 * n_params - 2 * total_ll)
        self.bic_ = float(n_params * np.log(max(self.n_, 1)) - 2 * total_ll)

        imr_p = float(self.out_p_[-1]) if self.out_p_ is not None else np.nan
        logger.info(
            f"  Heckman fitted: n={self.n_}, travellers={self.n_sel_}  "
            f"({100*self.participation_rate_:.1f}%)  "
            f"IMR coef={self.imr_coef_:.3f}  p={imr_p:.3f}  "
            f"LL_sel={self.ll_sel_:.1f}  LL_out={self.ll_out_:.1f}  "
            f"AIC={self.aic_:.1f}"
        )
        self.converged_ = True
        return self

    # ── Prediction methods ───────────────────────────────────────────────
    def _xb_sel(self, X_sel: np.ndarray) -> np.ndarray:
        """Compute Z'gamma (scaled) for new data."""
        Xs = (X_sel - self.sel_mu_) / self.sel_sd_
        return self.sel_intercept_ + Xs @ self.sel_coef_

    def _xb_out(self, X_out: np.ndarray) -> np.ndarray:
        """Compute X'beta (scaled) for new data (without IMR term)."""
        Xs = (X_out - self.out_mu_) / self.out_sd_
        return self.out_intercept_ + Xs @ self.out_coef_

    def predict_participation(self, X_sel: np.ndarray) -> np.ndarray:
        """P(travel) = Phi(Z'gamma) for new persons."""
        return norm.cdf(self._xb_sel(X_sel))

    def predict_conditional_budget(self, X_out: np.ndarray, X_sel: np.ndarray) -> np.ndarray:
        """E[B | travel, X] = X'beta + imr_coef * lambda(Z'gamma)."""
        xb     = self._xb_out(X_out)
        xb_sel = self._xb_sel(X_sel)
        phi    = norm.pdf(xb_sel)
        Phi    = norm.cdf(np.clip(xb_sel, -8, 8))
        imr    = phi / np.clip(Phi, 1e-10, None)
        return xb + self.imr_coef_ * imr

    def predict_unconditional_budget(self, X_out: np.ndarray, X_sel: np.ndarray) -> np.ndarray:
        """E[B | X] = P(travel) * E[B | travel, X] (marginal expected budget)."""
        p_travel = self.predict_participation(X_sel)
        cond_b   = self.predict_conditional_budget(X_out, X_sel)
        return p_travel * np.clip(cond_b, 0, None)

    def predict(self, X_out: np.ndarray, X_sel: np.ndarray) -> np.ndarray:
        """Alias for predict_unconditional_budget — expected B for all persons."""
        return self.predict_unconditional_budget(X_out, X_sel)


# ---------------------------------------------------------------------------
# 3.  BUDGET TOBIT TYPE-I  (thin wrapper for clarity)
# ---------------------------------------------------------------------------

class BudgetTobitModel:
    """
    Tobit Type-I for total out-of-home budget B, left-censored at 0.

    Includes non-travellers (B=0) in estimation so the model jointly
    identifies participation probability and conditional duration.

    P(travel)   = Phi(X'beta / sigma)
    E[B | B>0]  = X'beta + sigma * phi(X'beta/sigma) / Phi(X'beta/sigma)
    E[B]        = P(travel) * E[B | B>0]
    """

    def __init__(self, ridge: float = 0.01):
        self.ridge   = ridge
        self._tobit  = None
        self.features_     = None
        self.n_            = None
        self.participation_rate_ = None
        self.aic_          = None
        self.bic_          = None
        self.ll_           = None
        self.converged_    = None

    def fit(
        self,
        X: np.ndarray,
        B: np.ndarray,
        participated: np.ndarray,
        features: list = None,
    ) -> 'BudgetTobitModel':
        B_full = np.asarray(B, dtype=float)
        self.n_ = len(B_full)
        self.participation_rate_ = float(np.mean(np.asarray(participated) > 0))
        self.features_ = features or [f'x{i}' for i in range(X.shape[1])]

        self._tobit = TobitModel(lo=0.0, hi=np.inf, log_y=False, ridge=self.ridge)
        self._tobit.fit(X, B_full)
        self._tobit.feature_names_ = self.features_

        self.ll_       = self._tobit.ll_
        n_params       = X.shape[1] + 2   # coefs + intercept + log_sigma
        self.aic_      = float(2 * n_params - 2 * self.ll_)
        self.bic_      = float(n_params * np.log(max(self.n_, 1)) - 2 * self.ll_)
        self.converged_ = self._tobit.converged_

        sigma = self._tobit.sigma_
        xb    = self._tobit.predict(X)
        phi   = norm.pdf(xb / sigma)
        Phi   = np.clip(norm.cdf(xb / sigma), 1e-10, None)
        pred_prate = float(Phi.mean())
        logger.info(
            f"  Tobit-I fitted: n={self.n_}  "
            f"obs_prate={100*self.participation_rate_:.1f}%  "
            f"pred_prate={100*pred_prate:.1f}%  "
            f"sigma={sigma:.2f}  LL={self.ll_:.1f}  AIC={self.aic_:.1f}"
        )
        return self

    def predict_participation(self, X: np.ndarray) -> np.ndarray:
        """P(travel) = Phi(X'beta / sigma)."""
        sigma = self._tobit.sigma_
        xb    = self._tobit.predict(X)
        return norm.cdf(xb / sigma)

    def predict_conditional_budget(self, X: np.ndarray) -> np.ndarray:
        """E[B | B>0] = X'beta + sigma*phi(X'beta/sigma)/Phi(X'beta/sigma)."""
        sigma = self._tobit.sigma_
        xb    = self._tobit.predict(X)
        z     = xb / sigma
        phi   = norm.pdf(z)
        Phi   = np.clip(norm.cdf(z), 1e-10, None)
        return xb + sigma * phi / Phi

    def predict(self, X: np.ndarray) -> np.ndarray:
        """E[B] = P(travel) * E[B | B>0]  (unconditional expected budget)."""
        p = self.predict_participation(X)
        c = self.predict_conditional_budget(X)
        return p * np.clip(c, 0, None)

    @property
    def coef_(self):
        return self._tobit.coef_

    @property
    def intercept_(self):
        return self._tobit.intercept_

    @property
    def sigma(self):
        return self._tobit.sigma_

    @property
    def std_err_(self):
        return self._tobit.std_err_

    @property
    def t_stat_(self):
        return self._tobit.t_stat_

    @property
    def p_value_(self):
        return self._tobit.p_value_


# ---------------------------------------------------------------------------
# 4.  ESTIMATION  —  joint models
# ---------------------------------------------------------------------------

def _build_joint_X(
    df: pd.DataFrame,
    features: list,
) -> Tuple[np.ndarray, list]:
    """
    Build feature matrix for joint models.  Adds ptype dummies, fills NaN.
    Returns (X_array, actual_column_list).
    """
    df = _augment_day_features(df.copy())

    ptypes = sorted(df['ptype'].dropna().unique())
    for pt in ptypes:
        df[f'ptype_{pt}'] = (df['ptype'] == pt).astype(float)
    ptype_cols = [f'ptype_{pt}' for pt in ptypes]

    all_feats = list(dict.fromkeys(features + ptype_cols))
    avail = [c for c in all_feats if c in df.columns]
    if not avail:
        return np.ones((len(df), 1)), ['(intercept)']

    arrays = []
    for c in avail:
        arrays.append(pd.to_numeric(df[c], errors='coerce').fillna(0).values.astype(float))
    return np.column_stack(arrays), avail


def estimate_joint_models(
    person_day_full: pd.DataFrame,
    feature_override: list = None,
) -> dict:
    """
    Estimate both Tobit-I and Heckman models on the full person-day dataset
    (including non-travellers with B=0).

    Parameters
    ----------
    person_day_full : from aggregate_person_day_with_nontravellers()
    feature_override : override feature list (e.g. from GA)

    Returns
    -------
    dict with keys:
        'tobit1'  : BudgetTobitModel
        'heckman' : HeckmanModel
        'features_out' : list of feature column names used
        'features_sel' : list of selection feature column names
        'n_total'      : total persons in sample
        'n_travellers' : travellers
        'participation_rate' : float
    """
    df = person_day_full.copy()

    # Fill derived features
    df = _augment_day_features(df)
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            df[col] = 0
    if 'n_total' not in df.columns:
        df['n_total'] = df[[c for c in ACTIVITY_COLS if c in df.columns]].sum(axis=1)
    if 'n_tours' not in df.columns:
        df['n_tours'] = df.get('n_total', 0)

    # Feature set
    _pf = feature_override if feature_override else PERSON_FEATURES
    derived = ['n_mandatory', 'n_discretionary', 'mandatory_share',
               'escort_share', 'vehs_per_person', 'income_per_person']
    out_feats = ACTIVITY_COLS + derived + _pf + ['n_total', 'n_tours']
    sel_feats = list(dict.fromkeys(_pf + SELECTION_EXTRA))

    X_out, out_cols = _build_joint_X(df, out_feats)
    X_sel, sel_cols = _build_joint_X(df, sel_feats)

    B             = df['B'].fillna(0).values.astype(float)
    participated  = df['participated'].fillna(0).values.astype(float)

    n_total    = len(df)
    n_trav     = int(participated.sum())
    prate      = float(participated.mean())

    logger.info(f"  Estimating Tobit-I on {n_total:,} persons  "
                f"({n_trav:,} travellers, {100*prate:.1f}%)  "
                f"features={len(out_cols)}")

    # ── Tobit Type-I ──────────────────────────────────────────────────────
    tobit1 = BudgetTobitModel()
    tobit1.fit(X_out, B, participated, features=out_cols)

    # ── Heckman ───────────────────────────────────────────────────────────
    logger.info(f"  Estimating Heckman: sel_feats={len(sel_cols)}  out_feats={len(out_cols)}")
    heckman = HeckmanModel()
    heckman.fit(X_sel, X_out, B, participated,
                sel_features=sel_cols, out_features=out_cols)

    return {
        'tobit1':            tobit1,
        'heckman':           heckman,
        'features_out':      out_cols,
        'features_sel':      sel_cols,
        'n_total':           n_total,
        'n_travellers':      n_trav,
        'participation_rate': prate,
    }


# ---------------------------------------------------------------------------
# 5.  COMPARISON  —  metrics vs two-stage
# ---------------------------------------------------------------------------

def compare_models(
    tobit1: BudgetTobitModel,
    heckman: HeckmanModel,
    two_stage_models: dict,
    person_day_full: pd.DataFrame,
) -> dict:
    """
    Compute comparative metrics for all three approaches:
      - Tobit Type-I
      - Heckman
      - Sequential two-stage (Stage-1 ordLogit + Stage-2 Tobit start/end)

    Parameters
    ----------
    tobit1, heckman : estimated joint models
    two_stage_models : output of estimate_day_models() from stage2_day_duration
    person_day_full  : full dataset including non-travellers

    Returns
    -------
    dict of DataFrames / scalars for reporting.
    """
    df = _augment_day_features(person_day_full.copy())

    # --- Build feature matrices identical to estimation ---
    # Reuse the same feature lists from the fitted models
    def _reuse_X(fitted_features, df):
        arrays = []
        for c in fitted_features:
            if c in df.columns:
                arrays.append(pd.to_numeric(df[c], errors='coerce').fillna(0).values.astype(float))
            else:
                arrays.append(np.zeros(len(df), dtype=float))
        return np.column_stack(arrays) if arrays else np.ones((len(df), 1))

    X_out = _reuse_X(tobit1.features_, df)
    X_sel = _reuse_X(heckman.sel_features_, df)

    B_obs     = df['B'].fillna(0).values.astype(float)
    travelled = (df.get('participated', pd.Series(1, index=df.index))).fillna(0).values.astype(float)

    # ── Predictions ───────────────────────────────────────────────────────
    pred_t1    = tobit1.predict(X_out)
    pred_hk    = heckman.predict(X_out, X_sel)
    p_trav_t1  = tobit1.predict_participation(X_out)
    p_trav_hk  = heckman.predict_participation(X_sel)

    # Two-stage budget prediction (lognormal model on travellers)
    pred_2s = np.full(len(df), np.nan)
    if two_stage_models and 'budget' in two_stage_models:
        try:
            bm    = two_stage_models['budget']['model']
            feats = two_stage_models['budget']['features']
            X2    = _reuse_X(feats, df)
            # Normalise with the model's own mu/sd
            Xs2   = (X2 - bm.mu_) / bm.sd_
            raw   = bm.intercept_ + Xs2 @ bm.coef_
            pred_2s = np.exp(raw + 0.5 * bm.sigma_ ** 2)
        except Exception as e:
            logger.warning(f"  Two-stage predict failed: {e}")

    # ── Metrics function ──────────────────────────────────────────────────
    def _metrics(pred, y_obs, participated, label, p_trav=None):
        mask_t = participated == 1
        # Participation — use model's P(travel) if available, else threshold budget
        if p_trav is not None:
            pred_bin = (p_trav > 0.5).astype(float)
        else:
            pred_bin  = (pred > 0.1).astype(float)   # crude threshold for two-stage
        acc       = float(np.mean(pred_bin == participated))
        # Budget (travellers only)
        y_t       = y_obs[mask_t]
        p_t       = pred[mask_t]
        rmse_cond = float(np.sqrt(np.mean((y_t - p_t) ** 2))) if mask_t.any() else np.nan
        mae_cond  = float(np.mean(np.abs(y_t - p_t))) if mask_t.any() else np.nan
        corr_cond = float(np.corrcoef(y_t, p_t)[0, 1]) if mask_t.sum() > 1 else np.nan
        # Budget (all)
        rmse_all  = float(np.sqrt(np.mean((y_obs - pred) ** 2)))
        mae_all   = float(np.mean(np.abs(y_obs - pred)))
        return {
            'model': label,
            'participation_acc':   round(acc,  4),
            'budget_rmse_travellers': round(rmse_cond, 3),
            'budget_mae_travellers':  round(mae_cond,  3),
            'budget_corr_travellers': round(corr_cond, 4),
            'budget_rmse_all':     round(rmse_all, 3),
            'budget_mae_all':      round(mae_all,  3),
        }

    metrics_rows = [
        _metrics(pred_t1,   B_obs, travelled, 'Tobit-I',         p_trav=p_trav_t1),
        _metrics(pred_hk,   B_obs, travelled, 'Heckman',          p_trav=p_trav_hk),
    ]
    if not np.all(np.isnan(pred_2s)):
        metrics_rows.append(_metrics(pred_2s, B_obs, travelled, 'Two-stage (seq)'))

    metrics_df = pd.DataFrame(metrics_rows)

    # ── Coefficient tables ─────────────────────────────────────────────────
    def _coef_table(features, coef, intercept, se=None, tstat=None, pval=None):
        rows = [{'param': '(intercept)', 'coef': round(intercept, 4),
                 'se': round(float(se[0]), 4) if se is not None else np.nan,
                 't': round(float(tstat[0]), 3) if tstat is not None else np.nan,
                 'p': round(float(pval[0]), 4) if pval is not None else np.nan}]
        for i, nm in enumerate(features):
            rows.append({
                'param': nm,
                'coef': round(float(coef[i]), 4),
                'se':   round(float(se[i + 1]), 4) if se is not None else np.nan,
                't':    round(float(tstat[i + 1]), 3) if tstat is not None else np.nan,
                'p':    round(float(pval[i + 1]), 4) if pval is not None else np.nan,
            })
        return pd.DataFrame(rows)

    coef_tobit1 = _coef_table(
        tobit1.features_,
        tobit1.coef_,
        tobit1.intercept_,
        tobit1.std_err_,
        tobit1.t_stat_,
        tobit1.p_value_,
    )

    coef_heck_sel = _coef_table(
        heckman.sel_features_,
        heckman.sel_coef_,
        heckman.sel_intercept_,
        heckman.sel_se_,
        heckman.sel_t_,
        heckman.sel_p_,
    )
    # outcome coefs: out_coef_ + IMR
    hk_out_feats = list(heckman.out_features_) + ['IMR (lambda)']
    hk_out_coefs = np.append(heckman.out_coef_, heckman.imr_coef_)
    coef_heck_out = _coef_table(
        hk_out_feats,
        hk_out_coefs,
        heckman.out_intercept_,
        heckman.out_se_,
        heckman.out_t_,
        heckman.out_p_,
    )

    # ── IMR significance (used for recommendation) ─────────────────────────
    imr_p = float(heckman.out_p_[-1]) if heckman.out_p_ is not None else np.nan

    return {
        'metrics':         metrics_df,
        'coef_tobit1':     coef_tobit1,
        'coef_heck_sel':   coef_heck_sel,
        'coef_heck_out':   coef_heck_out,
        'imr_coef':        heckman.imr_coef_,
        'imr_p':           imr_p,
        'participation_rate': tobit1.participation_rate_,
        'n_total':         tobit1.n_,
        'aic_tobit1':      tobit1.aic_,
        'aic_heckman':     heckman.aic_,
        'bic_tobit1':      tobit1.bic_,
        'bic_heckman':     heckman.bic_,
        'pred_tobit1':     pred_t1,
        'pred_heckman':    pred_hk,
        'pred_twostage':   pred_2s,
        'p_trav_tobit1':   p_trav_t1,
        'p_trav_heckman':  p_trav_hk,
        'B_obs':           B_obs,
        'participated':    travelled,
    }


# ---------------------------------------------------------------------------
# 6.  PLOTS
# ---------------------------------------------------------------------------

def plot_joint_comparison(comparison: dict, save_dir: str) -> None:
    """
    Generate comparison figures for joint vs two-stage models.

    Figures produced
    ----------------
    joint_participation_comparison.png
        - Observed vs predicted participation rate per model
        - Predicted vs observed budget distribution (travellers only)
    joint_budget_scatter.png
        - Scatter predicted vs observed B for Tobit-I and Heckman
    joint_coefficient_imr.png
        - Heckman IMR coefficient bar with CI and p-value annotation
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)

    B_obs      = comparison['B_obs']
    parti      = comparison['participated']
    pred_t1    = comparison['pred_tobit1']
    pred_hk    = comparison['pred_heckman']
    pred_2s    = comparison['pred_twostage']
    prate      = comparison['participation_rate']
    p_trav_t1  = comparison['p_trav_tobit1']
    p_trav_hk  = comparison['p_trav_heckman']

    # ── Figure 1: participation bar + budget distribution ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

    # Panel A: participation rates
    ax = axes[0]
    obs_rate   = float(np.mean(parti))
    t1_rate    = float(np.mean(p_trav_t1))
    hk_rate    = float(np.mean(p_trav_hk))
    labels     = ['Observed', 'Tobit-I', 'Heckman']
    rates      = [obs_rate * 100, t1_rate * 100, hk_rate * 100]
    colors_bar = ['#2C6B9C', '#E07030', '#27AE60']
    bars = ax.bar(labels, rates, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Travel participation rate (%)')
    ax.set_title('Travel participation rate\nobserved vs predicted', fontsize=10)
    ax.set_ylim(0, max(rates) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: budget distribution (kernel density, travellers only)
    ax2 = axes[1]
    mask_t = parti == 1
    b_obs_t = B_obs[mask_t]
    from scipy.stats import gaussian_kde
    for pred, lbl, col in [
        (pred_t1, 'Tobit-I',  '#E07030'),
        (pred_hk, 'Heckman',  '#27AE60'),
    ]:
        b_pred_t = pred[mask_t]
        try:
            kde_p = gaussian_kde(b_pred_t[np.isfinite(b_pred_t) & (b_pred_t > 0)])
            xg    = np.linspace(0, MAX_BUDGET, 200)
            ax2.plot(xg, kde_p(xg), color=col, lw=2, label=lbl)
        except Exception:
            pass
    # observed
    try:
        kde_o = gaussian_kde(b_obs_t[np.isfinite(b_obs_t) & (b_obs_t > 0)])
        xg    = np.linspace(0, MAX_BUDGET, 200)
        ax2.plot(xg, kde_o(xg), color='#2C6B9C', lw=2, ls='--', label='Observed')
    except Exception:
        pass
    ax2.set_xlabel('Out-of-home budget B (hours)')
    ax2.set_ylabel('Density')
    ax2.set_title('Budget distribution — travellers\n(predicted vs observed)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, MAX_BUDGET)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'plots', 'joint_participation_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2: scatter observed vs predicted B ─────────────────────────
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    for ax, pred, lbl, col in [
        (axes2[0], pred_t1, 'Tobit-I',  '#E07030'),
        (axes2[1], pred_hk, 'Heckman',  '#27AE60'),
    ]:
        y_t = B_obs[mask_t]
        p_t = pred[mask_t]
        lim = MAX_BUDGET
        ax.scatter(p_t, y_t, alpha=0.15, s=5, color=col, rasterized=True)
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6)
        # metrics
        rmse = np.sqrt(np.mean((y_t - p_t) ** 2))
        corr = np.corrcoef(y_t, p_t)[0, 1] if len(y_t) > 1 else 0
        ax.text(0.05, 0.93, f'RMSE={rmse:.2f}h  r={corr:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(fc='white', alpha=0.8, ec='none'))
        ax.set_xlabel(f'{lbl} predicted B (hrs)')
        ax.set_ylabel('Observed B (hrs)')
        ax.set_title(f'{lbl}: predicted vs observed budget', fontsize=10)
        ax.set_xlim(0, lim);  ax.set_ylim(0, lim)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'plots', 'joint_budget_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 3: IMR coefficient with CI ─────────────────────────────────
    imr_coef = comparison['imr_coef']
    imr_p    = comparison['imr_p']
    # We need the SE of the IMR coef — extract from heckman.out_se_[-1]
    # This is stored in the comparison dict as part of heckman, but we only
    # have the numeric comparison dict here.  Use imr_p to recover approximately.
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='white')
    color   = '#E74C3C' if imr_p < 0.05 else '#95A5A6'
    ax.bar(['IMR coef (rho*sigma)'], [imr_coef], color=color, alpha=0.85, edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    sig_label = '*** p<0.001' if imr_p < 0.001 else \
                '** p<0.01'   if imr_p < 0.01  else \
                '* p<0.05'    if imr_p < 0.05  else \
                f'n.s. (p={imr_p:.3f})'
    ax.set_title(
        f'Heckman IMR coefficient: {imr_coef:.3f}  {sig_label}\n'
        f'{"Selection bias present — use Heckman" if imr_p < 0.05 else "No significant selection bias — Tobit-I sufficient"}',
        fontsize=9
    )
    ax.set_ylabel('Coefficient value')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'plots', 'joint_coefficient_imr.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info("  Joint model comparison plots saved.")


# ---------------------------------------------------------------------------
# 7.  MODEL SELECTION RECOMMENDATION
# ---------------------------------------------------------------------------

def joint_model_recommendation(comparison: dict) -> str:
    """
    Return a human-readable HTML string recommending which model to use.

    Decision tree
    -------------
    1. If non-traveller rate < 5%:
       => Sequential two-stage is fine.  Selection bias is negligible.
    2. Else if IMR p < 0.05:
       => Heckman is preferred (significant selection bias).
    3. Else if non-traveller rate >= 5%:
       => Tobit-I (simpler, joint, no selection bias correction needed).
    """
    prate  = comparison.get('participation_rate', 1.0)
    nt_pct = 100 * (1 - prate)
    imr_p  = comparison.get('imr_p', np.nan)
    aic_t1 = comparison.get('aic_tobit1', np.nan)
    aic_hk = comparison.get('aic_heckman', np.nan)
    bic_t1 = comparison.get('bic_tobit1',  np.nan)
    bic_hk = comparison.get('bic_heckman', np.nan)

    if nt_pct < 5:
        rec   = 'Sequential two-stage (current)'
        color = '#27AE60'
        reason = (
            f'Non-traveller rate is low ({nt_pct:.1f}%) — selection bias is negligible. '
            'The two-stage approach (Stage-1 ordered-logit + Stage-2 Tobit) is appropriate '
            'and simpler to interpret.'
        )
    elif (not np.isnan(imr_p)) and imr_p < 0.05:
        rec   = 'Heckman selection model'
        color = '#E74C3C'
        reason = (
            f'Non-traveller rate is {nt_pct:.1f}% and the IMR coefficient is significant '
            f'(p={imr_p:.3f}).  This indicates that the unobservables driving the decision '
            'to travel also affect duration — selection bias is present.  '
            'The Heckman model corrects for this and gives consistent estimates.'
        )
    else:
        rec   = 'Tobit Type-I (joint participation + duration)'
        color = '#E07030'
        reason = (
            f'Non-traveller rate is {nt_pct:.1f}% but the IMR coefficient is not significant '
            f'(p={"N/A" if np.isnan(imr_p) else f"{imr_p:.3f}"}).  '
            'Tobit-I jointly models participation and duration without the extra complexity '
            'of a separate selection equation, and is more parsimonious than Heckman.'
        )

    aic_row = (
        f'<tr><td>AIC</td>'
        f'<td>{aic_t1:.1f}</td><td>{aic_hk:.1f}</td></tr>'
        if not np.isnan(aic_t1) else ''
    )
    bic_row = (
        f'<tr><td>BIC</td>'
        f'<td>{bic_t1:.1f}</td><td>{bic_hk:.1f}</td></tr>'
        if not np.isnan(bic_t1) else ''
    )

    return f"""
<div class="recommendation" style="border:2px solid {color};padding:12px;margin:12px 0;
     border-radius:6px;background:#f9f9f9;">
<h3 style="color:{color};margin-top:0">Recommended approach: {rec}</h3>
<p>{reason}</p>
<table border="1" cellpadding="5" style="border-collapse:collapse;font-size:0.9em">
  <tr><th>Metric</th><th>Tobit-I</th><th>Heckman</th></tr>
  {aic_row}
  {bic_row}
  <tr><td>Non-traveller rate</td>
      <td colspan="2">{nt_pct:.1f}%</td></tr>
  <tr><td>IMR p-value</td>
      <td colspan="2">{'N/A' if np.isnan(imr_p) else f'{imr_p:.4f}'}</td></tr>
</table>
</div>
<div style="background:#fffde7;padding:10px;margin:8px 0;border-left:4px solid #f39c12">
<b>When to use each approach:</b>
<ul style="margin:4px 0">
  <li><b>Sequential two-stage</b>: when non-travellers are absent or rare (&lt;5%).
      Simpler estimation, no distributional assumptions on participation.
      Current pipeline default.</li>
  <li><b>Tobit Type-I</b>: when non-travellers are present (&ge;5%) and IMR is
      not significant.  One model, directly gives P(travel) and E[B|travel].</li>
  <li><b>Heckman selection</b>: when non-travellers are present AND IMR is
      significant (p&lt;0.05).  Use different variables in selection vs outcome
      equations (exclusion restriction) for proper identification.</li>
</ul>
</div>
"""


# ---------------------------------------------------------------------------
# 8.  HTML REPORT
# ---------------------------------------------------------------------------

def write_joint_html_report(
    comparison: dict,
    save_dir: str,
    two_stage_models: dict = None,
) -> str:
    """
    Write a self-contained HTML report comparing joint vs two-stage models.
    Returns the path to the saved file.
    """
    def _img(fname):
        path = os.path.join(save_dir, 'plots', fname)
        if not os.path.isfile(path):
            return ''
        import base64
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return (f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:100%;margin:8px 0">')

    def _coef_html(df, title, highlight_imr=False):
        if df is None or df.empty:
            return f'<p><em>No coefficients available for {title}.</em></p>'
        rows = []
        for _, r in df.iterrows():
            p = r.get('p', np.nan)
            t = r.get('t', np.nan)
            is_insig = (not np.isnan(p)) and abs(float(p)) >= 0.1
            is_sig   = (not np.isnan(p)) and abs(float(p)) < 0.05
            is_imr   = str(r['param']) == 'IMR (lambda)'
            row_style = 'background:#fff9c4' if (highlight_imr and is_imr) else ''
            name_style = 'color:#c0392b' if is_insig else ('color:#27ae60;font-weight:bold' if is_sig else '')
            p_str    = f'{p:.4f}' if not np.isnan(p) else '—'
            t_str    = f'{t:.3f}' if not np.isnan(t) else '—'
            rows.append(
                f'<tr style="{row_style}">'
                f'<td style="{name_style}">{r["param"]}</td>'
                f'<td style="text-align:right">{r["coef"]:.4f}</td>'
                f'<td style="text-align:right">{r.get("se", np.nan):.4f}</td>'
                f'<td style="text-align:right">{t_str}</td>'
                f'<td style="text-align:right">{p_str}</td>'
                f'</tr>'
            )
        return f'''
<h4>{title}</h4>
<table border="1" cellpadding="4" style="border-collapse:collapse;font-size:0.85em;width:100%">
  <tr style="background:#f0f0f0">
    <th>Parameter</th><th>Coef</th><th>SE</th><th>t/z</th><th>p-value</th>
  </tr>
  {''.join(rows)}
</table>
<p style="font-size:0.8em;color:#666">
  <span style="color:#c0392b">Red</span> = |t| &lt; 1.65 (not significant at 10%).
  <span style="color:#27ae60;font-weight:bold">Green bold</span> = p &lt; 0.05.
  <span style="background:#fff9c4;padding:2px">Yellow</span> = IMR (lambda) coefficient.
</p>'''

    metrics_df = comparison.get('metrics', pd.DataFrame())
    metrics_html = metrics_df.to_html(index=False, float_format=lambda x: f'{x:.4f}',
                                       classes='metrics-table', border=1) if not metrics_df.empty else ''

    recommendation_html = joint_model_recommendation(comparison)
    nt_pct = 100 * (1 - comparison.get('participation_rate', 1.0))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stage 2: Joint Participation + Duration Model Comparison</title>
<style>
  body  {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; }}
  h1    {{ color: #2c3e50; }}
  h2    {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 4px; }}
  h3    {{ color: #34495e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 8px; }}
  th    {{ background: #ecf0f1; }}
  .metrics-table td {{ text-align: right; }}
  .metrics-table td:first-child {{ text-align: left; }}
  .highlight {{ background: #eaf2ff; padding: 10px; border-radius: 4px; margin: 8px 0; }}
</style>
</head>
<body>
<h1>Stage 2: Joint Participation + Duration<br>
    <small>Tobit-I vs Heckman vs Sequential Two-Stage</small></h1>

<div class="highlight">
<b>Overview:</b>
The current pipeline uses a <em>sequential two-stage</em> approach — Stage 1 (ordered logit
for activity frequency) followed by Stage 2 (Tobit models for time window).
This report evaluates two joint alternatives that model participation and duration jointly:
<ol>
<li><b>Tobit Type-I</b>: left-censored budget B at zero; includes non-travellers</li>
<li><b>Heckman selection</b>: probit selection + OLS outcome with inverse Mills ratio (IMR)</li>
</ol>
Non-traveller rate in HHTS estimation sample: <b>{nt_pct:.1f}%</b>
</div>

<h2>1. Recommendation</h2>
{recommendation_html}

<h2>2. Comparative Fit Metrics</h2>
{metrics_html}
{_img('joint_participation_comparison.png')}

<h2>3. Budget Prediction Scatter</h2>
{_img('joint_budget_scatter.png')}

<h2>4. Heckman IMR Coefficient</h2>
<p>If the IMR (inverse Mills ratio) coefficient is statistically significant,
the decision to travel is correlated with how long the trip lasts — indicating
selection bias that the sequential two-stage approach does not account for.
The Heckman model corrects for this via the IMR term.</p>
{_img('joint_coefficient_imr.png')}

<h2>5. Tobit-I Coefficients</h2>
{_coef_html(comparison.get('coef_tobit1'), 'Tobit-I: budget B (left-censored at 0)')}

<h2>6. Heckman Coefficients</h2>
{_coef_html(comparison.get('coef_heck_sel'), 'Selection equation (probit): P(travel | Z)')}
{_coef_html(comparison.get('coef_heck_out'),
            'Outcome equation (OLS): E[B | travel, X, IMR]', highlight_imr=True)}

<h2>7. Technical Notes</h2>
<ul>
<li><b>Identification</b>: The Heckman model requires at least one variable in the selection
    equation not appearing in the outcome equation (exclusion restriction) for proper
    identification.  Here <code>STUDYING</code> and <code>nprd</code> (household size)
    serve as candidates — valid if they predict whether someone travels but not how long.</li>
<li><b>Tobit I consistency</b>: Tobit I is fully efficient under homoskedasticity and
    normality of errors.  If these fail (test with residual plots), consider a
    generalised Tobit or semi-parametric alternatives.</li>
<li><b>Two-step Heckman SEs</b>: Standard errors in Step 2 do not account for estimation
    error from Step 1 (generated regressor problem).  For inference, bootstrap SEs or
    MLE estimation is recommended if the Heckman model is adopted.</li>
<li><b>Prediction for synthetic population</b>: If Tobit-I is adopted, replace the
    Stage-2 budget prediction with <code>BudgetTobitModel.predict(X)</code> which gives
    E[B] for each person.  Participation is implicit (persons with predicted B≈0 are
    assigned zero activities in Stage 3).</li>
</ul>

<hr>
<p style="font-size:0.8em;color:#888">Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body>
</html>"""

    out_path = os.path.join(save_dir, 'stage2_joint_model_comparison.html')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    logger.info(f"  Joint model report → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 9.  SAVE / LOAD
# ---------------------------------------------------------------------------

def save_joint_models(models: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    logger.info(f"  Joint models saved → {path}")


def load_joint_models(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)
