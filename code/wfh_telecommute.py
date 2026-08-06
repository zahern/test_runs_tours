"""
wfh_telecommute.py  —  Work-from-home / telecommuting pre-day model
===================================================================
The `lr4` pre-day module in pipeline_model_workflow_tikz.tex. HHTS records daily
WFH (WFHDAY) and weekly frequency (WFHWEEK), and cdap.py derives `wfh` from
WFHDAY — but the SYNTHETIC population has no WFH assignment, so every synth
worker defaults to wfh=0 (no telecommuting shift on CDAP / activity frequency).

This module closes that gap:
  1. estimate_wfh_model(hhts_persons)  — fit P(WFH on a typical day | worker
     features) as a logistic regression (pure numpy IRLS) on HHTS workers, plus
     the mean WFHWEEK days for those who WFH.
  2. predict_wfh(synth, model, seed)   — assign `wfh` (0/1 Bernoulli draw) and
     an expected `WFHWEEK` onto synth workers; non-workers get wfh=0.
  3. run_wfh(hhts_persons, synth, ...)  — estimate + predict + log a summary.

Downstream: `wfh` already feeds cdap.py (a WFH worker is more likely Home /
non-mandatory) and is a candidate feature in Stage 1 (fewer work tours) and the
day-activity budget. Gated at the call site by PIPELINE_WFH.

Pure numpy/scipy — no jax/larch — so it runs and is testable in the base env.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Worker filter: WORKSTAT 1=full-time, 2=part-time (unified 2026-06 convention).
_WORKER_WORKSTAT = (1, 2)


def _worker_mask(df: pd.DataFrame) -> np.ndarray:
    if 'WORKSTAT' in df.columns:
        ws = pd.to_numeric(df['WORKSTAT'], errors='coerce').fillna(0).astype(int)
        return ws.isin(_WORKER_WORKSTAT).values
    for c in ('workstat_fulltime', 'workstat_parttime', 'mandatory_role',
              'ptype_FullTimeWorker', 'ptype_PartTimeWorker'):
        if c in df.columns:
            return (pd.to_numeric(df[c], errors='coerce').fillna(0) > 0).values
    return np.ones(len(df), dtype=bool)


def _wfh_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Feature matrix from columns present in BOTH HHTS and synth (0-filled if
    absent) so the fitted coefficients transfer. Intercept added by the model."""
    feats: list[str] = []
    cols: list[np.ndarray] = []

    def _num(name, default=0.0):
        return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)),
                             errors='coerce').fillna(default).values.astype(float)

    # income (higher → more WFH), standardised
    _inc = _num('incp', 7.0)
    cols.append((_inc - 7.0) / 4.0); feats.append('incp_z')
    # part-time (less WFH than FT baseline)
    if 'WORKSTAT' in df.columns:
        cols.append((pd.to_numeric(df['WORKSTAT'], errors='coerce').fillna(0) == 2)
                    .astype(float).values); feats.append('is_parttime')
    # female (often higher WFH)
    if 'sexp' in df.columns:
        cols.append((pd.to_numeric(df['sexp'], errors='coerce').fillna(0) == 2)
                    .astype(float).values); feats.append('is_female')
    # occupation: managers/professionals (occp 1,2) WFH more; labourers/trades less
    if 'occp' in df.columns:
        _occ = pd.to_numeric(df['occp'], errors='coerce').fillna(0)
        cols.append(_occ.isin([1, 2]).astype(float).values); feats.append('occ_prof')
        cols.append(_occ.isin([7, 8]).astype(float).values); feats.append('occ_manual')
    # industry: office/finance/IT/public WFH more; retail/health/agri less
    if 'indp' in df.columns:
        _ind = pd.to_numeric(df['indp'], errors='coerce').fillna(0)
        cols.append(_ind.isin([10, 11, 12, 13, 14]).astype(float).values); feats.append('ind_office')
        cols.append(_ind.isin([7, 8, 16, 17]).astype(float).values); feats.append('ind_frontline')
    # age band (mid-career WFH more than young/old)
    if 'age5p' in df.columns:
        _age = pd.to_numeric(df['age5p'], errors='coerce').fillna(8)
        cols.append(_age.between(6, 12).astype(float).values); feats.append('age_prime')

    if not cols:
        cols.append(np.zeros(len(df))); feats.append('_const0')
    return np.column_stack(cols), feats


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _as_binary(s: pd.Series) -> np.ndarray:
    """Robustly coerce a WFH-style column (bool / 0-1 int / 'Y'/'N' / 'True') to
    a float 0/1 array. Avoids the .map({True:1,1:1,...}) trap where Python folds
    True and 1 into a single dict key that pandas then fails to match on int64."""
    if s.dtype == bool:
        return s.astype(float).values
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().any():
        return (num.fillna(0) > 0).astype(float).values
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin(['1', 'y', 'yes', 'true', 't']).astype(float).values


class WfhModel:
    """Logistic P(WFH | features) fit by IRLS (ridge-stabilised); numpy only."""

    def __init__(self):
        self.coef_: Optional[np.ndarray] = None      # includes intercept as [0]
        self.feature_names_: list[str] = []
        self.wfhweek_mean_: float = 2.5              # mean WFH days/week among WFH-ers
        self.base_rate_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0, n_iter: int = 50):
        n, k = X.shape
        Xb = np.column_stack([np.ones(n), X])        # intercept
        w = np.zeros(k + 1)
        R = l2 * np.eye(k + 1); R[0, 0] = 0.0         # don't penalise intercept
        for _ in range(n_iter):
            p = _sigmoid(Xb @ w)
            W = np.clip(p * (1 - p), 1e-6, None)
            grad = Xb.T @ (p - y) + R @ w
            H = Xb.T @ (Xb * W[:, None]) + R
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            w_new = w - step
            if not np.all(np.isfinite(w_new)):
                break
            if np.max(np.abs(w_new - w)) < 1e-7:
                w = w_new; break
            w = w_new
        self.coef_ = w
        self.base_rate_ = float(y.mean())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            return np.full(len(X), self.base_rate_)
        Xb = np.column_stack([np.ones(len(X)), X])
        return _sigmoid(Xb @ self.coef_)


def estimate_wfh_model(hhts_persons: pd.DataFrame) -> WfhModel:
    """Fit the WFH model on HHTS workers (y = WFH on the surveyed day)."""
    df = hhts_persons
    wmask = _worker_mask(df)
    work = df.loc[wmask].copy()
    if len(work) < 50:
        logger.warning(f"WFH model: only {len(work)} workers in HHTS — using base rate")
        m = WfhModel()
        return m

    # Observed WFH today: prefer WFHDAY; else WFHWEEK>0 fraction of the week.
    if 'WFHDAY' in work.columns:
        y = _as_binary(work['WFHDAY'])
    elif 'WFHWEEK' in work.columns:
        wk = pd.to_numeric(work['WFHWEEK'], errors='coerce').fillna(0).clip(0, 7)
        # probability the surveyed weekday was a WFH day
        y = (wk / 5.0).clip(0, 1).values
        y = (np.random.default_rng(42).random(len(y)) < y).astype(float)
    else:
        logger.warning("WFH model: no WFHDAY/WFHWEEK in HHTS — base rate only")
        return WfhModel()

    X, names = _wfh_features(work)
    m = WfhModel()
    m.fit(X, y)
    m.feature_names_ = names
    if 'WFHWEEK' in work.columns:
        wk = pd.to_numeric(work['WFHWEEK'], errors='coerce').fillna(0).clip(0, 7)
        _wfh_wk = wk[wk > 0]
        m.wfhweek_mean_ = float(_wfh_wk.mean()) if len(_wfh_wk) else 2.5
    logger.info(f"WFH model fit on {len(work)} HHTS workers: base WFH rate="
                f"{m.base_rate_:.3f}, mean WFHWEEK|WFH={m.wfhweek_mean_:.1f}, "
                f"{len(names)} features")
    return m


def predict_wfh(synth: pd.DataFrame, model: WfhModel, seed: int = 42) -> pd.DataFrame:
    """Assign `wfh` (0/1) and expected `WFHWEEK` onto synth workers."""
    df = synth.copy()
    wmask = _worker_mask(df)
    p = np.zeros(len(df))
    if wmask.any():
        X, _ = _wfh_features(df.loc[wmask])
        p_workers = model.predict_proba(X)
        p[np.where(wmask)[0]] = p_workers
    rng = np.random.default_rng(seed)
    df['wfh'] = (rng.random(len(df)) < p).astype(np.int8)
    df.loc[~wmask, 'wfh'] = 0
    # Expected weekly WFH days: p(day) scaled to a 5-day week, or the fitted mean
    # for those drawn as WFH today.
    df['WFHWEEK'] = np.clip(np.round(p * 5.0), 0, 5).astype(np.int8)
    df.loc[df['wfh'] == 1, 'WFHWEEK'] = int(round(model.wfhweek_mean_))
    df.loc[~wmask, 'WFHWEEK'] = 0
    logger.info(f"WFH assigned: {int(df['wfh'].sum())}/{int(wmask.sum())} workers "
                f"WFH today ({df.loc[wmask, 'wfh'].mean()*100:.1f}%)")
    return df


def run_wfh(hhts_persons: pd.DataFrame, synth: pd.DataFrame,
            save_path: Optional[str] = None, seed: int = 42) -> pd.DataFrame:
    """Estimate on HHTS + assign onto synth. Returns synth with wfh / WFHWEEK."""
    try:
        model = estimate_wfh_model(hhts_persons)
        synth = predict_wfh(synth, model, seed=seed)
        if save_path:
            try:
                synth[['wfh', 'WFHWEEK']].describe().to_csv(save_path)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"run_wfh failed ({e}); leaving wfh unset (defaults to 0)")
    return synth


if __name__ == '__main__':
    # Self-check on synthetic data (base env, no jax/larch).
    rng = np.random.default_rng(0)
    n = 2000
    hh = pd.DataFrame({
        'WORKSTAT': rng.choice([0, 1, 2], n, p=[0.3, 0.55, 0.15]),
        'occp': rng.integers(1, 9, n), 'indp': rng.integers(1, 20, n),
        'incp': rng.integers(1, 14, n), 'sexp': rng.integers(1, 3, n),
        'age5p': rng.integers(4, 16, n),
    })
    # true WFH more likely for office/high-income
    pz = _sigmoid(-1.5 + 0.4 * (hh.incp - 7) / 4 + 0.9 * hh.indp.isin([10, 11, 12, 13]).astype(float))
    hh['WFHDAY'] = (rng.random(n) < pz).astype(int)
    hh['WFHWEEK'] = hh['WFHDAY'] * rng.integers(1, 5, n)
    m = estimate_wfh_model(hh)
    out = predict_wfh(hh.copy(), m)
    print(f"base rate={m.base_rate_:.3f}  predicted WFH share among workers="
          f"{out.loc[_worker_mask(out), 'wfh'].mean():.3f}")
    print("coef (intercept, then):", np.round(m.coef_, 3), "features:", m.feature_names_)
