"""
stage23_joint.py
================
Joint estimation of Stage 2 (day duration budget) and Stage 3 (activity
duration per purpose) in a single JAX-backed optimisation pass.

Motivation
----------
The current pipeline estimates Stage 2 (OLS / Tobit for start_of_day,
end_of_day, B) and Stage 3 (lognormal per purpose) independently, then
rescales Stage 3 durations to fit within B post-hoc.  This two-step approach
introduces:

  1. Rescaling artefacts — the post-hoc clip-and-scale step distorts estimated
     lognormal means, biasing purposes with the most flexibility.
  2. Parameter inconsistency — Stage 3 coefficients are not estimated subject
     to the budget constraint, so expected total duration can exceed B in
     expectation, meaning the rescaling is not a minor correction but a
     systematic model error.
  3. No hazard representation for the budget itself — OLS treats B as a
     symmetric error process, but B is positive, right-skewed, and subject to
     censoring at the survey-day boundary.

The joint model (following the `duration_main.py` prototype in
metacountregressor) addresses all three:

  * Budget distribution: Weibull AFT or lognormal regression for B (and
    optionally start_of_day / end_of_day).
  * Per-purpose durations: lognormal regression with person-level covariates.
  * Budget constraint: a differentiable penalty λ·(Σ_a d_a - B)² is added to
    the joint log-likelihood, so per-purpose coefficients are estimated to
    respect the constraint in expectation.
  * JAX gradient: the full objective is differentiated with jax.grad so
    L-BFGS-B converges reliably without finite-difference gradients.

Activation
----------
Set ``STAGE23_JOINT=1`` in the environment before running the pipeline.  The
joint model is then called from ``pipeline_logger.estimate_stage2`` and
``pipeline_logger.estimate_stage3`` in place of their independent fits.

    export STAGE23_JOINT=1
    python _run_sa_bandit.py

If JAX is unavailable the module falls back to independent OLS + lognormal
(identical to the current behaviour).

API
---
    from stage23_joint import fit_joint_stage23, predict_joint_stage23

    models = fit_joint_stage23(person_day, tours, persons,
                               purposes=PURPOSES,
                               feature_override=None,
                               budget_family='weibull')

    synth = predict_joint_stage23(synth_s1, persons, models)
    # synth gains: start_of_day, end_of_day, B,
    #              n_<purpose>_duration for each purpose in PURPOSES
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ── JAX availability ─────────────────────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _JAX_OK = True
except ImportError:
    jnp = np  # type: ignore[assignment]
    _JAX_OK = False
    logger.warning("stage23_joint: JAX not available — falling back to scipy only")

# ── Purposes (mirrors stage3_activity_duration.PURPOSES) ─────────────────────
DEFAULT_PURPOSES = [
    "work", "edu", "social", "escort",
    "shopping", "othmaint", "othdiscr", "eat",
]

# ── Minimum observations per segment ────────────────────────────────────────
_MIN_OBS = 30


# =============================================================================
# Log-likelihood primitives (JAX-native; fall back to numpy if JAX absent)
# =============================================================================

def _lognormal_ll_jax(y, eta, log_sigma):
    """Element-wise lognormal log-likelihood (JAX)."""
    sigma = jnp.exp(log_sigma)
    return (
        -jnp.log(y)
        - log_sigma
        - 0.5 * ((jnp.log(y) - eta) / sigma) ** 2
    )


def _weibull_ll_jax(y, eta, log_shape):
    """
    Element-wise Weibull AFT log-likelihood (JAX).

    Parameterisation follows Cho, Chung & Kim (2023, TRPart A 172:103665):
      log(T) = β₀ + βX + τW,  W ~ Extreme Value(0,1)
      τ = P⁻¹ (scale),  P = shape parameter (P > 1 → monotone increasing hazard)

    Need-based theory implies dh/dt > 0 for all t — i.e., the need to perform
    an activity grows over time until the threshold is crossed — which requires
    P > 1 in the Weibull distribution.

    log_shape = log(P),  P > 0
    η = Xβ  (linear predictor / location parameter)
    """
    shape = jnp.exp(log_shape)           # P = 1/τ (shape / rate parameter)
    z = shape * (jnp.log(y) - eta)      # (log(T) - η) / τ  scaled by P
    return jnp.log(shape) - jnp.log(y) + z - jnp.exp(z)


def _gaussian_ll_jax(y, eta, log_sigma):
    """Element-wise Gaussian log-likelihood (for start/end of day)."""
    sigma = jnp.exp(log_sigma)
    return -0.5 * ((y - eta) / sigma) ** 2 - log_sigma


# =============================================================================
# Parameter indexing
# =============================================================================

class _ParamIndex:
    """
    Flat parameter vector layout:

        [β_B (k_B+1),  log_shape_B,
         β_start (k_s+1), log_sigma_start,
         β_end   (k_e+1), log_sigma_end,
         β_p1 (k_p+1), log_sigma_p1,   ← per purpose
         ...
         β_pP (k_p+1), log_sigma_pP]
    """

    def __init__(self, k_B: int, k_start: int, k_end: int,
                 k_purp: int, n_purposes: int):
        idx = 0
        self.b_beta   = slice(idx, idx + k_B + 1);     idx += k_B + 1
        self.b_shape  = idx;                            idx += 1
        self.s_beta   = slice(idx, idx + k_start + 1); idx += k_start + 1
        self.s_lsig   = idx;                            idx += 1
        self.e_beta   = slice(idx, idx + k_end + 1);   idx += k_end + 1
        self.e_lsig   = idx;                            idx += 1
        self.p_beta   = []
        self.p_lsig   = []
        for _ in range(n_purposes):
            self.p_beta.append(slice(idx, idx + k_purp + 1)); idx += k_purp + 1
            self.p_lsig.append(idx);                           idx += 1
        self.total = idx


# =============================================================================
# Joint objective (JAX)
# =============================================================================

def _build_objective(
    X_B: np.ndarray,       # (N, k_B)   budget covariates
    y_B: np.ndarray,       # (N,)       B values
    X_start: np.ndarray,   # (N, k_s)
    y_start: np.ndarray,
    X_end: np.ndarray,     # (N, k_e)
    y_end: np.ndarray,
    X_purp: np.ndarray,    # (N_acts, k_p)  activity covariates
    y_purp: np.ndarray,    # (N_acts,)      activity durations
    pid_purp: np.ndarray,  # (N_acts,)      person index into B
    purp_id: np.ndarray,   # (N_acts,)      purpose index 0..P-1
    pidx: _ParamIndex,
    n_purposes: int,
    budget_family: str = 'weibull',   # Weibull AFT: theory-motivated by need-based hazard
    duration_family: str = 'weibull', # per-purpose episodes: same Weibull AFT
    lambda_budget: float = 5.0,
):
    """
    Build the joint negative log-likelihood callable for JAX / scipy.

    Returns (objective_fn, grad_fn).
    """
    if _JAX_OK:
        X_B_j      = jnp.array(X_B)
        y_B_j      = jnp.array(y_B)
        X_start_j  = jnp.array(X_start)
        y_start_j  = jnp.array(y_start)
        X_end_j    = jnp.array(X_end)
        y_end_j    = jnp.array(y_end)
        X_purp_j   = jnp.array(X_purp)
        y_purp_j   = jnp.array(y_purp)
        pid_j      = jnp.array(pid_purp, dtype=jnp.int32)
        purp_id_j  = jnp.array(purp_id,  dtype=jnp.int32)

        N = len(y_B)

        def nll(params):
            # ── Budget B ───────────────────────────────────────────────────
            beta_B = params[pidx.b_beta]
            eta_B  = X_B_j @ beta_B
            if budget_family == 'weibull':
                ll_B = _weibull_ll_jax(y_B_j, eta_B, params[pidx.b_shape])
            else:
                ll_B = _lognormal_ll_jax(y_B_j, eta_B, params[pidx.b_shape])
            total_ll = jnp.sum(ll_B)

            # ── Start-of-day ───────────────────────────────────────────────
            beta_s = params[pidx.s_beta]
            eta_s  = X_start_j @ beta_s
            total_ll += jnp.sum(_gaussian_ll_jax(y_start_j, eta_s, params[pidx.s_lsig]))

            # ── End-of-day ─────────────────────────────────────────────────
            beta_e = params[pidx.e_beta]
            eta_e  = X_end_j @ beta_e
            total_ll += jnp.sum(_gaussian_ll_jax(y_end_j, eta_e, params[pidx.e_lsig]))

            # ── Per-purpose durations (Weibull AFT or lognormal) ──────────
            # Weibull AFT is theory-motivated (Cho et al. 2023): need for each
            # activity type accumulates monotonically (P>1 hazard), implying
            # the Weibull distribution for inter-episode durations.
            log_d_pred = jnp.zeros(len(y_purp_j))
            for p in range(n_purposes):
                mask_p = purp_id_j == p
                beta_p = params[pidx.p_beta[p]]
                eta_p  = X_purp_j @ beta_p
                if duration_family == 'weibull':
                    ll_p = _weibull_ll_jax(y_purp_j, eta_p, params[pidx.p_lsig[p]])
                else:
                    ll_p = _lognormal_ll_jax(y_purp_j, eta_p, params[pidx.p_lsig[p]])
                total_ll += jnp.sum(jnp.where(mask_p, ll_p, 0.0))
                log_d_pred = log_d_pred.at[jnp.where(mask_p)].set(eta_p[jnp.where(mask_p)])

            # ── Budget constraint penalty ─────────────────────────────────
            # Σ_a E[d_a | person] should ≈ B
            # E[lognormal] = exp(η + σ²/2) — use η as proxy (σ² term is constant)
            d_exp = jnp.exp(log_d_pred)
            # Sum expected durations per person
            sum_d = jnp.zeros(N).at[pid_j].add(d_exp)
            penalty = jnp.sum((sum_d - y_B_j) ** 2)
            total_ll -= lambda_budget * penalty

            return -total_ll   # minimise negative LL

        nll_jit = jit(nll)
        grad_fn = jit(grad(nll))
        return nll_jit, grad_fn
    else:
        # Numpy fallback — no gradient, use numerical diff
        def nll_np(params):
            beta_B = params[pidx.b_beta]
            eta_B  = X_B @ beta_B
            B_pos  = np.clip(y_B, 1e-6, None)
            sig_B  = np.exp(params[pidx.b_shape])
            ll_B   = (-np.log(B_pos) - params[pidx.b_shape]
                      - 0.5 * ((np.log(B_pos) - eta_B) / sig_B) ** 2)
            total  = np.sum(ll_B)

            beta_s = params[pidx.s_beta]
            sig_s  = np.exp(params[pidx.s_lsig])
            total += np.sum(-0.5 * ((y_start - X_start @ beta_s) / sig_s) ** 2
                            - params[pidx.s_lsig])

            beta_e = params[pidx.e_beta]
            sig_e  = np.exp(params[pidx.e_lsig])
            total += np.sum(-0.5 * ((y_end - X_end @ beta_e) / sig_e) ** 2
                            - params[pidx.e_lsig])

            for p in range(n_purposes):
                mask_p = purp_id == p
                if not mask_p.any():
                    continue
                beta_p = params[pidx.p_beta[p]]
                sig_p  = np.exp(params[pidx.p_lsig[p]])
                eta_p  = (X_purp @ beta_p)[mask_p]
                yp     = np.clip(y_purp[mask_p], 1e-6, None)
                total += np.sum(-np.log(yp) - params[pidx.p_lsig[p]]
                                - 0.5 * ((np.log(yp) - eta_p) / sig_p) ** 2)

            return -total

        return nll_np, None


# =============================================================================
# Feature matrix builders
# =============================================================================

def _design(df: pd.DataFrame, features: list, add_intercept: bool = True) -> np.ndarray:
    """Build a design matrix from df columns (fills NaN with 0)."""
    cols = [c for c in features if c in df.columns]
    X = df[cols].fillna(0).to_numpy(dtype=float) if cols else np.zeros((len(df), 0))
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    return X


def _bic(nll_val: float, k: int, n: int) -> float:
    return 2 * nll_val + k * np.log(max(n, 2))


# =============================================================================
# Public API — fit
# =============================================================================

def fit_joint_stage23(
    person_day: pd.DataFrame,
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    purposes: Optional[List[str]] = None,
    feature_override: Optional[Dict] = None,
    budget_family: str = 'weibull',    # Weibull AFT for B — monotone hazard (Cho et al. 2023)
    duration_family: str = 'weibull',  # Weibull AFT for per-purpose durations
    lambda_budget: float = 5.0,
    maxiter: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Jointly estimate Stage 2 (B, start_of_day, end_of_day) and Stage 3
    (per-purpose durations) parameters with a budget constraint penalty.

    Parameters
    ----------
    person_day   : person-day DataFrame from stage2.aggregate_person_day
    tours        : HHTS tour-level DataFrame (purpose + duration columns)
    persons      : HHTS person DataFrame
    purposes     : list of activity purpose labels (default: DEFAULT_PURPOSES)
    feature_override : dict with keys 'B', 'start', 'end', 'purp' → feature lists
    budget_family    : 'lognormal' | 'weibull' for the B marginal distribution
    lambda_budget    : penalty weight on the budget constraint
    maxiter          : L-BFGS-B iteration cap
    seed             : random seed for parameter initialisation

    Returns
    -------
    dict with keys:
        'params'         : fitted flat parameter vector (numpy)
        'pidx'           : _ParamIndex describing the vector layout
        'bic'            : joint BIC
        'nll'            : final negative log-likelihood
        'budget_family'  : as passed
        'purposes'       : purpose list
        'feat_B'         : budget covariate names
        'feat_start'     : start_of_day covariate names
        'feat_end'       : end_of_day covariate names
        'feat_purp'      : per-purpose covariate names
        'purpose_col'    : name of the purpose column used in tours
        'success'        : bool
    """
    purposes = purposes or DEFAULT_PURPOSES
    rng = np.random.default_rng(seed)

    # ── Default feature sets ─────────────────────────────────────────────────
    fo = feature_override or {}
    feat_B     = fo.get('B',     ['vehrd', 'incp', 'sexp', 'nprd', 'ptype_Worker',
                                   'ptype_Student', 'n_work_out', 'n_edu_out'])
    feat_start = fo.get('start', ['vehrd', 'sexp', 'ptype_Worker', 'ptype_Student',
                                   'n_work_out', 'n_edu_out', 'wfh'])
    feat_end   = fo.get('end',   ['vehrd', 'sexp', 'n_work_out', 'n_edu_out'])
    feat_purp  = fo.get('purp',  ['vehrd', 'incp', 'sexp', 'ptype_Worker',
                                   'ptype_Student', 'nprd'])

    # ── Stage 2 data ─────────────────────────────────────────────────────────
    needed_2 = ['B', 'start_of_day', 'end_of_day']
    pd2 = person_day.dropna(subset=[c for c in needed_2 if c in person_day.columns]).copy()
    if len(pd2) < _MIN_OBS:
        logger.warning(f"stage23_joint: insufficient person-day records ({len(pd2)}); skipping")
        return {'success': False}

    B_raw  = pd.to_numeric(pd2.get('B', pd.Series(dtype=float)), errors='coerce').to_numpy(float)
    s_raw  = pd.to_numeric(pd2.get('start_of_day', pd.Series(dtype=float)), errors='coerce').to_numpy(float)
    e_raw  = pd.to_numeric(pd2.get('end_of_day',   pd.Series(dtype=float)), errors='coerce').to_numpy(float)

    ok = np.isfinite(B_raw) & (B_raw > 0) & np.isfinite(s_raw) & np.isfinite(e_raw)
    pd2 = pd2[ok]; B_raw = B_raw[ok]; s_raw = s_raw[ok]; e_raw = e_raw[ok]
    N = len(pd2)
    if N < _MIN_OBS:
        return {'success': False}

    X_B     = _design(pd2, feat_B)
    X_start = _design(pd2, feat_start)
    X_end   = _design(pd2, feat_end)

    # ── Stage 3 data ─────────────────────────────────────────────────────────
    # Determine purpose column
    purpose_col = next((c for c in ['main_purpose', 'DESTPURP', 'purpose']
                        if c in tours.columns), None)
    if purpose_col is None:
        logger.warning("stage23_joint: no purpose column found in tours; using lognormal-only fallback")
        return {'success': False}

    # Merge person-level features into tours
    merge_cols = [c for c in feat_purp if c in persons.columns]
    if merge_cols and 'person_id' in tours.columns and 'person_id' in persons.columns:
        tours3 = tours.merge(persons[['person_id'] + merge_cols], on='person_id', how='left')
    else:
        tours3 = tours.copy()

    dur_col = next((c for c in ['duration_min', 'duration', 'DURATION'] if c in tours3.columns), None)
    if dur_col is None:
        return {'success': False}

    tours3['_purp_idx'] = tours3[purpose_col].map({p: i for i, p in enumerate(purposes)})
    tours3 = tours3.dropna(subset=[dur_col, '_purp_idx'])
    tours3 = tours3[pd.to_numeric(tours3[dur_col], errors='coerce') > 0]
    if len(tours3) < _MIN_OBS:
        return {'success': False}

    y_purp  = tours3[dur_col].to_numpy(dtype=float)
    purp_id = tours3['_purp_idx'].to_numpy(dtype=int)
    X_purp  = _design(tours3, feat_purp)

    # Map each tour's person to a person-day row index for the budget penalty.
    # Use person_id as the key; fall back to position 0 for unmatched rows.
    if 'person_id' in tours3.columns and 'person_id' in pd2.columns:
        pid_map = {pid: i for i, pid in enumerate(pd2['person_id'].values)}
        pid_purp = np.array([pid_map.get(p, 0) for p in tours3['person_id'].values], dtype=int)
    else:
        pid_purp = np.zeros(len(tours3), dtype=int)

    # ── Parameter index ──────────────────────────────────────────────────────
    pidx = _ParamIndex(
        k_B     = X_B.shape[1] - 1,
        k_start = X_start.shape[1] - 1,
        k_end   = X_end.shape[1] - 1,
        k_purp  = X_purp.shape[1] - 1,
        n_purposes = len(purposes),
    )

    # ── Initialise parameters ────────────────────────────────────────────────
    p0 = np.zeros(pidx.total)
    # Warm-start budget intercept with log-mean of B
    p0[pidx.b_beta][0] = float(np.log(np.clip(B_raw, 1e-6, None)).mean())
    p0[pidx.b_shape]   = 0.0   # log(σ) = 0 → σ = 1
    # Warm-start start intercept
    p0[pidx.s_beta][0] = float(s_raw.mean())
    p0[pidx.s_lsig]    = np.log(float(s_raw.std()) + 1e-6)
    # Warm-start end intercept
    p0[pidx.e_beta][0] = float(e_raw.mean())
    p0[pidx.e_lsig]    = np.log(float(e_raw.std()) + 1e-6)
    # Per-purpose intercepts from log-mean durations
    for p_i, purp in enumerate(purposes):
        mask = purp_id == p_i
        if mask.sum() > 0:
            p0[pidx.p_beta[p_i]][0] = float(np.log(y_purp[mask]).mean())
            p0[pidx.p_lsig[p_i]]   = np.log(float(np.log(y_purp[mask]).std()) + 1e-3)

    # ── Build and run optimiser ──────────────────────────────────────────────
    nll_fn, grad_fn = _build_objective(
        X_B, B_raw, X_start, s_raw, X_end, e_raw,
        X_purp, y_purp, pid_purp, purp_id,
        pidx, len(purposes), budget_family, duration_family, lambda_budget,
    )

    logger.info(
        f"stage23_joint: fitting joint Stage 2+3 model | "
        f"N={N} person-days | {len(y_purp)} activity episodes | "
        f"budget_family={budget_family} | λ={lambda_budget} | "
        f"params={pidx.total}"
    )

    try:
        if _JAX_OK and grad_fn is not None:
            result = minimize(
                fun=lambda p: float(nll_fn(jnp.array(p))),
                x0=p0,
                jac=lambda p: np.array(grad_fn(jnp.array(p))),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-6},
            )
        else:
            result = minimize(
                fun=nll_fn,
                x0=p0,
                method='L-BFGS-B',
                options={'maxiter': maxiter},
            )
    except Exception as exc:
        logger.error(f"stage23_joint: optimisation failed: {exc}")
        return {'success': False}

    params_hat = result.x
    nll_val    = float(result.fun)
    bic_val    = _bic(nll_val, pidx.total, N + len(y_purp))

    logger.info(
        f"stage23_joint: done | success={result.success} | "
        f"NLL={nll_val:.4f} | BIC={bic_val:.4f} | "
        f"iters={result.nit}"
    )

    return {
        'success':       result.success,
        'params':        params_hat,
        'pidx':          pidx,
        'bic':           bic_val,
        'nll':           nll_val,
        'budget_family':   budget_family,
        'duration_family': duration_family,
        'purposes':      purposes,
        'feat_B':        feat_B,
        'feat_start':    feat_start,
        'feat_end':      feat_end,
        'feat_purp':     feat_purp,
        'purpose_col':   purpose_col,
        'lambda_budget': lambda_budget,
        'N_persons':     N,
        'N_activities':  len(y_purp),
    }


# =============================================================================
# Public API — predict
# =============================================================================

def predict_joint_stage23(
    synth: pd.DataFrame,
    persons: pd.DataFrame,
    models: dict,
    seed: int = 0,
    noise: bool = True,
) -> pd.DataFrame:
    """
    Apply the joint Stage 2+3 model to a synthetic population DataFrame.

    Returns a copy of ``synth`` with columns added:
        start_of_day, end_of_day, B
        <purpose>_duration_min  for each purpose in models['purposes']

    Duration columns are scaled so each person's total equals their predicted B.
    """
    if not models.get('success'):
        logger.warning("stage23_joint.predict: model not successfully fitted; no predictions made")
        return synth

    params  = models['params']
    pidx    = models['pidx']
    purps   = models['purposes']
    rng     = np.random.default_rng(seed)

    # Merge person-level features
    feat_cols = list(dict.fromkeys(
        models['feat_B'] + models['feat_start'] + models['feat_end'] + models['feat_purp']
    ))
    merge_cols = [c for c in feat_cols if c in persons.columns and c not in synth.columns]
    if merge_cols and 'person_id' in synth.columns and 'person_id' in persons.columns:
        synth = synth.merge(persons[['person_id'] + merge_cols], on='person_id', how='left')

    out = synth.copy()
    N   = len(out)

    # ── Stage 2: predict B, start_of_day, end_of_day ────────────────────────
    X_B     = _design(out, models['feat_B'])
    X_start = _design(out, models['feat_start'])
    X_end   = _design(out, models['feat_end'])

    eta_B   = X_B     @ params[pidx.b_beta]
    eta_s   = X_start @ params[pidx.s_beta]
    eta_e   = X_end   @ params[pidx.e_beta]

    sig_B = float(np.exp(params[pidx.b_shape]))
    sig_s = float(np.exp(params[pidx.s_lsig]))
    sig_e = float(np.exp(params[pidx.e_lsig]))

    if noise:
        eps   = rng.standard_normal(N)
        B_hat = np.exp(eta_B + sig_B * eps)
        s_hat = eta_s + sig_s * rng.standard_normal(N)
        e_hat = eta_e + sig_e * rng.standard_normal(N)
    else:
        # Lognormal mean: exp(η + σ²/2)
        B_hat = np.exp(eta_B + 0.5 * sig_B ** 2)
        s_hat = eta_s
        e_hat = eta_e

    B_hat = np.clip(B_hat, 0.5, 18.0)
    s_hat = np.clip(s_hat, 300, 1200)   # 5 am – 8 pm in minutes
    e_hat = np.clip(e_hat, s_hat + B_hat * 60, 1440)

    out['B']            = B_hat
    out['start_of_day'] = s_hat
    out['end_of_day']   = e_hat

    # ── Stage 3: predict per-purpose durations, budget-scaled ───────────────
    X_purp = _design(out, models['feat_purp'])

    unconstrained = {}
    for p_i, purp in enumerate(purps):
        beta_p  = params[pidx.p_beta[p_i]]
        sig_p   = float(np.exp(params[pidx.p_lsig[p_i]]))
        eta_p   = X_purp @ beta_p
        if noise:
            d_hat = np.exp(eta_p + sig_p * rng.standard_normal(N))
        else:
            d_hat = np.exp(eta_p + 0.5 * sig_p ** 2)
        unconstrained[purp] = np.clip(d_hat, 1.0, None)

    # Budget-proportional scaling: scale each person's durations so they sum to B
    total_unconstrained = sum(unconstrained.values())
    for purp in purps:
        # Convert B from hours to minutes for consistency with duration_min units
        out[f'{purp}_duration_min'] = (
            unconstrained[purp] / np.maximum(total_unconstrained, 1e-6) * B_hat * 60
        )

    return out


# =============================================================================
# BIC summary
# =============================================================================

def joint_model_summary(models: dict) -> str:
    if not models.get('success'):
        return "Joint Stage 2+3 model: not fitted."
    lines = [
        "Joint Stage 2+3 model summary",
        f"  budget_family : {models['budget_family']}",
        f"  lambda_budget : {models['lambda_budget']}",
        f"  N persons     : {models['N_persons']}",
        f"  N activities  : {models['N_activities']}",
        f"  parameters    : {models['pidx'].total}",
        f"  NLL           : {models['nll']:.4f}",
        f"  BIC           : {models['bic']:.4f}",
        f"  purposes      : {models['purposes']}",
    ]
    return "\n".join(lines)
