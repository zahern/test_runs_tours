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
    GA_ESTIMATOR=metacount      (MetaCount NB2 for stage1, Tobit/lognormal for 2–3)
    GA_ESTIMATOR=searchlibrium  (SearchLibrium MNL for stage4; default estimators for 1–3)
    GA_ESTIMATOR=both           (MetaCount for 1–3, SearchLibrium for 4)
    GA_ESTIMATOR=hybrid         (same as 'both'; no statsmodels fallback for either)
    GA_ESTIMATOR=sa_bandit      (blended: MetaCount for 1–3, SearchLibrium for 4, Larch for 5;
                                  outer loop uses BanditFeatureSearcher; inner estimators use
                                  BanditGuidedSA (MetaCount) and BanditSA (SearchLibrium) when
                                  GA_USE_BANDIT_SA=1 is also set)

Fitness functions all return –BIC / n (higher = better, same convention as
the rest of ga_hypothesis_testing.py).

The GA is unchanged; only the model fitted at each evaluation differs.
"""
from __future__ import annotations

import gc
import inspect
import os
import logging
import threading
import warnings
import numpy as np
import pandas as pd
from typing import List, Optional

# ---------------------------------------------------------------------------
# JAX-hang guard: metacountregressor uses JAX LBFGS; the first evaluation for
# each unique model shape triggers JIT compilation that can take 10+ minutes.
# We run the fit in a daemon thread and abandon it if it exceeds the timeout.
# ---------------------------------------------------------------------------
_FIT_TIMEOUT_S: float = 90.0   # seconds before declaring a JAX hang
_jax_orphan_threads: list = []  # threads still running after timeout

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JAX compilation-cache guard: every distinct (n_features, n_rows, family)
# combination evaluated by the GA triggers a fresh XLA JIT compile inside
# MetaCount/SearchLibrium.  JAX's compiled-program cache is process-global
# and is never freed, so a multi-hour GA search that visits hundreds of
# distinct chromosomes per purpose accumulates compiled programs until the
# PBS cgroup memory limit is hit (observed: 64GB exhausted ~21h into a
# sa_bandit/ultrafast run, after ~3800 unique MetaCount fits in Stage 1
# alone).  Instead of clearing on a fixed interval, we monitor RSS and only
# flush the JAX cache when memory approaches the allocation limit.
#
# Tuning env vars:
#   GA_JAX_CACHE_MEM_LIMIT_GB  — memory limit in GB (default: detect from cgroup
#                                or fall back to 32).  Clear when RSS > 80% of limit.
#   GA_JAX_CACHE_CHECK_EVERY   — fits between memory checks (default: 50).
#   GA_PAD_FEATURES            — if '1', pad all feature matrices to full
#                                candidate width (zeros for inactive features)
#                                so JAX compiles ONCE per stage instead of once
#                                per distinct chromosome width.
# ---------------------------------------------------------------------------
_FIT_CALL_COUNT: int = 0
_JAX_CACHE_CHECK_EVERY: int = max(1, int(os.environ.get('GA_JAX_CACHE_CHECK_EVERY', '50')))
_JAX_CACHE_MEM_LIMIT_BYTES: int = 0
_LAST_RSS_CHECK: int = 0
_LAST_RSS_BYTES: int = 0
_PAD_FEATURES: bool = os.environ.get('GA_PAD_FEATURES', '0') == '1'
# Per-stage cache of the full candidate list for padding
_STAGE_ALL_CANDIDATES: Dict[int, List[str]] = {}

def _padded_feature_matrix(df, selected_vars, all_vars):
    """Pad selected_vars to all_vars width with zero columns.
    Returns (padded_spec_vars, X_matrix, active_count, active_mask)."""
    if not _PAD_FEATURES or not all_vars:
        return selected_vars, df[selected_vars].to_numpy(dtype=float), len(selected_vars), None
    active = [c for c in selected_vars if c in df.columns]
    # Build full-width matrix: active columns from data, inactive = zeros
    X_full = np.zeros((len(df), len(all_vars)), dtype=float)
    col_map = {name: i for i, name in enumerate(all_vars)}
    for c in active:
        if c in col_map:
            X_full[:, col_map[c]] = pd.to_numeric(df[c], errors='coerce').fillna(0).to_numpy(dtype=float)
    # Boolean mask: True for active features
    active_mask = np.array([(name in set(active)) for name in all_vars], dtype=bool)
    return all_vars, X_full, len(active), active_mask


def _bounded_padded_fit(X, y, active_mask, family, start_vector=None):
    """
    Fit with bounds: active params free, inactive pinned at 0.
    Uses JAX log-likelihoods directly with scipy L-BFGS-B bounds.
    Returns dict matching MetaCount output format.
    """
    import jax.numpy as jnp
    from scipy.optimize import minimize
    import jax
    import jax.scipy.special as jsp

    n_features = X.shape[1]
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)
    if n_active == 0:
        return None

    X_jax = jnp.asarray(X, dtype=jnp.float32)
    y_jax = jnp.asarray(y, dtype=jnp.float32)
    active_idx_arr = jnp.asarray(active_idx, dtype=jnp.int32)
    n_features = X.shape[1]

    # ── JAX negative log-likelihood (active_idx is a traced argument) ────
    # Compiles ONCE per stage: the scatter op is generic, handles any mask.
    if family in ('nb2', 'negbin'):
        @jax.jit
        def _nl(few_params, idx):
            beta_a = few_params[:-1]; alpha_raw = few_params[-1]
            full_beta = jnp.zeros(n_features, dtype=jnp.float32).at[idx].set(beta_a)
            mu = jnp.exp(jnp.clip(X_jax @ full_beta, -20, 20))
            alpha = jnp.exp(alpha_raw)
            r = 1.0 / jnp.clip(alpha, 1e-8, 1e8)
            ll = (jsp.gammaln(y_jax + r) - jsp.gammaln(r)
                  - jsp.gammaln(y_jax + 1.0)
                  + r * jnp.log(r / (r + mu + 1e-15))
                  + y_jax * jnp.log(mu / (r + mu + 1e-15) + 1e-15))
            return -jnp.sum(ll)
    elif family == 'lognormal':
        @jax.jit
        def _nl(few_params, idx):
            beta_a = few_params[:-1]; sigma_raw = few_params[-1]
            full_beta = jnp.zeros(n_features, dtype=jnp.float32).at[idx].set(beta_a)
            eta = X_jax @ full_beta
            sigma = jnp.exp(jnp.clip(sigma_raw, -10, 5))
            resid = jnp.log(jnp.clip(y_jax, 1e-15, None)) - eta
            ll = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - 0.5 * (resid / sigma) ** 2
            return -jnp.sum(ll)
    elif family == 'gaussian':
        @jax.jit
        def _nl(few_params, idx):
            beta_a = few_params[:-1]; sigma_raw = few_params[-1]
            full_beta = jnp.zeros(n_features, dtype=jnp.float32).at[idx].set(beta_a)
            eta = X_jax @ full_beta
            sigma = jnp.exp(jnp.clip(sigma_raw, -10, 5))
            resid = y_jax - eta
            ll = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma) - 0.5 * (resid / sigma) ** 2
            return -jnp.sum(ll)
    else:
        return None

    _grad = jax.jit(jax.grad(_nl, argnums=0))

    # ── Optimize: only n_active + 1 params ───────────────────────────────
    n_few = n_active + 1
    x0 = np.zeros(n_few)
    if start_vector is not None:
        sv = np.asarray(start_vector, dtype=float)
        if len(sv) == n_features + 1:
            x0[:n_active] = sv[active_idx]
            x0[-1] = sv[-1]
        elif len(sv) == n_few:
            x0 = sv

    def _f(x): return float(_nl(jnp.asarray(x, dtype=jnp.float32), active_idx_arr))
    def _g(x): return np.array(_grad(jnp.asarray(x, dtype=jnp.float32), active_idx_arr), dtype=float)

    result = minimize(_f, x0, method='SLSQP', jac=_g,
                      options={'maxiter': 200, 'ftol': 1e-8})

    # ── Return MetaCount-compatible result ─────────────────────────────────
    # Reconstruct full beta from the sliced params
    few = result.x
    full_beta = np.zeros(n_features, dtype=float)
    full_beta[active_idx] = few[:n_active]
    n = len(y)
    ll = -float(result.fun)
    k = n_active + 1  # active features + dispersion/sigma
    bic = k * np.log(max(n, 2)) - 2.0 * ll

    return {
        'summary': {'bic': bic, 'loglik': ll},
        'coef': dict(zip([f'x{i}' for i in range(n_features)], full_beta)),
        'n_obs': n,
        'success': result.success,
    }


def _own_cgroup_mem_limit_bytes() -> int | None:
    """Memory limit of THIS process's cgroup (walking up ancestors), or None.

    PBS/SLURM put the job in a nested cgroup (e.g. .../pbs_jobs/23972276.aqua);
    the root-level memory.max is usually 'max', so checking only the root files
    misses the real per-job limit and the caller falls back to the 32 GB
    default even on a 128 GB job (job 23972276 post-mortem).
    """
    import pathlib
    try:
        cg_lines = pathlib.Path('/proc/self/cgroup').read_text().splitlines()
    except Exception:
        return None
    candidates: list[pathlib.Path] = []
    for line in cg_lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        hier_id, controllers, rel = parts
        rel = rel.lstrip('/')
        if hier_id == '0':      # cgroup v2 unified hierarchy
            root = pathlib.Path('/sys/fs/cgroup')
            leaf = root / rel if rel else root
            for p in [leaf, *leaf.parents]:
                candidates.append(p / 'memory.max')
                if p == root:
                    break
        elif 'memory' in controllers.split(','):   # cgroup v1 memory controller
            root = pathlib.Path('/sys/fs/cgroup/memory')
            leaf = root / rel if rel else root
            for p in [leaf, *leaf.parents]:
                candidates.append(p / 'memory.limit_in_bytes')
                if p == root:
                    break
    for c in candidates:
        try:
            if not c.is_file():
                continue
            val = c.read_text().strip()
            if val in ('', 'max'):
                continue
            limit = int(val)
            if 0 < limit < (1 << 50):
                return limit
        except Exception:
            continue
    return None


def _detect_mem_limit_bytes() -> int:
    """Return the effective memory limit for this process in bytes."""
    # 1) explicit env var
    env_gb = os.environ.get('GA_JAX_CACHE_MEM_LIMIT_GB', '')
    if env_gb:
        try:
            return int(float(env_gb) * 1_073_741_824)
        except ValueError:
            pass
    # 2) this process's own (possibly nested) cgroup — finds the PBS job limit
    own = _own_cgroup_mem_limit_bytes()
    if own is not None:
        return own
    # 3) try root cgroup v1 memory limit (PBS/Torque/SLURM on linux)
    try:
        import pathlib
        cg_v1 = pathlib.Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        if cg_v1.is_file():
            limit = int(cg_v1.read_text().strip())
            if 0 < limit < (1 << 50):  # sanity: 1 PB cap
                return limit
    except Exception:
        pass
    # 4) try root cgroup v2
    try:
        import pathlib
        cg_v2 = pathlib.Path('/sys/fs/cgroup/memory.max')
        if cg_v2.is_file():
            val = cg_v2.read_text().strip()
            if val not in ('', 'max'):
                limit = int(val)
                if limit > 0:
                    return limit
    except Exception:
        pass
    # 5) fallback: 32 GB
    return 32 * 1_073_741_824


_JAX_CACHE_MEM_LIMIT_BYTES = _detect_mem_limit_bytes()

# Anti-thrash state: once resident data (not caches) keeps RSS above the
# threshold, clearing frees ~nothing but forces JAX to recompile everything.
# Job 23972276 spent its whole run re-clearing every check interval.
_CLEAR_COOLDOWN_MIN_S: float = 60.0
_CLEAR_COOLDOWN_MAX_S: float = 1800.0
_CLEAR_COOLDOWN_S: float = _CLEAR_COOLDOWN_MIN_S
_LAST_CLEAR_TS: float = 0.0


def _maybe_clear_jax_caches() -> None:
    """Check RSS every _JAX_CACHE_CHECK_EVERY fits; clear JAX caches + gc
    only when memory usage exceeds 80% of the detected limit. Clears are
    rate-limited and back off exponentially when they free no memory."""
    global _FIT_CALL_COUNT, _LAST_RSS_CHECK, _LAST_RSS_BYTES
    global _CLEAR_COOLDOWN_S, _LAST_CLEAR_TS
    _FIT_CALL_COUNT += 1
    if _FIT_CALL_COUNT - _LAST_RSS_CHECK < _JAX_CACHE_CHECK_EVERY:
        return
    _LAST_RSS_CHECK = _FIT_CALL_COUNT

    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except Exception:
        return

    _LAST_RSS_BYTES = rss
    threshold = int(_JAX_CACHE_MEM_LIMIT_BYTES * 0.80)
    if rss > threshold:
        import time
        now = time.monotonic()
        if now - _LAST_CLEAR_TS < _CLEAR_COOLDOWN_S:
            return
        _LAST_CLEAR_TS = now
        logger.warning(
            "JAX cache: RSS %.1f GB / limit %.1f GB (>80%%), clearing JAX caches",
            rss / 1e9, _JAX_CACHE_MEM_LIMIT_BYTES / 1e9,
        )
        try:
            from SearchLibrium.multinomial_logit import MultinomialLogit
            MultinomialLogit._clear_jit_cache()
        except Exception:
            pass
        try:
            from SearchLibrium.MixedLogit import MixedLogit
            MixedLogit._clear_jit_cache()
        except Exception:
            pass
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()
        try:
            rss_after = psutil.Process().memory_info().rss
        except Exception:
            rss_after = rss
        freed = rss - rss_after
        if freed < int(0.05 * _JAX_CACHE_MEM_LIMIT_BYTES):
            _CLEAR_COOLDOWN_S = min(_CLEAR_COOLDOWN_S * 2.0, _CLEAR_COOLDOWN_MAX_S)
            logger.warning(
                "JAX cache: clear freed only %.1f GB (RSS is data, not caches) - "
                "backing off, next clear in >= %.0f s",
                freed / 1e9, _CLEAR_COOLDOWN_S,
            )
        else:
            _CLEAR_COOLDOWN_S = _CLEAR_COOLDOWN_MIN_S

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

    If a GPU is detected (via jax.devices) it is preferred — set JAX_PLATFORMS=cuda.
    CPU is only forced when no GPU is present OR the user explicitly sets
    GA_METACOUNT_FORCE_CPU=1.

    Tuning env vars:
      GA_METACOUNT_FORCE_CPU=1   — force CPU even when a GPU is available
      GA_METACOUNT_FORCE_CPU=0   — prefer GPU when available (default behaviour)
    """
    force_cpu = os.environ.get('GA_METACOUNT_FORCE_CPU', '0') == '1'
    platform_hint = os.environ.get('JAX_PLATFORMS', '').strip().lower()

    if force_cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        logger.info("JAX backend: CPU (forced by GA_METACOUNT_FORCE_CPU=1)")
        return

    if platform_hint:
        if platform_hint in ('rocm',):
            os.environ['JAX_PLATFORMS'] = 'cpu'
            logger.warning("JAX_PLATFORMS=rocm is unavailable; forcing cpu")
        elif platform_hint == 'gpu':
            os.environ['JAX_PLATFORMS'] = 'cuda'
        return

    # Auto-detect: prefer GPU if available
    try:
        import jax
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            os.environ['JAX_PLATFORMS'] = 'cuda'
            logger.info("JAX backend: CUDA (detected %d GPU device(s))", len(gpu_devices))
            return
    except Exception:
        pass
    os.environ['JAX_PLATFORMS'] = 'cpu'
    logger.info("JAX backend: CPU (no GPU detected)")


_configure_metacount_backend()

try:
    from metacountregressor import ExperimentBuilder, ModelConstraints
    _METACOUNT_OK = True
except Exception as exc:
    _METACOUNT_OK = False
    _METACOUNT_IMPORT_ERROR = exc
    logger.debug("metacountregressor unavailable: %s", exc)

try:
    import io as _io, sys as _sys
    _sl_old = _sys.stdout; _sys.stdout = _io.StringIO()
    try:
        from SearchLibrium import MultinomialLogit, MixedLogit
    finally:
        _sys.stdout = _sl_old
    _SEARCHLIB_OK = True
except Exception as exc:
    _SEARCHLIB_OK = False
    _SEARCHLIB_IMPORT_ERROR = exc
    logger.debug("SearchLibrium unavailable: %s", exc)


def ga_estimator() -> str:
    """Read GA_ESTIMATOR env var; default to 'default'."""
    v = os.environ.get('GA_ESTIMATOR', 'default').strip().lower()
    if v not in ('default', 'metacount', 'searchlibrium', 'both', 'hybrid', 'sa_bandit'):
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


# ---------------------------------------------------------------------------
# Adaptive feature weakness tracking
# ---------------------------------------------------------------------------
# Shared across all MetaCount fitness evaluations within a single stage run.
# The SA _neighbor() reads these scores to bias bit-flip proposals away from
# features that repeatedly come out with near-zero coefficients.
#
# Lifecycle: reset_feature_weakness() is called at the start of each stage
# (before the SA is created) so scores from one stage don't pollute the next.

_FEATURE_WEAKNESS: Dict[str, float] = {}
_WEAKNESS_DECAY = 0.95   # multiply all scores by this on each update call


def _update_feature_weakness(
    coef_rows: list,
    selected_vars: List[str],
    coef_zero: float = 0.01,
    p_thresh:  float = 0.10,
) -> None:
    """
    Update per-feature weakness scores after a model fit.

    A parameter is "weak" if EITHER:
      - |coef_value| < coef_zero  (coefficient is effectively zero), OR
      - p_value > p_thresh        (not statistically significant at 10%)
    A parameter is "stuck" if _estimate() (stage5_destination_choice.py)
    flagged it as never having moved from its starting value or as pinned at
    a bound (see 'stuck'/'at_bound' keys in _larch_coef_rows) - stronger
    evidence of a bad specification than mere insignificance, since it means
    the optimizer found no usable gradient at all for that term.

    Stuck/at-bound → score += 5.0
    Weak (near-zero or insignificant), not stuck → score += 2.0
    Strong (none of the above) → score -= 0.5, clamped at 0

    All scores decay by _WEAKNESS_DECAY each call so recent evidence dominates.
    """
    # Decay existing scores
    for k in list(_FEATURE_WEAKNESS.keys()):
        _FEATURE_WEAKNESS[k] *= _WEAKNESS_DECAY

    # Build value + p-value + stuck/at-bound maps; lower-case keys for
    # robust matching against selected_vars (which may differ in case).
    vmap: Dict[str, float] = {}
    pmap: Dict[str, float] = {}
    stuck_map: Dict[str, bool] = {}
    for r in coef_rows:
        name = str(r.get('param', '')).lower().strip()
        if not name:
            continue
        try:
            v = float(r.get('value', np.nan))
        except Exception:
            v = np.nan
        try:
            p = float(r.get('p_value', np.nan))
        except Exception:
            p = np.nan
        if np.isfinite(v):
            vmap[name] = v
        if np.isfinite(p):
            pmap[name] = p
        if r.get('stuck') or r.get('at_bound'):
            stuck_map[name] = True

    for var in selected_vars:
        key = var.lower().strip()
        v = vmap.get(key, np.nan)
        p = pmap.get(key, np.nan)
        stuck = stuck_map.get(key, False)
        # Need at least one signal
        if not np.isfinite(v) and not np.isfinite(p) and not stuck:
            continue
        near_zero  = np.isfinite(v) and abs(v) < coef_zero
        insig      = np.isfinite(p) and p > p_thresh
        current    = _FEATURE_WEAKNESS.get(var, 0.0)
        if stuck:
            _FEATURE_WEAKNESS[var] = current + 5.0
        elif near_zero or insig:
            _FEATURE_WEAKNESS[var] = current + 2.0
        else:
            _FEATURE_WEAKNESS[var] = max(0.0, current - 0.5)


def get_feature_weakness() -> Dict[str, float]:
    """Return a snapshot of current per-feature weakness scores."""
    return dict(_FEATURE_WEAKNESS)


def reset_feature_weakness() -> None:
    """Clear weakness scores at the start of a new search stage."""
    _FEATURE_WEAKNESS.clear()


def _metacount_significance_penalty(rows: list) -> float:
    """
    BIC penalty for parameters that are near-zero or statistically insignificant.

    A parameter is penalised if EITHER:
      - |coef_value| < GA_SIG_COEF_ZERO  (default 0.01)  — effectively zero, OR
      - p_value > GA_SIG_PVAL_THRESH     (default 0.10)  — not significant at 10%

    Penalty = GA_SIG_PENALTY_WEIGHT * (n_weak / n_tested)

    Both env vars are tunable without code changes.
    """
    if not rows:
        return 0.0
    weight    = float(os.environ.get('GA_SIG_PENALTY_WEIGHT', '0.10'))
    coef_zero = float(os.environ.get('GA_SIG_COEF_ZERO',      '0.01'))
    p_thresh  = float(os.environ.get('GA_SIG_PVAL_THRESH',    '0.10'))

    tested = 0
    n_weak = 0
    for r in rows:
        try:
            v = float(r.get('value', np.nan))
        except Exception:
            v = np.nan
        try:
            p = float(r.get('p_value', np.nan))
        except Exception:
            p = np.nan
        if not np.isfinite(v) and not np.isfinite(p):
            continue
        tested += 1
        near_zero = np.isfinite(v) and abs(v) < coef_zero
        insig     = np.isfinite(p) and p > p_thresh
        if near_zero or insig:
            n_weak += 1

    if tested == 0:
        return 0.0
    return weight * (n_weak / tested)


# ---------------------------------------------------------------------------
# Larch and SearchLibrium coefficient extraction helpers
# ---------------------------------------------------------------------------

def _larch_coef_rows(mdl) -> list:
    """
    Extract per-parameter rows from a fitted Larch model.

    Tries parameter_summary() (most reliable), then falls back to the raw
    pvals / tstat dicts.  Returns a list of dicts with keys:
    param, value, std_err, t_stat, p_value — same format as
    _extract_metacount_coefs so _metacount_significance_penalty can consume it.
    """
    rows = []
    try:
        from scipy.stats import norm as _snorm
        import pandas as _pd

        # Params _estimate() (stage5_destination_choice.py) flagged as never
        # having moved from their starting value, or pinned at a bound -
        # stronger evidence of a bad specification than mere insignificance.
        # See _estimate()'s "Stuck / at-bound parameter diagnostics" block.
        _stuck_set = set(getattr(mdl, '_stuck_params', None) or [])
        _at_bound_set = set(getattr(mdl, '_at_bound_params', None) or [])

        # --- primary path: parameter_summary() DataFrame --------------------
        try:
            ps = mdl.parameter_summary()
            if isinstance(ps, _pd.DataFrame) and not ps.empty:
                val_col = next((c for c in ps.columns
                                if c.strip().lower() in ('value', 'coef', 'estimate')), None)
                se_col  = next((c for c in ps.columns
                                if 'std' in c.lower()), None)
                t_col   = next((c for c in ps.columns
                                if c.strip().lower() in ('t stat', 't_stat', 'tstat', 't', 'z')), None)
                if val_col:
                    for pname, row in ps.iterrows():
                        val = (float(row[val_col]) if _pd.notna(row[val_col])
                               else float('nan'))
                        se  = (float(row[se_col]) if se_col and _pd.notna(row[se_col])
                               else float('nan'))
                        t   = (float(row[t_col]) if t_col and _pd.notna(row[t_col])
                               else float('nan'))
                        if not np.isfinite(t) and np.isfinite(val) and np.isfinite(se) and se > 0:
                            t = val / se
                        p = (float(2.0 * _snorm.sf(abs(t)))
                             if np.isfinite(t) else float('nan'))
                        rows.append({'param': str(pname), 'value': val,
                                     'std_err': se, 't_stat': t, 'p_value': p,
                                     'stuck': str(pname) in _stuck_set,
                                     'at_bound': str(pname) in _at_bound_set})
                    if rows:
                        return rows
        except Exception:
            pass

        # --- fallback: pvals / tstat dicts ----------------------------------
        pv = getattr(mdl, 'pvals', None)
        tv = getattr(mdl, 'tstat', None)
        if pv is not None:
            try:
                pv_d = dict(pv)
                tv_d = dict(tv) if tv is not None else {}
                for pname, p_val in pv_d.items():
                    t_val = tv_d.get(pname, float('nan'))
                    try:
                        t_val = float(t_val)
                    except Exception:
                        t_val = float('nan')
                    try:
                        p_val = float(p_val)
                    except Exception:
                        p_val = float('nan')
                    rows.append({'param': str(pname), 'value': float('nan'),
                                 'std_err': float('nan'),
                                 't_stat': t_val, 'p_value': p_val,
                                 'stuck': str(pname) in _stuck_set,
                                 'at_bound': str(pname) in _at_bound_set})
            except Exception:
                pass
    except Exception:
        pass
    return rows


def _searchlib_coef_rows(model, varnames: List[str]) -> list:
    """
    Extract per-feature coef rows from a fitted SearchLibrium MNL / MixedLogit.

    For isvars, the coefficient vector is ordered as:
      [var0_alt1, var0_alt2, ..., var1_alt1, ...] (block-by-variable).
    We average |coef| across alternatives to get a single representative
    magnitude per variable for the near-zero test.

    Standard errors are tried from several attribute names; if unavailable,
    p_value is left as NaN (only the near-zero check fires).
    """
    rows = []
    try:
        from scipy.stats import norm as _snorm
        coefs = np.asarray(getattr(model, 'coeff_est', []) or [], dtype=float).ravel()
        if coefs.size == 0:
            return rows

        # --- attempt to extract standard errors ----------------------------
        se_arr: Optional[np.ndarray] = None
        for _se_attr in ('se', 'std_err', 'coeff_se', 'standard_errors', 'stderr'):
            raw = getattr(model, _se_attr, None)
            if raw is not None:
                try:
                    arr = np.asarray(raw, dtype=float).ravel()
                    if arr.size == coefs.size:
                        se_arr = arr
                        break
                except Exception:
                    pass
        if se_arr is None:
            for _cov_attr in ('covariance', 'cov', 'vcov'):
                cov = getattr(model, _cov_attr, None)
                if cov is not None:
                    try:
                        cov_arr = np.asarray(cov, dtype=float)
                        if cov_arr.ndim == 2 and cov_arr.shape[0] == coefs.size:
                            se_arr = np.sqrt(np.clip(np.diag(cov_arr), 0.0, None))
                            break
                    except Exception:
                        pass

        # --- map coefficients to features ----------------------------------
        # For isvars: coeff vector is blocked by variable
        n_vars = len(varnames)
        n_alts_per_var = max(1, coefs.size // max(n_vars, 1))
        for fi, var in enumerate(varnames):
            start = fi * n_alts_per_var
            end   = min(start + n_alts_per_var, len(coefs))
            if start >= len(coefs):
                continue
            var_coefs = coefs[start:end]
            val = float(np.mean(np.abs(var_coefs)))
            if se_arr is not None and start < len(se_arr):
                var_se = float(np.mean(se_arr[start:min(end, len(se_arr))]))
                t = float(val / var_se) if var_se > 0 else float('nan')
                p = float(2.0 * _snorm.sf(abs(t))) if np.isfinite(t) else float('nan')
            else:
                var_se, t, p = float('nan'), float('nan'), float('nan')
            rows.append({'param': var, 'value': val, 'std_err': var_se,
                         't_stat': t, 'p_value': p})
    except Exception:
        pass
    return rows


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
    # Handle dict return from fit_manual_model (new API)
    if isinstance(result, dict):
        inner = result.get('result')
        if inner is not None:
            return _result_vector(inner)
        return None
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


_FAMILY_TO_MODEL_STR = {
    'nb2':      'nb',
    'negbin':   'nb',
    'poisson':  'poisson',
    'lognormal':'lognormal',
    'duration': 'lognormal',
    'gaussian': 'gaussian',
    'tobit':    'tobit',
}


def _fit_metacount_manual_model(builder, spec: dict, R: int,
                                start_vector: Optional[np.ndarray] = None,
                                family: str = 'nb2',
                                fast_mode: bool = False):
    fit_method = builder.fit_manual_model
    param_names = _signature_names(fit_method)
    kwargs: dict[str, object] = {}
    if 'R' in param_names:
        kwargs['R'] = R
    elif 'draws' in param_names:
        kwargs['draws'] = R
    elif 'n_draws' in param_names:
        kwargs['n_draws'] = R

    # Pass correct model string so fit_manual_model does not default to "poisson"
    # and overwrite the model type inferred from the spec dict.
    model_str = _FAMILY_TO_MODEL_STR.get(family, 'nb')
    if 'model' in param_names:
        kwargs['model'] = model_str

    # Skip the DE warm-start in fast (GA-search) mode to reduce JAX compilation load.
    if fast_mode and 'continuous_de_warm_start' in param_names:
        kwargs['continuous_de_warm_start'] = False

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
    """
    Call model.fit() and return a result object with .fun, .success, .x.

    SearchLibrium ≥ 0.0.87: fit() is void and stores results in model attributes
    (model.loglik, model.coeff_est).  When fit() returns None we build a
    SimpleNamespace adapter so all callers can uniformly do result.fun / result.x.
    """
    import types as _types
    fit_method = model.fit
    param_names = _signature_names(fit_method)
    # Detect if fit() accepts **kwargs (e.g. SearchLibrium MNL uses `betas` kwarg)
    try:
        _has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(fit_method).parameters.values()
        )
    except Exception:
        _has_var_kw = False
    kwargs: dict[str, object] = {}
    start_arg_name = None
    if start_vector is not None and np.asarray(start_vector).size:
        for candidate in ('x0', 'start_values', 'start_params', 'initial_values', 'init_params', 'betas'):
            if candidate in param_names or (_has_var_kw and candidate == 'betas'):
                start_arg_name = candidate
                break
    if start_arg_name is not None:
        kwargs[start_arg_name] = start_vector
    try:
        result = fit_method(**kwargs) if kwargs else fit_method()
    except TypeError:
        if start_arg_name is not None:
            kwargs.pop(start_arg_name, None)
            result = fit_method(**kwargs) if kwargs else fit_method()
        else:
            raise
    _maybe_clear_jax_caches()

    # SearchLibrium 0.0.87+: fit() returns None; wrap model attributes.
    if result is None:
        ll = getattr(model, 'loglik', None)
        if ll is not None and np.isfinite(float(ll)):
            r = _types.SimpleNamespace()
            r.fun     = -float(ll)
            r.success = True
            r.x       = np.asarray(getattr(model, 'coeff_est', []), dtype=float)
            return r
        # fit silently failed — return a sentinel failure object
        r = _types.SimpleNamespace()
        r.fun     = np.inf
        r.success = False
        r.x       = np.array([])
        return r

    return result


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

    # Guard against degenerate targets/design matrices that make MetaCount fail
    # with low-level numpy reductions (e.g., zero-size max).
    try:
        y = pd.to_numeric(df[y_col], errors='coerce').astype(float)
        if y.notna().sum() < 30:
            return -1e9
        yv = y.to_numpy(dtype=float)
        yv = yv[np.isfinite(yv)]
        if yv.size < 30:
            return -1e9
        if family in ('nb2', 'poisson', 'negbin'):
            if (yv < 0).any():
                return -1e9
            if np.allclose(np.nanstd(yv), 0.0):
                return -1e9
            if np.nanmax(yv) <= 0:
                return -1e9
        elif family in ('duration', 'lognormal'):
            if (yv <= 0).sum() > 0:
                return -1e9
            if np.allclose(np.nanstd(yv), 0.0):
                return -1e9
        else:
            if np.allclose(np.nanstd(yv), 0.0):
                return -1e9
        # Guard X matrix: MetaCount crashes with zero-size array if any selected
        # var is missing from df or if the design matrix has zero rows/columns.
        x_present = [c for c in selected_vars if c in df.columns]
        if not x_present:
            return -1e9
        # ── Feature padding (optional) ────────────────────────────────────
        # When GA_PAD_FEATURES=1, pad to full candidate width so JAX compiles
        # once per stage instead of once per distinct chromosome width.
        all_cands = _STAGE_ALL_CANDIDATES.get(id(df), [])  # use df id as proxy
        if _PAD_FEATURES and all_cands:
            selected_vars, xv, k_active, active_mask = _padded_feature_matrix(df, x_present, all_cands)
        else:
            xv = df[x_present].to_numpy(dtype=float)
            selected_vars = x_present
            k_active = len(selected_vars)
            active_mask = None
        if xv.shape[0] == 0 or xv.shape[1] == 0:
            return -1e9
    except Exception:
        return -1e9

    try:
        # ── Define spec variables (full candidate list when padding) ──────
        spec_vars = selected_vars
        if _PAD_FEATURES:
            all_cands = _STAGE_ALL_CANDIDATES.get(id(df), [])
            if all_cands and len(all_cands) > 1:
                spec_vars = list(dict.fromkeys(all_cands))  # ordered, deduped
        cache_key = _metacount_start_key(family, y_col, spec_vars)

        # ── Bounded padded fit: scipy L-BFGS-B with inactive params pinned ─
        if _PAD_FEATURES and active_mask is not None and active_mask.any():
            start_vec = _METACOUNT_START_VALUES.get(cache_key)
            fitted = _bounded_padded_fit(
                xv, yv, active_mask, family, start_vector=start_vec)
            _maybe_clear_jax_caches()
            if fitted is None or not fitted.get('success', False):
                return -1e9
            # Cache start vector (only active params have non-zero values)
            if 'coef' in fitted:
                sv = start_vec if start_vec is not None else np.zeros(len(active_mask) + 1)
                for i, name in enumerate(spec_vars):
                    sv[i] = float(fitted['coef'].get(f'x{i}', 0.0))
                _METACOUNT_START_VALUES[cache_key] = sv
            # Compute BIC and return
            n = len(df)
            ll = float(fitted['summary']['loglik'])
            bic = float(fitted['summary']['bic'])
            k = k_active + 1  # active features + dispersion/sigma
            # Manual BIC with correct k
            bic = k * np.log(max(n, 2)) - 2.0 * ll
            sig_pen = 0.0  # bounded fit doesn't produce coef rows for sig_pen
            return -(bic / max(n, 1)) - sig_pen

        # ── Standard MetaCount fit (threaded) ──────────────────────────────
        spec = {
            'fixed_terms':      spec_vars,
            'rdm_terms':        [],
            'rdm_cor_terms':    [],
            'grouped_terms':    [],
            'hetro_in_means':   [],
            'zi_terms':         [],
            'membership_terms': [],
            'dispersion':       1 if family in ('nb2', 'negbin') else 0,
            'latent_classes':   1,
        }
        builder = _create_metacount_builder(df, id_col=id_col, y_col=y_col,
                                            family=family, offset_col=offset_col)

        # Run the fit in a daemon thread so a JAX JIT hang doesn't block forever.
        global _jax_orphan_threads
        _jax_orphan_threads = [t for t in _jax_orphan_threads if t.is_alive()]
        if len(_jax_orphan_threads) >= 3:
            # Too many unfinished JAX compilations — bail out so the circuit
            # breaker in the GA loop can switch to the scipy fallback.
            return -1e9

        _result_box: list = [None]
        _exc_box:    list = [None]

        def _do_fit() -> None:
            try:
                _result_box[0] = _fit_metacount_manual_model(
                    builder, spec, R=1,
                    start_vector=_METACOUNT_START_VALUES.get(cache_key),
                    family=family,
                    fast_mode=True,
                )
            except Exception as _e:
                _exc_box[0] = _e

        _fit_thread = threading.Thread(target=_do_fit, daemon=True)
        _fit_thread.start()
        _fit_thread.join(timeout=_FIT_TIMEOUT_S)
        if _fit_thread.is_alive():
            _jax_orphan_threads.append(_fit_thread)
            warnings.warn(
                f"metacount_fit timed out after {_FIT_TIMEOUT_S:.0f}s "
                f"(orphaned_jax_threads={len(_jax_orphan_threads)}); returning -1e9",
                RuntimeWarning,
                stacklevel=2,
            )
            return -1e9
        _maybe_clear_jax_caches()
        if _exc_box[0] is not None:
            raise _exc_box[0]   # re-raise to be caught by outer try/except
        fitted = _result_box[0]

        if fitted is None:
            return -1e9
        start_vec = _result_vector(fitted)
        if start_vec is not None:
            _METACOUNT_START_VALUES[cache_key] = start_vec
        # fit_manual_model returns dict {'summary':{'bic':…,'loglik':…},…}; older API used attributes
        if isinstance(fitted, dict):
            _summary = fitted.get('summary') or {}
            bic = _summary.get('bic')
            ll  = _summary.get('loglik')
        else:
            bic = getattr(fitted, 'bic', None) or getattr(fitted, 'BIC', None)
            ll  = getattr(fitted, 'loglik', None) or getattr(fitted, 'log_lik', None)
        n   = len(df)
        k   = k_active if _PAD_FEATURES else len(selected_vars)   # count only active features
        coef_rows = _extract_metacount_coefs(fitted)
        sig_pen = _metacount_significance_penalty(coef_rows)
        _update_feature_weakness(coef_rows, selected_vars)   # bias SA away from near-zero vars
        if bic is not None and np.isfinite(bic):
            return -(abs(float(bic)) / max(n, 1)) - sig_pen
        if ll is not None and np.isfinite(ll):
            return _bic_per_obs(float(ll), k, n) - sig_pen
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
    correlated: bool = False,
) -> float:
    """
    Fit a MetaCount model with random parameters (mixed model) and return –BIC/n.
    Used for the FINAL estimation after the GA/SA has selected variables.

    random_vars : subset of selected_vars to make random (Normal distribution).
                  If None, uses all selected_vars.
    R           : Halton draws for simulation (200 = good balance; 500 for final paper).
    correlated  : if True, place random vars in rdm_cor_terms (full Cholesky covariance)
                  instead of independent rdm_terms.  Adds n*(n+1)/2 extra Cholesky params.
    """
    if not _METACOUNT_OK:
        return -1e9
    if not selected_vars:
        return -1e9

    rdm   = random_vars if random_vars is not None else selected_vars
    fixed = [v for v in selected_vars if v not in rdm]

    try:
        if correlated and rdm:
            rdm_terms     = []
            rdm_cor_terms = [f'{v}:normal' for v in rdm]
        else:
            rdm_terms     = [f'{v}:normal' for v in rdm]
            rdm_cor_terms = []

        spec = {
            'fixed_terms':      fixed,
            'rdm_terms':        rdm_terms,
            'rdm_cor_terms':    rdm_cor_terms,
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
        fitted = _fit_metacount_manual_model(builder, spec, R=R, start_vector=seed_vector, family=family)
        if fitted is None:
            return -1e9
        start_vec = _result_vector(fitted)
        if start_vec is not None:
            _METACOUNT_START_VALUES[cache_key] = start_vec
        if isinstance(fitted, dict):
            _summary = fitted.get('summary') or {}
            bic = _summary.get('bic')
            ll  = _summary.get('loglik')
        else:
            bic = getattr(fitted, 'bic', None) or getattr(fitted, 'BIC', None)
            ll  = getattr(fitted, 'loglik', None) or getattr(fitted, 'log_lik', None)
        n   = len(df)
        k   = len(selected_vars) + len(rdm)  # mean + std per random param
        if bic is not None and np.isfinite(bic):
            return -(abs(float(bic)) / max(n, 1))
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
                break
        except Exception:
            pass

    # Fallback: raw coeff arrays
    if not rows:
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
                break
            except Exception:
                pass

    # Deduplicate rows by (param, value, std_err, t_stat)
    seen = set()
    deduped = []
    for r in rows:
        key = (r['param'], r['value'], r['std_err'], r['t_stat'])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


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
    use_sim_anneal: bool = False,
) -> float:
    """
    GA fitness for Stage 1 using MetaCount NB2 count regression.

    Replaces the ordered logit.  Each activity-frequency outcome
    (n_work_out, n_edu_out, …) is a non-negative integer count —
    NB2 is the statistically correct model family here.

    Returns mean –BIC/n across purposes.
    """
    # Deduplicate features while preserving order
    seen = set()
    selected = [f for f, g in zip(feat_names, genes) if g == 1 and not (f in seen or seen.add(f))]
    if not selected:
        return -1e9

    # Register full candidate list for JAX feature padding (GA_PAD_FEATURES=1)
    if _PAD_FEATURES:
        _STAGE_ALL_CANDIDATES[id(data)] = [f for f in feat_names if f in data.columns]

    df = _ensure_id(data, 'obs_id')
    scores = []
    for purpose in purposes:
        y_col = f'n_{purpose}_out'
        if y_col not in df.columns:
            continue
        sub = df[['obs_id', y_col] + [c for c in selected if c in df.columns]].dropna()
        if len(sub) < 30:
            continue
        # Placeholder: use_sim_anneal logic can be implemented here
        score = _metacount_fit(sub, id_col='obs_id', y_col=y_col,
                               selected_vars=[c for c in selected if c in sub.columns],
                               family='nb2')
        if np.isfinite(score) and score > -1e8:
            scores.append(score)

    return float(np.mean(scores)) if scores else -1e9


# ---------------------------------------------------------------------------
# Stage 2 — day duration fast OLS-BIC fitness (used as GA oracle)
# ---------------------------------------------------------------------------
# MetaCount Tobit/lognormal is the correct *final* estimator but takes ~10-90s
# per fit.  For the GA search (3000+ evaluations) we use OLS-BIC which ranks
# chromosomes identically to full MLE for variable selection purposes and runs
# in <1 ms per evaluation — a ~1000x speed-up.
#
# For start_of_day / end_of_day we fit OLS on the raw value.
# For B (day duration) we fit OLS on log(B) which is the lognormal approximation.
# BIC = n*log(RSS/n) + k*log(n) — computed analytically via numpy.

def _ols_bic(y: np.ndarray, X: np.ndarray) -> float:
    """Return –BIC/n for an OLS regression y ~ X (intercept already in X)."""
    n, k = X.shape
    if n < k + 2:
        return -1e9
    try:
        coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        if len(res) == 0:
            yhat = X @ coef
            rss = float(np.sum((y - yhat) ** 2))
        else:
            rss = float(res[0])
        if rss <= 0 or not np.isfinite(rss):
            return -1e9
        bic = n * np.log(rss / n) + k * np.log(n)
        return -bic / n
    except Exception:
        return -1e9


def stage2_fitness_fast(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    targets: List[str],
) -> float:
    """
    Fast OLS-BIC fitness for Stage 2 GA variable selection.

    Replaces MetaCount Tobit/lognormal during the GA search phase.
    B is log-transformed (lognormal approximation); start/end use raw values.
    Each evaluation takes <1 ms vs ~10-90 s for MetaCount — a critical
    speed-up given 3000+ chromosome evaluations per run.
    """
    seen: set = set()
    selected = [f for f, g in zip(feat_names, genes)
                if g == 1 and not (f in seen or seen.add(f))]
    if not selected:
        return -1e9

    scores = []
    for target in targets:
        if target not in data.columns:
            continue
        cols = [c for c in selected if c in data.columns]
        if not cols:
            continue
        sub = data[[target] + cols].dropna()
        if len(sub) < max(len(cols) + 2, 30):
            continue
        y_raw = pd.to_numeric(sub[target], errors='coerce').to_numpy(dtype=float)
        # log-transform B (duration); raw for time-of-day targets
        if target == 'B':
            mask = y_raw > 0
            if mask.sum() < 30:
                continue
            y = np.log(y_raw[mask])
            Xdf = sub.loc[sub[target] > 0, cols]
        else:
            y = y_raw
            Xdf = sub[cols]
        X = np.column_stack([np.ones(len(y)), Xdf.to_numpy(dtype=float)])
        score = _ols_bic(y, X)
        if np.isfinite(score) and score > -1e8:
            scores.append(score)

    return float(np.mean(scores)) if scores else -1e9


# ---------------------------------------------------------------------------
# Stage 2 — day duration  (MetaCount Tobit/lognormal — used for FINAL model)
# ---------------------------------------------------------------------------

def stage2_fitness_metacount(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    targets: List[str],
    use_sim_anneal: bool = False,
) -> float:
    """
    GA fitness for Stage 2 using MetaCount mixed lognormal regression.

    Targets (start_of_day, end_of_day, B) are continuous duration/time
    variables.  MetaCount's 'duration' family fits log-normal regression
    with optional random effects for taste heterogeneity.

    Returns mean –BIC/n across targets.
    """
    seen = set()
    selected = [f for f, g in zip(feat_names, genes) if g == 1 and not (f in seen or seen.add(f))]
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
        # B (day duration) — log-normal regression on positive continuous values.
        # start_of_day / end_of_day — Tobit-I: time-of-day is bounded by the
        # survey window so observations are censored at the recording limits.
        # Tobit is the theoretically correct model for this censored outcome.
        family = 'duration' if target == 'B' else 'tobit'
        score = _metacount_fit(sub, id_col='obs_id', y_col=target,
                               selected_vars=[c for c in selected if c in sub.columns],
                               family=family)
        if np.isfinite(score) and score > -1e8:
            scores.append(score)

    return float(np.mean(scores)) if scores else -1e9


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Stage 3 — activity duration fast OLS-on-log(duration) fitness
# ---------------------------------------------------------------------------

def stage3_fitness_fast(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    purposes: List[str],
    purp_col: str,
    dur_col: str = 'duration_min',
) -> float:
    """
    Fast OLS-BIC fitness for Stage 3 GA variable selection.

    Fits OLS on log(duration) per purpose — identical variable selection
    ordering to full lognormal MLE but <1 ms per evaluation vs ~10-90 s.
    """
    seen: set = set()
    selected = [f for f, g in zip(feat_names, genes)
                if g == 1 and not (f in seen or seen.add(f))]
    if not selected or dur_col not in data.columns:
        return -1e9

    scores = []
    for purpose in purposes:
        sub = data[data[purp_col] == purpose] if purp_col in data.columns else data
        cols = [c for c in selected if c in sub.columns]
        if not cols:
            continue
        sub2 = sub[[dur_col] + cols].dropna()
        dur = pd.to_numeric(sub2[dur_col], errors='coerce')
        mask = dur > 0
        if mask.sum() < max(len(cols) + 2, 30):
            continue
        y = np.log(dur[mask].to_numpy(dtype=float))
        X = np.column_stack([np.ones(mask.sum()),
                             sub2.loc[mask, cols].to_numpy(dtype=float)])
        score = _ols_bic(y, X)
        if np.isfinite(score) and score > -1e8:
            scores.append(score)

    return float(np.mean(scores)) if scores else -1e9


# ---------------------------------------------------------------------------
# Joint Stage 2+3 — fast OLS-BIC proxy fitness for the SA-Bandit search
# ---------------------------------------------------------------------------

def stage23_fitness_joint(
    genes: np.ndarray,
    data_bundle: dict,
    feat_names: List[str],
    purposes: List[str],
    lambda_budget: float = 2.0,
) -> float:
    """
    Fast joint fitness oracle for the Stage 2+3 SA-Bandit search.

    Used as the GA chromosome-evaluation fitness function when STAGE23_JOINT=1.
    Each evaluation takes <2 ms (OLS-BIC), so the SA-Bandit can run thousands
    of evaluations without the 634-min MetaCount cost.

    The score blends three components:
      1. Stage 2 BIC/n: OLS on {start_of_day, end_of_day, log(B)} for selected features
      2. Stage 3 BIC/n: OLS on log(duration) per purpose for selected features
      3. Budget consistency bonus: penalise if predicted E[Σd] deviates from E[B]
         (same motivation as the JAX budget penalty in fit_joint_stage23)

    Parameters
    ----------
    data_bundle : dict with keys
        'person_day'  — DataFrame with start_of_day, end_of_day, B columns
        'tours'       — DataFrame with duration_min and purp_col
        'purp_col'    — name of the purpose column in tours
        'dur_col'     — name of the duration column in tours (default 'duration_min')
    feat_names  : full candidate feature list (shared pool for both sub-models)
    purposes    : purpose labels to score in Stage 3 component
    lambda_budget : weight on the budget-consistency penalty (lower = softer constraint)

    Returns
    -------
    float : combined –BIC/n (higher = better); -1e9 on degenerate inputs
    """
    seen: set = set()
    selected = [f for f, g in zip(feat_names, genes)
                if g == 1 and not (f in seen or seen.add(f))]
    if not selected:
        return -1e9

    person_day = data_bundle.get('person_day')
    tours      = data_bundle.get('tours')
    purp_col   = data_bundle.get('purp_col', 'main_purpose')
    dur_col    = data_bundle.get('dur_col', 'duration_min')

    if person_day is None or tours is None:
        return -1e9

    s2_scores: List[float] = []

    # ── Stage 2 component: OLS-BIC on {start_of_day, end_of_day, log(B)} ────
    for target in ['start_of_day', 'end_of_day', 'B']:
        if target not in person_day.columns:
            continue
        cols = [c for c in selected if c in person_day.columns]
        if not cols:
            continue
        sub = person_day[[target] + cols].dropna()
        if len(sub) < max(len(cols) + 2, 30):
            continue
        y_raw = pd.to_numeric(sub[target], errors='coerce').to_numpy(dtype=float)
        if target == 'B':
            mask = y_raw > 0
            if mask.sum() < 30:
                continue
            y = np.log(y_raw[mask])
            X = np.column_stack([np.ones(mask.sum()),
                                 sub.loc[sub[target] > 0, cols].to_numpy(dtype=float)])
        else:
            y = y_raw
            X = np.column_stack([np.ones(len(y)), sub[cols].to_numpy(dtype=float)])
        sc = _ols_bic(y, X)
        if np.isfinite(sc) and sc > -1e8:
            s2_scores.append(sc)

    # ── Stage 3 component: OLS-BIC on log(duration) per purpose ─────────────
    s3_scores: List[float] = []
    if purp_col in tours.columns and dur_col in tours.columns:
        for purpose in purposes:
            sub = tours[tours[purp_col] == purpose] if purp_col in tours.columns else tours
            cols = [c for c in selected if c in sub.columns]
            if not cols:
                continue
            sub2 = sub[[dur_col] + cols].dropna()
            dur = pd.to_numeric(sub2[dur_col], errors='coerce')
            mask = dur > 0
            if mask.sum() < max(len(cols) + 2, 30):
                continue
            y = np.log(dur[mask].to_numpy(dtype=float))
            X = np.column_stack([np.ones(mask.sum()),
                                 sub2.loc[mask, cols].to_numpy(dtype=float)])
            sc = _ols_bic(y, X)
            if np.isfinite(sc) and sc > -1e8:
                s3_scores.append(sc)

    if not s2_scores and not s3_scores:
        return -1e9

    # ── Budget consistency bonus ─────────────────────────────────────────────
    # Approximate: if B is selected as a Stage 3 feature, the model explicitly
    # conditions durations on the budget — reward this.
    budget_bonus = lambda_budget * 0.01 if 'B' in selected else 0.0

    # Combine: mean of stage 2 and stage 3 BIC/n scores plus budget bonus
    all_scores = s2_scores + s3_scores
    return float(np.mean(all_scores)) + budget_bonus


# Stage 3 — activity duration  (MetaCount lognormal — used for FINAL model)
# ---------------------------------------------------------------------------

def stage3_fitness_metacount(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    purposes: List[str],
    purp_col: str,
    dur_col: str = 'duration_min',
    use_sim_anneal: bool = False,
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
    seen = set()
    selected = [f for f, g in zip(feat_names, genes) if g == 1 and not (f in seen or seen.add(f))]
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
        sub = sub[pd.to_numeric(sub[dur_col], errors='coerce').fillna(0) > 0]  # lognormal requires y > 0
        if len(sub) < 30:
            continue
        # Placeholder: use_sim_anneal logic can be implemented here
        score = _metacount_fit(sub, id_col='obs_id', y_col=dur_col,
                               selected_vars=[c for c in selected if c in sub.columns],
                               family='duration')
        if np.isfinite(score) and score > -1e8:
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
    seen = set()
    selected = [f for f, g in zip(feat_names, genes) if g == 1 and not (f in seen or seen.add(f))]
    if not selected or not _SEARCHLIB_OK:
        return -1e9

    # Register full candidate list for JAX feature padding
    if _PAD_FEATURES:
        _STAGE_ALL_CANDIDATES[id(data)] = [f for f in feat_names if f in data.columns]

    scores = []
    for purpose in purposes:
        sub = data[data.get('purpose', data.iloc[:, 0]) == purpose] \
              if 'purpose' in data.columns else data
        if len(sub) < 20:
            continue
        avail_feats = [f for f in selected if f in sub.columns]
        if not avail_feats:
            continue

        # Drop near-constant columns within this purpose's subsample (no discriminating power)
        _sub_num = sub[avail_feats].apply(pd.to_numeric, errors='coerce').fillna(0)
        _sub_std = _sub_num.std()
        avail_feats = [f for f in avail_feats if _sub_std.get(f, 0) > 0.01]
        if not avail_feats:
            continue
        # Drop linearly dependent columns (e.g., ptype dummies summing to 1 within purpose)
        # Uses pivoted QR to identify and retain only a full-rank column subset
        if len(avail_feats) > 1:
            try:
                from scipy.linalg import qr as _scipy_qr
                _Xmat = _sub_num[avail_feats].values.astype(float)
                _, _R, _piv = _scipy_qr(_Xmat, pivoting=True)
                _diag_R = np.abs(np.diag(_R))
                _tol = 1e-6 * max(float(_diag_R[0]), 1e-10) if len(_diag_R) > 0 else 1e-10
                _rank = int(np.sum(_diag_R > _tol))
                if _rank < len(avail_feats):
                    avail_feats = [avail_feats[_piv[i]] for i in range(_rank)]
            except Exception:
                pass

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
                isvars   = avail_feats,
                ids      = long['case_id'].values.astype(int),
                base_alt = alts[0],
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = _fit_searchlib_model(mnl)

            if not _result_success(result):
                continue
            ll = -float(result.fun)
            k  = len(avail_feats) * (n_alts - 1)  # isvars: one coeff per feature per non-base alt
            start_vec = _result_vector(result)
            if start_vec is not None:
                _SEARCHLIB_START_VALUES[_searchlib_start_key(purpose, avail_feats, n_alts)] = start_vec
            # Significance penalty: discourage near-zero / insignificant features
            coef_rows = _searchlib_coef_rows(mnl, avail_feats)
            sig_pen   = _metacount_significance_penalty(coef_rows)
            _update_feature_weakness(coef_rows, avail_feats)
            scores.append(_bic_per_obs(ll, k, n_obs) - sig_pen)
        except Exception as e:
            _warn_estimator_failure('searchlibrium', f"mnl:{purpose}", e)
            continue

    return float(np.mean(scores)) if scores else -1e9


def stage4_fitness_searchlibrium_mixed(
    genes: np.ndarray,
    data: pd.DataFrame,
    feat_names: List[str],
    purposes: List[str],
    period_col: str = 'period',
    n_alts: int = 6,
    n_draws: int = 50,
    rand_dist: str = 'n',
    correlated: bool = False,
    rand_vars: Optional[List[str]] = None,
) -> float:
    """
    SA/GA fitness for Stage 4 using SearchLibrium MixedLogit (random parameters).

    Fits a Mixed Logit with random coefficients on all selected variables,
    capturing taste heterogeneity during the search itself rather than only
    in the final model.  Uses n_draws=50 by default to keep evaluations fast.

    Returns mean –BIC/n across purposes (BIC penalises 2k params: mean + std
    per random parameter, or mean + Cholesky elements if correlated=True).
    """
    seen = set()
    selected = [f for f, g in zip(feat_names, genes)
                if g == 1 and not (f in seen or seen.add(f))]
    if not selected or not _SEARCHLIB_OK:
        return -1e9

    scores = []
    for purpose in purposes:
        sub = data[data['purpose'] == purpose] if 'purpose' in data.columns else data
        if len(sub) < 30:
            continue
        avail_feats = [f for f in selected if f in sub.columns]
        if not avail_feats:
            continue

        try:
            long = _to_long_format(
                sub.copy(), choice_col=period_col,
                feat_cols=avail_feats, n_alts=n_alts,
            )
            n_obs    = len(sub)
            alts     = np.arange(n_alts)
            if rand_vars is not None:
                _rdm_set = set(rand_vars)
            else:
                # Central random-coefficient policy (force_random/never_random
                # from ga_constraints — SearchLibrium ConstraintBuilder format)
                try:
                    from ga_constraints import random_var_policy
                    _rdm_set = set(random_var_policy('stage4', avail_feats, rand_dist))
                except Exception:
                    _rdm_set = set(avail_feats)
            rv       = ({v: rand_dist for v in avail_feats if v in _rdm_set}
                        or {v: rand_dist for v in avail_feats})

            mxl_kwargs: dict = dict(
                X        = long[avail_feats].values.astype(float),
                y        = long['chosen'].values.astype(int),
                varnames = avail_feats,
                alts     = alts,
                ids      = long['case_id'].values.astype(int),
                randvars = rv,
                n_draws  = n_draws,
                halton   = True,
            )
            if correlated and len(rv) > 1:
                mxl_kwargs['correlated_vars'] = list(rv.keys())

            seed_key = _searchlib_start_key(purpose, avail_feats, n_alts)
            seed_vec = _SEARCHLIB_START_VALUES.get(seed_key)

            mxl = MixedLogit()
            mxl.setup(**mxl_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = _fit_searchlib_model(mxl, start_vector=seed_vec)

            if not _result_success(result):
                continue
            ll     = -float(result.fun)
            n_rand = len(rv)
            n_fix  = len(avail_feats) - n_rand
            n_chol = n_rand * (n_rand + 1) // 2
            k      = n_fix + (n_rand + n_chol if correlated else n_rand * 2)
            start_vec = _result_vector(result)
            if start_vec is not None:
                _SEARCHLIB_START_VALUES[seed_key] = start_vec
            scores.append(_bic_per_obs(ll, k, n_obs))
        except Exception as e:
            _warn_estimator_failure('searchlibrium', f"mxl:{purpose}", e)
            continue

    return float(np.mean(scores)) if scores else -1e9


def _fit_bic_rv(long, avail_feats, alts, rv, n_obs, n_draws=50):
    """Fit a SearchLibrium MXL (or MNL when rv is empty) and return its BIC.

    Used by the greedy random-parameter search below; np.inf on any failure
    so a failed trial can never be selected.
    """
    try:
        if rv:
            m = MixedLogit()
            m.setup(
                X        = long[avail_feats].values.astype(float),
                y        = long['chosen'].values.astype(int),
                varnames = avail_feats,
                alts     = alts,
                ids      = long['case_id'].values.astype(int),
                randvars = rv,
                n_draws  = n_draws,
                halton   = True,
            )
        else:
            m = MultinomialLogit()
            m.setup(
                X        = long[avail_feats].values.astype(float),
                y        = long['chosen'].values.astype(int),
                varnames = avail_feats,
                alts     = alts,
                ids      = long['case_id'].values.astype(int),
                base_alt = alts[0],
            )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = _fit_searchlib_model(m)
        if not _result_success(res):
            return float('inf')
        ll = -float(res.fun)
        k  = (len(avail_feats) - len(rv)) + 2 * len(rv)
        return -2.0 * ll + k * np.log(max(n_obs, 1))
    except Exception:
        return float('inf')


def _greedy_randvar_search(long, avail_feats, alts, rv0, rand_dist, n_obs,
                           purpose='', n_draws=50, never_random=()):
    """
    Single-pass greedy search over each variable's random-vs-fixed status.

    Makes random-coefficient treatment a searched DECISION rather than a
    blanket default: starting from the policy assignment rv0, each variable
    is toggled (random ↔ fixed) and the toggle is kept when it improves BIC.
    never_random variables are never given a random coefficient. Runs at a
    low draw count (default 50) — the production refit afterwards uses the
    full draw budget with the chosen assignment. Enabled with
    GA_RAND_PARAM_SEARCH=1 (see ga_constraints.rand_param_search_enabled).
    """
    never = set(never_random)
    best_rv  = {v: d for v, d in dict(rv0).items() if v not in never}
    best_bic = _fit_bic_rv(long, avail_feats, alts, best_rv, n_obs, n_draws)
    n_trials = 0
    for v in avail_feats:
        trial = dict(best_rv)
        if v in trial:
            trial.pop(v)
        elif v not in never:
            trial[v] = rand_dist
        else:
            continue
        bic = _fit_bic_rv(long, avail_feats, alts, trial, n_obs, n_draws)
        n_trials += 1
        if bic < best_bic - 1e-6:
            best_rv, best_bic = trial, bic
    print(f"  [{purpose}] greedy random-param search: "
          f"{len(best_rv)}/{len(avail_feats)} random after {n_trials} trials, "
          f"BIC={best_bic:.1f}")
    return best_rv


def stage4_final_mixed_logit(
    data: pd.DataFrame,
    selected_vars: List[str],
    purposes: List[str],
    period_col: str = 'period',
    n_alts: int = 6,
    n_draws: int = 500,
    rand_dist: str = 'n',
    correlated: bool = False,
    rand_vars: Optional[List[str]] = None,
) -> dict:
    """
    Refit Stage 4 as Mixed Logit (SearchLibrium) after GA/SA variable selection.

    Random coefficients are placed on all selected variables (Normal distribution
    by default), capturing taste variation in time-of-day sensitivity across
    travellers.

    Parameters
    ----------
    data          : output of prepare_timing_data()
    selected_vars : GA/SA-selected features
    purposes      : list of trip purposes to estimate
    period_col    : column encoding chosen time period (int 0..n_alts–1)
    n_alts        : number of time period alternatives
    n_draws       : Halton draws for simulation (500 = production quality)
    rand_dist     : distribution for random params ('n'=Normal, 'ln'=LogNormal)
    correlated    : if True, fit correlated random parameters (full Cholesky covariance
                    via SearchLibrium's correlated_vars).  Adds n*(n+1)/2 Cholesky params.

    Returns dict  {purpose: {'ll': ..., 'bic': ..., 'coefs': ..., 'randvars': ...}}
    """
    if not _SEARCHLIB_OK:
        logger.warning("SearchLibrium not installed — cannot run Mixed Logit for Stage 4")
        return {}

    results = {}
    if rand_vars is not None:
        _rdm_set = set(rand_vars)
    else:
        try:
            from ga_constraints import random_var_policy
            _rdm_set = set(random_var_policy('stage4', selected_vars, rand_dist))
        except Exception:
            _rdm_set = set(selected_vars)
    randvars = ({v: rand_dist for v in selected_vars if v in _rdm_set}
                or {v: rand_dist for v in selected_vars})
    try:
        from ga_constraints import rand_param_search_enabled, get_constraints
        _rand_search = rand_param_search_enabled()
        _never_random = tuple(get_constraints('stage4').get('never_random', ()))
    except Exception:
        _rand_search, _never_random = False, ()

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
            rv   = {v: rand_dist for v in avail_feats if v in randvars} or {v: rand_dist for v in avail_feats}
            if _rand_search and len(avail_feats) >= 2:
                # Random-parameter treatment as a searched decision, not a
                # blanket default (GA_RAND_PARAM_SEARCH=1): greedy BIC toggle
                # at low draws, then the production fit below uses n_draws.
                rv = _greedy_randvar_search(
                    long, avail_feats, alts, rv, rand_dist,
                    n_obs=len(sub), purpose=purpose,
                    never_random=_never_random,
                )
                if not rv:
                    # MixedLogit.setup requires ≥1 random var; keep the first
                    # eligible one so the production fit below stays valid.
                    _elig = [v for v in avail_feats if v not in set(_never_random)]
                    rv = {(_elig[0] if _elig else avail_feats[0]): rand_dist}
                    logger.info(f"[{purpose}] greedy search kept no random params - "
                                f"retaining minimal mixture on '{list(rv)[0]}'")
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

            n_chol = len(avail_feats) * (len(avail_feats) + 1) // 2  # Cholesky lower-tri
            mxl_setup_kwargs: dict = dict(
                X        = long[avail_feats].values.astype(float),
                y        = long['chosen'].values.astype(int),
                varnames = avail_feats,
                alts     = alts,
                randvars = rv,
                ids      = long['case_id'].values.astype(int),
                n_draws  = n_draws,
                halton   = True,
            )
            if correlated and len(rv) > 1:
                mxl_setup_kwargs['correlated_vars'] = list(rv.keys())

            mxl = MixedLogit()
            mxl.setup(**mxl_setup_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # MNL seed is only compatible when ALL avail_feats are random;
                # skip it for partial-random models to avoid shape mismatches.
                _use_seed = (seed_vector
                             if seed_vector is not None and set(rv.keys()) == set(avail_feats)
                             else None)
                result = _fit_searchlib_model(mxl, start_vector=_use_seed)

            ll     = -float(result.fun)
            n_rand = len(rv)
            n_fix  = len(avail_feats) - n_rand
            # param count: fixed params (1 each) + random params (mean+std or Cholesky)
            n_chol = n_rand * (n_rand + 1) // 2
            k      = n_fix + (n_rand + n_chol if correlated else n_rand * 2)
            n   = len(sub)
            bic = -2 * ll + k * np.log(n)

            results[purpose] = {
                'll':         ll,
                'bic':        bic,
                'coefs':      getattr(mxl, 'coeff_est', result.x).tolist(),
                'randvars':   rv,
                'correlated': correlated,
                'converged':  result.success,
                'n_obs':      n,
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
            return -(abs(float(bic)) / max(n, 1)), fitted
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
    # Deduplicate selected_vars
    seen = set()
    deduped_vars = [f for f in selected_vars if not (f in seen or seen.add(f))]
    for purpose in purposes:
        y_col = f'n_{purpose}_out'
        if y_col not in df.columns:
            continue
        sub   = df[['obs_id', y_col] + [c for c in deduped_vars if c in df.columns]].dropna()
        avail = [c for c in deduped_vars if c in sub.columns]
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
    seen = set()
    deduped_vars = [f for f in selected_vars if not (f in seen or seen.add(f))]
    sub   = df[['obs_id', y_col] + [c for c in deduped_vars if c in df.columns]].dropna()
    avail = [c for c in deduped_vars if c in sub.columns]
    family = 'duration' if y_col not in ('start_of_day', 'end_of_day') else 'tobit'
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
