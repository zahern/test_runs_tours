"""
larch_jax_enforce.py — JAX-only policy for LARGE Larch models (371-zone pipeline).

Large models (destination MNL on 371 zones, mode NL, MXL, trip-dest, parking)
MUST be JAX-based so they fit and converge:

  * exact autodiff gradients (no numba finite-difference stalls / zero-gradient
    SLSQP-at-iteration-0 failures from jobs 23540863 / e25042407),
  * float64 (JAX defaults to float32 → flat LL → gtol stall),
  * CPU-only XLA (avoids cusolver_gesvdj_ffi CUDA Hessian error that the old
    GA_LARCH_ENGINE=numba workaround was papering over),
  * importance sampling (DEST_EST_SAMPLE_SIZE=20 → n_obs×20 rows, not n_obs×371).

Usage
-----
    import larch_jax_enforce; larch_jax_enforce.apply()   # once at import time

    # or explicitly:
    from larch_jax_enforce import ensure_jax_environment, make_jax_model, check_jax_health

Env knobs
---------
  LARCH_FORCE_JAX=1 (default) — force jax for large models; numba only as
    last-resort fallback. Set 0 to restore legacy OS-dependent ordering.
  LARCH_JAX_THRESHOLD=50000 — cases×alts above this counts as "large".
  JAX_ENABLE_X64=1, JAX_PLATFORMS=cpu, JAX_PLATFORM_NAME=cpu,
  CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL=3 — set idempotently here
    BEFORE jax is first imported (must precede any jax array creation).

What apply() does
-----------------
  1. ensure_jax_environment() — env vars + jax_enable_x64.
  2. Patch lx.Model.__init__ so explicit compute_engine='numba' is upgraded
     to 'jax' when LARCH_FORCE_JAX=1 (logged, fallback to numba if jax build
     fails so small/debug models never hard-crash).
  3. Patch known builder helpers (_lx_model, _lx_model_s6) in already-imported
     stage modules to JAX-first ordering (best-effort, no-op if absent).
  4. Expose check_jax_health(model) + estimate_row_guard() for pre-fit checks.

Idempotent, no-op safely when jax/larch are missing.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_APPLIED = False


def ensure_jax_environment() -> bool:
    """Set CPU-only float64 JAX env vars + enable x64. Returns True if jax importable."""
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=1')
    # Honour explicit JAX_ENABLE_X64=0 opt-out for anyone debugging float32.
    if os.environ.get('JAX_ENABLE_X64', '1') == '0':
        try:
            import jax  # noqa: F401
            return True
        except Exception:
            return False
    try:
        import jax
        try:
            jax.config.update('jax_enable_x64', True)
        except Exception as e:  # noqa: BLE001
            log.warning(f"larch_jax_enforce: jax_enable_x64 failed ({e})")
        try:
            import jax.numpy as jnp  # noqa: F401
            # Touching a tiny array forces XLA init now (fail fast, not mid-fit).
            _ = jnp.zeros(1, dtype=jnp.float64).block_until_ready()  # type: ignore[attr-defined]
        except Exception:
            pass
        if not jax.config.jax_enable_x64:
            log.warning("larch_jax_enforce: jax_enable_x64 is False — gradients unreliable in float32")
        return True
    except ImportError:
        log.warning("larch_jax_enforce: jax not installed — large JAX-only models cannot fit "
                    "(pip install 'jax[cpu]')")
        return False
    except Exception as e:  # noqa: BLE001
        log.warning(f"larch_jax_enforce: jax init failed ({e})")
        return False


def is_large_model(n_cases: int | None = None, n_alts: int | None = None,
                   n_cells: int | None = None) -> bool:
    """True when cases×alts exceeds LARCH_JAX_THRESHOLD (default 50k cells)."""
    try:
        thresh = int(os.environ.get('LARCH_JAX_THRESHOLD', '50000'))
    except Exception:
        thresh = 50000
    try:
        if n_cells is not None:
            return int(n_cells) > thresh
        if n_cases is not None and n_alts is not None:
            return int(n_cases) * int(n_alts) > thresh
    except Exception:
        pass
    # Unknown size → assume large when forcing (safe side: JAX handles both).
    return os.environ.get('LARCH_FORCE_JAX', '1').strip() != '0'


def estimate_row_guard(n_obs: int, n_zones: int, sample_size: int = 0) -> dict:
    """Memory guard for dense n_obs × n_zones idca expansions.

    Returns dict with full_rows, sampled_rows, est_gb (float64, ~30 cols) and
    a recommendation. Large dest models must use importance sampling
    (sample_size=DEST_EST_SAMPLE_SIZE, default 20) — full 371 enumeration at
    50k obs = 18.5M rows ≈ 4-5 GB and OOMs (job 23661073 @ 261 GB).
    """
    full = int(n_obs) * int(n_zones)
    samp = int(n_obs) * int(sample_size) if sample_size and sample_size > 0 else full
    # ~30 float64 cols × 8 bytes ≈ 240 B/row + pandas/xarray overhead ×2.
    est_gb = samp * 240 * 2 / 1e9
    return {
        'n_obs': int(n_obs), 'n_zones': int(n_zones),
        'sample_size': int(sample_size),
        'full_rows': full, 'sampled_rows': samp,
        'est_gb': round(float(est_gb), 2),
        'use_sampling': bool(sample_size and sample_size > 0 and full > samp),
        'oom_risk': bool(est_gb > 4.0 or full > 10_000_000),
    }


def check_jax_health(model, label: str = "model") -> tuple[bool, float]:
    """Probe penalized d_logloss at start values: finite & non-zero?

    Returns (ok, grad_norm). Mirrors smoke_jax_models._probe_start_gradient
    but without the elasticnet wrapper (caller wraps if needed).
    """
    try:
        import numpy as np
        x0 = np.asarray(model.pvals, dtype=float)
        g = np.asarray(model.d_logloss(x0), dtype=float)
        finite = bool(np.all(np.isfinite(g)))
        norm = float(np.linalg.norm(g)) if finite else float('nan')
        ok = finite and norm > 0.0
        if not ok:
            log.warning(f"larch_jax_enforce: {label} gradient unhealthy "
                        f"(finite={finite}, |g|={norm:.3e}, engine={getattr(model, 'compute_engine', '?')})")
        return ok, norm
    except Exception as e:  # noqa: BLE001
        log.warning(f"larch_jax_enforce: {label} gradient probe failed ({e!r})")
        return False, float('nan')


def make_jax_model(dt, lx_module=None, **kwargs):
    """Build lx.Model(dt, compute_engine='jax') with numba fallback.

    Ensures JAX env first, forces jax unless LARCH_FORCE_JAX=0 and an explicit
    engine was passed. Raises RuntimeError only if BOTH engines fail.
    """
    ensure_jax_environment()
    if lx_module is None:
        try:
            import larch as lx_module  # type: ignore[no-redef]
        except ImportError as e:
            raise RuntimeError("larch not installed") from e
    force = os.environ.get('LARCH_FORCE_JAX', '1').strip() != '0'
    req = kwargs.pop('compute_engine', 'jax')
    if force and str(req).lower() == 'numba':
        log.info("larch_jax_enforce: upgrading requested numba → jax (LARCH_FORCE_JAX=1)")
        req = 'jax'
    try:
        return lx_module.Model(dt, compute_engine=req, **kwargs)
    except Exception as e:  # noqa: BLE001
        log.debug(f"larch_jax_enforce: engine='{req}' failed ({e}), trying fallback")
        fallback = 'numba' if req == 'jax' else 'jax'
        return lx_module.Model(dt, compute_engine=fallback, **kwargs)


def _patch_lx_model_init() -> None:
    """Wrap lx.Model.__init__ to upgrade numba→jax under LARCH_FORCE_JAX=1."""
    try:
        import larch as lx
    except ImportError:
        return
    Model = getattr(lx, 'Model', None)
    if Model is None or getattr(Model.__init__, '_jax_enforced', False):
        return
    _orig_init = Model.__init__

    def _jax_init(self, *args, **kwargs):
        force = os.environ.get('LARCH_FORCE_JAX', '1').strip() != '0'
        if force and str(kwargs.get('compute_engine', '')).lower() == 'numba':
            log.info("larch_jax_enforce: lx.Model(compute_engine='numba') → 'jax' (large-model policy)")
            kwargs['compute_engine'] = 'jax'
        # Default to jax when no engine given and forcing is on.
        if force and 'compute_engine' not in kwargs:
            kwargs['compute_engine'] = 'jax'
        try:
            return _orig_init(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            # If jax build fails, retry once on numba so small models survive.
            if force and kwargs.get('compute_engine') == 'jax':
                log.warning(f"larch_jax_enforce: jax Model build failed ({e}) — retrying numba")
                kwargs['compute_engine'] = 'numba'
                return _orig_init(self, *args, **kwargs)
            raise

    _jax_init._jax_enforced = True  # type: ignore[attr-defined]
    Model.__init__ = _jax_init
    log.info("larch_jax_enforce: lx.Model.__init__ → JAX-enforced")


def _patch_known_builders() -> None:
    """Best-effort: nudge already-imported stage builders to JAX-first."""
    import sys
    for mod_name, fn_name in (
        ('stage5_destination_choice', '_lx_model'),
        ('stage6_mode_choice', '_lx_model_s6'),
        ('model_zoo', '_lx_engine'),
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        fn = getattr(mod, fn_name, None)
        if fn is None or getattr(fn, '_jax_enforced', False):
            continue
        # Builders already patched via source edits; just tag to avoid re-wrap.
        try:
            fn._jax_enforced = True  # type: ignore[attr-defined]
        except Exception:
            pass


def apply() -> None:
    """Apply all JAX-enforcement patches (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    ensure_jax_environment()
    try:
        _patch_lx_model_init()
    except Exception as e:  # noqa: BLE001
        log.debug(f"larch_jax_enforce: Model patch skipped ({e})")
    try:
        _patch_known_builders()
    except Exception as e:  # noqa: BLE001
        log.debug(f"larch_jax_enforce: builder patch skipped ({e})")
    # Also apply the ridge-aware covariance + logsums shims (safe, idempotent).
    try:
        import larch_patches
        larch_patches.apply()
    except Exception as e:  # noqa: BLE001
        log.debug(f"larch_jax_enforce: larch_patches.apply skipped ({e})")
    log.info("larch_jax_enforce: applied (large Larch models → JAX)")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ok = ensure_jax_environment()
    print(f"JAX available: {ok}")
    try:
        import jax
        print(f"jax {jax.__version__}, x64={jax.config.jax_enable_x64}")
    except Exception as e:
        print(f"jax import failed: {e}")
    apply()
    print("apply() done")
