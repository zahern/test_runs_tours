"""
larch_patches.py — runtime patches for Larch 6.0.x ("driftless" 6 line).

Apply once at import time from any module that drives Larch:

    import larch_patches; larch_patches.apply()

What it fixes
=============
1. Ridge-aware parameter covariance (the big one).
   Larch's `calculate_parameter_covariance` differentiates the PURE
   log-likelihood (`d2_loglike` on numba, `jax_param_cov` on jax). It cannot
   see the elastic-net/ridge penalty that `larch_regularization.elasticnet_objective`
   applies during estimation, so any collinear/flat direction keeps its ~0
   curvature → singular Hessian → missing/absurd std_err and p_value.
   This patch adds the quadratic penalty's second derivative
   (2·α·(1−l1_ratio)·w on the free-parameter diagonal — the L1 part
   contributes nothing to the Hessian away from 0) using the EXACT alpha /
   l1_ratio / reference / weights stashed on the model by
   `elasticnet_objective.__enter__`, so fit and inference differentiate the
   SAME penalised objective. With the lift, every eigenvalue is ≥ 2α(1−ratio):
   perfectly invertible; a genuinely flat direction gets a large-but-finite SE
   and correctly shows as insignificant instead of NaN.

2. Version guard. The pipeline (DataTree APIs, sharrow flows, logsums
   signatures) targets the Larch 6.0 line — a loud warning is logged for any
   other major version so environment drift is visible in the logs.

The patches are idempotent and no-op safely on Larch versions whose internals
differ (guarded by hasattr).
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_APPLIED = False


def _reg_terms(model):
    """Penalty spec stashed by elasticnet_objective.__enter__ (with defaults)."""
    alpha = float(getattr(model, "_larch_reg_alpha",
                          os.environ.get("GA_L1_L2_ALPHA", "1e-3")))
    l1_ratio = float(getattr(model, "_larch_reg_l1_ratio",
                             os.environ.get("GA_L1_RATIO", "0.5")))
    ref = getattr(model, "_larch_reg_ref", None)
    w = getattr(model, "_larch_reg_w", None)
    return alpha, l1_ratio, ref, w


def _penalty_diag(model, n) -> np.ndarray:
    """Diagonal of d²/dβ² of the quadratic elastic-net term, scaled into the
    TOTAL log-likelihood space (logloss penalty × total_weight) so it matches
    the sign/scale conventions of d2_loglike / jax_param_cov."""
    alpha, l1_ratio, _ref, w = _reg_terms(model)
    try:
        tw = float(model.total_weight())
    except Exception:                                          # noqa: BLE001
        tw = 1.0
    if not np.isfinite(tw) or tw <= 0:
        tw = 1.0
    diag = np.full(n, 2.0 * alpha * max(0.0, (1.0 - l1_ratio)) * tw,
                   dtype=float)
    if w is not None and len(w) == n:
        diag *= np.asarray(w, dtype=float)
    # holdfasted parameters are zeroed out of the covariance downstream anyway
    hf = getattr(model, "pholdfast", None)
    if hf is not None:
        hf = np.asarray(hf, dtype=bool)
        if hf.shape == (n,):
            diag[hf] = 0.0
    return diag


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    import larch

    # ── version guard ────────────────────────────────────────────────────
    ver = str(getattr(larch, "__version__", "?"))
    if not ver.startswith("6."):
        logger.warning(
            f"larch_patches: expected Larch 6.0.x ('driftless' 6 line), "
            f"found {ver!r} — DataTree/sharrow/logsums APIs may not match "
            "this pipeline.")

    # ── numba engine: ridge-aware d2_loglike ─────────────────────────────
    try:
        Model = larch.Model
    except Exception as e:                                      # noqa: BLE001
        Model = None
        logger.debug(f"larch_patches: larch.Model unavailable ({e})")

    try:
        from larch.model.numbamodel import NumbaModel
    except Exception as e:                                     # noqa: BLE001
        NumbaModel = None
        logger.debug(f"larch_patches: NumbaModel unavailable ({e})")

    if NumbaModel is not None and hasattr(NumbaModel, "d2_loglike") \
            and not getattr(NumbaModel.d2_loglike, "_ridge_patched", False):
        _orig_d2 = NumbaModel.d2_loglike

        def _d2_loglike_ridge(self, pvals=None, *args, **kwargs):
            hess = _orig_d2(self, pvals, *args, **kwargs)
            try:
                hess = np.asarray(hess, dtype=float)
                lift = _penalty_diag(self, hess.shape[0])
                if lift.any():
                    # d2_loglike returns d²(loglike)/dθ² (negative-definite
                    # information has the opposite sign); the penalty is added
                    # on the SAME sign convention as the base term so the
                    # caller's `hess = -d2_loglike` keeps its information
                    # interpretation with the ridge included.
                    hess = hess - lift
                return hess
            except Exception:                                  # noqa: BLE001
                return hess

        _d2_loglike_ridge._ridge_patched = True
        NumbaModel.d2_loglike = _d2_loglike_ridge
        logger.info("larch_patches: NumbaModel.d2_loglike → ridge-aware")

    # ── jax engine: ridge-aware jax_param_cov ────────────────────────────
    try:
        from larch.model.jaxmodel import Model as JaxModel
    except Exception as e:                                     # noqa: BLE001
        JaxModel = None
        logger.debug(f"larch_patches: jax Model unavailable ({e})")

    if JaxModel is not None and hasattr(JaxModel, "jax_param_cov") \
            and not getattr(JaxModel.jax_param_cov, "_ridge_patched", False):
        _orig_jpc = JaxModel.jax_param_cov

        def _jax_param_cov_ridge(self, pvals=None, *args, **kwargs):
            se, hess, ihess = _orig_jpc(self, pvals, *args, **kwargs)
            try:
                hess = np.asarray(hess, dtype=float).copy()
                lift = _penalty_diag(self, hess.shape[0])
                if lift.any():
                    hess = hess + lift          # information-matrix convention
                ihess = np.linalg.pinv(hess)
                se = np.sqrt(np.clip(np.diag(ihess), 0.0, None))
            except Exception:                                  # noqa: BLE001
                pass
            return se, hess, ihess

        _jax_param_cov_ridge._ridge_patched = True
        JaxModel.jax_param_cov = _jax_param_cov_ridge
        logger.info("larch_patches: jax_param_cov → ridge-aware")

    # ── logsums signature shim (Larch 5 → 6 API change) ──────────────────
    # Larch 5.x: Model.logsums(datatree=...).  Larch 6.x: logsums(x=None)
    # evaluates against the model's attached `.datatree`.  Wrap both model
    # classes so legacy `logsums(datatree=tree)` calls transparently attach
    # the tree (mangle/unmangle rebuilds the compiled flow) — this removes the
    # per-zone TypeError that made _logsums abandon the vectorised path.
    def _logsums_shim(orig):
        import functools

        @functools.wraps(orig)
        def wrapper(self, x=None, *, datatree=None, **kwargs):
            if datatree is not None:
                self.datatree = datatree
                try:
                    self.mangle()
                    self.unmangle()
                except Exception:                          # noqa: BLE001
                    pass
                x = None                       # evaluate against the new tree
            return orig(self, x, **kwargs)
        wrapper._datatree_shim = True
        return wrapper

    for _cls in (NumbaModel, JaxModel):
        if _cls is not None and hasattr(_cls, "logsums") \
                and not getattr(_cls.logsums, "_datatree_shim", False):
            setattr(_cls, "logsums", _logsums_shim(_cls.logsums))
            logger.info(f"larch_patches: {_cls.__name__}.logsums → datatree shim")

    # ── sync covariance results into pf (std_err / t_stat / p_value) ─────
    # Larch 6 stores SEs in the `pstderr` attribute + `covariance_matrix`
    # parameter array but does NOT populate the pf frame, so every downstream
    # reader of pf['std_err'] saw nothing and fell back to proxies. Wrap
    # calculate_parameter_covariance to write the (now ridge-aware) inference
    # columns straight into pf.
    def _cov_sync_shim(orig):
        import functools

        @functools.wraps(orig)
        def wrapper(self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            try:
                import pandas as _pd
                from scipy.stats import norm as _norm
                se = getattr(self, "pstderr", None)
                if se is None:
                    return result
                se = np.asarray(se, dtype=float)
                names = list(self.pnames)[:len(se)]
                se_s = _pd.Series(se, index=names)
                with np.errstate(divide="ignore", invalid="ignore"):
                    # BUGFIX 2026-09-04: t = value / se (was se / value, inverted).
                    t = _pd.Series(
                        np.asarray(self.pf["value"], dtype=float)[:len(se)],
                        index=names) / se_s.replace(0.0, np.nan)
                t = t.fillna(0.0)
                pv = 2.0 * _norm.sf(np.abs(t.to_numpy()))
                enriched = _pd.DataFrame({
                    "value": _pd.Series(np.asarray(self.pf["value"], dtype=float)[:len(se)],
                                        index=names),
                    "std_err": se_s, "t_stat": t, "p_value": _pd.Series(pv, index=names),
                })
                # Carry lock/bound flags into the enriched frame. It is served
                # through the pf property shim below, and every downstream
                # holdfast/min/max reader (stuck diagnostics, SE 0/0/1 stamps,
                # _final_pf, convergence gates) would otherwise see locked
                # params as free-but-unmoved (seen on larch 6.0.44: locked
                # c_size_term/log_correction reported free). self.pf here
                # resolves to the BASE frame (enriched not yet assigned).
                try:
                    _bpf = self.pf
                    for _fc in ('holdfast', 'minimum', 'maximum'):
                        if _fc in getattr(_bpf, 'columns', ()):
                            enriched[_fc] = _bpf[_fc].reindex(enriched.index)
                except Exception:
                    pass
                # pf is REBUILT from the parameter bucket on every access and
                # hardcodes its column list, so the enriched frame is served
                # through the pf property shim below (and cdap._inf_pf).
                self._inference_pf = enriched
            except Exception as _e:                        # noqa: BLE001
                logger.debug(f"larch_patches: pf covariance sync skipped: {_e}")
            return result
        wrapper._cov_sync = True
        return wrapper

    for _cls in (NumbaModel, JaxModel):
        if _cls is not None and hasattr(_cls, "calculate_parameter_covariance") \
                and not getattr(_cls.calculate_parameter_covariance, "_cov_sync", False):
            setattr(_cls, "calculate_parameter_covariance",
                    _cov_sync_shim(_cls.calculate_parameter_covariance))
            logger.info(f"larch_patches: {_cls.__name__}"
                        ".calculate_parameter_covariance → pf sync")

    # ── pf property shim ─────────────────────────────────────────────────
    # Serve the enriched inference frame (std_err/t_stat/p_value from the
    # ridge-aware covariance) through `model.pf` itself, so every direct
    # reader (`model.pf['std_err']` in stage 5's inference fallbacks, the
    # coefficient exporters, ...) sees finite values with no call-site
    # changes. The enriched frame is only served while it still matches the
    # live parameter values — any re-fit/re-set_cap automatically falls back
    # to the base frame.
    def _install_pf_shim(cls):
        if getattr(cls.pf.fget, "_enriched_shim", False):
            return
        base_fget = cls.pf.fget

        def fget2(self):
            base = base_fget(self)
            enr = getattr(self, "_inference_pf", None)
            if enr is not None:
                try:
                    if (len(enr) == len(base)
                            and {"std_err", "t_stat", "p_value"} <= set(enr.columns)
                            and not {"std_err"} <= set(base.columns)
                            and np.allclose(
                                np.asarray(enr["value"], dtype=float),
                                np.asarray(base["value"], dtype=float),
                                atol=1e-9, equal_nan=True)):
                        return enr
                except Exception:                          # noqa: BLE001
                    pass
            return base

        orig_prop = cls.pf
        pf = property(fget2, orig_prop.__set__, doc=orig_prop.__doc__)
        pf.fget._enriched_shim = True
        try:
            cls.pf = pf
            logger.info(f"larch_patches: {cls.__name__}.pf → enriched-frame shim")
        except Exception as e:                             # noqa: BLE001
            logger.debug(f"larch_patches: pf shim skipped for {cls.__name__}: {e}")

    for _cls in (NumbaModel, JaxModel):
        if _cls is not None:
            _install_pf_shim(_cls)

    # ── rho_sq_null() compatibility shim ───────────────────────────────────
    # Larch 6.0.44 has NO `rho_sq_null()` convenience method: confirmed on the
    # live install with `hasattr(larch.Model, 'rho_sq_null')` -> False and no
    # 'rho'-named members on Model at all. `loglike_null()` DOES exist.
    #
    # Every project call site therefore raised AttributeError and reported
    # rho2=nan for every model -- runs/exampville_ab_x2 (CDAP MNL and CDAP
    # Mixed Logit) and runs/mtc_ab_x2 both show it, while the baseline arm
    # reported rho2=None, which is what made it look like a real result rather
    # than a crash. stage5_destination_choice._rho_sq_null already worked
    # around it; the other call sites (cdap, ga_hypothesis_testing,
    # paper_model_forms, pipeline_logger, mtc_compare, exampville_compare) did
    # not. Install the method on the class so all of them work unchanged.
    def _rho_sq_null(self) -> float:
        """McFadden rho-squared vs the null (equal-shares) model: 1 - LL/LL_null."""
        ll = getattr(self, "_cached_ll", None)
        if ll is None:
            ll = self.loglike()
        ll = float(ll)
        if not np.isfinite(ll):
            return float("nan")
        ll_null = float(self.loglike_null())
        if ll_null == 0 or not np.isfinite(ll_null):
            return float("nan")
        return 1.0 - (ll / ll_null)

    _rho_sq_null._rho_sq_null_shim = True
    for _cls in (Model, NumbaModel, JaxModel):
        if _cls is None:
            continue
        if hasattr(_cls, "rho_sq_null") and getattr(
                getattr(_cls, "rho_sq_null", None), "_rho_sq_null_shim", False):
            continue
        try:
            setattr(_cls, "rho_sq_null", _rho_sq_null)
            logger.info(f"larch_patches: {_cls.__name__}"
                        ".rho_sq_null → compatibility shim")
        except Exception as e:                                 # noqa: BLE001
            logger.debug(f"larch_patches: rho_sq_null shim skipped for "
                         f"{_cls.__name__}: {e}")

    logger.info("larch_patches: applied")
