"""
model_zoo.py — Unified estimator registry for the SEQ travel-model pipeline.

Wraps the following estimator families under a common ModelResult interface:

  Discrete-choice (DCM)
  ---------------------
  larch_mnl       — Larch 6 MNL (JAX autodiff)
  larch_nl        — Larch 6 Nested Logit (larch6 NestingTree, JAX autodiff)
  larch_mxl       — Larch 6 Mixed Logit (Halton-MC, JAX)
  sl_mnl          — SearchLibrium MNL (JAX backend when SL_JAX=1)
  sl_nested       — SearchLibrium NestedLogit (JAX backend)
  sl_ordered      — SearchLibrium Ordered Logit (JAX backend)
  pylogit_mnl     — pylogit MNL (scipy backend, cross-check)

  Count / continuous (via MCR)
  ----------------------------
  mcr_nb2         — NB2 via MetaCountRegressor (JAX)
  mcr_zip         — ZIP via MetaCountRegressor (JAX)
  mcr_tobit       — Tobit via MetaCountRegressor (JAX)
  mcr_lognormal   — Log-Normal via MetaCountRegressor (JAX)

    Survival / duration analysis
    ----------------------------
    aft_weibull     — Weibull AFT (MetaCountRegressor JAX)
    aft_lognormal   — Log-Normal AFT (MetaCountRegressor JAX)
    aft_loglogistic — Log-Logistic AFT (MetaCountRegressor JAX)
    cox_ph          — Legacy placeholder (disabled in JAX-only builds)

    Selection / binary
    ------------------
    probit          — Binary Probit (SearchLibrium JAX)
    heckman         — Heckman Two-Step selection model (SearchLibrium JAX + OLS step)

Usage
-----
>>> from model_zoo import ModelZoo, ESTIMATOR_REGISTRY
>>> zoo = ModelZoo(timing_df=df, y_col="chosen_id", feat_cols=["age","dist"])
>>> results = zoo.fit_all(["larch_mnl", "larch_nl", "probit"])
>>> best = zoo.select_best(results, criterion="bic")
>>> print(best.backend, best.bic)

For survival models, use the duration-specific interface:
>>> zoo2 = ModelZoo(timing_df=df, duration_col="duration", event_col="event")
>>> results2 = zoo2.fit_all(["aft_weibull", "aft_lognormal", "cox_ph"])

For Heckman:
>>> zoo3 = ModelZoo(timing_df=df)
>>> r = zoo3.fit_one("heckman",
...     participation_col="made_trip", sel_cols=["inc","veh"],
...     outcome_col="log_dist",       out_cols=["purpose","time_of_day"])
"""

# ── std / CPU env (must be first) ───────────────────────────────────────────
import os, sys, warnings
os.environ.setdefault("JAX_PLATFORM_NAME",     "cpu")
os.environ.setdefault("JAX_PLATFORMS",         "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",  "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")

for _src in [
    r"C:\Users\ahernz\source\SearchLibrium\src",
    r"C:\Users\ahernz\source\MetaCount\metacountregressor",
]:
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

import logging
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Optional heavy imports (lazy) ────────────────────────────────────────────
def _try_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ModelResult  — common output container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Standardised estimation result returned by every estimator."""
    backend:   str
    loglik:    float
    aic:       float
    bic:       float
    n_params:  int
    n_obs:     int
    params_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    t_stats:   Optional[pd.Series] = None
    converged: bool = True
    model_obj: Any = None
    extra:     Dict[str, Any] = field(default_factory=dict)

    # convenience ──────────────────────────────────────────────────────────────
    def significant_vars(self, t_thresh: float = 1.96) -> List[str]:
        """Return variable names where |t-stat| >= t_thresh."""
        if self.t_stats is None or self.t_stats.empty:
            return []
        return list(self.t_stats[self.t_stats.abs() >= t_thresh].index)

    def summary(self) -> pd.DataFrame:
        tbl = pd.DataFrame([{
            "backend":  self.backend,
            "loglik":   round(self.loglik, 2),
            "aic":      round(self.aic, 2),
            "bic":      round(self.bic, 2),
            "n_params": self.n_params,
            "n_obs":    self.n_obs,
            "converged": self.converged,
        }])
        return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Helper — Larch6 engine selector
# ─────────────────────────────────────────────────────────────────────────────

_LARCH_ENGINE = os.environ.get("LARCH_ENGINE", "jax")
_LARCH_FALLBACK = "numba"

def _lx_engine(require_jax: bool = False) -> str:
    """JAX-first engine selector (large-model policy: always JAX).

    Large Larch models (MNL/NL/MXL on 371 zones) must be JAX-based to fit.
    Honors LARCH_ENGINE only for small/debug models; large callers should pass
    require_jax=True. Ensures CPU float64 env via larch_jax_enforce.
    """
    # Default large-model policy: require JAX unless explicitly disabled.
    _force = os.environ.get("LARCH_FORCE_JAX", "1").strip() != "0"
    if _force:
        require_jax = True
    try:
        import larch_jax_enforce as _jfe  # noqa: F401
        try:
            _jfe.ensure_jax_environment()
        except Exception:
            pass
    except Exception:
        pass
    try:
        import jax  # noqa: F401
        return "jax"
    except ImportError:
        if require_jax:
            raise RuntimeError("JAX unavailable; cannot run JAX-only estimator "
                               "(large models require JAX: pip install jax[cpu])")
        return _LARCH_FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
# Period / IDCO helpers (shared with main_2.py logic)
# ─────────────────────────────────────────────────────────────────────────────

_PERIOD_MAP = {
    "early": 1, "am_peak": 2, "midday": 3,
    "pm_peak": 4, "evening": 5, "late": 6,
}

def _build_idco(df: pd.DataFrame,
                feat_cols: List[str],
                choice_col: str,
                period_map: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Convert long-format df to wide IDCO; encode choice as integer alt id."""
    pmap = period_map or _PERIOD_MAP
    out = df[feat_cols].copy()
    if choice_col in df.columns:
        out["chosen_id"] = df[choice_col].map(pmap).fillna(df[choice_col])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ── DCM estimators ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _fit_larch_mnl(df: pd.DataFrame,
                   feat_cols: List[str],
                   choice_col: str,
                   period_map: Optional[Dict[str, int]] = None) -> Optional[ModelResult]:
    """Larch 6 MNL — JAX autodiff, IDCO format."""
    try:
        import larch as lx
        from larch import P, X as LX
    except ImportError:
        log.warning("larch not installed; skipping larch_mnl")
        return None

    pmap = period_map or _PERIOD_MAP
    idco = _build_idco(df, feat_cols, choice_col, pmap)
    if len(idco.dropna()) < 30:
        return None
    try:
        alts = sorted(pmap.values())
        dt = lx.DataTree(main=lx.Dataset.construct.from_idco(idco, alts=alts))
        m = lx.Model(dt, compute_engine=_lx_engine())
        m.choice_co_code = "chosen_id"

        avail = [c for c in feat_cols if c in idco.columns]
        if avail:
            m.utility_co[alts[0]] = sum(P(f"b_{f}_base") * LX(f) for f in avail)
        inv_pmap = {v: k for k, v in pmap.items()}
        for alt_id in alts[1:]:
            label = inv_pmap[alt_id]
            m.utility_co[alt_id] = P(f"asc_{label}") + (
                sum(P(f"b_{f}_{label}") * LX(f) for f in avail) if avail else 0
            )

        m.maximize_loglike(quiet=True)
        ll = float(m.loglike())
        k  = int(m.n_params)
        n  = int(m.n_cases)
        pf = m.pf if hasattr(m, "pf") else pd.DataFrame()
        t_stats = pf["t_stat"] if "t_stat" in pf.columns else None

        return ModelResult(
            backend="larch_mnl", loglik=ll,
            aic=2*k - 2*ll, bic=k*np.log(n) - 2*ll,
            n_params=k, n_obs=n,
            params_df=pf, t_stats=t_stats, model_obj=m,
        )
    except Exception as e:
        log.debug(f"larch_mnl failed: {e}")
        return None


def _fit_larch_nl(df: pd.DataFrame,
                  feat_cols: List[str],
                  choice_col: str,
                  nests: Optional[Dict[str, List[int]]] = None,
                  period_map: Optional[Dict[str, int]] = None) -> Optional[ModelResult]:
    """
    Larch 6 Nested Logit using larch6-native NestingTree API.

    Default nests mirror TEMPORAL_NESTS:
        off_peak  -> [1, 6]  (early, late)
        peak      -> [2, 4]  (am_peak, pm_peak)
        shoulder  -> [3, 5]  (midday, evening)
    """
    try:
        import larch as lx
        from larch import P, X as LX
    except ImportError:
        log.warning("larch not installed; skipping larch_nl")
        return None

    pmap  = period_map or _PERIOD_MAP
    _nests = nests or {
        "off_peak":  [1, 6],
        "peak":      [2, 4],
        "shoulder":  [3, 5],
    }

    idco = _build_idco(df, feat_cols, choice_col, pmap)
    if len(idco.dropna()) < 30:
        return None
    try:
        alts = sorted(pmap.values())
        inv_pmap = {v: k for k, v in pmap.items()}

        # larch6 NestingTree (standalone, NOT m.graph)
        g = lx.NestingTree()
        for alt_id in alts:
            g.add_node(alt_id, name=inv_pmap.get(alt_id, str(alt_id)))
        for nest_name, children in _nests.items():
            g.new_node(name=nest_name, children=children,
                       parameter=f"Mu_{nest_name}")

        dt = lx.DataTree(main=lx.Dataset.construct.from_idco(idco, alts=alts))
        m  = lx.Model(dt, graph=g, compute_engine=_lx_engine())
        m.choice_co_code = "chosen_id"

        avail = [c for c in feat_cols if c in idco.columns]
        if avail:
            m.utility_co[alts[0]] = sum(P(f"b_{f}_base") * LX(f) for f in avail)
        for alt_id in alts[1:]:
            label = inv_pmap[alt_id]
            m.utility_co[alt_id] = P(f"asc_{label}") + (
                sum(P(f"b_{f}_{label}") * LX(f) for f in avail) if avail else 0
            )

        m.maximize_loglike(quiet=True)
        ll  = float(m.loglike())
        k   = int(m.n_params)
        n   = int(m.n_cases)
        pf  = m.pf if hasattr(m, "pf") else pd.DataFrame()
        t_s = pf["t_stat"] if "t_stat" in pf.columns else None

        return ModelResult(
            backend="larch_nl", loglik=ll,
            aic=2*k - 2*ll, bic=k*np.log(n) - 2*ll,
            n_params=k, n_obs=n,
            params_df=pf, t_stats=t_s, model_obj=m,
            extra={"nest_mus": {
                f"Mu_{nm}": float(pf.loc[f"Mu_{nm}", "value"])
                for nm in _nests if f"Mu_{nm}" in pf.index
            }},
        )
    except Exception as e:
        log.debug(f"larch_nl failed: {e}")
        return None


def _fit_larch_mxl(df: pd.DataFrame,
                   feat_cols: List[str],
                   choice_col: str,
                   n_draws: int = 200,
                   period_map: Optional[Dict[str, int]] = None) -> Optional[ModelResult]:
    """Larch 6 Mixed Logit (Halton draws, JAX-only)."""
    try:
        import larch as lx
        from larch import P, X as LX
    except ImportError:
        return None
    try:
        engine = _lx_engine(require_jax=True)
    except RuntimeError:
        log.debug("larch_mxl skipped: JAX unavailable")
        return None

    pmap = period_map or _PERIOD_MAP
    idco = _build_idco(df, feat_cols, choice_col, pmap)
    if len(idco.dropna()) < 60:
        return None
    try:
        alts = sorted(pmap.values())
        inv_pmap = {v: k for k, v in pmap.items()}
        avail = [c for c in feat_cols if c in idco.columns]
        rdm   = avail[:2]
        fixed = avail[2:]

        dt = lx.DataTree(main=lx.Dataset.construct.from_idco(idco, alts=alts))
        m  = lx.Model(dt, compute_engine=engine)
        m.choice_co_code = "chosen_id"

        # Base utility (alt 1)
        u_base = sum(P(f"b_{f}_base") * LX(f) for f in fixed)
        for rf in rdm:
            u_base = u_base + P(f"b_{rf}_base") * LX(rf)
        m.utility_co[alts[0]] = u_base

        for alt_id in alts[1:]:
            label = inv_pmap[alt_id]
            u = P(f"asc_{label}")
            for f in fixed:
                u = u + P(f"b_{f}_{label}") * LX(f)
            for rf in rdm:
                u = u + P(f"b_{rf}_{label}") * LX(rf)
            m.utility_co[alt_id] = u

        # Random parameters — Normal mixing
        for rf in rdm:
            for suffix in ["base"] + [inv_pmap[a] for a in alts[1:]]:
                pname = f"b_{rf}_{suffix}"
                if pname in m.pf.index:
                    m.mixtures[pname] = lx.mixtures.Normal(
                        P(pname), P(f"sigma_{rf}_{suffix}")
                    )
        m.n_draws = n_draws
        m.maximize_loglike(quiet=True)

        ll = float(m.loglike())
        k  = int(m.n_params)
        n  = int(m.n_cases)
        pf = m.pf if hasattr(m, "pf") else pd.DataFrame()

        return ModelResult(
            backend="larch_mxl", loglik=ll,
            aic=2*k - 2*ll, bic=k*np.log(n) - 2*ll,
            n_params=k, n_obs=n,
            params_df=pf, model_obj=m,
        )
    except Exception as e:
        log.debug(f"larch_mxl failed: {e}")
        return None


def _sl_setup(m, X, y_onehot, varnames, alts, **extra):
    """Version-tolerant SearchLibrium setup (one-hot → long-format fallback).

    SearchLibrium ≥ ~0.0.2xx requires y as a 1-D long-format chosen-alt vector
    ("y must be an array of one dimension in long format"); older releases
    take the one-hot matrix. Try one-hot first (old canonical), and on that
    specific ValueError retry with argmax long format. Anything else raises.
    """
    try:
        m.setup(X, y_onehot, varnames=varnames, alts=alts, **extra)
        return
    except ValueError as e:
        if 'long format' not in str(e).lower():
            raise
    import numpy as _np
    y_long = _np.asarray(y_onehot).argmax(axis=1).astype(int)
    m.setup(X, y_long, varnames=varnames, alts=alts, **extra)


def _fit_sl_mnl(df: pd.DataFrame,
                feat_cols: List[str],
                choice_col: str,
                use_jax: bool = True) -> Optional[ModelResult]:
    """SearchLibrium MNL (JAX autodiff when use_jax=True)."""
    try:
        from SearchLibrium import MultinomialLogit as SL_MNL
    except ImportError:
        return None
    try:
        X = df[feat_cols].values
        J = df[choice_col].nunique()
        y = pd.get_dummies(df[choice_col]).values.astype(float)
        m = SL_MNL(_jax=use_jax)
        _sl_setup(m, X, y, varnames=feat_cols, alts=list(range(J)))
        m.fit()
        ll  = float(m.loglik)
        k   = X.shape[1] * (J - 1) + (J - 1)
        n   = len(df)
        bic = k * np.log(n) - 2 * ll
        return ModelResult(
            backend="sl_mnl", loglik=ll,
            aic=2*k - 2*ll, bic=bic,
            n_params=k, n_obs=n, model_obj=m,
        )
    except Exception as e:
        log.debug(f"sl_mnl failed: {e}")
        return None


def _fit_sl_nested(df: pd.DataFrame,
                   feat_cols: List[str],
                   choice_col: str,
                   nests: Optional[Dict[str, List[int]]] = None,
                   use_jax: bool = True) -> Optional[ModelResult]:
    """SearchLibrium NestedLogit (JAX autodiff when use_jax=True)."""
    try:
        from SearchLibrium import NestedLogit as SL_NL
    except ImportError:
        return None
    _nests = nests or {"off_peak": [0, 5], "peak": [1, 3], "shoulder": [2, 4]}
    try:
        X = df[feat_cols].values
        J = df[choice_col].nunique()
        y = pd.get_dummies(df[choice_col]).values.astype(float)
        m = SL_NL(_jax=use_jax)
        _sl_setup(m, X, y, varnames=feat_cols, alts=list(range(J)), nests=_nests)
        m.fit()
        ll  = float(m.loglik)
        k   = X.shape[1] + (J - 1) + len(_nests)
        n   = len(df)
        bic = k * np.log(n) - 2 * ll
        return ModelResult(
            backend="sl_nested", loglik=ll,
            aic=2*k - 2*ll, bic=bic,
            n_params=k, n_obs=n, model_obj=m,
        )
    except Exception as e:
        log.debug(f"sl_nested failed: {e}")
        return None


def _fit_sl_ordered(df: pd.DataFrame,
                    feat_cols: List[str],
                    choice_col: str,
                    use_jax: bool = True) -> Optional[ModelResult]:
    """SearchLibrium Ordered Logit (JAX autodiff)."""
    try:
        from SearchLibrium import OrderedLogit as SL_ORL
    except ImportError:
        return None
    try:
        X = df[feat_cols].values
        y = df[choice_col].values.astype(int)
        J = int(y.max()) + 1
        m = SL_ORL(_jax=use_jax, X=X, y=y, J=J)
        m.fit()
        ll  = float(m.loglik)
        k   = X.shape[1] + (J - 1)
        n   = len(df)
        bic = k * np.log(n) - 2 * ll
        return ModelResult(
            backend="sl_ordered", loglik=ll,
            aic=2*k - 2*ll, bic=bic,
            n_params=k, n_obs=n, model_obj=m,
        )
    except Exception as e:
        log.debug(f"sl_ordered failed: {e}")
        return None


def _fit_pylogit_mnl(df: pd.DataFrame,
                     feat_cols: List[str],
                     choice_col: str) -> Optional[ModelResult]:
    """pylogit MNL cross-check (scipy backend)."""
    try:
        import pylogit as pl
    except ImportError:
        return None
    try:
        choices = df[choice_col].astype("category")
        alts = list(choices.cat.categories)
        rows = []
        for i, row in df.iterrows():
            for a in alts:
                r = {"obs_id": i, "alt": a, "chosen": int(row[choice_col] == a)}
                for f in feat_cols:
                    r[f] = row[f]
                rows.append(r)
        long_df = pd.DataFrame(rows)
        alt_spec = {f: "all_same" for f in feat_cols}
        m = pl.create_choice_model(
            data=long_df, alt_id_col="alt", obs_id_col="obs_id",
            choice_col="chosen", specification=alt_spec,
            names={f: [f"{f}"] for f in feat_cols},
            model_type="MNL",
        )
        m.fit_mle(np.zeros(len(feat_cols)))
        ll  = float(m.log_likelihood)
        k   = len(feat_cols)
        n   = len(df)
        bic = k * np.log(n) - 2 * ll
        params_df = pd.DataFrame({"coef": m.params, "stderr": m.standard_errors})
        t_stats   = pd.Series(m.tvalues, index=params_df.index)
        return ModelResult(
            backend="pylogit_mnl", loglik=ll,
            aic=2*k - 2*ll, bic=bic,
            n_params=k, n_obs=n,
            params_df=params_df, t_stats=t_stats, model_obj=m,
        )
    except Exception as e:
        log.debug(f"pylogit_mnl failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ── MCR estimators ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _mcr_fit(df: pd.DataFrame,
             y_col: str,
             feat_cols: List[str],
             family: str = "nb2",
             engine: str = "jax") -> Optional[ModelResult]:
    """
    Generic MCR wrapper for count/continuous families:
    nb2, zip, zinb, tobit, lognormal, gaussian, poisson.

    Delegates construction/fitting to ga_external_estimators'
    version-tolerant layer (_create_metacount_builder +
    _fit_metacount_manual_model) so model_zoo tracks the INSTALLED
    metacountregressor API instead of a hardcoded one — the 1.0.151 release
    replaced ExperimentBuilder(X=, y=, varnames=, ...) with
    ExperimentBuilder(df, id_col=, y_col=, ...) + fit_manual_model, which
    broke every direct old-style call. Returns None when unusable; the
    registry treats None as backend-unavailable and tries the next backend.
    (engine is accepted for signature compatibility; JAX is the package
    default in 1.0.151.)
    """
    try:
        from ga_external_estimators import (
            _create_metacount_builder, _fit_metacount_manual_model,
            _extract_metacount_coefs, _METACOUNT_OK,
        )
    except ImportError:
        return None
    if not _METACOUNT_OK:
        return None
    try:
        feats = [c for c in list(feat_cols or []) if c in df.columns]
        if not feats or y_col not in df.columns:
            return None
        y = pd.to_numeric(df[y_col], errors='coerce')
        if int(y.notna().sum()) < 30:
            return None
        df2 = df[[y_col] + feats].copy()
        df2['_mcr_id'] = np.arange(len(df2))
        builder = _create_metacount_builder(
            df2, id_col='_mcr_id', y_col=y_col, family=family)
        spec = {
            'fixed_terms':      feats,
            'rdm_terms':        [],
            'rdm_cor_terms':    [],
            'grouped_terms':    [],
            'hetro_in_means':   [],
            'zi_terms':         [],
            'membership_terms': [],
            'dispersion':       1 if family in ('nb2', 'negbin') else 0,
            'latent_classes':   1,
        }
        R = int(os.environ.get('MCR_FIT_DRAWS', '200'))
        fitted = _fit_metacount_manual_model(
            builder, spec, R=R, family=family)
        if fitted is None:
            return None
        # fit_manual_model returns dict {'summary':{'bic':…,'loglik':…},…};
        # older API used attributes — accept both, derive what's missing.
        if isinstance(fitted, dict):
            _summary = fitted.get('summary') or {}
            ll = _summary.get('loglik')
            bic = _summary.get('bic')
        else:
            ll = (getattr(fitted, 'loglik', None)
                  or getattr(fitted, 'log_lik', None))
            bic = (getattr(fitted, 'bic', None)
                   or getattr(fitted, 'BIC', None))
        n = len(df2)
        k = len(feats) + 1
        try:
            ll = float(ll)
        except Exception:
            ll = float('nan')
        if not np.isfinite(ll):
            return None
        try:
            bic = float(bic)
        except Exception:
            bic = float('nan')
        if not np.isfinite(bic):
            bic = k * np.log(max(n, 2)) - 2.0 * ll
        rows = _extract_metacount_coefs(fitted)
        if rows:
            pf = pd.DataFrame(rows)
            try:
                t_s = pd.Series(
                    [float(r.get('t_stat', float('nan'))) for r in rows],
                    index=[str(r.get('param', f'x{i}'))
                           for i, r in enumerate(rows)])
            except Exception:
                t_s = None
        else:
            pf, t_s = pd.DataFrame(), None
        return ModelResult(
            backend=f"mcr_{family}", loglik=ll,
            aic=2*k - 2*ll, bic=bic,
            n_params=k, n_obs=n,
            params_df=pf, t_stats=t_s, model_obj=fitted,
        )
    except Exception as e:
        log.debug(f"mcr_{family} failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ── Survival / duration estimators ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _fit_aft(df: pd.DataFrame,
             duration_col: str,
             feat_cols: List[str],
             event_col: Optional[str] = None,
             family: str = "lognormal") -> Optional[ModelResult]:
    """
    MetaCountRegressor JAX-native survival wrapper with correlated
    random-parameter simulation.
    """
    try:
        from metacountregressor import (
            LogLogisticRandomEffectsAFTFitter,
            LogNormalRandomEffectsAFTFitter,
            WeibullRandomEffectsAFTFitter,
        )
    except ImportError:
        log.warning("metacountregressor survival models not installed; AFT skipped")
        return None

    _fitters = {
        "lognormal":   LogNormalRandomEffectsAFTFitter,
        "weibull":     WeibullRandomEffectsAFTFitter,
        "loglogistic": LogLogisticRandomEffectsAFTFitter,
    }
    if family not in _fitters:
        raise ValueError(f"Unknown AFT family '{family}'. Choose from {list(_fitters)}")

    try:
        cols = feat_cols + [duration_col]
        if event_col and event_col in df.columns:
            cols = cols + [event_col]
        data = df[list(dict.fromkeys(cols))].dropna()

        random_terms = feat_cols[: min(2, len(feat_cols))]
        fitter = _fitters[family](
            random_terms=random_terms,
            correlated_random_terms=random_terms,
            n_draws=200,
        )
        fitter.fit(
            data,
            duration_col=duration_col,
            event_col=event_col if event_col and event_col in data.columns else None,
            feature_cols=feat_cols,
        )

        ll  = float(fitter.log_likelihood_)
        k   = int(getattr(fitter, "params_", pd.Series(dtype=float)).shape[0])
        n   = len(data)
        bic = k * np.log(len(data)) - 2 * ll
        converged = bool(getattr(fitter, "converged_", True))
        if not converged:
            log.warning(
                "aft_%s: optimizer did not fully converge — results may be unreliable. "
                "Consider increasing maxiter or checking data quality.", family
            )

        summary = fitter.summary_frame()
        t_stats: Optional[pd.Series] = None
        if "z" in summary.columns:
            t_stats = summary["z"].rename("t_stat")
        elif "t" in summary.columns:
            t_stats = summary["t"].rename("t_stat")

        return ModelResult(
            backend=f"mcr_jax_aft_{family}", loglik=ll,
            aic=2*k - 2*ll, bic=bic,
            n_params=k, n_obs=n,
            params_df=summary, t_stats=t_stats,
            converged=converged,
            model_obj=fitter,
            extra={
                "random_terms": list(getattr(fitter, "random_terms", [])),
                "correlated_random_terms": list(getattr(fitter, "correlated_random_terms", [])),
            },
        )
    except Exception as e:
        log.debug(f"aft_{family} failed: {e}")
        return None


def _fit_cox_ph(df: pd.DataFrame,
                duration_col: str,
                feat_cols: List[str],
                event_col: Optional[str] = None) -> Optional[ModelResult]:
    """Legacy placeholder: CoxPH disabled to keep survival stack JAX-only."""
    log.warning("cox_ph is disabled in this build; use mcr_jax_aft_* models instead")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ── Selection / binary estimators ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _fit_probit(df: pd.DataFrame,
                y_col: str,
                feat_cols: List[str]) -> Optional[ModelResult]:
    """SearchLibrium JAX Binary Probit."""
    try:
        from SearchLibrium import BinaryProbit
    except ImportError:
        log.warning("SearchLibrium BinaryProbit not installed; probit skipped")
        return None
    try:
        data = df[[y_col] + feat_cols].dropna()
        model = BinaryProbit(_jax=True)
        model.setup(data[feat_cols].to_numpy(), data[y_col].to_numpy(), varnames=feat_cols)
        result = model.fit(disp=False)
        ll  = float(model.loglik)
        k   = int(len(model.coeff_est))
        n   = int(len(data))
        bic = float(model.bic)
        params_df = model.summary_frame().rename(columns={"z": "t_stat"})
        return ModelResult(
            backend="sl_jax_probit", loglik=ll,
            aic=float(model.aic), bic=bic,
            n_params=k, n_obs=n,
            params_df=params_df, t_stats=params_df["t_stat"],
            converged=model.converged, model_obj=result,
        )
    except Exception as e:
        log.debug(f"probit failed: {e}")
        return None


def _fit_heckman(df: pd.DataFrame,
                 participation_col: str,
                 outcome_col: str,
                 sel_cols: List[str],
                 out_cols: List[str]) -> Optional[ModelResult]:
    """SearchLibrium JAX Heckman two-step selection model."""
    try:
        from SearchLibrium import HeckmanTwoStep
    except ImportError:
        log.warning("SearchLibrium HeckmanTwoStep not installed; heckman skipped")
        return None
    try:
        data = df[[participation_col, outcome_col] + sel_cols + out_cols].dropna()
        model = HeckmanTwoStep(_jax=True)
        model.setup(
            data[sel_cols].to_numpy(),
            data[participation_col].to_numpy(),
            data[out_cols].to_numpy(),
            data[outcome_col].to_numpy(),
            selection_varnames=sel_cols,
            outcome_varnames=out_cols,
        )
        result = model.fit(disp=False)
        ll  = float(model.loglik)
        k   = int(len(model.coeff_est))
        n   = int(len(data))
        bic = float(model.bic)
        params_df = model.summary_frame().rename(columns={"z": "t_stat"})
        probit_result = result["probit"]
        ols_result = result["ols"]
        lambda_val = float(ols_result.params.get("IMR", np.nan))
        lambda_t   = float(ols_result.tvalues.get("IMR", np.nan))

        return ModelResult(
            backend="sl_jax_heckman", loglik=ll,
            aic=float(model.aic), bic=bic,
            n_params=k, n_obs=n,
            params_df=params_df,
            t_stats=params_df["t_stat"],
            converged=model.converged,
            model_obj=result,
            extra={
                "probit_result": probit_result,
                "lambda_coef":   lambda_val,
                "lambda_t":      lambda_t,
                "imr_col":       "IMR",
            },
        )
    except Exception as e:
        log.debug(f"heckman failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ── Registry ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

#: Map estimator key → callable(df, feat_cols, **kwargs) → Optional[ModelResult]
ESTIMATOR_REGISTRY: Dict[str, Any] = {
    # DCM
    "larch_mnl":    _fit_larch_mnl,
    "larch_nl":     _fit_larch_nl,
    "larch_mxl":    _fit_larch_mxl,
    "sl_mnl":       _fit_sl_mnl,
    "sl_nested":    _fit_sl_nested,
    "sl_ordered":   _fit_sl_ordered,
    "sl_probit":    _fit_probit,
    "sl_heckman":   _fit_heckman,
    "pylogit_mnl":  _fit_pylogit_mnl,
    # Count/continuous
    "mcr_nb2":      lambda df, y, f, **kw: _mcr_fit(df, y, f, family="nb2",      **kw),
    "mcr_zip":      lambda df, y, f, **kw: _mcr_fit(df, y, f, family="zip",      **kw),
    "mcr_zinb":     lambda df, y, f, **kw: _mcr_fit(df, y, f, family="zinb",     **kw),
    "mcr_tobit":    lambda df, y, f, **kw: _mcr_fit(df, y, f, family="tobit",    **kw),
    "mcr_lognormal":lambda df, y, f, **kw: _mcr_fit(df, y, f, family="lognormal",**kw),
    "mcr_gaussian": lambda df, y, f, **kw: _mcr_fit(df, y, f, family="gaussian", **kw),
    # Survival
    "aft_lognormal":  lambda df, dc, f, **kw: _fit_aft(df, dc, f, family="lognormal",   **kw),
    "aft_weibull":    lambda df, dc, f, **kw: _fit_aft(df, dc, f, family="weibull",      **kw),
    "aft_loglogistic":lambda df, dc, f, **kw: _fit_aft(df, dc, f, family="loglogistic",  **kw),
    "cox_ph":         _fit_cox_ph,
    # Binary / selection
    "probit":   _fit_probit,
    "heckman":  _fit_heckman,
}


SEARCH_PRESETS: Dict[str, Dict[str, Any]] = {
    "core_fixed": {
        "description": "HHTS stages 0-4 core run with fixed historical specs.",
        "pipeline_mode": "core",
        "estimators": ["larch_mnl", "larch_nl", "sl_mnl", "sl_nested"],
    },
    "nested_fast": {
        "description": "Fast nested-logit SA search for stage-4 timing cells.",
        "pipeline_mode": "search",
        "search_model": "nested",
        "n_iter": 150,
        "temperature": 2.0,
        "estimators": ["larch_nl", "sl_nested"],
    },
    "nested_standard": {
        "description": "Standard nested-logit SA search budget.",
        "pipeline_mode": "search",
        "search_model": "nested",
        "n_iter": 400,
        "temperature": 2.0,
        "estimators": ["larch_nl", "sl_nested", "aft_lognormal"],
    },
    "mnl_fast": {
        "description": "Fast multinomial-logit SA search budget.",
        "pipeline_mode": "search",
        "search_model": "mnl",
        "n_iter": 150,
        "temperature": 1.5,
        "estimators": ["larch_mnl", "sl_mnl"],
    },
    "selection_screen": {
        "description": "Binary selection screen using SearchLibrium probit + Heckman.",
        "pipeline_mode": "search",
        "search_model": "mnl",
        "n_iter": 120,
        "temperature": 1.25,
        "estimators": ["sl_probit", "sl_heckman"],
    },
    "survival_screen": {
        "description": "Duration screen using MetaCount random-effects AFT models.",
        "pipeline_mode": "search",
        "search_model": "nested",
        "n_iter": 200,
        "temperature": 1.75,
        "estimators": ["aft_lognormal", "aft_weibull", "aft_loglogistic"],
    },
}


def get_search_presets() -> Dict[str, Dict[str, Any]]:
    return {key: value.copy() for key, value in SEARCH_PRESETS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# ── ModelZoo — high-level orchestrator ────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class ModelZoo:
    """
    High-level orchestrator that fits multiple estimator families on a
    single dataset and selects the best model by BIC (or other criterion).

    Parameters
    ----------
    timing_df : pd.DataFrame
        Input data frame.
    feat_cols : list[str]
        Covariate column names.
    choice_col : str, optional
        Column with choice/category labels (for DCM models).
    y_col : str, optional
        Column with numeric outcome (for count/continuous models).
    duration_col : str, optional
        Column with duration/time-to-event (for survival models).
    event_col : str, optional
        Column with event indicator 0/1 (for survival models).
    period_map : dict, optional
        {period_label: alt_id} mapping; defaults to _PERIOD_MAP.
    nests : dict, optional
        {nest_name: [alt_ids]} for nested logit; defaults to temporal nests.
    use_jax_sl : bool
        Whether to pass _jax=True to SearchLibrium (default True).
    """

    def __init__(self,
                 timing_df: pd.DataFrame,
                 feat_cols: Optional[List[str]] = None,
                 choice_col: Optional[str] = None,
                 y_col: Optional[str] = None,
                 duration_col: Optional[str] = None,
                 event_col: Optional[str] = None,
                 period_map: Optional[Dict[str, int]] = None,
                 nests: Optional[Dict[str, List[int]]] = None,
                 use_jax_sl: bool = True):
        self.df          = timing_df
        self.feat_cols   = feat_cols or []
        self.choice_col  = choice_col
        self.y_col       = y_col
        self.duration_col = duration_col
        self.event_col   = event_col
        self.period_map  = period_map or _PERIOD_MAP
        self.nests       = nests
        self.use_jax_sl  = use_jax_sl
        self._results: Dict[str, ModelResult] = {}

    # ── internals ─────────────────────────────────────────────────────────────

    def _dispatch(self, key: str, **override_kwargs) -> Optional[ModelResult]:
        """Route to the appropriate estimator with correct positional args."""
        df  = self.df
        fc  = self.feat_cols

        # DCM estimators
        if key in ("larch_mnl", "sl_mnl", "pylogit_mnl"):
            return ESTIMATOR_REGISTRY[key](df, fc, self.choice_col or "choice",
                                           **override_kwargs)
        if key == "larch_nl":
            return _fit_larch_nl(df, fc, self.choice_col or "choice",
                                  nests=self.nests, period_map=self.period_map,
                                  **override_kwargs)
        if key == "larch_mxl":
            return _fit_larch_mxl(df, fc, self.choice_col or "choice",
                                   period_map=self.period_map, **override_kwargs)
        if key == "sl_nested":
            return _fit_sl_nested(df, fc, self.choice_col or "choice",
                                   nests=self.nests, use_jax=self.use_jax_sl,
                                   **override_kwargs)
        if key == "sl_ordered":
            return _fit_sl_ordered(df, fc, self.choice_col or "choice",
                                    use_jax=self.use_jax_sl, **override_kwargs)

        # Count / continuous
        if key.startswith("mcr_"):
            family = key[4:]
            return _mcr_fit(df, self.y_col or "", fc, family=family,
                            **override_kwargs)

        # Survival
        if key.startswith("aft_"):
            family = key[4:]
            return _fit_aft(df, self.duration_col or "", fc,
                            event_col=self.event_col, family=family,
                            **override_kwargs)
        if key == "cox_ph":
            return _fit_cox_ph(df, self.duration_col or "", fc,
                               event_col=self.event_col, **override_kwargs)

        # Binary / selection
        if key in ("probit", "sl_probit"):
            return _fit_probit(df, self.y_col or "", fc, **override_kwargs)
        if key in ("heckman", "sl_heckman"):
            kw = dict(
                participation_col=override_kwargs.pop("participation_col",
                                                      self.y_col or ""),
                outcome_col=override_kwargs.pop("outcome_col", ""),
                sel_cols=override_kwargs.pop("sel_cols", fc),
                out_cols=override_kwargs.pop("out_cols", fc),
            )
            return _fit_heckman(df, **kw)

        raise KeyError(f"Unknown estimator key: '{key}'. "
                       f"Available: {list(ESTIMATOR_REGISTRY)}")

    # ── public API ────────────────────────────────────────────────────────────

    def fit_one(self, key: str, **kwargs) -> Optional[ModelResult]:
        """Fit a single estimator. Returns ModelResult or None on failure."""
        result = self._dispatch(key, **kwargs)
        if result is not None:
            self._results[key] = result
        return result

    def fit_all(self,
                keys: Optional[List[str]] = None,
                **kwargs) -> Dict[str, ModelResult]:
        """
        Fit all requested estimators (default: all registered).

        Returns dict {key: ModelResult} for successfully fitted models.
        """
        keys = keys or list(ESTIMATOR_REGISTRY.keys())
        for k in keys:
            try:
                r = self._dispatch(k, **kwargs)
                if r is not None:
                    self._results[k] = r
            except Exception as exc:
                log.warning(f"ModelZoo.fit_all: '{k}' raised {exc}")
        return dict(self._results)

    def select_best(
                    self,
                    results: Optional[Dict[str, ModelResult]] = None,
                    criterion: str = "bic",
                    exclude_non_converged: bool = True,
                    non_converged_penalty: float = 1e6,
                    ) -> Optional[ModelResult]:
        """
        Return the ModelResult with the lowest criterion value.

        criterion : "bic" | "aic" | "neg_loglik"
        exclude_non_converged : if True (default), non-converged fits are skipped
            entirely; if False they are penalised by *non_converged_penalty* instead.
        non_converged_penalty : additive BIC/AIC penalty applied when a fit has
            converged=False and exclude_non_converged=False.
        """
        pool = results if results is not None else self._results
        if not pool:
            return None

        _NON_CONVERGED = {k for k, v in pool.items() if not v.converged}
        if _NON_CONVERGED:
            log.warning(
                "select_best: %d non-converged fit(s) in pool: %s",
                len(_NON_CONVERGED), sorted(_NON_CONVERGED),
            )

        def _score(key: str, r: ModelResult) -> float:
            if criterion == "bic":        base = r.bic
            elif criterion == "aic":      base = r.aic
            elif criterion == "neg_loglik": base = -r.loglik
            else:
                raise ValueError(f"Unknown criterion '{criterion}'")
            if not r.converged:
                if exclude_non_converged:
                    return float("inf")
                base += non_converged_penalty
            return base

        valid = {k: v for k, v in pool.items()
                 if np.isfinite(_score(k, v)) and _score(k, v) < float("inf")}
        if not valid:
            log.warning("select_best: no converged finite-criterion fits available")
            return None
        best_key = min(valid, key=lambda k: _score(k, valid[k]))
        return valid[best_key]

    def comparison_table(self,
                         results: Optional[Dict[str, ModelResult]] = None
                         ) -> pd.DataFrame:
        """Return a comparison DataFrame sorted by BIC."""
        pool = results if results is not None else self._results
        rows = []
        for key, r in pool.items():
            rows.append({
                "estimator": key,
                "backend":   r.backend,
                "loglik":    round(r.loglik, 2),
                "aic":       round(r.aic, 2),
                "bic":       round(r.bic, 2),
                "n_params":  r.n_params,
                "n_obs":     r.n_obs,
                "converged": r.converged,
                "note":      "" if r.converged else "[non-converged]",
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Re-export BestSpecStore from main_2 if available
# ─────────────────────────────────────────────────────────────────────────────
try:
    from main_2 import BestSpecStore  # noqa: F401
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    # Minimal synthetic smoke test
    rng = np.random.default_rng(42)
    N   = 200
    df  = pd.DataFrame({
        "age":        rng.normal(35, 10, N),
        "dist":       rng.exponential(5, N),
        "income":     rng.normal(60, 15, N),
        "choice":     rng.choice(["early","am_peak","midday","pm_peak","evening","late"], N),
        "count_y":    rng.negative_binomial(3, 0.5, N),
        "duration":   rng.exponential(20, N),
        "event":      rng.binomial(1, 0.7, N),
        "binary_y":   rng.binomial(1, 0.4, N),
    })
    # Numeric choice for ordered/probit
    df["choice_id"] = df["choice"].map(_PERIOD_MAP)

    feat = ["age", "dist", "income"]

    zoo = ModelZoo(
        timing_df=df,
        feat_cols=feat,
        choice_col="choice",
        y_col="binary_y",
        duration_col="duration",
        event_col="event",
        period_map=_PERIOD_MAP,
    )

    keys = ["larch_mnl", "larch_nl", "aft_lognormal", "aft_weibull", "cox_ph",
            "probit"]
    results = zoo.fit_all(keys)
    tbl = zoo.comparison_table()
    print("\n── Model Zoo Comparison ──")
    print(tbl.to_string(index=False))
    best = zoo.select_best()
    if best:
        print(f"\nBest model: {best.backend}  BIC={best.bic:.2f}")
