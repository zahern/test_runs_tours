"""
main_2.py — Core-Model Validation Pipeline  (Stages 0–4, HHTS-only)
======================================================================
Objectives
----------
1.  Estimate stages 0–4 using *core* (non-searched) models drawn directly
    from SearchLibrium, MetaCountRegressor and Larch rather than the custom
    numpy/scipy estimators in stage1–4_*.py.
2.  Run on CPUs only — no GPU, no JAX XLA device conflicts.
3.  Validate every model back against the **full HHTS** (in-sample).
    The synthetic population is NOT loaded, and no predictions are pushed
    to population records.
4.  Write a self-contained HTML + CSV validation report.

Package sources (local editable installs)
------------------------------------------
    SearchLibrium       : C:/Users/ahernz/source/SearchLibrium
    MetaCountRegressor  : C:/Users/ahernz/source/MetaCount/metacountregressor
    Larch               : pip install larch5   (larch package)

CPU enforcement
---------------
JAX_PLATFORM_NAME / JAX_PLATFORMS must be set *before* any jax import.
We set them here unconditionally — even if a GPU node is available the
intent of this script is deterministic CPU validation.

Larch backend: tries 'numba' first (pure CPU, no JAX/XLA), then 'jax'
(XLA on cpu device).  'numba' is strongly preferred on Windows to avoid
hwloc / cusolver_gesvdj_ffi errors.

Stage → model mapping
----------------------
Stage 0 :  Data loading (Synth_Connect / HHTS only)
Stage 1 :  Activity frequency
              Primary   → MetaCountRegressor  NB2 (negative-binomial)
              Secondary → SearchLibrium       OrderedLogit
Stage 2 :  Day duration
              start_of_day / end_of_day  → MetaCountRegressor Tobit
              budget B                   → MetaCountRegressor Lognormal
Stage 3 :  Activity duration
              per purpose                → MetaCountRegressor mixed lognormal
Stage 4 :  Schedule placement (time-of-day choice)
              Primary   → Larch          MNL (numba engine; IDCO format)
              Secondary → Larch          Nested Logit (temporal nests)
              Tertiary  → Larch          Mixed Logit (numba; random time params)
              Cross-ck  → SearchLibrium  MNL / NestedLogit (BFGS)
              Cross-ck  → pylogit        MNL

GA bridge (LarchSABridge)
--------------------------
Larch log-likelihood is a direct drop-in for the fitness function in both
SearchLibrium SA/HS and MetaCountRegressor SA.  LarchSABridge wraps a Larch
MNL/NL fit and exposes a .fitness(variable_mask) → BIC interface compatible
with the call_siman / StructureEvaluatorLC APIs.  This allows variable
selection (which time-of-day covariates to include) to be driven by
evolutionary search while still using Larch's exact gradients for coefficient
estimation.

Additional integratable alternatives (see FRAMEWORK_PLAN.md):
    lifelines   (AFT / Weibull / Gamma / Cox)  → Stage 3
    statsmodels (Probit + IMR / OLS)            → Stage 2 joint Heckman
    pylogit     (MNL / Nested / Mixed)          → Stage 4 cross-validation
    biogeme     (MNL / Cross-nested / Mixed)    → Stage 4 cross-validation
    PyMC        (Bayesian hierarchical)         → any stage for posteriors
"""

# ── 0. CPU enforcement — MUST precede any jax / SearchLibrium import ─────────
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"]     = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("PIPELINE_ACCELERATOR", "cpu")

# Force non-interactive matplotlib BEFORE any plot import
import matplotlib
matplotlib.use("Agg")

# ── Standard library ─────────────────────────────────────────────────────────
import sys
import json
import traceback
import inspect
import time
import pickle
import logging
import pathlib
import datetime
import warnings
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Pipeline utilities ────────────────────────────────────────────────────────
from data_connect            import Synth_Connect
from stage1_activity_frequency import (
    prepare_frequency_data, add_total_activity_counts,
    assign_ptype_column, build_activity_counts,
    PURPOSES,                  # list of 8 purpose strings
)
from stage2_day_duration     import (
    aggregate_person_day, _augment_day_features, _build_X as _s2_build_X,
    MIN_START, MAX_END,
)
from stage3_activity_duration import (
    prepare_duration_data, _augment_duration_features,
)
from stage4_schedule_placement import (
    prepare_timing_data, TIME_PERIODS, _period_of,
)

# ── SearchLibrium (JAX path where available, else CPU BFGS) ─────────────────
# SL models accept _jax=True to use JAX autodiff for gradient computation.
# On the HPC this is the preferred path.  Set SL_JAX=0 to disable.
_SL_USE_JAX: bool = os.environ.get("SL_JAX", "1") == "1"

try:
    from SearchLibrium import MultinomialLogit as SL_MNL
    from SearchLibrium import OrderedLogit     as SL_ORL
    try:
        from SearchLibrium import NestedLogit  as SL_NL
    except ImportError:
        SL_NL = None
    _SL_AVAILABLE = True
except ImportError as _e:
    _SL_AVAILABLE = False
    warnings.warn(f"SearchLibrium not importable: {_e}. Stage 1 ORL and Stage 4 MNL will be skipped.")
    SL_MNL = SL_ORL = SL_NL = None

# ── MetaCountRegressor (JAX-cpu path) ────────────────────────────────────────
try:
    from metacountregressor import ExperimentBuilder, ModelConstraints
    _MCR_AVAILABLE = True
except ImportError as _e:
    _MCR_AVAILABLE = False
    warnings.warn(f"metacountregressor not importable: {_e}. NB/Tobit/Lognormal fits will be skipped.")
    ExperimentBuilder = ModelConstraints = None

# ── Larch (JAX engine preferred for exact autodiff gradients) ───────────────
# Larch is the primary estimator for discrete choice in stages 5+6 of the
# existing pipeline.  Here we use it for Stage 4 time-of-day choice.
#
# Engine priority: jax > numba
#   compute_engine='jax'   → XLA on cpu device — exact gradients via autodiff;
#                            required for MXL / LatentClass; default on HPC.
#   compute_engine='numba' → pure numba JIT, no JAX/XLA; set LARCH_ENGINE=numba
#                            on Windows if you hit hwloc/cusolver_gesvdj_ffi errors.
#
# Override with env var: LARCH_ENGINE=numba python main_2.py
_LARCH_ENGINE: str = "jax"    # default: JAX autodiff on cpu device
_LARCH_ENGINE = os.environ.get("LARCH_ENGINE", _LARCH_ENGINE).lower()
_LARCH_ENGINE_FALLBACK: str = "numba"   # used if JAX unavailable

try:
    import larch as lx
    from larch import P, X as LX
    import xarray as xr
    _LARCH_AVAILABLE = True
except ImportError as _e:
    lx = None
    P = LX = xr = None
    _LARCH_AVAILABLE = False
    warnings.warn(f"larch not importable: {_e}. Larch MNL/NL/MXL will be skipped.")

# ── SearchLibrium SA search (for GA bridge) ───────────────────────────────────
try:
    from SearchLibrium.call_meta import call_siman as _SL_SIMAN
    from SearchLibrium.siman    import SimulatedAnnealing as _SL_SA
    _SL_SEARCH_AVAILABLE = True
except ImportError:
    _SL_SIMAN = _SL_SA = None
    _SL_SEARCH_AVAILABLE = False

# ── Optional extra packages (graceful degradation) ───────────────────────────
try:
    import lifelines                       # AFT / Weibull / Cox for Stage 3
    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False

try:
    import statsmodels.api as sm           # Probit/Heckman / OLS for Stage 2
    _SM_AVAILABLE = True
except ImportError:
    _SM_AVAILABLE = False

try:
    import pylogit                         # alternative MNL for Stage 4
    _PYLOGIT_AVAILABLE = True
except ImportError:
    _PYLOGIT_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PERSON_FEATURES = ["sexp", "vehrd", "nprd", "STUDYING", "WORKSTAT", "incp"]

PTYPES = [
    "FullTimeWorker", "PartTimeWorker", "WorkingStudent",
    "UniversityStudent", "SchoolChild", "Retired", "NonWorker",
]

# Temporal nests for nested logit (Stage 4)
TEMPORAL_NESTS = {
    "off_peak": ["early", "late"],
    "peak":     ["am_peak", "pm_peak"],
    "shoulder": ["midday", "evening"],
}

# Minimum observations per (purpose, ptype) cell for individual model fitting
MIN_CELL_N = 30

# Number of Halton draws for mixed / random-parameter fits
N_DRAWS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup(run_tag: str = "main2") -> pathlib.Path:
    """Create output directory and configure logging. Returns out_dir."""
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("runs") / f"{run_tag}__{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "main2.log", mode="w"),
        ],
    )
    # Write manifest
    manifest = {
        "mode": run_tag,
        "out_dir": str(out_dir),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "cpu_only": True,
        "population_prediction": False,
        "searchlibrium_available": _SL_AVAILABLE,
        "metacountregressor_available": _MCR_AVAILABLE,
        "lifelines_available": _LIFELINES_AVAILABLE,
        "statsmodels_available": _SM_AVAILABLE,
        "pylogit_available": _PYLOGIT_AVAILABLE,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


log = logging.getLogger("main2")


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

_TIMING: List[dict] = []


class _Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._t0 = time.perf_counter()
        log.info(f"▶  {self.label}")
        return self

    def __exit__(self, et, ev, tb):
        elapsed = time.perf_counter() - self._t0
        status  = "ok" if et is None else "FAILED"
        log.info(f"{'✓' if et is None else '✗'}  {self.label}  [{elapsed:.1f}s] — {status}")
        _TIMING.append({"step": self.label, "elapsed_s": round(elapsed, 2), "status": status})
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — data loading (HHTS only, no synth population)
# ─────────────────────────────────────────────────────────────────────────────

def stage0_load_data() -> dict:
    """
    Load and process the HHTS survey data.

    Returns the standard ``out`` dict:
        out["tours"]   — tour-level records (activity chains)
        out["trips"]   — trip-level records
        out["persons"] — person-level records

    The synthetic population is deliberately NOT loaded here.
    All downstream validation operates exclusively on HHTS observations.
    """
    with _Timer("Stage 0 — HHTS data loading"):
        sc  = Synth_Connect()
        out = sc.process_survey(save=False)
        # Ensure full-day activity counts are present
        out["tours"] = add_total_activity_counts(out["tours"])
        out["tours"] = assign_ptype_column(out["tours"])
        n_tours   = len(out["tours"])
        n_persons = len(out["persons"])
        log.info(f"  HHTS loaded: {n_persons:,} persons, {n_tours:,} tours")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Activity frequency
# ─────────────────────────────────────────────────────────────────────────────

def _mcr_nb2_fit(df: pd.DataFrame, y_col: str, feat_cols: List[str],
                 person_id_col: str = "person_id") -> Optional[dict]:
    """
    Fit a Negative-Binomial 2 (NB2) model using MetaCountRegressor.

    NB2 is the canonical model for non-negative integer counts with
    overdispersion (variance > mean), which is typical of daily activity
    frequency data.

    Parameters
    ----------
    df          : DataFrame with one row per HHTS person
    y_col       : count outcome column
    feat_cols   : predictor variable names
    person_id_col: unique individual ID column

    Returns dict with keys: loglik, aic, bic, summary, backend="MCR-NB2"
    """
    if not _MCR_AVAILABLE:
        return None
    df_fit = df[[person_id_col] + feat_cols + [y_col]].dropna()
    if len(df_fit) < MIN_CELL_N:
        return None
    try:
        builder = ExperimentBuilder(
            df_fit,
            id_col=person_id_col,
            y_col=y_col,
            default_engine="jax",   # JAX forced to cpu via env var above
        )
        spec = builder.make_manual_spec(
            fixed_terms=feat_cols,
            rdm_terms=[],
            rdm_cor_terms=[],
            grouped_terms=[],
            hetro_in_means=[],
            zi_terms=[],
            membership_terms=[],
            dispersion=1,          # 1 → NB2 (overdispersed count)
            latent_classes=1,
        )
        fit = builder.fit_manual_model(spec, model="nb", R=N_DRAWS, print_report=False)
        fit["backend"] = "MCR-NB2"
        return fit
    except Exception as exc:
        log.warning(f"  MCR NB2 fit failed for {y_col}: {exc}\n{traceback.format_exc()}")
        return None


def _mcr_zip_fit(df: pd.DataFrame, y_col: str, feat_cols: List[str],
                 person_id_col: str = "person_id") -> Optional[dict]:
    """
    Fit a Zero-Inflated Poisson (ZIP) model using MetaCountRegressor.

    Appropriate when many persons generate zero activities for a given
    purpose (e.g. 'work' for non-workers, 'edu' for adults).
    The ZI component uses all features as zero-inflation regressors.
    """
    if not _MCR_AVAILABLE:
        return None
    df_fit = df[[person_id_col] + feat_cols + [y_col]].dropna()
    if len(df_fit) < MIN_CELL_N:
        return None
    zero_rate = (df_fit[y_col] == 0).mean()
    if zero_rate < 0.05:
        return None  # not enough zeros to warrant ZI treatment
    try:
        builder = ExperimentBuilder(df_fit, id_col=person_id_col, y_col=y_col,
                                    default_engine="jax")
        spec = builder.make_manual_spec(
            fixed_terms=feat_cols,
            rdm_terms=[],
            rdm_cor_terms=[],
            grouped_terms=[],
            hetro_in_means=[],
            zi_terms=feat_cols[:2],   # first two features enter ZI equation
            membership_terms=[],
            dispersion=0,             # 0 → Poisson (ZIP uses Poisson base)
            latent_classes=1,
        )
        fit = builder.fit_manual_model(spec, model="poisson", R=N_DRAWS, print_report=False)
        fit["backend"] = "MCR-ZIP"
        return fit
    except Exception as exc:
        log.warning(f"  MCR ZIP fit failed for {y_col}: {exc}\n{traceback.format_exc()}")
        return None


def _sl_ordered_logit_fit(X: np.ndarray, y: np.ndarray,
                          feat_names: List[str]) -> Optional[dict]:
    """
    Fit a proportional-odds Ordered Logit using SearchLibrium.

    The ordered logit model is appropriate for ordinal count outcomes
    (0, 1, 2, …, K) because it respects the ordering without assuming
    equal intervals.  SearchLibrium's implementation estimates K−1
    threshold parameters jointly with the utility coefficients.

    Parameters
    ----------
    X          : (N, K) feature matrix (already normalised / encoded)
    y          : (N,)   ordinal integer outcomes  {0, 1, …, J-1}
    feat_names : variable names matching columns of X

    Returns dict with loglik, aic, bic, params, stderr, backend="SL-ORL"
    """
    if not _SL_AVAILABLE or SL_ORL is None:
        return None
    if len(X) < MIN_CELL_N:
        return None
    try:
        J = int(y.max()) + 1
        X_df = pd.DataFrame(X, columns=feat_names)
        model = SL_ORL(
            _jax=_SL_USE_JAX,
            X=X_df,
            y=y.astype(int),
            J=J,
            varnames=feat_names,
            fit_intercept=False,
            normalize=True,
        )
        model.fit()
        result = {
            "backend":  "SL-ORL",
            "loglik":   getattr(model, "loglik", np.nan),
            "aic":      getattr(model, "aic",    np.nan),
            "bic":      getattr(model, "bic",    np.nan),
            "params":   getattr(model, "params", None),
            "stderr":   getattr(model, "stderr", None),
            "labels":   getattr(model, "labels", feat_names),
            "J":        J,
            "model_obj": model,
        }
        return result
    except Exception as exc:
        log.warning(f"  SL OrderedLogit fit failed: {exc}\n{traceback.format_exc()}")
        return None


def stage1_activity_frequency(out: dict, out_dir: pathlib.Path) -> dict:
    """
    Stage 1 — activity frequency estimation on HHTS.

    For each (purpose, ptype) cell:
        1. Fit MetaCountRegressor NB2       (primary)
        2. Fit MetaCountRegressor ZIP        (if high zero-rate)
        3. Fit SearchLibrium OrderedLogit    (secondary / cross-check)

    Returns nested dict:
        models[purpose][ptype] = {
            "nb2":  fit dict or None,
            "zip":  fit dict or None,
            "orl":  fit dict or None,
            "best": "nb2" | "zip" | "orl",   # by BIC
            "n":    int,
        }
    """
    with _Timer("Stage 1 — activity frequency"):
        tours   = out["tours"]
        persons = out["persons"]

        # Build person-level feature + count table (same logic as stage1)
        freq_df  = prepare_frequency_data(tours, persons)
        count_df = build_activity_counts(tours)

        # Merge counts onto person features.
        #
        # BUG (fixed): prepare_frequency_data() already calls build_activity_counts()
        # internally, so freq_df ALREADY carries every n_*_out column. Merging
        # count_df on top made all ten n_* columns collide, and pandas renamed
        # each to _x/_y -- so `n_work_out not in merged.columns` was true for
        # every purpose and stage 1 skipped all 8 while still logging "ok".
        # Only bring across columns freq_df does not already have, and coalesce
        # any that still collide, so the unsuffixed names survive.
        if "person_id" not in freq_df.columns:
            freq_df = freq_df.reset_index().rename(columns={"index": "person_id"})
        if "person_id" not in count_df.columns:
            count_df = count_df.reset_index().rename(columns={"index": "person_id"})

        _extra = [c for c in count_df.columns
                  if c != "person_id" and c not in freq_df.columns]
        if _extra:
            merged = freq_df.merge(count_df[["person_id"] + _extra],
                                   on="person_id", how="left")
        else:
            merged = freq_df
        # Safety net: if any count column still got suffixed, coalesce it back.
        for _c in [c for c in merged.columns
                   if c.endswith(("_x", "_y")) and c[:-2] not in merged.columns]:
            _base = _c[:-2]
            merged[_base] = merged[_c].combine_first(
                merged.get(f"{_base}_x" if _c.endswith("_y") else f"{_base}_y"))
        _missing_counts = [f"n_{p}_out" for p in PURPOSES
                           if f"n_{p}_out" not in merged.columns]
        if _missing_counts:
            raise KeyError(
                f"stage1: count columns missing after the person/count merge: "
                f"{_missing_counts}. Refusing to report 'ok' with zero fitted "
                f"purposes -- this silently produced 0-cell stage-1 runs.")

        feat_cols = [c for c in PERSON_FEATURES if c in merged.columns]

        models: dict = {}
        validation_rows: list = []

        for purpose in PURPOSES:
            y_col = f"n_{purpose}_out"
            if y_col not in merged.columns:
                log.warning(f"  Stage 1: column {y_col} not found — skipping {purpose}")
                continue
            models[purpose] = {}

            for ptype in PTYPES:
                mask    = merged.get("ptype", pd.Series(dtype=str)) == ptype
                cell_df = merged[mask].copy() if mask.any() else merged.copy()
                cell_df = cell_df.dropna(subset=feat_cols + [y_col])
                n_cell  = len(cell_df)

                # Encode features
                X_raw = cell_df[feat_cols].values.astype(float)
                y_raw = cell_df[y_col].clip(lower=0).values.astype(int)

                # ── Fit models ──────────────────────────────────────────────
                if "person_id" not in cell_df.columns:
                    cell_df = cell_df.copy()
                    cell_df["person_id"] = np.arange(len(cell_df))

                nb2_fit  = _mcr_nb2_fit(cell_df, y_col, feat_cols)
                zip_fit  = _mcr_zip_fit(cell_df, y_col, feat_cols)
                orl_fit  = _sl_ordered_logit_fit(X_raw, y_raw, feat_cols)

                # Choose best by BIC (lower is better)
                candidates = {k: v for k, v in
                              [("nb2", nb2_fit), ("zip", zip_fit), ("orl", orl_fit)]
                              if v is not None}
                if candidates:
                    best_key = min(
                        candidates,
                        key=lambda k: candidates[k].get("bic", np.inf),
                    )
                else:
                    best_key = None

                models[purpose][ptype] = {
                    "nb2":  nb2_fit,
                    "zip":  zip_fit,
                    "orl":  orl_fit,
                    "best": best_key,
                    "n":    n_cell,
                }

                # ── Validation (in-sample HHTS) ─────────────────────────────
                for model_key, fit in candidates.items():
                    bic = fit.get("bic", np.nan)
                    aic = fit.get("aic", np.nan)
                    ll  = fit.get("loglik", np.nan)
                    validation_rows.append({
                        "stage": 1,
                        "purpose": purpose,
                        "ptype":   ptype,
                        "model":   model_key,
                        "n":       n_cell,
                        "loglik":  ll,
                        "aic":     aic,
                        "bic":     bic,
                        "best":    model_key == best_key,
                    })

        # Save
        with open(out_dir / "stage1_core_models.pkl", "wb") as fh:
            pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)
        val_df = pd.DataFrame(validation_rows)
        val_df.to_csv(out_dir / "stage1_validation.csv", index=False)
        log.info(f"  Stage 1: fitted {len(validation_rows)} (purpose×ptype×model) cells")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Day duration budget
# ─────────────────────────────────────────────────────────────────────────────

def _mcr_tobit_fit(df: pd.DataFrame, y_col: str, feat_cols: List[str],
                   censoring: str = "left",
                   person_id_col: str = "person_id") -> Optional[dict]:
    """
    Fit a Tobit I censored regression using MetaCountRegressor.

    Tobit handles outcomes that are bounded (left-censored at 0 or some
    minimum; right-censored at a maximum), which is exactly the case for:
        start_of_day : left-censored  at MIN_START (4am)
        end_of_day   : right-censored at MAX_END   (midnight)

    Parameters
    ----------
    censoring : "left" or "right"
    """
    if not _MCR_AVAILABLE:
        return None
    df_fit = df[[person_id_col] + feat_cols + [y_col]].dropna()
    if len(df_fit) < MIN_CELL_N:
        return None
    try:
        builder = ExperimentBuilder(
            df_fit, id_col=person_id_col, y_col=y_col,
            default_engine="jax",
        )
        spec = builder.make_manual_spec(
            fixed_terms=feat_cols,
            rdm_terms=[],
            rdm_cor_terms=[],
            grouped_terms=[],
            hetro_in_means=[],
            zi_terms=[],
            membership_terms=[],
            dispersion=0,
            latent_classes=1,
        )
        fit = builder.fit_manual_model(spec, model="tobit", R=N_DRAWS, print_report=False)
        fit["backend"] = f"MCR-Tobit-{censoring}"
        return fit
    except Exception as exc:
        log.warning(f"  MCR Tobit fit failed for {y_col}: {exc}\n{traceback.format_exc()}")
        return None


def _mcr_lognormal_fit(df: pd.DataFrame, y_col: str, feat_cols: List[str],
                       rdm_terms: Optional[List[str]] = None,
                       person_id_col: str = "person_id") -> Optional[dict]:
    """
    Fit a (Mixed) Lognormal regression using MetaCountRegressor.

    Lognormal is appropriate for strictly-positive continuous durations
    and budget hours (Stage 2 budget B, Stage 3 activity durations) because:
        - support is (0, ∞)
        - right-skewed empirical distributions
        - log-transformation yields near-Gaussian residuals

    Setting rdm_terms adds random (individual heterogeneity) parameters for
    those variables, producing a *Mixed* Lognormal model.
    """
    if not _MCR_AVAILABLE:
        return None
    df_fit = df[[person_id_col] + feat_cols + [y_col]].dropna()
    df_fit = df_fit[df_fit[y_col] > 0]
    if len(df_fit) < MIN_CELL_N:
        return None
    rdm_terms = rdm_terms or []
    rdm_str   = [f"{v}:normal" for v in rdm_terms if v in feat_cols]
    # Lognormal needs an explicit right-censoring indicator.
    #
    # MCR's ALL_SURVIVAL_FAMILIES includes "lognormal", so fit_manual_model()
    # routes model="lognormal" down the survival branch and hard-requires
    # event_col:
    #     ValueError: survival/hazard refit needs an event_col on the builder
    # that branch never ran here, so every lognormal fit (Stage 2 budget B and
    # Stage 3 durations) returned None -- 10 silent failures per run.
    #
    # Activity durations and budget hours are FULLY OBSERVED: there is no
    # right-censoring, so every row is an event. MCR's own docstring says a
    # None event_col means data["event"] is all ones, which is the semantics we
    # want -- it just refuses to infer that for a survival-family name. Supply
    # the all-ones column explicitly.
    df_fit = df_fit.copy()
    df_fit["_event"] = np.ones(len(df_fit), dtype=np.int8)
    try:
        builder = ExperimentBuilder(
            df_fit, id_col=person_id_col, y_col=y_col,
            event_col="_event",
            default_engine="jax",
        )
        spec = builder.make_manual_spec(
            fixed_terms=[c for c in feat_cols if c not in rdm_terms],
            rdm_terms=rdm_str,
            rdm_cor_terms=[],
            grouped_terms=[],
            hetro_in_means=[],
            zi_terms=[],
            membership_terms=[],
            dispersion=0,
            latent_classes=1,
        )
        fit = builder.fit_manual_model(spec, model="lognormal", R=N_DRAWS, print_report=False)
        fit["backend"] = "MCR-Lognormal" + ("-Mixed" if rdm_terms else "")
        return fit
    except Exception as exc:
        log.warning(f"  MCR Lognormal fit failed for {y_col}: {exc}\n{traceback.format_exc()}")
        return None


def _sm_heckman_fit(df: pd.DataFrame,
                    participation_col: str,
                    budget_col: str,
                    sel_cols: List[str],
                    out_cols: List[str]) -> Optional[dict]:
    """
    Fit a two-step Heckman (Heckit) selection model using statsmodels.

    Jointly models:
        Step 1 — Probit:  P(participated | sel_cols)   (all persons)
        Step 2 — OLS+IMR: E(log B | participated, out_cols, IMR)

    The Inverse Mills Ratio (IMR) corrects for selection bias; persons who
    do not travel are excluded from the budget regression but their
    participation probability feeds the IMR term.

    Required exclusion restriction: sel_cols ⊃ out_cols (extra instruments).
    """
    if not _SM_AVAILABLE:
        return None
    df_all   = df.dropna(subset=sel_cols + [participation_col])
    df_trav  = df_all[df_all[participation_col] == 1].dropna(subset=out_cols + [budget_col])
    if len(df_all) < MIN_CELL_N or len(df_trav) < MIN_CELL_N // 2:
        return None
    try:
        # Step 1 — probit participation
        X_sel  = sm.add_constant(df_all[sel_cols].values.astype(float))
        y_part = df_all[participation_col].values.astype(int)
        probit = sm.Probit(y_part, X_sel).fit(disp=False, maxiter=200)

        # Compute IMR on traveller subsample
        Xp_trav  = sm.add_constant(df_trav[sel_cols].values.astype(float))
        xb_trav  = Xp_trav @ probit.params
        phi       = sp_stats.norm.pdf(xb_trav)
        Phi       = sp_stats.norm.cdf(xb_trav)
        imr       = phi / np.clip(Phi, 1e-10, None)

        # Step 2 — OLS with IMR
        log_B     = np.log(np.clip(df_trav[budget_col].values, 1e-3, None))
        X_out     = np.column_stack([
            sm.add_constant(df_trav[out_cols].values.astype(float)), imr
        ])
        ols       = sm.OLS(log_B, X_out).fit()
        return {
            "backend":   "SM-Heckman",
            "probit":    probit,
            "ols":       ols,
            "aic":       ols.aic,
            "bic":       ols.bic,
            "loglik":    float(ols.llf),
            "r2":        float(ols.rsquared),
        }
    except Exception as exc:
        log.warning(f"  Heckman fit failed: {exc}\n{traceback.format_exc()}")
        return None


def stage2_day_duration(out: dict, out_dir: pathlib.Path) -> dict:
    """
    Stage 2 — day duration estimation on HHTS.

    Models estimated:
        start_of_day  : Tobit-left  (MCR)
        end_of_day    : Tobit-right (MCR)
        budget B      : Lognormal   (MCR)  — primary
        budget B      : Mixed Lognormal (MCR) with random intercept
        budget B      : Heckman (SM)         — if statsmodels available

    Validation: RMSE and KDE agreement between predicted and observed HHTS
    distributions for all three outcomes.
    """
    with _Timer("Stage 2 — day duration"):
        tours   = out["tours"]
        persons = out["persons"]

        person_day = aggregate_person_day(tours, persons)
        person_day = _augment_day_features(person_day)

        if "person_id" not in person_day.columns:
            person_day = person_day.reset_index().rename(columns={"index": "person_id"})

        base_feats = ["n_mandatory", "n_discretionary", "mandatory_share",
                      "escort_share", "vehs_per_person", "income_per_person"]
        feat_cols  = [c for c in base_feats if c in person_day.columns]
        # Add ptype dummies if present
        ptype_dummies = [c for c in person_day.columns if c.startswith("ptype_")]
        feat_cols += ptype_dummies

        models: dict = {}
        val_rows: list = []

        for outcome, censoring, rdm in [
            ("start_of_day", "left",  []),
            ("end_of_day",   "right", []),
            ("B",            "none",  ["income_per_person"]),
        ]:
            if outcome not in person_day.columns:
                log.warning(f"  Stage 2: column '{outcome}' missing — skipping")
                continue

            tobit_fit  = None
            logn_fit   = None
            mlogn_fit  = None
            heck_fit   = None

            if censoring in ("left", "right"):
                tobit_fit = _mcr_tobit_fit(person_day, outcome, feat_cols,
                                           censoring=censoring)
            else:
                # Budget B: lognormal variants
                logn_fit  = _mcr_lognormal_fit(person_day, outcome, feat_cols)
                mlogn_fit = _mcr_lognormal_fit(person_day, outcome, feat_cols,
                                               rdm_terms=rdm)
                # Heckman (joint model) for B
                participation_col = "participated"
                if participation_col not in person_day.columns:
                    person_day["participated"] = (person_day["B"] > 0).astype(int)
                excl = ["n_mandatory", "n_discretionary"]
                sel_c = [c for c in feat_cols + excl if c in person_day.columns]
                heck_fit = _sm_heckman_fit(
                    person_day,
                    participation_col=participation_col,
                    budget_col="B",
                    sel_cols=sel_c,
                    out_cols=feat_cols,
                )

            candidates = {k: v for k, v in [
                ("tobit", tobit_fit), ("lognormal", logn_fit),
                ("mixed_lognormal", mlogn_fit), ("heckman", heck_fit),
            ] if v is not None}

            best_key = (min(candidates, key=lambda k: candidates[k].get("bic", np.inf))
                        if candidates else None)

            models[outcome] = {
                **{k: v for k, v in candidates.items()},
                "best": best_key,
                "n":    len(person_day.dropna(subset=[outcome])),
            }

            for k, fit in candidates.items():
                val_rows.append({
                    "stage":   2,
                    "outcome": outcome,
                    "model":   k,
                    "n":       models[outcome]["n"],
                    "loglik":  fit.get("loglik", np.nan),
                    "aic":     fit.get("aic",    np.nan),
                    "bic":     fit.get("bic",    np.nan),
                    "best":    k == best_key,
                })

        with open(out_dir / "stage2_core_models.pkl", "wb") as fh:
            pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame(val_rows).to_csv(out_dir / "stage2_validation.csv", index=False)
        log.info(f"  Stage 2: fitted {len(val_rows)} model variants")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Activity duration
# ─────────────────────────────────────────────────────────────────────────────

def _lifelines_aft_fit(df: pd.DataFrame, duration_col: str,
                       feat_cols: List[str],
                       family: str = "lognormal") -> Optional[dict]:
    """
    Fit an Accelerated Failure Time (AFT) duration model using lifelines.

    Available parametric families:
        lognormal  → LogNormalAFTFitter   (default; most common for activity duration)
        weibull    → WeibullAFTFitter     (hazard monotone increasing/decreasing)
        loglogistic → LogLogisticAFTFitter (non-monotone hazard; unimodal)

    The event indicator is set to 1 for all records (no censoring in HHTS).
    Budget rescaling is applied post-prediction (not here).
    """
    if not _LIFELINES_AVAILABLE:
        return None
    df_fit = df[feat_cols + [duration_col]].dropna()
    df_fit = df_fit[df_fit[duration_col] > 0]
    if len(df_fit) < MIN_CELL_N:
        return None
    try:
        df_fit = df_fit.copy()
        df_fit["_event"] = 1
        family_map = {
            "lognormal":   lifelines.LogNormalAFTFitter,
            "weibull":     lifelines.WeibullAFTFitter,
            "loglogistic": lifelines.LogLogisticAFTFitter,
        }
        Fitter = family_map.get(family, lifelines.LogNormalAFTFitter)
        m = Fitter()
        m.fit(df_fit, duration_col=duration_col, event_col="_event",
              ancillary=True)
        return {
            "backend": f"LL-AFT-{family}",
            "model_obj": m,
            "loglik": float(m.log_likelihood_),
            "aic":    float(m.AIC_),
            "bic":    float(m.BIC_),
        }
    except Exception as exc:
        log.warning(f"  lifelines AFT {family} failed for {duration_col}: {exc}\n{traceback.format_exc()}")
        return None


def stage3_activity_duration(out: dict, out_dir: pathlib.Path) -> dict:
    """
    Stage 3 — activity duration estimation on HHTS.

    For each activity purpose:
        1. MCR Mixed Lognormal         (primary)
        2. lifelines AFT Lognormal     (cross-check / robustness)
        3. lifelines AFT Weibull       (hazard-based alternative)
        4. lifelines AFT LogLogistic   (non-monotone hazard)

    Validation: bias%, scatter RMSE against HHTS observed durations.
    """
    with _Timer("Stage 3 — activity duration"):
        tours   = out["tours"]
        persons = out["persons"]
        trips   = out.get("trips")

        dur_df = prepare_duration_data(tours, persons, trips)
        dur_df = _augment_duration_features(dur_df)
        if "person_id" not in dur_df.columns:
            dur_df = dur_df.reset_index().rename(columns={"index": "person_id"})

        base_feats = ["start_hr", "log_dist", "B", "budget_per_activity_hr",
                      "day_span_hr", "n_mandatory", "n_discretionary"]
        feat_cols  = [c for c in base_feats if c in dur_df.columns]
        feat_cols += [c for c in dur_df.columns if c.startswith("ptype_")]

        models: dict = {}
        val_rows: list = []

        # prepare_duration_data() emits the duration column as `duration`
        # (minutes at the activity destination). Older revisions of this stage
        # expected `duration_min`, and the mismatch was a hard KeyError at the
        # dropna below -- stage 3 could not run at all. Resolve the name once
        # and accept either contract so the two modules stop drifting.
        _y_src = "duration_min" if "duration_min" in dur_df.columns else (
            "duration" if "duration" in dur_df.columns else None)
        if _y_src is None:
            raise KeyError(
                "stage3: no duration column in the estimation frame -- expected "
                f"'duration' or 'duration_min'; got {list(dur_df.columns)[:20]}"
            )
        if _y_src != "duration_min":
            dur_df = dur_df.rename(columns={_y_src: "duration_min"})
        y_col = "duration_min"

        for purpose in PURPOSES:
            mask    = dur_df.get("purpose", pd.Series(dtype=str)) == purpose
            cell_df = dur_df[mask].copy() if mask.any() else dur_df.copy()
            cell_df = cell_df.dropna(subset=feat_cols + ["duration_min"])

            rdm   = ["B"] if "B" in feat_cols else []

            mlogn_fit = _mcr_lognormal_fit(cell_df, y_col, feat_cols, rdm_terms=rdm)
            aft_logn  = _lifelines_aft_fit(cell_df, y_col, feat_cols, "lognormal")
            aft_weib  = _lifelines_aft_fit(cell_df, y_col, feat_cols, "weibull")
            aft_llogi = _lifelines_aft_fit(cell_df, y_col, feat_cols, "loglogistic")

            candidates = {k: v for k, v in [
                ("mcr_mixed_lognormal", mlogn_fit),
                ("aft_lognormal",       aft_logn),
                ("aft_weibull",         aft_weib),
                ("aft_loglogistic",     aft_llogi),
            ] if v is not None}

            best_key = (min(candidates, key=lambda k: candidates[k].get("bic", np.inf))
                        if candidates else None)

            models[purpose] = {**candidates, "best": best_key, "n": len(cell_df)}

            for k, fit in candidates.items():
                val_rows.append({
                    "stage": 3, "purpose": purpose, "model": k,
                    "n": len(cell_df),
                    "loglik": fit.get("loglik", np.nan),
                    "aic":    fit.get("aic",    np.nan),
                    "bic":    fit.get("bic",    np.nan),
                    "best":   k == best_key,
                })

        with open(out_dir / "stage3_core_models.pkl", "wb") as fh:
            pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame(val_rows).to_csv(out_dir / "stage3_validation.csv", index=False)
        log.info(f"  Stage 3: fitted {len(val_rows)} (purpose×model) cells")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Schedule placement (time-of-day choice)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mnl_long_format(timing_df: pd.DataFrame,
                           feat_cols: List[str],
                           period_col: str = "time_period",
                           id_col: str = "obs_id") -> Tuple[np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
    """
    Transform wide-format HHTS timing data into the long format required by
    SearchLibrium MultinomialLogit.

    Long format: one row per (observation × alternative), with:
        X     — (N_obs × N_alts, K_vars)  feature matrix
        y     — (N_obs × N_alts,)         1=chosen, 0=not-chosen
        alts  — (N_obs × N_alts,)         alternative labels
        ids   — (N_obs × N_alts,)         choice-situation IDs

    Individual-specific variables (isvars) are identical across alternatives;
    alternative-specific constants (ASCs) are identified via fit_intercept=True.
    """
    # TIME_PERIODS is a list of (label, start_min, end_min) TUPLES, so taking it
    # wholesale yielded tuples as the alternative codes:
    #   * rows_alt became a (N*J, 3) array -> SearchLibrium's
    #     `[j for j in self.alts if j != self.base_alt]` raised
    #     "truth value of an array is ambiguous" (base_alt = alts[0] = a 3-tuple)
    #   * `1 if chosen == period` compared a label ('am_peak') to a tuple, so y
    #     was ALL ZEROS -- no alternative was ever marked chosen.
    # Use the label (p[0]); stage4_schedule_placement does exactly this with
    # PERIOD_LABELS = [p[0] for p in TIME_PERIODS], and the data's time_period
    # column holds labels ('am_peak', 'midday', ...).
    if hasattr(TIME_PERIODS, "keys"):
        periods = list(TIME_PERIODS.keys())
    else:
        periods = [p[0] if isinstance(p, (tuple, list)) else p
                   for p in TIME_PERIODS]
    if not isinstance(periods, list):
        periods = ["early", "am_peak", "midday", "pm_peak", "evening", "late"]

    avail_feats = [c for c in feat_cols if c in timing_df.columns]
    if id_col not in timing_df.columns:
        timing_df = timing_df.reset_index().rename(columns={"index": id_col})

    rows_X, rows_y, rows_alt, rows_id = [], [], [], []
    for _, row in timing_df.iterrows():
        chosen = row.get(period_col, None)
        obs_id = row[id_col]
        x_vals = row[avail_feats].values.astype(float)
        for period in periods:
            rows_X.append(x_vals)
            rows_y.append(1 if chosen == period else 0)
            rows_alt.append(period)
            rows_id.append(obs_id)

    X    = np.array(rows_X)
    y    = np.array(rows_y)
    alts = np.array(rows_alt)
    ids  = np.array(rows_id)
    return X, y, alts, ids


# ---------------------------------------------------------------------------
# Larch helper: engine factory (numba → jax fallback, per user memory note)
# ---------------------------------------------------------------------------

def _lx_engine(require_jax: bool = False) -> str:
    """Return the Larch compute engine to use.

    Default: JAX (XLA on cpu) for exact autodiff gradients.  Falls back to
    numba if JAX is unavailable (e.g. a Windows node with no XLA build).

    Parameters
    ----------
    require_jax : bool
        If True (used for MXL / LatentClass) and JAX is unavailable, the
        caller should skip the model rather than degrade to numba silently.

    On Windows without a JAX build, set LARCH_ENGINE=numba to force numba.
    On HPC Linux (default): JAX runs fine on cpu device.
    """
    if _LARCH_ENGINE == "jax":
        try:
            import jax  # noqa: F401 — just test importability
            return "jax"
        except Exception:
            if require_jax:
                raise RuntimeError("JAX not available — cannot run MXL/LatentClass")
            return _LARCH_ENGINE_FALLBACK
    return _LARCH_ENGINE  # "numba" was explicitly requested


def _lx_build_idco_timing(timing_df: pd.DataFrame,
                           feat_cols: List[str],
                           period_col: str = "time_period") -> Tuple:
    """
    Build Larch IDCO-format data for time-of-day choice.

    IDCO layout: one row per choice situation (observation).
    Alternatives (6 time periods) are declared via integer alt-ids.
    Person/activity covariates are identical across alternatives — they enter
    as utility_co[j] terms, allowing alternative-specific slopes on IS vars.

    Period → integer alt-id mapping
    ---------------------------------
        early=1, am_peak=2, midday=3, pm_peak=4, evening=5, late=6

    Returns (idco_df, period_map, avail_feats)
        idco_df    : DataFrame with index=obs_id, columns=feat_cols + ['chosen_id']
        period_map : dict mapping period label → alt-id integer
        avail_feats: list of feature columns actually present in timing_df
    """
    period_map = {
        "early": 1, "am_peak": 2, "midday": 3,
        "pm_peak": 4, "evening": 5, "late": 6,
    }
    avail_feats = [c for c in feat_cols if c in timing_df.columns]

    idco_df = timing_df[avail_feats + [period_col]].copy()
    idco_df = idco_df.dropna(subset=[period_col])
    idco_df["chosen_id"] = idco_df[period_col].map(period_map)
    idco_df = idco_df.dropna(subset=["chosen_id"])
    idco_df["chosen_id"] = idco_df["chosen_id"].astype(int)
    idco_df = idco_df.drop(columns=[period_col])
    idco_df.index = pd.RangeIndex(len(idco_df))  # clean integer obs_id index
    idco_df.index.name = "obs_id"

    return idco_df, period_map, avail_feats


def _larch_mnl_fit(timing_df: pd.DataFrame,
                   feat_cols: List[str],
                   period_col: str = "time_period") -> Optional[dict]:
    """
    Fit a Multinomial Logit over 6 time periods using Larch.

    Data format: IDCO (one row per observation; utility_co[j] encodes
    alternative-specific linear-in-parameters utility with IS covariates).
    Each of the 6 time-period alternatives gets its own ASC + its own slopes
    on every IS covariate.  The early/1 alternative is the base (no ASC).

    Engine: numba by default (no JAX/XLA required; see _lx_engine()).

    Returns dict with loglik, aic, bic, params, backend='Larch-MNL'
    """
    if not _LARCH_AVAILABLE:
        return None
    if len(timing_df.dropna(subset=feat_cols)) < MIN_CELL_N:
        return None
    try:
        idco_df, period_map, avail_feats = _lx_build_idco_timing(
            timing_df, feat_cols, period_col
        )
        if len(idco_df) < MIN_CELL_N:
            return None

        # Build Larch DataTree from IDCO data
        # lx.Dataset.construct.from_idco requires the index to be the case id
        dt = lx.DataTree(main=lx.Dataset.construct.from_idco(
            idco_df, alts=list(period_map.values())
        ))

        m = lx.Model(dt, compute_engine=_lx_engine())
        m.choice_co_code = "chosen_id"

        # Utility for base alternative (early=1): no ASC, only covariate slopes
        if avail_feats:
            m.utility_co[1] = sum(
                P(f"b_{f}_base") * LX(f) for f in avail_feats
            )

        # Utility for alternatives 2–6: ASC + alt-specific slopes on IS vars
        for label, alt_id in period_map.items():
            if alt_id == 1:
                continue   # base alternative — no ASC
            if avail_feats:
                m.utility_co[alt_id] = (
                    P(f"asc_{label}")
                    + sum(P(f"b_{f}_{label}") * LX(f) for f in avail_feats)
                )
            else:
                m.utility_co[alt_id] = P(f"asc_{label}")

        m.maximize_loglike(quiet=True)

        ll  = float(m.loglike())
        k   = int(m.n_params)
        n   = int(m.n_cases)
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        pf = m.pf.copy() if hasattr(m, "pf") else pd.DataFrame()

        return {
            "backend":   "Larch-MNL",
            "loglik":    ll,
            "aic":       aic,
            "bic":       bic,
            "n_params":  k,
            "n_obs":     n,
            "params":    pf,
            "model_obj": m,
        }
    except Exception as exc:
        log.warning(f"  Larch MNL fit failed: {exc}\n{traceback.format_exc()}")
        return None


def _larch_nested_logit_fit(timing_df: pd.DataFrame,
                             feat_cols: List[str],
                             period_col: str = "time_period") -> Optional[dict]:
    """
    Fit a Nested Logit over 6 time periods using Larch.

    Nest structure (mirrors TEMPORAL_NESTS):
        off_peak  [early=1, late=6]     → logsum param Mu_offpeak
        peak      [am_peak=2, pm_peak=4] → logsum param Mu_peak
        shoulder  [midday=3, evening=5]  → logsum param Mu_shoulder

    The logsum (inclusive value) parameter μ_k ∈ (0, 1] is estimated per nest.
    μ_k < 1 indicates positive cross-nest correlation (nesting is significant).
    μ_k = 1 reduces to MNL.

    All IS covariates enter via utility_co with the same alternative-specific
    slope structure as the MNL above.
    """
    if not _LARCH_AVAILABLE:
        return None
    if len(timing_df.dropna(subset=feat_cols)) < MIN_CELL_N:
        return None
    try:
        idco_df, period_map, avail_feats = _lx_build_idco_timing(
            timing_df, feat_cols, period_col
        )
        if len(idco_df) < MIN_CELL_N:
            return None

        # larch6 NestingTree: standalone graph object, NOT m.graph.new_node(...)
        g = lx.NestingTree()
        for label, alt_id in period_map.items():
            g.add_node(alt_id, name=label)
        # off_peak nest:  early(1), late(6)
        g.new_node(name="off_peak",  children=[1, 6], parameter="Mu_offpeak")
        # peak nest:      am_peak(2), pm_peak(4)
        g.new_node(name="peak",      children=[2, 4], parameter="Mu_peak")
        # shoulder nest:  midday(3), evening(5)
        g.new_node(name="shoulder",  children=[3, 5], parameter="Mu_shoulder")

        dt = lx.DataTree(main=lx.Dataset.construct.from_idco(
            idco_df, alts=list(period_map.values())
        ))
        m  = lx.Model(dt, graph=g, compute_engine=_lx_engine())
        m.choice_co_code = "chosen_id"

        # Utility specification: same structure as MNL
        if avail_feats:
            m.utility_co[1] = sum(P(f"b_{f}_base") * LX(f) for f in avail_feats)
        for label, alt_id in period_map.items():
            if alt_id == 1:
                continue
            m.utility_co[alt_id] = (
                P(f"asc_{label}")
                + (sum(P(f"b_{f}_{label}") * LX(f) for f in avail_feats)
                   if avail_feats else 0)
            )

        m.maximize_loglike(quiet=True)

        ll  = float(m.loglike())
        k   = int(m.n_params)
        n   = int(m.n_cases)
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        return {
            "backend":   "Larch-NL",
            "loglik":    ll,
            "aic":       aic,
            "bic":       bic,
            "n_params":  k,
            "n_obs":     n,
            "params":    m.pf.copy() if hasattr(m, "pf") else pd.DataFrame(),
            "nest_mus":  {
                "Mu_offpeak":  float(m.pf.loc["Mu_offpeak", "value"]) if hasattr(m, "pf") else np.nan,
                "Mu_peak":     float(m.pf.loc["Mu_peak",    "value"]) if hasattr(m, "pf") else np.nan,
                "Mu_shoulder": float(m.pf.loc["Mu_shoulder","value"]) if hasattr(m, "pf") else np.nan,
            },
            "model_obj": m,
        }
    except Exception as exc:
        log.warning(f"  Larch NestedLogit fit failed: {exc}\n{traceback.format_exc()}")
        return None


def _larch_mxl_fit(timing_df: pd.DataFrame,
                   feat_cols: List[str],
                   period_col: str = "time_period",
                   n_draws: int = 200) -> Optional[dict]:
    """
    Fit a Mixed Logit (random-parameter MNL) over 6 time periods via Larch.

    Random parameters are placed on the first two IS covariates (typically
    day_position and time_consumed), drawn from Normal distributions using
    Halton quasi-random sequences.  The remaining features enter as fixed.

    Note: MXL unconditionally requires JAX engine in larch5.  If
    JAX is unavailable and LARCH_ENGINE=numba, this returns None.
    """
    if not _LARCH_AVAILABLE:
        return None
    if len(timing_df.dropna(subset=feat_cols)) < MIN_CELL_N * 2:
        return None
    # MXL unconditionally requires JAX for Halton-MC integration in larch5
    try:
        engine = _lx_engine(require_jax=True)
    except RuntimeError:
        return None
    try:
        idco_df, period_map, avail_feats = _lx_build_idco_timing(
            timing_df, feat_cols, period_col
        )
        if len(idco_df) < MIN_CELL_N * 2:
            return None

        dt = lx.DataTree(main=lx.Dataset.construct.from_idco(
            idco_df, alts=list(period_map.values())
        ))
        m  = lx.Model(dt, compute_engine=engine)
        m.choice_co_code = "chosen_id"

        # Designate first two features as random (Normal distribution)
        rdm_feats  = avail_feats[:2]
        fixed_feats = avail_feats[2:]

        # Base utility for alternative 1 (early): fixed slopes + random slopes
        u_base = sum(P(f"b_{f}_base") * LX(f) for f in fixed_feats)
        for rf in rdm_feats:
            u_base = u_base + P(f"b_{rf}_base") * LX(rf)
        m.utility_co[1] = u_base

        for label, alt_id in period_map.items():
            if alt_id == 1:
                continue
            u = P(f"asc_{label}")
            for f in fixed_feats:
                u = u + P(f"b_{f}_{label}") * LX(f)
            for rf in rdm_feats:
                u = u + P(f"b_{rf}_{label}") * LX(rf)
            m.utility_co[alt_id] = u

        # Set Normal mixing distributions for the random params
        for rf in rdm_feats:
            for suffix in ["base"] + [lbl for lbl in period_map if lbl != "early"]:
                pname = f"b_{rf}_{suffix}"
                if pname in m.pf.index:
                    m.mixtures[pname] = lx.mixtures.Normal(
                        P(pname), P(f"sigma_{rf}_{suffix}")
                    )

        m.n_draws = n_draws
        m.maximize_loglike(quiet=True)

        ll  = float(m.loglike())
        k   = int(m.n_params)
        n   = int(m.n_cases)
        bic = k * np.log(n) - 2 * ll

        return {
            "backend":   "Larch-MXL",
            "loglik":    ll,
            "aic":       2 * k - 2 * ll,
            "bic":       bic,
            "n_params":  k,
            "n_obs":     n,
            "params":    m.pf.copy() if hasattr(m, "pf") else pd.DataFrame(),
            "model_obj": m,
        }
    except Exception as exc:
        log.warning(f"  Larch MXL fit failed (engine={engine}): {exc}\n{traceback.format_exc()}")
        return None


# ---------------------------------------------------------------------------
# Larch ↔ SearchLibrium / MetaCountRegressor GA bridge
# ---------------------------------------------------------------------------

class LarchSABridge:
    """
    Bridge between Larch discrete-choice estimation and the simulated-annealing
    variable-selection search engines in SearchLibrium and MetaCountRegressor.

    Purpose
    -------
    Both SL's ``call_siman`` and MCR's SA accept a fitness callable that maps a
    candidate solution to a scalar score (lower = better).  This class wraps a
    Larch MNL or Nested Logit fit and exposes ``.fitness(variable_mask)→float``
    so that the SA engine can search over *which* time-of-day covariates to
    include while using Larch's exact-gradient MLE for coefficient estimation.

    This is more powerful than fixing the variable set up front: the GA engine
    can discard noisy or collinear features, reducing BIC while Larch's
    log-likelihood ensures estimation quality.

    Parameters
    ----------
    timing_df   : HHTS timing DataFrame in wide format (one row per activity)
    all_feat_cols: candidate feature columns to search over
    period_col  : column name of the chosen time period label
    model_type  : 'mnl' | 'nested'  — which Larch model type to fit
    min_vars    : minimum number of variables that must be active (default 1)

    Usage with SearchLibrium call_siman
    ------------------------------------
    ::

        bridge  = LarchSABridge(timing_df, all_feat_cols, model_type='nested')
        result  = bridge.run_sl_search(n_iter=300, temperature=2.0)
        best_vars = result["best_vars"]
        best_bic  = result["best_bic"]

    Usage as a raw fitness callable
    ---------------------------------
    ::

        bridge = LarchSABridge(timing_df, feat_cols)
        mask   = np.ones(len(feat_cols), dtype=bool)   # start with all vars
        bic    = bridge.fitness(mask)                  # evaluate

    Usage with MetaCountRegressor SA evaluator
    --------------------------------------------
    ::

        from metacountregressor import StructureEvaluatorLC
        bridge = LarchSABridge(timing_df, feat_cols, model_type='nested')
        # Decision vector: binary mask over features
        evaluator = bridge.as_mcr_evaluator()
        # Then pass evaluator to MCR's run() with algo='sa'
    """

    def __init__(
        self,
        timing_df: pd.DataFrame,
        all_feat_cols: List[str],
        period_col: str = "time_period",
        model_type: str = "mnl",
        min_vars: int = 1,
    ):
        if not _LARCH_AVAILABLE:
            raise RuntimeError("larch not installed — LarchSABridge requires larch5")
        self.timing_df     = timing_df.copy()
        self.all_feat_cols = [c for c in all_feat_cols if c in timing_df.columns]
        self.period_col    = period_col
        self.model_type    = model_type.lower()
        self.min_vars      = min_vars
        self._n_vars       = len(self.all_feat_cols)
        self._best_bic     = np.inf
        self._best_mask    = np.ones(self._n_vars, dtype=bool)
        self._best_fit     = None
        self._history: List[dict] = []

    def _active_feats(self, variable_mask: np.ndarray) -> List[str]:
        """Map a boolean mask over all_feat_cols to the active feature list."""
        mask = np.asarray(variable_mask, dtype=bool)
        active = [f for f, m in zip(self.all_feat_cols, mask) if m]
        if len(active) < self.min_vars:
            # Force at least min_vars by re-enabling the first ones
            active = self.all_feat_cols[: self.min_vars]
        return active

    def fitness(self, variable_mask: np.ndarray) -> float:
        """
        Evaluate a candidate variable set.

        Parameters
        ----------
        variable_mask : boolean array of length len(all_feat_cols)
                        True → include this covariate, False → exclude

        Returns
        -------
        BIC value (float).  Lower is better.  Returns np.inf on fit failure.
        """
        active = self._active_feats(variable_mask)
        if self.model_type == "nested":
            fit = _larch_nested_logit_fit(self.timing_df, active, self.period_col)
        else:
            fit = _larch_mnl_fit(self.timing_df, active, self.period_col)

        if fit is None:
            return np.inf

        bic = fit.get("bic", np.inf)
        self._history.append({
            "n_vars": int(np.sum(variable_mask)),
            "active": active,
            "bic":    bic,
        })
        if bic < self._best_bic:
            self._best_bic  = bic
            self._best_mask = np.asarray(variable_mask, dtype=bool).copy()
            self._best_fit  = fit
        return float(bic)

    def run_sl_search(
        self, n_iter: int = 500, temperature: float = 2.0
    ) -> dict:
        """
        Run a simple simulated-annealing variable-selection search using
        the Larch BIC as fitness, without depending on SearchLibrium's
        internal model structures.

        At each step a random variable is toggled; the move is accepted by
        the Metropolis criterion.  This mirrors the SA approach in SearchLibrium
        but operates on the variable-selection mask directly.

        Parameters
        ----------
        n_iter      : number of SA iterations
        temperature : initial temperature (controls acceptance of worse moves)

        Returns
        -------
        dict with keys: best_vars, best_bic, best_fit, history
        """
        rng         = np.random.default_rng(42)
        mask        = np.ones(self._n_vars, dtype=bool)
        current_bic = self.fitness(mask)
        T           = float(temperature)

        for i in range(n_iter):
            # Toggle a random variable
            j         = rng.integers(0, self._n_vars)
            new_mask  = mask.copy()
            new_mask[j] = not new_mask[j]

            new_bic = self.fitness(new_mask)

            # Metropolis acceptance
            delta = new_bic - current_bic
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                mask        = new_mask
                current_bic = new_bic

            # Geometric cooling
            T *= 0.995

            if i % 50 == 0:
                log.debug(
                    f"  LarchSABridge iter={i}  T={T:.3f}  "
                    f"current_bic={current_bic:.1f}  best_bic={self._best_bic:.1f}"
                )

        return {
            "best_vars": self._active_feats(self._best_mask),
            "best_bic":  self._best_bic,
            "best_fit":  self._best_fit,
            "history":   self._history,
        }

    def as_mcr_evaluator(self):
        """
        Return a lightweight evaluator object compatible with
        MetaCountRegressor's StructureEvaluatorLC protocol.

        The returned object has a .fitness(decision_vector) → float method
        where decision_vector is treated as a boolean variable-selection mask.
        This can be passed to MCR's run() method with algo='sa' or algo='de'.

        Example
        -------
        ::

            from metacountregressor import CMFExperimentBuilder
            bridge    = LarchSABridge(timing_df, feat_cols, model_type='nested')
            evaluator = bridge.as_mcr_evaluator()
            # CMFExperimentBuilder or run() call would use evaluator.fitness
        """
        bridge = self

        class _MCREvaluator:
            def fitness(self, decision_vector) -> float:
                mask = np.asarray(decision_vector, dtype=bool)
                return bridge.fitness(mask)

        return _MCREvaluator()


def _sl_mnl_fit(timing_df: pd.DataFrame, feat_cols: List[str],
                period_col: str = "time_period") -> Optional[dict]:
    """
    Fit a Multinomial Logit model over 6 time periods using SearchLibrium.

    MNL is the baseline discrete-choice specification for time-of-day choice.
    It satisfies the IIA property (independence of irrelevant alternatives),
    which is a testable assumption — violation suggests a nested or mixed
    specification.

    SearchLibrium MNL uses BFGS optimisation with analytical gradient (CPU).
    """
    if not _SL_AVAILABLE or SL_MNL is None:
        return None
    timing_df = timing_df.dropna(subset=feat_cols)
    if len(timing_df) < MIN_CELL_N:
        return None
    try:
        X, y, alts, ids = _build_mnl_long_format(timing_df, feat_cols, period_col)
        model = SL_MNL(_jax=_SL_USE_JAX)
        model.setup(
            X, y,
            varnames=feat_cols,
            alts=alts,
            isvars=feat_cols,         # all vars are individual-specific (IS)
            ids=ids,
            fit_intercept=True,
            maxiter=2000,
            method="bfgs",
            scipy_optimisation=not _SL_USE_JAX,
        )
        model.fit()
        return {
            "backend":   "SL-MNL",
            "loglik":    getattr(model, "loglik",    np.nan),
            "aic":       getattr(model, "aic",       np.nan),
            "bic":       getattr(model, "bic",       np.nan),
            "coeff_est": getattr(model, "coeff_est", None),
            "coeff_names": getattr(model, "coeff_names", None),
            "model_obj": model,
        }
    except Exception as exc:
        log.warning(f"  SL MNL fit failed: {exc}\n{traceback.format_exc()}")
        return None


def _sl_nested_logit_fit(timing_df: pd.DataFrame, feat_cols: List[str],
                         period_col: str = "time_period") -> Optional[dict]:
    """
    Fit a Nested Logit model using SearchLibrium with temporal nests.

    Nesting structure (TEMPORAL_NESTS):
        off_peak  → {early, late}
        peak      → {am_peak, pm_peak}
        shoulder  → {midday, evening}

    The Nested Logit relaxes the IIA assumption *within* nests: alternatives
    in the same nest can share unobserved utility components, producing more
    realistic substitution patterns between adjacent time periods.

    The inclusive value parameter (λ) per nest is estimated; λ ∈ (0, 1]
    indicates consistent utility maximisation.
    """
    if not _SL_AVAILABLE or SL_NL is None:
        return None
    timing_df = timing_df.dropna(subset=feat_cols)
    if len(timing_df) < MIN_CELL_N:
        return None
    try:
        X, y, alts, ids = _build_mnl_long_format(timing_df, feat_cols, period_col)
        model = SL_NL(_jax=_SL_USE_JAX)
        # SearchLibrium's NestedLogit.setup signature is
        #     setup(self, X, X_nest, y, varnames=None, ...)
        # -- it gained a nest-level design matrix as positional #2. Calling
        # setup(X, y, ...) bound y to X_nest and left the required y missing:
        #     TypeError: NestedLogit.setup() missing 1 required positional
        #                argument: 'y'
        # We pass no nest-specific covariates, and setup handles X_nest=None
        # explicitly (F=0; utilities_Upper=0), so insert the placeholder only
        # when the installed version declares that parameter -- keeps this
        # working against older SearchLibrium builds too.
        _nl_params = inspect.signature(model.setup).parameters
        _setup_args = (X, None, y) if "X_nest" in _nl_params else (X, y)
        model.setup(
            *_setup_args,
            varnames=feat_cols,
            alts=alts,
            isvars=feat_cols,
            ids=ids,
            nests=TEMPORAL_NESTS,
            fit_intercept=True,
            maxiter=2000,
        )
        model.fit()
        return {
            "backend":   "SL-NL",
            "loglik":    getattr(model, "loglik",    np.nan),
            "aic":       getattr(model, "aic",       np.nan),
            "bic":       getattr(model, "bic",       np.nan),
            "coeff_est": getattr(model, "coeff_est", None),
            "model_obj": model,
        }
    except Exception as exc:
        log.warning(f"  SL NestedLogit fit failed: {exc}\n{traceback.format_exc()}")
        return None


def _pylogit_mnl_fit(timing_df: pd.DataFrame, feat_cols: List[str],
                     period_col: str = "time_period") -> Optional[dict]:
    """
    Fit a MNL model using pylogit as a cross-validation alternative.

    pylogit uses a different optimisation backend (scipy L-BFGS-B by default)
    which provides an independent log-likelihood check against SearchLibrium.
    """
    if not _PYLOGIT_AVAILABLE:
        return None
    timing_df = timing_df.dropna(subset=feat_cols)
    if len(timing_df) < MIN_CELL_N:
        return None
    periods = ["early", "am_peak", "midday", "pm_peak", "evening", "late"]
    try:
        # Build observation dataframe in pylogit format
        timing_df = timing_df.reset_index(drop=True)
        timing_df["obs_id"] = timing_df.index

        long_rows = []
        for _, row in timing_df.iterrows():
            chosen = row.get(period_col)
            for pid, period in enumerate(periods):
                r = {"obs_id": row["obs_id"], "alt_id": pid,
                     "choice": int(chosen == period)}
                for fc in feat_cols:
                    r[fc] = row.get(fc, 0.0)
                long_rows.append(r)
        ldf = pd.DataFrame(long_rows)

        spec   = {fc: "all_same" for fc in feat_cols}
        names  = {fc: [fc] for fc in feat_cols}
        model  = pylogit.create_choice_model(
            data=ldf, alt_id_col="alt_id", obs_id_col="obs_id",
            choice_col="choice", specification=spec, names=names,
            model_type="MNL",
        )
        init_coef = np.zeros(len(feat_cols))
        model.fit_mle(init_coef, disp=False)
        return {
            "backend": "pylogit-MNL",
            "loglik":  float(model.log_likelihood),
            "aic":     float(2 * len(feat_cols) - 2 * model.log_likelihood),
            "bic":     float(np.log(len(timing_df)) * len(feat_cols)
                             - 2 * model.log_likelihood),
            "model_obj": model,
        }
    except Exception as exc:
        log.warning(f"  pylogit MNL failed: {exc}\n{traceback.format_exc()}")
        return None


def stage4_schedule_placement(out: dict, out_dir: pathlib.Path) -> dict:
    """
    Stage 4 — time-period choice estimation on HHTS.

    For each activity purpose, fits all available discrete-choice estimators
    and selects the best by BIC (Kass–Raftery ΔBIc > 10 threshold):

        Larch MNL         — baseline MNL, exact gradients, numba engine
        Larch Nested      — temporal nests (off-peak/peak/shoulder), Larch NL
        Larch MXL         — mixed logit with random time parameters (JAX)
        SearchLibrium MNL — independent BFGS cross-check
        SearchLibrium NL  — SL nested logit cross-check
        pylogit MNL       — pylogit cross-check

    Additionally, a LarchSABridge variable-selection search is run on the full
    combined dataset (all purposes pooled) to identify the globally optimal
    feature set for the Larch MNL using simulated annealing.

    Validation: Bhattacharyya coefficient on period shares (observed vs
    in-sample predicted), plus a confusion matrix across time periods.
    """
    with _Timer("Stage 4 — schedule placement"):
        trips   = out.get("trips")
        tours   = out["tours"]
        persons = out["persons"]

        if trips is None:
            log.warning("  Stage 4: no trips data — timing data quality reduced")
            trips = pd.DataFrame()

        timing_df = prepare_timing_data(trips, tours, persons)
        if "obs_id" not in timing_df.columns:
            timing_df = timing_df.reset_index().rename(columns={"index": "obs_id"})

        base_feats = ["day_position", "time_consumed", "duration_min",
                      "time_remaining"]
        feat_cols  = [c for c in base_feats if c in timing_df.columns]
        feat_cols += [c for c in timing_df.columns
                      if c.startswith("purpose_") or c.startswith("ptype_")]

        period_col = "time_period" if "time_period" in timing_df.columns else "period"

        models: dict = {}
        val_rows: list = []

        for purpose in PURPOSES:
            mask    = timing_df.get("purpose", pd.Series(dtype=str)) == purpose
            cell_df = timing_df[mask].copy() if mask.any() else timing_df.copy()

            # ── Larch models (primary) ────────────────────────────────────
            lx_mnl_fit = _larch_mnl_fit(cell_df, feat_cols, period_col)
            lx_nl_fit  = _larch_nested_logit_fit(cell_df, feat_cols, period_col)
            lx_mxl_fit = _larch_mxl_fit(cell_df, feat_cols, period_col, n_draws=N_DRAWS)

            # ── SearchLibrium cross-check ─────────────────────────────────
            mnl_fit  = _sl_mnl_fit(cell_df, feat_cols, period_col)
            nl_fit   = _sl_nested_logit_fit(cell_df, feat_cols, period_col)
            plg_fit  = _pylogit_mnl_fit(cell_df, feat_cols, period_col)

            candidates = {k: v for k, v in [
                ("larch_mnl",    lx_mnl_fit),
                ("larch_nested", lx_nl_fit),
                ("larch_mxl",    lx_mxl_fit),
                ("sl_mnl",       mnl_fit),
                ("sl_nested",    nl_fit),
                ("pylogit_mnl",  plg_fit),
            ] if v is not None}

            best_key = (min(candidates, key=lambda k: candidates[k].get("bic", np.inf))
                        if candidates else None)

            models[purpose] = {**candidates, "best": best_key, "n": len(cell_df)}

            for k, fit in candidates.items():
                val_rows.append({
                    "stage": 4, "purpose": purpose, "model": k,
                    "n": len(cell_df),
                    "loglik": fit.get("loglik", np.nan),
                    "aic":    fit.get("aic",    np.nan),
                    "bic":    fit.get("bic",    np.nan),
                    "best":   k == best_key,
                })

        # ── Larch SA variable-selection search (pooled across all purposes) ──
        if _LARCH_AVAILABLE and len(feat_cols) > 1:
            log.info("  Stage 4: running LarchSABridge variable selection (pooled)...")
            try:
                bridge = LarchSABridge(
                    timing_df, feat_cols, period_col=period_col, model_type="mnl"
                )
                sa_result = bridge.run_sl_search(n_iter=300, temperature=2.0)
                models["_sa_variable_selection"] = sa_result
                log.info(
                    f"  Stage 4: SA best vars={sa_result['best_vars']}  "
                    f"BIC={sa_result['best_bic']:.1f}"
                )
                pd.DataFrame(sa_result["history"]).to_csv(
                    out_dir / "stage4_sa_history.csv", index=False
                )
            except Exception as exc:
                log.warning(f"  Stage 4: LarchSABridge failed: {exc}")

        with open(out_dir / "stage4_core_models.pkl", "wb") as fh:
            pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame(val_rows).to_csv(out_dir / "stage4_validation.csv", index=False)
        log.info(f"  Stage 4: fitted {len(val_rows)} (purpose×model) cells")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Validation report compiler
# ─────────────────────────────────────────────────────────────────────────────

def compile_validation_report(out_dir: pathlib.Path) -> None:
    """
    Merge all per-stage CSV validation files into a single summary and write
    an HTML report.  No population prediction involved — this is purely an
    in-sample HHTS model fit comparison.
    """
    with _Timer("Validation report"):
        frames = []
        for stage in range(1, 5):
            csv_path = out_dir / f"stage{stage}_validation.csv"
            if csv_path.exists():
                frames.append(pd.read_csv(csv_path))
        if not frames:
            log.warning("  No validation CSVs found — skipping report")
            return

        summary = pd.concat(frames, ignore_index=True)
        summary.to_csv(out_dir / "all_stages_validation.csv", index=False)

        # ── Best-per-cell pivot ─────────────────────────────────────────────
        best = summary[summary["best"] == True].copy()

        html_rows = []
        for _, row in best.iterrows():
            bg = "#e8f5e9" if row.get("status") == "ok" else "#ffebee" \
                if "status" in row else "#e8f5e9"
            html_rows.append(
                f"<tr style='background:{bg}'>"
                + "".join(f"<td>{v}</td>" for v in row.values)
                + "</tr>"
            )

        header = "".join(f"<th>{c}</th>" for c in best.columns)
        html = textwrap.dedent(f"""
        <!DOCTYPE html><html><head>
        <style>
          body{{font-family:sans-serif;margin:2em}}
          table{{border-collapse:collapse;width:100%}}
          th,td{{border:1px solid #ccc;padding:6px 10px;font-size:0.85em}}
          th{{background:#1565c0;color:#fff}}
          h2{{color:#1565c0}}
        </style></head><body>
        <h1>main_2 Core-Model Validation Report (HHTS only)</h1>
        <p>Generated: {datetime.datetime.now().isoformat(timespec='seconds')}</p>
        <p>CPU-only execution. Synthetic population NOT loaded.
           All metrics are <strong>in-sample</strong> on the full HHTS.</p>
        <h2>Best model per cell (by BIC)</h2>
        <table><thead><tr>{header}</tr></thead><tbody>
        {''.join(html_rows)}
        </tbody></table>
        <h2>All model variants</h2>
        {summary.to_html(index=False, border=0, classes='summary')}
        </body></html>
        """)
        (out_dir / "validation_report.html").write_text(html)
        log.info(f"  Validation report written → {out_dir / 'validation_report.html'}")

        # ── Timing summary ──────────────────────────────────────────────────
        if _TIMING:
            pd.DataFrame(_TIMING).to_csv(out_dir / "timings.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Best-specification persistence  (hall-of-fame across runs)
# ─────────────────────────────────────────────────────────────────────────────

_BEST_SPEC_PATH = pathlib.Path("saved_data") / "best_specs.json"

# Significance threshold for t-statistics  |t| ≥ this → significant
_T_THRESH = float(os.environ.get("SIG_T_THRESH", "1.96"))

# A model passes parsimony if it has ≤ this many more params than the stored best
_PARSIMONY_SLACK = int(os.environ.get("PARSIMONY_SLACK", "0"))


class BestSpecStore:
    """
    Persistent JSON store for the best discrete-choice model specification
    across pipeline runs.

    The store records — per (stage, purpose) cell — the best variable set,
    its BIC, the number of parameters, and parameter t-statistics.  A new
    candidate replaces the stored specification only if it satisfies *all*:

        1. BIC improvement  : new_bic < stored_bic  (strictly better)
        2. Significance     : all |t| ≥ _T_THRESH  (no insignificant params)
        3. Parsimony        : new_k ≤ stored_k + _PARSIMONY_SLACK

    This ensures the store always contains the most parsimonious, significant,
    and well-fitting specification — which can be used as a warm-start initial
    solution for subsequent SA search runs.

    File layout (JSON)
    ------------------
    {
      "meta": {"updated": "2026-05-06T...", "t_thresh": 1.96},
      "specs": {
        "stage4:work": {
          "vars":    ["day_position", "time_consumed"],
          "bic":     -1234.5,
          "n_params": 12,
          "t_stats": {"asc_am_peak": 3.2, "b_day_position_base": 2.1, ...},
          "backend": "Larch-NL",
          "updated": "2026-05-06T12:34:56"
        },
        ...
      }
    }

    Usage
    -----
    ::
        store = BestSpecStore()
        init_vars = store.get_vars("stage4", "work", default=BASE_FEAT_COLS)
        ...
        store.maybe_update("stage4", "work", fit_result, active_vars)
        store.save()
    """

    def __init__(self, path: pathlib.Path = _BEST_SPEC_PATH):
        self.path = path
        self._data: dict = {"meta": {}, "specs": {}}
        if path.exists():
            try:
                with open(path) as fh:
                    self._data = json.load(fh)
                log.info(f"BestSpecStore: loaded {len(self._data['specs'])} specs from {path}")
            except Exception as exc:
                log.warning(f"BestSpecStore: could not read {path}: {exc}. Starting fresh.")

    @staticmethod
    def _key(stage: str, purpose: str) -> str:
        return f"{stage}:{purpose}"

    def get_vars(self, stage: str, purpose: str,
                 default: Optional[List[str]] = None) -> List[str]:
        """Return the stored best variable list, or *default* if not yet stored."""
        key = self._key(stage, purpose)
        entry = self._data["specs"].get(key)
        if entry and entry.get("vars"):
            log.info(f"  BestSpecStore [{key}]: warm-start vars={entry['vars']}  BIC={entry['bic']:.1f}")
            return list(entry["vars"])
        return list(default) if default else []

    def get_entry(self, stage: str, purpose: str) -> Optional[dict]:
        """Return the full stored entry or None."""
        return self._data["specs"].get(self._key(stage, purpose))

    def maybe_update(
        self,
        stage: str,
        purpose: str,
        fit: dict,
        active_vars: List[str],
    ) -> bool:
        """
        Evaluate *fit* against the stored best and update the store if the
        candidate is strictly better.

        Parameters
        ----------
        stage      : e.g. "stage4"
        purpose    : e.g. "work"
        fit        : dict returned by _larch_mnl_fit / _larch_nested_logit_fit
        active_vars: list of variable names used in this fit

        Returns True if the store was updated, False otherwise.
        """
        if fit is None:
            return False

        new_bic   = fit.get("bic", np.inf)
        new_k     = fit.get("n_params", np.inf)
        new_back  = fit.get("backend", "unknown")

        # Extract t-statistics from Larch param frame
        t_stats: dict = {}
        pf = fit.get("params")
        if pf is not None and hasattr(pf, "iterrows"):
            for pname, row in pf.iterrows():
                se = float(row.get("std_err", 0) or 0)
                v  = float(row.get("value",   0) or 0)
                t_stats[str(pname)] = round(abs(v / se), 3) if se > 0 else np.inf

        # ── Gate 1: significance ───────────────────────────────────────────
        insig_params = [p for p, t in t_stats.items() if abs(t) < _T_THRESH]
        if insig_params:
            log.debug(
                f"  BestSpecStore [{stage}:{purpose}]: REJECTED — "
                f"insignificant params: {insig_params}"
            )
            return False

        # ── Gate 2: parsimony + BIC vs stored ─────────────────────────────
        key   = self._key(stage, purpose)
        entry = self._data["specs"].get(key)

        if entry:
            stored_bic = entry.get("bic", np.inf)
            stored_k   = entry.get("n_params", np.inf)

            if new_bic >= stored_bic:
                log.debug(
                    f"  BestSpecStore [{key}]: REJECTED — "
                    f"BIC {new_bic:.1f} ≥ stored {stored_bic:.1f}"
                )
                return False

            if new_k > stored_k + _PARSIMONY_SLACK:
                log.debug(
                    f"  BestSpecStore [{key}]: REJECTED — "
                    f"params {new_k} > stored {stored_k} + slack {_PARSIMONY_SLACK}"
                )
                return False

        # ── Accept ─────────────────────────────────────────────────────────
        self._data["specs"][key] = {
            "vars":     active_vars,
            "bic":      round(new_bic, 4),
            "n_params": int(new_k),
            "t_stats":  t_stats,
            "backend":  new_back,
            "updated":  datetime.datetime.now().isoformat(timespec="seconds"),
        }
        log.info(
            f"  BestSpecStore [{key}]: UPDATED  BIC={new_bic:.1f}  "
            f"k={new_k}  backend={new_back}  vars={active_vars}"
        )
        return True

    def save(self) -> None:
        """Write the store to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data["meta"] = {
            "updated":  datetime.datetime.now().isoformat(timespec="seconds"),
            "t_thresh": _T_THRESH,
            "parsimony_slack": _PARSIMONY_SLACK,
        }
        with open(self.path, "w") as fh:
            json.dump(self._data, fh, indent=2, default=str)
        log.info(f"BestSpecStore: saved {len(self._data['specs'])} specs to {self.path}")

    def report_df(self) -> pd.DataFrame:
        """Return a summary DataFrame for inclusion in the HTML report."""
        rows = []
        for key, entry in self._data["specs"].items():
            stage, purpose = key.split(":", 1)
            rows.append({
                "stage":   stage,
                "purpose": purpose,
                "backend": entry.get("backend", ""),
                "n_vars":  len(entry.get("vars", [])),
                "n_params": entry.get("n_params", ""),
                "bic":     entry.get("bic", ""),
                "updated": entry.get("updated", ""),
                "vars":    ", ".join(entry.get("vars", [])),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 warm-start helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stage4_core_fit(
    cell_df: pd.DataFrame,
    feat_cols: List[str],
    period_col: str,
    store: BestSpecStore,
    purpose: str,
) -> dict:
    """
    Fit Stage 4 time-of-day models for one purpose using a **fixed** variable
    set drawn from BestSpecStore (warm-start / core mode).

    The stored variable list is used as-is — no SA search.  All available
    estimators (Larch MNL, Larch NL, SL-MNL, …) are run on the fixed vars
    and the best by BIC is recorded.  The best fit is offered to the store
    for update (will only be accepted if it passes significance + parsimony).

    Returns the candidates dict (same schema as stage4_schedule_placement).
    """
    active_vars = store.get_vars("stage4", purpose, default=feat_cols)
    active_vars = [c for c in active_vars if c in cell_df.columns] or feat_cols

    lx_mnl  = _larch_mnl_fit(cell_df, active_vars, period_col)
    lx_nl   = _larch_nested_logit_fit(cell_df, active_vars, period_col)
    sl_mnl  = _sl_mnl_fit(cell_df, active_vars, period_col)
    sl_nl   = _sl_nested_logit_fit(cell_df, active_vars, period_col)
    plg     = _pylogit_mnl_fit(cell_df, active_vars, period_col)

    candidates = {k: v for k, v in [
        ("larch_mnl",    lx_mnl),
        ("larch_nested", lx_nl),
        ("sl_mnl",       sl_mnl),
        ("sl_nested",    sl_nl),
        ("pylogit_mnl",  plg),
    ] if v is not None}

    if candidates:
        best_key = min(candidates, key=lambda k: candidates[k].get("bic", np.inf))
        best_fit = candidates[best_key]
        store.maybe_update("stage4", purpose, best_fit, active_vars)

    return candidates


def _stage4_search_fit(
    cell_df: pd.DataFrame,
    feat_cols: List[str],
    period_col: str,
    store: BestSpecStore,
    purpose: str,
    n_iter: int = 400,
    temperature: float = 2.0,
    model_type: str = "nested",
) -> dict:
    """
    Fit Stage 4 time-of-day models for one purpose using **SA variable
    selection** warm-started from BestSpecStore.

    1. Load stored variable mask as initial SA state (warm start).
    2. Run LarchSABridge SA search to find a better variable subset.
    3. Re-fit all estimators on the SA-optimal variable set.
    4. Offer the best fit to BestSpecStore — replaces stored spec only if it
       passes BIC < stored_bic AND all |t| ≥ threshold AND parsimony gate.
    5. Return the full candidates dict (same schema as stage4_schedule_placement).

    Parameters
    ----------
    n_iter      : number of SA iterations per purpose cell
    temperature : initial SA temperature
    model_type  : 'mnl' | 'nested' — which Larch model to use inside SA
    """
    # ── Warm-start: build initial mask from stored vars ────────────────────
    init_vars = store.get_vars("stage4", purpose, default=feat_cols)
    avail     = [c for c in feat_cols if c in cell_df.columns]
    init_mask = np.array([c in init_vars for c in avail], dtype=bool)
    if not init_mask.any():
        init_mask[:] = True  # fall back to all-vars if nothing stored

    # ── SA search ─────────────────────────────────────────────────────────
    bridge = LarchSABridge(
        cell_df, avail, period_col=period_col, model_type=model_type
    )
    # Seed the bridge with the stored initial solution
    bridge._best_mask = init_mask.copy()
    bridge._best_bic  = bridge.fitness(init_mask)

    sa_result = bridge.run_sl_search(n_iter=n_iter, temperature=temperature)
    best_vars = sa_result["best_vars"]

    log.info(
        f"  SA [{purpose}]: best_vars={best_vars}  "
        f"BIC={sa_result['best_bic']:.1f}"
    )

    # ── Re-fit all estimators on the SA-selected vars ──────────────────────
    lx_mnl  = _larch_mnl_fit(cell_df, best_vars, period_col)
    lx_nl   = _larch_nested_logit_fit(cell_df, best_vars, period_col)
    lx_mxl  = _larch_mxl_fit(cell_df, best_vars, period_col, n_draws=N_DRAWS)
    sl_mnl  = _sl_mnl_fit(cell_df, best_vars, period_col)
    sl_nl   = _sl_nested_logit_fit(cell_df, best_vars, period_col)
    plg     = _pylogit_mnl_fit(cell_df, best_vars, period_col)

    candidates = {k: v for k, v in [
        ("larch_mnl",    lx_mnl),
        ("larch_nested", lx_nl),
        ("larch_mxl",    lx_mxl),
        ("sl_mnl",       sl_mnl),
        ("sl_nested",    sl_nl),
        ("pylogit_mnl",  plg),
    ] if v is not None}

    # ── Offer best fit to the hall-of-fame store ───────────────────────────
    if candidates:
        best_key = min(candidates, key=lambda k: candidates[k].get("bic", np.inf))
        store.maybe_update("stage4", purpose, candidates[best_key], best_vars)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Run-mode entry points
# ─────────────────────────────────────────────────────────────────────────────

def run_core(out_dir: Optional[pathlib.Path] = None) -> None:
    """
    **Core mode** — no variable-selection search.

    Uses the variable sets stored in BestSpecStore (saved_data/best_specs.json)
    for Stage 4 (time-of-day choice).  All other stages run their full model
    catalogue as usual.

    Suitable for:
    • Routine validation runs to confirm historical specs still fit well.
    • Lightweight PBS jobs that don't need SA walltime.
    • Generating stable comparison BICs before a search run.

    Override stored specs entirely by deleting saved_data/best_specs.json.
    """
    if out_dir is None:
        out_dir = _setup("main2_core")

    store = BestSpecStore()

    log.info("=" * 70)
    log.info("main_2  MODE=core  (fixed vars from BestSpecStore, no SA search)")
    log.info(f"  Larch engine    : {_LARCH_ENGINE}  (SL_JAX={_SL_USE_JAX})")
    log.info(f"  BestSpecStore   : {_BEST_SPEC_PATH}")
    log.info(f"  Stored specs    : {len(store._data['specs'])}")
    log.info(f"  Output          : {out_dir}")
    log.info("=" * 70)

    out      = stage0_load_data()
    s1       = stage1_activity_frequency(out, out_dir)
    s2       = stage2_day_duration(out, out_dir)
    s3       = stage3_activity_duration(out, out_dir)

    # Stage 4 with fixed vars
    with _Timer("Stage 4 — schedule placement (core, no search)"):
        trips  = out.get("trips", pd.DataFrame())
        timing = prepare_timing_data(trips, out["tours"], out["persons"])
        if "obs_id" not in timing.columns:
            timing = timing.reset_index().rename(columns={"index": "obs_id"})

        base_feats = ["day_position", "time_consumed", "duration_min", "time_remaining"]
        feat_cols  = [c for c in base_feats if c in timing.columns]
        feat_cols += [c for c in timing.columns
                      if c.startswith("purpose_") or c.startswith("ptype_")]
        period_col = "time_period" if "time_period" in timing.columns else "period"

        models: dict = {}
        val_rows: list = []

        for purpose in PURPOSES:
            mask    = timing.get("purpose", pd.Series(dtype=str)) == purpose
            cell_df = timing[mask].copy() if mask.any() else timing.copy()

            candidates = _stage4_core_fit(
                cell_df, feat_cols, period_col, store, purpose
            )
            best_key = (min(candidates, key=lambda k: candidates[k].get("bic", np.inf))
                        if candidates else None)
            models[purpose] = {**candidates, "best": best_key, "n": len(cell_df)}

            for k, fit in candidates.items():
                val_rows.append({
                    "stage": 4, "purpose": purpose, "model": k,
                    "n": len(cell_df),
                    "loglik": fit.get("loglik", np.nan),
                    "aic":    fit.get("aic",    np.nan),
                    "bic":    fit.get("bic",    np.nan),
                    "best":   k == best_key,
                })

        with open(out_dir / "stage4_core_models.pkl", "wb") as fh:
            pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame(val_rows).to_csv(out_dir / "stage4_validation.csv", index=False)

    # Persist any improvements found even in core mode
    store.save()
    store.report_df().to_csv(out_dir / "best_specs_snapshot.csv", index=False)
    compile_validation_report(out_dir)

    log.info("=" * 70)
    log.info("run_core complete.")
    log.info(f"  Results : {out_dir}")
    log.info("=" * 70)


def run_search(
    out_dir: Optional[pathlib.Path] = None,
    n_iter: int = 400,
    temperature: float = 2.0,
    search_model: str = "nested",
) -> None:
    """
    **Search mode** — SA variable selection warm-started from BestSpecStore.

    For each (purpose, stage4) cell:
    1. Load the stored best variable set as the SA initial solution.
    2. Run LarchSABridge simulated annealing to explore the variable space.
    3. Re-fit all estimators on the best-found variable set.
    4. Update BestSpecStore if the new spec beats the stored one on
       BIC + significance + parsimony.

    The updated store is written back to saved_data/best_specs.json — so the
    next run (core or search) automatically inherits the improved solution.

    Parameters (also settable via env vars)
    ----------------------------------------
    n_iter        : SA iterations per purpose  [env: SA_N_ITER]
    temperature   : initial SA temperature     [env: SA_TEMP]
    search_model  : 'mnl' | 'nested'           [env: SA_MODEL]
    """
    n_iter       = int(os.environ.get("SA_N_ITER",  str(n_iter)))
    temperature  = float(os.environ.get("SA_TEMP",  str(temperature)))
    search_model = os.environ.get("SA_MODEL", search_model).lower()

    if out_dir is None:
        out_dir = _setup("main2_search")

    store = BestSpecStore()

    log.info("=" * 70)
    log.info("main_2  MODE=search  (SA variable selection, warm-started)")
    log.info(f"  Larch engine    : {_LARCH_ENGINE}  (SL_JAX={_SL_USE_JAX})")
    log.info(f"  SA model        : {search_model}")
    log.info(f"  SA iterations   : {n_iter}  temperature={temperature}")
    log.info(f"  BestSpecStore   : {_BEST_SPEC_PATH}  ({len(store._data['specs'])} stored)")
    log.info(f"  sig. threshold  : |t| ≥ {_T_THRESH}")
    log.info(f"  Output          : {out_dir}")
    log.info("=" * 70)

    out = stage0_load_data()
    stage1_activity_frequency(out, out_dir)
    stage2_day_duration(out, out_dir)
    stage3_activity_duration(out, out_dir)

    # Stage 4 with SA search
    with _Timer("Stage 4 — schedule placement (search mode)"):
        trips  = out.get("trips", pd.DataFrame())
        timing = prepare_timing_data(trips, out["tours"], out["persons"])
        if "obs_id" not in timing.columns:
            timing = timing.reset_index().rename(columns={"index": "obs_id"})

        base_feats = ["day_position", "time_consumed", "duration_min", "time_remaining"]
        feat_cols  = [c for c in base_feats if c in timing.columns]
        feat_cols += [c for c in timing.columns
                      if c.startswith("purpose_") or c.startswith("ptype_")]
        period_col = "time_period" if "time_period" in timing.columns else "period"

        models: dict = {}
        val_rows: list = []
        sa_all_history: list = []

        for purpose in PURPOSES:
            mask    = timing.get("purpose", pd.Series(dtype=str)) == purpose
            cell_df = timing[mask].copy() if mask.any() else timing.copy()

            candidates = _stage4_search_fit(
                cell_df, feat_cols, period_col, store, purpose,
                n_iter=n_iter, temperature=temperature,
                model_type=search_model,
            )
            best_key = (min(candidates, key=lambda k: candidates[k].get("bic", np.inf))
                        if candidates else None)
            models[purpose] = {**candidates, "best": best_key, "n": len(cell_df)}

            for k, fit in candidates.items():
                val_rows.append({
                    "stage": 4, "purpose": purpose, "model": k,
                    "n": len(cell_df),
                    "loglik": fit.get("loglik", np.nan),
                    "aic":    fit.get("aic",    np.nan),
                    "bic":    fit.get("bic",    np.nan),
                    "best":   k == best_key,
                })

        with open(out_dir / "stage4_core_models.pkl", "wb") as fh:
            pickle.dump(models, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame(val_rows).to_csv(out_dir / "stage4_validation.csv", index=False)

    # Persist the hall-of-fame
    store.save()
    store.report_df().to_csv(out_dir / "best_specs_snapshot.csv", index=False)
    compile_validation_report(out_dir)

    log.info("=" * 70)
    log.info("run_search complete.  BestSpecStore updated where improvements found.")
    log.info(f"  Hall-of-fame    : {_BEST_SPEC_PATH}")
    log.info(f"  Results         : {out_dir}")
    log.info("=" * 70)


def main() -> None:
    """
    Dispatch to run_core or run_search based on the MODE env var.

    MODE=core   (default) — fixed variables from BestSpecStore, no SA search
    MODE=search           — SA variable selection warm-started from BestSpecStore

    All other env-var overrides:
        LARCH_ENGINE=jax|numba   (default: jax)
        SL_JAX=0|1               (default: 1)
        SA_N_ITER=400            iterations per purpose in search mode
        SA_TEMP=2.0              initial SA temperature
        SA_MODEL=nested|mnl      which Larch model to use inside SA
        SIG_T_THRESH=1.96        significance threshold for spec acceptance
        PARSIMONY_SLACK=0        extra params allowed vs stored best

    Execution contract (all modes)
    ----------------------------------
    • JAX_PLATFORM_NAME=cpu enforced — CPU-only, no GPU
    • HHTS only — synthetic population is never loaded
    • No population prediction — validation against HHTS observations only
    """
    mode = os.environ.get("MODE", "core").lower().strip()
    log.info(f"main_2 dispatch: MODE={mode}")

    if mode == "search":
        run_search()
    else:
        if mode != "core":
            log.warning(f"Unknown MODE='{mode}' — defaulting to core")
        run_core()


if __name__ == "__main__":
    print(
        "main_2.py is deprecated as a direct entrypoint. "
        "Use pipeline_logger.py hhts_core or pipeline_logger.py hhts_search instead."
    )
    main()
