"""
47_segment_speeding_predictive.py - PURE speeding-count segment model (no risk weighting).

Replaces the risk-weighted segment layer (hard-braking EB + weighted S_j) with a
predictive model whose ONLY estimand is finable speeding counts:

    Y_{i,g,p,h} = # vehicles >= MARGIN (default 10) km/h over the posted limit
                  on segment i, fixture g, phase p, clock-hour bin h.

Why this structure (sparsity facts, from results/finable_speeding_audit.json):
  only ~987 of 40,177 finable exceedances fall inside the 41x7h fixture
  windows (~3.4 events/hour network-wide). A direct segment x fixture x hour
  NB random-parameters fit on ~1M mostly-zero rows cannot converge. So:

    Stage A (metacountregressor NB-RP, current PyPI release 1.0.159 or newer):
    segment baseline lambda_i from ALL train-period bounded finable totals
      (event + background: attendance>0 and attendance=0 alike), with the
      segment-characteristic structure itself SEARCHED by metacountregressor
    (ExperimentBuilder SparseEA-AGDS search minimising BIC over candidates):
        continuous road characteristics: lanes, posted speed (maxspeed_kmh),
          carriageway width proxy (width_m = lanes x 3.5 m), curvature
          (curvature_deg_km), sinuosity, segment length, road-class rank,
          node degree, distance to nearest intersection, distance to stadium;
        Overture block (Polaris-style build, script 25): edge-to-edge width
          (ov_width_m, width_rules), directional speed limits
          (ov_speed_fwd/bwd, speed_limits), finer class rank (ov_class_rank),
          connector counts (n_ov_connectors, incl. mid-segment routing
          decision points); missing lanes/speeds KNN-imputed within class
          with a minimum-one-lane rule, lengths from projected geometry,
          connector-shared endpoints refining intersection topology;
        countable OSM features: has_name (named vs unnamed), has_ref
          (numbered route shield), oneway, is_link (ramp/link), lane-count
          missing flag, speed-band indicators, dead-end proxy (node_degree<=2
          and far from intersection), lanes>=3 multi-lane flag;
        countable Overture flags: paved/unpaved (road_surface), ov_is_link,
          is_tunnel/is_bridge/is_toll (road_flags), has_ov_ref (routes),
          ov_mid_connector;
        optional POI counts: data/segment_poi_counts.csv (road_osmid +
          count columns, e.g. n_poi_100m, n_food_100m; nearest-segment join
          done offline) is consumed column-wise when present, otherwise
          skipped gracefully and noted in the audit (never fabricated).
      Constraints force exposure in (logexp fixed), forbid zero-inflation on
      geometry/countable terms (all rows have trips>0: zeros are sampling
      zeros), and allow normal random parameters on continuous geometry plus
      grouped effects by ROAD TYPE (highway functional class: group-level
      random effects shared within class + class-driven heterogeneity in
      random-parameter means). Best searched spec is refit with more draws
      for final lambda_base; --no-search falls back to the hand-built
      P1-style manual spec.
  Stage B1 (network event model, train games only):
      log E[Y_g] = c0 + eta*log(A_g+1) + DOW/weekend + kickoff cyclic +
                   fixture-type effects. This is the per-time / per-day-of-week /
                   attendance model that turns a segment baseline into a
                   predicted number of speeding events per hour.
  Stage B2 (background surface, attendance = 0): day-of-week x clock-hour
      network rates B(d,h) from train non-window hours; per-segment
      h_bg(i,d,h) = lambda_i * B(d,h) / sum_j lambda_j. The model therefore
      predicts EVERY hour of the week, not just fixture windows.
  Stage C (hour shares): empirical phase shares x within-phase hourly profile
      from train event rows only, so h_{i,g,p,h} is an hourly prediction.

  h_{i,g,p,h} = lambda_i * crowd_time_scale(g) * phase_share(p) * hour_share(h|p)
  h_bg(i,d,h) = lambda_i * B(d,h) / sum_j lambda_j   (attendance = 0)

Game holdout + convergence (the crucial validation for the MIP roster):
  - Stratified split: ~32 train / ~9 holdout fixtures (holdout spans small
    A-League, mid SuperRugby, large NRL so the crowd gradient is tested).
  - Fit A+B+C on train only; predict holdout segment totals and thresholds.
  - Report: Spearman rho, top-K Jaccard, RMSE, calibration slope/intercept,
    predicted-vs-real threshold table, and a learning curve (holdout skill vs
    # train games) = the convergence of the segment speeding model.
  - Artefact guards: abort (non-zero exit, no paper artefacts) if holdout
    Spearman < 0.5 or calibration slope outside [0.3, 3.0].

Deployment use (ABSOLUTELY NO risk weighting):
  b_j for P4 and r_{jt} for P5/P6 are pure predicted speeding counts summed
  per km over adjoining segments. No hard-braking risk, no flow/degree/
  betweenness weights, no distance-decay weights. Script 48 consumes the
  *_post_hourly outputs of this script.

Inputs:
    results/finable_exceedances.csv.gz  (per-event: road_osmid, dt, exceedance,
                                                                             fixture, phase; bounded by the upper
                                                                             exceedance limit in 43)
  data/fixture_calendar.csv           (label, kickoff_iso, attendance, fixture_type)
  data/road_geometry_features.csv     (road_osmid, maxspeed_kmh, lanes, ...)
  data/near_miss_hb_ecf.csv           (road_osmid, exposure_trips, highway, ...)

Outputs (all train-fit; holdout scored only):
  results/segment_speeding_pred_base.csv        (road_osmid, lambda_base)
  results/segment_speeding_pred_fixture.csv.gz  (fixture x segment + phase totals)
  results/segment_speeding_pred_fixture_hourly.csv.gz
      (segment x fixture x phase x hour: h_pred per before/during/after hour,
       kickoff/attendance/capacity/utilisation/flow_compass/asof; feeds 48)
  results/segment_speeding_pred_phase_totals.csv (fixture x phase predicted totals)
  results/segment_speeding_copula_compare.csv   (phase-pair copula hypothesis tests)
  results/segment_search47_progress.csv         (recoverable search trace)
  figures/segment_search47_progress.png         (search convergence plot)
    figures/segment_search47_dynamic_harmony.png  (latest library search plot)
  results/segment_speeding_holdout.csv          (holdout: observed vs predicted per segment)
  results/segment_speeding_pred_post_hourly.csv (post x bucket predicted speeding; feeds 48)
  results/segment_speeding_pred.tex             (paper equation + estimates)
  results/segment_speeding_convergence.tex      (holdout + learning-curve table)
  results/segment_speeding_pred_audit.json
  figures/segment_speeding_calibration.png      (predicted vs real thresholds)
  figures/segment_speeding_convergence.png      (learning curve)

Usage:
  python 47_segment_speeding_predictive.py [--margin 10] [--max-exceedance 30]
      [--n-holdout 9] [--seed 7]
         [--search-algo sa] [--search-iters 500] [--search-chunk-iters 100]
         [--search-R 150] [--refit-R 500] [--asof-iso ISO] [--no-copula]
         [--copula-max-rows 120000] [--no-search] [--resume-from JSON]

Requires: the latest PyPI metacountregressor release (currently 1.0.159 or newer).
    python -m pip install --upgrade metacountregressor

Optional POI enrichment (countable, never fabricated):
  data/segment_poi_counts.csv with columns road_osmid + one or more
  non-negative count columns (e.g. n_poi_100m, n_food_100m, n_shop_100m).
  Build it offline by buffering segment midpoints (e.g. 100 m) against an
  OSM POI extract and counting points per segment; the script consumes
  whatever count columns it finds as additional search candidates.
"""
import argparse
import gc
import importlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import metacountregressor as _mcr
    from importlib.metadata import PackageNotFoundError, version as _package_version
    try:
        # Distribution metadata is authoritative; older wheels exposed a stale
        # module __version__ even when the PyPI distribution was newer.
        _MCR_VERSION = _package_version("metacountregressor")
    except PackageNotFoundError:
        _MCR_VERSION = getattr(_mcr, "__version__", "unknown")
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        "FATAL: metacountregressor not installed. "
        "Run: python -m pip install --upgrade metacountregressor "
        f"(import error: {e})")

_TUP = tuple(int(x) for x in str(_MCR_VERSION).split(".")[:3] if x.isdigit())
try:
    importlib.import_module("metacountregressor.numba_count")
    _LOCAL_NUMBA_BACKEND = True
except ImportError:
    _LOCAL_NUMBA_BACKEND = False


def _validate_mcr_version(engine):
    """Require current PyPI MetaCount, or the tested local Numba checkout."""
    if _TUP >= (1, 0, 188):
        return
    if (str(engine).lower() == "numba" and _TUP >= (1, 0, 158)
            and _LOCAL_NUMBA_BACKEND):
        print(
            f"   using local MetaCount Numba backend {_MCR_VERSION}; "
            "PyPI JAX runs require >=1.0.188")
        return
    raise SystemExit(
        f"FATAL: metacountregressor {_MCR_VERSION} < 1.0.188. "
        "Install the latest release from PyPI: "
        "python -m pip install --upgrade metacountregressor")


def _stage_b_design(data, dummy_types=None):
    """Build the Stage B design matrix without a statsmodels dependency."""
    frame = pd.DataFrame(index=data.index)
    frame["const"] = 1.0
    for name in ("log_att", "weekend", "dow_sin", "dow_cos",
                 "ko_sin", "ko_cos"):
        values = (data[name] if name in data.columns
                  else pd.Series(0.0, index=data.index))
        frame[name] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    types = (data["fixture_type"].fillna("").astype(str)
             if "fixture_type" in data.columns
             else pd.Series("", index=data.index))
    if dummy_types is None:
        dummy_types = sorted(types.unique().tolist())[1:]
    for name in dummy_types:
        type_name = name.removeprefix("fixture_type_")
        frame[name] = (types == type_name).astype(float)
    return frame.to_numpy(dtype=float), list(frame.columns), dummy_types


class _StageBPoissonFallback:
    def __init__(self, columns, params, fittedvalues, aic, resid_pearson,
                 dummy_types):
        self._columns = columns
        self._dummy_types = dummy_types
        self.params = pd.Series(params, index=columns)
        self.fittedvalues = np.asarray(fittedvalues, dtype=float)
        self.aic = float(aic)
        self.resid_pearson = np.asarray(resid_pearson, dtype=float)
        self.scale = 1.0

    def predict(self, data):
        x, columns, _ = _stage_b_design(data, self._dummy_types)
        if columns != self._columns:
            aligned = np.zeros((len(data), len(self._columns)), dtype=float)
            for j, name in enumerate(columns):
                if name in self._columns:
                    aligned[:, self._columns.index(name)] = x[:, j]
            x = aligned
        return np.exp(np.clip(x @ self.params.to_numpy(), -30.0, 30.0))


def _fit_stage_b_poisson_fallback(data):
    from scipy.optimize import minimize
    from scipy.special import gammaln

    x, columns, dummy_types = _stage_b_design(data)
    y = pd.to_numeric(data["n"], errors="coerce").fillna(0.0).to_numpy()
    beta0 = np.linalg.lstsq(x, np.log1p(y), rcond=None)[0]

    def objective(beta):
        eta = np.clip(x @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta))

    def gradient(beta):
        eta = np.clip(x @ beta, -30.0, 30.0)
        return x.T @ (np.exp(eta) - y)

    fit = minimize(objective, beta0, jac=gradient, method="L-BFGS-B",
                   options={"maxiter": 2000})
    beta = np.asarray(fit.x if fit.success else beta0, dtype=float)
    mu = np.exp(np.clip(x @ beta, -30.0, 30.0))
    loglik = float(np.sum(y * np.log(np.maximum(mu, 1e-12)) - mu
                         - gammaln(y + 1.0)))
    aic = 2.0 * len(beta) - 2.0 * loglik
    resid = (y - mu) / np.sqrt(np.maximum(mu, 1e-12))
    model = _StageBPoissonFallback(columns, beta, mu, aic, resid,
                                    dummy_types)
    return model, float(max((np.sum(resid ** 2) / max(len(y) - len(beta), 1)
                             - 1.0) / max(np.mean(mu), 1e-9), 1e-3))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import prepare_metacount_frame, fit_metacount_nb

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
os.chdir(PROJ)

EXCEED = os.path.join(RES, "finable_exceedances.csv.gz")
EXCEED_OUTLIERS = os.path.join(RES, "finable_exceedances_outliers.csv.gz")
CAL = os.path.join(DATA, "fixture_calendar.csv")
FEATS = os.path.join(DATA, "road_geometry_features.csv")
SCREEN = os.path.join(DATA, "near_miss_hb_ecf.csv")
META = os.path.join(DATA, "road_meta.csv")
GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
POI = os.path.join(DATA, "segment_poi_counts.csv")  # optional, see header
POSTS = os.path.join(RES, "post_selection_mip.csv")
# Optional operational inputs (never fabricated; skipped gracefully if absent).
# Compass flow amounts, e.g. loop/SCATS-derived counts joined to segments:
#   road_osmid (required int), flow_veh (required >=0),
#   optional fixture (calendar label), phase (pre/during/post), hbin (-3..3).
COMPASS = os.path.join(DATA, "compass_flow.csv")
# Stadium/game info per fixture (capacity lets attendance enter as utilisation):
#   fixture or label (required calendar label), stadium_capacity (required >0),
#   stadium (optional name).
STADGAMES = os.path.join(DATA, "stadium_games.csv")
# Recoverable search artefacts (written per chunk so a walltime kill still
# leaves a usable best-spec checkpoint + convergence trace for --resume-from).
SEARCH_BEST = os.path.join(RES, "segment_search47_best.json")
SEARCH_PROGRESS = os.path.join(RES, "segment_search47_progress.csv")
SEARCH_PROGRESS_PNG = os.path.join(FIG, "segment_search47_progress.png")
# Fixture-relative hourly occurrence predictions feeding the MIP (script 48):
# one row per segment x fixture x phase x hour bin.
FIXTURE_HOURLY = os.path.join(RES, "segment_speeding_pred_fixture_hourly.csv.gz")
PHASE_TOTALS = os.path.join(RES, "segment_speeding_pred_phase_totals.csv")
COPULA_COMPARE = os.path.join(RES, "segment_speeding_copula_compare.csv")
COPULA_TEX = os.path.join(RES, "segment_speeding_copula.tex")
COPULA_AUDIT = os.path.join(RES, "segment_speeding_copula_audit.json")

MARGIN_DEFAULT = 10.0
MAX_EXCEEDANCE_DEFAULT = 30.0
# JAX runs the full random-parameter search. Numba is a local fixed-effects
# Poisson/NB2 search backend; final production refits remain JAX-based.
ENGINE_DEFAULT = "jax"
# 7 one-hour bins covering KO-3h..KO+4h; phase map matches paper (pre/during/post).
PHASE_HOURS = {"pre": [-3, -2, -1], "during": [0, 1], "post": [2, 3]}
HOUR_BINS = [-3, -2, -1, 0, 1, 2, 3]


def _rget(res, key, default=None):
    """builder.run() returns an object in some builds, a dict in others."""
    if hasattr(res, key):
        return getattr(res, key)
    if isinstance(res, dict):
        return res.get(key, default)
    return default


def _load_search_result_file(path):
    """Read a SearchOutputConfig JSON (nested result.model_spec format).

    Returns (best_score, best_spec) with Nones where absent.
    """
    with open(path) as f:
        d = json.load(f)
    r = d.get("result", d)
    return r.get("best_score"), r.get("model_spec")


def _newest_search_json(out_dir, pattern="seg_speed_*.json"):
    """Newest SearchOutputConfig JSON in out_dir, or None."""
    import glob
    cands = sorted(glob.glob(os.path.join(out_dir, pattern)),
                   key=os.path.getmtime)
    return cands[-1] if cands else None


def _inspect_groups(mc, col="highway", min_n=20):
    """Inspect group_id column BEFORE enabling grouped effects (roles 4/5).

    Grouped proposals error with `ValueError: Grouped effects require
    group_id_col` and waste search iters (fitness=1e12) when the group
    column is missing, null, or has only rare levels. Only return True
    when `col` is a true common term: present, >=2 levels each with
    >= min_n segments.
    Returns (use_grouped, counts_dict).
    """
    if col not in mc.columns:
        print(f"   group inspection ({col}): column absent; grouped OFF")
        return False, {}
    s = mc[col].fillna("").astype(str).str.strip()
    s = s[s != ""]
    vc = s.value_counts()
    counts = {str(k): int(v) for k, v in vc.items()}
    print(f"   group inspection ({col}): {counts}")
    n_common = int((vc >= int(min_n)).sum())
    if len(vc) >= 2 and n_common >= 2:
        print(f"   grouped ON: {n_common} common {col} groups >= {min_n} segs "
              f"({len(vc)} total levels)")
        return True, counts
    print(f"   grouped OFF: only {n_common} {col} groups >= {min_n} segs "
          f"(need >=2 common groups); falling back to roles [0,1,2]")
    return False, counts


def _build_evaluator(builder, variables, constraints, roles, R, engine):
    """Build the requested count evaluator with compatibility fallback.

    Mixed structural equations for the random parameters require roles
    including correlated (3) + heterogeneity (5); grouped (4) is only
    included when the group inspection passed. Numba intentionally supports
    only fixed-effects single-class count models.
    Returns (evaluator, engine_used).
    """
    if str(engine).lower() == "numba":
        ev = builder.build_evaluator(
            variables=variables,
            constraints=constraints,
            default_roles=[0, 1],
            max_latent_classes=1,
            R=1,
            engine="numba")
        print("   engine: numba (fixed-effects Poisson/NB2)")
        return ev, "numba"
    try:
        ev = builder.build_evaluator(
            variables=variables,
            constraints=constraints,
            default_roles=roles,
            max_latent_classes=1,
            R=int(R),
            engine=engine)
        print(f"   engine: {engine}")
        return ev, engine
    except (TypeError, ValueError) as e:
        # Older builds without explicit engine support can raise here.
        print(f"   WARN: engine={engine} unsupported ({e}); falling back to default JAX-first engine")
        ev = builder.build_evaluator(
            variables=variables,
            constraints=constraints,
            default_roles=roles,
            max_latent_classes=1,
            R=int(R))
        return ev, "default-jax"


def _numba_spec_from_decision(decision, variables):
    """Convert a fixed-effects Numba decision vector to a refit spec."""
    variables = list(variables)
    decision = np.asarray(decision, dtype=int).reshape(-1)
    n_vars = len(variables)
    if decision.size < 2 * n_vars + 1:
        raise ValueError(
            f"Numba decision has {decision.size} entries; expected at least "
            f"{2 * n_vars + 1} for {n_vars} variables")
    roles = decision[:n_vars]
    return {
        "fixed_terms": [
            var for var, role in zip(variables, roles) if int(role) == 1
        ],
        "rdm_terms": [],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "hetro_in_variances": [],
        "zi_terms": [],
        "membership_terms": [],
        "class_membership": None,
        "dispersion": int(decision[2 * n_vars]) % 2,
        "latent_classes": 1,
        "group_id_col": None,
    }


def _fit_numba_selected_spec(mc, spec, refit_R):
    """Refit a Numba fixed-effects winner through the normal JAX fitter."""
    fixed_terms = [c for c in spec.get("fixed_terms", []) if c in mc.columns]
    model = "nb" if int(spec.get("dispersion", 1)) else "poisson"
    return fit_metacount_nb(
        mc,
        y_col="Y",
        id_col="road_osmid",
        fixed_terms=fixed_terms,
        rdm_terms=[],
        model=model,
        R=int(refit_R),
        verbose=True,
    )


class _TracingEvaluator:
    """Proxy the evaluator so long fitness calls are visible immediately."""

    # Bound JAX/XLA memory across long searches: every fitness eval fits a
    # new model structure (fresh JIT compiles) and JAX pools the buffers
    # instead of returning them, so RSS ratchets per eval (OOM-killed at
    # ~66GB in o25828879 after 510 evals with no clearing anywhere).
    # Periodic gc + XLA cache clear flattens the footprint; repeated
    # structures recompile cheaply against the evaluator's fitness cache.
    _JAX_CLEAR_EVERY = 25

    def __init__(self, evaluator):
        self._evaluator = evaluator
        self._calls = 0
        self._best = float("inf")

    def fitness(self, decision):
        self._calls += 1
        call = self._calls
        pid = os.getpid()
        print(f"   [fitness] eval={call} pid={pid} start", flush=True)
        try:
            value = self._evaluator.fitness(decision)
        except Exception as e:  # noqa: BLE001 - preserve the original failure
            print(f"   [fitness] eval={call} FAILED: {type(e).__name__}: {e}",
                  flush=True)
            raise
        values = np.asarray(value, dtype=float).reshape(-1)
        finite = values[np.isfinite(values)]
        score = float(np.min(finite)) if finite.size else float("inf")
        self._best = min(self._best, score)
        score_text = "nan" if not np.isfinite(score) else f"{score:.4f}"
        best_text = "nan" if not np.isfinite(self._best) else f"{self._best:.4f}"
        print(f"   [fitness] eval={call} score={score_text} "
              f"best={best_text}", flush=True)
        if call % self._JAX_CLEAR_EVERY == 0:
            try:
                gc.collect()
            except Exception:
                pass
            try:
                import jax as _jax
                _jax.clear_caches()
            except Exception:
                pass
        return value

    def __getattr__(self, name):
        return getattr(self._evaluator, name)


class _SparseEAAGDSObjective:
    """Adapt the segment evaluator to MetaCount's generic AGDS primitive."""

    def __init__(self, evaluator, variables):
        source = evaluator
        while hasattr(source, "_evaluator"):
            source = source._evaluator
        self.evaluator = evaluator
        self.source = source
        self.variables = list(variables)
        self.n_variables = len(self.variables)
        self._obj_1 = "bic"
        self._obj_2 = None
        self.is_multi = False
        self.algorithm = "agds"
        self.instance_name = os.path.join(RES, "search47_agds")
        self.Last_Sol = None
        self.calls = 0
        self._max_time = float("inf")

        role_values = getattr(source, "allowed_roles", {})
        distribution_values = getattr(source, "allowed_distributions", {})
        self._discrete_values = [
            [int(value) for value in role_values.get(name, [0, 1])]
            for name in self.variables
        ]
        # Encode categorical distributions as 1..N because AGDS treats 0 as
        # an inactive sparse gene; decode back to the evaluator's 0..N-1 code.
        self._discrete_values.extend(
            [list(range(1, len(distribution_values.get(name, ["normal"]))+1))
             for name in self.variables]
        )
        self._discrete_values.append([1, 2])

    def get_num_parameters(self):
        return len(self._discrete_values)

    def get_value(self, index):
        import random
        return random.choice(self._discrete_values[index])

    def is_discrete(self, index):
        return True

    def _decode(self, layout):
        layout = np.asarray(layout, dtype=int).reshape(-1)
        expected = 2 * self.n_variables + 1
        if layout.size != expected:
            raise ValueError(
                f"AGDS layout has {layout.size} genes; expected {expected}")
        decision = np.zeros(expected, dtype=int)
        decision[:self.n_variables] = layout[:self.n_variables]
        decision[self.n_variables:2 * self.n_variables] = (
            layout[self.n_variables:2 * self.n_variables] - 1)
        decision[2 * self.n_variables] = int(layout[-1]) - 1
        return decision

    def decode(self, layout):
        return self._decode(layout)

    def get_fitness(self, layout, multi=False, max_routine=2):
        decision = self._decode(layout)
        self.calls += 1
        raw = self.evaluator.fitness(decision)
        if isinstance(raw, dict):
            score = raw.get("bic", raw.get("best_score", float("inf")))
        else:
            values = np.asarray(raw, dtype=float).reshape(-1)
            score = values[0] if values.size else float("inf")
        score = float(score)
        if not np.isfinite(score):
            score = 1e12
        self.Last_Sol = {"layout": list(layout), "bic": score}
        return {"layout": list(layout), "bic": score}

    def _get_obj1(self):
        return self._obj_1

    def _get_obj2(self):
        return self._obj_2

    def use_random_seed(self):
        return False

    def set_random_seed(self):
        return None

    def get_max_time(self):
        return self._max_time


def _run_sparse_ea_agds(evaluator, variables, max_iter, seed):
    """Run generic SparseEA-AGDS and return the builder-style best result."""
    import random
    random.seed(int(seed))
    np.random.seed(int(seed))
    from metacountregressor.metaheuristics import SparseEA_AGDS

    objective = _SparseEAAGDSObjective(evaluator, variables)
    solver = SparseEA_AGDS(
        objective, _hms=20, _max_iter=int(max_iter), verbose=False)
    result = solver.run()
    layout, best_score = result[0], float(result[1])
    decision = objective.decode(layout)
    best_spec = evaluator.build_spec(decision)
    if best_spec is None or not np.isfinite(best_score) or best_score >= 1e12:
        raise RuntimeError("SparseEA-AGDS returned no valid segment model")
    return {
        "algorithm": "agds",
        "best_score": best_score,
        "best_solution": decision,
        "best_spec": best_spec,
        "n_evals": objective.calls,
    }


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 10:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy(dtype=float, copy=True)
    rb = pd.Series(b).rank().to_numpy(dtype=float, copy=True)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()), 1e-12))


def _jaccard_topk(obs, pred, k):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    k = int(min(k, len(obs)))
    if k <= 0:
        return float("nan")
    so = set(np.argsort(-obs)[:k].tolist())
    sp = set(np.argsort(-pred)[:k].tolist())
    return float(len(so & sp) / max(len(so | sp), 1))


_STOP = {"flag": False, "signo": None}


def _handle_stop(signo, frame):  # noqa: ARG001 - PBS walltime gives SIGTERM first
    _STOP["flag"] = True
    _STOP["signo"] = int(signo)
    print(f"\n[stop] signal {signo}: checkpointing search progress, "
          "then stopping after the current chunk...", flush=True)


def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_csv(df, path, **kwargs):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def _extract_search_best(evaluator):
    """Best decision/spec/score seen so far in the evaluator cache.

    Lets every search chunk checkpoint a usable best-spec (for
    --resume-from) even if the job is killed before the full run ends.
    Returns (best_bic, best_spec_or_None, n_cache_evals).
    """
    best_bic, best_key, n_eval = float("inf"), None, 0
    try:
        items = list(getattr(evaluator, "cache", {}).items())
    except Exception:  # noqa: BLE001
        items = []
    n_eval = len(items)
    want = None
    try:
        want = len(getattr(evaluator, "vars", []) or [])
    except Exception:  # noqa: BLE001
        want = None
    for k, v in items:
        if not isinstance(k, (tuple, list)):
            continue
        if want and len(k) not in (2 * want + 1, 3 * want + 2):
            continue
        try:
            val = float(np.min(np.asarray(v, dtype=float)))
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(val) and val < 1e12 and val < best_bic:
            best_bic, best_key = val, k
    if best_key is None:
        return float("nan"), None, n_eval
    try:
        spec = evaluator.build_spec(np.asarray(best_key, dtype=float))
    except Exception:  # noqa: BLE001
        spec = None
    return float(best_bic), spec, n_eval


def _write_search_checkpoint(path, *, spec, best_bic, algo, iters_completed,
                             iters_total, R, seed, engine, roles, partial):
    _atomic_write_json(path, {
        "best_spec": spec,
        "best_bic": (None if spec is None or not np.isfinite(best_bic)
                     else float(best_bic)),
        "algo": algo,
        "iters_completed": int(iters_completed),
        "iters_total": int(iters_total),
        "partial": bool(partial),
        "R": int(R),
        "seed": int(seed),
        "engine": str(engine),
        "roles": list(roles or []),
        "updated_iso": _utcnow_iso(),
    })


def _append_progress_row(path, row):
    import csv
    header = ["chunk", "iters_completed", "elapsed_s", "best_bic",
              "n_cache_evals", "timestamp_iso"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:  # noqa: BLE001 - non-critical on some filesystems
            pass


def _plot_search_progress(csv_path, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        df = pd.read_csv(csv_path)
        df = df[np.isfinite(pd.to_numeric(df["best_bic"], errors="coerce"))]
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(df["iters_completed"].to_numpy(),
                df["best_bic"].to_numpy(), "o-", color="#0B6E4F", lw=2)
        ax.set_xlabel("Search iterations completed")
        ax.set_ylabel("Best BIC so far (lower is better)")
        ax.set_title("Segment search convergence (recoverable checkpoints)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        tmp = os.path.join(os.path.dirname(os.path.abspath(png_path)),
                           ".tmp_progress.png")
        fig.savefig(tmp, dpi=150)
        plt.close(fig)
        os.replace(tmp, png_path)
    except Exception as e:  # noqa: BLE001 - progress plot must never kill a run
        print(f"   WARN: progress plot failed ({e})")


def _publish_dynamic_harmony_plot():
    """Publish the library's relative-path plot at a stable project path."""
    import glob
    import shutil

    roots = [RES, os.path.join(os.path.dirname(__file__), "results")]
    candidates = []
    for root in roots:
        candidates.extend(glob.glob(os.path.join(
            root, "DynamicHarmony_convergence_*.png")))
    if not candidates:
        return None
    source = max(set(candidates), key=os.path.getmtime)
    target = os.path.join(FIG, "segment_search47_dynamic_harmony.png")
    tmp = target + ".tmp"
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
        print(f"   Published DynamicHarmony plot -> {target}", flush=True)
        return target
    except Exception as e:  # noqa: BLE001 - publishing must not kill search
        if os.path.exists(tmp):
            os.unlink(tmp)
        print(f"   WARN: DynamicHarmony plot publish failed ({e})",
              flush=True)
        return None


def _load_compass_flow():
    """Optional Compass flow amounts. Returns (df_or_None, note).

    Never fabricated: absent/invalid file -> (None, reason) and the caller
    proceeds with exposure/flow proxies, recording the gap in the audit.
    """
    if not os.path.exists(COMPASS):
        return None, "absent (COMPASS flow not provided; using exposure proxies)"
    try:
        df = pd.read_csv(COMPASS)
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable ({e}); using exposure proxies"
    if "road_osmid" not in df.columns or "flow_veh" not in df.columns:
        return None, ("present but missing required columns "
                      "road_osmid/flow_veh; using exposure proxies")
    df = df.copy()
    df["road_osmid"] = pd.to_numeric(df["road_osmid"], errors="coerce")
    df["flow_veh"] = pd.to_numeric(df["flow_veh"], errors="coerce")
    df = df.dropna(subset=["road_osmid", "flow_veh"])
    df = df[df["flow_veh"] >= 0]
    if df.empty:
        return None, "present but no valid rows; using exposure proxies"
    df["road_osmid"] = df["road_osmid"].astype(np.int64)
    for c in ("phase", "fixture"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    if "hbin" in df.columns:
        df["hbin"] = pd.to_numeric(df["hbin"], errors="coerce").astype("Int64")
    keep = ["road_osmid", "flow_veh"] + [c for c in ("fixture", "phase", "hbin")
                                         if c in df.columns]
    return df[keep].copy(), f"loaded ({len(df):,} rows)"


def _load_stadium_games(cal_labels):
    """Optional per-fixture stadium/game info. Returns (cap_map_or_None, note).

    Schema: fixture|label (calendar label), stadium_capacity (>0),
    stadium (optional). Missing -> capacities NaN; attendance still enters
    via log_att/fixture-type terms (never imputed as capacity).
    """
    if not os.path.exists(STADGAMES):
        return None, "absent (no stadium capacities; attendance-only game effects)"
    try:
        df = pd.read_csv(STADGAMES)
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable ({e}); attendance-only game effects"
    lab = "label" if "label" in df.columns else ("fixture" if "fixture" in df.columns else None)
    if lab is None or "stadium_capacity" not in df.columns:
        return None, ("present but missing required columns "
                      "fixture|label/stadium_capacity; attendance-only game effects")
    df = df.copy()
    df[lab] = df[lab].fillna("").astype(str)
    df["stadium_capacity"] = pd.to_numeric(df["stadium_capacity"], errors="coerce")
    df = df[(df[lab] != "") & np.isfinite(df["stadium_capacity"])
            & (df["stadium_capacity"] > 0)]
    if df.empty:
        return None, "present but no valid rows; attendance-only game effects"
    cap = dict(zip(df[lab], df["stadium_capacity"].astype(float)))
    n_hit = sum(1 for lab_ in cal_labels if lab_ in cap)
    return cap, (f"loaded ({len(cap)} fixtures, {n_hit} matched to calendar)")


# Candidate segment characteristics searched by metacountregressor.
# OSM/geometry block + Overture block (ov_*; Polaris-style build in script 25:
# width_rules, directional speed_limits, surface/flags, class/subclass,
# routes, connectors) + optional POI counts (segment_poi_counts.csv).
# width_m (=lanes x 3.5 exactly) and road_class_rank (collinear with highway
# dummies) are EXCLUDED by construction: the evaluator flags them as
# near-perfectly collinear and they only inflate the search dimension.
CONT_VARS = ["lanes", "maxspeed_kmh", "curvature_deg_km",
             "sinuosity", "length_m", "node_degree",
             "dist_to_intersection_m", "dist_stadium_km",
             "ov_width_m", "ov_speed_fwd", "ov_speed_bwd", "ov_class_rank",
             "n_ov_connectors"]
COUNT_VARS = ["has_name", "has_ref", "oneway", "is_link", "lanes_missing",
              "multilane", "dead_end", "hi_speed", "lo_speed",
              "paved", "unpaved", "ov_is_link", "is_tunnel", "is_bridge",
              "is_toll", "has_ov_ref", "ov_mid_connector"]

# Human-readable labels for every searchable variable (audit + paper tables).
# Standardised (_z) columns inherit the base label; hw_* are highway-class
# dummies against the trunk reference; poi_* are per-segment POI counts.
VAR_LABELS = {
    "logexp": "log exposure (telematics trips)",
    "lanes": "lane count",
    "maxspeed_kmh": "posted speed limit (km/h)",
    "curvature_deg_km": "curvature (deg/km)",
    "sinuosity": "sinuosity",
    "length_m": "segment length (m)",
    "node_degree": "node degree",
    "dist_to_intersection_m": "distance to intersection (m)",
    "dist_stadium_km": "distance to stadium (km)",
    "ov_width_m": "carriageway width, Overture (m)",
    "ov_speed_fwd": "speed limit forward, Overture (km/h)",
    "ov_speed_bwd": "speed limit backward, Overture (km/h)",
    "ov_class_rank": "road class rank, Overture",
    "n_ov_connectors": "connector count, Overture",
    "has_name": "named road (vs unnamed)",
    "has_ref": "numbered route shield",
    "oneway": "one-way",
    "is_link": "ramp/link segment",
    "lanes_missing": "lane count missing in OSM",
    "multilane": "3+ lanes",
    "dead_end": "dead-end proxy",
    "hi_speed": "high speed band (>=60 km/h)",
    "lo_speed": "low speed band (<=40 km/h)",
    "paved": "paved surface, Overture",
    "unpaved": "unpaved surface, Overture",
    "ov_is_link": "link flag, Overture",
    "is_tunnel": "tunnel, Overture",
    "is_bridge": "bridge, Overture",
    "is_toll": "toll, Overture",
    "has_ov_ref": "route shield, Overture",
    "ov_mid_connector": "mid-segment connector, Overture",
    "flow_seg_veh": "flow per segment, Compass (veh)",
}


def var_label(col):
    """Plain-English label for a model column (strips _z, hw_ prefix, poi_)."""
    c = col
    if c.endswith("_z"):
        c = c[:-2] + " (std)"
        base = col[:-2]
    else:
        base = col
    if base.startswith("hw_"):
        return f"highway class: {base[3:].replace('_', ' ')} (vs trunk)"
    if base.startswith("poi_") or base.startswith("n_poi") or base.startswith("n_food") or base.startswith("n_shop"):
        return f"POI count: {base.replace('_', ' ')}"
    lab = VAR_LABELS.get(base, base.replace("_", " "))
    return lab + (" [standardised]" if col.endswith("_z") else "")


def _run_phase_copulas(*, tr_ev, base, cal, train_lab, compass_df,
                       max_rows, seed):
    """Pairwise phase-dependence tests via metacountregressor copulas.

    Panel: one row per (segment, train fixture) with finable counts in
    pre/during/post (Y_pre/Y_during/Y_post) plus segment, game/stadium and
    optional Compass-flow covariates. For each phase pair we fit the
    univariate-independence baseline, Famoye and Marshall-Olkin joint
    models, and Frank / Normal / Kimeldorf-Samson bivariate-NB copulas via
    metacountregressor.bivariate_copula.compare_bivariate_copulas, and
    select by BIC among converged fits (LR-vs-independence p reported).
    Metacount is used here for hypothesis testing and dependence-structure
    selection only -- never for ranking weights.

    Returns (compare_df, audit_dict, phase_long_df) where phase_long_df has
    per (segment, fixture) copula-marginal mean occurrences for pre/during/
    post (during = mean of its two pairwise marginals, flagged in audit).
    """
    from metacountregressor.bivariate_copula import compare_bivariate_copulas
    rng = np.random.default_rng(int(seed))
    phases = ["pre", "during", "post"]
    ev = tr_ev[tr_ev["phase"].isin(phases)][["road_osmid", "fixture",
                                             "phase"]].copy()
    ev["road_osmid"] = pd.to_numeric(ev["road_osmid"],
                                     errors="coerce").astype("Int64")
    ev = ev.dropna(subset=["road_osmid"])
    ev["road_osmid"] = ev["road_osmid"].astype(np.int64)
    cnt = (ev.groupby(["road_osmid", "fixture", "phase"]).size()
             .unstack(fill_value=0))
    for p in phases:
        if p not in cnt.columns:
            cnt[p] = 0
    cnt = cnt.reset_index().rename(
        columns={"pre": "Y_pre", "during": "Y_during", "post": "Y_post"})
    segs = base[["road_osmid"]].drop_duplicates().copy()
    segs["road_osmid"] = pd.to_numeric(segs["road_osmid"],
                                       errors="coerce").astype("Int64")
    segs = segs.dropna().astype({"road_osmid": np.int64})
    fixs = sorted(set(train_lab) & set(cnt["fixture"].unique().tolist()))
    if not fixs:
        raise RuntimeError("copula panel: no train fixtures with event rows")
    panel = (segs.assign(_k=1).merge(
        pd.DataFrame({"fixture": fixs, "_k": 1}), on="_k").drop(columns="_k"))
    panel = panel.merge(cnt, on=["road_osmid", "fixture"], how="left")
    for c in ("Y_pre", "Y_during", "Y_post"):
        panel[c] = panel[c].fillna(0).astype(float)
    # segment covariates
    seg_cov = base[["road_osmid", "exposure_trips"]].copy()
    seg_cov["exposure_trips"] = pd.to_numeric(
        seg_cov["exposure_trips"], errors="coerce").fillna(0.0)
    if "dist_stadium_km" in base.columns:
        seg_cov["dist_stadium_km"] = pd.to_numeric(
            base["dist_stadium_km"], errors="coerce")
        med = float(seg_cov["dist_stadium_km"].median())
        seg_cov["dist_stadium_km"] = seg_cov["dist_stadium_km"].fillna(
            med if np.isfinite(med) else 0.0)
    else:
        seg_cov["dist_stadium_km"] = 0.0
    panel = panel.merge(seg_cov.drop_duplicates("road_osmid"),
                        on="road_osmid", how="left")
    # game/stadium covariates (generalised: attendance, capacity use,
    # kickoff/day-of-week cyclic terms, fixture type)
    g = cal[cal["label"].isin(fixs)].copy()
    g["dow_sin"] = np.sin(2 * np.pi * g["dow"].astype(float) / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * g["dow"].astype(float) / 7.0)
    gcols = ["label", "log_att", "dow_sin", "dow_cos", "ko_sin", "ko_cos",
             "weekend", "fixture_type"]
    for c in ("utilization", "stadium_capacity"):
        if c in g.columns:
            gcols.append(c)
    panel = panel.merge(g[gcols], left_on="fixture", right_on="label",
                        how="left").drop(columns=["label"])
    panel["utilization"] = pd.to_numeric(
        panel.get("utilization", np.nan), errors="coerce")
    if not np.isfinite(panel["utilization"]).any():
        panel["utilization"] = panel["log_att"] / max(
            float(panel["log_att"].max()), 1e-9)
        util_note = ("no stadium capacities: utilisation proxied by "
                     "log_att/max(log_att); capacity file recommended")
    else:
        util_note = "utilisation = attendance/stadium_capacity"
    td = pd.get_dummies(panel["fixture_type"].fillna(""),
                        prefix="type", dtype=float)
    for c in list(td.columns):
        panel[c] = td[c].values
    type_cols = sorted([c for c in td.columns])[:2]  # avoid intercept collinearity
    # Compass flow amounts (optional conditioning info, never a weight)
    flow_note = "compass absent"
    if compass_df is not None and len(compass_df):
        cf = compass_df.copy()
        if "fixture" in cf.columns:
            cf = cf[cf["fixture"].isin(fixs)]
            fl = cf.groupby(["road_osmid", "fixture"],
                            as_index=False)["flow_veh"].mean()
            panel = panel.merge(fl.rename(
                columns={"flow_veh": "flow_compass"}),
                on=["road_osmid", "fixture"], how="left")
        else:
            fl = cf.groupby("road_osmid", as_index=False)["flow_veh"].mean()
            panel = panel.merge(fl.rename(
                columns={"flow_veh": "flow_compass"}),
                on="road_osmid", how="left")
        seg_mean = panel.groupby("road_osmid")["flow_compass"].transform("mean")
        panel["flow_compass"] = panel["flow_compass"].fillna(seg_mean)
        flow_note = "compass flow_veh joined (segment-fixture mean where keyed)"
    # standardise continuous covariates (stats recorded for prediction reuse)
    std_stats = {}
    for c in ("dist_stadium_km", "log_att", "utilization"):
        v = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)
        mu, sd = float(v.mean()), float(v.std())
        panel[c + "_z"] = (v - mu) / (sd if sd and sd > 0 else 1.0)
        std_stats[c] = {"mean": mu, "sd": sd if sd and sd > 0 else 1.0}
    feat = ["dist_stadium_km_z", "log_att_z", "utilization_z", "dow_sin",
            "dow_cos", "ko_sin", "ko_cos", "weekend"] + type_cols
    if "flow_compass" in panel.columns and np.isfinite(
            pd.to_numeric(panel["flow_compass"],
                          errors="coerce")).any():
        fv = pd.to_numeric(panel["flow_compass"], errors="coerce").fillna(0.0)
        mu, sd = float(fv.mean()), float(fv.std())
        panel["flow_compass_z"] = (fv - mu) / (sd if sd and sd > 0 else 1.0)
        std_stats["flow_compass"] = {"mean": mu,
                                     "sd": sd if sd and sd > 0 else 1.0}
        feat.append("flow_compass_z")
        flow_note += "; flow_compass_z in copula design"
    else:
        flow_note += "; no usable compass values in design"
    # deterministic per-fixture cap (keeps fits tractable, all games kept)
    fit_panel = panel
    if len(panel) > int(max_rows):
        per = max(1, int(int(max_rows) // max(len(fixs), 1)))
        parts = []
        for fx, gp in panel.groupby("fixture"):
            if len(gp) > per:
                gp = gp.sample(n=per, random_state=int(seed))
            parts.append(gp)
        fit_panel = pd.concat(parts, ignore_index=True)
    Xn = ["intercept"] + feat
    Xf = np.column_stack(
        [np.ones(len(fit_panel))] +
        [pd.to_numeric(fit_panel[c], errors="coerce").fillna(0.0).to_numpy()
         for c in feat])
    off = np.log(pd.to_numeric(fit_panel["exposure_trips"],
                               errors="coerce").fillna(0.0).to_numpy() + 1.0)
    Xp = np.column_stack(
        [np.ones(len(panel))] +
        [pd.to_numeric(panel[c], errors="coerce").fillna(0.0).to_numpy()
         for c in feat])
    offp = np.log(pd.to_numeric(panel["exposure_trips"],
                                errors="coerce").fillna(0.0).to_numpy() + 1.0)
    pairs = [("pre", "during"), ("during", "post"), ("pre", "post")]
    cmp_rows, audit_pairs, mean_frames = [], {}, {}
    for a, b in pairs:
        y1 = fit_panel[f"Y_{a}"].to_numpy(dtype=float)
        y2 = fit_panel[f"Y_{b}"].to_numpy(dtype=float)
        try:
            cdf, fits = compare_bivariate_copulas(
                y1, y2, Xf, Xf, off, off,
                feature_names_1=Xn, feature_names_2=Xn)
        except Exception as e:  # noqa: BLE001 - one pair must not kill the run
            audit_pairs[f"{a}/{b}"] = {"ran": False,
                                       "error": str(e)[:300]}
            continue
        cdf = cdf.copy()
        cdf.insert(0, "pair", f"{a}/{b}")
        conv = {}
        for key, fit in (fits or {}).items():
            if hasattr(fit, "converged"):
                conv[str(key)] = bool(fit.converged)
            elif isinstance(fit, dict):
                conv[str(key)] = bool(fit.get("converged", True))
            else:
                conv[str(key)] = True
        # map convergence onto rows by model label
        def _row_conv(model_label):
            ml = str(model_label).lower()
            if "univariate" in ml or "baseline" in ml or "sum of" in ml:
                return conv.get("univar_sum", True)
            if "marshall" in ml:
                return conv.get("marshall_olkin", True)
            if "famoye" in ml:
                return conv.get("famoye", True)
            for k, v in conv.items():
                if k.lower() in ml:
                    return v
            return True
        cdf["converged"] = [bool(_row_conv(m)) for m in cdf["model"]]
        cdf["bic_finite"] = pd.to_numeric(cdf["BIC"], errors="coerce")
        ok = cdf[cdf["converged"] & np.isfinite(cdf["bic_finite"])]
        best_model, best_bic = None, float("nan")
        if len(ok):
            br = ok.loc[ok["bic_finite"].idxmin()]
            best_model, best_bic = str(br["model"]), float(br["bic_finite"])
        cdf["best_flag"] = [(str(m) == best_model) for m in cdf["model"]]
        cmp_rows.append(cdf)
        # marginal-mean occurrences from the winning copula fit when it
        # carries coefficients (else lowest-BIC copula with coefficients)
        means, mean_src = {}, None
        order = ([best_model] if best_model else []) + [
            r["model"] for _, r in
            cdf.sort_values("bic_finite").iterrows()]
        for ml in order:
            if ml is None:
                continue
            key = None
            for k in (fits or {}):
                if str(k).lower() in str(ml).lower() or str(ml).lower() in str(k).lower():
                    key = k
                    break
            fit = (fits or {}).get(key)
            if fit is None or not hasattr(fit, "coef_1"):
                continue
            try:
                m1 = np.exp(Xp @ np.asarray(fit.coef_1, dtype=float) + offp)
                m2 = np.exp(Xp @ np.asarray(fit.coef_2, dtype=float) + offp)
            except Exception:  # noqa: BLE001
                continue
            means = {"m1": np.maximum(m1, 0.0), "m2": np.maximum(m2, 0.0)}
            mean_src = str(ml)
            break
        mean_frames[(a, b)] = means
        audit_pairs[f"{a}/{b}"] = {
            "ran": True, "n_rows": int(len(fit_panel)),
            "best_model": best_model,
            "best_bic": (None if not np.isfinite(best_bic) else float(best_bic)),
            "mean_source": mean_src,
            "models": cdf[["model", "n", "k", "loglik", "AIC", "BIC", "rho",
                           "LR_independence", "p_independence", "converged",
                           "best_flag"]].to_dict(orient="records"),
        }
    compare_df = (pd.concat(cmp_rows, ignore_index=True) if cmp_rows
                  else pd.DataFrame(columns=["pair", "model", "n", "k",
                                             "loglik", "AIC", "BIC", "rho",
                                             "LR_independence",
                                             "p_independence", "converged",
                                             "best_flag"]))
    # long-format per (segment, fixture) occurrence means for pre/during/post
    base_long = panel[["road_osmid", "fixture"]].copy()
    for p, key in (("pre", (("pre", "during"), "m1")),
                   ("post", (("during", "post"), "m2"))):
        arr = mean_frames.get(key[0], {}).get(key[1])
        base_long[f"h_cop_{p}"] = (np.asarray(arr, dtype=float)
                                   if arr is not None else np.nan)
    d1 = mean_frames.get(("pre", "during"), {}).get("m2")
    d2 = mean_frames.get(("during", "post"), {}).get("m1")
    if d1 is not None and d2 is not None:
        base_long["h_cop_during"] = (np.asarray(d1) + np.asarray(d2)) / 2.0
        during_note = "mean of the two during-marginals"
    elif d1 is not None:
        base_long["h_cop_during"] = np.asarray(d1)
        during_note = "pre/during marginal only"
    elif d2 is not None:
        base_long["h_cop_during"] = np.asarray(d2)
        during_note = "during/post marginal only"
    else:
        base_long["h_cop_during"] = np.nan
        during_note = "no converged copula marginals"
    audit = {"ran": True, "pairs": audit_pairs,
             "design_features": Xn,
             "standardisation": std_stats,
             "utilization_note": util_note, "flow_note": flow_note,
             "n_panel_rows": int(len(panel)),
             "n_fit_rows": int(len(fit_panel)),
             "n_fixtures": int(len(fixs)),
             "during_note": during_note}
    return compare_df, audit, base_long


def build_segment_features(screen, feats, meta, geo_props=None, poi=None):
    """Merge all segment characteristics + countable OSM/POI features.

    screen: near_miss_hb_ecf (road_osmid, highway, exposure_trips, ...)
    feats:  road_geometry_features (road_osmid, length_m, curvature, ...)
    meta:   road_meta (osmid, name, highway, dist_stadium_km, ...)
    geo_props: optional DataFrame indexed by osmid with OSM tag columns
               (name, ref, oneway, lanes raw) for countable flags.
    poi:    optional DataFrame with road_osmid + count columns (>=0).
    Returns (df, cont_present, count_present, poi_cols).
    """
    df = screen.copy()
    f2 = feats.drop(columns=["highway"], errors="ignore")
    df = df.merge(f2, on="road_osmid", how="left", suffixes=("", "_feat"))
    # Prefer the Polaris-style feature build (script 25: KNN-imputed lanes /
    # speeds, Overture-backed fields) over stale screening-table copies.
    for c in list(f2.columns):
        if c == "road_osmid":
            continue
        fc = c + "_feat"
        if fc in df.columns:
            df[c] = df[fc].combine_first(df[c])
            df = df.drop(columns=[fc])
    m2 = meta.rename(columns={"osmid": "road_osmid"})
    keep = ["road_osmid", "name", "dist_stadium_km"]
    keep = [c for c in keep if c in m2.columns]
    # coalesce: screening table may already carry these columns (stale copies
    # would otherwise become name_x/name_y and vanish from candidates).
    for c in keep[1:]:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.merge(m2[keep].drop_duplicates("road_osmid"), on="road_osmid",
                  how="left")
    if geo_props is not None:
        gp = geo_props.copy()
        gp["road_osmid"] = pd.to_numeric(gp["road_osmid"], errors="coerce")
        df = df.merge(gp.drop_duplicates("road_osmid"), on="road_osmid",
                      how="left", suffixes=("", "_osm"))
    # --- continuous road characteristics ----------------------------------
    lanes_raw = pd.to_numeric(df.get("lanes"), errors="coerce")
    df["lanes_missing"] = lanes_raw.isna().astype(float)
    df["lanes"] = lanes_raw.fillna(float(lanes_raw.median()))
    df["width_m"] = df["lanes"] * 3.5  # standard lane-width proxy
    for c in ["maxspeed_kmh", "curvature_deg_km", "sinuosity", "length_m",
              "road_class_rank", "node_degree", "dist_to_intersection_m",
              "dist_stadium_km", "ov_width_m", "ov_speed_fwd", "ov_speed_bwd",
              "ov_class_rank", "n_ov_connectors"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = float(df[c].median())
            df[c] = df[c].fillna(med if np.isfinite(med) else 0.0)
    # --- countable OSM features --------------------------------------------
    nm = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    df["has_name"] = (nm.str.strip() != "").astype(float)
    if "ref" in df.columns:
        rf = df["ref"].fillna("").astype(str)
    else:
        rf = pd.Series("", index=df.index)
    df["has_ref"] = (rf.str.strip() != "").astype(float)
    for c in ["oneway", "is_link"]:
        if c in df.columns:
            v = df[c].astype(str).str.lower().str.strip()
            df[c] = v.isin(["1", "1.0", "yes", "true", "y"]).astype(float)
        else:
            df[c] = 0.0
    df["multilane"] = (df["lanes"] >= 3).astype(float)
    df["hi_speed"] = (df.get("maxspeed_kmh", 50) >= 60).astype(float)
    df["lo_speed"] = (df.get("maxspeed_kmh", 50) <= 40).astype(float)
    nd = pd.to_numeric(df.get("node_degree"), errors="coerce").fillna(3)
    di = pd.to_numeric(df.get("dist_to_intersection_m"),
                       errors="coerce").fillna(0)
    df["dead_end"] = ((nd <= 2) & (di > 200)).astype(float)
    # --- optional POI counts (pluggable, never fabricated) -------------------
    poi_cols = []
    if poi is not None and len(poi):
        pc = poi.copy()
        pc["road_osmid"] = pd.to_numeric(pc["road_osmid"], errors="coerce")
        pc = pc.dropna(subset=["road_osmid"])
        for c in pc.columns:
            if c == "road_osmid":
                continue
            pc[c] = pd.to_numeric(pc[c], errors="coerce").fillna(0.0).clip(lower=0)
            poi_cols.append(c)
        df = df.merge(pc.drop_duplicates("road_osmid"), on="road_osmid",
                      how="left")
        for c in poi_cols:
            df[c] = df[c].fillna(0.0)
    for c in CONT_VARS + COUNT_VARS + poi_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            df[c] = df[c].replace([np.inf, -np.inf], 0.0)
    cont = [c for c in CONT_VARS if c in df.columns]
    cnt = [c for c in COUNT_VARS if c in df.columns]
    return df, cont, cnt, poi_cols


def standardise_cols(df, cols):
    for c in cols:
        z = c + "_z"
        if c in df.columns and z not in df.columns:
            sd = float(df[c].std())
            df[z] = ((df[c] - float(df[c].mean())) /
                     (sd if sd and sd > 0 else 1.0))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=MARGIN_DEFAULT)
    ap.add_argument("--max-exceedance", type=float,
                    default=MAX_EXCEEDANCE_DEFAULT,
                    help="upper bound on km/h over posted limit; higher "
                         "values are excluded and audited")
    ap.add_argument("--n-holdout", type=int, default=9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-search", action="store_true",
                    help="skip metacountregressor structure search; use manual P1 spec")
    ap.add_argument("--search-algo", type=str, default="agds",
                    help="search algorithm: agds (SparseEA-AGDS), sa, de, or hs")
    ap.add_argument("--search-iters", type=int, default=150)
    ap.add_argument("--search-chunk-iters", type=int, default=30,
                    help="search iterations per recoverable chunk; each chunk "
                         "checkpoints the best spec so a walltime kill still "
                         "leaves a --resume-from checkpoint + progress plot; "
                         "HS/DE need at least one population (30) per chunk")
    ap.add_argument("--search-R", type=int, default=100)
    ap.add_argument("--refit-R", type=int, default=300)
    ap.add_argument("--asof-iso", type=str, default="",
                    help="operational as-of timestamp (ISO); default = now UTC. "
                         "Recorded on hourly predictions with hours_to_kickoff.")
    ap.add_argument("--no-copula", action="store_true",
                    help="skip the metacountregressor phase-pair copula "
                         "hypothesis tests (Stage C2)")
    ap.add_argument("--copula-max-rows", type=int, default=120000,
                    help="cap on segment-fixture rows for copula fits "
                         "(deterministic per-fixture subsample beyond this)")
    ap.add_argument("--search-jobs", type=int, default=4,
                    help="parallel fitness evaluations for the search "
                         "(4 = memory-safe on 64GB nodes)")
    ap.add_argument("--trace-fitness", action="store_true",
                    help="print every evaluator fitness start/result; useful "
                         "for local smoke tests and diagnosing stalled searches")
    ap.add_argument("--search-only", action="store_true",
                    help="run Stage A search and save best spec, then exit")
    ap.add_argument("--resume-from", type=str, default="",
                    help="best-spec JSON from a prior --search-only run; "
                         "skips the search and refits directly")
    ap.add_argument("--engine", type=str, default=ENGINE_DEFAULT,
                    help="count search engine: jax (full model) or numba "
                        "(fixed-effects local mode)")
    ap.add_argument("--no-panel", action="store_true",
                    help="skip the Stage B* metacount game panel; "
                         "use the statsmodels crowd scale only")
    ap.add_argument("--panel-R", type=int, default=100,
                    help="Halton draws for the Stage B* panel fit")
    args = ap.parse_args()
    _validate_mcr_version(args.engine)
    M, X = float(args.margin), float(args.max_exceedance)
    if X < M:
        raise SystemExit("FATAL: --max-exceedance must be >= --margin")
    rng = np.random.default_rng(int(args.seed))
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    for _sig in ("SIGTERM", "SIGINT"):
        try:
            signal.signal(getattr(signal, _sig), _handle_stop)
        except Exception:  # noqa: BLE001 - platform may lack the signal
            pass
    audit = {"margin_kph": M, "max_exceedance_kph": X,
             "metacountregressor": _MCR_VERSION,
             "engine_requested": str(args.engine),
             "estimand": ("bounded finable speeding counts (margin <= "
                          "exceedance <= max_exceedance); no risk weighting")}
    print("=" * 80)
    print(f"47_segment_speeding_predictive.py - bounded pure speeding counts "
          f"({M:.0f}-{X:.0f} kph over limit)")
    print(f"   metacountregressor={_MCR_VERSION} engine={args.engine}")
    print("=" * 80)

    for f in (EXCEED, CAL):
        if not os.path.exists(f):
            raise SystemExit(f"FATAL: missing input {f} (run 43_finable_speeding.py first)")

    # ---- Fixtures: attendance + DOW + kickoff + type -------------------------
    cal = pd.read_csv(CAL)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"], errors="coerce")
    cal = cal.dropna(subset=["ko"]).copy()
    cal["attendance"] = pd.to_numeric(cal["attendance"], errors="coerce")
    cal_med = float(cal["attendance"].median())
    cal["attendance"] = cal["attendance"].fillna(cal_med)
    audit["attendance_imputed_median"] = cal_med
    cal["dow"] = cal["ko"].dt.dayofweek.astype(int)
    cal["weekend"] = (cal["dow"] >= 5).astype(int)
    cal["ko_h"] = cal["ko"].dt.hour + cal["ko"].dt.minute / 60.0
    cal["ko_sin"] = np.sin(2 * np.pi * cal["ko_h"] / 24.0)
    cal["ko_cos"] = np.cos(2 * np.pi * cal["ko_h"] / 24.0)
    cal["log_att"] = np.log(cal["attendance"] + 1.0)
    cal["fixture_type"] = cal.get("fixture_type", "NRL").fillna("NRL").astype(str)
    print(f"   fixtures={len(cal)} attendance {cal['attendance'].min():,.0f}-"
          f"{cal['attendance'].max():,.0f} types={sorted(cal['fixture_type'].unique())}")
    # ---- Operational game context (optional files; never fabricated) --------
    asof_iso = str(args.asof_iso).strip() or _utcnow_iso()
    cap_map, cap_note = _load_stadium_games(set(cal["label"]))
    cal["stadium_capacity"] = cal["label"].map(
        cap_map if cap_map else {}).astype(float)
    cal["utilization"] = np.where(
        np.isfinite(cal["stadium_capacity"]) & (cal["stadium_capacity"] > 0),
        cal["attendance"] / cal["stadium_capacity"], np.nan)
    cal["asof_iso"] = asof_iso
    # hours_to_kickoff is computed per prediction row below from kickoff_iso
    # + asof_iso strings (parsed once, tz-naive Brisbane local).
    audit["asof_iso"] = asof_iso
    audit["stadium_games"] = cap_note
    compass_df, compass_note = _load_compass_flow()
    audit["compass_flow"] = compass_note
    print(f"   game context: asof={asof_iso}; stadium: {cap_note}; "
          f"compass: {compass_note}")

    # ---- Stratified game split: most games train, rest holdout ---------------
    hold_idx = []
    for t, g in cal.groupby("fixture_type"):
        g = g.sort_values("attendance")
        n = len(g)
        nh = max(1, int(round(n * args.n_holdout / max(len(cal), 1))))
        # spread across crowd gradient: smallest, median-ish, largest
        picks = sorted(set([0, n // 2, n - 1]))[:nh]
        hold_idx += g.iloc[picks].index.tolist()
    hold_idx = sorted(set(hold_idx))
    # top-up / trim to requested size deterministically
    if len(hold_idx) < args.n_holdout:
        rest = [i for i in cal.index if i not in hold_idx]
        extra = rng.choice(rest, size=args.n_holdout - len(hold_idx), replace=False)
        hold_idx = sorted(hold_idx + extra.tolist())
    hold_idx = hold_idx[:args.n_holdout]
    train_idx = [i for i in cal.index if i not in hold_idx]
    train_lab = set(cal.loc[train_idx, "label"])
    hold_lab = set(cal.loc[hold_idx, "label"])
    audit["train_fixtures"] = sorted(train_lab)
    audit["holdout_fixtures"] = sorted(hold_lab)
    n_tr_fx = int(cal.loc[train_idx].shape[0])
    n_ho_fx = int(cal.loc[hold_idx].shape[0])
    audit["n_train_fixtures"] = n_tr_fx
    audit["n_holdout_fixtures"] = n_ho_fx
    print(f"   train={len(train_lab)} labels ({n_tr_fx} fixtures) "
          f"holdout={len(hold_lab)} labels ({n_ho_fx} fixtures)")
    print(f"   holdout: {sorted(hold_lab)}")

    # ---- Per-event exceedances -> segment x fixture x phase x hour -----------
    print("\n[1/6] Loading finable exceedances...")
    ex = pd.read_csv(EXCEED)
    ex["road_osmid"] = pd.to_numeric(ex["road_osmid"], errors="coerce")
    ex = ex.dropna(subset=["road_osmid"]).copy()
    ex["road_osmid"] = ex["road_osmid"].astype(np.int64)
    ex["dt"] = pd.to_datetime(ex["dt"], errors="coerce")
    ex = ex.dropna(subset=["dt"])
    if "exceedance" not in ex.columns:
        raise SystemExit("FATAL: finable exceedance input lacks the "
                         "exceedance column; cannot enforce upper bound")
    ex["exceedance"] = pd.to_numeric(ex["exceedance"], errors="coerce")
    ex = ex.dropna(subset=["exceedance"]).copy()
    high_tail = ex[ex["exceedance"] > X].copy()
    if len(high_tail):
        high_tail["reason"] = "exceedance_above_max"
        high_tail.to_csv(EXCEED_OUTLIERS, index=False)
    elif not os.path.exists(EXCEED_OUTLIERS):
        pd.DataFrame(columns=list(ex.columns) + ["reason"]).to_csv(
            EXCEED_OUTLIERS, index=False)
    tail_count = int(len(high_tail))
    if not tail_count and os.path.exists(EXCEED_OUTLIERS):
        try:
            tail_count = int(len(pd.read_csv(EXCEED_OUTLIERS,
                                             usecols=["exceedance"])))
        except Exception:  # noqa: BLE001 - QA artifact must not block modeling
            tail_count = 0
    audit["n_exceedance_above_max"] = tail_count
    audit["share_exceedance_above_max"] = float(
        tail_count / max(len(ex), 1))
    audit["share_exceedance_above_max_of_exceedances"] = float(
        tail_count / max(len(ex), 1))
    ex = ex[(ex["exceedance"] >= M) & (ex["exceedance"] <= X)].copy()
    # ALL hours, not just fixture windows: outside a window the stadium
    # draws no crowd (attendance = 0) and speeding follows the background
    # day-of-week x clock-hour surface. Event rows carry a fixture label +
    # phase; everything else is background with attendance 0.
    ex["phase"] = ex.get("phase", "").fillna("").astype(str)
    ex["fixture"] = ex.get("fixture", "").fillna("").astype(str)
    in_window = (ex["fixture"].isin(set(cal["label"]))
                 & ex["phase"].isin(["pre", "during", "post"]))
    ex["window"] = np.where(in_window, "event", "background")
    ko_map = dict(zip(cal["label"], cal["ko"]))
    ex["ko"] = ex["fixture"].map(ko_map)
    ex["rel_h"] = ((ex["dt"] - ex["ko"]).dt.total_seconds() / 3600.0).astype(float)
    # event-relative hourly bin (background rows get hbin=-99: attendance 0,
    # scored on the day-of-week x clock-hour surface instead)
    ex["hbin"] = -99
    mev = ex["window"] == "event"
    ex.loc[mev, "hbin"] = (np.floor(ex.loc[mev, "rel_h"].clip(-3, 3.999))
                           .astype(int).clip(-3, 3))
    ex["dow"] = ex["dt"].dt.dayofweek.astype(int)
    ex["hour"] = ex["dt"].dt.hour.astype(int)
    # holdout = holdout fixtures (event rows) + background hours in the
    # holdout-game weeks (+-3 days around each holdout kickoff): unseen hours.
    hold_ko = pd.to_datetime(cal.loc[cal["label"].isin(hold_lab), "ko"])
    hold_days = set()
    for k in hold_ko:
        for d in range(-3, 4):
            hold_days.add((k + pd.Timedelta(days=d)).date())
    ex["in_hold_week"] = ex["dt"].dt.date.isin(hold_days)
    ex["split"] = np.where(
        ((ex["window"] == "event") & ex["fixture"].isin(hold_lab))
        | ((ex["window"] == "background") & ex["in_hold_week"]),
        "holdout", "train")
    nev = int((ex["window"] == "event").sum())
    nbg = int((ex["window"] == "background").sum())
    print(f"   exceedances: event-window {nev:,} + background {nbg:,} "
          f"(attendance=0 outside windows)")
    print(f"   train {int((ex['split']=='train').sum()):,}, "
          f"holdout {int((ex['split']=='holdout').sum()):,}")
    audit["n_exceed_event"] = nev
    audit["n_exceed_background"] = nbg
    audit["n_exceed_train"] = int((ex["split"] == "train").sum())
    audit["n_exceed_holdout"] = int((ex["split"] == "holdout").sum())

    # ---- Stage A: SEARCHED segment baseline NB-RP (train, metacount) --------
    print("\n[2/6] Stage A: metacountregressor SEARCH over segment "
          "characteristics + countable OSM/POI features...")
    tr = ex[ex["split"] == "train"]
    tr_ev = tr[tr["window"] == "event"].copy()
    tr_bg = tr[tr["window"] == "background"].copy()
    print(f"   train rows: event {len(tr_ev):,} + background {len(tr_bg):,}")
    # Baseline uses ALL train-period exceedances (event + background,
    # attendance>0 and attendance=0 alike): ~40k counts, far more stable
    # than the ~1k in-window events alone.
    agg = tr.groupby("road_osmid").size().reset_index(name="Y")
    sc = pd.read_csv(SCREEN)
    feats = pd.read_csv(FEATS) if os.path.exists(FEATS) else pd.DataFrame()
    meta = pd.read_csv(META) if os.path.exists(META) else pd.DataFrame()
    poi = (pd.read_csv(POI) if os.path.exists(POI) else None)
    audit["poi_available"] = bool(poi is not None and len(poi))
    if poi is None:
        print("   no data/segment_poi_counts.csv: POI terms skipped "
              "(add road_osmid + count columns to include them)")
    base, cont, cnt, poi_cols = build_segment_features(sc, feats, meta,
                                                       poi=poi)
    # OSM tag countable flags (name/ref) straight from the geojson so
    # has_name/has_ref vary (they are constant without this join).
    try:
        import geopandas as gpd
        _g = gpd.read_file(GEOJSON)
        _gp = pd.DataFrame({"road_osmid": pd.to_numeric(_g["osmid"],
                                                        errors="coerce"),
                            "name_osm": _g.get("name", "").fillna("").astype(str),
                            "ref": _g.get("ref", "").fillna("").astype(str)
                            if "ref" in _g.columns else ""})
        _gp = _gp.dropna(subset=["road_osmid"]).drop_duplicates("road_osmid")
        base = base.merge(_gp, on="road_osmid", how="left")
        nm = base["name_osm"].fillna(base.get("name", "").fillna("").astype(str))
        base["has_name"] = (nm.fillna("").astype(str).str.strip() != "").astype(float)
        base["has_ref"] = (base["ref"].fillna("").astype(str).str.strip() != "").astype(float)
        base = base.drop(columns=["name_osm", "ref"], errors="ignore")
    except Exception as e:  # noqa: BLE001 - countable flags stay as built
        print(f"   WARN: geojson tag join failed ({e}); has_name/has_ref as built")
    # Flow per segment in the SEARCH: Compass segment means (veh) when the
    # file exists; otherwise the exposure-trips level already forced into the
    # model (logexp) carries segment flow. Never fabricated: absent file ->
    # candidate skipped and the gap recorded (see audit["flow_seg"]).
    if compass_df is not None and len(compass_df):
        _cf = compass_df.copy()
        _cf["road_osmid"] = pd.to_numeric(_cf["road_osmid"],
                                          errors="coerce").astype("Int64")
        _cf = _cf.dropna(subset=["road_osmid"])
        _cf["road_osmid"] = _cf["road_osmid"].astype(np.int64)
        _fm = _cf.groupby("road_osmid")["flow_veh"].mean().reset_index(
            ).rename(columns={"flow_veh": "flow_seg_veh"})
        base = base.merge(_fm, on="road_osmid", how="left")
        _med = float(base["flow_seg_veh"].median())
        base["flow_seg_veh"] = base["flow_seg_veh"].fillna(
            _med if np.isfinite(_med) else 0.0)
        if "flow_seg_veh" not in cont:
            cont.append("flow_seg_veh")
        audit["flow_seg"] = {"source": "compass_flow.csv",
                             "segments": int(base["flow_seg_veh"].notna().sum()),
                             "mean_veh": float(base["flow_seg_veh"].mean())}
        print(f"   flow/segment candidate: flow_seg_veh "
              f"(mean {audit['flow_seg']['mean_veh']:.0f} veh)")
    else:
        audit["flow_seg"] = {"source": "exposure_trips (logexp forced in); "
                                       "compass_flow.csv absent",
                             "segments": 0}
        print("   flow/segment candidate: exposure level via logexp "
              "(compass_flow.csv absent)")
    base = base.merge(agg, on="road_osmid", how="left")
    base["Y"] = base["Y"].fillna(0.0)
    base = base[base["exposure_trips"] > 0].copy()
    base = standardise_cols(base, cont + poi_cols)
    mc, _ = prepare_metacount_frame(base, exposure_col="exposure_trips",
                                    highway_col="highway")
    cand = [c + "_z" for c in cont if c + "_z" in mc.columns]
    cand += [c for c in cnt if c in mc.columns]
    cand += [c for c in poi_cols if c in mc.columns]
    hw = [c for c in mc.columns if c.startswith("hw_")]
    search_variables = ["logexp"] + hw + cand
    audit["search_candidates"] = {"continuous": cont, "countable": cnt,
                                  "poi": poi_cols}
    audit["overture_available"] = bool(
        any(base[c].notna().sum() > 0 for c in
            ["ov_width_m", "ov_speed_fwd", "paved", "n_ov_connectors"]
            if c in base.columns))
    audit["overture_coverage"] = {
        c: int(base[c].notna().sum()) for c in
        ["ov_width_m", "ov_speed_fwd", "ov_speed_bwd", "paved",
         "n_ov_connectors", "has_ov_ref"] if c in base.columns}
    print(f"   candidates: continuous={cont}")
    print(f"   candidates: countable={cnt} poi={poi_cols}")
    if args.no_search:
        fixed = ["logexp"] + [c for c in mc.columns if c.startswith("hw_")]
        for c in ["maxspeed_kmh_z", "curvature_deg_km_z"]:
            if c in mc.columns:
                fixed.append(c)
        rdm = [f"{c}:normal" for c in ["lanes_z", "dist_to_intersection_m_z"]
               if c in mc.columns]
        info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                                fixed_terms=fixed, rdm_terms=rdm,
                                model="nb", R=150, de_maxiter=8,
                                de_popsize=8, verbose=True)
        audit["stageA_search"] = {"ran": False, "reason": "--no-search"}
    else:
        from metacountregressor import ExperimentBuilder, ModelConstraints
        _resume_active = bool(args.resume_from and os.path.exists(args.resume_from))
        if args.resume_from and os.path.exists(args.resume_from):
            # resume: skip the search, refit the checkpointed best spec.
            print(f"   resuming from {args.resume_from} (no search)...")
            with open(args.resume_from) as f:
                saved = json.load(f)
            # Accept our checkpoint {"best_spec":...} OR a raw
            # SearchOutputConfig JSON {"result":{"model_spec":...}}.
            if "best_spec" not in saved and "result" in saved:
                _rs = saved.get("result", {})
                saved = {"best_spec": _rs.get("model_spec"),
                         "best_bic": _rs.get("best_score"),
                         "algo": saved.get("algorithm", "?"),
                         "iters": None, "R": None,
                         "seed": _rs.get("seed")}
            if not saved.get("best_spec"):
                raise SystemExit(
                    f"FATAL: resume file has no spec: {args.resume_from}")
            _saved_spec = saved["best_spec"]
            _resume_is_numba = (
                str(saved.get("engine", "")).lower() == "numba"
                or (isinstance(_saved_spec, str)
                    and _saved_spec.lstrip().startswith("["))
            )
            if _resume_is_numba:
                if isinstance(_saved_spec, dict):
                    _refit_spec = _saved_spec
                else:
                    _decision = np.fromstring(
                        str(_saved_spec).replace("[", " ").replace("]", " "),
                        sep=" ", dtype=int)
                    _refit_spec = _numba_spec_from_decision(
                        _decision, search_variables)
                info = _fit_numba_selected_spec(
                    mc, _refit_spec, int(args.refit_R))
                info["spec_label"] = (
                    f"resumed-numba-{saved.get('algo', '?')}:"
                    f"{saved.get('best_bic', float('nan')):.0f}")
                _resume_best_spec = _refit_spec
            else:
                _use_g, _gcounts = _inspect_groups(mc, col="highway", min_n=20)
                audit["group_inspection"] = _gcounts
                # If the checkpointed spec itself needs grouped terms, still pass
                # the group col so the refit can run (even when fresh search
                # would gate grouped OFF).
                _spec_txt = str(_saved_spec)
                _needs_g = ("group" in _spec_txt.lower()) or ("hetro" in _spec_txt.lower())
                if _use_g or (_needs_g and "highway" in mc.columns):
                    if not _use_g:
                        print("   resume: spec needs grouped but groups are rare - "
                              "passing group_id_col anyway for refit")
                    builder = ExperimentBuilder(mc, id_col="road_osmid",
                                                y_col="Y", group_id_col="highway")
                else:
                    builder = ExperimentBuilder(mc, id_col="road_osmid", y_col="Y")
                fit = builder.fit_manual_model(manual_spec=_saved_spec,
                                               model="nb", R=int(args.refit_R))
                from metacount_nb import _parse_summary_text as _parse
                parsed = _parse(fit.get("_summary_text", ""))
                dp = parsed.get("parte", {}).get("dispersion", np.nan)
                alpha = float(dp) if np.isfinite(dp) and 0 < dp < 1e4 else 1.0
                expected = np.maximum(
                    np.asarray(fit["predictions"], dtype=float), 1e-9)
                summ = fit.get("summary", {})
                info = {"expected": expected, "alpha": alpha,
                        "aic": float(summ.get("aic", np.nan)),
                        "bic": float(summ.get("bic", np.nan)),
                        "parsed": parsed,
                        "spec_label": (f"resumed-{saved.get('algo', '?')}:"
                                       f"{saved.get('best_bic', float('nan')):.0f}")}
                _resume_best_spec = _saved_spec
            _resume_best_score = saved.get("best_bic", float("nan"))
            audit["engine_used"] = "numba" if _resume_is_numba else "jax"
            audit["stageA_search"] = {
                "ran": False, "resumed_from": args.resume_from,
                "best_bic": saved.get("best_bic"),
                "best_spec": str(saved.get("best_spec"))}
            base["lambda_base"] = info["expected"]
            print(f"   AIC={info['aic']:.0f} alpha={info['alpha']:.3f} "
                  f"[{info['spec_label']}] n={len(base):,}")
            audit["stageA"] = {"aic": info["aic"], "bic": info["bic"],
                               "alpha": info["alpha"], "n_seg": len(base),
                               "spec": info["spec_label"]}
            base[["road_osmid", "lambda_base"]].to_csv(
                os.path.join(RES, "segment_speeding_pred_base.csv"), index=False)
        else:
            c = (ModelConstraints()
                 .force_include("logexp")
                 .no_random("logexp", *hw)
                 .no_zi("logexp", *cand, *hw)
                 .no_random(*hw, *cnt))
            allow_r = [x for x in cand
                       if x not in cnt and not x.startswith("hw_")]
            if allow_r:
                c = c.allow_random(*allow_r, distributions=["normal"])
            # Grouped structure by ROAD TYPE: segments sharing a highway
            # functional class share a group-level random effect (role 4) and
            # class-driven heterogeneity in random-parameter means (role 5).
            # Inspect FIRST: only enable roles 4/5 when highway is a true
            # common term (>=2 levels each with >=20 segs). Otherwise every
            # grouped proposal errors (fitness=1e12) and wastes the search.
            _use_g, _gcounts = _inspect_groups(mc, col="highway", min_n=20)
            audit["group_inspection"] = _gcounts
            if _use_g:
                builder = ExperimentBuilder(mc, id_col="road_osmid",
                                            y_col="Y", group_id_col="highway")
                # Mixed structural equations for random parameters:
                # 3 correlated + 4 grouped + 5 heterogeneity-in-means.
                roles = [0, 1, 2, 3, 4, 5]
            else:
                builder = ExperimentBuilder(mc, id_col="road_osmid", y_col="Y")
                # Grouped (4) off; still allow correlated (3) + hetero (5).
                roles = [0, 1, 2, 3, 5]
            # Run the count search on the supported JAX evaluator.
            evaluator, _eng_used = _build_evaluator(
                builder, search_variables, c, roles,
                int(args.search_R), str(args.engine))
            if args.trace_fitness:
                evaluator = _TracingEvaluator(evaluator)
            audit["engine_used"] = _eng_used
            audit["default_roles"] = list(roles)
            print(f"   search algo={args.search_algo} "
                  f"iters={args.search_iters} R={args.search_R} "
                  f"jobs={args.search_jobs} engine={_eng_used} roles={roles} "
                f"trace_fitness={args.trace_fitness} "
                f"({len(search_variables)} variables)")
            try:
                from metacountregressor import SearchOutputConfig
                out_cfg = SearchOutputConfig(
                    output_dir=os.path.join(RES, "search47"),
                    experiment_name="seg_speed",
                    search_description="segment speeding baseline structure search")
            except Exception:  # noqa: BLE001 - older builds lack SearchOutputConfig
                out_cfg = None
        # Chunked search: checkpoint the best spec after every chunk so
        # usable results are returned before the job finishes (or is killed
        # at walltime); a later run resumes via --resume-from. In-memory
        # evaluator-cache warm starts carry progress across chunks.
        total_iters = int(args.search_iters)
        chunk_iters = max(1, int(args.search_chunk_iters))
        done_iters, n_chunk = 0, 0
        t_search0 = time.time()
        sres = None
        if _resume_active:
            total_iters = int(saved.get("iters_completed") or total_iters)
            done_iters = total_iters
            sres = {"best_score": _resume_best_score,
                "best_spec": _resume_best_spec}
        sa_only = str(args.search_algo).lower() in ("sa", "hc")
        while done_iters < total_iters and not _STOP["flag"]:
            this_iters = min(chunk_iters, total_iters - done_iters)
            n_chunk += 1
            run_kw = dict(n_jobs=int(args.search_jobs))
            if sa_only:
                run_kw["n_starts"] = 1  # one restart per chunk; chunks chain
            run_kw["refit"] = False
            print(f"   [search] starting iteration {done_iters + 1}/"
                  f"{total_iters} (chunk {n_chunk}, cache="
                  f"{len(getattr(evaluator, 'cache', {}) or {})})",
                  flush=True)
            try:
                if str(args.search_algo).lower() == "agds":
                    sres = _run_sparse_ea_agds(
                        evaluator, search_variables, this_iters,
                        int(args.seed) + done_iters)
                    _bbic = float(sres["best_score"])
                    _bspec = sres["best_spec"]
                    _neval = int(sres["n_evals"])
                else:
                    sres = builder.run(
                        evaluator, algo=args.search_algo, max_iter=this_iters,
                        seed=int(args.seed) + done_iters, output_config=out_cfg,
                        **run_kw)
                    _bbic, _bspec, _neval = _extract_search_best(evaluator)
            except Exception as e:  # noqa: BLE001 - expose failed local/HPC runs
                print(f"   [search] iteration {done_iters + 1} FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                raise
            done_iters += this_iters
            _write_search_checkpoint(
                SEARCH_BEST, spec=_bspec, best_bic=_bbic,
                algo=args.search_algo, iters_completed=done_iters,
                iters_total=total_iters, R=int(args.search_R),
                seed=int(args.seed), engine=audit.get("engine_used", "?"),
                roles=audit.get("default_roles", []), partial=True)
            _append_progress_row(SEARCH_PROGRESS, {
                "chunk": n_chunk, "iters_completed": done_iters,
                "elapsed_s": round(time.time() - t_search0, 1),
                "best_bic": ("" if not np.isfinite(_bbic)
                             else round(float(_bbic), 2)),
                "n_cache_evals": int(_neval),
                "timestamp_iso": _utcnow_iso()})
            _plot_search_progress(SEARCH_PROGRESS, SEARCH_PROGRESS_PNG)
            _bic_text = ("nan" if not np.isfinite(_bbic)
                         else f"{float(_bbic):.2f}")
            print(f"   [fitness] iteration {done_iters}/{total_iters}: "
                  f"best_BIC={_bic_text} cache_evals={_neval} "
                  f"elapsed={time.time() - t_search0:.1f}s", flush=True)
            print(f"   [checkpoint] chunk {n_chunk}: iters "
                  f"{done_iters}/{total_iters} best_BIC={_bbic:.1f} "
                  f"-> {SEARCH_BEST}", flush=True)
        if _STOP["flag"] and done_iters < total_iters:
            print(f"   STOP requested after {done_iters}/{total_iters} iters; "
                  "refitting from the checkpointed best spec.")
        best_score = _rget(sres, "best_score", float("nan"))
        # SA returns best_spec; HS returns best_solution/model_spec.
        best_spec = _rget(sres, "best_spec", None)
        if best_spec is None:
            best_spec = _rget(sres, "best_solution", None)
        if best_spec is None:
            best_spec = _rget(sres, "model_spec", None)
        # This build returns best_*=None in the run() dict even on convergence;
        # the SearchOutputConfig JSON on disk always carries the real result.
        if best_spec is None or not np.isfinite(float(best_score or float("nan"))):
            _jf = _rget(sres, "saved_to", None) or _newest_search_json(
                os.path.join(RES, "search47"))
            if _jf and os.path.exists(_jf):
                print(f"   run() dict empty; recovering from {_jf}...")
                _fs, _fx = _load_search_result_file(_jf)
                if _fx is not None:
                    best_spec = _fx
                if _fs is not None and np.isfinite(float(_fs)):
                    best_score = float(_fs)
        if best_spec is None:
            raise SystemExit("FATAL: search returned no best_spec "
                             f"(keys={list(sres) if isinstance(sres, dict) else type(sres)})")
        if str(audit.get("engine_used", args.engine)).lower() == "numba":
            if not (isinstance(best_spec, dict)
                    and "fixed_terms" in best_spec):
                best_spec = _numba_spec_from_decision(
                    best_spec, search_variables)
        print(f"   best BIC={best_score:.1f} spec={best_spec}")
        spec_path = SEARCH_BEST
        try:
            _write_search_checkpoint(
                spec_path, spec=best_spec, best_bic=best_score,
                algo=args.search_algo, iters_completed=done_iters,
                iters_total=total_iters, R=int(args.search_R),
                seed=int(args.seed),
                engine=audit.get("engine_used", str(args.engine)),
                roles=audit.get("default_roles", []),
                partial=bool(done_iters < total_iters))
            print(f"   checkpointed best spec -> {spec_path}")
        except Exception as e:  # noqa: BLE001
            print(f"   WARN: spec checkpoint failed ({e})")
        if args.search_only:
            print("DONE! Search-only run: best spec checkpointed.")
            return
        if str(audit.get("engine_used", args.engine)).lower() == "numba":
            info = _fit_numba_selected_spec(
                mc, best_spec, int(args.refit_R))
            info["spec_label"] = f"searched-numba-{args.search_algo}"
        else:
            fit = builder.fit_manual_model(manual_spec=best_spec,
                                           model="nb", R=int(args.refit_R))
            from metacount_nb import _parse_summary_text as _parse
            parsed = _parse(fit.get("_summary_text", ""))
            dp = parsed.get("parte", {}).get("dispersion", np.nan)
            alpha = float(dp) if np.isfinite(dp) and 0 < dp < 1e4 else 1.0
            expected = np.maximum(
                np.asarray(fit["predictions"], dtype=float), 1e-9)
            summ = fit.get("summary", {})
            info = {"expected": expected, "alpha": alpha,
                    "aic": float(summ.get("aic", np.nan)),
                    "bic": float(summ.get("bic", np.nan)),
                    "parsed": parsed,
                    "spec_label": f"searched-{args.search_algo}"}
        audit["stageA_search"] = {
            "ran": True, "algo": args.search_algo,
            "iters": int(args.search_iters),
            "iters_completed": int(done_iters),
            "iters_total": int(total_iters),
            "partial": bool(done_iters < total_iters),
            "search_chunks": int(n_chunk),
            "R": int(args.search_R),
            "refit_R": int(args.refit_R), "best_bic": float(best_score),
            "best_spec": str(best_spec),
            "engine": audit.get("engine_used", str(args.engine)),
            "roles": audit.get("default_roles", []),
            "progress_csv": SEARCH_PROGRESS,
            "progress_png": SEARCH_PROGRESS_PNG}
    base["lambda_base"] = info["expected"]
    print(f"   AIC={info['aic']:.0f} alpha={info['alpha']:.3f} "
          f"[{info['spec_label']}] n={len(base):,}")
    audit["stageA"] = {"aic": info["aic"], "bic": info["bic"],
                       "alpha": info["alpha"], "n_seg": len(base),
                       "spec": info["spec_label"]}
    base[["road_osmid", "lambda_base"]].to_csv(
        os.path.join(RES, "segment_speeding_pred_base.csv"), index=False)
    # labelled coefficient table for the paper (plain-English variables).
    audit["variable_labels"] = {c: var_label(c)
                                for c in (["logexp"] + cont + cnt + poi_cols)}
    try:
        _pp = info.get("parsed", {})
        _rows = []
        for _nm, _b in _pp.get("mle", {}).items():
            _rows.append({
                "variable": var_label(_nm),
                "coef": float(_b),
                "se": float(_pp.get("se", {}).get(_nm, float("nan"))),
                "p": float(_pp.get("p", {}).get(_nm, float("nan"))),
            })
        pd.DataFrame(_rows).to_csv(
            os.path.join(RES, "segment_speeding_coef.csv"), index=False)
        with open(os.path.join(RES, "segment_speeding_coef.tex"), "w") as f:
            f.write("% segment_speeding_coef.tex - auto-generated\n")
            f.write("\\begin{table}[htbp]\n\\centering\n")
            f.write("\\caption{\\textcolor{Highlight}{Searched segment "
                    "speeding baseline: coefficients (plain-English "
                    "variables).}}\n")
            f.write("\\label{tab:speedcoef}\n")
            f.write("\\begin{tabular}{lrrr}\n\\hline\n")
            f.write("Variable & Coef. & SE & $p$ \\\\\n\\hline\n")
            for _r in _rows:
                _se = f"{_r['se']:.3f}" if np.isfinite(_r['se']) else "--"
                _p = f"{_r['p']:.3f}" if np.isfinite(_r['p']) else "--"
                f.write(f"{_r['variable']} & {_r['coef']:.3f} & {_se} & {_p} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
        print(f"   wrote labelled coef table ({len(_rows)} variables)")
    except Exception as e:  # noqa: BLE001
        print(f"   WARN: coef table failed ({e})")

    # ---- Stage B: network time/attendance/DOW model (train games) ------------
    # Event rows only: each fixture contributes one network total, scaled by
    # published attendance (Stage B1). Background (attendance=0) is modelled
    # separately in Stage B2 on the day-of-week x clock-hour surface.
    print("\n[3/6] Stage B: crowd x time-of-week model (train games)...")
    fx = tr_ev.groupby("fixture").size().reset_index(name="n")
    fx = fx.merge(cal[["label", "attendance", "log_att", "dow", "weekend",
                        "ko_sin", "ko_cos", "fixture_type"]],
                  left_on="fixture", right_on="label", how="left")
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        _statsmodels_error = None
    except Exception as e:  # noqa: BLE001 - SciPy/statsmodels may mismatch
        smf, sm = None, None
        _statsmodels_error = e
    fx["dow_sin"] = np.sin(2 * np.pi * fx["dow"] / 7.0)
    fx["dow_cos"] = np.cos(2 * np.pi * fx["dow"] / 7.0)
    form = ("n ~ log_att + weekend + dow_sin + dow_cos + ko_sin + ko_cos"
            " + C(fixture_type)")
    if smf is None:
        print(f"   statsmodels unavailable ({_statsmodels_error}); "
              "using local Poisson fallback")
        nb, alpha_b = _fit_stage_b_poisson_fallback(fx)
        lam_b = np.asarray(nb.fittedvalues, dtype=float)
        tag_b = f"Poisson-fallback({form})"
        aic_b = float(nb.aic)
    else:
        try:
            nb = smf.negativebinomial(
                form, data=fx).fit(disp=False, method="lbfgs", maxiter=2000)
            conv = bool(getattr(nb, "mle_retvals", {}).get("converged", True))
            alpha_b = (float(nb.params["alpha"])
                       if "alpha" in nb.params else float(nb.scale))
            lam_b = np.asarray(nb.predict(fx), dtype=float)
            ok = conv and np.all(np.isfinite(lam_b)) and 0 < alpha_b <= 100
            tag_b = f"NB({form})"
            aic_b = float(nb.aic)
        except Exception as e:  # noqa: BLE001
            print(f"   NB time model failed ({e}); Poisson-GLM fallback")
            nb, ok = None, False
        if not ok:
            poi = smf.glm(form, data=fx,
                          family=sm.families.Poisson()).fit(disp=False)
            lam_b = np.asarray(poi.fittedvalues, dtype=float)
            pearson = float(((poi.resid_pearson) ** 2).sum()
                            / max(poi.df_resid, 1))
            alpha_b = float(max((pearson - 1.0)
                                / max(np.mean(lam_b), 1e-9), 1e-3))
            nb, tag_b, aic_b = poi, f"Poisson-GLM({form})", float(poi.aic)
    fx["fitted"] = lam_b
    eta = float(nb.params["log_att"]) if "log_att" in nb.params else float("nan")
    print(f"   {tag_b} AIC={aic_b:.0f} alpha={alpha_b:.3f} eta={eta:.3f}")
    audit["stageB"] = {"spec": tag_b, "aic": aic_b, "alpha": alpha_b,
                       "eta_log_att": eta,
                       "params": {k: float(v) for k, v in nb.params.items()}}
    abar = float(cal.loc[train_idx, "attendance"].mean())

    # ---- Stage B*: metacount game panel (flow x time x stadium size) --------
    # Segment x train-fixture NB panel fit IN metacountregressor: finable
    # counts ~ per-segment flow/exposure + game start time (kickoff cyclic) +
    # stadium size (attendance, capacity utilisation, near-sellout) +
    # day-of-week + fixture type + highway class. Time, capacity and flow per
    # segment thus live inside the metacount model (not just the network
    # scale); per-segment predictions for all 41 fixtures feed the MIP.
    # Any failure keeps the Stage B statsmodels scale (never fatal).
    use_panel = False
    panel_pred_by_fix = {}
    audit["stageBstar"] = {"ran": False}
    if not args.no_panel:
        try:
            from metacount_nb import (prepare_metacount_frame as _pmf,
                                      fit_metacount_nb as _fmb,
                                      predict_metacount as _pmd)
            _pfix = sorted(set(train_lab)
                           & set(tr_ev["fixture"].unique().tolist()))
            if not _pfix:
                raise RuntimeError("no train fixtures with event rows")
            _cnt = (tr_ev[tr_ev["fixture"].isin(_pfix)]
                    .groupby(["road_osmid", "fixture"]).size()
                    .reset_index(name="Y"))
            _segs = base[["road_osmid", "exposure_trips", "highway",
                          "dist_stadium_km"]].drop_duplicates().copy()
            _segs["exposure_trips"] = pd.to_numeric(
                _segs["exposure_trips"], errors="coerce").fillna(0.0)
            _segs = _segs[_segs["exposure_trips"] > 0].copy()
            _med_ds = float(pd.to_numeric(_segs["dist_stadium_km"],
                                          errors="coerce").median())
            _segs["dist_stadium_km"] = pd.to_numeric(
                _segs["dist_stadium_km"], errors="coerce").fillna(_med_ds)
            _pan = (_segs.assign(_k=1).merge(
                pd.DataFrame({"fixture": _pfix, "_k": 1}),
                on="_k").drop(columns="_k"))
            _pan = _pan.merge(_cnt, on=["road_osmid", "fixture"], how="left")
            _pan["Y"] = _pan["Y"].fillna(0.0)
            # game/stadium/time covariates from the calendar
            _cap_ref = float(cal.loc[train_idx, "attendance"].max())
            _g = cal[["label", "attendance", "log_att", "utilization",
                      "ko_sin", "ko_cos", "dow", "weekend",
                      "fixture_type"]].copy()
            _g["utilization"] = pd.to_numeric(_g["utilization"],
                                              errors="coerce")
            _proxy = _g["attendance"] / max(_cap_ref, 1e-9)
            _g["utilization"] = _g["utilization"].fillna(_proxy)
            _g["near_sellout"] = ((_g["utilization"] >= 0.9)
                                  ).astype(float)
            _g["dow_sin"] = np.sin(2 * np.pi * _g["dow"].astype(float) / 7.0)
            _g["dow_cos"] = np.cos(2 * np.pi * _g["dow"].astype(float) / 7.0)
            _pan = _pan.merge(
                _g, left_on="fixture", right_on="label", how="left").drop(
                columns=["label"])
            # per-segment flow for this fixture (Compass where keyed,
            # else the segment exposure level)
            if (compass_df is not None and len(compass_df)
                    and "fixture" in compass_df.columns):
                _fl = (compass_df[compass_df["fixture"].isin(_pfix)]
                       .groupby(["road_osmid", "fixture"],
                                as_index=False)["flow_veh"].mean())
                _pan = _pan.merge(
                    _fl.rename(columns={"flow_veh": "flow_fix"}),
                    on=["road_osmid", "fixture"], how="left")
                _pan["flow_fix"] = _pan["flow_fix"].fillna(
                    _pan.groupby("road_osmid")["flow_fix"].transform("mean"))
                _flow_src = "compass fixture means"
            else:
                _pan["flow_fix"] = _pan["exposure_trips"]
                _flow_src = "exposure_trips (compass fixture keys absent)"
            _pan["log_flow"] = np.log(_pan["flow_fix"].fillna(0.0) + 1.0)
            # standardise continuous (train stats recorded for prediction)
            _std = {}
            for _cc in ("log_flow", "log_att", "utilization",
                        "dist_stadium_km"):
                _v = pd.to_numeric(_pan[_cc], errors="coerce").fillna(0.0)
                _mu, _sd = float(_v.mean()), float(_v.std())
                _sd = _sd if _sd and _sd > 0 else 1.0
                _pan[_cc + "_z"] = (_v - _mu) / _sd
                _std[_cc] = {"mean": _mu, "sd": _sd}
            _td = pd.get_dummies(_pan["fixture_type"].fillna(""),
                                 prefix="type", dtype=float, drop_first=True)
            _type_cols = sorted(_td.columns.tolist())
            for _cc in _type_cols:
                _pan[_cc] = _td[_cc].values
            if len(_pan) > 200_000:
                raise RuntimeError(f"panel too large ({len(_pan):,} rows)")
            _mc2, _ = _pmf(_pan, exposure_col="exposure_trips",
                           highway_col="highway")
            _fixed = (["logexp"]
                      + [c for c in _mc2.columns if c.startswith("hw_")]
                      + ["log_flow_z", "log_att_z", "utilization_z",
                         "dist_stadium_km_z", "near_sellout", "dow_sin",
                         "dow_cos", "ko_sin", "ko_cos", "weekend"]
                      + _type_cols)
            _fixed = [c for c in _fixed if c in _mc2.columns]
            _rdm = (["dist_stadium_km_z:normal"]
                    if "dist_stadium_km_z" in _mc2.columns else [])
            _pan["row_id"] = (_pan["road_osmid"].astype(str) + "__"
                              + _pan["fixture"].astype(str))
            _pinfo = _fmb(_mc2, y_col="Y", id_col="row_id",
                          fixed_terms=_fixed, rdm_terms=_rdm,
                          model="nb", R=int(args.panel_R),
                          de_maxiter=8, de_popsize=8, verbose=True)
            # predict all 41 fixtures with TRAIN standardisation
            _Pall = []
            for _lab, _grow in cal.groupby("label"):
                _d = _segs[["road_osmid", "exposure_trips", "highway",
                            "dist_stadium_km"]].copy()
                _d["fixture"] = _lab
                _d["Y"] = 0.0
                for _cc in ("log_att", "utilization", "ko_sin", "ko_cos",
                            "dow_sin", "dow_cos", "weekend", "near_sellout"):
                    _d[_cc] = float(_grow[_cc].iloc[0]) if _cc in _grow \
                        else 0.0
                for _cc in _type_cols:
                    _t = str(_grow["fixture_type"].iloc[0]
                             ) if "fixture_type" in _grow else ""
                    _d[_cc] = 1.0 if _cc == f"type_{_t}" else 0.0
                _d["log_flow"] = np.log(
                    _d["exposure_trips"].fillna(0.0) + 1.0)
                for _cc in ("log_flow", "log_att", "utilization",
                            "dist_stadium_km"):
                    _s = _std[_cc]
                    _d[_cc + "_z"] = ((pd.to_numeric(_d[_cc],
                                                     errors="coerce")
                                       .fillna(0.0) - _s["mean"]) / _s["sd"])
                _d["row_id"] = (_d["road_osmid"].astype(str) + "__"
                                + _d["fixture"].astype(str))
                Pall.append(_d)
            _Pall = pd.concat(Pall, ignore_index=True)
            _Mpred, _ = _pmf(_Pall, exposure_col="exposure_trips",
                             highway_col="highway")
            for _cc in _pinfo.get("fixed_terms", []):
                if _cc not in _Mpred.columns:
                    _Mpred[_cc] = 0.0
            _pred = np.maximum(
                np.asarray(_pmd(_Mpred, _pinfo), dtype=float), 0.0)
            _Pall["h_total"] = _pred
            for _lab, _gp in _Pall.groupby("fixture"):
                panel_pred_by_fix[str(_lab)] = dict(
                    zip(_gp["road_osmid"].astype(np.int64),
                        _gp["h_total"].astype(float)))
            use_panel = True
            print(f"   Stage B* metacount panel: n={len(_pan):,} "
                  f"AIC={_pinfo['aic']:.0f} alpha={_pinfo['alpha']:.3f} "
                  f"[{_pinfo['spec_label']}] flow={_flow_src}")
            audit["stageBstar"] = {
                "ran": True, "n_rows": int(len(_pan)),
                "n_fixtures": len(_pfix), "aic": _pinfo["aic"],
                "bic": _pinfo["bic"], "alpha": _pinfo["alpha"],
                "spec": _pinfo["spec_label"], "flow": _flow_src,
                "r_draws": int(args.panel_R),
                "fixed": _fixed, "random": _rdm}
        except Exception as e:  # noqa: BLE001 - fallback keeps the run alive
            print(f"   WARN: Stage B* panel failed ({e}); "
                  "statsmodels crowd scale kept")
            audit["stageBstar"] = {"ran": False, "error": str(e)[:300]}
            use_panel = False
            panel_pred_by_fix = {}

    # ---- Stage C: phase x hour shares (train event rows only) ---------------
    print("\n[4/6] Stage C: phase x hour shares (train events only)...")
    ph = tr_ev.groupby("phase").size()
    ph = ph / ph.sum()
    phase_shares = {str(k): float(v) for k, v in ph.items()}
    hr = tr_ev.groupby(["phase", "hbin"]).size().reset_index(name="n")
    hr["tot_phase"] = hr.groupby("phase")["n"].transform("sum")
    hr["share"] = hr["n"] / hr["tot_phase"]
    hour_share = {(str(p), int(h)): float(s)
                  for p, h, s in zip(hr["phase"], hr["hbin"], hr["share"])}
    print(f"   phase shares: {phase_shares}")
    audit["phase_shares"] = phase_shares
    audit["hour_shares"] = {f"{p}/{h}": s for (p, h), s in hour_share.items()}

    # ---- Stage C2: phase-pair copula dependence (metacountregressor) ------
    # Hypothesis testing + dependence-structure selection for
    # (pre,during), (during,post), (pre,post): univariate baseline vs
    # Famoye / Marshall-Olkin / Frank / Normal / Kimeldorf-Samson, chosen
    # by BIC among converged fits with LR-vs-independence p-values.
    # Metacount selects structure here; it never sets ranking weights.
    print("\n[4c/6] Stage C2: phase-pair copula tests (metacountregressor)...")
    if args.no_copula:
        print("   skipped (--no-copula)")
        audit["copula"] = {"ran": False, "reason": "--no-copula"}
    else:
        try:
            _cop_compare, _cop_audit, _cop_phase = _run_phase_copulas(
                tr_ev=tr_ev, base=base, cal=cal, train_lab=train_lab,
                compass_df=compass_df, max_rows=int(args.copula_max_rows),
                seed=int(args.seed))
            _atomic_write_csv(_cop_compare, COPULA_COMPARE)
            _atomic_write_json(COPULA_AUDIT, _cop_audit)
            with open(COPULA_TEX, "w") as f:
                f.write("% segment_speeding_copula.tex - auto-generated by "
                        "47_segment_speeding_predictive.py\n")
                f.write("\\begin{table}[htbp]\n\\centering\n")
                f.write("\\caption{\\textcolor{Highlight}{Phase-pair speeding "
                        "dependence: copula hypothesis tests (train-fit).} "
                        "BIC selects among converged fits; LR tests each "
                        "copula against independence.}\n")
                f.write("\\label{tab:speedcop}\n")
                f.write("\\begin{tabular}{lllrrrr}\n\\hline\n")
                f.write("Pair & Model & $k$ & logLik & AIC & BIC & "
                        "$p$ (indep.) \\\\\n\\hline\n")
                for _, _r in _cop_compare.iterrows():
                    _p = _r.get("p_independence", float("nan"))
                    _ps = (f"{float(_p):.3g}" if pd.notna(_p) and
                           np.isfinite(float(_p)) else "--")
                    _mk = "*" if bool(_r.get("best_flag", False)) else ""
                    f.write(f"{_r.get('pair','')} & {_r.get('model','')}{_mk} & "
                            f"{int(_r.get('k', 0))} & "
                            f"{float(_r.get('loglik', float('nan'))):.1f} & "
                            f"{float(_r.get('AIC', float('nan'))):.1f} & "
                            f"{float(_r.get('BIC', float('nan'))):.1f} & {_ps} \\\\\n")
                f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
            _cop_phase.to_csv(os.path.join(
                RES, "segment_speeding_pred_copula_phase.csv.gz"), index=False)
            audit["copula"] = _cop_audit
            for _pk, _pv in _cop_audit.get("pairs", {}).items():
                print(f"   {_pk}: best={_pv.get('best_model')} "
                      f"BIC={_pv.get('best_bic')}")
        except Exception as e:  # noqa: BLE001 - copula must not kill the run
            print(f"   WARN: copula stage failed ({e}); continuing without it")
            audit["copula"] = {"ran": False, "error": str(e)[:300]}

    # ---- Stage B2: background surface (attendance = 0, all other hours) ------
    # Outside fixture windows the stadium draws no crowd, so speeding follows
    # the background day-of-week x clock-hour network rate B(d,h) (events per
    # clock-hour, train period). Per-segment background hazard reuses the
    # searched propensity: h_bg(i,d,h) = lambda_i * B(d,h) / L, L = sum lambda.
    print("\n[4b/6] Stage B2: background day-of-week x clock-hour surface...")
    tr_bg["date"] = tr_bg["dt"].dt.date
    dd = tr_bg.drop_duplicates("date")
    n_dates = dd.groupby(dd["dt"].dt.dayofweek).size().to_dict()
    bg_c = tr_bg.groupby(["dow", "hour"]).size().reset_index(name="n")
    bg_c["n_dates"] = bg_c["dow"].map(n_dates).fillna(1).astype(float)
    bg_c["bg_rate_per_hour"] = bg_c["n"] / bg_c["n_dates"]
    # fill unobserved dow x hour cells with the dow mean (never zero-fill:
    # a gap means no observation, and the dow mean is the honest prior)
    full = pd.MultiIndex.from_product([range(7), range(24)],
                                      names=["dow", "hour"]).to_frame(index=False)
    bg_c = full.merge(bg_c, on=["dow", "hour"], how="left")
    bg_c["bg_rate_per_hour"] = bg_c["bg_rate_per_hour"].fillna(
        bg_c.groupby("dow")["bg_rate_per_hour"].transform("mean"))
    bg_c["bg_rate_per_hour"] = bg_c["bg_rate_per_hour"].fillna(
        float(bg_c["bg_rate_per_hour"].mean()))
    bg_c.to_csv(os.path.join(RES, "segment_speeding_pred_background.csv"),
                index=False)
    bg_map = {(int(d), int(h)): float(r) for d, h, r in
              zip(bg_c["dow"], bg_c["hour"], bg_c["bg_rate_per_hour"])}
    print(f"   background network mean: "
          f"{float(bg_c['bg_rate_per_hour'].mean()):.2f} events/hour "
          f"(train dates={int(tr_bg['date'].nunique())})")
    audit["background"] = {
        "train_dates": int(tr_bg["date"].nunique()),
        "mean_rate_per_hour": float(bg_c["bg_rate_per_hour"].mean())}

    # ---- Hazard predictions for every calendar fixture ------------------------
    print("\n[5/6] Predicting segment x fixture x phase x hour...")
    lam_map = dict(zip(base["road_osmid"].values, base["lambda_base"].values))
    tot_lam = float(base["lambda_base"].sum())
    # network scale per fixture from Stage B (recomputed per fixture row)
    cal_f = cal.copy()
    cal_f["dow_sin"] = np.sin(2 * np.pi * cal_f["dow"] / 7.0)
    cal_f["dow_cos"] = np.cos(2 * np.pi * cal_f["dow"] / 7.0)
    cal_f["net_pred"] = np.asarray(nb.predict(cal_f), dtype=float)
    net_map = dict(zip(cal_f["label"], cal_f["net_pred"]))
    net_train_mean = float(fx["fitted"].mean())
    rows = []
    for _, f in cal_f.iterrows():
        h = base[["road_osmid"]].copy()
        h["fixture"] = f["label"]
        h["kickoff_iso"] = str(f["ko"])
        h["attendance"] = float(f["attendance"])
        h["stadium_capacity"] = float(f.get("stadium_capacity", np.nan))
        h["utilization"] = float(f.get("utilization", np.nan))
        h["fixture_type"] = str(f.get("fixture_type", ""))
        if use_panel and str(f["label"]) in panel_pred_by_fix:
            # Stage B* metacount panel: absolute per-segment predictions
            # (flow x start time x stadium size x road type inside metacount).
            h["h_total"] = h["road_osmid"].map(
                panel_pred_by_fix[str(f["label"])]).fillna(0.0)
        else:
            scale = float(net_map[f["label"]]) / max(net_train_mean, 1e-9)
            h["h_total"] = h["road_osmid"].map(lam_map) * scale
        for p, s in phase_shares.items():
            h[f"h_{p}"] = h["h_total"] * float(s)
        rows.append(h)
    hz = pd.concat(rows, ignore_index=True)
    hz.to_csv(os.path.join(RES, "segment_speeding_pred_fixture.csv.gz"), index=False)
    print(f"   wrote segment_speeding_pred_fixture.csv.gz ({len(hz):,} rows)")

    # hourly expansion (per-hour predictions for the MIP roster):
    # one row per segment x fixture x phase x event-relative hour bin.
    hrows = []
    for _, f in cal_f.iterrows():
        if use_panel and str(f["label"]) in panel_pred_by_fix:
            _pp = panel_pred_by_fix[str(f["label"])]
        else:
            _pp = None
            scale = float(net_map[f["label"]]) / max(net_train_mean, 1e-9)
        for (p, hb), s in hour_share.items():
            d = base[["road_osmid"]].copy()
            d["fixture"] = f["label"]
            d["kickoff_iso"] = str(f["ko"])
            d["attendance"] = float(f["attendance"])
            d["stadium_capacity"] = float(f.get("stadium_capacity", np.nan))
            d["utilization"] = float(f.get("utilization", np.nan))
            d["fixture_type"] = str(f.get("fixture_type", ""))
            d["dow"] = int(f.get("dow", -1))
            d["phase"] = p
            d["hbin"] = int(hb)
            d["rel_hour"] = int(hb)  # hours relative to kickoff (-3..3)
            if _pp is not None:
                d["h_pred"] = (d["road_osmid"].map(_pp).fillna(0.0)
                               * float(phase_shares.get(p, 0)) * float(s))
            else:
                d["h_pred"] = (d["road_osmid"].map(lam_map) * scale
                               * float(phase_shares.get(p, 0)) * float(s))
            hrows.append(d)
    hh = pd.concat(hrows, ignore_index=True)
    # per-km intensity + operational context (asof, hours-to-kickoff)
    seg_len = base.set_index("road_osmid")["length_m"].to_dict() \
        if "length_m" in base.columns else {}
    hh["length_m"] = hh["road_osmid"].map(seg_len).fillna(200.0)
    hh["h_pred_per_km"] = hh["h_pred"] / (hh["length_m"] / 1000.0).clip(lower=0.05)
    hh["asof_iso"] = asof_iso
    try:
        _ko = pd.to_datetime(hh["kickoff_iso"]).dt.tz_localize(
            "Australia/Brisbane", ambiguous="NaT", nonexistent="shift_forward")
        _asof = pd.to_datetime(asof_iso, utc=True)
        hh["hours_to_kickoff"] = ((_ko - _asof).dt.total_seconds() / 3600.0)
    except Exception:  # noqa: BLE001 - tz data may be missing; not fatal
        hh["hours_to_kickoff"] = np.nan
    # Compass flow amounts alongside predictions (conditioning/audit info for
    # the MIP; never a ranking weight).
    if compass_df is not None and len(compass_df):
        _cf = compass_df.copy()
        _keys = ["road_osmid"] + [c for c in ("fixture", "phase", "hbin")
                                  if c in _cf.columns]
        if _cf.duplicated(subset=_keys).any():
            _cf = _cf.groupby(_keys, as_index=False)["flow_veh"].mean()
        hh = hh.merge(_cf.rename(columns={"flow_veh": "flow_compass"}),
                      on=_keys, how="left")
        audit["compass_join_keys"] = _keys
    else:
        hh["flow_compass"] = np.nan
        audit["compass_join_keys"] = []
    hh.to_csv(FIXTURE_HOURLY, index=False)
    print(f"   wrote {os.path.basename(FIXTURE_HOURLY)} ({len(hh):,} rows: "
          "segment x fixture x phase x hour, before/during/after per hour)")
    audit["n_hourly_pred_rows"] = int(len(hh))
    audit["hourly_outputs"] = {
        "fixture_hourly": os.path.basename(FIXTURE_HOURLY),
        "phase_totals": os.path.basename(PHASE_TOTALS),
    }
    # per-fixture phase totals: predicted occurrence counts before/during/after
    pt = (hh.groupby(["fixture", "kickoff_iso", "attendance",
                      "stadium_capacity", "utilization", "phase"],
                     as_index=False, dropna=False)["h_pred"].sum()
            .rename(columns={"h_pred": "pred_total"}))
    pt["n_segments"] = int(hh["road_osmid"].nunique())
    _atomic_write_csv(pt, PHASE_TOTALS)
    audit["n_phase_total_rows"] = int(len(pt))
    print(f"   wrote {os.path.basename(PHASE_TOTALS)} ({len(pt):,} rows)")

    # ---- Holdout validation: predicted vs real thresholds ----------------------
    # Event holdout: holdout fixtures' segment totals vs leave-games-out
    # predictions. Background holdout: holdout-week non-window hours segment
    # totals vs propensity shares of the realised background total.
    print("\n[6/6] Holdout validation (predicted vs real)...")
    ho = ex[(ex["split"] == "holdout") & (ex["window"] == "event")]
    ho_seg = ho.groupby("road_osmid").size().reset_index(name="Y_obs")
    pr_ho = hz[hz["fixture"].isin(hold_lab)].groupby("road_osmid")["h_total"].sum().reset_index()
    pr_ho = pr_ho.rename(columns={"h_total": "Y_pred"})
    # align onto modelled segments (zeros where unobserved in holdout)
    val = base[["road_osmid"]].merge(ho_seg, on="road_osmid", how="left").merge(
        pr_ho, on="road_osmid", how="left")
    val["Y_obs"] = val["Y_obs"].fillna(0.0)
    val["Y_pred"] = val["Y_pred"].fillna(0.0)
    rho = _spearman(val["Y_obs"], val["Y_pred"])
    jk = {f"top{k}": _jaccard_topk(val["Y_obs"], val["Y_pred"], k)
          for k in (20, 50, 100)}
    rmse = float(np.sqrt(np.mean((val["Y_obs"] - val["Y_pred"]) ** 2)))
    mae = float(np.mean(np.abs(val["Y_obs"] - val["Y_pred"])))
    # calibration: OLS obs ~ pred
    X = np.column_stack([np.ones(len(val)), val["Y_pred"].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, val["Y_obs"].to_numpy(), rcond=None)
    # threshold table: predicted vs real counts above cutoffs
    thr = {}
    for c in (1, 2, 5):
        thr[f"ge{c}"] = {"observed_segments": int((val['Y_obs'] >= c).sum()),
                         "predicted_segments": int((val['Y_pred'] >= c).sum())}
    print(f"   holdout Spearman={rho:.3f} RMSE={rmse:.3f} MAE={mae:.3f} "
          f"calib slope={beta[1]:.3f} J={jk}")
    audit["holdout"] = {"spearman": rho, "rmse": rmse, "mae": mae,
                        "calib_intercept": float(beta[0]),
                        "calib_slope": float(beta[1]),
                        "jaccard": jk, "thresholds": thr,
                        "n_holdout_events": int(len(ho))}
    # background holdout (attendance=0 hours of holdout weeks)
    ho_bg = ex[(ex["split"] == "holdout") & (ex["window"] == "background")]
    ho_bg_seg = ho_bg.groupby("road_osmid").size().reset_index(name="Y_obs_bg")
    btot = float(len(ho_bg))
    L = max(float(base["lambda_base"].sum()), 1e-12)
    vb = base[["road_osmid", "lambda_base"]].merge(ho_bg_seg, on="road_osmid",
                                                  how="left")
    vb["Y_obs_bg"] = vb["Y_obs_bg"].fillna(0.0)
    vb["Y_pred_bg"] = vb["lambda_base"] / L * btot
    rho_bg = _spearman(vb["Y_obs_bg"], vb["Y_pred_bg"])
    rmse_bg = float(np.sqrt(np.mean((vb["Y_obs_bg"] - vb["Y_pred_bg"]) ** 2)))
    print(f"   background holdout: n={btot:,.0f} Spearman={rho_bg:.3f} "
          f"RMSE={rmse_bg:.3f}")
    audit["holdout_background"] = {"spearman": rho_bg, "rmse": rmse_bg,
                                   "n_holdout_bg_events": btot}
    val.to_csv(os.path.join(RES, "segment_speeding_holdout.csv"), index=False)

    # learning curve: refit Stage-B scale on growing train subsets (A fixed) ---
    lrn = []
    order = cal.loc[train_idx].sort_values("ko")["label"].tolist()
    for k in range(8, len(order) + 1, 4):
        sub = set(order[:k])
        trk = ex[(ex["split"] == "train") & (ex["fixture"].isin(sub))]
        if trk.empty:
            continue
        # rescale baseline by sub-train network mean (cheap, no refit of A)
        fsub = trk.groupby("fixture").size().mean()
        fho_pred_scale = fsub / max(fx["n"].mean(), 1e-9)
        pv = val[["road_osmid"]].copy()
        pv["Y_pred_k"] = pv["road_osmid"].map(lam_map) * fho_pred_scale * (len(hold_lab))
        lrn.append({"n_train_games": k,
                    "spearman": _spearman(val["Y_obs"], pv["Y_pred_k"]),
                    "rmse": float(np.sqrt(np.mean((val["Y_obs"] - pv["Y_pred_k"]) ** 2)))})
    lrn = pd.DataFrame(lrn)
    audit["learning_curve"] = lrn.to_dict(orient="records")

    ok_hold = np.isfinite(rho) and rho >= 0.5 and 0.3 <= beta[1] <= 3.0
    audit["holdout_pass"] = bool(ok_hold)
    print(f"   holdout {'PASS' if ok_hold else 'FAIL'} "
          f"(needs rho>=0.5 and slope in [0.3,3.0])")

    # ---- Post x bucket predicted speeding (pure counts, feeds script 48) ------
    if os.path.exists(POSTS):
        import geopandas as gpd
        from scipy.spatial import cKDTree
        g = gpd.read_file(os.path.join(DATA, "osm_roads_suncorp.geojson"))
        g["osmid"] = pd.to_numeric(g["osmid"], errors="coerce")
        g = g.dropna(subset=["osmid"])
        gu = g.to_crs(28356)
        mid = gu.geometry.centroid
        seg_xy = np.column_stack([mid.x.values, mid.y.values])
        seg_ids = g["osmid"].values.astype(np.int64)
        posts = pd.read_csv(POSTS)
        pc = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(posts["lon"].values, posts["lat"].values),
            crs=4326).to_crs(28356)
        pxy = np.column_stack([pc.geometry.x.values, pc.geometry.y.values])
        dd, ii = cKDTree(pxy).query(seg_xy)
        seg_post = dict(zip(seg_ids.tolist(), ii.tolist()))
        # mean predicted speeding per segment-hour over TRAIN fixtures
        tr_hh = hh[hh["fixture"].isin(train_lab)]
        seg_h = tr_hh.groupby(["road_osmid", "hbin"])["h_pred"].mean().reset_index()
        seg_h["post"] = seg_h["road_osmid"].map(seg_post).fillna(-1).astype(int)
        seg_h = seg_h[seg_h["post"] >= 0]
        post_h = seg_h.groupby(["post", "hbin"])["h_pred"].sum().reset_index()
        post_h.to_csv(os.path.join(RES, "segment_speeding_pred_post_hourly.csv"),
                      index=False)
        audit["n_post_hourly_rows"] = int(len(post_h))
        print(f"   post-hourly surface: {len(post_h):,} rows "
              f"(pure speeding counts, no weights)")
    else:
        print("   WARN: post_selection_mip.csv missing; skipping post surface")

    # ---- Figures ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # calibration: binned predicted vs observed means
    fig, ax = plt.subplots(figsize=(6.5, 5))
    q = pd.qcut(val["Y_pred"].rank(method="first"), 10, duplicates="drop")
    cb = val.groupby(q, observed=True).agg(p=("Y_pred", "mean"),
                                           o=("Y_obs", "mean"))
    ax.plot(cb["p"], cb["o"], "o-", color="#0B6E4F", lw=2)
    mx = max(cb["p"].max(), cb["o"].max(), 1e-9)
    ax.plot([0, mx], [0, mx], "--", color="grey")
    ax.set_xlabel("Predicted holdout speeding per segment (train-fit model)")
    ax.set_ylabel("Observed holdout speeding per segment")
    ax.set_title(f"Convergence: predicted vs real thresholds (rho={rho:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "segment_speeding_calibration.png"), dpi=150)
    plt.close(fig)
    if len(lrn):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(lrn["n_train_games"], lrn["spearman"], "o-", color="#0B6E4F", lw=2)
        ax.set_xlabel("# train games")
        ax.set_ylabel("Holdout Spearman")
        ax.set_title("Segment speeding model convergence over games")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "segment_speeding_convergence.png"), dpi=150)
        plt.close(fig)

    # ---- LaTeX ------------------------------------------------------------------
    with open(os.path.join(RES, "segment_speeding_pred.tex"), "w") as f:
        f.write("% segment_speeding_pred.tex - auto-generated by "
                "47_segment_speeding_predictive.py\n")
        f.write("%% PURE speeding counts (>=%.0f kph); no risk weighting.\n" % M)
        f.write("\\begin{equation}\n")
        f.write("\\log h_{i,g,p,h} = \\hat\\beta_0 + \\hat\\beta_1\\log(E_i+1) "
                "+ \\mathbf{x}_i'\\hat{\\boldsymbol{\\beta}} "
                "+ \\hat\\eta\\log(A_g+1) + \\hat\\delta_{\\mathrm{dow}(g)} "
                "+ \\hat\\kappa(\\mathrm{ko}_g) + \\hat\\gamma_p "
                "+ \\log s_{h|p} + b_i\n")
        f.write("\\end{equation}\n")
        f.write("\\[ h_{\\mathrm{bg}}(i,d,h) = \\lambda_i\\, "
                "B(d,h)\\,/\\,\\textstyle\\sum_j\\lambda_j "
                "\\quad (A{=}0\\ \\mathrm{outside\\ windows}) \\]\n")
        f.write("\\begin{equation}\n")
        f.write("\\log h^{\\mathrm{panel}}_{i,g} = \\hat\\beta_0 + "
                "\\hat\\beta_1\\log(F_{ig}{+}1) + \\mathbf{z}_{ig}'"
                "\\hat{\\boldsymbol{\\gamma}} + b_i "
                "\\quad (\\mathrm{Stage\\ B^*:\\ metacount\\ NB\\ panel;\\ "
                "F{=}flow,\\ z{=}start\\ time{+}capacity{+}dow{+}type})\\\\\n")
        f.write("\\end{equation}\n")
        f.write(f"% StageA NB-RP AIC={info['aic']:.0f} alpha={info['alpha']:.3f} "
                f"[{info['spec_label']}] n={len(base):,} "
                f"metacountregressor={_MCR_VERSION}; "
                f"StageB {tag_b} AIC={aic_b:.0f} eta={eta:.3f}; "
                f"train={len(train_lab)} holdout={len(hold_lab)}.\n")
    with open(os.path.join(RES, "segment_speeding_convergence.tex"), "w") as f:
        f.write("% segment_speeding_convergence.tex - auto-generated\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{\\textcolor{Highlight}{Segment speeding model: "
                "holdout convergence (train-fit, holdout-scored).}}\n")
        f.write("\\label{tab:speedconv}\n\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write("Metric & Value \\\\\n\\hline\n")
        f.write(f"Holdout games & {len(hold_lab)} \\\\\n")
        f.write(f"Spearman $\\\\rho$ & {rho:.3f} \\\\\n")
        for k, v in jk.items():
            f.write(f"Jaccard {k} & {v:.3f} \\\\\n")
        f.write(f"RMSE & {rmse:.3f} \\\\\nMAE & {mae:.3f} \\\\\n")
        f.write(f"Calibration slope & {beta[1]:.3f} \\\\\n")
        f.write(f"Background holdout $\\\\rho$ (A=0) & {rho_bg:.3f} \\\\\n")
        f.write(f"Background holdout RMSE & {rmse_bg:.3f} \\\\\n")
        for c, d in thr.items():
            f.write(f"Segments {c}: obs/pred & "
                    f"{d['observed_segments']}/{d['predicted_segments']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    with open(os.path.join(RES, "segment_speeding_pred_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print("DONE! Segment speeding predictive model complete. "
          f"holdout_pass={bool(ok_hold)}")
    if not ok_hold:
        raise SystemExit("FATAL: holdout convergence guards failed; "
                         "refusing paper artefacts.")


if __name__ == "__main__":
    main()
