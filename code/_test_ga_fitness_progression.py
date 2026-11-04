"""
_test_ga_fitness_progression.py
================================
Purely SA (Simulated Annealing) fitness progression test across three estimators.

All three modes use SimulatedAnnealing as the searcher -- only the fitness
function (estimator) differs.  The test verifies that best_so_far actually
changes (improves) across SA iterations for each estimator.

Modes tested
------------
  normal        -- SA + default ordered-logit fitness (Stage 1)
  metacount     -- SA + MetaCount NB2 fitness         (Stage 1)
  searchlibrium -- SA + SearchLibrium MNL fitness     (Stage 4)

Per-iteration output shows:
    Gen | best_so_far | delta_from_prev | n_feats | accept%

PASS -- best_so_far improved at least once during the SA run
FAIL -- best_so_far never changed (estimator stuck at sentinel -1e9)

Run directly in VS Code (Run Python File button) or terminal:
  python _test_ga_fitness_progression.py

  # limit to specific modes:
  $env:GA_TEST_MODES='normal,metacount'
  python _test_ga_fitness_progression.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import contextlib
import io

# ── env setup (must come before any JAX/larch imports) ──────────────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("GA_METACOUNT_FORCE_CPU", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── core imports (fast) ───────────────────────────────────────────────────────
print("Loading ga_hypothesis_testing ...", flush=True)
from ga_hypothesis_testing import (
    SimulatedAnnealing,
    _stage1_fitness_aggregate,
    _stage2_fitness_aggregate,
    _stage3_fitness_aggregate,
    _stage5_trip_fitness_aggregate,
    _stage6_fitness_aggregate,
    STAGE_COLLINEAR_GROUPS,
)
print("  done.", flush=True)

# ── context manager: silence library-internal print() spam ────────────────────────
@contextlib.contextmanager
def _quiet():
    """Suppress stdout + stderr from library internals during fitness evals."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield

# ── external estimators loaded lazily in runner functions ───────────────────────
_METACOUNT_OK = None
_SEARCHLIB_OK = None

# ── configuration ─────────────────────────────────────────────────────────────
#  Tiny budget so the test finishes in < 3 minutes on a laptop CPU.
#  Increase POP_SIZE / N_GENS / N_RESTARTS for a more thorough run.
POP_SIZE   = 10
N_GENS     = 16
N_RESTARTS = 1
SEED       = 42

# Default is 'normal' only (no JAX, runs in < 1 min).
# For all modes:  $env:GA_TEST_MODES='normal,metacount,searchlibrium,random_params,stage2,stage2_metacount,stage3,stage3_metacount,stage6'
MODES_AVAILABLE = [
    "normal", "metacount", "searchlibrium", "random_params",
    "stage2", "stage2_metacount",
    "stage3", "stage3_metacount",
    "stage5",
    "stage6",
]
_env_modes = os.environ.get("GA_TEST_MODES", "").strip()
MODES_TO_RUN = (
    [m.strip().lower() for m in _env_modes.split(",") if m.strip()]
    if _env_modes else [
        "normal", "metacount", "searchlibrium", "random_params",
        "stage2", "stage2_metacount",
        "stage3", "stage3_metacount",
        "stage5",
        "stage6",
    ]
)

# ── synthetic dataset ─────────────────────────────────────────────────────────
RNG = np.random.default_rng(SEED)
N   = 600   # generous N for clean signal

# ── expanded feature set ─────────────────────────────────────────────────────
#  Collinear groups (e.g. age + age groups) keep the SA from selecting
#  redundant representations simultaneously.
FEAT = [
    # age — continuous or categorical age groups
    "age",       # raw age 18-75 (continuous)
    "age_z",     # standardised age
    "age_u30",   # dummy: under 30
    "age_30_60", # dummy: 30–59
    "age_60p",   # dummy: 60+
    # income
    "incp",      # standardised income (continuous)
    "incp_pos",  # above-median income dummy
    "incp_hi",   # top-quartile income dummy
    # vehicles
    "vehrd",     # vehicle count (continuous)
    "veh_per_pp",# vehicles per person
    "veh_0",     # zero-vehicle household dummy
    # household structure — continuous or categories
    "nprd",      # persons in HH
    "HHSIZE",    # HH size (continuous)
    "hhsize_2",  # exactly 2 persons
    "hhsize_3p", # 3+ persons
    "lone_hh",   # lives alone (HHSIZE==1)
    # person attributes
    "rlhp",      # relationship to head
    "is_female", # gender
    "worker",    # worker status
]

# ── base features (random) ────────────────────────────────────────────────────
_age      = RNG.uniform(18, 75, N)
_incp     = RNG.standard_normal(N)
_vehrd    = RNG.integers(0, 4, N).astype(float)
_nprd     = RNG.integers(1, 6, N).astype(float)
_rlhp     = RNG.integers(0, 2, N).astype(float)
_HHSIZE   = RNG.integers(1, 7, N).astype(float)
_is_female= RNG.integers(0, 2, N).astype(float)
_worker   = RNG.integers(0, 2, N).astype(float)

# ── derived features ──────────────────────────────────────────────────────────
_age_z      = (_age - _age.mean()) / _age.std()
_incp_pos   = (_incp > 0).astype(float)
_veh_per_pp = (_vehrd / np.maximum(_nprd, 1)).clip(0, 3)
_lone_hh    = (_HHSIZE == 1).astype(float)

# ── categorical breakdowns (allow SA to choose continuous vs grouped form) ────
_age_u30   = (_age < 30).astype(float)
_age_30_60 = ((_age >= 30) & (_age < 60)).astype(float)
_age_60p   = (_age >= 60).astype(float)
_hhsize_2  = (_HHSIZE == 2).astype(float)
_hhsize_3p = (_HHSIZE >= 3).astype(float)
_incp_hi   = (_incp > 1.0).astype(float)   # top ~16% for N(0,1)
_veh_0     = (_vehrd == 0).astype(float)

# ── Stage 5 / 6: persons table columns needed by prepare_mode_data ──────────
#  prepare_mode_data merges sexp, vehrd, nprd, WORKSTAT, STUDYING, incp, age5p,
#  CARLICENCE, indp from a persons table.  Map our synthetic variables:
_sexp      = np.where(_is_female == 1, 2, 1)          # 2=female, 1=male
_WORKSTAT  = (_worker * 2).astype(float)               # 2=FT worker, 0=non-worker
_age5p     = np.clip((_age / 5).astype(int), 0, 15)   # 5-yr age band index
_carlicence= (_vehrd > 0).astype(int).astype(float)   # licence ≈ has vehicle
_incp_band = np.clip(_incp * 3 + 8, 0, 20)            # scale N(0,1) → realistic band
_indp      = np.random.default_rng(SEED + 99).integers(1, 20, N).astype(float)

synth_persons = pd.DataFrame({
    "person_id":  np.arange(N),
    "vehrd":      _vehrd,
    "sexp":       _sexp,
    "nprd":       _nprd,
    "WORKSTAT":   _WORKSTAT,
    "STUDYING":   np.zeros(N),
    "incp":       _incp_band,
    "age5p":      _age5p,
    "CARLICENCE": _carlicence,
    "indp":       _indp,
})

# ── Stage 6 vehicle flags & structured mode choice (DA/SR/Transit/Walk/Bike) ─
_has_vehicle = (_vehrd > 0).astype(float)
_no_vehicle  = (_vehrd == 0).astype(float)
_mode_logit  = np.column_stack([
    np.zeros(N),                             # DA   (reference)
    0.3  * (_vehrd > 1).astype(float),       # SR   — multi-vehicle HH
    -0.8 * _vehrd + 0.6 * (1 - _worker),    # Transit — no car + non-worker
    -1.5 * _vehrd + 0.3 * _age_u30,         # Walk    — no car + young
    -2.0 * _vehrd + 0.5 * _age_u30,         # Bike    — no car + very young
]) + RNG.gumbel(0, 1, (N, 5))
_chosen_mode = np.array(['DA', 'SR', 'Transit', 'Walk', 'Bike'])[
    np.argmax(_mode_logit, axis=1)
]

# ── structured targets so SA has signal to chase ─────────────────────────────
#  Stage 1: trip counts driven by real predictors + noise
_mu_work   = np.exp(0.7 * _worker  - 0.012 * (_age - 40) + 0.25 * _veh_per_pp
                    + RNG.normal(0, 0.35, N))
_mu_edu    = np.exp(0.9 * _nprd    - 0.6   * _rlhp        + 0.35 * (1 - _worker)
                    + RNG.normal(0, 0.35, N))
_mu_social = np.exp(0.45 * _HHSIZE + 0.55  * _is_female   - 0.35 * _incp
                    + RNG.normal(0, 0.35, N))
_n_work    = RNG.poisson(np.clip(_mu_work,   0.1, 6))
_n_edu     = RNG.poisson(np.clip(_mu_edu,    0.1, 4))
_n_social  = RNG.poisson(np.clip(_mu_social, 0.1, 5))

#  Stage 4: timing period (0-5) driven by worker / age_z / gender / incp + Gumbel
#    0=early AM, 1=AM peak, 2=midday, 3=PM peak, 4=PM, 5=evening
_util = np.column_stack([
    np.zeros(N),                                                       # period 0 (base)
    1.4 * _worker  - 0.9  * _age_z,                                   # period 1 AM peak
    -0.7 * _worker + 0.5  * _incp_pos,                                # period 2 midday
    1.0  * _worker + 0.5  * _is_female,                               # period 3 PM peak
    0.4  * _is_female - 0.25 * _worker + 0.3 * _incp_pos,            # period 4 PM
    -0.5 * _worker + 0.4  * (1 - _is_female) - 0.3 * _age_z,        # period 5 evening
]) + RNG.gumbel(0, 1, (N, 6))
_period = np.argmax(_util, axis=1)

synth = pd.DataFrame({
    "obs_id":       np.arange(N),
    "person_id":    np.arange(N),
    "ptype":        RNG.choice(["FT worker", "PT worker", "Student", "Non-worker"], N),
    "purpose":      RNG.choice(["work", "edu", "social", "shopping"], N),
    # raw features
    "age":          _age,
    "age_z":        _age_z,
    "incp":         _incp,
    "incp_pos":     _incp_pos,
    "vehrd":        _vehrd,
    "veh_per_pp":   _veh_per_pp,
    "nprd":         _nprd,
    "rlhp":         _rlhp,
    "HHSIZE":       _HHSIZE,
    "is_female":    _is_female,
    "worker":       _worker,
    "lone_hh":      _lone_hh,
    # categorical breakdowns
    "age_u30":      _age_u30,
    "age_30_60":    _age_30_60,
    "age_60p":      _age_60p,
    "hhsize_2":     _hhsize_2,
    "hhsize_3p":    _hhsize_3p,
    "incp_hi":      _incp_hi,
    "veh_0":        _veh_0,
    # Stage 1 targets — structured signal
    "n_work_out":     _n_work,
    "n_edu_out":      _n_edu,
    "n_social_out":   _n_social,
    "n_shopping_out": RNG.poisson(0.6, N),   # no signal — acts as noise
    # Stage 4 target — structured signal
    "period":         _period,
    # extras
    "start_of_day": RNG.normal(7.5, 1.5, N).clip(4, 14),
    "end_of_day":   RNG.normal(17.5, 1.5, N).clip(14, 23),
    "B":            np.maximum(RNG.lognormal(4.5, 0.8, N), 60.0),
    "duration_min": np.maximum(RNG.lognormal(4.0, 0.7, N), 5.0),
    "tour_id":      np.arange(N),
    # Stage 3: activity duration columns
    "main_purpose": RNG.choice(["work", "edu", "social", "shopping"], N),
    "duration":     np.maximum(RNG.lognormal(4.0, 0.7, N), 5.0),
    # Stage 5 / 6: mode choice columns
    "has_vehicle":  _has_vehicle,
    "no_vehicle":   _no_vehicle,
    "chosen_mode":  _chosen_mode,
})

# ── Stage 5: trip table (prepare_mode_data needs a 'mode' or 'MAINMODE' column) ─
synth_trips = synth[["person_id", "purpose", "chosen_mode"]].copy()
synth_trips["mode"] = synth_trips["chosen_mode"]   # RAW_MODE_MAP maps these directly

# ── derived datasets for single-model (pooled) fitness ─────────────────────────
#  synth_pooled: total trips (work+edu+social) so Stage 1 uses ONE count model
#                instead of a separate model per purpose.
synth_pooled = synth.copy()
synth_pooled["ptype"]       = "all"
synth_pooled["n_total_out"] = (
    synth["n_work_out"] + synth["n_edu_out"] + synth["n_social_out"]
)

#  synth_timing: synth without the 'purpose' column so Stage 4 fitness uses
#                ALL N observations in one MNL/MXL instead of purpose subsets.
synth_timing = synth.drop(columns=["purpose"])

# ── collinear groups for expanded feature set ─────────────────────────────────
#  Groups prevent the SA from selecting redundant representations together.
_EXTRA_COLLINEAR = [
    ["age", "age_z", "age_u30", "age_30_60", "age_60p"],  # age: continuous vs groups
    ["incp", "incp_pos", "incp_hi"],                       # income: continuous vs dummies
    ["vehrd", "veh_per_pp", "veh_0"],                      # vehicles: continuous vs dummy
    ["HHSIZE", "lone_hh", "hhsize_2", "hhsize_3p"],        # HH size: continuous vs categories
]

# ── Stage 5 feature set (trip mode choice — candidates for GA selection) ──────
#  Mode candidates are the enrichments the GA can add/remove around the backbone.
#  All skim-derived columns (SOVTOLL_TIME__AM, DIST, etc.) default to 0 via
#  prepare_mode_data's standardised fallbacks — that's fine for a smoke test.
MODE_CANDS_S5 = [
    "has_vehicle", "no_vehicle", "vehrd", "CARLICENCE",
    "incp", "income_hi", "income_lo",
    "is_female", "sexp",
    "age_working", "age_older",
    "nprd", "n_workers_hh",
    "mandatory_role",
]
DEST_CANDS_S5: list[str] = []   # skim_df is empty → dest branch always skips
FEAT_S5 = MODE_CANDS_S5 + DEST_CANDS_S5   # full gene vector

# ── Stage 6 feature set (mode choice — includes vehicle binary flags) ─────────
#  Person attributes that have mapped utility terms in build_utility_matrix():
#    has_vehicle → hv, no_vehicle → nv, incp → incp, vehrd (via has_vehicle/no_vehicle),
#    is_female (via sexp), worker (via WORKSTAT proxy), age_u30/age_60p (via age5p proxy).
FEAT_S6 = [
    "incp", "incp_hi",
    "vehrd", "has_vehicle", "no_vehicle", "veh_0",
    "nprd", "HHSIZE", "is_female", "worker",
    "age_u30", "age_30_60", "age_60p",
    "rlhp", "lone_hh",
]
_EXTRA_COLLINEAR_S6 = _EXTRA_COLLINEAR + [
    ["has_vehicle", "no_vehicle", "veh_0"],   # vehicle status — pick at most one
]

# Merge with any pipeline-defined groups (keyed by stage)
def _merged_collinear(stage_key: int) -> list[list[str]]:
    base = STAGE_COLLINEAR_GROUPS.get(stage_key, [])
    return base + _EXTRA_COLLINEAR

# ── shared GA / SA kwargs ─────────────────────────────────────────────────────
_BASE_KWARGS = dict(
    pop_size      = POP_SIZE,
    n_generations = N_GENS,
    min_features  = 2,
    max_features  = len(FEAT),
    seed          = SEED,
    verbose       = True,
    fitness_metric= "mean(-BIC/n)",
)

# ── helper: print per-generation fitness table ────────────────────────────────

def _print_history_table(history: list[dict], mode: str) -> None:
    header = f"{'Gen':>4}  {'best_so_far':>12}  {'delta':>9}  {'n_feats':>7}  {'div/accept':>10}"
    print(f"\n  {header}", flush=True)
    print(f"  {'-'*len(header)}", flush=True)
    prev_bsf = None
    for h in history:
        bsf   = h.get("best_so_far", float("nan"))
        delta = (bsf - prev_bsf) if prev_bsf is not None else float("nan")
        div   = h.get("diversity", float("nan"))
        nf    = h.get("n_features", 0)
        gen   = h.get("generation", 0)
        # Arrow to highlight improvements
        arrow = " <--" if (delta is not None and np.isfinite(delta) and delta > 1e-8) else ""
        delta_str = f"{delta:+.6f}" if np.isfinite(delta) else "     ---"
        bsf_str   = f"{bsf:.6f}" if np.isfinite(bsf) else "    N/A"
        div_str = f"{div:>10.3f}" if np.isfinite(div) else "       ---"
        print(f"  {gen:>4}  {bsf_str:>12}  {delta_str:>9}  {nf:>7}  {div_str}{arrow}", flush=True)
        prev_bsf = bsf

# ── helper: verdict ───────────────────────────────────────────────────────────

def _verdict(history: list[dict]) -> tuple[bool, str]:
    bsf = [h.get("best_so_far", float("nan")) for h in history]
    finite = [v for v in bsf if np.isfinite(v) and v > -1e8]
    if len(finite) < 2:
        return False, f"only {len(finite)} valid fitness entries (estimator may have failed)"
    improved = any(finite[i] > finite[i - 1] + 1e-10 for i in range(1, len(finite)))
    start_bic = -finite[0] if finite else float("nan")
    end_bic   = -finite[-1] if finite else float("nan")
    delta_bic = start_bic - end_bic   # positive = BIC went down = improvement
    detail = (
        f"BIC/n start={start_bic:.4f}  end={end_bic:.4f}  "
        f"improvement={delta_bic:+.4f}  ({'improved' if improved else 'flat'})"
    )
    return improved, detail


# ── coefficient printing helpers ─────────────────────────────────────────────

def _print_feature_decisions(best_genes: np.ndarray,
                              rnd_genes: np.ndarray | None = None,
                              feat: list[str] | None = None) -> None:
    """Print every feature with its SA decision (IN / out).
    feat defaults to FEAT.  If rnd_genes is provided (random_params mode),
    selected features also show whether the coefficient was random or fixed.
    """
    _feat = feat if feat is not None else FEAT
    n_sel = int(best_genes.sum())
    maxw  = max(len(f) for f in _feat)
    print(f"\n  SA feature decisions  ({n_sel} of {len(_feat)} selected):", flush=True)
    print(f"  {'feature':{maxw}s}  decision", flush=True)
    print(f"  {'-'*(maxw + 20)}", flush=True)
    for i, (fname, g) in enumerate(zip(_feat, best_genes)):
        if g:
            if rnd_genes is not None:
                rnd = bool(rnd_genes[i]) if i < len(rnd_genes) else False
                label = "IN  [RANDOM coeff]" if rnd else "IN  [fixed  coeff]"
            else:
                label = "IN  [selected]"
        else:
            label = "out"
        print(f"  {fname:{maxw}s}  {label}", flush=True)
    print(flush=True)


def _print_best_coefs_normal(best_genes: np.ndarray) -> None:
    """Re-fit OrderedLogit per purpose; print per-purpose coefs + effect direction."""
    selected = [f for f, g in zip(FEAT, best_genes) if g == 1]
    if not selected:
        return
    try:
        from stage1_activity_frequency import OrderedLogit
    except ImportError as e:
        print(f"  [coefs] Cannot import OrderedLogit: {e}", flush=True)
        return

    print(f"\n  Coefficients (OrderedLogit, Stage 1 — one model per purpose)", flush=True)
    for purpose in ["work", "edu", "social"]:
        y_col = f"n_{purpose}_out"
        if y_col not in synth.columns:
            continue
        sub = synth[selected + [y_col]].dropna()
        if len(sub) < 30:
            continue
        y = sub[y_col].values.astype(int)
        if len(np.unique(y)) < 2:
            continue
        X = sub[selected].fillna(0).values.astype(float)
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        X = (X - mu) / sd
        try:
            m = OrderedLogit()
            m.fit(X, y)
        except Exception as e:
            print(f"  {purpose}: fit failed ({e})", flush=True)
            continue
        print(f"\n  [{purpose.upper()}]  N={len(sub)}", flush=True)
        print(f"  {'variable':14s}  {'coef':>8s}  direction", flush=True)
        print(f"  {'-'*36}", flush=True)
        for name, coef in zip(selected, m.coef_):
            dirn = "more trips" if coef > 0 else "fewer trips"
            print(f"  {name:14s}  {coef:+8.4f}  {dirn}", flush=True)
    print(flush=True)


def _print_best_coefs_metacount(best_genes: np.ndarray) -> None:
    """Re-fit MetaCount NB2 per purpose; print per-purpose coefs + effect direction."""
    selected = [f for f, g in zip(FEAT, best_genes) if g == 1]
    if not selected:
        return
    try:
        from ga_external_estimators import (
            _create_metacount_builder, _fit_metacount_manual_model,
            _extract_metacount_coefs, _ensure_id,
        )
    except ImportError as e:
        print(f"  [coefs] import failed: {e}", flush=True)
        return

    df = _ensure_id(synth.copy(), "obs_id")
    print(f"\n  Coefficients (MetaCount NB2, Stage 1 — one model per purpose)", flush=True)
    for purpose in ["work", "edu", "social"]:
        y_col = f"n_{purpose}_out"
        if y_col not in df.columns:
            continue
        avail = [c for c in selected if c in df.columns]
        sub   = df[["obs_id", y_col] + avail].dropna()
        if len(sub) < 30 or not avail:
            continue
        try:
            spec = {"fixed_terms": avail, "rdm_terms": [], "rdm_cor_terms": [],
                    "grouped_terms": [], "hetro_in_means": [], "zi_terms": [],
                    "membership_terms": [], "dispersion": 1, "latent_classes": 1}
            builder = _create_metacount_builder(sub, id_col="obs_id", y_col=y_col, family="nb2")
            fitted  = _fit_metacount_manual_model(builder, spec, R=1, family="nb2")
            rows    = _extract_metacount_coefs(fitted)
        except Exception as e:
            print(f"  {purpose}: fit failed ({e})", flush=True)
            continue
        print(f"\n  [{purpose.upper()}]  N={len(sub)}", flush=True)
        if rows:
            print(f"  {'param':14s}  {'coef':>8s}  {'std_err':>8s}  {'t':>6s}  direction", flush=True)
            print(f"  {'-'*52}", flush=True)
            for r in rows:
                t   = r.get("t_stat", float("nan"))
                sig = "*" if np.isfinite(t) and abs(t) > 1.96 else " "
                dirn = "higher count" if r["value"] > 0 else "lower count"
                print(f"  {r['param']:14s}  {r['value']:+8.4f}  "
                      f"{r['std_err']:8.4f}  {t:6.2f}{sig}  {dirn}", flush=True)
        else:
            print(f"  no coefficients extracted", flush=True)
    print(flush=True)


def _print_best_coefs_searchlibrium(best_genes: np.ndarray) -> None:
    """OLS(period ~ selected_features) per purpose — gives non-zero directional
    coefficients for person-level attributes.  MNL with person-level attributes
    produces zero coefficients because features are constant across alternatives.
    OLS is used here for interpretability; the SA still uses MNL for selection.
    """
    selected = [f for f, g in zip(FEAT, best_genes) if g == 1]
    if not selected:
        return

    print(f"\n  Coefficients (OLS on period, Stage 4 — one model per purpose)", flush=True)
    print(f"  Note: OLS displayed for interpretability; SA fitness used MNL.", flush=True)
    for purpose in ["work", "edu", "social"]:
        sub   = synth[synth["purpose"] == purpose].copy() if "purpose" in synth.columns else synth
        avail = [f for f in selected if f in sub.columns]
        if len(sub) < 20 or not avail:
            continue
        sub = sub[avail + ["period"]].dropna()
        if len(sub) < 20:
            continue
        X = sub[avail].fillna(0).values.astype(float)
        y = sub["period"].values.astype(float)
        # standardise X for comparable magnitudes
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        X_std = (X - mu) / sd
        # OLS via least-squares with an intercept
        X_aug = np.column_stack([np.ones(len(X_std)), X_std])
        coefs_full, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        coefs = coefs_full[1:]    # drop intercept
        y_hat = X_aug @ coefs_full
        r_sq  = 1 - np.var(y - y_hat) / max(np.var(y), 1e-12)
        print(f"\n  [{purpose.upper()}]  N={len(sub)}  R²={r_sq:.3f}", flush=True)
        print(f"  {'variable':14s}  {'std-coef':>8s}  direction", flush=True)
        print(f"  {'-'*36}", flush=True)
        for name, coef in zip(avail, coefs):
            dirn = "later timing" if coef > 0 else "earlier timing"
            print(f"  {name:14s}  {coef:+8.4f}  {dirn}", flush=True)
    print(flush=True)


def _print_best_coefs_mixed_logit(best_genes: np.ndarray) -> None:
    """OLS(period ~ features) per purpose showing mean effect (direction) and
    cross-person std of predictions (heterogeneity proxy).  The SA used
    MixedLogit during selection; OLS is shown here because person-level
    attributes are unidentified in standard MNL/MXL without isvars.
    """
    selected = [f for f, g in zip(FEAT, best_genes) if g == 1]
    if not selected:
        return

    print(f"\n  Coefficients (OLS on period, Stage 4 — per purpose, random-params proxy)", flush=True)
    print(f"  Note: SA fitted MixedLogit for selection; OLS shown for interpretability.", flush=True)
    for purpose in ["work", "edu", "social"]:
        sub   = synth[synth["purpose"] == purpose].copy() if "purpose" in synth.columns else synth
        avail = [f for f in selected if f in sub.columns]
        if len(sub) < 30 or not avail:
            continue
        sub = sub[avail + ["period"]].dropna()
        if len(sub) < 30:
            continue
        X = sub[avail].fillna(0).values.astype(float)
        y = sub["period"].values.astype(float)
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        X_std = (X - mu) / sd
        X_aug = np.column_stack([np.ones(len(X_std)), X_std])
        coefs_full, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        coefs = coefs_full[1:]
        # Heterogeneity proxy: std of each variable's contribution across persons
        # = |beta_k| * std(x_k)  — after standardisation this is just |beta_k|
        het_sd = np.abs(coefs)   # already on std-x scale
        r_sq = 1 - np.var(y - X_aug @ coefs_full) / max(np.var(y), 1e-12)
        print(f"\n  [{purpose.upper()}]  N={len(sub)}  R²={r_sq:.3f}", flush=True)
        print(f"  {'variable':14s}  {'std-coef':>8s}  {'|effect|':>8s}  direction / heterogeneity", flush=True)
        print(f"  {'-'*56}", flush=True)
        for name, coef, hs in zip(avail, coefs, het_sd):
            dirn = "later timing" if coef > 0 else "earlier timing"
            het  = "heterogeneous" if hs > 0.05 else "homogeneous"
            print(f"  {name:14s}  {coef:+8.4f}  {hs:8.4f}  RANDOM — {dirn}, {het}", flush=True)
    print(flush=True)


# ── mode runners ─────────────────────────────────────────────────────────────

def _sa_run(fitness_fn, label: str, collinear_key=1,
            feat=None, collinear_extra=None) -> tuple[list[dict], str, np.ndarray]:
    """Shared SA runner — wraps fitness with _quiet() and prints results.
    feat: feature list (defaults to FEAT).
    collinear_extra: extra collinear groups (defaults to _EXTRA_COLLINEAR).
    """
    _feat   = feat if feat is not None else FEAT
    _extra  = collinear_extra if collinear_extra is not None else _EXTRA_COLLINEAR
    _collin = STAGE_COLLINEAR_GROUPS.get(collinear_key, []) + _extra

    # ── track each new best during the SA search and print the proposed set ──
    # Also capture the very first evaluation so we can inject a pre-SA baseline
    # entry into history — this ensures _verdict() sees the true starting point
    # even when the SA finds its optimum within the first generation's steps.
    _tracker   = {'best': -np.inf, 'n_evals': 0}
    _first_eval = {'fitness': None, 'n_genes': None}

    def _tracked(genes: np.ndarray) -> float:
        result = fitness_fn(genes)
        _tracker['n_evals'] += 1
        if _first_eval['fitness'] is None:
            _first_eval['fitness'] = result
            _first_eval['n_genes'] = int(genes.sum())
        if result > _tracker['best'] and result > -1e8:
            _tracker['best'] = result
            active = [f for f, g in zip(_feat, genes) if g]
            bic_n  = -result
            print(f"    #{_tracker['n_evals']:>4}  BIC/n={bic_n:.4f}  proposed: {active}", flush=True)
        return result

    sa = SimulatedAnnealing(
        len(_feat), _tracked,
        feature_names    = _feat,
        collinear_groups = _collin,
        **{**_BASE_KWARGS, 'max_features': len(_feat)},
    )
    t0 = time.perf_counter()
    best, history, _ = sa.run_with_restarts(n_restarts=N_RESTARTS)
    # Prepend pre-SA baseline so _verdict() can see the true starting fitness,
    # even when the optimum is found before the first generation log entry.
    if _first_eval['fitness'] is not None and history:
        pre_entry = {
            'generation': -1,
            'best_so_far': _first_eval['fitness'],
            'n_features':  _first_eval['n_genes'],
            'diversity':   float('nan'),
            'accept_rate': float('nan'),
        }
        history = [pre_entry] + list(history)
    elapsed = time.perf_counter() - t0
    active = [f for f, g in zip(_feat, best.genes) if g == 1]
    print(f"\n  Best features : {active}", flush=True)
    print(f"  Best fitness  : {best.fitness:.6f}  (BIC/n={-best.fitness:.4f})", flush=True)
    print(f"  Cache hits    : {sa.cache_hits}  |  Elapsed: {elapsed:.1f}s", flush=True)
    _print_feature_decisions(best.genes, feat=_feat)
    _print_history_table(history, label)
    return history, f"{label} | {elapsed:.1f}s", best.genes


def run_normal() -> tuple[list[dict], str]:
    """SA + default ordered-logit fitness on Stage 1, separate model per purpose."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: normal  (SA + ordered-logit, Stage 1, per purpose)", flush=True)
    print("=" * 66, flush=True)

    purposes = ["work", "edu", "social"]
    ptypes   = synth["ptype"].unique().tolist()

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _stage1_fitness_aggregate(genes, synth, FEAT, purposes, ptypes)

    history, label, best_genes = _sa_run(fitness, "normal | SA | Stage1-ordered-logit", collinear_key=1)
    _print_best_coefs_normal(best_genes)
    return history, label


def run_metacount() -> tuple[list[dict], str]:
    """SA + MetaCount NB2 fitness on Stage 1."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: metacount  (SA + MetaCount NB2, Stage 1)", flush=True)
    print("=" * 66, flush=True)
    print("  Initialising MetaCount/JAX (may take 1-2 min on first run) ...", flush=True)
    try:
        from ga_external_estimators import _METACOUNT_OK, stage1_fitness_metacount as _mc_fit
    except Exception as exc:
        print(f"  [SKIP] import failed: {exc}", flush=True)
        return [], f"metacount | SKIPPED: {exc}"
    if not _METACOUNT_OK:
        print("  [SKIP] metacountregressor not installed", flush=True)
        return [], "metacount | SKIPPED: metacountregressor not installed"
    print("  done.", flush=True)

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _mc_fit(genes, synth, FEAT, ["work", "edu", "social"])

    history, label, best_genes = _sa_run(fitness, "metacount | SA | Stage1-NB2", collinear_key=1)
    _print_best_coefs_metacount(best_genes)
    return history, label


def run_searchlibrium() -> tuple[list[dict], str]:
    """SA + SearchLibrium MNL fitness on Stage 4."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: searchlibrium  (SA + SearchLibrium MNL, Stage 4)", flush=True)
    print("=" * 66, flush=True)
    print("  Initialising SearchLibrium (may take 1-2 min on first run) ...", flush=True)
    try:
        from ga_external_estimators import _SEARCHLIB_OK, stage4_fitness_searchlibrium as _sl_fit
    except Exception as exc:
        print(f"  [SKIP] import failed: {exc}", flush=True)
        return [], f"searchlibrium | SKIPPED: {exc}"
    if not _SEARCHLIB_OK:
        print("  [SKIP] SearchLibrium not installed", flush=True)
        return [], "searchlibrium | SKIPPED: SearchLibrium not installed"
    print("  done.", flush=True)

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _sl_fit(genes, synth, FEAT, ["work", "edu", "social"], period_col="period", n_alts=6)

    history, label, best_genes = _sa_run(fitness, "searchlibrium | SA | Stage4-MNL", collinear_key=4)
    _print_best_coefs_searchlibrium(best_genes)
    return history, label


def run_random_params() -> tuple[list[dict], str]:
    """SA + SearchLibrium MixedLogit (random parameters) on Stage 4."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: random_params  (SA + MixedLogit random coefficients, Stage 4)", flush=True)
    print("=" * 66, flush=True)
    print("  Initialising SearchLibrium MixedLogit ...", flush=True)
    try:
        from ga_external_estimators import (
            _SEARCHLIB_OK,
            stage4_fitness_searchlibrium_mixed as _mxl_fit,
        )
    except Exception as exc:
        print(f"  [SKIP] import failed: {exc}", flush=True)
        return [], f"random_params | SKIPPED: {exc}"
    if not _SEARCHLIB_OK:
        print("  [SKIP] SearchLibrium not installed", flush=True)
        return [], "random_params | SKIPPED: SearchLibrium not installed"
    print("  done.", flush=True)
    print("  Note: each eval fits a Mixed Logit (n_draws=30) — slower than MNL", flush=True)

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _mxl_fit(
                genes, synth, FEAT, ["work", "edu", "social"],
                period_col="period", n_alts=6, n_draws=30,
            )

    history, label, best_genes = _sa_run(fitness, "random_params | SA | Stage4-MixedLogit", collinear_key=4)
    _print_best_coefs_mixed_logit(best_genes)
    return history, label


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — OLS day duration (start_of_day / end_of_day / B)
# ─────────────────────────────────────────────────────────────────────────────

_TARGETS_S2 = ["start_of_day", "end_of_day", "B"]


def _print_best_coefs_stage2(best_genes: np.ndarray) -> None:
    """OLS on each Stage 2 target; print standardised coefficients."""
    selected = [f for f, g in zip(FEAT, best_genes) if g == 1]
    if not selected:
        return
    print(f"\n  Coefficients (OLS, Stage 2 — day duration targets)", flush=True)
    for tgt in _TARGETS_S2:
        sub = synth[selected + [tgt]].dropna()
        if len(sub) < 20:
            continue
        X = sub[selected].fillna(0).values.astype(float)
        y = sub[tgt].values.astype(float)
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        X_std = (X - mu) / sd
        X_aug = np.column_stack([np.ones(len(X_std)), X_std])
        coefs_full, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        coefs = coefs_full[1:]
        r_sq = 1 - np.var(y - X_aug @ coefs_full) / max(np.var(y), 1e-12)
        print(f"\n  [{tgt.upper()}]  N={len(sub)}  R²={r_sq:.3f}", flush=True)
        print(f"  {'variable':14s}  {'std-coef':>9s}  direction", flush=True)
        print(f"  {'-'*38}", flush=True)
        for name, coef in zip(selected, coefs):
            dirn = f"higher {tgt}" if coef > 0 else f"lower {tgt}"
            print(f"  {name:14s}  {coef:+9.4f}  {dirn}", flush=True)
    print(flush=True)


def run_stage2() -> tuple[list[dict], str]:
    """SA + OLS fitness on Stage 2 day duration targets."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: stage2  (SA + OLS, Stage 2 — start/end of day + time budget)", flush=True)
    print("=" * 66, flush=True)

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _stage2_fitness_aggregate(genes, synth, FEAT, _TARGETS_S2)

    history, label, best_genes = _sa_run(fitness, "stage2 | SA | Stage2-OLS", collinear_key=2)
    _print_best_coefs_stage2(best_genes)
    return history, label


def run_stage2_metacount() -> tuple[list[dict], str]:
    """SA + MetaCount gaussian/duration fitness on Stage 2."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: stage2_metacount  (SA + MetaCount, Stage 2 day duration)", flush=True)
    print("=" * 66, flush=True)
    try:
        from ga_external_estimators import _METACOUNT_OK, stage2_fitness_metacount as _mc2_fit
    except Exception as exc:
        print(f"  [SKIP] import failed: {exc}", flush=True)
        return [], f"stage2_metacount | SKIPPED: {exc}"
    if not _METACOUNT_OK:
        print("  [SKIP] metacountregressor not installed", flush=True)
        return [], "stage2_metacount | SKIPPED: metacountregressor not installed"

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _mc2_fit(genes, synth, FEAT, _TARGETS_S2, use_sim_anneal=True)

    history, label, best_genes = _sa_run(fitness, "stage2_metacount | SA | Stage2-MetaCount", collinear_key=2)
    _print_best_coefs_stage2(best_genes)   # reuse OLS display for comparison
    return history, label


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — BudgetLognormal activity duration
# ─────────────────────────────────────────────────────────────────────────────

_PURPOSES_S3 = ["work", "edu", "social"]


def _print_best_coefs_stage3(best_genes: np.ndarray) -> None:
    """OLS on log(duration) per purpose; print standardised coefs."""
    selected = [f for f, g in zip(FEAT, best_genes) if g == 1]
    if not selected:
        return
    print(f"\n  Coefficients (OLS on log-duration, Stage 3 — per purpose)", flush=True)
    for purpose in _PURPOSES_S3:
        sub = synth[synth["main_purpose"] == purpose][selected + ["duration"]].dropna()
        if len(sub) < 20:
            continue
        X = sub[selected].fillna(0).values.astype(float)
        y = np.log(np.maximum(sub["duration"].values, 1.0))
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        X_std = (X - mu) / sd
        X_aug = np.column_stack([np.ones(len(X_std)), X_std])
        coefs_full, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        coefs = coefs_full[1:]
        r_sq = 1 - np.var(y - X_aug @ coefs_full) / max(np.var(y), 1e-12)
        print(f"\n  [{purpose.upper()}]  N={len(sub)}  R²(log-dur)={r_sq:.3f}", flush=True)
        print(f"  {'variable':14s}  {'std-coef':>9s}  direction", flush=True)
        print(f"  {'-'*38}", flush=True)
        for name, coef in zip(selected, coefs):
            dirn = "longer duration" if coef > 0 else "shorter duration"
            print(f"  {name:14s}  {coef:+9.4f}  {dirn}", flush=True)
    print(flush=True)


def run_stage3() -> tuple[list[dict], str]:
    """SA + BudgetLognormal fitness on Stage 3 activity duration."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: stage3  (SA + BudgetLognormal, Stage 3 — activity duration)", flush=True)
    print("=" * 66, flush=True)

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _stage3_fitness_aggregate(genes, synth, FEAT, _PURPOSES_S3, "main_purpose")

    history, label, best_genes = _sa_run(fitness, "stage3 | SA | Stage3-Lognormal", collinear_key=3)
    _print_best_coefs_stage3(best_genes)
    return history, label


def run_stage3_metacount() -> tuple[list[dict], str]:
    """SA + MetaCount lognormal fitness on Stage 3 activity duration."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: stage3_metacount  (SA + MetaCount, Stage 3 activity duration)", flush=True)
    print("=" * 66, flush=True)
    try:
        from ga_external_estimators import _METACOUNT_OK, stage3_fitness_metacount as _mc3_fit
    except Exception as exc:
        print(f"  [SKIP] import failed: {exc}", flush=True)
        return [], f"stage3_metacount | SKIPPED: {exc}"
    if not _METACOUNT_OK:
        print("  [SKIP] metacountregressor not installed", flush=True)
        return [], "stage3_metacount | SKIPPED: metacountregressor not installed"

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _mc3_fit(
                genes, synth, FEAT, _PURPOSES_S3,
                purp_col="main_purpose", dur_col="duration",
                use_sim_anneal=True,
            )

    history, label, best_genes = _sa_run(fitness, "stage3_metacount | SA | Stage3-MetaCount", collinear_key=3)
    _print_best_coefs_stage3(best_genes)   # OLS display for comparison
    return history, label


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — NestedLogit mode choice
# ─────────────────────────────────────────────────────────────────────────────

_PURPOSES_S6 = ["work", "edu", "social"]


def _print_best_coefs_stage6(best_genes: np.ndarray) -> None:
    """Mode share by purpose for chosen_mode, plus OLS on mode-code proxy."""
    selected = [f for f, g in zip(FEAT_S6, best_genes) if g == 1]
    if not selected:
        return
    print(f"\n  Mode shares per purpose (Stage 6 diagnosis):", flush=True)
    mode_order = ['DA', 'SR', 'Transit', 'Walk', 'Bike']
    for purpose in _PURPOSES_S6:
        sub = synth[synth["purpose"] == purpose]
        if len(sub) < 5:
            continue
        shares = sub["chosen_mode"].value_counts(normalize=True)
        row = "  ".join(
            f"{m}:{shares.get(m, 0.0)*100:4.1f}%" for m in mode_order
        )
        print(f"  [{purpose.upper():8s}]  {row}", flush=True)
    print(f"\n  OLS(mode_code ~ selected, Stage 6 — per purpose):", flush=True)
    mode_idx = {m: i for i, m in enumerate(mode_order)}
    for purpose in _PURPOSES_S6:
        sub = synth[synth["purpose"] == purpose].copy()
        avail = [f for f in selected if f in sub.columns]
        if len(sub) < 20 or not avail:
            continue
        sub["_mc"] = sub["chosen_mode"].map(mode_idx).fillna(0).astype(float)
        X = sub[avail].fillna(0).values.astype(float)
        y = sub["_mc"].values
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        X_aug = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        coefs_full, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        coefs = coefs_full[1:]
        r_sq = 1 - np.var(y - X_aug @ coefs_full) / max(np.var(y), 1e-12)
        print(f"\n  [{purpose.upper()}]  N={len(sub)}  R²(mode-code)={r_sq:.3f}", flush=True)
        print(f"  {'variable':14s}  {'std-coef':>9s}  direction", flush=True)
        print(f"  {'-'*38}", flush=True)
        for name, coef in zip(avail, coefs):
            dirn = "higher mode (Walk/Bike/Transit)" if coef > 0 else "lower mode (DA/SR)"
            print(f"  {name:14s}  {coef:+9.4f}  {dirn}", flush=True)
    print(flush=True)


def run_stage6() -> tuple[list[dict], str]:
    """SA + NestedLogit mode choice fitness on Stage 6."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: stage6  (SA + NestedLogitModeModel, Stage 6 — mode choice)", flush=True)
    print("=" * 66, flush=True)
    # Verify enough mode diversity per purpose before running
    for purpose in _PURPOSES_S6:
        sub = synth[synth["purpose"] == purpose]
        n_modes = sub["chosen_mode"].nunique()
        print(f"  [{purpose}]  N={len(sub)},  unique modes={n_modes},"
              f"  shares={dict(sub['chosen_mode'].value_counts())}", flush=True)

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _stage6_fitness_aggregate(genes, synth, FEAT_S6, _PURPOSES_S6)

    history, label, best_genes = _sa_run(
        fitness, "stage6 | SA | Stage6-NestedLogit",
        collinear_key=6,
        feat=FEAT_S6,
        collinear_extra=_EXTRA_COLLINEAR_S6,
    )
    _print_best_coefs_stage6(best_genes)
    return history, label


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — trip-level mode choice (NestedLogitModeModel via prepare_mode_data)
# ─────────────────────────────────────────────────────────────────────────────

_PURPOSES_S5 = ["work", "edu", "social"]


def _print_best_coefs_stage5(best_genes: np.ndarray) -> None:
    """Mode share by purpose for synth_trips, plus OLS on mode-code proxy."""
    selected = [f for f, g in zip(FEAT_S5, best_genes) if g == 1]
    mode_order = ["DA", "SR", "Transit", "Walk", "Bike"]
    print(f"\n  Mode shares per purpose (Stage 5 trip diagnosis):", flush=True)
    for purpose in _PURPOSES_S5:
        sub = synth_trips[synth_trips["purpose"] == purpose]
        if len(sub) < 5:
            continue
        shares = sub["chosen_mode"].value_counts(normalize=True)
        row = "  ".join(
            f"{m}:{shares.get(m, 0.0)*100:4.1f}%" for m in mode_order
        )
        print(f"  [{purpose.upper():8s}]  {row}", flush=True)
    if selected:
        print(f"\n  Mode candidates selected by SA: {selected}", flush=True)
    print(flush=True)


def run_stage5() -> tuple[list[dict], str]:
    """SA + Stage 5 trip-mode fitness (NestedLogitModeModel via prepare_mode_data)."""
    print("\n" + "=" * 66, flush=True)
    print("MODE: stage5  (SA + Stage5 trip mode, NestedLogitModeModel)", flush=True)
    print("=" * 66, flush=True)

    _n_mode = len(MODE_CANDS_S5)
    _n_dest = len(DEST_CANDS_S5)
    print(
        f"  Candidates: {_n_mode} mode + {_n_dest} dest = {_n_mode + _n_dest} genes",
        flush=True,
    )

    def fitness(genes: np.ndarray) -> float:
        with _quiet():
            return _stage5_trip_fitness_aggregate(
                genes           = genes,
                n_mode          = _n_mode,
                mode_candidates = MODE_CANDS_S5,
                dest_candidates = DEST_CANDS_S5,
                purposes        = _PURPOSES_S5,
                trips           = synth_trips,
                persons         = synth_persons,
                landuse         = pd.DataFrame(),
                skim_df         = pd.DataFrame(),
            )

    history, label, best_genes = _sa_run(
        fitness,
        "stage5 | SA | Stage5-trip-NestedLogit",
        collinear_key="5_mode",
        feat=FEAT_S5,
        collinear_extra=[],
    )
    _print_best_coefs_stage5(best_genes)
    return history, label


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 66, flush=True)
    print("GA/SA Fitness Progression Test", flush=True)
    print(f"  N features    : {len(FEAT)}  {FEAT}", flush=True)
    print(f"  N features S5 : {len(FEAT_S5)}  {FEAT_S5}", flush=True)
    print(f"  N features S6 : {len(FEAT_S6)}  {FEAT_S6}", flush=True)
    print(f"  Budget        : pop={POP_SIZE}, gens={N_GENS}, restarts={N_RESTARTS}", flush=True)
    print(f"  Synthetic N   : {N}", flush=True)
    print(f"  Modes to run  : {MODES_TO_RUN}", flush=True)
    print(f"  Tip: $env:GA_TEST_MODES='normal,stage2,stage3,stage6'", flush=True)
    print("=" * 66, flush=True)

    runners = {
        "normal":           run_normal,
        "metacount":        run_metacount,
        "searchlibrium":    run_searchlibrium,
        "random_params":    run_random_params,
        "stage2":           run_stage2,
        "stage2_metacount": run_stage2_metacount,
        "stage3":           run_stage3,
        "stage3_metacount": run_stage3_metacount,
        "stage5":           run_stage5,
        "stage6":           run_stage6,
    }

    results: dict[str, tuple[list, str]] = {}
    for mode in MODES_TO_RUN:
        if mode not in runners:
            print(f"\n[WARN] Unknown mode '{mode}' — skipping", flush=True)
            continue
        try:
            history, label = runners[mode]()
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {mode} raised an exception:", flush=True)
            traceback.print_exc()
            history, label = [], f"{mode} | ERROR: {exc}"
        results[mode] = (history, label)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 66, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 66, flush=True)
    all_pass = True
    rows = []
    for mode, (history, label) in results.items():
        if not history:
            passed, detail = False, "no history (skipped or error)"
        else:
            passed, detail = _verdict(history)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        rows.append({"mode": mode, "status": status, "detail": detail, "label": label})
        print(f"  [{status}]  {mode:<15} {detail}", flush=True)

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"), flush=True)
    print("=" * 66, flush=True)

    # Optional: print fitness comparison table across modes
    if len(results) > 1 and any(h for h, _ in results.values()):
        print("\nPer-generation best_so_far comparison (BIC/n, lower = better)")
        all_gens = max(
            (len(h) for h, _ in results.values() if h), default=0
        )
        cols = ["gen"] + [m for m in MODES_TO_RUN if results[m][0]]
        df_data: dict[str, list] = {"gen": list(range(all_gens))}
        for mode in MODES_TO_RUN:
            history, _ = results[mode]
            if not history:
                continue
            # pad with last value if shorter
            bsf = [h.get("best_so_far", float("nan")) for h in history]
            while len(bsf) < all_gens:
                bsf.append(bsf[-1] if bsf else float("nan"))
            df_data[mode] = [-v for v in bsf]   # negate: show BIC/n (positive, lower=better)
        cmp_df = pd.DataFrame(df_data)
        print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
