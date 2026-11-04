"""
stage3_activity_duration.py
============================
Stage 3 of the activity-based model pipeline.

Predicts the DURATION (minutes) of each individual activity for every
person, constrained so durations fit both the day budget B and the day window
[start_of_day, end_of_day] predicted by Stage 2.

Pipeline position
-----------------
    Stage 1: n_{purpose}_out  (how many activities)
    Stage 2: start_of_day, end_of_day, B  (when and total budget)
        → Stage 3: activity duration      ← YOU ARE HERE
        → Stage 4: schedule placement
        → Stage 5: destination choice
        → Stage 6: mode choice

Model structure
---------------
    For each purpose a budget-constrained lognormal is fitted:
        log(duration_i) ~ N(X_i @ beta, sigma^2)
    subject to: sum(duration_i for person p) <= min(B_p, day-window room)

    Estimation uses the tour_duration_model.py approach (already built).
    Prediction expands each person into one row per activity, runs the
    lognormal draw, then reconciles the person's total with the smaller of B
    and the time their predicted day window leaves for activities
    (window minus one placement travel allowance per activity).

Outputs per person
------------------
    One row per activity instance with columns:
        person_id, activity_seq, purpose, predicted_duration_min,
        start_of_day, end_of_day, B, n_tours, OTAZ
    (ready for Stage 4 schedule placement)

Usage
-----
    from stage3_activity_duration import run_stage3

    out      = Synth_Connect().process_survey(save=False)
    synth_s2 = pd.read_csv('saved_data/stage2_output.csv')

    activities, models = run_stage3(out["tours"], out["persons"], synth_s2)
    activities.to_csv(os.path.join(_OUT, 'stage3_output.csv'), index=False)
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
import sys
from typing import Optional
from scipy.stats import norm

_SEARCHLIBRIUM_SRC = os.environ.get(
    'SEARCHLIBRIUM_SOURCE', r'C:\Users\ahernz\source\SearchLibrium\src'
)
if os.path.isdir(_SEARCHLIBRIUM_SRC) and _SEARCHLIBRIUM_SRC not in sys.path:
    sys.path.insert(0, _SEARCHLIBRIUM_SRC)

try:
    from SearchLibrium import MDCEVModel
    _MDCEV_AVAILABLE = True
except Exception:
    MDCEVModel = None
    _MDCEV_AVAILABLE = False

try:
    from SearchLibrium import RandomRegret
    _RRM_AVAILABLE = True
except Exception:
    RandomRegret = None
    _RRM_AVAILABLE = False


def _install_searchlibrium_pickle_support() -> None:
    """Make installed SearchLibrium models portable across Python processes."""
    if not _RRM_AVAILABLE:
        return
    try:
        from SearchLibrium._choice_model import DiscreteChoiceModel
    except Exception:
        return
    if '__getstate__' in DiscreteChoiceModel.__dict__:
        return

    def _getstate(model):
        state = model.__dict__.copy()
        state.pop('backend', None)
        state['_backend_jax'] = bool(state.get('_jax', True))
        return state

    def _setstate(model, state):
        backend_jax = bool(state.pop('_backend_jax', True))
        model.__dict__.update(state)
        from SearchLibrium._choice_model import get_backend
        model.backend = get_backend(backend_jax)

    DiscreteChoiceModel.__getstate__ = _getstate
    DiscreteChoiceModel.__setstate__ = _setstate


_install_searchlibrium_pickle_support()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PURPOSES = [
    'work', 'edu', 'social', 'escort',
    'shopping', 'othmaint', 'othdiscr', 'eat',
]

ACTIVITY_COLS = [f'n_{p}_out' for p in PURPOSES]

# Valid person types per purpose — mirrors Stage 1 VALID_PTYPES_FOR_PURPOSE.
# Preschool excluded from work; school-age children excluded from adult-only purposes.
# NOTE: no per-purpose ptype whitelist any more.  Each purpose's duration
# model used to be fitted only on a hand-listed set of person types, but every
# activity the pipeline predicts (Stage 1 draws shopping for a child as
# readily as for a retiree) has to be covered.  Fitting on a truncated
# population made the model reproduce that population's mean and then
# undershoot every episode outside it — shopping by ~60%, escort/social/
# othmaint/eat by ~45% in sample.  Person type enters as a dummy instead, so
# sub-population differences are estimated rather than assumed away.

# Duration floors / ceilings per purpose (minutes).
# Upper bounds are SOFT — the model can predict above them (lognormal tail),
# and the budget rescaling may stretch up to 1.5× the bound before clipping.
# Work raised to 840 min (14 h) so that shift workers, overtime, and
# continuous-day workers are captured in the training distribution; the soft
# ceiling becomes 1260 min.  Observed HHTS work durations reach ~900 min.
# Edu raised to 600 min (10 h) to capture university full-day timetables,
# TAFE block-days, and after-school programs; soft ceiling = 900 min.
DURATION_BOUNDS = {
    'work':      (30,  840),   # 14 h max — covers shift workers and long days
    'edu':       (30,  600),   # 10 h max — covers full uni/school days
    'social':    (15,  300),
    'escort':    (10,  180),   # widened from (5, 120)
    'shopping':  (5,   180),
    'othmaint':  (5,   120),
    'othdiscr':  (5,   180),
    'eat':       (10,  120),
}

COEF_CLIP_LO = -20.0
COEF_CLIP_HI = 20.0
INTERCEPT_CLIP_LO = -1000.0
INTERCEPT_CLIP_HI = 1000.0


# =============================================================================
# EXPECTED COEFFICIENT SIGNS (behavioural priors for duration models)
# =============================================================================
# Positive coefficient = LONGER predicted duration (AFT-lognormal: eta enters
# log(duration) additively, and z-score standardisation never flips a sign).
# Convention mirrors stage5 _check_behavioural_signs: each entry is
#   feature -> (op, purposes-or-None, note)
# with purposes None = applies to every purpose. ONLY features with a solid
# behavioural prior are listed — everything else is still REPORTED by
# check_stage3_signs (pass=None) but never penalised. Deliberately excluded:
#   start_hr / log_dist (commute-vs-duration is confounded: full-timers both
#     travel farther and work longer — no clean prior), income / vehicle /
#     licence / gender / accessibility / industry / WFHDAY / mandatory_role /
#     *_out counts (ambiguous or purpose-dependent directions).
EXPECTED_S3_SIGNS = {
    'B':                      ('>', None, 'Larger day time-budget funds longer spells'),
    'budget_per_activity_hr': ('>', None, 'Larger per-activity budget funds longer spells'),
    'day_span_hr':            ('>', None, 'Wider awake span accommodates longer spells'),
    'n_mandatory':            ('<', None, 'Conditional on B: more mandatory acts dilute each spell'),
    'n_discretionary':        ('<', None, 'Conditional on B: more discretionary acts dilute each spell'),
    'ptype_FullTimeWorker':   ('>', ('work',), 'Full-timers work longer spells'),
    'workstat_parttime':      ('<', ('work',), 'Part-time flag shortens work spells'),
    'age_retired':            ('<', ('work',), 'Working retirees do shorter spells'),
}


def s3_sign_violations(feats, coef, purpose: str) -> int:
    """Count fitted coefficients violating EXPECTED_S3_SIGNS for a purpose.

    feats/coef align positionally (BudgetLognormal.coef_ order). NaN
    coefficients are not counted (unidentified, not wrong-signed).
    """
    n_viol = 0
    coef = np.asarray(coef, dtype=float).ravel()
    for i, f in enumerate(feats):
        spec = EXPECTED_S3_SIGNS.get(str(f))
        if spec is None or i >= len(coef):
            continue
        op, scopes, _note = spec
        if scopes is not None and purpose not in scopes:
            continue
        v = float(coef[i])
        if not np.isfinite(v):
            continue
        if op == '>' and v <= 0:
            n_viol += 1
        elif op == '<' and v >= 0:
            n_viol += 1
    return int(n_viol)


def check_stage3_signs(coef_by_purpose: dict) -> list:
    """Audit fitted duration coefficients against EXPECTED_S3_SIGNS.

    coef_by_purpose: {purpose: {feature: value}}. Returns rows with keys
    purpose/parameter/value/expected/pass/note — pass None means "no prior
    asserted" (reported, never penalised). Mirrors the stage-5 sign table.
    """
    rows = []
    for purpose, coefs in (coef_by_purpose or {}).items():
        for feat, val in (coefs or {}).items():
            spec = EXPECTED_S3_SIGNS.get(str(feat))
            try:
                v = float(val)
            except Exception:
                v = float('nan')
            if spec is None:
                rows.append({'purpose': purpose, 'parameter': str(feat),
                             'value': v, 'expected': 'no prior',
                             'pass': None, 'note': 'reported only'})
                continue
            op, scopes, note = spec
            if scopes is not None and purpose not in scopes:
                rows.append({'purpose': purpose, 'parameter': str(feat),
                             'value': v, 'expected': 'no prior (out of scope)',
                             'pass': None, 'note': note})
                continue
            if not np.isfinite(v):
                ok = None
            elif op == '>':
                ok = bool(v > 0)
            else:
                ok = bool(v < 0)
            rows.append({'purpose': purpose, 'parameter': str(feat),
                         'value': v, 'expected': f'{op} 0',
                         'pass': ok, 'note': note})
    return rows


def attach_stage3_sign_audits(models: dict) -> int:
    """Attach per-purpose sign audits to final estimate_duration_models output.

    models: {purpose: {'model': BudgetLognormal, 'features': [...]}}. Adds a
    'sign_audit' row-list per purpose and logs violations loudly. Never
    raises (audits must not break estimation). Returns total violations.
    """
    total = 0
    try:
        for purpose, entry in (models or {}).items():
            if not isinstance(entry, dict):
                continue
            mdl = entry.get('model')
            feats = entry.get('features') or getattr(mdl, 'feature_names_', None) or []
            coef = getattr(mdl, 'coef_', None)
            if coef is None or feats is None:
                entry['sign_audit'] = [{'purpose': purpose, 'parameter': '(model)',
                                        'value': float('nan'), 'expected': 'n/a',
                                        'pass': None,
                                        'note': 'no coef_ available for audit'}]
                continue
            coef = np.asarray(coef, dtype=float).ravel()
            cmap = {str(f): (float(coef[i]) if i < len(coef) else float('nan'))
                    for i, f in enumerate(feats)}
            rows = check_stage3_signs({purpose: cmap})
            entry['sign_audit'] = rows
            nv = sum(1 for r in rows if r['pass'] is False)
            total += nv
            if nv:
                bad = ', '.join(f"{r['parameter']}={r['value']:.3f} "
                                f"(expected {r['expected']})"
                                for r in rows if r['pass'] is False)
                logger.warning(f"[{purpose}] {nv} sign violation(s): {bad}")
        if total:
            logger.warning(f"Stage 3 sign audit: {total} violation(s) across "
                           f"{len(models)} purposes — see per-model sign_audit")
        else:
            logger.info("Stage 3 sign audit: no violations across "
                        f"{len(models)} purposes")
    except Exception as e:                                      # noqa: BLE001
        logger.warning(f"Stage 3 sign audit failed (non-fatal): {e}")
    return int(total)

PERSON_FEATURES = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp']

DEFAULT_FEATURES_BY_PURPOSE = {
    'work': ['WORKSTAT', 'incp', 'vehrd', 'CARLICENCE', 'ptype_FullTimeWorker', 'ptype_WorkingStudent', 'n_workers_hh'],
    'edu': ['STUDYING', 'age_child', 'age_teen', 'ptype_SchoolChild', 'ptype_UniversityStudent', 'transit_acc'],
    'escort': ['hh_child', 'nprd', 'vehrd', 'has_vehicle', 'WORKSTAT'],
    'shopping': ['incp', 'vehrd', 'nprd', 'has_vehicle', 'auto_shop_acc'],
    'social': ['nprd', 'is_female', 'age_teen', 'age_older', 'walk_acc'],
    'othmaint': ['nprd', 'has_vehicle', 'WORKSTAT', 'auto_nonmand_acc'],
    'othdiscr': ['incp', 'vehrd', 'age_older', 'walk_acc'],
    'eat': ['incp', 'nprd', 'age_working', 'walk_acc', 'transit_acc'],
}


def _duration_person_features(df: pd.DataFrame, purpose: str, feature_override=None) -> list:
    if isinstance(feature_override, dict):
        names = feature_override.get(purpose, feature_override.get('*', []))
    elif feature_override is not None:
        names = feature_override
    else:
        names = DEFAULT_FEATURES_BY_PURPOSE.get(purpose, PERSON_FEATURES)
    if not names:
        names = PERSON_FEATURES
    return [c for c in dict.fromkeys(names) if c in df.columns]


def _numerical_hessian(func, params: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    n = len(params)
    hess = np.zeros((n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = eps
        for j in range(i, n):
            ej = np.zeros(n, dtype=float)
            ej[j] = eps
            fpp = func(params + ei + ej)
            fpm = func(params + ei - ej)
            fmp = func(params - ei + ej)
            fmm = func(params - ei - ej)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
            hess[j, i] = hess[i, j]
    return hess


def _quantile_profile(values: pd.Series, q=None) -> dict:
    """Compact empirical distribution summary for rank-based remapping."""
    if q is None:
        q = np.linspace(0.001, 0.999, 199)
    s = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {'q': np.asarray(q, dtype=float), 'v': np.zeros(len(q), dtype=float)}
    return {'q': np.asarray(q, dtype=float), 'v': np.quantile(s.values.astype(float), q)}


def _apply_quantile_profile(values: np.ndarray, profile: dict) -> np.ndarray:
    """Monotonic rank remapping that preserves ordering while matching marginals."""
    arr = np.asarray(values, dtype=float)
    if profile is None or len(arr) == 0:
        return arr
    q = np.asarray(profile.get('q', []), dtype=float)
    v = np.asarray(profile.get('v', []), dtype=float)
    if q.size == 0 or v.size == 0:
        return arr
    order = np.argsort(arr, kind='mergesort')
    ranks = (np.arange(len(arr), dtype=float) + 0.5) / max(len(arr), 1)
    mapped_sorted = np.interp(ranks, q, v, left=v[0], right=v[-1])
    out = np.empty_like(arr, dtype=float)
    out[order] = mapped_sorted
    return out


# Columns that carry a genuine trip/tour distance (same units within a frame).
# OTAZ/DTAZ-style zone identifiers must NEVER be differenced: with SA2 codes
# the difference is meaningless, and with ORDER indices it is at best an
# ordinal proxy. Prefer a real distance column; otherwise emit an explicit
# zero (no distance signal) with a loud warning instead of ID arithmetic.
_DISTANCE_CANDIDATES = (
    'tour_distance', 'tour_distance_km', 'DIST', 'dist', 'dist_km',
    'distance_km', 'distance', 'TRIPDIST', 'TRIPDIST_KM', 'trip_distance',
    'trip_dist_km', 'od_dist', 'od_dist_km', 'skim_dist',
    # skim-merged auto distance (present when trips carry the OD skim columns)
    'SOVTOLL_DIST__AM',
)

_logged_stage3_warnings: set = set()

# ── OD distance lookup from the configured skim ────────────────────────────
# The HHTS extract carries no distance column. Rather than fall back to a
# zero signal (or, worse, abs(DTAZ-OTAZ) zone-ID arithmetic), derive the trip
# distance by joining each origin→destination pair to the auto road-distance
# matrix (SOVTOLL_DIST / DIST) of the active 371 OD skim. Key-space safe: the
# lookup is only used when a meaningful share of rows actually match, so a
# different zone system (SA2 vs ORDER) degrades to the 0-fallback + warning
# instead of silently returning garbage.
_OD_DIST_CACHE = None
_OD_DIST_SIGNATURE = None


def od_distance_lookup(skim_df: Optional[pd.DataFrame] = None):
    """Return {(otaz, dtaz): dist_km} for the active OD skim (cached), or None.

    Pass ``skim_df`` (a flat OTAZ/DTAZ/…/DIST frame) to build the lookup from
    an in-memory skim; otherwise the path in STAGE3_SKIMS_PATH /
    PIPELINE_SKIMS_PATH (or ../demand_/skims_371.omx) is read via read_skims.
    """
    global _OD_DIST_CACHE, _OD_DIST_SIGNATURE

    def _build(frame):
        _dcol = next((c for c in ('DIST', 'SOVTOLL_DIST__AM', 'SOVTOLL_DIST__PM',
                                  'SOVTOLL_DIST__MD', 'SOVTOLL_DIST__EA',
                                  'SOVTOLL_DIST__EV')
                      if c in frame.columns), None)
        if _dcol is None or 'OTAZ' not in frame.columns or 'DTAZ' not in frame.columns:
            return None
        _o = pd.to_numeric(frame['OTAZ'], errors='coerce')
        _d = pd.to_numeric(frame['DTAZ'], errors='coerce')
        _v = pd.to_numeric(frame[_dcol], errors='coerce')
        _m = _o.notna() & _d.notna() & _v.notna()
        return {(int(a), int(b)): float(c)
                for a, b, c in zip(_o[_m], _d[_m], _v[_m])}

    if skim_df is not None:
        try:
            _lk = _build(skim_df)
            if _lk:
                logger.info(f"stage3: OD distance lookup from supplied skim_df "
                            f"({len(_lk):,} OD pairs)")
                return _lk
        except Exception as _e:
            logger.debug(f"stage3: OD distance lookup from skim_df skipped ({_e!r})")
        return None

    if _OD_DIST_CACHE is not None:
        return _OD_DIST_CACHE or None
    _OD_DIST_CACHE = {}
    try:
        _p = (os.environ.get('STAGE3_SKIMS_PATH')
              or os.environ.get('PIPELINE_SKIMS_PATH') or '').strip()
        if not _p:
            _here = os.path.dirname(os.path.abspath(__file__))
            _cand = os.path.normpath(os.path.join(_here, '..', 'demand_', 'skims_371.omx'))
            _p = _cand if os.path.isfile(_cand) else ''
        if _p and os.path.isfile(_p):
            import read_skims as _rs
            _df = _rs.read_omx_skims(_p, matrices='all', add_derived=False)
            _lk = _build(_df)
            if _lk:
                _OD_DIST_CACHE = _lk
                logger.info(f"stage3: OD distance lookup built from "
                            f"{os.path.basename(_p)} ({len(_lk):,} OD pairs)")
            else:
                logger.warning(f"stage3: skim '{os.path.basename(_p)}' has no "
                               f"DIST/SOVTOLL_DIST column — log_dist stays 0")
        else:
            logger.info("stage3: no OD skim configured (PIPELINE_SKIMS_PATH) — "
                        "log_dist falls back to 0")
    except Exception as _e:
        logger.warning(f"stage3: OD distance lookup unavailable ({_e!r}) — "
                       f"log_dist falls back to 0")
    return _OD_DIST_CACHE or None


# Origin/destination zone columns to try (first present wins).
_OD_ORIG_COLS = ('OTAZ', 'tour_origin', 'ORIGSA2_2021', 'origin', 'from_zone')
_OD_DEST_COLS = ('DTAZ', 'tour_destination', 'DESTSA2_2021', 'destination', 'to_zone')


def _resolve_log_dist(df: pd.DataFrame, skim_df: Optional[pd.DataFrame] = None) -> pd.Series:
    """log1p distance (km) for duration features, estimation/prediction-safe.

    Priority:
      1. a real distance column already on the frame (_DISTANCE_CANDIDATES);
      2. OTAZ→DTAZ joined to the active OD skim's DIST matrix;
      3. explicit 0.0 with a one-time warning.

    Never ``abs(DTAZ - OTAZ)`` (zone IDs are not distances).
    """
    for col in _DISTANCE_CANDIDATES:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0.0).clip(lower=0.0)
            return np.log1p(vals)

    # ── derive from the OD skim (origin→destination road distance) ────────
    try:
        _lk = od_distance_lookup(skim_df)
        if _lk:
            _o = next((c for c in _OD_ORIG_COLS if c in df.columns), None)
            _d = next((c for c in _OD_DEST_COLS if c in df.columns), None)
            if _o and _d:
                _oi = pd.to_numeric(df[_o], errors='coerce')
                _di = pd.to_numeric(df[_d], errors='coerce')
                _dist = pd.Series(
                    [(_lk.get((int(a), int(b)), np.nan))
                     if (pd.notna(a) and pd.notna(b)) else np.nan
                     for a, b in zip(_oi, _di)],
                    index=df.index, dtype='float64')
                _hit = float(_dist.notna().mean()) if len(_dist) else 0.0
                if _hit > 0.05:
                    _med = float(_dist.median()) if _dist.notna().any() else 0.0
                    _dist = _dist.fillna(_med).clip(lower=0.0)
                    if 'stage3_log_dist' not in _logged_stage3_warnings:
                        _logged_stage3_warnings.add('stage3_log_dist')
                        logger.info(
                            f"prepare_duration_data: log_dist derived from OD skim "
                            f"({_hit:.0%} of {len(_dist):,} rows matched on {_o}->{_d}; "
                            f"unmatched filled with median {_med:.1f} km)")
                    return np.log1p(_dist)
                if 'stage3_log_dist_keyspace' not in _logged_stage3_warnings:
                    _logged_stage3_warnings.add('stage3_log_dist_keyspace')
                    logger.warning(
                        f"prepare_duration_data: OD-skim distance lookup matched only "
                        f"{_hit:.1%} of rows on {_o}->{_d} (key-space mismatch: frame "
                        f"zones vs skim OTAZ/DTAZ) — log_dist stays 0")
    except Exception as _e:
        logger.debug(f"stage3 log_dist OD lookup skipped ({_e!r})")

    if 'stage3_log_dist' not in _logged_stage3_warnings:
        _logged_stage3_warnings.add('stage3_log_dist')
        logger.warning(
            "prepare_duration_data: no distance column %s and no usable OD skim — "
            "log_dist=0 (zone-ID differencing disabled as invalid).",
            list(_DISTANCE_CANDIDATES),
        )
    return pd.Series(0.0, index=df.index)


# ---------------------------------------------------------------------------
# 0.  PREPARE ESTIMATION DATA  (one row per tour from HHTS)
# ---------------------------------------------------------------------------

def prepare_duration_data(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    trips: Optional[pd.DataFrame] = None,
    skim_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build activity-level estimation dataset with purpose, duration,
    and person attributes.

    Estimation source priority:
        1. trips table (DESTPURP + ACTIVITY_DURATION) — one row per
           individual activity stop, covers ALL purposes including
           eat, othdiscr which rarely appear as tour primary purpose.
        2. tours table fallback — one row per tour (main purpose only).

    Each row = one activity instance.
    Duration = time spent at the activity destination (minutes).

    ``skim_df`` (optional): a flat OTAZ/DTAZ/…/DIST frame used to derive
    ``log_dist`` from the OD skim when the frame carries no distance column.
    When omitted, the active skim is read from PIPELINE_SKIMS_PATH.
    """
    # ── person attributes ──────────────────────────────────────────
    pers = persons.copy()
    if 'person_id' not in pers.columns and 'PERSID' in pers.columns:
        pers = pers.rename(columns={'PERSID': 'person_id'})
    try:
        from stage1_activity_frequency import assign_ptype_column
        pers = assign_ptype_column(pers)
    except ImportError:
        pers['ptype'] = 'NonWorker'
    keep = ['person_id', 'ptype'] + [c for c in PERSON_FEATURES if c in pers.columns]

    # ── day budget per person from tours ───────────────────────────
    t = tours.copy()
    if 'person_id' not in t.columns and 'PERSID' in t.columns:
        t = t.rename(columns={'PERSID': 'person_id'})
    t['_dur_hrs'] = (t['duration'] / 60).clip(lower=0.05)
    budget = t.groupby('person_id')['_dur_hrs'].sum().reset_index(name='B')

    # tour-level activity counts (n_work_out etc.)
    act_count_cols = [c for c in ACTIVITY_COLS if c in t.columns]
    tour_counts = t.groupby('person_id')[act_count_cols].sum().reset_index()

    # ── person-level day aggregates (GA candidates) ──────────────────
    # budget_per_activity_hr, day_span_hr, n_mandatory, n_discretionary.
    # The GA spec lists them but nothing ever built them — the search could
    # only see per-activity attributes, never day context. Built once here,
    # merged in both branches below (trips branch refines the per-activity
    # budget with actual activity-row counts).
    _n_act = t.groupby('person_id').size().rename('n_activities')
    _bkey = budget.set_index('person_id')['B']
    day_aggs = pd.DataFrame({'n_activities': _n_act})
    day_aggs['budget_per_activity_hr'] = (
        _bkey.reindex(day_aggs.index).fillna(0)
        / day_aggs['n_activities'].clip(lower=1))
    if 'start_time' in t.columns and 'end_time' in t.columns:
        _st = pd.to_numeric(t['start_time'], errors='coerce')
        _et = pd.to_numeric(t['end_time'], errors='coerce')
        day_aggs['day_span_hr'] = (
            (_et.groupby(t['person_id']).max()
             - _st.groupby(t['person_id']).min()) / 60.0)
    else:
        day_aggs['day_span_hr'] = np.nan
    # span covers gaps too, so the budget is a safe lower-bound fallback
    day_aggs['day_span_hr'] = day_aggs['day_span_hr'].fillna(
        _bkey.reindex(day_aggs.index)).clip(lower=0.5)
    _tc = tour_counts.set_index('person_id')
    _mand_cols = [c for c in ('n_work_out', 'n_edu_out') if c in _tc.columns]
    day_aggs['n_mandatory'] = _tc[_mand_cols].sum(axis=1).fillna(0) \
        if _mand_cols else 0.0
    _dis_cols = [c for c in ACTIVITY_COLS
                 if c not in ('n_work_out', 'n_edu_out') and c in _tc.columns]
    day_aggs['n_discretionary'] = _tc[_dis_cols].sum(axis=1).fillna(0) \
        if _dis_cols else 0.0
    day_aggs = day_aggs.reset_index()

    # start_time for start_hr feature
    tour_start = (
        t.groupby('person_id')['start_time'].min().reset_index()
        if 'start_time' in t.columns
        else None
    )

    # ── build activity rows ────────────────────────────────────────
    if trips is not None and len(trips) > 0:
        df = trips.copy()
        if 'person_id' not in df.columns and 'PERSID' in df.columns:
            df = df.rename(columns={'PERSID': 'person_id'})

        # purpose column — prefer DESTPURP
        purp_col = next((c for c in ['DESTPURP', 'purpose'] if c in df.columns), None)
        if purp_col is None:
            logger.warning("trips has no DESTPURP or purpose column — falling back to tours")
            trips = None
        else:
            df = df.rename(columns={purp_col: 'main_purpose'}) if purp_col != 'main_purpose' else df

            # duration at activity = ACTIVITY_DURATION if available,
            # else DURATION - TRAVTIME, else DURATION
            if 'ACTIVITY_DURATION' in df.columns:
                df['duration'] = pd.to_numeric(
                    df['ACTIVITY_DURATION'], errors='coerce'
                ).fillna(0).clip(lower=0)
            elif 'TRAVTIME' in df.columns and 'DURATION' in df.columns:
                df['duration'] = (
                    pd.to_numeric(df['DURATION'], errors='coerce').fillna(0)
                    - pd.to_numeric(df['TRAVTIME'], errors='coerce').fillna(0)
                ).clip(lower=0)
            elif 'DURATION' in df.columns:
                df['duration'] = pd.to_numeric(
                    df['DURATION'], errors='coerce'
                ).fillna(0).clip(lower=0)
            else:
                df['duration'] = 30.0

            # drop home trips — we don't model time "at home"
            home_mask = df['main_purpose'].isin(['home', 'Home', None])
            df = df[~home_mask].copy()

            # OTAZ / DTAZ
            for col in ['OTAZ', 'DTAZ']:
                if col not in df.columns:
                    df[col] = 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            # Never abs(DTAZ - OTAZ): zone IDs are not distances.
            df['log_dist'] = _resolve_log_dist(df, skim_df=skim_df)

            # start hour
            st_col = next((c for c in ['STARTIME', 'start_time', 'depart']
                           if c in df.columns), None)
            df['start_hr'] = (
                pd.to_numeric(df[st_col], errors='coerce').fillna(480) / 60
                if st_col else 8.0
            )

            # merge person attrs + budget
            df = df.drop(columns=[c for c in ACTIVITY_COLS if c in df.columns])
            df = df.merge(pers[keep], on='person_id', how='left')
            df = df.merge(budget, on='person_id', how='left')
            df = df.merge(tour_counts, on='person_id', how='left')
            df = df.merge(day_aggs, on='person_id', how='left')
            # refine with actual activity-row counts (post home-filter rows).
            # LEAKAGE FIX: B sums ALL of the person's durations including this
            # row's own target, so B/n leaks the target into its features.
            # Use leave-one-out budget (B minus own duration, n minus 1) —
            # the day-context signal without the target feeding itself.
            df['duration_hr'] = (pd.to_numeric(df['duration'], errors='coerce')
                                 .fillna(0) / 60.0).clip(lower=0)
            _act_n = df.groupby('person_id').size().rename('_n_act_rows')
            df = df.merge(_act_n, on='person_id', how='left')
            df['budget_per_activity_hr'] = (
                (df['B'] - df['duration_hr']) / (df['_n_act_rows'] - 1).clip(lower=1))
            df = df.drop(columns=['_n_act_rows', 'duration_hr'])
            for _cc in ('n_mandatory', 'n_discretionary'):
                df[_cc] = pd.to_numeric(df[_cc], errors='coerce').fillna(0)
            df['budget_per_activity_hr'] = pd.to_numeric(
                df['budget_per_activity_hr'], errors='coerce').fillna(df['B'])
            df['day_span_hr'] = pd.to_numeric(
                df['day_span_hr'], errors='coerce').fillna(df['B']).clip(lower=0.5)

            logger.info(
                f"prepare_duration_data: using TRIPS table "
                f"({len(df)} activity rows across "
                f"{df['person_id'].nunique()} persons)"
            )

    if trips is None or len(trips) == 0:
        # ── fallback: tour-level ───────────────────────────────────
        logger.info("prepare_duration_data: falling back to TOURS table")
        df = t.copy()
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(60)

        purp_col = next(
            (c for c in ['main_purpose', 'primary_purpose', 'tour_primary_purpose']
             if c in df.columns), None
        )
        if purp_col and purp_col != 'main_purpose':
            df = df.rename(columns={purp_col: 'main_purpose'})
        elif purp_col is None:
            df['main_purpose'] = 'othdiscr'

        orig = df.get('tour_origin', pd.Series(0, index=df.index)).fillna(0)
        dest = df.get('tour_destination', pd.Series(0, index=df.index)).fillna(0)
        # tour_origin/_destination are zone IDs, not coordinates — a real
        # distance column (e.g. tour_distance) is used when present, else the
        # OD skim's OTAZ->DTAZ DIST is joined in.
        df['log_dist'] = _resolve_log_dist(df, skim_df=skim_df)
        df['start_hr'] = df.get('start_time', pd.Series(480, index=df.index)) / 60

        # The tours table carries PER-ROW n_*_out flags; the person-level
        # tour_counts aggregates carry the same names. Merging without
        # dropping first collides them into _x/_y suffixes and leaves the
        # canonical column as the per-row flag (or all-zero) — killing every
        # activity-count feature and their taste interactions. Drop the
        # per-row copies so the person-level aggregates win the name.
        df = df.drop(columns=[c for c in ACTIVITY_COLS if c in df.columns])

        df = df.merge(pers[keep], on='person_id', how='left')
        df = df.merge(budget, on='person_id', how='left')
        df = df.merge(tour_counts, on='person_id', how='left')
        df = df.merge(day_aggs, on='person_id', how='left')
        # Same leave-one-out budget correction as the trips branch: B includes
        # this row's own target duration, so B/n would leak the target.
        _fallback_dur_hr = (
            pd.to_numeric(df['duration'], errors='coerce').fillna(0) / 60.0
        ).clip(lower=0)
        _fallback_n = df.groupby('person_id').size().rename('_n_act_rows')
        df = df.merge(_fallback_n, on='person_id', how='left')
        df['budget_per_activity_hr'] = (
            (df['B'] - _fallback_dur_hr) / (df['_n_act_rows'] - 1).clip(lower=1)
        )
        df = df.drop(columns=['_n_act_rows'])

    # ── common cleanup ─────────────────────────────────────────────
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(float)

    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2,
                'WORKSTAT': 0, 'STUDYING': 0, 'incp': 8,
                'B': 8.0, 'start_hr': 8.0}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    ptypes = sorted(df['ptype'].dropna().unique()) if 'ptype' in df.columns else []
    for pt in ptypes:
        df[f'ptype_{pt}'] = (df['ptype'] == pt).astype(float)

    df.attrs['ptypes'] = ptypes
    return df


# ---------------------------------------------------------------------------
# 1.  BUDGET-CONSTRAINED LOGNORMAL  (pure numpy, no JAX required)
# ---------------------------------------------------------------------------

class BudgetLognormal:
    """
    Accelerated Failure Time (AFT) lognormal duration model.
    """

    def __init__(self, l2: float = None):
        # Ridge on the z-scored coefficients, EMBEDDED in the objective that is
        # both fitted AND differentiated: _neg_ll carries the penalty, so the
        # numerical Hessian used for std_err/p_value inherits a ≈λ eigenvalue
        # lift on every coefficient direction — collinear/flat directions stay
        # invertible instead of producing NaN inference (mirrors SearchLibrium's
        # l2_penalty=0.5 on z-scored coefficients). GA_S3_RIDGE overrides.
        self.l2 = float(os.environ.get('GA_S3_RIDGE', '0.5')) if l2 is None \
            else float(l2)
        self.coef_ = None
        self.intercept_ = None
        self.log_sigma_ = None
        self.sigma_ = None
        self.feature_names_ = None
        self.mu_ = None
        self.sd_ = None
        self.std_err_ = None
        self.t_stat_ = None
        self.p_value_ = None
        self.ll_ = None
        self.aic_ = None
        self.rmse_ = None
        self.n_ = None
        self.converged_ = None

    def _clip_coefficients(self):
        if self.coef_ is not None:
            self.coef_ = np.clip(self.coef_, COEF_CLIP_LO, COEF_CLIP_HI)
        if self.intercept_ is not None:
            self.intercept_ = float(np.clip(self.intercept_, INTERCEPT_CLIP_LO, INTERCEPT_CLIP_HI))

    def _neg_ll(self, params: np.ndarray, Xs: np.ndarray, log_y: np.ndarray) -> float:
        n_feat = Xs.shape[1] - 1
        intercept = params[0]
        beta = params[1:n_feat + 1]
        log_sigma = params[n_feat + 1]
        sigma = np.exp(log_sigma)
        eta = intercept + Xs[:, 1:] @ beta
        resid = log_y - eta
        N = len(log_y)
        ll = (-N * log_sigma - 0.5 * np.sum((resid / sigma) ** 2) - np.sum(log_y))
        ll -= 0.5 * self.l2 * np.dot(beta, beta)
        return -float(ll)

    def _neg_ll_grad(self, params: np.ndarray, Xs: np.ndarray, log_y: np.ndarray) -> np.ndarray:
        n_feat = Xs.shape[1] - 1
        beta = params[1:n_feat + 1]
        log_sigma = params[n_feat + 1]
        sigma = np.exp(log_sigma)
        eta = params[0] + Xs[:, 1:] @ beta
        resid = log_y - eta
        s2 = sigma ** 2
        N = len(log_y)
        d_intercept = np.sum(resid) / s2
        d_beta = Xs[:, 1:].T @ resid / s2 - self.l2 * beta
        d_logsigma = -N + np.sum(resid ** 2) / s2
        return np.concatenate([[d_intercept], d_beta, [d_logsigma]])

    def fit(self, X: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize

        n_feat = X.shape[1]

        # Drop zero-variance columns from the MLE to avoid wasting parameters
        # and inflating AIC/BIC.  coef_ is padded back to n_feat with zeros so
        # _eta / predict_mean accept the full feature matrix unchanged.
        col_var = X.var(axis=0)
        active = col_var > 1e-12          # bool mask, shape (n_feat,)
        n_active = int(active.sum())

        # Standardisation stats for all columns (inactive sd stays 1).
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0

        # Design matrix restricted to informative columns.
        X_act = X[:, active]
        Xs = np.column_stack([np.ones(len(X)),
                              (X_act - self.mu_[active]) / self.sd_[active]])

        log_y = np.log(np.clip(y, 1e-6, None))
        ols = np.linalg.lstsq(Xs, log_y, rcond=None)[0]
        resid_std = max(np.std(log_y - Xs @ ols), 0.05)
        init = np.concatenate([ols, [np.log(resid_std)]])

        def neg_ll_fn(p): return self._neg_ll(p, Xs, log_y)
        def neg_gr_fn(p): return -self._neg_ll_grad(p, Xs, log_y)

        res = minimize(neg_ll_fn, init, jac=neg_gr_fn, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-10})

        self.intercept_ = res.x[0]

        # Full-length coef_ (zero for inactive features).
        self.coef_ = np.zeros(n_feat)
        self.coef_[active] = res.x[1:n_active + 1]

        self.log_sigma_ = res.x[n_active + 1]
        self.sigma_ = float(np.exp(self.log_sigma_))
        self.ll_ = float(-res.fun)
        self.aic_ = float(2 * (n_active + 2) - 2 * self.ll_)   # k = n_active + 2
        self.rmse_ = float(np.sqrt(np.mean((y - self.predict_mean(X)) ** 2)))
        self.n_ = len(y)
        self.converged_ = bool(res.success)
        self._clip_coefficients()

        try:
            params_active = res.x                   # intercept + n_active coefs + log_sigma
            hess = _numerical_hessian(lambda p: self._neg_ll(p, Xs, log_y), params_active)
            cov = np.linalg.pinv(hess)
            se_active = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            ts_active = np.divide(params_active, se_active,
                                  out=np.full_like(params_active, np.nan),
                                  where=se_active > 0)
            pv_active = 2.0 * norm.sf(np.abs(ts_active))

            # Pad to n_feat + 2 so callers can zip against feature_names_.
            size = n_feat + 2
            self.std_err_ = np.full(size, np.nan)
            self.t_stat_  = np.full(size, np.nan)
            self.p_value_ = np.full(size, np.nan)
            active_idx = np.where(active)[0]
            self.std_err_[0]              = se_active[0]           # intercept
            self.std_err_[active_idx + 1] = se_active[1:n_active + 1]
            self.std_err_[-1]             = se_active[-1]          # log_sigma
            self.t_stat_[0]               = ts_active[0]
            self.t_stat_[active_idx + 1]  = ts_active[1:n_active + 1]
            self.t_stat_[-1]              = ts_active[-1]
            self.p_value_[0]              = pv_active[0]
            self.p_value_[active_idx + 1] = pv_active[1:n_active + 1]
            self.p_value_[-1]             = pv_active[-1]
        except Exception:
            size = n_feat + 2
            self.std_err_ = np.full(size, np.nan)
            self.t_stat_  = np.full(size, np.nan)
            self.p_value_ = np.full(size, np.nan)
        return self

    def _eta(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2 and X.shape[1] != len(self.mu_):
            raise ValueError(
                f"Feature mismatch: model expects {len(self.mu_)} columns "
                f"but received {X.shape[1]}. "
                f"Delete saved_data/stage3_models.pkl and re-run."
            )
        Xs = (X - self.mu_) / self.sd_
        return self.intercept_ + Xs @ self.coef_

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self._eta(X) + 0.5 * self.sigma_ ** 2)

    def predict_sampled(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.exp(rng.normal(self._eta(X), self.sigma_))


def _rank_to_unit(values: np.ndarray) -> np.ndarray:
    """Map arbitrary order values to the open unit interval."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 0.5, dtype=float)
    filled = values.copy()
    filled[~finite] = np.nanmedian(filled[finite])
    ranks = pd.Series(filled).rank(method='average').to_numpy(dtype=float)
    n = float(len(values))
    return np.clip((ranks - 0.5) / max(n, 1.0), 0.001, 0.999)


def _activity_order_quantile(frame: pd.DataFrame) -> np.ndarray:
    """Return each activity's observed within-person order quantile."""
    if frame.empty:
        return np.zeros(0, dtype=float)
    time_col = next(
        (c for c in (
            'planning_order', 'activity_seq', 'STARTIME', 'start_time',
            'depart', 'start_hr',
        ) if c in frame.columns),
        None,
    )
    work = frame.copy()
    work['_order_time'] = (
        pd.to_numeric(work[time_col], errors='coerce').fillna(0.0)
        if time_col is not None else np.arange(len(work), dtype=float)
    )
    if 'person_id' not in work.columns:
        return _rank_to_unit(work['_order_time'].to_numpy())

    work['_order_row'] = np.arange(len(work))
    ordered = work.sort_values(['person_id', '_order_time', '_order_row'], kind='stable')
    ranks = ordered.groupby('person_id', sort=False).cumcount().astype(float)
    counts = ordered.groupby('person_id', sort=False)['person_id'].transform('size').astype(float)
    ordered['_order_u'] = (ranks + 0.5) / counts.clip(lower=1.0)
    return ordered.set_index('_order_row')['_order_u'].reindex(
        work['_order_row']
    ).fillna(0.5).to_numpy(dtype=float)


class CopulaLinkedHazardDurationModel:
    """Joint activity-duration model with an order-linked Gaussian copula.

    The duration margin is an AFT lognormal model.  A discrete-time completion
    hazard is fitted on the same observations.  The standardized AFT residual
    and the observed within-person activity-order quantile identify a Gaussian
    copula correlation.  At prediction time, the copula supplies a shared
    latent duration/completion propensity, so earlier or later planned
    activities can receive systematically different duration draws.

    This is intentionally separate from ``BudgetLognormal``.  It can be
    enabled for experiments without changing saved legacy duration models.
    """

    estimator_ = 'copula_linked_discrete_hazard'

    def __init__(self, l2: float = 1e-4, n_bins: int = 16,
                 hazard_l2: float = 1e-3):
        self.l2 = l2
        self.n_bins = max(int(n_bins), 4)
        self.hazard_l2 = hazard_l2
        self.marginal_ = BudgetLognormal(l2=l2)
        self.feature_names_ = None
        self.rho_ = 0.0
        self.bin_edges_ = None
        self.hazard_intercept_ = None
        self.hazard_coef_ = None
        self.hazard_ll_ = None
        self.hazard_converged_ = False
        self.n_ = 0

    @property
    def sigma_(self):
        return self.marginal_.sigma_

    @property
    def coef_(self):
        return self.marginal_.coef_

    @property
    def intercept_(self):
        return self.marginal_.intercept_

    @property
    def ll_(self):
        return self.marginal_.ll_

    @property
    def rmse_(self):
        return self.marginal_.rmse_

    @property
    def converged_(self):
        return bool(self.marginal_.converged_ and self.hazard_converged_)

    def _eta_standardized(self, X: np.ndarray) -> np.ndarray:
        eta = self.marginal_._eta(np.asarray(X, dtype=float))
        mu = float(getattr(self, 'eta_mu_', 0.0))
        sd = max(float(getattr(self, 'eta_sd_', 1.0)), 1e-6)
        return (eta - mu) / sd

    def _latent_order(self, order_signal, size: int, rng=None) -> np.ndarray:
        if order_signal is None:
            if rng is None:
                return np.zeros(size, dtype=float)
            return rng.normal(size=size)
        z_order = norm.ppf(_rank_to_unit(order_signal))
        if rng is None:
            return self.rho_ * z_order
        residual_sd = np.sqrt(max(1.0 - self.rho_ ** 2, 1e-6))
        return self.rho_ * z_order + residual_sd * rng.normal(size=size)

    def fit(self, X: np.ndarray, y: np.ndarray, order_signal=None,
            bounds=None):
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y) or len(y) == 0:
            raise ValueError('X and y must contain the same non-zero number of rows')

        self.marginal_.fit(X, y)
        self.feature_names_ = list(getattr(self, 'feature_names_', []) or [])
        self.n_ = len(y)
        eta = self.marginal_._eta(X)
        sigma = max(float(self.marginal_.sigma_), 0.05)
        z_duration = np.clip((np.log(np.clip(y, 1e-6, None)) - eta) / sigma, -6.0, 6.0)
        z_order = norm.ppf(_rank_to_unit(order_signal)) if order_signal is not None \
            else np.zeros(len(y), dtype=float)
        if np.std(z_order) > 1e-8 and np.std(z_duration) > 1e-8:
            self.rho_ = float(np.clip(np.corrcoef(z_duration, z_order)[0, 1], -0.95, 0.95))
        else:
            self.rho_ = 0.0

        eta_sd = max(float(np.std(eta)), 1e-6)
        self.eta_mu_ = float(np.mean(eta))
        self.eta_sd_ = eta_sd
        eta_std = (eta - self.eta_mu_) / eta_sd

        lo = float(bounds[0]) if bounds is not None else max(float(np.min(y)), 1.0)
        hi = float(bounds[1]) if bounds is not None else float(np.max(y))
        hi = max(hi, float(np.max(y)) * 1.02, lo * 1.05)
        self.bin_edges_ = np.geomspace(max(lo, 1.0), hi, self.n_bins + 1)
        self.n_bins_ = len(self.bin_edges_) - 1

        event_bins = np.searchsorted(self.bin_edges_[1:], y, side='right')
        event_bins = np.clip(event_bins, 0, self.n_bins_ - 1)
        row_idx = np.concatenate([
            np.full(event_bin + 1, row, dtype=int)
            for row, event_bin in enumerate(event_bins)
        ])
        bin_idx = np.concatenate([
            np.arange(event_bin + 1, dtype=int)
            for event_bin in event_bins
        ])
        events = (bin_idx == event_bins[row_idx]).astype(float)
        n_at_risk = np.bincount(bin_idx, minlength=self.n_bins_).astype(float)
        n_events = np.bincount(bin_idx, weights=events, minlength=self.n_bins_)
        base_rate = np.clip(n_events / np.maximum(n_at_risk, 1.0), 1e-4, 1.0 - 1e-4)
        initial = np.concatenate([
            np.log(base_rate / (1.0 - base_rate)), np.zeros(2, dtype=float)
        ])

        def objective(params):
            intercept = params[:self.n_bins_]
            coef = params[self.n_bins_:]
            linear = intercept[bin_idx] + coef[0] * eta_std[row_idx] \
                + coef[1] * z_duration[row_idx]
            return float(
                np.sum(np.logaddexp(0.0, linear) - events * linear)
                + 0.5 * self.hazard_l2 * np.dot(coef, coef)
            )

        def gradient(params):
            intercept = params[:self.n_bins_]
            coef = params[self.n_bins_:]
            linear = intercept[bin_idx] + coef[0] * eta_std[row_idx] \
                + coef[1] * z_duration[row_idx]
            residual = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0))) - events
            grad_intercept = np.bincount(
                bin_idx, weights=residual, minlength=self.n_bins_
            )
            grad_coef = np.array([
                np.dot(residual, eta_std[row_idx]),
                np.dot(residual, z_duration[row_idx]),
            ]) + self.hazard_l2 * coef
            return np.concatenate([grad_intercept, grad_coef])

        result = minimize(
            objective, initial, jac=gradient, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-9},
        )
        self.hazard_intercept_ = result.x[:self.n_bins_]
        self.hazard_coef_ = result.x[self.n_bins_:]
        self.hazard_ll_ = float(-result.fun)
        self.hazard_converged_ = bool(result.success)
        return self

    def _hazards(self, X: np.ndarray, z_duration: np.ndarray) -> np.ndarray:
        eta_std = self._eta_standardized(X)
        linear = (
            self.hazard_intercept_[None, :]
            + self.hazard_coef_[0] * eta_std[:, None]
            + self.hazard_coef_[1] * z_duration[:, None]
        )
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))

    def completion_hazard(self, X: np.ndarray, order_signal=None) -> np.ndarray:
        """Return completion hazard by duration bin for each activity."""
        X = np.asarray(X, dtype=float)
        z = self._latent_order(order_signal, len(X))
        return self._hazards(X, z)

    def predict_sampled(self, X: np.ndarray, rng: np.random.Generator,
                        order_signal=None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z_duration = self._latent_order(order_signal, len(X), rng=rng)
        hazards = self._hazards(X, z_duration)
        cdf = 1.0 - np.cumprod(1.0 - hazards, axis=1)
        uniforms = rng.random(len(X))
        bins = np.sum(uniforms[:, None] > cdf, axis=1)
        bins = np.clip(bins, 0, self.n_bins_ - 1).astype(int)

        aft_duration = np.exp(self.marginal_._eta(X) + self.sigma_ * z_duration)
        low = self.bin_edges_[bins]
        high = self.bin_edges_[bins + 1]
        log_low = np.log(np.maximum(low, 1e-6))
        log_high = np.log(np.maximum(high, low + 1e-6))
        width = np.maximum(log_high - log_low, 0.05)
        midpoint = 0.5 * (log_low + log_high)
        frac = 1.0 / (1.0 + np.exp(-np.clip((np.log(aft_duration) - midpoint) / width, -30.0, 30.0)))
        return np.exp(log_low + frac * (log_high - log_low))

    def predict_mean(self, X: np.ndarray, order_signal=None) -> np.ndarray:
        """Return the hazard-implied mean duration in minutes."""
        X = np.asarray(X, dtype=float)
        z_duration = self._latent_order(order_signal, len(X))
        hazards = self._hazards(X, z_duration)
        survival_before = np.column_stack([
            np.ones(len(X)), np.cumprod(1.0 - hazards[:, :-1], axis=1)
        ])
        mass = survival_before * hazards
        mid = np.sqrt(self.bin_edges_[:-1] * self.bin_edges_[1:])
        tail = np.prod(1.0 - hazards, axis=1)
        return mass @ mid + tail * self.bin_edges_[-1]


def _augment_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    if 'B' not in df.columns:
        df['B'] = 8.0
    df['B'] = pd.to_numeric(df['B'], errors='coerce').fillna(8.0)
    total_acts = df[ACTIVITY_COLS].sum(axis=1).clip(lower=1.0)
    # Preserve a leave-one-out budget supplied by prepare_duration_data(): that
    # column already removes the current activity's own target duration.
    # Recomputing B/total_acts here would overwrite it and reintroduce leakage.
    if 'budget_per_activity_hr' not in df.columns:
        df['budget_per_activity_hr'] = df['B'] / total_acts
    else:
        df['budget_per_activity_hr'] = pd.to_numeric(
            df['budget_per_activity_hr'], errors='coerce'
        ).fillna(df['B'] / total_acts)
    if 'start_of_day' in df.columns and 'end_of_day' in df.columns:
        df['day_span_hr'] = np.clip((df['end_of_day'] - df['start_of_day']) / 60.0, 0.25, 18.0)
    else:
        df['day_span_hr'] = df['B']
    df['n_mandatory'] = df.get('n_work_out', 0.0) + df.get('n_edu_out', 0.0)
    df['n_discretionary'] = total_acts - df['n_mandatory']
    return df


def _feature_cols(df: pd.DataFrame, ptypes: list,
                  feature_override: list = None) -> list:
    person_feats = feature_override if feature_override else PERSON_FEATURES
    base = (['start_hr', 'log_dist', 'B', 'budget_per_activity_hr', 'day_span_hr',
             'n_mandatory', 'n_discretionary'] +
            ACTIVITY_COLS +
            [c for c in person_feats if c in df.columns] +
            [f'ptype_{pt}' for pt in ptypes if f'ptype_{pt}' in df.columns])
    return base


def _feature_cols_for_purpose(df: pd.DataFrame, ptypes: list, purpose: str,
                              feature_override=None) -> list:
    person_feats = _duration_person_features(df, purpose, feature_override)
    base = (['start_hr', 'log_dist', 'B', 'budget_per_activity_hr', 'day_span_hr',
             'n_mandatory', 'n_discretionary'] +
            ACTIVITY_COLS +
            [c for c in person_feats if c in df.columns] +
            [f'ptype_{pt}' for pt in ptypes if f'ptype_{pt}' in df.columns])
    return list(dict.fromkeys([c for c in base if c in df.columns]))


def duration_design_matrix(
    frame: pd.DataFrame,
    features: list,
    model,
    label: str = '',
    log_missing: bool = True,
) -> tuple:
    """Build a duration model's design matrix in the fitted column order.

    A fitted feature the frame does not carry is filled with the model's
    TRAINING MEAN, not zero.  The model is fitted on z-scored columns, so a
    raw zero is not a neutral value: ``walk_acc`` has mean 5.17 and sd 0.53,
    so zero-filling it is a z of -9.7 and swung the shopping mean from the
    fitted 58 min to 11 min — a fifth of the observed length.  Filling with
    the mean puts an absent column at z = 0 (the average person), which
    leaves the prediction at the specification's own baseline instead of
    inventing an extreme covariate value.

    The exception is a person-type dummy (``ptype_*``): if the frame has no
    such person type, the indicator is genuinely 0 and filling it with the
    training proportion would invent a fraction of a person.

    Returns ``(X, imputed)`` where ``imputed`` lists the filled features, so
    callers can report a design matrix that is incomplete rather than let it
    bias the fit silently.
    """
    features = list(features or [])
    # The copula-hazard variant keeps its lognormal marginal in `marginal_`,
    # so the standardisation stats live one level down there.
    _stats = getattr(model, 'mu_', None)
    if _stats is None:
        _stats = getattr(getattr(model, 'marginal_', None), 'mu_', None)
    mu = (np.asarray(_stats, dtype=float) if _stats is not None
          else np.zeros(len(features)))
    X = np.zeros((len(frame), len(features)), dtype=float)
    imputed: list = []
    for j, name in enumerate(features):
        default = float(mu[j]) if j < len(mu) else 0.0
        if name.startswith('ptype_'):
            default = 0.0
        if name in frame.columns:
            X[:, j] = pd.to_numeric(frame[name], errors='coerce') \
                .fillna(default).to_numpy(dtype=float)
        else:
            X[:, j] = default
            imputed.append(name)
    if imputed and log_missing:
        logger.warning(
            f"Stage 3 {label or 'duration'} design matrix: "
            f"{len(imputed)}/{len(features)} fitted features absent from the "
            f"frame ({', '.join(imputed)}) — imputed at the training mean "
            f"(z=0), so predictions for those rows rest on the specification "
            f"baseline."
        )
    return X, imputed


def estimate_duration_models(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    trips: Optional[pd.DataFrame] = None,
    min_sample: int = 50,
    use_mdcev: bool = False,
    feature_override: list = None,
    duration_model: str = None,
    skim_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Fit one BudgetLognormal per activity purpose.

    Parameters
    ----------
    tours, persons    : from process_survey()
    min_sample        : skip purposes with fewer observations
    feature_override  : GA-selected feature list (replaces PERSON_FEATURES)
    duration_model    : ``'lognormal'`` (default) or ``'copula_hazard'``.
                        The latter adds an order-linked discrete hazard
                        layer around the lognormal marginal.
    """
    data    = _augment_duration_features(
        prepare_duration_data(tours, persons, trips=trips, skim_df=skim_df))
    # ── GA dict override: recreate the engineered search-frame columns ────
    # The GA search frame (see _run_ga_stage3) carries ptype dummies,
    # is_female, income flags, mandatory_role and the _enrich_ga_data
    # additions (vehicle/licence/household/age-band columns + interactions).
    # prepare_duration_data only keeps PERSON_FEATURES, so without this step
    # every GA-selected engineered feature is silently dropped by the
    # "c in df.columns" filter and the refit degrades to the default spec.
    if isinstance(feature_override, dict) and feature_override:
        _attrs   = dict(data.attrs)
        _want_pt = {str(f) for vals in feature_override.values()
                    for f in (vals or []) if str(f).startswith('ptype_')}
        try:
            from ga_hypothesis_testing import ensure_ga_search_columns
            data = ensure_ga_search_columns(data, persons, ptype_subset=_want_pt,
                                            stage_id=3, zone_col='DESTSA2_2021')
            _missing = sorted(
                {str(f) for vals in feature_override.values() for f in (vals or [])}
                - set(data.columns)
            )
            logger.info(
                f"GA feature override: ensured search-frame columns "
                f"({_want_pt and len(_want_pt) or 0} ptype dummies); "
                f"unresolvable features: {_missing or 'none'}"
            )
        except Exception as _ee:
            logger.warning(f"GA feature-column enrichment failed: {_ee}")
        finally:
            # merges inside the helper drop DataFrame.attrs ('ptypes' needed below)
            data.attrs.update(_attrs)
    duration_model = (duration_model or os.environ.get(
        'STAGE3_DURATION_MODEL', 'lognormal'
    )).strip().lower()
    if duration_model in {'copula', 'hazard', 'copula-linked-hazard'}:
        duration_model = 'copula_hazard'
    if duration_model not in {'lognormal', 'copula_hazard'}:
        logger.warning(
            f"Unknown STAGE3_DURATION_MODEL={duration_model!r}; using lognormal"
        )
        duration_model = 'lognormal'
    data['_activity_order_u'] = _activity_order_quantile(data)
    ptypes  = data.attrs.get('ptypes', [])
    # ---- diagnose purpose column before fitting -------------------------
    purpose_col = None
    for candidate in ['main_purpose', 'primary_purpose', 'tour_primary_purpose']:
        if candidate in data.columns and data[candidate].notna().any():
            purpose_col = candidate
            break

    if purpose_col is None:
        logger.error("No purpose column found in tours. Checked: "                     "main_purpose, primary_purpose, tour_primary_purpose")
        return {}

    unique_vals = data[purpose_col].dropna().unique()
    logger.info(f"Using purpose column: '{purpose_col}' "                f"with {len(unique_vals)} unique values: {unique_vals[:10]}")

    # map to coded purposes if column contains raw survey strings
    raw_to_code = {
        'Work (my workplace)': 'work', 'Work (other work reason)': 'work',
        'Direct Work Commute': 'work', 'Work Related': 'work',
        'Full-time Work': 'work', 'Part-time Work': 'work', 'Work': 'work',
        'Education': 'edu', 'Childcare or Kindergarten': 'edu',
        'Childcare or kindergarten': 'edu',
        'Primary School': 'edu', 'Secondary School': 'edu',
        'Full-time TAFE/Uni': 'edu', 'Part-time TAFE/Uni': 'edu',
        'Childcare/Kindergarten': 'edu',
        'Social Visit': 'social', 'Recreation/leisure activity': 'social',
        'Recreation': 'social', 'Social': 'social',
        'Accompanying Someone': 'social', 'Accompany Someone': 'social',
        'Pick up or drop off a passenger': 'escort',
        'Pickup/Dropoff Someone': 'escort', 'Pickup/Deliver Something': 'escort',
        'Just accompanying someone': 'escort',
        'Shopping': 'shopping',
        'Personal Business': 'othmaint', 'Personal business': 'othmaint',
        'Pickup/deliver something (not work-related)': 'othmaint',
        'Other place (please specify)': 'othdiscr', 'Other Place': 'othdiscr',
        'Refuel my vehicle': 'othdiscr', 'Other Purpose': 'othdiscr',
        'At Home': 'home', 'Go home': 'home',
        'Eat/drink': 'eat',
    }
    # if values are already codes (work, edu…) mapping is identity
    already_coded = set(unique_vals).issubset(set(PURPOSES) | {'home', None})
    if not already_coded:
        logger.info("Mapping raw survey strings → purpose codes")
        data['_purpose_coded'] = data[purpose_col].map(raw_to_code)
        # token fallback for raw HHTS strings not in the exact map
        # ('Full-time Work', 'Primary School', 'Full-time TAFE/Uni', ...)
        _miss = data['_purpose_coded'].isna() & data[purpose_col].notna()
        if _miss.any():
            def _token_code(v):
                s = str(v).strip().lower()
                if not s or s == 'nan':
                    return None
                if 'work' in s:
                    return 'work'
                if any(t in s for t in ('school', 'tafe', 'univers',
                                        'kinder', 'childcare', 'educat')):
                    return 'edu'
                if 'escort' in s or 'pickup' in s or 'dropoff' in s \
                        or 'accompan' in s:
                    return 'escort'
                if 'shop' in s:
                    return 'shopping'
                if 'social' in s or 'recreation' in s or 'visit' in s \
                        or 'leisure' in s:
                    return 'social'
                if 'business' in s or 'personal' in s:
                    return 'othmaint'
                if 'eat' in s or 'drink' in s or 'dine' in s:
                    return 'eat'
                return 'othdiscr'
            data.loc[_miss, '_purpose_coded'] = data.loc[
                _miss, purpose_col].map(_token_code)
            logger.info(
                f"  token fallback coded {int(data.loc[_miss, '_purpose_coded'].notna().sum())}"
                f"/{int(_miss.sum())} remaining raw purposes")
    else:
        data['_purpose_coded'] = data[purpose_col]

    n_unmapped = data['_purpose_coded'].isna().sum()
    if n_unmapped > 0:
        unmapped_vals = data.loc[data['_purpose_coded'].isna(), purpose_col].unique()
        logger.warning(f"{n_unmapped} tours have unmapped purpose: {unmapped_vals}")

    models = {}

    _s3_zthresh = float(os.environ.get('GA_SIG_COEF_ZERO', '0.01'))
    _s3_pthresh = float(os.environ.get('GA_SIG_PVAL_THRESH', '0.10'))

    for purpose in PURPOSES:
        lo, hi = DURATION_BOUNDS[purpose]

        # Fit on every observed episode of the purpose.  Person type is a
        # feature (dummies below), not a row filter: the prediction population
        # includes all person types, so the estimation population must too.
        subset = data[data['_purpose_coded'] == purpose].copy()

        if len(subset) < min_sample:
            logger.warning(f"  {purpose}: {len(subset)} obs < {min_sample} — skipping")
            continue

        y = subset['duration'].clip(lo, hi).values.astype(float)

        X_cols = _feature_cols_for_purpose(data, ptypes, purpose, feature_override)
        X = subset[X_cols].fillna(0).values.astype(float)

        m = BudgetLognormal()
        m.fit(X, y)
        m.feature_names_ = X_cols

        # Iterative p-value + magnitude pruning (up to 3 rounds)
        for _prune_round in range(3):
            if len(X_cols) <= 1:
                break
            _pvals = getattr(m, 'p_value_', None)
            _coefs = np.asarray(m.coef_).ravel()
            # Robustness: still prune near-zero coefficients even when p-values
            # are unavailable (Hessian failed). Previously this `break`'d on a
            # missing p_value_, leaving zero-fit terms in the model.
            _have_pv = _pvals is not None and len(_pvals) >= len(X_cols) + 1
            # layout: [0]=intercept, [1:n+1]=features, [-1]=log_sigma
            _feat_pvals = _pvals[1:len(X_cols) + 1] if _have_pv else None
            _keep = np.ones(len(X_cols), dtype=bool)
            for _fi in range(len(X_cols)):
                _near_zero = _fi < len(_coefs) and abs(_coefs[_fi]) < _s3_zthresh
                _insig = (_have_pv and np.isfinite(_feat_pvals[_fi])
                          and _feat_pvals[_fi] > _s3_pthresh)
                if _near_zero or _insig:
                    _keep[_fi] = False
            _keep_idx = np.where(_keep)[0]
            if len(_keep_idx) == 0 or len(_keep_idx) == len(X_cols):
                break
            X_cols = [X_cols[i] for i in _keep_idx]
            X = subset[X_cols].fillna(0).values.astype(float)
            m = BudgetLognormal()
            m.fit(X, y)
            m.feature_names_ = X_cols

        if duration_model == 'copula_hazard':
            try:
                copula_model = CopulaLinkedHazardDurationModel()
                copula_model.feature_names_ = X_cols
                copula_model.fit(
                    X, y,
                    order_signal=subset.get('_activity_order_u'),
                    bounds=(lo, hi),
                )
                m = copula_model
            except Exception as exc:
                logger.warning(
                    f"  {purpose}: copula-hazard fit failed; using lognormal: {exc}"
                )

        if isinstance(m, CopulaLinkedHazardDurationModel):
            mean_values = m.predict_mean(
                X, order_signal=subset.get('_activity_order_u')
            )
        else:
            mean_values = m.predict_mean(X)
        mean_pred = float(np.nanmean(mean_values)) if len(X) else float('nan')
        mean_obs = float(np.nanmean(y)) if len(y) else float('nan')
        calib = 1.0
        if np.isfinite(mean_pred) and mean_pred > 0 and np.isfinite(mean_obs):
            # Work allows a higher ceiling: full-time workers regularly have 8–12h
            # activities; the lognormal mean undershoots because sigma is estimated
            # on a clipped distribution.  Allow up to 3.0× correction for work
            # to capture long-duration outliers.
            _calib_hi = 3.0 if purpose == 'work' else 1.50
            calib = float(np.clip(mean_obs / mean_pred, 0.67, _calib_hi))

        models[purpose] = {
            'model':    m,
            'features': X_cols,
            'bounds':   (lo, hi),
            'n_obs':    len(subset),
            'sigma':    m.sigma_,
            'll':       m.ll_,
            'rmse':     m.rmse_,
            'calibration_factor': calib,
            'mean_obs': mean_obs,
            'mean_pred_uncal': mean_pred,
            'observed_profile': _quantile_profile(pd.Series(y)),
            'estimator': getattr(m, 'estimator_', 'budget_lognormal'),
        }
        bias_pct = (mean_pred - mean_obs) / mean_obs * 100 if mean_obs > 0 else float('nan')
        bias_str = f"{bias_pct:+.1f}%" if np.isfinite(bias_pct) else "n/a"
        logger.info(f"  {purpose}: n={len(subset)}, "
                    f"sigma={m.sigma_:.3f}, "
                    f"obs_mean={mean_obs:.1f} min, pred_mean={mean_pred:.1f} min "
                    f"(bias={bias_str}), calib={calib:.3f}")

    print(f"Stage 3: fitted {len(models)} duration models "
          f"({', '.join(models.keys())})")

    # Behavioural sign audit on the FINAL models (paper tables must be
    # sign-defensible; the GA penalty only steers selection, this REPORTS it).
    try:
        attach_stage3_sign_audits(models)
    except Exception:
        pass

    if use_mdcev and _MDCEV_AVAILABLE and not data.empty and 'person_id' in data.columns:
        try:
            pp = (data.groupby(['person_id', purpose_col])['duration']
                    .sum()
                    .unstack(fill_value=0.0)
                    .reindex(columns=PURPOSES, fill_value=0.0))
            if len(pp) > 0 and (pp.sum(axis=1) > 0).any():
                mdcev = MDCEVModel(outside_good=PURPOSES.index('work') if 'work' in PURPOSES else 0)
                if hasattr(mdcev, 'fit_mle'):
                    mdcev.fit_mle(pp.values, labels=PURPOSES)
                else:
                    mdcev.fit(pp.values, labels=PURPOSES)
                models['_mdcev'] = {
                    'model': mdcev,
                    'labels': PURPOSES,
                    'n_obs': int(len(pp)),
                    'method': 'mle' if hasattr(mdcev, 'fit_mle') else 'heuristic',
                }
                logger.info(f"Stage 3: fitted MDCEV allocator on {len(pp)} person-day budgets")
        except Exception as _mdcev_e:
            logger.warning(f"Stage 3 MDCEV fit failed, falling back to lognormal-only allocation: {_mdcev_e}")

    planning = estimate_planning_order_model(data)
    if planning is not None:
        models['_planning_order'] = planning

    return models


# ---------------------------------------------------------------------------
# 2b.  POLARIS-STYLE ACTIVITY PLANNING ORDER
# ---------------------------------------------------------------------------

_PLANNING_FEATURES = [
    'mandatory', 'flexibility', 'duration_ratio',
    'remaining_ratio', 'horizon_pressure', 'empirical_rank',
]


class PlanningOrderModel:
    """Sequential next-activity choice model with RRM and availability."""

    def __init__(self, rrm=None, duration_by_purpose=None, rank_by_purpose=None):
        self.rrm = rrm
        self.duration_by_purpose = {
            p: float((duration_by_purpose or {}).get(p, 30.0))
            for p in PURPOSES
        }
        self.rank_by_purpose = {
            p: float((rank_by_purpose or {}).get(p, i))
            for i, p in enumerate(PURPOSES)
        }
        self.feature_names = list(_PLANNING_FEATURES)
        self.n_obs = 0
        self.estimator = 'searchlibrium_rrm' if rrm is not None else 'empirical'

    @staticmethod
    def _is_mandatory(purpose):
        return float(purpose in {'work', 'edu'})

    @staticmethod
    def _is_flexible(purpose):
        return float(purpose not in {'work', 'edu'})

    def _candidate_features(self, remaining, durations, horizon_min):
        total_remaining = max(sum(remaining.values()), 1)
        horizon = max(float(horizon_min), 60.0)
        features = np.zeros((len(PURPOSES), len(self.feature_names)), dtype=float)
        for i, purpose in enumerate(PURPOSES):
            duration = max(float(durations.get(purpose, self.duration_by_purpose[purpose])), 1.0)
            features[i] = [
                self._is_mandatory(purpose),
                self._is_flexible(purpose),
                duration / horizon,
                float(remaining.get(purpose, 0)) / total_remaining,
                duration / max(horizon * total_remaining, 1.0),
                self.rank_by_purpose[purpose],
            ]
        return features

    def _probabilities(self, features, available):
        if self.rrm is None:
            scores = -features[:, self.feature_names.index('empirical_rank')]
            scores = np.where(available, scores, -np.inf)
            scores -= np.max(scores)
            probs = np.exp(scores)
            return probs / max(float(probs.sum()), 1e-12)
        try:
            probs = self.rrm.predict_proba(
                features[None, :, :], avail=available.astype(float)[None, :]
            )[0]
            return np.asarray(probs, dtype=float)
        except Exception as exc:
            logger.debug(f'Planning RRM prediction failed; using empirical order: {exc}')
            scores = -features[:, self.feature_names.index('empirical_rank')]
            scores = np.where(available, scores, -np.inf)
            scores -= np.max(scores)
            probs = np.exp(scores)
            return probs / max(float(probs.sum()), 1e-12)

    def plan_order_arrays(self, purpose_arr, duration_arr,
                          horizon_min: float) -> np.ndarray:
        """Step assigned to each row of ONE person's activities.

        Pure numpy over the person's narrow columns: no frame copies, no
        per-step ``.loc`` writes.  This is deliberately separate from the wide
        activity frame, because the earlier DataFrame-based version copied,
        sorted and dropped the full 610-column frame once per person and cost
        320 s of the 415 s a 20k-person prediction chunk took.
        """
        n = int(len(purpose_arr))
        order = np.zeros(n, dtype=np.int64)
        if n == 0:
            return order
        purpose_index = {p: i for i, p in enumerate(PURPOSES)}
        # rows of an unknown purpose are kept but never planned (step 0)
        idx = np.array([purpose_index.get(p, -1) for p in purpose_arr],
                       dtype=np.int64)
        known = idx >= 0
        counts = np.bincount(idx[known], minlength=len(PURPOSES)).astype(np.int64)
        dur_sum = np.bincount(idx[known], weights=np.asarray(duration_arr,
                                                             dtype=float)[known],
                              minlength=len(PURPOSES))
        means = dur_sum / np.maximum(counts, 1)
        remaining = {p: int(c) for p, c in zip(PURPOSES, counts)}
        durations = {p: float(m) for p, m in zip(PURPOSES, means)}
        rows_by_purpose = {p: list(np.flatnonzero(idx == i))
                           for i, p in enumerate(PURPOSES)}

        for step in range(n):
            available = np.array([remaining.get(p, 0) > 0 for p in PURPOSES],
                                 dtype=bool)
            if not available.any():
                break
            features = self._candidate_features(remaining, durations, horizon_min)
            probs = self._probabilities(features, available)
            probs = np.where(available & np.isfinite(probs), probs, -np.inf)
            chosen = PURPOSES[int(np.argmax(probs))]
            pos = rows_by_purpose[chosen].pop(0)
            order[pos] = step
            remaining[chosen] -= 1
        return order

    def plan_order(self, group: pd.DataFrame) -> np.ndarray:
        """Step per row for one person's activity slice (see plan_order_arrays)."""
        if len(group) == 0:
            return np.zeros(0, dtype=np.int64)
        if {'start_of_day', 'end_of_day'}.issubset(group.columns):
            horizon = float(group['end_of_day'].iloc[0] - group['start_of_day'].iloc[0])
        else:
            horizon = 480.0
        return self.plan_order_arrays(
            group['purpose'].to_numpy(),
            pd.to_numeric(group['predicted_duration_min'], errors='coerce')
              .fillna(0.0).to_numpy(dtype=float),
            horizon,
        )

    def order_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Sort one person's activities into planned order (kept for callers)."""
        out = group.copy()
        if out.empty:
            return out
        out['planning_order'] = self.plan_order(out)
        out = out.sort_values('planning_order', kind='stable')
        out['activity_seq'] = np.arange(len(out), dtype=int)
        return out


def _planning_purpose_column(data: pd.DataFrame) -> Optional[str]:
    if '_purpose_coded' in data.columns:
        return '_purpose_coded'
    for col in ('main_purpose', 'purpose', 'primary_purpose'):
        if col in data.columns:
            return col
    return None


def _planning_order_rows(data: pd.DataFrame) -> pd.DataFrame:
    purpose_col = _planning_purpose_column(data)
    if purpose_col is None or 'person_id' not in data.columns:
        return pd.DataFrame()
    rows = data.copy()
    rows['purpose'] = rows[purpose_col]
    rows = rows[rows['purpose'].isin(PURPOSES)].copy()
    if rows.empty:
        return rows
    time_col = next((c for c in ('STARTIME', 'start_time', 'depart') if c in rows.columns), None)
    if time_col is not None:
        rows['_planning_time'] = pd.to_numeric(rows[time_col], errors='coerce').fillna(0.0)
    else:
        rows['_planning_time'] = rows.groupby('person_id').cumcount()
    return rows.sort_values(['person_id', '_planning_time'], kind='stable')


def estimate_planning_order_model(data: pd.DataFrame, max_obs: int = 50000):
    """Estimate next-activity order from observed person-day sequences."""
    rows = _planning_order_rows(data)
    if rows.empty:
        return None

    duration_by_purpose = rows.groupby('purpose')['duration'].median().to_dict() \
        if 'duration' in rows.columns else {}
    rank_sums = rows.groupby('person_id').cumcount()
    rank_by_purpose = rows.assign(_rank=rank_sums).groupby('purpose')['_rank'].mean().to_dict()
    model = PlanningOrderModel(
        duration_by_purpose=duration_by_purpose,
        rank_by_purpose=rank_by_purpose,
    )

    choices = []
    attributes = []
    availabilities = []
    grouped = rows.groupby('person_id', sort=False)
    for _, group in grouped:
        sequence = group['purpose'].tolist()
        remaining = group['purpose'].value_counts().to_dict()
        horizon = float(group['B'].iloc[0] * 60.0) if 'B' in group.columns else 480.0
        durations = group.groupby('purpose')['duration'].median().to_dict() \
            if 'duration' in group.columns else {}
        for purpose in sequence:
            available = np.array([remaining.get(p, 0) > 0 for p in PURPOSES], dtype=bool)
            if available.sum() < 2:
                remaining[purpose] -= 1
                continue
            attributes.append(model._candidate_features(remaining, durations, horizon))
            availabilities.append(available)
            choices.append(PURPOSES.index(purpose))
            remaining[purpose] -= 1
            if len(choices) >= max_obs:
                break
        if len(choices) >= max_obs:
            break

    model.n_obs = len(choices)
    if _RRM_AVAILABLE and choices:
        try:
            X = np.asarray(attributes, dtype=float)
            y_idx = np.asarray(choices, dtype=int)
            avail = np.asarray(availabilities, dtype=float)
            n_obs, n_alt, n_attr = X.shape
            long_X = X.reshape(n_obs * n_alt, n_attr)
            long_y = np.zeros(n_obs * n_alt, dtype=int)
            long_y[np.arange(n_obs) * n_alt + y_idx] = 1
            long_alts = np.tile(np.arange(1, n_alt + 1), n_obs)
            long_ids = np.repeat(np.arange(1, n_obs + 1), n_alt)
            rrm = RandomRegret()
            rrm.setup(
                X=long_X,
                y=long_y,
                varnames=model.feature_names,
                alts=long_alts,
                ids=long_ids,
                isvars=[],
                avail=avail.ravel(),
                maxiter=300,
            )
            rrm.fit(compute_inference=False)
            if np.isfinite(getattr(rrm, 'loglik', np.nan)):
                model.rrm = rrm
                model.estimator = 'searchlibrium_rrm'
                # maxiter=300 can cut the fit short; SearchLibrium now
                # re-polishes and judges convergence on the gradient, so say
                # when it did not reach a stationary point.
                _conv = bool(getattr(rrm, 'converged', True))
                _g = float(getattr(rrm, 'grad_norm_inf', float('nan')))
                logger.info(
                    f'  planning/order: fitted SearchLibrium RRM on {len(choices)} choices'
                    f' (converged={_conv}, max|g|={_g:.1e})'
                )
                if not _conv:
                    logger.warning(
                        '  planning/order: RRM stopped short of a stationary point: '
                        f'{getattr(rrm, "fit_message", "")}'
                    )
        except Exception as exc:
            logger.warning(f'  planning/order RRM fit failed; using empirical order: {exc}')

    logger.info(
        f"  planning/order: {model.estimator}, n={model.n_obs}, "
        f"rank={{{', '.join(f'{p}:{model.rank_by_purpose[p]:.1f}' for p in PURPOSES)}}}"
    )
    return model


def apply_planning_order(activities: pd.DataFrame, model: Optional[PlanningOrderModel]) -> pd.DataFrame:
    """Stamp ``planning_order`` and ``activity_seq`` onto the activity frame.

    The only things the plan changes are those two columns: the old version
    sorted each person's block and then restored the original row order, so the
    returned frame is identical row-for-row.  This version therefore runs the
    order model on per-person numpy slices and writes the two columns back,
    instead of copying/sorting/dropping the whole activity frame once per
    person (which dominated runtime on the 610-column frame).
    """
    if model is None or activities.empty or 'person_id' not in activities.columns:
        return activities

    acts = activities.reset_index(drop=True)
    n = len(acts)
    purpose_arr = acts['purpose'].astype(str).to_numpy()
    duration_arr = pd.to_numeric(acts['predicted_duration_min'], errors='coerce') \
        .fillna(0.0).to_numpy(dtype=float)
    if {'start_of_day', 'end_of_day'}.issubset(acts.columns):
        horizon_arr = (pd.to_numeric(acts['end_of_day'], errors='coerce')
                       - pd.to_numeric(acts['start_of_day'], errors='coerce')) \
            .fillna(480.0).to_numpy(dtype=float)
    else:
        horizon_arr = np.full(n, 480.0)

    pid_codes, _ = pd.factorize(acts['person_id'], sort=False)
    order_rows = np.argsort(pid_codes, kind='stable')
    sizes = np.bincount(pid_codes)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    planning = np.zeros(n, dtype=np.int64)
    seq = np.zeros(n, dtype=np.int64)
    for start, size in zip(offsets, sizes):
        if size == 0:
            continue
        rows = order_rows[start:start + size]
        step = model.plan_order_arrays(purpose_arr[rows], duration_arr[rows],
                                       float(horizon_arr[rows[0]]))
        planning[rows] = step
        # activity_seq is the row's rank in planned order (stable ties)
        rank = np.empty(size, dtype=np.int64)
        rank[np.argsort(step, kind='stable')] = np.arange(size)
        seq[rows] = rank

    acts['planning_order'] = planning
    acts['activity_seq'] = seq
    return acts


# ---------------------------------------------------------------------------
# 3.  EXPAND PERSON → ACTIVITY ROWS
# ---------------------------------------------------------------------------

def expand_to_activities(synth: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the person-level DataFrame (Stage 2 output) to one row per
    activity instance.

    Each person with n_work_out=2, n_eat_out=1 gets 3 rows.

    Returns
    -------
    DataFrame with columns:
        person_id, activity_seq, purpose,
        + all original person columns carried through
    """
    df = synth.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    count_cols = [f'n_{p}_out' for p in PURPOSES]
    present = [c for c in count_cols if c in df.columns]
    if not present:
        logger.warning("expand_to_activities: no n_{purpose}_out columns "
                       "present and no other enumeration available.")
        return pd.DataFrame()
    ns = df[present].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    ns = ns.to_numpy(dtype=np.int64)
    ns = np.maximum(ns, 0)

    total_per_person = ns.sum(axis=1)
    n_acts = int(total_per_person.sum())
    if n_acts == 0:
        logger.warning("expand_to_activities: all n_{purpose}_out counts are "
                       "zero — returning empty frame")
        return pd.DataFrame()

    person_idx = np.repeat(np.arange(len(df)), total_per_person)
    starts = np.repeat(np.cumsum(total_per_person) - total_per_person,
                       total_per_person)
    k = np.arange(n_acts, dtype=np.int64) - starts
    if len(present) == 1:
        purpose_pos = np.zeros(n_acts, dtype=np.int64)
    else:
        # activity k-of-person falls on the first purpose whose cumulative
        # count exceeds k ( PurposeS order = column order, matches old loop)
        cum_counts = np.cumsum(ns, axis=1)
        purpose_pos = np.argmax(k[:, None] < cum_counts[person_idx], axis=1)
        assert np.all(purpose_pos < len(present))
    purpose_col_of = [PURPOSES.index(c[len('n_'):-len('_out')])
                      for c in present]
    purpose_labels = np.asarray([PURPOSES[i] for i in purpose_col_of],
                                dtype=object)[purpose_pos]

    acts = df.take(person_idx).reset_index(drop=True)
    acts['purpose'] = purpose_labels
    acts['activity_seq'] = k
    logger.info(f"Expanded {len(df)} persons → {n_acts} activities")
    return acts


# ---------------------------------------------------------------------------
# 4.  PREDICT DURATIONS WITH BUDGET RESCALING
# ---------------------------------------------------------------------------

def _placement_link_gap_min() -> float:
    """Minimum inter-activity travel allowance used by Stage 4 placement.

    Read from the placement module so the two stages cannot drift apart; the
    literal is only a fallback for running Stage 3 without Stage 4 importable.
    """
    try:
        from stage4_schedule_placement import MIN_TRAVEL_GAP
        return float(MIN_TRAVEL_GAP)
    except Exception:
        return 10.0


def reconcile_day_budget(durations, purposes, budget_min: float,
                         window_min: Optional[float] = None,
                         link_gap: Optional[float] = None,
                         budget_tolerance: float = 1.20):
    """Fit one person's activity durations into the Stage 2 budget AND day window.

    Stage 2 predicts the day window [start_of_day, end_of_day] and an
    out-of-home budget B; Stage 3 predicts every duration independently of
    both.  Nothing then stops a chain from asking for more time than the
    predicted day holds, and Stage 4 can only honour it by pushing activities
    past end_of_day.

    The activity time actually available is the window minus one minimum
    travel allowance per activity (placement inserts a gap before every
    activity, including the first):

        room    = (end_of_day - start_of_day) - n_acts * link_gap
        ceiling = min(budget_tolerance * B, room)   # compress above this

    B stays a soft budget exactly as before (`budget_tolerance` keeps the
    +20% band that preserves cross-person variation); the window is the hard
    one, so nothing is ever allocated more activity time than the day holds.
    The pass only ever *reduces* time: an episode's duration is a model output
    about that episode, not a share of the day to be topped up.

    Durations above `ceiling` are compressed on the headroom above their
    per-purpose floors, so the person's relative pattern survives and the
    floors cannot push the total back over the ceiling.  When the floors alone
    exceed `room` they are kept and the chain is reported as floor-bound
    rather than silently clipped.

    Returns
    -------
    (duration_array, info) where info carries `ceiling`, `window_bound`
    (the window, not B, set the ceiling) and `floor_bound` (floors exceed
    the room available).
    """
    d = np.asarray(durations, dtype=float).copy()
    n = d.size
    if link_gap is None:
        link_gap = _placement_link_gap_min()

    floors = np.array([DURATION_BOUNDS.get(p, (5, 480))[0] for p in purposes],
                      dtype=float)
    if n:
        d = np.maximum(d, floors)
    floor_sum = float(floors.sum())

    budget_min = float(np.clip(budget_min, 30, 960))
    room = np.inf
    floor_bound = False
    if window_min is not None and np.isfinite(window_min):
        room = float(window_min) - n * float(link_gap)
        if room < floor_sum:
            # Floors are the practical minimum for this activity list; the
            # window cannot absorb them, so report it and keep them.
            room = floor_sum
            floor_bound = True

    budget_ceiling = budget_min * float(budget_tolerance)
    ceiling = min(budget_ceiling, room)
    # Only a compression that the window forced counts as window-bound; a chain
    # that stays under both ceilings is untouched whatever the ceiling was.
    window_bound = bool(np.isfinite(room) and room < budget_ceiling
                        and float(d.sum()) > ceiling)

    total = float(d.sum()) if n else 0.0
    if n and total > ceiling:
        # Compress the headroom above the floors rather than the totals: plain
        # proportional scaling followed by a floor pass can push the sum back
        # above the ceiling, which is exactly the inconsistency being removed.
        headroom = total - floor_sum
        allowed  = max(ceiling - floor_sum, 0.0)
        if headroom > 1e-9:
            d = floors + (d - floors) * (allowed / headroom)
        else:
            d = floors.copy()

    return d, {
        'ceiling': float(ceiling),
        'window_bound': window_bound,
        'floor_bound': bool(floor_bound),
        'total': float(d.sum()) if n else 0.0,
    }


def reconcile_predicted_durations(
    acts: pd.DataFrame,
    dur_col: str = 'predicted_duration_min',
    pid_col: str = 'person_id',
    purpose_col: str = 'purpose',
    budget_col: str = 'B',
    link_gap: Optional[float] = None,
    log_audit: bool = True,
) -> pd.DataFrame:
    """Fit each person's predicted durations into the predicted day.

    This is the post-pass every set of Stage 3 durations goes through before
    Stage 4 sees it, kept in one function so validation measures the durations
    the pipeline actually emits rather than the raw model draws.  In order:

      1. enforce the per-purpose floors;
      2. reconcile the chain with the smaller of the soft budget (``B`` plus
         its tolerance band) and the day window's room, via
         :func:`reconcile_day_budget`;
      3. clip extreme outliers at the soft ceiling (1.5x the purpose bound).

    Nothing here scales a chain *up*.  An earlier version topped thin days up
    towards ``B`` (discretionary durations x1-2 whenever the chain sat under
    half the budget), which was really a calibration patch for draws that
    undershot: it lifted shopping from 11 to 20 min and escort from 13 to 24,
    but once the draws were fixed it inflated shopping by +29%, escort by
    +59% and the whole panel's mean |bias| from 7% to 25%.  ``B`` is a soft
    guide for how much time the day holds, not a target each episode has to
    fill.

    The frame needs ``person_id``, ``purpose``, ``B`` and the duration column;
    ``start_of_day``/``end_of_day`` are used as the window when present, and
    their absence only disables the window constraint (budget-only rescaling).

    Returns the frame with ``dur_col`` rewritten (same index, same rows).
    """
    n_rows = len(acts)
    if n_rows == 0:
        return acts

    dur_col_orig = pd.to_numeric(acts[dur_col], errors='coerce').to_numpy(dtype=float)
    _dur_before = float(np.nanmean(dur_col_orig))

    # ── per-purpose floors, computed once on the duration column ──────────────
    # Do NOT clip the upper bound here — hard upper-bound clipping before the
    # budget rescaling creates a spike (pile-up) at the ceiling when many
    # lognormal draws fall above the bound.  The budget rescaling pulls
    # over-long activities back proportionally, which is smoother; the soft
    # ceiling is applied after it.
    lo_map = {p: DURATION_BOUNDS.get(p, (5, 480))[0] for p in PURPOSES}
    hi_map = {p: DURATION_BOUNDS.get(p, (5, 480))[1] for p in PURPOSES}
    floor_arr = acts[purpose_col].map(lo_map).fillna(5.0).to_numpy(dtype=float)
    dur_arr = np.maximum(np.where(np.isfinite(dur_col_orig), dur_col_orig, 0.0),
                         floor_arr)

    purpose_arr = acts[purpose_col].astype(str).to_numpy()
    b_arr = (pd.to_numeric(acts[budget_col], errors='coerce').fillna(8.0)
             .to_numpy(dtype=float)
             if budget_col in acts.columns else np.full(n_rows, 8.0))
    has_window = {'start_of_day', 'end_of_day'}.issubset(acts.columns)
    if has_window:
        sod = pd.to_numeric(acts['start_of_day'], errors='coerce').to_numpy(dtype=float)
        eod = pd.to_numeric(acts['end_of_day'], errors='coerce').to_numpy(dtype=float)

    # The reconciliation is per person and reads only person_id, purpose,
    # duration, B and the window.  Grouping the WIDE activity frame (610
    # columns) once per person cost 45 s of every 20k-person chunk; the same
    # logic over position arrays costs a fraction of a second.
    _link_gap = _placement_link_gap_min() if link_gap is None else float(link_gap)
    n_window_bound = 0
    n_floor_bound  = 0
    pid_codes, _ = pd.factorize(acts[pid_col], sort=False)
    order_rows = np.argsort(pid_codes, kind='stable')
    sizes = np.bincount(pid_codes)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    for start, size in zip(offsets, sizes):
        if size == 0:
            continue
        rows = order_rows[start:start + size]
        B_hrs = float(b_arr[rows[0]])
        # B is in hours from Stage 2 — convert to minutes, and guard against B
        # already being in minutes (if > 24 hrs it's minutes)
        B_min = B_hrs * 60 if B_hrs <= 24 else B_hrs
        # cap at a realistic day length for this model (up to 16 hours = 960 min)
        B_min = float(np.clip(B_min, 30, 960))
        window_min = (float(eod[rows[0]] - sod[rows[0]]) if has_window else None)
        arr, info = reconcile_day_budget(dur_arr[rows], purpose_arr[rows], B_min,
                                         window_min=window_min,
                                         link_gap=_link_gap)
        # No upward scaling: the durations are the model's estimate of each
        # episode, and the pass only has to make the chain fit the day.
        dur_arr[rows] = arr
        n_window_bound += int(info['window_bound'])
        n_floor_bound  += int(info['floor_bound'])

    # ── Final soft ceiling: only clip extreme outliers (> 1.5x upper bound) ──
    # This prevents genuinely unrealistic values (e.g. 20-hour work activity)
    # while not creating a spike at the nominal upper bound (480, 420, etc).
    soft_hi_arr = acts[purpose_col].map(
        {p: h * 1.5 for p, h in hi_map.items()}).fillna(720.0).to_numpy(dtype=float)
    acts[dur_col] = np.clip(dur_arr, floor_arr, soft_hi_arr)

    # ── reconciliation audit ─────────────────────────────────────────────────
    if log_audit:
        _n_p   = max(int(len(sizes)), 1)
        _mean  = float(acts[dur_col].mean())
        logger.info(
            f"Stage 3 day-window reconciliation: {n_window_bound:,}/{_n_p:,} persons "
            f"({100.0 * n_window_bound / _n_p:.1f}%) had their activity time capped "
            f"by the day window rather than by B; {n_floor_bound:,} "
            f"({100.0 * n_floor_bound / _n_p:.1f}%) are floor-bound (minimum "
            f"durations already exceed the window). Mean duration "
            f"{_dur_before:.0f} → {_mean:.0f} min."
        )

    return acts


def predict_activity_durations(
    synth_s2: pd.DataFrame,
    models: dict,
    sample: bool = True,
    seed: int = 42,
    use_mdcev: bool = False,
) -> pd.DataFrame:
    """
    Predict duration for every activity instance, then reconcile each
    person's total with both the day budget B and the day window room
    (see reconcile_day_budget), so Stage 4 can place the chain inside
    [start_of_day, end_of_day].

    Parameters
    ----------
    synth_s2 : Stage 2 output — one row per person with
               n_{purpose}_out, start_of_day, end_of_day, B
    models   : output of estimate_duration_models()
    sample   : sample from lognormal (True) or use mean (False)
    seed     : random seed

    Returns
    -------
    DataFrame with one row per activity, columns:
        person_id, activity_seq, purpose,
        predicted_duration_min, predicted_duration_hrs,
        start_of_day, end_of_day, B, OTAZ, ptype, …
    """
    rng = np.random.default_rng(seed)

    # expand persons → activities
    acts = expand_to_activities(synth_s2)
    if acts.empty:
        return acts

    # ptype if missing
    if 'ptype' not in acts.columns:
        try:
            from stage1_activity_frequency import assign_ptype_column
            acts = assign_ptype_column(acts)
        except ImportError:
            acts['ptype'] = 'NonWorker'

    # derived features
    acts['start_hr'] = acts.get('start_of_day', pd.Series(480, index=acts.index)) / 60

    orig = acts.get('tour_origin',
           acts.get('OTAZ', pd.Series(0, index=acts.index))).fillna(0)
    dest = acts.get('tour_destination',
           acts.get('DTAZ', pd.Series(0, index=acts.index))).fillna(0)
    # Zone IDs are not distances — same resolution rule as estimation.
    acts['log_dist'] = _resolve_log_dist(acts)

    # budget column
    if 'B' not in acts.columns:
        acts['B'] = 8.0  # default 8-hr budget

    # ptype dummies
    all_ptypes = sorted(acts['ptype'].dropna().unique())
    for pt in all_ptypes:
        acts[f'ptype_{pt}'] = (acts['ptype'] == pt).astype(float)

    # person feature defaults
    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2,
                'WORKSTAT': 0, 'STUDYING': 0, 'incp': 8}
    for col, val in defaults.items():
        if col not in acts.columns:
            acts[col] = val
        acts[col] = acts[col].fillna(val)

    acts = _augment_duration_features(acts)
    acts['_copula_order_u'] = _activity_order_quantile(acts)
    # Rebuild GA-searchable derived features (workstat/studying/age flags,
    # INDUSTRY/occp, household flags, day aggregates, home-zone acc standing
    # in for destination acc pre-stage-5, taste interactions) so GA-selected
    # terms resolve here instead of being silently zero-filled below.
    try:
        from ga_hypothesis_testing import ensure_prediction_features
        acts = ensure_prediction_features(acts, stage_id=3)
    except Exception as _e:
        logger.warning(f"prediction feature enrichment skipped: {_e}")
    acts['predicted_duration_min'] = np.nan
    _imputed_features: dict = {}

    # predict per purpose
    for purpose in PURPOSES:
        mask = acts['purpose'] == purpose

        if not mask.any():
            continue

        if purpose not in models:
            # fallback: use median duration from bounds
            lo, hi = DURATION_BOUNDS[purpose]
            acts.loc[mask, 'predicted_duration_min'] = (lo + hi) / 2
            continue

        entry    = models[purpose]
        feats    = entry['features']
        lo, hi   = entry['bounds']
        model    = entry['model']

        subset = acts.loc[mask]
        # Absent features are imputed at the training mean (z=0); a raw 0 on a
        # z-scored column would fabricate an extreme covariate value.
        X, _imputed = duration_design_matrix(subset, feats, model, purpose)
        if _imputed:
            _imputed_features.setdefault(purpose, list(_imputed))

        _order_signal = subset['_copula_order_u'].to_numpy(dtype=float) \
            if '_copula_order_u' in subset.columns else None

        if isinstance(model, CopulaLinkedHazardDurationModel):
            if sample:
                preds = model.predict_sampled(
                    X, rng, order_signal=_order_signal
                )
            else:
                preds = model.predict_mean(
                    X, order_signal=_order_signal
                )
        elif sample:
            # ActivitySim-style: use a minimum effective sigma so individual
            # durations vary even when the model fits tightly (small sigma_).
            # Purpose-specific floors:
            #   work: 0.45 — right-skewed (shift workers, overtime, long meetings)
            #   eat:  0.40 — wider spread (quick lunch vs long dinner)
            #   others: 0.30 default
            _sigma_floors = {'work': 0.45, 'eat': 0.40}
            _eff_sigma = max(float(model.sigma_), _sigma_floors.get(purpose, 0.30))
            eta = model._eta(X)   # log-scale mean, shape (n_mask,)

            # Person-level random intercept: each person gets a consistent
            # time-preference draw (positive = prefers longer activities today,
            # negative = prefers shorter).  This creates realistic cross-person
            # variation even within the same ptype/purpose cell.
            # Aligned to acts index so the same person gets the same draw
            # across different purpose rows.
            person_ids = subset['person_id'].values if 'person_id' in subset.columns else None
            if person_ids is not None:
                # Vectorised per-person random intercept (no dict allocation):
                # np.unique returns sorted ids, so the draw sequence matches
                # the legacy {pid: rng.normal(...)} comprehension over
                # unique_pids — same values, ~90 MB per million persons
                # and several minutes of dict building saved.
                unique_pids, pid_codes = np.unique(person_ids, return_inverse=True)
                pid_offsets = rng.normal(0, 0.15, size=len(unique_pids))
                person_offsets = pid_offsets[pid_codes]
            else:
                person_offsets = rng.normal(0, 0.15, size=len(eta))

            preds = np.exp(rng.normal(eta + person_offsets, _eff_sigma))
        else:
            preds = model.predict_mean(X)

        calib = float(entry.get('calibration_factor', 1.0) or 1.0)
        preds = preds * calib
        preds = _apply_quantile_profile(preds, entry.get('observed_profile'))
        preds = np.clip(preds, lo, hi)
        acts.loc[mask, 'predicted_duration_min'] = preds

    # fill any remaining NaN with purpose midpoint
    for purpose in PURPOSES:
        lo, hi = DURATION_BOUNDS[purpose]
        mask = (acts['purpose'] == purpose) & acts['predicted_duration_min'].isna()
        acts.loc[mask, 'predicted_duration_min'] = (lo + hi) / 2

    # carry the design-matrix completeness audit out to Stage 4 callers
    acts.attrs['imputed_features'] = {
        p: list(v) for p, v in _imputed_features.items()
    }

    pid_col = 'person_id'

    # ── optional MDCEV allocation before legacy budget rescaling ───────────
    _mdcev_entry = models.get('_mdcev') if isinstance(models, dict) else None
    _mdcev_model = _mdcev_entry.get('model') if isinstance(_mdcev_entry, dict) else None
    if use_mdcev and _mdcev_model is not None:
        try:
            _updated = []
            _purpose_idx = {p: i for i, p in enumerate(PURPOSES)}
            for pid, grp in acts.groupby(pid_col):
                g = grp.copy()
                B_hrs = float(g['B'].iloc[0])
                B_min = B_hrs * 60 if B_hrs <= 24 else B_hrs
                B_min = float(np.clip(B_min, 30, 960))

                raw_by_p = g.groupby('purpose')['predicted_duration_min'].sum().reindex(PURPOSES, fill_value=0.0)
                util_shift = np.log1p(raw_by_p.values)
                util_shift = util_shift - float(np.nanmean(util_shift))
                alloc_by_p = _mdcev_model.predict(np.array([B_min]), utility_shift=util_shift)[0]

                for p in PURPOSES:
                    pm = g['purpose'] == p
                    if not pm.any():
                        continue
                    target = float(alloc_by_p[_purpose_idx[p]])
                    raw = g.loc[pm, 'predicted_duration_min'].astype(float).values
                    rsum = float(raw.sum())
                    if rsum <= 1e-9:
                        g.loc[pm, 'predicted_duration_min'] = target / max(int(pm.sum()), 1)
                    else:
                        g.loc[pm, 'predicted_duration_min'] = raw * (target / rsum)
                _updated.append(g)

            acts = pd.concat(_updated, axis=0).sort_index()
            logger.info("Stage 3: applied MDCEV allocation to person-day budgets")
        except Exception as _mdcev_pred_e:
            logger.warning(f"Stage 3 MDCEV predict failed, using legacy rescaling: {_mdcev_pred_e}")

    # ── budget rescaling per person ──────────────────────────────────────────
    # ActivitySim approach: treat the budget as a SOFT guide, not a hard
    # constraint — do NOT force every person's total to equal it, because that
    # collapses cross-person variation and makes all plans look identical.
    #
    # The budget is not the only constraint, though.  The day window from
    # Stage 2 has to hold the durations plus one placement travel allowance per
    # activity, or Stage 4 cannot honour end_of_day, so each chain is
    # reconciled against the smaller of the two.  That pass lives in
    # reconcile_predicted_durations so validation can measure the very same
    # durations the pipeline emits.
    acts = reconcile_predicted_durations(acts, pid_col=pid_col)

    planning_model = models.get('_planning_order') if isinstance(models, dict) else None
    if isinstance(planning_model, PlanningOrderModel):
        acts = apply_planning_order(acts, planning_model)

    acts = acts.drop(columns=['_copula_order_u'], errors='ignore')
    acts['predicted_duration_hrs'] = acts['predicted_duration_min'] / 60

    # ── debug: show variation achieved ───────────────────────────────────────
    logger.info(
        "Stage 3 duration variation summary:\n" +
        "\n".join(
            f"  {p}: mean={acts.loc[acts['purpose']==p,'predicted_duration_min'].mean():.0f} min  "
            f"std={acts.loc[acts['purpose']==p,'predicted_duration_min'].std():.0f} min  "
            f"[{acts.loc[acts['purpose']==p,'predicted_duration_min'].min():.0f}–"
            f"{acts.loc[acts['purpose']==p,'predicted_duration_min'].max():.0f}]"
            for p in PURPOSES if (acts['purpose'] == p).any()
        )
    )
    # Warn if any purpose has very low variation (std < 5 min)
    for p in PURPOSES:
        pmask = acts['purpose'] == p
        if pmask.sum() > 10:
            std_p = acts.loc[pmask, 'predicted_duration_min'].std()
            if std_p < 5.0:
                logger.warning(
                    f"Stage 3: {p} duration std = {std_p:.1f} min — very low. "
                    "Check that sample=True is set and sigma_ > 0 in the model."
                )

    print(
        f"Stage 3 predictions: {len(acts)} activities "
        f"across {acts[pid_col].nunique()} persons\n"
        + "\n".join(
            f"  {p}: n={int((acts['purpose']==p).sum())}, "
            f"mean={acts.loc[acts['purpose']==p,'predicted_duration_min'].mean():.1f} min"
            for p in PURPOSES if (acts['purpose'] == p).any()
        )
    )

    return acts


# ---------------------------------------------------------------------------
# 5.  SAVE / LOAD
# ---------------------------------------------------------------------------


def _write_model_html(models: dict, path: str, stage_name: str, notes: str = "") -> None:
    """Write a human-readable HTML summary of fitted model coefficients."""
    import os as _os, html as _html
    _os.makedirs(_os.path.dirname(path) if _os.path.dirname(path) else '.', exist_ok=True)
    rows_html = []
    for key1, val1 in models.items():
        if isinstance(val1, dict):
            for key2, entry in val1.items():
                if isinstance(entry, dict):
                    _emit_model_entry(rows_html, entry, f"{key1}/{key2}")
                else:
                    rows_html.append(
                        f"<tr><td>{_html.escape(str(key1))}</td>"
                        f"<td>{_html.escape(str(key2))}</td>"
                        f"<td colspan=4>{_html.escape(str(entry)[:200])}</td></tr>")
    body = "\n".join(rows_html)
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{stage_name}</title>"
        "<style>body{font-family:Arial,sans-serif;font-size:13px;margin:20px}"
        "h1{color:#2c3e50}table{border-collapse:collapse;width:100%}"
        "th{background:#2c3e50;color:white;padding:6px 10px;text-align:left}"
        "td{border:1px solid #ddd;padding:5px 10px}"
        "tr:nth-child(even){background:#f9f9f9}"
        ".ok{color:green}.warn{color:orange}.note{color:#555;font-style:italic}"
        "</style></head><body>"
        f"<h1>{stage_name} — Estimated Model Coefficients</h1>"
        f"<p class='note'>{_html.escape(notes)}</p>"
        "<table><tr><th>Group</th><th>Sub-group</th>"
        "<th>n obs</th><th>OK?</th><th>Features</th><th>Fit stat</th></tr>"
        f"{body}</table></body></html>"
    )
    with open(path, 'w', encoding='utf-8') as _f:
        _f.write(doc)
    print(f"  HTML model report → {path}")


def _emit_model_entry(rows: list, entry: dict, label: str) -> None:
    """Format one model entry into an HTML table row."""
    import html as _h
    g1, g2 = (label.split('/', 1) + [''])[:2]
    n       = entry.get('n_obs', entry.get('n', '?'))
    conv    = entry.get('converged', entry.get('converged_', None))
    feats   = entry.get('features', [])
    rmse    = entry.get('rmse', entry.get('rmse_', None))
    ll      = entry.get('ll', None)
    classes = entry.get('classes', None)
    stat = (f"RMSE={float(rmse):.3f}" if rmse is not None else
            f"LL={float(ll):.1f}"     if ll   is not None else "")
    if classes is not None:
        stat += f"  classes={list(classes)}"
    conv_s = ('<span class="ok">&#10003;</span>' if conv is True  else
              '<span class="warn">&#10007;</span>' if conv is False else '&mdash;')
    feat_s = _h.escape(', '.join(str(f) for f in feats[:12]))
    if len(feats) > 12:
        feat_s += f' &hellip; (+{len(feats)-12})'
    rows.append(
        f"<tr><td>{_h.escape(g1)}</td><td>{_h.escape(g2)}</td>"
        f"<td>{n}</td><td>{conv_s}</td>"
        f"<td style='font-size:11px'>{feat_s}</td>"
        f"<td>{_h.escape(stat)}</td></tr>"
    )
def save_models(models: dict, path: str = 'saved_data/stage3_models.pkl'):
    # Save pkl first — artifact generation is non-fatal.
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Stage 3 models saved → {path}")
    try:
        html_path = path.replace('.pkl', '_report.html')
        flat = {}
        for purpose, entry in models.items():
            if isinstance(entry, dict):
                flat[purpose] = {
                    'model': entry.get('model'),
                    'n_obs': entry.get('n_obs', '?'),
                    'features': entry.get('features', []),
                    'converged': getattr(entry.get('model'), 'converged_', True),
                    'll': entry.get('ll'),
                    'rmse': entry.get('rmse'),
                    'stat': f"sigma={entry.get('sigma','?')}"
                }
            else:
                flat[purpose] = {
                    'model': entry,
                    'n_obs': getattr(entry, 'n_obs', '?'),
                    'features': getattr(entry, 'feature_names', []),
                    'converged': getattr(entry, 'estimator', None) == 'searchlibrium_rrm',
                    'll': getattr(entry, 'll_', None),
                    'rmse': None,
                    'stat': getattr(entry, 'estimator', '')
                }
        _write_model_html({'lognormal': flat}, html_path,
                          'Stage 3 — Activity Duration',
                          notes=f'Lognormal duration models. Saved: {path}')
    except Exception as _html_e:
        print(f"  Stage 3 HTML report failed (non-fatal): {_html_e}")
    try:
        from paper_model_forms import write_stage_paper_artifacts
        write_stage_paper_artifacts('stage3', os.path.dirname(path) or 'saved_data')
    except Exception as _paper_e:
        print(f"  Stage 3 paper artifacts failed (non-fatal): {_paper_e}")


def load_models(path: str = 'saved_data/stage3_models.pkl') -> dict:
    with open(path, 'rb') as f:
        models = pickle.load(f)
    print(f"Stage 3 models loaded ← {path}")
    return models


# ---------------------------------------------------------------------------
# 6.  FULL PIPELINE WRAPPER
# ---------------------------------------------------------------------------

def run_stage3(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    synth_s2: pd.DataFrame,
    trips: Optional[pd.DataFrame] = None,
    sample: bool = True,
    seed: int = 42,
    save_path: str = 'saved_data/stage3_models.pkl',
    use_mdcev: bool = False,
    feature_override: list = None,
    duration_model: str = None,
) -> tuple:
    """Estimate duration models on HHTS, predict for Stage 2 output."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    print("=" * 55)
    print("STAGE 3 — Activity Duration")
    print("=" * 55)
    if feature_override:
        print(f"  Feature override: {feature_override}")
    if duration_model:
        print(f"  Duration model: {duration_model}")

    print("\n[1/2] Estimating duration models from HHTS...")
    models = estimate_duration_models(tours, persons, trips=trips,
                                      use_mdcev=use_mdcev,
                                      feature_override=feature_override,
                                      duration_model=duration_model)

    if save_path:
        save_models(models, save_path)

    print("\n[2/2] Predicting activity durations for synthetic population...")
    # Chunked by person to bound peak memory (same knob as predict_stage3 in
    # pipeline_logger.py): expand_to_activities / per-purpose feature builds
    # scale with rows, and a single 3.8M-person pass pushed 194 GB nodes
    # OOM (16/10 exit-137).  STAGE3_PREDICT_CHUNK=0 restores single-shot.
    try:
        _chunk = int(os.environ.get('STAGE3_PREDICT_CHUNK', '500000') or '500000')
    except ValueError:
        _chunk = 500000
    import gc as _gc
    if _chunk > 0 and len(synth_s2) > _chunk:
        _parts = []
        for _i, _start in enumerate(range(0, len(synth_s2), _chunk)):
            _sub = synth_s2.iloc[_start:_start + _chunk]
            print(f"  chunk {_i + 1}: persons {_start:,}..{_start + len(_sub) - 1:,} "
                  f"({len(_sub):,})")
            _parts.append(predict_activity_durations(_sub, models,
                                                     sample=sample, seed=seed,
                                                     use_mdcev=use_mdcev))
            del _sub
            _gc.collect()
        activities = pd.concat(_parts, ignore_index=True)
        del _parts
        _gc.collect()
    else:
        activities = predict_activity_durations(synth_s2, models,
                                                sample=sample, seed=seed,
                                                use_mdcev=use_mdcev)

    print("\nStage 3 complete. Pass activities to run_stage4().")
    return activities, models


# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------
# Usage:
#   python stage3_activity_duration.py                        # resume from saved_data/stage2_output.csv
#   python stage3_activity_duration.py --fresh                # fresh: re-run Stages 1+2 first
#   python stage3_activity_duration.py --input path/to/s2.csv # explicit Stage 2 input
#
# Resume mode (default) reads saved_data/stage2_output.csv and runs Stage 3 only.
# Stage 3 is the slowest pipeline stage (~52 min on 3.7M persons) — resuming from
# saved CSVs avoids re-running expensive upstream stages unnecessarily.

if __name__ == '__main__':
    import sys
    import os
    import logging
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('stage3_main')

    _OUT   = os.environ.get('PIPELINE_OUT_DIR', 'saved_data')
    S2_CSV = os.path.join(_OUT, 'stage2_output.csv')

    args  = sys.argv[1:]
    fresh = '--fresh' in args
    args  = [a for a in args if a != '--fresh']

    input_csv = S2_CSV
    if '--input' in args:
        idx = args.index('--input')
        input_csv = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    zone_arg = args[0] if args else None

    # ── load HHTS ────────────────────────────────────────────────────────
    log.info("Loading HHTS survey data...")
    from data_connect import Synth_Connect
    sc  = Synth_Connect()
    out = sc.process_survey(save=False)
    log.info(f"  HHTS loaded: {len(out['persons'])} persons | "
             f"{len(out['tours'])} tours | {len(out['trips'])} trips")

    if fresh:
        log.info("Fresh mode: running Stages 1 and 2 first...")
        import subprocess, sys as _sys
        cmd = [_sys.executable, 'stage1_activity_frequency.py', '--fresh']
        if zone_arg:
            cmd.append(zone_arg)
        subprocess.run(cmd, check=True)
        subprocess.run([_sys.executable, 'stage2_day_duration.py'], check=True)

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Stage 2 output not found at '{input_csv}'.\n"
            f"  Run stage2_day_duration.py first, or use --fresh to run all prior stages."
        )
    log.info(f"Loading Stage 2 output from {input_csv} ...")
    synth_s2 = pd.read_csv(input_csv)
    log.info(f"  {len(synth_s2):,} rows loaded")

    # ── run Stage 3 ───────────────────────────────────────────────────────
    # NOTE: Stage 3 expands persons → activities. On the full 3.7M-person
    # population this produces ~7M activity rows and takes ~52 minutes.
    os.makedirs(_OUT, exist_ok=True)
    activities, models = run_stage3(
        out['tours'], out['persons'], synth_s2, trips=out['trips']
    )
    activities.to_csv('saved_data/stage3_output.csv', index=False)
    log.info(f"Stage 3 output saved → {_OUT}/stage3_output.csv")
    log.info(f"  {len(activities):,} activities across "
             f"{activities['person_id'].nunique():,} persons")
    log.info(f"Pass {_OUT}/stage3_output.csv to stage4_schedule_placement.py")

