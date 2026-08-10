"""
usual_location.py
==================
Usual Work Location and Usual School Location models.

ActivitySim equivalents: workplace_location, school_location

These run ONCE per person (not per tour) and assign each worker/student
a single fixed zone as their usual workplace or school. All subsequent
work/edu tours in Stage 5 destination choice use this as the KNOWN
destination rather than re-running MNL destination choice each time.

Why this matters
----------------
Without usual locations:
  - Every work tour runs destination MNL independently → same person
    goes to a different employment zone each day, which is unrealistic
  - Tour chains (home→work→home) have inconsistent DTAZ across days
  - Stage 5 destination model can't distinguish "worker in CBD" from
    "worker in suburb" — both look the same in the utility

With usual locations:
  - Each worker has one WORK_TAZ; each student has one SCHOOL_TAZ
  - Stage 5 skips destination choice for work/edu tours and uses these
  - Mode choice still runs (same person may drive vs PT on different days)
  - Non-mandatory destination choice is unaffected

Model structure
---------------
Simple weighted sampling from zone employment/enrolment attractors,
distance-decayed from the person's home zone.

  P(work zone z | home zone o) ∝ size_term(z, 'work') × exp(-b × tt_oz)
  P(school zone z | home zone o) ∝ size_term(z, 'edu') × exp(-b × tt_oz)

This is the ActivitySim workplace_location / school_location MNL pattern
but implemented as weighted sampling (faster, same expected distribution).

Outputs
-------
Adds to synth population DataFrame:
    WORK_TAZ    : usual work zone (int, 0 if not a worker)
    SCHOOL_TAZ  : usual school zone (int, 0 if not a student)

These columns are then used in Stage 5 to bypass destination choice
for work and edu segments.
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
from typing import Optional

logger = logging.getLogger(__name__)


def _location_model_selector(name: str) -> str:
        """
        ActivitySim-style model selector normalisation.

        Mirrors activitysim.estimation.larch.location_choice.py:
            - strips common suffixes (_location, _destination, _subtour, _tour)
            - maps 'joint' to 'non_mandatory'
        """
        selector = str(name or '').strip().lower()
        selector = selector.replace('_location', '')
        selector = selector.replace('_destination', '')
        selector = selector.replace('_subtour', '')
        selector = selector.replace('_tour', '')
        if selector == 'joint':
                selector = 'non_mandatory'
        return selector


# =============================================================================
# SIZE TERM ATTRACTORS
# =============================================================================

WORK_ATTRACTOR_COLS = {
    'TOTEMP':  1.0,
    'FPSEMPN': 0.1,
    'HEREMPN': 0.05,
    'RETEMPN': 0.05,
}

# Industry-specific work attractor weights (indp codes from synth pop)
# Maps synth pop indp code → preferred landuse attractor column
INDUSTRY_ATTRACTOR = {
    # indp: {landuse_col: extra_weight}
    7:  {'RETEMPN': 0.5},          # retail
    8:  {'OTHEMP': 0.3},           # accommodation/food
    11: {'FPSEMPN': 0.5},          # financial
    12: {'FPSEMPN': 0.3},          # real estate
    13: {'FPSEMPN': 0.5},          # professional
    16: {'HEREMPN': 0.5},          # education
    17: {'HEREMPN': 0.5},          # health
}

EDU_ATTRACTOR_COLS = {
    # STUDYING code → columns and weights
    # Land-use columns: Primary/Secondary are facility-based enrollment counts,
    # university/college are OSM POI counts for higher-ed campuses.
    # AGE0519 = children aged 5-19 in zone (proxy for school-age population),
    # HSENROLL = high-school enrollment count,
    # TOTPOP = total population (baseline for zones with no recorded facilities).
    # TOTEMP  = total employment (weak proxy for tertiary student attractor).
    1: {'Primary': 1.0, 'HSENROLL': 0.25, 'AGE0519': 0.10, 'TOTPOP': 0.05},
    2: {'Secondary': 1.0, 'HSENROLL': 0.40, 'AGE0519': 0.15, 'TOTPOP': 0.05},
    3: {'university': 1.0, 'college': 0.8, 'Education': 0.3, 'TOTPOP': 0.05},
    4: {'university': 1.0, 'college': 0.8, 'Education': 0.5, 'TOTPOP': 0.05},
}

# Decay coefficients (exp(-b * tt_minutes))
# Primary → very local catchment, Secondary → wider, Tertiary → region-wide
# Default values are fallbacks; preferred values are calibrated from HHTS
# person-level SCHOOLZONE data via _calibrate_school_decay().
WORK_DECAY_B   = 0.04   # ~30 min mean commute
SCHOOL_DECAY_B = {
    1: 0.15,   # primary: ~7 min mean (strict local catchment)
    2: 0.10,   # secondary: ~10 min mean
    3: 0.04,   # tertiary PT: ~25 min mean (commuter students)
    4: 0.04,   # tertiary FT: ~25 min mean
}

# Per-study-level MAXIMUM travel time (minutes from AM skims) for school
# assignments.  Zones beyond this threshold get zero probability — prevents
# primary students being assigned 30+ km away.  0 = no hard constraint.
SCHOOL_MAX_IVT_MIN = {
    1: 20,    # primary:  ~6 km at 18 km/h (very local)
    2: 35,    # secondary: ~11 km
    3: 999,   # tertiary PT: no hard limit (region-wide)
    4: 999,   # tertiary FT: no hard limit
}

# =============================================================================
# LAND-USE CAPACITY (workplace / school place equilibrium)
# =============================================================================
# The model is a doubly-constrained gravity / spatial-interaction equilibrium:
# every home zone's workers (students) must be placed somewhere (origin totals
# preserved), while no zone may absorb more workers than it has jobs, nor more
# students than it has school places. The balancing is done with destination
# shadow prices (ARC-style), iterated so that:
#
#     expected_j = Σ_o origin_totals[o] · p[o, j]  ≤  capacity[j]
#
# for every zone j, exactly the equilibrium concept used by the stage-5
# destination model — but with land-use capacity as the target instead of
# observed HHTS marginals. Set CAPACITY_CONSTRAINED=0 to disable.
WORK_CAPACITY_COLS = ['TOTEMP', 'FPSEMPN', 'HEREMPN', 'RETEMPN',
                      'AGREMPN', 'MWTEMPN', 'OTHEMP']
# school_enrol = Primary + Secondary (see compute_accessibility.enrich_landuse);
# tertiary capacity enters via Education per the equilibrium design decision.
SCHOOL_CAPACITY_COLS = ['school_enrol', 'Education', 'Primary', 'Secondary']

# IPF / shadow-price tuning for the capacity equilibrium.
CAP_N_ITER   = 400     # max Furness-style iterations
CAP_ALPHA    = 0.6     # shadow-price step damping (0-1)
CAP_TOL      = 0.005   # allowed relative overshoot band before a zone "binds"
CAP_ABS_TOL  = 0.5     # absolute expected-count floor for (near-)zero capacity zones
CAP_CLIP_SP  = 6.0     # max |shadow price| (log balancing factor)


# =============================================================================
# ATTRACTOR VECTORS
# =============================================================================

def _work_attractor(landuse: pd.DataFrame,
                    zone_ids: np.ndarray,
                    indp: int = 0) -> np.ndarray:
    """
    Compute work attractor for each zone.
    indp: industry code from synth pop (adds industry-specific bonus).
    """
    lu = landuse.set_index('ORDER') if 'ORDER' in landuse.columns else landuse
    att = np.zeros(len(zone_ids), dtype=np.float32)

    for col, w in WORK_ATTRACTOR_COLS.items():
        if col in lu.columns:
            vals = lu.reindex(zone_ids)[col].fillna(0).clip(lower=0).values
            att += w * vals.astype(np.float32)

    # industry bonus
    extra = INDUSTRY_ATTRACTOR.get(int(indp), {})
    for col, w in extra.items():
        if col in lu.columns:
            vals = lu.reindex(zone_ids)[col].fillna(0).clip(lower=0).values
            att += w * vals.astype(np.float32)

    return np.maximum(att, 0.0)


def _edu_attractor(landuse: pd.DataFrame,
                   zone_ids: np.ndarray,
                   studying: int = 4) -> np.ndarray:
    """
    Compute education attractor for each zone.
    studying: STUDYING code (1=primary, 2=secondary, 3=PT, 4=FT).
    Falls back to TOTEMP-based attractor if education columns missing.
    """
    lu = landuse.set_index('ORDER') if 'ORDER' in landuse.columns else landuse
    cols = EDU_ATTRACTOR_COLS.get(int(studying), {'Education': 1.0})
    att = np.zeros(len(zone_ids), dtype=np.float32)

    found_any = False
    for col, w in cols.items():
        if col in lu.columns:
            vals = lu.reindex(zone_ids)[col].fillna(0).clip(lower=0).values
            if vals.sum() > 0:
                att += w * vals.astype(np.float32)
                found_any = True

    # Fallback: if education-specific columns are missing or all-zero,
    # use school-age population or enrolment proxies rather than TOTEMP.
    # Primary/secondary schools cluster where children live, not where
    # jobs are — TOTEMP is a poor proxy for school locations.
    if not found_any:
        for fallback in ['HSENROLL', 'AGE0519', 'school_enrol', 'Education',
                         'Secondary', 'Primary', 'TOTEMP']:
            if fallback in lu.columns:
                vals = lu.reindex(zone_ids)[fallback].fillna(0).clip(lower=0).values
                if vals.sum() > 0:
                    att = vals.astype(np.float32)
                    logger.debug(f"_edu_attractor: using {fallback} as fallback for STUDYING={studying}")
                    break

    return np.maximum(att, 0.0)


# =============================================================================
# SKIM LOOKUP
# =============================================================================

def _get_tt_vector(skim_flat: pd.DataFrame,
                   otaz: int,
                   zone_ids: np.ndarray,
                   tt_col: str = 'SOVTOLL_TIME__AM') -> np.ndarray:
    """
    Get travel time from otaz to all zone_ids using the flat skim DataFrame.
    Returns array of travel times (minutes), 0 for missing OD pairs.
    """
    if skim_flat is None:
        return np.ones(len(zone_ids), dtype=np.float32) * 30.0

    row = skim_flat[skim_flat['OTAZ'] == otaz]
    if len(row) == 0:
        return np.ones(len(zone_ids), dtype=np.float32) * 30.0

    # build a zone→tt mapping
    zone_tt = row.set_index('DTAZ')[tt_col]
    tt = np.array([float(zone_tt.get(int(z), 30.0)) for z in zone_ids],
                  dtype=np.float32)

    # unit normalisation
    med = np.median(tt[tt > 0]) if (tt > 0).any() else 30.0
    if med > 600:
        tt = tt / 60.0
    elif med > 200:
        tt = tt / 10.0

    return tt


# =============================================================================
# SAMPLE ONE ZONE
# =============================================================================

def _sample_zone(attractor: np.ndarray,
                 tt: np.ndarray,
                 zone_ids: np.ndarray,
                 decay_b: float,
                 rng: np.random.Generator) -> int:
    """
    Sample one destination zone proportional to attractor × exp(-b×tt).
    """
    weights = attractor * np.exp(-decay_b * tt)
    total = weights.sum()
    if total <= 0:
        mask = attractor > 0
        if mask.any():
            return int(rng.choice(zone_ids[mask]))
        return int(zone_ids[rng.integers(0, len(zone_ids))])

    probs = (weights / total).astype(np.float64)
    # renormalise to fix floating-point drift (numpy requires sum == 1 exactly)
    probs = probs / probs.sum()
    return int(rng.choice(zone_ids, p=probs))


def _zone_probs(attractor: np.ndarray,
                tt: np.ndarray,
                decay_b: float,
                max_ivt_min: float = 999.0) -> np.ndarray:
    """Return normalized destination probabilities for one chooser segment.

    Zones with tt > max_ivt_min get zero probability (hard distance cap).
    Set max_ivt_min=999 to disable the cap (all zones reachable).
    """
    weights = attractor * np.exp(-decay_b * tt)
    if max_ivt_min > 0 and max_ivt_min < 900:
        weights = np.where(tt <= max_ivt_min, weights, 0.0)
    total = weights.sum()
    if total <= 0:
        return np.array([], dtype=np.float64)
    probs = (weights / total).astype(np.float64)
    return probs / probs.sum()


# =============================================================================
# LAND-USE CAPACITY / EQUILIBRIUM HELPERS
# =============================================================================

def _zone_capacity(landuse: pd.DataFrame,
                   zone_ids: np.ndarray,
                   kind: str) -> np.ndarray:
    """
    Per-zone land-use capacity (an upper bound on assignments).

    kind == 'work'   → TOTEMP (jobs/zone) if present, else the sum of the six
                       sectoral employment columns (RETEMPN, FPSEMPN, ...).
    kind == 'school' → school_enrol (Primary + Secondary) plus Education
                       (tertiary) if present, else the raw enrolment columns.

    Returns a float array aligned with ``zone_ids``, NaN-safe, clipped at 0.
    """
    lu = landuse.set_index('ORDER') if 'ORDER' in landuse.columns else landuse
    if kind == 'work':
        if 'TOTEMP' in lu.columns:
            cols = ['TOTEMP']
        else:
            cols = [c for c in WORK_CAPACITY_COLS if c in lu.columns]
    else:
        if 'school_enrol' in lu.columns:
            cols = ['school_enrol']
            if 'Education' in lu.columns:
                cols.append('Education')
        else:
            cols = [c for c in SCHOOL_CAPACITY_COLS if c in lu.columns]

    vals = np.zeros(len(zone_ids), dtype=np.float64)
    for c in cols:
        if c in lu.columns:
            vals += lu.reindex(zone_ids)[c].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    if not cols:
        logger.warning(
            f"_zone_capacity(kind='{kind}'): no capacity columns found in landuse "
            "— capacity equilibrium disabled for this assignment"
        )
    return np.maximum(np.nan_to_num(vals, nan=0.0), 0.0)


def _capacity_equilibrium_probs(seed_probs: np.ndarray,
                                origin_totals: np.ndarray,
                                capacity: np.ndarray, *,
                                n_iter: int = None, alpha: float = None,
                                tol: float = None, clip_sp: float = None,
                                abs_tol: float = None, eps: float = 1e-12) -> np.ndarray:
    """
    Doubly-constrained gravity with destination upper bounds (equilibrium).

    Adjusts per-origin choice probabilities with destination shadow prices
    (log balancing factors) so that expected assigned counts respect each
    zone's land-use capacity while preserving every origin's total:

        expected_j = Σ_o origin_totals[o] · p[o, j]  ≤  capacity[j]

    seed_probs    : (n_orig, n_zone) MNL probabilities (rows sum to 1)
    origin_totals : (n_orig,) chooser counts per origin (fixed, preserved)
    capacity      : (n_zone,) per-zone upper bound (jobs / school places)

    Returns adjusted probabilities with identical row sums. This is the
    ARC-style shadow-price / IPF equilibrium from the stage-5 destination
    model, applied here against land-use capacity rather than observed counts.
    """
    n_iter  = CAP_N_ITER if n_iter  is None else n_iter
    alpha   = CAP_ALPHA  if alpha   is None else alpha
    tol     = CAP_TOL    if tol     is None else tol
    abs_tol = CAP_ABS_TOL if abs_tol is None else abs_tol
    clip_sp = CAP_CLIP_SP if clip_sp is None else clip_sp

    P = np.asarray(seed_probs, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] == 0:
        return P
    O = np.asarray(origin_totals, dtype=np.float64).reshape(-1)
    C = np.asarray(capacity, dtype=np.float64).reshape(-1)
    if O.shape[0] != P.shape[0] or C.shape[0] != P.shape[1]:
        return P
    if np.nanmax(np.abs(O)) == 0:
        return P

    # Zones with zero attractor cannot receive mass — leave them untouched.
    active = (np.nan_to_num(P.sum(axis=0)) > 0) & np.isfinite(C)
    lam    = np.zeros(P.shape[1], dtype=np.float64)
    logP   = np.log(np.clip(P, eps, None))

    def _renorm(u):
        u = u - u.max(axis=1, keepdims=True)
        e = np.exp(np.clip(u, -500, 500))
        s = e.sum(axis=1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        return e / s

    p    = _renorm(logP + lam)
    pred = O @ p
    # A zone "binds" when expected assignment exceeds its capacity — relative
    # to capacity, with an absolute floor so (near-)zero-capacity zones only
    # need expected mass near zero (realised hard-cap draw makes it exactly 0).
    _over_test = lambda _pred: active & (_pred > np.maximum(C * (1.0 + tol), abs_tol))
    for _it in range(max(int(n_iter), 1)):
        p    = _renorm(logP + lam)
        pred = O @ p
        over = _over_test(pred)
        if not over.any():
            break
        ratio = np.clip(C[over] / np.maximum(pred[over], eps), 1e-4, 1e4)
        lam[over] += alpha * np.log(ratio)
        lam = np.clip(lam, -clip_sp, 0.0)

    if _it >= max(int(n_iter), 1) - 1 and _over_test(pred).any():
        logger.warning(
            f"_capacity_equilibrium_probs: did not fully converge in {int(n_iter)} "
            "iterations — residual zone overcapacity will be handled by the "
            "realised (hard-cap) draw/fallback"
        )

    return _renorm(logP + lam)


def _capped_multinomial_draws(n: int,
                              probs: np.ndarray,
                              caps: np.ndarray,
                              rng: np.random.Generator):
    """
    Allocate ``n`` choosers to zones proportionally to ``probs`` while leaving
    no zone with more than its remaining cap. Returns (alloc, leftover):
    alloc[j] choosers assigned to zone j; leftover = n - alloc.sum() if total
    capacity was exhausted before all ``n`` choosers could be placed.
    """
    n = int(n)
    n_zone = len(probs)
    if n <= 0 or n_zone == 0:
        return np.zeros(n_zone, dtype=np.int64), 0
    q     = np.asarray(probs, dtype=np.float64).reshape(-1)
    avail = np.rint(np.clip(np.asarray(caps, dtype=np.float64), 0, None)).astype(np.int64)
    alloc = np.zeros(n_zone, dtype=np.int64)
    n_left = n
    while n_left > 0:
        w = q * (avail > 0).astype(np.float64)
        s = w.sum()
        if s <= 0:
            break
        w    = w / s
        take = min(n_left, int(avail.sum()))
        if take <= 0:
            break
        draw = np.minimum(rng.multinomial(take, w), avail)
        alloc += draw
        avail -= draw
        n_left -= int(draw.sum())
    return alloc, n_left


def _assign_choosers_capacity(probs_dict: dict,
                              chooser_df: pd.DataFrame,
                              otaz_col: str,
                              combo_col: str,
                              zone_ids: np.ndarray,
                              capacity: np.ndarray,
                              rng: np.random.Generator,
                              logger,
                              label: str = 'work'):
    """
    Assign one destination zone per chooser under a doubly-constrained
    capacity equilibrium.

    probs_dict : {(otaz, combo): np.ndarray} seed MNL probabilities per origin
                 class; overwritten in place with the equilibrium version.
    chooser_df : DataFrame with ``otaz_col`` + ``combo_col``, one row/chooser
                 (order matches the caller's row mask).
    capacity   : (n_zone,) land-use upper bound (jobs / school places).

    Returns (assign, report):
        assign   int array, len == len(chooser_df), per-chooser zone
        report   {'overflow': # choosers that could not fit and fell back home,
                  'zones_at_cap': # zones at their capacity limit}
    """
    zc   = np.asarray(zone_ids, dtype=np.int64)
    n_zone = len(zc)
    cap  = np.rint(np.clip(np.asarray(capacity, dtype=np.float64), 0, None)).astype(np.int64)
    if len(chooser_df) == 0:
        return np.zeros(0, dtype=np.int64), {'overflow': 0, 'zones_at_cap': 0}

    grp     = chooser_df[[otaz_col, combo_col]].astype(int)
    indices = grp.groupby([otaz_col, combo_col]).indices
    keys    = [(int(k[0]), int(k[1])) for k in indices]
    counts  = {k: int(len(indices[k])) for k in keys}

    # ── destination shadow-price equilibrium across ALL origin classes ──
    def _p1d(k):
        v = np.asarray(probs_dict[k], dtype=np.float64).reshape(-1)
        return v

    valid = [k for k in keys
             if probs_dict.get(k) is not None
             and not np.isnan(_p1d(k)).any()
             and _p1d(k).sum() > 0
             and len(_p1d(k)) == n_zone]
    if valid:
        P     = np.stack([_p1d(k) for k in valid])
        O     = np.array([counts[k] for k in valid], dtype=np.float64)
        P_adj = _capacity_equilibrium_probs(P, O, cap.astype(np.float64))
        for i, k in enumerate(valid):
            probs_dict[k] = P_adj[i]

    # ── realised sampling with hard per-zone caps ──
    remaining = cap.copy()
    assign    = np.zeros(len(chooser_df), dtype=np.int64)
    overflow  = 0
    for k in keys:
        rows = indices[k]
        cnt  = counts[k]
        p    = _p1d(k)
        if np.isfinite(p).all() and p.sum() > 0:
            alloc, leftover = _capped_multinomial_draws(cnt, p, remaining, rng)
            zone_vals = zc[np.repeat(np.arange(n_zone), alloc)]
        else:
            alloc     = np.zeros(n_zone, dtype=np.int64)
            leftover  = cnt
            zone_vals = np.empty(0, dtype=np.int64)
        if leftover > 0:
            home_idx = int(np.argmin(np.abs(zc - int(k[0]))))
            zone_vals = np.append(zone_vals, np.full(leftover, zc[home_idx], dtype=np.int64))
            overflow += leftover
            logger.warning(
                f"    [{label}] {leftover} choosers from home {int(k[0])} "
                "exceeded remaining zone capacity — home-zone fallback")
        assign[rows] = zone_vals
        remaining -= alloc

    report = {'overflow': int(overflow),
              'zones_at_cap': int((remaining <= 0).sum())}
    return assign, report


# =============================================================================
# DECAY CALIBRATION FROM HHTS
# =============================================================================

def _calibrate_school_decay_from_hhts(
    hhts_persons: pd.DataFrame,
    landuse: pd.DataFrame,
    zone_ids: np.ndarray,
    get_tt_fn,
    logger,
) -> dict:
    """
    Calibrate school decay parameters from HHTS person-level SCHOOLZONE data.

    For each STUDYING code with enough observations, performs a 1-D
    line search for the decay_b that maximises the log-likelihood of
    assigned school zones under the gravity model:

        P(z | o, s) ∝ edu_attractor(z, s) × exp(-b_s × tt(o, z))

    Returns a dict {studying_code: decay_b} for codes with ≥10 observed
    school zones.  Returns None if HHTS data is unavailable or missing
    the SCHOOLZONE column.
    """
    if hhts_persons is None or hhts_persons.empty:
        return None

    # Find the school-zone column in HHTS persons
    sz_col = next((c for c in ['SCHOOL_TAZ', 'SCHOOLZONE', 'SCHOOL_ZONE', 'staz']
                   if c in hhts_persons.columns), None)
    otaz_col = next((c for c in ['OTAZ', 'ORDER']
                     if c in hhts_persons.columns), None)
    studying_col = 'STUDYING'

    if sz_col is None or otaz_col is None:
        logger.info(
            "_calibrate_school_decay: HHTS persons missing SCHOOLZONE/OTAZ — "
            "using heuristic decay defaults"
        )
        return None

    persons = hhts_persons.copy()
    persons = persons[
        persons[sz_col].notna() &
        persons[otaz_col].notna() &
        (persons[sz_col].astype(float) > 0) &
        (persons[otaz_col].astype(float) > 0)
    ]
    persons['otaz_i'] = persons[otaz_col].astype(int)
    persons['szone'] = persons[sz_col].astype(int)

    # STUDYING column may not exist in all HHTS formats
    if studying_col not in persons.columns:
        # Try to infer from AGE / study status
        logger.info(
            "_calibrate_school_decay: no STUDYING column — "
            "using heuristic decay defaults"
        )
        return None

    persons['studying'] = persons[studying_col].fillna(4).astype(int)

    zone_arr = np.asarray(zone_ids, dtype=int)
    zone_to_idx = {z: i for i, z in enumerate(zone_arr)}

    # For each studying code, build observed (otaz, szone) and calibrate
    calibrated = {}
    from scipy.optimize import minimize_scalar

    for sc in sorted(persons['studying'].unique()):
        sc_data = persons[persons['studying'] == sc]
        if len(sc_data) < 10:
            continue

        # Build OTAZ → observed zone list
        obs_otaz = sc_data['otaz_i'].values.astype(int)
        obs_zone = sc_data['szone'].values.astype(int)

        # Pre-compute attractor and TT for unique OTAZs
        unique_otaz = np.unique(obs_otaz)
        ott_map = {}
        for oz in unique_otaz:
            if oz not in zone_to_idx:
                continue
            try:
                tt = get_tt_fn(int(oz))
                att = _edu_attractor(landuse, zone_arr, studying=int(sc))
                ott_map[oz] = (att, tt)
            except Exception:
                continue

        # Only zones reachable per max constraint
        max_ivt = SCHOOL_MAX_IVT_MIN.get(int(sc), 999)

        def _neg_loglik(db):
            db = float(db)
            if db <= 0.001 or db >= 0.5:
                return 1e9
            ll = 0.0
            for i in range(len(obs_otaz)):
                oz, sz = int(obs_otaz[i]), int(obs_zone[i])
                if oz not in ott_map or sz not in zone_to_idx:
                    continue
                att_i, tt_i = ott_map[oz]
                weights = att_i * np.exp(-db * tt_i)
                if max_ivt > 0 and max_ivt < 900:
                    weights = np.where(tt_i <= max_ivt, weights, 0.0)
                total = weights.sum()
                if total <= 0:
                    continue
                probs = weights / total
                p = probs[zone_to_idx[sz]]
                if p <= 0:
                    ll -= 10.0  # penalise zones with zero probability
                else:
                    ll += np.log(p)
            return -ll

        # Bracketed line search — heuristic range for decay_b
        try:
            res = minimize_scalar(
                _neg_loglik,
                bounds=(0.005, 0.30),
                method='bounded',
                options={'xatol': 0.001},
            )
            calibrated[int(sc)] = float(res.x)
            logger.info(
                f"_calibrate_school_decay: STUDYING={sc} n={len(sc_data)} "
                f"decay_b={res.x:.4f} LL={-res.fun:,.1f}"
            )
        except Exception as e:
            logger.debug(f"_calibrate_school_decay: STUDYING={sc} failed: {e}")

    return calibrated if calibrated else None


# =============================================================================
# MAIN MODEL
# =============================================================================

def assign_usual_locations(
    synth: pd.DataFrame,
    landuse: pd.DataFrame,
    skim_flat: pd.DataFrame = None,
    seed: int = 42,
    save_path: str = 'saved_data/usual_locations.csv',
    tt_col: str = 'SOVTOLL_TIME__AM',
    hhts_tours: pd.DataFrame = None,
    hhts_persons: pd.DataFrame = None,
    plots_dir: str = None,
    assign_work: bool = True,
    assign_school: bool = True,
    enforce_capacity: bool = None,
) -> pd.DataFrame:
    """
    Assign usual work zone (WORK_TAZ) and usual school zone (SCHOOL_TAZ)
    to every person in the synthetic population.

    Run BEFORE Stage 5. Stage 5 then uses these to skip destination choice
    for work and edu tours.

    Parameters
    ----------
    synth      : synthetic population DataFrame (after enrich_synth_population)
    landuse    : zone landuse DataFrame (from sc.get_land(), enriched)
    skim_flat  : flat OD skim DataFrame from _skim_flat() — if None, uses
                 distance-decay only based on zone order proximity
    seed       : random seed for reproducibility
    save_path  : CSV output path
    hhts_tours : HHTS tours DataFrame for validation plots (optional)
    hhts_persons : HHTS persons DataFrame with SCHOOLZONE / WORKZONE columns
                   for decay calibration and validation (optional).
    plots_dir  : directory to save validation plots (defaults to save_path dir)
    enforce_capacity : if True, cap assigned workers per zone by TOTEMP and
                 students by school_enrol + Education (doubly-constrained
                 shadow-price equilibrium). Default None → falls back to the
                 CAPACITY_CONSTRAINED env var; 0/1 disable/enable.

    Returns
    -------
    synth with WORK_TAZ and SCHOOL_TAZ columns added
    """
    print("=" * 55)
    print("USUAL LOCATIONS — Work & School Zone Assignment")
    print("=" * 55)

    if enforce_capacity is None:
        _env = os.environ.get('CAPACITY_CONSTRAINED', '1').strip().lower()
        enforce_capacity = _env not in ('0', 'false', 'no', 'off')
    print(f"  Capacity equilibrium: {'ON' if enforce_capacity else 'OFF'}"
          "  (set CAPACITY_CONSTRAINED=0 to disable)")

    rng = np.random.default_rng(seed)
    df  = synth.copy()
    df['WORK_TAZ']   = 0
    df['SCHOOL_TAZ'] = 0

    # Zone universe for work/school assignment, taken from the landuse — so in
    # NEW_POP=1 (371 SA2) mode these are the 371 SA2 zones, matching skims and
    # accessibility. ZONE MIGRATION (371): the old hardcoded `!= 330` (drop the
    # 330-ORDER catch-all) is gone; any genuine external stations go in
    # EXCLUDE_ZONES (comma-separated). Harmless no-op on SA2 codes anyway.
    lu = landuse.copy()
    zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]
    zone_ids = lu[zone_col].astype(int).values
    zone_ids = zone_ids[zone_ids > 0]
    _excl = os.environ.get('EXCLUDE_ZONES', '').strip()
    if _excl:
        _excl_ids = {int(x) for x in _excl.split(',') if x.strip()}
        zone_ids = zone_ids[~np.isin(zone_ids, list(_excl_ids))]
    zone_ids = np.sort(zone_ids)
    if len(zone_ids) == 0:
        raise ValueError("assign_usual_locations: no valid zone ids found in landuse")
    print(f"  Zone universe: {len(zone_ids)} zones")

    _work_cols_present = [c for c in WORK_ATTRACTOR_COLS if c in lu.columns]
    _edu_cols_present = sorted({c for cc in EDU_ATTRACTOR_COLS.values() for c in cc.keys() if c in lu.columns})
    if not _work_cols_present:
        logger.warning(
            "assign_usual_locations: no work attractor columns found in landuse "
            f"(expected one of {list(WORK_ATTRACTOR_COLS.keys())})"
        )
    if not _edu_cols_present:
        logger.warning(
            "assign_usual_locations: no school-specific attractor columns found in landuse "
            "(Education/Primary/Secondary). Fallback will be used."
        )
    logger.info(
        f"assign_usual_locations: landuse work cols={_work_cols_present} "
        f"school cols={_edu_cols_present}"
    )

    # ── identify workers and students ──────────────────────────────────
    is_worker  = df['WORKSTAT'].isin([1, 2]) if 'WORKSTAT' in df.columns \
                 else df.get('hrsp', pd.Series(1, index=df.index)).isin([2,3,4,5,6,7,8])
    is_student = df['STUDYING'].isin([1,2,3,4]) if 'STUDYING' in df.columns \
                 else df.get('stup', pd.Series(1, index=df.index)).isin([2,3,4])

    n_workers  = is_worker.sum()
    n_students = is_student.sum()
    print(f"  Workers: {n_workers} | Students: {n_students} | "
          f"assign_work={assign_work} assign_school={assign_school}")

    # ── pre-compute skim rows per home zone ────────────────────────────
    # group by OTAZ to avoid re-reading skim for every person
    otaz_col = 'OTAZ' if 'OTAZ' in df.columns else 'ORDER'
    skim_cache = {}   # otaz → tt array

    def get_tt(otaz):
        if otaz not in skim_cache:
            skim_cache[otaz] = _get_tt_vector(skim_flat, otaz, zone_ids, tt_col)
        return skim_cache[otaz]

    # ── assign WORK_TAZ ────────────────────────────────────────────────
    if assign_work:
        print("\n  Assigning work locations...")

        # Pre-compute one attractor vector per unique (indp, otaz) combination
        # rather than recomputing per-person — workers with same industry/home share it
        worker_df = df.loc[is_worker, [otaz_col, 'indp'] if 'indp' in df.columns else [otaz_col]].copy()
        if 'indp' not in worker_df.columns:
            worker_df['indp'] = 0

        work_probs = {}   # (otaz, indp) -> np.array probs over zone_ids
        unique_combos = worker_df[[otaz_col, 'indp']].drop_duplicates().values
        print(f"    {len(unique_combos)} unique (home_zone, industry) combinations")

        for otaz, indp in unique_combos:
            key = (int(otaz), int(indp))
            tt  = get_tt(int(otaz))
            att = _work_attractor(lu, zone_ids, indp=int(indp))
            work_probs[key] = _zone_probs(att, tt, WORK_DECAY_B)

        # Sample per person from cached combo probabilities, subject to
        # per-zone job capacity (TOTEMP) when the equilibrium is enabled.
        if enforce_capacity:
            work_capacity = _zone_capacity(lu, zone_ids, kind='work')
            work_assign, work_report = _assign_choosers_capacity(
                work_probs, worker_df, otaz_col, 'indp', zone_ids,
                work_capacity, rng, logger, label='work')
            print(
                f"    Capacity: {work_report['zones_at_cap']}/{len(zone_ids)} "
                "zones at cap | "
                f"{work_report['overflow']} workers fell back to home zone")
        else:
            work_assign = np.zeros(len(worker_df), dtype=np.int64)
            for i, (otaz, indp) in enumerate(worker_df[[otaz_col, 'indp']].itertuples(index=False, name=None)):
                key = (int(otaz), int(indp))
                p = work_probs.get(key)
                if p is not None and len(p) == len(zone_ids) and np.isfinite(p).all() and p.sum() > 0:
                    work_assign[i] = int(rng.choice(zone_ids, p=p))
                else:
                    tt = get_tt(int(otaz))
                    att = _work_attractor(lu, zone_ids, indp=int(indp))
                    work_assign[i] = _sample_zone(att, tt, zone_ids, WORK_DECAY_B, rng)

        df.loc[is_worker, 'WORK_TAZ'] = work_assign

        missing_work = is_worker & df['WORK_TAZ'].fillna(0).astype(int).le(0)
        if missing_work.any():
            print(f"  WARNING: {missing_work.sum()} workers had no WORK_TAZ assigned; using home zone as fallback.")
            df.loc[missing_work, 'WORK_TAZ'] = df.loc[missing_work, otaz_col].fillna(0).astype(int)

        work_dist = df.loc[is_worker, 'WORK_TAZ']
        print(f"  Work TAZ: {work_dist.nunique()} unique zones | "
              f"mode={work_dist.mode().iloc[0]} | "
              f"median={int(work_dist.median())}")
    else:
        print("\n  Skipping work location assignment (assign_work=False)")

    # ── assign SCHOOL_TAZ ──────────────────────────────────────────────
    if assign_school:
        print("\n  Assigning school locations...")

        student_df = df.loc[is_student, [otaz_col, 'STUDYING'] if 'STUDYING' in df.columns else [otaz_col]].copy()
        if 'STUDYING' not in student_df.columns:
            student_df['STUDYING'] = 4

        school_probs = {}  # (otaz, studying) -> np.array probs over zone_ids
        unique_stu_combos = student_df[[otaz_col, 'STUDYING']].drop_duplicates().values
        print(f"    {len(unique_stu_combos)} unique (home_zone, study_type) combinations")

        # ── Calibrate school decay from HHTS person-level SCHOOLZONE ────
        _calibrated_db = _calibrate_school_decay_from_hhts(
            hhts_persons, landuse, zone_ids, get_tt, logger)
        if _calibrated_db is not None:
            _n_updated = 0
            for _sc, _b in _calibrated_db.items():
                _old = SCHOOL_DECAY_B.get(_sc, _b)
                SCHOOL_DECAY_B[_sc] = _b
                if abs(_old - _b) > 0.001:
                    _n_updated += 1
            print(f"    School decay calibrated from HHTS: {_n_updated} study-levels updated")
            for _sc in sorted(_calibrated_db):
                print(f"      STUDYING={_sc}: decay_b={_calibrated_db[_sc]:.4f} (was {SCHOOL_DECAY_B.get(_sc, 0.0):.4f})")

        for otaz, studying in unique_stu_combos:
            key = (int(otaz), int(studying))
            tt  = get_tt(int(otaz))
            att = _edu_attractor(lu, zone_ids, studying=int(studying))
            _db = SCHOOL_DECAY_B.get(int(studying), 0.06)
            _mx = SCHOOL_MAX_IVT_MIN.get(int(studying), 999)
            school_probs[key] = _zone_probs(att, tt, _db, max_ivt_min=_mx)

        # Sample per person from cached combo probabilities, subject to
        # per-zone school capacity (school_enrol + Education) when enabled.
        if enforce_capacity:
            school_capacity = _zone_capacity(lu, zone_ids, kind='school')
            school_assign, school_report = _assign_choosers_capacity(
                school_probs, student_df, otaz_col, 'STUDYING', zone_ids,
                school_capacity, rng, logger, label='school')
            print(
                f"    Capacity: {school_report['zones_at_cap']}/{len(zone_ids)} "
                "zones at cap | "
                f"{school_report['overflow']} students fell back to home zone")
        else:
            school_assign = np.zeros(len(student_df), dtype=np.int64)
            for i, (otaz, studying) in enumerate(student_df[[otaz_col, 'STUDYING']].itertuples(index=False, name=None)):
                key = (int(otaz), int(studying))
                p = school_probs.get(key)
                if p is not None and len(p) == len(zone_ids) and np.isfinite(p).all() and p.sum() > 0:
                    school_assign[i] = int(rng.choice(zone_ids, p=p))
                else:
                    tt = get_tt(int(otaz))
                    att = _edu_attractor(lu, zone_ids, studying=int(studying))
                    school_assign[i] = _sample_zone(att, tt, zone_ids, SCHOOL_DECAY_B.get(int(studying), 0.06), rng)

        df.loc[is_student, 'SCHOOL_TAZ'] = school_assign

        missing_school = is_student & df['SCHOOL_TAZ'].fillna(0).astype(int).le(0)
        if missing_school.any():
            print(f"  WARNING: {missing_school.sum()} students had no SCHOOL_TAZ assigned; using home zone as fallback.")
            df.loc[missing_school, 'SCHOOL_TAZ'] = df.loc[missing_school, otaz_col].fillna(0).astype(int)

        school_dist = df.loc[is_student, 'SCHOOL_TAZ']
        print(f"  School TAZ: {school_dist.nunique()} unique zones | "
              f"mode={school_dist.mode().iloc[0]} | "
              f"median={int(school_dist.median())}")
    else:
        print("\n  Skipping school location assignment (assign_school=False)")

    # ── capacity equilibrium summary ──────────────────────────────────────
    if enforce_capacity:
        try:
            _zmap = {int(z): i for i, z in enumerate(zone_ids)}
            _cap_work = _zone_capacity(lu, zone_ids, kind='work')
            _cap_school = _zone_capacity(lu, zone_ids, kind='school')
            _cnt = np.zeros((2, len(zone_ids)), dtype=np.int64)
            col_idx = {'WORK_TAZ': 0, 'SCHOOL_TAZ': 1}
            for _col, _mask, _kind, _cap in (
                ('WORK_TAZ', is_worker, 'work', _cap_work),
                ('SCHOOL_TAZ', is_student, 'school', _cap_school)):
                _vals = df.loc[_mask, _col].to_numpy(dtype=np.int64)
                for _z in np.unique(_vals):
                    _i = _zmap.get(int(_z))
                    if _i is not None:
                        _cnt[col_idx[_col], _i] += int((_vals == _z).sum())
            print("\n  Capacity equilibrium summary:")
            if assign_work:
                _over = int((_cnt[0] > _cap_work).sum())
                print(f"    Work:   {_cnt[0].sum()} workers, "
                      f"{_over} zones assigned above TOTEMP "
                      f"(of {len(zone_ids)}) | worst cap utilisation = "
                      f"{(_cnt[0] / np.maximum(_cap_work, 1)).max():.0%}")
            if assign_school:
                _over = int((_cnt[1] > _cap_school).sum())
                print(f"    School: {_cnt[1].sum()} students, "
                      f"{_over} zones assigned above school_enrol+Education "
                      f"(of {len(zone_ids)}) | worst cap utilisation = "
                      f"{(_cnt[1] / np.maximum(_cap_school, 1)).max():.0%}")
        except Exception as _ce:
            logger.warning(f"  Capacity summary failed (non-fatal): {_ce}")

    # ── save ───────────────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                    exist_ok=True)
        out_cols = [c for c in [otaz_col, 'person_id', 'abspid',
                                'WORK_TAZ', 'SCHOOL_TAZ'] if c in df.columns]
        df[out_cols].to_csv(save_path, index=False)
        print(f"\n  Usual locations saved → {save_path}")

    # ── Validation: compare HHTS observed destinations to synth assignments ──
    _val_dir = plots_dir or (os.path.dirname(save_path) if save_path else 'saved_data')
    _validation = {}
    if hhts_tours is not None and not hhts_tours.empty:
        try:
            _validation = _validate_usual_locations(
                df, hhts_tours, is_worker, is_student, _val_dir,
                hhts_persons=hhts_persons)
        except Exception as _ve:
            logger.warning(f"  Usual location validation failed (non-fatal): {_ve}")

    _save_location_html(df, landuse, is_worker, is_student, otaz_col,
                        save_path.replace('.csv', '.html') if save_path else None,
                        validation=_validation)

    print("\nUsual location assignment complete.")
    return df


# =============================================================================
# HHTS VALIDATION
# =============================================================================

def _validate_usual_locations(
    df: pd.DataFrame,
    hhts_tours: pd.DataFrame,
    is_worker,
    is_student,
    save_dir: str,
    hhts_persons: pd.DataFrame = None,
) -> dict:
    """
    Compare synthetic WORK_TAZ / SCHOOL_TAZ assignments to observed HHTS
    usual work and school locations.

    Prefers person-level WORKZONE/SCHOOLZONE columns (more accurate) over
    tour-level destinations (which may include multi-stop chains).

    Returns a dict with:
        work_plots   : list of saved plot paths
        school_plots : list of saved plot paths
        work_metrics : {overlap, top10_coverage_obs, top10_coverage_syn}
        school_metrics: same
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    result = {'work_plots': [], 'school_plots': [], 'work_metrics': {}, 'school_metrics': {}}

    # ── Try person-level SCHOOLZONE / WORKZONE first ────────────────────────
    obs_work_person  = None
    obs_school_person = None
    if hhts_persons is not None and not hhts_persons.empty:
        for wcol in ['WORK_TAZ', 'WORKZONE', 'WORK_ZONE', 'wtaz']:
            if wcol in hhts_persons.columns:
                obs_work_person = hhts_persons[wcol].dropna().astype(int)
                obs_work_person = obs_work_person[obs_work_person > 0]
                logger.info(f"_validate: using HHTS persons {wcol} for work (n={len(obs_work_person)})")
                break
        for scol in ['SCHOOL_TAZ', 'SCHOOLZONE', 'SCHOOL_ZONE', 'staz']:
            if scol in hhts_persons.columns:
                obs_school_person = hhts_persons[scol].dropna().astype(int)
                obs_school_person = obs_school_person[obs_school_person > 0]
                logger.info(f"_validate: using HHTS persons {scol} for school (n={len(obs_school_person)})")
                break

    # ── Fall back to tour-level destinations ────────────────────────────────
    if obs_work_person is None or obs_school_person is None:
        if hhts_tours is None or hhts_tours.empty:
            return result
        purp_col = next((c for c in ['primary_purpose', 'tour_primary_purpose',
                                      'main_purpose', 'TOURPURP', 'purpose']
                         if c in hhts_tours.columns), None)
        dtaz_col = next((c for c in ['DTAZ', 'dtaz', 'tour_destination', 'dest_zone']
                         if c in hhts_tours.columns), None)
        if purp_col is None or dtaz_col is None:
            logger.warning("_validate: no person-level zones and HHTS tours missing purpose/DTAZ — skipping")
            return result

    def _top_zones(series: pd.Series, n: int = 30):
        vc = series.dropna().astype(int).value_counts()
        return vc.head(n)

    def _plot_comparison(obs_vc: pd.Series, syn_vc: pd.Series,
                          title: str, save_path: str, top_n: int = 20):
        top_zones = obs_vc.head(top_n).index.union(syn_vc.head(top_n).index)
        top_zones = sorted(top_zones)

        obs_p = obs_vc.reindex(top_zones, fill_value=0) / max(obs_vc.sum(), 1) * 100
        syn_p = syn_vc.reindex(top_zones, fill_value=0) / max(syn_vc.sum(), 1) * 100

        x = np.arange(len(top_zones)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(10, len(top_zones) * 0.55), 5))
        ax.bar(x - w / 2, obs_p.values, w, label='HHTS observed', color='#2E86AB', alpha=0.85)
        ax.bar(x + w / 2, syn_p.values, w, label='Synth assigned', color='#E84855', alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([str(z) for z in top_zones], rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Zone (SA2)')
        ax.set_ylabel('Share (%)')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _overlap_metric(obs_vc, syn_vc) -> float:
        """Bhattacharyya overlap between two zone distributions."""
        all_zones = obs_vc.index.union(syn_vc.index)
        p = obs_vc.reindex(all_zones, fill_value=0).values.astype(float)
        q = syn_vc.reindex(all_zones, fill_value=0).values.astype(float)
        p = p / max(p.sum(), 1e-9)
        q = q / max(q.sum(), 1e-9)
        return float(np.sum(np.sqrt(p * q)))

# ── Work comparison ───────────────────────────────────────────────────
    if obs_work_person is not None:
        obs_work = obs_work_person
        _work_source = 'persons'
    else:
        work_mask_hhts = hhts_tours[purp_col].astype(str).str.lower().isin(
            ['work', 'full-time work', 'part-time work', 'workfulltime', 'workparttime'])
        obs_work = hhts_tours.loc[work_mask_hhts, dtaz_col]
        _work_source = 'tours'
    syn_work_taz  = df.loc[is_worker, 'WORK_TAZ']

    if obs_work.notna().any() and syn_work_taz.gt(0).any():
        obs_vc = _top_zones(obs_work, 50)
        syn_vc = _top_zones(syn_work_taz, 50)
        p = _plot_comparison(
            obs_vc, syn_vc,
            title=f'Usual Work Location — HHTS {_work_source} (n={obs_work.notna().sum()}) vs Synth (n={syn_work_taz.gt(0).sum()})',
            save_path=os.path.join(save_dir, 'val_usual_work_location.png'),
        )
        result['work_plots'].append(p)
        ov = _overlap_metric(obs_vc, syn_vc)
        result['work_metrics'] = {
            'bhattacharyya_overlap': round(ov, 4),
            'source': _work_source,
            'n_obs': int(obs_work.notna().sum()),
            'n_syn': int(syn_work_taz.gt(0).sum()),
            'n_obs_zones': int(obs_vc.shape[0]),
            'n_syn_zones': int(syn_vc.shape[0]),
        }
        logger.info(f"  Work location validation [{_work_source}]: overlap={ov:.3f} "
                    f"({int(obs_work.notna().sum())} HHTS, "
                    f"{int(syn_work_taz.gt(0).sum())} synth)")

    # ── School comparison ─────────────────────────────────────────────────
    if obs_school_person is not None:
        obs_school = obs_school_person
        _sch_source = 'persons'
    else:
        school_mask_hhts = hhts_tours[purp_col].astype(str).str.lower().isin(
            ['edu', 'education', 'primary school', 'secondary school',
             'full-time tafe/uni', 'part-time tafe/uni', 'school', 'university'])
        obs_school = hhts_tours.loc[school_mask_hhts, dtaz_col]
        _sch_source = 'tours'
    syn_school_taz  = df.loc[is_student, 'SCHOOL_TAZ']

    if obs_school.notna().any() and syn_school_taz.gt(0).any():
        obs_vc = _top_zones(obs_school, 50)
        syn_vc = _top_zones(syn_school_taz, 50)
        p = _plot_comparison(
            obs_vc, syn_vc,
            title=f'Usual School Location — HHTS {_sch_source} (n={obs_school.notna().sum()}) vs Synth (n={syn_school_taz.gt(0).sum()})',
            save_path=os.path.join(save_dir, 'val_usual_school_location.png'),
        )
        result['school_plots'].append(p)
        ov = _overlap_metric(obs_vc, syn_vc)
        result['school_metrics'] = {
            'bhattacharyya_overlap': round(ov, 4),
            'source': _sch_source,
            'n_obs': int(obs_school.notna().sum()),
            'n_syn': int(syn_school_taz.gt(0).sum()),
            'n_obs_zones': int(obs_vc.shape[0]),
            'n_syn_zones': int(syn_vc.shape[0]),
        }
        logger.info(f"  School location validation [{_sch_source}]: overlap={ov:.3f} "
                    f"({int(obs_school.notna().sum())} HHTS, "
                    f"{int(syn_school_taz.gt(0).sum())} synth)")

    return result


# =============================================================================
# HTML REPORT
# =============================================================================

def _save_location_html(df, landuse, is_worker, is_student, otaz_col, path,
                         validation: dict = None):
    if path is None:
        return
    try:
        from collections import Counter

        work_zones = Counter(df.loc[is_worker, 'WORK_TAZ'].astype(int))
        school_zones = Counter(df.loc[is_student, 'SCHOOL_TAZ'].astype(int))
        top_work = sorted(work_zones.items(), key=lambda x: -x[1])[:10]
        top_school = sorted(school_zones.items(), key=lambda x: -x[1])[:10]

        zone_ids = (landuse['ORDER'] if 'ORDER' in landuse.columns else landuse.index).astype(int)
        work_attr = _work_attractor(landuse, zone_ids.values, indp=0)
        edu_attr = _edu_attractor(landuse, zone_ids.values, studying=4)
        work_attr_top = sorted(zip(zone_ids, work_attr), key=lambda x: -x[1])[:10]
        edu_attr_top = sorted(zip(zone_ids, edu_attr), key=lambda x: -x[1])[:10]

        def rows_html(items):
            return '\n'.join(
                f'<tr><td>{z}</td><td style="text-align:right">{n}</td></tr>'
                for z, n in items)

        def coeff_rows_html(items):
            return '\n'.join(
                f'<tr><td>{name}</td><td style="text-align:right">{value}</td></tr>'
                for name, value in items)

        work_coeffs = list(WORK_ATTRACTOR_COLS.items())
        industry_coeffs = [
            (f'indp={indp} +{col}', weight)
            for indp, extras in INDUSTRY_ATTRACTOR.items()
            for col, weight in extras.items()
        ]
        school_coeffs = [
            (f'STUDYING={studying}:{col}', weight)
            for studying, cols in EDU_ATTRACTOR_COLS.items()
            for col, weight in cols.items()
        ]
        decay_rows = [
            ('WORK_DECAY_B', WORK_DECAY_B),
            ('SCHOOL_DECAY_B (primary)', SCHOOL_DECAY_B.get(1, 0.06)),
            ('SCHOOL_DECAY_B (secondary)', SCHOOL_DECAY_B.get(2, 0.06)),
            ('SCHOOL_DECAY_B (tertiary PT)', SCHOOL_DECAY_B.get(3, 0.06)),
            ('SCHOOL_DECAY_B (tertiary FT)', SCHOOL_DECAY_B.get(4, 0.06)),
        ]

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Usual Work &amp; School Location Model</title>
<style>
  body {{font-family: Arial,sans-serif; margin:40px;}}
  h1 {{border-bottom:2px solid #1565c0; padding-bottom:8px;}}
  h2 {{color:#1565c0; margin-top:28px;}}
  h3 {{color:#333; margin-top:22px;}}
  .stats {{background:#e3f2fd; padding:8px 12px; border-radius:4px; margin-bottom:12px;}}
  table {{border-collapse:collapse; width:100%; max-width:860px; margin-bottom:18px;}}
  th {{background:#1565c0; color:white; padding:8px 12px; text-align:left;}}
  td {{padding:6px 12px; border-bottom:1px solid #eee;}}
  .half {{display:inline-block; vertical-align:top; width:48%; margin-right:2%;}}
  .note {{color:#555; font-size:0.95em; margin-top:18px;}}
</style></head>
<body>
<h1>Usual Work &amp; School Location Model</h1>
<div class="stats">
  <b>Workers assigned:</b> {is_worker.sum()} &nbsp;|&nbsp;
  <b>Unique work zones:</b> {df.loc[is_worker, 'WORK_TAZ'].nunique()} &nbsp;|&nbsp;
  <b>Students assigned:</b> {is_student.sum()} &nbsp;|&nbsp;
  <b>Unique school zones:</b> {df.loc[is_student, 'SCHOOL_TAZ'].nunique()}
</div>

<h2>Model coefficients</h2>
<p class="note">These weights define the attractor functions used to sample usual work and school zones.</p>
<div class="half">
  <h3>Work attractor size terms</h3>
  <table><thead><tr><th>Attractor term</th><th>Weight</th></tr></thead>
  <tbody>{coeff_rows_html(work_coeffs)}</tbody></table>
</div>
<div class="half">
  <h3>School attractor size terms</h3>
  <table><thead><tr><th>Study type / term</th><th>Weight</th></tr></thead>
  <tbody>{coeff_rows_html(school_coeffs)}</tbody></table>
</div>

<h3>Industry-specific work bonuses</h3>
<table><thead><tr><th>Industry (indp)</th><th>Bonus term</th><th>Weight</th></tr></thead>
<tbody>"""
        + '\n'.join(
            f'<tr><td>{indp}</td><td>{col}</td><td style="text-align:right">{weight}</td></tr>'
            for indp, extras in INDUSTRY_ATTRACTOR.items() for col, weight in extras.items()
        )
        + "</tbody></table>"
        + f"\n<h3>Decay coefficients</h3>\n<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>{coeff_rows_html(decay_rows)}</tbody></table>"
        + "\n<h2>Attractor strength — generic top zones</h2>"
        + "\n<div class=\"half\">\n  <h3>Top work attractor zones</h3>\n  <table><thead><tr><th>Zone</th><th>Attractor</th></tr></thead>\n  <tbody>"
        + '\n'.join(
            f'<tr><td>{z}</td><td style="text-align:right">{val:.1f}</td></tr>'
            for z, val in work_attr_top)
        + "</tbody></table>\n</div>\n"
        + "\n<div class=\"half\">\n  <h3>Top school attractor zones</h3>\n  <table><thead><tr><th>Zone</th><th>Attractor</th></tr></thead>\n  <tbody>"
        + '\n'.join(
            f'<tr><td>{z}</td><td style="text-align:right">{val:.1f}</td></tr>'
            for z, val in edu_attr_top)
        + "</tbody></table>\n</div>\n"
        + "\n<h2>Assigned usual locations</h2>\n"
        + "\n<div class=\"half\">\n  <h3>Top assigned work zones</h3>\n  <table><thead><tr><th>Zone</th><th>Workers</th></tr></thead>\n  <tbody>" + rows_html(top_work) + "</tbody></table>\n</div>\n"
        + "\n<div class=\"half\">\n  <h3>Top assigned school zones</h3>\n  <table><thead><tr><th>Zone</th><th>Students</th></tr></thead>\n  <tbody>" + rows_html(top_school) + "</tbody></table>\n</div>\n"
        + _validation_html_section(validation)
        + "\n<p class=\"note\">Work and school locations are sampled from attractors that combine zone size and travel-time decay. Industry bonuses are applied for specific worker types.</p>\n"
        + "<p style=\"color:#888;font-size:0.85em;margin-top:24px;\">Generated by usual_location.py — QTS/HHTS ABM Pipeline</p>\n"
        + "</body></html>"

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',
                    exist_ok=True)
        with open(path, 'w') as f:
            f.write(html)
        print(f"  HTML report saved → {path}")
    except Exception as e:
        logger.debug(f"HTML report failed: {e}")


def _validation_html_section(validation: dict) -> str:
    """Render the HHTS validation section for the HTML report."""
    if not validation:
        return ''
    parts = ['\n<h2>Validation — HHTS Observed vs Synthetic Assigned</h2>\n']
    parts.append('<p class="note">Bhattacharyya overlap measures how closely the '
                 'synthetic assignment distribution matches the observed HHTS '
                 'trip destinations (1.0 = perfect match).</p>\n')

    def _metric_html(label: str, m: dict) -> str:
        if not m:
            return f'<p><em>{label}: no data available</em></p>'
        ov = m.get('bhattacharyya_overlap', float('nan'))
        color = '#155724' if ov > 0.7 else '#856404' if ov > 0.4 else '#721c24'
        return (
            f'<h3>{label}</h3>'
            f'<div class="stats">'
            f'<b>Overlap:</b> <span style="color:{color};font-weight:bold">{ov:.3f}</span>'
            f' &nbsp;|&nbsp; <b>HHTS trips:</b> {m.get("n_obs","?")} '
            f'&nbsp;|&nbsp; <b>Synth assigned:</b> {m.get("n_syn","?")}'
            f' &nbsp;|&nbsp; <b>HHTS zones:</b> {m.get("n_obs_zones","?")} '
            f'&nbsp;|&nbsp; <b>Synth zones:</b> {m.get("n_syn_zones","?")}'
            f'</div>'
        )

    parts.append(_metric_html('Work Location Validation', validation.get('work_metrics', {})))
    if validation.get('work_plots'):
        for p in validation['work_plots']:
            img = os.path.basename(p)
            parts.append(f'<p><img src="{img}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;margin-top:8px" alt="Work validation"></p>\n')

    parts.append(_metric_html('School Location Validation', validation.get('school_metrics', {})))
    if validation.get('school_plots'):
        for p in validation['school_plots']:
            img = os.path.basename(p)
            parts.append(f'<p><img src="{img}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;margin-top:8px" alt="School validation"></p>\n')

    return ''.join(parts)


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def run_usual_locations(
    synth: pd.DataFrame,
    landuse: pd.DataFrame,
    skims=None,
    seed: int = 42,
    save_path: str = 'saved_data/usual_locations.csv',
    hhts_tours: pd.DataFrame = None,
    hhts_persons: pd.DataFrame = None,
    plots_dir: str = None,
    assign_work: bool = True,
    assign_school: bool = True,
    enforce_capacity: bool = None,
) -> pd.DataFrame:
    """
    Full pipeline entry point.  Called from pipeline_logger after CDAP,
    before Stage 1.

    If skims is provided (OMX path or handle), reads the flat OD skim table
    for travel-time-weighted zone sampling.  If None, falls back to
    proximity-only (zones sorted by ORDER distance from home zone).

    hhts_tours   : optional HHTS tours table for HHTS vs synth validation plots
    hhts_persons : optional HHTS persons table for decay calibration
                   (SCHOOLZONE/WORKZONE) and person-level validation.
    plots_dir    : where to save validation plots; defaults to save_path directory
    enforce_capacity : passes through to assign_usual_locations (True → cap
                 workers by TOTEMP, students by school_enrol + Education).
                 Default None → CAPACITY_CONSTRAINED env var.

    Returns synth with WORK_TAZ and SCHOOL_TAZ added.
    """
    skim_flat = None
    close_skims = False

    if skims is not None:
        try:
            from stage5_destination_choice import _skim_flat, _open_skims
            skims_handle, close_skims = _open_skims(skims)
            print("  Reading skims for travel-time decay...")
            skim_flat = _skim_flat(skims_handle)
            print(f"  Skim loaded: {len(skim_flat)} OD pairs")
        except Exception as e:
            logger.warning(f"Could not load skims for usual locations: {e} "
                           "— using proximity fallback")

    result = assign_usual_locations(
        synth, landuse, skim_flat=skim_flat,
        seed=seed, save_path=save_path,
        hhts_tours=hhts_tours,
        hhts_persons=hhts_persons,
        plots_dir=plots_dir,
        assign_work=assign_work,
        assign_school=assign_school,
        enforce_capacity=enforce_capacity,
    )

    if close_skims:
        try:
            skims_handle.close()
        except Exception:
            pass

    return result


def workplace_location_model(
    synth: pd.DataFrame,
    landuse: pd.DataFrame,
    skims=None,
    seed: int = 42,
    save_path: str = 'saved_data/workplace_location.csv',
    hhts_tours: pd.DataFrame = None,
    plots_dir: str = None,
    enforce_capacity: bool = None,
) -> pd.DataFrame:
    """ActivitySim-style workplace_location wrapper over usual-location assignment."""
    return run_usual_locations(
        synth=synth,
        landuse=landuse,
        skims=skims,
        seed=seed,
        save_path=save_path,
        hhts_tours=hhts_tours,
        plots_dir=plots_dir,
        assign_work=True,
        assign_school=False,
        enforce_capacity=enforce_capacity,
    )


def school_location_model(
    synth: pd.DataFrame,
    landuse: pd.DataFrame,
    skims=None,
    seed: int = 42,
    save_path: str = 'saved_data/school_location.csv',
    hhts_tours: pd.DataFrame = None,
    plots_dir: str = None,
    enforce_capacity: bool = None,
) -> pd.DataFrame:
    """ActivitySim-style school_location wrapper over usual-location assignment."""
    return run_usual_locations(
        synth=synth,
        landuse=landuse,
        skims=skims,
        seed=seed,
        save_path=save_path,
        hhts_tours=hhts_tours,
        plots_dir=plots_dir,
        assign_work=False,
        assign_school=True,
        enforce_capacity=enforce_capacity,
    )


def location_choice_model(name: str, **kwargs):
    """
    ActivitySim-style location-choice entry point.

    Supported in this module:
      - workplace_location
      - school_location
    Tour- and trip-level destination wrappers are provided in stage5_destination_choice.py.
    """
    selector = _location_model_selector(name)
    if selector == 'workplace':
        return workplace_location_model(**kwargs)
    if selector == 'school':
        return school_location_model(**kwargs)
    raise ValueError(
        f"Unsupported location model '{name}' (selector='{selector}') in usual_location.py"
    )
