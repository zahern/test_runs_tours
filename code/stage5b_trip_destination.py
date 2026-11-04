"""
stage5b_trip_destination.py
============================
Stage 5b — Trip-Level (Intermediate Stop) Destination Choice

What this stage does
---------------------
Stage 5 assigns a PRIMARY DESTINATION to each tour (e.g. the workplace
for a work tour, or the main shop for a shopping tour).  But many persons
make intermediate stops on the way to or from their primary destination.

Stage 5b assigns destinations to those INTERMEDIATE STOP trips, i.e. trips
where ``is_tour_dest_leg == 0``.  The key difference from tour-level
destination choice is that a stop must be *detour-feasible*: its location
is constrained by the need to travel from the current origin to a downstream
anchor (the tour destination on the outbound leg, or home on the return).

Architecture borrowed from ActivitySim trip_destination.py
------------------------------------------------------------
* Two-stage: (1) destination SAMPLE (importance sampling), (2) simulate from sample.
* ``is_tour_dest_leg`` flag distinguishes primary-destination legs from stops.
* Detour distance  Δd = d(origin→stop) + d(stop→anchor_down) − d(origin→anchor_down)
  is the primary deterrence measure for stops.
* Zone size term S_z is purpose-specific (same gravity spec as tour level).
* Mode logsum L_{iz} enters as accessibility from the stop zone.

Trip record schema
-------------------
Each row output by this stage has:
    trip_id, tour_id, person_id, trip_num_in_tour, is_outbound,
    is_tour_dest_leg, depart_time_min, arrive_time_min,
    origin_taz, dest_taz, purpose, mode, travel_time_min,
    detour_dist_km, anchor_down_taz

GA sub-chromosome
-----------------
``g_tripdest`` is a bit-vector over the candidate variables listed in
Section 9.3 of the mathematical programme.  BIC on the trip-level MNL
(estimated on ``is_tour_dest_leg == 0`` trips) is the fitness criterion.

Usage
-----
    from stage5b_trip_destination import run_stage5b

    trips_with_dest = run_stage5b(
        trips       = stage5_trips_df,      # Stage 5 output trip table
        persons     = persons_df,
        skims_path  = 'downcasted_matrices_base.omx',
        landuse     = landuse_df,
        g_tripdest  = None,                 # None = use all candidates
    )
"""

import os
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import openmatrix as omx

try:
    import larch_jax_enforce as _jfe5b
    _jfe5b.apply()
except Exception:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Skim column aliases  →  OMX matrix names
# (see Appendix A.1 of ga_mathematical_program.tex)
SKIM_ALIAS = {
    "AutoIVT_AM":       "SOVTOLL_TIME__AM",
    "NetworkDist":      "DIST",
    "WalkDist":         "DISTWALK",
    "BikeDist":         "DISTBIKE",
    "TransitIVT_AM":    "WLK_LOC_WLK_TOTIVT__AM",
    "AutoCrashRisk":    "CRASH",
}

# ── Trip-boundary (trip-square) constraint for intermediate stops ────────────
# A stop must lie within the trip's own origin↔anchor spatial envelope: its
# detour Δd = d(o→s)+d(s→a)−d(o→a) cannot exceed a bound that SCALES WITH the
# direct origin→anchor distance (the per-trip "square"/ellipse), plus a small
# base so short trips still allow a nearby stop — capped by an absolute maximum.
# Previously only the absolute cap applied, so a 2 km trip could detour 50 km to
# a cross-city zone; the per-trip term keeps predicted stops geographically
# plausible. All three are env-tunable.
DETOUR_MAX_KM    = float(os.environ.get('STAGE5B_DETOUR_MAX_KM',    '50.0'))  # absolute cap
DETOUR_MAX_RATIO = float(os.environ.get('STAGE5B_DETOUR_MAX_RATIO', '1.0'))   # Δd ≤ ratio·d(o→a)
DETOUR_BASE_KM   = float(os.environ.get('STAGE5B_DETOUR_BASE_KM',   '3.0'))   # + slack for short trips

# Minimum number of stop observations required to estimate a segment model.
MIN_STOP_OBS = 30

# Importance sampling of destination alternatives for estimation
# (Manski & Lerman 1977; Ben-Akiva & Lerman 1985 §8.4). The sampler is fully
# implemented (`_sample_alt_pairs`, correction enters as a locked offset), but
# is DEFAULT OFF: Larch 6.0.46 cannot fit ragged choice sets safely — the
# dense from_idca unstacks missing cells to NaN and the jax gradient guard
# aborts, while the sparse from_idce path zero-gradients on jax and segfaults
# on numba (_test_larch_sparse.py). Opt in with STAGE5B_DEST_SAMPLE_SIZE=N
# once a fixed Larch is available; 0 = full choice set (default).
DEST_SAMPLE_SIZE = int(os.environ.get('STAGE5B_DEST_SAMPLE_SIZE', '0'))

# Maximum stop observations used to *estimate* a segment's MNL. The idca
# expansion in _estimate_mnl_segment is dense (n_obs x n_zones), so at
# full-population scale (millions of stop trips - e.g. job 23661073 OOM'd
# at ~261GB estimating n=2,600,393 home stops) it must be bounded to a
# calibration-sized sample. Prediction (_predict_stop_dest) still runs over
# every stop trip, just batched, so this only affects parameter estimation.
MAX_ESTIMATION_OBS = 50_000

# Zone sample size for importance sampling (0 = all zones).
SAMPLE_SIZE = 0

# ---------------------------------------------------------------------------
# Purpose-specific size-term variable lists
# (mirrors Stage 5 tour-level gravity spec; see eq. size_term in tex)
# ---------------------------------------------------------------------------
SIZE_TERM_VARS = {
    "work":     ["TotalEmployment", "FinProfSciEmp", "HealthEduRetEmp"],
    "edu":      ["Education", "Primary", "Secondary", "HighSchoolEnrol"],
    "shopping": ["RetailEmp", "Shopping", "EATOUT"],
    "social":   ["TotalPop", "recreation_ground", "sport", "park"],
    "escort":   ["TotalHH", "childcare", "ESCORT"],
    "othmaint": ["TotalPop", "Healthcare", "community"],
    "eat":      ["EATOUT", "Entertainment", "RetailEmp"],
    "othdiscr": ["TotalPop", "Entertainment"],
}

# Land-use alias → CSV column name  (see Appendix A.2)
LANDUSE_ALIAS = {
    "TotalEmployment":  "TOTEMP",
    "FinProfSciEmp":    "FPSEMPN",
    "HealthEduRetEmp":  "HEREMPN",
    "RetailEmp":        "RETEMPN",
    "AgriMiningEmp":    "AGREMPN",
    "MfgWhlTptEmp":     "MWTEMPN",
    "OtherEmp":         "OTHEMP",
    "TotalPop":         "TOTPOP",
    "TotalHH":          "TOTHH",
    "HighSchoolEnrol":  "HSENROLL",
}

# Fixed backbone covariates (always included; not tested by GA)
BACKBONE_VARS = [
    "detour_dist_km",       # primary stop deterrence
    "log_size",             # ln(S_z) — purpose-specific gravity size term
    "logsum",               # mode-choice logsum from stop zone
    "AutoCrashRisk",        # safety skim (core hypothesis)
    "depart_time_min",      # timing constraint on stop feasibility
    "is_outbound",          # direction flag
]

# GA candidate covariates (tested by g_tripdest sub-chromosome)
#
# LEGACY list: original candidates. When ``g_tripdest is None`` only these are
# active, so legacy estimation/prediction paths are unchanged.
LEGACY_GA_CANDIDATE_VARS = [
    "TotalEmployment",
    "RetailEmp",
    "HealthEduRetEmp",
    "Shopping",
    "EATOUT",
    "Entertainment",
    "NetworkDist",          # absolute dist (collinear with AutoIVT_AM)
    "AutoIVT_AM",           # auto IVT from origin to stop
    "TransitIVT_AM",        # transit IVT from origin to stop
    "TotalPop",             # collinear with TotalHH
    "TotalHH",
    "income_hi",            # person × zone interaction
    "is_female",
    "hh_child",
    "occp_group",
    "ptype_FullTimeWorker",
    "ptype_UniversityStudent",
    "mtwp",
    "WFHWEEK",
]

# Conditional trip-destination candidates (stage-51 search):
#   * conditional on the TOUR  -> anchor distance / "same as tour anchor"
#     and tour-mode dummies (mode-specific stop behaviour),
#   * conditional on the PREVIOUS stop -> origin==previous location, so
#     "same_as_prev" (return-to-previous) and is_inbound complement the
#     absolute origin->alt distance already in NetworkDist.
CONDITIONING_GA_CANDIDATE_VARS = [
    "dist_to_anchor",       # d(alt -> downstream anchor)
    "log_dist_to_anchor",
    "same_as_anchor",       # alt == tour destination / home anchor
    "same_as_prev",         # alt == origin (previous stop)
    "is_inbound",           # return leg of the tour
    "tour_mode_auto",       # tour made by car (DA/SR)
    "tour_mode_transit",
    "tour_mode_active",     # tour made by walk/bike
]

GA_CANDIDATE_VARS = LEGACY_GA_CANDIDATE_VARS + CONDITIONING_GA_CANDIDATE_VARS


def trip_destination_feature_mask(selected_features=None):
    """Return a Stage 5b GA mask in the canonical candidate order.

    Stage 51 stores its consensus selection as feature names, while the
    estimator accepts a bit-vector.  ``None`` preserves the legacy Stage 5b
    specification; an existing binary vector is accepted for callers that
    already use the low-level contract.
    """
    if selected_features is None:
        return None

    values = list(selected_features)
    if len(values) == len(GA_CANDIDATE_VARS) and all(
        value in (0, 1, False, True, np.int8(0), np.int8(1))
        for value in values
    ):
        return [int(value) for value in values]

    selected = {str(value) for value in values}
    unknown = selected.difference(GA_CANDIDATE_VARS)
    if unknown:
        raise ValueError(
            "Unknown Stage 5b destination feature(s): "
            f"{sorted(unknown)}"
        )
    return [int(name in selected) for name in GA_CANDIDATE_VARS]

# Tour-mode codes used throughout the pipeline: 1=DA 2=SR 3=Walk 4=Bike 5=Transit
_TOUR_MODE_AUTO_CODES = (1, 2)
_TOUR_MODE_TRANSIT_CODES = (5,)
_TOUR_MODE_ACTIVE_CODES = (3, 4)
_MODE_NAME_CODES = {
    "DA": 1, "SR": 2, "WALK": 3, "BIKE": 4, "TRANSIT": 5,
}


# ---------------------------------------------------------------------------
# Skim helpers
# ---------------------------------------------------------------------------

def _load_skims(skims_path: str) -> dict[str, np.ndarray]:
    """Load required skim matrices into memory as numpy arrays.

    Keys in returned dict use the readable aliases defined in SKIM_ALIAS.
    """
    needed = set(SKIM_ALIAS.values())
    arrays: dict[str, np.ndarray] = {}
    with omx.open_file(skims_path, "r") as f:
        available = set(f.list_matrices())
        for alias, orig in SKIM_ALIAS.items():
            if orig in available:
                arrays[alias] = np.array(f[orig])
            else:
                logger.warning("Skim %s (%s) not found in %s", orig, alias, skims_path)
    return arrays


def _skim_lookup(
    skim: np.ndarray,
    origin_idx: np.ndarray,
    dest_idx: np.ndarray,
) -> np.ndarray:
    """Vectorised OD skim lookup (0-based integer indices)."""
    return skim[origin_idx, dest_idx]


def _normalise_mode_codes(rows: pd.DataFrame) -> np.ndarray:
    """Return numeric mode codes, preferring trip mode over tour mode."""
    source = "mode" if "mode" in rows.columns else "tour_mode"
    if source not in rows.columns:
        return np.ones(len(rows), dtype=int)
    values = rows[source]
    # Guard against duplicated column labels: rows[source] is then a
    # DataFrame and `.str` fails with "'DataFrame' object has no attribute
    # 'str'". Keep the first occurrence.
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    numeric = pd.to_numeric(values, errors="coerce")
    if float(numeric.notna().mean()) < 0.5:
        numeric = values.astype(str).str.strip().str.upper().map(_MODE_NAME_CODES)
    return numeric.fillna(1).astype(int).to_numpy()


def _choice_set_metrics(
    obs_rows: pd.DataFrame,
    alt_zone_ids: np.ndarray,
    skims: dict[str, np.ndarray],
    taz_index: dict[int, int],
) -> dict[str, np.ndarray]:
    """Compute mode-aware detours and feasibility for case-alternative rows.

    Active-mode trips use the available walk/bike skim rather than the road
    network skim. This makes a short work-related walk trip inherit a genuinely
    short reachable choice set instead of being treated like an ordinary
    road-based stop. Other modes retain the existing NetworkDist envelope.
    """
    n_rows = len(obs_rows)
    alt_zone_ids = np.asarray(alt_zone_ids)
    mode_codes = _normalise_mode_codes(obs_rows)
    origin = obs_rows["origin_taz"].map(taz_index).to_numpy()
    anchor = obs_rows["anchor_down_taz"].map(taz_index).to_numpy()
    alt = pd.Series(alt_zone_ids).map(taz_index).to_numpy()

    def _indices(values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        valid = pd.notna(values)
        indices = np.where(valid, values, 0).astype(int, copy=False)
        valid &= (indices >= 0) & (indices < size)
        return np.clip(indices, 0, max(size - 1, 0)), valid

    if not skims:
        return {
            "d_oa": np.zeros(n_rows), "d_aa": np.zeros(n_rows),
            "d_oo": np.zeros(n_rows), "detour": np.zeros(n_rows),
            "feasible": np.ones(n_rows, dtype=bool),
        }

    fallback = skims.get("NetworkDist")
    if fallback is None:
        return {
            "d_oa": np.zeros(n_rows), "d_aa": np.zeros(n_rows),
            "d_oo": np.zeros(n_rows), "detour": np.zeros(n_rows),
            "feasible": np.ones(n_rows, dtype=bool),
        }

    size = int(fallback.shape[0])
    o_idx, o_valid = _indices(origin, size)
    a_idx, a_valid = _indices(anchor, size)
    z_idx, z_valid = _indices(alt, size)

    def _detour(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_oa = _skim_lookup(matrix, o_idx, z_idx)
        d_aa = _skim_lookup(matrix, z_idx, a_idx)
        d_oo = _skim_lookup(matrix, o_idx, a_idx)
        return d_oa, d_aa, d_oo

    d_oa, d_aa, d_oo = _detour(fallback)
    active = np.isin(mode_codes, _TOUR_MODE_ACTIVE_CODES)
    for code, alias in ((3, "WalkDist"), (4, "BikeDist")):
        use = mode_codes == code
        if not use.any() or alias not in skims:
            continue
        mode_matrix = skims[alias]
        mode_size = int(mode_matrix.shape[0])
        if mode_size != size:
            continue
        mode_d_oa, mode_d_aa, mode_d_oo = _detour(mode_matrix)
        d_oa = np.where(use, mode_d_oa, d_oa)
        d_aa = np.where(use, mode_d_aa, d_aa)
        d_oo = np.where(use, mode_d_oo, d_oo)

    detour = np.maximum(np.asarray(d_oa + d_aa - d_oo, dtype=float), 0.0)
    direct = np.maximum(np.asarray(d_oo, dtype=float), 0.0)
    bound = np.minimum(DETOUR_MAX_KM, DETOUR_MAX_RATIO * direct + DETOUR_BASE_KM)
    feasible = (
        o_valid & a_valid & z_valid
        & np.isfinite(detour) & (detour <= bound)
    )
    return {
        "d_oa": np.asarray(d_oa), "d_aa": np.asarray(d_aa),
        "d_oo": np.asarray(d_oo), "detour": detour,
        "feasible": feasible,
    }


# ---------------------------------------------------------------------------
# Trip record builder
# ---------------------------------------------------------------------------

def build_trip_records(
    trips: pd.DataFrame,
    tour_dest_col: str = "dest_taz",
    home_taz_col: str = "home_taz",
) -> pd.DataFrame:
    """Add stop-trip metadata to a trip table.

    Expects columns:
        tour_id, person_id, trip_num_in_tour, is_outbound,
        origin_taz, dest_taz, purpose, mode,
        depart_time_min, arrive_time_min,
        travel_time_min   (optional; filled 0 if missing)

    Adds:
        is_tour_dest_leg    — 1 if this is the primary destination leg
        anchor_down_taz     — downstream anchor TAZ
    """
    df = trips.copy()

    if "travel_time_min" not in df.columns:
        df["travel_time_min"] = 0.0

    # Identify tour-level primary destination: the last outbound leg
    # (highest trip_num_in_tour among is_outbound==1 per tour).
    tour_dest_leg = (
        df[df["is_outbound"] == 1]
        .groupby("tour_id")["trip_num_in_tour"]
        .max()
        .rename("max_out_leg")
    )
    df = df.merge(tour_dest_leg, on="tour_id", how="left")
    df["is_tour_dest_leg"] = (
        (df["is_outbound"] == 1) & (df["trip_num_in_tour"] == df["max_out_leg"])
    ).astype(int)
    df.drop(columns=["max_out_leg"], inplace=True)

    # Downstream anchor:
    #   outbound stops → tour destination (dest_taz of the primary dest leg)
    #   return stops   → home_taz of the person
    tour_primary_dest = (
        df[df["is_tour_dest_leg"] == 1][["tour_id", "dest_taz"]]
        .rename(columns={"dest_taz": "tour_dest_taz"})
        .drop_duplicates("tour_id")
    )
    df = df.merge(tour_primary_dest, on="tour_id", how="left")

    if home_taz_col in df.columns:
        df["anchor_down_taz"] = np.where(
            df["is_outbound"] == 1,
            df["tour_dest_taz"].fillna(df[home_taz_col]),
            df[home_taz_col],
        )
    else:
        df["anchor_down_taz"] = df["tour_dest_taz"].fillna(df["dest_taz"])

    df.drop(columns=["tour_dest_taz"], inplace=True, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Detour distance
# ---------------------------------------------------------------------------

def compute_detour(
    df: pd.DataFrame,
    skim_dist: np.ndarray,
    taz_index: dict[int, int],
) -> pd.Series:
    """
    Compute detour distance (km) for each trip row.

    Δd = d(origin→stop) + d(stop→anchor_down) − d(origin→anchor_down)

    where 'stop' = dest_taz of the stop trip,
          'anchor_down' = anchor_down_taz of the trip.

    Returns a pd.Series with the same index as df. Rows touching an
    unmapped/out-of-range zone get NaN (not a fabricated value) so callers can
    exclude them from estimation; validation summaries skip NaN by default.
    """
    # Skim is N×N (N = number of zones in the skim, e.g. 371). taz_index maps a
    # zone id to its 0-based position, but when it is built from a landuse table
    # that carries MORE zones than the skim (or a different ordering), the mapped
    # position can exceed N-1 and overflow the skim (IndexError: index 485 out of
    # bounds for size 371). Old behavior silently substituted index 0, computing
    # a WRONG detour from zone-0 distances. Now: NaN (honest, excludable).
    _n_skim = int(skim_dist.shape[0])
    _oor_mask = None

    def _idx(col: str) -> np.ndarray:
        nonlocal _oor_mask
        raw = df[col].map(taz_index)
        bad = ~(raw.notna() & (raw >= 0) & (raw < _n_skim)).values
        _oor_mask = bad if _oor_mask is None else (_oor_mask | bad)
        return raw.where(pd.Series(~bad, index=df.index), 0).astype(int).values

    o_idx  = _idx("origin_taz")
    s_idx  = _idx("dest_taz")
    a_idx  = _idx("anchor_down_taz")
    _oor_total = int(_oor_mask.sum()) if _oor_mask is not None else 0
    if _oor_total:
        logger.warning(
            f"compute_detour: {_oor_total}/{len(df)} rows touch unmapped or "
            f"out-of-range zones for the {_n_skim}-zone skim and get NaN detour "
            "(excluded from estimation, skipped in validation means). "
            "Check that taz_index/landuse zone set matches the skim zone set.")

    d_os = _skim_lookup(skim_dist, o_idx, s_idx)
    d_sa = _skim_lookup(skim_dist, s_idx, a_idx)
    d_oa = _skim_lookup(skim_dist, o_idx, a_idx)

    detour = np.asarray(d_os + d_sa - d_oa, dtype=float)
    # Negative detour (numerical noise) → set to 0; OOR rows → NaN
    detour = np.maximum(detour, 0.0)
    if _oor_total:
        detour = np.where(_oor_mask, np.nan, detour)
    return pd.Series(detour, index=df.index, name="detour_dist_km")


# ---------------------------------------------------------------------------
# Size term (gravity)
# ---------------------------------------------------------------------------

def compute_size_term(
    alts: pd.DataFrame,
    purpose: str,
    landuse: pd.DataFrame,
    zone_id_col: str = "zone_id",
) -> np.ndarray:
    """
    Log-sum size term  ln S_z  for the given purpose.

    S_z = sum_k ( exp(delta_k) * LandUseVar_k_z )

    For simplicity the deltas are all 1.0 (equal weighting); the GA
    selects which variables to include.  Returns ln(S_z + 1) to avoid
    log(0).
    """
    vars_for_purpose = SIZE_TERM_VARS.get(purpose, ["TotalPop"])
    size = pd.Series(0.0, index=alts.index)

    lu = landuse.copy()
    # Resolve land-use aliases
    for alias, orig in LANDUSE_ALIAS.items():
        if orig in lu.columns and alias not in lu.columns:
            lu[alias] = lu[orig]

    lu_indexed = lu.set_index(zone_id_col) if zone_id_col in lu.columns else lu

    for var in vars_for_purpose:
        if var in lu_indexed.columns:
            vals = alts[zone_id_col].map(lu_indexed[var]).fillna(0.0)
            size = size + vals.values

    return np.log(np.maximum(size.values, 0.0) + 1.0)


# ---------------------------------------------------------------------------
# Destination choice estimation (MNL with gravity)
# ---------------------------------------------------------------------------

def _sample_alt_pairs(
    dest_taz,
    zone_ids,
    log_size,
    sample_size: int,
    seed: int = 42,
    feasible_by_obs: Optional[np.ndarray] = None,
):
    """Build (obs_idx, alt_idx, log_correction) choice-set pairs.

    ``sample_size`` <= 0 (or >= n_zones) returns the full cartesian set with no
    correction (legacy behaviour). Otherwise draws ``sample_size - 1`` extra
    alternatives per observation from q_j ∝ exp(log_size_j), always forcing the
    observed alternative in, and returns the Manski-Lerman correction
    ``-log(q_j)`` per pair. Callers must drop observations whose chosen zone is
    outside the choice universe first (a case with no chosen row is
    unidentifiable).
    """
    dest_arr = np.asarray(dest_taz)
    zone_arr = np.asarray(zone_ids)
    n_obs = len(dest_arr)
    n_zones = len(zone_arr)
    requested_sampling = sample_size is not None and 0 < int(sample_size) < n_zones

    if feasible_by_obs is not None:
        feasible_by_obs = np.asarray(feasible_by_obs, dtype=bool)
        if feasible_by_obs.shape != (n_obs, n_zones):
            raise ValueError(
                "feasible_by_obs must have shape (n_obs, n_zones), got "
                f"{feasible_by_obs.shape}"
            )

    if feasible_by_obs is None and (
        sample_size is None or sample_size <= 0 or sample_size >= n_zones
    ):
        obs_idx = np.repeat(np.arange(n_obs), n_zones)
        alt_idx = np.tile(np.arange(n_zones), n_obs)
        return obs_idx, alt_idx, None

    zone_to_idx = {int(z): i for i, z in enumerate(zone_arr)}
    sample_size = (n_zones if sample_size is None else int(sample_size))
    sample_size = max(2, min(sample_size, n_zones))
    w = np.exp(np.clip(np.asarray(log_size, dtype=float), -30.0, 30.0))
    w = np.clip(w, 1e-12, None)
    rng = np.random.default_rng(seed)

    all_idx = np.arange(n_zones)
    _oi, _ai, _corr = [], [], []
    for i, z in enumerate(dest_arr):
        c = zone_to_idx[int(z)]
        available = (np.flatnonzero(feasible_by_obs[i])
                     if feasible_by_obs is not None else all_idx)
        if c not in available:
            available = np.unique(np.append(available, c))
        if len(available) <= sample_size:
            alts = available
            _oi.append(np.full(len(alts), i, dtype=int))
            _ai.append(alts)
            if requested_sampling:
                _corr.append(np.zeros(len(alts), dtype=float))
            continue
        others = available[available != c]
        q = w[available]
        q = q / q.sum()
        q_by_idx = {int(a): float(prob) for a, prob in zip(available, q)}
        draw = min(sample_size - 1, len(others))
        p = np.asarray([q_by_idx[int(a)] for a in others], dtype=float)
        p = p / p.sum()
        picks = (rng.choice(others, size=draw, replace=False, p=p)
                 if draw > 0 else np.empty(0, dtype=int))
        alts = np.concatenate(([c], picks))
        _oi.append(np.full(len(alts), i, dtype=int))
        _ai.append(alts)
        _corr.append(np.asarray([-np.log(q_by_idx[int(a)]) for a in alts]))
    if not _oi:
        return (np.empty(0, dtype=int), np.empty(0, dtype=int), None)
    if not requested_sampling:
        return np.concatenate(_oi), np.concatenate(_ai), None
    if not _corr or sum(len(x) for x in _corr) != sum(len(x) for x in _ai):
        return np.concatenate(_oi), np.concatenate(_ai), None
    return (np.concatenate(_oi), np.concatenate(_ai),
            np.concatenate(_corr))


def _key_variants(value) -> tuple[str, ...]:
    """Return string forms used by xarray/pandas case and zone coordinates."""
    if pd.isna(value):
        return ()
    values = [str(value)]
    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            values.append(str(int(numeric)))
    except (TypeError, ValueError):
        pass
    return tuple(dict.fromkeys(values))


def _resolve_logsum_values(logsums, obs_rows: pd.DataFrame,
                           alt_ids: np.ndarray) -> np.ndarray:
    """Align a case-by-zone logsum surface to prepared CA rows."""
    if logsums is None:
        if "logsum" in obs_rows.columns:
            return (pd.to_numeric(obs_rows["logsum"], errors="coerce")
                    .fillna(0.0).to_numpy(float))
        return np.zeros(len(obs_rows), dtype=float)

    case_map: dict[str, int] = {}
    zone_map: dict[str, int] = {}
    if hasattr(logsums, "coords") and hasattr(logsums, "values"):
        coords = logsums.coords
        if "cases" not in coords or "zones" not in coords:
            raise ValueError("logsums must have 'cases' and 'zones' coordinates")
        values = np.asarray(logsums.values, dtype=float)
        case_values = coords["cases"].values
        zone_values = coords["zones"].values
    elif isinstance(logsums, pd.DataFrame):
        if not {"case_id", "alt_id", "logsum"}.issubset(logsums.columns):
            raise ValueError("logsum DataFrame must contain case_id, alt_id, and logsum")
        lookup = {}
        for case_id, alt_id, value in logsums[["case_id", "alt_id", "logsum"]].itertuples(index=False):
            for case_key in _key_variants(case_id):
                for alt_key in _key_variants(alt_id):
                    lookup[(case_key, alt_key)] = float(value)
        result = np.zeros(len(obs_rows), dtype=float)
        for index, (_, row) in enumerate(obs_rows.iterrows()):
            case_keys = tuple(
                key for column in ("case_id", "ptour_id", "tour_id", "trip_id")
                if column in row.index for key in _key_variants(row[column])
            )
            alt_keys = _key_variants(alt_ids[index])
            result[index] = next(
                (lookup[(case_key, alt_key)]
                 for case_key in case_keys for alt_key in alt_keys
                 if (case_key, alt_key) in lookup),
                0.0,
            )
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise TypeError("logsums must be an xarray-like surface or long DataFrame")

    if values.ndim != 2 or values.shape != (len(case_values), len(zone_values)):
        raise ValueError("logsums values must match the cases-by-zones coordinates")
    for index, value in enumerate(case_values):
        for key in _key_variants(value):
            case_map[key] = index
    for index, value in enumerate(zone_values):
        for key in _key_variants(value):
            zone_map[key] = index

    result = np.zeros(len(obs_rows), dtype=float)
    matched = 0
    for index, (_, row) in enumerate(obs_rows.iterrows()):
        case_keys = tuple(
            key for column in ("case_id", "ptour_id", "tour_id", "trip_id")
            if column in row.index for key in _key_variants(row[column])
        )
        case_index = next((case_map[key] for key in case_keys if key in case_map), None)
        zone_index = next(
            (zone_map[key] for key in _key_variants(alt_ids[index]) if key in zone_map),
            None,
        )
        if case_index is not None and zone_index is not None:
            result[index] = float(values[case_index, zone_index])
            matched += 1
    if matched == 0:
        logger.warning("Stage 5b: supplied logsums did not match any CA case/zone rows")
    elif matched < len(result):
        logger.warning("Stage 5b: aligned logsums for %d/%d CA rows; unmatched rows use 0",
                       matched, len(result))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _estimate_mnl_segment(
    stop_trips: pd.DataFrame,
    all_zones: pd.DataFrame,
    skims: dict[str, np.ndarray],
    landuse: pd.DataFrame,
    purpose: str,
    taz_index: dict[int, int],
    g_tripdest: Optional[list[int]],
    zone_id_col: str = "zone_id",
    sample_size: Optional[int] = None,
    seed: int = 42,
    ca_dir: Optional[str] = None,
    persons: Optional[pd.DataFrame] = None,
    logsums=None,
) -> Optional[dict]:
    """
    Estimate a trip-level MNL destination choice model for one purpose segment.

    Parameters
    ----------
    stop_trips  : rows from the trip table with is_tour_dest_leg == 0
                  and this purpose
    all_zones   : zone alternative table with at least zone_id column
    skims       : dict of skim matrices (alias-keyed)
    landuse     : zone-level land-use DataFrame
    purpose     : trip purpose string
    taz_index   : dict mapping TAZ code → 0-based skim matrix index
    g_tripdest  : bit vector (len = len(GA_CANDIDATE_VARS)); None = all 1s
    zone_id_col : column name for zone identifier in all_zones / landuse

    Returns a dict with keys: params, log_likelihood, bic, n_obs, purpose
    or None if insufficient observations.
    """
    n_obs = len(stop_trips)
    if n_obs < MIN_STOP_OBS:
        logger.warning(
            "Segment %s: only %d stop observations; skipping estimation",
            purpose, n_obs,
        )
        return None

    # ── Drop unmapped-zone rows (OOR detour is NaN, and the idca skim lookups
    # below map unknown zones to index 0 — same silent bias compute_detour had).
    # A row whose origin, chosen dest, or anchor is outside the skim cannot
    # inform distance deterrence; keeping it fabricates a zero detour.
    try:
        _n_skim = int(next(iter(skims.values())).shape[0]) if skims else 0
    except Exception:
        _n_skim = 0
    if _n_skim > 0:
        try:
            _mapped = lambda c: stop_trips[c].map(taz_index)
            _ok = pd.Series(True, index=stop_trips.index)
            for _c in ("origin_taz", "dest_taz", "anchor_down_taz"):
                if _c in stop_trips.columns:
                    _r = _mapped(_c)
                    _ok &= _r.notna() & (_r >= 0) & (_r < _n_skim)
            if not bool(_ok.all()):
                _drop = stop_trips.index[~_ok]
                logger.warning(
                    f"Segment {purpose}: dropping {len(_drop)}/{len(stop_trips)} "
                    f"stop(s) with zones outside the {_n_skim}-zone skim "
                    f"(e.g. {_drop.tolist()[:5]}) — unidentifiable detour")
                stop_trips = stop_trips.loc[_ok].copy()
                n_obs = len(stop_trips)
                if n_obs < MIN_STOP_OBS:
                    logger.warning(
                        f"Segment {purpose}: only {n_obs} stops left after "
                        "zone-drop; skipping estimation")
                    return None
        except Exception as _zd_e:
            logger.debug(f"Segment {purpose}: zone-drop skipped ({_zd_e})")

    # ── Person-level covariates (person-specific models) ─────────────────
    # The GA's person genes (income_hi, ptype_*, mtwp, ...) are silent no-ops
    # unless these columns reach the design: production stop tables carried
    # only WFHWEEK, so every person gene was ignored and chromosomes were
    # indistinguishable on those arms. Left-merge from persons (deduped);
    # missing -> 0. In-memory and DuckDB paths both benefit downstream.
    _PERSON_ATTACH = ['income_hi', 'income_lo', 'is_female', 'hh_child',
                      'occp_group', 'ptype_FullTimeWorker',
                      'ptype_UniversityStudent', 'mtwp', 'WFHWEEK', 'incp']
    _stop_pid = next((c for c in ('person_id', 'PERSID', 'persid')
                      if c in stop_trips.columns), None)
    if persons is not None and _stop_pid is not None:
        try:
            _pid = next((c for c in ('person_id', 'PERSID', 'persid')
                         if c in persons.columns), None)
            if _pid is not None:
                _have = [c for c in _PERSON_ATTACH
                         if c in persons.columns and c not in stop_trips.columns]
                if _have:
                    _m = persons[[_pid] + _have].copy()
                    _m[_pid] = _m[_pid].astype(str)
                    _m = _m.drop_duplicates(_pid, keep='first')
                    _s = stop_trips.copy()
                    _s['_ppkey'] = _s[_stop_pid].astype(str)
                    _s = _s.merge(_m.rename(columns={_pid: '_ppkey'}),
                                  on='_ppkey', how='left')
                    _s.drop(columns=['_ppkey'], inplace=True)
                    for _c in _have:
                        _s[_c] = pd.to_numeric(
                            _s[_c], errors='coerce').fillna(0.0)
                    stop_trips = _s
                    logger.info(f"  {purpose}: attached {len(_have)} person "
                                f"covariate(s) from persons table")
        except Exception as _pae:
            logger.warning(f"  {purpose}: person attach skipped ({_pae})")

    if g_tripdest is None:
        # Legacy default: only the ORIGINAL candidates are active so callers
        # without a gene vector reproduce the legacy estimated model exactly.
        # Conditioning candidates activate only when the stage-51 GA selects
        # them (or a caller passes them explicitly).
        active_ga = {v: 1 for v in LEGACY_GA_CANDIDATE_VARS}
        active_ga.update({v: 0 for v in CONDITIONING_GA_CANDIDATE_VARS})
    else:
        active_ga = {v: int(b) for v, b in zip(GA_CANDIDATE_VARS, g_tripdest)}

    # ── DuckDB-backed CA materialisation (optional) ────────────────────────
    # When a cache dir is supplied and duckdb is available, the wide
    # (case, alt) feature table is built ONCE per purpose with SQL joins and
    # read back with column projection, so GA re-evaluations do not rebuild
    # the design matrix in pandas.
    if ca_dir:
        try:
            from stage5b_ca_duckdb import (
                available as _ca_avail, build_or_get as _ca_build,
                load_ca as _ca_load)
            if _ca_avail():
                import os as _os3
                # Full rectangular grid (see build_or_get): the ragged sampled
                # CA unstacks to NaN in dense from_idca and every engine fails.
                # Cap cases to keep the grid tractable (full choice sets need
                # no Manski-Lerman correction, so nothing is lost but sampled
                # alternatives). Tune with STAGE5B_CA_FULLGRID_CASES.
                try:
                    _fg_cases = int(_os3.environ.get(
                        'STAGE5B_CA_FULLGRID_CASES', '1500'))
                except (ValueError, TypeError):
                    _fg_cases = 1500
                _fg_seg = stop_trips
                if _fg_cases > 0 and len(stop_trips) > _fg_cases:
                    _fg_seg = stop_trips.sample(n=_fg_cases, random_state=seed)
                _estimator = os.environ.get(
                    "STAGE5B_DEST_ESTIMATOR", "larch").strip().lower()
                _sparse_estimator = _estimator in {"mh", "gibbs", "sgd"}
                _ssz = int(sample_size if sample_size is not None
                           else DEST_SAMPLE_SIZE)
                _use_sparse = (
                    _sparse_estimator and 0 < _ssz < len(all_zones))
                _info = _ca_build(ca_dir, purpose, _fg_seg, all_zones, skims,
                                  landuse, taz_index, zone_id_col, _ssz, seed,
                                  full_grid=not _use_sparse, logsums=logsums)
                _cols = list(BACKBONE_VARS) + [v for v, a in active_ga.items() if a]
                _ca, _n = _ca_load(_info["path"], _cols,
                                   has_correction=_info["sampled"])
                if _ca is not None:
                    return _fit_idca(_ca, active_ga, _n, purpose)
        except Exception as _ca_exc:  # noqa: BLE001
            logger.warning(f"  {purpose}: duckdb CA path failed ({_ca_exc}); "
                           "falling back to in-memory design")

    # ── Build case-alternative (idca) dataset ───────────────────────────────
    zones = all_zones[[zone_id_col]].copy().reset_index(drop=True)
    n_zones = len(zones)

    # Purpose size term over the full zone universe (also the importance
    # distribution for alternative sampling).
    log_size_all = compute_size_term(
        zones.assign(**{zone_id_col: zones[zone_id_col]}),
        purpose, landuse, zone_id_col)

    # Observations whose chosen zone is outside the universe cannot enter the
    # choice sets (no chosen alternative -> unidentifiable case).
    _z2i = {int(z): i for i, z in enumerate(zones[zone_id_col].values)}
    _valid = stop_trips["dest_taz"].map(
        lambda z: pd.notna(z) and int(z) in _z2i).fillna(False)
    if not bool(_valid.all()):
        logger.warning(
            f"Segment {purpose}: dropping "
            f"{int((~_valid).sum())}/{len(stop_trips)} stop(s) whose DTAZ is "
            f"outside the {n_zones}-zone universe")
        stop_trips = stop_trips.loc[_valid].copy()
        n_obs = len(stop_trips)
        if n_obs < MIN_STOP_OBS:
            logger.warning(
                f"Segment {purpose}: only {n_obs} stops left after "
                "universe-drop; skipping estimation")
            return None

    # Choice sets: importance-sampled (Manski-Lerman corrected) or full.
    _sample_size = DEST_SAMPLE_SIZE if sample_size is None else int(sample_size)
    _sampled_choice_set = 0 < _sample_size < n_zones
    _feasible_by_obs = None
    if not _sampled_choice_set:
        # Build the same mode-aware feasibility mask used by prediction before
        # drawing alternatives. Preserve an observed destination if the survey
        # contradicts the physical envelope.
        _full_obs_idx = np.repeat(np.arange(n_obs), n_zones)
        _full_alt_ids = np.tile(zones[zone_id_col].to_numpy(), n_obs)
        _full_obs_rows = stop_trips.iloc[_full_obs_idx].reset_index(drop=True)
        _full_metrics = _choice_set_metrics(
            _full_obs_rows, _full_alt_ids, skims, taz_index)
        _feasible_by_obs = _full_metrics["feasible"].reshape(n_obs, n_zones)
        _chosen_idx = stop_trips["dest_taz"].map(_z2i).astype(int).to_numpy()
        _chosen_feasible = _feasible_by_obs[np.arange(n_obs), _chosen_idx]
        _feasible_by_obs[np.arange(n_obs), _chosen_idx] = True
        _n_repaired = int((~_chosen_feasible).sum())
        if _n_repaired:
            logger.warning(
                "Segment %s: retaining %d observed destinations outside the "
                "mode-aware detour envelope", purpose, _n_repaired)
    obs_idx, alt_idx, log_corr = _sample_alt_pairs(
        stop_trips["dest_taz"].values, zones[zone_id_col].values,
        log_size_all, _sample_size, seed,
        feasible_by_obs=_feasible_by_obs)
    if log_corr is not None:
        logger.info(
            f"  {purpose}: importance-sampled {_sample_size}/{n_zones} zones "
            f"x {n_obs} obs = {len(obs_idx):,} (case, alt) rows "
            f"(STAGE5B_DEST_SAMPLE_SIZE)")
    else:
        logger.info(f"  {purpose}: full choice set {n_zones} zones x {n_obs} obs "
                    f"= {len(obs_idx):,} rows (no sampling)")

    obs_rows = stop_trips.iloc[obs_idx].reset_index(drop=True)
    alt_rows = zones.iloc[alt_idx].reset_index(drop=True)

    ca = pd.DataFrame()
    # case_id MUST repeat across alternatives of the same observation: a unique
    # per-row id makes every case a single-alternative choice (LL=0) and the
    # model unidentifiable. (Previous code used obs_rows.index after
    # reset_index -> unique ids.)
    ca["case_id"] = obs_idx
    ca["alt_id"]  = alt_rows[zone_id_col].values
    ca["chosen"]  = (obs_rows["dest_taz"].values == alt_rows[zone_id_col].values).astype(int)
    if log_corr is not None:
        ca["log_correction"] = log_corr

    ca["logsum"] = _resolve_logsum_values(logsums, obs_rows, ca["alt_id"].to_numpy())

    # Skim lookups: origin → alternative zone
    o_idx = obs_rows["origin_taz"].map(taz_index).fillna(0).astype(int).values
    a_idx = alt_rows[zone_id_col].map(taz_index).fillna(0).astype(int).values

    # Anchor-down indices (for detour)
    anchor_idx = obs_rows["anchor_down_taz"].map(taz_index).fillna(0).astype(int).values

    # Detour and feasibility use the same mode-aware metric as prediction.
    _metrics = _choice_set_metrics(
        obs_rows, alt_rows[zone_id_col].to_numpy(), skims, taz_index)
    d_oa = _metrics["d_oa"]
    d_aa = _metrics["d_aa"]
    d_oo = _metrics["d_oo"]
    ca["detour_dist_km"] = _metrics["detour"]

    # ── Tour / previous-stop conditioning columns ───────────────────────────
    # (stage-51 candidates above; computed always, activated by the gene vector)
    ca["dist_to_anchor"] = np.asarray(d_aa, dtype=float)
    ca["log_dist_to_anchor"] = np.log1p(np.maximum(ca["dist_to_anchor"].values, 0.0))
    _alt_vals = alt_rows[zone_id_col].values
    _origin_vals = (obs_rows["origin_taz"].values if "origin_taz" in obs_rows.columns
                    else np.zeros(n_obs))
    _anchor_vals = (obs_rows["anchor_down_taz"].values if "anchor_down_taz" in obs_rows.columns
                    else np.zeros(n_obs))
    ca["same_as_anchor"] = (_alt_vals == _anchor_vals).astype(float)
    ca["same_as_prev"] = (_alt_vals == _origin_vals).astype(float)
    if "is_outbound" in obs_rows.columns:
        ca["is_inbound"] = (1 - pd.to_numeric(obs_rows["is_outbound"], errors="coerce")
                            .fillna(1).astype(int).values).astype(float)
    else:
        ca["is_inbound"] = 0.0
    _tm_arr = np.ones(n_obs, dtype=int)
    if "tour_mode" in obs_rows.columns:
        _tm_col = obs_rows["tour_mode"]
        if isinstance(_tm_col, pd.DataFrame):
            _tm_col = _tm_col.iloc[:, 0]
        _tm_num = pd.to_numeric(_tm_col, errors="coerce")
        if float(_tm_num.notna().mean()) < 0.5:
            _tm_num = (_tm_col.astype(str).str.strip()
                       .map({"DA": 1, "SR": 2, "Walk": 3, "Bike": 4, "Transit": 5}))
        _tm_arr = _tm_num.fillna(1).astype(int).values
    ca["tour_mode_auto"] = np.isin(_tm_arr, _TOUR_MODE_AUTO_CODES).astype(float)
    ca["tour_mode_transit"] = np.isin(_tm_arr, _TOUR_MODE_TRANSIT_CODES).astype(float)
    ca["tour_mode_active"] = np.isin(_tm_arr, _TOUR_MODE_ACTIVE_CODES).astype(float)

    # ── Backbone skims with collinearity guard ──────────────────────────────
    # If WalkDist/BikeDist are identical to NetworkDist (common OMX issue),
    # drop them to avoid perfect collinearity poisoning the objective.
    backbone_aliases = ["AutoIVT_AM", "NetworkDist", "AutoCrashRisk", "TransitIVT_AM"]
    for alias in backbone_aliases:
        if alias in skims:
            ca[alias] = _skim_lookup(skims[alias], o_idx, a_idx)
        else:
            ca[alias] = 0.0

    # Collinearity check: if WalkDist/BikeDist exist and equal NetworkDist, drop them
    for alias in ["WalkDist", "BikeDist"]:
        if alias in skims and "NetworkDist" in skims:
            if np.array_equal(skims[alias], skims["NetworkDist"]):
                logger.warning(
                    f"Stage 5b: {alias} identical to NetworkDist in OMX — dropping "
                    f"{alias} from model to avoid perfect collinearity"
                )
                if alias in ca.columns:
                    ca.drop(columns=[alias], inplace=True, errors="ignore")
                # Also remove from GA candidates if present
                if alias in active_ga:
                    active_ga[alias] = 0

    # Size term
    ca["log_size"] = log_size_all[alt_idx]

    # Person covariates (idco → broadcast to idca)
    person_cols = [
        "depart_time_min", "is_outbound", "is_female", "income_hi",
        "hh_child", "occp_group", "ptype_FullTimeWorker",
        "ptype_UniversityStudent", "mtwp", "WFHWEEK",
        "incp", "income_lo",
    ]
    for col in person_cols:
        if col in obs_rows.columns:
            ca[col] = obs_rows[col].values
        else:
            ca[col] = 0

    # Land-use at alternative zone
    lu = landuse.copy()
    for alias, orig in LANDUSE_ALIAS.items():
        if orig in lu.columns and alias not in lu.columns:
            lu[alias] = lu[orig]
    if zone_id_col in lu.columns:
        lu_idx = lu.set_index(zone_id_col)
        for var in ["TotalEmployment", "RetailEmp", "HealthEduRetEmp",
                    "TotalPop", "TotalHH", "Shopping", "EATOUT", "Entertainment"]:
            if var in lu_idx.columns:
                ca[var] = alt_rows[zone_id_col].map(lu_idx[var]).fillna(0.0).values
            else:
                ca[var] = 0.0

    # ── Numerical guards on design matrix ───────────────────────────────────
    # Replace any NaN/Inf with 0, clip extreme values
    for col in ca.columns:
        if ca[col].dtype in (np.float64, np.float32, np.int64, np.int32):
            ca[col] = ca[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Clip to reasonable range to avoid overflow in exp()
            if ca[col].dtype in (np.float64, np.float32):
                ca[col] = ca[col].clip(-1e6, 1e6)

    # Ensure chosen column is clean
    ca["chosen"] = ca["chosen"].astype(int).clip(0, 1)

    return _fit_idca(ca, active_ga, n_obs, purpose)


def _fit_idca(ca, active_ga, n_obs, purpose):
    """Fit the trip-destination MNL on a prepared (case_id, alt_id) frame.

    Shared by the in-memory design builder and the DuckDB CA materialiser so
    both paths run the identical engine chain / regularisation / sampling
    correction.
    """
    estimator = os.environ.get("STAGE5B_DEST_ESTIMATOR", "larch").strip().lower()
    if estimator in {"mh", "gibbs", "sgd"}:
        result = _fit_searchlibrium(ca, active_ga, n_obs, purpose, estimator)
        if result is not None:
            return result
        logger.warning("%s destination fit failed; falling back to Larch", estimator)

    try:
        import larch as lx
        from larch import P, X
    except ImportError:
        logger.error("larch not installed; cannot estimate trip destination model")
        return None

    # ── Build Larch model ──────────────────────────────────────────────────
    ca = ca.set_index(["case_id", "alt_id"])

    # Sampled (ragged) choice sets *can* use Larch's sparse constructor
    # (`from_idce`, which stores only the actual (case, alt) rows) but that
    # path is UNSAFE on Larch 6.0.46: jax hands SLSQP a zero gradient, and the
    # numba engine segfaults (double free / core dump; repro in
    # _test_larch_sparse.py). Default OFF; opt in with STAGE5B_IDCE=1 once a
    # fixed Larch is installed.
    _sparse_env = os.environ.get('STAGE5B_IDCE', '0').strip() != '0'
    _sparse_mode = bool(_sparse_env and 'log_correction' in ca.columns)

    _attempt_errors: list = []
    def _try_estimate(engine: str, utility_terms: list[str]) -> Optional[dict]:
        """Try estimation with given engine and utility terms (JAX-first)."""
        try:
            # JAX-first policy: ensure CPU float64 env before build.
            try:
                import larch_jax_enforce as _jfe
                _jfe.ensure_jax_environment()
            except Exception:
                pass
            if _sparse_mode:
                d = lx.Dataset.construct.from_idce(ca, crack=False)
            else:
                d = lx.Dataset.construct.from_idca(ca, crack=False)
            # BUGFIX: Larch Model takes compute_engine at construction, NOT
            # maximize_loglike(engine=...). Old code passed engine to
            # maximize_loglike (TypeError on Larch 6 → all 4 attempts failed).
            try:
                m = lx.Model(d, compute_engine=engine)
            except Exception:
                m = lx.Model(d)
                try:
                    m.compute_engine = engine
                except Exception:
                    pass

            for var in utility_terms:
                if var in ca.columns:
                    m.utility_ca = m.utility_ca + P(var) * X(var)

            # Manski-Lerman sampling correction enters as a FIXED offset
            # (coefficient locked at 1, not estimated), so prediction over the
            # full choice set stays uncorrected. Larch rejects a bare X() term
            # in utility_ca, so bind it to a parameter then lock that parameter.
            if "log_correction" in ca.columns:
                m.utility_ca = m.utility_ca + P("c_log_correction") * X("log_correction")

            m.choice_ca_var = "chosen"

            if "log_correction" in ca.columns:
                try:
                    m.mangle()
                    m.lock_value("c_log_correction", 1.0)
                except Exception as _lce:
                    logger.warning(f"  {engine}: could not lock log_correction "
                                   f"offset ({_lce}) — continuing")

            # Ridge (pure-L2) regularisation so trip-destination coefficients are
            # shrunk/stabilised (avoids the flat/blown estimates a bare MNL gives
            # under collinear utility columns), consistent with the tour-level dest
            # model. STAGE5B_L1_RATIO>0 adds lasso selection; STAGE5B_ALPHA scales.
            _ennet = None
            try:
                import os as _os
                from larch_regularization import elasticnet_objective
                # Sparse datasets are optimized on the numba engine; the
                # elastic-net wrapper's jax branch then sees a zero/NaN jax
                # gradient and aborts the fit, so disable the penalty in
                # sparse mode (STAGE5B_ALPHA override still honoured for dense).
                _alpha = (0.0 if _sparse_mode else
                          float(_os.environ.get('STAGE5B_ALPHA', '1e-3')))
                _ennet = elasticnet_objective(
                    m, alpha=_alpha,
                    l1_ratio=float(_os.environ.get('STAGE5B_L1_RATIO', '0.0')))
                _ennet.__enter__()
            except Exception:
                _ennet = None
            try:
                # Ensure engine is set on the model (not as maximize kwarg).
                try:
                    m.compute_engine = engine
                    m.mangle()
                    m.unmangle()
                except Exception:
                    pass
                result = m.maximize_loglike(method="slsqp", quiet=True)
            finally:
                if _ennet is not None:
                    try:
                        _ennet.__exit__(None, None, None)
                    except Exception:
                        pass

            # Guard: check for NaN/Inf log-likelihood
            if not np.isfinite(result.loglike):
                logger.warning(f"  {engine}: non-finite loglike ({result.loglike})")
                return None

            # Guard against a dead gradient (silent all-zero fit): Larch's jax
            # engine hands SLSQP a zero gradient on sparse datasets (verified
            # minimal repro), which "converges" with every coefficient at 0.
            try:
                _pfv = m.pf["value"].astype(float)
                _hf = (m.pf["holdfast"].fillna(False).astype(bool)
                       if "holdfast" in m.pf.columns
                       else pd.Series(False, index=m.pf.index))
                _free = _pfv.index[~_hf]
                if len(_free) and float(np.abs(_pfv[_free]).max()) < 1e-12:
                    logger.warning(f"  {engine}: all free params at zero "
                                   "(dead gradient) — rejecting fit")
                    return None
            except Exception:
                pass

            n_params = len(m.pf)
            bic = -2 * result.loglike + n_params * np.log(n_obs)

            return {
                "purpose":        purpose,
                "params":         m.pf.copy(),
                "log_likelihood": result.loglike,
                "bic":            bic,
                "n_obs":          n_obs,
                "n_params":       n_params,
                "model":          m,
                "engine":         engine,
            }
        except Exception as exc:
            # Promoted from debug: per-attempt causes were invisible in .e
            # files (2536 silent failures in one Stage-51 shard), making the
            # terminal "ALL estimation attempts failed" undiagnosable.
            logger.warning(f"  {engine} estimation failed "
                           f"({len(utility_terms)} terms): {exc}")
            if not getattr(_try_estimate, '_tb_logged', False):
                _try_estimate._tb_logged = True
                try:
                    import traceback as _tb
                    logger.warning(
                        f"  {engine} estimation FULL TRACEBACK "
                        f"(first failure only):\n{_tb.format_exc()}")
                except Exception:
                    pass
            _attempt_errors.append(f"{engine}/{len(terms)}t:{exc}")
            return None

    # Build utility term lists for fallback chain
    backbone_terms = [v for v in BACKBONE_VARS if v in ca.columns]
    ga_terms = [v for v, active in active_ga.items() if active and v in ca.columns]
    all_terms = backbone_terms + ga_terms

    # Estimation chain. Sparse datasets (from_idce) ONLY work with the numba
    # engine — jax silently returns a zero gradient on them (minimal repro in
    # _test_larch_sparse.py). Dense keeps the JAX-first policy; numba is the
    # last-resort fallback for tiny segments (disable with LARCH_FORCE_JAX=0).
    import os as _os2
    _force_jax_5b = _os2.environ.get('LARCH_FORCE_JAX', '1').strip() != '0'

    def _engines_for(sparse_mode: bool):
        if sparse_mode:
            return [
                ("numba", all_terms),           # sparse: numba only
                ("numba", backbone_terms),
            ]
        if _force_jax_5b:
            return [
                ("jax", all_terms),             # Full model, jax (large-model default)
                ("jax", backbone_terms),        # Backbone only, jax
                ("numba", all_terms),           # Full model, numba fallback
                ("numba", backbone_terms),      # Backbone only, numba fallback
            ]
        return [
            ("numba", all_terms),
            ("jax", all_terms),
            ("numba", backbone_terms),
            ("jax", backbone_terms),
        ]

    for _sparse_try in ([True, False] if _sparse_mode else [False]):
        if _sparse_try != _sparse_mode:
            logger.warning("  sparse (idce) dataset failed; retrying with dense idca")
            _sparse_mode = _sparse_try
        for engine, terms in _engines_for(_sparse_mode):
            logger.debug(f"  Trying {engine} with {len(terms)} terms...")
            result = _try_estimate(engine, terms)
            if result is not None:
                logger.info(f"  {purpose}: converged with {engine} ({len(terms)} terms), LL={result['log_likelihood']:.1f}, BIC={result['bic']:.1f}")
                return result

    _seen: list = []
    for _e in _attempt_errors:
        if _e not in _seen:
            _seen.append(_e)
    logger.error(f"  {purpose}: ALL estimation attempts failed (jax/numba, full/backbone)"
                 f" causes=[{'; '.join(_seen[:4])}]")
    return None


def _fit_searchlibrium(ca, active_ga, n_obs, purpose, method):
    """Fit a ragged CA frame and normalize it to the Stage 5b result contract."""
    try:
        from SearchLibrium.mh_choice_estimator import ChoiceSetFrame, estimate_dest_choice
    except (ImportError, ModuleNotFoundError):
        import importlib.util
        source_root = os.environ.get(
            "SEARCHLIBRIUM_SRC", r"C:\Users\ahernz\source\SearchLibrium\src")
        source_file = os.path.join(source_root, "SearchLibrium", "mh_choice_estimator.py")
        if not os.path.isfile(source_file):
            logger.warning("SearchLibrium estimator is unavailable at %s", source_file)
            return None
        spec = importlib.util.spec_from_file_location(
            "_stage5b_searchlibrium_mh_choice_estimator", source_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ChoiceSetFrame = module.ChoiceSetFrame
        estimate_dest_choice = module.estimate_dest_choice

    feature_terms = [term for term in BACKBONE_VARS if term in ca.columns]
    feature_terms.extend(
        term for term, active in active_ga.items()
        if active and term in ca.columns and term not in feature_terms
    )
    if not feature_terms:
        logger.warning("%s destination fit has no utility features", purpose)
        return None

    frame_data = ca.copy()
    raw_values = frame_data[feature_terms].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(raw_values.to_numpy(float)).all():
        logger.warning("%s SearchLibrium CA features contain non-finite values", purpose)
        return None
    scales = raw_values.std(axis=0).to_numpy(float)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    for _index, _term in enumerate(feature_terms):
        frame_data[_term] = (raw_values[_term] / scales[_index]).astype(float)
    frame = ChoiceSetFrame.from_long(frame_data, feature_terms)

    options = {
        "mh": {
            "draws": int(os.environ.get("STAGE5B_DEST_MH_DRAWS", "1000")),
            "burn_in": int(os.environ.get("STAGE5B_DEST_MH_BURN_IN", "500")),
        },
        "gibbs": {
            "draws": int(os.environ.get("STAGE5B_DEST_GIBBS_DRAWS", "500")),
            "burn_in": int(os.environ.get("STAGE5B_DEST_GIBBS_BURN_IN", "250")),
            "set_size": int(os.environ.get("STAGE5B_DEST_GIBBS_SET_SIZE", "30")),
            "inner_steps": int(os.environ.get("STAGE5B_DEST_GIBBS_INNER_STEPS", "2")),
        },
        "sgd": {
            "iterations": int(os.environ.get("STAGE5B_DEST_SGD_ITERATIONS", "500")),
            "batch_size": int(os.environ.get("STAGE5B_DEST_SGD_BATCH_SIZE", "64")),
            "k_min": int(os.environ.get("STAGE5B_DEST_SGD_K_MIN", "8")),
            "k_max": int(os.environ.get("STAGE5B_DEST_SGD_K_MAX", "64")),
            "learning_rate": float(os.environ.get("STAGE5B_DEST_SGD_LR", "0.01")),
        },
    }
    estimate = estimate_dest_choice(
        frame, method=method, feature_cols=feature_terms,
        seed=int(os.environ.get("STAGE5B_DEST_SEED", "42")), **options[method])
    beta = np.asarray(estimate.coef, dtype=float) / scales
    std_err = np.asarray(estimate.std_err, dtype=float) / scales
    params = pd.DataFrame({
        "value": beta,
        "std_err": std_err,
        "t_stat": np.divide(
            beta, std_err, out=np.full(len(beta), np.nan), where=std_err > 0),
    }, index=feature_terms)
    loglike = float(frame.loglike(np.asarray(estimate.coef, dtype=float)))
    bic = -2.0 * loglike + len(feature_terms) * np.log(max(int(n_obs), 1))
    return {
        "purpose": purpose,
        "params": params,
        "coefficients": dict(zip(feature_terms, beta)),
        "log_likelihood": loglike,
        "bic": float(bic),
        "n_obs": int(n_obs),
        "n_params": len(feature_terms),
        "model": None,
        "engine": f"searchlibrium-{method}",
        "searchlibrium": {
            "method": method,
            "feature_names": feature_terms,
            "diagnostics": estimate.diagnostics,
            "metadata": estimate.metadata,
        },
    }


# ---------------------------------------------------------------------------
# Prediction / simulation
# ---------------------------------------------------------------------------

def _predict_stop_dest(
    stop_trips: pd.DataFrame,
    all_zones: pd.DataFrame,
    model_result: dict,
    skims: dict[str, np.ndarray],
    landuse: pd.DataFrame,
    taz_index: dict[int, int],
    zone_id_col: str = "zone_id",
    rng: np.random.Generator = None,
    batch_size: int = 20_000,
    logsums=None,
) -> pd.Series:
    """
    Predict destination TAZ for stop trips using estimated model parameters.

    Returns a pd.Series indexed like stop_trips with predicted dest_taz.

    Processes ``stop_trips`` in batches of ``batch_size`` observations: the
    observations x zones expansion below is dense, so at full-population
    scale (millions of stop trips) building it in one shot exhausts memory
    the same way full re-estimation did (job 23661073: OOM at ~261GB
    estimating n=2,600,393 home stops - the identical expansion pattern
    used here for prediction would reproduce that on the very next stage).
    Batching bounds peak memory to batch_size x n_zones regardless of the
    total population size.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Prediction only needs the fitted parameter values (below); the live
    # Larch model object is never called on. This also lets prediction run
    # against a reused/pickled model_result whose "model" key was stripped
    # before saving (see run_stage5b's reuse_models_path path).
    pf_result = model_result.get("params")
    if pf_result is None or (hasattr(pf_result, "empty") and pf_result.empty):
        logger.warning("No fitted parameters available for prediction; keeping observed destinations")
        return stop_trips["dest_taz"].copy()

    zones = all_zones[[zone_id_col]].reset_index(drop=True)
    n_obs_total = len(stop_trips)
    n_zones     = len(zones)

    # Build utility terms using fitted parameters
    pf = model_result["params"]["value"]

    def _p(name: float) -> float:
        return float(pf.get(name, 0.0))

    results: list[pd.Series] = []
    for batch_start in range(0, n_obs_total, batch_size):
        batch = stop_trips.iloc[batch_start:batch_start + batch_size]
        n_obs = len(batch)

        obs_idx = np.repeat(np.arange(n_obs), n_zones)
        alt_idx = np.tile(np.arange(n_zones), n_obs)

        obs_rows = batch.iloc[obs_idx].reset_index(drop=True)
        alt_rows = zones.iloc[alt_idx].reset_index(drop=True)

        if "NetworkDist" in skims:
            o_i = obs_rows["origin_taz"].map(taz_index).fillna(0).astype(int).values
            a_i = alt_rows[zone_id_col].map(taz_index).fillna(0).astype(int).values
            anc_i = obs_rows["anchor_down_taz"].map(taz_index).fillna(0).astype(int).values
        else:
            o_i = np.zeros(len(obs_rows), dtype=int)
            a_i = np.zeros(len(obs_rows), dtype=int)
            anc_i = np.zeros(len(obs_rows), dtype=int)

        # Use the same mode-aware trip-square mask as estimation.  In
        # particular, short active-mode work trips are bounded by their walk or
        # bike network when that skim is available.
        _metrics = _choice_set_metrics(
            obs_rows, alt_rows[zone_id_col].to_numpy(), skims, taz_index)
        d_oa = _metrics["d_oa"]
        d_aa = _metrics["d_aa"]
        d_oo = _metrics["d_oo"]
        detour = _metrics["detour"]
        feasible = _metrics["feasible"]
        feasible_matrix = feasible.reshape(n_obs, n_zones)
        if "dest_taz" in batch.columns:
            zone_to_idx = {int(zone): i for i, zone in enumerate(zones[zone_id_col])}
            for row_idx, observed in enumerate(batch["dest_taz"]):
                if pd.notna(observed) and int(observed) in zone_to_idx:
                    feasible_matrix[row_idx, zone_to_idx[int(observed)]] = True

        log_size = compute_size_term(
            alt_rows.assign(**{zone_id_col: alt_rows[zone_id_col]}),
            model_result["purpose"], landuse, zone_id_col,
        )

        # ── Build every feature the fitted model may carry, then accumulate
        # generically so GA-selected conditioning terms are honoured (the old
        # hard-coded detour+size+crash formula silently dropped every other
        # fitted parameter — including the stage-51 tour/anchor/previous-stop
        # terms).
        _alt_vals = alt_rows[zone_id_col].values
        _origin_vals = (obs_rows["origin_taz"].values if "origin_taz" in obs_rows.columns
                        else np.zeros(len(obs_rows)))
        _anchor_vals = (obs_rows["anchor_down_taz"].values if "anchor_down_taz" in obs_rows.columns
                        else np.zeros(len(obs_rows)))
        feats = {
            "detour_dist_km": detour,
            "log_size": log_size,
            "dist_to_anchor": d_aa,
            "log_dist_to_anchor": np.log1p(np.maximum(d_aa, 0.0)),
            "same_as_anchor": (_alt_vals == _anchor_vals).astype(float),
            "same_as_prev": (_alt_vals == _origin_vals).astype(float),
        }
        if "is_outbound" in obs_rows.columns:
            feats["is_inbound"] = (1 - pd.to_numeric(obs_rows["is_outbound"], errors="coerce")
                                   .fillna(1).astype(int).values).astype(float)
        else:
            feats["is_inbound"] = np.zeros(len(obs_rows))
        for _name in ("depart_time_min", "is_outbound"):
            if _name in obs_rows.columns:
                feats[_name] = pd.to_numeric(
                    obs_rows[_name], errors="coerce").fillna(0.0).to_numpy(float)
            else:
                feats[_name] = np.zeros(len(obs_rows))
        feats["logsum"] = _resolve_logsum_values(
            logsums, obs_rows, alt_rows[zone_id_col].to_numpy())
        _tm_arr = np.ones(len(obs_rows), dtype=int)
        if "tour_mode" in obs_rows.columns:
            _tm_col2 = obs_rows["tour_mode"]
            if isinstance(_tm_col2, pd.DataFrame):
                _tm_col2 = _tm_col2.iloc[:, 0]
            _tm_num = pd.to_numeric(_tm_col2, errors="coerce")
            if float(_tm_num.notna().mean()) < 0.5:
                _tm_num = (_tm_col2.astype(str).str.strip()
                           .map({"DA": 1, "SR": 2, "Walk": 3, "Bike": 4, "Transit": 5}))
            _tm_arr = _tm_num.fillna(1).astype(int).values
        feats["tour_mode_auto"] = np.isin(_tm_arr, _TOUR_MODE_AUTO_CODES).astype(float)
        feats["tour_mode_transit"] = np.isin(_tm_arr, _TOUR_MODE_TRANSIT_CODES).astype(float)
        feats["tour_mode_active"] = np.isin(_tm_arr, _TOUR_MODE_ACTIVE_CODES).astype(float)

        # skim-based candidates
        for _alias in ("NetworkDist", "AutoIVT_AM", "TransitIVT_AM", "AutoCrashRisk"):
            if _alias in skims:
                feats[_alias] = _skim_lookup(skims[_alias], o_i, a_i)
        # land-use candidates at the alternative zone
        _lu = landuse.copy()
        for _alias, _orig in LANDUSE_ALIAS.items():
            if _orig in _lu.columns and _alias not in _lu.columns:
                _lu[_alias] = _lu[_orig]
        _lu_idx = _lu.set_index(zone_id_col) if zone_id_col in _lu.columns else _lu
        for _v in ("TotalEmployment", "RetailEmp", "HealthEduRetEmp", "Shopping",
                   "EATOUT", "Entertainment", "TotalPop", "TotalHH"):
            if _v in _lu_idx.columns:
                feats[_v] = alt_rows[zone_id_col].map(_lu_idx[_v]).fillna(0.0).values
        # person covariates broadcast over alternatives
        for _v in ("income_hi", "is_female", "hh_child", "occp_group",
                   "ptype_FullTimeWorker", "ptype_UniversityStudent", "mtwp", "WFHWEEK"):
            if _v in obs_rows.columns:
                feats[_v] = pd.to_numeric(obs_rows[_v], errors="coerce").fillna(0.0).values

        V = np.zeros(len(obs_rows), dtype=float)
        for _name in pf.index:
            _arr = feats.get(str(_name))
            if _arr is not None:
                V = V + _p(str(_name)) * np.asarray(_arr, dtype=float)

        # Mask infeasible alts, retaining an observed destination as a
        # deterministic fallback when the input contradicts the envelope.
        V[~feasible_matrix.ravel()] = -np.inf

        # Softmax per observation → draw
        V_matrix = V.reshape(n_obs, n_zones)
        has_feasible = feasible_matrix.any(axis=1)
        row_max = np.max(V_matrix, axis=1, keepdims=True)
        row_max = np.where(has_feasible[:, None], row_max, 0.0)
        V_matrix = V_matrix - row_max
        exp_V = np.exp(V_matrix)
        exp_V[~feasible_matrix] = 0.0
        row_sums = exp_V.sum(axis=1, keepdims=True)
        no_feasible = row_sums[:, 0] == 0.0
        if no_feasible.any():
            logger.warning(
                "Stage 5b prediction: %d rows have no feasible destination; "
                "using a uniform zone fallback", int(no_feasible.sum()))
            exp_V[no_feasible] = 1.0
            row_sums = exp_V.sum(axis=1, keepdims=True)
        probs = exp_V / row_sums

        chosen_zone_indices = np.array([
            rng.choice(n_zones, p=probs[i])
            for i in range(n_obs)
        ])
        chosen_taz = zones.iloc[chosen_zone_indices][zone_id_col].values
        results.append(pd.Series(chosen_taz, index=batch.index, name="dest_taz_predicted"))

    return pd.concat(results) if results else pd.Series(dtype=object, name="dest_taz_predicted")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_trip_destinations(
    trips: pd.DataFrame,
    observed_col: str = "dest_taz",
    predicted_col: str = "dest_taz_predicted",
    detour_col: str = "detour_dist_km",
) -> pd.DataFrame:
    """
    Compute validation summaries:
    - Detour distance distribution (observed vs predicted)
    - Destination zone-type frequency comparison

    Returns a DataFrame with summary statistics.
    """
    rows = []

    for grp_name, grp in trips.groupby("purpose"):
        obs_stops   = grp[grp.get("is_tour_dest_leg", pd.Series(1, index=grp.index)) == 0]
        if obs_stops.empty:
            continue

        if detour_col in obs_stops.columns:
            rows.append({
                "purpose":           grp_name,
                "metric":            "mean_detour_km",
                "observed":          obs_stops[detour_col].mean(),
                "predicted":         np.nan,
            })
            rows.append({
                "purpose":           grp_name,
                "metric":            "p75_detour_km",
                "observed":          obs_stops[detour_col].quantile(0.75),
                "predicted":         np.nan,
            })

        if predicted_col in obs_stops.columns and observed_col in obs_stops.columns:
            # Share of stops that match the observed destination TAZ
            match_rate = (obs_stops[predicted_col] == obs_stops[observed_col]).mean()
            rows.append({
                "purpose":  grp_name,
                "metric":   "dest_match_rate",
                "observed": 1.0,
                "predicted": match_rate,
            })

    return pd.DataFrame(rows, columns=["purpose", "metric", "observed", "predicted"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stage5b(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    skims_path: str,
    landuse: pd.DataFrame,
    g_tripdest: Optional[list[int]] = None,
    zone_id_col: str = "zone_id",
    home_taz_col: str = "home_taz",
    predict: bool = True,
    validate: bool = True,
    save_models: bool = True,
    output_dir: str = "saved_data",
    reuse_models_path: Optional[str] = None,
    logsums=None,
) -> dict:
    """
    Run Stage 5b: trip-level intermediate-stop destination choice.

    Parameters
    ----------
    trips       : trip table from Stage 5 (must include tour_id, person_id,
                  trip_num_in_tour, is_outbound, origin_taz, dest_taz, purpose,
                  mode, depart_time_min, arrive_time_min, travel_time_min)
    persons     : person table with person-level covariates
    skims_path  : path to OMX skim file
    landuse     : zone-level land-use DataFrame
    g_tripdest  : GA bit-vector over GA_CANDIDATE_VARS; None = all included
    zone_id_col : column name for zone identifiers in landuse / zone table
    home_taz_col: column name for home TAZ in persons table
    predict     : if True, run prediction on stop trips after estimation
    validate    : if True, compute validation statistics
    save_models : if True, pickle model results to output_dir
    output_dir  : directory for saving outputs
    reuse_models_path : path to a previously-saved stage5b_models.pkl. When
                  given and the file exists, estimation is skipped entirely
                  and these fitted per-purpose parameters are used for
                  prediction instead. Needed for full-population runs: the
                  per-purpose MNL estimation builds a dense
                  observations x zones design matrix (see
                  ``_estimate_mnl_segment``), which is fine for a HHTS-scale
                  calibration sample but exhausts memory when ``trips`` is
                  the entire predicted population (millions of stop rows).
    logsums     : optional xarray-like cases-by-zones logsum surface or long
                  case_id/alt_id/logsum DataFrame to include in the CA design.

    Returns
    -------
    dict with keys:
        'trips'       : full trip table augmented with detour_dist_km,
                        is_tour_dest_leg, anchor_down_taz, dest_taz_predicted
        'models'      : dict of purpose → model result dict
        'validation'  : validation summary DataFrame (if validate=True)
        'bic_total'   : sum of segment BICs (fitness score for GA)
        'metadata'    : feature-selection and choice-set provenance
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Stage 5b: trip-level destination choice starting")

    g_tripdest = trip_destination_feature_mask(g_tripdest)
    if g_tripdest is None:
        _feature_mask = [
            int(name in LEGACY_GA_CANDIDATE_VARS)
            for name in GA_CANDIDATE_VARS
        ]
    else:
        _feature_mask = list(g_tripdest)
    _feature_metadata = {
        "candidate_vars": list(GA_CANDIDATE_VARS),
        "mask": _feature_mask,
        "selected_vars": [
            name for name, active in zip(GA_CANDIDATE_VARS, _feature_mask)
            if active
        ],
        "source": "stage51_trip_destination" if g_tripdest is not None else "legacy_default",
    }
    _choice_set_metadata = {
        "detour_feasible": True,
        "mode_aware": True,
        "active_mode_skim_candidates": ["WalkDist", "BikeDist"],
        "active_mode_skim_fallback": "NetworkDist",
        "observed_destination_repair": True,
        "anchor_field": "anchor_down_taz",
        "sample_size": int(DEST_SAMPLE_SIZE),
        "sampling_correction": (
            "manski_lerman_log_correction"
            if DEST_SAMPLE_SIZE > 0 else "none_full_choice_set"
        ),
        "max_estimation_obs": int(MAX_ESTIMATION_OBS),
    }

    # Plan-format bookkeeping: when the input is a Stage-5 plan (trip AND
    # activity rows, OTAZ/DTAZ/travtime names), work on the trip rows with
    # renamed columns internally, but hand the caller back a frame with the
    # original plan shape and dest_taz_predicted merged on.
    _plan_full = None

    # ── Guard: is_outbound is required for tour structure; derive if missing ─
    if 'is_outbound' not in trips.columns:
        if 'tour_id' in trips.columns and 'trip_num_in_tour' in trips.columns:
            max_leg = trips.groupby('tour_id')['trip_num_in_tour'].transform('max')
            trips = trips.copy()
            trips['is_outbound'] = (trips['trip_num_in_tour'] <= max_leg / 2).astype(int)
            logger.warning("Stage 5b: 'is_outbound' column missing — derived from trip_num_in_tour")
        elif 'ptour_id' in trips.columns and 'seq' in trips.columns:
            # Plan-format input from predict_stage5: seq is position in person's day.
            # Trips before the midpoint of the tour's seq range are outbound.
            # The plan interleaves activity rows with trip rows and uses
            # OTAZ/DTAZ/travtime column names — previously this branch derived
            # is_outbound but skipped both the row filter and the rename, so
            # build_trip_records() crashed with "['dest_taz'] not in index"
            # (job 23540863: Stage 5b FAILED after 27.7s, swallowed non-fatal).
            _plan_full = trips
            trips = trips.copy()
            if 'row_type' in trips.columns:
                trips = trips[trips['row_type'] == 'trip'].copy()
            trips['_plan_row_idx'] = trips.index   # survives the persons merge
            _renames = {src: dst for src, dst in {
                'OTAZ': 'origin_taz', 'DTAZ': 'dest_taz',
                'travtime': 'travel_time_min',
                'STARTIME': 'depart_time_min', 'ENDTIME': 'arrive_time_min',
            }.items() if src in trips.columns and dst not in trips.columns}
            if _renames:
                trips = trips.rename(columns=_renames)
            _missing = [c for c in ('origin_taz', 'dest_taz', 'purpose')
                        if c not in trips.columns]
            if _missing:
                logger.error(
                    f"Stage 5b: plan-format input still missing {_missing} "
                    f"after column mapping (had: {sorted(_plan_full.columns)[:20]}…) — aborting"
                )
                return {
                    'trips': _plan_full,
                    'models': {},
                    'validation': None,
                    'bic_total': 0.0,
                    'metadata': {
                        'feature_selection': _feature_metadata,
                        'choice_set_provenance': _choice_set_metadata,
                        'estimation_source': 'input_trips',
                        'prediction_source': 'input_trips',
                    },
                }
            min_seq = trips.groupby('ptour_id')['seq'].transform('min')
            max_seq = trips.groupby('ptour_id')['seq'].transform('max')
            mid_seq = (min_seq + max_seq) / 2.0
            trips['is_outbound'] = (trips['seq'] <= mid_seq).astype(int)
            trips['tour_id'] = trips['ptour_id']
            trips['trip_num_in_tour'] = trips.groupby('ptour_id').cumcount() + 1
            logger.warning("Stage 5b: 'is_outbound' derived from ptour_id/seq (plan-format input)")
        else:
            logger.error("Stage 5b: 'is_outbound' column missing and cannot be derived — aborting")
            return {
                'trips': trips,
                'models': {},
                'validation': None,
                'bic_total': 0.0,
                'metadata': {
                    'feature_selection': _feature_metadata,
                    'choice_set_provenance': _choice_set_metadata,
                    'estimation_source': 'input_trips',
                    'prediction_source': 'input_trips',
                },
            }


    # ── Merge person covariates onto trips ──────────────────────────────────
    person_cols_needed = [
        "person_id", home_taz_col,
        "is_female", "income_hi", "income_lo", "hh_child", "occp_group",
        "ptype_FullTimeWorker", "ptype_UniversityStudent",
        "mtwp", "WFHWEEK", "incp",
    ]
    person_cols_available = [c for c in person_cols_needed if c in persons.columns]
    trips_aug = trips.merge(
        persons[person_cols_available],
        on="person_id", how="left",
    )

    # ── Build trip record with is_tour_dest_leg and anchor ─────────────────
    trips_aug = build_trip_records(trips_aug, home_taz_col=home_taz_col)

    # ── Load skims ─────────────────────────────────────────────────────────
    skims = _load_skims(skims_path)

    # ── Skim validation: check for collinearity issues ──────────────────────
    if "NetworkDist" in skims:
        for alias in ["WalkDist", "BikeDist"]:
            if alias in skims:
                if np.array_equal(skims[alias], skims["NetworkDist"]):
                    logger.warning(
                        f"OMX COLLINEARITY: {alias} identical to NetworkDist — "
                        f"walk/bike distance skim carries same auto-network data. "
                        f"Model will drop {alias} to avoid perfect collinearity. "
                        f"Fix: supply true pedestrian/cycling distance skim in OMX."
                    )

    # ── Build TAZ index ────────────────────────────────────────────────────
    if zone_id_col in landuse.columns:
        all_zones = landuse[[zone_id_col]].drop_duplicates().reset_index(drop=True)
    else:
        # Fall back: derive from all TAZ codes seen in trips
        all_taz = pd.Series(
            pd.concat([trips_aug["origin_taz"], trips_aug["dest_taz"]]).unique()
        ).sort_values().reset_index(drop=True)
        all_zones = pd.DataFrame({zone_id_col: all_taz})

    taz_index: dict[int, int] = {
        int(z): i for i, z in enumerate(all_zones[zone_id_col])
    }

    # ── Compute detour distances for all trips ─────────────────────────────
    if "NetworkDist" in skims:
        trips_aug["detour_dist_km"] = compute_detour(trips_aug, skims["NetworkDist"], taz_index)
    else:
        trips_aug["detour_dist_km"] = 0.0

    # ── Estimation: one MNL per purpose, on stop trips only ────────────────
    stop_mask = trips_aug["is_tour_dest_leg"] == 0
    # Return-home legs are not intermediate stops: their destination is the
    # home zone by construction, so they must never enter stop-destination
    # estimation or prediction. In plan-format input they carry
    # purpose == 'home' and dominate the trip table (2.6M rows in job
    # 23661073, whose estimation attempt on them was OOM-killed at 250 GB).
    if "purpose" in trips_aug.columns:
        stop_mask &= trips_aug["purpose"].astype(str).str.lower() != "home"
    stop_trips = trips_aug[stop_mask].copy()

    models: dict[str, dict] = {}
    purposes = stop_trips["purpose"].dropna().unique().tolist()

    if reuse_models_path and os.path.isfile(reuse_models_path):
        import pickle
        logger.info(
            "Stage 5b: reusing fitted models from %s — skipping re-estimation "
            "(predict-only; stop_trips n=%d)",
            reuse_models_path, len(stop_trips),
        )
        with open(reuse_models_path, "rb") as fh:
            models = pickle.load(fh)
        # Older pickles may lack diagnostic keys (job 23726229 died here on a
        # KeyError('log_likelihood') raised by this *logging* loop, killing the
        # whole predict-only path). Log defensively and drop entries that lack
        # what prediction actually needs.
        _unusable = [p for p, r in models.items()
                     if not isinstance(r, dict) or "params" not in r]
        for p in _unusable:
            logger.warning("Stage 5b: reused model '%s' lacks 'params' — dropped", p)
            models.pop(p, None)
        for purpose, result in models.items():
            result.setdefault("purpose", purpose)   # _predict_stop_dest needs it
            result.setdefault("feature_selection", _feature_metadata)
            result.setdefault("choice_set_provenance", _choice_set_metadata)
            _saved_mask = (result.get("feature_selection") or {}).get("mask")
            if _saved_mask is not None and list(_saved_mask) != _feature_mask:
                raise ValueError(
                    f"Stage 5b reused model '{purpose}' was estimated with a "
                    "different destination feature mask"
                )
            logger.info(
                "  %s: LL=%.1f  BIC=%.1f  n=%s (reused)",
                purpose,
                float(result.get("log_likelihood", float("nan"))),
                float(result.get("bic", float("nan"))),
                result.get("n_obs", "?"),
            )
    else:
        for purpose in purposes:
            seg = stop_trips[stop_trips["purpose"] == purpose].copy()
            if len(seg) > MAX_ESTIMATION_OBS:
                logger.info(
                    "Stage 5b: estimating %s stop destination (n=%d, sampled down "
                    "from %d — idca expansion is dense at n x n_zones)",
                    purpose, MAX_ESTIMATION_OBS, len(seg),
                )
                seg = seg.sample(n=MAX_ESTIMATION_OBS, random_state=42)
            else:
                logger.info(
                    "Stage 5b: estimating %s stop destination (n=%d)", purpose, len(seg)
                )
            result = _estimate_mnl_segment(
                stop_trips   = seg,
                all_zones    = all_zones,
                skims        = skims,
                landuse      = landuse,
                purpose      = purpose,
                taz_index    = taz_index,
                g_tripdest   = g_tripdest,
                zone_id_col  = zone_id_col,
                persons      = persons,
                logsums      = logsums,
            )
            if result is not None:
                result["feature_selection"] = _feature_metadata
                result["choice_set_provenance"] = _choice_set_metadata
                models[purpose] = result
                logger.info(
                    "  %s: LL=%.1f  BIC=%.1f  n=%d",
                    purpose, result["log_likelihood"], result["bic"], result["n_obs"],
                )

    # ── Prediction ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    if predict and models:
        predicted_parts = []
        for purpose, result in models.items():
            seg = stop_trips[stop_trips["purpose"] == purpose].copy()
            pred = _predict_stop_dest(
                stop_trips   = seg,
                all_zones    = all_zones,
                model_result = result,
                skims        = skims,
                landuse      = landuse,
                taz_index    = taz_index,
                zone_id_col  = zone_id_col,
                rng          = rng,
                logsums      = logsums,
            )
            predicted_parts.append(pred)

        if predicted_parts:
            all_preds = pd.concat(predicted_parts)
            trips_aug["dest_taz_predicted"] = trips_aug.index.map(all_preds)
            # For primary-dest legs: predicted = observed (already assigned in Stage 5)
            trips_aug.loc[trips_aug["is_tour_dest_leg"] == 1, "dest_taz_predicted"] = (
                trips_aug.loc[trips_aug["is_tour_dest_leg"] == 1, "dest_taz"]
            )

    # ── Validation ─────────────────────────────────────────────────────────
    validation_df = pd.DataFrame()
    if validate and "dest_taz_predicted" in trips_aug.columns:
        validation_df = validate_trip_destinations(trips_aug)
        val_path = os.path.join(output_dir, "stage5b_validation.csv")
        validation_df.to_csv(val_path, index=False)
        logger.info("Stage 5b validation written to %s", val_path)

    # ── BIC total (GA fitness) ─────────────────────────────────────────────
    bic_total = sum(r["bic"] for r in models.values()) if models else np.inf

    # ── Save model results ─────────────────────────────────────────────────
    if save_models and models:
        import pickle
        model_path = os.path.join(output_dir, "stage5b_models.pkl")
        # Store without larch Model object (not always picklable)
        saveable = {
            p: {k: v for k, v in r.items() if k != "model"}
            for p, r in models.items()
        }
        with open(model_path, "wb") as fh:
            pickle.dump(saveable, fh)
        logger.info("Stage 5b models saved to %s", model_path)

    # ── Save trip table ────────────────────────────────────────────────────
    trip_path = os.path.join(output_dir, "stage5b_trips.csv")
    trips_aug.to_csv(trip_path, index=False)
    logger.info("Stage 5b trip table written to %s", trip_path)

    # ── Plan-format input: return the caller's plan shape, not the internal
    # trips-only working table. Downstream (Stage 6) expects the full plan
    # with activity rows intact; only dest_taz_predicted is added here.
    trips_out = trips_aug
    if _plan_full is not None:
        trips_out = _plan_full.copy()
        if "dest_taz_predicted" in trips_aug.columns and "_plan_row_idx" in trips_aug.columns:
            _pred_by_row = (trips_aug.dropna(subset=["dest_taz_predicted"])
                            .set_index("_plan_row_idx")["dest_taz_predicted"])
            trips_out["dest_taz_predicted"] = trips_out.index.map(_pred_by_row)
            logger.info(
                "Stage 5b: %d stop-trip destination predictions merged back "
                "onto plan (%d rows)", int(_pred_by_row.notna().sum()), len(trips_out)
            )

    return {
        "trips":      trips_out,
        "models":     models,
        "validation": validation_df,
        "bic_total":  bic_total,
        "metadata": {
            "feature_selection": _feature_metadata,
            "choice_set_provenance": _choice_set_metadata,
            "estimation_source": "input_trips",
            "prediction_source": "input_trips",
            "logsum_source": "argument" if logsums is not None else "none_or_trip_column",
            "destination_estimator": os.environ.get("STAGE5B_DEST_ESTIMATOR", "larch"),
            "models_reused": bool(reuse_models_path and os.path.isfile(reuse_models_path)),
        },
    }


# ---------------------------------------------------------------------------
# Trip-level mode choice estimation (per purpose)
# ---------------------------------------------------------------------------

def estimate_trip_mode_models(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    output_dir: str = "saved_data",
    save_models: bool = True,
) -> dict:
    """
    Estimate trip-level nested-logit mode choice models, one per purpose.

    This mirrors Stage 6 tour-level mode estimation but runs on individual
    trips rather than tours.  Used by the Stage 5b pipeline to produce
    separate trip-purpose destination and mode coefficient reports.

    Parameters
    ----------
    trips      : trip table (must contain a mode column and purpose column)
    persons    : person table with person-level covariates
    landuse    : zone-level land-use DataFrame
    output_dir : where to save pkl and HTML reports
    save_models: if True, write stage5b_trip_mode_models.pkl + HTML

    Returns
    -------
    dict  {purpose: NestedLogitModeModel}
    """
    from stage6_mode_choice import estimate_mode_models, _save_s6_coefficients

    logger.info("Stage 5b: estimating trip-level mode models per purpose")
    os.makedirs(output_dir, exist_ok=True)

    mode_models = {}
    try:
        mode_models = estimate_mode_models(trips, persons, landuse)
        logger.info(
            "Stage 5b trip mode models estimated for: %s",
            list(mode_models.keys()),
        )
    except Exception as exc:
        logger.warning("Stage 5b trip mode estimation failed: %s", exc)
        return mode_models

    if save_models and mode_models:
        pkl_path  = os.path.join(output_dir, "stage5b_trip_mode_models.pkl")
        coef_path = os.path.join(output_dir, "stage5b_trip_mode_models_coefficients.csv")
        try:
            with open(pkl_path, "wb") as fh:
                pickle.dump(mode_models, fh)
            logger.info("Stage 5b trip mode models saved → %s", pkl_path)
        except Exception as exc:
            logger.warning("Stage 5b trip mode pkl save failed: %s", exc)
        try:
            _save_s6_coefficients(mode_models, coef_path)
            logger.info("Stage 5b trip mode coefficients saved → %s", coef_path)
        except Exception as exc:
            logger.warning("Stage 5b trip mode coefficient export failed: %s", exc)

    return mode_models


# ---------------------------------------------------------------------------
# Trip-level transit access choice (access + submode constrained)
# ---------------------------------------------------------------------------
# Gated by STAGE5B_TRANSIT_ACCESS=1 (default 0: none of this runs, no new
# columns, byte-identical runs). With the flag on, Transit trip legs get:
#   1. an access label from HHTS MODE1 (Walk/Drive/Ride/Bike) for estimation,
#   2. hard availability constraints (drive needs licence+vehicle+PnR supply;
#      walk needs stop proximity; bike needs ownership; ride needs a driver),
#   3. an estimated access MNL (Larch; heuristic-share fallback), and
#   4. predicted access + submode columns at prediction time, with drive-access
#      trips flagged pnr_trip=1 for the stage-7 park-and-ride link.
# Tour level already predicts access (_predict_transit_access in stage5);
# this is the trip-leg counterpart using observed MODE1 access labels.

#: trip-level access codes (superset of the tour TRANSIT_ACCESS_MAP, + Ride)
TRIP_TRANSIT_ACCESS = {'Walk': 1, 'Drive': 2, 'Bike': 3, 'Ride': 4}
TRIP_TRANSIT_ACCESS_INV = {v: k for k, v in TRIP_TRANSIT_ACCESS.items()}

#: raw MODE1 strings that imply each access mode (pre-MODES-mapping values)
_TRIP_ACCESS_RAW = {
    'Walk': {'Walking', 'Mobility scooter'},
    'Drive': {'Car driver', 'Truck driver', 'Motorcycle driver'},
    'Ride': {'Car passenger', 'Taxi', 'Uber / Other Ride Share',
             'Truck passenger', 'Motorcycle passenger'},
    'Bike': {'Bicycle', 'E-bike or E-scooter'},
}
_TRIP_ACCESS_WALK_KM = 6.0   # walk-access feasibility cap (mirrors tour aval)


def _trip_transit_access_enabled() -> bool:
    return os.environ.get('STAGE5B_TRANSIT_ACCESS', '0').strip() == '1'


def trip_transit_access_labels(trips: pd.DataFrame) -> pd.Series:
    """Normalise trip-level transit access labels to TRIP_TRANSIT_ACCESS codes.

    Prefers the data_connect `transit_access` column (raw MODE1 mapping, built
    with STAGE5B_TRANSIT_ACCESS=1); falls back to deriving from raw MODE1
    strings when present; 0 = non-transit leg or unknown access.
    """
    out = pd.Series(0, index=trips.index, dtype=int)
    try:
        if 'transit_access' in trips.columns:
            _m = {'Walk': 1, 'Drive': 2, 'Bike': 3, 'Ride': 4}
            out = trips['transit_access'].map(_m).fillna(0).astype(int)
            if int((out > 0).sum()) > 0:
                return out
        if 'MODE1' in trips.columns:
            _raw = trips['MODE1'].astype(str)
            for _name, _vals in _TRIP_ACCESS_RAW.items():
                _mask = _raw.isin(_vals)
                if _mask.any():
                    out.loc[_mask] = TRIP_TRANSIT_ACCESS[_name]
    except Exception as exc:
        logger.debug(f"trip_transit_access_labels skipped ({exc})")
    return out.fillna(0).astype(int)


def trip_transit_availability(
    trip_rows: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    skim_lookup: dict = None,
) -> pd.DataFrame:
    """Hard availability (0/1) per trip leg × access mode.

    Drive: licence + household vehicle + PnR supply (landuse parking_capacity
        in the origin zone when available; without the column, drive stays
        available but a one-time warning is logged).
    Walk: DISTWALK short (< 6 km) when a skim lookup resolves, else allowed.
    Bike: micromobility ownership when observed, else allowed.
    Ride: multi-person household (needs a driver) when observed, else allowed.
    Columns: allow_Walk/allow_Drive/allow_Bike/allow_Ride, index = trip_rows.
    Never raises; unknown inputs degrade to allowed (soft), never to blocked.
    """
    n = len(trip_rows)
    allow = pd.DataFrame(
        {'allow_Walk': np.ones(n, dtype=int),
         'allow_Drive': np.ones(n, dtype=int),
         'allow_Bike': np.ones(n, dtype=int),
         'allow_Ride': np.ones(n, dtype=int)}, index=trip_rows.index)
    if n == 0:
        return allow
    try:
        pp = persons.copy()
        if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
            pp = pp.rename(columns={'PERSID': 'person_id'})
        _pid = next((c for c in ('person_id', 'PERSID')
                     if c in trip_rows.columns), None)
        _pidx = None
        if _pid is not None and 'person_id' in pp.columns:
            try:
                _pidx = pp.drop_duplicates('person_id').set_index('person_id')
            except Exception:
                _pidx = None

        def _pcol(col, default=np.nan):
            if _pidx is None or _pid is None or col not in _pidx.columns:
                return pd.Series(default, index=trip_rows.index)
            try:
                return pd.to_numeric(
                    trip_rows[_pid].astype(str).map(_pidx[col]),
                    errors='coerce')
            except Exception:
                return pd.Series(default, index=trip_rows.index)

        lic = _pcol('CARLICENCE', 1.0).fillna(1.0)
        veh = _pcol('vehrd', 1.0).fillna(1.0)
        nprd = _pcol('nprd', 2.0).fillna(2.0)
        # Drive: licence + vehicle + PnR supply at origin
        _drive_ok = ((lic > 0) & (veh > 0)).astype(int)
        try:
            _zcol = next((c for c in ('ORDER', 'DTAZ', 'SA2_MAIN16', 'zone_id')
                          if c in landuse.columns), None)
            _ocol = next((c for c in ('origin_taz', 'OTAZ')
                          if c in trip_rows.columns), None)
            if (_zcol is not None and _ocol is not None
                    and 'parking_capacity' in landuse.columns):
                _cap = pd.to_numeric(
                    landuse.set_index(_zcol)['parking_capacity'],
                    errors='coerce').fillna(0.0)
                _oz = pd.to_numeric(trip_rows[_ocol], errors='coerce')
                _has_pnr = _oz.map(_cap).fillna(0.0) > 0
                _drive_ok = (_drive_ok & _has_pnr.fillna(True)).astype(int)
            else:
                logger.warning(
                    "trip_transit_availability: no parking_capacity landuse "
                    "column — drive access unconstrained by PnR supply")
        except Exception as _pnr_e:
            logger.debug(f"trip_transit_availability PnR check skipped ({_pnr_e})")
        allow['allow_Drive'] = _drive_ok.values.astype(int)
        # Walk: short access distance when a skim resolves
        try:
            _o2 = next((c for c in ('origin_taz', 'OTAZ')
                        if c in trip_rows.columns), None)
            _d2 = next((c for c in ('dest_taz', 'DTAZ')
                        if c in trip_rows.columns), None)
            if skim_lookup and _o2 is not None and _d2 is not None:
                _wd = []
                for _o, _d in zip(
                        pd.to_numeric(trip_rows[_o2], errors='coerce'),
                        pd.to_numeric(trip_rows[_d2], errors='coerce')):
                    try:
                        _sk = skim_lookup.get((int(_o), int(_d)), {})
                        _wd.append(float(_sk.get('DISTWALK', 0.0) or 0.0))
                    except Exception:
                        _wd.append(0.0)
                _wd = np.asarray(_wd, dtype=float)
                _known = _wd > 0
                allow.loc[_known, 'allow_Walk'] = (
                    _wd[_known] < _TRIP_ACCESS_WALK_KM).astype(int).values
        except Exception as _w_e:
            logger.debug(f"trip_transit_availability walk check skipped ({_w_e})")
        # Bike: ownership-gated when observed
        try:
            _owned = None
            for _bc in ('bikes', 'ebikes', 'escooter'):
                _bv = _pcol(_bc, np.nan)
                if _bv.notna().any():
                    _owned = (_bv.fillna(0) > 0) if _owned is None \
                        else (_owned | (_bv.fillna(0) > 0))
            if _owned is not None:
                allow['allow_Bike'] = _owned.astype(int).values
        except Exception as _b_e:
            logger.debug(f"trip_transit_availability bike check skipped ({_b_e})")
        # Ride: needs a potential driver in the household
        try:
            if nprd.notna().any():
                allow['allow_Ride'] = (nprd.fillna(2.0) > 1).astype(int).values
        except Exception as _r_e:
            logger.debug(f"trip_transit_availability ride check skipped ({_r_e})")
    except Exception as exc:
        logger.debug(f"trip_transit_availability failed open ({exc})")
    return allow


def estimate_trip_transit_access(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    skim_lookup: dict = None,
    output_dir: str = "saved_data",
) -> dict:
    """Estimate a trip-leg transit access MNL (Walk/Drive/Bike/Ride).

    Uses HHTS MODE1-derived labels; unavailable alternatives are masked.
    Returns {'coefs': {alt: {var: val}}, 'features': [...], 'n_obs': int,
    'method': 'larch_mnl'|'shares'} and saves pkl + CSV (best-effort).
    Falls back to observed access shares (heuristic) when Larch is missing
    or the choice is degenerate — never raises.
    """
    out = {"coefs": {}, "features": [], "n_obs": 0, "method": "shares",
           "shares": {}}
    try:
        _lab = trip_transit_access_labels(trips)
        # access choice is conditional on the leg being a Transit trip —
        # drop labelled non-transit legs (their MODE1 is not access).
        try:
            _mc = next((c for c in ('mode', 'chosen_mode', 'MAINMODE')
                        if c in trips.columns), None)
            if _mc is not None:
                _is_tr = trips[_mc].astype(str).isin(
                    ['Transit', '5', 'transit'])
                _lab = _lab.where(_is_tr, 0)
        except Exception:
            pass
        _tt = trips[_lab.isin(list(TRIP_TRANSIT_ACCESS.values()))].copy()
        _tt['_access'] = _lab[_lab.isin(list(TRIP_TRANSIT_ACCESS.values()))].values
        out["n_obs"] = int(len(_tt))
        _shares = (_tt['_access'].value_counts(normalize=True).to_dict()
                   if len(_tt) else {})
        out["shares"] = {int(k): float(v) for k, v in _shares.items()}
        if len(_tt) < 30 or _tt['_access'].nunique() < 2:
            logger.warning(
                "estimate_trip_transit_access: only %d labelled legs "
                "(%d alts) — heuristic shares", len(_tt),
                _tt['_access'].nunique() if len(_tt) else 0)
            return out
        # skim-fed time features per leg
        try:
            _o = next((c for c in ('origin_taz', 'OTAZ') if c in _tt.columns), None)
            _d = next((c for c in ('dest_taz', 'DTAZ') if c in _tt.columns), None)
            _wt, _dt, _ivt = [], [], []
            for _ro, _rd in zip(
                    pd.to_numeric(_tt[_o], errors='coerce') if _o else [np.nan] * len(_tt),
                    pd.to_numeric(_tt[_d], errors='coerce') if _d else [np.nan] * len(_tt)):
                try:
                    _sk = (skim_lookup or {}).get((int(_ro), int(_rd)), {})
                    _wt.append(float(_sk.get('DISTWALK', 1.0) or 1.0) * 12.0)
                    _dt.append(float(_sk.get('SOVTOLL_TIME__AM', 30.0) or 30.0))
                    _ivt.append(float(_sk.get('WLK_LOC_WLK_TOTIVT__AM', 45.0) or 45.0))
                except Exception:
                    _wt.append(12.0)
                    _dt.append(30.0)
                    _ivt.append(45.0)
            _tt['_walk_min'] = np.asarray(_wt, dtype=float)
            _tt['_drive_min'] = np.asarray(_dt, dtype=float)
            _tt['_ivt_min'] = np.asarray(_ivt, dtype=float)
        except Exception:
            _tt['_walk_min'] = 12.0
            _tt['_drive_min'] = 30.0
            _tt['_ivt_min'] = 45.0
        _pp = persons.copy()
        if 'person_id' not in _pp.columns and 'PERSID' in _pp.columns:
            _pp = _pp.rename(columns={'PERSID': 'person_id'})
        _pid = next((c for c in ('person_id', 'PERSID') if c in _tt.columns), None)
        try:
            _pidx = _pp.drop_duplicates('person_id').set_index('person_id')
            _tt['_veh'] = pd.to_numeric(
                _tt[_pid].astype(str).map(_pidx.get('vehrd', pd.Series(dtype=float))),
                errors='coerce').fillna(1.0).values if _pid else 1.0
            _tt['_incp'] = pd.to_numeric(
                _tt[_pid].astype(str).map(_pidx.get('incp', pd.Series(dtype=float))),
                errors='coerce').fillna(8.0).values if _pid else 8.0
        except Exception:
            _tt['_veh'] = 1.0
            _tt['_incp'] = 8.0
        # availability mask (hard)
        try:
            _av = trip_transit_availability(_tt, persons, landuse, skim_lookup)
        except Exception:
            _av = None
        # Larch MNL over access alts present in the data
        try:
            import larch as lx
            from larch import P as _P, X as _X
            _alts = sorted(int(v) for v in _tt['_access'].unique())
            _ca_rows = []
            for _i, _row in _tt.iterrows():
                for _a in _alts:
                    _ca_rows.append({
                        'case_id': _i, 'alt_id': _a,
                        'chosen': int(_row['_access'] == _a),
                        'walk_min': float(_row['_walk_min']),
                        'drive_min': float(_row['_drive_min']),
                        'ivt_min': float(_row['_ivt_min']),
                        'veh': float(_row['_veh']),
                        'incp': float(_row['_incp']),
                        'is_walk': int(_a == 1), 'is_drive': int(_a == 2),
                        'is_bike': int(_a == 3), 'is_ride': int(_a == 4),
                    })
            _ca = pd.DataFrame(_ca_rows).set_index(['case_id', 'alt_id'])
            # hard availability: drop unavailable non-chosen alts BEFORE the
            # model is built (keeps every chosen row so the likelihood stays
            # well-defined). Falls back to the full CA on any failure.
            if _av is not None:
                try:
                    _keep = pd.Series(True, index=_ca.index)
                    for _a, _col in ((1, 'allow_Walk'), (2, 'allow_Drive'),
                                     (3, 'allow_Bike'), (4, 'allow_Ride')):
                        if _a in _alts and _col in _av.columns:
                            _ok = _av[_col].reindex(
                                _ca.index.get_level_values(0)).fillna(1).values
                            _rows_a = (_ca.index.get_level_values(1)
                                       == int(_a)).values
                            _keep = _keep & ~(_rows_a & (_ok == 0))
                    _chosen = _ca['chosen'].values.astype(bool)
                    _keep = (_keep | _chosen)
                    if float(np.asarray(_keep).mean()) < 1.0:
                        logger.info(
                            "estimate_trip_transit_access: availability drops "
                            f"{int((~np.asarray(_keep)).sum())}/{len(_ca)} CA rows")
                    _ca = _ca[np.asarray(_keep)]
                except Exception as _av_e:
                    logger.debug(f"access availability mask skipped ({_av_e})")
            _m = lx.Model(lx.Dataset.construct.from_idca(_ca, crack=False))
            _m.utility_ca = (
                _P('asc_drive') * _X('is_drive') + _P('asc_bike') * _X('is_bike')
                + _P('asc_ride') * _X('is_ride')
                + _P('c_walk_time') * _X('walk_min') * _X('is_walk')
                + _P('c_drive_time') * _X('drive_min') * _X('is_drive')
                + _P('c_ivt') * _X('ivt_min')
                + _P('c_veh_drive') * _X('veh') * _X('is_drive')
                + _P('c_incp_drive') * _X('incp') * _X('is_drive'))
            _m.choice_ca_var = 'chosen'
            _m.maximize_loglike(method='slsqp')
            _pf = _m.pf
            for _a in _alts:
                _nm = TRIP_TRANSIT_ACCESS_INV.get(_a, str(_a))
                out["coefs"][_nm] = {
                    str(_p): float(_pf.loc[_p, 'value'])
                    for _p in _pf.index if 'value' in _pf.columns}
            out["features"] = ['walk_min', 'drive_min', 'ivt_min',
                               'veh', 'incp']
            out["method"] = "larch_mnl"
            logger.info(f"estimate_trip_transit_access: Larch MNL fitted "
                        f"(n={len(_tt)}, alts={_alts})")
        except Exception as _lx_e:
            logger.warning(f"estimate_trip_transit_access: Larch fit failed "
                           f"({_lx_e}) — heuristic shares")
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "stage5b_transit_access.pkl"),
                      "wb") as _fh:
                pickle.dump(out, _fh)
            pd.DataFrame(
                [{"alt": _a, "share": out["shares"].get(_a, 0.0)}
                 for _a in sorted(TRIP_TRANSIT_ACCESS.values())]
            ).to_csv(os.path.join(output_dir, "stage5b_transit_access.csv"),
                     index=False)
        except Exception as _sv_e:
            logger.debug(f"transit-access artifact save skipped ({_sv_e})")
    except Exception as exc:
        logger.warning(f"estimate_trip_transit_access failed ({exc})")
    return out


def predict_trip_transit_access(
    trip_rows: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    access_model: dict,
    skim_lookup: dict = None,
    seed: int = 42,
    sample: bool = True,
) -> pd.DataFrame:
    """Assign access (+PnR marker) to Transit trip legs.

    Constrains by trip_transit_availability, then scores with the estimated
    MNL (or heuristic shares). Attaches `transit_access` (codes),
    `transit_access_name`, and `pnr_trip` (1 = drive access → park-and-ride
    demand for the stage-7 link). Non-transit rows get 0/''/0. Never raises.
    """
    out = trip_rows.copy()
    out['transit_access'] = 0
    out['transit_access_name'] = ''
    out['pnr_trip'] = 0
    try:
        _mode_col = next((c for c in ('mode', 'chosen_mode', 'MAINMODE')
                          if c in out.columns), None)
        if _mode_col is None:
            return out
        _is_tr = out[_mode_col].astype(str).isin(
            ['Transit', '5', 'transit'])
        if not _is_tr.any():
            return out
        _tr = out.loc[_is_tr].copy()
        _av = trip_transit_availability(_tr, persons, landuse, skim_lookup)
        _alts = [1, 2, 3, 4]
        _names = {1: 'Walk', 2: 'Drive', 3: 'Bike', 4: 'Ride'}
        # utility from estimated coefs (alt-specific constants + time terms)
        _coefs = (access_model or {}).get('coefs', {})
        _V = np.zeros((len(_tr), len(_alts)), dtype=float)
        try:
            _o = next((c for c in ('origin_taz', 'OTAZ') if c in _tr.columns),
                      None)
            _d = next((c for c in ('dest_taz', 'DTAZ') if c in _tr.columns),
                      None)
            _wt = np.full(len(_tr), 12.0)
            _dt = np.full(len(_tr), 30.0)
            _ivt = np.full(len(_tr), 45.0)
            if skim_lookup and _o is not None and _d is not None:
                for _i, (_ro, _rd) in enumerate(zip(
                        pd.to_numeric(_tr[_o], errors='coerce'),
                        pd.to_numeric(_tr[_d], errors='coerce'))):
                    try:
                        _sk = skim_lookup.get((int(_ro), int(_rd)), {})
                        _wt[_i] = float(_sk.get('DISTWALK', 1.0) or 1.0) * 12.0
                        _dt[_i] = float(_sk.get('SOVTOLL_TIME__AM', 30.0) or 30.0)
                        _ivt[_i] = float(_sk.get('WLK_LOC_WLK_TOTIVT__AM', 45.0)
                                         or 45.0)
                    except Exception:
                        pass
        except Exception:
            _wt = np.full(len(_tr), 12.0)
            _dt = np.full(len(_tr), 30.0)
            _ivt = np.full(len(_tr), 45.0)
        for _j, _a in enumerate(_alts):
            _cc = _coefs.get(_names[_a], {})
            _g = (lambda n, d=0.0: float(_cc.get(n, d))
                  if _cc else d)
            if _a == 1:
                _V[:, _j] = (_g('asc_walk', 0.0) + _g('c_walk_time', -0.1) * _wt)
            elif _a == 2:
                _V[:, _j] = (_g('asc_drive', 0.0) + _g('c_drive_time', -0.1) * _dt
                             + _g('c_veh_drive', 0.0) * 1.0
                             + _g('c_incp_drive', 0.0) * 8.0)
            elif _a == 3:
                _V[:, _j] = _g('asc_bike', 0.0) + _g('c_bike_time', -0.1) * _wt
            else:
                _V[:, _j] = _g('asc_ride', 0.0) + _g('c_ivt', -0.05) * _ivt
        # hard availability: -inf unavailable (softmax-safe sentinel)
        for _j, _a in enumerate(_alts):
            _col = {1: 'allow_Walk', 2: 'allow_Drive',
                    3: 'allow_Bike', 4: 'allow_Ride'}[_a]
            if _col in _av.columns:
                _ok = _av[_col].reindex(_tr.index).fillna(1).values.astype(bool)
                _V[~_ok, _j] = -1e20
        _V = _V - _V.max(axis=1, keepdims=True)
        _P = np.exp(np.clip(_V, -500, 0))
        _P = _P / _P.sum(axis=1, keepdims=True)
        _rng = np.random.default_rng(seed)
        if sample:
            _draw = np.array([_rng.choice(len(_alts), p=_P[i] / _P[i].sum())
                              for i in range(len(_tr))])
        else:
            _draw = _P.argmax(axis=1)
        _chosen = np.array(_alts)[_draw]
        out.loc[_is_tr, 'transit_access'] = _chosen
        out.loc[_is_tr, 'transit_access_name'] = [
            _names[int(v)] for v in _chosen]
        out.loc[_is_tr, 'pnr_trip'] = (_chosen == 2).astype(int)
    except Exception as exc:
        logger.debug(f"predict_trip_transit_access skipped ({exc})")
    return out


# ---------------------------------------------------------------------------
# Stage 5b HTML coefficient report (destination + mode)
# ---------------------------------------------------------------------------

def _write_stage5b_html(
    dest_models: dict,
    mode_models: dict,
    output_dir: str,
) -> None:
    """
    Write HTML coefficient report for Stage 5b trip destination and mode models.

    Produces:
      stage5b_trip_dest_models_coefficients.html   — trip stop destination MNL
      stage5b_trip_mode_models_coefficients.html   — trip-level mode NL (if available)
      stage5b_trip_models_coefficients.html        — combined report
    """
    import html as _h

    def _sig_stars(t):
        try:
            at = abs(float(t))
            return '★★★' if at > 3.3 else '★★' if at > 2.6 else '★' if at > 1.96 else ''
        except Exception:
            return ''

    def _sig_color(t):
        try:
            at = abs(float(t))
            return '#1a7a1a' if at > 3.3 else '#2e7d32' if at > 2.6 \
                else '#558b2f' if at > 1.96 else '#b71c1c'
        except Exception:
            return '#888'

    CSS = """
<style>
body{font-family:Arial,sans-serif;margin:30px;max-width:1300px}
h1{background:#2c3e50;color:#fff;padding:12px 18px;border-radius:4px}
h2{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:4px;margin-top:28px}
h3{color:#444;margin-top:16px}
table{border-collapse:collapse;width:100%;margin:8px 0 18px}
th{background:#2c3e50;color:#fff;padding:6px 12px;text-align:left}
td{border:1px solid #ddd;padding:5px 12px;font-size:13px}
tr:nth-child(even){background:#f9f9f9}
.stats{color:#555;font-size:13px;margin:4px 0 10px}
.sig-note{color:#666;font-size:11px;margin-top:6px}
</style>"""

    def _dest_section(purpose: str, result: dict) -> str:
        coef = result.get("coefficients", {})
        ll   = result.get("log_likelihood", float("nan"))
        bic  = result.get("bic", float("nan"))
        n    = result.get("n_obs", "?")
        rows = "\n".join(
            f'<tr><td>{_h.escape(str(p))}</td>'
            f'<td style="text-align:right">{float(v):.4f}</td>'
            f'<td></td><td></td><td></td></tr>'
            for p, v in coef.items()
        )
        return (
            f'<h3>{_h.escape(purpose.title())} — stop destination MNL'
            f'  <span class="stats">LL={ll:.1f}  BIC={bic:.1f}  n={n}</span></h3>'
            '<table><tr><th>Parameter</th><th>Value</th>'
            '<th>Std Err</th><th>t-stat</th><th>Sig</th></tr>'
            f'{rows}</table>'
        )

    def _mode_section(purpose: str, model) -> str:
        coef  = model.coef_
        se    = model.se_    if model.se_    is not None else np.full(len(coef), np.nan)
        tstat = model.tstat_ if model.tstat_ is not None else np.full(len(coef), np.nan)
        ll    = model.ll_final_ if model.ll_final_ is not None else float("nan")
        rho2  = model.rho2_     if model.rho2_     is not None else float("nan")
        n     = model.n_obs_
        rows = "\n".join(
            f'<tr><td>{_h.escape(str(p))}</td>'
            f'<td style="text-align:right">{float(c):.4f}</td>'
            f'<td style="text-align:right">{float(s):.4f if np.isfinite(float(s)) else "—"}</td>'
            f'<td style="text-align:right; color:{_sig_color(t)}">'
            f'{float(t):.2f if np.isfinite(float(t)) else "—"} {_sig_stars(t)}</td>'
            f'<td>{_sig_stars(t)}</td></tr>'
            for p, c, s, t in zip(model.PARAM_NAMES, coef, se, tstat)
        )
        return (
            f'<h3>{_h.escape(purpose.title())} — trip mode NL'
            f'  <span class="stats">LL={ll:.1f}  ρ²={rho2:.3f}  n={n}</span></h3>'
            '<table><tr><th>Parameter</th><th>Value</th>'
            '<th>Std Err</th><th>t-stat</th><th>Sig</th></tr>'
            f'{rows}</table>'
            '<p class="sig-note">★★★ |t|>3.3  ★★ |t|>2.6  ★ |t|>1.96</p>'
        )

    def _write(fname: str, title: str, body: str) -> None:
        path = os.path.join(output_dir, fname)
        doc = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>{CSS}</head><body>"
            f"<h1>{_h.escape(title)}</h1>"
            f"<p>Stage 5b — trip-level models | generated by stage5b_trip_destination.py</p>"
            f"{body}</body></html>"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        logger.info("Stage 5b HTML report → %s", path)

    os.makedirs(output_dir, exist_ok=True)

    # ── destination HTML ──────────────────────────────────────────────────
    dest_body = "<h2>Trip Stop Destination Models (MNL per purpose)</h2>" + "".join(
        _dest_section(p, r) for p, r in dest_models.items()
    ) if dest_models else "<p>No destination models estimated.</p>"
    _write("stage5b_trip_dest_models_coefficients.html",
           "Stage 5b — Trip Stop Destination Choice Coefficients", dest_body)

    # ── mode HTML ─────────────────────────────────────────────────────────
    mode_body = "<h2>Trip Mode Choice Models (NL per purpose)</h2>" + "".join(
        _mode_section(p, m) for p, m in mode_models.items()
    ) if mode_models else "<p>No trip mode models estimated.</p>"
    _write("stage5b_trip_mode_models_coefficients.html",
           "Stage 5b — Trip Mode Choice Coefficients", mode_body)

    # ── combined HTML ─────────────────────────────────────────────────────
    _write("stage5b_trip_models_coefficients.html",
           "Stage 5b — Trip Destination + Mode Choice Coefficients",
           dest_body + mode_body)


# ---------------------------------------------------------------------------
# Full Stage 5b pipeline (destination + mode)
# ---------------------------------------------------------------------------

def run_stage5b_pipeline(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    skims_path: str,
    landuse: pd.DataFrame,
    output_dir: str = "saved_data",
    g_tripdest: Optional[list] = None,
    validate: bool = True,
    logsums=None,
) -> dict:
    """
    Run the complete Stage 5b pipeline:
      1. Trip stop destination choice (MNL per purpose)
      2. Trip-level mode choice (nested logit per purpose)
      3. Save HTML coefficient reports for both

    Returns
    -------
    dict with keys:
        'trips'            : augmented trip table
        'dest_models'      : {purpose: dest model result dict}
        'mode_models'      : {purpose: NestedLogitModeModel}
        'validation'       : destination validation DataFrame
        'bic_total'        : sum of destination segment BICs
    """
    logger.info("=" * 60)
    logger.info("Stage 5b pipeline — trip destination + trip mode")
    logger.info("=" * 60)

    # ── Step 1: trip stop destination ────────────────────────────────────
    dest_result = run_stage5b(
        trips        = trips,
        persons      = persons,
        skims_path   = skims_path,
        landuse      = landuse,
        g_tripdest   = g_tripdest,
        output_dir   = output_dir,
        validate     = validate,
        save_models  = True,
        logsums      = logsums,
    )
    dest_models = dest_result.get("models", {})

    # ── Step 2: trip-level mode choice ────────────────────────────────────
    mode_models = estimate_trip_mode_models(
        trips      = trips,
        persons    = persons,
        landuse    = landuse,
        output_dir = output_dir,
        save_models = True,
    )

    # ── Step 2b: trip-level transit access (constrained) ───────────────
    # Gated by STAGE5B_TRANSIT_ACCESS=1. Estimates access choice on Transit
    # legs with MODE1 labels, then assigns access + pnr_trip markers.
    access_model = {}
    try:
        import os as _os2b
        if _os2b.environ.get('STAGE5B_TRANSIT_ACCESS', '0').strip() == '1':
            access_model = estimate_trip_transit_access(
                trips, persons, landuse, skim_lookup=None,
                output_dir=output_dir)
    except Exception as _acc_e:
        logger.warning(f"Stage 5b transit-access estimation skipped ({_acc_e})")
        access_model = {}

    # ── Step 3: HTML coefficient reports ─────────────────────────────────
    try:
        _write_stage5b_html(dest_models, mode_models, output_dir)
    except Exception as exc:
        logger.warning("Stage 5b HTML report generation failed: %s", exc)

    return {
        "trips":       dest_result["trips"],
        "dest_models": dest_models,
        "mode_models": mode_models,
        "access_model": access_model,
        "validation":  dest_result.get("validation", pd.DataFrame()),
        "bic_total":   dest_result.get("bic_total", float("inf")),
        "metadata":    dest_result.get("metadata", {}),
    }

