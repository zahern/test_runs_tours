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
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import openmatrix as omx

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
GA_CANDIDATE_VARS = [
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

    Returns a pd.Series with the same index as df.
    """
    # Skim is N×N (N = number of zones in the skim, e.g. 371). taz_index maps a
    # zone id to its 0-based position, but when it is built from a landuse table
    # that carries MORE zones than the skim (or a different ordering), the mapped
    # position can exceed N-1 and overflow the skim (IndexError: index 485 out of
    # bounds for size 371). Guard every mapped index to the valid [0, N-1] range;
    # out-of-range / unmapped zones fall back to 0 (their detour contribution is
    # dropped rather than crashing the whole trip-destination stage).
    _n_skim = int(skim_dist.shape[0])
    _oor_total = 0

    def _idx(col: str) -> np.ndarray:
        nonlocal _oor_total
        raw = df[col].map(taz_index)
        bad = ~(raw.notna() & (raw >= 0) & (raw < _n_skim))
        _oor_total += int(bad.sum())
        return raw.where(~bad, 0).astype(int).values

    o_idx  = _idx("origin_taz")
    s_idx  = _idx("dest_taz")
    a_idx  = _idx("anchor_down_taz")
    if _oor_total:
        logger.warning(
            f"compute_detour: {_oor_total} zone index(es) were unmapped or outside "
            f"the {_n_skim}-zone skim and were set to 0 (detour dropped for those). "
            "Check that taz_index/landuse zone set matches the skim zone set.")

    d_os = _skim_lookup(skim_dist, o_idx, s_idx)
    d_sa = _skim_lookup(skim_dist, s_idx, a_idx)
    d_oa = _skim_lookup(skim_dist, o_idx, a_idx)

    detour = d_os + d_sa - d_oa
    # Negative detour (numerical noise) → set to 0
    detour = np.maximum(detour, 0.0)
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

def _estimate_mnl_segment(
    stop_trips: pd.DataFrame,
    all_zones: pd.DataFrame,
    skims: dict[str, np.ndarray],
    landuse: pd.DataFrame,
    purpose: str,
    taz_index: dict[int, int],
    g_tripdest: Optional[list[int]],
    zone_id_col: str = "zone_id",
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
    try:
        import larch as lx
        from larch import P, X
    except ImportError:
        logger.error("larch not installed; cannot estimate trip destination model")
        return None

    n_obs = len(stop_trips)
    if n_obs < MIN_STOP_OBS:
        logger.warning(
            "Segment %s: only %d stop observations; skipping estimation",
            purpose, n_obs,
        )
        return None

    if g_tripdest is None:
        active_ga = {v: 1 for v in GA_CANDIDATE_VARS}
    else:
        active_ga = {v: int(b) for v, b in zip(GA_CANDIDATE_VARS, g_tripdest)}

    # ── Build case-alternative (idca) dataset ───────────────────────────────
    zones = all_zones[[zone_id_col]].copy().reset_index(drop=True)
    n_zones = len(zones)

    # Repeat each observation over all zone alternatives
    obs_idx = np.repeat(np.arange(n_obs), n_zones)
    alt_idx = np.tile(np.arange(n_zones), n_obs)

    obs_rows = stop_trips.iloc[obs_idx].reset_index(drop=True)
    alt_rows = zones.iloc[alt_idx].reset_index(drop=True)

    ca = pd.DataFrame()
    ca["case_id"] = obs_rows.index
    ca["alt_id"]  = alt_rows[zone_id_col].values
    ca["chosen"]  = (obs_rows["dest_taz"].values == alt_rows[zone_id_col].values).astype(int)

    # Skim lookups: origin → alternative zone
    o_idx = obs_rows["origin_taz"].map(taz_index).fillna(0).astype(int).values
    a_idx = alt_rows[zone_id_col].map(taz_index).fillna(0).astype(int).values

    # Anchor-down indices (for detour)
    anchor_idx = obs_rows["anchor_down_taz"].map(taz_index).fillna(0).astype(int).values

    # Detour
    if "NetworkDist" in skims:
        d_oa = _skim_lookup(skims["NetworkDist"], o_idx, a_idx)
        d_aa = _skim_lookup(skims["NetworkDist"], a_idx, anchor_idx)
        d_oo = _skim_lookup(skims["NetworkDist"], o_idx, anchor_idx)
        ca["detour_dist_km"] = np.maximum(d_oa + d_aa - d_oo, 0.0)
    else:
        ca["detour_dist_km"] = 0.0

    # Backbone skims
    for alias in ["AutoIVT_AM", "NetworkDist", "AutoCrashRisk", "TransitIVT_AM"]:
        if alias in skims:
            ca[alias] = _skim_lookup(skims[alias], o_idx, a_idx)
        else:
            ca[alias] = 0.0

    # Size term
    ca["log_size"] = compute_size_term(alt_rows.assign(**{zone_id_col: alt_rows[zone_id_col]}),
                                       purpose, landuse, zone_id_col)

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

    # ── Build Larch model ──────────────────────────────────────────────────
    ca = ca.set_index(["case_id", "alt_id"])

    try:
        d = lx.Dataset.construct.from_idca(ca, crack=False)
        m = lx.Model(d)

        # Fixed backbone utilities
        for var in BACKBONE_VARS:
            if var in ca.columns:
                m.utility_ca = m.utility_ca + P(var) * X(var)

        # GA-selected candidates
        for var, active in active_ga.items():
            if active and var in ca.columns:
                m.utility_ca = m.utility_ca + P(var) * X(var)

        m.choice_ca_var = "chosen"

        # Prefer numba on Windows (avoids hwloc/XLA binding errors)
        engine = "numba" if os.name == "nt" else "jax"
        try:
            result = m.maximize_loglike(method="slsqp", engine=engine)
        except Exception:
            result = m.maximize_loglike(method="slsqp", engine="numba")

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
        }

    except Exception as exc:
        logger.error("MNL estimation failed for purpose %s: %s", purpose, exc)
        return None


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
            d_oa = _skim_lookup(skims["NetworkDist"], o_i, a_i)
            d_aa = _skim_lookup(skims["NetworkDist"], a_i, anc_i)
            d_oo = _skim_lookup(skims["NetworkDist"], o_i, anc_i)
            detour = np.maximum(d_oa + d_aa - d_oo, 0.0)
        else:
            detour = np.zeros(len(obs_rows))

        # Trip-boundary ("trip square") constraint: a stop is feasible only if its
        # detour stays within a PER-TRIP envelope scaled by the direct origin→anchor
        # distance (ratio·d_oa + base), capped by the absolute maximum. So a short
        # trip gets a tight square and a long trip a proportionally larger one —
        # no cross-city stop for a 2 km trip. d_oo here is d(origin→anchor).
        _trip_bound = np.minimum(DETOUR_MAX_KM, DETOUR_MAX_RATIO * d_oo + DETOUR_BASE_KM)
        feasible = detour <= _trip_bound

        log_size = compute_size_term(
            alt_rows.assign(**{zone_id_col: alt_rows[zone_id_col]}),
            model_result["purpose"], landuse, zone_id_col,
        )

        V = (
            _p("detour_dist_km") * detour
            + _p("log_size") * log_size
        )
        if "AutoCrashRisk" in skims and "AutoCrashRisk" in pf.index:
            V += _p("AutoCrashRisk") * _skim_lookup(
                skims["AutoCrashRisk"],
                obs_rows["origin_taz"].map(taz_index).fillna(0).astype(int).values,
                alt_rows[zone_id_col].map(taz_index).fillna(0).astype(int).values,
            )

        # Mask infeasible alts
        V[~feasible] = -np.inf

        # Softmax per observation → draw
        V_matrix = V.reshape(n_obs, n_zones)
        V_matrix -= V_matrix.max(axis=1, keepdims=True)
        exp_V = np.exp(V_matrix)
        # Zero out any all-infeasible rows to avoid /0 — fall back to uniform
        row_sums = exp_V.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
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

    Returns
    -------
    dict with keys:
        'trips'       : full trip table augmented with detour_dist_km,
                        is_tour_dest_leg, anchor_down_taz, dest_taz_predicted
        'models'      : dict of purpose → model result dict
        'validation'  : validation summary DataFrame (if validate=True)
        'bic_total'   : sum of segment BICs (fitness score for GA)
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Stage 5b: trip-level destination choice starting")

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
                return {'trips': _plan_full, 'models': {}, 'validation': None, 'bic_total': 0.0}
            min_seq = trips.groupby('ptour_id')['seq'].transform('min')
            max_seq = trips.groupby('ptour_id')['seq'].transform('max')
            mid_seq = (min_seq + max_seq) / 2.0
            trips['is_outbound'] = (trips['seq'] <= mid_seq).astype(int)
            trips['tour_id'] = trips['ptour_id']
            trips['trip_num_in_tour'] = trips.groupby('ptour_id').cumcount() + 1
            logger.warning("Stage 5b: 'is_outbound' derived from ptour_id/seq (plan-format input)")
        else:
            logger.error("Stage 5b: 'is_outbound' column missing and cannot be derived — aborting")
            return {'trips': trips, 'models': {}, 'validation': None, 'bic_total': 0.0}


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
            )
            if result is not None:
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

    # ── Step 3: HTML coefficient reports ─────────────────────────────────
    try:
        _write_stage5b_html(dest_models, mode_models, output_dir)
    except Exception as exc:
        logger.warning("Stage 5b HTML report generation failed: %s", exc)

    return {
        "trips":       dest_result["trips"],
        "dest_models": dest_models,
        "mode_models": mode_models,
        "validation":  dest_result.get("validation", pd.DataFrame()),
        "bic_total":   dest_result.get("bic_total", float("inf")),
    }

