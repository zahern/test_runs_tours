#!/usr/bin/env python
"""
stage7_parking_choice.py
========================
Stage 7 - Parking Choice + Precise Trip End Location
=======================================================

POLARIS-style parking choice model for auto trips (DA, SR, Park-and-Ride).

What this stage does
--------------------
Stage 6 output = full trip chain with activities and trips.
Stage 7 adds parking choice for every auto-mode trip leg:

    auto trip ──> parking location ──walk leg──> activity

Parking alternatives (per POLARIS):
  1. ONSTREET      - free/cheap, high search time, walk to destination
  2. GARAGE        - paid, reserved/guaranteed, walk to destination
  3. PARK_AND_RIDE - transit access, only for Transit mode

For Park-and-Ride: the parking choice IS the transit access mode choice.

Output format (MATSim/SUMO compatible):
  - trip row: adds parking_location_id, parking_zone, parking_cost, parking_walk_time
  - adds walk leg: parking -> activity (mode=Walk)
  - precise coordinates (x, y) for parking location

Dependencies:
  - saved_data/location_table.csv (from build_location_table.py)
  - saved_data/zone_landuse_overture.csv or zone_order_and_landuse.csv (parking supply)
  - Skims for walk distance from parking to destination

Usage:
    from stage7_parking_choice import run_stage7

    plan_with_parking = run_stage7(
        plan=stage6_plan,
        persons=persons,
        location_table=location_df,
        landuse=landuse_df,
        skims_path='downcasted_matrices_base.omx',
        skim_lookup=skim_lookup,
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
    import larch as lx
    from larch import P, X
    LARCH_AVAILABLE = True
except ImportError:
    LARCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("larch not installed - parking choice will use heuristic fallback")

logger = logging.getLogger(__name__)

# ── Parking Types (POLARIS convention) ───────────────────────────────────────
PARKING_ONSTREET = 1      # free/cheap on-street, search required
PARKING_GARAGE = 2        # paid garage/lot, guaranteed space
PARKING_PNR = 3           # park-and-ride (transit access)

PARKING_NAMES = {
    PARKING_ONSTREET: 'OnStreet',
    PARKING_GARAGE: 'Garage',
    PARKING_PNR: 'ParkAndRide',
}

# ── Parking Choice Model Variables (from POLARIS parking_choice_records) ──────
# Variables used in utility:
#   parking_fee (cost)
#   distance_to_G (walk distance from garage to destination)
#   act_dur_hr (activity duration in hours)
#   ln_dest_empdens, ln_dest_popdens (destination density)
#   female, hhsize, ln_hhinc (person)
#   urgent (time pressure)
#   age_above_60
#   type (parking type: onstreet/garage/pnr)

PARKING_UEC = {
    # On-street parking
    'c_onstreet_fee': ('parking_fee', None),
    'c_onstreet_walk': ('walk_dist_km', None),
    'c_onstreet_search': ('search_time_min', None),
    'c_onstreet_duration': ('act_dur_hr', None),
    'c_onstreet_empdens': ('ln_dest_empdens', None),
    'c_onstreet_popdens': ('ln_dest_popdens', None),
    'c_onstreet_female': ('female', None),
    'c_onstreet_hhsize': ('hhsize', None),
    'c_onstreet_income': ('ln_hhinc', None),
    'c_onstreet_urgent': ('urgent', None),
    'c_onstreet_senior': ('age_above_60', None),

    # Garage parking
    'c_garage_fee': ('parking_fee', None),
    'c_garage_walk': ('walk_dist_km', None),
    'c_garage_duration': ('act_dur_hr', None),
    'c_garage_empdens': ('ln_dest_empdens', None),
    'c_garage_popdens': ('ln_dest_popdens', None),
    'c_garage_female': ('female', None),
    'c_garage_hhsize': ('hhsize', None),
    'c_garage_income': ('ln_hhinc', None),
    'c_garage_urgent': ('urgent', None),
    'c_garage_senior': ('age_above_60', None),

    # Park-and-Ride (only for transit)
    'c_pnr_fee': ('parking_fee', None),
    'c_pnr_walk': ('walk_dist_km', None),
    'c_pnr_transit_ivt': ('transit_ivt_hr', None),
    'c_pnr_female': ('female', None),
    'c_pnr_income': ('ln_hhinc', None),
}


def _open_skims(skims):
    if isinstance(skims, str):
        return omx.open_file(skims, 'r'), True
    return skims, False


def _load_parking_supply(location_table: pd.DataFrame, landuse: pd.DataFrame) -> pd.DataFrame:
    """
    Build parking supply table from location_table and landuse.

    Returns DataFrame with one row per parking facility:
        location, zone, x, y, land_use, parking_type,
        capacity (spaces), avg_parking_cost, walk_dist_to_dest (filled per query)
    """
    # Filter location_table for parking-relevant land uses.
    # Real structures first (land_use='parking' from OSM/Overture); legacy
    # proxy backs it up where no structures are mapped. On-street: all zones
    # have implicit on-street capacity.
    parking_lu = ['parking', 'industrial', 'civic', 'retail', 'office', 'education']

    # Known parking facilities from location table
    facilities = location_table[location_table['land_use'].isin(parking_lu)].copy()

    if facilities.empty:
        logger.warning("No parking facilities found in location_table; creating synthetic")
        # Create synthetic garage per zone
        zones = landuse['ORDER'].unique() if 'ORDER' in landuse.columns else landuse['zone_id'].unique()
        facilities = pd.DataFrame({
            'location': range(1, len(zones) + 1),
            'zone': zones,
            'x': 0.0, 'y': 0.0,  # will be filled from centroids
            'land_use': 'garage',
            'parking_type': PARKING_GARAGE,
            'capacity': 200,
            'avg_parking_cost': 5.0,
        })

    # Add parking type
    facilities['parking_type'] = PARKING_GARAGE

    # Add capacity estimate from landuse (parking spaces per zone)
    if 'capacity' not in facilities.columns:
        _zkey = 'ORDER' if 'ORDER' in landuse.columns else 'zone_id'
        # Prefer observed OSM capacity (parking_capacity per zone, apportioned
        # across that zone's facilities) over the employment proxy.
        if 'parking_capacity' in landuse.columns:
            _cap = landuse.set_index(_zkey)['parking_capacity']
            _nfac = facilities.groupby('zone').size()
            facilities['_zone_cap'] = facilities['zone'].map(_cap).fillna(0.0)
            facilities['_zone_n'] = facilities['zone'].map(_nfac).fillna(1.0).clip(lower=1.0)
            _apport = (facilities['_zone_cap'] / facilities['_zone_n']).fillna(0.0)
            # fall back to the employment proxy only where OSM mapped nothing
            zone_emp = landuse.set_index(_zkey)['TOTEMP']
            _proxy = facilities['zone'].map(zone_emp).fillna(200).astype(int) * 0.1
            facilities['capacity'] = np.where(_apport > 0, _apport, _proxy)
            facilities['capacity'] = pd.Series(facilities['capacity']).clip(lower=50, upper=2000).astype(int)
            facilities = facilities.drop(columns=['_zone_cap', '_zone_n'])
        else:
            # Estimate from zone employment/retail
            zone_emp = landuse.set_index(_zkey)['TOTEMP']
            facilities['capacity'] = facilities['zone'].map(zone_emp).fillna(200).astype(int) * 0.1
            facilities['capacity'] = facilities['capacity'].clip(lower=50, upper=2000).astype(int)

    # Add avg_parking_cost from location_table or estimate
    if 'avg_parking_cost' not in facilities.columns:
        # CBD zones higher cost
        cbd_zones = [1, 2, 3, 4, 5]  # inner Brisbane CBD zones
        facilities['avg_parking_cost'] = facilities['zone'].apply(
            lambda z: 8.0 if z in cbd_zones else 3.0
        )

    return facilities


def _build_parking_ca(plan: pd.DataFrame, persons: pd.DataFrame,
                      location_table: pd.DataFrame, landuse: pd.DataFrame,
                      skim_lookup: dict) -> pd.DataFrame:
    """
    Build case-alternative dataset for parking choice.

    Each case = one auto trip leg (DA or SR) ending at an activity.
    Alternatives = available parking facilities in/near destination zone.
    """
    # Filter to auto trip legs that END at an activity (not home)
    auto_trips = plan[
        (plan['row_type'] == 'trip') &
        (plan['mode'].isin([1, 2])) &  # DA=1, SR=2
        (plan['purpose'] != 'home')
    ].copy()

    if auto_trips.empty:
        return pd.DataFrame()

    # Precompute next activity per person using shift(-1) instead of per-trip scan
    plan_sorted = plan.sort_values(['person_id', 'seq']).reset_index(drop=True)
    # Build next activity columns per person
    next_act = plan_sorted[plan_sorted['row_type'] == 'activity'].copy()
    next_act = next_act.rename(columns={
        'seq': 'next_act_seq',
        'predicted_duration_min': 'next_act_dur_min'
    })[['person_id', 'next_act_seq', 'next_act_dur_min']]
    # Keep each trip's original plan row so the immediate next activity can be
    # selected unambiguously after the person-level join below.
    auto_trips['_trip_row'] = auto_trips.index
    # Join next activity onto trips (each trip gets its person's next activity)
    auto_trips = auto_trips.merge(
        next_act,
        left_on='person_id',
        right_on='person_id',
        how='left'
    )
    # Keep only trips where next activity exists and seq < next_act_seq
    # (some trips may not have a following activity)
    auto_trips = auto_trips[
        auto_trips['next_act_seq'].notna() &
        (auto_trips['seq'] < auto_trips['next_act_seq'])
    ].copy()
    # A person-level join matches every later activity; keep only the immediate
    # next activity per trip. Otherwise one trip becomes several CA cases with
    # duplicated case_ids and ambiguous durations.
    auto_trips = auto_trips.sort_values(
        ['person_id', 'seq', 'next_act_seq']
    ).drop_duplicates(subset=['_trip_row'], keep='first')
    if auto_trips.empty:
        return pd.DataFrame()
    auto_trips['act_dur_hr'] = auto_trips['next_act_dur_min'].fillna(60) / 60.0

    facilities = _load_parking_supply(location_table, landuse)

    # Person attributes - vectorized merge
    pp = persons.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})
    pp['person_id'] = pp['person_id'].astype(str)
    person_cols = ['person_id', 'sexp', 'nprd', 'incp', 'age5p']
    pp = pp[person_cols].drop_duplicates('person_id')
    auto_trips = auto_trips.merge(pp, on='person_id', how='left')

    # Compute person attributes vectorized
    auto_trips['female'] = (auto_trips['sexp'] == 2).astype(int).fillna(0)
    auto_trips['hhsize'] = auto_trips['nprd'].fillna(1).astype(int)
    auto_trips['incp'] = auto_trips['incp'].fillna(50000).astype(float)
    auto_trips['ln_hhinc'] = np.log(np.maximum(auto_trips['incp'], 1))
    auto_trips['age'] = auto_trips['age5p'].fillna(30).astype(int)
    auto_trips['age_above_60'] = (auto_trips['age'] >= 60).astype(int)

    # Urgent: work/edu with short duration
    auto_trips['urgent'] = (
        auto_trips['purpose'].isin(['work', 'edu']) &
        (auto_trips['act_dur_hr'] < 4)
    ).astype(int)

    # Destination zone density - vectorized merge
    zone_col = 'ORDER' if 'ORDER' in landuse.columns else 'zone_id'
    lu_cols = [zone_col, 'TOTEMP', 'TOTPOP']
    landuse_ = landuse[lu_cols].drop_duplicates(zone_col)
    landuse_ = landuse_.rename(columns={zone_col: 'dest_zone'})
    auto_trips['dest_zone'] = auto_trips['DTAZ'].astype(int)
    auto_trips = auto_trips.merge(landuse_, on='dest_zone', how='left')
    auto_trips['empdens'] = auto_trips['TOTEMP'].fillna(1000).astype(float)
    auto_trips['popdens'] = auto_trips['TOTPOP'].fillna(2000).astype(float)
    auto_trips['ln_dest_empdens'] = np.log(np.maximum(auto_trips['empdens'], 1))
    auto_trips['ln_dest_popdens'] = np.log(np.maximum(auto_trips['popdens'], 1))

    # Precompute walk_dist_km per destination zone (intra-zonal skim only)
    walk_dist_map = {}
    if skim_lookup:
        for key, vals in skim_lookup.items():
            if key[0] == key[1]:  # intra-zonal
                walk_dist_map[key[0]] = vals.get('DISTWALK', 500) / 1000.0  # m to km
    auto_trips['walk_dist_km'] = auto_trips['dest_zone'].map(walk_dist_map).fillna(0.5)

    # Precompute facilities per zone once (dict of small frames)
    zone_fac_map = {}
    for zone, grp in facilities.groupby('zone'):
        zone_fac_map[zone] = grp
    # Add fallback synthetic facilities for zones missing
    fallback_fac = pd.DataFrame({
        'location': [-1, -2],
        'parking_type': [PARKING_ONSTREET, PARKING_GARAGE],
        'capacity': [100, 200],
        'avg_parking_cost': [0.0, 5.0],
        'x': [0.0, 0.0], 'y': [0.0, 0.0],
    })

    # Build CA by cross-joining auto_trips with their zone's facilities
    ca_rows = []
    for idx, trip in auto_trips.iterrows():
        dest_zone = int(trip['dest_zone'])
        zone_facilities = zone_fac_map.get(dest_zone, fallback_fac)
        zone_facilities = zone_facilities.assign(zone=dest_zone)

        # Vectorized per-facility attributes
        zone_facilities = zone_facilities.copy()
        zone_facilities['ptype'] = zone_facilities['parking_type'].astype(int)
        zone_facilities['parking_fee'] = zone_facilities['avg_parking_cost'].fillna(0.0).astype(float)
        zone_facilities['search_time_min'] = np.where(
            zone_facilities['ptype'] == PARKING_ONSTREET, 5.0, 0.0
        )
        zone_facilities['walk_dist_km'] = trip['walk_dist_km']
        zone_facilities['act_dur_hr'] = trip['act_dur_hr']
        zone_facilities['ln_dest_empdens'] = trip['ln_dest_empdens']
        zone_facilities['ln_dest_popdens'] = trip['ln_dest_popdens']
        zone_facilities['female'] = trip['female']
        zone_facilities['hhsize'] = trip['hhsize']
        zone_facilities['ln_hhinc'] = trip['ln_hhinc']
        zone_facilities['urgent'] = trip['urgent']
        zone_facilities['age_above_60'] = trip['age_above_60']
        zone_facilities['is_onstreet'] = (zone_facilities['ptype'] == PARKING_ONSTREET).astype(int)
        zone_facilities['is_garage'] = (zone_facilities['ptype'] == PARKING_GARAGE).astype(int)
        zone_facilities['is_pnr'] = (zone_facilities['ptype'] == PARKING_PNR).astype(int)
        zone_facilities['fac_zone'] = dest_zone
        zone_facilities['fac_x'] = zone_facilities.get('x', 0.0).fillna(0.0)
        zone_facilities['fac_y'] = zone_facilities.get('y', 0.0).fillna(0.0)
        zone_facilities['case_id'] = trip['_trip_row']
        zone_facilities['person_id'] = trip['person_id']
        zone_facilities['alt_id'] = zone_facilities['location'].astype(int)
        zone_facilities['chosen'] = 0

        # Keep only needed columns
        ca_rows.append(zone_facilities[[
            'case_id', 'person_id', 'alt_id', 'parking_type', 'chosen',
            'parking_fee', 'walk_dist_km', 'search_time_min', 'act_dur_hr',
            'ln_dest_empdens', 'ln_dest_popdens', 'female', 'hhsize',
            'ln_hhinc', 'urgent', 'age_above_60', 'is_onstreet',
            'is_garage', 'is_pnr', 'fac_zone', 'fac_x', 'fac_y'
        ]])

    if not ca_rows:
        return pd.DataFrame()

    ca = pd.concat(ca_rows, ignore_index=True)
    # Auto-trip cases are only offered on-street/garage (PNR is a transit
    # access mode, not a parking choice for DA/SR legs).
    ca = ca[ca['parking_type'].isin([PARKING_ONSTREET, PARKING_GARAGE])].reset_index(drop=True)
    return ca


def map_vehparked_choice(ca_df: pd.DataFrame, vehparked: pd.Series) -> pd.DataFrame:
    """Attach observed parking choice from QTS-HHTS VEHPARKED to a CA table.

    vehparked: indexed like ca_df's source trips ('On the street' -> the
    on-street alt chosen; 'Car park (off-street)' -> garage alt chosen).
    Rows with any other value keep chosen=0 and should be excluded before
    estimation. Estimation with a constant-chosen column is degenerate —
    _estimate_parking_model() refuses it (returns None -> heuristic).
    """
    ca_df = ca_df.copy()
    obs = vehparked.map({'On the street': PARKING_ONSTREET,
                         'Car park (off-street)': PARKING_GARAGE})
    want = obs.reindex(ca_df['case_id']).values if 'case_id' in ca_df.columns else None
    if want is not None:
        ca_df['chosen'] = (ca_df['parking_type'].values == want).astype(int)
    return ca_df


def _estimate_parking_model(ca_df: pd.DataFrame) -> Optional[object]:
    """Estimate parking choice MNL using Larch.

    Identified spec: every coefficient is interacted with its parking-type
    dummy (on-street is the reference for ASCs). The previous spec summed
    on-street + garage + PNR coefficients on EVERY alternative row, so only
    the sums were identified and prediction evaluated to uniform noise.
    Refuses degenerate input (constant `chosen`) instead of fitting garbage.
    """
    if not LARCH_AVAILABLE or ca_df.empty:
        return None

    try:
        if ca_df['chosen'].nunique(dropna=False) < 2:
            logger.warning("parking chosen has no variation (all 0?) — "
                           "map observed choice via map_vehparked_choice() first; "
                           "skipping estimation (heuristic fallback will be used)")
            return None
        # Need at least 2 alternatives per case
        alt_counts = ca_df.groupby('case_id')['alt_id'].count()
        if (alt_counts < 2).any():
            logger.warning("Some cases have <2 parking alternatives; estimation may be weak")

        # Create Larch dataset
        ca_df_idx = ca_df.set_index(['case_id', 'alt_id'])
        dt = lx.Dataset.construct.from_idca(ca_df_idx, crack=False)
        m = lx.Model(dt)

        # Utility: type-interacted terms + garage ASC (on-street = reference).
        # Only on-street/garage alts are offered to auto cases (see
        # _build_parking_ca), so no PNR block is estimated here.
        m.utility_ca = (
            P('asc_garage') * X('is_garage')
            + P('c_onstreet_fee') * X('parking_fee') * X('is_onstreet')
            + P('c_onstreet_walk') * X('walk_dist_km') * X('is_onstreet')
            + P('c_onstreet_search') * X('search_time_min') * X('is_onstreet')
            + P('c_onstreet_duration') * X('act_dur_hr') * X('is_onstreet')
            + P('c_onstreet_empdens') * X('ln_dest_empdens') * X('is_onstreet')
            + P('c_onstreet_popdens') * X('ln_dest_popdens') * X('is_onstreet')
            + P('c_onstreet_female') * X('female') * X('is_onstreet')
            + P('c_onstreet_hhsize') * X('hhsize') * X('is_onstreet')
            + P('c_onstreet_income') * X('ln_hhinc') * X('is_onstreet')
            + P('c_onstreet_urgent') * X('urgent') * X('is_onstreet')
            + P('c_onstreet_senior') * X('age_above_60') * X('is_onstreet')
            + P('c_garage_fee') * X('parking_fee') * X('is_garage')
            + P('c_garage_walk') * X('walk_dist_km') * X('is_garage')
            + P('c_garage_duration') * X('act_dur_hr') * X('is_garage')
            + P('c_garage_empdens') * X('ln_dest_empdens') * X('is_garage')
            + P('c_garage_popdens') * X('ln_dest_popdens') * X('is_garage')
            + P('c_garage_female') * X('female') * X('is_garage')
            + P('c_garage_hhsize') * X('hhsize') * X('is_garage')
            + P('c_garage_income') * X('ln_hhinc') * X('is_garage')
            + P('c_garage_urgent') * X('urgent') * X('is_garage')
            + P('c_garage_senior') * X('age_above_60') * X('is_garage')
        )

        # All offered alternatives are available (PNR filtered upstream).
        m.availability_ca_var = 1

        m.choice_ca_var = 'chosen'

        # Estimate — JAX-first (large-model policy). BUGFIX: engine belongs on
        # the Model (compute_engine), not maximize_loglike(engine=...) which
        # raises TypeError on Larch 6 and broke all estimation.
        import os as _os7
        _force_jax7 = _os7.environ.get('LARCH_FORCE_JAX', '1').strip() != '0'
        _engines7 = ('jax', 'numba') if _force_jax7 else (('numba', 'jax') if os.name == 'nt' else ('jax', 'numba'))
        _last_e = None
        for engine in _engines7:
            try:
                try:
                    m.compute_engine = engine
                    m.mangle()
                    m.unmangle()
                except Exception:
                    pass
                m.maximize_loglike(method='slsqp', quiet=True)
                break
            except Exception as _e7:
                _last_e = _e7
                continue
        else:
            raise _last_e if _last_e is not None else RuntimeError("parking estimation failed")

        logger.info(f"Parking model estimated: LL={m.loglike:.1f}, params={len(m.pf)}")
        return m

    except Exception as e:
        logger.error(f"Parking model estimation failed: {e}")
        return None


# Coefficient -> (data column, parking-type mask column). Mirrors the
# type-interacted estimation spec above; prediction MUST use this map so
# estimated parameters actually enter the utilities (previously coef names
# were looked up directly in ca_df, matched nothing, utility was 0).
PARKING_COEF_MAP = {
    'asc_garage': (None, 'is_garage'),
    'c_onstreet_fee': ('parking_fee', 'is_onstreet'),
    'c_onstreet_walk': ('walk_dist_km', 'is_onstreet'),
    'c_onstreet_search': ('search_time_min', 'is_onstreet'),
    'c_onstreet_duration': ('act_dur_hr', 'is_onstreet'),
    'c_onstreet_empdens': ('ln_dest_empdens', 'is_onstreet'),
    'c_onstreet_popdens': ('ln_dest_popdens', 'is_onstreet'),
    'c_onstreet_female': ('female', 'is_onstreet'),
    'c_onstreet_hhsize': ('hhsize', 'is_onstreet'),
    'c_onstreet_income': ('ln_hhinc', 'is_onstreet'),
    'c_onstreet_urgent': ('urgent', 'is_onstreet'),
    'c_onstreet_senior': ('age_above_60', 'is_onstreet'),
    'c_garage_fee': ('parking_fee', 'is_garage'),
    'c_garage_walk': ('walk_dist_km', 'is_garage'),
    'c_garage_duration': ('act_dur_hr', 'is_garage'),
    'c_garage_empdens': ('ln_dest_empdens', 'is_garage'),
    'c_garage_popdens': ('ln_dest_popdens', 'is_garage'),
    'c_garage_female': ('female', 'is_garage'),
    'c_garage_hhsize': ('hhsize', 'is_garage'),
    'c_garage_income': ('ln_hhinc', 'is_garage'),
    'c_garage_urgent': ('urgent', 'is_garage'),
    'c_garage_senior': ('age_above_60', 'is_garage'),
}


def _predict_parking(plan: pd.DataFrame, persons: pd.DataFrame,
                     parking_model: object,
                     location_table: pd.DataFrame, landuse: pd.DataFrame,
                     skim_lookup: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Predict parking choice for auto trip legs and add walk legs."""
    if parking_model is None or not LARCH_AVAILABLE:
        return _heuristic_parking(plan, location_table, landuse, skim_lookup, rng)

    # Build CA dataset for prediction (same builder as estimation).
    ca_df = _build_parking_ca(plan, persons, location_table, landuse, skim_lookup)
    if ca_df.empty:
        return plan

    # Compute utilities via the explicit coef map (estimation-spec mirror).
    pf = parking_model.pf
    coef = {name: float(pf.loc[name, 'value']) for name in pf.index if 'value' in pf.columns}

    ca_df['utility'] = 0.0
    for coef_name, value in coef.items():
        spec = PARKING_COEF_MAP.get(coef_name)
        if spec is None:
            logger.warning(f"parking coef '{coef_name}' not in COEF map — skipped")
            continue
        col, mask_col = spec
        mask = ca_df[mask_col].values.astype(bool) if mask_col in ca_df.columns else np.ones(len(ca_df), bool)
        if col is None:
            ca_df.loc[mask, 'utility'] += value  # ASC
        elif col in ca_df.columns:
            ca_df.loc[mask, 'utility'] += value * ca_df.loc[mask, col].values
        else:
            logger.warning(f"parking utility column '{col}' missing — term skipped")

    # Softmax choice per case - vectorized
    util = ca_df['utility'].values
    case_ids = ca_df['case_id'].values
    unique_cases = np.unique(case_ids)

    chosen_indices = []
    for case_id in unique_cases:
        mask = case_ids == case_id
        u = util[mask]
        u = u - u.max()
        probs = np.exp(u)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            logger.warning(f"Zero probability sum for case {case_id}; using uniform")
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / prob_sum
        # Vectorized categorical draw
        chosen_local = rng.choice(len(probs), p=probs)
        global_idx = np.where(mask)[0][chosen_local]
        chosen_indices.append(global_idx)

    chosen_df = ca_df.iloc[chosen_indices].copy()

    # Merge back to plan - vectorized
    plan = plan.copy()
    merge_cols = ['parking_type', 'parking_location_id', 'parking_zone',
                  'parking_cost', 'parking_walk_dist_km', 'parking_walk_time_min',
                  'parking_x', 'parking_y']
    for col in merge_cols:
        if col not in plan.columns:
            plan[col] = np.nan

    # Map chosen values by case_id (which is the original plan index)
    for col in merge_cols:
        source_col = col if col in chosen_df.columns else (
            'parking_type' if col == 'parking_type' else
            'alt_id' if col == 'parking_location_id' else
            'fac_zone' if col == 'parking_zone' else
            'parking_fee' if col == 'parking_cost' else
            'walk_dist_km' if col == 'parking_walk_dist_km' else
            'walk_dist_km' if col == 'parking_walk_time_min' else  # will multiply after
            'fac_x' if col == 'parking_x' else
            'fac_y' if col == 'parking_y' else None
        )
        if source_col is None or source_col not in chosen_df.columns:
            continue
        # case_id in chosen_df maps to plan index
        plan.loc[chosen_df['case_id'].values, col] = chosen_df[source_col].values

    # Fix parking_walk_time_min (was set to walk_dist_km, need * 12)
    if 'parking_walk_time_min' in plan.columns and 'parking_walk_dist_km' in plan.columns:
        mask = plan['parking_walk_dist_km'].notna()
        plan.loc[mask, 'parking_walk_time_min'] = plan.loc[mask, 'parking_walk_dist_km'] * 12.0

    return plan


def _heuristic_parking(plan: pd.DataFrame, location_table: pd.DataFrame,
                       landuse: pd.DataFrame, skim_lookup: dict,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Heuristic parking assignment when Larch unavailable."""
    plan = plan.copy()
    auto_mask = (plan['row_type'] == 'trip') & (plan['mode'].isin([1, 2])) & (plan['purpose'] != 'home')

    # CBD zones = garage preferred
    cbd_zones = {1, 2, 3, 4, 5}

    # Vectorized: get all auto trip indices at once
    auto_indices = plan.index[auto_mask].to_numpy()
    if len(auto_indices) == 0:
        return plan

    # Extract arrays
    dest_zones = plan.loc[auto_indices, 'DTAZ'].astype(int).to_numpy()
    purposes = plan.loc[auto_indices, 'purpose'].to_numpy()
    act_durs = plan.loc[auto_indices, 'predicted_duration_min'].to_numpy() if 'predicted_duration_min' in plan.columns else np.full(len(auto_indices), 60.0)

    # Vectorized heuristic
    is_cbd = np.isin(dest_zones, list(cbd_zones))
    is_long = act_durs > 240
    is_garage = is_cbd | is_long

    ptypes = np.where(is_garage, PARKING_GARAGE, PARKING_ONSTREET)
    costs = np.where(is_garage, 5.0, 0.0)
    walk_kms = np.where(is_garage, 0.3, 0.5)

    # Batch write
    plan.loc[auto_indices, 'parking_type'] = ptypes
    plan.loc[auto_indices, 'parking_location_id'] = -1
    plan.loc[auto_indices, 'parking_zone'] = dest_zones
    plan.loc[auto_indices, 'parking_cost'] = costs
    plan.loc[auto_indices, 'parking_walk_dist_km'] = walk_kms
    plan.loc[auto_indices, 'parking_walk_time_min'] = walk_kms * 12.0

    # Facility coordinates - vectorized via map
    # Precompute zone -> (x,y) map
    zone_coords = {}
    for _, row in location_table.groupby('zone').first().iterrows():
        zone_coords[int(row['zone'])] = (float(row.get('x', 0.0)), float(row.get('y', 0.0)))

    if zone_coords:
        # Map coordinates
        coords = np.array([zone_coords.get(int(z), (0.0, 0.0)) for z in dest_zones])
        plan.loc[auto_indices, 'parking_x'] = coords[:, 0]
        plan.loc[auto_indices, 'parking_y'] = coords[:, 1]
    else:
        plan.loc[auto_indices, 'parking_x'] = 0.0
        plan.loc[auto_indices, 'parking_y'] = 0.0

    return plan


def _add_walk_legs(plan: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Insert walk legs from parking to activity for auto trips.

    For each auto trip ending at activity:
      1. Trip ends at parking location (update DTAZ to parking zone)
      2. Insert walk leg: parking -> activity destination
    """
    plan = plan.sort_values(['person_id', 'seq']).reset_index(drop=True)

    # Identify qualifying auto trips vectorized
    auto_mask = (
        (plan['row_type'] == 'trip') &
        (plan['mode'].isin([1, 2])) &  # DA, SR
        (plan['purpose'] != 'home') &
        ('parking_type' in plan.columns) &
        (plan['parking_type'].notna())
    )

    if not auto_mask.any():
        return plan

    # Get qualifying trip indices
    trip_indices = plan.index[auto_mask].to_numpy()

    # Build walk legs vectorized
    walk_legs = pd.DataFrame({
        'person_id': plan.loc[trip_indices, 'person_id'].values,
        'row_type': 'trip',
        'purpose': plan.loc[trip_indices, 'purpose'].values,
        'mode': 3,  # Walk
        'mode_name': 'Walk',
        'OTAZ': plan.loc[trip_indices, 'parking_zone'].fillna(plan.loc[trip_indices, 'DTAZ']).astype(int).values,
        'DTAZ': plan.loc[trip_indices, 'DTAZ'].astype(int).values,
        'STARTIME': plan.loc[trip_indices, 'ENDTIME'].values,
        'ENDTIME': plan.loc[trip_indices, 'ENDTIME'].values + plan.loc[trip_indices, 'parking_walk_time_min'].fillna(5.0).values,
        'travtime': plan.loc[trip_indices, 'parking_walk_time_min'].fillna(5.0).values,
        'predicted_duration_min': plan.loc[trip_indices, 'parking_walk_time_min'].fillna(5.0).values,
        'parking_walk_leg': True,
        'parking_location_id': plan.loc[trip_indices, 'parking_location_id'].fillna(-1).astype(int).values,
        'OTAZ_original': plan.loc[trip_indices, 'OTAZ'].values,
    })

    # Assign sequential seq numbers starting after max existing
    seq_start = plan['seq'].max() + 1 if 'seq' in plan.columns else len(plan)
    walk_legs['seq'] = np.arange(seq_start, seq_start + len(walk_legs))

    # Update the auto trips' DTAZ to parking zone (trip ends at parking facility)
    plan.loc[trip_indices, 'DTAZ'] = plan.loc[trip_indices, 'parking_zone'].fillna(plan.loc[trip_indices, 'DTAZ']).astype(int).values
    plan.loc[trip_indices, 'parking_trip_end'] = True

    # Concat and resort
    out = pd.concat([plan, walk_legs], ignore_index=True)
    out = out.sort_values(['person_id', 'seq']).reset_index(drop=True)
    return out


def run_stage7(
    plan: pd.DataFrame,
    persons: pd.DataFrame,
    location_table: pd.DataFrame = None,
    landuse: pd.DataFrame = None,
    skims_path: str = 'downcasted_matrices_base.omx',
    skim_lookup: dict = None,
    seed: int = 42,
    save_path: str = 'saved_data/stage7_parking_models.pkl',
    output_plan_path: str = 'saved_data/stage7_output.csv',
) -> dict:
    """
    Run Stage 7: Parking choice for auto trip legs.

    Parameters
    ----------
    plan : DataFrame
        Stage 6 output trip chain (trip + activity rows)
    persons : DataFrame
        Synthetic population
    location_table : DataFrame
        Parking facilities from build_location_table.py
    landuse : DataFrame
        Zone landuse (for density, employment)
    skims_path : str
        OMX skim file path
    skim_lookup : dict
        Pre-built skim lookup {(otaz, dtaz): {col: val}}
    seed : int
        Random seed
    save_path : str
        Where to save fitted parking model
    output_plan_path : str
        Where to save enhanced plan with parking

    Returns
    -------
    dict with keys: 'plan', 'parking_model', 'parking_summary'
    """
    rng = np.random.default_rng(seed)

    logger.info("Stage 7: Parking choice starting")

    # Load location table if not provided
    if location_table is None:
        loc_path = 'saved_data/location_table.csv'
        if os.path.exists(loc_path):
            location_table = pd.read_csv(loc_path)
            logger.info(f"Loaded location_table: {len(location_table)} facilities")
        else:
            logger.warning(f"Location table not found at {loc_path}; creating minimal")
            location_table = pd.DataFrame(columns=['location', 'zone', 'x', 'y', 'land_use',
                                                    'lu_area', 'avg_parking_cost'])

    # Load landuse if not provided
    if landuse is None:
        lu_path = 'saved_data/zone_landuse_overture.csv'
        if not os.path.exists(lu_path):
            lu_path = 'saved_data/zone_order_and_landuse.csv'
        if os.path.exists(lu_path):
            landuse = pd.read_csv(lu_path)
        else:
            logger.warning("Landuse not found; using empty")
            landuse = pd.DataFrame()

    # Load skims if not provided
    if skim_lookup is None:
        skims, _ = _open_skims(skims_path)
        skim_lookup = _skim_lookup(skims)

    # Estimate parking model from auto trips in plan
    ca_df = _build_parking_ca(plan, persons, location_table, landuse, skim_lookup)

    parking_model = None
    if not ca_df.empty and LARCH_AVAILABLE:
        parking_model = _estimate_parking_model(ca_df)

        # Save model
        if save_path and parking_model is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'model': parking_model,
                    'coef': {name: float(parking_model.pf.loc[name, 'value'])
                             for name in parking_model.pf.index},
                }, f)
            logger.info(f"Parking model saved to {save_path}")

    # Predict parking for all auto trips
    plan_with_parking = _predict_parking(plan, persons, parking_model, location_table, landuse, skim_lookup, rng)

    # Add walk legs from parking to activity
    plan_with_walk = _add_walk_legs(plan_with_parking, rng)

    # Summary
    auto_trips = plan_with_walk[
        (plan_with_walk['row_type'] == 'trip') &
        (plan_with_walk['mode'].isin([1, 2])) &
        (plan_with_walk['purpose'] != 'home')
    ]
    parking_summary = {
        'total_auto_trips': len(auto_trips),
        'onstreet_share': (auto_trips['parking_type'] == PARKING_ONSTREET).mean() if 'parking_type' in auto_trips.columns else 0,
        'garage_share': (auto_trips['parking_type'] == PARKING_GARAGE).mean() if 'parking_type' in auto_trips.columns else 0,
        'avg_parking_cost': auto_trips['parking_cost'].mean() if 'parking_cost' in auto_trips.columns else 0,
        'avg_walk_time_min': auto_trips['parking_walk_time_min'].mean() if 'parking_walk_time_min' in auto_trips.columns else 0,
    }
    logger.info(f"Parking summary: {parking_summary}")

    # Save enhanced plan
    if output_plan_path:
        os.makedirs(os.path.dirname(output_plan_path), exist_ok=True)
        plan_with_walk.to_csv(output_plan_path, index=False)
        logger.info(f"Stage 7 plan saved to {output_plan_path}")

    return {
        'plan': plan_with_walk,
        'parking_model': parking_model,
        'parking_summary': parking_summary,
    }


def _skim_lookup(skims) -> dict:
    """Build skim lookup dict from OMX."""
    lookup = {}
    try:
        mapping = dict(skims.mapping(skims.listMappings()[0]))
        zone_ids = sorted(mapping, key=lambda k: mapping[k])
        for name in skims.listMatrices():
            mat = np.array(skims[name])
            break
        N = len(zone_ids)
        for i, o in enumerate(zone_ids):
            for j, d in enumerate(zone_ids):
                key = (int(o), int(d))
                if key not in lookup:
                    lookup[key] = {}
                for name in skims.listMatrices():
                    mat = np.array(skims[name])
                    lookup[key][name] = mat[i, j]
    except Exception as e:
        logger.warning(f"Skim lookup build failed: {e}")
    return lookup


# ── Pipeline integration ──────────────────────────────────────────────────────

def stage7(plan: pd.DataFrame, persons: pd.DataFrame, landuse: pd.DataFrame,
           location_table: pd.DataFrame = None, skims_path: str = None,
           skim_lookup: dict = None, output_dir: str = 'saved_data') -> pd.DataFrame:
    """
    Pipeline wrapper for Stage 7 (used by pipeline_logger.py).

    Returns enhanced plan DataFrame.
    """
    if location_table is None:
        loc_path = os.path.join(output_dir, 'location_table.csv')
        if os.path.exists(loc_path):
            location_table = pd.read_csv(loc_path)

    if skims_path is None:
        skims_path = os.path.join(output_dir, '..', 'downcasted_matrices_base.omx')

    result = run_stage7(
        plan=plan,
        persons=persons,
        location_table=location_table,
        landuse=landuse,
        skims_path=skims_path,
        skim_lookup=skim_lookup,
        output_plan_path=os.path.join(output_dir, 'stage7_output.csv'),
        save_path=os.path.join(output_dir, 'stage7_parking_models.pkl'),
    )
    return result['plan']


if __name__ == '__main__':
    # Test run
    logging.basicConfig(level=logging.INFO)
    print("Stage 7 parking choice module loaded")
    print("Usage: from stage7_parking_choice import run_stage7, stage7")