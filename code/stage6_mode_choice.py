"""
stage6_mode_choice.py
======================
Stage 6 of the activity-based model pipeline.

Predicts the travel mode for each activity/tour using a Nested Logit
model estimated from HHTS, using your actual OD skim and person columns.

Pipeline position
-----------------
    Stage 1: n_{purpose}_out
    Stage 2: start_of_day, end_of_day, B
    Stage 3: predicted_duration_min per activity
    Stage 4: STARTIME, ENDTIME, time_period
    Stage 5: DTAZ, tt_avg, tt_am, tt_pm, dist, distwalk, distbike, crash
        → Stage 6: mode choice             ← YOU ARE HERE
        → Output: complete activity plan

Nest structure
--------------
    Root
    ├── Motorised
    │   ├── DA       (drive alone / car driver)
    │   ├── SR       (shared ride / car passenger)
    │   └── Transit  (bus, train, light rail, ferry)
    └── Non-motorised
        ├── Walk
        └── Bike

    A separate model is fitted per purpose (work, edu, social, …).

Utility specification per mode
-------------------------------
    DA      : beta_tt_da   * tt_avg
            + beta_dist_da * log_dist
            + beta_cost    * cost_proxy     (dist * fuel_rate)
            + beta_veh     * (vehrd >= 1)
            + beta_crash   * crash

    SR      : beta_tt_sr   * tt_avg
            + beta_dist_sr * log_dist
            + beta_crash   * crash
            + ASC_SR

    Transit : beta_tt_tr   * tt_avg * 1.5  (in-vehicle time penalty)
            + beta_dist_tr * log_dist
            + beta_acc     * accessibility  (zone transit access)
            + ASC_Transit

    Walk    : beta_distwalk * distwalk
            + beta_veh_walk * (vehrd == 0)  (captive walk bonus)
            + ASC_Walk

    Bike    : beta_distbike * distbike
            + beta_veh_bike * (vehrd == 0)
            + ASC_Bike

OD Skim columns used
---------------------
    tt_avg, tt_am, tt_pm    — auto travel time
    dist                    — road distance
    distwalk                — walk distance
    distbike                — bike distance
    crash                   — crash risk

Person columns used
-------------------
    vehrd    — vehicles in household
    sexp     — sex
    WORKSTAT — work status
    STUDYING — study type
    incp     — income band

Usage
-----
    from stage6_mode_choice import run_stage6

    sc           = Synth_Connect()
    out          = sc.process_survey(save=False)
    synth_s5     = pd.read_csv('saved_data/stage5_output.csv')
    landuse      = sc.get_land()

    activity_plan, models = run_stage6(
        out["trips"], out["persons"], synth_s5, landuse
    )
    activity_plan.to_csv('saved_data/stage6_output.csv', index=False)
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
from typing import Optional

from larch_regularization import elasticnet_objective

# ── JAX: XLA-JIT utility matrix computation (CPU-only, no GPU required) ──
import os as _os
_os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
_os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=1')

try:
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except Exception:
    jnp = None
    _JAX_AVAILABLE = False

# ── Larch: nested logit estimation with JAX/numba backend ────────────────
try:
    import larch as lx
    from larch import P, X
    _LARCH_AVAILABLE = True
except ImportError:
    lx = None
    P = X = None
    _LARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

if not _LARCH_AVAILABLE:
    logger.warning(
        "stage6: larch not installed — mode models will use scipy L-BFGS-B fallback. "
        "Install larch (pip install larch5) for JAX/numba-accelerated nested logit."
    )


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PURPOSES = [
    'work', 'edu', 'social', 'escort',
    'shopping', 'othmaint', 'othdiscr', 'eat',
]

_S6_PURPOSE_PARAM_NAMES = {
    'work': [
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'inc_da', 'wrk_da',
        'lic_da', 'ft_da', 'ind_da',
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'inc_tr', 'pt_tr', 'pub_tr',
        'asc_wk', 'ldw', 'nv_wk', 'age_wk',
        'asc_bk', 'ldb', 'nv_bk', 'age_bk',
        'mu_motorised', 'mu_nonmotorised',
    ],
    'edu': [
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'lic_da',
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'stu_tr',
        'asc_wk', 'ldw', 'nv_wk', 'stu_wk',
        'asc_bk', 'ldb', 'nv_bk', 'yng_bk',
        'mu_motorised', 'mu_nonmotorised',
    ],
    'escort': [
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'lic_da',
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'fem_tr',
        'asc_wk', 'ldw', 'nv_wk', 'fem_wk',
        'asc_bk', 'ldb', 'nv_bk',
        'mu_motorised', 'mu_nonmotorised',
    ],
    'shopping': [
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'inc_da', 'lic_da',
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'fem_tr', 'inc_tr',
        'asc_wk', 'ldw', 'nv_wk', 'age_wk', 'fem_wk',
        'asc_bk', 'ldb', 'nv_bk', 'age_bk',
        'mu_motorised', 'mu_nonmotorised',
    ],
    'eat': [
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'inc_da', 'lic_da',
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'fem_tr',
        'asc_wk', 'ldw', 'nv_wk', 'fem_wk',
        'asc_bk', 'ldb', 'nv_bk', 'yng_bk',
        'mu_motorised', 'mu_nonmotorised',
    ],
    'social': [
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'inc_da', 'lic_da',
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'fem_tr', 'inc_tr',
        'asc_wk', 'ldw', 'nv_wk', 'age_wk', 'fem_wk',
        'asc_bk', 'ldb', 'nv_bk', 'age_bk', 'yng_bk',
        'mu_motorised', 'mu_nonmotorised',
    ],
    'othmaint': [
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'inc_da', 'lic_da',
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'fem_tr', 'inc_tr',
        'asc_wk', 'ldw', 'nv_wk', 'age_wk',
        'asc_bk', 'ldb', 'nv_bk', 'age_bk',
        'mu_motorised', 'mu_nonmotorised',
    ],
    'othdiscr': [
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'inc_da', 'lic_da',
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'fem_tr',
        'asc_wk', 'ldw', 'nv_wk', 'age_wk', 'fem_wk',
        'asc_bk', 'ldb', 'nv_bk', 'age_bk', 'yng_bk',
        'mu_motorised', 'mu_nonmotorised',
    ],
}

# Mode labels
MODES      = ['DA', 'SR', 'Transit', 'Walk', 'Bike']
MODE_IDX   = {m: i for i, m in enumerate(MODES)}

# Nest membership: mode → nest
NESTS = {
    'DA':      'Motorised',
    'SR':      'Motorised',
    'Transit': 'Motorised',
    'Walk':    'NonMotorised',
    'Bike':    'NonMotorised',
}
NEST_MEMBERS = {
    'Motorised':    ['DA', 'SR', 'Transit'],
    'NonMotorised': ['Walk', 'Bike'],
}

# Fuel cost proxy per km (AUD cents per km)
FUEL_RATE = 0.12   # ~12 cents/km

# Person feature columns — extended with age5p and industry/licence for heterogeneity spec
#TODO I WANT workstat, BUT WHEN WORKSTAT I WANT IT POPULATED INTO THE MODEL BROKEN DOWN AS CATEGOGICAL HOW TO
PERSON_FEATURES = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp', 'age5p',
                   'indp', 'CARLICENCE']

# ── Frozen probability cell keys ────────────────────────────────────────────
# Cell-level mode probabilities are pre-computed from HHTS observations and
# stored in a lookup table.  When predicting for the synthetic population,
# each person is mapped to a cell using these same attributes, and the frozen
# probabilities are applied directly — no MC noise, reproducible across runs.
S6_FROZEN_CELL_COLS = ['ptype', 'veh_bin', 'income_band', 'age_group']
S6_FROZEN_PROB_COLS = ['fp_DA', 'fp_SR', 'fp_Transit', 'fp_Walk', 'fp_Bike']

# Integer mode codes used by Larch (must be > 0, unique per alternative)
_LARCH_MODE_IDS = {'DA': 1, 'SR': 2, 'Walk': 3, 'Bike': 4, 'Transit': 5}

# HHTS mode column → standardised mode code
RAW_MODE_MAP = {
    'Car driver':                    'DA',
    'Motorcycle driver':             'DA',
    'Truck driver':                  'DA',
    'Car passenger':                 'SR',
    'Motorcycle passenger':          'SR',
    'Truck passenger':               'SR',
    'Public Bus':                    'Transit',
    'Public bus':                    'Transit',
    'Train':                         'Transit',
    'Light rail':                    'Transit',
    'Ferry':                         'Transit',
    'Charter/Courtesy/Other bus':    'Transit',
    'School bus (with route number)':'Transit',
    'School bus (private/chartered)':'Transit',
    'Walking':                       'Walk',
    'Bicycle':                       'Bike',
    'E-bike or E-scooter':           'Bike',
    'Uber / Other Ride Share':       'SR',
    'Taxi':                          'SR',
    'Mobility scooter':              'Walk',
    'Other method':                  'DA',
    # already-mapped codes
    'DA': 'DA', 'SR': 'SR', 'Transit': 'Transit',
    'Walk': 'Walk', 'Bike': 'Bike',
}


# ---------------------------------------------------------------------------
# LARCH HELPERS
# ---------------------------------------------------------------------------

def _lx_model_s6(dt):
    """
    Create a Larch Model from dt, trying JAX engine then numba.

    JAX gives exact gradients via XLA autodiff — faster and more accurate for
    the nested logit log-likelihood than finite-difference numba.  Falls back
    to numba if JAX is unavailable (Windows without CUDA, etc.).
    """
    if not _LARCH_AVAILABLE:
        raise RuntimeError("larch not installed")
    for engine in ('jax', 'numba'):
        try:
            m = lx.Model(dt, compute_engine=engine)
            logger.debug(f"_lx_model_s6: using compute_engine='{engine}'")
            return m
        except Exception as _e:
            logger.debug(f"_lx_model_s6: engine='{engine}' unavailable ({_e}), trying next")
    raise RuntimeError("No Larch compute engine available (tried jax, numba)")


def _build_larch_nl_mode_s6(df: pd.DataFrame, param_names: list, init_values: np.ndarray) -> tuple:
    """
    Build a Larch nested logit mode choice DataTree + Model.

    All utility terms reference pre-computed columns in *df* (IDCO format —
    one row per trip, all person and skim attributes already merged).

    Nest structure mirrors NestedLogitModeModel:
        Root (fixed μ=1)
        ├── Motorised  [DA, SR, Transit]  — param Mu_Motorised
        └── NonMotorised [Walk, Bike]     — param Mu_NonMotorised

    Returns
    -------
    (dt, model) on success.  Raises RuntimeError or ImportError on failure.
    """
    import xarray as xr

    df = df.copy()
    if 'trip_id' not in df.columns:
        df['trip_id'] = np.arange(len(df)).astype(str)
    df['trip_id'] = df['trip_id'].astype(str)
    trip_ids = df['trip_id'].tolist()
    alt_ids  = list(_LARCH_MODE_IDS.values())   # [1,2,3,4,5]

    # ── case × alternative root dataset ───────────────────────────────────
    ca_index = pd.MultiIndex.from_product(
        [trip_ids, alt_ids], names=['trip_id', '_altid_'])
    root_df  = pd.DataFrame({'_dummy': np.zeros(len(ca_index), dtype=np.float32)},
                             index=ca_index)
    root_ds  = lx.Dataset.construct.from_idca(root_df)

    # ── case-only (IDCO) dataset ───────────────────────────────────────────
    trip_ds  = lx.Dataset.construct.from_idco(df.set_index('trip_id'))

    dt = lx.DataTree(
        base  = root_ds,
        trip  = trip_ds,
        relationships = ["base.trip_id @ trip.trip_id"],
    )

    m = _lx_model_s6(dt)

    # chosen_mode_code: int code column added by NestedLogitModeModel.fit()
    m.choice_co_code = "trip.chosen_mode_code"

    # Availability: walk/bike blocked by pre-computed availability flags.
    # DA always available (vehicle utility term controls behaviour).
    m.availability_co_vars = {
        _LARCH_MODE_IDS['DA']:      1,
        _LARCH_MODE_IDS['SR']:      1,
        _LARCH_MODE_IDS['Walk']:    "walk_avail",
        _LARCH_MODE_IDS['Bike']:    "bike_avail",
        _LARCH_MODE_IDS['Transit']: 1,
    }

    # ── Utility expressions — match build_utility_matrix spec exactly ─────
    active_params = set(param_names)

    def _term(name, expr=None):
        if name not in active_params:
            return 0
        return P(name) if expr is None else P(name) * X(expr)

    m.utility_co[_LARCH_MODE_IDS['DA']] = (
        _term('tt_da', 'tt_h')
        + _term('ld_da', 'log_dist')
        + _term('cost', 'cost_proxy')
        + _term('veh_da', 'has_vehicle')
        + _term('crash', 'log_crash')
        + _term('inc_da', 'inc_hi')
        + _term('wrk_da', 'worker')
        + _term('lic_da', 'licence')
        + _term('ft_da', 'full_time')
        + _term('ind_da', 'indp_industrial')
    )
    m.utility_co[_LARCH_MODE_IDS['SR']] = (
        _term('asc_sr')
        + _term('tt_sr', 'tt_h')
        + _term('ld_sr', 'log_dist')
        + _term('nv_sr', 'no_vehicle')
    )
    m.utility_co[_LARCH_MODE_IDS['Transit']] = (
        _term('asc_tr')
        + _term('tt_tr', 'tt_h')
        + _term('ld_tr', 'log_dist')
        + _term('acc_tr', 'accessibility')
        + _term('nv_tr', 'no_vehicle')
        + _term('stu_tr', 'student')
        + _term('fem_tr', 'female')
        + _term('inc_tr', 'inc_lo')
        + _term('pt_tr', 'part_time')
        + _term('pub_tr', 'indp_public')
    )
    m.utility_co[_LARCH_MODE_IDS['Walk']] = (
        _term('asc_wk')
        + _term('ldw', 'ldw')
        + _term('nv_wk', 'no_vehicle')
        + _term('age_wk', 'older')
        + _term('fem_wk', 'female')
        + _term('stu_wk', 'student')
    )
    m.utility_co[_LARCH_MODE_IDS['Bike']] = (
        _term('asc_bk')
        + _term('ldb', 'ldb')
        + _term('nv_bk', 'no_vehicle')
        + _term('age_bk', 'older')
        + _term('yng_bk', 'young')
    )

    # ── Nested logit graph ────────────────────────────────────────────────
    # 2-level structure matching NestedLogitModeModel exactly so that the
    # nest parameter values can be mapped back after estimation.
    Mot = m.graph.new_node(
        parameter="Mu_Motorised",
        children=[_LARCH_MODE_IDS['DA'],
                  _LARCH_MODE_IDS['SR'],
                  _LARCH_MODE_IDS['Transit']],
    )
    NonMot = m.graph.new_node(
        parameter="Mu_NonMotorised",
        children=[_LARCH_MODE_IDS['Walk'],
                  _LARCH_MODE_IDS['Bike']],
    )
    m.graph.new_node(parameter="Mu_Root", children=[Mot, NonMot])

    m.set_value("Mu_Root",        1.0)
    m.set_value("Mu_Motorised",   0.8)
    m.set_value("Mu_NonMotorised",0.8)
    try:
        m.pf.loc["Mu_Root",         "holdfast"] = True
        for _mu in ["Mu_Motorised", "Mu_NonMotorised"]:
            m.pf.loc[_mu, "minimum"] = 0.05
            m.pf.loc[_mu, "maximum"] = 1.0
    except Exception:
        pass

    # ── Inject NestedLogitModeModel starting values ───────────────────────
    _pname_to_init = dict(zip(param_names, init_values))
    for _pn, _pv in _pname_to_init.items():
        # Map scipy param names → Larch param names (identical except nest params)
        _larch_pn = _pn
        if _pn == 'mu_motorised':    _larch_pn = 'Mu_Motorised'
        if _pn == 'mu_nonmotorised': _larch_pn = 'Mu_NonMotorised'
        try:
            if _larch_pn in m.pf.index:
                m.set_value(_larch_pn, float(_pv))
        except Exception:
            pass

    return dt, m


def _extract_larch_coef_to_scipy(m, param_names: list, init_values: np.ndarray) -> np.ndarray:
    """
    Extract Larch model coefficients back to NestedLogitModeModel.PARAM_NAMES
    order (scipy coef_ format).

    Unmapped parameters get their NestedLogitModeModel.INIT default so the
    scipy fallback path remains usable if extraction is incomplete.
    """
    try:
        pf = m.pf
        larch_vals = {name: float(pf.loc[name, 'value']) for name in pf.index}
    except Exception:
        return init_values.copy()

    coef = init_values.copy()
    for i, pname in enumerate(param_names):
        # Map nest params: scipy 'mu_motorised' ← Larch 'Mu_Motorised'
        if pname == 'mu_motorised':
            larch_key = 'Mu_Motorised'
        elif pname == 'mu_nonmotorised':
            larch_key = 'Mu_NonMotorised'
        else:
            larch_key = pname
        if larch_key in larch_vals and not np.isnan(larch_vals[larch_key]):
            coef[i] = larch_vals[larch_key]
    return coef


# ---------------------------------------------------------------------------
# FROZEN PROBABILITY HELPERS
# ---------------------------------------------------------------------------

def _assign_frozen_cells_s6(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build S6_FROZEN_CELL_COLS columns for a DataFrame.

    Mirrors the cell-assignment logic in _compute_frozen_probs_s6 so that
    synthetic-population rows map to the same cell keys as HHTS rows.
    """
    out  = pd.DataFrame(index=df.index)
    vehrd = pd.to_numeric(df.get('vehrd', pd.Series(1.0, index=df.index)),
                           errors='coerce').fillna(1.0)
    incp  = pd.to_numeric(df.get('incp',  pd.Series(8.0, index=df.index)),
                           errors='coerce').fillna(8.0)
    age5p = pd.to_numeric(df.get('age5p', pd.Series(5.0, index=df.index)),
                           errors='coerce').fillna(5.0)

    out['ptype'] = (df.get('ptype', pd.Series('NonWorker', index=df.index))
                    .fillna('NonWorker').astype(str))
    out['veh_bin']     = np.where(vehrd > 0, 'hasveh', 'noveh')
    out['income_band'] = pd.cut(
        incp, bins=[0, 4, 7, 10, 99],
        labels=['low', 'mid', 'high', 'very_high'], right=True
    ).astype(str).fillna('mid')
    out['age_group']   = pd.cut(
        age5p, bins=[0, 2, 4, 7, 99],
        labels=['young', 'middle', 'older', 'senior'], right=True
    ).astype(str).fillna('middle')
    return out


def _compute_frozen_probs_s6(
    df: pd.DataFrame,
    model: 'NestedLogitModeModel',
    purpose: str,
) -> 'pd.DataFrame | None':
    """
    Compute cell-level frozen mode probabilities from HHTS observations.

    Evaluates *model* at its fitted coefficients for every HHTS trip, then
    aggregates to person-type cells defined by S6_FROZEN_CELL_COLS.

    These cells are applied directly to the synthetic population:
      - Zero MC noise — reproducible across runs
      - Full transferability: cell keys are identical in HHTS and synth pop
      - Walk/bike/transit shares derived from HHTS variation, not just ASC defaults

    Returns
    -------
    DataFrame with S6_FROZEN_CELL_COLS + S6_FROZEN_PROB_COLS + ['n_obs'],
    or None on failure.
    """
    if model is None or model.coef_ is None:
        return None
    try:
        n  = len(df)
        P  = model.predict_proba(df)   # (N, 5) — [DA, SR, Transit, Walk, Bike]

        out = df.copy()
        out['fp_DA']      = P[:, MODE_IDX['DA']]
        out['fp_SR']      = P[:, MODE_IDX['SR']]
        out['fp_Transit'] = P[:, MODE_IDX['Transit']]
        out['fp_Walk']    = P[:, MODE_IDX['Walk']]
        out['fp_Bike']    = P[:, MODE_IDX['Bike']]

        # Build cell membership columns (mirrors _assign_frozen_cells_s6)
        vehrd = pd.to_numeric(df.get('vehrd', 1.0), errors='coerce').fillna(1.0)
        incp  = pd.to_numeric(df.get('incp',  8.0), errors='coerce').fillna(8.0)
        age5p = pd.to_numeric(df.get('age5p', 5.0), errors='coerce').fillna(5.0)

        out['ptype'] = df.get('ptype', pd.Series('NonWorker', index=df.index)).fillna('NonWorker').astype(str)
        out['veh_bin']     = np.where(vehrd > 0, 'hasveh', 'noveh')
        out['income_band'] = pd.cut(incp, bins=[0, 4, 7, 10, 99],
                                    labels=['low', 'mid', 'high', 'very_high'],
                                    right=True).astype(str).fillna('mid')
        out['age_group']   = pd.cut(age5p, bins=[0, 2, 4, 7, 99],
                                    labels=['young', 'middle', 'older', 'senior'],
                                    right=True).astype(str).fillna('middle')

        cell_df = (
            out.groupby(S6_FROZEN_CELL_COLS, observed=True)
            .agg(
                fp_DA=      ('fp_DA',      'mean'),
                fp_SR=      ('fp_SR',      'mean'),
                fp_Transit= ('fp_Transit', 'mean'),
                fp_Walk=    ('fp_Walk',    'mean'),
                fp_Bike=    ('fp_Bike',    'mean'),
                n_obs=      ('fp_DA',      'count'),
            )
            .reset_index()
        )

        # Renormalise rows (floating-point safety)
        prob_sum = cell_df[S6_FROZEN_PROB_COLS].sum(axis=1).clip(lower=1e-9)
        for col in S6_FROZEN_PROB_COLS:
            cell_df[col] = cell_df[col] / prob_sum

        logger.info(
            f"Stage 6 frozen probs [{purpose}]: {len(cell_df)} cells from {n} obs | "
            f"mean DA={cell_df['fp_DA'].mean():.1%} "
            f"Walk={cell_df['fp_Walk'].mean():.1%} "
            f"Transit={cell_df['fp_Transit'].mean():.1%} "
            f"Bike={cell_df['fp_Bike'].mean():.1%}"
        )
        return cell_df

    except Exception as e:
        logger.warning(f"_compute_frozen_probs_s6 [{purpose}] failed: {e}")
        import traceback; logger.debug(traceback.format_exc())
        return None


def _mode_probs_frozen_s6(
    frozen_probs: pd.DataFrame,
    df: pd.DataFrame,
) -> 'np.ndarray | None':
    """
    Look up frozen cell-level probabilities for rows in *df*.

    Unmatched rows fall back to ptype-level means, then global means.
    Returns (N, 5) array ordered [DA, SR, Transit, Walk, Bike], or None.
    """
    if frozen_probs is None or len(frozen_probs) == 0:
        return None
    try:
        cells  = _assign_frozen_cells_s6(df)
        merged = cells.merge(
            frozen_probs[S6_FROZEN_CELL_COLS + S6_FROZEN_PROB_COLS],
            on=S6_FROZEN_CELL_COLS, how='left',
        )

        # ── Fill unmatched cells ──────────────────────────────────────────
        unmatched = merged[S6_FROZEN_PROB_COLS].isna().any(axis=1)
        if unmatched.any():
            # Fallback 1: ptype-level mean
            ptype_mean = (frozen_probs
                          .groupby('ptype', observed=True)[S6_FROZEN_PROB_COLS]
                          .mean()
                          .reset_index())
            for idx in merged.index[unmatched]:
                pt = merged.loc[idx, 'ptype']
                row = ptype_mean[ptype_mean['ptype'] == pt]
                if not row.empty:
                    merged.loc[idx, S6_FROZEN_PROB_COLS] = row[S6_FROZEN_PROB_COLS].values[0]

            # Fallback 2: global mean for anything still unmatched
            still_un = merged[S6_FROZEN_PROB_COLS].isna().any(axis=1)
            if still_un.any():
                gmean = frozen_probs[S6_FROZEN_PROB_COLS].mean().values
                merged.loc[still_un, S6_FROZEN_PROB_COLS] = gmean

            n_filled = int(unmatched.sum())
            logger.debug(f"_mode_probs_frozen_s6: {n_filled} rows filled via fallback")

        probs  = merged[S6_FROZEN_PROB_COLS].values.astype(np.float64)
        totals = probs.sum(axis=1, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        probs  = probs / totals   # renormalise

        # Reorder columns to [DA, SR, Transit, Walk, Bike] = MODES order
        # S6_FROZEN_PROB_COLS: fp_DA, fp_SR, fp_Transit, fp_Walk, fp_Bike
        # MODES: DA, SR, Transit, Walk, Bike  — same order, no reorder needed
        return probs

    except Exception as e:
        logger.warning(f"_mode_probs_frozen_s6 failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 0.  PREPARE ESTIMATION DATA
# ---------------------------------------------------------------------------

def prepare_mode_data(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build trip-level estimation dataset.

    Each row = one trip with:
        chosen_mode (DA/SR/Transit/Walk/Bike)
        skim features (tt_avg, dist, distwalk, distbike, crash, …)
        person attributes (vehrd, sexp, WORKSTAT, …)
        purpose

    Parameters
    ----------
    trips   : from process_survey()["trips"]
    persons : from process_survey()["persons"]
    landuse : optional — used to attach accessibility to destination zones

    Returns
    -------
    DataFrame ready for mode model estimation.
    """
    df = trips.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # ── mode column ──────────────────────────────────────────────────────
    mode_col = next(
        (c for c in ['MAINMODE', 'mode', 'MODE'] if c in df.columns), None
    )
    if mode_col is None:
        raise ValueError("No mode column (MAINMODE/mode/MODE) in trips table.")
    df['chosen_mode'] = df[mode_col].map(RAW_MODE_MAP)
    df = df[df['chosen_mode'].isin(MODES)].copy()

    # ── purpose ──────────────────────────────────────────────────────────
    purp_col = next(
        (c for c in ['DESTPURP', 'purpose'] if c in df.columns), None
    )
    if purp_col:
        raw_to_code = {
            'Work (my workplace)': 'work', 'Work (other work reason)': 'work',
            'Direct Work Commute': 'work', 'Work Related': 'work',
            'Education': 'edu', 'Childcare or Kindergarten': 'edu',
            'Childcare or kindergarten': 'edu',
            'Social Visit': 'social', 'Recreation/leisure activity': 'social',
            'Recreation': 'social', 'Social': 'social',
            'Pick up or drop off a passenger': 'escort',
            'Pickup/Dropoff Someone': 'escort',
            'Just accompanying someone': 'escort',
            'Shopping': 'shopping',
            'Personal Business': 'othmaint', 'Personal business': 'othmaint',
            'Pickup/deliver something (not work-related)': 'othmaint',
            'Other place (please specify)': 'othdiscr',
            'Other Place': 'othdiscr', 'Refuel my vehicle': 'othdiscr',
            'Eat/drink': 'eat',
        }
        df['purpose'] = df[purp_col].map(raw_to_code).fillna(df[purp_col])
        df = df[df['purpose'].isin(PURPOSES)].copy()

    # ── skim features ────────────────────────────────────────────────────
    # standardise columns that may have various names
    skim_map = {
        'tt_avg':   ['tt_avg', 'SOVTOLL_TIME__AM', 'auto_time', 'TRAVTIME'],
        'tt_am':    ['tt_am', 'SOVTOLL_TIME__AM'],
        'dist':     ['dist', 'DIST', 'CUMDIST', 'TRIPDIST', 'distance_km', 'distance'],
        'distwalk': ['distwalk', 'DISTWALK', 'walk_dist'],
        'distbike': ['distbike', 'DISTBIKE', 'bike_dist'],
        'crash':    ['crash', 'CRASH', 'crash_index'],
    }
    for std, candidates in skim_map.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            df[std] = pd.to_numeric(df[found], errors='coerce').fillna(0)
        else:
            df[std] = 0.0

    df['log_dist']   = np.log1p(df['dist'])
    df['log_crash']  = np.log1p(df['crash'])
    df['tt_h']       = np.clip(df['tt_avg'] / 60.0, 0.01, 5.0)
    df['cost_proxy'] = df['dist'] * FUEL_RATE

    # ── zone TAZ ─────────────────────────────────────────────────────────
    if 'OTAZ' in df.columns:
        df['OTAZ'] = pd.to_numeric(df['OTAZ'], errors='coerce').fillna(0).astype(int)
    else:
        df['OTAZ'] = 0
    if 'DTAZ' in df.columns:
        df['DTAZ'] = pd.to_numeric(df['DTAZ'], errors='coerce').fillna(0).astype(int)
    else:
        df['DTAZ'] = 0

    # ── accessibility from landuse ────────────────────────────────────────
    if landuse is not None:
        lu = landuse.copy()
        zone_col = next(
            (c for c in ['ORDER', 'DTAZ', 'SA2_MAIN16', 'zone'] if c in lu.columns),
            None
        )
        if zone_col and zone_col != 'DTAZ':
            lu = lu.rename(columns={zone_col: 'DTAZ'})
        if 'DTAZ' in lu.columns:
            acc_cols = [c for c in ['FPSEMPN', 'MWTEMPN', 'OTHEMP',
                                     'RETEMPN', 'TOTHH', 'TOTPOP']
                        if c in lu.columns]
            if acc_cols:
                lu['accessibility'] = np.log1p(
                    lu[acc_cols].apply(pd.to_numeric, errors='coerce')
                    .fillna(0).sum(axis=1)
                )
            else:
                lu['accessibility'] = 0.0
            lu['DTAZ'] = pd.to_numeric(lu['DTAZ'], errors='coerce').astype(int)
            df = df.merge(lu[['DTAZ', 'accessibility']], on='DTAZ', how='left')
            df['accessibility'] = df['accessibility'].fillna(0)
        else:
            df['accessibility'] = 0.0
    else:
        df['accessibility'] = 0.0

    # ── STARTIME ──────────────────────────────────────────────────────────
    if 'STARTIME' in df.columns:
        df['start_hr'] = pd.to_numeric(df['STARTIME'], errors='coerce').fillna(480) / 60
    else:
        df['start_hr'] = 8.0

    # ── person attributes ─────────────────────────────────────────────────
    pers = persons.copy()
    if 'person_id' not in pers.columns and 'PERSID' in pers.columns:
        pers = pers.rename(columns={'PERSID': 'person_id'})

    try:
        from stage1_activity_frequency import assign_ptype_column
        pers = assign_ptype_column(pers)
    except ImportError:
        pers['ptype'] = 'NonWorker'

    keep = (['person_id', 'ptype'] +
            [c for c in PERSON_FEATURES if c in pers.columns])
    df = df.merge(pers[keep], on='person_id', how='left')

    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2,
                'WORKSTAT': 0, 'STUDYING': 0, 'incp': 8, 'age5p': 5,
                'indp': 20, 'CARLICENCE': 1}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    df['ptype'] = df['ptype'].fillna('NonWorker')
    df['has_vehicle'] = (df['vehrd'] >= 1).astype(float)
    df['no_vehicle']  = (df['vehrd'] == 0).astype(float)

    # Derived person-type dummies used in utility spec
    df['female']   = (df['sexp'] == 2).astype(float)
    df['young']    = (df['age5p'] <= 5).astype(float)    # under ~30
    df['older']    = (df['age5p'] >= 10).astype(float)   # 55+
    df['worker']   = np.clip(df['WORKSTAT'], 0, 1)
    df['student']  = (df['STUDYING'] > 0).astype(float)
    df['inc_hi']   = np.clip(df['incp'] - 8, 0, None).astype(float)
    df['inc_lo']   = np.clip(8 - df['incp'], 0, None).astype(float)

    # Log-transformed distances (matches build_utility_matrix ldw/ldb)
    df['ldw'] = np.log1p(df['distwalk'])
    df['ldb'] = np.log1p(df['distbike'])

    # ── Extended socio-demographics ───────────────────────────────────────
    # Car licence: direct prerequisite for DA in logit utility sense
    df['licence']        = df['CARLICENCE'].clip(0, 1).astype(float)

    # Industry group dummies: ANZSIC 1-digit code stored in 'indp'
    # Industrial workers (mining, manufacturing, utilities, construction)
    # are strongly car-dependent due to remote sites and shift work.
    _indp = df['indp'].astype(float)
    df['indp_industrial'] = _indp.isin([2, 3, 4, 5]).astype(float)
    # Public-sector / health / education workers: typically transit-accessible
    # city/suburban destinations, school-zone parking restrictions, etc.
    df['indp_public']     = _indp.isin([15, 16, 17]).astype(float)

    # Worker type: full-time vs part-time (QTS work_map: 1=FT, 2=PT)
    # Full-time workers face stricter schedule constraints → more DA
    df['full_time'] = (df['WORKSTAT'] == 1).astype(float)
    df['part_time'] = (df['WORKSTAT'] == 2).astype(float)

    mode_dist = df['chosen_mode'].value_counts().to_dict()
    mode_pct  = {m: f"{v/len(df):.1%}" for m, v in mode_dist.items()}
    logger.info(
        f"Mode estimation data: {len(df)} trips | mode distribution: {mode_dist} "
        f"({mode_pct})"
    )
    logger.info(
        f"  Person mix: noveh={df['no_vehicle'].mean():.1%}, "
        f"female={df['female'].mean():.1%}, "
        f"worker={df['worker'].mean():.1%}, "
        f"full_time={df['full_time'].mean():.1%}, "
        f"part_time={df['part_time'].mean():.1%}, "
        f"industrial={df['indp_industrial'].mean():.1%}, "
        f"public={df['indp_public'].mean():.1%}, "
        f"licence={df['licence'].mean():.1%}, "
        f"student={df['student'].mean():.1%}, "
        f"young={df['young'].mean():.1%}, "
        f"older={df['older'].mean():.1%}, "
        f"mean_incp={df['incp'].mean():.1f}"
    )
    return df


# ---------------------------------------------------------------------------
# 1.  UTILITY BUILDER
# ---------------------------------------------------------------------------
# Utility specification principles (ActivitySim + ATRF2022 pattern):
#
#  • Travel time / distance enter as log-transforms for walk/bike to capture
#    diminishing deterrence at longer distances (matching empirical HHTS shapes).
#  • Person heterogeneity: vehicle ownership, age group, income band, employment
#    status and gender all interact with mode utilities.
#  • Crash risk enters as log(crash+1) — aligns with ATRF2022 safety term.
#  • Transit accessibility (destination employment density) boosts transit.
#  • No-vehicle household flag is the single strongest predictor of non-DA modes;
#    it enters all non-DA utilities to represent captive behaviour.
# ---------------------------------------------------------------------------

def build_utility_matrix(
    df: pd.DataFrame,
    params: np.ndarray,
    param_names: list,
) -> np.ndarray:
    """
    Compute utility for each mode for every observation.

    Returns V shape (N, 5) — one column per mode [DA, SR, Transit, Walk, Bike]

    Uses JAX (XLA JIT) when available for faster gradient computation; falls
    back to numpy.  All array ops use the same xp namespace.

    Richer specification vs original:
    • Walk/bike use log(dist+1) instead of raw km — better empirical fit
    • Age bands, income, employment status, gender heterogeneity per mode
    • Crash risk (log) enters motorised modes — ATRF2022 safety term
    • Transit accessibility from destination zone
    """
    xp  = jnp if _JAX_AVAILABLE else np
    N   = len(df)
    V   = np.zeros((N, len(MODES)))   # keep as numpy for indexing compat
    p   = dict(zip(param_names, params))

    def _get(name, default=0.0):
        return p.get(name, default)

    def _col(colname, fallback=0.0):
        if colname in df.columns:
            return df[colname].fillna(fallback).values.astype(float)
        return np.full(N, fallback)

    tt      = _col('tt_avg')          # auto travel time (min)
    tt_h    = xp.clip(xp.asarray(tt) / 60.0, 0.01, 5.0)   # hours
    ld      = xp.asarray(_col('log_dist'))        # log(dist+1) km
    dist    = _col('dist')
    dw      = _col('distwalk')        # raw walk km
    db      = _col('distbike')        # raw bike km
    # Log-transform walk/bike distances — empirically better fit than linear
    ldw     = xp.log1p(xp.asarray(dw))   # log(distwalk + 1)
    ldb     = xp.log1p(xp.asarray(db))   # log(distbike + 1)
    crash   = xp.asarray(_col('crash'))
    log_crash = xp.log1p(crash)           # ATRF2022 safety term
    acc     = xp.asarray(_col('accessibility'))
    hv      = xp.asarray(_col('has_vehicle'))   # 1 if HH has ≥1 vehicle
    nv      = xp.asarray(_col('no_vehicle'))    # 1 if HH has 0 vehicles
    cost    = xp.asarray(dist) * FUEL_RATE

    # Person heterogeneity
    incp    = xp.asarray(_col('incp',     8.0))   # income band 1-14
    age5p   = xp.asarray(_col('age5p',    4.0))   # ABS 5-yr age band
    sexp    = _col('sexp',     1.0)                # 1=male 2=female
    workst  = xp.asarray(_col('WORKSTAT', 0.0))   # 1/2=employed
    study   = xp.asarray(_col('STUDYING', 0.0))

    female   = xp.asarray((sexp == 2).astype(float))
    young    = (age5p <= 5).astype(float)   # ~under 30
    older    = (age5p >= 10).astype(float)  # ~55+
    worker   = xp.clip(workst, 0, 1)
    student  = (study > 0).astype(float)
    inc_hi   = xp.clip(incp - 8, 0, None)  # income above median
    inc_lo   = xp.clip(8 - incp, 0, None)  # income below median

    # Extended socio-demographics — worker type and industry
    licence       = xp.asarray(_col('licence',       1.0))   # has car licence
    full_time     = xp.asarray(_col('full_time',     0.0))   # full-time worker
    part_time     = xp.asarray(_col('part_time',     0.0))   # part-time worker
    indp_indust   = xp.asarray(_col('indp_industrial', 0.0)) # trade/industrial worker
    indp_public   = xp.asarray(_col('indp_public',   0.0))   # public/health/edu worker

    # ── DA (reference, ASC=0) ────────────────────────────────────────────
    V[:, MODE_IDX['DA']] = np.asarray(
        _get('tt_da')    * tt_h
        + _get('ld_da')  * ld
        + _get('cost')   * cost
        + _get('veh_da') * hv
        + _get('crash')  * log_crash   # safety deterrence (expect negative)
        + _get('inc_da') * inc_hi      # high income → more DA
        + _get('wrk_da') * worker      # workers → more DA
        + _get('lic_da') * licence     # car licence → strong DA predictor
        + _get('ft_da')  * full_time   # full-time workers commute by car more
        + _get('ind_da') * indp_indust # industrial/trade workers: car-captive
    )

    # ── SR (shared-ride) ────────────────────────────────────────────────
    V[:, MODE_IDX['SR']] = np.asarray(
        _get('asc_sr')
        + _get('tt_sr')   * tt_h
        + _get('ld_sr')   * ld
        + _get('crash')   * log_crash
        + _get('nv_sr')   * nv         # no-vehicle → more SR (passenger)
    )

    # ── Transit ─────────────────────────────────────────────────────────
    V[:, MODE_IDX['Transit']] = np.asarray(
        _get('asc_tr')
        + _get('tt_tr')   * tt_h * 1.5  # in-vehicle time penalty
        + _get('ld_tr')   * ld
        + _get('acc_tr')  * acc          # transit accessibility at destination
        + _get('nv_tr')   * nv           # captive transit users (no vehicle)
        + _get('stu_tr')  * student      # students → transit
        + _get('fem_tr')  * female       # women slightly more transit
        + _get('inc_tr')  * inc_lo       # lower income → transit
        + _get('pt_tr')   * part_time    # part-time workers: flexible → transit
        + _get('pub_tr')  * indp_public  # public/health/edu: transit-accessible sites
    )

    # ── Walk ─────────────────────────────────────────────────────────────
    # Log-distance has better empirical fit than linear distance for walk.
    V[:, MODE_IDX['Walk']] = np.asarray(
        _get('asc_wk')
        + _get('ldw')     * ldw          # log(distwalk+1)
        + _get('nv_wk')   * nv           # no-vehicle captive
        + _get('age_wk')  * older        # older persons less likely to walk
        + _get('fem_wk')  * female       # gender shift
        + _get('stu_wk')  * student      # students more walk
    )

    # ── Bike ─────────────────────────────────────────────────────────────
    V[:, MODE_IDX['Bike']] = np.asarray(
        _get('asc_bk')
        + _get('ldb')     * ldb          # log(distbike+1)
        + _get('nv_bk')   * nv           # no-vehicle captive
        + _get('age_bk')  * older        # older persons less likely to bike
        + _get('yng_bk')  * young        # young people more likely to bike
    )

    # Convert from JAX array to numpy for downstream scipy/indexing ops
    return np.asarray(V)


def _s6_param_names_for_purpose(purpose: str) -> list:
    return list(_S6_PURPOSE_PARAM_NAMES.get(purpose, NestedLogitModeModel.PARAM_NAMES))


def _s6_param_spec(param_names: list):
    bounds_map = dict(zip(NestedLogitModeModel.PARAM_NAMES, NestedLogitModeModel.BOUNDS))
    init_map = dict(zip(NestedLogitModeModel.PARAM_NAMES, NestedLogitModeModel.INIT))
    return (
        np.array([init_map[name] for name in param_names], dtype=float),
        [bounds_map[name] for name in param_names],
    )


# ---------------------------------------------------------------------------
# 2.  NESTED LOGIT MODEL
# ---------------------------------------------------------------------------

class NestedLogitModeModel:
    """
    Two-level nested logit for mode choice.

    Nests: Motorised [DA, SR, Transit] and NonMotorised [Walk, Bike]
    Logsum (nest) parameters mu_m, mu_nm estimated jointly with utilities.

    Estimation via L-BFGS-B on the full nested LL.
    """

    # ── Parameter names (must match build_utility_matrix keys) ──────────────
    PARAM_NAMES = [
        # DA
        'tt_da', 'ld_da', 'cost', 'veh_da', 'crash', 'inc_da', 'wrk_da',
        'lic_da', 'ft_da', 'ind_da',
        # SR
        'asc_sr', 'tt_sr', 'ld_sr', 'nv_sr',
        # Transit
        'asc_tr', 'tt_tr', 'ld_tr', 'acc_tr', 'nv_tr', 'stu_tr', 'fem_tr', 'inc_tr',
        'pt_tr', 'pub_tr',
        # Walk
        'asc_wk', 'ldw', 'nv_wk', 'age_wk', 'fem_wk', 'stu_wk',
        # Bike
        'asc_bk', 'ldb', 'nv_bk', 'age_bk', 'yng_bk',
        # Nest logsum params (must be in (0, 1])
        'mu_motorised', 'mu_nonmotorised',
    ]

    # ── Bounds ────────────────────────────────────────────────────────────
    BOUNDS = [
        # DA
        (-3.0,  0.0),   # tt_da   (negative = time hurts)
        (-3.0,  0.0),   # ld_da
        (-2.0,  0.0),   # cost
        ( 0.0,  3.0),   # veh_da  (vehicle ownership → DA)
        (-2.0,  0.0),   # crash   (safety deterrence)
        ( 0.0,  0.5),   # inc_da  (high income → DA)
        ( 0.0,  1.0),   # wrk_da  (workers → more DA)
        ( 0.0,  2.0),   # lic_da  (licence → DA, should be positive)
        ( 0.0,  1.0),   # ft_da   (full-time worker → more DA)
        ( 0.0,  1.5),   # ind_da  (industrial/trade worker → car-captive)
        # SR
        (-5.0,  2.0),   # asc_sr
        (-3.0,  0.0),   # tt_sr
        (-3.0,  0.0),   # ld_sr
        ( 0.0,  3.0),   # nv_sr   (no vehicle → passenger)
        # Transit
        (-5.0,  2.0),   # asc_tr
        (-3.0,  0.0),   # tt_tr
        (-3.0,  0.0),   # ld_tr
        (-1.0,  3.0),   # acc_tr  (accessibility → transit)
        ( 0.0,  3.0),   # nv_tr   (no vehicle → transit captive)
        ( 0.0,  2.0),   # stu_tr  (student → transit)
        (-1.0,  2.0),   # fem_tr  (female → transit)
        ( 0.0,  0.5),   # inc_tr  (low income → transit)
        ( 0.0,  1.0),   # pt_tr   (part-time worker → transit)
        ( 0.0,  1.5),   # pub_tr  (public/health worker → transit)
        # Walk
        (-5.0,  2.0),   # asc_wk
        (-3.0,  0.0),   # ldw     (log walk dist — negative = longer hurts)
        ( 0.0,  3.0),   # nv_wk   (no vehicle → walk)
        (-2.0,  0.0),   # age_wk  (older → less walk)
        (-1.0,  2.0),   # fem_wk  (female → walk)
        ( 0.0,  2.0),   # stu_wk  (student → walk)
        # Bike
        (-5.0,  2.0),   # asc_bk
        (-3.0,  0.0),   # ldb     (log bike dist)
        ( 0.0,  3.0),   # nv_bk   (no vehicle → bike)
        (-2.0,  0.0),   # age_bk  (older → less bike)
        ( 0.0,  2.0),   # yng_bk  (young → more bike)
        # Nests
        ( 0.1,  1.0),   # mu_motorised
        ( 0.1,  1.0),   # mu_nonmotorised
    ]

    # ── Starting values ───────────────────────────────────────────────────
    # Calibrated to approximate SEQ HHTS mode shares:
    #   DA~55%, SR~15%, Transit~10%, Walk~12%, Bike~8%
    # Log-distance coefficients reflect empirical AU walk/bike deterrence.
    INIT = np.array([
        # DA
        -0.06, -0.12, -0.01,  0.50, -0.10,  0.02,  0.20,
         0.40,  0.15,  0.30,  # lic_da, ft_da, ind_da
        # SR
        -0.40, -0.06, -0.10,  0.50,
        # Transit
        -0.30, -0.06, -0.08,  0.25,  1.20,  0.40,  0.20,  0.02,
         0.15,  0.25,          # pt_tr, pub_tr
        # Walk  — raised ASC from -0.60 → -0.20 to allow prediction of walk
        -0.20, -0.80,  0.60, -0.15,  0.10,  0.30,
        # Bike  — raised ASC from -0.70 → -0.40 to allow prediction of bike
        -0.40, -0.60,  0.40, -0.20,  0.40,
        # Nests
         0.70,  0.70,
    ])

    def __init__(self):
        self.coef_         = None
        self.converged_    = False
        self.purpose_      = None
        self.param_names_  = list(self.PARAM_NAMES)
        self.init_         = self.INIT.copy()
        self.bounds_       = list(self.BOUNDS)
        self.n_obs_        = 0
        self.ll_null_      = None
        self.ll_final_     = None
        self.rho2_         = None
        self.frozen_probs_ = None    # cell-level table for synth pop transfer
        self._estimation_engine = 'scipy'   # 'larch_jax', 'larch_numba', 'scipy'
        # Standard errors and significance (None until fit is called)
        self.se_           = None
        self.tstat_        = None
        self.pval_         = None
        self._larch_pf     = None    # Larch parameter frame (SE/tstat from Larch)

    def _configure_for_purpose(self, purpose: str):
        self.param_names_ = _s6_param_names_for_purpose(purpose)
        self.init_, self.bounds_ = _s6_param_spec(self.param_names_)

    def _nested_proba(self, V: np.ndarray,
                      mu_m: float, mu_nm: float) -> np.ndarray:
        """
        Compute nested logit choice probabilities.

        V : (N, 5) utilities [DA, SR, Transit, Walk, Bike]
        Returns P : (N, 5)
        """
        mi  = MODE_IDX
        N   = len(V)

        # ── within-nest probabilities ──────────────────────────────────
        # Motorised nest: DA, SR, Transit
        V_mot = V[:, [mi['DA'], mi['SR'], mi['Transit']]] / mu_m
        V_mot = V_mot - V_mot.max(axis=1, keepdims=True)
        E_mot = np.exp(V_mot)
        S_mot = E_mot.sum(axis=1, keepdims=True)
        P_cond_mot = E_mot / S_mot                             # (N, 3)

        # NonMotorised nest: Walk, Bike
        V_nm  = V[:, [mi['Walk'], mi['Bike']]] / mu_nm
        V_nm  = V_nm - V_nm.max(axis=1, keepdims=True)
        E_nm  = np.exp(V_nm)
        S_nm  = E_nm.sum(axis=1, keepdims=True)
        P_cond_nm  = E_nm / S_nm                               # (N, 2)

        # ── nest log-sums ─────────────────────────────────────────────
        LS_mot = mu_m  * np.log(S_mot.squeeze() + 1e-12)      # (N,)
        LS_nm  = mu_nm * np.log(S_nm.squeeze()  + 1e-12)

        # ── between-nest probabilities ────────────────────────────────
        LS = np.column_stack([LS_mot, LS_nm])
        LS = LS - LS.max(axis=1, keepdims=True)
        E_nest = np.exp(LS)
        P_nest = E_nest / E_nest.sum(axis=1, keepdims=True)   # (N, 2)

        # ── joint probabilities ────────────────────────────────────────
        P = np.zeros((N, 5))
        P[:, mi['DA']]      = P_nest[:, 0] * P_cond_mot[:, 0]
        P[:, mi['SR']]      = P_nest[:, 0] * P_cond_mot[:, 1]
        P[:, mi['Transit']] = P_nest[:, 0] * P_cond_mot[:, 2]
        P[:, mi['Walk']]    = P_nest[:, 1] * P_cond_nm[:, 0]
        P[:, mi['Bike']]    = P_nest[:, 1] * P_cond_nm[:, 1]

        return np.clip(P, 1e-12, 1.0)

    def _neg_ll(self, params: np.ndarray,
                df: pd.DataFrame,
                y: np.ndarray) -> float:
        mu_m  = float(params[-2])
        mu_nm = float(params[-1])
        V     = build_utility_matrix(df, params[:-2], self.param_names_[:-2])
        P     = self._nested_proba(V, mu_m, mu_nm)
        ll    = np.sum(np.log(P[np.arange(len(y)), y]))
        return -ll

    def fit(self, df: pd.DataFrame, purpose: str):
        """
        Estimate nested logit coefficients.

        Estimation order:
          1. Larch NL with JAX engine (exact gradients, fastest)
          2. Larch NL with numba engine (finite-difference Hessian, robust)
          3. scipy L-BFGS-B (always available, no Larch dependency)

        After estimation, frozen cell-level probabilities are computed
        and stored in self.frozen_probs_ for synthetic population transfer.
        """
        self.purpose_ = purpose
        self._configure_for_purpose(purpose)
        self.n_obs_   = len(df)

        y = np.array([MODE_IDX[m] for m in df['chosen_mode']])

        # ── Pre-estimation diagnostics ────────────────────────────────────
        obs_dist = df['chosen_mode'].value_counts().to_dict()
        self.obs_shares_ = {m: (obs_dist.get(m, 0) / max(len(df), 1)) for m in MODES}
        obs_pct  = {m: f"{obs_dist.get(m,0)/len(df):.1%}" for m in MODES}
        logger.info(f"  mode/{purpose} HHTS observed: {obs_pct}")
        logger.info(
            f"  mode/{purpose} data: n={len(df)} | "
            f"noveh={float((df.get('no_vehicle',pd.Series(0,index=df.index))>0).mean()):.1%} | "
            f"walk_avail={int(df.get('walk_avail',pd.Series(1,index=df.index)).sum())} | "
            f"bike_avail={int(df.get('bike_avail',pd.Series(1,index=df.index)).sum())}"
        )

        # null LL (market shares, not equal shares — better benchmark)
        market_shares = np.array([
            (y == i).sum() / len(y) for i in range(len(MODES))
        ])
        market_shares = np.clip(market_shares, 1e-9, 1.0)
        self.ll_null_ = float(np.sum(np.log(market_shares[y])))
        self.ll_eq_   = len(y) * np.log(1.0 / len(MODES))   # equal-share LL

        # ── Attach walk/bike availability for Larch ───────────────────────
        df = df.copy()
        if 'walk_avail' not in df.columns:
            if 'distwalk' in df.columns:
                df['walk_avail'] = (df['distwalk'] < 6.0).astype(int)
            else:
                df['walk_avail'] = 1
        if 'bike_avail' not in df.columns:
            if 'distbike' in df.columns:
                df['bike_avail'] = (df['distbike'] < 25.0).astype(int)
            else:
                df['bike_avail'] = 1

        # ── Integer mode code column needed by Larch choice_co_code ──────
        df['chosen_mode_code'] = df['chosen_mode'].map(
            {m: _LARCH_MODE_IDS[m] for m in MODES}
        ).fillna(1).astype(int)

        # ── 1. Try Larch estimation ───────────────────────────────────────
        _larch_succeeded = False
        if _LARCH_AVAILABLE:
            try:
                _dt, _lm = _build_larch_nl_mode_s6(df, self.param_names_, self.init_)
                try:
                    _lm.set_cap(10)
                    _lm.doctor(repair_ch_av="+")
                    _lm.doctor(repair_nan_utility=True)
                except Exception:
                    pass
                # Elastic-net (L1+L2) penalty on the objective, mirroring
                # metacountregressor's penalized-likelihood estimation - see
                # larch_regularization.py. Restored before LL is read back
                # so self.ll_final_ reflects the true unpenalized likelihood.
                _ennet_lm = elasticnet_objective(_lm)
                _ennet_lm.__enter__()
                try:
                    _lm_result = _lm.maximize_loglike(
                        options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-7}
                    )
                    _ll_larch  = float(_lm_result.loglike)
                    # Fallback to relaxed tolerances if LL is NaN or very bad
                    if not np.isfinite(_ll_larch) or _ll_larch < self.ll_null_ * 2:
                        logger.debug(
                            f"  mode/{purpose}: Larch LL={_ll_larch:.1f} — retrying with "
                            "relaxed tolerances"
                        )
                        _lm_result = _lm.maximize_loglike(
                            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-4}
                        )
                        _ll_larch = float(_lm_result.loglike)
                finally:
                    _ennet_lm.__exit__(None, None, None)

                self.coef_   = _extract_larch_coef_to_scipy(_lm, self.param_names_, self.init_)
                self.ll_final_ = _ll_larch
                self.converged_ = True
                # Detect which engine was actually used
                _eng = getattr(_lm, '_compute_engine', 'unknown')
                self._estimation_engine = f"larch_{_eng}"
                logger.info(
                    f"  mode/{purpose}: Larch ({_eng}) converged — "
                    f"LL={_ll_larch:.1f}"
                )
                # ── Extract standard errors from Larch pf ─────────────────
                try:
                    _lm.calculate_parameter_covariance()
                except Exception:
                    pass
                try:
                    self._larch_pf = _lm.pf.copy()
                    _pf = self._larch_pf
                    _k  = len(self.param_names_)
                    _se    = np.full(_k, np.nan)
                    _tstat = np.full(_k, np.nan)
                    _pval  = np.full(_k, np.nan)
                    from scipy.stats import norm as _snorm
                    for _i, _pname in enumerate(self.param_names_):
                        _lk = 'Mu_Motorised'    if _pname == 'mu_motorised' else \
                              'Mu_NonMotorised'  if _pname == 'mu_nonmotorised' else _pname
                        if _lk not in _pf.index:
                            continue
                        _row = _pf.loc[_lk]
                        for _sc in ('std_err', 'std_err_opg', 'se', 'stderr'):
                            if _sc in _row.index:
                                _v = float(_row[_sc])
                                if np.isfinite(_v) and _v > 1e-10:
                                    _se[_i] = _v
                                    break
                        if np.isfinite(_se[_i]):
                            _tstat[_i] = float(self.coef_[_i]) / _se[_i]
                            _pval[_i]  = float(2.0 * _snorm.sf(abs(_tstat[_i])))
                    self.se_    = _se
                    self.tstat_ = _tstat
                    self.pval_  = _pval
                    logger.debug(
                        f"  mode/{purpose}: Larch SEs extracted — "
                        f"{int(np.isfinite(_se).sum())}/{_k} valid"
                    )
                except Exception as _se_e:
                    logger.debug(f"  mode/{purpose}: Larch SE extraction failed: {_se_e}")
                    self._larch_pf = None
                _larch_succeeded = True
            except Exception as _larch_e:
                # Emit detailed diagnostics so failures can be diagnosed
                _diag = []
                _diag.append(f"error={_larch_e!r}")
                try:
                    _n_modes = df['chosen_mode'].nunique()
                    _diag.append(f"n_modes_observed={_n_modes}")
                    if _n_modes < 2:
                        _diag.append("CRITICAL: only 1 mode in data — NL cannot estimate")
                except Exception:
                    pass
                try:
                    _numeric_df = df.select_dtypes(include=[np.number])
                    _zero_var   = [c for c in _numeric_df.columns
                                   if _numeric_df[c].std() < 1e-8
                                   and c not in ('chosen_mode_code', 'walk_avail', 'bike_avail')]
                    if _zero_var:
                        _diag.append(f"zero-variance cols={_zero_var[:5]}")
                    _nan_cols = [c for c in _numeric_df.columns
                                 if _numeric_df[c].isna().any()]
                    if _nan_cols:
                        _diag.append(f"NaN cols={_nan_cols[:5]}")
                    _diag.append(
                        f"walk_avail={int(df['walk_avail'].sum())} "
                        f"bike_avail={int(df['bike_avail'].sum())} of {len(df)}"
                    )
                except Exception:
                    pass
                logger.warning(
                    f"  mode/{purpose}: Larch estimation failed — falling back to scipy.\n"
                    f"  Diagnosis: {'; '.join(_diag)}\n"
                    "  Common causes: near-collinear columns (check log_dist vs tt_h), "
                    "zero walk/bike availability blocking gradient, or larch DataTree build error."
                )

        # ── 2. scipy L-BFGS-B fallback ───────────────────────────────────
        if not _larch_succeeded:
            from scipy.optimize import minimize
            try:
                res = minimize(
                    self._neg_ll,
                    self.init_.copy(),
                    args=(df, y),
                    method='L-BFGS-B',
                    bounds=self.bounds_,
                    options={'maxiter': 1000, 'ftol': 1e-10},
                )
                self.coef_      = res.x
                self.converged_ = res.success
                self.ll_final_  = -res.fun
                self._estimation_engine = 'scipy'
                # ── Extract SEs from L-BFGS-B inverse Hessian approximation ─
                # L-BFGS-B stores hess_inv ≈ (H_{-LL})^{-1} = Var(θ) approximation.
                try:
                    _V = np.asarray(res.hess_inv.todense())
                    _diag = np.diag(_V)
                    _se   = np.where(_diag > 0, np.sqrt(_diag), np.nan)
                    _se   = np.where(np.isfinite(_se), _se, np.nan)
                    _tstat = np.where(
                        np.isfinite(_se) & (_se > 1e-10),
                        self.coef_ / _se, np.nan
                    )
                    from scipy.stats import norm as _snorm2
                    _pval = np.where(
                        np.isfinite(_tstat),
                        2.0 * _snorm2.sf(np.abs(_tstat)), np.nan
                    )
                    self.se_    = _se
                    self.tstat_ = _tstat
                    self.pval_  = _pval
                    _n_valid = int(np.isfinite(_se).sum())
                    logger.debug(
                        f"  mode/{purpose}: scipy SEs from hess_inv — "
                        f"{_n_valid}/{len(self.param_names_)} valid"
                    )
                    # Warn about near-zero SEs (potential collinearity)
                    _near_bound = [
                        self.param_names_[i]
                        for i in range(len(self.param_names_))
                        if np.isfinite(_se[i]) and _se[i] < 0.002
                    ]
                    if _near_bound:
                        logger.warning(
                            f"  mode/{purpose}: very small SEs (collinearity risk): "
                            f"{_near_bound[:5]}. Check log_dist vs tt_h correlation, "
                            "or veh_da vs lic_da. Consider removing one of each pair."
                        )
                except Exception as _se_e:
                    logger.debug(f"  mode/{purpose}: scipy SE extraction failed: {_se_e}")
                    self.se_    = np.full(len(self.param_names_), np.nan)
                    self.tstat_ = np.full(len(self.param_names_), np.nan)
                    self.pval_  = np.full(len(self.param_names_), np.nan)
                if not res.success:
                    logger.warning(
                        f"  mode/{purpose}: scipy L-BFGS-B did not fully converge "
                        f"(status={res.status}, message={res.message}).\n"
                        f"  LL={self.ll_final_:.1f}, n={len(df)}\n"
                        f"  Likely cause: {res.message}\n"
                        "  Consider: (a) more HHTS data for this purpose, "
                        "(b) fixing near-zero variance predictors, "
                        "(c) increasing min_sample threshold to skip sparse purposes."
                    )
            except Exception as _scipy_e:
                logger.error(
                    f"  mode/{purpose}: scipy fallback also failed: {_scipy_e!r}\n"
                    "  Model will use INIT starting values — predictions will be uninformative."
                )
                self.coef_      = self.init_.copy()
                self.converged_ = False
                self.ll_final_  = self.ll_null_
                self._estimation_engine = 'init_fallback'
                self.se_        = np.full(len(self.param_names_), np.nan)
                self.tstat_     = np.full(len(self.param_names_), np.nan)
                self.pval_      = np.full(len(self.param_names_), np.nan)

        # ── Compute rho² ─────────────────────────────────────────────────
        if self.ll_null_ and self.ll_null_ != 0:
            self.rho2_ = 1 - (self.ll_final_ / self.ll_null_)
        else:
            self.rho2_ = float('nan')

        # ── Post-estimation diagnostics ───────────────────────────────────
        P_train   = self.predict_proba(df)
        pred_dist = dict(zip(MODES, P_train.mean(axis=0)))
        self.train_pred_shares_ = pred_dist
        pred_pct  = {m: f"{pred_dist[m]:.1%}" for m in MODES}
        logger.info(
            f"  mode/{purpose}: engine={self._estimation_engine}, "
            f"n={len(df)}, "
            f"converged={self.converged_}, "
            f"rho²={self.rho2_:.3f}, "
            f"LL={self.ll_final_:.1f}"
        )
        logger.info(f"  mode/{purpose} predicted (mean proba): {pred_pct}")

        # Detailed under-prediction diagnostics
        for mode_name in MODES:
            obs_s = obs_dist.get(mode_name, 0) / len(df)
            prd_s = pred_dist[mode_name]
            if obs_s > 0.01 and prd_s < obs_s / 4:
                # Explain likely cause
                if mode_name == 'Walk':
                    _why = (
                        f"walk_avail count={int(df.get('walk_avail',pd.Series(1,index=df.index)).sum())}. "
                        "Check DISTWALK skim values and walk_avail threshold (should be <6km)."
                    )
                elif mode_name == 'Bike':
                    _why = (
                        f"bike_avail count={int(df.get('bike_avail',pd.Series(1,index=df.index)).sum())}. "
                        "Check DISTBIKE skim values and bike_avail threshold (should be <25km)."
                    )
                elif mode_name == 'Transit':
                    _why = (
                        "Check WLK_LOC_WLK_TOTIVT__AM skim values (0=no service). "
                        "Transit IVT defaults to 45min when skim missing — may over-suppress transit."
                    )
                else:
                    _why = "Check starting values and bounds."
                logger.warning(
                    f"  mode/{purpose}: {mode_name} under-predicted "
                    f"(obs={obs_s:.1%}, pred={prd_s:.1%}). {_why}"
                )

        # ── Compute frozen probabilities for synth pop transfer ───────────
        self.frozen_probs_ = _compute_frozen_probs_s6(df, self, purpose)
        if self.frozen_probs_ is None:
            logger.warning(
                f"  mode/{purpose}: frozen probability computation failed — "
                "synth pop will use live model.predict_proba() (slightly noisy)"
            )

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, 5) probability matrix."""
        mu_m  = float(self.coef_[-2])
        mu_nm = float(self.coef_[-1])
        V     = build_utility_matrix(
            df, self.coef_[:-2], self.param_names_[:-2]
        )
        return self._nested_proba(V, mu_m, mu_nm)

    def predict_sampled(self, df: pd.DataFrame,
                        rng: np.random.Generator) -> np.ndarray:
        """Sample one mode per row from predicted distribution."""
        P = self.predict_proba(df)
        return np.array([
            MODES[rng.choice(len(MODES), p=P[i])]
            for i in range(len(df))
        ])

    def predict_mode(self, df: pd.DataFrame) -> np.ndarray:
        """Return modal (most likely) mode per row."""
        P = self.predict_proba(df)
        return np.array([MODES[i] for i in P.argmax(axis=1)])

    def summary(self) -> pd.DataFrame:
        """Return coefficient summary table."""
        return pd.DataFrame({
            'parameter': self.param_names_,
            'estimate':  np.round(self.coef_, 4),
        })


def _rebalance_edu_walk_sr(P: np.ndarray, model) -> np.ndarray:
    """Shift some probability mass Walk->SR for edu when walk is over-predicted."""
    try:
        obs = getattr(model, 'obs_shares_', None) or {}
        obs_walk = float(obs.get('Walk', np.nan))
        obs_sr = float(obs.get('SR', np.nan))
        if not (np.isfinite(obs_walk) and np.isfinite(obs_sr)):
            return P
        idx_walk = MODE_IDX['Walk']
        idx_sr = MODE_IDX['SR']
        pred_walk = float(np.nanmean(P[:, idx_walk]))
        pred_sr = float(np.nanmean(P[:, idx_sr]))
        # Trigger only when mismatch is material.
        if not (pred_walk > obs_walk + 0.03 and pred_sr + 0.02 < obs_sr):
            return P
        shift_share = min(pred_walk - obs_walk, obs_sr - pred_sr, 0.12)
        if shift_share <= 0:
            return P
        walk_col = P[:, idx_walk].copy()
        take = walk_col * (shift_share / max(pred_walk, 1e-9))
        P[:, idx_walk] = np.clip(P[:, idx_walk] - take, 1e-12, 1.0)
        P[:, idx_sr] = np.clip(P[:, idx_sr] + take, 1e-12, 1.0)
        P = np.clip(P, 1e-12, 1.0)
        P /= P.sum(axis=1, keepdims=True)
        logger.info(
            f"  edu share rebalance applied: Walk {pred_walk:.1%}->{float(np.mean(P[:, idx_walk])):.1%}, "
            f"SR {pred_sr:.1%}->{float(np.mean(P[:, idx_sr])):.1%}"
        )
        return P
    except Exception as _e:
        logger.debug(f"edu share rebalance skipped: {_e}")
        return P


def _rebalance_to_observed_shares(P: np.ndarray, model, purpose: str) -> np.ndarray:
    """
    Stabilise purpose-level predicted shares toward observed HHTS shares.

    Uses iterative column scaling (IPF-like) on the probability matrix so rows
    remain proper probabilities while the purpose-level mean shares become more
    representative. This prevents DA-collapse in sparse/noisy transfer cases.
    """
    try:
        if P is None or len(P) == 0:
            return P
        obs = getattr(model, 'obs_shares_', None) or {}
        obs_vec = np.array([float(obs.get(m, np.nan)) for m in MODES], dtype=float)
        if not np.isfinite(obs_vec).all() or obs_vec.sum() <= 0:
            return P
        obs_vec = obs_vec / obs_vec.sum()

        pred_vec = np.nanmean(P, axis=0)
        pred_sum = float(np.nansum(pred_vec))
        if not np.isfinite(pred_sum) or pred_sum <= 0:
            return P
        pred_vec = pred_vec / pred_sum

        pred_da = float(pred_vec[MODE_IDX['DA']])
        obs_da = float(obs_vec[MODE_IDX['DA']])
        pred_non_da = float(1.0 - pred_da)
        obs_non_da = float(1.0 - obs_da)

        # Strong corrective blending when DA dominates unrealistically.
        severe_da_collapse = (
            (pred_da > 0.90 and obs_da < 0.80) or
            (pred_non_da < max(0.12, obs_non_da * 0.45))
        )
        if severe_da_collapse:
            alpha = 0.45
        else:
            max_gap = float(np.max(np.abs(pred_vec - obs_vec)))
            if max_gap < 0.06:
                return P
            alpha = 0.22 if purpose != 'edu' else 0.30

        target = (1.0 - alpha) * pred_vec + alpha * obs_vec

        # Keep each mode alive relative to observed distribution.
        floor = np.minimum(obs_vec * 0.55, obs_vec)
        target = np.maximum(target, floor)
        target = target / target.sum()

        P_adj = np.clip(P.astype(float), 1e-12, 1.0)
        for _ in range(4):
            cur = P_adj.mean(axis=0)
            cur = np.clip(cur, 1e-12, None)
            scale = target / cur
            P_adj *= scale[None, :]
            P_adj = np.clip(P_adj, 1e-12, 1.0)
            P_adj /= P_adj.sum(axis=1, keepdims=True)

        new_vec = P_adj.mean(axis=0)
        logger.info(
            f"  {purpose} share stabilisation: "
            f"DA {pred_da:.1%}->{float(new_vec[MODE_IDX['DA']]):.1%} (obs {obs_da:.1%}), "
            f"non-DA {pred_non_da:.1%}->{float(1.0-new_vec[MODE_IDX['DA']]):.1%}"
        )
        return P_adj
    except Exception as _e:
        logger.debug(f"share stabilisation skipped for {purpose}: {_e}")
        return P


# ---------------------------------------------------------------------------
# 3.  ESTIMATE — ONE MODEL PER PURPOSE
# ---------------------------------------------------------------------------

def estimate_mode_models(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: Optional[pd.DataFrame] = None,
    min_sample: int = 20,
) -> dict:
    """
    Fit one NestedLogitModeModel per activity purpose.

    Returns dict {purpose: NestedLogitModeModel}
    """
    data = prepare_mode_data(trips, persons, landuse)

    models = {}

    for purpose in PURPOSES:
        subset = data[data['purpose'] == purpose].copy()

        if len(subset) < min_sample:
            logger.warning(f"  mode/{purpose}: {len(subset)} obs — skipping")
            continue

        # check mode variety
        n_modes = subset['chosen_mode'].nunique()
        if n_modes < 2:
            logger.warning(
                f"  mode/{purpose}: only {n_modes} mode(s) observed — skipping"
            )
            continue

        m = NestedLogitModeModel()
        m.fit(subset, purpose)
        models[purpose] = m

        # ── Model validity check ─────────────────────────────────────────
        _warns = []
        if not m.converged_:
            _warns.append(f"NOT converged (engine={m._estimation_engine})")
        if m.rho2_ is not None and not np.isnan(m.rho2_) and m.rho2_ < 0.05:
            _warns.append(f"rho²={m.rho2_:.3f} — barely better than market shares")
        # Key parameter sign checks
        _pdict = dict(zip(m.param_names_, m.coef_))
        for _pname, _expected_sign in [('tt_da', 'neg'), ('ldw', 'neg'), ('ldb', 'neg')]:
            _v = _pdict.get(_pname, 0.0)
            if _expected_sign == 'neg' and _v > 0.001:
                _warns.append(
                    f"param '{_pname}'={_v:.3f} is positive — should be negative "
                    "(longer travel time/distance should deter the mode)"
                )
        if _warns:
            for _w in _warns:
                logger.warning(f"  mode/{purpose} validity: {_w}")
        else:
            logger.info(
                f"  mode/{purpose} validity: OK — "
                f"engine={m._estimation_engine}, rho²={m.rho2_:.3f}"
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    _summary_lines = []
    for p in models:
        _m  = models[p]
        _fp = "YES" if _m.frozen_probs_ is not None else "no"
        _summary_lines.append(
            f"  {p:10s}: engine={_m._estimation_engine:15s}, "
            f"rho²={_m.rho2_:.3f}, n={_m.n_obs_:6d}, frozen_probs={_fp}"
        )
    print(
        f"\nStage 6: fitted {len(models)} mode models "
        f"({', '.join(models.keys())})\n" +
        "\n".join(_summary_lines)
    )
    return models


# ---------------------------------------------------------------------------
# 4.  PREDICT — SYNTHETIC POPULATION
# ---------------------------------------------------------------------------

def predict_modes(
    scheduled: pd.DataFrame,
    models: dict,
    landuse: Optional[pd.DataFrame] = None,
    sample: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign a travel mode to every activity in the Stage 5 output.

    Parameters
    ----------
    scheduled : Stage 5 output — one row per activity with
                OTAZ, DTAZ, STARTIME, purpose, tt_avg, dist,
                distwalk, distbike, crash, vehrd, …
    models    : from estimate_mode_models()
    landuse   : optional, used for accessibility
    sample    : sample from predicted distribution (True) or argmax (False)
    seed      : random seed

    Returns
    -------
    scheduled with mode, mode_proba_DA, …Bike columns added.
    """
    rng = np.random.default_rng(seed)
    df  = scheduled.copy()

    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # derive convenience columns if missing
    for col, default in [('has_vehicle', None), ('no_vehicle', None)]:
        if col not in df.columns:
            if 'vehrd' in df.columns:
                df['has_vehicle'] = (df['vehrd'] >= 1).astype(float)
                df['no_vehicle']  = (df['vehrd'] == 0).astype(float)
            else:
                df['has_vehicle'] = 1.0
                df['no_vehicle']  = 0.0

    # fill skim columns
    for col in ['tt_avg', 'dist', 'distwalk', 'distbike', 'crash', 'log_dist']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # If distbike is missing or zero, use road distance as proxy (bikes broadly
    # follow road OD). This prevents bike utility collapsing to pure ASC.
    if (df['distbike'] <= 0).all() and (df['dist'] > 0).any():
        df['distbike'] = df['dist'] * 0.85
        logger.info("predict_modes: distbike was all-zero — using dist*0.85 as proxy")
    elif (df['distbike'] <= 0).any():
        zero_bike = df['distbike'] <= 0
        df.loc[zero_bike, 'distbike'] = df.loc[zero_bike, 'dist'] * 0.85

    # Likewise for distwalk: if all zero use dist as rough proxy
    if (df['distwalk'] <= 0).all() and (df['dist'] > 0).any():
        df['distwalk'] = df['dist'] * 1.10
        logger.info("predict_modes: distwalk was all-zero — using dist*1.10 as proxy")

    df['log_dist']   = np.log1p(df['dist'])
    df['cost_proxy'] = df['dist'] * FUEL_RATE

    # attach accessibility
    if 'accessibility' not in df.columns:
        if landuse is not None and 'DTAZ' in df.columns:
            lu = landuse.copy()
            zc = next(
                (c for c in ['ORDER', 'DTAZ', 'SA2_MAIN16'] if c in lu.columns),
                None
            )
            if zc and zc != 'DTAZ':
                lu = lu.rename(columns={zc: 'DTAZ'})
            if 'DTAZ' in lu.columns:
                acc_cols = [c for c in ['FPSEMPN', 'MWTEMPN', 'OTHEMP',
                                         'RETEMPN', 'TOTHH']
                            if c in lu.columns]
                if acc_cols:
                    lu['accessibility'] = np.log1p(
                        lu[acc_cols].apply(pd.to_numeric, errors='coerce')
                        .fillna(0).sum(axis=1)
                    )
                lu['DTAZ'] = pd.to_numeric(lu['DTAZ'], errors='coerce').astype(int)
                df = df.merge(lu[['DTAZ', 'accessibility']], on='DTAZ', how='left')
        df['accessibility'] = df.get('accessibility', pd.Series(0.0, index=df.index)).fillna(0)

    # ptype
    if 'ptype' not in df.columns:
        try:
            from stage1_activity_frequency import assign_ptype_column
            df = assign_ptype_column(df)
        except ImportError:
            df['ptype'] = 'NonWorker'

    # person features — extended with age5p and industry/licence spec
    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2, 'WORKSTAT': 0,
                'STUDYING': 0, 'incp': 8, 'age5p': 5, 'indp': 20, 'CARLICENCE': 1}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    # STARTIME
    df['start_hr'] = pd.to_numeric(
        df.get('STARTIME', 480), errors='coerce'
    ).fillna(480) / 60

    # ── Derived person-type columns for utility spec ──────────────────────
    df['has_vehicle'] = (df['vehrd'] >= 1).astype(float)
    df['no_vehicle']  = (df['vehrd'] == 0).astype(float)
    df['female']      = (df['sexp'] == 2).astype(float)
    df['young']       = (df['age5p'] <= 5).astype(float)
    df['older']       = (df['age5p'] >= 10).astype(float)
    df['worker']      = np.clip(df['WORKSTAT'], 0, 1)
    df['student']     = (df['STUDYING'] > 0).astype(float)
    df['inc_hi']      = np.clip(df['incp'] - 8, 0, None).astype(float)
    df['inc_lo']      = np.clip(8 - df['incp'], 0, None).astype(float)

    # Log-transform walk/bike distances — matches build_utility_matrix
    df['ldw'] = np.log1p(df['distwalk'])
    df['ldb'] = np.log1p(df['distbike'])

    # ── Extended socio-demographics (synth pop transfer) ──────────────────
    _indp_s = pd.to_numeric(
        df.get('indp', pd.Series(20, index=df.index)), errors='coerce'
    ).fillna(20)
    df['indp_industrial'] = _indp_s.isin([2, 3, 4, 5]).astype(float)
    df['indp_public']     = _indp_s.isin([15, 16, 17]).astype(float)

    _lic_s = pd.to_numeric(
        df.get('CARLICENCE', pd.Series(1, index=df.index)), errors='coerce'
    ).fillna(1).clip(0, 1)
    df['licence']    = _lic_s.astype(float)

    _ws_s = pd.to_numeric(df.get('WORKSTAT', 0), errors='coerce').fillna(0)
    df['full_time']  = (_ws_s == 2).astype(float)
    df['part_time']  = (_ws_s == 1).astype(float)

    logger.info(
        f"predict_modes: {len(df)} activities | "
        f"noveh={df['no_vehicle'].mean():.1%}, "
        f"female={df['female'].mean():.1%}, "
        f"worker={df['worker'].mean():.1%}, "
        f"student={df['student'].mean():.1%}"
    )

    df['mode'] = 'DA'   # default

    # initialise probability columns
    for m in MODES:
        df[f'prob_{m}'] = 0.0

    # Fallback order when a purpose model is missing: prefer similar purposes,
    # then aggregate over all available models (avoids DA=100% everywhere).
    _fallback_purpose_order = {
        'work':     ['work', 'othmaint', 'shopping', 'social'],
        'edu':      ['edu', 'work', 'social'],
        'social':   ['social', 'othdiscr', 'eat', 'shopping'],
        'escort':   ['escort', 'social', 'shopping'],
        'shopping': ['shopping', 'othmaint', 'social'],
        'othmaint': ['othmaint', 'shopping', 'social'],
        'othdiscr': ['othdiscr', 'social', 'eat'],
        'eat':      ['eat', 'social', 'othdiscr'],
    }

    def _get_fallback_mode_model(purpose: str) -> object:
        for p in _fallback_purpose_order.get(purpose, []):
            if p in models:
                return models[p]
        # last resort: any model
        return next(iter(models.values())) if models else None

    # Check whether any model has frozen probs available
    _n_frozen = sum(1 for m in models.values()
                    if getattr(m, 'frozen_probs_', None) is not None)
    _using_frozen = _n_frozen > 0
    if _using_frozen:
        logger.info(
            f"predict_modes: {_n_frozen}/{len(models)} models have frozen probability "
            "tables — applying cell-level HHTS→synth lookup for reproducible predictions."
        )
    else:
        logger.info(
            "predict_modes: no frozen probability tables found — "
            "using live model.predict_proba() (slightly noisier, fine for production)."
        )

    for purpose in PURPOSES:
        mask = df['purpose'] == purpose
        if not mask.any():
            continue

        if purpose not in models:
            fallback_model = _get_fallback_mode_model(purpose)
            if fallback_model is None:
                logger.warning(
                    f"  No mode model for {purpose} — defaulting to DA. "
                    "To fix: ensure HHTS estimation data has ≥50 trips for this purpose."
                )
                df.loc[mask, 'mode'] = 'DA'
                df.loc[mask, 'prob_DA'] = 1.0
                continue
            logger.warning(
                f"  No mode model for {purpose} — "
                f"using {fallback_model.purpose_} model as fallback"
            )
            model = fallback_model
        else:
            model = models[purpose]

        subset = df.loc[mask].copy()

        # add dummy chosen_mode column (needed by build_utility_matrix API)
        subset['chosen_mode'] = 'DA'

        # ── Apply frozen probabilities if available ───────────────────────
        # Frozen probs are cell-level HHTS averages keyed on person-type cells.
        # They give reproducible, MC-noise-free predictions for the synth pop.
        _frozen = getattr(model, 'frozen_probs_', None)
        if _frozen is not None:
            P_frozen = _mode_probs_frozen_s6(_frozen, subset)
        else:
            P_frozen = None

        if P_frozen is not None:
            P = P_frozen
            _prob_source = 'frozen'
        else:
            P = model.predict_proba(subset)
            _prob_source = 'live'

        if purpose == 'edu':
            P = _rebalance_edu_walk_sr(P, model)
        P = _rebalance_to_observed_shares(P, model, purpose)

        if purpose in models and not hasattr(models[purpose], '_prob_source_logged'):
            logger.debug(
                f"  {purpose}: using {_prob_source} probabilities "
                f"(n={int(mask.sum())})"
            )
            models[purpose]._prob_source_logged = True

        # Store mode-level probabilities in output columns
        # S6_FROZEN_PROB_COLS order: fp_DA, fp_SR, fp_Transit, fp_Walk, fp_Bike
        # MODES order: DA, SR, Transit, Walk, Bike — same, no reorder needed
        for j, mode_name in enumerate(MODES):
            df.loc[mask, f'prob_{mode_name}'] = P[:, j]

        if sample:
            preds = np.array([
                MODES[rng.choice(len(MODES), p=np.clip(P[i], 1e-12, 1.0) /
                                              np.clip(P[i], 1e-12, 1.0).sum())]
                for i in range(len(subset))
            ])
        else:
            preds = np.array([MODES[i] for i in P.argmax(axis=1)])

        df.loc[mask, 'mode'] = preds

    # ── Final summary with per-purpose breakdown ─────────────────────────
    mode_dist  = df['mode'].value_counts().to_dict()
    total      = len(df)
    mode_pct   = {m: f"{mode_dist.get(m,0)/total:.1%}" for m in MODES}

    print(
        f"Stage 6 predictions: {total} activities with modes\n"
        f"  Overall mode share: {mode_pct}"
    )
    logger.info(f"Stage 6 final mode distribution (counts): {mode_dist}")

    # Per-purpose breakdown
    for purpose in PURPOSES:
        pmask = df['purpose'] == purpose
        if not pmask.any():
            continue
        pdist = df.loc[pmask, 'mode'].value_counts().to_dict()
        pn    = int(pmask.sum())
        ppct  = {m: f"{pdist.get(m,0)/pn:.1%}" for m in MODES}
        logger.info(f"  {purpose:10s} (n={pn:6d}): {ppct}")

    # Warn if walk+bike+transit together < 5% of total
    active_tr = sum(mode_dist.get(m, 0) for m in ['Walk', 'Bike', 'Transit'])
    if active_tr / total < 0.05:
        logger.warning(
            f"Stage 6: Walk+Bike+Transit combined = {active_tr/total:.1%} — "
            "very low. Check skim distances and person attributes passed to "
            "predict_modes. DA-dominated output suppresses model realism."
        )

    return df


# ---------------------------------------------------------------------------
# 4b. TOUR-MODE CONSISTENCY
# ---------------------------------------------------------------------------
# Priority for deriving a single tour mode when the primary-leg heuristic can't
# be used: a private vehicle taken on ANY leg must serve the whole tour (you
# cannot strand a car or bicycle mid-tour), so DA and Bike dominate; SR/Transit/
# Walk have no vehicle-continuity constraint.
_TOUR_MODE_PRIORITY = ['DA', 'Bike', 'SR', 'Transit', 'Walk']


def enforce_tour_mode_consistency(df: pd.DataFrame, mode_col: str = 'mode') -> pd.DataFrame:
    """Constrain every leg of a tour to a single, consistent tour mode.

    Stage 6 picks each leg's mode independently, which can yield physically
    impossible tours (a drive-alone leg stranded inside a transit tour, or a car
    abandoned mid-tour). ActivitySim instead chooses ONE tour mode and constrains
    the trip legs to it. Here the tour mode is the mode of the PRIMARY
    (max-distance) leg — the main-destination leg — falling back to the
    vehicle-continuity priority (DA/Bike dominate) when distances are unavailable.

    Two modes via STAGE6_TOUR_MODE_CONSISTENCY:
      • 'relaxed' / '1' (default) — the ActivitySim-style nuance: a genuine WALK
        leg (walk access/egress) is allowed to remain inside a tour whose mode is
        NOT a private vehicle (Transit / SR / Walk), because walking a sub-leg
        never strands anything. Every other non-conforming leg is set to the tour
        mode. Private-vehicle tours (DA, Bike) stay strict — a walk leg there
        would abandon the car/bicycle — so ALL their legs take the tour mode.
      • 'strict' — every leg is set to the tour mode, no exceptions.
      • '0' / 'off' — disabled.

    The raw per-leg prediction is preserved in ``<mode_col>_leg`` and the chosen
    tour mode in ``tour_mode``. Grouping key preference:
    ptour_id → person_id+tour_id → person_id+tour_num → person_id (whole day)."""
    _cfg = os.environ.get('STAGE6_TOUR_MODE_CONSISTENCY', '1').strip().lower()
    if _cfg in ('0', 'off', 'false', 'no'):
        return df
    _strict = _cfg in ('strict', '2')
    if mode_col not in df.columns or df.empty:
        return df
    cols = set(df.columns)
    if 'ptour_id' in cols:
        tk = df['ptour_id'].astype(str)
    elif {'person_id', 'tour_id'} <= cols:
        tk = df['person_id'].astype(str) + '_' + df['tour_id'].astype(str)
    elif {'person_id', 'tour_num'} <= cols:
        tk = df['person_id'].astype(str) + '_' + df['tour_num'].astype(str)
    elif 'person_id' in cols:
        tk = df['person_id'].astype(str)
        logger.info("tour-mode consistency: no tour id — grouping by person (whole day)")
    else:
        logger.warning("tour-mode consistency: no person/tour key — skipping")
        return df

    df = df.copy()
    df['_tk'] = tk.values
    _dist = pd.to_numeric(df.get('dist', pd.Series(np.nan, index=df.index)),
                          errors='coerce')

    _prio = {m: i for i, m in enumerate(_TOUR_MODE_PRIORITY)}

    def _tour_mode(sub: pd.DataFrame):
        modes = sub[mode_col].tolist()
        d = _dist.loc[sub.index]
        if d.notna().any() and d.max() > 0:
            return sub.loc[d.idxmax(), mode_col]          # primary (max-dist) leg
        valid = [m for m in modes if m in _prio]
        return min(valid, key=lambda m: _prio[m]) if valid else (modes[0] if modes else None)

    tmode = pd.Series(index=df.index, dtype=object)
    for _k, sub in df.groupby('_tk'):
        tmode.loc[sub.index] = _tour_mode(sub)

    raw = df[mode_col].astype(object)
    df[mode_col + '_leg'] = df[mode_col]                   # keep raw per-leg mode
    df['tour_mode'] = tmode.values

    new = tmode.copy()
    n_walk_kept = 0
    if not _strict:
        # relaxed: keep a genuine walk access/egress leg when the tour mode is not
        # a private vehicle (walking never strands a car/bike).
        keep_walk = (raw == 'Walk') & tmode.notna() & (~tmode.isin(['DA', 'Bike']))
        new[keep_walk] = 'Walk'
        n_walk_kept = int(keep_walk.sum())
    _ok = new.notna()
    df.loc[_ok, mode_col] = new[_ok].values
    changed = int((raw != df[mode_col].astype(object)).sum())
    df = df.drop(columns=['_tk'])
    n_tours = int(pd.Series(tk.values).nunique())
    logger.info(f"tour-mode consistency [{'strict' if _strict else 'relaxed'}]: "
                f"{changed}/{len(df)} legs reconciled across {n_tours} tours "
                f"(primary max-dist leg; DA/Bike continuity)"
                + ('' if _strict else f"; {n_walk_kept} walk access/egress legs kept")
                + f". Raw per-leg mode preserved in '{mode_col}_leg'.")
    return df


# ---------------------------------------------------------------------------
# 5.  BUILD FINAL ACTIVITY PLAN
# ---------------------------------------------------------------------------

def build_activity_plan(scheduled: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean, ordered activity plan from the Stage 6 output.

    Columns in output:
        person_id, activity_seq, purpose, OTAZ, DTAZ,
        STARTIME, ENDTIME, predicted_duration_min,
        mode, time_period,
        prob_DA, prob_SR, prob_Transit, prob_Walk, prob_Bike,
        start_of_day, end_of_day, B, ptype

    Rows are sorted by person_id then STARTIME.
    """
    keep = [
        'person_id', 'activity_seq', 'purpose',
        'OTAZ', 'DTAZ', 'STARTIME', 'ENDTIME',
        'predicted_duration_min', 'mode', 'time_period',
        'prob_DA', 'prob_SR', 'prob_Transit', 'prob_Walk', 'prob_Bike',
        'start_of_day', 'end_of_day', 'B', 'ptype',
        'tt_avg', 'dist', 'distwalk', 'distbike', 'crash',
        'sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING',
    ]
    present = [c for c in keep if c in scheduled.columns]
    plan = (
        scheduled[present]
        .sort_values(['person_id', 'STARTIME'])
        .reset_index(drop=True)
    )
    return plan

def export_for_dashboard(activity_plan: pd.DataFrame,
                         path: str = 'saved_data/dashboard_data.json'):
    import os, json
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = activity_plan.copy()

    # compute dwell time = time at destination before next trip departs
    df = df.sort_values(['person_id', 'STARTIME']).reset_index(drop=True)
    df['next_STARTIME'] = df.groupby('person_id')['STARTIME'].shift(-1)
    df['dwell_min'] = (df['next_STARTIME'] - df['ENDTIME']).clip(lower=0).fillna(0).astype(int)

    cols = ['person_id', 'ptype', 'purpose', 'STARTIME', 'ENDTIME',
            'predicted_duration_min', 'dwell_min', 'mode',
            'vehrd', 'start_of_day', 'end_of_day', 'OTAZ', 'DTAZ']
    present = [c for c in cols if c in df.columns]

    records = df[present].to_dict(orient='records')
    with open(path, 'w') as f:
        json.dump(records, f)
    print(f"Dashboard data saved → {path}  ({len(records)} activities)")
# ---------------------------------------------------------------------------
# 6.  SAVE / LOAD
# ---------------------------------------------------------------------------

def _save_s6_destinations_html(activity_plan: pd.DataFrame, path: str) -> None:
    """Save an HTML report showing predicted destination distributions by purpose.

    Shows for each trip purpose: top-10 destination zones by trip count, median
    distance, and a mini bar chart of mode share, sourced from the activity_plan
    produced by run_stage6.
    """
    trips = activity_plan[activity_plan.get('row_type', pd.Series('trip')) == 'trip'].copy() \
        if 'row_type' in activity_plan.columns else activity_plan.copy()

    if trips.empty or 'purpose' not in trips.columns or 'DTAZ' not in trips.columns:
        logger.warning("_save_s6_destinations_html: no trip rows / missing columns — skipped")
        return

    PURPS = [p for p in PURPOSES if p in trips['purpose'].values]
    rows_html = []
    for purp in PURPS:
        sub = trips[trips['purpose'] == purp]
        if sub.empty:
            continue
        n_total = len(sub)
        top_zones = (sub['DTAZ'].value_counts().head(10)
                     .reset_index().rename(columns={'DTAZ': 'zone', 'count': 'n_trips'}))
        top_zones['pct'] = (top_zones['n_trips'] / n_total * 100).round(1)

        dist_med = f"{sub['dist'].median():.1f} km" if 'dist' in sub.columns else 'n/a'

        mode_share = ''
        if 'mode' in sub.columns:
            mc = sub['mode'].value_counts(normalize=True).mul(100).round(1)
            mode_share = '  '.join(f'{m}: {v:.1f}%' for m, v in mc.items())

        zone_rows = ''.join(
            f"<tr><td>{r['zone']}</td><td>{r['n_trips']:,}</td>"
            f"<td>{r['pct']:.1f}%</td></tr>"
            for _, r in top_zones.iterrows()
        )
        rows_html.append(f"""
        <h3>{purp} <span class="n">n={n_total:,}</span></h3>
        <p>Median trip distance: <b>{dist_med}</b> &nbsp;|&nbsp; Mode share: {mode_share}</p>
        <table>
          <tr><th>DTAZ</th><th>Trips</th><th>Share</th></tr>
          {zone_rows}
        </table>""")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Stage 6 — Destination distributions by purpose</title>
<style>
  body{{font-family:Arial,sans-serif;font-size:13px;margin:20px}}
  h2{{color:#2c3e50}} h3{{color:#34495e;margin-top:18px}}
  .n{{font-size:11px;color:#888;font-weight:normal}}
  table{{border-collapse:collapse;margin-bottom:12px;font-size:12px}}
  th,td{{border:1px solid #ddd;padding:4px 10px;text-align:left}}
  th{{background:#eef2f7}}
</style></head><body>
<h2>Stage 6 — Predicted destinations by purpose</h2>
<p>Top-10 DTAZ zones per purpose from synthetic population prediction.</p>
{''.join(rows_html)}
</body></html>"""

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Stage 6 destinations → {path}")


def save_models(models: dict, path: str = 'saved_data/stage6_models.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Stage 6 models saved → {path}")
    # Always write coefficient CSV + HTML alongside the pickle
    _coef_path = path.replace('.pkl', '_coefficients.csv')
    try:
        _save_s6_coefficients(models, _coef_path)
    except Exception as _e:
        logger.warning(f"Stage 6 coefficient export failed: {_e}")
    # Write paper-section HTML with narrative
    _paper_html = path.replace('.pkl', '_paper_section.html')
    try:
        from paper_model_forms import write_all_paper_outputs
        write_all_paper_outputs(stage6_models=models)
    except Exception as _e:
        logger.debug(f"Stage 6 paper HTML export failed: {_e}")


# ---------------------------------------------------------------------------
# HTML / CSV COEFFICIENT EXPORT — Stage 6 trip-level mode models
# ---------------------------------------------------------------------------

def _s6_sig_stars(t):
    try:
        at = abs(float(t))
        if not np.isfinite(at):
            return ''
        return '★★★' if at > 3.3 else '★★' if at > 2.6 else '★' if at > 1.96 else ''
    except Exception:
        return ''


def _s6_sig_color(t):
    try:
        at = abs(float(t))
        if not np.isfinite(at):
            return '#888888'
        return '#1a7a1a' if at > 3.3 else '#2e7d32' if at > 2.6 \
            else '#558b2f' if at > 1.96 else '#b71c1c'
    except Exception:
        return '#888888'


def _s6_model_table_html(purpose: str, model: 'NestedLogitModeModel') -> str:
    """Build one HTML table section for a single trip-level mode model."""
    coef  = model.coef_
    se    = model.se_    if model.se_    is not None else np.full(len(coef), np.nan)
    tstat = model.tstat_ if model.tstat_ is not None else np.full(len(coef), np.nan)
    pval  = model.pval_  if model.pval_  is not None else np.full(len(coef), np.nan)

    def _fv(v): return f'{float(v):.4f}' if np.isfinite(float(v)) else '—'
    def _ft(v): return f'{float(v):.2f}' if np.isfinite(float(v)) else '—'

    rows_html = '\n'.join(
        f'<tr>'
        f'<td>{p}</td>'
        f'<td style="text-align:right">{_fv(c)}</td>'
        f'<td style="text-align:right">{_fv(s)}</td>'
        f'<td style="text-align:right; color:{_s6_sig_color(t)}">'
        f'{_ft(t)} {_s6_sig_stars(t)}</td>'
        f'<td style="text-align:right">{_fv(pv)}</td>'
        f'</tr>'
        for p, c, s, t, pv in zip(
            model.param_names_, coef, se, tstat, pval
        )
    )
    rho2_str  = f'{model.rho2_:.3f}' if model.rho2_ is not None and np.isfinite(model.rho2_) else 'n/a'
    ll_str    = f'{model.ll_final_:.1f}' if model.ll_final_ is not None else 'n/a'
    null_str  = f'{model.ll_null_:.1f}'  if model.ll_null_  is not None else 'n/a'

    # p-value diagnosis note
    _sig_count = int(np.sum(np.abs(tstat[np.isfinite(tstat)]) > 1.96))
    _tot_count = int(np.isfinite(tstat).sum())
    _diag_note = ''
    if _tot_count > 0 and _sig_count < _tot_count / 2:
        _diag_note = (
            '<p class="warn">⚠ Fewer than half of parameters are statistically significant '
            '(|t|&gt;1.96). Possible causes: (1) small HHTS sample for this purpose, '
            '(2) near-collinear predictors (log_dist ↔ tt_h; veh_da ↔ lic_da), '
            '(3) parameters hitting bounds. Consider aggregating rare purposes or '
            'dropping one of each correlated pair before the next run.</p>'
        )

    return f"""
<div class="model-section">
  <h2>Trip-level Mode Choice — {purpose.upper()} (Nested Logit)</h2>
  <p class="stats">
    <b>Log-likelihood:</b> {ll_str} &nbsp;|&nbsp;
    <b>LL(null market shares):</b> {null_str} &nbsp;|&nbsp;
    <b>&rho;&sup2; (null):</b> {rho2_str} &nbsp;|&nbsp;
    <b>Obs:</b> {model.n_obs_} &nbsp;|&nbsp;
    <b>Engine:</b> {model._estimation_engine} &nbsp;|&nbsp;
    <b>Converged:</b> {'Yes' if model.converged_ else 'No'}
  </p>
  {_diag_note}
  <table>
    <thead>
      <tr><th>Parameter</th><th>Estimate</th><th>Std Err</th>
          <th>t-stat</th><th>p-value</th></tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
  <p class="sig-note">★ p&lt;0.05 &nbsp; ★★ p&lt;0.01 &nbsp; ★★★ p&lt;0.001</p>
  <p class="sig-note"><b>Nest structure:</b> Motorised [DA, SR, Transit] | Non-Motorised [Walk, Bike].
  <b>Safety:</b> <i>crash</i> coefficient (negative = crash risk deters DA/SR).
  <b>Worker type:</b> <i>lic_da</i>=licence→DA, <i>ft_da</i>=full-time→DA, <i>ind_da</i>=industrial worker→DA,
  <i>pt_tr</i>=part-time→transit, <i>pub_tr</i>=public/health worker→transit.</p>
</div>"""


def _write_s6_html_report(models: dict, path: str):
    """Write full HTML report for all Stage 6 trip-level mode models."""
    sections = [_s6_model_table_html(purpose, model)
                for purpose, model in models.items()]
    body = '\n'.join(sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stage 6 — Trip-level Mode Choice Model Results</title>
  <style>
    body   {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
    h1     {{ border-bottom: 2px solid #1565c0; padding-bottom: 8px; }}
    h2     {{ color: #1565c0; margin-top: 32px; }}
    .model-section {{ margin-bottom: 40px; border: 1px solid #ccc;
                      border-radius: 6px; padding: 20px; }}
    .stats {{ background: #e3f2fd; padding: 8px 12px; border-radius: 4px;
              font-size: 0.95em; margin-bottom: 12px; }}
    .warn  {{ background: #fff3e0; border-left: 4px solid #e65100;
              padding: 8px 12px; font-size: 0.9em; margin: 8px 0; }}
    table  {{ border-collapse: collapse; width: 100%; }}
    th     {{ background: #1565c0; color: white; padding: 8px 12px;
              text-align: left; }}
    td     {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f5f5f5; }}
    .sig-note {{ font-size: 0.82em; color: #555; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Stage 6 — Trip-Level Mode Choice Models (Nested Logit per Purpose)</h1>
  <p>
    These are <b>trip-level</b> nested logit mode choice models estimated from
    the SEQ Household Travel Survey (HHTS). One model is estimated per activity
    purpose. They complement the <b>tour-level</b> mode/destination models in
    Stage 5. Utility specification includes travel time, distance, vehicle ownership,
    car licence, income, age, gender, employment status (worker/student),
    worker type (full-time, part-time, industrial, public/health), and
    <b>crash risk</b> (safety deterrence term).
  </p>
  <p>Generated by <code>stage6_mode_choice.py</code> — QTS/HHTS ABM Pipeline.</p>
{body}
</body>
</html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Stage 6 HTML report → {path}")


def _save_s6_coefficients(models: dict, path: str):
    """Save Stage 6 coefficient table as CSV + HTML report."""
    rows = []
    for purpose, model in models.items():
        coef  = model.coef_
        se    = model.se_    if model.se_    is not None else np.full(len(coef), np.nan)
        tstat = model.tstat_ if model.tstat_ is not None else np.full(len(coef), np.nan)
        pval  = model.pval_  if model.pval_  is not None else np.full(len(coef), np.nan)
        for param, c, s, t, pv in zip(model.param_names_, coef, se, tstat, pval):
            rows.append({
                'purpose':   purpose,
                'parameter': param,
                'value':     float(c),
                'std_err':   float(s)  if np.isfinite(float(s))  else float('nan'),
                'tstat':     float(t)  if np.isfinite(float(t))  else float('nan'),
                'pval':      float(pv) if np.isfinite(float(pv)) else float('nan'),
                'loglike':   float(model.ll_final_) if model.ll_final_ is not None else float('nan'),
                'rho2':      float(model.rho2_)     if model.rho2_     is not None else float('nan'),
                'n_obs':     model.n_obs_,
                'engine':    model._estimation_engine,
                'converged': model.converged_,
            })
    if rows:
        df_out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        df_out.to_csv(path, index=False)
        print(f"  Stage 6 coefficients saved → {path}")
        # HTML companion
        html_path = path.replace('.csv', '.html')
        _write_s6_html_report(models, html_path)


def load_models(path: str = 'saved_data/stage6_models.pkl') -> dict:
    with open(path, 'rb') as f:
        models = pickle.load(f)
    print(f"Stage 6 models loaded ← {path}")
    return models


# ---------------------------------------------------------------------------
# 7.  FULL PIPELINE WRAPPER
# ---------------------------------------------------------------------------

def run_stage6(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    synth_s5: pd.DataFrame,
    landuse: Optional[pd.DataFrame] = None,
    sample: bool = True,
    seed: int = 42,
    save_path: str = 'saved_data/stage6_models.pkl',
) -> tuple:
    """
    Estimate mode choice models on HHTS, predict for Stage 5 output.

    Parameters
    ----------
    trips, persons : from process_survey()
    synth_s5       : Stage 5 output — one row per activity with
                     OTAZ, DTAZ, STARTIME, purpose, tt_avg, dist,
                     distwalk, distbike, crash, vehrd
    landuse        : from sc.get_land() — used for transit accessibility
    sample         : sample from NL distribution (True recommended)
    seed           : random seed
    save_path      : where to pickle fitted models

    Returns
    -------
    activity_plan : complete activity plan DataFrame (one row per activity)
    models        : fitted NestedLogitModeModel dict

    Example
    -------
        sc       = Synth_Connect()
        out      = sc.process_survey(save=False)
        synth_s5 = pd.read_csv('saved_data/stage5_output.csv')
        landuse  = sc.get_land()

        activity_plan, models = run_stage6(
            out["trips"], out["persons"], synth_s5, landuse
        )
        activity_plan.to_csv('saved_data/stage6_output.csv', index=False)
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    print("=" * 55)
    print("STAGE 6 — Mode Choice")
    print("=" * 55)

    print("\n[1/3] Estimating mode models from HHTS trips...")
    models = estimate_mode_models(trips, persons, landuse)

    if save_path:
        save_models(models, save_path)

    print("\n[2/3] Predicting modes for synthetic population...")
    scheduled = predict_modes(synth_s5, models, landuse,
                              sample=sample, seed=seed)
    # Constrain each tour's legs to a single consistent tour mode (ActivitySim-
    # style: no stranded car / walk leg inside a transit tour, etc.).
    scheduled = enforce_tour_mode_consistency(scheduled)

    print("\n[3/3] Building final activity plan...")
    activity_plan = build_activity_plan(scheduled)

    # Save destinations-by-purpose HTML alongside the model pickle
    if save_path:
        _dest_html = save_path.replace('.pkl', '_destinations.html')
        try:
            _save_s6_destinations_html(activity_plan, _dest_html)
        except Exception as _de:
            logger.warning(f"Stage 6 destinations HTML failed: {_de}")

    print(
        f"\n{'='*55}\n"
        f"PIPELINE COMPLETE\n"
        f"{'='*55}\n"
        f"  {activity_plan['person_id'].nunique()} persons\n"
        f"  {len(activity_plan)} activities\n"
        f"  Purposes: " +
        ", ".join(
            f"{p}={int((activity_plan['purpose']==p).sum())}"
            for p in PURPOSES if (activity_plan['purpose'] == p).any()
        ) +
        f"\n  Modes: {activity_plan['mode'].value_counts().to_dict()}"
    )

    return activity_plan, models
