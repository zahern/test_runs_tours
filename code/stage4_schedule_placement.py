"""
stage4_schedule_placement.py
=============================
Stage 4 of the activity-based model pipeline.

Places each activity into a time slot within the person's day window
[start_of_day, end_of_day], assigning a STARTIME and ENDTIME to every
activity in the activity list from Stage 3.

Pipeline position
-----------------
    Stage 1: n_{purpose}_out
    Stage 2: start_of_day, end_of_day, B
    Stage 3: predicted_duration_min per activity
        → Stage 4: schedule placement     ← YOU ARE HERE
        → Stage 5: destination choice
        → Stage 6: mode choice

Approach
--------
Activities are placed sequentially within the day window using a
multinomial logit (MNL) time-of-day model estimated from the survey.

For each activity the model predicts which hourly time period it starts
in, conditional on:
  - purpose (work / edu / social / etc.)
  - ptype
  - position in day (first, middle, last)
  - time already consumed

After period choice, exact STARTIME is sampled uniformly within the
chosen period. ENDTIME = STARTIME + predicted_duration_min.

Activities are ordered by purpose priority before placement:
    work → edu → escort → shopping → othmaint → othdiscr → eat → social

Outputs
-------
activities DataFrame (Stage 3 output) with added columns:
    STARTIME  (minutes from midnight)
    ENDTIME   (minutes from midnight)
    time_period  ('am_peak', 'midday', 'pm_peak', 'evening', 'early')

Usage
-----
    from stage4_schedule_placement import run_stage4

    out         = Synth_Connect().process_survey(save=False)
    synth_s3    = pd.read_csv('saved_data/stage3_output.csv')

    scheduled, models = run_stage4(out["trips"], out["tours"],
                                   out["persons"], synth_s3)
    scheduled.to_csv(os.path.join(_OUT, 'stage4_output.csv'), index=False)
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
import sys
from typing import Optional
from scipy.stats import norm as _norm

logger = logging.getLogger(__name__)

_SEARCHLIBRIUM_SRC = os.environ.get(
    'SEARCHLIBRIUM_SOURCE', r'C:\Users\ahernz\source\SearchLibrium\src'
)
if os.path.isdir(_SEARCHLIBRIUM_SRC) and _SEARCHLIBRIUM_SRC not in sys.path:
    sys.path.insert(0, _SEARCHLIBRIUM_SRC)

_SEARCHLIB_OK = False
_RRM_OK = False
try:
    import io as _io, sys as _sys
    _sl_old = _sys.stdout; _sys.stdout = _io.StringIO()
    try:
        from SearchLibrium import MultinomialLogit as _SLMNL
        from SearchLibrium import RandomRegret as _SLRRM
    finally:
        _sys.stdout = _sl_old
    _SEARCHLIB_OK = True
    _RRM_OK = True
except Exception:
    _SLRRM = None

STAGE4_ESTIMATOR = os.environ.get('STAGE4_ESTIMATOR', 'scipy').strip().lower()
if STAGE4_ESTIMATOR == 'searchlibrium' and not _SEARCHLIB_OK:
    logger.warning("STAGE4_ESTIMATOR=searchlibrium but SearchLibrium not installed — falling back to scipy")
    STAGE4_ESTIMATOR = 'scipy'
elif STAGE4_ESTIMATOR not in ('scipy', 'searchlibrium', 'jaxopt'):
    logger.warning(f"Unknown STAGE4_ESTIMATOR='{STAGE4_ESTIMATOR}' — falling back to scipy")
    STAGE4_ESTIMATOR = 'scipy'


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Activity placement priority (lower = placed first)
PURPOSE_PRIORITY = {
    'work':      1,
    'edu':       2,
    'escort':    3,
    'shopping':  4,
    'othmaint':  5,
    'othdiscr':  6,
    'eat':       7,
    'social':    8,
}

# Time periods — defined as (label, start_min, end_min)
TIME_PERIODS = [
    ('early',    240,  420),   # 04:00–07:00
    ('am_peak',  420,  540),   # 07:00–09:00
    ('midday',   540,  840),   # 09:00–14:00
    ('pm_peak',  840,  1020),  # 14:00–17:00
    ('evening',  1020, 1320),  # 17:00–22:00
    ('late',     1320, 1440),  # 22:00–24:00
]

PERIOD_LABELS  = [p[0] for p in TIME_PERIODS]
PERIOD_STARTS  = np.array([p[1] for p in TIME_PERIODS])
PERIOD_ENDS    = np.array([p[2] for p in TIME_PERIODS])

# Minimum gap between activities (minutes)
MIN_TRAVEL_GAP = 10

# ── Joint time-of-day-departure (tdd) alternative set ───────────────────────
# ActivitySim mandatory_tour_scheduling models a JOINT (start, end, duration)
# choice. Here the alternatives are all (start_period, end_period) pairs with
# start ≤ end over the 6 TIME_PERIODS → 21 tdd alternatives, each with a derived
# duration. Selected via STAGE4_TIMING_MODEL=tdd (see TDDChoiceModel).
_PERIOD_CENTERS = np.array([(s + e) / 2.0 for _, s, e in TIME_PERIODS])  # minutes


def _build_tdd_alts():
    """(start_period_idx, end_period_idx, duration_hours) for start ≤ end pairs."""
    alts = []
    for i in range(len(TIME_PERIODS)):
        for j in range(i, len(TIME_PERIODS)):
            dur_hr = max((_PERIOD_CENTERS[j] - _PERIOD_CENTERS[i]) / 60.0, 0.5)
            alts.append((i, j, float(dur_hr)))
    return alts


TDD_ALTS = _build_tdd_alts()          # 21 (start, end, duration) alternatives
_TDD_START = np.array([a[0] for a in TDD_ALTS])
_TDD_END   = np.array([a[1] for a in TDD_ALTS])
_TDD_DUR   = np.array([a[2] for a in TDD_ALTS])
_N_PERIODS = len(TIME_PERIODS)
# map a period index → its commute-LOS column (mandatory-scheduling keystone)
_PERIOD_LOS_COL = {0: 'commute_am', 1: 'commute_am', 2: 'commute_md',
                   3: 'commute_pm', 4: 'commute_ev', 5: 'commute_ev'}

PURPOSES = list(PURPOSE_PRIORITY.keys())

PERSON_FEATURES = ['sexp', 'vehrd', 'nprd']

# Home-zone landuse accessibility columns — merged from persons if available
ACC_FEATURES = [
    'auto_work_acc', 'auto_edu_acc', 'auto_shop_acc', 'auto_nonmand_acc',
    'walk_acc', 'bike_acc', 'transit_acc',
]

DEFAULT_FEATURES_BY_PURPOSE = {
    'work': ['CARLICENCE', 'vehrd', 'ptype_FullTimeWorker', 'ptype_WorkingStudent', 'wfh', 'auto_work_acc', 'transit_acc',
             'job_fulltime', 'has_work_anchor', 'same_zone_work', 'occp_prof',
             'cdap_M', 'cdap_hh_n_children', 'is_joint_traveller'],
    'edu': ['ptype_SchoolChild', 'ptype_UniversityStudent', 'age_child', 'age_teen', 'transit_acc', 'walk_acc', 'auto_edu_acc',
            'has_school_anchor', 'same_zone_school', 'cdap_M', 'cdap_is_child'],
    'escort': ['hh_child', 'vehrd', 'has_vehicle', 'nprd', 'auto_nonmand_acc', 'walk_acc', 'has_school_anchor',
               'escort_duty', 'pickup_anchor', 'is_joint_act', 'joint_party_n'],
    'shopping': ['incp', 'vehrd', 'has_vehicle', 'nprd', 'walk_acc', 'auto_shop_acc', 'job_fulltime', 'is_joint_act'],
    'social': ['is_female', 'nprd', 'age_teen', 'age_older', 'walk_acc', 'transit_acc', 'wfh', 'is_joint_act', 'joint_party_n'],
    'othmaint': ['vehrd', 'nprd', 'auto_nonmand_acc', 'walk_acc', 'has_work_anchor', 'is_joint_act'],
    'othdiscr': ['incp', 'vehrd', 'age_older', 'walk_acc', 'auto_nonmand_acc', 'job_fulltime', 'is_joint_act', 'joint_party_n'],
    'eat': ['incp', 'nprd', 'walk_acc', 'transit_acc', 'job_fulltime', 'is_joint_act', 'joint_party_n'],
}


def _attach_origin_job_features(df: pd.DataFrame) -> pd.DataFrame:
    """Origin-wise + job-wise timing conditionals (all safe, col-gated).

    Creates 0/1 flags only from columns already present — never raises for
    missing inputs. Conditioning timing on home origin (OTAZ), usual
    job/school anchors (WORK_TAZ/SCHOOL_TAZ) and job type (WORKSTAT /
    occp_group / INDUSTRY / wfh) makes work/edu departure peaks sound:
    fixed-workplace full-timers anchor to am_peak, wfh/part-time spread out.
    """
    try:
        if 'OTAZ' in df.columns and 'WORK_TAZ' in df.columns:
            _o = pd.to_numeric(df['OTAZ'], errors='coerce')
            _w = pd.to_numeric(df['WORK_TAZ'], errors='coerce')
            df['has_work_anchor'] = ((_w > 0) & _w.notna()).astype(float)
            df['same_zone_work'] = ((_o == _w) & _o.notna() & _w.notna()).astype(float)
        if 'OTAZ' in df.columns and 'SCHOOL_TAZ' in df.columns:
            _o = pd.to_numeric(df['OTAZ'], errors='coerce')
            _s = pd.to_numeric(df['SCHOOL_TAZ'], errors='coerce')
            df['has_school_anchor'] = ((_s > 0) & _s.notna()).astype(float)
            df['same_zone_school'] = ((_o == _s) & _o.notna() & _s.notna()).astype(float)
        if 'WORKSTAT' in df.columns:
            _ws = pd.to_numeric(df['WORKSTAT'], errors='coerce').fillna(0)
            df['job_fulltime'] = (_ws == 1).astype(float)
            df['job_parttime'] = (_ws == 2).astype(float)
            df['is_worker'] = _ws.isin([1, 2]).astype(float)
        if 'occp_group' in df.columns:
            _oc = df['occp_group'].astype(str).str.lower()
            df['occp_prof'] = _oc.str.contains('prof|manag|techn', regex=True, na=False).astype(float)
            df['occp_blue'] = _oc.str.contains('labour|machine|trade|driver|service', regex=True, na=False).astype(float)
        if 'wfh' not in df.columns and 'WFHWEEK' in df.columns:
            df['wfh'] = (pd.to_numeric(df['WFHWEEK'], errors='coerce').fillna(0) > 0).astype(float)
    except Exception as e:
        logger.warning('_attach_origin_job_features failed (non-fatal): %s', e)
    return df


def _timing_person_features(df: pd.DataFrame, purpose: str, feature_override=None) -> list:
    if isinstance(feature_override, dict):
        names = feature_override.get(purpose)
        if names is None and purpose in {'escort', 'shopping', 'social',
                                         'othmaint', 'othdiscr', 'eat'}:
            names = feature_override.get('nonmand')
        if names is None:
            names = feature_override.get('*', [])
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


# ---------------------------------------------------------------------------
# 0.  HELPERS
# ---------------------------------------------------------------------------

def _period_of(time_min: float) -> int:
    """Return period index for a time in minutes from midnight."""
    for i, (_, s, e) in enumerate(TIME_PERIODS):
        if s <= time_min < e:
            return i
    return len(TIME_PERIODS) - 1


def _period_label(time_min: float) -> str:
    return PERIOD_LABELS[_period_of(time_min)]


def _encode_purpose(purpose: str) -> np.ndarray:
    """One-hot encode purpose."""
    vec = np.zeros(len(PURPOSES))
    if purpose in PURPOSES:
        vec[PURPOSES.index(purpose)] = 1.0
    return vec


def _encode_period(period_idx: int) -> np.ndarray:
    """One-hot encode time period."""
    vec = np.zeros(len(TIME_PERIODS))
    vec[period_idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# ActivitySim mandatory-tour-scheduling keystone: time-period-varying commute LOS
# ---------------------------------------------------------------------------
_MANDATORY_LOS_COLS = ['commute_am', 'commute_md', 'commute_pm', 'commute_ev',
                       'commute_peak_ratio']


def _attach_mandatory_los(df: pd.DataFrame) -> pd.DataFrame:
    """Attach period-varying auto commute LOS for work/school (mandatory) tours.

    ActivitySim's ``mandatory_tour_scheduling`` keystone: because the destination
    of a work/school tour is the KNOWN usual location, the schedule can respond to
    time-period congestion — a long PM-peak commute pushes departure earlier/later.
    For each work/edu row we look up the home→usual-location auto time in each
    period (SOVTOLL_TIME__AM/MD/PM/EV) and expose:
        commute_am/md/pm/ev   – minutes by period
        commute_peak_ratio    – PM-peak / midday (congestion pressure)
    These become GA-searchable scheduling covariates (respecting the metaheuristic
    — see ga_hypothesis_testing _S4_PURPOSE_OPTIONAL work/edu). Gated by
    STAGE4_MANDATORY_LOS; leaves the columns at 0 on any failure (safe no-op), so
    it never breaks a run when skims/OD are unavailable. Needs OTAZ + WORK_TAZ/
    SCHOOL_TAZ in df (merged by prepare_timing_data) and the period skims in the OMX.
    """
    for _c in _MANDATORY_LOS_COLS:
        if _c not in df.columns:
            df[_c] = np.float32(0.0)
    if os.environ.get('STAGE4_MANDATORY_LOS', '0') != '1':
        return df
    try:
        if 'OTAZ' not in df.columns or 'purpose' not in df.columns:
            return df
        _o = pd.to_numeric(df['OTAZ'], errors='coerce')
        _d = pd.to_numeric(df.get('WORK_TAZ', pd.Series(np.nan, index=df.index)), errors='coerce')
        if 'SCHOOL_TAZ' in df.columns:
            _d = _d.where(_d > 0, pd.to_numeric(df['SCHOOL_TAZ'], errors='coerce'))
        _mand = df['purpose'].isin(['work', 'edu'])
        _valid = _mand & _o.notna() & _d.notna() & (_d > 0)
        if int(_valid.sum()) == 0:
            logger.info("  _attach_mandatory_los: no work/edu rows with OTAZ+usual-location; skipped")
            return df
        from stage5_destination_choice import _open_skims, _skim_flat
        _sk_obj, _close = _open_skims(os.environ.get('PIPELINE_SKIMS_PATH'))
        _sk = _skim_flat(_sk_obj)
        if _close:
            try:
                _sk_obj.close()
            except Exception:
                pass
        # one (OTAZ,DTAZ) → row map; look up each query pair
        _pos = pd.Series(np.arange(len(_sk)),
                         index=pd.MultiIndex.from_arrays(
                             [_sk['OTAZ'].astype(int), _sk['DTAZ'].astype(int)]))
        _qk = pd.MultiIndex.from_arrays([_o.fillna(-1).astype(int), _d.fillna(-1).astype(int)])
        _row = _pos.reindex(_qk).to_numpy()
        _ok = np.isfinite(_row)
        _periods = {'commute_am': 'SOVTOLL_TIME__AM', 'commute_md': 'SOVTOLL_TIME__MD',
                    'commute_pm': 'SOVTOLL_TIME__PM', 'commute_ev': 'SOVTOLL_TIME__EV'}
        _mand_np = _mand.to_numpy()
        for _col, _skcol in _periods.items():
            if _skcol not in _sk.columns:
                continue
            _sv = _sk[_skcol].to_numpy()
            _t = np.zeros(len(df), dtype=np.float64)
            _t[_ok] = _sv[_row[_ok].astype(int)]
            _pos_t = _t[_t > 0]
            if _pos_t.size and float(np.median(_pos_t)) > 600:   # seconds → minutes
                _t = _t / 60.0
            df[_col] = np.where(_mand_np, _t, 0.0).astype(np.float32)
        _md = df['commute_md'].replace(0, np.nan)
        df['commute_peak_ratio'] = np.where(
            _mand_np, (df['commute_pm'] / _md).fillna(1.0).clip(0, 5), 0.0).astype(np.float32)
        logger.info(f"  _attach_mandatory_los: period commute LOS attached for "
                    f"{int(_valid.sum())} work/edu tours "
                    f"(pm/md ratio μ={df.loc[_valid, 'commute_peak_ratio'].mean():.2f})")
    except Exception as _e:
        logger.debug(f"_attach_mandatory_los skipped ({_e})")
    return df


def _attach_joint_features(df: pd.DataFrame, tours: Optional[pd.DataFrame] = None,
                         persons: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Joint-travel conditionals for timing (pickup / travelling together).

    Estimation: needs to know who travels together. HHTS joint tours come
    from cdap_households_estimation.derive_observed_joint_tours (flag column
    or HHID+OD+time heuristic); prediction frames already carry
    is_joint / joint_tour_id / joint_participants tagged from the joint map
    (pipeline_logger.predict_stage3 via tag_joint_activities).

    Creates 0/1 flags only from columns present — never raises, and every
    flag column is ALWAYS created (zeros when there is no evidence) so
    downstream code (GA candidates, timing features) never KeyErrors:
      is_joint_act     this activity is a shared household tour
      joint_party_n    party size (from joint_participants list length)
      escort_duty      escort-purpose row in a household with children
      pickup_anchor    escort row with school-pickup duty: the row's school
                       anchor when known (derived inline from SCHOOL_TAZ if
                       has_school_anchor is absent), else a child in the
                       household as the school-run proxy
      joint_worker     joint row by a worker (tight peak coupling)
    """
    try:
        if 'is_joint' not in df.columns or df['is_joint'].isna().all():
            _jt = None
            try:
                from cdap_households_estimation import derive_observed_joint_tours
                if tours is not None and len(tours):
                    _jt = derive_observed_joint_tours(tours)
            except Exception:
                _jt = None
            if _jt is not None and len(_jt) and 'participants' in _jt.columns:
                _members = set()
                for _lst in _jt['participants'].dropna():
                    _l = _lst if isinstance(_lst, (list, tuple)) else [ _lst ]
                    for _p in _l:
                        _members.add(str(_p))
                if 'person_id' in df.columns and _members:
                    df['is_joint_act'] = df['person_id'].astype(str).isin(_members).astype(float)
        else:
            df['is_joint_act'] = pd.to_numeric(df['is_joint'], errors='coerce').fillna(0).clip(0, 1).astype(float)
        # Contract: the flag must always exist downstream (GA candidates,
        # timing features). Default 0 when no joint evidence was found.
        if 'is_joint_act' not in df.columns:
            df['is_joint_act'] = 0.0
        if 'joint_party_n' not in df.columns:
            if 'joint_participants' in df.columns:
                def _plen(v):
                    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ('', 'nan'):
                        return 0.0
                    s = str(v)
                    return float(len([x for x in s.replace('|', ',').split(',') if x.strip()]) or 1.0)
                df['joint_party_n'] = df['joint_participants'].map(_plen).astype(float)
            else:
                df['joint_party_n'] = df.get('is_joint_act', 0.0)
        _has_child = None
        for _cc in ('hh_child', 'n_children', 'has_child'):
            if _cc in df.columns:
                _has_child = pd.to_numeric(df[_cc], errors='coerce').fillna(0) > 0
                break
        _is_escort = (df['purpose'] == 'escort') if 'purpose' in df.columns else False
        if _has_child is not None:
            df['escort_duty'] = ((_is_escort) & _has_child).astype(float)
        elif 'purpose' in df.columns:
            df['escort_duty'] = (_is_escort).astype(float)
        else:
            df['escort_duty'] = 0.0
        # School anchor: derive inline when absent — callers may run joint
        # features before _attach_origin_job_features (which normally creates
        # has_school_anchor from SCHOOL_TAZ).
        if 'has_school_anchor' not in df.columns and 'SCHOOL_TAZ' in df.columns:
            _s = pd.to_numeric(df['SCHOOL_TAZ'], errors='coerce')
            df['has_school_anchor'] = ((_s > 0) & _s.notna()).astype(float)
        # pickup_anchor: escort trip with school-pickup duty — the row's school
        # anchor when known, else a child in the household as the school-run
        # proxy. Always created (zeros when no evidence) so downstream code
        # never KeyErrors.
        if 'has_school_anchor' in df.columns:
            _school = pd.to_numeric(df['has_school_anchor'], errors='coerce').fillna(0) > 0
            if _has_child is not None:
                df['pickup_anchor'] = ((_is_escort) & (_school | _has_child)).astype(float)
            else:
                df['pickup_anchor'] = ((_is_escort) & _school).astype(float)
        elif _has_child is not None:
            df['pickup_anchor'] = ((_is_escort) & _has_child).astype(float)
        else:
            df['pickup_anchor'] = 0.0
        if 'is_joint_act' in df.columns and 'WORKSTAT' in df.columns:
            _ws = pd.to_numeric(df['WORKSTAT'], errors='coerce').fillna(0)
            df['joint_worker'] = ((df['is_joint_act'] > 0) & _ws.isin([1, 2])).astype(float)
        elif 'joint_worker' not in df.columns:
            df['joint_worker'] = 0.0
    except Exception as e:
        logger.warning('_attach_joint_features failed (non-fatal): %s', e)
    return df


# ---------------------------------------------------------------------------
# 1.  PREPARE ESTIMATION DATA FROM SURVEY TRIPS
# ---------------------------------------------------------------------------

def prepare_timing_data(
    trips: pd.DataFrame,
    tours: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build estimation dataset: one row per trip/activity with
    STARTIME period as the outcome variable.

    Parameters
    ----------
    trips   : from process_survey()["trips"]
    tours   : from process_survey()["tours"]
    persons : from process_survey()["persons"]

    Returns
    -------
    DataFrame with columns: purpose, period (int), period_label,
    STARTIME, duration_min, day_position, time_consumed,
    person attributes.
    """
    df = trips.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # deduplicate columns — COLUMN_MAP_R may have already created
    # a 'purpose' col from OVERALL_PURPOSE; DESTPURP rename would then
    # produce two 'purpose' columns, making df['purpose'] return a DataFrame
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]

    # resolve purpose column: prefer DESTPURP, fall back to existing 'purpose'
    if 'DESTPURP' in df.columns:
        if 'purpose' in df.columns:
            df = df.drop(columns=['purpose'])
        df = df.rename(columns={'DESTPURP': 'purpose'})
    elif 'purpose' not in df.columns:
        raise ValueError('No purpose column found. Expected DESTPURP or purpose.')

    # map raw survey strings to codes if needed
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
        'Pick up or drop off a passenger': 'escort',
        'Pickup/Dropoff Someone': 'escort', 'Just accompanying someone': 'escort',
        'Shopping': 'shopping',
        'Personal Business': 'othmaint', 'Personal business': 'othmaint',
        'Pickup/deliver something (not work-related)': 'othmaint',
        'Other place (please specify)': 'othdiscr', 'Other Place': 'othdiscr',
        'Refuel my vehicle': 'othdiscr', 'Other Purpose': 'othdiscr',
        'Eat/drink': 'eat',
    }
    unique_vals = df['purpose'].dropna().unique()
    if not set(unique_vals).issubset(set(PURPOSES) | {'home'}):
        df['purpose'] = df['purpose'].map(raw_to_code).fillna(df['purpose'])
        # token fallback for raw HHTS strings missing from the exact map
        _miss = ~df['purpose'].isin(set(PURPOSES) | {'home'}) & df['purpose'].notna()
        if _miss.any():
            def _token_code(v):
                s = str(v).strip().lower()
                if not s or s in ('nan', 'none'):
                    return None
                if 'home' in s:
                    return 'home'
                if 'work' in s:
                    return 'work'
                if any(t in s for t in ('school', 'tafe', 'univers',
                                        'kinder', 'childcare', 'educat')):
                    return 'edu'
                if 'eat' in s or 'drink' in s or 'dine' in s:
                    return 'eat'
                if 'shop' in s:
                    return 'shopping'
                if 'social' in s or 'recreation' in s or 'visit' in s \
                        or 'leisure' in s:
                    return 'social'
                if 'escort' in s or 'pickup' in s or 'dropoff' in s \
                        or 'accompan' in s:
                    return 'escort'
                if 'business' in s or 'personal' in s:
                    return 'othmaint'
                return 'othdiscr'
            df.loc[_miss, 'purpose'] = df.loc[_miss, 'purpose'].map(_token_code)
            logger.info(f"  stage4 token fallback coded "
                        f"{int(df.loc[_miss, 'purpose'].isin(PURPOSES).sum())}"
                        f"/{int(_miss.sum())} raw purposes")

    # keep only trips with valid times and known purposes
    df = df[df['STARTIME'].notna() & df['purpose'].notna()].copy()
    df = df[df['purpose'].isin(PURPOSES)].copy()

    # duration in minutes
    if 'DURATION' in df.columns:
        df['duration_min'] = df['DURATION'].clip(1, 600)
    elif 'TRAVTIME' in df.columns:
        df['duration_min'] = df['TRAVTIME'].clip(1, 600)
    else:
        df['duration_min'] = 30

    # time period of departure
    df['period']       = df['STARTIME'].apply(_period_of)
    df['period_label'] = df['STARTIME'].apply(_period_label)

    # day-level context
    day_ctx = (
        df.groupby('person_id')
        .agg(day_start=('STARTIME', 'min'),
             day_end=('STARTIME', 'max'),
             n_trips=('trip_id', 'count'))
        .reset_index()
    )
    df = df.merge(day_ctx, on='person_id', how='left')

    # position in day (0=first, 1=last, 0.5=middle)
    df = df.sort_values(['person_id', 'STARTIME'])
    df['trip_rank'] = df.groupby('person_id').cumcount()
    df['day_position'] = df['trip_rank'] / df['n_trips'].clip(lower=1)

    # time already consumed by previous activities
    df['time_consumed'] = (
        df.groupby('person_id')['duration_min']
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    df['time_remaining'] = (
        (df['day_end'] - df['STARTIME']).clip(lower=0)
    )

    # person attributes
    pers = persons.copy()
    if 'person_id' not in pers.columns and 'PERSID' in pers.columns:
        pers = pers.rename(columns={'PERSID': 'person_id'})

    try:
        from stage1_activity_frequency import assign_ptype_column
        pers = assign_ptype_column(pers)
    except ImportError:
        pers['ptype'] = 'NonWorker'

    _extra_pers = [c for c in ['AGEGROUP'] + ACC_FEATURES if c in pers.columns]
    # Usual-location OD for the mandatory-scheduling LOS keystone (only merged
    # from persons when not already on the trips frame).
    _od_pers = [c for c in ('OTAZ', 'WORK_TAZ', 'SCHOOL_TAZ')
                if c in pers.columns and c not in df.columns]
    # Origin/job conditionals: carry job + origin attributes so timing can be
    # conditioned on where people live/work and what job they hold. Only merged
    # when present — _timing_person_features filters to df.columns, so absent
    # cols are safely ignored at fit/predict.
    _job_pers = [c for c in ('WORKSTAT', 'STUDYING', 'occp_group', 'INDUSTRY',
                             'wfh', 'WFHWEEK', 'incp', 'CARLICENCE')
                 if c in pers.columns and c not in df.columns]
    # Observed-CDAP conditionals (attached in process_survey via
    # attach_cdap_to_hhts): household coordination signal for timing —
    # DAP dummies, hh composition, joint-tour counts. Same filter applies.
    _cdap_pers = [c for c in ('cdap_M', 'cdap_N', 'cdap_H',
                              'cdap_hh_n_workers', 'cdap_hh_n_children',
                              'cdap_hh_has_home_adult',
                              'cdap_child_x_home_adult', 'cdap_is_child',
                              'n_joint_tours_hh', 'is_joint_traveller')
                  if c in pers.columns and c not in df.columns]
    keep = (['person_id', 'ptype'] +
            [c for c in PERSON_FEATURES if c in pers.columns] +
            _extra_pers + _od_pers + _job_pers + _cdap_pers)
    df = df.merge(pers[keep], on='person_id', how='left')
    # Origin/job-derived timing conditionals (safe: only from cols present).
    df = _attach_origin_job_features(df)
    # Joint-travel conditionals: who travels together / pickup duty. Uses HHTS
    # tours on the estimation path, tagged is_joint cols on predict frames.
    try:
        df = _attach_joint_features(df, tours=tours, persons=persons)
    except Exception:
        df = _attach_joint_features(df)
    # ActivitySim mandatory-scheduling keystone: period-varying commute LOS
    # (gated STAGE4_MANDATORY_LOS; safe no-op otherwise).
    df = _attach_mandatory_los(df)

    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2, 'WORKSTAT': 0, 'STUDYING': 0}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    df['ptype'] = df['ptype'].fillna('NonWorker')

    # Reclassify adult edu travellers with STUDYING=0 → 3 (post-secondary/TAFE/Uni).
    # HHTS codes many adult students as STUDYING=0 (none/unclear).  When their
    # purpose is Education and age band >= 4 (18+), they are almost certainly
    # tertiary students.  Without this fix the edu timing model trains on a
    # mixture of "school-age" and "adult" departure patterns labelled identically,
    # degrading fit.  Matches the same reclassification applied in stage5 _dest_tree.
    if 'AGEGROUP' in df.columns:
        _adult_edu = (df['purpose'] == 'edu') & (df['STUDYING'] == 0) & (df['AGEGROUP'] >= 4)
        if _adult_edu.any():
            n_reclassed = int(_adult_edu.sum())
            df.loc[_adult_edu, 'STUDYING'] = 3
            logger.info(
                f"  prepare_timing_data: reclassified {n_reclassed} adult edu trips "
                "from STUDYING=0 → 3 (post-secondary)"
            )

    logger.info(f"Timing dataset: {len(df)} trip observations")
    return df


# ---------------------------------------------------------------------------
# 2.  MNL TIME-OF-DAY MODEL
# ---------------------------------------------------------------------------

class MNLTimingModel:
    """
    Multinomial logit model for time period choice.

    Alternatives : TIME_PERIODS (6 periods)
    Features     : purpose dummies, ptype dummies, day_position,
                   time_consumed, duration_min, person attributes
    """

    PTYPE_ALLOWLIST: dict = {
        'work':     {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent'},
        'edu':      {'SchoolChild', 'UniversityStudent', 'WorkingStudent', 'Preschool'},
        'escort':   None,
        'shopping': None,
        'social':   None,
        'othmaint': None,
        'othdiscr': None,
        'eat':      None,
    }

    def __init__(self):
        self.coef_          = None   # (n_periods, n_features)
        self.feature_names_ = None
        self.ptypes_        = None
        self.feature_override_ = None   # set externally to override PERSON_FEATURES
        self.mu_            = None
        self.sd_            = None
        self._const_mask    = np.array([], dtype=bool)   # set by _fit_scipy
        self.n_periods_     = len(TIME_PERIODS)
        # Ridge on the (z-scored) coefficients, EMBEDDED in _neg_ll so the fit
        # AND the numerical Hessian used for std_err/p_value both carry the
        # ≈λ eigenvalue lift — keeps inference finite under collinearity
        # (mirrors SearchLibrium's l2_penalty on z-scored coefficients).
        self.ridge          = float(os.environ.get('GA_S4_RIDGE', '0.5'))
        self.coef_export_names_ = None
        self.coef_export_values_ = None
        self.std_err_export_ = None
        self.t_stat_export_ = None
        self.p_value_export_ = None
        self.within_period_draws_ = {}
        self.observed_period_shares_ = None
        self.period_blend_alpha_ = 0.0
        self.include_person_features_ = True
        self.fit_status_ = 'not_fit'
        self.purpose_ = None
        # Purpose dummies actually kept for this model (the ones that VARY in the
        # fit data). Decided once at fit-time in _build_X and reused at predict so
        # the design is consistent. None => not yet computed.
        self.kept_purposes_ = None

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix for MNL estimation / prediction.

        The fit-time design is locked on first call (kept_purposes_ plus the
        column order below); later calls align to that locked design with
        zeros for missing columns instead of silently dropping them (a
        dropped middle column used to shift every later column left against
        coef_, and a fully-missed purpose collapsed to the default period).
        """
        arrays = []
        cols   = []

        # purpose dummies — keep only purposes that VARY in the fit data. A
        # per-purpose segment (e.g. a work-tour timing model) has no shopping
        # activities, so purp_shopping is all-zero -> a holdfast-0 junk term.
        # Building only the informative dummies removes those "_other purpose"
        # coefficients the user flagged. The kept set is decided once (first
        # _build_X call = fit) and reused at predict for a consistent design.
        if self.kept_purposes_ is None:
            _kp = []
            for p in PURPOSES:
                _c = (df['purpose'] == p).astype(float).values
                if _c.size and _c.min() != _c.max():   # non-constant => informative
                    _kp.append(p)
            self.kept_purposes_ = _kp if _kp else (
                [self.purpose_] if self.purpose_ in PURPOSES else list(PURPOSES))
        for p in self.kept_purposes_:
            arrays.append((df['purpose'] == p).astype(float).values)
            cols.append(f'purp_{p}')

        # ptype dummies  — filter by purpose to avoid nonsensical predictors
        # (e.g. Preschool in the work timing model, FullTimeWorker in edu)
        allowed = self.PTYPE_ALLOWLIST.get(self.purpose_, None)
        ptype_pool = (self.ptypes_ or [])
        if allowed is not None:
            ptype_pool = [pt for pt in ptype_pool if pt in allowed]
        for pt in ptype_pool:
            arrays.append((df['ptype'] == pt).astype(float).values)
            cols.append(f'ptype_{pt}')

        # continuous features
        feature_list = (self.feature_override_ or []) + ['day_position',
                        'time_consumed', 'duration_min', 'time_remaining']
        if self.include_person_features_:
            feature_list = feature_list + PERSON_FEATURES
        feature_list = list(dict.fromkeys(feature_list))  # deduplicate, preserve order
        _avail = {}
        for col in feature_list:
            if col in df.columns:
                _avail[col] = pd.to_numeric(
                    df[col], errors='coerce').fillna(0).values
        if self.feature_names_ is None:
            # fit call: lock the design actually built (purp/ptype dummies
            # first, in the fixed order above, then present features).
            self.feature_names_ = cols + [c for c in feature_list if c in _avail]
        for col in self.feature_names_:
            if col in cols:
                continue  # dummy already appended above
            arrays.append(_avail[col] if col in _avail
                          else np.zeros(len(df), dtype=float))
            cols.append(col)

        return np.column_stack(arrays).astype(float)

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        """Row-wise softmax. Z shape (N, K)."""
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def _neg_ll(self, coef_flat: np.ndarray,
                X: np.ndarray, y: np.ndarray) -> float:
        K = self.n_periods_
        F = X.shape[1]
        # coef shape (K-1, F) — base period fixed at 0
        coef = coef_flat.reshape(K - 1, F)
        coef_full = np.vstack([np.zeros((1, F)), coef])
        Z    = X @ coef_full.T           # (N, K)
        prob = self._softmax(Z)
        ll   = np.sum(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)))
        # embedded ridge: lifts every Hessian eigenvalue by ≈2·self.ridge so the
        # inference step stays invertible on flat/collinear directions
        ll -= 0.5 * self.ridge * float(np.dot(coef_flat, coef_flat))
        return -ll

    def _neg_ll_grad(self, coef_flat: np.ndarray,
                     X: np.ndarray, y: np.ndarray) -> np.ndarray:
        K = self.n_periods_
        F = X.shape[1]
        N = len(y)
        coef = coef_flat.reshape(K - 1, F)
        coef_full = np.vstack([np.zeros((1, F)), coef])
        Z    = X @ coef_full.T
        prob = self._softmax(Z)

        # indicator matrix
        ind = np.zeros((N, K))
        ind[np.arange(N), y] = 1.0

        # gradient w.r.t. coef_full: X.T @ (prob - ind)  shape (F, K)
        grad_full = X.T @ (prob - ind)    # (F, K)
        # drop gradient for base period (first column)
        grad = grad_full[:, 1:].T         # (K-1, F)
        return grad.ravel()

    def fit(self, X: np.ndarray, y: np.ndarray):
        if STAGE4_ESTIMATOR == 'searchlibrium' and _SEARCHLIB_OK:
            self._fit_searchlibrium(X, y)
        elif STAGE4_ESTIMATOR == 'jaxopt':
            self._fit_jaxopt(X, y)
        else:
            self._fit_scipy(X, y)

        ll_val = float(self.ll_)
        rho2   = float(1.0 - ll_val / self.null_ll_) if self.null_ll_ < 0 else float('nan')

        self.rho2_     = rho2
        self.converged_ = self.converged_ or (np.isfinite(rho2) and rho2 > 0.05)
        self.fit_status_ = 'ok' if self.converged_ else 'optimizer_stopped'

        # ── Numerical Hessian → standard errors ──────────────────────────
        try:
            _n_feats = len(self.feature_names_ or [])
            _K = self.n_periods_
            _all_coef_flat = np.zeros((_K - 1) * _n_feats)
            if self.coef_ is not None:
                _all_coef_flat[:] = self.coef_.ravel()
            if _n_feats > 0 and np.isfinite(_all_coef_flat).all():

                def _neg_ll_wrapper(coef_flat):
                    return self._neg_ll(coef_flat, X, y)

                _hess = _numerical_hessian(_neg_ll_wrapper, _all_coef_flat)
                _se   = np.sqrt(np.diag(np.linalg.pinv(_hess)))
                _se   = np.where(np.isfinite(_se) & (_se > 1e-10), _se, float('nan'))
                _t    = _all_coef_flat / _se

                # Export friendly: one row per (alt, feature) pair
                export_names  = []
                export_values = []
                for alt_idx in range(1, _K):
                    for feat_idx, fn in enumerate(self.feature_names_ or []):
                        i = (alt_idx - 1) * _n_feats + feat_idx
                        export_names.append(f'{PERIOD_LABELS[alt_idx]}::{fn}')
                        export_values.append(_all_coef_flat[i])
                self.coef_export_names_  = export_names
                self.coef_export_values_ = export_values
                self.std_err_export_ = np.where(np.isfinite(_se), _se, float('nan'))
                self.t_stat_export_  = np.where(np.isfinite(_t),  _t,  float('nan'))
                self.p_value_export_ = 2.0 * (1.0 - _norm.cdf(np.abs(self.t_stat_export_)))
        except Exception as _hess_exc:
            logger.debug(f"  timing/{self.purpose_}: hessian/export failed: {_hess_exc}")

    def _fit_scipy(self, X: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self._const_mask = (self.sd_ == 0)
        self.sd_[self._const_mask] = 1.0
        Xs = np.where(self._const_mask, X, (X - self.mu_) / self.sd_)

        K = self.n_periods_
        F = Xs.shape[1]

        period_counts = np.bincount(y, minlength=K).astype(float)
        period_counts = np.maximum(period_counts, 1.0)
        self.observed_period_shares_ = (period_counts / period_counts.sum()).astype(float)
        log_shares    = np.log(period_counts / period_counts.sum())
        init_intercepts = log_shares[1:] - log_shares[0]

        init = np.zeros((K - 1) * F)
        for k in range(K - 1):
            init[k * F] = init_intercepts[k]

        # SAME embedded ridge as self._neg_ll (used by the Hessian for
        # std_err/p_value) — fit and inference must differentiate the same
        # penalised objective or flat directions stay flat.
        L2 = float(self.ridge)

        def neg_ll_reg(coef_flat):
            coef = coef_flat.reshape(K - 1, F)
            coef_full = np.vstack([np.zeros((1, F)), coef])
            Z    = Xs @ coef_full.T
            Z   -= Z.max(axis=1, keepdims=True)
            E    = np.exp(Z)
            prob = E / E.sum(axis=1, keepdims=True)
            ll   = np.sum(np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)))
            ll  -= 0.5 * L2 * np.dot(coef_flat, coef_flat)
            return -ll

        def neg_ll_grad(coef_flat):
            coef = coef_flat.reshape(K - 1, F)
            coef_full = np.vstack([np.zeros((1, F)), coef])
            Z    = Xs @ coef_full.T
            Z   -= Z.max(axis=1, keepdims=True)
            E    = np.exp(Z)
            prob = E / E.sum(axis=1, keepdims=True)
            N    = len(y)
            ind  = np.zeros((N, K)); ind[np.arange(N), y] = 1.0
            grad_full = Xs.T @ (prob - ind)
            grad      = grad_full[:, 1:].T
            return grad.ravel() + L2 * coef_flat

        res = minimize(
            neg_ll_reg, init, jac=neg_ll_grad,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-10},
        )

        if not res.success and len(y) < 500:
            res2 = minimize(
                neg_ll_reg, res.x, jac=neg_ll_grad,
                method='CG',
                options={'maxiter': 3000, 'gtol': 1e-7},
            )
            if -res2.fun < -res.fun:
                res = res2

        null_ll = float(np.sum(np.log(period_counts / period_counts.sum())[y]))
        ll_val  = float(-res.fun)

        self.coef_     = res.x.reshape(K - 1, F)
        self.ll_       = ll_val
        self.null_ll_  = null_ll
        self.n_obs_    = len(y)
        self.converged_ = bool(res.success)

    def _fit_jaxopt(self, X: np.ndarray, y: np.ndarray):
        try:
            import jax.numpy as jnp
            import jax
        except ImportError:
            self._fit_scipy(X, y)
            return

        K = self.n_periods_
        F = X.shape[1]
        period_counts = np.bincount(y, minlength=K).astype(float)
        period_counts = np.maximum(period_counts, 1.0)
        self.observed_period_shares_ = (period_counts / period_counts.sum()).astype(float)

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self._const_mask = (self.sd_ == 0)
        self.sd_[self._const_mask] = 1.0
        Xs = np.where(self._const_mask, X, (X - self.mu_) / self.sd_).astype(np.float32)
        X_jax = jnp.asarray(Xs, dtype=jnp.float32)
        y_jax = jnp.asarray(y, dtype=jnp.int32)
        L2 = float(self.ridge)

        @jax.jit
        def _nll(params):
            coef = params.reshape(K - 1, F)
            coef_full = jnp.vstack([jnp.zeros((1, F), dtype=jnp.float32), coef])
            Z = X_jax @ coef_full.T
            Z = Z - jax.lax.stop_gradient(jnp.max(Z, axis=1, keepdims=True))
            prob = jax.nn.softmax(Z, axis=1)
            ll = jnp.sum(jnp.log(jnp.clip(prob[jnp.arange(len(y)), y_jax], 1e-12, 1)))
            return -ll + 0.5 * L2 * jnp.sum(params ** 2)

        _grad_fn = jax.jit(jax.grad(_nll))

        init = np.zeros((K - 1) * F, dtype=np.float32)
        log_shares = np.log(period_counts / period_counts.sum())
        init_intercepts = log_shares[1:] - log_shares[0]
        for k in range(K - 1):
            init[k * F] = float(init_intercepts[k])

        def _f(x): return float(_nll(jnp.asarray(x, dtype=jnp.float32)))
        def _g(x): return np.array(_grad_fn(jnp.asarray(x, dtype=jnp.float32)), dtype=float)

        from scipy.optimize import minimize
        res = minimize(_f, init.astype(float), method='L-BFGS-B', jac=_g,
                       options={'maxiter': 2000, 'ftol': 1e-10})

        if not res.success and len(y) < 500:
            res2 = minimize(_f, res.x, method='CG', jac=_g,
                            options={'maxiter': 3000, 'gtol': 1e-7})
            if res2.fun < res.fun:
                res = res2

        null_ll = float(np.sum(np.log(period_counts / period_counts.sum())[y]))
        ll_val  = float(-res.fun + 0.5 * L2 * np.sum(res.x ** 2))

        self.coef_     = res.x.reshape(K - 1, F)
        self.ll_       = ll_val
        self.null_ll_  = null_ll
        self.n_obs_    = len(y)
        self.converged_ = bool(res.success)

    def _fit_searchlibrium(self, X: np.ndarray, y: np.ndarray):
        _SLMNL = None
        try:
            from SearchLibrium import MultinomialLogit as _SLMNL
        except ImportError:
            pass
        if _SLMNL is None:
            self._fit_scipy(X, y)
            return

        period_counts = np.bincount(y, minlength=self.n_periods_).astype(float)
        period_counts = np.maximum(period_counts, 1.0)
        self.observed_period_shares_ = (period_counts / period_counts.sum()).astype(float)

        F = X.shape[1]
        K = self.n_periods_
        feat_names = self.feature_names_ or [f'x{i}' for i in range(F)]

        mnl = _SLMNL()
        try:
            # Use wide_setup if available (SearchLibrium >= 0.0.111), otherwise
            # fall back to manual long-format construction.
            if hasattr(mnl, 'wide_setup'):
                mnl.wide_setup(
                    X_wide=X, y_chosen=y, varnames=feat_names, n_alts=K,
                    empirical_init=True,
                    l2_penalty=float(self.ridge),
                )
            else:
                # ── Manual wide → long (backward compat) ──────────────────
                N = len(y)
                rows = []
                for alt in range(K):
                    chunk = np.zeros((N, F + 2))
                    chunk[:, :F] = X
                    chunk[:, F] = float(alt)
                    chunk[:, F + 1] = (y == alt).astype(float)
                    rows.append(chunk)
                long_mat = np.vstack(rows)
                ids = np.tile(np.arange(N, dtype=np.int32), K)
                sort_idx = np.lexsort((long_mat[:, F].astype(np.int32), ids))
                long_sorted = long_mat[sort_idx]
                ids_sorted = ids[sort_idx]
                mnl.setup(
                    X=long_sorted[:, :F].astype(float),
                    y=long_sorted[:, F + 1].astype(int),
                    varnames=feat_names,
                    alts=np.arange(K),
                    isvars=feat_names,
                    ids=ids_sorted,
                    base_alt=0,
                    l2_penalty=float(self.ridge),
                )
        except Exception:
            self._fit_scipy(X, y)
            return

        try:
            mnl.fit()
        except Exception:
            self._fit_scipy(X, y)
            return

        ll_val = float(getattr(mnl, 'loglik', float('-inf')))
        if not np.isfinite(ll_val) or ll_val > 0:
            self._fit_scipy(X, y)
            return

        coeff = np.asarray(getattr(mnl, 'coeff_est', []), dtype=float)
        n_expected = (K - 1) * F
        if len(coeff) == n_expected:
            coef_mat = coeff.reshape(K - 1, F)
        elif len(coeff) == K * F and F > 0:
            coef_mat = coeff.reshape(K, F)[1:, :]
        else:
            coef_mat = np.zeros((K - 1, F))

        null_ll = float(np.sum(np.log(period_counts / period_counts.sum())[y]))

        self.coef_     = coef_mat
        self.ll_       = ll_val
        self.null_ll_  = null_ll
        self.n_obs_    = len(y)
        self.converged_ = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(period=k) for each row. Shape (N, K)."""
        K  = self.n_periods_
        F  = X.shape[1]
        if self.mu_ is not None and self.sd_ is not None:
            const_mask = self._const_mask if len(self._const_mask) == F else np.zeros(F, dtype=bool)
            Xs = np.where(const_mask, X, (X - self.mu_) / np.where(self.sd_ > 0, self.sd_, 1.0))
        else:
            Xs = X
        coef_full = np.vstack([np.zeros((1, F)), self.coef_])
        Z  = Xs @ coef_full.T
        return self._softmax(Z)

    def predict_sampled(self, X: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
        """Sample a period index for each row."""
        proba = self.predict_proba(X)
        return np.array([
            rng.choice(self.n_periods_, p=proba[i])
            for i in range(len(X))
        ])

    def fit_within_period_offsets(self, df: pd.DataFrame) -> None:
        """Store empirical within-period start-time fractions from survey data."""
        draws = {}
        if 'STARTIME' not in df.columns or 'period' not in df.columns:
            self.within_period_draws_ = draws
            return
        for pidx, (_, p_start, p_end) in enumerate(TIME_PERIODS):
            sub = df[df['period'] == pidx]
            if sub.empty:
                continue
            width = max(float(p_end - p_start), 1.0)
            frac = ((pd.to_numeric(sub['STARTIME'], errors='coerce') - p_start) / width)
            frac = frac.replace([np.inf, -np.inf], np.nan).dropna().clip(0.0, 0.999)
            if len(frac) >= 5:
                draws[pidx] = frac.to_numpy(dtype=float)
        self.within_period_draws_ = draws


def _sample_within_period_fraction(model: Optional[MNLTimingModel],
                                   period_idx: int,
                                   rng: np.random.Generator) -> float:
    """Draw a start-time fraction within a chosen period."""
    if model is not None:
        draws = getattr(model, 'within_period_draws_', {}) or {}
        arr = draws.get(int(period_idx))
        if arr is not None and len(arr) > 0:
            return float(arr[rng.integers(0, len(arr))])
    return float(rng.triangular(0.05, 0.5, 0.95))


def _attach_schedule_context(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stable sequence, window, and known-anchor fields for Stage 5."""
    if df is None or df.empty or 'person_id' not in df.columns:
        return df
    if 'STARTIME' not in df.columns or 'ENDTIME' not in df.columns:
        return df

    work = df.copy()
    work['_schedule_orig'] = np.arange(len(work), dtype=int)
    order_col = next((c for c in ('planning_order', 'activity_seq', 'STARTIME')
                      if c in work.columns), None)
    if order_col is None:
        work['_schedule_order'] = work['_schedule_orig']
    else:
        work['_schedule_order'] = pd.to_numeric(
            work[order_col], errors='coerce').fillna(work['_schedule_orig'])
    work = work.sort_values(
        ['person_id', '_schedule_order', 'STARTIME', '_schedule_orig'],
        kind='stable').reset_index(drop=True)
    work['schedule_seq'] = work.groupby('person_id', sort=False).cumcount().astype(int)
    work['schedule_n_activities'] = work.groupby('person_id', sort=False)['person_id'].transform('size').astype(int)
    work['sched_prev_end'] = work.groupby('person_id', sort=False)['ENDTIME'].shift(1)
    work['sched_prev_end'] = work['sched_prev_end'].fillna(
        pd.to_numeric(work.get('start_of_day', 0.0), errors='coerce').fillna(0.0))
    work['sched_next_start'] = work.groupby('person_id', sort=False)['STARTIME'].shift(-1)
    work['sched_next_start'] = work['sched_next_start'].fillna(
        pd.to_numeric(work.get('end_of_day', 1440.0), errors='coerce').fillna(1440.0))

    home = pd.Series(np.nan, index=work.index, dtype=float)
    for col in ('OTAZ', 'HOME_TAZ', 'home_taz', 'home_zone'):
        if col in work.columns:
            values = pd.to_numeric(work[col], errors='coerce')
            home = home.where(home.notna() & (home > 0), values)
    known = pd.Series(np.nan, index=work.index, dtype=float)
    if 'DTAZ' in work.columns:
        values = pd.to_numeric(work['DTAZ'], errors='coerce')
        known = known.where(known.notna() & (known > 0), values)
    if 'purpose' in work.columns:
        purpose = work['purpose'].astype(str).str.lower()
        for purpose_name, col in (('work', 'WORK_TAZ'), ('edu', 'SCHOOL_TAZ')):
            if col in work.columns:
                values = pd.to_numeric(work[col], errors='coerce')
                known = known.where(
                    ~((purpose == purpose_name) & values.gt(0)), values)

    prev_anchor = np.full(len(work), np.nan, dtype=float)
    next_anchor = np.full(len(work), np.nan, dtype=float)
    for positions in work.groupby('person_id', sort=False).groups.values():
        positions = np.asarray(list(positions), dtype=int)
        anchors = known.to_numpy(dtype=float)[positions]
        homes = home.to_numpy(dtype=float)[positions]
        prev_value = np.nan
        for offset, position in enumerate(positions):
            prev_anchor[position] = prev_value if np.isfinite(prev_value) else homes[offset]
            if np.isfinite(anchors[offset]):
                prev_value = anchors[offset]
        next_value = np.nan
        for offset in range(len(positions) - 1, -1, -1):
            position = positions[offset]
            next_anchor[position] = next_value if np.isfinite(next_value) else homes[offset]
            if np.isfinite(anchors[offset]):
                next_value = anchors[offset]
    work['sched_prev_taz'] = prev_anchor
    work['sched_next_taz'] = next_anchor
    if 'ptour_id' not in work.columns:
        work['ptour_id'] = (
            work['person_id'].astype(str) + '_' + work['schedule_seq'].astype(str))
    else:
        work['ptour_id'] = work['ptour_id'].astype(str)
    work = work.sort_values('_schedule_orig', kind='stable')
    return work.drop(columns=['_schedule_orig', '_schedule_order'], errors='ignore')


class RRMTimeModel:
    """SearchLibrium Random Regret model for start-period choice."""

    FEATURE_NAMES = [
        'period_center_hr', 'period_center_sq', 'period_peak',
        'period_start_rel', 'fit_slack_rel', 'duration_pressure',
    ]

    def __init__(self, purpose: str):
        self.purpose_ = purpose
        self.n_periods_ = len(TIME_PERIODS)
        self.feature_names_ = list(self.FEATURE_NAMES)
        self.rrm_ = None
        self.within_period_draws_ = {}
        self.observed_period_shares_ = None
        self.period_blend_alpha_ = 0.0
        self.fit_status_ = 'not_fit'
        self.converged_ = False
        self.n_obs_ = 0
        self.ll_ = float('nan')
        self.rho2_ = float('nan')

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        duration = pd.to_numeric(
            df.get('duration_min', pd.Series(30.0, index=df.index)), errors='coerce'
        ).fillna(30.0).to_numpy(dtype=float)
        day_start = pd.to_numeric(
            df.get('day_start', df.get('start_of_day', pd.Series(420.0, index=df.index))),
            errors='coerce'
        ).fillna(420.0).to_numpy(dtype=float)
        day_end = pd.to_numeric(
            df.get('day_end', df.get('end_of_day', pd.Series(1200.0, index=df.index))),
            errors='coerce'
        ).fillna(1200.0).to_numpy(dtype=float)
        day_span = np.maximum(day_end - day_start, 60.0)

        X = np.zeros((len(df), self.n_periods_, len(self.FEATURE_NAMES)), dtype=float)
        for pidx, (_, p_start, p_end) in enumerate(TIME_PERIODS):
            center_hr = (p_start + p_end) / 120.0
            width = max(float(p_end - p_start), 1.0)
            X[:, pidx, :] = np.column_stack([
                np.full(len(df), center_hr),
                np.full(len(df), center_hr ** 2),
                np.full(len(df), float(pidx in (1, 3))),
                (p_start - day_start) / day_span,
                (p_end - duration - day_start) / day_span,
                duration / width,
            ])
        return X

    def fit(self, df: pd.DataFrame) -> None:
        if _SLRRM is None or len(df) < 30:
            return
        y = pd.to_numeric(df['period'], errors='coerce').dropna().astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            return
        fit_df = df.loc[df['period'].notna()].copy()
        X = self._build_X(fit_df)
        if len(X) > 20000:
            keep = np.linspace(0, len(X) - 1, 20000).astype(int)
            X = X[keep]
            y = y[keep]

        rrm = _SLRRM()
        rrm.X = X
        rrm.y = y
        rrm.nb_samples = len(X)
        rrm.nb_alt = self.n_periods_
        rrm.nb_attr = X.shape[2]
        rrm.attrs = list(self.FEATURE_NAMES)
        rrm.avail = np.ones((len(X), self.n_periods_), dtype=float)
        rrm.initialise()
        rrm.normalize = False
        rrm.fit(compute_inference=False)

        if not np.isfinite(getattr(rrm, 'loglik', np.nan)):
            return
        self.rrm_ = rrm
        self.n_obs_ = len(X)
        self.ll_ = float(rrm.loglik)
        counts = np.bincount(y, minlength=self.n_periods_).astype(float)
        self.observed_period_shares_ = counts / max(float(counts.sum()), 1.0)
        self.converged_ = bool(getattr(rrm, 'converged', True))
        self.fit_status_ = 'ok' if self.converged_ else 'optimizer_stopped'

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.rrm_ is None:
            return np.full((len(X), self.n_periods_), 1.0 / self.n_periods_)
        probs = np.asarray(self.rrm_.predict_proba(X), dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        return np.divide(
            probs,
            row_sums,
            out=np.full_like(probs, 1.0 / self.n_periods_),
            where=row_sums > 0,
        )

    def predict_sampled(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array([rng.choice(self.n_periods_, p=p / p.sum()) for p in proba])

    def fit_within_period_offsets(self, df: pd.DataFrame) -> None:
        draws = {}
        if 'STARTIME' not in df.columns or 'period' not in df.columns:
            self.within_period_draws_ = draws
            return
        for pidx, (_, p_start, p_end) in enumerate(TIME_PERIODS):
            sub = df[df['period'] == pidx]
            if sub.empty:
                continue
            frac = ((pd.to_numeric(sub['STARTIME'], errors='coerce') - p_start)
                    / max(float(p_end - p_start), 1.0))
            frac = frac.replace([np.inf, -np.inf], np.nan).dropna().clip(0.0, 0.999)
            if len(frac) >= 5:
                draws[pidx] = frac.to_numpy(dtype=float)
        self.within_period_draws_ = draws


def _fit_rrm_timing_model(subset: pd.DataFrame, purpose: str):
    if not _RRM_OK:
        return None
    model = RRMTimeModel(purpose)
    try:
        model.fit(subset)
    except Exception as exc:
        logger.warning(f'  timing/{purpose}: SearchLibrium RRM failed: {exc}')
        return None
    return model if model.rrm_ is not None else None


class RegretTimingModel(RRMTimeModel):
    """POLARIS-style activity-timing (start-period) choice via the in-house
    regret_choice engine. Reuses RRMTimeModel._build_X / predict_sampled /
    fit_within_period_offsets, but fits one of:
        'rum'    — utility maximisation over periods
        'rrm'    — classic Random Regret
        'hybrid' — the Golshani et al. (2018) hybrid RUM-RRM used in POLARIS
                   (a convex blend with an estimated weight λ; λ says how
                   regret-driven the timing decision is)
    Pure numpy/scipy (no SearchLibrium dependency), so it always runs and, unlike
    the SearchLibrium path, exposes a BIC for the auto decision."""

    def __init__(self, purpose: str, rule: str = 'hybrid'):
        super().__init__(purpose)
        self.rule_ = rule
        self.model_ = None
        self.lam_ = float('nan')
        self.bic_ = float('nan')

    def fit(self, df: pd.DataFrame) -> None:
        from regret_choice import RegretChoiceModel
        y = pd.to_numeric(df.get('period'), errors='coerce')
        fit_df = df.loc[y.notna()].copy()
        y = y.dropna().astype(int).to_numpy()
        if len(fit_df) < 30 or len(np.unique(y)) < 2:
            return
        X = self._build_X(fit_df)
        if len(X) > 20000:
            keep = np.linspace(0, len(X) - 1, 20000).astype(int)
            X, y = X[keep], y[keep]
        try:
            m = RegretChoiceModel(rule=self.rule_, fit_ascs=True).fit(
                X, y, names=list(self.FEATURE_NAMES))
        except Exception as exc:
            logger.warning(f'  timing/{self.purpose_}: regret ({self.rule_}) failed: {exc}')
            return
        if not np.isfinite(m.ll_):
            return
        self.model_ = m
        self.n_obs_ = int(m.n_)
        self.ll_ = float(m.ll_)
        self.bic_ = float(m.bic)
        self.lam_ = float(getattr(m, 'lam_', float('nan')))
        counts = np.bincount(y, minlength=self.n_periods_).astype(float)
        self.observed_period_shares_ = counts / max(float(counts.sum()), 1.0)
        self.converged_ = True
        self.fit_status_ = f'ok:{self.rule_}'

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            return np.full((len(X), self.n_periods_), 1.0 / self.n_periods_)
        p = self.model_.predict_proba(np.asarray(X, dtype=float))
        return np.nan_to_num(p, nan=1.0 / self.n_periods_)


def _fit_regret_timing_model(subset: pd.DataFrame, purpose: str, mode: str):
    """Fit an engine-based timing model. mode ∈ {'rum','rrm','hybrid'} fits that
    single rule; mode == 'auto' fits all three and keeps the best-BIC one — the
    decision rule becomes a data-driven (searched) choice, POLARIS-style."""
    rules = ('rum', 'rrm', 'hybrid') if mode == 'auto' else (mode,)
    best, best_bic = None, np.inf
    _bics = {}
    for r in rules:
        mdl = RegretTimingModel(purpose, rule=r)
        try:
            mdl.fit(subset)
        except Exception as exc:
            logger.warning(f'  timing/{purpose}: regret {r} failed: {exc}')
            continue
        if mdl.model_ is None or not np.isfinite(mdl.bic_):
            continue
        _bics[r] = mdl.bic_
        if mdl.bic_ < best_bic:
            best, best_bic = mdl, mdl.bic_
    if best is not None and mode == 'auto':
        logger.info(f'  timing/{purpose}: decision-rule search BIC='
                    f'{{{", ".join(f"{r}:{b:.1f}" for r, b in _bics.items())}}} '
                    f'→ {best.rule_}' + (f' (λ={best.lam_:.2f})'
                                         if best.rule_ == "hybrid" else ""))
    return best


class TDDChoiceModel:
    """Joint (start, end, duration) time-of-day-departure MNL — ActivitySim
    ``mandatory_tour_scheduling`` structure.

    Alternatives are the 21 (start_period, end_period) pairs in TDD_ALTS (start ≤
    end over the 6 TIME_PERIODS); each defines a duration. Unlike the departure-
    period-only softmax (MNLTimingModel), this chooses START and END jointly, so
    duration is an OUTPUT of scheduling rather than an input from Stage 3.

    Utility (idca) per alternative a:
        V_a = Σ dep_asc[start_a]  (period≠midday ref)
            + Σ arr_asc[end_a]
            + b_dur·dur_a + b_dur2·dur_a²
            + Σ_f person_f·(g_f_dep·start_norm_a + g_f_dur·dur_a)   # GA-selected f
            + b_los·commute_los(start_a)                            # period congestion

    The person×schedule interactions are the metaheuristic-searchable part
    (feature_override_ from the GA, same convention as MNLTimingModel). Estimated
    by a scipy MNL with ridge; predicted with day-window availability.
    """
    REF_PERIOD = 2   # midday reference for departure/arrival constants

    def __init__(self):
        self.beta_ = None
        self.param_names_ = None
        self.person_feats_ = None
        self.feature_override_ = None
        self.purpose_ = None
        self.fit_status_ = 'not_fit'
        self.n_obs_ = 0
        self.ll_final_ = None
        self._has_los_ = False
        self.coef_export_names_ = None
        self.coef_export_values_ = None
        self.std_err_export_ = None
        self.t_stat_export_ = None
        self.p_value_export_ = None

    def _person_feats(self, df: pd.DataFrame):
        names = None
        if self.feature_override_:
            names = self.feature_override_.get(self.purpose_,
                                               self.feature_override_.get('*'))
        if not names:
            names = DEFAULT_FEATURES_BY_PURPOSE.get(self.purpose_, PERSON_FEATURES)
        return [f for f in names if f in df.columns]

    def _design(self, df: pd.DataFrame, person_feats):
        """Return (D, names) with D of shape (n_obs, n_alts, n_params)."""
        N = len(df)
        A = len(TDD_ALTS)
        cols, names = [], []
        for p in range(_N_PERIODS):                       # departure ASCs
            if p == self.REF_PERIOD:
                continue
            cols.append(np.tile((_TDD_START == p).astype(float), (N, 1)))
            names.append(f'dep_{PERIOD_LABELS[p]}')
        for p in range(_N_PERIODS):                       # arrival ASCs
            if p == self.REF_PERIOD:
                continue
            cols.append(np.tile((_TDD_END == p).astype(float), (N, 1)))
            names.append(f'arr_{PERIOD_LABELS[p]}')
        cols.append(np.tile(_TDD_DUR, (N, 1)));       names.append('duration_hr')
        cols.append(np.tile(_TDD_DUR ** 2, (N, 1)));  names.append('duration_hr_sq')
        _start_norm = (_TDD_START - 2.5) / 2.5            # centered start index
        for f in person_feats:
            # defensive: a feature present at fit-time may be absent from a
            # single-row prediction frame → treat as 0 (interaction inactive).
            if f in df.columns:
                fv = pd.to_numeric(df[f], errors='coerce').fillna(0.0).to_numpy().reshape(N, 1)
            else:
                fv = np.zeros((N, 1))
            cols.append(fv * np.tile(_start_norm, (N, 1))); names.append(f'{f}_x_departure')
            cols.append(fv * np.tile(_TDD_DUR, (N, 1)));    names.append(f'{f}_x_duration')
        # period-varying commute LOS at each alt's START period (keystone)
        _los = np.zeros((N, A), dtype=float)
        for a in range(A):
            _c = _PERIOD_LOS_COL[int(_TDD_START[a])]
            if _c in df.columns:
                _los[:, a] = pd.to_numeric(df[_c], errors='coerce').fillna(0.0).to_numpy()
        if np.abs(_los).sum() > 0:
            cols.append(_los / 60.0); names.append('commute_los_hr'); self._has_los_ = True
        return np.stack(cols, axis=-1), names

    @staticmethod
    def _chosen_idx(df: pd.DataFrame):
        _idx = {(s, e): k for k, (s, e, _) in enumerate(TDD_ALTS)}
        sp = pd.to_numeric(df['period'], errors='coerce').fillna(2).to_numpy().astype(int)
        sp = np.clip(sp, 0, _N_PERIODS - 1)
        _end_t = (pd.to_numeric(df['STARTIME'], errors='coerce').fillna(540.0)
                  + pd.to_numeric(df.get('duration_min', 60.0), errors='coerce').fillna(60.0))
        ep = np.clip(np.array([_period_of(float(t)) for t in _end_t]), 0, _N_PERIODS - 1)
        ep = np.maximum(ep, sp)
        return np.array([_idx[(int(s), int(e))] for s, e in zip(sp, ep)], dtype=int)

    def fit(self, df: pd.DataFrame):
        from scipy.optimize import minimize
        person_feats = self._person_feats(df)
        self.person_feats_ = person_feats
        D, names = self._design(df, person_feats)
        y = self._chosen_idx(df)
        N, A, P = D.shape
        self.n_obs_ = N
        _l2 = float(os.environ.get(
            'GA_S4_RIDGE', os.environ.get('STAGE4_TDD_L2', '0.5')))
        _rows = np.arange(N)

        def _negll(beta):
            V = D @ beta
            V -= V.max(axis=1, keepdims=True)
            eV = np.exp(V)
            p = eV / eV.sum(axis=1, keepdims=True)
            ll = np.log(p[_rows, y] + 1e-12).sum()
            ED = np.einsum('na,nap->np', p, D)
            grad = -(D[_rows, y, :] - ED).sum(axis=0) + 2.0 * _l2 * beta
            return -ll + _l2 * float(beta @ beta), grad

        res = minimize(_negll, np.zeros(P), jac=True, method='L-BFGS-B',
                       options={'maxiter': 400})
        self.beta_ = res.x
        self.param_names_ = names
        self.ll_final_ = -float(res.fun)
        self.fit_status_ = 'fit'
        # standard errors via numerical Hessian of the PENALISED objective
        # (_negll[0] includes the _l2 ridge — the ≈2·_l2 eigenvalue lift is what
        # keeps the pinv well-conditioned on flat directions)
        try:
            H = _numerical_hessian(lambda b: _negll(b)[0], self.beta_)
            _cov = np.linalg.pinv(H)
            se = np.sqrt(np.clip(np.diag(_cov), 0.0, None))
            with np.errstate(divide='ignore', invalid='ignore'):
                t = self.beta_ / se
            from scipy.stats import norm as _norm
            self.std_err_export_ = se
            self.t_stat_export_ = t
            self.p_value_export_ = 2.0 * _norm.sf(np.abs(t))
        except Exception:
            pass
        self.coef_export_names_ = names
        self.coef_export_values_ = self.beta_
        return self

    def predict_tdd(self, feat_row: pd.DataFrame, start_of_day: float,
                    end_of_day: float, current_time: float,
                    rng: np.random.Generator):
        """Sample a joint (start_min, end_min) respecting the remaining day window."""
        D, _ = self._design(feat_row, self.person_feats_ or [])
        V = (D @ self.beta_)[0]
        avail = np.ones(len(TDD_ALTS), dtype=bool)
        for a in range(len(TDD_ALTS)):
            st, en = _PERIOD_CENTERS[int(_TDD_START[a])], _PERIOD_CENTERS[int(_TDD_END[a])]
            if st < current_time - 90.0 or en > end_of_day + 1e-6 or en < st:
                avail[a] = False
        if not avail.any():
            avail[:] = True
        V = np.where(avail, V, -1e9)
        V -= V.max()
        p = np.exp(V)
        p = p / p.sum()
        a = int(rng.choice(len(TDD_ALTS), p=p))
        st = max(float(_PERIOD_CENTERS[int(_TDD_START[a])]), current_time, start_of_day)
        en = max(float(_PERIOD_CENTERS[int(_TDD_END[a])]), st + 30.0)
        return st, min(en, end_of_day)


# ---------------------------------------------------------------------------
# 3.  ESTIMATE — ONE MODEL PER PURPOSE
# ---------------------------------------------------------------------------

def estimate_timing_models(
    trips: pd.DataFrame,
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    min_sample: int = 30,
    feature_override: list = None,
) -> dict:
    """Fit one MNLTimingModel per activity purpose."""
    data   = prepare_timing_data(trips, tours, persons)
    # Recreate GA search-frame columns (engineered flags, acc join) so a
    # GA-selected feature resolves at refit exactly as it did in the search.
    # Previously absent here (unlike stages 1-3), so any GA-selected engineered
    # or accessibility feature was silently dropped from the final fit.
    try:
        from ga_hypothesis_testing import ensure_ga_search_columns
        data = ensure_ga_search_columns(data, persons, stage_id=4,
                                        zone_col='HOME_SA2_2021')
    except Exception as _ee:
        logger.warning(f"GA feature-column enrichment failed: {_ee}")
    ptypes = sorted(data['ptype'].dropna().unique())

    models = {}
    _alpha_scale = float(os.environ.get('STAGE4_PERIOD_BLEND', '1.0') or 1.0)
    _alpha_scale = float(np.clip(_alpha_scale, 0.0, 2.0))
    _timing_estimator = os.environ.get('STAGE4_TIMING_MODEL', 'rrm').strip().lower()
    _purpose_alpha = {
        'work': 0.08,
        'edu': 0.08,
        'social': 0.20,
        'escort': 0.35,
        'shopping': 0.30,
        'othmaint': 0.30,
        'othdiscr': 0.25,
        'eat': 0.30,
    }

    for purpose in PURPOSES:
        subset = data[data['purpose'] == purpose].copy()
        # Filter rows to ptypes valid for this purpose (e.g. no Preschool in work).
        _allowed_ptypes = MNLTimingModel.PTYPE_ALLOWLIST.get(purpose)
        if _allowed_ptypes is not None and 'ptype' in subset.columns:
            subset = subset[subset['ptype'].isin(_allowed_ptypes)].copy()
        if len(subset) < min_sample:
            logger.warning(f"  timing/{purpose}: {len(subset)} obs — skipping")
            continue
        y = subset['period'].values.astype(int)
        present_periods = np.unique(y)
        if len(present_periods) < 2:
            logger.warning(f"  timing/{purpose}: only 1 period observed — skipping")
            continue

        if _timing_estimator == 'tdd':
            # Full joint (start, end, duration) tdd choice (ActivitySim
            # mandatory_tour_scheduling). Duration becomes an OUTPUT of scheduling.
            try:
                _tdd = TDDChoiceModel()
                _tdd.purpose_ = purpose
                # resolve GA/override person features the same way the MNL path
                # does (handles list-or-dict + ptype filtering), wrapped as a dict
                # so TDDChoiceModel._person_feats picks up this purpose's set.
                _tdd.feature_override_ = {
                    purpose: _timing_person_features(subset, purpose, feature_override)}
                _tdd.fit(subset)
                if _tdd.fit_status_ == 'fit':
                    models[purpose] = _tdd
                    logger.info(f"  timing/{purpose}: joint TDD choice, n={_tdd.n_obs_}, "
                                f"LL={_tdd.ll_final_:.1f}, {len(_tdd.param_names_)} params"
                                + ("  (+commute LOS)" if _tdd._has_los_ else ""))
                    continue
                logger.warning(f"  timing/{purpose}: TDD fit failed — falling back to RRM")
            except Exception as _tdde:
                logger.warning(f"  timing/{purpose}: TDD error ({_tdde}) — falling back to RRM")

        if _timing_estimator in ('rrm', 'searchlibrium', 'tdd'):
            rrm_model = _fit_rrm_timing_model(subset, purpose)
            if rrm_model is not None:
                rrm_model.fit_within_period_offsets(subset)
                rrm_model.period_blend_alpha_ = float(
                    np.clip(_purpose_alpha.get(purpose, 0.20) * _alpha_scale, 0.0, 0.60)
                )
                models[purpose] = rrm_model
                logger.info(
                    f"  timing/{purpose}: SearchLibrium RRM, n={rrm_model.n_obs_}, "
                    f"status={rrm_model.fit_status_}, "
                    f"blend_alpha={rrm_model.period_blend_alpha_:.2f}"
                )
                continue

        # POLARIS-style hybrid RUM-RRM (or pure rum/rrm), and 'auto' which searches
        # the decision rule by BIC — via the in-house regret_choice engine.
        if _timing_estimator in ('hybrid', 'rum', 'auto'):
            reg_model = _fit_regret_timing_model(subset, purpose, _timing_estimator)
            if reg_model is not None:
                reg_model.fit_within_period_offsets(subset)
                reg_model.period_blend_alpha_ = float(
                    np.clip(_purpose_alpha.get(purpose, 0.20) * _alpha_scale, 0.0, 0.60)
                )
                models[purpose] = reg_model
                _lam = (f", λ={reg_model.lam_:.2f}"
                        if reg_model.rule_ == 'hybrid' and np.isfinite(reg_model.lam_) else "")
                logger.info(
                    f"  timing/{purpose}: regret[{reg_model.rule_}], n={reg_model.n_obs_}, "
                    f"BIC={reg_model.bic_:.1f}{_lam}, "
                    f"blend_alpha={reg_model.period_blend_alpha_:.2f}"
                )
                continue

        m = MNLTimingModel()
        m.purpose_ = purpose
        m.ptypes_ = ptypes
        # Filter feature_override to exclude ptype dummies for disallowed ptypes
        _fo = _timing_person_features(data, purpose, feature_override)
        if _allowed_ptypes is not None:
            _fo = [f for f in _fo
                   if not (f.startswith('ptype_') and f[len('ptype_'):] not in _allowed_ptypes)]
        m.feature_override_ = _fo
        X = m._build_X(subset)
        m.fit(X, y)
        _X_for_prune = X

        if (not m.converged_) or (not np.isfinite(m.rho2_)) or (m.rho2_ < 0):
            logger.warning(f"  timing/{purpose}: retrying with reduced person features")
            m2 = MNLTimingModel()
            m2.purpose_ = purpose
            m2.ptypes_ = ptypes
            m2.feature_override_ = _fo
            m2.include_person_features_ = False
            X2 = m2._build_X(subset)
            m2.fit(X2, y)
            if np.isfinite(m2.rho2_) and (not np.isfinite(m.rho2_) or m2.rho2_ > m.rho2_):
                logger.info(
                    f"  timing/{purpose}: reduced-feature retry improved rho\u00b2 "
                    f"{m.rho2_ if np.isfinite(m.rho2_) else float('nan'):.3f} -> {m2.rho2_:.3f}"
                )
                m = m2
                _X_for_prune = X2

        # Iterative p-value + magnitude pruning \u2014 drop features insignificant
        # across ALL non-base periods.  Purpose dummies and structural timing
        # features are never pruned (they encode intercepts and activity state).
        _s4_zthresh = float(os.environ.get('GA_SIG_COEF_ZERO', '0.01'))
        _s4_pthresh = float(os.environ.get('GA_SIG_PVAL_THRESH', '0.10'))
        _S4_STRUCTURAL = frozenset({'day_position', 'time_consumed', 'duration_min', 'time_remaining'})
        for _s4_round in range(3):
            _s4_pv = getattr(m, 'p_value_', None)
            _s4_coef = getattr(m, 'coef_', None)
            if _s4_coef is None:
                break
            _s4_F = _s4_coef.shape[1]
            _s4_K1 = _s4_coef.shape[0]
            if _s4_F <= 1:
                break
            # p-values optional: when missing/short (Hessian failed) still prune
            # on coefficient magnitude instead of bailing out — so zero-fit terms
            # are removed regardless of whether SEs were computed.
            _s4_have_pv = _s4_pv is not None and len(_s4_pv) >= _s4_K1 * _s4_F
            _s4_pv_mat = (np.asarray(_s4_pv[:_s4_K1 * _s4_F]).reshape(_s4_K1, _s4_F)
                          if _s4_have_pv else None)
            _s4_coef_mat = np.asarray(_s4_coef)
            _s4_fnames = m.feature_names_ or []
            _s4_keep = np.ones(_s4_F, dtype=bool)
            for _s4_fi in range(_s4_F):
                _s4_fn = _s4_fnames[_s4_fi] if _s4_fi < len(_s4_fnames) else ''
                if _s4_fn.startswith('purp_') or _s4_fn in _S4_STRUCTURAL:
                    continue
                _s4_fi_co = _s4_coef_mat[:, _s4_fi]
                _all_small = np.all(np.abs(_s4_fi_co) < _s4_zthresh)
                _all_insig = bool(_s4_have_pv) and np.all(
                    ~np.isfinite(_s4_pv_mat[:, _s4_fi]) | (_s4_pv_mat[:, _s4_fi] > _s4_pthresh))
                if _all_insig or _all_small:
                    _s4_keep[_s4_fi] = False
            _s4_keep_idx = np.where(_s4_keep)[0]
            if len(_s4_keep_idx) == _s4_F or len(_s4_keep_idx) == 0:
                break
            _s4_new_names = [_s4_fnames[i] for i in _s4_keep_idx]
            _s4_Xp = _X_for_prune[:, _s4_keep_idx]
            # Decompose kept names back into model config so _build_X at
            # prediction time produces the same pruned columns.
            # Include structural features in feature_override_ at their original
            # position so that deduplication in _build_X preserves column order
            # (structural features must stay between GA features and PERSON_FEATURES).
            _s4_kept_ptypes = [n[len('ptype_'):] for n in _s4_new_names
                                if n.startswith('ptype_')]
            _s4_kept_cont = [n for n in _s4_new_names
                             if not n.startswith('purp_')
                             and not n.startswith('ptype_')]
            _s4_mp = MNLTimingModel()
            _s4_mp.purpose_ = m.purpose_
            # Inherit the kept-purpose set so the pruned refit builds the SAME
            # purpose-dummy columns as the pre-built pruned matrix (and predict).
            _s4_mp.kept_purposes_ = m.kept_purposes_
            _s4_mp.ptypes_ = _s4_kept_ptypes
            _s4_mp.include_person_features_ = False
            _s4_mp.feature_override_ = _s4_kept_cont
            _s4_mp.feature_names_ = _s4_new_names
            _s4_mp.fit(_s4_Xp, y)
            _s4_mp.feature_names_ = _s4_new_names
            if not np.isfinite(_s4_mp.rho2_) or _s4_mp.rho2_ < 0:
                break
            logger.info(
                f"  timing/{purpose} prune round {_s4_round + 1}: "
                f"{_s4_F} \u2192 {len(_s4_keep_idx)} features, "
                f"rho\u00b2={_s4_mp.rho2_:.3f}"
            )
            m = _s4_mp
            _X_for_prune = _s4_Xp

        m.fit_within_period_offsets(subset)
        m.period_blend_alpha_ = float(
            np.clip(_purpose_alpha.get(purpose, 0.20) * _alpha_scale, 0.0, 0.60)
        )
        models[purpose] = m
        _rho2 = m.rho2_ if np.isfinite(m.rho2_) else float('nan')
        logger.info(
            f"  timing/{purpose}: n={len(subset)}, periods observed={present_periods}, "
            f"rho\u00b2={_rho2:.3f}, status={getattr(m, 'fit_status_', 'na')}, "
            f"blend_alpha={getattr(m, 'period_blend_alpha_', 0.0):.2f}"
        )

    print(f"Stage 4: fitted {len(models)} timing models")
    return models


# ---------------------------------------------------------------------------
# 4.  SCHEDULE BUILDER
# ---------------------------------------------------------------------------

def _sample_startime_in_period(
    period_idx: int,
    duration_min: float,
    current_time: float,
    end_of_day: float,
    rng: np.random.Generator,
    model: Optional[MNLTimingModel] = None,
) -> float:
    """
    Sample a start time uniformly within the chosen period,
    subject to:
      - start >= current_time + MIN_TRAVEL_GAP
      - start + duration_min <= end_of_day
    """
    p_start = PERIOD_STARTS[period_idx]
    p_end   = PERIOD_ENDS[period_idx]

    lo = max(p_start, current_time + MIN_TRAVEL_GAP)
    hi = min(p_end,   end_of_day   - duration_min)

    if lo >= hi:
        # period doesn't fit — just use lo
        return lo

    frac = _sample_within_period_fraction(model, period_idx, rng)
    return float(lo + frac * (hi - lo))


def build_schedule(
    person_activities: pd.DataFrame,
    timing_models: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Place activities for a single person into time slots.

    Vectorised: builds feature arrays once per person rather than
    constructing a pd.DataFrame per activity (old approach was ~3.5h for
    7.6M activities; this runs in ~20 min by avoiding per-row overhead).
    """
    df = person_activities.reset_index(drop=True)
    n  = len(df)

    start_of_day = float(df['start_of_day'].iloc[0])
    end_of_day   = float(df['end_of_day'].iloc[0])

    # Pre-extract columns as numpy arrays to avoid per-row attribute lookup
    purposes      = df['purpose'].tolist()
    durations     = df['predicted_duration_min'].to_numpy(dtype=float)
    person_feats  = {col: df[col].to_numpy(dtype=float)
                     if col in df.columns else np.zeros(n)
                     for col in PERSON_FEATURES}
    acc_feats     = {col: df[col].to_numpy(dtype=float)
                     if col in df.columns else np.zeros(n)
                     for col in ACC_FEATURES}
    ptypes        = df['ptype'].tolist() if 'ptype' in df.columns else ['NonWorker'] * n

    startimes = np.empty(n, dtype=float)
    endtimes  = np.empty(n, dtype=float)
    periods   = []
    current_time = start_of_day

    for i in range(n):
        purpose      = purposes[i]
        duration_min = durations[i]
        model        = timing_models.get(purpose)

        if isinstance(model, TDDChoiceModel):
            # Full joint (start, end, duration) choice: the model sets BOTH the
            # start and the end (duration is an output, overriding Stage 3). Pass
            # the real activity row so person interactions + commute LOS apply.
            startime, endtime = model.predict_tdd(
                df.iloc[[i]], start_of_day, end_of_day, current_time, rng)
            startime = max(startime, current_time + MIN_TRAVEL_GAP, start_of_day)
            endtime  = max(endtime, startime + 1.0)
            if endtime > end_of_day:
                endtime = end_of_day
            period_idx = _period_of(startime)
        else:
            if model is not None:
                # Build feature dict using pre-extracted arrays — avoids pd.DataFrame
                feat_dict = {
                    'purpose':        purpose,
                    'ptype':          ptypes[i],
                    'day_position':   i / max(n - 1, 1),
                    'time_consumed':  current_time - start_of_day,
                    'duration_min':   duration_min,
                    'time_remaining': max(end_of_day - current_time, 0.0),
                }
                for col in PERSON_FEATURES:
                    feat_dict[col] = float(person_feats[col][i])
                for col in ACC_FEATURES:
                    feat_dict[col] = float(acc_feats[col][i])

                # Single-row DataFrame built from pre-computed dict
                feat_row   = pd.DataFrame([feat_dict])
                X          = model._build_X(feat_row)
                period_idx = int(model.predict_sampled(X, rng)[0])
            else:
                period_idx = _period_of(current_time)

            startime = _sample_startime_in_period(
                period_idx, duration_min, current_time, end_of_day, rng, model=model)
            startime = max(startime, current_time + MIN_TRAVEL_GAP)
            endtime  = startime + duration_min

            if startime >= end_of_day:
                startime = end_of_day - duration_min
                endtime  = end_of_day
            startime = max(startime, start_of_day)
            endtime  = max(endtime,  startime + 1.0)

        startimes[i]  = round(startime)
        endtimes[i]   = round(endtime)
        periods.append(PERIOD_LABELS[period_idx])
        current_time  = endtime

    out = df.copy()
    out['STARTIME']    = startimes.astype(int)
    out['ENDTIME']     = endtimes.astype(int)
    out['time_period'] = periods
    return _attach_schedule_context(out)


# ---------------------------------------------------------------------------
# 5.  PREDICT — SYNTHETIC POPULATION
# ---------------------------------------------------------------------------

def _schedule_batch(args):
    """
    Process one batch of (pid, grp) pairs inside a worker process.
    Must be a module-level function so ProcessPoolExecutor can pickle it.
    """
    batch, models, batch_seed = args
    batch_rng = np.random.default_rng(batch_seed)
    out = []
    for _, grp in batch:
        out.append(build_schedule(grp, models, batch_rng))
    return out


def predict_schedule(
    activities: pd.DataFrame,
    timing_models: dict,
    sample: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign STARTIME and ENDTIME to every activity in the Stage 3 output.

    Parameters
    ----------
    activities    : Stage 3 output — one row per activity with
                    purpose, predicted_duration_min,
                    start_of_day, end_of_day, B, person attributes
    timing_models : output of estimate_timing_models()
    sample        : use sampled predictions (True) or modal period (False)
    seed          : random seed

    Returns
    -------
    activities with STARTIME, ENDTIME, time_period columns added.
    """
    df  = activities.copy()

    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    pid_col = 'person_id'

    # ptype
    if 'ptype' not in df.columns:
        try:
            from stage1_activity_frequency import assign_ptype_column
            df = assign_ptype_column(df)
        except ImportError:
            df['ptype'] = 'NonWorker'

    # person feature defaults
    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2, 'WORKSTAT': 0, 'STUDYING': 0}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    # Origin/job conditionals at predict time (same builder as estimation —
    # keeps fit/predict columns aligned; safe no-op when sources absent).
    df = _attach_origin_job_features(df)
    # Joint-travel flags at predict time (from tagged is_joint /
    # joint_participants cols when the joint map exists; else zeros).
    df = _attach_joint_features(df)
    # Predict-side CDAP equivalents: synth persons carry COORDINATED dap (not
    # observed), so mirror the estimation cdap_* names from prediction columns.
    # Keeps fit/predict aligned when models selected cdap_* features.
    try:
        if 'cdap_M' not in df.columns and 'dap' in df.columns:
            _dap = df['dap'].astype(str)
            df['cdap_M'] = (_dap == 'M').astype(float)
            df['cdap_N'] = (_dap == 'N').astype(float)
            df['cdap_H'] = (_dap == 'H').astype(float)
        if 'is_joint_traveller' not in df.columns and 'is_joint_act' in df.columns:
            df['is_joint_traveller'] = df['is_joint_act']
        if 'n_joint_tours_hh' not in df.columns and 'joint_tour_ids' in df.columns:
            df['n_joint_tours_hh'] = df['joint_tour_ids'].astype(str).ne('').astype(float)
    except Exception as _cdap_p_e:
        logger.warning('_attach predict-CDAP mirror skipped (%s)', _cdap_p_e)

    # Rebuild GA-searchable derived features (workstat/studying/age flags,
    # INDUSTRY/occp, household flags, home-zone acc, taste interactions) so
    # GA-selected terms resolve here. Without this, _build_X silently drops
    # missing columns and the whole purpose can collapse to the default period.
    try:
        from ga_hypothesis_testing import ensure_prediction_features
        df = ensure_prediction_features(df, stage_id=4)
    except Exception as _e:
        logger.warning('prediction feature enrichment skipped (%s)', _e)

    # Reclassify adult edu activities with STUDYING=0 → 3 (post-secondary).
    # Mirrors the same fix in prepare_timing_data and stage5 _dest_tree.
    if 'AGEGROUP' in df.columns:
        _adult_edu = (df['purpose'] == 'edu') & (df['STUDYING'] == 0) & (df['AGEGROUP'] >= 4)
        if _adult_edu.any():
            df.loc[_adult_edu, 'STUDYING'] = 3

    # ensure day window columns exist
    if 'start_of_day' not in df.columns:
        df['start_of_day'] = 420    # default 07:00
    if 'end_of_day' not in df.columns:
        df['end_of_day']   = 1200   # default 20:00

    # Prefer the Stage 3 learned planning order; retain the priority fallback
    # for older Stage 3 files that do not carry planning_order.
    df['_priority'] = df['purpose'].map(PURPOSE_PRIORITY).fillna(9)
    if 'planning_order' in df.columns:
        df = df.sort_values(['person_id', 'planning_order', '_priority']).reset_index(drop=True)
    elif 'activity_seq' in df.columns:
        df = df.sort_values(['person_id', 'activity_seq', '_priority']).reset_index(drop=True)
    else:
        df = df.sort_values(['person_id', '_priority']).reset_index(drop=True)

    # ── Vectorised scheduling ─────────────────────────────────────────────
    # PHASE A — Predict time_period for ALL activities in one batch per purpose.
    #   For 8 purposes × ~1M activities: 8 model.predict_proba() calls
    #   instead of 7.6M single-row _build_X calls. ~100× speedup.
    #
    # PHASE B — Apply time arithmetic per person (sequential within person,
    #   required for STARTIME monotonicity). Pure numpy: no DataFrames.
    #   Expected total: ~2-5 min vs 3.5h previously.
    rng = np.random.default_rng(seed)
    n_acts = len(df)
    n_persons = df[pid_col].nunique()
    logger.info(f"Stage 4 vectorised: {n_acts:,} activities, {n_persons:,} persons")

    # ── Phase A: batch period prediction ───────────────────────────────
    # Build a placeholder time_consumed / time_remaining using day_position proxy.
    # These are re-computed accurately in Phase B; here we just need plausible values
    # for the MNL to select a period bucket.
    df = df.reset_index(drop=True)

    # day_position: rank of activity within person (0=first, 1=last)
    df['_act_rank'] = df.groupby(pid_col).cumcount()
    df['_n_acts']   = df.groupby(pid_col)[pid_col].transform('count')
    df['_day_pos']  = df['_act_rank'] / (df['_n_acts'] - 1).clip(lower=1)

    budget_min      = (df['end_of_day'] - df['start_of_day']).clip(lower=60)
    df['_t_consumed']   = df['_day_pos'] * budget_min * 0.5   # proxy
    df['_t_remaining']  = budget_min * (1 - df['_day_pos'])    # proxy

    period_col = np.full(n_acts, 2, dtype=int)   # default AM_PEAK

    for purpose, model in timing_models.items():
        mask = (df['purpose'] == purpose).values
        if not mask.any():
            continue
        sub = df[mask].copy()
        sub_feat = sub.copy()
        sub_feat['day_position']   = sub['_day_pos']
        sub_feat['time_consumed']  = sub['_t_consumed']
        sub_feat['time_remaining'] = sub['_t_remaining']
        sub_feat['duration_min']   = sub['predicted_duration_min']

        try:
            X = model._build_X(sub_feat)
            proba = model.predict_proba(X)           # (n, K) — one batch call
            _alpha = float(getattr(model, 'period_blend_alpha_', 0.0) or 0.0)
            _target = getattr(model, 'observed_period_shares_', None)
            if _alpha > 0.0 and _target is not None:
                _target = np.asarray(_target, dtype=float)
                if (
                    _target.ndim == 1
                    and len(_target) == proba.shape[1]
                    and np.isfinite(_target).all()
                    and _target.sum() > 0
                ):
                    _target = _target / _target.sum()
                    proba = (1.0 - _alpha) * proba + _alpha * _target.reshape(1, -1)
                    proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
            idxs = np.empty(len(proba), dtype=int)
            # Vectorized categorical sampling: (cumsum > u).argmax(1)
            u = rng.random((len(proba), 1))
            cumsum = np.cumsum(proba, axis=1)
            idxs = (cumsum > u).argmax(axis=1)
            period_col[mask] = idxs
        except Exception as e:
            logger.warning(f"Stage 4 batch predict failed for '{purpose}': {e}")

    df['_period_idx'] = period_col
    # ── Joint coupling: members of one joint tour travel together, so they
    # share one period draw (else parent pickup + child drop-off split across
    # periods). One draw = first member's; safe no-op when untagged.
    try:
        if 'joint_tour_id' in df.columns:
            _jt = df['joint_tour_id'].astype(str)
            _has_jt = _jt.ne('') & _jt.ne('nan') & _jt.ne('None')
            if _has_jt.any():
                _first = df.loc[_has_jt].groupby(_jt[_has_jt])['_period_idx'].transform('first')
                df.loc[_has_jt, '_period_idx'] = _first.astype(int)
                logger.info(f"  Joint coupling: shared period for {int(_has_jt.sum()):,} joint rows "
                            f"({int(_jt[_has_jt].nunique()):,} tours)")
    except Exception as _jc_e:
        logger.warning(f"  Joint period coupling skipped ({_jc_e})")
    logger.info(f"  Phase A complete: periods assigned for {n_acts:,} activities")

    # ── Phase B: time arithmetic per person ──────────────────────────────
    # Sort by person and the planning sequence already chosen above.
    _order_cols = [pid_col]
    if 'planning_order' in df.columns:
        _order_cols.append('planning_order')
    elif 'activity_seq' in df.columns:
        _order_cols.append('activity_seq')
    _order_cols.append('_priority')
    df = df.sort_values(_order_cols).reset_index(drop=True)

    # Extract numpy arrays for arithmetic — avoids any DataFrame overhead
    pids_arr    = df[pid_col].to_numpy()
    sod_arr     = df['start_of_day'].to_numpy(dtype=float)
    eod_arr     = df['end_of_day'].to_numpy(dtype=float)
    dur_arr     = df['predicted_duration_min'].to_numpy(dtype=float)
    pidx_arr    = df['_period_idx'].to_numpy(dtype=int)
    purpose_arr = df['purpose'].astype(str).to_numpy()

    startimes = np.empty(n_acts, dtype=float)
    endtimes  = np.empty(n_acts, dtype=float)
    period_labels = []

    # Process each person's block with pure numpy — no DataFrames
    pid_changes = np.where(np.diff(pids_arr, prepend=pids_arr[0]-1) != 0)[0]
    pid_changes = np.append(pid_changes, n_acts)

    LOG_EVERY = 50_000
    n_done = 0

    for p_start, p_end in zip(pid_changes[:-1], pid_changes[1:]):
        n_done += 1
        if n_done % LOG_EVERY == 0:
            pct = 100.0 * n_done / n_persons
            logger.info(f"  Stage 4 progress: {n_done:,}/{n_persons:,} persons ({pct:.1f}%)")

        current_time = float(sod_arr[p_start])
        eod          = float(eod_arr[p_start])

        for j in range(p_start, p_end):
            pidx     = int(pidx_arr[j])
            dur      = float(dur_arr[j])
            model    = timing_models.get(purpose_arr[j])
            p_start_min = float(PERIOD_STARTS[pidx]) if pidx < len(PERIOD_STARTS) else current_time
            p_end_min   = float(PERIOD_ENDS[pidx]) if pidx < len(PERIOD_ENDS) else eod
            lo = max(p_start_min, current_time + MIN_TRAVEL_GAP)
            hi = min(p_end_min, eod - dur)
            if lo >= hi:
                startime = lo
            else:
                frac = _sample_within_period_fraction(model, pidx, rng)
                startime = lo + frac * (hi - lo)
            endtime  = startime + dur

            if startime >= eod:
                startime = max(eod - dur, float(sod_arr[p_start]))
                endtime  = eod
            startime = max(startime, float(sod_arr[p_start]))
            endtime  = max(endtime,  startime + 1.0)

            startimes[j] = round(startime)
            endtimes[j]  = round(endtime)
            period_labels.append(PERIOD_LABELS[pidx])
            current_time = endtime

    logger.info(f"  Stage 4 done: {n_persons:,}/{n_persons:,} persons (100%)")

    df['STARTIME']    = startimes.astype(int)
    df['ENDTIME']     = endtimes.astype(int)
    df['time_period'] = period_labels
    df = _attach_schedule_context(df)
    # ── Joint time sync: co-travellers share one realised start (group median)
    # so pickup/drop-off pairs stay feasible together. Safe no-op untagged.
    try:
        if 'joint_tour_id' in df.columns:
            _jt2 = df['joint_tour_id'].astype(str)
            _m2 = _jt2.ne('') & _jt2.ne('nan') & _jt2.ne('None')
            if _m2.any():
                _med = df.loc[_m2].groupby(_jt2[_m2])['STARTIME'].transform('median')
                _dur = (df.loc[_m2, 'ENDTIME'] - df.loc[_m2, 'STARTIME']).clip(lower=1)
                df.loc[_m2, 'STARTIME'] = _med.astype(int).values
                df.loc[_m2, 'ENDTIME'] = (_med.astype(int) + _dur.astype(int)).values
    except Exception as _js_e:
        logger.warning(f"  Joint time sync skipped ({_js_e})")
    out = df
    out = out.drop(columns=['_priority'], errors='ignore')

    # summary
    print(
        f"Stage 4 predictions: {len(out)} activities scheduled\n"
        f"  Mean STARTIME : {out['STARTIME'].mean()/60:.2f} hrs\n"
        f"  Mean ENDTIME  : {out['ENDTIME'].mean()/60:.2f} hrs\n"
        f"  Period distribution:\n" +
        "\n".join(
            f"    {p}: {int((out['time_period']==p).sum())}"
            for p in PERIOD_LABELS if (out['time_period'] == p).any()
        )
    )

    return out


# ---------------------------------------------------------------------------
# 6.  SAVE / LOAD
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
def save_models(models: dict, path: str = 'saved_data/stage4_models.pkl'):
    # Save the pkl FIRST — HTML/paper artifact generation is non-fatal.
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Stage 4 models saved → {path}")
    # ── Optional artifacts (HTML report + paper LaTeX) ───────────────────
    # Wrapped in try/except so that a missing dependency (e.g. paper_model_forms,
    # matplotlib backend) never prevents the pkl from being reported as saved.
    try:
        html_path = path.replace('.pkl', '_report.html')
        flat = {}
        for purpose, mdl in models.items():
            n_rows = getattr(mdl, 'n_obs_', getattr(mdl, 'n_', '?'))
            ll_val = getattr(mdl, 'll_', getattr(mdl, 'loglik_', None))
            flat[purpose] = {
                'model': mdl,
                'n_obs': n_rows,
                'features': list(getattr(mdl, 'coef_export_names_', [])),
                'converged': getattr(mdl, 'converged_', True),
                'll': ll_val,
            }
        _write_model_html({'mnl_timing': flat}, html_path,
                          'Stage 4 — Schedule Placement (MNL Timing)',
                          notes=f'MNL timing models per purpose. Saved: {path}')
    except Exception as _html_e:
        print(f"  Stage 4 HTML report failed (non-fatal): {_html_e}")
    try:
        from paper_model_forms import write_stage_paper_artifacts
        write_stage_paper_artifacts('stage4', os.path.dirname(path) or 'saved_data')
    except Exception as _paper_e:
        print(f"  Stage 4 paper artifacts failed (non-fatal): {_paper_e}")


def load_models(path: str = 'saved_data/stage4_models.pkl') -> dict:
    with open(path, 'rb') as f:
        models = pickle.load(f)
    print(f"Stage 4 models loaded ← {path}")
    return models


# ---------------------------------------------------------------------------
# 7.  FULL PIPELINE WRAPPER
# ---------------------------------------------------------------------------

def run_stage4(
    trips: pd.DataFrame,
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    synth_s3: pd.DataFrame,
    sample: bool = True,
    seed: int = 42,
    save_path: str = 'saved_data/stage4_models.pkl',
    feature_override: list = None,
) -> tuple:
    """Estimate timing models on HHTS, schedule activities for Stage 3 output."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    print("=" * 55)
    print("STAGE 4 — Schedule Placement")
    print("=" * 55)
    if feature_override:
        print(f"  Feature override: {feature_override}")

    print("\n[1/2] Estimating timing models from HHTS trips...")
    models = estimate_timing_models(trips, tours, persons,
                                    feature_override=feature_override)

    if save_path:
        save_models(models, save_path)

    print("\n[2/2] Building schedules for synthetic population...")
    scheduled = predict_schedule(synth_s3, models, sample=sample, seed=seed)

    print("\nStage 4 complete. Pass scheduled to run_stage5().")
    return scheduled, models

# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------
# Usage:
#   python stage4_schedule_placement.py                        # resume from saved_data/stage3_output.csv
#   python stage4_schedule_placement.py --fresh                # fresh: re-run Stages 1-3 first
#   python stage4_schedule_placement.py --input path/to/s3.csv # explicit Stage 3 input

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
    log = logging.getLogger('stage4_main')

    _OUT   = os.environ.get('PIPELINE_OUT_DIR', 'saved_data')
    S3_CSV = os.path.join(_OUT, 'stage3_output.csv')

    args  = sys.argv[1:]
    fresh = '--fresh' in args
    args  = [a for a in args if a != '--fresh']

    input_csv = S3_CSV
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
        log.info("Fresh mode: running Stages 1-3 first...")
        import subprocess, sys as _sys
        cmd = [_sys.executable, 'stage1_activity_frequency.py', '--fresh']
        if zone_arg:
            cmd.append(zone_arg)
        subprocess.run(cmd, check=True)
        subprocess.run([_sys.executable, 'stage2_day_duration.py'], check=True)
        subprocess.run([_sys.executable, 'stage3_activity_duration.py'], check=True)

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Stage 3 output not found at '{input_csv}'.\n"
            f"  Run stage3_activity_duration.py first, or use --fresh."
        )
    log.info(f"Loading Stage 3 output from {input_csv} ...")
    activities = pd.read_csv(input_csv)
    log.info(f"  {len(activities):,} activity rows loaded")

    # ── run Stage 4 ───────────────────────────────────────────────────────
    os.makedirs(_OUT, exist_ok=True)
    scheduled, models = run_stage4(
        out['trips'], out['tours'], out['persons'], activities
    )
    scheduled.to_csv('saved_data/stage4_output.csv', index=False)
    log.info(f"Stage 4 output saved → {_OUT}/stage4_output.csv")
    log.info(f"  {len(scheduled):,} scheduled activities")
    log.info(f"Pass {_OUT}/stage4_output.csv to stage5_destination_choice.py")


