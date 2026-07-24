"""
stage1_activity_frequency.py
=============================
Stage 1 of the activity-based model pipeline.

Predicts the NUMBER of activities of each purpose a person makes in a day,
using an ordered logit model estimated separately per person type (ptype).

Pipeline position
-----------------
    Synth persons (attributes + home zone)
        → Stage 1: activity frequency   ← YOU ARE HERE
        → Stage 2: day duration budget
        → Stage 3: activity duration
        → Stage 4: schedule placement
        → Stage 5: destination choice
        → Stage 6: mode choice

Outputs per person
------------------
    n_work_out, n_edu_out, n_social_out, n_escort_out,
    n_shopping_out, n_othmaint_out, n_othdiscr_out, n_eat_out

Usage — estimate from HHTS
---------------------------
    from stage1_activity_frequency import (
        prepare_frequency_data,
        estimate_frequency_models,
        predict_activity_frequency,
        run_stage1,
    )

    out    = Synth_Connect().process_survey(save=False)
    models = estimate_frequency_models(out["tours"], out["persons"])
    models  # dict of {purpose: {ptype: OrderedLogitResult}}

Usage — predict synthetic population
--------------------------------------
    synth       = pd.read_csv('saved_data/zone_ids_sa2/order_40.csv')
    synth_pred  = predict_activity_frequency(synth, models)
    # synth_pred has n_work_out, n_edu_out, … columns added
"""

import numpy as np
import pandas as pd
import logging
import os
from typing import Optional
from scipy.stats import norm

logger = logging.getLogger(__name__)


def add_total_activity_counts(tours: pd.DataFrame) -> pd.DataFrame:
    """
    Combine n_{purpose}_out + n_{purpose}_in → n_{purpose}_out (and _total).

    count_stops() splits activities into outbound (on way TO main dest)
    and inbound (on way HOME). Stage 1 should see the full daily count.
    Call this on as_tours right after build_as_tours() in process_survey.

    Also applies realistic upper caps to guard against data errors.
    """
    CAPS = {
        'work': 3, 'edu': 2, 'social': 5, 'escort': 6,
        'shopping': 5, 'othmaint': 5, 'othdiscr': 5, 'eat': 4,
    }
    for purp in PURPOSES:
        out_col = f'n_{purp}_out'
        in_col  = f'n_{purp}_in'
        tot_col = f'n_{purp}_total'
        out_v = tours[out_col].fillna(0) if out_col in tours.columns \
                else pd.Series(0, index=tours.index)
        in_v  = tours[in_col].fillna(0)  if in_col  in tours.columns \
                else pd.Series(0, index=tours.index)
        combined = (out_v + in_v).clip(upper=CAPS.get(purp, 5))
        tours[tot_col] = combined
        tours[out_col] = combined   # overwrite _out with full day count
    return tours


# ---------------------------------------------------------------------------
# CONFIGURATION — edit here if your column names differ
# ---------------------------------------------------------------------------

# Activity purposes to model
PURPOSES = [
    'work',
    'edu',
    'social',
    'escort',
    'shopping',
    'othmaint',
    'othdiscr',
    'eat',
]

# All non-mandatory purposes consolidated into a single count for GA search,
# matching the work/edu/nonmand structure of Stage 5.  Individual ordered-logit
# models are ALSO fitted for each nonmand purpose separately so predictions are
# behaviourally grounded rather than relying on Dirichlet proportions only.
NONMAND_PURPOSES = ['escort', 'shopping', 'social', 'othmaint', 'othdiscr', 'eat']

# Hard constraints on which person types are valid estimators for each purpose.
# Models are only estimated (and evaluated in GA fitness) for these ptypes.
# This prevents spurious models such as a work model fitted on SchoolChild rows.
VALID_PTYPES_FOR_PURPOSE: dict = {
    # Mandatory purposes: only people with that obligation
    'work':     {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent'},
    'edu':      {'SchoolChild', 'UniversityStudent', 'WorkingStudent'},
    # Escort: adults only — SchoolChild/Preschool are the ones being escorted
    'escort':   {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
                 'UniversityStudent', 'Retired', 'NonWorker'},
    # All other non-mandatory: all adult ptypes (school children rarely shop/eat alone)
    'shopping': {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
                 'UniversityStudent', 'Retired', 'NonWorker'},
    'social':   {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
                 'UniversityStudent', 'Retired', 'NonWorker'},
    'othmaint': {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
                 'UniversityStudent', 'Retired', 'NonWorker'},
    'othdiscr': {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
                 'UniversityStudent', 'Retired', 'NonWorker'},
    'eat':      {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
                 'UniversityStudent', 'SchoolChild', 'Retired', 'NonWorker'},
    # Consolidated nonmand (GA search segment): same adult set as individual purposes
    'nonmand':  {'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
                 'UniversityStudent', 'Retired', 'NonWorker'},
}

# Max count cap per purpose — hard ceiling on raw data values before any model.
# Guards against data errors; individual purposes can genuinely reach these levels.
MAX_COUNT = {
    'work':      3,
    'edu':       2,
    'social':    5,
    'escort':    6,
    'shopping':  5,
    'othmaint':  5,
    'othdiscr':  5,
    'eat':       4,
    'nonmand':  18,   # combined raw cap; binned to ORDERED_LOGIT_MAX_CLASSES below
}

# Maximum number of ordered-logit outcome classes per purpose.
# Bins the count outcome to at most this many distinct values before model fit,
# so the number of threshold cut-points = ORDERED_LOGIT_MAX_CLASSES - 1.
# 4 classes → 3 thresholds representing (0 = none, 1 = low, 2 = medium, 3+ = high).
ORDERED_LOGIT_MAX_CLASSES = {
    'work':     4,   # 0, 1, 2, 3+   → 3 thresholds
    'edu':      3,   # 0, 1, 2+      → 2 thresholds
    'nonmand':  4,   # 0, 1, 2, 3+   → 3 thresholds (was up to 18 classes)
    'escort':   4,
    'shopping': 4,
    'social':   4,
    'othmaint': 4,
    'othdiscr': 4,
    'eat':      4,
}

# Person type definitions — maps ptype label → filter on persons DataFrame
# Uses the assign_ptype() function from your existing code if available,
# otherwise falls back to these column-based rules
# Unified ptype labels — MUST match cdap._derive_ptype() output so synth pop
# predictions align with models estimated from HHTS.
#   WORKSTAT codes (from data_connect.work_map): 1=FullTime, 2=PartTime, 5=Retired
#   STUDYING codes (from data_connect.study_map): 1=primary, 2=secondary,
#                                                  3=tertiaryPartTime, 4=tertiaryFullTime
PTYPE_COLS = {
    'FullTimeWorker':    lambda p: (p['WORKSTAT'] == 1) & ~p['STUDYING'].isin([3, 4]),
    'PartTimeWorker':    lambda p: (p['WORKSTAT'] == 2) & ~p['STUDYING'].isin([3, 4]),
    'WorkingStudent':    lambda p: p['WORKSTAT'].isin([1, 2]) & p['STUDYING'].isin([3, 4]),
    'UniversityStudent': lambda p: (
        ~p['WORKSTAT'].isin([1, 2, 5]) & p['STUDYING'].isin([3, 4])
    ),
    'SchoolChild':       lambda p: p['STUDYING'].isin([1, 2]),
    'Retired':           lambda p: (
        p['WORKSTAT'] == 5
        if 'WORKSTAT' in p.columns
        else pd.Series(False, index=p.index)
    ),
    'NonWorker':         lambda p: ~(
        p['WORKSTAT'].isin([1, 2, 5]) | p['STUDYING'].isin([1, 2, 3, 4])
    ),
}

# Map infrequent ptypes to the closest model ptype for prediction.
# Preschool children attend childcare (edu) like school children but cannot work.
# WorkingStudent/UniversityStudent fall back to each other if one is absent.
PTYPE_PREDICTION_MAP = {
    'FullTimeWorker':    'FullTimeWorker',
    'PartTimeWorker':    'PartTimeWorker',
    'WorkingStudent':    'WorkingStudent',
    'UniversityStudent': 'UniversityStudent',
    'SchoolChild':       'SchoolChild',
    'Retired':           'Retired',
    'NonWorker':         'NonWorker',
    'Preschool':         'SchoolChild',      # preschool → SchoolChild for edu model
    # Legacy labels (old assign_ptype output) — keep for backward compat
    'School':            'SchoolChild',
    'Student':           'UniversityStudent',
    'JointWorkStudy':    'FullTimeWorker',
    'Unemployed / Other': 'NonWorker',
}

# All ptype labels that can appear in the 'ptype' column (assign_ptype_column /
# data_connect.assign_ptype) — used to build the ptype_* one-hot columns.
ALL_PTYPES = [
    'FullTimeWorker', 'PartTimeWorker', 'WorkingStudent', 'UniversityStudent',
    'SchoolChild', 'Retired', 'NonWorker', 'Preschool',
]

# Per-(ptype, purpose) overrides that take precedence over PTYPE_PREDICTION_MAP.
# SchoolChild predicts near-zero escort (school-age children rarely escort others)
# but preschool children in the HHTS often accompany parents on errands, producing
# escort trips in their diary.  NonWorker captures this accompanying behaviour better.
PTYPE_PURPOSE_PREDICTION_MAP: dict = {
    ('Preschool', 'escort'): 'NonWorker',
    ('WorkingStudent', 'work'): 'PartTimeWorker',
    ('WorkingStudent', 'edu'): 'UniversityStudent',
    ('WorkingStudent', 'othdiscr'): 'UniversityStudent',
    ('WorkingStudent', 'eat'): 'UniversityStudent',
}

# Person-level predictor columns used in the frequency model
PERSON_FEATURES = [
    'sexp',        # sex (1=male, 2=female)
    'vehrd',       # vehicles in household (clipped to 4)
    'nprd',        # household size
    'STUDYING',    # study type code
    'WORKSTAT',    # work status code
    'incp',        # income band (1-17)
]

# WORKSTAT and STUDYING are nominal codes (see PTYPE_COLS comment above) — a
# raw ordinal term implies e.g. "retired" (5) is a multiple of "part-time
# worker" (1), which has no meaning. Use the per-category binary columns
# built by _add_categorical_person_features() instead, so each category gets
# its own coefficient:
#   ptype_*            — one-hot of the 'ptype' column
#   mandatory_role     — worker or student (daily out-of-home obligation)
#   studying_primary / _secondary / _tertiary_pt / _tertiary_ft  — STUDYING category
#   workstat_unemployed / _carer / _illness / _volunteer / _other_reason
#                      — WORKSTAT "not working" reason (1/2/5 are already
#                        implied by ptype, so are not split out separately)
# *_acc columns are home-zone accessibility, attached by
# prepare_frequency_data() when `landuse` is supplied.
DEFAULT_FEATURES_BY_PURPOSE = {
    # Combined non-mandatory count — all six nonmand purposes aggregated.
    # Mirrors Stage 5's work/edu/nonmand segment structure.
    'nonmand': ['incp', 'vehrd', 'nprd', 'walk_acc', 'transit_acc', 'is_female',
                'has_vehicle', 'auto_nonmand_acc', 'auto_shop_acc', 'age_older',
                'mandatory_role', 'hh_child', 'ptype_Retired', 'age_senior',
                'age_working', 'age_working'],
    'work': ['incp', 'vehrd', 'CARLICENCE', 'mandatory_role',
             'ptype_FullTimeWorker', 'ptype_PartTimeWorker', 'ptype_WorkingStudent',
             'workstat_unemployed', 'workstat_carer', 'workstat_illness',
             'workstat_volunteer', 'workstat_other_reason',
             'auto_work_acc', 'n_workers_hh', 'wfh'],
    'edu': ['studying_primary', 'studying_secondary', 'studying_tertiary_pt', 'studying_tertiary_ft',
            'ptype_SchoolChild', 'ptype_UniversityStudent',
            'ptype_WorkingStudent', 'age_child', 'age_teen', 'transit_acc',
            'walk_acc', 'vehrd'],
    'escort': ['hh_child', 'nprd', 'vehrd', 'has_vehicle', 'is_female',
               'workstat_unemployed', 'workstat_carer', 'workstat_illness',
               'workstat_volunteer', 'workstat_other_reason', 'auto_nonmand_acc'],
    'shopping': ['incp', 'vehrd', 'has_vehicle', 'auto_shop_acc', 'nprd', 'age_older'],
    'social': ['nprd', 'is_female', 'vehrd', 'walk_acc', 'transit_acc', 'age_teen'],
    'othmaint': ['nprd', 'vehrd', 'has_vehicle', 'auto_nonmand_acc',
                  'workstat_unemployed', 'workstat_carer', 'workstat_illness',
                  'workstat_volunteer', 'workstat_other_reason'],
    'othdiscr': ['incp', 'vehrd', 'auto_nonmand_acc', 'age_older', 'walk_acc'],
    'eat': ['incp', 'nprd', 'walk_acc', 'transit_acc', 'is_female'],
}


def _stage1_features_for_purpose(data: pd.DataFrame, purpose: str, feature_override=None) -> list:
    if isinstance(feature_override, dict):
        names = feature_override.get(purpose, feature_override.get('*', []))
    elif feature_override is not None:
        names = feature_override
    else:
        names = DEFAULT_FEATURES_BY_PURPOSE.get(purpose, PERSON_FEATURES)
    if not names:
        names = PERSON_FEATURES
    return [c for c in dict.fromkeys(names) if c in data.columns]


def _numerical_hessian(func, params: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian for small MLE problems."""
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
# 0.  PERSON TYPE ASSIGNMENT
# ---------------------------------------------------------------------------

def assign_ptype_column(persons: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'ptype' column to the persons DataFrame.

    Uses assign_ptype() from your existing code if the columns are present,
    otherwise falls back to PTYPE_COLS rules above.

    Parameters
    ----------
    persons : processed persons DataFrame from process_survey()

    Returns
    -------
    persons with 'ptype' column added
    """
    df = persons.copy()

    # try the row-wise assign_ptype if the right columns exist
    required = {'AGEGROUP', 'SEX', 'WORKSTATUS', 'MAINACT'}
    if required.issubset(df.columns):
        try:
            from data_connect import assign_ptype   # your existing function
            df['ptype'] = df.apply(assign_ptype, axis=1)
            logger.info("ptype assigned via assign_ptype()")
            return df
        except ImportError:
            pass

    # ensure required numeric columns exist before applying masks
    if 'WORKSTAT' not in df.columns:
        df['WORKSTAT'] = 0
    if 'STUDYING' not in df.columns:
        df['STUDYING'] = 0
    df['WORKSTAT'] = pd.to_numeric(df['WORKSTAT'], errors='coerce').fillna(0).astype(int)
    df['STUDYING']  = pd.to_numeric(df['STUDYING'],  errors='coerce').fillna(0).astype(int)

    # fallback: rule-based from processed columns
    df['ptype'] = 'NonWorker'
    for label, mask_fn in PTYPE_COLS.items():
        try:
            mask = mask_fn(df)
            df.loc[mask, 'ptype'] = label
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"ptype rule '{label}' skipped: {e}")

    counts = df['ptype'].value_counts().to_dict()
    logger.info(f"ptype distribution: {counts}")
    return df


def _add_categorical_person_features(persons: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-category binary columns from 'ptype', WORKSTAT and STUDYING,
    plus a few other behavioural dummies referenced by
    DEFAULT_FEATURES_BY_PURPOSE. WORKSTAT/STUDYING are nominal codes — see the
    comment above DEFAULT_FEATURES_BY_PURPOSE for why these replace the raw
    ordinal columns as model features.

    Requires 'ptype' to already be set (call assign_ptype_column() first).
    """
    df = persons.copy()

    for pt in ALL_PTYPES:
        df[f'ptype_{pt}'] = (df['ptype'] == pt).astype(int)

    studying = pd.to_numeric(df.get('STUDYING', 0), errors='coerce').fillna(0).astype(int)
    for code, name in {1: 'primary', 2: 'secondary', 3: 'tertiary_pt', 4: 'tertiary_ft'}.items():
        df[f'studying_{name}'] = (studying == code).astype(int)

    # WORKSTAT 1/2/5 (part-time/full-time/retired) are already implied by
    # ptype via PTYPE_COLS — only the remaining "not working" reasons carry
    # information not captured by the ptype segmentation.
    workstat = pd.to_numeric(df.get('WORKSTAT', 0), errors='coerce').fillna(0).astype(int)
    for code, name in {0: 'unemployed', 3: 'other_reason', 4: 'carer', 8: 'volunteer'}.items():
        df[f'workstat_{name}'] = (workstat == code).astype(int)
    df['workstat_illness'] = workstat.isin([6, 7]).astype(int)

    df['mandatory_role'] = df['ptype'].isin([
        'FullTimeWorker', 'PartTimeWorker', 'SchoolChild',
        'UniversityStudent', 'WorkingStudent']).astype(int)

    if 'sexp' in df.columns:
        df['is_female'] = (pd.to_numeric(df['sexp'], errors='coerce') == 2).astype(int)

    if 'vehrd' in df.columns:
        df['has_vehicle'] = (pd.to_numeric(df['vehrd'], errors='coerce').fillna(0) > 0).astype(int)

    if 'AGEGROUP' in df.columns:
        # Same AGEGROUP cutoffs as _enrich_ga_data() in ga_hypothesis_testing.py
        # (5-yr bands, 1=0-4 .. 21=100+) — kept consistent so these columns mean
        # the same thing whichever code path creates them first.
        age = pd.to_numeric(df['AGEGROUP'], errors='coerce').fillna(6)
        df['age_child']  = (age <= 2).astype(int)   # 0-9
        df['age_teen']   = (age == 3).astype(int)   # 10-14
        df['age_older']  = (age >= 8).astype(int)   # 40+
        df['age_senior'] = (age >= 10).astype(int)  # 50+

    if 'RELATIONSHIP' in df.columns:
        rel = df['RELATIONSHIP'].astype(str).str.lower()
        df['hh_child'] = rel.isin(['child', 'son', 'daughter']).astype(int)

    return df


# ---------------------------------------------------------------------------
# 1.  BUILD ACTIVITY COUNTS FROM TOURS
# ---------------------------------------------------------------------------

def build_activity_counts(tours: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-tour activity counts to one row per person.

    Expects tours to have n_work_out, n_edu_out … columns (from your
    build_as_tours / count_stops pipeline). Falls back to counting from
    main_purpose if the n_*_out columns are absent.

    Returns
    -------
    One-row-per-person DataFrame with n_{purpose}_out columns.
    """
    df = tours.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # ── resolve count columns ─────────────────────────────────────
    # Prefer _total (outbound + inbound combined) if available,
    # fall back to _out (outbound only), then derive from main_purpose.
    # This ensures inbound stops (activities on way HOME) are counted.
    for purpose in PURPOSES:
        total_col = f'n_{purpose}_total'
        out_col   = f'n_{purpose}_out'
        in_col    = f'n_{purpose}_in'
        if total_col in df.columns:
            # already combined — use as-is
            df[out_col] = df[total_col]
        elif out_col in df.columns and in_col in df.columns:
            # combine outbound + inbound
            df[out_col] = df[out_col].fillna(0) + df[in_col].fillna(0)
        elif out_col not in df.columns:
            # last resort — derive from main_purpose
            purp_series = df['main_purpose'] if 'main_purpose' in df.columns \
                          else pd.Series('', index=df.index)
            df[out_col] = (purp_series == purpose).astype(int)

    count_cols = [f'n_{p}_out' for p in PURPOSES]

    agg = {col: (col, 'sum') for col in count_cols}
    agg['n_tours'] = ('tour_id', 'max')

    person_counts = (
        df.groupby('person_id')
        .agg(**agg)
        .reset_index()
    )

    # cap at realistic maximums — guard against data errors only
    for purpose in PURPOSES:
        col = f'n_{purpose}_out'
        cap = MAX_COUNT.get(purpose, 5)
        person_counts[col] = person_counts[col].clip(upper=cap)

    # Aggregate combined non-mandatory count for the consolidated model
    _dc = [f'n_{p}_out' for p in NONMAND_PURPOSES if f'n_{p}_out' in person_counts.columns]
    if _dc:
        person_counts['n_nonmand_out'] = person_counts[_dc].sum(axis=1).clip(upper=MAX_COUNT['nonmand'])

    return person_counts


# ---------------------------------------------------------------------------
# 2.  PREPARE ESTIMATION DATA
# ---------------------------------------------------------------------------

def prepare_frequency_data(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Join activity counts with person attributes ready for model estimation.

    Parameters
    ----------
    tours   : from process_survey()["tours"]
    persons : from process_survey()["persons"]  (already processed)
    landuse : enriched landuse from compute_accessibility.run_accessibility(),
              with *_acc columns. If supplied, home-zone accessibility is
              attached to each person via OTAZ/ORDER for use as features.

    Returns
    -------
    One-row-per-person DataFrame with ptype, n_{purpose}_out, and every
    column referenced by DEFAULT_FEATURES_BY_PURPOSE that is available.
    """
    counts = build_activity_counts(tours)

    pers = persons.copy()
    if 'person_id' not in pers.columns and 'PERSID' in pers.columns:
        pers = pers.rename(columns={'PERSID': 'person_id'})

    pers = assign_ptype_column(pers)
    pers = _add_categorical_person_features(pers)

    if landuse is not None:
        from synth_attributes import attach_person_accessibility
        pers = attach_person_accessibility(pers, landuse)

    # keep every column any purpose's feature list might reference, plus the
    # base PERSON_FEATURES — _stage1_features_for_purpose() filters down to
    # whatever's actually present, but a column has to survive this join to
    # be eligible at all. All *_acc columns are kept too (regardless of
    # whether a purpose's default list mentions them yet) so GA feature
    # search can consider any accessibility measure as a candidate.
    wanted = set(PERSON_FEATURES)
    for feats in DEFAULT_FEATURES_BY_PURPOSE.values():
        wanted.update(feats)
    wanted.update(c for c in pers.columns if c.endswith('_acc'))
    keep = ['person_id', 'ptype'] + [c for c in wanted if c in pers.columns]
    pers = pers[keep]

    df = counts.merge(pers, on='person_id', how='left')

    # fill any missing feature columns with sensible defaults
    defaults = {'sexp': 1, 'vehrd': 1, 'nprd': 2, 'STUDYING': 0,
                'WORKSTAT': 0, 'incp': 8}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = df[col].fillna(val)

    logger.info(f"Estimation dataset: {len(df)} persons, "
                f"{df['ptype'].nunique()} ptypes")
    return df


# ---------------------------------------------------------------------------
# 3.  ORDERED LOGIT MODEL
# ---------------------------------------------------------------------------

class OrderedLogit:
    """
    Simple ordered logit (proportional odds) model.
    Estimated via scipy L-BFGS-B with analytical gradients.

    For each purpose we model P(count >= k) using a cumulative logit.
    Thresholds are stored as log-diffs to keep them ordered.
    """

    def __init__(self):
        self.coef_      = None   # (K,) beta coefficients
        self.thresholds_= None   # (J-1,) ordered thresholds
        self.classes_   = None   # sorted unique outcome values
        self.feature_names_ = None
        self.converged_ = False
        self.std_err_   = None
        self.t_stat_    = None
        self.p_value_   = None

    def _pack(self, beta, raw_cuts):
        return np.concatenate([beta, raw_cuts])

    def _unpack(self, params, K):
        return params[:K], params[K:]

    def _cuts_from_raw(self, raw_cuts):
        """Ensure thresholds are strictly ordered via cumulative exp."""
        return np.cumsum(np.concatenate([[raw_cuts[0]],
                                          np.exp(raw_cuts[1:])]))

    def _log_likelihood(self, params, X, y, K, ridge=0.0):
        from scipy.special import expit
        beta, raw_cuts = self._unpack(params, K)
        cuts = self._cuts_from_raw(raw_cuts)
        eta  = X @ beta                          # (N,)

        ll = 0.0
        for j, cls in enumerate(self.classes_):
            mask = (y == cls)
            if not mask.any():
                continue
            eta_j = eta[mask]
            if j == 0:
                p = expit(cuts[0] - eta_j)
            elif j == len(self.classes_) - 1:
                p = 1 - expit(cuts[-1] - eta_j)
            else:
                p = expit(cuts[j] - eta_j) - expit(cuts[j-1] - eta_j)
            p = np.clip(p, 1e-10, 1.0)
            ll += np.sum(np.log(p))

        nll = -ll   # minimise negative LL
        # ── Ridge (L2) penalty on slopes (NOT thresholds) ────────────────────
        # Quasi-complete separation in tiny (purpose, ptype) segments otherwise
        # sends coefficients to ±hundreds (e.g. walk_acc=-404 with t≈0). A small
        # absolute ridge adds curvature 2·ridge to the objective: negligible for
        # a well-identified coefficient (LL curvature ~ N dominates) but it pins
        # a coefficient the data cannot identify toward 0. Slopes only — the
        # ordinal thresholds must stay free.
        if ridge:
            nll = nll + 0.5 * float(ridge) * float(np.dot(beta, beta))
        return nll

    def fit(self, X: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize

        self.classes_ = np.sort(np.unique(y))
        J = len(self.classes_)
        K = X.shape[1]

        if J < 2:
            # degenerate — all outcomes the same, skip
            self.coef_       = np.zeros(K)
            self.thresholds_ = np.array([0.0])
            self.converged_  = False
            return self

        # initialise thresholds at empirical quantiles of eta=0
        probs = np.array([(y <= c).mean() for c in self.classes_[:-1]])
        probs = np.clip(probs, 0.01, 0.99)
        from scipy.special import logit as logit_fn
        init_cuts = logit_fn(probs)
        # convert to raw (log-diff) parameterisation
        raw_cuts = np.concatenate([[init_cuts[0]],
                                    np.log(np.diff(init_cuts) + 1e-4)])
        init = np.concatenate([np.zeros(K), raw_cuts])

        # Ridge weight + hard slope bounds guard against separation blow-up in
        # tiny segments. GA_S1_RIDGE tunes shrinkage; GA_S1_COEF_CAP the bound.
        _ridge = float(os.environ.get('GA_S1_RIDGE', '2.0'))
        _cap   = float(os.environ.get('GA_S1_COEF_CAP', '10.0'))
        _bounds = [(-_cap, _cap)] * K + [(None, None)] * (len(init) - K)

        res = minimize(
            self._log_likelihood,
            init,
            args=(X, y, K, _ridge),
            method='L-BFGS-B',
            bounds=_bounds,
            options={'maxiter': 500, 'ftol': 1e-9},
        )

        self.coef_       = res.x[:K]
        self.thresholds_ = self._cuts_from_raw(res.x[K:])
        self.converged_  = res.success
        self.n_obs_      = int(len(y))
        self.ll_         = float(-res.fun)
        # null LL: uniform probabilities
        self.ll0_        = float(len(y) * np.log(1.0 / J))
        try:
            hess = _numerical_hessian(lambda p: self._log_likelihood(p, X, y, K, _ridge), res.x)
            cov = np.linalg.pinv(hess)
            params = np.concatenate([self.coef_, self.thresholds_])
            raw_se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            # Report thresholds on the transformed scale; use the same variance slots.
            self.std_err_ = raw_se
            self.t_stat_ = np.divide(
                params,
                self.std_err_,
                out=np.full_like(params, np.nan, dtype=float),
                where=self.std_err_ > 0,
            )
            self.p_value_ = 2.0 * norm.sf(np.abs(self.t_stat_))
        except Exception:
            size = K + max(J - 1, 0)
            self.std_err_ = np.full(size, np.nan)
            self.t_stat_ = np.full(size, np.nan)
            self.p_value_ = np.full(size, np.nan)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y == class_j) for each row. Shape (N, J)."""
        from scipy.special import expit
        eta  = X @ self.coef_
        cuts = self.thresholds_
        J    = len(self.classes_)
        probs = np.zeros((len(X), J))

        for j in range(J):
            if j == 0:
                probs[:, j] = expit(cuts[0] - eta)
            elif j == J - 1:
                probs[:, j] = 1 - expit(cuts[-1] - eta)
            else:
                probs[:, j] = expit(cuts[j] - eta) - expit(cuts[j-1] - eta)

        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return most-likely count (argmax of predicted probabilities)."""
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def predict_sampled(self, X: np.ndarray, seed: int = None) -> np.ndarray:
        """Sample a count from the predicted probability distribution."""
        rng   = np.random.default_rng(seed)
        proba = self.predict_proba(X)
        J     = len(self.classes_)
        out   = np.array([
            self.classes_[rng.choice(J, p=proba[i])]
            for i in range(len(X))
        ])
        return out


# ---------------------------------------------------------------------------
# 4.  ESTIMATE MODELS
# ---------------------------------------------------------------------------

def _get_feature_matrix(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Build standardised feature matrix from DataFrame."""
    X = df[feature_cols].fillna(0).values.astype(float)
    # standardise each column
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def estimate_frequency_models(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    min_sample: int = 30,
    feature_override: list = None,
    landuse: pd.DataFrame = None,
    fit_nonmand_combined: bool = True,
) -> dict:
    """
    Estimate one ordered logit per (purpose, ptype) combination.

    Parameters
    ----------
    tours, persons       : from process_survey()
    min_sample           : skip estimation if fewer than this many observations
    feature_override     : if supplied, use this list of features instead of
                           PERSON_FEATURES — passed from GA results to apply
                           GA-selected variable sets.
    landuse              : enriched landuse with *_acc columns, passed through
                           to prepare_frequency_data() for home-zone
                           accessibility features.
    fit_nonmand_combined : if True (default), also estimate a single ordered
                           logit for n_nonmand_out = sum(all six nonmand purposes)
                           and store Dirichlet split proportions in models['_nonmand']
                           and models['_nonmand_split'].  predict_activity_frequency()
                           uses these to replace the six individual predictions
                           with a combined-count + multinomial-split prediction.
    """
    data = prepare_frequency_data(tours, persons, landuse=landuse)
    models = {}

    for purpose in PURPOSES:
        col = f'n_{purpose}_out'
        models[purpose] = {}

        # Only estimate for ptypes that logically can make trips of this purpose
        _valid_set = VALID_PTYPES_FOR_PURPOSE.get(purpose, None)
        ptypes = [
            p for p in data['ptype'].unique()
            if _valid_set is None or p in _valid_set
        ]

        for ptype in ptypes:
            subset = data[data['ptype'] == ptype].copy()
            feature_cols = _stage1_features_for_purpose(data, purpose, feature_override)

            if len(subset) < min_sample:
                logger.warning(
                    f"  {purpose}/{ptype}: only {len(subset)} obs — skipping"
                )
                continue

            y = subset[col].values.astype(int)
            # Bin to at most ORDERED_LOGIT_MAX_CLASSES categories so thresholds
            # represent low / medium / high counts, not every individual value.
            _max_cls = ORDERED_LOGIT_MAX_CLASSES.get(purpose, 4)
            y = np.where(y >= _max_cls - 1, _max_cls - 1, y)

            # skip if everyone has 0 activities (not relevant for this ptype)
            if y.max() == 0:
                logger.info(f"  {purpose}/{ptype}: all zeros — skipping")
                continue

            X_raw = subset[feature_cols].fillna(0).values.astype(float)
            # ── Drop zero-variance columns BEFORE fitting ────────────────────
            # A feature that is constant within this (purpose, ptype) segment
            # (e.g. ptype_FullTimeWorker inside the WorkingStudent segment, or a
            # workstat_* dummy that is all-0 here) carries no information: its
            # coefficient is stuck at 0.0 with p≈1/NaN and it survives the
            # magnitude/p-value prune below because removing it can fail the
            # threshold-gap acceptance check. Removing it up front is the only
            # reliable way to keep these dead 0.0 terms out of the final model.
            _sd0 = X_raw.std(axis=0)
            _keep0 = _sd0 > 1e-9
            if not _keep0.all():
                _dropped0 = [feature_cols[i] for i in range(len(feature_cols)) if not _keep0[i]]
                logger.info(f"  {purpose}/{ptype}: dropping {len(_dropped0)} zero-variance "
                            f"feature(s): {_dropped0}")
                feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if _keep0[i]]
                X_raw = X_raw[:, _keep0]
            if X_raw.shape[1] == 0:
                logger.info(f"  {purpose}/{ptype}: all features zero-variance — skipping")
                continue
            mu = X_raw.mean(axis=0)
            sd = X_raw.std(axis=0)
            sd[sd == 0] = 1.0
            X = (X_raw - mu) / sd

            model = OrderedLogit()
            model.fit(X, y)
            # ── Prune insignificant parameters (magnitude AND p-value) ────────
            # Drop features that are near-zero OR statistically insignificant,
            # then re-fit so standard errors reflect the trimmed specification.
            # Iterate until no further pruning is possible (convergence in ≤3 rounds).
            _zthresh  = float(os.environ.get('GA_SIG_COEF_ZERO',   '0.01'))
            _pthresh  = float(os.environ.get('GA_SIG_PVAL_THRESH',  '0.10'))
            for _prune_round in range(3):
                if len(feature_cols) <= 1:
                    break
                coef_arr = np.asarray(model.coef_).ravel()
                pvals    = getattr(model, 'p_value_', None)
                # Keep a feature if its coefficient is large enough AND significant
                keep_mask = np.ones(len(feature_cols), dtype=bool)
                for _i in range(len(feature_cols)):
                    near_zero = abs(coef_arr[_i]) < _zthresh
                    insig     = (pvals is not None and _i < len(pvals)
                                 and np.isfinite(pvals[_i]) and pvals[_i] > _pthresh)
                    if near_zero or insig:
                        keep_mask[_i] = False
                keep_idx = np.where(keep_mask)[0]
                if len(keep_idx) == len(feature_cols):
                    break  # nothing to prune
                if len(keep_idx) == 0:
                    break  # would remove everything — stop
                pruned_cols = [feature_cols[_i] for _i in keep_idx]
                Xp = X[:, keep_idx]
                try:
                    mp = OrderedLogit()
                    mp.fit(Xp, y)
                    if hasattr(mp, 'thresholds_') and len(mp.thresholds_) >= 2:
                        gaps = np.diff(np.sort(mp.thresholds_))
                        if gaps.min() >= 0.05:
                            model        = mp
                            feature_cols = pruned_cols
                            X            = Xp
                            mu           = mu[keep_idx]
                            sd           = sd[keep_idx]
                except Exception:
                    break
            model.feature_names_ = feature_cols

            models[purpose][ptype] = {
                'model':    model,
                'mu':       mu,
                'sd':       sd,
                'features': feature_cols,
                'n_obs':    len(subset),
                'converged': model.converged_,
            }

            status = 'OK' if model.converged_ else 'WARN'
            logger.info(
                f"  [{status}] {purpose}/{ptype}: "
                f"n={len(subset)}, classes={model.classes_}"
            )

    n_fitted = sum(len(v) for v in models.values())
    print(f"Stage 1: fitted {n_fitted} ordered logit models "
          f"({len(PURPOSES)} purposes × ptypes)")

    # ── Combined non-mandatory count model (work/edu/nonmand GA segment) ────────
    # This gives the total n_nonmand_out count.  Individual nonmand purposes are
    # already fitted above in the PURPOSES loop with their own ptype-filtered
    # ordered-logit models.  The Dirichlet split is kept as a fallback only.
    if fit_nonmand_combined and 'n_nonmand_out' in data.columns:
        models['_nonmand'] = {}
        _nm_valid = VALID_PTYPES_FOR_PURPOSE.get('nonmand', None)
        _nm_ptypes = [
            p for p in data['ptype'].dropna().unique()
            if _nm_valid is None or p in _nm_valid
        ]
        for ptype in _nm_ptypes:
            subset = data[data['ptype'] == ptype].copy()
            if len(subset) < min_sample:
                continue
            y = subset['n_nonmand_out'].values.astype(int)
            _nm_max_cls = ORDERED_LOGIT_MAX_CLASSES.get('nonmand', 4)
            y = np.where(y >= _nm_max_cls - 1, _nm_max_cls - 1, y)
            if y.max() == 0:
                continue
            feature_cols = _stage1_features_for_purpose(data, 'nonmand', feature_override)
            if not feature_cols:
                break
            X_raw = subset[feature_cols].fillna(0).values.astype(float)
            mu = X_raw.mean(axis=0); sd = X_raw.std(axis=0); sd[sd == 0] = 1.0
            X = (X_raw - mu) / sd
            m = OrderedLogit()
            m.fit(X, y)
            # p-value + magnitude pruning (same as per-purpose loop above)
            _zthresh = float(os.environ.get('GA_SIG_COEF_ZERO',  '0.01'))
            _pthresh = float(os.environ.get('GA_SIG_PVAL_THRESH', '0.10'))
            for _ in range(3):
                if len(feature_cols) <= 1:
                    break
                _c = np.asarray(m.coef_).ravel()
                _pv = getattr(m, 'p_value_', None)
                _km = np.ones(len(feature_cols), dtype=bool)
                for _i in range(len(feature_cols)):
                    if abs(_c[_i]) < _zthresh or (
                        _pv is not None and _i < len(_pv)
                        and np.isfinite(_pv[_i]) and _pv[_i] > _pthresh
                    ):
                        _km[_i] = False
                _ki = np.where(_km)[0]
                if len(_ki) in (0, len(feature_cols)):
                    break
                try:
                    _mp = OrderedLogit()
                    _mp.fit(X[:, _ki], y)
                    m = _mp; feature_cols = [feature_cols[i] for i in _ki]
                    X = X[:, _ki]; mu = mu[_ki]; sd = sd[_ki]
                except Exception:
                    break
            m.feature_names_ = feature_cols
            models['_nonmand'][ptype] = {
                'model': m, 'mu': mu, 'sd': sd, 'features': feature_cols,
                'n_obs': len(subset), 'converged': m.converged_,
            }
            logger.info(f"  [nonmand/{ptype}] classes={m.classes_} feats={len(feature_cols)} n={len(subset)}")

        # Dirichlet split kept as fallback; individual nonmand models (fitted in the
        # PURPOSES loop above) are the primary prediction path.
        models['_nonmand_split'] = estimate_nonmand_split(data)
        n_nonmand_fitted = len(models['_nonmand'])
        print(f"  + nonmand count model: {n_nonmand_fitted} ptype models "
              f"(individual purpose models fitted in main loop above)")

    return models


# ---------------------------------------------------------------------------
# 4b.  NON-MANDATORY SPLIT HELPERS
# ---------------------------------------------------------------------------

def estimate_nonmand_split(data: pd.DataFrame) -> dict:
    """
    Estimate ptype-conditional proportions for splitting n_nonmand_out into
    the six individual non-mandatory purposes via multinomial sampling.

    Proportions are the observed share of each component in total nonmand
    activity count per ptype, from HHTS persons with at least one nonmand trip.

    Returns
    -------
    dict mapping ptype → np.ndarray shape (6,) summing to 1.0.
    '_default' key holds global proportions as fallback.
    """
    nonmand_cols = [f'n_{p}_out' for p in NONMAND_PURPOSES]
    _avail = [c for c in nonmand_cols if c in data.columns]
    if not _avail:
        unif = np.ones(len(NONMAND_PURPOSES)) / len(NONMAND_PURPOSES)
        return {'_default': unif}

    participants = data[data[_avail].sum(axis=1) > 0].copy()
    split_props: dict = {}

    for ptype, grp in participants.groupby('ptype'):
        totals = np.array([grp[c].fillna(0).sum() for c in _avail], dtype=float)
        if totals.sum() > 0:
            split_props[ptype] = totals / totals.sum()
        else:
            split_props[ptype] = np.ones(len(NONMAND_PURPOSES)) / len(NONMAND_PURPOSES)

    global_totals = np.array([participants[c].fillna(0).sum() for c in _avail], dtype=float)
    split_props['_default'] = (global_totals / global_totals.sum()
                               if global_totals.sum() > 0
                               else np.ones(len(NONMAND_PURPOSES)) / len(NONMAND_PURPOSES))

    logger.info("Nonmand split proportions by ptype:")
    for pt, props in split_props.items():
        if pt == '_default':
            continue
        logger.info("  %-22s %s" % (pt, "  ".join(f"{p}={v:.3f}" for p, v in zip(NONMAND_PURPOSES, props))))

    return split_props


def predict_nonmand_split(
    synth: pd.DataFrame,
    split_props: dict,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Distribute each person's n_nonmand_out across the six non-mandatory purposes
    by ptype-conditional multinomial sampling.

    Overwrites n_escort_out, n_shopping_out, n_social_out, n_othmaint_out,
    n_othdiscr_out, n_eat_out.  Persons with n_nonmand_out == 0 get 0 for all.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = synth.copy()
    for p in NONMAND_PURPOSES:
        df[f'n_{p}_out'] = 0

    _default_props = split_props.get('_default',
                                     np.ones(len(NONMAND_PURPOSES)) / len(NONMAND_PURPOSES))

    for ptype, grp in df.groupby('ptype'):
        props = split_props.get(ptype, _default_props)
        n_vals = grp['n_nonmand_out'].fillna(0).values.astype(int)
        results = np.zeros((len(grp), len(NONMAND_PURPOSES)), dtype=int)

        # Batch by unique n to vectorise multinomial sampling
        for n in np.unique(n_vals):
            if n <= 0:
                continue
            mask = n_vals == n
            results[mask] = rng.multinomial(n, props, size=int(mask.sum()))

        for j, purpose in enumerate(NONMAND_PURPOSES):
            df.loc[grp.index, f'n_{purpose}_out'] = results[:, j]

    return df


# ---------------------------------------------------------------------------
# 5.  PREDICT — SYNTHETIC POPULATION
# ---------------------------------------------------------------------------

def _get_fallback_model(models: dict, purpose: str) -> Optional[dict]:
    """
    Return the model entry with the most observations for this purpose.
    Used when the person's ptype has no fitted model.
    """
    if not models.get(purpose):
        return None
    best = max(models[purpose].values(), key=lambda m: m['n_obs'])
    return best


def predict_activity_frequency(
    synth: pd.DataFrame,
    models: dict,
    sample: bool = True,
    seed: int = 42,
    landuse: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Apply fitted models to a synthetic population.

    Parameters
    ----------
    synth   : synthetic population DataFrame. Must have person attributes
              matching PERSON_FEATURES (missing ones default to 0).
              Must have 'ptype' column — or we derive it from available cols.
    models  : output of estimate_frequency_models()
    sample  : if True, sample from predicted distribution (recommended);
              if False, return the modal prediction
    seed    : random seed for sampling
    landuse : enriched landuse with *_acc columns — attached via OTAZ/ORDER
              so synth predictions use the same accessibility features the
              models were estimated with.

    Returns
    -------
    synth copy with n_{purpose}_out columns added.
    """
    df = synth.copy()
    rng = np.random.default_rng(seed)

    # ensure person_id exists
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})
    if 'person_id' not in df.columns:
        df['person_id'] = df.index

    # derive ptype if not present
    if 'ptype' not in df.columns:
        df = assign_ptype_column(df)

    # build the same categorical/accessibility features used at estimation
    # time so the fitted coefficients apply meaningfully to the synth pop.
    df = _add_categorical_person_features(df)
    if landuse is not None:
        from synth_attributes import attach_person_accessibility
        df = attach_person_accessibility(df, landuse)

    # initialise output columns
    for purpose in PURPOSES:
        df[f'n_{purpose}_out'] = 0
    df['n_nonmand_out'] = 0

    # When the combined nonmand model is present, skip all six nonmand purposes
    # in the per-purpose loop and predict them via the combined path below.
    _use_nonmand_model = '_nonmand' in models and bool(models['_nonmand'])
    _skip_purposes = set(NONMAND_PURPOSES) if _use_nonmand_model else set()

    # process each ptype group separately
    for ptype, grp in df.groupby('ptype'):
        idx = grp.index

        # Map new HHTS ptypes (Preschool, JointWorkStudy) to closest estimation ptype
        lookup_ptype = PTYPE_PREDICTION_MAP.get(ptype, ptype)

        for purpose in PURPOSES:
            if purpose in _skip_purposes:
                continue
            col = f'n_{purpose}_out'

            # per-(ptype, purpose) override takes precedence over the global map
            purpose_lookup = PTYPE_PURPOSE_PREDICTION_MAP.get(
                (ptype, purpose), lookup_ptype)

            # use purpose-specific lookup first, then global map, then fallback
            entry = (models.get(purpose, {}).get(purpose_lookup)
                     or models.get(purpose, {}).get(lookup_ptype)
                     or models.get(purpose, {}).get(ptype)
                     or _get_fallback_model(models, purpose))

            if entry is None:
                # no model at all for this purpose — leave as 0
                continue

            feature_cols = entry['features']
            mu           = entry['mu']
            sd           = entry['sd']
            model        = entry['model']

            # build feature matrix — fill missing columns with 0
            X_raw = np.zeros((len(grp), len(feature_cols)))
            for j, col_name in enumerate(feature_cols):
                if col_name in grp.columns:
                    X_raw[:, j] = grp[col_name].fillna(0).values.astype(float)

            X = (X_raw - mu) / sd

            if sample:
                proba = model.predict_proba(X)
                J     = len(model.classes_)
                preds = np.array([
                    model.classes_[rng.choice(J, p=proba[i])]
                    for i in range(len(grp))
                ])
            else:
                preds = model.predict(X)

            df.loc[idx, col] = preds

    # ── Combined non-mandatory count prediction ───────────────────────────────
    if _use_nonmand_model:
        _nonmand_models = models['_nonmand']
        _nonmand_fallback = (max(_nonmand_models.values(), key=lambda m: m['n_obs'])
                             if _nonmand_models else None)
        for ptype, grp in df.groupby('ptype'):
            idx = grp.index
            lookup_ptype = PTYPE_PREDICTION_MAP.get(ptype, ptype)
            entry = (_nonmand_models.get(lookup_ptype)
                     or _nonmand_models.get(ptype)
                     or _nonmand_fallback)
            if entry is None:
                continue
            feature_cols = entry['features']
            X_raw = np.zeros((len(grp), len(feature_cols)))
            for j, col_name in enumerate(feature_cols):
                if col_name in grp.columns:
                    X_raw[:, j] = grp[col_name].fillna(0).values.astype(float)
            X = (X_raw - entry['mu']) / entry['sd']
            if sample:
                proba = entry['model'].predict_proba(X)
                J = len(entry['model'].classes_)
                preds = np.array([
                    entry['model'].classes_[rng.choice(J, p=proba[i])]
                    for i in range(len(grp))
                ])
            else:
                preds = entry['model'].predict(X)
            df.loc[idx, 'n_nonmand_out'] = preds

    # ── Preschool constraints ────────────────────────────────────────────────
    # Preschool children attend childcare/kindergarten (modelled as edu).
    # They cannot work or participate in independent discretionary activities.
    # The NonWorker fallback model may assign work or othdiscr — zero those out.
    preschool_mask = df['ptype'] == 'Preschool'
    if preschool_mask.any():
        df.loc[preschool_mask, 'n_work_out']    = 0
        df.loc[preschool_mask, 'n_othdiscr_out'] = 0
        # Preschool children attend childcare ~35% of survey days (observed
        # HHTS rate).  SchoolChild model predicts ~1.0 edu/day which leads
        # to ~100% attendance — wrong.  Instead: clip at 1, then
        # probabilistically target 35% attendance by randomly zeroing excess
        # assignments or randomly promoting zeros to 1.
        df.loc[preschool_mask, 'n_edu_out'] = (
            df.loc[preschool_mask, 'n_edu_out'].clip(upper=1)
        )
        n_ps = int(preschool_mask.sum())
        n_target_edu = max(1, int(round(0.35 * n_ps)))
        ps_edu1_mask = preschool_mask & (df['n_edu_out'] == 1)
        ps_edu0_mask = preschool_mask & (df['n_edu_out'] == 0)
        n_current_1  = int(ps_edu1_mask.sum())

        if n_current_1 > n_target_edu:
            # too many edu=1 — randomly zero out the excess
            ps_edu1_idx = df.index[ps_edu1_mask]
            n_zero = n_current_1 - n_target_edu
            sel = rng.choice(len(ps_edu1_idx), size=n_zero, replace=False)
            df.loc[ps_edu1_idx[sel], 'n_edu_out'] = 0
        elif n_current_1 < n_target_edu:
            # too few edu=1 — randomly promote zeros to 1
            ps_edu0_idx = df.index[ps_edu0_mask]
            n_add = min(n_target_edu - n_current_1, len(ps_edu0_idx))
            if n_add > 0:
                sel = rng.choice(len(ps_edu0_idx), size=n_add, replace=False)
                df.loc[ps_edu0_idx[sel], 'n_edu_out'] = 1

        n_ps_edu_final = int((preschool_mask & (df['n_edu_out'] == 1)).sum())
        logger.info(
            f"Preschool constraints applied to {n_ps} persons: "
            f"work→0, othdiscr→0, edu targeted ~35% attendance "
            f"({n_ps_edu_final}/{n_ps} assigned edu=1)"
        )

    # ── Mandatory education constraint: full-time university students ────────
    # ActivitySim mandatory_tour_frequency logic: students always have ≥1
    # mandatory (education) tour per simulated day.
    # STUDYING codes: 3=tertiaryPartTime, 4=tertiaryFullTime.
    # Full-time (STUDYING==4): attend campus almost every day → enforce ≥1.
    # Part-time (STUDYING==3): realistically attend 2-3 days/week; the
    #   ordered logit may legitimately predict 0 on some days — leave those.
    # SchoolChild: primary/secondary always attend — enforce ≥1 as well.
    if 'STUDYING' in df.columns:
        _studying = pd.to_numeric(df['STUDYING'], errors='coerce').fillna(0).astype(int)
        # Full-time tertiary (UniversityStudent or WorkingStudent STUDYING==4)
        _ft_uni_mask = (df['ptype'].isin(['UniversityStudent', 'WorkingStudent'])
                        & (_studying == 4)
                        & (df['n_edu_out'] == 0))
        if _ft_uni_mask.any():
            df.loc[_ft_uni_mask, 'n_edu_out'] = 1
            logger.info(
                f"Mandatory edu constraint: {int(_ft_uni_mask.sum())} full-time "
                f"tertiary students (STUDYING=4) with n_edu=0 corrected → 1"
            )
        # School children (primary/secondary) always attend
        _school_mask = (df['ptype'] == 'SchoolChild') & (df['n_edu_out'] == 0)
        if _school_mask.any():
            df.loc[_school_mask, 'n_edu_out'] = 1
            logger.info(
                f"Mandatory edu constraint: {int(_school_mask.sum())} SchoolChild "
                f"with n_edu=0 corrected → 1"
            )
    else:
        # No STUDYING column — fall back to ptype-only: all UniversityStudent get ≥1
        _uni_mask = (df['ptype'] == 'UniversityStudent') & (df['n_edu_out'] == 0)
        if _uni_mask.any():
            df.loc[_uni_mask, 'n_edu_out'] = 1
            logger.info(
                f"Mandatory edu constraint (no STUDYING col): {int(_uni_mask.sum())} "
                f"UniversityStudent with n_edu=0 corrected → 1"
            )

    # ── Mandatory work constraint for employed persons ───────────────────────
    # FullTimeWorker: works every weekday → enforce ≥1 work tour.
    # PartTimeWorker: works ~3 days/week on average → ~60% of sampled days
    #   should have a work tour; use a probabilistic correction rather than
    #   forcing 100% (which would over-produce for a 5-day discrete model).
    _ftw_mask = (df['ptype'] == 'FullTimeWorker') & (df['n_work_out'] == 0)
    if _ftw_mask.any():
        df.loc[_ftw_mask, 'n_work_out'] = 1
        logger.info(
            f"Mandatory work constraint: {int(_ftw_mask.sum())} FullTimeWorker "
            f"with n_work=0 corrected → 1"
        )
    _ptw_zero_idx = df.index[(df['ptype'] == 'PartTimeWorker') & (df['n_work_out'] == 0)]
    if len(_ptw_zero_idx) > 0:
        # Flip ~60% of part-time-worker zeros to 1 (stochastic)
        _n_flip = int(round(0.60 * len(_ptw_zero_idx)))
        if _n_flip > 0:
            _flip_sel = rng.choice(len(_ptw_zero_idx), size=_n_flip, replace=False)
            df.loc[_ptw_zero_idx[_flip_sel], 'n_work_out'] = 1
            logger.info(
                f"Work tour constraint (PTW): {_n_flip}/{len(_ptw_zero_idx)} "
                f"PartTimeWorker with n_work=0 probabilistically corrected → 1 (p=0.60)"
            )

    # ── Debug: show predictions by ptype so under-production is visible ──────
    total_acts = df[[f'n_{p}_out' for p in PURPOSES]].sum(axis=1)
    print(f"\nStage 1 predictions: {len(df)} persons")
    print(f"  Mean activities per person: {total_acts.mean():.2f}  "
          f"(median={total_acts.median():.2f}, "
          f"std={total_acts.std():.2f}, "
          f"min={total_acts.min():.0f}, max={total_acts.max():.0f})")
    print(f"  Activity totals by purpose:")
    for p in PURPOSES:
        col = f'n_{p}_out'
        vals = df[col]
        print(f"    {p:12s}: total={int(vals.sum()):6d}  "
              f"mean={vals.mean():.3f}  "
              f"zero_share={100*(vals==0).mean():.1f}%")

    print(f"\n  Activity rates by ptype:")
    print(f"  {'ptype':20s}  {'n':>6}  " +
          "  ".join(f"{p[:5]:>5}" for p in PURPOSES))
    print(f"  {'-'*20}  {'------':>6}  " + "  ".join("-----" for _ in PURPOSES))
    for pt, grp in df.groupby('ptype'):
        means = [f"{grp[f'n_{p}_out'].mean():5.2f}" for p in PURPOSES]
        print(f"  {str(pt):20s}  {len(grp):6d}  " + "  ".join(means))

    # Warn if any ptype produces zero activities for all mandatory purposes
    for pt, grp in df.groupby('ptype'):
        work_mean = grp['n_work_out'].mean() if 'n_work_out' in grp.columns else 0
        edu_mean  = grp['n_edu_out'].mean()  if 'n_edu_out'  in grp.columns else 0
        if pt in ('FullTimeWorker', 'PartTimeWorker') and work_mean < 0.5:
            logger.warning(
                f"Stage 1: ptype={pt} has mean work trips = {work_mean:.3f} < 0.5 — "
                "workers are under-producing work tours. Check model or fallback."
            )
        if pt in ('SchoolChild',) and edu_mean < 0.5:
            logger.warning(
                f"Stage 1: ptype={pt} has mean edu trips = {edu_mean:.3f} < 0.5 — "
                "students are under-producing education tours."
            )
        # UniversityStudent: mix of full-time and part-time students; part-time
        # students (STUDYING=3) may legitimately have 0 edu on many days.
        # Use a lower threshold of 0.3 to catch genuine model failures.
        if pt == 'UniversityStudent' and edu_mean < 0.30:
            logger.warning(
                f"Stage 1: ptype=UniversityStudent has mean edu trips = {edu_mean:.3f} < 0.30 — "
                "university students are severely under-producing education tours. "
                "Check STUDYING code mapping or mandatory constraint."
            )

    return df


# ---------------------------------------------------------------------------
# 6.  SAVE / LOAD MODELS
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
def save_models(models: dict, path: str = 'saved_data/stage1_models.pkl'):
    """Pickle the fitted models dict and write an HTML coefficient report."""
    import pickle, os
    # Save pkl first — artifact generation is non-fatal.
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {path}")
    try:
        html_path = path.replace('.pkl', '_report.html')
        _write_model_html(models, html_path, 'Stage 1 — Activity Frequency',
                          notes=f'Saved: {path}')
    except Exception as _html_e:
        print(f"  Stage 1 HTML report failed (non-fatal): {_html_e}")
    try:
        from paper_model_forms import write_stage_paper_artifacts
        write_stage_paper_artifacts('stage1', os.path.dirname(path) or 'saved_data')
    except Exception as _paper_e:
        print(f"  Stage 1 paper artifacts failed (non-fatal): {_paper_e}")


def load_models(path: str = 'saved_data/stage1_models.pkl') -> dict:
    """Load pickled models dict."""
    import pickle
    with open(path, 'rb') as f:
        models = pickle.load(f)
    print(f"Models loaded from {path}")
    return models


# ---------------------------------------------------------------------------
# 7.  FULL PIPELINE WRAPPER
# ---------------------------------------------------------------------------

def calibrate_activity_rates(
    synth_pred: pd.DataFrame,
    hhts_persons: pd.DataFrame,
    hhts_tours: pd.DataFrame,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Post-prediction calibration: scale synthetic activity counts so that
    aggregate rates by ptype match HHTS observed rates.

    ActivitySim uses shadow-pricing in destination choice for a similar
    purpose.  Here we apply a simpler but effective correction at the
    frequency stage: for each (ptype, purpose) cell, compute the ratio
    of HHTS mean to predicted mean and use it to probabilistically
    adjust predictions up or down.

    Parameters
    ----------
    synth_pred    : output of predict_activity_frequency()
    hhts_persons  : persons table from process_survey()
    hhts_tours    : tours table from process_survey()
    rng           : random generator (for stochastic rounding)

    Returns
    -------
    synth_pred with n_{purpose}_out columns adjusted.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Build HHTS observed rates per (ptype, purpose)
    try:
        from stage1_activity_frequency import assign_ptype_column as _apc
        hhts_p = assign_ptype_column(hhts_persons.copy())
    except Exception:
        hhts_p = hhts_persons.copy()
        if 'ptype' not in hhts_p.columns:
            hhts_p['ptype'] = 'NonWorker'

    if 'person_id' not in hhts_p.columns and 'PERSID' in hhts_p.columns:
        hhts_p = hhts_p.rename(columns={'PERSID': 'person_id'})
    if 'person_id' not in hhts_tours.columns and 'PERSID' in hhts_tours.columns:
        hhts_tours = hhts_tours.rename(columns={'PERSID': 'person_id'})

    # Compute per-person activity counts from HHTS tours
    hhts_counts = hhts_persons[['person_id']].copy() if 'person_id' in hhts_persons.columns \
        else hhts_p[['person_id']].copy()
    for p in PURPOSES:
        col = f'n_{p}_out'
        if col in hhts_tours.columns:
            cnts = hhts_tours.groupby('person_id')[col].sum().reset_index()
        else:
            # count rows where primary purpose matches
            purp_col = next(
                (c for c in ['primary_purpose', 'main_purpose', 'purpose']
                 if c in hhts_tours.columns), None
            )
            if purp_col:
                mask = hhts_tours[purp_col].str.lower().str.startswith(p[:4])
                cnts = hhts_tours[mask].groupby('person_id').size().reset_index(name=col)
            else:
                hhts_counts[col] = 0
                continue
        hhts_counts = hhts_counts.merge(cnts, on='person_id', how='left')
        hhts_counts[col] = hhts_counts[col].fillna(0)

    # Merge ptype onto hhts_counts
    hhts_counts = hhts_counts.merge(hhts_p[['person_id', 'ptype']], on='person_id', how='left')
    hhts_counts['ptype'] = hhts_counts['ptype'].fillna('NonWorker')

    df = synth_pred.copy()
    n_adj_total = 0

    print("\n  Stage 1 calibration (HHTS target vs predicted):")
    print(f"  {'ptype':20s}  {'purpose':12s}  {'hhts_mean':>9}  {'pred_mean':>9}  {'ratio':>6}  adj")

    for pt in df['ptype'].dropna().unique():
        obs_grp  = hhts_counts[hhts_counts['ptype'] == pt]
        pred_grp = df[df['ptype'] == pt]
        if obs_grp.empty or pred_grp.empty:
            continue

        for p in PURPOSES:
            col = f'n_{p}_out'
            hhts_mean = float(obs_grp[col].mean()) if col in obs_grp.columns else 0.0
            pred_mean = float(pred_grp[col].mean()) if col in pred_grp.columns else 0.0

            if hhts_mean < 0.01 and pred_mean < 0.01:
                continue  # both near zero — nothing to calibrate

            ratio = hhts_mean / max(pred_mean, 0.001)
            adj_flag = ''

            # Only adjust when ratio is meaningfully off (outside ±25%)
            if ratio < 0.75 or ratio > 1.35:
                pred_idx = df.index[df['ptype'] == pt]
                current  = df.loc[pred_idx, col].values.astype(float)

                if ratio > 1.35:
                    # Under-production: probabilistically increment some zeros→1
                    # or ones→2 etc. using stochastic rounding
                    target_total = hhts_mean * len(pred_idx)
                    current_total = current.sum()
                    gap = target_total - current_total
                    # Randomly increment gap persons by 1 (capped at MAX_COUNT)
                    n_increment = min(int(round(gap)), len(pred_idx))
                    if n_increment > 0:
                        inc_idx = rng.choice(len(pred_idx), size=n_increment, replace=False)
                        for ii in inc_idx:
                            current[ii] = min(current[ii] + 1, MAX_COUNT.get(p, 5))
                        adj_flag = f'+{n_increment}'
                        n_adj_total += n_increment

                elif ratio < 0.75:
                    # Over-production: probabilistically decrement
                    target_total = hhts_mean * len(pred_idx)
                    current_total = current.sum()
                    gap = current_total - target_total
                    n_decrement = min(int(round(gap)), int((current > 0).sum()))
                    if n_decrement > 0:
                        eligible = np.where(current > 0)[0]
                        dec_idx = rng.choice(eligible, size=min(n_decrement, len(eligible)), replace=False)
                        for ii in dec_idx:
                            current[ii] = max(current[ii] - 1, 0)
                        adj_flag = f'-{n_decrement}'
                        n_adj_total += n_decrement

                df.loc[pred_idx, col] = current

            new_pred_mean = float(df.loc[df['ptype'] == pt, col].mean())
            if ratio < 0.75 or ratio > 1.35:
                print(f"  {str(pt):20s}  {p:12s}  {hhts_mean:9.3f}  {pred_mean:9.3f}  "
                      f"{ratio:6.2f}  {adj_flag}")

    logger.info(f"Stage 1 calibration: {n_adj_total} activity counts adjusted")
    return df


def run_stage1(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    synth: pd.DataFrame,
    sample: bool = True,
    seed: int = 42,
    save_path: str = 'saved_data/stage1_models.pkl',
    feature_override: list = None,
    landuse: pd.DataFrame = None,
) -> tuple:
    """
    Estimate on HHTS, predict on synthetic population.

    Parameters
    ----------
    tours, persons    : from process_survey()
    synth             : synthetic population (person attributes + OTAZ)
    sample            : sample from distribution (True) or take modal pred (False)
    seed              : random seed
    save_path         : where to pickle the fitted models
    feature_override  : GA-selected feature list — passed to estimate_frequency_models
                        to use the GA's winning variables instead of PERSON_FEATURES.
    landuse           : enriched landuse with *_acc columns — passed through to
                        estimation and prediction so home-zone accessibility is
                        used as a feature.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
    print("=" * 55)
    print("STAGE 1 — Activity Frequency")
    print("=" * 55)
    if feature_override:
        print(f"  Feature override: {feature_override}")

    print("\n[1/2] Estimating models from HHTS...")
    models = estimate_frequency_models(tours, persons,
                                       feature_override=feature_override,
                                       landuse=landuse)

    if save_path:
        save_models(models, save_path)

    print("\n[2/2] Predicting synthetic population...")
    synth_pred = predict_activity_frequency(synth, models,
                                            sample=sample, seed=seed,
                                            landuse=landuse)

    if '_nonmand_split' in models and '_nonmand' in models and bool(models['_nonmand']):
        print("  Splitting n_nonmand_out → escort/shopping/social/othmaint/othdiscr/eat (multinomial)...")
        synth_pred = predict_nonmand_split(synth_pred, models['_nonmand_split'],
                                           rng=np.random.default_rng(seed + 2))

    print("\n[3/3] Calibrating to HHTS aggregate rates...")
    rng_cal = np.random.default_rng(seed + 1)
    synth_pred = calibrate_activity_rates(synth_pred, persons, tours, rng=rng_cal)

    # Re-sync the combined count after calibration adjusts the individual columns
    if 'n_nonmand_out' in synth_pred.columns:
        _dc = [f'n_{p}_out' for p in NONMAND_PURPOSES if f'n_{p}_out' in synth_pred.columns]
        if _dc:
            synth_pred['n_nonmand_out'] = synth_pred[_dc].sum(axis=1)

    print("\nStage 1 complete. Pass synth_pred to run_stage2().")
    return synth_pred, models

# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------
# Usage:
#   python stage1_activity_frequency.py                    # resume: load saved_data/synth from SYNTH_DIR (all zones)
#   python stage1_activity_frequency.py 40                 # resume: zone 40 only
#   python stage1_activity_frequency.py 40,124,200         # resume: specific zones
#   python stage1_activity_frequency.py order_40.csv       # resume: single zone CSV
#   python stage1_activity_frequency.py --fresh            # fresh: load HHTS + all zones from scratch
#   python stage1_activity_frequency.py --fresh 40         # fresh: zone 40 from scratch
#
# Resume mode (default) loads synth from saved_data/zone_ids_sa2/order_N.csv
# and runs Stage 1 only. Useful for re-running with new GA features without
# waiting for upstream stages.
#
# Fresh mode runs data_connect.Synth_Connect().process_survey() + synth_attributes
# + cdap + usual_locations before Stage 1 — the full upstream chain.

if __name__ == '__main__':
    import sys
    import os
    import re
    import glob
    import logging
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('stage1_main')

    SYNTH_DIR    = 'saved_data/zone_ids_sa2'
    _OUT         = os.environ.get('PIPELINE_OUT_DIR', 'saved_data')
    CDAP_CSV     = os.path.join('saved_data', 'cdap_output.csv')
    UL_CSV       = 'saved_data/usual_locations.csv'

    args  = sys.argv[1:]
    fresh = '--fresh' in args
    args  = [a for a in args if a != '--fresh']
    zone_arg = args[0] if args else None

    # ── helper: load and enrich synth ────────────────────────────────────
    def _load_synth(zone_arg, landuse):
        from synth_attributes import enrich_synth_population

        def _enrich(df, otaz=None):
            if otaz is not None:
                df['OTAZ'] = otaz
            elif 'ORDER' in df.columns:
                df['OTAZ'] = df['ORDER'].astype(int)
            return enrich_synth_population(df, landuse=landuse)

        if zone_arg is None:
            # all zones
            files = sorted(glob.glob(os.path.join(SYNTH_DIR, 'order_*.csv')))
            if not files:
                raise FileNotFoundError(f"No order_*.csv found in {SYNTH_DIR}")
            frames = []
            for fp in files:
                m = re.search(r'order_(\d+)', fp)
                df = pd.read_csv(fp)
                frames.append(_enrich(df, int(m.group(1)) if m else None))
            return pd.concat(frames, ignore_index=True)

        if os.path.isfile(zone_arg):
            df = pd.read_csv(zone_arg)
            m = re.search(r'order_(\d+)', zone_arg)
            return _enrich(df, int(m.group(1)) if m else None)

        if re.fullmatch(r'[\d, ]+', zone_arg):
            zones = [int(z.strip()) for z in zone_arg.split(',')]
            frames = []
            for z in zones:
                fp = os.path.join(SYNTH_DIR, f'order_{z}.csv')
                if not os.path.isfile(fp):
                    log.warning(f"Zone file not found: {fp} — skipping")
                    continue
                frames.append(_enrich(pd.read_csv(fp), z))
            if not frames:
                raise FileNotFoundError(f"None of the requested zones found in {SYNTH_DIR}")
            return pd.concat(frames, ignore_index=True)

        fp = os.path.join(SYNTH_DIR, zone_arg)
        df = pd.read_csv(fp)
        m = re.search(r'order_(\d+)', zone_arg)
        return _enrich(df, int(m.group(1)) if m else None)

    # ── load HHTS survey ─────────────────────────────────────────────────
    log.info("Loading HHTS survey data...")
    from data_connect import Synth_Connect
    sc  = Synth_Connect()
    out = sc.process_survey(save=False)
    log.info(f"  HHTS loaded: {len(out['persons'])} persons, {len(out['tours'])} tours")

    if fresh:
        # ── full upstream chain ──────────────────────────────────────────
        from compute_accessibility import run_accessibility
        log.info("Running accessibility...")
        landuse = run_accessibility(
            sc.get_land(),
            skims='downcasted_matrices_base.omx',
            save_path='saved_data/accessibility.csv',
        )
        log.info("Loading and enriching synthetic population...")
        synth = _load_synth(zone_arg, landuse)
        log.info(f"  {len(synth):,} persons loaded | running CDAP...")
        from cdap import run_cdap
        synth = run_cdap(out['persons'], synth)
        synth.to_csv(CDAP_CSV, index=False)
        log.info("  Running usual locations...")
        from usual_location import run_usual_locations
        synth = run_usual_locations(
            synth, landuse,
            skims='downcasted_matrices_base.omx',
            save_path=UL_CSV,
        )
    else:
        # ── resume: attach cached accessibility (if available) to landuse ─
        ACC_CSV = 'saved_data/accessibility.csv'
        if os.path.isfile(ACC_CSV):
            from compute_accessibility import attach_to_landuse
            log.info(f"Resume mode: attaching cached accessibility from {ACC_CSV}")
            acc_df = pd.read_csv(ACC_CSV).set_index('zone_id')
            landuse = attach_to_landuse(sc.get_land(), acc_df)
        else:
            log.warning(f"No cached accessibility found at {ACC_CSV} — "
                        f"Stage 1 will run without accessibility features")
            landuse = sc.get_land()

        # ── resume: load CDAP/UL output if available ─────────────────────
        if os.path.isfile(UL_CSV):
            log.info(f"Resume mode: loading {UL_CSV}")
            synth = pd.read_csv(UL_CSV)
        elif os.path.isfile(CDAP_CSV):
            log.info(f"Resume mode: loading {CDAP_CSV}")
            synth = pd.read_csv(CDAP_CSV)
        else:
            log.warning("No saved CDAP/UL output found — loading raw synth (no CDAP filter applied)")
            synth = _load_synth(zone_arg, landuse)

    # ── CDAP traveller filter ────────────────────────────────────────────
    if 'CDAP' in synth.columns:
        travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
        log.info(f"  CDAP filter: {len(travellers):,}/{len(synth):,} travellers")
    else:
        log.warning("  'CDAP' column not found — using full synth population")
        travellers = synth.copy()

    # ── run Stage 1 ──────────────────────────────────────────────────────
    os.makedirs(_OUT, exist_ok=True)
    synth_s1, models = run_stage1(out['tours'], out['persons'], travellers, landuse=landuse)
    synth_s1.to_csv(os.path.join(_OUT, 'stage1_output.csv'), index=False)
    log.info(f"Stage 1 output saved → {_OUT}/stage1_output.csv")
    log.info(f"Pass {_OUT}/stage1_output.csv to stage2_day_duration.py")
