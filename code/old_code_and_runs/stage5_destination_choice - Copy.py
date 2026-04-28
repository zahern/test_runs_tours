"""
stage5_destination_choice.py
=============================
Stage 5 of the activity-based model pipeline.

Predicts destination zone (DTAZ) for each activity using a size-weighted
MNL destination choice model estimated from HHTS, using your actual
OD skim and landuse column names.

OD Skim columns used
---------------------
    DIST              — straight-line / road distance (km or m)
    DISTWALK          — walk distance
    DISTBIKE          — bike distance
    SOVTOLL_TIME__AM  — AM peak auto travel time
    SOVTOLL_TIME__PM  — PM peak auto travel time
    CRASH             — crash risk index (used as safety deterrent)

Landuse columns used (per purpose)
-----------------------------------
    work       : FPSEMPN, MWTEMPN, OTHEMP, HEREMPN  (employment types)
    edu        : HSENROLL, Education, childcare, education (land use)
    social     : Entertainment, EATOUT, recreation_ground, sport, park
    escort     : TOTHH, AGE0519
    shopping   : RETEMPN, Shopping, retail, commercial
    othmaint   : OTHERMAINTENANCE, Healthcare, civic, community
    othdiscr   : OTHERDISCRFacilities, TOTPOP, residential
    eat        : EATOUT, RETEMPN, retail

Accessibility = log-sum of all employment + households in zone.

Usage
-----
    from stage5_destination_choice import run_stage5

    sc       = Synth_Connect()
    out      = sc.process_survey(save=False)
    synth_s4 = pd.read_csv('saved_data/stage4_output.csv')
    skims    = pd.read_csv('OD_skims.csv')
    landuse  = sc.get_land()

    scheduled_dest, models = run_stage5(
        out["trips"], out["persons"], synth_s4, skims, landuse
    )
    scheduled_dest.to_csv('saved_data/stage5_output.csv', index=False)
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIGURATION — skim and landuse column mappings
# ---------------------------------------------------------------------------

PURPOSES = [
    'work', 'edu', 'social', 'escort',
    'shopping', 'othmaint', 'othdiscr', 'eat',
]

# ── Skim columns ──────────────────────────────────────────────────────────
# Each entry: (standardised_name, candidate_column_names, sign_in_utility)
# sign: -1 = deterrent (time/distance/crash), +1 = attractor
SKIM_SPECS = [
    ('tt_am',    ['SOVTOLL_TIME__AM', 'SOVTIME_AM', 'auto_time_am', 'tt_am'],  -1),
    ('tt_pm',    ['SOVTOLL_TIME__PM', 'SOVTIME_PM', 'auto_time_pm', 'tt_pm'],  -1),
    ('dist',     ['DIST', 'distance_km', 'dist_km', 'distance'],               -1),
    ('distwalk', ['DISTWALK', 'walk_dist', 'dist_walk'],                       -1),
    ('distbike', ['DISTBIKE', 'bike_dist', 'dist_bike'],                       -1),
    ('crash',    ['CRASH', 'crash_index', 'crash_risk'],                       -1),
]

# ── Landuse size variables per purpose ───────────────────────────────────
# Each entry is a list of columns to SUM for that purpose's size variable.
# Missing columns are silently ignored (filled with 0).
PURPOSE_SIZE_COLS = {
    'work':     ['FPSEMPN', 'MWTEMPN', 'OTHEMP', 'HEREMPN', 'AGREMPN',
                 'commercial', 'industrial', 'government', 'governmental'],
    'edu':      ['HSENROLL', 'Education', 'childcare', 'education',
                 'Primary_total_priv', 'Secondary_total_priv',
                 'Primary', 'Secondary', 'AGE0519'],
    'social':   ['Entertainment', 'EATOUT', 'recreation_ground', 'sport',
                 'park', 'village_green', 'community', 'civic'],
    'escort':   ['TOTHH', 'AGE0519', 'TOTPOP', 'childcare',
                 'Primary', 'Secondary'],
    'shopping': ['RETEMPN', 'Shopping', 'retail', 'commercial', 'EATOUT'],
    'othmaint': ['OTHERMAINTENANCE', 'Healthcare', 'civic', 'community',
                 'civic_admin', 'Public_Services', 'TOTHH'],
    'othdiscr': ['OTHERDISCRFacilities', 'TOTPOP', 'residential',
                 'entertainment', 'sport', 'park', 'OTHEMP'],
    'eat':      ['EATOUT', 'RETEMPN', 'retail', 'commercial'],
}

# ── Accessibility columns (log-sum of broad attractors) ──────────────────
ACCESSIBILITY_COLS = [
    'FPSEMPN', 'MWTEMPN', 'OTHEMP', 'HEREMPN', 'RETEMPN', 'AGREMPN',
    'TOTHH', 'TOTPOP', 'HSENROLL',
]

# Zone column candidates in landuse table
ZONE_COL_CANDIDATES = [
    'ORDER', 'DTAZ', 'SA2_MAIN16', 'zone', 'zone_id', 'TAZ', 'SA2'
]

N_ALTS       = 25   # sampled alternatives per observation
PERSON_FEATURES = ['sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING']


# ---------------------------------------------------------------------------
# 0.  SKIM + LANDUSE PREPARATION
# ---------------------------------------------------------------------------

def prepare_skims(skims) -> dict:
    """
    Standardise OD skim table.

    Accepts:
        - flat OTAZ/DTAZ DataFrame  (original behaviour)
        - path string ending in .omx (auto-converted via read_skims)
        - open omx.File object       (auto-converted)

    Returns
    -------
    dict with keys:
        'df'     : standardised DataFrame
        'lookup' : dict (OTAZ, DTAZ) -> row dict
        'cols'   : list of skim feature names present
    """
    # ── OMX path ──────────────────────────────────────────────────────
    if isinstance(skims, str) and skims.lower().endswith('.omx'):
        from read_skims import read_omx_skims
        skims = read_omx_skims(skims)
    # ── open omx.File object ──────────────────────────────────────────
    elif not isinstance(skims, pd.DataFrame):
        try:
            import openmatrix as _omx_mod
            if isinstance(skims, _omx_mod.File):
                from read_skims import read_omx_skims
                path = skims.filename
                skims.close()
                skims = read_omx_skims(path)
        except ImportError:
            pass
    # ── final type check ──────────────────────────────────────────────
    if not isinstance(skims, pd.DataFrame):
        raise TypeError(
            f"skims must be a DataFrame or .omx path/file. Got: {type(skims)}"
        )
    df = skims.copy()

    # normalise origin/destination column names
    for old, new in [('origin', 'OTAZ'), ('destination', 'DTAZ'),
                     ('orig',   'OTAZ'), ('dest',        'DTAZ'),
                     ('O',      'OTAZ'), ('D',           'DTAZ')]:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if 'OTAZ' not in df.columns or 'DTAZ' not in df.columns:
        raise ValueError(
            "OD skim table must have OTAZ and DTAZ columns (or "
            "origin/destination, orig/dest, O/D)."
        )

    present_cols = []
    for std_name, candidates, _ in SKIM_SPECS:
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            df[std_name] = pd.to_numeric(df[found], errors='coerce').fillna(0)
            present_cols.append(std_name)
            logger.info(f"  skim '{std_name}' ← '{found}'")
        else:
            df[std_name] = 0.0
            logger.warning(f"  skim '{std_name}' not found — set to 0")

    # average of AM/PM for a generic travel time
    df['tt_avg']   = (df['tt_am'] + df['tt_pm']) / 2.0
    df['log_dist'] = np.log1p(df['dist'])

    all_skim_cols = [s[0] for s in SKIM_SPECS] + ['tt_avg', 'log_dist']
    keep = ['OTAZ', 'DTAZ'] + all_skim_cols
    df   = df[keep].drop_duplicates(subset=['OTAZ', 'DTAZ'])

    df['OTAZ'] = df['OTAZ'].astype(int)
    df['DTAZ'] = df['DTAZ'].astype(int)

    lookup = df.set_index(['OTAZ', 'DTAZ'])[all_skim_cols].to_dict('index')

    logger.info(f"Skims ready: {len(df)} OD pairs, "
                f"features: {present_cols}")
    return {
        'df':     df,
        'lookup': lookup,
        'cols':   present_cols + ['tt_avg', 'log_dist'],
    }


def prepare_landuse(landuse: pd.DataFrame) -> dict:
    """
    Standardise landuse table.

    Returns
    -------
    dict with keys:
        'df'         : standardised DataFrame
        'zone_col'   : name of the zone ID column used
        'size_cols'  : dict {purpose: computed size column name}
    """
    df = landuse.copy()

    # find zone column
    zone_col = next(
        (c for c in ZONE_COL_CANDIDATES if c in df.columns), None
    )
    if zone_col is None:
        raise ValueError(
            f"Landuse table has no recognisable zone column. "
            f"Expected one of: {ZONE_COL_CANDIDATES}"
        )
    df = df.rename(columns={zone_col: 'DTAZ'}) if zone_col != 'DTAZ' else df
    df['DTAZ'] = pd.to_numeric(df['DTAZ'], errors='coerce')
    df = df.dropna(subset=['DTAZ'])
    df['DTAZ'] = df['DTAZ'].astype(int)
    df = df.drop_duplicates(subset=['DTAZ'])

    # convert all numeric columns
    for col in df.columns:
        if col != 'DTAZ':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # build size variable per purpose (sum of available columns)
    size_cols = {}
    for purpose, cols in PURPOSE_SIZE_COLS.items():
        available = [c for c in cols if c in df.columns]
        col_name  = f'size_{purpose}'
        if available:
            df[col_name] = df[available].sum(axis=1).clip(lower=0)
        else:
            logger.warning(f"  landuse: no size columns for '{purpose}' — using 1")
            df[col_name] = 1.0
        df[col_name] = np.log1p(df[col_name])   # log-transform size
        size_cols[purpose] = col_name

    # accessibility = log-sum of broad attractors
    acc_available = [c for c in ACCESSIBILITY_COLS if c in df.columns]
    if acc_available:
        df['accessibility'] = np.log1p(df[acc_available].sum(axis=1))
    else:
        df['accessibility'] = 0.0

    # zone density proxy
    if 'sa2_area' in df.columns and df['sa2_area'].gt(0).any():
        df['log_density'] = np.log1p(
            df.get('TOTPOP', df.get('POPULATION', 0)) /
            df['sa2_area'].replace(0, np.nan).fillna(1)
        )
    else:
        df['log_density'] = 0.0

    logger.info(
        f"Landuse ready: {len(df)} zones, "
        f"size columns built for: {list(size_cols.keys())}"
    )
    return {
        'df':        df,
        'zone_col':  'DTAZ',
        'size_cols': size_cols,
    }


# ---------------------------------------------------------------------------
# 1.  BUILD FEATURE VECTOR FOR ONE OD PAIR
# ---------------------------------------------------------------------------

def _od_features(
    otaz: int,
    dtaz: int,
    purpose: str,
    skim_lookup: dict,
    lu_df: pd.DataFrame,
    lu_size_cols: dict,
    feature_names: list,
) -> np.ndarray:
    """
    Build utility feature vector for one (origin, destination, purpose).
    Returns 1-D array aligned to feature_names.
    """
    sk = skim_lookup.get((otaz, dtaz), {})

    # try to get landuse row for this DTAZ
    lu_rows = lu_df[lu_df['DTAZ'] == dtaz]
    lu = lu_rows.iloc[0].to_dict() if len(lu_rows) > 0 else {}

    feat = {}

    # skim features
    for std_name, _, _ in SKIM_SPECS:
        feat[std_name] = float(sk.get(std_name, 0.0))
    feat['tt_avg']   = (feat['tt_am'] + feat['tt_pm']) / 2.0
    feat['log_dist'] = np.log1p(feat.get('dist', 0.0))

    # landuse size for this purpose
    size_col = lu_size_cols.get(purpose, None)
    feat['log_size'] = float(lu.get(size_col, 0.0)) if size_col else 0.0

    # accessibility and density
    feat['accessibility'] = float(lu.get('accessibility', 0.0))
    feat['log_density']   = float(lu.get('log_density', 0.0))

    return np.array([feat.get(f, 0.0) for f in feature_names])


# ---------------------------------------------------------------------------
# 2.  MNL DESTINATION CHOICE MODEL
# ---------------------------------------------------------------------------

class MNLDestinationModel:
    """
    MNL destination choice with importance-sampled alternatives.

    Utility function:
        V(d) = beta_tt_avg    * tt_avg(o→d)         [auto time deterrent]
             + beta_tt_am     * tt_am(o→d)           [AM peak penalty]
             + beta_log_dist  * log_dist(o→d)        [distance deterrent]
             + beta_distwalk  * distwalk(o→d)        [walk distance]
             + beta_distbike  * distbike(o→d)        [bike distance]
             + beta_crash     * crash(o→d)           [crash deterrent]
             + beta_log_size  * log_size(d, purpose) [attraction]
             + beta_acc       * accessibility(d)     [zone attractiveness]
             + beta_density   * log_density(d)       [urban density]
    """

    FEATURE_NAMES = [
        'tt_avg', 'tt_am', 'log_dist',
        'distwalk', 'distbike', 'crash',
        'log_size', 'accessibility', 'log_density',
    ]

    # coefficient bounds: (lo, hi) per feature
    BOUNDS = [
        (-3.0, 0.0),   # tt_avg      — must be negative
        (-2.0, 0.0),   # tt_am       — must be negative
        (-3.0, 0.0),   # log_dist    — must be negative
        (-2.0, 0.0),   # distwalk    — must be negative
        (-2.0, 0.0),   # distbike    — must be negative
        (-2.0, 0.0),   # crash       — must be negative
        ( 0.0, 3.0),   # log_size    — must be positive
        (-2.0, 3.0),   # accessibility
        (-1.0, 2.0),   # log_density
    ]

    # sensible starting values
    INIT = np.array([-0.05, -0.02, -0.20,
                     -0.01, -0.01, -0.10,
                      0.50,  0.30,  0.10])

    def __init__(self):
        self.coef_      = None
        self.converged_ = False
        self.purpose_   = None
        self.n_obs_     = 0

    def _V(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def _softmax_rows(self, Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def _neg_ll(self, params, X_chosen, X_alts, n_alts):
        N = len(X_chosen)
        V_ch  = X_chosen @ params                           # (N,)
        V_alt = (X_alts @ params).reshape(N, n_alts)       # (N, n_alts)
        V_all = np.column_stack([V_ch, V_alt])
        V_all -= V_all.max(axis=1, keepdims=True)
        denom = np.exp(V_all).sum(axis=1)
        return -np.sum(V_ch - np.log(denom))

    def _grad(self, params, X_chosen, X_alts, n_alts):
        N = len(X_chosen)
        V_ch  = X_chosen @ params
        V_alt = (X_alts @ params).reshape(N, n_alts)
        V_all = np.column_stack([V_ch, V_alt])
        V_all -= V_all.max(axis=1, keepdims=True)
        E     = np.exp(V_all)
        P     = E / E.sum(axis=1, keepdims=True)           # (N, n_alts+1)
        X_all = np.concatenate(
            [X_chosen[:, np.newaxis, :],
             X_alts.reshape(N, n_alts, -1)], axis=1
        )
        exp_x = (P[:, :, np.newaxis] * X_all).sum(axis=1) # (N, K)
        return -(X_chosen - exp_x).sum(axis=0)

    def fit(
        self,
        obs: pd.DataFrame,
        purpose: str,
        skim_lookup: dict,
        lu_df: pd.DataFrame,
        lu_size_cols: dict,
        seed: int = 42,
    ):
        from scipy.optimize import minimize

        rng = np.random.default_rng(seed)
        self.purpose_ = purpose
        self.n_obs_   = len(obs)

        feats  = self.FEATURE_NAMES
        zones  = lu_df['DTAZ'].values

        # attraction weights for importance sampling
        size_col = lu_size_cols.get(purpose)
        raw_size = lu_df[size_col].values if size_col else np.ones(len(zones))
        weights  = np.exp(raw_size.astype(float)) + 1.0
        weights /= weights.sum()

        # chosen features
        X_chosen = np.array([
            _od_features(int(r['OTAZ']), int(r['DTAZ']),
                         purpose, skim_lookup, lu_df, lu_size_cols, feats)
            for _, r in obs.iterrows()
        ])

        N       = len(obs)
        alt_idx = rng.choice(len(zones), size=(N, N_ALTS),
                             p=weights, replace=True)
        otaz_arr = obs['OTAZ'].values

        X_alts = np.array([
            [_od_features(int(otaz_arr[i]), int(zones[j]),
                          purpose, skim_lookup, lu_df, lu_size_cols, feats)
             for j in alt_idx[i]]
            for i in range(N)
        ]).reshape(N * N_ALTS, len(feats))

        res = minimize(
            self._neg_ll,
            self.INIT.copy(),
            args=(X_chosen, X_alts, N_ALTS),
            jac=self._grad,
            method='L-BFGS-B',
            bounds=self.BOUNDS,
            options={'maxiter': 400, 'ftol': 1e-9},
        )

        self.coef_      = res.x
        self.converged_ = res.success
        return self

    def predict_dtaz(
        self,
        otaz_arr: np.ndarray,
        purpose: str,
        skim_lookup: dict,
        lu_df: pd.DataFrame,
        lu_size_cols: dict,
        rng: np.random.Generator,
        n_sample: int = 30,
    ) -> np.ndarray:
        """Sample destination zone for each activity via MNL."""
        feats  = self.FEATURE_NAMES
        zones  = lu_df['DTAZ'].values

        size_col = lu_size_cols.get(purpose)
        raw_size = lu_df[size_col].values if size_col else np.ones(len(zones))
        weights  = np.exp(raw_size.astype(float).clip(-10, 10)) + 1.0
        weights /= weights.sum()

        N      = len(otaz_arr)
        chosen = np.zeros(N, dtype=int)
        pool   = rng.choice(len(zones), size=(N, n_sample),
                            p=weights, replace=True)

        for i in range(N):
            otaz = int(otaz_arr[i])
            X = np.array([
                _od_features(otaz, int(zones[j]),
                             purpose, skim_lookup, lu_df, lu_size_cols, feats)
                for j in pool[i]
            ])
            V = X @ self.coef_
            V -= V.max()
            p  = np.exp(V)
            p /= p.sum()
            chosen[i] = int(zones[pool[i][rng.choice(n_sample, p=p)]])

        return chosen


# ---------------------------------------------------------------------------
# 3.  PREPARE ESTIMATION DATA
# ---------------------------------------------------------------------------

def prepare_destination_data(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    skim_info: dict,
    lu_info: dict,
) -> pd.DataFrame:
    """
    Build trip-level estimation dataset with chosen OTAZ, DTAZ,
    purpose, and skim/landuse attributes for the chosen pair.
    """
    df = trips.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # ── deduplicate columns first — prevents df['purpose'] returning DataFrame
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]

    # ── resolve purpose column ────────────────────────────────────────
    # Prefer DESTPURP (individual trip destination purpose).
    # If both DESTPURP and 'purpose' exist, drop 'purpose' first to avoid
    # the duplicate-column bug that makes df['purpose'] return a DataFrame.
    if 'DESTPURP' in df.columns:
        if 'purpose' in df.columns:
            df = df.drop(columns=['purpose'])
        df = df.rename(columns={'DESTPURP': 'purpose'})
    elif 'purpose' not in df.columns:
        purp_col = next(
            (c for c in ['OVERALL_PURPOSE', 'ORIGPURP'] if c in df.columns), None
        )
        if purp_col is None:
            raise ValueError("No purpose column in trips table.")
        df = df.rename(columns={purp_col: 'purpose'})

    # ── map raw strings → codes ───────────────────────────────────────
    raw_to_code = {
        'Work (my workplace)': 'work', 'Work (other work reason)': 'work',
        'Direct Work Commute': 'work', 'Work Related': 'work',
        'Education': 'edu', 'Childcare or Kindergarten': 'edu',
        'Childcare or kindergarten': 'edu',
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
        'At Home': 'home', 'Go home': 'home', 'My home': 'home',
        'Eat/drink': 'eat',
    }
    # check on a scalar-safe Series — avoids the DataFrame.unique() error
    purpose_series = df['purpose']
    if isinstance(purpose_series, pd.DataFrame):
        # still duplicated somehow — take last column
        purpose_series = purpose_series.iloc[:, -1]
        df['purpose'] = purpose_series
    unique_vals = set(purpose_series.dropna().unique())
    if not unique_vals.issubset(set(PURPOSES) | {'home'}):
        df['purpose'] = purpose_series.map(raw_to_code).fillna(purpose_series)

    # ── filter to modelled purposes, valid zones ───────────────────────
    df = df[df['purpose'].isin(PURPOSES)].copy()
    df = df[df['OTAZ'].notna() & df['DTAZ'].notna()].copy()
    df['OTAZ'] = pd.to_numeric(df['OTAZ'], errors='coerce').fillna(0).astype(int)
    df['DTAZ'] = pd.to_numeric(df['DTAZ'], errors='coerce').fillna(0).astype(int)
    df = df[(df['OTAZ'] > 0) & (df['DTAZ'] > 0)].copy()

    # ── STARTIME ─────────────────────────────────────────────────────────
    st_col = next((c for c in ['STARTIME', 'depart', 'start_time']
                   if c in df.columns), None)
    df['start_hr'] = (
        pd.to_numeric(df[st_col], errors='coerce').fillna(480) / 60
        if st_col else 8.0
    )

    logger.info(f"Destination estimation: {len(df)} trips across "
                f"{df['person_id'].nunique()} persons")
    return df


# ---------------------------------------------------------------------------
# 4.  ESTIMATE
# ---------------------------------------------------------------------------

def estimate_destination_models(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    skim_info: dict,
    lu_info: dict,
    min_sample: int = 50,
    seed: int = 42,
) -> dict:
    """
    Fit one MNLDestinationModel per purpose.

    Returns dict {purpose: MNLDestinationModel}
    """
    data = prepare_destination_data(trips, persons, skim_info, lu_info)

    skim_lookup  = skim_info['lookup']
    lu_df        = lu_info['df']
    lu_size_cols = lu_info['size_cols']

    models = {}

    for purpose in PURPOSES:
        subset = data[data['purpose'] == purpose].copy()

        if len(subset) < min_sample:
            logger.warning(f"  dest/{purpose}: {len(subset)} obs — skipping")
            continue

        m = MNLDestinationModel()
        m.fit(subset, purpose, skim_lookup, lu_df, lu_size_cols, seed=seed)

        models[purpose] = m
        status = 'OK' if m.converged_ else 'WARN'

        coef_summary = dict(zip(MNLDestinationModel.FEATURE_NAMES,
                                np.round(m.coef_, 3)))
        logger.info(
            f"  [{status}] dest/{purpose}: n={len(subset)}\n"
            f"    coef: {coef_summary}"
        )

    print(
        f"Stage 5: fitted {len(models)} destination models "
        f"({', '.join(models.keys())})"
    )
    return models


# ---------------------------------------------------------------------------
# 5.  PREDICT
# ---------------------------------------------------------------------------

def predict_destinations(
    scheduled: pd.DataFrame,
    models: dict,
    skim_info: dict,
    lu_info: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign DTAZ to every activity in the Stage 4 output.

    For each person, activities are chained in departure-time order:
      - Activity 0  : origin = person home zone (OTAZ from persons table)
      - Activity 1+ : origin = DTAZ of preceding activity

    Parameters
    ----------
    scheduled  : Stage 4 output — one row per activity per person.
                 Must contain: person_id, OTAZ, STARTIME, purpose.
    models     : {purpose: DestinationModel} from estimate_destination_models()
    skim_info  : from prepare_skims()
    lu_info    : from prepare_landuse()
    seed       : random seed for Monte-Carlo sampling

    Returns
    -------
    DataFrame with DTAZ assigned and OD skim attributes attached.
    """
    rng = np.random.default_rng(seed)
    df  = scheduled.copy()

    # ── column normalisation ──────────────────────────────────────────
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]

    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # ── OTAZ ─────────────────────────────────────────────────────────
    if 'OTAZ' not in df.columns:
        raise ValueError("predict_destinations: OTAZ column missing. "
                         "Merge persons['OTAZ'] onto scheduled before calling.")
    df['OTAZ'] = pd.to_numeric(df['OTAZ'], errors='coerce').fillna(0).astype(int)

    # ── STARTIME fallback ─────────────────────────────────────────────
    st_col = next((c for c in ['STARTIME', 'depart', 'start_time']
                   if c in df.columns), None)
    if st_col is None:
        df['STARTIME'] = 480
    elif st_col != 'STARTIME':
        df['STARTIME'] = df[st_col]

    # ── purpose column ────────────────────────────────────────────────
    if 'purpose' not in df.columns and 'DESTPURP' in df.columns:
        df = df.rename(columns={'DESTPURP': 'purpose'})

    # ── sort ──────────────────────────────────────────────────────────
    df = df.sort_values(['person_id', 'STARTIME']).reset_index(drop=True)

    # ── home zone lookup ──────────────────────────────────────────────
    # Use max OTAZ per person — avoids 0 from the first-activity row
    # which may not have OTAZ populated yet.
    home_otaz_raw = df.groupby('person_id')['OTAZ'].max().to_dict()
    home_otaz     = {pid: (int(z) if z > 0 else 1)
                     for pid, z in home_otaz_raw.items()}
    n_zero = sum(1 for z in home_otaz_raw.values() if z == 0)
    if n_zero > 0:
        logger.warning(
            f"predict_destinations: {n_zero} persons have OTAZ=0 — "
            f"defaulting to zone 1. Ensure persons['OTAZ'] is merged onto "
            f"synth_s4 before calling run_stage5()."
        )

    skim_lookup  = skim_info['lookup']
    lu_df        = lu_info['df']
    lu_size_cols = lu_info['size_cols']

    # ── initialise output columns ─────────────────────────────────────
    df['DTAZ']        = 0
    df['_otaz_chain'] = 0   # actual origin used for each activity

    # ── chain trip destinations person by person ──────────────────────
    all_idx       = []
    all_dtaz      = []
    all_otaz_used = []

    for pid, grp in df.groupby('person_id', sort=False):
        grp = grp.sort_values('STARTIME')

        # first activity origin is always home zone
        home_zone = home_otaz.get(pid, 1)
        prev_dtaz = home_zone

        for i, (idx, row) in enumerate(grp.iterrows()):
            purpose   = str(row.get('purpose', '')).strip()
            trip_otaz = home_zone if i == 0 else prev_dtaz

            # ensure valid zone
            if trip_otaz <= 0:
                trip_otaz = home_zone if home_zone > 0 else 1

            # predict destination
            if purpose in models:
                try:
                    dtaz_arr = models[purpose].predict_dtaz(
                        otaz_arr     = np.array([trip_otaz]),
                        purpose      = purpose,
                        skim_lookup  = skim_lookup,
                        lu_df        = lu_df,
                        lu_size_cols = lu_size_cols,
                        rng          = rng,
                    )
                    dtaz = int(dtaz_arr[0])
                except Exception as e:
                    logger.warning(f"predict_dtaz failed pid={pid} purpose={purpose}: {e}")
                    dtaz = trip_otaz
            else:
                # no model for this purpose — stay in origin zone
                dtaz = trip_otaz

            all_idx.append(idx)
            all_dtaz.append(dtaz)
            all_otaz_used.append(trip_otaz)
            prev_dtaz = dtaz

    df.loc[all_idx, 'DTAZ']        = all_dtaz
    df.loc[all_idx, '_otaz_chain'] = all_otaz_used

    # update OTAZ to reflect the chained origin (not the raw home-zone value)
    df['OTAZ'] = df['_otaz_chain'].where(df['_otaz_chain'] > 0, df['OTAZ'])
    df = df.drop(columns=['_otaz_chain'])

    df['OTAZ'] = df['OTAZ'].astype(int)
    df['DTAZ'] = df['DTAZ'].astype(int)

    # ── attach OD skim attributes for Stage 6 ────────────────────────
    # Always drop existing skim cols first — prevents _x/_y duplicates
    # when log_dist (or others) was already added by prepare_destination_data.
    SKIM_COLS = ['tt_avg', 'tt_am', 'tt_pm', 'log_dist', 'dist',
                 'distwalk', 'distbike', 'crash']
    skim_df   = skim_info['df']

    existing_skim = [c for c in SKIM_COLS if c in df.columns]
    if existing_skim:
        df = df.drop(columns=existing_skim)

    avail_skim = [c for c in SKIM_COLS if c in skim_df.columns]
    df = df.merge(
        skim_df[['OTAZ', 'DTAZ'] + avail_skim].drop_duplicates(['OTAZ', 'DTAZ']),
        on=['OTAZ', 'DTAZ'],
        how='left'
    )
    for col in SKIM_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0.0   # column absent from skim file — use zero

    # ── summary ───────────────────────────────────────────────────────
    n_valid_dtaz = (df['DTAZ'] > 0).sum()
    n_zero_dtaz  = (df['DTAZ'] == 0).sum()
    logger.info(
        f"predict_destinations: {len(df)} activities | "
        f"{n_valid_dtaz} valid DTAZ | {n_zero_dtaz} zero DTAZ | "
        f"{df['DTAZ'].nunique()} unique zones"
    )
    print(f"Stage 5 predictions: {len(df)} activities")
    for p in PURPOSES:
        n = int((df['purpose'] == p).sum())
        if n > 0:
            n_dest = int((df.loc[df['purpose'] == p, 'DTAZ'] > 0).sum())
            print(f"  {p}: {n} activities, {n_dest} with valid DTAZ")
    print(f"  Unique DTAZ: {df['DTAZ'].nunique()}")
    if n_zero_dtaz > 0:
        logger.warning(f"  {n_zero_dtaz} activities have DTAZ=0 "
                       f"(no model or prediction failed)")
    return df


# ---------------------------------------------------------------------------
# 6.  SAVE / LOAD
# ---------------------------------------------------------------------------

def save_models(models: dict, path: str = 'saved_data/stage5_models.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Stage 5 models saved → {path}")


def load_models(path: str = 'saved_data/stage5_models.pkl') -> dict:
    with open(path, 'rb') as f:
        models = pickle.load(f)
    print(f"Stage 5 models loaded ← {path}")
    return models


# ---------------------------------------------------------------------------
# 7.  FULL PIPELINE WRAPPER
# ---------------------------------------------------------------------------

def run_stage5(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    synth_s4: pd.DataFrame,
    skims: pd.DataFrame,
    landuse: pd.DataFrame,
    seed: int = 42,
    save_path: str = 'saved_data/stage5_models.pkl',
) -> tuple:
    """
    Estimate destination choice models on HHTS, predict for Stage 4 output.

    Parameters
    ----------
    trips, persons : from process_survey()
    synth_s4       : Stage 4 output (OTAZ, STARTIME, purpose per activity)
    skims          : OD skim table — needs OTAZ, DTAZ, plus any of:
                     SOVTOLL_TIME__AM, SOVTOLL_TIME__PM, DIST,
                     DISTWALK, DISTBIKE, CRASH
    landuse        : zone landuse — needs ORDER or SA2_MAIN16 as zone key,
                     plus any of the PURPOSE_SIZE_COLS columns
    seed           : random seed
    save_path      : where to pickle fitted models

    Returns
    -------
    scheduled_dest : Stage 4 activities with DTAZ + skim attributes
    models         : fitted models dict (pass to Stage 6)

    Example
    -------
        sc       = Synth_Connect()
        out      = sc.process_survey(save=False)
        synth_s4 = pd.read_csv('saved_data/stage4_output.csv')
        skims    = pd.read_csv('OD_skims.csv')
        landuse  = sc.get_land()

        scheduled_dest, models = run_stage5(
            out["trips"], out["persons"], synth_s4, skims, landuse
        )
        scheduled_dest.to_csv('saved_data/stage5_output.csv', index=False)
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    print("=" * 55)
    print("STAGE 5 — Destination Choice")
    print("=" * 55)

    print("\n[0/2] Preparing skims and landuse...")
    skim_info = prepare_skims(skims)
    lu_info   = prepare_landuse(landuse)

    print("\n[1/2] Estimating destination models from HHTS trips...")
    models = estimate_destination_models(
        trips, persons, skim_info, lu_info, seed=seed
    )

    if save_path:
        save_models(models, save_path)

    print("\n[2/2] Predicting destinations for synthetic population...")

    # ── ensure OTAZ is present on every activity row ──────────────
    # Stage 4 output has one row per activity but OTAZ (home zone)
    # lives on the persons table. Merge it on if missing.
    synth_s4 = synth_s4.copy()
    if 'OTAZ' not in synth_s4.columns or synth_s4['OTAZ'].isna().all() or (synth_s4['OTAZ'] == 0).all():
        # try to get OTAZ from persons
        pers_otaz = persons.copy()
        if 'person_id' not in pers_otaz.columns and 'PERSID' in pers_otaz.columns:
            pers_otaz = pers_otaz.rename(columns={'PERSID': 'person_id'})
        if 'OTAZ' in pers_otaz.columns and 'person_id' in pers_otaz.columns:
            otaz_map = pers_otaz.set_index('person_id')['OTAZ'].to_dict()
            pid_col = 'person_id' if 'person_id' in synth_s4.columns else 'PERSID'
            synth_s4['OTAZ'] = synth_s4[pid_col].map(otaz_map).fillna(0).astype(int)
            print(f"  Merged OTAZ from persons: "
                  f"{(synth_s4['OTAZ'] > 0).sum()}/{len(synth_s4)} activities have valid OTAZ")
        else:
            raise ValueError(
                "OTAZ not found in synth_s4 or persons. "
                "Ensure persons table has OTAZ (home zone ORDER) column."
            )
    else:
        synth_s4['OTAZ'] = pd.to_numeric(synth_s4['OTAZ'], errors='coerce').fillna(0).astype(int)
        # backfill any zeros from persons
        zero_mask = synth_s4['OTAZ'] == 0
        if zero_mask.any():
            pers_otaz = persons.copy()
            if 'person_id' not in pers_otaz.columns and 'PERSID' in pers_otaz.columns:
                pers_otaz = pers_otaz.rename(columns={'PERSID': 'person_id'})
            if 'OTAZ' in pers_otaz.columns:
                otaz_map = pers_otaz.set_index('person_id')['OTAZ'].to_dict()
                pid_col = 'person_id' if 'person_id' in synth_s4.columns else 'PERSID'
                synth_s4.loc[zero_mask, 'OTAZ'] = (
                    synth_s4.loc[zero_mask, pid_col].map(otaz_map).fillna(0).astype(int)
                )

    scheduled_dest = predict_destinations(
        synth_s4, models, skim_info, lu_info, seed=seed
    )

    print("\nStage 5 complete. Pass scheduled_dest to run_stage6().")
    return scheduled_dest, models