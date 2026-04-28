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
from typing import Optional

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

# Max count cap per purpose (counts above this are rare and break the model)
MAX_COUNT = {
    'work':      3,   # rare to have >3 work trips in a day
    'edu':       2,
    'social':    5,   # can have multiple social visits
    'escort':    6,   # school drop-off + pick-up etc
    'shopping':  5,
    'othmaint':  5,
    'othdiscr':  5,
    'eat':       4,   # breakfast + lunch + dinner possible
}

# Person type definitions — maps ptype label → filter on persons DataFrame
# Uses the assign_ptype() function from your existing code if available,
# otherwise falls back to these column-based rules
# Unified ptype labels — MUST match cdap._derive_ptype() output so synth pop
# predictions align with models estimated from HHTS.
#   WORKSTAT codes (from data_connect.work_map): 1=PartTime, 2=FullTime, 5=Retired
#   STUDYING codes (from data_connect.study_map): 1=primary, 2=secondary,
#                                                  3=tertiaryPartTime, 4=tertiaryFullTime
PTYPE_COLS = {
    'FullTimeWorker':    lambda p: (p['WORKSTAT'] == 2) & ~p['STUDYING'].isin([3, 4]),
    'PartTimeWorker':    lambda p: (p['WORKSTAT'] == 1) & ~p['STUDYING'].isin([3, 4]),
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

    return person_counts


# ---------------------------------------------------------------------------
# 2.  PREPARE ESTIMATION DATA
# ---------------------------------------------------------------------------

def prepare_frequency_data(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join activity counts with person attributes ready for model estimation.

    Parameters
    ----------
    tours   : from process_survey()["tours"]
    persons : from process_survey()["persons"]  (already processed)

    Returns
    -------
    One-row-per-person DataFrame with ptype, n_{purpose}_out, and
    all PERSON_FEATURES columns.
    """
    counts = build_activity_counts(tours)

    pers = persons.copy()
    if 'person_id' not in pers.columns and 'PERSID' in pers.columns:
        pers = pers.rename(columns={'PERSID': 'person_id'})

    pers = assign_ptype_column(pers)

    # keep only columns we need
    keep = ['person_id', 'ptype'] + [c for c in PERSON_FEATURES if c in pers.columns]
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

    def _pack(self, beta, raw_cuts):
        return np.concatenate([beta, raw_cuts])

    def _unpack(self, params, K):
        return params[:K], params[K:]

    def _cuts_from_raw(self, raw_cuts):
        """Ensure thresholds are strictly ordered via cumulative exp."""
        return np.cumsum(np.concatenate([[raw_cuts[0]],
                                          np.exp(raw_cuts[1:])]))

    def _log_likelihood(self, params, X, y, K):
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

        return -ll   # minimise negative LL

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

        res = minimize(
            self._log_likelihood,
            init,
            args=(X, y, K),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-9},
        )

        self.coef_       = res.x[:K]
        self.thresholds_ = self._cuts_from_raw(res.x[K:])
        self.converged_  = res.success
        self.n_obs_      = int(len(y))
        self.ll_         = float(-res.fun)
        # null LL: uniform probabilities
        self.ll0_        = float(len(y) * np.log(1.0 / J))
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
) -> dict:
    """
    Estimate one ordered logit per (purpose, ptype) combination.

    Parameters
    ----------
    tours, persons    : from process_survey()
    min_sample        : skip estimation if fewer than this many observations
    feature_override  : if supplied, use this list of features instead of
                        PERSON_FEATURES — passed from GA results to apply
                        GA-selected variable sets.
    """
    data = prepare_frequency_data(tours, persons)
    if feature_override is not None:
        feature_cols = [c for c in feature_override if c in data.columns]
        if not feature_cols:
            feature_cols = [c for c in PERSON_FEATURES if c in data.columns]
    else:
        feature_cols = [c for c in PERSON_FEATURES if c in data.columns]

    models = {}

    for purpose in PURPOSES:
        col = f'n_{purpose}_out'
        models[purpose] = {}

        ptypes = data['ptype'].unique()

        for ptype in ptypes:
            subset = data[data['ptype'] == ptype].copy()

            if len(subset) < min_sample:
                logger.warning(
                    f"  {purpose}/{ptype}: only {len(subset)} obs — skipping"
                )
                continue

            y = subset[col].values.astype(int)

            # skip if everyone has 0 activities (not relevant for this ptype)
            if y.max() == 0:
                logger.info(f"  {purpose}/{ptype}: all zeros — skipping")
                continue

            X_raw = subset[feature_cols].fillna(0).values.astype(float)
            mu = X_raw.mean(axis=0)
            sd = X_raw.std(axis=0)
            sd[sd == 0] = 1.0
            X = (X_raw - mu) / sd

            model = OrderedLogit()
            model.fit(X, y)
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
    return models


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

    # initialise output columns
    for purpose in PURPOSES:
        df[f'n_{purpose}_out'] = 0

    # process each ptype group separately
    for ptype, grp in df.groupby('ptype'):
        idx = grp.index

        # Map new HHTS ptypes (Preschool, JointWorkStudy) to closest estimation ptype
        lookup_ptype = PTYPE_PREDICTION_MAP.get(ptype, ptype)

        for purpose in PURPOSES:
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
    from paper_model_forms import write_stage_paper_artifacts
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {path}")
    html_path = path.replace('.pkl', '_report.html')
    _write_model_html(models, html_path, 'Stage 1 — Activity Frequency',
                      notes=f'Saved: {path}')
    write_stage_paper_artifacts('stage1', os.path.dirname(path) or 'saved_data')


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
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
    print("=" * 55)
    print("STAGE 1 — Activity Frequency")
    print("=" * 55)
    if feature_override:
        print(f"  Feature override: {feature_override}")

    print("\n[1/2] Estimating models from HHTS...")
    models = estimate_frequency_models(tours, persons,
                                       feature_override=feature_override)

    if save_path:
        save_models(models, save_path)

    print("\n[2/2] Predicting synthetic population...")
    synth_pred = predict_activity_frequency(synth, models,
                                            sample=sample, seed=seed)

    print("\n[3/3] Calibrating to HHTS aggregate rates...")
    rng_cal = np.random.default_rng(seed + 1)
    synth_pred = calibrate_activity_rates(synth_pred, persons, tours, rng=rng_cal)

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
        # ── resume: load CDAP/UL output if available ─────────────────────
        if os.path.isfile(UL_CSV):
            log.info(f"Resume mode: loading {UL_CSV}")
            synth = pd.read_csv(UL_CSV)
        elif os.path.isfile(CDAP_CSV):
            log.info(f"Resume mode: loading {CDAP_CSV}")
            synth = pd.read_csv(CDAP_CSV)
        else:
            log.warning("No saved CDAP/UL output found — loading raw synth (no CDAP filter applied)")
            landuse = sc.get_land()
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
    synth_s1, models = run_stage1(out['tours'], out['persons'], travellers)
    synth_s1.to_csv(os.path.join(_OUT, 'stage1_output.csv'), index=False)
    log.info(f"Stage 1 output saved → {_OUT}/stage1_output.csv")
    log.info(f"Pass {_OUT}/stage1_output.csv to stage2_day_duration.py")
