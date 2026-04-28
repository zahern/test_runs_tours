"""
tour_duration_model.py
======================
Budget-constrained lognormal duration model for HHTS tour data.
Features are activity counts (n_eat_out, n_work_out, etc.) rather than
purpose dummies — matching the stop_frequency output of your pipeline.

Usage — fit on HHTS
--------------------
    from tour_duration_model import run_tour_duration_model

    out = Synth_Connect().process_survey(save=False)

    # aggregate tours to person-day level
    person_day = aggregate_to_person_day(out["tours"])

    tours_pred, res, feats, dummy_info = run_tour_duration_model(
        out["tours"], person_day
    )

Usage — apply to synthetic population
--------------------------------------
    synth = pd.read_csv('saved_data/ativities_time_synthetic_population.csv')
    synth_pred = predict_synthetic(res.x, synth, feats, dummy_info)
    # synth_pred has predicted_duration_min per row (one row = one tour)
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from jax import grad


# ---------------------------------------------------------------------------
# ACTIVITY COUNT COLUMNS — edit here if your column names differ
# ---------------------------------------------------------------------------

ACTIVITY_COLS = [
    'n_eat_out',
    'n_edu_out',
    'n_social_out',
    'n_escort_out',
    'n_shopping_out',
    'n_othdiscr_out',
    'n_othmaint_out',
    'n_work_out',
]


# ---------------------------------------------------------------------------
# 0. AGGREGATE TOURS TO PERSON-DAY
# ---------------------------------------------------------------------------

def aggregate_to_person_day(tours: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the tours table to one row per person with:
      - start_of_day / end_of_day (hours)
      - n_*_out activity counts summed across tours
      - n_tours
      - B (total tour duration in hours — the daily budget)

    Parameters
    ----------
    tours : DataFrame from process_survey()["tours"]
            Must have: person_id, start_time, end_time, duration,
                       n_eat_out ... n_work_out, tour_id

    Returns
    -------
    One-row-per-person DataFrame.
    """
    df = tours.copy()

    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # convert times to hours if still in minutes
    df['start_time_hour'] = df['start_time'] / 60
    df['end_time_hour']   = df['end_time']   / 60
    df['duration_hrs']    = (df['duration']  / 60).clip(lower=0.05)

    agg_dict = {
        'start_of_day':  ('start_time_hour', 'min'),
        'end_of_day':    ('end_time_hour',   'max'),
        'n_tours':       ('tour_id',         'max'),
        'B':             ('duration_hrs',    'sum'),   # daily budget
    }
    for col in ACTIVITY_COLS:
        if col in df.columns:
            agg_dict[col] = (col, 'sum')
        else:
            df[col] = 0
            agg_dict[col] = (col, 'sum')

    person_day = (
        df.groupby('person_id')
        .agg(**agg_dict)
        .reset_index()
    )

    return person_day


# ---------------------------------------------------------------------------
# 1. FEATURE PREPARATION  (tour-level — one row per tour)
# ---------------------------------------------------------------------------

def prepare_tour_features(
    tours: pd.DataFrame,
    person_day: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the feature matrix at tour level.

    Each tour gets:
      - its own start_hr and log_dist
      - the person-level activity counts (n_eat_out, n_work_out, …)
      - start_of_day / end_of_day / n_tours from person_day

    Parameters
    ----------
    tours      : from process_survey()["tours"]
    person_day : from aggregate_to_person_day(tours)

    Returns
    -------
    DataFrame with feature columns and df.attrs['feature_cols'].
    """
    df = tours.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    # ---- duration target ----------------------------------------------
    df['duration_hrs'] = (df['duration'] / 60).clip(lower=0.05)

    # ---- merge person-day summary onto each tour row ------------------
    merge_cols = ['person_id', 'B', 'start_of_day', 'end_of_day', 'n_tours']
    merge_cols += [c for c in ACTIVITY_COLS if c in person_day.columns]
    df = df.merge(person_day[merge_cols], on='person_id', how='left')

    # ---- drop single-tour persons ------------------------------------
    tour_counts = df.groupby('person_id')['tour_id'].transform('count')
    n_before = len(df)
    df = df[tour_counts > 1].copy().reset_index(drop=True)
    print(f"  Dropped {n_before - len(df)} rows (single-tour persons).")

    # ---- tour-level features -----------------------------------------
    df['start_hr']  = df['start_time'] / 60
    df['log_dist']  = np.log1p(np.abs(
        df['tour_destination'].fillna(0) - df['tour_origin'].fillna(0)
    ))
    df['day_span']  = (df['end_of_day'] - df['start_of_day']).clip(lower=0)
    df['n_tours_f'] = df['n_tours'].fillna(1).clip(lower=1)

    # ---- activity count features (already on df from merge) ----------
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(float)

    # ---- record feature columns --------------------------------------
    base_feats    = ['start_hr', 'log_dist', 'day_span', 'n_tours_f']
    activity_feats = [c for c in ACTIVITY_COLS if c in df.columns]
    df.attrs['feature_cols'] = base_feats + activity_feats

    return df


# ---------------------------------------------------------------------------
# 2. ARRAY EXTRACTION
# ---------------------------------------------------------------------------

def prepare_model_arrays(tours_feat: pd.DataFrame):
    """
    Convert prepared DataFrame to JAX-safe numpy arrays.

    person_id is always encoded to a dense 0-based int32 index,
    handling string / composite keys like '100582/1000'.

    Returns
    -------
    X, y, ids (int32), budgets, unique_persons
    """
    feature_cols = tours_feat.attrs.get('feature_cols')
    if feature_cols is None:
        raise ValueError("Run prepare_tour_features first.")

    X = tours_feat[feature_cols].fillna(0).values.astype(np.float64)
    y = tours_feat['duration_hrs'].values.astype(np.float64)

    raw_ids = tours_feat['person_id'].values
    unique_persons, ids = np.unique(raw_ids, return_inverse=True)
    ids = ids.astype(np.int32)

    budget_map = (
        tours_feat
        .groupby('person_id')['B']
        .first()
        .reindex(unique_persons)
        .values
        .astype(np.float64)
    )

    return X, y, ids, budget_map, unique_persons


# ---------------------------------------------------------------------------
# 3. MODEL
# ---------------------------------------------------------------------------

def _lognormal_ll(y, eta, sigma):
    return (
        - jnp.log(y)
        - jnp.log(sigma)
        - 0.5 * ((jnp.log(y) - eta) / sigma) ** 2
    )


def ll_with_budget_penalty(params, X, y, ids, budgets, lambda_penalty=50.0):
    K     = X.shape[1]
    beta  = params[:K]
    sigma = jax.nn.softplus(params[K])

    eta      = X @ beta
    total_ll = jnp.sum(_lognormal_ll(y, eta, sigma))

    pred_unscaled = jnp.exp(eta + 0.5 * sigma ** 2)
    person_totals = jax.ops.segment_sum(
        pred_unscaled, ids, num_segments=len(budgets)
    )
    penalty = jnp.sum((person_totals - budgets) ** 2)

    return -(total_ll - lambda_penalty * penalty)


# ---------------------------------------------------------------------------
# 4. ESTIMATION
# ---------------------------------------------------------------------------

def estimate_duration_model(
    X, y, ids, budgets,
    lambda_penalty=50.0,
    init_params=None,
):
    K = X.shape[1]
    if init_params is None:
        init_params = np.zeros(K + 1)

    X_j   = jnp.array(X)
    y_j   = jnp.array(y)
    ids_j = jnp.array(ids)
    b_j   = jnp.array(budgets)

    obj = lambda p: ll_with_budget_penalty(p, X_j, y_j, ids_j, b_j, lambda_penalty)

    result = minimize(obj, init_params, jac=grad(obj), method='L-BFGS-B')

    if not result.success:
        print(f"Warning: did not converge — {result.message}")
    else:
        print(f"Converged. Final loss: {result.fun:.4f}")

    return result


# ---------------------------------------------------------------------------
# 5. PREDICTION — HHTS tours (budget-rescaled)
# ---------------------------------------------------------------------------

def predict_tour_durations(params, tours_feat, X, ids):
    """
    Predict and budget-rescale durations for HHTS tours.
    Returns tours_feat copy with predicted_duration_hrs / _min columns.
    """
    K     = X.shape[1]
    beta  = params[:K]
    sigma = float(jax.nn.softplus(jnp.array(params[K])))

    unscaled  = np.exp(X @ beta + 0.5 * sigma ** 2)
    df        = tours_feat.copy().reset_index(drop=True)
    predicted = np.zeros(len(df))

    for int_id in np.unique(ids):
        mask  = ids == int_id
        B     = df.loc[mask, 'B'].iloc[0]
        total = unscaled[mask].sum()
        scale = B / total if total > 0 else 1.0
        predicted[mask] = unscaled[mask] * scale

    df['predicted_duration_hrs'] = predicted
    df['predicted_duration_min'] = predicted * 60
    return df


# ---------------------------------------------------------------------------
# 6. PREDICTION — SYNTHETIC POPULATION
# ---------------------------------------------------------------------------

def build_synth_features(
    synth: pd.DataFrame,
    feature_cols: list,
) -> np.ndarray:
    """
    Build the feature matrix X for a synthetic population.

    The synthetic population only needs to have the columns that appear
    in feature_cols — missing ones are filled with 0.

    Typical synthetic population columns required:
        n_eat_out, n_edu_out, n_social_out, n_escort_out,
        n_shopping_out, n_othdiscr_out, n_othmaint_out, n_work_out,
        start_hr (or start_of_day in minutes → divide by 60),
        OTAZ / tour_origin (for log_dist — set to 0 if unknown)

    Parameters
    ----------
    synth        : synthetic population DataFrame
    feature_cols : list returned by run_tour_duration_model

    Returns
    -------
    X_synth : (N, K) float64 array aligned to feature_cols
    """
    df = synth.copy()

    # ---- derive any missing features from available columns -----------

    # start_hr: accept start_of_day (minutes) or STARTIME
    if 'start_hr' not in df.columns:
        if 'start_of_day' in df.columns:
            df['start_hr'] = df['start_of_day'] / 60
        elif 'STARTIME' in df.columns:
            df['start_hr'] = df['STARTIME'] / 60
        else:
            df['start_hr'] = 0.0

    # log_dist: if origin known but no explicit distance
    if 'log_dist' not in df.columns:
        orig = df.get('tour_origin', df.get('OTAZ', pd.Series(0, index=df.index)))
        dest = df.get('tour_destination', df.get('DTAZ', pd.Series(0, index=df.index)))
        df['log_dist'] = np.log1p(np.abs(dest.fillna(0) - orig.fillna(0)))

    # day_span
    if 'day_span' not in df.columns:
        if 'start_of_day' in df.columns and 'end_of_day' in df.columns:
            df['day_span'] = (df['end_of_day'] - df['start_of_day']).clip(lower=0)
        else:
            df['day_span'] = 0.0

    # n_tours_f
    if 'n_tours_f' not in df.columns:
        col = 'n_tours' if 'n_tours' in df.columns else None
        df['n_tours_f'] = df[col].fillna(1).clip(lower=1) if col else 1.0

    # activity counts — fill missing with 0
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0).astype(float)

    # ---- assemble X aligned to training feature order ----------------
    X_synth = np.column_stack([
        df[col].fillna(0).values.astype(np.float64)
        for col in feature_cols
    ])

    return X_synth


def predict_synthetic(
    params: np.ndarray,
    synth: pd.DataFrame,
    feature_cols: list,
    budget_col: str = None,
) -> pd.DataFrame:
    """
    Apply fitted model to a synthetic population.

    Because the synthetic population has no observed budget B,
    prediction is unconstrained — we return the raw lognormal mean
    (not rescaled). If you have a budget estimate (e.g. from a
    separate daily budget model), pass its column name as budget_col
    to apply rescaling.

    Parameters
    ----------
    params       : fitted params from estimate_duration_model (.x)
    synth        : synthetic population DataFrame
    feature_cols : list returned by run_tour_duration_model
    budget_col   : optional column in synth containing person budget (hrs)

    Returns
    -------
    synth copy with columns:
        predicted_duration_hrs, predicted_duration_min
    """
    X_synth = build_synth_features(synth, feature_cols)

    K     = len(feature_cols)
    beta  = params[:K]
    sigma = float(jax.nn.softplus(jnp.array(params[K])))

    eta      = X_synth @ beta
    unscaled = np.exp(eta + 0.5 * sigma ** 2)

    df = synth.copy().reset_index(drop=True)

    if budget_col is not None and budget_col in df.columns:
        # rescale per person if budget known
        pid_col = 'person_id' if 'person_id' in df.columns else (
                  'PERSID'    if 'PERSID'    in df.columns else None)

        if pid_col:
            predicted = np.zeros(len(df))
            for pid, grp in df.groupby(pid_col):
                idx   = grp.index
                B     = grp[budget_col].iloc[0]
                total = unscaled[idx].sum()
                scale = B / total if total > 0 else 1.0
                predicted[idx] = unscaled[idx] * scale
        else:
            predicted = unscaled   # can't group without person id
    else:
        predicted = unscaled        # raw lognormal mean

    df['predicted_duration_hrs'] = predicted
    df['predicted_duration_min'] = predicted * 60

    return df


# ---------------------------------------------------------------------------
# 7. FULL PIPELINE
# ---------------------------------------------------------------------------

def run_tour_duration_model(
    tours: pd.DataFrame,
    person_day: pd.DataFrame = None,
    lambda_penalty: float = 50.0,
) -> tuple:
    """
    Full fit pipeline: aggregate → prepare → fit → predict.

    Parameters
    ----------
    tours          : from process_survey()["tours"]
    person_day     : from aggregate_to_person_day(tours), or None to
                     compute it automatically
    lambda_penalty : budget constraint weight (try 10–100)

    Returns
    -------
    tours_pred   : tours DataFrame with predicted_duration_min
    result       : scipy OptimizeResult (.x = fitted params)
    feature_cols : list of feature column names (pass to predict_synthetic)

    Example — fit
    -------------
        out        = Synth_Connect().process_survey(save=False)
        person_day = aggregate_to_person_day(out["tours"])
        tours_pred, res, feats = run_tour_duration_model(
            out["tours"], person_day
        )

    Example — predict synthetic
    ---------------------------
        synth      = pd.read_csv('saved_data/ativities_time_synthetic_population.csv')
        synth_pred = predict_synthetic(res.x, synth, feats)
        print(synth_pred[['person_id','predicted_duration_min']].head())
    """
    if person_day is None:
        print("Aggregating tours to person-day...")
        person_day = aggregate_to_person_day(tours)

    print("Preparing features...")
    tours_feat   = prepare_tour_features(tours, person_day)
    feature_cols = tours_feat.attrs['feature_cols']
    print(f"  {len(tours_feat)} tours, {tours_feat['person_id'].nunique()} persons")
    print(f"  Features: {feature_cols}")

    print("Extracting arrays...")
    X, y, ids, budgets, unique_persons = prepare_model_arrays(tours_feat)
    print(f"  {len(unique_persons)} persons with 2+ tours")

    print("Fitting model...")
    result = estimate_duration_model(X, y, ids, budgets, lambda_penalty)

    print("Predicting HHTS tours...")
    tours_pred = predict_tour_durations(result.x, tours_feat, X, ids)

    print("Done.")
    return tours_pred, result, feature_cols