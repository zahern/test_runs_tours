"""
logsum_accessibility.py
=======================
Compute per-home-zone logsum accessibility from fitted Stage-5 mode models.

Used to unify the accessibility definition across the pipeline: the same
fitted mode-logsum inclusive value enters Stage 1 (activity generation) and
Stage 5 (mode-destination coupling via c_logsum), replacing the simplified
log1p(attractor*decay) proxy for the behavioural chain.

Following ActivitySim _accumulate_accessibility:

    acc[o] = log( sum_z [ exp( LS_mode(o, z) ) ] )

where LS_mode(o,z) is the mode-choice logsum for a representative tour from
home zone o to destination zone z, computed with the purpose-specific Stage-5
mode model.

Columns follow the *_acc convention so attach_person_accessibility and
prepare_frequency_data pick them up automatically.
"""

import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

# ── Environment gates ───────────────────────────────────────────────────────
ENV_UNIFIED_LOGSUM_ACC  = 'UNIFIED_LOGSUM_ACC'      # '1' → enable reorder + logsum
ENV_EQUILIBRIUM_LOOP    = 'EQUILIBRIUM_LOOP'         # '1' → iterate Phase 2
ENV_EQUILIBRIUM_N_ITER  = 'EQUILIBRIUM_N_ITER'       # int, default 3

LOGSUM_COL_PREFIX = 'logsum_'

# Stage-1 purpose → preferred Stage-5 segment (fallback chain if missing)
_PURPOSE_SEGMENT_MAP = {
    'work':     ('work',),
    'edu':      ('edu',),
    'shopping': ('shopping', 'othdiscr', 'nonmand'),
    'social':   ('social', 'nonmand'),
    'escort':   ('escort', 'nonmand'),
    'othmaint': ('othmaint', 'nonmand'),
    'othdiscr': ('othdiscr', 'nonmand'),
    'eat':      ('nonmand', 'othdiscr'),
    'nonmand':  ('nonmand',),
}


def is_unified_enabled() -> bool:
    return os.environ.get(ENV_UNIFIED_LOGSUM_ACC, '0').strip().lower() in (
        '1', 'true', 'on', 'yes')


def is_equilibrium_enabled() -> bool:
    return os.environ.get(ENV_EQUILIBRIUM_LOOP, '0').strip().lower() in (
        '1', 'true', 'on', 'yes')


def equilibrium_n_iter() -> int:
    try:
        n = int(os.environ.get(ENV_EQUILIBRIUM_N_ITER, '3'))
        return max(n, 1)
    except ValueError:
        return 3


def _segment_for_purpose(purpose: str, available_segments: set) -> str:
    candidates = _PURPOSE_SEGMENT_MAP.get(purpose, (purpose,))
    for seg in candidates:
        if seg in available_segments:
            return seg
    return next(iter(available_segments)) if available_segments else purpose


def compute_logsum_accessibility(
    stage5_models: dict,
    landuse: pd.DataFrame,
    skims,
    *,
    persons: pd.DataFrame = None,
    zone_ids_override: list = None,
) -> pd.DataFrame:
    """
    Per-home-zone logsum accessibility from fitted Stage-5 mode models.

    For each purpose with a fitted mode model, computes:

        acc[o] = log( Σ_z exp( mode_logsum(o, z) ) )

    Parameters
    ----------
    stage5_models : {segment: {mode_model, dest_model, ...}} from run_stage5()
    landuse       : zone landuse DataFrame (ORDER column for zone ids)
    skims         : flat OD skim DataFrame, or OMX handle
    persons       : optional person DataFrame for mode-tree attributes
    zone_ids_override : explicit zone ORDER list (None → from landuse)

    Returns
    -------
    acc_df : DataFrame indexed by zone ORDER, columns like
             'logsum_work_acc', 'logsum_edu_acc', 'logsum_shopping_acc'
    """
    from stage5_destination_choice import (
        _logsums, _mode_tree, _skim_flat, _default_zone_universe,
    )
    import xarray as xr

    zone_col = 'ORDER' if 'ORDER' in landuse.columns else (
        landuse.index.name or landuse.columns[0])
    # 371-only: no `!= 330` drop — ORDER 330 is a real zone in the 371 system.
    if zone_ids_override is not None:
        zone_ids = sorted(int(z) for z in zone_ids_override if int(z) > 0)
    else:
        zone_ids = sorted(
            int(z) for z in landuse[zone_col].astype(int).unique()
            if int(z) > 0)
    if len(zone_ids) < 10:
        zone_ids = _default_zone_universe()
    n_zones = len(zone_ids)

    model_segments = set(stage5_models.keys())
    logger.info(
        f"logsum_accessibility: {n_zones} zones, "
        f"segments={sorted(model_segments)}")

    # ── flat skim ────────────────────────────────────────────────────────
    if isinstance(skims, pd.DataFrame):
        skim_df = skims
    else:
        try:
            skim_df = _skim_flat(skims) if skims is not None else pd.DataFrame()
        except Exception:
            logger.warning("logsum_accessibility: could not flatten skims")
            skim_df = pd.DataFrame()

    # ── representative tours ─────────────────────────────────────────────
    if persons is not None and len(persons) > 0:
        template = persons.iloc[0:1].copy()
        if 'OTAZ' in template.columns:
            template = template.drop(columns=['OTAZ'])
        if 'DTAZ' in template.columns:
            template = template.drop(columns=['DTAZ'])
    else:
        template = pd.DataFrame({
            'person_id': 1, 'incp': 8, 'vehrd': 1, 'nprd': 2,
            'sexp': 1, 'WORKSTAT': 1, 'STUDYING': 0, 'ptype': 'FullTimeWorker',
            'AGEGROUP': 30, 'CARLICENCE': 1, 'n_workers_hh': 1,
            'wfh': 0, 'is_female': 0, 'has_vehicle': 1, 'hh_child': 0,
            'ptype_FullTimeWorker': 1, 'mandatory_role': 1,
        }, index=[0])

    rows = []
    for oz in zone_ids:
        r = template.copy()
        r['OTAZ'] = oz
        r['DTAZ'] = oz
        r['ptour_id'] = f"ls_oz_{oz}"
        r['weight'] = 1
        r['tour_mode'] = 1
        rows.append(r)
    tours = pd.concat(rows, ignore_index=True)
    persons_frame = tours[template.columns.intersection(tours.columns)].copy()
    if 'person_id' not in persons_frame.columns:
        persons_frame['person_id'] = persons_frame.index + 1

    # ── build mode tree ──────────────────────────────────────────────────
    dt_mode = None
    try:
        dt_mode = _mode_tree(tours, persons_frame, landuse, skim_df)
    except Exception as _e1:
        logger.debug(f"  _mode_tree failed with skim_df: {_e1}")
        try:
            dt_mode = _mode_tree(tours, persons_frame, landuse, skims)
        except Exception as _e2:
            logger.warning(
                f"logsum_accessibility: could not build mode tree ({_e2})")
            return pd.DataFrame({'ORDER': zone_ids}).set_index('ORDER')

    # ── compute per-purpose logsum columns ───────────────────────────────
    purposes_to_compute = ['work', 'edu', 'shopping', 'social', 'escort',
                           'othdiscr', 'othmaint', 'eat', 'nonmand']
    results = {}
    for purp in purposes_to_compute:
        seg = _segment_for_purpose(purp, model_segments)
        seg_data = stage5_models.get(seg)
        if seg_data is None:
            continue
        mode_mdl = seg_data.get('mode_model')
        if mode_mdl is None:
            continue
        try:
            ls_da = _logsums(
                mode_mdl, dt_mode, zone_ids_override=zone_ids, verbose=False)
            ls_np = ls_da.values.astype(np.float64)
            if ls_np.ndim != 2 or ls_np.shape[0] != n_zones:
                logger.debug(
                    f"  logsum shape mismatch for '{purp}' "
                    f"(seg='{seg}'): {ls_np.shape}, expected ({n_zones},{n_zones})")
                continue

            ls_max = ls_np.max(axis=1, keepdims=True)
            ls_exp = np.exp(ls_np - np.maximum(ls_max, 0))
            acc = np.log(np.maximum(ls_exp.sum(axis=1), 1e-12))
            acc = acc + ls_max.ravel()
            acc = np.nan_to_num(acc, nan=0.0, posinf=1e6, neginf=-1e6)

            col = f"{LOGSUM_COL_PREFIX}{purp}_acc"
            results[col] = acc
            logger.info(
                f"  logsum_accessibility: '{col}' "
                f"(seg='{seg}', range=[{acc.min():.1f}, {acc.max():.1f}])")
        except Exception as e:
            logger.warning(
                f"  logsum_accessibility: {purp} (seg='{seg}') failed: {e}")

    if not results:
        logger.warning("logsum_accessibility: no columns computed")
        return pd.DataFrame({'ORDER': zone_ids}).set_index('ORDER')

    acc_df = pd.DataFrame(results, index=zone_ids)
    acc_df.index.name = 'ORDER'

    # Add ORDER column for merge compatibility
    acc_df_out = acc_df.reset_index()
    return acc_df_out


def attach_logsum_to_landuse(
    landuse: pd.DataFrame,
    acc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge logsum accessibility columns onto landuse.
    Follows compute_accessibility.attach_to_landuse convention.

    Parameters
    ----------
    landuse : raw landuse DataFrame
    acc_df  : output of compute_logsum_accessibility (ORDER + *_acc cols)

    Returns
    -------
    landuse copy with logsum_*_acc columns merged in
    """
    lu = landuse.copy()
    zone_col = 'ORDER' if 'ORDER' in lu.columns else None
    join_col = 'ORDER' if 'ORDER' in acc_df.columns else acc_df.index.name

    if zone_col and zone_col in lu.columns:
        lu = lu.merge(acc_df, on=zone_col, how='left')
    else:
        lu = lu.merge(
            acc_df, left_index=True, right_on=join_col, how='left')
        if join_col in lu.columns:
            lu = lu.drop(columns=[join_col])

    for col in (c for c in acc_df.columns if c != 'ORDER'):
        if col in lu.columns and lu[col].isna().any():
            median_val = lu[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            lu[col] = lu[col].fillna(median_val)
    return lu
