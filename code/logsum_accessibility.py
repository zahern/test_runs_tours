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

Diagnostics: each purpose's surface is checked for coverage/flatness
(LOGSUM_ACC_REQUIRE_VARIANCE=1 → raise) and, with verify_pt_monotonicity,
for the PT-on >= PT-off nest invariant (STAGE5_PT_MODES=1).
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
ENV_LOGSUM_ACC_REQUIRE_VARIANCE = 'LOGSUM_ACC_REQUIRE_VARIANCE'  # '1' → raise on flat columns

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
    # Default ON: the model-consistent fitted mode-logsum inclusive value is the
    # accessibility measure that feeds Stage 1 and the Stage-5 c_logsum coupling,
    # replacing the fixed-decay gravity proxy (whose hand-set decay coefficients
    # and collinear employment inputs destabilised c_logsum identification).
    # Set UNIFIED_LOGSUM_ACC=0 to fall back to the gravity proxy.
    return os.environ.get(ENV_UNIFIED_LOGSUM_ACC, '1').strip().lower() in (
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


def _check_pt_monotonicity(acc_pt, mode_mdl, dt_mode, zone_ids, *, skims=None,
                           purp: str, seg: str):
    """Paired PT-on/off consistency check for one purpose's logsum surface.

    Re-evaluates the SAME fitted mode model with the PT alternatives dropped
    (runtime toggle of stage5_destination_choice.STAGE5_PT_MODES): adding
    alternatives to a nest can only raise the inclusive value, so
    acc_pt >= acc_off is a hard invariant.  Catches PT terms not reaching
    the nest (propagation broken) and accidental double counting.

    Returns (acc_pt, diag).  Raises under LOGSUM_ACC_REQUIRE_VARIANCE=1 on
    breach; warns otherwise.
    """
    try:
        import stage5_destination_choice as _sdc
        from stage5_destination_choice import _logsums as _ls_fn
    except Exception as _ie:
        logger.warning("logsum_accessibility: PT monotonicity check skipped "
                       "for '%s' (%s)", purp, _ie)
        return acc_pt, {'status': 'skipped', 'reason': str(_ie)}
    if not getattr(_sdc, 'STAGE5_PT_MODES', False):
        logger.info("logsum_accessibility: PT monotonicity check skipped "
                    "for '%s' (STAGE5_PT_MODES=0)", purp)
        return acc_pt, {'status': 'skipped', 'reason': 'STAGE5_PT_MODES=0'}

    _sdc.STAGE5_PT_MODES = False
    try:
        ls_off = _ls_fn(mode_mdl, dt_mode, skims=skims,
                        zone_ids_override=zone_ids, verbose=False)
    finally:
        _sdc.STAGE5_PT_MODES = True

    _off = np.asarray(ls_off.values, dtype=np.float64)
    _pt_arr = np.asarray(acc_pt, dtype=np.float64)
    if _off.ndim != 2 or _off.shape[0] != _pt_arr.size:
        _msg = (f"logsum_accessibility: PT monotonicity check for '{purp}' "
                f"shape mismatch (PT {_pt_arr.shape} vs off {_off.shape})")
        logger.warning(_msg)
        return acc_pt, {'status': 'skipped', 'reason': _msg}
    _om = _off.max(axis=1, keepdims=True)
    _oe = np.exp(_off - np.maximum(_om, 0))
    acc_off = np.log(np.maximum(_oe.sum(axis=1), 1e-12)) + _om.ravel()
    acc_off = np.nan_to_num(acc_off, nan=0.0, posinf=1e6, neginf=-1e6)

    delta = _pt_arr - acc_off
    _tol = 1e-6 * np.maximum(1.0, np.abs(acc_off))
    _n_viol = int((delta < -_tol).sum())
    diag = {
        'status': 'ok',
        'n_zones': int(delta.size),
        'min_delta': float(delta.min()),
        'median_delta': float(np.median(delta)),
        'max_delta': float(delta.max()),
        'frac_pt_adds': float((delta > 1e-9).mean()),
        'n_violations': _n_viol,
    }
    if _n_viol:
        _msg = (
            f"logsum_accessibility: PT monotonicity BREACH for '{purp}' "
            f"(seg='{seg}'): {_n_viol}/{delta.size} zones have PT-on "
            f"logsum below PT-off (min delta={delta.min():.3g}) - the PT "
            "alternatives are not reaching the inclusive value correctly")
        if os.environ.get(ENV_LOGSUM_ACC_REQUIRE_VARIANCE,
                          '0').strip().lower() in ('1', 'true', 'on', 'yes'):
            raise RuntimeError(_msg)
        logger.error(_msg)
    else:
        logger.info(
            f"logsum_accessibility: PT monotonicity OK for '{purp}' "
            f"(seg='{seg}'): delta [{delta.min():.4g}, {delta.max():.4g}], "
            f"median={float(np.median(delta)):.4g}, adds in "
            f"{diag['frac_pt_adds']:.0%} of zones")
    return acc_pt, diag


def compute_logsum_accessibility(
    stage5_models: dict,
    landuse: pd.DataFrame,
    skims,
    *,
    persons: pd.DataFrame = None,
    zone_ids_override: list = None,
    verify_pt_monotonicity: bool = False,
) -> pd.DataFrame:
    """
    Per-home-zone logsum accessibility from fitted Stage-5 mode models.

    For each purpose with a fitted mode model, computes:

        acc[o] = log( Σ_z exp( mode_logsum(o, z) ) )

    Parameters
    ----------
    stage5_models : {segment: {mode_model, dest_model, ...}} from run_stage5()
    landuse       : zone landuse DataFrame (ORDER column for zone ids)
    skims         : flat OD skim DataFrame, OMX handle, or OMX path.  The
                    _logsums numpy fallback (default path) reads matrices BY
                    NAME through a handle — a flat DataFrame alone silently
                    degrades every time/impedance term to a constant, so a
                    handle is resolved/reopened here.
    persons       : optional person DataFrame for mode-tree attributes
    zone_ids_override : explicit zone ORDER list (None → from landuse)
    verify_pt_monotonicity : paired PT-on/off nest-invariant check per
                    purpose (STAGE5_PT_MODES=1; doubles logsum cost — used
                    by verify_logsum_accessibility.py)

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

    # ── OMX handle for _logsums matrix reads ─────────────────────────────
    # _logsums' numpy fallback (the default path since STAGE5_LARCH_LOGSUM=0)
    # reads OMX matrices BY NAME via the skims handle; a flat DataFrame has
    # no matrix access, so every time/impedance term degrades to a constant
    # and the surface flattens.  String paths are opened (and closed at
    # every return), handles pass through, DataFrames pass through unchanged.
    _ls_skims, _ls_opened = skims, False
    try:
        if isinstance(skims, str) and skims and os.path.isfile(skims):
            import openmatrix as _omx
            _ls_skims = _omx.open_file(skims, 'r')
            _ls_opened = True
    except Exception as _sk_e:
        logger.warning(
            "logsum_accessibility: could not open OMX handle for logsums "
            "(%s) - time/impedance terms may degrade to constants", _sk_e)
        _ls_skims, _ls_opened = skims, False

    def _close_ls_skims():
        if _ls_opened:
            try:
                _ls_skims.close()
            except Exception:
                pass

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
            _close_ls_skims()
            return pd.DataFrame({'ORDER': zone_ids}).set_index('ORDER')

    # ── compute per-purpose logsum columns ───────────────────────────────
    purposes_to_compute = ['work', 'edu', 'shopping', 'social', 'escort',
                           'othdiscr', 'othmaint', 'eat', 'nonmand']
    results = {}
    _acc_diagnostics: dict = {}
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
                mode_mdl, dt_mode, skims=_ls_skims,
                zone_ids_override=zone_ids, verbose=False)
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
            _n_bad = int((~np.isfinite(acc)).sum())
            acc = np.nan_to_num(acc, nan=0.0, posinf=1e6, neginf=-1e6)

            # ── Propagation diagnostics ─────────────────────────────────
            # Downstream (Stage 1 frequency + Stage 5 c_logsum) needs a
            # zone-varying finite surface: a flat column silently fits a
            # dead c_logsum.  Report coverage/variance per purpose; fail
            # loudly under LOGSUM_ACC_REQUIRE_VARIANCE=1.
            _std = float(np.std(acc))
            _flat_frac = (float((np.abs(acc - acc[0]) < 1e-9).mean())
                          if acc.size else 1.0)
            _diag = _acc_diagnostics.setdefault(purp, {})
            _diag.update({
                'seg': seg, 'n_zones': int(acc.size),
                'n_nonfinite': _n_bad, 'std': _std,
                'flat_frac': _flat_frac,
                'range': (float(acc.min()), float(acc.max())),
            })
            if _n_bad:
                logger.warning(
                    f"  logsum_accessibility: '{LOGSUM_COL_PREFIX}{purp}_acc' "
                    f"had {_n_bad}/{acc.size} non-finite zone values "
                    "(nan_to_num-filled) - check skims/model for that segment")
            if _flat_frac > 0.5 or _std < 1e-9:
                _msg = (
                    f"logsum_accessibility: '{LOGSUM_COL_PREFIX}{purp}_acc' "
                    f"(seg='{seg}') is FLAT across zones "
                    f"(std={_std:.3g}, flat_frac={_flat_frac:.0%}) - the "
                    "upstream logsum surface is degenerate and c_logsum / "
                    "Stage-1 frequency cannot identify zone-variation")
                if os.environ.get(ENV_LOGSUM_ACC_REQUIRE_VARIANCE,
                                  '0').strip().lower() in ('1', 'true', 'on', 'yes'):
                    raise RuntimeError(_msg)
                logger.warning(_msg)

            if verify_pt_monotonicity:
                acc, _pt_diag = _check_pt_monotonicity(
                    acc, mode_mdl, dt_mode, zone_ids, skims=_ls_skims,
                    purp=purp, seg=seg)
                _diag['pt_monotonicity'] = _pt_diag

            col = f"{LOGSUM_COL_PREFIX}{purp}_acc"
            results[col] = acc
            logger.info(
                f"  logsum_accessibility: '{col}' "
                f"(seg='{seg}', range=[{acc.min():.1f}, {acc.max():.1f}], "
                f"std={_std:.3g}, nonfinite={_n_bad})")
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(
                f"  logsum_accessibility: {purp} (seg='{seg}') failed: {e}")

    if results:
        _flat_n = sum(1 for d in _acc_diagnostics.values()
                      if d['flat_frac'] > 0.5 or d['std'] < 1e-9)
        _pt_viol_n = sum(1 for d in _acc_diagnostics.values()
                         if d.get('pt_monotonicity', {}).get('n_violations'))
        logger.info(
            f"logsum_accessibility: propagation summary — {len(results)} "
            f"purposes computed, {_flat_n} flat, "
            f"{_pt_viol_n} PT-monotonicity breaches, "
            f"{sum(d['n_nonfinite'] for d in _acc_diagnostics.values())} "
            "non-finite zone values total")
    _close_ls_skims()
    if not results:
        logger.warning("logsum_accessibility: no columns computed")
        return pd.DataFrame({'ORDER': zone_ids}).set_index('ORDER')

    acc_df = pd.DataFrame(results, index=zone_ids)
    acc_df.index.name = 'ORDER'

    # Add ORDER column for merge compatibility
    acc_df_out = acc_df.reset_index()
    # Diagnostics ride along on .attrs for verification scripts (pandas
    # propagates attrs through reset_index); regular callers ignore them.
    acc_df_out.attrs['diagnostics'] = _acc_diagnostics
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
