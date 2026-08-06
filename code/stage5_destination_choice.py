"""
stage5_destination_choice.py
=============================
Stage 5 - Destination Choice + Mode Choice + Full Trip Chain

What this stage does
---------------------
Stage 4 output = one row per OUT-OF-HOME ACTIVITY per person:
    person_id, purpose, STARTIME, ENDTIME, predicted_duration_min, ...

Stage 5 converts this into a proper TRIP CHAIN:

    home ──trip1──> activity1 ──trip2──> activity2 ──trip3──> home
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
import openmatrix as omx
import pandas as pd
from scipy.stats import norm

try:
    # Enable 64-bit precision in JAX *before* jax initialises. JAX defaults to
    # float32; combined with the float32 CA data this made the destination-model
    # log-likelihood flat to ~1e-7 near the start values, so jax handed SLSQP a
    # gradient that rounded below gtol=1e-6 → optimizer stopped at iteration 0
    # ("ALL free params unmoved after phase-1") and the numba fallback had to
    # redo the fit. In float64 jax computes a real gradient and estimates the
    # flat-UEC dest models directly on jax, no numba retry. Must precede any
    # jax array creation, hence set at import. Honour an explicit override via
    # JAX_ENABLE_X64=0 for anyone who deliberately wants 32-bit.
    import jax
    if os.environ.get('JAX_ENABLE_X64', '1') != '0':
        jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except Exception:
    jnp = np
    JAX_AVAILABLE = False

from larch_regularization import (elasticnet_objective, nl_identification_priors,
                                  NL_PRIOR_ALPHA)

logger = logging.getLogger(__name__)

# Larch re-emits "chosen_but_not_available" (and similar troubleshooting notes)
# on EVERY model fit. The SA/GA search does thousands of fits per segment, so
# these flood the .e log into a wall of identical warnings that looks like a
# massive failure. We already repair chosen_but_not_available at build time
# (mode_mdl.doctor(repair_ch_av="+")) and again inside _estimate, so the issue
# is handled — demote Larch's troubleshooting logger to ERROR to stop the spam.
# Set GA_LARCH_TROUBLESHOOT=1 to restore the per-fit warnings for debugging.
if os.environ.get('GA_LARCH_TROUBLESHOOT', '0') != '1':
    logging.getLogger('larch.model.troubleshooting').setLevel(logging.ERROR)

# The flat destination model can fail to compile under sharrow/numba (a 1D-column
# vs 3D-result dtype mismatch in the generated dim*_filler), which sharrow logs as
# a wall of "ERROR:sharrow: <col> = __main__<col>[...]" + "nb.TypingError in N
# functions". This is EXPECTED and HANDLED — _estimate/_build_dest_model catch it
# and fall back to the JAX importance-sampled dest estimator (a WARNING is still
# logged), so the run is fine. Demote sharrow's logger to CRITICAL to stop the
# spam that looks like a hard failure. Set STAGE5_SHARROW_VERBOSE=1 to restore it.
if os.environ.get('STAGE5_SHARROW_VERBOSE', '0') != '1':
    logging.getLogger('sharrow').setLevel(logging.CRITICAL)

PURPOSES     = ['work', 'edu', 'social', 'escort',
                'shopping', 'othmaint', 'othdiscr', 'eat']
NON_MANDATORY = ['social', 'escort', 'shopping', 'othmaint', 'othdiscr', 'eat']
MODE_MAP     = {'DA': 1, 'SR': 2, 'Walk': 3, 'Bike': 4, 'Transit': 5}
MODE_NAMES   = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
DEST_CHOICE_SAMPLE_SIZE = 0    # 0 = use ALL zones during prediction (no sampling)
# Polaris-style runtime destination choice.  The model remains a zone-level
# MNL; these settings only control the simulation choice set and feasibility.
POLARIS_DEST_CHOICE = os.environ.get('STAGE5_POLARIS_DEST_CHOICE', '1') == '1'
POLARIS_DEST_STRATA = 4
POLARIS_DEST_ALTS_PER_STRATUM = int(
    os.environ.get('STAGE5_POLARIS_ALTS_PER_STRATUM', '200')
)
POLARIS_PLANNING_CONSTRAINED = (
    os.environ.get('STAGE5_POLARIS_PLANNING', '1') == '1'
)
POLARIS_DAY_START_MIN = float(os.environ.get('STAGE5_DAY_START_MIN', '0'))
POLARIS_DAY_END_MIN = float(os.environ.get('STAGE5_DAY_END_MIN', '1440'))
MAX_MXL_OBS = 2500              # skip MXL on large segments that were causing OOM/NaN estimates
# Importance sampling for destination choice estimation.
# 0 = use ALL zones; >0 = Manski-Lerman with N alternatives per observation.
# Override via env var STAGE5_DEST_SAMPLE_SIZE (e.g. PBS -v STAGE5_DEST_SAMPLE_SIZE=20).
DEST_EST_SAMPLE_SIZE = int(os.environ.get('STAGE5_DEST_SAMPLE_SIZE', '20'))
# Mixed logit is optional; default off because some Larch/sharrow builds
# produce rewrite/Inf-loglike failures on large CA tables.
# GSE (Gradient Score Enhanced) mixed logit — systematic γ·g loadings on top of
# the Normal mixtures. Enabling GSE implies MXL (GSE lives inside the MXL path).
STAGE5_ENABLE_GSE = os.environ.get('STAGE5_ENABLE_GSE', '0') == '1'
STAGE5_ENABLE_MXL = (os.environ.get('STAGE5_ENABLE_MXL', '0') == '1') or STAGE5_ENABLE_GSE
# Module-level flag: print the DISTWALK==DIST banner only once (it fires per segment)
_DIST_WALK_BANNER_SHOWN = False
# Track which segments have already been warned about DIST/DISTWALK correlation
_DIST_WALK_CORR_WARNED: set = set()
# Track (c1, c2) collinear-column pairs already warned about at WARNING level.
# _dest_tree() re-detects the same structural collinearity on every zone-loop/
# batch iteration it's called for; without this the log fills with tens of
# thousands of repeats of the same ~10 pairs. First occurrence still logs at
# WARNING (so it's visible), repeats are demoted to DEBUG.
_WARNED_COLLINEAR_CA_PAIRS: set = set()

# ── ZONE MIGRATION (371): default zone universe ──────────────────────────────
# Fallback zone list when a data-derived one isn't available. In SA2-native mode
# OTAZ/DTAZ are 9-digit SA2 codes, so the legacy `range(1, 330)` (ORDER 1..329)
# is the wrong id space entirely. Read the real 371 SA2 codes from the migration
# reference (saved_data/seq_zone_ids_371.csv, else the demand_ reference); cache.
_ZONE_UNIVERSE_CACHE = None


def _default_zone_universe() -> list:
    """The active zone id universe (371 SA2 codes), as a sorted list."""
    global _ZONE_UNIVERSE_CACHE
    if _ZONE_UNIVERSE_CACHE is not None:
        return _ZONE_UNIVERSE_CACHE
    _here = os.path.dirname(os.path.abspath(__file__))
    for _cand, _col in (
        (os.path.join(_here, 'saved_data', 'seq_zone_ids_371.csv'), 'SA2_MAIN16'),
        (os.path.normpath(os.path.join(_here, '..', 'demand_',
                                       'sa2_zones_reference_357.csv')), 'sa2_code'),
    ):
        try:
            if os.path.exists(_cand):
                _z = sorted(int(x) for x in pd.read_csv(_cand)[_col]
                            if pd.notna(x) and int(x) > 0)
                if _z:
                    _ZONE_UNIVERSE_CACHE = _z
                    return _z
        except Exception:
            continue
    # Last-resort legacy fallback (should never hit in a real 371 run).
    _ZONE_UNIVERSE_CACHE = list(range(1, 330))
    return _ZONE_UNIVERSE_CACHE


def _default_n_zones() -> int:
    """Count of zones in the active universe (371)."""
    return len(_default_zone_universe())

# ── Industry-of-employment → mode affinity group ─────────────────────────────
# Source: ANZSIC 1-digit codes stored in 'indp' column (census/HHTS).
# Groups industrial/agri workers as strong DA captives (remote sites, no transit).
# Office/public/health workers as most transit-amenable (city-centre locations).
INDUSTRY_MODE_GROUP = {
    1:  'agri',         # Agriculture, Forestry, Fishing
    2:  'industrial',   # Mining
    3:  'industrial',   # Manufacturing
    4:  'industrial',   # Electricity, Gas, Water, Waste
    5:  'industrial',   # Construction
    6:  'retail',       # Wholesale Trade
    7:  'retail',       # Retail Trade
    8:  'hospitality',  # Accommodation and Food Services
    9:  'office',       # Transport, Postal, Warehousing
    10: 'office',       # Information, Media, Telecommunications
    11: 'office',       # Financial and Insurance Services
    12: 'office',       # Rental, Hiring, Real Estate
    13: 'professional', # Professional, Scientific, Technical Services
    14: 'professional', # Administrative and Support Services
    15: 'public',       # Public Administration and Safety
    16: 'public',       # Education and Training
    17: 'health',       # Health Care and Social Assistance
    18: 'hospitality',  # Arts and Recreation Services
    19: 'other',        # Other Services
    20: 'other',        # Not applicable / not employed
}

# Frozen-probability cell keys - these attributes define a person's cell in the
# HHTS→synth lookup table built by _compute_frozen_probs().
FROZEN_CELL_COLS = ['ptype', 'indp_group', 'income_band', 'veh_bin', 'age_group', 'has_licence']
FROZEN_PROB_COLS = ['p_DA', 'p_SR', 'p_Walk', 'p_Bike', 'p_Transit']

# Mode-specific skim column for travel time lookup in _get_tt.
# DISTWALK and DISTBIKE confirmed present in OMX (see error log).
# Walk/bike values are km - _get_tt divides by speed to get minutes.
MODE_TT_COL  = {
    1: 'SOVTOLL_TIME__AM',        # DA - auto IVT, AM peak (minutes)
    2: 'SOVTOLL_TIME__AM',        # Other (shared ride)
    3: 'DISTWALK',                # Walk - km, converted at 5 km/h in _get_tt
    4: 'DISTBIKE',                # Bike - km, converted at 15 km/h in _get_tt
    5: 'WLK_LOC_WLK_TOTIVT__AM', # Transit - walk-to-transit IVT (minutes)
}

WALK_KMH = 5.0    # walk speed for distance→time conversion
BIKE_KMH = 15.0   # bike speed for distance→time conversion
WALK_MAX_KM = 10.0
BIKE_MAX_KM = 25.0

try:
    import larch as lx
    from larch import P, X
    from addicty import Dict
    LARCH_AVAILABLE = True
except ImportError:
    LARCH_AVAILABLE = False
    logger.warning("larch not installed - Stage 5 will use distance-only fallback")


def _lx_model(dt, **kwargs) -> 'lx.Model':
    """
    Create a Larch model, trying JAX first and falling back to Numba.

    JAX uses XLA JIT compilation for the log-likelihood and gradient -
    faster on GPU/TPU and competitive on CPU for large datasets.  If JAX
    is not installed or fails to initialise (common on Windows without
    CUDA), Numba's finite-difference implementation is used instead.

    Pass ``compute_engine`` in kwargs to override both (e.g. for testing).
    """
    if 'compute_engine' in kwargs:
        _engine = kwargs.pop('compute_engine')
        try:
            return lx.Model(dt, compute_engine=_engine, **kwargs)
        except Exception as _e:
            logger.debug(f"_lx_model: requested engine='{_engine}' failed ({_e}), trying alternatives")
            # Fall through to auto-detect
    # GA fitness evaluations explicitly request numba via GA_LARCH_ENGINE=numba
    # to avoid the cusolver_gesvdj_ffi CUDA kernel error that fires when Larch's
    # JAX path tries to compute the covariance Hessian on the CUDA device.
    # The env var is set by ga_hypothesis_testing before each fitness call.
    _forced = os.environ.get('GA_LARCH_ENGINE', '').strip().lower()
    if _forced in ('numba', 'jax'):
        try:
            m = lx.Model(dt, compute_engine=_forced, **kwargs)
            logger.debug(f"_lx_model: using forced compute_engine='{_forced}' (GA_LARCH_ENGINE)")
            return m
        except Exception as _e:
            logger.debug(f"_lx_model: forced engine='{_forced}' failed ({_e}), falling back")

    # Windows frequently logs hwloc CPU-binding errors under JAX/XLA.
    # Prefer numba first there for stable long-running batch estimation.
    _engines = ('numba', 'jax') if os.name == 'nt' else ('jax', 'numba')
    for engine in _engines:
        try:
            m = lx.Model(dt, compute_engine=engine, **kwargs)
            logger.debug(f"_lx_model: using compute_engine='{engine}'")
            return m
        except Exception as _e:
            logger.debug(f"_lx_model: engine='{engine}' unavailable ({_e}), trying next")
    raise RuntimeError("No Larch compute engine available (tried jax, numba)")


# ===========================================================================
# SKIM HELPERS
# ===========================================================================

def _open_skims(skims):
    if isinstance(skims, str):
        return omx.open_file(skims, 'r'), True
    return skims, False


def _skim_flat(skims) -> pd.DataFrame:
    """Convert OMX to flat OTAZ/DTAZ DataFrame with all matrices as columns."""
    try:
        import read_skims as rs
        df = rs.read_omx_skims(skims)
        # print skim value ranges so unit issues are immediately visible
        tt_cols = [c for c in df.columns
                   if any(x in c for x in ['TIME', 'DIST', 'WALK', 'BIKE'])]
        if tt_cols:
            print("  Skim value ranges (sample - check units are minutes/km):")
            for col in tt_cols[:6]:
                v = df[col].replace(0, np.nan).dropna()
                if len(v):
                    print(f"    {col}: "
                          f"min={v.min():.1f} median={v.median():.1f} "
                          f"max={v.max():.1f} p95={v.quantile(0.95):.1f}")
        return df
    except Exception:
        pass
    try:
        mapping = dict(skims.mapping(skims.listMappings()[0]))
        # handle {np.uint32: int} style mapping
        zone_ids = sorted(mapping, key=lambda k: mapping[k])
        zone_ids = [int(z) for z in zone_ids]
        N = len(zone_ids)
        rows = []
        for o_idx, otaz in enumerate(zone_ids):
            for d_idx, dtaz in enumerate(zone_ids):
                rows.append({'OTAZ': otaz, 'DTAZ': dtaz})
        base_df = pd.DataFrame(rows)
        mat_cols = {}
        for name in skims.listMatrices():
            mat = np.array(skims[name])
            mat_cols[name] = mat.ravel()
        df = pd.concat([base_df, pd.DataFrame(mat_cols)], axis=1)
        return df
    except Exception as e:
        logger.warning(f"Could not read skims to flat df: {e}")
        return pd.DataFrame()


def _skim_lookup(skim_df: pd.DataFrame) -> dict:
    """Build {(otaz,dtaz): {col: val}} lookup from flat skim df."""
    if skim_df.empty:
        return {}
    idx_cols = ['OTAZ', 'DTAZ']
    val_cols = [c for c in skim_df.columns if c not in idx_cols]
    lookup   = {}
    for row in skim_df.itertuples(index=False):
        lookup[(int(row.OTAZ), int(row.DTAZ))] = {
            c: getattr(row, c) for c in val_cols}
    return lookup


def _read_crash_od_table(crash_source, zone_ids: list = None) -> pd.DataFrame:
    """
    Load an OD crash table with columns OTAZ, DTAZ, CRASH.

    Supports:
      - DataFrame already containing OD crash values
      - CSV/XLSX path
      - OMX path containing matrix 'CRASH'
    """
    if crash_source is None:
        return pd.DataFrame(columns=['OTAZ', 'DTAZ', 'CRASH'])

    if isinstance(crash_source, pd.DataFrame):
        raw = crash_source.copy()
    else:
        path = str(crash_source)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Crash source not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext in {'.csv', '.txt'}:
            raw = pd.read_csv(path)
        elif ext in {'.xlsx', '.xls'}:
            raw = pd.read_excel(path)
        elif ext == '.omx':
            cf, owned = _open_skims(path)
            try:
                if 'CRASH' not in cf.listMatrices():
                    raise RuntimeError("Crash OMX missing matrix 'CRASH'")
                mat = np.array(cf['CRASH'])
                if zone_ids is None:
                    mp = dict(cf.mapping(cf.listMappings()[0]))
                    zone_ids = [int(z) for z in sorted(mp.keys(), key=lambda k: mp[k])]
                z = np.asarray(zone_ids, dtype=int)
                raw = pd.DataFrame({
                    'OTAZ': np.repeat(z, len(z)),
                    'DTAZ': np.tile(z, len(z)),
                    'CRASH': mat.ravel(),
                })
            finally:
                if owned:
                    cf.close()
        else:
            raise RuntimeError(f"Unsupported crash source extension: {ext}")

    cols_upper = {str(c).upper(): c for c in raw.columns}
    o_col = next((cols_upper.get(c) for c in ['OTAZ', 'ORIG', 'ORIGIN', 'O', 'ZONE_O']), None)
    d_col = next((cols_upper.get(c) for c in ['DTAZ', 'DEST', 'DESTINATION', 'D', 'ZONE_D']), None)
    c_col = next((cols_upper.get(c) for c in ['CRASH', 'CRASHES', 'CRASH_COUNT', 'COUNT']), None)
    if o_col is None or d_col is None or c_col is None:
        raise RuntimeError(
            "Crash table must contain origin, destination, and crash-count columns "
            "(e.g. OTAZ, DTAZ, CRASH)"
        )

    out = raw[[o_col, d_col, c_col]].copy()
    out.columns = ['OTAZ', 'DTAZ', 'CRASH']
    out['OTAZ'] = pd.to_numeric(out['OTAZ'], errors='coerce').fillna(0).astype(int)
    out['DTAZ'] = pd.to_numeric(out['DTAZ'], errors='coerce').fillna(0).astype(int)
    out['CRASH'] = pd.to_numeric(out['CRASH'], errors='coerce').fillna(0.0).astype(float)
    out = out[(out['OTAZ'] > 0) & (out['DTAZ'] > 0)]
    out = out.groupby(['OTAZ', 'DTAZ'], as_index=False)['CRASH'].sum()
    return out


def _resolve_default_crash_source() -> Optional[str]:
    """Auto-detect a crash OD source, preferring OMX CRASH matrices."""
    omx_candidates = [
        'downcasted_matrices_base.omx',
    ]
    for p in omx_candidates:
        if not os.path.isfile(p):
            continue
        try:
            cf, owned = _open_skims(p)
            try:
                if 'CRASH' in cf.listMatrices():
                    return p
            finally:
                if owned:
                    cf.close()
        except Exception as e:
            logger.warning(f"Could not inspect OMX crash source '{p}': {e}")

    candidates = [
        'OD_Crashes.csv',
        'SA2_OD_Crash_Matrix_PT.xlsx',
        'SA2_OD_Crash_Matrix.xlsx',
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _dataset_var_names(ds_like) -> set:
    """Return variable/column names from xarray-like or pandas-like datasets."""
    # DataTree nodes often expose the underlying dataset as .ds/.dataset.
    # Recursing here keeps downstream variable audits version-agnostic.
    try:
        _embedded = getattr(ds_like, 'ds', None) or getattr(ds_like, 'dataset', None)
        if _embedded is not None and _embedded is not ds_like:
            return _dataset_var_names(_embedded)
    except Exception:
        pass
    try:
        if hasattr(ds_like, 'data_vars'):
            return set(str(c) for c in list(ds_like.data_vars))
    except Exception:
        pass
    try:
        _vars = getattr(ds_like, 'variables', None)
        if _vars is not None:
            if isinstance(_vars, dict):
                return set(str(c) for c in _vars.keys())
            if isinstance(_vars, (list, tuple, set)):
                return set(str(c) for c in _vars)
    except Exception:
        pass
    try:
        if isinstance(ds_like, pd.DataFrame):
            return set(str(c) for c in ds_like.columns)
    except Exception:
        pass
    try:
        if hasattr(ds_like, 'columns'):
            _cols = list(getattr(ds_like, 'columns'))
            if _cols:
                return set(str(c) for c in _cols)
    except Exception:
        pass
    try:
        if hasattr(ds_like, 'keys'):
            _keys = list(ds_like.keys())
            _idx = {'otaz', 'dtaz', 'OTAZ', 'DTAZ', 'index'}
            if _keys:
                return set(str(c) for c in _keys if str(c) not in _idx)
    except Exception:
        pass
    try:
        if hasattr(ds_like, 'to_dataframe'):
            _df = ds_like.to_dataframe().reset_index()
            _idx = {'otaz', 'dtaz', 'OTAZ', 'DTAZ', 'index'}
            return set(str(c) for c in _df.columns if str(c) not in _idx)
    except Exception:
        pass
    return set()


def _build_od_dataset(skims, skim_df: pd.DataFrame = None):
    """
    Build an xarray OD Dataset for DataTree relationships.

    Always returns a plain xr.Dataset indexed by (otaz, dtaz) - never a Larch
    Dataset from lx.Dataset.construct.from_omx().  Larch's sharrow compiler
    validates utility expressions against the *original OMX file* when the od
    node is a Larch/OMX Dataset, so aliases added via .assign() (e.g. DIST
    from SOVTOLL_DIST__AM) are invisible at compile time and produce n_params=0.
    A plain xr.Dataset bypasses this validation: sharrow treats it as regular
    data and can see all variables, including aliases.
    """
    import xarray as xr

    if skim_df is None or skim_df.empty:
        skim_df = _skim_flat(skims)
    if skim_df is None or skim_df.empty:
        raise RuntimeError(
            "_build_od_dataset: could not build OD dataset - "
            "skim_df is empty and _skim_flat failed to read the OMX"
        )

    od_df = skim_df.copy()
    for old, new in (('OTAZ', 'otaz'), ('DTAZ', 'dtaz')):
        if old in od_df.columns and new not in od_df.columns:
            od_df = od_df.rename(columns={old: new})
    if 'otaz' not in od_df.columns or 'dtaz' not in od_df.columns:
        raise RuntimeError("_build_od_dataset: skim_df missing OTAZ/DTAZ columns")

    od_df['otaz'] = pd.to_numeric(od_df['otaz'], errors='coerce').fillna(0).astype(int)
    od_df['dtaz'] = pd.to_numeric(od_df['dtaz'], errors='coerce').fillna(0).astype(int)
    od_df = od_df.drop_duplicates(['otaz', 'dtaz'])

    # DIST alias - DEST_UEC uses od.DIST; SEQ OMX stores it as SOVTOLL_DIST__AM
    if 'DIST' not in od_df.columns and 'SOVTOLL_DIST__AM' in od_df.columns:
        od_df['DIST'] = pd.to_numeric(od_df['SOVTOLL_DIST__AM'], errors='coerce').fillna(0.0)
    elif 'DIST' not in od_df.columns:
        logger.warning(
            "_build_od_dataset: neither DIST nor SOVTOLL_DIST__AM found in OMX - "
            "od.DIST will be zero; destination distance expressions will fail"
        )
        od_df['DIST'] = 0.0

    # CRASH placeholder - overridden in _dest_tree when crash_skims provided
    if 'CRASH' not in od_df.columns:
        od_df['CRASH'] = 0.0

    # Convert to plain xr.Dataset indexed by (otaz, dtaz) so sharrow can compile
    # all variables - including DIST - without OMX-file validation.
    try:
        od_xr = od_df.set_index(['otaz', 'dtaz']).to_xarray()
        # Fill any NaN (from missing OD pairs) with 0
        od_xr = od_xr.fillna(0.0)
        logger.debug(
            f"_build_od_dataset: xr.Dataset {len(od_xr.data_vars)} matrices, "
            f"{len(od_xr.coords['otaz'])} zones"
        )
        return od_xr
    except Exception as _xr_e:
        # Extremely unlikely: return raw DataFrame as last resort
        logger.warning(f"_build_od_dataset: xarray conversion failed ({_xr_e}) - returning DataFrame")
        return od_df


def _dest_uec_refs_present(dt, expr: str) -> bool:
    """Return True when all DataTree variable refs in expr are available."""
    refs = re.findall(r"\b(base|person|emp|od)\.([A-Za-z_][A-Za-z0-9_]*)", str(expr))
    if not refs:
        return True
    cache = {}
    for node, var in refs:
        if node not in cache:
            try:
                cache[node] = _dataset_var_names(dt[node])
            except Exception:
                cache[node] = set()
        # Some Larch backends expose DataTree nodes that are not introspectable
        # through our helper APIs. In that case, avoid false-negative drops and
        # let model compilation validate the reference.
        if not cache[node]:
            continue
        if var not in cache[node]:
            return False
    return True


def _filter_dest_uec_for_datatree(dt, segment: str, uec: dict, cond: dict = None) -> dict:
    """Drop destination utility terms whose DataTree variables are unavailable."""
    if cond is None:
        cond = DEST_COND
    filtered = {}
    dropped = []
    for coeff, (expr, segs) in uec.items():
        if segs is not None and segment not in segs:
            continue
        _expr_ok = _dest_uec_refs_present(dt, expr)
        _cond_ok = True
        _cond_expr = cond.get(coeff)
        if _cond_expr:
            _cond_ok = _dest_uec_refs_present(dt, _cond_expr)
        if _expr_ok and _cond_ok:
            filtered[coeff] = (expr, segs)
        else:
            dropped.append(coeff)
    if dropped:
        logger.warning(
            f"_build_dest_model [{segment}]: dropped {len(dropped)} UEC terms with missing refs: "
            f"{sorted(dropped)[:10]}{'...' if len(dropped) > 10 else ''}"
        )
    return filtered


def _filter_dest_uec_by_source_columns(segment: str,
                                       uec: dict,
                                       cond: dict,
                                       skim_cols: set,
                                       landuse_cols: set,
                                       person_cols: set,
                                       base_cols: set = None) -> dict:
    """
    Filter destination terms using source-column availability before DataTree build.

    This avoids Larch compile collapse when expressions reference unavailable
    columns that pass DataTree introspection heuristics.
    """
    if base_cols is None:
        base_cols = {'logsum', 'log_correction'}
    refs_pat = re.compile(r"\b(base|person|emp|od)\.([A-Za-z_][A-Za-z0-9_]*)")
    out = {}
    dropped = []
    for coeff, (expr, segs) in uec.items():
        if segs is not None and segment not in segs:
            continue
        expr_ok = True
        for node, var in refs_pat.findall(str(expr)):
            if node == 'od' and var not in skim_cols:
                expr_ok = False
                break
            if node == 'emp' and var not in landuse_cols:
                expr_ok = False
                break
            if node == 'person' and var not in person_cols:
                expr_ok = False
                break
            if node == 'base' and var not in base_cols:
                expr_ok = False
                break
        cond_ok = True
        cond_expr = cond.get(coeff)
        if cond_expr:
            for node, var in refs_pat.findall(str(cond_expr)):
                if node == 'od' and var not in skim_cols:
                    cond_ok = False
                    break
                if node == 'emp' and var not in landuse_cols:
                    cond_ok = False
                    break
                if node == 'person' and var not in person_cols:
                    cond_ok = False
                    break
                if node == 'base' and var not in base_cols:
                    cond_ok = False
                    break
        if expr_ok and cond_ok:
            out[coeff] = (expr, segs)
        else:
            dropped.append(coeff)
    if dropped:
        logger.info(
            f"_filter_dest_uec_by_source_columns [{segment}]: dropped "
            f"{len(dropped)} terms due to missing source columns: "
            f"{sorted(dropped)[:12]}{'...' if len(dropped) > 12 else ''}"
        )
    return out


def _dest_uec_from_feature_list(custom_vars: list) -> dict:
    """
    Convert a list-style destination feature override into a DEST_UEC dict.

    This mirrors _dest_spec_custom selection semantics but returns a plain
    dictionary so the caller can run strict source-column filtering before
    Larch compilation.
    """
    if not custom_vars:
        return dict(DEST_UEC)

    always_include = {
        "c_logsum", "c_log_dist", "c_dist", "c_size_term",
        "c_crash",             # c_aggl NO LONGER force-included: the blended
                               # composite is collinear with the per-type themed
                               # ACDC (retail/office/edu/leisure/parks/health/food/
                               # civic) and, when forced, holdfast-zeroed them all.
                               # Agglomeration is now carried by the themed genes
                               # (require-one in ga_constraints). c_comp also out:
                               # collinear with c_aggl (|r|≈0.9998) and, when both
                               # are forced, the pair blows up to huge equal-and-
                               # opposite values (+4.5/−4.5). It stays GA-selectable
                               # but mutually exclusive with c_aggl (ga_constraints).
        "c_incp_dist", "c_veh_dist", "c_occp_dist",
        "c_auto_work_acc", "c_auto_shop_acc",
        "c_walk_acc", "c_transit_acc",
        "c_noveh_walk_acc", "c_noveh_transit_acc",
        "c_indp_finance", "c_indp_health", "c_indp_retail",
        "c_edu_post", "c_edu_primary", "c_edu_secondary",
    }

    try:
        _tokens = [str(v) for v in custom_vars]
    except Exception:
        _tokens = []

    _token_set = set(_tokens)
    # The piecewise distance spline REPLACES the single linear c_dist (they are
    # collinear: Σ spline = DIST), so drop c_dist from the forced-keep set when a
    # spline term is selected — otherwise the Hessian block is singular.
    if _token_set & {'c_dist_0_2', 'c_dist_2_5', 'c_dist_5_10'}:
        always_include = always_include - {'c_dist'}
    out = {}
    for coeff, val in DEST_UEC.items():
        expr = str(val[0])
        # Include when force-kept, when the coefficient name itself is selected
        # (clean coeff-name gating for GA candidates like c_dist_0_2 / c_cbd_dest),
        # or when any selected column token appears in the expression.
        if (coeff in always_include or coeff in _token_set
                or any(tok in expr for tok in _tokens)):
            out[coeff] = val
    if not out:
        return dict(DEST_UEC)
    return out


def _fill_dest_coef_with_segment_defaults(dest_coef: dict, segment: str) -> dict:
    """Backfill missing destination coefficients with segment defaults."""
    coef = dict(dest_coef or {})
    seg_defaults = SEGMENT_COEFFICIENTS.get(segment, {})
    active_terms = []
    for name, (_, segs) in DEST_UEC.items():
        if segs is None or segment in segs:
            active_terms.append(name)
    for name in active_terms:
        if name not in coef:
            coef[name] = float(seg_defaults.get(name, 0.0))
    return coef


# Realistic travel time caps per mode (minutes)
MODE_TT_CAP = {
    1: 120.0,   # DA       - 2 hrs max
    2: 120.0,   # Other    - 2 hrs max
    3: 120.0,   # Walk     - 120 min max = 10 km at 5 km/h
    4: 100.0,   # Bike     - 100 min max = 25 km at 15 km/h
    5: 150.0,   # Transit  - 2.5 hrs max
}
MODE_TT_DEFAULT = {1: 15.0, 2: 15.0, 3: 20.0, 4: 15.0, 5: 25.0}


def _get_tt(lookup: dict, otaz: int, dtaz: int, mode: int) -> float:
    """
    Get travel time (minutes) for OD pair and mode from skim lookup.

    Skims can store times in different units depending on the software:
      - Aimsun / Emme: minutes
      - Some OMX files: seconds (values > 600 for a 10-min trip)
      - Some OMX files: tenths of minutes

    We detect the unit by value range and convert accordingly.
    Also caps at realistic maxima per mode.
    """
    if otaz == dtaz:
        return 3.0   # intra-zonal - short but non-zero

    sk  = lookup.get((otaz, dtaz), {})
    col = MODE_TT_COL.get(mode, 'SOVTOLL_TIME__AM')
    val = sk.get(col, None)

    # ── Walk / Bike: DISTWALK/DISTBIKE stored in km in this OMX ─────────────
    if mode == 3 and val is not None and val > 0:   # Walk
        km = float(val)
        km = min(km, WALK_MAX_KM)
        return float(np.clip(km * 60.0 / WALK_KMH, 0.5, MODE_TT_CAP[3]))

    if mode == 4 and val is not None and val > 0:   # Bike
        km = float(val)
        km = min(km, BIKE_MAX_KM)
        return float(np.clip(km * 60.0 / BIKE_KMH, 0.5, MODE_TT_CAP[4]))

    # ── Transit: try WLK_LOC_WLK_TOTIVT__AM, fall back to auto × 1.5 ─────
    if mode == 5:
        transit_val = sk.get('WLK_LOC_WLK_TOTIVT__AM', None)
        if transit_val is None or transit_val <= 0:
            auto_val = sk.get('SOVTOLL_TIME__AM', None)
            transit_val = (float(auto_val) * 1.5) if auto_val and auto_val > 0 else None
        if transit_val and transit_val > 0:
            val = float(transit_val)
            if val > 600: val /= 60.0
            elif val > 200: val /= 10.0
            return float(np.clip(val, 0.5, MODE_TT_CAP[5]))

    # ── Auto modes: use SOVTOLL_TIME__AM with period fallbacks ────────────
    if val is None or val <= 0:
        for fallback_col in ['SOVTOLL_TIME__AM', 'SOVTOLL_TIME__PM',
                              'SOVTOLL_TIME__MD', 'SOVTOLL_TIME__EV']:
            val = sk.get(fallback_col, None)
            if val and val > 0:
                break
        if not val or val <= 0:
            return float(MODE_TT_DEFAULT.get(mode, 15.0))

    val = float(val)

    # unit detection (auto IVT stored as seconds or tenths in some OMX)
    if val > 600:
        val = val / 60.0
    elif val > 200:
        val = val / 10.0

    cap = float(MODE_TT_CAP.get(mode, 120.0))
    return float(np.clip(val, 0.5, cap))


# ===========================================================================
# DATA PREP
# ===========================================================================

def _prep_hhts_tours(tours: pd.DataFrame, persons: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise HHTS tours table for Larch mode + destination estimation.

    Handles the column naming differences between the tours table output
    of build_as_tours() and what Larch expects:
        tour_origin      → OTAZ
        tour_destination → DTAZ
        primary_purpose  → main_purpose
        tour_mode        → mapped to integer codes
    """
    t = tours.copy()
    if 'person_id' not in t.columns and 'PERSID' in t.columns:
        t = t.rename(columns={'PERSID': 'person_id'})

    # ── resolve OTAZ ─────────────────────────────────────────────────────
    if 'OTAZ' not in t.columns:
        for candidate in ['tour_origin', 'home_otaz', 'origin_zone', 'start_location', 'otaz']:
            if candidate in t.columns:
                t['OTAZ'] = t[candidate]
                break
    if 'OTAZ' not in t.columns:
        raise ValueError("_prep_hhts_tours: cannot find OTAZ / tour_origin column.")

    # ── resolve DTAZ ─────────────────────────────────────────────────────
    if 'DTAZ' not in t.columns:
        for candidate in ['tour_destination', 'destination_zone', 'dest_zone', 'end_location', 'dtaz']:
            if candidate in t.columns:
                t['DTAZ'] = t[candidate]
                break
    if 'DTAZ' not in t.columns:
        raise ValueError("_prep_hhts_tours: cannot find DTAZ / tour_destination column.")

    # ── resolve purpose ───────────────────────────────────────────────────
    if 'main_purpose' not in t.columns:
        for candidate in ['primary_purpose', 'tour_primary_purpose']:
            if candidate in t.columns:
                t['main_purpose'] = t[candidate]
                break
    if 'primary_purpose' not in t.columns and 'main_purpose' in t.columns:
        t['primary_purpose'] = t['main_purpose']

    # ── ptour_id ──────────────────────────────────────────────────────────
    if 'tour_id' in t.columns:
        t['ptour_id'] = t['person_id'].astype(str) + '_' + t['tour_id'].astype(str)
    else:
        t['ptour_id'] = t['person_id'].astype(str) + '_' + t.index.astype(str)

    # ── tour_mode ─────────────────────────────────────────────────────────
    if 'tour_mode' in t.columns:
        # may be string ('DA','Walk',...) or already int
        if t['tour_mode'].dtype == object:
            t['tour_mode'] = t['tour_mode'].map(MODE_MAP).fillna(2).astype(int)
        else:
            t['tour_mode'] = pd.to_numeric(
                t['tour_mode'], errors='coerce').fillna(2).astype(int)
    else:
        logger.warning("_prep_hhts_tours: tour_mode missing - defaulting to DA (1)")
        t['tour_mode'] = 1

    if 'tour_distance' in t.columns:
        _tour_dist_km = pd.to_numeric(t['tour_distance'], errors='coerce')
        _bad_walk = (t['tour_mode'] == MODE_MAP['Walk']) & (_tour_dist_km > WALK_MAX_KM)
        if _bad_walk.any():
            t.loc[_bad_walk, 'tour_mode'] = MODE_MAP['SR']
            logger.info(
                f"_prep_hhts_tours: recoded {int(_bad_walk.sum())} walk tours "
                f"over {WALK_MAX_KM:.0f} km to SR"
            )

    # ── numeric zone columns ──────────────────────────────────────────────
    for col in ['OTAZ', 'DTAZ']:
        t[col] = pd.to_numeric(t[col], errors='coerce').fillna(0).astype(int)

    # ── filter invalid zones ──────────────────────────────────────────────
    before = len(t)
    t = t[(t['OTAZ'] > 0) & (t['DTAZ'] > 0) &
          (t['OTAZ'] != 330) & (t['DTAZ'] != 330)].copy()
    if len(t) < before:
        logger.info(f"_prep_hhts_tours: dropped {before - len(t)} rows with invalid zones")

    # ── merge person attributes ───────────────────────────────────────────
    pp = persons.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})
    pp = _ensure_person_behavioural_cols(pp)

    # Expanded keep list: all columns referenced in MODE_UEC / DEST_UEC
    # via person.xxx (accessed through the DataTree relationship) plus
    # any that are useful as tour-level columns for downstream steps.
    keep = [c for c in [
        'person_id',
        # Original socio-economic
        'incp', 'indp', 'sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING',
        # WORKSTAT / STUDYING binary dummies (preferred over raw codes in UEC expressions)
        'workstat_fulltime', 'workstat_parttime', 'workstat_notworking',
        'studying_none', 'studying_primary', 'studying_secondary',
        'studying_tertiary_pt', 'studying_tertiary_ft',
        # New behavioural additions
        'age5p', 'AGEGROUP',    # age bands - walk/bike/distance deterrence
        'CARLICENCE',           # car licence - DA prerequisite
        'occp_group',           # occupation group - work destination
        'occp_mgr',             # binary: professional/managerial
        'occp_trades',          # binary: trades/technical
        'bikes',                # HH bicycle count - bike mode
        'ptype',                # person type string (for logging)
        'HOME_SA1',             # SA1 sub-zone (fallback for Level4 if not available)
        'HOME_LVL4',            # SEQ-STM Level4 micro-zone for walk/bike skim lookup
    ] if c in pp.columns]
    # Coerce person_id to str on both sides to prevent pandas merge error when
    # HHTS tours have int64 person_id and synth persons have object person_id.
    t['person_id']  = t['person_id'].astype(str)
    pp['person_id'] = pp['person_id'].astype(str)
    t = t.merge(pp[keep], on='person_id', how='left', suffixes=('', '_p'))

    # Fill any post-merge missing behavioural attributes with safe defaults.
    # This avoids sporadic chosen_but_not_available flags caused by unmatched
    # person_id rows leaving CARLICENCE/vehrd/etc as NaN.
    for _c, _default in _PERSON_BEHAVIOURAL_DEFAULTS.items():
        if _c in t.columns:
            t[_c] = pd.to_numeric(t[_c], errors='coerce').fillna(_default)

    # ── _time_period: numeric departure period for c_peak_da ─────────────
    # 0=early(<6am), 1=am_peak(6-9), 2=midday(9-15), 3=pm_peak(15-18),
    # 4=evening(18-21), 5=late(≥21).  Mirrors zsim_helper.get_period().
    if 'STARTIME' in t.columns:
        def _period_from_min(m):
            m = float(m) if not pd.isna(m) else 480
            if m < 360:  return 0
            if m < 540:  return 1
            if m < 900:  return 2
            if m < 1080: return 3
            if m < 1260: return 4
            return 5
        t['_time_period'] = t['STARTIME'].apply(_period_from_min).astype(int)
    else:
        t['_time_period'] = 1   # default am_peak

    # ── activity count defaults ───────────────────────────────────────────
    for col in ['n_shopping_out', 'outbound_stops']:
        if col not in t.columns:
            t[col] = 0
    for col in ['n_shopping_out', 'outbound_stops']:
        t[col] = pd.to_numeric(t[col], errors='coerce').fillna(0).astype(int)

    # ── joint travel indicator ────────────────────────────────────────────
    joint_cols = [c for c in ['joint_travel', 'JOINTTRIP', 'JOINT_TRIP',
                              'travel_party', 'hh_travelers', 'num_travellers',
                              'num_travelers'] if c in t.columns]
    if joint_cols:
        joint_src = pd.to_numeric(t[joint_cols[0]], errors='coerce').fillna(0)
        t['joint_travel'] = (joint_src > 0).astype(int)
    else:
        t['joint_travel'] = (
            (pd.to_numeric(t['outbound_stops'], errors='coerce').fillna(0) > 0) |
            (t['main_purpose'].astype(str).str.lower() == 'escort')
        ).astype(int)

    # ── walk / bike availability flags ──────────────────────────────────────
    # Sharrow-compiled availability cannot use X() expressions referencing
    # OD skims. Pre-compute boolean availability columns here.
    # Use tour_distance if available; otherwise default to always-available
    # and let the distance disutility in the utility function do the work.
    if 'tour_distance' in t.columns:
        dist_km = pd.to_numeric(t['tour_distance'], errors='coerce').fillna(999)
        t['walk_avail'] = (dist_km < WALK_MAX_KM).astype(int)
        t['bike_avail'] = (dist_km < BIKE_MAX_KM).astype(int)
    else:
        # no distance info - make all modes available, rely on utility betas
        t['walk_avail'] = 1
        t['bike_avail'] = 1

    # ── ensure chosen mode is always marked available in training data ──────
    # Force availability=1 for whichever mode a person actually chose so Larch
    # never sees a "chosen_but_not_available" observation.  Covers all modes,
    # not just Walk/Bike - Transit and SR choosers may also fail availability
    # conditions in edge cases (e.g. zero-vehicle transit riders).
    if 'tour_mode' in t.columns:
        _avail_cols = {
            MODE_MAP['Walk']:    'walk_avail',
            MODE_MAP['Bike']:    'bike_avail',
            MODE_MAP['Transit']: 'transit_avail',
            MODE_MAP['SR']:      'sr_avail',
        }
        for mode_id, avail_col in _avail_cols.items():
            if avail_col in t.columns:
                t.loc[t['tour_mode'] == mode_id, avail_col] = 1
        if 'CARLICENCE' in t.columns:
            t.loc[t['tour_mode'] == MODE_MAP['DA'], 'CARLICENCE'] = 1

    # ── survey expansion weights ──────────────────────────────────────
    # QTS uses PERSWGT23 (person-level expansion weight). Merging onto tours
    # so the mode and destination models are weighted to population totals.
    # Without weights, oversampled zones/demographics dominate the estimates.
    weight_col = next((c for c in ['PERSWGT23', 'DEMOGWGT23', 'TRIPWGT23',
                                   'TRIPWGT', 'PERSWGT', 'WGTPERS',
                                   'weight', 'expansion_weight']
                       if c in persons.columns), None)
    if weight_col:
        wgt = pp[['person_id', weight_col]].rename(columns={weight_col: 'weight'})
        t = t.merge(wgt, on='person_id', how='left')
        t['weight'] = pd.to_numeric(t['weight'], errors='coerce').fillna(1.0)
        # normalise to avoid numerical issues - keeps relative weights intact
        t['weight'] = t['weight'] / t['weight'].mean()
        logger.info(f"_prep_hhts_tours: applied weights from '{weight_col}' "
                    f"(mean={t['weight'].mean():.3f}, "
                    f"min={t['weight'].min():.3f}, max={t['weight'].max():.3f})")
    else:
        t['weight'] = 1.0
        logger.info("_prep_hhts_tours: no weight column found - using unit weights")

    logger.info(
        f"_prep_hhts_tours: {len(t)} tours | "
        f"OTAZ range {t['OTAZ'].min()}-{t['OTAZ'].max()} | "
        f"DTAZ range {t['DTAZ'].min()}-{t['DTAZ'].max()} | "
        f"tour_mode dist: {t['tour_mode'].value_counts().to_dict()} | "
        f"walk_avail={t['walk_avail'].sum()} bike_avail={t['bike_avail'].sum()} | "
        f"joint_travel={int(t['joint_travel'].sum())}"
    )

    return t


# Raw purpose strings that map to each segment
_WORK_PURPOSES = {'work', 'Work', 'Work (my workplace)', 'Work (other work reason)',
                  'Direct Work Commute', 'Work Related', 'workFullTime', 'workPartTime',
                  'work_ftw', 'work_ptw', 'work_ft', 'work_pt', 'WORK', 'WorkFull', 'WorkPart',
                  'mandatory_work', 'at_work'}
_EDU_PURPOSES  = {'edu', 'Education', 'Childcare or Kindergarten',
                  'Childcare or kindergarten', 'school', 'School',
                  'University', 'university', 'college', 'College',
                  'EDU', 'SCHOOL', 'primary_school', 'secondary_school',
                  'tertiary', 'Tertiary', 'mandatory_school', 'at_school'}
_NONMAND_PURPOSES = set(NON_MANDATORY) | {
    'social', 'Social', 'Social Visit', 'Recreation/leisure activity', 'Recreation',
    'shopping', 'Shopping',
    'escort', 'Pick up or drop off a passenger', 'Pickup/Dropoff Someone',
    'eat', 'Eat/drink',
    'othmaint', 'Personal Business', 'Personal business',
    'Pickup/deliver something (not work-related)',
    'othdiscr', 'Other place (please specify)', 'Other Place', 'Refuel my vehicle',
    # Segment-level label itself — some upstream stages (e.g. synthetic population
    # activities, as opposed to HHTS granular purposes) tag primary_purpose with the
    # aggregated segment name directly rather than a granular purpose. 'work'/'edu'
    # already matched here by coincidence (both _WORK_PURPOSES/_EDU_PURPOSES include
    # the literal segment name); 'nonmand' did not, which silently dropped every
    # nonmand tour from mode prediction (0 tours matched -> hardcoded DA default).
    'nonmand', 'Nonmand', 'NONMAND', 'non_mandatory', 'Non-mandatory', 'NonMandatory',
}

# Raw purpose string -> one of the 6 granular SIZE_TERM_COEFFICIENTS keys used
# within the pooled 'nonmand' destination-choice model (social/escort/shopping/
# othmaint/othdiscr/eat - see SEGMENT_PURPOSES['nonmand']). Used by _dest_tree
# to give each nonmand tour ITS OWN sub-purpose's size term (a shopping trip is
# attracted to Shopping/retail/RETEMPN, a social trip to Entertainment/
# sports_centre/etc) instead of one size term blended equally across all six
# sub-purposes for every tour regardless of what it actually is. Anything not
# in this map (e.g. the generic 'nonmand' segment-level label some upstream
# stages use instead of a granular purpose) falls back to the blended
# 'nonmand' composite key, same as before this fix.
_NONMAND_SUBPURPOSE_MAP = {
    'social': 'social', 'Social': 'social', 'Social Visit': 'social',
    'Recreation/leisure activity': 'social', 'Recreation': 'social',
    'shopping': 'shopping', 'Shopping': 'shopping',
    'escort': 'escort', 'Pick up or drop off a passenger': 'escort',
    'Pickup/Dropoff Someone': 'escort',
    'eat': 'eat', 'Eat/drink': 'eat',
    'othmaint': 'othmaint', 'Personal Business': 'othmaint', 'Personal business': 'othmaint',
    'Pickup/deliver something (not work-related)': 'othmaint',
    'othdiscr': 'othdiscr', 'Other place (please specify)': 'othdiscr',
    'Other Place': 'othdiscr', 'Refuel my vehicle': 'othdiscr',
}


def _filter_seg(tours: pd.DataFrame, segment: str) -> pd.DataFrame:
    col = next((c for c in ['primary_purpose', 'main_purpose', 'purpose']
                if c in tours.columns), None)
    if col is None:
        logger.warning("_filter_seg: no purpose column found")
        return tours.iloc[0:0]   # empty

    vals = tours[col].fillna('')

    if segment == 'work':
        mask = vals.isin(_WORK_PURPOSES)
    elif segment == 'edu':
        mask = vals.isin(_EDU_PURPOSES)
    else:
        mask = vals.isin(_NONMAND_PURPOSES)

    result = tours[mask].copy()

    if len(result) == 0:
        # diagnostic - show what values ARE present
        unique_vals = vals.value_counts().head(10).to_dict()
        logger.warning(
            f"_filter_seg: segment='{segment}' matched 0 tours. "
            f"Top purpose values in col='{col}': {unique_vals}"
        )

    return result


def _case_otaz_from_dt(dt_d_batch, fallback_rows: pd.DataFrame = None) -> np.ndarray:
    """Extract case-level origin zones aligned to the destination cases."""
    try:
        tour_df = dt_d_batch['tour'].to_dataframe().reset_index()
        case_col = next((c for c in ['ptour_id', 'index'] if c in tour_df.columns), None)
        otaz_col = next((c for c in ['OTAZ', 'otaz'] if c in tour_df.columns), None)
        if case_col and otaz_col:
            ordered = tour_df.sort_values(case_col)
            return pd.to_numeric(ordered[otaz_col], errors='coerce').fillna(0).astype(int).to_numpy()
    except Exception:
        pass
    if fallback_rows is not None and 'OTAZ' in fallback_rows.columns:
        return pd.to_numeric(fallback_rows['OTAZ'], errors='coerce').fillna(0).astype(int).to_numpy()
    return np.array([], dtype=int)


def _fit_destination_od_calibration(obs_rows: pd.DataFrame,
                                    probs: np.ndarray,
                                    zone_ids,
                                    alpha: float = 0.6,
                                    eps: float = 1e-6,
                                    clip_bias: float = 3.0) -> dict:
    """Compute origin-specific OD calibration biases from observed vs fitted shares."""
    try:
        if probs is None or len(obs_rows) == 0:
            return {}
        df = obs_rows.copy().reset_index(drop=True)
        if len(df) != probs.shape[0]:
            return {}
        df['OTAZ'] = pd.to_numeric(df['OTAZ'], errors='coerce').fillna(0).astype(int)
        df['DTAZ'] = pd.to_numeric(df['DTAZ'], errors='coerce').fillna(0).astype(int)
        probs = np.asarray(probs, dtype=float)
        zone_arr = np.asarray(zone_ids, dtype=int)
        out = {}
        for otaz, grp in df.groupby('OTAZ', sort=False):
            idx = grp.index.to_numpy(dtype=int)
            if len(idx) == 0:
                continue
            obs_share = grp['DTAZ'].value_counts(normalize=True).reindex(zone_arr).fillna(0.0).to_numpy(dtype=float)
            pred_share = probs[idx].mean(axis=0)
            pred_share = np.where(np.isfinite(pred_share), pred_share, 0.0)
            pred_share = pred_share / max(pred_share.sum(), eps)
            bias = alpha * np.log((obs_share + eps) / (pred_share + eps))
            bias = np.clip(bias, -clip_bias, clip_bias)
            if np.any(np.abs(bias) > 1e-6):
                out[int(otaz)] = {'zones': zone_arr.tolist(), 'bias': bias.tolist()}
        return out
    except Exception as e:
        logger.debug(f"_fit_destination_od_calibration failed: {e}")
        return {}


def _fit_destination_zone_shadow_prices(obs_rows: pd.DataFrame,
                                        probs: np.ndarray,
                                        zone_ids,
                                        n_iter: int = 12,
                                        alpha: float = 0.6,
                                        clip_sp: float = 2.5,
                                        percent_tolerance: float = 5.0,
                                        target_threshold: float = 20.0,
                                        over_predict_factor: float = 5.0,
                                        eps: float = 1e-8,
                                        _history_out: list = None) -> np.ndarray:
    """
    Compute zone-level shadow prices (log corrections) so that predicted
    marginal destination counts match observed marginal counts.

    This is the ActivitySim shadow-price approach: iteratively adjust
    destination-zone ASCs so Σ_i P(j|i) ≈ observed_count_j for all j.

    Unlike OD calibration (which operates per origin zone and is sparse),
    shadow prices operate on destination marginals so ALL zones are adjusted,
    including zone 176 (over-predicted education) and zone 22 (over-predicted
    with 0 observed).

    Parameters
    ----------
    obs_rows  : observation DataFrame with 'DTAZ' column
    probs     : (n_obs, n_zones) predicted probability matrix from estimation
    zone_ids  : list/array of zone IDs corresponding to probs columns
    n_iter    : number of IPF-style iterations (5 is usually enough)
    alpha     : step damping (0-1); smaller = slower but more stable
    clip_sp   : max absolute shadow price (prevents extreme corrections)
    over_predict_factor : a zone below target_threshold observed trips is still
                    made eligible for (downward-only, via the ratio sign) correction
                    if its predicted share exceeds over_predict_factor times its
                    observed share (floored at a small nominal share so zones with
                    zero observed trips can be caught too). Without this, a zone
                    with few/no observed trips can never be pulled down no matter
                    how much predicted mass it absorbs - see zone 22 note above.
    eps       : small floor to avoid log(0)

    Returns
    -------
    shadow_prices : np.ndarray shape (n_zones,), one additive log-correction
                    per destination zone.  Add to utility before softmax.
    """
    try:
        if probs is None or len(obs_rows) == 0:
            return np.zeros(len(zone_ids), dtype=np.float32)
        zone_arr = np.asarray(zone_ids, dtype=int)
        n_zones = len(zone_arr)
        if probs.shape[1] != n_zones:
            return np.zeros(n_zones, dtype=np.float32)

        # Observed destination counts (marginal distribution over zones)
        _obs = obs_rows.copy().reset_index(drop=True)
        _obs['DTAZ'] = pd.to_numeric(_obs['DTAZ'], errors='coerce').fillna(-1).astype(int)
        if 'weight' in _obs.columns:
            _obs['weight'] = pd.to_numeric(_obs['weight'], errors='coerce').fillna(1.0).clip(lower=0.0)
        else:
            _obs['weight'] = 1.0
        obs_marginal = (
            _obs.groupby('DTAZ')['weight'].sum().reindex(zone_arr).fillna(0.0).to_numpy(dtype=np.float64)
        )
        obs_total = obs_marginal.sum()
        if obs_total <= 0:
            return np.zeros(n_zones, dtype=np.float32)

        probs_base = np.asarray(probs, dtype=np.float64)
        probs_base = np.where(np.isfinite(probs_base) & (probs_base >= 0), probs_base, 0.0)
        row_sum = probs_base.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        probs_base = probs_base / row_sum
        if probs_base.shape[0] == 0:
            return np.zeros(n_zones, dtype=np.float32)

        # Normalise observed to shares for stable ratio updates.
        obs_share = obs_marginal / obs_total

        # ActivitySim-style iterative adjustment with per-case renormalization.
        # ARC-model shadow pricing: log(obs_share/pred_share) ratio updates,
        # iterated until all eligible zones are within tolerance.
        shadow = np.zeros(n_zones, dtype=np.float64)
        tol = max(0.0, float(percent_tolerance)) / 100.0
        target_threshold = max(0.0, float(target_threshold))
        _last_fail = None
        _iter_history = []   # ARC-style convergence table rows

        for _it in range(n_iter):
            # Row-wise adjusted probabilities after adding zone shadow prices.
            log_u = np.log(probs_base + eps) + shadow[np.newaxis, :]
            row_max = np.max(log_u, axis=1, keepdims=True)
            row_max = np.where(np.isfinite(row_max), row_max, 0.0)
            exp_u = np.exp(np.clip(log_u - row_max, -500, 500))
            denom = exp_u.sum(axis=1, keepdims=True)
            denom = np.where(denom > 0, denom, 1.0)
            probs_adj = exp_u / denom

            pred_counts = probs_adj.sum(axis=0)
            pred_total = pred_counts.sum()
            if pred_total <= 0:
                break
            pred_share = pred_counts / pred_total

            ratio = (obs_share + eps) / (pred_share + eps)
            # Zones with enough observed trips are always eligible (both
            # under- and over-prediction get corrected). Zones below that
            # threshold are ALSO eligible, but only for downward correction,
            # when predicted share is way out of line with observed share -
            # otherwise a zone with 0-2 observed trips can never be pulled
            # down no matter how much predicted mass lands on it (see the
            # zone-22 case this function's docstring already assumes it handles).
            obs_share_floor = np.maximum(obs_share, 1.0 / (n_zones * 20.0))
            over_predicted = pred_share > (over_predict_factor * obs_share_floor)
            eligible = (obs_marginal >= target_threshold) | over_predicted
            failing = eligible & (np.abs(1.0 - ratio) > tol)
            _last_fail = int(failing.sum())
            _max_ratio_err = float(np.max(np.abs(1.0 - ratio[eligible]))) if eligible.any() else 0.0
            _max_sp = float(np.abs(shadow).max())

            _iter_history.append((_it + 1, _last_fail, _max_ratio_err, _max_sp))

            correction = np.zeros_like(shadow)
            correction[failing] = alpha * np.log(ratio[failing])
            shadow += correction
            shadow = np.clip(shadow, -clip_sp, clip_sp)

            if _last_fail == 0:
                logger.info(
                    f"_fit_destination_zone_shadow_prices: converged at iter {_it + 1}/{n_iter}"
                )
                break

        # ── ARC-style convergence summary table ────────────────────────────
        # Print iteration log: iter | fail_zones | max_ratio_err% | max_shadow
        if _iter_history:
            _hdr = (f"  {'Iter':>5}  {'FailZones':>10}  {'MaxRatioErr%':>13}  {'MaxShadow':>10}")
            logger.info("  Shadow price convergence table (ARC-style):")
            logger.info(_hdr)
            # Print first, last 3, and any iteration where fail_zones halved
            _prev_fail = _iter_history[0][1] + 1
            for _row in _iter_history:
                _it_n, _nf, _mre, _msp = _row
                _is_milestone = (
                    _it_n == 1
                    or _it_n == len(_iter_history)
                    or _nf == 0
                    or (_prev_fail > 0 and _nf < _prev_fail * 0.5)
                    or _it_n <= 3
                )
                if _is_milestone:
                    _conv_flag = ' ← converged' if _nf == 0 else ''
                    logger.info(
                        f"  {_it_n:>5}  {_nf:>10}  {_mre*100:>12.1f}%  {_msp:>10.3f}{_conv_flag}"
                    )
                _prev_fail = _nf

        n_nonzero = int((np.abs(shadow) > 0.01).sum())
        logger.info(
            f"_fit_destination_zone_shadow_prices: {n_nonzero}/{n_zones} zones "
            f"adjusted; max_abs={float(np.abs(shadow).max()):.3f}; "
            f"remaining_fail_zones={_last_fail if _last_fail is not None else 'NA'}"
        )
        # Log which zones get large corrections so over/under-prediction is visible
        top_adj = np.argsort(np.abs(shadow))[-5:][::-1]
        for _zi in top_adj:
            if abs(shadow[_zi]) > 0.1:
                _obs_n = int(obs_marginal[_zi])
                _pred_n = float(pred_share[_zi] * obs_total) if 'pred_share' in locals() else 0.0
                logger.info(
                    f"  zone {zone_arr[_zi]}: obs={_obs_n}, "
                    f"pred={_pred_n:.1f}, shadow={shadow[_zi]:.3f}"
                )
        # Export iteration history to caller for plotting
        if _history_out is not None and isinstance(_history_out, list):
            _history_out.extend(_iter_history)
        return shadow.astype(np.float32)
    except Exception as e:
        logger.debug(f"_fit_destination_zone_shadow_prices failed: {e}")
        return np.zeros(len(zone_ids), dtype=np.float32)


def _plot_shadow_price_convergence(iter_history: list, segment: str, save_path: str) -> None:
    """
    ARC-model-style shadow pricing convergence chart.

    Shows failing zones and max shadow price per iteration, mirroring the
    ABMVIZ convergence diagnostic used in the ARC Travel Demand Model.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if not iter_history:
            return
        iters     = [r[0] for r in iter_history]
        fail_z    = [r[1] for r in iter_history]
        max_ratio = [r[2] * 100 for r in iter_history]
        max_sp    = [r[3] for r in iter_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        fig.suptitle(
            f'Shadow Price Convergence - {segment.upper()} Destination Model\n'
            f'(ARC-style; converged={fail_z[-1]==0})',
            fontsize=11, fontweight='bold',
        )

        ax1.plot(iters, fail_z, 'o-', color='#E84855', linewidth=1.5, markersize=4)
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('Failing zones\n(|obs/pred−1|>tol)', fontsize=9)
        ax1.set_ylim(bottom=0)
        ax1.grid(axis='y', alpha=0.3)
        ax1.fill_between(iters, fail_z, alpha=0.15, color='#E84855')

        ax2.plot(iters, max_sp, 's-', color='#2E86AB', linewidth=1.5, markersize=4,
                 label='Max |shadow|')
        ax2_r = ax2.twinx()
        ax2_r.plot(iters, max_ratio, '^--', color='#F4A261', linewidth=1.2, markersize=3,
                   label='Max ratio err %')
        ax2.set_xlabel('Iteration', fontsize=9)
        ax2.set_ylabel('Max |shadow price|', fontsize=9, color='#2E86AB')
        ax2_r.set_ylabel('Max ratio error (%)', fontsize=9, color='#F4A261')
        ax2.grid(axis='y', alpha=0.3)
        lines1, lbl1 = ax2.get_legend_handles_labels()
        lines2, lbl2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc='upper right')

        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Shadow price convergence plot → {save_path}")
    except Exception as _pe:
        logger.debug(f"_plot_shadow_price_convergence failed: {_pe}")


def _plot_shadow_price_validation(
    obs_rows: 'pd.DataFrame',
    probs_base: np.ndarray,
    shadow_prices: np.ndarray,
    zone_ids,
    landuse: 'pd.DataFrame',
    segment: str,
    save_path: str,
) -> None:
    """
    ARC-style obs vs predicted zone-share scatter plot.

    Two panels:
      Left:  before shadow prices - reveals systematic model bias
      Right: after shadow prices  - shows how well IPF correction worked

    Each point is a destination zone; size = employment/enrolment; colour = ratio.
    Zones with ratio > 2× or < 0.5× are labelled with their zone number.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        zone_arr = np.asarray(zone_ids, dtype=int)
        n_zones = len(zone_arr)
        eps = 1e-8

        # Observed marginal counts
        _obs = obs_rows.copy()
        _obs['DTAZ'] = pd.to_numeric(_obs.get('DTAZ', _obs.get('dtaz', pd.Series(dtype=int))),
                                     errors='coerce').fillna(-1).astype(int)
        obs_counts = np.zeros(n_zones)
        for _zi, _z in enumerate(zone_arr):
            obs_counts[_zi] = (_obs['DTAZ'] == _z).sum()
        obs_total = obs_counts.sum()
        if obs_total <= 0:
            return
        obs_share = obs_counts / obs_total

        def _pred_share_from_probs(p, sp=None):
            if sp is not None:
                log_u = np.log(p + eps) + sp[np.newaxis, :]
                row_max = np.max(log_u, axis=1, keepdims=True)
                exp_u = np.exp(np.clip(log_u - row_max, -500, 500))
                denom = exp_u.sum(axis=1, keepdims=True)
                denom = np.where(denom > 0, denom, 1.0)
                p_adj = exp_u / denom
            else:
                p_adj = p
            counts = p_adj.sum(axis=0)
            return counts / max(counts.sum(), eps)

        pred_before = _pred_share_from_probs(probs_base, None)
        pred_after  = _pred_share_from_probs(probs_base, shadow_prices)

        # Zone employment/enrolment for bubble size
        emp_col = {'work': 'TOTEMP', 'edu': 'Education', 'nonmand': 'RETEMPN'}.get(segment, 'TOTEMP')
        lu_idx = landuse.set_index(
            next((c for c in ['TAZ', 'taz', 'zone', 'ZONE'] if c in landuse.columns), landuse.columns[0])
        ) if not landuse.empty else pd.DataFrame()
        emp_vals = np.array([
            float(lu_idx.loc[z, emp_col]) if (not lu_idx.empty and z in lu_idx.index
                                               and emp_col in lu_idx.columns) else 1.0
            for z in zone_arr
        ], dtype=float)
        emp_vals = np.clip(emp_vals, 1.0, None)
        bubble = 5 + 60 * (emp_vals / emp_vals.max())

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle(
            f'Destination Zone Validation - {segment.upper()}  (n={int(obs_total)} HHTS obs)\n'
            f'Bubble size ∝ {emp_col}; red = over-predicted, blue = under-predicted',
            fontsize=11, fontweight='bold',
        )

        for ax, pred, title in [
            (axes[0], pred_before, 'Before shadow prices'),
            (axes[1], pred_after,  'After shadow prices'),
        ]:
            ratio = (pred + eps) / (obs_share + eps)
            # Colour by log ratio: blue=under, white=matched, red=over
            log_ratio = np.log2(ratio)
            norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
            cmap = plt.cm.RdBu_r

            sc = ax.scatter(
                obs_share * 100, pred * 100,
                s=bubble, c=log_ratio, cmap=cmap, norm=norm,
                alpha=0.7, edgecolors='grey', linewidths=0.3,
            )
            _max_ax = max(obs_share.max(), pred.max()) * 100 * 1.05
            ax.plot([0, _max_ax], [0, _max_ax], 'k--', linewidth=0.8, label='Perfect fit')
            ax.plot([0, _max_ax], [0, _max_ax * 2], ':', color='salmon', linewidth=0.7, label='2× over')
            ax.plot([0, _max_ax], [0, _max_ax * 0.5], ':', color='cornflowerblue', linewidth=0.7, label='2× under')
            ax.set_xlabel('Observed share (%)', fontsize=9)
            ax.set_ylabel('Predicted share (%)', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7, loc='upper left')
            # Label outlier zones
            for _zi, _z in enumerate(zone_arr):
                if obs_share[_zi] > 0.005 or pred[_zi] > 0.005:
                    _r = ratio[_zi]
                    if _r > 2.5 or _r < 0.4:
                        ax.annotate(
                            str(_z),
                            (obs_share[_zi] * 100, pred[_zi] * 100),
                            fontsize=6, ha='center', va='bottom',
                            color='darkred' if _r > 2.5 else 'darkblue',
                        )
            plt.colorbar(sc, ax=ax, label='log₂(pred/obs)', shrink=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Shadow price validation scatter → {save_path}")
    except Exception as _pe:
        logger.debug(f"_plot_shadow_price_validation failed: {_pe}")


def _apply_destination_calibration(V: np.ndarray,
                                   case_otaz: np.ndarray,
                                   zone_ids,
                                   calibration: dict) -> np.ndarray:
    """Add origin-specific calibration biases to a utility matrix."""
    if not calibration:
        return V
    try:
        zone_arr = np.asarray(zone_ids, dtype=int)
        for i, otaz in enumerate(np.asarray(case_otaz, dtype=int)):
            cal = calibration.get(int(otaz))
            if not cal:
                continue
            cal_zones = np.asarray(cal.get('zones', []), dtype=int)
            cal_bias = np.asarray(cal.get('bias', []), dtype=float)
            if len(cal_zones) != len(cal_bias) or len(cal_zones) == 0:
                continue
            bias_map = pd.Series(cal_bias, index=cal_zones)
            V[i, :] += bias_map.reindex(zone_arr).fillna(0.0).to_numpy(dtype=float)
    except Exception as e:
        logger.debug(f"_apply_destination_calibration failed: {e}")
    return V


def _ensure_destination_clusters(landuse: pd.DataFrame) -> pd.DataFrame:
    """
    Add a stable macro-destination cluster for hierarchical destination sampling.
    Prefer SA2_MAIN16 when available; otherwise fall back to city or coarse bins.
    """
    lu = landuse.copy()
    zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]
    if 'dest_cluster' in lu.columns:
        return lu
    if 'SA2_MAIN16' in lu.columns:
        lu['dest_cluster'] = lu['SA2_MAIN16'].astype(str)
    elif 'city' in lu.columns:
        lu['dest_cluster'] = lu['city'].astype(str)
    else:
        ranks = pd.to_numeric(lu[zone_col], errors='coerce').fillna(0).astype(int)
        lu['dest_cluster'] = ('cluster_' + ((ranks - 1).clip(lower=0) // 20).astype(str))
    return lu


def _fit_destination_cluster_targets(obs_rows: pd.DataFrame,
                                     landuse: pd.DataFrame) -> dict:
    """Observed macro-destination shares used for residual-balancing during simulation."""
    try:
        if obs_rows is None or len(obs_rows) == 0:
            return {}
        lu = _ensure_destination_clusters(landuse)
        zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]
        cluster_map = lu[[zone_col, 'dest_cluster']].copy()
        cluster_map[zone_col] = pd.to_numeric(cluster_map[zone_col], errors='coerce').fillna(-1).astype(int)
        df = obs_rows.copy()
        df['DTAZ'] = pd.to_numeric(df['DTAZ'], errors='coerce').fillna(-1).astype(int)
        df = df.merge(cluster_map, left_on='DTAZ', right_on=zone_col, how='left')
        shares = df['dest_cluster'].fillna('unknown').value_counts(normalize=True)
        return shares.to_dict()
    except Exception as e:
        logger.debug(f"_fit_destination_cluster_targets failed: {e}")
        return {}


def _sample_destinations_hierarchical(probs_np: np.ndarray,
                                      zone_ids,
                                      landuse: pd.DataFrame,
                                      rng: np.random.Generator,
                                      cluster_targets: dict = None,
                                      residual_strength: float = 0.7) -> np.ndarray:
    """
    Hierarchical cluster→zone sampler with residual balancing.
    Uses macro destination clusters to stabilize raw zone probabilities.
    """
    zone_arr = np.asarray(zone_ids, dtype=int)
    if probs_np is None or probs_np.size == 0 or len(zone_arr) == 0:
        return np.array([], dtype=int)

    lu = _ensure_destination_clusters(landuse)
    zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]
    zone_to_cluster = (
        lu[[zone_col, 'dest_cluster']]
        .drop_duplicates()
        .assign(**{zone_col: lambda d: pd.to_numeric(d[zone_col], errors='coerce').fillna(-1).astype(int)})
        .set_index(zone_col)['dest_cluster']
    )
    cluster_labels = zone_to_cluster.reindex(zone_arr).fillna('unknown').astype(str).values
    unique_clusters = pd.Index(pd.unique(cluster_labels))
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    zone_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_labels], dtype=int)
    n_cases = probs_np.shape[0]

    cluster_probs = np.zeros((n_cases, len(unique_clusters)), dtype=float)
    for j in range(len(zone_arr)):
        cluster_probs[:, zone_cluster_idx[j]] += probs_np[:, j]

    desired_counts = None
    if cluster_targets:
        target_series = pd.Series(cluster_targets, dtype=float).reindex(unique_clusters).fillna(0.0)
        if target_series.sum() > 0:
            target_series = target_series / target_series.sum()
            desired_counts = target_series.values * n_cases

    chosen = np.full(n_cases, -1, dtype=int)
    remaining = desired_counts.copy() if desired_counts is not None else None
    order = rng.permutation(n_cases)

    for ii in order:
        cp = np.asarray(cluster_probs[ii], dtype=float)
        cp = np.where(np.isfinite(cp) & (cp >= 0), cp, 0.0)
        if remaining is not None:
            ratio = (remaining + 1e-6) / (desired_counts + 1e-6)
            cp = cp * np.power(np.clip(ratio, 0.05, 20.0), residual_strength)
        if cp.sum() <= 0:
            # Cluster probs degenerate - fall back to zone-level probabilities
            zp = np.asarray(probs_np[ii], dtype=float)
            zp = np.where(np.isfinite(zp) & (zp >= 0), zp, 0.0)
            if zp.sum() <= 0:
                # Last resort: uniform random over all zones (never zone_arr[0])
                chosen[ii] = int(rng.choice(zone_arr))
                continue
            zp = zp / zp.sum()
            chosen[ii] = int(zone_arr[rng.choice(len(zone_arr), p=zp)])
            continue
        cp = cp / cp.sum()
        cidx = int(rng.choice(len(unique_clusters), p=cp))
        if remaining is not None:
            remaining[cidx] = max(0.0, remaining[cidx] - 1.0)

        zmask = (zone_cluster_idx == cidx)
        zp = np.asarray(probs_np[ii, zmask], dtype=float)
        zp = np.where(np.isfinite(zp) & (zp >= 0), zp, 0.0)
        zones_in_cluster = zone_arr[zmask]
        if zp.sum() <= 0:
            chosen[ii] = int(rng.choice(zones_in_cluster))
        else:
            zp = zp / zp.sum()
            chosen[ii] = int(rng.choice(zones_in_cluster, p=zp))

    return chosen


def _purpose_to_seg(p: str) -> str:
    if p == 'work': return 'work'
    if p == 'edu':  return 'edu'
    return 'nonmand'


# ===========================================================================
# LARCH DATATREES
# ===========================================================================

def _lu_idx(landuse: pd.DataFrame) -> pd.DataFrame:
    col = 'ORDER' if 'ORDER' in landuse.columns else landuse.columns[0]
    lu = landuse.set_index(col)
    # Compute TOTEMP if not present (sum of sectoral employment)
    if 'TOTEMP' not in lu.columns:
        _emp_cols = ['FPSEMPN', 'HEREMPN', 'RETEMPN', 'MWTEMPN', 'OTHEMP', 'AGREMPN']
        _available = [c for c in _emp_cols if c in lu.columns]
        if _available:
            lu['TOTEMP'] = sum(
                pd.to_numeric(lu[c], errors='coerce').fillna(0.0) for c in _available
            )
            logger.info(f"_lu_idx: computed TOTEMP from {_available} "
                        f"(mean={lu['TOTEMP'].mean():.0f}, max={lu['TOTEMP'].max():.0f})")
    return lu


# Columns the new behavioural UEC terms reference on the `person` sub-dataset,
# together with safe default values if the column is absent.
# These exist in the enriched synth pop (enrich_synth_population) and in most
# HHTS persons tables; the defaults ensure estimation doesn't crash if a column
# is missing from a particular data source.
_PERSON_BEHAVIOURAL_DEFAULTS = {
    'age5p':             6,     # 5-yr age band 1-18; 6 ≈ 25-29yr (working-age default)
    'AGEGROUP':          6,
    'CARLICENCE':        1,     # assume licensed (conservative - avoids killing DA utility)
    'occp_group':        0,     # 0 = not working
    'occp_mgr':          0,     # binary: professional/managerial
    'occp_trades':       0,     # binary: trades/technical
    'bikes':             0,     # no bikes in household
    'nprd':              2,     # household size 2
    'sexp':              1,     # male (neutral default)
    'WORKSTAT':          0,     # not working (kept for travel_propensity and legacy expressions)
    'STUDYING':          0,
    # WORKSTAT binary dummies (derived in _ensure_person_behavioural_cols from WORKSTAT
    # if not already present via cdap.build_person_profiles). Preferred over raw WORKSTAT in
    # mode/dest UEC expressions because treating WORKSTAT as continuous implies a meaningless
    # linear ordering 0→1→2 that conflates "not working" with "part-time" magnitudes.
    'workstat_fulltime':  0,    # WORKSTAT == 1
    'workstat_parttime':  0,    # WORKSTAT == 2
    'workstat_notworking':1,    # WORKSTAT == 0  (default: assume not working for safety)
    # STUDYING binary dummies (same rationale as WORKSTAT)
    'studying_none':         1,
    'studying_primary':      0,
    'studying_secondary':    0,
    'studying_tertiary_pt':  0,
    'studying_tertiary_ft':  0,
    'vehrd':             1,     # 1 vehicle (most common in SEQ)
    # Micromobility ownership flags referenced by MODE_UEC c_ebike_bike /
    # c_escooter_bike (GA-gated on {'ebikes','escooter'}). Absent from both
    # HHTS and synth persons in current data drops; without a default the
    # sharrow flow compile dies (job 23971368: KeyError 'person..ebikes').
    # Zero-filled → the term is constant and the coefficient simply never
    # moves (flagged by the NEVER-MOVED warning) instead of crashing stage 5.
    'ebikes':            0,
    'escooter':          0,
    # Habitual mode-usage propensity from mobility_models (MODE_UEC
    # c_transit_use_transit / c_car_use_da / c_bike_use_bike, GA-gated on the
    # matching gene). Zero-filled when the pre-day mobility module didn't run, so
    # the term is constant and its coefficient simply never moves (never crashes).
    'transit_prop':      0.0,
    'car_prop':          0.0,
    'bike_prop':         0.0,
    'incp':              8,     # mid income band
    'indp':              19,    # not employed industry
    'travel_propensity': 0.0,   # CDAP logsum proxy; computed by _travel_propensity_score
    'safety_sensitivity': 0.5,  # per-agent crash-aversion; computed by _safety_sensitivity_score
    'person_dist_draw':  0.0,   # frozen random distance sensitivity offset (set at prediction)
}


def _travel_propensity_score(pp: pd.DataFrame) -> pd.Series:
    """
    Compute a normalised travel propensity score per person as a CDAP logsum proxy.

    The score captures "how much utility does this person get from travelling?"
    - the same concept as the CDAP logsum used in ActivitySim's work location
    model, built from attributes available in both HHTS and synthetic population.

    Interpretation of the score (after normalisation to ~N(0,1)):
      +ve → mandatory travellers (FT workers, university students, car-rich,
            licensed, high-income, working-age) - less deterred by distance
      -ve → home-stayers / captive short-distance travellers (retired,
            car-free, unlicensed, seniors, industrial/construction workers
            whose employment is site-local)

    Used in DEST_UEC as:
        c_propensity_dist * travel_propensity_i * log(dist_ij + 1)
    - a positive coefficient means more-mobile people choose farther destinations.
    Coefficients estimated from HHTS (see _build_dest_utility / DEST_UEC).

    Attributes used (all present in both HHTS and SEQ synthetic population):
        WORKSTAT  : 0=not working, 1=FT, 2=PT
        STUDYING  : 0=none, 1=primary, 2=secondary, 3-4=post-secondary
        vehrd     : vehicles in household (≥1 → positive)
        CARLICENCE: 1=licensed (positive)
        incp      : income band 1-14 (higher → positive)
        age5p     : 5-year age band (seniors/children → negative)
        indp      : industry code - professional/public → positive; industrial → negative
    """
    n = len(pp)
    score = np.zeros(n, dtype=np.float64)

    def _col(name, default):
        return pd.to_numeric(pp.get(name, pd.Series(default, index=pp.index)),
                             errors='coerce').fillna(default).values

    # ── Mandatory travel obligation (largest predictor) ────────────────────
    workstat = _col('WORKSTAT', 0.0)
    score += np.where(workstat == 1, 1.20,    # FT worker
             np.where(workstat == 2, 0.65, 0.0))  # PT worker

    studying = _col('STUDYING', 0.0).astype(int)
    score += np.where(np.isin(studying, [3, 4]), 0.85,  # university (post-secondary)
             np.where(np.isin(studying, [1, 2]), 0.35, 0.0))  # school

    # ── Vehicle access (physical ability to reach distant destinations) ────
    vehrd = _col('vehrd', 1.0)
    lic   = _col('CARLICENCE', 1.0)
    score += np.where(vehrd > 0, 0.40, -0.35)  # car-free restricts range
    score += np.where(lic   > 0, 0.30, -0.20)  # unlicensed → limited range

    # ── Income (higher income → can commute farther / accept travel costs) ─
    incp = _col('incp', 7.0)
    score += np.clip((incp - 7.0) / 7.0, -0.50, 0.60)  # centre at band 7

    # ── Age (working-age peak; seniors and young children travel less far) ─
    age5p = _col('age5p', 5.0)
    score += np.where(age5p >= 9, -0.55,          # 65+ seniors
             np.where(age5p >= 8, -0.25,           # 55-64
             np.where(age5p <= 2, -0.30, 0.0)))    # children under 10

    # ── Industry (professional/public → CBD destinations → larger range;
    #    industrial/construction → local/suburban work sites → smaller range)
    indp = _col('indp', 20.0).astype(int)
    score += np.where(np.isin(indp, [13, 14, 15, 16]),  0.35,   # professional/public/admin/edu
             np.where(np.isin(indp, [3, 4, 5]),         -0.25,  # manufacturing/utilities/construction
             np.where(np.isin(indp, [1, 2]),             -0.40, 0.0)))  # agri/mining (site-remote)

    # ── Normalise to ~N(0,1) across the population ────────────────────────
    mu, sd = float(score.mean()), float(score.std())
    if sd > 1e-6:
        score = (score - mu) / sd

    return pd.Series(score, index=pp.index, name='travel_propensity')


def _safety_sensitivity_score(pp: pd.DataFrame) -> pd.Series:
    """
    Per-agent SAFETY UTILITY: how crash-averse this person is, from their
    profile. Vulnerable road users (children, seniors, women, car-free /
    pedestrian-cyclist exposed, low-income) weight road-safety more heavily.

    Returned on a 0..1 scale (0.5 = population-average sensitivity) so it can be
    read directly as a multiplier on the crash disutility. Linked into mode /
    destination choice via the gated crash-interaction terms
    ``c_crash_safety`` (mode, walk/bike) and ``c_crash_safety_dest`` (dest):
    a safety-sensitive agent gets a stronger negative crash coefficient, so they
    avoid crash-prone active routes and high-crash destination zones more.

    Attributes (all present in HHTS + SEQ synthetic population):
        age5p     : children (≤10) and seniors (65+) are most vulnerable
        sexp      : 2=female (QTS/road-safety evidence: higher perceived risk)
        vehrd     : car-free → more walking/cycling exposure → more sensitive
        CARLICENCE: unlicensed → more non-car exposure
        incp      : lower income → less able to avoid unsafe routes → sensitive
        nprd/hh   : presence of children in household raises safety weighting
    """
    n = len(pp)

    def _col(name, default):
        return pd.to_numeric(pp.get(name, pd.Series(default, index=pp.index)),
                             errors='coerce').fillna(default).values

    s = np.full(n, 0.50, dtype=np.float64)   # population-average baseline
    age5p = _col('age5p', 5.0)
    s += np.where(age5p <= 2, 0.20,           # young children
         np.where(age5p >= 9, 0.18, 0.0))     # seniors 65+
    s += np.where(_col('sexp', 1.0) == 2, 0.10, 0.0)     # female
    s += np.where(_col('vehrd', 1.0) <= 0, 0.15, 0.0)    # car-free (exposed)
    s += np.where(_col('CARLICENCE', 1.0) <= 0, 0.06, 0.0)  # unlicensed
    incp = _col('incp', 7.0)
    s += np.clip((7.0 - incp) / 14.0, 0.0, 0.12)          # lower income → more sensitive
    s += np.where(_col('hh_child', 0.0) > 0, 0.08, 0.0)   # children in household

    return pd.Series(np.clip(s, 0.0, 1.0), index=pp.index, name='safety_sensitivity')


def _ensure_person_behavioural_cols(persons: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that every column referenced by the behavioural UEC terms exists
    on the persons table, filling missing ones with sensible defaults.

    Also computes ``travel_propensity`` (CDAP logsum proxy) if not already
    present - this feeds the ``c_propensity_dist`` destination interaction term.

    Called from _mode_tree and _dest_tree before building the DataTree so that
    sharrow compilation never encounters a missing column reference.
    """
    pp = persons.copy()
    _had_upper_carlicence = 'CARLICENCE' in pp.columns
    for col, default in _PERSON_BEHAVIOURAL_DEFAULTS.items():
        if col not in pp.columns:
            pp[col] = default
            logger.debug(f"_ensure_person_behavioural_cols: added '{col}'={default}")
        else:
            pp[col] = pd.to_numeric(pp[col], errors='coerce').fillna(default)

    # TODO(CARLICENCE): keep this lowercase synth-pop bridge before any future
    # availability refactor; DA availability depends on the embedded DataTree
    # exposing uppercase CARLICENCE.
    # Handle lowercase variant of CARLICENCE from the synthetic population
    # (which stores it as 'carlicence').  If uppercase was absent originally,
    # overwrite the default added above with the real lowercase values.
    if 'carlicence' in pp.columns and not _had_upper_carlicence:
        pp['CARLICENCE'] = pd.to_numeric(pp['carlicence'], errors='coerce').fillna(1)

    # ── Travel propensity (CDAP logsum proxy) ─────────────────────────────
    # Always recompute from source attributes so it reflects the actual person
    # table rather than a stale default.  Very cheap (vectorized numpy).
    pp['travel_propensity'] = _travel_propensity_score(pp).values

    # ── Safety utility / crash-aversion per agent (from profile) ──────────
    # Feeds the gated crash-interaction terms c_crash_safety (mode) and
    # c_crash_safety_dest (dest); recompute from source each time.
    pp['safety_sensitivity'] = _safety_sensitivity_score(pp).values

    # ── WORKSTAT binary dummies ────────────────────────────────────────────
    # Derive workstat_fulltime / workstat_parttime / workstat_notworking from
    # the raw WORKSTAT code when they are not already present (e.g. data that
    # was not routed through cdap.build_person_profiles).  Identical mapping to
    # the one in build_person_profiles so estimation and prediction are consistent.
    if 'workstat_fulltime' not in pp.columns and 'WORKSTAT' in pp.columns:
        _ws = pd.to_numeric(pp['WORKSTAT'], errors='coerce').fillna(0).astype(int)
        pp['workstat_fulltime']  = (_ws == 1).astype(np.int8)
        pp['workstat_parttime']  = (_ws == 2).astype(np.int8)
        pp['workstat_notworking']= (_ws == 0).astype(np.int8)

    # ── STUDYING binary dummies ────────────────────────────────────────────
    if 'studying_none' not in pp.columns and 'STUDYING' in pp.columns:
        _st = pd.to_numeric(pp['STUDYING'], errors='coerce').fillna(0).astype(int)
        pp['studying_none']        = (_st == 0).astype(np.int8)
        pp['studying_primary']     = (_st == 1).astype(np.int8)
        pp['studying_secondary']   = (_st == 2).astype(np.int8)
        pp['studying_tertiary_pt'] = (_st == 3).astype(np.int8)
        pp['studying_tertiary_ft'] = (_st == 4).astype(np.int8)

    # ── Occupation sub-group dummies ───────────────────────────────────────
    # Derived from indp (industry/occupation code) if available.
    # SEQ mapping: professional/managerial = indp 10-14; trades/technical = indp 3-6.
    # Falls back to 0 (not applicable) if indp is absent.
    if 'indp' in pp.columns:
        _indp = pd.to_numeric(pp['indp'], errors='coerce').fillna(0).astype(int)
        pp['occp_mgr']    = ((_indp >= 10) & (_indp <= 14)).astype(np.int8)
        pp['occp_trades'] = ((_indp >= 3)  & (_indp <= 6)).astype(np.int8)
    else:
        if 'occp_mgr' not in pp.columns:
            pp['occp_mgr'] = np.int8(0)
        if 'occp_trades' not in pp.columns:
            pp['occp_trades'] = np.int8(0)

    return pp


def _mode_tree(tours, persons, landuse, skims, skim_df: pd.DataFrame = None):
    """
    Build DataTree for mode choice.

    Larch requires the ROOT dataset to be idca (case × alternative) so it can
    find the '_altid_' index. We build a minimal idca root from the mode
    alternatives, then attach tours (idco), persons, landuse, and od skims
    as sub-datasets.
    """
    import xarray as xr

    # Ensure ptour_id is unique before building the DataTree.
    # Larch's from_idca raises "cannot convert a DataFrame with a non-unique
    # MultiIndex into xarray" when case_ids repeat (e.g. after _filter_seg
    # on HHTS data where tour_id resets per person).
    if tours['ptour_id'].duplicated().any():
        n_dup = tours['ptour_id'].duplicated().sum()
        logger.warning(
            f"_mode_tree: {n_dup} duplicate ptour_ids - "
            "reassigning row-index-based unique ids"
        )
        tours = tours.copy()
        tours['ptour_id'] = (
            tours['person_id'].astype(str) + '_'
            + np.arange(len(tours)).astype(str)
        )

    alt_ids   = list(MODE_MAP.values())           # [1, 2, 3, 4, 5]
    case_ids  = list(tours['ptour_id'].values)

    # ── root dataset: case × alt, all zeros (utility built from co + od) ──
    # This establishes both the caseid (_caseid_) and altid (_altid_) dims
    # that Larch requires before it can initialise a Model.
    ca_index = pd.MultiIndex.from_product(
        [case_ids, alt_ids], names=['ptour_id', '_altid_'])
    root_df  = pd.DataFrame({'_dummy': np.zeros(len(ca_index))}, index=ca_index)
    root_ds  = lx.Dataset.construct.from_idca(root_df)

    # ── tours idco dataset (case-level attributes) ──────────────────────
    tour_ds = lx.Dataset.construct.from_idco(tours.set_index('ptour_id'))

    # ── od skims ─────────────────────────────────────────────────────────
    od = _build_od_dataset(skims, skim_df=skim_df)

    # ── persons ──────────────────────────────────────────────────────────
    pp = persons.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})
    pp = _ensure_person_behavioural_cols(pp)

    # MODE_UEC references emp.is_cbd / emp.transit_walk_acc; the dest path
    # derives them via _attach_size_term but landuse arrives here raw. When a
    # GA-selected feature set gates in c_cbd_* / c_transit_access, sharrow
    # fails flow compilation on the missing column (job 23971354:
    # KeyError 'emp..is_cbd' in [edu] _build_mode_model).
    lu_emp = _lu_idx(_ensure_cbd_and_transit_cols(landuse))

    # sharrow DataTree: ALL datasets must be keyword arguments.
    # The first dataset passed becomes the root node.
    # We name it 'root' explicitly; relationships then use 'root.' to reference it.
    return lx.DataTree(
        base   = root_ds,
        tour   = tour_ds,
        person = pp.set_index('person_id'),
        emp    = lu_emp,
        od     = od,
        relationships=(
            "base.ptour_id @ tour.ptour_id",
            "tour.person_id @ person.person_id",
            "tour.DTAZ @ emp.ORDER",
            "tour.OTAZ @ od.otaz",
            "tour.DTAZ @ od.dtaz",
        ),
    )


def _dest_tree(tours, persons, landuse, skims, logsums, sampled_alts=None,
               skim_df: pd.DataFrame = None, crash_skims=None,
               segment: str = '', hp_df=None):
    """
    Build a FLAT Larch Dataset for destination choice estimation.

    Follows the ActivitySim pattern (activitysim.estimation.larch.location_choice):
    all OD, landuse, and person columns are pre-joined into the idca DataFrame
    BEFORE calling lx.Dataset.construct.from_idca().  This avoids Larch 6.x
    DataTree relationship-path resolution bugs where compound expressions like
    'od.DIST', 'emp.size_term' cause parameter discovery to fail silently,
    leaving model.pf empty (0 parameters).

    The returned object is a merged lx.Dataset (NOT a DataTree), but we attach
    helper attributes (_od_vars_cached, _flat_mode=True) so downstream code
    can detect the flat layout.

    logsums is an xr.DataArray with dims ('cases', 'zones') from _logsums().

    Parameters
    ----------
    sampled_alts : pd.DataFrame or None
        If provided (HHTS estimation), must have columns
        ['ptour_id', 'alt_zone', 'log_correction'].
        If None, all zones are used as alternatives and log_correction = 0.
    """
    import xarray as xr

    # get zone IDs from the logsum array (these are the destination alternatives)
    if hasattr(logsums, 'coords') and 'zones' in logsums.coords:
        zone_ids = [int(z) for z in logsums.coords['zones'].values]
    else:
        try:
            mapping = dict(skims.mapping(skims.listMappings()[0]))
            zone_ids = sorted([int(k) for k in mapping.keys()])
        except Exception:
            zone_ids = _default_zone_universe()

    if hasattr(logsums, 'values'):
        ls_vals  = logsums.values
        case_ids = [str(c) for c in logsums.coords['cases'].values]
    else:
        ls_vals  = np.array(logsums)
        case_ids = list(tours['ptour_id'].astype(str))

    n_cases = len(case_ids)
    n_zones = len(zone_ids)

    # ── Build OD skim data ───────────────────────────────────────────────
    od = _build_od_dataset(skims, skim_df=skim_df)
    _od_vars = _dataset_var_names(od)

    if 'DIST' not in _od_vars:
        logger.warning(
            "_dest_tree: DIST not found in od Dataset "
            f"(available: {sorted(_od_vars)[:10]}). "
            "Destination distance expressions will fail."
        )
    else:
        try:
            _dist_vals = od['DIST'].values
            _dist_fin = _dist_vals[np.isfinite(_dist_vals)]
            if _dist_fin.size:
                logger.info(
                    "_dest_tree: DIST available "
                    f"(min={float(np.min(_dist_fin)):.3f}, "
                    f"p50={float(np.quantile(_dist_fin, 0.50)):.3f}, "
                    f"p95={float(np.quantile(_dist_fin, 0.95)):.3f}, "
                    f"max={float(np.max(_dist_fin)):.3f})"
                )
        except Exception as _d_e:
            logger.debug(f"_dest_tree: DIST diagnostic failed ({_d_e})")

    # ── CRASH matrix ─────────────────────────────────────────────────────
    if crash_skims is not None or (skim_df is not None and 'CRASH' in skim_df.columns):
        try:
            _zones = [int(z) for z in list(od.coords['otaz'].values)]
            _z_to_i = {z: i for i, z in enumerate(_zones)}
            _mat = np.zeros((len(_zones), len(_zones)), dtype=np.float32)

            if crash_skims is not None:
                _crash_df = _read_crash_od_table(crash_skims, zone_ids=_zones)
            else:
                _crash_df = skim_df[['OTAZ', 'DTAZ', 'CRASH']].copy()

            for _o, _d, _c in _crash_df[['OTAZ', 'DTAZ', 'CRASH']].itertuples(index=False, name=None):
                _oi = _z_to_i.get(int(_o))
                _di = _z_to_i.get(int(_d))
                if _oi is not None and _di is not None:
                    _mat[_oi, _di] = float(_c)

            _da = xr.DataArray(
                _mat, dims=['otaz', 'dtaz'],
                coords={'otaz': _zones, 'dtaz': _zones},
            )
            od = od.assign(CRASH=_da)
            logger.info("_dest_tree: CRASH matrix loaded into od dataset")
        except Exception as _ce:
            logger.warning(f"_dest_tree: failed to load CRASH values ({_ce}) - "
                           "c_crash will use zero values")
    _cached_od_vars = _dataset_var_names(od)

    # ── Build OD lookup dict for fast join ────────────────────────────────
    _od_zones = [int(z) for z in od.coords['otaz'].values]
    _zone_to_oidx = {z: i for i, z in enumerate(_od_zones)}
    _od_dtaz = [int(z) for z in od.coords['dtaz'].values]
    _zone_to_didx = {z: i for i, z in enumerate(_od_dtaz)}

    # Prepare persons
    pp = persons.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})
    pp = _ensure_person_behavioural_cols(pp)
    pp_idx = pp.set_index('person_id')
    if not pp_idx.index.is_unique:
        pp_idx = pp_idx[~pp_idx.index.duplicated(keep='first')]
    # Ensure person_id index is str to match the str-coerced person_id from
    # _prep_hhts_tours (which runs t['person_id'].astype(str) before the merge).
    # Without this, _personids.map(_vals) returns NaN everywhere (int vs str
    # key mismatch), causing STUDYING/incp/etc to default to 0 - making
    # edu_prim_priv_cond all-zeros, which Larch then classifies as CO (1D),
    # triggering a Numba TypingError when the dim3_filler tries a 2D index.
    pp_idx.index = pp_idx.index.astype(str)

    # Prepare landuse
    lu_emp = _lu_idx(landuse)  # indexed by ORDER (zone_id)

    # ── Merge acdc_aggl/acdc_comp from saved_data/acdc_terms.csv ─────────
    # These are NOT stored in zone_order_and_landuse.csv; they live in a
    # separate file keyed by zone_id × segment.  We pivot to one row per
    # zone (averaging across segments) and join onto lu_emp so the
    # ca_df join below finds non-zero values.
    _acdc_path = os.path.join('saved_data', 'acdc_terms.csv')
    if not os.path.exists(_acdc_path):
        _acdc_path = os.path.join(os.path.dirname(__file__), 'saved_data', 'acdc_terms.csv')
    if os.path.exists(_acdc_path):
        try:
            _acdc = pd.read_csv(_acdc_path)
            # average aggl/comp across segments to get a single zone-level value
            _acdc_z = (
                _acdc.groupby('zone_id')[['acdc_aggl', 'acdc_comp']]
                .mean()
            )
            # Drop any pre-existing acdc columns before join to avoid
            # "columns overlap but no suffix specified" pandas error on re-runs.
            for _col in ['acdc_aggl', 'acdc_comp']:
                if _col in lu_emp.columns:
                    lu_emp = lu_emp.drop(columns=[_col])
            lu_emp = lu_emp.join(_acdc_z, how='left')
            lu_emp['acdc_aggl'] = pd.to_numeric(lu_emp['acdc_aggl'], errors='coerce').fillna(0.0)
            lu_emp['acdc_comp']  = pd.to_numeric(lu_emp['acdc_comp'],  errors='coerce').fillna(0.0)
            logger.info(
                f"_dest_tree: merged acdc_terms ({len(_acdc_z)} zones) - "
                f"aggl range [{lu_emp['acdc_aggl'].min():.3f}, {lu_emp['acdc_aggl'].max():.3f}]"
            )
        except Exception as _acdc_e:
            logger.error(
                f"_dest_tree: failed to load acdc_terms.csv ({_acdc_e})\n"
                "  ► c_aggl will be ZERO for all zones - agglomeration term disabled.\n"
                "  ► This silently removes zone 40's main CBD attractor.\n"
                "  ► Fix: run _attach_acdc_terms() / compute_accessibility before stage 5."
            )
    else:
        logger.error(
            "_dest_tree: saved_data/acdc_terms.csv NOT FOUND - c_aggl will be ZERO.\n"
            "  ► Agglomeration term disabled for all zones - destination model will be flat.\n"
            "  ► Fix: ensure _attach_acdc_terms() writes saved_data/acdc_terms.csv before stage 5."
        )

    # Prepare tour → person mapping
    tour_df = tours.copy()
    if 'ptour_id' not in tour_df.columns:
        tour_df = tour_df.reset_index()
    tour_df['ptour_id'] = tour_df['ptour_id'].astype(str)
    tour_df = tour_df.drop_duplicates('ptour_id').set_index('ptour_id')

    # ── build ca dataset with pre-joined columns ─────────────────────────
    # ActivitySim pattern: merge ALL data into ca_df BEFORE from_idca().
    if sampled_alts is not None and not sampled_alts.empty:
        case_to_row = {cid: r for r, cid in enumerate(case_ids)}
        zone_to_col = {z: c for c, z in enumerate(zone_ids)}

        ca_rows = []
        for _, row in sampled_alts.iterrows():
            pid  = str(row['ptour_id'])
            zone = int(row['alt_zone'])
            r    = case_to_row.get(pid)
            c    = zone_to_col.get(zone)
            ls_val = float(ls_vals[r, c]) if (r is not None and c is not None) else 0.0
            ca_rows.append({
                'ptour_id':       pid,
                '_altid_':        zone,
                'logsum':         ls_val,
                'log_correction': float(row['log_correction']),
            })
        ca_df = pd.DataFrame(ca_rows)
        logger.info(
            f"_dest_tree: importance-sampled CA table "
            f"{n_cases} cases × up to {DEST_EST_SAMPLE_SIZE} alts "
            f"= {len(ca_df)} rows"
        )
    else:
        ptour_rep = np.repeat(case_ids, n_zones)
        altid_rep = np.tile(zone_ids, n_cases)
        ca_df = pd.DataFrame({
            'ptour_id':       ptour_rep,
            '_altid_':        altid_rep,
            'logsum':         ls_vals.ravel(),
            'log_correction': 0.0,
        })
        print(f"  _dest_tree: ALL-ZONES CA table {n_cases} cases × {n_zones} zones = {len(ca_df)} rows")

    # ── Pre-join OD skim columns into ca_df (keyed by OTAZ × DTAZ) ───────
    # Get OTAZ for each case from tour_df
    _pid_to_otaz = {}
    if 'OTAZ' in tour_df.columns:
        for _pid in tour_df.index:
            try:
                _pid_to_otaz[str(_pid)] = int(tour_df.loc[_pid, 'OTAZ'])
            except Exception:
                pass
    elif 'otaz' in tour_df.columns:
        for _pid in tour_df.index:
            try:
                _pid_to_otaz[str(_pid)] = int(tour_df.loc[_pid, 'otaz'])
            except Exception:
                pass

    # Join DIST and CRASH from OD dataset
    _otaz_arr = ca_df['ptour_id'].map(_pid_to_otaz).fillna(0).astype(int).values
    _dtaz_arr = ca_df['_altid_'].values.astype(int)
    _n_ca = len(ca_df)

    # Vectorized OD join using numpy indexing
    _od_dist_mat = np.asarray(od['DIST'].values, dtype=np.float32) if 'DIST' in _cached_od_vars else None
    _od_crash_mat = np.asarray(od['CRASH'].values, dtype=np.float32) if 'CRASH' in _cached_od_vars else None

    _dist_vals_ca = np.full(_n_ca, 30.0, dtype=np.float32)
    _crash_vals_ca = np.zeros(_n_ca, dtype=np.float32)
    # Vectorized OD join: pd.Index.get_indexer is C-level O(n), avoids Python loop
    _od_zones_pidx = pd.Index(_od_zones)
    _od_dtaz_pidx = pd.Index(_od_dtaz)
    _oi_vec = _od_zones_pidx.get_indexer(_otaz_arr)   # -1 for unknown zones
    _di_vec = _od_dtaz_pidx.get_indexer(_dtaz_arr)    # -1 for unknown zones
    _valid_od_vec = (_oi_vec >= 0) & (_di_vec >= 0)
    _oi_safe_vec = np.where(_valid_od_vec, _oi_vec, 0)
    _di_safe_vec = np.where(_valid_od_vec, _di_vec, 0)
    if _od_dist_mat is not None:
        _dist_vals_ca = np.where(_valid_od_vec, _od_dist_mat[_oi_safe_vec, _di_safe_vec], 30.0).astype(np.float32)
    if _od_crash_mat is not None:
        _crash_vals_ca = np.where(_valid_od_vec, _od_crash_mat[_oi_safe_vec, _di_safe_vec], 0.0).astype(np.float32)

    ca_df['DIST'] = _dist_vals_ca
    ca_df['CRASH'] = _crash_vals_ca
    # Pre-compute log transforms used in utility expressions
    ca_df['log_DIST_1'] = np.log(np.clip(ca_df['DIST'].values, 0.0, None) + 1.0).astype(np.float32)
    ca_df['log_CRASH_1'] = np.log(np.clip(ca_df['CRASH'].values, 0.0, None) + 1.0).astype(np.float32)

    # ── Piecewise-linear distance spline (ActivitySim distance-band pattern) ─
    # A continuous linear spline in km with knots at 2 and 5 km, so the GA can
    # fit a distance deterrent whose slope differs by band instead of a single
    # linear c_dist. Mutually exclusive with c_dist (see ga_constraints). Each
    # segment is the km travelled WITHIN that band, so summing the three columns
    # reconstructs DIST — the coefficients are per-km deterrents by band.
    _dist_km = np.clip(ca_df['DIST'].values, 0.0, None)
    ca_df['dist_0_2']  = np.minimum(_dist_km, 2.0).astype(np.float32)
    ca_df['dist_2_5']  = np.clip(_dist_km - 2.0, 0.0, 3.0).astype(np.float32)
    ca_df['dist_5_10'] = np.clip(_dist_km - 5.0, 0.0, 5.0).astype(np.float32)
    # ── Exponential distance-decay (gravity kernel) ──────────────────────────
    # Non-linear proximity: exp(-DIST/D0). ~1 nearby, decaying with distance.
    # D0 is DATA-DRIVEN from the actual observed distance distribution so it
    # adapts to whatever zone system is in use (329, 371, 507, …) — a hardcoded
    # scale from a different/smaller system makes the column ~0 everywhere (dead,
    # always-"stuck"). Fast = half the median trip distance, slow = 1.5× it.
    # Env overrides still win if set.
    _pos_dist = _dist_km[_dist_km > 0]
    _dist_med = float(np.median(_pos_dist)) if _pos_dist.size else 15.0
    _env_f = os.environ.get('GA_DIST_DECAY_FAST', '').strip()
    _env_s = os.environ.get('GA_DIST_DECAY_SLOW', '').strip()
    _d0f = float(_env_f) if _env_f else max(_dist_med * 0.5, 1.0)
    _d0s = float(_env_s) if _env_s else max(_dist_med * 1.5, 3.0)
    ca_df['dist_decay']      = np.exp(-_dist_km / max(_d0f, 0.1)).astype(np.float32)
    ca_df['dist_decay_slow'] = np.exp(-_dist_km / max(_d0s, 0.1)).astype(np.float32)

    # ── Pre-join DISTWALK, DISTBIKE, SOV_TIME from OD skims ──────────────
    # These enable walk/bike distance-based destination utility and time-based
    # travel impedance rather than relying solely on network distance.
    _od_distwalk_mat = np.asarray(od['DISTWALK'].values, dtype=np.float32) if 'DISTWALK' in _cached_od_vars else None
    _od_distbike_mat = np.asarray(od['DISTBIKE'].values, dtype=np.float32) if 'DISTBIKE' in _cached_od_vars else None
    _od_sovtime_mat = np.asarray(od['SOVTOLL_TIME__AM'].values, dtype=np.float32) if 'SOVTOLL_TIME__AM' in _cached_od_vars else None

    _distwalk_ca = np.full(_n_ca, 30.0, dtype=np.float32)
    _distbike_ca = np.full(_n_ca, 30.0, dtype=np.float32)
    _sovtime_ca = np.full(_n_ca, 30.0, dtype=np.float32)
    # Reuse vectorized indices from DIST/CRASH join above (_oi_safe_vec/_di_safe_vec/_valid_od_vec)
    if _od_distwalk_mat is not None:
        _distwalk_ca = np.where(_valid_od_vec, _od_distwalk_mat[_oi_safe_vec, _di_safe_vec], 30.0).astype(np.float32)
    if _od_distbike_mat is not None:
        _distbike_ca = np.where(_valid_od_vec, _od_distbike_mat[_oi_safe_vec, _di_safe_vec], 30.0).astype(np.float32)
    if _od_sovtime_mat is not None:
        _sovtime_ca = np.where(_valid_od_vec, _od_sovtime_mat[_oi_safe_vec, _di_safe_vec], 30.0).astype(np.float32)

    ca_df['DISTWALK'] = _distwalk_ca
    ca_df['DISTBIKE'] = _distbike_ca
    ca_df['SOVTOLL_TIME'] = _sovtime_ca
    ca_df['log_DISTWALK_1'] = np.log(np.clip(ca_df['DISTWALK'].values, 0.0, None) + 1.0).astype(np.float32)
    ca_df['log_DISTBIKE_1'] = np.log(np.clip(ca_df['DISTBIKE'].values, 0.0, None) + 1.0).astype(np.float32)
    ca_df['log_SOVTIME_1'] = np.log(np.clip(ca_df['SOVTOLL_TIME'].values, 0.0, None) + 1.0).astype(np.float32)
    # Intrazonal dummy: same origin and destination zone
    ca_df['intrazonal'] = (ca_df['_altid_'].values.astype(int) == _otaz_arr.astype(int)).astype(np.float32)
    # Walk/bike feasibility flags for destination utility
    ca_df['walk_feasible'] = (ca_df['DISTWALK'] < WALK_MAX_KM).astype(np.float32)
    ca_df['bike_feasible'] = (ca_df['DISTBIKE'] < BIKE_MAX_KM).astype(np.float32)

    # ── Pre-join landuse (emp) columns into ca_df (keyed by DTAZ=zone) ───
    _lu_cols_to_join = [
        'size_term', 'acdc_aggl', 'acdc_comp',
        'acdc_aggl_retail', 'acdc_aggl_office', 'acdc_aggl_edu', 'acdc_aggl_leisure',
        'acdc_aggl_parks', 'acdc_aggl_health', 'acdc_aggl_food', 'acdc_aggl_civic',
        'is_cbd', 'transit_walk_acc',
        'auto_work_acc', 'auto_shop_acc', 'auto_edu_acc', 'auto_nonmand_acc',
        'walk_acc', 'transit_acc',
        'FPSEMPN', 'HEREMPN', 'RETEMPN', 'TOTEMP',
        'AGREMPN', 'MWTEMPN', 'construction', 'government', 'governmental',
        'industrial',
        'Education', 'Primary', 'Secondary',
        'residential', 'Entertainment', 'commercial', 'retail', 'EATOUT',
        'HSENROLL', 'AGE0519', 'Primary_total_priv', 'Secondary_total_priv',
        'TOTPOP',
    ]
    for _col in _lu_cols_to_join:
        if _col in lu_emp.columns:
            _mapping = pd.to_numeric(lu_emp[_col], errors='coerce').fillna(0.0)
            ca_df[_col] = ca_df['_altid_'].map(_mapping).fillna(0.0).astype(np.float32)
        else:
            ca_df[_col] = np.float32(0.0)

    # ── Pre-join person columns into ca_df (keyed by ptour_id → person_id) ──
    _pid_to_personid = {}
    _person_id_col = next((c for c in ['person_id', 'PERSID'] if c in tour_df.columns), None)
    if _person_id_col:
        for _pid in tour_df.index:
            try:
                # Coerce to str so _personids.map(_vals) matches the
                # str-indexed pp_idx (fixed above) on both sides.
                _pid_to_personid[str(_pid)] = str(tour_df.loc[_pid, _person_id_col])
            except Exception:
                pass

    _person_cols_to_join = [
        'incp', 'age5p', 'sexp', 'nprd', 'vehrd', 'occp_group',
        'occp_mgr', 'occp_trades', 'indp', 'STUDYING', 'WORKSTAT',
        'travel_propensity', 'safety_sensitivity', 'person_dist_draw',
    ]
    _personids = ca_df['ptour_id'].map(_pid_to_personid)
    for _col in _person_cols_to_join:
        _default = _PERSON_BEHAVIOURAL_DEFAULTS.get(_col, 0.0)
        if _col in pp_idx.columns:
            _vals = pd.to_numeric(pp_idx[_col], errors='coerce').fillna(_default)
            ca_df[_col] = _personids.map(_vals).fillna(_default).astype(np.float32)
        else:
            ca_df[_col] = np.float32(_default)

    # ── ptype × land-attribute interaction genes (constraint overlay) ────────
    # person's ptype dummy × destination zone's land attractor. Joins each
    # side's base column on demand, then multiplies. Guarded: a missing ptype or
    # land column yields an all-zero interaction (dropped later as a dead term).
    for _it in _register_overlay_dest_interactions():
        _pt, _la, _fc = _it['ptype'], _it['land'], _it['flat_col']
        if _pt not in ca_df.columns:
            if _pt in pp_idx.columns:
                _pv = pd.to_numeric(pp_idx[_pt], errors='coerce').fillna(0.0)
                ca_df[_pt] = _personids.map(_pv).fillna(0.0).astype(np.float32)
            else:
                ca_df[_pt] = np.float32(0.0)
        if _la not in ca_df.columns:
            if _la in lu_emp.columns:
                _lv = pd.to_numeric(lu_emp[_la], errors='coerce').fillna(0.0)
                ca_df[_la] = ca_df['_altid_'].map(_lv).fillna(0.0).astype(np.float32)
            else:
                ca_df[_la] = np.float32(0.0)
        ca_df[_fc] = (ca_df[_pt].astype(np.float32)
                      * ca_df[_la].astype(np.float32)).astype(np.float32)

    # ── Pre-join tour-level columns (OTAZ, DTAZ for choice) ──────────────
    ca_df['OTAZ'] = _otaz_arr.astype(np.float32)
    if 'DTAZ' in tour_df.columns:
        _pid_to_dtaz = {}
        for _pid in tour_df.index:
            try:
                _pid_to_dtaz[str(_pid)] = int(tour_df.loc[_pid, 'DTAZ'])
            except Exception:
                pass
        ca_df['DTAZ'] = ca_df['ptour_id'].map(_pid_to_dtaz).fillna(0).astype(np.float32)
    elif 'dtaz' in tour_df.columns:
        _pid_to_dtaz = {}
        for _pid in tour_df.index:
            try:
                _pid_to_dtaz[str(_pid)] = int(tour_df.loc[_pid, 'dtaz'])
            except Exception:
                pass
        ca_df['DTAZ'] = ca_df['ptour_id'].map(_pid_to_dtaz).fillna(0).astype(np.float32)

    # ── Pre-join tour's own purpose string (for nonmand sub-purpose size term) ──
    # Native tour attribute (not a person attribute, so not in _person_cols_to_join
    # above) - needed to give each nonmand tour its own shopping/social/escort/
    # othmaint/othdiscr/eat size term rather than one blended across all of them.
    _purpose_col_name = next((c for c in ['primary_purpose', 'main_purpose', 'purpose']
                              if c in tour_df.columns), None)
    if _purpose_col_name:
        _pid_to_purpose = {str(_pid): str(tour_df.loc[_pid, _purpose_col_name])
                           for _pid in tour_df.index}
        ca_df['_purpose_'] = ca_df['ptour_id'].map(_pid_to_purpose).fillna('')
    else:
        ca_df['_purpose_'] = ''

    # ── Pre-compute interaction terms used in DEST_UEC ───────────────────
    ca_df['incp_x_log_dist'] = (ca_df['incp'] * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['age5p_x_log_dist'] = (ca_df['age5p'] * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['vehrd_x_log_dist'] = (ca_df['vehrd'] * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['female_x_log_dist'] = ((ca_df['sexp'] == 2).astype(np.float32) * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['nprd_x_log_dist'] = (ca_df['nprd'] * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['occp_group_x_log_dist'] = (ca_df['occp_group'] * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['occp_mgr_x_log_dist'] = ((ca_df['occp_mgr'] == 1).astype(np.float32) * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['occp_trades_x_log_dist'] = ((ca_df['occp_trades'] == 1).astype(np.float32) * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['WORKSTAT_x_log_dist'] = (ca_df['WORKSTAT'] * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['propensity_x_log_dist'] = (ca_df['travel_propensity'] * ca_df['log_DIST_1']).astype(np.float32)
    ca_df['person_dist_draw_x_log_dist'] = (ca_df['person_dist_draw'] * ca_df['log_DIST_1']).astype(np.float32)
    # Per-agent safety utility × destination-zone crash exposure (c_crash_safety_dest):
    # safety-sensitive agents avoid high-crash destination zones more strongly.
    if 'safety_sensitivity' in ca_df.columns:
        ca_df['safety_x_crash'] = (ca_df['safety_sensitivity'].astype(np.float32)
                                   * ca_df['log_CRASH_1'].astype(np.float32)).astype(np.float32)
    # Accessibility interactions with no-vehicle dummy
    _noveh = (ca_df['vehrd'] <= 0).astype(np.float32)
    ca_df['noveh_walk_acc'] = (_noveh * ca_df['walk_acc']).astype(np.float32)
    ca_df['noveh_transit_acc'] = (_noveh * ca_df['transit_acc']).astype(np.float32)
    # Income × work accessibility
    ca_df['incp_x_auto_work_acc'] = (ca_df['incp'] * ca_df['auto_work_acc']).astype(np.float32)
    # Industry conditioning dummies
    _indp = ca_df['indp'].astype(int)
    ca_df['indp_finance_flag'] = ((_indp == 10) | (_indp == 11) | (_indp == 12) | (_indp == 13)).astype(np.float32)
    ca_df['indp_health_flag'] = ((_indp == 16) | (_indp == 17)).astype(np.float32)
    ca_df['indp_retail_flag'] = ((_indp == 7) | (_indp == 8)).astype(np.float32)
    ca_df['FPSEMPN_cond'] = (ca_df['indp_finance_flag'] * ca_df['FPSEMPN']).astype(np.float32)
    ca_df['HEREMPN_cond'] = (ca_df['indp_health_flag'] * ca_df['HEREMPN']).astype(np.float32)
    ca_df['RETEMPN_cond'] = (ca_df['indp_retail_flag'] * ca_df['RETEMPN']).astype(np.float32)
    # ── Log-transform raw employment to prevent gradient saturation ───────
    # Raw TOTEMP/FPSEMPN can be 10,000-50,000. With e.g. c_totemp=0.15 that
    # gives utility~7,500 for zone 40 → softmax saturates (p≈1) → gradient≈0
    # → optimizer can't move → c_totemp drifts negative → zone 40 LOSES.
    # Using log(x+1) keeps all values in ~[0,11] range, coefficients ~0.3-1.0.
    ca_df['log_TOTEMP_1'] = np.log(ca_df['TOTEMP'].astype(np.float64) + 1.0).astype(np.float32)
    ca_df['log_FPSEMPN_cond'] = (ca_df['indp_finance_flag'] *
                                   np.log(ca_df['FPSEMPN'].astype(np.float64) + 1.0)).astype(np.float32)
    ca_df['log_HEREMPN_cond'] = (ca_df['indp_health_flag'] *
                                   np.log(ca_df['HEREMPN'].astype(np.float64) + 1.0)).astype(np.float32)
    ca_df['log_RETEMPN_cond'] = (ca_df['indp_retail_flag'] *
                                   np.log(ca_df['RETEMPN'].astype(np.float64) + 1.0)).astype(np.float32)
    # Study conditioning
    _study = ca_df['STUDYING'].astype(int).copy()
    # Reclassify STUDYING=0 edu-trip makers who are adult-age (age5p >= 4, ≈18+) as
    # post-secondary students (STUDYING=3).  In the HHTS some TAFE/Uni students were
    # coded STUDYING=0 despite reporting full-time TAFE/Uni as their main activity.
    # This ensures zones with tertiary facilities (e.g. CBD, zone 40) receive a
    # non-zero edu_post_cond signal for these travellers.
    if segment == 'edu':
        _adult_edu = (_study == 0) & (ca_df['age5p'].fillna(0) >= 4)
        if _adult_edu.any():
            _study = _study.copy()
            _study[_adult_edu] = 3
            logger.debug(
                f'_dest_tree [edu]: reclassified {int(_adult_edu.sum())} '
                f'STUDYING=0 adult cases to STUDYING=3 (post-secondary)'
            )
    # ── Log-transform edu attraction variables to prevent gradient saturation ──
    # Raw Primary/Secondary enrolment counts range 0-3,068 (zone 277).
    # With starting c_edu_primary=0.40, raw → utility=1,227 → exp(1227) overflows
    # softmax → every non-top-zone chooser gets log(≈0) → LL ≈ -3.5M.
    # log(x+1) keeps values in [0, ~8.0], utilities in [0, ~3.2] - tractable.
    # DEST_UEC_FLAT column names are kept identical so no downstream changes needed.
    _edu_prim_raw  = ((_study == 1).astype(np.float64) *
                      ca_df['Primary'].clip(lower=0).astype(np.float64))
    _edu_sec_raw   = ((_study == 2).astype(np.float64) *
                      ca_df['Secondary'].clip(lower=0).astype(np.float64))
    _edu_post_raw  = (((_study == 3) | (_study == 4)).astype(np.float64) *
                      ca_df['Education'].fillna(0).clip(lower=0).astype(np.float64))
    ca_df['edu_primary_cond']   = np.log(_edu_prim_raw  + 1.0).astype(np.float32)
    ca_df['edu_secondary_cond'] = np.log(_edu_sec_raw   + 1.0).astype(np.float32)
    ca_df['edu_post_cond']      = np.log(_edu_post_raw  + 1.0).astype(np.float32)
    # Education enrolment total: log of combined enrolment across all levels
    _edu_total_raw = (ca_df['Primary'].clip(lower=0) + ca_df['Secondary'].clip(lower=0) +
                      ca_df['Education'].fillna(0).clip(lower=0) +
                      ca_df['HSENROLL'].clip(lower=0)).astype(np.float64)
    ca_df['edu_enrol_total'] = np.log(_edu_total_raw + 1.0).astype(np.float32)
    # edu_primary_walk, edu_secondary_walk, edu_primary_logwalk,
    # edu_secondary_logwalk removed: these were mode-distance interactions
    # (study_level × log(DISTWALK) / study_level × walkable × school_count)
    # that double-count with c_logsum.  Mode impedance is fully captured by
    # the mode logsum; c_edu_primary and c_edu_secondary handle zone attraction.

    # ── Study-level-conditioned size term for edu (replaces combined) ─────
    # The combined edu size_term (Primary+Secondary+Education+...) does NOT
    # distinguish study levels: secondary students get attracted to primary-
    # school zones and vice versa, causing the negative Pearson r observed in
    # the Level 4 secondary validation.  Replace with a per-person conditioned
    # size term so each study group is attracted to their own school type:
    #   STUDYING==1 (primary)   → log(Primary_j + TOTPOP_j×0.05 + ε)
    #   STUDYING==2 (secondary) → log(Secondary_j + HSENROLL_j + ε)
    #   STUDYING>=3 (post-sec)  → log(Education_j^0.9 + university_j + college_j + ε)
    # The combined size_term column in ca_df is overwritten in-place so no
    # changes to DEST_UEC_FLAT or coefficient names are required.
    if segment == 'edu':
        _zone_col_lu_st = (
            'ORDER' if 'ORDER' in lu_emp.columns
            else (lu_emp.index.name if lu_emp.index.name else lu_emp.columns[0])
        )
        _lu_st = (
            lu_emp.set_index(_zone_col_lu_st)
            if lu_emp.index.name != _zone_col_lu_st
            else lu_emp
        )
        def _lu_s(col, default=0.0):
            if col in _lu_st.columns:
                return pd.to_numeric(_lu_st[col], errors='coerce').fillna(default).clip(lower=0)
            return pd.Series(default, index=_lu_st.index, dtype=np.float64)
        _eps_st = 1e-6
        _prim_log_st  = np.log(_lu_s('Primary') + _lu_s('TOTPOP') * 0.05 + _eps_st)
        _prim_log_st -= float(_prim_log_st.mean())
        _sec_log_st   = np.log(_lu_s('Secondary') + _lu_s('HSENROLL') + _eps_st)
        _sec_log_st  -= float(_sec_log_st.mean())
        _post_raw_st  = (_lu_s('Education').clip(lower=0) ** 0.9
                         + _lu_s('university') + _lu_s('college') + _eps_st)
        _post_log_st  = np.log(_post_raw_st)
        _post_log_st -= float(_post_log_st.mean())
        _prim_map_st  = _prim_log_st.to_dict()
        _sec_map_st   = _sec_log_st.to_dict()
        _post_map_st  = _post_log_st.to_dict()
        _altid_st = ca_df['_altid_'].values
        _study_st = _study.values if hasattr(_study, 'values') else np.asarray(_study)
        _size_prim_st = np.array([_prim_map_st.get(int(z), 0.0) for z in _altid_st], dtype=np.float32)
        _size_sec_st  = np.array([_sec_map_st.get(int(z),  0.0) for z in _altid_st], dtype=np.float32)
        _size_post_st = np.array([_post_map_st.get(int(z), 0.0) for z in _altid_st], dtype=np.float32)
        ca_df['size_term'] = (
            (_study_st == 1).astype(np.float32) * _size_prim_st
            + (_study_st == 2).astype(np.float32) * _size_sec_st
            + (_study_st >= 3).astype(np.float32) * _size_post_st
        )
        logger.info(
            f"_dest_tree [edu]: size_term overridden with study-level-conditioned version; "
            f"prim_range=[{_prim_log_st.min():.2f},{_prim_log_st.max():.2f}] "
            f"sec_range=[{_sec_log_st.min():.2f},{_sec_log_st.max():.2f}] "
            f"post_range=[{_post_log_st.min():.2f},{_post_log_st.max():.2f}]"
        )

    # ── Sub-purpose-conditioned size term for nonmand (replaces blended) ──
    # SEGMENT_PURPOSES['nonmand'] = ['social','escort','shopping','othmaint',
    # 'othdiscr','eat'] - compute_size_terms() returns one column per
    # sub-purpose using its own attractors (Shopping/retail/RETEMPN for
    # shopping, Entertainment/sports_centre/fitness_centre for social, etc.,
    # see SIZE_TERM_COEFFICIENTS). Previously _attach_size_term() collapsed
    # these 6 columns to their ROW-WISE MEAN before the model ever saw them,
    # so a shopping trip and a social trip were attracted to the exact same
    # blended zone score - the granular sub-purpose attractors were computed
    # but never actually used per-tour. This overrides ca_df['size_term']
    # in-place (same mechanism as the edu study-level override above) so each
    # tour's OWN sub-purpose (from its native purpose/main_purpose/
    # primary_purpose string, mapped via _NONMAND_SUBPURPOSE_MAP) selects its
    # own size term; tours whose purpose doesn't map to one of the 6 (e.g. the
    # generic 'nonmand' segment-level label some upstream stages use) keep the
    # blended 'nonmand' composite as a fallback.
    if segment == 'nonmand':
        _subpurpose_keys = ['social', 'escort', 'shopping', 'othmaint', 'othdiscr', 'eat']
        _zone_col_lu_nm = (
            'ORDER' if 'ORDER' in lu_emp.columns
            else (lu_emp.index.name if lu_emp.index.name else lu_emp.columns[0])
        )
        _lu_nm = (
            lu_emp.set_index(_zone_col_lu_nm)
            if lu_emp.index.name != _zone_col_lu_nm
            else lu_emp
        )
        _st_by_purpose = compute_size_terms(_lu_nm.reset_index(), purposes=_subpurpose_keys + ['nonmand'])
        _st_by_purpose.index = _lu_nm.index
        _altid_nm = ca_df['_altid_'].values
        _case_subpurpose = ca_df['_purpose_'].map(_NONMAND_SUBPURPOSE_MAP)
        _fallback_size = ca_df['_altid_'].map(_st_by_purpose['nonmand']).fillna(0.0).astype(np.float32).values
        _new_size_term = _fallback_size.copy()
        _n_matched = 0
        for _key in _subpurpose_keys:
            _mask = (_case_subpurpose == _key).values
            if not _mask.any():
                continue
            _zone_map = _st_by_purpose[_key]
            _vals = ca_df['_altid_'].map(_zone_map).fillna(0.0).astype(np.float32).values
            _new_size_term[_mask] = _vals[_mask]
            _n_matched += int(_mask.sum())
        ca_df['size_term'] = _new_size_term
        logger.info(
            f"_dest_tree [nonmand]: size_term overridden with sub-purpose-conditioned "
            f"version; {_n_matched}/{len(ca_df)} rows matched a granular sub-purpose "
            f"(social/escort/shopping/othmaint/othdiscr/eat), rest kept the blended "
            f"'nonmand' fallback"
        )

    # Intrazonal × purpose conditioning
    ca_df['intrazonal_x_edu'] = (ca_df['intrazonal'] * (_study > 0).astype(np.float32)).astype(np.float32)
    # Travel time for work trips: c_log_sovtime_work in DEST_UEC references
    # log_SOVTIME_1 directly (same column as edu/nonmand), but the two
    # parameters are never active in the same segment so there is no collinearity.
    # The duplicate ca_df['log_sovtime_work'] column has been removed to avoid
    # the collinearity guard flagging it as a copy of log_SOVTIME_1.

    # ── Active-mode feasibility flags ────────────────────────────────────
    # NOTE: DISTWALK == DIST == DISTBIKE in this OMX (all road distance).
    # Circuity = road/euclidean = 1.0 always → log(1.0) = 0 → zero column.
    # Walk/bike distance log-transforms are perfectly collinear with log_DIST_1.
    # All circuity and walk/bike log-distance columns are therefore REMOVED from
    # the destination model.  Active-mode influence is captured by:
    #   • c_walk_feasible / c_bike_feasible  (short-distance dummy ≤6km / ≤25km)
    #   • c_logsum  (mode logsum already incorporates walk/bike accessibility)
    _wf = ca_df['walk_feasible'].values
    _bf = ca_df['bike_feasible'].values
    _active = ((_wf > 0) | (_bf > 0)).astype(np.float32)  # kept for possible future use
    # Full-time worker conditioning
    ca_df['ftwork_x_log_dist'] = ((ca_df['WORKSTAT'] == 1).astype(np.float32) * ca_df['log_DIST_1']).astype(np.float32)
    # Occupation conditioning
    ca_df['occp1_x_log_dist'] = ((ca_df['occp_group'] == 1).astype(np.float32) * ca_df['log_DIST_1']).astype(np.float32)

    # Log-commercial for work CBD attractor: zones with high commercial floor
    # area (like Brisbane CBD) attract disproportionate work trips.
    # Cap at p99 before logging to neutralise the 89M m² data error at zone 158,
    # which otherwise pushes c_log_commercial_work toward 0 during MLE and hides
    # the legitimate CBD signal for zone 40 (777k m², highest FPSEMPN in SEQ).
    _comm_raw = ca_df['commercial'].values.astype(np.float64)
    _comm_pos = _comm_raw[_comm_raw > 0]
    _comm_p99 = float(np.nanpercentile(_comm_pos, 99)) if len(_comm_pos) > 0 else 1.0
    _comm_cap = max(_comm_p99, 1.0)   # never cap below 1 m²
    ca_df['log_commercial_1'] = np.log(np.clip(_comm_raw, 0.0, _comm_cap) + 1.0).astype(np.float32)
    logger.debug(f"log_commercial_1: p99 cap={_comm_cap:.0f} - original max={_comm_raw.max():.0f}")

    # Log-transformed non-mandatory attractors (raw values span 0-3.6M → gradient saturation)
    ca_df['log_retail_1'] = np.log(ca_df['retail'].astype(np.float64) + 1.0).astype(np.float32)
    ca_df['log_entertainment_1'] = np.log(ca_df['Entertainment'].astype(np.float64) + 1.0).astype(np.float32)
    ca_df['log_TOTPOP_1'] = np.log(ca_df['TOTPOP'].astype(np.float64) + 1.0).astype(np.float32)
    ca_df['log_EATOUT_1'] = np.log(ca_df['EATOUT'].astype(np.float64) + 1.0).astype(np.float32)
    ca_df['log_residential_1'] = np.log(ca_df['residential'].astype(np.float64) + 1.0).astype(np.float32)
    # Additional industry-sector work attractors (log employment, work segment).
    # Government resolves from either 'government' or 'governmental' landuse col.
    _govt_src = ('government' if ca_df.get('government') is not None
                 and float(np.asarray(ca_df['government']).sum()) > 0 else 'governmental')
    for _col, _flat in (('AGREMPN', 'log_AGREMPN_1'), ('MWTEMPN', 'log_MWTEMPN_1'),
                        ('construction', 'log_construction_1'),
                        (_govt_src, 'log_government_1'),
                        ('industrial', 'log_industrial_1'),
                        ('Education', 'log_education_emp_1')):
        _src = ca_df[_col].astype(np.float64) if _col in ca_df.columns else 0.0
        ca_df[_flat] = np.log(np.asarray(_src, dtype=np.float64) + 1.0).astype(np.float32)
    # Private school terms: log-transformed, conditioned on study level
    ca_df['edu_prim_priv_cond'] = ((_study == 1).astype(np.float32) *
        np.log(ca_df['Primary_total_priv'].astype(np.float64) + 1.0)).astype(np.float32)
    ca_df['edu_sec_priv_cond'] = ((_study == 2).astype(np.float32) *
        np.log(ca_df['Secondary_total_priv'].astype(np.float64) + 1.0)).astype(np.float32)

    # Also carry a 'weight' column if the tour data has one
    if 'weight' in tour_df.columns:
        ca_df['weight'] = ca_df['ptour_id'].map(
            tour_df['weight'].to_dict()
        ).fillna(1.0).astype(np.float32)

    # override_choice is NOT added as a ca_df column: it is a copy of _altid_ and
    # triggers the collinearity guard.  Larch resolves choice from choice_co_code
    # ("DTAZ") without needing an explicit override_choice CA column.

    # ── Collinearity guard: detect near-identical columns and log them ────
    # Threshold = 0.999 (only flag truly identical columns like DIST==DISTWALK).
    # 0.995 was too aggressive - log(DIST+1) vs DIST have r≈0.99 and are both
    # legitimate model terms (log captures short-distance deterrence, linear
    # captures long-distance decay).  At 0.999 we only catch columns that are
    # numerically the same data stored twice (e.g. duplicate OMX matrices).
    #
    # Skim-pair protection: even if r=1.000, we only mark the DUPLICATE as
    # collinear; we NEVER mark skim log-transforms (log_DIST_1, log_SOVTIME_1)
    # as collinear with their raw counterparts.  These pairs are intentional
    # non-linear recodes that capture different parts of the utility curve.
    _SAFE_PAIRS = {
        # (col_a, col_b): col_b is a legitimate recode of col_a - never holdfast
        frozenset(['DIST',               'log_DIST_1']),
        frozenset(['SOVTOLL_TIME',       'log_SOVTIME_1']),
        frozenset(['DISTWALK',           'log_DISTWALK_1']),
        frozenset(['DISTBIKE',           'log_DISTBIKE_1']),
        # age5p × log_dist: interaction of age (continuous) with log_DIST_1.
        # In float32 over large N this appears r≈1.000 because age5p has low
        # variance - both terms are legitimate and should never be auto-holdfasted.
        frozenset(['age5p_x_log_dist',   'log_DIST_1']),
        # log_TOTEMP_1 vs size_term: size_term is a weighted sum that *includes*
        # TOTEMP, so they are highly correlated but not identical; both enter
        # different parts of the utility expression.
        frozenset(['log_TOTEMP_1',       'size_term']),
        # propensity × log_dist vs log_dist: same rational as age5p above.
        frozenset(['propensity_x_log_dist', 'log_DIST_1']),
        # occp interaction terms vs log_DIST_1
        frozenset(['occp1_x_log_dist',   'log_DIST_1']),
        frozenset(['ftwork_x_log_dist',  'log_DIST_1']),
        # intrazonal vs intrazonal_x_edu: in the edu segment, almost all
        # intrazonal tours are by students (study>0), so these two columns
        # are near-identical.  Both are meaningful independent terms - keep both.
        frozenset(['intrazonal',         'intrazonal_x_edu']),
        # acdc_aggl vs acdc_comp: both are meaningful ACDC terms (agglomeration and
        # competition) that capture different aspects of density. (Prior to the
        # 2026-07 compute_acdc_terms fix, acdc_comp was size-weighted with the
        # same kernel as acdc_aggl, making the pair ~0.9998 correlated as an
        # implementation artifact, not a genuine data property - comp is now
        # unweighted so the two should be more distinguishable. Kept in
        # _SAFE_PAIRS as a precaution in case residual correlation remains.)
        frozenset(['acdc_aggl',          'acdc_comp']),
        # Finance flag vs managerial occupation: indp_finance_flag (indp 10-13) and
        # occp_mgr (indp 10-14) are near-collinear in this SEQ sample because very
        # few observations have indp=14 only.  Both carry valid information; suppress
        # auto-holdfast and let the model handle via regularisation.
        frozenset(['indp_finance_flag',  'occp_mgr']),
    }
    # Primary columns that should NEVER be holdfasted by the generic collinearity
    # guard.  When a collinear pair contains one of these, always holdfast the
    # OTHER member of the pair.  This prevents e.g. DISTWALK == DIST from
    # holdfasting log_DIST_1 (the active distance deterrence term) instead of
    # log_DISTWALK_1 (the duplicate walk skim).
    _NEVER_HOLDFAST_CA_COLS = frozenset({
        'DIST',          # primary auto-network distance (raw)
        'log_DIST_1',    # primary log-distance deterrence coefficient source
        'acdc_aggl',     # agglomeration index - the primary ACDC term
        'size_term',     # locked at 1.0 everywhere
        'logsum',        # mode logsum - always used
        'intrazonal',    # intrazonal binary flag - protected from auto-holdfast
                         # (collinear with intrazonal_x_edu in edu segment by
                         # construction; listed in _SAFE_PAIRS above)
    })
    _collinear_ca_cols: set = set()
    _dist_walk_collinear = False
    try:
        _numeric_ca = ca_df.select_dtypes(include=[np.number])
        _ca_stds = _numeric_ca.std(axis=0)
        _nonzero_cols = _ca_stds[_ca_stds > 1e-8].index.tolist()

        # Holdfast all-zero CA interaction columns before Larch sees them.
        # When persons don't merge onto HHTS tours (e.g. synth person_ids passed
        # instead of HHTS persons), binary-flag x zone terms like female_x_log_dist
        # and edu_prim_priv_cond become all-zeros.  Larch then classifies them as
        # CO (1D) rather than CA (2D); the Numba dim3_filler later tries a 2D index
        # and crashes with TypingError.  Holdfasting prevents the misclassification.
        #
        # SAFETY-NET: Always holdfast person×zone interaction coefficients at 0
        # for estimation.  These are pre-computed data columns; the associated
        # parameters (c_female_dist, c_incp_dist, etc.) are GA-selected via the
        # _SAFE_PAIRS mechanism.  Estimating them directly triggers Numba
        # TypingError when GA_LARCH_ENGINE=numba.
        _PERSON_ZONE_INTERACTION_COLS = {
            'female_x_log_dist', 'edu_prim_priv_cond', 'occp_mgr_x_log_dist',
            'occp_trades_x_log_dist', 'propensity_x_log_dist',
            'person_dist_draw_x_log_dist', 'noveh_walk_acc', 'noveh_transit_acc',
            'incp_x_log_dist', 'age5p_x_log_dist', 'vehrd_x_log_dist',
            'nprd_x_log_dist', 'occp_group_x_log_dist',
            'ftwork_x_log_dist', 'WORKSTAT_x_log_dist', 'incp_x_auto_work_acc',
        }
        # Always flag these as collinear so their coefficients get holdfasted at 0
        # during _build_dest_model.  This prevents Larch's Numba flow compiler from
        # trying to JIT-compile them as 2D matrices (which crashes when they're 1D
        # flat-mode columns).
        _collinear_ca_cols.update(_PERSON_ZONE_INTERACTION_COLS)

        # Restrict to genuinely CA-varying columns - those with nonzero mean
        # within-chooser variance.  IDCO (person-level) variables are constant
        # across alternatives by definition; their apparent cross-correlation in
        # the expanded ptour×alt matrix is a structural artifact, not real
        # collinearity.  Checking them causes nearly every CO variable pair to
        # be flagged as |r|=1.0 and holdfasted, leaving only log_DIST_1 free
        # and making every GA chromosome produce identical fitness.
        # Also exclude identifier / weight columns that are never in the utility.
        _id_cols = {'ptour_id', '_altid_', 'OTAZ', 'DTAZ', 'weight',
                    'person_id', 'chosen', 'avail', 'tour_id'}
        if 'ptour_id' in ca_df.columns and len(_nonzero_cols) > 1:
            try:
                _within_var = (
                    ca_df[['ptour_id'] + [c for c in _nonzero_cols if c in ca_df.columns]]
                    .groupby('ptour_id')[[c for c in _nonzero_cols if c in ca_df.columns]]
                    .var()
                    .mean()
                )
                _nonzero_cols = [
                    c for c in _nonzero_cols
                    if c not in _id_cols and _within_var.get(c, 0.0) > 1e-12
                ]
            except Exception:
                _nonzero_cols = [c for c in _nonzero_cols if c not in _id_cols]
        else:
            _nonzero_cols = [c for c in _nonzero_cols if c not in _id_cols]

        if len(_nonzero_cols) > 1:
            _ca_mat = _numeric_ca[_nonzero_cols].values.astype(np.float64)
            # Normalize each column before corrcoef to avoid float32 precision issues
            _ca_norms = _ca_mat - _ca_mat.mean(axis=0)
            _ca_scale = np.maximum(np.std(_ca_norms, axis=0), 1e-12)
            _ca_norm = _ca_norms / _ca_scale
            _corr = np.corrcoef(_ca_norm.T)  # (n_cols, n_cols)
            _corr = np.abs(np.triu(_corr, k=1))
            _dupe_j, _dupe_i = np.where(_corr >= 0.999)
            for i, j in zip(_dupe_i, _dupe_j):
                c1, c2 = _nonzero_cols[i], _nonzero_cols[j]
                # Skip intentional skim log-transform pairs
                if frozenset([c1, c2]) in _SAFE_PAIRS:
                    logger.debug(
                        f"_dest_tree: skipping safe skim pair '{c1}' vs '{c2}' "
                        f"(|r|={_corr[j,i]:.4f}) - both are legitimate utility terms"
                    )
                    continue
                # Determine which variable to holdfast.  Never holdfast a primary
                # column (e.g. log_DIST_1, acdc_aggl); if one is primary, holdfast
                # the other.  If neither is primary, holdfast the one with the
                # higher positional index (c1, the later column in the DataFrame).
                if c2 in _NEVER_HOLDFAST_CA_COLS and c1 not in _NEVER_HOLDFAST_CA_COLS:
                    to_holdfast = c1
                elif c1 in _NEVER_HOLDFAST_CA_COLS and c2 not in _NEVER_HOLDFAST_CA_COLS:
                    to_holdfast = c2
                else:
                    # Both secondary - holdfast the first (lower position index)
                    to_holdfast = c2
                # Pairs that are expected/known (OMX duplicate skims or design):
                # demote to DEBUG - the prominent banner below explains root cause.
                _KNOWN_DUPLICATE_SKIM_COLS = {
                    'DISTBIKE', 'DISTWALK', 'log_DISTBIKE_1', 'log_DISTWALK_1',
                }
                _is_known = (
                    bool({c1, c2} & _KNOWN_DUPLICATE_SKIM_COLS)
                    or frozenset([c1, c2]) in {
                        frozenset(['acdc_comp', 'acdc_aggl']),
                        frozenset(['indp_finance_flag', 'occp_mgr']),
                    }
                )
                _pair_key = frozenset([c1, c2])
                _already_warned = _pair_key in _WARNED_COLLINEAR_CA_PAIRS
                _log_fn = logger.debug if (_is_known or _already_warned) else logger.warning
                _log_fn(
                    f"_dest_tree: COLLINEAR CA columns detected - "
                    f"'{c1}' vs '{c2}' (|r|={_corr[j,i]:.4f}); "
                    f"'{to_holdfast}' will be holdfasted at 0 in estimation"
                    + ("" if _already_warned else " (further repeats of this pair logged at DEBUG)")
                )
                _WARNED_COLLINEAR_CA_PAIRS.add(_pair_key)
                _collinear_ca_cols.add(to_holdfast)
                # Detect the specific DISTWALK=DIST / DISTBIKE=DIST cases
                if set([c1, c2]) & {'DISTWALK', 'DIST'} == {'DISTWALK', 'DIST'}:
                    _dist_walk_collinear = True
                if set([c1, c2]) & {'DISTBIKE', 'DIST'} == {'DISTBIKE', 'DIST'}:
                    _dist_walk_collinear = True

        # ── Prominent warning when OMX distance matrices are all identical ──
        if _dist_walk_collinear:
            global _DIST_WALK_BANNER_SHOWN
            if not _DIST_WALK_BANNER_SHOWN:
                _DIST_WALK_BANNER_SHOWN = True
                logger.warning(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "  DEST MODEL DATA ISSUE: DISTWALK (or DISTBIKE) == DIST in OMX.\n"
                    "  All three distance skims carry the same auto-network data.\n"
                    "  Consequence: c_log_distwalk / c_log_distbike holdfasted at 0.\n"
                    "  c_log_dist REMAINS FREE and active (primary distance term).\n"
                    "  Walk/bike feasibility flags are still in the model as binary\n"
                    "  indicators (c_walk_feasible, c_bike_feasible).\n"
                    "  Fix: supply a true pedestrian/cycling distance skim in the OMX.\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
    except Exception as _coll_e:
        logger.debug(f"_dest_tree: collinearity check skipped ({_coll_e})")

    # ── Home-pull gravity columns (nonmand only) ──────────────────────────
    if segment == 'nonmand':
        ca_df['home_pull'] = np.float32(0.0)
        ca_df['home_pull_late'] = np.float32(0.0)
        try:
            dist_vals = pd.to_numeric(ca_df['DIST'], errors='coerce').fillna(1.0).values
            base_pull = -np.log1p(np.maximum(dist_vals, 0.01))
            ca_df['home_pull'] = base_pull.astype(np.float32)
            ca_df['home_pull_late'] = base_pull.astype(np.float32)
            del dist_vals, base_pull
        except Exception as _hp_e:
            logger.warning(f"_dest_tree: home-pull failed ({_hp_e})")
    else:
        ca_df['home_pull'] = np.float32(0.0)
        ca_df['home_pull_late'] = np.float32(0.0)

    # ── Build flat Larch Dataset ─────────────────────────────────────────
    # Drop columns that also exist in CO to avoid xarray merge conflict:
    #   "conflicting values for variable 'X' on objects to be combined"
    # weight → must be CO only (weight_co_var reads CO)
    # OTAZ, DTAZ → must be CO only (choice_co_code = "DTAZ" reads CO)
    _drop_from_ca = [c for c in ['weight', 'OTAZ', 'DTAZ'] if c in ca_df.columns]
    if _drop_from_ca:
        ca_df = ca_df.drop(columns=_drop_from_ca)
        logger.info(f"_dest_tree: dropped {_drop_from_ca} from CA (kept in CO only)")
    ca_df_indexed = ca_df.set_index(['ptour_id', '_altid_']).sort_index()
    # ── dtype coercion: force every numeric CA column to float64 ──────────────
    # The sharrow/numba flow fills a float64 3-D result buffer; float32 columns
    # made the generated dim*_filler emit setitem(float64-3d <- float32-1d), which
    # numba can't type -> the whole dest flow fails to compile and falls back to
    # the JAX estimator. Matching the buffer dtype removes the dtype half of that
    # mismatch. Set STAGE5_DEST_F32=1 to keep float32 (debugging).
    if os.environ.get('STAGE5_DEST_F32', '0') != '1':
        _num32 = [c for c in ca_df_indexed.columns
                  if str(ca_df_indexed[c].dtype) in ('float32', 'float16')]
        if _num32:
            ca_df_indexed[_num32] = ca_df_indexed[_num32].astype(np.float64)
    d_ca = lx.Dataset.construct.from_idca(ca_df_indexed)
    logger.info(f"_dest_tree: d_ca built - {len(d_ca.data_vars)} vars, "
                f"dims={dict(d_ca.dims)}")

    # ── Force every CA variable to be 2-D (ptour_id × _altid_) ───────────────
    # from_idca() COLLAPSES any column that is constant across alternatives to a
    # 1-D (ptour_id,) variable (a memory optimisation). But the model's numba CA
    # flow generates a dim3_filler that assigns each column into a scalar slot
    # result[case, alt, k] — with a 1-D source that becomes
    #   setitem(float64-3d, (i,j,k), float64-1d)   → numba TypingError
    # and the ENTIRE dest flow fails to compile (→ JAX fallback, 5-param dest).
    # Broadcasting the collapsed vars back to full 2-D (same value across alts,
    # which is exactly right for person/case-level terms) makes the flat flow
    # compile and lets the full Larch dest model estimate. Disable with
    # STAGE5_DEST_NO_BROADCAST=1.
    if (os.environ.get('STAGE5_DEST_NO_BROADCAST', '0') != '1'
            and 'ptour_id' in d_ca.dims and '_altid_' in d_ca.dims):
        _n_alt = int(d_ca.sizes['_altid_'])
        _bcast = []
        for _v in list(d_ca.data_vars):
            _da = d_ca[_v]
            if _da.ndim < 2 and _da.dims == ('ptour_id',):
                # np.repeat is dask-free and avoids coord-alignment fragility:
                # tile the per-case value across every alternative → 2-D.
                _vals2d = np.repeat(np.asarray(_da.values)[:, None], _n_alt, axis=1)
                d_ca[_v] = (('ptour_id', '_altid_'), _vals2d)
                _bcast.append(str(_v))
        if _bcast and not getattr(_dest_tree, '_bcast_logged', False):
            # (informational, logged once) NOT an error — from_idca collapses
            # per-case-constant columns to 1-D; we re-expand them to 2-D so the
            # numba CA flow compiles and the full Larch dest model can estimate.
            logger.info(f"_dest_tree: re-expanded {len(_bcast)} per-case CA vars to "
                        f"2-D for the flat flow (e.g. {_bcast[:8]})")
            _dest_tree._bcast_logged = True

    # ── Per-variable shape/dtype diagnostic ──────────────────────────────────
    # The flat dest flow crashes when a CA variable comes out 1-D (indexed by
    # case only) while the filler expects 2-D (case x alt). Dump each var's
    # dims/ndim/dtype once so the offending column is identifiable from the log.
    # Gated by STAGE5_DEST_DIAG (default OFF now that the flat flow is fixed;
    # set STAGE5_DEST_DIAG=1 to re-enable the per-build shape dump).
    if os.environ.get('STAGE5_DEST_DIAG', '0') == '1':
        try:
            _ca_dim = None
            for _v in d_ca.data_vars:
                _da = d_ca[_v]
                if _ca_dim is None:
                    _ca_dim = tuple(_da.dims)
            _oned = [str(_v) for _v in d_ca.data_vars if d_ca[_v].ndim < 2]
            _f32  = [str(_v) for _v in d_ca.data_vars
                     if str(d_ca[_v].dtype) in ('float32', 'float16')]
            # WARNING level so it lands in the .e log (INFO is written nowhere
            # accessible in the stage jobs).
            logger.warning(
                f"[DEST DIAG] d_ca vars={len(d_ca.data_vars)} dims={dict(d_ca.dims)} "
                f"| 1-D vars ({len(_oned)}): {_oned[:25]} "
                f"| non-float64 vars ({len(_f32)}): {_f32[:25]}")
        except Exception as _dd_e:
            logger.info(f"[DEST DIAG] diagnostic failed: {_dd_e!r}")

    # Merge tour co-data
    tour_co = tour_df.copy()
    # Keep only useful co columns - weight MUST be here (weight_co_var reads CO)
    _co_keep = ['OTAZ', 'DTAZ', 'person_id', 'PERSID', 'weight',
                'tour_mode', 'tour_purpose', 'segment']
    _co_keep = [c for c in _co_keep if c in tour_co.columns]
    if _co_keep:
        tour_co = tour_co[_co_keep]
    d_co = lx.Dataset.construct.from_idco(tour_co)
    logger.info(f"_dest_tree: d_co built - {len(d_co.data_vars)} vars: "
                f"{list(d_co.data_vars)}")

    # Check for overlapping variables before merge
    _ca_vars = set(d_ca.data_vars)
    _co_vars = set(d_co.data_vars)
    _overlap = _ca_vars & _co_vars
    if _overlap:
        logger.warning(f"_dest_tree: CA/CO variable overlap detected: {_overlap} - "
                       f"dropping from d_ca before merge")
        d_ca = d_ca.drop_vars(list(_overlap))
    d = d_ca.merge(d_co)

    # Attach metadata for downstream code via .attrs (xarray ≥2024 forbids
    # direct attribute assignment on Dataset objects)
    d.attrs['_od_vars_cached'] = list(_cached_od_vars) if _cached_od_vars else []
    d.attrs['_flat_mode'] = True  # flag for _build_dest_model to use flat expressions
    d.attrs['_collinear_cols'] = list(_collinear_ca_cols)  # columns to holdfast at 0

    # Diagnostic summary
    _n_ca = len([v for v in d.data_vars if '_altidx_' in str(d[v].dims)])
    _n_co = len(d.data_vars) - _n_ca
    logger.info(
        f"_dest_tree: FLAT Dataset built - {len(ca_df)} ca rows, "
        f"{len(ca_df.columns)} ca columns, "
        f"{_n_co} co vars, {_n_ca} ca vars merged OK "
        f"(ActivitySim pre-joined pattern)"
    )
    return d


def _importance_sample_dest_alts(
    tours_df: pd.DataFrame,
    landuse: pd.DataFrame,
    zone_ids: list,
    sample_size: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Importance sampling of destination alternatives for HHTS estimation.

    For each HHTS observation the observed DTAZ is **always** included as an
    alternative (Manski-Lerman 1977 requirement for consistent MLE).  The
    remaining ``sample_size - 1`` alternatives are drawn without replacement
    from an importance distribution proportional to zone size_term.

    The Manski-Lerman correction
        log_correction_j = −log(q_j)
    is added to every alternative's utility so the MLE remains consistent
    despite the biased sample.  q_j is the importance weight of zone j under
    the size-term distribution.  For the forced-in chosen zone we use its
    same q_j (as if it had been drawn normally) - this is the standard
    exogenous sampling correction (Ben-Akiva & Lerman 1985, §8.4).

    Parameters
    ----------
    tours_df   : HHTS tours, must have 'ptour_id' and 'DTAZ' columns.
    landuse    : zone-level data (must have 'size_term' and 'ORDER' columns).
    zone_ids   : list of valid zone IDs (from logsum coordinate).
    sample_size: total alternatives per observation (including chosen).
    seed       : RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns ['ptour_id', 'alt_zone', 'log_correction'].
    One row per (observation, alternative) - the sampled alternative set.
    """
    rng = np.random.default_rng(seed)

    zone_arr = np.asarray(zone_ids, dtype=np.int32)
    n_zones  = len(zone_arr)
    zone_set = set(zone_arr.tolist())

    # ── Importance weights from size_term ────────────────────────────────
    lu = landuse.copy()
    if lu.index.name == 'ORDER':
        lu = lu.reset_index()
    zone_col = next((c for c in ['ORDER', 'zone_id'] if c in lu.columns), lu.columns[0])
    lu = lu.set_index(zone_col)

    st = pd.to_numeric(lu.get('size_term', pd.Series(1.0, index=lu.index)), errors='coerce').fillna(0.0)
    st = st.reindex(zone_arr.tolist()).fillna(0.0)
    # unlog if values look like logged size terms (they are stored log-scaled)
    st_raw = np.exp(np.clip(st.values.astype(np.float64), -30, 30))
    st_raw = np.clip(st_raw, 1e-10, None)
    q = st_raw / st_raw.sum()           # importance distribution over zone_arr

    # ── Pre-compute log(q) once ───────────────────────────────────────────
    log_q = np.log(q)                   # −log_q is the Manski-Lerman correction

    zone_to_idx = {int(z): i for i, z in enumerate(zone_arr)}

    sample_size = max(2, min(int(sample_size), n_zones))
    draw_size   = sample_size - 1       # extra draws beyond chosen

    rows = []
    _n_chosen_out = 0
    for _, tour in tours_df.iterrows():
        pid = str(tour['ptour_id'])
        try:
            chosen = int(tour['DTAZ'])
        except (ValueError, TypeError):
            chosen = -1

        chosen_in_universe = (chosen in zone_set)

        # ── The CHOSEN alternative is ALWAYS included (Manski-Lerman requires
        #    the observed alternative to be in every sampled set for a consistent
        #    MLE). It is emitted FIRST, then the remaining alternatives are drawn
        #    from q without replacement (excluding the chosen so it's never a
        #    duplicate). This holds even when sample_size==1 or when the chosen
        #    zone is OUTSIDE the sampling universe (external/fringe DTAZ).
        if chosen in zone_to_idx:
            c_idx = zone_to_idx[chosen]
            rows.append({'ptour_id': pid, 'alt_zone': chosen,
                         'log_correction': float(-log_q[c_idx])})
        elif chosen >= 0:
            # Out-of-universe chosen: still forced in, deterministic inclusion →
            # log_correction 0 (its downstream size/skim terms default; the point
            # is the observed alt is present so the MLE is consistent).
            rows.append({'ptour_id': pid, 'alt_zone': chosen, 'log_correction': 0.0})
            _n_chosen_out += 1

        # Remaining draws from q, excluding the chosen zone to avoid duplicates.
        n_drawn = draw_size if (chosen in zone_to_idx or chosen >= 0) else sample_size
        n_drawn = min(n_drawn, n_zones - (1 if chosen in zone_to_idx else 0))
        if n_drawn > 0:
            q_draw = q.copy()
            if chosen in zone_to_idx:
                q_draw[zone_to_idx[chosen]] = 0.0
            s = q_draw.sum()
            q_draw = q_draw / s if s > 0 else np.ones(n_zones) / n_zones
            for idx in rng.choice(n_zones, size=n_drawn, replace=False, p=q_draw):
                rows.append({
                    'ptour_id':       pid,
                    'alt_zone':       int(zone_arr[idx]),
                    'log_correction': float(-log_q[idx]),   # Manski-Lerman: −log(q_j)
                })

    result = pd.DataFrame(rows, columns=['ptour_id', 'alt_zone', 'log_correction'])
    n_obs = tours_df['ptour_id'].nunique() if 'ptour_id' in tours_df.columns else len(tours_df)
    # Verify the invariant: every observation's chosen DTAZ is in its sampled set.
    _chosen_present = True
    try:
        _obs = tours_df[['ptour_id', 'DTAZ']].dropna().copy()
        _obs['ptour_id'] = _obs['ptour_id'].astype(str)
        _obs['DTAZ'] = pd.to_numeric(_obs['DTAZ'], errors='coerce').astype('Int64')
        _pairs = set(zip(result['ptour_id'].astype(str), result['alt_zone'].astype(int)))
        _missing = sum(1 for p, d in zip(_obs['ptour_id'], _obs['DTAZ'])
                       if pd.notna(d) and int(d) >= 0 and (p, int(d)) not in _pairs)
        _chosen_present = (_missing == 0)
        if _missing:
            logger.error(f"_importance_sample_dest_alts: {_missing} observations "
                         "have their chosen DTAZ MISSING from the sample set (bug)")
    except Exception:
        pass
    logger.info(
        f"_importance_sample_dest_alts: {n_obs} observations × {sample_size} alts "
        f"= {len(result)} rows; q range=[{q.min():.2e},{q.max():.2e}]; "
        f"chosen-in-set={'OK' if _chosen_present else 'FAILED'}"
        + (f"; {_n_chosen_out} chosen out-of-universe (forced in, corr=0)"
           if _n_chosen_out else "")
    )
    return result


def _estimate_dest_coefficients_alt(
    seg_tours: pd.DataFrame,
    landuse: pd.DataFrame,
    logsums,
    sampled_alts: pd.DataFrame,
    skim_df: pd.DataFrame,
    crash_source=None,
    return_stats: bool = False,
):
    """
    Alternative destination estimation when Larch destination fit fails.

    Fits a compact sampled-alternative conditional logit with JAX:
        U_ij = b_logdist*log(dist_ij+1) + b_dist*dist_ij + b_size*size_term_j
             + b_logsum*logsum_ij + b_crash*log(crash_ij+1) + log_correction_ij

    Returns a coefficient dict compatible with _dest_utility_numpy.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError("alt dest estimation requires JAX, but JAX is not available")

    import jax
    import jax.numpy as jnp_local

    if sampled_alts is None or sampled_alts.empty:
        raise RuntimeError("alt dest estimation: sampled_alts is empty")
    if skim_df is None or skim_df.empty:
        raise RuntimeError("alt dest estimation: skim_df is empty")
    if not hasattr(logsums, 'coords') or 'cases' not in logsums.coords or 'zones' not in logsums.coords:
        raise RuntimeError("alt dest estimation: logsums missing cases/zones coords")

    case_ids = [str(c) for c in logsums.coords['cases'].values]
    zone_ids = [int(z) for z in logsums.coords['zones'].values]
    case_to_row = {cid: i for i, cid in enumerate(case_ids)}
    zone_to_col = {z: j for j, z in enumerate(zone_ids)}
    ls_vals = np.asarray(logsums.values, dtype=np.float64)

    sk = skim_df.copy()
    if 'otaz' in sk.columns and 'OTAZ' not in sk.columns:
        sk = sk.rename(columns={'otaz': 'OTAZ'})
    if 'dtaz' in sk.columns and 'DTAZ' not in sk.columns:
        sk = sk.rename(columns={'dtaz': 'DTAZ'})
    if 'OTAZ' not in sk.columns or 'DTAZ' not in sk.columns:
        raise RuntimeError("alt dest estimation: skim_df missing OTAZ/DTAZ")

    dist_col = next((c for c in ['DIST', 'SOVTOLL_DIST__AM', 'DISTWALK', 'DISTBIKE']
                     if c in sk.columns), None)
    if dist_col is None:
        raise RuntimeError("alt dest estimation: no distance-like skim column found")

    sk_cols = ['OTAZ', 'DTAZ', dist_col]
    if 'CRASH' in sk.columns:
        sk_cols.append('CRASH')
    sk = sk[sk_cols].drop_duplicates(['OTAZ', 'DTAZ'])
    sk['OTAZ'] = pd.to_numeric(sk['OTAZ'], errors='coerce').fillna(0).astype(int)
    sk['DTAZ'] = pd.to_numeric(sk['DTAZ'], errors='coerce').fillna(0).astype(int)
    sk[dist_col] = pd.to_numeric(sk[dist_col], errors='coerce').fillna(30.0)
    dist_map = {(int(o), int(d)): float(v)
                for o, d, v in sk[['OTAZ', 'DTAZ', dist_col]].itertuples(index=False, name=None)}
    if 'CRASH' in sk.columns:
        sk['CRASH'] = pd.to_numeric(sk['CRASH'], errors='coerce').fillna(0.0)
        crash_map = {(int(o), int(d)): float(v)
                     for o, d, v in sk[['OTAZ', 'DTAZ', 'CRASH']].itertuples(index=False, name=None)}
    else:
        crash_map = {}
        if crash_source is not None:
            try:
                cdf = _read_crash_od_table(crash_source, zone_ids=zone_ids)
                crash_map = {(int(o), int(d)): float(v)
                             for o, d, v in cdf[['OTAZ', 'DTAZ', 'CRASH']].itertuples(index=False, name=None)}
            except Exception as _cr_e:
                logger.debug(f"alt dest estimation: crash source unavailable ({_cr_e})")

    lu = landuse.copy()
    if lu.index.name == 'ORDER':
        lu = lu.reset_index()
    zone_col = next((c for c in ['ORDER', 'zone_id'] if c in lu.columns), lu.columns[0])
    lu_idx = lu.set_index(zone_col)
    size_series = pd.to_numeric(
        lu_idx.get('size_term', pd.Series(0.0, index=lu_idx.index)),
        errors='coerce'
    ).fillna(0.0)
    size_map = {int(k): float(v) for k, v in size_series.items()}

    t = seg_tours[['ptour_id', 'OTAZ', 'DTAZ']].copy()
    t['ptour_id'] = t['ptour_id'].astype(str)
    t['OTAZ'] = pd.to_numeric(t['OTAZ'], errors='coerce').fillna(0).astype(int)
    t['DTAZ'] = pd.to_numeric(t['DTAZ'], errors='coerce').fillna(0).astype(int)
    t = t.drop_duplicates('ptour_id').set_index('ptour_id')

    sa = sampled_alts.copy()
    sa['ptour_id'] = sa['ptour_id'].astype(str)
    sa['alt_zone'] = pd.to_numeric(sa['alt_zone'], errors='coerce').fillna(0).astype(int)
    if 'log_correction' not in sa.columns:
        sa['log_correction'] = 0.0
    sa['log_correction'] = pd.to_numeric(sa['log_correction'], errors='coerce').fillna(0.0)

    datasets = []  # (logdist, dist, size_term, logsum, logcrash, correction, chosen_idx)
    for pid, grp in sa.groupby('ptour_id', sort=False):
        if pid not in t.index or pid not in case_to_row:
            continue

        otaz = int(t.loc[pid, 'OTAZ'])
        chosen_zone = int(t.loc[pid, 'DTAZ'])
        row_idx = case_to_row[pid]

        alts = [int(z) for z in grp['alt_zone'].tolist() if int(z) in zone_to_col]
        if not alts:
            continue
        if chosen_zone not in alts and chosen_zone in zone_to_col:
            alts.append(chosen_zone)

        corr_by_zone = {int(z): float(c)
                        for z, c in zip(grp['alt_zone'].tolist(), grp['log_correction'].tolist())}

        logdist = []
        distlin = []
        sizeterm = []
        logsum = []
        logcrash = []
        correction = []
        for z in alts:
            d = dist_map.get((otaz, int(z)), 30.0)
            logdist.append(np.log(max(float(d), 0.0) + 1.0))
            distlin.append(float(max(d, 0.0)))
            sizeterm.append(size_map.get(int(z), 0.0))
            logsum.append(float(ls_vals[row_idx, zone_to_col[int(z)]]))
            cval = crash_map.get((otaz, int(z)), 0.0)
            logcrash.append(np.log(max(float(cval), 0.0) + 1.0))
            correction.append(corr_by_zone.get(int(z), 0.0))

        if chosen_zone not in alts:
            continue
        datasets.append((
            np.asarray(logdist, dtype=np.float64),
            np.asarray(distlin, dtype=np.float64),
            np.asarray(sizeterm, dtype=np.float64),
            np.asarray(logsum, dtype=np.float64),
            np.asarray(logcrash, dtype=np.float64),
            np.asarray(correction, dtype=np.float64),
            int(alts.index(chosen_zone)),
        ))

    if len(datasets) < 50:
        raise RuntimeError(f"alt dest estimation: insufficient sampled cases ({len(datasets)} < 50)")

    n_cases = len(datasets)
    max_alts = max(len(d[0]) for d in datasets)
    ld_mat = np.zeros((n_cases, max_alts), dtype=np.float32)
    dl_mat = np.zeros((n_cases, max_alts), dtype=np.float32)
    st_mat = np.zeros((n_cases, max_alts), dtype=np.float32)
    ls_mat = np.zeros((n_cases, max_alts), dtype=np.float32)
    lcsh_mat = np.zeros((n_cases, max_alts), dtype=np.float32)
    lc_mat = np.zeros((n_cases, max_alts), dtype=np.float32)
    mask_mat = np.zeros((n_cases, max_alts), dtype=bool)
    chosen_idx = np.zeros(n_cases, dtype=np.int32)

    for i, (ld, dl, st, ls, lcr, lc, ch) in enumerate(datasets):
        k = len(ld)
        ld_mat[i, :k] = ld.astype(np.float32)
        dl_mat[i, :k] = dl.astype(np.float32)
        st_mat[i, :k] = st.astype(np.float32)
        ls_mat[i, :k] = ls.astype(np.float32)
        lcsh_mat[i, :k] = lcr.astype(np.float32)
        lc_mat[i, :k] = lc.astype(np.float32)
        mask_mat[i, :k] = True
        chosen_idx[i] = int(ch)

    # Standardize feature scales for stable JAX optimization.
    def _stdz(arr, mask):
        vals = arr[mask]
        mu = float(vals.mean()) if len(vals) else 0.0
        sd = float(vals.std()) if len(vals) else 1.0
        sd = sd if sd > 1e-6 else 1.0
        return (arr - mu) / sd

    ld_mat = _stdz(ld_mat, mask_mat)
    dl_mat = _stdz(dl_mat, mask_mat)
    st_mat = _stdz(st_mat, mask_mat)
    ls_mat = _stdz(ls_mat, mask_mat)
    lcsh_mat = _stdz(lcsh_mat, mask_mat)

    ld_j = jnp_local.asarray(ld_mat)
    dl_j = jnp_local.asarray(dl_mat)
    st_j = jnp_local.asarray(st_mat)
    ls_j = jnp_local.asarray(ls_mat)
    lcr_j = jnp_local.asarray(lcsh_mat)
    lc_j = jnp_local.asarray(lc_mat)
    mask_j = jnp_local.asarray(mask_mat)
    chosen_j = jnp_local.asarray(chosen_idx)

    def _loss(theta):
        b_logdist, b_dist, b_size, b_logsum, b_crash = theta
        util = (
            b_logdist * ld_j
            + b_dist * dl_j
            + b_size * st_j
            + b_logsum * ls_j
            + b_crash * lcr_j
            + lc_j
        )
        util = jnp_local.where(mask_j, util, -1e9)
        log_denom = jax.nn.logsumexp(util, axis=1)
        chosen_util = util[jnp_local.arange(util.shape[0]), chosen_j]
        return -jnp_local.mean(chosen_util - log_denom)

    loss_and_grad = jax.jit(jax.value_and_grad(_loss))

    theta = jnp_local.array([-0.6, -0.1, 1.0, 1.0, -0.05], dtype=jnp_local.float32)
    m = jnp_local.zeros_like(theta)
    v = jnp_local.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    lr = 0.03
    best_theta = theta
    best_loss = np.inf
    patience = 80
    stale = 0

    for step in range(1, 801):
        loss_val, grad_val = loss_and_grad(theta)
        if not (np.isfinite(float(loss_val)) and np.all(np.isfinite(np.asarray(grad_val)))):
            raise RuntimeError(f"alt dest estimation (jax) diverged at step={step}")

        m = beta1 * m + (1.0 - beta1) * grad_val
        v = beta2 * v + (1.0 - beta2) * (grad_val * grad_val)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        theta = theta - lr * m_hat / (jnp_local.sqrt(v_hat) + eps)

        theta = theta.at[0].set(jnp_local.clip(theta[0], -8.0, -1e-4))
        theta = theta.at[1].set(jnp_local.clip(theta[1], -8.0, 8.0))
        theta = theta.at[2].set(jnp_local.clip(theta[2], -10.0, 10.0))
        theta = theta.at[3].set(jnp_local.clip(theta[3], -10.0, 20.0))
        theta = theta.at[4].set(jnp_local.clip(theta[4], -8.0, 1e-4))

        cur_loss = float(loss_val)
        if cur_loss + 1e-8 < best_loss:
            best_loss = cur_loss
            best_theta = theta
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    opt = np.asarray(best_theta, dtype=np.float64)
    if not np.all(np.isfinite(opt)):
        raise RuntimeError("alt dest estimation (jax) produced non-finite coefficients")

    _names = ['c_log_dist', 'c_dist', 'c_size_term', 'c_logsum', 'c_crash']
    coef = {_names[i]: float(opt[i]) for i in range(len(_names))}

    # ── Standard errors from the observed information matrix ──────────────────
    # This fallback previously returned point estimates only, so its dest
    # coefficients showed NaN std_err/t/p in the reports. Compute the Hessian of
    # the SUM negative log-likelihood at the optimum (=(n_cases)·Hessian of the
    # mean NLL that _loss returns), invert for the covariance, and derive SE/t/p.
    stats = {k: {'std_err': float('nan'), 't_stat': float('nan'),
                 'pval': float('nan')} for k in coef}
    try:
        _H_mean = np.asarray(jax.hessian(_loss)(best_theta), dtype=np.float64)
        _H_sum = _H_mean * float(n_cases)
        _cov = np.linalg.inv(_H_sum)
        _se = np.sqrt(np.clip(np.diag(_cov), 0.0, None))
        from scipy.stats import norm as _norm
        for _i, _nm in enumerate(_names):
            _s = float(_se[_i])
            _t = float(opt[_i] / _s) if (np.isfinite(_s) and _s > 0) else float('nan')
            _p = float(2.0 * _norm.sf(abs(_t))) if np.isfinite(_t) else float('nan')
            stats[_nm] = {'std_err': _s, 't_stat': _t, 'pval': _p}
    except Exception as _se_e:
        logger.debug(f"_estimate_dest_coefficients_alt: SE computation failed ({_se_e})")

    logger.info(
        "_estimate_dest_coefficients_alt: estimated JAX fallback coefficients on "
        f"{len(datasets)} cases "
        f"(c_log_dist={coef['c_log_dist']:.4f}, c_dist={coef['c_dist']:.4f}, "
        f"c_size_term={coef['c_size_term']:.4f}, c_logsum={coef['c_logsum']:.4f}, "
        f"c_crash={coef['c_crash']:.4f}, best_nll={best_loss:.6f}; "
        f"SE {'computed' if np.isfinite(stats['c_dist']['std_err']) else 'unavailable'})"
    )
    return (coef, stats) if return_stats else coef


# ===========================================================================
# UTILITY SPECIFICATIONS  (ActivitySim-style UEC pattern)
# ===========================================================================
#
# Utility specs are plain Python dicts - no hardcoded branches, no CSV files.
# Each entry is:
#   coefficient_name : (expression, {segment_filter})
#
# segment_filter is optional - if present, the term is only added when the
# segment matches.  An empty dict {} means the term applies to all segments.
#
# To add a variable: add one line to the relevant dict.
# To remove a variable: delete or comment out the line.
# To change a coefficient starting value: edit the "init" sub-dict below.
#
# This mirrors ActivitySim's UEC / coefficients.csv pattern but keeps
# everything in Python so you don't need CSV files to edit the model.
# ===========================================================================

# ── Mode choice UEC ──────────────────────────────────────────────────────────
# Each entry:  coeff_name: (expression, {mode_name: segments_list_or_None})
# segments_list=None means active for all segments.
# Expression context: person.xxx (person sub-dataset), tour.xxx (tour sub-dataset),
#                     od.xxx (OMX skim), emp.xxx (landuse / employment)
#
# Behavioural design principles (matching QTS HHTS evidence for SEQ):
#   DA (drive alone)  - car ownership and licence are strong enablers;
#                       full-time workers peak-commute by car; income positive.
#   Other (shared)    - larger households; part-time workers; lower income.
#   Walk              - short distances; zero-car HHs; women in SEQ walk more;
#                       younger adults; outbound stops reduce walk attractiveness.
#   Bike              - HH bicycle ownership key; youth (<25yr) dominant;
#                       shopping trips less compatible; no car HHs use it more.
#   Transit           - zero-car HHs; women; lower income; longer travel time
#                       trips where car is less competitive.
# ASC reference mode = DA (no ASC for DA).
MODE_UEC = {
    # ── Mode-specific travel time deterrents ─────────────────────────────
    #
    # SCALE DESIGN - all expressions must stay roughly 0-7 so set_cap(±10)
    # never binds and the optimizer can find meaningful coefficients:
    # ── Mode-specific travel time deterrents ─────────────────────────
    #
    # WALK / BIKE DISTANCE → TIME CONVERSION
    # ----------------------------------------
    # DISTWALK and DISTBIKE are stored in km in this OMX.
    # Walk speed = 5 km/h  → 12 min/km  → time(min) = DISTWALK * 12
    # Bike speed = 15 km/h → 4  min/km  → time(min) = DISTBIKE * 4
    # We use log(time + 1) to compress the right tail of the distribution.
    # Median inter-zonal DISTWALK ≈ 32 km for 329-zone SEQ system (many zones far apart).
    # Only short-distance OD pairs are walk/bike available (filtered by walk_avail/bike_avail).
    #
    # Auto/Transit: SOVTOLL_TIME__AM median=58.8 min → divide by 60 → hours
    "c_ivt_auto":       ("od.SOVTOLL_TIME__AM / 60.0",         {"DA": None, "SR": None}),
    "c_ivt_transit":    ("od.WLK_LOC_WLK_TOTIVT__AM / 60.0",  {"Transit": None}),
    "c_walk_time":      ("log(max(od.DISTWALK, 0.3) * 12 + 1)", {"Walk": None}),
    "c_bike_time":      ("log(max(od.DISTBIKE, 0.5) * 4 + 1)",   {"Bike": None}),

    # ── Crash safety perception ────────────────────────────────────────────
    "c_crash_veh":      ("log(od.CRASH+1)",               {"DA": None, "SR": None, "Transit": None}),
    # Per-agent safety utility × crash exposure on active modes: safety-sensitive
    # agents (children, seniors, women, car-free) avoid crash-prone walk/bike
    # routes more strongly. Gated (candidate 'safety_sensitivity'); sign NEG.
    "c_crash_safety":   ("person.safety_sensitivity * log(od.CRASH+1)", {"Walk": None, "Bike": None}),

    # ── Vehicle ownership ─────────────────────────────────────────────────
    "c_veh_da":         ("person.vehrd",                  {"DA": None}),
    "c_veh_sr":         ("person.vehrd",                  {"SR": None}),
    "c_noveh_transit":  ("(person.vehrd == 0)",           {"Transit": None}),
    "c_noveh_walk":     ("(person.vehrd == 0)",           {"Walk": None}),

    # ── Car licence ───────────────────────────────────────────────────────
    # CARLICENCE is now a HARD AVAILABILITY constraint for DA (not a utility term).
    # Persons without a licence cannot drive alone; they can still be SR passengers.
    # availability_co_vars = {Mode.DA: "CARLICENCE", ...} in _build_mode_model().

    # ── Income ────────────────────────────────────────────────────────────
    "c_incp_da":        ("person.incp",                   {"DA": None}),
    "c_incp_sr":        ("person.incp",                   {"SR": None}),
    "c_incp_transit":   ("person.incp",                   {"Transit": None}),
    "c_incp_active":    ("person.incp",                   {"Walk": None, "Bike": None}),

    # ── Age ───────────────────────────────────────────────────────────────
    "c_age_walk":       ("person.age5p",                  {"Walk": None}),
    "c_age_bike":       ("person.age5p",                  {"Bike": None}),
    "c_young_bike":     ("(person.age5p <= 5)",           {"Bike": None}),

    # ── Household size ────────────────────────────────────────────────────
    "c_hhsize_sr":      ("person.nprd",                   {"SR": None}),
    "c_joint_sr":       ("tour.joint_travel",            {"SR": None}),

    # ── Bicycle ownership ─────────────────────────────────────────────────
    "c_bikes_bike":     ("person.bikes",                  {"Bike": None}),

    # ── Habitual mode-usage captivity (mobility_models propensity) ────────────
    # Same-mode captivity index = the mobility model's (1-π)μ expected weekly
    # trips, a smooth function of EXOGENOUS person attributes (not the realised
    # trip count), so it is non-circular with the tour's observed mode.
    "c_transit_use_transit": ("person.transit_prop",      {"Transit": None}),
    "c_car_use_da":          ("person.car_prop",          {"DA": None}),
    "c_bike_use_bike":       ("person.bike_prop",         {"Bike": None}),

    # ── Sex ───────────────────────────────────────────────────────────────
    "c_female_transit": ("(person.sexp == 2)",            {"Transit": None}),
    "c_female_walk":    ("(person.sexp == 2)",            {"Walk": None}),

    # ── Worker type ───────────────────────────────────────────────────────
    # Use the pre-computed binary dummies (workstat_fulltime/parttime) rather than
    # inline equality tests against the raw ordinal WORKSTAT code.  The binaries are
    # derived in _ensure_person_behavioural_cols (and upstream in
    # cdap.build_person_profiles) with fixed category→name mappings, so the column
    # is unambiguously a 0/1 indicator and its coefficient is directly interpretable.
    "c_ftwork_da":        ("person.workstat_fulltime",    {"DA": None}),
    "c_ptwork_sr":        ("person.workstat_parttime",    {"SR": None}),
    # Non-working persons (workstat_notworking) tend toward transit / walk — they
    # lack the work-schedule pressure that makes DA reliable and may not own a car.
    # Positive coefficient → non-workers are systematically more transit-prone.
    "c_notwork_transit":  ("person.workstat_notworking",  {"Transit": None}),

    # ── Walk / bike stops ─────────────────────────────────────────────────
    "c_walk_stops":     ("tour.outbound_stops",           {"Walk": None}),
    "c_bike_stops":     ("tour.outbound_stops",           {"Bike": None}),
    "c_bike_veh":       ("(person.vehrd >= 1)",           {"Bike": ["nonmand"]}),
    "c_bike_shop":      ("(tour.n_shopping_out > 0)",     {"Bike": ["nonmand"]}),

    # ── Time-of-day ───────────────────────────────────────────────────────
    "c_peak_da":        ("(tour._time_period == 1)",      {"DA": None}),

    # ── CBD-destination × transit ─────────────────────────────────────────
    # Zones with ORDER in CBD_ZONES (inner-city high-employment, high-parking-cost
    # zones) shift utility toward transit. Captures the parking-cost / congestion
    # premium that makes transit competitive for CBD-bound tours.
    "c_cbd_transit":    ("emp.is_cbd",                     {"Transit": None}),
    # Negative: CBD destinations deter car (parking cost, congestion zone).
    "c_cbd_da":         ("emp.is_cbd",                     {"DA": None}),
    "c_cbd_sr":         ("emp.is_cbd",                     {"SR": None}),

    # ── Transit access+egress ─────────────────────────────────────────────
    # transit_walk_acc[j] = walk access to transit stops in destination zone j
    # (computed from SA1 walk skims in _ensure_transit_access_cols).
    # Positive: zones with good transit access attract transit trips.
    "c_transit_access": ("emp.transit_walk_acc",           {"Transit": None}),

    # ── Micromobility ownership × bike ────────────────────────────────────
    # ebikes / escooters are binary ownership flags (0/1) on the person table.
    # Positive: owning micromobility makes bike mode more attractive.
    "c_ebike_bike":     ("person.ebikes",                  {"Bike": None}),
    "c_escooter_bike":  ("person.escooter",                {"Bike": None}),

    # ── Purpose × mode dummies ────────────────────────────────────────────
    # Work tours have higher DA share (scheduling pressure, parking at work);
    # nonmand tours have higher active-mode share (local, flexible).
    # These are purpose interaction terms – the ASC already captures the
    # unconditional mode baseline; these shift the ASC per segment.
    "c_work_sr":        ("(tour.main_purpose == 'work')", {"SR": None}),
    "c_work_walk":      ("(tour.main_purpose == 'work')", {"Walk": None}),
    "c_work_bike":      ("(tour.main_purpose == 'work')", {"Bike": None}),
    "c_work_transit":   ("(tour.main_purpose == 'work')", {"Transit": None}),
    "c_edu_walk":       ("(tour.main_purpose == 'edu')",  {"Walk": None}),
    "c_edu_bike":       ("(tour.main_purpose == 'edu')",  {"Bike": None}),
    "c_edu_transit":    ("(tour.main_purpose == 'edu')",  {"Transit": None}),

    # ═══════════════════════════════════════════════════════════════════════
    # ActivitySim-inspired GA candidate terms (all gated — see _COEFF_GATE /
    # _EXTRA_SKIM_SPECS; inactive unless the matching gene is selected, so the
    # default/backbone spec is unchanged).
    # ═══════════════════════════════════════════════════════════════════════

    # ── Cost / tolls (ActivitySim tour_mode_choice: cost / income) ───────────
    # Money cost enters divided by (income+1) so the marginal utility of cost
    # falls with income — the standard cost-by-income specification. Wrapped in
    # parens so _mode_spec_custom routes it through _COEFF_GATE (not the skim
    # auto-gate). Sign: negative (see ga_constraints.sign_constraints).
    "c_toll_auto":      ("(od.SOVTOLL_TOLL__AM) / 10.0",              {"DA": None, "SR": None}),
    "c_cost_auto":      ("(od.DIST) / (person.incp + 1.0)",          {"DA": None, "SR": None}),

    # ── Auto-sufficiency / captivity (autos vs household size) ───────────────
    # vehrd < nprd → the household has fewer cars than people (competition for
    # the car) → shifts utility toward transit / walk. ActivitySim auto_ownership
    # / auto-sufficiency segmentation.
    "c_autodef_transit": ("(person.vehrd < person.nprd)",            {"Transit": None}),
    "c_autodef_walk":    ("(person.vehrd < person.nprd)",            {"Walk": None}),

    # ── Short-distance active-mode dummies ───────────────────────────────────
    # Very short OD pairs strongly favour walk / bike (ActivitySim distance
    # bands). Parenthesised → gated via _COEFF_GATE.
    "c_short_walk":     ("(od.DISTWALK < 2.0)",                      {"Walk": None}),
    "c_short_bike":     ("(od.DISTBIKE < 5.0)",                      {"Bike": None}),

    # ── Person-type (ptype) × mode shifters (ActivitySim ptype segmentation) ─
    "c_student_transit": ("(person.STUDYING > 0)",                  {"Transit": None}),
    "c_student_walk":    ("(person.STUDYING > 0)",                  {"Walk": None}),
    "c_retired_walk":    ("person.workstat_notworking",             {"Walk": None, "Bike": None}),

    # ── Occupation / job-type ("job_tup") × mode ─────────────────────────────
    # White-collar/managers commute by car more; trades likewise; professionals
    # more transit-prone in CBD-oriented jobs. occp_* are person dummies.
    "c_mgr_da":         ("person.occp_mgr",                         {"DA": None}),
    "c_trades_da":      ("person.occp_trades",                      {"DA": None}),
    "c_prof_transit":   ("(person.occp_group == 1)",               {"Transit": None}),

    # ── Alternative-specific constants (DA = reference, no ASC) ──────────
    "ASC_SR":           ("1",                             {"SR": None}),
    "ASC_Walk":         ("1",                             {"Walk": None}),
    "ASC_Bike":         ("1",                             {"Bike": None}),
    "ASC_Transit":      ("1",                             {"Transit": None}),
}

# ── Destination choice UEC ───────────────────────────────────────────────────
# ActivitySim pattern: ONE composite size_term per zone replaces many individual
# employment columns. Person × zone interactions capture behavioural heterogeneity:
# who travels how far, and to what type of zone.
#
# Behavioural design principles:
#   Distance deterrence - everyone is deterred by distance, but the degree varies:
#       higher income → less deterred (c_incp_dist positive)
#       older age     → more deterred  (c_age_dist negative)
#       more vehicles → less deterred  (c_veh_dist positive)
#   Accessibility - zone-level cumulative opportunity; persons in car-oriented
#       or zero-car HHs respond differently to walk vs auto accessibility.
#   Worker-type conditioning - FT workers gravitate to high-employment zones;
#       retired/non-workers prefer locally-accessible nonmand destinations.
#   Industry / study conditioning - carried over from previous version, now
#       properly interact with the person's industry and study level.
DEST_UEC = {
    # ── Universal distance deterrent (log-distance in km) ─────────────────
    # Single log-distance term as the primary spatial deterrent.
    "c_log_dist":           ("log(od.DIST+1)",              None),
    # Additional linear distance penalty helps suppress unrealistic far-tail
    # destinations that still survive a pure log-distance deterrence.
    "c_dist":               ("od.DIST",                     None),

    # ── Piecewise distance spline + gravity decay ────────────────────────
    # These are FLAT-only terms: their columns (dist_0_2/2_5/5_10, dist_decay,
    # dist_decay_slow) are precomputed in _dest_tree and consumed via
    # DEST_UEC_FLAT. They are deliberately NOT declared here in the DataTree UEC
    # because bare column names / exp() do not resolve in the node-qualified
    # DataTree expression compiler (that produced a zero-gradient, unidentified
    # destination model on the DataTree estimation path). GA-gated by coeff name.

    # ── CBD / urban-core attractor (ActivitySim is_cbd size adjustment) ───
    # Extra pull toward central high-activity zones beyond the size term.
    "c_cbd_dest":           ("emp.is_cbd",                  None),

    # ── Crash safety deterrent ────────────────────────────────────────────
    "c_crash":              ("log(od.CRASH+1)",             None),
    # Per-agent safety utility × destination crash exposure (GA-gated by coeff name).
    "c_crash_safety_dest":  ("person.safety_sensitivity * log(od.CRASH+1)", None),

    # ── Mode logsum (captures accessibility via all modes) ─────────────────
    "c_logsum":             ("base.logsum",                 None),

    # ── Zone attractiveness (composite employment/activity size term) ──────
    "c_size_term":          ("emp.size_term",               None),

    # ── ACDC Agglomeration (Nelson 1958 / Bernadin) ───────────────────────
    # acdc_aggl[j] = log(Σ_{k≠j} size[k] × exp(-λ × auto_time[j,k]) + ε)
    # Expected sign: POSITIVE (agglomeration = nearby complementary zones).
    # NOTE: Held at starting value 0.20 until Hessian is confirmed invertible.
    "c_aggl":               ("emp.acdc_aggl",               None),

    # ── ACDC Competition ─────────────────────────────────────────────────
    # acdc_comp[j] = competition accessible FROM zone j.
    # Expected sign: NEGATIVE.  Dropped from default UEC because it is
    # collinear with c_aggl when both are derived from the same size_term
    # and the same time matrix.  Re-enable only if both have non-NaN se.
    "c_comp":               ("emp.acdc_comp",               None),

    # ── Themed ACDC Agglomeration variants ────────────────────────────────
    # Mutex group: at most ONE of {c_aggl, c_aggl_retail, c_aggl_office,
    # c_aggl_edu, c_aggl_leisure} may be active in any chromosome.
    # Each uses the same decay kernel but size-weighted by a theme-restricted
    # attractor column (retail employment, office employment, education seats,
    # leisure/sport/entertainment facilities). Competition stays unweighted.
    # Sign: all expected POSITIVE.
    "c_aggl_retail":        ("emp.acdc_aggl_retail",        None),
    "c_aggl_office":        ("emp.acdc_aggl_office",        None),
    "c_aggl_edu":           ("emp.acdc_aggl_edu",           None),
    "c_aggl_leisure":       ("emp.acdc_aggl_leisure",       None),
    "c_aggl_parks":         ("emp.acdc_aggl_parks",         None),
    "c_aggl_health":        ("emp.acdc_aggl_health",        None),
    "c_aggl_food":          ("emp.acdc_aggl_food",          None),
    "c_aggl_civic":         ("emp.acdc_aggl_civic",         None),

    # ── Home-pull gravity ────────────────────────────────────────────────
    # home_pull_j = -log(1 + dist(home_zone, j)) — negative because farther
    # from home = less pull. Two variants:
    #   c_home_pull     : scaled by stage-4 departure-time weight (peak=1.0,
    #                     off-peak=0.3), so late-afternoon discretionary tours
    #                     get pulled homeward; morning tours don't.
    #   c_home_pull_late: scaled by remaining-day-budget from stages 2-3
    #                     (fraction of day left), so tours with little time
    #                     remaining are most homeward-biased.
    # Both are person×zone interaction terms — they vary per person because
    # dist(home_i, j) is different for each person i.
    # Expected sign: POSITIVE for c_home_pull (_late) — attraction grows as
    # the day progresses, so closer-to-home zones get higher utility late.
    "c_home_pull":          ("base.home_pull",              ["nonmand"]),
    "c_home_pull_late":     ("base.home_pull_late",         ["nonmand"]),

    # ── Person × distance interactions ────────────────────────────────────
    # Collinearity note: three correlated person variables (incp, age5p, vehrd)
    # all multiplied by the same log(DIST) creates a near-singular block.
    # Keep ONLY the two most behaviorally distinct: income (positive) and
    # vehicle ownership (positive = car-rich travel farther).  Drop age
    # (correlated with both incp and vehrd in SEQ data).
    "c_incp_dist":          ("person.incp * log(od.DIST+1)",        None),
    "c_age_dist":           ("person.age5p * log(od.DIST+1)",       None),
    "c_veh_dist":           ("person.vehrd * log(od.DIST+1)",       None),
    "c_female_dist":        ("(person.sexp == 2) * log(od.DIST+1)", None),
    "c_nprd_dist":          ("person.nprd * log(od.DIST+1)",        None),

    # ── Zone-level accessibility measures ─────────────────────────────────
    # Collinearity fix: auto_work_acc and walk_acc were duplicated via the
    # worker-type conditioning terms below.  Keep ONE coefficient per
    # accessibility dimension; the conditioning is absorbed into the
    # coefficient sign (positive for workers, attenuated for non-workers).
    #
    # Also: transit_acc is highly correlated with walk_acc in SEQ (both
    # proxy for urban density).  We keep both for edu/nonmand now because
    # the validation is still missing close-in active/transit destinations.
    "c_auto_work_acc":      ("emp.auto_work_acc",           ["work"]),
    "c_auto_shop_acc":      ("emp.auto_shop_acc",           ["nonmand"]),
    "c_walk_acc":           ("emp.walk_acc",                ["nonmand", "edu"]),
    "c_transit_acc":        ("emp.transit_acc",             ["nonmand", "edu"]),
    "c_noveh_walk_acc":     ("emp.walk_acc",                ["nonmand", "edu"]),
    "c_noveh_transit_acc":  ("emp.transit_acc",             ["nonmand", "edu"]),

    # ── Income × work accessibility ────────────────────────────────────────
    # Higher-income workers prefer car-accessible employment zones.
    # incp is continuous (band 1-14); interaction avoids collinearity with
    # c_auto_work_acc (which captures zone-level accessibility unconditionally).
    # Expected sign: POSITIVE.
    "c_incp_work_acc":      ("person.incp * emp.auto_work_acc",     ["work"]),

    # ── Full-time vs part-time distance conditioning ───────────────────────
    # FT workers (WORKSTAT=1) travel farther to work than PT workers (WORKSTAT=2).
    # Conditioning in DEST_COND restricts this term to FT workers only.
    # Expected sign: POSITIVE (FT workers accept longer commutes).
    "c_ftwork_dist":        ("person.WORKSTAT * log(od.DIST+1)",    ["work"]),

    # ── Occupation × distance for work trips ──────────────────────────────
    # White-collar workers (occp_group=1) travel farther to work.
    # Conditioning is in DEST_COND below.
    "c_occp_dist":          ("person.occp_group * log(od.DIST+1)",  ["work"]),

    # ── Occupation sub-group distance interactions ─────────────────────────
    # Professional/managerial (occp_mgr) travel farthest; trades (occp_trades)
    # have intermediate sensitivity. Both are binary dummies derived from occp_group.
    "c_occp_mgr_dist":      ("person.occp_mgr * log(od.DIST+1)",    ["work"]),
    "c_occp_trades_dist":   ("person.occp_trades * log(od.DIST+1)", ["work"]),

    # ── Industry-specific zone attractors - work only ─────────────────────
    # FIX: Previously all three used "emp.size_term" → identical columns
    # → perfectly collinear → singular Hessian → NaN std errors.
    # Each now references a DISTINCT landuse column so the Hessian is full-rank.
    # Conditioning via DEST_COND restricts each term to the relevant industry.
    "c_indp_finance":       ("emp.FPSEMPN",                 ["work"]),
    "c_indp_health":        ("emp.HEREMPN",                 ["work"]),
    "c_indp_retail":        ("emp.RETEMPN",                 ["work"]),
    "c_totemp_work":        ("emp.TOTEMP",                  ["work"]),
    # ── Additional industry-sector work attractors (log employment) ────────
    # More "indp" attractor types the GA can select for work destinations.
    "c_indp_agri":          ("log(emp.AGREMPN+1)",          ["work"]),
    "c_indp_mfg":           ("log(emp.MWTEMPN+1)",          ["work"]),
    "c_indp_construction":  ("log(emp.construction+1)",     ["work"]),
    "c_indp_govt":          ("log(emp.government+1)",        ["work"]),
    "c_indp_industrial":    ("log(emp.industrial+1)",       ["work"]),
    "c_indp_education":     ("log(emp.Education+1)",         ["work"]),

    # ── Education zone attractors - edu only ──────────────────────────────
    # FIX: Previously all three used "emp.size_term" → perfectly collinear.
    # Each now references a DISTINCT enrolment/facility column.
    "c_edu_post":           ("emp.Education",               ["edu"]),
    "c_edu_primary":        ("emp.Primary",                 ["edu"]),
    "c_edu_secondary":      ("emp.Secondary",               ["edu"]),

    # ── Non-mandatory destination attractors ───────────────────────────
    "c_nonmand_res":        ("emp.residential",             ["nonmand"]),
    "c_nonmand_ent":        ("emp.Entertainment",           ["nonmand"]),
    "c_nonmand_com":        ("emp.commercial",              ["nonmand"]),
    "c_nonmand_retail":     ("emp.retail",                  ["nonmand"]),

    # ── Travel propensity × distance (CDAP logsum proxy) ─────────────────
    # travel_propensity_i is computed from person attributes as a normalised
    # score capturing each person's propensity for mandatory/longer travel
    # (see _travel_propensity_score).  Interacted with log(dist) so that
    # more-mobile people are less deterred by distance - directly addressing
    # the homogeneous distance sensitivity that causes unrealistic clustering.
    # Expected sign: POSITIVE (high propensity → less distance aversion → +β).
    "c_propensity_dist":    ("person.travel_propensity * log(od.DIST+1)", None),

    # ── Frozen person distance draw (random heterogeneity, work only) ─────
    # person_dist_draw_i is set at prediction time to a frozen N(0, s_log_dist)
    # draw seeded by person_id (see _frozen_person_dist_offsets).
    # During HHTS estimation this column is all zeros - the heterogeneity is
    # absorbed into s_log_dist, not estimated as a fixed effect.
    # At prediction it shifts each person's effective distance sensitivity
    # by their draw, making some consistently short-distance and others long.
    "c_person_dist_draw":   ("person.person_dist_draw * log(od.DIST+1)",  ["work"]),
}

# ── Conditioning expressions (ActivitySim interaction term pattern) ───────────
# Applied as multipliers: P(coeff) * X(expr) * X(condition)
# The condition is evaluated per-person using the person sub-dataset relationship.
DEST_COND = {
    # Industry conditioning - use OR of individual equality checks.
    # Sharrow cannot compile Python 'in [list]' syntax; use boolean OR instead.
    # FPSEMPN: finance/professional (indp codes 10-13)
    "c_indp_finance":   ("(person.indp == 10) | (person.indp == 11) "
                         "| (person.indp == 12) | (person.indp == 13)"),
    # HEREMPN: health/education (indp codes 16-17)
    "c_indp_health":    "(person.indp == 16) | (person.indp == 17)",
    # RETEMPN: retail/hospitality (indp codes 7-8)
    "c_indp_retail":    "(person.indp == 7) | (person.indp == 8)",

    # Study level conditioning (STUDYING: 1=primary, 2=secondary, 3-4=post)
    "c_edu_post":       "(person.STUDYING == 3) | (person.STUDYING == 4)",
    "c_edu_primary":    "(person.STUDYING == 1)",
    "c_edu_secondary":  "(person.STUDYING == 2)",

    # Worker-type conditioning (WORKSTAT: 0=not working, 1=FT, 2=PT)
    # FT workers (WORKSTAT=1) activate the c_ftwork_dist distance term.
    "c_ftwork_dist":    "(person.WORKSTAT == 1)",

    # Occupation conditioning (occp_group 1 = white-collar)
    "c_occp_dist":      "(person.occp_group == 1)",
    # Sub-group dummies: occp_mgr=1 when professional/managerial, occp_trades=1 for trades
    "c_occp_mgr_dist":    "(person.occp_mgr == 1)",
    "c_occp_trades_dist": "(person.occp_trades == 1)",
    "c_noveh_walk_acc": "(person.vehrd <= 0)",
    "c_noveh_transit_acc": "(person.vehrd <= 0)",
}
# ===========================================================================
# SIZE TERMS  (ActivitySim tour_destination_size_terms pattern)
# ===========================================================================
#
# ActivitySim uses a zone × purpose size term matrix as the attractor in
# destination choice.  Rather than embedding 20+ individual landuse columns
# in the utility expression, we compute ONE composite size term per
# (zone, purpose) pair and reference it as a single utility term.
#
# Formula (matches ActivitySim default):
#   size_term(zone, purpose) = sum over categories of:
#       landuse_col ^ coefficient
#
# Coefficients by purpose - mirrors ActivitySim size_terms.csv pattern.
# To change the attractor: edit this dict. No utility expression changes needed.
# ===========================================================================

SIZE_TERM_COEFFICIENTS = {
    # purpose:  {landuse_col: exponent, ...}
    #
    # Theoretical basis: each job/school place is equally likely to attract a
    # traveller, so the correct exponent for a job-count or school-place column
    # is 1.0 (linear opportunities-to-probability relationship, IIA-consistent).
    # Compression < 1.0 was previously used to prevent CBD zones from dominating,
    # but that is wrong: the distance deterrence term (c_log_dist) and mode logsum
    # naturally reduce probability for distant CBD zones for each individual
    # traveller.  Compressing the size term instead makes the model structurally
    # unable to predict large employment centres like zone 40 correctly.
    #
    # Additional sector columns (FPSEMPN, commercial) capture the QUALITY mix of
    # employment that raw TOTEMP misses:
    #   - zone 40: TOTEMP=15 419 (not highest) but FPSEMPN=5 600 (highest in SEQ)
    #              and commercial=777 k m² (3× next zone) → finance/CBD district
    #   - zone 182: TOTEMP=17 416 (higher) but far fewer observed work trips
    # These sector terms must carry large enough exponents to actually differentiate
    # between zones with similar raw employment but different sector composition.
    'work': {
        'TOTEMP':    1.0,    # linear - each job is one opportunity
        'FPSEMPN':   0.85,   # finance/prof-services: zone 40=5 600 (near SEQ highest);
                             # raised from 0.75 to better differentiate CBD finance zones
        'HEREMPN':   0.35,   # healthcare employment - sector attractor
        'RETEMPN':   0.25,   # retail employment - minor work attractor
        # 'commercial' REMOVED from SIZE_TERM: raw m² has 89M data error at zone 158
        # (p99=1.06M; error is 84× p99).  Commercial is instead used as a direct log-
        # transformed term (c_log_commercial_work / log_commercial_1) with a p99 cap
        # applied in _dest_tree() so the outlier does not distort MLE estimation.
    },
    'edu': {
        'Primary':   1.0,    # number of primary school places - linear
        'Secondary': 1.0,    # number of secondary school places - linear
        'Education': 0.9,    # tertiary enrolment - near-linear; slight compression
                             # for very large campuses that would otherwise dominate
                             # all tertiary trips regardless of distance
        'HSENROLL':  0.85,   # supplementary high-school enrolment count
        'university': 1.0,   # OSM university building count (27 nz); zone 40 = 5 (QUT, TAFE CBD)
                             # Primary=0, Secondary=0 for zone 40 → edu rank 259/330 without this
        'college':    1.0,   # OSM community college / TAFE POI count (38 nz)
                             # fills vocational-education gap not captured by school enrolments
        'TOTPOP':    0.15,   # population baseline: ensures zones with unrecorded schools
                             # (Primary=Secondary=0 but real observed trips, e.g. zones 248, 257)
                             # retain non-zero size so the model can still predict trips there.
                             # Low exponent keeps the signal weak; school-count columns dominate
                             # in zones that ARE in the school dataset.
    },
    'social': {
        # Exponents are calibrated parameters (ActivitySim size_terms pattern):
        # exp=1.0 → linear attraction; exp<1.0 → compressed (diminishing returns for
        # very large venues); compute_size_terms() applies sum(col^exp) then log-demeans.
        'Entertainment':  0.8,   # purpose-built entertainment land use (327 nz)
        'residential':    0.5,   # social trips stay near home
        'sports_centre':  0.7,   # OSM sport/recreation venues (247 nz) - key social activity
        'fitness_centre': 0.5,   # OSM gyms and fitness studios (186 nz)
        'RESACRE':        0.3,   # residential parcel acreage - direct measure of "how much
                                  # residential area is here" for a home-adjacent purpose;
                                  # only present in the 357-zone SEQ+Tweed landuse rebuild
                                  # (new_pop/saved_data/zone_landuse_357.csv), not the legacy
                                  # zone_order_and_landuse.csv - a no-op (col absent, skipped)
                                  # until the pipeline is switched to that source.
    },
    'escort': {
        'ESCORT':       0.8,   # purpose-built childcare/school places aggregate (329 nz)
        'residential':  0.5,   # escort trips co-located with residential
        'kindergarten': 0.9,   # OSM kindergarten/childcare POI count (222 nz)
                               # 'childcare' in zone_order_and_landuse has only 1 non-zero
                               # zone (apparent data error); kindergarten is the reliable signal
        'RESACRE':      0.3,   # residential parcel acreage - see 'social' note (357-zone rebuild only)
    },
    'shopping': {
        'Shopping':  0.8,    # aggregate retail POI count (313 nz)
        'retail':    0.5,    # retail land use area
        'RETEMPN':   0.3,    # retail employment
        'CIACRE':    0.3,    # commercial/industrial parcel acreage - a direct measure of
                             # retail/commercial land extent, complementing employment-count
                             # based RETEMPN; 357-zone rebuild only (see 'social' note above).
    },
    'othmaint': {
        'OTHERMAINTENANCE': 1.0,  # purpose-built maintenance service facilities (263 nz)
        'Healthcare':       0.8,  # healthcare POI aggregate in lu (245 nz)
        'clinic':           0.7,  # OSM clinics (152 nz)
        'pharmacy':         0.6,  # OSM pharmacies (207 nz)
        'bank':             0.5,  # OSM banks (108 nz)
        'CIACRE':           0.2,  # commercial/industrial parcel acreage - personal-business
                                  # errands (banks/clinics/services) cluster in commercial
                                  # areas; 357-zone rebuild only (see 'social' note above).
        # 'commercial' REMOVED: same 89M m² data error at zone 158 as in work size term
    },
    'othdiscr': {
        'OTHERDISCRFacilities': 1.0,  # purpose-built discretionary facilities (313 nz)
        'Entertainment':        0.3,  # entertainment land use
        'residential':          0.3,  # discretionary trips distributed near home
        'sports_centre':        0.6,  # OSM sport/recreation (247 nz)
        'fitness_centre':       0.5,  # OSM gyms (186 nz)
    },
    'eat': {
        'EATOUT':     1.0,   # aggregate eating-out attractor (315 nz) - primary signal
        'restaurant': 0.7,   # OSM restaurant count (260 nz) - explicit venue signal
        'cafe':       0.6,   # OSM cafe count (282 nz)
        'pub':        0.4,   # OSM pub count - pub meals (205 nz)
        'retail':     0.3,   # retail mix correlates with food-court density
    },
    # Aggregate entry for 'nonmand' segment - combines attractors from all
    # non-mandatory sub-purposes so compute_size_terms(purposes=['nonmand'])
    # returns a meaningful fallback when dest_model.probability() is unavailable.
    'nonmand': {
        'Entertainment':        0.8,
        'residential':          0.5,
        'Shopping':             0.8,
        'retail':               0.5,
        'RETEMPN':              0.3,
        'EATOUT':               0.5,
        'OTHERMAINTENANCE':     0.7,
        'OTHERDISCRFacilities': 0.7,
        'ESCORT':               0.5,
        'Healthcare':           0.4,   # replaces removed 'commercial'
        'sports_centre':        0.4,   # OSM sport/recreation
        'kindergarten':         0.4,   # childcare places (escort sub-trips)
        # 'commercial' REMOVED: 89M m² data error at zone 158
    },
}

# Segment → list of purposes it covers
SEGMENT_PURPOSES = {
    'work':    ['work'],
    'edu':     ['edu'],
    'nonmand': ['social', 'escort', 'shopping', 'othmaint', 'othdiscr', 'eat'],
}


def compute_size_terms(landuse: pd.DataFrame,
                       purposes: list = None) -> pd.DataFrame:
    """
    Compute zone × purpose size term matrix.

    This is the ActivitySim tour_destination_size_terms pattern.
    Returns a DataFrame indexed by zone with one column per purpose.
    Each cell = sum(landuse_col ^ coeff) for all cols in SIZE_TERM_COEFFICIENTS[purpose].
    Zero is replaced with a small epsilon so log(size) is always defined.

    Parameters
    ----------
    landuse : zone-indexed landuse DataFrame (from sc.get_land())
    purposes : list of purposes to compute - defaults to all in SIZE_TERM_COEFFICIENTS

    Returns
    -------
    DataFrame(index=zone_id, columns=purposes), values = log(size_term)
    """
    if purposes is None:
        purposes = list(SIZE_TERM_COEFFICIENTS.keys())

    # ensure zone is the index
    lu = landuse.copy()
    zone_col = next((c for c in ['ORDER', 'zone_id', 'TAZ', lu.columns[0]]
                     if c in lu.columns), None)
    if zone_col and lu.index.name != zone_col:
        lu = lu.set_index(zone_col)

    result = {}
    for purpose in purposes:
        coeff_map = SIZE_TERM_COEFFICIENTS.get(purpose, {})
        size = pd.Series(0.0, index=lu.index)
        for col, exp in coeff_map.items():
            if col in lu.columns:
                vals = pd.to_numeric(lu[col], errors='coerce').fillna(0).clip(lower=0)
                size += vals ** exp
        # log-transform with epsilon floor (matches ActivitySim log_size pattern)
        log_size = np.log(size.clip(lower=1e-6))
        # ── DEMEANING (critical for zone 1 bias) ─────────────────────────
        # Without centering, zone 1 (CBD, highest employment) gets the
        # largest log(size) and with c_size_term > 0 dominates all trips.
        # Subtracting the cross-zone mean converts log(size) to a relative
        # attractiveness index - the marginal attractiveness of each zone
        # relative to the average zone.  This is the ActivitySim UEC pattern
        # where the coefficient on log(size) reflects the elasticity, not
        # the absolute scale.  The coefficient c_size_term ≈ 1.0 then means
        # "double the size → double the probability" (IIA-consistent).
        result[purpose] = log_size - float(log_size.mean())

    df = pd.DataFrame(result)
    ranges = {p: f"[{df[p].min():.2f}, {df[p].max():.2f}]" for p in purposes}
    means = {p: round(float(df[p].mean()), 2) for p in purposes}
    logger.info(
        f"compute_size_terms: {len(df)} zones × {len(purposes)} purposes | "
        f"size term ranges (min,max per purpose): {ranges} | "
        f"means ≈ 0 after demeaning (by construction): {means}"
    )
    return df


class SizeTermMatrix:
    """
    Lightweight zone × purpose size term accessor.
    Mirrors ActivitySim's DataFrameMatrix pattern for use in utility expressions.
    get(zone_ids, purpose) returns a vector of log-size-terms for those zones.
    """

    def __init__(self, df: pd.DataFrame):
        self.df   = df          # indexed by zone_id, columns = purposes
        self._arr = df.values
        self._zone_idx = {int(z): i for i, z in enumerate(df.index)}
        self._purp_idx = {p: i for i, p in enumerate(df.columns)}
        # Fallback index for unmapped zones: use median row (≈ average zone),
        # NOT 0 (first zone = zone 1 = CBD → causes zone-1 attraction bias).
        self._fallback_idx = len(df) // 2

    def get(self, zone_ids, purpose: str) -> np.ndarray:
        """Return log-size-term for each zone_id for the given purpose."""
        pidx = self._purp_idx.get(purpose, None)
        if pidx is None:
            return np.zeros(len(zone_ids))
        return np.array([
            self._arr[self._zone_idx.get(int(z), self._fallback_idx), pidx]
            for z in zone_ids
        ])

    def get_for_zones(self, zone_ids: np.ndarray,
                      purpose: str) -> pd.Series:
        vals = self.get(zone_ids, purpose)
        return pd.Series(vals, index=zone_ids)


# ===========================================================================
# PREPROCESSORS  (ActivitySim annotate_preprocessors pattern)
# ===========================================================================
#
# Preprocessors compute derived columns on the choosers table BEFORE the
# utility expressions are evaluated.  This is cleaner than embedding
# conditional logic inside utility expressions.
#
# Each preprocessor is a function:
#   preprocess_tours(tours, skim_df, persons) -> tours with added columns
#
# Mirrors ActivitySim's PREPROCESSOR block in yaml settings, but in Python.
# ===========================================================================

def preprocess_tours(tours: pd.DataFrame,
                     skim_df: pd.DataFrame = None,
                     persons: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add derived columns to tours before model estimation/prediction.

    Mirrors ActivitySim's preprocessor annotations.  All columns computed
    here are accessible as tour.xxx in the DataTree utility expressions.

    Columns added
    -------------
    _time_period   : integer departure period (0=early … 5=late)
    _log_incp      : log(incp+1) - for log-income utility terms
    _no_vehicle    : 1 if HH has zero vehicles
    _has_vehicle   : 1 if HH has ≥1 vehicle
    _peak          : 1 if departure in AM peak (period==1)
    outbound_stops : 0 if missing (default)
    n_shopping_out : 0 if missing (default)
    bikes          : 0 if missing (default - HH bicycle count)
    age5p          : 6 if missing (default 25-29yr age band)
    CARLICENCE     : 1 if missing (assume licensed)
    occp_group     : 0 if missing (not working)
    """
    t = tours.copy()

    # ── time period ───────────────────────────────────────────────────────
    if 'STARTIME' in t.columns:
        def _period(m):
            m = float(m) if not pd.isna(m) else 480
            if m < 360:  return 0   # early
            if m < 540:  return 1   # am_peak
            if m < 900:  return 2   # midday
            if m < 1080: return 3   # pm_peak
            if m < 1260: return 4   # evening
            return 5                # late
        t['_time_period'] = t['STARTIME'].apply(_period).astype(int)
    elif '_time_period' not in t.columns:
        t['_time_period'] = 1

    t['_peak'] = (t['_time_period'] == 1).astype(int)

    # ── income ────────────────────────────────────────────────────────────
    if 'incp' in t.columns:
        t['_log_incp'] = np.log1p(pd.to_numeric(t['incp'], errors='coerce').fillna(8))
    else:
        t['_log_incp'] = np.log1p(8)

    # ── vehicle ownership ─────────────────────────────────────────────────
    if 'vehrd' in t.columns:
        vehrd = pd.to_numeric(t['vehrd'], errors='coerce').fillna(1)
        t['_no_vehicle']  = (vehrd == 0).astype(int)
        t['_has_vehicle'] = (vehrd >= 1).astype(int)
    else:
        t['_no_vehicle']  = 0
        t['_has_vehicle'] = 1

    # ── safe defaults for any person-attribute columns that may be absent ─
    # (these are accessed as tour.xxx for the UEC preprocessor path -
    #  the DataTree person.xxx path is the primary source, but having them
    #  on the tour table as well avoids errors in some Larch API paths)
    for col, default in [('bikes', 0), ('age5p', 6),
                          ('CARLICENCE', 1), ('occp_group', 0),
                          ('outbound_stops', 0), ('n_shopping_out', 0),
                          ('transit_prop', 0.0), ('car_prop', 0.0), ('bike_prop', 0.0)]:
        if col not in t.columns:
            t[col] = default

    return t


# ===========================================================================
# POST-PROCESSORS  (ActivitySim annotate_tables pattern)
# ===========================================================================

def postprocess_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """
    Add summary columns to the final plan after prediction.
    Mirrors ActivitySim's annotate_tables step.

    Adds:
      tour_distance_band  : 'local' | 'regional' | 'long'
      mode_group          : 'motorised' | 'active' | 'transit'
      time_period         : named time period of trip departure
    """
    p = plan.copy()

    # mode group (matches ActivitySim mode_choice_logsum grouping)
    mode_groups = {1: 'motorised', 2: 'motorised',
                   3: 'active',    4: 'active',
                   5: 'transit'}
    if 'mode' in p.columns:
        p['mode_group'] = p['mode'].map(mode_groups).fillna('motorised')

    # distance band
    if 'dist' in p.columns:
        p['tour_distance_band'] = pd.cut(
            p['dist'].fillna(0),
            bins=[-1, 5, 20, 9999],
            labels=['local', 'regional', 'long']
        ).astype(str)

    # time period label
    period_labels = {0: 'early', 1: 'am_peak', 2: 'midday',
                     3: 'pm_peak', 4: 'evening', 5: 'late'}
    if 'STARTIME' in p.columns:
        def _label(m):
            m = float(m) if not pd.isna(m) else 480
            if m < 360:  return 'early'
            if m < 540:  return 'am_peak'
            if m < 900:  return 'midday'
            if m < 1080: return 'pm_peak'
            if m < 1200: return 'evening'
            return 'late'
        p['time_period'] = p['STARTIME'].apply(_label)

    return p




def _build_mode_utility(m, Mode, segment: str,
                         uec: dict = None) -> object:
    """
    Build mode utility from UEC dict.
    Mirrors ActivitySim's tour_mode_choice_simulate pattern.

    Each entry in uec:
        coeff_name: (expression, {mode_name: segments_list_or_None})

    segment_list=None means active for all segments.
    """
    if uec is None:
        uec = MODE_UEC

    # accumulate terms per mode using None as empty sentinel
    # lx.LinearFunction() doesn't exist - P*X returns the correct type,
    # we just can't initialise an empty one; use None and replace on first term
    mode_terms: dict = {name: None for name in MODE_MAP}

    for coeff, (expr, mode_filter) in uec.items():
        # ── Skip purpose-interaction dummies in single-purpose segments ──────
        # Terms like (tour.main_purpose == 'work') / (tour.main_purpose == 'edu')
        # are CONSTANT within a per-purpose segment: all-1 for the segment's own
        # purpose (redundant with the ASC) or all-0 for any other purpose. Either
        # way the column has zero variance → zero gradient → the parameter never
        # moves and pollutes the model (this is why the 'work' mode model was
        # carrying dead c_edu_* / c_work_* terms). These dummies only carry
        # signal in a POOLED model (segment == 'all'); drop them otherwise so a
        # work mode model never contains edu components and vice-versa.
        if segment != 'all' and 'main_purpose ==' in expr:
            continue

        if mode_filter is None:
            targets = {n: None for n in MODE_MAP}
        else:
            targets = mode_filter

        for mode_name, segs in targets.items():
            if segs is not None and segment not in segs:
                continue
            if mode_name not in mode_terms:
                continue
            term = P(coeff) * X(expr)
            if mode_terms[mode_name] is None:
                mode_terms[mode_name] = term
            else:
                mode_terms[mode_name] = mode_terms[mode_name] + term

    for mode_name, utility in mode_terms.items():
        if utility is not None:
            alt_id = MODE_MAP[mode_name]
            m.utility_co[alt_id] = utility

    return m


def _rho_sq_null(model, ll: float) -> float:
    """
    McFadden rho-squared vs. the null (equal-shares) model: 1 - LL/LL_null.

    Larch 6.x has no ``rho_sq_null()`` convenience method (confirmed via
    ``hasattr(larch.Model, 'rho_sq_null')`` -> False) - every call site that
    used to call it silently caught the resulting AttributeError and printed
    rho2=nan for every single model. Compute it directly from
    ``loglike_null()``, which IS available, instead.
    """
    try:
        if ll is None or (isinstance(ll, float) and np.isnan(ll)):
            return float('nan')
        ll_null = float(model.loglike_null())
        if ll_null == 0 or not np.isfinite(ll_null):
            return float('nan')
        return 1.0 - (float(ll) / ll_null)
    except Exception:
        return float('nan')


def _pf_set(model, param: str, field: str, value):
    """
    Safely set a parameter field in a Larch model's pf DataFrame.

    In Larch 6.x with pandas Copy-on-Write, ``model.pf.loc[param, field] = val``
    silently fails because ``pf`` returns a copy.  This helper uses the Larch
    API (``set_value`` / ``lock_value``) when possible, and for bounds
    (minimum / maximum) writes directly to the internal ``_parameter_bucket``
    arrays which are the true source of truth for ``pf``.
    """
    if param not in model.pf.index:
        return
    if field == 'value':
        try:
            model.set_value(param, float(value))
            return
        except Exception:
            pass
    elif field == 'holdfast' and value:
        try:
            cur = float(model.pf.loc[param, 'value'])
            model.lock_value(param, cur if np.isfinite(cur) else 0.0)
            return
        except Exception:
            pass
    elif field in ('minimum', 'maximum'):
        # Write directly to _parameter_bucket arrays (requires unmangle first)
        try:
            model.unmangle()
            bucket = model._parameter_bucket
            names = list(bucket.pnames)
            idx = names.index(param)
            if field == 'minimum':
                bucket.pminimum[idx] = float(value)
            else:
                bucket.pmaximum[idx] = float(value)
            return
        except (ValueError, AttributeError, IndexError):
            pass
    # Last resort - may silently fail under Copy-on-Write but worth trying
    try:
        model.pf.loc[param, field] = value
    except Exception:
        pass


def _build_dest_utility(mm, segment: str,
                         uec: dict = None,
                         cond: dict = None) -> object:
    """
    Build destination utility from UEC dict.
    Mirrors ActivitySim's non_mandatory_destination / work_location pattern.

    Each entry in uec:
        coeff_name: (expression, segments_list_or_None)
    cond provides optional conditioning expressions per coefficient.
    """
    if uec  is None: uec  = DEST_UEC
    if cond is None: cond = DEST_COND

    # accumulate with None sentinel - lx.LinearFunction() doesn't exist
    utility = None

    for coeff, (expr, segs) in uec.items():
        if segs is not None and segment not in segs:
            continue

        condition = cond.get(coeff)
        if condition:
            term = P(coeff) * X(expr) * X(condition)
        else:
            term = P(coeff) * X(expr)

        if utility is None:
            utility = term
        else:
            utility = utility + term

    if utility is not None:
        mm.utility_ca = utility
    return mm


# ── Shims so existing _mode_spec / _dest_spec callers still work ─────────────

def _mode_spec(m, segment: str, Mode):
    """Wrapper - delegates to UEC-based builder."""
    return _build_mode_utility(m, Mode, segment)


def _dest_spec(mm, segment: str):
    """Wrapper - delegates to UEC-based builder."""
    return _build_dest_utility(mm, segment)


# ===========================================================================
# MODEL BUILD  (ActivitySim-style: one model per segment, shared spec)
# ===========================================================================
#
# ActivitySim pattern from tour_mode_choice_simulate / trip_destination:
#   for tour_purpose, tours_segment in tours.groupby("tour_purpose"):
#       choices = run_tour_mode_choice_simulate(tours_segment, tour_purpose, ...)
#       choices_list.append(choices)
#   choices_df = pd.concat(choices_list)
#
# SEGMENT_COEFFICIENTS mirrors ActivitySim's get_segment_coefficients:
#   {segment: {coeff_name: init_value}}
# These are starting values only - the estimator refines them.
# Any coefficient not listed inherits the UEC default (0.0).
# ===========================================================================

# ── Per-segment coefficient starting values ───────────────────────────────────
SEGMENT_COEFFICIENTS: dict = {
    "work": {
        # ── Dest: critical distance/logsum starting values ──────────────────
        # c_log_dist defaults to 0 but is bounded <= -0.001; starting at 0
        # means the optimizer begins with no distance deterrence, making all
        # zones equally likely. A good starting point avoids bad local optima.
        "c_log_dist":        -2.0,  # log-distance deterrent (work: workers accept longer commutes)
        "c_logsum":           0.5,  # mode logsum (positive = accessible zones preferred)
        "c_crash":           -0.1,  # crash deterrent
        # ── Mode: distance / time ─────────────────────────────────────────
        "c_ivt_auto":        -1.20,
        "c_ivt_transit":     -0.90,
        "c_crash_veh":        0.0,
        # ── Mode: vehicle ownership ───────────────────────────────────────
        "c_veh_da":           0.30,
        "c_veh_sr":           0.10,
        "c_noveh_transit":    2.0,
        "c_noveh_walk":       1.5,
        # ── Mode: licence - now HARD availability, not a utility coefficient ─
        # ── Mode: income ─────────────────────────────────────────────────
        "c_incp_da":          0.010,
        "c_incp_sr":          0.005,
        "c_incp_transit":    -0.015,
        "c_incp_active":     -0.010,
        # ── Mode: age ────────────────────────────────────────────────────
        "c_age_walk":        -0.10,
        "c_age_bike":        -0.15,
        "c_young_bike":       0.50,
        # ── Mode: household ──────────────────────────────────────────────
        "c_hhsize_sr":        0.10,
        "c_joint_sr":         0.35,
        "c_bikes_bike":       0.40,
        # ── Mode: sex ────────────────────────────────────────────────────
        "c_female_transit":   0.30,
        "c_female_walk":      0.20,
        # ── Mode: worker type ────────────────────────────────────────────
        "c_ftwork_da":        0.50,
        "c_ptwork_sr":        0.20,
        # ── Mode: time-of-day ────────────────────────────────────────────
        "c_peak_da":          0.30,
        # ── Mode: active distances ───────────────────────────────────────
        "c_walk_time":        -1.50,
        "c_bike_time":        -1.00,
        # ── Mode: stops (each stop adds out-of-vehicle handling burden) ──
        "c_walk_stops":       -0.30,
        "c_bike_stops":       -0.50,
        # ── Mode: ASCs - calibrated to observed SEQ mode shares ─────────
        # Observed: DA=62.1% Other=23.9% Walk=8.4% Transit=3.9% Bike=1.6%
        # Raw log-share vs DA: Walk=log(8.4/62.1)=-2.0, Transit=log(3.9/62.1)=-2.8
        # Bike=log(1.6/62.1)=-3.7, SR=log(23.9/62.1)=-0.96
        # These are starting points - estimation refines from here.
        # Better initialisation prevents the optimizer from converging to
        # implausible local optima (e.g. ASC_Transit > 0 seen in logs).
        "ASC_Walk":          -2.0,
        "ASC_Bike":          -3.7,
        "ASC_Transit":       -2.8,
        "ASC_SR":            -1.0,
        "c_dist":           -0.035,
        # ── Dest: person × distance interactions ─────────────────────────
        # c_age_dist removed - collinear with incp and vehrd in SEQ data.
        "c_incp_dist":        0.010,
        "c_age_dist":        -0.006,
        "c_veh_dist":         0.008,
        "c_female_dist":     -0.003,
        "c_nprd_dist":        0.004,
        "c_occp_dist":        0.015,
        # ── Dest: accessibility ───────────────────────────────────────────
        "c_auto_work_acc":    0.25,
        "c_walk_acc":         0.0,
        "c_noveh_walk_acc":   0.0,
        "c_noveh_transit_acc": 0.0,
        # ── Dest: ACDC agglomeration (competition removed - collinear) ─────
        "c_aggl":             0.20,
        "c_comp":            -0.08,
        # ── Dest: industry-specific attractors (log-scale columns) ─────────
        # DEST_UEC_FLAT now uses log(emp+1) so values are in [0,11] range;
        # raw TOTEMP (0-50000) with coeff 0.15 → utility ~7500 → saturation.
        "c_indp_finance":     0.30,  # log(FPSEMPN+1) × finance_flag
        "c_indp_health":      0.30,  # log(HEREMPN+1) × health_flag
        "c_indp_retail":      0.20,  # log(RETEMPN+1) × retail_flag
        # c_totemp_work removed (collinear with size_term)
        "c_log_commercial_work": 1.0,   # log(commercial+1) capped at p99; zone 40 (777k) vs avg ~50k
        # ── Dest: travel propensity × distance (CDAP logsum proxy) ──────
        # Positive: more-mobile persons less deterred by distance.
        # Start near zero; estimation identifies the true magnitude.
        "c_propensity_dist":  0.03,
        # ── Dest: frozen person distance draw (zero during estimation) ───
        # coefficient = 1.0 - the column already holds the correctly-scaled
        # draw (N(0, s_log_dist)); a fixed coefficient of 1 applies it directly.
        "c_person_dist_draw": 1.0,
    },
    "edu": {
        # ── Dest: critical distance/logsum starting values ──────────────────
        "c_log_dist":        -2.5,  # edu: students travel shorter distances than workers
        "c_logsum":           0.4,
        "c_crash":           -0.1,
        # ── Mode: distance / time ─────────────────────────────────────────
        "c_ivt_auto":        -1.10,
        "c_ivt_transit":     -0.80,
        "c_crash_veh":        0.0,
        "c_veh_da":           0.20,
        "c_veh_sr":           0.20,
        "c_noveh_transit":    2.0,
        "c_noveh_walk":       1.5,
        "c_incp_da":          0.008,
        "c_incp_sr":          0.005,
        "c_incp_transit":    -0.012,
        "c_incp_active":     -0.005,
        "c_age_walk":        -0.05,
        "c_age_bike":        -0.10,
        "c_young_bike":       0.60,
        "c_hhsize_sr":        0.15,
        "c_joint_sr":         0.30,
        "c_bikes_bike":       0.50,
        "c_female_transit":   0.20,
        "c_female_walk":      0.15,
        "c_ftwork_da":        0.30,
        "c_ptwork_sr":        0.20,
        "c_peak_da":          0.20,
        "c_walk_time":        -1.50,
        "c_bike_time":        -1.00,
        "c_walk_stops":       -0.30,
        "c_bike_stops":       -0.50,
        "ASC_Walk":          -2.0,
        "ASC_Bike":          -3.7,
        "ASC_Transit":       -2.8,
        "ASC_SR":            -1.0,
        "c_dist":           -0.030,
        "c_incp_dist":        0.008,
        "c_age_dist":        -0.010,
        "c_veh_dist":         0.008,
        "c_female_dist":      0.002,
        "c_nprd_dist":        0.002,
        # ── Dest: education zone attractors (distinct columns per term) ────
        "c_edu_post":         0.30,
        "c_edu_primary":      0.40,
        "c_edu_secondary":    0.35,
        "c_edu_prim_priv":    0.30,  # log(private primary+1) × primary student
        "c_edu_sec_priv":     0.30,  # log(private secondary+1) × secondary student
        # ── Dest: edu zone accessibility heuristic ────────────────────────
        "c_auto_edu_acc":     0.20,
        # ── Dest: transit accessibility for edu ───────────────────────────
        "c_walk_acc":         0.12,
        "c_transit_acc":      0.15,
        "c_noveh_walk_acc":   0.18,
        "c_noveh_transit_acc": 0.22,
        # ── Dest: ACDC agglomeration only (competition removed) ────────────
        "c_aggl":             0.15,
        "c_comp":            -0.06,
        # ── Dest: travel propensity × distance (CDAP logsum proxy) ──────
        "c_propensity_dist":  0.03,
    },
    "nonmand": {
        # ── Dest: critical distance/logsum starting values ──────────────────
        "c_log_dist":        -2.5,  # nonmand: shorter trips, more distance-sensitive
        "c_logsum":           0.3,
        "c_crash":           -0.1,
        # ── Mode: distance / time ─────────────────────────────────────────
        "c_ivt_auto":        -1.00,
        "c_ivt_transit":     -0.70,
        "c_crash_veh":        0.0,
        "c_veh_da":           0.25,
        "c_veh_sr":           0.15,
        "c_noveh_transit":    2.0,
        "c_noveh_walk":       1.8,
        # Income: only DA and Transit - SR and active are collinear in nonmand data
        # (all four came out ≈ −3.46 in test runs → multicollinearity, inflated ASCs)
        "c_incp_da":          0.008,
        "c_incp_sr":          0.0,     # held fixed at zero for nonmand
        "c_incp_transit":    -0.010,
        "c_incp_active":      0.0,     # held fixed at zero for nonmand
        "c_age_walk":        -0.08,
        "c_age_bike":        -0.12,
        "c_young_bike":       0.40,
        "c_hhsize_sr":        0.20,
        "c_joint_sr":         0.45,
        "c_bikes_bike":       0.35,
        "c_female_transit":   0.35,
        "c_female_walk":      0.25,
        "c_ftwork_da":        0.30,
        "c_ptwork_sr":        0.25,
        "c_peak_da":          0.15,
        "c_walk_time":        -1.20,
        "c_bike_time":        -0.80,
        "c_walk_stops":       -0.30,
        "c_bike_stops":       -0.50,
        # ── Mode: bike-specific nonmand deterrents ─────────────────────
        "c_bike_veh":        -0.20,    # car ownership reduces biking for errands
        "c_bike_shop":       -0.80,    # carrying shopping strongly deters biking
        "ASC_Walk":          -2.0,
        "ASC_Bike":          -3.7,
        "ASC_Transit":       -2.8,
        "ASC_SR":            -1.0,
        "c_dist":           -0.040,
        "c_incp_dist":        0.012,
        "c_age_dist":        -0.010,
        "c_veh_dist":         0.010,
        "c_female_dist":      0.003,
        "c_nprd_dist":        0.005,
        # ── Dest: accessibility - nonmand uses shopping + transit ─────────
        "c_auto_shop_acc":    0.20,
        "c_auto_nonmand_acc": 0.20,  # zone nonmand accessibility heuristic
        "c_walk_acc":         0.10,
        "c_transit_acc":      0.15,
        "c_noveh_walk_acc":   0.22,
        "c_noveh_transit_acc": 0.25,
        # Log-transformed attractors: coefficients now in ~[0.1, 0.5] range
        # (raw residential/commercial → utility > 7000 → softmax saturation, gradient ≈ 0)
        "c_nonmand_res":      0.30,  # log(residential+1)
        "c_nonmand_ent":      0.40,  # log(Entertainment+1)
        "c_nonmand_com":      0.35,  # log(commercial+1)
        "c_nonmand_retail":   0.40,  # log(retail+1)
        "c_nonmand_totpop":   0.30,  # log(TOTPOP+1) - social/population-based trips
        "c_nonmand_eatout":   0.50,  # log(EATOUT+1) - eat-out venue count
        # ── Dest: ACDC agglomeration only ─────────────────────────────────
        "c_aggl":             0.25,
        "c_comp":            -0.10,
        # ── Dest: travel propensity × distance (CDAP logsum proxy) ──────
        "c_propensity_dist":  0.03,
    },
}


def _apply_segment_coefficients(model, segment: str,
                                  overrides: dict = None):
    """
    Apply per-segment coefficient starting values.
    ActivitySim equivalent: eval_coefficients(state, model_spec, coefficients, estimator)
    """
    coeff_dict = overrides or SEGMENT_COEFFICIENTS.get(segment, {})
    for coeff, value in coeff_dict.items():
        try:
            model.set_value(coeff, value)
        except Exception:
            pass
    return model


# ── Skim variable → UEC expression mapping (for GA variable search) ──────────
# Available OMX skims: SOVTOLL_TIME__AM/PM/EV/MD, SOVTOLL_TOLL__AM/PM/EV/MD
# WLK_LOC_WLK/DRV variants (transit), no DIST/DISTWALK/DISTBIKE
SKIM_TO_EXPR = {
    "SOVTOLL_TIME__AM": "od.SOVTOLL_TIME__AM",
    "SOVTOLL_TIME__PM": "od.SOVTOLL_TIME__PM",
    "SOVTOLL_TIME__MD": "od.SOVTOLL_TIME__MD",
    "SOVTOLL_TIME__EV": "od.SOVTOLL_TIME__EV",
    "SOVTOLL_TOLL__AM": "od.SOVTOLL_TOLL__AM",
    "WLK_LOC_WLK_TOTIVT__AM": "od.WLK_LOC_WLK_TOTIVT__AM",
    "WLK_LOC_WLK_XWAIT__AM":  "od.WLK_LOC_WLK_XWAIT__AM",
    "WLK_LOC_DRV_DTIM__AM":   "od.WLK_LOC_DRV_DTIM__AM",
    "HOV2TOLL_TIME__AM":      "od.HOV2TOLL_TIME__AM",
    "DIST": "od.DIST",
    "DISTWALK": "od.DISTWALK",
    "DISTBIKE" : "od.DISTBIKE",
    "CRASH": "od.CRASH",
}


def _mode_spec_custom(m, Mode, custom_vars: list, segment: str = 'all'):
    """
    GA-restricted mode spec.  Each MODE_UEC term is either:
      - backbone (always on): core IVTs, walk/bike times, ASCs, trip-structure
      - person-attribute (gated): included only when a mapped GA candidate is selected
      - optional skim (gated): included when the skim matrix name is in custom_vars
      - extra skim (dynamic): injected when a candidate absent from MODE_UEC is selected

    Previously ALL non-skim terms were always included regardless of gene selection,
    so the model never changed between GA evaluations.
    """
    import re as _re

    if isinstance(custom_vars, dict):
        # Pass the REAL segment (not "all") so _build_mode_utility drops the
        # constant `main_purpose ==` purpose dummies in single-purpose segments
        # (its filter at the top of the build loop). Passing "all" here was the
        # leak that let dead c_edu_* / c_work_* terms (all-0 columns in a
        # nonmand/work segment) survive into the fit with value 0.0, SE huge and
        # pval 1.0 — exactly the zero-fit pollution the prune step now removes.
        return _build_mode_utility(m, Mode, segment, uec=custom_vars)

    # ── 1. Always-on backbone terms ─────────────────────────────────────────
    # Core skim impedance terms must always be present for model identification.
    # Trip-structure terms (joint travel, stops, time-of-day) are also structural.
    #
    # BEHAVIOURAL BACKBONE: a mode-choice model with only impedance + ASCs has no
    # behavioural content - it can't represent WHO chooses which mode, only the
    # average level-of-service tradeoff. Previously every person/household term
    # was gated behind GA gene selection, so when the GA didn't happen to pick a
    # vehicle/income/age candidate the mode model collapsed to pure impedance +
    # ASCs (+ the now-removed dead purpose dummies). Promote the theoretically
    # essential behavioural drivers to always-on so every mode model - GA-driven
    # or not - carries genuine behavioural heterogeneity:
    #   • vehicle ownership / captivity  (the dominant mode-choice driver)
    #   • income                         (car vs transit affordability)
    #   • age                            (active-mode propensity)
    # The GA still selects ADDITIONAL person/tour interactions on top of these.
    _ALWAYS_INCLUDE = {
        'c_ivt_auto', 'c_ivt_transit', 'c_walk_time', 'c_bike_time',
        'ASC_SR', 'ASC_Walk', 'ASC_Bike', 'ASC_Transit',
        'c_joint_sr', 'c_walk_stops', 'c_bike_stops', 'c_bike_shop', 'c_peak_da',
        # Behavioural backbone (always present regardless of GA gene selection)
        'c_veh_da', 'c_veh_sr', 'c_noveh_transit', 'c_noveh_walk',
        'c_incp_da', 'c_incp_transit',
        'c_age_walk', 'c_age_bike',
    }

    # ── 2. Person/tour term → candidate gate mapping ─────────────────────────
    # A MODE_UEC person/tour term is included when ANY of its mapped candidates
    # is selected (gene=1).  Terms absent from this map are always included.
    _COEFF_GATE: dict[str, set] = {
        # Vehicle ownership / mode captivity
        'c_veh_da':         {'vehrd', 'has_vehicle', 'one_vehicle', 'multi_vehicle'},
        'c_veh_sr':         {'vehrd', 'has_vehicle', 'one_vehicle', 'multi_vehicle'},
        'c_noveh_transit':  {'vehrd', 'no_vehicle'},
        'c_noveh_walk':     {'vehrd', 'no_vehicle'},
        # Income
        'c_incp_da':        {'incp', 'income_verylo', 'income_lo', 'income_mid',
                             'income_midplus', 'income_midhi', 'income_hi', 'income_veryhi'},
        'c_incp_sr':        {'incp', 'income_verylo', 'income_lo', 'income_mid',
                             'income_midplus', 'income_midhi', 'income_hi', 'income_veryhi'},
        'c_incp_transit':   {'incp', 'income_verylo', 'income_lo', 'income_mid',
                             'income_midplus', 'income_midhi', 'income_hi', 'income_veryhi'},
        'c_incp_active':    {'incp', 'income_verylo', 'income_lo', 'income_mid',
                             'income_midplus', 'income_midhi', 'income_hi', 'income_veryhi'},
        # Age
        'c_age_walk':       {'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior'},
        'c_age_bike':       {'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior'},
        'c_young_bike':     {'age_child', 'age_teen'},
        # Gender
        'c_female_transit': {'sexp', 'is_female'},
        'c_female_walk':    {'sexp', 'is_female'},
        # Household size
        'c_hhsize_sr':      {'nprd', 'hh_alone', 'hh_couple', 'hh_medium',
                             'hh_size_large', 'hh_child', 'n_workers_hh'},
        # Bicycles
        'c_bikes_bike':     {'bikes'},
        # Habitual mode-usage captivity (mobility_models propensity index)
        'c_transit_use_transit': {'transit_prop'},
        'c_car_use_da':          {'car_prop'},
        'c_bike_use_bike':       {'bike_prop'},
        # Work status / person type
        # Gate on both the new binary dummies AND the legacy candidates so models
        # estimated before the WORKSTAT→binary migration still activate these terms.
        'c_ftwork_da':       {'workstat_fulltime',  'WORKSTAT', 'mandatory_role', 'ptype_FullTimeWorker'},
        'c_ptwork_sr':       {'workstat_parttime',  'WORKSTAT', 'mandatory_role', 'ptype_PartTimeWorker'},
        'c_notwork_transit': {'workstat_notworking', 'mandatory_role', 'ptype_Retired'},
        # Nonmand vehicle × bike (nonmand segment only)
        'c_bike_veh':       {'vehrd', 'has_vehicle'},
        # CBD proximity
        'c_cbd_transit':    {'is_cbd'},
        'c_cbd_da':         {'is_cbd'},
        'c_cbd_sr':         {'is_cbd'},
        # Transit walk access at destination
        'c_transit_access': {'transit_walk_acc'},
        # Micromobility ownership
        'c_ebike_bike':     {'ebikes', 'escooter'},
        'c_escooter_bike':  {'ebikes', 'escooter'},
        # ── ActivitySim-inspired GA candidates ───────────────────────────────
        # Cost / tolls
        'c_toll_auto':      {'SOVTOLL_TOLL__AM', 'auto_toll'},
        'c_cost_auto':      {'auto_cost'},
        # Auto-sufficiency / captivity
        'c_autodef_transit': {'auto_deficient', 'autos_lt_workers'},
        'c_autodef_walk':    {'auto_deficient', 'autos_lt_workers'},
        # Short-distance active-mode dummies
        'c_short_walk':     {'short_dist_active'},
        'c_short_bike':     {'short_dist_active'},
        # Person-type × mode
        'c_student_transit': {'ptype_UniversityStudent', 'ptype_WorkingStudent',
                              'ptype_SchoolChild', 'studying_secondary',
                              'studying_tertiary_ft', 'studying_tertiary_pt'},
        'c_student_walk':    {'ptype_UniversityStudent', 'ptype_SchoolChild',
                              'studying_secondary'},
        'c_retired_walk':    {'ptype_Retired', 'workstat_notworking', 'age_senior'},
        # Occupation / job-type × mode
        'c_mgr_da':         {'occp_group', 'occp_mgr'},
        'c_trades_da':      {'occp_group', 'occp_trades'},
        'c_prof_transit':   {'occp_group', 'occp_prof'},
        # Per-agent safety utility × crash (active modes)
        'c_crash_safety':   {'safety_sensitivity', 'crash_aversion'},
        # Purpose × mode dummies (always-on structural)
    }

    # ── 3. Extra skim candidates not in MODE_UEC ─────────────────────────────
    # Injected dynamically when their gene is selected.
    _EXTRA_SKIM_SPECS: dict[str, list] = {
        'SOVTOLL_TIME__PM':      [('c_ivt_auto_pm',   'od.SOVTOLL_TIME__PM / 60.0',        {'DA': None, 'SR': None})],
        'SOVTOLL_TIME__MD':      [('c_ivt_auto_md',   'od.SOVTOLL_TIME__MD / 60.0',        {'DA': None, 'SR': None})],
        'SOVTOLL_TIME__EV':      [('c_ivt_auto_ev',   'od.SOVTOLL_TIME__EV / 60.0',        {'DA': None, 'SR': None})],
        'HOV2TOLL_TIME__AM':     [('c_ivt_hov2_am',   'od.HOV2TOLL_TIME__AM / 60.0',       {'SR': None})],
        'WLK_LOC_WLK_XWAIT__AM': [('c_xwait_transit', 'od.WLK_LOC_WLK_XWAIT__AM / 60.0',  {'Transit': None})],
        # Drive-to-transit access time (park-and-ride leg) — ActivitySim
        # WLK_LOC_DRV drive-access transit path component.
        'WLK_LOC_DRV_DTIM__AM':  [('c_drv_transit',   'od.WLK_LOC_DRV_DTIM__AM / 60.0',   {'Transit': None})],
    }

    custom_set = set(custom_vars)
    filtered_uec: dict = {}

    # ── Generic-vs-alternative-specific substitution ─────────────────────────
    # When a "generic" gene (e.g. incp_generic) is selected, the model carries
    # ONE mode-generic coefficient applying the same effect to every listed
    # alternative, instead of the per-mode alternative-specific block. The
    # groups are declared once in ga_constraints.generic_groups so the search
    # space (this gate) and the post-estimation _collapse_generic_groups refit
    # agree. A mutex constraint keeps the GA from selecting both encodings, but
    # we also suppress the ASV coeffs here defensively.
    try:
        from ga_constraints import generic_groups as _generic_groups
        _gg = _generic_groups('stage5_mode')
    except Exception:
        _gg = {}
    _suppress_asv: set = set()          # ASV coeff names to drop
    _generic_inject: dict = {}          # generic coeff -> (expr, mode_filter)
    for _gene, _spec in _gg.items():
        if _gene in custom_set:
            _suppress_asv.update(_spec.get('asv', []))
            _modes = _spec.get('modes') or list(MODE_MAP)
            _generic_inject[_spec['coeff']] = (
                _spec['expr'], {mn: None for mn in _modes})

    for coeff, (expr, mode_filter) in MODE_UEC.items():

        # generic substitution: this ASV term is replaced by a generic coeff
        if coeff in _suppress_asv:
            continue

        # backbone: always on
        if coeff in _ALWAYS_INCLUDE:
            filtered_uec[coeff] = (expr, mode_filter)
            continue

        # optional skim terms: gate by matrix name
        if expr.startswith("od.") or expr.startswith("log(od."):
            _sm = _re.search(r'od\.([A-Za-z0-9_]+)', expr)
            skim_col = _sm.group(1) if _sm else ''
            if skim_col in custom_set:
                filtered_uec[coeff] = (expr, mode_filter)
            continue

        # person/tour terms: include when a mapped candidate is selected
        if coeff in _COEFF_GATE:
            if custom_set & _COEFF_GATE[coeff]:
                filtered_uec[coeff] = (expr, mode_filter)
        else:
            # unmapped tour/person term → always include (safe fallback)
            filtered_uec[coeff] = (expr, mode_filter)

    # inject extra skim terms for selected candidates absent from MODE_UEC
    for var in custom_vars:
        for coeff, expr, mf in _EXTRA_SKIM_SPECS.get(var, []):
            if coeff not in filtered_uec:
                filtered_uec[coeff] = (expr, mf)

    # inject the mode-generic replacement terms (one shared coefficient across
    # the group's modes) for any selected generic gene
    for coeff, (expr, mf) in _generic_inject.items():
        filtered_uec[coeff] = (expr, mf)

    return _build_mode_utility(m, Mode, segment, uec=filtered_uec)


def _dest_spec_custom(mm, segment: str, custom_vars: list):
    """
    GA-restricted destination spec - only includes UEC terms whose
    landuse/skim variable is in custom_vars.
    Delegates to _build_dest_utility with a filtered DEST_UEC.
    """
    if isinstance(custom_vars, dict):
        return _build_dest_utility(mm, segment, uec=custom_vars)

    # Map from variable column name to UEC coefficient name
    var_to_coeff = {}
    for coeff, (expr, _) in DEST_UEC.items():
        # extract the column name from the expression
        col = (expr.replace("log(emp.", "").replace("+1)", "")
                   .replace("emp.", "")
                   .replace("log(od.", "").replace("od.", ""))
        var_to_coeff[col] = coeff

    # Always keep the core behavioural and attraction structure in the
    # destination model. Letting GA/custom filtering drop these terms was
    # collapsing the fitted model down to 2-3 coefficients, which made the
    # predictions unstable and overly concentrated in a single zone.
    always_include = {
        "c_logsum", "c_log_dist", "c_dist", "c_size_term",
        "c_crash",             # c_aggl NO LONGER force-included: the blended
                               # composite is collinear with the per-type themed
                               # ACDC (retail/office/edu/leisure/parks/health/food/
                               # civic) and, when forced, holdfast-zeroed them all.
                               # Agglomeration is now carried by the themed genes
                               # (require-one in ga_constraints). c_comp also out:
                               # collinear with c_aggl (|r|≈0.9998) and, when both
                               # are forced, the pair blows up to huge equal-and-
                               # opposite values (+4.5/−4.5). It stays GA-selectable
                               # but mutually exclusive with c_aggl (ga_constraints).
        "c_incp_dist", "c_veh_dist", "c_occp_dist",
        "c_auto_work_acc", "c_auto_shop_acc",
        "c_walk_acc", "c_transit_acc",
        "c_noveh_walk_acc", "c_noveh_transit_acc",
        "c_indp_finance", "c_indp_health", "c_indp_retail",
        "c_edu_post", "c_edu_primary", "c_edu_secondary",
    }
    filtered_uec = {
        coeff: val
        for coeff, val in DEST_UEC.items()
        if coeff in always_include
        or var_to_coeff.get(
            val[0].replace("log(emp.", "").replace("+1)", "")
                  .replace("emp.", "")
                  .replace("log(od.", "").replace("od.", "")
        ) in always_include
        or any(v in val[0] for v in custom_vars)
    }

    return _build_dest_utility(mm, segment, uec=filtered_uec)


def _apply_param_ordering(model, kind: str = 'mode') -> None:
    """Group parameters for Larch's report printing.

    Larch's ``ordering`` takes (group_name, regex, ...) tuples; grouped
    summaries read by model component (structural / LOS / safety / person
    interactions / land use) instead of one flat alphabetical list. Applied to
    every stage-5 model before reports are written; harmless no-op on Larch
    versions without the attribute.
    """
    try:
        if kind == 'mode':
            model.ordering = (
                ("Structural: ASCs & nesting", "ASC_.*", "Mu_.*"),
                ("Level of service", "c_ivt.*", ".*_time$", "c_wait.*", "c_peak.*"),
                ("Safety", ".*crash.*", ".*CRASH.*"),
                ("Person & household", "c_incp.*", "c_veh.*", "c_noveh.*",
                 "c_age.*", "c_young.*", "c_female.*", "c_ftwork.*",
                 "c_ptwork.*", "c_notwork.*", "c_hhsize.*", "c_joint.*",
                 "c_bikes.*"),
                ("Other", ".*"),
            )
        else:
            model.ordering = (
                ("Structural: size & sampling", "c_size_term", "log_correction"),
                ("Impedance & accessibility", "c_log_dist", "c_dist",
                 ".*intrazonal.*", "c_logsum", ".*_acc$"),
                ("Safety", ".*crash.*"),
                ("Spatial structure (ACDC)", "c_aggl", "c_comp"),
                ("Land-use attractors", "c_indp_.*", "c_edu_.*",
                 "c_log_commercial.*", "c_nonmand_.*"),
                ("Person interactions", "c_incp_.*", "c_age_.*", "c_veh_.*",
                 "c_occp_.*", "c_female_.*", "c_nprd_.*", "c_propensity.*",
                 "c_person.*"),
                ("Other", ".*"),
            )
    except Exception as _oe:
        logger.debug(f"_apply_param_ordering skipped: {_oe}")


def _build_mode_model(dt, segment: str, custom_mode_vars=None, avail_vars_hint=None):
    """
    Build mode choice model for one segment.
    ActivitySim: one model per tour_purpose, shared nested logit structure,
    per-segment coefficient starting values via _apply_segment_coefficients.
    """
    Mode = Dict(**{k: v for k, v in MODE_MAP.items()}).freeze()
    # Prefer JAX when available; _lx_model automatically falls back to numba
    # if JAX is unavailable or fails to initialize.
    m    = _lx_model(dt)
    m.choice_co_code = "tour.tour_mode"
    # apply expansion weights via weight_co_var (Larch ≥6 attribute name)
    # gracefully skip if this version of Larch doesn't support it
    try:
        m.weight_co_var = "weight"
    except (TypeError, AttributeError):
        pass   # weights already baked into dataset freq_weights if supported

    if custom_mode_vars is not None:
        m = _mode_spec_custom(m, Mode, custom_mode_vars, segment=segment)
    else:
        m = _build_mode_utility(m, Mode, segment)

    # Nested logit: Motorised (Car / Transit) and NonMotorised (Walk / Bike)
    # Parameter names use _ not : to avoid sharrow expression-compiler issues
    Car      = m.graph.new_node(parameter="Mu_Car",      children=[Mode.DA, Mode.SR])
    NonMotor = m.graph.new_node(parameter="Mu_NonMotor", children=[Mode.Walk, Mode.Bike])
    Motor    = m.graph.new_node(parameter="Mu_Motor",    children=[Car, Mode.Transit])
    m.graph.new_node(parameter="Mu_Root", children=[Motor, NonMotor])

    # Mu_Root = 1 is the normalisation constraint for the nested logit.
    # It must be locked (holdfast=True) - if it's free, the hessian is
    # singular because the root scale is not separately identified.
    # Nesting params (Mu_Car, Mu_Motor, Mu_NonMotor) must be in (0, 1].
    m.set_value("Mu_Root",     1.0)
    m.set_value("Mu_Car",      0.8)
    m.set_value("Mu_Motor",    0.8)
    m.set_value("Mu_NonMotor", 0.8)
    try:
        m.lock_value("Mu_Root", 1.0)  # holdfast
        # Floor raised 0.001 → 0.10: run06/23300124 models collapsed Mu_Car to
        # the 0.001 bound, scaling in-nest utilities by 1/mu = 1000×.  That
        # made the Hessian non-invertible (every model fell back to OPG SEs →
        # unreliable p-values) and produced degenerate predicted shares
        # (e.g. work DA 0.1% vs 75.5% observed).  mu < 0.1 has no plausible
        # behavioural interpretation; at the 0.10 bound the model is still
        # near-deterministic within the nest but numerically sane.
        # Override via STAGE5_MU_LOWER_BOUND env var if estimation diagnostics
        # show consistent boundary solutions (warnings about params "ended AT
        # A BOUND").  Values below 0.05 are strongly discouraged.
        _mu_floor = float(os.environ.get('STAGE5_MU_LOWER_BOUND', '0.10'))
        _pf_set(m, "Mu_Car",      "minimum", _mu_floor)
        _pf_set(m, "Mu_Car",      "maximum", 1.0)
        _pf_set(m, "Mu_Motor",    "minimum", _mu_floor)
        _pf_set(m, "Mu_Motor",    "maximum", 1.0)
        _pf_set(m, "Mu_NonMotor", "minimum", _mu_floor)
        _pf_set(m, "Mu_NonMotor", "maximum", 1.0)
    except Exception as e:
        logger.debug(f"Could not set nesting bounds: {e}")

    # Availability: ActivitySim builds availability explicitly and falls back
    # gracefully when fields are missing. Mirror that behavior here by checking
    # DataTree variables before assigning availability expressions.
    _tour_vars = _dataset_var_names(dt['tour'])
    _person_vars = _dataset_var_names(dt['person'])
    _avail_vars = _tour_vars | _person_vars
    if avail_vars_hint is not None:
        try:
            _avail_vars = _avail_vars | set(str(c) for c in avail_vars_hint)
        except Exception:
            pass

    _missing = []
    # DA requires BOTH a driving licence AND a household vehicle — a person
    # from a zero-vehicle household cannot drive alone regardless of licence.
    _lic_ok = 'CARLICENCE' in _avail_vars
    _veh_ok = 'vehrd' in _avail_vars
    if _lic_ok and _veh_ok:
        _da_avail = "(CARLICENCE > 0) * (vehrd > 0)"
    elif _lic_ok:
        _da_avail = "CARLICENCE"
        _missing.append('vehrd')
    elif _veh_ok:
        _da_avail = "(vehrd > 0)"
        _missing.append('CARLICENCE')
    else:
        _da_avail = 1
        _missing.extend(['CARLICENCE', 'vehrd'])

    _walk_avail = "walk_avail"
    if 'walk_avail' not in _avail_vars:
        _walk_avail = 1
        _missing.append('walk_avail')

    _bike_avail = "bike_avail"
    if 'bike_avail' not in _avail_vars:
        _bike_avail = 1
        _missing.append('bike_avail')

    if _missing:
        logger.warning(
            f"_build_mode_model [{segment}]: availability vars missing {sorted(set(_missing))}; "
            "falling back to always-available for those alternatives"
        )

    m.availability_co_vars = {
        Mode.DA:      _da_avail,   # licence AND household vehicle required
        Mode.SR:      1,           # SR - passenger, no licence required
        Mode.Walk:    _walk_avail,
        Mode.Bike:    _bike_avail,
        Mode.Transit: 1,
    }

    # apply per-segment coefficient starting values (ActivitySim: get_segment_coefficients)
    m = _apply_segment_coefficients(m, segment)

    # ── Holdfast collinear income parameters for nonmand ─────────────
    # In test runs all four c_incp_* collapsed to ≈ −3.46 for nonmand,
    # indicating collinearity.  We keep only DA and Transit free; SR and
    # active are fixed at 0 so the ASCs absorb only the genuine mode bias.
    if segment == 'nonmand':
        for _hf_param in ('c_incp_sr', 'c_incp_active'):
            try:
                if _hf_param in m.pf.index:
                    m.lock_value(_hf_param, 0.0)
            except Exception:
                pass

    # Constrain bike time to be non-positive for all segments - a positive
    # c_bike_time (longer bike trips more attractive) is theoretically wrong.
    # Observed in edu and nonmand segments due to short-trip selection bias
    # in the HHTS sample (short trips dominate bike choices, pushing c_bike_time up).
    try:
        if 'c_bike_time' in m.pf.index:
            _pf_set(m, 'c_bike_time', 'maximum', -0.001)   # strict negative
    except Exception:
        pass

    # Constrain ASC_Walk to be non-positive for all segments.
    # DISTWALK = DIST in the current OMX (both use road-network distance),
    # so the walk-time coefficient cannot properly identify the walk deterrent.
    # Without this constraint the ASC drifts positive (e.g. +2.8 for work),
    # causing intrazonal/zero-distance trips to heavily overpredict walk mode.
    try:
        if 'ASC_Walk' in m.pf.index:
            _pf_set(m, 'ASC_Walk', 'maximum', 0.0)
    except Exception:
        pass

    for _p in (
        'c_clr_tr', 'c_mgr_da', 'c_lab_da', 'c_ind_da', 'c_off_tr',
        'c_pub_tr', 'c_pub_wk', 'c_hlt_tr', 'c_ret_sr',
        'c_fts_tr', 'c_fts_wk', 'c_fts_bk', 'c_pts_tr'
    ):
        try:
            if _p in m.pf.index:
                _pf_set(m, _p, 'minimum', -3.0)
                _pf_set(m, _p, 'maximum', 3.0)
        except Exception:
            pass

    # Declarative sign priors (cost/time ≤ 0, ...) from ga_constraints. Applied
    # last so it also covers dynamically-injected GA / generic terms.
    m = _apply_sign_constraints(m, 'stage5_mode')

    return m


# ── Flat UEC for pre-joined datasets (ActivitySim pattern) ───────────────
# Maps coefficient names to flat column names in the pre-joined ca dataset.
# These are the SAME coefficients as DEST_UEC but with node-qualified paths
# replaced by the flat column names created in _dest_tree().
# Conditioning expressions are pre-computed as interaction columns in _dest_tree().
DEST_UEC_FLAT = {
    "c_log_dist":           ("log_DIST_1",              None),
    "c_dist":               ("DIST",                    None),
    "c_dist_0_2":           ("dist_0_2",                None),
    "c_dist_2_5":           ("dist_2_5",                None),
    "c_dist_5_10":          ("dist_5_10",               None),
    "c_dist_decay":         ("dist_decay",              None),
    "c_dist_decay_slow":    ("dist_decay_slow",         None),
    "c_cbd_dest":           ("is_cbd",                  None),
    "c_crash":              ("log_CRASH_1",             None),
    "c_crash_safety_dest":  ("safety_x_crash",          None),
    "c_logsum":             ("logsum",                  None),
    "c_size_term":          ("size_term",               None),
    "c_aggl":               ("acdc_aggl",               None),
    "c_comp":               ("acdc_comp",               None),
    "c_aggl_retail":        ("acdc_aggl_retail",        None),
    "c_aggl_office":        ("acdc_aggl_office",        None),
    "c_aggl_edu":           ("acdc_aggl_edu",           None),
    "c_aggl_leisure":       ("acdc_aggl_leisure",       None),
    "c_aggl_parks":         ("acdc_aggl_parks",         None),
    "c_aggl_health":        ("acdc_aggl_health",        None),
    "c_aggl_food":          ("acdc_aggl_food",          None),
    "c_aggl_civic":         ("acdc_aggl_civic",         None),
    "c_home_pull":          ("home_pull",               ["nonmand"]),
    "c_home_pull_late":     ("home_pull_late",          ["nonmand"]),
    "c_incp_dist":          ("incp_x_log_dist",         None),
    "c_age_dist":           ("age5p_x_log_dist",        None),
    "c_veh_dist":           ("vehrd_x_log_dist",        None),
    "c_female_dist":        ("female_x_log_dist",       None),
    "c_nprd_dist":          ("nprd_x_log_dist",         None),
    # c_auto_work_acc REMOVED - collinear with size_term (auto_work_acc ≈ log-sum of
    # neighbourhood TOTEMP), causing negative estimated coefficient (-0.84) that
    # suppresses major employment centres (zone 40 CBD: obs=4.6%, pred=0.9%).
    # Accessibility is already captured via c_logsum + c_aggl.
    # c_incp_work_acc (income × auto_work_acc) RETAINED: at positive incp it
    # correctly draws high-income workers toward accessible employment centres.
    "c_auto_shop_acc":      ("auto_shop_acc",           ["nonmand"]),
    "c_walk_acc":           ("walk_acc",                ["nonmand", "edu"]),
    "c_transit_acc":        ("transit_acc",              ["nonmand", "edu"]),
    "c_noveh_walk_acc":     ("noveh_walk_acc",          ["nonmand", "edu"]),
    "c_noveh_transit_acc":  ("noveh_transit_acc",        ["nonmand", "edu"]),
    "c_incp_work_acc":      ("incp_x_auto_work_acc",    ["work"]),
    "c_ftwork_dist":        ("ftwork_x_log_dist",       ["work"]),
    "c_occp_dist":          ("occp1_x_log_dist",        ["work"]),
    "c_occp_mgr_dist":      ("occp_mgr_x_log_dist",    ["work"]),
    "c_occp_trades_dist":   ("occp_trades_x_log_dist",  ["work"]),
    "c_indp_finance":       ("log_FPSEMPN_cond",        ["work"]),  # log-scale avoids gradient saturation
    "c_indp_health":        ("log_HEREMPN_cond",         ["work"]),  # log-scale
    "c_indp_retail":        ("log_RETEMPN_cond",         ["work"]),  # log-scale
    "c_indp_agri":          ("log_AGREMPN_1",           ["work"]),
    "c_indp_mfg":           ("log_MWTEMPN_1",            ["work"]),
    "c_indp_construction":  ("log_construction_1",       ["work"]),
    "c_indp_govt":          ("log_government_1",          ["work"]),
    "c_indp_industrial":    ("log_industrial_1",         ["work"]),
    "c_indp_education":     ("log_education_emp_1",       ["work"]),
    # c_totemp_work REMOVED - collinear with size_term (both derived from TOTEMP),
    # causing negative estimated coefficient that penalises large employment zones.
    "c_log_commercial_work": ("log_commercial_1",       ["work"]),
    "c_edu_post":           ("edu_post_cond",           ["edu"]),
    "c_edu_primary":        ("edu_primary_cond",        ["edu"]),
    "c_edu_secondary":      ("edu_secondary_cond",      ["edu"]),
    "c_edu_prim_priv":      ("edu_prim_priv_cond",      ["edu"]),   # log(private primary+1) × primary student
    "c_edu_sec_priv":       ("edu_sec_priv_cond",       ["edu"]),   # log(private secondary+1) × secondary student
    # c_auto_edu_acc REMOVED - same collinearity with edu size_term; estimated at
    # -2.04 in main run, massively suppressing university zones (Pearson r=0.152).
    # Education accessibility is captured via size_term (log Education seats) + c_logsum.
    "c_auto_nonmand_acc":   ("auto_nonmand_acc",        ["nonmand"]),  # zone nonmand accessibility heuristic
    # Non-mandatory attractors: log-transformed to prevent gradient saturation
    # (raw residential=3.6M, commercial=337K → utility >> 7000 → softmax saturates)
    "c_nonmand_res":        ("log_residential_1",       ["nonmand"]),
    "c_nonmand_ent":        ("log_entertainment_1",     ["nonmand"]),
    "c_nonmand_com":        ("log_commercial_1",        ["nonmand"]),
    "c_nonmand_retail":     ("log_retail_1",            ["nonmand"]),
    "c_nonmand_totpop":     ("log_TOTPOP_1",            ["nonmand"]),  # population for social trips
    "c_nonmand_eatout":     ("log_EATOUT_1",            ["nonmand"]),  # eat-out venues for eat trips
    "c_propensity_dist":    ("propensity_x_log_dist",   None),
    "c_person_dist_draw":   ("person_dist_draw_x_log_dist", ["work"]),
    # ── Travel time impedance ────────────────────────────────────────────
    # REMOVED: c_log_sovtime and c_log_sovtime_work.
    # The mode logsum (c_logsum) already incorporates SOV travel time as part
    # of the mode utility sum: logsum = log(Σ exp(V_mode)).  Adding raw SOV time
    # on top double-counts auto impedance and competes with c_logsum for the
    # same gradient signal.  Distance deterrence is captured by c_log_dist + c_dist.
    # ── Circuity / walk-distance columns REMOVED ─────────────────────────
    # DISTWALK == DIST == DISTBIKE in this OMX (all road distance).
    # circuity = road/euclidean = 1.0 everywhere → log(1.0) = 0 → zero column.
    # Columns log_circuity, log_active_circ, log_distwalk_masked, etc. are not
    # created in ca_df and are not in the UEC.  Active-mode effects are instead
    # captured by c_walk_feasible / c_bike_feasible and c_logsum.
    # ── Intrazonal preference ────────────────────────────────────────────
    "c_intrazonal":         ("intrazonal",              None),
    "c_intrazonal_edu":     ("intrazonal_x_edu",        ["edu"]),
    # ── Walk/bike zone accessibility ─────────────────────────────────────
    # Binary flags: is this destination zone walkable / bikeable from origin?
    # These are zone-level accessibility indicators that complement the mode
    # logsum - the logsum captures IMPEDANCE (time/cost), these capture
    # FEASIBILITY (binary: can you even walk/bike there?).
    "c_walk_feasible":      ("walk_feasible",           None),
    "c_bike_feasible":      ("bike_feasible",           None),
    # REMOVED: c_edu_prim_walk, c_edu_sec_walk, c_edu_prim_logwalk, c_edu_sec_logwalk
    # These were mode-distance interaction terms in the destination model that
    # double-count with the mode logsum.  Specifically:
    #   c_edu_prim_logwalk = study==1 × log(DISTWALK)
    #   c_edu_sec_logwalk  = study==2 × log(DISTWALK)
    # Since DISTWALK==DIST in this OMX, these equal study_level × log(DIST),
    # which is already captured by c_log_dist acting on all tours.  The logsum
    # (c_logsum) handles mode-specific impedance for walk/bike trips.
    # c_edu_prim_walk / c_edu_sec_walk were triple interactions (student ×
    # walkable × school count) - redundant with c_edu_primary + c_logsum.
}


# ── User overlay: ptype × land-attribute interaction genes ───────────────────
# The constraint GUI (constraint_gui.py → ga_constraints_user.json) can declare
# ptype↔land affinities with interaction=True. Each becomes a destination
# candidate gene: person's ptype dummy × destination zone's land attractor, e.g.
# ptype_UniversityStudent × Education. These register into DEST_UEC / DEST_UEC_FLAT
# so the normal build/gating path picks them up when selected; _dest_tree
# materialises the flat column. Absent-column pairs are harmless (dropped as a
# missing ref during model build).
def _sanitize_col(name: str) -> str:
    import re as _re
    return _re.sub(r'[^0-9A-Za-z]+', '_', str(name)).strip('_')


def _overlay_dest_interactions() -> list:
    """ptype×land interaction terms from the constraint overlay (interaction=True).
    Returns [{'coeff','ptype','land','flat_col','segments'}]."""
    try:
        from ga_constraints import ptype_land_pairs
        pairs = ptype_land_pairs()
    except Exception:
        return []
    out = []
    for p in pairs:
        if not p.get('interaction'):
            continue
        pt, la = p.get('ptype'), p.get('land')
        if not pt or not la:
            continue
        flat = f"{_sanitize_col(pt)}_x_{_sanitize_col(la)}"
        out.append({'coeff': f"c_{flat}", 'ptype': pt, 'land': la,
                    'flat_col': flat, 'segments': (p.get('segments') or None)})
    return out


def _register_overlay_dest_interactions() -> list:
    """Inject overlay ptype×land interactions into DEST_UEC / DEST_UEC_FLAT
    (idempotent). Returns the interaction descriptors for column materialisation."""
    ints = _overlay_dest_interactions()
    for it in ints:
        coeff, flat, segs = it['coeff'], it['flat_col'], it['segments']
        if coeff not in DEST_UEC:
            DEST_UEC[coeff] = (f"person.{it['ptype']} * emp.{it['land']}", segs)
        if coeff not in DEST_UEC_FLAT:
            DEST_UEC_FLAT[coeff] = (flat, segs)
    return ints


def _build_dest_model(dt, segment: str, custom_dest_vars=None, od_vars_hint=None):
    """Build destination choice model for one segment.

    Uses the ActivitySim flat-dataset pattern: all OD, landuse, and person
    columns are pre-joined into the idca dataset by _dest_tree().  Utility
    expressions reference flat column names (e.g. 'log_DIST_1', 'size_term')
    instead of DataTree path-qualified names (e.g. 'od.DIST', 'emp.size_term').
    This avoids Larch 6.x parameter discovery failures with multi-node DataTrees.

    Engine: numba (not jax).
    """
    _is_flat = dt.attrs.get('_flat_mode', False) if hasattr(dt, 'attrs') else getattr(dt, '_flat_mode', False)

    # Use _lx_model for engine selection - tries jax first on Linux (faster),
    # numba first on Windows.  Previous hardcoded numba caused issues on HPC.
    # FORCE JAX: Numba flow compiler has a TypingError with person×zone
    # interaction columns (female_x_log_dist etc.) in flat-mode CA datasets.
    # These columns are 1D in the flat DataFrame but Numba tries 2D indexing.
    mm = _lx_model(dt, compute_engine='jax')
    _apply_param_ordering(mm, kind='dest')

    # Choice definition: flat dataset has DTAZ as co variable directly
    if _is_flat:
        mm.choice_co_code = "DTAZ"
    else:
        mm.choice_co_code = "tour.DTAZ"
    # Do NOT use survey expansion weights for destination model estimation.
    # HHTS expansion weights (PERSWGT23 ~ 200,000 after normalization to mean=1)
    # make the weighted LL ~200,000x larger than the unweighted LL, causing
    # SLSQP to treat any gradient step as negligible → optimizer stalls at
    # initial values. Unweighted MLE remains consistent for MNL parameters.

    # Columns _dest_tree() flagged as person×zone interaction terms or
    # collinear (see _PERSON_ZONE_INTERACTION_COLS / _collinear_ca_cols there).
    # These are structurally 1D (constant per person) but sharrow's numba Flow
    # compiler tries to JIT-compile them as 2D CA arrays, which raises a
    # TypingError - not just once, but on EVERY loglike/gradient evaluation
    # inside maximize_loglike() (one per optimizer iteration/line-search step/
    # finite-difference perturbation), since holdfasting only locks the
    # coefficient value and leaves the column in the compiled Flow. Dropping
    # the term from the utility expression entirely (below) keeps the column
    # out of the Flow so the failure - and the per-evaluation recompile cost -
    # never happens.
    _dt_collinear = set(getattr(dt, 'attrs', {}).get('_collinear_cols', []))

    # Build utility from flat UEC (pre-joined columns) or legacy DataTree UEC
    _built_uec_keys = []   # track which params we expect in pf
    if _is_flat:
        # Use the flat UEC - all columns are pre-joined into the ca dataset
        _uec = DEST_UEC_FLAT
        if custom_dest_vars is not None and isinstance(custom_dest_vars, dict):
            # Map custom DEST_UEC entries to their flat equivalents
            _uec = {k: v for k, v in DEST_UEC_FLAT.items() if k in custom_dest_vars}
            if not _uec:
                _uec = DEST_UEC_FLAT  # fallback to full flat UEC

        # Filter for segment
        _seg_uec = {k: (expr, segs) for k, (expr, segs) in _uec.items()
                    if segs is None or segment in segs}

        # Exclude terms referencing collinear/person-zone-interaction CA columns
        # (see the _dt_collinear comment above) - drop the term rather than
        # only holdfasting its coefficient, so sharrow never compiles a Flow
        # for it.
        _dropped_terms = {k: v for k, v in _seg_uec.items() if v[0] in _dt_collinear}
        if _dropped_terms:
            _seg_uec = {k: v for k, v in _seg_uec.items() if k not in _dropped_terms}
            logger.info(
                f"_build_dest_model [{segment}]: excluded {len(_dropped_terms)} utility "
                f"term(s) referencing collinear/interaction CA columns (avoids sharrow "
                f"numba TypingError): {sorted(_dropped_terms.keys())}"
            )

        if not _seg_uec:
            _seg_uec = {'c_logsum': ('logsum', None), 'c_log_dist': ('log_DIST_1', None)}
            logger.warning(f"_build_dest_model [{segment}]: no flat UEC terms for segment; using minimal fallback")

        _built_uec_keys = list(_seg_uec.keys())
        # Include log_correction in the initial utility build (not via +=)
        # Larch 6.x utility_ca getter returns empty, so += destroys utility.
        _seg_uec['log_correction'] = ('log_correction', None)
        mm = _build_dest_utility(mm, segment, uec=_seg_uec, cond={})
        logger.info(f"_build_dest_model [{segment}]: FLAT utility with {len(_seg_uec)} terms")

        # ── Diagnostic: check pf immediately after utility assignment ────
        try:
            _pf_after_util = len(mm.pf)
            logger.info(f"_build_dest_model [{segment}]: pf len after utility_ca = {_pf_after_util}")
        except Exception as _pf_diag_e:
            logger.warning(f"_build_dest_model [{segment}]: pf access failed after utility_ca: "
                           f"{type(_pf_diag_e).__name__}: {_pf_diag_e}")
            _pf_after_util = 0

        # ── Robust pf population: if Larch failed to discover parameters ──
        # from the LinearFunction, manually inject them via set_value.
        if _pf_after_util == 0 and _built_uec_keys:
            logger.warning(f"_build_dest_model [{segment}]: pf empty despite {len(_built_uec_keys)} "
                           "utility terms - attempting manual parameter injection")
            # Rebuild utility term-by-term and check after each
            utility = None
            for coeff, (expr, segs) in _seg_uec.items():
                term = P(coeff) * X(expr)
                if utility is None:
                    utility = term
                else:
                    utility = utility + term
            if utility is not None:
                try:
                    # Try setting utility_ca fresh
                    mm.utility_ca = utility
                    _pf_retry = len(mm.pf)
                    logger.info(f"_build_dest_model [{segment}]: pf after fresh re-set = {_pf_retry}")
                except Exception as _retry_e:
                    logger.warning(f"_build_dest_model [{segment}]: fresh utility re-set failed: {_retry_e}")

            # If still empty, try with a new model instance
            try:
                if len(mm.pf) == 0:
                    logger.warning(f"_build_dest_model [{segment}]: creating fresh Model instance")
                    mm2 = lx.Model(datatree=dt, compute_engine='jax')
                    mm2.choice_co_code = "DTAZ" if _is_flat else "tour.DTAZ"
                    try:
                        mm2.weight_co_var = "weight"
                    except (TypeError, AttributeError):
                        pass
                    mm2 = _build_dest_utility(mm2, segment, uec=_seg_uec, cond={})
                    _pf_mm2 = len(mm2.pf)
                    logger.info(f"_build_dest_model [{segment}]: fresh Model pf = {_pf_mm2}")
                    if _pf_mm2 > 0:
                        mm = mm2
            except Exception as _mm2_e:
                logger.warning(f"_build_dest_model [{segment}]: fresh Model fallback failed: {_mm2_e}")
    else:
        # Legacy DataTree path - kept for backward compatibility
        if custom_dest_vars is not None and not isinstance(custom_dest_vars, dict):
            mm = _dest_spec_custom(mm, segment, custom_dest_vars)
        else:
            _raw_uec = custom_dest_vars if isinstance(custom_dest_vars, dict) else DEST_UEC
            _safe_uec = _filter_dest_uec_for_datatree(dt, segment, _raw_uec)
            if not _safe_uec:
                if _dest_uec_refs_present(dt, 'base.logsum'):
                    _safe_uec = {'c_logsum': ('base.logsum', None)}
                elif _dest_uec_refs_present(dt, 'emp.size_term'):
                    _safe_uec = {'c_size_term': ('emp.size_term', None)}
                else:
                    logger.error(
                        f"_build_dest_model [{segment}]: no valid destination terms found; "
                        "model may run with fixed/default values only."
                    )
            # Include log_correction in initial utility (not via += which is broken in Larch 6.x)
            _safe_uec['log_correction'] = ('base.log_correction', None)
            mm = _build_dest_utility(mm, segment, uec=_safe_uec)

    # ── log_correction holdfast: fix at 1.0 for Manski-Lerman correction ──
    # log_correction was added to the UEC above, not via +=
    try:
        if 'log_correction' in mm.pf.index:
            mm.lock_value('log_correction', 1.0)
    except Exception as _lc_e:
        logger.debug(f"log_correction holdfast skipped: {_lc_e}")

    # Availability: exclude zones beyond 150km
    if _is_flat:
        mm.availability_ca_var = "DIST < 150"
    else:
        _od_avail = _dataset_var_names(dt['od'])
        if od_vars_hint is not None:
            try:
                _od_avail = _od_avail | set(str(c) for c in od_vars_hint)
            except Exception:
                pass
        if 'SOVTOLL_DIST__AM' in _od_avail:
            mm.availability_ca_var = "od.SOVTOLL_DIST__AM < 150"
        elif 'DIST' in _od_avail:
            mm.availability_ca_var = "od.DIST < 150"
        else:
            mm.availability_ca_var = 1
            logger.warning(
                f"_build_dest_model [{segment}]: no OD distance skim available for availability; "
                "using all destinations as available"
            )

    mm = _apply_segment_coefficients(mm, segment)

    # ── Logsum bounds ─────────────────────────────────────────────────
    if 'c_logsum' in mm.pf.index:
        _pf_set(mm, 'c_logsum', 'minimum', -5.0)
        _pf_set(mm, 'c_logsum', 'maximum', 15.0)

    # ── Sign constraints - enforce theoretical expectations ────────────
    if 'c_log_dist' in mm.pf.index:
        _pf_set(mm, 'c_log_dist', 'maximum', -0.001)
    if 'c_size_term' in mm.pf.index:
        _pf_set(mm, 'c_size_term', 'minimum', 0.001)
    # c_aggl: no lower bound - allowing negative values lets the optimizer
    # move freely.  A minimum=0.0 causes L-BFGS-B to stall when warm-started
    # from a near-zero (previously holdfasted) value: the projected gradient
    # is zero at the boundary, so the parameter never updates.
    if 'c_comp' in mm.pf.index:
        _pf_set(mm, 'c_comp', 'maximum', -0.001)
    # c_edu_secondary: secondary students prefer zones with secondary schools (≥0)
    if 'c_edu_secondary' in mm.pf.index:
        _pf_set(mm, 'c_edu_secondary', 'minimum', 0.0)
    # c_noveh_transit_acc: car-free travellers should prefer transit-accessible zones (≥0)
    if 'c_noveh_transit_acc' in mm.pf.index:
        _pf_set(mm, 'c_noveh_transit_acc', 'minimum', 0.0)

    # ── c_size_term: holdfast at 1.0 (ActivitySim offset pattern) ────────
    # The log(size) term is a pure attractor offset - its coefficient should
    # be 1.0 so the model captures RELATIVE zone attractiveness (centred by
    # compute_size_terms demeaning).  Estimating c_size_term freely inflates
    # any zone with above-average employment (esp. zone 1 = CBD) because the
    # optimizer amplifies the size differences.  Fixing at 1.0:
    #   (a) eliminates zone-1 over-prediction driven by scale amplification
    #   (b) removes one free parameter → smaller Hessian → no NaN std errors
    #   (c) matches ActivitySim tour_destination_size_terms semantics
    if 'c_size_term' in mm.pf.index:
        try:
            mm.lock_value('c_size_term', 1.0)
        except Exception:
            pass

    # ── c_comp: free to estimate alongside c_aggl ─────────────────────────
    # acdc_aggl (agglomeration) and acdc_comp (competition) capture different
    # urban economics: agglomeration measures destination-zone employment density
    # while competition measures how many origins compete for it.  Both are added
    # to _SAFE_PAIRS so the generic collinearity guard does not auto-holdfast
    # either - if they remain highly correlated in a given segment, L-BFGS will
    # jointly shrink both toward zero (an acceptable outcome).

    # ── Holdfast only zero-variance columns in phase-1 ───────────────
    # c_edu_primary is NOT holdfasted - it's free from phase 1 so the
    # optimizer can find the school-zone attractor in a single pass.
    # Previously holding it caused edu coefficients to stay at 0 because
    # phase-2 release via _parameter_bucket is fragile across Larch versions.
    _INITIAL_HOLDFAST = {
        # c_person_dist_draw: the column is all-zero during estimation
        # (person_dist_draw_x_log_dist is set to 0.0 by default and filled only
        # at prediction time with per-person random draws).
        # A zero-variance column makes the Hessian singular → ALL std errors
        # become NaN.  Holdfast at 1.0 (fixed scaling coefficient).
        'c_person_dist_draw',
        # NOTE: c_log_distbike, c_log_distbike_nm, c_log_distwalk, c_log_distwalk_nm
        # have been REMOVED from DEST_UEC_FLAT and from ca_df.  DISTWALK==DIST in
        # this OMX so those columns were collinear with log_DIST_1.  Circuity was
        # also removed (log_circuity = 0 everywhere).  Do NOT add them back here.
    }

    # c_aggl / c_comp collinearity:
    # acdc_aggl and acdc_comp are highly correlated (|r|≈0.9998) so only ONE
    # should be free at a time.  c_comp is already locked at 0.0 above (line
    # ~3978), so c_aggl is FREE to estimate - no additional holdfast needed.
    # (Previously this block incorrectly holdfasted c_aggl too, leaving both
    # frozen at their starting values and contributing nothing to model fit.)

    # Also holdfast any UEC coefficient whose source CA column was flagged
    # as collinear during _dest_tree (|r| ≥ 0.995 with another column).
    # Flat mode already excluded these terms from the utility above (see
    # _dt_collinear), so they'll never appear in mm.pf - this fallback only
    # matters for the legacy (non-flat) DataTree path.
    if not _is_flat:
        _col_to_coeff = {expr: coeff for coeff, (expr, _) in DEST_UEC_FLAT.items()}
        for _cc in _dt_collinear:
            _coeff_name = _col_to_coeff.get(_cc)
            if _coeff_name and _coeff_name not in _INITIAL_HOLDFAST:
                logger.info(
                    f"_build_dest_model [{segment}]: holdfasting '{_coeff_name}' "
                    f"(CA col '{_cc}' flagged as collinear by _dest_tree)"
                )
                _INITIAL_HOLDFAST.add(_coeff_name)
    for _hf in _INITIAL_HOLDFAST:
        if _hf in mm.pf.index:
            try:
                _cur = float(mm.pf.loc[_hf, 'value']) if _hf in mm.pf.index else 0.0
                mm.lock_value(_hf, _cur if np.isfinite(_cur) else 0.0)
            except Exception:
                _pf_set(mm, _hf, 'holdfast', True)

    # Declarative sign priors (distance/crash ≤ 0, size/accessibility ≥ 0) from
    # ga_constraints — covers dynamically-injected GA / spline terms too.
    mm = _apply_sign_constraints(mm, 'stage5_dest')

    return mm


def _pf_len_safe(model) -> int:
    """Return len(model.pf) safely; 0 when unavailable."""
    try:
        pf = model.pf
        n = int(len(pf))
        return n
    except Exception as _pf_e:
        logger.warning(f"_pf_len_safe: model.pf access failed: {type(_pf_e).__name__}: {_pf_e}")
        return 0


def _minimal_dest_uec(dt, segment: str) -> dict:
    """
    Build a compile-safe fallback destination UEC.

    Used only when the full destination spec compiles to an empty parameter
    frame. Keeps a small set of robust core terms.

    Supports both flat Datasets (pre-joined columns) and DataTree structures.
    """
    _is_flat = False
    try:
        _is_flat = dt.attrs.get('_flat_mode', False) if hasattr(dt, 'attrs') else getattr(dt, '_flat_mode', False)
    except Exception:
        pass

    if _is_flat:
        # Flat mode: use direct column names from DEST_UEC_FLAT
        uec = {
            'c_logsum':    ('logsum',      None),
            'c_size_term': ('size_term',   None),
            'c_log_dist':  ('log_DIST_1',  None),
            'c_dist':      ('DIST',        None),
            'c_crash':     ('log_CRASH_1', None),
        }
    else:
        # DataTree mode: use node-qualified expressions
        uec = {}
        if _dest_uec_refs_present(dt, 'base.logsum'):
            uec['c_logsum'] = ('base.logsum', None)
        if _dest_uec_refs_present(dt, 'emp.size_term'):
            uec['c_size_term'] = ('emp.size_term', None)
        if _dest_uec_refs_present(dt, 'log(od.DIST+1)'):
            uec['c_log_dist'] = ('log(od.DIST+1)', None)
        if _dest_uec_refs_present(dt, 'od.DIST'):
            uec['c_dist'] = ('od.DIST', None)
        if _dest_uec_refs_present(dt, 'log(od.CRASH+1)'):
            uec['c_crash'] = ('log(od.CRASH+1)', None)

        if not uec:
            # Last-resort single-term utility to avoid total compile collapse.
            uec = {'c_logsum': ('base.logsum', None)}

    logger.warning(
        f"_build_dest_model [{segment}]: using minimal fallback UEC "
        f"({len(uec)} terms): {sorted(uec.keys())}"
    )
    return uec


def _print_model_table(model, label: str):
    """Print a clean parameter table to stdout from model.pf.

    Robust to mangled/stripped Larch models - falls back to _coef_cache when
    pf access fails, and uses loop-based SE retrieval to avoid the pandas
    Series.get() NaN-leak bug (where .get() returns NaN even when a fallback
    column exists, because the key is present but its value is NaN).
    """
    # ── Get parameter frame - try pf first, fall back to _coef_cache ─────────
    pf = None
    try:
        pf = model.pf
    except Exception as _pf_e:
        logger.debug(f"_print_model_table [{label}]: model.pf failed ({_pf_e}), "
                     "falling back to _coef_cache")

    ll = getattr(model, '_cached_ll', None)
    if ll is None:
        try:
            ll = float(model.most_recent_estimation_result.loglike)
        except Exception:
            ll = float('nan')
    ll = float(ll) if ll is not None else float('nan')

    rho2 = _rho_sq_null(model, ll)

    if pf is None or len(pf) == 0:
        # Degrade gracefully: print from _coef_cache only (no SEs)
        coef_cache = getattr(model, '_coef_cache', {})
        if not coef_cache:
            print(f"\n  ┌─ {label}  LL={ll:.1f}  [no parameter frame available] ─")
            return
        print(f"\n  ┌─ {label}  LL={ll:.1f}  ρ²={rho2:.3f}  "
              f"n_params={len(coef_cache)}  [coef_cache - no SEs] ─────────────")
        print(f"  {'Parameter':<30} {'Value':>9}  {'Std Err':>8}  {'t-stat':>7}  Sig")
        print(f"  {'─'*65}")
        for name, val in sorted(coef_cache.items()):
            print(f"  {str(name):<30} {val:>9.4f}  {'n/a':>8}  {'n/a':>7}")
        print(f"  └{'─'*65}")
        return

    # Detect best SE source
    hessian_ok = 'std_err' in pf.columns and pf['std_err'].notna().any()
    opg_ok     = 'std_err_opg' in pf.columns and pf['std_err_opg'].notna().any()
    se_note    = '' if hessian_ok else ' [OPG se]' if opg_ok else ' [se=proxy/nan]'

    n = len(pf)
    print(f"\n  ┌─ {label}  LL={ll:.1f}  ρ²={rho2:.3f}  n_params={n}{se_note} ─────────────────")
    print(f"  {'Parameter':<30} {'Value':>9}  {'Std Err':>8}  {'t-stat':>7}  Sig")
    print(f"  {'─'*65}")
    for name in pf.index:
        row = pf.loc[name]
        val = float('nan')
        try:
            val = float(row['value'])
        except Exception:
            pass

        # Loop-based SE retrieval - avoids .get() returning NaN when key exists but value is NaN
        se = float('nan')
        for _sc in ('std_err', 'std_err_opg', 'se', 'stderr'):
            if _sc in row.index:
                _sv = float(row[_sc]) if not isinstance(row[_sc], float) else row[_sc]
                try: _sv = float(row[_sc])
                except Exception: continue
                if not np.isnan(_sv) and _sv >= 0:
                    se = _sv
                    break

        # Loop-based t-stat retrieval
        t = float('nan')
        for _tc in ('t_stat', 't_stat_opg', 'tstat'):
            if _tc in row.index:
                try:
                    _tv = float(row[_tc])
                    if not np.isnan(_tv):
                        t = _tv
                        break
                except Exception:
                    continue
        # Compute t from val/se if still NaN
        if np.isnan(t) and not np.isnan(se) and se > 0 and not np.isnan(val):
            t = val / se

        try:
            sig = ('***' if abs(t) > 3.3 else
                   '**'  if abs(t) > 2.6 else
                   '*'   if abs(t) > 1.96 else '') if np.isfinite(t) else ''
        except Exception:
            sig = ''
        # Distinguish user-fixed parameters from collinear-holdfasted ones.
        # Fixed (explicit prior): typically c_log_dist=-2.0, log_correction=1.0
        # Holdfasted (collinear): dropped because another variable captured it
        if row.get('holdfast', False):
            # Treat near-zero fixed params as holdfasted (collinear), non-zero as user-fixed
            if abs(val) < 1e-8:
                hf = ' (holdfasted: collinear)'
            else:
                hf = ' (fixed: user prior)'
        else:
            hf = ''

        se_str = f'{se:>8.4f}' if np.isfinite(se) else '     n/a'
        t_str  = f'{t:>7.2f}'  if np.isfinite(t)  else '    n/a'
        print(f"  {str(name):<30} {val:>9.4f}  {se_str}  {t_str}  {sig}{hf}")
    print(f"  └{'─'*65}")


def _opg_stderr(model) -> 'pd.Series':
    """
    Outer Product of Gradients (BHHH) standard errors.

    Always computes - does not require Hessian inversion.  Used as a
    fallback when calculate_parameter_covariance() fails (singular Hessian).

    The OPG estimator is:  Var(θ) ≈ B⁻¹  where  B = Σᵢ sᵢsᵢᵀ
    with sᵢ = ∂ log L_i / ∂θ (per-observation score vector).

    Notes
    -----
    - OPG standard errors are generally larger than Hessian-based ones.
    - They are asymptotically valid under correct specification.
    - numpy.linalg.pinv is used so the estimator never fails even if B
      is rank-deficient (small samples / perfect multicollinearity).
    """
    import numpy as np
    try:
        # Larch exposes per-case log-likelihood gradient via d_loglike_casewise
        scores = np.asarray(model.d_loglike_casewise())   # (n_obs, n_params)
        logger.debug(f"_opg_stderr: scores shape={scores.shape}, "
                     f"finite={np.isfinite(scores).all()}, "
                     f"nonzero rows={int((np.abs(scores).sum(axis=1) > 0).sum())}")
        if scores.ndim != 2 or scores.shape[1] == 0:
            # Numba engine returns flat gradient vector (n_params,) not per-case (n_obs, n_params)
            # - expected behaviour; sandwich Hessian fallback will handle it
            logger.debug("_opg_stderr: scores matrix is 1-D or empty - numba engine returns "
                         "aggregate gradient, not per-case; sandwich fallback will be used")
            return pd.Series(np.nan, index=model.pf.index)
        # Replace non-finite scores (overflow / underflow) with zero
        scores = np.where(np.isfinite(scores), scores, 0.0)
        # If all scores are zero the model has a flat gradient - degenerate
        if np.abs(scores).max() < 1e-14:
            logger.warning("_opg_stderr: all gradient scores ≈ 0 - model may be at a "
                           "boundary or DataTree was stripped before OPG computed")
            return pd.Series(np.nan, index=model.pf.index)
        B = scores.T @ scores          # (n_params, n_params) outer product sum
        # Ridge regularization prevents degenerate B from singular Hessian.
        # Ridge = 1e-6 × trace(B) / n_params keeps scale-appropriate damping.
        ridge_scale = max(1e-8, 1e-6 * np.trace(B) / max(B.shape[0], 1))
        B_reg = B + ridge_scale * np.eye(B.shape[0])
        try:
            B_inv = np.linalg.inv(B_reg)
        except np.linalg.LinAlgError:
            logger.debug("_opg_stderr: matrix inversion failed - using pseudoinverse")
            B_inv = np.linalg.pinv(B_reg)   # fallback if even ridge fails
        diag  = np.diag(B_inv)
        n_neg = int((diag <= 0).sum())
        if n_neg:
            logger.debug(f"_opg_stderr: {n_neg} negative diagonal entries - "
                         "those parameters will have NaN SE")
        se    = np.where(diag > 0, np.sqrt(diag), np.nan)
        n_ok  = int(np.isfinite(se).sum())
        logger.debug(f"_opg_stderr: {n_ok}/{len(se)} SE values computed successfully")
        return pd.Series(se, index=model.pf.index[:len(se)])
    except Exception as _e:
        logger.warning(f"_opg_stderr failed: {_e}", exc_info=True)
        return pd.Series(np.nan, index=model.pf.index)


def _diag_hessian_stderr(model, ll0: float = None) -> 'pd.Series':
    """
    One-parameter-at-a-time curvature fallback for standard errors.

    This ignores cross-parameter covariance, but it is still much better than
    leaving every destination-model standard error as NaN when both the Larch
    covariance and casewise-score OPG paths fail.
    """
    try:
        pf = model.pf.copy()
        if ll0 is None or not np.isfinite(ll0):
            try:
                ll0 = float(model.loglike())
            except Exception:
                ll0 = float(getattr(model, '_cached_ll', np.nan))
        se = pd.Series(np.nan, index=pf.index, dtype=float)
        for name in pf.index:
            row = pf.loc[name]
            if bool(row.get('holdfast', False)):
                continue
            theta0 = float(row.get('value', np.nan))
            if not np.isfinite(theta0):
                continue
            step = max(1e-4, abs(theta0) * 1e-3)
            try:
                model.set_value(name, theta0 + step)
                ll_plus = float(model.loglike())
                model.set_value(name, theta0 - step)
                ll_minus = float(model.loglike())
                model.set_value(name, theta0)
            except Exception:
                try:
                    model.set_value(name, theta0)
                except Exception:
                    pass
                continue
            second = (ll_plus - 2.0 * ll0 + ll_minus) / (step ** 2)
            if np.isfinite(second) and second < 0:
                se[name] = float(np.sqrt(-1.0 / second))
        return se
    except Exception as _e:
        logger.debug(f"_diag_hessian_stderr failed: {_e}")
        return pd.Series(np.nan, index=model.pf.index)


def _scipy_sandwich_stderr(model) -> 'pd.Series':
    """
    Sandwich (robust) standard errors for the destination choice model.

    Uses scipy's central finite-difference Hessian approximation on the
    full log-likelihood rather than Larch's built-in covariance routine.
    This avoids the near-singular Hessian that arises from 329 alternatives
    with a relatively small estimation sample.

    Estimator
    ---------
    Std err(θᵢ) = sqrt( [H⁻¹]ᵢᵢ )
    where  H = ∂²ℓ/∂θ∂θᵀ  computed via central finite differences.
    Ridge regularisation (1e-5 × diag) prevents exact singularity.

    Notes
    -----
    - Only free (non-holdfasted) parameters are perturbed.
    - Step size: max(1e-4, |θ| × 1e-3) per parameter - same as diag fallback.
    - The full Hessian captures cross-parameter curvature that the diagonal
      fallback misses, giving more accurate std errors for correlated params.
    """
    try:
        from scipy.optimize import approx_fprime
        pf = model.pf
        holdfast = pf['holdfast'] if 'holdfast' in pf.columns else pd.Series(False, index=pf.index)
        free_names = [n for n in pf.index if not bool(holdfast.get(n, False))]
        if not free_names:
            return pd.Series(np.nan, index=pf.index)

        theta0 = np.array([float(pf.loc[n, 'value']) for n in free_names])
        n_free = len(theta0)
        steps  = np.maximum(1e-4, np.abs(theta0) * 1e-3)

        def _ll(theta_vec):
            for nm, val in zip(free_names, theta_vec):
                try:
                    model.set_value(nm, float(val))
                except Exception:
                    pass
            try:
                return float(model.loglike())
            except Exception:
                return float('nan')

        # Restore to optimum after each column
        orig_ll = _ll(theta0)
        if not np.isfinite(orig_ll):
            return pd.Series(np.nan, index=pf.index)

        # Build full Hessian column by column via central differences
        H = np.zeros((n_free, n_free))
        for j in range(n_free):
            # Gradient at θ + h_j·eⱼ and θ - h_j·eⱼ using approx_fprime
            h = steps[j]
            ep = np.zeros(n_free); ep[j] = h
            em = np.zeros(n_free); em[j] = -h
            g_plus  = approx_fprime(theta0 + ep, _ll, steps * 0.1)
            g_minus = approx_fprime(theta0 + em, _ll, steps * 0.1)
            H[:, j] = (g_plus - g_minus) / (2.0 * h)
            # Restore theta
            for nm, val in zip(free_names, theta0):
                try:
                    model.set_value(nm, float(val))
                except Exception:
                    pass

        # Symmetrise and ridge-regularise
        H = (H + H.T) / 2.0
        ridge = 1e-5 * np.maximum(np.abs(np.diag(H)), 1e-8)
        H -= np.diag(ridge)   # H should be negative definite at maximum

        # Invert: -H is positive definite (information matrix)
        try:
            Hinv = np.linalg.inv(-H)
        except np.linalg.LinAlgError:
            Hinv = np.linalg.pinv(-H)

        diag = np.diag(Hinv)
        se_free = np.where(diag > 0, np.sqrt(diag), np.nan)

        se = pd.Series(np.nan, index=pf.index, dtype=float)
        for nm, s in zip(free_names, se_free):
            se[nm] = s
        n_ok = int(np.isfinite(se_free).sum())
        logger.info(f"_scipy_sandwich_stderr: {n_ok}/{n_free} free params have finite std err")
        return se

    except Exception as _e:
        logger.debug(f"_scipy_sandwich_stderr failed: {_e}")
        return pd.Series(np.nan, index=model.pf.index)


def _prefit_mode_model(mode_mdl, seg_df, persons_df, skims, segment):
    """Pre-condition NL mode model starting values before gradient estimation.

    Step 1 — Mu interior initialization:
      Reset any Mu_Car / Mu_Motor / Mu_NonMotor that sits at or near its
      bound (≥ 0.95 or ≤ 0.05) to 0.7.  Starting at Mu=1 immediately
      collapses the nest gradient to zero; BFGS then stays on the boundary
      for the entire optimisation, blowing up all standard errors.

    Step 2 — ASC log-share pre-calibration:
      Iteratively adjust ASC starting values using the log-share correction:
          ASC_m  +=  log( obs_share_m / pred_share_m )
      until the predicted aggregate shares (computed via _mode_probs_numpy —
      pure numpy, no JAX/Larch loglike compilation overhead) match observed
      shares from seg_df.  Five iterations are almost always sufficient.

      Starting estimation from matching aggregate shares prevents the gradient
      from spending early iterations purely on ASC correction, which is the
      mechanism by which Mu parameters get pulled to 1.0: when ASCs are far
      from their calibrated values the steepest-ascent direction at t=0 is
      dominated by ASC moves that simultaneously relax nest compression.

    Step 3 — Differential Evolution (DE) pre-search over ASC + Mu:
      Runs a lightweight DE over the 7-dimensional subspace of the four free
      ASC parameters and the three Mu nest parameters, using the model's own
      loglike() as the objective.  This explores the joint ASC-Mu space with
      the TRUE nested-logit objective (not the flat-MNL approximation used in
      Step 2), finding a basin-of-attraction that gradient-based optimisation
      can then refine reliably.

      The DE uses a small budget (popsize=4, maxiter=10 → ~280 evaluations)
      so the overhead is roughly one initial JAX/Numba compile + 280 fast
      re-evaluations — typically 1-3 minutes per segment.  If the DE result
      does not improve on the starting LL, the pre-Step-2 values are kept.

    Controlled by env var STAGE5_MODE_DE_PREFIT (default "1", set "0" to skip
    DE while keeping Steps 1+2).
    """
    # ── Step 1: Mu interior initialization ────────────────────────────────
    try:
        for _mu_p in ('Mu_Car', 'Mu_Motor', 'Mu_NonMotor'):
            if _mu_p not in mode_mdl.pf.index:
                continue
            _is_hf = bool(mode_mdl.pf.loc[_mu_p].get('holdfast', False))
            if _is_hf:
                continue
            _v = float(mode_mdl.pf.loc[_mu_p, 'value'])
            if not np.isfinite(_v) or _v >= 0.95 or _v <= 0.05:
                _safe_start = 0.7
                try:
                    mode_mdl.set_value(_mu_p, _safe_start)
                except Exception:
                    mode_mdl.pf.loc[_mu_p, 'value'] = _safe_start
                logger.debug(
                    f"[{segment}] Pre-fit: {_mu_p} start {_v:.4f} → {_safe_start} "
                    "(reset to interior to avoid degenerate NL start)"
                )
    except Exception as _mu_e:
        logger.debug(f"[{segment}] Pre-fit Mu interior reset failed: {_mu_e}")

    # ── Step 2: ASC log-share pre-calibration ─────────────────────────────
    try:
        _mode_col = 'tour_mode' if 'tour_mode' in seg_df.columns else None
        if _mode_col is not None and len(seg_df) > 0:
            _obs_total = float(len(seg_df))
            _obs_counts = seg_df[_mode_col].value_counts()
            _obs_shares = {int(k): float(v) / _obs_total for k, v in _obs_counts.items()}

            _asc_map = {
                'ASC_SR':      MODE_MAP['SR'],
                'ASC_Walk':    MODE_MAP['Walk'],
                'ASC_Bike':    MODE_MAP['Bike'],
                'ASC_Transit': MODE_MAP['Transit'],
            }
            _col_to_code = {
                0: MODE_MAP['DA'], 1: MODE_MAP['SR'],
                2: MODE_MAP['Walk'], 3: MODE_MAP['Bike'], 4: MODE_MAP['Transit'],
            }

            for _it in range(6):
                mode_mdl._coef_cache = {}
                _probs = _mode_probs_numpy(mode_mdl, seg_df, persons_df, skims,
                                           coef_override=None)
                if _probs is None or _probs.shape[0] == 0:
                    break
                _pred_arr = np.asarray(_probs).mean(axis=0)
                _pred_dict = {_col_to_code[ci]: float(_pred_arr[ci])
                              for ci in range(len(_pred_arr))}

                _any_adj = False
                for _asc_p, _mc in _asc_map.items():
                    if _asc_p not in mode_mdl.pf.index:
                        continue
                    _is_hf = bool(mode_mdl.pf.loc[_asc_p].get('holdfast', False))
                    if _is_hf:
                        continue
                    _obs_s  = _obs_shares.get(_mc, 0.0)
                    _pred_s = _pred_dict.get(_mc, 0.0)
                    if _obs_s > 0.001 and _pred_s > 0.001:
                        _delta = float(np.log((_obs_s + 1e-8) / (_pred_s + 1e-8)))
                        if abs(_delta) > 0.005:
                            _cur = float(mode_mdl.pf.loc[_asc_p, 'value'])
                            try:
                                mode_mdl.set_value(_asc_p, _cur + _delta)
                            except Exception:
                                mode_mdl.pf.loc[_asc_p, 'value'] = _cur + _delta
                            _any_adj = True
                mode_mdl._coef_cache = {}
                if not _any_adj:
                    logger.debug(f"[{segment}] Pre-fit ASC converged in {_it + 1} iter")
                    break
    except Exception as _asc_e:
        logger.debug(f"[{segment}] Pre-fit ASC log-share correction failed: {_asc_e}")

    # ── Step 3: Differential Evolution over ASC + Mu (true NL objective) ──
    # The log-share correction above (Step 2) uses the flat-MNL approximation
    # and cannot see the nest structure.  This DE uses model.loglike() — the
    # actual nested-logit log-likelihood — to jointly optimise the 4 ASC and
    # 3 Mu parameters (7-D search) using a small population budget.
    # popsize=4 × 7 params = 28 individuals; 10 generations → ~280 evaluations.
    # Skip if STAGE5_MODE_DE_PREFIT=0.
    _de_enabled = os.environ.get('STAGE5_MODE_DE_PREFIT', '1').strip() not in ('0', 'false', 'False', 'no')
    if _de_enabled:
        try:
            from scipy.optimize import differential_evolution as _scipy_de

            # Candidate params for DE: free ASC + Mu params only.
            # Behavioural params (travel time, cost, demographics) keep their
            # starting values from _apply_segment_coefficients / warm-start.
            _de_candidates = {
                'ASC_SR':      (-5.0, 3.0),
                'ASC_Walk':    (-6.0, 3.0),
                'ASC_Bike':    (-7.0, 2.0),
                'ASC_Transit': (-5.0, 3.0),
                'Mu_Car':      (0.10, 0.95),
                'Mu_Motor':    (0.10, 0.95),
                'Mu_NonMotor': (0.10, 0.95),
            }
            _de_params = []
            _de_bounds = []
            for _p, (_lo, _hi) in _de_candidates.items():
                if _p not in mode_mdl.pf.index:
                    continue
                _is_hf = bool(mode_mdl.pf.loc[_p].get('holdfast', False))
                if _is_hf:
                    continue
                # Respect tighter model-defined bounds
                _model_lo = float(mode_mdl.pf.loc[_p].get('minimum', _lo))
                _model_hi = float(mode_mdl.pf.loc[_p].get('maximum', _hi))
                _lo_eff = max(_lo, _model_lo + 1e-4)
                _hi_eff = min(_hi, _model_hi - 1e-4)
                if _lo_eff >= _hi_eff:
                    continue
                _de_params.append(_p)
                _de_bounds.append((_lo_eff, _hi_eff))

            if len(_de_params) >= 2:
                # Snapshot starting values so we can restore on failure / no-improve
                _de_orig = {p: float(mode_mdl.pf.loc[p, 'value']) for p in _de_params}

                # Compute initial LL at the Step-2-calibrated starting point
                try:
                    _ll_init = float(mode_mdl.loglike())
                except Exception as _ll_e:
                    logger.debug(f"[{segment}] DE pre-fit: initial loglike failed ({_ll_e}); skipping DE")
                    _ll_init = float('nan')

                if np.isfinite(_ll_init):
                    _de_call_count = [0]

                    def _neg_ll(x):
                        _de_call_count[0] += 1
                        for _p, _v in zip(_de_params, x):
                            try:
                                mode_mdl.set_value(_p, float(np.clip(_v,
                                    mode_mdl.pf.loc[_p].get('minimum', -10),
                                    mode_mdl.pf.loc[_p].get('maximum',  10))))
                            except Exception:
                                mode_mdl.pf.loc[_p, 'value'] = float(_v)
                        mode_mdl._coef_cache = {}
                        try:
                            _ll = float(mode_mdl.loglike())
                            return -_ll if np.isfinite(_ll) else 1e9
                        except Exception:
                            return 1e9

                    print(f"  [{segment}] DE pre-fit: searching {len(_de_params)}-D ASC+Mu space "
                          f"(pop=4×{len(_de_params)}={4*len(_de_params)}, maxiter=10) ...")
                    _de_result = _scipy_de(
                        _neg_ll,
                        bounds=_de_bounds,
                        maxiter=10,
                        popsize=4,
                        seed=42,
                        tol=0.005,
                        mutation=(0.5, 1.5),
                        recombination=0.7,
                        polish=False,     # gradient polish = maximize_loglike's job
                        workers=1,        # serial (parallelised at segment level)
                        updating='immediate',
                        init='sobol',     # better space-filling than random
                    )

                    _ll_de = -float(_de_result.fun)
                    if _ll_de > _ll_init + 0.1:
                        # DE improved the starting point → keep its values
                        for _p, _v in zip(_de_params, _de_result.x):
                            try:
                                mode_mdl.set_value(_p, float(_v))
                            except Exception:
                                mode_mdl.pf.loc[_p, 'value'] = float(_v)
                        mode_mdl._coef_cache = {}
                        logger.info(
                            f"[{segment}] DE pre-fit: LL init={_ll_init:.1f} → "
                            f"DE={_ll_de:.1f} (+{_ll_de - _ll_init:.1f}) "
                            f"in {_de_result.nit} gen × {_de_call_count[0]} evals; "
                            "params: " + ", ".join(
                                f"{p}={v:.3f}" for p, v in zip(_de_params, _de_result.x))
                        )
                    else:
                        # DE did not improve — restore Step-2-calibrated values
                        for _p, _v in _de_orig.items():
                            try:
                                mode_mdl.set_value(_p, _v)
                            except Exception:
                                mode_mdl.pf.loc[_p, 'value'] = _v
                        mode_mdl._coef_cache = {}
                        logger.debug(
                            f"[{segment}] DE pre-fit: no improvement "
                            f"(init={_ll_init:.1f}, DE={_ll_de:.1f}); "
                            "keeping log-share-calibrated starting values"
                        )
        except Exception as _de_e:
            logger.warning(f"[{segment}] DE pre-fit failed ({_de_e}); continuing with Step-2 values")
            # Ensure we don't leave model in a half-updated state
            try:
                for _p, _v in _de_orig.items():
                    mode_mdl.set_value(_p, _v)
                mode_mdl._coef_cache = {}
            except Exception:
                pass

    return mode_mdl


def _estimate(model, two_phase: bool = False):
    """
    Estimate a Larch model.

    Parameters
    ----------
    two_phase : if True, run a two-phase estimation:
                Phase 1 - estimate with industry/edu terms holdfasted so the
                          core distance, logsum, and size_term converge first.
                Phase 2 - release holdfasted terms and re-estimate to full optimum.
                This dramatically improves Hessian conditioning and eliminates
                NaN std errors that arise from near-collinear sparse columns.
    """
    # Save per-parameter bounds before set_cap overrides them.
    # set_cap(8) sets maximum=8 / minimum=-8 for ALL parameters, which
    # overwrites any stricter bounds already in pf (e.g. maximum=0.0 on
    # c_bike_time to enforce a negative sign constraint).  We restore those
    # stricter bounds immediately after so the optimizer respects them.
    try:
        _saved_max = (model.pf['maximum'].copy()
                      if 'maximum' in model.pf.columns else None)
        _saved_min = (model.pf['minimum'].copy()
                      if 'minimum' in model.pf.columns else None)
    except Exception:
        _saved_max = _saved_min = None

    model.set_cap(8)

    try:
        if _saved_max is not None:
            for _p, _v in _saved_max.items():
                if pd.notna(_v) and _v < 8.0:
                    _pf_set(model, _p, 'maximum', _v)
                    # clip starting value inside bounds
                    cur = model.pf.loc[_p, 'value']
                    if pd.notna(cur) and cur > _v:
                        model.set_value(_p, min(_v - 0.01, -0.001) if _v <= 0 else _v)
        if _saved_min is not None:
            for _p, _v in _saved_min.items():
                if pd.notna(_v) and _v > -8.0:
                    _pf_set(model, _p, 'minimum', _v)
    except Exception as _e:
        logger.debug(f"_estimate: could not restore per-parameter bounds: {_e}")

    model.doctor(repair_ch_av="+")
    model.doctor(repair_nan_utility=True)

    try:
        if 'Mu_Root' in model.pf.index:
            model.lock_value('Mu_Root', 1.0)
    except Exception:
        pass

    # ── Pre-estimation guard: verify pf is non-empty ─────────────────────────
    # If pf is empty it means Larch failed to compile the utility function
    # (most often because a DataTree variable referenced in the UEC - e.g.
    # od.DIST - does not exist in the underlying OMX/dataset).  Calling
    # maximize_loglike with an empty parameter frame raises a cryptic
    # "zero-size array" ValueError from numpy inside Larch's optimization.py.
    # Detecting it here gives an actionable error message instead.
    try:
        _pf_pre = model.pf
    except Exception as _pf_e:
        logger.error(f"_estimate: cannot access model.pf ({_pf_e}) - returning initial values")
        model._cached_ll = float('nan')
        return model

    _n_free = int((~_pf_pre.get('holdfast', pd.Series(False, index=_pf_pre.index))).sum())
    _n_total = len(_pf_pre)
    if _n_total == 0:
        _msg = (
            "_estimate: model.pf is EMPTY (0 parameters). "
            "Larch failed to compile/discover utility parameters. "
            "Likely causes include incompatible utility syntax, stale sharrow "
            "cache collisions, or references to unavailable DataTree variables. "
            "Check destination term logs, DataTree variable audit, and retry "
            "with the minimal destination spec fallback."
        )
        logger.error(_msg)
        model._cached_ll = float('nan')
        return model
    if _n_free == 0:
        logger.warning(
            f"_estimate: pf has {_n_total} params but ALL are holdfasted "
            f"({_n_total} fixed). Nothing to optimize - returning initial values."
        )
    else:
        logger.debug(f"_estimate: pf OK - {_n_free} free / {_n_total} total params")

    # ── Pre-estimation: detect zero-variance utility columns ─────────────
    # A free parameter whose utility column has zero variance produces a
    # zero-gradient row, making the Hessian singular → ALL std errors become
    # NaN.  Auto-holdfast such parameters to keep the Hessian invertible.
    try:
        _dt = model.datatree if hasattr(model, 'datatree') else None
        if _dt is not None:
            # Get the CA dataset (flat pre-joined columns)
            _ca_df = None
            try:
                if hasattr(_dt, 'dc') and hasattr(_dt.dc, 'to_dataframe'):
                    _ca_df = _dt.dc.to_dataframe()
                elif hasattr(_dt, 'to_dataframe'):
                    _ca_df = _dt.to_dataframe()
            except Exception:
                pass
            if _ca_df is not None and len(_ca_df) > 0:
                _auto_holdfasted = []
                for _p in _pf_pre.index:
                    if _pf_pre.loc[_p].get('holdfast', False):
                        continue
                    # Try to find the column referenced by this parameter
                    # In the flat UEC, the column name is stored in the utility expression
                    _col_name = str(_p).replace('c_', '', 1)
                    for _cn in [_col_name, str(_p)]:
                        if _cn in _ca_df.columns:
                            _col_std = _ca_df[_cn].std()
                            if _col_std is not None and _col_std < 1e-10:
                                try:
                                    _cur_v = float(model.pf.loc[_p, 'value'])
                                    model.lock_value(_p, _cur_v if np.isfinite(_cur_v) else 0.0)
                                except Exception:
                                    _pf_set(model, _p, 'holdfast', True)
                                _auto_holdfasted.append(_p)
                                logger.warning(
                                    f"_estimate: auto-holdfast {_p} - "
                                    f"utility column '{_cn}' has zero variance"
                                )
                            break
                if _auto_holdfasted:
                    logger.warning(
                        f"_estimate: auto-holdfasted {len(_auto_holdfasted)} zero-variance params: "
                        f"{_auto_holdfasted}"
                    )
    except Exception as _zv_e:
        logger.debug(f"_estimate: zero-variance detection failed: {_zv_e}")

    # Snapshot starting values so unmoved ("stuck") parameters can be flagged
    # after estimation — a free param that ends exactly at its start value has
    # a zero gradient (constant/absent utility column) and is not identified.
    try:
        _start_vals = model.pf['value'].astype(float).copy()
    except Exception:
        _start_vals = None

    # ── Elastic-net (L1+L2) penalty on the estimation objective ───────────
    # Mirrors metacountregressor's penalized-likelihood objective (see
    # larch_regularization.py) so Larch-estimated coefficients get the same
    # shrinkage toward zero as the metacountregressor-fitted count models
    # elsewhere in this pipeline. Patches model.logloss/d_logloss (the
    # objective/gradient scipy.optimize.minimize actually calls here, since
    # set_cap() above gives every parameter finite bounds and Larch picks
    # SLSQP over BHHH). Restored to the true unpenalized likelihood before
    # std errors/covariance are computed below.
    # Nested-logit identification priors: when the model carries nesting
    # parameters, ridge-shrink Mu_* toward 1.0 (flat/MNL limit) and ASC_*
    # toward 0 with heavier weights than slopes. Job 23880279's unpenalized
    # fits collapsed Mu_Car to its 0.10 bound and blew ASC_SR up to +1.96
    # (SE~12) - statistically unidentified values that predicted SR=96% /
    # DA=2.6% for nonmand. Pure L2 (l1_ratio=0) - we shrink, never select.
    _nl_ref, _nl_w = nl_identification_priors(model)
    if any(str(k).startswith('Mu_') for k in _nl_ref):
        _ennet_ctx = elasticnet_objective(model, alpha=NL_PRIOR_ALPHA,
                                          l1_ratio=0.0,
                                          reference=_nl_ref, weights=_nl_w)
    else:
        _ennet_ctx = elasticnet_objective(model)
    _ennet_ctx.__enter__()

    # ── Phase 1: core terms only (industry/edu holdfasted if two_phase) ───
    # tighter gtol → optimizer walks further from the boundary so the Hessian
    # at convergence is better conditioned and std errors less likely to be NaN.
    est_options = {'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-6}
    _n_free_now = sum(1 for p in model.pf.index
                      if not model.pf.loc[p].get('holdfast', False))
    print(f"    _estimate phase-1: {_n_free_now} free params, maxiter={est_options['maxiter']} ...")
    try:
        result = model.maximize_loglike(options=est_options)
        #todo : why candidate pool mode=5 and dest = 2
        print(f"    _estimate phase-1 done: LL={getattr(result, 'loglike', '?')}")
    except Exception as _mll_e:
        logger.warning(
            f"_estimate phase-1 maximize_loglike failed: {_mll_e!r}\n"
            "  Trying with relaxed tolerances (ftol=1e-6, gtol=1e-4)..."
        )
        try:
            result = model.maximize_loglike(
                options={'maxiter': 600, 'ftol': 1e-6, 'gtol': 1e-4}
            )
        except Exception as _mll_e2:
            # ── Emit detailed diagnostics so the failure can be diagnosed ──
            # Common causes: near-constant utility columns (zero variance),
            # insufficient choice variation (all choices the same mode),
            # collinear predictors, or NaN/Inf in the data.
            _fail_reason = []
            try:
                _pf_idx = list(model.pf.index) if hasattr(model, 'pf') else []
                _fail_reason.append(f"n_params={len(_pf_idx)}")
            except Exception:
                pass
            try:
                _dt = model.datatree if hasattr(model, 'datatree') else None
                if _dt is not None and hasattr(_dt, '__getitem__'):
                    _tour = _dt['tour'].to_dataframe()
                    _fail_reason.append(f"n_cases={len(_tour)}")
                    # Check for degenerate columns
                    _numeric = _tour.select_dtypes(include=[np.number])
                    _numeric = _numeric.loc[:, ~_numeric.columns.duplicated()]
                    _zero_var = [c for c in _numeric.columns
                                 if _numeric[c].std() < 1e-6 and _numeric[c].mean() != 0]
                    if _zero_var:
                        _fail_reason.append(f"zero-variance cols={_zero_var[:5]}")
                    _nan_cols = [c for c in _numeric.columns
                                 if _numeric[c].isna().any()]
                    if _nan_cols:
                        _fail_reason.append(f"NaN cols={_nan_cols[:5]}")
                    # Check choice variation
                    if 'tour_mode' in _tour.columns:
                        _mode_counts = _tour['tour_mode'].value_counts().to_dict()
                        _fail_reason.append(f"mode_counts={_mode_counts}")
                        if len(_mode_counts) < 2:
                            _fail_reason.append("ONLY 1 MODE IN DATA - NL cannot estimate")
            except Exception as _diag_e:
                _fail_reason.append(f"diagnostic failed: {_diag_e}")
            logger.error(
                f"_estimate phase-1 fallback also failed: {_mll_e2!r}\n"
                f"  Diagnosis: {'; '.join(_fail_reason)}\n"
                "  Model will have initial values only - predictions will be weak.\n"
                "  To fix: check that HHTS estimation data has ≥2 modes, no zero-variance\n"
                "  columns, and that skim values are non-zero for a reasonable fraction of OD pairs."
            )
            # Don't raise - return model with initial values so pipeline continues
            result = type('_FakeResult', (), {'loglike': float('nan')})()

    # ── Zero-gradient engine fallback ─────────────────────────────────────
    # Job 23540863: every destination model (and the nonmand MXL) finished
    # phase-1 with ALL free params exactly at their starting values and LL
    # unchanged to machine precision — the jax compute engine handed SLSQP a
    # zero gradient, so the optimizer stopped at iteration 0 and the
    # "estimated" coefficients were just the hand-set priors. The mode MNLs
    # (numba engine) estimated fine in the same run. The person×zone
    # interaction columns that originally forced compute_engine='jax' in
    # _build_dest_model are excluded from the flat UEC now, so numba is a
    # viable retry. Detect the all-stuck outcome and re-run phase-1 once on
    # the numba engine; keep the (prior-valued) phase-1 result if that also
    # fails.
    def _all_free_params_stuck() -> bool:
        if _start_vals is None:
            return False
        try:
            _pf_now = model.pf
            _hf_now = (_pf_now['holdfast'].fillna(False).astype(bool)
                       if 'holdfast' in _pf_now.columns
                       else pd.Series(False, index=_pf_now.index))
            _free = [p for p in _pf_now.index if not _hf_now.get(p, False)]
            if not _free:
                return False
            _vals = _pf_now['value'].astype(float)
            return all(
                p in _start_vals.index and np.isfinite(_vals[p])
                and abs(_vals[p] - float(_start_vals[p])) < 1e-12
                for p in _free
            )
        except Exception:
            return False

    _cur_engine = str(getattr(model, 'compute_engine', '') or '')
    # Bidirectional: jax-stuck models retry on numba (the destination-model
    # failure mode), numba-stuck models retry on jax (job 23726229: the
    # nonmand mode NL zero-gradiented on numba — all 32 params incl. every
    # ASC — and was previously left at priors with only ASC calibration
    # papering over it).
    _retry_engine = 'numba' if 'numba' not in _cur_engine.lower() else 'jax'
    if _all_free_params_stuck():
        logger.warning(
            f"_estimate: ALL free params unmoved after phase-1 (zero gradient, "
            f"engine={_cur_engine or '?'}) — retrying once with "
            f"compute_engine='{_retry_engine}'"
        )
        # Unpatch the elastic-net wrapper BEFORE touching the engine: its
        # closures call the phase-1 engine's logloss/d_logloss, so leaving it
        # active would route the retry's gradient straight back through the
        # broken jax path (and re-wrapping on top would stack the penalty
        # twice). Re-applied on the new engine below so the retry optimizes
        # the same penalized objective as phase-1/phase-2.
        _ennet_ctx.__exit__(None, None, None)
        try:
            model.compute_engine = _retry_engine
            try:
                model.mangle()
                model.unmangle()
            except Exception:
                pass
        except Exception as _zg_e:
            logger.warning(
                f"_estimate: could not switch to {_retry_engine} engine ({_zg_e!r}) — "
                "keeping phase-1 (prior) values on the previous engine"
            )
            try:
                model.compute_engine = _cur_engine or 'jax'
                model.mangle()
                model.unmangle()
            except Exception:
                pass
        # Re-apply the elastic-net penalty on whichever engine is now active.
        _ennet_ctx = elasticnet_objective(model)
        _ennet_ctx.__enter__()
        try:
            result = model.maximize_loglike(options=est_options)
            print(f"    _estimate {_retry_engine} retry done: LL={getattr(result, 'loglike', '?')}")
            if _all_free_params_stuck():
                logger.error(
                    f"_estimate: params STILL unmoved after {_retry_engine} retry — "
                    "model is not identified on this data (constant/missing "
                    "utility columns), coefficients remain at starting priors."
                )
            else:
                logger.warning(
                    f"_estimate: {_retry_engine} retry moved the parameters — "
                    f"{_cur_engine or 'the previous'} engine gradient is broken "
                    "for this model family."
                )
        except Exception as _zg_e2:
            logger.warning(
                f"_estimate: {_retry_engine} zero-gradient retry failed ({_zg_e2!r}) — "
                "keeping phase-1 (prior) values"
            )

    # ── Phase 2 (dest model only): release holdfasted attractor terms ─────
    if two_phase:
        _RELEASE = {
            'c_indp_finance', 'c_indp_health', 'c_indp_retail',
            'c_edu_post', 'c_edu_primary', 'c_edu_secondary',
            # c_log_distwalk / c_log_distwalk_nm are no longer in DEST_UEC_FLAT
            # (DISTWALK==DIST collinearity).  Left in _RELEASE so that old pkl
            # files with these params are handled gracefully; the guard
            # `if _p in model.pf.index` will skip them in new runs.
            'c_log_distwalk', 'c_log_distwalk_nm',
        }
        released = []
        for _p in _RELEASE:
            if _p in model.pf.index:
                _released_ok = False
                # Method A: direct _parameter_bucket access (preferred, Larch ≥5.5)
                try:
                    model.unmangle()
                    bucket = model._parameter_bucket
                    names = list(bucket.pnames)
                    idx = names.index(_p)
                    bucket.pholdfast[idx] = 0
                    bucket.pminimum[idx] = -8.0
                    bucket.pmaximum[idx] = 8.0
                    _released_ok = True
                except Exception:
                    pass
                # Method B: pf-based holdfast toggle (older Larch / ParameterFrame)
                if not _released_ok:
                    try:
                        _pf_set(model, _p, 'holdfast', 0)
                        _pf_set(model, _p, 'minimum', -8.0)
                        _pf_set(model, _p, 'maximum', 8.0)
                        _released_ok = True
                    except Exception:
                        pass
                # Method C: set_value with bounds - at least un-freezes the value
                if not _released_ok:
                    try:
                        _cur = float(model.pf.loc[_p, 'value'])
                        model.set_value(_p, _cur if np.isfinite(_cur) else 0.0)
                        _released_ok = True
                    except Exception:
                        pass
                if _released_ok:
                    released.append(_p)
        if released:
            logger.info(f"_estimate phase-2: released {released} - re-estimating")
            print(f"    _estimate phase-2: released {released}, re-estimating ...")
            try:
                result = model.maximize_loglike(options=est_options)
                print(f"    _estimate phase-2 done: LL={getattr(result, 'loglike', '?')}")
            except Exception as _ph2_e:
                logger.warning(
                    f"_estimate phase-2 maximize_loglike failed: {_ph2_e!r} - "
                    "keeping phase-1 values"
                )

    # ── Post-estimation sign enforcement ─────────────────────────────────────
    # If the optimizer escaped the bounds for travel-time parameters (can happen
    # when L-BFGS-B bounds are not propagated correctly through set_cap), clip
    # them here so downstream validity checks don't fire false alarms.
    _SIGN_NEG_PARAMS = ('c_bike_time', 'c_walk_time', 'c_ivt_auto',
                        'c_ivt_transit', 'c_bike', 'c_walk', 'c_ivt')
    try:
        for _sn in _SIGN_NEG_PARAMS:
            if _sn in model.pf.index:
                _sv = float(model.pf.loc[_sn, 'value'])
                if not np.isnan(_sv) and _sv > -0.001:
                    try:
                        model.set_value(_sn, -0.001)
                    except Exception:
                        _pf_set(model, _sn, 'value', -0.001)
                    logger.debug(
                        f"_estimate: post-clip {_sn}={_sv:.4f} → -0.001 "
                        "(sign constraint enforced after optimization)"
                    )
    except Exception as _sc_e:
        logger.debug(f"_estimate: post-clip sign constraint failed: {_sc_e}")

    # ── Snapshot estimated coefficients BEFORE any JAX JIT side-effects ────────
    # model.pf is a Larch property that reconstructs from internal parameter
    # storage.  After JAX JIT compiles the log-likelihood gradient, the internal
    # storage can be clobbered (all values become NaN on next access).  We grab
    # a plain-Python dict of coefficient values here, immediately after
    # maximize_loglike() returns and before anything else can trigger JIT.
    # This dict is stored as model._coef_cache so _extract_coef_early() can
    # always find valid coefficients for the numpy logsum fallback.
    try:
        _coef_snap = {}
        for _p in model.pf.index:
            try:
                _v = float(model.pf.loc[_p, 'value'])
                if not np.isnan(_v):
                    _coef_snap[str(_p)] = _v
            except Exception:
                pass
        if _coef_snap:
            model._coef_cache = _coef_snap
            logger.debug(f"_estimate: coefficient snapshot saved "
                         f"({len(_coef_snap)} params) before JAX JIT side-effects")
    except Exception as _snap_e:
        logger.debug(f"_estimate: coefficient snapshot failed: {_snap_e}")

    # ── Pre-compute OPG scores while DataTree is still attached ──────────────
    # _opg_stderr calls model.d_loglike_casewise() which requires an active
    # DataTree.  If we wait until _compute_inference() the DataTree may have
    # been stripped, causing OPG to silently return all-NaN and forcing the
    # proxy fill.  Cache the OPG SE series immediately after estimation.
    _opg_se_early = pd.Series(np.nan, index=model.pf.index)
    try:
        # Ensure model is in a callable state
        try:
            model.mangle()
            model.unmangle()
        except Exception:
            pass
        _opg_se_early = _opg_stderr(model)
        if _opg_se_early.notna().any():
            logger.debug("Pre-estimation OPG scores cached successfully")
    except Exception as _opg_pre_e:
        logger.debug(f"Pre-estimation OPG cache failed: {_opg_pre_e}")

    def _compute_inference() -> None:
        try:
            model._cached_ll = float(result.loglike)
        except Exception:
            try:
                model._cached_ll = float(model.loglike())
            except Exception:
                model._cached_ll = float('nan')

        try:
            model.calculate_parameter_covariance()
        except Exception as e:
            logger.warning(f"Covariance calculation failed ({e}) - trying OPG fallback")

        try:
            needs_opg = (
                'std_err' not in model.pf.columns or
                model.pf['std_err'].isna().any() or
                (model.pf['std_err'] <= 0).any()
            )
            if needs_opg:
                # Use pre-cached OPG SE if available, otherwise re-try live
                se_opg = _opg_se_early if _opg_se_early.notna().any() else _opg_stderr(model)
                if se_opg.isna().all():
                    logger.debug("OPG std errors unavailable - trying diagonal-curvature fallback")
                    se_opg = _diag_hessian_stderr(model, ll0=getattr(model, '_cached_ll', np.nan))
                if se_opg.isna().all():
                    logger.debug("Diagonal fallback also failed - trying scipy sandwich Hessian")
                    se_opg = _scipy_sandwich_stderr(model)
                model.pf['std_err_opg'] = se_opg.reindex(model.pf.index).values
                with np.errstate(divide='ignore', invalid='ignore'):
                    t_opg = model.pf['value'] / model.pf['std_err_opg'].replace(0, np.nan)
                model.pf['t_stat_opg'] = t_opg.values
                model.pf['pval_opg'] = 2.0 * norm.sf(np.abs(model.pf['t_stat_opg'].astype(float)))
                if 'std_err' not in model.pf.columns:
                    model.pf['std_err'] = np.nan
                if 't_stat' not in model.pf.columns:
                    model.pf['t_stat'] = np.nan
                if 'pval' not in model.pf.columns:
                    model.pf['pval'] = np.nan
                bad_se = model.pf['std_err'].isna() | (model.pf['std_err'] <= 0)
                bad_t = model.pf['t_stat'].isna()
                bad_p = model.pf['pval'].isna()
                model.pf.loc[bad_se, 'std_err'] = model.pf.loc[bad_se, 'std_err_opg']
                model.pf.loc[bad_t, 't_stat'] = model.pf.loc[bad_t, 't_stat_opg']
                model.pf.loc[bad_p, 'pval'] = model.pf.loc[bad_p, 'pval_opg']
                n_ok = int(model.pf['std_err_opg'].notna().sum())
                logger.info(f"OPG std errors computed for {n_ok}/{len(model.pf)} params "
                            f"(used to fill missing/non-positive Hessian std errors)")
        except Exception as _e:
            logger.debug(f"OPG fallback failed: {_e}")

        if 'std_err' in model.pf.columns:
            holdfast = model.pf['holdfast'] if 'holdfast' in model.pf.columns else pd.Series(False, index=model.pf.index)
            fixed = holdfast.fillna(False).astype(bool)
            model.pf.loc[fixed, 'std_err'] = 0.0
            model.pf.loc[fixed, 't_stat'] = 0.0
            model.pf.loc[fixed, 'pval'] = 1.0

    _compute_inference()

    # If sparse destination-specific terms are still unidentified, lock them at
    # zero and re-estimate once so the core destination model has usable SEs.
    try:
        pf = model.pf
        if two_phase and 'c_size_term' in pf.index and 'std_err' in pf.columns:
            sparse_terms = {
                'c_indp_finance', 'c_indp_health', 'c_indp_retail',
                'c_edu_post', 'c_edu_primary', 'c_edu_secondary',
                'c_walk_acc', 'c_transit_acc', 'c_noveh_walk_acc', 'c_noveh_transit_acc',
            }
            bad = pf.index[(pf['std_err'].isna() | (pf['std_err'] <= 0))]
            to_fix = [name for name in bad if name in sparse_terms]
            if to_fix:
                logger.warning(f"Destination inference unstable for {to_fix} - refitting with those terms fixed at 0")
                for name in to_fix:
                    model.lock_value(name, 0.0)
                result = model.maximize_loglike(options=est_options)
                _compute_inference()
    except Exception as _e:
        logger.debug(f"Destination stabilization refit skipped: {_e}")

    # ── Drop stuck (never-moved) components and refit without them ─────────
    # User directive: a FREE coefficient that ends exactly at its starting value
    # has a zero gradient - its utility column is constant / absent / perfectly
    # collinear, so the term is not identified and contributes only a garbage SE
    # and a dead entry in the coefficient table. Remove those components
    # (holdfast at 0.0, which zeroes their utility contribution) and refit once
    # so the reported model reflects only identified terms. The dropped names
    # are exposed on the model (unioned into _stuck_params below) so the SA /
    # bandit search penalises them via update_feature_weakness and stops
    # re-proposing them on the next iteration. Disable with STAGE5_DROP_STUCK=0.
    _dropped_stuck: list = []
    try:
        if _start_vals is not None and os.environ.get('STAGE5_DROP_STUCK', '1') != '0':
            # Structural terms are never dropped even if they appear unmoved:
            # ASCs anchor the mode constants, the Mu nest scales are kept free
            # per the nest policy, and the destination backbone (size term,
            # Manski-Lerman correction, logsum, distance) must always remain.
            _NEVER_DROP = {
                'Mu_Root', 'Mu_Car', 'Mu_Motor', 'Mu_NonMotor',
                'ASC_SR', 'ASC_Walk', 'ASC_Bike', 'ASC_Transit',
                # Level-of-service impedance backbone (mode model)
                'c_ivt_auto', 'c_ivt_transit', 'c_walk_time', 'c_bike_time',
                # Destination backbone
                'c_size_term', 'log_correction',
                'c_logsum', 'c_log_dist', 'c_dist',
            }
            _pf_d = model.pf
            _hf_d = (_pf_d['holdfast'].fillna(False).astype(bool)
                     if 'holdfast' in _pf_d.columns
                     else pd.Series(False, index=_pf_d.index))
            _cur_d = _pf_d['value'].astype(float)
            # Only ZERO-drop a stuck component when it is ALSO effectively zero.
            # A parameter stuck at a NON-zero value (e.g. an availability/ASC-like
            # term the optimiser left at its start) carries real utility — forcing
            # it to 0 floored the chosen alternative to ~0 probability and blew the
            # refit LL to -4.5e34 (poisoning the search fitness AND leaving the
            # final model degenerate → singular Hessian → all-OPG t≈const p-values).
            _drop_zero = float(os.environ.get('STAGE5_DROP_STUCK_ZERO', '0.05'))
            _dropped_stuck = [
                p for p in _pf_d.index
                if not _hf_d.get(p, False)
                and p not in _NEVER_DROP
                and p in _start_vals.index
                and np.isfinite(_cur_d[p])
                and abs(_cur_d[p] - float(_start_vals[p])) < 1e-12
                and abs(_cur_d[p]) < _drop_zero          # dead term only
            ]
            if _dropped_stuck:
                # Snapshot the pre-drop optimum so a degrading refit can be reverted.
                _predrop_ll = float(getattr(result, 'loglike', float('nan')))
                _predrop_vals = _cur_d.copy()
                logger.warning(
                    f"_estimate: dropping {len(_dropped_stuck)} stuck near-zero "
                    f"component(s) and refitting without them: {_dropped_stuck}"
                )
                for _p in _dropped_stuck:
                    try:
                        model.lock_value(_p, 0.0)
                    except Exception:
                        try:
                            model.set_value(_p, 0.0)
                        except Exception:
                            pass
                        _pf_set(model, _p, 'holdfast', True)
                try:
                    result = model.maximize_loglike(options=est_options)
                    _new_ll = float(getattr(result, 'loglike', float('nan')))
                    # Revert if the refit is non-finite, the blown-LL sentinel, or
                    # materially worse than the pre-drop fit. Dropping a genuinely
                    # dead term should leave LL essentially unchanged.
                    if (not np.isfinite(_new_ll) or _new_ll < -1e12
                            or (np.isfinite(_predrop_ll)
                                and _new_ll < _predrop_ll - 1e-3)):
                        logger.warning(
                            f"_estimate: stuck-drop refit worsened LL "
                            f"({_predrop_ll:.2f} → {_new_ll:.2e}) — reverting to "
                            "the pre-drop fit (terms left free)")
                        for _p in _dropped_stuck:
                            try:
                                _pf_set(model, _p, 'holdfast', False)
                            except Exception:
                                pass
                        for _p in _predrop_vals.index:
                            try:
                                model.set_value(_p, float(_predrop_vals[_p]))
                            except Exception:
                                pass
                        _compute_inference()
                    else:
                        _compute_inference()
                        print(f"    _estimate stuck-drop refit done: LL={_new_ll}")
                except Exception as _rf_e:
                    logger.warning(
                        f"_estimate: stuck-drop refit failed ({_rf_e!r}) - "
                        "keeping pre-drop fit with those terms fixed at 0"
                    )
    except Exception as _drop_e:
        logger.debug(f"_estimate: stuck-drop step skipped: {_drop_e}")
        _dropped_stuck = []

    # Restore the true (unpenalized) logloss/d_logloss now that all
    # maximize_loglike() calls for this model are done - everything below
    # (std errors, covariance, diagnostics) must reflect the real likelihood.
    _ennet_ctx.__exit__(None, None, None)

    # RE-COMPUTE inference on the UNPENALIZED likelihood. Every _compute_inference()
    # call above ran while the elastic-net (L1/L2) penalty was still active, so
    # calculate_parameter_covariance() saw the penalized objective — its Hessian is
    # ill-defined under the non-smooth L1 term, which failed silently (every fit
    # logged "Hessian: 0") and fell back to OPG on the penalized scores, producing
    # the degenerate near-constant t≈0.65 (all coefficients spuriously
    # "insignificant"). Recomputing here, with the penalty removed, gives the real
    # observed-information Hessian and correct standard errors / p-values.
    try:
        _compute_inference()
    except Exception as _post_inf_e:
        logger.debug(f"_estimate: unpenalized inference recompute failed ({_post_inf_e})")

    # ── Build definitive parameter frame (nuclear guarantee + final cache) ──────
    #
    # IMPORTANT: in Larch 6.x + pandas 2.x, model.pf is a PROPERTY that
    # reconstructs a DataFrame from internal Larch parameter storage on each
    # access.  Assignments like:
    #
    #     model.pf.loc[mask, 'std_err'] = values   # ← DOES NOT PERSIST
    #
    # only modify the temporary DataFrame returned by the property - the
    # underlying Larch parameter store is unchanged, and the next call to
    # model.pf returns fresh NaN values again.
    #
    # Solution: snapshot model.pf ONCE into a plain local DataFrame (_pf),
    # ── Stuck / at-bound parameter diagnostics ────────────────────────────
    # Free params that never moved from their starting values (zero gradient →
    # unidentified: constant utility column, missing data merge, or placeholder
    # zero-fill) and params pinned at the ±cap or a bound.  These are the
    # coefficients that show up downstream with garbage SEs / p-values.
    try:
        if _start_vals is not None:
            _pf_chk = model.pf
            _hf_chk = (_pf_chk['holdfast'].fillna(False).astype(bool)
                       if 'holdfast' in _pf_chk.columns
                       else pd.Series(False, index=_pf_chk.index))
            _cur = _pf_chk['value'].astype(float)
            _stuck = [p for p in _pf_chk.index
                      if not _hf_chk.get(p, False)
                      and p in _start_vals.index
                      and np.isfinite(_cur[p])
                      and abs(_cur[p] - float(_start_vals[p])) < 1e-12]
            _at_bound = []
            for p in _pf_chk.index:
                if _hf_chk.get(p, False) or not np.isfinite(_cur[p]):
                    continue
                _mx = float(_pf_chk.loc[p].get('maximum', np.nan)) if 'maximum' in _pf_chk.columns else np.nan
                _mn = float(_pf_chk.loc[p].get('minimum', np.nan)) if 'minimum' in _pf_chk.columns else np.nan
                if (np.isfinite(_mx) and abs(_cur[p] - _mx) < 1e-9) or \
                   (np.isfinite(_mn) and abs(_cur[p] - _mn) < 1e-9):
                    _at_bound.append(p)
            if _stuck:
                logger.warning(
                    f"_estimate: {len(_stuck)} free params NEVER MOVED from their "
                    f"starting values (zero gradient — likely constant/missing "
                    f"utility column; not identified): {_stuck}"
                )
            if _at_bound:
                logger.warning(
                    f"_estimate: {len(_at_bound)} params ended AT A BOUND "
                    f"(estimate is censored; SE/p-value unreliable): {_at_bound}"
                )
            # Expose to callers (e.g. ga_external_estimators._larch_coef_rows)
            # so the GA/SA/Bandit search can bias perturbations away from
            # these params - a param stuck at its start value or pinned at a
            # bound is stronger evidence of a bad specification than mere
            # statistical insignificance.
            # Union in components dropped above: they've been holdfast at 0 so
            # they no longer appear as free-and-stuck here, but the SA search
            # still needs to see them flagged so it stops re-proposing the
            # feature that produced a dead term.
            model._stuck_params = list(dict.fromkeys(list(_stuck) + list(_dropped_stuck)))
            model._at_bound_params = list(_at_bound)
            model._dropped_params = list(_dropped_stuck)
    except Exception as _sd_e:
        logger.debug(f"_estimate: stuck-param diagnostic failed: {_sd_e}")

    # apply all fills to _pf, then store _pf as model._final_pf.
    # All downstream code (CSV, HTML, validation) uses _final_pf exclusively.
    try:
        _pf = model.pf.copy()   # snapshot - safe to modify in-place

        # Ensure required columns exist
        for _col in ('std_err', 't_stat', 'pval'):
            if _col not in _pf.columns:
                _pf[_col] = np.nan
        if 'std_err_source' not in _pf.columns:
            _pf['std_err_source'] = 'hessian_opg'

        _holdfast = (_pf['holdfast'] if 'holdfast' in _pf.columns
                     else pd.Series(False, index=_pf.index))
        _free = ~_holdfast.fillna(False).astype(bool)

        # Stamp holdfast params with 0.0 / 1.0
        _pf.loc[~_free, 'std_err'] = 0.0
        _pf.loc[~_free, 't_stat']  = 0.0
        _pf.loc[~_free, 'pval']    = 1.0
        _pf.loc[~_free, 'std_err_source'] = 'holdfast'

        # Merge OPG SE if available (pre-cached before Phase 2 re-estimation)
        if _opg_se_early.notna().any():
            _opg_vals = _opg_se_early.reindex(_pf.index)
            _needs_opg = _free & (_pf['std_err'].isna() | (_pf['std_err'] <= 0))
            _have_opg  = _needs_opg & _opg_vals.notna() & (_opg_vals > 0)
            if _have_opg.any():
                _pf.loc[_have_opg, 'std_err'] = _opg_vals[_have_opg].values
                _pf.loc[_have_opg, 'std_err_source'] = 'opg'
                with np.errstate(divide='ignore', invalid='ignore'):
                    _pf.loc[_have_opg, 't_stat'] = (
                        _pf.loc[_have_opg, 'value'].astype(float).values /
                        _pf.loc[_have_opg, 'std_err'].astype(float).values
                    )
                _pf.loc[_have_opg, 'pval'] = (
                    2.0 * norm.sf(np.abs(_pf.loc[_have_opg, 't_stat'].astype(float)))
                )

        # Nuclear guarantee: fill any remaining NaN/zero std_err with proxy
        _bad = _free & (_pf['std_err'].isna() | (_pf['std_err'] <= 0))
        if _bad.any():
            _good_se = _pf.loc[_free & ~_bad, 'std_err'].dropna()
            _proxy_base = float(np.nanmedian(_good_se)) if len(_good_se) > 0 else 0.25
            _proxy_base = max(_proxy_base, 0.25)
            _vals = _pf.loc[_bad, 'value'].astype(float)
            _proxy = np.maximum(np.abs(_vals) * 0.50, _proxy_base).clip(lower=0.25)
            _pf.loc[_bad, 'std_err'] = _proxy.values
            _pf.loc[_bad, 'std_err_source'] = 'proxy'
            with np.errstate(divide='ignore', invalid='ignore'):
                _t = _vals.values / _proxy.values
            _pf.loc[_bad, 't_stat'] = np.where(np.isfinite(_t), _t, 0.0)
            _pf.loc[_bad, 'pval'] = 2.0 * norm.sf(np.abs(_pf.loc[_bad, 't_stat']))
            logger.warning(
                f"SE proxy fill for {int(_bad.sum())} params "
                f"({list(_pf.index[_bad])}): "
                f"proxy_base={_proxy_base:.3f}. "
                "These SEs are NOT from the Hessian - model may be unidentified."
            )

        # Cache definitive frame on the model object so it survives pickle/reload
        model._final_pf = _pf

        # ── SE diagnostic summary ──────────────────────────────────────────
        _n_nan  = int(_pf.loc[_free, 'std_err'].isna().sum())
        _n_prx  = int((_pf.loc[_free, 'std_err_source'] == 'proxy').sum())
        _n_opg  = int((_pf.loc[_free, 'std_err_source'] == 'opg').sum())
        _n_hess = int(_free.sum()) - _n_prx - _n_opg - _n_nan
        _ll_val = getattr(model, '_cached_ll', float('nan'))
        _src_counts = _pf.loc[_free, 'std_err_source'].value_counts().to_dict()
        print(
            f"\n  ┌─ SE diagnostic ─────────────────────────────────────────\n"
            f"  │  LL = {_ll_val:.2f}\n"
            f"  │  Free params: {int(_free.sum())} "
            f"(Hessian: {_n_hess}, OPG: {_n_opg}, proxy: {_n_prx}, NaN: {_n_nan})\n"
            f"  │  SE sources: {_src_counts}\n"
            f"  └─────────────────────────────────────────────────────────"
        )
        if _n_nan:
            logger.error(
                f"_estimate: {_n_nan} free parameter(s) still have NaN std_err "
                f"in _final_pf - this should never happen; report as a bug."
            )
        if _n_prx > 0:
            logger.warning(
                f"  Proxy SEs for: {list(_pf.index[_free & (_pf['std_err_source']=='proxy')])} "
                "- parameters not identified from Hessian/OPG. "
                "Consider holding them fast or reviewing data coverage."
            )
    except Exception as _nuke_e:
        logger.warning(f"_estimate: final SE fill/cache failed: {_nuke_e!r}")
        import traceback as _tb; logger.debug(_tb.format_exc())
        # Best-effort: cache whatever pf we have
        try:
            model._final_pf = model.pf.copy()
        except Exception:
            pass

    return model


# ===========================================================================
# POST-ESTIMATION REFIT RULES
#   1. _prune_and_refit          — drop zero-fit / insignificant terms, refit
#   2. _collapse_generic_groups  — swap a collinear ASV block for one mode-
#                                  generic coefficient, keep the better BIC
#   3. _apply_sign_constraints   — enforce declarative sign priors (build+refit)
# All three read their rules from ga_constraints so the search space, the model
# build, and the refit share one source of truth.
# ===========================================================================

def _model_ll(model) -> float:
    """Log-likelihood of a fitted Larch model (cached value preferred)."""
    try:
        v = getattr(model, '_cached_ll', None)
        if v is not None and np.isfinite(float(v)):
            return float(v)
    except Exception:
        pass
    try:
        return float(model.loglike())
    except Exception:
        return float('nan')


def _model_n_free(model) -> int:
    """Number of free (non-holdfast) parameters."""
    try:
        pf = model.pf
        hf = pf.get('holdfast', pd.Series(False, index=pf.index)).astype(bool)
        return int((~hf).sum())
    except Exception:
        return 0


def _model_n_cases(model) -> int:
    """Estimation sample size (choice observations)."""
    for _attr in ('n_cases',):
        try:
            v = getattr(model, _attr, None)
            if v:
                return int(v)
        except Exception:
            pass
    for _acc in (lambda m: m.dataframes.n_cases,
                 lambda m: len(m.dataset['_caseid_']),
                 lambda m: m.datatree.n_cases):
        try:
            v = _acc(model)
            if v:
                return int(v)
        except Exception:
            pass
    return 0


def _model_bic(model, ll=None, n=None) -> float:
    """Bayesian information criterion (lower is better). inf when LL is NaN."""
    ll = _model_ll(model) if ll is None else ll
    if ll is None or not np.isfinite(ll):
        return float('inf')
    k = _model_n_free(model)
    n = _model_n_cases(model) if n is None else n
    n_eff = n if (n and n > 1) else max(k + 2, 2)
    return -2.0 * float(ll) + k * float(np.log(n_eff))


def _param_tstats(model) -> dict:
    """{param_name: t-statistic} from the model's final parameter frame.

    Prefers value/std_err; falls back to any precomputed t-stat column
    (t_stat, t_stat_opg) or a p-value column (pval, pval_opg -> |t|) so that
    insignificance pruning still fires when std_err isn't on the frame being
    read (the silent case that left insignificant coefficients un-pruned)."""
    out: dict = {}
    try:
        pf = getattr(model, '_final_pf', None)
        if pf is None:
            pf = model.pf
        _cols = set(pf.columns)
        for p in pf.index:
            t = float('nan')
            try:
                v = float(pf.loc[p, 'value']) if 'value' in _cols else float('nan')
                se = float(pf.loc[p, 'std_err']) if 'std_err' in _cols else float('nan')
                if np.isfinite(v) and np.isfinite(se) and se > 0:
                    t = v / se
            except Exception:
                t = float('nan')
            if not np.isfinite(t):
                for _tc in ('t_stat', 't_stat_opg'):
                    if _tc in _cols:
                        try:
                            _tv = float(pf.loc[p, _tc])
                            if np.isfinite(_tv):
                                t = _tv
                                break
                        except Exception:
                            pass
            if not np.isfinite(t):
                for _pc in ('pval', 'pval_opg'):
                    if _pc in _cols:
                        try:
                            _pv = float(pf.loc[p, _pc])
                            if np.isfinite(_pv):
                                t = float(norm.isf(min(max(_pv, 1e-12), 1.0) / 2.0))
                                break
                        except Exception:
                            pass
            out[p] = t
    except Exception:
        pass
    return out


def _apply_sign_constraints(model, stage: str):
    """
    Enforce declarative sign priors from ga_constraints.sign_constraints(stage):
    'neg' bounds the coefficient ≤ 0 (deterrents), 'pos' bounds it ≥ 0
    (attractors). Applied at model-build time and re-applied after any refit.
    """
    try:
        from ga_constraints import sign_constraints as _sc
        signs = _sc(stage)
    except Exception:
        signs = {}
    for p, s in signs.items():
        if p not in model.pf.index:
            continue
        try:
            cur = float(model.pf.loc[p, 'value'])
        except Exception:
            cur = 0.0
        try:
            if s == 'neg':
                _hi = -0.001 if p == 'c_bike_time' else 0.0
                _pf_set(model, p, 'maximum', _hi)
                if np.isfinite(cur) and cur > _hi:
                    model.set_value(p, _hi - 0.001)
            elif s == 'pos':
                _pf_set(model, p, 'minimum', 0.0)
                if np.isfinite(cur) and cur < 0:
                    model.set_value(p, 0.001)
        except Exception:
            pass
    return model


def _prune_and_refit(model, estimate_fn, stage: str, *,
                     tmin: float = None, zero_eps: float = None,
                     max_iters: int = 4, label: str = ''):
    """
    Iterative backward elimination of zero-fit / insignificant coefficients.

    A FREE, non-`protected_backbone` parameter is dropped (locked at 0.0, so its
    utility term leaves the likelihood) when it fits to ~0 (|value| < zero_eps —
    the dead pval≈1 terms) or is statistically insignificant (|t| < tmin). The
    model is re-estimated after each round; the loop stops when nothing more is
    prunable or `max_iters` is hit. Dropping a true zero cannot reduce LL, and
    an insignificant term's LL contribution is tiny relative to the k·log(n) BIC
    saving, so removal is accepted (a BIC regression is logged, not reverted —
    the whole point of the rule is to strip statistically-dead structure).

    Threshold sourcing (shared pruning contract across ALL stages):
      * zero_eps  ← GA_SIG_COEF_ZERO   (same "near-zero coefficient" cut the
                    stage 1-4 refit loops use), overridable via GA_S5_PRUNE_EPS.
      * tmin      ← the two-sided |t| implied by GA_SIG_PVAL_THRESH (the same
                    p-value cut stages 1-4 use, e.g. p=0.10 → |t|≈1.645), or an
                    explicit GA_S5_PRUNE_TMIN if set. This keeps one knob
                    (GA_SIG_PVAL_THRESH / GA_SIG_COEF_ZERO) controlling pruning
                    aggressiveness for stages 1-5 alike.
    """
    import os as _os
    if tmin is None:
        _tmin_env = _os.environ.get('GA_S5_PRUNE_TMIN', '').strip()
        if _tmin_env:
            try:
                tmin = float(_tmin_env)
            except Exception:
                tmin = 1.6
        else:
            # Derive from the shared p-value threshold so all stages agree.
            try:
                from scipy.stats import norm as _norm
                _pth = float(_os.environ.get('GA_SIG_PVAL_THRESH', '0.10'))
                tmin = float(_norm.isf(_pth / 2.0))   # two-sided |t| for p
            except Exception:
                tmin = 1.6
    if zero_eps is None:
        _eps_env = _os.environ.get('GA_S5_PRUNE_EPS', '').strip()
        if _eps_env:
            try:
                zero_eps = float(_eps_env)
            except Exception:
                zero_eps = 1e-4
        else:
            try:
                zero_eps = float(_os.environ.get('GA_SIG_COEF_ZERO', '1e-4'))
            except Exception:
                zero_eps = 1e-4
    try:
        from ga_constraints import protected_backbone as _pb
        protected = set(_pb(stage))
    except Exception:
        protected = set()

    _tag = f"[prune {label or stage}]"
    try:
        _pf0 = getattr(model, '_final_pf', None)
        _pf0 = model.pf if _pf0 is None else _pf0
        _nfree0 = int((~_pf0.get('holdfast', pd.Series(False, index=_pf0.index))
                       .fillna(False).astype(bool)).sum())
    except Exception:
        _nfree0 = -1
    logger.info(f"{_tag} entering: {_nfree0} free params, tmin={tmin:.2f}, zero_eps={zero_eps:g}")
    for _it in range(max_iters):
        try:
            pf = getattr(model, '_final_pf', None)
            if pf is None:
                pf = model.pf
        except Exception as _pf_e:
            logger.warning(f"{_tag} cannot read param frame ({_pf_e!r}) — prune skipped")
            break
        tstats = _param_tstats(model)
        bic_before = _model_bic(model)
        drop = []
        for p in list(pf.index):
            if p in protected:
                continue
            try:
                if bool(pf.loc[p].get('holdfast', False)):
                    continue
            except Exception:
                pass
            try:
                val = float(pf.loc[p, 'value'])
            except Exception:
                continue
            t = tstats.get(p, float('nan'))
            if abs(val) < zero_eps:
                drop.append((p, 'zero'))
            elif np.isfinite(t) and abs(t) < tmin:
                drop.append((p, 'insig'))
        if not drop:
            if _it == 0:
                logger.info(f"{_tag} no zero/insignificant free params to remove")
            break
        for p, _why in drop:
            try:
                model.lock_value(p, 0.0)
            except Exception:
                _pf_set(model, p, 'value', 0.0)
                _pf_set(model, p, 'holdfast', True)
        logger.info(f"{_tag} iter {_it + 1}: removing {len(drop)} term(s) "
                    f"{[d[0] for d in drop]}")
        try:
            model = estimate_fn(model)
        except Exception as _e:
            logger.warning(f"{_tag} re-estimation failed after prune: {_e!r}")
            break
        bic_after = _model_bic(model)
        if (np.isfinite(bic_before) and np.isfinite(bic_after)
                and bic_after > bic_before + 2.0):
            logger.info(f"{_tag} BIC {bic_before:.1f}→{bic_after:.1f} after prune "
                        "(kept — removed terms were statistically dead)")
    return model


def _collapse_generic_groups(model, stage: str, rebuild_and_fit, *,
                             collinear_frac: float = 0.30, label: str = ''):
    """
    Refit substitution: replace a collinear / jointly-weak alternative-specific
    (per-mode) coefficient block with ONE mode-generic coefficient that applies
    the same effect to every alternative in the group — keeping whichever form
    has the better BIC ("the ones that best do it").

    Generalises the hardcoded income-collinearity collapse in _run_segment. The
    groups come from ga_constraints.generic_groups(stage).

    Parameters
    ----------
    rebuild_and_fit : callable(generic_genes: list[str]) -> fitted model | None
        Rebuilds the model with the given generic genes selected (their ASV
        blocks are auto-suppressed by _mode_spec_custom) and returns it fitted.
        Only meaningful for the mode model; dest choice is idca (naturally
        alternative-specific) and declares no generic groups.
    """
    try:
        from ga_constraints import generic_groups as _gg
        groups = _gg(stage)
    except Exception:
        groups = {}
    if not groups or rebuild_and_fit is None:
        return model
    try:
        pf = getattr(model, '_final_pf', None)
        if pf is None:
            pf = model.pf
    except Exception:
        return model
    tstats = _param_tstats(model)
    _tag = f"[generic {label or stage}]"
    # Compute the sample size once from the incumbent model so the ASV vs
    # generic BIC comparison uses an identical n (both fit the same data).
    _n_shared = _model_n_cases(model) or None

    eligible = []
    for gene, spec in groups.items():
        free_vals = []
        for c in spec.get('asv', []):
            if c not in pf.index:
                continue
            try:
                if bool(pf.loc[c].get('holdfast', False)):
                    continue
                v = float(pf.loc[c, 'value'])
                if np.isfinite(v):
                    free_vals.append((c, v))
            except Exception:
                pass
        if len(free_vals) < 2:
            continue
        vals = [v for _c, v in free_vals]
        mean_v = float(np.mean(vals))
        spread = float(max(vals) - min(vals))
        collinear = abs(mean_v) > 0.05 and spread < collinear_frac * abs(mean_v)
        all_weak = all(
            (not np.isfinite(tstats.get(c, float('nan'))))
            or abs(tstats.get(c, 0.0)) < 1.0
            for c, _v in free_vals
        )
        if collinear or all_weak:
            eligible.append(gene)

    if not eligible:
        return model

    bic_asv = _model_bic(model, n=_n_shared)
    logger.info(f"{_tag} collinear/weak ASV block(s) {eligible} "
                f"(ASV BIC={bic_asv:.1f}); trying mode-generic substitution")
    try:
        cand = rebuild_and_fit(eligible)
    except Exception as _e:
        logger.warning(f"{_tag} generic rebuild failed: {_e!r}")
        return model
    if cand is None:
        return model
    bic_gen = _model_bic(cand, n=(_n_shared or _model_n_cases(cand)))
    if np.isfinite(bic_gen) and bic_gen <= bic_asv:
        logger.info(f"{_tag} generic form wins BIC {bic_asv:.1f}→{bic_gen:.1f}; "
                    f"applying same effect across modes for {eligible}")
        return cand
    logger.info(f"{_tag} alternative-specific form wins "
                f"(generic BIC={bic_gen:.1f} ≥ {bic_asv:.1f}); keeping per-mode block")
    return model


def _logsums(mode_model, dt_mode, skims=None, verbose: bool = False,
               coef_override: dict = None, zone_ids_override: list = None):
    """
    Compute mode choice logsums across all destination zones.

    The mode model has 5 mode alternatives as its alts dimension.
    We need a (n_tours × n_zones) logsum matrix - one logsum per tour
    per potential destination zone.

    For each zone:
      1. Set DTAZ = that zone for all tours (xarray broadcast)
      2. Evaluate mode model → get log-sum-of-exponentials across modes
      3. Store as a column in the output DataArray

    ActivitySim pattern from trip_destination.compute_logsums:
      logsums stored once per segment, fed into destination choice as ca variable.

    zone_ids_override : optional explicit zone list. Without this, the zone
        universe below is derived from whichever DTAZ values are actually
        present in dt_mode's tours - at ESTIMATION time that's only the
        HHTS-observed destination zones (a self-selected, already-popular
        subset, e.g. 243/329 for edu), not the true full zone universe. That
        mismatch is what made post-estimation calibration (_run_segment's
        probs_cal/shadow-price/pred_zone_shares) evaluate the fitted model
        over a truncated, biased zone set while real simulation (_predict_batch)
        evaluates it over all zones - producing a much more concentrated
        distribution during calibration than in the real prediction. Pass
        the full landuse zone list here when computing logsums specifically
        for that calibration step.
    """
    import xarray as xr

    case_ids = list(dt_mode.caseids())

    # zone IDs from the OMX od sub-dataset
    zone_ids = None
    if zone_ids_override is not None:
        zone_ids = sorted({int(z) for z in zone_ids_override if int(z) > 0 and int(z) != 330})
        if len(zone_ids) < 10:
            zone_ids = None   # override was degenerate - fall through to normal derivation
    if zone_ids is None:
        try:
            od_ds = dt_mode['od']
            coord = od_ds.coords.get('dtaz', od_ds.coords.get('otaz', None))
            if coord is not None:
                zone_ids = sorted([int(z) for z in coord.values if int(z) > 0 and int(z) != 330])
        except Exception:
            pass

    if zone_ids is None:
        try:
            zone_ids = sorted(
                dt_mode['tour']['DTAZ'].to_series().unique().tolist())
            zone_ids = [int(z) for z in zone_ids if int(z) > 0 and int(z) != 330]
        except Exception:
            zone_ids = _default_zone_universe()

    # If the DataTree only has one unique DTAZ (e.g. work/edu batch where all
    # persons share the same usual-location zone) the logsum matrix would be
    # (n_cases × 1) - that's a scalar per person, not a distribution over zones.
    # Fall back to the full zone list so Sharrow sees the same DataTree shape
    # across all batches and reuses the compiled JIT flow.
    if len(zone_ids) < 10:
        # Try to get full list from the od sub-dataset coords first
        try:
            od_ds = dt_mode['od']
            coord = od_ds.coords.get('dtaz', od_ds.coords.get('otaz', None))
            if coord is not None:
                full_zones = sorted([int(z) for z in coord.values
                                     if int(z) > 0 and int(z) != 330])
                if len(full_zones) >= 10:
                    zone_ids = full_zones
        except Exception:
            pass
        # Final fallback: sequential 1-329
        if len(zone_ids) < 10:
            zone_ids = _default_zone_universe()
        logger.debug(f"_logsums: zone list was <10 - expanded to {len(zone_ids)} zones")

    if verbose:
        print(f"    Computing logsums: {len(case_ids)} cases × {len(zone_ids)} zones")

    import numpy as np
    ls_array = None

    # ── Attempt 1: vectorised Larch logsums over all zones at once ──────────
    # Larch ≥ 5.7 supports logsums() called once on the fitted DataTree and
    # returns an (n_cases,) array for the current DTAZ values.  We can't use
    # this to get a full (n_cases × n_zones) matrix in one call, but we can
    # try a quick smoke-test (one call) to see if the API is working at all
    # before attempting the per-zone loop - which is very slow on 295 zones.
    try:
        m_test = mode_model.copy()
        try:
            m_test.weight_co_var = None
        except Exception:
            pass
        # Try both API variants
        try:
            _ = np.asarray(m_test.logsums(datatree=dt_mode), dtype=np.float32)
        except TypeError:
            m_test.datatree = dt_mode
            m_test.mangle()
            m_test.unmangle()
            _ = np.asarray(m_test.logsums(), dtype=np.float32)
        # If we reach here the API works - but we still need per-zone matrix.
        # The single-call result is only for the observed DTAZs, not all zones.
        # Fall through to per-zone loop only if API is confirmed working.
        larch_api_ok = True
    except Exception as _e:
        _e_str = str(_e)
        if 'availability_any' in _e_str or 'availability_co' in _e_str:
            # Larch removed internal _availability_any attribute in a recent version.
            # This is a known API change - the numpy logsum path is functionally
            # equivalent and is the correct fallback.  Log at DEBUG only.
            logger.debug(
                f"_logsums: Larch API changed ({_e}) - using numpy logsum path (normal)."
            )
        else:
            logger.info(
                f"_logsums: Larch logsums() unavailable ({_e}) - "
                "using numpy logsum path (mode coefficients from cache)."
            )
        larch_api_ok = False

    # ── Attempt 2: per-zone Larch loop (only if API confirmed working) ──────
    if larch_api_ok:
        try:
            template = dt_mode['tour']['DTAZ']
        except Exception:
            template = None

        try:
            m_copy = mode_model.copy()
            try:
                m_copy.weight_co_var = None
            except Exception:
                pass
        except Exception:
            m_copy = mode_model

        # Helper: replace the 'tour' dataset inside a DataTree with one that has
        # DTAZ fixed at `dtaz`.  Tries three API variants across larch/xarray versions.
        def _set_dtaz(dt, dtaz_val):
            _set_dtaz.variant = 'unchanged'
            # Variant 1: larch replace_datasets (larch <5.8 / older xarray DataTree)
            if template is not None:
                try:
                    new_tour = dt['tour'].assign(DTAZ=xr.full_like(template, int(dtaz_val)))
                    _set_dtaz.variant = 'replace_datasets_node'
                    return dt.replace_datasets(tour=new_tour)
                except (AttributeError, TypeError, Exception):
                    pass
                # Variant 2: access underlying .ds Dataset, then rebuild child node
                try:
                    _ds = (getattr(dt['tour'], 'ds', None)
                           or getattr(dt['tour'], 'dataset', None))
                    if _ds is not None:
                        new_ds = _ds.assign(DTAZ=xr.full_like(_ds['DTAZ'], int(dtaz_val)))
                        _set_dtaz.variant = 'replace_datasets_dataset'
                        return dt.replace_datasets(tour=new_ds)
                except Exception:
                    pass
                # Variant 3: in-place mutation on the model copy's datatree
                # (safe because m_copy is already a copy for each outer iteration)
                try:
                    _ds = (getattr(m_copy.datatree['tour'], 'ds', None)
                           or getattr(m_copy.datatree['tour'], 'dataset', None))
                    if _ds is not None and 'DTAZ' in _ds:
                        import xarray as _xr
                        _ds['DTAZ'].values[:] = int(dtaz_val)
                        _set_dtaz.variant = 'in_place_model_copy'
                        return m_copy.datatree
                except Exception:
                    pass
            return dt

        ls_array_larch = np.zeros((len(case_ids), len(zone_ids)), dtype=np.float32)
        n_failed = 0
        _datatree_kwarg_unsupported = False
        for j, dtaz in enumerate(zone_ids):
            try:
                tmp = _set_dtaz(dt_mode, dtaz)
                ls_col = None
                try:
                    ls_col = np.asarray(m_copy.logsums(datatree=tmp), dtype=np.float32)
                except TypeError as e:
                    # logsums(datatree=...) isn't supported by the installed Larch
                    # version. The fallback below (reassigning m_copy.datatree then
                    # calling logsums()) "succeeds" but forces Larch to recompile its
                    # evaluation flow from scratch on EVERY zone - confirmed via
                    # production logs showing continuous "writing fresh flow code"
                    # output that never progressed past zone 1 of ~300 in 24h wall
                    # time. Abandon the per-zone loop after the first zone and drop
                    # straight to the numpy fallback (Attempt 3) instead of limping
                    # through the rest of the zones at that cost.
                    if j == 0:
                        logger.warning(
                            "_logsums: logsums(datatree=...) unsupported by installed Larch "
                            f"(DTAZ variant {_set_dtaz.variant}: {e}). The per-zone fallback "
                            "recompiles the model flow on every zone - abandoning the Larch "
                            "per-zone loop after zone 1 and using the numpy fallback instead."
                        )
                        _datatree_kwarg_unsupported = True
                        n_failed = len(zone_ids)
                        break
                    raise
                if ls_col is not None and not np.all(np.isnan(ls_col)):
                    ls_array_larch[:, j] = np.nan_to_num(ls_col, nan=0.0)
                else:
                    n_failed += 1
            except Exception as e:
                logger.debug(f"Logsum failed for zone {dtaz}: {e}")
                n_failed += 1

        if _datatree_kwarg_unsupported:
            pass   # ls_array stays None -> falls through to numpy fallback below
        elif n_failed < len(zone_ids):
            ls_array = ls_array_larch
        else:
            logger.error(
                "_logsums: Larch per-zone loop FAILED for every zone despite API smoke-test passing.\n"
                "  ► Falling back to numpy - mode impedance may be less accurate.\n"
                "  ► Check Larch model state and skim data consistency."
            )

    # ── Attempt 3: numpy fallback - always works regardless of Larch API ────
    # ls_array is None here when both Larch attempts failed/were skipped.
    # The numpy fallback reads OMX matrices directly and never returns None.
    if ls_array is None:
        logger.info("_logsums: using numpy fallback (direct OMX coefficient computation).")
        ls_array = _logsums_numpy_fallback(
            mode_model, dt_mode, case_ids, zone_ids, skims=skims, verbose=verbose,
            coef_override=coef_override)

    # wrap as xarray DataArray with named dims for use in _dest_tree
    ls = xr.DataArray(
        ls_array,
        dims=['cases', 'zones'],
        coords={'cases': case_ids, 'zones': zone_ids},
        name='logsum',
    )

    # sanity check - warn if logsums look degenerate
    nonzero = (ls_array != 0).mean()
    mean_ls = float(np.nanmean(ls_array))
    logger.info(f"Logsums: mean={mean_ls:.3f}, non-zero={nonzero*100:.1f}%")
    if nonzero < 0.1:
        logger.warning(
            "WARNING: >90% of logsums are zero - mode model may not be "
            "evaluating correctly. Check that tour.weight, tour.DTAZ, and "
            "skim columns are all present in the DataTree.")
        # ── Diagnostic dump ─────────────────────────────────────────────
        # Columns are missing when the DataTree was built without joining skims
        # or when tour sub-dataset lacks OTAZ/weight. Check that _build_dt_mode
        # receives a skims object and that tours have OTAZ populated from stage4.
        try:
            logger.warning("  LOGSUM DIAGNOSTIC:")
            logger.warning(f"    ls shape: {ls_array.shape}  "
                           f"sample[:2,:3]: {ls_array[:2,:3].tolist()}")

            # Build a name→dataset mapping that works across Larch/xarray versions.
            # Older xarray-datatree: dt.datasets  (dict of xr.Dataset)
            # Newer xarray DataTree: dt.children  (dict of DataTree nodes, each with .ds)
            def _dt_sub_datasets(dt):
                """Return {name: xr.Dataset} for all child nodes, trying multiple APIs."""
                try:
                    d = dt.datasets
                    if d:
                        return dict(d)
                except Exception:
                    pass
                try:
                    out = {}
                    for name, child in dt.children.items():
                        ds = getattr(child, 'ds', None) or getattr(child, 'dataset', None)
                        if ds is not None:
                            out[name] = ds
                    if out:
                        return out
                except Exception:
                    pass
                # Direct subscription fallback - try known sub-dataset names
                out = {}
                for name in ('tour', 'person', 'od', 'emp', 'base'):
                    try:
                        node = dt[name]
                        ds   = getattr(node, 'ds',      None) \
                            or getattr(node, 'dataset', None) \
                            or getattr(node, '_ds',     None)
                        if ds is not None:
                            out[name] = ds
                    except Exception:
                        pass
                return out

            sub_datasets = _dt_sub_datasets(dt_mode)
            for ds_name, ds in sub_datasets.items():
                cols = list(getattr(ds, 'data_vars', {}).keys())[:10]
                logger.warning(f"    DataTree[{ds_name}] vars: {cols}")

            for key_col in ['DTAZ','OTAZ','weight','SOVTOLL_TIME__AM',
                            'WLK_LOC_WLK_TOTIVT__AM','DISTWALK','DISTBIKE']:
                present_in = [
                    dsn for dsn, ds in sub_datasets.items()
                    if key_col in getattr(ds, 'data_vars', {})
                ]
                status = f"found in {present_in}" if present_in else "MISSING"
                logger.warning(f"    '{key_col}': {status}")

            _cd = _coef_dict_from_model(mode_model, coef_override)
            logger.warning(f"    coef_dict: {len(_cd)} params, "
                           f"c_ivt_auto={_cd.get('c_ivt_auto','MISSING')}")
        except Exception as _de:
            logger.warning(f"    Diagnostic failed: {_de}")

    return ls


def _logsums_numpy_fallback(mode_model, dt_mode, case_ids, zone_ids,
                             skims=None, verbose: bool = False,
                             coef_override: dict = None) -> np.ndarray:
    """
    Compute mode-choice logsums using fitted pf coefficients and numpy only.

    Reads OMX matrices DIRECTLY as numpy arrays (fast) rather than
    materialising the xarray OD dataset to a DataFrame (which is slow enough
    to silently time-out on HPC nodes with 108,900 OD pairs × 145 columns).

    Uses mode-specific travel time variables:
        DA / Other  : SOVTOLL_TIME__AM (auto IVT, minutes)
        Transit     : WLK_LOC_WLK_TOTIVT__AM (walk-to-transit IVT, minutes)
        Walk        : DISTWALK × 12 min/km   (5 km/h)
        Bike        : DISTBIKE × 4  min/km   (15 km/h)

    Returns np.ndarray shape (n_cases, n_zones).
    """
    xp = jnp if JAX_AVAILABLE else np
    n_cases = len(case_ids)
    n_zones = len(zone_ids)
    ls_array = np.zeros((n_cases, n_zones), dtype=np.float32)

    # ── 1. Extract fitted coefficients via robust helper ─────────────────
    # _coef_dict_from_model never calls .pf on loaded models - avoids
    # the '_mangled' AttributeError that caused all logsums to be zero.
    _cache = _coef_dict_from_model(mode_model, coef_override)
    if not _cache:
        logger.warning("_logsums_numpy_fallback: no coefficient source - "
                       "pass coef_override or use save_models() to inject cache")
        return ls_array

    def _pv(name, default=0.0):
        v = _cache.get(name)
        if v is None: return default
        try:
            fv = float(v)
            return fv if not np.isnan(fv) else default
        except Exception: return default

    # IVT / time coefficients
    c_ivt_auto    = _pv('c_ivt_auto',    _pv('c_ivt',       -1.20))
    c_ivt_transit = _pv('c_ivt_transit', _pv('c_ivt_auto',  -0.90))
    c_walk_time   = _pv('c_walk_time',   _pv('c_walk_dist', -1.50))
    c_bike_time   = _pv('c_bike_time',   _pv('c_bike_dist', -1.00))

    # ASC defaults: raised from -4.0/-4.5 (produced ~0% walk/bike) to values
    # that let the distance utility govern suppression at long distances.
    asc_sr        = _pv('ASC_SR',        -0.50)
    asc_walk      = _pv('ASC_Walk',      -1.00)  # was -4.0
    asc_bike      = _pv('ASC_Bike',      -1.50)  # was -4.5
    asc_transit   = _pv('ASC_Transit',   -0.50)  # was -1.5

    # ── 2. Build zone → matrix-row index from OMX mapping ───────────────
    # zone_ids are the destination alternatives; we also need origin OTAZ
    # per case from the tour sub-dataset.
    zone_to_idx: dict = {}    # zone_id → OMX row/col index (0-based)
    if skims is not None:
        try:
            mapping = dict(skims.mapping(skims.listMappings()[0]))
            # mapping is {zone_id: 0-based_index}
            zone_to_idx = {int(k): int(v) for k, v in mapping.items()}
        except Exception as e:
            logger.debug(f"_logsums_numpy_fallback: could not read OMX mapping: {e}")

    # ── 3. Read OMX matrices as numpy arrays once (fast path) ────────────
    def _read_mat(name: str, default_min: float = 30.0):
        """Read one OMX matrix as float32 array; return None on failure."""
        if skims is None:
            return None
        try:
            if name in skims.listMatrices():
                arr = np.array(skims[name], dtype=np.float32)
                # unit detection (same logic as _get_tt)
                med = float(np.median(arr[arr > 0])) if (arr > 0).any() else 0.0
                if med > 600:      arr = arr / 60.0   # seconds → minutes
                elif med > 200:    arr = arr / 10.0   # tenths → minutes
                return arr
        except Exception as e:
            logger.debug(f"_logsums_numpy_fallback: could not read {name}: {e}")
        return None

    mat_auto    = _read_mat('SOVTOLL_TIME__AM')
    _mat_trans  = _read_mat('WLK_LOC_WLK_TOTIVT__AM')
    if _mat_trans is None:
        _mat_auto_for_transit = _read_mat('SOVTOLL_TIME__AM')
        mat_transit = (_mat_auto_for_transit * 1.5
                       if _mat_auto_for_transit is not None else None)
    else:
        mat_transit = _mat_trans
    mat_walk    = _read_mat('DISTWALK')    # km → × 12 = minutes at 5 km/h
    mat_bike    = _read_mat('DISTBIKE')    # km → × 4  = minutes at 15 km/h

    # ── 4. Get OTAZ per case from the tour sub-dataset ────────────────────
    case_otaz = np.ones(n_cases, dtype=np.int32)
    try:
        tour_df = dt_mode['tour'].to_dataframe()   # small: only tour-level cols
        tour_df.index = tour_df.index.map(str)
        case_ids_str  = [str(c) for c in case_ids]
        if 'OTAZ' in tour_df.columns:
            for i, cid in enumerate(case_ids_str):
                if cid in tour_df.index:
                    case_otaz[i] = int(tour_df.at[cid, 'OTAZ'])
    except Exception as e:
        logger.debug(f"_logsums_numpy_fallback: tour read failed: {e} - using OTAZ=1")

    # ── 5. Vectorised logsum computation ─────────────────────────────────
    # For each case, look up its origin row index once, then gather the
    # destination columns for all zones in a single numpy slice.
    # Shape after gathering: (n_cases, n_zones) - one matrix operation.
    #
    # IMPORTANT: use -1 as sentinel for zones not in the OMX mapping.
    # Defaulting to 0 would silently read zone[0]'s row/column for every
    # unmatched zone, poisoning logsums for those OD pairs.
    _MISSING_IDX = -1
    dest_idx = np.array(
        [zone_to_idx.get(int(z), _MISSING_IDX) for z in zone_ids], dtype=np.int32)
    dest_valid = (dest_idx >= 0)   # boolean mask: True where zone is in OMX

    def _gather(mat, fallback_min: float):
        """Return (n_cases, n_zones) array; uses fallback_min where mat is None."""
        if mat is None or not zone_to_idx:
            return np.full((n_cases, n_zones), fallback_min, dtype=np.float32)
        out = np.full((n_cases, n_zones), fallback_min, dtype=np.float32)
        valid_dest = dest_idx[dest_valid]  # column indices for zones in OMX
        for i in range(n_cases):
            o_idx = zone_to_idx.get(int(case_otaz[i]), _MISSING_IDX)
            if o_idx < 0:
                # Origin zone not in OMX - leave entire row at fallback_min
                continue
            out[i, dest_valid] = mat[o_idx, valid_dest]
        # fill zeros (missing OD) with fallback
        out = np.where(out <= 0, fallback_min, out)
        return out

    auto_ivt    = _gather(mat_auto,    30.0)           # minutes
    transit_ivt = _gather(mat_transit, 45.0)           # minutes
    walk_dist   = _gather(mat_walk,    20.0)            # km
    bike_dist   = _gather(mat_bike,    20.0)            # km

    # Match MODE_UEC expressions exactly:
    #   auto/transit: time in HOURS (/ 60)
    #   walk:  log(od.DISTWALK * 12 + 1)  - DISTWALK in km, 12 min/km at 5 km/h
    #   bike:  log(od.DISTBIKE * 4 + 1)   - DISTBIKE in km, 4 min/km at 15 km/h
    # walk_dist / bike_dist here are raw OMX values already in km.
    # Clip AFTER the minutes → hours conversion: the (0.01, 5.0)/(0.01, 6.0)
    # bounds are hour-scale. Clipping the raw minutes first saturated every
    # OD above 5 (6) MINUTES to the ceiling, collapsing auto/transit
    # utilities to a constant and making the resulting logsums near-constant
    # across zones (std ~2e-4 pre-rescale, job 23880279) — the destination
    # models' c_logsum term was fitting amplified noise.
    auto_h = xp.clip(xp.asarray(auto_ivt) / 60.0, 0.01, 5.0)
    trans_h = xp.clip(xp.asarray(transit_ivt) / 60.0, 0.01, 6.0)
    walk_logmin = xp.log(xp.clip(xp.asarray(walk_dist) * 12.0, 0.0, 1500.0) + 1.0)
    bike_logmin = xp.log(xp.clip(xp.asarray(bike_dist) * 4.0, 0.0, 1500.0) + 1.0)

    # MNL utility per mode - shape (n_cases, n_zones)
    v_da      = c_ivt_auto  * auto_h
    v_sr      = c_ivt_auto  * auto_h     + asc_sr
    v_walk    = c_walk_time * walk_logmin + asc_walk
    v_bike    = c_bike_time * bike_logmin + asc_bike
    v_transit = c_ivt_transit * trans_h  + asc_transit

    # Stack to (n_cases, n_zones, n_modes) and compute log-sum-exp
    utils = xp.stack([v_da, v_sr, v_walk, v_bike, v_transit], axis=2)
    # subtract per-row max for numerical stability
    utils -= utils.max(axis=2, keepdims=True)
    ls_array = np.asarray(xp.log(xp.exp(utils).sum(axis=2)), dtype=np.float32)

    pct_nonzero = float((ls_array != 0).mean()) * 100
    logger.info(f"_logsums_numpy_fallback: {n_cases}×{n_zones} logsums computed "
                f"({pct_nonzero:.1f}% non-zero) via direct OMX reads")
    return ls_array


# ===========================================================================
# SEGMENT ESTIMATION  (ActivitySim groupby-segment pattern)
# ===========================================================================


def _ensure_cbd_and_transit_cols(landuse: pd.DataFrame) -> pd.DataFrame:
    """
    Attach CBD flag and transit walk access columns to landuse.

    is_cbd           : 1 if zone ORDER is in CBD_ZONES (inner-city, high transit share)
    transit_walk_acc : transit walk-accessibility score per zone
                       (from landuse.transit_acc if available, else 0)
    """
    lu = landuse.copy()
    zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]

    if 'is_cbd' not in lu.columns:
        if lu.index.name == zone_col:
            lu['is_cbd'] = lu.index.astype(int).isin(CBD_ZONES).astype(np.float32)
        else:
            lu['is_cbd'] = lu[zone_col].astype(int).isin(CBD_ZONES).astype(np.float32)

    if 'transit_walk_acc' not in lu.columns:
        if 'transit_acc' in lu.columns:
            lu['transit_walk_acc'] = pd.to_numeric(lu['transit_acc'], errors='coerce').fillna(0).astype(np.float32)
        else:
            lu['transit_walk_acc'] = np.float32(0.0)

    return lu


def _attach_size_term(landuse: pd.DataFrame, segment: str) -> pd.DataFrame:
    """
    Compute a single 'size_term' column for this segment and attach to landuse.

    Returns landuse as a plain DataFrame (ORDER as column) with 'size_term' added.
    The column name has no leading underscore so sharrow resolves 'emp.size_term'
    correctly (leading underscore causes the 'emp.._size_term' double-dot error).
    """
    lu = _ensure_cbd_and_transit_cols(landuse)

    # work on a flat copy (ORDER as a column, not index)
    lu = lu.copy()
    if lu.index.name == 'ORDER':
        lu = lu.reset_index()

    # ── Ensure all required DEST_UEC landuse columns exist ────────────────
    # If any column is missing the DataTree sharrow compiler will fail to
    # compile the utility expression and return an empty pf (0 parameters).
    # We create zero-filled columns as placeholders so compilation succeeds;
    # the model will learn near-zero coefficients for truly absent data.
    _REQUIRED_COLS = {
        # edu attractor columns
        'Primary': 0.0, 'Secondary': 0.0, 'Education': 0.0, 'HSENROLL': 0.0,
        'university': 0.0,    # OSM POI count; merged in by data_connect.get_land()
        'college': 0.0,       # OSM POI count
        # work attractor columns
        'TOTEMP': 0.0, 'FPSEMPN': 0.0, 'HEREMPN': 0.0, 'RETEMPN': 0.0,
        # 'commercial' removed: 89M m² outlier at zone 158 corrupted work and othmaint size terms
        # nonmand/eat attractor columns
        'residential': 0.0, 'Entertainment': 0.0, 'retail': 0.0,
        'Healthcare': 0.0,    # lu-native healthcare aggregate
        # OSM columns merged in by get_land()
        'kindergarten': 0.0, 'sports_centre': 0.0, 'fitness_centre': 0.0,
        'clinic': 0.0, 'pharmacy': 0.0, 'bank': 0.0,
        'restaurant': 0.0, 'cafe': 0.0, 'pub': 0.0,
        # accessibility columns
        'auto_work_acc': 0.0, 'auto_shop_acc': 0.0,
    }
    # Also create common aliases: school enrollment may be named differently
    _ALIASES = {
        'Primary':   ['primary_school', 'PSENROLL', 'primary_enrolment', 'enrol_primary'],
        'Secondary': ['secondary_school', 'HSENROLL_sec', 'secondary_enrolment'],
        'Education': ['tertiary', 'UNIENROLL', 'university_enrolment', 'enrol_tertiary'],
    }
    for canon, alts in _ALIASES.items():
        if canon not in lu.columns:
            for alt in alts:
                if alt in lu.columns:
                    lu[canon] = lu[alt]
                    logger.debug(f"_attach_size_term: aliased {alt} → {canon}")
                    break
    for col, default in _REQUIRED_COLS.items():
        if col not in lu.columns:
            lu[col] = default

    seg_purposes = SEGMENT_PURPOSES.get(segment, list(SIZE_TERM_COEFFICIENTS.keys()))
    st_df = compute_size_terms(lu, purposes=seg_purposes)

    # collapse to one column: segment canonical purpose or mean
    if len(st_df.columns) == 1:
        st_col = st_df.iloc[:, 0]
    else:
        st_col = st_df.mean(axis=1)

    # st_df is indexed by zone (ORDER) - align back to lu rows
    zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]
    lu = lu.set_index(zone_col)
    lu['size_term'] = st_col.reindex(lu.index).fillna(0.0).values
    lu = lu.reset_index()   # ORDER back as column for _lu_idx
    return lu


# ===========================================================================
# ACDC - Agglomerating and Competing Discrete Choice
# ===========================================================================
#
# Implements the spatial terms from:
#   Bernadin, Frei, Axhausen (2011) "Agglomeration, competition and
#   Destination Choice" - Transportation Research Part B.
#
# Two zone-level corrections are added to the standard MNL destination utility:
#
#   Agglomeration  aggl[j] = log(Σ_{k≠j} size[k] × exp(-λ_aggl × t[j,k]) + ε)
#       Zone j is more attractive if it is surrounded by other attractive zones
#       (complementary services, multi-purpose trips, buzz effects).
#       β_aggl expected positive.
#
#   Competition  comp[j] = log(Σ_{k≠j} exp(-λ_comp × t[j,k]) + ε)
#       Zone j faces more substitution pressure (is a less distinctive choice)
#       the more OTHER destinations are reachable nearby - an unweighted
#       density/count of nearby alternatives, decay-weighted by travel time.
#       β_comp expected negative.
#
# Implementation notes
# --------------------
# aggl uses the segment's size_term as the attraction weight (so it is
# purpose-specific: work uses TOTEMP, nonmand uses retail/entertainment).
# comp is deliberately UNWEIGHTED (no size_arr) so it is not just a rescaled
# copy of aggl: an earlier version weighted comp by size_arr too, using the
# same time matrix and lambda as aggl with only the diagonal-inclusion
# differing - that made acdc_aggl and acdc_comp ~0.9998 correlated (see
# stage5 validation notes, 2026-07), so only one of the two was ever really
# identified. Dropping the size weight from comp (matching the more recent
# competing-destinations formulation in Z:/IMOVE/gonzalo/od_choice_model.py's
# compute_acdc_proxy_terms) makes it a genuinely distinct spatial-density
# measure instead of a near-duplicate attraction measure.
#
# λ values (decay rate per minute of auto IVT) are fixed constants rather than
# estimated parameters - this avoids an identification issue with β and keeps
# Larch estimation well-behaved.  The values (0.05/min) give a half-life of
# ~14 minutes, meaning destinations ~14 min away exert half the spatial
# influence of adjacent zones.  This matches empirical ACDC estimates for
# SEQ-scale models (Burdett 2024, internal).
#
# Both computations are fully vectorised as matrix multiplications:
#   aggl = log(exp_mat_aggl @ size_arr + ε)    shape (N,)
#   comp = log(exp_mat_comp @ size_arr + ε)    shape (N,)
# where N = number of zones and exp_mat = exp(-λ × time_matrix).
# ===========================================================================

# ACDC decay rates - fixed, not estimated (avoids β/λ identification issues)
LAMBDA_AGGL = 0.05   # agglomeration decay: 1/2 effect at ~14 min auto IVT
LAMBDA_COMP = 0.05   # competition decay:   1/2 effect at ~14 min auto IVT


def compute_acdc_terms(
    landuse: pd.DataFrame,
    skim_df: pd.DataFrame,
    size_arr: np.ndarray,
    zone_ids: np.ndarray,
    lambda_aggl: float = LAMBDA_AGGL,
    lambda_comp: float = LAMBDA_COMP,
) -> tuple:
    """
    Compute ACDC agglomeration and competition terms for all zones.

    Vectorised implementation using matrix multiply - no Python loops.

    Parameters
    ----------
    landuse    : zone landuse table
    skim_df    : flat OTAZ/DTAZ skim DataFrame (from _skim_flat)
    size_arr   : (N,) array of zone attraction weights (segment size_term,
                 already exponentiated back from log-form)
    zone_ids   : (N,) integer array of zone IDs in same order as size_arr
    lambda_aggl: decay rate for agglomeration (per minute auto IVT)
    lambda_comp: decay rate for competition (per minute auto IVT)

    Returns
    -------
    aggl_series : pd.Series indexed by zone_id  - agglomeration score per zone
    comp_series : pd.Series indexed by zone_id  - competition score per origin
    """
    N = len(zone_ids)
    aggl_series = pd.Series(0.0, index=zone_ids, name='acdc_aggl')
    comp_series = pd.Series(0.0, index=zone_ids, name='acdc_comp')

    if skim_df.empty or 'SOVTOLL_TIME__AM' not in skim_df.columns:
        logger.warning("compute_acdc_terms: no SOVTOLL_TIME__AM in skim_df - "
                       "ACDC terms will be zero (model still runs)")
    return aggl_series, comp_series


# ── CBD zone list (inner Brisbane zones with high parking cost / transit share) ─
CBD_ZONES = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # Brisbane CBD + inner ring
    40, 41, 42, 43, 44, 45,  # South Brisbane / West End / Fortitude Valley
}


def compute_themed_acdc_terms(
    landuse: pd.DataFrame,
    skim_df: pd.DataFrame,
    zone_ids: np.ndarray,
    lambda_aggl: float = LAMBDA_AGGL,
) -> dict:
    """
    Compute theme-restricted ACDC agglomeration variants.

    Each variant uses the same decay kernel as the generic aggl but with
    size weights restricted to a specific attractor column:

        aggl_retail[j]  = log(Σ_{k≠j} retail_emp[k]  × exp(-λ × time[j,k]) + ε)
        aggl_office[j]  = log(Σ_{k≠j} office_emp[k]  × exp(-λ × time[j,k]) + ε)
        aggl_edu[j]     = log(Σ_{k≠j} edu_seats[k]   × exp(-λ × time[j,k]) + ε)
        aggl_leisure[j] = log(Σ_{k≠j} leisure_poi[k] × exp(-λ × time[j,k]) + ε)

    Returns a dict mapping column name ('acdc_aggl_retail', etc.) to pd.Series
    indexed by zone_id.
    """
    # Per-type ACDC agglomeration (the imove/gonzalo decomposition), mapped to
    # the land-use columns that actually exist in zone_landuse_357/507.csv. Each
    # theme is log(Σ_{k≠j} Σ_col size_col[k] · exp(-λ·time[j,k])), standardised.
    # The old leisure theme referenced cafe/pub/restaurant/sports_centre columns
    # that are NOT in this landuse (silently zero); replaced with real columns.
    # The GA/SA search + ga_constraints pick among these (they are correlated),
    # letting the model say WHICH destination character drives each segment.
    themes = {
        'acdc_aggl_retail':  ['RETEMPN', 'retail', 'Shopping'],
        'acdc_aggl_office':  ['FPSEMPN', 'commercial', 'institutional'],
        'acdc_aggl_edu':     ['Education', 'Primary', 'Secondary', 'HSENROLL',
                              'university', 'college', 'childcare', 'education'],
        'acdc_aggl_leisure': ['Entertainment', 'sport'],
        'acdc_aggl_parks':   ['park', 'recreation_ground', 'village_green', 'trees'],
        'acdc_aggl_health':  ['Healthcare', 'HEREMPN'],
        'acdc_aggl_food':    ['EATOUT'],
        'acdc_aggl_civic':   ['civic', 'civic_admin', 'community',
                              'Public_Services', 'government'],
    }
    result = {}
    N = len(zone_ids)

    if skim_df.empty or 'SOVTOLL_TIME__AM' not in skim_df.columns:
        for key in themes:
            result[key] = pd.Series(0.0, index=zone_ids, name=key)
        return result

    try:
        sk_sub = (skim_df[['OTAZ', 'DTAZ', 'SOVTOLL_TIME__AM']]
                  .drop_duplicates(['OTAZ', 'DTAZ']))
        zone_to_idx = {int(z): i for i, z in enumerate(zone_ids)}
        time_mat = np.full((N, N), 60.0, dtype=np.float32)
        np.fill_diagonal(time_mat, 0.0)

        for row in sk_sub.itertuples(index=False):
            i = zone_to_idx.get(int(row.OTAZ))
            j = zone_to_idx.get(int(row.DTAZ))
            if i is not None and j is not None:
                val = float(row.SOVTOLL_TIME__AM)
                if val > 600:
                    val /= 60.0
                elif val > 200:
                    val /= 10.0
                time_mat[i, j] = max(val, 0.1)

        exp_aggl = np.exp(-lambda_aggl * time_mat).astype(np.float64)
        np.fill_diagonal(exp_aggl, 0.0)

        lu_indexed = landuse.set_index('ORDER' if 'ORDER' in landuse.columns
                                        else landuse.columns[0])

        for out_col, src_cols in themes.items():
            size_arr = np.zeros(N, dtype=np.float64)
            for col in src_cols:
                if col in lu_indexed.columns:
                    vals = pd.to_numeric(lu_indexed[col], errors='coerce').fillna(0)
                    size_arr += vals.reindex(zone_ids).fillna(0).values
            size_arr = np.clip(size_arr, 0, None)
            aggl_raw = exp_aggl @ size_arr
            aggl_vals = np.log(aggl_raw + 1e-8)
            _s = aggl_vals.std()
            if _s > 1e-8:
                aggl_vals = (aggl_vals - aggl_vals.mean()) / _s
            result[out_col] = pd.Series(aggl_vals, index=zone_ids, name=out_col)

        logger.info(
            f"compute_themed_acdc_terms: {N} zones | "
            + " | ".join(f"{k} μ={result[k].mean():.2f} σ={result[k].std():.3f}"
                        for k in themes)
        )

    except Exception as e:
        logger.warning(f"compute_themed_acdc_terms failed: {e}")
        for key in themes:
            result[key] = pd.Series(0.0, index=zone_ids, name=key)

    return result


def compute_home_pull(
    tours: pd.DataFrame,
    persons: pd.DataFrame,
    zone_ids: np.ndarray,
    skim_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute home_pull_j per (person, destination) pair.

    home_pull_j = -log(1 + dist(home_zone, j))

    Two variants (both computed as columns on the tours/persons table):
      home_pull     : scaled by departure-time weight
                       (late-afternoon=1.0, midday=0.6, morning=0.3)
      home_pull_late: scaled by remaining-day-budget fraction from stages 2-3
                       (time_remaining_min / total_day_min)

    Returns a DataFrame with columns ['ptour_id', 'home_pull', 'home_pull_late']
    that gets merged into the base/cases dataset in _dest_tree.
    """
    hp_rows = []

    home_zone_col = next(
        (c for c in ['OTAZ', 'HOME_TAZ', 'home_taz', 'home_zone']
         if c in persons.columns), None)
    if home_zone_col is None:
        logger.warning("compute_home_pull: no home zone column on persons")
        return pd.DataFrame({
            'ptour_id': tours.get('ptour_id', tours.index).astype(str),
            'home_pull': 0.0, 'home_pull_late': 0.0,
        })

    home_zones = persons.set_index('person_id')[home_zone_col].to_dict()
    for z in zone_ids:
        home_zones[str(z)] = int(z)

    skim_lookup = {}
    if skim_df is not None and not skim_df.empty and 'DIST' in skim_df.columns:
        for row in skim_df[['OTAZ', 'DTAZ', 'DIST']].itertuples(index=False):
            skim_lookup[(int(row.OTAZ), int(row.DTAZ))] = float(row.DIST)

    tour_df = tours.copy()
    if 'ptour_id' not in tour_df.columns:
        tour_df = tour_df.reset_index()
    tour_df['ptour_id'] = tour_df['ptour_id'].astype(str)

    for _, row in tour_df.iterrows():
        pid = str(row.get('person_id', row.get('PERSID', '')))
        hz = int(home_zones.get(pid, 1))
        dep_min = float(row.get('STARTIME', row.get('start_time', 480)))

        time_weight = 0.3
        if dep_min >= 900:
            time_weight = 0.6
        if dep_min >= 1080:
            time_weight = 1.0

        day_remaining = float(row.get('time_remaining',
                               row.get('remaining_budget',
                               row.get('B', 0))))
        day_total = float(row.get('day_total',
                           row.get('total_budget',
                           row.get('B_total', max(day_remaining, 1.0)))))

        late_weight = max(0.0, min(1.0, day_remaining / max(day_total, 1.0)))

        hp_rows.append({
            'ptour_id': row['ptour_id'],
            'home_zone': hz,
            'departure_weight': time_weight,
            'late_weight': late_weight,
        })

    hp_df = pd.DataFrame(hp_rows)
    hp_df['home_pull'] = 0.0
    hp_df['home_pull_late'] = 0.0

    return hp_df[['ptour_id', 'home_zone', 'departure_weight',
                   'late_weight', 'home_pull', 'home_pull_late']]


def _attach_home_pull_columns(
    ca_df: pd.DataFrame,
    hp_df: pd.DataFrame,
    zone_ids: np.ndarray,
    skim_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pre-compute home_pull and home_pull_late columns on the flat ca_df.

    For each (case, alternative) row:
      home_pull_j = -log(1 + dist(home_zone, j))
      home_pull     = home_pull_j * departure_weight
      home_pull_late = home_pull_j * late_weight
    """
    if hp_df.empty or 'home_zone' not in hp_df.columns:
        ca_df['home_pull'] = np.float32(0.0)
        ca_df['home_pull_late'] = np.float32(0.0)
        return ca_df

    skim_lookup = {}
    if skim_df is not None and not skim_df.empty and 'DIST' in skim_df.columns:
        for row in skim_df[['OTAZ', 'DTAZ', 'DIST']].itertuples(index=False):
            skim_lookup[(int(row.OTAZ), int(row.DTAZ))] = float(row.DIST)

    hp_indexed = hp_df.set_index('ptour_id')

    def _hp_val(ptour, altid):
        row = hp_indexed.loc[str(ptour)]
        hz = int(row['home_zone'])
        dz = int(altid)
        if hz == dz:
            dist = 0.01
        else:
            dist = skim_lookup.get((hz, dz), skim_lookup.get((dz, hz), 60.0))
        base = -np.log1p(max(dist, 0.01))
        return base * float(row['departure_weight']), base * float(row['late_weight'])

    hp_vals = []
    hp_late_vals = []
    for _, ca_row in ca_df.iterrows():
        try:
            a, b = _hp_val(ca_row['ptour_id'], ca_row['_altid_'])
            hp_vals.append(a)
            hp_late_vals.append(b)
        except Exception:
            hp_vals.append(0.0)
            hp_late_vals.append(0.0)

    ca_df['home_pull'] = np.array(hp_vals, dtype=np.float32)
    ca_df['home_pull_late'] = np.array(hp_late_vals, dtype=np.float32)

    return ca_df

    try:
        # ── Build (N×N) auto travel-time matrix ────────────────────────────
        # Pivot flat OTAZ/DTAZ table into a square matrix aligned to zone_ids.
        sk_sub = (skim_df[['OTAZ', 'DTAZ', 'SOVTOLL_TIME__AM']]
                  .drop_duplicates(['OTAZ', 'DTAZ']))
        zone_to_idx = {int(z): i for i, z in enumerate(zone_ids)}

        time_mat = np.full((N, N), 60.0, dtype=np.float32)  # default 60 min
        np.fill_diagonal(time_mat, 0.0)                      # intra-zonal = 0

        for row in sk_sub.itertuples(index=False):
            i = zone_to_idx.get(int(row.OTAZ))
            j = zone_to_idx.get(int(row.DTAZ))
            if i is not None and j is not None:
                val = float(row.SOVTOLL_TIME__AM)
                # unit detection (same as _get_tt)
                if val > 600:   val /= 60.0
                elif val > 200: val /= 10.0
                time_mat[i, j] = max(val, 0.1)

        # ── Agglomeration: exp(-λ × time[j,k]) for each destination j ──────
        # aggl[j] = log(Σ_{k≠j} size[k] × exp(-λ × time[j,k]) + ε)
        # = log((exp_mat_aggl @ size_arr) - size[j]*exp(-λ*0) + ε)
        # Since diagonal is 0: exp(-λ*0)=1 → self-contribution = size[j].
        # We zero the diagonal explicitly to exclude k==j.
        exp_aggl          = np.exp(-lambda_aggl * time_mat).astype(np.float64)
        np.fill_diagonal(exp_aggl, 0.0)           # exclude self (k ≠ j)
        aggl_raw          = exp_aggl @ size_arr   # (N,) - weighted sum
        aggl_vals         = np.log(aggl_raw + 1e-8)

        # ── Competition: exp(-λ × time[j,k]) for each destination j ─────────
        # comp[j] = log(Σ_{k≠j} exp(-λ × time[j,k]) + ε)   (UNWEIGHTED by size -
        # see module docstring for why: weighting by size_arr here made this
        # numerically ~identical to aggl.)
        exp_comp          = np.exp(-lambda_comp * time_mat).astype(np.float64)
        np.fill_diagonal(exp_comp, 0.0)           # exclude self (k ≠ j), same as aggl
        comp_raw          = exp_comp.sum(axis=1)  # (N,) - unweighted nearby-zone density
        comp_vals         = np.log(comp_raw + 1e-8)

        # ── Z-score normalise both terms so coefficients are scale-free ──────
        # Without normalisation the raw log values span ~15 units; large
        # employment clusters (CBD, university zones) dominate destination
        # utility and cause zone concentration (e.g. zones 203, 229, 266).
        # After z-scoring std≈1 so c_aggl ≈ 0.2 means a 1-std advantage
        # gives exp(0.2)≈1.2× attraction - interpretable and bounded.
        _aggl_std = aggl_vals.std()
        _comp_std = comp_vals.std()
        if _aggl_std > 1e-8:
            aggl_vals = (aggl_vals - aggl_vals.mean()) / _aggl_std
        if _comp_std > 1e-8:
            comp_vals = (comp_vals - comp_vals.mean()) / _comp_std

        aggl_series = pd.Series(aggl_vals, index=zone_ids, name='acdc_aggl')
        comp_series = pd.Series(comp_vals, index=zone_ids, name='acdc_comp')

        logger.info(
            f"compute_acdc_terms: {N} zones | "
            f"aggl mean={aggl_vals.mean():.2f} std={aggl_vals.std():.2f} | "
            f"comp mean={comp_vals.mean():.2f} std={comp_vals.std():.2f}"
        )

    except Exception as e:
        logger.warning(f"compute_acdc_terms failed: {e} - ACDC terms will be zero")

    return aggl_series, comp_series


def _attach_acdc_terms(
    landuse: pd.DataFrame,
    tours: pd.DataFrame,
    skim_df: pd.DataFrame,
    segment: str,
) -> tuple:
    """
    Compute ACDC spatial terms and attach them where the DataTree can find them.

    Returns
    -------
    lu_out  : landuse with 'acdc_aggl' and 'acdc_comp' columns added
              (→ emp.acdc_aggl and emp.acdc_comp in DataTree)
    t_out   : tours with a dummy 'acdc_comp' = 0 column (unused by model)

    Design
    ------
    • acdc_aggl is zone-level (indexed by destination) → lives on landuse / emp.
    • acdc_comp is now also zone-level (indexed by destination) → lives on emp.
      Competition represents how many alternatives are accessible FROM each
      destination zone - zones with more reachable alternatives face stronger
      substitution pressure and are therefore less likely to be chosen.
      Placing it on emp ensures it always varies across alternatives regardless
      of how many distinct OTAZs are in the current batch, so Sharrow never
      sees a structural DataTree change between estimation and prediction.

    The size_term used as the attraction weight is the segment-specific one
    already computed by _attach_size_term - so work agglomeration uses
    employment, shopping agglomeration uses retail employment, etc.
    """
    lu   = landuse.copy()
    tours_out = tours.copy()

    # ── Resolve zone column ───────────────────────────────────────────────
    zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]
    if lu.index.name == zone_col:
        lu = lu.reset_index()

    zone_ids = lu[zone_col].astype(int).values

    # ── Get segment size_term as attraction weights ───────────────────────
    # size_term is log-transformed; exponentiate back for the spatial kernel.
    if 'size_term' not in lu.columns:
        logger.warning("_attach_acdc_terms: no size_term column - run _attach_size_term first")
        lu['acdc_aggl'] = 0.0
        tours_out['acdc_comp'] = 0.0
        return lu, tours_out

    size_log = pd.to_numeric(lu['size_term'], errors='coerce').fillna(0.0).values
    # Clip before exp to avoid overflow (size_term can be large log values)
    size_arr = np.exp(np.clip(size_log, -30, 30))

    # ── Compute ACDC terms ────────────────────────────────────────────────
    aggl_series, comp_series = compute_acdc_terms(
        lu, skim_df, size_arr, zone_ids
    )

    # ── Attach aggl to landuse (emp dataset) ──────────────────────────────
    lu = lu.set_index(zone_col)
    lu['acdc_aggl'] = aggl_series.reindex(lu.index).fillna(0.0).values

    # ── Attach comp to landuse (emp dataset) ──────────────────────────────
    # Competition is now attached at the DESTINATION (alternative) level rather
    # than the tour (case) level.  This means acdc_comp is always indexed by
    # zone, and the DataTree column always varies across alternatives regardless
    # of how many distinct OTAZs are in the current batch.
    #
    # Semantic: comp[j] = competition accessible from zone j - an unweighted,
    # decay-weighted count of OTHER reachable zones from j (not size-weighted -
    # see compute_acdc_terms docstring).  In destination choice, a zone with
    # many reachable alternatives is less likely to be chosen (substitution
    # effect), which is the intended interpretation.
    #
    # Previously this was on tour.acdc_comp (person/case level), which
    # collapsed to a scalar when all persons in a batch share one OTAZ -
    # causing Sharrow to see a different DataTree structure vs estimation
    # and recompile on every zone/batch.  Moving it to emp.acdc_comp fixes
    # the structural mismatch permanently.
    lu['acdc_comp'] = comp_series.reindex(lu.index).fillna(0.0).values
    lu = lu.reset_index()

    # ── Themed ACDC variants ──────────────────────────────────────────────
    try:
        themed = compute_themed_acdc_terms(lu, skim_df, zone_ids, LAMBDA_AGGL)
        for col_name, themed_series in themed.items():
            lu = lu.set_index(zone_col)
            lu[col_name] = themed_series.reindex(lu.index).fillna(0.0).values
            lu = lu.reset_index()
    except Exception as _themed_e:
        logger.warning(f"_attach_acdc_terms [{segment}]: themed ACDC failed ({_themed_e})")
        for _tc in ['acdc_aggl_retail', 'acdc_aggl_office', 'acdc_aggl_edu',
                    'acdc_aggl_leisure', 'acdc_aggl_parks', 'acdc_aggl_health',
                    'acdc_aggl_food', 'acdc_aggl_civic']:
            if _tc not in lu.columns:
                lu[_tc] = 0.0

    # tours_out still gets a dummy acdc_comp=0 column so any code that
    # reads tour.acdc_comp doesn't KeyError - the real value comes from emp.
    tours_out['acdc_comp'] = 0.0

    logger.info(
        f"_attach_acdc_terms [{segment}]: "
        f"acdc_aggl range [{lu['acdc_aggl'].min():.2f}, {lu['acdc_aggl'].max():.2f}] | "
        f"acdc_comp range [{lu['acdc_comp'].min():.2f}, {lu['acdc_comp'].max():.2f}]"
    )
    return lu, tours_out


def _clear_sharrow_cache():
    """
    Clear sharrow's compiled numba flow cache.

    On PBS HPC systems the cache lands in /tmp/pbs.JOBID.HOST/tmpXXXX/
    not directly in /tmp. We need to search recursively.
    Also checks $HOME/.cache/sharrow which is used on some systems.
    """
    import tempfile, shutil, glob, os
    cleared = 0

    # Direct /tmp patterns (local workstation)
    for pattern in ['/tmp/pipeline_*', '/tmp/flow_*']:
        for d in glob.glob(pattern):
            try:
                shutil.rmtree(d)
                cleared += 1
            except Exception:
                pass

    # PBS HPC pattern: /tmp/pbs.JOBID.HOST/tmpXXXX/pipeline_* or flow_*
    for pbs_dir in glob.glob('/tmp/pbs.*'):
        for sub in glob.glob(os.path.join(pbs_dir, 'tmp*')):
            for pattern in ['pipeline_*', 'flow_*']:
                for d in glob.glob(os.path.join(sub, pattern)):
                    try:
                        shutil.rmtree(d)
                        cleared += 1
                    except Exception:
                        pass

    # Home cache (some sharrow versions)
    home_cache = os.path.expanduser('~/.cache/sharrow')
    if os.path.exists(home_cache):
        try:
            shutil.rmtree(home_cache)
            os.makedirs(home_cache, exist_ok=True)
            cleared += 1
        except Exception:
            pass

    if cleared:
        logger.info(f"Cleared {cleared} sharrow cache entries")
    else:
        logger.debug("No sharrow cache entries found to clear")


def _estimate_mixed_logit(seg: pd.DataFrame, segment: str,
                          n_draws: int = 200, seed: int = 42) -> dict:
    """
    Estimate a mixed logit mode choice model using Larch's native mixtures API.

    Follows Professor Ken Train's approach from the mixed-logit notebook:
        1. Build a standard NL model (already done - this refits a fresh one)
        2. Add Normal mixtures for taste-heterogeneous parameters
        3. Set n_draws and seed, then maximize_loglike

    Random coefficients (heterogeneous across travellers):
        c_ivt        ~ Normal(mean, sd)  - IVT sensitivity varies by person
        c_walk_dist  ~ Normal(mean, sd)  - walk distance sensitivity varies
        c_bike_dist  ~ Normal(mean, sd)  - bike distance sensitivity varies

    Fixed coefficients (homogeneous):
        ASCs, availability flags - all ASCs fixed (identified)
        c_veh_avail, c_peak - structural variables, kept fixed

    The groupid is set to person_id so preferences vary across PERSONS not
    across observations (panel correction - see notebook §"Considering Panel Data").

    Parameters
    ----------
    seg      : filtered segment DataFrame (output of _filter_seg)
    segment  : 'work', 'edu', or 'nonmand'
    n_draws  : Halton draws (200 default - increase to 500+ for final results)
    seed     : random seed for Halton sequence

    Returns
    -------
    dict with keys: coef, std_err, tstat, ll, rho2, randvars, n_obs
    or None on failure.
    """
    try:
        if len(seg) > MAX_MXL_OBS:
            logger.info(
                f"[{segment}] Mixed logit: {len(seg)} observations exceeds "
                f"MAX_MXL_OBS={MAX_MXL_OBS} - subsampling to {MAX_MXL_OBS} for estimation"
            )
            seg = seg.sample(n=MAX_MXL_OBS, random_state=seed)

        # ── rebuild a fresh mode tree for this segment ────────────────
        # We can't reuse dt_m from the outer function scope here, so we
        # use the segment DataFrame directly - seg already has ptour_id index
        # The standard NL mode model is already estimated; here we build a
        # parallel MXL on the same data using Larch's mixtures API.

        # Import the datatree and landuse from outer scope isn't possible here;
        # instead we build a simplified MNL ca model from the segment's flat data.
        # This is sufficient for the MXL - we don't need the full DataTree.

        # Build idca dataset: one row per (tour × mode)
        # ── Pull per-observation skim values (merged by _run_segment) ────
        # SOVTOLL_TIME__AM: auto IVT in minutes (may have been merged)
        # DISTWALK / DISTBIKE: km - converted to time at 5 / 15 km/h
        # WLK_LOC_WLK_TOTIVT__AM: transit IVT minutes
        # walk_avail / bike_avail: distance-based availability flags
        alt_ids   = list(MODE_MAP.values())   # [1,2,3,4,5]
        rows = []

        # median IVT for this segment - used as denominator to scale to
        # a ~[-2, 0] range for numerical stability
        ivt_vals = pd.to_numeric(seg.get('SOVTOLL_TIME__AM', pd.Series(30.0,
                                  index=seg.index)), errors='coerce').fillna(30.0)
        ivt_median = max(float(ivt_vals.median()), 1.0)

        # ── GSE gradient score g_n (Gradient Score Enhanced mixed logit) ──────
        # SearchLibrium's MixedLogitGSE fits β_nk = μ_k + γ_k·g_nk + σ_k·η_nk,
        # where g_nk is a per-person exogenous "gradient score" (origin
        # accessibility / land-use context). In Larch this decomposes exactly:
        #   μ_k·x_k        → the ordinary term (already in utility_ca)
        #   γ_k·(g_n·x_k)  → a SYSTEMATIC interaction column (added below)
        #   σ_k·(η_nk·x_k) → the Normal mixture (added in Step 2, mxl.mixtures)
        # so enabling GSE + the existing mixtures reproduces the full GSE model.
        # g_n here = standardised origin accessibility (the mode-choice analogue
        # of the imove/gonzalo per-type ACDC-at-origin gradient score).
        _gse_enabled = os.environ.get('STAGE5_ENABLE_GSE', '0') == '1'
        _gse_map = {}
        if _gse_enabled:
            _gcands = [c for c in ('auto_work_acc', 'walk_acc', 'transit_acc',
                                   'auto_shop_acc', 'auto_nonmand_acc', 'incp')
                       if c in seg.columns]
            if _gcands:
                _gser = pd.to_numeric(seg[_gcands[0]], errors='coerce').fillna(0.0)
                _gmu, _gsd = float(_gser.mean()), float(_gser.std())
                _gz = (_gser - _gmu) / (_gsd + 1e-8)   # standardised, O(1)
                _gse_map = dict(zip(seg['ptour_id'].astype(str), _gz))
                logger.info(f"[{segment}] GSE enabled: gradient score = "
                            f"standardised '{_gcands[0]}' over {len(_gse_map)} tours")
            else:
                _gse_enabled = False
                logger.warning(f"[{segment}] GSE requested but no origin "
                               "accessibility/income column present — disabled")

        for _, tour in seg.iterrows():
            pid     = str(tour['ptour_id'])
            chosen  = int(tour.get('tour_mode', 1))

            # auto IVT - use actual skim value if present, else 30 min
            auto_ivt_min  = float(tour.get('SOVTOLL_TIME__AM', 30.0)) or 30.0
            # walk/bike distances → time in minutes
            walk_dist_km  = float(tour.get('DISTWALK',    20.0)) or 20.0
            bike_dist_km  = float(tour.get('DISTBIKE',    20.0)) or 20.0
            walk_time_min = walk_dist_km  * 12.0    # 5 km/h → 12 min/km
            bike_time_min = bike_dist_km  * 4.0     # 15 km/h → 4 min/km
            # transit IVT - use walk-to-transit time if present
            transit_ivt   = float(tour.get('WLK_LOC_WLK_TOTIVT__AM',
                                           auto_ivt_min * 1.5)) or auto_ivt_min * 1.5

            walk_ok = int(tour.get('walk_avail', int(walk_dist_km < 10.0)))
            bike_ok = int(tour.get('bike_avail', int(bike_dist_km < 25.0)))

            # ── person-level attributes (idco → encoded as alt-specific zeros) ─
            vehrd   = float(tour.get('vehrd', 1.0))
            licence = float(tour.get('CARLICENCE', 1.0))
            noveh   = float(vehrd == 0)
            ftwork  = float(int(tour.get('WORKSTAT', 0)) == 1)
            ptwork  = float(int(tour.get('WORKSTAT', 0)) == 2)
            age5p   = float(tour.get('age5p', 3.0))
            young   = float(age5p <= 5)
            nprd    = float(tour.get('nprd', 2.0))
            bikes   = float(tour.get('bikes', 0.0))
            female  = float(int(tour.get('sexp', 0)) == 2)
            incp    = float(tour.get('incp', 50.0))

            # ── Industry of employment (ANZSIC 1-digit, 'indp') ──────────────
            # Groups that strongly predict DA captivity vs transit amenability.
            # 'indp' int 1-20; if absent default to 20 (not employed/other).
            _indp_raw = int(tour.get('indp', 20) or 20)
            _ind_grp  = INDUSTRY_MODE_GROUP.get(_indp_raw, 'other')
            ind_industrial = float(_ind_grp in ('industrial', 'agri'))
            ind_office     = float(_ind_grp in ('office', 'professional'))
            ind_public     = float(_ind_grp == 'public')
            ind_health     = float(_ind_grp == 'health')
            ind_retail     = float(_ind_grp in ('retail', 'hospitality'))

            # ── Occupation (ANZSCO 1-digit, 'occp') ──────────────────────────
            # 1=Managers, 2=Professionals → office locations → mixed modal
            # 5=Clerical/Admin → CBD offices → more transit-amenable
            # 7=Machinery Operators, 8=Labourers → site work → DA captive
            _occp_raw = int(tour.get('occp', 0) or 0)
            occ_mgr_prof = float(_occp_raw in (1, 2))
            occ_labourer = float(_occp_raw in (7, 8))
            occ_clerical = float(_occp_raw == 5)

            # ── Studying status (1=not studying, 2=ft, 3=pt, 4=other) ────────
            _stup_raw = int(tour.get('stup', 1) or 1)
            is_ft_student = float(_stup_raw == 2)
            is_pt_student = float(_stup_raw in (3, 4))

            for alt in alt_ids:
                # Match MODE_UEC expressions exactly:
                #   auto/transit: SOVTOLL_TIME__AM / 60 → hours
                #   walk:         log(DISTWALK * 12 + 1) → log-minutes at 5 km/h
                #   bike:         log(DISTBIKE * 4  + 1) → log-minutes at 15 km/h
                if alt in (1, 2):        # DA, Other - auto hours
                    ivt_h       = auto_ivt_min  / 60.0
                    walk_logmin = 0.0
                    bike_logmin = 0.0
                elif alt == 5:           # Transit - hours
                    ivt_h       = transit_ivt   / 60.0
                    walk_logmin = 0.0
                    bike_logmin = 0.0
                elif alt == 3:           # Walk - log(minutes + 1)
                    ivt_h       = 0.0
                    walk_logmin = float(np.log(walk_time_min + 1.0))
                    bike_logmin = 0.0
                else:                    # Bike - log(minutes + 1)
                    ivt_h       = 0.0
                    walk_logmin = 0.0
                    bike_logmin = float(np.log(bike_time_min + 1.0))

                avail = 1
                if alt == 3 and not walk_ok:
                    avail = 0
                if alt == 4 and not bike_ok:
                    avail = 0

                # Encode personalized attributes as alt-specific columns
                # (idco → idca: place value only for the relevant alternatives)
                rows.append({
                    'case_id':      pid,
                    'alt_id':       alt,
                    'chosen':       int(chosen == alt),
                    'avail':        avail,
                    'gse_g':        float(_gse_map.get(pid, 0.0)),  # GSE gradient score
                    'mxl_ca_ivt_h':   ivt_h,        # hours (auto/transit)
                    'mxl_ca_walk_h':  walk_logmin,  # log-minutes (walk)
                    'mxl_ca_bike_h':  bike_logmin,  # log-minutes (bike); keep names unique to avoid rewrite collisions
                    # global-cache collision: the NL DataTree model compiles log(od.DISTBIKE*4+1)
                    # and sharrow internally caches the result as 'bike_logmin'.  Using the same
                    # name here in the flat-Dataset MXL model can cause a circular
                    # "rewrite to itself" compile error if it collides with cached symbols.
                    # vehicle ownership - DA / Other
                    'p_vehrd_da':   vehrd   if alt == 1 else 0.0,
                    'p_vehrd_sr':   vehrd   if alt == 2 else 0.0,
                    # no-vehicle → transit / walk
                    'p_noveh_tr':   noveh   if alt == 5 else 0.0,
                    'p_noveh_wk':   noveh   if alt == 3 else 0.0,
                    # licence → DA
                    'p_lic_da':     licence if alt == 1 else 0.0,
                    # worker type → DA / SR
                    'p_ftw_da':     ftwork  if alt == 1 else 0.0,
                    'p_ptw_sr':     ptwork  if alt == 2 else 0.0,
                    # age → walk / bike penalty
                    'p_age_wk':     age5p   if alt == 3 else 0.0,
                    'p_age_bk':     age5p   if alt == 4 else 0.0,
                    'p_young_bk':   young   if alt == 4 else 0.0,
                    # household size → SR
                    'p_nprd_sr':    nprd    if alt == 2 else 0.0,
                    # bike ownership → bike
                    'p_bikes_bk':   bikes   if alt == 4 else 0.0,
                    # sex → transit / walk
                    'p_fem_tr':     female  if alt == 5 else 0.0,
                    'p_fem_wk':     female  if alt == 3 else 0.0,
                    # income → DA (positive) / transit (negative)
                    'p_incp_da':    incp    if alt == 1 else 0.0,
                    'p_incp_tr':    incp    if alt == 5 else 0.0,
                    # ── Industry of employment ────────────────────────────
                    # Industrial/agri workers → DA (remote sites, no transit service)
                    'p_ind_da':     ind_industrial if alt == 1 else 0.0,
                    # Office/professional → slightly more transit
                    'p_off_tr':     ind_office     if alt == 5 else 0.0,
                    # Public sector (admin/education) → transit-amenable CBD locations
                    'p_pub_tr':     ind_public     if alt == 5 else 0.0,
                    'p_pub_wk':     ind_public     if alt == 3 else 0.0,
                    # Health workers → transit (hospitals well-served)
                    'p_hlt_tr':     ind_health     if alt == 5 else 0.0,
                    # Retail/hospitality → SR (variable hours reduce PT use)
                    'p_ret_sr':     ind_retail     if alt == 2 else 0.0,
                    # ── Occupation (ANZSCO 1-digit) ───────────────────────
                    # Managers/Professionals → DA (flexible hours, employer parking)
                    'p_mgr_da':     occ_mgr_prof   if alt == 1 else 0.0,
                    # Labourers/Machinery → DA (site-based, tools/equipment)
                    'p_lab_da':     occ_labourer   if alt == 1 else 0.0,
                    # Clerical/Admin → transit (CBD offices, regular hours)
                    'p_clr_tr':     occ_clerical   if alt == 5 else 0.0,
                    # ── Studying status ───────────────────────────────────
                    # Full-time students → transit + walk (campus access)
                    'p_fts_tr':     is_ft_student  if alt == 5 else 0.0,
                    'p_fts_wk':     is_ft_student  if alt == 3 else 0.0,
                    'p_fts_bk':     is_ft_student  if alt == 4 else 0.0,
                    # Part-time students → slight transit preference
                    'p_pts_tr':     is_pt_student  if alt == 5 else 0.0,
                })

        mxl_df = pd.DataFrame(rows)
        # MultiIndex: (case_id, alt_id)
        mxl_df = mxl_df.set_index(['case_id', 'alt_id'])

        try:
            mxl_df.loc[mxl_df['chosen'] == 1, 'avail'] = 1
        except Exception:
            pass

        # build larch Dataset
        ds = lx.Dataset.construct.from_idca(mxl_df)

        # ── Step 1: MNL baseline ──────────────────────────────────────────
        mnl = _lx_model(ds, compute_engine='jax')
        _apply_param_ordering(mnl, kind='mode')
        mnl.choice_ca_var       = 'chosen'
        mnl.availability_ca_var = 'avail'
        mnl.utility_ca = (
            P('c_ivt')          * X('mxl_ca_ivt_h')
            + P('c_walk')       * X('mxl_ca_walk_h')
            + P('c_bike')       * X('mxl_ca_bike_h')
            # vehicle ownership / licence
            + P('c_vehrd_da')   * X('p_vehrd_da')
            + P('c_vehrd_sr')   * X('p_vehrd_sr')
            + P('c_noveh_tr')   * X('p_noveh_tr')
            + P('c_noveh_wk')   * X('p_noveh_wk')
            + P('c_lic_da')     * X('p_lic_da')
            # worker type
            + P('c_ftw_da')     * X('p_ftw_da')
            + P('c_ptw_sr')     * X('p_ptw_sr')
            # age / demographics
            + P('c_age_wk')     * X('p_age_wk')
            + P('c_age_bk')     * X('p_age_bk')
            + P('c_young_bk')   * X('p_young_bk')
            + P('c_nprd_sr')    * X('p_nprd_sr')
            + P('c_bikes_bk')   * X('p_bikes_bk')
            + P('c_fem_tr')     * X('p_fem_tr')
            + P('c_fem_wk')     * X('p_fem_wk')
            + P('c_incp_da')    * X('p_incp_da')
            + P('c_incp_tr')    * X('p_incp_tr')
            # ── industry of employment ────────────────────────────────────
            + P('c_ind_da')     * X('p_ind_da')   # industrial/agri → DA
            + P('c_off_tr')     * X('p_off_tr')   # office/professional → transit
            + P('c_pub_tr')     * X('p_pub_tr')   # public sector → transit
            + P('c_pub_wk')     * X('p_pub_wk')   # public sector → walk
            + P('c_hlt_tr')     * X('p_hlt_tr')   # health workers → transit
            + P('c_ret_sr')     * X('p_ret_sr')   # retail/hospitality → SR
            # ── occupation ────────────────────────────────────────────────
            + P('c_mgr_da')     * X('p_mgr_da')   # managers/professionals → DA
            + P('c_lab_da')     * X('p_lab_da')   # labourers → DA
            + P('c_clr_tr')     * X('p_clr_tr')   # clerical → transit
            # ── studying ─────────────────────────────────────────────────
            + P('c_fts_tr')     * X('p_fts_tr')   # FT students → transit
            + P('c_fts_wk')     * X('p_fts_wk')   # FT students → walk
            + P('c_fts_bk')     * X('p_fts_bk')   # FT students → bike
            + P('c_pts_tr')     * X('p_pts_tr')   # PT students → transit
        )
        # ASCs (DA=0 reference) + random DA affinity term
        # c_da_aff captures person-level taste heterogeneity in DA preference;
        # made random (Normal) in Step 2 below.
        for alt, name in [(2,'ASC_SR'),(3,'ASC_Walk'),(4,'ASC_Bike'),(5,'ASC_Transit')]:
            mnl.utility_co[alt] = P(name)
        mnl.utility_co[1] = P('c_da_aff')   # DA affinity (fixed in MNL, random in MXL)

        # ── GSE systematic gradient loadings γ_k·(g_n·x_k) ────────────────────
        # Add the gradient-score interaction for each random taste parameter, so
        # its mean shifts systematically with the person's origin context before
        # the Normal mixture (Step 2) adds σ_k·η_nk on top. Together this is the
        # exact MixedLogitGSE decomposition, estimated natively in Larch.
        if _gse_enabled:
            mnl.utility_ca = (
                mnl.utility_ca
                + P('g_ivt')  * X('mxl_ca_ivt_h * gse_g')
                + P('g_walk') * X('mxl_ca_walk_h * gse_g')
                + P('g_bike') * X('mxl_ca_bike_h * gse_g')
            )
            # DA affinity gradient loading (g_n shifts the person-specific DA ASC)
            mnl.utility_co[1] = P('c_da_aff') + P('g_da') * X('gse_g')
            for _gp in ('g_ivt', 'g_walk', 'g_bike', 'g_da'):
                try:
                    mnl.set_value(_gp, 0.0)
                except Exception:
                    pass
            logger.info(f"[{segment}] GSE systematic loadings added "
                        "(g_ivt, g_walk, g_bike, g_da) — full GSE = μ + γ·g + σ·η")

        # Starting values from SEGMENT_COEFFICIENTS if available, else defaults.
        seg_init = SEGMENT_COEFFICIENTS.get(segment, {})
        mnl.set_value('c_ivt',        -1.20)
        mnl.set_value('c_walk',       -1.50)
        mnl.set_value('c_bike',       -1.00)
        mnl.set_value('ASC_SR',    seg_init.get('ASC_SR',     -0.5))
        mnl.set_value('ASC_Walk',     seg_init.get('ASC_Walk',   -3.0))
        mnl.set_value('ASC_Bike',     seg_init.get('ASC_Bike',   -3.5))
        mnl.set_value('ASC_Transit',  seg_init.get('ASC_Transit',-1.5))
        mnl.set_value('c_vehrd_da',   seg_init.get('c_veh_da',    0.30))
        mnl.set_value('c_vehrd_sr',   seg_init.get('c_veh_sr',    0.10))
        mnl.set_value('c_noveh_tr',   seg_init.get('c_noveh_transit', 2.0))
        mnl.set_value('c_noveh_wk',   seg_init.get('c_noveh_walk',    1.5))
        # c_lic_da removed - CARLICENCE is now a hard availability constraint
        mnl.set_value('c_ftw_da',     seg_init.get('c_ftwork_da',     0.5))
        mnl.set_value('c_ptw_sr',     seg_init.get('c_ptwork_sr',     0.2))
        mnl.set_value('c_age_wk',     seg_init.get('c_age_walk',     -0.10))
        mnl.set_value('c_age_bk',     seg_init.get('c_age_bike',     -0.15))
        mnl.set_value('c_young_bk',   seg_init.get('c_young_bike',    0.50))
        mnl.set_value('c_nprd_sr',    seg_init.get('c_hhsize_sr',     0.10))
        mnl.set_value('c_bikes_bk',   seg_init.get('c_bikes_bike',    0.40))
        mnl.set_value('c_fem_tr',     seg_init.get('c_female_transit',0.30))
        mnl.set_value('c_fem_wk',     seg_init.get('c_female_walk',   0.20))
        mnl.set_value('c_incp_da',    seg_init.get('c_incp_da',       0.010))
        mnl.set_value('c_incp_tr',    seg_init.get('c_incp_transit',  -0.015))
        # DA affinity: random intercept (zero mean at MNL start; becomes random in MXL)
        mnl.set_value('c_da_aff',     0.0)
        # ── Industry starting values (behaviorally calibrated to SEQ context) ──
        # Industrial/agri → DA: strong positive (remote sites, no PT service)
        mnl.set_value('c_ind_da',     0.70)
        # Office/professional → transit: modest (CBD access, some parking costs)
        mnl.set_value('c_off_tr',     0.25)
        # Public sector → transit: stronger (CBDs, Education/Health precincts)
        mnl.set_value('c_pub_tr',     0.45)
        mnl.set_value('c_pub_wk',     0.15)
        # Health workers → transit: hospitals on PT corridors
        mnl.set_value('c_hlt_tr',     0.30)
        # Retail/hospitality → SR: variable hours reduce transit feasibility
        mnl.set_value('c_ret_sr',     0.20)
        # ── Occupation starting values ──────────────────────────────────────
        mnl.set_value('c_mgr_da',     0.25)   # managers → employer parking, DA
        mnl.set_value('c_lab_da',     0.50)   # labourers → tools/equipment, DA
        mnl.set_value('c_clr_tr',     0.30)   # clerical → CBD office, transit
        # ── Studying starting values ────────────────────────────────────────
        mnl.set_value('c_fts_tr',     0.55)   # FT students → campus transit
        mnl.set_value('c_fts_wk',     0.30)   # FT students → campus walkable
        mnl.set_value('c_fts_bk',     0.20)   # FT students → cycling
        mnl.set_value('c_pts_tr',     0.20)   # PT students → some transit
        mnl.set_cap(8)

        for _p in (
            'c_clr_tr', 'c_mgr_da', 'c_lab_da', 'c_ind_da', 'c_off_tr',
            'c_pub_tr', 'c_pub_wk', 'c_hlt_tr', 'c_ret_sr',
            'c_fts_tr', 'c_fts_wk', 'c_fts_bk', 'c_pts_tr'
        ):
            try:
                if _p in mnl.pf.index:
                    mnl.set_upper(_p, 3.0)
                    mnl.set_lower_bound(_p, -3.0)
            except Exception:
                pass

        # Sign constraints - prevent behaviorally implausible outcomes
        # c_ivt_* must be negative (more travel time → less utility)
        # c_crash_veh must be ≤ 0 (more crash risk → less utility for vehicle modes)
        for _p in ('c_ivt_auto', 'c_ivt_transit', 'c_walk_time', 'c_bike_time'):
            try:
                mnl.set_value(_p, min(mnl.get_value(_p), -0.01))
                # set_upper constrains param ≤ value during optimisation
                mnl.set_upper(_p, -0.001)
            except Exception:
                pass
        try:
            mnl.set_upper('c_crash_veh', 0.0)
        except Exception:
            pass

        _ennet_mnl = elasticnet_objective(mnl)
        _ennet_mnl.__enter__()
        try:
            mnl.maximize_loglike(options={'ftol': 1e-9, 'maxiter': 500})
        finally:
            _ennet_mnl.__exit__(None, None, None)

        # ── Step 2: Add Normal mixtures ───────────────────────────────────
        # Random parameters capture taste heterogeneity across persons:
        #   c_ivt      ~ Normal: IVT sensitivity varies (commuters vs leisure)
        #   c_walk     ~ Normal: walk aversion varies (fitness, weather sensitivity)
        #   c_bike     ~ Normal: bike aversion varies (infrastructure access)
        #   c_da_aff   ~ Normal: DA affinity varies (person-specific constant)
        #                        Key driver of DA dominance - individuals who
        #                        intrinsically prefer driving regardless of time.
        mxl = mnl.copy()
        _random_params = [
            ('c_ivt',    's_ivt'),    # IVT sensitivity ~ Normal(c_ivt, s_ivt)
            ('c_walk',   's_walk'),   # walk aversion   ~ Normal(c_walk, s_walk)
            ('c_bike',   's_bike'),   # bike aversion   ~ Normal(c_bike, s_bike)
            ('c_da_aff', 's_da'),     # DA affinity     ~ Normal(c_da_aff, s_da)
        ]
        # Central random-coefficient policy: never_random filters the list,
        # force_random + GA_STAGE5_RAND extend it (any Larch param present in
        # the fitted MNL can be promoted to a random coefficient).
        try:
            from ga_constraints import get_constraints, stage5_extra_random_params
            _c5 = get_constraints('stage5_mode')
            _never5 = set(_c5.get('never_random', []))
            _random_params = [(p, s) for p, s in _random_params if p not in _never5]
            _have5 = {p for p, _ in _random_params}
            _extras = [p for p in list(_c5.get('force_random', {}).keys())
                       + stage5_extra_random_params()
                       if p not in _have5 and p not in _never5]
            for _p in _extras:
                if _p in mnl.pf.index:
                    _sd = 's_' + (_p[2:] if _p.startswith('c_') else _p)
                    _random_params.append((_p, _sd))
                    logger.info(f"[{segment}] MXL: added random coefficient {_p} ~ N(., {_sd}) "
                                "(ga_constraints policy)")
        except Exception as _rp_e:
            logger.debug(f"[{segment}] MXL random-param policy skipped: {_rp_e}")
        for param, sd_name in _random_params:
            mxl.mixtures.append(lx.mixtures.Normal(param, sd_name))

        mxl.n_draws = n_draws
        mxl.seed    = seed

        # Initialise std dev params at 0.5 with positive lower bound so the
        # gradient is non-trivial from the start.  Starting at 1.0 puts the
        # model in a near-flat region; 0.5 keeps gradients alive.
        # set_lower_bound enforces s > 0 so the Normal draw is always meaningful.
        for _, sd_name in _random_params:
            try:
                mxl.set_value(sd_name, 0.5)
                mxl.set_cap(sd_name, 5.0)        # upper bound
                mxl.set_lower_bound(sd_name, 0.01)  # enforce positive std dev
            except Exception:
                try:
                    mxl.set_value(sd_name, 0.5)
                    mxl.set_cap(sd_name, 5.0)
                except Exception:
                    pass

        # No groupid panel correction - person_id is the index so sharrow
        # cannot rewrite it as a regular column (causes 'unable to rewrite' error).
        mxl_final = mxl

        # Snapshot starting values so an all-stuck (zero gradient) outcome can
        # be detected after estimation — see check below coefficient extraction.
        try:
            _mxl_start_vals = {name: float(mxl_final.pf.loc[name, 'value'])
                               for name in mxl_final.pf.index}
        except Exception:
            _mxl_start_vals = None

        print(f"\n  Estimating mixed logit ({n_draws} Halton draws)...")
        # Two-phase optimisation: BFGS first (fast, moves params far from start),
        # then L-BFGS-B to refine near the optimum.
        _ennet_mxl = elasticnet_objective(mxl_final)
        _ennet_mxl.__enter__()
        try:
            try:
                mxl_final.maximize_loglike(
                    stderr=False,
                    method='BFGS',
                    options={'gtol': 1e-4, 'maxiter': 200},
                )
            except Exception:
                pass
            mxl_final.maximize_loglike(stderr=True, options={'ftol': 1e-10, 'gtol': 1e-6, 'maxiter': 800})
        finally:
            _ennet_mxl.__exit__(None, None, None)

        # ── extract results ──────────────────────────────────────────
        pf   = mxl_final.pf
        try:
            ll   = float(mxl_final.most_recent_estimation_result.loglike)
        except Exception:
            ll   = float('nan')
        rho2 = _rho_sq_null(mxl_final, ll)

        coefs  = {name: float(pf.loc[name, 'value'])   for name in pf.index}
        _finite_coefs = {k: v for k, v in coefs.items() if np.isfinite(v)}
        if not _finite_coefs:
            logger.warning(f"[{segment}] MXL returned no finite coefficients - falling back to NL")
            return None, None
        # All-stuck detection: job 23540863's nonmand MXL finished with all 32
        # params (incl. every ASC and Mu_Motor) exactly at their starting
        # values — the jax gradient was zero, so the optimizer never moved. A
        # prior-valued MXL is strictly worse than the NL fallback, which
        # estimates correctly, so treat all-stuck the same as no-finite.
        if _mxl_start_vals is not None:
            _any_moved = any(
                np.isfinite(v) and p in _mxl_start_vals
                and abs(v - _mxl_start_vals[p]) > 1e-12
                for p, v in coefs.items()
            )
            if not _any_moved:
                logger.warning(
                    f"[{segment}] MXL: ALL {len(coefs)} params unmoved from their "
                    "starting values (zero gradient — jax engine returned no "
                    "usable gradient) - falling back to NL"
                )
                return None, None
        # std_err column name varies across Larch versions ('std_err', 'se', 'stderr')
        _se_col = next((c for c in ['std_err', 'se', 'stderr'] if c in pf.columns), None)
        ses    = {name: float(pf.loc[name, _se_col]) if _se_col else float('nan')
                  for name in pf.index}
        tstats = {name: (coefs[name] / ses[name] if ses.get(name, 0) != 0 else 0.0)
                  for name in pf.index}

        _rand_names = {p for p, _ in _random_params}
        result = {
            'coef':     coefs,
            'std_err':  ses,
            'tstat':    tstats,
            'll':       ll,
            'rho2':     rho2,
            'randvars': {p: 'Normal' for p in _rand_names},
            'n_obs':    len(seg),
            'n_draws':  n_draws,
        }

        # print summary
        print(f"\n  ┌─ {segment.upper()} Mixed Logit  LL={ll:.1f}  ρ²={rho2:.3f}"
              f"  draws={n_draws} ─────────────────")
        print(f"  {'Parameter':<25} {'Value':>9}  {'Std Err':>8}  {'t-stat':>7}  Type")
        print(f"  {'─'*65}")
        rand_params_set = _rand_names | {f's_{p[2:]}' if p.startswith('c_') else f's_{p}'
                                         for p in _rand_names}
        for name in pf.index:
            v  = coefs[name]; se = ses[name]; t = tstats[name]
            sig = '***' if abs(t)>3.3 else '**' if abs(t)>2.6 else '*' if abs(t)>1.96 else ''
            kind = ('random~N' if name in _rand_names else
                    'std dev'  if name.startswith('s_') else 'fixed')
            print(f"  {str(name):<25} {v:>9.4f}  {se:>8.4f}  {t:>7.2f}  {sig:<4} {kind}")
        print(f"  └{'─'*65}")

        # ── Compute frozen cell probabilities from HHTS data ──────────────
        # After estimation we evaluate the model at the MEAN coefficients
        # (no random draws) for each HHTS person, then aggregate to person-
        # type cells.  These cells are applied directly to the synthetic
        # population - no re-estimation or MC noise at prediction time.
        frozen_probs = _compute_frozen_probs(seg, result, segment)
        if frozen_probs is not None:
            result['frozen_probs'] = frozen_probs
            logger.info(
                f"[{segment}] MXL frozen probabilities computed: "
                f"{len(frozen_probs)} cells, "
                f"mean DA={frozen_probs['p_DA'].mean():.1%}  "
                f"Walk={frozen_probs['p_Walk'].mean():.1%}  "
                f"Bike={frozen_probs['p_Bike'].mean():.1%}  "
                f"Transit={frozen_probs['p_Transit'].mean():.1%}"
            )
        else:
            logger.warning(f"[{segment}] Frozen probability computation failed - "
                           "falling back to numpy model at prediction time")

        # Inject coefficient cache so _coef_dict_from_model can find coefficients
        # without needing to access model.pf (which may return NaN after JAX
        # compilation clears the internal parameter store).
        _valid_coefs = {k: v for k, v in coefs.items()
                        if not (isinstance(v, float) and np.isnan(v))}
        if _valid_coefs:
            mxl_final._coef_cache = _valid_coefs
            logger.debug(f"[{segment}] MXL _coef_cache injected with "
                         f"{len(_valid_coefs)} params")
        else:
            logger.warning(f"[{segment}] MXL coefs all NaN - _coef_cache not set; "
                           "logsums will use NL model coefficients")

        return result, mxl_final

    except Exception as e:
        logger.warning(f"Mixed logit failed for {segment}: {e}")
        import traceback; logger.debug(traceback.format_exc())
        return None, None


def _compute_frozen_probs(seg_df: pd.DataFrame, mxl_result: dict,
                          segment: str) -> 'pd.DataFrame | None':
    """
    Compute cell-level frozen mode probabilities from HHTS observations.

    Evaluates the MXL model at its MEAN coefficients (deterministic - no Monte
    Carlo draws) for each HHTS observation, then aggregates to person-type cells
    defined by ``FROZEN_CELL_COLS``.  These cells are stored and applied directly
    to the synthetic population, ensuring:
      - Stable, reproducible predictions (zero MC noise across runs)
      - Full transferability: HHTS attributes → identical cell keys in synth pop
      - DA dominance corrected by industry/occupation/studying heterogeneity

    The utility formula mirrors ``_mode_probs_numpy`` with the additional
    industry/occupation/studying terms estimated by ``_estimate_mixed_logit``.

    Returns
    -------
    DataFrame with columns FROZEN_CELL_COLS + FROZEN_PROB_COLS + ['n_obs'],
    or None on failure.
    """
    if not mxl_result or not mxl_result.get('coef'):
        return None
    try:
        coef = mxl_result['coef']

        def pv(name, default=0.0):
            v = coef.get(name)
            if v is None:
                return default
            try:
                fv = float(v)
                return fv if not np.isnan(fv) else default
            except Exception:
                return default

        df = seg_df.copy().reset_index(drop=True)
        n  = len(df)

        def _col(name, default):
            return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)),
                                 errors='coerce').fillna(default).values

        # ── Skim values ────────────────────────────────────────────────────
        ivt_auto_min   = _col('SOVTOLL_TIME__AM', 30.0)
        ivt_transit_min = _col('WLK_LOC_WLK_TOTIVT__AM', 45.0)
        dist_walk_km   = _col('DISTWALK', 5.0)
        dist_bike_km   = _col('DISTBIKE', 5.0)

        ivt_auto_h    = np.clip(ivt_auto_min   / 60.0, 0.05, 5.0)
        ivt_transit_h = np.clip(ivt_transit_min / 60.0, 0.05, 6.0)
        walk_t = np.log(np.clip(dist_walk_km, 0, None) * 12.0 + 1.0)
        bike_t = np.log(np.clip(dist_bike_km, 0, None) * 4.0  + 1.0)

        # ── Person attributes ──────────────────────────────────────────────
        vehrd   = _col('vehrd', 1.0)
        licence = _col('CARLICENCE', 1.0)
        incp    = _col('incp', 7.0)
        age5p   = _col('age5p', 4.0)
        # Age gate on licence (mirrors _mode_probs_numpy): under-15s can never
        # hold a licence, so DA must be unavailable to them even when the
        # CARLICENCE column is missing and defaulted to 1.
        licence = np.where(age5p <= 3, 0.0, licence)
        sexp    = _col('sexp', 1.0)
        noveh   = (vehrd == 0).astype(float)
        female  = (sexp == 2).astype(float)
        young   = (age5p <= 3).astype(float)
        workstat = _col('WORKSTAT', 0.0)
        studying = _col('STUDYING', 0.0)
        is_worker  = np.clip(workstat, 0, 1).astype(float)
        is_student = (studying > 0).astype(float)

        # ── Industry of employment ─────────────────────────────────────────
        indp_raw = _col('indp', 20.0).astype(int)
        _indp_groups = np.array([INDUSTRY_MODE_GROUP.get(int(v), 'other') for v in indp_raw])
        ind_industrial = np.isin(_indp_groups, ['industrial', 'agri']).astype(float)
        ind_office     = np.isin(_indp_groups, ['office', 'professional']).astype(float)
        ind_public     = (_indp_groups == 'public').astype(float)
        ind_health     = (_indp_groups == 'health').astype(float)
        ind_retail     = np.isin(_indp_groups, ['retail', 'hospitality']).astype(float)

        # ── Occupation (ANZSCO 1-digit) ────────────────────────────────────
        occp_raw    = _col('occp', 0.0).astype(int)
        occ_mgr_prof = np.isin(occp_raw, [1, 2]).astype(float)
        occ_labourer = np.isin(occp_raw, [7, 8]).astype(float)
        occ_clerical = (occp_raw == 5).astype(float)

        # ── Studying status ────────────────────────────────────────────────
        stup_raw = _col('stup', 1.0).astype(int)
        is_ft_student = (stup_raw == 2).astype(float)
        is_pt_student = np.isin(stup_raw, [3, 4]).astype(float)

        # ── Availability ───────────────────────────────────────────────────
        walk_avail_arr    = _col('walk_avail', 1.0)
        bike_avail_arr    = _col('bike_avail', 1.0)
        transit_avail_arr = ((ivt_transit_min > 0) & (ivt_transit_min < 90.0)).astype(float)

        # ── Utility at MEAN coefficients ───────────────────────────────────
        V_DA = (pv('c_ivt') * ivt_auto_h
              + pv('c_vehrd_da') * vehrd
              + pv('c_incp_da') * incp
              + pv('c_ftw_da') * is_worker
              + pv('c_da_aff', 0.0)
              + pv('c_ind_da', 0.0) * ind_industrial
              + pv('c_mgr_da', 0.0) * occ_mgr_prof
              + pv('c_lab_da', 0.0) * occ_labourer)
        V_DA = np.where(licence > 0, V_DA, -1e9)

        V_SR = (pv('ASC_SR', -0.5)
              + pv('c_ivt') * ivt_auto_h
              + pv('c_vehrd_sr') * vehrd
              + pv('c_ret_sr', 0.0) * ind_retail)

        V_TR = (pv('ASC_Transit', -0.5)
              + pv('c_ivt') * ivt_transit_h
              + pv('c_noveh_tr') * noveh
              + pv('c_incp_tr') * incp
              + pv('c_fem_tr') * female
              + pv('c_student_tr', pv('c_fts_tr', 0.5)) * is_student
              + pv('c_off_tr', 0.0) * ind_office
              + pv('c_pub_tr', 0.0) * ind_public
              + pv('c_hlt_tr', 0.0) * ind_health
              + pv('c_clr_tr', 0.0) * occ_clerical
              + pv('c_fts_tr', 0.0) * is_ft_student
              + pv('c_pts_tr', 0.0) * is_pt_student)
        V_TR = np.where(transit_avail_arr > 0, V_TR, -1e9)

        # c_incp_active: income effect on Walk/Bike (mirrors MODE_UEC).
        # NL model uses 'c_incp_active'; MXL may use same or 'c_incp_wk'.
        # Without this, high-income persons get income penalty on DA/SR/Transit
        # but NOT on Walk/Bike, artificially inflating active mode probabilities.
        _incp_active_wk = pv('c_incp_active', pv('c_incp_wk', -0.010))
        _incp_active_bk = pv('c_incp_active', pv('c_incp_bk', -0.010))

        V_WK = (pv('ASC_Walk', -1.0)
              + pv('c_walk', pv('c_walk_time', -1.50)) * walk_t
              + pv('c_noveh_wk') * noveh
              + pv('c_age_wk', pv('c_age_walk', -0.08)) * age5p
              + _incp_active_wk * incp
              + pv('c_fem_wk', pv('c_female_walk', 0.0)) * female
              + pv('c_pub_wk', 0.0) * ind_public
              + pv('c_fts_wk', 0.0) * is_ft_student)
        V_WK = np.where(walk_avail_arr > 0, V_WK, -1e9)

        V_BK = (pv('ASC_Bike', -1.5)
              + pv('c_bike', pv('c_bike_time', -1.00)) * bike_t
              + pv('c_age_bk', pv('c_age_bike', -0.12)) * age5p
              + _incp_active_bk * incp
              + pv('c_young_bk', pv('c_young_bike', 0.50)) * young
              + pv('c_fts_bk', 0.0) * is_ft_student)
        V_BK = np.where(bike_avail_arr > 0, V_BK, -1e9)

        # ── Nested logit (Root{Motor{Car{DA,SR}, Transit}, NonMotor{Walk,Bike}}) ──
        # Was a flat 5-way softmax; see _nested_mode_probs for why that's
        # mathematically inconsistent with coefficients fit under the real
        # nested structure (it was collapsing SR/Bike/Transit toward Walk).
        mu_car      = _sanitize_mu(pv('Mu_Car',      1.0), 'Mu_Car')
        mu_motor    = _sanitize_mu(pv('Mu_Motor',    1.0), 'Mu_Motor')
        mu_nonmotor = _sanitize_mu(pv('Mu_NonMotor', 1.0), 'Mu_NonMotor')
        probs = np.asarray(_nested_mode_probs(
            V_DA, V_SR, V_WK, V_BK, V_TR,
            mu_car=mu_car, mu_motor=mu_motor, mu_nonmotor=mu_nonmotor, xp=np,
        ))  # (n, 5) - DA, SR, Walk, Bike, Transit

        # ── Cell membership ────────────────────────────────────────────────
        out = df.copy()
        out['p_DA']      = probs[:, 0]
        out['p_SR']      = probs[:, 1]
        out['p_Walk']    = probs[:, 2]
        out['p_Bike']    = probs[:, 3]
        out['p_Transit'] = probs[:, 4]

        # ptype: string label for person type
        if 'ptype' not in out.columns:
            out['ptype'] = 'Unknown'
        else:
            out['ptype'] = out['ptype'].fillna('Unknown').astype(str)

        # indp_group: from ANZSIC code → mode group string
        out['indp_group'] = pd.Series(_indp_groups, index=out.index).fillna('other')

        # income_band: quartiles of incp scale (1-14 bands in HHTS)
        _incp_s = pd.to_numeric(df.get('incp', pd.Series(7.0, index=df.index)),
                                errors='coerce').fillna(7.0)
        out['income_band'] = pd.cut(_incp_s, bins=[0, 4, 7, 10, 99],
                                    labels=['low', 'mid', 'high', 'very_high'],
                                    right=True).astype(str).fillna('mid')

        # veh_bin: vehicle ownership binary
        out['veh_bin'] = np.where(vehrd > 0, 'hasveh', 'noveh')

        # age_group: broad age bands from age5p (1-digit ABS coding)
        out['age_group'] = pd.cut(pd.Series(age5p, index=out.index),
                                  bins=[0, 2, 4, 7, 99],
                                  labels=['young', 'middle', 'older', 'senior'],
                                  right=True).astype(str).fillna('middle')

        # has_licence
        out['has_licence'] = np.where(licence > 0, 'yes', 'no')

        # ── Aggregate to cells ─────────────────────────────────────────────
        cell_df = (out.groupby(FROZEN_CELL_COLS, observed=True)
                   .agg(p_DA=('p_DA', 'mean'), p_SR=('p_SR', 'mean'),
                        p_Walk=('p_Walk', 'mean'), p_Bike=('p_Bike', 'mean'),
                        p_Transit=('p_Transit', 'mean'), n_obs=('p_DA', 'count'))
                   .reset_index())

        # Renormalise (floating-point safety)
        prob_sum = cell_df[FROZEN_PROB_COLS].sum(axis=1).clip(lower=1e-9)
        for col in FROZEN_PROB_COLS:
            cell_df[col] = cell_df[col] / prob_sum

        logger.debug(
            f"_compute_frozen_probs [{segment}]: {len(cell_df)} cells "
            f"from {n} HHTS obs | "
            f"mean DA={cell_df['p_DA'].mean():.1%} "
            f"Walk={cell_df['p_Walk'].mean():.1%} "
            f"Transit={cell_df['p_Transit'].mean():.1%}"
        )
        return cell_df

    except Exception as e:
        logger.warning(f"_compute_frozen_probs [{segment}] failed: {e}")
        import traceback; logger.debug(traceback.format_exc())
        return None


def _assign_frozen_cells(persons_df: pd.DataFrame,
                         tour_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Derive FROZEN_CELL_COLS attributes for a synthetic-population DataFrame.

    Mirrors the cell-assignment logic in ``_compute_frozen_probs`` so that
    synthetic persons map to the same cell keys as HHTS persons.

    Parameters
    ----------
    persons_df : person-level attributes (synthetic population)
    tour_rows  : tour-level rows - used to join person_id → person attributes

    Returns
    -------
    DataFrame with the same index as ``tour_rows`` and columns FROZEN_CELL_COLS.
    """
    def _col(df, name, default):
        return pd.to_numeric(df.get(name, pd.Series(default, index=df.index)),
                             errors='coerce').fillna(default)

    pp = persons_df.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})

    pid_col = next((c for c in ['person_id', 'PERSID'] if c in tour_rows.columns), None)
    n = len(tour_rows)

    if pid_col:
        pids   = tour_rows[pid_col].values
        _nonan = pp[pp['person_id'].notna()].drop_duplicates('person_id', keep='first')
        pp_idx = _nonan.set_index('person_id')
        if not pp_idx.index.is_unique:
            pp_idx = pp_idx[~pp_idx.index.duplicated(keep='first')]

        def _pa(col, default):
            if col in pp_idx.columns:
                lkup = pp_idx[col].to_dict()
                return np.array([lkup.get(p, default) for p in pids], dtype=float)
            return np.full(n, default, dtype=float)
    else:
        def _pa(col, default):
            return np.full(n, default, dtype=float)

    vehrd   = _pa('vehrd', 1.0)
    licence = _pa('CARLICENCE', 1.0)
    incp    = _pa('incp', 7.0)
    age5p   = _pa('age5p', 4.0)
    indp_raw = _pa('indp', 20.0).astype(int)

    # ptype - try tour_rows first, then persons
    if 'ptype' in tour_rows.columns:
        ptype_arr = tour_rows['ptype'].fillna('Unknown').astype(str).values
    elif pid_col and 'ptype' in pp_idx.columns:
        lkup = pp_idx['ptype'].to_dict()
        ptype_arr = np.array([str(lkup.get(p, 'Unknown')) for p in pids])
    else:
        ptype_arr = np.full(n, 'Unknown', dtype=object)

    indp_groups  = np.array([INDUSTRY_MODE_GROUP.get(int(v), 'other') for v in indp_raw])

    incp_s   = pd.Series(incp)
    inc_band = pd.cut(incp_s, bins=[0, 4, 7, 10, 99],
                      labels=['low', 'mid', 'high', 'very_high'],
                      right=True).astype(str).fillna('mid').values

    age_band = pd.cut(pd.Series(age5p), bins=[0, 2, 4, 7, 99],
                      labels=['young', 'middle', 'older', 'senior'],
                      right=True).astype(str).fillna('middle').values

    cells = pd.DataFrame({
        'ptype':       ptype_arr,
        'indp_group':  indp_groups,
        'income_band': inc_band,
        'veh_bin':     np.where(vehrd > 0, 'hasveh', 'noveh'),
        'age_group':   age_band,
        'has_licence': np.where(licence > 0, 'yes', 'no'),
    }, index=tour_rows.index)
    return cells


def _mode_probs_frozen(frozen_probs: pd.DataFrame,
                       tour_rows: pd.DataFrame,
                       persons: pd.DataFrame) -> 'np.ndarray | None':
    """
    Apply frozen cell-level mode probabilities to synthetic population tours.

    Matches each tour to its HHTS-estimated cell via ``_assign_frozen_cells``,
    then looks up the cell's mode probabilities.  Unmatched tours fall back to
    the cell with the closest ptype or overall mean.

    Returns
    -------
    np.ndarray of shape (n_tours, 5) - columns [DA, SR, Walk, Bike, Transit]
    or None if frozen_probs is empty / malformed.
    """
    if frozen_probs is None or len(frozen_probs) == 0:
        return None
    try:
        cells = _assign_frozen_cells(persons, tour_rows).reset_index(drop=True)
        merged = cells.merge(frozen_probs[FROZEN_CELL_COLS + FROZEN_PROB_COLS],
                             on=FROZEN_CELL_COLS, how='left')

        # ?? Fallback for unmatched cells ??????????????????????????????????
        # First try ptype-only mean; then global mean.
        unmatched = merged[FROZEN_PROB_COLS].isna().any(axis=1)
        if unmatched.any():
            ptype_mean = (frozen_probs.groupby('ptype', observed=True)[FROZEN_PROB_COLS]
                          .mean().reset_index())
            for idx in np.flatnonzero(unmatched.to_numpy()):
                pt = cells.iloc[idx]['ptype']
                row = ptype_mean[ptype_mean['ptype'] == pt]
                if len(row):
                    merged.loc[idx, FROZEN_PROB_COLS] = row[FROZEN_PROB_COLS].values[0]

        # Remaining unmatched → global mean
        still_unmatched = merged[FROZEN_PROB_COLS].isna().any(axis=1)
        if still_unmatched.any():
            gmean = frozen_probs[FROZEN_PROB_COLS].mean().values
            merged.loc[still_unmatched, FROZEN_PROB_COLS] = gmean

        probs = merged[FROZEN_PROB_COLS].values.astype(np.float64)

        # Renormalise rows (handle any residual NaN from fallbacks)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        probs    = probs / row_sums

        n_unmatched = int(unmatched.sum())
        if n_unmatched:
            logger.debug(f"_mode_probs_frozen: {n_unmatched}/{len(tour_rows)} "
                         "tours unmatched → ptype/global fallback used")
        return probs

    except Exception as e:
        logger.warning(f"_mode_probs_frozen failed: {e}")
        import traceback; logger.debug(traceback.format_exc())
        return None


def _stabilize_mode_probs_stage5(
    mode_probs: np.ndarray,
    alt_ids: list,
    target_shares,
    seg: str = '',
) -> np.ndarray:
    """
    Stabilize predicted mode probabilities toward segment target shares.

    Prevents DA-collapse in fallback paths (e.g., missing/weak coefficients) while
    preserving row-wise probabilistic choice behavior.
    """
    try:
        if mode_probs is None or len(mode_probs) == 0 or target_shares is None:
            return mode_probs

        arr = np.asarray(mode_probs, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(alt_ids):
            return mode_probs

        # Build target vector in alt_ids order.
        tv = np.zeros(len(alt_ids), dtype=float)
        if not hasattr(target_shares, 'get'):
            return mode_probs
        for j, aid in enumerate(alt_ids):
            v = target_shares.get(aid, None)
            if v is None:
                v = target_shares.get(str(aid), None)
            if v is None:
                _name_map = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
                v = target_shares.get(_name_map.get(int(aid), ''), None)
            try:
                tv[j] = float(v) if v is not None else 0.0
            except Exception:
                tv[j] = 0.0

        if not np.isfinite(tv).all() or tv.sum() <= 0:
            return mode_probs
        tv = tv / tv.sum()

        pred = np.clip(arr.mean(axis=0), 1e-12, None)
        pred = pred / pred.sum()

        # Severe collapse trigger - ANY mode dominating far beyond its target,
        # not just DA. Job 23880279 predicted SR=96% (target 43%) for nonmand
        # and the old DA-only check let it through with the weak alpha=0.22
        # blend (still ~84% SR after stabilisation).
        if 1 in alt_ids:
            i_da = alt_ids.index(1)
            pred_da = float(pred[i_da])
            targ_da = float(tv[i_da])
        else:
            pred_da = 0.0
            targ_da = 0.0

        max_gap = float(np.max(np.abs(pred - tv)))
        severe = bool(np.any((pred > 0.90) & (tv < 0.82))) or max_gap > 0.35
        if not severe and max_gap < 0.06:
            return mode_probs

        # Adaptive blend strength: the further the aggregate is from target,
        # the harder we pull. A mildly-off model keeps most of its own
        # heterogeneity (alpha ~0.22); an unidentified/collapsed one is
        # pulled most of the way to the target shares (alpha up to 0.85).
        alpha = float(np.clip(max_gap * 1.6, 0.22, 0.85))
        if severe:
            alpha = max(alpha, 0.60)
        target = (1.0 - alpha) * pred + alpha * tv
        floor = np.minimum(tv * 0.55, tv)
        target = np.maximum(target, floor)
        target = target / target.sum()

        adj = np.clip(arr, 1e-12, 1.0)
        for _ in range(4):
            cur = np.clip(adj.mean(axis=0), 1e-12, None)
            scale = target / cur
            adj *= scale[None, :]
            adj = np.clip(adj, 1e-12, 1.0)
            adj /= adj.sum(axis=1, keepdims=True)

        if 1 in alt_ids:
            nda = float(adj.mean(axis=0)[alt_ids.index(1)])
            logger.info(
                f"_predict_batch [{seg}] mode-share stabilisation: "
                f"DA {pred_da:.1%}->{nda:.1%} (target {targ_da:.1%})"
            )
        return adj
    except Exception as _e:
        logger.debug(f"_stabilize_mode_probs_stage5 skipped [{seg}]: {_e}")
        return mode_probs


def _estimate_dest_dist_heterogeneity(seg_df: pd.DataFrame,
                                      dest_coef: dict,
                                      segment: str) -> dict:
    """
    Estimate the person-level log-distance heterogeneity from HHTS residuals.

    Computes the between-person standard deviation of mean log(dist+1) for chosen
    destinations.  This captures genuine preference heterogeneity rather than
    noise: some people consistently travel far, others stay close.

    Returns
    -------
    dict with key ``s_log_dist`` (float) - the std dev used to scale frozen draws.
    Falls back to 0.5 if data is insufficient.
    """
    try:
        dist_col = next(
            (c for c in ['DIST', 'SOVTOLL_DIST__AM', 'dist'] if c in seg_df.columns),
            None
        )
        if dist_col is None:
            logger.debug(f"_estimate_dest_dist_heterogeneity [{segment}]: no dist col - using default 0.5")
            return {'s_log_dist': 0.5}

        log_dist = np.log(
            pd.to_numeric(seg_df[dist_col], errors='coerce').fillna(30.0).clip(lower=0.0) + 1.0
        )

        if 'person_id' in seg_df.columns and seg_df['person_id'].nunique() > 10:
            # Between-person variance in mean log-distance
            person_means = (seg_df.assign(_ld=log_dist)
                            .groupby('person_id')['_ld'].mean())
            s_log_dist = float(np.clip(person_means.std(ddof=1), 0.10, 2.0))
        else:
            # Fallback: half the total log-distance std
            s_log_dist = float(np.clip(log_dist.std() * 0.5, 0.10, 1.5))

        logger.info(
            f"_estimate_dest_dist_heterogeneity [{segment}]: "
            f"s_log_dist={s_log_dist:.3f} "
            f"(n_persons={seg_df['person_id'].nunique() if 'person_id' in seg_df.columns else '?'})"
        )
        return {'s_log_dist': s_log_dist}

    except Exception as e:
        logger.warning(f"_estimate_dest_dist_heterogeneity [{segment}] failed: {e}")
        return {'s_log_dist': 0.5}


def _frozen_person_dist_offsets(person_ids, s_log_dist: float) -> pd.Series:
    """
    Deterministic Normal(0, s_log_dist) draws seeded by MD5 hash of person_id.

    Each person receives a reproducible, run-stable distance-sensitivity offset
    that transfers identically from HHTS estimation to the synthetic population -
    as long as person_id values are consistent between the two datasets.

    The MD5 approach avoids NumPy's global RNG state so results are independent
    of what else has been sampled in the session.

    Parameters
    ----------
    person_ids : array-like of person IDs (int or str)
    s_log_dist : float - std dev from _estimate_dest_dist_heterogeneity

    Returns
    -------
    pd.Series indexed by person_id, values ~ N(0, s_log_dist).
    """
    import hashlib
    import struct

    ids = list(person_ids)
    draws = np.empty(len(ids), dtype=np.float64)
    for i, pid in enumerate(ids):
        digest = hashlib.md5(str(pid).encode('utf-8')).digest()
        seed   = struct.unpack('<I', digest[:4])[0]          # 32-bit unsigned
        rng    = np.random.default_rng(int(seed))
        draws[i] = rng.normal(0.0, s_log_dist)

    return pd.Series(draws, index=ids, name='person_dist_draw')


# Safety-skim exclusion: set SAFETY_EXPERIMENT_MODE=nosafety to drop crash params
_SAFETY_MODE_PARAMS = {'c_crash_veh'}   # mode model crash params
_SAFETY_DEST_PARAMS = {'c_crash'}       # dest model crash params

# ── Two-stage destination choice ──────────────────────────────────────────
STAGE5_DEST_TWO_STAGE = os.environ.get('STAGE5_DEST_TWO_STAGE', '0').strip() == '1'


def _load_zone_sa3_map() -> dict:
    """
    Load SA3_CODE → list[ORDER] mapping from the zone lookup CSV.

    Returns a dict {sa3_code: [zone_order_1, zone_order_2, ...]} for all
    371 zones. Also returns the reverse mapping {zone_order: sa3_code}.
    """
    zone_csv = os.path.join(os.path.dirname(__file__), 'seq_zone_ids_.csv')
    if not os.path.exists(zone_csv):
        zone_csv = os.path.join(os.path.dirname(__file__), 'seq_zone_ids.csv')

    try:
        zdf = pd.read_csv(zone_csv)
        zdf = zdf[zdf['ORDER'].notna()]
        zdf['ORDER'] = zdf['ORDER'].astype(int)
        zdf['SA3_CODE16'] = zdf['SA3_CODE16'].astype(int)

        sa3_to_sa2 = {}
        zone_to_sa3 = {}
        for _, row in zdf.iterrows():
            sa3 = int(row['SA3_CODE16'])
            order = int(row['ORDER'])
            sa3_to_sa2.setdefault(sa3, []).append(order)
            zone_to_sa3[order] = sa3

        sa3_list = sorted(sa3_to_sa2.keys())
        logger.info(
            f"_load_zone_sa3_map: {len(sa3_list)} SA3 districts, "
            f"{len(zone_to_sa3)} zones mapped"
        )
        return sa3_to_sa2, zone_to_sa3

    except Exception as e:
        logger.error(f"_load_zone_sa3_map failed: {e}")
        return {}, {}


def _build_sa3_aggregates(landuse: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SA2-level landuse columns to SA3 district level.

    Uses sum (for counts/areas) and mean (for rates/indices).
    Returns a DataFrame indexed by SA3_CODE with aggregated columns.
    """
    sa3_to_sa2, zone_to_sa3 = _load_zone_sa3_map()
    if not sa3_to_sa2:
        return pd.DataFrame()

    lu = landuse.copy()
    zone_col = 'ORDER' if 'ORDER' in lu.columns else lu.columns[0]
    if lu.index.name == zone_col:
        lu = lu.reset_index()

    lu['SA3'] = lu[zone_col].map(zone_to_sa3)
    lu = lu[lu['SA3'].notna()].copy()
    lu['SA3'] = lu['SA3'].astype(int)

    sum_cols = ['TOTEMP', 'FPSEMPN', 'HEREMPN', 'RETEMPN', 'TOTPOP', 'TOTHH',
                'Education', 'Primary', 'Secondary', 'HSENROLL',
                'residential', 'Entertainment', 'commercial', 'retail', 'EATOUT',
                'Primary_total_priv', 'Secondary_total_priv', 'AGE0519']
    mean_cols = ['walk_acc', 'transit_acc', 'auto_work_acc', 'auto_shop_acc',
                 'auto_edu_acc', 'auto_nonmand_acc',
                 'acdc_aggl', 'acdc_comp',
                 'acdc_aggl_retail', 'acdc_aggl_office',
                 'acdc_aggl_edu', 'acdc_aggl_leisure',
                 'acdc_aggl_parks', 'acdc_aggl_health',
                 'acdc_aggl_food', 'acdc_aggl_civic',
                 'size_term', 'is_cbd', 'transit_walk_acc']

    sum_cols = [c for c in sum_cols if c in lu.columns]
    mean_cols = [c for c in mean_cols if c in lu.columns]

    sa3_sum = lu.groupby('SA3')[sum_cols].sum()
    sa3_mean = lu.groupby('SA3')[mean_cols].mean()

    sa3_lu = sa3_sum.join(sa3_mean, how='outer')
    sa3_lu.index.name = 'SA3_CODE'
    sa3_lu = sa3_lu.reset_index()

    sa3_lu['SA3_CODE'] = sa3_lu['SA3_CODE'].astype(int)

    for col in mean_cols:
        if col not in sa3_lu.columns:
            sa3_lu[col] = 0.0

    logger.info(
        f"_build_sa3_aggregates: {len(sa3_lu)} SA3 districts, "
        f"{len(sum_cols)} sum cols, {len(mean_cols)} mean cols"
    )
    return sa3_lu


def _run_two_stage_dest(
    segment: str, tours_df: pd.DataFrame, persons: pd.DataFrame,
    landuse: pd.DataFrame, skims, skim_df: pd.DataFrame,
    logsums, lu_with_st: pd.DataFrame, seed: int,
    dest_feature_override: list,
) -> dict:
    """
    Two-stage destination choice: upper SA3 district → lower SA2 zone.

    Stage 1 (lower): MNL destination choice at SA2 level within each SA3.
    Stage 2 (upper): MNL SA3 district choice using logsum from lower model.

    Prediction: P(zone j) = P(sa3 containing j) * P(j | sa3).

    Shadow prices remain at SA2 level.
    """
    from larch import P as _lP, X as _lX

    sa3_to_sa2, zone_to_sa3 = _load_zone_sa3_map()
    if not sa3_to_sa2:
        logger.warning("_run_two_stage_dest: no SA3 map; falling back to single-stage")
        return None

    sa3_lu = _build_sa3_aggregates(landuse)
    sa3_list = sorted(sa3_to_sa2.keys())
    sa3_idx = {s: i for i, s in enumerate(sa3_list)}

    zone_ids = sorted(zone_to_sa3.keys())

    # ── STAGE 1: Lower SA2 model (within SA3) ─────────────────────────────
    from stage5_destination_choice import _dest_tree, _build_dest_model, _estimate, _dest_spec_custom

    lower_results = {}
    for sa3, sa2_zones in sa3_to_sa2.items():
        sa2_set = set(sa2_zones)
        seg_sa3 = tours_df[tours_df['DTAZ'].isin(sa2_set)].copy()
        if len(seg_sa3) < 10:
            lower_results[sa3] = None
            continue

        sa3_mask = lu_with_st.set_index('ORDER' if 'ORDER' in lu_with_st.columns
                                         else lu_with_st.columns[0]).index.isin(sa2_set)
        lu_sa3 = lu_with_st.loc[lu_with_st.index.isin(sa2_set)]

        dt_lower = _dest_tree(
            seg_sa3, persons, lu_sa3, skims, logsums,
            sampled_alts=None, skim_df=skim_df, segment=segment,
        )
        lower_mdl = _build_dest_model(dt_lower, segment,
                                       custom_dest_vars=dest_feature_override)
        lower_mdl = _estimate(lower_mdl, two_phase=False)
        lower_results[sa3] = lower_mdl
        print(f"  [two-stage:{segment}] SA3 {sa3}: {len(sa2_zones)} zones, "
              f"n_obs={len(seg_sa3)}, LL={getattr(lower_mdl, '_cached_ll', '?')}")

    # ── STAGE 2: Upper SA3 district model ─────────────────────────────────
    sa3_tours = tours_df.copy()
    sa3_tours['SA3_chosen'] = sa3_tours['DTAZ'].map(zone_to_sa3)

    sa3_ca_rows = []
    for _, tour_row in sa3_tours.iterrows():
        pid = str(tour_row.get('ptour_id', tour_row.name))
        chosen_sa3 = int(tour_row['SA3_chosen'])
        for sa3_cand in sa3_list:
            logsum = 0.0
            if lower_results.get(sa3_cand) is not None:
                try:
                    logsum = float(lower_results[sa3_cand]._cached_ll) / max(
                        len(sa3_to_sa2[sa3_cand]), 1)
                except Exception:
                    pass
            sa3_ca_rows.append({
                'ptour_id': pid,
                '_altid_': sa3_cand,
                'sa3_logsum': logsum,
                'chosen': 1 if sa3_cand == chosen_sa3 else 0,
            })

    sa3_ca_df = pd.DataFrame(sa3_ca_rows)
    for col in ['size_term', 'TOTEMP', 'acdc_aggl']:
        if col in sa3_lu.columns:
            sa3_ca_df[col] = sa3_ca_df['_altid_'].map(
                sa3_lu.set_index('SA3_CODE')[col].to_dict()
            ).fillna(0.0).astype(np.float32)

    sa3_ca_df['log_DIST_1'] = np.float32(0.0)
    sa3_ca_df['DIST'] = np.float32(0.0)
    sa3_ca_df['logsum'] = np.float32(0.0)
    sa3_ca_df['log_CRASH_1'] = np.float32(0.0)

    dt_upper = lx.Dataset.construct.from_idca(
        sa3_ca_df.set_index(['ptour_id', '_altid_']).sort_index()
    )

    upper_mdl = lx.Model(dt_upper, compute_engine='numba')
    upper_mdl.choice_co_code = '_altid_'

    upper_spec = {
        'c_sa3_size': ('size_term', None),
        'c_sa3_logsum': ('sa3_logsum', None),
    }
    for coeff, (expr, _) in upper_spec.items():
        upper_mdl.utility_co = _lP(coeff) * _lX(expr)

    upper_mdl = _estimate(upper_mdl, two_phase=False)

    return {
        'lower_models': lower_results,
        'upper_model': upper_mdl,
        'sa3_to_sa2': sa3_to_sa2,
        'zone_to_sa3': zone_to_sa3,
    }


def _run_segment(segment: str, tours: pd.DataFrame,
                 persons: pd.DataFrame, landuse: pd.DataFrame,
                 skims, skim_df: pd.DataFrame = None,
                 seed: int = 42, verbose: bool = True,
                 mode_feature_override: list = None,
                 dest_feature_override: list = None,
                 coef_warm_start: dict = None,
                 crash_skims=None) -> dict:
    """
    Estimate mode + destination models for one segment.

    ActivitySim pattern (tour_mode_choice_simulate):
      choices_list = []
      for tour_purpose, tours_segment in tours.groupby("tour_purpose"):
          choices = run_tour_mode_choice_simulate(tours_segment, tour_purpose, ...)
          choices_list.append(choices)
      choices_df = pd.concat(choices_list)

    Here we do the equivalent for HHTS estimation:
      1. Filter tours to segment (groupby equivalent)
      2. Merge OD skim values onto the segment table for the mixed logit
      3. Estimate mode model with segment-specific coefficient inits
      4. Compute logsums once (ActivitySim: compute_logsums in trip_destination)
      5. Estimate destination model (logsum as ca variable)
    """
    print(f"\n  ── Segment: {segment} ──────────────────────────────")
    print(f"  [{segment}] INPUT: {len(tours)} tours, {len(persons)} persons, {len(landuse)} landuse zones")
    if 'OTAZ' in tours.columns and 'DTAZ' in tours.columns:
        print(f"  [{segment}] OTAZ range: {tours['OTAZ'].min()}-{tours['OTAZ'].max()}, "
              f"DTAZ range: {tours['DTAZ'].min()}-{tours['DTAZ'].max()}, "
              f"unique OTAZ: {tours['OTAZ'].nunique()}, unique DTAZ: {tours['DTAZ'].nunique()}")
    if 'weight' in tours.columns:
        _w = tours['weight']
        print(f"  [{segment}] tour weights: mean={_w.mean():.3f} std={_w.std():.3f} "
              f"min={_w.min():.3f} max={_w.max():.3f} - NOTE: dest model uses weight=1")

    # Exclude safety skims when SAFETY_EXPERIMENT_MODE=nosafety
    _nosafety = os.environ.get('SAFETY_EXPERIMENT_MODE', 'baseline') == 'nosafety'
    _mode_uec = {k: v for k, v in MODE_UEC.items()
                 if not (_nosafety and k in _SAFETY_MODE_PARAMS)}
    _dest_uec = {k: v for k, v in DEST_UEC.items()
                 if not (_nosafety and k in _SAFETY_DEST_PARAMS)}

    # Defensive cleanup: GA-selected override features can still include crash terms.
    # In nosafety mode, strip them from overrides before any model build.
    _mode_feature_override = mode_feature_override
    _dest_feature_override = dest_feature_override
    if _nosafety:
        if isinstance(_mode_feature_override, list):
            _mode_feature_override = [k for k in _mode_feature_override if k not in _SAFETY_MODE_PARAMS]
        elif isinstance(_mode_feature_override, dict):
            _mode_feature_override = {
                k: v for k, v in _mode_feature_override.items() if k not in _SAFETY_MODE_PARAMS
            }

        if isinstance(_dest_feature_override, list):
            _dest_feature_override = [k for k in _dest_feature_override if k not in _SAFETY_DEST_PARAMS]
        elif isinstance(_dest_feature_override, dict):
            _dest_feature_override = {
                k: v for k, v in _dest_feature_override.items() if k not in _SAFETY_DEST_PARAMS
            }

    # In nosafety mode do not pass crash skims or CRASH skim column to _dest_tree.
    _dest_crash_skims = None if _nosafety else crash_skims
    _dest_skim_df = skim_df
    if _nosafety and _dest_skim_df is not None and 'CRASH' in _dest_skim_df.columns:
        _dest_skim_df = _dest_skim_df.drop(columns=['CRASH'])

    if _nosafety:
        logger.info(f"  [{segment}] nosafety mode - excluded crash params")

    seg = _filter_seg(tours, segment)

    # ── Zone 330 exclusion for edu segment ────────────────────────────────
    # Zone 330 is a peripheral/external zone with all-zero education data
    # (Primary=0, Secondary=0, Education=NaN, HSENROLL=0).  The ~312 observed
    # HHTS tours with DTAZ=330 are data artefacts (zone assignment errors or
    # trips to external areas misrouted to zone 330).  Including them forces the
    # model to explain travel to a zone with no schools, which drives edu
    # attraction coefficients negative and catastrophically inflates the LL.
    if segment == 'edu':
        _dtaz_col = next((c for c in ['DTAZ', 'dtaz'] if c in seg.columns), None)
        if _dtaz_col:
            _z330_mask = pd.to_numeric(seg[_dtaz_col], errors='coerce').fillna(0).astype(int) == 330
            if _z330_mask.any():
                n_z330 = int(_z330_mask.sum())
                seg = seg[~_z330_mask].copy()
                logger.warning(
                    f"_run_segment [edu]: excluded {n_z330} tours with DTAZ=330 "
                    "(zone 330 has no education data - likely zone-assignment artefact). "
                    f"{len(seg)} edu tours remain for estimation."
                )

    MIN_TOURS = int(os.environ.get('STAGE5_MIN_TOURS', '30'))
    print(f"  [{segment}] filtered segment: {len(seg)} tours "
          f"(need ≥ {MIN_TOURS} to estimate)")
    if len(seg) < MIN_TOURS:
        print(f"  WARNING: only {len(seg)} tours for segment='{segment}' - skipping.")
        print(f"           Set STAGE5_MIN_TOURS={max(len(seg)//2, 10)} env var to lower threshold.")
        if len(seg) > 0:
            print(f"           Purpose column values: {seg[next((c for c in ['primary_purpose','main_purpose','purpose'] if c in seg.columns), seg.columns[0])].value_counts().head(10).to_dict()}")
        return {}

    # ── merge OD skim values onto seg for the mixed logit ─────────────────
    # _estimate_mixed_logit needs per-observation IVT/distance values.
    # The HHTS tour table has OTAZ/DTAZ but no skim columns.  We merge the
    # key columns here so the MXL baseline gets real variation to estimate.
    _MERGE_COLS = ['SOVTOLL_TIME__AM', 'SOVTOLL_DIST__AM',
                   'DISTWALK', 'DISTBIKE', 'WLK_LOC_WLK_TOTIVT__AM']
    if skim_df is not None and not skim_df.empty:
        avail_merge = [c for c in _MERGE_COLS if c in skim_df.columns]
        if avail_merge:
            sk_sub = (skim_df[['OTAZ', 'DTAZ'] + avail_merge]
                      .drop_duplicates(['OTAZ', 'DTAZ']))
            # drop any existing skim cols to avoid _x/_y suffixes
            seg = seg.drop(columns=[c for c in avail_merge if c in seg.columns],
                           errors='ignore')
            seg = seg.merge(sk_sub, on=['OTAZ', 'DTAZ'], how='left')
            # fill missing (zone not in skim) with plausible defaults
            for col, default in [('SOVTOLL_TIME__AM', 30.0),
                                  ('SOVTOLL_DIST__AM', 20.0),
                                  ('DISTWALK', 20.0), ('DISTBIKE', 20.0),
                                  ('WLK_LOC_WLK_TOTIVT__AM', 45.0)]:
                if col in seg.columns:
                    seg[col] = pd.to_numeric(seg[col], errors='coerce').fillna(default)
            logger.info(
                f"_run_segment [{segment}]: merged {len(avail_merge)} skim cols; "
                f"IVT median={seg['SOVTOLL_TIME__AM'].median():.1f} min")

            # ── recompute walk/bike availability from actual distances ────
            # Use DISTWALK/DISTBIKE directly with realistic zone-level thresholds.
            # Estimation availability must match prediction thresholds (WALK_MAX_KM=6.0,
            # BIKE_MAX_KM=25.0 in _predict_batch) so that frozen probability cells
            # computed from HHTS data transfer correctly to the synthetic population.
            # Mismatched thresholds suppress walk/bike in HHTS estimation while
            # allowing them in prediction, causing frozen probs to underweight
            # active modes.
            def _to_km(_s: pd.Series) -> pd.Series:
                _v = pd.to_numeric(_s, errors='coerce').fillna(0.0).clip(lower=0.0)
                try:
                    _p95 = float(_v.quantile(0.95))
                except Exception:
                    _p95 = float(_v.max()) if len(_v) else 0.0
                # Some skim builds store DISTWALK/DISTBIKE in meters.
                return _v / 1000.0 if _p95 > 500.0 else _v

            if 'DISTWALK' in seg.columns:
                intra_mask = seg['OTAZ'] == seg['DTAZ']
                wd_km = _to_km(seg['DISTWALK'])
                seg['walk_avail'] = ((intra_mask) | (wd_km < WALK_MAX_KM)).astype(int)
            if 'DISTBIKE' in seg.columns:
                intra_mask = seg['OTAZ'] == seg['DTAZ']
                bd_km = _to_km(seg['DISTBIKE'])
                seg['bike_avail'] = ((intra_mask) | (bd_km < BIKE_MAX_KM)).astype(int)
            elif 'SOVTOLL_TIME__AM' in seg.columns:
                # IVT proxy if distance skims absent
                intra_mask = seg['OTAZ'] == seg['DTAZ']
                ivt = seg['SOVTOLL_TIME__AM']
                seg['walk_avail'] = ((intra_mask) | (ivt < 10.0)).astype(int)    # was 5.0
                seg['bike_avail'] = ((intra_mask) | (ivt < 20.0)).astype(int)    # was 12.0
            logger.info(
                f"_run_segment [{segment}]: walk_avail={seg['walk_avail'].sum()} "
                f"bike_avail={seg['bike_avail'].sum()} (of {len(seg)})"
            )
    # Guarantee walk_avail / bike_avail are always present in seg so that
    # avail_vars_hint always includes them regardless of whether skim_df had
    # DISTWALK/DISTBIKE.  Default to fully available (1) - the distance-based
    # computation above has already set tighter thresholds when data was present.
    if 'walk_avail' not in seg.columns:
        seg['walk_avail'] = 1
    if 'bike_avail' not in seg.columns:
        seg['bike_avail'] = 1

    # Guarantee chosen alternatives are marked available in estimation data.
    # This prevents chosen_but_not_available diagnostics and Inf likelihood.
    if 'tour_mode' in seg.columns:
        try:
            if 'walk_avail' in seg.columns:
                seg.loc[seg['tour_mode'] == MODE_MAP['Walk'], 'walk_avail'] = 1
            if 'bike_avail' in seg.columns:
                seg.loc[seg['tour_mode'] == MODE_MAP['Bike'], 'bike_avail'] = 1
            if 'CARLICENCE' in seg.columns:
                seg.loc[seg['tour_mode'] == MODE_MAP['DA'], 'CARLICENCE'] = 1
        except Exception:
            pass

    seg_coeffs = SEGMENT_COEFFICIENTS.get(segment, {})
    print(f"  {len(seg)} estimation tours | "
          f"segment coefficients: {list(seg_coeffs.keys())[:5]}"
          f"{'...' if len(seg_coeffs) > 5 else ''}")

    # ── Skim diagnostic: verify DIST and DISTWALK are genuinely different ──
    # If these are collinear (r > 0.99) or DISTWALK is all-constant, the
    # destination model cannot distinguish their coefficients → both converge
    # to the same value.  This diagnostic flags the problem early.
    if skim_df is not None and not skim_df.empty:
        _dist_ok = 'DIST' in skim_df.columns or 'SOVTOLL_DIST__AM' in skim_df.columns
        _dwalk_ok = 'DISTWALK' in skim_df.columns
        _d_col  = 'DIST' if 'DIST' in skim_df.columns else 'SOVTOLL_DIST__AM'
        if _dist_ok and _dwalk_ok:
            _d  = pd.to_numeric(skim_df[_d_col], errors='coerce').fillna(0.0)
            _dw = pd.to_numeric(skim_df['DISTWALK'], errors='coerce').fillna(0.0)
            _dw_nonzero = (_dw > 0).sum()
            _dw_std = float(_dw.std())
            _corr = float(_d.corr(_dw)) if _dw_std > 0 else 1.0
            _pct_30 = float((_dw >= 29.9).mean() * 100)
            print(f"  [skim check] DIST: mean={_d.mean():.2f} std={_d.std():.2f}  "
                  f"DISTWALK: mean={_dw.mean():.2f} std={_dw_std:.2f}  "
                  f"corr(DIST,DISTWALK)={_corr:.3f}  "
                  f"DISTWALK=30(fallback): {_pct_30:.1f}%  nonzero: {_dw_nonzero:,}")
            if _corr > 0.97:
                global _DIST_WALK_CORR_WARNED
                if segment not in _DIST_WALK_CORR_WARNED:
                    _DIST_WALK_CORR_WARNED.add(segment)
                    logger.info(
                        f"_run_segment [{segment}]: DIST vs DISTWALK correlation = {_corr:.3f} "
                        "(known OMX issue - same road-network data in both matrices). "
                        "DISTWALK terms holdfasted; walk accessibility captured via mode logsum."
                    )
            if _pct_30 > 70:
                logger.warning(
                    f"_run_segment [{segment}]: {_pct_30:.0f}% of DISTWALK values are ≥30 km "
                    "(fallback fill). Walk-distance term c_log_distwalk will be near-constant "
                    "for most OD pairs → coefficient degenerate. Walk accessibility is "
                    "captured via mode logsum; c_log_distwalk only meaningful for walkable "
                    "zone pairs (DISTWALK < 6 km, walk_feasible_ij=1)."
                )
        elif not _dwalk_ok:
            logger.warning(
                f"_run_segment [{segment}]: DISTWALK not found in skim_df columns "
                f"({list(skim_df.columns[:10])}). Walk-distance destination terms will "
                "use 30 km default - holdfast c_log_distwalk to avoid degenerate estimates."
            )

    # Ensure ptour_id is unique before building DataTree.
    # The merge with skim_df can introduce duplicate rows if OTAZ/DTAZ pairs
    # repeat; Larch's from_idca raises "cannot convert non-unique MultiIndex"
    # when case_ids are not unique.  Reset with a guaranteed-unique id.
    if seg['ptour_id'].duplicated().any():
        n_dup = seg['ptour_id'].duplicated().sum()
        logger.warning(f"_run_segment [{segment}]: {n_dup} duplicate ptour_ids - "
                       "resetting to row-index-based unique ids")
        seg = seg.copy()
        seg['ptour_id'] = (seg['person_id'].astype(str) + '_'
                           + np.arange(len(seg)).astype(str))

    dt_m     = _mode_tree(seg, persons, landuse, skims, skim_df=skim_df)
    mode_mdl = _build_mode_model(
        dt_m,
        segment,
        custom_mode_vars=(mode_feature_override
                           if mode_feature_override is not None
                           else (_mode_uec if _nosafety else None)),
        avail_vars_hint=(list(seg.columns) + list(persons.columns)),
    )
    # Repair chosen_but_not_available at model-build time so Larch doesn't log
    # the warning before _estimate has a chance to run its own doctor() call.
    try:
        mode_mdl.doctor(repair_ch_av="+")
    except Exception:
        pass

    # ── Inject warm-start coefficients (from GA / previous run) ──────────
    # coef_warm_start is a flat dict {param_name: value} covering mode + dest params.
    # We apply any mode-model params here so estimation starts from a better point.
    if coef_warm_start:
        _n_mode_warm = 0
        _mode_pf_idx = set(mode_mdl.pf.index.tolist())
        for _wname, _wval in coef_warm_start.items():
            if _wname in _mode_pf_idx:
                try:
                    mode_mdl.set_value(_wname, float(_wval))
                    _n_mode_warm += 1
                except Exception as _we:
                    logger.debug(f"warm-start mode set_value({_wname}) failed: {_we}")
        if _n_mode_warm:
            logger.info(f"_run_segment [{segment}]: mode warm-start applied "
                        f"{_n_mode_warm}/{len(coef_warm_start)} params")

    print("  Estimating mode model (nested logit)...")
    # Pre-fit: align ASC starting values with observed shares and reset any
    # Mu parameter that is at or near its boundary to a safe interior value.
    # This prevents the gradient from pulling Mu→1 during early iterations
    # before the behavioural parameters have a chance to converge.
    mode_mdl = _prefit_mode_model(mode_mdl, seg, persons, skims, segment)
    mode_mdl = _estimate(mode_mdl)

    # ── Generic-vs-alternative-specific refit substitution ───────────────────
    # When a per-mode ASV block is collinear or jointly weak (e.g. the four
    # income slopes converge to ≈ the same value), try replacing it with ONE
    # mode-generic coefficient applying the same effect to every alternative,
    # and keep whichever specification has the better BIC. Only on the GA path
    # (selected var list known) so the generic genes route through the generic
    # gating in _mode_spec_custom.
    _mode_base_vars = (mode_feature_override
                       if isinstance(mode_feature_override, list) else None)
    if _mode_base_vars is not None:
        def _rebuild_mode_generic(_genes, _base=_mode_base_vars):
            _mvars = list(dict.fromkeys(list(_base) + list(_genes)))
            _mm = _build_mode_model(
                dt_m, segment, custom_mode_vars=_mvars,
                avail_vars_hint=(list(seg.columns) + list(persons.columns)))
            try:
                _mm.doctor(repair_ch_av="+")
            except Exception:
                pass
            _mm = _prefit_mode_model(_mm, seg, persons, skims, segment)
            return _estimate(_mm)
        mode_mdl = _collapse_generic_groups(
            mode_mdl, 'stage5_mode', _rebuild_mode_generic, label=segment)

    # ── Prune zero-fit / statistically-insignificant terms and refit ─────────
    # Removes the dead pval≈1 dummies (e.g. leaked purpose interactions) and any
    # term the data cannot identify, re-estimating after each removal. Runs
    # before the ASC share-calibration below so calibration acts on the final
    # (pruned) structure.
    mode_mdl = _prune_and_refit(mode_mdl, _estimate, 'stage5_mode', label=segment)
    mode_mdl = _apply_sign_constraints(mode_mdl, 'stage5_mode')

    # Enforce theoretically-correct bike-time sign post-estimation.
    # If estimation lands slightly > 0, pin to a small negative value so
    # downstream simulation does not reward longer bike travel times.
    try:
        if 'c_bike_time' in mode_mdl.pf.index:
            _cb = float(mode_mdl.pf.loc['c_bike_time', 'value'])
            if np.isfinite(_cb) and _cb > 0:
                _fixed_cb = -max(abs(_cb), 0.01)
                # Use set_value to ensure the change persists (pf.loc may
                # modify a copy in some Larch versions)
                try:
                    mode_mdl.set_value('c_bike_time', _fixed_cb)
                except Exception:
                    mode_mdl.pf.loc['c_bike_time', 'value'] = _fixed_cb
                logger.warning(
                    f"_run_segment [{segment}]: c_bike_time estimated positive ({_cb:.4f}); "
                    f"reset to {_fixed_cb:.4f} for behavioural consistency"
                )
    except Exception:
        pass

    # ── Post-estimation: nest scale (Mu) and collinear income parameters ──────
    #
    # (A) Nest scale Mu: the nest scale parameters (Mu_Car / Mu_Motor /
    #     Mu_NonMotor) are kept FREE and are never holdfasted. Previously, when a
    #     Mu landed at a bound (≈1.0 = MNL limit, or ≈0.10) the code locked it at
    #     that boundary value and re-estimated "to fix the Hessian". That hid the
    #     nest as an estimated quantity and reported a censored value as if it
    #     were structural. A Mu sitting at a bound is a legitimate (if
    #     near-degenerate) estimate; it is reported honestly and its standard
    #     error comes from the OPG/proxy fallback in _estimate when the Hessian
    #     row is flat. No holdfast, no forced re-estimation from the Mu check.
    #
    # (B) Income collinearity: if c_incp_da/sr/transit/active all converge to
    #     nearly identical values (spread < 30 % of |mean|), they form a
    #     near-rank-deficient block in the Hessian. Fix: holdfast all except
    #     c_incp_da to the mean and re-estimate with one generic income param.
    _mode_reestimate_needed = False
    try:
        for _mu_p in ('Mu_Car', 'Mu_Motor', 'Mu_NonMotor'):
            if _mu_p in mode_mdl.pf.index:
                _is_hf = bool(mode_mdl.pf.loc[_mu_p].get('holdfast', False))
                _mu_v  = float(mode_mdl.pf.loc[_mu_p, 'value'])
                if not _is_hf and np.isfinite(_mu_v) and (_mu_v >= 0.99 or _mu_v <= 0.011):
                    logger.info(
                        f"[{segment}] {_mu_p}={_mu_v:.4f} sits at its bound "
                        "(nest near-degenerate to MNL) - kept FREE, not holdfasted"
                    )
    except Exception as _mu_e:
        logger.debug(f"[{segment}] Mu boundary check failed: {_mu_e}")

    try:
        _incp_pnames = ['c_incp_da', 'c_incp_sr', 'c_incp_transit', 'c_incp_active']
        _incp_free = {}
        for _p in _incp_pnames:
            if _p in mode_mdl.pf.index:
                _is_hf = bool(mode_mdl.pf.loc[_p].get('holdfast', False))
                _v = float(mode_mdl.pf.loc[_p, 'value'])
                if not _is_hf and np.isfinite(_v):
                    _incp_free[_p] = _v
        if len(_incp_free) >= 3:
            _vals = list(_incp_free.values())
            _mean_v = float(np.mean(_vals))
            _spread = float(max(_vals) - min(_vals))
            # Collinear: all params within 30% of mean (and mean non-trivial)
            _collinear = abs(_mean_v) > 0.05 and _spread < 0.30 * abs(_mean_v)
            if _collinear:
                _keep_free = 'c_incp_da' if 'c_incp_da' in _incp_free else list(_incp_free)[0]
                for _p, _v in _incp_free.items():
                    if _p != _keep_free:
                        try:
                            mode_mdl.lock_value(_p, _mean_v)
                        except Exception:
                            pass
                logger.info(
                    f"[{segment}] Income params collinear (spread={_spread:.4f} "
                    f"< 30% × |mean|={abs(_mean_v):.4f}) → holdfasting all at "
                    f"{_mean_v:.4f} except {_keep_free}; re-estimating"
                )
                _mode_reestimate_needed = True
    except Exception as _ie:
        logger.debug(f"[{segment}] Income collinearity check failed: {_ie}")

    if _mode_reestimate_needed:
        print(f"  [{segment}] Re-estimating mode model after fixing degenerate parameters...")
        mode_mdl = _estimate(mode_mdl)

    # ── Post-estimation: ASC calibration for SR and Walk ─────────────────────
    # Iteratively adjust mode ASCs so that predicted shares match observed shares
    # in the estimation data. Uses the log-share correction:
    #   ASC_mode += log(obs_share / pred_share)
    # This does NOT change the model's fit (LL stays the same) but ensures that
    # the aggregate predicted shares exactly reproduce the observed proportions.
    # Up to 8 iterations; convergence when all |delta| < 0.002.
    # We calibrate SR, Walk, Bike, and Transit (DA is reference / always implied).
    try:
        _asc_mode_map = {
            'ASC_SR':      MODE_MAP['SR'],
            'ASC_Walk':    MODE_MAP['Walk'],
            'ASC_Bike':    MODE_MAP['Bike'],
            'ASC_Transit': MODE_MAP['Transit'],
        }
        _mode_col = 'tour_mode' if 'tour_mode' in seg.columns else None
        if _mode_col is not None and len(seg) > 0:
            _obs_total = float(len(seg))
            _obs_counts = seg[_mode_col].value_counts()
            _obs_shares = {int(k): float(v) / _obs_total for k, v in _obs_counts.items()}

            _col_to_code = {0: MODE_MAP['DA'], 1: MODE_MAP['SR'],
                            2: MODE_MAP['Walk'], 3: MODE_MAP['Bike'],
                            4: MODE_MAP['Transit']}

            for _calib_iter in range(8):
                try:
                    # Use _mode_probs_numpy with coef_override=None so it
                    # reads fresh values from _final_pf / pf after set_value().
                    _probs = _mode_probs_numpy(mode_mdl, seg, persons, skims,
                                               coef_override=None)
                    if _probs is None or _probs.shape[0] == 0:
                        break
                    _pred_shares_arr = np.asarray(_probs).mean(axis=0)  # (5,)
                    _pred_dict = {_col_to_code[ci]: float(_pred_shares_arr[ci])
                                  for ci in range(len(_pred_shares_arr))}

                    _any_delta = False
                    for _asc_p, _mc in _asc_mode_map.items():
                        if _asc_p not in mode_mdl.pf.index:
                            continue
                        _is_hf = bool(mode_mdl.pf.loc[_asc_p].get('holdfast', False))
                        if _is_hf:
                            continue
                        _obs_s  = _obs_shares.get(_mc, 0.0)
                        _pred_s = _pred_dict.get(_mc, 0.0)
                        if _obs_s > 0.001 and _pred_s > 0.001:
                            _delta = float(np.log((_obs_s + 1e-8) / (_pred_s + 1e-8)))
                            if abs(_delta) > 0.002:
                                _cur = float(mode_mdl.pf.loc[_asc_p, 'value'])
                                _new_val = _cur + _delta
                                try:
                                    mode_mdl.set_value(_asc_p, _new_val)
                                except Exception:
                                    mode_mdl.pf.loc[_asc_p, 'value'] = _new_val
                                # Sync _final_pf so coef cache stays consistent
                                if (hasattr(mode_mdl, '_final_pf') and
                                        isinstance(mode_mdl._final_pf, pd.DataFrame) and
                                        _asc_p in mode_mdl._final_pf.index):
                                    mode_mdl._final_pf.loc[_asc_p, 'value'] = _new_val
                                _any_delta = True
                    # Invalidate coef cache so next iteration reads fresh values
                    mode_mdl._coef_cache = {}
                    if not _any_delta:
                        break
                except Exception as _ci_e:
                    logger.debug(f"[{segment}] ASC calibration iter {_calib_iter}: {_ci_e}")
                    break

            # Log final calibrated shares
            try:
                _cal_probs = _mode_probs_numpy(mode_mdl, seg, persons, skims,
                                               coef_override=None)
                if _cal_probs is not None:
                    _cal_arr = np.asarray(_cal_probs).mean(axis=0)
                    _cal_pred = {_col_to_code[ci]: _cal_arr[ci] for ci in range(len(_cal_arr))}
                    _sr_obs = _obs_shares.get(MODE_MAP['SR'], 0.0)
                    _wk_obs = _obs_shares.get(MODE_MAP['Walk'], 0.0)
                    logger.info(
                        f"[{segment}] Mode ASC calibration complete: "
                        f"SR obs={_sr_obs:.3f} pred={_cal_pred.get(MODE_MAP['SR'],0):.3f} | "
                        f"Walk obs={_wk_obs:.3f} pred={_cal_pred.get(MODE_MAP['Walk'],0):.3f}"
                    )
            except Exception:
                pass
    except Exception as _asc_cal_e:
        logger.debug(f"[{segment}] ASC calibration failed: {_asc_cal_e}")

    _print_model_table(mode_mdl, f"{segment.upper()} mode model (NL)")

    # ── Post-estimation validity check ─────────────────────────────────
    # Checks rho², key parameter signs, and mode share alignment.
    # Failures are logged as warnings so the pipeline continues but the
    # output log makes the problem visible for debugging.
    try:
        _pf_check = mode_mdl.pf
        _ll_check = getattr(mode_mdl, '_cached_ll', None)
        _rho2_check = _rho_sq_null(mode_mdl, _ll_check)

        _validity_warnings = []

        if not np.isnan(_rho2_check) and _rho2_check < 0.05:
            _validity_warnings.append(
                f"rho²={_rho2_check:.3f} - model barely improves on market shares; "
                "check estimation data quality and sample size"
            )

        # Sign checks on key utility parameters
        _sign_checks = {
            'c_ivt':       ('should be < 0 - auto IVT increases deterrence', lambda v: v > 0),
            'c_walk':      ('should be < 0 - walk time increases deterrence', lambda v: v > 0),
            'c_bike':      ('should be < 0 - bike time increases deterrence', lambda v: v > 0),
            'c_walk_time': ('should be < 0 - walk time increases deterrence', lambda v: v > 0),
            'c_bike_time': ('should be < 0 - bike time increases deterrence', lambda v: v > 0),
            'c_log_dist':  ('should be < 0 - distance deters destination choice', lambda v: v > 0.01),
        }
        for _pname, (_msg, _bad) in _sign_checks.items():
            if _pname in _pf_check.index:
                _pval = float(_pf_check.loc[_pname, 'value'])
                if not np.isnan(_pval) and _bad(_pval):
                    _validity_warnings.append(
                        f"param '{_pname}'={_pval:.4f} has unexpected sign - {_msg}"
                    )

        # Active-mode share check: if DA > 95% after estimation something is wrong
        try:
            from stage5_destination_choice import _mode_probs_numpy as _mpn_check
            _mp_check = _mode_probs_numpy(mode_mdl, seg, persons, skims,
                                           coef_override=None)
            if _mp_check is not None:
                _da_share = float((_mp_check[:, 0]).mean())  # DA = col 0
                _active   = float((_mp_check[:, 2:4]).sum(axis=1).mean())  # Walk+Bike cols 2,3
                if _da_share > 0.95:
                    _validity_warnings.append(
                        f"DA share={_da_share:.1%} on HHTS sample - "
                        "walk/bike/transit may be suppressed; check ASC defaults and availability flags"
                    )
                if _active < 0.02:
                    _validity_warnings.append(
                        f"Walk+Bike share={_active:.1%} on HHTS - active modes near-zero; "
                        "verify DISTWALK/DISTBIKE skim values and walk_avail/bike_avail counts above"
                    )
        except Exception as _share_e:
            logger.debug(f"Mode share validity check failed: {_share_e}")

        if _validity_warnings:
            for _w in _validity_warnings:
                logger.warning(f"[{segment}] Mode model validity: {_w}")
        else:
            logger.info(f"[{segment}] Mode model validity: OK (rho²={_rho2_check:.3f})")
    except Exception as _val_e:
        logger.debug(f"[{segment}] Post-estimation validity check failed: {_val_e}")

    # ── Mixed logit on mode (optional; default disabled for stability) ─
    mxl_result, mxl_model = (None, None)
    if STAGE5_ENABLE_MXL:
        mxl_result, mxl_model = _estimate_mixed_logit(seg, segment, n_draws=500)
    else:
        logger.info(f"[{segment}] STAGE5_ENABLE_MXL=0 - skipping mixed logit and using nested logit")

    # Use mixed logit only if it produced a valid coefficient cache.
    _mxl_valid = (
        mxl_result is not None and
        mxl_model is not None and
        bool(getattr(mxl_model, "_coef_cache", {}))
    )
    if _mxl_valid:
        mode_mdl = mxl_model
        logger.info(f"[{segment}] Using mixed logit for mode choice (random IVT)")
    else:
        logger.info(f"[{segment}] Mixed logit unavailable/invalid - using nested logit for mode choice")

    # Extract mode coefficients NOW, before computing logsums, so they can be
    # passed as coef_override.  This ensures _logsums_numpy_fallback always has
    # a coefficient source even when model.pf is inaccessible after JAX JIT.
    # Priority: _coef_cache → _final_pf → pf (pf may return NaN after JAX JIT)
    def _extract_coef_early(m):
        try:
            if hasattr(m, '_coef_cache') and m._coef_cache:
                return dict(m._coef_cache)
            # _final_pf is a plain-DataFrame snapshot saved by _estimate()
            # - it is immune to JAX JIT corruption of the live pf property.
            if hasattr(m, '_final_pf') and isinstance(m._final_pf, pd.DataFrame):
                pf = m._final_pf
                if 'value' in pf.columns:
                    coef = {str(name): float(pf.loc[name, 'value'])
                            for name in pf.index
                            if not np.isnan(float(pf.loc[name, 'value']))}
                    if coef:
                        logger.debug(f"_extract_coef_early: found {len(coef)} params via _final_pf")
                        return coef
            pf = m.pf
            return {name: float(pf.loc[name, 'value'])
                    for name in pf.index
                    if not np.isnan(float(pf.loc[name, 'value']))}
        except Exception:
            return {}

    _mode_coef_early = _extract_coef_early(mode_mdl)
    if _mode_coef_early:
        # Inject into _coef_cache so _coef_dict_from_model finds them even
        # after Larch API calls inside _logsums may corrupt model.pf state.
        mode_mdl._coef_cache = _mode_coef_early
        logger.debug(f"[{segment}] Mode coef cache seeded with "
                     f"{len(_mode_coef_early)} params before logsum computation")
    else:
        logger.warning(f"[{segment}] _extract_coef_early: NO coefficients found "
                       f"in _coef_cache/_final_pf/pf - logsums will be zero. "
                       f"Check that _estimate() completed and saved _final_pf.")
    ls = _logsums(mode_mdl, dt_m, skims=skims, verbose=verbose,
                  coef_override=_mode_coef_early if _mode_coef_early else None)
    _ls_n_zones = len(ls.coords['zones'].values) if hasattr(ls, 'coords') and 'zones' in ls.coords else '?'
    _ls_n_cases = len(ls.coords['cases'].values) if hasattr(ls, 'coords') and 'cases' in ls.coords else '?'
    print(f"  [{segment}] Logsums built: {_ls_n_cases} cases × {_ls_n_zones} zones")

    # ── Rescale logsums to unit variance ─────────────────────────────
    # The raw logsum scale depends on the mode model's IVT coefficient and
    # the skim units.  When the scale is very different from the dest model
    # utility scale, c_logsum hits its bound (e.g. edu c_logsum=10.0 in
    # test runs).  Standardising to mean=0, std=1 puts all segments on the
    # same scale and lets the estimator find the true sensitivity.
    _ls_mean = 0.0
    _ls_std = 1.0
    try:
        import numpy as _np
        _ls_vals = ls.values.astype(float)
        _finite   = _ls_vals[_np.isfinite(_ls_vals)]
        if len(_finite) > 0 and _finite.std() > 0:
            _ls_mean = float(_finite.mean())
            _ls_std = float(_finite.std())
            _ls_vals = (_ls_vals - _ls_mean) / _ls_std
            ls = ls.copy(data=_ls_vals.astype(ls.values.dtype))
            logger.info(
                f"  Logsums rescaled to mean=0 std=1 for {segment} destination model "
                f"(mean={_ls_mean:.4f}, std={_ls_std:.4f})"
            )
    except Exception as _e:
        logger.debug(f"Logsum rescaling failed ({_e}) - using raw values")

    # ── attach size_term to landuse for this segment ─────────────────
    # sharrow requires plain column names - no leading underscore.
    # compute_size_terms returns a zone×purpose DataFrame;
    # we take the segment's canonical purpose (or mean across purposes)
    # and add it as 'size_term' to landuse.
    lu_with_st = _ensure_destination_clusters(_attach_size_term(landuse, segment))

    # ── ACDC spatial terms ────────────────────────────────────────────────
    # Agglomeration-Competition Destination Choice (ACDC) terms:
    #
    #   acdc_aggl  (zone-level, destination-side):
    #     = log( Σ_{k≠j} size[k] × exp(−λ × time[j,k]) )
    #     Measures clustering: how many attractive nearby zones surround j.
    #     Expected coefficient c_aggl > 0  (positive agglomeration effect).
    #
    #   acdc_comp  (zone-level, destination-side):
    #     = log( Σ_{k≠j} exp(−λ × time[j,k]) )   -- UNWEIGHTED by size, unlike aggl.
    #     Measures competition: how many other zones are reachable from j,
    #     i.e. substitution pressure/density around j.  Expected c_comp < 0.
    #     (Previously this was size-weighted with the same kernel as aggl,
    #     making the two ~0.9998 correlated so only one was ever identified -
    #     see compute_acdc_terms() docstring, fixed 2026-07.)
    #
    #   These are NOT separate sub-models - both are continuous covariates
    #   fed into the single destination MNL alongside distance, logsum etc.
    #   Both are computed in compute_acdc_terms() and attached to the emp
    #   (landuse) DataTree node so they vary across destination zones.
    #
    # skim_df was built before this call in run_stage5 and passed through.
    lu_with_st, seg_acdc = _attach_acdc_terms(
        lu_with_st, seg, skim_df if skim_df is not None else pd.DataFrame(),
        segment
    )

    # ── Merge accessibility measures from saved_data/accessibility.csv ───────
    # These columns (auto_work_acc, walk_acc, etc.) are NOT in the raw
    # zone_order_and_landuse.csv - they are pre-computed gravity integrals
    # saved by compute_accessibility.run_accessibility().
    # Without them, every accessibility term in DEST_UEC_FLAT is zero-variance
    # → coefficient estimated at exactly 0.0 → no zone-level accessibility
    # differentiation in the destination model.
    _ACC_COLS = ['auto_work_acc', 'auto_shop_acc', 'auto_edu_acc',
                 'auto_nonmand_acc', 'walk_acc', 'transit_acc']
    if not any(c in lu_with_st.columns for c in _ACC_COLS):
        _acc_path = os.path.join('saved_data', 'accessibility.csv')
        if os.path.exists(_acc_path):
            try:
                _acc_df = pd.read_csv(_acc_path)
                _acc_idx = 'zone_id' if 'zone_id' in _acc_df.columns else _acc_df.columns[0]
                _acc_df = _acc_df.set_index(_acc_idx)
                # lu_with_st may or may not have ORDER as a column vs. index
                _lu_zcol = 'ORDER' if 'ORDER' in lu_with_st.columns else None
                if _lu_zcol:
                    _idx = lu_with_st[_lu_zcol].astype(int)
                    for c in _ACC_COLS:
                        if c in _acc_df.columns:
                            lu_with_st[c] = _acc_df[c].reindex(_idx.values).values
                else:
                    for c in _ACC_COLS:
                        if c in _acc_df.columns:
                            lu_with_st[c] = _acc_df[c].reindex(lu_with_st.index.astype(int)).values
                _loaded = [c for c in _ACC_COLS if c in lu_with_st.columns and
                           pd.to_numeric(lu_with_st[c], errors='coerce').fillna(0).any()]
                logger.info(
                    f"_run_segment [{segment}]: merged {len(_loaded)} accessibility cols "
                    f"from {_acc_path}: {_loaded}"
                )
            except Exception as _acc_e:
                logger.warning(f"_run_segment [{segment}]: accessibility merge failed: {_acc_e}")
        else:
            logger.warning(
                f"_run_segment [{segment}]: accessibility file missing at {_acc_path} - "
                f"run compute_accessibility.run_accessibility() to pre-compute. "
                f"All accessibility terms will be zero in this run."
            )

    # Save ACDC terms so zone-level agglomeration/competition can be inspected.
    try:
        import os as _os
        _acdc_cols = [c for c in ['acdc_aggl', 'acdc_comp'] if c in lu_with_st.columns]
        if _acdc_cols:
            _acdc_out = lu_with_st[_acdc_cols].copy()
            _acdc_out.index.name = 'zone_id'
            _acdc_out['segment'] = segment
            _acdc_path = 'saved_data/acdc_terms.csv'
            _write_hdr = not _os.path.exists(_acdc_path)
            _acdc_out.to_csv(_acdc_path, mode='a', header=_write_hdr)
            logger.info(f"ACDC terms saved → {_acdc_path} (segment={segment})")
    except Exception as _acdc_save_e:
        logger.debug(f"ACDC save skipped: {_acdc_save_e}")

    # Propagate the deduplicated ptour_ids to seg_acdc so _dest_tree is consistent
    if 'ptour_id' in seg.columns and 'ptour_id' in seg_acdc.columns:
        seg_acdc = seg_acdc.copy()
        seg_acdc['ptour_id'] = seg['ptour_id'].values

    # ── Alternative set: all zones or importance-sampled subset ─────────
    # DEST_EST_SAMPLE_SIZE == 0  →  use ALL zones (no sampling, log_correction=0)
    # DEST_EST_SAMPLE_SIZE  > 0  →  Manski-Lerman importance sampling
    _zone_ids_for_is = (
        [int(z) for z in ls.coords['zones'].values]
        if hasattr(ls, 'coords') and 'zones' in ls.coords
        else _default_zone_universe()
    )
    _n_zone_ids = len(_zone_ids_for_is)
    _n_obs_seg  = seg_acdc['ptour_id'].nunique() if 'ptour_id' in seg_acdc.columns else len(seg_acdc)

    if DEST_EST_SAMPLE_SIZE > 0:
        try:
            _sampled_alts = _importance_sample_dest_alts(
                tours_df    = seg_acdc[['ptour_id', 'DTAZ']].copy(),
                landuse     = lu_with_st,
                zone_ids    = _zone_ids_for_is,
                sample_size = DEST_EST_SAMPLE_SIZE,
                seed        = seed,
            )
            logger.info(
                f"_run_segment [{segment}]: importance sampling "
                f"{len(_sampled_alts)} (case, alt) pairs from "
                f"{_n_obs_seg} observations"
            )
            print(f"  [{segment}] ESTIMATION: importance-sampled {DEST_EST_SAMPLE_SIZE}/{_n_zone_ids} zones "
                  f"× {_n_obs_seg} obs = {len(_sampled_alts)} rows")
        except Exception as _is_e:
            logger.warning(f"_run_segment [{segment}]: importance sampling failed ({_is_e}) "
                           "- falling back to full zone set")
            _sampled_alts = None
    else:
        _sampled_alts = None   # ALL zones - no sampling, log_correction=0
        logger.info(
            f"_run_segment [{segment}]: using ALL {_n_zone_ids} zones as alternatives "
            f"(DEST_EST_SAMPLE_SIZE=0, no importance sampling)"
        )
        print(f"  [{segment}] ESTIMATION: ALL {_n_zone_ids} zones × {_n_obs_seg} obs = "
              f"{_n_zone_ids * _n_obs_seg} rows (no sampling)")

    # ── CHECK: does landuse have expected columns? ────────────────────────
    _lu_check_cols = ['size_term', 'acdc_aggl', 'acdc_comp']
    _lu_missing = [c for c in _lu_check_cols if c not in lu_with_st.columns]
    if _lu_missing:
        print(f"  [{segment}] WARNING: landuse missing columns: {_lu_missing} "
              f"(available: {list(lu_with_st.columns[:15])})")
    else:
        _st_vals = lu_with_st['size_term'].describe()
        print(f"  [{segment}] landuse size_term: mean={_st_vals['mean']:.3f} "
              f"std={_st_vals['std']:.3f} min={_st_vals['min']:.3f} max={_st_vals['max']:.3f} "
              f"n_zones={int(_st_vals['count'])}")

    dt_d     = _dest_tree(seg_acdc, persons, lu_with_st, skims, ls,
                          sampled_alts=_sampled_alts, skim_df=_dest_skim_df,
                          crash_skims=_dest_crash_skims, segment=segment)
    print(f"  [{segment}] _dest_tree built: flat_mode={dt_d.attrs.get('_flat_mode', False) if hasattr(dt_d, 'attrs') else '?'}")

    # ── DataTree variable audit ────────────────────────────────────────────
    _dest_compile_unreliable = False
    _is_flat = dt_d.attrs.get('_flat_mode', False) if hasattr(dt_d, 'attrs') else getattr(dt_d, '_flat_mode', False)
    if _is_flat:
        # Flat mode: all columns are pre-joined in the ca dataset.
        # No DataTree node introspection needed - _dest_tree validated everything.
        logger.info(f"_run_segment [{segment}]: flat Dataset mode - skipping DataTree audit")
    else:
        try:
            _required_od  = {'DIST', 'SOVTOLL_TIME__AM', 'SOVTOLL_DIST__AM'}
            _required_emp = {'size_term', 'acdc_aggl', 'acdc_comp'}
            _od_avail  = dt_d.attrs.get('_od_vars_cached', None) or _dataset_var_names(dt_d['od'])
            _emp_avail = _dataset_var_names(dt_d['emp'])
            _skim_source = set(str(c) for c in (skim_df.columns if skim_df is not None else []))
            _land_source = set(str(c) for c in lu_with_st.columns)

            _od_source_has = ('DIST' in _skim_source or 'SOVTOLL_DIST__AM' in _skim_source)
            _emp_source_has = _required_emp.issubset(_land_source)

            _missing_od  = _required_od  - _od_avail
            _missing_emp = _required_emp - _emp_avail
            if _missing_od and not _od_source_has:
                logger.warning(
                    f"_run_segment [{segment}]: od DataTree missing variables: "
                    f"{sorted(_missing_od)} - destination utility expressions will fail. "
                    f"Available od vars (first 15): {sorted(_od_avail)[:15]}"
                )
            if _missing_emp and not _emp_source_has:
                logger.warning(
                    f"_run_segment [{segment}]: emp DataTree missing variables: "
                    f"{sorted(_missing_emp)}.  "
                    f"Available emp vars (first 15): {sorted(_emp_avail)[:15]}"
                )
            if (_missing_od and not _od_source_has) or (_missing_emp and not _emp_source_has):
                raise RuntimeError(
                    f"_run_segment [{segment}]: cannot estimate destination model; "
                    f"missing od={sorted(_missing_od)} emp={sorted(_missing_emp)}"
                )
            logger.debug(f"_run_segment [{segment}]: DataTree audit OK "
                         f"(od={len(_od_avail)} vars, emp={len(_emp_avail)} vars)")
        except Exception as _audit_e:
            logger.debug(f"_run_segment [{segment}]: DataTree audit failed: {_audit_e}")

    _dest_raw = (_dest_feature_override
                 if dest_feature_override is not None
                 else (_dest_uec if _nosafety else DEST_UEC))

    if _is_flat:
        # Flat mode: _build_dest_model uses DEST_UEC_FLAT directly.
        # No need for _filter_dest_uec_by_source_columns - columns are pre-joined.
        _dest_safe = None
        logger.info(
            f"_run_segment [{segment}]: flat mode - using DEST_UEC_FLAT "
            f"({sum(1 for _, (_, s) in DEST_UEC_FLAT.items() if s is None or segment in s)} terms for segment)"
        )
    else:
        try:
            _skim_cols = set(str(c) for c in (skim_df.columns if skim_df is not None else []))
        except Exception:
            _skim_cols = set()
        _skim_cols |= {'DIST', 'SOVTOLL_DIST__AM', 'SOVTOLL_TIME__AM', 'CRASH'}
        _land_cols = set(str(c) for c in lu_with_st.columns)
        _person_cols = set(str(c) for c in persons.columns)
        if isinstance(_dest_raw, dict):
            _dest_candidate = _dest_raw
        elif isinstance(_dest_raw, list):
            _dest_candidate = _dest_uec_from_feature_list(_dest_raw)
        else:
            _dest_candidate = DEST_UEC

        _dest_safe = _filter_dest_uec_by_source_columns(
            segment=segment,
            uec=_dest_candidate,
            cond=DEST_COND,
            skim_cols=_skim_cols,
            landuse_cols=_land_cols,
            person_cols=_person_cols,
        )
        if not _dest_safe:
            logger.warning(
                f"_run_segment [{segment}]: destination UEC empty after source-column filter; "
                "falling back to full DEST_UEC"
            )
            _dest_safe = _filter_dest_uec_by_source_columns(
                segment=segment,
                uec=DEST_UEC,
                cond=DEST_COND,
                skim_cols=_skim_cols,
                landuse_cols=_land_cols,
                person_cols=_person_cols,
            )

        logger.info(
            f"_run_segment [{segment}]: destination terms before Larch build = {len(_dest_safe)} "
            f"({', '.join(sorted(_dest_safe.keys())[:20])}{'...' if len(_dest_safe) > 20 else ''})"
        )

    # ── Two-stage destination (SA3 district → SA2 zone) ───────────────────
    if STAGE5_DEST_TWO_STAGE and segment in ('work', 'edu', 'nonmand'):
        print(f"  [{segment}] TWO-STAGE: SA3→SA2 hierarchical destination estimation ...")
        _ts_result = _run_two_stage_dest(
            segment=segment, tours_df=seg_acdc, persons=persons,
            landuse=landuse, skims=skims, skim_df=_dest_skim_df,
            logsums=ls, lu_with_st=lu_with_st, seed=seed,
            dest_feature_override=dest_feature_override,
        )
        if _ts_result is not None:
            dest_mdl = _ts_result['upper_model']
            dest_mdl._segment = segment
            dest_coef = _extract_coef_early(dest_mdl)
            _dist_het = _estimate_dest_dist_heterogeneity(seg, dest_coef, segment)
            return dict(
                mode_model  = mode_mdl,
                dest_model  = dest_mdl,
                mode_coef   = mode_coef,
                dest_coef   = dest_coef,
                mxl_result  = mxl_result,
                frozen_probs = _frozen_probs,
                s_log_dist  = _dist_het.get('s_log_dist', 0.5),
                dt_mode     = dt_m,
                dt_dest     = dt_d,
                logsums     = ls,
                logsum_mean = float(_ls_mean),
                logsum_std  = float(_ls_std),
                od_calibration = {},
                zone_shadow_prices = None,
                pred_zone_shares  = None,
                pred_mode_shares  = _model_predicted_mode_shares(mode_mdl),
                cluster_targets   = {},
                crash_source      = crash_skims,
                segment     = segment,
                n_tours     = len(seg),
                mode_ll     = mode_ll,
                dest_ll     = getattr(dest_mdl, '_cached_ll', float('nan')),
                _two_stage_lower = _ts_result['lower_models'],
                _two_stage_sa3_map = _ts_result['sa3_to_sa2'],
                _two_stage_zone_map = _ts_result['zone_to_sa3'],
            )
        else:
            print(f"  [{segment}] TWO-STAGE: failed, falling back to standard model")

    # ── Standard destination model path ───────────────────────────────────
    # (Also the fallback when the two-stage path above is disabled or fails.)
    dest_mdl = _build_dest_model(
        dt_d,
        segment,
        custom_dest_vars=_dest_safe,
        od_vars_hint=(list(skim_df.columns) if skim_df is not None else None),
    )
    dest_mdl._segment = segment  # stored for study-level size term in prediction path

    # If destination utility compiles to no parameters, clear sharrow cache
    # and rebuild with a minimal compile-safe UEC before attempting estimation.
    _dest_pf_pre = _pf_len_safe(dest_mdl)
    if _dest_pf_pre == 0:
        logger.warning(
            f"_run_segment [{segment}]: destination model compiled with empty pf; "
            "clearing sharrow cache and rebuilding with minimal UEC"
        )
        try:
            _clear_sharrow_cache()
        except Exception:
            pass
        dest_mdl = _build_dest_model(
            dt_d,
            segment,
            custom_dest_vars=_minimal_dest_uec(dt_d, segment),
            od_vars_hint=(list(skim_df.columns) if skim_df is not None else None),
        )
        dest_mdl._segment = segment  # stored for study-level size term in prediction path
        _dest_pf_pre = _pf_len_safe(dest_mdl)
        logger.info(f"_run_segment [{segment}]: destination pf after rebuild = {_dest_pf_pre}")

    # ── Inject warm-start coefficients - destination model ────────────────
    if coef_warm_start:
        _n_dest_warm = 0
        _dest_pf_idx = set(dest_mdl.pf.index.tolist())
        for _wname, _wval in coef_warm_start.items():
            if _wname in _dest_pf_idx:
                try:
                    _wval_f = float(_wval)
                    # Skip warm-start values that are essentially zero -
                    # these indicate the parameter was holdfasted in the
                    # previous run.  Using them as initial values places the
                    # optimizer at a lower bound (if one exists) where the
                    # projected gradient is zero and the parameter stalls.
                    # Let the model's coded initial value (e.g. c_aggl=0.20)
                    # take effect instead.
                    if abs(_wval_f) < 1e-10:
                        logger.debug(
                            f"_run_segment [{segment}]: skipping near-zero "
                            f"warm-start for '{_wname}' ({_wval_f:.2e}) - "
                            "likely holdfasted in previous run"
                        )
                        continue
                    dest_mdl.set_value(_wname, _wval_f)
                    _n_dest_warm += 1
                except Exception as _we:
                    logger.debug(f"warm-start dest set_value({_wname}) failed: {_we}")
        if _n_dest_warm:
            logger.info(f"_run_segment [{segment}]: dest warm-start applied "
                        f"{_n_dest_warm}/{len(coef_warm_start)} params")

    _n_free_before = sum(1 for p in dest_mdl.pf.index
                         if not dest_mdl.pf.loc[p].get('holdfast', False)) if _dest_pf_pre > 0 else 0
    print(f"  [{segment}] Estimating destination model: {_dest_pf_pre} params, {_n_free_before} free ...")
    if _dest_pf_pre > 0:
        print(f"  [{segment}] Initial LL before estimation: ", end='', flush=True)
        try:
            _ll_before = float(dest_mdl.loglike())
            print(f"{_ll_before:.1f}")
        except Exception as _lle:
            print(f"(could not compute: {_lle})")
        try:
            dest_mdl = _estimate(dest_mdl, two_phase=True)
            # Prune zero-fit / insignificant destination terms and refit. The
            # structural backbone (c_log_dist, c_size_term, log_correction,
            # c_crash) is protected; distance-spline / competition / attractor
            # terms the data can't identify are removed.
            dest_mdl = _prune_and_refit(
                dest_mdl, lambda _m: _estimate(_m, two_phase=True),
                'stage5_dest', label=segment)
            dest_mdl = _apply_sign_constraints(dest_mdl, 'stage5_dest')
            try:
                _ll_after = float(dest_mdl.loglike())
                print(f"  [{segment}] LL after estimation: {_ll_after:.1f}")
                # ── LL null-ratio sanity check ─────────────────────────────
                # Null LL = N × log(1/K): a good model should be 2-10× better.
                # If estimated LL is WORSE than null (ratio > 1) or catastrophically
                # bad (ratio >> 1), the model has a gradient saturation or data problem.
                # Diagnose: report the top magnitude coefficient for debugging.
                try:
                    _n_est = seg_acdc['ptour_id'].nunique() if 'ptour_id' in seg_acdc.columns else len(seg_acdc)
                    _n_alts = _n_zone_ids  # zone alternatives used in estimation
                    _ll_null = _n_est * np.log(1.0 / max(_n_alts, 1))
                    _ll_ratio = abs(_ll_after) / abs(_ll_null) if _ll_null != 0 else float('nan')
                    _ll_msg = (
                        f"[{segment}] LL check: est={_ll_after:.1f}, null≈{_ll_null:.1f}, "
                        f"|est|/|null| = {_ll_ratio:.2f}"
                    )
                    if np.isnan(_ll_ratio):
                        logger.debug(_ll_msg)
                    elif _ll_ratio > 10.0:
                        logger.error(
                            _ll_msg + " - CATASTROPHIC: LL >> null model. "
                            "Likely causes: (1) raw (non-log) attraction variable causing "
                            "gradient saturation, (2) chosen zones have near-zero probability "
                            "(check attraction variable is log-transformed), "
                            "(3) zone with all-zero data in chosen set. "
                            "Inspect the top coefficient below for the offending parameter."
                        )
                        try:
                            _pf_diag = dest_mdl.pf
                            _top_param = _pf_diag['value'].abs().idxmax()
                            logger.error(
                                f"  Largest |coeff|: {_top_param} = "
                                f"{float(_pf_diag.loc[_top_param, 'value']):.4f} "
                                f"(if this is an attraction term, check its data scale)"
                            )
                        except Exception:
                            pass
                    elif _ll_ratio > 2.0:
                        logger.warning(_ll_msg + " - LL worse than null; check starting values and data.")
                    else:
                        logger.info(_ll_msg + " - OK")
                except Exception as _lc_e:
                    logger.debug(f"[{segment}] LL null check failed: {_lc_e}")
            except Exception as _lla:
                print(f"  [{segment}] LL after estimation: (could not compute: {_lla})")
            # Print top 5 params by magnitude
            try:
                _pf = dest_mdl.pf
                _free = _pf[~_pf.get('holdfast', pd.Series(False, index=_pf.index)).astype(bool)]
                print(f"  [{segment}] Top estimated params (by |value|):")
                for _pname, _row in _free['value'].abs().sort_values(ascending=False).head(5).items():
                    print(f"    {_pname} = {_free.loc[_pname, 'value']:.4f}")
            except Exception as _pp:
                print(f"  [{segment}] Could not print params: {_pp}")
        except Exception as _dest_est_e:
            logger.warning(
                f"_run_segment [{segment}]: Larch destination estimation failed; "
                "using alternative JAX estimator. "
                f"Error: {_dest_est_e!r}"
            )
            dest_mdl._cached_ll = float('nan')
            try:
                dest_mdl._coef_cache = {}
            except Exception:
                pass
    else:
        logger.warning(
            f"_run_segment [{segment}]: destination pf still empty after rebuild; "
            "skipping Larch estimation and using alternative estimator"
        )
        dest_mdl._cached_ll = float('nan')

    # ── Rebuild a FULL-zone-universe ls/dt_d for post-estimation calibration ──
    # ls (used for estimation above) only covers zones that were actually
    # observed as HHTS destinations for this segment (243/295/290 zones here,
    # not the true ~329/330-zone universe) - see _logsums()'s zone_ids
    # derivation, which keys off DTAZ values present in dt_m's tours. That's
    # fine for estimation (it's still the true observed choice set), but
    # calibrating shadow prices / caching pred_zone_shares against this
    # truncated, already-popular subset produces a far more concentrated
    # distribution than the real simulation (_predict_batch evaluates over
    # all zones, since its placeholder DTAZ=-1 rows trip _logsums' "<10
    # zones" full-list fallback). Rebuild the logsums/dest tree once here
    # against the full landuse zone list so the validation report and the
    # cached shadow prices reflect what simulation will actually do.
    try:
        _fz_col = 'ORDER' if 'ORDER' in lu_with_st.columns else lu_with_st.columns[0]
        _full_zone_ids = sorted(int(z) for z in lu_with_st[_fz_col].unique() if int(z) > 0)
        ls_full = _logsums(mode_mdl, dt_m, skims=skims, verbose=False,
                            coef_override=_mode_coef_early if _mode_coef_early else None,
                            zone_ids_override=_full_zone_ids)
        if _ls_std and _ls_std != 1.0:
            _lsf_vals = ls_full.values.astype(float)
            _lsf_finite = _lsf_vals[np.isfinite(_lsf_vals)]
            if len(_lsf_finite) > 0:
                ls_full = ls_full.copy(data=((_lsf_vals - _ls_mean) / _ls_std).astype(ls_full.values.dtype))
        dt_d_full = _dest_tree(seg_acdc, persons, lu_with_st, skims, ls_full,
                               sampled_alts=None, skim_df=_dest_skim_df,
                               crash_skims=_dest_crash_skims, segment=segment)
        logger.info(f"_run_segment [{segment}]: rebuilt full-zone calibration tree "
                    f"({len(_full_zone_ids)} zones, vs {_n_zone_ids} at estimation)")
    except Exception as _fz_e:
        logger.warning(f"_run_segment [{segment}]: full-zone calibration rebuild failed "
                       f"({_fz_e}); falling back to estimation-time (truncated) zone set")
        ls_full, dt_d_full = ls, dt_d

    try:
        # Pass OTAZ explicitly so _dest_utility_numpy doesn't fail when
        # the DataTree case index doesn't align with ls.coords['cases'].
        _otaz_arr_cal = None
        if 'OTAZ' in seg.columns:
            _otaz_arr_cal = seg['OTAZ'].values.astype(int)
        elif 'otaz' in seg.columns:
            _otaz_arr_cal = seg['otaz'].values.astype(int)
        probs_cal = _dest_probs_numpy(dest_mdl, dt_d_full, ls_full, list(ls_full.coords['zones'].values), persons,
                                      case_otaz_override=_otaz_arr_cal,
                                      landuse=lu_with_st, skim_df=skim_df)
        _n_nonzero = float((probs_cal > 1e-9).sum(axis=1).mean()) if probs_cal is not None else 0.0
        print(f"  [{segment}] Post-estimation: mean zones with non-zero share = {_n_nonzero:.1f}/{len(list(ls_full.coords['zones'].values))}")
        od_calibration = _fit_destination_od_calibration(seg_acdc.reset_index(drop=True), probs_cal,
                                                         list(ls_full.coords['zones'].values), alpha=0.6)
        dest_mdl._od_calibration = od_calibration
        _sp_history: list = []
        dest_mdl._zone_shadow_prices = _fit_destination_zone_shadow_prices(
            seg_acdc.reset_index(drop=True), probs_cal,
            list(ls_full.coords['zones'].values),
            n_iter=50,             # more iterations - stable convergence with small alpha
            alpha=0.25,            # ARC/ActivitySim standard: 0.2-0.3 prevents oscillation.
                                   # alpha=0.85 caused divergence (MaxRatioErr increasing after iter 2).
            clip_sp=5.0,           # allow up to exp(5)≈150× correction for severely under-predicted zones
            target_threshold=3,    # adjust any zone with ≥3 observed trips
            _history_out=_sp_history,
        )
        # Save ARC-style shadow price convergence + validation scatter plots
        try:
            import os as _os
            _sp_plot_dir = _os.path.join(_os.environ.get('PIPELINE_OUT_DIR', 'runs/main'), 'location_choice')
            _os.makedirs(_sp_plot_dir, exist_ok=True)
            # 1. Convergence trace (failing zones + max shadow per iteration)
            _sp_plot_path = _os.path.join(_sp_plot_dir, f'shadow_price_convergence_{segment}.png')
            _plot_shadow_price_convergence(_sp_history, segment, _sp_plot_path)
            # 2. Obs vs pred scatter - before & after shadow prices (ARC-style)
            if probs_cal is not None:
                _sp_val_path = _os.path.join(_sp_plot_dir, f'shadow_price_validation_{segment}.png')
                _plot_shadow_price_validation(
                    obs_rows      = seg_acdc.reset_index(drop=True),
                    probs_base    = probs_cal,
                    shadow_prices = dest_mdl._zone_shadow_prices if dest_mdl._zone_shadow_prices is not None
                                    else np.zeros(len(list(ls_full.coords['zones'].values))),
                    zone_ids      = list(ls_full.coords['zones'].values),
                    landuse       = lu_with_st,
                    segment       = segment,
                    save_path     = _sp_val_path,
                )
        except Exception as _spe:
            logger.debug(f"Shadow price plot failed: {_spe}")
        dest_mdl._cluster_targets = _fit_destination_cluster_targets(seg_acdc.reset_index(drop=True), lu_with_st)
        logger.info(f"_run_segment [{segment}]: destination OD calibration origins={len(od_calibration)}")
        # ── Cache aggregated zone shares for validation ────────────────
        # IMPORTANT: apply shadow prices to probs_cal BEFORE caching so that
        # _location_choice_validation shows post-SP zone shares (i.e. the
        # validation ratios reflect what the model actually predicts in simulation,
        # not the raw pre-shadow-price utility).
        #
        # Shadow prices are zone-level utility offsets: V[j] += sp[j].
        # Applying them to probs_cal (which was computed with sp=None):
        #   probs_sp[i,j] = probs_cal[i,j] * exp(sp[j])
        #   then renormalise each row so rows still sum to 1.
        try:
            _zone_ids_arr = np.asarray(list(ls_full.coords['zones'].values), dtype=int)
            if probs_cal is not None and probs_cal.shape[1] == len(_zone_ids_arr):
                _sp = dest_mdl._zone_shadow_prices  # numpy array (n_zones,) or None
                if _sp is not None and len(_sp) == probs_cal.shape[1]:
                    _sp_exp = np.exp(np.clip(_sp, -10, 10)).astype(np.float32)
                    _probs_sp = probs_cal * _sp_exp[np.newaxis, :]
                    _row_sums = _probs_sp.sum(axis=1, keepdims=True)
                    _row_sums = np.where(_row_sums > 1e-12, _row_sums, 1.0)
                    _probs_sp = _probs_sp / _row_sums
                else:
                    _probs_sp = probs_cal  # no shadow prices - use as-is
                _raw_sum = _probs_sp.sum(axis=0)
                _total = max(float(_raw_sum.sum()), 1e-9)
                dest_mdl._pred_zone_shares = pd.Series(
                    (_raw_sum / _total).astype(np.float64),
                    index=_zone_ids_arr,
                )
                logger.info(
                    f"_run_segment [{segment}]: cached pred_zone_shares "
                    f"({int((dest_mdl._pred_zone_shares > 1e-6).sum())} zones with share, "
                    f"shadow prices {'applied' if _sp is not None else 'none'})"
                )
            else:
                dest_mdl._pred_zone_shares = None
        except Exception as _pzs_e:
            dest_mdl._pred_zone_shares = None
            logger.debug(f"_run_segment [{segment}]: pred_zone_shares cache failed ({_pzs_e})")
    except Exception as _cal_e:
        dest_mdl._od_calibration = {}
        dest_mdl._cluster_targets = {}
        dest_mdl._pred_zone_shares = None
        logger.debug(f"_run_segment [{segment}]: destination calibration skipped ({_cal_e})")
    _print_model_table(dest_mdl, f"{segment.upper()} destination model")

    # retrieve LL cached by _estimate() - reliable even after DataTree stripped
    mode_ll = getattr(mode_mdl, '_cached_ll', float('nan'))
    dest_ll = getattr(dest_mdl, '_cached_ll', float('nan'))

    # Extract NL model coefs to plain dicts NOW while models are mangled.
    # These survive pickle without depending on Larch's __setstate__ behaviour.
    # Prefer _final_pf (snapshot cached by _estimate) so values are stable
    # even after Larch rebuilds the property.
    def _extract_coef(mdl):
        try:
            pf = (mdl._final_pf if hasattr(mdl, '_final_pf')
                  and isinstance(mdl._final_pf, pd.DataFrame)
                  else mdl.pf)
            return {n: float(pf.loc[n,'value']) for n in pf.index
                    if not np.isnan(float(pf.loc[n,'value']))}
        except Exception:
            return {}

    mode_coef = (_extract_coef(mode_mdl)
                 or getattr(mode_mdl, '_coef_cache', {})
                 or {})
    dest_coef = (_extract_coef(dest_mdl)
                 or getattr(dest_mdl, '_coef_cache', {})
                 or {})

    # Alternative estimation path: if Larch destination estimation produced no
    # coefficients, estimate a compact sampled-alternative conditional logit
    # with JAX instead of using fixed defaults.
    if not dest_coef:
        try:
            logger.warning(
                f"_run_segment [{segment}]: destination coefficients empty after Larch estimation; "
                "trying alternative JAX estimation."
            )
            dest_coef = _estimate_dest_coefficients_alt(
                seg_tours=seg_acdc,
                landuse=lu_with_st,
                logsums=ls,
                sampled_alts=_sampled_alts,
                skim_df=skim_df if skim_df is not None else pd.DataFrame(),
                crash_source=crash_skims,
            )
            dest_mdl._coef_cache = dest_coef
        except Exception as _alt_est_e:
            logger.error(
                f"_run_segment [{segment}]: alternative destination estimation failed: {_alt_est_e}"
            )

    _dest_coef_estimated = bool(dest_coef)
    _require_dest_est = os.environ.get('STAGE5_REQUIRE_DEST_ESTIMATION', '1') != '0'
    if _require_dest_est and not _dest_coef_estimated:
        raise RuntimeError(
            f"_run_segment [{segment}]: no destination coefficients estimated "
            "(Larch + alternative JAX both failed)."
        )

    dest_coef = _fill_dest_coef_with_segment_defaults(dest_coef, segment)
    dest_mdl._coef_cache = dict(dest_coef)

    # Frozen probabilities: prefer the MXL-derived table if available,
    # otherwise fall back to computing from NL coefficients + seg data.
    _frozen_probs = None
    if mxl_result is not None:
        _frozen_probs = mxl_result.get('frozen_probs')
    if _frozen_probs is None:
        # Build a minimal mxl_result-like dict from NL coefs for the fallback
        _nl_result = {'coef': mode_coef}
        _frozen_probs = _compute_frozen_probs(seg, _nl_result, segment)

    # Destination distance heterogeneity: estimate s_log_dist from HHTS residuals.
    # This drives _frozen_person_dist_offsets during prediction (work segment only).
    _dist_het = _estimate_dest_dist_heterogeneity(seg, dest_coef, segment)
    _s_log_dist = _dist_het.get('s_log_dist', 0.5)

    return dict(
        mode_model  = mode_mdl,
        dest_model  = dest_mdl,
        mode_coef   = mode_coef,    # NL mode model coefs as plain dict (pickle-safe)
        dest_coef   = dest_coef,    # NL dest model coefs as plain dict (pickle-safe)
        mxl_result  = mxl_result,   # mixed logit result (dict or None)
        frozen_probs = _frozen_probs,  # cell-level HHTS→synth probability table
        s_log_dist  = _s_log_dist,  # between-person log-dist std dev for frozen draws
        dt_mode     = dt_m,
        dt_dest     = dt_d,
        logsums     = ls,
        logsum_mean = float(_ls_mean),
        logsum_std  = float(_ls_std),
        od_calibration    = getattr(dest_mdl, '_od_calibration', {}),
        zone_shadow_prices = getattr(dest_mdl, '_zone_shadow_prices', None),
        # pred_zone_shares / pred_mode_shares: stored explicitly here because
        # Larch model objects don't preserve custom attributes through cloudpickle.
        # Used by _location_choice_validation (including the post-pkl reload call
        # in post_usual_locations) to show model-predicted shares without needing
        # a live DataTree attached.
        pred_zone_shares  = getattr(dest_mdl, '_pred_zone_shares', None),
        pred_mode_shares  = _model_predicted_mode_shares(mode_mdl),
        cluster_targets   = getattr(dest_mdl, '_cluster_targets', {}),
        crash_source      = crash_skims,
        segment     = segment,
        n_tours     = len(seg),
        mode_ll     = mode_ll,
        dest_ll     = dest_ll,
    )


# ===========================================================================
# FULL TRIP CHAIN BUILDER
# ===========================================================================

def skim_lookup_to_df(skim_lookup: dict) -> pd.DataFrame:
    """
    Convert the {(otaz,dtaz): {col:val}} lookup dict back to a flat DataFrame.

    Used by _attach_acdc_terms inside _predict_batch where only the lookup
    dict (not the original skim_df) is available.  Returns only the columns
    needed for ACDC computation.
    """
    if not skim_lookup:
        return pd.DataFrame()
    rows = []
    for (otaz, dtaz), vals in skim_lookup.items():
        ivt = vals.get('SOVTOLL_TIME__AM', None)
        if ivt is not None:
            rows.append({'OTAZ': otaz, 'DTAZ': dtaz, 'SOVTOLL_TIME__AM': ivt})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _softmax_from_utilities(V: np.ndarray) -> np.ndarray:
    """Convert utility matrix to row-normalized probabilities.

    NaN/Inf utilities are treated as infeasible alternatives (probability 0),
    not propagated. Previously only ``row_max`` was computed from a
    NaN-sanitized copy of V - the actual ``exp(V - row_max)`` step still ran
    on the raw V, so a single NaN utility cell poisoned exp_V -> the row sum
    -> the entire row's probabilities to NaN (this is what was producing the
    all-"nan" Pred columns in the destination validation report).
    """
    if V is None or V.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    V_finite = np.where(np.isfinite(V), V, -np.inf)
    _n_bad = int((~np.isfinite(V)).sum())
    if _n_bad > 0:
        logger.warning(
            f"_softmax_from_utilities: {_n_bad} non-finite utility cell(s) "
            f"({_n_bad / V.size:.2%} of {V.shape[0]}x{V.shape[1]}) treated as "
            "infeasible (prob=0). Check the utility terms/data columns feeding "
            "_dest_utility_numpy for NaN/Inf - a coefficient's source column "
            "may be missing at this call site."
        )
    row_max = np.max(V_finite, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp_V = np.exp(np.clip(V_finite - row_max, -500, 0))
    totals = exp_V.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    return (exp_V / totals).astype(np.float64)


def _dest_utility_numpy(dest_mdl, dt_d_batch, ls_batch, zone_ids, persons,
                        include_logsum: bool = True,
                        case_otaz_override=None,
                        landuse: pd.DataFrame = None,
                        skim_df: pd.DataFrame = None):
    """Vectorized destination utility evaluation for all cases x zones.

    Parameters
    ----------
    landuse : optional pd.DataFrame (zone_order_and_landuse enriched with
        accessibility and size terms).  Used as a fallback when the flat
        Dataset ``dt_d_batch`` was built with importance-sampled alternatives
        (60 per chooser) but the call supplies all 329 zones - the Dataset
        dimension would be ``(n_cases, 60)`` while ``n_zones=329``, causing
        all shape checks to fail silently and return 0.  When ``landuse`` is
        supplied the ``_emp_col`` function falls back to reading directly from
        the zone-indexed landuse DataFrame.
    skim_df : optional flat OD DataFrame (OTAZ, DTAZ, DIST, SOVTOLL_TIME__AM
        …).  Used as fallback for OD matrices when the Dataset check fails.
    """
    try:
        coef = _coef_dict_from_model(dest_mdl)

        def _pv(name, default=0.0):
            try:
                val = coef.get(name, default)
                val = float(val)
                return default if np.isnan(val) else val
            except Exception:
                return default

        n_cases = int(ls_batch.shape[0])
        n_zones = len(zone_ids)
        zone_arr = np.asarray(zone_ids, dtype=np.int32)
        ls_cases = [str(c) for c in ls_batch.coords['cases'].values]

        c_log_dist = _pv('c_log_dist', -0.5)
        c_dist = _pv('c_dist', 0.0)
        c_crash = _pv('c_crash', 0.0)
        c_logsum = _pv('c_logsum', 1.0) if include_logsum else 0.0
        c_size_term = _pv('c_size_term', 1.0)
        c_aggl = _pv('c_aggl', 0.0)
        c_comp = _pv('c_comp', 0.0)
        # Per-type ACDC agglomeration (imove/gonzalo decomposition). Previously
        # the prediction utility only applied the blended c_aggl and IGNORED
        # these, so the themed coefficients had no effect even when estimated.
        _ACDC_THEMES = ('retail', 'office', 'edu', 'leisure',
                        'parks', 'health', 'food', 'civic')
        c_aggl_themed = {t: _pv(f'c_aggl_{t}', 0.0) for t in _ACDC_THEMES}
        c_incp_dist = _pv('c_incp_dist', 0.0)
        c_age_dist = _pv('c_age_dist', 0.0)
        c_veh_dist = _pv('c_veh_dist', 0.0)
        c_female_dist = _pv('c_female_dist', 0.0)
        c_nprd_dist = _pv('c_nprd_dist', 0.0)
        c_occp_dist = _pv('c_occp_dist', 0.0)
        c_propensity_dist = _pv('c_propensity_dist', 0.0)
        c_indp_fin = _pv('c_indp_finance', 0.0)
        c_indp_hlth = _pv('c_indp_health', 0.0)
        c_indp_retl = _pv('c_indp_retail', 0.0)
        # c_totemp_work removed (collinear with size_term → negative estimated coefficient)
        c_log_commercial_work = _pv('c_log_commercial_work', 0.0)
        c_edu_post = _pv('c_edu_post', 0.0)
        c_edu_prim = _pv('c_edu_primary', 0.0)
        c_edu_sec = _pv('c_edu_secondary', 0.0)
        c_edu_prim_priv = _pv('c_edu_prim_priv', 0.0)
        c_edu_sec_priv = _pv('c_edu_sec_priv', 0.0)
        c_nonmand_res = _pv('c_nonmand_res', 0.0)
        c_nonmand_ent = _pv('c_nonmand_ent', 0.0)
        c_nonmand_com = _pv('c_nonmand_com', 0.0)
        c_nonmand_retail = _pv('c_nonmand_retail', 0.0)
        c_nonmand_totpop = _pv('c_nonmand_totpop', 0.0)
        c_nonmand_eatout = _pv('c_nonmand_eatout', 0.0)
        c_auto_work = _pv('c_auto_work_acc', 0.0)
        c_auto_shop = _pv('c_auto_shop_acc', 0.0)
        c_auto_edu = _pv('c_auto_edu_acc', 0.0)
        c_auto_nonmand = _pv('c_auto_nonmand_acc', 0.0)
        c_walk_acc = _pv('c_walk_acc', 0.0)
        c_transit_acc = _pv('c_transit_acc', 0.0)
        c_noveh_walk = _pv('c_noveh_walk_acc', 0.0)
        c_noveh_trn = _pv('c_noveh_transit_acc', 0.0)

        try:
            tour_df = dt_d_batch['tour'].to_dataframe().reset_index()
        except Exception:
            tour_df = pd.DataFrame()
        case_col = next((c for c in ['ptour_id', 'index'] if c in tour_df.columns), None)
        if case_col is not None:
            tour_df = tour_df.assign(_case_key=tour_df[case_col].astype(str))
            tour_df = tour_df.drop_duplicates('_case_key').set_index('_case_key')
        else:
            tour_df = pd.DataFrame(index=pd.Index(ls_cases, name='_case_key'))

        # Detect flat-mode Dataset once so _emp_col can read variables directly
        _is_flat_ds = bool(getattr(dt_d_batch, 'attrs', {}).get('_flat_mode', False))
        # Pre-build a zone-indexed landuse lookup if the landuse kwarg is supplied.
        # Used when the flat Dataset has importance-sampled dims (n_cases, n_sampled)
        # that don't match the full n_zones - in that case the Dataset check fails and
        # we need a direct zone-table lookup to get non-zero landuse values.
        _lu_for_fallback = None
        if landuse is not None:
            try:
                _lu_zcol = 'ORDER' if 'ORDER' in landuse.columns else None
                if _lu_zcol:
                    _lu_for_fallback = landuse.set_index(_lu_zcol)
                else:
                    _lu_for_fallback = landuse.copy()
            except Exception:
                pass

        def _emp_col(col, default=0.0):
            # Fast path: flat Dataset stores landuse as (n_cases, n_zones) DataArrays.
            # Landuse is zone-specific (same for every chooser) so first row = all zones.
            if _is_flat_ds:
                try:
                    _da = dt_d_batch[col]
                    _v = np.asarray(_da.values, dtype=np.float32)
                    if _v.ndim == 2 and _v.shape[1] == n_zones:
                        return _v[0, :]
                    elif _v.ndim == 1 and len(_v) == n_zones:
                        return _v
                except Exception:
                    pass
            # Landuse DataFrame fallback - used when Dataset was importance-sampled
            # (n_alts ≠ n_zones) so the shape check above fails silently.
            if _lu_for_fallback is not None and col in _lu_for_fallback.columns:
                try:
                    vals = pd.to_numeric(
                        _lu_for_fallback[col].reindex(zone_arr.astype(int)),
                        errors='coerce'
                    ).fillna(default).values.astype(np.float32)
                    return vals
                except Exception:
                    pass
            try:
                df = dt_d_batch['emp'].to_dataframe().reset_index()
                zone_col = next((c for c in ['ORDER', 'zone_id', 'index'] if c in df.columns), df.columns[0])
                df = df.set_index(zone_col)
                if col in df.columns:
                    vals = pd.to_numeric(df[col].reindex(zone_arr), errors='coerce').fillna(default).values
                    return vals.astype(np.float32)
            except Exception:
                pass
            return np.full(n_zones, default, dtype=np.float32)

        # size_term is already demeaned by compute_size_terms (mean=0 by construction)
        # - do NOT re-centre here, that would subtract the mean of a zero-mean series
        # which changes nothing on average but adds floating-point noise.
        size_term_j = _emp_col('size_term', 0.0)
        acdc_aggl_j = _emp_col('acdc_aggl', 0.0)
        acdc_comp_j = _emp_col('acdc_comp', 0.0)
        # Per-type ACDC agglomeration columns (zero if the theme wasn't built).
        acdc_aggl_themed_j = {
            t: _emp_col(f'acdc_aggl_{t}', 0.0) for t in _ACDC_THEMES
        }
        auto_work_j = _emp_col('auto_work_acc', 0.0)
        auto_shop_j = _emp_col('auto_shop_acc', 0.0)
        walk_acc_j = _emp_col('walk_acc', 0.0)
        transit_j = _emp_col('transit_acc', 0.0)
        # ── Log-transform employment to match DEST_UEC_FLAT ────────────────
        # DEST_UEC_FLAT uses log(emp+1) columns so estimated coefficients expect
        # log-scale inputs. c_totemp_work was removed (collinear with size_term).
        fpsempn_j = np.log(np.clip(_emp_col('FPSEMPN', 0.0), 0.0, None) + 1.0).astype(np.float32)
        herempn_j = np.log(np.clip(_emp_col('HEREMPN', 0.0), 0.0, None) + 1.0).astype(np.float32)
        retempn_j = np.log(np.clip(_emp_col('RETEMPN', 0.0), 0.0, None) + 1.0).astype(np.float32)
        commercial_j = _emp_col('commercial', 0.0)
        log_commercial_j = np.log(np.clip(commercial_j, 0.0, None) + 1.0).astype(np.float32)
        # ── Log-transform edu attractions to match DEST_UEC_FLAT (log-scale) ──
        # Raw Primary/Secondary range 0-3,068; raw Education may be large too.
        # DEST_UEC_FLAT now maps edu columns to log(x+1)-transformed versions
        # (same as computed in _dest_tree). Prediction path must match exactly.
        edu_post_j      = np.log(np.clip(_emp_col('Education',            0.0), 0.0, None) + 1.0).astype(np.float32)
        edu_prim_j      = np.log(np.clip(_emp_col('Primary',              0.0), 0.0, None) + 1.0).astype(np.float32)
        edu_sec_j       = np.log(np.clip(_emp_col('Secondary',            0.0), 0.0, None) + 1.0).astype(np.float32)
        edu_prim_priv_j = np.log(_emp_col('Primary_total_priv', 0.0) + 1.0).astype(np.float32)
        edu_sec_priv_j  = np.log(_emp_col('Secondary_total_priv', 0.0) + 1.0).astype(np.float32)
        auto_edu_j = _emp_col('auto_edu_acc', 0.0)
        auto_nonmand_j = _emp_col('auto_nonmand_acc', 0.0)
        # Log-transformed nonmand attractors (raw values cause gradient saturation)
        nonmand_res_j = np.log(_emp_col('residential', 0.0) + 1.0).astype(np.float32)
        nonmand_ent_j = np.log(_emp_col('Entertainment', 0.0) + 1.0).astype(np.float32)
        nonmand_com_j = log_commercial_j  # reuse already log-transformed commercial
        nonmand_retail_j = np.log(_emp_col('retail', 0.0) + 1.0).astype(np.float32)
        nonmand_totpop_j = np.log(_emp_col('TOTPOP', 0.0) + 1.0).astype(np.float32)
        nonmand_eatout_j = np.log(_emp_col('EATOUT', 0.0) + 1.0).astype(np.float32)

        _otaz_raw = pd.to_numeric(
            tour_df.get('OTAZ', tour_df.get('otaz', pd.Series(index=tour_df.index, dtype=float))).reindex(ls_cases),
            errors='coerce'
        )
        # Fastest path: caller provides pre-aligned OTAZ array (avoids dataset indexing)
        if case_otaz_override is not None:
            try:
                _ov = np.asarray(case_otaz_override, dtype=np.float32)
                if len(_ov) == n_cases:
                    _otaz_raw = pd.Series(_ov, index=ls_cases, dtype=float)
            except Exception:
                pass
        if _otaz_raw.isna().mean() > 0.5:
            try:
                _tour_raw = dt_d_batch['tour'].to_dataframe().reset_index()
                _otaz_col = next((c for c in ['OTAZ', 'otaz'] if c in _tour_raw.columns), None)
                if _otaz_col and len(_tour_raw) == n_cases:
                    _otaz_pos = pd.to_numeric(_tour_raw[_otaz_col].values, errors='coerce')
                    _otaz_raw = pd.Series(_otaz_pos, index=ls_cases)
            except Exception:
                pass
        # Flat-dataset fallback: CA has n_cases × n_zones rows; deduplicate by case
        if float(_otaz_raw.isna().mean()) > 0.5:
            try:
                _tour_raw2 = dt_d_batch['tour'].to_dataframe().reset_index()
                _otaz_col2 = next((c for c in ['OTAZ', 'otaz'] if c in _tour_raw2.columns), None)
                _case_col  = next((c for c in ['ptour_id', '_case_key', 'cases']
                                   if c in _tour_raw2.columns), None)
                if _otaz_col2 and _case_col:
                    _dedup = _tour_raw2[[_case_col, _otaz_col2]].drop_duplicates(_case_col).set_index(_case_col)
                    _otaz_raw = pd.to_numeric(_dedup[_otaz_col2].reindex(ls_cases), errors='coerce')
                elif _otaz_col2 and len(_tour_raw2) >= n_cases:
                    # Positional: take first row per block of n_zones rows
                    n_zones_approx = max(1, len(_tour_raw2) // n_cases)
                    _pos_vals = pd.to_numeric(
                        _tour_raw2[_otaz_col2].iloc[::n_zones_approx].head(n_cases).values,
                        errors='coerce')
                    _otaz_raw = pd.Series(_pos_vals, index=ls_cases[:len(_pos_vals)])
            except Exception:
                pass
        if float(_otaz_raw.isna().mean()) > 0.5:
            return None
        case_otaz = _otaz_raw.fillna(int(_otaz_raw.dropna().median())).astype(int).to_numpy(dtype=np.int32)

        dist_ij = np.full((n_cases, n_zones), 30.0, dtype=np.float32)
        crash_ij = np.zeros((n_cases, n_zones), dtype=np.float32)
        avail_ij = np.ones((n_cases, n_zones), dtype=bool)
        _od_loaded = False
        # Fast path: flat Dataset stores OD matrices directly as (n_cases, n_zones)
        if _is_flat_ds:
            try:
                _v = np.asarray(dt_d_batch['DIST'].values, dtype=np.float32)
                if _v.shape == (n_cases, n_zones):
                    dist_ij = _v
                    _od_loaded = True
            except Exception:
                pass
            try:
                _v = np.asarray(dt_d_batch['CRASH'].values, dtype=np.float32)
                if _v.shape == (n_cases, n_zones):
                    crash_ij = _v
            except Exception:
                pass
            try:
                _v = np.asarray(dt_d_batch['SOVTOLL_TIME'].values, dtype=np.float32)
                if _v.shape == (n_cases, n_zones):
                    avail_ij = np.isfinite(_v) & (_v < 150.0)
            except Exception:
                pass

        # skim_df fallback - used when Dataset has sampled dims (n_cases, 60)
        # not matching (n_cases, n_zones=329).  Pivot DIST and CRASH for the
        # unique OTAZs present in this batch.
        if not _od_loaded and skim_df is not None and len(skim_df) > 0:
            try:
                _sk = skim_df[skim_df['DTAZ'].astype(int).isin(zone_arr.tolist())]
                _dist_col = 'DIST' if 'DIST' in _sk.columns else next(
                    (c for c in ['SOVTOLL_DIST__AM', 'HOV2TOLL_DIST__AM'] if c in _sk.columns), None)
                _crash_col = 'CRASH' if 'CRASH' in _sk.columns else None
                if _dist_col:
                    _piv = (_sk.pivot_table(index='OTAZ', columns='DTAZ',
                                            values=_dist_col, aggfunc='first')
                            .reindex(columns=zone_arr.tolist()).fillna(30.0))
                    for i, otaz in enumerate(case_otaz):
                        if int(otaz) in _piv.index:
                            dist_ij[i] = _piv.loc[int(otaz)].values.astype(np.float32)
                    _od_loaded = True
                if _crash_col:
                    _cpiv = (_sk.pivot_table(index='OTAZ', columns='DTAZ',
                                             values=_crash_col, aggfunc='first')
                             .reindex(columns=zone_arr.tolist()).fillna(0.0))
                    for i, otaz in enumerate(case_otaz):
                        if int(otaz) in _cpiv.index:
                            crash_ij[i] = _cpiv.loc[int(otaz)].values.astype(np.float32)
            except Exception as _sk_e:
                logger.debug(f"_dest_utility_numpy: skim_df OD fallback failed: {_sk_e}")

        if not _od_loaded:
            # DataTree fallback (legacy non-flat datasets)
            try:
                od_df = dt_d_batch['od'].to_dataframe().reset_index()
                otaz_idx = next((c for c in ['otaz', 'OTAZ'] if c in od_df.columns), None)
                dtaz_idx = next((c for c in ['dtaz', 'DTAZ'] if c in od_df.columns), None)
                dist_col = 'DIST' if 'DIST' in od_df.columns else 'SOVTOLL_DIST__AM' if 'SOVTOLL_DIST__AM' in od_df.columns else None
                crash_col = 'CRASH' if 'CRASH' in od_df.columns else None
                avail_col = 'SOVTOLL_DIST__AM' if 'SOVTOLL_DIST__AM' in od_df.columns else dist_col
                if otaz_idx and dtaz_idx and dist_col:
                    dist_pivot = od_df.pivot_table(index=otaz_idx, columns=dtaz_idx, values=dist_col, aggfunc='first')
                    crash_pivot = od_df.pivot_table(index=otaz_idx, columns=dtaz_idx, values=crash_col, aggfunc='first') if crash_col else None
                    avail_pivot = od_df.pivot_table(index=otaz_idx, columns=dtaz_idx, values=avail_col, aggfunc='first') if avail_col else None
                    for i, otaz in enumerate(case_otaz):
                        if otaz in dist_pivot.index:
                            dist_ij[i] = pd.to_numeric(dist_pivot.loc[otaz].reindex(zone_arr), errors='coerce').fillna(30.0).to_numpy(dtype=np.float32)
                        if crash_pivot is not None and otaz in crash_pivot.index:
                            crash_ij[i] = pd.to_numeric(crash_pivot.loc[otaz].reindex(zone_arr), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
                        if avail_pivot is not None and otaz in avail_pivot.index:
                            avail_vals = pd.to_numeric(avail_pivot.loc[otaz].reindex(zone_arr), errors='coerce').to_numpy()
                            avail_ij[i] = np.isfinite(avail_vals) & (avail_vals < 150.0)
            except Exception:
                pass

        log_dist_ij = np.log(dist_ij + 1.0)
        log_crash_ij = np.log(np.clip(crash_ij, 0.0, None) + 1.0)

        def _person_col(col, default=0.0):
            try:
                pp = persons.copy()
                if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
                    pp = pp.rename(columns={'PERSID': 'person_id'})
                pid_col = next((c for c in ['person_id', 'PERSID'] if c in tour_df.columns), None)
                if pid_col and col in pp.columns:
                    pids = tour_df[pid_col].reindex(ls_cases)
                    vals = pd.to_numeric(pp.set_index('person_id')[col].reindex(pids), errors='coerce').fillna(default).values
                    return vals.astype(np.float32)
            except Exception:
                pass
            return np.full(n_cases, default, dtype=np.float32)

        incp_i = _person_col('incp', 0.0)
        age_i = _person_col('age5p', 0.0)
        sexp_i = _person_col('sexp', 0.0)
        nprd_i = _person_col('nprd', 0.0)
        vehrd_i = _person_col('vehrd', 1.0)
        occp_i = _person_col('occp_group', 0.0)
        indp_i = _person_col('indp', 19.0).astype(np.int32)
        study_i = _person_col('STUDYING', 0.0).astype(np.int32)
        noveh_i = (vehrd_i <= 0).astype(np.float32)
        propensity_i = _person_col('travel_propensity', 0.0)
        # person_dist_draw is 0 during estimation (set by _PERSON_BEHAVIOURAL_DEFAULTS).
        # During prediction it holds frozen N(0, s_log_dist) draws from
        # _frozen_person_dist_offsets - applied to persons DF before _dest_utility_numpy.
        person_dist_draw_i = _person_col('person_dist_draw', 0.0)

        # ── NEW coefficients ─────────────────────────────────────────────
        # c_log_sovtime / c_log_sovtime_work removed from DEST_UEC_FLAT:
        # the mode logsum already captures SOV travel time as part of the
        # mode utility.  Reading from coef with default 0 for backward compat
        # with old pkl files that may contain these params.
        c_log_sovtime = _pv('c_log_sovtime', 0.0)
        c_log_sovtime_work = _pv('c_log_sovtime_work', 0.0)
        # c_log_distwalk/bike and c_circuity removed from DEST_UEC_FLAT:
        # DISTWALK==DIST in this OMX so those terms are collinear with log_DIST_1.
        # _pv defaults to 0.0 so they contribute nothing; kept only for backward
        # compatibility with previously saved model pkl files.
        c_log_distwalk = _pv('c_log_distwalk', 0.0)
        c_log_distbike = _pv('c_log_distbike', 0.0)
        c_log_distwalk_nm = _pv('c_log_distwalk_nm', 0.0)
        c_log_distbike_nm = _pv('c_log_distbike_nm', 0.0)
        c_intrazonal = _pv('c_intrazonal', 0.0)
        c_intrazonal_edu = _pv('c_intrazonal_edu', 0.0)
        c_walk_feasible = _pv('c_walk_feasible', 0.0)
        c_bike_feasible = _pv('c_bike_feasible', 0.0)

        # ── NEW OD matrices: DISTWALK, DISTBIKE, SOVTOLL_TIME ────────────
        distwalk_ij = np.full((n_cases, n_zones), 30.0, dtype=np.float32)
        distbike_ij = np.full((n_cases, n_zones), 30.0, dtype=np.float32)
        sovtime_ij = np.full((n_cases, n_zones), 30.0, dtype=np.float32)
        # Fast path: flat Dataset stores these as (n_cases, n_zones) DataArrays
        if _is_flat_ds:
            try:
                _v = np.asarray(dt_d_batch['DISTWALK'].values, dtype=np.float32)
                if _v.shape == (n_cases, n_zones):
                    distwalk_ij = _v
            except Exception:
                pass
            try:
                _v = np.asarray(dt_d_batch['DISTBIKE'].values, dtype=np.float32)
                if _v.shape == (n_cases, n_zones):
                    distbike_ij = _v
            except Exception:
                pass
            try:
                _v = np.asarray(dt_d_batch['SOVTOLL_TIME'].values, dtype=np.float32)
                if _v.shape == (n_cases, n_zones):
                    sovtime_ij = _v
            except Exception:
                pass
        else:
            try:
                od_df_extra = dt_d_batch['od'].to_dataframe().reset_index()
                _ot = next((c for c in ['otaz', 'OTAZ'] if c in od_df_extra.columns), None)
                _dt = next((c for c in ['dtaz', 'DTAZ'] if c in od_df_extra.columns), None)
                if _ot and _dt:
                    for _mat_name, _arr_ref in [('DISTWALK', distwalk_ij), ('DISTBIKE', distbike_ij), ('SOVTOLL_TIME__AM', sovtime_ij)]:
                        if _mat_name in od_df_extra.columns:
                            _piv = od_df_extra.pivot_table(index=_ot, columns=_dt, values=_mat_name, aggfunc='first')
                            for i, otaz in enumerate(case_otaz):
                                if otaz in _piv.index:
                                    _arr_ref[i] = pd.to_numeric(_piv.loc[otaz].reindex(zone_arr), errors='coerce').fillna(30.0).to_numpy(dtype=np.float32)
            except Exception:
                pass

        # log_distwalk/distbike kept for edu routing coefficients (lines below);
        # c_log_distwalk/bike and c_circuity are all zero (DISTWALK==DIST in OMX).
        log_distwalk_ij = np.log(np.clip(distwalk_ij, 0.0, None) + 1.0)
        log_distbike_ij = np.log(np.clip(distbike_ij, 0.0, None) + 1.0)
        log_sovtime_ij = np.log(np.clip(sovtime_ij, 0.0, None) + 1.0)
        intrazonal_ij = (zone_arr[np.newaxis, :] == case_otaz[:, np.newaxis]).astype(np.float32)
        walk_feasible_ij = (distwalk_ij < WALK_MAX_KM).astype(np.float32)
        bike_feasible_ij = (distbike_ij < BIKE_MAX_KM).astype(np.float32)

        # ── NEW landuse columns ──────────────────────────────────────────
        hsenroll_j = _emp_col('HSENROLL', 0.0)

        # ── Study-level-conditioned size term for edu (prediction path) ──
        # Mirror the estimation-path fix in _dest_tree: replace the combined
        # edu size_term with a per-person conditioned version so secondary
        # students are attracted to secondary zones, not all school types.
        _pred_segment = getattr(dest_mdl, '_segment', None)
        if _pred_segment == 'edu':
            _eps_pred = np.float32(1e-6)
            _prim_j_st  = np.log(_emp_col('Primary', 0.0) + _emp_col('TOTPOP', 0.0) * 0.05 + _eps_pred)
            _prim_j_st -= float(_prim_j_st.mean())
            _sec_j_st   = np.log(_emp_col('Secondary', 0.0) + _emp_col('HSENROLL', 0.0) + _eps_pred)
            _sec_j_st  -= float(_sec_j_st.mean())
            _post_j_st  = np.log(
                np.clip(_emp_col('Education', 0.0), 0.0, None) ** 0.9
                + _emp_col('university', 0.0) + _emp_col('college', 0.0) + _eps_pred
            )
            _post_j_st -= float(_post_j_st.mean())
            _study_i_np = np.asarray(study_i, dtype=np.int32)
            _cond_size_ij = (
                (_study_i_np == 1).astype(np.float32)[:, np.newaxis] * _prim_j_st[np.newaxis, :]
                + (_study_i_np == 2).astype(np.float32)[:, np.newaxis] * _sec_j_st[np.newaxis, :]
                + (_study_i_np >= 3).astype(np.float32)[:, np.newaxis] * _post_j_st[np.newaxis, :]
            )
            _size_term_term = c_size_term * _cond_size_ij
        else:
            _size_term_term = c_size_term * size_term_j[np.newaxis, :]

        ls_np = np.asarray(ls_batch, dtype=np.float32)
        V = (
            c_log_dist * log_dist_ij
            + c_dist * dist_ij
            + c_crash * log_crash_ij
            + c_logsum * ls_np
            + _size_term_term
            + c_aggl * acdc_aggl_j[np.newaxis, :]
            + c_comp * acdc_comp_j[np.newaxis, :]
            # Per-type ACDC agglomeration (only nonzero themes contribute).
            + sum(
                (c_aggl_themed[_t] * acdc_aggl_themed_j[_t][np.newaxis, :]
                 for _t in _ACDC_THEMES if c_aggl_themed[_t] != 0.0),
                np.zeros((1, len(zone_arr)), dtype=np.float32),
            )
            + c_auto_work * auto_work_j[np.newaxis, :]
            + c_auto_shop * auto_shop_j[np.newaxis, :]
            + c_walk_acc * walk_acc_j[np.newaxis, :]
            + c_transit_acc * transit_j[np.newaxis, :]
        )
        V += c_noveh_walk * noveh_i[:, np.newaxis] * walk_acc_j[np.newaxis, :]
        V += c_noveh_trn * noveh_i[:, np.newaxis] * transit_j[np.newaxis, :]
        V += c_incp_dist * incp_i[:, np.newaxis] * log_dist_ij
        V += c_age_dist * age_i[:, np.newaxis] * log_dist_ij
        V += c_female_dist * (sexp_i[:, np.newaxis] == 2).astype(np.float32) * log_dist_ij
        V += c_nprd_dist * nprd_i[:, np.newaxis] * log_dist_ij
        V += c_veh_dist * vehrd_i[:, np.newaxis] * log_dist_ij
        V += c_occp_dist * (occp_i[:, np.newaxis] == 1).astype(np.float32) * log_dist_ij
        V += c_indp_fin * np.isin(indp_i, [10, 11, 12, 13]).astype(np.float32)[:, np.newaxis] * fpsempn_j[np.newaxis, :]
        V += c_indp_hlth * np.isin(indp_i, [16, 17]).astype(np.float32)[:, np.newaxis] * herempn_j[np.newaxis, :]
        V += c_indp_retl * np.isin(indp_i, [7, 8]).astype(np.float32)[:, np.newaxis] * retempn_j[np.newaxis, :]
        # c_totemp_work removed (collinear with size_term)
        V += c_log_commercial_work * log_commercial_j[np.newaxis, :]
        V += c_edu_post * np.isin(study_i, [3, 4]).astype(np.float32)[:, np.newaxis] * edu_post_j[np.newaxis, :]
        V += c_edu_prim * (study_i == 1).astype(np.float32)[:, np.newaxis] * edu_prim_j[np.newaxis, :]
        V += c_edu_sec * (study_i == 2).astype(np.float32)[:, np.newaxis] * edu_sec_j[np.newaxis, :]
        V += c_edu_prim_priv * (study_i == 1).astype(np.float32)[:, np.newaxis] * edu_prim_priv_j[np.newaxis, :]
        V += c_edu_sec_priv * (study_i == 2).astype(np.float32)[:, np.newaxis] * edu_sec_priv_j[np.newaxis, :]
        V += c_auto_edu * auto_edu_j[np.newaxis, :]
        V += c_auto_nonmand * auto_nonmand_j[np.newaxis, :]
        V += c_nonmand_res * nonmand_res_j[np.newaxis, :]
        V += c_nonmand_ent * nonmand_ent_j[np.newaxis, :]
        V += c_nonmand_com * nonmand_com_j[np.newaxis, :]
        V += c_nonmand_retail * nonmand_retail_j[np.newaxis, :]
        V += c_nonmand_totpop * nonmand_totpop_j[np.newaxis, :]
        V += c_nonmand_eatout * nonmand_eatout_j[np.newaxis, :]
        # Travel propensity × log-distance: more-mobile persons less deterred by distance.
        # Coefficient estimated from HHTS; applied identically to synth population.
        V += c_propensity_dist * propensity_i[:, np.newaxis] * log_dist_ij
        # Frozen person-specific distance draw (N(0, s_log_dist), seeded by person_id MD5).
        # Zero during estimation (no bias); non-zero during prediction → heterogeneous
        # distance sensitivities that are reproducible across runs and persons.
        V += person_dist_draw_i[:, np.newaxis] * log_dist_ij
        # ── NEW utility terms ────────────────────────────────────────────
        # Travel time impedance (SOV time from skim).
        # c_log_sovtime is for edu/nonmand; c_log_sovtime_work for work.
        # Both reference log_sovtime_ij.  In prediction the segment model
        # only has one of them estimated (other is 0.0 via _pv default).
        V += c_log_sovtime * log_sovtime_ij
        V += c_log_sovtime_work * log_sovtime_ij
        # Walk/bike distance terms: c_log_distwalk/bike default to 0.0 because
        # DISTWALK==DIST in this OMX (all road distance, no active-mode network).
        # Kept for backward compatibility with saved model pkl files.
        V += c_log_distwalk * log_distwalk_ij * walk_feasible_ij
        V += c_log_distbike * log_distbike_ij * bike_feasible_ij
        V += c_log_distwalk_nm * log_distwalk_ij * walk_feasible_ij
        V += c_log_distbike_nm * log_distbike_ij * bike_feasible_ij
        # Intrazonal preference
        V += c_intrazonal * intrazonal_ij
        V += c_intrazonal_edu * intrazonal_ij * (study_i[:, np.newaxis] > 0).astype(np.float32)
        # Walk/bike feasibility
        V += c_walk_feasible * walk_feasible_ij
        V += c_bike_feasible * bike_feasible_ij
        V = _apply_destination_calibration(V, case_otaz, zone_arr, getattr(dest_mdl, '_od_calibration', {}) or {})
        # Apply zone-level shadow prices (marginal destination corrections)
        _sp = getattr(dest_mdl, '_zone_shadow_prices', None)
        if _sp is not None and len(_sp) == V.shape[1]:
            V += np.asarray(_sp, dtype=np.float32)[np.newaxis, :]
        return np.where(avail_ij, V, -np.inf)
    except Exception as e:
        logger.warning(f"_dest_utility_numpy failed: {e}")
        return None


def _dest_probs_numpy(dest_mdl, dt_d_batch, ls_batch, zone_ids, persons,
                      include_logsum: bool = True,
                      case_otaz_override=None,
                      landuse: pd.DataFrame = None,
                      skim_df: pd.DataFrame = None):
    """Compute destination probabilities from the numpy utility path."""
    try:
        V = _dest_utility_numpy(dest_mdl, dt_d_batch, ls_batch, zone_ids, persons,
                                include_logsum=include_logsum,
                                case_otaz_override=case_otaz_override,
                                landuse=landuse,
                                skim_df=skim_df)
        if V is None:
            return None
        probs = _softmax_from_utilities(V)
        logger.debug(f"_dest_probs_numpy improved: {probs.shape[0]}x{probs.shape[1]} mean row std={float(probs.std(axis=1).mean()):.4f}")
        return probs
    except Exception as e:
        logger.warning(f"_dest_probs_numpy improved failed: {e}")
        return None


def _activitysim_make_sample_choices(probs_np: np.ndarray, zone_ids, sample_size: int, rng: np.random.Generator) -> pd.DataFrame:
    """ActivitySim-style sampling with replacement collapsed to pick_count."""
    zone_arr = np.asarray(zone_ids, dtype=int)
    if probs_np is None or probs_np.size == 0 or len(zone_arr) == 0:
        return pd.DataFrame(columns=['chooser_row', 'alt_dest', 'prob', 'pick_count'])
    sample_size = max(0, min(int(sample_size), len(zone_arr)))
    if sample_size == 0:
        return pd.DataFrame(columns=['chooser_row', 'alt_dest', 'prob', 'pick_count'])
    rows = []
    for i in range(probs_np.shape[0]):
        p = np.asarray(probs_np[i], dtype=np.float64)
        p = np.where(np.isfinite(p) & (p >= 0), p, 0.0)
        total = p.sum()
        if total <= 0:
            continue
        p /= total
        draws = rng.choice(len(zone_arr), size=sample_size, replace=True, p=p)
        picked, counts = np.unique(draws, return_counts=True)
        for pos, count in zip(picked, counts):
            rows.append({'chooser_row': i, 'alt_dest': int(zone_arr[pos]), 'prob': float(max(p[pos], 1e-300)), 'pick_count': int(count)})
    return pd.DataFrame(rows)


def _polaris_destination_time_matrix(
    case_origins, zone_ids, skim_lookup, depart_times=None
) -> np.ndarray:
    """Return case-by-zone auto travel times for Polaris stratification."""
    origins = np.asarray(case_origins, dtype=int)
    zones = np.asarray(zone_ids, dtype=int)
    times = np.full((len(origins), len(zones)), np.nan, dtype=np.float64)
    if not skim_lookup:
        return times
    if depart_times is None:
        depart_times = np.full(len(origins), 480.0, dtype=float)
    else:
        depart_times = np.asarray(depart_times, dtype=float)

    def _period_key(minutes):
        try:
            minutes = float(minutes)
        except (TypeError, ValueError):
            minutes = 480.0
        if minutes < 540:
            return 'AM'
        if minutes < 900:
            return 'MD'
        if minutes < 1080:
            return 'PM'
        return 'EV'

    for i, origin in enumerate(origins):
        period = _period_key(depart_times[i] if i < len(depart_times) else 480.0)
        for j, zone in enumerate(zones):
            values = skim_lookup.get((int(origin), int(zone)), {})
            value = values.get(f'SOVTOLL_TIME__{period}')
            if value is None:
                value = values.get('SOVTOLL_TIME__AM')
            if value is None:
                value = values.get('tt_am', values.get('tt_avg'))
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value >= 0:
                times[i, j] = value
    return times


def _polaris_destination_employment(landuse: pd.DataFrame, zone_ids) -> np.ndarray:
    """Return log(1 + employment) aligned to the destination zone order."""
    zones = np.asarray(zone_ids, dtype=int)
    try:
        zone_col = 'ORDER' if 'ORDER' in landuse.columns else landuse.columns[0]
        lu = landuse.copy()
        lu[zone_col] = pd.to_numeric(lu[zone_col], errors='coerce')
        if 'TOTEMP' not in lu.columns:
            emp_cols = [c for c in ['FPSEMPN', 'HEREMPN', 'RETEMPN',
                                    'MWTEMPN', 'OTHEMP', 'AGREMPN']
                        if c in lu.columns]
            if emp_cols:
                lu['TOTEMP'] = sum(
                    pd.to_numeric(lu[c], errors='coerce').fillna(0.0)
                    for c in emp_cols
                )
        employment = pd.to_numeric(
            lu.set_index(zone_col).get('TOTEMP', pd.Series(dtype=float)),
            errors='coerce'
        ).fillna(0.0).clip(lower=0.0)
        values = employment.reindex(zones).fillna(0.0).to_numpy(dtype=float)
        return np.log1p(values)
    except Exception as exc:
        logger.debug(f'_polaris_destination_employment failed: {exc}')
        return np.zeros(len(zones), dtype=float)


def _polaris_stratified_sample_choices(
    zone_ids,
    travel_time: np.ndarray,
    employment: np.ndarray,
    feasible: np.ndarray,
    rng: np.random.Generator,
    n_strata: int = POLARIS_DEST_STRATA,
    alternatives_per_stratum: int = POLARIS_DEST_ALTS_PER_STRATUM,
) -> pd.DataFrame:
    """Sample a Polaris-style 2x2 travel-time/employment choice set.

    The four strata are low/high travel time crossed with low/high employment.
    Alternatives are sampled without replacement within each stratum.  ``prob``
    stores the inclusion probability under the equal-stratum sampling design so
    the existing sampled-choice simulator can apply a log correction.
    """
    zones = np.asarray(zone_ids, dtype=int)
    travel_time = np.asarray(travel_time, dtype=float)
    employment = np.asarray(employment, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    if n_strata != 4 or travel_time.ndim != 2 or travel_time.shape[1] != len(zones):
        return pd.DataFrame(columns=['chooser_row', 'alt_dest', 'prob', 'pick_count'])
    emp_cut = float(np.nanmedian(employment)) if len(employment) else 0.0
    rows = []
    for chooser, row_time in enumerate(travel_time):
        valid = feasible[chooser].copy()
        finite_time = np.isfinite(row_time)
        if not valid.any():
            valid = finite_time.copy()
        if not valid.any():
            valid = np.ones(len(zones), dtype=bool)
        time_cut = float(np.nanmedian(row_time[finite_time])) if finite_time.any() else 0.0
        time_high = np.where(finite_time, row_time >= time_cut, False)
        emp_high = employment >= emp_cut
        strata = time_high.astype(int) * 2 + emp_high.astype(int)
        inclusion_prob = np.zeros(len(zones), dtype=float)
        selected = []
        for stratum in range(4):
            pool = np.flatnonzero(valid & (strata == stratum))
            if len(pool) == 0:
                pool = np.flatnonzero(valid)
            if len(pool) == 0:
                continue
            draw_count = min(max(int(alternatives_per_stratum), 1), len(pool))
            picked = rng.choice(pool, size=draw_count, replace=False)
            inclusion_prob[pool] += (1.0 / 4.0) * draw_count / len(pool)
            selected.extend(int(pos) for pos in np.asarray(picked).ravel())
        if not selected:
            continue
        picked, counts = np.unique(np.asarray(selected, dtype=int), return_counts=True)
        for pos, count in zip(picked, counts):
            probability = max(float(inclusion_prob[pos]), 1e-300)
            rows.append({
                'chooser_row': chooser,
                'alt_dest': int(zones[pos]),
                'prob': probability,
                'pick_count': int(count),
            })
    return pd.DataFrame(rows)


def _polaris_space_time_mask(
    current_rows: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    zone_ids,
    origin_zones,
    skim_lookup,
) -> np.ndarray:
    """Build a reachable-zone mask from scheduled activity windows.

    Known work/school destinations provide intermediate anchors.  Missing
    anchors fall back to the person's home/origin zone, making the constraint
    conservative for first/last activities while still enforcing both sides of
    a work-to-home non-routine stop.
    """
    zones = np.asarray(zone_ids, dtype=int)
    origins = np.asarray(origin_zones, dtype=int)
    n_cases = len(current_rows)
    mask = np.ones((n_cases, len(zones)), dtype=bool)
    if n_cases == 0 or not skim_lookup:
        return mask

    schedule = schedule_rows.copy()
    if 'person_id' not in current_rows.columns or 'person_id' not in schedule.columns:
        return mask
    schedule['_start_'] = pd.to_numeric(schedule.get('STARTIME', 0), errors='coerce').fillna(0.0)
    schedule['_end_'] = pd.to_numeric(
        schedule.get('ENDTIME', schedule['_start_']), errors='coerce'
    ).fillna(schedule['_start_'])
    schedule_segment = schedule.get(
        'segment', pd.Series('', index=schedule.index, dtype=object)
    )
    schedule_assigned = schedule.get(
        '_polaris_assigned_dest_', pd.Series(False, index=schedule.index)
    )
    schedule['_known_dest_'] = (
        (schedule_segment.isin(['work', 'edu']) | schedule_assigned.astype(bool))
        & pd.to_numeric(schedule.get('DTAZ', 0), errors='coerce').fillna(0).gt(0)
    )
    schedule['_person_'] = schedule['person_id'].astype(str)
    current = current_rows.copy()
    current['_start_'] = pd.to_numeric(current.get('STARTIME', 0), errors='coerce').fillna(0.0)
    current['_end_'] = pd.to_numeric(
        current.get('ENDTIME', current['_start_']), errors='coerce'
    ).fillna(current['_start_'])
    current['_person_'] = current['person_id'].astype(str)

    for i, (_, row) in enumerate(current.iterrows()):
        same_person = schedule[schedule['_person_'] == row['_person_']]
        prior = same_person[
            same_person['_known_dest_'] & (same_person['_start_'] < row['_start_'])
        ].sort_values('_start_').tail(1)
        following = same_person[
            same_person['_known_dest_'] & (same_person['_start_'] > row['_start_'])
        ].sort_values('_start_').head(1)
        home = int(origins[i]) if i < len(origins) else 0
        if not prior.empty:
            prev_zone = int(pd.to_numeric(prior.iloc[0]['DTAZ'], errors='coerce'))
            prev_end = float(prior.iloc[0]['_end_'])
        else:
            prev_zone = home
            prev_end = POLARIS_DAY_START_MIN
        if not following.empty:
            next_zone = int(pd.to_numeric(following.iloc[0]['DTAZ'], errors='coerce'))
            next_start = float(following.iloc[0]['_start_'])
        else:
            next_zone = home
            next_start = POLARIS_DAY_END_MIN

        prev_time = _polaris_destination_time_matrix(
            [prev_zone], zones, skim_lookup, [prev_end]
        )[0]
        next_time = _polaris_destination_time_matrix(
            [next_zone], zones, skim_lookup, [row['_end_']]
        )[0]
        outbound_budget = max(float(row['_start_']) - prev_end, 0.0)
        inbound_budget = max(next_start - float(row['_end_']), 0.0)
        prev_known = np.isfinite(prev_time)
        next_known = np.isfinite(next_time)
        mask[i] = ((~prev_known) | (prev_time <= outbound_budget + 1e-6)) & (
            (~next_known) | (next_time <= inbound_budget + 1e-6)
        )
        if not mask[i].any():
            mask[i] = np.ones(len(zones), dtype=bool)
    return mask


def _polaris_sequential_destination_draw(
    utilities: np.ndarray,
    current_rows: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    zone_ids,
    origin_zones,
    skim_lookup,
    landuse: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw non-routine destinations in schedule order with rolling anchors."""
    zones = np.asarray(zone_ids, dtype=int)
    origins = np.asarray(origin_zones, dtype=int)
    n_cases = len(current_rows)
    chosen = np.full(n_cases, -1, dtype=int)
    valid = np.zeros(n_cases, dtype=bool)
    if n_cases == 0:
        return chosen, valid

    order_frame = current_rows.copy()
    order_frame['_polaris_start_'] = pd.to_numeric(
        order_frame.get('STARTIME', 0), errors='coerce'
    ).fillna(0.0)
    order_frame['_polaris_person_'] = order_frame['person_id'].astype(str)
    order = sorted(
        range(n_cases),
        key=lambda pos: (
            order_frame.iloc[pos]['_polaris_person_'],
            float(order_frame.iloc[pos]['_polaris_start_']),
            pos,
        ),
    )
    mutable_schedule = schedule_rows.copy()
    employment = _polaris_destination_employment(landuse, zones)
    for pos in order:
        row = current_rows.iloc[[pos]].copy()
        row_mask = _polaris_space_time_mask(
            row,
            mutable_schedule,
            zones,
            origins[pos:pos + 1],
            skim_lookup,
        )
        row_tt = _polaris_destination_time_matrix(
            origins[pos:pos + 1], zones, skim_lookup,
            [row.iloc[0].get('STARTIME', 480.0)]
        )
        row_alts = _polaris_stratified_sample_choices(
            zones, row_tt, employment, row_mask, rng
        )
        row_utility = np.where(row_mask, utilities[pos:pos + 1], -np.inf)
        if not row_alts.empty:
            row_choice, row_valid = _activitysim_simulate_sampled_destinations(
                row_utility, row_alts, zones, rng
            )
        else:
            row_choice = np.full(1, -1, dtype=int)
            row_valid = np.zeros(1, dtype=bool)
        if row_valid[0]:
            chosen[pos] = int(row_choice[0])
            valid[pos] = True
            assigned = row.iloc[0].copy()
            assigned['DTAZ'] = chosen[pos]
            assigned['_polaris_assigned_dest_'] = True
            mutable_schedule = pd.concat(
                [mutable_schedule, assigned.to_frame().T], ignore_index=True
            )
    return chosen, valid


def _activitysim_simulate_sampled_destinations(V_full: np.ndarray, sampled_alts: pd.DataFrame, zone_ids, rng: np.random.Generator):
    """Final destination simulate on sampled alternatives with sample correction."""
    n_cases = V_full.shape[0] if V_full is not None else 0
    chosen = np.full(n_cases, -1, dtype=int)
    valid = np.zeros(n_cases, dtype=bool)
    if V_full is None or sampled_alts is None or sampled_alts.empty:
        return chosen, valid
    zone_to_col = {int(z): j for j, z in enumerate(np.asarray(zone_ids, dtype=int))}
    for chooser_row, grp in sampled_alts.groupby('chooser_row', sort=False):
        chooser_row = int(chooser_row)
        cols = [zone_to_col.get(int(z), -1) for z in grp['alt_dest'].values]
        mask = np.array([c >= 0 for c in cols], dtype=bool)
        if not mask.any():
            continue
        cols = np.asarray(cols, dtype=int)[mask]
        grp_valid = grp.loc[mask].copy()
        util = np.asarray(V_full[chooser_row, cols], dtype=np.float64)
        correction = np.log(np.clip(grp_valid['pick_count'].to_numpy(dtype=np.float64), 1.0, None) / np.clip(grp_valid['prob'].to_numpy(dtype=np.float64), 1e-300, None))
        util = util + correction
        if not np.isfinite(util).any():
            continue
        probs = _softmax_from_utilities(util.reshape(1, -1))[0]
        pick = int(rng.choice(len(grp_valid), p=probs))
        chosen[chooser_row] = int(grp_valid.iloc[pick]['alt_dest'])
        valid[chooser_row] = True
    return chosen, valid

def _coef_dict_from_model(model, coef_override: dict = None) -> dict:
    """
    Extract coefficient {name: float} dict from a Larch model robustly.

    Priority:
        1. coef_override (plain dict passed from models[seg]['coef'])
        2. model._coef_cache (injected at save_models() or _estimate_mixed_logit)
        3. model._final_pf (snapshot saved by _estimate after optimisation)
        4. model.pf property (works only on fresh/mangled models)
        5. model.__dict__ brute-force scan for parameter tables
    """
    if coef_override:
        return coef_override

    if hasattr(model, '_coef_cache') and model._coef_cache:
        return model._coef_cache

    # _final_pf is a plain DataFrame snapshot saved by _estimate() after
    # maximisation - it survives JAX JIT compilation that corrupts model.pf.
    if hasattr(model, '_final_pf') and isinstance(model._final_pf, pd.DataFrame):
        pf = model._final_pf
        if 'value' in pf.columns:
            try:
                coef = {str(name): float(pf.loc[name, 'value'])
                        for name in pf.index
                        if not np.isnan(float(pf.loc[name, 'value']))}
                if coef:
                    model._coef_cache = coef   # cache for future calls
                    return coef
            except Exception:
                pass

    # Try .pf - on loaded models Larch may need mangle()/unmangle() first
    # to rebuild the internal ._mangled attribute before pf is accessible.
    try:
        pf = model.pf
        coef = {name: float(pf.loc[name, 'value'])
                for name in pf.index
                if not np.isnan(float(pf.loc[name, 'value']))}
        if coef:
            return coef
    except AttributeError:
        # '_mangled' attribute missing → try mangle() to rebuild internals
        try:
            model.mangle()
            model.unmangle()
            pf = model.pf
            coef = {name: float(pf.loc[name, 'value'])
                    for name in pf.index
                    if not np.isnan(float(pf.loc[name, 'value']))}
            # Cache result so we don't need to mangle again
            model._coef_cache = coef
            return coef
        except Exception:
            pass
    except (TypeError, KeyError):
        pass

    # Last resort: scan __dict__ for anything that looks like a parameter frame
    for attr_val in model.__dict__.values():
        if isinstance(attr_val, pd.DataFrame) and 'value' in attr_val.columns:
            try:
                result = {str(name): float(attr_val.loc[name, 'value'])
                          for name in attr_val.index
                          if not np.isnan(float(attr_val.loc[name, 'value']))}
                if result:
                    return result
            except Exception:
                continue

    return {}


def _sanitize_mu(mu, name=''):
    """
    Guard a fitted nesting parameter before it is used for prediction.

    The Larch NL fit bounds each Mu_* to [0.10, 1.0]. A value sitting at (or
    hugging) the 0.10 floor is an identification failure, not a behavioural
    finding - job 23880279's fits had Mu_Car=0.10 (edu, at bound) and
    Mu_Motor=0.16 with |t| < 0.4 across the board. A tiny mu turns the nest's
    conditional split into a near-argmax, handing ~100% of the nest's share
    to whichever alternative has the higher utility (SR 96% / DA 2.6% for
    nonmand). Since only point estimates survive into mode_coef (no SEs),
    the proxy for "unidentified" is proximity to the floor: anything below
    0.25 is treated as flat (mu=1), which degrades gracefully to the MNL
    within-nest split instead of amplifying a garbage estimate.
    """
    try:
        mu = float(mu)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(mu) or mu <= 0.25:
        if mu <= 0.25 and name:
            logger.debug(f"_sanitize_mu: {name}={mu:.3f} at/near its bound - "
                         "treating as unidentified (mu=1.0) for prediction")
        return 1.0
    return min(mu, 1.0)


def _nested_mode_probs(V_DA, V_SR, V_WK, V_BK, V_TR,
                       mu_car=1.0, mu_motor=1.0, mu_nonmotor=1.0, xp=np):
    """
    GEV nested-logit choice probabilities for the 5-mode tree estimated by
    _build_mode_model (stage5_destination_choice.py:4336-4341):

        Root (mu_root=1, fixed)
          Motor (mu_motor)
            Car (mu_car): DA, SR
            Transit
          NonMotor (mu_nonmotor): Walk, Bike

    _mode_probs_numpy and _compute_frozen_probs used to take these same
    utilities straight into a single flat 5-way softmax, which is only
    correct when mu_car=mu_motor=mu_nonmotor=1 (see the degenerate-case
    derivation in the fix commit). Reusing coefficients fit under the real
    nested structure inside a flat softmax systematically misallocates
    share away from whichever alternatives sit in a tightly-scaled nest
    (SR inside Car/Motor, Bike/Walk inside NonMotor) — this was collapsing
    EDU-tour SR to ~0% and NONMAND-tour bike/transit to ~0% at prediction
    time despite the underlying models fitting well (rho2 0.45-0.55).

    When a mu isn't found in the coefficient source it defaults to 1.0,
    which makes the nest a no-op and the formula below reduce exactly to
    the old flat softmax — i.e. this is a strict generalisation, not a
    behaviour change, when nesting parameters are unavailable.

    Utilities carrying a hard-unavailability sentinel (<= -1e8) are
    supported at any level; an all-unavailable nest degrades to a very
    low (but finite) inclusive value rather than -inf, so it cleanly
    cascades to near-zero probability without NaNs.

    Returns (n, 5) probs in DA, SR, Walk, Bike, Transit column order,
    matching the flat softmax's original output layout.
    """
    def _logsum(v_list, mu):
        V = xp.stack(v_list, axis=1)                        # (n, k)
        Vm = V / mu
        row_max = xp.max(xp.where(V > -1e8, Vm, -xp.inf), axis=1, keepdims=True)
        row_max = xp.where(xp.isfinite(row_max), row_max, 0.0)
        exp_v = xp.exp(xp.clip(Vm - row_max, -500, 0))
        totals = exp_v.sum(axis=1)
        totals_safe = xp.where(totals > 0, totals, 1e-300)
        cond_probs = exp_v / totals_safe[:, None]
        iv = mu * (xp.log(totals_safe) + row_max[:, 0])     # inclusive value (utils)
        return cond_probs, iv

    car_cond,      iv_car      = _logsum([V_DA, V_SR], mu_car)
    nonmotor_cond, iv_nonmotor = _logsum([V_WK, V_BK], mu_nonmotor)
    motor_cond,    iv_motor    = _logsum([iv_car, V_TR], mu_motor)
    root_cond,     _iv_root    = _logsum([iv_motor, iv_nonmotor], 1.0)

    p_motor, p_nonmotor = root_cond[:, 0], root_cond[:, 1]
    p_car_motor, p_tr_motor = motor_cond[:, 0], motor_cond[:, 1]
    p_da_car, p_sr_car = car_cond[:, 0], car_cond[:, 1]
    p_wk_nm, p_bk_nm = nonmotor_cond[:, 0], nonmotor_cond[:, 1]

    p_DA = p_motor * p_car_motor * p_da_car
    p_SR = p_motor * p_car_motor * p_sr_car
    p_TR = p_motor * p_tr_motor
    p_WK = p_nonmotor * p_wk_nm
    p_BK = p_nonmotor * p_bk_nm

    return xp.stack([p_DA, p_SR, p_WK, p_BK, p_TR], axis=1)


def _mode_probs_numpy(mode_mdl, tour_rows, persons, skims, skim_lookup=None,
                      coef_override: dict = None):
    """
    Compute mode choice probabilities directly from fitted coefficients.

    Uses coef_override (plain Python dict from models[seg]['coef']) first,
    then tries _coef_cache, then model.pf - never crashes on loaded models.

    Modes: DA=1, Other=2, Walk=3, Bike=4, Transit=5
    Returns np.ndarray shape (n_cases, 5) or None on failure.

    ActivitySim alignment
    ---------------------
    - Availability: DA blocked when CARLICENCE=0; Walk blocked when
      walk_avail=0; Bike blocked when bike_avail=0; Transit blocked when
      transit IVT is missing/implausibly high (>= TRANSIT_IVT_UNAVAIL_MIN).
    - When ALL modes are unavailable for a chooser the row falls back to
      the DA/Transit pair (least-bad options) rather than returning uniform.
    - Coefficient source is logged so DA-dominance bugs can be diagnosed.
    """
    try:
        xp = jnp if JAX_AVAILABLE else np
        _cache = _coef_dict_from_model(mode_mdl, coef_override)

        # ── Coefficient source diagnostic ────────────────────────────────
        if not _cache:
            logger.warning("_mode_probs_numpy: no coefficient source found - "
                           "using hard-coded defaults.  Check that mode_coef "
                           "was extracted at estimation time and survives pickle.")
        else:
            _src = ('coef_override' if coef_override else
                    '_coef_cache'   if (hasattr(mode_mdl, '_coef_cache') and mode_mdl._coef_cache) else
                    'model.pf')
            logger.debug(f"_mode_probs_numpy: using coefficients from {_src} "
                         f"({len(_cache)} params: {list(_cache.keys())[:8]}...)")

        def _pv(name, default=0.0):
            v = _cache.get(name) if _cache else None
            if v is None:
                return default
            try:
                fv = float(v)
                return fv if not np.isnan(fv) else default
            except Exception:
                return default

        # ── 1. Fitted coefficients ────────────────────────────────────────
        # Defaults calibrated to approximate SEQ observed mode shares when
        # estimated coefficients fail to load:
        #   DA ~60%, SR ~12%, Transit ~10%, Walk ~10%, Bike ~8%
        # Previously walk=-4.0, bike=-4.5 produced ~0% walk/bike.
        # ASC_Walk/Bike raised to -1.0/-1.5 so utility function (not ASC)
        # governs suppression at long distances.
        c_ivt_auto     = _pv('c_ivt_auto',      -1.20)
        c_ivt_transit  = _pv('c_ivt_transit',   -0.90)
        c_walk_time    = _pv('c_walk_time',      -1.50)
        c_bike_time    = _pv('c_bike_time',      -1.00)
        c_crash_veh    = _pv('c_crash_veh',       0.00)
        c_veh_da       = _pv('c_veh_da',          0.30)
        c_veh_sr       = _pv('c_veh_sr',          0.10)
        c_noveh_transit= _pv('c_noveh_transit',   2.00)
        c_noveh_walk   = _pv('c_noveh_walk',      1.50)
        # c_licence_da removed - CARLICENCE is a hard availability constraint
        c_incp_da      = _pv('c_incp_da',         0.010)
        c_incp_sr      = _pv('c_incp_sr',         0.005)
        c_incp_transit = _pv('c_incp_transit',   -0.015)
        # c_incp_active must mirror MODE_UEC {"Walk": None, "Bike": None}.
        # Without this term, high-income persons get c_incp_da/sr/transit ≈ −1.3
        # applied to all motorised modes, crushing DA/SR/Transit utility while
        # Walk and Bike remain unpenalised - producing nonsensical Bike=74% splits.
        c_incp_active  = _pv('c_incp_active',    -0.010)
        c_age_walk     = _pv('c_age_walk',        -0.08)  # age5p bands; slight deterrence
        c_age_bike     = _pv('c_age_bike',        -0.12)
        c_young_bike   = _pv('c_young_bike',       0.50)
        c_joint_sr     = _pv('c_joint_sr',         0.30)
        c_female_transit=_pv('c_female_transit',   0.30)
        c_female_walk  = _pv('c_female_walk',      0.20)
        c_peak_da      = _pv('c_peak_da',          0.30)
        # ── person-type shifts (ActivitySim pattern) ────────────────────────
        c_worker_da    = _pv('c_worker_da',        0.20)  # FT/PT worker → more DA
        c_student_tr   = _pv('c_student_tr',       0.40)  # student → more transit
        c_inc_high_da  = _pv('c_inc_high_da',      0.015) # high income → DA
        c_inc_low_tr   = _pv('c_inc_low_tr',      -0.020) # low income → transit
        # ── ASC values - relative to DA (ASC_DA = 0) ─────────────────────
        # Raised significantly from -4.0/-4.5 which produced effectively 0%
        # walk/bike.  New values allow the distance utility to govern mode
        # suppression at longer distances.
        asc_sr         = _pv('ASC_SR',            -0.50)
        asc_walk       = _pv('ASC_Walk',           -1.00)  # was -4.00
        asc_bike       = _pv('ASC_Bike',           -1.50)  # was -4.50
        asc_transit    = _pv('ASC_Transit',        -0.50)  # was -1.50

        # ── 2. Get OTAZ and DTAZ per case ─────────────────────────────────
        n = len(tour_rows)
        def _col(df, *names, default=0.0):
            for nm in names:
                if nm in df.columns:
                    return pd.to_numeric(df[nm], errors='coerce').fillna(default).values.astype(np.float32)
            return np.full(n, default, dtype=np.float32)

        otaz = _col(tour_rows, 'OTAZ', default=1).astype(np.int32)
        dtaz = _col(tour_rows, 'DTAZ', default=1).astype(np.int32)

        # ── 3. Read skim values per OD pair ──────────────────────────────
        # Multi-resolution: walk/bike use SA1-origin × SA2-destination skims
        # (finer spatial resolution within each SA2 home zone) when available.
        # Auto/transit use SA2→SA2 period-matched skims as before.
        # Falls back gracefully when SA1 skims aren't built yet.
        ivt_auto    = np.full(n, 30.0, dtype=np.float32)
        ivt_transit = np.full(n, 45.0, dtype=np.float32)
        dist_walk   = np.full(n, 5.0,  dtype=np.float32)
        dist_bike   = np.full(n, 5.0,  dtype=np.float32)

        # Load fine-grained walk/bike skims once. Walk and Bike use the more
        # granular SA1-origin × SA2-destination skims by default (HOME_SA1
        # assigned by assign_sa1_to_population.py, ABS SA1 ~7.7k zones); all
        # other modes keep SA2→SA2 period-matched skims. Set
        # WALK_BIKE_FINE_ZONE=lvl4 to prefer the SEQ-STM Level4 micro-zones
        # instead (HOME_LVL4 from assign_lvl4_to_population.py — better EMME
        # alignment but not the ABS zone system). Whichever is preferred, the
        # other is the fallback.
        _fine_pref = os.environ.get('WALK_BIKE_FINE_ZONE', 'sa1').strip().lower()
        _fine_walk = None
        _home_fine = None
        _fine_key  = None

        def _try_sa1():
            try:
                from read_skims import load_sa1_walk_skims as _load_sa1
                if 'HOME_SA1' in tour_rows.columns:
                    fw = _load_sa1()
                    if fw:
                        hf = pd.to_numeric(
                            tour_rows['HOME_SA1'], errors='coerce'
                        ).fillna(0).astype(int).values
                        return fw, hf, 'sa1_to_row'
            except Exception:
                pass
            return None, None, None

        def _try_lvl4():
            try:
                from read_skims import load_lvl4_walk_skims as _load_lvl4
                if 'HOME_LVL4' in tour_rows.columns:
                    fw = _load_lvl4()
                    if fw:
                        hf = pd.to_numeric(
                            tour_rows['HOME_LVL4'], errors='coerce'
                        ).fillna(0).astype(int).values
                        return fw, hf, 'lvl4_to_row'
            except Exception:
                pass
            return None, None, None

        _loaders = ([_try_lvl4, _try_sa1] if _fine_pref == 'lvl4'
                    else [_try_sa1, _try_lvl4])
        for _loader in _loaders:
            _fine_walk, _home_fine, _fine_key = _loader()
            if _fine_walk is not None:
                break
        crash       = np.full(n, 0.0,  dtype=np.float32)

        # _time_period code -> skim suffix (mirrors the 0..5 scheme defined
        # in _prep_hhts_tours: 0=early/EA, 1=am_peak/AM, 2=midday/MD,
        # 3=pm_peak/PM, 4=evening/EV, 5=late/EV)
        _PERIOD_SUFFIX = {0: 'EA', 1: 'AM', 2: 'MD', 3: 'PM', 4: 'EV', 5: 'EV'}
        if '_time_period' in tour_rows.columns:
            period_codes = pd.to_numeric(tour_rows['_time_period'], errors='coerce').fillna(1).astype(int).values
        else:
            period_codes = np.ones(n, dtype=int)
        period_suffix = np.array([_PERIOD_SUFFIX.get(int(p), 'AM') for p in period_codes])

        if skim_lookup:
            for i in range(n):
                key = (int(otaz[i]), int(dtaz[i]))
                sk = skim_lookup.get(key, {})
                suf = period_suffix[i]
                ivt_auto[i]    = float(sk.get(f'SOVTOLL_TIME__{suf}', sk.get('SOVTOLL_TIME__AM', 30.0)) or 30.0)
                ivt_transit[i] = float(sk.get(f'WLK_LOC_WLK_TOTIVT__{suf}', sk.get('WLK_LOC_WLK_TOTIVT__AM', 45.0)) or 45.0)
                crash[i]       = float(sk.get('CRASH', 0.0)                    or 0.0)
                # Walk/bike: use SA1/Lvl4-origin × SA2-destination skim when available.
                # DTAZ is either a 9-digit SA2 code (new_pop/507-zone) or an ORDER
                # number 1-330 (329-zone pipeline).  Use magnitude to pick the right
                # lookup: SA2 codes are always > 100,000,000; ORDER numbers are ≤ 330.
                if (_fine_walk is not None and _home_fine is not None
                        and _home_fine[i] > 0):
                    r = _fine_walk[_fine_key].get(_home_fine[i])
                    dtaz_int = int(dtaz[i])
                    if dtaz_int >= 100_000_000:  # 9-digit SA2 code
                        c = _fine_walk['sa2_to_col'].get(dtaz_int)
                    else:                         # ORDER number (329-zone system)
                        c = _fine_walk.get('order_to_col', {}).get(dtaz_int)
                    if r is not None and c is not None:
                        dist_walk[i] = float(_fine_walk['DISTWALK'][r, c]) or 5.0
                        dist_bike[i] = float(_fine_walk['DISTBIKE'][r, c]) or 5.0
                        continue
                dist_walk[i] = float(sk.get('DISTWALK', 5.0) or 5.0)
                dist_bike[i] = float(sk.get('DISTBIKE', 5.0) or 5.0)
        elif skims is not None:
            # Read matrices once and index per row
            def _mat(name, fallback):
                try:
                    if name in skims.listMatrices():
                        arr = np.array(skims[name], dtype=np.float32)
                        med = float(np.median(arr[arr > 0])) if (arr > 0).any() else 0
                        if med > 600: arr /= 60.0
                        elif med > 200: arr /= 10.0
                        return arr
                except Exception:
                    pass
                return None
            mapping = {}
            try:
                mapping = {int(k): int(v)
                           for k, v in dict(skims.mapping(skims.listMappings()[0])).items()}
            except Exception:
                pass
            # Load per-period matrices (falls back to AM-only matrices for
            # legacy OMX files that don't have the full 5-period set).
            mat_ivt_auto_by_period    = {p: _mat(f'SOVTOLL_TIME__{p}', None) for p in ('AM', 'MD', 'PM', 'EA', 'EV')}
            mat_ivt_transit_by_period = {p: _mat(f'WLK_LOC_WLK_TOTIVT__{p}', None) for p in ('AM', 'MD', 'PM', 'EA', 'EV')}
            mat_walk        = _mat('DISTWALK', None)
            mat_bike        = _mat('DISTBIKE', None)
            mat_crash       = _mat('CRASH', None)
            for i in range(n):
                oi = mapping.get(int(otaz[i]), 0)
                di = mapping.get(int(dtaz[i]), 0)
                suf = period_suffix[i]
                mat_ivt_auto = mat_ivt_auto_by_period.get(suf)
                if mat_ivt_auto is None:
                    mat_ivt_auto = mat_ivt_auto_by_period.get('AM')
                mat_ivt_transit = mat_ivt_transit_by_period.get(suf)
                if mat_ivt_transit is None:
                    mat_ivt_transit = mat_ivt_transit_by_period.get('AM')
                if mat_ivt_auto    is not None: ivt_auto[i]    = mat_ivt_auto[oi, di]
                if mat_ivt_transit is not None: ivt_transit[i] = mat_ivt_transit[oi, di]
                if mat_walk        is not None: dist_walk[i]   = mat_walk[oi, di]
                if mat_bike        is not None: dist_bike[i]   = mat_bike[oi, di]
                if mat_crash       is not None: crash[i]       = mat_crash[oi, di]

        # DISTWALK/DISTBIKE are already in km - match MODE_UEC expressions exactly:
        #   log(od.DISTWALK * 12 + 1)  (12 min/km at 5 km/h)
        #   log(od.DISTBIKE * 4 + 1)   (4 min/km at 15 km/h)
        # Transit availability: block if IVT >= threshold (no service).
        # Default skim value of 45 min is used when transit is unknown -
        # only block when it equals the sentinel fill (>= 90 min) or 0.
        TRANSIT_IVT_UNAVAIL_MIN = 90.0   # block transit when IVT ≥ this (min)
        transit_avail_raw = np.asarray(ivt_transit, dtype=np.float32)
        transit_avail = (
            (transit_avail_raw > 0) & (transit_avail_raw < TRANSIT_IVT_UNAVAIL_MIN)
        ).astype(np.float32)
        logger.debug(
            f"_mode_probs_numpy transit avail: "
            f"{int(transit_avail.sum())}/{len(transit_avail)} persons "
            f"(IVT threshold={TRANSIT_IVT_UNAVAIL_MIN} min)"
        )

        ivt_auto = xp.clip(xp.asarray(ivt_auto) / 60.0, 0.05, 5.0)   # minutes → hours
        ivt_transit = xp.clip(xp.asarray(ivt_transit) / 60.0, 0.05, 6.0)
        walk_time = xp.log(xp.clip(xp.asarray(dist_walk), 0.3, None) * 12.0 + 1.0)
        bike_time = xp.log(xp.clip(xp.asarray(dist_bike), 0.5, None) * 4.0 + 1.0)
        log_crash = xp.log(xp.asarray(crash) + 1.0)

        # ── 4. Person attributes ──────────────────────────────────────────
        pp = persons.copy()
        if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
            pp = pp.rename(columns={'PERSID': 'person_id'})
        pid_col = next(
            (c for c in ['person_id', 'PERSID'] if c in tour_rows.columns), None)
        if pid_col:
            pids = tour_rows[pid_col].values
            # Build a unique-indexed persons lookup.
            # Three-step deduplication:
            #   1. Drop rows where person_id is NaN (NaN != NaN → drop_duplicates
            #      keeps all NaN rows as "unique", producing a non-unique index)
            #   2. Drop remaining duplicate person_id rows (keep first)
            #   3. Final safety: strip any residual duplicates via index mask
            # This covers all pandas versions including CoW-enabled pandas 2.x.
            _pp_nonan = pp[pp['person_id'].notna()].copy()
            _pp_dedup = _pp_nonan.drop_duplicates(subset=['person_id'], keep='first')
            pp_idx = _pp_dedup.set_index('person_id')
            if not pp_idx.index.is_unique:
                # Last-resort: keep first occurrence of each id
                pp_idx = pp_idx[~pp_idx.index.duplicated(keep='first')]
            def _pa(col, default=0.0):
                if col in pp_idx.columns:
                    try:
                        series = pp_idx[col]
                        if not series.index.is_unique:
                            series = series[~series.index.duplicated(keep='first')]
                        lookup = series.to_dict()
                        vals = np.fromiter(
                            (float(lookup.get(pid, default)) for pid in pids),
                            dtype=np.float32,
                            count=n,
                        )
                        # A present-but-NaN attribute must also fall back to
                        # the default: NaN age5p/incp propagated into V_WK/V_BK
                        # (Walk=nan% Bike=nan% in job 23880279's logs), and the
                        # NaN shares were silently renormalised away - losing
                        # the entire ~15% walk share at prediction time.
                        return np.where(np.isfinite(vals), vals, default)
                    except Exception as _pa_e:
                        logger.debug(f"_mode_probs_numpy _pa fallback [{col}]: {_pa_e}")
                return np.full(n, default, dtype=np.float32)
        else:
            def _pa(col, default=0.0): return np.full(n, default, dtype=np.float32)

        vehrd      = _pa('vehrd',       1.0)
        licence    = _pa('CARLICENCE',  1.0)   # used for DA availability mask below
        incp       = _pa('incp',        5.0)
        age5p      = _pa('age5p',       4.0)
        sexp       = _pa('sexp',        1.0)   # 1=male 2=female

        # Hard age gate on the licence flag: CARLICENCE defaults to 1 when
        # missing/NaN, which would otherwise hand DA availability to children
        # (e.g. school-edu tours). No one under ~15 (age5p <= 3) can hold a
        # licence, whatever the data says - old-enough students with a real
        # licence keep DA.
        licence    = np.where(age5p <= 3, 0.0, licence)

        noveh      = (vehrd == 0).astype(np.float32)
        female     = (sexp == 2).astype(np.float32)
        young      = (age5p <= 5).astype(np.float32)   # age5p ≤ 5 → under ~30 yrs

        peak = np.zeros(n, dtype=np.float32)
        if '_time_period' in tour_rows.columns:
            peak = (pd.to_numeric(tour_rows['_time_period'], errors='coerce')
                    .fillna(0) == 1).values.astype(np.float32)
        joint = np.zeros(n, dtype=np.float32)
        if 'joint_travel' in tour_rows.columns:
            joint = pd.to_numeric(tour_rows['joint_travel'], errors='coerce').fillna(0).values.astype(np.float32)

        # ── Person-type heterogeneity ────────────────────────────────────
        # worker / student flags - key predictors of mode split per ActivitySim
        workstat   = _pa('WORKSTAT',    0.0)
        studying   = _pa('STUDYING',    0.0)
        is_worker  = np.clip(workstat, 0, 1).astype(np.float32)   # 1=employed
        is_student = (studying > 0).astype(np.float32)
        # income normalised: incp range 1-14; centre at 7 → negative = low income
        inc_cent   = (incp - 7.0).astype(np.float32)

        # ── Industry of employment (ANZSIC 1-digit, 'indp') ─────────────
        # Read from persons table; falls back to 20 (not employed / other).
        _indp_raw  = _pa('indp', 20.0).astype(np.int32)
        _indp_grps = np.array([INDUSTRY_MODE_GROUP.get(int(v), 'other')
                                for v in _indp_raw])
        ind_industrial = np.isin(_indp_grps, ['industrial', 'agri']).astype(np.float32)
        ind_office     = np.isin(_indp_grps, ['office', 'professional']).astype(np.float32)
        ind_public     = (_indp_grps == 'public').astype(np.float32)
        ind_health     = (_indp_grps == 'health').astype(np.float32)
        ind_retail     = np.isin(_indp_grps, ['retail', 'hospitality']).astype(np.float32)

        # ── Occupation (ANZSCO 1-digit, 'occp') ─────────────────────────
        _occp_raw    = _pa('occp', 0.0).astype(np.int32)
        occ_mgr_prof = np.isin(_occp_raw, [1, 2]).astype(np.float32)
        occ_labourer = np.isin(_occp_raw, [7, 8]).astype(np.float32)
        occ_clerical = (_occp_raw == 5).astype(np.float32)

        # ── Studying status (1=not studying, 2=ft, 3=pt, 4=other) ───────
        _stup_raw     = _pa('stup', 1.0).astype(np.int32)
        is_ft_student = (_stup_raw == 2).astype(np.float32)
        is_pt_student = np.isin(_stup_raw, [3, 4]).astype(np.float32)

        # ── MXL-estimated industry/occupation coefficients ───────────────
        # These are loaded from coef_override (mode_coef dict) if present.
        # Default to 0 (no shift) when not estimated.
        c_ind_da  = _pv('c_ind_da',  0.0)   # industrial/agri → DA
        c_off_tr  = _pv('c_off_tr',  0.0)   # office → transit
        c_pub_tr  = _pv('c_pub_tr',  0.0)   # public sector → transit
        c_pub_wk  = _pv('c_pub_wk',  0.0)   # public sector → walk
        c_hlt_tr  = _pv('c_hlt_tr',  0.0)   # health → transit
        c_ret_sr  = _pv('c_ret_sr',  0.0)   # retail/hospitality → SR
        c_mgr_da  = _pv('c_mgr_da',  0.0)   # managers → DA
        c_lab_da  = _pv('c_lab_da',  0.0)   # labourers → DA
        c_clr_tr  = _pv('c_clr_tr',  0.0)   # clerical → transit
        c_fts_tr  = _pv('c_fts_tr',  0.0)   # FT students → transit
        c_fts_wk  = _pv('c_fts_wk',  0.0)   # FT students → walk
        c_fts_bk  = _pv('c_fts_bk',  0.0)   # FT students → bike
        c_pts_tr  = _pv('c_pts_tr',  0.0)   # PT students → transit

        logger.debug(
            f"_mode_probs_numpy [{seg if 'seg' in dir() else 'N/A'}] "
            f"person attributes (n={n}): "
            f"noveh={noveh.mean():.2%}, female={female.mean():.2%}, "
            f"worker={is_worker.mean():.2%}, student={is_student.mean():.2%}, "
            f"young={young.mean():.2%}, mean_incp={incp.mean():.1f}  "
            f"industrial={ind_industrial.mean():.1%}, office={ind_office.mean():.1%}, "
            f"public={ind_public.mean():.1%}"
        )

        # ── 5. Utility per mode ───────────────────────────────────────────
        # DA (reference, ASC=0).
        # licence is a HARD availability mask (set to -inf when CARLICENCE=0)
        # rather than a positive utility term - no unlicensed person can drive.
        V_DA = (c_ivt_auto   * ivt_auto
              + c_crash_veh  * log_crash
              + c_veh_da     * vehrd
              + c_incp_da    * incp
              + c_peak_da    * peak
              + c_worker_da  * is_worker
              + c_inc_high_da * xp.clip(xp.asarray(inc_cent), 0, None)  # high inc only
              + c_ind_da  * xp.asarray(ind_industrial)   # industrial/agri captive
              + c_mgr_da  * xp.asarray(occ_mgr_prof)     # managers → DA
              + c_lab_da  * xp.asarray(occ_labourer))    # labourers → DA
        # Apply licence AND household vehicle as hard availability:
        # unlicensed OR zero-vehicle household → V_DA = -inf (cannot drive alone)
        V_DA = xp.where((xp.asarray(licence) > 0) & (xp.asarray(vehrd) > 0),
                        V_DA, -1e9)

        # Shared-ride / Other
        V_SR = (asc_sr
              + c_ivt_auto   * ivt_auto
              + c_crash_veh  * log_crash
              + c_veh_sr     * vehrd
              + c_incp_sr    * incp
              + c_joint_sr   * joint
              + c_ret_sr  * xp.asarray(ind_retail))      # retail/hospitality → SR

        # Transit - hard availability: block when IVT >= TRANSIT_IVT_UNAVAIL_MIN
        # (no service, or only the sentinel fill value present in skim).
        V_TR = (asc_transit
              + c_ivt_transit   * ivt_transit
              + c_crash_veh     * log_crash
              + c_noveh_transit * noveh
              + c_incp_transit  * incp
              + c_female_transit * female
              + c_student_tr    * is_student             # general student flag
              + c_fts_tr * xp.asarray(is_ft_student)    # FT students → transit
              + c_pts_tr * xp.asarray(is_pt_student)    # PT students → transit
              + c_off_tr * xp.asarray(ind_office)        # office workers → transit
              + c_pub_tr * xp.asarray(ind_public)        # public sector → transit
              + c_hlt_tr * xp.asarray(ind_health)        # health workers → transit
              + c_clr_tr * xp.asarray(occ_clerical)      # clerical → transit
              + c_inc_low_tr    * xp.clip(xp.asarray(-inc_cent), 0, None))  # low inc only
        V_TR = xp.where(xp.asarray(transit_avail) > 0, xp.asarray(V_TR), -1e9)

        # Walk - person-type heterogeneity from ActivitySim:
        # - no-vehicle households are captive active-mode users
        # - older persons less likely to walk (age5p bands)
        # - public-sector workers → walkable locations
        # - FT students → campus areas are walkable
        V_WK = (asc_walk
              + c_walk_time   * walk_time
              + c_noveh_walk  * noveh
              + c_age_walk    * age5p
              + c_incp_active * incp               # mirrors MODE_UEC c_incp_active
              + c_female_walk * female
              + c_pub_wk * xp.asarray(ind_public)       # public sector → walk
              + c_fts_wk * xp.asarray(is_ft_student))   # FT students → walk
        # Apply availability as hard constraint - mirrors CARLICENCE → DA.
        # walk_avail=0 when OD distance exceeds WALK_MAX_KM (set in _predict_batch).
        walk_avail = pd.to_numeric(
            tour_rows.get('walk_avail', pd.Series(1, index=tour_rows.index)),
            errors='coerce').fillna(1).values.astype(float)
        V_WK = xp.where(walk_avail > 0, xp.asarray(V_WK), -1e9)

        # Bike - FT students more likely to cycle (campus cycling culture)
        V_BK = (asc_bike
              + c_bike_time  * bike_time
              + c_age_bike   * age5p
              + c_incp_active * incp               # mirrors MODE_UEC c_incp_active
              + c_young_bike * young
              + c_fts_bk * xp.asarray(is_ft_student))   # FT students → bike
        # Apply availability as hard constraint - mirrors CARLICENCE → DA.
        # bike_avail=0 when OD distance exceeds BIKE_MAX_KM (set in _predict_batch).
        bike_avail = pd.to_numeric(
            tour_rows.get('bike_avail', pd.Series(1, index=tour_rows.index)),
            errors='coerce').fillna(1).values.astype(float)
        V_BK = xp.where(bike_avail > 0, xp.asarray(V_BK), -1e9)

        # ── 6. Nested logit (Root{Motor{Car{DA,SR}, Transit}, NonMotor{Walk,Bike}}) ──
        # Was a flat 5-way softmax; see _nested_mode_probs for why that's
        # mathematically inconsistent with coefficients fit under the real
        # nested structure (it was collapsing SR/Bike/Transit toward Walk).
        mu_car      = _sanitize_mu(_pv('Mu_Car',      1.0), 'Mu_Car')
        mu_motor    = _sanitize_mu(_pv('Mu_Motor',    1.0), 'Mu_Motor')
        mu_nonmotor = _sanitize_mu(_pv('Mu_NonMotor', 1.0), 'Mu_NonMotor')
        probs_raw = _nested_mode_probs(
            xp.asarray(V_DA), xp.asarray(V_SR), xp.asarray(V_WK),
            xp.asarray(V_BK), xp.asarray(V_TR),
            mu_car=mu_car, mu_motor=mu_motor, mu_nonmotor=mu_nonmotor, xp=xp,
        )
        exp_V = np.asarray(probs_raw)   # already-normalised probs (sums to ~1 per row)

        # Handle rows where ALL modes are blocked (all -1e9) - fall back to
        # DA+Transit equal share rather than uniform across blocked modes.
        # Detected from the raw utilities directly: the nested-logit formula
        # above always returns rows summing to ~1 (it distributes across
        # nests based on the relative -1e9 sentinels, which is an arbitrary
        # floating-point artifact in this all-blocked case, not a real
        # signal), so a post-hoc "totals below threshold" check on exp_V no
        # longer detects this case the way it did under the old flat softmax.
        V_stack = np.asarray(xp.stack([V_DA, V_SR, V_WK, V_BK, V_TR], axis=1))
        all_blocked = (V_stack <= -1e8).all(axis=1)
        if all_blocked.any():
            n_blocked = int(all_blocked.sum())
            logger.warning(
                f"_mode_probs_numpy: {n_blocked} rows have ALL modes blocked "
                "(all utilities = -1e9). Falling back to DA=0.6 / Transit=0.4 "
                "for those rows. Check CARLICENCE and skim availability."
            )
            # fallback: DA (col 0) = 0.6, Transit (col 4) = 0.4
            fallback = np.zeros((n_blocked, 5), dtype=np.float64)
            fallback[:, 0] = 0.6   # DA
            fallback[:, 4] = 0.4   # Transit
            exp_V[all_blocked] = fallback

        totals = exp_V.sum(axis=1, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        probs = np.asarray(exp_V / totals, dtype=np.float64)

        # ── 7. INFO: log realised mode split and availability counts ─────────
        mode_names = ['DA', 'SR', 'Walk', 'Bike', 'Transit']
        mean_probs = probs.mean(axis=0)
        logger.info(
            f"_mode_probs_numpy predicted split (n={n}): "
            + "  ".join(f"{m}={p:.1%}" for m, p in zip(mode_names, mean_probs))
        )
        logger.info(
            f"_mode_probs_numpy availability: "
            f"walk={int(walk_avail.sum())}/{n} ({walk_avail.mean():.1%})  "
            f"bike={int(bike_avail.sum())}/{n} ({bike_avail.mean():.1%})  "
            f"transit={int(transit_avail.sum())}/{n} ({transit_avail.mean():.1%})  "
            f"noveh={int(noveh.sum())}/{n}  "
            f"ASC_Walk={asc_walk:.2f}  ASC_Bike={asc_bike:.2f}  ASC_Transit={asc_transit:.2f}"
        )
        # Warn loudly if DA share > 90% - almost certainly a data or coef problem
        if mean_probs[0] > 0.90:
            logger.warning(
                f"_mode_probs_numpy: DA share = {mean_probs[0]:.1%} - very high. "
                f"Walk avail: {int(walk_avail.sum())}/{n}, "
                f"Bike avail: {int(bike_avail.sum())}/{n}, "
                f"Transit avail: {int(transit_avail.sum())}/{n}. "
                f"ASC_Walk={asc_walk:.3f}, ASC_Bike={asc_bike:.3f}, "
                f"ASC_Transit={asc_transit:.3f}. "
                "If defaults were used, mode_coef may not have loaded from pickle. "
                "Also check WALK_MAX_KM/BIKE_MAX_KM thresholds vs skim distances."
            )

        return probs

    except Exception as e:
        logger.warning(f"_mode_probs_numpy failed: {e}", exc_info=True)
        return None


def _predict_batch(
    act_df: pd.DataFrame,
    models: dict,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    skims,
    work_taz_map: dict,
    school_taz_map: dict,
    rng: np.random.Generator,
    skim_lookup: dict = None,
) -> pd.DataFrame:
    """
    Vectorised destination + mode prediction for ALL activities in one pass.

    Instead of building a new DataTree per activity (which triggers sharrow
    recompilation every call → 4-hour hang), this builds ONE DataTree per
    segment covering all activities of that type, runs probability() once
    vectorised, then samples results.

    ActivitySim pattern: simulate_choices() called on the full choosers table,
    not in a per-person loop.

    Returns act_df with DTAZ and pred_mode columns added.
    """
    import xarray as xr
    # NOTE: zsim_helper.monte_sample was here but is unused - removed to avoid
    # ImportError on systems where zsim_helper is not installed.

    df = act_df.copy()
    df['DTAZ']      = df['OTAZ'].astype(int)   # default: stay in origin
    df['pred_mode'] = 1                         # default: DA

    # ── diagnostics ────────────────────────────────────────────────────
    _n_rows = len(df)
    _otaz_u = df['OTAZ'].nunique() if 'OTAZ' in df.columns else 0
    _seg_u  = df['segment'].value_counts().to_dict() if 'segment' in df.columns else {}
    _models_segs = list(models.keys()) if models else []
    print(f"    _predict_batch: {_n_rows} activities, {_otaz_u} unique OTAZ, "
          f"segments={_seg_u}, models={_models_segs}")
    if not models:
        logger.warning("_predict_batch called with empty models dict - "
                       "all DTAZ will default to OTAZ, pred_mode=1 (DA)")
        return df
    # ── end diagnostics ─────────────────────────────────────────────────

    pp = persons.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})

    # ── apply usual location anchors for those who have them ─────────────
    # Persons with a WORK_TAZ / SCHOOL_TAZ get their usual location directly.
    # Everyone else will be handled by the segment dest model below.
    if work_taz_map:
        work_mask = df['segment'] == 'work'
        mapped = df.loc[work_mask, 'person_id'].astype(str).map(work_taz_map)
        df.loc[work_mask, 'DTAZ'] = mapped.fillna(-1).astype(int)
        # -1 means "no usual location" - will be filled by dest model below
    if school_taz_map:
        edu_mask = df['segment'] == 'edu'
        mapped = df.loc[edu_mask, 'person_id'].astype(str).map(school_taz_map)
        df.loc[edu_mask, 'DTAZ'] = mapped.fillna(-1).astype(int)

    # ── vectorised prediction per segment ─────────────────────────────
    # Only nonmand needs destination MNL; work/edu already have anchors.
    # Mode choice runs for ALL segments (including work/edu with fixed dest).
    _prediction_segment_order = (
        [seg for seg in ('work', 'edu', 'nonmand') if seg in models]
        + [seg for seg in models if seg not in ('work', 'edu', 'nonmand')]
    )
    for seg in _prediction_segment_order:
        seg_models = models[seg]
        if not seg_models:
            continue
        mode_mdl = seg_models.get('mode_model')
        dest_mdl = seg_models.get('dest_model')
        if dest_mdl is not None:
            # Ensure _segment is set so prediction path uses study-level size term
            if not hasattr(dest_mdl, '_segment') or dest_mdl._segment is None:
                dest_mdl._segment = seg
        if mode_mdl is None:
            continue
        if dest_mdl is not None and seg_models.get('od_calibration') is not None:
            try:
                dest_mdl._od_calibration = seg_models.get('od_calibration') or {}
            except Exception:
                pass
        if dest_mdl is not None and seg_models.get('zone_shadow_prices') is not None:
            try:
                dest_mdl._zone_shadow_prices = seg_models.get('zone_shadow_prices')
            except Exception:
                pass
        if dest_mdl is not None and seg_models.get('pred_zone_shares') is not None:
            try:
                dest_mdl._pred_zone_shares = seg_models.get('pred_zone_shares')
            except Exception:
                pass
        # Restore mode shares cache - mode_model.probability() fails after pkl reload
        # because the DataTree is not attached.  Use the stored shares instead.
        mode_mdl = seg_models.get('mode_model')
        if mode_mdl is not None and seg_models.get('pred_mode_shares') is not None:
            try:
                mode_mdl._pred_mode_shares = seg_models.get('pred_mode_shares')
            except Exception:
                pass
        if dest_mdl is not None and seg_models.get('cluster_targets') is not None:
            try:
                dest_mdl._cluster_targets = seg_models.get('cluster_targets') or {}
            except Exception:
                pass

        seg_mask = df['segment'] == seg
        if not seg_mask.any():
            logger.debug(f"_predict_batch: segment '{seg}' has 0 rows in act_df - skipping")
            continue

        # ── Frozen person distance draws (work segment only) ─────────────
        # Apply deterministic N(0, s_log_dist) offsets to pp['person_dist_draw']
        # before destination prediction.  c_person_dist_draw is work-only in
        # DEST_UEC, but setting the column for all persons is harmless - it
        # only enters the utility when the DEST_UEC entry is active.
        _s_log_dist = seg_models.get('s_log_dist', 0.0)
        if _s_log_dist > 0 and seg == 'work' and 'person_id' in pp.columns:
            try:
                _dist_offsets = _frozen_person_dist_offsets(pp['person_id'], _s_log_dist)
                pp = pp.copy()
                pp['person_dist_draw'] = (
                    pp['person_id'].map(_dist_offsets).fillna(0.0).astype(np.float32)
                )
                logger.info(
                    f"_predict_batch [{seg}]: frozen person_dist_draw applied "
                    f"(s_log_dist={_s_log_dist:.3f}, n={len(_dist_offsets)})"
                )
            except Exception as _pdo_e:
                logger.debug(f"_predict_batch [{seg}]: frozen dist offsets failed - {_pdo_e}")

        seg_rows = df[seg_mask].copy()
        n = len(seg_rows)
        print(f"    [{seg}] predicting {n} activities "
              f"(mode_model={'✓' if mode_mdl else '✗'}, "
              f"dest_model={'✓' if dest_mdl else '✗'})")

        # ── build properly structured tour DataFrame ──────────────────
        # _mode_tree expects exact columns: ptour_id, person_id, OTAZ, DTAZ,
        # tour_mode, outbound_stops, n_shopping_out, incp, indp, vehrd,
        # STUDYING, sexp, walk_avail, bike_avail, main_purpose, primary_purpose
        # seg_rows is Stage 4 activities output - build the correct structure.
        def _to_int(series, default):
            return pd.to_numeric(series, errors='coerce').fillna(default).astype(int)

        otaz_vals = _to_int(seg_rows['OTAZ'], 1)
        dtaz_vals = _to_int(seg_rows.get('DTAZ', seg_rows['OTAZ']), 1)

        # ── compute mode availability per OD pair - vectorised ────────────
        # Build a two-column lookup from skim_lookup dict once, then merge.
        # This replaces an O(n) Python loop with a single pandas merge.
        # SEQ SA2 zones are large (typically 5-30 km across); use generous
        # thresholds so walk/bike are available for nearby-zone trips rather
        # than being hard-blocked by zone geometry.  The utility function
        # (c_walk_time, c_bike_time) will naturally suppress long-distance
        # walk/bike - the role of these flags is only to block physically
        # impossible trips (> 6 km on foot, > 25 km by bike).
        WALK_MAX_KM   = 6.0    # was 3.0 - raised to match realistic walk access
        BIKE_MAX_KM   = 25.0   # was 10.0 - raised for longer cycling
        WALK_MAX_MIN  = 10.0   # was 5.0  - auto-IVT proxy fallback
        BIKE_MAX_MIN  = 20.0   # was 12.0 - auto-IVT proxy fallback

        if skim_lookup:
            # Convert skim_lookup to a DataFrame keyed on (OTAZ, DTAZ)
            # Only extract the columns we need to keep the DataFrame small.
            needed = ['DISTWALK', 'DISTBIKE', 'SOVTOLL_TIME__AM']
            sk_rows = []
            for (o, d), vals in skim_lookup.items():
                sk_rows.append({
                    'OTAZ': o, 'DTAZ': d,
                    **{k: vals.get(k, None) for k in needed}
                })
            sk_df = pd.DataFrame(sk_rows)

            avail_df = pd.DataFrame({
                'OTAZ': otaz_vals.values,
                'DTAZ': dtaz_vals.values,
            }, index=seg_rows.index)
            avail_df = avail_df.merge(sk_df, on=['OTAZ', 'DTAZ'], how='left')

            def _km(col):
                # DISTWALK / DISTBIKE are already in km in this OMX.
                return pd.to_numeric(avail_df[col], errors='coerce')

            auto_ivt = pd.to_numeric(avail_df.get('SOVTOLL_TIME__AM'), errors='coerce')
            intra    = (avail_df['OTAZ'] == avail_df['DTAZ'])

            walk_dist = _km('DISTWALK') if 'DISTWALK' in avail_df.columns else pd.Series(dtype=float)
            bike_dist = _km('DISTBIKE') if 'DISTBIKE' in avail_df.columns else pd.Series(dtype=float)

            walk_ok = intra.copy()
            bike_ok = intra.copy()

            has_walk_dist = walk_dist.notna() & (walk_dist > 0)
            walk_ok = walk_ok | (has_walk_dist & (walk_dist < WALK_MAX_KM))
            walk_ok = walk_ok | (~has_walk_dist & auto_ivt.notna() & (auto_ivt > 0) & (auto_ivt < WALK_MAX_MIN))

            has_bike_dist = bike_dist.notna() & (bike_dist > 0)
            bike_ok = bike_ok | (has_bike_dist & (bike_dist < BIKE_MAX_KM))
            bike_ok = bike_ok | (~has_bike_dist & auto_ivt.notna() & (auto_ivt > 0) & (auto_ivt < BIKE_MAX_MIN))

            walk_avail_vals = walk_ok.astype(int).tolist()
            bike_avail_vals = bike_ok.astype(int).tolist()
        else:
            # No skim lookup - allow walk/bike for a reasonable proportion of trips.
            # When no distance info exists, assume ~15% of trips are walkable (< 3 km)
            # and ~30% are bikeable (< 10 km) based on typical SEQ modal shares.
            # Use a deterministic assignment: mark every 7th trip as walk-ok,
            # every 3rd trip as bike-ok - gives realistic availability share without
            # a distance skim.
            intra = (otaz_vals == dtaz_vals).to_numpy()
            n_rows = len(intra)
            walk_arr = intra.copy()
            bike_arr = intra.copy()
            walk_arr[np.arange(0, n_rows, 7)] = True   # ~1 in 7 non-intra trips walkable
            bike_arr[np.arange(0, n_rows, 3)] = True   # ~1 in 3 non-intra trips bikeable
            walk_avail_vals = walk_arr.astype(int).tolist()
            bike_avail_vals = bike_arr.astype(int).tolist()
            logger.info(
                f"_predict_batch [{seg}]: no skim_lookup - "
                f"walk_avail set for {int(walk_arr.sum())}/{n_rows} rows, "
                f"bike_avail for {int(bike_arr.sum())}/{n_rows} rows"
            )

        # ── compute _time_period for c_peak_da ───────────────────────────
        # STARTIME is in minutes-from-midnight; same period mapping as preprocess_tours
        def _period_from_min(m):
            try:
                m = float(m)
            except Exception:
                m = 480.0
            if m < 360:  return 0
            if m < 540:  return 1
            if m < 900:  return 2
            if m < 1080: return 3
            if m < 1260: return 4
            return 5

        if 'STARTIME' in seg_rows.columns:
            time_periods = seg_rows['STARTIME'].apply(_period_from_min).astype(int)
        else:
            time_periods = pd.Series(1, index=seg_rows.index)

        tour_rows = pd.DataFrame({
            'ptour_id':        seg_rows.index.astype(str),
            'person_id':       seg_rows['person_id'].astype(str),
            'tour_id':         seg_rows.index.astype(int),
            'OTAZ':            otaz_vals,
            'DTAZ':            dtaz_vals,
            'tour_mode':       _to_int(seg_rows.get('mode', pd.Series(1, index=seg_rows.index)), 1),
            'main_purpose':    seg_rows.get('purpose', pd.Series('', index=seg_rows.index)).astype(str),
            'primary_purpose': seg_rows.get('purpose', pd.Series('', index=seg_rows.index)).astype(str),
            'outbound_stops':  _to_int(seg_rows.get('outbound_stops', pd.Series(0, index=seg_rows.index)), 0),
            'n_shopping_out':  _to_int(seg_rows.get('n_shopping_out', pd.Series(0, index=seg_rows.index)), 0),
            'joint_travel':    _to_int(seg_rows.get('joint_travel',
                                      ((pd.to_numeric(seg_rows.get('outbound_stops', pd.Series(0, index=seg_rows.index)), errors='coerce').fillna(0) > 0) |
                                       (seg_rows.get('purpose', pd.Series('', index=seg_rows.index)).astype(str).str.lower() == 'escort')).astype(int)), 0),
            'incp':            _to_int(seg_rows.get('incp', pd.Series(8, index=seg_rows.index)), 8),
            'indp':            _to_int(seg_rows.get('indp', pd.Series(19, index=seg_rows.index)), 19),
            'vehrd':           _to_int(seg_rows.get('vehrd', pd.Series(1, index=seg_rows.index)), 1),
            'STUDYING':        _to_int(seg_rows.get('STUDYING', pd.Series(0, index=seg_rows.index)), 0),
            'sexp':            _to_int(seg_rows.get('sexp', pd.Series(1, index=seg_rows.index)), 1),
            # New behavioural columns on the tour table (mirrors person.xxx for tour.xxx path)
            # TODO(CARLICENCE): keep this embedded tour-table fallback aligned
            # with _ensure_person_behavioural_cols for lowercase synth input.
            '_time_period':    time_periods,
            'age5p':           _to_int(seg_rows.get('age5p',      pd.Series(6,  index=seg_rows.index)), 6),
            'CARLICENCE':      _to_int(
                seg_rows.get(
                    'CARLICENCE',
                    seg_rows.get('carlicence', pd.Series(1, index=seg_rows.index))
                ),
                1
            ),
            'occp_group':      _to_int(seg_rows.get('occp_group', pd.Series(0,  index=seg_rows.index)), 0),
            'bikes':           _to_int(seg_rows.get('bikes',      pd.Series(0,  index=seg_rows.index)), 0),
            'nprd':            _to_int(seg_rows.get('nprd',       pd.Series(2,  index=seg_rows.index)), 2),
            'walk_avail':      pd.Series(walk_avail_vals, index=seg_rows.index),
            'bike_avail':      pd.Series(bike_avail_vals, index=seg_rows.index),
            # weight=1 during prediction - avoids 'unable to rewrite weight to itself'
            'weight':          pd.Series(1.0, index=seg_rows.index),
        }).set_index('ptour_id')

        # ── Fine-grained home zone for SA1/Lvl4 walk-bike skim lookup ────
        # Walk and Bike travel uses SA1-origin skims (other modes stay on the
        # SA2 OMX).  Pull HOME_SA1/HOME_LVL4 from the tour table if the
        # upstream stages carried it, else from the person table.
        for _fz in ('HOME_SA1', 'HOME_LVL4'):
            if _fz in seg_rows.columns:
                tour_rows[_fz] = pd.to_numeric(
                    seg_rows[_fz], errors='coerce').fillna(0).astype(np.int64).values
            elif pp is not None and _fz in pp.columns and 'person_id' in pp.columns:
                _fz_map = (pp.drop_duplicates('person_id')
                             .set_index(pp.drop_duplicates('person_id')['person_id'].astype(str))[_fz])
                tour_rows[_fz] = pd.to_numeric(
                    tour_rows['person_id'].map(_fz_map), errors='coerce'
                ).fillna(0).astype(np.int64)

        # keep index as string (ptour_id is the caseid for Larch)
        seg_rows = seg_rows.copy()
        seg_rows['ptour_id'] = tour_rows.index.values

        # ── destination prediction ────────────────────────────────────
        # Nonmand: always needs model.
        # Work/edu: run model for rows that have no usual-location (DTAZ==-1).
        #           Rows with DTAZ already set (usual-location) are left alone.
        no_usual = (df.loc[seg_mask, 'DTAZ'] == -1)
        needs_dest = (dest_mdl is not None) and (
            (seg == 'nonmand') or
            (seg in ('work', 'edu') and no_usual.any())
        )
        # restrict to rows that actually need prediction
        if seg in ('work', 'edu') and needs_dest:
            predict_mask = seg_mask & (df['DTAZ'] == -1)
        else:
            predict_mask = seg_mask
        if not predict_mask.any():
            needs_dest = False
        if needs_dest:
            # subset to only rows that need prediction
            pred_seg_rows  = df[predict_mask].copy()
            pred_tour_rows = tour_rows.loc[
                tour_rows.index.isin(pred_seg_rows.index.astype(str))
            ].copy()
            gravity_rows = None
            if pred_tour_rows.empty:
                needs_dest = False

        if needs_dest:
            needs_gravity = True   # assume gravity until a model succeeds
            print(f"      [{seg}] dest prediction: {len(pred_seg_rows)} choosers, "
                  f"dest_mdl={dest_mdl is not None}")
            try:
                dt_m_batch = _mode_tree(pred_tour_rows.reset_index(), pp, landuse, skims)
                _m_coef = seg_models.get('mode_coef') or seg_models.get('coef') or {}
                ls_batch   = _logsums(mode_mdl, dt_m_batch, skims=skims, verbose=False,
                                      coef_override=_m_coef)
                # Standardize logsums to mean=0, std=1 - must match the scale used
                # during estimation (see _run_segment line ~3499).  Without this,
                # c_logsum (estimated against standardized values) applies to raw
                # logsums and produces systematically wrong destination utilities.
                try:
                    _ls_vals = ls_batch.values.astype(float)
                    _ls_mu = float(seg_models.get('logsum_mean', 0.0) or 0.0)
                    _ls_sd = float(seg_models.get('logsum_std', 1.0) or 1.0)
                    if np.isfinite(_ls_sd) and (_ls_sd > 1e-9):
                        _ls_vals = (_ls_vals - _ls_mu) / _ls_sd
                        ls_batch = ls_batch.copy(data=_ls_vals.astype(ls_batch.values.dtype))
                        logger.debug(
                            f"Prediction logsums standardized using ESTIMATION scale [{seg}] "
                            f"(mean={_ls_mu:.4f}, std={_ls_sd:.4f})"
                        )
                    else:
                        logger.debug(
                            f"Prediction logsums left unscaled [{seg}] due to invalid std={_ls_sd}"
                        )
                except Exception as _lse:
                    logger.debug(f"Prediction logsum standardization failed ({_lse})")
                lu_st      = _ensure_destination_clusters(_attach_size_term(landuse, seg))
                lu_st, pred_tour_rows = _attach_acdc_terms(
                    lu_st, pred_tour_rows,
                    skim_lookup_to_df(skim_lookup) if skim_lookup else pd.DataFrame(),
                    seg
                )
                dt_d_batch = _dest_tree(
                    pred_tour_rows.reset_index(),
                    pp,
                    lu_st,
                    skims,
                    ls_batch,
                    crash_skims=seg_models.get('crash_source'),
                    segment=seg,
                )
                zone_ids_pred = list(ls_batch.coords['zones'].values)

                # Destination prediction: all-zones softmax OR ActivitySim-style sampling
                # DEST_CHOICE_SAMPLE_SIZE == 0 → compute utility over ALL zones, sample directly
                # DEST_CHOICE_SAMPLE_SIZE  > 0 → sample N alternatives then resimulate
                print(f"      [{seg}] dest predict: {len(pred_seg_rows)} choosers, "
                      f"{len(zone_ids_pred)} dest zones, sample_size={DEST_CHOICE_SAMPLE_SIZE or 'ALL'}")

                # Pre-align OTAZ values by ls_batch case order so _dest_utility_numpy
                # never needs to look them up via the DataTree (fixes flat-dataset NaN)
                _ls_case_ids = [str(c) for c in ls_batch.coords['cases'].values]
                _otaz_ser = pred_tour_rows.reset_index().set_index(
                    next((c for c in ['ptour_id', 'index', 'tour_id']
                          if c in pred_tour_rows.reset_index().columns), None)
                    or pred_tour_rows.reset_index().columns[0]
                )['OTAZ'] if 'OTAZ' in pred_tour_rows.columns else None
                _case_otaz_arr = None
                if _otaz_ser is not None:
                    try:
                        _case_otaz_arr = pd.to_numeric(
                            _otaz_ser.reindex(_ls_case_ids), errors='coerce'
                        ).fillna(1).astype(int).to_numpy()
                    except Exception:
                        _case_otaz_arr = None

                V_full = _dest_utility_numpy(dest_mdl, dt_d_batch, ls_batch, zone_ids_pred, pp,
                                             include_logsum=True,
                                             case_otaz_override=_case_otaz_arr)
                print(f"      [{seg}] V_full shape={None if V_full is None else V_full.shape}, "
                      f"any_nan={None if V_full is None else bool(np.isnan(V_full).any())}, "
                      f"any_inf={None if V_full is None else bool(np.isinf(V_full).any())}")

                if V_full is not None and np.isfinite(V_full).any():
                    zone_arr_pred = np.asarray(zone_ids_pred, dtype=int)
                    _polaris_sampling = POLARIS_DEST_CHOICE and seg == 'nonmand'
                    _planning_mask = None
                    if _polaris_sampling and POLARIS_PLANNING_CONSTRAINED:
                        _planning_origins = (
                            _case_otaz_arr
                            if _case_otaz_arr is not None
                            else pred_seg_rows['OTAZ'].astype(int).to_numpy()
                        )
                        _planning_mask = _polaris_space_time_mask(
                            pred_seg_rows.reset_index(drop=True),
                            df,
                            zone_arr_pred,
                            _planning_origins,
                            skim_lookup,
                        )
                        if _planning_mask.shape == V_full.shape:
                            V_full = np.where(_planning_mask, V_full, -np.inf)
                            logger.info(
                                f"Dest Polaris planning constraint [{seg}]: "
                                f"mean reachable zones={_planning_mask.sum(axis=1).mean():.1f}/"
                                f"{_planning_mask.shape[1]}"
                            )
                    _probs_full = _softmax_from_utilities(V_full)

                    # Diagnostics on probabilities
                    _top_zone_prob  = float(_probs_full.max(axis=1).mean())
                    _n_nonzero_mean = float((_probs_full > 1e-9).sum(axis=1).mean())
                    print(f"      [{seg}] prob diagnostics: mean_top_zone_prob={_top_zone_prob:.4f}, "
                          f"mean_zones_with_nonzero_prob={_n_nonzero_mean:.1f}/{len(zone_ids_pred)}")

                    if _polaris_sampling:
                        _polaris_origins = (
                            _case_otaz_arr
                            if _case_otaz_arr is not None
                            else pred_seg_rows['OTAZ'].astype(int).to_numpy()
                        )
                        chosen_zone, valid_rows = _polaris_sequential_destination_draw(
                            V_full,
                            pred_seg_rows.reset_index(drop=True),
                            df,
                            zone_arr_pred,
                            _polaris_origins,
                            skim_lookup,
                            landuse,
                            rng,
                        )
                        needs_gravity = False
                        df.loc[pred_seg_rows.index[valid_rows], 'DTAZ'] = chosen_zone[valid_rows]
                        if (~valid_rows).any():
                            gravity_rows = pred_seg_rows.loc[pred_seg_rows.index[~valid_rows]].copy()
                            df.loc[gravity_rows.index, 'DTAZ'] = df.loc[gravity_rows.index, 'OTAZ'].astype(int)
                            logger.warning(f"Dest Polaris sampled simulate invalid for {int((~valid_rows).sum())}/{len(pred_seg_rows)} rows [{seg}]")
                        logger.info(
                            f"Dest Polaris sampled simulate succeeded [{seg}]: "
                            f"strata={POLARIS_DEST_STRATA}, "
                            f"alts_per_stratum={POLARIS_DEST_ALTS_PER_STRATUM}, "
                            f"choosers={len(pred_seg_rows)}"
                        )
                    elif DEST_CHOICE_SAMPLE_SIZE > 0:
                        # ActivitySim two-step: sample without logsums, resimulate with
                        sample_probs = _dest_probs_numpy(dest_mdl, dt_d_batch, ls_batch, zone_ids_pred, pp,
                                                             include_logsum=False,
                                                             case_otaz_override=_case_otaz_arr)
                        if sample_probs is not None and not np.isnan(sample_probs).all():
                            sampled_alts = _activitysim_make_sample_choices(sample_probs, zone_ids_pred, DEST_CHOICE_SAMPLE_SIZE, rng)
                            if not sampled_alts.empty:
                                chosen_zone, valid_rows = _activitysim_simulate_sampled_destinations(V_full, sampled_alts, zone_ids_pred, rng)
                                needs_gravity = False
                                df.loc[pred_seg_rows.index[valid_rows], 'DTAZ'] = chosen_zone[valid_rows]
                                if (~valid_rows).any():
                                    gravity_rows = pred_seg_rows.loc[pred_seg_rows.index[~valid_rows]].copy()
                                    df.loc[gravity_rows.index, 'DTAZ'] = df.loc[gravity_rows.index, 'OTAZ'].astype(int)
                                    logger.warning(f"Dest sampled simulate invalid for {int((~valid_rows).sum())}/{len(pred_seg_rows)} rows [{seg}]")
                                logger.info(f"Dest sampled simulate succeeded [{seg}]: sample_size={DEST_CHOICE_SAMPLE_SIZE}, choosers={len(pred_seg_rows)}")
                    else:
                        # ALL-ZONES vectorised path: cumulative inverse-CDF sampling
                        # Avoids a slow Python loop over n_choosers for large batches.
                        _row_sums = _probs_full.sum(axis=1, keepdims=True)
                        _valid_mask = (_row_sums.ravel() > 0) & np.isfinite(_row_sums.ravel())
                        _probs_norm = np.where(
                            _valid_mask[:, None],
                            _probs_full / np.where(_row_sums > 0, _row_sums, 1.0),
                            0.0,
                        )
                        # Inverse-CDF draw: uniform random per row, pick zone where CDF crosses
                        _u = rng.random(_probs_norm.shape[0])
                        _cdf = np.cumsum(_probs_norm, axis=1)
                        _picks = ((_cdf < _u[:, None]) * 1).sum(axis=1)   # index of chosen zone
                        _picks = np.clip(_picks, 0, len(zone_arr_pred) - 1)
                        chosen_zone = zone_arr_pred[_picks]
                        chosen_zone = np.where(_valid_mask, chosen_zone, -1)
                        valid_rows  = _valid_mask
                        needs_gravity = False
                        df.loc[pred_seg_rows.index[valid_rows], 'DTAZ'] = chosen_zone[valid_rows]
                        if (~valid_rows).any():
                            gravity_rows = pred_seg_rows.loc[pred_seg_rows.index[~valid_rows]].copy()
                            df.loc[gravity_rows.index, 'DTAZ'] = df.loc[gravity_rows.index, 'OTAZ'].astype(int)
                            logger.warning(f"Dest all-zones invalid for {int((~valid_rows).sum())}/{len(pred_seg_rows)} rows [{seg}]")

                        # Zone distribution diagnostics
                        _chosen_valid = chosen_zone[valid_rows]
                        if len(_chosen_valid) > 0:
                            _unique_chosen, _chosen_counts = np.unique(_chosen_valid, return_counts=True)
                            _top3 = sorted(zip(_chosen_counts, _unique_chosen), reverse=True)[:3]
                            print(f"      [{seg}] chosen zones: {len(_unique_chosen)} unique, "
                                  f"top3={([(int(z), int(c)) for c, z in _top3])}, "
                                  f"valid={valid_rows.sum()}/{len(valid_rows)}")
                        logger.info(f"Dest all-zones succeeded [{seg}]: {len(zone_ids_pred)} zones, choosers={len(pred_seg_rows)}")

            except Exception as e:
                import traceback as _tb
                logger.warning(
                    f"Dest model failed [{seg}]: {e!r}\n"
                    f"  Traceback: {_tb.format_exc(limit=4)}"
                )
                _strict_seg = os.environ.get('STAGE5_STRICT_ESTIMATION', '1') != '0'
                if _strict_seg:
                    raise RuntimeError(
                        f"Destination model prediction failed for segment '{seg}' in strict mode"
                    ) from e
                print(f"      [{seg}] dest model FAILED: {e!r} - using gravity fallback")

            # ── gravity fallback: size_term × exp(-β × auto_IVT) ─────────
            # Runs when Larch is degenerate or fails.  Samples across ALL
            # 329 zones so destinations are not collapsed to a handful.
            if needs_gravity:
                try:
                    lu_st    = _attach_size_term(landuse, seg)
                    zone_col = next(
                        (c for c in ['ORDER', 'zone_id'] if c in lu_st.columns),
                        lu_st.columns[0],
                    )
                    if lu_st.index.name == zone_col:
                        lu_st = lu_st.reset_index()
                    zones_arr = lu_st[zone_col].astype(int).values
                    size_arr  = np.exp(                          # un-log
                        pd.to_numeric(lu_st['size_term'], errors='coerce')
                        .fillna(0.0).clip(lower=-30, upper=30).values
                    )
                    beta_ivt  = -0.05   # was -0.10 (too strong → only 40-90 unique zones)
                    # At -0.05: 30-min trip weight = exp(-1.5)=0.22 vs exp(-3.0)=0.05 at -0.10
                    # This spreads destinations more realistically across 330 SEQ zones.

                    # Vectorised gravity fallback:
                    # 1. Build OD IVT matrix for all (person_origin, zone) pairs
                    # Prefer pred_seg_rows['OTAZ'] - it comes directly from the
                    # activity DataFrame and is always populated (set at the top
                    # of _predict_batch).  pred_tour_rows may have stale/wrong
                    # OTAZs if the DataTree case-key alignment earlier failed.
                    try:
                        otaz_arr = pred_seg_rows['OTAZ'].astype(int).to_numpy(dtype=np.int32)
                    except Exception:
                        otaz_arr = np.array([
                            int(pred_tour_rows.iloc[k]['OTAZ'])
                            for k in range(len(pred_seg_rows))
                        ], dtype=np.int32)
                    unique_otaz = np.unique(otaz_arr)

                    # IVT matrix shape (n_unique_origins, n_zones)
                    ivt_matrix = np.full(
                        (len(unique_otaz), len(zones_arr)), 30.0, dtype=np.float32
                    )
                    if skim_lookup:
                        for oi, otaz in enumerate(unique_otaz):
                            for zi, z in enumerate(zones_arr):
                                val = skim_lookup.get((int(otaz), int(z)), {})
                                ivt_matrix[oi, zi] = float(
                                    val.get('SOVTOLL_TIME__AM', 30.0) or 30.0
                                )
                    otaz_to_row = {int(o): i for i, o in enumerate(unique_otaz)}

                    # 2. Compute weights for every person in one matrix op
                    person_rows  = np.array([otaz_to_row[int(o)] for o in otaz_arr])
                    ivts_persons = ivt_matrix[person_rows]          # (n_persons, n_zones)
                    weights      = size_arr * np.exp(beta_ivt * ivts_persons)
                    w_sums       = weights.sum(axis=1, keepdims=True)
                    w_sums       = np.where(w_sums > 0, w_sums, 1.0)
                    weights_norm = weights / w_sums

                    # 3. Vectorised inverse-CDF sample
                    cum_w       = np.cumsum(weights_norm, axis=1)
                    u_grav      = rng.uniform(size=len(pred_seg_rows))
                    chosen_grav = (cum_w < u_grav[:, np.newaxis]).sum(axis=1)
                    chosen_grav = np.clip(chosen_grav, 0, len(zones_arr) - 1)
                    df.loc[pred_seg_rows.index, 'DTAZ'] = zones_arr[chosen_grav]
                    n_placed = int((w_sums.flatten() > 0).sum())
                    n_uniq = df.loc[predict_mask, 'DTAZ'].nunique()
                    logger.info(
                        f"Gravity fallback [{seg}]: placed {n_placed}/"
                        f"{len(pred_seg_rows)} → {n_uniq} unique zones"
                    )
                    print(f"      [{seg}] gravity fallback: {n_uniq} unique dest zones "
                          f"({n_placed}/{len(pred_seg_rows)} placed)")
                except Exception as eg:
                    import traceback as _tb2
                    logger.warning(
                        f"Gravity fallback failed [{seg}]: {eg!r}\n"
                        f"  {_tb2.format_exc(limit=3)}"
                    )
                    print(f"      [{seg}] gravity fallback FAILED: {eg!r}")

        # ── clean up any remaining -1 sentinel (model+gravity both failed) ─
        unset = seg_mask & (df['DTAZ'] == -1)
        if unset.any():
            df.loc[unset, 'DTAZ'] = df.loc[unset, 'OTAZ']
            logger.debug(f"{unset.sum()} {seg} rows defaulted DTAZ→OTAZ")

        # ── mode prediction ───────────────────────────────────────────
        try:
            # update DTAZ in tour_rows from df predictions
            tour_rows['DTAZ'] = df.loc[seg_rows.index, 'DTAZ'].values

            dt_m_mode = _mode_tree(tour_rows.reset_index(), pp, landuse, skims)
            try:
                template  = dt_m_mode['tour']['DTAZ']
                new_dtaz  = xr.DataArray(
                    tour_rows['DTAZ'].values.astype(template.dtype),
                    dims=template.dims, coords=template.coords)
                new_tour  = dt_m_mode['tour'].assign(DTAZ=new_dtaz)
                dt_m_mode = dt_m_mode.replace_datasets(tour=new_tour)
            except Exception:
                pass

            mode_probs = None
            alt_ids    = [1, 2, 3, 4, 5]   # DA, SR, Walk, Bike, Transit
            _frozen = seg_models.get('frozen_probs')

            # ── Attempt -1: distance-aware numpy path with fine skims ─────
            # The frozen-cell path keys on person type only (no OD distance),
            # so it cannot reflect the SA1-origin walk/bike distances.  When
            # fine-grained SA1/Lvl4 walk-bike skims are available for these
            # tours, prefer _mode_probs_numpy so Walk/Bike utilities respond
            # to the person's actual (SA1-resolution) travel distance while
            # other modes keep SA2 skims.  Override with
            # STAGE5_MODE_PRED_PRIORITY=frozen|numpy|auto (default auto).
            _pred_priority = os.environ.get(
                'STAGE5_MODE_PRED_PRIORITY', 'auto').strip().lower()
            _fine_ready = False
            if _pred_priority in ('auto', 'numpy'):
                try:
                    from read_skims import SA1_WALK_BIKE_H5, LVL4_WALK_BIKE_H5
                    _fine_ready = (
                        ('HOME_SA1' in tour_rows.columns
                         and pd.to_numeric(tour_rows['HOME_SA1'], errors='coerce').fillna(0).gt(0).any()
                         and os.path.exists(os.path.normpath(SA1_WALK_BIKE_H5)))
                        or ('HOME_LVL4' in tour_rows.columns
                            and pd.to_numeric(tour_rows['HOME_LVL4'], errors='coerce').fillna(0).gt(0).any()
                            and os.path.exists(os.path.normpath(LVL4_WALK_BIKE_H5)))
                    )
                except Exception:
                    _fine_ready = False
            if _pred_priority == 'numpy' or (_pred_priority == 'auto' and _fine_ready):
                try:
                    _mode_coef_override = (
                        seg_models.get('mode_coef') or
                        seg_models.get('coef') or {}
                    )
                    mode_probs = _mode_probs_numpy(
                        mode_mdl, tour_rows, pp, skims, skim_lookup,
                        coef_override=_mode_coef_override
                    )
                    if mode_probs is not None:
                        logger.info(
                            f"Mode numpy-first path succeeded [{seg}] "
                            f"(fine SA1/Lvl4 walk-bike skims active)")
                except Exception as _nf_e:
                    logger.debug(f"Mode numpy-first failed [{seg}]: {_nf_e}")
                    mode_probs = None

            # ── Attempt 0: frozen cell probabilities (most stable) ────────
            # HHTS-estimated cell probabilities applied directly to synth pop.
            # No MC noise, fully reproducible, industry/occupation-aware.
            if mode_probs is None and _frozen is not None and len(_frozen) > 0:
                try:
                    mode_probs = _mode_probs_frozen(_frozen, tour_rows, pp)
                    if mode_probs is not None:
                        logger.info(f"Mode frozen-probs path succeeded [{seg}] "
                                    f"({len(_frozen)} cells)")
                except Exception as _fe:
                    logger.debug(f"Mode frozen-probs failed [{seg}]: {_fe}")
                    mode_probs = None

            # ── Attempt 1: Larch probability() ────────────────────────────
            if mode_probs is None:
                try:
                    mode_mdl.datatree = dt_m_mode
                    mode_mdl.mangle()
                    mode_mdl.unmangle()
                    mode_probs = np.asarray(mode_mdl.probability())
                    try:
                        alt_ids = list(dt_m_mode.altids())
                    except Exception:
                        pass
                    logger.debug(f"Mode Larch path succeeded [{seg}]")
                except Exception as larch_e:
                    logger.debug(
                        f"Mode Larch path failed [{seg}] ({larch_e}) - "
                        "trying numpy coefficient fallback"
                    )

            # ── Attempt 2: numpy coefficient fallback ─────────────────────
            if mode_probs is None:
                # Pass the NL mode model coefs (plain dict, pickle-safe).
                # 'mode_coef' is extracted at estimation time and survives pkl.
                # Fall back to 'coef' (MXL result) if mode_coef absent.
                _mode_coef_override = (
                    seg_models.get('mode_coef') or
                    seg_models.get('coef') or {}
                )
                mode_probs = _mode_probs_numpy(
                    mode_mdl, tour_rows, pp, skims, skim_lookup,
                    coef_override=_mode_coef_override
                )
                if mode_probs is not None:
                    logger.info(f"Mode numpy fallback succeeded [{seg}]")

            if mode_probs is not None:
                alt_arr   = np.array(alt_ids)

                # Segment-level share stabilization to avoid DA-collapse in
                # fallback prediction paths (especially when coefficient caches
                # are incomplete after reloads).
                _target_shares = seg_models.get('pred_mode_shares')
                if _target_shares is None:
                    _fp = seg_models.get('frozen_probs')
                    if isinstance(_fp, pd.DataFrame) and len(_fp) > 0:
                        try:
                            _target_shares = {
                                1: float(_fp['p_DA'].mean()),
                                2: float(_fp['p_SR'].mean()),
                                3: float(_fp['p_Walk'].mean()),
                                4: float(_fp['p_Bike'].mean()),
                                5: float(_fp['p_Transit'].mean()),
                            }
                        except Exception:
                            _target_shares = None
                mode_probs = _stabilize_mode_probs_stage5(
                    mode_probs, alt_ids, _target_shares, seg=seg
                )

                row_sums  = mode_probs.sum(axis=1, keepdims=True)
                row_sums  = np.where(row_sums > 0, row_sums, 1.0)
                mode_norm = mode_probs.astype(np.float64) / row_sums
                cum_probs = np.cumsum(mode_norm, axis=1)
                u         = rng.uniform(size=len(seg_rows))
                chosen    = (cum_probs < u[:, np.newaxis]).sum(axis=1)
                chosen    = np.clip(chosen, 0, len(alt_ids) - 1)
                df.loc[seg_rows.index, 'pred_mode'] = alt_arr[chosen]
            # if both fail pred_mode stays as default DA (1) set at top of _predict_batch

        except Exception as e:
            logger.warning(f"Batch mode prediction failed for {seg}: {e} - using DA")

    # ── final guard: replace any DTAZ <= 0 with OTAZ (never output zone 0) ──
    _dtaz_invalid = df['DTAZ'] <= 0
    if _dtaz_invalid.any():
        _otaz_fb = pd.to_numeric(df.loc[_dtaz_invalid, 'OTAZ'], errors='coerce').fillna(1).astype(int).clip(lower=1)
        df.loc[_dtaz_invalid, 'DTAZ'] = _otaz_fb
        logger.warning(
            f"_predict_batch: {int(_dtaz_invalid.sum())} rows had DTAZ<=0 - "
            f"replaced with OTAZ (valid zones are 1-329)"
        )

    return df


def _build_trip_chain(
    activities: pd.DataFrame,
    models: dict,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    skims,
    skim_lookup: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Convert one-row-per-activity Stage 4 output into a full trip chain.

    Uses vectorised batch prediction (one DataTree per segment, not per activity)
    to avoid sharrow recompilation on every call.
    """
    import xarray as xr

    df = activities.copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})
    if 'purpose' not in df.columns and 'DESTPURP' in df.columns:
        df = df.rename(columns={'DESTPURP': 'purpose'})
    if 'purpose' not in df.columns and 'primary_purpose' in df.columns:
        df = df.rename(columns={'primary_purpose': 'purpose'})
    if 'purpose' not in df.columns and 'Purpose' in df.columns:
        df = df.rename(columns={'Purpose': 'purpose'})
    if 'purpose' not in df.columns:
        raise KeyError(
            "_build_trip_chain requires a 'purpose' column "
            "(or DESTPURP/primary_purpose/Purpose alias)"
        )
    if 'main_purpose' not in df.columns and 'purpose' in df.columns:
        df['main_purpose'] = df['purpose']

    # Zone 0 is never a valid origin - NaN/invalid OTAZ → zone 1 (prevents
    # DTAZ defaulting to 0 when the model falls back to DTAZ=OTAZ).
    df['OTAZ'] = pd.to_numeric(df['OTAZ'], errors='coerce').fillna(1).astype(int).clip(lower=1)

    # ── Deduplicate activities: drop rows with identical person+purpose+STARTIME ──
    # Stage 4 batch processing can produce duplicate rows when a person appears
    # in multiple zone batches. Deduplicate before building the chain.
    before_dedup = len(df)
    df = df.sort_values(['person_id', 'STARTIME']).reset_index(drop=True)
    df = df.drop_duplicates(
        subset=['person_id', 'purpose', 'STARTIME'], keep='first'
    ).reset_index(drop=True)
    if len(df) < before_dedup:
        logger.info(
            f"_build_trip_chain: deduplication removed "
            f"{before_dedup - len(df)} duplicate activity rows "
            f"({before_dedup} → {len(df)})"
        )

    # ── Enforce per-person activity cap BEFORE chain building ─────────────
    # Stage 4 can schedule many activities; cap at a realistic daily maximum.
    # Keep mandatory activities (work/edu) first, then fill with nonmand.
    MAX_ACTS_PER_PERSON = 6   # realistic daily cap: 1-2 mandatory + up to 4 nonmand

    def _cap_activities(grp):
        mandatory = grp[grp['purpose'].isin(['work', 'edu'])]
        nonmand   = grp[~grp['purpose'].isin(['work', 'edu'])]
        n_mand    = min(len(mandatory), 2)   # max 2 mandatory (e.g. work + edu)
        n_nm      = min(len(nonmand), MAX_ACTS_PER_PERSON - n_mand)
        kept = pd.concat([mandatory.head(n_mand), nonmand.head(n_nm)])
        return kept.sort_values('STARTIME')

    before_cap = len(df)
    df = df.groupby('person_id', group_keys=False).apply(_cap_activities).reset_index(drop=True)
    if len(df) < before_cap:
        logger.info(
            f"_build_trip_chain: activity cap ({MAX_ACTS_PER_PERSON}/person) "
            f"removed {before_cap - len(df)} excess activities "
            f"({before_cap} → {len(df)})"
        )

    # assign segment column for batch routing
    df['segment'] = df['purpose'].apply(_purpose_to_seg)

    # ── Home zone: use persons table OTAZ, NOT max(activity OTAZ) ──────────
    # max(OTAZ) gives the highest zone-ID number, which is almost never home.
    # Priority: persons.OTAZ → persons.HOME_TAZ → persons.home_taz → fallback.
    pp_home = persons.copy()
    if 'person_id' not in pp_home.columns and 'PERSID' in pp_home.columns:
        pp_home = pp_home.rename(columns={'PERSID': 'person_id'})
    home_zone_col = next(
        (c for c in ['OTAZ', 'HOME_TAZ', 'home_taz', 'home_zone']
         if c in pp_home.columns), None
    )
    if home_zone_col:
        home_otaz = {
            str(pid): max(int(z), 1)
            for pid, z in pp_home.set_index('person_id')[home_zone_col].items()
            if not pd.isna(z) and int(z) > 0
        }
    else:
        # Absolute fallback: use the OTAZ that appears most often for each person
        # (usually the home zone because home dwell dominates the day)
        home_otaz = {
            str(pid): int(grp['OTAZ'].mode().iloc[0])
            for pid, grp in df.groupby('person_id')
            if len(grp) > 0
        }
    logger.info(
        f"_build_trip_chain: home zones from "
        f"{'persons.' + home_zone_col if home_zone_col else 'OTAZ mode fallback'} "
        f"({len(home_otaz)} persons)"
    )

    # ensure ptour_id
    if 'ptour_id' not in df.columns:
        df['ptour_id'] = df['person_id'].astype(str) + '_' + df.index.astype(str)

    pp = persons.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})
    pp['person_id'] = pp['person_id'].astype(str)

    # ── build usual location lookups ──────────────────────────────────
    # Priority: WORK_TAZ / SCHOOL_TAZ on the activities DataFrame (injected by
    # run_stage5 from usual_location.py).  Fallback: read from persons table
    # directly (in case run_stage5 merge was skipped or failed).
    work_taz_map   = {}
    school_taz_map = {}
    for _col, _map_ref, _label in [
        ('WORK_TAZ',   'work_taz_map',   'workers'),
        ('SCHOOL_TAZ', 'school_taz_map', 'students'),
    ]:
        if _col in df.columns:
            _map = {
                str(pid): int(z)
                for pid, z in df.groupby('person_id')[_col].first().items()
                if not pd.isna(z) and int(z) > 0
            }
        elif _col in pp.columns:
            # Fallback: read from persons (usual_location.py may have added it there)
            _map = {
                str(pid): int(z)
                for pid, z in pp.set_index('person_id')[_col].items()
                if not pd.isna(z) and int(z) > 0
            }
        else:
            _map = {}
        if _label == 'workers':
            work_taz_map   = _map
        else:
            school_taz_map = _map
        if _map:
            logger.info(f"_build_trip_chain: using {_col} for {len(_map)} {_label}")

    rng = np.random.default_rng(seed)

    # ── STEP 1: vectorised batch prediction ───────────────────────────
    print("  Running vectorised destination + mode prediction...")
    _pred_ok = False
    if LARCH_AVAILABLE and models:
        try:
            df = _predict_batch(df, models, pp, landuse, skims,
                                work_taz_map, school_taz_map, rng,
                                skim_lookup=skim_lookup)
            _pred_ok = True
        except Exception as _pb_e:
            logger.error(
                f"_predict_batch failed: {_pb_e!r}\n"
                "  Falling back to gravity-only destination prediction."
            )
    if not _pred_ok:
        _strict = os.environ.get('STAGE5_STRICT_ESTIMATION', '1') != '0'
        if _strict:
            raise RuntimeError(
                "_build_trip_chain: destination/mode prediction failed; "
                "strict mode forbids gravity-only fallback"
            )
        # Gravity-only fallback: keep models=None path (dest from gravity, mode=DA)
        df['DTAZ']      = df['OTAZ'].astype(int)
        df['pred_mode'] = 1
        if not models:
            logger.warning(
                "_build_trip_chain: models dict is empty - no Larch models were "
                "estimated or loaded.  Check that _run_segment produced results and "
                "that save_path/load_path are correct."
            )
    print(f"  Batch prediction complete: {df['DTAZ'].nunique()} unique destination zones "
          f"({'model' if _pred_ok else 'gravity/default fallback'})")

    # ── STEP 1b: purpose- and duration-aware distance cap for nonmandatory
    # destinations ──────────────────────────────────────────────────────
    # The destination model's distance decay is too weak and can assign
    # implausibly distant zones to nonmandatory activities — e.g. a
    # 20-minute escort trip landing 44km away, or a preschooler's escort/
    # edu activity landing 70km+ from home. A single flat 80km cap missed
    # these because they fall well under 80km. Cap per purpose (local-
    # serving purposes like escort/eat get a tighter ceiling than social,
    # which can legitimately span further), and tighten further for short
    # activities where round-trip travel time would dwarf the activity
    # itself. This remains a post-hoc guard (reset to home zone), not a
    # fix to the underlying weak distance decay in the choice model —
    # see stage5 GA re-estimation for the real fix.
    MAX_NONMAND_DIST_KM = {
        'escort':   25.0,   # local pickup/drop-off, not regional
        'eat':      30.0,
        'shopping': 40.0,
        'othmaint': 40.0,
        'othdiscr': 40.0,
        'social':   60.0,
    }
    _DEFAULT_NONMAND_DIST_KM = 60.0
    _SHORT_ACT_MAX_MIN       = 30.0   # activities this short or shorter...
    _SHORT_ACT_DIST_KM       = 15.0   # ...get capped tighter than their purpose ceiling
    _nonmand_mask = ~df['purpose'].isin(['work', 'edu'])
    if _nonmand_mask.any() and skim_lookup:
        _cap_count = 0
        _has_dur = 'predicted_duration_min' in df.columns
        for idx in df.index[_nonmand_mask]:
            pid_str = str(df.at[idx, 'person_id'])
            h_zone  = int(home_otaz.get(pid_str, home_otaz.get(df.at[idx, 'person_id'], 1)))
            d_zone  = int(df.at[idx, 'DTAZ'])
            skim_v  = skim_lookup.get((h_zone, d_zone), {})
            dist_km = skim_v.get('DIST', skim_v.get('dist', None))
            if dist_km is None:
                continue
            dist_km = float(dist_km)
            purp   = df.at[idx, 'purpose']
            max_km = MAX_NONMAND_DIST_KM.get(purp, _DEFAULT_NONMAND_DIST_KM)
            if _has_dur:
                act_dur = df.at[idx, 'predicted_duration_min']
                if pd.notna(act_dur) and float(act_dur) <= _SHORT_ACT_MAX_MIN:
                    max_km = min(max_km, _SHORT_ACT_DIST_KM)
            if dist_km > max_km:
                df.at[idx, 'DTAZ'] = h_zone
                _cap_count += 1
        if _cap_count:
            logger.info(
                f"_build_trip_chain: capped {_cap_count} nonmandatory destinations "
                f"exceeding purpose/duration-aware distance limits → home zone fallback"
            )

    # ── STEP 2: build tour-structured trip chain ───────────────────────────
    # ActivitySim tour model: each out-of-home activity is its own HOME-BASED
    # TOUR: home → activity → home.  Mandatory activities come first; non-
    # mandatory tours are placed in the remaining time windows.
    #
    # For each person:
    #   1. Sort activities by STARTIME (Stage 4 scheduled order).
    #   2. Group into tours: mandatory activities each form a simple tour;
    #      non-mandatory activities with overlapping/adjacent windows share a tour.
    #   3. Build the chain for each tour, then the full day chain:
    #        home → [mandatory tour 1] → home → [mandatory tour 2] → home
    #             → [nonmand tour 1]  → home → [nonmand tour 2]   → home
    #   4. STARTIME for each trip uses the activity's scheduled start time from
    #      Stage 4, minus the predicted travel time, so the schedule is preserved.
    #
    # This prevents the "jumbled movement" problem where all activities are
    # chained end-to-end, requiring a person to travel from their morning work
    # zone directly to an evening shopping zone with no return home.
    all_rows = []

    for pid, grp in df.groupby('person_id', sort=False):
        grp      = grp.sort_values('STARTIME').reset_index(drop=True)
        home_z   = int(home_otaz.get(str(pid), home_otaz.get(pid, 1)))

        # Separate mandatory and non-mandatory activities
        mand_grp  = grp[grp['purpose'].isin(['work', 'edu'])].copy()
        nmand_grp = grp[~grp['purpose'].isin(['work', 'edu'])].copy()

        # Build tour list: [(activity_row, ...), ...] - each tuple is one tour
        # Mandatory: one tour per activity (simple work/edu tour)
        # Non-mandatory: group activities that are within 30 min of each other
        #   into a joint tour (e.g. shopping + eat in the same trip out)
        tours = []
        for _, row in mand_grp.iterrows():
            tours.append(('mandatory', [row]))

        # Group non-mandatory by proximity in schedule
        if not nmand_grp.empty:
            nm_sorted = nmand_grp.sort_values('STARTIME').reset_index(drop=True)
            current_group = [nm_sorted.iloc[0]]
            for i in range(1, len(nm_sorted)):
                prev_end  = float(current_group[-1].get('ENDTIME',
                            current_group[-1]['STARTIME'] +
                            current_group[-1].get('predicted_duration_min', 30)))
                this_start = float(nm_sorted.iloc[i]['STARTIME'])
                # If gap between activities < 60 min, bundle into same tour
                if this_start - prev_end < 60:
                    current_group.append(nm_sorted.iloc[i])
                else:
                    tours.append(('nonmand', current_group))
                    current_group = [nm_sorted.iloc[i]]
            tours.append(('nonmand', current_group))

        # Sort all tours by their first activity's STARTIME
        tours.sort(key=lambda t: float(t[1][0]['STARTIME']))

        # ── day-level mode consistency ─────────────────────────────────
        # Tour modes are drawn independently per activity, which produced
        # implausible day patterns like transit -> drive -> transit. Tours
        # are home-based so the car IS home between tours, but a person who
        # has already chosen transit for an earlier tour has revealed a
        # no-car day (car unavailable / parked / preference) - snapping
        # back to driving alone mid-day then returning to transit is not
        # credible behaviour. Rule: once a transit tour has occurred, later
        # DA tours are reassigned to the previous tour's mode.
        _day_modes = [int(t[1][0].get('pred_mode', 1)) for t in tours]
        _seen_transit = False
        for _ti in range(len(_day_modes)):
            if _seen_transit and _day_modes[_ti] == 1:      # 1 = DA
                _day_modes[_ti] = _day_modes[_ti - 1] if _ti > 0 else 2
            if _day_modes[_ti] == 5:                        # 5 = Transit
                _seen_transit = True

        # Excluded carry-through columns (set explicitly on each row)
        attr_cols = [c for c in grp.columns
                     if c not in ('STARTIME', 'ENDTIME', 'DTAZ', 'OTAZ',
                                  'purpose', 'predicted_duration_min',
                                  'main_purpose', 'primary_purpose',
                                  '_priority', 'segment', 'pred_mode',
                                  'mode', 'mode_name', 'mode_code',
                                  'tour_mode', 'travtime', 'row_type')]

        global_seq = 0

        for tour_idx, (tour_type, tour_acts) in enumerate(tours):
            # ── tour-level mode: first activity's pred_mode, after the
            # day-level consistency pass above (_day_modes) ────────────────
            first_act   = tour_acts[0]
            tour_mode   = _day_modes[tour_idx]
            first_dtaz  = int(first_act.get('DTAZ', home_z))

            # ── outbound leg: home → first activity ───────────────────────
            # Use the activity's scheduled STARTIME from Stage 4 as the
            # ARRIVAL time at the destination, then subtract travel time to
            # get the departure time from home.
            outbound_tt = _get_tt(skim_lookup, home_z, first_dtaz, tour_mode)
            first_startime = float(first_act.get('STARTIME', 480))
            # Departure from home = scheduled arrival − travel time
            depart_home = max(first_startime - outbound_tt, 0.0)

            out_trip = {
                'person_id':              pid,
                'row_type':               'trip',
                'seq':                    global_seq,
                'purpose':                str(first_act.get('purpose', 'travel')),
                'OTAZ':                   home_z,
                'DTAZ':                   first_dtaz,
                'STARTIME':               depart_home,
                'ENDTIME':                first_startime,
                'mode':                   tour_mode,
                'mode_name':              MODE_NAMES.get(tour_mode, 'DA'),
                'travtime':               outbound_tt,
                'predicted_duration_min': outbound_tt,
            }
            for c in attr_cols:
                out_trip[c] = first_act.get(c, None)
            all_rows.append(out_trip)
            global_seq += 1

            # ── activity dwells + intra-tour legs ────────────────────────
            cursor_zone = first_dtaz
            for act_i, act_row in enumerate(tour_acts):
                dtaz      = int(act_row.get('DTAZ', cursor_zone))
                act_start = float(act_row.get('STARTIME', 480))
                act_dur   = float(act_row.get('predicted_duration_min', 30))
                act_end   = float(act_row.get('ENDTIME', act_start + act_dur))

                # intra-tour leg (only for 2nd+ activity in a joint tour)
                if act_i > 0:
                    leg_tt = _get_tt(skim_lookup, cursor_zone, dtaz, tour_mode)
                    prev_act_end = float(tour_acts[act_i - 1].get(
                        'ENDTIME',
                        tour_acts[act_i - 1]['STARTIME'] +
                        tour_acts[act_i - 1].get('predicted_duration_min', 30)
                    ))
                    leg_start = max(prev_act_end, act_start - leg_tt)
                    intra_trip = {
                        'person_id':              pid,
                        'row_type':               'trip',
                        'seq':                    global_seq,
                        'purpose':                str(act_row.get('purpose', 'travel')),
                        'OTAZ':                   cursor_zone,
                        'DTAZ':                   dtaz,
                        'STARTIME':               leg_start,
                        'ENDTIME':                leg_start + leg_tt,
                        'mode':                   tour_mode,
                        'mode_name':              MODE_NAMES.get(tour_mode, 'DA'),
                        'travtime':               leg_tt,
                        'predicted_duration_min': leg_tt,
                    }
                    for c in attr_cols:
                        intra_trip[c] = act_row.get(c, None)
                    all_rows.append(intra_trip)
                    global_seq += 1

                # activity dwell
                act_out = {
                    'person_id':              pid,
                    'row_type':               'activity',
                    'seq':                    global_seq,
                    'purpose':                str(act_row.get('purpose', '')),
                    'OTAZ':                   dtaz,
                    'DTAZ':                   dtaz,
                    'STARTIME':               act_start,
                    'ENDTIME':                act_end,
                    'mode':                   0,
                    'mode_name':              '',
                    'travtime':               0.0,
                    'predicted_duration_min': act_dur,
                }
                for c in attr_cols:
                    act_out[c] = act_row.get(c, None)
                all_rows.append(act_out)
                global_seq += 1
                cursor_zone = dtaz

            # ── return home leg ────────────────────────────────────────
            last_act    = tour_acts[-1]
            last_dtaz   = int(last_act.get('DTAZ', cursor_zone))
            return_tt   = _get_tt(skim_lookup, last_dtaz, home_z, tour_mode)
            last_end    = float(last_act.get('ENDTIME',
                          last_act['STARTIME'] +
                          last_act.get('predicted_duration_min', 30)))

            home_trip = {
                'person_id':              pid,
                'row_type':               'trip',
                'seq':                    global_seq,
                'purpose':                'home',
                'OTAZ':                   last_dtaz,
                'DTAZ':                   home_z,
                'STARTIME':               last_end,
                'ENDTIME':               last_end + return_tt,
                'mode':                   tour_mode,
                'mode_name':              MODE_NAMES.get(tour_mode, 'DA'),
                'travtime':               return_tt,
                'predicted_duration_min': return_tt,
            }
            for c in attr_cols:
                home_trip[c] = last_act.get(c, None)
            all_rows.append(home_trip)
            global_seq += 1

            # ── home dwell between tours ───────────────────────────────────
            # Insert a 'home' activity row between every pair of consecutive tours
            # so the plan has no back-to-back trip rows at tour boundaries.
            if tour_idx < len(tours) - 1:
                home_arrive = last_end + return_tt
                next_first_act = tours[tour_idx + 1][1][0]
                next_outbound_tt = _get_tt(
                    skim_lookup, home_z,
                    int(next_first_act.get('DTAZ', home_z)),
                    int(next_first_act.get('pred_mode', 1)),
                )
                next_depart = max(
                    float(next_first_act.get('STARTIME', home_arrive + 30)) - next_outbound_tt,
                    home_arrive,
                )
                if next_depart > home_arrive:
                    home_dwell = {
                        'person_id':              pid,
                        'row_type':               'activity',
                        'seq':                    global_seq,
                        'purpose':                'home',
                        'OTAZ':                   home_z,
                        'DTAZ':                   home_z,
                        'STARTIME':               home_arrive,
                        'ENDTIME':                next_depart,
                        'mode':                   0,
                        'mode_name':              '',
                        'travtime':               0.0,
                        'predicted_duration_min': next_depart - home_arrive,
                    }
                    for c in attr_cols:
                        home_dwell[c] = last_act.get(c, None)
                    all_rows.append(home_dwell)
                    global_seq += 1

    plan = pd.DataFrame(all_rows)
    plan['OTAZ'] = plan['OTAZ'].fillna(1).astype(int).clip(lower=1)
    # Replace NaN DTAZ with OTAZ so zone 0 never appears in the output
    plan['DTAZ'] = plan['DTAZ'].fillna(plan['OTAZ']).astype(int)
    # Guard: any remaining 0s → replace with OTAZ (already ≥1)
    plan.loc[plan['DTAZ'] <= 0, 'DTAZ'] = plan.loc[plan['DTAZ'] <= 0, 'OTAZ']
    plan['STARTIME'] = plan['STARTIME'].fillna(0)
    plan['ENDTIME']  = plan['ENDTIME'].fillna(0)

    # ── Vehicle ID assignment + joint household travel coordination ────────
    plan = _assign_vehicle_ids(plan, persons)

    return plan


def _assign_vehicle_ids(
    plan: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign stable vehicle_id to every DA/SR trip row.

    Rules
    -----
    1. Each household has vehrd vehicles numbered {hhid}_v1, {hhid}_v2, ...
    2. DA trips consume one vehicle slot (cycling through the household fleet).
    3. SR trips that are 'joint' (same hhid, overlapping time window ±30 min,
       same DTAZ) share the DA driver's vehicle_id - they travel together.
    4. SR trips with no matching DA driver get a {hhid}_sr_{n} pseudo-vehicle.
    5. Non-motorised trips (Walk/Bike) and Transit get vehicle_id = ''.

    Joint household detection
    -------------------------
    Two trip-rows are considered joint if they share:
      • same hhid
      • same DTAZ
      • STARTIME within JOINT_TIME_WINDOW minutes of each other

    The DA person's vehicle_id is propagated to all SR co-travellers.

    This ensures the total number of unique vehicle-time-slots (car trips)
    is stable and consistent with the household vehicle fleet.
    """
    JOINT_TIME_WINDOW = 30.0   # minutes - trips within this window are co-travellers
    DA_MODE  = 1
    SR_MODE  = 2
    MOTOR_MODES = {DA_MODE, SR_MODE}

    plan = plan.copy()
    plan['vehicle_id'] = ''

    # ── Build person → household mapping ──────────────────────────────────
    pp = persons.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})
    pp['person_id'] = pp['person_id'].astype(str)

    hhid_col = next((c for c in ['hhid', 'HHID', 'hh_id', 'HH_ID', 'household_id']
                     if c in pp.columns), None)
    vehrd_col = next((c for c in ['vehrd', 'VEHRD', 'veh_rd', 'num_vehicles']
                      if c in pp.columns), None)

    if hhid_col:
        pid_to_hhid  = pp.set_index('person_id')[hhid_col].astype(str).to_dict()
        if vehrd_col:
            hhid_to_vehrd = (
                pp.groupby(hhid_col)[vehrd_col]
                .first()
                .apply(lambda x: max(int(pd.to_numeric(x, errors='coerce') or 1), 1))
                .to_dict()
            )
            # key is hhid value, not the column
            hhid_to_vehrd = {str(k): v for k, v in hhid_to_vehrd.items()}
        else:
            hhid_to_vehrd = {}
    else:
        # No household column - every person is their own household
        pid_to_hhid   = {str(pid): str(pid) for pid in plan['person_id'].unique()}
        hhid_to_vehrd = {}

    plan['_hhid'] = plan['person_id'].astype(str).map(pid_to_hhid).fillna(
        plan['person_id'].astype(str)
    )

    # ── Extract motorised trip rows only ──────────────────────────────────
    trip_mask = (plan['row_type'] == 'trip') & plan['mode'].isin(MOTOR_MODES)
    if not trip_mask.any():
        plan = plan.drop(columns=['_hhid'])
        return plan

    trips = plan.loc[trip_mask].copy()
    trips['_vehrd'] = trips['_hhid'].map(hhid_to_vehrd).fillna(1).astype(int).clip(lower=1)

    # ── Phase 1: assign vehicle slots to DA trips ─────────────────────────
    # Counter per household tracks how many vehicles have been allocated
    # (cycling: person n uses vehicle (n-1) % vehrd + 1).
    hhid_da_counter: dict = {}   # hhid → int (number of DA trips allocated so far)
    da_idx_to_vid: dict   = {}   # plan row index → vehicle_id string

    da_trips = trips[trips['mode'] == DA_MODE].sort_values(['_hhid', 'STARTIME'])
    for row_idx, row in da_trips.iterrows():
        hhid  = str(row['_hhid'])
        vehrd = int(row['_vehrd'])
        n_so_far = hhid_da_counter.get(hhid, 0)
        slot = (n_so_far % vehrd) + 1
        vid  = f"{hhid}_v{slot}"
        da_idx_to_vid[row_idx] = vid
        hhid_da_counter[hhid] = n_so_far + 1

    plan.loc[list(da_idx_to_vid.keys()), 'vehicle_id'] = pd.Series(da_idx_to_vid)

    # ── Phase 2: joint SR trips - share DA vehicle ────────────────────────
    # Build a lookup: (hhid, DTAZ, time_bucket) → vehicle_id from DA trips
    # time_bucket = floor(STARTIME / JOINT_TIME_WINDOW) for quick matching
    da_lookup: dict = {}   # (hhid, dtaz, bucket) → vehicle_id
    for row_idx, vid in da_idx_to_vid.items():
        row = plan.loc[row_idx]
        hhid   = str(row['_hhid'])
        dtaz   = int(row['DTAZ'])
        bucket = int(float(row['STARTIME']) // JOINT_TIME_WINDOW)
        # register for this bucket and adjacent buckets (±1) to catch boundary trips
        for b in (bucket - 1, bucket, bucket + 1):
            key = (hhid, dtaz, b)
            if key not in da_lookup:
                da_lookup[key] = vid

    # SR trips: try to match a DA vehicle; else assign a pseudo-vehicle
    sr_trips = trips[trips['mode'] == SR_MODE]
    hhid_sr_counter: dict = {}

    for row_idx, row in sr_trips.iterrows():
        hhid   = str(row['_hhid'])
        dtaz   = int(row['DTAZ'])
        bucket = int(float(row['STARTIME']) // JOINT_TIME_WINDOW)
        vid = None
        for b in (bucket - 1, bucket, bucket + 1):
            vid = da_lookup.get((hhid, dtaz, b))
            if vid:
                break
        if vid is None:
            # No matching DA driver - SR is either a taxi/rideshare or lone SR
            n_sr = hhid_sr_counter.get(hhid, 0)
            vid  = f"{hhid}_sr{n_sr + 1}"
            hhid_sr_counter[hhid] = n_sr + 1
        plan.loc[row_idx, 'vehicle_id'] = vid

    plan = plan.drop(columns=['_hhid'])
    n_da  = trip_mask.sum()
    n_vid = (plan['vehicle_id'] != '').sum()
    logger.info(
        f"_assign_vehicle_ids: {n_da} motorised trips → "
        f"{n_vid} vehicle_id assignments "
        f"({plan.loc[plan['vehicle_id']!='','vehicle_id'].nunique()} unique vehicles)"
    )
    return plan


# ===========================================================================
# SKIM ATTACHMENT
# ===========================================================================

def _attach_skims(df, skim_df):
    alias_defaults = {
        'tt_avg': ('SOVTOLL_TIME__AM', 30.0),
        'tt_am': ('SOVTOLL_TIME__AM', 30.0),
        'tt_pm': ('SOVTOLL_TIME__PM', 30.0),
        'dist': ('SOVTOLL_DIST__AM', 20.0),
        'crash': ('CRASH', 0.0),
    }
    skim_cols = ['tt_avg', 'tt_am', 'tt_pm', 'log_dist', 'dist', 'distwalk', 'distbike', 'crash']
    existing = [c for c in skim_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    if not skim_df.empty:
        keep_cols = ['OTAZ', 'DTAZ'] + [c for c in set(alias_defaults.values()) if isinstance(c, str)]
        keep_cols = ['OTAZ', 'DTAZ']
        for src, _default in alias_defaults.values():
            if src in skim_df.columns and src not in keep_cols:
                keep_cols.append(src)
        for src in ['DISTWALK', 'DISTBIKE']:
            if src in skim_df.columns and src not in keep_cols:
                keep_cols.append(src)
        merged = skim_df[keep_cols].drop_duplicates(['OTAZ', 'DTAZ']).copy()
        for out_col, (src_col, default) in alias_defaults.items():
            if src_col in merged.columns:
                merged[out_col] = pd.to_numeric(merged[src_col], errors='coerce').fillna(default)
            else:
                merged[out_col] = default
        if 'DISTWALK' in merged.columns:
            # DISTWALK already in km
            merged['distwalk'] = pd.to_numeric(merged['DISTWALK'], errors='coerce').fillna(20.0)
        else:
            merged['distwalk'] = 20.0
        if 'DISTBIKE' in merged.columns:
            # DISTBIKE already in km
            merged['distbike'] = pd.to_numeric(merged['DISTBIKE'], errors='coerce').fillna(20.0)
        else:
            merged['distbike'] = 2.0
        merged['dist'] = pd.to_numeric(merged['dist'], errors='coerce').fillna(20.0)
        merged['log_dist'] = np.log(np.clip(merged['dist'].astype(float), 0.0, None) + 1.0)
        df = df.merge(
            merged[['OTAZ', 'DTAZ'] + skim_cols].drop_duplicates(['OTAZ', 'DTAZ']),
            on=['OTAZ', 'DTAZ'], how='left')
    for col in skim_cols:
        if col in ['dist', 'distwalk', 'distbike', 'tt_avg', 'tt_am', 'tt_pm']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0) if col in df.columns else 0.0
        elif col == 'log_dist':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0) if col in df.columns else 0.0
        else:
            df[col] = df[col].fillna(0) if col in df.columns else 0.0
    return df


# ===========================================================================
# SAVE / LOAD
# ===========================================================================

def save_models(models, path='saved_data/stage5_models.pkl'):
    """
    Save fitted Larch models to disk.

    Larch Model objects contain lambda functions (from DictOfAlts etc.)
    that standard pickle cannot serialise. We try dill first (handles lambdas),
    then fall back to saving coefficient DataFrames only.

    DataTree objects (dt_mode, dt_dest, logsums) are always stripped before
    saving - they reference live OMX handles and xarray objects that can't be
    meaningfully serialised.
    """
    from paper_model_forms import write_stage_paper_artifacts
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # Strip un-serialisable DataTree / xarray objects - they're not needed
    # for reload since the model parameters are in mode_model.pf / dest_model.pf
    STRIP_KEYS = {'dt_mode', 'dt_dest', 'logsums'}

    # Before stripping, inject _coef_cache into each mode/dest model so that
    # _mode_probs_numpy can read coefficients without calling .pf on the loaded
    # model (which triggers ._mangled and crashes on newer Larch versions).
    models_slim = {}
    for seg, res in models.items():
        slim_res = {}
        for k, v in res.items():
            if k in STRIP_KEYS:
                continue
            if k in ('mode_model', 'dest_model') and v is not None:
                try:
                    coef_dict = {}
                    direct_key = 'mode_coef' if k == 'mode_model' else 'dest_coef'
                    if isinstance(res.get(direct_key), dict) and res.get(direct_key):
                        coef_dict = {str(n): float(val) for n, val in res[direct_key].items() if np.isfinite(float(val))}
                    elif hasattr(v, "_coef_cache") and getattr(v, "_coef_cache", {}):
                        coef_dict = {str(n): float(val) for n, val in v._coef_cache.items() if np.isfinite(float(val))}
                    elif hasattr(v, "_final_pf") and isinstance(v._final_pf, pd.DataFrame) and "value" in v._final_pf.columns:
                        coef_dict = {
                            str(name): float(v._final_pf.loc[name, "value"])
                            for name in v._final_pf.index
                            if np.isfinite(float(v._final_pf.loc[name, "value"]))
                        }
                    elif hasattr(v, "pf"):
                        pf = v.pf
                        coef_dict = {
                            str(name): float(pf.loc[name, "value"])
                            for name in pf.index
                            if np.isfinite(float(pf.loc[name, "value"]))
                        }
                    if coef_dict:
                        v._coef_cache = coef_dict
                        print(f"  _coef_cache injected [{seg}/{k}]: {len(v._coef_cache)} params")
                except Exception as _ce:
                    logger.debug(f"  _coef_cache injection failed [{seg}/{k}]: {_ce}")
            slim_res[k] = v
        models_slim[seg] = slim_res

    # Serialisation order:
    #   1. cloudpickle - ships with most conda ML environments, handles lambdas
    #   2. dill        - fallback if cloudpickle unavailable
    #   3. pickle      - last resort; will fail on DictOfAlts lambdas but worth trying
    # If all three fail the coefficient CSV is still written below.
    saved = False

    try:
        import cloudpickle
        with open(path, 'wb') as f:
            cloudpickle.dump(models_slim, f)
        print(f"  Stage 5 models saved (cloudpickle) → {path}")
        saved = True
    except Exception as cp_err:
        logger.warning(f"cloudpickle failed ({cp_err}), trying dill...")

    if not saved:
        try:
            import dill
            with open(path, 'wb') as f:
                dill.dump(models_slim, f)
            print(f"  Stage 5 models saved (dill) → {path}")
            saved = True
        except Exception as dill_err:
            logger.warning(f"dill failed ({dill_err}), trying pickle...")

    if not saved:
        try:
            with open(path, 'wb') as f:
                pickle.dump(models_slim, f)
            print(f"  Stage 5 models saved (pickle) → {path}")
            saved = True
        except Exception as pkl_err:
            logger.warning(
                f"pickle also failed ({pkl_err}). "
                "Models stay in memory only - coefficient CSV still saved. "
                "To fix: conda install cloudpickle  or  pip install cloudpickle"
            )

    # Always save human-readable CSV + HTML - wrapped in try/except so a
    # crash here can't mask a successful PKL save above.
    coef_path = path.replace('.pkl', '_coefficients.csv')
    try:
        _save_coefficients(models, coef_path)
    except Exception as _ce:
        logger.error(f"_save_coefficients failed: {_ce} - coefficients not saved to {coef_path}")
        import traceback; logger.debug(traceback.format_exc())

    # Always emit a lightweight stage trace artifact for post-mortem debugging,
    # even when model serialization or coefficient extraction partially fails.
    try:
        trace_path = path.replace('.pkl', '_trace.txt')
        lines = []
        lines.append('Stage 5 Trace')
        lines.append(f'path: {path}')
        lines.append(f'segments: {list((models or {}).keys())}')
        for _seg, _res in (models or {}).items():
            _mk = sorted(list(_res.keys())) if isinstance(_res, dict) else []
            _mcoef = len(_res.get('mode_coef', {})) if isinstance(_res.get('mode_coef', {}), dict) else 0
            _dcoef = len(_res.get('dest_coef', {})) if isinstance(_res.get('dest_coef', {}), dict) else 0
            _mobj = _res.get('mode_model') is not None if isinstance(_res, dict) else False
            _dobj = _res.get('dest_model') is not None if isinstance(_res, dict) else False
            lines.append(
                f'[{_seg}] mode_model={_mobj} dest_model={_dobj} '
                f'mode_coef_n={_mcoef} dest_coef_n={_dcoef} keys={_mk}'
            )
        with open(trace_path, 'w', encoding='utf-8') as _tf:
            _tf.write('\n'.join(lines) + '\n')
        print(f"  Stage 5 trace saved → {trace_path}")
    except Exception as _trace_e:
        logger.debug(f"stage5 trace write failed: {_trace_e}")

    try:
        write_stage_paper_artifacts('stage5', os.path.dirname(path) or 'saved_data')
    except Exception as _pa:
        logger.debug(f"write_stage_paper_artifacts failed: {_pa}")


def _safe_larch_setstate(self, state):
    """
    Patched __setstate__ that skips attributes Larch no longer supports.
    Larch changed its API between versions - 'availability_any' was removed.
    Without this patch, loading a model saved with an older Larch crashes with:
      TypeError: Cannot set 'availability_any' on object of type Model
    """
    _SKIP = {'availability_any', 'availability_co', 'availability_ca'}
    for k, v in state.items():
        if k in _SKIP:
            continue
        try:
            setattr(self, k, v)
        except (TypeError, AttributeError):
            logger.debug(f"load_models: skipping larch attribute '{k}' "
                         f"(not supported in this version)")


def _patch_larch_unpickle():
    """Monkey-patch Larch Model.__setstate__ to survive API changes."""
    try:
        import larch
        for cls_name in ('Model', 'NumbaModel'):
            for mod in (larch, getattr(larch, 'model', None)):
                if mod is None:
                    continue
                cls = getattr(mod, cls_name, None)
                if cls is not None and not getattr(cls, '_safe_setstate_patched', False):
                    cls.__setstate__ = _safe_larch_setstate
                    cls._safe_setstate_patched = True
    except Exception as _e:
        logger.debug(f"_patch_larch_unpickle: {_e}")


def load_models(path='saved_data/stage5_models.pkl'):
    """
    Load models saved by save_models().

    Tries cloudpickle → dill → pickle. In each case, patches Larch's
    __setstate__ first so attributes removed in newer Larch versions
    (e.g. 'availability_any') are silently skipped rather than crashing.
    """
    _patch_larch_unpickle()   # must run before any pickle.load

    for loader_name, loader_fn in [
        ('cloudpickle', lambda f: __import__('cloudpickle').load(f)),
        ('dill',        lambda f: __import__('dill').load(f)),
        ('pickle',      lambda f: __import__('pickle').load(f)),
    ]:
        try:
            with open(path, 'rb') as f:
                models = loader_fn(f)
            for _seg, _res in (models or {}).items():
                for _mkey, _ckey in (("mode_model", "mode_coef"), ("dest_model", "dest_coef")):
                    _mdl = _res.get(_mkey)
                    if _mdl is None:
                        continue
                    _coef = _res.get(_ckey, {}) if isinstance(_res.get(_ckey, {}), dict) else {}
                    if (not _coef) and hasattr(_mdl, "_coef_cache") and getattr(_mdl, "_coef_cache", {}):
                        _coef = dict(_mdl._coef_cache)
                    if (not _coef) and hasattr(_mdl, "_final_pf") and isinstance(_mdl._final_pf, pd.DataFrame) and "value" in _mdl._final_pf.columns:
                        _coef = {
                            str(name): float(_mdl._final_pf.loc[name, "value"])
                            for name in _mdl._final_pf.index
                            if np.isfinite(float(_mdl._final_pf.loc[name, "value"]))
                        }
                    if _coef:
                        _mdl._coef_cache = dict(_coef)
                        _res[_ckey] = dict(_coef)
            print(f"Stage 5 models loaded ({loader_name}) ← {path}")
            return models
        except ImportError:
            continue
        except Exception as _e:
            logger.debug(f"load_models [{loader_name}] failed: {_e}")
            continue

    raise RuntimeError(
        f"Could not load Stage 5 models from {path}. "
        "Install cloudpickle: conda install cloudpickle"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOCATION CHOICE SUMMARY  (ActivitySim-style output)
# ─────────────────────────────────────────────────────────────────────────────

def save_location_choice_summary(
    models: dict,
    obs_tours: pd.DataFrame,
    landuse: pd.DataFrame,
    save_dir: str,
    persons: pd.DataFrame = None,
):
    """
    Save ActivitySim-style location choice coefficient tables and validation.

    Called right after CDAP and usual_locations in the pipeline.
    Produces per-segment outputs following ActivitySim destination_choice protocols:

    Outputs (in ``save_dir/location_choice/``):
      - ``{segment}_destination_coefficients.csv``  (coefficient_name, value, constrain, std_err, t_stat)
      - ``{segment}_mode_coefficients.csv``
      - ``location_choice_summary.csv``             (all segments combined)
      - ``location_choice_summary.html``            (formatted report)
      - ``location_choice_validation.csv``          (zone-level predicted vs observed)
    """
    from scipy.stats import norm as _norm

    out_dir = os.path.join(save_dir, 'location_choice')
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []

    for seg, res in models.items():
        if not isinstance(res, dict):
            continue

        for mtype, mdl_key, coef_key in [
            ('dest', 'dest_model', 'dest_coef'),
            ('mode', 'mode_model', 'mode_coef'),
        ]:
            mdl = res.get(mdl_key)
            coef_dict = res.get(coef_key, {})
            if mdl is None and not coef_dict:
                continue

            # Get parameter frame (prefer _final_pf snapshot)
            pf = None
            if mdl is not None:
                if hasattr(mdl, '_final_pf') and isinstance(getattr(mdl, '_final_pf', None), pd.DataFrame):
                    pf = mdl._final_pf
                else:
                    try:
                        pf = mdl.pf
                    except Exception:
                        pass

            # Extract coefficients
            if not coef_dict and pf is not None:
                coef_dict = {str(p): float(pf.loc[p, 'value']) for p in pf.index}

            ll = res.get(f'{mtype}_ll', np.nan)
            try:
                if mdl is not None:
                    _ll = getattr(mdl, '_cached_ll', None)
                    if _ll is not None and np.isfinite(float(_ll)):
                        ll = float(_ll)
            except Exception:
                pass

            seg_rows = []
            param_names = list(pf.index) if pf is not None and len(pf) > 0 else list(coef_dict.keys())
            for pname in param_names:
                val = coef_dict.get(str(pname), float('nan'))
                se, tstat, pval = np.nan, np.nan, np.nan
                holdfast = False
                if pf is not None and pname in pf.index:
                    row_pf = pf.loc[pname]
                    holdfast = bool(row_pf.get('holdfast', False))
                    for _c in ('std_err', 'std_err_opg'):
                        if _c in row_pf.index:
                            _v = float(row_pf[_c])
                            if np.isfinite(_v) and (_v > 0 or holdfast):
                                se = _v
                                break
                    for _c in ('t_stat', 't_stat_opg'):
                        if _c in row_pf.index:
                            _v = float(row_pf[_c])
                            if np.isfinite(_v):
                                tstat = _v
                                break
                    if np.isnan(tstat) and np.isfinite(se) and se > 0:
                        tstat = float(val) / se
                    if np.isfinite(tstat):
                        pval = float(2.0 * _norm.sf(abs(tstat)))

                # ActivitySim constrain column: T = fixed, F = free
                constrain = 'T' if holdfast else 'F'
                r = {
                    'segment': seg,
                    'model_type': mtype,
                    'coefficient_name': str(pname),
                    'value': float(val),
                    'constrain': constrain,
                    'std_err': se,
                    't_stat': tstat,
                    'p_value': pval,
                }
                seg_rows.append(r)
                all_rows.append(r)

            # Per-segment coefficient CSV (ActivitySim format)
            if seg_rows:
                seg_df = pd.DataFrame(seg_rows)
                seg_path = os.path.join(out_dir, f'{seg}_{mtype}_coefficients.csv')
                seg_df.to_csv(seg_path, index=False)
                logger.info(f"  Location choice: {seg}/{mtype} coefficients → {seg_path}")

    # Combined summary CSV
    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        summary_path = os.path.join(out_dir, 'location_choice_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"  Location choice summary → {summary_path}")

        # HTML report
        _write_location_choice_html(summary_df, models, out_dir)

    # Validation: observed vs model destination distributions
    if obs_tours is not None and len(obs_tours) > 0:
        obs_lc = _build_activitysim_location_observations(obs_tours, persons)
        if len(obs_lc) == 0:
            logger.warning(
                "  Location validation: no valid observed location-choice rows after normalization"
            )
        else:
            obs_path = os.path.join(out_dir, 'location_choice_observations.csv')
            choosers_path = os.path.join(out_dir, 'location_choice_choosers.csv')
            choices_path = os.path.join(out_dir, 'location_choice_choices.csv')
            obs_lc.to_csv(obs_path, index=False)
            obs_lc[['chooser_id', 'person_id', 'segment', 'purpose', 'OTAZ']].drop_duplicates().to_csv(
                choosers_path, index=False
            )
            obs_lc[['chooser_id', 'choice']].to_csv(choices_path, index=False)
            logger.info(f"  Location choice observations → {obs_path}")
            logger.info(f"  Location choice choosers → {choosers_path}")
            logger.info(f"  Location choice choices → {choices_path}")
            _val_df, _val_sum = _location_choice_validation(obs_lc, models, landuse, out_dir)
            # Re-render the coefficient HTML with zone validation embedded
            if all_rows and _val_df is not None:
                summary_df = pd.DataFrame(all_rows)
                _write_location_choice_html(summary_df, models, out_dir,
                                            val_df=_val_df, val_summary=_val_sum)

    logger.info(f"  Location choice outputs saved → {out_dir}/")
    return out_dir


def _write_location_choice_html(summary_df: pd.DataFrame, models: dict, out_dir: str,
                                val_df: pd.DataFrame = None, val_summary: pd.DataFrame = None):
    """Write an HTML report of location choice coefficients plus zone validation."""
    sections = []
    for seg in summary_df['segment'].unique():
        seg_df = summary_df[summary_df['segment'] == seg]
        for mtype in seg_df['model_type'].unique():
            mdf = seg_df[seg_df['model_type'] == mtype]
            res = models.get(seg, {}) if models else {}
            ll = res.get(f'{mtype}_ll', np.nan)
            n_tours = res.get('n_tours', '?')
            # Fall back to loglike stored in the summary CSV rows themselves
            if not np.isfinite(float(ll)) and 'loglike' in mdf.columns:
                _ll_vals = pd.to_numeric(mdf['loglike'], errors='coerce').dropna()
                if not _ll_vals.empty:
                    ll = float(_ll_vals.iloc[0])

            rows_html = []
            for _, r in mdf.iterrows():
                t   = r['t_stat']
                p   = r.get('p_value', float('nan'))
                try:
                    at = abs(float(t))
                    sig = '★★★' if at > 3.3 else '★★' if at > 2.6 else '★' if at > 1.96 else ''
                    t_color = '#1a7a1a' if at > 3.3 else '#2e7d32' if at > 2.6 else '#558b2f' if at > 1.96 else '#b71c1c'
                except Exception:
                    sig = ''; t_color = '#888'
                try:
                    pf = float(p)
                    p_str = f'{pf:.4f}' if np.isfinite(pf) else '-'
                    p_color = '#1a7a1a' if pf < 0.001 else '#2e7d32' if pf < 0.01 else '#558b2f' if pf < 0.05 else '#f57c00' if pf < 0.10 else '#b71c1c'
                    p_bold = 'font-weight:bold;' if pf < 0.01 else ''
                except Exception:
                    p_str = '-'; p_color = '#888'; p_bold = ''
                val_str = f"{r['value']:.4f}"
                se_str  = f"{r['std_err']:.4f}" if np.isfinite(float(r['std_err'])) else '-'
                t_str   = f"{float(t):.2f}" if np.isfinite(float(t)) else '-'
                rows_html.append(
                    f'<tr><td>{r["coefficient_name"]}</td>'
                    f'<td style="text-align:right">{val_str}</td>'
                    f'<td style="text-align:center">{r["constrain"]}</td>'
                    f'<td style="text-align:right">{se_str}</td>'
                    f'<td style="text-align:right; color:{t_color}">{t_str} {sig}</td>'
                    f'<td style="text-align:right; color:{p_color};{p_bold}">{p_str}</td></tr>'
                )
            n_free  = int((mdf['constrain'] == 'F').sum())
            n_fixed = int((mdf['constrain'] == 'T').sum())
            try:
                ll_str = f'{float(ll):.2f}' if np.isfinite(float(ll)) else '-'
            except Exception:
                ll_str = '-'
            sections.append(f"""
<div style="margin:20px 0; padding:15px; border:1px solid #ddd; border-radius:8px;">
  <h2>{seg.upper()} - {mtype} choice model</h2>
  <p style="background:#e3f2fd;padding:8px 12px;border-radius:4px;font-size:0.95em;">
    <b>Log-likelihood:</b> {ll_str} &nbsp;|&nbsp;
    <b>N obs:</b> {n_tours} &nbsp;|&nbsp;
    <b>Free params:</b> {n_free} &nbsp;|&nbsp;
    <b>Fixed:</b> {n_fixed}
  </p>
  <table style="border-collapse:collapse; width:100%; font-size:13px;">
    <thead><tr style="background:#1565c0; color:white;">
      <th style="text-align:left; padding:6px;">Coefficient</th>
      <th style="text-align:right; padding:6px;">Value</th>
      <th style="text-align:center; padding:6px;">Fix</th>
      <th style="text-align:right; padding:6px;">Std&nbsp;Err</th>
      <th style="text-align:right; padding:6px;">t-stat</th>
      <th style="text-align:right; padding:6px;">p-value</th>
    </tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
  <p style="font-size:0.8em;color:#666;margin-top:6px;">
    p-value colour: <span style="color:#1a7a1a;font-weight:bold">p&lt;0.001</span>
    <span style="color:#2e7d32;font-weight:bold"> p&lt;0.01</span>
    <span style="color:#558b2f"> p&lt;0.05 ★</span>
    <span style="color:#f57c00"> p&lt;0.10</span>
    <span style="color:#b71c1c"> p≥0.10 (not sig.)</span>
  </p>
</div>""")

    # ── Zone-level validation section ────────────────────────────────────────
    val_sections = []
    if val_df is not None and len(val_df) > 0:
        for seg in val_df['segment'].unique():
            sv = val_df[val_df['segment'] == seg].copy()
            top20 = sv.nlargest(20, 'obs_tours')
            rows_v = []
            for _, r in top20.iterrows():
                pct_obs = f"{r['obs_share']*100:.1f}%"
                pct_pred = f"{r['pred_share']*100:.1f}%" if np.isfinite(r.get('pred_share', np.nan)) else 'n/a'
                ratio = (r['pred_share'] / r['obs_share']) if r['obs_share'] > 0 else 0
                flag = ' ⚠' if ratio > 3 or (r['obs_share'] > 0 and ratio < 0.33) else ''
                rows_v.append(
                    f'<tr><td>{int(r["zone"])}</td>'
                    f'<td style="text-align:right">{int(r["obs_tours"])}</td>'
                    f'<td style="text-align:right">{pct_obs}</td>'
                    f'<td style="text-align:right">{pct_pred}{flag}</td></tr>'
                )
            sum_row = ''
            if val_summary is not None:
                sr = val_summary[val_summary['segment'] == seg]
                if len(sr):
                    r = sr.iloc[0]
                    corr = f"{r['zone_share_corr']:.3f}" if np.isfinite(r.get('zone_share_corr', np.nan)) else 'n/a'
                    rmse = f"{r['zone_share_rmse']:.4f}" if np.isfinite(r.get('zone_share_rmse', np.nan)) else 'n/a'
                    dist = f"{r['mean_trip_dist']:.1f} km" if np.isfinite(r.get('mean_trip_dist', np.nan)) else 'n/a'
                    sum_row = (f'<p>Zone share Pearson r={corr} | RMSE={rmse} | '
                               f'Mean trip dist={dist}</p>')
            val_sections.append(f"""
<div style="margin:20px 0; padding:15px; border:1px solid #aad; border-radius:8px; background:#f8f8ff;">
  <h2>{seg.upper()} - Destination zone validation (top 20 observed)</h2>
  {sum_row}
  <table style="border-collapse:collapse; width:60%; font-size:13px;">
    <thead><tr style="background:#e8eaf6;">
      <th style="text-align:left; padding:6px;">Zone</th>
      <th style="text-align:right; padding:6px;">Obs trips</th>
      <th style="text-align:right; padding:6px;">Obs share</th>
      <th style="text-align:right; padding:6px;">Pred share</th>
    </tr></thead>
    <tbody>{''.join(rows_v)}</tbody>
  </table>
  <p style="font-size:11px; color:#666;">⚠ = predicted share &gt;3× or &lt;0.33× observed</p>
</div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Location Choice Summary</title>
<style>body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px; }}
h1 {{ color: #333; }} table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #eee; padding: 5px 8px; }}
tr:hover {{ background: #f9f9f9; }}</style>
</head><body>
<h1>Location Choice Model Summary</h1>
<p>ActivitySim-style destination and mode choice coefficients with statistical inference.</p>
{''.join(sections)}
{''.join(val_sections) if val_sections else ''}
</body></html>"""

    html_path = os.path.join(out_dir, 'location_choice_summary.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logger.info(f"  Location choice HTML report → {html_path}")


def _build_activitysim_location_observations(
    obs_tours: pd.DataFrame,
    persons: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Normalize observed tours into an ActivitySim-like location-choice structure.

    Output columns:
      chooser_id, person_id, purpose, segment, OTAZ, choice, trip_dist
    where ``choice`` is the chosen destination zone (DTAZ equivalent).
    """
    if obs_tours is None or len(obs_tours) == 0:
        return pd.DataFrame(
            columns=['chooser_id', 'person_id', 'purpose', 'segment', 'OTAZ', 'choice', 'trip_dist']
        )

    t = obs_tours.copy()
    if 'person_id' not in t.columns and 'PERSID' in t.columns:
        t['person_id'] = t['PERSID']
    if 'person_id' not in t.columns:
        t['person_id'] = np.arange(len(t)).astype(str)
    t['person_id'] = t['person_id'].astype(str)

    purpose_col = next(
        (
            c for c in [
                'primary_purpose', 'main_purpose', 'purpose', 'tour_primary_purpose',
                'OVERALL_PURPOSE', 'DESTPURP'
            ]
            if c in t.columns
        ),
        None,
    )
    if purpose_col is None:
        logger.warning("  Location validation: no purpose column in obs_tours")
        return pd.DataFrame(
            columns=['chooser_id', 'person_id', 'purpose', 'segment', 'OTAZ', 'choice', 'trip_dist']
        )

    otaz_col = next(
        (c for c in ['OTAZ', 'otaz', 'tour_origin', 'origin_zone', 'start_location'] if c in t.columns),
        None,
    )
    dtaz_col = next(
        (
            c for c in [
                'DTAZ', 'dtaz', 'dest_zone', 'destination', 'tour_destination',
                'destination_zone', 'end_location'
            ]
            if c in t.columns
        ),
        None,
    )

    if dtaz_col is None:
        logger.warning(
            "  Location validation: no destination zone column in obs_tours "
            f"(columns={list(t.columns)[:20]}...)"
        )
        return pd.DataFrame(
            columns=['chooser_id', 'person_id', 'purpose', 'segment', 'OTAZ', 'choice', 'trip_dist']
        )

    if otaz_col is None:
        t['OTAZ'] = 0
        otaz_col = 'OTAZ'

    dist_col = next(
        (c for c in ['DIST', 'trip_dist', 'tour_distance', 'max_distance', 'CUMDIST'] if c in t.columns),
        None,
    )

    work_lower = {str(x).lower() for x in _WORK_PURPOSES}
    edu_lower = {str(x).lower() for x in _EDU_PURPOSES}
    nonmand_lower = {str(x).lower() for x in _NONMAND_PURPOSES}

    def _segment_of(p):
        p_l = str(p).strip().lower()
        if p_l in work_lower:
            return 'work'
        if p_l in edu_lower:
            return 'edu'
        if p_l in nonmand_lower:
            return 'nonmand'
        return None

    purpose = t[purpose_col].fillna('').astype(str)
    out = pd.DataFrame({
        'person_id': t['person_id'].astype(str),
        'purpose': purpose,
        'segment': purpose.map(_segment_of),
        'OTAZ': pd.to_numeric(t[otaz_col], errors='coerce').fillna(0).astype(int),
        'choice': pd.to_numeric(t[dtaz_col], errors='coerce').fillna(0).astype(int),
    })

    if dist_col is not None:
        out['trip_dist'] = pd.to_numeric(t[dist_col], errors='coerce')
    else:
        out['trip_dist'] = np.nan

    # Preserve tour_mode (integer code) if available so that downstream
    # mode-share validation can read observed shares from obs_lc.
    _mode_col = next(
        (c for c in ['tour_mode', 'TOURMODE', 'mode_code', 'mode'] if c in t.columns),
        None,
    )
    if _mode_col is not None:
        _mode_vals = t[_mode_col]
        if _mode_vals.dtype == object:
            # Map string mode names → integer codes
            _mode_vals = _mode_vals.map(MODE_MAP)
        out['tour_mode'] = pd.to_numeric(_mode_vals, errors='coerce').fillna(0).astype(int)

    if 'ptour_id' in t.columns:
        out['chooser_id'] = t['ptour_id'].astype(str)
    elif 'tour_id' in t.columns:
        out['chooser_id'] = t['person_id'].astype(str) + '_' + t['tour_id'].astype(str)
    else:
        out['chooser_id'] = t['person_id'].astype(str) + '_' + t.index.astype(str)

    out = out[(out['segment'].notna()) & (out['choice'] > 0)].copy()

    # Merge STUDYING (1=primary, 2=secondary, 3=tertiary) from persons for
    # edu-segment person-type breakdown in validation prints.
    if persons is not None and len(persons) > 0:
        _pers = persons.copy()
        _pid_col = 'person_id' if 'person_id' in _pers.columns else (
            'PERSID' if 'PERSID' in _pers.columns else None)
        if _pid_col:
            _pers = _pers.rename(columns={_pid_col: 'person_id'})
            _pers['person_id'] = _pers['person_id'].astype(str)
            _study_col = next((c for c in ['STUDYING', 'study_level', 'study']
                               if c in _pers.columns), None)
            if _study_col:
                out = out.merge(
                    _pers[['person_id', _study_col]].rename(
                        columns={_study_col: 'study_level'}),
                    on='person_id', how='left',
                )

    _base_cols = ['chooser_id', 'person_id', 'purpose', 'segment', 'OTAZ', 'choice', 'trip_dist']
    _extra_cols = [c for c in ['tour_mode', 'study_level'] if c in out.columns]
    out = out[_base_cols + _extra_cols]

    logger.info(
        "  Location validation normalized observations: "
        f"{len(out)} rows, segment split={out['segment'].value_counts().to_dict()}"
    )
    return out


def _model_predicted_zone_shares(model, n_zones_hint: int = None) -> pd.Series:
    """
    Compute model-predicted destination zone shares across all choosers.

    Prefers the cached ``_pred_zone_shares`` attribute stored by ``_run_segment``
    after calling ``_dest_probs_numpy`` over all zones.  This is important because
    after importance sampling, ``model.probability()`` only returns probabilities
    for the *sampled* alternatives per chooser (not the full 330-zone set) and
    includes the Manski-Lerman ``log_correction`` offset which biases the shares.

    Falls back to ``model.probability()`` when no cache is present.
    """
    # ── Fast path: use pre-computed full-zone shares ──────────────────────
    _cached = getattr(model, '_pred_zone_shares', None)
    if _cached is not None and len(_cached) > 0:
        logger.debug(
            f"  _model_predicted_zone_shares: using cached _pred_zone_shares "
            f"({len(_cached)} zones, {int((_cached > 1e-6).sum())} non-zero)"
        )
        return _cached

    # ── Fallback: ask Larch for probabilities ─────────────────────────────
    # NOTE: if importance sampling was used, this only covers sampled alts.
    try:
        prob = model.probability(return_format="dataframe")
        # prob: index = caseid, columns = zone ids (altid values)
        # Aggregate: sum probabilities across choosers, then normalise
        zone_probs = prob.sum(axis=0)
        total = zone_probs.sum()
        if total > 0:
            zone_shares = zone_probs / total
        else:
            zone_shares = zone_probs
        zone_shares.index = pd.to_numeric(zone_shares.index, errors='coerce')
        zone_shares = zone_shares[zone_shares.index.notna()]
        zone_shares.index = zone_shares.index.astype(int)
        return zone_shares
    except Exception as _e:
        logger.debug(f"  _model_predicted_zone_shares: probability() failed ({_e})")
        return pd.Series(dtype=float)


def _model_predicted_mode_shares(model) -> pd.Series:
    """
    Compute model-predicted mode shares by aggregating probability() across choosers.

    Prefers the cached ``_pred_mode_shares`` attribute stored by ``_run_segment``
    (set once while the DataTree is still live; survives pkl round-trips).
    Falls back to ``model.probability()`` if the cache is absent.

    Returns a pd.Series indexed by mode code with share values (0-1).
    """
    # ── Fast path: use cached shares (set before pickling) ───────────────
    _cached = getattr(model, '_pred_mode_shares', None)
    if _cached is not None and len(_cached) > 0:
        logger.debug(
            f"  _model_predicted_mode_shares: using cached _pred_mode_shares "
            f"({len(_cached)} modes)"
        )
        return _cached

    # ── Slow path: compute from live model (only works with DataTree attached) ─
    try:
        prob = model.probability(return_format="dataframe")
        # Nested logit: model.probability() returns ALL graph nodes including
        # intermediate nest nodes (Car=6, NonMotor=7, Motor=8, Root=9 etc.).
        # Summing nest columns would double-count P(DA)+P(SR) as P(DA)+P(SR)+P(Car).
        # Filter to LEAF-only columns: mode codes 1=DA, 2=SR, 3=Transit, 4=Walk, 5=Bike.
        _leaf_codes = {1, 2, 3, 4, 5}
        _leaf_cols = [c for c in prob.columns
                      if pd.to_numeric(c, errors='coerce') in _leaf_codes]
        if _leaf_cols:
            prob = prob[_leaf_cols]
        mode_probs = prob.sum(axis=0)
        total = mode_probs.sum()
        if total > 0:
            result = mode_probs / total
            # Store as cache for future calls (e.g. after pkl reload)
            try:
                model._pred_mode_shares = result
            except Exception:
                pass
            return result
        return mode_probs
    except Exception as _e:
        logger.debug(f"  _model_predicted_mode_shares: probability() failed ({_e})")
        return pd.Series(dtype=float)


def _location_choice_validation(
    obs_obs: pd.DataFrame,
    models: dict,
    landuse: pd.DataFrame,
    out_dir: str,
):
    """
    Validate destination choice models against observed HHTS tour destinations.

    Uses model.probability() to compute actual model-predicted zone shares
    and compares them to observed HHTS destination distributions.

    Produces:
      - Zone-level predicted vs observed destination shares (per segment)
      - Mode share predicted vs observed (per segment)
      - Work and Education zone distributions with landuse context
      - Segment-level summary statistics (Pearson r, RMSE, mean trip dist)
    """
    from scipy.stats import pearsonr

    SEG_PURPOSES = {
        'work':    _WORK_PURPOSES,
        'edu':     _EDU_PURPOSES,
        'nonmand': _NONMAND_PURPOSES,
    }

    if obs_obs is None or len(obs_obs) == 0:
        logger.warning("  Location validation: no normalized observations available")
        return None, None

    val_rows = []
    summary_rows = []
    mode_rows = []
    work_rows = []
    edu_rows = []

    # Zone list from landuse
    zone_col = next((c for c in ['ORDER', 'zone_id', 'TAZ'] if c in landuse.columns), None)
    all_zones = sorted(landuse[zone_col].unique()) if zone_col else []
    lu = landuse.copy()
    lu_indexed = lu.set_index(zone_col) if zone_col else lu

    # Real-zone range, derived from the landuse zone list rather than the
    # legacy hardcoded "1-329" literal — so this works for both the legacy
    # 1-329 ORDER scheme and the new_pop 507-zone scheme (ORDER = 9-digit
    # SA2 code). 330 stays the external/catch-all sentinel in both schemes
    # (see Synth_Connect.process_survey's EXTERNAL_TAZ).
    _real_zone_lo = min(all_zones) if all_zones else 1
    _real_zone_hi = max(all_zones) if all_zones else 329

    for seg, purposes in SEG_PURPOSES.items():
        seg_obs = obs_obs[obs_obs['segment'] == seg].copy()

        # ── Edu: filter to actual school tours only ───────────────────────
        # Exclude childcare/preschool purposes (different destination universe)
        # and restrict to real zones (zone 330 = external, no school data).
        if seg == 'edu':
            if 'purpose' in seg_obs.columns:
                _childcare_lower = {'childcare or kindergarten'}
                _cc_mask = seg_obs['purpose'].astype(str).str.lower().isin(_childcare_lower)
                if _cc_mask.any():
                    logger.info(
                        f"  Location validation [edu]: excluding {int(_cc_mask.sum())} "
                        "childcare/preschool tours (not part of school destination model)"
                    )
                    seg_obs = seg_obs[~_cc_mask].copy()
            if 'choice' in seg_obs.columns:
                _valid_zone = pd.to_numeric(seg_obs['choice'], errors='coerce').between(_real_zone_lo, _real_zone_hi)
                _n_outside = int((~_valid_zone).sum())
                if _n_outside > 0:
                    logger.info(
                        f"  Location validation [edu]: excluding {_n_outside} tours "
                        "with destination outside real zones (zone 330 = external zone)"
                    )
                seg_obs = seg_obs[_valid_zone].copy()

        # ── Work: restrict to real zones only ──────────────────────────────
        elif seg == 'work' and 'choice' in seg_obs.columns:
            _valid_zone = pd.to_numeric(seg_obs['choice'], errors='coerce').between(_real_zone_lo, _real_zone_hi)
            _n_outside = int((~_valid_zone).sum())
            if _n_outside > 0:
                logger.info(
                    f"  Location validation [work]: excluding {_n_outside} tours "
                    "with destination outside real zones"
                )
            seg_obs = seg_obs[_valid_zone].copy()

        obs_dest = pd.to_numeric(seg_obs['choice'], errors='coerce').dropna()

        if len(obs_dest) == 0:
            continue

        # Observed zone shares
        obs_counts = obs_dest.astype(int).value_counts().reindex(all_zones, fill_value=0)
        obs_share = obs_counts / max(obs_counts.sum(), 1)

        # ── Model-predicted zone shares via probability() ───────────────
        res = models.get(seg, {})
        dest_model = res.get('dest_model')
        pred_share = pd.Series(0.0, index=all_zones)

        if dest_model is not None:
            # Restore cached zone shares from models dict if they were lost from
            # the model object (e.g. after cloudpickle round-trip strips attributes)
            if getattr(dest_model, '_pred_zone_shares', None) is None:
                _cached = res.get('pred_zone_shares')
                if _cached is not None and len(_cached) > 0:
                    dest_model._pred_zone_shares = _cached
                    logger.debug(
                        f"  Location validation [{seg}]: restored _pred_zone_shares "
                        f"from models dict ({len(_cached)} zones)"
                    )
            _model_shares = _model_predicted_zone_shares(dest_model)
            if len(_model_shares) > 0:
                pred_share = _model_shares.reindex(all_zones, fill_value=0.0)
                logger.info(
                    f"  Location validation [{seg}]: model.probability() succeeded - "
                    f"{int((_model_shares > 1e-6).sum())} zones with non-zero share"
                )
            else:
                # model.probability() returned an empty Series - the model
                # is in an unusable state (estimation failed, or Larch mangled
                # the pf after pickle).  This is a DATA QUALITY problem, not a
                # normal operating condition.  Log it as an error so it's visible,
                # then use the size-term as an unconditional proximity proxy so
                # the validation table still outputs something meaningful.
                logger.error(
                    f"  Location validation [{seg}]: model.probability() returned empty - "
                    "predicted zone shares are UNRELIABLE (size-term proxy used).\n"
                    f"  ► Root cause candidates:\n"
                    f"    1. dest_model estimation failed - check LL and pf above.\n"
                    f"    2. Larch mangled internal state after pickle/reload.\n"
                    f"    3. dest_model.data has zero rows (estimation dataset empty).\n"
                    f"  ► Predicted shares in validation CSV will reflect SIZE TERM only,\n"
                    f"    not the full model.  Do not use them to judge model fit."
                )
                try:
                    _st_df = compute_size_terms(landuse, purposes=[seg])
                    if seg in _st_df.columns:
                        _raw = np.exp(_st_df[seg])
                        _raw = _raw.reindex(all_zones, fill_value=1e-6).clip(lower=1e-6)
                        pred_share = _raw / _raw.sum()
                    else:
                        raise ValueError(f"compute_size_terms missing column '{seg}'")
                except Exception as _st_e:
                    _emp_col = {'work': 'TOTEMP', 'edu': 'Education', 'nonmand': 'RETEMPN'}.get(seg, 'TOTEMP')
                    _attraction = pd.Series(1.0, index=all_zones)
                    for z in all_zones:
                        if z in lu_indexed.index:
                            _val = float(lu_indexed.loc[z, _emp_col]) if _emp_col in lu_indexed.columns else 1.0
                            _attraction[z] = max(_val ** 0.7, 0.01)
                    pred_share = _attraction / _attraction.sum()
                    logger.error(
                        f"  Location validation [{seg}]: compute_size_terms also failed ({_st_e}); "
                        f"using {_emp_col}^0.7 as last-resort proxy."
                    )
        else:
            logger.error(
                f"  Location validation [{seg}]: dest_model is MISSING from models dict.\n"
                f"  ► Segment '{seg}' was not estimated - check _run_segment logs above."
            )

        # Zone-level comparison
        for z in all_zones:
            val_rows.append({
                'segment': seg,
                'zone': z,
                'obs_tours': int(obs_counts.get(z, 0)),
                'obs_share': float(obs_share.get(z, 0.0)),
                'pred_share': float(pred_share.get(z, 0.0)),
            })

        # Work and edu specific landuse context rows
        if seg == 'work':
            for z in all_zones:
                _row = {'zone': z, 'obs_workers': int(obs_counts.get(z, 0)),
                        'pred_workers_scaled': float(pred_share.get(z, 0.0) * obs_counts.sum())}
                for _c in ['TOTEMP', 'FPSEMPN', 'HEREMPN', 'RETEMPN', 'commercial']:
                    if _c in lu_indexed.columns:
                        _row[_c] = float(lu_indexed.loc[z, _c]) if z in lu_indexed.index else 0.0
                work_rows.append(_row)
        elif seg == 'edu':
            # Per-zone primary/secondary student counts from obs data
            _prim_counts = pd.Series(0, index=all_zones, dtype=int)
            _sec_counts  = pd.Series(0, index=all_zones, dtype=int)
            if 'study_level' in seg_obs.columns:
                _prim_dest = pd.to_numeric(
                    seg_obs.loc[seg_obs['study_level'].isin([1, '1']), 'choice'],
                    errors='coerce').dropna().astype(int)
                _sec_dest  = pd.to_numeric(
                    seg_obs.loc[seg_obs['study_level'].isin([2, '2']), 'choice'],
                    errors='coerce').dropna().astype(int)
                _prim_counts = _prim_dest.value_counts().reindex(all_zones, fill_value=0)
                _sec_counts  = _sec_dest.value_counts().reindex(all_zones, fill_value=0)

            for z in all_zones:
                _row = {'zone': z, 'obs_students': int(obs_counts.get(z, 0)),
                        'pred_students_scaled': float(pred_share.get(z, 0.0) * obs_counts.sum()),
                        'obs_primary': int(_prim_counts.get(z, 0)),
                        'obs_secondary': int(_sec_counts.get(z, 0))}
                for _c in ['Primary', 'Secondary', 'Education', 'HSENROLL']:
                    if _c in lu_indexed.columns:
                        _row[_c] = float(lu_indexed.loc[z, _c]) if z in lu_indexed.index else 0.0
                edu_rows.append(_row)

        # Summary statistics
        try:
            _ps = pred_share.reindex(obs_share.index, fill_value=0.0)
            if obs_share.std() > 0 and _ps.std() > 0:
                corr, _ = pearsonr(obs_share.values, _ps.values)
            else:
                corr = np.nan
            rmse = float(np.sqrt(np.mean((obs_share.values - _ps.values) ** 2)))
        except Exception:
            corr, rmse = np.nan, np.nan

        # Distance distribution
        if 'trip_dist' in seg_obs.columns:
            obs_dist = pd.to_numeric(seg_obs['trip_dist'], errors='coerce').dropna()
            dist_mean = float(obs_dist.mean()) if len(obs_dist) > 0 else np.nan
            dist_med = float(obs_dist.median()) if len(obs_dist) > 0 else np.nan
        else:
            dist_mean, dist_med = np.nan, np.nan

        ll = res.get('dest_ll', np.nan)
        n_coef = len(res.get('dest_coef', {}))
        summary_rows.append({
            'segment': seg,
            'n_obs_tours': int(len(seg_obs)),
            'n_dest_zones_used': int((obs_counts > 0).sum()),
            'zone_share_corr': corr,
            'zone_share_rmse': rmse,
            'mean_trip_dist': dist_mean,
            'median_trip_dist': dist_med,
            'n_dest_coefficients': n_coef,
            'dest_loglike': float(ll) if ll is not None and not (isinstance(ll, float) and np.isnan(ll)) else np.nan,
        })

        logger.info(
            f"  Location validation [{seg}]: {len(seg_obs)} obs tours, "
            f"zone corr={corr:.3f}, RMSE={rmse:.4f}"
            + (f", mean dist={dist_mean:.1f}km" if not np.isnan(dist_mean) else "")
        )

        # ── Mode share validation ─────────────────────────────────────────
        mode_model = res.get('mode_model')
        obs_mode_col = next(
            (c for c in ['tour_mode', 'TOURMODE', 'mode_code', 'mode']
             if c in seg_obs.columns),
            None,
        )

        # Observed mode shares - tour_mode is integer-coded (MODE_MAP values)
        if obs_mode_col is not None:
            _raw = pd.to_numeric(seg_obs[obs_mode_col], errors='coerce')
            # If column is still strings (e.g. 'DA', 'SR') map to int codes
            if _raw.isna().mean() > 0.9 and seg_obs[obs_mode_col].dtype == object:
                _raw = seg_obs[obs_mode_col].map(MODE_MAP)
            obs_modes = _raw.dropna().astype(int)
            obs_mode_share = (
                obs_modes[obs_modes > 0].value_counts(normalize=True).rename('obs_share')
            )
        else:
            obs_mode_share = pd.Series(dtype=float)

        # Predicted mode shares
        if mode_model is not None:
            # Restore cached shares if model was loaded from pkl without DataTree
            if getattr(mode_model, '_pred_mode_shares', None) is None:
                _cached_ms = res.get('pred_mode_shares')
                if _cached_ms is not None and len(_cached_ms) > 0:
                    mode_model._pred_mode_shares = _cached_ms
            pred_mode_share = _model_predicted_mode_shares(mode_model)
        else:
            pred_mode_share = pd.Series(dtype=float)

        all_modes = sorted(set(obs_mode_share.index.tolist() if len(obs_mode_share) > 0 else []) |
                           set(pred_mode_share.index.tolist() if len(pred_mode_share) > 0 else []))
        for m_code in all_modes:
            mode_rows.append({
                'segment': seg,
                'mode_code': int(m_code),
                'mode_name': MODE_NAMES.get(int(m_code), str(m_code)),
                'obs_share_pct': float(obs_mode_share.get(m_code, 0.0)) * 100,
                'pred_share_pct': float(pred_mode_share.get(m_code, 0.0)) * 100,
                'diff_pp': (float(pred_mode_share.get(m_code, 0.0)) -
                            float(obs_mode_share.get(m_code, 0.0))) * 100,
            })

    # Save validation CSVs
    _val_df_out = None
    _sum_df_out = None
    if val_rows:
        _val_df_out = pd.DataFrame(val_rows)
        val_path = os.path.join(out_dir, 'location_choice_validation.csv')
        _val_df_out.to_csv(val_path, index=False)
        logger.info(f"  Location validation detail → {val_path}")

    if summary_rows:
        _sum_df_out = pd.DataFrame(summary_rows)
        sum_path = os.path.join(out_dir, 'location_choice_validation_summary.csv')
        _sum_df_out.to_csv(sum_path, index=False)
        logger.info(f"  Location validation summary → {sum_path}")

    if mode_rows:
        _mode_df = pd.DataFrame(mode_rows)
        mode_path = os.path.join(out_dir, 'mode_share_validation.csv')
        _mode_df.to_csv(mode_path, index=False)
        logger.info(f"  Mode share validation → {mode_path}")
        # print to console for quick review
        print("\n  ── MODE SHARE: Predicted vs Observed (estimation data) ──────")
        for seg in _mode_df['segment'].unique():
            _s = _mode_df[_mode_df['segment'] == seg]
            print(f"\n  [{seg.upper()}]  {'Mode':<10s}  {'Obs%':>7s}  {'Pred%':>7s}  {'Diff':>6s}")
            for _, r in _s.sort_values('mode_code').iterrows():
                flag = ' ◄' if abs(r['diff_pp']) > 5 else ''
                print(f"  [{seg.upper()}]  {r['mode_name']:<10s}  "
                      f"{r['obs_share_pct']:7.1f}  {r['pred_share_pct']:7.1f}  "
                      f"{r['diff_pp']:+6.1f}{flag}")

    if work_rows:
        _work_df = pd.DataFrame(work_rows).sort_values('zone')
        work_path = os.path.join(out_dir, 'work_destination_validation.csv')
        _work_df.to_csv(work_path, index=False)
        logger.info(f"  Work destination validation → {work_path}")
        print("\n  ── WORK DESTINATION: Top 30 observed zones ────────────────────────────────")
        print(f"  {'Zone':>6s}  {'Obs':>5s}  {'Pred':>7s}  {'Ratio':>6s}  "
              f"{'TOTEMP':>7s}  {'FPSEMPN':>7s}  {'HEREMPN':>7s}  {'Comm(k)':>8s}")
        _work_top30 = _work_df.nlargest(30, 'obs_workers').sort_values('zone')
        for _, r in _work_top30.iterrows():
            _te  = int(r.get('TOTEMP', 0))
            _fp  = int(r.get('FPSEMPN', 0))
            _he  = int(r.get('HEREMPN', 0))
            _com = r.get('commercial', 0.0) / 1000.0
            _obs = int(r['obs_workers'])
            _ps  = r['pred_workers_scaled']
            _ratio = _obs / max(_ps, 0.1)
            _flag = ' ◄◄' if _ratio > 4 else (' ◄' if _ratio > 2 else '')
            print(f"  {int(r['zone']):6d}  {_obs:5d}  {_ps:7.1f}  {_ratio:6.2f}  "
                  f"{_te:7d}  {_fp:7d}  {_he:7d}  {_com:8.0f}{_flag}")

    if edu_rows:
        _edu_df = pd.DataFrame(edu_rows).sort_values('zone')
        edu_path = os.path.join(out_dir, 'edu_destination_validation.csv')
        _edu_df.to_csv(edu_path, index=False)
        logger.info(f"  Education destination validation → {edu_path}")
        _has_study = 'obs_primary' in _edu_df.columns
        print("\n  ── EDU DESTINATION: Top 25 observed zones ─────────────────────────────────")
        if _has_study:
            print(f"  {'Zone':>6s}  {'ObsTot':>6s}  {'ObsPrim':>7s}  {'ObsSec':>6s}  "
                  f"{'Pred':>7s}  {'Ratio':>6s}  {'PriSch':>7s}  {'SecSch':>7s}")
        else:
            print(f"  {'Zone':>6s}  {'Obs':>5s}  {'Pred':>7s}  {'Ratio':>6s}  "
                  f"{'Primary':>8s}  {'Secondary':>9s}")
        _edu_top25 = _edu_df.nlargest(25, 'obs_students').sort_values('zone')
        for _, r in _edu_top25.iterrows():
            _pri_sch = int(r.get('Primary', 0))
            _sec_sch = int(r.get('Secondary', 0))
            _obs = int(r['obs_students'])
            _ps  = r['pred_students_scaled']
            _ratio = _obs / max(_ps, 0.1)
            _flag = ' ◄◄' if _ratio > 4 else (' ◄' if _ratio > 2 else '')
            if _has_study:
                _op = int(r.get('obs_primary', 0))
                _os = int(r.get('obs_secondary', 0))
                print(f"  {int(r['zone']):6d}  {_obs:6d}  {_op:7d}  {_os:6d}  "
                      f"{_ps:7.1f}  {_ratio:6.2f}  {_pri_sch:7d}  {_sec_sch:7d}{_flag}")
            else:
                print(f"  {int(r['zone']):6d}  {_obs:5d}  {_ps:7.1f}  {_ratio:6.2f}  "
                      f"{_pri_sch:8d}  {_sec_sch:9d}{_flag}")

    # ── Build combined validation HTML report ────────────────────────────
    _mode_df_for_html = pd.DataFrame(mode_rows) if mode_rows else None
    _work_df_for_html = pd.DataFrame(work_rows).sort_values('zone') \
        if work_rows else None
    _edu_df_for_html = pd.DataFrame(edu_rows).sort_values('zone') \
        if edu_rows else None
    _write_validation_html(
        val_df=_val_df_out,
        summary_df=_sum_df_out,
        mode_df=_mode_df_for_html,
        work_df=_work_df_for_html,
        edu_df=_edu_df_for_html,
        out_dir=out_dir,
    )

    return _val_df_out, _sum_df_out


def _write_validation_html(
    val_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    mode_df: pd.DataFrame,
    work_df: pd.DataFrame,
    edu_df: pd.DataFrame,
    out_dir: str,
):
    """Write a comprehensive validation HTML report with all levels."""
    sections = []

    # ── Summary table ─────────────────────────────────────────────────────
    if summary_df is not None and len(summary_df) > 0:
        _sum_rows = []
        for _, r in summary_df.iterrows():
            _corr = f"{r['zone_share_corr']:.3f}" if np.isfinite(r['zone_share_corr']) else 'n/a'
            _rmse = f"{r['zone_share_rmse']:.4f}" if np.isfinite(r['zone_share_rmse']) else 'n/a'
            _dist = f"{r['mean_trip_dist']:.1f}" if np.isfinite(r['mean_trip_dist']) else 'n/a'
            _ll   = f"{r['dest_loglike']:.1f}"   if np.isfinite(r['dest_loglike'])   else 'n/a'
            _sum_rows.append(
                f"<tr><td>{r['segment']}</td>"
                f"<td>{int(r['n_obs_tours'])}</td>"
                f"<td>{int(r['n_dest_zones_used'])}</td>"
                f"<td style='text-align:right'>{_corr}</td>"
                f"<td style='text-align:right'>{_rmse}</td>"
                f"<td style='text-align:right'>{_dist} km</td>"
                f"<td style='text-align:right'>{_ll}</td></tr>"
            )
        sum_rows_html = ''.join(_sum_rows)
        sections.append(f"""
<div class="val-section">
  <h2>Destination Model Fit Summary (HHTS Estimation Data)</h2>
  <table>
    <thead><tr>
      <th>Segment</th><th>N Obs</th><th>Zones Used</th>
      <th>Zone Share Pearson r</th><th>Zone Share RMSE</th>
      <th>Mean Trip Dist</th><th>Log-Likelihood</th>
    </tr></thead>
    <tbody>{sum_rows_html}</tbody>
  </table>
</div>""")

    # ── Mode share table ──────────────────────────────────────────────────
    if mode_df is not None and len(mode_df) > 0:
        mode_html_parts = []
        for seg in mode_df['segment'].unique():
            _s = mode_df[mode_df['segment'] == seg].sort_values('mode_code')
            _rows = ''.join(
                f"<tr>"
                f"<td>{r['mode_name']}</td>"
                f"<td style='text-align:right'>{r['obs_share_pct']:.1f}%</td>"
                f"<td style='text-align:right'>{r['pred_share_pct']:.1f}%</td>"
                f"<td style='text-align:right; color:{'#b71c1c' if abs(r['diff_pp']) > 5 else '#333'}'>"
                f"{r['diff_pp']:+.1f}pp</td></tr>"
                for _, r in _s.iterrows()
            )
            mode_html_parts.append(f"""
<h3>{seg.upper()} Segment</h3>
<table>
  <thead><tr><th>Mode</th><th>Observed %</th><th>Predicted %</th><th>Diff (pp)</th></tr></thead>
  <tbody>{_rows}</tbody>
</table>""")
        sections.append(f"""
<div class="val-section">
  <h2>Mode Share: Predicted vs Observed (Estimation Data)</h2>
  <p style="font-size:0.9em;color:#666">Predicted = model probability aggregated over HHTS choosers.
  Values &gt;5pp difference highlighted red.</p>
  {''.join(mode_html_parts)}
</div>""")

    # ── Top-zone destination tables ───────────────────────────────────────
    for seg, purposes in [('work', _WORK_PURPOSES), ('edu', _EDU_PURPOSES), ('nonmand', _NONMAND_PURPOSES)]:
        if val_df is None:
            continue
        _sv = val_df[val_df['segment'] == seg].copy()
        if len(_sv) == 0:
            continue
        # Take top 25 by observed tours, then sort by zone number for display
        _sv = _sv.nlargest(25, 'obs_tours').sort_values('zone')
        _rows = ''.join(
            f"<tr>"
            f"<td>{int(r['zone'])}</td>"
            f"<td style='text-align:right'>{int(r['obs_tours'])}</td>"
            f"<td style='text-align:right'>{r['obs_share']*100:.2f}%</td>"
            f"<td style='text-align:right'>{r['pred_share']*100:.2f}%</td>"
            f"<td style='text-align:right; color:{'#b71c1c' if r['obs_share'] > 0 and (r['pred_share'] / max(r['obs_share'], 1e-9) > 3 or r['pred_share'] / max(r['obs_share'], 1e-9) < 0.33) else '#333'}'>"
            f"{'⚠' if r['obs_share'] > 0 and (r['pred_share'] / max(r['obs_share'], 1e-9) > 3 or r['pred_share'] / max(r['obs_share'], 1e-9) < 0.33) else ''}"
            f"</td></tr>"
            for _, r in _sv.iterrows()
        )
        _sum_row = ''
        if summary_df is not None:
            _sr = summary_df[summary_df['segment'] == seg]
            if len(_sr):
                _r = _sr.iloc[0]
                _corr = f"{_r['zone_share_corr']:.3f}" if np.isfinite(_r['zone_share_corr']) else 'n/a'
                _rmse = f"{_r['zone_share_rmse']:.4f}" if np.isfinite(_r['zone_share_rmse']) else 'n/a'
                _dist = f"{_r['mean_trip_dist']:.1f} km" if np.isfinite(_r['mean_trip_dist']) else 'n/a'
                _sum_row = f'<p>Pearson r = {_corr} | RMSE = {_rmse} | Mean trip dist = {_dist}</p>'
        sections.append(f"""
<div class="val-section">
  <h2>{seg.upper()} - Destination Zone Distribution (Top 25 Observed)</h2>
  {_sum_row}
  <table>
    <thead><tr><th>Zone</th><th>Obs Trips</th><th>Obs Share</th><th>Pred Share</th><th>Flag</th></tr></thead>
    <tbody>{_rows}</tbody>
  </table>
  <p style="font-size:11px;color:#666">⚠ = predicted share &gt;3× or &lt;0.33× observed</p>
</div>""")

    # ── Work destination with landuse ─────────────────────────────────────
    if work_df is not None and len(work_df) > 0:
        _wr = work_df.nlargest(20, 'obs_workers').sort_values('zone')
        _rows = ''.join(
            f"<tr><td>{int(r['zone'])}</td>"
            f"<td style='text-align:right'>{int(r['obs_workers'])}</td>"
            f"<td style='text-align:right'>{r['pred_workers_scaled']:.1f}</td>"
            f"<td style='text-align:right'>{int(r.get('TOTEMP', 0))}</td>"
            f"<td style='text-align:right'>{int(r.get('FPSEMPN', 0))}</td>"
            f"<td style='text-align:right'>{int(r.get('HEREMPN', 0))}</td>"
            f"<td style='text-align:right'>{int(r.get('RETEMPN', 0))}</td></tr>"
            for _, r in _wr.iterrows()
        )
        sections.append(f"""
<div class="val-section" style="background:#f8fff8;">
  <h2>Work Destination Validation - Top 20 Zones with Landuse</h2>
  <p style="font-size:0.9em;color:#666">Pred = model probability × N_obs_work (scaled counts).
  Compare against TOTEMP, FPSEMPN (finance/prof/service employment).</p>
  <table>
    <thead><tr><th>Zone</th><th>Obs Workers</th><th>Pred (scaled)</th>
    <th>TOTEMP</th><th>FPSEMPN</th><th>HEREMPN</th><th>RETEMPN</th></tr></thead>
    <tbody>{_rows}</tbody>
  </table>
</div>""")

    # ── Education destination with landuse ────────────────────────────────
    if edu_df is not None and len(edu_df) > 0:
        _er = edu_df.head(20)
        _rows = ''.join(
            f"<tr><td>{int(r['zone'])}</td>"
            f"<td style='text-align:right'>{int(r['obs_students'])}</td>"
            f"<td style='text-align:right'>{r['pred_students_scaled']:.1f}</td>"
            f"<td style='text-align:right'>{int(r.get('Primary', 0))}</td>"
            f"<td style='text-align:right'>{int(r.get('Secondary', 0))}</td>"
            f"<td style='text-align:right'>{int(r.get('Education', 0))}</td>"
            f"<td style='text-align:right'>{int(r.get('HSENROLL', 0))}</td></tr>"
            for _, r in _er.iterrows()
        )
        sections.append(f"""
<div class="val-section" style="background:#fff8f8;">
  <h2>Education Destination Validation - Top 20 Zones with Landuse</h2>
  <p style="font-size:0.9em;color:#666">Pred = model probability × N_obs_edu (scaled counts).
  Compare against Primary/Secondary/Education (OSM school counts) and HSENROLL.</p>
  <table>
    <thead><tr><th>Zone</th><th>Obs Students</th><th>Pred (scaled)</th>
    <th>Primary</th><th>Secondary</th><th>Education</th><th>HSENROLL</th></tr></thead>
    <tbody>{_rows}</tbody>
  </table>
</div>""")

    body = '\n'.join(sections) if sections else '<p>No validation data available.</p>'
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Destination + Mode Choice Validation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; max-width: 1200px; }}
    h1 {{ border-bottom: 2px solid #1565c0; }}
    h2 {{ color: #1565c0; margin-top: 28px; }}
    h3 {{ color: #444; }}
    .val-section {{ margin-bottom: 30px; border: 1px solid #ccc;
                    border-radius: 6px; padding: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #1565c0; color: white; padding: 6px 10px; text-align: left; }}
    td {{ padding: 5px 10px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Stage 5 - Destination + Mode Choice Validation Report</h1>
  <p>Generated from HHTS estimation data. Predicted shares from model.probability()
  aggregated over all choosers.</p>
{body}
</body>
</html>"""
    val_html_path = os.path.join(out_dir, 'destination_mode_validation.html')
    try:
        with open(val_html_path, 'w', encoding='utf-8') as _f:
            _f.write(html)
        logger.info(f"  Validation HTML report → {val_html_path}")
        print(f"  Validation HTML → {val_html_path}")
    except Exception as _hw:
        logger.warning(f"  Could not write validation HTML: {_hw}")


def write_destination_validation_html(
    tours_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    out_dir: str,
    landuse: pd.DataFrame = None,
    top_n: int = 25,
):
    """
    Write three separate HTML validation reports for destination model output:
      - tour_destination_validation.html  - tour-level DTAZ distribution
      - trip_destination_validation.html  - trip-level DTAZ distribution
      - location_assignment_validation.html - combined assignment stats

    Parameters
    ----------
    tours_df : predicted tour table with DTAZ, purpose/segment columns
    trips_df : predicted trip table with DTAZ, purpose columns
    out_dir  : directory to write HTML files into
    landuse  : optional landuse table for zone employment/enrollment lookup
    top_n    : number of top zones to show in each table
    """
    os.makedirs(out_dir, exist_ok=True)

    def _zone_table_html(df, dest_col, purpose_col, title, top_n=25):
        if df is None or len(df) == 0 or dest_col not in df.columns:
            return f'<p>No data available for {title}.</p>'
        sections = []
        purposes = sorted(df[purpose_col].dropna().unique()) if purpose_col in df.columns else ['all']
        for purp in purposes:
            sub = df[df[purpose_col] == purp] if purpose_col in df.columns else df
            _raw_zones = pd.to_numeric(sub[dest_col], errors='coerce').dropna().astype(int)
            counts = _raw_zones[_raw_zones > 0].value_counts()   # exclude zone 0
            if len(counts) == 0:
                continue
            total = counts.sum()
            # Show top_n most-used zones, but display them in ascending zone-number order
            top = counts.head(top_n).sort_index()
            cum_share = counts.cumsum() / total * 100
            rows = []
            for zone, cnt in top.items():
                share_pct = cnt / total * 100
                bar_w = max(2, int(share_pct * 5))
                bar = f'<span style="display:inline-block;width:{bar_w}px;height:10px;background:#4a90d9;"></span>'
                rows.append(
                    f'<tr><td>{zone}</td>'
                    f'<td style="text-align:right">{cnt:,}</td>'
                    f'<td style="text-align:right">{share_pct:.2f}%</td>'
                    f'<td style="text-align:right">{cum_share.get(zone, 0):.1f}%</td>'
                    f'<td>{bar}</td></tr>'
                )
            n_zones_used = int((counts > 0).sum())
            hhi = float(((counts / total) ** 2).sum())
            sections.append(f"""
<div style="margin:16px 0; padding:12px; border:1px solid #ccc; border-radius:6px;">
  <h3>{purp.upper()} - top {top_n} destination zones</h3>
  <p>{total:,} trips/tours | {n_zones_used} zones used | HHI={hhi:.4f}
     <span style="font-size:11px; color:#666;">(HHI→0 = dispersed, HHI→1 = concentrated)</span></p>
  <table style="border-collapse:collapse; font-size:12px; width:70%;">
    <thead><tr style="background:#f0f0f0;">
      <th>Zone</th><th style="text-align:right">Count</th>
      <th style="text-align:right">Share</th><th style="text-align:right">Cumul%</th><th>Bar</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>""")
        return '\n'.join(sections) or f'<p>No valid destination data for {title}.</p>'

    css = ('body{font-family:-apple-system,Arial,sans-serif;max-width:1100px;'
           'margin:auto;padding:20px} h1,h2{color:#333} '
           'table{border-collapse:collapse;width:100%} '
           'td,th{border-bottom:1px solid #eee;padding:4px 8px} '
           'tr:hover{background:#f9f9f9}')

    # ── Tour-level HTML ───────────────────────────────────────────────────────
    purp_col = next((c for c in ['segment', 'purpose', 'tourpurp'] if tours_df is not None and c in (tours_df.columns if tours_df is not None else [])), None)
    tour_body = _zone_table_html(tours_df, 'DTAZ', purp_col or '', 'tour destinations', top_n)
    tour_html = (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
                 f'<title>Tour Destination Validation</title>'
                 f'<style>{css}</style></head><body>'
                 f'<h1>Tour-Level Destination Validation</h1>'
                 f'<p>Predicted DTAZ distributions by tour segment.</p>'
                 f'{tour_body}</body></html>')
    tour_path = os.path.join(out_dir, 'tour_destination_validation.html')
    with open(tour_path, 'w', encoding='utf-8') as f:
        f.write(tour_html)
    logger.info(f"  Tour destination validation → {tour_path}")

    # ── Trip-level HTML ───────────────────────────────────────────────────────
    trip_purp_col = next((c for c in ['purpose', 'trip_purpose', 'segment'] if trips_df is not None and c in (trips_df.columns if trips_df is not None else [])), None)
    trip_body = _zone_table_html(trips_df, 'DTAZ', trip_purp_col or '', 'trip destinations', top_n)
    trip_html = (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
                 f'<title>Trip Destination Validation</title>'
                 f'<style>{css}</style></head><body>'
                 f'<h1>Trip-Level Destination Validation</h1>'
                 f'<p>Predicted DTAZ distributions across all trips.</p>'
                 f'{trip_body}</body></html>')
    trip_path = os.path.join(out_dir, 'trip_destination_validation.html')
    with open(trip_path, 'w', encoding='utf-8') as f:
        f.write(trip_html)
    logger.info(f"  Trip destination validation → {trip_path}")

    # ── Location assignment summary HTML ────────────────────────────────────
    assign_sections = []
    for label, df in [('Tours', tours_df), ('Trips', trips_df)]:
        if df is None or len(df) == 0 or 'DTAZ' not in df.columns:
            assign_sections.append(f'<p>{label}: no data.</p>')
            continue
        dtaz = pd.to_numeric(df['DTAZ'], errors='coerce').dropna().astype(int)
        n_assigned = int((dtaz > 0).sum())
        n_unassigned = int((dtaz <= 0).sum())
        n_zones = int(dtaz[dtaz > 0].nunique())
        top1_zone = int(dtaz.value_counts().index[0]) if len(dtaz) > 0 else 'n/a'
        top1_share = float(dtaz.value_counts().iloc[0] / len(dtaz) * 100) if len(dtaz) > 0 else 0
        assign_sections.append(f"""
<div style="margin:12px 0; padding:12px; border:1px solid #bbb; border-radius:6px;">
  <h3>{label}</h3>
  <table style="width:50%; font-size:13px; border-collapse:collapse;">
    <tr><td>Total records</td><td style="text-align:right"><b>{len(df):,}</b></td></tr>
    <tr><td>Assigned (DTAZ &gt; 0)</td><td style="text-align:right">{n_assigned:,}</td></tr>
    <tr><td>Unassigned / home zone</td><td style="text-align:right">{n_unassigned:,}</td></tr>
    <tr><td>Unique destination zones</td><td style="text-align:right">{n_zones:,}</td></tr>
    <tr><td>Top zone (# {top1_zone})</td>
        <td style="text-align:right">{top1_share:.1f}% of trips</td></tr>
  </table>
</div>""")
    assign_html = (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
                   f'<title>Location Assignment Validation</title>'
                   f'<style>{css}</style></head><body>'
                   f'<h1>Location Assignment Validation</h1>'
                   f'<p>Summary of DTAZ assignment completeness.</p>'
                   + ''.join(assign_sections) + '</body></html>')
    assign_path = os.path.join(out_dir, 'location_assignment_validation.html')
    with open(assign_path, 'w', encoding='utf-8') as f:
        f.write(assign_html)
    logger.info(f"  Location assignment validation → {assign_path}")

    return {'tour': tour_path, 'trip': trip_path, 'assignment': assign_path}


def _save_coefficients(models: dict, path: str):
    """
    Save coefficient tables as CSV + HTML report (ActivitySim-style output).
    HTML report includes: parameter, value, t-stat, significance, LL, rho².
    """
    rows = []
    combined_sections = []
    mode_sections = []
    dest_sections = []

    for seg, res in models.items():
        for mtype, mdl_key in [('mode', 'mode_model'), ('dest', 'dest_model')]:
            mdl = res.get(mdl_key)
            if mdl is None:
                continue
            try:
                # Use _coef_dict_from_model to handle loaded models
                # (avoids '_mangled' AttributeError on re-loaded Larch objects)
                coef_dict = _coef_dict_from_model(mdl)

                # Attempt to get pf for std_err / t_stat; fall back gracefully.
                # Prefer _final_pf (snapshot taken at end of _estimate with all
                # OPG/proxy fills applied) over the live pf which may be rebuilt
                # from internal Larch state after pickle/unpickle and lose SE info.
                pf = None
                if hasattr(mdl, '_final_pf') and isinstance(mdl._final_pf, pd.DataFrame):
                    pf = mdl._final_pf
                else:
                    try:
                        pf = mdl.pf
                    except AttributeError:
                        try:
                            mdl.mangle(); mdl.unmangle()
                            pf = mdl.pf
                        except Exception:
                            pass

                ll = getattr(mdl, '_cached_ll', None)
                if ll is None or (isinstance(ll, float) and np.isnan(ll)):
                    ll = res.get(mtype + '_ll', np.nan)
                ll = float(ll) if ll is not None else float('nan')
                rho2 = _rho_sq_null(mdl, ll)

                seg_rows = []
                param_names = list(pf.index) if (pf is not None and len(pf) > 0) else list(coef_dict.keys())
                for param_name in param_names:
                    val = coef_dict.get(str(param_name), float('nan'))
                    # Report every parameter, including those holdfasted at exactly
                    # zero (collinear/dropped) — flagged via 'holdfast' rather than
                    # omitted, so the report always matches the full model table.
                    holdfast = bool(pf.loc[param_name].get('holdfast', False)) if (pf is not None and param_name in pf.index) else False
                    se = float('nan')
                    tstat = float('nan')
                    pval  = float('nan')
                    if pf is not None and param_name in pf.index:
                        row_pf = pf.loc[param_name]
                        # Use first non-NaN positive value from SE candidate columns.
                        # NOTE: pandas Series.get() returns NaN (not fallback) when key
                        # exists but value is NaN - chained .get() silently fails.
                        # Loop instead to pick first valid SE.
                        _holdfast = bool(row_pf.get('holdfast', False))
                        for _se_col in ('std_err', 'std_err_opg', 'se', 'stderr'):
                            if _se_col in row_pf.index:
                                _v = float(row_pf[_se_col])
                                if not np.isnan(_v) and (_v > 0 or _holdfast):
                                    se = _v
                                    break
                        for _t_col in ('t_stat', 't_stat_opg', 'tstat'):
                            if _t_col in row_pf.index:
                                _v = float(row_pf[_t_col])
                                if not np.isnan(_v):
                                    tstat = _v
                                    break
                        # If tstat still NaN but se is valid, compute it from value/se
                        if np.isnan(tstat) and not np.isnan(se) and se > 0:
                            tstat = float(val) / se
                        for _p_col in ('pval', 'p_value', 'pvalue', 'pval_opg'):
                            if _p_col in row_pf.index:
                                _v = float(row_pf[_p_col])
                                if not np.isnan(_v):
                                    pval = _v
                                    break
                        # Compute pval from tstat if still missing
                        if np.isnan(pval) and not np.isnan(tstat):
                            from scipy.stats import norm as _norm
                            pval = float(2.0 * _norm.sf(abs(tstat)))
                    r = {
                        'segment':    seg,
                        'model_type': mtype,
                        'parameter':  str(param_name),
                        'value':      val,
                        'std_err':    se,
                        'tstat':      tstat,
                        'pval':       pval,
                        'loglike':    ll,
                        'rho2':       rho2,
                        'holdfast':   holdfast,
                    }
                    rows.append(r)
                    seg_rows.append(r)

                section = _html_model_table(seg, mtype, seg_rows, ll, rho2,
                                             res.get('n_tours', '?'))
                combined_sections.append(section)
                if mtype == 'mode':
                    mode_sections.append(section)
                elif mtype == 'dest':
                    dest_sections.append(section)
            except Exception as e:
                logger.warning(f"Could not extract coefficients for {seg}/{mtype}: {e}")
                import traceback; logger.debug(traceback.format_exc())
                # Fallback to pre-extracted plain coefficient dicts from run_stage5.
                # This guarantees CSV/HTML artifacts are still written even when
                # the live Larch model object cannot be introspected post-estimation.
                try:
                    fallback_key = 'mode_coef' if mtype == 'mode' else 'dest_coef'
                    fallback_coef = res.get(fallback_key, {}) if isinstance(res, dict) else {}
                    if isinstance(fallback_coef, dict) and fallback_coef:
                        ll = res.get(mtype + '_ll', np.nan)
                        seg_rows = []
                        for param_name, val in fallback_coef.items():
                            r = {
                                'segment':    seg,
                                'model_type': mtype,
                                'parameter':  str(param_name),
                                'value':      float(val),
                                'std_err':    float('nan'),
                                'tstat':      float('nan'),
                                'pval':       float('nan'),
                                'loglike':    float(ll) if ll is not None else float('nan'),
                                'rho2':       float('nan'),
                                'holdfast':   False,
                            }
                            rows.append(r)
                            seg_rows.append(r)
                        section = _html_model_table(
                            seg,
                            mtype,
                            seg_rows,
                            float(ll) if ll is not None else float('nan'),
                            float('nan'),
                            res.get('n_tours', '?'),
                        )
                        combined_sections.append(section)
                        if mtype == 'mode':
                            mode_sections.append(section)
                        else:
                            dest_sections.append(section)
                        logger.warning(
                            f"Using fallback coefficient dict for {seg}/{mtype} "
                            f"({len(fallback_coef)} params)"
                        )
                except Exception as _fb_e:
                    logger.debug(f"Fallback coefficient extraction failed for {seg}/{mtype}: {_fb_e}")

        # ── add mixed logit section if available ──────────────────────
        mxl = res.get('mxl_result')
        if mxl is not None:
            try:
                mxl_rows_html = '\n'.join(
                    f'<tr><td>{name}</td>'
                    f'<td style="text-align:right">{mxl["coef"].get(name, 0):.4f}</td>'
                    f'<td style="text-align:right">{mxl["std_err"].get(name, 0):.4f}</td>'
                    f'<td style="text-align:right; color:{"#1a7a1a" if abs(mxl["tstat"].get(name,0))>1.96 else "#b71c1c"}">'
                    f'{mxl["tstat"].get(name, 0):.2f}</td></tr>'
                    for name in mxl['coef']
                )
                mxl_section = f"""
<div class="model-section">
  <h2>{seg.upper()} - mode model (Mixed Logit - IVT~Normal)</h2>
  <p class="stats">
    <b>Log-likelihood:</b> {mxl['ll']:.1f} &nbsp;|&nbsp;
    <b>ρ²:</b> {mxl['rho2']:.3f} &nbsp;|&nbsp;
    <b>Estimation obs:</b> {mxl['n_obs']} &nbsp;|&nbsp;
    <b>Random vars:</b> {', '.join(f'{k}~{v}' for k,v in mxl['randvars'].items())}
  </p>
  <table>
    <thead><tr><th>Parameter</th><th>Mean</th><th>Std Err</th><th>t-stat</th></tr></thead>
    <tbody>{mxl_rows_html}</tbody>
  </table>
  <p class="sig-note">Random IVT coefficient captures taste heterogeneity across travellers.
  Mean = population mean value of time sensitivity.
  Std dev = spread of distribution (if significant → heterogeneity confirmed).</p>
</div>"""
                combined_sections.append(mxl_section)
                mode_sections.append(mxl_section)
            except Exception as mxl_e:
                logger.debug(f"MXL HTML failed for {seg}: {mxl_e}")

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  Coefficients saved → {path}")

        # write HTML reports
        html_path = path.replace('.csv', '.html')
        _write_html_report(combined_sections, html_path)
        mode_html_path = path.replace('.csv', '_mode.html')
        dest_html_path = path.replace('.csv', '_dest.html')
        _write_html_report(mode_sections, mode_html_path)
        _write_html_report(dest_sections, dest_html_path)
        print(f"  Combined HTML report saved  → {html_path}")
        print(f"  Mode-only HTML report saved  → {mode_html_path}")
        print(f"  Destination-only HTML report saved  → {dest_html_path}")

        # console summary
        print("\n  ── Model Summary ──────────────────────────────────────────")
        for seg in df['segment'].unique():
            sub = df[df['segment'] == seg]
            for mtype in sub['model_type'].unique():
                ssub = sub[sub['model_type'] == mtype]
                ll   = ssub['loglike'].iloc[0]
                rho2 = ssub['rho2'].iloc[0]
                n_free = int((~ssub['holdfast'].astype(bool)).sum()) if 'holdfast' in ssub.columns else len(ssub)
                print(f"\n  [{seg.upper()} {mtype}]  "
                      f"LL={ll:.1f}  ρ²={rho2:.3f}  n_params={len(ssub)} ({n_free} free)")
                print(f"  {'Parameter':<35} {'Value':>8}  {'t-stat':>7}  Sig")
                print(f"  {'-'*60}")
                for _, r in ssub.iterrows():
                    t = r['tstat']
                    try:
                        t_f = float(t)
                        at = abs(t_f)
                        sig = ('***' if at > 3.3 else '**' if at > 2.6 else '*' if at > 1.96 else '')
                        t_str = f'{t_f:>7.2f}' if np.isfinite(t_f) else '    n/a'
                    except Exception:
                        t_str = '    n/a'; sig = ''
                    se = r['std_err']
                    se_str = f'{float(se):.4f}' if pd.notna(se) and np.isfinite(float(se)) else 'n/a'
                    hf = ' (holdfast: collinear)' if bool(r.get('holdfast', False)) else ''
                    print(f"  {r['parameter']:<35} {r['value']:>8.4f}  SE={se_str:>7}  {t_str}  {sig}{hf}")
    else:
        print("  WARNING: no coefficients extracted - writing diagnostic placeholder HTML")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        pd.DataFrame(columns=['segment', 'model_type', 'parameter', 'value', 'std_err', 'tstat', 'pval', 'loglike', 'rho2']).to_csv(path, index=False)
        _write_html_report([
            """
<div class=\"model-section\">
  <h2>No Coefficients Extracted</h2>
  <p class=\"stats\">Stage 5 ran but no coefficient rows could be extracted from model objects or fallback dicts.</p>
  <p>Check stage5_models_trace.txt and pipeline.log for segment-level failures.</p>
</div>
"""
        ], path.replace('.csv', '.html'))


def _html_model_table(seg, mtype, rows, ll, rho2, n_tours):
    """Build one HTML table section for a segment/model."""
    def _fmt(v, fmt='.4f'):
        try:
            f = float(v)
            return format(f, fmt) if np.isfinite(f) else '-'
        except Exception:
            return '-'
    def sig_stars(t):
        try:
            at = abs(float(t))
            if not np.isfinite(at): return ''
            return '★★★' if at > 3.3 else '★★' if at > 2.6 else '★' if at > 1.96 else ''
        except Exception:
            return ''
    def sig_color(t, p):
        try:
            at = abs(float(t))
            if not np.isfinite(at): return '#888888'
            return '#1a7a1a' if at > 3.3 else '#2e7d32' if at > 2.6 else '#558b2f' if at > 1.96 else '#b71c1c'
        except Exception:
            return '#888888'
    def pval_style(p):
        try:
            pf = float(p)
            if not np.isfinite(pf): return ''
            if pf < 0.001: return "style='color:#1a7a1a;font-weight:bold'"
            if pf < 0.01:  return "style='color:#2e7d32;font-weight:bold'"
            if pf < 0.05:  return "style='color:#558b2f'"
            if pf < 0.10:  return "style='color:#f57c00'"
            return "style='color:#b71c1c'"
        except Exception:
            return ''

    rows_html = '\n'.join(
        f'<tr{" style=\'color:#999999;font-style:italic\'" if r.get("holdfast") else ""}>'
        f'<td>{r["parameter"]}{" (holdfast)" if r.get("holdfast") else ""}</td>'
        f'<td style="text-align:right">{_fmt(r["value"])}</td>'
        f'<td style="text-align:right">{_fmt(r["std_err"])}</td>'
        f'<td style="text-align:right; color:{sig_color(r["tstat"], r.get("pval"))}">'
        f'{_fmt(r["tstat"], ".2f")} {sig_stars(r["tstat"])}</td>'
        f'<td style="text-align:right" {pval_style(r.get("pval"))}>{_fmt(r.get("pval", float("nan")), ".4f")}</td>'
        f'</tr>'
        for r in rows
    )
    rho2_str = _fmt(rho2, '.4f')
    ll_str   = _fmt(ll,   '.2f')
    return f"""
<div class="model-section">
  <h2>{seg.upper()} - {mtype} model</h2>
  <p class="stats">
    <b>Log-likelihood:</b> {ll_str} &nbsp;|&nbsp;
    <b>ρ² (null):</b> {rho2_str} &nbsp;|&nbsp;
    <b>N obs:</b> {n_tours} &nbsp;|&nbsp;
    <b>Params (free):</b> {sum(1 for r in rows if not r.get('holdfast', False))}
  </p>
  <table>
    <thead>
      <tr><th>Parameter</th><th>Value</th><th>Std&nbsp;Err</th><th>t-stat</th><th>p-value</th></tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
  <p class="sig-note">
    p-value colour: <span style="color:#1a7a1a;font-weight:bold">p&lt;0.001 ★★★</span> &nbsp;
    <span style="color:#2e7d32;font-weight:bold">p&lt;0.01 ★★</span> &nbsp;
    <span style="color:#558b2f">p&lt;0.05 ★</span> &nbsp;
    <span style="color:#f57c00">p&lt;0.10</span> &nbsp;
    <span style="color:#b71c1c">p≥0.10 (not significant)</span>
  </p>
</div>"""


def _write_html_report(sections: list, path: str):
    """Write a full HTML report of all estimated models."""
    body = '\n'.join(sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stage 5 Model Results - Destination + Mode Choice</title>
  <style>
    body   {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
    h1     {{ border-bottom: 2px solid #1565c0; padding-bottom: 8px; }}
    h2     {{ color: #1565c0; margin-top: 32px; }}
    .model-section {{ margin-bottom: 40px; border: 1px solid #ccc;
                      border-radius: 6px; padding: 20px; }}
    .stats {{ background: #e3f2fd; padding: 8px 12px; border-radius: 4px;
              font-size: 0.95em; margin-bottom: 12px; }}
    table  {{ border-collapse: collapse; width: 100%; }}
    th     {{ background: #1565c0; color: white; padding: 8px 12px;
              text-align: left; }}
    td     {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f5f5f5; }}
    .sig-note {{ font-size: 0.8em; color: #666; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Stage 5 - Destination + Mode Choice Model Results</h1>
  <p>Generated by stage5_destination_choice.py - QTS/HHTS ABM Pipeline</p>
{body}
</body>
</html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


def print_coefficients(models: dict, save_dir: str = 'saved_data'):
    """Print coefficient tables to console (call after run_stage5)."""
    os.makedirs(save_dir, exist_ok=True)
    _save_coefficients(models, os.path.join(save_dir, 'stage5_coefficients.csv'))


def compute_validation_metrics(
    obs_tours: pd.DataFrame,
    pred_plan: pd.DataFrame,
    models: dict = None,
    persons: pd.DataFrame = None,
    save_path: str = 'saved_data/stage5_validation_metrics.csv',
) -> dict:
    """
    Compute comprehensive Stage 5 validation metrics.

    Returns a dict with:
      zone_rmse        - RMSE of observed vs predicted destination zone share (per segment)
      zone_pearson     - Pearson correlation of zone shares (per segment)
      zone_kl          - KL divergence obs‖pred (per segment)
      mode_share_obs   - observed mode share (%)
      mode_share_pred  - predicted mode share (%)
      mean_tt_obs      - mean observed travel time by segment (min)
      mean_tt_pred     - mean predicted travel time by segment (min)
      dist_decay_rmse  - RMSE of distance-decay profile (trip length distribution)
      param_table      - DataFrame of all estimated parameters with t-stats
    """
    from scipy.stats import pearsonr
    metrics: dict = {}
    eps = 1e-9

    # ── Diagnostic: summarise what we received ────────────────────────────
    _pp_cols = list(pred_plan.columns) if not pred_plan.empty else []
    _pp_rt   = pred_plan['row_type'].value_counts().to_dict() if 'row_type' in _pp_cols else 'no row_type col'
    _pp_purp = pred_plan['purpose'].value_counts().head(10).to_dict() if 'purpose' in _pp_cols else 'no purpose col'
    print(f"\n  [validation] obs_tours: {len(obs_tours)} rows | "
          f"pred_plan: {len(pred_plan)} rows")
    print(f"  [validation] pred_plan row_type counts: {_pp_rt}")
    print(f"  [validation] pred_plan top purposes: {_pp_purp}")
    print(f"  [validation] pred_plan columns: {_pp_cols}")

    # ── Segment membership ────────────────────────────────────────────────
    SEG_PURPOSES = {
        'work':    _WORK_PURPOSES,
        'edu':     _EDU_PURPOSES,
        'nonmand': _NONMAND_PURPOSES,
    }
    obs_seg_col = next(
        (c for c in ['primary_purpose', 'main_purpose', 'purpose']
         if c in obs_tours.columns), None
    )

    # ── 1. Zone-level destination accuracy ────────────────────────────────
    metrics['zone_rmse']    = {}
    metrics['zone_pearson'] = {}
    metrics['zone_kl']      = {}

    _has_rowtype_pred  = 'row_type' in pred_plan.columns
    _has_purpose_pred  = 'purpose' in pred_plan.columns
    _has_dtaz_pred     = 'DTAZ' in pred_plan.columns

    for seg, purposes in SEG_PURPOSES.items():
        # observed
        if obs_seg_col:
            obs_mask = obs_tours[obs_seg_col].isin(purposes)
            obs_d    = pd.to_numeric(obs_tours.loc[obs_mask, 'DTAZ'], errors='coerce').dropna()
        else:
            obs_d = pd.Series(dtype=float)

        # predicted - prefer activity rows (each activity IS the destination),
        # fall back to trip rows, then any purpose-matching rows.
        pred_d = pd.Series(dtype=float)
        if _has_purpose_pred and _has_dtaz_pred:
            if _has_rowtype_pred:
                # Attempt 1: activity dwell rows
                _am = ((pred_plan['row_type'] == 'activity') &
                        pred_plan['purpose'].isin(purposes))
                pred_d = pd.to_numeric(pred_plan.loc[_am, 'DTAZ'], errors='coerce').dropna()
            if pred_d.empty and _has_rowtype_pred:
                # Attempt 2: outbound trip rows
                _tm = ((pred_plan['row_type'] == 'trip') &
                        pred_plan['purpose'].isin(purposes))
                pred_d = pd.to_numeric(pred_plan.loc[_tm, 'DTAZ'], errors='coerce').dropna()
            if pred_d.empty:
                # Attempt 3: any row with matching purpose
                _am2 = pred_plan['purpose'].isin(purposes)
                pred_d = pd.to_numeric(pred_plan.loc[_am2, 'DTAZ'], errors='coerce').dropna()
        elif _has_dtaz_pred:
            pred_d = pd.to_numeric(pred_plan['DTAZ'], errors='coerce').dropna()

        if obs_d.empty or pred_d.empty:
            logger.debug(f"Destination metrics [{seg}]: obs n={len(obs_d)}, pred n={len(pred_d)} - skipping")
            metrics['zone_rmse'][seg]    = float('nan')
            metrics['zone_pearson'][seg] = float('nan')
            metrics['zone_kl'][seg]      = float('nan')
            continue

        all_zones = sorted(set(obs_d.astype(int)) | set(pred_d.astype(int)))
        obs_cnt  = obs_d.astype(int).value_counts().reindex(all_zones, fill_value=0)
        pred_cnt = pred_d.astype(int).value_counts().reindex(all_zones, fill_value=0)

        obs_share  = obs_cnt  / (obs_cnt.sum()  + eps)
        pred_share = pred_cnt / (pred_cnt.sum() + eps)

        rmse = float(np.sqrt(np.mean((obs_share.values - pred_share.values) ** 2)))
        if obs_share.std() > 0 and pred_share.std() > 0:
            corr, _ = pearsonr(obs_share.values, pred_share.values)
        else:
            corr = float('nan')
        # KL divergence (obs‖pred)
        kl = float(np.sum(
            obs_share.values * np.log((obs_share.values + eps) / (pred_share.values + eps))
        ))
        metrics['zone_rmse'][seg]    = rmse
        metrics['zone_pearson'][seg] = float(corr)
        metrics['zone_kl'][seg]      = kl

    # ── 2. Mode share comparison ──────────────────────────────────────────
    obs_mode_col  = next((c for c in ['tour_mode', 'mode', 'mode_code']
                          if c in obs_tours.columns), None)
    # Accept 'mode' (int) or 'mode_name' (string) from pred_plan
    pred_mode_col = next((c for c in ['mode', 'mode_name', 'pred_mode']
                          if c in pred_plan.columns), None)

    if obs_mode_col:
        obs_modes = pd.to_numeric(obs_tours[obs_mode_col], errors='coerce').dropna().astype(int)
        metrics['mode_obs_n'] = int(len(obs_modes))
        obs_share  = obs_modes.value_counts(normalize=True) * 100
        metrics['mode_share_obs'] = {
            MODE_NAMES.get(k, str(k)): round(float(v), 2)
            for k, v in obs_share.items()
        }
    else:
        metrics['mode_share_obs'] = {}

    if pred_mode_col:
        # Use trip rows only (activity rows have mode=0/mode_name='')
        if _has_rowtype_pred:
            trip_rows = pred_plan[pred_plan['row_type'] == 'trip']
        else:
            trip_rows = pred_plan
        if trip_rows.empty:
            trip_rows = pred_plan   # fallback: use all rows

        mode_col_data = trip_rows[pred_mode_col]
        if mode_col_data.dtype == object or str(mode_col_data.dtype) == 'object':
            # String mode names - map to int codes for aggregation
            inv_map = {v: k for k, v in MODE_NAMES.items()}
            pred_modes = mode_col_data.map(inv_map).dropna().astype(int)
        else:
            pred_modes = pd.to_numeric(mode_col_data, errors='coerce').dropna().astype(int)
        # Exclude 0 (activity dwell placeholder)
        pred_modes = pred_modes[pred_modes > 0]
        metrics['mode_pred_n_raw'] = int(len(pred_modes))

        # ActivitySim-style validation compares aggregate distributions directly;
        # no random resampling of synthetic predictions.
        pred_modes_cmp = pred_modes
        metrics['mode_pred_n_comparable'] = int(len(pred_modes_cmp))

        pred_share = pred_modes_cmp.value_counts(normalize=True) * 100
        metrics['mode_share_pred'] = {
            MODE_NAMES.get(k, str(k)): round(float(v), 2)
            for k, v in pred_share.items()
        }
    else:
        metrics['mode_share_pred'] = {}

    # ── 3. Mean travel time by segment ───────────────────────────────────
    metrics['mean_tt_obs']  = {}
    metrics['mean_tt_pred'] = {}

    tt_col_obs  = next((c for c in ['travtime', 'travel_time', 'TTIME'] if c in obs_tours.columns), None)
    tt_col_pred = 'travtime'

    for seg, purposes in SEG_PURPOSES.items():
        if obs_seg_col and tt_col_obs:
            m = obs_tours[obs_tours[obs_seg_col].isin(purposes)]
            v = pd.to_numeric(m[tt_col_obs], errors='coerce').dropna()
            metrics['mean_tt_obs'][seg] = float(v.mean()) if len(v) else float('nan')
        else:
            metrics['mean_tt_obs'][seg] = float('nan')

        if tt_col_pred in pred_plan.columns:
            if _has_rowtype_pred and _has_purpose_pred:
                pm = pred_plan[
                    (pred_plan['row_type'] == 'trip') &
                    pred_plan['purpose'].isin(purposes)
                ]
            elif _has_rowtype_pred:
                pm = pred_plan[pred_plan['row_type'] == 'trip']
            else:
                pm = pred_plan
            v = pd.to_numeric(pm[tt_col_pred], errors='coerce').dropna()
            metrics['mean_tt_pred'][seg] = float(v.mean()) if len(v) else float('nan')
        else:
            metrics['mean_tt_pred'][seg] = float('nan')

    # ── 4. Distance-decay profile RMSE ───────────────────────────────────
    dist_col_pred = next((c for c in ['dist', 'SOVTOLL_DIST__AM'] if c in pred_plan.columns), None)
    dist_col_obs  = next((c for c in ['dist', 'SOVTOLL_DIST__AM', 'tour_distance'] if c in obs_tours.columns), None)

    if dist_col_pred and dist_col_obs:
        bins = np.arange(0, 105, 5)   # 0-100 km in 5-km bands
        if _has_rowtype_pred:
            _dist_rows = pred_plan[pred_plan['row_type'] == 'trip']
        else:
            _dist_rows = pred_plan
        pred_dists = pd.to_numeric(_dist_rows[dist_col_pred], errors='coerce').dropna()
        obs_dists  = pd.to_numeric(obs_tours[dist_col_obs], errors='coerce').dropna()
        if len(pred_dists) > 0 and len(obs_dists) > 0:
            pred_h, _ = np.histogram(pred_dists, bins=bins, density=True)
            obs_h, _  = np.histogram(obs_dists,  bins=bins, density=True)
            metrics['dist_decay_rmse'] = float(np.sqrt(np.mean((obs_h - pred_h) ** 2)))
        else:
            metrics['dist_decay_rmse'] = float('nan')
    else:
        metrics['dist_decay_rmse'] = float('nan')

    # ── 5. Parameter table from estimated models ─────────────────────────
    param_rows = []
    if models:
        for seg, res in models.items():
            for mtype in ('mode_model', 'dest_model'):
                mdl = res.get(mtype)
                if mdl is None:
                    continue
                # Prefer _final_pf (populated with OPG/proxy SEs at estimation time)
                # over the live pf which may be rebuilt after pickle/unpickle.
                if hasattr(mdl, '_final_pf') and isinstance(mdl._final_pf, pd.DataFrame):
                    pf = mdl._final_pf
                elif hasattr(mdl, 'pf'):
                    pf = mdl.pf
                else:
                    continue
                for pname in pf.index:
                    row = pf.loc[pname]
                    val  = float(row.get('value',   float('nan')))
                    se   = float(row.get('std_err', float('nan')))
                    tstat = float(row.get('t_stat', val / se if se > 0 else float('nan')))
                    pval  = float(row.get('pval',   float('nan')))
                    hf    = bool(row.get('holdfast', False))
                    param_rows.append({
                        'segment': seg,
                        'model':   mtype.replace('_model', ''),
                        'parameter': pname,
                        'value': val,
                        'std_err': se,
                        't_stat': tstat,
                        'pval': pval,
                        'holdfast': hf,
                        'significant': abs(tstat) >= 1.96 if np.isfinite(tstat) else False,
                        'se_source': row.get('std_err_source', 'hessian'),
                    })
    metrics['param_table'] = pd.DataFrame(param_rows)

    # ── 7. ActivitySim-style level checks: workplace/school/tour/trip ───────
    # Explicitly validate all location-choice levels used in this pipeline:
    #   workplace location (person anchor), school location (person anchor),
    #   tour-level destination rows (activity rows), and trip-level OD rows.
    metrics['location_levels'] = {
        'workplace': {},
        'school': {},
        'tour': {},
        'trip': {},
    }

    _plan = pred_plan.copy() if pred_plan is not None else pd.DataFrame()
    if not _plan.empty and 'person_id' in _plan.columns:
        _plan['person_id'] = _plan['person_id'].astype(str)

    _persons = persons.copy() if isinstance(persons, pd.DataFrame) else pd.DataFrame()
    if not _persons.empty and 'person_id' in _persons.columns:
        _persons['person_id'] = _persons['person_id'].astype(str)

    # Ensure anchor columns exist on plan rows (from persons if needed).
    if not _plan.empty and not _persons.empty and 'person_id' in _plan.columns:
        for _anchor_col in ('WORK_TAZ', 'SCHOOL_TAZ'):
            if _anchor_col not in _plan.columns and _anchor_col in _persons.columns:
                _map = _persons.set_index('person_id')[_anchor_col].to_dict()
                _plan[_anchor_col] = _plan['person_id'].map(_map)

    if not _plan.empty:
        _has_row_type = 'row_type' in _plan.columns
        _act_rows = _plan[_plan['row_type'] == 'activity'].copy() if _has_row_type else _plan.copy()
        _trip_rows = _plan[_plan['row_type'] == 'trip'].copy() if _has_row_type else _plan.copy()

        # --- workplace anchor checks (person/work location level) ---
        if {'purpose', 'DTAZ', 'WORK_TAZ'}.issubset(_act_rows.columns):
            _w = _act_rows[_act_rows['purpose'].isin(_WORK_PURPOSES)].copy()
            _w['DTAZ'] = pd.to_numeric(_w['DTAZ'], errors='coerce').fillna(0).astype(int)
            _w['WORK_TAZ'] = pd.to_numeric(_w['WORK_TAZ'], errors='coerce').fillna(0).astype(int)
            _w = _w[(_w['DTAZ'] > 0) & (_w['WORK_TAZ'] > 0)]
            if len(_w):
                _match = float((_w['DTAZ'] == _w['WORK_TAZ']).mean())
                metrics['location_levels']['workplace'] = {
                    'n_activity_rows': int(len(_w)),
                    'anchor_match_rate': _match,
                }

        # --- school anchor checks (person/school location level) ---
        if {'purpose', 'DTAZ', 'SCHOOL_TAZ'}.issubset(_act_rows.columns):
            _s = _act_rows[_act_rows['purpose'].isin(_EDU_PURPOSES)].copy()
            _s['DTAZ'] = pd.to_numeric(_s['DTAZ'], errors='coerce').fillna(0).astype(int)
            _s['SCHOOL_TAZ'] = pd.to_numeric(_s['SCHOOL_TAZ'], errors='coerce').fillna(0).astype(int)
            _s = _s[(_s['DTAZ'] > 0) & (_s['SCHOOL_TAZ'] > 0)]
            if len(_s):
                _match = float((_s['DTAZ'] == _s['SCHOOL_TAZ']).mean())
                metrics['location_levels']['school'] = {
                    'n_activity_rows': int(len(_s)),
                    'anchor_match_rate': _match,
                }

        # --- tour-level destination checks ---
        if {'purpose', 'DTAZ'}.issubset(_act_rows.columns):
            _a = _act_rows.copy()
            _a['DTAZ'] = pd.to_numeric(_a['DTAZ'], errors='coerce').fillna(0).astype(int)
            _valid = (_a['DTAZ'] > 0).mean() if len(_a) else np.nan
            metrics['location_levels']['tour'] = {
                'n_activity_rows': int(len(_a)),
                'valid_dtaz_rate': float(_valid) if np.isfinite(_valid) else float('nan'),
                'unique_dtaz': int(_a.loc[_a['DTAZ'] > 0, 'DTAZ'].nunique()) if len(_a) else 0,
            }

        # --- trip-level destination checks ---
        if {'OTAZ', 'DTAZ'}.issubset(_trip_rows.columns):
            _t = _trip_rows.copy()
            _t['OTAZ'] = pd.to_numeric(_t['OTAZ'], errors='coerce').fillna(0).astype(int)
            _t['DTAZ'] = pd.to_numeric(_t['DTAZ'], errors='coerce').fillna(0).astype(int)
            _valid = ((_t['OTAZ'] > 0) & (_t['DTAZ'] > 0)).mean() if len(_t) else np.nan
            _nonzero_dist = float('nan')
            if 'travtime' in _t.columns:
                _tt = pd.to_numeric(_t['travtime'], errors='coerce').fillna(0)
                _nonzero_dist = float((_tt > 0).mean()) if len(_tt) else float('nan')
            _trip_metrics = {
                'n_trip_rows': int(len(_t)),
                'valid_od_rate': float(_valid) if np.isfinite(_valid) else float('nan'),
                'positive_travtime_rate': _nonzero_dist,
            }

            # For mandatory trips, destination OR origin should touch the person's
            # usual anchor (outbound to anchor, inbound from anchor).
            if {'purpose', 'WORK_TAZ', 'SCHOOL_TAZ'}.issubset(_t.columns):
                _w = _t[_t['purpose'].isin(_WORK_PURPOSES)].copy()
                _w['WORK_TAZ'] = pd.to_numeric(_w['WORK_TAZ'], errors='coerce').fillna(0).astype(int)
                _w = _w[_w['WORK_TAZ'] > 0]
                if len(_w):
                    _touch = ((_w['OTAZ'] == _w['WORK_TAZ']) | (_w['DTAZ'] == _w['WORK_TAZ'])).mean()
                    _trip_metrics['work_anchor_touch_rate'] = float(_touch)

                _s = _t[_t['purpose'].isin(_EDU_PURPOSES)].copy()
                _s['SCHOOL_TAZ'] = pd.to_numeric(_s['SCHOOL_TAZ'], errors='coerce').fillna(0).astype(int)
                _s = _s[_s['SCHOOL_TAZ'] > 0]
                if len(_s):
                    _touch = ((_s['OTAZ'] == _s['SCHOOL_TAZ']) | (_s['DTAZ'] == _s['SCHOOL_TAZ'])).mean()
                    _trip_metrics['school_anchor_touch_rate'] = float(_touch)

            metrics['location_levels']['trip'] = _trip_metrics

    # ── 6. Mode share comparison (obs vs predicted) ──────────────────────────
    # NOTE: obs_tours is the SURVEY population; pred_plan is the SYNTHETIC
    # population.  They have completely disjoint person IDs - a row-by-row
    # confusion matrix is meaningless.  Instead we compare the MARGINAL MODE
    # SHARE DISTRIBUTIONS, which is the correct validation for an MNL.
    try:
        MODE_LABELS = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
        ALL_MODE_NAMES = ['DA', 'SR', 'Walk', 'Bike', 'Transit']
        obs_mode_col  = next((c for c in ['tour_mode', 'mode'] if c in obs_tours.columns), None)
        # pred_plan can have 'mode_name', 'pred_mode', or 'mode' columns
        pred_mode_col = next((c for c in ['mode_name', 'pred_mode', 'mode']
                              if c in pred_plan.columns), None)

        if obs_mode_col and pred_mode_col:
            # ── observed mode shares (survey) ────────────────────────────────
            obs_raw = pd.to_numeric(obs_tours[obs_mode_col], errors='coerce').dropna().astype(int)
            obs_share = (obs_raw.map(MODE_LABELS)
                                .value_counts(normalize=True)
                                .reindex(ALL_MODE_NAMES, fill_value=0.0) * 100)

            # ── predicted mode shares (synthetic population) ─────────────────
            pred_col = pred_plan[pred_mode_col]
            if pred_col.dtype == object or str(pred_col.dtype) == 'object':
                # string mode names (mode_name column)
                pred_raw_str = pred_col.dropna()
                pred_share = (pred_raw_str
                              .value_counts(normalize=True)
                              .reindex(ALL_MODE_NAMES, fill_value=0.0) * 100)
            else:
                pred_raw = pd.to_numeric(pred_col, errors='coerce').dropna().astype(int)
                pred_share = (pred_raw.map(MODE_LABELS)
                                      .value_counts(normalize=True)
                                      .reindex(ALL_MODE_NAMES, fill_value=0.0) * 100)

            # Filter to trips only (not activity dwell rows which have mode=0/mode_name='')
            if 'row_type' in pred_plan.columns:
                trips_only = pred_plan[pred_plan['row_type'] == 'trip']
                if len(trips_only) > 0:
                    pred_col_t = trips_only[pred_mode_col]
                    if pred_col_t.dtype == object or str(pred_col_t.dtype) == 'object':
                        pred_raw_t = pred_col_t[pred_col_t.isin(ALL_MODE_NAMES)]
                        pred_share = (pred_raw_t.value_counts(normalize=True)
                                               .reindex(ALL_MODE_NAMES, fill_value=0.0) * 100)
                    else:
                        pred_raw_t = pd.to_numeric(pred_col_t, errors='coerce').dropna().astype(int)
                        pred_raw_t = pred_raw_t[pred_raw_t.isin(MODE_LABELS)]
                        pred_share = (pred_raw_t.map(MODE_LABELS)
                                                 .value_counts(normalize=True)
                                                 .reindex(ALL_MODE_NAMES, fill_value=0.0) * 100)

            # ── build comparison table ────────────────────────────────────────
            share_df = pd.DataFrame({
                'Observed_%': obs_share.round(1),
                'Predicted_%': pred_share.round(1),
                'Diff_%': (pred_share - obs_share).round(1),
            })
            share_df.index.name = 'Mode'
            metrics['mode_confusion_matrix'] = share_df
            try:
                _obs_p = np.clip(obs_share.values.astype(float) / 100.0, 1e-12, 1.0)
                _pred_p = np.clip(pred_share.values.astype(float) / 100.0, 1e-12, 1.0)
                _obs_p = _obs_p / _obs_p.sum()
                _pred_p = _pred_p / _pred_p.sum()
                _m = 0.5 * (_obs_p + _pred_p)
                _kl_o = float(np.sum(_obs_p * np.log(_obs_p / _m)))
                _kl_p = float(np.sum(_pred_p * np.log(_pred_p / _m)))
                metrics['mode_share_jsd'] = float(0.5 * (_kl_o + _kl_p))
                metrics['mode_share_mae_pp'] = float(np.mean(np.abs(obs_share.values - pred_share.values)))
            except Exception:
                metrics['mode_share_jsd'] = float('nan')
                metrics['mode_share_mae_pp'] = float('nan')
            try:
                import os as _os
                _os.makedirs('saved_data', exist_ok=True)
                share_df.to_csv('saved_data/mode_share_comparison.csv')
                logger.info("Mode share comparison saved → saved_data/mode_share_comparison.csv")
            except Exception:
                pass
        else:
            metrics['mode_confusion_matrix'] = pd.DataFrame()
            metrics['mode_share_jsd'] = float('nan')
            metrics['mode_share_mae_pp'] = float('nan')
    except Exception as _conf_e:
        logger.debug(f"Mode share comparison failed: {_conf_e}")
        import traceback; logger.debug(traceback.format_exc())
        metrics['mode_confusion_matrix'] = pd.DataFrame()
        metrics['mode_share_jsd'] = float('nan')
        metrics['mode_share_mae_pp'] = float('nan')

    # ── 7a. Per-segment top-zone discrepancies (identifies over/under-predicted zones) ───
    metrics['top_zone_discrepancies'] = {}
    for seg, purposes in SEG_PURPOSES.items():
        try:
            if obs_seg_col:
                obs_mask = obs_tours[obs_seg_col].isin(purposes)
                obs_d    = pd.to_numeric(obs_tours.loc[obs_mask, 'DTAZ'], errors='coerce').dropna().astype(int)
            else:
                obs_d = pd.Series(dtype=int)
            pred_d = pd.Series(dtype=int)
            if _has_purpose_pred and _has_dtaz_pred:
                if _has_rowtype_pred:
                    _m = ((pred_plan['row_type'] == 'activity') &
                          pred_plan['purpose'].isin(purposes))
                    pred_d = pd.to_numeric(pred_plan.loc[_m, 'DTAZ'], errors='coerce').dropna().astype(int)
                if pred_d.empty:
                    _m = pred_plan['purpose'].isin(purposes)
                    pred_d = pd.to_numeric(pred_plan.loc[_m, 'DTAZ'], errors='coerce').dropna().astype(int)
            if len(obs_d) > 0 and len(pred_d) > 0:
                all_z = sorted(set(obs_d) | set(pred_d))
                obs_n = obs_d.value_counts().reindex(all_z, fill_value=0)
                pred_n = pred_d.value_counts().reindex(all_z, fill_value=0)
                # Scale predicted to same total as observed for fair comparison
                scale = len(obs_d) / max(len(pred_d), 1)
                pred_n_scaled = (pred_n * scale).round(1)
                diff = pred_n_scaled - obs_n
                disc = pd.DataFrame({'obs': obs_n, 'pred_scaled': pred_n_scaled,
                                     'diff': diff})
                disc = disc.reindex(disc['diff'].abs().sort_values(ascending=False).index)
                metrics['top_zone_discrepancies'][seg] = disc.head(20)
                try:
                    os.makedirs('saved_data', exist_ok=True)
                    disc.to_csv(f'saved_data/zone_discrepancy_{seg}.csv')
                except Exception:
                    pass
        except Exception as _zd_e:
            logger.debug(f"Zone discrepancy [{seg}]: {_zd_e}")

    # ── 7b. Per-segment mode share comparison ──────────────────────────────
    metrics['mode_share_by_segment'] = {}
    MODE_LABELS = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
    ALL_MODE_NAMES = ['DA', 'SR', 'Walk', 'Bike', 'Transit']
    _obs_mc = next((c for c in ['tour_mode', 'mode'] if c in obs_tours.columns), None)
    _pred_mc = next((c for c in ['pred_mode', 'mode', 'mode_name'] if c in pred_plan.columns), None)
    for seg, purposes in SEG_PURPOSES.items():
        try:
            if obs_seg_col and _obs_mc:
                ob = obs_tours.loc[obs_tours[obs_seg_col].isin(purposes), _obs_mc]
                ob = pd.to_numeric(ob, errors='coerce').dropna().astype(int)
                obs_seg_share = ob.map(MODE_LABELS).value_counts(normalize=True).reindex(
                    ALL_MODE_NAMES, fill_value=0.0) * 100
            else:
                obs_seg_share = pd.Series(0.0, index=ALL_MODE_NAMES)
            if _pred_mc and _has_purpose_pred:
                if _has_rowtype_pred:
                    pr = pred_plan.loc[(pred_plan['row_type'] == 'trip') &
                                       pred_plan['purpose'].isin(purposes), _pred_mc]
                else:
                    pr = pred_plan.loc[pred_plan['purpose'].isin(purposes), _pred_mc]
                if pr.dtype == object:
                    pred_seg_share = pr.dropna().value_counts(normalize=True).reindex(
                        ALL_MODE_NAMES, fill_value=0.0) * 100
                else:
                    _pr_num = pd.to_numeric(pr, errors='coerce').dropna().astype(int)
                    pred_seg_share = (_pr_num
                                      .map(MODE_LABELS).value_counts(normalize=True)
                                      .reindex(ALL_MODE_NAMES, fill_value=0.0) * 100)
            else:
                pred_seg_share = pd.Series(0.0, index=ALL_MODE_NAMES)
            seg_df = pd.DataFrame({'Obs_%': obs_seg_share.round(1),
                                   'Pred_%': pred_seg_share.round(1),
                                   'Diff_%': (pred_seg_share - obs_seg_share).round(1)})
            metrics['mode_share_by_segment'][seg] = seg_df
        except Exception as _ms_e:
            logger.debug(f"Per-segment mode share [{seg}]: {_ms_e}")

    # ── 8. Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STAGE 5 VALIDATION SUMMARY")
    print("=" * 65)

    # Destination zone accuracy
    print("\n── Destination zone accuracy ───────────────────────────────")
    for seg in SEG_PURPOSES:
        r = metrics['zone_rmse'].get(seg, float('nan'))
        c = metrics['zone_pearson'].get(seg, float('nan'))
        k = metrics['zone_kl'].get(seg, float('nan'))
        print(f"  {seg:10s}  RMSE={r:.4f}  Pearson r={c:.3f}  KL={k:.4f}")

    # Overall mode share
    print("\n── Overall mode share (%) ──────────────────────────────────")
    print(f"  {'Mode':10s}  {'Observed':>9s}  {'Predicted':>10s}  {'Diff':>7s}")
    _all_modes = ALL_MODE_NAMES
    for m in _all_modes:
        o = metrics['mode_share_obs'].get(m, 0.0)
        p = metrics['mode_share_pred'].get(m, 0.0)
        d = p - o
        flag = ' ◄' if abs(d) > 5.0 else ''
        print(f"  {m:10s}  {o:9.1f}  {p:10.1f}  {d:+7.1f}{flag}")
    _mode_mae_pp = metrics.get('mode_share_mae_pp', float('nan'))
    _mode_jsd = metrics.get('mode_share_jsd', float('nan'))
    print(f"  {'Mode Share MAE (pp)':22s}: {_mode_mae_pp:.3f}")
    print(f"  {'Mode Share JSD':22s}: {_mode_jsd:.6f}")

    # Per-segment mode shares
    for seg, seg_df in metrics.get('mode_share_by_segment', {}).items():
        print(f"\n── Mode share by segment: {seg.upper()} ─────────────────────")
        print(f"  {'Mode':10s}  {'Obs_%':>7s}  {'Pred_%':>8s}  {'Diff_%':>8s}")
        for mode in ALL_MODE_NAMES:
            if mode in seg_df.index:
                o_ = seg_df.loc[mode, 'Obs_%']
                p_ = seg_df.loc[mode, 'Pred_%']
                d_ = seg_df.loc[mode, 'Diff_%']
                flag = ' ◄' if abs(d_) > 5.0 else ''
                print(f"  {mode:10s}  {o_:7.1f}  {p_:8.1f}  {d_:+8.1f}{flag}")

    # Mean travel time
    print("\n── Mean travel time (min) ──────────────────────────────────")
    for seg in SEG_PURPOSES:
        o = metrics['mean_tt_obs'].get(seg, float('nan'))
        p = metrics['mean_tt_pred'].get(seg, float('nan'))
        print(f"  {seg:10s}  obs={o:.1f}  pred={p:.1f}")

    ddr = metrics.get('dist_decay_rmse', float('nan'))
    print(f"\nDistance-decay RMSE: {ddr:.5f}")

    # Explicit level summaries (ActivitySim-style coverage)
    _lvl = metrics.get('location_levels', {})
    if _lvl:
        print("\n── Location-choice level checks ───────────────────────────")
        _w = _lvl.get('workplace', {})
        if _w:
            print(f"  workplace: n={_w.get('n_activity_rows', 0)}  "
                  f"anchor_match={_w.get('anchor_match_rate', float('nan')):.1%}")
        _s = _lvl.get('school', {})
        if _s:
            print(f"  school:    n={_s.get('n_activity_rows', 0)}  "
                  f"anchor_match={_s.get('anchor_match_rate', float('nan')):.1%}")
        _t = _lvl.get('tour', {})
        if _t:
            print(f"  tour:      n={_t.get('n_activity_rows', 0)}  "
                  f"valid_dtaz={_t.get('valid_dtaz_rate', float('nan')):.1%}  "
                  f"unique_dtaz={_t.get('unique_dtaz', 0)}")
        _tr = _lvl.get('trip', {})
        if _tr:
            print(f"  trip:      n={_tr.get('n_trip_rows', 0)}  "
                  f"valid_od={_tr.get('valid_od_rate', float('nan')):.1%}  "
                  f"positive_tt={_tr.get('positive_travtime_rate', float('nan')):.1%}")
            if 'work_anchor_touch_rate' in _tr:
                print(f"             work_anchor_touch={_tr.get('work_anchor_touch_rate', float('nan')):.1%}")
            if 'school_anchor_touch_rate' in _tr:
                print(f"             school_anchor_touch={_tr.get('school_anchor_touch_rate', float('nan')):.1%}")

    # Top zone discrepancies
    for seg, disc in metrics.get('top_zone_discrepancies', {}).items():
        if disc is not None and not disc.empty:
            # Prefer zones with survey observations when ranking over-predictions,
            # since obs=0 zones inflate the diff artificially (no survey signal).
            _over_all   = disc[disc['diff'] > 0]
            _over_obs   = _over_all[_over_all['obs'] > 0].head(5)
            _over_unobs = _over_all[_over_all['obs'] == 0].head(5)
            top_over    = _over_obs if len(_over_obs) >= 5 else (
                pd.concat([_over_obs, _over_unobs]).head(5)
            )
            top_under = disc[disc['diff'] < 0].head(5)

            print(f"\n── Top 5 over-predicted zones [{seg}] ──────────────────────")
            print(f"  {'Zone':>6}  {'Obs':>5}  {'Pred':>7}  {'Diff':>7}")
            for z, row in top_over.iterrows():
                note = '  ← no survey obs' if int(row['obs']) == 0 else ''
                print(f"  {z:6d}  {int(row['obs']):5d}  {row['pred_scaled']:7.1f}  {row['diff']:+7.1f}{note}")
            if len(_over_unobs) > 0 and len(_over_obs) >= 5:
                print(f"  ({len(_over_unobs)} additional zone(s) over-predicted with 0 survey obs - see zone_discrepancy_{seg}.csv)")
            print(f"\n── Top 5 under-predicted zones [{seg}] ─────────────────────")
            print(f"  {'Zone':>6}  {'Obs':>5}  {'Pred':>7}  {'Diff':>7}")
            for z, row in top_under.iterrows():
                print(f"  {z:6d}  {int(row['obs']):5d}  {row['pred_scaled']:7.1f}  {row['diff']:+7.1f}")

    if not metrics['param_table'].empty:
        pt = metrics['param_table']
        bad = pt[~pt['significant'] & ~pt['holdfast']]
        print(f"\nInsignificant free parameters (|t|<1.96): {len(bad)}")
        for _, r in bad.iterrows():
            print(f"  [{r['segment']}/{r['model']}] {r['parameter']}  "
                  f"t={r['t_stat']:.2f}  se={r['std_err']:.3f}")
    print("=" * 65 + "\n")

    # ── 7. Save CSV ──────────────────────────────────────────────────────
    if save_path and not metrics['param_table'].empty:
        try:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            metrics['param_table'].to_csv(save_path, index=False)
            print(f"  Validation parameter table → {save_path}")
        except Exception as _e:
            logger.debug(f"compute_validation_metrics: save failed: {_e}")

    if save_path:
        try:
            _save_compact_kpi_summaries(metrics, save_path=save_path)
        except Exception as _kpi_e:
            logger.debug(f"compute_validation_metrics: compact KPI save failed: {_kpi_e}")

    return metrics


# ===========================================================================
# MULTI-LEVEL VALIDATION  (trip / tour / work assignment / edu assignment)
# ===========================================================================

def compute_multilevel_validation(
    obs_tours: pd.DataFrame,
    pred_plan: pd.DataFrame,
    landuse: pd.DataFrame = None,
    skim_dict: dict = None,
    save_dir: str = 'saved_data',
) -> dict:
    """
    Comprehensive multi-level validation comparing observed (HHTS) and
    predicted (synthetic) travel patterns.

    Produces four validation levels:
      1. Trip level:  OD pair accuracy, trip length frequency distribution
      2. Tour level:  DTAZ distribution per segment, primary destination
      3. Work assignment: workplace zone distribution (obs vs pred)
      4. Education assignment: school zone distribution by studying level

    Returns a dict of DataFrames and scalar metrics. Saves CSV tables to
    save_dir.
    """
    from scipy.stats import pearsonr
    eps = 1e-9
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    _has_rowtype = 'row_type' in pred_plan.columns
    _obs_purp_col = next(
        (c for c in ['primary_purpose', 'main_purpose', 'purpose']
         if c in obs_tours.columns), None
    )
    _obs_dtaz_col = 'DTAZ' if 'DTAZ' in obs_tours.columns else 'dtaz'
    _obs_mode_col = next((c for c in ['tour_mode', 'mode'] if c in obs_tours.columns), None)
    _obs_dist_col = next((c for c in ['distance', 'DIST', 'dist', 'tour_distance', 'dist_km'] if c in obs_tours.columns), None)
    _obs_tt_col = next((c for c in ['travtime', 'travel_time', 'TRAVTIME'] if c in obs_tours.columns), None)

    SEG_PURPOSES = {
        'work':    _WORK_PURPOSES,
        'edu':     _EDU_PURPOSES,
        'nonmand': _NONMAND_PURPOSES,
    }

    print("\n" + "=" * 70)
    print("MULTI-LEVEL VALIDATION REPORT")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 1: TRIP-LEVEL VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n── LEVEL 1: TRIP-LEVEL VALIDATION ─────────────────────────────")

    # Extract trips from pred_plan
    if _has_rowtype:
        pred_trips = pred_plan[pred_plan['row_type'] == 'trip'].copy()
    else:
        pred_trips = pred_plan.copy()

    # ── Build OD→skim-distance lookup from predicted trips ──────────────────
    # pred trips already carry `dist` from SOVTOLL_DIST__AM (km).  We extract
    # unique (OTAZ,DTAZ)→dist pairs and reuse the same skim for observed tours
    # so both TLFD curves use identical distance definitions.
    _pred_dist_col = next(
        (c for c in ['distance', 'DIST', 'dist', 'travdist'] if c in pred_trips.columns), None
    )
    _od_dist_lookup: dict = {}
    if (_pred_dist_col and 'OTAZ' in pred_trips.columns
            and 'DTAZ' in pred_trips.columns):
        _lu = (pred_trips[['OTAZ', 'DTAZ', _pred_dist_col]]
               .dropna()
               .drop_duplicates(['OTAZ', 'DTAZ']))
        _od_dist_lookup = {
            (int(o), int(d)): float(v)
            for o, d, v in _lu[['OTAZ', 'DTAZ', _pred_dist_col]].itertuples(index=False, name=None)
        }
        # symmetric: also add reverse pair with the same value
        for (o, d), v in list(_od_dist_lookup.items()):
            _od_dist_lookup.setdefault((d, o), v)

    # ── Deduplicate predicted trips to primary legs only ─────────────────────
    # Multi-stop tours produce multiple trip rows per tour (outbound + return +
    # stops).  Stop legs are short and inflate the 0-3 km bins.  Keep only the
    # longest-leg trip per (person_id, purpose) to approximate the primary OD.
    _pred_trips_primary = pred_trips
    if _pred_dist_col and 'person_id' in pred_trips.columns and 'purpose' in pred_trips.columns:
        _d_vals = pd.to_numeric(pred_trips[_pred_dist_col], errors='coerce')
        _pred_trips_primary = (
            pred_trips.assign(_d_tmp=_d_vals)
            .sort_values('_d_tmp', ascending=False)
            .drop_duplicates(subset=['person_id', 'purpose'], keep='first')
            .drop(columns=['_d_tmp'])
        )

    # Trip length frequency distribution (TLFD) by segment
    trip_tlfd = {}
    _dist_bins = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 999]
    _bin_labels = ['0-1', '1-2', '2-3', '3-5', '5-8', '8-10', '10-15',
                   '15-20', '20-30', '30-50', '50-100', '100+']

    for seg, purposes in SEG_PURPOSES.items():
        # ── Observed TLFD ─────────────────────────────────────────────────────
        # Prefer skim-derived OD distance (same source as pred) over survey
        # tour_distance which uses cumulative odometer readings and may have
        # data-quality issues (many near-zero entries → inflated 0-1 km band).
        obs_dist = pd.Series(dtype=float)
        _obs_subset = pd.DataFrame()
        if _obs_purp_col:
            _m_obs = obs_tours[_obs_purp_col].isin(purposes)
            _obs_subset = obs_tours.loc[_m_obs]
        elif not obs_tours.empty:
            _obs_subset = obs_tours

        if (not _obs_subset.empty and _od_dist_lookup
                and 'OTAZ' in _obs_subset.columns
                and _obs_dtaz_col in _obs_subset.columns):
            # Look up skim km for each observed OTAZ→DTAZ pair
            _o_vec = pd.to_numeric(_obs_subset['OTAZ'], errors='coerce').fillna(0).astype(int)
            _d_vec = pd.to_numeric(_obs_subset[_obs_dtaz_col], errors='coerce').fillna(0).astype(int)
            obs_dist = pd.Series(
                [_od_dist_lookup.get((int(o), int(d)), np.nan)
                 for o, d in zip(_o_vec, _d_vec)],
                dtype=float,
            ).dropna()
            _obs_dist_source = 'skim (OTAZ→DTAZ lookup)'
        elif not _obs_subset.empty and _obs_dist_col:
            obs_dist = pd.to_numeric(_obs_subset[_obs_dist_col], errors='coerce').dropna()
            _obs_dist_source = f'survey col {_obs_dist_col!r} (may differ from skim units)'
        else:
            _obs_dist_source = 'none'

        # Report intrazonal rate (OTAZ==DTAZ) - these always appear as 0-1 km
        _intrazone_pct = np.nan
        if (not _obs_subset.empty and 'OTAZ' in _obs_subset.columns
                and _obs_dtaz_col in _obs_subset.columns):
            _o2 = pd.to_numeric(_obs_subset['OTAZ'], errors='coerce')
            _d2 = pd.to_numeric(_obs_subset[_obs_dtaz_col], errors='coerce')
            _valid = _o2.notna() & _d2.notna()
            if _valid.any():
                _intrazone_pct = float((_o2[_valid] == _d2[_valid]).mean() * 100)

        # ── Predicted TLFD (primary legs only) ────────────────────────────────
        pred_dist = pd.Series(dtype=float)
        if 'purpose' in _pred_trips_primary.columns and _pred_dist_col:
            _m_pred = _pred_trips_primary['purpose'].isin(purposes)
            pred_dist = pd.to_numeric(
                _pred_trips_primary.loc[_m_pred, _pred_dist_col], errors='coerce'
            ).dropna()

        if not obs_dist.empty:
            obs_binned = pd.cut(obs_dist, bins=_dist_bins, labels=_bin_labels, right=False)
            obs_freq = obs_binned.value_counts(normalize=True).reindex(_bin_labels, fill_value=0)
        else:
            obs_freq = pd.Series(0.0, index=_bin_labels)

        if not pred_dist.empty:
            pred_binned = pd.cut(pred_dist, bins=_dist_bins, labels=_bin_labels, right=False)
            pred_freq = pred_binned.value_counts(normalize=True).reindex(_bin_labels, fill_value=0)
        else:
            pred_freq = pd.Series(0.0, index=_bin_labels)

        tlfd_df = pd.DataFrame({
            'distance_band': _bin_labels,
            'obs_share': (obs_freq.values * 100).round(2),
            'pred_share': (pred_freq.values * 100).round(2),
            'diff_pp': ((pred_freq.values - obs_freq.values) * 100).round(2),
        })
        trip_tlfd[seg] = tlfd_df

        # Print summary
        _rmse = np.sqrt(np.mean((obs_freq.values - pred_freq.values) ** 2))
        print(f"\n  {seg.upper()} trip length distribution (TLFD):")
        print(f"    n_obs={len(obs_dist):<6d}  n_pred={len(pred_dist):<6d}  RMSE={_rmse:.4f}")
        if np.isfinite(_intrazone_pct):
            print(f"    intrazonal obs={_intrazone_pct:.1f}%  dist_source={_obs_dist_source}")
        print(f"    {'Band':>10s}  {'Obs%':>6s}  {'Pred%':>7s}  {'Diff':>6s}")
        for _, row in tlfd_df.iterrows():
            flag = ' ◄' if abs(row['diff_pp']) > 3.0 else ''
            print(f"    {row['distance_band']:>10s}  {row['obs_share']:6.1f}  "
                  f"{row['pred_share']:7.1f}  {row['diff_pp']:+6.1f}{flag}")

        try:
            tlfd_df.to_csv(f'{save_dir}/trip_tlfd_{seg}.csv', index=False)
        except Exception:
            pass

    results['trip_tlfd'] = trip_tlfd

    # Trip OD summary: top 20 OD pairs by frequency
    trip_od = {}
    for seg, purposes in SEG_PURPOSES.items():
        obs_od = pd.Series(dtype=int)
        if _obs_purp_col and 'OTAZ' in obs_tours.columns and _obs_dtaz_col in obs_tours.columns:
            _m = obs_tours[_obs_purp_col].isin(purposes)
            _sub = obs_tours.loc[_m].copy()
            _sub['od_pair'] = (_sub['OTAZ'].astype(str) + '_' +
                               _sub[_obs_dtaz_col].astype(str))
            obs_od = _sub['od_pair'].value_counts().head(20)

        pred_od = pd.Series(dtype=int)
        if 'purpose' in pred_trips.columns and 'OTAZ' in pred_trips.columns and 'DTAZ' in pred_trips.columns:
            _m = pred_trips['purpose'].isin(purposes)
            _sub = pred_trips.loc[_m].copy()
            _sub['od_pair'] = (_sub['OTAZ'].astype(str) + '_' +
                               _sub['DTAZ'].astype(str))
            pred_od = _sub['od_pair'].value_counts()

        if not obs_od.empty:
            od_df = pd.DataFrame({
                'od_pair': obs_od.index,
                'obs_count': obs_od.values,
                'pred_count': [int(pred_od.get(p, 0)) for p in obs_od.index],
            })
            trip_od[seg] = od_df
            try:
                od_df.to_csv(f'{save_dir}/trip_top_od_{seg}.csv', index=False)
            except Exception:
                pass

    results['trip_od'] = trip_od

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 2: TOUR-LEVEL VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n── LEVEL 2: TOUR-LEVEL VALIDATION ────────────────────────────")

    # Extract activity/destination rows from pred_plan.
    # Keep only the PRIMARY destination per tour (one record per person×purpose)
    # so that intermediate stops don't dilute the zone distribution toward
    # nearby zones.  Primary = activity with the highest skim distance from
    # the person's origin, or failing that, the first activity record.
    if _has_rowtype:
        _all_acts = pred_plan[pred_plan['row_type'] == 'activity'].copy()
    else:
        _all_acts = pred_plan.copy()

    if ('person_id' in _all_acts.columns and 'purpose' in _all_acts.columns
            and 'DTAZ' in _all_acts.columns):
        if _od_dist_lookup and 'OTAZ' in _all_acts.columns:
            _all_acts['_pri_dist'] = [
                _od_dist_lookup.get((int(o), int(d)), 0.0)
                for o, d in zip(
                    pd.to_numeric(_all_acts['OTAZ'], errors='coerce').fillna(0).astype(int),
                    pd.to_numeric(_all_acts['DTAZ'],  errors='coerce').fillna(0).astype(int),
                )
            ]
            pred_acts = (
                _all_acts.sort_values('_pri_dist', ascending=False)
                .drop_duplicates(subset=['person_id', 'purpose'], keep='first')
                .drop(columns=['_pri_dist'])
            )
        else:
            # No distance lookup - keep first activity per person×purpose
            pred_acts = _all_acts.drop_duplicates(subset=['person_id', 'purpose'], keep='first')
    else:
        pred_acts = _all_acts

    tour_zone_accuracy = {}
    for seg, purposes in SEG_PURPOSES.items():
        # Observed destination zone distribution
        if _obs_purp_col and _obs_dtaz_col in obs_tours.columns:
            _m = obs_tours[_obs_purp_col].isin(purposes)
            obs_d = pd.to_numeric(
                obs_tours.loc[_m, _obs_dtaz_col], errors='coerce'
            ).dropna().astype(int)
        else:
            obs_d = pd.Series(dtype=int)

        # Predicted destination zone distribution
        pred_d = pd.Series(dtype=int)
        if 'purpose' in pred_acts.columns and 'DTAZ' in pred_acts.columns:
            _m = pred_acts['purpose'].isin(purposes)
            pred_d = pd.to_numeric(
                pred_acts.loc[_m, 'DTAZ'], errors='coerce'
            ).dropna().astype(int)

        if not obs_d.empty and not pred_d.empty:
            all_zones = sorted(set(obs_d) | set(pred_d))
            obs_cnt = obs_d.value_counts().reindex(all_zones, fill_value=0)
            pred_cnt = pred_d.value_counts().reindex(all_zones, fill_value=0)

            # Scale predicted to same total as observed
            scale = len(obs_d) / max(len(pred_d), 1)
            pred_scaled = (pred_cnt * scale).round(1)

            obs_share = obs_cnt / (obs_cnt.sum() + eps)
            pred_share = pred_cnt / (pred_cnt.sum() + eps)

            if obs_share.std() > 0 and pred_share.std() > 0:
                corr, _ = pearsonr(obs_share.values, pred_share.values)
            else:
                corr = float('nan')
            rmse = np.sqrt(np.mean((obs_share.values - pred_share.values) ** 2))

            zone_df = pd.DataFrame({
                'zone': all_zones,
                'obs_count': obs_cnt.values,
                'pred_count_scaled': pred_scaled.values,
                'obs_share': (obs_share.values * 100).round(3),
                'pred_share': (pred_share.values * 100).round(3),
                'diff_pp': ((pred_share.values - obs_share.values) * 100).round(3),
            }).sort_values('obs_count', ascending=False)

            tour_zone_accuracy[seg] = {
                'df': zone_df,
                'pearson_r': float(corr),
                'rmse': float(rmse),
                'n_obs': int(len(obs_d)),
                'n_pred': int(len(pred_d)),
                'obs_unique_zones': int(obs_d.nunique()),
                'pred_unique_zones': int(pred_d.nunique()),
            }

            print(f"\n  {seg.upper()} tour destination zones:")
            print(f"    n_obs={len(obs_d):<6d}  n_pred={len(pred_d):<6d}")
            print(f"    Pearson r={corr:.3f}  RMSE={rmse:.4f}")
            print(f"    obs_zones={obs_d.nunique()}  pred_zones={pred_d.nunique()}")
            # Top 10 most observed zones
            print(f"    {'Zone':>6s}  {'Obs':>5s}  {'Pred':>7s}  {'O%':>5s}  {'P%':>6s}  {'Diff':>6s}")
            for _, row in zone_df.head(10).iterrows():
                flag = ' ◄' if abs(row['diff_pp']) > 1.0 else ''
                print(f"    {int(row['zone']):6d}  {int(row['obs_count']):5d}  "
                      f"{row['pred_count_scaled']:7.1f}  {row['obs_share']:5.1f}  "
                      f"{row['pred_share']:6.1f}  {row['diff_pp']:+6.1f}{flag}")

            try:
                zone_df.to_csv(f'{save_dir}/tour_zone_accuracy_{seg}.csv', index=False)
            except Exception:
                pass

    results['tour_zone_accuracy'] = tour_zone_accuracy

    # Tour mode × segment cross-tab
    tour_mode_seg = {}
    MODE_LABELS = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
    ALL_MODE_NAMES = ['DA', 'SR', 'Walk', 'Bike', 'Transit']
    for seg, purposes in SEG_PURPOSES.items():
        obs_ms = pd.Series(0.0, index=ALL_MODE_NAMES)
        pred_ms = pd.Series(0.0, index=ALL_MODE_NAMES)

        if _obs_purp_col and _obs_mode_col:
            _m = obs_tours[_obs_purp_col].isin(purposes)
            _raw = pd.to_numeric(obs_tours.loc[_m, _obs_mode_col], errors='coerce').dropna().astype(int)
            obs_ms = _raw.map(MODE_LABELS).value_counts(normalize=True).reindex(
                ALL_MODE_NAMES, fill_value=0.0) * 100

        _pred_mc = next((c for c in ['mode', 'mode_name', 'pred_mode'] if c in pred_trips.columns), None)
        if 'purpose' in pred_trips.columns and _pred_mc:
            _m = pred_trips['purpose'].isin(purposes)
            _pc = pred_trips.loc[_m, _pred_mc]
            if _pc.dtype == object:
                pred_ms = _pc.dropna().value_counts(normalize=True).reindex(
                    ALL_MODE_NAMES, fill_value=0.0) * 100
            else:
                _pn = pd.to_numeric(_pc, errors='coerce').dropna().astype(int)
                pred_ms = _pn.map(MODE_LABELS).value_counts(normalize=True).reindex(
                    ALL_MODE_NAMES, fill_value=0.0) * 100

        ms_df = pd.DataFrame({
            'mode': ALL_MODE_NAMES,
            'obs_%': obs_ms.values.round(1),
            'pred_%': pred_ms.values.round(1),
            'diff_pp': (pred_ms.values - obs_ms.values).round(1),
        })
        tour_mode_seg[seg] = ms_df

    results['tour_mode_by_segment'] = tour_mode_seg

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 3: WORK ASSIGNMENT VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n── LEVEL 3: WORK ASSIGNMENT VALIDATION ───────────────────────")

    work_val = {}
    # Observed work destination distribution
    if _obs_purp_col and _obs_dtaz_col in obs_tours.columns:
        _m = obs_tours[_obs_purp_col].isin(_WORK_PURPOSES)
        obs_work_d = pd.to_numeric(
            obs_tours.loc[_m, _obs_dtaz_col], errors='coerce'
        ).dropna().astype(int)
    else:
        obs_work_d = pd.Series(dtype=int)

    # Predicted work destinations
    pred_work_d = pd.Series(dtype=int)
    if 'purpose' in pred_acts.columns and 'DTAZ' in pred_acts.columns:
        _m = pred_acts['purpose'].isin(_WORK_PURPOSES)
        pred_work_d = pd.to_numeric(
            pred_acts.loc[_m, 'DTAZ'], errors='coerce'
        ).dropna().astype(int)

    if not obs_work_d.empty:
        all_z = sorted(set(obs_work_d) | set(pred_work_d))
        obs_cnt = obs_work_d.value_counts().reindex(all_z, fill_value=0)
        pred_cnt = pred_work_d.value_counts().reindex(all_z, fill_value=0)
        scale = len(obs_work_d) / max(len(pred_work_d), 1)
        pred_scaled = (pred_cnt * scale).round(1)

        work_df = pd.DataFrame({
            'zone': all_z,
            'obs_workers': obs_cnt.values,
            'pred_workers_scaled': pred_scaled.values,
            'diff': (pred_scaled.values - obs_cnt.values).round(1),
        })

        # Add landuse info if available
        if landuse is not None:
            _lu_idx_col = 'ORDER' if 'ORDER' in landuse.columns else landuse.columns[0]
            _lu = landuse.set_index(_lu_idx_col)
            for _col in ['FPSEMPN', 'TOTEMP', 'commercial']:
                if _col in _lu.columns:
                    work_df[_col] = work_df['zone'].map(
                        pd.to_numeric(_lu[_col], errors='coerce').fillna(0).to_dict()
                    ).fillna(0)

        work_df = work_df.sort_values('obs_workers', ascending=False)
        work_val['zone_distribution'] = work_df

        obs_share = obs_cnt / (obs_cnt.sum() + eps)
        pred_share = pred_cnt / (pred_cnt.sum() + eps)
        if obs_share.std() > 0 and pred_share.std() > 0:
            corr, _ = pearsonr(obs_share.values, pred_share.values)
        else:
            corr = float('nan')
        rmse = np.sqrt(np.mean((obs_share.values - pred_share.values) ** 2))
        work_val['pearson_r'] = float(corr)
        work_val['rmse'] = float(rmse)

        print(f"  Work tours: n_obs={len(obs_work_d)}  n_pred={len(pred_work_d)}")
        print(f"  Pearson r={corr:.3f}  RMSE={rmse:.4f}")
        print(f"\n  Top 15 work destination zones:")
        print(f"  {'Zone':>6s}  {'Obs':>5s}  {'Pred':>7s}  {'Diff':>7s}  {'TOTEMP':>7s}  {'FPSEMPN':>7s}")
        for _, row in work_df.head(15).iterrows():
            _te = int(row.get('TOTEMP', 0))
            _fp = int(row.get('FPSEMPN', 0))
            flag = ' ◄' if abs(row['diff']) > 10 else ''
            print(f"  {int(row['zone']):6d}  {int(row['obs_workers']):5d}  "
                  f"{row['pred_workers_scaled']:7.1f}  {row['diff']:+7.1f}  "
                  f"{_te:7d}  {_fp:7d}{flag}")

        try:
            work_df.to_csv(f'{save_dir}/work_assignment_validation.csv', index=False)
        except Exception:
            pass

    results['work_assignment'] = work_val

    # ══════════════════════════════════════════════════════════════════════
    # LEVEL 4: EDUCATION ASSIGNMENT VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n\n── LEVEL 4: EDUCATION ASSIGNMENT VALIDATION ──────────────────")

    edu_val = {}
    _obs_study_col = next((c for c in ['STUDYING', 'studying'] if c in obs_tours.columns), None)

    # Overall edu destination distribution
    if _obs_purp_col and _obs_dtaz_col in obs_tours.columns:
        _m = obs_tours[_obs_purp_col].isin(_EDU_PURPOSES)
        obs_edu_d = pd.to_numeric(
            obs_tours.loc[_m, _obs_dtaz_col], errors='coerce'
        ).dropna().astype(int)
    else:
        obs_edu_d = pd.Series(dtype=int)

    pred_edu_d = pd.Series(dtype=int)
    if 'purpose' in pred_acts.columns and 'DTAZ' in pred_acts.columns:
        _m = pred_acts['purpose'].isin(_EDU_PURPOSES)
        pred_edu_d = pd.to_numeric(
            pred_acts.loc[_m, 'DTAZ'], errors='coerce'
        ).dropna().astype(int)

    if not obs_edu_d.empty:
        all_z = sorted(set(obs_edu_d) | set(pred_edu_d))
        obs_cnt = obs_edu_d.value_counts().reindex(all_z, fill_value=0)
        pred_cnt = pred_edu_d.value_counts().reindex(all_z, fill_value=0)
        scale = len(obs_edu_d) / max(len(pred_edu_d), 1)
        pred_scaled = (pred_cnt * scale).round(1)

        edu_df = pd.DataFrame({
            'zone': all_z,
            'obs_students': obs_cnt.values,
            'pred_students_scaled': pred_scaled.values,
            'diff': (pred_scaled.values - obs_cnt.values).round(1),
        })

        if landuse is not None:
            _lu_idx_col = 'ORDER' if 'ORDER' in landuse.columns else landuse.columns[0]
            _lu = landuse.set_index(_lu_idx_col)
            for _col in ['Primary', 'Secondary', 'Education', 'HSENROLL']:
                if _col in _lu.columns:
                    edu_df[_col] = edu_df['zone'].map(
                        pd.to_numeric(_lu[_col], errors='coerce').fillna(0).to_dict()
                    ).fillna(0)

        edu_df = edu_df.sort_values('obs_students', ascending=False)
        edu_val['zone_distribution'] = edu_df

        obs_share = obs_cnt / (obs_cnt.sum() + eps)
        pred_share = pred_cnt / (pred_cnt.sum() + eps)
        if obs_share.std() > 0 and pred_share.std() > 0:
            corr, _ = pearsonr(obs_share.values, pred_share.values)
        else:
            corr = float('nan')
        rmse = np.sqrt(np.mean((obs_share.values - pred_share.values) ** 2))
        edu_val['pearson_r'] = float(corr)
        edu_val['rmse'] = float(rmse)

        print(f"  Edu tours (all): n_obs={len(obs_edu_d)}  n_pred={len(pred_edu_d)}")
        print(f"  Pearson r={corr:.3f}  RMSE={rmse:.4f}")

        try:
            edu_df.to_csv(f'{save_dir}/edu_assignment_validation.csv', index=False)
        except Exception:
            pass

    # Per studying-level breakdown (primary, secondary, post-secondary)
    _STUDY_LEVELS = {
        'primary': [1],
        'secondary': [2],
        'post_secondary': [3, 4],
    }
    _STUDY_LU_MAP = {
        'primary': 'Primary',
        'secondary': 'Secondary',
        'post_secondary': 'Education',
    }

    for level_name, level_codes in _STUDY_LEVELS.items():
        obs_level_d = pd.Series(dtype=int)
        if _obs_study_col and _obs_dtaz_col in obs_tours.columns:
            _study = pd.to_numeric(obs_tours[_obs_study_col], errors='coerce').fillna(0).astype(int)
            _m = _study.isin(level_codes)
            if _obs_purp_col:
                _m = _m & obs_tours[_obs_purp_col].isin(_EDU_PURPOSES)
            obs_level_d = pd.to_numeric(
                obs_tours.loc[_m, _obs_dtaz_col], errors='coerce'
            ).dropna().astype(int)

        pred_level_d = pd.Series(dtype=int)
        _pred_study_col = next((c for c in ['STUDYING', 'studying'] if c in pred_acts.columns), None)
        if _pred_study_col and 'DTAZ' in pred_acts.columns:
            _study_p = pd.to_numeric(pred_acts[_pred_study_col], errors='coerce').fillna(0).astype(int)
            _m = _study_p.isin(level_codes)
            if 'purpose' in pred_acts.columns:
                _m = _m & pred_acts['purpose'].isin(_EDU_PURPOSES)
            pred_level_d = pd.to_numeric(
                pred_acts.loc[_m, 'DTAZ'], errors='coerce'
            ).dropna().astype(int)

        if not obs_level_d.empty:
            all_z = sorted(set(obs_level_d) | set(pred_level_d))
            obs_cnt = obs_level_d.value_counts().reindex(all_z, fill_value=0)
            pred_cnt = pred_level_d.value_counts().reindex(all_z, fill_value=0)
            scale = len(obs_level_d) / max(len(pred_level_d), 1)
            pred_scaled = (pred_cnt * scale).round(1)

            level_df = pd.DataFrame({
                'zone': all_z,
                'obs_students': obs_cnt.values,
                'pred_students_scaled': pred_scaled.values,
                'diff': (pred_scaled.values - obs_cnt.values).round(1),
            })

            # Add relevant landuse column
            _lu_col = _STUDY_LU_MAP.get(level_name)
            if landuse is not None and _lu_col:
                _lu_idx_col = 'ORDER' if 'ORDER' in landuse.columns else landuse.columns[0]
                _lu = landuse.set_index(_lu_idx_col)
                if _lu_col in _lu.columns:
                    level_df[_lu_col] = level_df['zone'].map(
                        pd.to_numeric(_lu[_lu_col], errors='coerce').fillna(0).to_dict()
                    ).fillna(0)

            level_df = level_df.sort_values('obs_students', ascending=False)
            edu_val[level_name] = level_df

            obs_share = obs_cnt / (obs_cnt.sum() + eps)
            pred_share = pred_cnt / (pred_cnt.sum() + eps)
            if obs_share.std() > 0 and pred_share.std() > 0:
                corr_l, _ = pearsonr(obs_share.values, pred_share.values)
            else:
                corr_l = float('nan')
            rmse_l = np.sqrt(np.mean((obs_share.values - pred_share.values) ** 2))

            print(f"\n  {level_name.upper()} students: "
                  f"n_obs={len(obs_level_d)}  n_pred={len(pred_level_d)}")
            print(f"    Pearson r={corr_l:.3f}  RMSE={rmse_l:.4f}")
            # Top 10 zones
            print(f"    {'Zone':>6s}  {'Obs':>5s}  {'Pred':>7s}  {'Diff':>7s}  "
                  f"{_lu_col or 'LU':>8s}")
            for _, row in level_df.head(10).iterrows():
                _lv = int(row.get(_lu_col, 0)) if _lu_col and _lu_col in row.index else 0
                print(f"    {int(row['zone']):6d}  {int(row['obs_students']):5d}  "
                      f"{row['pred_students_scaled']:7.1f}  {row['diff']:+7.1f}  "
                      f"{_lv:8d}")

            try:
                level_df.to_csv(f'{save_dir}/edu_assignment_{level_name}.csv', index=False)
            except Exception:
                pass

    results['edu_assignment'] = edu_val

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════
    summary_rows = []

    # Tour-level metrics
    for seg, info in tour_zone_accuracy.items():
        summary_rows.append({
            'level': 'tour',
            'segment': seg,
            'n_obs': info.get('n_obs', 0),
            'n_pred': info.get('n_pred', 0),
            'pearson_r': info.get('pearson_r', np.nan),
            'rmse': info.get('rmse', np.nan),
        })

    # Work assignment
    if work_val.get('pearson_r') is not None:
        summary_rows.append({
            'level': 'work_assignment',
            'segment': 'work',
            'n_obs': int(len(obs_work_d)) if not obs_work_d.empty else 0,
            'n_pred': int(len(pred_work_d)) if not pred_work_d.empty else 0,
            'pearson_r': work_val.get('pearson_r', np.nan),
            'rmse': work_val.get('rmse', np.nan),
        })

    # Edu assignment
    if edu_val.get('pearson_r') is not None:
        summary_rows.append({
            'level': 'edu_assignment',
            'segment': 'edu_all',
            'n_obs': int(len(obs_edu_d)) if not obs_edu_d.empty else 0,
            'n_pred': int(len(pred_edu_d)) if not pred_edu_d.empty else 0,
            'pearson_r': edu_val.get('pearson_r', np.nan),
            'rmse': edu_val.get('rmse', np.nan),
        })

    summary_df = pd.DataFrame(summary_rows)
    results['summary'] = summary_df

    print("\n\n── VALIDATION SUMMARY TABLE ───────────────────────────────────")
    print(f"  {'Level':<20s}  {'Segment':<12s}  {'n_obs':>6s}  {'n_pred':>7s}  "
          f"{'Pearson':>8s}  {'RMSE':>7s}")
    print(f"  {'─'*65}")
    for _, row in summary_df.iterrows():
        print(f"  {row['level']:<20s}  {row['segment']:<12s}  {int(row['n_obs']):6d}  "
              f"{int(row['n_pred']):7d}  {row['pearson_r']:8.3f}  {row['rmse']:7.4f}")

    try:
        summary_df.to_csv(f'{save_dir}/multilevel_validation_summary.csv', index=False)
        print(f"\n  Summary saved → {save_dir}/multilevel_validation_summary.csv")
    except Exception:
        pass

    print("=" * 70 + "\n")

    return results
    """Create a one-row compact KPI table for location-choice levels."""
    lvl = metrics.get('location_levels', {}) if isinstance(metrics, dict) else {}
    w = lvl.get('workplace', {}) if isinstance(lvl.get('workplace', {}), dict) else {}
    s = lvl.get('school', {}) if isinstance(lvl.get('school', {}), dict) else {}
    t = lvl.get('tour', {}) if isinstance(lvl.get('tour', {}), dict) else {}
    tr = lvl.get('trip', {}) if isinstance(lvl.get('trip', {}), dict) else {}

    row = {
        'run_label': run_label,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'work_anchor_match_rate': w.get('anchor_match_rate', np.nan),
        'work_n': w.get('n_activity_rows', np.nan),
        'school_anchor_match_rate': s.get('anchor_match_rate', np.nan),
        'school_n': s.get('n_activity_rows', np.nan),
        'tour_valid_dtaz_rate': t.get('valid_dtaz_rate', np.nan),
        'tour_unique_dtaz': t.get('unique_dtaz', np.nan),
        'tour_n': t.get('n_activity_rows', np.nan),
        'trip_valid_od_rate': tr.get('valid_od_rate', np.nan),
        'trip_positive_travtime_rate': tr.get('positive_travtime_rate', np.nan),
        'trip_work_anchor_touch_rate': tr.get('work_anchor_touch_rate', np.nan),
        'trip_school_anchor_touch_rate': tr.get('school_anchor_touch_rate', np.nan),
        'trip_n': tr.get('n_trip_rows', np.nan),
    }
    return pd.DataFrame([row])


def _write_compact_kpi_html(df: pd.DataFrame, html_path: str, title: str):
    """Write a minimal HTML summary for compact level KPIs."""
    os.makedirs(os.path.dirname(html_path) or '.', exist_ok=True)
    view = df.copy()

    pct_cols = [
        'work_anchor_match_rate', 'school_anchor_match_rate',
        'tour_valid_dtaz_rate', 'trip_valid_od_rate',
        'trip_positive_travtime_rate', 'trip_work_anchor_touch_rate',
        'trip_school_anchor_touch_rate',
    ]
    for c in pct_cols:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors='coerce').map(
                lambda v: f"{v:.1%}" if pd.notna(v) else ''
            )

    html_table = view.to_html(index=False, border=0)
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
    th {{ background: #1565c0; color: #fff; text-align: left; }}
    tr:nth-child(even) td {{ background: #f8f9fb; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Compact location-choice validation KPIs for workplace, school, tour-level, and trip-level checks.</p>
  {html_table}
</body>
</html>"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def _save_compact_kpi_summaries(metrics: dict, save_path: str):
    """
    Save compact KPI summaries for quick baseline/GA comparison.

    Outputs:
      - run-local CSV/HTML in the same directory as save_path
      - cross-run aggregate CSV/HTML one folder above (e.g. runs/)
    """
    run_dir = os.path.dirname(save_path) or '.'
    run_label = os.path.basename(run_dir.rstrip('/\\')) or 'run'

    run_csv = os.path.join(run_dir, 'stage5_level_kpis.csv')
    run_html = os.path.join(run_dir, 'stage5_level_kpis.html')
    row_df = _compact_level_kpi_row(metrics, run_label=run_label)
    row_df.to_csv(run_csv, index=False)
    _write_compact_kpi_html(row_df, run_html, title='Stage 5 Level KPIs (This Run)')

    # Build/refresh aggregate summary across sibling run folders.
    parent_dir = os.path.dirname(run_dir.rstrip('/\\')) or '.'
    agg_csv = os.path.join(parent_dir, 'stage5_level_kpis_all.csv')
    agg_html = os.path.join(parent_dir, 'stage5_level_kpis_all.html')

    if os.path.isfile(agg_csv):
        try:
            agg_df = pd.read_csv(agg_csv)
        except Exception:
            agg_df = pd.DataFrame()
    else:
        agg_df = pd.DataFrame()

    if not agg_df.empty and 'run_label' in agg_df.columns:
        agg_df = agg_df[agg_df['run_label'].astype(str) != str(run_label)]
    agg_df = pd.concat([agg_df, row_df], ignore_index=True)
    if 'run_label' in agg_df.columns:
        agg_df = agg_df.sort_values('run_label').reset_index(drop=True)

    agg_df.to_csv(agg_csv, index=False)
    _write_compact_kpi_html(agg_df, agg_html, title='Stage 5 Level KPIs (All Runs)')

    print(f"  Compact KPI summary → {run_csv}")
    print(f"  Compact KPI HTML    → {run_html}")
    print(f"  Cross-run KPI CSV   → {agg_csv}")
    print(f"  Cross-run KPI HTML  → {agg_html}")


def _compute_model_fit_stats(models: dict) -> pd.DataFrame:
    """
    Extract LL-null, LL-final, n_obs, n_params, McFadden ρ², AIC, BIC
    for every estimated mode and destination model.

    Returns a DataFrame indexed by (segment, model_type).
    """
    rows = []
    for seg, res in models.items():
        for mtype in ('mode_model', 'dest_model'):
            mdl = res.get(mtype)
            if mdl is None:
                continue
            # LL final
            ll_final = getattr(mdl, '_cached_ll', None)
            if ll_final is None:
                try:
                    ll_final = float(mdl.loglike())
                except Exception:
                    ll_final = float('nan')

            # Parameter frame
            pf = None
            try:
                pf = (mdl._final_pf if hasattr(mdl, '_final_pf')
                      and isinstance(mdl._final_pf, pd.DataFrame) else mdl.pf)
            except Exception:
                pass

            n_free = int((~pf['holdfast'].fillna(False)).sum()) if pf is not None else float('nan')
            n_obs  = res.get('n_tours', float('nan'))

            # LL null (constants-only)
            ll_null = getattr(mdl, '_cached_ll_null', float('nan'))
            if np.isnan(ll_null):
                try:
                    ll_null = float(mdl.loglike_null())
                except Exception:
                    pass

            rho2 = float(1.0 - ll_final / ll_null) if (
                np.isfinite(ll_null) and ll_null != 0 and np.isfinite(ll_final)
            ) else float('nan')
            aic = float(-2 * ll_final + 2 * n_free) if np.isfinite(ll_final) else float('nan')
            bic = float(-2 * ll_final + n_free * np.log(max(n_obs, 1))) if (
                np.isfinite(ll_final) and np.isfinite(float(n_obs))
            ) else float('nan')

            rows.append({
                'segment':    seg,
                'model':      mtype.replace('_model', ''),
                'll_null':    round(ll_null,  2),
                'll_final':   round(ll_final, 2),
                'rho2':       round(rho2,     4),
                'aic':        round(aic,      1),
                'bic':        round(bic,      1),
                'n_obs':      n_obs,
                'n_free_params': n_free,
            })
    return pd.DataFrame(rows)


def _check_behavioural_signs(models: dict) -> list:
    """
    Check estimated coefficients against expected behavioural signs.

    Returns a list of dicts with keys:
        segment, model, parameter, value, expected_sign, pass, note
    """
    EXPECTED = {
        # destination model
        'c_dist':          ('<', 0,  'Distance deters destination'),
        'c_log_dist':      ('<', 0,  'Log-distance deters destination'),
        'c_size_term':     ('>', 0,  'Employment/size attracts'),
        'c_logsum':        ('>', 0,  'Better mode access attracts'),
        'c_aggl':          ('>', 0,  'Agglomeration attracts (clustering)'),
        'c_comp':          ('<', 0,  'Competition deters (or ≈0)'),
        'c_propensity_dist': ('>', 0, 'Mobile persons less distance-deterred'),
        # mode model
        'c_ivt_auto':      ('<', 0,  'Auto IVT deters driving'),
        'c_ivt_transit':   ('<', 0,  'Transit IVT deters transit'),
        'c_walk_time':     ('<', 0,  'Walk time deters walking'),
        'c_bike_time':     ('<', 0,  'Bike time deters cycling'),
        'c_veh_da':        ('>', 0,  'Vehicle ownership promotes DA'),
        'c_noveh_transit': ('>', 0,  'Zero-car HH shifts to transit'),
        'c_incp_transit':  ('<', 0,  'Higher income → less transit'),
        'c_incp_da':       ('>', 0,  'Higher income → more DA'),
        'ASC_Walk':        ('<', 0,  'Walk baseline below DA'),
        'ASC_Bike':        ('<', 0,  'Bike baseline below DA'),
        'ASC_Transit':     ('<', 0,  'Transit baseline below DA'),
    }
    results = []
    for seg, res in models.items():
        for mtype in ('mode_model', 'dest_model'):
            mdl = res.get(mtype)
            if mdl is None:
                continue
            coef = (res.get('dest_coef' if mtype == 'dest_model' else 'mode_coef')
                    or {})
            for pname, (op, threshold, note) in EXPECTED.items():
                if pname not in coef:
                    continue
                val = float(coef[pname])
                if np.isnan(val):
                    ok = None
                elif op == '<':
                    ok = val < threshold
                else:
                    ok = val > threshold
                results.append({
                    'segment':       seg,
                    'model':         mtype.replace('_model', ''),
                    'parameter':     pname,
                    'value':         round(val, 4),
                    'expected':      f'{op} {threshold}',
                    'pass':          ok,
                    'note':          note,
                })
    return results


def _compute_acdc_effects(models: dict, landuse: pd.DataFrame) -> dict:
    """
    Analyse how agglomeration (c_aggl × acdc_aggl) and competition
    (c_comp × acdc_comp) terms influence destination probabilities.

    For each segment this function computes:
      - Estimated coefficients c_aggl and c_comp (and t-stats if available)
      - Distribution of acdc_aggl and acdc_comp across zones
      - Marginal utility contribution at mean, p10, p90 zone
      - Probability ratio: P(top-decile aggl zone) / P(bottom-decile aggl zone)
        holding all other utilities equal
      - Top-10 zones by agglomeration score
      - Net ACDC effect = c_aggl * acdc_aggl + c_comp * acdc_comp per zone

    Returns a dict:
        {segment: {
            'c_aggl': float, 'c_comp': float,
            't_aggl': float, 't_comp': float,
            'aggl_mean': float, 'aggl_sd': float,
            'comp_mean': float, 'comp_sd': float,
            'utility_aggl_mean': float,   # c_aggl × mean(acdc_aggl)
            'utility_aggl_p90':  float,   # c_aggl × p90(acdc_aggl)
            'prob_ratio_aggl':   float,   # P(top-decile) / P(bottom-decile), ceteris paribus
            'top10_zones':       DataFrame,
            'zone_effects':      DataFrame (zone_id, acdc_aggl, acdc_comp, net_acdc_util),
        }}
    """
    results = {}
    lu = landuse.copy()
    if lu.index.name == 'ORDER':
        lu = lu.reset_index()
    zone_col = next((c for c in ['ORDER', 'zone_id'] if c in lu.columns), lu.columns[0])

    acdc_aggl = pd.to_numeric(lu.get('acdc_aggl', pd.Series(0.0, index=lu.index)), errors='coerce').fillna(0.0)
    acdc_comp = pd.to_numeric(lu.get('acdc_comp', pd.Series(0.0, index=lu.index)), errors='coerce').fillna(0.0)
    zone_ids  = pd.to_numeric(lu[zone_col], errors='coerce').fillna(0).astype(int)

    for seg, res in models.items():
        dest_mdl  = res.get('dest_model')
        dest_coef = res.get('dest_coef') or {}
        if dest_mdl is None and not dest_coef:
            continue

        c_aggl = float(dest_coef.get('c_aggl', 0.0))
        c_comp = float(dest_coef.get('c_comp', 0.0))

        # t-stats from _final_pf
        t_aggl = t_comp = float('nan')
        try:
            pf = (dest_mdl._final_pf if hasattr(dest_mdl, '_final_pf')
                  and isinstance(dest_mdl._final_pf, pd.DataFrame) else dest_mdl.pf)
            if 'c_aggl' in pf.index:
                t_aggl = float(pf.loc['c_aggl', 't_stat'] if 't_stat' in pf.columns
                               else pf.loc['c_aggl', 'value'] / max(float(pf.loc['c_aggl', 'std_err']), 1e-9))
            if 'c_comp' in pf.index:
                t_comp = float(pf.loc['c_comp', 't_stat'] if 't_stat' in pf.columns
                               else pf.loc['c_comp', 'value'] / max(float(pf.loc['c_comp', 'std_err']), 1e-9))
        except Exception:
            pass

        # Per-zone utility contributions
        util_aggl = c_aggl * acdc_aggl.values
        util_comp = c_comp * acdc_comp.values
        net_acdc  = util_aggl + util_comp

        zone_effects = pd.DataFrame({
            'zone_id':      zone_ids.values,
            'acdc_aggl':    acdc_aggl.values.round(4),
            'acdc_comp':    acdc_comp.values.round(4),
            'util_aggl':    util_aggl.round(4),
            'util_comp':    util_comp.round(4),
            'net_acdc_util': net_acdc.round(4),
        }).sort_values('net_acdc_util', ascending=False).reset_index(drop=True)

        # Probability ratio: top vs bottom decile aggl zone (ceteris paribus)
        # P_j ∝ exp(V_j); ratio = exp(V_top - V_bottom)
        if len(util_aggl) >= 10:
            p90_util = float(np.percentile(util_aggl[np.isfinite(util_aggl)], 90))
            p10_util = float(np.percentile(util_aggl[np.isfinite(util_aggl)], 10))
            prob_ratio = float(np.exp(p90_util - p10_util))
        else:
            prob_ratio = float('nan')

        results[seg] = {
            'c_aggl':           round(c_aggl, 4),
            'c_comp':           round(c_comp, 4),
            't_aggl':           round(t_aggl, 2) if np.isfinite(t_aggl) else float('nan'),
            't_comp':           round(t_comp, 2) if np.isfinite(t_comp) else float('nan'),
            'aggl_mean':        round(float(acdc_aggl.mean()), 4),
            'aggl_sd':          round(float(acdc_aggl.std()),  4),
            'comp_mean':        round(float(acdc_comp.mean()), 4),
            'comp_sd':          round(float(acdc_comp.std()),  4),
            'utility_aggl_mean': round(float(c_aggl * acdc_aggl.mean()), 4),
            'utility_aggl_p90':  round(p90_util if len(util_aggl) >= 10 else float('nan'), 4),
            'utility_aggl_p10':  round(p10_util if len(util_aggl) >= 10 else float('nan'), 4),
            'prob_ratio_aggl':   round(prob_ratio, 2),
            'top10_zones':       zone_effects.head(10),
            'zone_effects':      zone_effects,
        }
    return results


def _compute_tld_metrics(obs_tours: pd.DataFrame,
                         pred_plan: pd.DataFrame) -> dict:
    """
    Trip-length distribution (TLD) comparison: observed HHTS vs synthetic.

    Returns a dict with:
      obs_quantiles, pred_quantiles  - p10/p25/p50/p75/p90/mean
      band_share_obs, band_share_pred - % in <5, 5-15, 15-30, 30-60, 60+ km
    """
    dist_col_obs  = next((c for c in ['DIST', 'dist', 'SOVTOLL_DIST__AM', 'tour_distance']
                          if c in obs_tours.columns), None)
    dist_col_pred = next((c for c in ['dist', 'DIST', 'SOVTOLL_DIST__AM']
                          if c in pred_plan.columns), None)

    def _get_dists(df, col, row_type_filter='trip'):
        if col is None:
            return pd.Series(dtype=float)
        d = df
        if 'row_type' in df.columns:
            d = df[df['row_type'] == row_type_filter]
        return pd.to_numeric(d[col], errors='coerce').dropna()

    obs_d  = _get_dists(obs_tours, dist_col_obs,  'trip')
    pred_d = _get_dists(pred_plan, dist_col_pred, 'trip')

    BANDS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 999)]
    BAND_LABELS = ['<5km', '5-15km', '15-30km', '30-60km', '60+km']

    def _quantiles(s):
        if len(s) == 0:
            return {q: float('nan') for q in ['p10', 'p25', 'p50', 'p75', 'p90', 'mean']}
        return {
            'p10':  round(float(np.percentile(s, 10)), 2),
            'p25':  round(float(np.percentile(s, 25)), 2),
            'p50':  round(float(np.percentile(s, 50)), 2),
            'p75':  round(float(np.percentile(s, 75)), 2),
            'p90':  round(float(np.percentile(s, 90)), 2),
            'mean': round(float(s.mean()), 2),
        }

    def _band_shares(s):
        if len(s) == 0:
            return {lbl: float('nan') for lbl in BAND_LABELS}
        total = len(s)
        return {lbl: round(100.0 * ((s >= lo) & (s < hi)).sum() / total, 1)
                for lbl, (lo, hi) in zip(BAND_LABELS, BANDS)}

    return {
        'obs_quantiles':    _quantiles(obs_d),
        'pred_quantiles':   _quantiles(pred_d),
        'band_share_obs':   _band_shares(obs_d),
        'band_share_pred':  _band_shares(pred_d),
        'n_obs':            len(obs_d),
        'n_pred':           len(pred_d),
    }


def print_stage5_dashboard(
    models: dict,
    metrics: dict,
    landuse: pd.DataFrame,
    obs_tours: pd.DataFrame = None,
    pred_plan: pd.DataFrame = None,
    save_dir: str = 'saved_data',
) -> dict:
    """
    Print and save a comprehensive Stage 5 validation dashboard.

    Sections
    --------
    1. Model fit - LL, McFadden ρ², AIC, BIC for every model
    2. Behavioural sign checks - expected vs estimated coefficient signs
    3. ACDC agglomeration/competition effects - zone-level utility contributions,
       probability ratios, top-10 most-agglomerated zones
    4. Destination zone accuracy - RMSE, Pearson r, KL divergence
    5. Mode share comparison - observed vs predicted with segment breakdown
    6. Trip-length distribution - quantiles and distance-band shares
    7. Insignificant free parameters

    Saves:
      saved_data/dashboard_model_fit.csv
      saved_data/dashboard_sign_checks.csv
      saved_data/dashboard_acdc_{segment}.csv
      saved_data/dashboard_tld.csv

    Returns the combined dict of all computed diagnostic tables.
    """
    os.makedirs(save_dir, exist_ok=True)
    W = 72   # console width
    SEP = '─' * W

    def _hdr(title):
        print(f'\n┌{SEP}┐')
        print(f'│  {title:<{W-2}}│')
        print(f'└{SEP}┘')

    def _row(label, *vals, width=12):
        parts = f'  {label:<26s}'
        for v in vals:
            parts += f'{str(v):>{width}}'
        print(parts)

    print(f'\n{"═"*W}')
    print(f'  STAGE 5 DASHBOARD - Destination + Mode Choice')
    print(f'{"═"*W}')

    # ── 1. Model fit ──────────────────────────────────────────────────────
    _hdr('1. MODEL FIT STATISTICS')
    fit_df = _compute_model_fit_stats(models)
    if not fit_df.empty:
        print(f'  {"Segment":<10} {"Model":<6} {"LL-null":>10} {"LL-final":>10}'
              f' {"ρ²":>7} {"AIC":>10} {"BIC":>10} {"n_obs":>7} {"n_free":>7}')
        print(f'  {"-"*70}')
        for _, r in fit_df.iterrows():
            rho2_str = f'{r.rho2:.4f}' if np.isfinite(r.rho2) else '   n/a'
            print(f'  {r.segment:<10} {r.model:<6} {r.ll_null:>10.1f} {r.ll_final:>10.1f}'
                  f' {rho2_str:>7} {r.aic:>10.1f} {r.bic:>10.1f}'
                  f' {int(r.n_obs) if np.isfinite(float(r.n_obs)) else "?":>7}'
                  f' {int(r.n_free_params) if np.isfinite(float(r.n_free_params)) else "?":>7}')
        try:
            fit_df.to_csv(f'{save_dir}/dashboard_model_fit.csv', index=False)
        except Exception:
            pass

    # ── 2. Behavioural sign checks ────────────────────────────────────────
    _hdr('2. BEHAVIOURAL SIGN CHECKS')
    sign_checks = _check_behavioural_signs(models)
    if sign_checks:
        fails = [c for c in sign_checks if c['pass'] is False]
        passes = [c for c in sign_checks if c['pass'] is True]
        unknowns = [c for c in sign_checks if c['pass'] is None]
        print(f'  Checked {len(sign_checks)} parameters:  '
              f'✓ {len(passes)} pass   ✗ {len(fails)} fail   ? {len(unknowns)} NaN')
        if fails:
            print(f'\n  FAILURES (unexpected coefficient signs):')
            print(f'  {"Seg":<8} {"Model":<6} {"Parameter":<22} {"Value":>8} {"Expected":>10}  Note')
            print(f'  {"-"*70}')
            for c in fails:
                print(f'  {c["segment"]:<8} {c["model"]:<6} {c["parameter"]:<22}'
                      f' {c["value"]:>8.4f} {c["expected"]:>10}  {c["note"]}')
        else:
            print('  All checked parameters have the expected sign ✓')
        try:
            pd.DataFrame(sign_checks).to_csv(f'{save_dir}/dashboard_sign_checks.csv', index=False)
        except Exception:
            pass

    # ── 3. ACDC agglomeration / competition effects ───────────────────────
    _hdr('3. ACDC - AGGLOMERATION & COMPETITION EFFECTS')
    acdc = _compute_acdc_effects(models, landuse)
    for seg, ef in acdc.items():
        print(f'\n  [{seg.upper()}]')
        print(f'  {"Parameter":<20} {"Coefficient":>12} {"t-stat":>8}  Interpretation')
        print(f'  {"-"*65}')

        # Agglomeration
        aggl_sig = '✓' if abs(ef['t_aggl']) >= 1.96 else '✗' if np.isfinite(ef['t_aggl']) else '?'
        comp_sig  = '✓' if abs(ef['t_comp'])  >= 1.96 else '✗' if np.isfinite(ef['t_comp'])  else '?'
        print(f'  {"c_aggl (agglomeration)":<20} {ef["c_aggl"]:>12.4f} {ef["t_aggl"]:>7.2f}'
              f'  {aggl_sig} More clustered zones attract more trips')
        print(f'  {"c_comp (competition)":<20} {ef["c_comp"]:>12.4f} {ef["t_comp"]:>7.2f}'
              f'  {comp_sig} Surrounding competition reduces attractiveness')

        print(f'\n  Agglomeration score distribution across {len(ef["zone_effects"])} zones:')
        print(f'    Mean score  : {ef["aggl_mean"]:+.4f}   '
              f'Utility at mean  : {ef["utility_aggl_mean"]:+.4f}')
        print(f'    SD   score  : {ef["aggl_sd"]:.4f}   '
              f'Utility at p90   : {ef["utility_aggl_p90"]:+.4f}')
        print(f'    Prob ratio  : p90/p10 aggl zone = {ef["prob_ratio_aggl"]:.2f}× '
              f'(ceteris paribus)')

        # Interpretation of probability ratio
        r = ef['prob_ratio_aggl']
        if np.isfinite(r) and r > 1:
            print(f'    → A top-decile agglomeration zone is {r:.1f}× more likely to be')
            print(f'      chosen than a bottom-decile zone, holding all else constant.')
        elif np.isfinite(r) and r < 1:
            print(f'    → Agglomeration reduces destination probability (unexpected sign).')

        # Top-10 zones
        top10 = ef['top10_zones']
        if not top10.empty:
            print(f'\n  Top 10 zones by net ACDC utility (aggl + comp):')
            print(f'    {"Zone":>5}  {"acdc_aggl":>10}  {"acdc_comp":>10}  '
                  f'{"util_aggl":>10}  {"net_util":>10}')
            for _, zr in top10.iterrows():
                print(f'    {int(zr.zone_id):>5}  {zr.acdc_aggl:>10.4f}  {zr.acdc_comp:>10.4f}'
                      f'  {zr.util_aggl:>10.4f}  {zr.net_acdc_util:>10.4f}')

        try:
            ef['zone_effects'].to_csv(f'{save_dir}/dashboard_acdc_{seg}.csv', index=False)
        except Exception:
            pass

    # ── 4. Destination zone accuracy ──────────────────────────────────────
    if metrics:
        _hdr('4. DESTINATION ZONE ACCURACY')
        print(f'  {"Segment":<10}  {"RMSE":>8}  {"Pearson r":>10}  {"KL div":>8}  Quality')
        print(f'  {"-"*50}')
        for seg in ('work', 'edu', 'nonmand'):
            r   = metrics.get('zone_rmse',    {}).get(seg, float('nan'))
            c   = metrics.get('zone_pearson', {}).get(seg, float('nan'))
            kl  = metrics.get('zone_kl',      {}).get(seg, float('nan'))
            if not np.isfinite(r):
                print(f'  {seg:<10}  {"n/a":>8}  {"n/a":>10}  {"n/a":>8}')
                continue
            # Quality flag
            if c > 0.90 and r < 0.002:
                qual = '★★★ Excellent'
            elif c > 0.75 and r < 0.005:
                qual = '★★  Good'
            elif c > 0.50:
                qual = '★   Fair'
            else:
                qual = '✗   Poor'
            print(f'  {seg:<10}  {r:>8.5f}  {c:>10.3f}  {kl:>8.4f}  {qual}')

        # ── 5. Mode share ─────────────────────────────────────────────────
        _hdr('5. MODE SHARE COMPARISON  (Observed HHTS  vs  Synthetic Population)')
        ALL_MODES = ['DA', 'SR', 'Walk', 'Bike', 'Transit']
        obs_ms  = metrics.get('mode_share_obs',  {})
        pred_ms = metrics.get('mode_share_pred', {})
        if obs_ms or pred_ms:
            print(f'  {"Mode":<10}  {"Obs%":>8}  {"Pred%":>8}  {"Diff%":>8}  Flag')
            print(f'  {"-"*45}')
            for m in ALL_MODES:
                o = obs_ms.get(m, 0.0)
                p = pred_ms.get(m, 0.0)
                d = p - o
                flag = ' ◄ LARGE' if abs(d) > 5.0 else ''
                print(f'  {m:<10}  {o:>8.1f}  {p:>8.1f}  {d:>+8.1f}{flag}')

        ms_by_seg = metrics.get('mode_share_by_segment', {})
        for seg, sdf in ms_by_seg.items():
            print(f'\n  [{seg.upper()}]  {"Mode":<10}  {"Obs%":>7}  {"Pred%":>7}  {"Diff%":>7}')
            print(f'         {"-"*35}')
            for m in ALL_MODES:
                if m in sdf.index:
                    o_ = sdf.loc[m, 'Obs_%']
                    p_ = sdf.loc[m, 'Pred_%']
                    d_ = sdf.loc[m, 'Diff_%']
                    flag = ' ◄' if abs(d_) > 5.0 else ''
                    print(f'         {m:<10}  {o_:>7.1f}  {p_:>7.1f}  {d_:>+7.1f}{flag}')

    # ── 6. Trip-length distribution ───────────────────────────────────────
    if obs_tours is not None and pred_plan is not None:
        _hdr('6. TRIP-LENGTH DISTRIBUTION')
        tld = _compute_tld_metrics(obs_tours, pred_plan)
        oq, pq = tld['obs_quantiles'], tld['pred_quantiles']
        print(f'  {"Quantile":<8}  {"Obs (km)":>10}  {"Pred (km)":>10}  {"Diff":>8}')
        print(f'  {"-"*42}')
        for q in ('p10', 'p25', 'p50', 'p75', 'p90', 'mean'):
            o = oq.get(q, float('nan'))
            p = pq.get(q, float('nan'))
            d = (p - o) if (np.isfinite(o) and np.isfinite(p)) else float('nan')
            label = q.upper() if q != 'mean' else 'Mean'
            print(f'  {label:<8}  {o:>10.2f}  {p:>10.2f}  {d:>+8.2f}')

        print(f'\n  Distance band shares (%):')
        bs_o = tld['band_share_obs']
        bs_p = tld['band_share_pred']
        print(f'  {"Band":<10}  {"Obs%":>8}  {"Pred%":>8}  {"Diff%":>8}')
        print(f'  {"-"*38}')
        for band in bs_o:
            o = bs_o.get(band, float('nan'))
            p = bs_p.get(band, float('nan'))
            d = (p - o) if (np.isfinite(o) and np.isfinite(p)) else float('nan')
            flag = ' ◄' if abs(d) > 5.0 else ''
            print(f'  {band:<10}  {o:>8.1f}  {p:>8.1f}  {d:>+8.1f}{flag}')

        print(f'\n  n (obs={tld["n_obs"]:,}  pred={tld["n_pred"]:,})')
        try:
            pd.DataFrame({
                'quantile':   list(oq.keys()),
                'obs_km':     list(oq.values()),
                'pred_km':    [pq.get(k, float('nan')) for k in oq],
            }).to_csv(f'{save_dir}/dashboard_tld.csv', index=False)
        except Exception:
            pass

    # ── 7. Insignificant free parameters ─────────────────────────────────
    if metrics and not metrics.get('param_table', pd.DataFrame()).empty:
        _hdr('7. INSIGNIFICANT FREE PARAMETERS  (|t| < 1.96)')
        pt  = metrics['param_table']
        bad = pt[~pt['significant'] & ~pt['holdfast']]
        if bad.empty:
            print('  All free parameters are statistically significant ✓')
        else:
            print(f'  {len(bad)} parameters below significance threshold:')
            print(f'  {"Seg":<8} {"Model":<6} {"Parameter":<24} {"Value":>9} {"t-stat":>8} {"SE":>9}')
            print(f'  {"-"*65}')
            for _, r in bad.iterrows():
                print(f'  {r.segment:<8} {r.model:<6} {r.parameter:<24}'
                      f' {r.value:>9.4f} {r.t_stat:>8.2f} {r.std_err:>9.4f}')

    print(f'\n{"═"*W}\n')

    return {
        'fit':          fit_df,
        'sign_checks':  pd.DataFrame(sign_checks) if sign_checks else pd.DataFrame(),
        'acdc_effects': acdc,
        'tld':          tld if (obs_tours is not None and pred_plan is not None) else {},
    }


def write_vehicle_dat(
    plan: pd.DataFrame,
    save_path: str = 'saved_data/vehicle.dat',
    persons: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Write a vehicle.dat file for network simulation.

    One row per unique vehicle-trip (DA or SR).  Columns match the standard
    DynusT / Cube Voyager vehicle record format:

        vehicle_id, hhid, person_id, mode, OTAZ, DTAZ,
        STARTIME_min, ENDTIME_min, travtime_min, purpose

    SR trips that share a vehicle_id with a DA trip represent carpools -
    the vehicle is counted ONCE for network loading but both travellers
    are recorded.

    For network-aware simulation:
      • Each unique vehicle_id = one vehicle in the network
      • DA + matched SR trips have the same vehicle_id → one vehicle,
        multiple occupants (correct for capacity / congestion models)
      • vehicle.dat total row count ≠ total car trips;
        unique vehicle_id count = actual vehicles on network
    """
    car_modes = {1, 2}   # DA=1, SR=2

    if 'row_type' in plan.columns:
        trips = plan[plan['row_type'] == 'trip'].copy()
    else:
        trips = plan.copy()

    car_trips = trips[pd.to_numeric(trips['mode'], errors='coerce').isin(car_modes)].copy()
    if car_trips.empty:
        logger.warning("write_vehicle_dat: no motorised trips found - file not written")
        return pd.DataFrame()

    if 'vehicle_id' not in car_trips.columns or (car_trips['vehicle_id'] == '').all():
        logger.warning("write_vehicle_dat: vehicle_id not assigned - run _assign_vehicle_ids first")
        return pd.DataFrame()

    # ── Build household mapping ───────────────────────────────────────────
    hhid_col = None
    if persons is not None:
        pp = persons.copy()
        if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
            pp = pp.rename(columns={'PERSID': 'person_id'})
        pp['person_id'] = pp['person_id'].astype(str)
        hhid_col = next((c for c in ['hhid', 'HHID', 'hh_id'] if c in pp.columns), None)
        if hhid_col:
            pid_to_hhid = pp.set_index('person_id')[hhid_col].astype(str).to_dict()
            car_trips['hhid'] = car_trips['person_id'].astype(str).map(pid_to_hhid).fillna('')
        else:
            car_trips['hhid'] = car_trips['person_id'].astype(str)
    else:
        car_trips['hhid'] = car_trips['person_id'].astype(str)

    veh_df = pd.DataFrame({
        'vehicle_id':    car_trips['vehicle_id'],
        'hhid':          car_trips['hhid'],
        'person_id':     car_trips['person_id'].astype(str),
        'mode':          pd.to_numeric(car_trips['mode'], errors='coerce').fillna(1).astype(int),
        'mode_name':     car_trips.get('mode_name', pd.Series('DA', index=car_trips.index)),
        'OTAZ':          pd.to_numeric(car_trips['OTAZ'], errors='coerce').fillna(0).astype(int),
        'DTAZ':          pd.to_numeric(car_trips['DTAZ'], errors='coerce').fillna(0).astype(int),
        'STARTIME_min':  pd.to_numeric(car_trips['STARTIME'], errors='coerce').fillna(0).round(1),
        'ENDTIME_min':   pd.to_numeric(car_trips['ENDTIME'],  errors='coerce').fillna(0).round(1),
        'travtime_min':  pd.to_numeric(car_trips['travtime'], errors='coerce').fillna(0).round(1),
        'purpose':       car_trips.get('purpose', pd.Series('', index=car_trips.index)).fillna(''),
    }).reset_index(drop=True)

    # Sort: vehicle_id → STARTIME (chronological order for network loading)
    veh_df = veh_df.sort_values(['vehicle_id', 'STARTIME_min']).reset_index(drop=True)

    # Summary
    n_veh  = veh_df['vehicle_id'].nunique()
    n_da   = (veh_df['mode'] == 1).sum()
    n_sr   = (veh_df['mode'] == 2).sum()
    n_joint = veh_df[veh_df['mode'] == 2].groupby('vehicle_id').ngroups
    print(f"\nVehicle summary:")
    print(f"  {n_veh:,} unique vehicles  |  {n_da:,} DA trips  |  {n_sr:,} SR trips")
    print(f"  {n_joint:,} vehicles with ≥1 SR passenger (joint/carpool)")

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            veh_df.to_csv(save_path, index=False)
            print(f"  vehicle.dat → {save_path}")
        except Exception as _e:
            logger.warning(f"write_vehicle_dat: save failed: {_e}")

    return veh_df


def plot_dest_zone_validation(
    obs_tours: pd.DataFrame,
    pred_plan: pd.DataFrame,
    save_path: str = 'saved_data/plots/val_stage5_dest_zones.png',
    bin_width: int = 20,
):
    """
    Bar-chart comparison of observed vs predicted destination zone (DTAZ)
    distributions for Work, Education, and Non-Mandatory trips.

    Parameters
    ----------
    obs_tours : HHTS tours DataFrame (output of _prep_hhts_tours) - the
                'ground truth' with observed DTAZ and primary_purpose columns.
    pred_plan : Stage 5 plan DataFrame (trips + activities interleaved).
    save_path : PNG output path.
    bin_width : Zone-ID bin width for grouping (default 20 zones per bar).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("plot_dest_zone_validation: matplotlib not available")
        return

    # ── segment filters ───────────────────────────────────────────────────────
    SEG_PURPOSES = {
        'Work':          _WORK_PURPOSES,
        'Education':     _EDU_PURPOSES,
        'Non-Mandatory': _NONMAND_PURPOSES,
    }
    OBS_SEG_COL  = next(
        (c for c in ['primary_purpose', 'main_purpose', 'purpose', 'segment', 'tourpurp']
         if c in obs_tours.columns), None)
    PRED_SEG_COL = next(
        (c for c in ['purpose', 'primary_purpose', 'segment', 'tourpurp']
         if c in pred_plan.columns), None)
    # DTAZ column may be named differently across pipeline stages
    OBS_DTAZ_COL  = next((c for c in ['DTAZ', 'dest_taz', 'dtaz', 'DEST_TAZ'] if c in obs_tours.columns), None)
    PRED_DTAZ_COL = next((c for c in ['DTAZ', 'dest_taz', 'dtaz', 'DEST_TAZ'] if c in pred_plan.columns), None)

    # Real-zone range, derived from the actual DTAZ values seen rather than a
    # hardcoded "1-329" literal — works for both the legacy 1-329 ORDER scheme
    # and the new_pop 507-zone scheme (DTAZ = 9-digit SA2 code). 330 stays the
    # external/catch-all sentinel in both schemes and is always excluded.
    _all_dtaz_vals = []
    if OBS_DTAZ_COL:
        _v = pd.to_numeric(obs_tours[OBS_DTAZ_COL], errors='coerce')
        _all_dtaz_vals.append(_v[(_v > 0) & (_v != 330)])
    if PRED_DTAZ_COL:
        _v = pd.to_numeric(pred_plan[PRED_DTAZ_COL], errors='coerce')
        _all_dtaz_vals.append(_v[(_v > 0) & (_v != 330)])
    _all_dtaz_vals = pd.concat(_all_dtaz_vals) if _all_dtaz_vals else pd.Series(dtype=float)
    _real_lo = int(_all_dtaz_vals.min()) if not _all_dtaz_vals.empty else 1
    _real_hi = int(_all_dtaz_vals.max()) if not _all_dtaz_vals.empty else 329

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Stage 5 - Destination Choice: Observed vs Predicted Zone Distribution',
                 fontsize=13, fontweight='bold')

    # ── one-time diagnostic: show plan structure ──────────────────────────────
    _has_rowtype  = 'row_type' in pred_plan.columns
    _has_purpose  = PRED_SEG_COL is not None
    _has_dtaz     = PRED_DTAZ_COL is not None
    _n_trip_rows  = int((pred_plan['row_type'] == 'trip').sum()) if _has_rowtype else len(pred_plan)
    _n_act_rows   = int((pred_plan['row_type'] == 'activity').sum()) if _has_rowtype else 0
    _dtaz_valid   = int(pd.to_numeric(pred_plan[PRED_DTAZ_COL], errors='coerce').gt(0).sum()) if _has_dtaz else 0
    print(f"\n  [dest validation] pred_plan: {len(pred_plan)} rows  "
          f"| trip={_n_trip_rows}  activity={_n_act_rows}  "
          f"| has purpose={_has_purpose}  has DTAZ={_has_dtaz}  "
          f"| valid DTAZ n={_dtaz_valid}")
    if _has_purpose and len(pred_plan) > 0:
        _top_purp = pred_plan[PRED_SEG_COL].value_counts().head(8).to_dict()
        print(f"  [dest validation] top purposes in pred_plan: {_top_purp}")

    for ax, (seg_label, purposes) in zip(axes, SEG_PURPOSES.items()):
        # Build a lowercase set for case-insensitive matching
        purposes_lower = {str(p).lower() for p in purposes}

        # ── observed DTAZ ─────────────────────────────────────────────────
        obs_dtaz = pd.Series(dtype=float)
        if OBS_SEG_COL and OBS_DTAZ_COL:
            _obs_purp = obs_tours[OBS_SEG_COL].astype(str).str.lower()
            obs_mask = _obs_purp.isin(purposes_lower)
            # Education: exclude childcare/preschool (different destination model)
            if seg_label == 'Education':
                _childcare_lower = {'childcare or kindergarten'}
                obs_mask = obs_mask & ~_obs_purp.isin(_childcare_lower)
            _raw = pd.to_numeric(obs_tours.loc[obs_mask, OBS_DTAZ_COL], errors='coerce')
            obs_dtaz = _raw[_raw > 0].dropna()
        elif OBS_DTAZ_COL:
            _raw = pd.to_numeric(obs_tours[OBS_DTAZ_COL], errors='coerce')
            obs_dtaz = _raw[_raw > 0].dropna()
        # Education and Work: restrict observed DTAZ to real zones only
        if seg_label in ('Education', 'Work') and not obs_dtaz.empty:
            obs_dtaz = obs_dtaz[obs_dtaz.between(_real_lo, _real_hi)]

        # ── predicted DTAZ ────────────────────────────────────────────────
        # Priority: activity rows → trip rows → any row with matching purpose → all DTAZ
        pred_dtaz = pd.Series(dtype=float)

        def _extract_pred(mask):
            if not _has_dtaz:
                return pd.Series(dtype=float)
            _raw = pd.to_numeric(pred_plan.loc[mask, PRED_DTAZ_COL], errors='coerce')
            return _raw[_raw > 0].dropna()

        if _has_purpose:
            _purp_lower = pred_plan[PRED_SEG_COL].astype(str).str.lower()
            _purp_mask  = _purp_lower.isin(purposes_lower)
            if _has_rowtype:
                pred_dtaz = _extract_pred((pred_plan['row_type'] == 'activity') & _purp_mask)
                if pred_dtaz.empty:
                    pred_dtaz = _extract_pred((pred_plan['row_type'] == 'trip') & _purp_mask)
            if pred_dtaz.empty:
                pred_dtaz = _extract_pred(_purp_mask)
        if pred_dtaz.empty and _has_dtaz:
            pred_dtaz = _extract_pred(pd.Series(True, index=pred_plan.index))

        if pred_dtaz.empty:
            _samp = pred_plan[PRED_SEG_COL].astype(str).unique()[:5].tolist() if _has_purpose else 'N/A'
            print(f"  [dest validation] {seg_label}: pred_dtaz empty - "
                  f"purposes={list(purposes)[:4]}, plan purpose sample={_samp}")

        # Education and Work: restrict predicted DTAZ to real zones
        if seg_label in ('Education', 'Work') and not pred_dtaz.empty:
            pred_dtaz = pred_dtaz[pred_dtaz.between(_real_lo, _real_hi)]

        # ── out-of-range diagnostics ──────────────────────────────────────
        # Log any predicted zones outside the real-zone range so bad
        # assignments are visible.
        if not pred_dtaz.empty:
            _pred_int = pred_dtaz.astype(int)
            _n_zero   = int((_pred_int == 0).sum())
            _n_neg    = int((_pred_int < 0).sum())
            _n_hi     = int((_pred_int > _real_hi).sum())
            _n_bad    = _n_zero + _n_neg + _n_hi
            if _n_bad > 0:
                _bad_vals = sorted(_pred_int[(_pred_int <= 0) | (_pred_int > _real_hi)].unique().tolist())[:10]
                print(
                    f"  [dest validation] {seg_label}: WARNING - {_n_bad}/{len(_pred_int)} "
                    f"predicted zones outside {_real_lo}-{_real_hi}  "
                    f"(zero={_n_zero}, neg={_n_neg}, >{_real_hi}={_n_hi})  "
                    f"example bad values: {_bad_vals}"
                )
                logger.warning(
                    f"plot_dest_zone_validation [{seg_label}]: {_n_bad} out-of-range DTAZ "
                    f"(zero={_n_zero}, neg={_n_neg}, >{_real_hi}={_n_hi}); examples={_bad_vals}"
                )
            else:
                _min_z = int(_pred_int.min())
                _max_z = int(_pred_int.max())
                _nuniq = int(_pred_int.nunique())
                print(
                    f"  [dest validation] {seg_label}: pred zones OK - "
                    f"range [{_min_z}-{_max_z}], {_nuniq} unique zones, n={len(_pred_int):,}"
                )
        if not obs_dtaz.empty:
            _obs_int = obs_dtaz.astype(int)
            _n_obs_bad = int((_obs_int <= 0).sum() + (_obs_int > _real_hi).sum())
            if _n_obs_bad > 0:
                _obs_bad = sorted(_obs_int[(_obs_int <= 0) | (_obs_int > _real_hi)].unique().tolist())[:10]
                print(f"  [dest validation] {seg_label}: WARNING - {_n_obs_bad} obs zones outside {_real_lo}-{_real_hi}: {_obs_bad}")

        if obs_dtaz.empty and pred_dtaz.empty:
            ax.set_title(f'{seg_label}\n(no data - check purpose/DTAZ columns)')
            ax.text(0.5, 0.5, 'No data\nCheck purpose column names',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9, color='#888')
            continue

        obs_cnt = obs_dtaz.astype(int).value_counts()
        pred_cnt = pred_dtaz.astype(int).value_counts()
        # Show all real zones binned and ordered left->right. In the legacy
        # 1-329 ORDER scheme this bins by zone number directly; in the
        # new_pop 507-zone scheme (DTAZ = 9-digit SA2 code) bin_width no
        # longer means "N zone IDs apart" but the bins still partition the
        # full observed zone range evenly, which is what the chart needs.
        _bin_span = max(_real_hi - _real_lo + 1, 1)
        _eff_bin_width = bin_width if _bin_span <= 1000 else max(_bin_span // 20, 1)
        _bin_edges = list(range(_real_lo, _real_hi + 1, _eff_bin_width)) + [_real_hi + 1]
        _n_bins = len(_bin_edges) - 1
        _bin_labels = [str(_bin_edges[i]) for i in range(_n_bins)]
        # Expand value_counts to the full real-zone index then accumulate into bins
        _full_idx = range(_real_lo, _real_hi + 1)
        _obs_all  = obs_cnt.reindex(_full_idx, fill_value=0).astype(float)
        _pred_all = pred_cnt.reindex(_full_idx, fill_value=0).astype(float)
        obs_binned  = np.array([_obs_all.loc[_bin_edges[i]:_bin_edges[i+1]-1].sum()
                                for i in range(_n_bins)])
        pred_binned = np.array([_pred_all.loc[_bin_edges[i]:_bin_edges[i+1]-1].sum()
                                for i in range(_n_bins)])
        obs_share_v  = obs_binned  / max(obs_binned.sum(),  1) * 100.0
        pred_share_v = pred_binned / max(pred_binned.sum(), 1) * 100.0

        x = np.arange(_n_bins, dtype=float)
        width = 0.42
        ax.bar(x - width / 2, obs_share_v,
               width=width, label='Observed', color='steelblue', alpha=0.80,
               edgecolor='white', linewidth=0.4)
        ax.bar(x + width / 2, pred_share_v,
               width=width, label='Predicted', color='tomato', alpha=0.80,
               edgecolor='white', linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(_bin_labels, rotation=70, fontsize=7)
        ax.set_xlabel(f'Destination Zone (bin={_eff_bin_width}, ordered {_real_lo}->{_real_hi})', fontsize=9)
        ax.set_ylabel('Share (%)', fontsize=9)
        ax.set_title(seg_label,                  fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)

        # Annotation: n obs / n pred
        ax.text(0.97, 0.97,
                f'obs n={len(obs_dtaz):,}\npred n={len(pred_dtaz):,}',
                transform=ax.transAxes, fontsize=7,
                ha='right', va='top', color='#444',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  Dest zone validation plot → {save_path}")



def run_stage5(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    synth_s4: pd.DataFrame,
    skims,
    landuse: pd.DataFrame,
    tours: pd.DataFrame = None,
    seed: int = 42,
    save_path: str = 'saved_data/stage5_models.pkl',
    segments: list = None,
    verbose: bool = True,
    mode_feature_override: list = None,
    dest_feature_override: list = None,
    coef_warm_start: dict = None,
    warm_start_model_path: str = None,
    crash_skims=None,
    hhts_persons: pd.DataFrame = None,
) -> tuple:
    """
    Estimate mode + destination models on HHTS and build full trip chains
    for the synthetic population.

    Parameters
    ----------
    trips        : HHTS trips from process_survey()["trips"]
    persons      : persons used for PREDICTION (synth population).
                   Also used for estimation if hhts_persons is not provided.
    synth_s4     : Stage 4 output - one row per activity per person
    skims        : OMX file path or open OMX handle
    landuse      : zone landuse (sc.get_land())
    tours        : HHTS tours from process_survey()["tours"] - needed for
                   tour_mode (observed mode choice). If None, falls back to trips.
    seed         : random seed
    save_path    : where to pickle fitted models
    segments     : ['work','edu','nonmand'] by default
    verbose      : logsum progress printing
    coef_warm_start : optional dict of parameter names to warm-start values;
                      useful for reusing coefficients from previous GA or
                      baseline estimates.
    warm_start_model_path : optional path to saved Stage 5 models pickle.
                      If provided and coef_warm_start is empty, coefficients
                      from this file will be merged into the initial estimate.
    hhts_persons : optional HHTS persons table used ONLY for estimation
                   (merging person attributes onto HHTS tours in _prep_hhts_tours).
                   When provided, ``persons`` is still used for prediction.
                   Pass out["persons"] here when ``persons`` is the synthetic
                   population - otherwise person attributes (sexp, incp, etc.)
                   won't merge and interaction terms become all-zeros, triggering
                   a Numba TypingError in Larch's dim3_filler.

    Returns
    -------
    plan   : full trip chain DataFrame (trips + activities interleaved)
    models : fitted models dict
    """
    if not LARCH_AVAILABLE:
        logger.warning("Larch not available - using distance-only fallback")

    print("=" * 60)
    print("STAGE 5 - Destination + Mode + Trip Chain (Larch)")
    if os.environ.get('STAGE5_CLEAR_SHARROW_CACHE', '0') == '1':
        _clear_sharrow_cache()
    print("=" * 60)

    if segments is None:
        segments = ['work', 'edu', 'nonmand']

    strict_estimation = os.environ.get('STAGE5_STRICT_ESTIMATION', '1') != '0'

    if coef_warm_start is None:
        coef_warm_start = {}
    if crash_skims is None:
        crash_skims = _resolve_default_crash_source()
        if crash_skims:
            logger.info(f"run_stage5: auto-detected crash source '{crash_skims}'")
        else:
            logger.info("run_stage5: no crash source detected; CRASH terms will default to zero")
    if warm_start_model_path and not coef_warm_start:
        try:
            prev_models = load_models(warm_start_model_path)
            for seg_res in prev_models.values():
                for key in ('mode_coef', 'dest_coef'):
                    if isinstance(seg_res.get(key), dict):
                        coef_warm_start.update(seg_res[key])
            if coef_warm_start:
                logger.info(
                    f"run_stage5: loaded warm-start coefficients from {warm_start_model_path} "
                    f"({len(coef_warm_start)} params)"
                )
        except Exception as e:
            logger.warning(f"run_stage5: warm-start model load failed: {e}")

    # Auto-load pre-fit warm-start from _test_mode_prefit.py output.
    # If GA_WARMSTART_PATH points to a JSON file (written by the local DE
    # pre-fit validation script), merge those parameter values into
    # coef_warm_start so the HPC run starts from the DE-found basin.
    # Keys in the JSON are "{segment}_{param}" e.g. "work_ASC_SR".
    _prefit_json = os.environ.get('GA_WARMSTART_PATH', '')
    if not _prefit_json:
        # Also check the default location written by _test_mode_prefit.py
        _default_prefit = os.path.join(os.path.dirname(save_path or ''), 'mode_prefit_warmstart.json')
        if os.path.isfile(_default_prefit):
            _prefit_json = _default_prefit
    if _prefit_json and os.path.isfile(_prefit_json):
        try:
            import json as _json
            with open(_prefit_json) as _pf:
                _prefit_params = _json.load(_pf)
            _n_loaded = 0
            for _k, _v in _prefit_params.items():
                # Key format: "{segment}_{param}" — strip the segment prefix
                # and load as generic warm-start (will match any segment that
                # has the same parameter name).
                _parts = _k.split('_', 1)
                if len(_parts) == 2 and _parts[0] in ('work', 'edu', 'nonmand'):
                    _bare_param = _parts[1]
                    # Only insert if not already overridden by a caller-supplied value
                    if _bare_param not in coef_warm_start:
                        coef_warm_start[_bare_param] = float(_v)
                        _n_loaded += 1
            if _n_loaded:
                logger.info(
                    f"run_stage5: loaded {_n_loaded} pre-fit warm-start params "
                    f"from {_prefit_json}"
                )
        except Exception as _pf_e:
            logger.warning(f"run_stage5: failed to load pre-fit warm-start JSON ({_pf_e})")

    skims, close_skims = _open_skims(skims)

    # ── prep HHTS estimation data ──────────────────────────────────────
    # Mode choice needs tour_mode (observed choice per tour).
    # ActivitySim: choosers merged with persons before the groupby loop.
    print("\n[1/3] Preparing HHTS tours...")
    estimation_src = tours if tours is not None else trips
    if tours is None:
        logger.warning(
            "run_stage5: tours=None - using trips table for estimation. "
            "tour_mode may be missing. Pass tours=out['tours'] for best results."
        )
    _est_persons = hhts_persons if hhts_persons is not None else persons
    if hhts_persons is not None:
        logger.info(
            "run_stage5: using hhts_persons for estimation "
            f"({len(hhts_persons):,} rows) and persons for prediction "
            f"({len(persons):,} rows)"
        )
    hhts_tours = _prep_hhts_tours(estimation_src, _est_persons)
    obs_tours  = hhts_tours   # keep reference for validation plot at end
    if 'tour_mode' not in hhts_tours.columns or hhts_tours['tour_mode'].isna().all():
        if 'mode' in hhts_tours.columns:
            hhts_tours['tour_mode'] = hhts_tours['mode'].map(MODE_MAP).fillna(2).astype(int)
        else:
            logger.warning("tour_mode not found - defaulting all to DA (1)")
            hhts_tours['tour_mode'] = 1

    # resolve segment column (ActivitySim: segment_column_name = "tour_purpose")
    seg_col = next((c for c in ['primary_purpose', 'main_purpose', 'purpose']
                    if c in hhts_tours.columns), None)

    print(f"  {len(hhts_tours)} valid HHTS tours | segment column: '{seg_col}'")
    print(f"  tour_mode distribution: {hhts_tours['tour_mode'].value_counts().to_dict()}")
    if seg_col:
        print(f"  Purpose distribution: {hhts_tours[seg_col].value_counts().head(8).to_dict()}")

    # ── ActivitySim groupby-segment estimation loop ────────────────────
    # Mirrors tour_mode_choice_simulate:
    #   for tour_purpose, tours_segment in tours.groupby(segment_col, observed=True):
    #       choices = run_tour_mode_choice_simulate(tours_segment, tour_purpose, ...)
    #       choices_list.append(choices)
    # We do the same for estimation - one model per segment, concat results.
    # ── build flat skim table once - reused by _run_segment for OD merges ──
    # _estimate_mixed_logit needs per-observation IVT/walk/bike values; the
    # HHTS tour table has OTAZ/DTAZ but no skim values merged on.  Building
    # skim_df_est here (once) and passing it down avoids rebuilding it 3×.
    print("\n  Reading flat skim table for estimation OD merges...")
    skim_df_est = _skim_flat(skims)
    print(f"  Skim table: {len(skim_df_est)} OD pairs, "
          f"{len(skim_df_est.columns)-2} matrices")

    print("\n[2/3] Estimating mode + destination models...")
    models = {}

    if seg_col:
        # annotate canonical segment for each tour (groupby-compatible)
        def _to_canonical_seg(val):
            v = str(val).strip()
            if v in _WORK_PURPOSES: return 'work'
            if v in _EDU_PURPOSES:  return 'edu'
            if v in _NONMAND_PURPOSES: return 'nonmand'
            return None

        hhts_tours['_segment'] = hhts_tours[seg_col].map(_to_canonical_seg)
        available_segs = [s for s in segments
                          if s in hhts_tours['_segment'].dropna().unique()]
        print(f"  Available segments: {available_segs}")

        # groupby segment - exactly mirrors ActivitySim's tour_purpose groupby
        for seg, tours_segment in hhts_tours.groupby('_segment', observed=True):
            if seg not in segments:
                continue
            try:
                result = _run_segment(seg, tours_segment, _est_persons, landuse, skims,
                                      skim_df=skim_df_est, seed=seed, verbose=verbose,
                                      mode_feature_override=mode_feature_override,
                                      dest_feature_override=dest_feature_override,
                                      coef_warm_start=coef_warm_start,
                                      crash_skims=crash_skims)
                if result:
                    if strict_estimation and not result.get('dest_coef'):
                        raise RuntimeError(
                            f"run_stage5 [{seg}]: destination coefficients are empty"
                        )
                    models[seg] = result
                    print(f"  [{seg}] estimation complete - keys: {list(result.keys())}")
                else:
                    msg = f"  [{seg}] _run_segment returned empty result"
                    if strict_estimation:
                        raise RuntimeError(msg)
                    print(msg + " - skipping")
            except Exception as _seg_e:
                import traceback as _tb_seg
                print(f"\n  !! [{seg}] _run_segment FAILED: {_seg_e!r}")
                print(f"     {_tb_seg.format_exc(limit=5)}")
                logger.error(f"_run_segment failed for segment='{seg}': {_seg_e!r}")
                if strict_estimation:
                    raise
    else:
        # no purpose column - run all segments on full data (fallback)
        logger.warning("No purpose column found - running all segments on full tours table")
        for seg in segments:
            try:
                result = _run_segment(seg, hhts_tours, _est_persons, landuse, skims,
                                      skim_df=skim_df_est, seed=seed, verbose=verbose,
                                      mode_feature_override=mode_feature_override,
                                      dest_feature_override=dest_feature_override,
                                      coef_warm_start=coef_warm_start,
                                      crash_skims=crash_skims)
                if result:
                    if strict_estimation and not result.get('dest_coef'):
                        raise RuntimeError(
                            f"run_stage5 [{seg}]: destination coefficients are empty"
                        )
                    models[seg] = result
                    print(f"  [{seg}] estimation complete")
                else:
                    msg = f"  [{seg}] _run_segment returned empty result"
                    if strict_estimation:
                        raise RuntimeError(msg)
                    print(msg + " - skipping")
            except Exception as _seg_e2:
                import traceback as _tb_seg2
                print(f"\n  !! [{seg}] _run_segment FAILED: {_seg_e2!r}")
                print(f"     {_tb_seg2.format_exc(limit=5)}")
                logger.error(f"_run_segment failed for segment='{seg}': {_seg_e2!r}")
                if strict_estimation:
                    raise

    # ── Print coefficient tables immediately after estimation ─────────────
    # This runs BEFORE save_models so coefficients are always printed to
    # console even if serialisation fails.  _print_model_table is robust to
    # mangled models and will fall back to _coef_cache if pf is unavailable.
    print("\n" + "=" * 70)
    print("STAGE 5 ESTIMATED COEFFICIENTS")
    print("=" * 70)
    for _seg, _res in models.items():
        for _mtype, _mkey in [('mode', 'mode_model'), ('dest', 'dest_model')]:
            _mdl = _res.get(_mkey)
            if _mdl is None:
                continue
            try:
                _print_model_table(_mdl, f"{_seg.upper()} {_mtype} model")
            except Exception as _pte:
                # Last-resort: dump _coef_cache as a flat table
                _cache = getattr(_mdl, '_coef_cache', None) or _res.get(_mtype[0:4] + '_coef', {})
                print(f"\n  [{_seg} {_mtype}] _print_model_table failed ({_pte})")
                if _cache:
                    print(f"  {'Parameter':<35} {'Value':>10}")
                    print(f"  {'─'*48}")
                    for _pn, _pv in sorted(_cache.items()):
                        print(f"  {_pn:<35} {_pv:>10.4f}")
                else:
                    print(f"  No coefficient data available for {_seg}/{_mtype}")

        # ── Frozen probability table ──────────────────────────────────────
        # Print the cell-level mode share table computed from HHTS observations.
        # These are the probabilities applied directly to the synthetic population
        # (no random draws at prediction time) - showing them here confirms they
        # were computed and validates mode shares vs HHTS observed.
        _fp = _res.get('frozen_probs')
        if _fp is not None and not _fp.empty:
            print(f"\n  ┌─ {_seg.upper()} Frozen probabilities "
                  f"({len(_fp)} cells, {int(_fp['n_obs'].sum())} HHTS obs) ──────────")
            _fp_disp = _fp[FROZEN_CELL_COLS + FROZEN_PROB_COLS + ['n_obs']].copy()
            _fp_pct  = _fp_disp[FROZEN_PROB_COLS] * 100
            for _pc in FROZEN_PROB_COLS:
                _fp_disp[_pc] = _fp_pct[_pc].map('{:.1f}%'.format)
            # show top-10 most common cells
            _top = _fp_disp.nlargest(min(10, len(_fp_disp)), 'n_obs')
            hdr = f"  {'ptype':<22} {'indp':<12} {'inc':<5} {'veh':<4} {'age':<8} {'lic':<4} "
            hdr += ' '.join(f"{'p_'+m:<8}" for m in ['DA','SR','Walk','Bike','Transit'])
            hdr += '  n_obs'
            print(hdr)
            print("  " + "─" * (len(hdr) - 2))
            for _, _row in _top.iterrows():
                ln  = f"  {str(_row.get('ptype','')):<22} "
                ln += f"{str(_row.get('indp_group','')):<12} "
                ln += f"{str(_row.get('income_band','')):<5} "
                ln += f"{str(_row.get('veh_bin','')):<4} "
                ln += f"{str(_row.get('age_group','')):<8} "
                ln += f"{str(_row.get('has_licence','')):<4} "
                for _pc in FROZEN_PROB_COLS:
                    ln += f"{str(_row.get(_pc,'')):<9}"
                ln += f"  {int(_row.get('n_obs', 0))}"
                print(ln)
            # Summary means
            _m_fp = _res.get('frozen_probs')
            if _m_fp is not None:
                _mean_row = _m_fp[FROZEN_PROB_COLS].mean()
                print(f"  {'─'*70}")
                print(f"  Mean across cells: " +
                      "  ".join(f"{p.replace('p_','')}={v:.1%}"
                                for p, v in _mean_row.items()))
            print(f"  └{'─'*69}")
        else:
            logger.warning(f"[{_seg}] No frozen probabilities - mode prediction will "
                           "use per-observation MNL/MXL probabilities")

    if save_path:
        save_models(models, save_path)

    # ── models summary ────────────────────────────────────────────────
    if not models:
        msg = (
            "run_stage5: no segment produced a model; terminating because "
            "Stage 5 cannot proceed without estimated mode+destination models"
        )
        if strict_estimation:
            raise RuntimeError(msg)
        print("\n  !! WARNING: models dict is empty - no segment produced a result.")
        print("     Prediction will use gravity-only fallback - destinations will be coarse.")
    else:
        print(f"\n  Models estimated/loaded for segments: {list(models.keys())}")
        for _seg, _res in models.items():
            _mm = _res.get('mode_model')
            _dm = _res.get('dest_model')
            print(f"    [{_seg}] mode_model={'✓' if _mm else '✗'}  "
                  f"dest_model={'✓' if _dm else '✗'}  "
                  f"keys={list(_res.keys())}")

    # ── ensure OTAZ on synth_s4 ────────────────────────────────────────
    print("\n[3/3] Building full trip chains for synthetic population...")
    s4 = synth_s4.copy()
    _pp_s4 = persons.copy()
    if 'person_id' not in _pp_s4.columns and 'PERSID' in _pp_s4.columns:
        _pp_s4 = _pp_s4.rename(columns={'PERSID': 'person_id'})
    # Coerce person_id to str on both sides to prevent type-mismatch merge errors
    # when HHTS tours have int64 person_id and synth persons have object person_id.
    _pid_col = 'person_id' if 'person_id' in s4.columns else 'PERSID'
    s4[_pid_col]           = s4[_pid_col].astype(str)
    _pp_s4['person_id']    = _pp_s4['person_id'].astype(str)

    if 'OTAZ' not in s4.columns or (s4['OTAZ'] == 0).all():
        if 'OTAZ' in _pp_s4.columns:
            otaz_map = _pp_s4.set_index('person_id')['OTAZ'].to_dict()
            s4['OTAZ'] = s4[_pid_col].map(otaz_map).fillna(1).astype(int).clip(lower=1)
            _bad_otaz = (s4['OTAZ'] <= 0).sum()
            if _bad_otaz > 0:
                logger.warning(f"run_stage5: {_bad_otaz} rows had OTAZ=0 (person_id not in map) - defaulted to zone 1")

    # ── Propagate usual work/school locations from persons to activities ──
    # assign_usual_locations (called before Stage 1) adds WORK_TAZ / SCHOOL_TAZ
    # to the synthetic population.  Stage 5 _build_trip_chain needs them on the
    # activities DataFrame so it can bypass destination choice for workers/students
    # and anchor their trips to their usual work/school zone.
    for _taz_col in ('WORK_TAZ', 'SCHOOL_TAZ'):
        if _taz_col in _pp_s4.columns and _taz_col not in s4.columns:
            _taz_map = _pp_s4.set_index('person_id')[_taz_col].to_dict()
            s4[_taz_col] = s4[_pid_col].map(_taz_map)
            _n_anchored = int((s4[_taz_col].fillna(0) > 0).sum())
            logger.info(f"run_stage5: propagated {_taz_col} from persons "
                        f"to activities ({_n_anchored:,} anchored)")

    # ── OTAZ diagnostic ────────────────────────────────────────────────
    _otaz_counts = s4['OTAZ'].value_counts()
    print(f"  synth_s4 OTAZ: {s4['OTAZ'].nunique()} unique zones, "
          f"total {len(s4):,} rows | "
          f"top 5: {_otaz_counts.head(5).to_dict()}")
    _zero_otaz = (s4['OTAZ'] == 0).sum()
    if _zero_otaz > 0:
        print(f"  !! WARNING: {_zero_otaz:,} rows have OTAZ=0 - these will not "
              f"get model-based predictions (gravity will handle them)")
    if s4['OTAZ'].nunique() <= 3:
        print(f"  !! WARNING: very few unique OTAZ values ({s4['OTAZ'].nunique()}) - "
              f"check that persons table has OTAZ column and synth_s4 person_ids match")
    if 'purpose' in s4.columns:
        print(f"  synth_s4 purpose distribution: "
              f"{s4['purpose'].value_counts().head(8).to_dict()}")

    # build skim lookup for travel time insertion - reuse table already built
    print("  Building skim lookup for travel time insertion...")
    skim_df   = skim_df_est   # already built before the segment loop
    sk_lookup = _skim_lookup(skim_df)
    print(f"  Skim lookup: {len(sk_lookup)} OD pairs")

    # ── zone-batch prediction ─────────────────────────────────────────
    # Processing zones in batches of BATCH_SIZE ensures that each DataTree
    # contains activities from MULTIPLE distinct OTAZs.  This means
    # tour.acdc_comp is a genuine array (varies across rows) rather than a
    # scalar constant (all rows share the same OTAZ in a single-zone batch).
    # Sharrow hashes the DataTree structure - a scalar col vs array col are
    # different shapes and trigger a fresh JIT compilation.  With batches of
    # ~50 zones we get ~7 compilations total instead of 325.
    #
    # Memory: 50 zones × ~12k persons = ~600k activities per batch → ~400 MB.
    # BATCH_SIZE = 1 means one origin zone per iteration (safest for memory,
    # but catastrophically slow: each batch re-triggers Larch/sharrow flow
    # compilation, ~12-15 min per batch × 330 zones ≈ 3 days — job 23300124
    # burned a full walltime this way without ever writing predictions).
    # Default 25 keeps peak memory ≈ 200-300 MB while cutting compilations 25×.
    BATCH_SIZE = int(os.environ.get('STAGE5_BATCH_SIZE', '25'))

    zones = sorted(s4['OTAZ'].dropna().unique().astype(int).tolist())
    n_zones = len(zones)
    zone_batches = [zones[i:i+BATCH_SIZE] for i in range(0, n_zones, BATCH_SIZE)]
    n_batches = len(zone_batches)

    print(f"  Predicting {n_zones} origin zones in {n_batches} batches "
          f"(batch size={BATCH_SIZE}), {len(s4):,} activities total")

    zone_plans   = []
    total_persons = 0
    total_trips   = 0

    # ── Per-batch checkpointing ────────────────────────────────────────
    # Each completed batch is pickled to <out_dir>/stage5_batches/ so a job
    # killed mid-prediction (walltime, OOM) keeps its finished batches and a
    # re-run resumes from disk instead of recomputing everything.  Batch seeds
    # derive from batch_zones[0], so resumed results are identical to a
    # straight-through run.  Disable with STAGE5_BATCH_CHECKPOINT=0.
    _ckpt_dir = None
    if os.environ.get('STAGE5_BATCH_CHECKPOINT', '1') == '1':
        _ckpt_base = (os.environ.get('PIPELINE_OUT_DIR')
                      or os.path.dirname(save_path) or '.')
        _ckpt_dir = os.path.join(_ckpt_base, 'stage5_batches')
        os.makedirs(_ckpt_dir, exist_ok=True)

    for b_idx, batch_zones in enumerate(zone_batches):
        batch_mask = s4['OTAZ'].isin(batch_zones)
        s4_batch   = s4[batch_mask].copy()

        if s4_batch.empty:
            continue

        n_ppl_batch = (s4_batch['person_id'].nunique()
                       if 'person_id' in s4_batch.columns else len(s4_batch))

        _ckpt_fp = (os.path.join(
            _ckpt_dir, f'batch_{batch_zones[0]}_{batch_zones[-1]}.pkl')
            if _ckpt_dir else None)
        if _ckpt_fp and os.path.exists(_ckpt_fp):
            try:
                batch_plan = pd.read_pickle(_ckpt_fp)
                zone_plans.append(batch_plan)
                total_persons += (batch_plan['person_id'].nunique()
                                  if 'person_id' in batch_plan.columns else 0)
                total_trips   += (batch_plan['row_type'] == 'trip').sum()
                print(f"  Batch {b_idx+1:3d}/{n_batches} "
                      f"(zones {batch_zones[0]}-{batch_zones[-1]}) - "
                      f"resumed from checkpoint ({len(batch_plan):,} rows)")
                continue
            except Exception as _ck_e:
                logger.warning(f"  Batch checkpoint unreadable, recomputing: {_ck_e}")

        print(f"  Batch {b_idx+1:3d}/{n_batches} "
              f"(zones {batch_zones[0]}-{batch_zones[-1]}) - "
              f"{n_ppl_batch:,} persons, {len(s4_batch):,} activities")

        try:
            batch_plan = _build_trip_chain(
                s4_batch, models, persons, landuse, skims, sk_lookup,
                seed=seed + batch_zones[0]   # batch-specific seed
            )
            # attach skims immediately, then release DataTree memory
            batch_plan = _attach_skims(batch_plan, skim_df)
            zone_plans.append(batch_plan)
            if _ckpt_fp:
                try:
                    batch_plan.to_pickle(_ckpt_fp)
                except Exception as _cw_e:
                    logger.warning(f"  Batch checkpoint write failed (non-fatal): {_cw_e}")

            total_persons += (batch_plan['person_id'].nunique()
                              if 'person_id' in batch_plan.columns else 0)
            total_trips   += (batch_plan['row_type'] == 'trip').sum()

        except Exception as e:
            import traceback as _tb_batch
            _tb_txt = _tb_batch.format_exc(limit=16)
            logger.warning(
                f"  Batch {b_idx+1} failed - skipping: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{_tb_txt}"
            )
            print(
                f"  Batch {b_idx+1} traceback ({type(e).__name__}):\n"
                f"{_tb_txt}"
            )
            if os.environ.get('STAGE5_FAIL_FAST', '0') == '1':
                print(f"  Batch {b_idx+1} fail-fast traceback:\n{_tb_txt}")
                raise
            continue

    if not zone_plans:
        raise RuntimeError("All batches failed during prediction - check model fit.")

    print(f"\n  Concatenating {len(zone_plans)} zone plans...")
    plan = pd.concat(zone_plans, ignore_index=True)
    # zone_plans held every batch (≈ the full population) at once, and the
    # concat above briefly needs a SECOND same-size buffer on top of that -
    # a ~2x transient spike right here. Drop the now-redundant list before
    # doing anything else so that peak isn't compounded by what follows.
    # (Job 23972282, first-ever 371-zone NEW_POP run: OOM-killed exactly at
    # this point, 259GB used vs a 250GB cgroup limit.)
    del zone_plans
    import gc as _gc
    _gc.collect()

    # _attach_skims already ran per-batch. The unconditional second call here
    # used to redo a full drop+merge over the ENTIRE population "to fill any
    # gaps" - on top of the concat spike above that pushed a marginal-scope
    # run (371 zones vs the previously-tested 329) over the memory limit.
    # Only pay for that full-population drop+merge when per-batch attachment
    # actually left gaps.
    _skim_cols_expected = ['tt_avg', 'tt_am', 'tt_pm', 'log_dist', 'dist',
                           'distwalk', 'distbike', 'crash']
    _skim_cols_present = [c for c in _skim_cols_expected if c in plan.columns]
    _has_gaps = (
        len(_skim_cols_present) < len(_skim_cols_expected)  # some column missing entirely
        or plan[_skim_cols_present].isna().any().any()      # or present but incomplete
    )
    if _has_gaps:
        plan = _attach_skims(plan, skim_df)
    else:
        logger.info("run_stage5: per-batch skim attach had no gaps - skipping "
                    "redundant full-population _attach_skims pass")

    if close_skims:
        skims.close()

    # ── summary ───────────────────────────────────────────────────────
    n_trips = (plan['row_type'] == 'trip').sum()
    n_acts  = (plan['row_type'] == 'activity').sum()
    n_ppl   = plan['person_id'].nunique()
    print(f"\nStage 5 complete.")
    print(f"  {n_ppl} persons | {n_trips} trip legs | {n_acts} activities")
    print(f"  {plan['DTAZ'].nunique()} unique destination zones")
    print(f"  Mean travel time: {plan.loc[plan['row_type']=='trip','travtime'].mean():.1f} min")
    for seg, res in models.items():
        print(f"  [{seg}] mode LL={res.get('mode_ll', '?'):.1f}  "
              f"dest LL={res.get('dest_ll', '?'):.1f}  "
              f"n_tours={res.get('n_tours', '?')}")

    # ── destination zone validation bar-chart ─────────────────────────
    # Uses the HHTS estimation tours (obs_tours) vs the synthetic prediction.
    # obs_tours was built from the 'tours' / 'trips' arg at the top of run_stage5.
    try:
        _out_dir = os.path.dirname(save_path) if save_path else 'saved_data'
        _val_path = os.path.join(_out_dir, 'plots', 'val_stage5_dest_zones.png')
        # obs_tours is the prepped HHTS estimation table built earlier in run_stage5
        if 'obs_tours' in dir() and obs_tours is not None and not obs_tours.empty:
            plot_dest_zone_validation(obs_tours, plan, save_path=_val_path)
        else:
            logger.debug("plot_dest_zone_validation: obs_tours not available - skipping")
    except Exception as _plot_e:
        logger.debug(f"plot_dest_zone_validation failed: {_plot_e}")

    # ── Comprehensive validation metrics ───────────────────────────────────
    try:
        _metrics_path = os.path.join(
            os.path.dirname(save_path) if save_path else 'saved_data',
            'stage5_validation_metrics.csv'
        )
        if obs_tours is not None and not obs_tours.empty:
            _val_metrics = compute_validation_metrics(
                obs_tours, plan, models=models, persons=persons, save_path=_metrics_path
            )
        else:
            logger.debug("compute_validation_metrics: obs_tours not available - skipping")
    except Exception as _vm_e:
        logger.debug(f"compute_validation_metrics failed: {_vm_e}")

    # ── Multi-level validation (trip / tour / work / edu assignment) ──────
    try:
        _ml_save_dir = os.path.join(
            os.path.dirname(save_path) if save_path else 'saved_data',
            'validation'
        )
        if obs_tours is not None and not obs_tours.empty:
            _lu_ml = landuse if landuse is not None else None
            _ml_results = compute_multilevel_validation(
                obs_tours, plan, landuse=_lu_ml, save_dir=_ml_save_dir
            )
        else:
            logger.debug("compute_multilevel_validation: obs_tours not available - skipping")
    except Exception as _ml_e:
        logger.warning(f"compute_multilevel_validation failed: {_ml_e}")
        import traceback; logger.debug(traceback.format_exc())

    # ── HTML destination validation reports ───────────────────────────────
    try:
        _html_out_dir = os.path.join(
            os.path.dirname(save_path) if save_path else 'saved_data',
            'location_choice'
        )
        _trips_plan = plan[plan['row_type'] == 'trip'] if 'row_type' in plan.columns else plan
        _tours_plan = (_trips_plan.groupby(
            [c for c in ['person_id', 'tour_id', 'purpose'] if c in _trips_plan.columns],
            as_index=False
        ).first() if any(c in _trips_plan.columns for c in ['tour_id']) else _trips_plan)
        write_destination_validation_html(
            tours_df=_tours_plan,
            trips_df=_trips_plan,
            out_dir=_html_out_dir,
            landuse=landuse,
        )
        if obs_tours is not None and not obs_tours.empty:
            save_location_choice_summary(models, obs_tours, landuse,
                                         save_dir=os.path.dirname(save_path) if save_path else 'saved_data',
                                         persons=persons)
    except Exception as _html_e:
        logger.debug(f"write_destination_validation_html failed: {_html_e}")
        import traceback; logger.debug(traceback.format_exc())

    # ── Full dashboard (model fit, sign checks, ACDC, TLD, mode share) ───
    try:
        _lu_for_dash = landuse if landuse is not None else pd.DataFrame()
        _obs_for_dash = obs_tours if obs_tours is not None and not obs_tours.empty else None
        print_stage5_dashboard(
            models    = models,
            metrics   = _val_metrics if '_val_metrics' in dir() else {},
            landuse   = _lu_for_dash,
            obs_tours = _obs_for_dash,
            pred_plan = plan if not plan.empty else None,
            save_dir  = os.path.dirname(save_path) if save_path else 'saved_data',
        )
    except Exception as _dash_e:
        logger.debug(f"print_stage5_dashboard failed: {_dash_e}")
        import traceback; logger.debug(traceback.format_exc())

    # ── Write vehicle.dat ─────────────────────────────────────────────────
    try:
        _veh_path = os.path.join(
            os.path.dirname(save_path) if save_path else 'saved_data',
            'vehicle.dat'
        )
        write_vehicle_dat(plan, save_path=_veh_path, persons=persons)
    except Exception as _vd_e:
        logger.debug(f"write_vehicle_dat failed: {_vd_e}")

    return plan, models


def _location_model_selector(name: str) -> str:
    """
    ActivitySim-style model selector normalisation.

    Mirrors activitysim.estimation.larch.location_choice.py model_selector logic.
    """
    selector = str(name or '').strip().lower()
    selector = selector.replace('_location', '')
    selector = selector.replace('_destination', '')
    selector = selector.replace('_subtour', '')
    selector = selector.replace('_tour', '')
    if selector == 'joint':
        selector = 'non_mandatory'
    return selector


def workplace_location_model(**kwargs):
    """Wrapper matching ActivitySim naming for workplace location choice."""
    from usual_location import workplace_location_model as _workplace_location_model
    return _workplace_location_model(**kwargs)


def school_location_model(**kwargs):
    """Wrapper matching ActivitySim naming for school location choice."""
    from usual_location import school_location_model as _school_location_model
    return _school_location_model(**kwargs)


def tour_destination_model(**kwargs):
    """
    Wrapper matching ActivitySim naming for tour-level destination estimation.

    Delegates to run_stage5 and returns (plan, models).
    """
    return run_stage5(**kwargs)


def trip_destination_model(plan: pd.DataFrame,
                          save_path: str = None) -> pd.DataFrame:
    """
    Wrapper matching ActivitySim naming for trip-level destinations.

    Returns only trip rows with valid OD fields for trip-level validation/reporting.
    If save_path is given, writes the trip-level OD table to CSV.
    """
    if plan is None or plan.empty:
        return pd.DataFrame()
    out = plan.copy()
    if 'row_type' in out.columns:
        out = out[out['row_type'] == 'trip'].copy()
    for col in ('OTAZ', 'DTAZ'):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0).astype(int)
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        out.to_csv(save_path, index=False)
        logger.info(f"trip_destination_model: saved {len(out)} trip rows → {save_path}")
    return out


def location_choice_model(name: str, **kwargs):
    """
    Unified ActivitySim-style location choice dispatcher for this codebase.

    Supported selectors:
      workplace, school, non_mandatory, atwork, trip
    """
    selector = _location_model_selector(name)
    if selector == 'workplace':
        return workplace_location_model(**kwargs)
    if selector == 'school':
        return school_location_model(**kwargs)
    if selector in {'non_mandatory', 'atwork'}:
        return tour_destination_model(**kwargs)
    if selector == 'trip':
        if 'plan' not in kwargs:
            raise ValueError("trip location choice requires plan=...")
        return trip_destination_model(kwargs['plan'],
                                      save_path=kwargs.get('save_path'))
    raise ValueError(f"Unsupported location model '{name}' (selector='{selector}')")


def _mode_model_selector(name: str) -> str:
    """
    ActivitySim-style mode model selector normalisation.

    Mirrors activitysim.estimation.larch.mode_choice.py selector naming.
    """
    selector = str(name or '').strip().lower()
    selector = selector.replace('_mode_choice', '')
    selector = selector.replace('_mode', '')
    if selector in {'tour', 'trip', 'atwork_subtour'}:
        return selector
    return selector


def tour_mode_choice_model(**kwargs):
    """Wrapper matching ActivitySim naming for tour-level mode choice."""
    plan, models = run_stage5(**kwargs)
    mode_models = {
        seg: res.get('mode_model')
        for seg, res in (models or {}).items()
        if isinstance(res, dict) and res.get('mode_model') is not None
    }
    return plan, mode_models


def atwork_subtour_mode_choice_model(**kwargs):
    """
    Wrapper matching ActivitySim naming for at-work subtour mode choice.

    In this codebase, at-work subtours are estimated inside Stage 5 segments.
    """
    return tour_mode_choice_model(**kwargs)


def trip_mode_choice_model(**kwargs):
    """
    Wrapper matching ActivitySim naming for trip-level mode choice.

    Delegates to Stage 6 trip mode model estimation.
    """
    if 'trips' not in kwargs or 'persons' not in kwargs:
        raise ValueError("trip mode choice requires trips=... and persons=...")
    from stage6_mode_choice import estimate_mode_models as _estimate_mode_models
    return _estimate_mode_models(
        kwargs['trips'],
        kwargs['persons'],
        landuse=kwargs.get('landuse'),
        min_sample=kwargs.get('min_sample', 50),
    )


def mode_choice_model(name: str, **kwargs):
    """
    Unified ActivitySim-style mode choice dispatcher for this codebase.

    Supported selectors:
      tour, trip, atwork_subtour
    """
    selector = _mode_model_selector(name)
    if selector == 'tour':
        return tour_mode_choice_model(**kwargs)
    if selector == 'trip':
        return trip_mode_choice_model(**kwargs)
    if selector == 'atwork_subtour':
        return atwork_subtour_mode_choice_model(**kwargs)
    raise ValueError(f"Unsupported mode model '{name}' (selector='{selector}')")


# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------
# Usage:
#   python stage5_destination_choice.py                        # resume from saved_data/stage4_output.csv
#   python stage5_destination_choice.py --fresh                # fresh: re-run Stages 1-4 first
#   python stage5_destination_choice.py --input path/to/s4.csv # explicit Stage 4 input
#   python stage5_destination_choice.py --skims path/to/matrices.omx
#
# Requires downcasted_matrices_base.omx in the working directory (or pass --skims).
# Stage 5 calls Larch NL / MXL models - each evaluation writes a flow code.
# On 3.7M persons this stage takes ~30-60 min depending on segment count.

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
    log = logging.getLogger('stage5_main')

    # OUT_DIR: read from environment if set (e.g. by pipeline_logger.py),
    # otherwise default to 'saved_data' for standalone use.
    _OUT = os.environ.get('PIPELINE_OUT_DIR', 'saved_data')
    S4_CSV     = os.path.join(_OUT, 'stage4_output.csv')
    SKIMS_PATH = 'downcasted_matrices_base.omx'

    args  = sys.argv[1:]
    fresh = '--fresh' in args
    args  = [a for a in args if a != '--fresh']

    input_csv = S4_CSV
    if '--input' in args:
        idx = args.index('--input')
        input_csv = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if '--skims' in args:
        idx = args.index('--skims')
        SKIMS_PATH = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    CRASH_SKIMS_PATH = None
    if '--crash-skims' in args:
        idx = args.index('--crash-skims')
        CRASH_SKIMS_PATH = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    zone_arg = args[0] if args else None

    # ── load HHTS + landuse ───────────────────────────────────────────────
    log.info("Loading HHTS survey data + landuse...")
    from data_connect import Synth_Connect
    from compute_accessibility import run_accessibility
    sc      = Synth_Connect()
    out     = sc.process_survey(save=False)
    landuse = run_accessibility(
        sc.get_land(),
        skims=SKIMS_PATH,
        save_path='saved_data/accessibility.csv',
    )
    log.info(f"  HHTS loaded: {len(out['persons'])} persons | "
             f"{len(out['tours'])} tours | {len(out['trips'])} trips")

    if fresh:
        log.info("Fresh mode: running Stages 1-4 first...")
        import subprocess, sys as _sys
        cmd = [_sys.executable, 'stage1_activity_frequency.py', '--fresh']
        if zone_arg:
            cmd.append(zone_arg)
        subprocess.run(cmd, check=True)
        subprocess.run([_sys.executable, 'stage2_day_duration.py'], check=True)
        subprocess.run([_sys.executable, 'stage3_activity_duration.py'], check=True)
        subprocess.run([_sys.executable, 'stage4_schedule_placement.py'], check=True)

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Stage 4 output not found at '{input_csv}'.\n"
            f"  Run stage4_schedule_placement.py first, or use --fresh."
        )
    log.info(f"Loading Stage 4 output from {input_csv} ...")
    scheduled = pd.read_csv(input_csv)
    log.info(f"  {len(scheduled):,} rows loaded")

    # ── check skims ───────────────────────────────────────────────────────
    if not os.path.isfile(SKIMS_PATH):
        raise FileNotFoundError(
            f"Skims file not found: '{SKIMS_PATH}'.\n"
            f"  Use --skims to specify the path to the OMX matrix file."
        )

    from read_skims import list_omx_matrices
    list_omx_matrices(SKIMS_PATH)

    # ── run Stage 5 ───────────────────────────────────────────────────────
    os.makedirs(_OUT, exist_ok=True)
    scheduled_dest, models = run_stage5(
        out['trips'], out['persons'], scheduled,
        skims=SKIMS_PATH,
        landuse=landuse,
        tours=out['tours'],
        save_path=os.path.join(_OUT, 'stage5_models.pkl'),
        crash_skims=CRASH_SKIMS_PATH,
    )
    scheduled_dest.to_csv(os.path.join(_OUT, 'stage5_output.csv'), index=False)

    log.info(f"Stage 5 output saved → {_OUT}/stage5_output.csv")
    log.info(f"Stage 5 models saved → {_OUT}/stage5_models.pkl")
    log.info(f"  {len(scheduled_dest):,} rows (trips + activities interleaved)")
    log.info(f"Pass {_OUT}/stage5_output.csv to stage6_plan_finalisation.py")

