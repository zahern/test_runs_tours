"""
read_skims.py
=============
Read OMX skim files and convert to the flat OTAZ/DTAZ DataFrame
format expected by stage5_destination_choice.prepare_skims().

OMX files store matrices as 2D numpy arrays indexed by zone number.
Each matrix (DIST, DISTWALK, DISTBIKE, SOVTOLL_TIME__AM, etc.) is a
separate dataset inside the HDF5 file.

This module:
    1. Opens the OMX file
    2. Lists all available matrices
    3. Reads the zone mapping (row/col index → TAZ ID)
    4. Converts each matrix to long-form (OTAZ, DTAZ, value)
    5. Merges all matrices into one flat DataFrame
    6. Returns it in the exact shape prepare_skims() expects

Requirements
------------
    pip install openmatrix numpy pandas

Usage
-----
    from read_skims import read_omx_skims, omx_to_dataframe

    # quick: read all matrices
    skims_df = read_omx_skims('downcasted_matrices_base.omx')

    # or selective: only the matrices you need
    skims_df = read_omx_skims(
        'downcasted_matrices_base.omx',
        matrices=['DIST', 'DISTWALK', 'DISTBIKE',
                  'SOVTOLL_TIME__AM', 'SOVTOLL_TIME__PM', 'CRASH']
    )

    # then pass straight into the pipeline
    from stage5_destination_choice import run_stage5
    scheduled_dest, models = run_stage5(
        trips, persons, synth_s4, skims_df, landuse
    )
"""

import numpy as np
import pandas as pd
import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ZONE MIGRATION (371) — hard guard against the retired 330-zone system.
# The pipeline now runs exclusively on the 371-zone SA2 skims (sa2_skims_507).
# read_omx_skims() raises if a loaded skim does not have this many zones, so
# the legacy 330-zone downcasted_matrices_base.omx can never be silently used.
# Overridable via ZONE_SYSTEM_N_ZONES for a deliberate alternate zone set.
# ---------------------------------------------------------------------------
EXPECTED_N_ZONES = int(os.environ.get('ZONE_SYSTEM_N_ZONES', '371'))


# ---------------------------------------------------------------------------
# SKIM UNIT SIDECAR (demand_/skim_units.json) — measured, not guessed.
# Value-range guessing (val>600 → seconds, >200 → tenths) misclassifies
# genuine long trips: skims_371.omx stores MINUTES, so a real 250-min
# Toowoomba trip was read as 25 min. The sidecar records the measured divisor
# per matrix (raw/divisor = minutes). Applies only when the loaded file's
# zone count matches the sidecar (default 371); otherwise None (= unknown →
# callers keep legacy guessing). Override path via SKIM_UNITS_FILE.
# ---------------------------------------------------------------------------
_SKIM_UNITS_CACHE = None
_SKIM_UNITS_N_ZONES = None


def _skim_units_path() -> Optional[str]:
    env = os.environ.get('SKIM_UNITS_FILE', '').strip()
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, '..', 'demand_', 'skim_units.json'))
    return cand if os.path.isfile(cand) else None


def load_skim_unit_divisors() -> dict:
    """{matrix_name: divisor} from the sidecar (cached; {} when unavailable)."""
    global _SKIM_UNITS_CACHE, _SKIM_UNITS_N_ZONES
    if _SKIM_UNITS_CACHE is not None:
        return _SKIM_UNITS_CACHE
    _SKIM_UNITS_CACHE = {}
    try:
        import json as _json
        _p = _skim_units_path()
        if _p:
            with open(_p, 'r', encoding='utf-8') as _fh:
                _doc = _json.load(_fh)
            if isinstance(_doc, dict) and isinstance(_doc.get('time_divisors'), dict):
                _SKIM_UNITS_CACHE = {str(k): float(v)
                                     for k, v in _doc['time_divisors'].items()}
                _SKIM_UNITS_N_ZONES = int(_doc.get('n_zones', -1))
    except Exception as _e:
        logger.debug(f"load_skim_unit_divisors skipped ({_e})")
    return _SKIM_UNITS_CACHE


def skim_time_divisor(matrix_name: str, n_zones=None) -> Optional[float]:
    """Divisor converting a raw time-matrix value to minutes, or None if unknown.

    Sidecar-gated: returns None unless the loaded file's zone count matches
    the sidecar's n_zones (371). Pass n_zones=None only when the caller has
    already validated the file (e.g. post-guard paths) — otherwise the legacy
    value-guessing fallback applies.
    """
    try:
        _divs = load_skim_unit_divisors()
        if not _divs or str(matrix_name) not in _divs:
            return None
        if n_zones is not None and int(n_zones) != int(_SKIM_UNITS_N_ZONES or -1):
            return None
        return float(_divs[str(matrix_name)])
    except Exception:
        return None


_ANNOUNCED_UNITS: set = set()


def to_minutes(matrix_name: str, values, n_zones=None):
    """Convert raw skim value(s) to minutes. Scalars stay scalar.

    Sidecar path (measured file): divide by the recorded divisor (1.0 =
    already minutes for every skims_371.omx time matrix). Unknown matrix or
    zone-count mismatch: legacy value-guessing (>600 → /60, >200 → /10).
    The legacy path is a known misclassifier in [200, 600] (a genuine 250-min
    trip reads as 25 min) — it only survives for unmeasured files.
    """
    try:
        import numpy as _np
        _div = skim_time_divisor(matrix_name, n_zones=n_zones)
        _arr = _np.asarray(values, dtype=float)
        _scalar = (_arr.ndim == 0)
        if _div is not None and _div > 0:
            if matrix_name not in _ANNOUNCED_UNITS:
                _ANNOUNCED_UNITS.add(matrix_name)
                logger.info(f"skim units: '{matrix_name}' per sidecar "
                            f"(/{_div:g} -> minutes, no value-guessing)")
            out = _arr / _div
            return float(out) if _scalar else out
        # — legacy fallback for unmeasured matrices/files —
        if matrix_name not in _ANNOUNCED_UNITS:
            _ANNOUNCED_UNITS.add(matrix_name)
            logger.warning(f"skim units: '{matrix_name}' NOT in sidecar — "
                           "legacy value-guessing applies (misclassifies "
                           "genuine 200–600 min trips)")
        if _scalar:
            _v = float(_arr)
            if _v > 600:
                return _v / 60.0
            if _v > 200:
                return _v / 10.0
            return _v
        _med = float(_np.median(_arr[_arr > 0])) if (_arr > 0).any() else 0.0
        if _med > 600:
            return _arr / 60.0
        if _med > 200:
            return _arr / 10.0
        return _arr
    except Exception:
        return values


def _active_zone_ref_csv() -> Optional[str]:
    """Path to the active ORDER↔SA2 zone reference CSV.

    371-only: the 371-zone seq_zone_ids_371.csv is required. The legacy
    seq_zone_ids_.csv (330 system) is NEVER consulted — ORDER 330 is a real
    zone in the 371 system. Returns None if the 371 file is absent.
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    _cand = os.path.join(_here, 'saved_data', 'seq_zone_ids_371.csv')
    if os.path.exists(_cand):
        return _cand
    return None


# ---------------------------------------------------------------------------
# CONFIGURATION — skim matrix name candidates
# (the module searches for these names in the OMX file)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 0.  SA1-LEVEL WALK/BIKE SKIMS  (multi-resolution active-mode LOS)
# ---------------------------------------------------------------------------

SA1_WALK_BIKE_H5 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'demand_', 'walk_bike_sa1_to_sa2_skims.h5'
)

# Level 4 micro-zone skim (preferred over SA1 — uses SEQ-STM's own ~7,000-zone
# system that was purpose-built for fine-grain travel analysis, with proper
# alignment to the EMME network coordinate system).
LVL4_WALK_BIKE_H5 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'demand_', 'walk_bike_lvl4_to_sa2_skims.h5'
)


def load_lvl4_walk_skims(path: Optional[str] = None) -> Optional[dict]:
    """
    Load the Level4-origin x SA2-destination walk/bike skims built by
    demand_/19_build_walk_bike_lvl4_skims.py.

    The SEQ-STM model's Level 4 micro-zones (~7,000 zones) are the native
    fine-grain zone system for active-mode modelling — they map directly
    onto the EMME network and give better spatial resolution for walk/bike
    than ABS SA1 zones, which can have boundary mismatches with the SEQSTM
    geometry.

    Returns a dict with keys:
        'DISTWALK'    : float32 ndarray (n_lvl4, n_sa2) km
        'DISTBIKE'    : float32 ndarray (n_lvl4, n_sa2) km
        'lvl4_to_row' : dict {lvl4_id (int): row_idx}
        'sa2_to_col'  : dict {sa2_code (int): col_idx}
    Returns None if the file doesn't exist (run 18/19 scripts first).
    """
    p = os.path.normpath(path or LVL4_WALK_BIKE_H5)
    if not os.path.exists(p):
        logger.warning(
            f"load_lvl4_walk_skims: {p} not found — "
            f"run demand_/18_build_lvl4_emme_crosswalk.py then "
            f"demand_/19_build_walk_bike_lvl4_skims.py to generate it. "
            f"Walk/bike will fall back to SA2-averaged values."
        )
        return None
    try:
        import h5py as _h5py
        with _h5py.File(p, 'r') as f:
            lvl4_ids  = f['lvl4_ids'][...]
            sa2_codes = f['sa2_codes'][...]
            distwalk  = np.array(f['DISTWALK'], dtype=np.float32)
            distbike  = np.array(f['DISTBIKE'],  dtype=np.float32)
        sa2_to_col = {int(c): i for i, c in enumerate(sa2_codes)}
        order_to_col = {}
        try:
            _seq_csv = _active_zone_ref_csv()
            if _seq_csv and os.path.exists(_seq_csv):
                _sdf = pd.read_csv(_seq_csv, usecols=['SA2_MAIN16', 'ORDER'])
                _sdf = _sdf[pd.to_numeric(_sdf['SA2_MAIN16'], errors='coerce').fillna(0) > 0]
                for _, row in _sdf.iterrows():
                    col = sa2_to_col.get(int(row['SA2_MAIN16']))
                    if col is not None:
                        order_to_col[int(row['ORDER'])] = col
        except Exception as _oe:
            logger.warning(f"load_lvl4_walk_skims: could not build order_to_col: {_oe}")
        logger.info(
            f"Loaded Level4 walk/bike skims: {len(lvl4_ids)} Level4 origins x "
            f"{len(sa2_codes)} SA2 destinations "
            f"(ORDER→col coverage: {len(order_to_col)}/{EXPECTED_N_ZONES})"
        )
        return {
            'DISTWALK':    distwalk,
            'DISTBIKE':    distbike,
            'lvl4_to_row': {int(c): i for i, c in enumerate(lvl4_ids)},
            'sa2_to_col':  sa2_to_col,
            'order_to_col': order_to_col,
        }
    except Exception as e:
        logger.warning(f"load_lvl4_walk_skims: error loading {p}: {e}")
        return None


def load_sa1_walk_skims(path: Optional[str] = None) -> Optional[dict]:
    """
    Load the SA1-origin × SA2-destination walk and bike distance skims built
    by demand_/17_build_walk_bike_sa1_skims.py.

    Returns a dict with keys:
        'DISTWALK'   : float32 ndarray (n_sa1, n_sa2) — km
        'DISTBIKE'   : float32 ndarray (n_sa1, n_sa2) — km
        'sa1_to_row' : dict {sa1_code (int): row_idx} — fast SA1 → row lookup
        'sa2_to_col' : dict {sa2_code (int): col_idx} — fast SA2 → col lookup

    Returns None (with a warning) if the file doesn't exist yet
    (demand_/16_*.py and 17_*.py haven't been run).

    Usage in stage5 _mode_probs_numpy:
        sa1_skim = load_sa1_walk_skims()
        if sa1_skim and 'HOME_SA1' in tour_rows.columns:
            home_sa1 = tour_rows['HOME_SA1'].values.astype(int)
            for i in range(n):
                r = sa1_skim['sa1_to_row'].get(home_sa1[i])
                c = sa1_skim['sa2_to_col'].get(int(dtaz[i]))
                if r is not None and c is not None:
                    dist_walk[i] = sa1_skim['DISTWALK'][r, c]
                    dist_bike[i] = sa1_skim['DISTBIKE'][r, c]
    """
    p = os.path.normpath(path or SA1_WALK_BIKE_H5)
    if not os.path.exists(p):
        logger.warning(
            f"load_sa1_walk_skims: {p} not found — "
            f"run demand_/16_build_sa1_emme_crosswalk.py then "
            f"demand_/17_build_walk_bike_sa1_skims.py to generate it. "
            f"Walk/bike distances will fall back to SA2-averaged values."
        )
        return None
    try:
        import h5py
        with h5py.File(p, 'r') as f:
            sa1_codes = f['sa1_codes'][...]
            sa2_codes = f['sa2_codes'][...]
            distwalk  = np.array(f['DISTWALK'], dtype=np.float32)
            distbike  = np.array(f['DISTBIKE'],  dtype=np.float32)
        sa2_to_col = {int(c): i for i, c in enumerate(sa2_codes)}
        # Build ORDER→col for the 371-zone pipeline (ORDER number → skim col index).
        # The destination dimension of the SA1 skim uses 9-digit SA2 codes; in the
        # 371-zone pipeline DTAZ is an ORDER number (1-371).  Read the 371 zone
        # reference (ORDER → SA2_MAIN16) to build the bridge so stage5 can look
        # up either zone system.
        order_to_col = {}
        try:
            _seq_csv = _active_zone_ref_csv()
            if _seq_csv and os.path.exists(_seq_csv):
                _sdf = pd.read_csv(_seq_csv, usecols=['SA2_MAIN16', 'ORDER'])
                _sdf = _sdf[pd.to_numeric(_sdf['SA2_MAIN16'], errors='coerce').fillna(0) > 0]
                for _, row in _sdf.iterrows():
                    col = sa2_to_col.get(int(row['SA2_MAIN16']))
                    if col is not None:
                        order_to_col[int(row['ORDER'])] = col
        except Exception as _oe:
            logger.warning(f"load_sa1_walk_skims: could not build order_to_col: {_oe}")
        logger.info(
            f"Loaded SA1 walk/bike skims: {len(sa1_codes)} SA1 origins × "
            f"{len(sa2_codes)} SA2 destinations "
            f"(ORDER→col coverage: {len(order_to_col)}/{EXPECTED_N_ZONES})"
        )
        return {
            'DISTWALK':    distwalk,
            'DISTBIKE':    distbike,
            'sa1_to_row':  {int(c): i for i, c in enumerate(sa1_codes)},
            'sa2_to_col':  sa2_to_col,
            'order_to_col': order_to_col,
        }
    except Exception as e:
        logger.warning(f"load_sa1_walk_skims: error loading {p}: {e}")
        return None


def _build_default_matrices():
    """
    Full set of relevant LOS variables across all 5 time periods (AM/MD/PM/EA/EV)
    and all auto/transit submodes available in the SEQ-STM skims — matches the
    matrix set aggregated by demand_/04_aggregate_skims.py into sa2_skims_507.omx.
    Harmless for the legacy ~330-zone downcasted_matrices_base.omx (which only
    has a handful of these) since omx_to_dataframe() silently drops any
    candidate name not actually present in the file.
    """
    periods = ['AM', 'MD', 'PM', 'EA', 'EV']
    auto = [f'{mode}_{metric}__{p}'
            for mode in ('SOVTOLL', 'HOV2TOLL', 'HOV3TOLL')
            for metric in ('TIME', 'DIST', 'TOLL')
            for p in periods]
    transit = [f'{mode}_{metric}__{p}'
               for mode in ('DRV_LOC_WLK', 'WLK_LOC_DRV', 'WLK_LOC_WLK')
               for metric in ('TOTIVT', 'FAR', 'WAIT', 'BOARDS')
               for p in periods]
    legacy_alternates = [
        'CRASH', 'SOV_TIME__AM', 'SOV_TIME__PM', 'SOVTIME_AM', 'SOVTIME_PM',
        'AUTO_TIME', 'WALK_DIST', 'BIKE_DIST',
    ]
    return ['DIST', 'DISTWALK', 'DISTBIKE'] + auto + transit + legacy_alternates


DEFAULT_MATRICES = _build_default_matrices()


# ---------------------------------------------------------------------------
# 1.  LIST MATRICES IN AN OMX FILE
# ---------------------------------------------------------------------------

def list_omx_matrices(omx_path: str) -> List[str]:
    """
    Print and return the names of all matrices in an OMX file.

    Parameters
    ----------
    omx_path : path to .omx file

    Returns
    -------
    list of matrix names

    Example
    -------
        from read_skims import list_omx_matrices
        names = list_omx_matrices('downcasted_matrices_base.omx')
        print(names)
    """
    try:
        import openmatrix as omx
    except ImportError:
        raise ImportError("pip install openmatrix")

    f        = omx.open_file(omx_path, 'r')
    names    = f.list_matrices()
    mappings = f.list_mappings()

    # collect everything we need BEFORE closing
    try:
        shape_str = str(f.shape())
    except Exception:
        shape_str = 'unknown'

    matrix_info = {}
    for n in names:
        try:
            matrix_info[n] = getattr(f[n], 'shape', '?')
        except Exception:
            matrix_info[n] = '?'

    f.close()   # safe to close now — all data collected

    print(f"OMX file : {os.path.basename(omx_path)}")
    print(f"  Shape  : {shape_str}")
    print(f"  Zone mappings : {mappings}")
    print(f"  Matrices ({len(names)}):")
    for n in names:
        print(f"    {n}  {matrix_info.get(n, '')}")
    return names


# ---------------------------------------------------------------------------
# 2.  CORE CONVERTER
# ---------------------------------------------------------------------------

def _is_h5py_omx(omx_path: str) -> bool:
    """
    Detect demand_'s lightweight h5py-written OMX files (see
    Z:/test_runs_tours/demand_/omx_utils.py — used for sa2_skims_507.omx and
    other new_pop / 507-zone-scope skims to avoid an openmatrix/tables binary
    incompatibility). These have an OMX_VERSION file attribute and a
    /lookup/<name> string-array dataset instead of the standard tagged
    OMX_Lookup mapping that the `openmatrix` package expects.
    """
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(omx_path, 'r') as f:
            return 'OMX_VERSION' in f.attrs
    except Exception:
        return False


def _omx_to_dataframe_h5py(
    omx_path: str,
    matrices: Optional[List[str]] = None,
    zone_index: Optional[str] = None,
    chunksize: int = 500,
) -> pd.DataFrame:
    """h5py-based reader for demand_'s lightweight OMX files (see _is_h5py_omx)."""
    import h5py

    with h5py.File(omx_path, 'r') as f:
        available = list(f['data'].keys())
        logger.info(f"OMX (h5py): {len(available)} matrices available: {available}")

        if matrices == 'all':
            matrices = list(available)
        if matrices is None:
            matrices = [m for m in DEFAULT_MATRICES if m in available]
            if not matrices:
                matrices = list(available)
                logger.warning(f"None of the default matrix names found. Reading all {len(matrices)} matrices.")
        else:
            missing = [m for m in matrices if m not in available]
            if missing:
                logger.warning(f"Requested matrices not found in OMX: {missing}")
            matrices = [m for m in matrices if m in available]
        if not matrices:
            raise ValueError(f"No readable matrices found in {omx_path}. Available: {available}")

        lookup_names = list(f['lookup'].keys()) if 'lookup' in f else []
        lookup_name = None
        if zone_index in lookup_names:
            lookup_name = zone_index
        else:
            # An OMX may carry several lookup tables (larch's Exampville file
            # ships TAZ_AREA_TYPE strings next to TAZ_ID). Taking
            # lookup_names[0] blindly crashed with int('SUB') — score ID-like
            # names first and only accept a table whose entries all parse.
            def _as_int(_z):
                try:
                    return int(_z.decode() if isinstance(_z, bytes) else _z)
                except Exception:
                    return None

            _id_like = ('TAZ_ID', 'TAZ', 'taz_id', 'taz', 'ZONE', 'zone',
                        'ZONES', 'zones', 'ID', 'id')
            _ordered = [n for n in lookup_names if n in _id_like]
            _ordered += [n for n in lookup_names if n not in _ordered]
            for _n in _ordered:
                try:
                    _raw = f['lookup'][_n][...]
                    if len(_raw) and all(_as_int(_z) is not None for _z in _raw):
                        lookup_name = _n
                        break
                except Exception:
                    continue
            if lookup_name is not None:
                logger.info(f"OMX (h5py): zone lookup '{lookup_name}' chosen "
                            f"from {lookup_names}")
            elif lookup_names:
                logger.warning(f"OMX (h5py): no integer-parseable lookup in "
                               f"{lookup_names} — 1-based fallback")

        if lookup_name:
            raw = f['lookup'][lookup_name][...]
            zone_ids = np.array([int(z.decode() if isinstance(z, bytes) else z) for z in raw], dtype=int)
            row_order = np.arange(len(zone_ids), dtype=int)
        else:
            n = f['data'][matrices[0]].shape[0]
            zone_ids = np.arange(1, n + 1, dtype=int)
            row_order = np.arange(n, dtype=int)
            logger.warning(f"No usable zone mapping — using 1-based indices 1..{n}.")

        print(f"  Zone mapping: {len(zone_ids)} zones, IDs {zone_ids.min()}-{zone_ids.max()}")
        n_zones = len(zone_ids)

        arrays = {}
        for name in matrices:
            arr = np.array(f['data'][name], dtype=np.float32)
            if arr.shape != (n_zones, n_zones):
                logger.warning(f"Matrix '{name}' shape {arr.shape} != ({n_zones},{n_zones}) — skipping")
                continue
            arrays[name] = arr
            logger.info(f"  Read '{name}': min={arr.min():.3f} max={arr.max():.3f}")

    if not arrays:
        raise ValueError("No matrices could be read from the OMX file.")

    chunks = []
    read_matrices = list(arrays.keys())
    for chunk_start in range(0, n_zones, chunksize):
        chunk_end = min(chunk_start + chunksize, n_zones)
        chunk_zids = zone_ids[chunk_start:chunk_end]
        chunk_rows = row_order[chunk_start:chunk_end]
        n_chunk = len(chunk_zids)

        otaz = np.repeat(chunk_zids, n_zones)
        dtaz = np.tile(zone_ids, n_chunk)
        chunk_dict = {'OTAZ': otaz, 'DTAZ': dtaz}
        for name in read_matrices:
            chunk_dict[name] = arrays[name][np.ix_(chunk_rows, row_order)].ravel()
        chunks.append(pd.DataFrame(chunk_dict))

    df = pd.concat(chunks, ignore_index=True)
    df['OTAZ'] = df['OTAZ'].astype(int)
    df['DTAZ'] = df['DTAZ'].astype(int)
    print(
        f"OMX (h5py) -> DataFrame complete\n  Shape  : {df.shape}\n"
        f"  Zones  : {n_zones} ({zone_ids.min()}-{zone_ids.max()})\n"
        f"  Matrices loaded: {read_matrices}"
    )
    return df


def omx_to_dataframe(
    omx_path: str,
    matrices: Optional[List[str]] = None,
    zone_index: Optional[str] = None,
    chunksize: int = 500,
) -> pd.DataFrame:
    """
    Convert an OMX file to a flat OTAZ/DTAZ DataFrame.

    Parameters
    ----------
    omx_path   : path to .omx file
    matrices   : list of matrix names to read; None = all matrices
                 that match DEFAULT_MATRICES (or all if none match)
    zone_index : name of the zone mapping in the OMX file
                 (e.g. 'NO', 'zone_id', 'TAZ', 'SA2_CODE'); None = auto-detect
    chunksize  : number of origin zones to process at once
                 (reduces peak memory for large matrices)

    Returns
    -------
    DataFrame with columns: OTAZ, DTAZ, <matrix_name>, ...

    Notes
    -----
    For a 330-zone model the full matrix has 330² = 108,900 rows.
    For the new_pop 507-zone model it has 507² = 257,049 rows.
    demand_'s skims (e.g. sa2_skims_507.omx) are written with a lightweight
    h5py-based writer (see Z:/test_runs_tours/demand_/omx_utils.py) to avoid
    an openmatrix/tables binary incompatibility — those are auto-detected
    and read directly via h5py instead of the `openmatrix` package.
    """
    if _is_h5py_omx(omx_path):
        return _omx_to_dataframe_h5py(omx_path, matrices=matrices, zone_index=zone_index, chunksize=chunksize)

    try:
        import openmatrix as omx
    except ImportError:
        raise ImportError("pip install openmatrix")

    f = omx.open_file(omx_path, 'r')

    # ── available matrices ────────────────────────────────────────────
    available = f.list_matrices()
    logger.info(f"OMX: {len(available)} matrices available: {available}")

    if matrices == 'all':
        matrices = list(available)
    if matrices is None:
        # try the default candidates first
        matrices = [m for m in DEFAULT_MATRICES if m in available]
        if not matrices:
            # fall back to all available
            matrices = list(available)
            logger.warning(
                f"None of the default matrix names found. "
                f"Reading all {len(matrices)} matrices."
            )
    else:
        missing = [m for m in matrices if m not in available]
        if missing:
            logger.warning(f"Requested matrices not found in OMX: {missing}")
        matrices = [m for m in matrices if m in available]

    if not matrices:
        f.close()
        raise ValueError(
            f"No readable matrices found in {omx_path}. "
            f"Available: {available}"
        )

    # ── zone mapping ──────────────────────────────────────────────────
    mappings = f.list_mappings()
    logger.info(f"OMX zone mappings: {mappings}")
    # An OMX may carry several mappings (larch's Exampville file ships a
    # TAZ_AREA_TYPE string table next to TAZ_ID). Score ID-like names first
    # and take the first fully int-parseable mapping instead of mappings[0].
    _id_like = ('TAZ_ID', 'TAZ', 'taz_id', 'taz', 'ZONE', 'zone',
                'ZONES', 'zones', 'ID', 'id')
    if zone_index is not None and zone_index in mappings:
        _map_cands = [zone_index]
    else:
        _map_cands = [m for m in mappings if m in _id_like]
        _map_cands += [m for m in mappings if m not in _map_cands]

    # ── zone mapping — handle different openmatrix API versions ──
    # The mapping dict is {zone_id (np.uint32): row_index (int)}
    # Keys = real TAZ IDs, values = 0-based matrix row/col positions.
    # We build two arrays:
    #   zone_ids      : TAZ IDs in row order  (used as OTAZ/DTAZ labels)
    #   row_order     : row indices sorted by zone_id (for correct slicing)
    zone_ids  = None
    row_order = None

    for _zi in _map_cands:
        try:
            raw = f.mapping(_zi)
            if isinstance(raw, dict):
                # sort by row index (value) so zone_ids[i] matches matrix row i
                sorted_pairs = sorted(raw.items(), key=lambda kv: kv[1])
                zone_ids  = np.array([int(k) for k, v in sorted_pairs], dtype=int)
                row_order = np.array([int(v) for k, v in sorted_pairs], dtype=int)
            else:
                zone_ids  = np.array(raw, dtype=int).ravel()
                row_order = np.arange(len(zone_ids), dtype=int)
            if len(zone_ids):
                logger.info(f"OMX zone mapping '{_zi}': {len(zone_ids)} zones")
                break
            zone_ids = None
        except Exception as e:
            logger.warning(f"Could not read zone mapping '{_zi}': {e}")
            zone_ids = None

    if zone_ids is None or len(zone_ids) == 0:
        try:
            shape = f[matrices[0]].shape
        except Exception:
            shape = np.array(f[matrices[0]]).shape
        n = shape[0]
        zone_ids  = np.arange(1, n + 1, dtype=int)
        row_order = np.arange(n, dtype=int)
        logger.warning(
            f"No usable zone mapping — using 1-based indices 1..{n}."
        )

    print(
        f"  Zone mapping: {len(zone_ids)} zones, "
        f"IDs {zone_ids.min()}–{zone_ids.max()}"
    )

    n_zones = len(zone_ids)
    logger.info(f"Matrix size: {n_zones} × {n_zones} = {n_zones**2:,} cells")

    # ── read matrices in chunks ───────────────────────────────────────
    # pre-load all matrices as numpy arrays (they're already in RAM via HDF5)
    arrays = {}
    for name in matrices:
        try:
            arr = np.array(f[name], dtype=np.float32)
            if arr.shape != (n_zones, n_zones):
                logger.warning(
                    f"Matrix '{name}' shape {arr.shape} != "
                    f"({n_zones},{n_zones}) — skipping"
                )
                continue
            arrays[name] = arr
            logger.info(f"  Read '{name}': "
                        f"min={arr.min():.3f} max={arr.max():.3f} "
                        f"mean={arr[arr>0].mean():.3f}")
        except Exception as e:
            logger.warning(f"  Could not read '{name}': {e}")

    f.close()

    if not arrays:
        raise ValueError("No matrices could be read from the OMX file.")

    # ── build long-form DataFrame in chunks ───────────────────────────
    # zone_ids[i]  = TAZ label for row i
    # row_order[i] = matrix row index for zone_ids[i]
    # We iterate over positions 0..n_zones-1 (i), slice the matrix
    # using row_order[chunk], and label with zone_ids[chunk].
    chunks = []
    read_matrices = list(arrays.keys())

    for chunk_start in range(0, n_zones, chunksize):
        chunk_end    = min(chunk_start + chunksize, n_zones)
        chunk_zids   = zone_ids[chunk_start:chunk_end]   # TAZ labels for origins
        chunk_rows   = row_order[chunk_start:chunk_end]  # matrix row indices
        n_chunk      = len(chunk_zids)

        # OTAZ repeats each origin zone_id n_zones times
        # DTAZ cycles through all zone_ids for each origin
        otaz = np.repeat(chunk_zids, n_zones)
        dtaz = np.tile(zone_ids, n_chunk)

        chunk_dict = {'OTAZ': otaz, 'DTAZ': dtaz}
        for name in read_matrices:
            # index matrix rows by actual row indices, columns by row_order
            chunk_dict[name] = arrays[name][np.ix_(chunk_rows, row_order)].ravel()

        chunks.append(pd.DataFrame(chunk_dict))

    df = pd.concat(chunks, ignore_index=True)

    # ── drop intra-zonal cells (diagonal) ────────────────────────────
    # optionally keep them — some models use intra-zonal travel time
    # df = df[df['OTAZ'] != df['DTAZ']].reset_index(drop=True)

    # ── cast zone IDs to int ──────────────────────────────────────────
    df['OTAZ'] = df['OTAZ'].astype(int)
    df['DTAZ'] = df['DTAZ'].astype(int)

    print(
        f"OMX -> DataFrame complete\n"
        f"  Shape  : {df.shape}\n"
        f"  Zones  : {n_zones} ({zone_ids.min()}-{zone_ids.max()})\n"
        f"  Matrices loaded: {read_matrices}"
    )
    return df


# ---------------------------------------------------------------------------
# 3.  HIGH-LEVEL READER  (main entry point)
# ---------------------------------------------------------------------------

def _overlay_real_walk_bike(df: pd.DataFrame) -> pd.DataFrame:
    """Overlay OSM-routed DISTWALK/DISTBIKE onto a loaded skim DataFrame.

    Source: demand_/walk_bike_osm_sa2_371.omx (built by
    21_build_osm_walk_bike_skims.py), keyed by SA2 code — the same id space as
    the 371-zone 507 skims, so the merge is on (OTAZ, DTAZ). Returns df
    unchanged if the file is absent or its zone ids don't overlap df's (so it
    never silently corrupts a mismatched zone system). Path overridable via
    REAL_WALK_BIKE_OMX.
    """
    path = os.environ.get(
        'REAL_WALK_BIKE_OMX',
        os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', 'demand_', 'walk_bike_osm_sa2_371.omx')))
    if not path or not os.path.exists(path):
        return df
    if 'OTAZ' not in df.columns or 'DTAZ' not in df.columns:
        return df
    try:
        wb = omx_to_dataframe(path, matrices=['DISTWALK', 'DISTBIKE'])
    except Exception as e:
        logger.warning(f"_overlay_real_walk_bike: could not read {path}: {e} — "
                       "keeping existing DISTWALK/DISTBIKE")
        return df
    # Refuse to overlay across mismatched zone systems.
    _shared = set(df['DTAZ'].unique()) & set(wb['DTAZ'].unique())
    if len(_shared) < 0.5 * df['DTAZ'].nunique():
        logger.warning(
            f"_overlay_real_walk_bike: zone ids in {os.path.basename(path)} "
            f"overlap only {len(_shared)} of {df['DTAZ'].nunique()} skim zones "
            "— skipping overlay (zone-system mismatch)")
        return df
    wb = wb.rename(columns={'DISTWALK': '_wb_walk', 'DISTBIKE': '_wb_bike'})
    df = df.merge(wb[['OTAZ', 'DTAZ', '_wb_walk', '_wb_bike']],
                  on=['OTAZ', 'DTAZ'], how='left')
    for src, dst in [('_wb_walk', 'DISTWALK'), ('_wb_bike', 'DISTBIKE')]:
        df[dst] = df[src].where(df[src].notna(),
                                df[dst] if dst in df.columns else df[src])
    df = df.drop(columns=['_wb_walk', '_wb_bike'])
    logger.info(
        f"_overlay_real_walk_bike: applied real OSM walk/bike from "
        f"{os.path.basename(path)} "
        f"(walk mean={df['DISTWALK'].mean():.3f} bike mean={df['DISTBIKE'].mean():.3f} "
        f"km; distinct={not np.allclose(df['DISTWALK'], df['DISTBIKE'])})")
    return df


def read_omx_skims(
    omx_path: str,
    matrices: Optional[List[str]] = None,
    zone_index: Optional[str] = None,
    add_derived: bool = True,
) -> pd.DataFrame:
    """
    Read an OMX skim file and return a DataFrame ready for
    stage5_destination_choice.prepare_skims().

    Parameters
    ----------
    omx_path    : path to .omx file
    matrices    : matrix names to read; None = auto-detect (SEQ defaults
                  first, all-available fallback); 'all' = every matrix
                  (transfer-test OMXs with non-SEQ names)
    zone_index  : OMX zone mapping name; None = auto-detect
    add_derived : if True, add derived columns:
                    tt_avg    = mean of AM and PM travel times
                    log_dist  = log1p(DIST)
                    cost_proxy= DIST * 0.12  (fuel cost AUD/km)

    Returns
    -------
    DataFrame with OTAZ, DTAZ, and one column per matrix,
    plus derived columns if add_derived=True.

    Example
    -------
        skims = read_omx_skims('downcasted_matrices_base.omx')
        # → use directly in run_stage5(... skims=skims ...)
    """
    df = omx_to_dataframe(omx_path, matrices=matrices,
                          zone_index=zone_index)

    # ── ZONE MIGRATION (371) hard guard ───────────────────────────────────
    # Refuse any skim that is not the active 371-zone system. This is the
    # single choke point every stage's skim load passes through, so a stray
    # 330-zone downcasted_matrices_base.omx (or any wrong-sized OMX) fails
    # loudly here instead of silently misaligning DTAZ/OTAZ downstream.
    try:
        _n_dtaz = int(df['DTAZ'].nunique()) if 'DTAZ' in df.columns else 0
        _n_otaz = int(df['OTAZ'].nunique()) if 'OTAZ' in df.columns else 0
        _n_zones_seen = max(_n_dtaz, _n_otaz)
    except Exception:
        _n_zones_seen = 0
    if _n_zones_seen != EXPECTED_N_ZONES:
        raise ValueError(
            f"read_omx_skims: '{os.path.basename(omx_path)}' has "
            f"{_n_zones_seen} zones, expected {EXPECTED_N_ZONES} (the 371-zone "
            f"SA2 system). The legacy 330-zone downcasted_matrices_base.omx has "
            f"been retired — point PIPELINE_SKIMS_PATH at sa2_skims_507(_omxspec)"
            f".omx, or set ZONE_SYSTEM_N_ZONES if you intend a different zone set."
        )

    # ── ZONE MIGRATION (371): overlay real OSM walk/bike distances ────────
    # If the OSM-routed walk/bike skim (demand_/21_build_osm_walk_bike_skims.py)
    # has been built, replace the placeholder DISTWALK/DISTBIKE (which upstream
    # are identical to each other and only 0.9x auto DIST) with the real
    # pedestrian-network walk distance and the detour-scaled bike distance.
    # No-op until that file exists, so this is inert on the branch and
    # self-activates once the builder has run. 371-system data ONLY: never
    # overlay onto a transfer-test zone system (ids 1..N overlap SA2 ids
    # numerically but mean different zones — a silent corruption).
    if int(EXPECTED_N_ZONES) == 371:
        df = _overlay_real_walk_bike(df)
    else:
        logger.info(
            f"read_omx_skims: skipping OSM walk/bike overlay "
            f"(ZONE_SYSTEM_N_ZONES={EXPECTED_N_ZONES}, not the 371 system)")

    if add_derived:
        # ── average travel time ───────────────────────────────────────
        am_col = next((c for c in ['SOVTOLL_TIME__AM', 'SOV_TIME__AM',
                                   'SOVTIME_AM', 'AUTO_TIME']
                       if c in df.columns), None)
        pm_col = next((c for c in ['SOVTOLL_TIME__PM', 'SOV_TIME__PM',
                                   'SOVTIME_PM']
                       if c in df.columns), None)

        if am_col and pm_col:
            df['tt_avg'] = (df[am_col] + df[pm_col]) / 2.0
        elif am_col:
            df['tt_avg'] = df[am_col]
            logger.warning("No PM skim found — tt_avg = AM only")
        else:
            df['tt_avg'] = 0.0
            logger.warning("No travel time skim found — tt_avg set to 0")

        # ── log distance ──────────────────────────────────────────────
        dist_col = next((c for c in ['DIST', 'distance_km', 'dist', 'AUTO_DIST']
                         if c in df.columns), None)
        if dist_col:
            df['log_dist'] = np.log1p(df[dist_col])
            df['dist']     = df[dist_col]
        else:
            df['log_dist'] = 0.0
            df['dist']     = 0.0

        # ── walk / bike distance ──────────────────────────────────────
        for src, dst in [('DISTWALK', 'distwalk'),
                          ('WALK_DIST', 'distwalk'),
                          ('DISTBIKE',  'distbike'),
                          ('BIKE_DIST', 'distbike')]:
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        if 'distwalk' not in df.columns:
            df['distwalk'] = df.get('dist', 0.0)
        if 'distbike' not in df.columns:
            df['distbike'] = df.get('dist', 0.0)

        # ── crash ─────────────────────────────────────────────────────
        if 'CRASH' not in df.columns:
            df['crash'] = 0.0
        else:
            df['crash'] = df['CRASH']

        # ── cost proxy ────────────────────────────────────────────────
        df['cost_proxy'] = df['dist'] * 0.12

        # ── AM / PM aliases ───────────────────────────────────────────
        if am_col and 'tt_am' not in df.columns:
            df['tt_am'] = df[am_col]
        if pm_col and 'tt_pm' not in df.columns:
            df['tt_pm'] = df[pm_col]

    # ── summary ───────────────────────────────────────────────────────
    print("\nSkim DataFrame summary:")
    for col in ['tt_avg', 'tt_am', 'tt_pm', 'dist',
                'distwalk', 'distbike', 'crash']:
        if col in df.columns:
            vals = df.loc[df[col] > 0, col]
            if not vals.empty:
                print(f"  {col:20s}: "
                      f"min={vals.min():.2f}  "
                      f"mean={vals.mean():.2f}  "
                      f"max={vals.max():.2f}  "
                      f"(n>{0}: {len(vals):,})")

    return df


# ---------------------------------------------------------------------------
# 4.  PATCH prepare_skims TO ACCEPT OMX PATH
# ---------------------------------------------------------------------------

def prepare_skims_from_omx(
    skims_input,
    matrices: Optional[List[str]] = None,
    zone_index: Optional[str] = None,
) -> dict:
    """
    Drop-in replacement for stage5_destination_choice.prepare_skims()
    that accepts either:
        - a path to an OMX file  (str ending in .omx)
        - a pre-loaded DataFrame  (already flat OTAZ/DTAZ format)
        - an open omx.File object

    Returns the same dict format as prepare_skims():
        {'df': DataFrame, 'lookup': dict, 'cols': list}

    Example
    -------
        from read_skims import prepare_skims_from_omx
        from stage5_destination_choice import (
            prepare_landuse, estimate_destination_models,
            predict_destinations
        )

        skim_info = prepare_skims_from_omx('downcasted_matrices_base.omx')
        lu_info   = prepare_landuse(landuse)

        models = estimate_destination_models(
            trips, persons, skim_info, lu_info
        )
    """
    from stage5_destination_choice import prepare_skims

    # ── case 1: OMX file path ─────────────────────────────────────────
    if isinstance(skims_input, str) and skims_input.endswith('.omx'):
        print(f"Reading OMX: {skims_input}")
        df = read_omx_skims(skims_input, matrices=matrices,
                            zone_index=zone_index)
        return prepare_skims(df)

    # ── case 2: open omx.File object ─────────────────────────────────
    try:
        import openmatrix as omx
        if isinstance(skims_input, omx.File):
            path = skims_input.filename
            skims_input.close()
            print(f"Converting open OMX file: {path}")
            df = read_omx_skims(path, matrices=matrices,
                                zone_index=zone_index)
            return prepare_skims(df)
    except ImportError:
        pass

    # ── case 3: already a DataFrame ──────────────────────────────────
    if isinstance(skims_input, pd.DataFrame):
        return prepare_skims(skims_input)

    raise TypeError(
        f"skims_input must be an OMX path (str), omx.File, "
        f"or DataFrame. Got: {type(skims_input)}"
    )


# ---------------------------------------------------------------------------
# 4.  CRASH / NEAR-MISS SAFETY FACTORS (507-zone scope only)
# ---------------------------------------------------------------------------

# demand_'s road-safety OD data lives in separate OMX files from the travel
# LOS skims (different build scripts: 05_crash_matrices.py / nearmiss
# pipeline), all at the same 507-zone scope. Not loaded by default — call
# merge_safety_skims() / merge_ecf_component() explicitly where a stage wants
# to use crash/near-miss exposure as a destination- or route-choice factor
# (e.g. alongside spatial_assignment.py / safety_experiment_analysis.py's
# existing safety angle). Matrices available:
#   crash_matrices_371.omx       : crash_total, crash_headon, crash_rearend,
#                                 crash_runoff, crash_intersection, crash_other,
#                                 crash_length_km, crash_rate_total — the 371-zone
#                                 cut of crash_matrices_507.omx (58 peripheral
#                                 zones donor-filled from nearest covered SA2 by
#                                 centroid; covered 313² block bit-identical).
#                                 crash_rate_total (crashes/km) is the
#                                 canonical ECF (Expected Crash Frequency)
#                                 safety component used by the ABM — see
#                                 merge_ecf_component() below.
#   nearmiss_ecf_skim_371.omx    : ecf_total, ecf_per_km, nm_total, nm_per_km_total,
#                                 nm_{am,op,pm,ev}, safety_index_total,
#                                 route_length_km — route-aggregated near-miss/
#                                 ExpCrashes exposure (371-zone cut of
#                                 nearmiss_ecf_skim_507.omx, 47 zones donor-filled).
#                                 Available for supplementary analysis via
#                                 merge_safety_skims(sources=['ecf']), but NOT
#                                 the model's ECF component.
#   nearmiss_skim_507.omx      : DEPRECATED — kept only for backward
#                                 compatibility. nearmiss_{origin,dest,total}_*
#                                 are flat per-zone counts broadcast across
#                                 rows/columns (every OD pair sharing an
#                                 origin or destination zone gets an identical
#                                 value) — NOT OD-pair-specific.
SAFETY_OMX_FILES = {
    'crash':    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demand_', 'crash_matrices_371.omx'),
    'ecf':      os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demand_', 'nearmiss_ecf_skim_371.omx'),
    'nearmiss': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'demand_', 'nearmiss_skim_507.omx'),
}

ECF_SOURCE_MATRIX = 'crash_rate_total'  # the model's single ECF safety component


def merge_ecf_component(skims_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge the model's single ECF (Expected Crash Frequency) safety
    component — crash_rate_total (crashes/km) from crash_matrices_371.omx —
    onto a flat OTAZ/DTAZ skims DataFrame, as a column named 'ECF'.

    Full 371² coverage (58 peripheral zones donor-filled); no fillna(0) gaps
    remain, but the fill is kept as a guard.
    """
    path = SAFETY_OMX_FILES['crash']
    if not os.path.exists(path):
        logger.warning(f"merge_ecf_component: crash OMX not found at {path} — ECF column not added")
        return skims_df
    src_df = omx_to_dataframe(path, matrices=[ECF_SOURCE_MATRIX])
    src_df = src_df.rename(columns={ECF_SOURCE_MATRIX: 'ECF'})
    out = skims_df.merge(src_df[['OTAZ', 'DTAZ', 'ECF']], on=['OTAZ', 'DTAZ'], how='left')
    out['ECF'] = out['ECF'].fillna(0)
    return out


def merge_safety_skims(skims_df: pd.DataFrame, sources: Optional[List[str]] = None,
                       matrices: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Left-merge crash/near-miss OD factors onto an existing flat OTAZ/DTAZ
    skims DataFrame (e.g. the output of read_omx_skims() for sa2_skims_507.omx).

    For the model's actual ECF safety component, prefer merge_ecf_component()
    (crash_rate_total only) — this function is for supplementary/exploratory
    analysis across the broader set of crash/near-miss factors.

    Parameters
    ----------
    skims_df : flat OTAZ/DTAZ DataFrame to merge into (must already exist —
               this does not build one on its own).
    sources  : which safety OMX files to pull from; None = 'crash' only
               (matches the model's ECF component). Pass ['crash', 'ecf'] or
               ['nearmiss'] explicitly for supplementary/exploratory factors.
    matrices : specific matrix names to merge; None = all available in each
               source file.

    Returns
    -------
    skims_df with additional crash_*/ecf_*/nm_* columns. The 371-zone safety
    files give full OD coverage (peripheral zones donor-filled at build);
    rows missing safety data for any other reason get 0, not NaN.
    """
    sources = sources or ['crash']
    out = skims_df
    for src in sources:
        path = SAFETY_OMX_FILES.get(src)
        if not path or not os.path.exists(path):
            logger.warning(f"merge_safety_skims: {src} OMX not found at {path} — skipping")
            continue
        src_df = omx_to_dataframe(path, matrices=matrices)
        merge_cols = [c for c in src_df.columns if c not in ('OTAZ', 'DTAZ')]
        out = out.merge(src_df[['OTAZ', 'DTAZ'] + merge_cols], on=['OTAZ', 'DTAZ'], how='left')
        for c in merge_cols:
            out[c] = out[c].fillna(0)
        logger.info(f"merge_safety_skims: merged {len(merge_cols)} columns from {src} ({path})")
    return out