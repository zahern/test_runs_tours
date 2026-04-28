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
# CONFIGURATION — skim matrix name candidates
# (the module searches for these names in the OMX file)
# ---------------------------------------------------------------------------

DEFAULT_MATRICES = [
    'DIST',
    'DISTWALK',
    'DISTBIKE',
    'SOVTOLL_TIME__AM',
    'SOVTOLL_TIME__PM',
    'CRASH',
    # common alternates — included so the reader is robust
    'SOV_TIME__AM',
    'SOV_TIME__PM',
    'SOVTIME_AM',
    'SOVTIME_PM',
    'AUTO_TIME',
    'WALK_DIST',
    'BIKE_DIST',
]


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
                 (e.g. 'NO', 'zone_id', 'TAZ'); None = auto-detect
    chunksize  : number of origin zones to process at once
                 (reduces peak memory for large matrices)

    Returns
    -------
    DataFrame with columns: OTAZ, DTAZ, <matrix_name>, ...

    Notes
    -----
    For a 330-zone model the full matrix has 330² = 108,900 rows.
    For a 3,000-zone model it has 9M rows — use chunksize to avoid
    loading everything into memory at once.
    """
    try:
        import openmatrix as omx
    except ImportError:
        raise ImportError("pip install openmatrix")

    f = omx.open_file(omx_path, 'r')

    # ── available matrices ────────────────────────────────────────────
    available = f.list_matrices()
    logger.info(f"OMX: {len(available)} matrices available: {available}")

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

    if zone_index is None:
        if mappings:
            zone_index = mappings[0]
        else:
            zone_index = None

    # ── zone mapping — handle different openmatrix API versions ──
    # The mapping dict is {zone_id (np.uint32): row_index (int)}
    # Keys = real TAZ IDs, values = 0-based matrix row/col positions.
    # We build two arrays:
    #   zone_ids      : TAZ IDs in row order  (used as OTAZ/DTAZ labels)
    #   row_order     : row indices sorted by zone_id (for correct slicing)
    zone_ids  = None
    row_order = None

    if zone_index and zone_index in mappings:
        try:
            raw = f.mapping(zone_index)
            if isinstance(raw, dict):
                # sort by row index (value) so zone_ids[i] matches matrix row i
                sorted_pairs = sorted(raw.items(), key=lambda kv: kv[1])
                zone_ids  = np.array([int(k) for k, v in sorted_pairs], dtype=int)
                row_order = np.array([int(v) for k, v in sorted_pairs], dtype=int)
            else:
                zone_ids  = np.array(raw, dtype=int).ravel()
                row_order = np.arange(len(zone_ids), dtype=int)
        except Exception as e:
            logger.warning(f"Could not read zone mapping '{zone_index}': {e}")
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
        f"OMX → DataFrame complete\n"
        f"  Shape  : {df.shape}\n"
        f"  Zones  : {n_zones} ({zone_ids.min()}–{zone_ids.max()})\n"
        f"  Matrices loaded: {read_matrices}"
    )
    return df


# ---------------------------------------------------------------------------
# 3.  HIGH-LEVEL READER  (main entry point)
# ---------------------------------------------------------------------------

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
    matrices    : matrix names to read; None = auto-detect
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
        dist_col = next((c for c in ['DIST', 'distance_km', 'dist']
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