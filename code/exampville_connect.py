#!/usr/bin/env python
"""
exampville_connect.py — Exampville data-source adapter for the Stage-5 pipeline.
================================================================================
Loads Larch's built-in Exampville synthetic town (40 TAZs, 5,000 households,
12,349 persons, 20,739 tours; Larch docs examples 200/201/203) and maps it
onto this pipeline's Stage-5 frames (tours / persons / landuse / skim_df).

Design rules
------------
1. Larch-gated in ONE place: only load_exampville_raw() / copy_package_skims()
   import larch. Every mapping helper below is pure pandas/numpy and runs
   anywhere (mapping is unit-tested without larch).
2. Artifact separation: every path this module builds lives under
   EXAMPVILLE_ROOT (default runs/exampville_ab, overridable via env or arg).
   guard_path() refuses any path escaping the root. NOTHING is written to
   saved_data/, runs/ga_parallel/, runs/safety_* or any QLD tree.
3. Scope honesty (also stated in the paper): Exampville ships tours without
   diaries/times/durations/stops/crash. Transfer coverage: Stage 0 CDAP
   (labels derived from tour counts), Stage 1 frequency (counts derived from
   tours; work + consolidated nonmand cells only), Stage 5 primary mode +
   destination. Out of scope by construction: stages 2-4 (need diary times /
   durations), 5b (needs stops), M3 (needs times), safety terms (nosafety).

Column mapping (Exampville -> pipeline canonical)
-------------------------------------------------
hh:  HHID, HOMETAZ->OTAZ/home, INCOME->incp, N_VEHICLES->vehrd,
     HHSIZE->HHSIZE/nprd, N_WORKERS->n_workers_hh
pp:  PERSONID->person_id, AGE (bands derived; age5p/AGEGROUP banded 1-18),
     WORKS->WORKSTAT/workstat_fulltime, CARLICENCE=(AGE>=16) [201 uses AGE>=16
     for DA availability; Exampville records no licence]
tour: TOURID->tour_id/ptour_id, PERSONID, DTAZ, TOURMODE (1-5, SAME coding as
     pipeline MODE_MAP), TOURPURP 1->'work', 2->'othdiscr' (routes to the
     nonmand segment; Exampville has no sub-purpose or education detail)
skims (OMX 40x40): AUTO_TIME->SOVTOLL_TIME__AM (+PM copy), AUTO_DIST->DIST,
     WALK_DIST->DISTWALK, BIKE_TIME/4->DISTBIKE, TRANSIT_IVTT->
     WLK_LOC_WLK_TOTIVT__AM, TRANSIT_FARE kept.
emp: TOTAL_EMP->TOTEMP, RETAIL_EMP->RETEMPN, NONRETAIL_EMP->FPSEMPN
     (missing attractor columns are zero-filled downstream already).
Times: Exampville tours carry no schedule — estimation runs without STARTIME
     (pipeline defaults to AM peak); synth_s4 application rows get uniform
     STARTIME=480/ENDTIME=540, documented as unscheduled application.
"""
import os
import shutil

import numpy as np
import pandas as pd

SOURCE_TAG = 'exampville'
SEGMENTS = ['work', 'nonmand']  # TOURPURP 1/2; no education detail
PURPOSE_MAP = {1: 'work', 2: 'othdiscr'}  # 'othdiscr' routes to nonmand
APP_STARTIME = 480.0   # uniform application placement (documented assumption)
APP_ENDTIME = 540.0

_PIPELINE_MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}


def exampville_root(outdir=None):
    """Resolve + create the artifact root. Refuses non-exampville roots."""
    root = os.path.abspath(outdir or os.environ.get(
        'EXAMPVILLE_ROOT', os.path.join('runs', 'exampville_ab')))
    if 'exampville' not in os.path.basename(root).lower():
        raise RuntimeError(
            f"Artifact separation: EXAMPVILLE_ROOT must contain 'exampville', "
            f"got {root}")
    os.makedirs(root, exist_ok=True)
    return root


def guard_path(root, *parts):
    """Join under root; raise if the result escapes the root."""
    root = os.path.abspath(root)
    p = os.path.abspath(os.path.join(root, *parts))
    if p != root and not p.startswith(root + os.sep):
        raise RuntimeError(f"Artifact separation: {p} escapes {root}")
    return p


def load_exampville_raw():
    """Load package data via larch (cluster-only). Returns dict of frames+handles.

    Uses lx.example(200, [...]) exactly as Larch docs examples 200/201/203.
    Raises RuntimeError with a clear message when larch is unavailable.
    """
    try:
        import larch as lx
    except Exception as e:
        raise RuntimeError(
            f"Exampville loading needs larch (run in zigenv on the cluster): {e}")
    hh, pp, tour, skims, emp = lx.example(
        200, ['hh', 'pp', 'tour', 'skims', 'emp'])
    try:
        import larch.exampville as _ev
        skim_src = getattr(getattr(_ev, 'files', None), 'skims', None)
    except Exception:
        skim_src = None
    if not skim_src or not os.path.exists(str(skim_src)):
        # larch 6.x dropped the files.skims handle attr; the OMX ships at
        # larch/data_warehouse/exampville_skims.omx (job 25302771).
        skim_src = _package_skim_path()
    return {'hh': hh, 'pp': pp, 'tour': tour, 'skims': skims, 'emp': emp,
            'skim_src': skim_src}


def _package_skim_path():
    """Locate larch's bundled exampville_skims.omx without the files API."""
    for _cand in ('data_warehouse/exampville_skims.omx',
                  'data_warehouse/exampville_skims.omx'.replace('/', os.sep)):
        try:
            from importlib import resources
            _p = resources.files('larch').joinpath(*_cand.split('/'))
            if os.path.exists(str(_p)):
                return str(_p)
        except Exception:
            continue
    try:  # last resort: beside the exampville module itself
        import larch.exampville as _ev
        _d = os.path.dirname(os.path.abspath(_ev.__file__))
        for _nm in ('exampville_skims.omx',):
            _p = os.path.join(_d, _nm)
            if os.path.exists(_p):
                return _p
    except Exception:
        pass
    return None


def _h5_omx_read_all(src):
    """Read an OMX (h5py) -> ({matrix_name: 2D array}, TAZ ids).

    Handles the standard OMX layout (``data/`` matrices, ``lookup/`` zone
    tables) and also flat files that hold matrices at the root.
    """
    import h5py
    mats, taz = {}, None
    with h5py.File(src, 'r') as f:
        def _collect(g, group):
            for k in g:
                item = g[k]
                if isinstance(item, h5py.Group):
                    _collect(item, k)
                elif item.ndim == 2:
                    mats[k] = np.asarray(item[()])
        if 'data' in f and isinstance(f['data'], h5py.Group):
            _collect(f['data'], 'data')
        else:
            _collect(f, None)
        lookup = f['lookup'] if 'lookup' in f else None
        if lookup is not None:
            for key in ('TAZ_ID', 'TAZ', 'ZONE'):
                if key in lookup:
                    taz = np.asarray(lookup[key][()]).astype(int).ravel()
                    break
        else:
            for key in ('TAZ_ID', 'TAZ', 'ZONE'):
                if key in f and isinstance(f[key], h5py.Dataset):
                    taz = np.asarray(f[key][()]).astype(int).ravel()
                    break
    if not mats:
        raise RuntimeError(f"No 2D matrices found in OMX: {src}")
    return mats, taz


def _write_omx(dst, mats, taz_ids):
    """Deprecated shim kept for callers; prefer append_canonical_aliases().

    Writing a file from scratch with h5py drops the root attributes
    (OMX_VERSION / CLASS / SHAPE) that make the OMX recognisable, which sent
    read_skims down the openmatrix branch and yielded ZERO matrices.  Use
    append_canonical_aliases(), which edits a copy of the real package OMX.
    """
    raise RuntimeError(
        "_write_omx() cannot reproduce a readable OMX (root attributes are "
        "required). Use append_canonical_aliases() instead.")


def append_canonical_aliases(dst, aliases, taz_ids):
    """Add canonical skim matrices to a COPY of the package OMX, in place.

    Editing a copy (rather than writing a new file) preserves the package
    file's root attributes (OMX_VERSION/CLASS/SHAPE) and lookup group, which
    read_skims._is_h5py_omx() keys on to pick the h5py reader.  A from-scratch
    h5py write omits those attributes and the OMX then reads as ZERO matrices.
    """
    import h5py
    with h5py.File(dst, 'r+') as f:
        if 'data' not in f:
            raise RuntimeError(f"OMX has no 'data' group: {dst}")
        dgrp = f['data']
        for name, arr in aliases.items():
            arr = np.asarray(arr, dtype=float)
            if name in dgrp:
                del dgrp[name]
            dgrp.create_dataset(name, data=arr)
        if taz_ids is not None and 'lookup' in f and 'TAZ_ID' not in f['lookup']:
            f['lookup'].create_dataset(
                'TAZ_ID', data=np.asarray(taz_ids).astype(int).ravel())
    return dst


def write_canonical_skims(dst, mats, taz_ids):
    """Return the canonical alias matrices for the Exampville skims.

    Larch's packaged exampville_skims.omx ships only the raw Exampville
    matrices (AUTO_TIME / AUTO_DIST / WALK_DIST / BIKE_TIME / TRANSIT_*).
    The stage-5 estimator hard-codes canonical names in MODE_UEC/DEST_UEC
    (e.g. c_ivt_auto -> 'od.SOVTOLL_TIME__AM / 60.0') and reads the OMX file,
    NOT build_skim_df()'s in-memory frame.  Copying the package OMX verbatim
    left SOVTOLL_TIME__AM absent: _run_segment raised
    KeyError('od..SOVTOLL_TIME__AM'), both stage-5 mode baselines scored the
    -1e9 failure sentinel (printed as the bogus 'BIC/n=1000000000.0000'),
    and the E1/E2 arms reported ok=False (job 25929034).

    Returns {alias: matrix} for append_canonical_aliases() to write into a
    copy of the package OMX.
    """
    def _mat(nm):
        a = mats.get(nm)
        if a is None:
            return None
        a = np.asarray(a, dtype=float)
        return a if a.shape == (n, n) else None

    n = int(np.asarray(taz_ids).size) if taz_ids is not None and len(taz_ids) else 0
    if not n:
        n = int(next(iter(np.asarray(v) for v in mats.values())).shape[0])

    out = {}
    auto_t = _mat('AUTO_TIME')
    for k in ('SOVTOLL_TIME__AM', 'SOVTOLL_TIME__PM', 'SOVTOLL_TIME__EV',
              'SOVTOLL_TIME__MD'):
        out[k] = auto_t          # Exampville has no period matrices: reuse AM
    auto_d = _mat('AUTO_DIST')
    for k in ('SOVTOLL_DIST__AM', 'SOVTOLL_DIST__PM', 'SOVTOLL_DIST__EV',
              'SOVTOLL_DIST__MD'):
        out[k] = auto_d          # same: reuse AM
    out['DIST'] = auto_d        # canonical distance used by MODE_UEC/DEST_UEC
    out['DISTWALK'] = _mat('WALK_DIST')
    _bike = _mat('BIKE_TIME')
    out['DISTBIKE'] = (_bike / 4.0 if _bike is not None else None)
    out['WLK_LOC_WLK_TOTIVT__AM'] = _mat('TRANSIT_IVTT')
    out['WLK_LOC_DRV_TOTIVT__AM'] = _mat('TRANSIT_IVTT')
    # Zero toll/CRASH: Exampville has no pricing or crash matrices.  Written as
    # real zeros (not omitted) so downstream constant-column checks see them
    # and holdfast the terms instead of raising a missing-column KeyError.
    zeros = np.zeros((n, n), dtype=float)
    for k in ('SOVTOLL_TOLL__AM', 'SOVTOLL_TOLL__PM', 'SOVTOLL_TOLL__EV',
              'SOVTOLL_TOLL__MD', 'CRASH', 'BTOLL', 'VTOLL'):
        out[k] = zeros
    out = {k: v for k, v in out.items() if v is not None}
    if 'SOVTOLL_TIME__AM' not in out:
        raise RuntimeError(
            "Cannot build canonical Exampville aliases: AUTO_TIME missing "
            f"from {sorted(mats)}")
    return out


def copy_package_skims(skim_src, root):
    """Materialise the Exampville OMX into the artifact root (canonical names).

    Copies the package OMX VERBATIM (preserving its root attributes, which
    read_skims._is_h5py_omx keys on) and appends the canonical aliases the
    stage-5 estimator expects.  A verbatim copy lacks SOVTOLL_TIME__AM and
    fails stage 5 with a KeyError; rewriting the file from scratch drops
    OMX_VERSION and makes the whole skim read back as zero matrices.
    """
    if not skim_src or not os.path.exists(str(skim_src)):
        raise RuntimeError(f"Exampville OMX source not found: {skim_src}")
    dst = guard_path(root, 'exampville_skims.omx')
    shutil.copy2(str(skim_src), dst)
    try:
        mats, taz = _h5_omx_read_all(str(skim_src))
        aliases = write_canonical_skims(dst, mats, taz)
        append_canonical_aliases(dst, aliases, taz)
    except Exception as e:
        # Alias injection is best-effort: an unreadable package OMX still
        # leaves a valid verbatim copy, and downstream will fail loudly on the
        # missing column rather than here.
        import warnings as _w
        _w.warn(f"Exampville canonical skim aliases not written ({e!r}); "
                "stage 5 will likely fail on missing SOVTOLL_TIME__AM")
    return dst


def map_households(hh: pd.DataFrame) -> pd.DataFrame:
    """hh -> pipeline-canonical household/person-level fields."""
    h = hh.copy()
    out = pd.DataFrame(index=h.index)
    out['HHID'] = pd.to_numeric(h['HHID'], errors='coerce').astype('Int64')
    out['HOMETAZ'] = pd.to_numeric(h['HOMETAZ'], errors='coerce').astype('Int64')
    out['OTAZ'] = out['HOMETAZ']
    out['incp'] = pd.to_numeric(h['INCOME'], errors='coerce')
    out['vehrd'] = pd.to_numeric(h['N_VEHICLES'], errors='coerce')
    out['HHSIZE'] = pd.to_numeric(h['HHSIZE'], errors='coerce')
    out['nprd'] = out['HHSIZE']
    out['n_workers_hh'] = pd.to_numeric(h['N_WORKERS'], errors='coerce')
    return out


def map_persons(pp: pd.DataFrame, hh_mapped: pd.DataFrame) -> pd.DataFrame:
    """pp + households -> pipeline-canonical persons.

    Licence is DERIVED (AGE>=16, matching example-201 DA availability);
    Exampville records no licence. Bands are explicit dummies plus real 5yr
    age5p/AGEGROUP bands (cdap needs finite bands); age5p doubles as the
    pipeline default. SEXP is absent -> NaN -> pipeline neutral default.
    CDAP/stage-1 linkage (derived, documented assumptions):
      ANYSTOPS  = (N_TOURS > 0); MAINACT = 'Work' if N_WORK_TOURS>0 else
      ('other' if N_OTHER_TOURS>0 else '') — prepare_cdap_data() derives
      H/N/M from these two columns via MANDATORY_ACTS (capital-'Work'
      spelling is load-bearing). STUDYING = 0 (no school detail: under-16s
      fall in NonWorker; no SchoolChild/edu cells — matches the documented
      'no education detail' scope). weight = 1.0 (no expansion weights).
    """
    p = pp.copy()
    out = pd.DataFrame(index=p.index)
    out['person_id'] = pd.to_numeric(p['PERSONID'], errors='coerce').astype('Int64')
    out['HHID'] = pd.to_numeric(p['HHID'], errors='coerce').astype('Int64')
    age = pd.to_numeric(p['AGE'], errors='coerce')
    out['AGE'] = age
    # Real 5yr bands (1-18), NOT NaN: cdap._derive_ptype does int(age-band or
    # 6) and NaN is truthy, so NaN bands crash it (job 25332564). Same value
    # in AGEGROUP (cdap prefers it) and age5p (pipeline default).
    _band = np.ceil((age.fillna(27.0) + 1.0) / 5.0).clip(1, 18).astype(int)
    out['age5p'] = _band
    out['AGEGROUP'] = _band
    out['age_child'] = (age < 16).astype(float)
    out['age_teen'] = ((age >= 16) & (age <= 17)).astype(float)
    out['age_working'] = ((age >= 18) & (age <= 64)).astype(float)
    out['age_older'] = ((age >= 65) & (age <= 74)).astype(float)
    out['age_senior'] = (age >= 75).astype(float)
    out['CARLICENCE'] = (age >= 16).astype(float)
    works = pd.to_numeric(p['WORKS'], errors='coerce').fillna(0).astype(int)
    out['WORKSTAT'] = works
    out['workstat_fulltime'] = (works == 1).astype(float)
    out['workstat_parttime'] = 0.0
    out['workstat_notworking'] = (works == 0).astype(float)
    out['STUDYING'] = 0
    out['sexp'] = np.nan
    out['weight'] = 1.0
    _nt = pd.to_numeric(p.get('N_TOURS', 0), errors='coerce').fillna(0)
    _nw = pd.to_numeric(p.get('N_WORK_TOURS', 0), errors='coerce').fillna(0)
    _no = pd.to_numeric(p.get('N_OTHER_TOURS', 0), errors='coerce').fillna(0)
    out['N_TOURS'] = _nt
    out['N_WORK_TOURS'] = _nw
    out['N_OTHER_TOURS'] = _no
    out['ANYSTOPS'] = (_nt > 0)
    out['MAINACT'] = np.where(_nw > 0, 'Work',
                              np.where(_no > 0, 'other', ''))
    hcols = ['HHID', 'HOMETAZ', 'OTAZ', 'incp', 'vehrd', 'HHSIZE', 'nprd',
             'n_workers_hh']
    hsub = hh_mapped[[c for c in hcols if c in hh_mapped.columns]].copy()
    out = out.merge(hsub, on='HHID', how='left')
    return out


def map_landuse(emp: pd.DataFrame) -> pd.DataFrame:
    """Employment (40 TAZs) -> pipeline landuse frame (ORDER-indexed)."""
    e = emp.copy()
    zone_col = next((c for c in ('ORDER', 'TAZ', 'zone_id', 'ZONE')
                     if c in e.columns), None)
    if zone_col is None:
        e = e.reset_index()
        zone_col = next((c for c in ('ORDER', 'TAZ', 'zone_id', 'ZONE',
                                     'index') if c in e.columns),
                        e.columns[0])
    if zone_col != 'ORDER':
        e = e.rename(columns={zone_col: 'ORDER'})
    lu = pd.DataFrame(index=pd.to_numeric(e['ORDER'], errors='coerce').astype(int))
    lu.index.name = 'ORDER'
    tot = pd.to_numeric(e.get('TOTAL_EMP', 0), errors='coerce').fillna(0.0)
    ret = pd.to_numeric(e.get('RETAIL_EMP', 0), errors='coerce').fillna(0.0)
    nret = pd.to_numeric(e.get('NONRETAIL_EMP', 0), errors='coerce').fillna(0.0)
    lu['TOTAL_EMP'] = np.asarray(tot, dtype=float)
    lu['TOTEMP'] = np.asarray(tot, dtype=float)
    lu['RETEMPN'] = np.asarray(ret, dtype=float)
    lu['retail'] = np.asarray(ret, dtype=float)
    lu['FPSEMPN'] = np.asarray(nret, dtype=float)
    lu['HEREMPN'] = 0.0
    return lu


def read_omx_matrices(omx_handle, names):
    """Read named matrices -> {name: (zones, 2D array)} + zone list.

    Works with larch.OMX handles and openmatrix handles (layered try/except
    per access pattern). Raises RuntimeError listing what was found.
    """
    found, missing = {}, []
    taz_ids = None
    for accessor in ('TAZ_ID', 'taz_id', 'TAZ', 'taz', 'zones', 'ZONE'):
        try:
            _t = omx_handle[accessor]
            _t = np.asarray(_t[:] if hasattr(_t, '__getitem__') else _t)
            taz_ids = np.asarray(_t).astype(int).ravel().tolist()
            break
        except Exception:
            continue
    for nm in names:
        arr = None
        try:
            _m = omx_handle[nm]
            try:
                arr = np.asarray(_m[:, :], dtype=float)
            except Exception:
                arr = np.asarray(_m, dtype=float)
        except Exception:
            missing.append(nm)
            continue
        found[nm] = arr
    if not found:
        raise RuntimeError(f"Exampville OMX: no matrices readable "
                           f"(tried {names}; missing={missing})")
    if taz_ids is None:
        n = next(iter(found.values())).shape[0]
        taz_ids = list(range(1, n + 1))
    return found, taz_ids


def build_skim_df(mats: dict, taz_ids: list) -> pd.DataFrame:
    """OMX matrices -> pipeline long skim_df (OTAZ/DTAZ + canonical names).

    Every column the stage-5 utility expressions reference must be present:
    DEST_UEC/MODE_UEC hard-code these names, so an absent one surfaces as
    KeyError('od..<name>') deep inside _run_segment rather than at load time.
    """
    zones = [int(z) for z in taz_ids]
    n = len(zones)
    rec = {'OTAZ': np.repeat(zones, n), 'DTAZ': np.tile(zones, n)}

    def _mat(nm, default=np.nan):
        a = mats.get(nm)
        if a is None or np.asarray(a).shape != (n, n):
            return np.full(n * n, default)
        return np.asarray(a, dtype=float).ravel()

    auto_t = _mat('AUTO_TIME')
    for _k in ('SOVTOLL_TIME__AM', 'SOVTOLL_TIME__PM', 'SOVTOLL_TIME__EV',
               'SOVTOLL_TIME__MD'):
        rec[_k] = auto_t      # no period matrices in Exampville: reuse AM
    auto_d = _mat('AUTO_DIST')
    for _k in ('SOVTOLL_DIST__AM', 'SOVTOLL_DIST__PM', 'SOVTOLL_DIST__EV',
               'SOVTOLL_DIST__MD'):
        rec[_k] = auto_d      # same: reuse AM
    rec['DIST'] = auto_d
    rec['DISTWALK'] = _mat('WALK_DIST')
    bike_t = _mat('BIKE_TIME')
    with np.errstate(invalid='ignore'):
        rec['DISTBIKE'] = np.where(np.isfinite(bike_t), bike_t / 4.0, np.nan)
    for _k in ('WLK_LOC_WLK_TOTIVT__AM', 'WLK_LOC_DRV_TOTIVT__AM'):
        rec[_k] = _mat('TRANSIT_IVTT')
    rec['TRANSIT_FARE'] = _mat('TRANSIT_FARE')
    rec['AUTO_COST'] = _mat('AUTO_COST')
    # Zero toll/CRASH columns: Exampville has no pricing or crash matrices.
    # Emitted as real zeros so constant-column checks holdfast those terms
    # instead of the expression raising on a missing column.
    for _k in ('SOVTOLL_TOLL__AM', 'SOVTOLL_TOLL__PM', 'BTOLL', 'VTOLL',
               'CRASH'):
        rec[_k] = np.zeros(n * n)
    return pd.DataFrame(rec)


def _mat_lookup(mats, taz_ids, name, o, d, default=np.nan):
    try:
        a = np.asarray(mats[name], dtype=float)
        zones = [int(z) for z in taz_ids]
        pos = {z: i for i, z in enumerate(zones)}
        return float(a[pos[int(o)], pos[int(d)]])
    except Exception:
        return default


def map_tours(tour: pd.DataFrame, persons_mapped: pd.DataFrame,
              mats: dict, taz_ids: list) -> pd.DataFrame:
    """Tours -> pipeline-canonical tour frame (estimation-ready).

    tour_distance / DISTWALK / DISTBIKE from home->dest skim lookups.
    No STARTIME (absent in Exampville; pipeline defaults to AM peak and M3
    skips loudly). tour_mode keeps the 1-5 coding (identical to pipeline).
    """
    t = tour.copy()
    out = pd.DataFrame(index=t.index)
    out['tour_id'] = pd.to_numeric(t['TOURID'], errors='coerce').astype('Int64')
    out['ptour_id'] = out['tour_id'].astype(str)
    out['person_id'] = pd.to_numeric(t['PERSONID'], errors='coerce').astype('Int64')
    out['HHID'] = pd.to_numeric(t['HHID'], errors='coerce').astype('Int64')
    out['DTAZ'] = pd.to_numeric(t['DTAZ'], errors='coerce').astype('Int64')
    out['tour_mode'] = pd.to_numeric(t['TOURMODE'], errors='coerce').astype('Int64')
    purp = pd.to_numeric(t['TOURPURP'], errors='coerce')
    out['main_purpose'] = purp.map(PURPOSE_MAP)
    # home zone per tour for distance lookups
    pmap = dict(zip(persons_mapped['person_id'].astype(str),
                    pd.to_numeric(persons_mapped['OTAZ'],
                                  errors='coerce').astype('Int64').astype(str)))
    homes = out['person_id'].astype(str).map(pmap)
    out['OTAZ'] = pd.to_numeric(homes, errors='coerce').astype('Int64')
    dists, walks, bikes = [], [], []
    for o, d in zip(out['OTAZ'].tolist(), out['DTAZ'].tolist()):
        dists.append(_mat_lookup(mats, taz_ids, 'AUTO_DIST', o, d))
        walks.append(_mat_lookup(mats, taz_ids, 'WALK_DIST', o, d))
        _bt = _mat_lookup(mats, taz_ids, 'BIKE_TIME', o, d)
        bikes.append(_bt / 4.0 if np.isfinite(_bt) else np.nan)
    out['tour_distance'] = np.asarray(dists, dtype=float)
    out['DISTWALK'] = np.asarray(walks, dtype=float)
    out['DISTBIKE'] = np.asarray(bikes, dtype=float)
    out['weight'] = 1.0  # no expansion weights in Exampville
    return out


def map_synth_s4(tours_mapped: pd.DataFrame) -> pd.DataFrame:
    """Application schedule = ALL tours (full synthetic population use).

    Uniform AM-peak placement: Exampville carries no diary times, so every
    tour is placed at STARTIME=480/ENDTIME=540 (documented assumption; mode
    and destination choice do not identify on within-day timing here).
    """
    s4 = tours_mapped.copy()
    s4['STARTIME'] = float(APP_STARTIME)
    s4['ENDTIME'] = float(APP_ENDTIME)
    return s4


def write_manifest(root, info: dict):
    """SOURCE.json: provenance + schema map (artifact-separation proof)."""
    import json
    manifest = {'source': SOURCE_TAG, 'segments': SEGMENTS,
                'purpose_map': {str(k): v for k, v in PURPOSE_MAP.items()},
                'mode_coding': _PIPELINE_MODE_NAMES,
                'assumptions': [
                    'CARLICENCE derived as AGE>=16 (no licence in Exampville)',
                    'TOURPURP 2 routed as othdiscr (nonmand segment; no sub-purpose)',
                    'MAINACT Work/other derived from tour counts for CDAP labels',
                    'STUDYING=0 (no school detail; no edu/SchoolChild cells)',
                    'PM skims reuse AM (no PM matrix)',
                    'Uniform STARTIME=480/ENDTIME=540 application (no diary times)',
                    'Stages 2-4, 5b, M3, safety terms out of scope (no times/durations/stops/crash)',
                ]}
    manifest.update(info or {})
    with open(guard_path(root, 'SOURCE.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    return guard_path(root, 'SOURCE.json')
