#!/usr/bin/env python
"""
mtc_connect.py — Prototype-MTC data-source adapter for the Stage-5 pipeline.
================================================================================
Loads ActivitySim's canonical prototype_mtc example (Bay Area prototype:
households/persons/land_use/skims + the regress final_trips.csv standard-model
outputs) and maps it onto this pipeline's Stage-5 frames (tours / persons /
landuse / skim_df), mirroring exampville_connect.py design rules:

1. Artifact separation: every path lives under MTC_ROOT (default
   runs/mtc_ab, overridable via env or arg). guard_path() refuses any path
   escaping the root. NOTHING is written to saved_data/, runs/ga_parallel/,
   runs/exampville_ab/ or any QLD tree.
2. Download-once: MTC files fetch from raw.githubusercontent.com on first run
   (needs egress; pre-download on an egress node otherwise) into
   <root>/source/. Reruns reuse the local copies.
3. Scope honesty (also stated in the paper): prototype_mtc ships trips without
   diaries/times/durations/stops/crash. Transfer coverage: Stage 0 CDAP
   (labels derived from tour counts), Stage 1 frequency (counts derived from
   tours; work/edu/nonmand cells), Stage 5 primary mode + destination.
   Out of scope by construction: stages 2-4 (need diary times / durations),
   5b (needs stops), M3 (needs times), safety terms (nosafety).

Column mapping (prototype_mtc -> pipeline canonical)
----------------------------------------------------
hh:  HHID, TAZ->HOMETAZ/OTAZ, income($)->incp(band 1-14, fixed log edges),
     VEHICL->vehrd, PERSONS->HHSIZE/nprd, workers->n_workers_hh
pp:  PERID->person_id, household_id->HHID, age (bands derived; age5p banded),
     sex->sexp, HOURS/WEEKS/EARNS->WORKSTAT (documented thresholds),
     GRADE/age->STUDYING, CARLICENCE=(age>=16)
tour (synthesized from regress trips, grouped by tour_id):
     origin=first outbound origin, DTAZ=main-leg destination,
     modelled purpose->main_purpose (work/edu/nonmand), collapsed 1-5 tour_mode
skims (MTC skims.omx, discovered at runtime): best-effort canonical mapping,
     PM reuses AM where absent (documented); hard error listing available
     matrices when core time/distance slots are unresolvable.
emp: TAZ->ORDER, TOTEMP/RETEMPN/FPSEMPN/HEREMPN/OTHEMPN/AGREMPN/MWTEMPN,
     TOTPOP/TOTHH/HSENROLL direct; Education=COLLFTE+COLLPTE;
     residential=RESACRE, commercial=CIACRE (documented proxies);
     Primary/Secondary/retail/EATOUT/Entertainment absent -> 0 downstream.
Reference: regress/final_trips.csv = ActivitySim standard-model outputs (M0).
"""
import os
import shutil
import urllib.request

import numpy as np
import pandas as pd

SOURCE_TAG = 'mtc'
SEGMENTS = ['work', 'edu', 'nonmand']
# MTC trip purposes -> pipeline segments ('othdiscr' routes to nonmand, as in
# the Exampville adapter; MTC has no finer nonmand sub-purpose split here).
PURPOSE_MAP = {
    'work': 'work', 'school': 'edu', 'univ': 'edu',
    'escort': 'nonmand', 'shopping': 'nonmand', 'othdiscr': 'nonmand',
    'othmaint': 'nonmand', 'eatout': 'nonmand', 'social': 'nonmand',
    'atwork': 'nonmand',
}
APP_STARTIME = 480.0   # uniform application placement (documented assumption)
APP_ENDTIME = 540.0

_PIPELINE_MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}

MTC_GITHUB_BASE = ('https://raw.githubusercontent.com/ActivitySim/activitysim'
                   '/main/activitysim/examples/prototype_mtc')
MTC_DATA_URLS = {
    'households.csv': f'{MTC_GITHUB_BASE}/data/households.csv',
    'persons.csv': f'{MTC_GITHUB_BASE}/data/persons.csv',
    'land_use.csv': f'{MTC_GITHUB_BASE}/data/land_use.csv',
    'skims.omx': f'{MTC_GITHUB_BASE}/data/skims.omx',
    'regress_final_trips.csv': f'{MTC_GITHUB_BASE}/test/regress/final_trips.csv',
}

# Fixed log edges mapping household income ($) -> incp band 1-14.
# Deterministic across runs (no quantiles); mirrors the banded semantics the
# pipeline's c_incp_* terms expect. Edges span $1 .. $1M in log space.
_INCP_LOG_EDGES = np.linspace(0.0, 6.0, 15)


def mtc_root(outdir=None):
    """Resolve + create the artifact root. Refuses non-mtc roots."""
    root = os.path.abspath(outdir or os.environ.get(
        'MTC_ROOT', os.path.join('runs', 'mtc_ab')))
    if 'mtc' not in os.path.basename(root).lower():
        raise RuntimeError(
            f"Artifact separation: MTC_ROOT must contain 'mtc', got {root}")
    os.makedirs(root, exist_ok=True)
    return root


def guard_path(root, *parts):
    """Join under root; raise if the result escapes the root."""
    root = os.path.abspath(root)
    p = os.path.abspath(os.path.join(root, *parts))
    if p != root and not p.startswith(root + os.sep):
        raise RuntimeError(f"Artifact separation: {p} escapes {root}")
    return p


def download_mtc_data(root, timeout=120):
    """Fetch MTC source files once into <root>/source/; return {name: path}.

    Skips files already present AND valid. Every file is size/row validated
    (job 25295180: regress_final_trips.csv silently truncated to 80 lines by
    a dropped connection — urllib reports success on a clean FIN). Mismatches
    retry up to 3x, then raise naming the URL (pre-download on an egress node
    if compute nodes lack internet).
    """
    # name -> minimum valid content (rows for CSVs, bytes for binaries).
    # NOTE: the prototype_mtc regress fixture is tiny BY DESIGN (33-tour
    # test set, ~80 rows) — a 5000-row threshold here once made every run
    # delete the valid file and refetch-loop to failure (job 25329144).
    _MIN_ROWS = {'households.csv': 1000, 'persons.csv': 1000,
                 'land_use.csv': 10, 'regress_final_trips.csv': 10}
    _MIN_BYTES = {'skims.omx': 1_000_000}
    srcdir = guard_path(root, 'source')
    os.makedirs(srcdir, exist_ok=True)

    def _valid(name, dst):
        try:
            if os.path.getsize(dst) == 0:
                return False
            if name in _MIN_BYTES:
                return os.path.getsize(dst) >= _MIN_BYTES[name]
            with open(dst, 'r', encoding='utf-8', errors='replace') as f:
                _n = sum(1 for _ in f)
            return _n >= _MIN_ROWS.get(name, 2)
        except Exception:
            return False

    paths = {}
    for name, url in MTC_DATA_URLS.items():
        dst = os.path.join(srcdir, name)
        if os.path.exists(dst) and _valid(name, dst):
            paths[name] = dst
            continue
        if os.path.exists(dst):
            os.remove(dst)  # stale/truncated — refetch
        _err = None
        for _attempt in (1, 2, 3):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r, \
                        open(dst, 'wb') as f:
                    shutil.copyfileobj(r, f)
                if _valid(name, dst):
                    break
                _err = 'downloaded file failed validation'
            except Exception as e:
                _err = e
        else:
            raise RuntimeError(
                f"MTC download failed for {url} ({_err}) — pre-download on an "
                f"egress node into {srcdir}/ and rerun")
        paths[name] = dst
    return paths


def map_households(hh: pd.DataFrame) -> pd.DataFrame:
    """MTC households -> pipeline-canonical household fields."""
    h = hh.copy()
    out = pd.DataFrame(index=h.index)
    out['HHID'] = pd.to_numeric(h['HHID'], errors='coerce').astype('Int64')
    out['HOMETAZ'] = pd.to_numeric(h['TAZ'], errors='coerce').astype('Int64')
    out['OTAZ'] = out['HOMETAZ']
    _inc = pd.to_numeric(h['income'], errors='coerce').fillna(0.0).clip(lower=0.0)
    _log = np.log10(_inc + 1.0).to_numpy()
    out['incp'] = np.clip(np.digitize(_log, _INCP_LOG_EDGES[1:-1]) + 1, 1, 14).astype(int)
    out['vehrd'] = pd.to_numeric(h['VEHICL'], errors='coerce')
    out['HHSIZE'] = pd.to_numeric(h['PERSONS'], errors='coerce')
    out['nprd'] = out['HHSIZE']
    out['n_workers_hh'] = pd.to_numeric(h['workers'], errors='coerce')
    return out


def map_persons(pp: pd.DataFrame, hh_mapped: pd.DataFrame) -> pd.DataFrame:
    """MTC persons + households -> pipeline-canonical persons.

    WORKSTAT derives from observed work intensity (HOURS/WEEKS/EARNS), not the
    MTC ptype code (whose semantics vary by example vintage): full-time =
    employed at HOURS>=30 (or employed with missing hours); part-time =
    employed below that; else not working. STUDYING = school-age (5-17) with
    GRADE>0, else 0. Licence derived as AGE>=16 (MTC records no licence);
    sexp from sex (1/2); weight = 1.0 (no expansion weights). CDAP linkage
    mirrors the Exampville adapter (N_TOURS/N_WORK_TOURS filled in by
    attach_tour_counts() after tour synthesis).
    """
    p = pp.copy()
    out = pd.DataFrame(index=p.index)
    out['person_id'] = pd.to_numeric(p['PERID'], errors='coerce').astype('Int64')
    out['HHID'] = pd.to_numeric(p['household_id'], errors='coerce').astype('Int64')
    age = pd.to_numeric(p['age'], errors='coerce')
    out['AGE'] = age
    out['age5p'] = np.ceil((age + 1.0) / 5.0).clip(1, 18)
    out['age_child'] = (age < 16).astype(float)
    out['age_teen'] = ((age >= 16) & (age <= 17)).astype(float)
    out['age_working'] = ((age >= 18) & (age <= 64)).astype(float)
    out['age_older'] = ((age >= 65) & (age <= 74)).astype(float)
    out['age_senior'] = (age >= 75).astype(float)
    out['CARLICENCE'] = (age >= 16).astype(float)
    _hours = pd.to_numeric(p.get('HOURS'), errors='coerce')
    _weeks = pd.to_numeric(p.get('WEEKS'), errors='coerce')
    _earns = pd.to_numeric(p.get('EARNS'), errors='coerce')
    _employed = ((_hours.fillna(0) > 0) | (_weeks.fillna(0) > 0) | (_earns.fillna(0) > 0))
    _ft = (_employed & (_hours.fillna(99) >= 30)).astype(float)
    _pt = (_employed & (_hours.fillna(0) < 30)).astype(float)
    out['WORKSTAT'] = np.where(_ft > 0, 1, np.where(_pt > 0, 2, 0)).astype(int)
    out['workstat_fulltime'] = _ft
    out['workstat_parttime'] = _pt
    out['workstat_notworking'] = (~_employed).astype(float)
    _grade = pd.to_numeric(p.get('GRADE'), errors='coerce').fillna(0)
    out['STUDYING'] = (((age >= 5) & (age <= 17)) & (_grade > 0)).astype(int)
    out['sexp'] = pd.to_numeric(p.get('sex'), errors='coerce')
    out['weight'] = 1.0  # no expansion weights in prototype_mtc (KeyError 'weight' otherwise)
    out['weight'] = 1.0
    for _c in ('N_TOURS', 'N_WORK_TOURS', 'N_OTHER_TOURS'):
        out[_c] = 0.0
    out['ANYSTOPS'] = False
    out['MAINACT'] = ''
    hcols = ['HHID', 'HOMETAZ', 'OTAZ', 'incp', 'vehrd', 'HHSIZE', 'nprd',
             'n_workers_hh']
    hsub = hh_mapped[[c for c in hcols if c in hh_mapped.columns]].copy()
    out = out.merge(hsub, on='HHID', how='left')
    return out


def map_landuse(lu: pd.DataFrame) -> pd.DataFrame:
    """MTC land_use (TAZ-keyed) -> pipeline landuse frame (ORDER-indexed).

    Documented proxies: Education=COLLFTE+COLLPTE (no primary/secondary split;
    HSENROLL kept direct); residential=RESACRE, commercial=CIACRE (acreage
    proxies — no floor-area/count columns in MTC land_use); Primary/Secondary/
    retail/EATOUT/Entertainment absent -> 0 downstream (documented; weakens
    nonmand attractors on MTC by construction).
    """
    e = lu.copy()

    def _num(col, default=0.0):
        if col in e.columns:
            return pd.to_numeric(e[col], errors='coerce').fillna(default).astype(float).values
        return np.full(len(e), default, dtype=float)

    zones = pd.to_numeric(e['TAZ'], errors='coerce').astype(int).values
    out = pd.DataFrame(index=zones)
    out.index.name = 'ORDER'
    for _c in ('TOTEMP', 'RETEMPN', 'FPSEMPN', 'HEREMPN', 'OTHEMPN',
               'AGREMPN', 'MWTEMPN', 'TOTPOP', 'TOTHH', 'HSENROLL'):
        out[_c] = _num(_c)
    out['Education'] = _num('COLLFTE') + _num('COLLPTE')
    out['Primary'] = 0.0
    out['Secondary'] = 0.0
    out['residential'] = _num('RESACRE')
    out['commercial'] = _num('CIACRE')
    out['retail'] = 0.0
    out['Entertainment'] = 0.0
    out['EATOUT'] = 0.0
    out['PRKCST'] = _num('PRKCST')
    out['area_type'] = _num('area_type')
    # Zone area for employment-density (log_emp_density): MTC reports TOTACRE
    # in acres — convert to sqkm so the density scale matches the pipeline's
    # SEQ convention. Without this every MTC zone gets density 0 (constant →
    # auto-holdfasted emp-density terms).
    out['sa2_area_sqkm'] = (_num('TOTACRE') * 0.00404686).astype(float)
    out['sa2_area'] = out['sa2_area_sqkm']
    return out


def collapse_mtc_mode(mode) -> int:
    """MTC trip_mode string -> pipeline 1-5 tour-mode group (documented rule).

    ActivitySim MTC modes carry the transit access as a suffix (WALK_LOC /
    WALK_LRF / WALK_HVY = walk access to local/rapid/heavy transit), so any
    suffixed WALK* or DRIVE_TRANSIT* leg is transit; bare WALK is walk.
    DRIVEALONE* -> DA; SHARED*/TNC* -> SR; BIKE* -> bike. Unknown -> SR
    (counted by the caller for the record).
    """
    try:
        s = str(mode).strip().upper()
    except Exception:
        return 2
    if not s or s in ('NAN', 'NONE'):
        return 2
    if 'TRANSIT' in s:
        return 5
    if s.startswith('WALK') and s != 'WALK':
        return 5
    if s.startswith('DRIVEALONE'):
        return 1
    if s.startswith('SHARED') or s.startswith('TNC'):
        return 2
    if s.startswith('BIKE'):
        return 4
    if s == 'WALK':
        return 3
    return 2


def tour_mode_from_legs(modes) -> tuple:
    """Main-mode rule over a tour's trip legs -> (mode_1_5, n_unmapped).

    Precedence: transit > DA > SR > bike > walk (a tour with any transit leg
    is a transit tour). Bare unknown strings fall to SR and are counted.
    """
    order = []
    n_unmapped = 0
    for _m in list(modes):
        _g = collapse_mtc_mode(_m)
        order.append(_g)
        try:
            _s = str(_m).strip().upper()
            if _g == 2 and not (_s.startswith('SHARED') or _s.startswith('TNC')):
                n_unmapped += 1
        except Exception:
            pass
    if 5 in order:
        return 5, n_unmapped
    if 1 in order:
        return 1, n_unmapped
    if 2 in order:
        return 2, n_unmapped
    if 4 in order:
        return 4, n_unmapped
    if 3 in order:
        return 3, n_unmapped
    return 2, n_unmapped


def build_tours_from_regress(trips: pd.DataFrame,
                             persons_mapped: pd.DataFrame) -> pd.DataFrame:
    """Regress trips grouped by tour_id -> pipeline-canonical tour frame.

    Origin = first outbound leg origin; destination = first outbound
    non-home-purpose leg destination (fallback: last outbound destination);
    purpose = that leg's purpose mapped to work/edu/nonmand; return-home legs
    (purpose == 'home') never seed a destination. Home zone from the person's
    household. tour_distance filled later from skims by the caller.
    """
    t = trips.copy()
    t['tour_id'] = pd.to_numeric(t['tour_id'], errors='coerce')
    t = t[t['tour_id'].notna()].copy()
    t['tour_id'] = t['tour_id'].astype('int64')
    t['_outbound'] = t['outbound'].astype(str).str.lower().isin(['true', '1'])
    t['_purpose'] = t['purpose'].fillna('').astype(str).str.lower()
    _pmap = dict(zip(persons_mapped['person_id'].astype(str),
                     persons_mapped['HHID'].astype(str)))
    _hmap = dict(zip(persons_mapped['HHID'].astype(str),
                     pd.to_numeric(persons_mapped['OTAZ'],
                                   errors='coerce').astype('Int64').astype(str)))
    rows, n_unmapped_modes = [], 0
    for _tid, _g in t.groupby('tour_id', sort=False):
        _g = _g.sort_values(['trip_num'])
        _pid = str(_g['person_id'].iloc[0])
        _out = _g[_g['_outbound']]
        if len(_out) == 0:
            _out = _g
        _origin = pd.to_numeric(_out['origin'], errors='coerce').dropna()
        _origin = int(_origin.iloc[0]) if len(_origin) else 0
        _legs = _out[~_out['_purpose'].isin(['home', ''])]
        if len(_legs) == 0:
            _legs = _out
        _dest = pd.to_numeric(_legs['destination'], errors='coerce').dropna()
        _dest = int(_dest.iloc[0]) if len(_dest) else 0
        _purp = str(_legs['primary_purpose'].iloc[0]).lower() \
            if 'primary_purpose' in _legs.columns else str(_legs['_purpose'].iloc[0])
        _mode, _nu = tour_mode_from_legs(_g['trip_mode'].tolist())
        n_unmapped_modes += _nu
        _hhid = _pmap.get(_pid, '')
        rows.append({
            'tour_id': int(_tid), 'ptour_id': str(int(_tid)),
            'person_id': _pid, 'HHID': _hhid,
            'OTAZ': pd.to_numeric(_hmap.get(str(_hhid), np.nan), errors='coerce'),
            'DTAZ': _dest, 'tour_mode': _mode,
            'main_purpose': PURPOSE_MAP.get(_purp, 'nonmand'),
            'weight': 1.0,
        })
    out = pd.DataFrame(rows)
    out['OTAZ'] = pd.to_numeric(out['OTAZ'], errors='coerce').astype('Int64')
    out['DTAZ'] = pd.to_numeric(out['DTAZ'], errors='coerce').astype('Int64')
    out.attrs['n_unmapped_modes'] = int(n_unmapped_modes)
    return out


def attach_tour_counts(persons_mapped: pd.DataFrame,
                       tours_mapped: pd.DataFrame) -> pd.DataFrame:
    """Derive N_TOURS/N_WORK_TOURS/ANYSTOPS/MAINACT from synthesized tours."""
    p = persons_mapped.copy()
    _nt = tours_mapped.groupby('person_id').size()
    _nw = tours_mapped[tours_mapped['main_purpose'] == 'work'].groupby('person_id').size()
    _no = tours_mapped[tours_mapped['main_purpose'] != 'work'].groupby('person_id').size()
    _key = p['person_id'].astype(str)
    p['N_TOURS'] = _key.map(_nt).fillna(0).astype(float)
    p['N_WORK_TOURS'] = _key.map(_nw).fillna(0).astype(float)
    p['N_OTHER_TOURS'] = _key.map(_no).fillna(0).astype(float)
    p['ANYSTOPS'] = (p['N_TOURS'] > 0)
    p['MAINACT'] = np.where(p['N_WORK_TOURS'] > 0, 'Work',
                            np.where(p['N_OTHER_TOURS'] > 0, 'other', ''))
    return p


def discover_skim_matrices(omx_handle) -> list:
    """List matrix names in an open OMX handle (openmatrix or larch)."""
    for _meth in ('listMatrices', 'list_matrices'):
        try:
            return [str(n) for n in getattr(omx_handle, _meth)()]
        except Exception:
            continue
    try:
        return [str(k) for k in omx_handle.keys()]
    except Exception as e:
        raise RuntimeError(f"MTC OMX: cannot list matrices ({e})")


def _norm_name(s: str) -> str:
    return ''.join(c for c in str(s).upper() if c.isalnum())


def pick_matrix(names: list, *token_groups: list, slot: str,
                required: bool = True) -> str:
    """Best-effort matrix pick: first name containing ALL tokens of the first
    matching group. Raises listing available matrices when required and lost."""
    _normed = [(_n, _norm_name(_n)) for _n in names]
    for _grp in token_groups:
        _gt = [_norm_name(_g) for _g in _grp]
        for _raw, _nn in _normed:
            if all(_t in _nn for _t in _gt):
                return _raw
    if required:
        raise RuntimeError(
            f"MTC skims: no matrix for slot '{slot}' (tried {list(token_groups)}; "
            f"available={names}) — extend the slot map in mtc_connect.py")
    return None


def read_omx_matrices(omx_handle, names):
    """Read named matrices -> {name: 2D array} (openmatrix or larch handles)."""
    found = {}
    for nm in names:
        try:
            _m = omx_handle[nm]
            try:
                found[nm] = np.asarray(_m[:, :], dtype=float)
            except Exception:
                found[nm] = np.asarray(_m, dtype=float)
        except Exception as e:
            raise RuntimeError(f"MTC OMX: cannot read matrix '{nm}' ({e})")
    return found


def zone_ids_from_omx(omx_handle, n: int) -> list:
    """TAZ ids from the OMX mapping layer; fallback 1..n."""
    for accessor in ('TAZ_ID', 'taz_id', 'TAZ', 'taz', 'zones', 'ZONE'):
        try:
            _t = omx_handle[accessor]
            _t = np.asarray(_t[:] if hasattr(_t, '__getitem__') else _t)
            return np.asarray(_t).astype(int).ravel().tolist()
        except Exception:
            continue
    try:
        _mp = omx_handle.mapping(next(iter(omx_handle.listMatrices())))
        return sorted(int(z) for z in _mp)
    except Exception:
        pass
    return list(range(1, int(n) + 1))


def build_skim_df(mats: dict, taz_ids: list, picked: dict) -> pd.DataFrame:
    """Picked MTC matrices -> pipeline long skim_df (canonical names).

    picked: {slot: matrix_name} with slots SOV_TIME, PM_TIME, DIST, DISTWALK,
    DISTBIKE, IVTT (only SOV_TIME/DIST required; rest NaN-tolerant).
    PM reuses AM where absent (documented).
    """
    zones = [int(z) for z in taz_ids]
    n = len(zones)
    rec = {'OTAZ': np.repeat(zones, n), 'DTAZ': np.tile(zones, n)}

    def _mat(slot, default=np.nan):
        _nm = picked.get(slot)
        _a = mats.get(_nm) if _nm else None
        if _a is None or np.asarray(_a).shape != (n, n):
            return np.full(n * n, default)
        return np.asarray(_a, dtype=float).ravel()

    _am = _mat('SOV_TIME')
    rec['SOVTOLL_TIME__AM'] = _am
    rec['SOVTOLL_TIME__PM'] = _mat('PM_TIME', default=np.nan)
    _pm = rec['SOVTOLL_TIME__PM']
    _pm[~np.isfinite(_pm)] = _am[~np.isfinite(_pm)]
    rec['SOVTOLL_TIME__PM'] = _pm
    rec['DIST'] = _mat('DIST')
    rec['DISTWALK'] = _mat('DISTWALK')
    rec['DISTBIKE'] = _mat('DISTBIKE')
    rec['WLK_LOC_WLK_TOTIVT__AM'] = _mat('IVTT')
    rec['TRANSIT_FARE'] = _mat('FARE', default=0.0)
    return pd.DataFrame(rec)


def add_tour_distances(tours_mapped: pd.DataFrame,
                       skim_df: pd.DataFrame) -> pd.DataFrame:
    """tour_distance/DISTWALK/DISTBIKE from home->dest skim lookups."""
    t = tours_mapped.copy()
    _key = dict(zip(zip(skim_df['OTAZ'].astype(int), skim_df['DTAZ'].astype(int)),
                    range(len(skim_df))))
    _dist = skim_df['DIST'].to_numpy(dtype=float)
    _walk = skim_df['DISTWALK'].to_numpy(dtype=float) \
        if 'DISTWALK' in skim_df.columns else np.full(len(skim_df), np.nan)
    _bike = skim_df['DISTBIKE'].to_numpy(dtype=float) \
        if 'DISTBIKE' in skim_df.columns else np.full(len(skim_df), np.nan)

    def _look(arr, o, d):
        try:
            return float(arr[_key[(int(o), int(d))]])
        except Exception:
            return np.nan

    _o = pd.to_numeric(t['OTAZ'], errors='coerce')
    _d = pd.to_numeric(t['DTAZ'], errors='coerce')
    t['tour_distance'] = [_look(_dist, o, d) for o, d in zip(_o, _d)]
    t['DISTWALK'] = [_look(_walk, o, d) for o, d in zip(_o, _d)]
    t['DISTBIKE'] = [_look(_bike, o, d) for o, d in zip(_o, _d)]
    return t


def map_synth_s4(tours_mapped: pd.DataFrame) -> pd.DataFrame:
    """Application schedule = ALL tours (full-population use).

    Uniform AM-peak placement: prototype_mtc carries no diary times, so every
    tour is placed at STARTIME=480/ENDTIME=540 (documented assumption).
    """
    s4 = tours_mapped.copy()
    s4['STARTIME'] = float(APP_STARTIME)
    s4['ENDTIME'] = float(APP_ENDTIME)
    return s4


def write_manifest(root, info: dict):
    """SOURCE.json: provenance + schema map (artifact-separation proof)."""
    import json
    manifest = {'source': SOURCE_TAG, 'segments': SEGMENTS,
                'purpose_map': dict(PURPOSE_MAP),
                'mode_coding': _PIPELINE_MODE_NAMES,
                'reference': 'ActivitySim prototype_mtc regress/final_trips.csv '
                             '(standard-model outputs, M0 arm)',
                'assumptions': [
                    'Tours synthesized from regress trips grouped by tour_id '
                    '(no tour table in prototype_mtc)',
                    'Tour mode by main-mode rule: transit > DA > SR > bike > walk',
                    'MTC WALK_* suffixed legs are transit access (ActivitySim '
                    'convention), bare WALK is walk',
                    'incp bands 1-14 from fixed log-$ edges (MTC income is $)',
                    'WORKSTAT from HOURS/WEEKS/EARNS (>=30h = full-time)',
                    'STUDYING = school-age with GRADE>0; CARLICENCE = age>=16',
                    'Education=COLLFTE+COLLPTE; residential=RESACRE, '
                    'commercial=CIACRE (acreage proxies)',
                    'Primary/Secondary/retail/EATOUT/Entertainment absent -> 0 '
                    '(weakens MTC nonmand attractors by construction)',
                    'PM skims reuse AM where absent',
                    'Uniform STARTIME=480/ENDTIME=540 application (no diary times)',
                    'Stages 2-4, 5b, M3, safety terms out of scope '
                    '(no times/durations/stops/crash)',
                ]}
    manifest.update(info or {})
    with open(guard_path(root, 'SOURCE.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    return guard_path(root, 'SOURCE.json')
