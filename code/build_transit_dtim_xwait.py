#!/usr/bin/env python
"""
build_transit_dtim_xwait.py — exact drive-egress time + transfer-wait skims.
================================================================================
Fills the three transit-LOS matrices the 371-zone OMX lacks (verified absent):

    WLK_LOC_DRV_DTIM__AM    drive-egress leg minutes (park-and-ride egress)
    WLK_LOC_DRV_DDIST__AM   drive-egress leg distance in km (same CAR legs)
    WLK_LOC_WLK_XWAIT__AM   transfer-wait minutes on walk-access paths

Method: r5py DetailedItineraries (per-segment mode/duration/wait) instead of
the TravelTimeMatrix aggregate used by build_transit_gtfs_skims.py:
  * XWAIT = sum of wait_time over transit segments AFTER the first boarding
    (option 0 = best itinerary). Restricted to ODs with BOARDS>1 in the
    existing matrix — single-boarding ODs have XWAIT=0 by construction, which
    skips most of the grid.
  * DTIM  = sum of CAR-mode segment durations with access=[WALK],
    egress=[CAR]. DDIST = sum of the same CAR segments' `distance` (metres)
    converted to km — the r5py frame carries both, so the two always build
    together (the stage-5 drive_access_km feature divides nothing, so the
    matrix must be km; its fallback is a flat 8.0). A spread-origin probe
    runs first: if R5 yields no CAR-egress legs (unsupported engine path),
    DTIM+DDIST are skipped loudly and the pipeline keeps its TOTIVT proxy —
    no silent garbage.

Performance: itineraries cost ~1 s+/OD. The full 137k grid does NOT fit one
slot: origins run in batches with per-batch parquet checkpoints
(DTIM_WORK_DIR) and the job simply resubmits to resume (completed origins are
skipped). Expect ~2 chained 48 h slots for the full build; ORIGIN_STEP>1 (or
DTIM_ONLY/XWAIT_ONLY) for smoke tests — a strided grid never publishes to the
OMX (exit 3 by design; ALLOW_PARTIAL_PUBLISH=1 overrides, e.g. against a copy).

Safety: the OMX is timestamp-backuped before any write. Matrices are replaced
atomically (del + assign). The pipeline consumes them automatically:
_build_od_dataset prefers real matrices over its XWAIT derivation, and the
c_drv_transit term prefers DTIM over its TOTIVT proxy when present.

Usage:
    python build_transit_dtim_xwait.py            # full build (all three)
    DTIM_ONLY=1 python build_transit_dtim_xwait.py    # DTIM + DDIST only
    XWAIT_ONLY=1 python build_transit_dtim_xwait.py   # XWAIT only
    ORIGIN_STEP=20 python build_transit_dtim_xwait.py   # smoke (~19 origins):
        # validates JVM heap + routing/checkpoints, publishes NO OMX, exits 3

Env: OMX_PATH, GTFS_PATH, OSM_PATH, CENTROIDS_PATH (same defaults as
     build_transit_gtfs_skims.py), DEPARTURE_HOUR (8), DTIM_WORK_DIR,
     ORIGIN_STEP (1), BATCH_ORIGINS (6), MAX_TIME_HRS (4).
     PROBE_ORIGINS (6), MAX_TIME_WALKING_MIN (60), MAX_TIME_DRIVING_MIN (45).

Memory: JVM_MEMORY (48G) sizes the r5 Java heap. r5py ignores
     JAVA_TOOL_OPTIONS and starts the JVM itself with -Xmx = 80% of node RAM
     (r5py/util/jvm.py), so a 128 GB slot got a ~102 GB heap and job 25395626
     was cgroup-OOM-killed at 125 GB. We therefore inject --max-memory into
     r5py's own argument parser before the first `import r5py` (which starts
     the JVM as a side effect) and fail loudly if r5py did not honour it.
"""
import os
import sys
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('dtim_xwait')

_HERE = os.path.dirname(os.path.abspath(__file__))
OMX_PATH = os.environ.get('OMX_PATH', os.path.join(_HERE, '..', 'demand_', 'skims_371.omx'))
GTFS_PATH = os.environ.get('GTFS_PATH', os.path.join(_HERE, 'saved_data', 'osm_cache', 'seq_gtfs.zip'))
OSM_PATH = os.environ.get('OSM_PATH', os.path.join(_HERE, 'saved_data', 'osm_cache', 'seq_osm.pbf'))
CENTROIDS_PATH = os.environ.get('CENTROIDS_PATH', os.path.join(_HERE, 'saved_data', 'osm_cache', 'zone_centroids.csv'))
WORK_DIR = os.environ.get('DTIM_WORK_DIR', os.path.join(_HERE, 'saved_data', 'dtim_xwait_work'))
DEPARTURE_HOUR = int(os.environ.get('DEPARTURE_HOUR', '8'))
ORIGIN_STEP = int(os.environ.get('ORIGIN_STEP', '1'))
# Small batches: one DetailedItineraries frame holds batch x 371 ODs x options
# x segments (4 h window, dense SEQ GTFS). 19-origin frames OOMed a 64 GB slot
# during the probe (job 25329147: 65.4 GB RSS in 5 min) — 6 keeps frames small;
# checkpoints make resume cheap, so smaller batches only cost JVM call overhead.
BATCH_ORIGINS = int(os.environ.get('BATCH_ORIGINS', '6'))
PROBE_ORIGINS = int(os.environ.get('PROBE_ORIGINS', '6'))
MAX_TIME_HRS = float(os.environ.get('MAX_TIME_HRS', '4'))
# Per-leg ceilings. RegionalTask defaults max_time_walking / max_time_driving
# to max_time, i.e. 4 h of walking to reach a stop and 4 h of driving to leave
# one. That is what made R5's search state space explode (job 25395626: 125 GB
# RSS before the cgroup OOM kill). These bound only the access/egress legs —
# the 4 h total trip budget (MAX_TIME_HRS, which matches the TOTIVT skim build)
# is unchanged, so coverage stays comparable.
MAX_TIME_WALKING_MIN = float(os.environ.get('MAX_TIME_WALKING_MIN', '60'))
MAX_TIME_DRIVING_MIN = float(os.environ.get('MAX_TIME_DRIVING_MIN', '45'))
DTIM_ONLY = os.environ.get('DTIM_ONLY', '0') == '1'
XWAIT_ONLY = os.environ.get('XWAIT_ONLY', '0') == '1'

# r5py sizes the JVM heap itself (80% of node RAM) and ignores
# JAVA_TOOL_OPTIONS for -Xmx precedence. Keep both in sync with JVM_MEMORY so
# any Java helper process agrees with the heap we actually request.
JVM_MEMORY = os.environ.get('JVM_MEMORY', '48G')

# Between-batch RSS ceiling. Hitting the cgroup limit is an instant SIGKILL that
# loses the batch in flight; stopping at a batch boundary keeps every completed
# checkpoint and lets a resubmit resume. Must stay below the slot's mem= limit.
MAX_RSS_GB = float(os.environ.get('MAX_RSS_GB', '100'))

os.environ['JAVA_TOOL_OPTIONS'] = f"-Xmx{JVM_MEMORY}"

import numpy as np
import pandas as pd
import gc


def _rss_gb():
    """Peak RSS in GB (Linux ru_maxrss is KB). Logs only — never raises."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return float('nan')


def _cur_rss_gb():
    """Current RSS in GB (/proc/self/statm). NaN when unavailable.

    The batch guard must read *current* RSS: ru_maxrss is monotonic, so one
    freed spike would otherwise abort every remaining batch.
    """
    try:
        with open('/proc/self/statm') as _f:
            _resident_pages = int(_f.read().split()[1])
        return _resident_pages * os.sysconf('SC_PAGE_SIZE') / 1e9
    except Exception:
        return float('nan')


def _minutes(x):
    """Coerce an r5py duration (timedelta/str/float) to float minutes."""
    try:
        if x is None:
            return 0.0
        if isinstance(x, timedelta):
            return x.total_seconds() / 60.0
        v = pd.to_timedelta(x).total_seconds() / 60.0
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        try:
            v = float(x)
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0


def _f0(v):
    """Finite float or 0.0 (NaN/None-safe resume fill)."""
    try:
        f = float(v)
        return f if np.isfinite(f) else 0.0
    except Exception:
        return 0.0


def _pick_col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def _fail_on_schema(df, need=('from_id', 'to_id', 'option', 'segment', 'transport_mode')):
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"r5py itinerary schema mismatch — missing {missing}; "
            f"columns={list(df.columns)}. Refusing silent garbage.")


def _ckpt_backend():
    """Checkpoint format: parquet when pyarrow exists, else stdlib pickle.

    geotfenv has no pyarrow (job 25311809 died importing it before routing a
    single OD). Pickle is slightly larger but dependency-free and equally
    resumable — never let a checkpoint codec block the build.
    """
    try:
        import pyarrow  # noqa: F401
        return 'parquet'
    except Exception:
        return 'pickle'


_CKPT_FMT = _ckpt_backend()


def _ckpt_write(df, path_stem):
    if _CKPT_FMT == 'parquet':
        _p = path_stem + '.parquet'
        df.to_parquet(_p, index=False)
    else:
        _p = path_stem + '.pkl'
        df.to_pickle(_p)
    return _p


def _ckpt_read(path_stem):
    for _ext, _reader in (('.parquet', pd.read_parquet), ('.pkl', pd.read_pickle)):
        _p = path_stem + _ext
        if os.path.exists(_p):
            return _reader(_p)
    return None


def _jvm_memory_bytes(value):
    """Parse '48G' / '12000M' / plain bytes into an int byte count."""
    text = str(value).strip()
    unit = text[-1].upper()
    if unit in 'KMGTP':
        return int(float(text[:-1]) * 1024 ** ' KMGTP'.index(unit))
    return int(float(text))


def _pin_r5py_jvm_memory():
    """Make r5py start its JVM with JVM_MEMORY instead of 80% of node RAM.

    `import r5py` starts the JVM as an import side effect (r5py/r5/transport_network.py
    runs `import com.conveyal.r5` at module level) and `r5py.util.jvm.start_jvm()`
    passes `-Xmx{MAX_JVM_MEMORY}`, where MAX_JVM_MEMORY defaults to 80% of total
    RAM. Patching that global afterwards is too late, and JAVA_TOOL_OPTIONS does
    not win -Xmx precedence. The one reliable hook is r5py's own `--max-memory`
    argument, which r5py.util.memory_footprint reads from the argument parser
    (i.e. from sys.argv) while it is being imported. So: inject it into
    sys.argv, import, then verify — job 25395626 died because a ~102 GB default
    heap plus the Python-side frames exceeded the 128 GB cgroup.
    """
    want = _jvm_memory_bytes(JVM_MEMORY)
    if not any(_a.startswith('--max-memory') for _a in sys.argv[1:]):
        sys.argv += ['--max-memory', JVM_MEMORY]
    import r5py.util.memory_footprint as _mf
    got = int(_mf.MAX_JVM_MEMORY)
    if got > want * 1.05:
        raise RuntimeError(
            f"r5py JVM heap is {got / 1024**3:.1f} GB but JVM_MEMORY={JVM_MEMORY} "
            f"({want / 1024**3:.1f} GB) was requested — refusing to run; a ~102 GB "
            "default heap is what cgroup-OOM-killed job 25395626.")
    log.info(f"JVM heap pinned to {got / 1024**3:.1f} GB (JVM_MEMORY={JVM_MEMORY})")


def main():
    _pin_r5py_jvm_memory()
    import geopandas as gpd
    from shapely.geometry import Point
    from r5py import TransportNetwork, DetailedItineraries, TransportMode

    for _p, _tag in ((GTFS_PATH, 'GTFS'), (OSM_PATH, 'OSM'), (CENTROIDS_PATH, 'centroids')):
        if not os.path.exists(_p):
            log.error(f"Missing {_tag} input: {_p}")
            sys.exit(1)
    if not os.path.exists(OMX_PATH):
        log.error(f"OMX not found: {OMX_PATH}")
        sys.exit(1)
    os.makedirs(WORK_DIR, exist_ok=True)

    cents = pd.read_csv(CENTROIDS_PATH)
    cents = cents.rename(columns=str.lower)
    cents = cents[cents['lat'].notna() & cents['lon'].notna()]
    zone_ids = cents['order'].astype(int).tolist()
    log.info(f"Zone centroids loaded: {len(zone_ids)} zones")
    zidx = {z: i for i, z in enumerate(zone_ids)}
    n = len(zone_ids)

    origins = gpd.GeoDataFrame(
        {"id": zone_ids,
         "geometry": [Point(lon, lat) for lon, lat in zip(cents["lon"], cents["lat"])]},
        crs="EPSG:4326",
    )
    log.info("Building r5 transport network...")
    net = TransportNetwork(str(OSM_PATH), [str(GTFS_PATH)])
    log.info("  transport network ready")
    dep = datetime(2026, 8, 7, DEPARTURE_HOUR, 0)

    o_all = origins.iloc[::ORIGIN_STEP].reset_index(drop=True)
    log.info(f"Routing {len(o_all)} origins x {len(origins)} dest "
             f"(STEP={ORIGIN_STEP}, max_time={MAX_TIME_HRS}h)")

    do_xwait = not DTIM_ONLY
    do_dtim = not XWAIT_ONLY
    # Checkpoint tag: a checkpoint only ever feeds a run with the SAME matrix
    # selection — otherwise an XWAIT-only checkpoint's dummy zeros (dtim AND
    # ddist) would be mistaken for real values on a later full run (silent OMX
    # poison). DDIST rides with DTIM: both come from the walk-access/CAR-egress
    # frame, so the 'd' tag covers them together.
    _mode_tag = ('x' if do_xwait else '') + ('d' if do_dtim else '') or 'none'

    xwait = np.zeros((n, n), dtype=np.float32)
    dtim = np.zeros((n, n), dtype=np.float32)
    ddist = np.zeros((n, n), dtype=np.float32)
    have_dtim_evidence = [False]
    # DDIST is written only if the r5py frame actually carries segment
    # `distance`: an all-zero matrix would be worse than the pipeline's flat
    # 8 km fallback (silent zeros vs a conservative constant).
    _dd_col_found = [False]

    def _run_batch(o_batch, access, egress):
        det = DetailedItineraries(
            net, origins=o_batch, destinations=origins, departure=dep,
            transport_modes=[TransportMode.TRANSIT],
            access_modes=[access], egress_modes=[egress],
            max_time=timedelta(hours=MAX_TIME_HRS),
            # Bound the access/egress legs explicitly: RegionalTask defaults
            # them to max_time (4 h walk to a stop!), which is what blew the
            # R5 search state space up to ~100 GB. The total trip budget stays
            # at MAX_TIME_HRS so reachability matches the TOTIVT skim build.
            max_time_walking=timedelta(minutes=MAX_TIME_WALKING_MIN),
            max_time_driving=timedelta(minutes=MAX_TIME_DRIVING_MIN),
            snap_to_network=True, force_all_to_all=True,
        )
        return det

    # ── DTIM probe: spread origins first; abort DTIM loudly if no CAR legs ─
    # Spread sample (not the first N rows, which are all regional NSW with no
    # park-and-ride transit) so a "no CAR legs" verdict is trustworthy.
    if do_dtim:
        log.info("DTIM probe: spread ODs with walk-access / CAR-egress ...")
        # Explicit, env-tunable probe size. The old strided-by-len//10 sample was
        # 11 origins on the 371-zone grid and alone reached 65.4 GB RSS
        # (job 25329147) because the access/egress legs were still unbounded.
        _probe_n = max(1, min(PROBE_ORIGINS, len(origins)))
        _probe_o = origins.iloc[
            np.linspace(0, len(origins) - 1, _probe_n).astype(int)
        ].reset_index(drop=True)
        log.info(f"  probe origins={len(_probe_o)} (walk<="
                 f"{MAX_TIME_WALKING_MIN:.0f} min, drive<={MAX_TIME_DRIVING_MIN:.0f} min, "
                 f"total<={MAX_TIME_HRS:.1f} h)")
        try:
            _pd = _run_batch(_probe_o, TransportMode.WALK, TransportMode.CAR)
            _fail_on_schema(_pd)
            _car = _pd[_pd['transport_mode'].astype(str).str.upper() == 'CAR']
            log.info(f"  probe rows={len(_pd)} car-leg rows={len(_car)}")
            if len(_car) == 0:
                log.warning("DTIM probe found NO car-egress legs — R5 ignores "
                            "egress=CAR here; DTIM will be SKIPPED (pipeline keeps "
                            "its TOTIVT proxy). XWAIT build continues.")
                do_dtim = False
            else:
                have_dtim_evidence[0] = True
        except Exception as e:
            log.warning(f"DTIM probe failed ({e}) — DTIM will be SKIPPED.")
            do_dtim = False
        # Free the probe frame: DetailedItineraries materialises every option x
        # segment (4 h window) — holding it while batching doubles peak RSS.
        try:
            del _pd
        except NameError:
            pass
        gc.collect()
        log.info(f"  post-probe RSS={_rss_gb():.1f} GB")

    # ── main grid, origin-batched with checkpoints ─────────────────────────
    t0 = time.time()
    done = 0
    for _b0 in range(0, len(o_all), BATCH_ORIGINS):
        _rss = _cur_rss_gb()
        if not np.isfinite(_rss):
            _rss = _rss_gb()
        if np.isfinite(_rss) and _rss > MAX_RSS_GB:
            log.error(
                f"RSS={_rss:.1f} GB > MAX_RSS_GB={MAX_RSS_GB:.0f} — stopping at a batch "
                f"boundary ({done}/{len(o_all)} origins checkpointed). A cgroup OOM kill "
                "would lose the batch in flight; every completed checkpoint survives and "
                "a resubmit of the same command resumes here.")
            break
        _batch = o_all.iloc[_b0:_b0 + BATCH_ORIGINS]
        _oids = [int(i) for i in _batch['id'].tolist()]
        _stem = os.path.join(WORK_DIR, f"batch_{_oids[0]}_{_oids[-1]}_{_mode_tag}")
        _cdf = _ckpt_read(_stem)
        if _cdf is not None:
            try:
                for _r in _cdf.itertuples(index=False):
                    _oi = zidx.get(int(_r.o_id))
                    _di = zidx.get(int(_r.d_id))
                    if _oi is None or _di is None:
                        continue
                    if do_xwait:
                        xwait[_oi, _di] = _f0(_r.xwait)
                    if do_dtim:
                        dtim[_oi, _di] = _f0(_r.dtim)
                        ddist[_oi, _di] = _f0(_r.ddist)
                done += len(_oids)
                continue
            except Exception as e:
                log.warning(f"  checkpoint unreadable, recomputing: {e}")
        try:
            _rows = []
            if do_xwait:
                _wx = _run_batch(_batch, TransportMode.WALK, TransportMode.WALK)
                _fail_on_schema(_wx)
                _wcol = _pick_col(_wx, 'wait_time', 'wait', 'initial_wait_time')
                _dcol = _pick_col(_wx, 'duration', 'travel_time', 'segment_time')
                for (_o, _d, _op), _gg in _wx.groupby(['from_id', 'to_id', 'option']):
                    _gg = _gg.sort_values('segment')
                    _segs = _gg.to_dict('records')
                    _xw = 0.0
                    _seen_first_board = False
                    for _s in _segs:
                        _tm = str(_s.get('transport_mode', '')).upper()
                        if 'TRANSIT' in _tm or 'BUS' in _tm or 'RAIL' in _tm \
                                or 'TRAM' in _tm or 'SUBWAY' in _tm or 'FERRY' in _tm:
                            if _seen_first_board and _wcol:
                                _xw += _minutes(_s.get(_wcol))
                            _seen_first_board = True
                    _rows.append((_o, _d, _xw, None, None))
            if do_dtim:
                _dr = _run_batch(_batch, TransportMode.WALK, TransportMode.CAR)
                _fail_on_schema(_dr)
                _dcol = _pick_col(_dr, 'duration', 'travel_time', 'segment_time')
                _ddcol = _pick_col(_dr, 'distance', 'segment_distance', 'dist')
                if _ddcol:
                    _dd_col_found[0] = True
                _by_od = {}
                for (_o, _d, _op), _gg in _dr.groupby(['from_id', 'to_id', 'option']):
                    _segs = _gg.to_dict('records')
                    _dt = sum(_minutes(_s.get(_dcol)) for _s in _segs
                              if str(_s.get('transport_mode', '')).upper() == 'CAR')
                    _dd = (sum(_f0(_s.get(_ddcol)) for _s in _segs
                               if str(_s.get('transport_mode', '')).upper() == 'CAR')
                           / 1000.0) if _ddcol else 0.0
                    _by_od.setdefault((_o, _d), (_dt, _dd))
                _merged = {(_o, _d): (_xw, *_by_od.get((_o, _d), (0.0, 0.0)))
                           for (_o, _d, _xw, _dt_, _dd_) in _rows}
                for (_o, _d), (_dt, _dd) in _by_od.items():
                    if (_o, _d) not in _merged:
                        _merged[(_o, _d)] = (0.0, _dt, _dd)
                _rows = [(_o, _d, _xw, _dt, _dd)
                         for (_o, _d), (_xw, _dt, _dd) in _merged.items()]
            _cdf = pd.DataFrame(_rows, columns=['o_id', 'd_id', 'xwait', 'dtim', 'ddist'])
            _ckpt_write(_cdf, _stem)
            # Apply to the in-memory matrices BEFORE the del block below: the
            # checkpoint write above already made this batch durable, and the
            # frames are only freed once its values are in xwait/dtim.
            for _r in _cdf.itertuples(index=False):
                _oi = zidx.get(int(_r.o_id))
                _di = zidx.get(int(_r.d_id))
                if _oi is None or _di is None:
                    continue
                if do_xwait:
                    xwait[_oi, _di] = _f0(_r.xwait)
                if do_dtim:
                    dtim[_oi, _di] = _f0(_r.dtim)
                    ddist[_oi, _di] = _f0(_r.ddist)
            # Flush the batch frame now (checkpoint holds it): the next
            # _run_batch allocates a fresh multi-GB frame and the old one must
            # not pile up behind it (the 25329147 OOM). Explicit dels — each
            # name may be unbound on DTIM_ONLY/XWAIT_ONLY runs.
            try:
                del _wx
            except NameError:
                pass
            try:
                del _dr
            except NameError:
                pass
            try:
                del _rows
            except NameError:
                pass
            try:
                del _cdf
            except NameError:
                pass
            try:
                del _merged
            except NameError:
                pass
            try:
                del _by_od
            except NameError:
                pass
            gc.collect()
            log.info(f"  batch RSS={_rss_gb():.1f} GB")
        except Exception as e:
            log.warning(f"  batch {_oids[0]}-{_oids[-1]} failed (non-fatal, "
                        f"rerun resumes here): {e}")
            continue
        done += len(_oids)
        _el = (time.time() - t0) / 60.0
        log.info(f"  batch done {done}/{len(o_all)} origins ({_el:.0f} min elapsed)")

    # ── refuse to publish a partial matrix ─────────────────────────────────
    # A partial DTIM/XWAIT would be consumed by the pipeline as if it were the
    # real matrix, silently zeroing every unfinished OD. Require full coverage.
    if done < len(o_all):
        log.error(
            f"INCOMPLETE: {done}/{len(o_all)} origins checkpointed — NOT writing the "
            "OMX, so the pipeline keeps its derivations/proxies rather than reading a "
            "partial matrix. Resubmit the same command to resume from checkpoints.")
        sys.exit(2)

    # ── a strided smoke grid must never publish ────────────────────────────
    # ORIGIN_STEP>1 computes only a sample of origins: publishing would zero
    # every unsampled OD in the real OMX — silent garbage of exactly the kind
    # this script exists to avoid. Smoke runs are for validating the r5py/JVM
    # path; they exit 3 after their checkpoints are written. ALLOW_PARTIAL_
    # PUBLISH=1 is the explicit escape hatch (e.g. smoke against a throwaway
    # OMX copy via OMX_PATH).
    if ORIGIN_STEP != 1 and os.environ.get('ALLOW_PARTIAL_PUBLISH', '0') != '1':
        log.warning(
            f"SMOKE grid (ORIGIN_STEP={ORIGIN_STEP}): {done}/{len(origins)} sampled "
            "origins finished and checkpoints are on disk, but the OMX is NOT written "
            "(only a full ORIGIN_STEP=1 grid may publish). This is by design; the "
            "job's verify step will report the matrices as missing.")
        sys.exit(3)

    # ── write into OMX (backup first) ──────────────────────────────────────
    import openmatrix as omx
    import shutil as _sh
    _bak = str(OMX_PATH) + f".backup_{datetime.now():%Y%m%d_%H%M}"
    log.info(f"Backing up OMX -> {_bak}")
    _sh.copy2(str(OMX_PATH), _bak)
    f = omx.open_file(str(OMX_PATH), 'a')
    _prov = (f"walk<={MAX_TIME_WALKING_MIN:.0f}min, drive<={MAX_TIME_DRIVING_MIN:.0f}min, "
             f"total<={MAX_TIME_HRS:.1f}h, dep={DEPARTURE_HOUR:02d}:00")
    if do_xwait:
        if 'WLK_LOC_WLK_XWAIT__AM' in f:
            del f['WLK_LOC_WLK_XWAIT__AM']
        f['WLK_LOC_WLK_XWAIT__AM'] = np.asarray(xwait, dtype=np.float32)
        log.info(f"  wrote WLK_LOC_WLK_XWAIT__AM "
                 f"(mean={xwait.mean():.1f}, nonzero={(xwait > 0).mean():.1%}) [{_prov}]")
    if do_dtim:
        if 'WLK_LOC_DRV_DTIM__AM' in f:
            del f['WLK_LOC_DRV_DTIM__AM']
        f['WLK_LOC_DRV_DTIM__AM'] = np.asarray(dtim, dtype=np.float32)
        log.info(f"  wrote WLK_LOC_DRV_DTIM__AM "
                 f"(mean={dtim.mean():.1f}, nonzero={(dtim > 0).mean():.1%}) [{_prov}]")
    if do_dtim and not _dd_col_found[0]:
        log.warning(
            "r5py frames carried no segment `distance` column — NOT writing "
            "WLK_LOC_DRV_DDIST__AM (all-zero would replace the pipeline's flat "
            "8 km drive_access_km fallback with silent zeros).")
    if do_dtim and _dd_col_found[0]:
        if 'WLK_LOC_DRV_DDIST__AM' in f:
            del f['WLK_LOC_DRV_DDIST__AM']
        f['WLK_LOC_DRV_DDIST__AM'] = np.asarray(ddist, dtype=np.float32)
        log.info(f"  wrote WLK_LOC_DRV_DDIST__AM "
                 f"(mean={ddist.mean():.2f} km, nonzero={(ddist > 0).mean():.1%}) [{_prov}]")
    f.close()
    log.info("Done. Pipeline picks real matrices over derivations/proxies automatically.")


if __name__ == '__main__':
    main()
