#!/usr/bin/env python
"""
build_transit_dtim_xwait.py — exact drive-egress time + transfer-wait skims.
================================================================================
Fills the two transit-LOS matrices the 371-zone OMX lacks (verified absent):

    WLK_LOC_DRV_DTIM__AM   drive-egress leg minutes (park-and-ride egress)
    WLK_LOC_WLK_XWAIT__AM  transfer-wait minutes on walk-access paths

Method: r5py DetailedItineraries (per-segment mode/duration/wait) instead of
the TravelTimeMatrix aggregate used by build_transit_gtfs_skims.py:
  * XWAIT = sum of wait_time over transit segments AFTER the first boarding
    (option 0 = best itinerary). Restricted to ODs with BOARDS>1 in the
    existing matrix — single-boarding ODs have XWAIT=0 by construction, which
    skips most of the grid.
  * DTIM  = sum of CAR-mode segment durations with access=[WALK],
    egress=[CAR]. A spread-origin probe runs first: if R5 yields no CAR-egress legs
    (unsupported engine path), DTIM is skipped loudly and the pipeline keeps
    its TOTIVT proxy — no silent garbage.

Performance: itineraries cost ~1 s+/OD. The full 137k grid does NOT fit one
slot: origins run in batches with per-batch parquet checkpoints
(DTIM_WORK_DIR) and the job simply resubmits to resume (completed origins are
skipped). Expect ~2 chained 48 h slots for the full build; ORIGIN_STEP>1 (or
DTIM_ONLY/XWAIT_ONLY) for smoke tests.

Safety: the OMX is timestamp-backuped before any write. Matrices are replaced
atomically (del + assign). The pipeline consumes them automatically:
_build_od_dataset prefers real matrices over its XWAIT derivation, and the
c_drv_transit term prefers DTIM over its TOTIVT proxy when present.

Usage:
    python build_transit_dtim_xwait.py            # full build (both matrices)
    DTIM_ONLY=1 python build_transit_dtim_xwait.py
    XWAIT_ONLY=1 python build_transit_dtim_xwait.py
    ORIGIN_STEP=20 python build_transit_dtim_xwait.py   # smoke (~19 origins)

Env: OMX_PATH, GTFS_PATH, OSM_PATH, CENTROIDS_PATH (same defaults as
     build_transit_gtfs_skims.py), DEPARTURE_HOUR (8), DTIM_WORK_DIR,
     ORIGIN_STEP (1), BATCH_ORIGINS (19), MAX_TIME_HRS (4).
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
MAX_TIME_HRS = float(os.environ.get('MAX_TIME_HRS', '4'))
DTIM_ONLY = os.environ.get('DTIM_ONLY', '0') == '1'
XWAIT_ONLY = os.environ.get('XWAIT_ONLY', '0') == '1'

os.environ['JAVA_TOOL_OPTIONS'] = f"-Xmx{os.environ.get('JVM_MEMORY', '12G')}"

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


def main():
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
    # selection — otherwise an XWAIT-only checkpoint's dummy zeros would be
    # mistaken for real DTIM on a later full run (silent OMX poison).
    _mode_tag = ('x' if do_xwait else '') + ('d' if do_dtim else '') or 'none'

    xwait = np.zeros((n, n), dtype=np.float32)
    dtim = np.zeros((n, n), dtype=np.float32)
    have_dtim_evidence = [False]

    def _run_batch(o_batch, access, egress):
        det = DetailedItineraries(
            net, origins=o_batch, destinations=origins, departure=dep,
            transport_modes=[TransportMode.TRANSIT],
            access_modes=[access], egress_modes=[egress],
            max_time=timedelta(hours=MAX_TIME_HRS),
            snap_to_network=True, force_all_to_all=True,
        )
        return det

    # ── DTIM probe: spread origins first; abort DTIM loudly if no CAR legs ─
    # Spread sample (not the first N rows, which are all regional NSW with no
    # park-and-ride transit) so a "no CAR legs" verdict is trustworthy.
    if do_dtim:
        log.info("DTIM probe: spread ODs with walk-access / CAR-egress ...")
        _probe_o = origins.iloc[::max(1, len(origins) // 10)].reset_index(drop=True)
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
                    _rows.append((_o, _d, _xw, None))
            if do_dtim:
                _dr = _run_batch(_batch, TransportMode.WALK, TransportMode.CAR)
                _fail_on_schema(_dr)
                _dcol = _pick_col(_dr, 'duration', 'travel_time', 'segment_time')
                _by_od = {}
                for (_o, _d, _op), _gg in _dr.groupby(['from_id', 'to_id', 'option']):
                    _dt = sum(_minutes(_s.get(_dcol)) for _s in _gg.to_dict('records')
                              if str(_s.get('transport_mode', '')).upper() == 'CAR')
                    _by_od.setdefault((_o, _d), _dt)
                _merged = {(_o, _d): (_xw, _by_od.get((_o, _d), 0.0)) for (_o, _d, _xw, _) in _rows}
                for (_o, _d), _dt in _by_od.items():
                    if (_o, _d) not in _merged:
                        _merged[(_o, _d)] = (0.0, _dt)
                _rows = [(_o, _d, _xw, _dt) for (_o, _d), (_xw, _dt) in _merged.items()]
            _cdf = pd.DataFrame(_rows, columns=['o_id', 'd_id', 'xwait', 'dtim'])
            _ckpt_write(_cdf, _stem)
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
            for _r in _cdf.itertuples(index=False):
                _oi = zidx.get(int(_r.o_id))
                _di = zidx.get(int(_r.d_id))
                if _oi is None or _di is None:
                    continue
                if do_xwait:
                    xwait[_oi, _di] = _f0(_r.xwait)
                if do_dtim:
                    dtim[_oi, _di] = _f0(_r.dtim)
        except Exception as e:
            log.warning(f"  batch {_oids[0]}-{_oids[-1]} failed (non-fatal, "
                        f"rerun resumes here): {e}")
            continue
        done += len(_oids)
        _el = (time.time() - t0) / 60.0
        log.info(f"  batch done {done}/{len(o_all)} origins ({_el:.0f} min elapsed)")

    # ── write into OMX (backup first) ──────────────────────────────────────
    import openmatrix as omx
    import shutil as _sh
    _bak = str(OMX_PATH) + f".backup_{datetime.now():%Y%m%d_%H%M}"
    log.info(f"Backing up OMX -> {_bak}")
    _sh.copy2(str(OMX_PATH), _bak)
    f = omx.open_file(str(OMX_PATH), 'a')
    if do_xwait:
        if 'WLK_LOC_WLK_XWAIT__AM' in f:
            del f['WLK_LOC_WLK_XWAIT__AM']
        f['WLK_LOC_WLK_XWAIT__AM'] = np.asarray(xwait, dtype=np.float32)
        log.info(f"  wrote WLK_LOC_WLK_XWAIT__AM "
                 f"(mean={xwait.mean():.1f}, nonzero={(xwait > 0).mean():.1%})")
    if do_dtim:
        if 'WLK_LOC_DRV_DTIM__AM' in f:
            del f['WLK_LOC_DRV_DTIM__AM']
        f['WLK_LOC_DRV_DTIM__AM'] = np.asarray(dtim, dtype=np.float32)
        log.info(f"  wrote WLK_LOC_DRV_DTIM__AM "
                 f"(mean={dtim.mean():.1f}, nonzero={(dtim > 0).mean():.1%})")
    f.close()
    log.info("Done. Pipeline picks real matrices over derivations/proxies automatically.")


if __name__ == '__main__':
    main()
