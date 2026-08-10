#!/usr/bin/env python
"""
Build zone-to-zone transit travel-time matrices from TransLink GTFS
(r5py 1.1.7 API) and write them into the SA2 371-zone OMX that the model
actually consumes (../demand_/sa2_skims_507_omxspec.omx).

The GA predict chain reads its transit LOS from:
    WLK_LOC_WLK_TOTIVT__AM   (od travel time, minutes)   — main c_ivt_transit var
    WLK_LOC_WLK_WAIT__AM     (wait minutes)              — auxiliary
    WLK_LOC_WLK_BOARDS__AM   (transit boardings)         — auxiliary

r5py 1.1.7 measuring (verified on the login node):
  * TravelTimeMatrix        — full 371x371 door-to-door transit time in ~18 s.
    (No per-vehicle/wait split; `travel_time` is the R5 door-to-door time.)
  * DetailedItineraries     — exact IVT + wait + transfers per OD, but ~1+ s per
    OD pair; the full 137k OD grid would take many hours. It is only usable on
    a sample (see ORIGIN_STEP), and we therefore only use the fast matrix to
    ensure the transit LOS is real and its O-D structure is exact.

This module therefore:
  1. Computes the exact full TravelTimeMatrix into WLK_LOC_WLK_TOTIVT__AM
     (overwriting the old STM-derived values, which were largely 2.2-min
     constants / 0.01-correlated with reality).
  2. Leaves WLK_LOC_WLK_WAIT__AM / WLK_LOC_WLK_BOARDS__AM untouched — those
     already contain well-behaved values (mean ~28 min wait, ~2.6 boards).

Usage:
    python build_transit_gtfs_skims.py

Env:
    OMX_PATH       target OMX (default ../demand_/sa2_skims_507_omxspec.omx)
    DEPARTURE_HOUR AM-peak departure hour (default 8)
    JVM_MEMORY     Java heap for r5py (default 12G)
    ORIGIN_STEP    if set >1, compute the TravelTimeMatrix on a strided origin
                   subsample for a quick smoke (default 1 = full 371 x 371)
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OMX_PATH = os.environ.get(
    "OMX_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "demand_", "sa2_skims_507_omxspec.omx"),
)
CENTROIDS_PATH = os.environ.get(
    "CENTROIDS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "saved_data", "osm_cache", "zone_centroids.csv"),
)
GTFS_PATH = os.environ.get(
    "GTFS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "saved_data", "osm_cache", "seq_gtfs.zip"),
)
OSM_PATH = os.environ.get(
    "OSM_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "saved_data", "osm_cache", "seq_osm.pbf"),
)
DEPARTURE_HOUR = int(os.environ.get("DEPARTURE_HOUR", "8"))
JVM_MEMORY = os.environ.get("JVM_MEMORY", "12G")
ORIGIN_STEP = int(os.environ.get("ORIGIN_STEP", "1"))

os.environ["JAVA_TOOL_OPTIONS"] = f"-Xmx{JVM_MEMORY}"

import numpy as np
import pandas as pd


def main():
    gtfs_path = Path(GTFS_PATH)
    osm_path = Path(OSM_PATH)
    if not gtfs_path.exists() or not osm_path.exists():
        log.error(f"Missing GTFT outputs: {gtfs_path} / {osm_path}")
        sys.exit(1)

    # ── 1. Zone centroids ────────────────────────────────────────────────
    cents = pd.read_csv(CENTROIDS_PATH)
    cents = cents.rename(columns=str.lower)
    cents = cents[cents["lat"].notna() & cents["lon"].notna()]
    zone_ids = cents["order"].astype(int).tolist()
    log.info(f"Zone centroids loaded: {len(zone_ids)} zones")

    import geopandas as gpd
    from shapely.geometry import Point
    from r5py import TransportNetwork, TravelTimeMatrix

    origins = gpd.GeoDataFrame(
        {"id": zone_ids,
         "geometry": [Point(lon, lat) for lon, lat in zip(cents["lon"], cents["lat"])]},
        crs="EPSG:4326",
    )
    log.info(f"r5 model: {len(origins)}×{len(origins)} = {len(origins) ** 2} OD pairs")

    log.info("Building r5 transport network...")
    net = TransportNetwork(str(osm_path), [str(gtfs_path)])
    log.info("  transport network ready")

    dep = datetime(2026, 8, 7, DEPARTURE_HOUR, 0)
    from datetime import timedelta

    o_all = origins.iloc[::ORIGIN_STEP]
    log.info(f"Computing TravelTimeMatrix on {len(o_all)} origins × {len(origins)} dest...")
    m = TravelTimeMatrix(
        net,
        origins=o_all,
        destinations=origins,
        departure=dep,
        transport_modes=["TRANSIT"],
        # r5py defaults max_time=7200s (2h), which zeroes the long tail of the
        # O‑D matrix and collapsed coverage to ~21%. 4h matches the original
        # skim's reachable O‑D coverage (>90%).
        max_time=timedelta(hours=4),
    )
    log.info(f"  travel time matrix: {len(m)} rows (min={m.travel_time.min()}, "
             f"max={m.travel_time.max()})")

    zone_to_idx = {z: i for i, z in enumerate(zone_ids)}
    ivt = np.full((len(zone_ids), len(zone_ids)), 0.0, dtype=np.float32)
    for row in m.itertuples(index=False):
        o_id = int(row.from_id)
        d_id = int(row.to_id)
        v = float(row.travel_time)
        if o_id in zone_to_idx and d_id in zone_to_idx:
            ivt[zone_to_idx[o_id], zone_to_idx[d_id]] = v

    # diagonal (same-zone) transit loops make no sense → 0
    np.fill_diagonal(ivt, 0.0)
    n_pos = int((ivt > 0).sum())
    log.info(f"IVT matrix filled: {n_pos}/{ivt.size} positive pairs "
             f"(mean={(ivt[ivt > 0]).mean():.1f} min)")

    # ── 2. Write into the 507 OMX ────────────────────────────────────────
    omx_path = Path(OMX_PATH)
    if not omx_path.exists():
        log.error(f"OMX not found: {omx_path}")
        sys.exit(1)

    import openmatrix as omx
    log.info(f"Updating {omx_path}")
    f = omx.open_file(str(omx_path), "a")

    name = "WLK_LOC_WLK_TOTIVT__AM"
    if name in f:
        log.info(f"  replacing existing {name}")
        del f[name]
    f[name] = ivt
    log.info(f"  wrote {name}")

    # leave WAIT / BOARDS untouched (already well-behaved)
    for k in ("WLK_LOC_WLK_WAIT__AM", "WLK_LOC_WLK_BOARDS__AM"):
        if k in f:
            a = f[k][:]
            valid = int((a > 0).sum())
            log.info(f"  kept existing {k}: {valid}/{a.size} positive pairs")
    f.close()
    log.info("Done.")


if __name__ == "__main__":
    main()