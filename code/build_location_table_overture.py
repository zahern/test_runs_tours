#!/usr/bin/env python
"""
build_location_table_overture.py — Location table from OVERTURE MAPS Places.
================================================================================
Overture Maps Places is a cleaner, better-categorised POI dataset than raw OSM
(conflated from multiple sources with confidence scores). This builds the same
POLARIS-style location table as build_location_table.py but sources the POIs from
Overture, then reuses that module's shared downstream (SA2 zone assignment,
tod_distance, parking proxy, is_home/is_work/is_school/is_discretionary flags,
CSV write) so micro_location_placement.py consumes it unchanged.

Access (two options):
  • ONLINE (needs internet egress): DuckDB queries the public Overture S3
    GeoParquet directly, filtered to the SEQ bounding box (from zone centroids).
  • OFFLINE: set OVERTURE_PLACES=<local places.parquet> (pre-downloaded on a
    node with egress, e.g. `overturemaps download --bbox=... --type=place
    -f geoparquet -o places.parquet`). Compute nodes then need no internet.

Env: OVERTURE_PLACES (local parquet; else S3), OVERTURE_RELEASE (S3 release tag),
     OVERTURE_MIN_CONFIDENCE (default 0.5), ZONE_CENTROIDS, LOCATION_OUT.
Run (env with duckdb + duckdb spatial/httpfs, e.g. geotfenv):
     python build_location_table_overture.py
"""
import os
import logging

import numpy as np
import pandas as pd

from build_location_table import (finalize_location_table, _LU_DEFAULT_AREA,
                                   ZONE_CENTROIDS)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('location_table_ovm')

OVERTURE_PLACES = os.environ.get('OVERTURE_PLACES', '')        # local parquet (offline)
OVERTURE_RELEASE = os.environ.get('OVERTURE_RELEASE', '2026-08-19.0')
MIN_CONF = float(os.environ.get('OVERTURE_MIN_CONFIDENCE', '0.5'))


def _ovm_land_use(cat: str):
    """Overture Places primary category → land_use category (None = skip)."""
    # Overture rows with no primary category arrive as NaN (float). `cat or ''`
    # does NOT catch that — NaN is truthy in Python, so it would return the float
    # and .lower() raises AttributeError. Guard on real strings only.
    if not isinstance(cat, str) or not cat:
        return None
    c = cat.lower()
    # parking FIRST: 'parking' contains 'park' — checked after recreation it
    # would leak into recreation counts and parking supply would be lost.
    if 'parking' in c:
        return 'parking'
    if any(k in c for k in ('school', 'education', 'university', 'college',
                            'child_care', 'kindergarten', 'tutoring')):
        return 'education'
    if any(k in c for k in ('restaurant', 'cafe', 'coffee', 'food', 'bar_', 'pub',
                            'fast_food', 'bakery', 'eat_and_drink', 'dining', 'brewery')):
        return 'food'
    if any(k in c for k in ('hospital', 'medical', 'health', 'clinic', 'pharmacy',
                            'dentist', 'doctor', 'physician', 'nursing')):
        return 'health'
    if any(k in c for k in ('shop', 'retail', 'store', 'supermarket', 'market',
                            'mall', 'grocery', 'boutique', 'dealer')):
        return 'retail'
    if any(k in c for k in ('office', 'professional', 'financial', 'bank',
                            'insurance', 'real_estate', 'coworking', 'consulting')):
        return 'office'
    if any(k in c for k in ('park', 'recreation', 'sport', 'fitness', 'gym',
                            'entertainment', 'arts', 'cinema', 'theatre', 'museum',
                            'attraction', 'active_life', 'stadium', 'zoo')):
        return 'recreation'
    if any(k in c for k in ('government', 'public_service', 'community', 'library',
                            'police', 'fire_', 'post_office', 'civic', 'townhall',
                            'courthouse', 'religious', 'church')):
        return 'civic'
    if any(k in c for k in ('warehouse', 'industrial', 'manufacturing', 'logistics',
                            'factory', 'wholesale')):
        return 'industrial'
    if any(k in c for k in ('hotel', 'motel', 'accommodation', 'lodging', 'resort')):
        return 'office'   # accommodation is a workplace
    return None


def _bbox_from_centroids(margin=0.05):
    zc = pd.read_csv(ZONE_CENTROIDS)
    _lat = next((c for c in ('lat', 'latitude', 'y') if c in zc.columns), None)
    _lon = next((c for c in ('lon', 'lng', 'longitude', 'x') if c in zc.columns), None)
    lat = pd.to_numeric(zc[_lat], errors='coerce'); lon = pd.to_numeric(zc[_lon], errors='coerce')
    return (float(lon.min()) - margin, float(lat.min()) - margin,
            float(lon.max()) + margin, float(lat.max()) + margin)


def _read_places():
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    xmin, ymin, xmax, ymax = _bbox_from_centroids()
    log.info("SEQ bbox: [%.3f, %.3f, %.3f, %.3f]", xmin, ymin, xmax, ymax)
    if OVERTURE_PLACES and os.path.exists(OVERTURE_PLACES):
        src = f"read_parquet('{OVERTURE_PLACES}')"
        log.info("reading Overture places from local file %s", OVERTURE_PLACES)
    else:
        con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
        url = (f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"
               "/theme=places/type=place/*")
        src = f"read_parquet('{url}', hive_partitioning=1)"
        log.info("querying Overture S3 release %s (needs internet egress)", OVERTURE_RELEASE)
    # geometry storage differs by vintage: the 14/08 offline parquet serves a
    # native GEOMETRY type (ST_GeomFromWKB on it raises a Binder error), older
    # pulls serve WKB blobs (need the decode). Probe once, then query — same
    # pattern as the buildings step in build_zone_landuse_overture.py.
    _gtype = con.execute(f"SELECT typeof(geometry) FROM {src} LIMIT 1").fetchone()[0]
    _gtype_str = str(_gtype).upper()
    _is_native_geom = _gtype_str.startswith('GEOMETRY') or 'GEOMETRY' in _gtype_str
    _geom_expr = "geometry" if _is_native_geom else "ST_GeomFromWKB(geometry)"
    log.info("Overture geometry storage: %s (native_geom=%s)", _gtype, _is_native_geom)
    q = f"""
        SELECT categories.primary AS category,
                ST_X({_geom_expr}) AS x,
                ST_Y({_geom_expr}) AS y,
                confidence
        FROM {src}
        WHERE bbox.xmin BETWEEN {xmin} AND {xmax}
          AND bbox.ymin BETWEEN {ymin} AND {ymax}
          AND confidence >= {MIN_CONF}
    """
    df = con.execute(q).df()
    log.info("Overture returned %d places (confidence ≥ %.2f)", len(df), MIN_CONF)
    return df


def main():
    df = _read_places()
    if df is None or len(df) == 0:
        raise SystemExit("FATAL: no Overture places returned — check release tag / "
                         "internet egress, or set OVERTURE_PLACES to a local parquet")
    df['land_use'] = df['category'].map(_ovm_land_use)
    df = df[df['land_use'].notna()].copy()
    df['lu_area'] = df['land_use'].map(_LU_DEFAULT_AREA).astype(float)
    df['poi_category'] = df['category'].fillna('').astype(str)
    pois = df[['x', 'y', 'land_use', 'lu_area', 'poi_category']].reset_index(drop=True)
    log.info("Overture POIs kept: %d; land_use mix: %s",
             len(pois), pois['land_use'].value_counts().to_dict())
    finalize_location_table(pois)


if __name__ == '__main__':
    main()
