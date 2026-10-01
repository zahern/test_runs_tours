#!/usr/bin/env python
"""
build_zone_supply.py — parking, micromobility & transit-service supply per SA2.
================================================================================
Three independent sections (each skips cleanly when its input is absent):

1. OSM SUPPLY (OSM_PBF, default saved_data/osm_cache/seq_osm.pbf) via the
   GDAL OSM driver (vectorised, like build_location_table._read_pois_geopandas):
     n_parking / parking_capacity / parking_area_m2  amenity=parking
       (+parking_space); capacity parsed from other_tags HSTORE for parking
       rows only (cheap — small subset)
     n_bike_parking    amenity=bicycle_parking (+bicycle_rental)
     n_ev_charging     amenity=charging
     n_fuel            amenity=fuel
     n_car_share       amenity=car_sharing / car_rental
     n_taxi            amenity=taxi
     cycleway_km       highway=cycleway lines (midpoint → nearest SA2)
   → saved_data/zone_supply_osm.csv (one row per SA2 with supply)

2. TRANSIT SUPPLY (GTFS_PATH, default saved_data/osm_cache/seq_gtfs.zip):
     n_transit_stops   stops per SA2 (nearest centroid)
     transit_trips_am  weekday scheduled arrivals 07:00–09:00 per SA2
     transit_headway   120 / mean(trips per stop) — AM peak headway proxy
   → saved_data/transit_supply_sa2.csv

3. Nothing is merged here — build_zone_landuse_overture.py picks both CSVs
   up (like the transport step) when present.

Zone key: SA2 codes via nearest-centroid assignment (same as the location
table), so the merge keys match zone_landuse_overture.csv.

Run:  python build_zone_supply.py
Env:  OSM_PBF, GTFS_PATH, ZONE_CENTROIDS, SUPPLY_OUT, TRANSIT_OUT,
      SKIP_OSM / SKIP_TRANSIT (=1 to skip a section).
"""
import os
import re
import zipfile
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('zone_supply')

_HERE = os.path.dirname(os.path.abspath(__file__))
OSM_PBF = os.environ.get('OSM_PBF', os.path.join(_HERE, 'saved_data', 'osm_cache', 'seq_osm.pbf'))
GTFS_PATH = os.environ.get('GTFS_PATH', os.path.join(_HERE, 'saved_data', 'osm_cache', 'seq_gtfs.zip'))
ZONE_CENTROIDS = os.environ.get('ZONE_CENTROIDS',
                                os.path.join(_HERE, 'saved_data', 'osm_cache', 'zone_centroids.csv'))
SUPPLY_OUT = os.environ.get('SUPPLY_OUT', os.path.join(_HERE, 'saved_data', 'zone_supply_osm.csv'))
TRANSIT_OUT = os.environ.get('TRANSIT_OUT', os.path.join(_HERE, 'saved_data', 'transit_supply_sa2.csv'))

OSM_SUPPLY_COLS = ('n_parking', 'parking_capacity', 'parking_area_m2',
                   'n_bike_parking', 'n_ev_charging', 'n_fuel',
                   'n_car_share', 'n_taxi', 'cycleway_km')
TRANSIT_SUPPLY_COLS = ('n_transit_stops', 'transit_trips_am', 'transit_headway')

_CAP_RE = re.compile(r'"capacity"\s*=>\s*"([^"]+)"')


def _zone_lookup():
    """Return (zone_keys, zone_lonlat) from the centroids file."""
    from build_location_table import _nearest  # noqa: F401  (re-exported below)
    zc = pd.read_csv(ZONE_CENTROIDS)
    _zcol = next((c for c in ('zone_id', 'SA2_MAIN16', 'ORDER', 'zone') if c in zc.columns), zc.columns[0])
    _lat = next((c for c in ('lat', 'latitude', 'y', 'centroid_lat') if c in zc.columns), None)
    _lon = next((c for c in ('lon', 'lng', 'longitude', 'x', 'centroid_lon') if c in zc.columns), None)
    if _lat is None or _lon is None:
        raise SystemExit(f'FATAL: zone centroids need lat/lon columns; got {list(zc.columns)}')
    keys = pd.to_numeric(zc[_zcol], errors='coerce')
    ok = keys.notna()
    return keys[ok].astype('int64').to_numpy(), zc.loc[ok, [_lon, _lat]].to_numpy(dtype=float)


def _assign_zones(xs, ys):
    """Nearest-SA2 zone id per lon/lat array."""
    from build_location_table import _nearest
    keys, ref = _zone_lookup()
    idx, _ = _nearest(np.column_stack([np.asarray(xs, dtype=float),
                                       np.asarray(ys, dtype=float)]), ref)
    return keys[idx]


def _col_str(g, col, n):
    import pandas as _pd
    return (g[col].astype('string').str.lower() if col in g.columns
            else _pd.Series(_pd.array([_pd.NA] * n, dtype='string')))


def build_osm_supply():
    """Section 1: parking + micromobility + fuel/car-share + cycleways."""
    import geopandas as gpd
    counts, capacity, area, bike, ev, fuel, share, taxi = [], [], [], [], [], [], [], []
    cyc_len, cyc_zone = [], []
    for layer in ('points', 'multipolygons'):
        try:
            g = gpd.read_file(OSM_PBF, layer=layer)
        except Exception as e:
            log.warning('  OSM layer %s failed: %s', layer, e)
            continue
        if g is None or len(g) == 0:
            continue
        n = len(g)
        amenity = _col_str(g, 'amenity', n)
        is_park = amenity.isin(['parking', 'parking_space'])
        is_bike = amenity.isin(['bicycle_parking', 'bicycle_rental', 'bicycle_repair_station'])
        is_ev = amenity.isin(['charging', 'charging_station'])
        is_fuel = amenity.isin(['fuel'])
        is_share = amenity.isin(['car_sharing', 'car_rental', 'car_pooling'])
        is_taxi = amenity.isin(['taxi'])
        keep = (is_park | is_bike | is_ev | is_fuel | is_share | is_taxi)
        if not keep.any():
            log.info('  layer %s: 0 supply POIs', layer)
            try:
                log.info('  layer %s columns: %s', layer, list(g.columns)[:12])
                if 'amenity' in g.columns:
                    log.info('  layer %s top amenity values: %s',
                             layer, g['amenity'].astype(str).value_counts().head(8).to_dict())
            except Exception:
                pass
            continue
        gk = g.loc[keep.fillna(False).to_numpy()].copy()
        try:
            gm = gk.geometry.to_crs(3577)
            cent = gm.centroid.to_crs(4326)
            ar = gm.area.to_numpy() if layer == 'multipolygons' else np.zeros(len(gk))
        except Exception:
            cent = gk.geometry.centroid
            ar = np.zeros(len(gk))
        zones = _assign_zones(cent.x.to_numpy(), cent.y.to_numpy())
        gk = gk.assign(_zone=zones, _area=np.maximum(ar, 0.0))
        # capacity from other_tags HSTORE for parking rows only
        cap = np.zeros(len(gk))
        am = gk['amenity'].astype(str).str.lower()
        is_p = am.isin(['parking', 'parking_space']).to_numpy()
        if 'other_tags' in gk.columns:
            for i in np.where(is_p)[0]:
                m = _CAP_RE.search(str(gk['other_tags'].iloc[i]))
                if m:
                    try:
                        cap[i] = float(m.group(1).split(';')[0].strip())
                    except Exception:
                        pass
        gk = gk.assign(_cap=np.clip(cap, 0, 100000))
        pk = gk.loc[is_p].copy()  # parking rows only — counts/capacity/area share this set
        counts.append(pk.assign(_k=1)[['_zone', '_k']])
        capacity.append(pk.assign(_v=pk['_cap'])[['_zone', '_v']])
        area.append(pk.assign(_v=pk['_area'])[['_zone', '_v']])
        am_all = am
        bike.append(gk[am_all.isin(['bicycle_parking', 'bicycle_rental', 'bicycle_repair_station'])].assign(_k=1)[['_zone', '_k']])
        ev.append(gk[am_all.isin(['charging', 'charging_station'])].assign(_k=1)[['_zone', '_k']])
        fuel.append(gk[am_all.isin(['fuel'])].assign(_k=1)[['_zone', '_k']])
        share.append(gk[am_all.isin(['car_sharing', 'car_rental', 'car_pooling'])].assign(_k=1)[['_zone', '_k']])
        taxi.append(gk[am_all.isin(['taxi'])].assign(_k=1)[['_zone', '_k']])
        log.info('  layer %s: %d supply POIs (%d parking)', layer, int(keep.sum()), int(is_p.sum()))
    # cycleways (lines layer)
    try:
        gl = gpd.read_file(OSM_PBF, layer='lines')
        if gl is not None and len(gl):
            hw = _col_str(gl, 'highway', len(gl))
            cyc = gl[hw.fillna('').to_numpy() == 'cycleway']
            if len(cyc):
                gm = cyc.geometry.to_crs(3577)
                mid = gm.interpolate(gm.length / 2).to_crs(4326)
                cyc_zone = _assign_zones(mid.x.to_numpy(), mid.y.to_numpy())
                cyc_len = (gm.length.to_numpy() / 1000.0).tolist()
                log.info('  lines: %.1f cycleway-km over %d segments', float(np.sum(cyc_len)), len(cyc))
    except Exception as e:
        log.warning('  OSM lines layer failed (non-fatal): %s', e)

    def _sum(frames):
        if not frames:
            return pd.Series(dtype=float)
        f = pd.concat(frames, ignore_index=True)
        v = '_k' if '_k' in f.columns else '_v'
        return f.groupby('_zone')[v].sum()

    def _size(frames):
        if not frames:
            return pd.Series(dtype=float)
        f = pd.concat(frames, ignore_index=True)
        return f.groupby('_zone').size().astype(float)

    zones = set()
    for _s in (_sum(counts), _sum(bike), _sum(ev), _sum(fuel),
               _sum(share), _sum(taxi)):
        zones |= set(_s.index.tolist())
    idx = sorted(zones)
    out = pd.DataFrame({'zone': pd.Series(idx, dtype='int64')}).set_index('zone')
    out['n_parking'] = _size(counts)
    out['parking_capacity'] = _sum(capacity)
    out['parking_area_m2'] = _sum(area)
    out['n_bike_parking'] = _sum(bike)
    out['n_ev_charging'] = _sum(ev)
    out['n_fuel'] = _sum(fuel)
    out['n_car_share'] = _sum(share)
    out['n_taxi'] = _sum(taxi)
    if len(cyc_len):
        out['cycleway_km'] = pd.Series(cyc_len, index=pd.Series(cyc_zone)).groupby(level=0).sum()
    for c in OSM_SUPPLY_COLS:
        if c not in out.columns:
            out[c] = 0.0
    out = out[list(OSM_SUPPLY_COLS)].fillna(0.0).reset_index()
    # drop the placeholder artefacts
    out = out[['zone'] + list(OSM_SUPPLY_COLS)]
    # ── Full-universe reindex (job 25256774) ─────────────────────────────
    # Previously only zones WITH supply were emitted (310/371) — alarming in
    # the verify line and forcing every downstream merge to fillna. Emit all
    # 371 zones with zeros for no-supply zones so the CSV is self-describing.
    try:
        _keys, _ = _zone_lookup()
        _full = pd.DataFrame({'zone': pd.Series(sorted(set(int(k) for k in _keys)), dtype='int64')})
        out['zone'] = pd.to_numeric(out['zone'], errors='coerce').astype('Int64')
        out = _full.merge(out, on='zone', how='left')
        for c in OSM_SUPPLY_COLS:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
        log.info('reindexed OSM supply to full universe (%d zones, %d with supply)',
                 len(out), int((out[list(OSM_SUPPLY_COLS)].sum(axis=1) > 0).sum()))
    except Exception as _re:
        log.warning('OSM full-universe reindex skipped (non-fatal): %s', _re)
    os.makedirs(os.path.dirname(SUPPLY_OUT) or '.', exist_ok=True)
    out.to_csv(SUPPLY_OUT, index=False)
    log.info('wrote OSM supply (%d zones) → %s', len(out), SUPPLY_OUT)
    return out


def build_transit_supply():
    """Section 2: GTFS stop counts + weekday AM-peak scheduled arrivals per SA2."""
    with zipfile.ZipFile(GTFS_PATH) as z:
        names = z.namelist()
        _sn = next((n for n in names if n.endswith('stops.txt')), None)
        _stn = next((n for n in names if n.endswith('stop_times.txt')), None)
        _tn = next((n for n in names if n.endswith('trips.txt')), None)
        _cn = next((n for n in names if n.endswith('calendar.txt')), None)
        if _sn is None:
            log.warning('transit supply skipped (no stops.txt in %s)', GTFS_PATH)
            return None
        stops = pd.read_csv(__import__('io').BytesIO(z.read(_sn)))
    if not {'stop_lat', 'stop_lon'}.issubset(stops.columns):
        log.warning('transit supply skipped (stops lack lat/lon)')
        return None
    id_col = next((c for c in ('stop_id', 'stop_code') if c in stops.columns), stops.columns[0])
    stops = stops[np.isfinite(stops['stop_lat']) & np.isfinite(stops['stop_lon'])].copy()
    stops['_zone'] = _assign_zones(stops['stop_lon'].to_numpy(), stops['stop_lat'].to_numpy())
    n_stops = stops.groupby('_zone').size().rename('n_transit_stops')
    trips_am = pd.Series(dtype=float)
    try:
        if _stn is None or _tn is None:
            raise ValueError('stop_times/trips missing')
        with zipfile.ZipFile(GTFS_PATH) as z:
            st = pd.read_csv(__import__('io').BytesIO(z.read(_stn)),
                             usecols=lambda c: c in ('trip_id', 'stop_id', 'arrival_time'),
                             dtype=str)
            tp = pd.read_csv(__import__('io').BytesIO(z.read(_tn)),
                             usecols=lambda c: c in ('trip_id', 'service_id'),
                             dtype=str)
        hr = pd.to_numeric(st['arrival_time'].str.slice(0, 2), errors='coerce')
        st = st[hr.isin([7, 8])]
        if _cn is not None:
            with zipfile.ZipFile(GTFS_PATH) as z:
                cal = pd.read_csv(__import__('io').BytesIO(z.read(_cn)), dtype=str)
            if 'monday' in cal.columns and 'service_id' in cal.columns:
                wk = set(cal.loc[pd.to_numeric(cal['monday'], errors='coerce') == 1, 'service_id'])
                tp = tp[tp['service_id'].isin(wk)]
        st = st[st['trip_id'].isin(set(tp['trip_id']))]
        am = st.groupby('stop_id').size().rename('trips_am')
        stop_zone = stops.set_index(stops[id_col].astype(str))['_zone']
        amz = am.to_frame().assign(_zone=am.index.map(stop_zone)).dropna(subset=['_zone'])
        trips_am = amz.groupby('_zone')['trips_am'].sum().rename('transit_trips_am')
    except Exception as e:
        log.warning('transit AM trips skipped (non-fatal): %s', e)
    out = pd.DataFrame({'n_transit_stops': n_stops}).join(
        pd.DataFrame({'transit_trips_am': trips_am}), how='outer').fillna(0.0)
    per_stop = (out['transit_trips_am'] / out['n_transit_stops'].replace(0, np.nan)).fillna(0.0)
    out['transit_headway'] = np.where(per_stop > 0, 120.0 / per_stop, 0.0)
    out = out.reset_index().rename(columns={'_zone': 'zone'})
    out['zone'] = pd.to_numeric(out['zone'], errors='coerce').astype('Int64')
    out = out.dropna(subset=['zone'])
    out['zone'] = out['zone'].astype('int64')
    for c in TRANSIT_SUPPLY_COLS:
        if c not in out.columns:
            out[c] = 0.0
    out = out[['zone'] + list(TRANSIT_SUPPLY_COLS)].fillna(0.0)
    # ── Full-universe reindex (job 25256774: was 308/371) ────────────────
    try:
        _keys, _ = _zone_lookup()
        _full = pd.DataFrame({'zone': pd.Series(sorted(set(int(k) for k in _keys)), dtype='int64')})
        out['zone'] = pd.to_numeric(out['zone'], errors='coerce').astype('Int64')
        out = _full.merge(out, on='zone', how='left')
        for c in TRANSIT_SUPPLY_COLS:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
        log.info('reindexed transit supply to full universe (%d zones, %d with stops)',
                 len(out), int((out['n_transit_stops'] > 0).sum()))
    except Exception as _re:
        log.warning('transit full-universe reindex skipped (non-fatal): %s', _re)
    os.makedirs(os.path.dirname(TRANSIT_OUT) or '.', exist_ok=True)
    out.to_csv(TRANSIT_OUT, index=False)
    log.info('wrote transit supply (%d zones) → %s', len(out), TRANSIT_OUT)
    return out


def main():
    if os.environ.get('SKIP_OSM', '0') != '1':
        if os.path.exists(OSM_PBF):
            build_osm_supply()
        else:
            log.warning('OSM supply skipped (missing %s)', OSM_PBF)
    if os.environ.get('SKIP_TRANSIT', '0') != '1':
        if os.path.exists(GTFS_PATH):
            build_transit_supply()
        else:
            log.warning('transit supply skipped (missing %s)', GTFS_PATH)


if __name__ == '__main__':
    main()
