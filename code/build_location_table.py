#!/usr/bin/env python
"""
build_location_table.py — POLARIS-style sub-zonal Location table from OSM.
================================================================================
Extracts POI *points* (with coordinates) from the local OSM extract, assigns each
to its SA2 zone (nearest centroid — SEQ has no zone polygons on disk), tags a
land-use category + size, computes an average parking-cost proxy and the walk
distance to the nearest GTFS transit stop. Output mirrors POLARIS's Location table
so a within-zone location-choice model can consume it (structurally identical to
the destination quantity_ca, one level below the zone).

Output → saved_data/location_table.csv, columns (POLARIS analogue):
    location          unique id
    zone              parent SA2 zone (nearest centroid)
    x, y              coordinates (lon, lat — EPSG:4326)
    land_use          residential/retail/office/education/food/health/
                      recreation/civic/industrial
    lu_area           size weight (building footprint m² if a polygon, else a
                      per-type default) — the within-zone size term
    avg_parking_cost  proxy: scales with the zone's location density (CBD-like
                      zones cost more); replace with real data when available
    tod_distance      walk distance (m) to the nearest GTFS transit stop

Run (in geotfenv; NO internet — pbf + gtfs are pre-downloaded):
    python build_location_table.py
Env overrides: OSM_PBF, ZONE_CENTROIDS, GTFS_PATH, LOCATION_OUT.
"""
import os
import io
import zipfile
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('location_table')

_HERE = os.path.dirname(os.path.abspath(__file__))
OSM_PBF        = os.environ.get('OSM_PBF',        os.path.join(_HERE, 'saved_data', 'osm_cache', 'seq_osm.pbf'))
ZONE_CENTROIDS = os.environ.get('ZONE_CENTROIDS', os.path.join(_HERE, 'saved_data', 'osm_cache', 'zone_centroids.csv'))
GTFS_PATH      = os.environ.get('GTFS_PATH',      os.path.join(_HERE, 'saved_data', 'osm_cache', 'seq_gtfs.zip'))
OUT            = os.environ.get('LOCATION_OUT',   os.path.join(_HERE, 'saved_data', 'location_table.csv'))

# per-land_use default footprint (m²) when the POI is a point, and a size weight
# used by the within-zone location choice (relative attraction by activity type)
_LU_DEFAULT_AREA = {
    'retail': 300.0, 'office': 500.0, 'education': 2000.0, 'food': 150.0,
    'health': 400.0, 'recreation': 1000.0, 'civic': 600.0,
    'industrial': 1500.0, 'residential': 150.0, 'parking': 800.0,
}

# ── Location person-capacity model (POLARIS-style occupancy) ─────────────────
# Estimate how many people a location holds from its GROSS FLOOR AREA and land
# use, using standard planning occupancy ratios (m² of GFA per person). GFA =
# footprint (lu_area) × storeys. The person "kind" differs by land use:
#   residential → residents, education → students, everything else → workers.
# Aggregated per zone (build_zone_landuse_overture.py) this yields estimated
# employment / enrolment / residents straight from the map — a data-driven size
# and capacity measure, cross-checkable against the landuse and WorldPop.
_GFA_PER_PERSON = {      # m² gross floor area per person
    'office': 18.0, 'retail': 25.0, 'food': 18.0, 'health': 30.0,
    'education': 10.0, 'industrial': 80.0, 'civic': 25.0,
    'recreation': 40.0, 'residential': 45.0,
}
_DEFAULT_STOREYS = {     # assumed storeys when no building height is known (POIs)
    'office': 3, 'retail': 1, 'food': 1, 'health': 2, 'education': 2,
    'industrial': 1, 'civic': 2, 'recreation': 1, 'residential': 1,
}
_PERSON_KIND = {'residential': 'resident', 'education': 'student'}   # else 'worker'

# POLARIS land_use activity flags: (is_home, is_work, is_school, is_discretionary).
# These gate which activity types can be placed at a location (see
# micro_location_placement.py). is_work = any non-residential; is_school =
# educational; is_discretionary = non-educational non-home.
_LU_FLAGS = {
    'residential': (1, 0, 0, 0),
    'retail':      (0, 1, 0, 1),
    'office':      (0, 1, 0, 0),
    'education':   (0, 1, 1, 0),
    'food':        (0, 1, 0, 1),
    'health':      (0, 1, 0, 1),
    'recreation':  (0, 1, 0, 1),
    'civic':       (0, 1, 0, 1),
    'industrial':  (0, 1, 0, 0),
    # parking structures host no activities — all-zero flags keep them out of
    # micro-location choice while letting stage 7 use them as supply.
    'parking':     (0, 0, 0, 0),
}

_EDU_AMENITY = {'school', 'university', 'college', 'kindergarten', 'childcare'}
_FOOD_AMENITY = {'restaurant', 'cafe', 'fast_food', 'food_court', 'pub', 'bar', 'biergarten', 'ice_cream'}
_HEALTH_AMENITY = {'hospital', 'clinic', 'doctors', 'pharmacy', 'dentist', 'nursing_home'}
_CIVIC_AMENITY = {'townhall', 'courthouse', 'community_centre', 'library', 'police', 'fire_station', 'post_office'}
_REC_AMENITY = {'cinema', 'theatre', 'arts_centre', 'nightclub'}


def _classify(tags: dict):
    """OSM tags → POLARIS land_use category (None = skip)."""
    amenity = tags.get('amenity'); shop = tags.get('shop'); leisure = tags.get('leisure')
    tourism = tags.get('tourism'); landuse = tags.get('landuse'); building = tags.get('building')
    office = tags.get('office')
    if shop or amenity in {'marketplace', 'vending_machine'}:
        return 'retail'
    if amenity in _EDU_AMENITY or building in {'school', 'university', 'college'}:
        return 'education'
    if amenity in _FOOD_AMENITY:
        return 'food'
    if amenity in _HEALTH_AMENITY or building in {'hospital'}:
        return 'health'
    if office or amenity in {'bank'} or building in {'office', 'commercial'}:
        return 'office'
    if leisure or tourism or amenity in _REC_AMENITY or landuse in {'recreation_ground', 'park'}:
        return 'recreation'
    if amenity in _CIVIC_AMENITY or office == 'government':
        return 'civic'
    # motor-vehicle parking only — bicycle parking is counted separately in
    # build_zone_supply (n_bike_parking), not car supply for stage 7.
    if amenity in {'parking', 'parking_space', 'parking_entrance',
                   'motorcycle_parking'}:
        return 'parking'
    if landuse in {'industrial', 'logistics'} or building in {'industrial', 'warehouse'}:
        return 'industrial'
    if building in {'residential', 'house', 'apartments', 'detached', 'terrace'} or landuse == 'residential':
        return 'residential'
    return None


_TAG_KEYS = ('amenity', 'shop', 'leisure', 'tourism', 'landuse', 'building', 'office')


def _read_pois_pyrosm(path):
    from pyrosm import OSM
    log.info('reading POIs via pyrosm …')
    osm = OSM(path)
    gdf = osm.get_pois()
    if gdf is None or len(gdf) == 0:
        return None
    # pyrosm exposes common tags as columns; the rest live in a 'tags' dict column
    rows = []
    _has = {k: (k in gdf.columns) for k in _TAG_KEYS}
    geom = gdf.geometry
    cent = geom.centroid
    is_poly = geom.geom_type.isin(['Polygon', 'MultiPolygon']).to_numpy()
    # polygon areas in a metric CRS (Australia Albers) for lu_area
    try:
        areas = gdf.to_crs(3577).geometry.area.to_numpy()
    except Exception:
        areas = np.full(len(gdf), np.nan)
    xs = cent.x.to_numpy(); ys = cent.y.to_numpy()
    extra_tags = gdf['tags'] if 'tags' in gdf.columns else None
    for i in range(len(gdf)):
        tags = {k: gdf[k].iloc[i] for k in _TAG_KEYS if _has[k] and pd.notna(gdf[k].iloc[i])}
        if extra_tags is not None:
            _e = extra_tags.iloc[i]
            if isinstance(_e, dict):
                for k in _TAG_KEYS:
                    tags.setdefault(k, _e.get(k))
        lu = _classify(tags)
        if lu is None:
            continue
        rows.append((xs[i], ys[i], lu,
                     areas[i] if (is_poly[i] and np.isfinite(areas[i]) and areas[i] > 0)
                     else _LU_DEFAULT_AREA[lu]))
    return pd.DataFrame(rows, columns=['x', 'y', 'land_use', 'lu_area'])


def _read_pois_geopandas(path):
    """Fallback: GDAL OSM driver via geopandas (points + polygon centroids)."""
    import geopandas as gpd
    log.info('reading POIs via geopandas/GDAL OSM driver …')
    frames = []
    for layer in ('points', 'multipolygons'):
        try:
            g = gpd.read_file(path, layer=layer)
        except Exception as e:
            log.warning('  layer %s failed: %s', layer, e)
            continue
        if g is None or len(g) == 0:
            continue
        # ── VECTORISED classification (the per-row loop timed out on the full
        #    SEQ pbf). Classify from the GDAL named tag columns only — they cover
        #    the vast majority of POIs; skipping per-row other_tags HSTORE parsing
        #    is what makes this fast. ──────────────────────────────────────────
        lu = _classify_vec(g)
        keep = lu.notna().to_numpy()
        if not keep.any():
            log.info('  layer %s: 0 classified POIs', layer)
            continue
        gk = g.loc[keep]
        lu_k = lu[keep].to_numpy()
        # centroids in a projected CRS (correctness + no geographic-CRS warning)
        try:
            gm = gk.geometry.to_crs(3577)
            cent = gm.centroid.to_crs(4326)
            areas = gm.area.to_numpy() if layer == 'multipolygons' else None
        except Exception:
            cent = gk.geometry.centroid
            areas = None
        xs = cent.x.to_numpy(); ys = cent.y.to_numpy()
        default_area = np.array([_LU_DEFAULT_AREA[v] for v in lu_k], dtype=float)
        if areas is not None:
            area_k = np.where(np.isfinite(areas) & (areas > 0), areas, default_area)
        else:
            area_k = default_area
        frames.append(pd.DataFrame({'x': xs, 'y': ys, 'land_use': lu_k, 'lu_area': area_k}))
        log.info('  layer %s: %d classified POIs', layer, int(keep.sum()))
    return pd.concat(frames, ignore_index=True) if frames else None


def _classify_vec(g) -> pd.Series:
    """Vectorised OSM tag → land_use over a GeoDataFrame's named tag columns."""
    n = len(g)

    def _c(col):
        return (g[col].astype('string').str.lower() if col in g.columns
                else pd.Series(pd.array([pd.NA] * n, dtype='string')))
    amenity, shop, leisure = _c('amenity'), _c('shop'), _c('leisure')
    tourism, landuse, building, office = _c('tourism'), _c('landuse'), _c('building'), _c('office')
    lu = pd.Series([None] * n, dtype='object')

    def _set(mask, val):
        m = mask.fillna(False).to_numpy() & lu.isna().to_numpy()
        lu.iloc[np.where(m)[0]] = val

    _set(shop.notna() | amenity.isin(['marketplace', 'vending_machine']), 'retail')
    _set(amenity.isin(_EDU_AMENITY) | building.isin(['school', 'university', 'college']), 'education')
    _set(amenity.isin(_FOOD_AMENITY), 'food')
    _set(amenity.isin(_HEALTH_AMENITY) | building.isin(['hospital']), 'health')
    _set(office.notna() | amenity.isin(['bank']) | building.isin(['office', 'commercial']), 'office')
    _set(leisure.notna() | tourism.notna() | amenity.isin(_REC_AMENITY)
         | landuse.isin(['recreation_ground', 'park']), 'recreation')
    _set(amenity.isin(_CIVIC_AMENITY) | (office == 'government'), 'civic')
    _set(amenity.isin(['parking', 'parking_space', 'parking_entrance',
                       'motorcycle_parking']), 'parking')
    _set(landuse.isin(['industrial', 'logistics']) | building.isin(['industrial', 'warehouse']), 'industrial')
    _set(building.isin(['residential', 'house', 'apartments', 'detached', 'terrace'])
         | (landuse == 'residential'), 'residential')
    return lu


def _nearest(query_lonlat, ref_lonlat):
    """Index of nearest ref point for each query (haversine); returns (idx, dist_m)."""
    try:
        from sklearn.neighbors import BallTree
        _ref = np.radians(ref_lonlat[:, ::-1])   # (lat, lon)
        tree = BallTree(_ref, metric='haversine')
        d, idx = tree.query(np.radians(query_lonlat[:, ::-1]), k=1)
        return idx[:, 0], d[:, 0] * 6_371_000.0
    except Exception:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_lonlat)
        d, idx = tree.query(query_lonlat, k=1)
        # rough deg→m at SEQ latitude (~-27.5°)
        return idx, d * 111_000.0


def main():
    for _p in (OSM_PBF, ZONE_CENTROIDS):
        if not os.path.exists(_p):
            raise SystemExit(f'FATAL: missing input {_p}')

    pois = None
    try:
        pois = _read_pois_pyrosm(OSM_PBF)
    except Exception as e:
        log.warning('pyrosm path failed (%s) — trying geopandas', e)
    if pois is None or len(pois) == 0:
        pois = _read_pois_geopandas(OSM_PBF)
    if pois is None or len(pois) == 0:
        raise SystemExit('FATAL: no POIs extracted — install pyrosm (pip install pyrosm) in geotfenv')
    pois = pois[np.isfinite(pois['x']) & np.isfinite(pois['y'])].reset_index(drop=True)
    log.info('extracted %d POIs; land_use mix: %s', len(pois),
             pois['land_use'].value_counts().to_dict())
    finalize_location_table(pois)


def finalize_location_table(pois: pd.DataFrame, out: str = None) -> pd.DataFrame:
    """Shared downstream for any POI source (OSM or Overture): assign each POI to
    its SA2 zone, compute tod_distance + parking proxy + POLARIS activity flags,
    and write the location table. `pois` needs columns [x, y, land_use, lu_area].
    """
    out = out or OUT
    # ── zone assignment: nearest SA2 centroid ────────────────────────────────
    zc = pd.read_csv(ZONE_CENTROIDS)
    _zcol = next((c for c in ('zone_id', 'SA2_MAIN16', 'ORDER', 'zone') if c in zc.columns), zc.columns[0])
    _lat = next((c for c in ('lat', 'latitude', 'y', 'centroid_lat') if c in zc.columns), None)
    _lon = next((c for c in ('lon', 'lng', 'longitude', 'x', 'centroid_lon') if c in zc.columns), None)
    if _lat is None or _lon is None:
        raise SystemExit(f'FATAL: zone centroids need lat/lon columns; got {list(zc.columns)}')
    zone_ll = zc[[_lon, _lat]].to_numpy(dtype=float)
    zidx, _ = _nearest(pois[['x', 'y']].to_numpy(dtype=float), zone_ll)
    pois['zone'] = zc[_zcol].to_numpy()[zidx]

    # ── backfill zones with zero POIs (synthetic centroid placeholders) ────
    # Outer-Moreton-Bay SA2s have no OSM POIs; with no rows here every activity
    # in those zones micro-places to location_id=-1/NaN. Append a 4-role
    # placeholder set (home/work/school/discretionary) at the zone centroid so
    # every role pool is non-empty. est_persons is zeroed below so employment
    # aggregates don't move; lu_area keeps type defaults as choice weights.
    pois['_is_synth'] = False
    try:
        _seq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'saved_data', 'seq_zone_ids_371.csv')
        _have = set(pd.to_numeric(pois['zone'], errors='coerce')
                    .dropna().astype('int64').astype(str))
        _sa2_like = bool(_have) and min(len(z) for z in _have) >= 9
        if os.path.exists(_seq_path) and _sa2_like:
            _seq = pd.read_csv(_seq_path, dtype={'SA2_MAIN16': str})
            _missing = [z.strip() for z in _seq['SA2_MAIN16']
                        if z.strip() not in _have]
            _roles = [('residential', 'housing'),
                      ('office', 'professional_services'),
                      ('education', 'school'),
                      ('retail', 'shopping_centre')]
            _syn_rows = []
            for _z in _missing:
                _zr = _seq[_seq['SA2_MAIN16'].str.strip() == _z].iloc[0]
                _rng = np.random.default_rng(int(_z) % (2 ** 32))
                _jit = (_rng.random((len(_roles), 2)) - 0.5) * 0.0014
                for (_k, (_lu, _pc)), (_dj_lon, _dj_lat) in zip(
                        enumerate(_roles), _jit):
                    _syn_rows.append({
                        'zone': int(_z),
                        'x': float(_zr['longitude']) + float(_dj_lon),
                        'y': float(_zr['latitude']) + float(_dj_lat),
                        'land_use': _lu, 'lu_area': _LU_DEFAULT_AREA[_lu],
                        'poi_category': 'synthetic_centroid',
                        '_is_synth': True})
            if _syn_rows:
                if 'poi_category' not in pois.columns:
                    pois['poi_category'] = 'unspecified'
                pois = pd.concat([pois, pd.DataFrame(_syn_rows)],
                                 ignore_index=True)
                log.info('backfilled %d zones × %d roles with synthetic '
                         'centroid locations', len(_missing), len(_roles))
        elif not _sa2_like:
            log.warning('backfill skipped: location zones are not SA2-like '
                        '(centroids file may be ORDER-keyed) — refusing to mix '
                        'key systems')
    except Exception as _be:                            # noqa: BLE001
        log.warning('synthetic backfill failed (non-fatal): %s', _be)

    # ── tod_distance: walk metres to nearest GTFS stop ───────────────────────
    pois['tod_distance'] = np.nan
    try:
        if os.path.exists(GTFS_PATH):
            with zipfile.ZipFile(GTFS_PATH) as z:
                _sname = next((n for n in z.namelist() if n.endswith('stops.txt')), None)
                stops = pd.read_csv(io.BytesIO(z.read(_sname))) if _sname else None
            if stops is not None and {'stop_lat', 'stop_lon'}.issubset(stops.columns):
                _s = stops[['stop_lon', 'stop_lat']].to_numpy(dtype=float)
                _s = _s[np.isfinite(_s).all(axis=1)]
                _, dist_m = _nearest(pois[['x', 'y']].to_numpy(dtype=float), _s)
                pois['tod_distance'] = np.round(dist_m, 1)
                log.info('tod_distance: median %.0f m to nearest of %d GTFS stops',
                         np.nanmedian(pois['tod_distance']), len(_s))
    except Exception as e:
        log.warning('tod_distance skipped (%s)', e)

    # ── avg_parking_cost proxy: scales with per-zone location density ────────
    _dens = pois.groupby('zone')['land_use'].transform('size').astype(float)
    _dn = (_dens - _dens.min()) / max(_dens.max() - _dens.min(), 1.0)
    pois['avg_parking_cost'] = np.round(_dn * float(os.environ.get('LOC_MAX_PARKING', '8.0')), 2)

    # POLARIS activity-type flags (gate which activities can be placed here)
    _flags = pois['land_use'].map(_LU_FLAGS)
    pois['is_home']          = _flags.map(lambda t: t[0] if isinstance(t, tuple) else 0).astype('int8')
    pois['is_work']          = _flags.map(lambda t: t[1] if isinstance(t, tuple) else 0).astype('int8')
    pois['is_school']        = _flags.map(lambda t: t[2] if isinstance(t, tuple) else 0).astype('int8')
    pois['is_discretionary'] = _flags.map(lambda t: t[3] if isinstance(t, tuple) else 0).astype('int8')

    # ── Estimated person capacity from GFA × land-use occupancy ratio ────────
    # est_persons = (footprint × storeys) / GFA-per-person.  storeys from the
    # 'storeys' column if the POI source provided building floors, else a per-
    # land-use default.  person_kind: resident / student / worker.
    _lu = pois['land_use'].astype(str)
    _storeys = (pd.to_numeric(pois['storeys'], errors='coerce') if 'storeys' in pois.columns
                else pd.Series(np.nan, index=pois.index))
    _storeys = _storeys.fillna(_lu.map(_DEFAULT_STOREYS)).fillna(1.0).clip(lower=1.0)
    _gfa = pd.to_numeric(pois['lu_area'], errors='coerce').fillna(0.0).clip(lower=0.0) * _storeys
    _dens = _lu.map(_GFA_PER_PERSON).fillna(30.0)
    pois['est_persons'] = (_gfa / _dens).round(1)
    pois['person_kind'] = _lu.map(_PERSON_KIND).fillna('worker')
    # parking structures hold cars, not people — zero them so they never
    # inflate est_workers (stage 7 still sees them via land_use='parking').
    _park = (_lu == 'parking').to_numpy()
    pois.loc[_park, 'est_persons'] = 0.0
    # Synthetic placeholders carry no employment (see backfill above).
    if '_is_synth' in pois.columns:
        pois.loc[pois['_is_synth'].fillna(False).astype(bool), 'est_persons'] = 0.0
        pois['synthetic'] = pois['_is_synth'].fillna(False).astype(bool).astype('int8')
        pois = pois.drop(columns=['_is_synth'])
    elif 'synthetic' not in pois.columns:
        pois['synthetic'] = 0

    pois.insert(0, 'location', np.arange(1, len(pois) + 1))
    _source_columns = ['poi_category'] if 'poi_category' in pois.columns else []
    pois = pois[['location', 'zone', 'x', 'y', 'land_use', 'lu_area',
                 *_source_columns, 'avg_parking_cost', 'tod_distance',
                 'is_home', 'is_work', 'is_school', 'is_discretionary',
                 'est_persons', 'person_kind', 'synthetic']]
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    pois.to_csv(out, index=False)
    log.info('wrote %d locations across %d zones → %s',
             len(pois), pois['zone'].nunique(), out)
    return pois


if __name__ == '__main__':
    main()
