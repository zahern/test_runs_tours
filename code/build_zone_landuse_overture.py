#!/usr/bin/env python
"""
build_zone_landuse_overture.py — comprehensive, up-to-date zone land use.
================================================================================
Aggregates the Overture location table (build_location_table_overture.py →
location_table.csv, one row per POI with zone + land_use + lu_area + POLARIS
flags) into a per-SA2 land-use table richer and more current than the existing
zone_landuse: counts AND floor-area by activity category
    medical / education / retail / office / food / recreation / civic /
    industrial / residential
plus work- and school-location totals. Refreshes the matching columns of the
existing zone_landuse (so destination size terms use current Overture features)
while KEEPING its population/employment columns, and adds the new comprehensive
columns. Optionally folds in Overture BUILDINGS floor area (residential /
commercial m²) for a real capacity measure, and Overture TRANSPORTATION
segments (road_km / freeway_km / arterial_km / freeway_share per SA2) as
exogenous network-capacity proxies — geometry only, never traffic volumes
(Overture carries no counts; volumes are model outcomes, so they must not
enter utilities directly — see module docstring of the transport step).

Run (env with pandas; geopandas+duckdb only needed for the optional buildings step):
    python build_zone_landuse_overture.py
Env: LOCATION_TABLE, ZONE_LANDUSE_IN, ZONE_LANDUSE_OUT, OVERTURE_BUILDINGS(0/1),
     OVERTURE_SA1_LANDUSE(0/1), SA1_GEOJSON, SA2_GEOJSON, OVERTURE_RELEASE,
     OVERTURE_TRANSPORT(auto/0/1), OVERTURE_TRANSPORT_PARQUET.
     OVERTURE_TRANSPORT=auto (default): use a local transport parquet if
     present, else skip. =1: query Overture S3 (needs internet egress).
     =0: skip entirely. Offline pre-download on an egress node, e.g.:
       overturemaps download --bbox=152.0,-28.4,153.6,-26.5 \
           --type=segment -f geoparquet -o saved_data/overture_transport_seq.parquet
"""
import os
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('zone_landuse_ovm')

_HERE = os.path.dirname(os.path.abspath(__file__))
LOCATION_TABLE  = os.environ.get('LOCATION_TABLE',  os.path.join(_HERE, 'saved_data', 'location_table.csv'))
ZONE_LANDUSE_IN = os.environ.get('ZONE_LANDUSE_IN', os.path.join(_HERE, '..', 'new_pop', 'saved_data', 'zone_landuse_357.csv'))
ZONE_LANDUSE_OUT = os.environ.get('ZONE_LANDUSE_OUT', os.path.join(_HERE, 'saved_data', 'zone_landuse_overture.csv'))
SA2_GEOJSON     = os.environ.get('SA2_GEOJSON',
    (os.path.join(_HERE, 'saved_data', 'sa2_371.geojson')
     if os.path.exists(os.path.join(_HERE, 'saved_data', 'sa2_371.geojson'))
     else os.path.join(_HERE, 'saved_data', 'sa2_seq.geojson')))
SA1_GEOJSON     = os.environ.get(
    'SA1_GEOJSON',
    os.path.join(_HERE, '..', 'SA1_SHP', 'SA1_2021_AUST_GDA2020.shp'),
)
SA1_LANDUSE_OUT = os.environ.get(
    'SA1_LANDUSE_OUT', os.path.join(_HERE, 'saved_data', 'sa1_landuse_overture.csv')
)
OVERTURE_TRANSPORT_MODE = os.environ.get('OVERTURE_TRANSPORT', 'auto').strip().lower()
OVERTURE_TRANSPORT_PARQUET = os.environ.get(
    'OVERTURE_TRANSPORT_PARQUET',
    os.path.join(_HERE, 'saved_data', 'overture_transport_seq.parquet'),
)
# OSM + transit supply (build_zone_supply.py) merged when present.
SUPPLY_OSM_PATH = os.environ.get(
    'SUPPLY_OSM_PATH', os.path.join(_HERE, 'saved_data', 'zone_supply_osm.csv'))
TRANSIT_SUPPLY_PATH = os.environ.get(
    'TRANSIT_SUPPLY_PATH', os.path.join(_HERE, 'saved_data', 'transit_supply_sa2.csv'))
SUPPLY_COLS = ('n_parking', 'parking_capacity', 'parking_area_m2',
               'n_bike_parking', 'n_ev_charging', 'n_fuel',
               'n_car_share', 'n_taxi', 'cycleway_km')
TRANSIT_COLS = ('n_transit_stops', 'transit_trips_am', 'transit_headway')
# Exogenous network-capacity columns added by the transport step. Deliberately
# NOT in any utility spec by default — available for validation, the DTA
# feedback loop (skim updates), and later GA selection with sign constraints.
TRANSPORT_COLS = ('road_km', 'freeway_km', 'arterial_km', 'freeway_share')

# Overture land_use category → existing zone_landuse column it REFRESHES (so the
# comprehensive table is a drop-in for the current size-term inputs) + a POI count.
_LU_TO_EXISTING = {
    'retail':      'retail',
    'education':   'Education',
    'health':      'Healthcare',
    'office':      'commercial',
    'food':        'EATOUT',
    'recreation':  'Entertainment',
    'civic':       'Public_Services',
    'industrial':  'industrial',
    'residential': 'residential',
}


# All land-use categories the zone table promises, even when the POI source
# (e.g. Overture Places — businesses only, no homes) yields zero rows for one.
_EXPECTED_LU = ('retail', 'education', 'health', 'office', 'food',
                'recreation', 'civic', 'industrial', 'residential', 'parking')
_EXPECTED_EST = ('est_workers', 'est_students', 'est_residents')
_EXPECTED_AMENITY = ('n_parks', 'n_doctors_clinics', 'n_pharmacy',
                     'n_schools', 'n_universities')


def _aggregate_locations(loc: pd.DataFrame) -> pd.DataFrame:
    """Per-zone counts + floor area by land_use, plus work/school totals."""
    loc = loc.copy()
    loc['zone'] = pd.to_numeric(loc['zone'], errors='coerce')
    loc = loc[loc['zone'].notna()]
    loc['zone'] = loc['zone'].astype('int64')
    g = loc.groupby(['zone', 'land_use'])
    cnt = g.size().unstack(fill_value=0)
    area = g['lu_area'].sum().unstack(fill_value=0.0)
    cnt.columns = [f'n_{c}' for c in cnt.columns]
    area.columns = [f'area_{c}' for c in area.columns]
    out = cnt.join(area, how='outer').fillna(0.0)
    # Guarantee every expected n_/area_ column even when the source has zero
    # rows for that land use (Overture Places never yields 'residential').
    for _lu in _EXPECTED_LU:
        if f'n_{_lu}' not in out.columns:
            out[f'n_{_lu}'] = 0.0
        if f'area_{_lu}' not in out.columns:
            out[f'area_{_lu}'] = 0.0
    # work/school location totals (POLARIS flags)
    for _f in ('is_work', 'is_school', 'is_discretionary', 'is_home'):
        if _f in loc.columns:
            out[f'n_{_f}'] = loc.groupby('zone')[_f].sum()
    out['n_locations'] = loc.groupby('zone').size()
    _add_amenity_counts(out, loc, 'zone')
    # estimated person capacity by kind (worker / student / resident) from the
    # location occupancy model — a data-driven employment / enrolment / residents
    # estimate per zone straight from the map. Always emit all three columns
    # (zeros when the source has no rows of that kind — e.g. Overture Places
    # has no homes, so est_residents is 0) so the downstream verify gate passes.
    if 'est_persons' in loc.columns and 'person_kind' in loc.columns:
        _cap = (loc.groupby(['zone', 'person_kind'])['est_persons'].sum()
                   .unstack(fill_value=0.0))
        _cap = _cap.rename(columns={'worker': 'est_workers', 'student': 'est_students',
                                    'resident': 'est_residents'})
        for _c in _EXPECTED_EST:
            if _c not in _cap.columns:
                _cap[_c] = 0.0
        out = out.join(_cap, how='outer')
        out['est_persons_total'] = _cap[list(_EXPECTED_EST)].sum(axis=1)
    else:
        for _c in (*_EXPECTED_EST, 'est_persons_total'):
            if _c not in out.columns:
                out[_c] = 0.0
    for _c in (*_EXPECTED_AMENITY, *_EXPECTED_EST, 'est_persons_total'):
        if _c not in out.columns:
            out[_c] = 0.0
    out = out.fillna(0.0).reset_index()
    log.info('aggregated %d locations → %d zones, %d land-use columns',
             len(loc), len(out), out.shape[1] - 1)
    return out


def _add_amenity_counts(out: pd.DataFrame, loc: pd.DataFrame, group_col: str) -> None:
    """Add destination-facing counts that are lost by broad land-use labels."""
    if 'poi_category' not in loc.columns:
        for _name in ('parks', 'doctors_clinics', 'pharmacy', 'schools', 'universities'):
            if f'n_{_name}' not in out.columns:
                out[f'n_{_name}'] = 0.0
        return
    category = loc['poi_category'].fillna('').astype(str).str.lower()
    masks = {
        'parks': category.str.contains(r'park|garden|nature_reserve', regex=True),
        'doctors_clinics': category.str.contains(
            r'clinic|doctor|physician|medical|dentist', regex=True
        ),
        'pharmacy': category.str.contains('pharmacy', regex=False),
        'schools': category.str.contains(
            r'school|kindergarten|child_care|childcare', regex=True
        ),
        'universities': category.str.contains(r'university|college', regex=True),
    }
    for name, mask in masks.items():
        counts = loc.loc[mask].groupby(group_col).size()
        # counts may be empty (no POIs of that kind) — reindex is safe where
        # Index.map(Series) can KeyError / misalign on empty or dtype mismatch.
        out[f'n_{name}'] = counts.reindex(out.index).fillna(0.0).values


def _aggregate_sa1_rows(loc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Overture locations to SA1, retaining fine destination features."""
    if 'SA1_CODE21' not in loc.columns:
        raise ValueError('SA1_CODE21 is required for SA1 land-use aggregation')
    rows = loc.copy()
    rows['SA1_CODE21'] = pd.to_numeric(rows['SA1_CODE21'], errors='coerce')
    rows = rows[rows['SA1_CODE21'].notna()].copy()
    rows['zone'] = rows['SA1_CODE21'].astype('int64')
    out = _aggregate_locations(rows).rename(columns={'zone': 'SA1_CODE21'})
    if 'SA2_CODE21' in rows.columns:
        parent = rows.groupby('zone')['SA2_CODE21'].first()
        out['SA2_CODE21'] = out['SA1_CODE21'].map(parent)
    return out


def _write_sa1_landuse(loc: pd.DataFrame) -> pd.DataFrame | None:
    """Spatially join Overture points to SA1 polygons and write a SA1 table."""
    try:
        import geopandas as gpd
    except Exception as error:
        log.warning('SA1 land-use skipped (missing geopandas: %s)', error)
        return None
    if not os.path.exists(SA1_GEOJSON):
        log.warning('SA1 land-use skipped (boundary file missing: %s)', SA1_GEOJSON)
        return None
    if not {'x', 'y'}.issubset(loc.columns):
        log.warning('SA1 land-use skipped (location table has no x/y columns)')
        return None

    try:
        sa1 = gpd.read_file(SA1_GEOJSON)
        id_col = next((c for c in ('SA1_CODE21', 'SA1_MAIN21') if c in sa1.columns), None)
        if id_col is None:
            raise ValueError(f'no SA1 identifier in {list(sa1.columns)}')
        if sa1.crs is None:
            sa1 = sa1.set_crs('EPSG:4283')
        sa1 = sa1.to_crs('EPSG:4326')
        points = gpd.GeoDataFrame(
            loc.copy(),
            geometry=gpd.points_from_xy(loc['x'], loc['y']),
            crs='EPSG:4326',
        )
        bounds = points.total_bounds
        sa1 = sa1.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
        if sa1.empty:
            log.warning('SA1 land-use skipped (no SA1 polygons intersect Overture bbox)')
            return None

        keep = [id_col, 'geometry']
        parent_col = next((c for c in ('SA2_CODE21', 'SA2_MAIN21') if c in sa1.columns), None)
        if parent_col:
            keep.insert(1, parent_col)
        joined = gpd.sjoin(
            points, sa1[keep], how='left', predicate='within'
        ).drop(columns=['geometry', 'index_right'], errors='ignore')
        joined = joined.rename(columns={id_col: 'SA1_CODE21'})
        if parent_col and parent_col != 'SA2_CODE21':
            joined = joined.rename(columns={parent_col: 'SA2_CODE21'})

        aggregate = _aggregate_sa1_rows(joined)
        base_columns = [id_col]
        if parent_col:
            base_columns.append(parent_col)
        base = sa1[base_columns].drop_duplicates().rename(
            columns={id_col: 'SA1_CODE21', 'SA2_MAIN21': 'SA2_CODE21'}
        )
        base['SA1_CODE21'] = pd.to_numeric(base['SA1_CODE21'], errors='coerce')
        result = base.merge(aggregate, on='SA1_CODE21', how='left', suffixes=('', '_poi'))
        if 'SA2_CODE21_poi' in result.columns:
            result['SA2_CODE21'] = result['SA2_CODE21'].fillna(result['SA2_CODE21_poi'])
            result = result.drop(columns=['SA2_CODE21_poi'])
        numeric = result.select_dtypes(include='number').columns
        result[numeric] = result[numeric].fillna(0.0)
        os.makedirs(os.path.dirname(SA1_LANDUSE_OUT) or '.', exist_ok=True)
        result.to_csv(SA1_LANDUSE_OUT, index=False)
        log.info('wrote SA1 Overture land use (%d SA1s, %d columns) → %s',
                 len(result), result.shape[1], SA1_LANDUSE_OUT)
        return result
    except Exception as error:
        log.warning('SA1 land-use aggregation failed (%s)', error)
        return None


# Detailed building-use buckets. Token-based against the release's `class`
# strings (vocabulary varies by release), so unknown classes fall to 'other'
# instead of dying. Union of apartments+houses == legacy res rule.
_BLDG_USES = ('apartments', 'houses', 'retail', 'office', 'industrial',
              'public', 'other')
_BLDG_USE_COLS = {'apartments': 'bldg_apt_m2', 'houses': 'bldg_house_m2',
                  'retail': 'bldg_retail_m2', 'office': 'bldg_office_m2',
                  'industrial': 'bldg_industrial_m2', 'public': 'bldg_public_m2',
                  'other': 'bldg_other_m2'}
BLDG_FLOOR_COLS = ('res_floor_m2', 'com_floor_m2',
                   'bldg_apt_m2', 'bldg_house_m2', 'bldg_retail_m2',
                   'bldg_office_m2', 'bldg_industrial_m2', 'bldg_public_m2',
                   'bldg_other_m2', 'bldg_apt_share')


def _classify_building_use(class_val: str) -> str:
    """Overture building `class` string → detailed-use bucket (never raises)."""
    try:
        c = str(class_val or '').lower()
    except Exception:
        return 'other'
    if not c or c in ('nan', 'none', 'null'):
        return 'other'
    if any(t in c for t in ('apart', 'condo', 'flat', 'dormitory', 'tenement')):
        return 'apartments'
    if any(t in c for t in ('retail', 'shop', 'mall', 'market', 'store',
                            'supermarket', 'restaurant', 'dealership')):
        return 'retail'
    if any(t in c for t in ('office', 'bank', 'commercial')):
        return 'office'
    if any(t in c for t in ('industr', 'warehouse', 'factory', 'logistic',
                            'manufacturing', 'workshop', 'storage')):
        return 'industrial'
    if any(t in c for t in ('school', 'education', 'university', 'college',
                            'hospital', 'health', 'clinic', 'civic',
                            'government', 'religious', 'church', 'library')):
        return 'public'
    if any(t in c for t in ('residential', 'house', 'dwelling', 'cabin',
                            'bungalow', 'terrace', 'detached')):
        return 'houses'
    return 'other'


def _overture_buildings_floor_area():
    """Optional: Overture buildings floor area (m²) per SA2, via DuckDB (bbox)
    + geopandas spatial join. Returns df[zone, res_floor_m2, com_floor_m2,
    bldg_apt_m2, bldg_house_m2, bldg_retail_m2, bldg_office_m2,
    bldg_industrial_m2, bldg_public_m2, bldg_apt_share] or None.
    Needs internet egress (unless OVERTURE_BUILDINGS_PARQUET).

    res/com keep their legacy names but now derive from the clean use cut
    (apartments+houses == res by construction; this also fixes the old inline
    substring rule that filed e.g. 'warehouse' under res via 'house').
    The bldg_* splits are the detailed-use cut (apartments vs houses for resident capacity; retail/office/industrial
    floors as job-capacity cross-checks against the POI occupancy model).
    All bucket columns are always emitted (zeros when a release has no such
    class) so the schema is stable with/without buildings.
    """
    try:
        import duckdb
        import geopandas as gpd
        from shapely import wkb
    except Exception as e:
        log.warning('buildings step skipped (missing duckdb/geopandas: %s)', e)
        return None
    try:
        sa2 = gpd.read_file(SA2_GEOJSON).to_crs(4326)
        _zcol = next((c for c in ('SA2_MAIN16', 'SA2_MAIN21', 'ORDER', 'zone', 'zone_id')
                      if c in sa2.columns), sa2.columns[0])
        b = sa2.total_bounds  # xmin,ymin,xmax,ymax
        con = duckdb.connect(); con.execute("INSTALL spatial; LOAD spatial;")
        local = os.environ.get('OVERTURE_BUILDINGS_PARQUET', '')
        if local and os.path.exists(local):
            src = f"read_parquet('{local}')"
        else:
            con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
            rel = os.environ.get('OVERTURE_RELEASE', '2026-08-19.0')
            src = (f"read_parquet('s3://overturemaps-us-west-2/release/{rel}"
                   f"/theme=buildings/type=building/*', hive_partitioning=1)")
        # geometry storage differs by release: newer releases serve a native
        # GEOMETRY type (ST_GeomFromWKB on it raises a Binder error), older
        # ones serve WKB blobs (need the decode). Probe once, then query.
        _gtype = con.execute(
            f"SELECT typeof(geometry) FROM {src} LIMIT 1").fetchone()[0]
        # Native GEOMETRY (e.g. GEOMETRY('OGC:CRS84')) must be used as-is;
        # ST_GeomFromWKB on it raises a Binder error (location_table_ovm job
        # 25233620). Only WKB blobs need the decode.
        _geom_expr = ("geometry" if 'GEOMETRY' in str(_gtype).upper()
                      else "ST_GeomFromWKB(geometry)")
        q = f"""SELECT ST_AsWKB({_geom_expr}) AS geom,
                       COALESCE(num_floors, 1) AS floors, class
                FROM {src}
                WHERE bbox.xmin BETWEEN {b[0]} AND {b[2]}
                  AND bbox.ymin BETWEEN {b[1]} AND {b[3]}"""
        bdf = con.execute(q).df()
        if not len(bdf):
            return None
        # Defensive WKB decode: a few rows come back as non-bytes (null/odd
        # driver coercions on 1.9M-row pulls). Drop them instead of dying.
        def _wkb_safe(g):
            try:
                if isinstance(g, (bytes, bytearray, memoryview)):
                    return wkb.loads(bytes(g))
                if isinstance(g, str):
                    return wkb.loads(g)
            except Exception:
                pass
            return None
        bdf['geometry'] = bdf['geom'].apply(_wkb_safe)
        bdf = bdf[bdf['geometry'].notna()].reset_index(drop=True)
        if not len(bdf):
            return None
        gb = gpd.GeoDataFrame(bdf.drop(columns='geom'), geometry='geometry', crs=4326).to_crs(3577)
        gb['foot_m2'] = gb.geometry.area
        gb['floor_m2'] = gb['foot_m2'] * gb['floors'].clip(lower=1)
        # Detailed use first; legacy res/com derives from it. (The old inline
        # rule matched 'residential|house|apartments' by substring, which put
        # e.g. 'warehouse' into res via 'house' — fixed by construction here.)
        gb['use'] = gb['class'].map(_classify_building_use)
        gb['kind'] = np.where(gb['use'].isin(['apartments', 'houses']), 'res', 'com')
        j = gpd.sjoin(gb.to_crs(4326), sa2[[_zcol, 'geometry']], predicate='within', how='inner')
        piv = j.pivot_table(index=_zcol, columns='kind', values='floor_m2', aggfunc='sum', fill_value=0.0)
        # Buildings pull may contain only one kind (e.g. all 'com') — guarantee
        # both columns so the output schema is stable with/without buildings.
        for _k in ('res', 'com'):
            if _k not in piv.columns:
                piv[_k] = 0.0
        piv = piv.rename(columns={'res': 'res_floor_m2', 'com': 'com_floor_m2'})
        # Detailed-use cut (apartments/houses/retail/office/industrial/public).
        upiv = j.pivot_table(index=_zcol, columns='use', values='floor_m2', aggfunc='sum', fill_value=0.0)
        for _k in _BLDG_USES:
            if _k not in upiv.columns:
                upiv[_k] = 0.0
        for _u, _col in _BLDG_USE_COLS.items():
            piv[_col] = upiv[_u]
        _res = (piv['res_floor_m2'].replace(0, np.nan))
        piv['bldg_apt_share'] = (piv['bldg_apt_m2'] / _res).fillna(0.0).clip(0, 1)
        piv = piv.reset_index()
        piv = piv.rename(columns={_zcol: 'zone'})
        log.info('Overture buildings: floor area for %d zones (%.0f apt-m²)', len(piv), piv['bldg_apt_m2'].sum())
        return piv
    except Exception as e:
        log.warning('buildings step failed (%s)', e)
        return None


_FREEWAY_TOKENS = ('motorway', 'trunk')
_ARTERIAL_TOKENS = ('primary', 'secondary', 'tertiary')


def _classify_road_kind(class_val: str) -> str:
    """Road-class string → 'freeway' / 'arterial' / 'local' (never raises)."""
    try:
        c = str(class_val or '').lower()
    except Exception:
        return 'local'
    if any(t in c for t in _FREEWAY_TOKENS):
        return 'freeway'
    if any(t in c for t in _ARTERIAL_TOKENS):
        return 'arterial'
    return 'local'


def _transport_class_column(cols) -> str | None:
    """Pick the road-class column from whatever the release serves."""
    for cand in ('class', 'subtype', 'highway', 'road_class', 'functional_class'):
        if cand in cols:
            return cand
    # 'road' is a struct/array of road refs in some releases — stringified fallback
    if 'road' in cols:
        return 'road'
    return None


def _aggregate_road_by_zone(roads: pd.DataFrame, zone_key: pd.Series) -> pd.DataFrame:
    """Pure aggregation: per-zone road_km / freeway_km / arterial_km / share.

    `roads` needs ['length_km', 'kind']; `zone_key` the SA2 code per row.
    Returns df[zone, road_km, freeway_km, arterial_km, freeway_share].
    """
    r = pd.DataFrame({'zone': pd.to_numeric(zone_key, errors='coerce'),
                      'length_km': pd.to_numeric(roads['length_km'], errors='coerce').fillna(0.0),
                      'kind': roads['kind'].astype(str)})
    r = r[r['zone'].notna()].copy()
    tot = r.groupby('zone')['length_km'].sum()
    fw = r[r['kind'] == 'freeway'].groupby('zone')['length_km'].sum()
    art = r[r['kind'] == 'arterial'].groupby('zone')['length_km'].sum()
    out = pd.DataFrame({'road_km': tot}).join(
        pd.DataFrame({'freeway_km': fw}), how='left').join(
        pd.DataFrame({'arterial_km': art}), how='left').fillna(0.0)
    out['freeway_share'] = (out['freeway_km'] / out['road_km'].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    return out.reset_index()


def _overture_transport_network():
    """Optional: Overture TRANSPORTATION road length per SA2.

    Exogenous capacity proxies only (geometry, not volumes — Overture carries
    no traffic counts). Returns df[zone, road_km, freeway_km, arterial_km,
    freeway_share] or None. Mode via OVERTURE_TRANSPORT: 'auto' (default)
    uses a local parquet if present else skips; '1' queries Overture S3
    (needs internet egress); '0' skips.
    """
    if OVERTURE_TRANSPORT_MODE in ('0', 'no', 'off', 'false'):
        return None
    try:
        import duckdb
        import geopandas as gpd
        from shapely import wkb
    except Exception as e:
        log.warning('transport step skipped (missing duckdb/geopandas: %s)', e)
        return None
    try:
        sa2 = gpd.read_file(SA2_GEOJSON).to_crs(4326)
        _zcol = next((c for c in ('SA2_MAIN16', 'SA2_MAIN21', 'ORDER', 'zone', 'zone_id')
                      if c in sa2.columns), sa2.columns[0])
        b = sa2.total_bounds
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        local = OVERTURE_TRANSPORT_PARQUET
        if local and os.path.exists(local):
            src = f"read_parquet('{local}')"
            log.info('reading Overture transport segments from local file %s', local)
        elif OVERTURE_TRANSPORT_MODE == '1':
            con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
            rel = os.environ.get('OVERTURE_RELEASE', '2026-08-19.0')
            src = (f"read_parquet('s3://overturemaps-us-west-2/release/{rel}"
                   f"/theme=transportation/type=segment/*', hive_partitioning=1)")
            log.info('querying Overture S3 transport segments release %s (needs egress)', rel)
        else:
            log.info('transport step skipped (no local parquet at %s; set OVERTURE_TRANSPORT=1 for S3)', local)
            return None
        cols = [r[0] for r in con.execute(f"SELECT * FROM {src} LIMIT 0").description]
        _gtype = con.execute(f"SELECT typeof(geometry) FROM {src} LIMIT 1").fetchone()[0]
        _geom_expr = ("geometry" if 'GEOMETRY' in str(_gtype).upper()
                      else "ST_GeomFromWKB(geometry)")
        _class_col = _transport_class_column(cols)
        _extra = f", {_class_col} AS class_val" if _class_col else ", NULL AS class_val"
        q = f"""SELECT ST_AsWKB({_geom_expr}) AS geom{_extra}
                FROM {src}
                WHERE bbox.xmin BETWEEN {b[0]} AND {b[2]}
                  AND bbox.ymin BETWEEN {b[1]} AND {b[3]}"""
        sdf = con.execute(q).df()
        if not len(sdf):
            return None

        def _wkb_safe(g):
            try:
                if isinstance(g, (bytes, bytearray, memoryview)):
                    return wkb.loads(bytes(g))
                if isinstance(g, str):
                    return wkb.loads(g)
            except Exception:
                pass
            return None

        sdf['geometry'] = sdf['geom'].apply(_wkb_safe)
        sdf = sdf[sdf['geometry'].notna()].reset_index(drop=True)
        if not len(sdf):
            return None
        gdf = gpd.GeoDataFrame(sdf.drop(columns='geom'), geometry='geometry', crs=4326).to_crs(3577)
        gdf['length_km'] = gdf.geometry.length / 1000.0
        gdf['kind'] = gdf['class_val'].map(_classify_road_kind) if 'class_val' in gdf.columns \
            else 'local'
        j = gpd.sjoin(gdf.to_crs(4326)[['length_km', 'kind', 'geometry']],
                      sa2[[_zcol, 'geometry']], predicate='intersects', how='inner')
        out = _aggregate_road_by_zone(j, j[_zcol])
        log.info('Overture transport: %.0f road-km across %d zones (%.0f freeway-km)',
                 out['road_km'].sum(), len(out), out['freeway_km'].sum())
        return out
    except Exception as e:
        log.warning('transport step failed (%s)', e)
        return None


def main():
    if not os.path.exists(LOCATION_TABLE):
        raise SystemExit(f'FATAL: {LOCATION_TABLE} not found — run job_location_table_overture.pbs first')
    loc = pd.read_csv(LOCATION_TABLE)
    agg = _aggregate_locations(loc)

    if os.environ.get('OVERTURE_SA1_LANDUSE', '1') == '1':
        _write_sa1_landuse(loc)

    # merge onto existing landuse (keep population/employment; refresh POI cols)
    if os.path.exists(ZONE_LANDUSE_IN):
        lu = pd.read_csv(ZONE_LANDUSE_IN)
        _zcol = next((c for c in ('SA2_MAIN16', 'ORDER', 'zone', 'zone_id') if c in lu.columns), lu.columns[0])
        lu['_zone'] = pd.to_numeric(lu[_zcol], errors='coerce').astype('Int64')
        merged = lu.merge(agg, left_on='_zone', right_on='zone', how='left')
        # Left join leaves NaN overture cols for zones with zero POIs (350 of
        # 371 zones have POIs) — zero-fill so size terms are never NaN downstream.
        _agg_cols = [c for c in agg.columns if c != 'zone']
        merged[_agg_cols] = merged[_agg_cols].fillna(0.0)
        # refresh the matching existing size-term columns with the Overture counts
        # Batch the new *_ovm columns via concat (single insert → no
        # DataFrame fragmentation PerformanceWarning, job 25256775).
        _ovm_new = {}
        for _luc, _existing in _LU_TO_EXISTING.items():
            _src = f'n_{_luc}'
            # agg now always carries every n_<lu> (zeros if source had none),
            # so every <existing>_ovm is always emitted — no silent drops.
            if _src in merged.columns:
                _fill = merged[_src].fillna(0.0) if _src in merged.columns else 0.0
                _ovm_new[f'{_existing}_ovm'] = _fill
        if _ovm_new:
            merged = pd.concat([merged, pd.DataFrame(_ovm_new, index=merged.index)], axis=1)
        merged = merged.drop(columns=[c for c in ('_zone', 'zone') if c in merged.columns])
        log.info('merged Overture aggregates onto %d existing zones (added %d new cols)',
                 len(merged), agg.shape[1] - 1)
    else:
        merged = agg
        log.warning('no existing zone_landuse at %s — writing Overture aggregates only', ZONE_LANDUSE_IN)

    if os.environ.get('OVERTURE_BUILDINGS', '0') == '1':
        _b = _overture_buildings_floor_area()
        if _b is not None:
            _zcol = next((c for c in ('SA2_MAIN16', 'ORDER', 'zone', 'zone_id') if c in merged.columns), merged.columns[0])
            merged = merged.merge(_b, left_on=_zcol, right_on='zone', how='left').drop(
                columns=[c for c in ('zone',) if c in merged.columns])

    # Exogenous network capacity (Overture TRANSPORTATION geometry — never
    # volumes). Auto-uses a local parquet when present, else skips unless
    # OVERTURE_TRANSPORT=1 (S3, needs egress). Zero-fills zones with no
    # segments so the columns are never NaN downstream.
    _t = _overture_transport_network()
    if _t is not None:
        _zcol = next((c for c in ('SA2_MAIN16', 'ORDER', 'zone', 'zone_id') if c in merged.columns),
                     merged.columns[0])
        merged = merged.merge(_t, left_on=_zcol, right_on='zone', how='left')
        if _zcol != 'zone':
            merged = merged.drop(columns=[c for c in ('zone',) if c in merged.columns])
        for _c in TRANSPORT_COLS:
            if _c in merged.columns:
                merged[_c] = pd.to_numeric(merged[_c], errors='coerce').fillna(0.0)
        log.info('merged transport network onto %d zones (+%d cols)',
                 len(merged), sum(c in merged.columns for c in TRANSPORT_COLS))

    # OSM + transit supply (build_zone_supply.py — parking, micromobility,
    # GTFS service intensity). Merged when present, zero-filled otherwise.
    for _path, _cols, _tag in ((SUPPLY_OSM_PATH, SUPPLY_COLS, 'OSM supply'),
                               (TRANSIT_SUPPLY_PATH, TRANSIT_COLS, 'transit supply')):
        if os.path.exists(_path):
            try:
                _s = pd.read_csv(_path)
                _skey = next((c for c in ('zone', 'SA2_MAIN16', 'ORDER', 'zone_id') if c in _s.columns), None)
                if _skey is None:
                    log.warning('%s skipped (no zone key in %s)', _tag, _path)
                    continue
                _zcol = next((c for c in ('SA2_MAIN16', 'ORDER', 'zone', 'zone_id') if c in merged.columns),
                             merged.columns[0])
                _add = [c for c in _cols if c in _s.columns]
                _m = _s.assign(_sk=pd.to_numeric(_s[_skey], errors='coerce')).dropna(subset=['_sk'])
                _m['_sk'] = _m['_sk'].astype('int64')
                _mk = pd.to_numeric(merged[_zcol], errors='coerce')
                merged['_sk'] = _mk.astype('Int64')
                # Overlap rule: Overture POI counts win their names; the OSM
                # twin lands as {col}_osm (independent cross-check, e.g. the
                # same car parks mapped twice) instead of double-counting.
                _ren = {c: (f'{c}_osm' if c in merged.columns else c) for c in _add}
                _m = _m.rename(columns=_ren)
                _add2 = [_ren[c] for c in _add]
                merged = merged.merge(_m[['_sk'] + _add2], on='_sk', how='left')
                for _c in _add2:
                    if _c in merged.columns:
                        merged[_c] = pd.to_numeric(merged[_c], errors='coerce').fillna(0.0)
                merged = merged.drop(columns=['_sk'])
                log.info('merged %s onto %d zones (+%d cols)', _tag, len(merged), len(_add2))
            except Exception as _se:
                log.warning('%s merge failed (non-fatal): %s', _tag, _se)

    # ── Orthogonalised occupancy residuals ───────────────────────────────
    # Why: est_workers/est_students come from the POI occupancy model
    # (est_persons = lu_area×storeys/GFA-per-person with DEFAULT areas per
    # land_use — every Overture POI of one land_use gets the SAME persons, so
    # zone totals track establishment COUNTS, not employment). Totals align
    # with census (~1.9M) but the zone distribution doesn't (job 25256775:
    # r(est_workers,TOTEMP)=-0.015, r(est_students,Primary)=0.107).
    # Rather than recalibrating densities, split the signal honestly:
    #   census column  = the LEVEL (existing size terms keep using it)
    #   residual column = the Overture NEWS beyond census (mean-zero, ~uncorr
    #     with the base by construction), safe to enter utilities alongside
    #     the census term without double-counting size.
    # Same pooled-OLS residual semantics as the stage-5 orthogonalize_on
    # pairs (resid = y - beta*x on finite-mask, var-guarded). Log1p scale
    # matches the model's log-size handling. Disable with ORTHO_EST=0.
    try:
        if os.environ.get('ORTHO_EST', '1') == '1':
            _ortho_new = {}

            def _ortho_resid(y_col, x_cols, out_col):
                try:
                    if y_col not in merged.columns:
                        return
                    _y = np.log1p(pd.to_numeric(merged[y_col], errors='coerce').fillna(0.0).clip(lower=0.0).values.astype(float))
                    _x_parts = []
                    for _xc in x_cols:
                        if _xc in merged.columns:
                            _x_parts.append(pd.to_numeric(merged[_xc], errors='coerce').fillna(0.0).clip(lower=0.0).values.astype(float))
                    if not _x_parts:
                        return
                    _x = np.log1p(sum(_x_parts))
                    _mask = np.isfinite(_y) & np.isfinite(_x)
                    if int(_mask.sum()) <= 10:
                        return
                    if float(np.var(_x[_mask])) <= 1e-12 or float(np.var(_y[_mask])) <= 1e-12:
                        return
                    _beta = float(np.cov(_x[_mask], _y[_mask])[0, 1] / (float(np.var(_x[_mask])) + 1e-12))
                    if not np.isfinite(_beta):
                        return
                    _resid = _y - _beta * _x
                    _resid[~_mask] = 0.0
                    _ortho_new[out_col] = _resid.astype(np.float64)
                    _r = float(np.corrcoef(_y[_mask], _x[_mask])[0, 1]) if _mask.sum() > 2 else float('nan')
                    _rr = float(np.corrcoef(_resid[_mask], _x[_mask])[0, 1]) if _mask.sum() > 2 else float('nan')
                    log.info('ortho occupancy: %s = log1p(%s) - %.4f*log1p(%s) '
                             '[raw r=%.3f → resid r=%.3f, n=%d]',
                             out_col, y_col, _beta, '+'.join([c for c in x_cols if c in merged.columns]),
                             _r, _rr, int(_mask.sum()))
                except Exception as _oe:
                    log.warning('ortho occupancy %s skipped (non-fatal): %s', out_col, _oe)

            _ortho_resid('est_workers', ['TOTEMP'], 'est_workers_resid')
            _ortho_resid('est_students', ['Primary', 'Secondary', 'Education', 'HSENROLL'],
                         'est_students_resid')
            if _ortho_new:
                merged = pd.concat([merged, pd.DataFrame(_ortho_new, index=merged.index)], axis=1)
    except Exception as _oe:
        log.warning('orthogonalised occupancy residuals failed (non-fatal): %s', _oe)

    # Derived mix metrics (no new data — combinations of columns above).
    # lu_entropy: POI land-use diversity (destination variety attractor).
    # jobs_housing_ratio: est_workers vs residents (requires TOTPOP base col).
    # Built via concat (single insert) to avoid fragmentation warnings.
    try:
        _derived = {}
        _ncols = [f'n_{lu}' for lu in _EXPECTED_LU if f'n_{lu}' in merged.columns]
        if _ncols:
            _sh = merged[_ncols].div(merged[_ncols].sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                _derived['lu_entropy'] = (-_sh * np.log(_sh.replace(0, np.nan))).sum(axis=1).fillna(0.0)
        if 'est_workers' in merged.columns and 'TOTPOP' in merged.columns:
            _pop = pd.to_numeric(merged['TOTPOP'], errors='coerce').fillna(0.0)
            _w = pd.to_numeric(merged['est_workers'], errors='coerce').fillna(0.0)
            _derived['jobs_housing_ratio'] = (_w / _pop.replace(0, np.nan)).fillna(0.0).clip(0, 50)
        if _derived:
            merged = pd.concat([merged, pd.DataFrame(_derived, index=merged.index)], axis=1)
    except Exception as _de:
        log.warning('derived mix metrics failed (non-fatal): %s', _de)

    # Pre-write gate (07/09 lesson): a rebuild from a stale location table
    # silently DROPPED res_floor_m2/com_floor_m2 (previous buildings-derived
    # columns) because the no-buildings default never re-adds them. Carry
    # forward the PREVIOUS VALUES by zone key instead of zeros (job 25256775:
    # 34 cols were zeroed, wiping real buildings/jobs data). Zeros only as a
    # last resort when the previous file/key is unavailable.
    if os.path.exists(ZONE_LANDUSE_OUT):
        try:
            _prev_cols_only = pd.read_csv(ZONE_LANDUSE_OUT, nrows=0)
            _prev_cols = set(_prev_cols_only.columns)
            _dropped = sorted(_prev_cols - set(merged.columns))
            if _dropped:
                log.warning('PRE-WRITE GATE: %d columns absent from new build — '
                            'carrying forward PREVIOUS VALUES by zone (not zeros): %s.',
                            len(_dropped), _dropped)
                try:
                    _prev_full = pd.read_csv(ZONE_LANDUSE_OUT, usecols=lambda c: c in _dropped or True)
                    _z_merged = next((c for c in ('SA2_MAIN16', 'ORDER', 'zone', 'zone_id') if c in merged.columns), merged.columns[0])
                    _z_prev = next((c for c in ('SA2_MAIN16', 'ORDER', 'zone', 'zone_id') if c in _prev_full.columns), None)
                    if _z_prev is not None:
                        _keep = [_z_prev] + [c for c in _dropped if c in _prev_full.columns]
                        _pf = _prev_full[_keep].copy()
                        _pf['_join_key'] = pd.to_numeric(_pf[_z_prev], errors='coerce').astype('Int64')
                        _mk = pd.to_numeric(merged[_z_merged], errors='coerce').astype('Int64')
                        merged['_join_key'] = _mk
                        _carry = _pf.drop(columns=[_z_prev]).drop_duplicates('_join_key')
                        merged = merged.merge(_carry, on='_join_key', how='left', suffixes=('', '_carried'))
                        # Merged carry columns arrive as-is (no suffix since names
                        # were absent); any still-missing values → 0 with a note.
                        for _c in _dropped:
                            if _c in merged.columns:
                                _n_na = int(merged[_c].isna().sum())
                                if _n_na:
                                    merged[_c] = merged[_c].fillna(0.0)
                                    log.info('PRE-WRITE GATE: %s carried forward, %d zones without prior value → 0', _c, _n_na)
                            else:
                                merged[_c] = 0.0
                        merged = merged.drop(columns=['_join_key'])
                    else:
                        raise ValueError('no zone key in previous output')
                except Exception as _ce:
                    log.warning('PRE-WRITE GATE carry-forward join failed (%s) — zero-filling instead', _ce)
                    _zero_fill = { _c: 0.0 for _c in _dropped if _c not in merged.columns }
                    if _zero_fill:
                        merged = pd.concat([merged, pd.DataFrame(_zero_fill, index=merged.index)], axis=1)
        except Exception as _ge:                            # noqa: BLE001
            log.warning('pre-write gate check failed (non-fatal): %s', _ge)
    # Defrag before write (many merges/concats above fragment the frame).
    try:
        merged = merged.copy()
    except Exception:
        pass
    # Final NaN gate: out-of-network zones (60/371, in_emme_network=False)
    # miss the Overture amenity block entirely. Counts → 0 (no mapped jobs =
    # zero attraction, the correct semantic — NaN would poison any utility
    # using them); labels → 'Unknown'; sa2_area → reference area (never 0 —
    # densities divide by it).
    try:
        _num_cols = list(merged.select_dtypes(include=[np.number]).columns)
        if _num_cols:
            merged[_num_cols] = merged[_num_cols].fillna(0.0)
        for _c in merged.select_dtypes(exclude=[np.number]).columns:
            merged[_c] = merged[_c].fillna('Unknown')
        if 'sa2_area' in merged.columns and 'SA2_MAIN16' in merged.columns:
            _need_area = merged['sa2_area'].isna() | (merged['sa2_area'] <= 0)
            if _need_area.any():
                _ref_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'saved_data', 'seq_zone_ids_371.csv')
                _ref = pd.read_csv(_ref_path, usecols=['SA2_MAIN16', 'area_sqkm'],
                                   dtype={'SA2_MAIN16': str})
                _area_map = dict(zip(
                    _ref['SA2_MAIN16'].str.strip(),
                    pd.to_numeric(_ref['area_sqkm'], errors='coerce') * 1e6))
                _fill = (merged['SA2_MAIN16'].astype(str).str.strip().map(_area_map))
                merged.loc[_need_area, 'sa2_area'] = _fill[_need_area].fillna(1e7)
                log.info('final NaN gate: %d zones sa2_area from reference areas',
                         int(_need_area.sum()))
    except Exception as _fe:                            # noqa: BLE001
        log.warning('final NaN gate failed (non-fatal): %s', _fe)
    os.makedirs(os.path.dirname(ZONE_LANDUSE_OUT) or '.', exist_ok=True)
    merged.to_csv(ZONE_LANDUSE_OUT, index=False)
    log.info('wrote comprehensive zone land use (%d zones, %d cols) → %s',
             len(merged), merged.shape[1], ZONE_LANDUSE_OUT)


if __name__ == '__main__':
    main()
