"""
spatial_assignment.py
======================
Two-stage spatial assignment after the ABM schedule is complete:

PART 1 — HOUSEHOLD MATCHING
    Assigns every synthetic person to a real dwelling using a
    combinatorial matching process that considers:
        - household size (nprd) vs dwelling bedrooms
        - dwelling type vs household composition
        - income band vs dwelling value / rent proxy
        - zone (OTAZ) — dwelling must be in the correct SA2

PART 2 — ACTIVITY LOCATION PINNING
    Takes the completed schedule (Stage 6 output) and pins each
    activity to a real OSM point-of-interest in the destination zone,
    using purpose → OSM tag mappings:
        work      → office, industrial, commercial landuse
        edu       → school, university, kindergarten
        shopping  → shop=*, supermarket, mall
        eat       → amenity=restaurant/cafe/fast_food
        social    → leisure=*, amenity=cinema/theatre/park
        escort    → childcare, school (drop-off context)
        othmaint  → amenity=hospital/pharmacy/bank/post_office
        othdiscr  → amenity=fuel/car_repair, misc services

Both parts use OSM data fetched via the Overpass API.
Results are attached as lat/lon columns to the persons and schedule tables.

Requirements
------------
    pip install osmnx requests shapely

Usage
-----
    from spatial_assignment import (
        fetch_dwellings,
        match_households,
        fetch_activity_locations,
        pin_activity_locations,
        run_spatial_assignment,
    )

    persons  = pd.read_csv('saved_data/persons_processed.csv')
    schedule = pd.read_csv('saved_data/stage6_output.csv')
    landuse  = sc.get_land()

    # SA2 → bbox lookup (you need SA2 centroids or bboxes)
    sa2_geo  = pd.read_csv('SA1_SHP/sa2_centroids.csv')  # SA2, lat, lon

    persons_geo, schedule_geo = run_spatial_assignment(
        persons, schedule, landuse, sa2_geo
    )
    persons_geo.to_csv('saved_data/persons_with_locations.csv', index=False)
    schedule_geo.to_csv('saved_data/schedule_with_locations.csv', index=False)
"""

import numpy as np
import pandas as pd
import logging
import os
import json
import time
import warnings
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# OSM tags per activity purpose
PURPOSE_OSM_TAGS = {
    'work': [
        {'landuse': 'commercial'},
        {'landuse': 'industrial'},
        {'landuse': 'office'},
        {'building': 'office'},
        {'building': 'commercial'},
        {'office': True},
    ],
    'edu': [
        {'amenity': 'school'},
        {'amenity': 'university'},
        {'amenity': 'college'},
        {'amenity': 'kindergarten'},
        {'building': 'school'},
        {'building': 'university'},
    ],
    'shopping': [
        {'shop': True},
        {'amenity': 'marketplace'},
        {'building': 'retail'},
        {'landuse': 'retail'},
        {'shop': 'supermarket'},
        {'shop': 'mall'},
    ],
    'eat': [
        {'amenity': 'restaurant'},
        {'amenity': 'cafe'},
        {'amenity': 'fast_food'},
        {'amenity': 'food_court'},
        {'amenity': 'bar'},
        {'amenity': 'pub'},
    ],
    'social': [
        {'leisure': 'park'},
        {'leisure': 'sports_centre'},
        {'leisure': 'fitness_centre'},
        {'amenity': 'cinema'},
        {'amenity': 'theatre'},
        {'amenity': 'community_centre'},
        {'leisure': 'recreation_ground'},
        {'amenity': 'library'},
    ],
    'escort': [
        {'amenity': 'kindergarten'},
        {'amenity': 'school'},
        {'amenity': 'childcare'},
        {'amenity': 'community_centre'},
    ],
    'othmaint': [
        {'amenity': 'hospital'},
        {'amenity': 'clinic'},
        {'amenity': 'pharmacy'},
        {'amenity': 'bank'},
        {'amenity': 'post_office'},
        {'amenity': 'doctors'},
        {'office': 'government'},
    ],
    'othdiscr': [
        {'amenity': 'fuel'},
        {'shop': 'car_repair'},
        {'amenity': 'car_wash'},
        {'shop': 'hardware'},
        {'amenity': 'place_of_worship'},
        {'leisure': 'golf_course'},
    ],
}

# Dwelling type OSM tags
DWELLING_OSM_TAGS = [
    {'building': 'residential'},
    {'building': 'house'},
    {'building': 'apartments'},
    {'building': 'detached'},
    {'building': 'semidetached_house'},
    {'building': 'terrace'},
    {'building': 'flat'},
    {'landuse': 'residential'},
]

# Industry → work purpose OSM override
INDUSTRY_OSM_TAGS = {
    'Retail':          [{'shop': True}, {'landuse': 'retail'}],
    'Education':       [{'amenity': 'school'}, {'amenity': 'university'}],
    'Health':          [{'amenity': 'hospital'}, {'amenity': 'clinic'}],
    'Manufacturing':   [{'landuse': 'industrial'}, {'building': 'industrial'}],
    'Construction':    [{'landuse': 'construction'}, {'building': 'construction'}],
    'Government':      [{'office': 'government'}, {'building': 'government'}],
    'Transport':       [{'amenity': 'bus_station'}, {'landuse': 'transport'}],
    'Finance':         [{'office': 'financial'}, {'building': 'commercial'}],
    'Agriculture':     [{'landuse': 'farmyard'}, {'landuse': 'farm'}],
    'Accommodation':   [{'tourism': 'hotel'}, {'building': 'hotel'}],
}

# Overpass API endpoint
OVERPASS_URL = 'https://overpass-api.de/api/interpreter'


# ---------------------------------------------------------------------------
# 0.  OVERPASS / OSM HELPERS
# ---------------------------------------------------------------------------

def _overpass_query(query: str, retries: int = 3,
                    wait: float = 2.0) -> Optional[dict]:
    """Execute an Overpass QL query and return parsed JSON."""
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    for attempt in range(retries):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={'data': query},
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Overpass HTTP {resp.status_code}, retry {attempt+1}")
        except Exception as e:
            logger.warning(f"Overpass error: {e}, retry {attempt+1}")
        time.sleep(wait * (attempt + 1))
    return None


def _bbox_from_centroid(lat: float, lon: float,
                        radius_km: float = 3.0) -> Tuple[float, float, float, float]:
    """Approx bounding box around a centroid."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def _build_overpass_query(bbox: Tuple, tags: List[dict],
                          element_types: str = 'node way') -> str:
    """Build Overpass QL for a list of tag dicts within a bbox."""
    s, w, n, e = bbox
    union_parts = []
    for tag_dict in tags:
        for k, v in tag_dict.items():
            val_str = '' if v is True else f'="{v}"'
            for el in element_types.split():
                union_parts.append(
                    f'{el}["{k}"{val_str}]({s},{w},{n},{e});'
                )
    query = (
        '[out:json][timeout:30];'
        '(' + ''.join(union_parts) + ');'
        'out center;'
    )
    return query


def _parse_elements(data: dict) -> pd.DataFrame:
    """Parse Overpass JSON response to DataFrame with lat/lon."""
    rows = []
    for el in data.get('elements', []):
        if el['type'] == 'node':
            lat, lon = el.get('lat'), el.get('lon')
        else:
            center = el.get('center', {})
            lat, lon = center.get('lat'), center.get('lon')
        if lat is None or lon is None:
            continue
        tags = el.get('tags', {})
        rows.append({
            'osm_id':   el['id'],
            'osm_type': el['type'],
            'lat':      lat,
            'lon':      lon,
            'name':     tags.get('name', ''),
            **{k: v for k, v in tags.items()},
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['osm_id', 'osm_type', 'lat', 'lon', 'name']
    )


def fetch_osm_pois(
    lat: float, lon: float,
    tags: List[dict],
    radius_km: float = 3.0,
    cache_dir: Optional[str] = 'saved_data/osm_cache',
) -> pd.DataFrame:
    """
    Fetch OSM points of interest near a centroid.

    Results are cached to disk by (lat_r, lon_r, tag_hash) so repeated
    calls for the same zone don't hit Overpass again.

    Parameters
    ----------
    lat, lon    : centroid coordinates
    tags        : list of OSM tag dicts
    radius_km   : search radius
    cache_dir   : directory for caching; None = no cache

    Returns
    -------
    DataFrame with osm_id, lat, lon, name, and all OSM tags as columns.
    """
    # cache key
    tag_key = str(sorted(str(t) for t in tags))
    cache_key = f'{lat:.4f}_{lon:.4f}_{hash(tag_key) % 100000}'

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'{cache_key}.json')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return pd.DataFrame(json.load(f))

    bbox  = _bbox_from_centroid(lat, lon, radius_km)
    query = _build_overpass_query(bbox, tags)
    data  = _overpass_query(query)

    if data is None:
        return pd.DataFrame(columns=['osm_id', 'lat', 'lon', 'name'])

    df = _parse_elements(data)

    if cache_dir and not df.empty:
        with open(cache_path, 'w') as f:
            json.dump(df[['osm_id','lat','lon','name']].to_dict('records'), f)

    return df


# ---------------------------------------------------------------------------
# 1.  DWELLING FETCH + HOUSEHOLD MATCHING
# ---------------------------------------------------------------------------

def fetch_dwellings(
    sa2_geo: pd.DataFrame,
    zone_ids: List[int],
    radius_km: float = 2.5,
    cache_dir: str = 'saved_data/osm_cache',
) -> pd.DataFrame:
    """
    Fetch residential dwellings from OSM for a list of zone IDs.

    Parameters
    ----------
    sa2_geo  : DataFrame with columns: SA2_MAIN16 (or ORDER), lat, lon
    zone_ids : list of zone IDs to fetch
    radius_km: search radius around each zone centroid

    Returns
    -------
    DataFrame with columns: zone_id, osm_id, lat, lon,
                             building_type, est_bedrooms
    """
    # normalise zone column
    zc = next((c for c in ['SA2_MAIN16', 'ORDER', 'DTAZ', 'zone', 'zone_id']
               if c in sa2_geo.columns), None)
    if zc is None:
        raise ValueError("sa2_geo must have SA2_MAIN16, ORDER, DTAZ, zone, or zone_id column.")
    sa2_geo = sa2_geo.rename(columns={zc: 'zone_id'})

    all_dws = []
    total   = len(zone_ids)

    for i, zid in enumerate(zone_ids):
        row = sa2_geo[sa2_geo['zone_id'] == zid]
        if row.empty:
            continue

        lat = float(row['lat'].iloc[0])
        lon = float(row['lon'].iloc[0])
        print(f"  Fetching dwellings zone {zid} ({i+1}/{total}) "
              f"@ {lat:.4f},{lon:.4f}")

        dw = fetch_osm_pois(lat, lon, DWELLING_OSM_TAGS,
                            radius_km=radius_km, cache_dir=cache_dir)
        if dw.empty:
            continue

        dw['zone_id'] = zid

        # estimate bedrooms from building type
        def est_beds(btype):
            if btype in ['apartments', 'flat']:
                return 2
            if btype in ['house', 'detached', 'terrace', 'semidetached_house']:
                return 3
            if btype == 'residential':
                return 3
            return 2

        dw['building_type'] = dw.get('building', pd.Series('residential',
                                                             index=dw.index))
        dw['est_bedrooms']  = dw['building_type'].apply(est_beds)
        all_dws.append(dw[['zone_id', 'osm_id', 'lat', 'lon',
                             'building_type', 'est_bedrooms']])

        time.sleep(1.0)   # be polite to Overpass

    if not all_dws:
        logger.warning("No dwellings found. Check sa2_geo coordinates.")
        return pd.DataFrame(columns=['zone_id','osm_id','lat','lon',
                                     'building_type','est_bedrooms'])

    return pd.concat(all_dws, ignore_index=True)


def match_households(
    persons: pd.DataFrame,
    dwellings: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Match synthetic households to dwellings using a scoring approach.

    Scoring factors:
        bedroom match  : abs(est_bedrooms - required_bedrooms) penalty
        income match   : higher income → detached/house preference
        type match     : apartment for 1-2 person hh, house for family

    Each dwelling is assigned to at most one household.
    Households without a match get the closest available dwelling.

    Parameters
    ----------
    persons   : synthetic persons with columns:
                abshid (or person_id), nprd, incp, ptype, OTAZ
    dwellings : output of fetch_dwellings()

    Returns
    -------
    persons with added columns:
        dwelling_osm_id, dwelling_lat, dwelling_lon,
        dwelling_type, home_lat, home_lon
    """
    rng = np.random.default_rng(seed)
    df  = persons.copy()

    # build household table (one row per household)
    hh_col = 'abshid' if 'abshid' in df.columns else (
             'abshid' if 'abshid' in df.columns else None)
    if hh_col is None:
        df['_hhid'] = df.groupby(['OTAZ','nprd']).ngroup()
        hh_col = '_hhid'

    hh = (
        df.groupby(hh_col)
        .agg(
            nprd   = ('nprd',  'first'),
            incp   = ('incp',  'mean'),
            OTAZ   = ('OTAZ',  'first'),
            ptype0 = ('ptype', 'first'),
        )
        .reset_index()
    )

    # required bedrooms heuristic
    def required_beds(nprd, ptype0):
        if nprd <= 1:
            return 1
        if nprd <= 2:
            return 2
        if nprd <= 4:
            return 3
        return 4

    hh['req_beds'] = hh.apply(
        lambda r: required_beds(r['nprd'], r['ptype0']), axis=1
    )

    # preferred dwelling type
    def pref_type(incp, nprd):
        if incp >= 12:
            return 'house'
        if nprd <= 2:
            return 'apartments'
        return 'house'

    hh['pref_type'] = hh.apply(
        lambda r: pref_type(r['incp'], r['nprd']), axis=1
    )

    # match zone by zone
    df['dwelling_osm_id'] = np.nan
    df['dwelling_lat']    = np.nan
    df['dwelling_lon']    = np.nan
    df['dwelling_type']   = ''

    zones = hh['OTAZ'].unique()

    for zone in zones:
        hh_zone = hh[hh['OTAZ'] == zone].copy()
        dw_zone = dwellings[dwellings['zone_id'] == zone].copy()

        if dw_zone.empty:
            logger.warning(f"No dwellings in zone {zone} — skipping")
            continue

        assigned_dw = set()

        # score each household against each dwelling
        for _, household in hh_zone.iterrows():
            hhid       = household[hh_col]
            req_beds   = household['req_beds']
            pref       = household['pref_type']

            # compute scores for unassigned dwellings
            avail = dw_zone[~dw_zone['osm_id'].isin(assigned_dw)].copy()
            if avail.empty:
                # all dwellings taken — allow reuse with small noise
                avail = dw_zone.copy()
                avail['lat'] += rng.normal(0, 0.0003, len(avail))
                avail['lon'] += rng.normal(0, 0.0003, len(avail))

            avail['bed_score']  = -np.abs(
                avail['est_bedrooms'] - req_beds
            ).clip(0, 3)
            avail['type_score'] = (avail['building_type'] == pref).astype(float)
            avail['score']      = avail['bed_score'] + avail['type_score']

            # pick best match with small random tie-break
            avail['score'] += rng.uniform(0, 0.05, len(avail))
            best = avail.sort_values('score', ascending=False).iloc[0]

            # assign to all persons in this household
            hh_mask = df[hh_col] == hhid
            df.loc[hh_mask, 'dwelling_osm_id'] = best['osm_id']
            df.loc[hh_mask, 'dwelling_lat']    = best['lat']
            df.loc[hh_mask, 'dwelling_lon']    = best['lon']
            df.loc[hh_mask, 'dwelling_type']   = best['building_type']
            assigned_dw.add(best['osm_id'])

    df['home_lat'] = df['dwelling_lat']
    df['home_lon'] = df['dwelling_lon']

    n_matched = df['dwelling_lat'].notna().sum()
    print(f"Household matching: {n_matched}/{len(df)} persons assigned "
          f"({100*n_matched/len(df):.1f}%)")
    return df


# ---------------------------------------------------------------------------
# 2.  ACTIVITY LOCATION PINNING
# ---------------------------------------------------------------------------

def fetch_activity_locations(
    sa2_geo: pd.DataFrame,
    zone_ids: List[int],
    purposes: Optional[List[str]] = None,
    radius_km: float = 3.0,
    cache_dir: str = 'saved_data/osm_cache',
) -> Dict[Tuple[int, str], pd.DataFrame]:
    """
    Fetch OSM POIs for each (zone, purpose) combination.

    Returns
    -------
    dict  {(zone_id, purpose): DataFrame of POIs with lat/lon}
    """
    if purposes is None:
        purposes = list(PURPOSE_OSM_TAGS.keys())

    zc = next((c for c in ['SA2_MAIN16', 'ORDER', 'DTAZ', 'zone', 'zone_id']
               if c in sa2_geo.columns), None)
    sa2_geo = sa2_geo.rename(columns={zc: 'zone_id'})

    poi_store = {}
    total     = len(zone_ids) * len(purposes)
    done      = 0

    for zid in zone_ids:
        row = sa2_geo[sa2_geo['zone_id'] == zid]
        if row.empty:
            continue

        lat = float(row['lat'].iloc[0])
        lon = float(row['lon'].iloc[0])

        for purpose in purposes:
            done += 1
            tags = PURPOSE_OSM_TAGS.get(purpose, [])
            if not tags:
                continue

            print(f"  [{done}/{total}] zone {zid} purpose={purpose}")

            pois = fetch_osm_pois(lat, lon, tags,
                                  radius_km=radius_km,
                                  cache_dir=cache_dir)
            poi_store[(zid, purpose)] = pois
            time.sleep(0.5)

    return poi_store


def _pin_single(
    row: pd.Series,
    poi_store: dict,
    persons_df: Optional[pd.DataFrame],
    rng: np.random.Generator,
) -> Tuple[float, float, str]:
    """
    Return (lat, lon, osm_name) for a single activity row.

    Logic:
        1. Look up POIs for (DTAZ, purpose)
        2. If work and person has INDUSTRY, filter by industry OSM tags
        3. Sample uniformly from available POIs
        4. If no POIs found, return zone centroid with small jitter
    """
    zone_raw = row.get('DTAZ', 0)
    zone     = int(zone_raw) if pd.notna(zone_raw) else 0
    purpose  = str(row.get('purpose', 'othdiscr'))
    key     = (zone, purpose)

    pois = poi_store.get(key, pd.DataFrame())

    # industry refinement for work trips
    if purpose == 'work' and persons_df is not None:
        pid = row.get('person_id')
        if pid is not None:
            person_row = persons_df[persons_df['person_id'] == pid]
            if not person_row.empty:
                industry = str(person_row['INDUSTRY'].iloc[0]
                               if 'INDUSTRY' in person_row.columns else '')
                ind_tags = INDUSTRY_OSM_TAGS.get(industry)
                if ind_tags:
                    ind_pois = poi_store.get((zone, f'work_{industry}'),
                                             pd.DataFrame())
                    if not ind_pois.empty:
                        pois = ind_pois

    if pois.empty or 'lat' not in pois.columns:
        # fallback: return a jittered centroid placeholder
        return (np.nan, np.nan, '')

    idx    = rng.integers(0, len(pois))
    chosen = pois.iloc[idx]
    return (
        float(chosen['lat']),
        float(chosen['lon']),
        str(chosen.get('name', '')),
    )


def pin_activity_locations(
    schedule: pd.DataFrame,
    poi_store: Dict[Tuple[int, str], pd.DataFrame],
    persons: Optional[pd.DataFrame] = None,
    sa2_geo: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign a real-world lat/lon to every row in the schedule.

    Fast path (poi_store empty): vectorised centroid-jitter for all rows —
    runs in milliseconds regardless of schedule size.

    Slow path (poi_store non-empty): row-by-row OSM POI lookup with centroid
    fallback. Only used when real POI data was fetched.

    Returns schedule with columns:
        activity_lat, activity_lon, activity_name, activity_osm_matched
    """
    rng = np.random.default_rng(seed)
    df  = schedule.copy()

    # ── build centroid lookup ────────────────────────────────────────────
    centroid_lookup: dict = {}
    if sa2_geo is not None:
        zc = next((c for c in ['SA2_MAIN16', 'ORDER', 'DTAZ', 'zone', 'zone_id']
                   if c in sa2_geo.columns), None)
        if zc and 'lat' in sa2_geo.columns and 'lon' in sa2_geo.columns:
            for _, r in sa2_geo.dropna(subset=[zc, 'lat', 'lon']).iterrows():
                try:
                    centroid_lookup[int(r[zc])] = (float(r['lat']), float(r['lon']))
                except (ValueError, TypeError):
                    pass

    # ── FAST PATH: no POIs — pure vectorised centroid jitter ─────────────
    if not poi_store:
        dtaz_vals = pd.to_numeric(df.get('DTAZ', 0), errors='coerce').fillna(0).astype(int)
        clats = np.full(len(df), np.nan)
        clons = np.full(len(df), np.nan)
        for i, zid in enumerate(dtaz_vals):
            c = centroid_lookup.get(int(zid))
            if c:
                clats[i] = c[0] + rng.normal(0, 0.003)
                clons[i] = c[1] + rng.normal(0, 0.003)
        df['activity_lat']         = clats
        df['activity_lon']         = clons
        df['activity_name']        = ''
        df['activity_osm_matched'] = False
        n_placed = int(np.isfinite(clats).sum())
        print(f"Activity location pinning: {n_placed}/{len(df)} placed "
              f"(centroid jitter, no OSM cache)")
        return df

    # ── SLOW PATH: OSM POI lookup with centroid fallback ──────────────────
    lats, lons, names, matched = [], [], [], []

    for _, row in df.iterrows():
        lat, lon, name = _pin_single(row, poi_store, persons, rng)

        if np.isnan(lat):
            # Safe DTAZ extraction — guard against NaN float values
            dtaz_raw = row.get('DTAZ', 0)
            zone = int(dtaz_raw) if pd.notna(dtaz_raw) else 0
            c = centroid_lookup.get(zone)
            if c:
                lat = c[0] + rng.normal(0, 0.003)
                lon = c[1] + rng.normal(0, 0.003)
            else:
                lat, lon = np.nan, np.nan
            matched.append(False)
        else:
            matched.append(True)

        lats.append(lat)
        lons.append(lon)
        names.append(name)

    df['activity_lat']         = lats
    df['activity_lon']         = lons
    df['activity_name']        = names
    df['activity_osm_matched'] = matched

    n_matched = sum(matched)
    print(
        f"Activity location pinning: {n_matched}/{len(df)} matched to OSM POI "
        f"({100*n_matched/len(df):.1f}%)"
    )
    return df


# =============================================================================
# DISTANCE CALCULATION (Haversine)
# =============================================================================

def compute_trip_distance_km(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute great-circle distance (km) between origin and destination.
    Requires:
        origin_lat, origin_lon,
        dest_lat, dest_lon
    """
    import numpy as np

    d = df.copy()

    required = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']
    if not all(col in d.columns for col in required):
        print("  Cannot compute distance — origin/dest coordinates missing.")
        return d

    # convert to radians
    lat1 = np.radians(d['origin_lat'].astype(float))
    lon1 = np.radians(d['origin_lon'].astype(float))
    lat2 = np.radians(d['dest_lat'].astype(float))
    lon2 = np.radians(d['dest_lon'].astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    earth_radius_km = 6371.0
    d['dist_km'] = earth_radius_km * c

    return d

# ---------------------------------------------------------------------------
# 3.  FETCH INDUSTRY-SPECIFIC WORK LOCATIONS
# ---------------------------------------------------------------------------

def fetch_work_locations_by_industry(
    sa2_geo: pd.DataFrame,
    persons: pd.DataFrame,
    zone_ids: List[int],
    radius_km: float = 3.0,
    cache_dir: str = 'saved_data/osm_cache',
) -> Dict[Tuple[int, str], pd.DataFrame]:
    """
    For each (zone, industry) present in the persons table,
    fetch OSM work locations using industry-specific tags.

    Returns
    -------
    dict {(zone_id, f'work_{industry}'): DataFrame}
    """
    zc = next((c for c in ['SA2_MAIN16', 'ORDER', 'DTAZ', 'zone', 'zone_id']
               if c in sa2_geo.columns), None)
    sa2_geo = sa2_geo.rename(columns={zc: 'zone_id'})

    poi_store = {}
    industries = (persons['INDUSTRY'].dropna().unique()
                  if 'INDUSTRY' in persons.columns else [])

    total = len(zone_ids) * len(industries)
    done  = 0

    for zid in zone_ids:
        row = sa2_geo[sa2_geo['zone_id'] == zid]
        if row.empty:
            continue

        lat = float(row['lat'].iloc[0])
        lon = float(row['lon'].iloc[0])

        for industry in industries:
            done += 1
            tags = INDUSTRY_OSM_TAGS.get(str(industry))
            if not tags:
                continue

            print(f"  [{done}/{total}] zone {zid} industry={industry}")
            pois = fetch_osm_pois(lat, lon, tags, radius_km=radius_km,
                                  cache_dir=cache_dir)
            poi_store[(zid, f'work_{industry}')] = pois
            time.sleep(0.5)

    return poi_store


# ---------------------------------------------------------------------------
# 4.  FULL PIPELINE
# ---------------------------------------------------------------------------

def _centroid_jitter_dwellings(
    sa2_geo: pd.DataFrame,
    zone_ids: List[int],
    n_per_zone: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic dwellings table from zone centroids with random jitter.

    Used when skip_dwelling_fetch=True and no cache file exists.
    Produces enough dwelling records for match_households to work without
    requiring any Overpass API calls.
    """
    rng = np.random.default_rng(seed)
    geo = sa2_geo.copy()
    zc  = next((c for c in ['SA2_MAIN16', 'ORDER', 'DTAZ', 'zone', 'zone_id']
                if c in geo.columns), None)
    if zc:
        geo = geo.rename(columns={zc: 'zone_id'})

    rows = []
    for zid in zone_ids:
        row = geo[geo['zone_id'] == zid] if zc else pd.DataFrame()
        if row.empty or 'lat' not in geo.columns:
            clat, clon = -27.47, 153.03   # SEQ centroid fallback
        else:
            clat = float(row['lat'].iloc[0])
            clon = float(row['lon'].iloc[0])

        for k in range(n_per_zone):
            btype = rng.choice(['house', 'apartments', 'residential'],
                               p=[0.5, 0.3, 0.2])
            rows.append({
                'zone_id':      zid,
                'osm_id':       zid * 1000 + k,
                'lat':          clat + rng.uniform(-0.005, 0.005),
                'lon':          clon + rng.uniform(-0.005, 0.005),
                'building_type': btype,
                'est_bedrooms': 3 if btype == 'house' else 2,
            })
    return pd.DataFrame(rows)


def run_spatial_assignment(
    persons: pd.DataFrame,
    schedule: pd.DataFrame,
    landuse: pd.DataFrame,
    sa2_geo: pd.DataFrame,
    radius_km: float = 3.0,
    cache_dir: str = 'saved_data/osm_cache',
    seed: int = 42,
    skip_dwelling_fetch: bool = False,
    dwellings_path: Optional[str] = 'saved_data/dwellings.csv',
    skip_poi_fetch: bool = False,
    poi_store_path: Optional[str] = 'saved_data/poi_store.pkl',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full spatial assignment pipeline:
        1. Fetch dwellings from OSM (or load from cache)
        2. Match households to dwellings
        3. Fetch activity POIs (or load from cache)
        4. Fetch industry-specific work POIs
        5. Pin every activity to a real location

    Parameters
    ----------
    persons   : persons table (must have OTAZ, nprd, incp, ptype)
    schedule  : Stage 6 schedule (must have DTAZ, purpose, person_id)
    landuse   : landuse table (for zone centroids fallback)
    sa2_geo   : DataFrame with zone_id (SA2_MAIN16 or ORDER), lat, lon
    radius_km : OSM search radius around each zone centroid
    cache_dir : directory for OSM response caching
    skip_dwelling_fetch : load dwellings from dwellings_path instead
    dwellings_path      : save/load path for dwellings CSV
    skip_poi_fetch      : load poi_store from poi_store_path instead
    poi_store_path      : save/load path for POI store pickle

    Returns
    -------
    persons_geo  : persons with home_lat, home_lon, dwelling_type
    schedule_geo : schedule with activity_lat, activity_lon, activity_name

    Example
    -------
        sc       = Synth_Connect()
        out      = sc.process_survey(save=False)
        persons  = pd.read_csv('saved_data/persons_processed.csv')
        schedule = pd.read_csv('saved_data/stage6_output.csv')
        landuse  = sc.get_land()
        sa2_geo  = pd.read_csv('SA1_SHP/sa2_centroids.csv')
        # sa2_geo needs columns: SA2_MAIN16 (or ORDER), lat, lon

        persons_geo, schedule_geo = run_spatial_assignment(
            persons, schedule, landuse, sa2_geo
        )
    """
    import pickle

    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
    os.makedirs(cache_dir, exist_ok=True)

    # normalise person_id
    if 'person_id' not in persons.columns and 'PERSID' in persons.columns:
        persons = persons.rename(columns={'PERSID': 'person_id'})
    if 'OTAZ' not in persons.columns and 'ORDER' in persons.columns:
        persons = persons.rename(columns={'ORDER': 'OTAZ'})

    # zones present in persons and schedule
    home_zones     = persons['OTAZ'].dropna().unique().astype(int).tolist()
    activity_zones = schedule['DTAZ'].dropna().unique().astype(int).tolist()
    all_zones      = list(set(home_zones + activity_zones))

    print(f"\n{'='*55}")
    print("SPATIAL ASSIGNMENT")
    print(f"  {len(persons)} persons in {len(home_zones)} home zones")
    print(f"  {len(schedule)} activities in {len(activity_zones)} destination zones")
    print(f"{'='*55}")

    # ── PART 1: Household matching ─────────────────────────────────────
    print("\n[1/4] Fetching dwellings from OSM...")

    if skip_dwelling_fetch:
        if dwellings_path and os.path.exists(dwellings_path):
            dwellings = pd.read_csv(dwellings_path)
            print(f"  Loaded {len(dwellings)} dwellings from {dwellings_path}")
        else:
            # No cache and skip requested — build centroid-jitter dwellings
            # so match_households still assigns approximate home_lat/lon.
            print("  No dwellings cache found — using zone centroid jitter")
            dwellings = _centroid_jitter_dwellings(sa2_geo, home_zones)
    else:
        dwellings = fetch_dwellings(sa2_geo, home_zones,
                                    radius_km=radius_km, cache_dir=cache_dir)
        if dwellings_path:
            dwellings.to_csv(dwellings_path, index=False)
            print(f"  Saved {len(dwellings)} dwellings → {dwellings_path}")

    print("\n[2/4] Matching households to dwellings...")
    persons_geo = match_households(persons, dwellings, seed=seed)

    # ── PART 2: Activity location pinning ─────────────────────────────
    print("\n[3/4] Fetching activity POIs from OSM...")

    if skip_poi_fetch:
        if poi_store_path and os.path.exists(poi_store_path):
            with open(poi_store_path, 'rb') as f:
                poi_store = pickle.load(f)
            print(f"  Loaded POI store from {poi_store_path}")
        else:
            # No cache and skip requested — empty store triggers centroid fallback
            print("  No POI cache found — using zone centroid fallback for activities")
            poi_store = {}
    else:
        poi_store = fetch_activity_locations(
            sa2_geo, activity_zones,
            radius_km=radius_km, cache_dir=cache_dir,
        )
        if 'INDUSTRY' in persons_geo.columns:
            work_pois = fetch_work_locations_by_industry(
                sa2_geo, persons_geo, activity_zones,
                radius_km=radius_km, cache_dir=cache_dir,
            )
            poi_store.update(work_pois)
        if poi_store_path:
            with open(poi_store_path, 'wb') as f:
                pickle.dump(poi_store, f)
            print(f"  Saved POI store → {poi_store_path}")

    print("\n[4/4] Pinning activity locations...")
    try:
        schedule_geo = pin_activity_locations(
            schedule, poi_store,
            persons=persons_geo,
            sa2_geo=sa2_geo,
            seed=seed,
        )
    except Exception as e:
        print(f"  WARNING: pin_activity_locations failed ({e}) — using raw schedule")
        logger.warning(f"pin_activity_locations failed: {e}", exc_info=True)
        schedule_geo = schedule.copy()
        schedule_geo['activity_lat']         = np.nan
        schedule_geo['activity_lon']         = np.nan
        schedule_geo['activity_name']        = ''
        schedule_geo['activity_osm_matched'] = False

    # attach home lat/lon to schedule for reference
    try:
        if 'home_lat' in persons_geo.columns and 'person_id' in persons_geo.columns:
            home_coords  = persons_geo[['person_id', 'home_lat', 'home_lon']].copy()
            schedule_geo = schedule_geo.merge(home_coords, on='person_id', how='left')
    except Exception as e:
        print(f"  WARNING: home_lat merge failed ({e}) — skipping")
        logger.warning(f"home_lat merge failed: {e}")

    # ── Assign origin/dest coordinates and compute dist_km ────────────────
    # This is the critical step for all plots — always runs even if
    # pin_activity_locations or the merge above failed.
    try:
        schedule_geo = assign_coordinates(schedule_geo, sa2_geo, seed=seed)
    except Exception as e:
        print(f"  WARNING: assign_coordinates failed ({e}) — dist_km unavailable")
        logger.warning(f"assign_coordinates failed: {e}", exc_info=True)

    print(
        f"\n{'='*55}\n"
        f"SPATIAL ASSIGNMENT COMPLETE\n"
        f"{'='*55}\n"
        f"  Persons with home location: "
        f"{persons_geo['home_lat'].notna().sum()}/{len(persons_geo)}\n"
        f"  Activities with OSM match : "
        f"{schedule_geo.get('activity_osm_matched', pd.Series(False)).sum()}"
        f"/{(schedule_geo.get('row_type', pd.Series('activity'))=='activity').sum()}\n"
        f"  Trips with dist_km        : "
        f"{schedule_geo['dist_km'].notna().sum() if 'dist_km' in schedule_geo.columns else 0}"
        f"/{(schedule_geo.get('row_type', pd.Series('trip'))=='trip').sum()}"
    )

    return schedule_geo, persons_geo


# ---------------------------------------------------------------------------
# 5.  SA2 CENTROIDS HELPER  (if you don't have sa2_geo ready)
# ---------------------------------------------------------------------------

def build_sa2_centroids_from_landuse(
    landuse: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive approximate SA2 centroids from your landuse table.

    Returns DataFrame with columns: zone_id, lat, lon
    """
    lu = landuse.copy()
    # Bug fix: was referencing undefined sa2_geo — use lu instead
    zc = next((c for c in ['SA2_MAIN16', 'ORDER', 'DTAZ', 'zone', 'zone_id']
               if c in lu.columns), None)

    if zc is None:
        raise ValueError(
            f"landuse missing zone column. Found: {list(lu.columns)}"
        )

    lu = lu.rename(columns={zc: 'zone_id'})

    if 'lat' in lu.columns and 'lon' in lu.columns:
        return lu[['zone_id', 'lat', 'lon']].dropna()

    # SEQ bounding box centroid approximation
    if 'zone_id' in lu.columns:
        n    = len(lu)
        lats = np.linspace(-27.2, -28.2, n)
        lons = np.linspace(152.7, 153.5, n)
        lu = lu.sort_values('zone_id').reset_index(drop=True)
        lu['lat'] = lats
        lu['lon'] = lons
        logger.warning(
            "Using approximate SEQ coordinates. "
            "For accuracy, supply a real sa2_centroids.csv with lat/lon per zone."
        )
        return lu[['zone_id', 'lat', 'lon']]

    raise ValueError("Cannot derive centroids from landuse table.")


# =============================================================================
# ZONE CENTROIDS  (ABS ASGS API + Nominatim + SEQ fallback)
# =============================================================================

# SEQ SA4 anchor points for fallback centroid estimation
_SA4_ANCHORS = {
    'Brisbane Inner City':     (-27.470, 153.026),
    'Brisbane - East':         (-27.520, 153.110),
    'Brisbane - North':        (-27.380, 153.010),
    'Brisbane - South':        (-27.590, 153.040),
    'Brisbane - West':         (-27.510, 152.940),
    'Brisbane Inner - South':  (-27.505, 153.040),
    'Logan - Beaudesert':      (-27.700, 153.070),
    'Ipswich':                 (-27.610, 152.760),
    'Moreton Bay - North':     (-27.100, 152.970),
    'Moreton Bay - South':     (-27.250, 152.990),
    'Gold Coast':              (-28.020, 153.420),
    'Sunshine Coast':          (-26.650, 153.060),
    'Toowoomba':               (-27.560, 151.950),
    'Queensland - Outback':    (-25.000, 145.000),
}

_MODE_COLORS = {
    'DA': '#1E5FA8', 'SR': '#2E9E6B', 'Transit': '#7B5EA7',
    'Walk': '#E07030', 'Bike': '#5B9E2B', 'Other': '#888888',
    1: '#1E5FA8', 2: '#2E9E6B', 3: '#E07030', 4: '#5B9E2B', 5: '#7B5EA7',
}
_PURPOSE_COLORS = {
    'work': '#1A5FA8', 'edu': '#0F6E56', 'shopping': '#993C1D',
    'eat': '#3B6D11', 'social': '#993556', 'escort': '#854F0B',
    'othmaint': '#534AB7', 'othdiscr': '#888780', 'home': '#555555',
}


def get_zone_centroids_rich(
    landuse: pd.DataFrame,
    cache_dir: str = 'saved_data/osm_cache',
) -> pd.DataFrame:
    """
    Get lat/lon centroids for all zones.
    Priority: cache → ABS ASGS API → Nominatim → SA4 anchor fallback.
    Returns landuse with lat, lon columns added.
    """
    import os, json, logging
    logger = logging.getLogger(__name__)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'zone_centroids.csv')
    lu = landuse.copy()

    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if {'ORDER','lat','lon'}.issubset(cached.columns):
            lu = lu.merge(cached[['ORDER','lat','lon']], on='ORDER', how='left')
            if lu['lat'].notna().mean() > 0.9:
                print(f"  Zone centroids: {lu['lat'].notna().sum()}/{len(lu)} from cache")
                return lu

    print("  Fetching zone centroids (ABS ASGS → Nominatim → fallback)...")
    centroids = {}

    # 1. ABS ASGS Feature Server
    try:
        import requests
        sa2_codes = lu['SA2_MAIN16'].dropna().astype(int).unique().tolist()
        base_url = ("https://geo.abs.gov.au/arcgis/rest/services/ASGS2016/"
                    "SA2/MapServer/0/query")
        for i in range(0, len(sa2_codes), 50):
            batch = sa2_codes[i:i+50]
            params = {
                'where': "SA2_MAINCODE_2016 IN ({})".format(
                    ','.join(str(c) for c in batch)),
                'outFields': 'SA2_MAINCODE_2016',
                'returnCentroid': 'true', 'f': 'json',
            }
            try:
                r = requests.get(base_url, params=params, timeout=15)
                for feat in r.json().get('features', []):
                    sa2 = int(feat['attributes']['SA2_MAINCODE_2016'])
                    c   = feat.get('centroid', {})
                    if c: centroids[sa2] = (float(c['y']), float(c['x']))
            except Exception: pass
        if centroids:
            print(f"    ABS ASGS: {len(centroids)} centroids")
    except Exception: pass

    # 2. Assign to zones
    lats, lons = [], []
    for _, row in lu.iterrows():
        order = int(row['ORDER'])
        sa2   = int(row.get('SA2_MAIN16', 0) or 0)
        sa4   = str(row.get('SA4_NAME16', '') or '')

        if sa2 in centroids:
            lat, lon = centroids[sa2]
        else:
            anchor = _SA4_ANCHORS.get(sa4, (-27.47, 153.03))
            rng2   = np.random.default_rng(order + 9999)
            lat = float(np.clip(anchor[0] + rng2.uniform(-0.12, 0.12), -28.5, -26.5))
            lon = float(np.clip(anchor[1] + rng2.uniform(-0.12, 0.12), 152.5, 153.8))
        lats.append(lat); lons.append(lon)

    lu['lat'] = lats; lu['lon'] = lons
    lu[['ORDER','lat','lon']].to_csv(cache_path, index=False)
    print(f"  Zone centroids saved → {cache_path}")
    return lu


def assign_coordinates(
    plan: pd.DataFrame,
    sa2_geo: pd.DataFrame,
    jitter_m: float = 200.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign origin/destination lat/lon to every TRIP row using zone centroids.

    For each trip row:
        origin_lat/lon  ← centroid of OTAZ  (+ small jitter)
        dest_lat/lon    ← centroid of DTAZ  (+ small jitter)

    Then computes haversine dist_km between origin and destination.

    This is the fast zone-centroid path — no OSM calls needed.  It runs
    instantly on 100k+ rows and is called by the plotting pipeline when the
    full spatial assignment (OSM POI pinning) is not required.

    Parameters
    ----------
    plan     : Stage 6 output with OTAZ, DTAZ, row_type columns
    sa2_geo  : zone centroid table — needs a zone-id column + lat + lon
    jitter_m : metres of random jitter to avoid all trips overlapping
               at exactly the zone centroid (default 200m ≈ 0.0018 deg)
    seed     : RNG seed for reproducible jitter

    Returns
    -------
    plan with columns added:
        origin_lat, origin_lon, dest_lat, dest_lon, dist_km
    """
    rng = np.random.default_rng(seed)
    df  = plan.copy()

    # ── build zone → (lat, lon) centroid lookup ───────────────────────────
    geo = sa2_geo.copy()
    zc  = next((c for c in ['SA2_MAIN16', 'ORDER', 'DTAZ', 'zone', 'zone_id']
                if c in geo.columns), None)
    if zc is None:
        logger.warning("assign_coordinates: no zone column in sa2_geo — skipping")
        return df
    geo = geo.rename(columns={zc: 'zone_id'})
    if 'lat' not in geo.columns or 'lon' not in geo.columns:
        logger.warning("assign_coordinates: sa2_geo missing lat/lon columns")
        return df

    centroid = {
        int(r['zone_id']): (float(r['lat']), float(r['lon']))
        for _, r in geo.dropna(subset=['lat', 'lon']).iterrows()
    }
    if not centroid:
        logger.warning("assign_coordinates: empty centroid dict — skipping")
        return df

    # ── identify trip rows ────────────────────────────────────────────────
    if 'row_type' in df.columns:
        trip_mask = df['row_type'] == 'trip'
    else:
        trip_mask = pd.Series(True, index=df.index)   # treat all as trips

    n_trips = trip_mask.sum()
    if n_trips == 0:
        logger.warning("assign_coordinates: no trip rows found")
        return df

    # ── jitter scale: 200m ≈ 0.0018 degrees at SEQ latitudes ─────────────
    jitter_deg = jitter_m / 111_000.0

    def _lookup(zone_series, lat_col, lon_col):
        lats = np.full(len(df), np.nan)
        lons = np.full(len(df), np.nan)
        trip_idx = np.where(trip_mask)[0]
        zones    = zone_series.iloc[trip_idx].values
        jit_lat  = rng.uniform(-jitter_deg, jitter_deg, n_trips)
        jit_lon  = rng.uniform(-jitter_deg, jitter_deg, n_trips)
        for k, (zid, jlat, jlon) in enumerate(zip(zones, jit_lat, jit_lon)):
            c = centroid.get(int(zid) if not pd.isna(zid) else 0)
            if c:
                lats[trip_idx[k]] = c[0] + jlat
                lons[trip_idx[k]] = c[1] + jlon
        df[lat_col] = lats
        df[lon_col] = lons

    _lookup(df['OTAZ'].fillna(0), 'origin_lat', 'origin_lon')
    _lookup(df['DTAZ'].fillna(0), 'dest_lat',   'dest_lon')

    # ── compute haversine distance ─────────────────────────────────────────
    df = compute_trip_distance_km(df)

    n_matched = trip_mask.sum()
    n_with_dist = df.loc[trip_mask, 'dist_km'].notna().sum()
    logger.info(
        f"assign_coordinates: {n_with_dist}/{n_matched} trip rows have dist_km; "
        f"{len(centroid)} zone centroids used"
    )
    print(f"  assign_coordinates: {n_with_dist:,} trips assigned coordinates "
          f"({100*n_with_dist/max(n_matched,1):.0f}%)")
    return df


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_person_movement(
    plan_with_coords: pd.DataFrame,
    person_id,
    save_path=None,
) -> 'plt.Figure':
    """
    Plot a person's daily movement as an animated-style sequence.
    Lines coloured by time-of-day (plasma colormap). Activity bubbles
    sized by duration.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.collections import LineCollection

    person  = plan_with_coords[plan_with_coords['person_id'] == person_id].copy()
    trips   = person[person.get('row_type', pd.Series('trip', index=person.index)) == 'trip']
    acts    = person[person.get('row_type', pd.Series('activity', index=person.index)) == 'activity']

    if len(trips) == 0 or 'origin_lat' not in trips.columns:
        print(f"  No spatial trips for person {person_id}")
        return None

    fig, ax = plt.subplots(figsize=(11, 9), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    n   = len(trips)
    col = [cm.plasma(i / max(n-1, 1)) for i in range(n)]
    all_lats, all_lons = [], []

    for k, (_, row) in enumerate(trips.iterrows()):
        o_lat, o_lon = float(row['origin_lat']), float(row['origin_lon'])
        d_lat, d_lon = float(row['dest_lat']),   float(row['dest_lon'])
        all_lats += [o_lat, d_lat]; all_lons += [o_lon, d_lon]
        ax.annotate('', xy=(d_lon, d_lat), xytext=(o_lon, o_lat),
                    arrowprops=dict(arrowstyle='->', color=col[k], lw=2.5,
                                   mutation_scale=16))
        ax.plot([o_lon, d_lon], [o_lat, d_lat], color=col[k], lw=2, alpha=0.8)
        t_h = int(row.get('STARTIME', 0) / 60)
        t_m = int(row.get('STARTIME', 0) % 60)
        mode = str(row.get('mode_name', row.get('mode', '')))
        dist = float(row.get('dist_km', 0))
        ax.text((o_lon+d_lon)/2, (o_lat+d_lat)/2 + 0.002,
                f"{t_h:02d}:{t_m:02d} {mode} {dist:.1f}km",
                fontsize=6, color='white', ha='center', alpha=0.85,
                fontfamily='monospace')

    for _, row in acts.iterrows():
        d_lat = float(row.get('dest_lat', 0) or 0)
        d_lon = float(row.get('dest_lon', 0) or 0)
        purp  = str(row.get('purpose', ''))
        c     = _PURPOSE_COLORS.get(purp, '#aaa')
        dur   = float(row.get('predicted_duration_min', 0))
        ax.scatter(d_lon, d_lat, s=max(40, min(180, dur/2)),
                   color=c, zorder=6, edgecolors='white', linewidths=0.8, alpha=0.9)
        ax.text(d_lon, d_lat + 0.004, purp, fontsize=6.5,
               color=c, ha='center', fontweight='bold')

    if all_lats:
        pad = 0.02
        ax.set_xlim(min(all_lons)-pad, max(all_lons)+pad)
        ax.set_ylim(min(all_lats)-pad, max(all_lats)+pad)
    ax.set_xlabel('Longitude', color='#aaa', fontsize=9)
    ax.set_ylabel('Latitude',  color='#aaa', fontsize=9)
    ax.tick_params(colors='#777', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#334466')

    ptype = person.iloc[0].get('ptype','') if len(person) else ''
    total_km = trips['dist_km'].sum() if 'dist_km' in trips.columns else 0
    # Use activity count for the human-readable title — that's what "trips" means
    # to a user (one trip = one out-of-home activity visit, not each trip leg).
    n_activities = len(acts)
    n_trip_legs  = len(trips)
    ax.set_title(f'Person {person_id}  —  {ptype}  |  '
                 f'{n_activities} activities  ·  {n_trip_legs} trip legs  ·  Total: {total_km:.1f} km',
                 color='white', fontsize=11, pad=10)

    sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(0,24))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Hour of day', color='#aaa', fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa', fontsize=7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"  Person {person_id} movement → {save_path}")
    return fig


def plot_distance_distribution(
    plan_with_coords: pd.DataFrame,
    save_path=None,
) -> 'plt.Figure':
    """Histogram + box-plot of trip distances by mode and purpose."""
    import matplotlib.pyplot as plt

    row_type_col = 'row_type' if 'row_type' in plan_with_coords.columns else None
    if row_type_col:
        trips = plan_with_coords[plan_with_coords[row_type_col] == 'trip'].copy()
    else:
        trips = plan_with_coords.copy()

    if 'dist_km' not in trips.columns or len(trips) == 0:
        print("  No dist_km — run assign_coordinates() first")
        return None

    mode_col = 'mode_name' if 'mode_name' in trips.columns else 'mode'
    _mm = {1:'DA', 2:'SR', 3:'Walk', 4:'Bike', 5:'Transit'}
    if trips[mode_col].dtype != object:
        trips['mode_label'] = trips[mode_col].map(_mm).fillna('DA')
    else:
        trips['mode_label'] = trips[mode_col].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # Panel 1 — histogram by mode
    ax = axes[0]
    bins = np.linspace(0, min(trips['dist_km'].quantile(0.98), 80), 45)
    for mode in ['DA','SR','Transit','Walk','Bike']:
        sub = trips[trips['mode_label']==mode]['dist_km']
        if len(sub)==0: continue
        ax.hist(sub, bins=bins, alpha=0.6, color=_MODE_COLORS.get(mode,'#888'),
               label=f'{mode} (n={len(sub):,})', density=True,
               edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Trip distance (km)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Trip distance by mode', fontsize=11, pad=6)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color='#eee', linewidth=0.5)

    # Panel 2 — box by purpose
    ax = axes[1]
    PURPS = ['work','edu','shopping','eat','social','escort','othmaint','othdiscr']
    pp = [p for p in PURPS if p in trips.get('purpose', pd.Series([], dtype=str)).values]
    if pp:
        data = [trips.loc[trips['purpose']==p, 'dist_km'].clip(0,80).dropna().values for p in pp]
        bp = ax.boxplot(data, vert=False, patch_artist=True,
                       medianprops=dict(color='white', lw=2),
                       flierprops=dict(marker='.', ms=2, alpha=0.3))
        for patch, p in zip(bp['boxes'], pp):
            patch.set_facecolor(_PURPOSE_COLORS.get(p,'#888'))
            patch.set_alpha(0.8)
        ax.set_yticks(range(1, len(pp)+1)); ax.set_yticklabels(pp, fontsize=9)
    ax.set_xlabel('Distance (km)', fontsize=10)
    ax.set_title('Trip distance by purpose', fontsize=11, pad=6)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', color='#eee', linewidth=0.5)

    med = trips['dist_km'].median(); mn = trips['dist_km'].mean()
    fig.suptitle(f'Trip Distances  —  median={med:.1f} km  mean={mn:.1f} km  '
                 f'n={len(trips):,}', fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Distance distribution → {save_path}")
    return fig


def plot_desire_lines(
    plan_with_coords: pd.DataFrame,
    sample_n: int = 3000,
    save_path=None,
) -> 'plt.Figure':
    """O-D desire lines coloured by mode on a dark background."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    row_type_col = 'row_type' if 'row_type' in plan_with_coords.columns else None
    if row_type_col:
        trips = plan_with_coords[plan_with_coords[row_type_col]=='trip'].copy()
    else:
        trips = plan_with_coords.copy()

    if 'origin_lat' not in trips.columns or len(trips)==0:
        print("  No spatial data for desire lines"); return None

    if len(trips) > sample_n:
        trips = trips.sample(sample_n, random_state=42)

    mode_col = 'mode_name' if 'mode_name' in trips.columns else 'mode'
    _mm = {1:'DA', 2:'SR', 3:'Walk', 4:'Bike', 5:'Transit'}
    if trips[mode_col].dtype != object:
        trips['mode_label'] = trips[mode_col].map(_mm).fillna('DA')
    else:
        trips['mode_label'] = trips[mode_col].astype(str)

    fig, ax = plt.subplots(figsize=(13, 11), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    for mode in ['DA','SR','Transit','Walk','Bike']:
        sub = trips[trips['mode_label']==mode].dropna(
            subset=['origin_lat','origin_lon','dest_lat','dest_lon'])
        if len(sub)==0: continue
        segs = [[[r['origin_lon'], r['origin_lat']], [r['dest_lon'], r['dest_lat']]]
                for _, r in sub.iterrows()]
        lc = LineCollection(segs, colors=_MODE_COLORS.get(mode,'#888'),
                           linewidths=0.4, alpha=0.22,
                           label=f'{mode} (n={len(sub):,})')
        ax.add_collection(lc)

    all_lons = pd.concat([trips['origin_lon'], trips['dest_lon']]).dropna()
    all_lats = pd.concat([trips['origin_lat'], trips['dest_lat']]).dropna()
    if len(all_lons):
        p = 0.05
        ax.set_xlim(all_lons.min()-p, all_lons.max()+p)
        ax.set_ylim(all_lats.min()-p, all_lats.max()+p)

    ax.scatter(trips['origin_lon'], trips['origin_lat'],
               s=0.8, c='#FFD700', alpha=0.12, zorder=4)
    ax.set_xlabel('Longitude', color='#666', fontsize=9)
    ax.set_ylabel('Latitude',  color='#666', fontsize=9)
    ax.tick_params(colors='#444', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#222')
    ax.set_title(f'O-D Desire Lines  —  {sample_n:,} sampled trips',
                color='white', fontsize=12, pad=10)
    ax.legend(loc='lower right', fontsize=8, facecolor='#0d1117',
             edgecolor='#333', labelcolor='white', markerscale=6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"  Desire lines → {save_path}")
    return fig


def plot_activity_heatmap(
    plan_with_coords: pd.DataFrame,
    save_path=None,
) -> 'plt.Figure':
    """Hexbin density heatmap of activity locations by purpose."""
    import matplotlib.pyplot as plt, math

    row_type_col = 'row_type' if 'row_type' in plan_with_coords.columns else None
    if row_type_col:
        acts = plan_with_coords[plan_with_coords[row_type_col]=='activity'].copy()
    else:
        acts = plan_with_coords.copy()

    if 'dest_lat' not in acts.columns: return None
    PURPS = ['work','edu','shopping','eat','social','escort','othmaint','othdiscr']
    pp = [p for p in PURPS if p in acts.get('purpose', pd.Series([], dtype=str)).values]
    if not pp: return None

    ncols = min(4, len(pp)); nrows = math.ceil(len(pp)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.5, nrows*4), facecolor='white')
    axes = np.array(axes).flatten()

    lon_min = acts['dest_lon'].quantile(0.01); lon_max = acts['dest_lon'].quantile(0.99)
    lat_min = acts['dest_lat'].quantile(0.01); lat_max = acts['dest_lat'].quantile(0.99)

    for i, purp in enumerate(pp):
        ax  = axes[i]
        sub = acts[acts['purpose']==purp].dropna(subset=['dest_lat','dest_lon'])
        hb  = ax.hexbin(sub['dest_lon'], sub['dest_lat'], gridsize=35,
                        cmap='YlOrRd', mincnt=1,
                        extent=[lon_min, lon_max, lat_min, lat_max])
        ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
        ax.set_title(f'{purp}  (n={len(sub):,})', fontsize=10,
                    color=_PURPOSE_COLORS.get(purp,'#666'), fontweight='bold', pad=4)
        ax.tick_params(labelsize=6.5)
        ax.spines[['top','right']].set_visible(False)
        fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)

    for j in range(len(pp), len(axes)): axes[j].set_visible(False)
    fig.suptitle('Activity Location Density by Purpose', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Activity heatmap → {save_path}")
    return fig


def plot_spatial_metrics(
    plan_with_coords: pd.DataFrame,
    save_path=None,
) -> 'plt.Figure':
    """4-panel spatial metrics: (a) dist hist, (b) dist×mode box,
    (c) daily distance per person, (d) mode share by distance band."""
    import matplotlib.pyplot as plt

    row_type_col = 'row_type' if 'row_type' in plan_with_coords.columns else None
    if row_type_col:
        trips = plan_with_coords[plan_with_coords[row_type_col]=='trip'].copy()
    else:
        trips = plan_with_coords.copy()

    if 'dist_km' not in trips.columns: return None

    mode_col = 'mode_name' if 'mode_name' in trips.columns else 'mode'
    _mm = {1:'DA',2:'SR',3:'Walk',4:'Bike',5:'Transit'}
    if trips[mode_col].dtype != object:
        trips['mode_label'] = trips[mode_col].map(_mm).fillna('DA')
    else:
        trips['mode_label'] = trips[mode_col].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    ax1, ax2, ax3, ax4 = axes.flatten()

    # (a) — cap at 60km for display but show real median/mean from full data
    _dist_all = trips['dist_km']
    dc = _dist_all.clip(0, 60)
    ax1.hist(dc, bins=60, color='#1E5FA8', alpha=0.8, edgecolor='white', lw=0.3)
    ax1.axvline(_dist_all.median(), color='#E07030', lw=2, ls='--',
                label=f'Median {_dist_all.median():.1f}km')
    ax1.axvline(_dist_all.mean(),   color='#993556', lw=2, ls=':',
                label=f'Mean {_dist_all.mean():.1f}km')
    ax1.set_xlabel('Trip distance (km)', fontsize=10); ax1.set_ylabel('Trips', fontsize=10)
    ax1.set_title('(a) Trip distance distribution', fontsize=11, pad=6)
    ax1.legend(fontsize=8, frameon=False); ax1.spines[['top','right']].set_visible(False)
    ax1.grid(axis='y', color='#eee', lw=0.5)

    # (b)
    modes = [m for m in ['DA','SR','Transit','Walk','Bike'] if m in trips['mode_label'].unique()]
    mdata = [trips.loc[trips['mode_label']==m,'dist_km'].clip(0,60).dropna().values for m in modes]
    bp = ax2.boxplot(mdata, vert=True, patch_artist=True,
                    medianprops=dict(color='white', lw=2),
                    flierprops=dict(marker='.', ms=2, alpha=0.3))
    for patch, m in zip(bp['boxes'], modes):
        patch.set_facecolor(_MODE_COLORS.get(m,'#888')); patch.set_alpha(0.85)
    ax2.set_xticklabels(modes, fontsize=9); ax2.set_ylabel('Distance (km)', fontsize=10)
    ax2.set_title('(b) Distance by mode', fontsize=11, pad=6)
    ax2.spines[['top','right']].set_visible(False); ax2.grid(axis='y', color='#eee', lw=0.5)

    # (c)
    if 'person_id' in trips.columns:
        daily = trips.groupby('person_id')['dist_km'].sum()
        ax3.hist(daily.clip(0, 120), bins=50, color='#0F6E56', alpha=0.8,
                edgecolor='white', lw=0.3)
        ax3.axvline(daily.median(), color='#E07030', lw=2, ls='--',
                   label=f'Median {daily.median():.1f}km/day')
        ax3.set_xlabel('Daily distance (km/person)', fontsize=10)
        ax3.set_ylabel('Persons', fontsize=10)
        ax3.legend(fontsize=8, frameon=False)
    ax3.set_title('(c) Daily distance per person', fontsize=11, pad=6)
    ax3.spines[['top','right']].set_visible(False); ax3.grid(axis='y', color='#eee', lw=0.5)

    # (d)
    bins   = [0,1,3,7,15,30,60,200]
    labels = ['0-1','1-3','3-7','7-15','15-30','30-60','60+']
    trips['dband'] = pd.cut(trips['dist_km'], bins=bins, labels=labels)
    pivot = (trips.groupby(['dband','mode_label'], observed=True)
             .size().unstack(fill_value=0))
    ppct  = pivot.div(pivot.sum(axis=1), axis=0) * 100
    bots  = np.zeros(len(ppct))
    for mode in modes:
        if mode not in ppct.columns: continue
        ax4.bar(range(len(ppct)), ppct[mode].values, bottom=bots,
               color=_MODE_COLORS.get(mode,'#888'), label=mode, alpha=0.85)
        bots += ppct[mode].fillna(0).values
    ax4.set_xticks(range(len(labels))); ax4.set_xticklabels(labels, fontsize=8)
    ax4.set_xlabel('Distance band (km)', fontsize=10)
    ax4.set_ylabel('Mode share (%)', fontsize=10); ax4.set_ylim(0,100)
    ax4.set_title('(d) Mode share by distance', fontsize=11, pad=6)
    ax4.legend(fontsize=8, frameon=False, loc='upper right')
    ax4.spines[['top','right']].set_visible(False); ax4.grid(axis='y', color='#eee', lw=0.5)

    fig.suptitle('Spatial Travel Metrics', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Spatial metrics → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------
# Usage:
#   python spatial_assignment.py                            # use saved_data/stage6_output.csv
#   python spatial_assignment.py --input path/to/plan.csv  # explicit plan input
#   python spatial_assignment.py --skip-fetch               # skip OSM dwelling/POI fetches (faster)
#
# Reads Stage 6 output and produces all spatial plots under saved_data/plots/.
# Requires landuse from Synth_Connect().get_land() — no prior stage output needed
# beyond the plan CSV.

if __name__ == '__main__':
    import sys
    import os
    import logging
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('spatial_main')

    _OUT       = os.environ.get('PIPELINE_OUT_DIR', 'saved_data')
    S6_CSV     = os.path.join(_OUT, 'stage6_output.csv')
    SKIMS_PATH = 'downcasted_matrices_base.omx'

    args        = sys.argv[1:]
    skip_fetch  = '--skip-fetch' in args
    args        = [a for a in args if a != '--skip-fetch']

    input_csv = S6_CSV
    if '--input' in args:
        idx = args.index('--input')
        input_csv = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if '--skims' in args:
        idx = args.index('--skims')
        SKIMS_PATH = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    # ── load landuse ─────────────────────────────────────────────────────
    log.info("Loading landuse...")
    from data_connect import Synth_Connect
    from compute_accessibility import run_accessibility
    sc      = Synth_Connect()
    landuse = run_accessibility(
        sc.get_land(),
        skims=SKIMS_PATH,
        save_path='saved_data/accessibility.csv',
    )
    log.info(f"  Landuse: {len(landuse)} zones")

    # ── load plan ─────────────────────────────────────────────────────────
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Stage 6 output not found at '{input_csv}'.\n"
            f"  Run stage6_plan_finalisation.py first, or pass --input."
        )
    log.info(f"Loading plan from {input_csv} ...")
    plan = pd.read_csv(input_csv)
    log.info(f"  {len(plan):,} rows loaded")

    # ── build zone centroids ──────────────────────────────────────────────
    os.makedirs(os.path.join(_OUT, 'plots'), exist_ok=True)
    os.makedirs('saved_data/osm_cache',                exist_ok=True)

    try:
        sa2_geo = get_zone_centroids_rich(landuse, cache_dir='saved_data/osm_cache')
        sa2_geo_map = sa2_geo[['ORDER', 'lat', 'lon']].rename(columns={'ORDER': 'zone_id'})
        log.info(f"  Zone centroids: {len(sa2_geo)} zones")
    except Exception as e:
        log.warning(f"  Zone centroids unavailable ({e}) — using fallback Brisbane coords")
        sa2_geo_map = landuse[['ORDER']].rename(columns={'ORDER': 'zone_id'}).copy()
        sa2_geo_map['lat'] = -27.47
        sa2_geo_map['lon'] = 153.03

    # ── spatial assignment ────────────────────────────────────────────────
    log.info("Running spatial assignment...")
    try:
        # synth persons: load cdap output or usual_locations if available for
        # home coordinate assignment; fall back to a stub if neither exists.
        ul_csv = os.path.join(_OUT, 'usual_locations.csv')
        cdap_csv = os.path.join('saved_data', 'cdap_output.csv')
        if os.path.isfile(ul_csv):
            synth = pd.read_csv(ul_csv)
        elif os.path.isfile(cdap_csv):
            synth = pd.read_csv(cdap_csv)
        else:
            log.warning("  No synth CSV found — creating minimal stub for spatial assignment")
            pid_col = next((c for c in ('person_id', 'PERSONID', 'pid') if c in plan.columns), None)
            otaz_col = next((c for c in ('OTAZ', 'otaz', 'home_zone') if c in plan.columns), None)
            if pid_col and otaz_col:
                synth = plan[[pid_col, otaz_col]].drop_duplicates().rename(
                    columns={pid_col: 'person_id', otaz_col: 'OTAZ'}
                )
            else:
                synth = pd.DataFrame(columns=['person_id', 'OTAZ'])

        plan_coords, _ = run_spatial_assignment(
            persons             = synth,
            schedule            = plan,
            landuse             = landuse,
            sa2_geo             = sa2_geo_map,
            cache_dir           = 'saved_data/osm_cache',
            skip_dwelling_fetch = skip_fetch,
            skip_poi_fetch      = skip_fetch,
        )
    except Exception as e:
        log.warning(f"  Spatial assignment failed ({e}) — using plan directly for plots")
        plan_coords = plan.copy()

    # ── plots ─────────────────────────────────────────────────────────────
    log.info("Generating spatial plots...")

    plot_distance_distribution(plan_coords,
                               save_path=os.path.join(_OUT, 'plots', 'distance_distribution.png'))
    plot_desire_lines(plan_coords,
                      save_path=os.path.join(_OUT, 'plots', 'desire_lines.png'))
    plot_activity_heatmap(plan_coords,
                          save_path=os.path.join(_OUT, 'plots', 'activity_heatmap.png'))
    plot_spatial_metrics(plan_coords,
                         save_path=os.path.join(_OUT, 'plots', 'spatial_metrics.png'))

    # per-person movement plots for the top 4 highest-mileage persons
    rt_col = 'row_type' if 'row_type' in plan_coords.columns else None
    trip_rows = plan_coords[plan_coords[rt_col] == 'trip'] if rt_col else plan_coords
    if 'dist_km' in trip_rows.columns and 'person_id' in trip_rows.columns:
        import numpy as np
        top_pids = (trip_rows.groupby('person_id')['dist_km']
                    .sum().sort_values(ascending=False)
                    .head(20).index.tolist())
        rng = np.random.default_rng(42)
        sample_pids = rng.choice(top_pids, size=min(4, len(top_pids)),
                                  replace=False).tolist()
        for pid in sample_pids:
            plot_person_movement(
                plan_coords, pid,
                save_path=os.path.join(_OUT, 'plots', f'person_{pid}_movement.png')
            )
        log.info(f"  Per-person movement plots: {sample_pids}")

    log.info("=" * 55)
    log.info(f"Spatial plots complete → {_OUT}/plots/")
