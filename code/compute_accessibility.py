"""
compute_accessibility.py
=========================
Zone-level accessibility using the actual QTS/SEQ landuse columns.

Landuse columns available (from zone_order_and_landuse.csv):
  Employment:   RETEMPN, FPSEMPN, HEREMPN, AGREMPN, MWTEMPN, OTHEMP
  Education:    Primary, Secondary, Education (tertiary), HSENROLL, AGE0519
  Activities:   EATOUT, Shopping, Entertainment, ESCORT,
                OTHERMAINTENANCE, OTHERDISCRFacilities, Healthcare
  Land area:    residential, commercial, retail, industrial,
                recreation_ground, construction, sa2_area
  Population:   TOTPOP, TOTHH, POPULATION

Key design choices (informed by data analysis):
  - Employment cols are highly correlated (r≥0.88 with TOTEMP) so we use
    TOTEMP for generic work accessibility and add industry-specific bonuses
  - Activity counts (EATOUT, Shopping etc.) are facility counts, low-coverage
    (55-95%) but meaningful for non-mandatory destination choice
  - Land area columns (residential m², retail m²) are continuous measures
    of land use intensity — better for density-based measures than counts
  - Population density derived from residential/sa2_area correlates with
    non-mandatory activity generation
"""

import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ACCESSIBILITY SPECIFICATION
# Uses actual landuse columns from zone_order_and_landuse.csv
# =============================================================================

ACCESSIBILITY_SPEC = {

    # ── Auto work accessibility ───────────────────────────────────────────────
    # Attractor: total employment (TOTEMP = sum of all emp categories)
    # High correlation (r≥0.88) means TOTEMP is a valid aggregate
    'auto_work_acc': {
        'attractor_cols': {
            'TOTEMP':  1.0,   # total employment — computed as sum of below
            'FPSEMPN': 0.15,  # finance/professional bonus (white-collar)
            'HEREMPN': 0.10,  # healthcare bonus
            'n_office': 1.0,      # Overture office POIs (active w/ USE_OVERTURE_LANDUSE=1)
            'n_industrial': 0.2,  # Overture industrial POIs
            'com_floor_m2': 0.000001,  # Overture commercial floor area (job capacity)
            'jobs_office_n': 0.5,  # classified office job locations (count)
            'jobs_office_m2': 0.0000005,  # classified office floorspace
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.05,
        'cap_tt':      120.0,
    },

    # ── Auto education accessibility ──────────────────────────────────────────
    # Uses actual enrolment counts from landuse (Primary, Secondary, Education)
    # Education = tertiary facilities count; Primary/Secondary = enrolment
    'auto_edu_acc': {
        'attractor_cols': {
            'Primary':   1.0,   # primary school enrolment
            'Secondary': 1.0,   # secondary school enrolment
            'Education': 50.0,  # tertiary facilities (scale up — count vs enrolment)
            'HSENROLL':  0.5,   # high school enrolment (supplement)
            'n_education': 1.0,  # Overture education POIs (facility presence)
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.06,
        'cap_tt':      90.0,
    },

    # ── Auto shopping accessibility ───────────────────────────────────────────
    # Uses Shopping (facility count) and retail land area (m²)
    # retail area is continuous and highly correlated with shopping activity
    'auto_shop_acc': {
        'attractor_cols': {
            'Shopping': 10.0,   # facility count (scale up to match retail area)
            'retail':   0.001,  # retail floor area m² (normalise to comparable scale)
            'RETEMPN':  0.5,    # retail employment
            'n_retail': 3.0,      # Overture retail POIs (current, comprehensive)
            'area_retail': 0.0002,  # Overture retail floor area m²
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.08,
        'cap_tt':      60.0,
    },

    # ── Auto non-mandatory accessibility ─────────────────────────────────────
    # Composite of eat/entertainment/maintenance/discretionary facilities
    'auto_nonmand_acc': {
        'attractor_cols': {
            'EATOUT':               5.0,   # restaurant/cafe count
            'Entertainment':        3.0,   # entertainment venues
            'Shopping':             3.0,   # shopping facilities
            'OTHERMAINTENANCE':     2.0,   # maintenance services
            'OTHERDISCRFacilities': 2.0,  # discretionary facilities
            'Healthcare':           2.0,   # healthcare facilities
            'retail':              0.0005, # retail area contribution
            'n_food':               2.0,   # Overture food/dining POIs
            'n_recreation':         0.05,  # Overture leisure POIs (large counts)
            'n_health':             1.0,   # Overture health POIs
            'n_addresses':          0.002, # Overture address density (trip ends)
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.07,
        'cap_tt':      60.0,
    },

    # ── Walk accessibility ─────────────────────────────────────────────────────
    # Walk focuses on local density: residential area + retail + EATOUT
    # residential m² is a strong indicator of walkable neighbourhood density
    'walk_acc': {
        'attractor_cols': {
            'EATOUT':      3.0,
            'Shopping':    3.0,
            'retail':     0.0003,
            'residential': 0.000005,  # residential floor area m²
            'TOTPOP':      0.01,
            'n_food':       1.0,    # Overture dining POIs (walkable amenities)
            'n_retail':     1.0,    # Overture retail POIs
            'n_residential': 0.005, # Overture residential POI density
            'res_floor_m2': 0.00001,  # Overture residential floor area (density)
            'road_walk_km': 0.5,      # Overture footway/path/cycleway network
        },
        'skim_col':    'DISTWALK',
        'decay_type':  'logit',
        'decay_coeff': 1.5,
        'cap_tt':      7.5,
    },

    # ── Bike accessibility ────────────────────────────────────────────────────
    'bike_acc': {
        'attractor_cols': {
            'TOTEMP':  0.5,
            'retail':  0.0002,
            'EATOUT':  2.0,
            'n_retail': 0.5,  # Overture retail POIs
            'n_food':   0.5,  # Overture dining POIs
            'road_cycle_km': 1.0,  # Overture cycleway network
        },
        'skim_col':    'DISTBIKE',
        'decay_type':  'logit',
        'decay_coeff': 0.4,
        'cap_tt':      25.0,
    },

    # ── Transit accessibility ─────────────────────────────────────────────────
    # Employment reachable BY TRANSIT, time-decayed. Two fixes vs the old spec:
    #  (1) skim_col is the real transit skim (WLK_LOC_WLK_TOTIVT__AM, walk-local-
    #      walk transit in-vehicle minutes) — the old spec used SOVTOLL_TIME (an
    #      AUTO skim), so "transit accessibility" was actually auto accessibility.
    #  (2) steeper decay (0.06 vs 0.025) + lower cap (90 vs 150 min) so the measure
    #      DIFFERENTIATES zones instead of saturating to a near-constant (~13
    #      everywhere, CV≈3.5%). Transit access should be high near the CBD/rail and
    #      low on the fringe; the flat measure is why c_transit_access injected no
    #      cross-zone variation into the mode logsum → c_logsum could not identify.
    #  zero_unreachable: transit OD pairs with no path come through the skim as 0.
    #      Unlike auto (which reaches everywhere), a 0 here means NO SERVICE and
    #      must count as WORST access (tt→cap), not instant/best access.
    'transit_acc': {
        'attractor_cols': {
            'TOTEMP':      1.0,
            'commercial': 0.000005,  # commercial area as CBD proxy
            'n_office':    0.5,      # Overture office POIs (CBD/employment proxy)
            'n_bus_stops': 0.5,      # Overture transit stops (service density)
        },
        'skim_col':        'WLK_LOC_WLK_TOTIVT__AM',
        'decay_type':      'exp',
        'decay_coeff':     0.06,
        'cap_tt':          90.0,
        'zero_unreachable': True,
    },

    # ── Auto power-consumption accessibility ──────────────────────────────────
    # Attractor: diversified electrical load per SA2 (zone_power_371.csv).
    # Rationale: metered-equivalent consumption is a revealed activity-
    # intensity proxy — high-consumption zones concentrate the work, shopping
    # and social opportunities trips seek, complementing employment counts
    # (which miss informal/unmetered activity). Same gravity form as the
    # work measure; com_kw over-weights commercial (job-activity) load.
    'auto_power_acc': {
        'attractor_cols': {
            'energy_kwh_day': 1.0,
            'peak_kw':        10.0,
            'com_kw':         5.0,
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.05,
        'cap_tt':      120.0,
    },

    # ── Transit power-consumption accessibility ───────────────────────────────
    # Same consumption attractor through the transit skim (zero = no service).
    'transit_power_acc': {
        'attractor_cols': {
            'energy_kwh_day': 1.0,
            'com_kw':         5.0,
        },
        'skim_col':        'WLK_LOC_WLK_TOTIVT__AM',
        'decay_type':      'exp',
        'decay_coeff':     0.06,
        'cap_tt':          90.0,
        'zero_unreachable': True,
    },

    # ── Recreation accessibility ──────────────────────────────────────────────
    # Parks, recreation grounds, sport — relevant for social/othdiscr trips
    'recreation_acc': {
        'attractor_cols': {
            'recreation_ground': 0.00001,
            'Entertainment':     3.0,
            'park':              0.001,
            'sport':             0.01,
            'n_recreation':      0.05,        # Overture leisure POIs
            'area_recreation':   0.00000002, # Overture recreation area m² (large scale)
        },
        'skim_col':    'SOVTOLL_TIME__AM',
        'decay_type':  'exp',
        'decay_coeff': 0.06,
        'cap_tt':      45.0,
    },
}


# =============================================================================
# DERIVED LANDUSE COLUMNS
# Computed from raw columns before accessibility calculation
# =============================================================================

def enrich_landuse(landuse: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to landuse that improve accessibility measures.

    Derived:
        TOTEMP          total employment (sum of 6 categories)
        pop_density     population per km² (TOTPOP / sa2_area * 1e6)
        emp_density     employment per km² (TOTEMP / sa2_area * 1e6)
        res_density     residential floor area fraction (residential / sa2_area)
        school_enrol    combined primary + secondary enrolment
        activity_count  sum of all facility counts
    """
    lu = landuse.copy()

    def _col(name, default=0.0) -> pd.Series:
        """Numeric column or zero-filled placeholder — derived measures must
        not hard-fail on partial landuse tables."""
        if name in lu.columns:
            return pd.to_numeric(lu[name], errors='coerce').fillna(0.0)
        return pd.Series(default, index=lu.index, dtype=float)

    # total employment — not in raw file, must be computed
    emp_cols = ['RETEMPN', 'FPSEMPN', 'HEREMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMP']
    lu['TOTEMP'] = lu[[c for c in emp_cols if c in lu.columns]].sum(axis=1) \
        if any(c in lu.columns for c in emp_cols) else \
        _col('TOTEMP')

    # density measures (per km² where sa2_area is in m²)
    area = _col('sa2_area', 1e7).clip(lower=1e4)  # minimum 1 ha to avoid division noise
    lu['TOTPOP'] = _col('TOTPOP')
    lu['pop_density'] = lu['TOTPOP']       / area * 1e6
    lu['emp_density'] = lu['TOTEMP']       / area * 1e6
    lu['res_density'] = _col('residential')  / area  # fraction 0-1

    # combined school enrolment
    lu['school_enrol'] = (
        lu.get('Primary',   pd.Series(0, index=lu.index)).fillna(0) +
        lu.get('Secondary', pd.Series(0, index=lu.index)).fillna(0)
    )

    # total activity facility count
    act_cols = ['EATOUT','Shopping','Entertainment','ESCORT',
                'OTHERMAINTENANCE','OTHERDISCRFacilities','Healthcare']
    lu['activity_count'] = lu[[c for c in act_cols if c in lu.columns]].fillna(0).sum(axis=1) \
        if any(c in lu.columns for c in act_cols) else 0.0

    return lu


# Overture POI/area columns available in zone_landuse_overture.csv
# (build_zone_landuse_overture.py). These are joined in by
# _merge_overture_attractors() when USE_OVERTURE_LANDUSE=1 and become live
# attractor terms in ACCESSIBILITY_SPEC / _LOGSUM_PURPOSE_SPEC below.
# Missing columns are skipped silently by the spec loops, so the specs are
# safe to extend unconditionally — the terms only activate when present.
_OVERTURE_ATTRACTOR_COLS = [
    'n_civic', 'n_education', 'n_food', 'n_health', 'n_industrial',
    'n_office', 'n_recreation', 'n_residential', 'n_retail',
    'area_civic', 'area_education', 'area_food', 'area_health',
    'area_industrial', 'area_office', 'area_recreation', 'area_residential',
    'area_retail', 'n_is_work', 'n_is_school', 'n_is_discretionary',
    'n_is_home', 'n_locations',
    'road_km_motorway', 'road_km_primary', 'road_km_secondary',
    'road_km_tertiary', 'road_km_local', 'road_km_car_total',
    'road_speed_avg', 'road_walk_km', 'road_cycle_km',
    'lu_area_residential', 'lu_area_retail', 'lu_area_commercial',
    'lu_area_industrial', 'lu_area_school', 'lu_area_park', 'lu_area_civic',
    'n_bus_stops', 'n_addresses',
    'jobs_education_n', 'jobs_education_m2', 'jobs_health_n', 'jobs_health_m2',
    'jobs_hospitality_n', 'jobs_hospitality_m2', 'jobs_industrial_n',
    'jobs_industrial_m2', 'jobs_office_n', 'jobs_office_m2',
    'jobs_retail_n', 'jobs_retail_m2', 'jobs_services_n', 'jobs_services_m2',
    # Overture TRANSPORTATION capacity (build_zone_landuse_overture transport
    # step — geometry only, never volumes). Available for validation / DTA
    # feedback; NOT in ACCESSIBILITY_SPEC weights by design.
    'road_km', 'freeway_km', 'arterial_km', 'freeway_share',
    # Overture BUILDINGS detailed use (buildings step, OVERTURE_BUILDINGS=1 —
    # real floorspace by use). res/com_floor_m2 ARE in ACCESSIBILITY_SPEC
    # weights (walk/work); the finer bldg_* splits stay available for
    # validation / GA selection.
    'res_floor_m2', 'com_floor_m2', 'bldg_apt_m2', 'bldg_house_m2',
    'bldg_retail_m2', 'bldg_office_m2', 'bldg_industrial_m2',
    'bldg_public_m2', 'bldg_other_m2', 'bldg_apt_share',
    # OSM supply + transit service (build_zone_supply.py — observed supply and
    # scheduled service, NOT volumes). Available for validation / GA selection;
    # NOT in ACCESSIBILITY_SPEC weights by design.
    'n_parking', 'area_parking', 'n_parking_osm', 'parking_capacity', 'parking_area_m2',
    'n_bike_parking', 'n_ev_charging', 'n_fuel', 'n_car_share', 'n_taxi',
    'cycleway_km', 'n_transit_stops', 'transit_trips_am', 'transit_headway',
    # Derived mix (build_zone_landuse_overture.py).
    'lu_entropy', 'jobs_housing_ratio',
]


def _merge_overture_attractors(landuse: pd.DataFrame) -> pd.DataFrame:
    """Left-join Overture POI count/area columns into the landuse frame.

    Active only when USE_OVERTURE_LANDUSE=1 (same flag the chain already
    forwards to stage/pred jobs). Joins on SA2_MAIN16 (the 371-zone key),
    falling back to ORDER. Only ADDS columns missing from ``landuse`` — it
    never overwrites existing values. Fully non-fatal: any failure logs and
    returns the input unchanged.
    """
    if os.environ.get('USE_OVERTURE_LANDUSE', '0') != '1':
        return landuse
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _ov_path = os.environ.get(
            'OVERTURE_LANDUSE_PATH',
            os.path.join(_here, 'saved_data', 'zone_landuse_overture.csv'))
        if not os.path.exists(_ov_path):
            logger.warning(f"USE_OVERTURE_LANDUSE=1 but {_ov_path} missing — "
                           "accessibility falls back to legacy attractors")
            return landuse
        ov = pd.read_csv(_ov_path)
        key = None
        for cand in ('SA2_MAIN16', 'ORDER'):
            if cand in landuse.columns and cand in ov.columns:
                key = cand
                break
        if key is None:
            logger.warning("USE_OVERTURE_LANDUSE=1 but no join key "
                           "(SA2_MAIN16/ORDER) shared with the Overture table")
            return landuse
        add = [c for c in _OVERTURE_ATTRACTOR_COLS
               if c in ov.columns and c not in landuse.columns]
        if not add:
            return landuse
        lu = landuse.merge(ov[[key] + add], on=key, how='left')
        for c in add:
            lu[c] = pd.to_numeric(lu[c], errors='coerce').fillna(0.0)
        nz = int((lu[add].fillna(0) > 0).any(axis=1).sum())
        logger.info(f"[Accessibility] merged {len(add)} Overture attractor "
                    f"columns on {key} ({nz}/{len(lu)} zones have POIs)")
        return lu
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"Overture attractor merge failed (non-fatal): {e}")
        return landuse


# Power-network consumption columns (build_power_network_371.py ->
# saved_data/power/zone_power_371.csv). Joined into the landuse frame so the
# *_power_acc gravity measures see per-SA2 diversified load. Missing columns
# are skipped silently by the spec loops, mirroring the Overture pattern.
_POWER_CONSUMPTION_COLS = [
    'peak_kw', 'peak_kva', 'energy_kwh_day', 'pv_kw',
    'res_kw', 'com_kw', 'ind_kw', 'ev_kw',
]


def _merge_power_consumption(landuse: pd.DataFrame) -> pd.DataFrame:
    """Left-join per-zone power consumption into the landuse frame.

    Keys on SA2_MAIN16; falls back to ORDER when ORDER carries SA2 codes (the
    371 convention set by run_accessibility_371.py). Only ADDS columns missing
    from ``landuse`` — never overwrites. Fully non-fatal: missing file, missing
    keys, or zero coverage all log and return the input unchanged.
    """
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _pp = os.environ.get(
            'POWER_371_PATH',
            os.path.join(_here, 'saved_data', 'power', 'zone_power_371.csv'))
        if not os.path.exists(_pp):
            return landuse
        pw = pd.read_csv(_pp)
        if 'SA2_MAIN16' not in pw.columns:
            logger.warning("power merge skipped: zone_power file lacks SA2_MAIN16")
            return landuse
        pw = pw.copy()
        pw['SA2_MAIN16'] = pd.to_numeric(pw['SA2_MAIN16'], errors='coerce')
        pw = pw.dropna(subset=['SA2_MAIN16']).drop_duplicates('SA2_MAIN16')
        key = None
        if 'SA2_MAIN16' in landuse.columns:
            key = 'SA2_MAIN16'
        elif 'ORDER' in landuse.columns:
            _o = pd.to_numeric(landuse['ORDER'], errors='coerce')
            if float((_o > 100000).mean()) > 0.9:
                key = 'ORDER'
        if key is None:
            logger.warning("power merge skipped: no SA2 key "
                           "(SA2_MAIN16 / ORDER-as-SA2) in landuse")
            return landuse
        add = [c for c in _POWER_CONSUMPTION_COLS
               if c in pw.columns and c not in landuse.columns]
        if not add:
            return landuse
        lu = landuse.copy()
        lu['_pkey'] = pd.to_numeric(lu[key], errors='coerce')
        lu = lu.merge(pw[['SA2_MAIN16'] + add].rename(
            columns={'SA2_MAIN16': '_pkey'}), on='_pkey', how='left')
        lu = lu.drop(columns=['_pkey'])
        for c in add:
            lu[c] = pd.to_numeric(lu[c], errors='coerce').fillna(0.0)
        cov = float((lu[add].fillna(0) > 0).any(axis=1).sum())
        logger.info(f"[Accessibility] merged {len(add)} power-consumption "
                    f"columns on {key} ({cov:.0f}/{len(lu)} zones with load)")
        return lu
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"power-consumption merge failed (non-fatal): {e}")
        return landuse


# =============================================================================
# DECAY FUNCTIONS
# =============================================================================

def _decay(tt: np.ndarray, decay_type: str, b: float) -> np.ndarray:
    tt = np.asarray(tt, dtype=np.float32)
    if decay_type == 'exp':
        return np.exp(-b * tt)
    elif decay_type == 'logit':
        return 1.0 / (1.0 + np.exp(b * tt))
    elif decay_type == 'power':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(tt > 0, tt ** (-b), 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def _is_time_matrix(col: str) -> bool:
    """True if this skim column holds travel TIME (needs minutes conversion).

    Distance (DIST*, *_DIST__*) and count (CRASH, *BOARDS*, *FAR*) matrices
    are already in their native units — routing them through to_minutes()
    only emits spurious "NOT in sidecar" warnings (and risks the legacy
    value-guess dividing km data). Mirrors the codebase's own convention
    ('min' if 'TIME' in col else 'km').
    """
    c = str(col).upper()
    return ('TIME' in c) or ('TOTIVT' in c) or ('WAIT' in c)

def compute_accessibility(
    landuse: pd.DataFrame,
    skim_df: pd.DataFrame,
    spec: dict = None,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Compute zone-level accessibility measures.

    ActivitySim pattern:
      access(o) = log1p( sum_d [ attractor(d) × decay(tt_od) ] )
    """
    if spec is None:
        spec = ACCESSIBILITY_SPEC

    # enrich landuse with derived columns, then merge Overture POI/area
    # attractors when USE_OVERTURE_LANDUSE=1 (no-op otherwise), then merge
    # per-zone power consumption when zone_power_371.csv exists (no-op otherwise)
    lu = enrich_landuse(landuse)
    lu = _merge_overture_attractors(lu)
    lu = _merge_power_consumption(lu)

    # zone index
    lu_col = 'ORDER' if 'ORDER' in lu.columns else lu.index.name
    if lu_col in lu.columns:
        zone_ids = lu[lu_col].astype(int).values
        lu = lu.set_index(lu_col)
    else:
        zone_ids = lu.index.astype(int).values

    # ZONE MIGRATION (371): ORDER 330 was a dummy/external zone in the retired
    # 330-zone system; in the 371-zone SA2 system every ORDER is a real zone, so
    # the old hardcoded `!= 330` drop is gone. Any genuine external stations to
    # exclude can be listed in EXCLUDE_ZONES (comma-separated ORDER ids).
    _excl = os.environ.get('EXCLUDE_ZONES', '').strip()
    if _excl:
        _excl_ids = {int(x) for x in _excl.split(',') if x.strip()}
        zone_ids = zone_ids[~np.isin(zone_ids, list(_excl_ids))]
    zone_ids = np.sort(zone_ids)
    n_zones  = len(zone_ids)
    zone_idx = {int(z): i for i, z in enumerate(zone_ids)}

    print(f"[Accessibility] {n_zones} zones × {len(spec)} measures")

    # filter skims to our zone universe
    sk = skim_df.copy()
    sk['OTAZ'] = sk['OTAZ'].astype(int)
    sk['DTAZ'] = sk['DTAZ'].astype(int)
    sk = sk[sk['OTAZ'].isin(zone_idx) & sk['DTAZ'].isin(zone_idx)]

    # pivot each needed skim column to (n_zones × n_zones) matrix
    needed_cols = {entry['skim_col'] for entry in spec.values()}
    skim_matrices = {}
    for col in needed_cols:
        if col not in sk.columns:
            logger.warning(f"  skim '{col}' missing — using zeros")
            skim_matrices[col] = np.zeros((n_zones, n_zones), dtype=np.float32)
            continue
        piv = (sk.pivot_table(index='OTAZ', columns='DTAZ',
                               values=col, aggfunc='first')
                 .reindex(index=zone_ids, columns=zone_ids)
                 .fillna(0).values.astype(np.float32))

        # units via sidecar (measured) — legacy value-guessing only for
        # unmeasured matrices (it misclassifies genuine 200–600 values).
        # Distance/count matrices skip conversion entirely (already km/counts).
        _converted = False
        if _is_time_matrix(col):
            try:
                from read_skims import to_minutes as _to_minutes
                piv = np.asarray(_to_minutes(col, piv), dtype=np.float32)
                _converted = True
            except Exception:
                pass
        else:
            _converted = True  # native units (km/counts) — nothing to convert
        if not _converted:
            nonzero = piv[piv > 0]
            if len(nonzero):
                median_val = float(np.median(nonzero))
                if median_val > 600:
                    piv = piv / 60.0
                    logger.info(f"  '{col}': converted seconds → minutes (median was {median_val:.0f})")
                elif median_val > 200:
                    piv = piv / 10.0
                    logger.info(f"  '{col}': converted tenths → minutes")
            print(f"  Skim '{col}': "
                  f"median={np.median(nonzero):.1f}  "
                  f"p95={float(np.percentile(nonzero,95)):.1f}  "
                  f"(units: {'min' if 'TIME' in col else 'km'})")

        skim_matrices[col] = piv

    # compute each measure
    results = {}
    for acc_name, entry in spec.items():
        att_cols   = entry['attractor_cols']
        # Skip measures whose attractor columns are ALL absent (e.g. the
        # *_power_acc pair when no power build exists) instead of emitting a
        # degenerate all-zero column into the reports and GA pools.
        if not any(c in lu.columns for c in att_cols):
            logger.warning(f"  '{acc_name}': no attractor columns in landuse — skipped")
            continue
        skim_col   = entry['skim_col']
        decay_type = entry['decay_type']
        b          = entry['decay_coeff']
        cap_tt     = entry.get('cap_tt', 999.0)

        # build attractor vector (n_dest,)
        attractor = np.zeros(n_zones, dtype=np.float32)
        for col, weight in att_cols.items():
            if col in lu.columns:
                vals = (lu.reindex(zone_ids)[col]
                          .fillna(0).clip(lower=0).values
                          .astype(np.float32))
                attractor += weight * vals
            else:
                logger.debug(f"  '{acc_name}': attractor col '{col}' not in landuse")

        attractor = np.maximum(attractor, 0.0)

        tt_mat  = skim_matrices[skim_col].copy()
        # For transit-type skims a 0 means "no path" (unreachable), which must map
        # to WORST access — not the instant/best access that tt=0 would give under
        # exp(-b·0)=1. Push those pairs to the cap before decaying. (Off-diagonal
        # only: keep the intrazonal diagonal at 0 so a zone always accesses itself.)
        if entry.get('zero_unreachable'):
            _off = ~np.eye(tt_mat.shape[0], dtype=bool)
            tt_mat[_off & (tt_mat <= 0)] = cap_tt
        tt_mat  = np.minimum(tt_mat, cap_tt)
        weights = _decay(tt_mat, decay_type, b)

        # log1p of weighted sum — ActivitySim _accumulate_accessibility
        acc_vals = np.log1p((weights * attractor[np.newaxis, :]).sum(axis=1))
        results[acc_name] = acc_vals

        print(f"  {acc_name:20s}: "
              f"min={acc_vals.min():.3f}  "
              f"mean={acc_vals.mean():.3f}  "
              f"max={acc_vals.max():.3f}")

    # ── crash-exposure accessibility (safety as an upstream determinant) ──
    # ATRF reviewers: safety should enter decision layers beyond mode/
    # destination utilities. crash_acc summarises the expected crash
    # frequency a resident of zone i is exposed to across reachable
    # destinations, time-decayed like every other measure:
    #     crash_acc_i = log1p( Σ_j exp(-b·tt_ij) · CRASH_ij )
    # Unlike the spec-driven measures the "attractor" here is the OD-level
    # CRASH skim itself (crash exposure is a property of the OD movement,
    # not of the destination zone), hence the dedicated block. Joining this
    # onto persons via their home zone lets the GA test crash exposure in
    # CDAP, activity frequency, and duration models - not just in the
    # mode/destination utilities where the CRASH term already lives.
    if 'CRASH' in sk.columns:
        try:
            _crash_piv = np.zeros((n_zones, n_zones), dtype=np.float32)
            _oi = sk['OTAZ'].map(zone_idx).values
            _di = sk['DTAZ'].map(zone_idx).values
            _crash_piv[_oi, _di] = pd.to_numeric(
                sk['CRASH'], errors='coerce').fillna(0).clip(lower=0).values
            _tt_col = ('SOVTOLL_TIME__AM' if 'SOVTOLL_TIME__AM' in skim_matrices
                       else next(iter(skim_matrices)))
            _w = _decay(np.minimum(skim_matrices[_tt_col], 90.0), 'exp', 0.05)
            crash_vals = np.log1p((_w * _crash_piv).sum(axis=1))
            results['crash_acc'] = crash_vals
            print(f"  {'crash_acc':20s}: min={crash_vals.min():.3f}  "
                  f"mean={crash_vals.mean():.3f}  max={crash_vals.max():.3f}")
        except Exception as _ce:
            logger.warning(f"crash_acc computation failed (non-fatal): {_ce}")

    acc_df = pd.DataFrame(results, index=zone_ids)
    acc_df.index.name = 'zone_id'

    # ── ActivitySim-style mode-choice LOGSUM accessibility ────────────────
    # The gravity measures above use a single impedance per measure; the
    # ActivitySim accessibility module instead computes, per purpose, the log
    # of the destination-sum of exp(size + mode-choice logsum). See
    # compute_logsum_accessibility() for the formulation. Failures here are
    # non-fatal: the legacy measures remain in place.
    try:
        ls_df = compute_logsum_accessibility(landuse, skim_df, zone_ids=zone_ids)
        for c in ls_df.columns:
            results[c] = ls_df[c].values
            v = ls_df[c].values
            print(f"  {c:20s}: min={v.min():.3f}  mean={v.mean():.3f}  max={v.max():.3f}")
        acc_df = pd.DataFrame(results, index=zone_ids)
        acc_df.index.name = 'zone_id'
    except Exception as _le:
        logger.warning(f"asim logsum accessibility failed (non-fatal, "
                       f"legacy gravity measures kept): {_le}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        acc_df.to_csv(save_path)
        print(f"[Accessibility] saved → {save_path}")

    return acc_df


# =============================================================================
# ACTIVITYSIM-STYLE MODE-CHOICE LOGSUM ACCESSIBILITY
# =============================================================================
# ActivitySim's accessibility module computes, for each origin zone i and
# purpose p, the log of the destination-sum of exp(size term + mode-choice
# logsum):
#
#     acc_p(i) = ln Σ_j exp( size_p(j) + ln Σ_m exp(V_m(i,j)) )
#
#   size_p(j) = exp(γ·log1p(attractor_p(j)))   (ActivitySim size-term
#               convention: a destination with zero attractor keeps base
#               presence e⁰ = 1, so the sum never collapses)
#   V_m(i,j)  = ASC_m + β_m · time_m(i,j)      (systematic mode utility)
#
# This is the impedance ActivitySim feeds its CDAP and accessibility models —
# richer than single-mode gravity because it captures MODE COMPETITION (a
# destination reachable by fast transit OR car contributes one combined
# utility, not two) and MODE AVAILABILITY (walk/bike only over their feasible
# distances; transit only where service exists). All coefficients are
# env-tunable so the measures can be re-calibrated without code changes.

_LOGSUM_PURPOSE_SPEC = {
    'work':     {'TOTEMP': 1.0, 'FPSEMPN': 0.15, 'HEREMPN': 0.10,
                 'n_office': 1.0, 'com_floor_m2': 0.000001,
                 'jobs_office_n': 0.5, 'jobs_office_m2': 0.0000005},
    'edu':      {'Primary': 1.0, 'Secondary': 1.0, 'Education': 50.0,
                 'HSENROLL': 0.5, 'n_education': 1.0},
    'shop':     {'Shopping': 10.0, 'retail': 0.001, 'RETEMPN': 0.5,
                 'n_retail': 3.0},
    'nonmand':  {'EATOUT': 5.0, 'Entertainment': 3.0, 'Shopping': 3.0,
                 'OTHERMAINTENANCE': 2.0, 'OTHERDISCRFacilities': 2.0,
                 'Healthcare': 2.0, 'retail': 0.0005,
                 'n_food': 2.0, 'n_recreation': 0.05, 'n_addresses': 0.002},
    'social':   {'park': 0.001, 'recreation_ground': 0.00001,
                 'Entertainment': 3.0, 'n_recreation': 0.05},
    'eat':      {'EATOUT': 5.0, 'n_food': 2.0},
}


def _asim_beta(name: str, default: float) -> float:
    return float(os.environ.get(f'ASIM_{name}', default))


def compute_logsum_accessibility(landuse: pd.DataFrame,
                                 skim_df: pd.DataFrame,
                                 zone_ids: np.ndarray = None) -> pd.DataFrame:
    """
    ActivitySim-faithful accessibility: per purpose,
        acc_p(i) = ln Σ_j exp( size_p(j) + ln Σ_m exp(V_m(i,j)) )
    with size terms from landuse attractors and mode utilities from the
    real skims (car AM time, transit IVT, walk/bike distances converted to
    minutes at typical speeds). Returns a DataFrame indexed by zone with
    one `asim_<purpose>_acc` logsum column per purpose.
    """
    lu = enrich_landuse(landuse)
    lu = _merge_overture_attractors(lu)
    lu_col = 'ORDER' if 'ORDER' in lu.columns else lu.index.name
    if lu_col in lu.columns:
        lu = lu.set_index(lu_col)
    if zone_ids is None:
        zone_ids = np.sort(lu.index.astype(int).values)
    zone_ids = np.asarray(zone_ids, dtype=int)
    n = len(zone_ids)
    lu = lu.reindex(zone_ids)

    # ── skim matrices (minutes) ───────────────────────────────────────────
    sk = skim_df.copy()
    sk['OTAZ'] = sk['OTAZ'].astype(int)
    sk['DTAZ'] = sk['DTAZ'].astype(int)
    sk = sk[sk['OTAZ'].isin(set(zone_ids)) & sk['DTAZ'].isin(set(zone_ids))]

    try:
        from read_skims import to_minutes as _to_minutes
    except Exception:
        _to_minutes = None

    def _pivot(col):
        if col not in sk.columns:
            return None
        piv = (sk.pivot_table(index='OTAZ', columns='DTAZ', values=col,
                              aggfunc='first')
                 .reindex(index=zone_ids, columns=zone_ids))
        piv = piv.apply(pd.to_numeric, errors='coerce')
        if _to_minutes is not None and _is_time_matrix(col):
            # minutes via unit sidecar (measured); legacy guess only unknown.
            # Distance/count matrices bypass conversion (native km/counts).
            # _to_minutes returns a bare ndarray — re-wrap as DataFrame so
            # downstream _util / walk_km.values / bike_km.values keep
            # working. Without this, ANY sidecar-covered skim (always the
            # car skim in production) crashes _util with
            # "'numpy.ndarray' has no attribute 'values'", the outer
            # try/except in compute_accessibility() swallows it, and NO
            # asim_*_acc column is ever built (silent total loss of the
            # logsum accessibility layer).
            try:
                conv = _to_minutes(col, piv.values)
                return pd.DataFrame(np.asarray(conv, dtype=float),
                                    index=piv.index, columns=piv.columns)
            except Exception:
                pass
        nz = piv.values[piv.values > 0]
        if len(nz):                       # unit normalisation (s → min)
            med = float(np.median(nz))
            if med > 600:
                piv = piv / 60.0
            elif med > 200:
                piv = piv / 10.0
        return piv

    car_t = _pivot('SOVTOLL_TIME__AM')
    if car_t is None:
        car_t = _pivot('SOVTIME__AM')
    trn_t = _pivot('WLK_LOC_WLK_TOTIVT__AM')
    walk_km = _pivot('DISTWALK')
    bike_km = _pivot('DISTBIKE')

    b_car = _asim_beta('BETA_TIME_CAR', -0.035)
    b_trn = _asim_beta('BETA_TIME_TRANSIT', -0.045)
    b_wlk = _asim_beta('BETA_TIME_WALK', -0.060)
    b_bik = _asim_beta('BETA_TIME_BIKE', -0.045)
    asc_car = _asim_beta('ASC_CAR', 0.0)
    asc_trn = _asim_beta('ASC_TRANSIT', -0.8)
    asc_wlk = _asim_beta('ASC_WALK', -0.2)
    asc_bik = _asim_beta('ASC_BIKE', -0.4)
    walk_max_km = _asim_beta('WALK_MAX_KM', 5.0)
    bike_max_km = _asim_beta('BIKE_MAX_KM', 15.0)
    gamma = _asim_beta('SIZE_GAMMA', 0.1)
    neg_big = -700.0                     # exp(-700) underflows to 0 cleanly

    # ── mode utilities (n × n), NaN-safe ──────────────────────────────────
    def _util(mat, beta, asc, unavailable=None):
        if mat is None:
            return None
        v = mat.values.astype(float).copy()
        v[v <= 0] = np.nan               # 0 = unreachable for time skims
        u = asc + beta * v
        if unavailable is not None:
            u[unavailable] = neg_big
        u[~np.isfinite(u)] = neg_big
        return u

    modes = []
    if car_t is not None:
        modes.append(_util(car_t, b_car, asc_car))
    if trn_t is not None:
        off = ~np.eye(n, dtype=bool)
        u = _util(trn_t, b_trn, asc_trn)
        u[off & (u <= neg_big + 1)] = neg_big     # explicit no-service
        modes.append(u)
    if walk_km is not None:
        km = walk_km.values.astype(float)
        km[km <= 0] = np.nan
        mins = km / 4.8 * 60.0                    # 4.8 km/h walk speed
        u = asc_wlk + b_wlk * mins
        u[np.isnan(mins) | (km > walk_max_km)] = neg_big
        modes.append(u)
    if bike_km is not None:
        km = bike_km.values.astype(float)
        km[km <= 0] = np.nan
        mins = km / 14.0 * 60.0                   # 14 km/h bike speed
        u = asc_bik + b_bik * mins
        u[np.isnan(mins) | (km > bike_max_km)] = neg_big
        modes.append(u)
    if not modes:
        raise RuntimeError('no mode skims available for logsum accessibility')
    for _u, _lbl in zip(modes, ('car', 'transit', 'walk', 'bike')[:len(modes)]):
        _fin = _u[np.isfinite(_u) & (_u > neg_big + 1)]
        logger.info(f"[asim] mode {_lbl:8s}: available={int(len(_fin))} "
                    f"util min={_fin.min() if len(_fin) else float('nan'):.2f} "
                    f"max={_fin.max() if len(_fin) else float('nan'):.2f}")

    def _logsumexp(a, axis=0):
        m = np.max(a, axis=axis)
        m = np.where(np.isfinite(m), m, -700.0)
        shifted = a - np.expand_dims(m, axis)
        return m + np.log(np.clip(np.exp(shifted), 0, None).sum(axis=axis))

    mode_stack = np.stack(modes, axis=0)          # (M, n, n)
    mode_logsum = _logsumexp(mode_stack, axis=0)  # ln Σ_m exp(V_m)  (n, n)

    # ── per-purpose accessibility ─────────────────────────────────────────
    out = {}
    for purpose, att in _LOGSUM_PURPOSE_SPEC.items():
        attractor = np.zeros(n, dtype=float)
        for col, w in att.items():
            if col in lu.columns:
                attractor += w * pd.to_numeric(lu[col], errors='coerce') \
                    .fillna(0).clip(lower=0).values
        size = np.exp(gamma * np.log1p(np.maximum(attractor, 0.0)))  # base 1
        # acc_i = ln Σ_j exp(size_j + mode_logsum_ij)
        acc = _logsumexp(size[None, :] + mode_logsum, axis=1)
        out[f'asim_{purpose}_acc'] = acc

    df = pd.DataFrame(out, index=zone_ids)
    df.index.name = 'zone_id'
    return df


def attach_to_landuse(landuse: pd.DataFrame,
                      acc_df: pd.DataFrame) -> pd.DataFrame:
    """Merge accessibility measures into the landuse DataFrame."""
    lu = enrich_landuse(landuse)  # also add TOTEMP etc.
    lu_col = 'ORDER' if 'ORDER' in lu.columns else None
    if lu_col:
        lu = lu.merge(
            acc_df.reset_index().rename(columns={'zone_id': lu_col}),
            on=lu_col, how='left')
    else:
        lu = lu.join(acc_df, how='left')

    for col in acc_df.columns:
        if col in lu.columns:
            med = lu[col].median()
            lu[col] = lu[col].fillna(med if not pd.isna(med) else 0.0)

    print(f"[Accessibility] attached {len(acc_df.columns)} accessibility + "
          f"{5} derived columns to landuse")
    return lu


# Accessibility measure → the landuse attractor that most directly drives it.
# Used by plot_accessibility to show the acc↔landuse linkage explicitly.
_ACC_LANDUSE_PAIRS = [
    ('auto_work_acc',   'TOTEMP',     'total employment'),
    ('auto_edu_acc',    'Education',  'education facilities'),
    ('auto_shop_acc',   'Shopping',   'shopping attractors'),
    ('walk_acc',        'retail',     'retail landuse area'),
    ('bike_acc',        'TOTEMP',     'total employment'),
    ('transit_acc',     'TOTEMP',     'total employment'),
    # NOTE (2026-09): the SEQ landuse extract leaves the `park` / `sport` /
    # `trees` categories essentially unpopulated (327/329 zones are exactly 0;
    # verified in SEQ_SA2_2016_landuse_area.csv AND zone_order_and_landuse.csv,
    # which agree). Open space in this build lives in `recreation_ground`.
    # Pair recreation_acc against the populated column — plotting against
    # `park` produced a degenerate all-zero panel whose auto-annotation then
    # misdiagnosed a data gap as peripheral concentration (job s5check).
    ('recreation_acc',  'recreation_ground', 'recreation-ground area'),
    ('auto_nonmand_acc', 'TOTPOP',    'population'),
    # ActivitySim-style mode-choice logsum measures
    ('asim_work_acc',    'TOTEMP',    'work logsum vs employment'),
    ('asim_edu_acc',     'Education', 'edu logsum vs education'),
    ('asim_shop_acc',    'Shopping',  'shop logsum vs shopping attractors'),
    ('asim_nonmand_acc', 'TOTPOP',    'nonmand logsum vs population'),
]


def plot_accessibility(landuse_enriched: pd.DataFrame,
                       acc_cols: list,
                       plot_dir: str) -> None:
    """
    Two diagnostic figures for the zone-level accessibility measures:

    accessibility_distributions.png — per-measure histogram across zones, so
        a degenerate measure (all zones alike → useless in a choice model)
        is visible at a glance.
    accessibility_vs_landuse.png — each measure scattered against the landuse
        attractor that feeds it (Pearson r annotated). Low r flags a measure
        whose decay/attractor wiring isn't actually transmitting the landuse
        signal.

    Non-fatal: any failure logs and returns (plots must never kill a run).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(plot_dir, exist_ok=True)
        cols = [c for c in acc_cols if c in landuse_enriched.columns]
        if not cols:
            return

        # ── 1. distributions ─────────────────────────────────────────────
        ncols = 4
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).flatten()
        fig.suptitle('Zone accessibility measures — distribution across zones',
                     fontweight='bold')
        for i, c in enumerate(cols):
            v = pd.to_numeric(landuse_enriched[c], errors='coerce').dropna()
            axes[i].hist(v, bins=30, color='#2E86AB', alpha=0.85)
            axes[i].set_title(f'{c}\nmean={v.mean():.2f}  std={v.std():.2f}', fontsize=9)
            axes[i].grid(alpha=0.3)
        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        p1 = os.path.join(plot_dir, 'accessibility_distributions.png')
        fig.savefig(p1, dpi=110)
        plt.close(fig)
        print(f"[Accessibility] plot → {p1}")

        # ── 2. accessibility vs landuse attractors ───────────────────────
        # Area-like attractors are plotted as DENSITY (attractor / zone area):
        # raw areas let large peripheral zones dominate and can flip r negative
        # even when the measure is healthy. Accessibility aggregates OTHER
        # zones' attractors with distance decay, so a negative own-zone r is
        # plausible for peripheral attractors (e.g. parks) — annotated on the
        # panel rather than treated as an error.
        _area_col = next((c for c in ('sa2_area_sqkm', 'sa2_area', 'area_sqkm')
                          if c in landuse_enriched.columns), None)
        from scipy.stats import spearmanr as _spearman
        pairs = [(a, lc, lbl) for a, lc, lbl in _ACC_LANDUSE_PAIRS
                 if a in landuse_enriched.columns and lc in landuse_enriched.columns]
        if pairs:
            nrows2 = int(np.ceil(len(pairs) / ncols))
            fig, axes = plt.subplots(nrows2, ncols, figsize=(4 * ncols, 3.2 * nrows2))
            axes = np.atleast_1d(axes).flatten()
            fig.suptitle('Accessibility vs landuse attractor (per zone)',
                         fontweight='bold')
            for i, (a, lc, lbl) in enumerate(pairs):
                x = pd.to_numeric(landuse_enriched[lc], errors='coerce')
                y = pd.to_numeric(landuse_enriched[a], errors='coerce')
                ok = x.notna() & y.notna()
                if _area_col is not None:
                    area = pd.to_numeric(landuse_enriched[_area_col],
                                         errors='coerce')
                    # Zones with missing/non-positive area cannot form a
                    # density (the old fillna(1.0) silently inflated them —
                    # e.g. 60/371 zones lack sa2_area in zone_landuse_357.csv).
                    # Exclude them from density panels instead.
                    _ok_area = (area.reindex(x.index).notna()
                                & (area.reindex(x.index) > 0))
                    _n_drop = int((ok & ~_ok_area).sum())
                    if _n_drop:
                        print(f"[Accessibility] panel {a} vs {lc}: dropping "
                              f"{_n_drop} zone(s) with missing/non-positive "
                              f"'{_area_col}' (density undefined)")
                    ok = ok & _ok_area
                    x = x / area.reindex(x.index).clip(lower=1e-6)
                    xlabel = f'{lc} density (per km², log1p)'
                else:
                    xlabel = f'log1p({lc})'
                x = np.log1p(x.clip(lower=0))
                r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else float('nan')
                rs = float(_spearman(x[ok], y[ok])[0]) if ok.sum() > 2 else float('nan')
                axes[i].scatter(x[ok], y[ok], s=12, alpha=0.55, color='#1C7293')
                axes[i].set_xlabel(xlabel, fontsize=8)
                axes[i].set_ylabel(a, fontsize=8)
                note = ''
                # Degenerate attractor guard: a column that is ~all zeros
                # (e.g. `park`/`sport` in the SEQ extract — 327/329 zeros) is a
                # LANDUSE DATA gap, not peripheral concentration. Flag it
                # plainly instead of the neg-r story below, whose r is
                # meaningless on a near-constant x.
                _x_raw = pd.to_numeric(landuse_enriched[lc],
                                       errors='coerce')[ok]
                _n_nz = int(((_x_raw.fillna(0)) > 0).sum())
                _n_ok = int(ok.sum())
                if _n_ok > 0 and (_n_nz < 3 or _n_nz / _n_ok < 0.10):
                    note = (f'\nDEGENERATE attractor: {_n_nz}/{_n_ok} zones '
                            f'non-zero — check the landuse build; r is '
                            f'meaningless here')
                elif np.isfinite(r) and r < 0:
                    note = ('\nneg r: attractor concentrated in peripheral, '
                            'low-accessibility zones;\naccessibility aggregates '
                            "NEIGHBOURS' attractors (distance decay)")
                axes[i].set_title(f'{a} vs {lbl}\n'
                                  f'r={r:.2f}  ρ={rs:.2f}{note}', fontsize=8)
                axes[i].grid(alpha=0.3)
            for j in range(len(pairs), len(axes)):
                axes[j].set_visible(False)
            fig.tight_layout()
            p2 = os.path.join(plot_dir, 'accessibility_vs_landuse.png')
            fig.savefig(p2, dpi=110)
            plt.close(fig)
            print(f"[Accessibility] plot → {p2}")
    except Exception as e:
        logger.warning(f"plot_accessibility failed (non-fatal): {e}")


def run_accessibility(
    landuse: pd.DataFrame,
    skims,
    spec: dict = None,
    save_path: str = 'saved_data/accessibility.csv',
    plot_dir: str = None,
) -> pd.DataFrame:
    from stage5_destination_choice import _skim_flat, _open_skims

    print("=" * 60)
    print("ACCESSIBILITY — Zone-level access measures")
    print("=" * 60)

    skims, close_it = _open_skims(skims)
    print("\nReading skims...")
    skim_df = _skim_flat(skims)
    print(f"  {len(skim_df)} OD pairs, {len(skim_df.columns)-2} skim columns")
    if len(skim_df) == 0 or 'OTAZ' not in skim_df.columns:
        # Fail HERE with the real cause instead of a KeyError('OTAZ') three
        # frames deeper. The usual culprit: an LZF-compressed raw OMX
        # (sa2_skims_507.omx) that PyTables cannot decode - "required filter
        # 'lzf' is not registered" in the warning above. The readable,
        # omx-spec-compliant conversion is produced by
        # demand_/11_convert_skims_to_openmatrix.py.
        raise RuntimeError(
            "Skims could not be read (0 OD pairs) - see the HDF5 warning "
            "above. If it mentions the 'lzf' filter, generate the readable "
            "conversion first:  cd ../demand_ && "
            "python 11_convert_skims_to_openmatrix.py  "
            "(writes skims_371.omx, which pipeline_logger "
            "prefers automatically)."
        )

    acc_df = compute_accessibility(landuse, skim_df, spec=spec, save_path=save_path)
    landuse_enriched = attach_to_landuse(landuse, acc_df)

    if plot_dir:
        plot_accessibility(landuse_enriched, list(acc_df.columns), plot_dir)

    if close_it:
        skims.close()

    print("\nAccessibility complete.")
    return landuse_enriched
