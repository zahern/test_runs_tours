"""
stage6_plan_finalisation.py
============================
Stage 6 — Plan Validation, Post-Processing & Aimsun-Ready Output

What changed from the old Stage 6
-----------------------------------
Mode choice is now handled inside Stage 5 (jointly with destination choice
using proper logsums). Running another mode choice model here would
overwrite Stage 5's correctly-estimated modes.

Stage 6 now does three things:

   1. VALIDATION — check the trip chain for impossible records and fix them:
        - Schedule gaps (time between activity ENDTIME and next trip STARTIME)
        - Overnight trips (ENDTIME > 1439 / midnight)
        - Zero or negative travel times
        - Invalid zones (OTAZ or DTAZ = 0, 999 fringe/external, or outside 1-371.
          NOTE: ORDER 330 is a REAL zone in the 371 system, never a catch-all)
       - Mode-distance mismatches (Walk for 50km, Bike for 80km)
       - Duplicate rows per person

  2. LOS METRICS — compute level-of-service attributes per trip:
       - in_vehicle_time, walk_access_time, wait_time (from skims)
       - generalised_cost (mode-specific cost function)
       - time_period label for each trip departure

  3. AIMSUN OUTPUT — produce two clean tables:
       trips_for_aimsun.csv
           One row per trip leg, columns matching Aimsun demand format:
           person_id, trip_id, OTAZ, DTAZ, STARTIME, mode,
           travtime, purpose, time_period
       activities_summary.csv
           One row per activity, columns for plan visualisation and
           tour-level statistics.

Usage
-----
    from stage6_plan_finalisation import run_stage6

    sc      = Synth_Connect()
    out     = sc.process_survey(save=False)
    plan    = pd.read_csv('saved_data/stage5_output.csv')   # Stage 5 output
    landuse = sc.get_land()

    final_plan, trips_aimsun = run_stage6(plan, out["persons"], landuse)
    final_plan.to_csv('saved_data/stage6_output.csv', index=False)
    trips_aimsun.to_csv('saved_data/trips_for_aimsun.csv', index=False)
"""

import numpy as np
import pandas as pd
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

PURPOSES = ['work', 'edu', 'social', 'escort',
            'shopping', 'othmaint', 'othdiscr', 'eat']

MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
MODE_FROM_NAME = {v: k for k, v in MODE_NAMES.items()}

# Modes that need to be profiled as vehicles for DynusT
# DA (1) = drive alone, Other (2) = shared ride, Transit (5) = transit vehicle
MOTORISED_MODES    = {1, 2, 5}
MOTORISED_NAMES    = {'DA', 'SR', 'Transit'}

# DynusT vehicle type codes
# 1 = passenger car (DA), 2 = shared ride / HOV, 3 = transit / bus
DYNUS_VEH_TYPE = {
    1: 1,   # DA       → passenger car
    2: 2,   # SR       → shared ride / HOV
    5: 3,   # Transit  → transit / bus
    'DA':      1,
    'SR':      2,
    'Transit': 3,
}

# DynusT simulation start time (minutes from midnight).
# Departure times in vehicle.dat are in minutes from midnight — this constant
# is used as the default parameter value in run_stage6() and write_dynus_vehicle_dat().
# Default: 6:00 AM (360 min). Adjust to match your DynusT scenario file.
DYNUS_SIM_START_MIN = 360

# DynusT simulation start time (minutes from midnight).
# Departure times in vehicle.dat are seconds from SIM_START_MIN.
# Default: 6:00 AM (360 min) — adjust to match your DynusT scenario.

# maximum realistic distance per mode (km) — used to flag mode mismatches
MODE_MAX_DIST = {
    'Walk':   10.0,
    'Bike':   25.0,
    'Transit': 999.0,
    'DA':      999.0,
    'SR':      999.0,
}

# time periods (minutes from midnight)
TIME_PERIODS = [
    (240,  360, 'early'),
    (360,  540, 'am_peak'),
    (540,  900, 'midday'),
    (900, 1080, 'pm_peak'),
    (1080,1200, 'evening'),
    (1200,1440, 'late'),
]


def _time_period(minutes: float) -> str:
    for lo, hi, label in TIME_PERIODS:
        if lo <= minutes < hi:
            return label
    return 'late'


# ===========================================================================
# 1. VALIDATION
# ===========================================================================

def validate_and_fix(plan: pd.DataFrame,
                     skim_df: Optional[pd.DataFrame] = None,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Check the Stage 5 trip chain for impossible records and fix them.

    Fixes applied
    -------------
    - Negative or zero travtime → replaced with mode-default (10 min)
    - ENDTIME > 1439 → clipped to 1439 (midnight)
    - STARTIME < 240 → warn (pre-4am trips are unusual but not removed)
     - OTAZ or DTAZ == 0 → flag; set DTAZ = OTAZ (intra-zonal fallback)
     - OTAZ or DTAZ == 999 → fringe/external sentinel; warn (kept, excluded
       from valid-zone metrics). Values outside 1-371 (other than 999) → flag.
       ORDER 330 is a real 371-system zone and is NEVER treated as catch-all
    - Schedule gap between activity ENDTIME and next trip STARTIME →
      adjust next trip's STARTIME to close the gap
    - Mode-distance mismatch → downgrade to DA if walk/bike distance
      exceeds realistic maximum

    Returns
    -------
    Validated plan with added columns:
        _fixed (bool)  — True if this row had any fix applied
        _fix_reason    — description of what was fixed
    """
    df = plan.copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]

    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    df['_fixed']      = False
    df['_fix_reason'] = ''

    def _flag(mask, reason):
        df.loc[mask, '_fixed']      = True
        df.loc[mask, '_fix_reason'] = df.loc[mask, '_fix_reason'].where(
            df.loc[mask, '_fix_reason'] == '', df.loc[mask, '_fix_reason'] + '; '
        ) + reason

    # ── 1. travtime ──────────────────────────────────────────────────────
    if 'travtime' in df.columns:
        bad_tt = (df['row_type'] == 'trip') & (df['travtime'] <= 0)
        if bad_tt.any():
            _flag(bad_tt, 'zero_travtime')
            df.loc[bad_tt, 'travtime'] = 10.0
            if verbose:
                print(f"  Fixed {bad_tt.sum()} trips with zero/negative travtime")

    # ── 2. ENDTIME overflow ───────────────────────────────────────────────
    if 'ENDTIME' in df.columns:
        overflow = df['ENDTIME'] > 1439
        if overflow.any():
            _flag(overflow, 'endtime_overflow')
            df.loc[overflow, 'ENDTIME'] = 1439
            if verbose:
                print(f"  Fixed {overflow.sum()} rows with ENDTIME > 1439 (midnight)")

    # ── 3. invalid zones ──────────────────────────────────────────────────
    for col in ['OTAZ', 'DTAZ']:
        if col in df.columns:
            zero_mask = df[col] == 0
            if zero_mask.any():
                _flag(zero_mask, f'{col}=0')
                if verbose:
                    print(f"  WARNING: {zero_mask.sum()} rows have {col}=0")
                # intra-zonal fallback
                other = 'OTAZ' if col == 'DTAZ' else 'DTAZ'
                if other in df.columns:
                    df.loc[zero_mask, col] = df.loc[zero_mask, other]

            # 371-only: 999 = fringe/external sentinel; out-of-range (excl. 999)
            # is invalid. ORDER 330 is a real zone — never flagged.
            zext = (df[col] == 999) | (~df[col].between(1, N_VALID_ZONES, inclusive='both'))
            if zext.any():
                _flag(zext, f'{col}=external_or_out_of_range')
                if verbose:
                    print(f"  WARNING: {zext.sum()} rows have external/out-of-range {col} "
                          f"(999 fringe or outside 1-{N_VALID_ZONES})")

    # ── 4. schedule gap repair (per person) ───────────────────────────────
    # ── Tier-1 time-window scheduler: conflict-free per-person day ───────────
    # A person cannot be in two places at once. Sweep each person's rows in time
    # order with a running "cursor" (earliest free minute); any row that starts
    # before the cursor (an OVERLAP / conflict) is pushed forward to begin at the
    # cursor, PRESERVING its own duration (unlike the old fix, which truncated
    # activities to prev_end+travtime). Consumed time thus makes later slots
    # unavailable — the availability-propagation you asked for. Disable with
    # GA_RESOLVE_CONFLICTS=0.
    if ({'STARTIME', 'ENDTIME'}.issubset(df.columns)
            and os.environ.get('GA_RESOLVE_CONFLICTS', '1').strip() != '0'):
        df = df.sort_values(['person_id', 'STARTIME']).reset_index(drop=True)
        _st = pd.to_numeric(df['STARTIME'], errors='coerce').fillna(0.0).to_numpy(float)
        _en = pd.to_numeric(df['ENDTIME'], errors='coerce').fillna(0.0).to_numpy(float)
        _pid = df['person_id'].to_numpy()
        _conf = []
        _cursor, _cur = -1.0, None
        for _i in range(len(df)):
            if _pid[_i] != _cur:
                _cur, _cursor = _pid[_i], -1.0
            _dur = max(_en[_i] - _st[_i], 0.0)
            if _cursor >= 0 and _st[_i] < _cursor - 1e-6:      # overlap → shift, keep duration
                _st[_i] = _cursor
                _en[_i] = min(_st[_i] + _dur, 1439.0)
                _conf.append(_i)
            else:
                _en[_i] = min(_en[_i], 1439.0)
            _cursor = _en[_i]
        df['STARTIME'], df['ENDTIME'] = _st, _en
        if _conf:
            _flag(pd.Index(_conf), 'time_conflict')
            if verbose:
                print(f"  Tier-1 scheduler: resolved {len(_conf)} time-overlap "
                      f"conflict(s) (duration-preserving)")

    # ── 5. mode-distance mismatch ─────────────────────────────────────────
    if 'dist' in df.columns and 'mode_name' in df.columns:
        fallback_mode = {
            'Walk': ('SR', 2),
            'Bike': ('SR', 2),
        }
        for mode_name, max_dist in MODE_MAX_DIST.items():
            if max_dist >= 999:
                continue
            mask = ((df['mode_name'] == mode_name) &
                    (df['dist'] > max_dist) &
                    (df['row_type'] == 'trip'))
            if mask.any():
                _flag(mask, f'{mode_name}_dist_mismatch')
                _fallback_name, _fallback_code = fallback_mode.get(mode_name, ('DA', 1))
                df.loc[mask, 'mode']      = _fallback_code
                df.loc[mask, 'mode_name'] = _fallback_name
                if verbose:
                    print(f"  Fixed {mask.sum()} {mode_name} trips exceeding "
                          f"{max_dist}km → {_fallback_name}")

    n_fixed = df['_fixed'].sum()
    if verbose:
        print(f"  Validation complete: {n_fixed} rows fixed "
              f"({n_fixed/len(df)*100:.1f}%)")

    return df


# ===========================================================================
# 2. DESTINATION ZONE VALIDATION
# ===========================================================================

# 371-system: zones 1-371 are all real; 999 = fringe/external sentinel.
# ORDER 330 is a real zone and is NEVER a catch-all.
N_VALID_ZONES = 371
FRINGE_ZONE = 999


def compute_zone_validation_metrics(
    plan: pd.DataFrame,
    landuse: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> dict:
    """
    Compute destination-zone quality metrics for trip rows.

    Metrics returned
    ----------------
    pct_valid_dtaz        : % of trip rows with DTAZ in 1-371
    pct_catchall_dtaz     : % of trip rows with DTAZ == 999 (fringe/external)
    pct_intrazonal        : % of trips where OTAZ == DTAZ (stayed in zone)
    n_unique_dtaz         : number of distinct DTAZ values used
    mean_dtaz_per_person  : average number of unique destination zones per person
    entropy               : Shannon entropy of DTAZ distribution (higher = more spread)
    top5_dtaz_share       : % of all trips going to the top 5 most popular zones
    mean_travtime_by_valid: mean travel time for valid-zone trips vs catch-all trips
    catchall_by_mode      : dict of catch-all zone share per mode (to spot mode issues)
    """
    trips = plan[plan['row_type'] == 'trip'].copy() if 'row_type' in plan.columns else plan.copy()
    if trips.empty or 'DTAZ' not in trips.columns:
        if verbose:
            print("  Zone validation: no trip rows or no DTAZ column")
        return {}

    dtaz = pd.to_numeric(trips['DTAZ'], errors='coerce')
    otaz = pd.to_numeric(trips.get('OTAZ', pd.Series(dtype=float)), errors='coerce')
    n    = len(dtaz.dropna())
    if n == 0:
        return {}

    valid_mask    = dtaz.between(1, N_VALID_ZONES, inclusive='both')
    catchall_mask = dtaz == FRINGE_ZONE
    intra_mask    = (otaz == dtaz) & valid_mask

    pct_valid    = valid_mask.sum() / n * 100
    pct_catchall = catchall_mask.sum() / n * 100
    pct_intra    = intra_mask.sum() / n * 100
    n_unique     = int(dtaz[valid_mask].nunique())

    # Shannon entropy of DTAZ distribution (normalised by log(N_VALID_ZONES))
    counts = dtaz[valid_mask].value_counts()
    probs  = counts / counts.sum()
    import math
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(N_VALID_ZONES))

    # Top-5 destination concentration
    top5_share = float(counts.head(5).sum() / counts.sum() * 100) if len(counts) >= 5 else 100.0

    # Mean unique destinations per person
    if 'person_id' in trips.columns:
        mean_dtaz_per_person = float(
            trips[valid_mask].groupby('person_id')['DTAZ'].nunique().mean()
        )
    else:
        mean_dtaz_per_person = float('nan')

    # Travel time comparison valid vs catch-all
    tt_col = next((c for c in ['travtime', 'tt_avg'] if c in trips.columns), None)
    mean_tt_valid    = float(pd.to_numeric(trips.loc[valid_mask,    tt_col], errors='coerce').mean()) if tt_col else float('nan')
    mean_tt_catchall = float(pd.to_numeric(trips.loc[catchall_mask, tt_col], errors='coerce').mean()) if tt_col and catchall_mask.any() else float('nan')

    # Catch-all share per mode
    catchall_by_mode = {}
    if 'mode_name' in trips.columns:
        for mode, grp in trips.groupby('mode_name'):
            n_mode = len(grp)
            n_ca   = int((pd.to_numeric(grp['DTAZ'], errors='coerce') == FRINGE_ZONE).sum())
            catchall_by_mode[str(mode)] = round(n_ca / n_mode * 100, 2) if n_mode else 0.0

    metrics = {
        'pct_valid_dtaz':        round(pct_valid, 2),
        'pct_catchall_dtaz':     round(pct_catchall, 2),
        'pct_intrazonal':        round(pct_intra, 2),
        'n_unique_dtaz':         n_unique,
        'mean_dtaz_per_person':  round(mean_dtaz_per_person, 2),
        'entropy_normalised':    round(entropy, 4),
        'top5_dtaz_share_pct':   round(top5_share, 2),
        'mean_travtime_valid':   round(mean_tt_valid, 2),
        'mean_travtime_catchall':round(mean_tt_catchall, 2) if not math.isnan(mean_tt_catchall) else None,
        'catchall_by_mode':      catchall_by_mode,
        'n_trips_total':         n,
        'n_catchall':            int(catchall_mask.sum()),
    }

    if verbose:
        print("\n  ── Destination Zone Validation ──────────────────────────")
        print(f"     Valid DTAZ (1-{N_VALID_ZONES}): {pct_valid:.1f}%   "
              f"Fringe/external (999): {pct_catchall:.1f}%")
        print(f"     Intrazonal trips:  {pct_intra:.1f}%")
        print(f"     Unique zones used: {n_unique} / {N_VALID_ZONES}   "
              f"Entropy (0=concentrated, 1=uniform): {entropy:.3f}")
        print(f"     Top-5 zone concentration: {top5_share:.1f}% of all trips")
        print(f"     Mean travel time — valid: {mean_tt_valid:.1f} min  "
              f"catch-all: {mean_tt_catchall:.1f} min" if not math.isnan(mean_tt_catchall)
              else f"     Mean travel time — valid zones: {mean_tt_valid:.1f} min")
        if catchall_by_mode:
            bad_modes = {k: v for k, v in catchall_by_mode.items() if v > 0}
            if bad_modes:
                print(f"     Catch-all by mode: "
                      + "  ".join(f"{k}={v:.1f}%" for k, v in sorted(bad_modes.items())))
        print("  ─────────────────────────────────────────────────────────")

    return metrics


# ===========================================================================
# 3. LOS METRICS  (was 2)
# ===========================================================================

FUEL_COST_PER_KM = 0.15   # $/km approximate fuel cost proxy

def compute_los(plan: pd.DataFrame,
                skim_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add level-of-service attributes to each trip row:
        time_period      — am_peak / midday / pm_peak / evening / late / early
        generalised_cost — weighted time + distance cost proxy (minutes equivalent)
        in_vehicle_time  — auto IVT from skim (same as travtime for DA)
        cost_proxy       — fuel cost proxy (dist * FUEL_COST_PER_KM)
    """
    df = plan.copy()

    # time period
    df['time_period'] = df['STARTIME'].apply(_time_period)

    # only compute LOS for trip rows
    trip_mask = df['row_type'] == 'trip'

    # cost proxy (distance * fuel rate) — zero for walk/bike/transit
    if 'dist' in df.columns:
        motorised = df['mode_name'].isin(['DA', 'Other'])
        df.loc[trip_mask & motorised, 'cost_proxy'] = (
            df.loc[trip_mask & motorised, 'dist'] * FUEL_COST_PER_KM
        )
    # ensure cost_proxy column always exists as a numeric Series, not a Python float
    if 'cost_proxy' not in df.columns:
        df['cost_proxy'] = 0.0
    df['cost_proxy'] = pd.to_numeric(df['cost_proxy'], errors='coerce').fillna(0.0)

    # generalised cost (value of time = $15/hr → 1 min = $0.25)
    # GC = travtime + cost_proxy / (15/60)  → minutes equivalent
    VOT = 15.0 / 60.0   # $/min
    df['generalised_cost'] = df.get('travtime', 0.0).fillna(0.0)
    if 'cost_proxy' in df.columns:
        df.loc[trip_mask, 'generalised_cost'] += (
            df.loc[trip_mask, 'cost_proxy'] / VOT
        )

    # in-vehicle time = travtime for motorised; walk/bike use distance proxy
    df['in_vehicle_time'] = np.where(
        df['mode_name'].isin(['DA', 'Other', 'Transit']),
        df.get('travtime', 0.0).fillna(0.0),
        0.0
    )

    return df


# ===========================================================================
# 3. AIMSUN OUTPUT FORMAT
# ===========================================================================

def build_aimsun_trips(plan: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trip legs into Aimsun demand table format.

    Aimsun requires one row per trip with:
        person_id, trip_id, OTAZ, DTAZ,
        departure_time (minutes from midnight),
        mode, purpose, time_period,
        travtime, dist, generalised_cost

    Filters out:
        - return-home legs (purpose='home') if you want production-based
        - intra-zonal trips (OTAZ == DTAZ) — optional
        - trips with invalid zones
    """
    trips = plan[plan['row_type'] == 'trip'].copy()

    # generate sequential trip IDs per person
    trips = trips.sort_values(['person_id', 'STARTIME']).copy()
    trips['trip_id'] = (
        trips.groupby('person_id').cumcount() + 1
    )
    trips['trip_id'] = (
        trips['person_id'].astype(str) + '_' +
        trips['trip_id'].astype(str)
    )

    # rename for Aimsun
    out = trips[[
        'person_id', 'trip_id',
        'OTAZ', 'DTAZ',
        'STARTIME', 'ENDTIME',
        'mode', 'mode_name',
        'purpose', 'time_period',
        'travtime',
    ]].copy()

    # add optional columns if available
    for col in ['dist', 'generalised_cost', 'in_vehicle_time',
                'cost_proxy', 'crash', '_fixed', '_fix_reason']:
        if col in trips.columns:
            out[col] = trips[col].values

    out = out.rename(columns={'STARTIME': 'departure_time'})

    # filter invalid zones
    out = out[(out['OTAZ'] > 0) & (out['DTAZ'] > 0)].copy()

    return out


def build_activities_summary(plan: pd.DataFrame) -> pd.DataFrame:
    """
    Extract activity rows into a summary table for visualisation
    and tour-level statistics.
    """
    acts = plan[plan['row_type'] == 'activity'].copy()
    acts['dwell_min'] = (acts['ENDTIME'] - acts['STARTIME']).clip(lower=0)

    keep = ['person_id', 'purpose', 'OTAZ', 'DTAZ',
            'STARTIME', 'ENDTIME', 'dwell_min', 'time_period',
            'predicted_duration_min']
    keep = [c for c in keep if c in acts.columns]

    return acts[keep].copy()


# ===========================================================================
# LOCATION ATTACHMENT  (pin zone centroids onto trip rows)
# ===========================================================================

def attach_location_coords(
    trips: pd.DataFrame,
    sa2_geo: Optional[pd.DataFrame] = None,
    zone_centroids: Optional[dict] = None,
    jitter_m: float = 150.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Attach origin and destination lat/lon to every trip row.

    Adds columns:
        origin_lat, origin_lon   ← centroid of OTAZ + jitter
        dest_lat,   dest_lon     ← centroid of DTAZ + jitter
        origin_x,   origin_y     ← same as lat/lon (alias for DynusT)
        dest_x,     dest_y       ← same as lat/lon (alias for DynusT)

    Parameters
    ----------
    trips          : trip table from build_aimsun_trips()
    sa2_geo        : zone centroid DataFrame with zone-id + lat + lon columns.
                     If None, falls back to zone_centroids dict.
    zone_centroids : {zone_id: (lat, lon)} dict — used if sa2_geo is None.
                     Build this from get_zone_centroids_rich(landuse) in the
                     pipeline (spatial_assignment.py).
    jitter_m       : random jitter in metres so multiple trips from the same
                     zone don't all plot on exactly the same pixel.
    seed           : RNG seed for reproducible jitter.
    """
    rng = np.random.default_rng(seed)
    df  = trips.copy()

    # ── build centroid lookup ──────────────────────────────────────────────
    centroid: dict = {}

    if sa2_geo is not None:
        zc = next(
            (c for c in ['SA2_MAIN16', 'ORDER', 'DTAZ', 'zone', 'zone_id']
             if c in sa2_geo.columns), None
        )
        if zc and 'lat' in sa2_geo.columns and 'lon' in sa2_geo.columns:
            for _, r in sa2_geo.dropna(subset=[zc, 'lat', 'lon']).iterrows():
                try:
                    centroid[int(r[zc])] = (float(r['lat']), float(r['lon']))
                except (ValueError, TypeError):
                    pass

    if not centroid and zone_centroids:
        centroid = {int(k): v for k, v in zone_centroids.items()}

    if not centroid:
        logger.warning("attach_location_coords: no centroid data — lat/lon will be NaN")
        for col in ['origin_lat','origin_lon','dest_lat','dest_lon',
                    'origin_x','origin_y','dest_x','dest_y']:
            df[col] = np.nan
        return df

    jitter_deg = jitter_m / 111_000.0
    n = len(df)

    # ── vectorised centroid lookup ─────────────────────────────────────────
    def _coords(zone_series, suffix):
        zones  = pd.to_numeric(zone_series, errors='coerce').fillna(0).astype(int)
        lats   = np.full(n, np.nan)
        lons   = np.full(n, np.nan)
        jlat   = rng.uniform(-jitter_deg, jitter_deg, n)
        jlon   = rng.uniform(-jitter_deg, jitter_deg, n)
        for i, zid in enumerate(zones):
            c = centroid.get(int(zid))
            if c:
                lats[i] = c[0] + jlat[i]
                lons[i] = c[1] + jlon[i]
        df[f'{suffix}_lat'] = lats
        df[f'{suffix}_lon'] = lons
        df[f'{suffix}_x']   = lons   # DynusT uses x=longitude, y=latitude
        df[f'{suffix}_y']   = lats

    _coords(df['OTAZ'], 'origin')
    _coords(df['DTAZ'], 'dest')

    n_placed = df['origin_lat'].notna().sum()
    logger.info(f"attach_location_coords: {n_placed}/{n} trips have coordinates")
    return df


# ===========================================================================
# MOTORISED TRIP EXPORTS  (DA, Shared Ride, Transit)
# ===========================================================================

def save_motorised_trips(
    trips_all: pd.DataFrame,
    save_dir: str,
) -> dict:
    """
    Filter trip table to motorised modes and save separate CSV files.

    Modes saved:
        DA (mode=1)       → trips_DA.csv
        Other/SR (mode=2) → trips_SR.csv
        Transit (mode=5)  → trips_Transit.csv
        All motorised     → trips_motorised.csv   (combined)

    Parameters
    ----------
    trips_all  : full trip table from build_aimsun_trips() WITH location coords
                 attached (run attach_location_coords first).
    save_dir   : output directory.

    Returns
    -------
    dict of {label: DataFrame} for each motorised split.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = {}

    # resolve mode column (integer or string)
    if 'mode' in trips_all.columns:
        mode_int = pd.to_numeric(trips_all['mode'], errors='coerce').fillna(0).astype(int)
    else:
        mode_int = trips_all.get('mode_name', pd.Series('', index=trips_all.index)) \
                             .map(MODE_FROM_NAME).fillna(0).astype(int)

    splits = {
        'DA':      mode_int == 1,
        'SR':      mode_int == 2,
        'Transit': mode_int == 5,
    }

    combined_mask = mode_int.isin(MOTORISED_MODES)
    splits['motorised'] = combined_mask

    for label, mask in splits.items():
        sub = trips_all[mask].copy()
        if sub.empty:
            logger.info(f"save_motorised_trips: no trips for mode '{label}' — skipping")
            continue
        path = os.path.join(save_dir, f'trips_{label}.csv')
        sub.to_csv(path, index=False)
        saved[label] = sub
        print(f"  Saved: trips_{label}.csv  ({len(sub):,} trips)")

    return saved


# ===========================================================================
# ZONE CROSSWALK  (ABM zone numbering  ->  a specific DynusT network's zones)
# ===========================================================================

def apply_dynust_zone_crosswalk(motorised_trips: dict, crosswalk_path: str) -> dict:
    """
    Remap OTAZ/DTAZ from the ABM's zone numbering onto a target DynusT
    network's own zone numbering, and drop trips that fall outside that
    network's coverage.

    Needed because write_dynus_vehicle_dat() writes OTAZ/DTAZ straight into
    vehicle.dat's zone fields, which only works if the ABM's zone numbers
    already match the target network's numbering 1:1. For networks where
    they don't (e.g. a network covering a smaller sub-area, or with a
    different zone count) use build_dynust_zone_crosswalk.py to build a
    crosswalk CSV first (columns: ORDER, dynust_zone, dist_m, in_network),
    then pass its path here.

    Trips where the origin or destination ABM zone has in_network=False
    (i.e. no DynusT zone in this network is anywhere near it) are dropped
    entirely rather than forced onto a distant/wrong zone.

    Parameters
    ----------
    motorised_trips : dict from save_motorised_trips()  {label: DataFrame}
    crosswalk_path  : CSV from build_dynust_zone_crosswalk.py

    Returns
    -------
    dict, same shape as motorised_trips, with OTAZ/DTAZ remapped to the
    target network's zone numbers and out-of-coverage trips removed.
    """
    cw = pd.read_csv(crosswalk_path)
    order_to_dynust = cw.set_index('ORDER')['dynust_zone'].to_dict()
    in_network = set(cw.loc[cw['in_network'], 'ORDER'])
    _zone_name = (cw.set_index('ORDER')['SA2_NAME16'].to_dict()
                  if 'SA2_NAME16' in cw.columns else {})

    remapped = {}
    _oon_rows = []   # out-of-network audit rows, written next to vehicle.dat
    for label, df in motorised_trips.items():
        if df is None or len(df) == 0:
            remapped[label] = df
            continue
        d = df.copy()
        otaz = pd.to_numeric(d['OTAZ'], errors='coerce')
        dtaz = pd.to_numeric(d['DTAZ'], errors='coerce')
        keep = otaz.isin(in_network) & dtaz.isin(in_network)
        n_before = len(d)

        # ── audit the trips that FAIL the coverage test, per offending zone,
        # before dropping them - so vehicle.dat ships with an explicit
        # account of what demand the DTA network is not seeing.
        if (~keep).any():
            _bad = pd.concat([
                otaz[~otaz.isin(in_network)].rename('zone'),
                dtaz[~dtaz.isin(in_network)].rename('zone'),
            ])
            for z, n in _bad.value_counts().items():
                _oon_rows.append({
                    'segment': label, 'abm_zone': int(z) if pd.notna(z) else -1,
                    'zone_name': _zone_name.get(z, ''),
                    'n_trip_ends': int(n),
                    'reason': ('not in crosswalk' if z not in order_to_dynust
                               else 'in_network=False (outside DTA coverage)'),
                })

        d = d.loc[keep].copy()
        d['OTAZ'] = otaz.loc[keep].map(order_to_dynust).astype(int)
        d['DTAZ'] = dtaz.loc[keep].map(order_to_dynust).astype(int)
        logger.info(
            f"apply_dynust_zone_crosswalk[{label}]: {len(d):,}/{n_before:,} trips "
            f"kept (both endpoints inside target network's coverage)"
        )
        remapped[label] = d

    # Sidecar report next to the crosswalk: which zones carried demand that
    # never reached vehicle.dat, and how much. This is the "inform" channel
    # for out-of-network vehicles - read it whenever DTA volumes look light.
    if _oon_rows:
        _rep = pd.DataFrame(_oon_rows).sort_values('n_trip_ends', ascending=False)
        _rep_path = os.path.join(os.path.dirname(os.path.abspath(crosswalk_path)),
                                 'vehicle_dat_out_of_network.csv')
        try:
            _rep.to_csv(_rep_path, index=False)
            _tot = int(_rep['n_trip_ends'].sum())
            _top = _rep.head(5)[['abm_zone', 'zone_name', 'n_trip_ends']].to_dict('records')
            logger.warning(
                f"apply_dynust_zone_crosswalk: {_tot:,} trip-ends touched zones "
                f"outside the DTA network and were EXCLUDED from vehicle.dat. "
                f"Top zones: {_top}. Full report: {_rep_path}"
            )
        except Exception as _rep_e:
            logger.warning(f"out-of-network report write failed: {_rep_e}")
    return remapped


# ===========================================================================
# DYNUS-T  vehicle.dat  WRITER
# ===========================================================================

def write_dynus_vehicle_dat(
    motorised_trips: dict,
    save_dir: str,
    origin_link_map: Optional[dict] = None,
    user_class: int = 3,
    n_path_nodes: int = 1,
    indiff_band: float = 0.0,
    compliance_rate: float = 0.0,
    description: str = 'Generated by ABM pipeline stage6',
) -> str:
    """
    Write a DynusT/DYNASMART-P vehicle.dat file from the motorised trip splits.

    Format reference: DSP User's Guide v1.3.0, Section 4.23 (Table 4-27 /
    Figure 4-26) — page 78-79.  The 15 written fields mirror the layout of
    the reference output_vehicle.dat file.

    ┌──────────────────────────────────────────────────────────────────┐
    │  VEHICLE.DAT FORMAT (DYNASMART-P / DynusT)                       │
    ├──────────────────────────────────────────────────────────────────┤
    │  Record 1  (Basic data — one line)                               │
    │    Field 1  INT   Number of vehicles to load                     │
    │    Field 2  INT   Max destinations per vehicle (1 for O-D only)  │
    │    Field 3  Text  Optional description                           │
    │                                                                  │
    │  Record 2  (Header comment line)                                 │
    │                                                                  │
    │  Record 3+ (Vehicle characteristics — one line per vehicle)      │
    │    Field 1   veh_id     Vehicle ID (sequential from 1)           │
    │    Field 2   usec       Upstream node of generation link         │
    │    Field 3   dsec       Downstream node of generation link       │
    │    Field 4   stime      Starting time (min), ascending order     │
    │    Field 5   vehcls     Vehicle class: 5=car 6=HOV 2=truck       │
    │    Field 6   vehtype    1=car 2=truck                            │
    │    Field 7   ioc        Occupancy: 0=LOV 1=shared-ride           │
    │    Field 8   #ONode     Nodes in path (1 = no path.dat used)     │
    │    Field 9   #IntDe     Destinations incl. intermediate + final  │
    │    Field 10  info       En-route info: 0=none 1=available        │
    │    Field 11  ribf       Indifference band (min)                  │
    │    Field 12  comp       Compliance rate 0.0–1.0                  │
    │    Field 13  izone      Origin zone                              │
    │    Field 14  Evac       0=non-evacuation vehicle                 │
    │    Field 15  InitPos    Random initial position 0-1              │
    │                                                                  │
    │  Record 4+ (Destinations — one line per destination per vehicle) │
    │    Field 1   dest_zone  Zone number of the destination           │
    │    Field 2   dwell      Activity duration at destination (min);  │
    │                      0.00 for the final destination (always)     │
    └──────────────────────────────────────────────────────────────────┘

    Exact column widths (15 fields, 105 chars/record; veh_id starts each
    line with an 8-space leading offset, matching the reference file):

      veh_id %9d  usec %7d  dsec %7d  stime %8.2f  vehcls %6d
      vehtype %6d  ioc %6d  #ONode %6d  #IntDe %6d  info %6d
      ribf %8.4f  comp %8.4f  izone %5d  Evac %5d  InitPos %12.8f

    Mode → DynusT vehicle class + type + occupancy mapping
    --------------------------------------------------------
    DA (mode 1)       vehcls=5 car,  vehtype=1, ioc=0 LOV,  ribf=0.2000 comp=1.0000
    Other/SR (mode 2) vehcls=5 car,  vehtype=1, ioc=1 shared-ride
    Transit (mode 5)  vehcls=2 truck,vehtype=2, ioc=0 LOV

    Note: vehicles must be listed in ASCENDING departure-time order.
    usec/dsec are the upstream/downstream nodes of the generation link
    for the origin zone (DynusT rejects vehicles whose upstream ==
    downstream node).  If origin_link_map is not supplied, OTAZ is
    used as a proxy for both and such vehicles are dropped — always
    pass the map built from the network's origin.dat.  #ONode is the
    number of path nodes (1 = no pre-specified path).  InitPos is a
    random draw in [0,1], matching the reference output_vehicle.dat.

    A matching path.dat is ALWAYS written next to vehicle.dat (one line
    per vehicle: the upstream node of its generation link).  DynusT
    terminates at demand generation when path.dat is empty, even with
    #ONode=1.

    Parameters
    ----------
    motorised_trips  : dict from save_motorised_trips()  {label: DataFrame}
    save_dir         : output directory
    origin_link_map  : {OTAZ: (usec, dsec)} — upstream/downstream generation
                       link nodes per zone (from origin.dat).  If None,
                       OTAZ is used for both nodes.
    user_class       : default DynusT user class for DA and SR trips
                       (1=unresponsive, 2=SO, 3=UE — default 3)
    n_path_nodes     : number of path nodes per vehicle (1 = no path.dat)
    indiff_band      : indifference band in minutes (class 4 only, else 0)
    compliance_rate  : compliance rate (class 4 only; must be 0 for others)
    description      : text description written to Record 1

    Returns
    -------
    Path to the written vehicle.dat file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'vehicle.dat')

    # ── Collect all motorised trips, deduplicate, sort by departure time ──
    frames = [df for label, df in motorised_trips.items()
              if label != 'motorised' and df is not None and len(df) > 0]
    if not frames:
        logger.warning("write_dynus_vehicle_dat: no motorised trips — vehicle.dat not written")
        return path

    all_trips = (
        pd.concat(frames, ignore_index=True)
        .sort_values('departure_time')
        .reset_index(drop=True)
    )
    if 'trip_id' in all_trips.columns:
        all_trips = all_trips.drop_duplicates(subset='trip_id').reset_index(drop=True)

    n_vehicles = len(all_trips)
    max_stops  = 1   # one final destination per trip (no intermediate stops)

    # ── Normalise departure times to a 0-120 minute scale ──────────────────
    # Reference output_vehicle.dat uses stime 0-120; shift so the earliest
    # departure starts at 0 and scale the whole day to fit within 120 min.
    dep_min     = pd.to_numeric(all_trips['departure_time'], errors='coerce').fillna(0.0)
    _s0 = float(dep_min.min())
    _s1 = float(dep_min.max())
    _dep_range = (_s1 - _s0) or 1.0
    dep_min_120 = (dep_min - _s0) * (120.0 / _dep_range)
    otaz_arr = pd.to_numeric(all_trips['OTAZ'], errors='coerce').fillna(1).astype(int)
    dtaz_arr = pd.to_numeric(all_trips['DTAZ'], errors='coerce').fillna(1).astype(int)

    # ── Mode → DynusT vehicle type + occupancy + user class ───────────────
    if 'mode' in all_trips.columns:
        mode_int = pd.to_numeric(all_trips['mode'], errors='coerce').fillna(1).astype(int)
    else:
        mode_int = (
            all_trips.get('mode_name', pd.Series('DA', index=all_trips.index))
            .map({'DA': 1, 'Other': 2, 'Walk': 3, 'Bike': 4, 'Transit': 5})
            .fillna(1).astype(int)
        )

    # veh_type:  1=car(DA), 1=car(SR/HOV), 2=truck(Transit proxy)
    # occupancy: 1=LOV(DA/Transit), 2=HOV(SR)
    # ucls:      user_class param for DA/SR, 1=unresponsive for Transit
    veh_type_arr  = mode_int.map({1: 1, 2: 1, 5: 2}).fillna(1).astype(int)
    occupancy_arr = mode_int.map({1: 1, 2: 2, 5: 1}).fillna(1).astype(int)

    # ribf/compliance: 0.2000/1.0000 (matching the reference output_vehicle.dat)
    ribf_arr  = np.full(n_vehicles, 0.2000, dtype=float)
    comp_arr  = np.full(n_vehicles, 1.0000, dtype=float)

    # ── Generation-link nodes ──────────────────────────────────────────────
    # usec/dsec must be a real, distinct-node generation link for the origin
    # zone (DynusT rejects vehicles whose upstream == downstream node). Use the
    # origin_link_map when available; otherwise fall back to the zone number as
    # a node proxy. Any vehicle left with usec == dsec (e.g. a zone with no
    # entry in origin_link_map) is DROPPED — DynusT cannot load it.
    otaz_vals = otaz_arr.to_numpy()
    if origin_link_map:
        _lk = origin_link_map
        usec_arr = np.array([_lk.get(int(z), (int(z), int(z)))[0] for z in otaz_vals], dtype=int)
        dsec_arr = np.array([_lk.get(int(z), (int(z), int(z)))[1] for z in otaz_vals], dtype=int)
    else:
        usec_arr = otaz_arr.to_numpy()
        dsec_arr = otaz_arr.to_numpy()

    keep = usec_arr != dsec_arr
    if not bool(keep.all()):
        n_drop = int((~keep).sum())
        logger.warning(
            f"write_dynus_vehicle_dat: dropping {n_drop:,} vehicles whose "
            f"generation link has identical up/downstream nodes (DynusT cannot "
            f"load them)"
        )
        keep_ix = np.flatnonzero(keep)
        usec_arr   = usec_arr[keep_ix]
        dsec_arr   = dsec_arr[keep_ix]
        otaz_arr   = otaz_arr.iloc[keep_ix].reset_index(drop=True)
        dtaz_arr   = dtaz_arr.iloc[keep_ix].reset_index(drop=True)
        dep_min_120= dep_min_120.iloc[keep_ix].reset_index(drop=True)
        mode_int   = mode_int.iloc[keep_ix].reset_index(drop=True)
        veh_type_arr   = veh_type_arr.iloc[keep_ix].reset_index(drop=True)
        occupancy_arr  = occupancy_arr.iloc[keep_ix].reset_index(drop=True)
        ribf_arr   = ribf_arr[keep_ix]
        comp_arr   = comp_arr[keep_ix]
        n_vehicles = int(keep.sum())
        print(f"  Dropped {n_drop:,} vehicles with invalid generation links; "
              f"{n_vehicles:,} remaining")

    # InitPos: random initial position in [0,1] (as in the reference
    # output_vehicle.dat), seeded so repeated runs are reproducible.
    _seed_ip = int(n_vehicles) % (2**31 - 1)
    _initpos_arr = np.random.default_rng(_seed_ip).random(n_vehicles)

    # ── Write file ────────────────────────────────────────────────────────
    # newline='' keeps the explicit \r\n endings untranslated (avoids
    # \r\r\n on Windows); matches the reference file's CRLF line endings.
    # path.dat MUST be written too: DynusT terminates at demand generation
    # if path.dat is empty — one line per vehicle with the upstream node of
    # the generation link (matches #ONode=1; DynusT then builds the real
    # path itself, as seen in output_vehicle.dat).
    path_dat = os.path.join(save_dir, 'path.dat')
    with open(path, 'w', newline='') as f, open(path_dat, 'w', newline='') as fpath:

        # Record 1: vehicle count, max stops, description
        # (8-space leading offset; description matches reference file)
        f.write(f"{n_vehicles:15d}{max_stops:12d}    # of vehicles in the file, Max # of STOPs\r\n")

        # Record 2: column header (comment) — same as reference file
        f.write("        #   usec   dsec   stime vehcls vehtype ioc #ONode"
                " #IntDe info ribf    comp   izone Evac InitPos"
                "    VoT  tFlag pArrTime TP IniGas\r\n")

        for i in range(n_vehicles):
            usec = int(usec_arr[i])
            dsec = int(dsec_arr[i])
            otaz = int(otaz_arr.iloc[i])
            dtaz = int(dtaz_arr.iloc[i])

            stime   = float(dep_min_120.iloc[i])
            vt      = int(veh_type_arr.iloc[i])
            occ     = int(occupancy_arr.iloc[i])
            ribf    = float(ribf_arr[i])
            comp    = float(comp_arr[i])
            initpos = float(_initpos_arr[i])

            # vehcls: 5=passenger car, 2=truck (transit proxy)
            vehcls = 5 if vt == 1 else 2
            # ioc: 0=drive-alone (LOV), 1=shared-ride (HOV)
            ioc = 0 if occ == 1 else 1

            # 15-field vehicle record matching the reference output_vehicle.dat:
            # veh_id, usec, dsec, stime, vehcls, vehtype, ioc, #ONode, #IntDe,
            # info, ribf, comp, izone, Evac, InitPos.
            f.write(
                f"{i+1:9d}"           # veh_id (8-space leading offset)
                f"{usec:7d}"          # usec (upstream node of generation link)
                f"{dsec:7d}"          # dsec (downstream node)
                f"{stime:8.2f}"       # stime (departure minutes, normalised to 0-120)
                f"{vehcls:6d}"        # vehcls (5=car, 2=truck)
                f"{vt:6d}"            # vehtype (1=car, 2=truck)
                f"{ioc:6d}"           # ioc (0=LOV, 1=shared-ride)
                f"{1:6d}"             # #ONode (nodes in path; 1 = no path.dat)
                f"{max_stops:6d}"     # #IntDe (destinations incl. final)
                f"{0:6d}"             # info (en-route info: 0=none, 1=available)
                f"{ribf:8.4f}"        # ribf (indifference band, min)
                f"{comp:8.4f}"        # comp (compliance rate 0-1.0)
                f"{otaz:5d}"          # izone (origin zone number)
                f"{0:5d}"             # Evac (0=non-evacuation)
                f"{initpos:12.8f}"    # InitPos (random initial position 0-1)
                f"\r\n"
            )

            # Destination record — one line per destination (max_stops=1 here)
            # Activity duration = 0.00 for the final destination (always)
            f.write(f"{dtaz:12d}{0.0:7.2f}\r\n")

            # path.dat: one line per vehicle — initial path node = upstream
            # node of the generation link (per DSP User's Guide §4.24).
            fpath.write(f"{usec}\r\n")

    # ── Summary ───────────────────────────────────────────────────────────
    n_da      = int((mode_int == 1).sum())
    n_sr      = int((mode_int == 2).sum())
    n_transit = int((mode_int == 5).sum())
    print(
        f"  Saved: vehicle.dat + path.dat  ({n_vehicles:,} vehicles — "
        f"DA={n_da:,}  SR={n_sr:,}  Transit={n_transit:,})"
    )
    logger.info(
        f"write_dynus_vehicle_dat: {n_vehicles} vehicles → {path} (+ path.dat) | "
        f"DA={n_da} SR={n_sr} Transit={n_transit} | "
        f"dep_time range: {dep_min.min():.1f}–{dep_min.max():.1f} min"
    )
    return path


# ===========================================================================
# DYNUS-T  od.dat  WRITER  (aggregate OD demand by time period)
# ===========================================================================

# DynusT time-period intervals (minutes from midnight) for OD demand files.
# Each interval produces one OD matrix in od.dat.
# Matches the TIME_PERIODS defined at the top of this module.
DYNUS_OD_PERIODS = [
    (360,  540,  'am_peak'),
    (540,  900,  'midday'),
    (900, 1080,  'pm_peak'),
    (1080,1440,  'off_peak'),
]


def write_dynus_od_dat(
    trips: pd.DataFrame,
    save_dir: str,
    n_zones: int = 371,
    periods: list = None,
    modes: list = None,
    filename: str = 'od.dat',
) -> str:
    """
    Write a DynusT aggregate OD demand file (od.dat) from the trip table.

    Format: one matrix per time period, sparse (only non-zero OD pairs written).

    ┌──────────────────────────────────────────────────────────────────┐
    │  OD.DAT FORMAT (DYNASMART-P)                                     │
    ├──────────────────────────────────────────────────────────────────┤
    │  Line 1: [n_zones]                                               │
    │  Line 2: [n_periods]                                             │
    │  For each period:                                                │
    │    [start_min] [end_min]    ← period boundaries (minutes)       │
    │    [origin] [dest] [demand] ← one per non-zero OD pair          │
    │    -1 -1 -1                 ← end-of-period sentinel             │
    └──────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    trips      : Aimsun trip table (departure_time, OTAZ, DTAZ, mode columns)
    save_dir   : output directory
    n_zones    : number of zones (371 for SEQ)
    periods    : list of (start_min, end_min, label) tuples (default DYNUS_OD_PERIODS)
    modes      : restrict to these mode_name values (None = all motorised modes)
    filename   : output filename (default 'od.dat')

    Returns
    -------
    Path to the written od.dat file.
    """
    if periods is None:
        periods = DYNUS_OD_PERIODS

    if modes is None:
        modes = list(MOTORISED_NAMES)   # DA, Other, Transit

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    # ── Resolve mode filter ────────────────────────────────────────────────
    dep_col = next((c for c in ['departure_time', 'STARTIME'] if c in trips.columns), None)
    if dep_col is None:
        logger.warning("write_dynus_od_dat: no departure_time/STARTIME column — od.dat not written")
        return path

    dep_min  = pd.to_numeric(trips[dep_col], errors='coerce')
    otaz_arr = pd.to_numeric(trips['OTAZ'],  errors='coerce').fillna(0).astype(int)
    dtaz_arr = pd.to_numeric(trips['DTAZ'],  errors='coerce').fillna(0).astype(int)

    if 'mode_name' in trips.columns:
        mode_name = trips['mode_name'].fillna('DA')
    elif 'mode' in trips.columns:
        mode_name = pd.to_numeric(trips['mode'], errors='coerce').fillna(1).astype(int) \
                      .map(MODE_NAMES).fillna('DA')
    else:
        mode_name = pd.Series('DA', index=trips.index)

    # filter to valid zones and motorised modes
    valid = (
        otaz_arr.between(1, n_zones) &
        dtaz_arr.between(1, n_zones) &
        mode_name.isin(modes) &
        dep_min.notna()
    )
    trips_m = trips[valid].copy()
    dep_m   = dep_min[valid]
    otaz_m  = otaz_arr[valid]
    dtaz_m  = dtaz_arr[valid]

    n_periods = len(periods)
    period_rows = []
    for (t_start, t_end, label) in periods:
        mask = (dep_m >= t_start) & (dep_m < t_end)
        sub_o = otaz_m[mask]
        sub_d = dtaz_m[mask]
        od_counts = (
            pd.DataFrame({'o': sub_o.values, 'd': sub_d.values})
            .groupby(['o', 'd'])
            .size()
            .reset_index(name='demand')
        )
        period_rows.append((t_start, t_end, od_counts))
        logger.info(
            f"  OD period {label} ({t_start}-{t_end} min): "
            f"{len(sub_o):,} trips  {len(od_counts):,} non-zero OD pairs"
        )

    # ── Write file ────────────────────────────────────────────────────────
    with open(path, 'w') as f:
        f.write(f"{n_zones}\n")
        f.write(f"{n_periods}\n")
        for (t_start, t_end, od_counts) in period_rows:
            f.write(f"{t_start} {t_end}\n")
            for _, row in od_counts.iterrows():
                f.write(f"{int(row['o'])} {int(row['d'])} {row['demand']:.4f}\n")
            f.write("-1 -1 -1\n")   # end-of-period sentinel

    total_pairs  = sum(len(od) for _, _, od in period_rows)
    total_demand = sum(od['demand'].sum() for _, _, od in period_rows)
    print(
        f"  Saved: {filename}  ({n_periods} periods, {total_pairs:,} OD pairs, "
        f"{total_demand:,.0f} total demand trips)"
    )
    logger.info(
        f"write_dynus_od_dat: {path} | {n_periods} periods | "
        f"{total_pairs} OD pairs | {total_demand:.0f} trips"
    )
    return path


# ===========================================================================
# 4. TOUR AGGREGATION
# ===========================================================================

def aggregate_to_tours(trips: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Aimsun trip table to tour-level summary.

    A tour = all trips by one person between two consecutive home visits.
    Returns one row per tour with:
        tour_id, n_trips, primary_purpose, start_time, end_time,
        total_travtime, tour_mode (dominant mode), OTAZ, main_DTAZ
    """
    tours = []

    for pid, grp in trips.groupby('person_id', sort=False):
        grp = grp.sort_values('departure_time').reset_index(drop=True)

        tour_num   = 1
        tour_trips = []

        for row_idx, row in grp.iterrows():
            tour_trips.append(row)
            # end of tour = trip going home OR last trip for this person
            is_home_trip  = str(row.get('purpose', '')).lower() == 'home'
            is_last_trip  = (row_idx == grp.index[-1])

            if is_home_trip or is_last_trip:
                if tour_trips:
                    tt        = pd.DataFrame(tour_trips)
                    main_purp = (tt['purpose']
                                 .replace('home', np.nan)
                                 .dropna()
                                 .mode())
                    main_purp = main_purp.iloc[0] if len(main_purp) else 'other'
                    mode_col  = 'mode_name' if 'mode_name' in tt.columns else 'mode'
                    mode_mode = (tt[mode_col]
                                 .replace('', np.nan)
                                 .dropna()
                                 .mode())
                    mode_mode = str(mode_mode.iloc[0]) if len(mode_mode) else 'DA'

                    tours.append({
                        'person_id':      pid,
                        'tour_id':        f"{pid}_{tour_num}",
                        'n_trips':        len(tt),
                        'primary_purpose':main_purp,
                        'start_time':     tt['departure_time'].min(),
                        'end_time':       tt['ENDTIME'].max()
                                          if 'ENDTIME' in tt.columns else np.nan,
                        'total_travtime': tt['travtime'].sum()
                                          if 'travtime' in tt.columns else np.nan,
                        'tour_mode':      mode_mode,
                        'OTAZ':           int(tt['OTAZ'].iloc[0]),
                        'main_DTAZ':      int(tt['DTAZ'].iloc[0]),
                    })
                    tour_num  += 1
                    tour_trips = []

    return pd.DataFrame(tours)


# ===========================================================================
# 5. DIAGNOSTIC SUMMARY
# ===========================================================================

def print_plan_summary(plan: pd.DataFrame,
                       trips_aimsun: pd.DataFrame):
    """Print a readable summary of the final plan."""
    trips = plan[plan['row_type'] == 'trip']
    acts  = plan[plan['row_type'] == 'activity']

    print(f"\n{'='*60}")
    print("STAGE 6 — PLAN SUMMARY")
    print(f"{'='*60}")
    print(f"  Persons:    {plan['person_id'].nunique()}")
    print(f"  Trip legs:  {len(trips)}")
    print(f"  Activities: {len(acts)}")

    if 'travtime' in trips.columns:
        tt = trips['travtime'].dropna()
        print(f"\n  Travel time distribution (min):")
        print(f"    median={tt.median():.1f}  mean={tt.mean():.1f}  "
              f"p5={tt.quantile(0.05):.1f}  p95={tt.quantile(0.95):.1f}")

    if 'predicted_duration_min' in acts.columns:
        print(f"\n  Activity dwell time by purpose (median min):")
        for p in PURPOSES:
            sub = acts.loc[acts['purpose'] == p, 'predicted_duration_min'].dropna()
            if len(sub):
                print(f"    {p:12s}: n={len(sub):5d}  "
                      f"median={sub.median():5.0f}  "
                      f"p5={sub.quantile(0.05):4.0f}  "
                      f"p95={sub.quantile(0.95):5.0f}")

    if 'mode_name' in trips.columns:
        print(f"\n  Mode split:")
        mc = trips['mode_name'].value_counts()
        total = mc.sum()
        for mode, count in mc.items():
            print(f"    {mode:10s}: {count:6d}  ({100*count/total:.1f}%)")

    if 'time_period' in trips.columns:
        print(f"\n  Trips by time period:")
        tp = trips['time_period'].value_counts()
        for period, count in tp.sort_index().items():
            print(f"    {period:12s}: {count}")

    n_fixed = plan['_fixed'].sum() if '_fixed' in plan.columns else 0
    if n_fixed:
        print(f"\n  Records fixed by validation: {n_fixed}")
        if '_fix_reason' in plan.columns:
            reasons = (plan.loc[plan['_fixed'], '_fix_reason']
                       .str.split('; ')
                       .explode()
                       .value_counts())
            for reason, count in reasons.items():
                print(f"    {reason}: {count}")

    print(f"\n  Aimsun trips table: {len(trips_aimsun)} rows")
    print(f"  Zones used: {trips_aimsun['OTAZ'].nunique()} origin, "
          f"{trips_aimsun['DTAZ'].nunique()} destination")


# ===========================================================================
# 5. DESTINATION REPAIR LOOP
# ===========================================================================

# Minimum acceptable dwell time per purpose after travel is deducted (min)
MIN_DWELL = {
    'work':      60,   # at least 1 hour at work
    'edu':       60,   # at least 1 hour at school
    'social':    15,
    'escort':     5,
    'shopping':  10,
    'othmaint':  10,
    'othdiscr':  10,
    'eat':       15,
}

# How much to tighten the distance constraint each repair iteration
REPAIR_TT_REDUCTION = 0.75   # multiply max_tt by this each pass

# Runaway protection for the per-activity Larch repair loop. Each repair
# attempt builds a 1-case DataTree and triggers a fresh sharrow flow compile
# (nb.TypingError → recompile → numpy fallback, ~1.5-2 s per call). Job
# 23540863 spent ~17 h doing 32,586 such calls inside chunk 1/91 of Stage 6
# and was killed by the 24 h walltime without writing any output. These caps
# bound the damage: the loop stops after the time budget and reports how many
# persons were left unrepaired instead of silently burning the walltime.
REPAIR_TIME_BUDGET_S   = float(os.environ.get('STAGE6_REPAIR_TIME_BUDGET_S', '300'))
# If the first sampled activities average more than this many seconds per
# Larch call and nothing has been repaired yet, the per-case path is in the
# recompile-every-call failure mode — abandon instead of limping through.
REPAIR_SLOW_CALL_S     = float(os.environ.get('STAGE6_REPAIR_SLOW_CALL_S', '2.0'))
REPAIR_SLOW_CALL_PROBE = 10   # activities to sample before the slow-call check


def repair_compressed_dwells(
    plan: pd.DataFrame,
    stage5_models: dict,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    skims,
    skim_lookup: dict,
    max_iterations: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For persons whose mandatory activity dwell time is compressed below
    MIN_DWELL, re-run destination choice with a tighter distance constraint
    and rebuild that person's trip chain.

    How it works
    ------------
    1. After validation, compute actual dwell = ENDTIME - STARTIME for
       each activity. Compare to MIN_DWELL[purpose].
    2. For each person with at least one under-threshold dwell on a
       mandatory activity (work/edu), find the offending activity and
       its OTAZ (chained origin).
    3. Re-predict DTAZ using the Stage 5 destination model with the
       availability constraint tightened:
           od.SOVTOLL_DIST__AM < current_max_dist * REPAIR_TT_REDUCTION
       This forces a closer destination.
    4. Recompute travel time from the new DTAZ and check if dwell
       now exceeds MIN_DWELL. If yes, update the plan. If no, try again
       up to max_iterations times.
    5. If no closer destination improves the dwell, leave it unchanged
       and flag the person as 'unresolvable'.

    Parameters
    ----------
    plan           : validated plan from validate_and_fix()
    stage5_models  : dict returned by run_stage5() — {segment: {mode_model, dest_model,...}}
    persons        : HHTS persons DataFrame
    landuse        : zone landuse
    skims          : open OMX handle
    skim_lookup    : {(otaz,dtaz): {col: val}} from _skim_lookup()
    max_iterations : how many times to try tightening the constraint
    verbose        : print progress

    Returns
    -------
    plan with repaired trip chains where possible
    """
    try:
        from stage5_destination_choice import (
            _mode_tree, _dest_tree, _logsums,
            _get_tt, MODE_TT_CAP,
        )
        from zsim_helper import monte_sample
        import xarray as xr
        import larch as lx
        from larch import X
    except ImportError as e:
        logger.warning(f"Repair loop skipped — import failed: {e}")
        return plan

    df = plan.copy()

    # compute actual dwell per activity row
    act_mask = df['row_type'] == 'activity'
    df.loc[act_mask, '_actual_dwell'] = (
        df.loc[act_mask, 'ENDTIME'] - df.loc[act_mask, 'STARTIME']
    ).clip(lower=0)

    # find persons with compressed mandatory dwells
    mandatory = df[act_mask & df['purpose'].isin(['work', 'edu'])].copy()
    mandatory['_min_dwell'] = mandatory['purpose'].map(MIN_DWELL)
    compressed = mandatory[
        mandatory['_actual_dwell'] < mandatory['_min_dwell']
    ]

    if compressed.empty:
        if verbose:
            print("  No compressed dwells found — repair loop not needed")
        return df

    bad_pids = compressed['person_id'].unique()
    if verbose:
        print(f"  {len(bad_pids)} persons have compressed mandatory dwells")

    pp = persons.copy()
    if 'person_id' not in pp.columns and 'PERSID' in pp.columns:
        pp = pp.rename(columns={'PERSID': 'person_id'})

    repaired   = 0
    unresolved = 0
    n_attempt_errors = 0
    _first_attempt_error = None
    _n_acts_tried = 0
    _t_start = time.monotonic()
    _aborted_reason = None

    # The per-case Larch calls below re-trigger sharrow flow compilation on
    # every activity; larch/sharrow log an 18-line "nb.TypingError" ERROR
    # block plus "writing fresh flow code" INFO lines on each one (job
    # 23540863 produced a 1M-line log this way). Suppress third-party log
    # noise for the duration of the loop and restore afterwards.
    _muted_loggers = {}
    for _ln in ('larch', 'sharrow'):
        _lg = logging.getLogger(_ln)
        _muted_loggers[_ln] = _lg.level
        _lg.setLevel(logging.CRITICAL)

    try:
      for pid in bad_pids:
        if time.monotonic() - _t_start > REPAIR_TIME_BUDGET_S:
            _aborted_reason = (
                f"time budget exhausted ({REPAIR_TIME_BUDGET_S:.0f}s, "
                f"override with STAGE6_REPAIR_TIME_BUDGET_S)"
            )
            break
        if (_n_acts_tried >= REPAIR_SLOW_CALL_PROBE and repaired == 0
                and (time.monotonic() - _t_start) / _n_acts_tried > REPAIR_SLOW_CALL_S):
            _aborted_reason = (
                f"first {_n_acts_tried} activities averaged "
                f"{(time.monotonic() - _t_start) / _n_acts_tried:.1f}s/repair with 0 "
                "repaired — per-case Larch path is recompiling its flow on every "
                "call (see STAGE6_REPAIR_SLOW_CALL_S); not viable at this scale"
            )
            break
        person_rows = df[df['person_id'] == pid].copy()
        person_acts = person_rows[person_rows['row_type'] == 'activity']
        person_comp = compressed[compressed['person_id'] == pid]

        for _, bad_act in person_comp.iterrows():
            purpose = bad_act['purpose']
            segment = 'work' if purpose == 'work' else 'edu'
            seg_models = stage5_models.get(segment, {})
            if not seg_models:
                unresolved += 1
                continue

            dest_mdl  = seg_models['dest_model']
            mode_mdl  = seg_models['mode_model']

            # find the trip leg TO this activity to get OTAZ
            act_idx   = bad_act.name
            act_order = person_rows.index.tolist().index(act_idx)
            trip_otaz = (
                int(person_rows.iloc[act_order - 1]['OTAZ'])
                if act_order > 0
                else int(bad_act.get('OTAZ', 1))
            )
            if trip_otaz <= 0:
                trip_otaz = 1

            day_budget = float(bad_act.get('end_of_day', 1080)
                               - bad_act.get('STARTIME', 480))

            current_max_tt = float(MODE_TT_CAP.get(1, 120.0))  # start from DA cap
            success = False
            _n_acts_tried += 1

            for attempt in range(max_iterations):
                # tighten distance constraint
                current_max_tt *= REPAIR_TT_REDUCTION
                max_dist_km     = current_max_tt * (50.0 / 60.0)  # ~50 km/h speed

                act_row_df = pd.DataFrame([{
                    'ptour_id':        f"{pid}_repair_{attempt}",
                    'person_id':       pid,
                    'tour_id':         0,
                    'OTAZ':            trip_otaz,
                    'DTAZ':            trip_otaz,
                    'tour_mode':       int(bad_act.get('mode', 1)),
                    'main_purpose':    purpose,
                    'primary_purpose': purpose,
                    'outbound_stops':  0,
                    'n_shopping_out':  0,
                    'incp':            int(bad_act.get('incp', 8)),
                    'indp':            int(bad_act.get('indp', 19)),
                    'vehrd':           int(bad_act.get('vehrd', 1)),
                    'STUDYING':        int(bad_act.get('STUDYING', 0)),
                    'sexp':            int(bad_act.get('sexp', 1)),
                    '_walk_avail':     1,
                    '_bike_avail':     1,
                }])

                try:
                    dt_m  = _mode_tree(act_row_df, pp, landuse, skims)
                    ls    = _logsums(mode_mdl, dt_m, verbose=False)
                    dt_d  = _dest_tree(act_row_df, pp, landuse, skims, ls)

                    # temporarily tighten availability — plain string, not X()
                    # Must use SOVTOLL_DIST__AM — od.DIST is not a valid column
                    # in the sharrow-compiled expression context for this OMX.
                    orig_avail = dest_mdl.availability_ca_var
                    dest_mdl.availability_ca_var = f'od.SOVTOLL_DIST__AM < {max_dist_km:.1f}'
                    dest_mdl.datatree = dt_d
                    dest_mdl.mangle()
                    dest_mdl.unmangle()

                    new_dtaz  = int(monte_sample(dest_mdl.probability())[0]) + 1
                    dest_mdl.availability_ca_var = orig_avail  # restore

                    # compute new travel time
                    new_tt = _get_tt(skim_lookup, trip_otaz, new_dtaz,
                                     int(bad_act.get('mode', 1)))

                    # check if dwell now OK
                    remaining = day_budget - new_tt
                    min_d     = MIN_DWELL.get(purpose, 30)
                    if remaining >= min_d:
                        # update the activity row in df
                        df.at[act_idx, 'DTAZ'] = new_dtaz
                        df.at[act_idx, '_fixed'] = True
                        df.at[act_idx, '_fix_reason'] = (
                            (df.at[act_idx, '_fix_reason'] or '') +
                            f'; repaired_dest(attempt={attempt+1})')
                        # update the preceding trip leg DTAZ too
                        if act_order > 0:
                            trip_idx = person_rows.index[act_order - 1]
                            df.at[trip_idx, 'DTAZ']    = new_dtaz
                            df.at[trip_idx, 'travtime'] = new_tt
                            df.at[trip_idx, 'ENDTIME']  = (
                                df.at[trip_idx, 'STARTIME'] + new_tt
                            )
                        repaired += 1
                        success   = True
                        break

                except Exception as e:
                    n_attempt_errors += 1
                    if _first_attempt_error is None:
                        # First failure at WARNING with full context; the rest at
                        # DEBUG (they are almost always the same root cause and
                        # job 23540863 showed what per-attempt spam does to logs).
                        _first_attempt_error = repr(e)
                        logger.warning(
                            f"Repair attempt failed (first of possibly many) "
                            f"pid={pid} purpose={purpose}: {e!r} — further "
                            f"attempt errors logged at DEBUG"
                        )
                    else:
                        logger.debug(f"Repair attempt {attempt+1} failed "
                                     f"pid={pid} purpose={purpose}: {e}")
                    continue

            if not success:
                unresolved += 1
    finally:
        for _ln, _lvl in _muted_loggers.items():
            logging.getLogger(_ln).setLevel(_lvl)

    if _aborted_reason is not None:
        _n_skipped = len(bad_pids) - repaired - unresolved
        logger.warning(
            f"repair_compressed_dwells ABORTED: {_aborted_reason}. "
            f"{repaired} repaired, {unresolved} unresolvable, "
            f"~{max(_n_skipped, 0)} persons left unrepaired (their compressed "
            f"dwells pass through unchanged — same outcome as --no-repair)."
        )
    _summary = (f"  Repair loop: {repaired} activities repaired, "
                f"{unresolved} unresolvable, {n_attempt_errors} attempt errors"
                + (f", ABORTED ({_aborted_reason})" if _aborted_reason else ""))
    if verbose:
        print(_summary)
    logger.info(_summary.strip())

    return df


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def run_stage6(
    plan: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: Optional[pd.DataFrame] = None,
    skim_df: Optional[pd.DataFrame] = None,
    stage5_models: Optional[dict] = None,
    skims = None,
    skim_lookup: Optional[dict] = None,
    sa2_geo: Optional[pd.DataFrame] = None,
    save_dir: str = 'saved_data',
    dynust_zone_crosswalk: Optional[str] = None,
    repair_dwells: bool = True,
    max_repair_iterations: int = 3,
    dynus_sim_start_min: int = DYNUS_SIM_START_MIN,
    dynus_value_of_time: float = 15.0,
    origin_link_map: Optional[dict] = None,
    verbose: bool = True,
) -> tuple:
    """
    Validate, post-process and export the Stage 5 trip chain.

    Parameters
    ----------
    plan               : Stage 5 output — interleaved trip + activity rows
    persons            : from process_survey()["persons"]
    landuse            : from sc.get_land()
    skim_df            : flat skim DataFrame for LOS metrics
    stage5_models      : models dict from run_stage5() — enables repair loop
    skims              : open OMX handle — needed for repair loop
    skim_lookup        : {(otaz,dtaz): dict} — needed for repair loop travel times
    sa2_geo            : zone centroid table (ORDER/zone_id, lat, lon).
                         From get_zone_centroids_rich(landuse) in spatial_assignment.
                         Used to pin lat/lon onto every trip row.
    save_dir           : directory to write outputs
    dynust_zone_crosswalk : path to a crosswalk CSV from
                         build_dynust_zone_crosswalk.py (columns: ORDER,
                         dynust_zone, dist_m, in_network). Pass this when the
                         target DynusT network's zone numbering doesn't match
                         the ABM's own zone numbers 1:1 (e.g. a network that
                         only covers a sub-area) — remaps OTAZ/DTAZ and drops
                         trips outside that network's coverage before writing
                         vehicle.dat. Leave None to write OTAZ/DTAZ as-is
                         (only correct when the two zone systems already match).
    repair_dwells      : whether to run destination repair loop (default True)
    max_repair_iterations : how many times to tighten distance per person
    dynus_sim_start_min : DynusT simulation start (minutes from midnight, default 360=6am)
    dynus_value_of_time : value of time written to vehicle.dat ($/hour)
    origin_link_map   : {zone: (usec, dsec)} — generation-link node pairs per
                        zone (from the network's origin.dat). Pass this so
                        vehicle.dat gets real, distinct up/downstream nodes;
                        otherwise zones fall back to the zone number as a node
                        proxy and any vehicle left with identical nodes is
                        dropped (DynusT cannot load it).
    verbose            : print progress

    Returns
    -------
    final_plan    : validated plan with LOS metrics and location coords added
    trips_aimsun  : Aimsun-format trip table with location coords

    Output files
    ------------
    {save_dir}/stage6_output.csv          — full validated plan with coords
    {save_dir}/trips_for_aimsun.csv       — Aimsun demand table with coords
    {save_dir}/activities_summary.csv     — activity-level summary
    {save_dir}/tours_summary.csv          — tour-level aggregation
    {save_dir}/trips_DA.csv               — Drive-alone trips with coords
    {save_dir}/trips_SR.csv               — Shared-ride trips with coords
    {save_dir}/trips_Transit.csv          — Transit trips with coords
    {save_dir}/trips_motorised.csv        — All motorised trips combined
    {save_dir}/vehicle.dat                — DynusT vehicle demand file (DYNASMART-P format)
    {save_dir}/od.dat                     — DynusT aggregate OD demand by time period

    Per-zone output (one sub-folder per origin zone):
    {save_dir}/zones/zone_{otaz}/stage6_output.csv      — plan rows for persons from OTAZ
    {save_dir}/zones/zone_{otaz}/trips_for_aimsun.csv   — Aimsun trip rows originating from OTAZ
    """
    print("=" * 60)
    print("STAGE 6 — Plan Validation & Aimsun / DynusT Export")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # ── handle legacy input ────────────────────────────────────────────────
    if 'row_type' not in plan.columns:
        print("  NOTE: treating all rows as activities (no row_type column)")
        plan = plan.copy()
        plan['row_type'] = 'activity'
        plan['travtime'] = plan.get('tt_avg', 10.0).fillna(10.0)

    # ── ensure ENDTIME exists ──────────────────────────────────────────────────
    # Stage 5 always computes ENDTIME for trips (STARTIME + travtime).
    # If the column is missing (e.g. plan passed from Stage 4 directly, or
    # partial Stage 5 failure), reconstruct it from STARTIME + travtime/duration.
    if 'ENDTIME' not in plan.columns or plan['ENDTIME'].isna().all():
        plan = plan.copy()
        if 'STARTIME' in plan.columns:
            tt_source = next(
                (c for c in ['travtime', 'predicted_duration_min', 'tt_avg']
                 if c in plan.columns),
                None
            )
            if tt_source:
                plan['ENDTIME'] = (
                    pd.to_numeric(plan['STARTIME'], errors='coerce').fillna(0)
                    + pd.to_numeric(plan[tt_source], errors='coerce').fillna(10)
                )
                print(f"  NOTE: ENDTIME was missing — created from STARTIME + {tt_source}")
            else:
                plan['ENDTIME'] = pd.to_numeric(
                    plan['STARTIME'], errors='coerce').fillna(0) + 30
                print("  NOTE: ENDTIME was missing — set to STARTIME + 30 min (no travtime found)")
        else:
            plan['ENDTIME'] = 480.0   # 8am fallback
            print("  NOTE: ENDTIME and STARTIME both missing — defaulted to 480 (8am)")

    n_trips_total = int((plan['row_type'] == 'trip').sum())
    n_acts_total  = int((plan['row_type'] == 'activity').sum())
    n_persons     = plan['person_id'].nunique() if 'person_id' in plan.columns else 0
    print(f"\n  Input: {len(plan):,} rows "
          f"({n_trips_total:,} trips, {n_acts_total:,} activities, "
          f"{n_persons:,} persons)")

    # ── chunked processing ─────────────────────────────────────────────────
    # The full plan can be 17M+ rows and multiple plan.copy() calls within
    # validate_and_fix / compute_los / build_aimsun_trips would exhaust memory.
    # We process CHUNK_SIZE persons at a time and stream outputs to CSV files.
    # Each chunk is independently validated, LOS-computed and written out.
    #
    # CHUNK_SIZE = 20000 persons ≈ ~750k rows per chunk → ~500 MB peak.
    # Adjust down to 5000 if the node cgroup is tight.
    CHUNK_SIZE = 20_000

    all_pids = (plan['person_id'].dropna().unique().tolist()
                if 'person_id' in plan.columns else [])
    pid_chunks = [all_pids[i:i + CHUNK_SIZE]
                  for i in range(0, max(len(all_pids), 1), CHUNK_SIZE)]
    n_chunks = max(len(pid_chunks), 1)
    print(f"  Processing {n_persons:,} persons in {n_chunks} chunks "
          f"(chunk size ≤ {CHUNK_SIZE:,})")

    # Output file handles — write header once then append
    stage6_path   = os.path.join(save_dir, 'stage6_output.csv')
    aimsun_path   = os.path.join(save_dir, 'trips_for_aimsun.csv')
    acts_path     = os.path.join(save_dir, 'activities_summary.csv')
    tours_path    = os.path.join(save_dir, 'tours_summary.csv')
    header_written = {'stage6': False, 'aimsun': False,
                      'acts': False, 'tours': False}

    total_fixed = 0
    total_trips_out = 0
    all_aimsun_chunks = []   # kept for motorised splits + vehicle.dat at end

    # [1-5] per-chunk loop
    print("\n[1-5/7] Validating, repairing, computing LOS, attaching coords...")
    for c_idx, pids in enumerate(pid_chunks, 1):
        if pids:
            chunk = plan[plan['person_id'].isin(pids)].copy()
        else:
            chunk = plan.copy()

        n_c = len(chunk)
        n_p = chunk['person_id'].nunique() if 'person_id' in chunk.columns else '?'
        print(f"  Chunk {c_idx:3d}/{n_chunks} — {n_p} persons, {n_c:,} rows  ", end='', flush=True)

        # 1. validate
        chunk = validate_and_fix(chunk, skim_df=skim_df, verbose=False)
        n_fixed = int(chunk.get('_fixed', pd.Series(False, index=chunk.index)).sum())
        total_fixed += n_fixed

        # 2. repair dwells (skip repair if models unavailable — too expensive per-chunk)
        if repair_dwells and stage5_models is not None and skims is not None:
            sk_lookup = skim_lookup or {}
            try:
                chunk = repair_compressed_dwells(
                    plan           = chunk,
                    stage5_models  = stage5_models,
                    persons        = persons,
                    landuse        = landuse if landuse is not None else pd.DataFrame(),
                    skims          = skims,
                    skim_lookup    = sk_lookup,
                    max_iterations = max_repair_iterations,
                    verbose        = False,
                )
            except Exception as _re:
                logger.warning(f"Chunk {c_idx} repair failed ({_re}) — skipping repair")

        # 3. LOS metrics
        chunk = compute_los(chunk, skim_df=skim_df)

        # 4. build output tables
        c_trips   = build_aimsun_trips(chunk)
        c_acts    = build_activities_summary(chunk)
        c_tours   = aggregate_to_tours(c_trips)
        total_trips_out += len(c_trips)

        # 5. attach coordinates
        if sa2_geo is not None:
            c_trips = attach_location_coords(c_trips, sa2_geo=sa2_geo)
            trip_mask = chunk['row_type'] == 'trip'
            if trip_mask.any():
                fp_t = attach_location_coords(chunk[trip_mask].copy(), sa2_geo=sa2_geo)
                for _col in ['origin_lat','origin_lon','dest_lat','dest_lon']:
                    if _col in fp_t.columns:
                        chunk.loc[trip_mask, _col] = fp_t[_col].values

        # stream to CSV (header on first chunk only)
        chunk.to_csv(stage6_path, mode='w' if not header_written['stage6'] else 'a',
                     header=not header_written['stage6'], index=False)
        c_trips.to_csv(aimsun_path, mode='w' if not header_written['aimsun'] else 'a',
                       header=not header_written['aimsun'], index=False)
        c_acts.to_csv(acts_path,   mode='w' if not header_written['acts'] else 'a',
                      header=not header_written['acts'],  index=False)
        c_tours.to_csv(tours_path, mode='w' if not header_written['tours'] else 'a',
                       header=not header_written['tours'], index=False)
        for k in header_written: header_written[k] = True

        all_aimsun_chunks.append(c_trips)
        print(f"✓  ({len(c_trips):,} trip rows written)", flush=True)

        # free chunk memory explicitly
        del chunk, c_trips, c_acts, c_tours

    # ── 6. motorised splits + DynusT ──────────────────────────────────────
    print(f"\n[6/7] Saving motorised splits and DynusT vehicle.dat...")
    trips_aimsun_all = pd.concat(all_aimsun_chunks, ignore_index=True)
    del all_aimsun_chunks

    motorised_splits = save_motorised_trips(trips_aimsun_all, save_dir)
    if dynust_zone_crosswalk:
        motorised_splits = apply_dynust_zone_crosswalk(motorised_splits, dynust_zone_crosswalk)
    write_dynus_vehicle_dat(
        motorised_splits,
        save_dir        = save_dir,
        origin_link_map = origin_link_map,
    )
    write_dynus_od_dat(
        trips    = trips_aimsun_all,
        save_dir = save_dir,
    )

    # ── destination zone validation metrics ───────────────────────────────────
    zone_metrics = compute_zone_validation_metrics(
        plan    = pd.read_csv(stage6_path, nrows=500_000),  # sample for speed
        verbose = verbose,
    )
    if zone_metrics:
        # flatten catchall_by_mode dict for CSV
        flat = {k: v for k, v in zone_metrics.items() if k != 'catchall_by_mode'}
        for mode, share in zone_metrics.get('catchall_by_mode', {}).items():
            flat[f'catchall_pct_{mode}'] = share
        zone_metrics_path = os.path.join(save_dir, 'zone_validation_metrics.csv')
        pd.DataFrame([flat]).to_csv(zone_metrics_path, index=False)
        print(f"  zone_validation_metrics.csv → {zone_metrics_path}")

    print(f"\n  Stage 6 complete.")
    print(f"  stage6_output.csv        ({n_persons:,} persons, {n_trips_total:,} trips)")
    print(f"  trips_for_aimsun.csv     ({total_trips_out:,} trip rows)")
    print(f"  activities_summary.csv   → {acts_path}")
    print(f"  tours_summary.csv        → {tours_path}")
    print(f"  vehicle.dat              → {save_dir}/vehicle.dat")
    print(f"  od.dat                   → {save_dir}/od.dat")
    print(f"  Rows fixed by validator  : {total_fixed:,}")
    if sa2_geo is not None:
        n_coords = int(trips_aimsun_all['origin_lat'].notna().sum())
        print(f"  Trips with coordinates   : {n_coords:,}/{total_trips_out:,}")

    # ── 7. Per-zone output folders ─────────────────────────────────────────
    # Write one sub-folder per origin zone (OTAZ) so downstream tools can
    # load a single zone's worth of data without reading the full population.
    #
    # Layout:
    #   {save_dir}/zones/zone_{otaz}/stage6_output.csv    — full plan rows
    #   {save_dir}/zones/zone_{otaz}/trips_for_aimsun.csv — Aimsun trip rows
    #
    # Both files are written in a single pass over the already-saved flat
    # CSVs so no extra memory beyond one zone chunk at a time is required.
    print(f"\n[7/7] Writing per-zone output folders...")
    zones_root = os.path.join(save_dir, 'zones')
    os.makedirs(zones_root, exist_ok=True)

    try:
        # Read the full plan and aimsun tables back from disk (already streamed
        # there above) and group by OTAZ without holding both in RAM together.
        _pz_plan   = pd.read_csv(stage6_path,  low_memory=False)
        _pz_aimsun = pd.read_csv(aimsun_path,  low_memory=False)

        # Ensure OTAZ column is present in both tables
        for _df in (_pz_plan, _pz_aimsun):
            if 'OTAZ' not in _df.columns:
                _df['OTAZ'] = 0

        _pz_plan['OTAZ']   = pd.to_numeric(_pz_plan['OTAZ'],   errors='coerce').fillna(0).astype(int)
        _pz_aimsun['OTAZ'] = pd.to_numeric(_pz_aimsun['OTAZ'], errors='coerce').fillna(0).astype(int)

        # Build a person_id -> OTAZ map from plan rows that already have OTAZ set
        # so we can fill any activity rows (row_type='activity') that lack OTAZ.
        if 'person_id' in _pz_plan.columns:
            _pid_otaz = (
                _pz_plan[_pz_plan['OTAZ'] > 0][['person_id', 'OTAZ']]
                .drop_duplicates('person_id')
                .set_index('person_id')['OTAZ']
                .to_dict()
            )
            _zero_mask = _pz_plan['OTAZ'] == 0
            if _zero_mask.any():
                _pz_plan.loc[_zero_mask, 'OTAZ'] = (
                    _pz_plan.loc[_zero_mask, 'person_id']
                    .map(_pid_otaz)
                    .fillna(0)
                    .astype(int)
                )

        all_otaz = sorted(_pz_plan['OTAZ'].unique().tolist())
        n_zones_written = 0

        for _otaz in all_otaz:
            _zone_dir = os.path.join(zones_root, f'zone_{_otaz}')
            os.makedirs(_zone_dir, exist_ok=True)

            # Plan rows — all row types for persons originating from this zone
            _plan_slice = _pz_plan[_pz_plan['OTAZ'] == _otaz]
            _plan_slice.to_csv(os.path.join(_zone_dir, 'stage6_output.csv'), index=False)

            # Aimsun trip rows originating from this zone
            _aimsun_slice = _pz_aimsun[_pz_aimsun['OTAZ'] == _otaz]
            _aimsun_slice.to_csv(os.path.join(_zone_dir, 'trips_for_aimsun.csv'), index=False)

            n_zones_written += 1

        del _pz_plan, _pz_aimsun
        print(f"  Per-zone folders written: {n_zones_written} zones -> {zones_root}/")

    except Exception as _ze:
        logger.warning(f"Per-zone split failed ({_ze}) — flat CSVs still valid")
        import traceback as _tb
        _tb.print_exc()

    # Return lightweight summary objects (not the full 17M-row plan)
    # Load back from CSV only if caller needs the full plan in memory
    final_plan   = pd.read_csv(stage6_path)
    trips_aimsun = trips_aimsun_all
    del trips_aimsun_all

    print_plan_summary(final_plan, trips_aimsun)
    return final_plan, trips_aimsun


# ---------------------------------------------------------------------------
# STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------
# Usage:
#   python stage6_plan_finalisation.py                         # resume from saved_data/stage5_output.csv
#   python stage6_plan_finalisation.py --fresh                 # fresh: re-run Stages 1-5 first
#   python stage6_plan_finalisation.py --input path/to/s5.csv  # explicit Stage 5 input
#   python stage6_plan_finalisation.py --skims path/to/matrices.omx
#   python stage6_plan_finalisation.py --no-repair             # skip destination repair loop
#
# Outputs are written to saved_data/ (stage6_output.csv, trips_for_aimsun.csv,
# vehicle.dat, etc). See run_stage6() docstring for the full list.

if __name__ == '__main__':
    import sys
    import os
    import logging
    import pickle
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('stage6_main')

    _OUT        = os.environ.get('PIPELINE_OUT_DIR', 'saved_data')
    S5_CSV      = os.path.join(_OUT, 'stage5_output.csv')
    S5_MODELS   = os.path.join(_OUT, 'stage5_models.pkl')
    SKIMS_PATH  = 'downcasted_matrices_base.omx'

    args       = sys.argv[1:]
    fresh      = '--fresh' in args
    no_repair  = '--no-repair' in args
    args       = [a for a in args if a not in ('--fresh', '--no-repair')]

    input_csv = S5_CSV
    if '--input' in args:
        idx = args.index('--input')
        input_csv = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if '--skims' in args:
        idx = args.index('--skims')
        SKIMS_PATH = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    zone_arg = args[0] if args else None

    # ── load HHTS + landuse ───────────────────────────────────────────────
    log.info("Loading HHTS + landuse...")
    from data_connect import Synth_Connect
    from compute_accessibility import run_accessibility
    sc      = Synth_Connect()
    out     = sc.process_survey(save=False)
    landuse = run_accessibility(
        sc.get_land(),
        skims=SKIMS_PATH,
        save_path='saved_data/accessibility.csv',
    )
    log.info(f"  HHTS: {len(out['persons'])} persons")

    if fresh:
        log.info("Fresh mode: running Stages 1-5 first...")
        import subprocess, sys as _sys
        cmd = [_sys.executable, 'stage1_activity_frequency.py', '--fresh']
        if zone_arg:
            cmd.append(zone_arg)
        subprocess.run(cmd, check=True)
        for s in ('stage2_day_duration.py', 'stage3_activity_duration.py',
                  'stage4_schedule_placement.py', 'stage5_destination_choice.py'):
            subprocess.run([_sys.executable, s], check=True)

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Stage 5 output not found at '{input_csv}'.\n"
            f"  Run stage5_destination_choice.py first, or use --fresh."
        )
    log.info(f"Loading Stage 5 output from {input_csv} ...")
    plan = pd.read_csv(input_csv)
    log.info(f"  {len(plan):,} rows loaded")

    # ── load Stage 5 models for repair loop ───────────────────────────────
    stage5_models = None
    skims         = None
    skim_lookup   = None
    if not no_repair and os.path.isfile(S5_MODELS):
        log.info(f"Loading Stage 5 models from {S5_MODELS} for repair loop...")
        with open(S5_MODELS, 'rb') as _f:
            stage5_models = pickle.load(_f)
        import openmatrix as omx
        from stage5_destination_choice import _skim_flat, _skim_lookup
        skims      = omx.open_file(SKIMS_PATH, 'r')
        skim_df    = _skim_flat(skims)
        skim_lookup = _skim_lookup(skim_df)
    elif no_repair:
        log.info("  --no-repair: destination repair loop disabled")
    else:
        log.warning(f"  {S5_MODELS} not found — destination repair loop disabled")

    # ── zone centroids for coordinate pinning ─────────────────────────────
    sa2_geo = None
    try:
        from spatial_assignment import get_zone_centroids_rich
        sa2_geo = get_zone_centroids_rich(landuse, cache_dir='saved_data/osm_cache')
        sa2_geo = sa2_geo[['ORDER', 'lat', 'lon']].copy()
        log.info(f"  Zone centroids loaded: {len(sa2_geo)} zones")
    except Exception as e:
        log.warning(f"  Zone centroids unavailable ({e}) — skipping coord pinning")

    # ── run Stage 6 ───────────────────────────────────────────────────────
    os.makedirs(_OUT, exist_ok=True)
    final_plan, trips_aimsun = run_stage6(
        plan           = plan,
        persons        = out['persons'],
        landuse        = landuse,
        stage5_models  = stage5_models,
        skims          = skims,
        skim_lookup    = skim_lookup,
        sa2_geo        = sa2_geo,
        repair_dwells  = (stage5_models is not None),
    )

    if skims:
        skims.close()

    log.info("=" * 55)
    log.info("Stage 6 complete.")
    log.info(f"  {_OUT}/stage6_output.csv")
    log.info(f"  {_OUT}/trips_for_aimsun.csv")
    log.info(f"  {_OUT}/vehicle.dat")
    log.info("Run spatial_assignment.py for spatial plots.")
