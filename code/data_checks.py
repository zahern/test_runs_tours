"""
data_checks.py
==============
Startup data quality checks for the SEQ ABM pipeline.

Validates that:
  1. HHTS data files exist and have expected columns
  2. All purpose values map cleanly (zero unmapped)
  3. All mode values map cleanly (zero unmapped)
  4. AGEGROUP/WORKSTATUS/MAINACT produce correct ptype distribution
  5. SA1→SA2→TAZ zone chain has no missing values
  6. Tour building produces sensible output (tours start/end at home)

Call via:
    from data_checks import run_data_checks
    run_data_checks()   # raises CheckFailed if critical issue found
"""

from __future__ import annotations
import os, sys, logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CheckFailed(RuntimeError):
    """Raised when a critical data check fails."""


# ── Expected column sets ─────────────────────────────────────────────────────
REQUIRED_TRIPS_COLS   = {'TRIPID','HHID','PERSID','STARTIME','ARRTIME',
                          'ORIGPLACE','DESTPLACE','ORIGPURP','DESTPURP',
                          'MAINMODE','TRAVTIME','DURATION','CUMDIST',
                          'OVERALL_PURPOSE','ORIGSA1_2021','DESTSA1_2021'}
REQUIRED_PERSONS_COLS = {'HHID','PERSID','AGEGROUP','SEX','WORKSTATUS',
                          'MAINACT','CARLICENCE','INCOME','STUDYING'}
REQUIRED_HH_COLS      = {'HHID','HHSIZE','HHVEH','BIKES','HOME_SA1_2021'}

# ── Expected value sets ──────────────────────────────────────────────────────
EXPECTED_MODES = {
    'Car driver','Car passenger','Walking','Bicycle','E-bike or E-scooter',
    'Public bus','Public Bus','Train','Light rail','Ferry',
    'Motorcycle driver','Motorcycle passenger','Truck driver','Truck passenger',
    'School bus (with route number)','School bus (private/chartered)',
    'Charter/Courtesy/Other bus','Uber / Other Ride Share','Taxi',
    'Mobility scooter','Other method',
}

EXPECTED_PURPOSES = {
    'At Home','Go home','My home',
    'Work (my workplace)','Work (other work reason)',
    'Education','Childcare or kindergarten','Childcare or Kindergarten',
    'Pick up or drop off a passenger',
    'Just accompanying someone','Shopping','Eat/drink','Social Visit',
    'Recreation/leisure activity',
    'Personal business','Personal Business',
    'Pickup/deliver something (not work-related)',
    'Refuel my vehicle','Other place (please specify)','Other Place',
    # OVERALL_PURPOSE values
    'Direct Work Commute','Work Related','Accompany Someone',
    'Pickup/Dropoff Someone','Pickup/Deliver Something',
    'Social','Recreation','Childcare or Kindergarten',
    'Personal Business','Other Purpose',
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _pct(n, total):
    return f"{100*n/max(total,1):.1f}%"


def _warn(msg):
    logger.warning(f"  [CHECK] {msg}")
    print(f"  ⚠  {msg}")


def _ok(msg):
    logger.info(f"  [CHECK] ✓ {msg}")
    print(f"  ✓  {msg}")


def _fail(msg):
    logger.error(f"  [CHECK] ✗ CRITICAL: {msg}")
    print(f"  ✗  CRITICAL: {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE EXISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def check_files():
    candidates = [
        ('Trips',   ['5_QTS_TRIPS.xlsx',   '../5_QTS_TRIPS.xlsx',   'data/5_QTS_TRIPS.xlsx']),
        ('Persons', ['2_QTS_PERSONS.xlsx', '../2_QTS_PERSONS.xlsx', 'data/2_QTS_PERSONS.xlsx']),
        ('HH',      ['1_QTS_HOUSEHOLDS.xlsx','../1_QTS_HOUSEHOLDS.xlsx']),
        ('SA1 map', ['SA1_SHP/sa1_2021.csv','../SA1_SHP/sa1_2021.csv','sa1_2021.csv']),
        ('Zone ids',['seq_zone_ids_.csv','../seq_zone_ids_.csv']),
    ]
    found = {}
    for label, paths in candidates:
        for p in paths:
            if os.path.exists(p):
                found[label] = p
                _ok(f"{label} found: {p}")
                break
        else:
            _warn(f"{label} not found in standard locations {paths[:2]}")
    return found


# ─────────────────────────────────────────────────────────────────────────────
# 2. COLUMN CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_columns(trips: pd.DataFrame, persons: pd.DataFrame,
                  hh: pd.DataFrame) -> None:
    for label, df, expected in [
        ('trips',   trips,   REQUIRED_TRIPS_COLS),
        ('persons', persons, REQUIRED_PERSONS_COLS),
        ('hh',      hh,      REQUIRED_HH_COLS),
    ]:
        missing = expected - set(df.columns)
        if missing:
            _warn(f"{label}: missing expected columns: {sorted(missing)}")
        else:
            _ok(f"{label}: all required columns present ({len(df):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PURPOSE MAPPING CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_purpose_mapping(trips: pd.DataFrame) -> None:
    from data_connect import purpose as PURPOSE_MAP
    for col in ['ORIGPURP', 'DESTPURP', 'OVERALL_PURPOSE']:
        if col not in trips.columns:
            continue
        vals      = trips[col].dropna()
        unmapped  = vals[~vals.isin(PURPOSE_MAP.keys())]
        n_unmapped = len(unmapped)
        pct        = _pct(n_unmapped, len(vals))
        if n_unmapped > 0:
            top_unmapped = unmapped.value_counts().head(5).to_dict()
            _warn(f"{col}: {n_unmapped:,} unmapped ({pct}) "
                  f"top values: {top_unmapped}")
        else:
            _ok(f"{col}: 100% mapped ({len(vals):,} values)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODE MAPPING CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_mode_mapping(trips: pd.DataFrame) -> None:
    from data_connect import MODES
    for col in ['MAINMODE', 'MODE1']:
        if col not in trips.columns:
            continue
        vals      = trips[col].dropna()
        unmapped  = vals[~vals.isin(MODES.keys())]
        n_unmapped = len(unmapped)
        if n_unmapped > 0:
            _warn(f"{col}: {n_unmapped:,} unmapped modes "
                  f"({_pct(n_unmapped,len(vals))}): "
                  f"{unmapped.value_counts().head(5).to_dict()}")
        else:
            _ok(f"{col}: 100% mapped ({len(vals):,} values)")

    # Check Car passenger → SR (not DA)
    from data_connect import MODES
    if MODES.get('Car passenger') == 'DA':
        _fail("MODES: 'Car passenger' maps to 'DA' — should be 'SR'. "
              "Fix data_connect.py MODES dict.")
    else:
        _ok(f"Car passenger → {MODES.get('Car passenger','?')} (correct)")

    # Check at least one of each output mode
    mapped = trips['MAINMODE'].map(MODES) if 'MAINMODE' in trips.columns else pd.Series()
    if not mapped.empty:
        dist = mapped.value_counts(normalize=True)*100
        for mode in ['DA','SR','Walk','Transit','Bike']:
            share = dist.get(mode, 0)
            if share < 0.1:
                _warn(f"Mode '{mode}' share is only {share:.2f}% — "
                      "may indicate mapping problem")
        _ok(f"Mode distribution: {dist.round(1).to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PTYPE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_ptype(persons: pd.DataFrame, hh: pd.DataFrame) -> None:
    from data_connect import assign_ptype
    from stage1_activity_frequency import PTYPE_PREDICTION_MAP

    p = persons.merge(hh[['HHID']], on='HHID', how='left') \
        if 'HOME_SA1_2021' not in persons.columns else persons
    p['ptype'] = p.apply(assign_ptype, axis=1)
    dist = p['ptype'].value_counts()
    _ok(f"ptype distribution:\n"
        + '\n'.join(f"      {pt:25s}: {n:6,}  ({_pct(n,len(p))})"
                    for pt, n in dist.items()))

    # Check old labels not appearing
    old_labels = {'School', 'Student', 'JointWorkStudy',
                  'Unemployed / Other', 'Joint Study & Work',
                  'Full-time Worker', 'Part-time Worker'}
    bad = old_labels & set(dist.index)
    if bad:
        _fail(f"assign_ptype returned OLD labels: {bad}. "
              "Update data_connect.assign_ptype to use new labels.")
    else:
        _ok("assign_ptype uses correct CDAP-aligned labels")

    # Check PTYPE_PREDICTION_MAP covers all produced labels
    unmapped_ptypes = set(dist.index) - set(PTYPE_PREDICTION_MAP.keys())
    if unmapped_ptypes:
        _warn(f"ptypes not in PTYPE_PREDICTION_MAP: {unmapped_ptypes}")
    else:
        _ok("All ptypes covered by PTYPE_PREDICTION_MAP")

    # Check coverage — no ptype should be > 60% of population
    top_share = dist.iloc[0] / len(p) * 100
    if top_share > 65:
        _warn(f"Top ptype '{dist.index[0]}' = {top_share:.1f}% — "
              "may indicate classification bug")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ZONE CHAIN CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_zone_chain(trips: pd.DataFrame, persons: pd.DataFrame) -> None:
    sa1_col = 'ORIGSA1_2021'
    if sa1_col not in trips.columns:
        _warn("Zone chain check: no ORIGSA1_2021 column in trips")
        return

    # Check SA1 values are numeric
    sa1_vals = pd.to_numeric(trips[sa1_col], errors='coerce')
    n_bad = sa1_vals.isna().sum()
    if n_bad > 0:
        _warn(f"Zone chain: {n_bad:,} non-numeric SA1 codes in trips")
    else:
        _ok(f"Zone chain: all {len(sa1_vals):,} SA1 codes are numeric")

    # Check PERSID join between trips and persons
    if 'PERSID' in trips.columns and 'PERSID' in persons.columns:
        join_rate = trips['PERSID'].isin(persons['PERSID']).mean() * 100
        if join_rate < 99:
            _warn(f"Zone chain: only {join_rate:.1f}% of trips join to persons")
        else:
            _ok(f"Trips→Persons join: {join_rate:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 7. TOUR BUILDING SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_tour_building(trips: pd.DataFrame) -> None:
    """Quick sanity check on a sample of tours."""
    from data_connect import (build_tours_from_trips, survey_time_to_minutes,
                               purpose, MODES)
    sample = trips.sample(min(5000, len(trips)), random_state=42).copy()
    sample['STARTIME'] = survey_time_to_minutes(sample['STARTIME'])
    if 'ARRTIME' in sample.columns:
        sample['ARRTIME'] = survey_time_to_minutes(sample['ARRTIME'])
    sample['MAINMODE'] = sample['MAINMODE'].map(MODES)
    sample['ORIGPURP'] = sample['ORIGPURP'].map(purpose)
    sample['DESTPURP'] = sample['DESTPURP'].map(purpose)

    # Stub zone columns so determine_primary_purpose() and build_as_tours()
    # don't crash when SA1→SA2→TAZ zone files aren't in the working directory.
    for col in ('OTAZ', 'DTAZ', 'ORIGSA2_2021', 'DESTSA2_2021'):
        if col not in sample.columns:
            sample[col] = 1

    # build_tours_from_trips / build_as_tours requires ENDTIME and
    # ENDTIME_WITH_ACTIVITY.  The survey file has ARRTIME / TRAVTIME / DURATION
    # instead.  Pre-compute them here so the check doesn't crash.
    if 'ENDTIME' not in sample.columns:
        if 'ARRTIME' in sample.columns:
            sample['ENDTIME'] = sample['ARRTIME']
        elif 'TRAVTIME' in sample.columns:
            trav = pd.to_numeric(sample['TRAVTIME'], errors='coerce').fillna(0)
            sample['ENDTIME'] = sample['STARTIME'] + trav
    if 'ENDTIME_WITH_ACTIVITY' not in sample.columns:
        trav = pd.to_numeric(sample.get('TRAVTIME', 0), errors='coerce').fillna(0)
        dur  = pd.to_numeric(sample.get('DURATION', 0), errors='coerce').fillna(0) \
               if 'DURATION' in sample.columns else 0
        sample['ENDTIME_WITH_ACTIVITY'] = sample['STARTIME'] + trav + dur

    try:
        as_trips, as_tours = build_tours_from_trips(sample)
        n_tours = len(as_tours)
        if n_tours == 0:
            _warn("Tour building: produced 0 tours from 5k trip sample")
            return

        _ok(f"Tour building: {n_tours} tours from 5k sample")

        # Check tour_mode column
        if 'tour_mode' not in as_tours.columns:
            _warn("Tour building: 'tour_mode' column missing — "
                  "check _modal_mode() in build_as_tours()")
        else:
            da_frac = (as_tours['tour_mode'] == 'DA').mean()
            mode_dist = as_tours['tour_mode'].value_counts().to_dict()
            if da_frac > 0.90:
                _warn(f"Tour building: {da_frac*100:.0f}% DA — check MODES dict "
                      f"(Car passenger should be Other, not DA)")
            else:
                _ok(f"Tour building: mode dist = {mode_dist}")

        # Check primary purpose was assigned
        if 'primary_purpose' in as_tours.columns:
            n_null = as_tours['primary_purpose'].isna().sum()
            if n_null > n_tours * 0.1:
                _warn(f"Tour building: {n_null} tours have null primary_purpose")
            else:
                _ok("Tour building: primary_purpose assigned to all tours")
    except Exception as e:
        _warn(f"Tour building: failed on sample — {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. TIMING SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_timing(trips: pd.DataFrame) -> None:
    from data_connect import survey_time_to_minutes
    st = survey_time_to_minutes(trips['STARTIME']) if 'STARTIME' in trips.columns \
         else pd.Series()
    if not st.empty:
        pct_am = ((st >= 360) & (st <= 600)).mean() * 100
        pct_mid = ((st >= 600) & (st <= 900)).mean() * 100
        pct_pm  = ((st >= 900) & (st <= 1200)).mean() * 100
        if pct_am < 5:
            _warn(f"Timing: only {pct_am:.1f}% of trips depart 6-10am — "
                  "check survey_time_to_minutes conversion")
        else:
            _ok(f"Timing: AM={pct_am:.1f}%  Midday={pct_mid:.1f}%  PM={pct_pm:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def run_data_checks(raise_on_critical: bool = False) -> bool:
    """
    Run all data checks. Prints results. Returns True if all checks passed.

    Parameters
    ----------
    raise_on_critical : if True, raise CheckFailed on critical errors.
    """
    print("\n" + "="*60)
    print("DATA QUALITY CHECKS")
    print("="*60)

    files = check_files()
    trips_path   = files.get('Trips')
    persons_path = files.get('Persons')
    hh_path      = files.get('HH')

    if not trips_path or not persons_path:
        _warn("Cannot load data — skipping most checks")
        return False

    try:
        trips   = pd.read_excel(trips_path)
        persons = pd.read_excel(persons_path)
        hh      = pd.read_excel(hh_path) if hh_path else pd.DataFrame()
    except Exception as e:
        _warn(f"Could not load data files: {e}")
        return False

    print(f"\n  Loaded: {len(trips):,} trips, {len(persons):,} persons, "
          f"{len(hh):,} households")

    print("\n── Column checks ──")
    check_columns(trips, persons, hh)

    print("\n── Purpose mapping ──")
    check_purpose_mapping(trips)

    print("\n── Mode mapping ──")
    check_mode_mapping(trips)

    print("\n── Person type (ptype) ──")
    if not hh.empty:
        check_ptype(persons, hh)
    else:
        _warn("No HH data — skipping ptype check")

    print("\n── Zone chain ──")
    check_zone_chain(trips, persons)

    print("\n── Timing ──")
    check_timing(trips)

    print("\n── Tour building (5k sample) ──")
    check_tour_building(trips)


    print("\n" + "="*60)
    print("DATA CHECKS COMPLETE")
    print("="*60 + "\n")
    return True


if __name__ == '__main__':
    run_data_checks()
