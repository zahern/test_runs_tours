"""
build_observed_plans.py
=======================
Reconstruct OBSERVED daily activity plans from HHTS stop records so the paper
can show observed-vs-predicted on identical footing.

Input : trips_stops_merged.csv (stop-level HHTS: tour_id/STOPNO/OTAZ/DTAZ/MODE/
        STARTIME/ARRTIME/DURATION/DESTPURP per PERSID) + person_ss.csv
Output: observed_plans.csv in the EXACT schema plot_activity_plans.py expects
        (row_type/person_id/purpose/STARTIME/ENDTIME/mode/mode_name/OTAZ/DTAZ/
        ptype/vehrd, minutes from midnight, mode codes 0=dwell 1=DA 2=SR
        3=Walk 4=Bike 5=Transit) + plot_all() figures.

Mapping provenance (kept identical to estimation so observed-vs-predicted is
like-for-like):
  modes    : stage6_mode_choice.RAW_MODE_MAP (fallback TOURMODE 1..5 =
             DA/SR/Walk/Bike/Transit per _LARCH_MODE_IDS)
  purposes : ga_hypothesis_testing._HHTS_TRIP_PURPOSE_MAP ('Go home' -> 'home';
             PT-access/refuel legs inherit the next activity purpose in-tour)

Home dwell is NOT emitted as rows (plot_activity_plans infers home as the
complement of trips + non-home activities; emitting it would double-draw).

Usage:
  python build_observed_plans.py [--trips trips_stops_merged.csv]
      [--persons person_ss.csv] [--outdir observed_plans] [--plots 1]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Mode map: canonical source, embedded fallback ────────────────────────────
try:
    from stage6_mode_choice import RAW_MODE_MAP as _RAW_MODE_MAP
    OBSERVED_MODE_MAP = dict(_RAW_MODE_MAP)
except Exception:
    OBSERVED_MODE_MAP = {
        'Car driver': 'DA', 'Motorcycle driver': 'DA', 'Truck driver': 'DA',
        'Car passenger': 'SR', 'Motorcycle passenger': 'SR',
        'Truck passenger': 'SR', 'Public Bus': 'Transit', 'Public bus': 'Transit',
        'Train': 'Transit', 'Light rail': 'Transit', 'Ferry': 'Transit',
        'Charter/Courtesy/Other bus': 'Transit',
        'School bus (with route number)': 'Transit',
        'School bus (private/chartered)': 'Transit',
        'Walking': 'Walk', 'Bicycle': 'Bike', 'E-bike or E-scooter': 'Bike',
        'Uber / Other Ride Share': 'SR', 'Taxi': 'SR',
        'Mobility scooter': 'Walk', 'Other method': 'DA',
        'DA': 'DA', 'SR': 'SR', 'Transit': 'Transit', 'Walk': 'Walk', 'Bike': 'Bike',
    }
MODE_NAME_TO_INT = {'DA': 1, 'SR': 2, 'Walk': 3, 'Bike': 4, 'Transit': 5}
TOURMODE_TO_NAME = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}

# ── Purpose map: mirror of ga_hypothesis_testing._HHTS_TRIP_PURPOSE_MAP ──────
DESTPURP_TO_PURPOSE = {
    'Work (my workplace)': 'work', 'Work (other work reason)': 'work',
    'Direct Work Commute': 'work', 'Work Related': 'work',
    'Education': 'edu', 'Childcare or Kindergarten': 'edu',
    'Childcare or kindergarten': 'edu',
    'Social Visit': 'social', 'Recreation/leisure activity': 'social',
    'Recreation': 'social', 'Social': 'social',
    'Pick up or drop off a passenger': 'escort',
    'Pickup/Dropoff Someone': 'escort',
    'Just accompanying someone': 'escort',
    'Shopping': 'shopping',
    'Personal Business': 'othmaint', 'Personal business': 'othmaint',
    'Pickup/deliver something (not work-related)': 'othmaint',
    'Other place (please specify)': 'othdiscr',
    'Other Place': 'othdiscr', 'Refuel my vehicle': 'othdiscr',
    'Eat/drink': 'eat',
}
HOME_PURPS = {'Go home', 'At Home', 'At home', 'Home'}
ACCESS_LIKE = {'Get on/off public transport', 'Get on/off public transport '}  # en-route legs


def _map_purpose(destpurp: str) -> str:
    s = str(destpurp or '').strip()
    if s in HOME_PURPS:
        return 'home'
    if s in DESTPURP_TO_PURPOSE:
        return DESTPURP_TO_PURPOSE[s]
    if s in ACCESS_LIKE:
        return 'travel'  # backfilled to next activity purpose in-tour below
    return 'travel'


def _map_mode(mode_str, tourmode) -> str:
    s = str(mode_str or '').strip()
    if s in OBSERVED_MODE_MAP:
        return OBSERVED_MODE_MAP[s]
    try:
        return TOURMODE_TO_NAME.get(int(float(tourmode)), '')
    except Exception:
        return ''


def build_observed_plans(trips_csv: str, persons_csv: str | None = None,
                         ) -> tuple[pd.DataFrame, dict]:
    """Return (plan DataFrame in plot schema, diagnostics dict)."""
    stops = pd.read_csv(trips_csv, low_memory=False)
    stops = stops.sort_values(
        ['PERSID', 'tour_id', 'STOPNO']).reset_index(drop=True)

    # ── person attributes (same ptype logic as estimation) ───────────────────
    ptype_map, veh_map = {}, {}
    if persons_csv and os.path.isfile(persons_csv):
        try:
            from cdap import build_person_profiles
            per = pd.read_csv(persons_csv, low_memory=False)
            if 'HHVEH' in per.columns and 'vehrd' not in per.columns:
                per['vehrd'] = pd.to_numeric(
                    per['HHVEH'], errors='coerce').fillna(1)
            if 'HHSIZE' in per.columns and 'nprd' not in per.columns:
                per['nprd'] = pd.to_numeric(
                    per['HHSIZE'], errors='coerce').fillna(2)
            prof = build_person_profiles(per)
            ptype_map = dict(zip(prof['PERSID'].astype(str),
                                 prof['ptype'].astype(str)))
            if 'vehrd' in prof.columns:
                veh_map = dict(zip(prof['PERSID'].astype(str),
                                   pd.to_numeric(prof['vehrd'],
                                                 errors='coerce').fillna(0)))
        except Exception as e:
            print(f"  person-profile attach skipped ({e})")

    trip_rows, act_rows = [], []
    n_bad_time, n_unmapped_mode = 0, 0
    for (pid, tour), g in stops.groupby(['PERSID', 'tour_id'], sort=False):
        g = g.sort_values('STOPNO')
        for _, r in g.iterrows():
            try:
                t0 = float(r['STARTIME'])
                t1 = float(r['ARRTIME'])
            except Exception:
                n_bad_time += 1
                continue
            if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
                n_bad_time += 1
                continue
            purp = _map_purpose(r.get('DESTPURP', ''))
            mname = _map_mode(r.get('MODE', r.get('MAINMODE', '')),
                              r.get('TOURMODE', r.get('tourmode', '')))
            if not mname:
                n_unmapped_mode += 1
            trip_rows.append({
                'row_type': 'trip', 'person_id': pid, 'purpose': purp,
                'STARTIME': t0, 'ENDTIME': t1,
                'mode': MODE_NAME_TO_INT.get(mname, 0), 'mode_name': mname,
                'OTAZ': r.get('OTAZ', ''), 'DTAZ': r.get('DTAZ', ''),
                'travtime': r.get('TRAVTIME', t1 - t0),
                'dist': r.get('DISTANCE', np.nan),
            })
            # Activity at the destination (home arrivals left implicit —
            # the plot infers home dwell as the complement; access-leg dwells
            # are wait time, kept inside the trip leg, not as activities).
            if purp not in ('home', 'travel'):
                try:
                    dwell = float(r.get('DURATION', 0) or 0)
                except Exception:
                    dwell = 0.0
                if dwell > 0 and np.isfinite(dwell):
                    act_rows.append({
                        'row_type': 'activity', 'person_id': pid,
                        'purpose': purp, 'STARTIME': t1,
                        'ENDTIME': t1 + dwell,
                        'mode': 0, 'mode_name': '',
                        'OTAZ': r.get('DTAZ', ''), 'DTAZ': r.get('DTAZ', ''),
                    })
    plan = pd.DataFrame(trip_rows + act_rows)
    if plan.empty:
        return plan, {'n_persons': 0}

    # ── access/refuel legs inherit the next activity purpose in-person ───────
    # (vectorised: pandas 3 groupby.apply on mutated groups is unreliable)
    plan = plan.sort_values(['person_id', 'STARTIME']).reset_index(drop=True)
    _valid = plan['purpose'].where(
        ~plan['purpose'].isin(['travel', 'home']))
    _nxt = _valid.groupby(plan['person_id']).bfill()
    _is_access = ((plan['row_type'] == 'trip')
                  & (plan['purpose'] == 'travel') & _nxt.notna())
    plan.loc[_is_access, 'purpose'] = _nxt[_is_access]

    plan['ptype'] = plan['person_id'].astype(str).map(ptype_map).fillna('')
    plan['vehrd'] = plan['person_id'].astype(str).map(veh_map).fillna(0)

    diag = {
        'n_persons': int(plan['person_id'].nunique()),
        'n_tours': int(stops['tour_id'].nunique()),
        'n_trips': int((plan['row_type'] == 'trip').sum()),
        'n_activities': int((plan['row_type'] == 'activity').sum()),
        'n_bad_time': int(n_bad_time),
        'n_unmapped_mode': int(n_unmapped_mode),
        'n_travel_purpose_left': int(((plan['row_type'] == 'trip')
                                     & (plan['purpose'] == 'travel')).sum()),
        'mode_share': plan.loc[plan['row_type'] == 'trip', 'mode_name']
        .value_counts().to_dict(),
        'purpose_share': plan.loc[plan['row_type'] == 'activity', 'purpose']
        .value_counts().to_dict(),
    }
    return plan.reset_index(drop=True), diag


def main():
    ap = argparse.ArgumentParser(description="Build observed HHTS plans")
    ap.add_argument('--trips', default='trips_stops_merged.csv')
    ap.add_argument('--persons', default='person_ss.csv')
    ap.add_argument('--outdir', default='observed_plans')
    ap.add_argument('--plots', type=int, default=1)
    ap.add_argument('--sample', type=int, default=8)
    a = ap.parse_args()

    plan, diag = build_observed_plans(a.trips, a.persons)
    os.makedirs(a.outdir, exist_ok=True)
    plan.to_csv(os.path.join(a.outdir, 'observed_plans.csv'), index=False)
    print(f"  Observed plans → {a.outdir}/observed_plans.csv "
          f"({len(plan):,} rows)")
    for k, v in diag.items():
        print(f"    {k}: {v}")
    if a.plots and len(plan):
        from plot_activity_plans import plot_all
        plot_all(plan, n_sample=a.sample, save_dir=a.outdir)


if __name__ == '__main__':
    main()
