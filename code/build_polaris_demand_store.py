#!/usr/bin/env python3
"""Build a POLARIS-style demand SQLite store from SEQ framework outputs.

Combines the synthetic population (synth_population_mapped.csv /
synth_persons_mapped.csv), the predicted schedule (stage6_output.csv) and the
zone reference (seq_zone_ids.csv) into a single SQLite database following the
POLARIS demand data model:

    Household -> Person -> Plan -> Trip -> Activity
    plus Location, Mode and Activity_Type dictionary tables.

Information preservation: every source column that is not consumed by a
canonical POLARIS column is carried over verbatim as an extra column on the
same row (census codes, latent variables, accessibility, GA features, weights,
etc.). Nothing is dropped.

Usage:
  python build_polaris_demand_store.py [options]

Options:
  --population FILE   synth population CSV (default saved_data/synth_population_mapped.csv)
  --persons FILE      synth persons CSV (default saved_data/synth_persons_mapped.csv)
  --stage6 FILE       predicted schedule CSV (default saved_data/stage6_output.csv)
  --zones FILE        zone reference CSV (default seq_zone_ids.csv)
  --out FILE          output SQLite path (default saved_data/Demand.sqlite)
  --popid COL         household id column in the population file (default household_id)
  --pid COL           person id column in the population file (default person_id)
  --run TAG           optional run tag: if stage6/--out are unset, reads
                      runs/ga_parallel/<TAG>/stage6/stage6_output.csv
"""
import argparse
import os
import sqlite3
import sys

import pandas as pd

import polaris_schema as ps

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(msg, flush=True)


def load(path, required=True, chunksize=200000):
    if not os.path.exists(path):
        if required:
            log(f'ERROR: missing input {path}')
            sys.exit(2)
        return None
    parts = []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize,
                                          low_memory=False)):
        parts.append(downcast(chunk))
        if i and i % 5 == 0:
            log(f'  ...{i * chunksize:,} rows loaded')
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    del parts
    log(f'  loaded {path} ({len(df):,} rows x {len(df.columns)} cols)')
    return df


def downcast(df):
    """Shrink dtypes (float64->32, int64->32, low-cardinality object->category).
    Cuts the 1.2 GB synth population to a few hundred MB in RAM."""
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        elif df[c].dtype == 'int64':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('int32')
        elif df[c].dtype == 'object':
            n = df[c].nunique(dropna=True)
            if n <= 128 and n > 0 and n / max(len(df), 1) < 0.1:
                df[c] = df[c].astype('category')
            else:
                df[c] = df[c].astype(str)
    return df


def coerce_int(v):
    try:
        f = float(v)
        if f != f:  # NaN
            return 0
        return int(round(f))
    except (TypeError, ValueError):
        return 0


def write_table(conn, name, df):
    """Write df into table, ALTER ADD COLUMN any extended (non-canonical) cols
    first so no source information is dropped."""
    cur = conn.cursor()
    existing = {r[1].lower(): r[1]
                for r in cur.execute(f'PRAGMA table_info("{name}")').fetchall()}
    seen = set()
    keep = []
    for col in df.columns:
        if col.lower() in seen:
            continue  # drop case-insensitive duplicates
        seen.add(col.lower())
        keep.append(col)
        if col.lower() not in existing:
            if 'int' in str(df[col].dtype):
                ctype = 'INTEGER'
            elif 'float' in str(df[col].dtype):
                ctype = 'REAL'
            else:
                ctype = 'TEXT'
            cur.execute(f'ALTER TABLE "{name}" ADD COLUMN "{col}" {ctype}')
    conn.commit()
    df[keep].to_sql(name, conn, if_exists='append', index=False)
    n = len(df)
    log(f'  wrote {name}: {n:,} rows')
    return n


def build_households(pop, zone_col, person_index):
    """Household table from the synth population (grouped by household).

    The source's household_id is NaN for ~47% of rows; abshid (ABS household
    id) is complete and disjoint, so it fills the gap. family_id is a fallback."""
    pop = pop.copy()
    for k in ('household_id', 'abshid', 'family_id'):
        if k in pop:
            pop[k] = pop[k].astype('float64')
    if 'household_id' in pop:
        pop['_hhid'] = pop['household_id'].fillna(pop['abshid'])
        if pop['_hhid'].isna().any() and 'family_id' in pop:
            pop['_hhid'] = pop['_hhid'].fillna(pop['family_id'])
    elif 'abshid' in pop:
        pop['_hhid'] = pop['abshid']
    else:
        pop['_hhid'] = pop.index.astype('float64')
    pop['_hhid'] = pop['_hhid'].fillna(pop.groupby('person_id', sort=False)['person_id'].transform('first').astype('float64'))
    pop['_hhid'] = pop['_hhid'].astype('int64')
    pop['_loc'] = pop[zone_col].map(person_index).fillna(0).astype('int64')
    g = pop.groupby('_hhid', sort=False)
    rows = []
    for hid, idx in g.groups.items():
        sub = pop.loc[idx]
        mode_loc = sub['_loc'].mode()
        mode_pih = sub['persons_in_household'].mode() if 'persons_in_household' in sub else pd.Series(dtype='float64')
        mode_incp = sub['incp'].mode() if 'incp' in sub else pd.Series(dtype='float64')
        rows.append({
            'household': hid,
            'hhold': hid,
            'location': int(mode_loc.iloc[0]) if len(mode_loc) else 0,
            'persons': int(mode_pih.iloc[0])
                       if len(mode_pih) else int(len(sub)),
            'workers': 0,
            'vehicles': 0,
            'type': 0,
            'income': coerce_int(mode_incp.iloc[0])
                      if len(mode_incp) else 0,
            'bikes': 0,
            'housing_unit_type': 0,
            'ecom': 0,
            'delRat': 0.0,
            'dispose_veh': 0,
            'time_in_home': 0.0,
            'Has_Residential_Charging': 0,
            'num_groceries': 0,
            'num_meals': 0,
        })
    hh = pd.DataFrame(rows)
    hh['household_id'] = hh['household']
    return hh


def build_persons(persons, pop, zone_col, person_index):
    """Person table from synth persons; pulls location cols from stage6 later."""
    df = persons.copy()
    df['_loc'] = df[zone_col].map(person_index).fillna(0).astype('int64')
    df['person'] = df['person_id']
    # rebuild household key exactly as build_households did
    p = pop.copy()
    for k in ('household_id', 'abshid', 'family_id'):
        if k in p:
            p[k] = p[k].astype('float64')
    if 'household_id' in p:
        p['_hhid'] = p['household_id'].fillna(p['abshid'])
        if p['_hhid'].isna().any() and 'family_id' in p:
            p['_hhid'] = p['_hhid'].fillna(p['family_id'])
    elif 'abshid' in p:
        p['_hhid'] = p['abshid']
    else:
        p['_hhid'] = p.index.astype('float64')
    p['_hhid'] = p['_hhid'].fillna(
        p.groupby('person_id', sort=False)['person_id'].transform('first').astype('float64'))
    p['_hhid'] = p['_hhid'].astype('int64')
    hh_map = p.set_index('person_id')['_hhid']
    df['household'] = df['person_id'].map(hh_map).fillna(0).astype('int64')
    df['id'] = df.groupby('household').cumcount()

    df['age'] = df['agep'].map(coerce_int) if 'agep' in df else 0
    df['gender'] = df['sexp'].map({1: 1, 2: 2}).fillna(0).astype(int) \
        if 'sexp' in df else 0
    df['income'] = df['incp'].map(coerce_int) if 'incp' in df else 0
    df['work_hours'] = df['hrsp'].map(coerce_int) if 'hrsp' in df else 0
    # location (home)
    df['school_location_id'] = 0
    df['work_location_id'] = 0
    for c in ('worker_class', 'education', 'industry', 'employment',
              'journey_to_work_arrival_time', 'journey_to_work_mode',
              'journey_to_work_travel_time', 'journey_to_work_vehicle_occupancy',
              'marital_status', 'race', 'school_enrollment', 'school_grade_level',
              'telecommute_level', 'transit_pass', 'time_in_job', 'disability',
              'escooter_use_level', 'is_long_term_chooser'):
        if c not in df:
            df[c] = 0
    df = df.rename(columns={'_loc': 'location_home'})
    # drop the raw id column to avoid clobbering canonical 'id'
    for c in df.columns:
        if c == 'person_id':
            continue
    return df


def build_plan_from_stage6(stage6, trip_ids_by_key):
    """One plan row per person-tour aggregate (from trip legs)."""
    trips = stage6[stage6['row_type'] == 'trip'].copy()
    if not len(trips):
        return pd.DataFrame()
    trips['start'] = trips['STARTIME'] * 60.0
    trips['end'] = trips['ENDTIME'] * 60.0
    g = trips.groupby('person_id', sort=False)
    rows = []
    for pid, idx in g.groups.items():
        sub = trips.loc[idx].sort_values('seq')
        rows.append({
            'plan': pid,
            'depart': float(sub['start'].iloc[0]),
            'arrive': float(sub['end'].iloc[-1]),
            'activity': float(len(sub)),
            'walk': 0.0, 'drive': 0.0, 'transit': 0.0, 'wait': 0.0,
            'other': 0.0,
            'length': float(sub['dist'].fillna(0).sum() * 1000.0)
                      if 'dist' in sub else 0.0,
            'cost': float(sub['cost_proxy'].fillna(0).sum())
                    if 'cost_proxy' in sub else 0.0,
            'impedance': 0, 'leg_mode': 0, 'leg_type': 0, 'leg_id': 0,
            'leg_time': 0.0, 'leg_length': 0.0, 'leg_cost': 0.0, 'leg_imp': 0,
        })
    return pd.DataFrame(rows)


def build_trips(stage6, seq_mode_map):
    """Trip table from stage6 trip rows."""
    trips = stage6[stage6['row_type'] == 'trip'].copy()
    if not len(trips):
        return pd.DataFrame()
    trips = trips.rename(columns={'person_id': 'person'})
    trips['trip_id'] = trips.index + 1
    trips['start'] = trips['STARTIME'] * 60.0
    trips['end'] = trips['ENDTIME'] * 60.0
    trips['origin'] = trips['OTAZ'].fillna(0).astype('int64')
    trips['destination'] = trips['DTAZ'].fillna(0).astype('int64')
    trips['mode'] = trips['mode_name'].map(seq_mode_map).fillna(0).astype(int)
    trips['type'] = 11  # ABM
    trips['travel_distance'] = trips['dist'].fillna(0) * 1000.0
    trips['skim_travel_time'] = trips['travtime'].fillna(0) * 60.0
    trips['routed_travel_time'] = trips['travtime'].fillna(0) * 60.0
    trips['monetary_cost'] = trips['cost_proxy'].fillna(0)
    for c in ('hhold', 'path', 'path_multimodal', 'tour', 'trip', 'duration',
              'experienced_gap', 'purpose', 'constraint', 'priority', 'vehicle',
              'passengers', 'partition', 'access_egress_ovtt', 'toll',
              'has_artificial_trip', 'number_of_switches', 'request',
              'initial_energy_level', 'final_energy_level'):
        if c not in trips:
            trips[c] = 0
    return trips


def build_activities(stage6, seq_mode_map, seq_purpose_map):
    """Activity table from stage6 activity rows."""
    act = stage6[stage6['row_type'] == 'activity'].copy()
    if not len(act):
        return pd.DataFrame()
    act = act.rename(columns={'person_id': 'person'})
    act['id'] = act.index + 1
    act['seq_num'] = act['seq'].fillna(0).astype(int)
    act['location_id'] = act['DTAZ'].fillna(0).astype('int64')
    act['origin_id'] = act['OTAZ'].fillna(0).astype('int64')
    act['start_time'] = act['STARTIME'] * 60.0
    dur_col = 'predicted_duration_min' if 'predicted_duration_min' in act else None
    act['duration'] = (act[dur_col].fillna(0) * 60.0) if dur_col else 0.0
    act['mode'] = (act['mode_name'].astype('object')
                   .map(seq_mode_map).fillna(0).map(
                       lambda c: ps.POLARIS_MODE.get(int(c), '')).fillna(''))
    act['type'] = (act['purpose'].astype('object')
                   .map(seq_purpose_map).fillna(0).map(
                       lambda c: ps.POLARIS_ACTIVITY_TYPE.get(int(c), '')).fillna(''))
    act['trip'] = 0  # linked later from trip table
    return act


def link_activities_to_trips(act, trips):
    """For each activity row, set trip = id of the trip that reached it.
    stage6 interleaves rows by seq (activity 1, trip 2, activity 3, ...) so the
    reaching trip for an activity at seq s is the trip at seq s-1 (same person)."""
    if not len(act) or not len(trips):
        return act
    tr = trips[['person', 'seq', 'trip_id']].copy()
    act = act.copy()
    act['_reach_seq'] = act['seq'] - 1
    act = act.merge(tr.rename(columns={'seq': '_reach_seq',
                                       'trip_id': 'reaching_trip'}),
                    on=['person', '_reach_seq'], how='left')
    act['trip'] = act['reaching_trip'].fillna(0).astype('int64')
    act = act.drop(columns=['reaching_trip', '_reach_seq'])
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--population')
    ap.add_argument('--persons')
    ap.add_argument('--stage6')
    ap.add_argument('--zones')
    ap.add_argument('--out')
    ap.add_argument('--run', help='RUN_TAG for runs/ga_parallel/<tag>/stage6')
    args = ap.parse_args()

    if args.run:
        base = os.path.join(CODE_DIR, 'runs', 'ga_parallel', args.run)
        stage6_path = args.stage6 or os.path.join(base, 'stage6', 'stage6_output.csv')
        out_path = args.out or os.path.join(base, f'{args.run}-Demand.sqlite')
    else:
        stage6_path = args.stage6 or os.path.join(CODE_DIR, 'saved_data', 'stage6_output.csv')
        out_path = args.out or os.path.join(CODE_DIR, 'saved_data', 'Demand.sqlite')

    pop_path = args.population or os.path.join(CODE_DIR, 'saved_data', 'synth_population_mapped.csv')
    persons_path = args.persons or os.path.join(CODE_DIR, 'saved_data', 'synth_persons_mapped.csv')
    zones_path = args.zones or os.path.join(CODE_DIR, 'seq_zone_ids.csv')

    log('Building POLARIS-style demand store...')
    pop = load(pop_path)
    persons = load(persons_path)
    stage6 = load(stage6_path)
    zones = load(zones_path, required=False)

    zone_col = 'OTAZ' if 'OTAZ' in pop.columns else 'ORDER'
    person_index = {}
    if zones is not None:
        person_index = dict(zip(zones['ORDER'], range(1, len(zones) + 1)))

    if os.path.exists(out_path):
        os.remove(out_path)
    conn = sqlite3.connect(out_path)
    try:
        cur = conn.cursor()
        for stmt in ps.create_ddl():
            cur.execute(stmt)
        conn.commit()

        log('Households...')
        hh = build_households(pop, zone_col, person_index)
        write_table(conn, 'Household', hh)

        log('Persons...')
        per = build_persons(persons, pop, zone_col, person_index)
        write_table(conn, 'Person', per)

        log('Activities...')
        act = build_activities(stage6, ps.SEQ_MODE_MAP, ps.SEQ_PURPOSE_MAP)
        write_table(conn, 'Activity', act)

        log('Trips...')
        trips = build_trips(stage6, ps.SEQ_MODE_MAP)
        write_table(conn, 'Trip', trips)

        log('Linking activities to reaching trips...')
        act = link_activities_to_trips(act, trips)
        conn.execute('DROP TABLE IF EXISTS "Activity"')
        cur.execute(ps.create_ddl()[4])  # recreate Activity
        write_table(conn, 'Activity', act)
        log(f'  updated Activity.trip links ({len(act):,} rows)')

        log('Plans...')
        plans = build_plan_from_stage6(stage6, {})
        write_table(conn, 'Plan', plans)

        log('Dictionary tables...')
        mode_df = pd.DataFrame(
            [{'mode_id': k, 'mode_description': v}
             for k, v in sorted(ps.POLARIS_MODE.items())])
        write_table(conn, 'Mode', mode_df)
        at_df = pd.DataFrame(
            [{'activity_type_id': k, 'activity_type_description': v}
             for k, v in sorted(ps.POLARIS_ACTIVITY_TYPE.items())])
        write_table(conn, 'Activity_Type', at_df)

        if zones is not None:
            loc = zones[['ORDER', 'SA2_MAIN16', 'SA2_NAME16',
                         'longitude', 'latitude']].copy()
            loc = loc.rename(columns={'ORDER': 'location'})
            loc['location'] = loc['location'].map(person_index)
            write_table(conn, 'Location', loc)

        conn.commit()
        log(f'Done: {out_path}')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
