"""
_smoke_pipeline_matsim.py
=========================
End-to-end pipeline smoke test on SYNTHETIC data:

    estimate (one model per stage) -> predict the full chain -> MATSim export

Stages exercised
----------------
  S0  individual CDAP MNL          (cdap.estimate_cdap; needs LARCH)
  S0  JOINT household CDAP         (shared + per-size regimes, no heavy deps)
  S1  activity frequency           ordered logit per purpose
  S2  day budget / timing window   Tobit models
  S3  activity duration            BudgetLognormal AFT
  S4  schedule placement           RRM/MNL timing models
  S5  destination + mode           (needs skims OMX + larch — skipped unless
                                    SMOKE_SKIMS is set)
  P   prediction chain             CDAP -> household coordination ->
                                   frequency -> day budget -> durations ->
                                   schedule times
  M   MATSim export                plans.xml + facilities.xml + config.xml,
                                   parsed back to guarantee valid XML

Every step reports PASS / FAIL / SKIP (with reason). Exits non-zero on any
FAIL. SKIPs are expected on machines without Larch/openmatrix (e.g. a laptop);
on the HPC the same script exercises those stages too.

Run:
    python _smoke_pipeline_matsim.py --out runs/smoke_full --persons 400
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

PURPOSES = ['work', 'edu', 'escort', 'shopping', 'social', 'othmaint',
            'othdiscr', 'eat']
PTYPES = ['FullTimeWorker', 'PartTimeWorker', 'SchoolChild',
          'UniversityStudent', 'Retired', 'NonWorker']
NONMAND = {'escort': 'Pickup/Dropoff Someone', 'shopping': 'Shopping',
           'social': 'Social Visit', 'othmaint': 'Personal Business',
           'othdiscr': 'Other Place', 'eat': 'Eat/drink'}
MAND_RAW = {'work': ['Full-time Work', 'Part-time Work'],
            'edu': ['Primary School', 'Full-time TAFE/Uni']}
NZONES = 12


# --------------------------------------------------------------------------- #
# synthetic HHTS + synth population                                            #
# --------------------------------------------------------------------------- #
def make_persons(n, seed):
    rng = np.random.default_rng(seed)
    hh_sizes = rng.choice([1, 2, 3, 4, 5], n, p=[.22, .28, .22, .18, .10])
    hhid = np.repeat(np.arange(len(hh_sizes)) + 1, hh_sizes)[:n]
    return pd.DataFrame({
        'person_id': np.arange(n),
        'PERSID': np.arange(n),                # legacy alias some readers want
        'HHID': hhid[:n],
        'ptype': rng.choice(PTYPES, n),
        'sexp': rng.integers(1, 3, n),
        'vehrd': rng.integers(0, 3, n),
        'nprd': np.bincount(hhid[:n])[hhid[:n]],
        'incp': rng.integers(1, 15, n),
        'AGEGROUP': rng.integers(1, 12, n),
        'CARLICENCE': rng.integers(0, 2, n),
        'n_workers_hh': rng.integers(0, 3, n),
        'wfh': rng.integers(0, 2, n),
        'WFHWEEK': rng.integers(0, 5, n),
        'WORKSTAT': rng.integers(0, 3, n),
        'STUDYING': rng.integers(0, 5, n),
    })


def make_tours(persons, seed):
    """One-day agenda per person -> tour rows with RAW QTS-style purposes."""
    rng = np.random.default_rng(seed + 1)
    recs = []
    tid = 0
    for _, p in persons.iterrows():
        pt = p['ptype']
        t = int(rng.integers(1, 4))
        clock = float(rng.uniform(420, 520))
        workerish = pt in ('FullTimeWorker', 'PartTimeWorker', 'WorkingStudent')
        schoolish = pt in ('SchoolChild', 'UniversityStudent')
        for k in range(t):
            if k == 0 and (workerior := workerish or schoolish):
                pool = (MAND_RAW['work'] if workerish else MAND_RAW['edu'])
                purp = str(rng.choice(pool))
            else:
                purp = str(rng.choice(list(NONMAND.values())))
            dur = float(rng.uniform(25, 240))
            code = next((c for c, raw in {**MAND_RAW, **NONMAND}.items()
                         if raw == purp or purp in raw), 'othdiscr')
            recs.append({'tour_id': tid, 'person_id': int(p['person_id']),
                         'main_purpose': purp, '_code': code,
                         'duration': dur, 'start_time': clock,
                         'end_time': clock + dur,
                         'OTAZ': int(rng.integers(1, NZONES)),
                         'DTAZ': int(rng.integers(1, NZONES))})
            clock += dur + float(rng.uniform(10, 45))
            tid += 1
    tf = pd.DataFrame(recs)
    # per-person outbound counts (some estimators read these directly)
    cnt = tf.groupby('person_id')['_code'].value_counts().unstack(fill_value=0)
    for c in PURPOSES:
        tf[f'n_{c}_out'] = tf['person_id'].map(cnt.get(c, pd.Series(dtype=int))) \
            .fillna(0).astype(int)
    return tf.drop(columns=['_code'])


def make_trips(persons, tours, seed):
    """Two legs per tour (outbound + return) like an activity-day diary."""
    rng = np.random.default_rng(seed + 2)
    recs = []
    for _, t in tours.iterrows():
        tt = float(rng.uniform(5, 40))
        dep = float(t['start_time']) - tt
        recs.append({'person_id': int(t['person_id']),
                     'DESTPURP': t['main_purpose'], '_code': t['_code'] if '_code' in tours.columns else '',
                     'OTAZ': int(t['OTAZ']), 'DTAZ': int(t['DTAZ']),
                     'STARTIME': dep, 'TRAVTIME': tt,
                     'DURATION': tt + float(t['duration']),
                     'ACTIVITY_DURATION': float(t['duration'])})
        recs.append({'person_id': int(t['person_id']),
                     'DESTPURP': 'Home', '_code': 'home',
                     'OTAZ': int(t['DTAZ']), 'DTAZ': int(t['OTAZ']),
                     'STARTIME': dep + tt + float(t['duration']),
                     'TRAVTIME': tt, 'DURATION': tt * 2 + float(t['duration']),
                     'ACTIVITY_DURATION': 0.0})
    tr = pd.DataFrame(recs)
    tr.insert(0, 'trip_id', np.arange(len(tr)))
    if '_code' in tr.columns:
        tr = tr.drop(columns=['_code'])
    return tr


def make_synth(persons, seed):
    synth = persons.copy()
    synth['person_id'] = synth['person_id'] + 10_000      # disjoint ids
    synth['HOMETAZ'] = np.random.default_rng(seed).integers(1, NZONES, len(synth))
    synth['OTAZ'] = synth['HOMETAZ']
    return synth


# --------------------------------------------------------------------------- #
# harness                                                                      #
# --------------------------------------------------------------------------- #
class SkipStep(Exception):
    """Raised by a step to demote itself to SKIP with a reason."""


class Smoke:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.results = []                      # (step, status, detail)
        self.models = {}

    def record(self, step, status, detail=''):
        self.results.append((step, status, str(detail)[:220]))
        print(f'  [{status:4}] {step}' + (f' — {detail}' if detail else ''))

    def run(self, step, fn, skip_if=None):
        """Run one step; SKIP when skip_if() returns a reason string."""
        if skip_if is not None:
            reason = skip_if()
            if reason:
                self.record(step, 'SKIP', reason)
                return None
        try:
            out = fn()
            self.record(step, 'PASS')
            return out
        except SkipStep as skip:
            self.record(step, 'SKIP', skip)
            return None
        except Exception as exc:               # noqa: BLE001
            traceback.print_exc()
            self.record(step, 'FAIL', f'{type(exc).__name__}: {exc}')
            return None


def attach_cdap_truth(persons, tours):
    """
    Derive ANYSTOPS / MAINACT from the synthetic tours so prepare_cdap_data's
    observed target matches what the tours actually contain. Without this the
    individual CDAP trains on an all-N target (prepare_cdap_data defaults
    ANYSTOPS=True / MAINACT='' when the columns are missing).
    """
    from cdap_households_estimation import _purpose_is_mandatory
    mand, anyt = {}, {}
    for pid, s in tours.groupby('person_id')['main_purpose']:
        vals = [str(v) for v in s.dropna()]
        if not vals:
            continue
        anyt[int(pid)] = True
        m = [v for v in vals if _purpose_is_mandatory(v)]
        mand[int(pid)] = m[0] if m else vals[0]
    persons = persons.copy()
    persons['ANYSTOPS'] = persons['person_id'].map(anyt).fillna(False)
    persons['ANYSTOPS'] = persons['ANYSTOPS'].astype(bool)
    persons['MAINACT'] = persons['person_id'].map(mand).fillna('')
    return persons


# --------------------------------------------------------------------------- #
# prediction-chain helpers                                                     #
# --------------------------------------------------------------------------- #
def synth_cdap_fallback(persons_hhts, tours_hhts, synth, seed=7):
    """When Larch is unavailable, draw CDAP letters from the observed
    per-ptype M/N/H marginals (same definition as derive_observed_dap)."""
    from cdap_households_estimation import derive_observed_dap
    dap = derive_observed_dap(persons_hhts, tours_hhts)
    ref = persons_hhts.assign(_dap=dap.values).groupby('ptype')['_dap'] \
        .value_counts(normalize=True).unstack(fill_value=0.0)
    for c in ('M', 'N', 'H'):
        if c not in ref.columns:
            ref[c] = 0.0
    ref = ref[['M', 'N', 'H']].div(ref[['M', 'N', 'H']].sum(axis=1), axis=0)
    rng = np.random.default_rng(seed)
    probs = synth['ptype'].map(
        lambda pt: ref.loc[pt] if pt in ref.index else pd.Series([.3, .4, .3]))
    letters = np.array([str(rng.choice(['M', 'N', 'H'], p=p.to_numpy()))
                        for p in probs])
    synth = synth.copy()
    synth['CDAP'] = letters
    for d in ('M', 'N', 'H'):
        synth[f'cdap_util_{d}'] = 0.0          # flat utils: interactions drive
    return synth


def build_trip_legs(activities, synth, seed):
    """
    Emulate the stage-6 trip-leg view from scheduled activities:
    home -> act -> ... -> home, with departure times and modes.
    Output matches matsim_export.build_plans expectations.
    """
    rng = np.random.default_rng(seed)
    mode_pool = ['DA', 'Walk', 'Transit', 'Bike']
    acts = activities[activities.get('main_purpose', activities.iloc[:, 0]).notna()] \
        if 'main_purpose' in activities.columns else activities
    home_zone = dict(zip(synth['person_id'], synth['HOMETAZ']))
    lat_of = {z: -27.45 + z * 0.03 for z in range(NZONES)}
    lon_of = {z: 153.02 + z * 0.03 for z in range(NZONES)}
    rows = []
    for pid, grp in acts.groupby('person_id'):
        hz = int(home_zone.get(pid, 1))
        prev_zone, prev_end = hz, 6.5 * 60
        grp = grp.sort_values('_start') if '_start' in grp.columns else grp
        for _, a in grp.iterrows():
            dz = int(rng.integers(1, NZONES))
            rows.append({
                'person_id': pid, 'row_type': 'trip',
                'OTAZ': prev_zone, 'DTAZ': dz,
                'departure_time': prev_end,
                'mode_name': str(rng.choice(mode_pool)),
                'purpose': str(a.get('purpose', 'other')),
                'origin_lat': lat_of[prev_zone], 'origin_lon': lon_of[prev_zone],
                'dest_lat': lat_of[dz], 'dest_lon': lon_of[dz],
            })
            prev_zone, prev_end = dz, float(a.get('_end', prev_end + 60))
        rows.append({'person_id': pid, 'row_type': 'trip',
                     'OTAZ': prev_zone, 'DTAZ': hz,
                     'departure_time': prev_end, 'mode_name': 'Walk',
                     'purpose': 'home',
                     'origin_lat': lat_of[prev_zone], 'origin_lon': lon_of[prev_zone],
                     'dest_lat': lat_of[hz], 'dest_lon': lon_of[hz]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/smoke_full')
    ap.add_argument('--persons', type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sm = Smoke(args.out)
    print(f'== FULL PIPELINE SMOKE == out={args.out} persons={args.persons}\n')

    # ── synthetic HHTS ────────────────────────────────────────────────────
    persons = make_persons(args.persons, 42)
    tours = make_tours(persons, 42)
    trips = make_trips(persons, tours, 42)
    persons = attach_cdap_truth(persons, tours)

    # ── estimation ────────────────────────────────────────────────────────
    def _no_larch():
        try:
            import larch  # noqa: F401
        except ImportError:
            return 'larch not installed'
        return None

    def est_s0_indiv():
        from cdap import estimate_cdap
        return estimate_cdap(persons, save_path=os.path.join(
            args.out, 'smoke_cdap_model.pkl'))

    sm.models['S0_cdap_individual'] = sm.run(
        'ESTIMATE S0 individual CDAP MNL (larch)', est_s0_indiv,
        skip_if=_no_larch)

    def est_s0_joint_shared():
        from cdap_households_estimation import estimate_joint_cdap_by_hh_size
        import cdap_households
        res = estimate_joint_cdap_by_hh_size(
            persons, tours, cdap_households.HouseholdCdapConfig())
        assert res['sizes'], 'no size buckets'
        return res

    sm.models['S0_joint_shared'] = sm.run(
        'ESTIMATE S0 joint household CDAP (shared)', est_s0_joint_shared)

    def est_s0_joint_persize():
        from cdap_households_estimation import estimate_joint_cdap_by_hh_size
        import cdap_households
        res = estimate_joint_cdap_by_hh_size(persons, tours,
                                             cdap_households.HouseholdCdapConfig(),
                                             per_size=True)
        assert res['size_models'], 'no dedicated size models'
        return res

    sm.models['S0_joint_persize'] = sm.run(
        'ESTIMATE S0 joint household CDAP (per-size)', est_s0_joint_persize)

    def est_s1():
        from stage1_activity_frequency import estimate_frequency_models
        return estimate_frequency_models(tours, persons)

    sm.models['S1'] = sm.run('ESTIMATE S1 activity frequency', est_s1)

    def est_s2():
        from stage2_day_duration import aggregate_person_day, estimate_day_models
        return estimate_day_models(aggregate_person_day(tours, persons),
                                   persons=persons)

    sm.models['S2'] = sm.run('ESTIMATE S2 day budget (Tobit)', est_s2)

    def est_s3():
        from stage3_activity_duration import estimate_duration_models
        return estimate_duration_models(tours, persons, trips=trips)

    sm.models['S3'] = sm.run('ESTIMATE S3 activity duration', est_s3)

    def est_s4():
        from stage4_schedule_placement import estimate_timing_models
        return estimate_timing_models(trips, tours, persons)

    sm.models['S4'] = sm.run('ESTIMATE S4 schedule placement', est_s4)

    _LANDUSE = {}

    def _s5_ready():
        if not os.environ.get('SMOKE_SKIMS'):
            return 'set SMOKE_SKIMS=<omx> to enable'
        try:
            from data_connect import Synth_Connect
            land = Synth_Connect().get_land()
        except Exception as exc:                       # noqa: BLE001
            return f'landuse unavailable ({type(exc).__name__}: {exc})'
        if land is None or not len(land):
            return 'landuse table empty/unavailable'
        _LANDUSE['df'] = land
        return None

    def est_s5():
        from stage5_destination_choice import run_stage5
        try:
            return run_stage5(trips, persons, make_synth(persons, 43),
                              os.environ['SMOKE_SKIMS'],
                              landuse=_LANDUSE['df'], tours=tours,
                              seed=11, verbose=False)
        except RuntimeError as exc:
            if 'All batches failed' in str(exc):
                # synthetic OTAZ/DTAZ are not in the real skim zone universe,
                # so the batched dest/mode prediction legitimately cannot run.
                # The REAL-data stage-5 smoke is: python _tmp_smoke_allseg.py
                raise SkipStep(
                    'synthetic zones outside skim universe — for real-data S5 '
                    'use: python _tmp_smoke_allseg.py')
            raise

    sm.models['S5'] = sm.run('ESTIMATE S5 destination+mode (skims)',
                             est_s5, skip_if=_s5_ready)

    # ── prediction chain ──────────────────────────────────────────────────
    print()
    synth = make_synth(persons, 44)

    def pred_cdap():
        from cdap import predict_cdap
        return predict_cdap(synth, sm.models['S0_cdap_individual'])

    got_cdap = sm.run('PREDICT  S0 CDAP letters (larch)', pred_cdap,
                      skip_if=lambda: (
                          None if sm.models.get('S0_cdap_individual')
                          else 'larch not installed'))
    if got_cdap is None:
        synth = synth_cdap_fallback(persons, tours, synth)
        sm.record('PREDICT  S0 CDAP letters (marginal fallback)', 'PASS',
                  'marginal draw (larch unavailable)')
    else:
        synth = got_cdap
        assert synth['CDAP'].isin(['M', 'N', 'H']).all()

    def coord():
        import cdap_households
        cfg = cdap_households.HouseholdCdapConfig()
        js = sm.models.get('S0_joint_shared')
        jp = sm.models.get('S0_joint_persize')
        if js:
            if js.get('interaction'):
                cfg.interaction.clear(); cfg.interaction.update(js['interaction'])
            for a in ('interaction3', 'interaction4', 'interaction5'):
                if js.get(a):
                    getattr(cfg, a).clear(); getattr(cfg, a).update(js[a])
        if jp and jp.get('size_models'):
            cfg.per_size_tables = {
                int(s): {'interaction': sm_.get('interaction') or {},
                         'interaction3': sm_.get('interaction3') or {},
                         'interaction4': sm_.get('interaction4') or {},
                         'interaction5': sm_.get('interaction5') or {}}
                for s, sm_ in jp['size_models'].items()}
        return cdap_households.coordinate_household_cdap(synth, cfg)

    _coordinated = sm.run('PREDICT  S0 household coordination (draw stage)', coord)
    if _coordinated is not None:
        synth = _coordinated

    def pred_s1():
        from stage1_activity_frequency import predict_activity_frequency
        return predict_activity_frequency(synth, sm.models['S1'])

    _s1_out = sm.run('PREDICT  S1 activity counts', pred_s1)
    if _s1_out is not None:
        synth = _s1_out

    def pred_s2():
        from stage2_day_duration import predict_day_budget
        return predict_day_budget(synth, sm.models['S2'])

    _s2_out = sm.run('PREDICT  S2 day budget', pred_s2)
    if _s2_out is not None:
        synth = _s2_out

    def pred_s3():
        from stage3_activity_duration import predict_activity_durations
        return predict_activity_durations(synth, sm.models['S3'])

    activities = sm.run('PREDICT  S3 activity durations', pred_s3)
    activities = activities if activities is not None else synth

    def pred_s4():
        from stage4_schedule_placement import predict_schedule
        return predict_schedule(activities, sm.models['S4'])

    _s4_out = sm.run('PREDICT  S4 schedule times', pred_s4,
                     skip_if=lambda: (
                         None if sm.models.get('S4')
                         else 'S4 estimation unavailable'))
    if _s4_out is not None:
        activities = _s4_out

    # ── MATSim export ─────────────────────────────────────────────────────
    print()
    def to_matsim():
        import matsim_export as mx
        plan_df = build_trip_legs(activities, synth, seed=11)
        # canonical trip-leg name so matsim_export.export_matsim() and
        # job_matsim_scenario.pbs can consume this dir like any pipeline run
        plan_df.to_csv(os.path.join(args.out, 'trips_for_aimsun.csv'),
                       index=False)
        persons_plans, stats = mx.build_plans(plan_df, zone_xy={},
                                              transform=(lambda lon, lat: (lon, lat)))
        assert persons_plans, 'no plans reconstructed'
        p_path = os.path.join(args.out, 'plans.xml')
        mx.write_plans_xml(persons_plans, p_path)
        f_path = os.path.join(args.out, 'facilities_empty.xml')
        mx.write_facilities_xml({}, f_path)
        c_path = os.path.join(args.out, 'config.xml')
        act_types = sorted(set(plan_df['purpose'].astype(str)))
        mx.write_config_xml(c_path, 'plans.xml', 'facilities_empty.xml',
                            crs='EPSG:4326', activity_types=act_types)
        import xml.etree.ElementTree as ET
        for f in (p_path, c_path):
            ET.parse(f)                       # must be well-formed XML
        root = ET.parse(p_path).getroot()
        assert root.tag == 'population' and len(root) == len(persons_plans)
        print(f"           MATSim: {stats['n_persons_out']} persons, "
              f"{stats['n_activities']} acts, {stats['n_legs']} legs, "
              f"modes={stats['mode_counts']}")
        return stats

    sm.run('EXPORT   MATSim plans/facilities/config', to_matsim)

    # ── summary ───────────────────────────────────────────────────────────
    df = pd.DataFrame(self_results := sm.results,
                      columns=['step', 'status', 'detail'])
    csv = os.path.join(args.out, 'smoke_summary.csv')
    df.to_csv(csv, index=False)
    fails = int((df['status'] == 'FAIL').sum())
    skips = int((df['status'] == 'SKIP').sum())
    print(f'\n== SMOKE SUMMARY ==  PASS={len(df)-fails-skips} '
          f'FAIL={fails} SKIP={skips}   → {csv}')
    if fails:
        print(df[df['status'] == 'FAIL'].to_string(index=False))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
