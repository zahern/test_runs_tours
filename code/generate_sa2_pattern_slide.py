"""
generate_sa2_pattern_slide.py
=============================
Aggregate SA2-level travel patterns by person attribute from the NEWEST
pipeline plan output and inject the stats into the model-review presentation
deck (abm_safety/model_review_deck.html), so the "SA2 travel patterns" slide
always reflects the latest run.

What it computes
----------------
From the newest stage6_output.csv / stage5_output.csv (trip rows only), per
attribute class (income / age / gender / vehicles / person type / licence
holding / household size - see ATTRIBUTES):
  * mode shares (DA/SR/Walk/Bike/Transit) and mean travel time (unchanged)
  * purpose mix (work/edu/social/escort/shopping/othmaint/othdiscr/eat)
  * mean trip distance (km)
  * time-of-day split (AM peak / PM peak / off-peak), derived from STARTIME
    (NOT the 'start_hr' column - that's a stale Stage 3 duration-model
    feature that doesn't track true per-trip timing)
  * an income x vehicles cross-tab (car/active/transit share per cell) to
    surface car-dependency interaction effects a single-attribute
    breakdown can't show
  * top origin SA2 zones by trips, with dominant mode + active share
    (zone names from SEQ_SA2_2016_OSM_counts.csv when available)

Where it writes
---------------
  * abm_safety/Figures/sa2_travel_patterns.json          (provenance + stats)
  * abm_safety/model_review_deck.html                    (in-place, between
      /*SA2_PATTERN_DATA_START*/ ... /*SA2_PATTERN_DATA_END*/ sentinels)

Pipeline integration
--------------------
Called (non-fatally) at the end of pbs_pipeline_mode.pbs after the pipeline
run, so every completed HPC run refreshes the slide. Can also be run by hand:

    python generate_sa2_pattern_slide.py            # newest output, default deck
    python generate_sa2_pattern_slide.py --source path/to/stage5_output.csv
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DECK_HTML = os.path.normpath(os.path.join(CODE_DIR, '..', 'abm_safety', 'model_review_deck.html'))
JSON_OUT = os.path.normpath(os.path.join(CODE_DIR, '..', 'abm_safety', 'Figures', 'sa2_travel_patterns.json'))
ZONE_NAME_CSV = os.path.join(CODE_DIR, 'SEQ_SA2_2016_OSM_counts.csv')

# newest-first candidate patterns, relative to CODE_DIR
SOURCE_PATTERNS = [
    'runs/*/*/stage6_output.csv',
    'runs/*/stage6_output.csv',
    'saved_data/stage6_output.csv',
    'runs/*/*/stage5_output.csv',
    'runs/*/stage5_output.csv',
    'saved_data/stage5_output.csv',
]

MODE_NAMES = {1: 'DA', 2: 'SR', 3: 'Walk', 4: 'Bike', 5: 'Transit'}
MODE_ORDER = ['DA', 'SR', 'Walk', 'Bike', 'Transit']
ACTIVE_MODES = {'Walk', 'Bike'}

PURPOSE_ORDER = ['work', 'edu', 'social', 'escort', 'shopping', 'othmaint', 'othdiscr', 'eat']

# AM/PM peak windows (24h clock). Used only for the peak/offpeak split shown
# per attribute class - not the model's own period definitions.
AM_PEAK = (6, 9)
PM_PEAK = (15, 18)

USECOLS = ['row_type', 'person_id', 'OTAZ', 'mode', 'travtime',
           'incp', 'agep', 'sexp', 'vehrd', 'purpose',
           'ptype', 'dist', 'STARTIME', 'CARLICENCE', 'HHSIZE']

# ── attribute class definitions (ABS Census codes, see census_bridge.py) ──
def _incp_class(v):
    # incp: personal income band 1-17 (1 lowest). Grouped without asserting
    # dollar values (band boundaries differ across census years).
    if not np.isfinite(v):
        return None
    v = int(v)
    if v <= 4:  return 'Bands 1–4 (lowest)'
    if v <= 8:  return 'Bands 5–8'
    if v <= 12: return 'Bands 9–12'
    return 'Bands 13–17 (highest)'


def _agep_class(v):
    # agep: 5-year band 1=0-4 ... 18=85+
    if not np.isfinite(v):
        return None
    v = int(v)
    if v <= 3:  return '0–14'
    if v <= 6:  return '15–29'
    if v <= 9:  return '30–44'
    if v <= 12: return '45–59'
    if v <= 15: return '60–74'
    return '75+'


def _sexp_class(v):
    if not np.isfinite(v):
        return None
    return {1: 'Male', 2: 'Female'}.get(int(v))


def _vehrd_class(v):
    if not np.isfinite(v):
        return None
    v = int(v)
    if v <= 0: return '0 vehicles'
    if v == 1: return '1 vehicle'
    if v == 2: return '2 vehicles'
    return '3+ vehicles'


_PTYPE_LABELS = {
    'FullTimeWorker':    'Full-time worker',
    'PartTimeWorker':    'Part-time worker',
    'UniversityStudent': 'University student',
    'WorkingStudent':    'Working student',
    'SchoolChild':       'School child',
    'Preschool':         'Preschool',
    'Retired':           'Retired',
    'NonWorker':         'Non-worker',
}


def _ptype_class(v):
    # ptype is already a string label (not a numeric code) - passed through
    # the same 'numeric' gate as the other attributes via numeric=False.
    if v is None or v == 'nan' or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return _PTYPE_LABELS.get(v, str(v))


def _licence_class(v):
    if not np.isfinite(v):
        return None
    return 'Licensed' if int(v) == 1 else 'No licence'


def _hhsize_class(v):
    # HHSIZE (household size). hh_alone/hh_child/hh_spouse/hh_parent looked
    # like the natural "household composition" fields but are person-role
    # flags that come out essentially always-0 on the synth population
    # output (checked against runs/ga_parallel/sa_bandit__guided01) - not
    # hh_alone (~7.5%, consistent with HHSIZE==1's share), just the other
    # three. HHSIZE is reliably populated, so use it directly instead.
    if not np.isfinite(v):
        return None
    v = int(v)
    if v <= 1: return '1 person'
    if v == 2: return '2 people'
    if v <= 4: return '3–4 people'
    return '5+ people'


ATTRIBUTES = {
    'income':   {'col': 'incp',  'label': 'Income band',        'fn': _incp_class,
                 'order': ['Bands 1–4 (lowest)', 'Bands 5–8', 'Bands 9–12', 'Bands 13–17 (highest)']},
    'age':      {'col': 'agep',  'label': 'Age group',          'fn': _agep_class,
                 'order': ['0–14', '15–29', '30–44', '45–59', '60–74', '75+']},
    'gender':   {'col': 'sexp',  'label': 'Gender',             'fn': _sexp_class,
                 'order': ['Male', 'Female']},
    'vehicles': {'col': 'vehrd', 'label': 'Household vehicles', 'fn': _vehrd_class,
                 'order': ['0 vehicles', '1 vehicle', '2 vehicles', '3+ vehicles']},
    'ptype':    {'col': 'ptype', 'label': 'Person type', 'fn': _ptype_class, 'numeric': False,
                 'order': ['Full-time worker', 'Part-time worker', 'University student',
                           'Working student', 'School child', 'Preschool', 'Retired', 'Non-worker']},
    'licence':  {'col': 'CARLICENCE', 'label': 'Licence holding', 'fn': _licence_class,
                 'order': ['Licensed', 'No licence']},
    'household': {'col': 'HHSIZE', 'label': 'Household size', 'fn': _hhsize_class,
                  'order': ['1 person', '2 people', '3–4 people', '5+ people']},
}

SENTINEL_RE = re.compile(
    r'/\*SA2_PATTERN_DATA_START\*/.*?/\*SA2_PATTERN_DATA_END\*/', re.DOTALL)


def find_newest_source() -> str | None:
    cands = []
    for pat in SOURCE_PATTERNS:
        cands.extend(glob.glob(os.path.join(CODE_DIR, pat)))
    cands = [c for c in cands if os.path.getsize(c) > 1_000_000]  # skip stubs
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def load_zone_names() -> dict:
    try:
        z = pd.read_csv(ZONE_NAME_CSV, usecols=['ORDER', 'SA2_NAME16'])
        return dict(zip(z['ORDER'].astype(int), z['SA2_NAME16'].astype(str)))
    except Exception as e:
        print(f"[sa2-slide] zone names unavailable ({e}) — using zone numbers")
        return {}


def aggregate(source: str, chunksize: int = 2_000_000) -> dict:
    mode_total = {m: 0 for m in MODE_ORDER}
    # attr -> class -> stats dict: {mode_name: n, 'tt_sum', 'tt_n', 'dist_sum',
    # 'dist_n', 'purpose_n': {purpose: n}, 'peak_n': {peak_am/peak_pm/offpeak: n}}
    attr_counts = {k: {} for k in ATTRIBUTES}
    crosstab_counts = {}                             # (income_cls, veh_cls) -> {mode: n}
    zone_counts = {}                                 # otaz -> {mode: n}
    hourly_counts = [0] * 24                         # departures by hour of day
    purpose_hourly = {}                              # purpose -> [24 hour bins]
    n_trips = 0
    tt_sum = 0.0
    tt_n = 0
    persons = set()

    header = pd.read_csv(source, nrows=0)
    usecols = [c for c in USECOLS if c in header.columns]
    have_attr = {k: (v['col'] in usecols) for k, v in ATTRIBUTES.items()}
    have_purpose = 'purpose' in usecols
    have_dist = 'dist' in usecols
    have_peak = 'STARTIME' in usecols

    def _accumulate(key, cls, base):
        """base: DataFrame with 'mode' + optional 'purpose'/'dist'/'period'/'tt'
        columns, same row order as cls. Groups by cls and folds counts into
        attr_counts[key]."""
        g = base.copy()
        g['cls'] = cls.values
        g = g.dropna(subset=['cls'])
        if g.empty:
            return
        for (c, m), n in g.groupby(['cls', 'mode']).size().items():
            d = attr_counts[key].setdefault(c, {})
            d[m] = d.get(m, 0) + int(n)
        for c, s in g.groupby('cls')['tt'].agg(['sum', 'count']).iterrows():
            d = attr_counts[key].setdefault(c, {})
            d['tt_sum'] = d.get('tt_sum', 0.0) + (float(s['sum']) if np.isfinite(s['sum']) else 0.0)
            d['tt_n'] = d.get('tt_n', 0) + int(s['count'])
        if 'purpose' in g.columns:
            for (c, p), n in g.groupby(['cls', 'purpose']).size().items():
                d = attr_counts[key].setdefault(c, {})
                pn = d.setdefault('purpose_n', {})
                pn[p] = pn.get(p, 0) + int(n)
        if 'dist' in g.columns:
            gd = g.dropna(subset=['dist'])
            for c, s in gd.groupby('cls')['dist'].agg(['sum', 'count']).iterrows():
                d = attr_counts[key].setdefault(c, {})
                d['dist_sum'] = d.get('dist_sum', 0.0) + (float(s['sum']) if np.isfinite(s['sum']) else 0.0)
                d['dist_n'] = d.get('dist_n', 0) + int(s['count'])
        if 'period' in g.columns:
            gp = g.dropna(subset=['period'])
            for (c, p), n in gp.groupby(['cls', 'period']).size().items():
                d = attr_counts[key].setdefault(c, {})
                pk = d.setdefault('peak_n', {})
                pk[p] = pk.get(p, 0) + int(n)

    for chunk in pd.read_csv(source, usecols=usecols, chunksize=chunksize,
                             low_memory=True, on_bad_lines='skip'):
        if 'row_type' in chunk.columns:
            chunk = chunk[chunk['row_type'] == 'trip']
        if chunk.empty:
            continue
        mode = pd.to_numeric(chunk['mode'], errors='coerce')
        mname = mode.map(MODE_NAMES)
        ok = mname.notna()
        chunk, mname = chunk[ok], mname[ok]
        n_trips += len(chunk)
        if 'person_id' in chunk.columns:
            persons.update(chunk['person_id'].astype(str).unique())
        tt = pd.to_numeric(chunk.get('travtime'), errors='coerce')
        tt_ok = tt.notna() & (tt >= 0) & (tt < 600)
        tt_sum += float(tt[tt_ok].sum())
        tt_n += int(tt_ok.sum())

        for m, n in mname.value_counts().items():
            mode_total[m] = mode_total.get(m, 0) + int(n)

        # ── shared per-chunk frame reused across every attribute's groupby ──
        base = pd.DataFrame({'mode': mname.values, 'tt': tt.values}, index=chunk.index)
        if have_purpose:
            base['purpose'] = chunk['purpose'].astype(str).values
        if have_dist:
            base['dist'] = pd.to_numeric(chunk['dist'], errors='coerce').values
        if have_peak:
            # STARTIME is minutes-since-midnight, extended past 1440 for
            # trips that start after midnight (multi-day schedule
            # convention) - wrap to a 0-23 hour-of-day before bucketing.
            # (The 'start_hr' column looks tempting but is a stale Stage 3
            # duration-model feature that doesn't track true trip timing -
            # it clusters 5-12 for every trip regardless of purpose/time.)
            hr = (pd.to_numeric(chunk['STARTIME'], errors='coerce') / 60.0) % 24.0
            period = np.select([(hr >= AM_PEAK[0]) & (hr < AM_PEAK[1]),
                                 (hr >= PM_PEAK[0]) & (hr < PM_PEAK[1])],
                                ['peak_am', 'peak_pm'], default='offpeak')
            base['period'] = np.where(hr.notna(), period, None)
            # hourly departure histogram (24 bins), overall + per purpose —
            # feeds the deck's native "when does travel happen" schedule chart
            _hb = np.floor(hr.dropna()).astype(int).clip(0, 23)
            for h, n in _hb.value_counts().items():
                hourly_counts[int(h)] += int(n)
            if have_purpose:
                _ph = pd.DataFrame({'h': np.floor(hr).values,
                                    'p': chunk['purpose'].astype(str).values}).dropna()
                if not _ph.empty:
                    _ph['h'] = _ph['h'].astype(int).clip(0, 23)
                    for (p, h), n in _ph.groupby(['p', 'h']).size().items():
                        purpose_hourly.setdefault(p, [0] * 24)[int(h)] += int(n)

        # attributes (income/age/gender/vehicles/ptype/licence/household)
        for key, spec in ATTRIBUTES.items():
            if not have_attr[key]:
                continue
            if spec.get('numeric', True):
                raw = pd.to_numeric(chunk[spec['col']], errors='coerce')
                cls = raw.map(lambda v, fn=spec['fn']: fn(v) if np.isfinite(v) else None)
            else:
                cls = chunk[spec['col']].map(spec['fn'])
            _accumulate(key, cls, base)

        # income x vehicles cross-tab (mode share only - no purpose/dist/peak)
        if have_attr.get('income') and have_attr.get('vehicles'):
            inc_raw = pd.to_numeric(chunk['incp'], errors='coerce')
            inc_cls = inc_raw.map(lambda v: _incp_class(v) if np.isfinite(v) else None)
            veh_raw = pd.to_numeric(chunk['vehrd'], errors='coerce')
            veh_cls = veh_raw.map(lambda v: _vehrd_class(v) if np.isfinite(v) else None)
            xt = pd.DataFrame({'inc': inc_cls.values, 'veh': veh_cls.values,
                               'mode': mname.values}).dropna(subset=['inc', 'veh'])
            for (i, v, m), n in xt.groupby(['inc', 'veh', 'mode']).size().items():
                d = crosstab_counts.setdefault((i, v), {})
                d[m] = d.get(m, 0) + int(n)

        # zones
        otaz = pd.to_numeric(chunk.get('OTAZ'), errors='coerce')
        zgrp = pd.DataFrame({'z': otaz, 'mode': mname.values}).dropna()
        for (z, m), n in zgrp.groupby(['z', 'mode']).size().items():
            d = zone_counts.setdefault(int(z), {})
            d[m] = d.get(m, 0) + int(n)

    # ── shape the output ────────────────────────────────────────────────
    def _shares(d):
        tot = sum(d.get(m, 0) for m in MODE_ORDER)
        return ({m: round(100.0 * d.get(m, 0) / tot, 1) for m in MODE_ORDER}, tot)

    def _purpose_shares(d):
        pn = d.get('purpose_n', {})
        tot = sum(pn.get(p, 0) for p in PURPOSE_ORDER)
        if not tot:
            return None
        return {p: round(100.0 * pn.get(p, 0) / tot, 1) for p in PURPOSE_ORDER}

    def _peak_shares(d):
        pk = d.get('peak_n', {})
        tot = sum(pk.get(k, 0) for k in ('peak_am', 'peak_pm', 'offpeak'))
        if not tot:
            return None
        return {k: round(100.0 * pk.get(k, 0) / tot, 1) for k in ('peak_am', 'peak_pm', 'offpeak')}

    overall_shares, _ = _shares(mode_total)
    attrs_out = {}
    for key, spec in ATTRIBUTES.items():
        if not attr_counts.get(key):
            continue
        classes = []
        for c in spec['order']:
            if c not in attr_counts[key]:
                continue
            d = attr_counts[key][c]
            shares, tot = _shares(d)
            mean_tt = (d.get('tt_sum', 0.0) / d['tt_n']) if d.get('tt_n') else None
            mean_dist = (d.get('dist_sum', 0.0) / d['dist_n']) if d.get('dist_n') else None
            classes.append({
                'label': c, 'trips': tot, 'shares': shares,
                'mean_tt': round(mean_tt, 1) if mean_tt else None,
                'mean_dist': round(mean_dist, 1) if mean_dist else None,
                'purpose_shares': _purpose_shares(d),
                'peak_shares': _peak_shares(d),
            })
        if classes:
            attrs_out[key] = {'label': spec['label'], 'classes': classes}

    # income x vehicles cross-tab: car-dependency / active / transit share per cell
    crosstab_cells = []
    for (i, v), d in crosstab_counts.items():
        shares, tot = _shares(d)
        if tot == 0:
            continue
        crosstab_cells.append({
            'income': i, 'vehicles': v, 'trips': tot,
            'car_share': round(shares.get('DA', 0) + shares.get('SR', 0), 1),
            'active_share': round(shares.get('Walk', 0) + shares.get('Bike', 0), 1),
            'transit_share': shares.get('Transit', 0),
        })
    crosstab_out = {
        'row_label': 'Income band', 'row_order': ATTRIBUTES['income']['order'],
        'col_label': 'Household vehicles', 'col_order': ATTRIBUTES['vehicles']['order'],
        'cells': crosstab_cells,
    } if crosstab_cells else None

    zone_names = load_zone_names()
    zones = []
    for z, d in zone_counts.items():
        shares, tot = _shares(d)
        if tot == 0:
            continue
        active = shares.get('Walk', 0) + shares.get('Bike', 0)
        dom = max(MODE_ORDER, key=lambda m: d.get(m, 0))
        zones.append({'zone': z, 'name': zone_names.get(z, f'Zone {z}'),
                      'trips': tot, 'dominant': dom, 'active_pct': round(active, 1)})
    zones.sort(key=lambda r: -r['trips'])

    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(source))
    try:
        _src_rel = os.path.relpath(source, CODE_DIR)
    except ValueError:      # e.g. source on a different Windows drive
        _src_rel = source
    return {
        'source': _src_rel.replace('\\', '/'),
        'source_mtime': mtime.strftime('%Y-%m-%d %H:%M'),
        'generated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'n_trips': n_trips,
        'n_persons': len(persons),
        'mean_travtime': round(tt_sum / tt_n, 1) if tt_n else None,
        'overall_shares': overall_shares,
        'active_share': round(overall_shares.get('Walk', 0) + overall_shares.get('Bike', 0), 1),
        'attributes': attrs_out,
        'crosstab_income_vehicles': crosstab_out,
        'hourly': hourly_counts,
        'purpose_hourly': {p: v for p, v in purpose_hourly.items()
                           if p in PURPOSE_ORDER or p == 'home'},
        'top_zones': zones[:10],
        'n_zones': len(zones),
    }


def inject(deck_path: str, data: dict) -> bool:
    with open(deck_path, 'r', encoding='utf-8') as fh:
        html = fh.read()
    payload = ('/*SA2_PATTERN_DATA_START*/window.SA2_DATA = '
               + json.dumps(data, ensure_ascii=False, separators=(',', ':'))
               + ';/*SA2_PATTERN_DATA_END*/')
    new_html, n = SENTINEL_RE.subn(payload, html)
    if n == 0:
        print(f"[sa2-slide] ERROR: sentinels not found in {deck_path} — deck not updated")
        return False
    with open(deck_path, 'w', encoding='utf-8') as fh:
        fh.write(new_html)
    print(f"[sa2-slide] deck updated: {deck_path}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=None, help='plan CSV (default: newest run output)')
    ap.add_argument('--deck', default=DECK_HTML)
    ap.add_argument('--json-out', default=JSON_OUT)
    args = ap.parse_args(argv)

    source = args.source or find_newest_source()
    if not source or not os.path.exists(source):
        print("[sa2-slide] no plan output found — nothing to do")
        return 1
    print(f"[sa2-slide] aggregating {source} "
          f"({os.path.getsize(source) / 1e9:.1f} GB) ...")
    data = aggregate(source)
    print(f"[sa2-slide] {data['n_trips']:,} trips, {data['n_persons']:,} persons, "
          f"{data['n_zones']} zones, active share {data['active_share']}%")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"[sa2-slide] stats written: {args.json_out}")

    if os.path.exists(args.deck):
        inject(args.deck, data)
    else:
        print(f"[sa2-slide] deck not found at {args.deck} — JSON written only")
    return 0


if __name__ == '__main__':
    sys.exit(main())
