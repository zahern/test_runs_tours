#!/usr/bin/env python
"""
build_constraint_schema.py
==========================
Introspect the pipeline's REAL candidate variables, model coefficients, and
data attributes and dump them to ``constraint_schema.json``. The constraint GUI
(``constraint_gui.py``) reads that schema so its dropdowns are always linked to
the actual model/data — add a variable to the pipeline, re-run this, and it
appears in the GUI.

Usage
-----
    python build_constraint_schema.py                 # try to load real data
    python build_constraint_schema.py --no-data       # static pools only (fast)
    python build_constraint_schema.py --out foo.json

The schema is a plain JSON object::

    {
      "generated": "<iso timestamp>",
      "stages": {
         "stage1": {"label": ..., "name_kind": "feature", "candidates": [...]},
         ...,
         "stage5_mode": {"label": ..., "name_kind": "coefficient",
                          "segments": [...], "candidates": [...coeff names...]},
         "stage5_dest": {..., "land_attributes": [...]}
      },
      "ptypes": [...],            # ptype_* dummy columns
      "land_attributes": [...],   # land-use / employment attractor columns
      "person_attributes": [...], # person-level driver columns
      "constraint_types": [...]   # what the GUI can author
    }

``name_kind`` tells the GUI which name set a stage's constraints operate on:
``feature`` = GA candidate gene names (stages 1-4), ``coefficient`` = model
coefficient names like ``c_aggl`` (stage 5, whose constraints live on coeffs).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

STAGE_LABELS = {
    'stage1':      'Stage 1 — activity frequency',
    'stage2':      'Stage 2 — day pattern / duration',
    'stage3':      'Stage 3 — activity duration',
    'stage4':      'Stage 4 — schedule placement',
    'stage5_mode': 'Stage 5 — tour mode choice',
    'stage5_dest': 'Stage 5 — destination choice',
}

# Sensible default ptype ↔ land-attribute affinities (behavioural priors). The
# builder keeps only those whose ptype + land actually exist, so the GUI opens
# pre-populated with meaningful pairings the user can edit or remove.
_DEFAULT_PTYPE_LAND = [
    # Education destinations by student type
    {'ptype': 'ptype_UniversityStudent', 'land': 'Education',        'segments': ['edu']},
    {'ptype': 'ptype_UniversityStudent', 'land': 'HSENROLL',         'segments': ['edu']},
    {'ptype': 'ptype_WorkingStudent',    'land': 'Education',        'segments': ['edu']},
    {'ptype': 'ptype_SchoolChild',       'land': 'Secondary',       'segments': ['edu']},
    {'ptype': 'ptype_SchoolChild',       'land': 'Primary',         'segments': ['edu']},
    # Work destinations by worker sector
    {'ptype': 'ptype_FullTimeWorker',    'land': 'TOTEMP',          'segments': ['work']},
    {'ptype': 'ptype_FullTimeWorker',    'land': 'FPSEMPN',         'segments': ['work']},
    {'ptype': 'ptype_PartTimeWorker',    'land': 'RETEMPN',         'segments': ['work']},
    {'ptype': 'ptype_PartTimeWorker',    'land': 'TOTEMP',          'segments': ['work']},
    # Non-mandatory destinations by lifestage
    {'ptype': 'ptype_Retired',           'land': 'Healthcare',      'segments': ['nonmand']},
    {'ptype': 'ptype_Retired',           'land': 'recreation_ground', 'segments': ['nonmand']},
    {'ptype': 'ptype_Retired',           'land': 'community',       'segments': ['nonmand']},
    # ── Person-attribute → land (destination utility by driver) ─────────────
    # Uses continuous person columns that exist in the stage-5 person table, so
    # the interaction person.X × emp.Y always materialises. "Income → job land"
    # for all relevant industry types, plus vehicle / household / age drivers.
    {'ptype': 'incp',  'land': 'FPSEMPN',          'segments': ['work']},    # income → finance/professional
    {'ptype': 'incp',  'land': 'commercial',       'segments': ['work']},    # income → commercial/office
    {'ptype': 'incp',  'land': 'HEREMPN',          'segments': ['work']},    # income → health/education prof
    {'ptype': 'incp',  'land': 'RETEMPN',          'segments': ['work']},    # income → retail (−: low pref)
    {'ptype': 'incp',  'land': 'government',        'segments': ['work']},    # income → government/public
    {'ptype': 'vehrd', 'land': 'auto_work_acc',    'segments': ['work']},    # cars → auto-accessible work
    {'ptype': 'vehrd', 'land': 'auto_nonmand_acc', 'segments': ['nonmand']}, # cars → auto-accessible leisure
    {'ptype': 'nprd',  'land': 'residential',      'segments': ['nonmand']}, # household size → residential
    {'ptype': 'nprd',  'land': 'Primary',          'segments': ['edu']},     # family → primary schools
    {'ptype': 'age5p', 'land': 'Healthcare',       'segments': ['nonmand']}, # older → healthcare
    {'ptype': 'age5p', 'land': 'recreation_ground', 'segments': ['nonmand']}, # older → recreation
]

# Curated person "drivers" the GUI offers (alongside ptype_*) for person↔land
# affinities — continuous columns present in the stage-5 person table.
_PERSON_DRIVERS = ['incp', 'vehrd', 'nprd', 'age5p', 'sexp', 'travel_propensity',
                   'occp_group', 'STUDYING', 'WORKSTAT']

# Constraint keys the GUI can author (and therefore prefill from the built-ins).
# force_random (dict var→dist) and never_random (list) drive which coefficients
# are searched as RANDOM/mixed vs held fixed.
_EDITABLE_KEYS = ('force_include', 'force_exclude', 'mutually_exclusive',
                  'min_behavioral', 'sign_constraints',
                  'force_random', 'never_random')

# Land-use / employment attractor columns worth offering for ptype↔land and
# min_behavioral rules. Extended with whatever the loaded landuse actually has.
_DEFAULT_LAND_ATTRS = [
    'Education', 'Primary', 'Secondary', 'HSENROLL',
    'Healthcare', 'Public_Services', 'childcare',
    'FPSEMPN', 'HEREMPN', 'RETEMPN', 'AGREMPN', 'MWTEMPN', 'TOTEMP',
    'retail', 'commercial', 'industrial', 'government', 'residential',
    'Entertainment', 'Shopping', 'EATOUT', 'recreation_ground', 'sport',
    'park', 'community', 'civic', 'TOTPOP', 'TOTHH', 'is_cbd',
]


def _safe(fn, default):
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - introspection must never hard-fail
        print(f"  ! {getattr(fn, '__name__', 'introspect')}: {exc!r}")
        return default


def _stage5_names():
    """Coefficient + gene names for the two stage-5 models (no data needed)."""
    import stage5_destination_choice as s5
    import ga_hypothesis_testing as gh
    mode = {
        'label': STAGE_LABELS['stage5_mode'],
        'name_kind': 'coefficient',
        'segments': ['work', 'edu', 'nonmand'],
        # constraints operate on coefficient names; genes are also offered so
        # the user can force-include / mutex a candidate the GA toggles.
        'candidates': sorted(set(list(s5.MODE_UEC.keys())
                                 + gh._stage5_mode_candidates())),
        'coefficients': sorted(s5.MODE_UEC.keys()),
    }
    dest = {
        'label': STAGE_LABELS['stage5_dest'],
        'name_kind': 'coefficient',
        'segments': ['work', 'edu', 'nonmand'],
        'candidates': sorted(set(list(s5.DEST_UEC.keys())
                                 + gh._stage5_dest_candidates())),
        'coefficients': sorted(s5.DEST_UEC.keys()),
    }
    return mode, dest


def build_schema(use_data: bool = True) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    stages: dict = {}
    ptypes: list = []
    land_attrs: list = list(_DEFAULT_LAND_ATTRS)
    person_attrs: list = []

    # ── Optionally load real data to enumerate actual columns ────────────────
    data_persons = None
    landuse = None
    if use_data:
        print("Loading a data sample for real attribute names (use --no-data to skip)...")
        def _load():
            from data_connect import Synth_Connect
            sc = Synth_Connect()
            out = sc.process_survey(save=False)
            land = sc.get_land()
            return out, land
        loaded = _safe(_load, None)
        if loaded is not None:
            out, landuse = loaded
            data_persons = out.get('persons')
            print(f"  loaded persons={0 if data_persons is None else len(data_persons)}, "
                  f"landuse cols={0 if landuse is None else len(landuse.columns)}")

    # ── Stage 1-4 candidate pools (data-driven) ──────────────────────────────
    def _stage14():
        import ga_hypothesis_testing as gh
        import pandas as pd
        # Enrich a person frame the way each stage does, so interaction genes show up.
        base = data_persons
        if base is None:
            # minimal synthetic frame carrying the common driver columns so the
            # static candidate names still resolve
            base = pd.DataFrame({c: [0, 1] for c in (
                'sexp', 'vehrd', 'nprd', 'incp', 'CARLICENCE', 'occp_group', 'HHSIZE',
                'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
                'walk_acc', 'transit_acc', 'auto_work_acc', 'bike_acc',
                'hh_child', 'WFHWEEK', 'is_female', 'n_total', 'n_tours')})
        enr = gh._enrich_ga_data(base, base)
        out = {}
        for sid, fn in ((1, gh._stage1_candidates), (2, gh._stage2_candidates),
                        (3, gh._stage3_candidates), (4, gh._stage4_candidates)):
            key = f'stage{sid}'
            out[key] = {
                'label': STAGE_LABELS[key],
                'name_kind': 'feature',
                'candidates': sorted(fn(enr)),
            }
        return out
    stages.update(_safe(_stage14, {}))

    # ── Stage 5 coeff/gene names ─────────────────────────────────────────────
    s5names = _safe(_stage5_names, None)
    if s5names is not None:
        stages['stage5_mode'], stages['stage5_dest'] = s5names

    # ── ptypes, land attributes, person attributes from real data ────────────
    if data_persons is not None:
        ptypes = sorted([c for c in data_persons.columns if str(c).startswith('ptype_')])
        person_attrs = sorted([c for c in data_persons.columns
                               if data_persons[c].dtype.kind in 'if'])[:200]
    if not ptypes:
        ptypes = ['ptype_FullTimeWorker', 'ptype_PartTimeWorker',
                  'ptype_UniversityStudent', 'ptype_WorkingStudent',
                  'ptype_SchoolChild', 'ptype_Retired']
    if landuse is not None:
        land_attrs = sorted(set(land_attrs) | {
            c for c in landuse.columns
            if getattr(landuse[c], 'dtype', None) is not None
            and landuse[c].dtype.kind in 'if'})
    if 'stage5_dest' in stages:
        stages['stage5_dest']['land_attributes'] = sorted(land_attrs)

    # ── Built-in constraints per stage → schema 'defaults' (GUI prefill) ─────
    def _stage_defaults():
        from ga_constraints import PIPELINE_CONSTRAINTS, normalize_min_behavioral
        out = {}
        for key, cons in PIPELINE_CONSTRAINTS.items():
            d = {}
            for ek in _EDITABLE_KEYS:
                if ek not in cons:
                    continue
                if ek == 'min_behavioral':
                    d[ek] = [[n, list(v)] for n, v in
                             normalize_min_behavioral(cons[ek])]
                elif ek == 'mutually_exclusive':
                    d[ek] = [list(g) for g in cons[ek]]
                else:
                    d[ek] = cons[ek]
            if d:
                out[key] = d
        return out
    defaults = _safe(_stage_defaults, {})
    for key, d in defaults.items():
        if key in stages:
            stages[key]['defaults'] = d

    # ── Default person↔land affinities, filtered to a valid land attractor ───
    # (the person side may be a ptype_* dummy OR a continuous driver like incp;
    #  its column is guarded when the interaction is materialised, so we only
    #  require the land attractor to exist.)
    land_set = set(land_attrs)
    default_pl = [
        {**pair, 'pairing': True, 'interaction': True}
        for pair in _DEFAULT_PTYPE_LAND
        if pair['land'] in land_set
    ]

    return {
        'generated': _dt.datetime.now().isoformat(timespec='seconds'),
        'stages': stages,
        'ptypes': ptypes,
        'person_drivers': sorted(set(_PERSON_DRIVERS)),
        'land_attributes': sorted(land_attrs),
        'person_attributes': person_attrs,
        'default_ptype_land': default_pl,
        'constraint_types': [
            'force_include', 'force_exclude', 'mutually_exclusive',
            'min_behavioral', 'sign_constraints', 'ptype_land',
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='constraint_schema.json',
                    help='output JSON path (default: constraint_schema.json)')
    ap.add_argument('--no-data', action='store_true',
                    help='skip loading real data; use static candidate pools only')
    args = ap.parse_args()

    schema = build_schema(use_data=not args.no_data)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(schema, fh, indent=2)
    n_stages = len(schema['stages'])
    print(f"\nWrote {args.out}: {n_stages} stages, {len(schema['ptypes'])} ptypes, "
          f"{len(schema['land_attributes'])} land attributes, "
          f"{len(schema['person_attributes'])} person attributes.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
