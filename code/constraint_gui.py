#!/usr/bin/env python
"""
constraint_gui.py  —  Model constraint authoring GUI (Streamlit)
================================================================
Point-and-click editor for the ADDITIONAL model constraints the GA pipeline
reads at run time. It writes a JSON overlay (``ga_constraints_user.json``) that
``ga_constraints.get_constraints()`` merges on top of the hard-coded pipeline
constraints — so whatever you Generate here changes the next run, no code edits.

What you can author, per stage/model:
  • Must-have variables            → force_include
  • Never-use variables            → force_exclude
  • Mutually-exclusive groups      → mutually_exclusive  (at most one of a set)
  • "At least N of …" groups       → min_behavioral
  • Sign constraints (≤0 / ≥0)     → sign_constraints
And globally:
  • ptype ↔ land-attribute affinity → pairing rule (require the land attractor)
    AND a ptype×land interaction gene the GA can select.

The dropdowns are populated from ``constraint_schema.json`` (produced by
``build_constraint_schema.py``), so they always reflect your real model's
variables and your data's attributes.

Run
---
    pip install streamlit          # once
    streamlit run constraint_gui.py
Then edit and press **Generate overlay**.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, 'constraint_schema.json')
OVERLAY_PATH = os.path.join(HERE, 'ga_constraints_user.json')

SIGN_LABELS = {'(none)': None, 'negative (≤ 0)': 'neg', 'positive (≥ 0)': 'pos'}
SIGN_FROM_CODE = {None: '(none)', 'neg': 'negative (≤ 0)', 'pos': 'positive (≥ 0)'}


# ---------------------------------------------------------------------------
# Load schema + any existing overlay
# ---------------------------------------------------------------------------
def _load_json(path, default):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Variable categorisation + colour (for the catalog view)
# ---------------------------------------------------------------------------
# Category → (colour, help). Colour is a background chip colour that reads on
# both light and dark themes.
CATEGORY_STYLE = {
    'Land-use / attractor': ('#2e7d32', 'zone employment, enrolment, facilities, size terms'),
    'Accessibility':        ('#0277bd', 'zone reachability (walk/transit/auto acc)'),
    'Impedance / cost':     ('#c62828', 'travel time, distance, tolls, cost, crash'),
    'Person / behaviour':   ('#6a1b9a', 'demographics, income, vehicles, ptype, work status'),
    'Interaction':          ('#ad1457', 'products of two drivers (x)'),
    'Structural':           ('#455a64', 'ASCs, nest scales, constants, thresholds, size offset'),
    'Generic (mode)':       ('#ef6c00', 'one shared effect across modes'),
    'Other':                ('#616161', ''),
}


def categorize(name: str) -> str:
    n = str(name)
    nl = n.lower()
    if n.endswith('_generic') or n.startswith(('c_incp_gen', 'c_veh_gen', 'c_crash_gen', 'c_age_active_gen')):
        return 'Generic (mode)'
    if '_x_' in nl or nl.endswith('_dist') and 'x_' in nl:
        return 'Interaction'
    if n.startswith(('ASC_', 'Mu_')) or n in ('const', 'offset', 'log_correction',
                                              'c_size_term', 'size_term') or n.startswith('threshold'):
        return 'Structural'
    if 'acc' == nl[-3:] or '_acc' in nl or nl.startswith('c_walk_acc') or nl.startswith('c_transit_acc') \
            or nl.startswith('c_auto') or nl.startswith('c_noveh'):
        return 'Accessibility'
    if any(k in n for k in ('SOVTOLL', 'WLK_', 'HOV', 'DIST', 'CRASH')) \
            or nl.startswith(('c_ivt', 'c_cost', 'c_toll', 'c_dist', 'c_walk_time', 'c_bike_time',
                              'c_xwait', 'c_drv', 'c_crash', 'c_log_dist')):
        return 'Impedance / cost'
    if any(k in n for k in ('EMP', 'TOTEMP', 'Education', 'Primary', 'Secondary', 'residential',
                            'commercial', 'industrial', 'government', 'retail', 'Entertainment',
                            'Shopping', 'EATOUT', 'Healthcare', 'recreation', 'sport', 'park',
                            'community', 'civic', 'TOTPOP', 'TOTHH', 'size_term', 'aggl', 'construction',
                            'childcare', 'HSENROLL', 'AGE0519', 'is_cbd', 'transit_walk_acc')) \
            or nl.startswith(('c_indp', 'c_aggl', 'c_nonmand', 'c_edu_', 'c_cbd', 'c_totemp')):
        return 'Land-use / attractor'
    if any(k in nl for k in ('incp', 'income', 'age', 'sexp', 'female', 'veh', 'nprd', 'hh_',
                             'ptype', 'workstat', 'studying', 'occp', 'wfh', 'licence', 'carlicence',
                             'mandatory', 'bikes', 'ebikes', 'escoot', 'student', 'worker', 'child',
                             'n_workers', 'hhsize')):
        return 'Person / behaviour'
    return 'Other'


def chip(name: str) -> str:
    col = CATEGORY_STYLE.get(categorize(name), CATEGORY_STYLE['Other'])[0]
    return (f'<span style="background:{col};color:#fff;padding:1px 7px;border-radius:9px;'
            f'font-size:0.78em;margin:2px;display:inline-block;white-space:nowrap;">{name}</span>')


st.set_page_config(page_title='Model Constraint Editor', page_icon='🧩', layout='wide')
st.title('🧩 Model Constraint Editor')
st.caption('Author extra constraints for the GA pipeline. **Generate** writes '
           '`ga_constraints_user.json`, which the next run reads automatically.')

schema = _load_json(SCHEMA_PATH, None)
if not schema or not schema.get('stages'):
    st.error(
        'No `constraint_schema.json` found (or it is empty). Build it first so '
        'the dropdowns match your real model + data:\n\n'
        '```\npython build_constraint_schema.py            # real data\n'
        'python build_constraint_schema.py --no-data   # fast, static pools\n```')
    st.stop()

stages = schema['stages']
ptypes = schema.get('ptypes', [])
person_drivers = schema.get('person_drivers', [])
# person side of an affinity may be a ptype dummy OR a continuous driver (incp…)
person_options = list(dict.fromkeys(list(ptypes) + list(person_drivers)))
land_attrs = schema.get('land_attributes', [])
st.caption(f'Schema generated {schema.get("generated", "?")} · '
           f'{len(stages)} stages · {len(ptypes)} ptypes · {len(land_attrs)} land attributes')


# ===========================================================================
# Run plan — generate the qsub command for the parallel staged run + rejoin
# ===========================================================================
with st.expander('🚀 Run plan — parallel stages → rejoin → prediction (generate the qsub command)',
                 expanded=False):
    st.caption('Runs all GA stages **in parallel** as separate PBS jobs, then a '
               'merge job (depends on all stages), then the prediction job '
               '(depends on the merge) — via `job_submit_ga_parallel_chain.pbs`. '
               'Your generated `ga_constraints_user.json` is read automatically on '
               'the HPC as long as it sits in the code folder there.')
    rp = st.columns(3)
    with rp[0]:
        run_tag = st.text_input('RUN_TAG', value='gui_run')
        estimator = st.selectbox('GA_ESTIMATOR', ['sa_bandit', 'default', 'both', 'hybrid'])
        budget = st.selectbox('GA_BUDGET', ['standard', 'fast', 'quick', 'thorough'], index=0)
        conda_env = st.text_input('CONDA_ENV', value='zigenv')
    with rp[1]:
        stage_ncpus = st.number_input('STAGE_NCPUS', 1, 64, 8)
        stage_mem = st.text_input('STAGE_MEM', value='128GB')
        stage_wall = st.text_input('STAGE_WALLTIME', value='36:00:00')
        parallel_segs = st.number_input('GA_PARALLEL_SEGMENTS', 1, 16, 4)
    with rp[2]:
        pred_ncpus = st.number_input('PRED_NCPUS', 1, 64, 8)
        pred_mem = st.text_input('PRED_MEM', value='250GB')
        pred_wall = st.text_input('PRED_WALLTIME', value='24:00:00')
        pval = st.text_input('GA_SIG_PVAL_THRESH', value='0.05')
    rp2 = st.columns(3)
    with rp2[0]:
        early_stop = st.selectbox('GA_EARLY_STOP', ['1', '0'], index=0)
    with rp2[1]:
        s1_ridge = st.text_input('GA_S1_RIDGE (stage-1 shrinkage)', value='2.0')
    with rp2[2]:
        hpc_host = st.text_input('HPC login host', value='ahernz@lyra.qut.edu.au')
    code_dir = st.text_input('HPC code directory',
                             value='/mnt/hpccs01/home/ahernz/test_runs_tours/code')

    _vlist = (
        f'RUN_TAG={run_tag},GA_ESTIMATOR={estimator},GA_BUDGET={budget},'
        f'CONDA_ENV={conda_env},GA_PARALLEL_SEGMENTS={int(parallel_segs)},'
        f'STAGE_NCPUS={int(stage_ncpus)},STAGE_MEM={stage_mem},STAGE_WALLTIME={stage_wall},'
        f'PRED_NCPUS={int(pred_ncpus)},PRED_MEM={pred_mem},PRED_WALLTIME={pred_wall},'
        f'GA_SIG_PVAL_THRESH={pval},GA_EARLY_STOP={early_stop},GA_S1_RIDGE={s1_ridge}'
    )
    qsub_cmd = f'qsub -v "{_vlist}" job_submit_ga_parallel_chain.pbs'
    ssh_cmd = f'ssh {hpc_host} \'cd {code_dir} && {qsub_cmd}\''

    st.markdown('**1) The `qsub` command** (run on the HPC login node):')
    st.code(qsub_cmd, language='bash')
    st.markdown('**2) …or submit it over SSH from here** (you will be prompted for '
                'your password — it is never stored):')
    st.code(ssh_cmd, language='bash')
    st.info('Submits 6 stage jobs in parallel → merge (afterok all) → prediction '
            '(afterok merge). Check status on the HPC with `qstat -u <user>`. '
            'Make sure `ga_constraints_user.json` (Generate below) is synced to the '
            'HPC code folder so your constraints take effect.')

existing = _load_json(OVERLAY_PATH, {})


def _defaults_source() -> dict:
    """A source dict (same shape as the overlay) built from the schema's built-in
    'defaults' per stage plus the default ptype↔land affinities — so the GUI can
    open pre-populated with the current constraints."""
    src = {}
    for sk in stages:
        d = stages[sk].get('defaults', {})
        if d:
            src[sk] = d
    src['ptype_land'] = list(schema.get('default_ptype_land', []))
    return src


def _seed_state(source: dict) -> None:
    """(Re)seed all editor widgets from a source dict (overlay or defaults)."""
    for sk in stages:
        ov = source.get(sk, {}) if isinstance(source, dict) else {}
        st.session_state[f'fi_{sk}'] = list(ov.get('force_include', []))
        st.session_state[f'fe_{sk}'] = list(ov.get('force_exclude', []))
        st.session_state[f'mutex_{sk}'] = [list(g) for g in ov.get('mutually_exclusive', [])]
        mb = ov.get('min_behavioral', [])
        if mb and isinstance(mb, (list, tuple)) and len(mb) == 2 and isinstance(mb[0], (int, float)):
            mb = [mb]  # single (n,[vars]) → list form
        st.session_state[f'minb_{sk}'] = [[int(g[0]), list(g[1])] for g in mb] if mb else []
        st.session_state[f'sign_{sk}'] = dict(ov.get('sign_constraints', {}))
        st.session_state[f'frand_{sk}'] = list((ov.get('force_random', {}) or {}).keys())
        st.session_state[f'nrand_{sk}'] = list(ov.get('never_random', []))
        gui = (source.get('_gui', {}) or {}).get(sk, {}) if isinstance(source, dict) else {}
        st.session_state[f'groups_{sk}'] = {k: dict(v) if isinstance(v, dict) else {'members': list(v), 'self': False}
                                            for k, v in (gui.get('groups', {}) or {}).items()}
        st.session_state[f'conns_{sk}'] = [dict(c) for c in (gui.get('conns', []) or [])]
    st.session_state['ptype_land'] = list(source.get('ptype_land', [])) if isinstance(source, dict) else []


# Seed once: from a previously-saved overlay if it exists, otherwise from the
# built-in defaults + default pairings (so the GUI is never blank).
if 'seeded' not in st.session_state:
    st.session_state.seeded = True
    seed_from = existing if (isinstance(existing, dict) and existing) else _defaults_source()
    _seed_state(seed_from)

# Sidebar: reset to the built-in defaults at any time.
if st.sidebar.button('↩️ Reset to built-in defaults'):
    _seed_state(_defaults_source())
    st.rerun()


# ---------------------------------------------------------------------------
# Per-stage editor
# ---------------------------------------------------------------------------
stage_keys = list(stages.keys())
sk = st.sidebar.radio('Model / stage', stage_keys,
                      format_func=lambda k: stages[k].get('label', k))
sinfo = stages[sk]
names = sinfo.get('candidates', [])
coeffs = sinfo.get('coefficients', names)
name_kind = sinfo.get('name_kind', 'feature')

st.header(stages[sk].get('label', sk))
st.caption(f'Constraints here operate on **{name_kind} names** '
           f'({len(names)} available). Segments: {sinfo.get("segments", "—")}')

# ── Variable catalog: see everything you have to play with, coloured ─────────
with st.expander(f'📚 Variable catalog — all {len(names)} variables for this stage '
                 '(coloured by category)', expanded=False):
    # colour legend
    st.markdown('**Legend:** ' + '  '.join(
        f'<span style="background:{c};color:#fff;padding:1px 7px;border-radius:9px;'
        f'font-size:0.78em;">{cat}</span>' for cat, (c, _h) in CATEGORY_STYLE.items()),
        unsafe_allow_html=True)
    _filter = st.text_input('filter', key=f'catfilter_{sk}',
                            placeholder='type to filter (e.g. income, acc, ptype_)')
    shown = [n for n in names if _filter.lower() in n.lower()] if _filter else list(names)
    st.caption(f'{len(shown)} of {len(names)} shown, grouped by category.')
    # group by category, render coloured chips per category
    by_cat: dict = {}
    for n in shown:
        by_cat.setdefault(categorize(n), []).append(n)
    for cat in CATEGORY_STYLE:
        members = sorted(by_cat.get(cat, []))
        if not members:
            continue
        col, _hlp = CATEGORY_STYLE[cat]
        st.markdown(
            f'<div style="margin-top:6px"><b style="color:{col}">{cat}</b> '
            f'<span style="color:#888;font-size:0.8em">({len(members)}) {_hlp}</span><br>'
            + ' '.join(chip(m) for m in members) + '</div>',
            unsafe_allow_html=True)
    st.markdown('---')
    # quick multiselect to exclude straight from the catalog
    _cat_excl = st.multiselect('🚫 turn these OFF (never consider)', shown,
                               default=[v for v in st.session_state[f'fe_{sk}'] if v in shown],
                               key=f'catexcl_{sk}')
    _outside = [v for v in st.session_state[f'fe_{sk}'] if v not in shown]
    st.session_state[f'fe_{sk}'] = sorted(set(_cat_excl) | set(_outside))

col1, col2 = st.columns(2)
with col1:
    st.session_state[f'fi_{sk}'] = st.multiselect(
        '✅ Must-have (force_include)', names,
        default=[v for v in st.session_state[f'fi_{sk}'] if v in names],
        help='These are always kept in the model.')
with col2:
    st.session_state[f'fe_{sk}'] = st.multiselect(
        '🚫 Never-use (force_exclude)', names,
        default=[v for v in st.session_state[f'fe_{sk}'] if v in names],
        help='These are never allowed into the model. Same list as the catalog toggle above.')

# ── Mutually-exclusive groups ────────────────────────────────────────────────
st.subheader('↔️ Mutually-exclusive groups')
st.caption('At most ONE variable from each group may be active (competing encodings).')
mutex = st.session_state[f'mutex_{sk}']
_rm = None
for i, grp in enumerate(mutex):
    c1, c2 = st.columns([10, 1])
    mutex[i] = c1.multiselect(f'group {i + 1}', names,
                              default=[v for v in grp if v in names], key=f'mutex_{sk}_{i}')
    if c2.button('🗑', key=f'mutex_del_{sk}_{i}'):
        _rm = i
if _rm is not None:
    mutex.pop(_rm)
    st.rerun()
if st.button('➕ add mutex group', key=f'mutex_add_{sk}'):
    mutex.append([])
    st.rerun()

# ── Reusable groups + connect-a-variable mutual exclusivity ───────────────────
st.subheader('🔗 Reusable groups & connections')
st.caption('Define a named group of variables once, then **connect** any variable '
           'to it so that variable is mutually exclusive with every member of the '
           'group (expanded into mutex groups on Generate).')
groups = st.session_state[f'groups_{sk}']    # {name: {'members':[...], 'self':bool}}
conns = st.session_state[f'conns_{sk}']       # [{'var':.., 'group':..}]

gcolA, gcolB = st.columns(2)
with gcolA:
    st.markdown('**Define / update a group**')
    with st.form(f'grp_form_{sk}', clear_on_submit=True):
        gname = st.text_input('group name', placeholder='e.g. income_encodings')
        gmembers = st.multiselect('members', names, key=f'grp_mem_{sk}')
        gself = st.checkbox('also make the group itself mutually exclusive '
                            '(at most one member)', value=True)
        if st.form_submit_button('💾 save group') and gname and gmembers:
            groups[gname] = {'members': sorted(set(gmembers)), 'self': bool(gself)}
    if groups:
        st.markdown('**Groups**')
        _rm_g = None
        for gn, gd in list(groups.items()):
            st.markdown(f'`{gn}` {"⛓" if gd.get("self") else ""} — '
                        + ' '.join(chip(m) for m in gd.get('members', [])),
                        unsafe_allow_html=True)
            if st.button(f'🗑 delete {gn}', key=f'grp_del_{sk}_{gn}'):
                _rm_g = gn
        if _rm_g is not None:
            groups.pop(_rm_g, None)
            # drop connections that referenced it
            st.session_state[f'conns_{sk}'] = [c for c in conns if c.get('group') != _rm_g]
            st.rerun()
with gcolB:
    st.markdown('**Connect a variable → group (mutually exclusive)**')
    if not groups:
        st.info('Define a group first.')
    else:
        with st.form(f'conn_form_{sk}', clear_on_submit=True):
            cvars = st.multiselect('variable(s)', names, key=f'conn_var_{sk}')
            cgroup = st.selectbox('mutually exclusive with group', list(groups.keys()))
            if st.form_submit_button('🔗 connect') and cvars and cgroup:
                for v in cvars:
                    if {'var': v, 'group': cgroup} not in conns:
                        conns.append({'var': v, 'group': cgroup})
    if conns:
        st.markdown('**Connections**')
        _rm_c = None
        for i, c in enumerate(conns):
            st.write(f"{i + 1}. `{c['var']}`  ⇎  group `{c['group']}`")
            if st.button('remove', key=f'conn_del_{sk}_{i}'):
                _rm_c = i
        if _rm_c is not None:
            conns.pop(_rm_c)
            st.rerun()

# ── min_behavioral groups ────────────────────────────────────────────────────
st.subheader('📎 "At least N of …" groups (min_behavioral)')
st.caption('Require at least N variables from a group to be present (semantic pairing).')
minb = st.session_state[f'minb_{sk}']
_rm = None
for i, (n, grp) in enumerate(minb):
    c0, c1, c2 = st.columns([2, 9, 1])
    n_new = c0.number_input('N', min_value=1, max_value=max(1, len(names)), value=int(n),
                            key=f'minb_n_{sk}_{i}')
    grp_new = c1.multiselect(f'of these (group {i + 1})', names,
                             default=[v for v in grp if v in names], key=f'minb_g_{sk}_{i}')
    minb[i] = [int(n_new), grp_new]
    if c2.button('🗑', key=f'minb_del_{sk}_{i}'):
        _rm = i
if _rm is not None:
    minb.pop(_rm)
    st.rerun()
if st.button('➕ add min-behavioral group', key=f'minb_add_{sk}'):
    minb.append([1, []])
    st.rerun()

# ── Sign constraints ─────────────────────────────────────────────────────────
st.subheader('➖➕ Sign constraints')
st.caption('Force a coefficient ≤ 0 (deterrent) or ≥ 0 (attractor).')
signs = st.session_state[f'sign_{sk}']
sc1, sc2, sc3 = st.columns([6, 4, 1])
_new_sign_var = sc1.selectbox('coefficient', ['—'] + coeffs, key=f'sign_pick_{sk}')
_new_sign_dir = sc2.selectbox('sign', list(SIGN_LABELS.keys())[1:], key=f'sign_dir_{sk}')
if sc3.button('add', key=f'sign_add_{sk}') and _new_sign_var != '—':
    signs[_new_sign_var] = SIGN_LABELS[_new_sign_dir]
    st.rerun()
if signs:
    _rm_sign = None
    for v, code in list(signs.items()):
        d1, d2 = st.columns([10, 1])
        d1.write(f'`{v}` → **{SIGN_FROM_CODE.get(code, code)}**')
        if d2.button('🗑', key=f'sign_del_{sk}_{v}'):
            _rm_sign = v
    if _rm_sign is not None:
        signs.pop(_rm_sign, None)
        st.rerun()

# ── Random (mixed) coefficients ──────────────────────────────────────────────
st.subheader('🎲 Random (mixed-model) coefficients')
st.caption('Which coefficients the search treats as **random** (taste variation) '
           'vs **fixed**. Continuous drivers make good random terms; 0/1 dummies, '
           'constants and structural terms should stay fixed.')
rc1, rc2 = st.columns(2)
with rc1:
    st.session_state[f'frand_{sk}'] = st.multiselect(
        '🎲 Search as RANDOM (force_random)', names,
        default=[v for v in st.session_state[f'frand_{sk}'] if v in names],
        help='These get a random (normal) coefficient when selected.')
with rc2:
    st.session_state[f'nrand_{sk}'] = st.multiselect(
        '📌 Always FIXED (never_random)', names,
        default=[v for v in st.session_state[f'nrand_{sk}'] if v in names],
        help='These are never given a random coefficient (structure / dummies).')


# ---------------------------------------------------------------------------
# Global: ptype ↔ land-attribute affinities
# ---------------------------------------------------------------------------
st.sidebar.markdown('---')
st.sidebar.subheader('🧭 person ↔ land affinity')
st.sidebar.caption('Global — applies to destination choice. Person side may be a '
                   'ptype (e.g. student) OR a driver (e.g. incp, vehrd).')
pl = st.session_state['ptype_land']
with st.sidebar.form('add_pl', clear_on_submit=True):
    p = st.selectbox('person attribute', person_options)
    la = st.selectbox('land attractor', land_attrs)
    segs = st.multiselect('segments', ['work', 'edu', 'nonmand'], default=['nonmand'])
    pairing = st.checkbox('pairing rule (require the land attractor)', value=True)
    interaction = st.checkbox('interaction term (ptype × land gene)', value=True)
    if st.form_submit_button('➕ add affinity') and p and la:
        pl.append({'ptype': p, 'land': la, 'segments': segs,
                   'pairing': pairing, 'interaction': interaction})

if pl:
    st.sidebar.write('**Current affinities**')
    _rm_pl = None
    for i, a in enumerate(pl):
        st.sidebar.write(f"{i + 1}. `{a['ptype']}` → `{a['land']}` "
                         f"{a.get('segments', [])} "
                         f"{'⛓' if a.get('pairing') else ''}{'✳️' if a.get('interaction') else ''}")
        if st.sidebar.button('remove', key=f'pl_del_{i}'):
            _rm_pl = i
    if _rm_pl is not None:
        pl.pop(_rm_pl)
        st.rerun()


# ---------------------------------------------------------------------------
# Assemble + Generate
# ---------------------------------------------------------------------------
def _assemble_overlay() -> dict:
    overlay: dict = {}
    gui_state: dict = {}
    for k in stage_keys:
        sd: dict = {}
        fi = [v for v in st.session_state[f'fi_{k}'] if v]
        fe = [v for v in st.session_state[f'fe_{k}'] if v]
        mutex_groups = [sorted(set(g)) for g in st.session_state[f'mutex_{k}'] if len(g) >= 2]
        # ── expand reusable groups + connections into concrete mutex groups ──
        groups_k = st.session_state.get(f'groups_{k}', {})
        conns_k = st.session_state.get(f'conns_{k}', [])
        for gn, gd in groups_k.items():
            members = list(gd.get('members', []))
            if gd.get('self') and len(members) >= 2 and sorted(set(members)) not in mutex_groups:
                mutex_groups.append(sorted(set(members)))
        for c in conns_k:
            members = list(groups_k.get(c.get('group'), {}).get('members', []))
            grp = sorted(set([c.get('var')] + members))
            if len(grp) >= 2 and grp not in mutex_groups:
                mutex_groups.append(grp)
        if groups_k or conns_k:
            gui_state[k] = {'groups': groups_k, 'conns': conns_k}
        minb_groups = [[int(n), sorted(set(g))] for n, g in st.session_state[f'minb_{k}'] if g]
        signs = {v: c for v, c in st.session_state[f'sign_{k}'].items() if c in ('neg', 'pos')}
        frand = {v: 'n' for v in st.session_state.get(f'frand_{k}', []) if v}
        nrand = [v for v in st.session_state.get(f'nrand_{k}', []) if v]
        if fi:
            sd['force_include'] = fi
        if fe:
            sd['force_exclude'] = fe
        if mutex_groups:
            sd['mutually_exclusive'] = mutex_groups
        if minb_groups:
            sd['min_behavioral'] = minb_groups
        if signs:
            sd['sign_constraints'] = signs
        if frand:
            sd['force_random'] = frand
        if nrand:
            sd['never_random'] = nrand
        if sd:
            overlay[k] = sd

    # ptype↔land: pairing → a require-one land group on stage5_dest; keep the raw
    # section too (drives the ptype×land interaction genes in the backend).
    pairs = st.session_state['ptype_land']
    if pairs:
        overlay['ptype_land'] = pairs
        pairing_lands = sorted({a['land'] for a in pairs if a.get('pairing') and a.get('land')})
        if pairing_lands:
            dest = overlay.setdefault('stage5_dest', {})
            mb = dest.get('min_behavioral', [])
            grp = [1, pairing_lands]
            if grp not in mb:
                mb = list(mb) + [grp]
            dest['min_behavioral'] = mb

    # GUI-only state (reusable groups + connections) so the editor can reload
    # them; the pipeline ignores the '_gui' key (only stage dicts + ptype_land
    # are consumed by ga_constraints).
    if gui_state:
        overlay['_gui'] = gui_state
    return overlay


st.markdown('---')
overlay = _assemble_overlay()
gcol1, gcol2 = st.columns([1, 3])
with gcol1:
    if st.button('🚀 Generate overlay', type='primary', use_container_width=True):
        with open(OVERLAY_PATH, 'w', encoding='utf-8') as fh:
            json.dump(overlay, fh, indent=2)
        st.success(f'Wrote {OVERLAY_PATH} at {datetime.now():%H:%M:%S}. '
                   'The next pipeline run will read it automatically.')
    if os.path.exists(OVERLAY_PATH) and st.button('🗑 Clear overlay file',
                                                  use_container_width=True):
        os.remove(OVERLAY_PATH)
        st.warning('Overlay removed — the pipeline reverts to built-in constraints.')
with gcol2:
    st.write('**Overlay preview** (what Generate writes):')
    st.json(overlay)
