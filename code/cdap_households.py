"""
cdap_households.py  —  ActivitySim-style household coordination
================================================================
Self-contained module — always testable standalone (``python cdap_households.py``
runs a demo on synthetic households).  It is wired into the live pipeline via
``pipeline_logger.cdap()``:
  * ``CDAP_HOUSEHOLD_COORD=1`` → coordinate_household_cdap (M/N/H coordination).
  * ``CDAP_HOUSEHOLD_COORD=joint`` or ``JOINT_TOURS=1`` → full structural path:
    coordination + joint_tour_frequency + joint_tour_participation +
    joint_tour_purpose, then reconcile_joint_tours (Stage-1 purpose counts) and
    tag_joint_activities (Stage-3 rows).  The joint-tour PURPOSE is how the
    household CDAP is incorporated into the activity-purpose dimension.

What it models (mirrors ActivitySim's household sub-models):
  1. coordinate_household_cdap  — a coordinated Daily Activity Pattern (Mandatory
     / NonMandatory / Home) per person, with intra-household interactions so a
     household's members' patterns are consistent (e.g. a non-working adult at
     home raises the chance a child is escorted rather than travelling alone).
  2. joint_tour_frequency       — how many JOINT (whole-/part-household) tours a
     household makes.
  3. joint_tour_participation    — which members travel together on each joint
     tour (adults-only / with-children / mixed composition).
  3b. joint_tour_purpose          — STRUCTURAL: co-assign an ACTIVITY PURPOSE to
     each joint tour via a joint logit over the Stage-1 purpose set, conditioned
     on composition and the participants' coordinated DAPs (M-heavy households
     shift toward post-work purposes; children shift toward escort). This is how
     the household CDAP is incorporated into the activity-purpose dimension.
     reconcile_joint_tours() then folds these shared purposes into the per-person
     Stage-1 counts (n_joint_<purpose>_out flags + optional inflation), and
     tag_joint_activities() marks the shared activity rows downstream.
  4. allocate_household_vehicles — assign the household's vehicles to auto tours
     by AVAILABILITY: a tour gets a car only if one is free during its time
     window; otherwise its probability of an auto trip falls (→ passenger, or a
     non-car mode). This consumes the Tier-1 time windows from stage-6
     (stage6_plan_finalisation._resolve_time_conflicts) so choices deplete a
     shared, finite resource — exactly the "certain things become unavailable
     based on all the choices" behaviour.

Design contract (so plugging in is mechanical):
  * Everything operates on plain pandas DataFrames and is a pure function
    (no global state, no file/network side effects).
  * Reproducible: every stochastic step is seeded (per household / per person).
  * The coefficient tables are heuristic priors marked "TODO(estimate)"; replace
    them with HHTS-estimated coefficients without changing any call site.

PLUG-IN POINTS (implemented in pipeline_logger.cdap()):
  A. After person profiles + CDAP (cdap.predict_cdap) and BEFORE tour frequency
     (stage1_activity_frequency): coordinate_household_cdap overwrites the
     per-person DAP with the coordinated one; then joint_tour_frequency /
     joint_tour_participation / joint_tour_purpose generate shared tours with
     purposes, which reconcile_joint_tours folds into the Stage-1 purpose counts
     (predict_stage1) and tag_joint_activities marks on Stage-3 activity rows
     (predict_stage3).
  B. After stage-4 scheduling (STARTIME/ENDTIME assigned) and the Tier-1
     conflict resolver: allocate_household_vehicles sets each auto tour's
     vehicle id / downgrade mode when no vehicle is free.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DAP_CLASSES = ('M', 'N', 'H')   # Mandatory, NonMandatory, Home

# Activity purposes a joint (household) tour can take. This is the household
# co-assignment dimension: the same purpose set Stage 1 predicts per person, so
# a joint tour's purpose is directly comparable to n_{purpose}_out counts.
JOINT_PURPOSES = ('escort', 'shopping', 'social', 'eat', 'othdiscr')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class HouseholdCdapConfig:
    hh_id_col: str = 'HHID'
    person_id_col: str = 'person_id'
    ptype_col: str = 'ptype'
    vehicles_col: str = 'vehrd'
    seed: int = 42

    # DAP utility priors by person type (M, N, H). TODO(estimate) from HHTS.
    dap_utility: dict = field(default_factory=lambda: {
        'FullTimeWorker':    {'M': 2.2, 'N': 0.3, 'H': 0.0},
        'PartTimeWorker':    {'M': 1.4, 'N': 0.6, 'H': 0.0},
        'UniversityStudent': {'M': 1.6, 'N': 0.7, 'H': 0.0},
        'WorkingStudent':    {'M': 1.8, 'N': 0.6, 'H': 0.0},
        'SchoolChild':       {'M': 1.9, 'N': 0.4, 'H': 0.0},
        'Preschool':         {'M': 0.2, 'N': 0.9, 'H': 0.6},
        'Retired':           {'M': 0.0, 'N': 1.1, 'H': 0.5},
        'NonWorker':         {'M': 0.0, 'N': 1.0, 'H': 0.4},
    })
    # Household interaction adjustments (added to a person's DAP utility).
    # TODO(estimate): these encode coordination.
    adult_home_boosts_child_N: float = 0.5   # a home adult → child more N (escortable)
    more_workers_boost_M: float = 0.15        # each additional worker → slightly more M

    # ── ActivitySim-style 2-way interaction coefficients ─────────────────────
    # Added to a household pattern COMBINATION for every member pair, keyed by
    # (pair_class, {sorted pattern pair}). pair_class ∈ {AA adult-adult,
    # AC adult-child, CC child-child}. These are what make a household's members
    # co-determine their patterns (a true joint logit, not independent draws).
    # Priors below are behaviourally signed; TODO(estimate) from HHTS overwrites
    # them via cdap_households_estimation. Absent (class,pair) keys default to 0.
    n_cdap_persons: int = 5   # members modelled JOINTLY (ActivitySim cap); rest
                              # are drawn conditionally on the modelled members.
    interaction: dict = field(default_factory=lambda: {
        # both mandatory — schedule alignment among workers / students / pupils
        ('AA', 'M', 'M'): 0.30, ('AC', 'M', 'M'): 0.20, ('CC', 'M', 'M'): 0.40,
        # both non-mandatory — joint discretionary / shopping / social travel
        ('AA', 'N', 'N'): 0.35, ('AC', 'N', 'N'): 0.25, ('CC', 'N', 'N'): 0.20,
        # both home — shared at-home day
        ('AA', 'H', 'H'): 0.20,
        # escort: an at-home / non-mandatory adult lets a child travel N (escorted)
        ('AC', 'H', 'N'): 0.45, ('AC', 'M', 'N'): 0.15,
        # supervision: child home while every adult is mandatory is discouraged,
        # and two children home alone together even more so
        ('AC', 'M', 'H'): -0.30, ('CC', 'H', 'H'): -0.25,
    })
    # Joint-tour frequency: base Poisson rate + per-feature increments.
    joint_base_rate: float = 0.25
    joint_rate_per_child: float = 0.20
    joint_rate_per_adult: float = 0.05
    joint_rate_if_multi_vehicle: float = 0.10

    # ── Joint-tour PURPOSE (structural: household co-assigns a purpose to each
    #    joint tour, ActivitySim-style, so the household CDAP is incorporated
    #    into the activity-purpose dimension instead of only M/N/H).
    #    joint_purpose_utility holds baseline utilities on the logit scale keyed
    #    by (composition, purpose); the per-member adjustment tables co-determine
    #    the purpose with the participants' coordinated DAPs (M-heavy households
    #    shift toward post-work purposes, child participants shift toward
    #    escort/accompaniment).  TODO(estimate) from HHTS via
    #    cdap_households_estimation.estimate_joint_purpose.
    joint_purpose_utility: dict = field(default_factory=lambda: {
        ('adults', 'social'):     0.30,
        ('adults', 'eat'):        0.15,
        ('adults', 'shopping'):  -0.10,
        ('adults', 'othdiscr'):  -0.50,
        ('mixed', 'escort'):      0.45,
        ('mixed', 'shopping'):    0.10,
        ('mixed', 'social'):     -0.20,
        ('mixed', 'eat'):        -0.30,
        ('children', 'escort'):   0.60,
        ('children', 'shopping'): 0.05,
        ('children', 'eat'):     -0.30,
    })
    joint_purpose_m_adjust: dict = field(default_factory=lambda: {
        # per mandatory-participant shift toward post-work joint purposes
        'eat': 0.40, 'social': 0.30, 'shopping': 0.10,
        'escort': -0.30, 'othdiscr': 0.05,
    })
    joint_purpose_child_adjust: dict = field(default_factory=lambda: {
        # per child participant pull toward accompaniment / family activities
        'escort': 0.60, 'shopping': 0.20, 'eat': -0.05,
        'social': -0.10, 'othdiscr': -0.05,
    })
    # When reconciling joint tours with Stage-1 purpose counts:
    #   True  → each participant gains one extra {purpose} activity (ActivitySim:
    #           joint tours are additional to individual non-mandatory tours).
    #   False → each participant's {purpose} count is raised to at least 1
    #           (the joint tour consumes/replaces one of their independent
    #           activities, so total purpose counts are NOT inflated). Overridable
    #           per-run via JOINT_TOURS_ADDITIVE=1.
    joint_tour_additive: bool = False


def _rng(seed: int, *keys) -> np.random.Generator:
    """Deterministic per-entity RNG."""
    h = hash((seed, *keys)) & 0x7FFFFFFF
    return np.random.default_rng(h)


def _softmax_choice(util: dict, rng: np.random.Generator) -> str:
    keys = list(util.keys())
    v = np.array([util[k] for k in keys], dtype=float)
    v = v - v.max()
    p = np.exp(v)
    p = p / p.sum()
    return keys[int(rng.choice(len(keys), p=p))]


# ---------------------------------------------------------------------------
# ActivitySim-style joint-logit helpers
# ---------------------------------------------------------------------------
_CHILD_PTYPES = ('SchoolChild', 'Preschool')


def _pair_class(is_child_i: bool, is_child_j: bool) -> str:
    """Adult-Adult / Adult-Child / Child-Child bucket for a member pair."""
    n_child = int(is_child_i) + int(is_child_j)
    return 'CC' if n_child == 2 else ('AC' if n_child == 1 else 'AA')


def _interaction_util(cfg: 'HouseholdCdapConfig', cls: str,
                      d_i: str, d_j: str) -> float:
    """Symmetric 2-way interaction lookup (pattern order doesn't matter)."""
    a, b = sorted((d_i, d_j))
    return float(cfg.interaction.get((cls, a, b),
                 cfg.interaction.get((cls, b, a), 0.0)))


def _person_dap_utilities(row, cfg: 'HouseholdCdapConfig', n_workers: int) -> dict:
    """Individual M/N/H utility for one person.

    Uses per-person model utilities if the caller attached them as columns
    cdap_util_M / cdap_util_N / cdap_util_H (so this coordination layer sits ON
    TOP of the fitted individual CDAP, ActivitySim-style); otherwise falls back
    to the ptype utility priors. The household-worker count enters as an
    individual-level composition covariate; pairwise coordination is handled by
    the interaction terms, not here.
    """
    if all(f'cdap_util_{d}' in row.index for d in DAP_CLASSES):
        u = {d: float(row[f'cdap_util_{d}']) for d in DAP_CLASSES}
    else:
        pt = row[cfg.ptype_col]
        u = dict(cfg.dap_utility.get(pt, cfg.dap_utility['NonWorker']))
    u['M'] += cfg.more_workers_boost_M * max(n_workers - 1, 0)
    return u


def _joint_cdap_draw(members: list, cfg: 'HouseholdCdapConfig',
                     rng: np.random.Generator) -> dict:
    """Draw a coordinated pattern for a set of household members via the joint
    logit over the Cartesian product of their {M,N,H} patterns:

        U(combo) = Σ_i u_i(d_i) + Σ_{i<j} interaction(class_ij, d_i, d_j)

    `members` is a list of (key, util_dict, is_child). Returns {key: pattern}.
    """
    import itertools
    n = len(members)
    keys      = [m[0] for m in members]
    utils     = [m[1] for m in members]
    children  = [m[2] for m in members]
    combos    = list(itertools.product(DAP_CLASSES, repeat=n))
    logits    = np.empty(len(combos), dtype=float)
    for c, combo in enumerate(combos):
        s = 0.0
        for i in range(n):
            s += utils[i][combo[i]]
        for i in range(n):
            for j in range(i + 1, n):
                s += _interaction_util(cfg, _pair_class(children[i], children[j]),
                                       combo[i], combo[j])
        logits[c] = s
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    combo = combos[int(rng.choice(len(combos), p=p))]
    return {keys[i]: combo[i] for i in range(n)}


# ---------------------------------------------------------------------------
# 1. Coordinated household CDAP
# ---------------------------------------------------------------------------
def coordinate_household_cdap(persons: pd.DataFrame,
                              cfg: Optional[HouseholdCdapConfig] = None) -> pd.DataFrame:
    """
    Assign a coordinated Daily Activity Pattern ('dap' ∈ {M,N,H}) to every
    person, conditioning each person's pattern on their household's composition.

    Returns a copy of `persons` with a new 'dap' column (and 'dap_coordinated'
    flag). Non-destructive.
    """
    cfg = cfg or HouseholdCdapConfig()
    df = persons.copy()
    if cfg.ptype_col not in df.columns:
        raise KeyError(f"coordinate_household_cdap: missing '{cfg.ptype_col}'")

    is_child  = df[cfg.ptype_col].isin(_CHILD_PTYPES)
    is_worker = df[cfg.ptype_col].isin(
        ['FullTimeWorker', 'PartTimeWorker', 'WorkingStudent'])

    daps = np.empty(len(df), dtype=object)
    for hh, grp in df.groupby(cfg.hh_id_col, sort=False):
        n_workers = int(is_worker.loc[grp.index].sum())
        hh_rng = _rng(cfg.seed, 'cdap', hh)
        # Build (index, util, is_child) for every member.
        members = [
            (i, _person_dap_utilities(grp.loc[i], cfg, n_workers),
             bool(is_child.loc[i]))
            for i in grp.index
        ]
        # ActivitySim caps the JOINTLY-modelled set (default 5). When a household
        # is larger, model the highest-priority members jointly (adults first,
        # then children) and draw the remainder conditionally — each remaining
        # member joins the joint set one at a time so it still co-determines with
        # the already-fixed members (keeps coordination, bounds 3^n blow-up).
        order = sorted(range(len(members)),
                       key=lambda k: (members[k][2],  # adults (False) first
                                      str(grp.at[members[k][0], cfg.person_id_col])))
        cap = max(1, int(cfg.n_cdap_persons))
        core = [members[k] for k in order[:cap]]
        drawn = _joint_cdap_draw(core, cfg, hh_rng)
        for k in order[cap:]:
            idx, util, child = members[k]
            # condition on the already-drawn members via their interaction terms
            cond = {d: util[d] for d in DAP_CLASSES}
            for dkey, dval in drawn.items():
                dchild = bool(is_child.loc[dkey])
                cls = _pair_class(child, dchild)
                for d in DAP_CLASSES:
                    cond[d] += _interaction_util(cfg, cls, d, dval)
            drawn[idx] = _softmax_choice(cond, hh_rng)
        for idx, pattern in drawn.items():
            daps[df.index.get_loc(idx)] = pattern
    df['dap'] = daps
    df['dap_coordinated'] = True
    return df


# ---------------------------------------------------------------------------
# 2. Joint tour frequency  (per household)
# ---------------------------------------------------------------------------
def joint_tour_frequency(persons: pd.DataFrame,
                         cfg: Optional[HouseholdCdapConfig] = None) -> pd.DataFrame:
    """
    Number of joint (household) tours per household, as a Poisson draw whose
    rate rises with children, adults and vehicle availability. Returns a
    DataFrame indexed by household id with 'n_joint_tours'.
    """
    cfg = cfg or HouseholdCdapConfig()
    is_child = persons[cfg.ptype_col].isin(['SchoolChild', 'Preschool'])
    rows = []
    for hh, grp in persons.groupby(cfg.hh_id_col, sort=False):
        n_child = int(is_child.loc[grp.index].sum())
        n_adult = len(grp) - n_child
        veh = float(grp[cfg.vehicles_col].iloc[0]) if cfg.vehicles_col in grp else 1.0
        rate = (cfg.joint_base_rate + cfg.joint_rate_per_child * n_child
                + cfg.joint_rate_per_adult * n_adult
                + (cfg.joint_rate_if_multi_vehicle if veh >= 2 else 0.0))
        n = int(_rng(cfg.seed, 'jtf', hh).poisson(max(rate, 0.0)))
        # a joint tour needs ≥2 members
        n = min(n, max(len(grp) - 1, 0))
        rows.append({cfg.hh_id_col: hh, 'n_joint_tours': n})
    return pd.DataFrame(rows).set_index(cfg.hh_id_col)


# ---------------------------------------------------------------------------
# 3. Joint tour participation / composition
# ---------------------------------------------------------------------------
def joint_tour_participation(persons: pd.DataFrame,
                             joint_freq: pd.DataFrame,
                             cfg: Optional[HouseholdCdapConfig] = None) -> pd.DataFrame:
    """
    Expand each household's joint tours into participant lists. Returns one row
    per (household, joint_tour_idx) with a 'participants' list of person_ids and
    a 'composition' label (adults / children / mixed).
    """
    cfg = cfg or HouseholdCdapConfig()
    is_child = persons[cfg.ptype_col].isin(['SchoolChild', 'Preschool'])
    out = []
    for hh, grp in persons.groupby(cfg.hh_id_col, sort=False):
        n = int(joint_freq['n_joint_tours'].get(hh, 0))
        members = grp[cfg.person_id_col].tolist()
        child_ids = grp.loc[is_child.loc[grp.index], cfg.person_id_col].tolist()
        adult_ids = [m for m in members if m not in child_ids]
        for t in range(n):
            rng = _rng(cfg.seed, 'jtp', hh, t)
            # participation: each member joins with a role-dependent probability
            p_join = {m: (0.7 if m in adult_ids else 0.5) for m in members}
            party = [m for m in members if rng.random() < p_join[m]]
            if len(party) < 2:  # ensure a valid joint tour
                party = rng.choice(members, size=min(2, len(members)),
                                   replace=False).tolist()
            n_c = sum(1 for m in party if m in child_ids)
            comp = 'children' if n_c == len(party) else ('adults' if n_c == 0 else 'mixed')
            out.append({cfg.hh_id_col: hh, 'joint_tour_idx': t,
                        'participants': party, 'composition': comp,
                        'n_participants': len(party)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 3b. Joint-tour PURPOSE  (household co-assignment into the purpose dimension)
# ---------------------------------------------------------------------------
def _parse_participants(parts) -> list:
    """Parse a participants value (list, tuple, or CSV/pipe string) to a list."""
    if isinstance(parts, (list, tuple)):
        return list(parts)
    s = str(parts).strip()
    if not s:
        return []
    try:
        lst = ast.literal_eval(s)          # "[123, 456]" / "(123, 456)"
        if isinstance(lst, (list, tuple)):
            return list(lst)
    except (ValueError, SyntaxError):
        pass
    sep = '|' if '|' in s else ','
    return [x.strip().strip("'\" ").strip() for x in s.split(sep) if x.strip()]


def _participant_daps(persons: pd.DataFrame, cfg: 'HouseholdCdapConfig') -> dict:
    """Map person_id → coordinated DAP (M/N/H) for the participants.

    Prefers the 'dap' column written by coordinate_household_cdap, then the
    'CDAP' letter from the individual model, then a ptype-role heuristic.
    """
    if 'dap' in persons.columns:
        return persons.set_index(cfg.person_id_col)['dap'].to_dict()
    if 'CDAP' in persons.columns:
        return persons.set_index(cfg.person_id_col)['CDAP'].to_dict()
    mand = set(('FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
                'UniversityStudent', 'SchoolChild'))
    out = {}
    for pid, pt in zip(persons[cfg.person_id_col], persons[cfg.ptype_col]):
        out[pid] = 'M' if pt in mand else 'N'
    return out


def _draw_joint_purpose(row, dap_map: dict, child_ids: set, cfg: 'HouseholdCdapConfig',
                        rng: np.random.Generator) -> str:
    """Draw one joint tour's purpose via a logit over JOINT_PURPOSES.

    U(p) = joint_purpose_utility[(composition, p)]
         + joint_purpose_m_adjust[p] * n_m_participants
         + joint_purpose_child_adjust[p] * n_child_participants

    The composition pins the baseline, while the participants' coordinated DAPs
    and ages move the draw — this is how the household CDAP enters the activity
    purpose dimension structurally (a joint logit, not an independent draw).
    """
    composition = row['composition']
    participants = _parse_participants(row.get('participants', []))
    n_m = int(sum(1 for p in participants if dap_map.get(p) == 'M'))
    n_child = int(sum(1 for p in participants if p in child_ids))

    util = {p: float(cfg.joint_purpose_utility.get((composition, p), 0.0))
            for p in JOINT_PURPOSES}
    for p in JOINT_PURPOSES:
        util[p] += cfg.joint_purpose_m_adjust.get(p, 0.0) * n_m
        util[p] += cfg.joint_purpose_child_adjust.get(p, 0.0) * n_child

    v = np.array([util[p] for p in JOINT_PURPOSES], dtype=float)
    v = v - v.max()
    e = np.exp(v)
    probs = e / e.sum()
    return JOINT_PURPOSES[int(rng.choice(len(JOINT_PURPOSES), p=probs))]


def joint_tour_purpose(participation: pd.DataFrame,
                       persons: pd.DataFrame,
                       cfg: Optional['HouseholdCdapConfig'] = None) -> pd.DataFrame:
    """
    Assign an activity purpose to every joint tour, co-determined with the
    participants' coordinated DAPs and household composition.

    `participation` is the output of joint_tour_participation() (one row per
    HHID/joint_tour_idx with 'participants' and 'composition').  `persons` must
    carry the coordinated 'dap' column (from coordinate_household_cdap) so the
    draw conditions on each member's pattern.

    Returns a copy of `participation` with a 'purpose' column added.
    """
    cfg = cfg or HouseholdCdapConfig()
    if not len(participation):
        return participation.copy()
    dap_map = _participant_daps(persons, cfg)
    is_child = persons[cfg.ptype_col].isin(['SchoolChild', 'Preschool'])
    child_ids = set(persons.loc[is_child, cfg.person_id_col])

    out = participation.copy()
    purposes = []
    for _, row in out.iterrows():
        idx = int(row.get('joint_tour_idx', 0))
        rng = _rng(cfg.seed, 'jtpurp', row[cfg.hh_id_col], idx)
        purposes.append(_draw_joint_purpose(row, dap_map, child_ids, cfg, rng))
    out['purpose'] = purposes
    return out


# ---------------------------------------------------------------------------
# 4. Household vehicle allocation  (availability-based, uses time windows)
# ---------------------------------------------------------------------------
def allocate_household_vehicles(tours: pd.DataFrame,
                                persons: pd.DataFrame,
                                cfg: Optional[HouseholdCdapConfig] = None,
                                start_col: str = 'STARTIME',
                                end_col: str = 'ENDTIME',
                                auto_flag_col: Optional[str] = 'is_auto') -> pd.DataFrame:
    """
    Assign a household vehicle id to each auto tour, or mark it unallocated when
    no vehicle is free during the tour's [start, end] window.

    Greedy time-window allocator: within each household, process auto tours in
    start-time order; each vehicle carries a list of busy intervals; a tour is
    given the first vehicle free for its whole window. When all vehicles are
    busy the tour is left with vehicle_id = -1 (caller downgrades its mode /
    makes it a passenger). This is the probabilistic-availability behaviour: a
    tour's chance of a car equals the chance a vehicle happens to be free.

    Returns a copy of `tours` with columns:
        vehicle_id      — int, 0..(n_veh-1), or -1 if none was available
        vehicle_ok      — bool
    """
    cfg = cfg or HouseholdCdapConfig()
    df = tours.copy()
    if start_col not in df.columns or end_col not in df.columns:
        raise KeyError(f"allocate_household_vehicles: need '{start_col}'/'{end_col}'")

    veh_by_hh = (persons.groupby(cfg.hh_id_col)[cfg.vehicles_col].first()
                 if cfg.vehicles_col in persons.columns else pd.Series(dtype=float))

    df['vehicle_id'] = -1
    df['vehicle_ok'] = False
    is_auto = (df[auto_flag_col].astype(bool)
               if auto_flag_col and auto_flag_col in df.columns
               else pd.Series(True, index=df.index))

    for hh, grp in df.groupby(cfg.hh_id_col, sort=False):
        n_veh = int(veh_by_hh.get(hh, 1) or 0)
        if n_veh <= 0:
            continue
        busy = [[] for _ in range(n_veh)]      # per-vehicle list of (start,end)
        auto_idx = grp.index[is_auto.loc[grp.index]]
        order = grp.loc[auto_idx].sort_values(start_col).index
        for i in order:
            s = float(df.at[i, start_col]); e = float(df.at[i, end_col])
            for v in range(n_veh):
                if all(e <= bs or s >= be for bs, be in busy[v]):   # free this window
                    busy[v].append((s, e))
                    df.at[i, 'vehicle_id'] = v
                    df.at[i, 'vehicle_ok'] = True
                    break
    return df


# ---------------------------------------------------------------------------
# 4b. Structural bridge into the activity-purpose dimension
# ---------------------------------------------------------------------------
def reconcile_joint_tours(synth_s1: pd.DataFrame,
                          joint_tours: pd.DataFrame,
                          cfg: Optional['HouseholdCdapConfig'] = None,
                          additive: Optional[bool] = None) -> tuple:
    """
    Apply household joint tours (with purposes) to the Stage-1 per-person
    purpose counts — the point where the household CDAP structurally enters the
    activity-purpose dimension.

    `joint_tours` is the joint_tour_participation → joint_tour_purpose output:
    one row per (HHID, joint_tour_idx) with 'participants', 'composition' and
    'purpose'.  Every participant of a joint tour of purpose p gets:

      * n_joint_{p}_out  = 1      — the shared-purpose flag
      * joint_tour_ids / joint_purposes — person-level lists for downstream
      * n_{p}_out raised/inflated per cfg.joint_tour_additive:
          True  → +1 (joint tour is an ADDITIONAL shared activity)
          False → at least 1 (joint tour replaces one independent activity)

    Returns (synth, person_joint_map) where person_joint_map is one row per
    (participant, joint tour) and is the key used by tag_joint_activities().
    """
    cfg = cfg or HouseholdCdapConfig()
    if cfg.person_id_col not in synth_s1.columns:
        raise KeyError(f"reconcile_joint_tours: missing '{cfg.person_id_col}' in synth_s1")
    if joint_tours is None or not len(joint_tours):
        return synth_s1.copy(), pd.DataFrame()

    additive = cfg.joint_tour_additive if additive is None else additive
    df = synth_s1.copy()
    rows = []

    # map person_id → row index once (ids may be int or str after CSV round-trips)
    pid_col = cfg.person_id_col
    pid_to_idx = {str(v): i for i, v in enumerate(df[pid_col])}

    for _, jt in joint_tours.iterrows():
        purpose = str(jt.get('purpose', 'othdiscr')).lower().strip()
        if purpose not in JOINT_PURPOSES:
            purpose = 'othdiscr'
        parts = _parse_participants(jt.get('participants', []))
        hhv = jt.get(cfg.hh_id_col)
        idx_v = int(jt.get('joint_tour_idx', 0) or 0)
        jtid = f"{hhv}_{idx_v}"
        col = f'n_{purpose}_out'
        jcol = f'n_joint_{purpose}_out'
        if col not in df.columns:
            df[col] = 0
        if jcol not in df.columns:
            df[jcol] = 0
        for p in parts:
            pk = str(p)
            if pk not in pid_to_idx:
                continue
            i = pid_to_idx[pk]
            if additive:
                df.iat[i, df.columns.get_loc(col)] = df.iat[i, df.columns.get_loc(col)] + 1
            else:
                df.iat[i, df.columns.get_loc(col)] = max(
                    int(df.iat[i, df.columns.get_loc(col)]), 1)
            df.iat[i, df.columns.get_loc(jcol)] = 1
            rows.append({'person_id': p, 'joint_tour_id': jtid,
                         'purpose': purpose, 'composition': jt.get('composition', ''),
                         'n_participants': len(parts), 'participants': list(parts),
                         cfg.hh_id_col: hhv})

    if rows:
        pmap = pd.DataFrame(rows)
        _jids_raw = pmap.groupby('person_id')['joint_tour_id'].apply(list).to_dict()
        _jpur_raw = pmap.groupby('person_id')['purpose'].apply(list).to_dict()
        # normalise keys to str so mapping works whether person ids are int or str
        jids = {str(k): v for k, v in _jids_raw.items()}
        jpur = {str(k): v for k, v in _jpur_raw.items()}
        df['joint_tour_ids'] = df[pid_col].astype(str).map(jids).fillna('').apply(
            lambda x: ','.join(str(v) for v in x) if isinstance(x, list) else str(x))
        df['joint_purposes'] = df[pid_col].astype(str).map(jpur).fillna('').apply(
            lambda x: ','.join(str(v) for v in x) if isinstance(x, list) else str(x))
        logger_ = logging.getLogger('cdap_households')
        logger_.info(
            f"reconcile_joint_tours: {len(pmap):,} participant-links, "
            f"{df['joint_tour_ids'].astype(str).ne('').sum():,} persons tagged, "
            f"additive={additive}"
        )
    return df, (pd.DataFrame(rows) if rows else pd.DataFrame())


def tag_joint_activities(activities: pd.DataFrame,
                         person_joint_map: pd.DataFrame,
                         cfg: Optional['HouseholdCdapConfig'] = None) -> pd.DataFrame:
    """
    Mark the Stage-3 activity rows that belong to a household joint tour.

    For each (participant, purpose) pair in person_joint_map the FIRST activity
    row of that purpose for the person is tagged with is_joint=1, its
    joint_tour_id and the full participant list.  Only new columns are added —
    no existing column is modified, so this is safe to run over the Stage-3
    activity frame without disturbing downstream duration/scheduling models.

    Returns a copy of `activities` with 'is_joint' / 'joint_tour_id' /
    'joint_participants' columns.
    """
    cfg = cfg or HouseholdCdapConfig()
    if person_joint_map is None or not len(person_joint_map):
        return activities.copy()
    df = activities.copy()
    df['is_joint'] = 0
    df['joint_tour_id'] = ''
    df['joint_participants'] = ''

    seen = set()
    for _, jt in person_joint_map.iterrows():
        pid = jt.get('person_id')
        purpose = str(jt.get('purpose', 'othdiscr')).lower().strip()
        key = (str(pid), purpose)
        if key in seen:
            continue
        seen.add(key)
        m = ((df['person_id'].astype(str) == str(pid)) & (df['purpose'] == purpose)
             & (df['is_joint'] == 0))
        idx = df.index[m]
        if len(idx) == 0:
            continue
        parts = _parse_participants(jt.get('participants', []))
        df.at[idx[0], 'is_joint'] = 1
        df.at[idx[0], 'joint_tour_id'] = str(jt.get('joint_tour_id', ''))
        df.at[idx[0], 'joint_participants'] = ','.join(str(x) for x in parts)
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def apply_household_cdap(persons: pd.DataFrame,
                         tours: Optional[pd.DataFrame] = None,
                         cfg: Optional[HouseholdCdapConfig] = None) -> dict:
    """
    Run the household coordination pipeline end-to-end. Pure — returns a dict of
    new/augmented tables; wires nothing into the live pipeline.

        {'persons': <persons + dap>,
         'joint_freq': <hh → n_joint_tours>,
         'joint_tours': <participant lists + purpose>,
         'tours': <tours + vehicle_id> (only if `tours` given)}
    """
    cfg = cfg or HouseholdCdapConfig()
    persons2 = coordinate_household_cdap(persons, cfg)
    jf = joint_tour_frequency(persons2, cfg)
    jt = joint_tour_participation(persons2, jf, cfg)
    if len(jt):
        jt = joint_tour_purpose(jt, persons2, cfg)
    result = {'persons': persons2, 'joint_freq': jf, 'joint_tours': jt}
    if tours is not None:
        result['tours'] = allocate_household_vehicles(tours, persons2, cfg)
    return result


# ---------------------------------------------------------------------------
# Standalone demo / self-test  (run: python cdap_households.py)
# ---------------------------------------------------------------------------
def _demo() -> None:
    rng = np.random.default_rng(0)
    people, tours = [], []
    pid = 0
    for hh in range(200):
        size = int(rng.integers(1, 5))
        veh = int(rng.integers(0, 3))
        for k in range(size):
            pt = rng.choice(['FullTimeWorker', 'PartTimeWorker', 'SchoolChild',
                             'Preschool', 'Retired', 'NonWorker'])
            people.append({'HHID': hh, 'person_id': pid, 'ptype': pt, 'vehrd': veh})
            # a couple of auto tours per person with random windows
            for _ in range(int(rng.integers(1, 3))):
                s = int(rng.integers(360, 1000)); d = int(rng.integers(30, 180))
                tours.append({'HHID': hh, 'person_id': pid,
                              'STARTIME': s, 'ENDTIME': min(s + d, 1439), 'is_auto': True})
            pid += 1
    persons = pd.DataFrame(people)
    tours = pd.DataFrame(tours)

    res = apply_household_cdap(persons, tours)
    p = res['persons']
    print(f"households={persons['HHID'].nunique()}  persons={len(persons)}  tours={len(tours)}")
    print("DAP distribution:", p['dap'].value_counts().to_dict())
    print("total joint tours:", int(res['joint_freq']['n_joint_tours'].sum()),
          "| composition:", res['joint_tours']['composition'].value_counts().to_dict()
          if len(res['joint_tours']) else {})
    if len(res['joint_tours']):
        print("joint-purpose distribution:",
              res['joint_tours']['purpose'].value_counts().to_dict())
    t = res['tours']
    got = int(t['vehicle_ok'].sum()); tot = int(t['is_auto'].sum())
    print(f"vehicle allocation: {got}/{tot} auto tours got a car "
          f"({100.0 * got / max(tot, 1):.1f}%); {tot - got} unallocated (no vehicle free)")
    # sanity: no vehicle double-booked within a household
    ok = True
    for hh, g in t[t['vehicle_ok']].groupby('HHID'):
        for v, gv in g.groupby('vehicle_id'):
            iv = gv.sort_values('STARTIME')[['STARTIME', 'ENDTIME']].values
            if any(iv[k][0] < iv[k - 1][1] for k in range(1, len(iv))):
                ok = False
    print("no vehicle double-booked:", ok)

    # ── structural bridge check: joint purposes → Stage-1 style counts ─────
    from stage1_activity_frequency import PURPOSES
    synth = pd.DataFrame({
        'person_id': p['person_id'].tolist(),
        'HHID': p['HHID'].tolist(),
    })
    for purp in PURPOSES:
        synth[f'n_{purp}_out'] = rng.integers(0, 3, size=len(synth))
    synth2, pmap = reconcile_joint_tours(synth, res['joint_tours'],
                                         HouseholdCdapConfig(hh_id_col='HHID',
                                                            person_id_col='person_id'))
    print(f"reconciled: {len(pmap)} participant-links; "
          f"persons with joint_tour_ids: "
          f"{int(synth2['joint_tour_ids'].astype(str).ne('').sum())}")
    acts = pd.DataFrame({'person_id': synth2['person_id'].tolist(), 'purpose': 'shopping'})
    acts = tag_joint_activities(acts, pmap,
                                HouseholdCdapConfig(hh_id_col='HHID',
                                                    person_id_col='person_id'))
    print("joint-tagged activity rows:",
          int(acts['is_joint'].sum()) if 'is_joint' in acts else 0)


if __name__ == '__main__':
    _demo()
