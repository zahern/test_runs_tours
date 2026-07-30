"""
cdap_households.py  —  ActivitySim-style household coordination
================================================================
FUTURE WORK — a self-contained module that is **not wired into any pipeline job
yet**. Build and test it standalone (``python cdap_households.py`` runs a demo on
synthetic households); when it is ready, plug it into the jobs at the points
marked "PLUG-IN POINT" below.

What it models (mirrors ActivitySim's household sub-models):
  1. coordinate_household_cdap  — a coordinated Daily Activity Pattern (Mandatory
     / NonMandatory / Home) per person, with intra-household interactions so a
     household's members' patterns are consistent (e.g. a non-working adult at
     home raises the chance a child is escorted rather than travelling alone).
  2. joint_tour_frequency       — how many JOINT (whole-/part-household) tours a
     household makes.
  3. joint_tour_participation    — which members travel together on each joint
     tour (adults-only / with-children / mixed composition).
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

PLUG-IN POINTS (when complete):
  A. After person profiles + CDAP (cdap.predict_cdap) and BEFORE tour frequency
     (stage1_activity_frequency): call coordinate_household_cdap to overwrite the
     per-person DAP with the coordinated one; then joint_tour_frequency /
     joint_tour_participation add joint tours to the tour list.
  B. After stage-4 scheduling (STARTIME/ENDTIME assigned) and the Tier-1
     conflict resolver: call allocate_household_vehicles to set each auto tour's
     vehicle id / downgrade mode when no vehicle is free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DAP_CLASSES = ('M', 'N', 'H')   # Mandatory, NonMandatory, Home


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
         'joint_tours': <participant lists>,
         'tours': <tours + vehicle_id> (only if `tours` given)}
    """
    cfg = cfg or HouseholdCdapConfig()
    persons2 = coordinate_household_cdap(persons, cfg)
    jf = joint_tour_frequency(persons2, cfg)
    jt = joint_tour_participation(persons2, jf, cfg)
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


if __name__ == '__main__':
    _demo()
