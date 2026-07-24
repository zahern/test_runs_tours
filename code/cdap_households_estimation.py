"""
cdap_households_estimation.py  —  estimation layer for cdap_households (SKETCH)
==============================================================================
Fits the household-coordination models from HHTS and returns a populated
``HouseholdCdapConfig`` whose coefficient tables replace the heuristic priors —
so `cdap_households.apply_household_cdap` runs on estimated behaviour without any
call-site change.

This is a SKETCH: the model forms, the observed-outcome derivation, and the
fitting calls are concrete; the two places that need your HHTS specifics are
marked `TODO(hhts)` (the joint-travel flag column, and the tour→person link).

Three models
------------
1. Coordinated CDAP     : person DAP (M/N/H) ~ ptype + household-composition
                          features → multinomial logit. The composition
                          coefficients ARE the coordination effect.
2. Joint-tour frequency : n_joint_tours per household ~ hh features → Poisson.
3. Joint participation  : member joins a joint tour (0/1) ~ role + tour features
                          → binary logit (→ per-role join probabilities).

Run (on real HHTS):  python cdap_households_estimation.py
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from cdap_households import HouseholdCdapConfig, DAP_CLASSES


# ---------------------------------------------------------------------------
# 0. Observed-outcome derivation from HHTS
# ---------------------------------------------------------------------------
def derive_observed_dap(persons: pd.DataFrame, tours: pd.DataFrame,
                        cfg: Optional[HouseholdCdapConfig] = None) -> pd.Series:
    """
    Classify each person's OBSERVED daily activity pattern from their tours:
        M (Mandatory)     — has ≥1 work/education tour
        N (NonMandatory)  — has tours, none mandatory
        H (Home)          — made no tours
    Returns a Series 'dap' aligned to `persons[person_id]`.
    """
    cfg = cfg or HouseholdCdapConfig()
    pid = cfg.person_id_col
    purp = next((c for c in ('main_purpose', 'primary_purpose', 'purpose', 'PURPOSE')
                 if c in tours.columns), None)
    mand = {'work', 'edu', 'education', 'university', 'school',
            'Direct Work Commute', 'Education'}
    per_person = {}
    if purp is not None:
        g = tours.groupby(pid)[purp]
        for p, s in g:
            vals = set(str(x) for x in s.dropna().unique())
            per_person[p] = 'M' if (vals & mand) else 'N'
    dap = persons[pid].map(per_person).fillna('H')   # no tours → Home
    dap.name = 'dap'
    return dap


def derive_observed_joint_tours(tours: pd.DataFrame,
                                cfg: Optional[HouseholdCdapConfig] = None) -> pd.DataFrame:
    """
    Extract observed JOINT tours from HHTS. A joint tour is one flagged as
    shared/joint travel (TODO(hhts): confirm the flag column — commonly
    'joint_travel' / 'JOINT' / a shared tour_id across members).

    Returns one row per (HHID, joint tour) with the participant person_ids.
    """
    cfg = cfg or HouseholdCdapConfig()
    flag = next((c for c in ('joint_travel', 'JOINT', 'is_joint', 'joint')
                 if c in tours.columns), None)
    if flag is None:
        # Fallback heuristic: members of the same household sharing an
        # (OTAZ,DTAZ,departure) triple are travelling together.
        keys = [k for k in ('HHID', 'OTAZ', 'DTAZ', 'STARTIME', 'depart_hr')
                if k in tours.columns]
        if 'HHID' not in keys:
            return pd.DataFrame(columns=[cfg.hh_id_col, 'participants'])
        grp = tours.groupby(keys)[cfg.person_id_col].nunique()
        shared = grp[grp >= 2].index
        rows = []
        for key in shared:
            sub = tours
            for k, val in zip(keys, key if isinstance(key, tuple) else (key,)):
                sub = sub[sub[k] == val]
            rows.append({cfg.hh_id_col: sub[cfg.hh_id_col].iloc[0],
                         'participants': sub[cfg.person_id_col].unique().tolist()})
        return pd.DataFrame(rows)
    jt = tours[tours[flag].astype(bool)]
    return (jt.groupby([cfg.hh_id_col, 'tour_id'])[cfg.person_id_col]
              .apply(list).reset_index(name='participants')
            if 'tour_id' in jt.columns else
            jt.groupby(cfg.hh_id_col)[cfg.person_id_col].apply(list)
              .reset_index(name='participants'))


# ---------------------------------------------------------------------------
# 1. Coordinated CDAP  (multinomial logit)
# ---------------------------------------------------------------------------
def _hh_composition_features(persons: pd.DataFrame, cfg: HouseholdCdapConfig) -> pd.DataFrame:
    """Per-person household-composition covariates (the coordination signal)."""
    is_child = persons[cfg.ptype_col].isin(['SchoolChild', 'Preschool'])
    is_worker = persons[cfg.ptype_col].isin(['FullTimeWorker', 'PartTimeWorker', 'WorkingStudent'])
    is_home_adult = persons[cfg.ptype_col].isin(['Retired', 'NonWorker'])
    g = persons.assign(_child=is_child, _wk=is_worker, _ha=is_home_adult).groupby(cfg.hh_id_col)
    hh = pd.DataFrame({
        'hh_n_workers': g['_wk'].transform('sum'),
        'hh_n_children': g['_child'].transform('sum'),
        'hh_has_home_adult': g['_ha'].transform('max'),
    })
    hh['is_child'] = is_child.astype(int).values
    hh['child_x_home_adult'] = hh['is_child'] * hh['hh_has_home_adult']
    return hh


def estimate_cdap_coefficients(persons: pd.DataFrame, tours: pd.DataFrame,
                               cfg: Optional[HouseholdCdapConfig] = None) -> dict:
    """
    Fit person DAP (M/N/H) ~ ptype dummies + household-composition features via a
    multinomial logit. Returns {'dap_utility': {...}, 'interaction': {...}}
    reshaped into the HouseholdCdapConfig layout (H is the reference class).
    """
    from sklearn.linear_model import LogisticRegression
    cfg = cfg or HouseholdCdapConfig()
    y = derive_observed_dap(persons, tours, cfg)
    comp = _hh_composition_features(persons, cfg)
    pt_dummies = pd.get_dummies(persons[cfg.ptype_col], prefix='pt')
    X = pd.concat([pt_dummies, comp[['hh_n_workers', 'child_x_home_adult']]], axis=1).fillna(0.0)

    clf = LogisticRegression(multi_class='multinomial', max_iter=500, C=5.0)
    clf.fit(X.values, y.values)
    classes = list(clf.classes_)                     # e.g. ['H','M','N']
    coef = {c: dict(zip(X.columns, row)) for c, row in zip(classes, clf.coef_)}

    # reshape to per-ptype DAP utility (M/N/H), H as reference (utility 0)
    dap_utility = {}
    for pt_col in pt_dummies.columns:
        pt = pt_col[3:]
        dap_utility[pt] = {k: float(coef.get(k, {}).get(pt_col, 0.0)) for k in DAP_CLASSES}
    interaction = {
        'more_workers_boost_M': float(coef.get('M', {}).get('hh_n_workers', 0.0)),
        'adult_home_boosts_child_N': float(coef.get('N', {}).get('child_x_home_adult', 0.0)),
    }
    return {'dap_utility': dap_utility, 'interaction': interaction,
            'accuracy': float((clf.predict(X.values) == y.values).mean())}


# ---------------------------------------------------------------------------
# 2. Joint-tour frequency  (Poisson)
# ---------------------------------------------------------------------------
def estimate_joint_frequency(persons: pd.DataFrame, tours: pd.DataFrame,
                             cfg: Optional[HouseholdCdapConfig] = None) -> dict:
    """
    Fit n_joint_tours per household ~ n_children + n_adults + multi_vehicle via
    Poisson regression. Returns rate-coefficient fields for HouseholdCdapConfig.
    """
    from sklearn.linear_model import PoissonRegressor
    cfg = cfg or HouseholdCdapConfig()
    jt = derive_observed_joint_tours(tours, cfg)
    counts = (jt.groupby(cfg.hh_id_col).size() if len(jt) else pd.Series(dtype=int))

    is_child = persons[cfg.ptype_col].isin(['SchoolChild', 'Preschool'])
    hh = persons.assign(_child=is_child).groupby(cfg.hh_id_col).agg(
        n_children=('_child', 'sum'),
        n_persons=(cfg.person_id_col, 'size'),
        veh=(cfg.vehicles_col, 'first'))
    hh['n_adults'] = hh['n_persons'] - hh['n_children']
    hh['multi_veh'] = (hh['veh'] >= 2).astype(int)
    hh['y'] = counts.reindex(hh.index).fillna(0).astype(int)

    Xcols = ['n_children', 'n_adults', 'multi_veh']
    reg = PoissonRegressor(alpha=1e-3, max_iter=500)
    reg.fit(hh[Xcols].values, hh['y'].values)
    b = dict(zip(Xcols, reg.coef_))
    # Poisson is multiplicative (log link); expose as additive rate increments
    # around the fitted intercept for the config's simple linear-rate form.
    base = float(np.exp(reg.intercept_))
    return {'joint_base_rate': base,
            'joint_rate_per_child': base * float(b['n_children']),
            'joint_rate_per_adult': base * float(b['n_adults']),
            'joint_rate_if_multi_vehicle': base * float(b['multi_veh']),
            'mean_observed': float(hh['y'].mean())}


# ---------------------------------------------------------------------------
# 3. Joint participation  (binary logit → per-role join probability)
# ---------------------------------------------------------------------------
def estimate_joint_participation(persons: pd.DataFrame, tours: pd.DataFrame,
                                 cfg: Optional[HouseholdCdapConfig] = None) -> dict:
    """
    For households with an observed joint tour, model whether each member
    participated (1/0) ~ is_child. Returns {'p_join_adult','p_join_child'} used
    by joint_tour_participation. (Extend X with tour purpose/age when available.)
    """
    from sklearn.linear_model import LogisticRegression
    cfg = cfg or HouseholdCdapConfig()
    jt = derive_observed_joint_tours(tours, cfg)
    if not len(jt):
        return {'p_join_adult': 0.7, 'p_join_child': 0.5, 'note': 'no observed joint tours'}
    part_ids = set()
    for lst in jt['participants']:
        part_ids.update(lst)
    # candidate set = members of households that made a joint tour
    hh_with_jt = set(jt[cfg.hh_id_col])
    cand = persons[persons[cfg.hh_id_col].isin(hh_with_jt)].copy()
    cand['joined'] = cand[cfg.person_id_col].isin(part_ids).astype(int)
    cand['is_child'] = cand[cfg.ptype_col].isin(['SchoolChild', 'Preschool']).astype(int)
    clf = LogisticRegression(max_iter=300)
    clf.fit(cand[['is_child']].values, cand['joined'].values)
    p_adult = float(clf.predict_proba([[0]])[0, 1])
    p_child = float(clf.predict_proba([[1]])[0, 1])
    return {'p_join_adult': p_adult, 'p_join_child': p_child}


# ---------------------------------------------------------------------------
# Orchestrator: HHTS → estimated HouseholdCdapConfig
# ---------------------------------------------------------------------------
def estimate_household_cdap(persons: pd.DataFrame, tours: pd.DataFrame,
                            base: Optional[HouseholdCdapConfig] = None) -> HouseholdCdapConfig:
    """Fit all three models and fold the coefficients into a HouseholdCdapConfig."""
    cfg = base or HouseholdCdapConfig()
    dap = estimate_cdap_coefficients(persons, tours, cfg)
    freq = estimate_joint_frequency(persons, tours, cfg)
    part = estimate_joint_participation(persons, tours, cfg)

    cfg.dap_utility.update(dap['dap_utility'])
    cfg.more_workers_boost_M = dap['interaction']['more_workers_boost_M']
    cfg.adult_home_boosts_child_N = dap['interaction']['adult_home_boosts_child_N']
    cfg.joint_base_rate = freq['joint_base_rate']
    cfg.joint_rate_per_child = freq['joint_rate_per_child']
    cfg.joint_rate_per_adult = freq['joint_rate_per_adult']
    cfg.joint_rate_if_multi_vehicle = freq['joint_rate_if_multi_vehicle']
    # participation probs live in joint_tour_participation; stash for reference
    cfg._participation = part   # type: ignore[attr-defined]
    print(f"[cdap-hh estimate] DAP acc={dap['accuracy']:.3f}  "
          f"joint mean obs={freq['mean_observed']:.3f}  "
          f"p_join adult/child={part.get('p_join_adult'):.2f}/{part.get('p_join_child'):.2f}")
    return cfg


def _estimate_on_hhts() -> None:
    """Run the estimation on real HHTS (guarded; needs the pipeline data)."""
    from data_connect import Synth_Connect
    from cdap import build_person_profiles
    sc = Synth_Connect()
    out = sc.process_survey(save=False)
    persons = build_person_profiles(out['persons'])
    if 'person_id' not in persons.columns and 'PERSID' in persons.columns:
        persons = persons.rename(columns={'PERSID': 'person_id'})
    tours = out.get('tours', out.get('trips'))
    cfg = estimate_household_cdap(persons, tours)
    print("estimated dap_utility (sample):",
          {k: {kk: round(vv, 2) for kk, vv in v.items()}
           for k, v in list(cfg.dap_utility.items())[:3]})


if __name__ == '__main__':
    _estimate_on_hhts()
