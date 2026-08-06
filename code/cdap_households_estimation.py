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
4. Joint-tour purpose   : joint tour purpose ~ composition (adults/children/mixed)
                          → categorical (→ composition-conditional purpose utilities
                          folded into HouseholdCdapConfig.joint_purpose_utility).
                          This is the structural link between the household CDAP
                          and the Stage-1 activity-purpose dimension.

All logit choices (1 and 3) are fitted with **SearchLibrium** (github.com/zahern/
HypothesisX) discrete-choice MLE (MultinomialLogit; JAX/scipy) — ptype and
household-composition utilities are estimated as a proper alternative-specific
choice model rather than a generic ML classifier. If the package is unavailable
the estimators fall back to a plain sklearn multinomial/binary logit with a
warning.

Run (on real HHTS):  python cdap_households_estimation.py
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from cdap_households import HouseholdCdapConfig


_SEARCHLIBRIUM_AVAILABLE = None   # lazily resolved; True/False


def _searchlibrium_mnl():
    """
    Lazily import SearchLibrium's MultinomialLogit.

    SearchLibrium (github.com/zahern/HypothesisX) is the discrete-choice
    estimation framework used for all logit models in this file.  Returns the
    class, or None if the package is not installed (caller then falls back to
    a plain sklearn MNL so the pipeline still runs, but logs the shortfall).
    """
    global _SEARCHLIBRIUM_AVAILABLE
    if _SEARCHLIBRIUM_AVAILABLE is not None:
        return _SEARCHLIBRIUM_AVAILABLE or None
    try:
        from SearchLibrium.multinomial_logit import MultinomialLogit
        _SEARCHLIBRIUM_AVAILABLE = MultinomialLogit
    except Exception as exc:                       # noqa: BLE001
        _SEARCHLIBRIUM_AVAILABLE = False
        print(f"[cdap-hh estimate] SearchLibrium unavailable ({exc}); "
              f"falling back to sklearn MNL")
    return _SEARCHLIBRIUM_AVAILABLE or None


def _searchlibrium_logistic():
    """
    Lazily import SearchLibrium's JAX-based LogisticRegression class.

    Returns the class, or None if the package is not installed (caller then
    falls back to a plain sklearn binary logit so the pipeline still runs).
    """
    global _SEARCHLIBRIUM_AVAILABLE
    if _SEARCHLIBRIUM_AVAILABLE is not None:
        cls = _SEARCHLIBRIUM_AVAILABLE
    else:
        try:
            from SearchLibrium.multinomial_logit import MultinomialLogit
            _SEARCHLIBRIUM_AVAILABLE = MultinomialLogit
            cls = MultinomialLogit
        except Exception as exc:                   # noqa: BLE001
            _SEARCHLIBRIUM_AVAILABLE = False
            print(f"[cdap-hh estimate] SearchLibrium unavailable ({exc}); "
                  f"falling back to sklearn MNL")
            return None
    try:
        from SearchLibrium.logistic_regression import LogisticRegression
        return LogisticRegression
    except Exception as exc:                       # noqa: BLE001
        print(f"[cdap-hh estimate] SearchLibrium LogisticRegression "
              f"unavailable ({exc}); using MultinomialLogit binary path")
        return cls


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


def _tour_purpose(sub: pd.DataFrame) -> str:
    """Modal activity purpose for a group of trips/tours ('' if unknown)."""
    purp = next((c for c in ('main_purpose', 'primary_purpose', 'purpose', 'PURPOSE',
                             'DESTPURP')
                 if c in sub.columns), None)
    if purp is None:
        return ''
    s = sub[purp].dropna()
    if not len(s):
        return ''
    return str(s.value_counts().index[0])


def derive_observed_joint_tours(tours: pd.DataFrame,
                                cfg: Optional[HouseholdCdapConfig] = None) -> pd.DataFrame:
    """
    Extract observed JOINT tours from HHTS. A joint tour is one flagged as
    shared/joint travel (TODO(hhts): confirm the flag column — commonly
    'joint_travel' / 'JOINT' / a shared tour_id across members).

    Returns one row per (HHID, joint tour) with the participant person_ids
    and the tour's modal 'purpose' ('' when no purpose column is present).
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
                         'participants': sub[cfg.person_id_col].unique().tolist(),
                         'purpose': _tour_purpose(sub)})
        return pd.DataFrame(rows)
    jt = tours[tours[flag].astype(bool)]
    out = (jt.groupby([cfg.hh_id_col, 'tour_id'])[cfg.person_id_col]
              .apply(list).reset_index(name='participants')
            if 'tour_id' in jt.columns else
            jt.groupby(cfg.hh_id_col)[cfg.person_id_col].apply(list)
              .reset_index(name='participants'))
    # modal purpose per joint tour group
    purp = next((c for c in ('main_purpose', 'primary_purpose', 'purpose', 'PURPOSE',
                             'DESTPURP')
                 if c in jt.columns), None)
    if purp is not None:
        gkeys = ['tour_id'] if 'tour_id' in jt.columns else [cfg.hh_id_col]
        modals = jt.groupby(gkeys)[purp].apply(
            lambda s: str(s.dropna().value_counts().index[0])
            if len(s.dropna()) else '')
        out['purpose'] = out[gkeys[0]].map(modals)
    else:
        out['purpose'] = ''
    return out


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


def _cdap_model_matrix(persons: pd.DataFrame, cfg: HouseholdCdapConfig):
    """Build the wide CDAP choice matrix: ptype dummies + hh-composition features."""
    comp = _hh_composition_features(persons, cfg)
    pt_dummies = pd.get_dummies(persons[cfg.ptype_col], prefix='pt')
    X_wide = pd.concat([pt_dummies, comp[['hh_n_workers', 'child_x_home_adult']]],
                       axis=1).fillna(0.0)
    return X_wide, list(X_wide.columns)


def estimate_cdap_coefficients(persons: pd.DataFrame, tours: pd.DataFrame,
                               cfg: Optional[HouseholdCdapConfig] = None) -> dict:
    """
    Fit person DAP (M/N/H) as a *multinomial logit choice model* using
    SearchLibrium: each person chooses one of {H, M, N}; the choice-set covers
    a home-only pattern (H, the reference alternative) plus two non-mandatory
    variants, committee style.  Alternative-specific ptype and household-
    composition utilities are estimated by maximum likelihood (JAX/scipy).

    Returns {'dap_utility': {...}, 'interaction': {...}} shaped for
    HouseholdCdapConfig (H is the reference class ⇒ utility 0).
    """
    cfg = cfg or HouseholdCdapConfig()
    y = derive_observed_dap(persons, tours, cfg)          # per-person 'H'/'M'/'N'
    alt_of = {'H': 0, 'M': 1, 'N': 2}                     # H is the base alternative
    y_chosen = y.map(alt_of).fillna(0).astype(int).to_numpy()
    X_wide, varnames = _cdap_model_matrix(persons, cfg)

    mnl_cls = _searchlibrium_mnl()
    if mnl_cls is not None:
        mnl = mnl_cls(_jax=True)
        mnl.wide_setup(X_wide=X_wide.to_numpy(), y_chosen=y_chosen,
                       varnames=varnames, n_alts=3, ids=None,
                       empirical_init=True, base_alt=0,
                       maxiter=2000, method="slsqp")
        mnl.fit()
        coeff_names = list(getattr(mnl, 'coeff_names', []))
        coeff_est = np.asarray(getattr(mnl, 'coeff_est', []), dtype=float)
        fitted_loglik = float(getattr(mnl, 'loglik', float('nan')))
        conjugated = bool(getattr(mnl, 'converged', False))
        beta = dict(zip(coeff_names, coeff_est)) if len(coeff_names) else {}
        # 'pt.{alt}' and '<feature>.{alt}' per non-base alternative 1=M, 2=N
        def _b(name, alt):
            return float(beta.get(f"{name}.{alt}", 0.0))
        dap_utility = {
            pt: {'M': _b(f"pt_{pt}", 1), 'N': _b(f"pt_{pt}", 2), 'H': 0.0}
            for pt in pt_from_columns(varnames)
        }
        interaction = {
            'more_workers_boost_M': _b('hh_n_workers', 1),
            'adult_home_boosts_child_N': _b('child_x_home_adult', 2),
        }
        accuracy = _mnl_accuracy(X_wide, varnames, dap_utility, y_chosen)
        print(f"[cdap-hh estimate] SearchLibrium MNL: loglik={fitted_loglik:.2f} "
              f"converged={conjugated} acc={accuracy:.3f} "
              f"({len(coeff_names)} params)")
        return {'dap_utility': dap_utility, 'interaction': interaction,
                'accuracy': accuracy, 'framework': 'SearchLibrium',
                'loglik': fitted_loglik, 'coeff_names': coeff_names}
    # ── Fallback: sklearn multinomial MNL (SearchLibrium not installed) ─────
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, C=5.0)   # lbfgs ⇒ multinomial by default
    clf.fit(X_wide.to_numpy(), y_chosen)
    classes = list(map(int, clf.classes_))
    # sklearn gives per-class coefficients vs. its OWN reference class (the last
    # class, not necessarily H).  MNL utilities are only identified up to an
    # additive constant per alternative, so recentre on H:  U_X = raw_X - raw_H.
    coef = {c: dict(zip(varnames, row)) for c, row in zip(classes, clf.coef_)}
    dap_utility = {}
    for pt in pt_from_columns(varnames):
        cm = coef.get(1, {}).get(f"pt_{pt}", 0.0)   # raw M coefficient
        cn = coef.get(2, {}).get(f"pt_{pt}", 0.0)   # raw N coefficient
        ch = coef.get(0, {}).get(f"pt_{pt}", 0.0)   # raw H coefficient
        dap_utility[pt] = {'M': float(cm - ch), 'N': float(cn - ch), 'H': 0.0}
    interaction = {
        'more_workers_boost_M': float(coef.get(1, {}).get('hh_n_workers', 0.0)
                                      - coef.get(0, {}).get('hh_n_workers', 0.0)),
        'adult_home_boosts_child_N': float(coef.get(2, {}).get('child_x_home_adult', 0.0)
                                           - coef.get(0, {}).get('child_x_home_adult', 0.0)),
    }
    accuracy = _mnl_accuracy(X_wide, varnames, dap_utility, y_chosen)
    return {'dap_utility': dap_utility, 'interaction': interaction,
            'accuracy': accuracy, 'framework': 'sklearn'}


def pt_from_columns(varnames):
    """Recover ptype labels from the 'pt_<ptype>' design columns."""
    return [c[3:] for c in varnames if c.startswith('pt_')]


def _mnl_accuracy(X_wide, varnames, dap_utility, y_chosen):
    """Argmax class from the fitted utilities vs. the observed choices."""
    pt_cols = {i: v[3:] for i, v in enumerate(varnames) if v.startswith('pt_')}
    B = np.zeros((len(varnames), 2))                    # columns: M, N
    for i, v in enumerate(varnames):
        key = pt_cols.get(i, v)
        B[i, 0] = dap_utility.get(key, {}).get('M', 0.0)
        B[i, 1] = dap_utility.get(key, {}).get('N', 0.0)
    U = X_wide.to_numpy() @ B                            # (N, 2): [U_M, U_N]
    UH = np.zeros((U.shape[0], 1))                       # H reference ⇒ 0
    pred = np.argmax(np.concatenate([UH, U], axis=1), axis=1)
    return float((pred == y_chosen).mean())


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
    participated (1/0) as a binary logit choice (SearchLibrium MNL, J=2, base
    alternative = not joining).  Features: const (ASC) + is_child.
    Returns {'p_join_adult','p_join_child'} used by joint_tour_participation.
    (Extend X with tour purpose/age when available.)
    """
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
    y_chosen = cand['joined'].to_numpy()
    X_wide = np.column_stack([np.ones(len(cand)), cand['is_child'].to_numpy()])
    varnames = ['const', 'is_child']

    lg_cls = _searchlibrium_logistic()
    if lg_cls is not None:
        lg = lg_cls(_jax=True)
        lg.fit(X_wide, y_chosen, varnames=varnames,
               empirical_init=False, maxiter=1000, method="slsqp")
        coef = np.asarray(getattr(lg, 'coef_', []) or []).ravel()
        if len(coef) >= 2:
            b_const, b_child = float(coef[0]), float(coef[1])
        else:
            b_const = getattr(lg, 'intercept_', 0.0) or 0.0
            b_child = float(coef[-1]) if len(coef) else 0.0
        p_adult = 1.0 / (1.0 + np.exp(-b_const))
        p_child = 1.0 / (1.0 + np.exp(-(b_const + b_child)))
        print(f"[cdap-hh estimate] participation logit: b_const={b_const:.3f} "
              f"b_is_child={b_child:.3f}")
        return {'p_join_adult': float(p_adult), 'p_join_child': float(p_child),
                'framework': 'SearchLibrium'}
    # ── Fallback: sklearn binary logistic regression ────────────────────────
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_wide[:, 1:], y_chosen)                 # drop const; sklearn adds its own
    p_adult = float(clf.predict_proba([[0]])[0, 1])
    p_child = float(clf.predict_proba([[1]])[0, 1])
    return {'p_join_adult': p_adult, 'p_join_child': p_child, 'framework': 'sklearn'}


# ---------------------------------------------------------------------------
# 3b. Joint-tour PURPOSE  (composition → purpose choice, structural)
# ---------------------------------------------------------------------------
def _composition_label(participants, is_child_map: dict) -> str:
    """Classify a participant list as adults / children / mixed."""
    n_child = sum(1 for p in participants if is_child_map.get(p))
    n_tot = max(len(participants), 1)
    if n_child == n_tot:
        return 'children'
    if n_child == 0:
        return 'adults'
    return 'mixed'


def estimate_joint_purpose(persons: pd.DataFrame, tours: pd.DataFrame,
                           cfg: Optional[HouseholdCdapConfig] = None) -> dict:
    """
    Estimate the household joint-tour purpose choice from HHTS.

    Observed joint tours (derive_observed_joint_tours) are classified into
    adults / children / mixed composition and the purpose distribution per
    composition is used to populate HouseholdCdapConfig.joint_purpose_utility on
    the logit scale (U = ln share).

    Returns
    -------
    {'shares': {composition: {purpose: share}},
     'utility': {(composition, purpose): utility},
     'n_obs': int}
    """
    from cdap_households import JOINT_PURPOSES
    cfg = cfg or HouseholdCdapConfig()
    jt = derive_observed_joint_tours(tours, cfg)
    if not len(jt):
        return {'shares': {}, 'utility': dict(cfg.joint_purpose_utility), 'n_obs': 0}

    is_child = persons[cfg.ptype_col].isin(['SchoolChild', 'Preschool']).astype(bool)
    is_child.index = persons[cfg.person_id_col]
    is_child_map = is_child.to_dict()
    comps = []
    purps = []
    for _, row in jt.iterrows():
        comps.append(_composition_label(row['participants'], is_child_map))
        purps.append(str(row.get('purpose', '')).lower().strip())
    jt2 = jt.assign(_comp=comps, _purpose=purps)

    shares = {}
    for comp, grp in jt2.groupby('_comp'):
        vc = grp[grp['_purpose'].ne('')]['_purpose'].value_counts(normalize=True)
        shares[comp] = {p: float(v) for p, v in vc.items()}

    utility = {}
    for comp, probs in shares.items():
        for p in JOINT_PURPOSES:
            utility[(comp, p)] = float(np.log(max(probs.get(p, 1e-4), 1e-4)))

    print(f"[cdap-hh estimate] joint-purpose: n_joint={len(jt2)}, "
          f"compositions={ {c: int(n) for c, n in jt2['_comp'].value_counts().items()} }")
    for comp, probs in shares.items():
        print(f"    {comp:9s}: "
              + ", ".join(f"{p}={v:.3f}" for p, v in sorted(probs.items())))
    return {'shares': shares, 'utility': utility, 'n_obs': len(jt2)}


# ---------------------------------------------------------------------------
# Orchestrator: HHTS → estimated HouseholdCdapConfig
# ---------------------------------------------------------------------------
def estimate_household_cdap(persons: pd.DataFrame, tours: pd.DataFrame,
                            base: Optional[HouseholdCdapConfig] = None) -> HouseholdCdapConfig:
    """Fit all four models and fold the coefficients into a HouseholdCdapConfig."""
    cfg = base or HouseholdCdapConfig()
    dap = estimate_cdap_coefficients(persons, tours, cfg)
    freq = estimate_joint_frequency(persons, tours, cfg)
    part = estimate_joint_participation(persons, tours, cfg)
    purp = estimate_joint_purpose(persons, tours, cfg)

    cfg.dap_utility.update(dap['dap_utility'])
    cfg.more_workers_boost_M = dap['interaction']['more_workers_boost_M']
    cfg.adult_home_boosts_child_N = dap['interaction']['adult_home_boosts_child_N']
    cfg.joint_base_rate = freq['joint_base_rate']
    cfg.joint_rate_per_child = freq['joint_rate_per_child']
    cfg.joint_rate_per_adult = freq['joint_rate_per_adult']
    cfg.joint_rate_if_multi_vehicle = freq['joint_rate_if_multi_vehicle']
    # participation probs live in joint_tour_participation; stash for reference
    cfg._participation = part   # type: ignore[attr-defined]
    if purp['utility']:
        cfg.joint_purpose_utility.update(purp['utility'])
    cfg._joint_purpose = purp  # type: ignore[attr-defined]
    print(f"[cdap-hh estimate] DAP acc={dap['accuracy']:.3f}  "
          f"joint mean obs={freq['mean_observed']:.3f}  "
          f"p_join adult/child={part.get('p_join_adult'):.2f}/{part.get('p_join_child'):.2f}  "
          f"joint-purpose n={purp['n_obs']}")
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
