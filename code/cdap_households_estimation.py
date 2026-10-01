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

import os
import pickle
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
_MANDATORY_PURPOSE_TOKENS = ('work', 'school', 'educat', 'univers',
                             'tafe', 'kinder', 'childcare')


def _purpose_is_mandatory(p) -> bool:
    """
    True when a tour purpose denotes a MANDATORY (work/education) activity.

    Matches BOTH coded purposes ('work', 'edu') AND raw QTS/HHTS strings
    ('Full-time Work', 'Direct Work Commute', 'Primary School',
     'Full-time TAFE/Uni', 'Childcare/Kindergarten', ...), so the joint
    household CDAP uses the same mandatory definition as the individual
    model's MANDATORY_ACTS / prepare_cdap_data() — previously the narrow
    coded-only set misclassified every raw-string worker/student as 'N'.
    """
    s = str(p).strip().lower()
    if s in ('', 'nan', 'none'):
        return False
    return any(tok in s for tok in _MANDATORY_PURPOSE_TOKENS)


def derive_observed_dap(persons: pd.DataFrame, tours: pd.DataFrame,
                        cfg: Optional[HouseholdCdapConfig] = None) -> pd.Series:
    """
    Classify each person's OBSERVED daily activity pattern from their tours:
        M (Mandatory)     — has ≥1 work/education tour
        N (NonMandatory)  — has tours, none mandatory
        H (Home)          — made no tours
    Returns a Series 'dap' aligned to `persons[person_id]`. Diagnostics are
    attached via .attrs (purpose column used, counts of tour-less persons).
    """
    cfg = cfg or HouseholdCdapConfig()
    pid = cfg.person_id_col
    t = tours
    if pid not in t.columns and 'PERSID' in t.columns:
        t = t.rename(columns={'PERSID': pid})
    purp = next((c for c in ('main_purpose', 'primary_purpose', 'purpose',
                             'PURPOSE', 'DESTPURP')
                 if c in t.columns), None)
    per_person = {}
    if purp is not None:
        for p, s in t.groupby(pid)[purp]:
            vals = [v for v in s.dropna().unique()
                    if str(v).strip() not in ('', 'nan', 'None')]
            if not vals:
                continue
            per_person[p] = 'M' if any(_purpose_is_mandatory(v)
                                       for v in vals) else 'N'
    dap = persons[pid].map(per_person).fillna('H')   # no tours → Home
    dap.name = 'dap'
    dap.attrs['purpose_col'] = purp
    dap.attrs['n_persons_with_tours'] = len(per_person)
    return dap


def attach_cdap_to_hhts(persons: pd.DataFrame, tours: pd.DataFrame,
                        cfg: Optional[HouseholdCdapConfig] = None) -> pd.DataFrame:
    """Attach OBSERVED CDAP attributes to HHTS travellers for joint estimation.

    Estimation-side counterpart of the prediction path (`coordinate_household_
    cdap` + `joint_tour_frequency/participation` draw sampled values for the
    synthetic population). Here every attribute is a deterministic function of
    the respondent's OWN observed tours — the ActivitySim observed-vs-simulated
    pattern — so downstream estimators (stage 1/4/5, ...) can condition on
    household coordination with no leakage and no randomness:

      cdap_M / cdap_N / cdap_H  observed DAP dummies (from derive_observed_dap)
      hh_n_workers / hh_n_children / hh_has_home_adult / child_x_home_adult
                                household-composition coordination signal
      n_joint_tours_hh          observed joint-tour count of the household
      is_joint_traveller        in ≥1 observed joint tour
      joint_party_n             mean observed party size (0 when solo-only)

    Idempotent (skips when 'cdap_M' present) and never raises — returns
    `persons` unchanged on any failure. Callers merge the new columns onto
    trips/tours via person_id like any person covariate.
    """
    cfg = cfg or HouseholdCdapConfig()
    try:
        df = persons.copy()
        pid = cfg.person_id_col if cfg.person_id_col in df.columns else (
            'person_id' if 'person_id' in df.columns else None)
        if pid is None:
            return persons
        if 'cdap_M' in df.columns:
            return df  # already attached
        # 1. observed DAP + dummies
        try:
            _dap = derive_observed_dap(df, tours, cfg)
            df['cdap_dap'] = _dap.values
        except Exception:
            df['cdap_dap'] = 'H'
        for _d in ('M', 'N', 'H'):
            df[f'cdap_{_d}'] = (df['cdap_dap'] == _d).astype(float)
        # 2. household composition (same builder the CDAP logit estimates on)
        try:
            _comp = _hh_composition_features(df, cfg)
            for _c in ('hh_n_workers', 'hh_n_children', 'hh_has_home_adult',
                       'child_x_home_adult', 'is_child'):
                if _c in _comp.columns:
                    df[f'cdap_{_c}'] = pd.to_numeric(
                        _comp[_c], errors='coerce').fillna(0).values
        except Exception:
            pass
        for _c in ('cdap_hh_n_workers', 'cdap_hh_n_children',
                   'cdap_hh_has_home_adult', 'cdap_child_x_home_adult',
                   'cdap_is_child'):
            if _c not in df.columns:
                df[_c] = 0.0
        # 3. observed joint-tour attributes (flag or HHID+OD+time heuristic)
        try:
            _jt = derive_observed_joint_tours(tours, cfg)
            _hhid = cfg.hh_id_col if cfg.hh_id_col in df.columns else (
                'HHID' if 'HHID' in df.columns else None)
            if _jt is not None and len(_jt) and _hhid:
                _cnt = _jt.groupby(cfg.hh_id_col).size().to_dict() \
                    if cfg.hh_id_col in _jt.columns else {}
                df['n_joint_tours_hh'] = df[_hhid].map(_cnt).fillna(0).astype(float)
                _members, _psize = set(), {}
                for _, _r in _jt.iterrows():
                    _pl = _r.get('participants', [])
                    _pl = list(_pl) if isinstance(_pl, (list, tuple)) else [_pl]
                    for _p in _pl:
                        _members.add(str(_p))
                        _psize[str(_p)] = max(_psize.get(str(_p), 0), len(_pl))
                df['is_joint_traveller'] = df[pid].astype(str).isin(_members).astype(float)
                df['joint_party_n'] = df[pid].astype(str).map(_psize).fillna(0).astype(float)
            else:
                for _c in ('n_joint_tours_hh', 'is_joint_traveller', 'joint_party_n'):
                    df[_c] = 0.0
        except Exception:
            for _c in ('n_joint_tours_hh', 'is_joint_traveller', 'joint_party_n'):
                if _c not in df.columns:
                    df[_c] = 0.0
        return df
    except Exception:
        return persons


def validate_joint_cdap_inputs(persons: pd.DataFrame, tours: pd.DataFrame,
                               cfg: Optional[HouseholdCdapConfig] = None,
                               max_size: int = 5) -> dict:
    """
    Sanity-check the HHTS inputs that build the joint household CDAP models:
      * household-id column presence / coverage
      * duplicate person ids
      * purpose-column availability and how tour purposes classify to M/N
        (exact coded vs token-matched raw strings)
      * household-size buckets actually populated at estimation time
      * agreement with the individual CDAP ground truth (ANYSTOPS + MAINACT,
        the same source prepare_cdap_data uses) when those columns exist

    Returns a report dict; prints loud warnings for anything that would bias
    the estimated patterns.
    """
    cfg = cfg or HouseholdCdapConfig()
    rep = {'fatal': []}
    pid = cfg.person_id_col

    if persons[pid].duplicated().any():
        rep['duplicate_person_rows'] = int(persons[pid].duplicated().sum())
        print(f"[cdap-hh validate] WARNING {rep['duplicate_person_rows']} "
              f"duplicate person_id rows")

    hhc = next((c for c in (cfg.hh_id_col, 'HHID', 'household_id', 'hhid')
                if c in persons.columns), None)
    rep['household_col'] = hhc
    if hhc is None:
        print('[cdap-hh validate] WARNING no household-id column found — '
              'every person would be treated as a size-1 household and all '
              'interaction terms are UNIDENTIFIED. Joint estimation needs '
              'persons[HHID].')
        rep['fatal'].append('no household id column')
    elif persons[hhc].isna().any():
        rep['persons_missing_hhid'] = int(persons[hhc].isna().sum())
        print(f"[cdap-hh validate] WARNING {rep['persons_missing_hhid']} "
              f"persons have no {hhc} (excluded from joint buckets)")

    dap = derive_observed_dap(persons, tours, cfg)
    rep['persons'] = int(len(persons))
    rep['persons_with_tours'] = int(dap.attrs.get('n_persons_with_tours', 0))
    rep['dap_distribution'] = dap.value_counts().to_dict()

    purp_col = dap.attrs.get('purpose_col')
    rep['purpose_column'] = purp_col
    if purp_col is None:
        print('[cdap-hh validate] FATAL: tours table carries no purpose column '
              '(main_purpose/purpose/PURPOSE/DESTPURP)')
        rep['fatal'].append('no purpose column in tours')
    else:
        uniq = tours[purp_col].dropna().astype(str)
        uniq = uniq[~uniq.str.strip().str.lower().isin(('', 'nan', 'none'))]
        coded = {'work', 'edu', 'education', 'university', 'school'}
        low = uniq.str.strip().str.lower()
        cls = uniq.map(_purpose_is_mandatory)
        rep['unique_purposes'] = int(uniq.nunique())
        rep['mandatory_tour_share'] = round(float(cls.mean()), 4)
        rep['token_matched_share'] = round(
            float((~low.isin(coded)).mean()), 4)
        rep['top_purposes'] = {
            str(k): (int(v), 'M' if _purpose_is_mandatory(k) else 'N')
            for k, v in uniq.value_counts().head(12).items()}
        print(f"[cdap-hh validate] purpose col '{purp_col}': "
              f"{rep['unique_purposes']} unique purposes, "
              f"mandatory-tour share={rep['mandatory_tour_share']:.1%} "
              f"(token-matched non-coded: {rep['token_matched_share']:.0%})")
        if rep['mandatory_tour_share'] == 0.0:
            print('[cdap-hh validate] FATAL: zero tours classified mandatory — '
                  'check purpose strings/coding.')
            rep['fatal'].append('no mandatory tours detected')

    if hhc is not None:
        sizes = persons.groupby(hhc).size().clip(upper=max_size)
        buckets = sizes.value_counts().sort_index()
        rep['households_by_bucket'] = {int(k): int(v)
                                       for k, v in buckets.items()}
        thin = {int(k): int(v) for k, v in buckets.items() if v < 30}
        if thin:
            print(f'[cdap-hh validate] thin buckets (<30 households): {thin} '
                  '— prefer shared-coefficient mode there')

    # cross-check against the individual CDAP ground-truth columns
    if {'ANYSTOPS', 'MAINACT'}.issubset(persons.columns):
        anystops = persons['ANYSTOPS'].astype(str).str.lower().isin(
            ('true', '1', 'yes', 'y'))
        mand = persons['MAINACT'].fillna('').astype(str).map(
            _purpose_is_mandatory)
        ref = np.where(~anystops, 'H', np.where(mand, 'M', 'N'))
        agree = float((dap.to_numpy() == ref).mean())
        rep['individual_cdap_agreement'] = round(agree, 4)
        mm_mask = dap.to_numpy() != ref
        rep['mismatch_mainact_top'] = (
            persons.loc[mm_mask, 'MAINACT'].astype(str)
            .value_counts().head(8).to_dict())
        print(f"[cdap-hh validate] agreement with ANYSTOPS/MAINACT "
              f"CDAP truth: {agree:.1%}")
        if agree < 0.90:
            print('[cdap-hh validate] WARNING <90% agreement with the '
                  'individual CDAP target — top mismatching MAINACT values:')
            for k, v in rep['mismatch_mainact_top'].items():
                print(f'    {k!r}: {v}')
    return rep


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
        # standard errors, if SearchLibrium exposes them (attribute name varies)
        _se = next((np.asarray(getattr(mnl, a), dtype=float)
                    for a in ('coeff_se', 'stderr', 'std_err', 'se', 'coeff_stderr')
                    if getattr(mnl, a, None) is not None), None)
        coeff_se = dict(zip(coeff_names, _se)) if (_se is not None and len(_se) == len(coeff_names)) else {}
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
                'loglik': fitted_loglik, 'coeff_names': coeff_names,
                'coeff_est': beta, 'coeff_se': coeff_se}
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
def _save_household_cdap_html(df: pd.DataFrame, out_dir: str) -> None:
    """Render the estimated household-CDAP / joint-tour coefficients as an
    ActivitySim-style HTML report (household_cdap_coefficients.html): individual
    DAP utilities, the 2/3/4/5-way household INTERACTION_COEFFICIENTS, and the
    joint-tour frequency / participation / purpose coefficients — one table per
    model group, matching the layout of ActivitySim's CDAP component docs so the
    joint coefficients are inspectable next to the individual utilities."""
    # model-group → (section title, explanatory note), in display order
    _SECTIONS = [
        ('cdap_dap_utility',      'Individual DAP utilities',
         'Per person-type utility of M (mandatory) and N (non-mandatory); H (home) is the base = 0.'),
        ('interaction_2way',      'Household interaction coefficients — 2-way',
         'ActivitySim INTERACTION_COEFFICIENTS: utility added when a household pair shares an activity pattern.'),
        ('interaction_3way',      'Household interaction coefficients — 3-way',
         'Utility added when three household members share an activity pattern.'),
        ('interaction_4way',      'Household interaction coefficients — 4-way',
         'Utility added when four household members share an activity pattern.'),
        ('interaction_5way',      'Household interaction coefficients — 5-way',
         'Utility added when five household members share an activity pattern.'),
        ('cdap_interaction',      'Household interaction (scalar composition boosts)',
         'Legacy scalar coordination effects folded into the config.'),
        ('joint_frequency',       'Joint-tour frequency',
         'Poisson rate of joint tours per household by composition.'),
        ('joint_participation',   'Joint-tour participation',
         'Per-role probability that a member joins a joint tour.'),
        ('joint_purpose_utility', 'Joint-tour purpose utility',
         'Composition-conditional purpose utilities (link to Stage-1 activity purpose).'),
    ]

    def _fmt(x, nd=4):
        try:
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return '—'
            return f'{float(x):.{nd}f}'
        except Exception:
            return '—'

    def _stars(p):
        try:
            p = float(p)
        except Exception:
            return ''
        if not np.isfinite(p):
            return ''
        return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

    parts = [
        '<html><head><meta charset="utf-8"><title>Household CDAP &amp; joint-tour coefficients</title>',
        '<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;color:#1a1a1a}'
        'h2{margin-top:0}h3{margin:22px 0 4px;color:#0b5}'
        'p.note{margin:2px 0 8px;color:#555;font-size:13px}'
        'table{border-collapse:collapse;margin:4px 0 10px;font-size:13px}'
        'th,td{border:1px solid #ccc;padding:3px 9px;text-align:right}'
        'th{background:#f2f2f2}td.p{text-align:left}'
        '.sig{color:#0a7}.na{color:#bbb}</style></head><body>',
        '<h2>Household CDAP &amp; joint-tour coefficients</h2>',
        '<p class="note">Estimated from HHTS. Mirrors ActivitySim CDAP: individual DAP utilities, '
        'household interaction coefficients (2/3/4/5-way), and the joint-tour models. '
        '<code>t</code> and <code>p</code> are shown where the estimator exposed standard errors '
        '(stars: *** p&lt;.01, ** p&lt;.05, * p&lt;.10).</p>',
    ]
    has_col = set(df.columns)
    for _model, _title, _note in _SECTIONS:
        _sub = df[df['model'] == _model]
        if _sub.empty:
            continue
        parts.append(f'<h3>{_title}</h3><p class="note">{_note}</p>')
        parts.append('<table><tr><th class="p">parameter</th><th>value</th>'
                     '<th>std err</th><th>t</th><th>p</th></tr>')
        for _, r in _sub.iterrows():
            _se = r.get('std_err', np.nan) if 'std_err' in has_col else np.nan
            _t  = r.get('t_stat',  np.nan) if 't_stat'  in has_col else np.nan
            _p  = r.get('p_value', np.nan) if 'p_value' in has_col else np.nan
            _star = _stars(_p)
            parts.append(
                f'<tr><td class="p">{r["param"]}</td>'
                f'<td>{_fmt(r["value"])}</td>'
                f'<td class="{"na" if _se!=_se else ""}">{_fmt(_se)}</td>'
                f'<td>{_fmt(_t, 2)}</td>'
                f'<td class="sig">{_fmt(_p, 3)}{(" " + _star) if _star else ""}</td></tr>'
            )
        parts.append('</table>')
    parts.append('</body></html>')
    try:
        with open(os.path.join(out_dir, 'household_cdap_coefficients.html'), 'w',
                  encoding='utf-8') as fh:
            fh.write('\n'.join(parts))
        print(f"[cdap-hh estimate] wrote household_cdap_coefficients.html")
    except Exception as _e:
        print(f"[cdap-hh estimate] html save skipped: {_e}")


def save_household_cdap_coefficients(dap: dict, freq: dict, part: dict, purp: dict,
                                     out_dir: str, cfg=None) -> None:
    """Write the estimated household-CDAP / joint-tour coefficients to
    household_cdap_coefficients.csv (+ .pkl) so they are inspectable like the
    other stage models: the M/N DAP utilities per person-type, the household
    interaction coefficients (ActivitySim INTERACTION_COEFFICIENTS analogue), and
    the joint-tour frequency / participation / purpose coefficients. Includes a
    std_err / p_value column where the estimator exposed standard errors."""
    from scipy.stats import norm as _norm
    rows = []

    def _row(model, param, value, se=np.nan):
        se = float(se) if se is not None and np.isfinite(se) else np.nan
        t = value / se if (np.isfinite(se) and se > 0) else np.nan
        p = float(2.0 * _norm.sf(abs(t))) if np.isfinite(t) else np.nan
        rows.append({'model': model, 'param': param, 'value': float(value),
                     'std_err': se, 't_stat': t, 'p_value': p})

    se_map = dap.get('coeff_se', {}) or {}
    for pt, u in dap.get('dap_utility', {}).items():
        _row('cdap_dap_utility', f'{pt}.M', u.get('M', 0.0), se_map.get(f'pt_{pt}.1'))
        _row('cdap_dap_utility', f'{pt}.N', u.get('N', 0.0), se_map.get(f'pt_{pt}.2'))
    for k, v in (dap.get('interaction', {}) or {}).items():
        _row('cdap_interaction', k, v)
    for k in ('joint_base_rate', 'joint_rate_per_child', 'joint_rate_per_adult',
              'joint_rate_if_multi_vehicle'):
        if k in freq:
            _row('joint_frequency', k, freq[k])
    for k in ('p_join_adult', 'p_join_child'):
        if part.get(k) is not None:
            _row('joint_participation', k, part[k])
    for pur, u in (purp.get('utility', {}) or {}).items():
        _row('joint_purpose_utility', pur, u)
    # Household interaction coefficients (ActivitySim INTERACTION_COEFFICIENTS):
    # the 2-way table plus the higher-order 3/4/5-way tables from the config.
    if cfg is not None:
        for _order, _attr in ((2, 'interaction'), (3, 'interaction3'),
                              (4, 'interaction4'), (5, 'interaction5')):
            _tbl = getattr(cfg, _attr, None) or {}
            for _key, _v in _tbl.items():
                _row(f'interaction_{_order}way', '|'.join(map(str, _key)), _v)

    try:
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, 'household_cdap_coefficients.csv'), index=False)
        with open(os.path.join(out_dir, 'household_cdap_coefficients.pkl'), 'wb') as fh:
            pickle.dump({'dap': dap, 'freq': freq, 'participation': part, 'purpose': purp}, fh)
        _save_household_cdap_html(df, out_dir)
        print(f"[cdap-hh estimate] saved {len(df)} coefficients -> "
              f"{out_dir}/household_cdap_coefficients.csv")
    except Exception as _e:
        print(f"[cdap-hh estimate] save skipped: {_e}")


# ---------------------------------------------------------------------------
# 1b. JOINT household CDAP, stratified by household size 1..5 (ActivitySim)
# ---------------------------------------------------------------------------
# ActivitySim estimates CDAP as a JOINT choice per household: the alternatives
# are the Cartesian product of every member's {M,N,H} pattern, utilities are
# additive in member terms plus 2/3/4/5-way intra-household interaction terms,
# estimated SEPARATELY for each household size (1,2,3,4,5+). This function fits
# exactly that model on HHTS and emits coefficients in the same key format as
# HouseholdCdapConfig.interaction / interaction3 / interaction4 / interaction5,
# so the heuristic priors can be replaced by estimated behaviour with no
# call-site changes (coordinate_household_cdap consumes the same tables).

_JOINT_DAP_PROFILE_VARS = [
    # compact individual utility covariates — all derived defensively below
    'is_female', 'has_vehicle', 'has_licence', 'income_midplus',
    'age_child', 'age_teen', 'age_working', 'age_older', 'age_senior',
    'mandatory_role',
]
_JOINT_CHILD_PTYPES = ('SchoolChild', 'Preschool')


def _joint_dap_profile(persons: pd.DataFrame) -> pd.DataFrame:
    """Self-contained person-profile matrix for the joint CDAP (no Larch)."""
    df = persons.copy()
    if cfg_ptype_col() not in df.columns:
        for cand in ('ptype', 'PTYPE', 'person_type'):
            if cand in df.columns:
                df = df.rename(columns={cand: cfg_ptype_col()})
                break
    pt = df.get(cfg_ptype_col())
    if pt is not None:
        pt = pt.astype(str)
    age = pd.to_numeric(df.get('AGEGROUP', pd.Series(np.nan, index=df.index)),
                        errors='coerce')
    out = pd.DataFrame(index=df.index)
    out['is_female'] = (pd.to_numeric(df['sexp'], errors='coerce') == 2) \
        .astype(float) if 'sexp' in df.columns else 0.0
    out['has_vehicle'] = (pd.to_numeric(df['vehrd'], errors='coerce').fillna(0) > 0) \
        .astype(float) if 'vehrd' in df.columns else 0.0
    out['has_licence'] = pd.to_numeric(df['CARLICENCE'], errors='coerce') \
        .fillna(0).clip(0, 1).astype(float) if 'CARLICENCE' in df.columns else 0.0
    out['income_midplus'] = (pd.to_numeric(df['incp'], errors='coerce') >= 8) \
        .astype(float) if 'incp' in df.columns else 0.0
    if age.notna().any():
        out['age_child']   = (age <= 3).astype(float)    # 0-14
        out['age_teen']    = (age == 4).astype(float)    # 15-19
        out['age_working'] = age.between(5, 8).astype(float)
        out['age_older']   = (age >= 9).astype(float)    # 40+
        out['age_senior']  = (age >= 11).astype(float)   # 50+
    else:
        for c in ('age_child', 'age_teen', 'age_working', 'age_older', 'age_senior'):
            out[c] = 0.0
    if pt is not None:
        for d in pt.unique():
            out[f'pt_{d}'] = (pt == d).astype(float)
        out['mandatory_role'] = pt.isin(
            ['FullTimeWorker', 'PartTimeWorker', 'WorkingStudent',
             'UniversityStudent', 'SchoolChild']).astype(float)
        child_mask = pt.isin(_JOINT_CHILD_PTYPES).to_numpy()
    else:
        _ws = df['WORKSTAT'] if 'WORKSTAT' in df.columns \
            else pd.Series(0, index=df.index)
        out['mandatory_role'] = (pd.to_numeric(_ws, errors='coerce') >= 1) \
            .astype(float)
        child_mask = (age <= 3).fillna(False).to_numpy()
    out['_is_child'] = child_mask.astype(float)
    return out.fillna(0.0)


def cfg_ptype_col() -> str:
    """Default ptype column used by the joint CDAP builder."""
    return 'ptype'


def _fit_conditional_logit(situations, l2: float = 1e-4, maxiter: int = 800):
    """
    McFadden conditional logit on pre-built alternative designs.

    situations : list of (Z, y_idx) where Z is (n_alts × n_feat) and y_idx the
                 chosen row. One shared coefficient vector θ; P(a) ∝ exp(z_a·θ).
    Returns dict(theta, std_err, ll, k, n_sit, converged).
    Standard errors come from scipy's BFGS/L-BFGS inverse-Hessian approximation.
    """
    from scipy.optimize import minimize

    K = situations[0][0].shape[1]

    def neg_ll(th):
        ll = 0.0
        g = np.zeros(K)
        for Z, ych in situations:
            v = Z @ th
            v = v - v.max()
            p = np.exp(v)
            s = p.sum()
            ll += v[ych] - np.log(s)
            # g accumulates d(dataNegLL)/dth = -(z_y - sum_a p_a z_a)
            g -= Z[ych] - (p @ Z) / s
        return -ll + l2 * float(th @ th), g + 2.0 * l2 * th

    th0 = np.zeros(K)
    res = minimize(neg_ll, th0, jac=True, method='L-BFGS-B',
                   options={'maxiter': 2000, 'ftol': 1e-9})
    theta = res.x
    ll = -res.fun + l2 * float(theta @ theta)
    try:
        se = np.sqrt(np.abs(np.asarray(res.hess_inv.todense()).diagonal()))
    except Exception:
        se = np.full(K, np.nan)
    # status 1 = iteration cap: with a shared-θ stacked design this still yields
    # stable, usable estimates, so treat it as converged for the caller's gate.
    return {'theta': theta, 'std_err': se, 'll': float(ll), 'k': K,
            'n_sit': len(situations),
            'converged': bool(res.success or res.status in (0, 1))}


def estimate_joint_cdap_by_hh_size(persons: pd.DataFrame,
                                   tours: pd.DataFrame,
                                   cfg: Optional[HouseholdCdapConfig] = None,
                                   max_size: int = 5,
                                   l2: float = 1e-4,
                                   per_size: Optional[bool] = None) -> dict:
    """
    Estimate the ActivitySim-style JOINT household CDAP separately for each
    household size 1..max_size (5+ collapsed into the max_size bucket, members
    capped adults-first like coordinate_household_cdap's n_cdap_persons).

    Utility of a joint pattern combination:
        U(combo) = Σ_i [ASC_{d_i} + x_i·β_{d_i}]           (individual part;
                     H is the reference alternative)
                 + Σ_{i<j} θ2(class_ij, d_i, d_j)          (2-way interactions)
                 + Σ_{k-subsets, k=3..s} θk(gcls, d)       (all-same-pattern)

    Two coefficient regimes
    -----------------------
    * shared (default): ONE coefficient vector estimated across all sizes —
      robust when sizes 3–5 have few households. Structurally-inapplicable
      terms shrink to ~0 under L2.
    * per_size=True (or CDAP_HH_JOINT_PER_SIZE=1): a DEDICATED model per
      household size that perturbs ONLY the terms appropriate to that size —
        size 1: ASC + individual β            (no intra-household terms exist)
        size 2: + 2-way pair interactions
        size 3: + 3-way · size 4: + 4-way · size 5: + 5-way
      Each size's coefficients, interaction tables and fit statistics are
      reported independently under 'size_models'. The prediction-side config
      continues to use the shared tables (the runtime coordination layer holds
      one table); the dedicated per-size models are saved for inspection and
      future per-size runtime use.

    Returns a dict with:
      sizes       : {s: {n_hh, ll, acc, bic}}   fit diagnostics per bucket
      coef        : DataFrame(param, value, std_err, t_stat, p_value)  [shared]
      size_models : {} or {s: {coef, interaction, interaction3/4/5, ll, acc,
                                bic, n_hh, converged}}                [per_size]
      interaction / interaction3 / interaction4 / interaction5 :
              estimated SHARED tables keyed EXACTLY like HouseholdCdapConfig's
      dap_utility : {ptype: {M, N, H}} evaluated at each ptype's mean profile
    """
    import itertools
    from scipy.stats import norm as _norm

    if per_size is None:
        per_size = os.environ.get('CDAP_HH_JOINT_PER_SIZE', '0') == '1'

    cfg = cfg or HouseholdCdapConfig()
    hh_col = next((c for c in (cfg.hh_id_col, 'HHID', 'household_id', 'hhid')
                   if c in persons.columns), None)
    prof = _joint_dap_profile(persons)
    pt_series = persons[cfg_ptype_col()].astype(str) \
        if cfg_ptype_col() in persons.columns else None
    dap = derive_observed_dap(persons, tours, cfg)

    # ── HHTS input validation: are we really seeing the right patterns? ────
    validation = validate_joint_cdap_inputs(persons, tours, cfg,
                                            max_size=max_size)
    if validation.get('fatal'):
        raise RuntimeError(
            'joint household CDAP: input validation failed — '
            + '; '.join(validation['fatal']))

    # ── assemble estimation situations per household-size bucket ─────────
    member_rows = []   # one entry per household: (bucket_size, prof_rows, children, daps)
    if hh_col is not None:
        groups = persons.groupby(hh_col, sort=False)
    else:
        groups = [(i, persons.iloc[[i]]) for i in range(len(persons))]
    for _, grp in groups:
        idx = grp.index
        child = prof.loc[idx, '_is_child'].to_numpy().astype(bool)
        # deterministic ordering mirroring coordinate_household_cdap:
        # adults first, then by person_id string
        pid_key = [str(x) for x in idx]
        order = sorted(range(len(idx)),
                       key=lambda k: (bool(child[k]), str(pid_key[k])))
        core = order[:max(1, int(max_size))]
        bucket = len(core)
        member_rows.append((
            bucket,
            prof.loc[idx[core]],
            [bool(child[k]) for k in core],
            [dap.loc[i] for i in idx[core]],
        ))

    var_cols = [c for c in prof.columns if c != '_is_child']
    DAPS = ['H', 'M', 'N']                     # alphabetical: matches sorted() lookups
    param_names: list = []
    col_defs: list = []                        # name → builder fn(prof_rows, daps_tuple)

    # All column builders share one signature: (mlist, children, daps) where
    # mlist is a list of per-member profile dicts, children the matching
    # is-child flags, and daps the combination tuple being scored.
    def _add(name, fn):
        param_names.append(name)
        col_defs.append(fn)

    # ASC_N / ASC_M (H reference)
    _add('ASC_N', lambda ml, ch, ds: float(sum(1 for d in ds if d == 'N')))
    _add('ASC_M', lambda ml, ch, ds: float(sum(1 for d in ds if d == 'M')))
    # individual covariates, alternative-specific (N / M)
    for d in ('N', 'M'):
        for k in var_cols:
            _add(f'b_{d}_{k}',
                 lambda ml, ch, ds, d=d, k=k:
                 float(sum(r.get(k, 0.0) for r, dd in zip(ml, ds) if dd == d)))
    # 2-way interactions: count of pairs matching (class, unordered patterns)
    i2_start = len(param_names)
    for cls in ('AA', 'AC', 'CC'):
        for a, b in itertools.combinations_with_replacement(DAPS, 2):
            _add(f'i2_{cls}_{a}{b}',
                 lambda ml, ch, ds, cls=cls, a=a, b=b:
                 float(sum(1 for i in range(len(ds)) for j in range(i + 1, len(ds))
                           if (('CC' if (ch[i] and ch[j])
                               else 'AC' if (ch[i] or ch[j]) else 'AA') == cls
                               and tuple(sorted((ds[i], ds[j])))
                               == tuple(sorted((a, b)))))))
    i2_end = len(param_names)
    # higher-order all-same-pattern subsets (3/4/5-way). Term blocks are laid
    # out contiguously so a size-s model can restrict itself to the terms that
    # structurally exist at that size: core + i2 + ... + i{min(s,5)}.
    order_ends = {}
    for kk in (3, 4, 5):
        for n_c in range(kk, -1, -1):
            gcls = 'A' * (kk - n_c) + 'C' * n_c
            for d in DAPS:
                _add(f'i{kk}_{gcls}_{d}',
                     lambda ml, ch, ds, kk=kk, gcls=gcls, d=d:
                     float(sum(1 for sub in itertools.combinations(range(len(ds)), kk)
                               if ('A' * (kk - sum(1 for t in sub if ch[t]))
                                   + 'C' * sum(1 for t in sub if ch[t])) == gcls
                               and all(ds[t] == d for t in sub))))
        order_ends[kk] = len(param_names)

    def _term_slice_for_size(s: int) -> slice:
        """Columns appropriate for a household of size s: core terms plus every
        interaction order up to min(s, 5) — nothing else is estimable there."""
        top = min(s, 5)
        if s < 2:
            return slice(0, i2_start)          # ASC + individual β only
        if top < 3:
            return slice(0, i2_end)            # + 2-way
        return slice(0, order_ends[top])       # + 3/4/5-way

    situations = []
    sit_buckets = []
    for bucket, mrows, children, daps_list in member_rows:
        mlist = mrows[var_cols].to_dict('records') if hasattr(mrows, 'columns') \
            else mrows
        combos = list(itertools.product(DAPS, repeat=bucket))
        obs_key = tuple(daps_list)
        if obs_key not in combos:
            continue
        Zbase = np.empty((len(combos), len(param_names)), dtype=float)
        for ci, combo in enumerate(combos):
            Zbase[ci] = [fn(mlist, children, combo) for fn in col_defs]
        situations.append((Zbase, combos.index(obs_key)))
        sit_buckets.append(bucket)

    if not situations:
        raise RuntimeError('joint CDAP: no usable households after DAP derivation')

    fit = _fit_conditional_logit(situations, l2=l2)
    theta, se = fit['theta'], fit['std_err']

    # per-bucket diagnostics under the shared θ
    ll_by, acc_by, nh_by = {}, {}, {}
    for (Z, y_idx), bucket in zip(situations, sit_buckets):
        v = Z @ theta
        logsum = float(np.log(np.exp(v - v.max()).sum())) + float(v.max())
        ll_by[bucket] = ll_by.get(bucket, 0.0) + float(v[y_idx] - logsum)
        acc_by[bucket] = acc_by.get(bucket, 0) + int(int(np.argmax(v)) == y_idx)
        nh_by[bucket] = nh_by.get(bucket, 0) + 1
    sizes_out = {}
    for s in sorted(nh_by):
        sizes_out[s] = {
            'n_hh': int(nh_by[s]),
            'll': round(ll_by[s], 3),
            'acc': round(acc_by[s] / nh_by[s], 4),
            'bic': round(-2 * ll_by[s] + fit['k'] * np.log(max(nh_by[s], 1)), 3),
        }

    def _extract_tables(names, vals):
        """Interaction tables from a coefficient vector, keyed exactly like
        HouseholdCdapConfig.interaction / interaction3 / interaction4/5."""
        inter = {}
        for pname, val in zip(names, vals):
            if pname.startswith('i2_'):
                cls, ab = pname[3:].split('_', 1)
                inter[(cls, ab[0], ab[1])] = round(float(val), 6)

        def _ho(kk):
            tbl = {}
            pre = f'i{kk}_'
            for pname, val in zip(names, vals):
                if pname.startswith(pre):
                    body = pname[len(pre):]     # gcls run then '_' + pattern
                    gi = 0
                    while gi < len(body) and body[gi] in 'AC':
                        gi += 1
                    tbl[(body[:gi], body[gi + 1:])] = round(float(val), 6)
            return tbl
        return inter, {kk: _ho(kk) for kk in (3, 4, 5)}

    # ── dedicated per-size models: perturb ONLY the appropriate terms ──────
    # size 1 → ASC+β · size 2 → +2-way · size s → +{3..s}-way interactions.
    size_models = {}
    if per_size:
        _min_hh = max(1, int(os.environ.get('CDAP_HH_JOINT_MIN_HH', '30')))
        for s in sorted(set(sit_buckets)):
            sl = _term_slice_for_size(s)
            sits_s = [(Z[:, sl], y)
                      for (Z, y), b in zip(situations, sit_buckets) if b == s]
            n_hh_s = len(sits_s)
            if n_hh_s < _min_hh:
                print(f"[cdap-hh estimate] joint CDAP size {s}: only "
                      f"{n_hh_s} households (< {_min_hh}) — no dedicated model")
                continue
            fit_s = _fit_conditional_logit(sits_s, l2=l2)
            th_s = fit_s['theta']
            names_s = param_names[sl]
            ll_s, acc_s = 0.0, 0
            for Z, y_idx in sits_s:
                v = Z @ th_s
                logsum = float(np.log(np.exp(v - v.max()).sum())) + float(v.max())
                ll_s += float(v[y_idx] - logsum)
                acc_s += int(int(np.argmax(v)) == y_idx)
            coef_s = pd.DataFrame({'param': names_s, 'value': th_s,
                                   'std_err': fit_s['std_err']})
            coef_s['t_stat'] = coef_s['value'] / coef_s['std_err'].replace(0, np.nan)
            coef_s['p_value'] = [float(2.0 * _norm.sf(abs(t)))
                                 if np.isfinite(t) else np.nan
                                 for t in coef_s['t_stat']]
            inter_s, ho_s = _extract_tables(names_s, th_s)
            size_models[s] = {
                'n_hh': n_hh_s,
                'll': round(ll_s, 3),
                'acc': round(acc_s / n_hh_s, 4),
                'bic': round(-2 * ll_s + len(names_s) * np.log(n_hh_s), 3),
                'converged': fit_s['converged'],
                'coef': coef_s,
                'terms': {'core': True,
                          'i2': s >= 2, 'i3': s >= 3,
                          'i4': s >= 4, 'i5': s >= 5},
                'interaction': inter_s,
                'interaction3': ho_s[3], 'interaction4': ho_s[4],
                'interaction5': ho_s[5],
            }
            print(f"[cdap-hh estimate] joint CDAP size {s}: dedicated model on "
                  f"{len(names_s)} terms — n={n_hh_s} acc={size_models[s]['acc']:.3f}")

    coef_df = pd.DataFrame({
        'param': param_names,
        'value': theta,
        'std_err': se,
    })
    coef_df['t_stat'] = coef_df['value'] / coef_df['std_err'].replace(0, np.nan)
    coef_df['p_value'] = [float(2.0 * _norm.sf(abs(t))) if np.isfinite(t) else np.nan
                          for t in coef_df['t_stat']]

    interaction, ho_tables = _extract_tables(param_names, theta)

    # dap_utility per ptype: ASC + β·mean(x) evaluated at each ptype mean profile
    dap_utility = {}
    if pt_series is not None:
        bcols = {(d, k): f'b_{d}_{k}' for d in ('N', 'M') for k in var_cols}
        th_map = dict(zip(param_names, theta))
        for pt_val, grp_idx in prof.groupby(pt_series.values).groups.items():
            mx = prof.loc[grp_idx, var_cols].mean()
            u = {}
            for d in DAPS:
                if d == 'H':
                    u[d] = 0.0
                else:
                    u[d] = float(th_map.get(f'ASC_{d}', 0.0)
                                 + sum(th_map.get(bcols[(d, k)], 0.0) * float(mx[k])
                                       for k in var_cols))
            u = {dd: round(max(min(vv, 12.0), -12.0), 4) for dd, vv in u.items()}
            dap_utility[str(pt_val)] = u

    return {
        'sizes': sizes_out,
        'coef': coef_df,
        'theta': dict(zip(param_names, map(float, theta))),
        'interaction': interaction,
        'interaction3': ho_tables[3],
        'interaction4': ho_tables[4],
        'interaction5': ho_tables[5],
        'size_models': size_models,
        'per_size': bool(per_size),
        'input_validation': validation,
        'dap_utility': dap_utility,
        'converged': fit['converged'],
        'n_households': int(sum(nh_by.values())),
        'model_form': ('joint MNL over product(M,N,H)^s per household size s=1..%d; '
                       'individual terms (ASC + x·β, ref=H) + 2-way '
                       '(class, sorted-pattern-pair) + 3/4/5-way all-same-pattern '
                       'interactions; coefficient regime: %s; L2=%g'
                       % (max_size,
                          'dedicated per-size (appropriate terms only)'
                          if per_size else 'shared across sizes',
                          l2)),
    }


def save_joint_cdap_html(res: dict, out_dir: str,
                         fname: str = 'stage0_cdap_household_coefficients.html') -> None:
    """Render the per-household-size joint CDAP model as an HTML report."""
    import html as _h
    os.makedirs(out_dir, exist_ok=True)
    parts = [
        '<html><head><meta charset="utf-8">'
        '<title>Joint household CDAP by household size</title>'
        '<style>body{font-family:Arial;margin:20px;font-size:13px}'
        'table{border-collapse:collapse;margin-bottom:18px}'
        'th{background:#2c3e50;color:#fff;padding:5px 10px;text-align:left}'
        'td{border:1px solid #ddd;padding:4px 10px}'
        '</style></head><body>',
        '<h2>Joint household CDAP — one coordinated plan per household size</h2>',
        '<p class="note">ActivitySim-style joint logit: alternatives are the '
        'Cartesian product of each member\'s {M,N,H} pattern, stratified by '
        'household size 1..5 (5+ capped adults-first). Individual terms '
        '(ASC + x·β, reference = H) plus 2/3/4/5-way intra-household '
        'interaction coefficients, shared across sizes.</p>',
    ]
    if res.get('sizes'):
        rows = ''.join(
            f"<tr><td>{s}</td><td>{v['n_hh']}</td><td>{v['ll']}</td>"
            f"<td>{v['acc']}</td><td>{v['bic']}</td></tr>"
            for s, v in sorted(res['sizes'].items()))
        parts.append('<h3>Per-size fit (shared coefficients)</h3>'
                     '<table><tr><th>HH size</th><th>n households</th>'
                     '<th>LogLik</th><th>arg-max accuracy</th><th>BIC</th></tr>'
                     + rows + '</table>')
    coef = res.get('coef')
    if coef is not None and len(coef):
        def _star(p):
            if not np.isfinite(p):
                return ''
            return '***' if p < 0.001 else '**' if p < 0.01 else \
                '*' if p < 0.05 else '.' if p < 0.10 else ''
        rows = ''.join(
            f"<tr><td>{_h.escape(str(r.param))}</td><td>{r.value:.4f}</td>"
            f"<td>{r.std_err:.4f}</td><td>{r.t_stat:.2f}</td>"
            f"<td>{r.p_value:.3g} {_star(r.p_value)}</td></tr>"
            for r in coef.itertuples())
        parts.append('<h3>Coefficients (shared across sizes)</h3>'
                     '<table><tr><th>param</th><th>value</th><th>std_err</th>'
                     '<th>t</th><th>p</th></tr>' + rows + '</table>')
    for label, key in (('Estimated 2-way interactions '
                        '(pair class, pattern pair)', 'interaction'),
                       ('Estimated 3-way (all-same-pattern)',
                        'interaction3'),
                       ('Estimated 4-way', 'interaction4'),
                       ('Estimated 5-way', 'interaction5')):
        tbl = res.get(key) or {}
        if tbl:
            rows = ''.join(f"<tr><td>{_h.escape(str(k))}</td><td>{v:.4f}</td></tr>"
                           for k, v in sorted(tbl.items(), key=str))
            parts.append(f'<h3>{label}</h3><table><tr><th>key</th>'
                         f'<th>value</th></tr>{rows}</table>')

    # ── dedicated per-size models (CDAP_HH_JOINT_PER_SIZE=1) ───────────────
    # Each household size gets its OWN estimated model, perturbing only the
    # terms structurally appropriate to that size.
    for s, sm in sorted((res.get('size_models') or {}).items()):
        _terms = sm.get('terms', {})
        _tlist = ', '.join(t for t, on in _terms.items() if on)
        parts.append(
            f"<h3>Dedicated model — household size {s}</h3>"
            f"<p class='note'>perturbed terms: {_h.escape(_tlist)} | "
            f"n={sm['n_hh']} households | LogLik={sm['ll']} | "
            f"acc={sm['acc']} | BIC={sm['bic']}</p>")
        _c = sm.get('coef')
        if _c is not None and len(_c):
            def _star2(p):
                if not np.isfinite(p):
                    return ''
                return '***' if p < 0.001 else '**' if p < 0.01 else \
                    '*' if p < 0.05 else '.' if p < 0.10 else ''
            _rows = ''.join(
                f"<tr><td>{_h.escape(str(r.param))}</td>"
                f"<td>{r.value:.4f}</td><td>{_star2(r.p_value)}</td></tr>"
                for r in _c.itertuples())
            parts.append('<table><tr><th>param</th><th>value</th>'
                         '<th>sig</th></tr>' + _rows + '</table>')
    du = res.get('dap_utility') or {}
    if du:
        rows = ''.join(
            f"<tr><td>{_h.escape(str(pt))}</td><td>{u.get('M', 0):.3f}</td>"
            f"<td>{u.get('N', 0):.3f}</td><td>{u.get('H', 0):.3f}</td></tr>"
            for pt, u in sorted(du.items()))
        parts.append('<h3>Implied per-person-type DAP utilities (mean profile)</h3>'
                     '<table><tr><th>ptype</th><th>M</th><th>N</th><th>H</th>'
                     '</tr>' + rows + '</table>')
    parts.append('</body></html>')
    try:
        with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(parts))
        print(f"[cdap-hh estimate] wrote {fname}")
    except Exception as _e:
        print(f"[cdap-hh estimate] joint-cdap html save skipped: {_e}")



def estimate_household_cdap(persons: pd.DataFrame, tours: pd.DataFrame,
                            base: Optional[HouseholdCdapConfig] = None,
                            out_dir: Optional[str] = None) -> HouseholdCdapConfig:
    """Fit all household-CDAP models and fold the coefficients into a
    HouseholdCdapConfig.

    Models: person-level DAP MNL (dap_utility + composition boosts), the
    ActivitySim-style JOINT CDAP stratified by household size 1..5 (replaces
    the heuristic interaction / interaction3 / interaction4 / interaction5
    priors), joint-tour frequency (Poisson), participation and purpose.
    When out_dir is given, also save the estimated coefficients (incl.
    joint-tour) to household_cdap_coefficients.csv/.pkl."""
    cfg = base or HouseholdCdapConfig()
    dap = estimate_cdap_coefficients(persons, tours, cfg)
    freq = estimate_joint_frequency(persons, tours, cfg)
    part = estimate_joint_participation(persons, tours, cfg)
    purp = estimate_joint_purpose(persons, tours, cfg)

    # ── Joint-by-household-size CDAP (ActivitySim) ────────────────────────
    # Estimated joint pattern-combination logit; overwrites the prior
    # interaction tables with estimated behaviour when it fits cleanly.
    joint = None
    try:
        joint = estimate_joint_cdap_by_hh_size(persons, tours, cfg)
        if joint.get('converged'):
            if joint['interaction']:
                cfg.interaction.clear()
                cfg.interaction.update(joint['interaction'])
            for attr in ('interaction3', 'interaction4', 'interaction5'):
                tbl = joint.get(attr)
                if tbl:
                    getattr(cfg, attr).clear()
                    getattr(cfg, attr).update(tbl)
            # Draw-stage tables: dedicated per-size models (when the per-size
            # regime ran) let each household co-determine its members from
            # ITS OWN size's coefficients.
            if joint.get('size_models'):
                cfg.per_size_tables = {
                    int(s): {'interaction': sm.get('interaction') or {},
                             'interaction3': sm.get('interaction3') or {},
                             'interaction4': sm.get('interaction4') or {},
                             'interaction5': sm.get('interaction5') or {}}
                    for s, sm in joint['size_models'].items()}
        cfg._joint_cdap_by_size = joint   # type: ignore[attr-defined]
        _sz = joint.get('sizes', {})
        print("[cdap-hh estimate] joint-by-size CDAP: "
              + ", ".join(f"size {s}: n={v['n_hh']} acc={v['acc']:.3f}"
                          for s, v in sorted(_sz.items())))
    except Exception as _jhh_e:
        print(f"[cdap-hh estimate] joint-by-size CDAP skipped ({_jhh_e}); "
              "coordination keeps prior interaction coefficients")

    cfg.dap_utility.update(dap['dap_utility'])
    # Prefer the JOINT model's implied per-ptype utilities (they are identified
    # against observed household patterns, not just marginal ones).
    if joint is not None and joint.get('converged') and joint.get('dap_utility'):
        cfg.dap_utility.update(joint['dap_utility'])
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
    if out_dir:
        save_household_cdap_coefficients(dap, freq, part, purp, out_dir, cfg=cfg)
        try:
            import pickle as _pk
            with open(os.path.join(out_dir, 'stage0_cdap_household_models.pkl'),
                      'wb') as _fh:
                _pk.dump({'joint_by_size': joint, 'config': cfg}, _fh)
        except Exception as _pe:
            print(f"[cdap-hh estimate] stage0_cdap_household_models.pkl "
                  f"save skipped: {_pe}")
        if joint is not None:
            save_joint_cdap_html(
                joint, out_dir,
                fname='stage0_cdap_household_coefficients.html')
    return cfg


def load_household_cdap_config(pkl_path: str,
                               base: Optional[HouseholdCdapConfig] = None) -> HouseholdCdapConfig:
    """Load a fitted household-CDAP config saved by estimation.

    Handles both on-disk formats:
      * predict-time saves: {'joint_by_size': joint, 'config': cfg}
        → returns the saved config (column-name fields refreshed from
        `base` when given so hh/person id columns match the caller).
      * stage-0 shard saves: the joint-by-size result dict itself
        (keys sizes/coef/theta/interaction/interaction3/4/5/...) →
        folds the estimated interaction tables (+ per-size tables,
        dap_utility when present) into `base` (or a fresh config).

    Individual-level theta terms (ASC_N/M, b_N_*/b_M_*) are reported in
    the HTML/CSV but intentionally NOT mapped onto per-ptype
    dap_utility priors (different structure); the joint tables are the
    H/M/N coordination core. Raises informative errors (never silent
    priors) so callers fall back explicitly.
    """
    import pickle as _pk
    with open(pkl_path, 'rb') as _fh:
        obj = _pk.load(_fh)
    if isinstance(obj, dict) and 'config' in obj and obj['config'] is not None:
        cfg = obj['config']
        if base is not None:
            for _f in ('hh_id_col', 'person_id_col', 'ptype_col',
                       'vehicles_col', 'seed'):
                try:
                    setattr(cfg, _f, getattr(base, _f))
                except Exception:
                    pass
        return cfg
    joint = None
    if isinstance(obj, dict) and 'joint_by_size' in obj and obj['joint_by_size'] is not None:
        joint = obj['joint_by_size']
    elif isinstance(obj, dict) and 'interaction' in obj:
        joint = obj
    if joint is None:
        raise ValueError(f"{pkl_path}: unrecognised household-CDAP content "
                         f"(keys={sorted(obj.keys()) if isinstance(obj, dict) else type(obj)})")
    cfg = base or HouseholdCdapConfig()
    if joint.get('interaction'):
        cfg.interaction.clear()
        cfg.interaction.update({tuple(k): float(v)
                                for k, v in joint['interaction'].items()})
    for _attr in ('interaction3', 'interaction4', 'interaction5'):
        _tbl = joint.get(_attr)
        if _tbl:
            getattr(cfg, _attr).clear()
            getattr(cfg, _attr).update({tuple(k): float(v)
                                        for k, v in _tbl.items()})
    if joint.get('size_models'):
        cfg.per_size_tables = {
            int(s): {'interaction': {tuple(k): float(v) for k, v in (sm.get('interaction') or {}).items()},
                     'interaction3': {tuple(k): float(v) for k, v in (sm.get('interaction3') or {}).items()},
                     'interaction4': {tuple(k): float(v) for k, v in (sm.get('interaction4') or {}).items()},
                     'interaction5': {tuple(k): float(v) for k, v in (sm.get('interaction5') or {}).items()}}
            for s, sm in joint['size_models'].items()}
    if joint.get('dap_utility'):
        try:
            cfg.dap_utility.update(joint['dap_utility'])
        except Exception:
            pass
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
