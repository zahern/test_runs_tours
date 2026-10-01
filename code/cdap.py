"""
cdap.py  —  Coordinated Daily Activity Pattern
================================================
Predicts whether each person travels on a typical day (M/N/H).

  H  =  Home all day (no travel)
  N  =  Non-mandatory travel only
  M  =  Mandatory travel (work or school primary purpose)

QTS observed CDAP:
  ANYSTOPS=False                              → H
  ANYSTOPS=True + MAINACT in MANDATORY_ACTS  → M
  ANYSTOPS=True + other                      → N

Models
------
1. Larch MNL  — 3-alternative multinomial logit (H/N/M), used for prediction
2. Larch MXL  — Mixed logit with Normal random coefficients on key variables
                (ptype_FullTimeWorker, age_norm, has_vehicle)
                Reports taste heterogeneity across the population.

Person profiles (new — replaces ptype dummies that were all 0)
--------------------------------------------------------------
Previously: ptype derived from hrsp/typp/lfsp (synth pop integer codes)
            but HHTS uses WORKSTATUS/STUDYING string categories → all zeros

Now: build_person_profiles() reads QTS string categories directly and creates:
  ptype_*        : 7 person type dummies from WORKSTATUS + STUDYING strings
  age_*          : child/teen/working/older/senior age band dummies
  is_female      : from SEX column
  has_licence    : from CARLICENCE boolean
  hh_child/spouse: household role from RELATIONSHIP
  wfh            : works from home (WFHDAY)
  income_midplus : earns >= $800/wk (from INCOME string)
  has_vehicle    : vehrd > 0
  mandatory_role : worker or student (has daily obligation)
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
import traceback
from typing import Optional
from scipy.stats import norm

from larch_regularization import elasticnet_objective

# Larch 6.0 runtime patches: ridge-aware covariance (p_values from the same
# penalised objective that was fitted) + logsums(datatree=) shim. Idempotent.
import larch_patches as _larch_patches
_larch_patches.apply()

# UTF-8 console guard: CDAP prints box-drawing progress art; on a C-locale
# (or cp1252) stdout one fancy print raises UnicodeEncodeError and aborts an
# otherwise-healthy estimation (observed locally on Windows). Same pattern
# as ga_hypothesis_testing._ensure_utf8_console. No-op on failure.
try:
    import sys as _sys
    for _s in (_sys.stdout, _sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
except Exception:
    pass

logger = logging.getLogger(__name__)

CDAP_LABELS = {0: 'H', 1: 'N', 2: 'M'}
CDAP_CODES  = {'H': 0, 'N': 1, 'M': 2}

# Coefficient magnitude cap. The old tight ±10 pin created at-bound coefficients
# under quasi-separation (a ptype/studying dummy perfectly predicting M/N/H),
# which made the unpenalized Hessian singular → covariance failed → all
# std-errs/p-values came back NaN. Elastic-net regularization
# (elasticnet_objective) is already embedded, so the pin isn't needed for
# identification; loosen it to a wide numerical guard (won't bind for real
# coefficients) and let regularization + the SE fallback (_fill_larch_inference)
# do the work. Set CDAP_COEF_CAP very large to effectively disable it.
_CDAP_COEF_CAP = float(os.environ.get('CDAP_COEF_CAP', '25'))

MANDATORY_ACTS = {
    'Full-time Work', 'Part-time Work', 'Work',
    'Primary School', 'Secondary School',
    'Full-time TAFE/Uni', 'Part-time TAFE/Uni',
    'Childcare/Kindergarten',
    'workFullTime', 'workPartTime', 'school', 'education',
}

INCOME_SCORE = {
    'Nil or negative income': 0,
    '$1 - $149 per week ($1 - $7,799 per year)': 1,
    '$150 - $299 per week ($7,800 - $15,599 per year)': 2,
    '$300 - $399 per week ($15,600 - $20,799 per year)': 3,
    '$400 - $499 per week ($20,800 - $25,999 per year)': 4,
    '$500 - $649 per week ($26,000 - $33,799 per year)': 5,
    '$650 - $799 per week ($33,800 - $41,599 per year)': 6,
    '$800 - $999 per week ($41,600 - $51,999 per year)': 7,
    '$1,000 - $1,249 per week ($52,000 - $64,999 per year)': 8,
    '$1,250 - $1,499 per week ($65,000 - $77,999 per year)': 9,
    '$1,500 - $1,749 per week ($78,000 - $90,999 per year)': 10,
    '$1,750 - $1,999 per week ($91,000 - $103,999 per year)': 11,
    '$2,000 - $2,999 per week ($104,000 - $155,999 per year)': 12,
    '$3,000 - $3,499 per week ($156,000 - $181,999 per year)': 13,
    '$3,500 or more per week ($182,000 or more per year)': 14,
}

MNL_FEATURES = [
    'ptype_FullTimeWorker', 'ptype_PartTimeWorker',
    'ptype_UniversityStudent', 'ptype_WorkingStudent',
    'ptype_SchoolChild', 'ptype_Retired',
    'ptype_Preschool',       # must be explicit — reference category is NonWorker,
                             # so Preschool needs its own coefficient to stay home.
    'age_child', 'age_teen', 'age_older', 'age_senior',
    'age_norm',
    'is_female',
    'has_licence', 'has_vehicle',
    'hh_child', 'hh_spouse', 'hh_alone', 'hh_size_large',
    'wfh',
    'income_midplus',
    # Home-zone accessibility, attached in prepare_cdap_data. crash_acc is
    # the time-decayed expected crash exposure reachable from home
    # (compute_accessibility) - embedding it HERE means safety shapes the
    # day-pattern decision itself, and therefore cascades into every
    # downstream stage a CDAP letter gates (activity generation, timing,
    # mode, destination). walk/transit accessibility enter alongside so the
    # crash coefficient is not proxying general urbanity.
    'crash_acc', 'walk_acc', 'transit_acc',
]

MXL_RANDOM = ['ptype_FullTimeWorker', 'age_norm', 'has_vehicle']


def _nonconstant_features(df: pd.DataFrame, cols: list) -> list:
    """Drop all-zero/all-one feature columns before estimation."""
    keep = []
    dropped = []
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').fillna(0)
        nunique = int(s.nunique(dropna=False))
        if nunique <= 1:
            dropped.append(col)
        else:
            keep.append(col)
    if dropped:
        logger.info(f"CDAP dropped non-varying features: {dropped}")
    return keep


def _extract_cdap_coefs(mnl, feat_cols: list) -> dict:
    """
    Robustly extract fitted parameter values from a Larch CDAP MNL model.

    Tries multiple Larch internal representations in order:
      1. model.pf (standard parameter frame — may fail if model is stale)
      2. model._final_pf (cached by _estimate after fitting)
      3. model.pnames + model.pvals (lower-level arrays)
      4. model._pvals + model._pnames (internal storage)
      5. model.get_value() per expected parameter name (brute-force)
    """
    expected = ['ASC_N', 'ASC_M'] + [f'b_N_{f}' for f in feat_cols] + [f'b_M_{f}' for f in feat_cols]
    coefs: dict = {}

    # Attempt 1: model.pf
    try:
        pf = mnl.pf
        for name in pf.index:
            v = float(pf.loc[name, 'value'])
            if not np.isnan(v):
                coefs[str(name)] = v
        if coefs:
            return coefs
    except Exception:
        pass

    # Attempt 2: cached _final_pf
    try:
        pf2 = mnl._final_pf
        for name in pf2.index:
            v = float(pf2.loc[name, 'value'])
            if not np.isnan(v):
                coefs[str(name)] = v
        if coefs:
            return coefs
    except Exception:
        pass

    # Attempt 3: pnames + pvals arrays
    try:
        for name, val in zip(mnl.pnames, mnl.pvals):
            if np.isfinite(val):
                coefs[str(name)] = float(val)
        if coefs:
            return coefs
    except Exception:
        pass

    # Attempt 4: internal _pvals / _pnames
    try:
        for name, val in zip(mnl._pnames, mnl._pvals):
            if np.isfinite(val):
                coefs[str(name)] = float(val)
        if coefs:
            return coefs
    except Exception:
        pass

    # Attempt 5: brute-force get_value per expected parameter name
    for pname in expected:
        try:
            v = float(mnl.get_value(pname))
            if np.isfinite(v):
                coefs[pname] = v
        except Exception:
            pass

    return coefs


def _fill_larch_inference(model) -> None:
    """
    Populate finite std_err / t_stat / p_value for Larch models.
    """
    pf = model.pf
    if 'std_err' not in pf.columns:
        pf['std_err'] = np.nan
    if 't_stat' not in pf.columns:
        pf['t_stat'] = np.nan
    if 'p_value' not in pf.columns:
        pf['p_value'] = np.nan

    try:
        needs_fallback = pf['std_err'].isna().all() or (pf['std_err'] <= 0).all()
    except Exception:
        needs_fallback = True

    if 'std_err_source' not in pf.columns:
        pf['std_err_source'] = 'hessian'

    # Ridge lift used by every fallback below — SAME penalty strength the model
    # was fitted with (GA_L1_L2_ALPHA), so a flat direction gets a large but
    # FINITE standard error instead of NaN/inf.
    try:
        from larch_regularization import DEFAULT_L1_L2_ALPHA as _ridge_alpha
    except Exception:
        _ridge_alpha = 1e-3
    _ridge_alpha = float(os.environ.get('GA_L1_L2_ALPHA', _ridge_alpha))

    # ── Fallback 1: OPG (outer product of casewise gradients) ────────────────
    if needs_fallback:
        try:
            scores = np.asarray(model.d_loglike_casewise(), dtype=float)
            if scores.ndim == 2 and scores.shape[1] > 0:
                bread = scores.T @ scores
                bread = bread + _ridge_alpha * np.eye(bread.shape[0])
                vcov = np.linalg.pinv(bread)
                se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
                pf['std_err_opg'] = pd.Series(se, index=pf.index[:len(se)]).reindex(pf.index).values
        except Exception as e:
            logger.debug(f"CDAP OPG inference failed: {e}")

    if 'std_err_opg' in pf.columns:
        bad = pf['std_err'].isna() | (pf['std_err'] <= 0)
        pf.loc[bad, 'std_err'] = pf.loc[bad, 'std_err_opg']
        pf.loc[bad & pf['std_err'].notna(), 'std_err_source'] = 'opg'

    # ── Fallback 2: finite-difference DIAGONAL Hessian (real curvature) ──────
    # A meaningful SE per free param from the loglike curvature — unlike the
    # |value|×0.35 proxy, this actually discriminates identified from
    # unidentified terms (a separated / at-bound param has ~0 curvature → a huge
    # SE → correctly insignificant), which is what the significance penalty needs.
    bad = pf['std_err'].isna() | (pf['std_err'] <= 0)
    if bad.any():
        try:
            names = list(pf.index[bad])
            ll0 = float(getattr(model, '_cached_ll', model.loglike()))
            for nm in names:
                try:
                    v0 = float(pf.loc[nm, 'value'])
                    if bool(pf.loc[nm].get('holdfast', False)):
                        continue
                    h = max(1e-3, abs(v0) * 1e-3)
                    model.set_value(nm, v0 + h); lp = float(model.loglike())
                    model.set_value(nm, v0 - h); lm = float(model.loglike())
                    model.set_value(nm, v0)
                    curv = -(lp - 2.0 * ll0 + lm) / (h * h)   # −d²LL/dθ² = info
                    # ridge lift: flat direction → curv≈0 → SE≈1/√α
                    # (large but FINITE → correctly insignificant, never NaN)
                    curv = curv + _ridge_alpha
                    if np.isfinite(curv) and curv > 0:
                        pf.loc[nm, 'std_err'] = float(1.0 / np.sqrt(curv))
                        pf.loc[nm, 'std_err_source'] = 'fd_hessian'
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"CDAP finite-difference Hessian failed: {e}")

    # ── Fallback 3: last-resort proxy (flagged so it can be down-weighted) ───
    bad = pf['std_err'].isna() | (pf['std_err'] <= 0)
    if bad.any():
        proxy = np.maximum(np.abs(pd.to_numeric(pf.loc[bad, 'value'], errors='coerce').fillna(0.0)) * 0.35, 0.15)
        pf.loc[bad, 'std_err'] = proxy.values
        pf.loc[bad, 'std_err_source'] = 'proxy'

    with np.errstate(divide='ignore', invalid='ignore'):
        pf['t_stat'] = pd.to_numeric(pf['value'], errors='coerce') / pd.to_numeric(pf['std_err'], errors='coerce').replace(0, np.nan)
    pf['p_value'] = 2.0 * norm.sf(np.abs(pd.to_numeric(pf['t_stat'], errors='coerce')))

    # PERSIST: model.pf returns a COPY in this Larch version, so writes above do
    # NOT stick to the model. Stash the enriched frame so _print_larch_table /
    # _save_cdap_html / the summary read the computed SEs instead of raw NaNs.
    try:
        model._inference_pf = pf
    except Exception:
        pass
    return pf


def build_person_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rich person profiles from QTS string categories.
    Works for both HHTS estimation (string cols) and synth pop prediction
    (integer codes).
    """
    d = df.copy()

    def _derive_ptype(row):
        studying_raw = row.get('STUDYING', '')
        studying = str(studying_raw or '').strip().lower()
        mainact = str(row.get('MAINACT', '') or '').strip().lower()
        ws = str(row.get('WORKSTATUS', '') or '').strip().lower()
        try:
            workstat_num = int(float(row.get('WORKSTAT', 0) or 0))
        except Exception:
            workstat_num = 0
        is_worker = (ws in ('workfulltime', 'workparttime')) or (workstat_num in (1, 2))
        try:
            studying_num = int(float(studying_raw))
        except Exception:
            studying_num = None

        # Numeric HHTS study codes are common in the survey extract:
        # 1=childcare, 2=school, 3=tertiary. The previous logic only matched
        # string labels, leaving university/working-student dummies all zero.
        if studying_num == 1:
            # Age guard: AGEGROUP 3+ (10-14 yr) with STUDYING=1 is a data quality
            # issue in QTS HHTS — 10-14yr olds incorrectly coded as childcare.
            # Reclassify as SchoolChild when AGEGROUP >= 3.
            try:
                _ag_chk = int(row.get('AGEGROUP', row.get('age5p', 1)) or 1)
            except Exception:
                _ag_chk = 1
            return 'Preschool' if _ag_chk <= 2 else 'SchoolChild'
        if studying_num == 2:
            return 'SchoolChild'
        if studying_num in (3, 4):
            return 'WorkingStudent' if is_worker else 'UniversityStudent'

        if studying == 'childcare':
            return 'Preschool'
        if studying in ('primary', 'secondary'):
            return 'SchoolChild'
        if studying in ('tertiaryfulltime', 'tertiaryparttime'):
            return 'WorkingStudent' if is_worker else 'UniversityStudent'

        if mainact in ('full-time tafe/uni', 'part-time tafe/uni', 'other education'):
            return 'WorkingStudent' if is_worker else 'UniversityStudent'

        if ws == 'workfulltime':   return 'FullTimeWorker'
        if ws == 'workparttime':   return 'PartTimeWorker'
        if ws == 'retired':        return 'Retired'

        # integer fallbacks for synth pop
        try:
            wstat = int(float(row.get('WORKSTAT', 0) or 0))
        except Exception:
            wstat = 0
        try:
            stud = int(float(row.get('STUDYING_code', studying_num or 0) or 0))
        except Exception:
            stud = 0
        if stud in (3, 4):
            return 'WorkingStudent' if wstat in (1, 2) else 'UniversityStudent'
        if wstat == 1: return 'FullTimeWorker'
        if wstat == 2: return 'PartTimeWorker'
        if stud in (1, 2): return 'SchoolChild'

        ag = int(row.get('AGEGROUP', row.get('age5p', 6)) or 6)
        if ag <= 2: return 'Preschool'
        if ag <= 4: return 'SchoolChild'
        if ag >= 14: return 'Retired'
        return 'NonWorker'

    d['ptype'] = d.apply(_derive_ptype, axis=1)

    PTYPES = ['FullTimeWorker', 'PartTimeWorker', 'UniversityStudent',
              'WorkingStudent', 'SchoolChild', 'Retired', 'Preschool']
    for pt in PTYPES:
        d[f'ptype_{pt}'] = (d['ptype'] == pt).astype(np.int8)

    # WORKSTAT (0=not working, 1=fulltime, 2=parttime) and STUDYING (0=none,
    # 1=primary, 2=secondary, 3=tertiary PT, 4=tertiary FT) are nominal codes
    # with no meaningful linear scale — break them into per-category binary
    # indicators (like ptype_*) so the GA/estimators treat them as binaries
    # rather than as categorical/ordinal codes. Built here so estimation
    # (HHTS) and prediction (synth pop) derive identical columns.
    if 'WORKSTAT' in d.columns:
        ws_num = pd.to_numeric(d['WORKSTAT'], errors='coerce').fillna(0).astype(int)
        d['workstat_fulltime']   = (ws_num == 1).astype(np.int8)
        d['workstat_parttime']   = (ws_num == 2).astype(np.int8)
        d['workstat_notworking'] = (ws_num == 0).astype(np.int8)

    if 'STUDYING' in d.columns:
        stud_num = pd.to_numeric(d['STUDYING'], errors='coerce').fillna(0).astype(int)
        d['studying_none']        = (stud_num == 0).astype(np.int8)
        d['studying_primary']     = (stud_num == 1).astype(np.int8)
        d['studying_secondary']   = (stud_num == 2).astype(np.int8)
        d['studying_tertiary_pt'] = (stud_num == 3).astype(np.int8)
        d['studying_tertiary_ft'] = (stud_num == 4).astype(np.int8)

    ag = pd.to_numeric(
        d.get('AGEGROUP', d.get('age5p', pd.Series(6, index=d.index))),
        errors='coerce').fillna(6).astype(int)
    d['age_child']   = (ag <= 3).astype(np.int8)
    d['age_teen']    = ag.isin([4, 5]).astype(np.int8)
    d['age_working'] = ag.between(6, 13).astype(np.int8)
    d['age_older']   = ag.isin([14, 15, 16]).astype(np.int8)
    d['age_senior']  = (ag >= 17).astype(np.int8)
    d['age_norm']    = (ag - 1) / 20.0

    sex = d.get('SEX', d.get('sexp', pd.Series('male', index=d.index)))
    d['is_female'] = sex.astype(str).str.lower().isin(['female', '2']).astype(np.int8)

    if 'CARLICENCE' in d.columns:
        d['has_licence'] = pd.to_numeric(
            d['CARLICENCE'].map({True:1, False:0, 1:1, 0:0, '1':1, '0':0}),
            errors='coerce').fillna(0).astype(np.int8)
    else:
        d['has_licence'] = 0

    rel = d.get('RELATIONSHIP', pd.Series('', index=d.index)).astype(str).str.lower()
    d['hh_child']  = rel.isin(['child', 'son', 'daughter']).astype(np.int8)
    d['hh_spouse'] = rel.isin(['spouse', 'partner', 'defacto']).astype(np.int8)
    d['hh_parent'] = rel.isin(['parent']).astype(np.int8)

    if 'wfh' in d.columns:
        # already assigned upstream (synth pop via wfh_telecommute.predict_wfh)
        d['wfh'] = (pd.to_numeric(d['wfh'], errors='coerce').fillna(0) > 0).astype(np.int8)
    elif 'WFHDAY' in d.columns:
        # bool / 0-1 int / 'Y'/'N' — coerce robustly (don't use .map({True:1,1:1}):
        # Python folds True and 1 into one dict key and pandas fails to match int64)
        _w = d['WFHDAY']
        if _w.dtype == bool:
            d['wfh'] = _w.astype(np.int8)
        else:
            _wn = pd.to_numeric(_w, errors='coerce')
            if _wn.notna().any():
                d['wfh'] = (_wn.fillna(0) > 0).astype(np.int8)
            else:
                d['wfh'] = _w.astype(str).str.strip().str.lower().isin(
                    ['1', 'y', 'yes', 'true', 't']).astype(np.int8)
    else:
        d['wfh'] = 0

    if 'INCOME' in d.columns:
        inc_score = d['INCOME'].map(INCOME_SCORE).fillna(-1)
        d['income_midplus'] = (inc_score >= 7).astype(np.int8)
    else:
        incp = pd.to_numeric(d.get('incp', pd.Series(8, index=d.index)), errors='coerce').fillna(8)
        d['income_midplus'] = (incp >= 10).astype(np.int8)

    veh = pd.to_numeric(d.get('vehrd', d.get('VEHRD', pd.Series(1, index=d.index))), errors='coerce').fillna(1)
    d['has_vehicle'] = (veh > 0).astype(np.int8)

    nprd = pd.to_numeric(d.get('nprd', pd.Series(2, index=d.index)), errors='coerce').fillna(2)
    d['hh_size_large'] = (nprd >= 4).astype(np.int8)
    d['hh_alone']      = (nprd == 1).astype(np.int8)

    d['mandatory_role'] = d['ptype'].isin([
        'FullTimeWorker', 'PartTimeWorker', 'SchoolChild',
        'UniversityStudent', 'WorkingStudent']).astype(np.int8)

    if 'PERSWGT23' in d.columns:
        d['weight'] = pd.to_numeric(d['PERSWGT23'], errors='coerce').fillna(1.0)
        d['weight'] = d['weight'] / d['weight'].mean()
    elif 'weight' not in d.columns:
        d['weight'] = 1.0

    return d


def prepare_cdap_data(persons: pd.DataFrame) -> pd.DataFrame:
    df = persons.copy()
    if 'person_id' not in df.columns and 'PERSID' in df.columns:
        df = df.rename(columns={'PERSID': 'person_id'})

    anystops = df.get('ANYSTOPS', pd.Series(True, index=df.index))
    mainact  = df.get('MAINACT',  pd.Series('', index=df.index)).fillna('')
    is_mandatory = mainact.isin(MANDATORY_ACTS)

    df['cdap_obs'] = 0
    df.loc[anystops & ~is_mandatory, 'cdap_obs'] = 1
    df.loc[anystops &  is_mandatory, 'cdap_obs'] = 2

    df = build_person_profiles(df)

    # Attach home-zone accessibility so safety (and every other GA-selected
    # *_acc measure: asim logsums, power acc, fitted logsums) enters the CDAP
    # utility. Previously a hardcoded 3-column list, which silently dropped any
    # other GA-selected accessibility feature at refit even though the search
    # frame carried it. Non-fatal: _nonconstant_features silently drops the
    # columns when the accessibility CSV hasn't been built yet.
    try:
        _acc_csv = os.path.join('saved_data', 'accessibility.csv')
        if os.path.exists(_acc_csv):
            _acc = pd.read_csv(_acc_csv).set_index('zone_id')
            _acc_wanted = [c for c in _acc.columns
                           if str(c).endswith('_acc') and c not in df.columns]
            if _acc_wanted:
                _home = pd.to_numeric(
                    df.get('OTAZ', df.get('HOMETAZ')), errors='coerce'
                ).fillna(-1).astype(int)
                for c in _acc_wanted:
                    df[c] = _home.map(_acc[c]).fillna(float(_acc[c].median()))
                print(f"  CDAP: attached home-zone accessibility "
                      f"({[c for c in _acc_wanted if c in df.columns]})")
    except Exception as _acc_e:
        logger.warning(f"CDAP accessibility attach skipped: {_acc_e}")

    dist = df['cdap_obs'].value_counts().sort_index()
    print(f"  CDAP observed: H={dist.get(0,0)}  N={dist.get(1,0)}  M={dist.get(2,0)}")
    print(f"  Person type dist: {df['ptype'].value_counts().to_dict()}")
    return df


def _inf_pf(model):
    """Enriched parameter frame with computed std_err/t_stat/p_value if
    _fill_larch_inference stashed one (model.pf itself returns a fresh copy that
    lacks those columns), else the raw model.pf."""
    pf = getattr(model, '_inference_pf', None)
    return pf if pf is not None else model.pf


def _rho_sq_null_safe(model, ll=None) -> float:
    """rho-squared via larch_patches' compatibility shim, cause logged once.

    The underlying defect: larch 6.0.44 has no ``rho_sq_null()`` method at all
    (confirmed -- ``hasattr(larch.Model, 'rho_sq_null')`` is False and Model has
    no 'rho'-named members), so every call site raised AttributeError and
    printed rho2=nan for every model. Visible in runs/exampville_ab_x2 and
    runs/mtc_ab_x2, where all four CDAP arms report rho2=nan while the baseline
    arm reported None -- which is why it read as a result rather than a crash.
    larch_patches.apply() now installs the method (computing 1 - LL/LL_null
    from ``loglike_null()``), so this delegates to that single implementation.
    The warning stays so any *other* failure, e.g. loglike_null() raising, is
    still visible instead of silently degrading to nan.
    """
    global _RHO_SQ_NULL_WARNED
    try:
        return float(model.rho_sq_null())
    except Exception as _e:
        if not _RHO_SQ_NULL_WARNED:
            _RHO_SQ_NULL_WARNED = True
            print(f"  WARNING: rho_sq_null() failed ({type(_e).__name__}: {_e}); "
                  f"reporting nan. Further occurrences suppressed.\n"
                  f"{traceback.format_exc()}", flush=True)
        return float('nan')


_RHO_SQ_NULL_WARNED = False


def _print_larch_table(model, label: str):
    pf  = _inf_pf(model)
    ll  = getattr(model, '_cached_ll', float('nan'))
    rho2 = _rho_sq_null_safe(model)
    print(f"\n  ┌─ {label}  LL={ll:.1f}  ρ²={rho2:.3f}  n_params={len(pf)} ──────")
    print(f"  {'Parameter':<38} {'Value':>9}  {'Std Err':>8}  {'t-stat':>7}  Sig")
    print(f"  {'─'*74}")
    for name in pf.index:
        row = pf.loc[name]
        v = float(row.get('value',   float('nan')))
        s = float(row.get('std_err', row.get('std_err_opg', float('nan'))))
        t = float(row.get('t_stat',  float('nan')))
        sig = ('***' if abs(t)>3.3 else '**' if abs(t)>2.6 else '*' if abs(t)>1.96 else '')
        print(f"  {str(name):<38} {v:>9.4f}  {s:>8.4f}  {t:>7.2f}  {sig}")
    print(f"  └{'─'*74}")


def _save_cdap_html(mnl, mxl, path):
    if path is None: return
    def pf_html(model, title):
        if model is None: return f'<p><em>{title} — not available</em></p>'
        pf = _inf_pf(model)
        ll = getattr(model, '_cached_ll', float('nan'))
        rho2 = _rho_sq_null_safe(model)
        rows = ''
        for name in pf.index:
            row = pf.loc[name]
            v = float(row.get('value', float('nan')))
            s = float(row.get('std_err', row.get('std_err_opg', float('nan'))))
            t = float(row.get('t_stat', float('nan')))
            sig = ('★★★' if abs(t)>3.3 else '★★' if abs(t)>2.6 else '★' if abs(t)>1.96 else '')
            c = '#155724' if abs(t)>1.96 else '#721c24'
            rows += (f'<tr><td>{name}</td><td style="text-align:right">{v:.4f}</td>'
                     f'<td style="text-align:right">{s:.4f}</td>'
                     f'<td style="text-align:right;color:{c}">{t:.2f}</td>'
                     f'<td>{sig}</td></tr>\n')
        return (f'<h2>{title}</h2><p class="stats">LL={ll:.1f} | ρ²={rho2:.3f}</p>'
                f'<table><thead><tr><th>Parameter</th><th>Value</th><th>Std Err</th>'
                f'<th>t-stat</th><th>★</th></tr></thead><tbody>{rows}</tbody></table>')

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>CDAP Results</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;max-width:1200px}}
h1{{border-bottom:2px solid #1565c0;padding-bottom:8px}}h2{{color:#1565c0}}
.stats{{background:#e3f2fd;padding:6px 12px;border-radius:4px;margin-bottom:10px}}
table{{border-collapse:collapse;width:100%;margin-bottom:20px}}
th{{background:#1565c0;color:white;padding:8px 12px;text-align:left}}
td{{padding:5px 12px;border-bottom:1px solid #eee;font-size:.9em}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:32px}}</style></head>
<body><h1>CDAP — Coordinated Daily Activity Pattern</h1>
<p><b>H</b>=Home | <b>N</b>=Non-mandatory | <b>M</b>=Mandatory (work/school)</p>
<p>Reference alternative: H. Prefixes: <code>b_N_</code>=H→N effect, <code>b_M_</code>=H→M effect.
MXL random coefficients (s_*) show taste heterogeneity — significant std dev means people vary in that preference.</p>
<div class="grid"><div>{pf_html(mnl, "MNL — Base Model")}</div>
<div>{pf_html(mxl, "Mixed Logit — Random Coefficients")}</div></div>
</body></html>"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    open(path, 'w', encoding='utf-8').write(html)
    print(f"  CDAP HTML → {path}")


def estimate_cdap(persons: pd.DataFrame,
                  save_path: str = 'saved_data/cdap_model.pkl',
                  feature_override: list = None) -> dict:
    """
    Estimate CDAP model on QTS HHTS persons using Larch MNL + MXL.

    Uses compute_engine='jax' — Larch compiles the MNL log-likelihood
    and gradient via XLA/JAX for GPU acceleration.  The CDAP model is
    small (23k persons × 3 alts × 20 features ≈ 5.5 MB) so JAX memory
    footprint is negligible.  With XLA_PYTHON_CLIENT_PREALLOCATE=false
    (set in pbs_pipeline_mode.pbs for bandit+GPU), JAX allocates GPU
    memory on-demand rather than pre-allocating.

    feature_override: when given (e.g. the bandit-SA/GA feature-search
    winner for stage 0), replaces the hardcoded MNL_FEATURES list. Without
    this, the GA search that already runs for CDAP had no way to affect
    the production model — its result was written to a diagnostic JSON only.
    """
    print("\n[1/2] Estimating from HHTS...")
    data = prepare_cdap_data(persons)
    _candidates = feature_override if feature_override else MNL_FEATURES
    feat_cols = _nonconstant_features(data, [c for c in _candidates if c in data.columns])
    print(f"  Features ({len(feat_cols)}){' [GA-selected]' if feature_override else ''}: {feat_cols}")

    import larch as lx

    case_ids = data['person_id'].astype(str).tolist()
    alt_ids  = [0, 1, 2]

    # build idca dataset (chosen flag only — no skim attributes)
    ca_rows = []
    for _, row in data.iterrows():
        pid = str(row['person_id']); obs = int(row['cdap_obs'])
        for alt in alt_ids:
            ca_rows.append({'person_id': pid, '_altid_': alt,
                            'chosen': int(obs == alt)})
    ca_df = pd.DataFrame(ca_rows).set_index(['person_id', '_altid_'])
    ds_ca = lx.Dataset.construct.from_idca(ca_df)

    # build idco dataset (all features are case-level)
    co_df = data.set_index(data['person_id'].astype(str))[feat_cols + ['weight']].copy()
    co_df.index.name = 'person_id'
    ds_co = lx.Dataset.construct.from_idco(co_df)

    dt = lx.DataTree(base=ds_ca, person=ds_co,
                     relationships=('base.person_id @ person.person_id',))

    # ── MNL (JAX engine — GPU-accelerated) ──────────────────────────────────
    print("  Estimating MNL (JAX engine)...")
    mnl = lx.Model(dt, compute_engine='jax')
    mnl.choice_ca_var = 'chosen'
    mnl.utility_co[1] = lx.P('ASC_N')
    mnl.utility_co[2] = lx.P('ASC_M')
    for feat in feat_cols:
        mnl.utility_co[1] += lx.P(f'b_N_{feat}') * lx.X(f'person.{feat}')
        mnl.utility_co[2] += lx.P(f'b_M_{feat}') * lx.X(f'person.{feat}')
    try:
        mnl.weight_co_var = 'weight'
    except (TypeError, AttributeError):
        pass
    mnl.set_cap(_CDAP_COEF_CAP)
    # Elastic-net (L1+L2) penalty on the objective, mirroring
    # metacountregressor's penalized-likelihood estimation - see
    # larch_regularization.py. The covariance is computed INSIDE the context so
    # the Hessian includes the SAME penalty that was fitted: the ≈α eigenvalue
    # lift keeps collinear/flat directions invertible → std_err/p_value are
    # always finite (a genuinely flat direction gets a large-but-finite SE and
    # correctly shows as insignificant). _cached_ll still stores the pure LL
    # for reporting.
    with elasticnet_objective(mnl):
        r = mnl.maximize_loglike(options={'ftol': 1e-8}, quiet=True)
        try:
            mnl.calculate_parameter_covariance()
        except Exception as e:
            logger.warning(f"MNL covariance failed: {e}")
    mnl._cached_ll = float(r.loglike)
    _fill_larch_inference(mnl)
    _print_larch_table(mnl, "CDAP MNL")

    # ── MXL / heterogeneity reporting (bounded sample) ─────────────────────
    # The full-sample 200-draw JAX MXL OOM-killed the 250GB prediction node
    # (job 23970303: cgroup SIGKILL at ~259GB — uncatchable, so the except
    # below never fires). The MXL is report-only (predict_cdap uses the MNL).
    # CDAP_MXL=0  skips it entirely (MNL only — cheapest, shard default).
    # CDAP_MXL=mh estimates taste heterogeneity via the JAX-free RWMH driver
    #   (mh_choice_estimator.estimate_mnl_mh) on a weight-aware subsample:
    #   minutes + single-digit MBs instead of 10s of GBs. Posterior SDs are
    #   parameter uncertainty (not Larch s_* random-coefficient SDs) — the
    #   report is labelled accordingly.
    # otherwise   JAX MSL mixed logit; CDAP_MXL_MAX_N / CDAP_MXL_DRAWS bound it.
    mxl = None
    mxl_ok = False
    _mxl_mode = os.environ.get('CDAP_MXL', '1').strip().lower()
    if _mxl_mode == '0':
        print("\n  CDAP_MXL=0 — skipping Mixed Logit (MNL only)")
    elif _mxl_mode == 'mh':
        _mxl_max_n = int(os.environ.get('CDAP_MXL_MAX_N', '5000'))
        try:
            if len(data) > _mxl_max_n:
                _w = pd.to_numeric(data.get('weight', 1.0),
                                   errors='coerce').fillna(0.0).clip(lower=0.0)
                if float(_w.sum()) > 0:
                    mxl_sub = data.sample(n=_mxl_max_n, weights=_w,
                                          random_state=42)
                else:
                    mxl_sub = data.sample(n=_mxl_max_n, random_state=42)
                print(f"\n  CDAP heterogeneity via MH posterior (RWMH, "
                      f"weight-aware subsample n={len(mxl_sub)}/{len(data)})...")
            else:
                mxl_sub = data
            _mh_res = _estimate_cdap_mh(mxl_sub, feat_cols, mnl)
            if _mh_res is None:
                raise RuntimeError("MH heterogeneity returned no result")
            mxl = _mh_res['model']
            mxl_ok = True
        except Exception as e:
            logger.warning(f"MH heterogeneity failed: {e} — MNL only")
            mxl = None
    else:
        _mxl_draws = int(os.environ.get('CDAP_MXL_DRAWS', '100'))
        _mxl_max_n = int(os.environ.get('CDAP_MXL_MAX_N', '5000'))
        try:
            if len(data) > _mxl_max_n:
                mxl_sub = data.sample(n=_mxl_max_n, random_state=42)
                print(f"\n  Estimating Mixed Logit ({_mxl_draws} draws, JAX, "
                      f"subsample n={len(mxl_sub)}/{len(data)})...")
                sub_ids = mxl_sub['person_id'].astype(str).tolist()
                ds_ca_s = lx.Dataset.construct.from_idca(ca_df.loc[sub_ids])
                ds_co_s = lx.Dataset.construct.from_idco(co_df.loc[sub_ids])
                dt_s = lx.DataTree(
                    base=ds_ca_s, person=ds_co_s,
                    relationships=('base.person_id @ person.person_id',))
                mxl = lx.Model(dt_s, compute_engine='jax')
                mxl.choice_ca_var = 'chosen'
                mxl.utility_co[1] = lx.P('ASC_N')
                mxl.utility_co[2] = lx.P('ASC_M')
                for feat in feat_cols:
                    mxl.utility_co[1] += lx.P(f'b_N_{feat}') * lx.X(f'person.{feat}')
                    mxl.utility_co[2] += lx.P(f'b_M_{feat}') * lx.X(f'person.{feat}')
                mxl.set_cap(_CDAP_COEF_CAP)
                # warm-start at the full-sample MNL estimates
                for _p in mnl.pf.index:
                    try:
                        mxl.set_value(_p, float(mnl.pf.loc[_p, 'value']))
                    except Exception:
                        pass
            else:
                print(f"\n  Estimating Mixed Logit ({_mxl_draws} draws, JAX)...")
                mxl = mnl.copy()
            for feat in MXL_RANDOM:
                for prefix in ['b_N_', 'b_M_']:
                    pname = f'{prefix}{feat}'
                    if pname in mxl.pf.index:
                        mxl.mixtures.append(
                            lx.mixtures.Normal(pname, f's_{prefix[2:-1]}_{feat}'))
            mxl.n_draws = _mxl_draws
            mxl.seed    = 42
            try:
                mxl.weight_co_var = 'weight'
            except (TypeError, AttributeError):
                pass
            with elasticnet_objective(mxl):
                r2 = mxl.maximize_loglike(stderr=True, options={'ftol': 1e-8}, quiet=True)
                try:
                    mxl.calculate_parameter_covariance()
                except Exception:
                    pass
            mxl._cached_ll = float(r2.loglike)
            _fill_larch_inference(mxl)
            _print_larch_table(mxl, "CDAP Mixed Logit")
            mxl_ok = True
        except Exception as e:
            logger.warning(f"MXL failed: {e} — MNL only")
            mxl = None

    result = {
        'mnl':      mnl,
        'mxl':      mxl if mxl_ok else None,
        'sklearn':  None,
        'features': feat_cols,
        'll':       mnl._cached_ll,
        'rho2':     float('nan'),
    }
    _save_cdap_html(mnl, mxl if mxl_ok else None,
                    save_path.replace('.pkl', '.html'))
    from paper_model_forms import write_stage_paper_artifacts
    write_stage_paper_artifacts('cdap', os.path.dirname(save_path) or 'saved_data')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)
    try:
        import dill
        dill.dump(result, open(save_path, 'wb'))
        print(f"  Saved (dill) → {save_path}")
    except Exception:
        pickle.dump({'features': feat_cols,
                     'mnl_pf': mnl.pf.copy(),
                     'mxl_pf': mxl.pf.copy() if mxl_ok and mxl else None},
                    open(save_path, 'wb'))
        print(f"  Saved slim → {save_path}")
    return result


def _estimate_mxl_subsample(sub: pd.DataFrame,
                              feat_cols: list,
                              lx) -> Optional[dict]:
    """
    Estimate a Larch MXL on a small subsample for heterogeneity reporting.
    Uses idco-only DataTree (no alternatives expansion) — much smaller memory.
    Returns a summary dict with coef/se/t/ll, or None on failure.
    """
    alt_ids  = [0, 1, 2]
    case_ids = sub['person_id'].astype(str).tolist()

    # minimal idca: just chosen flag, no skim data
    ca_idx = pd.MultiIndex.from_product([case_ids, alt_ids],
                                        names=['person_id', '_altid_'])
    ca_df  = pd.DataFrame({'chosen': 0}, index=ca_idx)
    for _, row in sub.iterrows():
        ca_df.loc[(str(row['person_id']), int(row['cdap_obs'])), 'chosen'] = 1
    ds_ca = lx.Dataset.construct.from_idca(ca_df)

    co_df = sub.set_index(sub['person_id'].astype(str))[feat_cols + ['weight']].copy()
    co_df.index.name = 'person_id'
    ds_co = lx.Dataset.construct.from_idco(co_df)

    dt = lx.DataTree(base=ds_ca, person=ds_co,
                     relationships=('base.person_id @ person.person_id',))

    # simple MNL first (starting point for MXL)
    mnl = lx.Model(dt, compute_engine='jax')
    mnl.choice_ca_var = 'chosen'
    mnl.utility_co[1] = lx.P('ASC_N')
    mnl.utility_co[2] = lx.P('ASC_M')
    for feat in feat_cols:
        mnl.utility_co[1] += lx.P(f'b_N_{feat}') * lx.X(f'person.{feat}')
        mnl.utility_co[2] += lx.P(f'b_M_{feat}') * lx.X(f'person.{feat}')
    mnl.set_cap(_CDAP_COEF_CAP)
    with elasticnet_objective(mnl):
        mnl.maximize_loglike(options={'ftol': 1e-7}, quiet=True)
        try:
            mnl.calculate_parameter_covariance()
        except Exception:
            pass
    _fill_larch_inference(mnl)

    # add Normal mixtures on the 3 key random variables
    mxl = mnl.copy()
    mxl.n_draws = 100   # fewer draws since this is just for reporting
    mxl.seed    = 42
    for feat in MXL_RANDOM:
        for prefix in ['b_N_', 'b_M_']:
            pname = f'{prefix}{feat}'
            if pname in mxl.pf.index:
                mxl.mixtures.append(
                    lx.mixtures.Normal(pname, f's_{prefix[2:-1]}_{feat}'))
    try:
        mxl.weight_co_var = 'weight'
    except Exception:
        pass

    with elasticnet_objective(mxl):
        r = mxl.maximize_loglike(stderr=True, options={'ftol': 1e-7}, quiet=True)
        try:
            mxl.calculate_parameter_covariance()
        except Exception:
            pass
    mxl._cached_ll = float(r.loglike)
    _fill_larch_inference(mxl)

    _print_larch_table(mxl, f"CDAP MXL (subsample n={len(sub)})")

    pf = _inf_pf(mxl)
    return {
        'coef':    {name: float(pf.loc[name, 'value'])   for name in pf.index},
        'std_err': {name: float(pf.loc[name, 'std_err'])  for name in pf.index},
        'tstat':   {name: float(pf.loc[name, 't_stat'])   for name in pf.index},
        'll':      mxl._cached_ll,
        'n_sub':   len(sub),
    }


def _estimate_cdap_mh(mxl_sub: pd.DataFrame,
                      feat_cols: list,
                      mnl=None) -> Optional[dict]:
    """Heterogeneity reporting via MH posterior (JAX-free, low-memory).

    Runs the fixed-coefficient CDAP MNL through the adaptive RWMH driver
    (mh_choice_estimator.estimate_mnl_mh) warm-started at the full-sample MNL.
    Design is built directly in numpy — no Larch DataTree, no JAX/XLA, no
    Hessian: ~N*3*d floats (single-digit MBs) plus O(N*K*d) per iteration.

    Posterior SDs quantify parameter uncertainty, NOT Larch s_* random-
    coefficient taste SDs — callers must label the report accordingly
    (done here via the table title and result metadata).

    Returns a dict with coef/std_err/tstat/ll/n_sub/accept_rate/ess plus a
    'model' adapter exposing .pf/.rho_sq_null()/._cached_ll so the existing
    HTML/pickle plumbing works unchanged, or None on failure.
    """
    from types import SimpleNamespace
    from mh_choice_estimator import (
        estimate_mnl_mh, loglik_mat, pad_choice_data)

    feats = [f for f in (feat_cols or []) if f in mxl_sub.columns]
    if not feats:
        return None
    names = (['ASC_N', 'ASC_M']
             + [f'b_N_{f}' for f in feats]
             + [f'b_M_{f}' for f in feats])
    idx = {n: i for i, n in enumerate(names)}
    Xf = mxl_sub[feats].apply(
        pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    N, d = Xf.shape[0], len(names)
    if N < 30:
        return None

    # Per-case design: alt 0 = reference (all zeros); alt 1/2 carry their
    # intercept plus case attributes under alt-specific coefficients.
    Xarr = np.zeros((N, 3, d))
    Xarr[:, 1, idx['ASC_N']] = 1.0
    Xarr[:, 2, idx['ASC_M']] = 1.0
    _bN = [idx[f'b_N_{f}'] for f in feats]
    _bM = [idx[f'b_M_{f}'] for f in feats]
    Xarr[:, 1, _bN] = Xf
    Xarr[:, 2, _bM] = Xf
    Xlist = [Xarr[n] for n in range(N)]
    yidx = np.clip(pd.to_numeric(
        mxl_sub['cdap_obs'], errors='coerce').fillna(0).astype(int).to_numpy(),
        0, 2)

    # Warm-start at the full-sample MNL (cap respected by construction).
    init = np.zeros(d)
    if mnl is not None:
        try:
            _pf = _inf_pf(mnl)
            for n, i in idx.items():
                if n in _pf.index:
                    init[i] = float(_pf.loc[n, 'value'])
        except Exception:
            pass

    _burn = int(os.environ.get('CDAP_MH_BURN', '1000'))
    _draw = int(os.environ.get('CDAP_MH_DRAWS', '2000'))
    _thin = int(os.environ.get('CDAP_MH_THIN', '5'))
    _seed = int(os.environ.get('CDAP_MH_SEED', '42'))
    _psd = float(os.environ.get('CDAP_MH_PRIOR_SD', '5.0'))
    print(f"  Estimating CDAP MH posterior (RWMH burn={_burn} draws={_draw} "
          f"thin={_thin}, n={N}, d={d}, prior_sd={_psd})...")
    fit = estimate_mnl_mh(Xlist, yidx, offsets=None, prior_sd=_psd, init=init,
                          n_burn=_burn, n_draw=_draw, thin=_thin, seed=_seed,
                          verbose=True)
    draws = np.asarray(fit['draws'])
    mu = np.asarray(fit['posterior_mean'], dtype=float)
    sd = draws.std(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = mu / np.where(sd > 0, sd, np.nan)
    Xp, yp, opp, maskp = pad_choice_data(Xlist, yidx, None)
    ll_mean = loglik_mat(mu, Xp, yp, opp, maskp)
    ess = np.asarray(fit['ess'], dtype=float)

    pf = pd.DataFrame({'value': mu, 'std_err': sd, 't_stat': t}, index=names)
    model = SimpleNamespace(
        pf=pf, _inference_pf=pf, _cached_ll=float(ll_mean),
        rho_sq_null=lambda: float('nan'),
        mh_accept_rate=float(fit['accept_rate']),
        mh_ess_min=float(np.nanmin(ess)) if ess.size else float('nan'),
        mh_ess_mean=float(np.nanmean(ess)) if ess.size else float('nan'),
        mh_label='CDAP MH posterior (RWMH fixed-coefficient; s_* not estimated)',
        n_sub=int(N),
    )
    _print_larch_table(model, "CDAP MH posterior (RWMH, report-only)")
    print(f"  MH accept={fit['accept_rate']:.3f} "
          f"ESS min={model.mh_ess_min:.0f} mean={model.mh_ess_mean:.0f}")
    return {
        'coef':      {n: float(mu[i]) for i, n in enumerate(names)},
        'std_err':   {n: float(sd[i]) for i, n in enumerate(names)},
        'tstat':     {n: float(t[i]) for i, n in enumerate(names)},
        'll':        float(ll_mean),
        'n_sub':     int(N),
        'accept_rate': float(fit['accept_rate']),
        'ess_min':   float(model.mh_ess_min),
        'ess_mean':  float(model.mh_ess_mean),
        'model':     model,
    }


def _save_cdap_html_sklearn(sklearn_mdl, mxl_summary, feat_cols, path):
    """Save HTML coefficient report from sklearn result + optional MXL."""
    if path is None:
        return
    try:
        coef = sklearn_mdl.coef_
        se   = sklearn_mdl.se_dict_
        ll   = sklearn_mdl.ll_
        rho2 = sklearn_mdl.rho2_

        rows = ''
        for fi, feat in enumerate(feat_cols):
            vals = [coef[ci, fi] for ci in range(3)]
            t_n  = vals[1] / se[1][fi] if se[1][fi] > 0 else float('nan')
            t_m  = vals[2] / se[2][fi] if se[2][fi] > 0 else float('nan')
            sig  = ('★★★' if (abs(t_n)>3.3 or abs(t_m)>3.3) else
                    '★★'  if (abs(t_n)>2.6 or abs(t_m)>2.6) else
                    '★'   if (abs(t_n)>1.96 or abs(t_m)>1.96) else '')
            cn   = '#155724' if sig else '#721c24'
            rows += (f'<tr><td>{feat}</td>'
                     f'<td style="text-align:right">{vals[0]:.4f}</td>'
                     f'<td style="text-align:right">{vals[1]:.4f}</td>'
                     f'<td style="text-align:right">{vals[2]:.4f}</td>'
                     f'<td style="text-align:right;color:{cn}">{t_n:.2f}</td>'
                     f'<td style="text-align:right;color:{cn}">{t_m:.2f}</td>'
                     f'<td>{sig}</td></tr>\n')

        mxl_section = ''
        if mxl_summary:
            mxl_rows = ''
            for name, v in mxl_summary['coef'].items():
                se_v = mxl_summary['std_err'].get(name, float('nan'))
                t_v  = mxl_summary['tstat'].get(name, float('nan'))
                sig  = ('★★★' if abs(t_v)>3.3 else '★★' if abs(t_v)>2.6
                        else '★' if abs(t_v)>1.96 else '')
                is_rand = str(name).startswith('s_')
                kind = '<em>std dev (random)</em>' if is_rand else 'fixed'
                mxl_rows += (f'<tr><td>{name}</td>'
                              f'<td style="text-align:right">{v:.4f}</td>'
                              f'<td style="text-align:right">{se_v:.4f}</td>'
                              f'<td style="text-align:right">{t_v:.2f}</td>'
                              f'<td>{sig}</td><td>{kind}</td></tr>\n')
            mxl_section = f'''
<h2>Mixed Logit — Taste Heterogeneity (subsample n={mxl_summary["n_sub"]})</h2>
<p class="stats">LL={mxl_summary["ll"]:.1f}
  | Random coefficients: {", ".join(MXL_RANDOM)}</p>
<p>Parameters prefixed <code>s_</code> are standard deviations of the Normal
distribution. A significant <code>s_</code> means people genuinely differ
in that preference — it is not just noise.</p>
<table>
<thead><tr><th>Parameter</th><th>Value</th><th>Std Err</th>
<th>t-stat</th><th>★</th><th>Type</th></tr></thead>
<tbody>{mxl_rows}</tbody></table>'''

        html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>CDAP Results</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;max-width:1000px}}
h1{{border-bottom:2px solid #1565c0;padding-bottom:8px}}
h2{{color:#1565c0;margin-top:28px}}
.stats{{background:#e3f2fd;padding:6px 12px;border-radius:4px;margin-bottom:10px}}
table{{border-collapse:collapse;width:100%;margin-bottom:20px}}
th{{background:#1565c0;color:white;padding:8px 12px;text-align:left}}
td{{padding:5px 12px;border-bottom:1px solid #eee;font-size:.9em}}
</style></head>
<body>
<h1>CDAP — Coordinated Daily Activity Pattern</h1>
<p><b>H</b> = Home all day &nbsp;|&nbsp;
   <b>N</b> = Non-mandatory travel &nbsp;|&nbsp;
   <b>M</b> = Mandatory (work/school)</p>

<h2>Multinomial Logit — Full Sample</h2>
<p class="stats">LL={ll:.1f} &nbsp;|&nbsp; ρ²={rho2:.3f}
  &nbsp;|&nbsp; Reference: H (Home)</p>
<table>
<thead><tr><th>Feature</th><th>H coef</th><th>N coef</th><th>M coef</th>
<th>t(N)</th><th>t(M)</th><th>★</th></tr></thead>
<tbody>{rows}</tbody></table>

{mxl_section}
<p style="color:#888;font-size:.85em">Generated by cdap.py — QTS ABM Pipeline</p>
</body></html>"""

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',
                    exist_ok=True)
        open(path, 'w', encoding='utf-8').write(html)
        print(f"  CDAP HTML → {path}")
    except Exception as e:
        logger.debug(f"CDAP HTML failed: {e}")


def predict_cdap(synth: pd.DataFrame, result: dict, seed: int = 42) -> pd.DataFrame:
    """
    Predict CDAP for synthetic population.
    Uses Larch MNL with numba engine (avoids JAX OOM).
    Falls back to sklearn if MNL not available.
    """
    df  = synth.copy()
    rng = np.random.default_rng(seed)
    df  = build_person_profiles(df)

    mnl         = result.get('mnl')
    sklearn_mdl = result.get('sklearn')
    feat_cols   = result.get('features', MNL_FEATURES)

    # Prediction-side accessibility attach (mirrors prepare_cdap_data): the
    # estimated model may carry crash_acc/walk_acc/transit_acc coefficients;
    # a synth population without those columns would silently predict with
    # them zero-filled (reindex(...).fillna(0) below), biasing everyone
    # toward the reference pattern.
    try:
        _acc_needed = [c for c in feat_cols
                       if c.endswith('_acc') and c not in df.columns]
        _acc_csv = os.path.join('saved_data', 'accessibility.csv')
        if _acc_needed and os.path.exists(_acc_csv):
            _acc = pd.read_csv(_acc_csv).set_index('zone_id')
            _home = pd.to_numeric(
                df.get('OTAZ', df.get('HOMETAZ')), errors='coerce'
            ).fillna(-1).astype(int)
            for c in _acc_needed:
                if c in _acc.columns:
                    df[c] = _home.map(_acc[c]).fillna(float(_acc[c].median()))
            print(f"  predict_cdap: attached home-zone accessibility {_acc_needed}")
    except Exception as _acc_e:
        logger.warning(f"predict_cdap accessibility attach skipped: {_acc_e}")

    if mnl is not None:
        try:
            import larch as lx
            pid_col  = 'person_id' if 'person_id' in df.columns else None

            # ── deduplicate: DataTree requires one row per case ─────────────
            if pid_col is not None:
                df_uniq = df.drop_duplicates(subset=[pid_col]).reset_index(drop=True)
            else:
                df_uniq = df.copy()
            case_ids = (df_uniq[pid_col].astype(str).tolist()
                        if pid_col is not None else df_uniq.index.astype(str).tolist())
            alt_ids  = [0, 1, 2]

            co_df = df_uniq.reindex(columns=feat_cols).fillna(0).copy()
            co_df.index = case_ids
            co_df.index.name = 'person_id'
            ds_co = lx.Dataset.construct.from_idco(co_df)

            ca_idx = pd.MultiIndex.from_product(
                [case_ids, alt_ids], names=['person_id', '_altid_'])
            ca_df  = pd.DataFrame({'chosen': 0, '_dummy': 0}, index=ca_idx)
            ds_ca  = lx.Dataset.construct.from_idca(ca_df)
            dt_pred = lx.DataTree(base=ds_ca, person=ds_co,
                                  relationships=('base.person_id @ person.person_id',))

            # Build a fresh model against dt_pred, transfer fitted parameters.
            # Some Larch versions don't support mnl.datatree = ... on a loaded model.
            fresh = lx.Model(dt_pred, compute_engine='jax')
            fresh.choice_ca_var = 'chosen'
            fresh.utility_co[1] = lx.P('ASC_N')
            fresh.utility_co[2] = lx.P('ASC_M')
            for _feat in feat_cols:
                fresh.utility_co[1] += lx.P(f'b_N_{_feat}') * lx.X(f'person.{_feat}')
                fresh.utility_co[2] += lx.P(f'b_M_{_feat}') * lx.X(f'person.{_feat}')

            # Transfer parameter values from the fitted mnl
            _coefs = _extract_cdap_coefs(mnl, feat_cols)
            for _pname, _pval in _coefs.items():
                try:
                    fresh.set_value(_pname, _pval)
                except Exception:
                    pass
            fresh.mangle()
            fresh.unmangle()
            probs = np.asarray(fresh.probability())   # (n_uniq, 3)
            probs = probs / probs.sum(axis=1, keepdims=True)

            # Some cached Larch pickles load without a usable parameter state and
            # silently emit near-uniform probabilities for every person. Treat
            # that as a broken prediction path and fall back to the coefficient-
            # based numpy softmax below.
            _uniform_gap = float(np.mean(np.abs(probs - (1.0 / 3.0))))
            _row_std = float(np.mean(np.std(probs, axis=1)))
            if _uniform_gap < 0.02 and _row_std < 0.02:
                raise RuntimeError(
                    "Larch CDAP probability() returned near-uniform probabilities; "
                    "falling back to numpy softmax"
                )

            # NOTE: no preschool P(H) floor is applied here.
            # HHTS observed preschool rates: H=5.6%, N=12.6%, M=81.9%.
            # Preschool children make predominantly mandatory trips (childcare /
            # kindergarten coded M, escorted school trips coded M).  The MNL
            # estimates b_M_ptype_Preschool directly from this observed distribution;
            # a hard floor would invert the model's learnt behaviour.

            cdap_idx = np.array([rng.choice(3, p=p.astype(np.float64)) for p in probs])

            # Map predictions from unique persons back to all rows in df
            if pid_col is not None:
                pid_to_cdap = {
                    str(pid): CDAP_LABELS[cdap_idx[i]]
                    for i, pid in enumerate(df_uniq[pid_col].astype(str))
                }
                df['CDAP'] = df[pid_col].astype(str).map(pid_to_cdap).fillna('M')
            else:
                df['CDAP'] = pd.Series(cdap_idx, index=df.index).map(CDAP_LABELS)

            # Expose the per-person M/N/H log-utilities so an optional household
            # coordination layer (cdap_households.coordinate_household_cdap) can
            # sit ON TOP of this fitted individual model, ActivitySim-style,
            # rather than re-drawing from crude ptype priors. probs columns are
            # ordered [H, N, M] per CDAP_LABELS.
            try:
                _lp = np.log(np.clip(probs, 1e-9, None))
                _util = {d: _lp[:, k] for k, d in CDAP_LABELS.items()}  # {'H':..,'N':..,'M':..}
                if pid_col is not None:
                    _ids = df_uniq[pid_col].astype(str).values
                    for d in ('M', 'N', 'H'):
                        _m = {pid: _util[d][i] for i, pid in enumerate(_ids)}
                        df[f'cdap_util_{d}'] = df[pid_col].astype(str).map(_m)
                else:
                    for d in ('M', 'N', 'H'):
                        df[f'cdap_util_{d}'] = pd.Series(_util[d], index=df.index)
            except Exception as _ue:
                logger.debug(f"CDAP per-person utility export skipped: {_ue}")

            dist = df['CDAP'].value_counts()
            print(f"  CDAP: H={dist.get('H',0)} ({dist.get('H',0)/len(df)*100:.1f}%)  "
                  f"N={dist.get('N',0)} ({dist.get('N',0)/len(df)*100:.1f}%)  "
                  f"M={dist.get('M',0)} ({dist.get('M',0)/len(df)*100:.1f}%)")
            return df
        except Exception as e:
            logger.warning(f"Larch CDAP predict failed: {e} — trying sklearn fallback")

    if sklearn_mdl is not None:
        X = df.reindex(columns=feat_cols).fillna(0).values.astype(float)
        probs = sklearn_mdl.predict_proba(X)
        probs = probs / probs.sum(axis=1, keepdims=True)
        cdap_idx = np.array([rng.choice(3, p=p.astype(np.float64)) for p in probs])
        df['CDAP'] = pd.Series(cdap_idx, index=df.index).map(CDAP_LABELS)
        dist = df['CDAP'].value_counts()
        print(f"  CDAP (sklearn): H={dist.get('H',0)}  N={dist.get('N',0)}  M={dist.get('M',0)}")
        return df

    # ── Numpy softmax fallback — uses fitted MNL pf directly ─────────────────
    # When Larch's DataTree prediction fails (e.g. "expected flow {flow!r}"
    # from mismatched DataTree on loaded models) and sklearn is None, we compute
    # probabilities from the parameter frame directly.
    # Model spec: V_H=0 (reference), V_N = ASC_N + b_N_feat*X, V_M = ASC_M + b_M_feat*X
    if mnl is not None:
        try:
            coefs = _extract_cdap_coefs(mnl, feat_cols)

            # df_uniq / pid_col may not be set if Larch failed before defining them
            _pid_col = locals().get('pid_col',
                                    'person_id' if 'person_id' in df.columns else None)
            if 'df_uniq' in locals() and locals()['df_uniq'] is not None:
                _df_uniq = locals()['df_uniq']
            elif _pid_col is not None:
                _df_uniq = df.drop_duplicates(subset=[_pid_col]).reset_index(drop=True)
            else:
                _df_uniq = df.copy()

            X_df = _df_uniq.reindex(columns=feat_cols).fillna(0)
            X_np = X_df.values.astype(float)   # (n_persons, n_features)

            # Build V_N and V_M from b_N_* and b_M_* coefficients
            V_N = np.full(len(X_np), coefs.get('ASC_N', 0.0))
            V_M = np.full(len(X_np), coefs.get('ASC_M', 0.0))
            for j, feat in enumerate(feat_cols):
                bn = coefs.get(f'b_N_{feat}', 0.0)
                bm = coefs.get(f'b_M_{feat}', 0.0)
                if bn != 0.0:
                    V_N += bn * X_np[:, j]
                if bm != 0.0:
                    V_M += bm * X_np[:, j]

            V = np.stack([np.zeros(len(X_np)), V_N, V_M], axis=1)   # (n, 3)
            V -= V.max(axis=1, keepdims=True)   # numerical stability
            exp_V = np.exp(V)
            probs = exp_V / exp_V.sum(axis=1, keepdims=True)

            # No preschool floor — HHTS observed: H=6%, N=13%, M=82%.
            # Model learnt this distribution from ptype_Preschool feature directly.
            cdap_idx = np.array([rng.choice(3, p=p.astype(np.float64)) for p in probs])

            if _pid_col is not None:
                pid_to_cdap = {
                    str(pid): CDAP_LABELS[cdap_idx[i]]
                    for i, pid in enumerate(_df_uniq[_pid_col].astype(str))
                }
                df['CDAP'] = df[_pid_col].astype(str).map(pid_to_cdap).fillna('M')
            else:
                df['CDAP'] = pd.Series(cdap_idx, index=df.index).map(CDAP_LABELS)

            dist = df['CDAP'].value_counts()
            print(f"  CDAP (numpy fallback): H={dist.get('H',0)} ({dist.get('H',0)/len(df)*100:.1f}%)  "
                  f"N={dist.get('N',0)} ({dist.get('N',0)/len(df)*100:.1f}%)  "
                  f"M={dist.get('M',0)} ({dist.get('M',0)/len(df)*100:.1f}%)")
            logger.warning("CDAP used numpy softmax fallback — Larch predict unavailable")
            return df
        except Exception as _np_e:
            logger.warning(f"CDAP numpy fallback also failed: {_np_e}")

    logger.warning("No CDAP model available — defaulting to M")
    df['CDAP'] = 'M'
    return df


def run_cdap(persons: pd.DataFrame,
             synth: pd.DataFrame,
             save_path: str = 'saved_data/cdap_model.pkl',
             seed: int = 42,
             feature_override: list = None) -> pd.DataFrame:
    print("=" * 55)
    print("CDAP — Coordinated Daily Activity Pattern")
    print("=" * 55)

    if feature_override and os.path.exists(save_path):
        # save_path is a single shared cache reused across all runs
        # regardless of mode. If this run supplies a GA-selected feature
        # set that differs from what's cached, the cache is stale for our
        # purposes — re-estimate rather than silently keep serving the old
        # (hardcoded-MNL_FEATURES) model.
        try:
            with open(save_path, 'rb') as _f:
                _cached = pickle.load(_f)
            _cached_feats = _cached.get('features') if isinstance(_cached, dict) else None
            if _cached_feats is not None and set(_cached_feats) != set(feature_override):
                logger.info(
                    "CDAP cache features differ from GA-selected feature_override "
                    f"({len(_cached_feats)} cached vs {len(feature_override)} requested) "
                    "— re-estimating"
                )
                os.remove(save_path)
        except Exception:
            # Unreadable/legacy cache format — leave the existing validity
            # check below to decide whether to delete it.
            pass

    if os.path.exists(save_path):
        # Patch Larch __setstate__ before loading so removed attributes
        # (availability_any) don't crash unpickling on newer Larch versions.
        try:
            import larch as _lx
            def _safe_ss(self, state):
                _SKIP = {'availability_any', 'availability_co', 'availability_ca'}
                for k, v in state.items():
                    if k in _SKIP: continue
                    try: setattr(self, k, v)
                    except (TypeError, AttributeError): pass
            for _attr in ('Model', 'NumbaModel'):
                _cls = getattr(_lx, _attr, None)
                if _cls is None:
                    _cls = getattr(getattr(_lx, 'model', None), _attr, None)
                if _cls and not getattr(_cls, '_safe_ss_patched', False):
                    _cls.__setstate__ = _safe_ss
                    _cls._safe_ss_patched = True
        except Exception:
            pass

        loaded = False
        for _name, _loader in [
            ('cloudpickle', lambda: __import__('cloudpickle').load(open(save_path, 'rb'))),
            ('dill',        lambda: __import__('dill').load(open(save_path, 'rb'))),
            ('pickle',      lambda: pickle.load(open(save_path, 'rb'))),
        ]:
            try:
                result = _loader()
                print(f"  Loaded cached CDAP ({_name}) ← {save_path}")
                loaded = True
                break
            except (ImportError, Exception):
                continue
        if not loaded:
            result = estimate_cdap(persons, save_path, feature_override=feature_override)
    else:
        result = estimate_cdap(persons, save_path, feature_override=feature_override)

    print("\n[2/2] Predicting synthetic population...")
    synth = predict_cdap(synth, result, seed=seed)
    return synth
