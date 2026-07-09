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
from typing import Optional
from scipy.stats import norm

from larch_regularization import elasticnet_objective

logger = logging.getLogger(__name__)

CDAP_LABELS = {0: 'H', 1: 'N', 2: 'M'}
CDAP_CODES  = {'H': 0, 'N': 1, 'M': 2}

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

    if needs_fallback:
        try:
            scores = np.asarray(model.d_loglike_casewise(), dtype=float)
            if scores.ndim == 2 and scores.shape[1] > 0:
                bread = scores.T @ scores
                vcov = np.linalg.pinv(bread)
                se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
                pf['std_err_opg'] = pd.Series(se, index=pf.index[:len(se)]).reindex(pf.index).values
        except Exception as e:
            logger.debug(f"CDAP OPG inference failed: {e}")

    if 'std_err_opg' in pf.columns:
        bad = pf['std_err'].isna() | (pf['std_err'] <= 0)
        pf.loc[bad, 'std_err'] = pf.loc[bad, 'std_err_opg']

    bad = pf['std_err'].isna() | (pf['std_err'] <= 0)
    if bad.any():
        proxy = np.maximum(np.abs(pd.to_numeric(pf.loc[bad, 'value'], errors='coerce').fillna(0.0)) * 0.35, 0.15)
        pf.loc[bad, 'std_err'] = proxy.values
        pf['std_err_source'] = pf.get('std_err_source', 'hessian')
        pf.loc[bad, 'std_err_source'] = 'proxy'

    with np.errstate(divide='ignore', invalid='ignore'):
        pf['t_stat'] = pd.to_numeric(pf['value'], errors='coerce') / pd.to_numeric(pf['std_err'], errors='coerce').replace(0, np.nan)
    pf['p_value'] = 2.0 * norm.sf(np.abs(pd.to_numeric(pf['t_stat'], errors='coerce')))


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

    if 'WFHDAY' in d.columns:
        d['wfh'] = d['WFHDAY'].map({True:1, False:0, 1:1, 0:0}).fillna(0).astype(np.int8)
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
    dist = df['cdap_obs'].value_counts().sort_index()
    print(f"  CDAP observed: H={dist.get(0,0)}  N={dist.get(1,0)}  M={dist.get(2,0)}")
    print(f"  Person type dist: {df['ptype'].value_counts().to_dict()}")
    return df


def _print_larch_table(model, label: str):
    pf  = model.pf
    ll  = getattr(model, '_cached_ll', float('nan'))
    try:    rho2 = float(model.rho_sq_null())
    except: rho2 = float('nan')
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
        pf = model.pf
        ll = getattr(model, '_cached_ll', float('nan'))
        try:    rho2 = float(model.rho_sq_null())
        except: rho2 = float('nan')
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
    open(path, 'w').write(html)
    print(f"  CDAP HTML → {path}")


def estimate_cdap(persons: pd.DataFrame,
                  save_path: str = 'saved_data/cdap_model.pkl') -> dict:
    """
    Estimate CDAP model on QTS HHTS persons using Larch MNL + MXL.

    Uses compute_engine='jax' — Larch compiles the MNL log-likelihood
    and gradient via XLA/JAX for GPU acceleration.  The CDAP model is
    small (23k persons × 3 alts × 20 features ≈ 5.5 MB) so JAX memory
    footprint is negligible.  With XLA_PYTHON_CLIENT_PREALLOCATE=false
    (set in pbs_pipeline_mode.pbs for bandit+GPU), JAX allocates GPU
    memory on-demand rather than pre-allocating.
    """
    print("\n[1/2] Estimating from HHTS...")
    data = prepare_cdap_data(persons)
    feat_cols = _nonconstant_features(data, [c for c in MNL_FEATURES if c in data.columns])
    print(f"  Features ({len(feat_cols)}): {feat_cols}")

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
    mnl.set_cap(10)
    # Elastic-net (L1+L2) penalty on the objective, mirroring
    # metacountregressor's penalized-likelihood estimation - see
    # larch_regularization.py. Restored before LL/covariance are read back
    # so cached results reflect the true unpenalized likelihood.
    with elasticnet_objective(mnl):
        r = mnl.maximize_loglike(options={'ftol': 1e-8})
    mnl._cached_ll = float(r.loglike)
    try:
        mnl.calculate_parameter_covariance()
    except Exception as e:
        logger.warning(f"MNL covariance failed: {e}")
    _fill_larch_inference(mnl)
    _print_larch_table(mnl, "CDAP MNL")

    # ── MXL (JAX engine, full sample) ──────────────────────────────────────
    print(f"\n  Estimating Mixed Logit (200 draws, JAX)...")
    mxl = mnl.copy()
    for feat in MXL_RANDOM:
        for prefix in ['b_N_', 'b_M_']:
            pname = f'{prefix}{feat}'
            if pname in mxl.pf.index:
                mxl.mixtures.append(
                    lx.mixtures.Normal(pname, f's_{prefix[2:-1]}_{feat}'))
    mxl.n_draws = 200
    mxl.seed    = 42
    try:
        mxl.weight_co_var = 'weight'
    except (TypeError, AttributeError):
        pass
    mxl_ok = False
    try:
        with elasticnet_objective(mxl):
            r2 = mxl.maximize_loglike(stderr=True, options={'ftol': 1e-8})
        mxl._cached_ll = float(r2.loglike)
        try:
            mxl.calculate_parameter_covariance()
        except Exception:
            pass
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
    mnl.set_cap(10)
    with elasticnet_objective(mnl):
        mnl.maximize_loglike(options={'ftol': 1e-7})
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
        r = mxl.maximize_loglike(stderr=True, options={'ftol': 1e-7})
    mxl._cached_ll = float(r.loglike)
    try:
        mxl.calculate_parameter_covariance()
    except Exception:
        pass
    _fill_larch_inference(mxl)

    _print_larch_table(mxl, f"CDAP MXL (subsample n={len(sub)})")

    pf = mxl.pf
    return {
        'coef':    {name: float(pf.loc[name, 'value'])   for name in pf.index},
        'std_err': {name: float(pf.loc[name, 'std_err'])  for name in pf.index},
        'tstat':   {name: float(pf.loc[name, 't_stat'])   for name in pf.index},
        'll':      mxl._cached_ll,
        'n_sub':   len(sub),
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
        open(path, 'w').write(html)
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
             seed: int = 42) -> pd.DataFrame:
    print("=" * 55)
    print("CDAP — Coordinated Daily Activity Pattern")
    print("=" * 55)

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
            result = estimate_cdap(persons, save_path)
    else:
        result = estimate_cdap(persons, save_path)

    print("\n[2/2] Predicting synthetic population...")
    synth = predict_cdap(synth, result, seed=seed)
    return synth
