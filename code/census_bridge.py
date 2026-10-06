"""
census_bridge.py
=================
Three capabilities:

1. VARIABLE MAPPING
   Maps ABS census codes in the synthetic population to HHTS equivalents,
   with derived variables where direct mappings don't exist.

2. SYNTHETIC IMPUTATION
   Where the synthetic population lacks an HHTS variable entirely,
   imputes it from a fitted conditional distribution estimated from the
   survey data. Uses ordered logit / OLS / multinomial depending on
   the variable type.

3. EXTENDED GA
   Augments ga_hypothesis_testing.py to also search over:
     - landuse variable combinations (RETEMPN, FPSEMPN, EATOUT, etc.)
     - interaction terms between person attributes and landuse
     - skim variable combinations

ABS Census variable codes used
--------------------------------
    agep      age in 5-year bands (1=0-4, 2=5-9, ... 18=85+)
    age5p     exact 5-year band integer (0,5,10,…)
    sexp      sex (1=male, 2=female)
    nprd      household size (number of persons)
    vehrd     vehicles in household (0,1,2,3,4+)  ← direct match
    hrsp      hours worked per week (1=none, 2=1-15h, ..8=50h+)
    incp      personal income band (1-17)           ← direct match
    stup      student status (1=not studying, 2=ft, 3=pt, 4=other)
    typp      type of educational institution
    occp      occupation (1-digit ANZSCO)
    rlhp      relationship in household (1=head,2=spouse,3=child,…)
    mstp      marital status
    empp      employment status (1=employed ft, 2=employed pt, …)
    indp      industry of employment (1-digit ANZSIC)
    rntrd     rent/mortgage status
    tenlld    tenure type
    mqrd / mtwp  method of travel to work (can proxy mode preference)
    abspid    person ID
    abshid    household ID
    SA2_MAIN16  SA2 zone code
    ORDER     TAZ order index

Usage
-----
    from census_bridge import (
        map_census_to_hhts,
        impute_missing_attrs,
        run_ga_extended,
    )

    # 1. map and derive
    synth = pd.read_csv('saved_data/zone_ids_sa2/order_40.csv')
    synth_mapped = map_census_to_hhts(synth)

    # 2. impute remaining gaps from survey
    out          = Synth_Connect().process_survey(save=False)
    synth_full   = impute_missing_attrs(synth_mapped, out['persons'])

    # 3. extended GA including landuse variables
    skims   = pd.read_csv('OD_skims.csv')
    landuse = sc.get_land()
    result  = run_ga_extended(
        stage=5,
        trips=out['trips'], persons=out['persons'],
        skims=skims, landuse=landuse,
        synth=synth_full,
    )
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------------------------------------------------------
# 0.  ABS → HHTS VARIABLE MAPPING TABLES
# ---------------------------------------------------------------------------

# ── Age group ─────────────────────────────────────────────────────────────
# agep: 1=0-4, 2=5-9, 3=10-14, 4=15-19, 5=20-24, 6=25-29,
#        7=30-34, 8=35-39, 9=40-44, 10=45-49, 11=50-54,
#       12=55-59, 13=60-64, 14=65-69, 15=70-74, 16=75-79, 17=80-84, 18=85+
AGEP_TO_AGEGROUP = {
    1:  '0-4',   2:  '5-11',  3:  '5-11',  4:  '12-17',
    5:  '18-24', 6:  '25-34', 7:  '25-34', 8:  '35-44',
    9:  '35-44', 10: '45-54', 11: '45-54', 12: '55-64',
    13: '55-64', 14: '65-74', 15: '65-74', 16: '75+',
    17: '75+',   18: '75+',
}

# ── Work status (hrsp → WORKSTAT) ─────────────────────────────────────────
# ABS HRSP is HOURS WORKED PER WEEK (not a labour-force status):
#   1=0h (incl. N/A), 2=1-4, 3=5-9, 4=10-14, 5=15-19,
#   6=20-24, 7=25-29, 8=30-34, 9=35-39, 10=40h+
# WORKSTAT: 0=other/unemployed, 1=part-time, 2=full-time
# NOTE: prefer deriving WORKSTAT from LFSP where available — hours alone cannot
# distinguish "unemployed" from "employed 1-4h".
HRSP_TO_WORKSTAT = {
    1: 0,   # 0 hours / not applicable
    2: 1,   # 1-4h   → part-time
    3: 1,   # 5-9h   → part-time
    4: 1,   # 10-14h → part-time
    5: 1,   # 15-19h → part-time
    6: 1,   # 20-24h → part-time
    7: 1,   # 25-29h → part-time
    8: 1,   # 30-34h → part-time
    9: 2,   # 35-39h → full-time
    10: 2,  # 40h+    → full-time
}

# ── Study type (stup/typp → STUDYING) ─────────────────────────────────────
# stup: 1=not studying, 2=full-time, 3=part-time
# typp: 1=primary, 2=secondary, 3=combined, 4=special, 5=TAFE, 6=uni, 7=other
# STUDYING: 0=not studying, 1=primary/secondary, 2=secondary,
#            3=tertiary pt, 4=tertiary ft
def derive_studying(row) -> int:
    stup = row.get('stup', 1)
    typp = row.get('typp', 0)
    if stup == 1:
        return 0
    if typp in [1, 3]:           # primary / combined
        return 1
    if typp == 2:                # secondary
        return 2
    if typp in [4, 5, 6, 7]:    # TAFE / uni / other
        if stup == 3:
            return 3             # part-time tertiary
        return 4                 # full-time tertiary
    return 0

# ── Relationship (rlhp → HHTS RELATIONSHIP text) ──────────────────────────
# rlhp: 1=head, 2=spouse, 3=child, 4=other relative, 5=non-family
RLHP_TO_REL = {
    1: 'head', 2: 'spouse', 3: 'child', 4: 'relative', 5: 'nonFamily',
}

# ── Income band (incp is already 1-17, matches HHTS income_mapping) ───────
# No remapping needed — incp is a direct match.

# ── Sex (sexp: 1=male, 2=female — direct match) ───────────────────────────

# ── Vehicles (vehrd: 0,1,2,3,4+ — direct match) ──────────────────────────

# ── Employment status (empp → ptype helper) ───────────────────────────────
# empp: 1=employed ft, 2=employed pt, 3=unemployed looking,
#        4=unemployed not looking, 5=not in LF (retired/carer/student)
EMPP_TO_WORKSTAT = {
    1: 2,   # employed ft
    2: 1,   # employed pt
    3: 0,   # unemployed
    4: 0,
    5: 0,
}

# ── Method of travel to work (mtwp → mode preference proxy) ───────────────
# mtwp: 1=car driver, 2=car passenger, 3=train, 4=bus, 5=ferry,
#        6=tram/light rail, 7=walk, 8=bike, 9=motorbike, 10=taxi/rideshare,
#       11=worked from home, 12=did not go to work
MTWP_TO_MODE = {
    1: 'DA', 2: 'SR', 3: 'Transit', 4: 'Transit', 5: 'Transit',
    6: 'Transit', 7: 'Walk', 8: 'Bike', 9: 'DA', 10: 'SR',
    11: None, 12: None,
}

# ── Person type mapping ────────────────────────────────────────────────────
# Derived from combined agep + empp + stup/typp
def derive_ptype(row) -> str:
    agep  = row.get('agep', 10)
    empp  = row.get('empp', 5)
    stup  = row.get('stup', 1)
    typp  = row.get('typp', 0)
    hrsp  = row.get('hrsp', 1)

    if agep <= 1:                           # 0-4
        return 'Preschool'
    if agep in [2, 3, 4]:                  # 5-19 → school age
        if typp in [1, 2, 3]:
            return 'SchoolChild'
    if empp == 5 and agep >= 13:           # 60+ not in LF → retired proxy
        return 'Retired'
    if stup in [2, 3] and empp in [1, 2]:  # studying AND working
        return 'WorkingStudent'
    if stup in [2, 3]:                     # studying
        return 'UniversityStudent'
    if hrsp >= 6:                           # 35h+ → full time
        return 'FullTimeWorker'
    if hrsp in [3, 4, 5]:                  # 1-34h → part time
        return 'PartTimeWorker'
    return 'NonWorker'


# ---------------------------------------------------------------------------
# 1.  MAIN MAPPING FUNCTION
# ---------------------------------------------------------------------------

def map_census_to_hhts(
    synth: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Map ABS census codes in the synthetic population to HHTS equivalents.

    Adds or overwrites columns:
        AGEGROUP, WORKSTAT, STUDYING, RELATIONSHIP,
        ptype, preferred_mode,
        sexp, vehrd, incp, nprd  (pass-through if already present)

    Also derives:
        carlicence   : proxy from vehrd + agep
        has_vehicle  : binary
        no_vehicle   : binary
        income_hi    : incp >= 10  (>= $52k)
        income_lo    : incp <= 4   (<= $21k)
        age_working  : agep in working-age range (5-13, i.e. 20-64)
        age_retired  : agep >= 13 (60+)
        n_workers    : count of workers in household (approximated)

    Parameters
    ----------
    synth   : synthetic population DataFrame with ABS census columns
    verbose : print mapping summary

    Returns
    -------
    synth with new HHTS-equivalent columns added
    """
    df = synth.copy()

    # ── person_id alignment ───────────────────────────────────────────
    if 'person_id' not in df.columns:
        if 'abspid' in df.columns:
            df['person_id'] = df['abspid']
        else:
            df['person_id'] = df.index

    # ── AGEGROUP ──────────────────────────────────────────────────────
    if 'agep' in df.columns:
        _ap = pd.to_numeric(df['agep'], errors='coerce')
        # `agep` here means the 5-year BAND INDEX (1=0-4 ... 18=85+), NOT the raw
        # ABS AGEP code (0=age 0, 1=age 1, ... 25=25-29 ... 37=85+).  Feeding raw
        # AGEP silently mislabels every code outside 1-18 as '25-34' (~81% of the
        # population) because AGEP_TO_AGEGROUP has no keys for 0 or 19-37.
        # Use export_synth_pop_for_travel.py (or age5p below) to do the conversion.
        if _ap.max() is not None and _ap.max() > 18:
            raise ValueError(
                "agep looks like raw ABS AGEP codes (max=%s) but must be the 5-year "
                "band index 1..18 (1=0-4 ... 18=85+). Convert first, e.g. "
                "band = agep//5 + 1 for agep<=24 else agep-19, or supply 'age5p'."
                % _ap.max()
            )
        df['AGEGROUP'] = df['agep'].map(AGEP_TO_AGEGROUP).fillna('25-34')
    elif 'age5p' in df.columns:
        bins   = [0,5,12,18,25,35,45,55,65,75,200]
        labels = ['0-4','5-11','12-17','18-24','25-34','35-44',
                  '45-54','55-64','65-74','75+']
        df['AGEGROUP'] = pd.cut(df['age5p'], bins=bins,
                                labels=labels, right=False)

    # ── WORKSTAT ──────────────────────────────────────────────────────
    if 'hrsp' in df.columns:
        df['WORKSTAT'] = df['hrsp'].map(HRSP_TO_WORKSTAT).fillna(0).astype(int)
    elif 'empp' in df.columns:
        df['WORKSTAT'] = df['empp'].map(EMPP_TO_WORKSTAT).fillna(0).astype(int)

    # ── STUDYING ──────────────────────────────────────────────────────
    if 'stup' in df.columns:
        df['STUDYING'] = df.apply(derive_studying, axis=1)

    # ── RELATIONSHIP ──────────────────────────────────────────────────
    if 'rlhp' in df.columns:
        df['RELATIONSHIP'] = df['rlhp'].map(RLHP_TO_REL).fillna('nonFamily')

    # ── ptype ─────────────────────────────────────────────────────────
    df['ptype'] = df.apply(derive_ptype, axis=1)

    # ── carlicence proxy ──────────────────────────────────────────────
    if 'carlicence' not in df.columns:
        if 'agep' in df.columns and 'vehrd' in df.columns:
            df['carlicence'] = ((df['agep'] >= 4) & (df['vehrd'] >= 1)).astype(int)
        else:
            df['carlicence'] = 0

    # ── preferred mode from travel to work ────────────────────────────
    if 'mtwp' in df.columns:
        df['preferred_mode'] = df['mtwp'].map(MTWP_TO_MODE)

    # ── derived binary / numeric features ────────────────────────────
    if 'vehrd' in df.columns:
        df['has_vehicle'] = (df['vehrd'] >= 1).astype(float)
        df['no_vehicle']  = (df['vehrd'] == 0).astype(float)

    if 'incp' in df.columns:
        df['income_hi'] = (df['incp'] >= 10).astype(float)
        df['income_lo'] = (df['incp'] <= 4).astype(float)

    if 'agep' in df.columns:
        df['age_working'] = df['agep'].between(5, 13).astype(float)
        df['age_retired'] = (df['agep'] >= 13).astype(float)
        df['age_young']   = (df['agep'] <= 4).astype(float)

    # ── household-level aggregates ───────────────────────────────────
    if 'abshid' in df.columns and 'WORKSTAT' in df.columns:
        hh_workers = (
            df[df['WORKSTAT'] == 2]
            .groupby('abshid')
            .size()
            .reset_index(name='n_workers_hh')
        )
        df = df.merge(hh_workers, on='abshid', how='left')
        df['n_workers_hh'] = df['n_workers_hh'].fillna(0).astype(int)

    # ── OTAZ alignment ────────────────────────────────────────────────
    if 'OTAZ' not in df.columns:
        if 'ORDER' in df.columns:
            df['OTAZ'] = df['ORDER'].astype(int)
        elif 'SA2_MAIN16' in df.columns:
            df['OTAZ'] = df['SA2_MAIN16'].astype(int)

    if verbose:
        print("Census → HHTS mapping complete.")
        if 'ptype' in df.columns:
            print("  ptype distribution:")
            for k, v in df['ptype'].value_counts().items():
                print(f"    {k}: {v} ({100*v/len(df):.1f}%)")
        mapped = [c for c in ['AGEGROUP','WORKSTAT','STUDYING',
                               'RELATIONSHIP','ptype','carlicence']
                  if c in df.columns]
        print(f"  Columns mapped: {mapped}")

    return df


# ---------------------------------------------------------------------------
# 2.  SYNTHETIC IMPUTATION OF MISSING HHTS ATTRIBUTES
# ---------------------------------------------------------------------------

def impute_missing_attrs(
    synth: pd.DataFrame,
    survey_persons: pd.DataFrame,
    attrs_to_impute: Optional[List[str]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Impute HHTS attributes that are absent from the synthetic population
    by sampling from conditional distributions estimated from the survey.

    Strategy per variable type:
        Categorical / ordinal  : sample from empirical conditional given ptype
        Continuous / count     : sample from fitted normal given ptype

    Variables imputed if missing:
        WFHWEEK       : WFH days per week
        MAINACT       : main activity type
        ANYSTOPS      : any trips flag
        bikes         : bikes in household
        INDUSTRY      : industry code (uses indp from census if available)

    Parameters
    ----------
    synth           : synthetic population (already mapped with map_census_to_hhts)
    survey_persons  : HHTS persons DataFrame from process_survey()["persons"]
    attrs_to_impute : list of columns to impute; None = auto-detect missing
    seed            : random seed
    verbose         : print imputation summary

    Returns
    -------
    synth with imputed columns added
    """
    rng = np.random.default_rng(seed)
    df  = synth.copy()

    pers = survey_persons.copy()
    if 'person_id' not in pers.columns and 'PERSID' in pers.columns:
        pers = pers.rename(columns={'PERSID': 'person_id'})

    # ensure ptype exists
    if 'ptype' not in df.columns:
        df = map_census_to_hhts(df, verbose=False)
    if 'ptype' not in pers.columns:
        try:
            from stage1_activity_frequency import assign_ptype_column
            pers = assign_ptype_column(pers)
        except ImportError:
            pers['ptype'] = 'NonWorker'

    # auto-detect columns to impute
    if attrs_to_impute is None:
        candidates = ['WFHWEEK', 'MAINACT', 'ANYSTOPS', 'bikes',
                      'INDUSTRY', 'WORKSTATUS']
        attrs_to_impute = [c for c in candidates if c not in df.columns
                           and c in pers.columns]

    if not attrs_to_impute:
        if verbose:
            print("No attributes to impute — all present or none in survey.")
        return df

    if verbose:
        print(f"Imputing {len(attrs_to_impute)} attributes: {attrs_to_impute}")

    # build conditional distributions per ptype
    ptypes = df['ptype'].unique()

    for attr in attrs_to_impute:
        if attr not in pers.columns:
            continue

        df[attr] = np.nan
        imputed_count = 0

        for pt in ptypes:
            pt_mask_synth  = df['ptype']   == pt
            pt_mask_survey = pers['ptype'] == pt

            survey_vals = pers.loc[pt_mask_survey, attr].dropna()

            if len(survey_vals) == 0:
                # fall back to overall distribution
                survey_vals = pers[attr].dropna()

            if len(survey_vals) == 0:
                continue

            n_needed = pt_mask_synth.sum()
            if n_needed == 0:
                continue

            # categorical / low-cardinality: sample from empirical distribution
            unique_vals = survey_vals.unique()
            if len(unique_vals) <= 20 or survey_vals.dtype == object:
                val_counts = survey_vals.value_counts(normalize=True)
                sampled = rng.choice(
                    val_counts.index.values,
                    size=n_needed,
                    p=val_counts.values,
                )
            else:
                # continuous: sample from fitted normal
                mu, sd = survey_vals.mean(), survey_vals.std()
                sd     = sd if sd > 0 else 1.0
                sampled = rng.normal(mu, sd, size=n_needed)
                # round if the survey values are integers
                if pd.api.types.is_integer_dtype(survey_vals):
                    sampled = np.round(sampled).astype(int)

            df.loc[pt_mask_synth, attr] = sampled
            imputed_count += n_needed

        if verbose:
            n_missing = df[attr].isna().sum()
            print(f"  {attr}: imputed {imputed_count}, still missing {n_missing}")
            if n_missing > 0:
                # fill remaining with mode
                mode_val = pers[attr].mode().iloc[0] if not pers[attr].mode().empty else 0
                df[attr] = df[attr].fillna(mode_val)

    # ── special derivations that depend on imputed values ─────────────
    # INDUSTRY from indp (census) if available
    if 'INDUSTRY' not in df.columns and 'indp' in df.columns:
        indp_to_industry = {
            1: 'Agriculture', 2: 'Mining', 3: 'Manufacturing',
            4: 'Utilities', 5: 'Construction', 6: 'Wholesale',
            7: 'Retail', 8: 'Accommodation', 9: 'Transport',
            10: 'IT', 11: 'Finance', 12: 'Real Estate',
            13: 'Professional', 14: 'Administration', 15: 'Government',
            16: 'Education', 17: 'Health', 18: 'Arts',
            19: 'Other', 20: 'Not applicable',
        }
        df['INDUSTRY'] = df['indp'].map(indp_to_industry).fillna('Other')

    # MAINACT from ptype if still missing
    if 'MAINACT' not in df.columns:
        ptype_to_mainact = {
            'FullTimeWorker':    'Full-time Work',
            'PartTimeWorker':    'Part-time Work',
            'UniversityStudent': 'Full-time TAFE/Uni',
            'WorkingStudent':    'Full-time TAFE/Uni',
            'SchoolChild':       'Secondary School',
            'Preschool':         'Childcare/Kindergarten',
            'Retired':           'Retired',
            'NonWorker':         'Other Activity',
            # legacy labels — keep for backward compat with old data files
            'Student':           'Full-time TAFE/Uni',
            'School':            'Secondary School',
            'Joint Study & Work':'Full-time TAFE/Uni',  # old JointWorkStudy label
        }
        df['MAINACT'] = df['ptype'].map(ptype_to_mainact).fillna('Other Activity')

    return df


# ---------------------------------------------------------------------------
# 3.  EXTENDED FEATURE CANDIDATE LISTS WITH LANDUSE
# ---------------------------------------------------------------------------

# All landuse columns from your actual table
ALL_LANDUSE_COLS = [
    'RETEMPN', 'FPSEMPN', 'HEREMPN', 'AGREMPN', 'MWTEMPN', 'OTHEMP',
    'HSENROLL', 'AGE0519', 'TOTPOP', 'TOTHH',
    'Primary_total_priv', 'Secondary_total_priv', 'Primary', 'Secondary',
    'EATOUT', 'ESCORT', 'Education', 'Entertainment', 'Healthcare',
    'LandUse', 'OTHERDISCRFacilities', 'OTHERMAINTENANCE',
    'Public_Services', 'Shopping', 'airport', 'childcare', 'civic',
    'civic_admin', 'commercial', 'communications', 'community',
    'construction', 'education', 'government', 'governmental', 'harbour',
    'industrial', 'institutional', 'land_reclamation', 'logistics',
    'military', 'park', 'recreation_ground', 'residential', 'retail',
    'sport', 'trees', 'utility', 'village_green', 'sa2_area',
    'POPULATION',
]

# Purpose-relevant landuse subsets — GA starts from these, can explore wider
PURPOSE_LANDUSE_CANDIDATES = {
    'work':     ['FPSEMPN', 'MWTEMPN', 'OTHEMP', 'HEREMPN', 'AGREMPN',
                 'commercial', 'industrial', 'government', 'governmental',
                 'TOTPOP', 'sa2_area'],
    'edu':      ['HSENROLL', 'Education', 'education', 'childcare',
                 'Primary_total_priv', 'Secondary_total_priv',
                 'Primary', 'Secondary', 'AGE0519', 'TOTPOP'],
    'social':   ['Entertainment', 'EATOUT', 'recreation_ground', 'sport',
                 'park', 'village_green', 'community', 'civic', 'TOTHH',
                 'residential', 'TOTPOP'],
    'escort':   ['TOTHH', 'AGE0519', 'TOTPOP', 'childcare',
                 'Primary', 'Secondary', 'residential'],
    'shopping': ['RETEMPN', 'Shopping', 'retail', 'commercial', 'EATOUT',
                 'TOTPOP', 'TOTHH'],
    'othmaint': ['OTHERMAINTENANCE', 'Healthcare', 'civic', 'community',
                 'civic_admin', 'Public_Services', 'TOTHH', 'TOTPOP'],
    'othdiscr': ['OTHERDISCRFacilities', 'TOTPOP', 'residential',
                 'entertainment', 'sport', 'park', 'OTHEMP', 'EATOUT'],
    'eat':      ['EATOUT', 'RETEMPN', 'retail', 'commercial', 'TOTPOP',
                 'TOTHH', 'entertainment'],
}

# All skim candidates
ALL_SKIM_COLS = [
    'tt_avg', 'tt_am', 'tt_pm', 'log_dist',
    'distwalk', 'distbike', 'crash',
    'dist',
]

# Extended person candidates including ABS-derived columns
ALL_PERSON_CANDIDATES = [
    # HHTS core
    'sexp', 'vehrd', 'nprd', 'WORKSTAT', 'STUDYING', 'incp', 'rlhp',
    # ABS-derived
    'carlicence', 'has_vehicle', 'no_vehicle',
    'income_hi', 'income_lo',
    'age_working', 'age_retired', 'age_young',
    'n_workers_hh',
    # ABS direct
    'agep', 'empp', 'occp', 'rntrd', 'mstp',
    'mtwp',    # travel to work mode preference
    'bikes',   # bikes in hh
]


# ---------------------------------------------------------------------------
# 4.  EXTENDED GA — INCLUDES LANDUSE VARIABLE SELECTION
# ---------------------------------------------------------------------------

def _build_dest_fitness_extended(
    obs: pd.DataFrame,
    purpose: str,
    skim_lookup: dict,
    lu_df: pd.DataFrame,
    lu_size_cols: dict,
    landuse_candidates: List[str],
    skim_candidates: List[str],
    n_landuse_genes: int,
    n_skim_genes: int,
):
    """
    Returns a fitness function for destination choice that includes
    GA selection over BOTH skim features AND landuse size variables.

    Chromosome layout:
        [skim_genes (len=n_skim_genes) | landuse_genes (len=n_landuse_genes)]
    """
    from stage5_destination_choice import MNLDestinationModel, N_ALTS
    import numpy as np

    zones   = lu_df['DTAZ'].values
    N_ALTS_ = min(N_ALTS, len(zones))

    def fitness(genes: np.ndarray) -> float:
        skim_genes    = genes[:n_skim_genes]
        landuse_genes = genes[n_skim_genes:]

        sel_skims   = [f for f, g in zip(skim_candidates, skim_genes) if g == 1]
        sel_landuse = [f for f, g in zip(landuse_candidates, landuse_genes) if g == 1]

        if not sel_skims and not sel_landuse:
            return -1e9

        # build a custom size variable from selected landuse columns
        avail_lu = [c for c in sel_landuse if c in lu_df.columns]
        if avail_lu:
            lu_df_copy = lu_df.copy()
            lu_df_copy['_custom_size'] = np.log1p(
                lu_df_copy[avail_lu].fillna(0).sum(axis=1)
            )
            # compute accessibility from same cols
            lu_df_copy['_custom_acc'] = lu_df_copy['_custom_size']
        else:
            lu_df_copy = lu_df.copy()
            lu_df_copy['_custom_size'] = lu_df_copy.get(
                lu_size_cols.get(purpose, 'TOTPOP'),
                pd.Series(0, index=lu_df.index)
            )
            lu_df_copy['_custom_acc']  = lu_df_copy.get('accessibility', 0)

        # build feature matrix for chosen destinations
        rows = []
        for _, r in obs.iterrows():
            otaz = int(r['OTAZ']); dtaz = int(r['DTAZ'])
            sk   = skim_lookup.get((otaz, dtaz), {})

            tt_am   = sk.get('tt_am',   0)
            tt_pm   = sk.get('tt_pm',   0)
            tt_avg  = (tt_am + tt_pm) / 2
            log_dist= sk.get('log_dist', 0)
            dw      = sk.get('distwalk', 0)
            db      = sk.get('distbike', 0)
            crash   = sk.get('crash',    0)
            dist    = sk.get('dist',     0)

            lu_rows = lu_df_copy[lu_df_copy['DTAZ'] == dtaz]
            size = float(lu_rows['_custom_size'].iloc[0]) if len(lu_rows) else 0.0
            acc  = float(lu_rows['_custom_acc'].iloc[0])  if len(lu_rows) else 0.0

            skim_vals = {
                'tt_avg': tt_avg, 'tt_am': tt_am, 'tt_pm': tt_pm,
                'log_dist': log_dist, 'distwalk': dw, 'distbike': db,
                'crash': crash, 'dist': dist,
            }
            feat_row = [skim_vals.get(s, 0) for s in sel_skims] + [size, acc]
            rows.append(feat_row)

        if not rows:
            return -1e9

        X_chosen = np.array(rows)
        N = len(X_chosen)
        K = X_chosen.shape[1]

        # sample alternatives
        rng_ = np.random.default_rng(999)
        zone_weights = np.exp(
            lu_df_copy['_custom_size'].values.astype(float).clip(-10, 10)
        ) + 1.0
        zone_weights /= zone_weights.sum()

        alt_idx = rng_.choice(len(zones), size=(N, N_ALTS_),
                               p=zone_weights, replace=True)
        X_alts_list = []
        for i in range(N):
            otaz = int(obs['OTAZ'].iloc[i])
            alt_rows = []
            for j in alt_idx[i]:
                d  = int(zones[j])
                sk = skim_lookup.get((otaz, d), {})
                lu_row = lu_df_copy[lu_df_copy['DTAZ'] == d]

                tt_am  = sk.get('tt_am',   0); tt_pm = sk.get('tt_pm', 0)
                tt_avg = (tt_am + tt_pm) / 2
                ld     = sk.get('log_dist', 0)
                dw     = sk.get('distwalk', 0)
                db_    = sk.get('distbike', 0)
                cr     = sk.get('crash',    0)
                dist_  = sk.get('dist',     0)

                sz = float(lu_row['_custom_size'].iloc[0]) if len(lu_row) else 0.0
                ac = float(lu_row['_custom_acc'].iloc[0])  if len(lu_row) else 0.0

                sv = {'tt_avg':tt_avg,'tt_am':tt_am,'tt_pm':tt_pm,
                      'log_dist':ld,'distwalk':dw,'distbike':db_,
                      'crash':cr,'dist':dist_}
                alt_rows.append([sv.get(s, 0) for s in sel_skims] + [sz, ac])
            X_alts_list.append(alt_rows)

        X_alts = np.array(X_alts_list).reshape(N * N_ALTS_, K)

        # simple MNL fit
        from scipy.optimize import minimize
        init = np.zeros(K)

        def neg_ll(params):
            V_ch  = X_chosen @ params
            V_alt = (X_alts @ params).reshape(N, N_ALTS_)
            V_all = np.column_stack([V_ch, V_alt])
            V_all -= V_all.max(axis=1, keepdims=True)
            denom = np.exp(V_all).sum(axis=1)
            return -np.sum(V_ch - np.log(denom + 1e-12))

        try:
            res = minimize(neg_ll, init, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-7})
        except Exception:
            return -1e9

        ll_fitted = -res.fun
        ll_null   = N * np.log(1.0 / max(len(zones), 2))
        rho2      = 1 - ll_fitted / ll_null if ll_null != 0 else 0.0
        k_pen     = 0.001 * K
        return float(np.clip(rho2 - k_pen, -1, 1))

    return fitness


def run_ga_extended(
    stage: int,
    trips:         Optional[pd.DataFrame] = None,
    tours:         Optional[pd.DataFrame] = None,
    persons:       Optional[pd.DataFrame] = None,
    skims                                 = None,
    landuse:       Optional[pd.DataFrame] = None,
    synth:         Optional[pd.DataFrame] = None,
    purpose:       str   = 'work',
    pop_size:      int   = 40,
    n_generations: int   = 60,
    mutation_rate: float = 0.08,
    n_restarts:    int   = 2,
    seed:          int   = 42,
    verbose:       bool  = True,
) -> 'GAResult':
    """
    Extended GA searching over skim + landuse variable combinations for
    destination choice (stage 5) or person + landuse for mode choice (stage 6).

    Uses the improved GeneticAlgorithm from ga_hypothesis_testing:
    - Fitness caching (no re-estimation of duplicate chromosomes)
    - Adaptive mutation (boosts when diversity drops or run stagnates)
    - Multi-restart (n_restarts shorter re-runs after main run)
    - Aggregated fitness across ALL purposes (not just the requested one)
      so selected variables work broadly, not just for 'work' or 'shopping'

    For stages 1–4, delegates to run_ga_stage() from ga_hypothesis_testing.

    Parameters
    ----------
    stage         : 1–6 (5 and 6 use extended landuse+skim search)
    purpose       : primary purpose to weight more heavily in fitness
                    (still aggregates across all valid purposes)
    pop_size      : GA population size
    n_generations : generations in the main run
    mutation_rate : base per-gene flip probability (adapts upward automatically)
    n_restarts    : short restart runs after main run
    seed          : random seed
    verbose       : print generation progress
    """
    from ga_hypothesis_testing import (
        GeneticAlgorithm, GAResult, Individual,
        _stage6_fitness_aggregate,
    )

    if stage not in [5, 6]:
        from ga_hypothesis_testing import run_ga_stage
        return run_ga_stage(
            stage=stage, tours=tours, trips=trips, persons=persons,
            skims=skims, landuse=landuse,
            pop_size=pop_size, n_generations=n_generations,
            mutation_rate=mutation_rate, n_restarts=n_restarts,
            seed=seed, verbose=verbose,
        )

    ga_kwargs = dict(
        pop_size         = pop_size,
        n_generations    = n_generations,
        mutation_rate    = mutation_rate,
        crossover_rate   = 0.75,
        elitism_k        = 4,
        min_features     = 3,    # at least 3 features: prevents degenerate 1-2 var models
        tournament_k     = 4,
        stagnation_limit = 12,
        seed             = seed,
        verbose          = verbose,
    )

    # ── Stage 5: skim + PURPOSE-SPECIFIC landuse ──────────────────────
    if stage == 5:
        print(f"\n{'='*60}")
        print(f"EXTENDED GA — STAGE 5 DESTINATION  (estimation only)")
        print(f"  Purpose focus: {purpose}  |  all purposes aggregated in fitness")
        print(f"  pop={pop_size}  gens={n_generations}  restarts={n_restarts}")
        print(f"{'='*60}")

        import openmatrix as omx
        from stage5_destination_choice import (
            _prep_hhts_tours, _skim_flat, _skim_lookup, _filter_seg,
            _build_mode_model, _build_dest_model, _estimate,
            _logsums, _mode_tree, _dest_tree,
            _attach_size_term, _attach_acdc_terms,
        )

        # build skim flat once
        close_skims = False
        if isinstance(skims, str):
            skims       = omx.open_file(skims, 'r')
            close_skims = True
        skim_df = _skim_flat(skims)

        # Use tours table (has tour_mode) not trips (no tour_mode column).
        # Falls back to trips with dominant-mode derivation if tours is None.
        _src = tours if (tours is not None and 'tour_mode' in tours.columns) else trips
        if _src is trips:
            logger.warning("census_bridge run_ga_extended: "
                           "tours table not provided or has no tour_mode — "
                           "deriving mode from trips. Pass tours=out['tours'] for best results.")
        hhts_tours = _prep_hhts_tours(_src, persons)

        # ── Candidate sets: skim + full PURPOSE-aware landuse ─────────────
        skim_cands = ALL_SKIM_COLS                          # 8 skim vars
        # union of all purpose-relevant landuse cols (deduped, ordered)
        lu_cands_seen = {}
        for p_cands in PURPOSE_LANDUSE_CANDIDATES.values():
            for c in p_cands:
                lu_cands_seen.setdefault(c, 0)
                lu_cands_seen[c] += 1
        # Sort by how many purposes use this column (broadest first)
        lu_cands_ordered = sorted(lu_cands_seen, key=lu_cands_seen.get, reverse=True)
        # Filter to what actually exists in the landuse table
        if landuse is not None:
            lu_cands = [c for c in lu_cands_ordered if c in landuse.columns]
        else:
            lu_cands = lu_cands_ordered

        n_s = len(skim_cands)
        n_l = len(lu_cands)
        feat_names = skim_cands + lu_cands
        print(f"  Skim candidates ({n_s}): {skim_cands}")
        print(f"  Landuse candidates ({n_l}): {lu_cands[:10]}{'...' if n_l > 10 else ''}")

        segments = ['work', 'edu', 'nonmand']
        valid_segs = [s for s in segments if len(_filter_seg(hhts_tours, s)) >= 50]

        def fitness(genes):
            skim_genes = genes[:n_s]
            lu_genes   = genes[n_s:]
            sel_skims  = [c for c, g in zip(skim_cands, skim_genes)  if g == 1]
            sel_lu     = [c for c, g in zip(lu_cands,   lu_genes)    if g == 1]
            if not sel_skims and not sel_lu:
                return -1e9

            scores = []
            for seg in valid_segs:
                seg_tours = _filter_seg(hhts_tours, seg)
                try:
                    # mode model (uses skim vars from sel_skims)
                    dt_m     = _mode_tree(seg_tours, persons, landuse, skims)
                    mode_mdl = _build_mode_model(dt_m, seg,
                                                 custom_mode_vars=sel_skims or None)
                    mode_mdl = _estimate(mode_mdl)
                    mode_ll  = float(mode_mdl.loglike())

                    ls = _logsums(mode_mdl, dt_m, skims=skims, verbose=False)

                    # dest model (uses landuse vars from sel_lu)
                    lu_st = _attach_size_term(landuse, seg)
                    lu_st, seg_acdc = _attach_acdc_terms(
                        lu_st, seg_tours, skim_df, seg)
                    dt_d     = _dest_tree(seg_acdc, persons, lu_st, skims, ls)
                    dest_mdl = _build_dest_model(dt_d, seg,
                                                  custom_dest_vars=sel_lu or None)
                    dest_mdl = _estimate(dest_mdl)
                    dest_ll  = float(dest_mdl.loglike())

                    n_params = len(sel_skims) + len(sel_lu)
                    penalty  = 0.5 * n_params * np.log(max(len(seg_tours), 2))
                    scores.append(mode_ll + dest_ll - penalty)
                except Exception as e:
                    logger.debug(f"Extended GA stage5 [{seg}]: {e}")
            return float(np.mean(scores)) if scores else -1e9

        ga  = GeneticAlgorithm(len(feat_names), fitness, **ga_kwargs)
        best, history, pop = ga.run_with_restarts(n_restarts=n_restarts)

        if close_skims:
            skims.close()

        sel_skims = [f for f, g in zip(skim_cands, best.genes[:n_s]) if g == 1]
        sel_lu    = [f for f, g in zip(lu_cands,   best.genes[n_s:]) if g == 1]

        print(f"\n  Best skim features   : {sel_skims}")
        print(f"  Best landuse features: {sel_lu}")
        print(f"  Best fitness         : {best.fitness:.4f}")
        print(f"  Cache hits           : {ga.cache_hits}")

        return GAResult(
            stage=5,
            best_features={'skim': sel_skims, 'landuse': sel_lu},
            best_fitness=best.fitness, best_individual=best,
            history=history, all_candidates=pop,
            feature_names=feat_names, fitness_metric='mean(mode_LL+dest_LL)',
            n_restarts=n_restarts, cache_hits=ga.cache_hits,
        )

    # ── Stage 6: person + landuse accessibility → mode choice ────────
    if stage == 6:
        print(f"\n{'='*60}")
        print(f"EXTENDED GA — STAGE 6 MODE CHOICE  (estimation only)")
        print(f"  Person cols + landuse accessibility  |  all purposes aggregated")
        print(f"  pop={pop_size}  gens={n_generations}  restarts={n_restarts}")
        print(f"{'='*60}")

        try:
            from stage6_mode_choice import prepare_mode_data, PURPOSES
        except ImportError:
            PURPOSES = ['work', 'edu', 'shopping', 'social',
                        'escort', 'othmaint', 'othdiscr', 'eat']
            def prepare_mode_data(trips, persons, landuse):
                return trips

        data = prepare_mode_data(trips, persons, landuse)

        # attach log-transformed landuse accessibility columns per destination
        if landuse is not None:
            lu = landuse.copy()
            zc = next((c for c in ['ORDER', 'DTAZ', 'SA2_MAIN16']
                       if c in lu.columns), None)
            if zc and zc != 'DTAZ':
                lu = lu.rename(columns={zc: 'DTAZ'})
            lu['DTAZ'] = pd.to_numeric(lu['DTAZ'], errors='coerce').astype(int)
            # Use the union of all purpose-specific landuse columns
            all_lu_for_acc = list({c for cols in PURPOSE_LANDUSE_CANDIDATES.values()
                                   for c in cols if c in lu.columns})
            for col in all_lu_for_acc:
                merged = data.merge(lu[['DTAZ', col]], on='DTAZ', how='left')
                data[f'acc_{col}'] = np.log1p(
                    merged[col].fillna(0).values)

        # Full person + ABS candidate set
        person_cands = [c for c in ALL_PERSON_CANDIDATES if c in data.columns]
        acc_cands    = [c for c in data.columns if c.startswith('acc_')]
        feat_names   = person_cands + acc_cands
        n_p          = len(person_cands)

        valid_purposes = [p for p in PURPOSES
                          if len(data[data['purpose'] == p]) >= 30
                          and data[data['purpose'] == p].get(
                              'chosen_mode', pd.Series()).nunique() >= 2]

        print(f"  Person candidates ({n_p}): {person_cands}")
        print(f"  Accessibility candidates ({len(acc_cands)}): "
              f"{acc_cands[:8]}{'...' if len(acc_cands)>8 else ''}")
        print(f"  Valid purposes: {valid_purposes}")

        def fitness(genes):
            all_feats = [f for f, g in zip(feat_names, genes) if g == 1]
            if not all_feats:
                return -1e9
            return _stage6_fitness_aggregate(
                genes, data, feat_names, valid_purposes)

        ga  = GeneticAlgorithm(len(feat_names), fitness, **ga_kwargs)
        best, history, pop = ga.run_with_restarts(n_restarts=n_restarts)

        sel_person = [f for f, g in zip(person_cands, best.genes[:n_p]) if g == 1]
        sel_acc    = [f for f, g in zip(acc_cands,    best.genes[n_p:]) if g == 1]

        print(f"\n  Best person features : {sel_person}")
        print(f"  Best acc features    : {sel_acc}")
        print(f"  Best fitness         : {best.fitness:.4f}")
        print(f"  Cache hits           : {ga.cache_hits}")

        return GAResult(
            stage=6,
            best_features={'person': sel_person, 'accessibility': sel_acc},
            best_fitness=best.fitness, best_individual=best,
            history=history, all_candidates=pop,
            feature_names=feat_names, fitness_metric='mean-rho²',
            n_restarts=n_restarts, cache_hits=ga.cache_hits,
        )

# ---------------------------------------------------------------------------
# 5.  QUICK DIAGNOSTICS — compare mapped synth vs survey
# ---------------------------------------------------------------------------

def compare_distributions(
    synth: pd.DataFrame,
    survey: pd.DataFrame,
    cols: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot side-by-side distributions of key person attributes
    between the synthetic population and the survey, to validate
    the mapping.

    Parameters
    ----------
    synth   : mapped synthetic population
    survey  : HHTS persons from process_survey()["persons"]
    cols    : columns to compare; None = auto
    """
    import matplotlib.pyplot as plt

    if cols is None:
        cols = [c for c in
                ['WORKSTAT', 'STUDYING', 'vehrd', 'sexp', 'incp',
                 'age_working', 'carlicence', 'ptype']
                if c in synth.columns and c in survey.columns]

    n    = len(cols)
    cols_ = 3
    rows_ = int(np.ceil(n / cols_))
    fig, axes = plt.subplots(rows_, cols_,
                             figsize=(14, 3.5 * rows_),
                             facecolor='white')
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        sv = synth[col].dropna()
        su = survey[col].dropna()

        if su.dtype == object or su.nunique() <= 15:
            # categorical
            sv_p = sv.value_counts(normalize=True).sort_index()
            su_p = su.value_counts(normalize=True).sort_index()
            idx  = su_p.index.union(sv_p.index)
            sv_p = sv_p.reindex(idx, fill_value=0)
            su_p = su_p.reindex(idx, fill_value=0)

            x = np.arange(len(idx))
            w = 0.35
            ax.bar(x - w/2, su_p.values, w, label='survey',
                   color='#534AB7', alpha=0.75)
            ax.bar(x + w/2, sv_p.values, w, label='synthetic',
                   color='#1D9E75', alpha=0.75)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in idx],
                                rotation=45, ha='right', fontsize=8)
        else:
            # continuous
            bins = np.linspace(
                min(su.min(), sv.min()),
                max(su.max(), sv.max()), 25
            )
            ax.hist(su.values, bins=bins, density=True,
                    alpha=0.65, color='#534AB7', label='survey')
            ax.hist(sv.values, bins=bins, density=True,
                    alpha=0.65, color='#1D9E75', label='synthetic')

        ax.set_title(col, fontsize=10, fontweight='normal')
        ax.legend(fontsize=7, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Survey vs synthetic population — attribute distributions',
                 fontsize=12, fontweight='normal', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 6.  BACKWARD-COMPAT SHIMS
#     pipeline_logger.py calls  from census_bridge import run_ga_stageN
#     These delegate to run_ga_stage() from ga_hypothesis_testing so the
#     old call sites continue to work without any pipeline_logger edits.
# ---------------------------------------------------------------------------

def run_ga_stage1(tours, persons, **kw):
    """Shim — delegates to ga_hypothesis_testing.run_ga_stage(stage=1)."""
    from ga_hypothesis_testing import run_ga_stage
    return run_ga_stage(stage=1, tours=tours, persons=persons, **kw)


def run_ga_cdap(persons, **kw):
    """Shim — delegates to ga_hypothesis_testing.run_ga_stage(stage=0)."""
    from ga_hypothesis_testing import run_ga_stage
    return run_ga_stage(stage=0, persons=persons, **kw)


def run_ga_stage2(tours, persons, **kw):
    """Shim — delegates to ga_hypothesis_testing.run_ga_stage(stage=2)."""
    from ga_hypothesis_testing import run_ga_stage
    return run_ga_stage(stage=2, tours=tours, persons=persons, **kw)


def run_ga_stage3(tours, persons, **kw):
    """Shim — delegates to ga_hypothesis_testing.run_ga_stage(stage=3)."""
    from ga_hypothesis_testing import run_ga_stage
    return run_ga_stage(stage=3, tours=tours, persons=persons, **kw)


def run_ga_stage4(trips, tours, persons, **kw):
    """Shim — delegates to ga_hypothesis_testing.run_ga_stage(stage=4)."""
    from ga_hypothesis_testing import run_ga_stage
    return run_ga_stage(stage=4, trips=trips, tours=tours, persons=persons, **kw)


def run_ga_stage5(trips, persons, skims=None, landuse=None, **kw):
    """Shim — delegates to ga_hypothesis_testing.run_ga_stage(stage=5)."""
    from ga_hypothesis_testing import run_ga_stage
    return run_ga_stage(stage=5, trips=trips, persons=persons,
                        skims=skims, landuse=landuse, **kw)
