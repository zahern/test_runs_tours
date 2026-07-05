"""
synth_attributes.py
====================
Improved mapping from synthetic population (order_N.csv ABS Census columns)
to ABM model attributes, aligned with the QTS HHTS survey coding.

Why this matters
----------------
The current pipeline uses get_workstat/get_studying from data_connect.py
which work from QTS columns (hrsp, stup, typp). The synthetic population
has the same columns from the ABS Census but they encode the same
information differently. The mapping here ensures:

  1. WORKSTAT, STUDYING, ptype in the synth pop match what was estimated
     from the QTS HHTS (same categories, same numeric codes).
  2. New attributes (CARLICENCE, occp_group, lfsp_group) are added so
     Stage 5 destination/mode choice can use occupation-specific attractors.
  3. Person-level accessibility (home zone auto_work_acc etc.) is attached
     to the synth pop so Stage 1 and Stage 2 can use it as a feature.

Synth pop columns used
----------------------
agep / age5p  — age (agep = single year, age5p = 5-year band code 1-18)
sexp          — sex (1=male, 2=female)
vehrd         — household vehicles (0-5+)
nprd          — household size
incp          — income band (1-19)
hrsp          — hours worked per week (1-10 codes)
stup          — study status (1=not studying, 2-4=studying, 5=unknown)
typp          — study institution type (1=childcare...7=uni...10=not)
indp          — industry code (1-22, 22=not working)
occp          — occupation code (1-11, 11=not in labour force)
lfsp          — labour force status (1=ft emp, 2=pt emp, 3=unemp, 4=NILF, 5=student)
rlhp          — relationship to household head (1-5)
bikes         — bicycles in household
carlicence    — car licence holder (1=yes, 0=no)

QTS equivalent codes
---------------------
WORKSTAT: 0=not working, 1=fulltime, 2=parttime
STUDYING: 0=none, 1=primary, 2=secondary, 3=tertiary PT, 4=tertiary FT
ptype:    FullTimeWorker, PartTimeWorker, NonWorker, Student, School, Retired
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# WORKSTAT  (matches QTS work_map output: 0=not working, 1=FT, 2=PT)
# =============================================================================

def get_workstat(row) -> int:
    """
    Derive work status from synth pop hrsp (hours worked per week).

    hrsp codes:
        1  = Less than 1 hour / not applicable → not working
        2  = 1-15 hours  → part time
        3  = 16-24 hours → part time
        4  = 25-34 hours → part time
        5  = 35-39 hours → full time
        6  = 40 hours    → full time
        7  = 41-49 hours → full time
        8  = 50+ hours   → full time
        9  = Not stated
        10 = Overseas visitor / unknown
    """
    h = row.get('hrsp', 1)
    if h in (5, 6, 7, 8):
        return 1   # full time
    elif h in (2, 3, 4):
        return 2   # part time
    return 0       # not working


# =============================================================================
# STUDYING  (matches QTS study_map: 0=none,1=primary,2=secondary,3=PT,4=FT)
# =============================================================================

def get_studying(row) -> int:
    """
    Derive study status from synth pop stup + typp.

    stup: 1=not studying, 2=studying, 3=part-time, 4=full-time, 5=unknown
    typp: 1=childcare, 2=primary_gov, 3=primary_priv, 4=secondary_gov,
          5=secondary_priv, 6=TAFE, 7=university, 8=otherEdu, 9=other,
          10=notStudying, 11=unknown
    """
    stup = row.get('stup', 1)
    typp = row.get('typp', 10)

    if stup == 1 or typp == 10:
        return 0  # not studying

    if typp in (1, 2, 3):        # childcare / primary
        return 1
    elif typp in (4, 5):         # secondary
        return 2
    elif stup == 3 or typp == 6: # part-time tertiary / TAFE
        return 3
    elif stup == 4 or typp == 7: # full-time tertiary / university
        return 4

    return 0


# =============================================================================
# PTYPE  (ActivitySim person type for activity frequency segmentation)
# =============================================================================

PTYPE_LABELS = {
    1: 'FullTimeWorker',
    2: 'PartTimeWorker',
    3: 'NonWorker',
    4: 'Retired',
    5: 'UniversityStudent',  # tertiary — must match HHTS assign_ptype in data_connect.py
    6: 'SchoolChild',         # primary/secondary — must match HHTS assign_ptype
    7: 'Preschool',
    8: 'WorkingStudent',      # full-time work + study — must match HHTS assign_ptype
}

def get_ptype(row) -> int:
    """
    Derive person type from synth pop attributes.

    Priority order (matches ActivitySim and QTS assign_ptype):
      1. Preschool  (age5p=1, 0-4yr)
      2. School     (typp=2,3,4,5 — primary/secondary)
      3. Joint Work+Study
      4. Tertiary Student (typp=7, lfsp=5)
      5. Full-time Worker
      6. Part-time Worker
      7. Retired (lfsp=4 + older age)
      8. NonWorker (everyone else)
    """
    age5p = row.get('age5p', 6)
    typp  = row.get('typp', 10)
    hrsp  = row.get('hrsp', 1)
    lfsp  = row.get('lfsp', 4)

    # Preschool (0-4)
    if age5p == 1:
        return 7

    # Primary / Secondary school
    if typp in (2, 3, 4, 5):
        return 6

    # Full-time work AND tertiary study
    if hrsp in (5, 6, 7, 8) and typp in (6, 7, 8):
        return 8

    # Tertiary student (TAFE or uni, not primarily working)
    if typp in (6, 7) and hrsp in (1, 2, 3):
        return 5

    # Full-time worker
    if hrsp in (5, 6, 7, 8):
        return 1

    # Part-time worker
    if hrsp in (2, 3, 4):
        return 2

    # Retired: not in labour force + age 55+
    if lfsp == 4 and age5p >= 9:
        return 4

    return 3   # NonWorker


def assign_ptype_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'ptype' string column matching QTS person types."""
    df = df.copy()
    ptype_int = df.apply(get_ptype, axis=1)
    df['ptype'] = ptype_int.map(PTYPE_LABELS).fillna('NonWorker')
    return df


# =============================================================================
# OCCUPATION GROUP  (new — not in current pipeline)
# Maps synth occp to broad groups useful for destination choice
# =============================================================================

OCCP_GROUP = {
    1: 'white_collar',   # Manager
    2: 'white_collar',   # Professional
    3: 'skilled',        # Technician
    4: 'service',        # Community/Personal Service
    5: 'service',        # Sales
    6: 'clerical',       # Clerical/Administrative
    7: 'blue_collar',    # Machinery Operator
    8: 'blue_collar',    # Labourer
    9: 'other',
    10: 'other',
    11: 'not_working',
    12: 'unknown',
}

OCCP_GROUP_ID = {
    'white_collar': 1,
    'skilled':      2,
    'service':      3,
    'clerical':     4,
    'blue_collar':  5,
    'other':        6,
    'not_working':  0,
    'unknown':      0,
}

def get_occp_group(row) -> int:
    occp = row.get('occp', 11)
    label = OCCP_GROUP.get(int(occp), 'unknown')
    return OCCP_GROUP_ID.get(label, 0)


# =============================================================================
# CARLICENCE  (standardise to 0/1 int)
# =============================================================================

def get_carlicence(row) -> int:
    """
    Derive car licence from synth pop.
    Synth pop has 'carlicence' (1=yes, 0=no) already — just ensure int.
    Falls back to age-based heuristic if column missing.
    """
    cl = row.get('carlicence', None)
    if cl is not None:
        return int(cl)
    # fallback: assume licence if working age and has vehicle
    age5p = row.get('age5p', 1)
    vehrd = row.get('vehrd', 0)
    if age5p >= 5 and vehrd >= 1:   # age 20+ with household vehicle
        return 1
    return 0


# =============================================================================
# PERSON-LEVEL ACCESSIBILITY ATTACHMENT
# Attaches home zone accessibility to each person for use in Stage 1/2
# =============================================================================

def attach_person_accessibility(
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    acc_cols: list = None,
) -> pd.DataFrame:
    """
    Attach zone-level accessibility measures to each person based on
    their home zone (OTAZ / ORDER).

    After this call, persons has columns like:
        auto_work_acc, auto_nonmand_acc, walk_acc, transit_acc

    These are used by Stage 1 (activity frequency) and Stage 2 (day duration)
    as person-level features — people in more accessible zones make more trips.

    Parameters
    ----------
    persons  : synth pop DataFrame with OTAZ or ORDER column
    landuse  : enriched landuse with accessibility columns attached
    acc_cols : list of accessibility columns to attach (defaults to all acc_*cols)
    """
    if acc_cols is None:
        acc_cols = [c for c in landuse.columns if c.endswith('_acc')]

    if not acc_cols:
        logger.warning("attach_person_accessibility: no _acc columns found in landuse")
        return persons

    # find zone key
    zone_col = next((c for c in ['ORDER', 'OTAZ'] if c in landuse.columns), None)
    person_zone_col = next((c for c in ['OTAZ', 'ORDER'] if c in persons.columns), None)
    if zone_col is None or person_zone_col is None:
        logger.warning("attach_person_accessibility: could not find zone column")
        return persons

    lu_acc = landuse[[zone_col] + acc_cols].drop_duplicates(subset=[zone_col])
    lu_acc = lu_acc.rename(columns={zone_col: person_zone_col})

    persons = persons.copy()
    # drop existing acc cols to avoid duplication
    persons = persons.drop(columns=[c for c in acc_cols if c in persons.columns], errors='ignore')
    persons = persons.merge(lu_acc, on=person_zone_col, how='left')

    # fill missing zones with median
    for col in acc_cols:
        if col in persons.columns:
            med = persons[col].median()
            persons[col] = persons[col].fillna(med if not pd.isna(med) else 0.0)

    logger.info(f"Attached {len(acc_cols)} accessibility columns to {len(persons)} persons")
    return persons


# =============================================================================
# MAIN ENRICHMENT FUNCTION
# Called once on the synth pop at pipeline start
# =============================================================================

def enrich_synth_population(
    persons: pd.DataFrame,
    landuse: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Add all derived ABM attributes to the synthetic population.

    This replaces the per-row apply calls in pipeline_logger.py
    (get_workstat, get_studying, get_licence) with vectorised operations
    and adds new columns that improve model performance:
        WORKSTAT, STUDYING, CARLICENCE  — existing attributes, improved logic
        ptype                            — person type (new)
        occp_group                       — occupation group (new)
        auto_work_acc, walk_acc, ...     — home zone accessibility (new, if landuse passed)

    Parameters
    ----------
    persons  : raw synthetic population DataFrame (from order_N.csv)
    landuse  : enriched landuse with accessibility columns (optional)
    """
    df = persons.copy()

    # ── standardise person_id ────────────────────────────────────────────
    if 'person_id' not in df.columns:
        if 'abspid' in df.columns:
            df['person_id'] = df['abspid'].astype(str)
        elif 'PERSID' in df.columns:
            df['person_id'] = df['PERSID']

    # ── AGEGROUP — use age5p directly (5-year bands 1-18) ───────────────
    if 'age5p' in df.columns:
        df['AGEGROUP'] = df['age5p'].astype(int)
    elif 'agep' in df.columns:
        # map single-year agep to 5-year band
        df['AGEGROUP'] = (df['agep'].astype(int) // 5).clip(1, 18)

    # ── WORKSTAT (0=not working, 1=FT, 2=PT) ────────────────────────────
    df['WORKSTAT'] = df.apply(get_workstat, axis=1)

    # ── STUDYING (0=none, 1=primary, 2=secondary, 3=PT, 4=FT) ───────────
    df['STUDYING'] = df.apply(get_studying, axis=1)

    # ── CARLICENCE (0/1) ─────────────────────────────────────────────────
    if 'carlicence' in df.columns:
        df['CARLICENCE'] = df['carlicence'].astype(int)
    else:
        df['CARLICENCE'] = df.apply(get_carlicence, axis=1)

    # ── ptype ─────────────────────────────────────────────────────────────
    df = assign_ptype_column(df)

    # ── occp_group ────────────────────────────────────────────────────────
    if 'occp' in df.columns:
        df['occp_group'] = df.apply(get_occp_group, axis=1)

    # ── HHSIZE / household attributes ────────────────────────────────────
    if 'nprd' in df.columns:
        df['HHSIZE'] = df['nprd'].astype(int)

    # ── vehrd clip ────────────────────────────────────────────────────────
    if 'vehrd' in df.columns:
        df['vehrd'] = df['vehrd'].astype(int).clip(upper=4)

    # ── home zone accessibility ───────────────────────────────────────────
    if landuse is not None:
        acc_cols = [c for c in landuse.columns if c.endswith('_acc')]
        if acc_cols:
            df = attach_person_accessibility(df, landuse, acc_cols)

    logger.info(
        f"enrich_synth_population: {len(df)} persons | "
        f"WORKSTAT dist={df['WORKSTAT'].value_counts().to_dict()} | "
        f"STUDYING dist={df['STUDYING'].value_counts().to_dict()} | "
        f"ptype dist={df['ptype'].value_counts().to_dict()}"
    )
    return df
