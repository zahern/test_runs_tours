import pandas as pd
import larch as lx
import logging
#from pyspark.sql import functions as F
#from pyspark.sql.types import MapType, StringType, IntegerType
from addicty import Dict
import numpy as np
import random
import re
from pathlib import Path
import os

# Repo-relative default for the new_pop scope - a literal Z:\ drive path
# here broke every NEW_POP=1 run executed on the Linux HPC (Z: is only the
# Windows mount of /mnt/hpccs01/home/ahernz).
_TRT_ROOT_DC = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_DEFAULT_NEW_POP_DIR = os.path.join(_TRT_ROOT_DC, 'new_pop')
def time_hour(time):
    '''returns the time of starting..'''
    return int(time/60)
def time_hour_flat(time):
    return int(time/60)

class YourClass:

    @staticmethod
    def generate_alternatives(df, alt_method: str):

        if alt_method == "Mode":
            modes = pd.DataFrame({
                "mode": ["BUS", "CAR"],
                "mode_id": [1, 2]
            })

            # Cartesian product: every person × every mode
            df_out = df.merge(modes, how="cross")
            return df_out
        elif alt_method  == 'DEST_SAMPLE':
            det_sample = pd.DataFrame({
                'DTAZ': [np.random.choices(range(1,330),k=5)]
            })
            df_out = df.merge(det_sample, how = 'cross')

        # if you plan more methods later
        raise ValueError(f"Unknown alt_method: {alt_method}")

def survey_time_to_minutes(s: pd.Series) -> pd.Series:
    """
    Convert QTS STARTIME from 4am-based minutes to minutes from midnight.
    
    QTS travel day starts at 4:00am:
        0    = 4:00am  = 240 min from midnight
        60   = 5:00am  = 300 min from midnight
        480  = 12:00pm = 720 min from midnight
        840  = 2:00pm  = 960 min from midnight
        1200 = 8:00pm  = 1440 → wraps to next day
        1439 = 3:59am  = 1679 → clip to 1439 (end of day)
    
    TRAVTIME and DURATION are already in minutes — do NOT convert those.
    """
    s = pd.to_numeric(s, errors='coerce').fillna(0).astype(int)
    return (s + 240).clip(upper=1439)


# helper funcitons

def get_studying(row):
    if row['stup'] in [1, 5]:
        return 0
    elif row['typp'] in [1,3,2]: #PRIMARY
        return 1
    elif row['typp'] in [5,4]: #SECONDARY
        return 2
    elif row['typp'] == 7:
        if row['stup'] == 3: #PART TIME
            return 3
        elif row['stup'] == 4: #FULL TIME
            return 4
    # default fallback
    return 0


def safe_merge(left, right, on, how="left"):
    # find duplicate columns except join keys
    overlap = left.columns.intersection(right.columns).difference([on])

    if len(overlap) > 0:
        raise ValueError(f"Merge would create _x/_y columns. Overlapping columns: {list(overlap)}")

    return left.merge(right, on=on, how=how)

def clean_merge(left, right, on, how="left", validate=None):
    overlap = left.columns.intersection(right.columns).difference([on])

    if len(overlap) > 0:
        print("WARNING: dropping overlapping columns:", list(overlap))
        right = right.drop(columns=overlap)

    return left.merge(right, on=on, how=how, validate=validate)

def get_licence(row):
    """
    Return 1 if person is likely to hold a driver's licence, else 0.

    Heuristic: anyone aged 16+ in a vehicle-owning household is assumed
    to hold (or be eligible for) a licence. Children under 16 do not.
    Households without vehicles return 0 (no vehicle to licence for).
    """
    if row['vehrd'] >= 1:
        # Minimum driving age is 16; agep is in years
        return 1 if row['agep'] >= 16 else 0
    else:
        return 0

def get_workstat(row):
    '''
    "Less than 1 hour and Not applicable":1,
            "1-15 hours":2,
            "16-24 hours":3,
            "25-34 hours":4,
            "35-39 hours":5,
            "40 hours":6,
            "41-49 hours":7,
            "50 hours and over":8,
            "Not stated":9,
            "Overseas visitor":10}
    
    '''
    if row['hrsp'] in [5,6,7]: #NOT ATTENDING
        return 1
    elif row['hrsp'] in [2,3,4]: #FULL TIME
        return 2 # part time
    else:
        return 0

income_mapping = {'Negative income': 0,
'Nil or negative income':1,
'$1-$149 ($1-$7,799)':2,
'$150-$299 ($7,800-$15,599)':3,
'$300-$399 ($15,600-$20,799)':4,
'$400-$499 ($20,800-$25,999)':5,
'$500-$649 ($26,000-$33,799)':6,
'$650-$799 ($33,800-$41,599)':7,
'$800-$999 ($41,600-$51,999)':8,
'$1,000-$1,249 ($52,000-$64,999)':9,
'$1,250-$1,499 ($65,000-$77,999)':10,
'$1,500-$1,749 ($78,000-$90,999)':11,
'$1,750-$1,999 ($91,000-$103,999)':12,
'$2,000-$2,999 ($104,000-$155,999)':13,
'$3,000-$3,499 ($156,000-$181,999)':14,
'$3,500 or more ($182,000 or more)':15,
'Select One':17,
'Not applicable':17,
"Prefer not to say / can't say":16
}

# Income codes that represent NON-RESPONSE rather than a real income band.
_INCOME_NONRESPONSE = {16, 17}


def _impute_income(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Impute personal income band (incp, 1..15) for NON-RESPONSE records from the
    person's profile, so the income coefficient is estimable and applies
    sensibly to the population.

    In this HHTS ~88% of INCOME is 'Select One' / 'Not applicable' / 'Prefer not
    to say' → mapped to 16/17, which the income-band logic wrongly reads as high
    income (incp>=8). Left alone, every income band collapses (perfect
    correlation). We replace those non-response codes with a band drawn from a
    profile-conditioned distribution:

        expected band = base(WORKSTAT) + industry uplift + student penalty
        incp ~ round( Normal(expected, sd) ), clipped to [1, 15]

    Real responses (bands 1..15) are kept untouched. A per-person seed makes the
    draw reproducible. An 'incp_imputed' 0/1 flag marks which rows were filled.
    Disable via GA_IMPUTE_INCOME=0.
    """
    incp = pd.to_numeric(df.get('incp', pd.Series(17, index=df.index)),
                         errors='coerce')
    nonresp = incp.isin(_INCOME_NONRESPONSE) | incp.isna()
    n_nonresp = int(nonresp.sum())
    if n_nonresp == 0:
        df['incp_imputed'] = 0
        return df

    ws  = pd.to_numeric(df.get('WORKSTAT', 0), errors='coerce').fillna(0).values
    ind = pd.to_numeric(df.get('indp', 20),   errors='coerce').fillna(20).values

    # base expected band by work status: FT high, PT mid, not-working low
    mu = np.where(ws == 1, 9.0, np.where(ws == 2, 6.5, 4.0)).astype(float)
    # industry uplift: professional/finance/public/education → higher;
    # agriculture/manufacturing/utilities/construction → slightly lower
    mu += np.where(np.isin(ind, [10, 11, 12, 13, 14, 15, 16]), 2.0,
           np.where(np.isin(ind, [1, 2, 3, 4, 5, 6]), -0.5, 0.0))
    if 'STUDYING' in df.columns:
        st = pd.to_numeric(df['STUDYING'], errors='coerce').fillna(0).values
        mu += np.where(st > 0, -2.0, 0.0)   # students earn less

    # reproducible per-person draw (seed from a stable person id if available)
    _idcol = next((c for c in ('PERSID', 'person_id', 'PERSONID') if c in df.columns), None)
    if _idcol is not None:
        base = pd.util.hash_pandas_object(df[_idcol].astype(str), index=False).values
        rng_seed = (base % (2 ** 31)).astype(np.int64)
        noise = np.array([np.random.default_rng(int(s) + seed).normal(0.0, 2.5)
                          for s in rng_seed])
    else:
        noise = np.random.default_rng(seed).normal(0.0, 2.5, size=len(df))

    draw = np.clip(np.round(mu + noise), 1, 15).astype(int)
    out = incp.values.astype(float).copy()
    out[nonresp.values] = draw[nonresp.values]
    df = df.copy()
    df['incp'] = out.astype(int)
    df['incp_imputed'] = nonresp.astype(int).values
    try:
        logging.info(f"_impute_income: imputed {n_nonresp}/{len(df)} "
                     f"({100.0 * n_nonresp / max(len(df), 1):.1f}%) non-response "
                     f"income records from profile")
    except Exception:
        pass
    return df
    
    
study_map = {
    'other':0,
    'childcare':1,
    'primary':1,
             "secondary":2,
            "tertiaryPartTime":3,
             "tertiaryFullTime":4,
             
             }
sex_map = {'male':1,
           'female':2}


work_map = {'workFullTime':1,
            'workPartTime':2,
            'otherReason':3,
            'carer':4,
            'retired':5,
            'longTermIllness':6,
            'shortTermIllness':7,
            'volunteer':8,
            'unemployed':0,
            
                
    
}

industy_map = {
               'other':19,
               'publicAdmin':15,
               'health':17,
               'education':16,
                              'admin':14,
               'agriculture':1,
               'mining': 2,
               'manufacturing':3,
               'utility': 4,
               'construction': 5,
               'wholesale': 6,
                'retail':7,
               'accom':8,
               'transport':9,
                'it':10,
                'financial':11,
                'realEstate':12,
                'professional':13,
               'arts':18,
              "na":20
               }

def update_rlhp(group):
    """Apply logic at the household (HHID) level."""
    has_spouse = (group['RELATIONSHIP'] == 'spouse').any()
    has_head = (group['RELATIONSHIP'] == 'Head').any()
    has_child = (group['RELATIONSHIP'] == 'child').any()

    # Case 1: spouse exists and blank → assign 'Spouse'
    if has_spouse:
        group.loc[group['RELATIONSHIP'].isin(['', None]), 'RELATIONSHIP'] = 'spouse'

    # Case 2: no head but child present → assign 'Lone Parent' to blanks
    elif (not has_head) and has_child:
        group.loc[group['RELATIONSHIP'].isin(['', None]), 'RELATIONSHIP'] = 'lone parent'

    return group



rl_map = {'child':3,
                                                     'relative':4,
                                                     'spouse':2,
                                                     'nonFamily':5,
                                                     'head':1
}

###############################################################################
# BINARY DUMMY ENCODING
# Called after any person DataFrame is assembled (HHTS or synth pop).
# Adds proper binary indicator columns for marital status, household type,
# and industry sector so the GA can select them without ordinal bias.
###############################################################################

# ABS MSTP (marital status) codes used in the synthetic population
_MSTP_MARRIED   = {1}           # married (registered)
_MSTP_DEFACTO   = {2}           # de facto / common-law
_MSTP_SEPARATED = {3}           # separated
_MSTP_DIVORCED  = {4}           # divorced
_MSTP_WIDOWED   = {5}           # widowed
_MSTP_SINGLE    = {6}           # never married

# Industry (indp) groupings derived from industy_map
_IND_PROFESSIONAL  = {13}           # professional/scientific/technical
_IND_FINANCIAL     = {11, 12}       # financial services + real estate
_IND_IT            = {10}           # information & communication
_IND_HEALTH        = {17}           # health & social assistance
_IND_EDUCATION     = {16}           # education & training
_IND_PUBLIC_ADMIN  = {14, 15}       # public administration + administrative services
_IND_CONSTRUCTION  = {5}            # construction
_IND_MANUFACTURING = {3}            # manufacturing
_IND_BLUE_COLLAR   = {1, 2, 3, 4, 5}  # agriculture/mining/manufacturing/utilities/construction
_IND_RETAIL        = {6, 7}         # retail + wholesale trade
_IND_HOSPITALITY   = {8}            # accommodation & food
_IND_TRANSPORT     = {9}            # transport, postal, warehousing
_IND_ARTS_OTHER    = {18, 19}       # arts, recreation, other services
_IND_NOT_WORKING   = {20}           # not applicable / unknown


def add_person_binary_dummies(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Add binary dummy columns for marital status, household size type,
    and industry sector to a person DataFrame.

    Safe to call on both HHTS and synthetic-population DataFrames; each
    block only runs when the required source column is present.

    Columns added
    -------------
    Marital status (from 'mstp', ABS MSTP coding 1-6):
        is_married, is_defacto, is_separated, is_divorced,
        is_widowed, is_never_married, is_partnered

    Household size type (from 'nprd'):
        hh_1person, hh_couple, hh_small_group, hh_large_group

    Industry sector (from 'indp', derived from INDUSTRY text):
        ind_professional, ind_financial, ind_it,
        ind_health, ind_education, ind_public_admin,
        ind_construction, ind_manufacturing, ind_blue_collar,
        ind_retail, ind_hospitality, ind_transport,
        ind_arts_other, ind_not_working
    """
    import pandas as _pd
    import numpy as _np

    # -- Marital status -------------------------------------------------------
    if 'mstp' in df.columns:
        m = _pd.to_numeric(df['mstp'], errors='coerce').fillna(0).astype(int)
        df['is_married']      = m.isin(_MSTP_MARRIED).astype(int)
        df['is_defacto']      = m.isin(_MSTP_DEFACTO).astype(int)
        df['is_separated']    = m.isin(_MSTP_SEPARATED).astype(int)
        df['is_divorced']     = m.isin(_MSTP_DIVORCED).astype(int)
        df['is_widowed']      = m.isin(_MSTP_WIDOWED).astype(int)
        df['is_never_married']= m.isin(_MSTP_SINGLE).astype(int)
        # Convenience: currently partnered (married or de facto)
        df['is_partnered']    = m.isin(_MSTP_MARRIED | _MSTP_DEFACTO).astype(int)

    # -- Household size type (from nprd = household size count) ---------------
    if 'nprd' in df.columns:
        n = _pd.to_numeric(df['nprd'], errors='coerce').fillna(0).astype(int)
        df['hh_1person']    = (n == 1).astype(int)    # single-person
        df['hh_couple']     = (n == 2).astype(int)    # couple or lone-parent+1
        df['hh_small_group']= n.isin([3, 4]).astype(int)  # small family/group
        df['hh_large_group']= (n >= 5).astype(int)    # large family/houseshare

    # -- Industry sector (from indp numeric code) -----------------------------
    if 'indp' in df.columns:
        i = _pd.to_numeric(df['indp'], errors='coerce').fillna(20).astype(int)
        df['ind_professional'] = i.isin(_IND_PROFESSIONAL).astype(int)
        df['ind_financial']    = i.isin(_IND_FINANCIAL).astype(int)
        df['ind_it']           = i.isin(_IND_IT).astype(int)
        df['ind_health']       = i.isin(_IND_HEALTH).astype(int)
        df['ind_education']    = i.isin(_IND_EDUCATION).astype(int)
        df['ind_public_admin'] = i.isin(_IND_PUBLIC_ADMIN).astype(int)
        df['ind_construction'] = i.isin(_IND_CONSTRUCTION).astype(int)
        df['ind_manufacturing']= i.isin(_IND_MANUFACTURING).astype(int)
        df['ind_blue_collar']  = i.isin(_IND_BLUE_COLLAR).astype(int)
        df['ind_retail']       = i.isin(_IND_RETAIL).astype(int)
        df['ind_hospitality']  = i.isin(_IND_HOSPITALITY).astype(int)
        df['ind_transport']    = i.isin(_IND_TRANSPORT).astype(int)
        df['ind_arts_other']   = i.isin(_IND_ARTS_OTHER).astype(int)
        df['ind_not_working']  = i.isin(_IND_NOT_WORKING).astype(int)

    return df


###############################################################################
# HELPER FUNCTIONS -
###############################################################################
def count_stops(df):
    rows = []

    for (pid, tour_id), g in df[df["tour_id"] > 0].groupby(["person_id", "tour_id"]):

        if len(g) == 0:
            continue

        # find main tour destination (already computed)
        main_dest = g["primary_trip_for_tour"].iloc[0]

        # get the trip_id of the primary destination
        main_trip_id = int(main_dest)

        outbound = 0
        inbound  = 0    
        # activity-type counts
        outbound_by_act = {}
        inbound_by_act  = {}

        for _, r in g.iterrows():

            # skip trips finishing at home
            if r["dest_is_home"]:
                continue
            
            act = r["DESTPURP"]
            
            # OUTBOUND logic
            if r["trip_id"] <= main_trip_id:
                # skip the final outbound leg (arrival at main destination)
                #if r["trip_id"] == main_trip_id - 1:
                 #   continue
                outbound += 1
                outbound_by_act[act] = outbound_by_act.get(act, 0) + 1
                continue

            # INBOUND logic
            if r["trip_id"] > main_trip_id:
                inbound += 1
                inbound_by_act[act] = inbound_by_act.get(act, 0) + 1
                continue

        rows.append((pid, tour_id, outbound, inbound, outbound_by_act, inbound_by_act))

    df2 = pd.DataFrame(rows, columns=["person_id","tour_id","outbound_stops_adj","inbound_stops_adj", "outbound_by_activity", "inbound_by_activity"])
    out = pd.json_normalize(df2["outbound_by_activity"]).fillna(0).astype(int)
    out.columns = [f"n_{c}_out" for c in out.columns]

    # expand inbound dict
    inn = pd.json_normalize(df2["inbound_by_activity"]).fillna(0).astype(int)
    inn.columns = [f"n_{c}_in" for c in inn.columns]
    df_final = pd.concat([df2.drop(columns=["outbound_by_activity",
                                        "inbound_by_activity"]),
                      out, inn],
                     axis=1)
    return df_final


###############################################################################
# 0. COLUMN MAPPING — UPDATE ONLY THIS SECTION TO MATCH YOUR TRIPS FILE
###############################################################################

COLUMN_MAP = {
    "trip_id": "TRIPID",
    "person_id": "PERSONID",
    "origin": "ORIGIN_NAME",
    "destination": "DEST_NAME",
    "purpose": "PURPOSE_MAIN",
    "cumdist": "CUMDIST"
}
COLUMN_MAP_R = {
    "TRIPID":"trip_id",
    "PERSID":"person_id",
    "ORIGPLACE":"origin",
    "DESTPLACE":"destination",
    "OVERALL_PURPOSE":"purpose",
    "CUMDIST":"cumdist",
    "MAINMODE": "mode",
    
}


###############################################################################
# 1. LOAD + STANDARDIZE TRIPS
###############################################################################

def load_trips(df_trips):
    trips = df_trips.rename(columns=COLUMN_MAP_R).copy()
    trips = trips.sort_values(["person_id", "trip_id"]).reset_index(drop=True)

    trips = trips.reset_index(drop=True)
    return trips


###############################################################################
# 2. HOME DETECTION
###############################################################################

def is_home(x):
    """
    Detect if a place string represents the person's home.
    
    QTS data uses 'My home' as the ORIGPLACE/DESTPLACE value.
    'Go home' and 'At Home' appear in ORIGPURP/DESTPURP (different column).
    "Someone else's home" is NOT the person's own home.
    """
    if pd.isna(x):
        return False
    x = str(x).strip()
    # exact QTS place values that mean "this person's home"
    return x in ["My home", "my home", "My Home"]

def tag_home(df):
    df["origin_is_home"] = df["origin"].apply(is_home)
    df["dest_is_home"]   = df["destination"].apply(is_home)
    return df


###############################################################################
# 3. OUTBOUND / INBOUND LABELING (uses home + cumdist)
###############################################################################

def label_direction(df):
    df = df.sort_values(["person_id", "trip_id"]).copy()
    df["cumdist_prev"] = df.groupby("person_id")["cumdist"].shift(1)

    leaving_home = df["origin_is_home"]
    going_home   = df["dest_is_home"]

    dist_inc = df["cumdist"] > df["cumdist_prev"]
    dist_dec = df["cumdist"] < df["cumdist_prev"]

    df["direction"] = "outbound"
    df.loc[going_home, "direction"] = "inbound"
    df.loc[dist_dec,   "direction"] = "inbound"
    df.loc[leaving_home, "direction"] = "outbound"

    return df.drop(columns=["cumdist_prev"])


###############################################################################
# 4. ASSIGN HOME‑BASED TOURS
###############################################################################

def assign_tours(df):
    df = df.copy()
    df["tour_id"] = 0

    for pid, group in df.groupby("person_id"):
        tour = 0
        on_tour = False

        for idx, row in group.iterrows():

            if row["origin_is_home"] and row["direction"] == "outbound":
                tour += 1
                on_tour = True

            if on_tour:
                df.at[idx, "tour_id"] = tour

            if row["dest_is_home"] and row["direction"] == "inbound":
                on_tour = False

    return df


###############################################################################
# 5. PRIMARY PURPOSE ASSIGNMENT (ActivitySim hierarchy)
###############################################################################

PURPOSE_PRIORITY = {
    #"Work": 1,
    #"Primary or secondary school": 2,
    "Shopping": 5,
    "Recreation/leisure activity": 4,
    "Recreation": 4,
    "Pickup/Dropoff Someone": 15,
    "Pickup/Deliver Something": 15,
    "Childcare or Kindergarten": 90,
    "Childcare or kindergarten": 90,
    "Other Purpose": 99,
    "Direct Work Commute": 1,
    "Education": 2,
    "Social": 5, 
    'Accompany Someone': 15,
    "Personal Business": 9,
    "Work Related": 1,
    "Pick up or drop off a passenger":15,
    "Other": 99
}

def determine_primary_purpose(df):
    df["purpose_rank"] = df["purpose"].map(lambda x: PURPOSE_PRIORITY.get(x, 99))
    tours_only = df[df["tour_id"] > 0].copy()
    primary_idx = (
        tours_only
        .sort_values(["purpose_rank", "cumdist"], ascending=[True, False])
        .groupby(["tour_id", 'person_id'])
        .head(1)
    )
    primary_idx_h = (
        tours_only
        .sort_values(["origin_is_home"], ascending=[False])
        .groupby(["tour_id", 'person_id'])
        .head(1)
    )
    primary_idx = primary_idx.set_index(["person_id", "tour_id"])
    df_index = df.set_index(["person_id", "tour_id"])
    
    df_index["tour_primary_purpose"] = primary_idx["purpose"]
    df_index["tour_destination"]     = primary_idx["DTAZ"]
    df_index["primary_trip_for_tour"] = primary_idx["trip_id"]

    tour_origin_series = (
    df
    .sort_values(["person_id","tour_id","trip_id"])
    .groupby(["person_id","tour_id"])["OTAZ"]
    .first()
    )
    
    # Home anchor destination (your idx-h)
    df_index["tour_origin"] = tour_origin_series
    #df_index["home_anchor_trip"]     = primary_idx_h["trip_id"]
    
    
    # need to do by tour_id and person_id
    '''
    df["tour_primary_purpose"] = df["tour_id"].map(
        primary_idx.set_index("tour_id")["purpose"]
    )

    # map primary destination
    
    
    df["tour_destination"] = df["tour_id"].map(
        primary_idx.set_index(['person_id', "tour_id"])["DESTSA2_2021"]
    )

    # optional: keep which trip produced the primary purpose
    df["primary_trip_for_tour"] = df["tour_id"].map(
        primary_idx.set_index("tour_id")["trip_id"]
    )
    '''
    df = df_index.reset_index()

    return df


###############################################################################
# 6. SUBTOUR DETECTION (escort loops etc.)
###############################################################################

def assign_subtours(df):
    df = df.copy()
    df["subtour_id"] = 0

    for tour_id, group in df.groupby("tour_id"):

        anchor = None
        subtour = 0

        for idx, row in group.iterrows():

            if row["origin_is_home"]:
                anchor = None
                continue

            if anchor is None:
                anchor = row["origin"]

            # leaving anchor → new subtour
            if row["origin"] == anchor and not row["dest_is_home"]:
                subtour += 1

            df.at[idx, "subtour_id"] = subtour

            if row["destination"] == anchor:
                anchor = None

    return df


###############################################################################
# 7. BUILD ACTIVITYSIM TRIPS TABLE
###############################################################################

def build_as_trips(df):
    """
    Build ActivitySim-style trips table from QTS trips.
    
    QTS DURATION semantics (confirmed from data):
        DURATION = dwell time at destination (minutes)
        STARTIME + TRAVTIME + DURATION = next trip STARTIME  (exact chain)
    
    Therefore ACTIVITY_DURATION = DURATION directly.
    Do NOT subtract TRAVTIME — it is NOT included in DURATION.
    
    Negative DURATION (23 records) = data entry errors — clip to 0.
    """
    trips = df.copy()
    trips["depart"] = trips["STARTIME"]
    trips["arrive"] = trips["ARRTIME"]
    
    # DURATION is dwell time at destination — confirmed by time chain verification
    # clip negatives (23 recording errors in the QTS data)
    trips['DURATION'] = trips['DURATION'].fillna(0).clip(lower=0)
    trips['ACTIVITY_DURATION'] = trips['DURATION']
    
    # ENDTIME = arrival time (STARTIME + TRAVTIME)
    # ENDTIME_WITH_ACTIVITY = departure from destination (STARTIME + TRAVTIME + DURATION)
    trips['ENDTIME']              = trips['STARTIME'] + trips['TRAVTIME']
    trips['ENDTIME_WITH_ACTIVITY'] = trips['STARTIME'] + trips['TRAVTIME'] + trips['DURATION']
    
    return trips


###############################################################################
# 8. BUILD ACTIVITYSIM TOURS TABLE
###############################################################################

def _modal_mode(x):
    """Return the most common non-null mode value from a group. Robust to
    pandas passing scalars, Series, or single-column DataFrames."""
    if isinstance(x, str):
        return x
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if not hasattr(x, '__iter__'):
        return str(x) if x else 'DA'
    vals = pd.Series(x).dropna()
    if vals.empty:
        return 'DA'
    return vals.value_counts().index[0]


def build_as_tours(df):
    # Guard against duplicate columns (can happen if mode was set before load_trips rename)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    stops_fixed = count_stops(df)
    # Use ENDTIME_WITH_ACTIVITY for home-return so tour end = when person
    # departed the last non-home stop (not when they arrived home minus dwell)
    # ENDTIME_WITH_ACTIVITY = STARTIME + TRAVTIME + DURATION (departure from dest)
    et_col = 'ENDTIME_WITH_ACTIVITY' if 'ENDTIME_WITH_ACTIVITY' in df.columns else 'ENDTIME'
    home_returns = (
        df[df['dest_is_home'] == True]
        .groupby(['person_id', 'tour_id'])[et_col]
        .max()
        .reset_index(name='home_return_time')
    )

    tours = (
        df[df["tour_id"] > 0]
        .groupby(["person_id", "tour_id", 'HHID'])
        .agg(
            primary_purpose   = ("tour_primary_purpose", "first"),
            tour_origin       = ("tour_origin", "first"),
            tour_destination  = ("tour_destination", "first"),
            start_cumdist     = ("cumdist", "min"),
            start_time        = ('STARTIME', 'min'),
            end_time          = (et_col, 'max'),   # departure from last stop
            end_cumdist       = ("cumdist", "max"),
            num_trips         = ("trip_id", "count"),
            outbound_stops    = ("direction", lambda x: (x=="outbound").sum()),
            inbound_stops     = ("direction", lambda x: (x=="inbound").sum()),
            tour_mode         = ('mode', _modal_mode),
        )
        .reset_index()
    )

    # replace end_time with home-return time where available
    tours = tours.merge(home_returns, on=['person_id', 'tour_id'], how='left')
    tours['end_time'] = tours['home_return_time'].fillna(tours['end_time'])
    tours = tours.drop(columns=['home_return_time'])

    tours['duration'] = tours['end_time'] - tours['start_time']
    # Keep STARTIME and ENDTIME in MINUTES FROM MIDNIGHT (not hours)
    # Downstream stages (2,3,4) all expect minutes
    tours['STARTIME'] = tours['start_time'].astype(int)
    tours['ENDTIME']  = tours['end_time'].astype(int)
    
    tours = tours.merge(stops_fixed, on=["person_id","tour_id"], how="left")
    
    tours["tour_distance"] = tours["end_cumdist"] - tours["start_cumdist"]
    tours["tour_start"] = tours["tour_id"] * 100
    tours["tour_end"]   = tours["tour_start"] + 60
    tours["stop_frequency"] = (
    tours["outbound_stops_adj"].clip(upper=3).astype(str)
    + "out_"
    + tours["inbound_stops_adj"].clip(upper=3).astype(str)
    + "in"
    )
    '''
    tours["stop_frequency"] = (
    tours["outbound_stops"].clip(upper=3).astype(int).astype(str) + "out_" +
    tours["inbound_stops"].clip(upper=3).astype(int).astype(str) + "in"
    )'''

    return tours


###############################################################################
# 9. FULL PIPELINE FUNCTION
###############################################################################

def build_tours_from_trips(trips):
    trips = load_trips(trips)
    trips = tag_home(trips)
    trips = trim_to_home_sequence(trips)
    trips = label_direction(trips)
    trips = assign_tours(trips)
    trips = determine_primary_purpose(trips)
    trips = assign_subtours(trips)

    as_trips = build_as_trips(trips)
    as_tours = build_as_tours(trips)

    return as_trips, as_tours


def trim_to_home_sequence(df):
    df = df.copy()
    df["origin_is_home"] = df["origin"].apply(is_home)
    df["dest_is_home"]   = df["destination"].apply(is_home)

    cleaned = []

    for pid, g in df.groupby("person_id"):
        g = g.sort_values("trip_id")

        # first home origin
        origins = g.index[g["origin_is_home"]].tolist()
        if not origins:
            continue  # drop whole person

        first_home_idx = origins[0]

        # last home destination
        dests = g.index[g["dest_is_home"]].tolist()
        if not dests:
            continue  # drop whole person

        last_home_idx = dests[-1]

        g_clean = g.loc[first_home_idx:last_home_idx]
        cleaned.append(g_clean)

    # Recombine all people
    out = pd.concat(cleaned, ignore_index=True)
    return out



def assign_ptype(person):
    """
    Assign ActivitySim-style person type from QTS HHTS columns.

    QTS AGEGROUP codes (numeric):  1=0-4, 2=5-9, 3=10-14, 4=15-19,
                                    5=20-24, ... (5-year bands up to 21=100+)
    QTS WORKSTATUS values:  'workFullTime', 'workPartTime', 'retired',
                             'otherReason', 'carer', 'volunteer',
                             'longTermIllness', 'shortTermIllness'
    QTS MAINACT values:     'Full-time Work', 'Part-time Work', 'Retired',
                             'Primary School', 'Secondary School',
                             'Childcare/Kindergarten', 'Full-time TAFE/Uni',
                             'Part-time TAFE/Uni', 'Other Education',
                             'Volunteer/Carer', 'Illness', 'Other Activity'

    Matches PTYPE_COLS in stage1_activity_frequency.py so HHTS and synth
    population use consistent ptype labels:
        'FullTimeWorker', 'PartTimeWorker', 'SchoolChild', 'UniversityStudent',
        'WorkingStudent', 'Retired', 'Preschool', 'NonWorker'
    """
    age  = person["AGEGROUP"]   # int code 1-21
    work = person["WORKSTATUS"] # string e.g. 'workFullTime'
    act  = person["MAINACT"]    # string e.g. 'Full-time Work'

    # 1. Preschool (AGEGROUP=1 = 0-4 years)
    if age == 1 or act == "Childcare/Kindergarten":
        return "Preschool"

    # 2. School children (AGEGROUP 2-4 = 5-19 years, or school activity)
    if age in [2, 3, 4] or act in ["Primary School", "Secondary School"]:
        return "SchoolChild"

    # 3. Retired
    if work == "retired" or act == "Retired":
        return "Retired"

    # 4. Joint study + work
    if work in ["workFullTime", "workPartTime"] and act in [
        "Full-time TAFE/Uni", "Part-time TAFE/Uni", "Other Education"
    ]:
        return "WorkingStudent"

    # 5. Students
    if act in ["Full-time TAFE/Uni", "Part-time TAFE/Uni", "Other Education"]:
        return "UniversityStudent"

    # 6. Full-time workers
    if work == "workFullTime" or act == "Full-time Work":
        return "FullTimeWorker"

    # 7. Part-time workers
    if work == "workPartTime" or act == "Part-time Work":
        return "PartTimeWorker"

    # 8. Catch-all non-worker
    return "NonWorker"




purpose = {
    # Work
    'Work (my workplace)':                 'work',
    'Work (other work reason)':            'work',
    'Direct Work Commute':                 'work',
    'Work Related':                        'work',
    'My workplace':                        'work',
    'Other work-related place':            'work',
    # Education
    'Education':                           'edu',
    'Childcare or Kindergarten':           'edu',
    'Childcare or kindergarten':           'edu',
    'Primary or secondary school':         'edu',
    'My campus':                           'edu',
    'Other study-related place':           'edu',
    # Escort
    'Pick up or drop off a passenger':     'escort',
    'Pickup/Dropoff Someone':              'escort',
    'Pickup/Deliver Something':            'escort',
    'Just accompanying someone':           'escort',
    'Accompanying Someone':                'escort',
    'Accompany Someone':                   'escort',
    # Shopping
    'Shopping':                            'shopping',
    'Shop or supermarket':                 'shopping',
    # Eat
    'Eat/drink':                           'eat',
    'Restaurant, café, or food outlet':   'eat',
    # Social
    'Social Visit':                        'social',
    'Social':                              'social',
    'Recreation/leisure activity':         'social',
    'Recreation':                          'social',
    'Reacreation':                         'social',
    "Someone else's home":                 'social',
    # Othmaint
    'Personal Business':                   'othmaint',
    'Personal business':                   'othmaint',
    'Pickup/deliver something (not work-related)': 'othmaint',
    'Car park':                            'othmaint',
    'Petrol station':                      'othdiscr',
    # Escort
    'Pick up or drop off a passenger':     'escort',
    # Othdiscr
    'Other place (please specify)':        'othdiscr',
    'Other Place':                         'othdiscr',
    'Refuel my vehicle':                   'othdiscr',
    'Other Purpose':                       'othdiscr',
    'Pickup/Deliver Something':            'othmaint',
    # Home
    'At Home':                             'home',
    'Go home':                             'home',
    'My home':                             'home',
}
MODES = {
            'Car driver': 'DA',
            'Walking': 'Walk',
            'Car passenger': 'SR',
            'Public Bus': 'Transit',
            'Motorcycle driver': 'DA',
            'Truck passenger': 'SR',
            'Charter/Courtesy/Other bus':'Transit',
            'Light rail':'Transit',
            'Public bus':'Transit',
            'School bus (with route number)': 'Transit',
            'E-bike or E-scooter': 'Bike',
            'Truck driver':'DA',
            'Uber / Other Ride Share':'SR',
            'Train': 'Transit',
            'Other method':'DA',
            'Motorcycle passenger':'SR',
            'Taxi':'SR',
            'Bicycle':'Bike',
            'Mobility scooter':'Walk',
            'Ferry':'Transit',
            'School bus (private/chartered)': 'Transit'
        }

class Synth_Connect:
    
    
    
    
    
    
    
    def __init__(self, new_pop=False):
        # new_pop=True switches to the 507-SA2-zone scope (Z:/test_runs_tours/new_pop):
        # OTAZ/DTAZ/ORDER are the SA2 code itself (no ORDER indirection), landuse comes
        # from zone_landuse_507.csv, and OSM size-term columns come from
        # SA2_2016_OSM_counts_507.csv. See export_zone_sa2_files() in
        # synth_pop/run_pipeline.py for how the population side is keyed the same way.
        self.new_pop = new_pop
        self.data = None
        self.trips = None
        self.land = None
        self.tours = None
        
    
    def _clean_merge(self, left, right, on, how="left", validate=None):
        """
        Safe merge that prevents accidental _x/_y columns.
        Drops overlapping columns from right dataframe (except join keys).
        """

        # allow single key or list of keys
        if isinstance(on, str):
            join_keys = [on]
        else:
            join_keys = on

        overlap = left.columns.intersection(right.columns)
        overlap = overlap.difference(join_keys)

        if len(overlap) > 0:
            logging.warning(
                f"Dropping overlapping columns before merge: {list(overlap)}"
            )
            right = right.drop(columns=overlap)

        return left.merge(
            right,
            on=on,
            how=how,
            validate=validate
        )
    
    
    def update_files(self):
        '''Helper Function for rewriteing as trips tours'''
        self.trips_to_tours()
    
    
    def persons_to_active(self):
        person = pd.read_excel('../2_QTS_PERSONS.xlsx')
        household = pd.read_excel('../1_QTS_HOUSEHOLDS.xlsx')
        persons = person.merge(household, on='HHID', how='left')
        person = self.process_hhts_person(persons)
        sa1_to_sa2data = pd.read_csv('../SA1_SHP/sa1_2021.csv')
        sa1_to_sa2 = sa1_to_sa2data.set_index('SA1_CODE21')['SA2_CODE21'].astype(int).to_dict()
        person['HOME_SA2_2021'] = person['HOME_SA1_2021'].map(sa1_to_sa2)
        
        
        Main_ACT = {
            'Full-time Work': 'M',
            'Retired' : 'N',
            'Part-time Work': 'M',
            'Volunteer/Carer': 'N',
            'Other Activity': 'N',
            'Illness': 'N',
            'Secondary School': 'M',
            'Primary School': 'M',
            'Childcare/Kindergarten': 'M',
            'Full-time TAFE/Uni': 'M',
            'Part-time TAFE/Uni': 'M',
            'Other Education': 'M'
            

            
        }
        person["ACTC"] = "N"  # default

        person.loc[~person["ANYSTOPS"], "ACTC"] = "H"
        person.loc[person["ANYSTOPS"], "ACTC"] = person.loc[person["ANYSTOPS"], "MAINACT"].map(Main_ACT)
        
        #person = self.process_hhts_person(person)
        person = person.rename(columns={'PERSID':'person_id'})
            
        # I WANT TO MAKE A PTYPE 
        # based on AGREGRROUP, SEX, RELATIONSHIP, WORKSTATUS, MAINACT, WFHDAY, WFHOMEWEEKDAY
        
        
        return person
    
    
    def process_survey(self, save=True, output_path="saved_data"):
        """
        Process QTS survey data and return structured results.

        Parameters
        ----------
        save : bool
            Whether to save processed outputs to disk.
        output_path : str
            Directory where processed CSV files will be saved.

        Returns
        -------
        dict with keys: "tours", "trips", "persons"
        """

        # -------------------------------
        # LOAD DATA
        # -------------------------------
        person = pd.read_excel('../2_QTS_PERSONS.xlsx')
        household = pd.read_excel('../1_QTS_HOUSEHOLDS.xlsx')
        persons = person.merge(household, on='HHID', how='left')
        persons = self.process_hhts_person(persons)

        # align person_id column name
        if 'person_id' not in persons.columns and 'PERSID' in persons.columns:
            persons = persons.rename(columns={'PERSID': 'person_id'})

        trips = pd.read_excel('../5_QTS_TRIPS.xlsx')
        trips['STARTIME'] = survey_time_to_minutes(trips['STARTIME'])
        if 'ARRTIME' in trips.columns:
            trips['ARRTIME'] = survey_time_to_minutes(trips['ARRTIME'])
        trips['ENDTIME']  = trips['STARTIME'] + trips['TRAVTIME']
        

        # FIX 5: only replace negative rows, not all rows
        trips['DURATION'] = trips['DURATION'].fillna(trips['TRAVTIME'])
        trips.loc[trips['DURATION'] < 0, 'DURATION'] = trips.loc[
            trips['DURATION'] < 0, 'TRAVTIME'
        ]

        # -------------------------------
        # SA2 MAPPING
        # -------------------------------
        sa1_to_sa2data = pd.read_csv('../SA1_SHP/sa1_2021.csv')
        sa1_to_sa2 = (
            sa1_to_sa2data
            .set_index('SA1_CODE21')['SA2_CODE21']
            .astype(int)
            .to_dict()
        )

        trips['DESTSA2_2021'] = trips['DESTSA1_2021'].map(sa1_to_sa2)
        trips['ORIGSA2_2021'] = trips['ORIGSA1_2021'].map(sa1_to_sa2)
        persons['HOME_SA2_2021'] = persons['HOME_SA1_2021'].map(sa1_to_sa2)

        # Preserve SA1 code as HOME_SA1 (ABS sub-zone, fallback for Level4).
        if 'HOME_SA1_2021' in persons.columns:
            import numpy as _np
            persons['HOME_SA1'] = (
                pd.to_numeric(persons['HOME_SA1_2021'], errors='coerce')
                .fillna(0).astype(_np.int64)
            )

        # Assign HOME_LVL4 (SEQ-STM Level4 micro-zone) to HHTS respondents so
        # the walk/bike mode choice is TRAINED at the same fine-grain resolution
        # it is PREDICTED — avoiding a training/prediction LOS mismatch.
        # Strategy: sample uniformly from Level4 zones within each respondent's
        # home SA2 (the same approach as assign_lvl4_to_population.py), which
        # is consistent and produces the same within-SA2 distribution in training
        # and prediction even without knowing the respondent's exact Level4 zone.
        import os as _os
        _lvl4_xwalk_path = _os.path.join(
            (_os.environ.get('NEW_POP_DIR') or _DEFAULT_NEW_POP_DIR),
            '..', 'demand_', 'lvl4_to_emme_crosswalk.csv'
        )
        _lvl4_xwalk_path = _os.path.normpath(_lvl4_xwalk_path)
        if _os.path.exists(_lvl4_xwalk_path):
            import numpy as _np
            _lxw = pd.read_csv(_lvl4_xwalk_path)
            _lxw['sa2_code'] = pd.to_numeric(_lxw['sa2_code'], errors='coerce')
            _lxw = _lxw.dropna(subset=['sa2_code'])
            _lxw['sa2_code'] = _lxw['sa2_code'].astype(int)
            _sa2_to_lvl4s = _lxw.groupby('sa2_code')['LvL4_ID'].apply(list).to_dict()

            def _sample_lvl4(sa2):
                opts = _sa2_to_lvl4s.get(int(sa2) if not pd.isna(sa2) else -1)
                if opts:
                    return int(_np.random.choice(opts))
                return 0

            persons['HOME_LVL4'] = persons['HOME_SA2_2021'].apply(
                lambda s: _sample_lvl4(pd.to_numeric(s, errors='coerce') if s else -1)
            ).astype(_np.int64)

        if self.new_pop:
            # 507-SA2-zone scope: OTAZ/DTAZ are the SA2 code directly, no ORDER
            # lookup. Deliberately reuse 330 as the external/catch-all sentinel
            # (not e.g. 999999999): SA2 codes are always 9-digit (>100,000,000),
            # so 330 can never collide with a real new_pop zone either, and
            # every existing "!= 330" catch-all check already scattered through
            # stage5_destination_choice.py / stage6_plan_finalisation.py keeps
            # working unchanged in new_pop mode.
            import os as _os
            EXTERNAL_TAZ = 330
            # Valid-zone universe: the UNION of the operative crosswalk list
            # (sa2_crosswalk_507.csv, 324 zones) and the full-SEQ reference
            # (sa2_zones_reference_357.csv, 371 zones after the 2026-07
            # extension). Validating against the crosswalk alone wrongly
            # externalised every trip touching the 47 newly-added zones
            # (Jimboomba, Toowoomba SA4 317, NSW Richmond Valley, ...).
            _demand_dir = _os.path.normpath(_os.path.join(
                (_os.environ.get('NEW_POP_DIR') or _DEFAULT_NEW_POP_DIR),
                '..', 'demand_'))
            _valid_zones = set()
            for _zl_name, _zl_col in [('sa2_crosswalk_507.csv', 'sa2_code'),
                                       ('sa2_zones_reference_357.csv', 'sa2_code')]:
                _zl_path = _os.path.join(_demand_dir, _zl_name)
                if _os.path.exists(_zl_path):
                    _valid_zones |= set(
                        pd.to_numeric(pd.read_csv(_zl_path)[_zl_col],
                                      errors='coerce').dropna().astype(int))
            if not _valid_zones:
                _valid_zones = None

            def _to_otaz(sa2_series):
                sa2 = pd.to_numeric(sa2_series, errors='coerce')
                if _valid_zones is not None:
                    sa2 = sa2.where(sa2.isin(_valid_zones))
                return sa2.fillna(EXTERNAL_TAZ).astype(int)

            trips['OTAZ'] = _to_otaz(trips['ORIGSA2_2021'])
            trips['DTAZ'] = _to_otaz(trips['DESTSA2_2021'])
            persons['OTAZ'] = _to_otaz(persons['HOME_SA2_2021'])

            _n_external = int((trips['OTAZ'] == EXTERNAL_TAZ).sum() + (trips['DTAZ'] == EXTERNAL_TAZ).sum())
            if _n_external:
                # name the genuinely-external codes so a coverage regression
                # (like the 324-vs-371 one) is visible at a glance instead of
                # hiding behind an aggregate count
                _ext_codes = pd.concat([
                    pd.to_numeric(trips['ORIGSA2_2021'], errors='coerce'),
                    pd.to_numeric(trips['DESTSA2_2021'], errors='coerce'),
                ]).dropna().astype(int)
                _ext_codes = sorted(set(_ext_codes[~_ext_codes.isin(_valid_zones)])) \
                    if _valid_zones else []
                logging.warning(
                    f"process_survey (new_pop): {_n_external} trip endpoints outside "
                    f"the {len(_valid_zones or [])}-zone SEQ universe -> external "
                    f"sentinel {EXTERNAL_TAZ} (excluded from zone choice). "
                    f"{len(_ext_codes)} distinct external SA2s, first 10: {_ext_codes[:10]}"
                )
        else:
            # FIX 1: TAZ zone lookup (was missing entirely from process_survey)
            sa2_to_order = pd.read_csv('seq_zone_ids_.csv')[['SA2_MAIN16', 'ORDER']]
            o = sa2_to_order.set_index('SA2_MAIN16')['ORDER']
            trips['OTAZ'] = trips['ORIGSA2_2021'].map(o).fillna(330).astype(int)
            trips['DTAZ'] = trips['DESTSA2_2021'].map(o).fillna(330).astype(int)
            persons['OTAZ'] = persons['HOME_SA2_2021'].map(o).fillna(330).astype(int)

            _unmapped_o = trips['ORIGSA2_2021'].dropna().astype(int)
            _unmapped_origs = set(_unmapped_o[_unmapped_o.map(o).isna()])
            _unmapped_d = trips['DESTSA2_2021'].dropna().astype(int)
            _unmapped_dests = set(_unmapped_d[_unmapped_d.map(o).isna()])
            _unmapped = _unmapped_origs | _unmapped_dests
            if _unmapped:
                logging.warning(
                    f"process_survey: {len(_unmapped)} SA2 codes not in zone mapping "
                    f"(defaulted to TAZ 330): {sorted(_unmapped)[:10]}"
                )

        # -------------------------------
        # MODE & PURPOSE CLEANING
        # -------------------------------
        # Save raw MAINMODE before mapping for VEHOCC-based DA→SR reclassification
        _raw_mainmode = trips['MAINMODE'].copy()

        trips['MAINMODE'] = trips['MAINMODE'].map(MODES)
        # NOTE: do NOT set trips['mode'] here — load_trips() inside
        # build_tours_from_trips() renames MAINMODE→mode via COLUMN_MAP_R.
        # Pre-setting 'mode' here causes a duplicate column that crashes agg().
        trips['MODE1']    = trips['MODE1'].map(MODES)

        # Reclassify Car/Truck/Motorcycle driver → SR when vehicle had
        # multiple occupants (VEHOCC > 1).  VEHOCC is in the stops file;
        # aggregate max VEHOCC per trip and merge.
        _driver_modes = {'Car driver', 'Truck driver', 'Motorcycle driver'}
        _is_driver = _raw_mainmode.isin(_driver_modes)
        if _is_driver.any():
            try:
                _stops_path = Path('../4_QTS_STOPS.xlsx')
                if _stops_path.exists():
                    _stops_occ = pd.read_excel(
                        _stops_path,
                        usecols=['TRIPID', 'VEHOCC'],
                    )
                    _stops_occ['VEHOCC'] = pd.to_numeric(
                        _stops_occ['VEHOCC'], errors='coerce'
                    ).fillna(0)
                    _trip_occ = (
                        _stops_occ.groupby('TRIPID')['VEHOCC']
                        .max()
                        .reset_index()
                    )
                    trips = trips.merge(_trip_occ, on='TRIPID', how='left')
                    trips['VEHOCC'] = trips['VEHOCC'].fillna(0)
                    _reclassify = _is_driver & (trips['VEHOCC'] > 1)
                    _n_reclass = _reclassify.sum()
                    if _n_reclass > 0:
                        trips.loc[_reclassify, 'MAINMODE'] = 'SR'
                        logging.info(
                            f"DA→SR reclassification: {_n_reclass} driver trips "
                            f"with VEHOCC>1 reclassified from DA to SR"
                        )
                else:
                    logging.warning(
                        "4_QTS_STOPS.xlsx not found — skipping VEHOCC-based "
                        "DA→SR reclassification"
                    )
            except Exception as _vocc_e:
                logging.warning(f"VEHOCC reclassification failed: {_vocc_e}")

        trips['ORIGPURP'] = trips['ORIGPURP'].map(purpose)
        trips['DESTPURP'] = trips['DESTPURP'].map(purpose)

        # FIX 2: use TRAVTIME not DURATION for end time
        # use actual recorded arrival time if available
        if 'ARRTIME' in trips.columns:
            trips["ENDTIME"] = trips["ARRTIME"]
        else:
            trips["ENDTIME"] = trips["STARTIME"] + trips["TRAVTIME"]
        trips["ENDTIME_WITH_ACTIVITY"] = trips["STARTIME"] + trips["TRAVTIME"] + trips["DURATION"].fillna(0)

        # -------------------------------
        # BUILD TOURS
        # -------------------------------
        as_trips, as_tours = build_tours_from_trips(trips)

        as_tours['main_purpose'] = as_tours['primary_purpose']  # no .map — already coded
        as_trips['purpose'] = as_trips['purpose'].map(purpose)

        # FIX 4: STARTSTOP doesn't exist — build it, then compute max_stops
        as_trips['STARTSTOP'] = (
            as_trips
            .groupby(['person_id', 'tour_id'])
            .cumcount() + 1
        )
        as_trips['max_stops'] = (
            as_trips
            .groupby(['person_id', 'tour_id'])['STARTSTOP']
            .transform('max')
        )

        # FIX 3: merge person demographics onto tours
        # CARLICENCE carried through (process_hhts_person builds it, but it was
        # being dropped here) so the Stage 5 mode model can gate DA availability
        # (licence AND vehicle) instead of falling back to always-available —
        # that was the "availability vars missing [CARLICENCE, vehrd, ...]"
        # warning. Guarded below, so it's skipped if absent from persons.
        person_cols = ['person_id', 'sexp', 'vehrd', 'nprd', 'OTAZ', 'HOME_SA2_2021',
                       'HHID', 'CARLICENCE']
        person_cols = [c for c in person_cols if c in persons.columns]
        as_tours = as_tours.merge(persons[person_cols], on='person_id', how='left')

        # -------------------------------
        # SAVE OUTPUTS
        # -------------------------------
        if save:
            os.makedirs(output_path, exist_ok=True)
            as_trips.to_csv(f"{output_path}/trips_processed.csv", index=False)
            as_tours.to_csv(f"{output_path}/tours_processed.csv", index=False)
            persons.to_csv(f"{output_path}/persons_processed.csv", index=False)

        # -------------------------------
        # RETURN STRUCTURED OUTPUT
        # -------------------------------
        return {
            "tours": as_tours,
            "trips": as_trips,
            "persons": persons
        }
    
    
    def trips_to_tours(self, and_trips = False, and_persons = False):
        ## SETUP 
        '''
        Docstring for trips_to_tours
        
        :param self: Description
        :param and_trips: Description
        :param and_persons: Description
        '''
        person = pd.read_excel('../2_QTS_PERSONS.xlsx')
        household = pd.read_excel('../1_QTS_HOUSEHOLDS.xlsx')
        persons = person.merge(household, on='HHID', how='left')
        person = self.process_hhts_person(persons)

        trips = pd.read_excel('../5_QTS_TRIPS.xlsx')
        trips['STARTIME'] = survey_time_to_minutes(trips['STARTIME'])
        if 'ARRTIME' in trips.columns:
            trips['ARRTIME'] = survey_time_to_minutes(trips['ARRTIME'])
        trips['ENDTIME'] = trips['STARTIME'] + trips['TRAVTIME']
        
        trips['DURATION'] = trips['DURATION'].fillna(trips['TRAVTIME'])
        trips.loc[trips['DURATION'] < 0, 'DURATION'] = trips.loc[
             trips['DURATION'] < 0, 'TRAVTIME'
                    ]
        
        
       
        # if trips dont start at home delete the first records until they start at home
        
        # if trips font finish at home, delete teh last records until they finish at home
        
        #person_id_h = person[['PERSID', 'HHID']]
        #trips = trips.merge(person_id_h, on = 'PERSID' )

        # SA2 mapping
        sa1_to_sa2data = pd.read_csv('../SA1_SHP/sa1_2021.csv')
        sa1_to_sa2 = sa1_to_sa2data.set_index('SA1_CODE21')['SA2_CODE21'].astype(int).to_dict()

        trips['DESTSA2_2021'] = trips['DESTSA1_2021'].map(sa1_to_sa2)
        trips['ORIGSA2_2021'] = trips['ORIGSA1_2021'].map(sa1_to_sa2)
        #trips['OTAZ']
        #trips['DTAZ']
        
        sa2_to_order = pd.read_csv('seq_zone_ids_.csv')[['SA2_MAIN16', 'ORDER']]
        o = sa2_to_order.set_index('SA2_MAIN16')['ORDER']
        
        trips['OTAZ'] = trips['ORIGSA2_2021'].apply(lambda x: o.get(x)).fillna(330)
        trips['DTAZ'] = trips['DESTSA2_2021'].apply(lambda x: o.get(x)).fillna(330)
        person['HOME_SA2_2021'] = person['HOME_SA1_2021'].map(sa1_to_sa2)

        # -----------------------------------------
        # PURPOSE & MODE CLEANING
        # -----------------------------------------
        purpose = {
            'Pick up or drop off a passenger': 'escort',
            'Work (my workplace)': 'work',
            'Work (other work reason)': 'work',
            'Other place (please specify)': 'othdiscr',
            'At Home': 'home',
            'Go home': 'home',
            'Childcare or Kindergarten':'edu',
            'Childcare or kindergarten':'edu',
            'Other Place':'othdiscr',
            'Refuel my vehicle':'othdiscr',
            'Education':'edu',
            'Pickup/deliver something (not work-related)':'othmaint',
            'Social Visit':'social',
            'Recreation/leisure activity':'social',
            'Recreation': 'social',
            'Personal Business': "othmaint",
            'Pickup/Dropoff Someone': 'escort',
            'Pickup/Deliver Something': 'escort',
            'Other Purpose': 'othdiscr',
            'Accompanying Someone': 'social',
            'Accompany Someone': 'social',
            'Direct Work Commute': 'work',
            
            'Social': 'social',
            'Work Related': 'work',
            
            'Eat/drink':'eat',
            'Personal business':'othmaint',
            'Just accompanying someone':'escort',
            'Pick up or drop off a passenger':'escort',
            'Shopping':'shopping',
            'Pickup/Deliver Something': 'othmaint',
        }

        MODES = {
            'Car driver': 'DA',
            'Walking': 'Walk',
            'Car passenger': 'SR',
            'Public Bus': 'Transit',
            'Motorcycle driver': 'DA',
            'Truck passenger': 'SR',
            'Charter/Courtesy/Other bus':'Transit',
            'Light rail':'Transit',
            'Public bus':'Transit',
            'School bus (with route number)': 'Transit',
            'E-bike or E-scooter': 'Bike',
            'Truck driver':'DA',
            'Uber / Other Ride Share':'SR',
            'Train': 'Transit',
            'Other method':'DA',
            'Motorcycle passenger':'SR',
            'Taxi':'SR',
            'Bicycle':'Bike',
            'Mobility scooter':'Walk',
            'Ferry':'Transit',
            'School bus (private/chartered)': 'Transit'
        }
        WORK = {}
        

        trips['MAINMODE'] = trips['MAINMODE'].map(MODES)
        trips['MODE1'] = trips['MODE1'].map(MODES)
        trips['ORIGPURP'] = trips['ORIGPURP'].map(purpose)
        trips['DESTPURP'] = trips['DESTPURP'].map(purpose)
        
        
        

        # Add ENDTIME (needed for ordering)
        trips["ENDTIME"] = trips["STARTIME"]+ trips["TRAVTIME"]
        # now 
        as_trips, as_tours = build_tours_from_trips(trips)
        as_tours['main_purpose'] = as_tours['primary_purpose'].map(purpose)
        as_trips['purpose'] = as_trips['purpose'].map(purpose)
        as_trips['max_stops'] = (
            as_trips
                .groupby(['person_id', 'tour_id'])['STARTSTOP']
                .transform('max')
        )
        as_trips.to_csv('saved_data/trips_processed.csv')
        as_tours.to_csv('saved_data/tours_processed.csv')
        # get the number of indinvidual..
        if and_persons:
             return as_tours, as_trips, persons
        if and_trips:
            return as_tours, as_trips
    
        return as_tours
        
        
    def trips_to_tours_with_direction(self):
        """
        Build tours based ONLY on trip data.
        No stops.
        Adds: tour_id, main_trip_id, trip_direction (outbound/main/inbound)
        Returns: trips, tours, person
        """
        

        # -----------------------------------------
        # LOAD DATA
        # -----------------------------------------
        person = pd.read_excel('../2_QTS_PERSONS.xlsx')
        household = pd.read_excel('../1_QTS_HOUSEHOLDS.xlsx')
        persons = person.merge(household, on='HHID', how='left')
        person = self.process_hhts_person(persons)

        trips = pd.read_excel('../5_QTS_TRIPS.xlsx')
        # ADD after loading trips, before SA2 mapping
        trips['STARTIME'] = survey_time_to_minutes(trips['STARTIME'])
        if 'ARRTIME' in trips.columns:
            trips['ARRTIME'] = survey_time_to_minutes(trips['ARRTIME'])
        # then change this line:
        trips["ENDTIME"] = trips["STARTIME"] + trips["DURATION"]  # ← WRONG
        # to:
        trips["ENDTIME"] = trips["STARTIME"] + trips["TRAVTIME"]  # ← CORRECT

        # SA2 mapping
        sa1_to_sa2data = pd.read_csv('../SA1_SHP/sa1_2021.csv')
        sa1_to_sa2 = sa1_to_sa2data.set_index('SA1_CODE21')['SA2_CODE21'].astype(int).to_dict()

        trips['DESTSA2_2021'] = trips['DESTSA1_2021'].map(sa1_to_sa2)
        trips['ORIGSA2_2021'] = trips['ORIGSA1_2021'].map(sa1_to_sa2)
        person['HOME_SA2_2021'] = person['HOME_SA1_2021'].map(sa1_to_sa2)

        # -----------------------------------------
        # PURPOSE & MODE CLEANING
        # -----------------------------------------
        purpose = {
            'Pick up or drop off a passenger': 'escort',
            'Work (my workplace)': 'work',
            'Work (other work reason)': 'work',
            'Other place (please specify)': 'othdiscr',
            'At Home': 'home',
            'Go home': 'home',
            'Childcare or kindergarten':'escort',
            'Other Place':'othdiscr',
            'Refuel my vehicle':'othdiscr',
            'Education':'edu',
            'Pickup/deliver something (not work-related)':'othmaint',
            'Social Visit':'social',
            'Recreation/leisure activity':'social',
            'Recreation': 'social',
            'Eat/drink':'eat',
            'Personal Business':'othmaint',
            'Just accompanying someone':'escort',
            'Shopping':'shopping'
        }

        MODES = {
            'Car driver': 'DA',
            'Walking': 'Walk',
            'Car passenger': 'SR',
            'Public Bus': 'Transit',
            'Motorcycle driver': 'DA',
            'Truck passenger': 'SR',
            'Charter/Courtesy/Other bus':'Transit',
            'Light rail':'Transit',
            'Public bus':'Transit',
            'School bus (with route number)': 'Transit',
            'E-bike or E-scooter': 'Bike',
            'Truck driver':'DA',
            'Uber / Other Ride Share':'SR',
            'Train': 'Transit',
            'Other method':'DA',
            'Motorcycle passenger':'SR',
            'Taxi':'SR',
            'Bicycle':'Bike',
            'Mobility scooter':'Walk',
            'Ferry':'Transit',
            'School bus (private/chartered)': 'Transit'
        }

        trips['MAINMODE'] = trips['MAINMODE'].map(MODES)
        trips['MODE1'] = trips['MODE1'].map(MODES)
        trips['ORIGPURP'] = trips['ORIGPURP'].map(purpose)
        trips['DESTPURP'] = trips['DESTPURP'].map(purpose)

        # Add ENDTIME (needed for ordering)
        # ENDTIME = arrival time (STARTIME + TRAVTIME)
        # ENDTIME_WITH_ACTIVITY = departure from destination (STARTIME + TRAVTIME + DURATION)
        trips['DURATION'] = trips['DURATION'].fillna(0).clip(lower=0)
        trips["ENDTIME"]               = trips["STARTIME"] + trips["TRAVTIME"]
        trips["ENDTIME_WITH_ACTIVITY"] = trips["STARTIME"] + trips["TRAVTIME"] + trips["DURATION"]

        # -----------------------------------------
        # Sort trips
        # -----------------------------------------
        trips = trips.sort_values(["PERSID", "STARTIME"])

        # -----------------------------------------
        # STEP 1: Assign tour_id based on RETURNS HOME
        # -----------------------------------------
        trips["prev_dest"] = trips.groupby("PERSID")["DESTPLACE"].shift()

        # New tour if previous trip ENDED at HOME
        trips["new_tour_flag"] = (trips["prev_dest"] == "My home").astype(int)

        # First trip of each person starts a tour
        first_idx = trips.groupby("PERSID").head(1).index
        trips.loc[first_idx, "new_tour_flag"] = 1

        # Build cumulative tour IDs
        trips["tour_id"] = trips.groupby("PERSID")["new_tour_flag"].cumsum()

        trips = trips.drop(columns=["prev_dest", "new_tour_flag"])

        # -----------------------------------------
        # STEP 2: Main trip detection
        # -----------------------------------------
        purpose_priority = {
            'work': 1,
            'edu': 2,
            'escort': 6,
            'shopping': 4,
            'othmaint': 3,
            'othdiscr': 5,
            'eat': 7,
            'social': 8,
            'home': 9
        }

        trips["purpose_rank"] = trips["DESTPURP"].map(purpose_priority).fillna(5)

        main_idx = trips.loc[
            trips.groupby(["PERSID", "tour_id"])["purpose_rank"].idxmin()
        ]

        main_map = (
            main_idx[["PERSID", "tour_id", "TRIPID"]]
            .rename(columns={"TRIPID": "main_trip_id"})
        )

        trips = trips.merge(main_map, on=["PERSID", "tour_id"], how="left")

        # -----------------------------------------
        # STEP 3: Assign outbound / main / inbound
        # -----------------------------------------
        trips["trip_direction"] = np.where(
            trips["TRIPID"] < trips["main_trip_id"], "outbound",
            np.where(trips["TRIPID"] == trips["main_trip_id"], "main", "inbound")
        )

        # -----------------------------------------
        # STEP 4: Direction counts per tour
        # -----------------------------------------
        direction_counts = (
            trips.groupby(["PERSID", "tour_id", "trip_direction"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .rename(columns={
                "outbound": "num_outbound",
                "inbound": "num_inbound",
                "main": "num_main"
            })
        )

        for col in ["num_outbound", "num_inbound", "num_main"]:
            if col not in direction_counts:
                direction_counts[col] = 0

        # -----------------------------------------
        # STEP 5: Purpose counts per tour
        # -----------------------------------------
        purpose_counts = (
            trips.groupby(["PERSID", "tour_id", "DESTPURP"])
            .size()
            .unstack(fill_value=0)
        )
        purpose_counts.columns = [p + "_count" for p in purpose_counts.columns]
        purpose_counts = purpose_counts.reset_index()

        # -----------------------------------------
        # STEP 6: Build tours table
        # -----------------------------------------
        tours = (
            trips.groupby(["PERSID", "tour_id"])
            .agg(
                tour_start=("STARTIME", "min"),
                tour_end=("ENDTIME", "max"),
                total_travel_time=("TRAVTIME", "sum"),
                total_activity_time=("DURATION_ACTIVITY", "sum"),
                num_trips=("TRIPID", "count"),
                main_purpose=("DESTPURP", lambda x: x.value_counts().idxmax())
            )
            .reset_index()
        )

        # Add home SA2
        tours = tours.merge(
            person[["PERSID", "HOME_SA2_2021"]],
            on="PERSID",
            how="left"
        ).rename(columns={"HOME_SA2_2021": "tour_start_SA2"})

        # Merge direction and purpose counts
        tours = (
            tours
            .merge(direction_counts, on=["PERSID", "tour_id"], how="left")
            .merge(purpose_counts, on=["PERSID", "tour_id"], how="left")
            .fillna(0)
        )

        return trips, tours, person
        
    def generate_alternatives(self, df, alt_method:str, *args, **kwargs):
        
        if alt_method == 'Mode':
            modes = pd.DataFrame({
                "mode": ["BUS", "CAR"],
                "mode_id": [1, 2]
            })

            # Cartesian product: every person × every mode
            df_out = df.merge(modes, how="cross")
            return df_out
        elif alt_method  == 'DEST_SAMPLE':
            df = df.copy()
            df["DTAZ"] = df.apply(
                lambda row: random.sample(range(1, 330), 1),
                axis=1
            )
            df_out = df.explode('DTAZ').reset_index(drop=True)
            return df_out
        elif alt_method == 'Time':
            times = pd.DataFrame({
                "time_period": ["am_peak_end", "a", "PM", "EV"],
                "time_id": [1, 2, 3, 4]
            })
            

            # Cartesian prodnct: every person × every mode
            df_out = df.merge(times, how="cross")
            return df_out
        elif alt_method == 'TIME_SAMPLE':
            times = kwargs.get('times')
            a = len(times.index.values)
            print('check this please')
            df["duration_choices"] = df.apply(
                lambda row: random.sample(range(1, a), 1)[0],
                axis=1
            )
            
        return df # on person id add columns modes....
             
        
    def process_purpose(self):
        '''MAPS PURPOSE TO MORE FRIENDLY TERMS'''
        
        purpose =   {'escort': 'Just accompanying someone',
                    'work': 'Work (other work reason)',
                    'escort': 'Childcare or kindergarten',
                    'work':'Work (my workplace)',
                    'othdiscr':'Refuel my vehicle',
                    'home':'Go home',
                    'home': 'At Home',
                    'escort':'Pick up or drop off a passenger',
                    'edu':'Education',
                    'othmaint':'Pickup/deliver something (not work-related)',
                    'social':'Social Visit',
                    'social':'Recreation/leisure activity',
                    'eat':'Eat/drink',
                    'othmaint':'Personal Business',
                    'othdiscr' : 'Other Place',
                    'shopping':'Shopping'}
        
        
    def read_in_persons(self):
        a, b,c, pp = self.read_in_hhts()
        if 'PERSID' in pp.columns:
            pp = pp.rename(columns={'PERSID': 'person_id'})
        return pp
    
    
    def linked_stops_to_tours(self):
        '''
        
        returns trips, tours, stops, person
        '''
        #from data_connect import Synth_Connect
        
        

        # ------------------------------------------------------------
        # LOAD DATA
        # ------------------------------------------------------------
    
        person = pd.read_excel('../2_QTS_PERSONS.xlsx')
        household = pd.read_excel('../1_QTS_HOUSEHOLDS.xlsx')
        persons = person.merge(household, on='HHID', how='left')
        #cleanup person
        person = Synth_Connect().process_hhts_person(persons)
        
        stops = pd.read_excel('../4_QTS_STOPS.xlsx')
        trips = pd.read_excel('../5_QTS_TRIPS.xlsx')
        sa1_to_sa2data = pd.read_csv('../SA1_SHP/sa1_2021.csv')
        #sa1_to_sa2data['SA1_CODE21', 'SA2_CODE21'] #-> create teh map from this 
        sa1_to_sa2 = sa1_to_sa2data.set_index('SA1_CODE21')['SA2_CODE21'].astype(int).to_dict()
        trips['DESTSA2_2021'] = trips['DESTSA1_2021'].map(sa1_to_sa2)
        trips['ORIGSA2_2021'] = trips['ORIGSA1_2021'].map(sa1_to_sa2)
        stops['DESTSA2_2021'] = stops['DESTSA1_2021'].map(sa1_to_sa2)
        stops['ORIGSA2_2021'] = stops['ORIGSA1_2021'].map(sa1_to_sa2)
        person['HOME_SA2_2021'] = person['HOME_SA1_2021'].map(sa1_to_sa2)
        
        # ← ADD HERE — convert stops times from HHMM to minutes
        stops['STARTIME'] = survey_time_to_minutes(stops['STARTIME'])
        if 'ARRTIME' in stops.columns:
            stops['ARRTIME'] = survey_time_to_minutes(stops['ARRTIME'])
        if 'ENDTIME' in stops.columns:
            stops['ENDTIME'] = survey_time_to_minutes(stops['ENDTIME'])

        # then the trips conversion which you already have
        trips['STARTIME'] = survey_time_to_minutes(trips['STARTIME'])
        if 'ARRTIME' in trips.columns:
            trips['ARRTIME'] = survey_time_to_minutes(trips['ARRTIME'])
        trips['ENDTIME'] = trips['STARTIME'] + trips['TRAVTIME']
        
        purpose =   {   'escort': 'Just accompanying someone',
                        'work': 'Work (other work reason)',
                        'escort': 'Childcare or kindergarten',
                        'work':'Work (my workplace)',
                        'othdiscr':'Refuel my vehicle',
                        'home':'Go home',
                        'home': 'At Home',
                        'escort':'Pick up or drop off a passenger',
                        'edu':'Education',
                        'othmaint':'Pickup/deliver something (not work-related)',
                        'social':'Social Visit',
                        'social':'Recreation/leisure activity',
                        'eat':'Eat/drink',
                        'othmaint':'Personal business',
                        'othdiscr' : 'Other Place',
                        'othdiscr' :'Other place (please specify)',
                        'shopping':'Shopping'}
        
        purpose = {
        'Pick up or drop off a passenger': 'escort',
        'Work (my workplace)': 'work',
        'Work (other work reason)':'work',
        'Other place (please specify)': 'othdiscr',
        'At Home': 'home',
        'Go home': 'home',
        'Childcare or kindergarten':'escort',
        'Other Place': 'othdiscr',
        'Refuel my vehicle': 'othdiscr',
        'Education': 'edu',
        'Pickup/deliver something (not work-related)': 'othmaint',
        'Social Visit':'social',
        'Recreation/leisure activity': 'social',
        "Recreation": "social",
        'Eat/drink': 'eat',
        'Personal business': 'othmaint',
        'Just accompanying someone': 'escort',
        'Shopping': 'shopping'
        }
        
        MODES = {
            'Car driver': 'DA',
            'Walking': 'Walk',
            'Car passenger': 'SR',
            'Public Bus': 'Transit',
            'Motorcycle driver': 'DA',
            'Truck passenger': 'SR',
            'Charter/Courtesy/Other bus':'Transit',
            'Light rail':  'Transit',
            'Public bus': 'Transit',
            'School bus (with route number)': 'Transit',
            'E-bike or E-scooter': 'Bike',
            'Truck driver': 'DA',
            'Uber / Other Ride Share': 'SR',
            'Train':   'Transit',
            'Other method': 'DA',
            'Motorcycle passenger': 'SR',
            'Taxi': 'SR',
            'Bicycle': 'Bike',
            'Mobility scooter': 'Walk',
            'Ferry':  'Transit',
            'School bus (private/chartered)': 'Transit'
            }
        logging.info('reconsider modes')
        trips['MAINMODE'] = trips['MAINMODE'].map(MODES)
        missing = trips.loc[~trips['ORIGPURP'].isin(purpose.keys()), 'ORIGPURP'].unique()
        print(missing)
        trips['ORIGPURP'] = trips['ORIGPURP'].map(purpose)
        trips['DESTPURP'] = trips['DESTPURP'].map(purpose)
        

   

        # If you don’t have proper day info, you can set all to 1
        # or generate from STARTIME later if it’s datetime type
        trips["date"] = 1

        # ------------------------------------------------------------
        # DEFINE ANCHORS (places considered as tour endpoints)
        # ------------------------------------------------------------
        anchor_places = ["My home", "work", "school"]

        # ------------------------------------------------------------
        # STEP 1: PREPARE & FIND FIRST TRIPS NOT STARTING AT HOME
        # ------------------------------------------------------------
        trips = trips.sort_values(["PERSID", "STARTIME"])

        first_idx = trips.groupby(["PERSID", "date"]).head(1).index
        starts_not_home = trips.loc[first_idx]
        bad_start_mask = ~starts_not_home["ORIGPLACE"].isin(anchor_places)
        bad_first_trips_idx = starts_not_home[bad_start_mask].index

        # ------------------------------------------------------------
        # STEP 2: FIND SEQUENCES THAT START AWAY FROM ANCHOR
        # ------------------------------------------------------------
        bad_groups = []

        for idx in bad_first_trips_idx:
            pers = trips.loc[idx, "PERSID"]
            day = trips.loc[idx, "date"]

            day_trips = (
                trips[(trips["PERSID"] == pers) & (trips["date"] == day)]
                .sort_values("STARTIME")
            )

            # first time the person returns to an anchor place
            return_anchor = day_trips[day_trips["DESTPLACE"].isin(anchor_places)]

            if len(return_anchor) == 0:
                # never returns → discard
                bad_groups.append(day_trips.index)
                continue

            cutoff_time = return_anchor.iloc[0]["ENDTIME"]
            # include all trips before they reach the anchor
            morning_seq = day_trips[day_trips["ENDTIME"] <= cutoff_time]
            bad_groups.append(morning_seq.index)

        # ------------------------------------------------------------
        # STEP 3: TRY TO MOVE THOSE TRIPS TO PREVIOUS DAY
        # ------------------------------------------------------------
        to_discard = []

        for seq_idx in bad_groups:
            seq = trips.loc[seq_idx]
            pers = seq["PERSID"].iloc[0]
            day = seq["date"].iloc[0]

            person_days = sorted(trips[trips["PERSID"] == pers]["date"].unique())
            earlier_days = [d for d in person_days if d < day]

            if not earlier_days:
                to_discard.extend(seq_idx)
                continue

            # move to previous day
            new_day = earlier_days[-1]
            trips.loc[seq_idx, "date"] = new_day

        # ------------------------------------------------------------
        # STEP 4: DROP TRIPS THAT CAN’T BE FIXED
        # ------------------------------------------------------------
        trips = trips.drop(index=to_discard)

        # ------------------------------------------------------------
        # STEP 5: REBUILD TOURS BASED ON ANCHORS
        # ------------------------------------------------------------
        trips = trips.sort_values(["PERSID", "STARTIME"])

        # detect if previous destination was an anchor
        trips["prev_dest"] = trips.groupby("PERSID")["DESTPLACE"].shift()
        trips["new_tour_flag"] = trips["prev_dest"].isin(anchor_places).astype(int)

        # ensure first trip of each person starts a tour
        first_idx = trips.groupby("PERSID").head(1).index
        trips.loc[first_idx, "new_tour_flag"] = 1

        # generate tour IDs
        trips["tour_id"] = trips.groupby("PERSID")["new_tour_flag"].cumsum()

        trips = trips.drop(columns=["prev_dest", "new_tour_flag"])

        # ------------------------------------------------------------
        # STEP 6: CREATE TOURS TABLE
        # ------------------------------------------------------------
        tours = (
            trips.groupby(["PERSID", "tour_id"])
            .agg(
                tour_start=("STARTIME", "min"),
                tour_end=("ENDTIME", "max"),
                num_trips=("TRIPID", "count"),
                main_purpose=("DESTPURP", lambda x: x.value_counts().idxmax())
            )
            .reset_index()
        )
        
        # Tour start SA2 is always the person's home
        tours = tours.merge(
            person[['PERSID', 'HOME_SA2_2021']],
            on='PERSID',
            how='left'
        ).rename(columns={'HOME_SA2_2021': 'tour_start_SA2'})

        
        # ------------------------------------------------------------
        # STEP 7: MERGE STOPS WITH TOUR INFORMATION
        # ------------------------------------------------------------
        stops = stops.merge(
            trips[["TRIPID", "PERSID", "tour_id"]],
            on=["TRIPID", "PERSID"],
            how="left"
        )
        
        # ------------------------------------------------------------
        # STEP 8: COUNT STOPS BY DIRECTION (OUTBOUND/INBOUND)
        # ------------------------------------------------------------
        trips = trips.sort_values(["PERSID", "tour_id", "STARTIME", "ENDTIME"])

        purpose_priority = {
        'work': 1,
        'edu': 2,
        'escort': 3,
        'shopping': 4,
        'othmaint': 5,
        'othdiscr': 6,
        'eat': 7,
        'social': 8,
        'home': 9
        }
        trips["purpose_rank"] = trips["DESTPURP"].map(purpose_priority).fillna(5)
        
        
        # Find the main trip (longest duration or latest END)
        '''
        main_trip_indices = (
            trips.groupby(["PERSID", "tour_id"])["ENDTIME"]
            .idxmax()
        )
        '''
        main_trip_indices = (
        trips.loc[trips.groupby(["PERSID", "tour_id"])["purpose_rank"].idxmin()]
        .index
        )
        
        # extract MODE of the main trip
        main_mode = (
            trips.loc[main_trip_indices, ['PERSID', 'tour_id', 'MAINMODE']]
            .rename(columns={'MAINMODE': 'tour_mode'})
        )
        # merge into tours
        #tours = tours.merge(main_mode, on=['PERSID', 'tour_id'], how='left')
        tours = self._clean_merge(
                tours,
                main_mode,
                on=["PERSID", "tour_id"],
                how="left",
                validate="many_to_one"
            )
        
        
        
        trips["tour_leg"] = "outbound"
        trips.loc[main_trip_indices, "tour_leg"] = "main"
        
    
        
        
        # Identify the main trip's destination SA2
        main_sa2 = (
            trips.loc[main_trip_indices, ['PERSID', 'tour_id', 'DESTSA2_2021']]
            .rename(columns={'DESTSA2_2021': 'tour_dest_SA2'})
        )

        #tours = tours.merge(main_sa2, on=['PERSID', 'tour_id'], how='left')
        tours = self._clean_merge(
                tours,
               main_sa2,
                on=["PERSID", "tour_id"],
                how="left",
                validate="many_to_one"
            )

        
        # Identify main trip index per tour
        main_idx = trips.loc[main_trip_indices, ["PERSID", "tour_id", "TRIPID"]]
        main_idx = main_idx.rename(columns={"TRIPID": "main_trip_id"})

        # Merge main trip id into trip table
        trips = trips.merge(main_idx, on=["PERSID", "tour_id"], how="left")

        # Define trip direction
        trips["trip_direction"] = np.where(
            trips["TRIPID"] < trips["main_trip_id"], "outbound",
            np.where(trips["TRIPID"] == trips["main_trip_id"], "main", "inbound")
        )

        # Merge direction into stops
        '''
        stops = stops.merge(
            trips[["TRIPID", "PERSID", "tour_id", "trip_direction"]],
            on=["TRIPID", "PERSID", "tour_id"],
            how="left"
        )
        '''
        stops = self._clean_merge(
                stops,
                trips[["TRIPID", "PERSID", "tour_id", "trip_direction"]],
                on=["TRIPID", "PERSID", "tour_id"],
                how="left",
                validate="many_to_one"
        )

        # Rename for consistency
        stops["tour_leg"] = stops["trip_direction"]
        
        stops = stops.sort_values(["PERSID", "tour_id", "STARTIME"])

    # Collapse consecutive stops with same direction  
        stops["prev_leg"] = stops.groupby(["PERSID", "tour_id"])["tour_leg"].shift()

        stops["direction_change"] = (stops["tour_leg"] != stops["prev_leg"]).astype(int)

        # Keep only the first stop of each direction block
        #stops_collapsed = stops[stops["direction_change"] == 1].copy()

        # First stop of each tour must be counted
        first_stop_idx = stops.groupby(["PERSID","tour_id"]).head(1).index
        #stops_collapsed = pd.concat([stops_collapsed, stops.loc[first_stop_idx]]).drop_duplicates()
        #stops_collapsed = stops_collapsed.append(stops.loc[first_stop_idx]).drop_duplicates()

        # Count stops by direction
        dir_stops = (
            stops.groupby(["PERSID", "tour_id", "tour_leg"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .rename(columns={
                "outbound": "num_outbound_stops",
                "inbound": "num_inbound_stops",
                "main": "num_main_stops"  # optional
            })
        )

        # Merge into the tours table
        tours = tours.merge(dir_stops, on=["PERSID", "tour_id"], how="left")

        # Fill missing types with zeros
        for col in ["num_outbound_stops", "num_inbound_stops", "num_main_stops"]:
            if col in tours.columns:
                tours[col] = tours[col].fillna(0).astype(int)
        
        #tours["num_outbound"] = tours["num_outbound_stops"] + 1
        #tours["num_inbound"] = tours["num_inbound_stops"] + 1
        
        tours["stop_frequency"] = (
        tours["num_outbound_stops"].astype(str) + "out_" +
        tours["num_inbound_stops"].astype(str) + "in"
        )
        # ------------------------------------------------------------
        # DONE
        # ------------------------------------------------------------
        print("Processing complete.")
        print(f"Trips retained: {len(trips)}")
        print(f"Tours created: {len(tours)}")
        print(f"Stops linked: {len(stops)}")
        stops = stops.merge(
            trips[["TRIPID", "PERSID", "tour_id", "DURATION_ACTIVITY"]],
            on=["TRIPID", "PERSID", "tour_id"],
            how="left"
        )

        stops["DURATION_ACTIVITY"] = stops["DURATION_ACTIVITY"].fillna(0)
        
        # total number of activities (stops)
        acts_per_tour = (
            stops.groupby(["PERSID", "tour_id"])
            .size()
            .reset_index(name="num_activities")
        )

        # number of *unique* activity types per tour (distinct DESTPURP)
        unique_acts_per_tour = (
            stops.groupby(["PERSID", "tour_id"])["DESTPURP"]
            .nunique()
            .reset_index(name="num_unique_activities")
        )

        # average duration between stops, you can add more time variables later

        # merge back into tours
        tours = (
            tours.merge(acts_per_tour, on=["PERSID", "tour_id"], how="left")
            .merge(unique_acts_per_tour, on=["PERSID", "tour_id"], how="left")
            .fillna({"num_activities": 0, "num_unique_activities": 0})
        )

        print("Updated tours table with activity summaries:")
        print(tours[["PERSID", "tour_id", "main_purpose", "num_activities", "num_unique_activities",
                    "num_outbound_stops", "num_inbound_stops"]].head())
        
        
        purpose_counts = (
        stops.groupby(["PERSID", "tour_id", "DESTPURP"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        )

        # Rename columns safely (optional)
        purpose_counts.columns = [
            "PERSID",
            "tour_id",
            *[f"num_{c.lower().replace(' ', '_')}" for c in purpose_counts.columns[2:]],
        ]
        
        mode_counts = (
        stops.groupby(["PERSID", "tour_id", "MODE"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        )

        # rename mode columns for consistency
        mode_counts.columns = [
            "PERSID", 
            "tour_id",
            *[f"num_mode_{str(c).lower().replace(' ', '_')}" for c in mode_counts.columns[2:]]
        ]

        purpose_counts["num_work_total"] = (
        purpose_counts.get("num_work_(my_workplace)", 0)
        + purpose_counts.get("num_work_(other_work_reason)", 0)
        )

        purpose_counts["num_education_total"] = (
            purpose_counts.get("num_education", 0)
            + purpose_counts.get("num_childcare_or_kindergarten", 0)
        )

        purpose_counts["num_social_total"] = (
            purpose_counts.get("num_social_visit", 0)
            + purpose_counts.get("num_recreation/leisure_activity", 0)
        
        ).clip(upper = 1)
        
        purpose_counts["num_eat_total"] = (
            purpose_counts.get("num_eat/drink", 0)
        ).clip(upper = 1)

        purpose_counts["num_shopping_total"] = (
            purpose_counts.get("num_shopping", 0)
        ).clip(upper = 1)

        purpose_counts["num_othermaint_total"] = (
            #purpose_counts.get("num_go_home", 0)
        # + purpose_counts.get("num_refuel_my_vehicle", 0)
            #+ purpose_counts.get("num_just_accompanying_someone", 0)
            #+ purpose_counts.get("num_park_/_pick-up_a_vehicle", 0)
            #+ purpose_counts.get("num_get_on/off_public_transport", 0)
            purpose_counts.get("num_personal_business", 0)
        ).clip(upper = 1)
        
        purpose_counts["num_otherdiscr_total"] = (
            #purpose_counts.get("num_go_home", 0)
            purpose_counts.get("num_refuel_my_vehicle", 0)
            + purpose_counts.get("num_just_accompanying_someone", 0)
            #+ purpose_counts.get("num_park_/_pick-up_a_vehicle", 0)
            #+ purpose_counts.get("num_get_on/off_public_transport", 0)
        # + purpose_counts.get("num_personal_business", 0)
        ).clip(upper = 1)
        
        purpose_counts["num_escort_total"] = (
            purpose_counts.get("num_pick_up_or_drop_off_a_passenger", 0)
            + purpose_counts.get("num_pickup/deliver_something_(not_work-related)", 0)
        ).clip(upper = 2)

        # Total activities = sum of all broad categories
        purpose_counts["num_total_activities"] = (
            purpose_counts[
                [
                    "num_work_total",
                    "num_education_total",
                    "num_social_total",
                    "num_eat_total",
                    "num_shopping_total",
                    "num_othermaint_total",
                    "num_otherdiscr_total",
                    "num_escort_total"
                ]
            ].sum(axis=1)
        )
        logging.info(f'lenth of tours before merge:{len(tours)}')
        tours = tours.merge(purpose_counts, on=["PERSID", "tour_id"], how="left").fillna(0)
        logging.info(f'lenth of tours after merge:  {len(tours)}')
        tours = tours.merge(mode_counts, on=["PERSID", "tour_id"], how="left").fillna(0)
        logging.info(f'lenth of tours after additional merge: {len(tours)}')
        tours['num_bike'] = tours[['num_mode_bicycle']].sum(axis=1)
        tours['num_car'] = tours[['num_mode_car_driver','num_mode_car_passenger',
                        'num_mode_motorcycle_driver','num_mode_motorcycle_passenger',
                        'num_mode_truck_driver','num_mode_truck_passenger']].sum(axis=1)
        tours['num_emobility'] = tours[['num_mode_e-bike_or_e-scooter','num_mode_mobility_scooter']].sum(axis=1)
        tours['num_pt'] = tours[['num_mode_public_bus','num_mode_school_bus_(private/chartered)',
                    'num_mode_school_bus_(with_route_number)',
                    'num_mode_charter/courtesy/other_bus','num_mode_light_rail',
                    'num_mode_train','num_mode_ferry']].sum(axis=1)
        tours['num_walking'] = tours[['num_mode_walking']].sum(axis=1)
        
        
        person['RELATIONSHIP'] = person['RELATIONSHIP'].fillna('DNF')
        person['MCLICENCE'] = person['MCLICENCE'].fillna('noLicence')
        person['CARLICTYPE'] = person['CARLICTYPE'].fillna('DNF')
        person['MCLICTYPE'] = person['MCLICTYPE'].fillna('DNF')
        person['WFHWEEK'] = person['WFHWEEK'].fillna(0)
        person['REASON'] =person['REASON'].fillna('no')
        person['ASSISTANY'] =person['ASSISTANY'].fillna('no')
        person['INDUSTRY'] = person['INDUSTRY'].fillna('DNF')
        person['ED_TYPE'] = person['INDUSTRY'].fillna('DNF')
        person['ASSISTOTHER'] =person['ASSISTOTHER'].fillna('no')
        person['ASSISTAGE'] =person['ASSISTAGE'].fillna('no')
        person['WORKSTATUS'] = person['WORKSTATUS'].fillna('no')
        person = person.drop(columns = ['ASSISTENGLISH', 'ASSISTSTHC', 'LASTLEFT', 'ANZSCO_1-digit', 'ANZSCO_3-digit', 'ASSISTDISABILITY', 'ASSISTLTHC'])
        stops['NOONE'] = stops['NOONE'].fillna(0)
        stops['VEHOCC'] = stops['VEHOCC'].fillna(0)
        stops['VEHPARKED'] = stops['VEHPARKED'].fillna('No Vehicle')
        stops['VEHID'] = stops['VEHID'].fillna('-999')
        trips = trips.drop(columns = ['MODE1','MODE2', 'MODE3', 'MODE4', 'MODE5', 'MODE6', 'MODE7', 'MODE8'])
        print(len(trips))
        # need to fix the DURATION..
        return trips, tours, stops, person


        
    def read_in_hhts(self):
        '''
        Docstring for read_in_hhts
        process trips tours, stops and persons
        :param self: Description
        '''
        #from stop_freq import linked_stops_to_tours
        trips, tours, stops, persons = self.linked_stops_to_tours()
        sa1_to_sa2 = pd.read_csv('sa1_2021.csv')[['SA1_CODE21', 'SA2_CODE21']]
        m = sa1_to_sa2.set_index('SA1_CODE21')['SA2_CODE21']
        sa2_to_order = pd.read_csv('seq_zone_ids_.csv')[['SA2_MAIN16', 'ORDER']]
        o = sa2_to_order.set_index('SA2_MAIN16')['ORDER']
        
        persons['OTAZ'] = persons['HOME_SA2_2021'].apply(lambda x: o.get(x))
       # tours['start_sa2'] = tours['start_location'].apply(lambda x: m.get(x))
        #tours['end_sa2'] = tours['end_location'].apply(lambda x: m.get(x))
        trips['OTAZ'] = trips['ORIGSA2_2021'].apply(lambda x: o.get(x))
        trips['DTAZ'] = trips['DESTSA2_2021'].apply(lambda x: o.get(x))
        tours['OTAZ'] = tours['tour_start_SA2'].apply(lambda x: o.get(x))
        tours['DTAZ'] = tours['tour_dest_SA2'].apply(lambda x: o.get(x))
        stops['OTAZ'] = stops['ORIGSA2_2021'].apply(lambda x: o.get(x))
        stops['DTAZ'] = stops['DESTSA2_2021'].apply(lambda x: o.get(x))
        tours = tours.dropna()
        persons = persons.dropna()
        trips = trips.dropna()
        stops = stops.dropna()
        tours['OTAZ'] = tours['OTAZ'].astype(int)
        persons['OTAZ'] = persons['OTAZ'].astype(int)
        trips['OTAZ'] = trips['OTAZ'].astype(int)
        stops['OTAZ'] = stops['OTAZ'].astype(int)
        
        tours['DTAZ'] = tours['DTAZ'].astype(int)
       
        trips['DTAZ'] = trips['DTAZ'].astype(int)
        stops['DTAZ'] = stops['DTAZ'].astype(int)
        
        trips.to_csv('initial_trips_hhts.csv')
        return trips, tours, stops, persons
    
    def get_land(self):
        if self.land is not None:
            return self.land
        else:
            import os as _os
            if self.new_pop:
                _new_pop_dir = (_os.environ.get('NEW_POP_DIR') or _DEFAULT_NEW_POP_DIR)
                _landuse_path = _os.path.join(_new_pop_dir, 'saved_data', 'zone_landuse_507.csv')
                _osm_path = _os.path.join(_new_pop_dir, 'saved_data', 'SA2_2016_OSM_counts_507.csv')
                self.land = pd.read_csv(_landuse_path)
                # ORDER == SA2 code in the 507-zone scope (no ORDER indirection) so
                # every downstream .merge(..., on='ORDER') keeps working unchanged.
                self.land['ORDER'] = self.land['SA2_MAIN16']
            else:
                self.land = pd.read_csv('zone_order_and_landuse.csv')
            # Convert everything possible to numeric (keep column as-is if it can't)
            for col in self.land.columns:
                try:
                    self.land[col] = pd.to_numeric(self.land[col])
                except (ValueError, TypeError):
                    pass

            # Replace inf with NaN
            self.land = self.land.replace([np.inf, -np.inf], np.nan)

            # Convert numeric columns explicitly to float64
            numeric_cols = self.land.select_dtypes(include=["number"]).columns
            self.land[numeric_cols] = self.land[numeric_cols].astype("float64")

            # Remove NaNs in numeric columns (or fill with 0 if preferred)
            self.land[numeric_cols] = self.land[numeric_cols].fillna(0)

            # ── Enrich with OSM POI counts needed by size terms ───────────────
            # zone_order_and_landuse.csv (or zone_landuse_507.csv in new_pop mode)
            # lacks these OSM POI count columns. They live in
            # pddopulation_by_order_with_osm_and_landuse.csv (legacy, ORDER-keyed)
            # or SA2_2016_OSM_counts_507.csv (new_pop, SA2-keyed) and are merged
            # in once here so all downstream stages see them.
            #   university    edu:     zone 40 CBD has 5 (QUT, TAFE etc) — edu rank 259/330 without it
            #   college       edu:     community colleges / TAFE campuses (38 nz)
            #   kindergarten  escort:  early childhood centres (222 nz; lu 'childcare' = 1 nz)
            #   sports_centre social/othdiscr: sport & recreation venues (247 nz)
            #   fitness_centre social/othdiscr: gyms (186 nz)
            #   clinic        othmaint: medical clinics (152 nz)
            #   pharmacy      othmaint: pharmacies (207 nz)
            #   bank          othmaint: banks (108 nz)
            #   restaurant    eat:     restaurants (260 nz) — complements EATOUT
            #   cafe          eat:     cafes (282 nz)
            #   pub           eat:     pubs/restaurant-bars (205 nz)
            _OSM_SIZE_COLS = [
                'university', 'college', 'kindergarten',
                'sports_centre', 'fitness_centre',
                'clinic', 'pharmacy', 'bank',
                'restaurant', 'cafe', 'pub',
            ]
            _missing_osm = [c for c in _OSM_SIZE_COLS if c not in self.land.columns]
            if _missing_osm:
                _pop_file = _osm_path if self.new_pop else 'pddopulation_by_order_with_osm_and_landuse.csv'
                _key_col = 'SA2_MAIN16' if self.new_pop else 'ORDER'
                try:
                    if _os.path.exists(_pop_file):
                        _avail = [c for c in _missing_osm if c in pd.read_csv(_pop_file, nrows=0).columns]
                        _pop = pd.read_csv(_pop_file, usecols=[_key_col] + _avail)
                        for c in _avail:
                            _pop[c] = pd.to_numeric(_pop[c], errors='coerce').fillna(0)
                        self.land = self.land.merge(_pop[[_key_col] + _avail], on=_key_col, how='left')
                        for c in _avail:
                            self.land[c] = self.land[c].fillna(0)
                except Exception as _e:
                    import logging as _log
                    _log.getLogger(__name__).warning(f"get_land: could not merge OSM POI columns: {_e}")
                    for c in _missing_osm:
                        if c not in self.land.columns:
                            self.land[c] = 0.0
            # ── Validate OSM merge completeness ───────────────────────────────
            for c in _OSM_SIZE_COLS:
                if c not in self.land.columns:
                    self.land[c] = 0.0
            # ──────────────────────────────────────────────────────────────────

            return self.land
        
    def get_tours(self):
        if self.tours is not None:
            return self.tours
        else:
            from stop_freq import linked_stops_to_tours
            trips, tours, stops, persons = linked_stops_to_tours()
            self.tours = tours
            return self.tours
    def read_in_trips(self):
        if self.trips is not None:
            print('trips is no none..')
            return self.trips
        else:
            logging.info('readin to trips ')
            file_path = Path('saved_data/trips_processed.csv')
            if file_path.exists():
                self.trips = pd.read_csv('saved_data/trips_processed.csv')
            else:
                print('need to_process saved_data/trips_processed.csv agaom')
                
                
            MODES = {
                'Car driver': 'DA',
                'Walking': 'Walk',
                'Car passenger': 'SR',
                'Public Bus': 'Transit',
                'Motorcycle driver': 'DA',
                'Truck passenger': 'SR',
                'Charter/Courtesy/Other bus':'Transit',
                'Light rail':'Transit',
                'Public bus':'Transit',
                'School bus (with route number)': 'Transit',
                'E-bike or E-scooter': 'Bike',
                'Truck driver':'DA',
                'Uber / Other Ride Share':'SR',
                'Train': 'Transit',
                'Other method':'DA',
                'Motorcycle passenger':'SR',
                'Taxi':'SR',
                'Bicycle':'Bike',
                'Mobility scooter':'Walk',
                'Ferry':'Transit',
                'School bus (private/chartered)': 'Transit',
                'Car driver ': 'DA',
                'Walking ': 'Walk',
                'Bicycle ': 'Bike',
                'Public Bus ': 'Transit',
            }
            self.trips['MODE1'] = self.trips['MODE1'].map(MODES)
            self.trips['OTAZ'] = self.trips['OTAZ'].astype(int)
            self.trips['DTAZ'] = self.trips['DTAZ'].astype(int)
            self.trips['tour_origin'] =  self.trips['tour_origin'].fillna(self.trips['OTAZ']).astype(int)
            self.trips['TTAZ'] = self.trips['tour_destination'].fillna(self.trips['DTAZ']).astype(int)
            self.trips['chosen_trips_duration'] = self.trips['DURATION'].apply(time_hour_flat)
            return self.trips    
    
    def read_in_tours(self):
        if self.tours is not None:
            return self.tours
        else:
            logging.info('readin to tours ')
            self.tours = pd.read_csv('saved_data/tours_processed.csv')
            self.tours['OTAZ'] = self.tours['tour_origin'].astype(int)
            self.tours['DTAZ'] = self.tours['tour_destination'].astype(int)
            self.tours['duration'] = self.tours['duration'].fillna(0)
            #df.loc[df['end_time'].isna(), 'end_time'] = df['arr_time'] + df['duration']
            # Step 2 — confirm each person's unique choice set
            self.tours['start_time_hour'] = self.tours['start_time'].apply(time_hour)
            self.tours['end_time_hour'] = self.tours['end_time'].apply(time_hour)
            
            return self.tours
        
    
    
    def run_statistical_trip_generation(
        self,
        synthetic_persons,
        weight_col="PERSWGT23",
        purpose_col="DESTPURP",
        home_zone_col="ORDER",
        model_formula=None
    ):
        """
        Estimate purpose-specific Poisson trip generation models
        from HHTS and apply to synthetic population.

        Parameters
        ----------
        synthetic_persons : DataFrame
            Must contain same explanatory variables as HHTS persons.
        weight_col : str
            Person expansion weight column in HHTS.
        purpose_col : str
            Trip purpose column in trips file.
        home_zone_col : str
            Zone column in synthetic population.
        model_formula : str
            Optional statsmodels formula.
            If None, a default demographic formula is used.

        Returns
        -------
        synthetic_persons_with_trips : DataFrame
        zone_productions : DataFrame
        models : dict of fitted models by purpose
        """

        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        # -------------------------------------------------
        # 1️⃣ Load HHTS
        # -------------------------------------------------
        trips, tours, stops, persons = self.read_in_hhts()

        # -------------------------------------------------
        # 2️⃣ Count trips per person per purpose
        # -------------------------------------------------
        person_trips = (
            trips
            .groupby(["PERSID", purpose_col])
            .size()
            .reset_index(name="n_trips")
        )

        # Merge with persons
        person_model_df = persons.merge(
            person_trips,
            on="PERSID",
            how="left"
        )

        person_model_df["n_trips"] = (
            person_model_df["n_trips"]
            .fillna(0)
        )

        
        '''
        person["AGEGROUP"]
         sex = person["SEX"]
        rel = person["RELATIONSHIP"]
        work = person["WORKSTATUS"]
        act = person["MAINACT"]
        '''
        
        
        # -------------------------------------------------
        # 3️⃣ Default Model Formula
        # -------------------------------------------------
        if model_formula is None:
            model_formula = """
            n_trips ~
            C(AGEGROUP) +
            C(sexp) +

            C(vehrd)
            """

        purposes = trips[purpose_col].unique()

        models = {}
        synthetic_persons = synthetic_persons.copy()
        synthetic_persons.rename(columns = {'agep':'AGEGROUP'}, inplace = True)

        # Create storage for predictions
        synthetic_persons["expected_trips_total"] = 0

        # -------------------------------------------------
        # 4️⃣ Estimate Model Per Purpose
        # -------------------------------------------------
        for purpose in purposes:

            df_p = person_model_df[
                person_model_df[purpose_col] == purpose
            ].copy()

            if df_p.empty:
                continue

            # Fit weighted Poisson
            model = smf.glm(
                formula=model_formula,
                data=df_p,
                family=sm.families.Poisson(),
                freq_weights=df_p[weight_col]
            ).fit()

            models[purpose] = model

            # -------------------------------------------------
            # 5️⃣ Apply to Synthetic Population
            # -------------------------------------------------
            
            for col in ["AGEGROUP", "sexp", "vehrd"]:
    
                # Get categories seen in training
                train_cats = df_p[col].astype("category").cat.categories
                
                # Force synthetic persons to use same categories
                synthetic_persons[col] = pd.Categorical(
                    synthetic_persons[col],
                    categories=train_cats
                )
            
            
            
            synthetic_persons[f"expected_trips_{purpose}"] = (
                model.predict(synthetic_persons)
            )

            synthetic_persons["expected_trips_total"] += (
                synthetic_persons[f"expected_trips_{purpose}"]
            )

        # -------------------------------------------------
        # 6️⃣ Aggregate to Zone Productions
        # -------------------------------------------------
        zone_productions = (
            synthetic_persons
            .groupby(home_zone_col)["expected_trips_total"]
            .sum()
            .reset_index()
            .rename(columns={
                "expected_trips_total": "productions"
            })
        )

        return synthetic_persons, zone_productions, models
    
        
    def get_trip_rates_by_zone_and_purpose(
        self,
        weight_col="PERSON_WEIGHT",
        purpose_col="DESTPURP",
        origin_col="OTAZ",
        dest_col="DTAZ"
    ):
        """
        Returns:
            production_rates  (origin-based)
            destination_rates (attraction-based)

        Each contains:
            zone
            purpose
            weighted_trips
            total_weighted_persons
            trip_rate_per_person
        """

        # ----------------------------------------------
        # Load HHTS expanded data
        # ----------------------------------------------
        trips, tours, stops, persons = self.read_in_hhts()

        if weight_col not in persons.columns:
            raise ValueError(f"{weight_col} not found in persons file")

        # ----------------------------------------------
        # Merge weights onto trips
        # ----------------------------------------------
        trips = trips.merge(
            persons[["PERSID", weight_col]],
            on="PERSID",
            how="left"
        )

        # ----------------------------------------------
        # Total weighted persons
        # ----------------------------------------------
        total_weighted_persons = persons[weight_col].sum()

        # ==============================================
        # ✅ 1️⃣ PRODUCTION (Origin-Based)
        # ==============================================
        production = (
            trips
            .groupby([origin_col, purpose_col])[weight_col]
            .sum()
            .reset_index(name="weighted_trips")
        )

        production["total_weighted_persons"] = total_weighted_persons

        production["trip_rate_per_person"] = (
            production["weighted_trips"]
            / total_weighted_persons
        )

        production = production.rename(columns={
            origin_col: "zone"
        })

        # ==============================================
        # ✅ 2️⃣ DESTINATION (Attraction-Based)
        # ==============================================
        destination = (
            trips
            .groupby([dest_col, purpose_col])[weight_col]
            .sum()
            .reset_index(name="weighted_trips")
        )

        destination["total_weighted_persons"] = total_weighted_persons

        destination["trip_rate_per_person"] = (
            destination["weighted_trips"]
            / total_weighted_persons
        )

        destination = destination.rename(columns={
            dest_col: "zone"
        })

        return production, destination
        
    def get_trip_rate_to_zone_per_person(
        self,
        weight_col="PERSON_WEIGHT",
        zone_col="DTAZ"
    ):
        """
        Returns a dataframe with:
            zone
            weighted_trips
            total_weighted_persons
            trip_rate_per_person

        trip_rate_per_person =
            weighted_trips_to_zone /
            total_weighted_persons
        """

        # ---------------------------------------------------
        # Load HHTS expanded data
        # ---------------------------------------------------
        trips, tours, stops, persons = self.read_in_hhts()

        if weight_col not in persons.columns:
            raise ValueError(f"{weight_col} not found in persons file")

        # ---------------------------------------------------
        # Merge person weights into trips
        # ---------------------------------------------------
        trips = trips.merge(
            persons[["PERSID", weight_col]],
            on="PERSID",
            how="left"
        )

        # ---------------------------------------------------
        # Total weighted persons in region
        # ---------------------------------------------------
        total_weighted_persons = persons[weight_col].sum()

        # ---------------------------------------------------
        # Weighted trips to each destination zone
        # ---------------------------------------------------
        zone_weighted_trips = (
            trips
            .groupby(zone_col)[weight_col]
            .sum()
            .reset_index(name="weighted_trips")
        )

        # ---------------------------------------------------
        # Compute per-person trip rate
        # ---------------------------------------------------
        zone_weighted_trips["total_weighted_persons"] = total_weighted_persons

        zone_weighted_trips["trip_rate_per_person"] = (
            zone_weighted_trips["weighted_trips"]
            / total_weighted_persons
        )

        return zone_weighted_trips
    def get_dest_production_and_attraction(self):
        trips = self.get_trips()
        
        
        #summaries trips to get total trip counts per activity to type to zone, and then in a seperate dataframe from zone
        
        # 1. summarise trips per activity type and zone
        # ORIGIN PRODUCRION
        
        trip_summary = (
            trips
            .groupby(['DESTPURP', 'ORIGSA2_2021'])
            .size()                    # or .agg({'trip_id':'count'}) or sum of weights
            .reset_index(name='n_trips')
        )

        # 2. create a separate zone-level dataframe if needed
        zone_summary = (
            trip_summary
            .groupby('ORIGSA2_2021')['n_trips']
            .sum()
            .reset_index()
        )
        logging.info('test')
        
        
        #DESTINATION PRODUCTION
        
        trip_summary_d = (
            trips
            .groupby(['DESTPURP', 'DESTSA2_2021'])
            .size()                    # or .agg({'trip_id':'count'}) or sum of weights
            .reset_index(name='n_trips')
        )

        # 2. create a separate zone-level dataframe if needed
        zone_summary_d = (
            trip_summary_d
            .groupby('DESTSA2_2021')['n_trips']
            .sum()
            .reset_index()
        )
        logging.info('test')
            
        
    def get_trips(self):
        if self.trips is not None:
            return self.trips
        else:
            from stop_freq import linked_stops_to_tours
            trips, tours, stops, persons = linked_stops_to_tours()
            self.trips = trips
            return self.trips
        
    
    
    
    def process_hhts_person(self, df):
        df['incp'] = df['INCOME'].map(income_mapping).fillna(17)
        df['WORKSTAT'] = df['WORKSTATUS'].map(work_map).fillna(0)
        df['indp'] = df['INDUSTRY'].map(industy_map).fillna(20)
        df['sexp'] = df['SEX'].map(sex_map)
        # ── Impute non-response income (codes 16/17 = "prefer not to say" /
        # "select one" / "not applicable") from the person's profile so the
        # income coefficient is estimable. ~88% of HHTS income is non-response,
        # which otherwise collapses every income band. Disable with
        # GA_IMPUTE_INCOME=0.
        if os.environ.get('GA_IMPUTE_INCOME', '1').strip() != '0':
            df = _impute_income(df)
        df = df.rename(columns={'BIKES': 'bikes', 'HHSIZE': 'nprd'})
        df['vehrd'] = df['HHVEH'].clip(upper=4)
        # CARLICENCE is boolean True/False in QTS — convert to 0/1
        if df['CARLICENCE'].dtype == bool or df['CARLICENCE'].dtype == object:
            df['CARLICENCE'] = df['CARLICENCE'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)
        logging.info(f'lenth of list before processing: {len(df)}')
        # include_groups=False avoids DeprecationWarning in pandas >= 2.0
        # where groupby columns are passed to the function
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', message='.*DataFrameGroupBy.*')
            df = df.groupby('HHID', group_keys=False).apply(update_rlhp)
        logging.info(f'length of list after processing: {len(df)}')
        df['rlhp'] = df['RELATIONSHIP'].map(rl_map).fillna(5).astype(int)
        df['STUDYING'] = df['STUDYING'].map(study_map).fillna(0).astype(int)
        df = add_person_binary_dummies(df)
        return df
    
    def synth_filter_function(self, persons, filter_function):
        if filter_function is not None: #
            mask = filter_function(persons)
            # where filter function is the persons_data fram with person['STUDYING'].isin([1,2,3,4])
            persons = persons[mask]
        return persons
    
    def synth_data_pass(self, choice_id_set, persons, filter_function):
        '''Fully Pass in Dataframe'''
        persons['AGEGROUP'] = persons['age5p']
        
        persons['PERSID'] = persons['abspid']  # Ensure PERSID starts from 1
        persons['person_id'] = persons['PERSID']
        persons['STUDYING'] = persons.apply(get_studying, axis=1)
        persons['WORKSTAT'] = persons.apply(get_workstat, axis=1)
        persons['CARLICENCE'] = persons.apply(get_licence, axis=1)
        persons = add_person_binary_dummies(persons)
        persons['chosen'] = 1 #dummy
        if filter_function is not None: #
            mask = filter_function(persons)
            # where filter function is the persons_data fram with person['STUDYING'].isin([1,2,3,4])
            persons = persons[mask]
        
        persons['case_id'] = persons.index
        #tours['case_id'] = tours['PERSID'].astype(str) + "_" + tours['tour_id'].astype(str)
        persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        return persons_ds
    
    def synth_data_pass_2(self, choice_id_set, persons):
        '''Fully Pass in Dataframe'''
        
        #tours['case_id'] = tours['PERSID'].astype(str) + "_" + tours['tour_id'].astype(str)
        persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        return persons_ds
    
    
    def propose_schedule_changes_duration(self):
        '''place holder function'''
        trips_df, df = self.propose_shedule_changes()
        
        #durations
        dataset_duration = 'saved_data/trip_duration_synthetic_population.csv'
        duration = pd.read_csv(dataset_duration)
        trips_df = trips_df.merge(duration[['person_id', 'chosen_trip_duration', 'trip_seq', 'trip_id', 'DESTPURP']], on = ['person_id', 'trip_seq', 'DESTPURP', 'trip_id'], how='left')
        print(duration)
        return trips_df, df
        #then based on time and duration fill out schedule..
        # merge, 
        #from duration_fill_out
         
       
    
    
    def propose_shedule_changes(self):
        dataset = 'saved_data/ativities_time_synthetic_population.csv'
        persons = pd.read_csv(dataset)
        
        #dataset_actives = 'saved_data/ativities_locations_synthetic_population.csv'
        #actives = pd.read_csv(dataset_actives)
         
        # merge persons and activies..
        # then with mergd based on the activity counts generate a sequence final sequence being the DTAZ in persons, 
        # convert the first person to trips..
        ''' in each record we have this
        n_escort_out	n_shopping_out	n_othmaint_out	n_othdiscr_out	n_eat_out	n_social_out	n_work_out	n_edu_out
        '''
        # based on the Nsss. thats how many trips we have. persons homezone represents, origin zone, persons DTAZ represents final destination zone, then from remaining trips just put -1 as placeholds for OTAZ and DTAZ
        # then generate trip ids based on this.
        
                # columns in actives that are NOT already in persons
        #cols_to_add = actives.columns.difference(persons.columns)

        # keep the join key
       # cols_to_add = cols_to_add.union(['case_id'])
        '''
        df = persons.merge(
            actives[cols_to_add],
            on='case_id',
            how='left'
        )
        
        '''
        df = persons.copy()
        if 'person_id' not in df.columns:
            df = df.rename(columns={'PERSID':'person_id'})
        df['ptour_id'] = df['person_id'].astype(str) + '_' + df.index.astype(str)

        # -----------------------------
        # Activity count columns
        # -----------------------------
        activity_cols = {
            
            'escort': 'n_escort_out',
            'shopping': 'n_shopping_out',
            'othmaint': 'n_othmaint_out',
            'othdiscr': 'n_othdiscr_out',
            'eat': 'n_eat_out',
            'social': 'n_social_out',
            'work': 'n_work_out',
            'edu': 'n_edu_out'
        }

        df[list(activity_cols.values())] = (
            df[list(activity_cols.values())]
            .fillna(0)
            .astype(int)
        )

        df['n_trips'] = df[list(activity_cols.values())].sum(axis=1)
        df['num_trips'] = df['n_trips']
        # ensure missing values don't break counts
        #df[activity_cols] = df[activity_cols].fillna(0).astype(int)
        if 'person_id' not in df.columns:
            df = df.rename(columns={'case_id':'person_id'})
        # total number of trips per person
        #df['n_trips'] = df[activity_cols].sum(axis=1)
        # -----------------------------
        # Helper: build purpose list
        # -----------------------------
        def build_purpose_sequence(row):
            purposes = []
            for purp, col in activity_cols.items():
                purposes.extend([purp] * row[col])
            return purposes

        # -----------------------------
        # Build trips
        # -----------------------------
        trips = []

        for _, row in df.iterrows():

            person_id = row['person_id']
            n_trips = row['n_trips']

            if n_trips == 0:
                continue
            
            
            purposes = build_purpose_sequence(row)
            main_purpose = 'work' if 'work' in purposes else (
                'edu' if 'edu' in purposes else random.choice(purposes)
            )
            for t in range(n_trips+1):

                if t == 0:
                    # first trip
                    otaz = row['OTAZ']
                    dtaz = -1
                    if n_trips ==0:
                        dtaz = row['DTAZ']
                    destpurp = purposes[t]

                elif t == n_trips:
                    # final trip
                    if t ==0:
                        otaz = row['OTAZ']
                    else:
                        otaz = -1
                    dtaz = row['DTAZ']
                    dest_purp = main_purpose

                else:
                    # intermediate trips
                    otaz = -1
                    dtaz = -1
                    destpurp = purposes[t]

                trips.append({
                    'person_id': person_id,
                    'trip_seq': t + 1,
                    'OTAZ': otaz,
                    'TTAZ': row['DTAZ'],
                    'STARTSTOP': t+1,
                    'STARTIME' :row['start_of_day'],
                    'ENDTIME': row['end_of_day'],
                    'MAINPURP': main_purpose,
                    'DESTPURP': destpurp,# needs to be purpoose based on n_trips. the final trip is alway the gloabl purpse 
                    'DTAZ': dtaz,
                    'ptour_id': row['ptour_id'],
                    'ntrips': n_trips,
                    'mode': row['tour_mode']
                })

        trips_df = pd.DataFrame(trips)

        # -----------------------------
        # Generate trip IDs
        # -----------------------------
        trips_df['trip_id'] = trips_df['person_id'].astype(str) + '_' + trips_df['trip_seq'].astype(str)
        trips_df["OTAZ"] = trips_df["OTAZ"].astype(int)
        trips_df["DTAZ"] = trips_df["DTAZ"].astype(int)

        # optional ordering
        trips_df = trips_df.sort_values(['person_id', 'trip_seq'])
        trips_df['case_id'] = trips_df.index
       
        return trips_df, df
    
    
    def synth_data_test_stage_6(self, choice_id_set, dataset = 'saved_data/activities_locations_trips.csv', filter_function =None):
        persons, tours_p = self.propose_shedule_changes()
        persons['mode'] = tours_p['tour_mode']
        tours_p  = pd.read_csv(dataset)
        persons['DURATION'] = (persons['ENDTIME'] - persons['STARTIME'])*60
        persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        return persons_ds, persons, tours_p
        
    def synth_data_test_stage_4(self, choice_id_set, dataset = 'saved_data/ativities_time_synthetic_population.csv', filter_function =None):
        logging.info('copy next stage')
        persons, tours_p = self.propose_shedule_changes()
        #dataset_duration = pd.read_csv(dataset)['tour_mode']
        #persons['mode'] = dataset_duration
        persons['MODE1'] = -1
        persons['DURATION'] = (persons['ENDTIME'] - persons['STARTIME'])*60
       # dataset_duration['DURATION'] = (dataset_duration['end_of_day'].astype(int) - dataset_duration['start_of_day'].astype(int))*60
        #dataset_duration = dataset_duration.rename(columns={'PERSID':'person_id'})
        #persons = persons.merge(dataset_duration, on='person_id', how='left')
        
        
        
        
        #tours['case_id'] = tours['PERSID'].astype(str) + "_" + tours['tour_id'].astype(str)
        persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        persons['mode'] = -1
        return persons_ds, persons, tours_p
    
    def synth_data_test_stage_5(self, choice_id_set, dataset = 'saved_data/ativities_time_synthetic_population.csv', filter_function =None):
        logging.info('copy next stage')
        persons, tours_p = self.propose_schedule_changes_duration()
        persons['DURATION'] = persons['chosen_trip_duration'] *60
        
        
        
        
        #tours['case_id'] = tours['PERSID'].astype(str) + "_" + tours['tour_id'].astype(str)
        persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        persons['mode'] = -1
        return persons_ds, persons, tours_p
    
    def connected_data_syth(self, choice_id_set):
        logging.info('copy next stage')
        persons, tours_p = self.propose_shedule_changes()
        #person_g = pd.merge(persons, tours_p, on='person_id', how='left')
        # how to merger without making double cases like DTAZ_x and DTAZ_y
        dup_cols = persons.columns.intersection(tours_p.columns)
        dup_cols = dup_cols.drop('person_id')

        tours_clean = tours_p.drop(columns=dup_cols)

        person_g = persons.merge(
            tours_clean,
            on='person_id',
            how='left'
        )
        return person_g
        
    
    
    def synth_data_test_stage_3(self, choice_id_set, dataset = 'saved_data/ativities_time_synthetic_population.csv', filter_function =None):
        logging.info('copy next stage')
        persons = pd.read_csv(dataset)
        persons['AGEGROUP'] = persons['age5p']
        
        persons['PERSID'] = persons['abspid']  # Ensure PERSID starts from 1
        persons['person_id'] = persons['PERSID']
        persons['STUDYING'] = persons.apply(get_studying, axis=1)
        persons['WORKSTAT'] = persons.apply(get_workstat, axis=1)
        persons['CARLICENCE'] = persons.apply(get_licence, axis=1)
        persons = add_person_binary_dummies(persons)
        persons['HHSIZE'] = persons['nprd']
        persons['chosen'] = 1 #dummy
        
        # better way to do this ..
        

        persons['main_purpose'] = np.select(
            [
                persons['n_work_out'] > 0,
                persons['n_edu_out'] > 0
            ],
            [
                'work',
                'edu'
            ],
            default='othdiscr'
        )
        
        
        
        '''
        persons = persons.rename(columns ={
        'edu': 'n_edu_out', 'social': 'n_social_out', 'eatout': 'n_eat_out', 'shopping': 'n_shopping_out',
        'othmaint': 'n_othmaint_out', 'othdiscr': 'n_othdiscr_out', 'escort': 'n_escort_out', 'work': 'n_work_out'
        })
        '''
        if filter_function is not None: #
            mask = filter_function(persons)
            # where filter function is the persons_data fram with person['STUDYING'].isin([1,2,3,4])
            persons = persons[mask]
        '''
        persons['key'] = 1
        
        choice_id_set['key'] = 1

        persons = persons.merge(choice_id_set.rename(columns = {'escort':'num_escort_total', 'shopping':'num_shopping_total', 'othmaint':'num_othermaint_total',
                                                             'othdiscr':'num_otherdiscr_total', 'eatout':'num_eat_total', 'social':'num_social_total', 'tot_tours': 'num_total_activities'}), on='key').drop(columns='key')
        '''
        #logging
        #persons = -> give person all attirbutes from ->choice_id_set_df
        persons['case_id'] = persons.index
        #persons['choice_id'] = -1
        #persons['DTAZ'] = -int(1)
        persons['OTAZ'] = persons['OTAZ'].astype(int)
        #tours['case_id'] = tours['PERSID'].astype(str) + "_" + tours['tour_id'].astype(str)
        persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        return persons_ds, persons
    
    
    def synth_data_test_stage_2(self, choice_id_set, dataset = 'saved_data/persons_with_activity.csv', filter_function =None):
        logging.info('copy next stage')
        persons = pd.read_csv(dataset)
        persons['AGEGROUP'] = persons['age5p']
        
        persons['PERSID'] = persons['abspid']  # Ensure PERSID starts from 1
        persons['person_id'] = persons['PERSID']
        persons['STUDYING'] = persons.apply(get_studying, axis=1)
        persons['WORKSTAT'] = persons.apply(get_workstat, axis=1)
        persons['CARLICENCE'] = persons.apply(get_licence, axis=1)
        persons = add_person_binary_dummies(persons)
        persons['HHSIZE'] = persons['nprd']
        persons['chosen'] = 1 #dummy
        persons = persons.rename(columns ={
        'edu': 'n_edu_out', 'social': 'n_social_out', 'eatout': 'n_eat_out', 'shopping': 'n_shopping_out',
        'othmaint': 'n_othmaint_out', 'othdiscr': 'n_othdiscr_out', 'escort': 'n_escort_out', 'work': 'n_work_out'
        })
        if filter_function is not None: #
            mask = filter_function(persons)
            # where filter function is the persons_data fram with person['STUDYING'].isin([1,2,3,4])
            persons = persons[mask]
        '''
        persons['key'] = 1
        
        choice_id_set['key'] = 1
        
        persons = persons.merge(choice_id_set.rename(columns = {'escort':'num_escort_total', 'shopping':'num_shopping_total', 'othmaint':'num_othermaint_total',
                                                             'othdiscr':'num_otherdiscr_total', 'eatout':'num_eat_total', 'social':'num_social_total', 'tot_tours': 'num_total_activities'}), on='key').drop(columns='key')
        
        '''
        persons = persons[persons['ACT'] != 'H']
        #logging
        #persons = -> give person all attirbutes from ->choice_id_set_df
        persons['case_id'] = persons.index
        persons['choice_id'] = -1
        #tours['case_id'] = tours['PERSID'].astype(str) + "_" + tours['tour_id'].astype(str)
        if type(choice_id_set) != list:
            
            persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        else:
            persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set)
            
        return persons_ds, persons
    
    
    
    def synth_data_with_M(self, choice_id_set, activity_dataset = 'M'):
        dataset = 'saved_data/persons_with_activity.csv'
        persons = pd.read_csv(dataset)
        persons['AGEGROUP'] = persons['age5p']
        
        persons['PERSID'] = persons['abspid']  # Ensure PERSID starts from 1
        persons['person_id'] = persons['PERSID']
        persons['STUDYING'] = persons.apply(get_studying, axis=1)
        persons['WORKSTAT'] = persons.apply(get_workstat, axis=1)
        persons['CARLICENCE'] = persons.apply(get_licence, axis=1)
        persons = add_person_binary_dummies(persons)
        persons['HHSIZE'] = persons['nprd']
        persons['chosen'] = 1 #dummy
        persons['tour_mode'] = 'DA'
        persons['ACTC'] = -1
        persons['DTAZ'] = 1
        persons = persons.rename(columns ={
        'edu': 'n_edu_out', 'social': 'n_social_out', 'eatout': 'n_eat_out', 'shopping': 'n_shopping_out',
        'othmaint': 'n_othmaint_out', 'othdiscr': 'n_othdiscr_out', 'escort': 'n_escort_out', 'work': 'n_work_out'
        })
        if activity_dataset == 'M':
            persons = persons[persons['ACT'] != 'H']
            persons = persons[persons['ACT'] == 'M']
        elif activity_dataset == 'N':
            persons = persons[persons['ACT'] != 'H']
            persons = persons[persons['ACT'] == 'N']
            
        '''
        
        persons['key'] = 1
        
        choice_id_set['key'] = 1

        persons = persons.merge(choice_id_set.rename(columns = {'escort':'num_escort_total', 'shopping':'num_shopping_total', 'othmaint':'num_othermaint_total',
                                                             'othdiscr':'num_otherdiscr_total', 'eatout':'num_eat_total', 'social':'num_social_total', 'tot_tours': 'num_total_activities'}), on='key').drop(columns='key')
        '''
        #persons = -> give person all attirbutes from ->choice_id_set_df
        persons['outbound_stops'] = persons[['n_escort_out', 'n_shopping_out', 'n_othmaint_out', 'n_othdiscr_out', 'n_eat_out', 'n_social_out', 'n_work_out', 'n_edu_out']].sum(axis=1)
        persons['case_id'] = persons.index
        #tours['case_id'] = tours['PERSID'].astype(str) + "_" + tours['tour_id'].astype(str)
        choice_id_set
        if type(choice_id_set) == Dict:
            persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set)
        
        else:
            persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        return persons_ds, persons
    
    def synth_data_test(self, choice_id_set, data_set = "Z:/sa_run_again_n/zones_sa_4_copy/30901122736_synthetic_population.csv", filter_function =None):
        ### SYNTETHIC POPULATION ###
        # 1. Synthetic people
        # this is just one file. how do I process it so i can do multiple files... where each additioanl file, by the code {number}_synthetic_population
        '''Now from zones'''
        OLD = 0
        
        if OLD:
            try:
                
                persons = pd.read_csv(data_set)
            except Exception as e:
                print('could not access hpc')
                persons = pd.read_csv('saved_data/zones_csv/30901122736_synthetic_population.csv')
                
            match = re.search(r"(\d+)_synthetic_population\.csv", data_set)
            if match:
                pop_id = int(match.group(1))
                if len(str(pop_id)) ==11:
                    persons['SA1'] = int(pop_id)
                    sa1_to_sa2 = pd.read_csv('sa1_2021.csv')[['SA1_CODE21', 'SA2_CODE21']]
                    m = sa1_to_sa2.set_index('SA1_CODE21')['SA2_CODE21']
                    sa2_to_order = pd.read_csv('seq_zone_ids_.csv')[['SA2_MAIN16', 'ORDER']]
                    o = sa2_to_order.set_index('SA2_MAIN16')['ORDER']
                    persons['HOME_SA2_2021'] = persons['SA1'].apply(lambda x: m.get(x))
                    persons['OTAZ'] = persons['HOME_SA2_2021'].apply(lambda x: o.get(x))
                    #persons['OTAZ'] = persons['SA1'].map(sa1_to_o)
                elif len(pop_id) ==9:
                    persons['SA2'] = int(pop_id)
        else:
            persons = pd.read_csv('saved_data/zone_ids_sa2/order_40.csv')
            persons['OTAZ'] = persons['ORDER'].astype(int)
            persons['HOME_SA2_2021'] = persons['SA2_MAIN16'].astype(int)
        
        #get number::
        #excract number form this file name'''

        
        persons['AGEGROUP'] = persons['age5p']
        
        persons['PERSID'] = persons['abspid']  # Ensure PERSID starts from 1
        persons['STUDYING'] = persons.apply(get_studying, axis=1)
        persons['WORKSTAT'] = persons.apply(get_workstat, axis=1)
        persons['CARLICENCE'] = persons.apply(get_licence, axis=1)
        persons = add_person_binary_dummies(persons)
        persons['HHSIZE'] = persons['nprd']
        persons['chosen'] = 1 #dummy
        persons['tour_mode'] = 'DA'
        persons['ACTC'] = -1
        persons['DTAZ'] = 1
        if filter_function is not None: #
            mask = filter_function(persons)
            # where filter function is the persons_data fram with person['STUDYING'].isin([1,2,3,4])
            persons = persons[mask]
        '''
        persons['key'] = 1
        
        choice_id_set['key'] = 1

        persons = persons.merge(choice_id_set.rename(columns = {'escort':'num_escort_total', 'shopping':'num_shopping_total', 'othmaint':'num_othermaint_total',
                                                             'othdiscr':'num_otherdiscr_total', 'eatout':'num_eat_total', 'social':'num_social_total', 'tot_tours': 'num_total_activities'}), on='key').drop(columns='key')
        '''
        #persons = -> give person all attirbutes from ->choice_id_set_df
        persons['case_id'] = persons.index
        #tours['case_id'] = tours['PERSID'].astype(str) + "_" + tours['tour_id'].astype(str)
        choice_id_set
        if type(choice_id_set) == Dict:
            persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set)
        
        else:
            persons_ds = lx.Dataset.construct.from_idco(persons.set_index('case_id'), alts=choice_id_set.index.values)
        return persons_ds, persons
        
        
        
    def _process_single_population_file(
                                    self,
                                    data_set,
                                    choice_id_set,
                                    filter_function=None):
        try:
            persons = pd.read_csv(data_set)
        except Exception:
            print(f'Could not access {data_set}, trying fallback')
            persons = pd.read_csv(
                f'saved_data/zones_csv/{Path(data_set).name}'
            )

        # extract number from filename
        match = re.search(r"(\d+)_synthetic_population\.csv", str(data_set))
        if match:
            pop_id = match.group(1)

            if len(pop_id) == 11:
                pop_id = int(pop_id)
                persons['SA1'] = pop_id

                sa1_to_sa2 = pd.read_csv('sa1_2021.csv')[['SA1_CODE21', 'SA2_CODE21']]
                m = sa1_to_sa2.set_index('SA1_CODE21')['SA2_CODE21']

                sa2_to_order = pd.read_csv('seq_zone_ids_.csv')[['SA2_MAIN16', 'ORDER']]
                o = sa2_to_order.set_index('SA2_MAIN16')['ORDER']

                persons['HOME_SA2_2021'] = persons['SA1'].map(m)
                persons['OTAZ'] = persons['HOME_SA2_2021'].map(o)

            elif len(pop_id) == 9:
                persons['SA2'] = int(pop_id)

        # attributes
        persons['AGEGROUP'] = persons['age5p']
        persons['PERSID'] = persons['abspid']
        persons['STUDYING'] = persons.apply(get_studying, axis=1)
        persons['WORKSTAT'] = persons.apply(get_workstat, axis=1)
        persons['CARLICENCE'] = persons.apply(get_licence, axis=1)
        persons['HHSIZE'] = persons['nprd']
        persons['chosen'] = 1
        persons['tour_mode'] = 'DA'
        persons['DTAZ'] = 1

        if filter_function is not None:
            persons = persons[filter_function(persons)]

        persons['case_id'] = persons.index

        persons_ds = lx.Dataset.construct.from_idco(
            persons.set_index('case_id'),
            alts=choice_id_set.index.values
        )

        return persons_ds, persons
    
    def synth_data_test_batch(
        self,
        choice_id_set,
        data_dir="Z:/sa_run_again_n/zones_sa_4_copy/",
        filter_function=None
        ):
        data_dir = Path(data_dir)

        files = sorted(
            data_dir.glob("*_synthetic_population.csv")
        )

        if not files:
            raise FileNotFoundError("No synthetic population files found")

        all_datasets = []
        all_persons = []

        for file_path in files:
            print(f"Processing {file_path.name}")

            persons_ds, persons = self._process_single_population_file(
                data_set=file_path,
                choice_id_set=choice_id_set,
                filter_function=filter_function
            )

            all_datasets.append(persons_ds)
            all_persons.append(persons)

        return all_datasets, all_persons



if __name__ == '__main__':
    '''TESTING'''
    
    ## TEST FUCNTION 1
    DEBUG = 0
    if DEBUG:
        choice_full = pd.read_csv('alternative_configs/non_mandatory_tour_frequency_alternatives.csv')
        aa, bb = Synth_Connect().synth_data_test_batch(choice_full)
    
    
    
    ##### TEST FUNCTIONS 
    a = Synth_Connect().propose_schedule_changes_duration()
    b = Synth_Connect().update_files()
    a = Synth_Connect().propose_shedule_changes()
    print(a)
    print('done')
    