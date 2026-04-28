import pandas as pd
import logging
from addicty import Dict
import os
import sys
import math
import openmatrix as omx
import larch as lx
import numpy as np
from data_connect import Synth_Connect
from larch import P, X

def add_time_dummies(df, start_col="start_of_day", end_col="end_of_day"):
    """
    Add binary time-of-day dummy variables for departure (start) and arrival (end) times.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing start and end columns.
    start_col : str
        Column name for start time (integer hours).
    end_col : str
        Column name for end time (integer hours).

    Returns
    -------
    pd.DataFrame
        Original dataframe with binary (0/1) dummy columns added.
    """

    # Copy so we don't overwrite
    df = df.copy()

    # --- Departure (Start) Dummies ---
    df["util_early_start_at_5"]             = (df[start_col] < 6).astype(int)
    df["util_am_peak_start_at_6"]           = (df[start_col] == 6).astype(int)
    df["util_am_peak_start_at_7"]           = (df[start_col] == 7).astype(int)
    df["util_am_peak_start_at_8"]           = (df[start_col] == 8).astype(int)
    df["util_am_peak_start_at_9"]           = (df[start_col] == 9).astype(int)
    df["util_midday_start_at_10_11_12"]     = ((df[start_col] > 9) & (df[start_col] < 13)).astype(int)
    df["util_midday_start_at_13_14_15"]     = ((df[start_col] > 12) & (df[start_col] < 16)).astype(int)
    df["util_pm_peak_start_at_16_17_18"]    = ((df[start_col] > 15) & (df[start_col] < 19)).astype(int)
    df["util_evening_start_at_19_20_21"]    = ((df[start_col] > 18) & (df[start_col] < 22)).astype(int)
    df["util_late_start_at_22_23"]          = (df[start_col] > 21).astype(int)

    # --- Arrival (End) Dummies ---
    df["util_early_end_at_5_6"]             = (df[end_col] < 7).astype(int)
    df["util_am_peak_end"]                  = ((df[end_col] > 6) & (df[end_col] < 10)).astype(int)
    df["util_midday_end_at_10_11_12"]       = ((df[end_col] > 9) & (df[end_col] < 13)).astype(int)
    df["util_midday_end_at_13_14"]          = ((df[end_col] > 12) & (df[end_col] < 15)).astype(int)
    df["util_pm_peak_end_at_15"]            = (df[end_col] == 15).astype(int)
    df["util_pm_peak_end_at_16"]            = (df[end_col] == 16).astype(int)
    df["util_pm_peak_end_at_17"]            = (df[end_col] == 17).astype(int)
    df["util_pm_peak_end_at_18"]            = (df[end_col] == 18).astype(int)
    df["util_evening_end_at_19_20_21"]      = ((df[end_col] > 18) & (df[end_col] < 22)).astype(int)
    df["util_late_end_at_22_23"]            = (df[end_col] > 21).astype(int)
    
    
    df['early_start']= (df[start_col] <6).astype(int)
    df['am_peak_start']= ((df[start_col] >=6) & (df[start_col] <=9)).astype(int)
    df['midday_start']= ((df[start_col] >=10) & (df[start_col] <=15)).astype(int)
    df['pm_peak_start']= ((df[start_col] >=16) & (df[start_col] <=18)).astype(int)
    df['evening_start']= ((df[start_col] >=19) & (df[start_col] <=21)).astype(int)
    df['late_start']= (df[start_col] >=22).astype(int)  
    df['early_end']= (df[end_col] <6).astype(int)
    df['am_peak_end']= ((df[end_col] >=6) & (df[end_col] <=9)).astype(int)
    df['midday_end']= ((df[end_col] >=10) & (df[end_col] <=15)).astype(int)
    df['pm_peak_end']= ((df[end_col] >=16) & (df[end_col] <=18)).astype(int)
    df['evening_end']= ((df[end_col] >=19) & (df[end_col] <=21)).astype(int)
    df['late_end']= (df[end_col] >=22).astype(int)
            
    

    return df


def time_hour(time):
    '''returns the time of starting..'''
    return int(time/60) +4

def get_start_dummy_name(start):
    if start < 6: return "util_early_start_at_5"
    elif start == 6: return "util_am_peak_start_at_6"
    elif start == 7: return "util_am_peak_start_at_7"
    elif start == 8: return "util_am_peak_start_at_8"
    elif start == 9: return "util_am_peak_start_at_9"
    elif 10 <= start <= 12: return "util_midday_start_at_10_11_12"
    elif 13 <= start <= 15: return "util_midday_start_at_13_14_15"
    elif 16 <= start <= 18: return "util_pm_peak_start_at_16_17_18"
    elif 19 <= start <= 21: return "util_evening_start_at_19_20_21"
    elif start >= 22: return "util_late_start_at_22_23"
    else: return None

def get_period(time, start = 'start'):
    if time <6: return f"early_{start}"
    elif 6 <= time <=9: return f"am_peak_{start}"
    elif 10 <= time <=15: return f"midday_{start}"
    elif 16 <= time <=18: return f"pm_peak_{start}"
    elif 19 <= time <=21: return f"evening_{start}"
    elif time >=22: return f"late_{start}"
    else: return None


def get_end_dummy_name(end):
    if end < 7: return "util_early_end_at_5_6"
    elif 7 <= end <= 9: return "util_am_peak_end"
    elif 10 <= end <= 12: return "util_midday_end_at_10_11_12"
    elif 13 <= end <= 14: return "util_midday_end_at_13_14"
    elif end == 15: return "util_pm_peak_end_at_15"
    elif end == 16: return "util_pm_peak_end_at_16"
    elif end == 17: return "util_pm_peak_end_at_17"
    elif end == 18: return "util_pm_peak_end_at_18"
    elif 19 <= end <= 21: return "util_evening_end_at_19_20_21"
    elif end >= 22: return "util_late_end_at_22_23"
    else: return None

def get_is_worker(df, workstatus_col="WORKSTATUS"):
    """
    Tag whether a person is a worker.
    WORKSTATUS convention:
      1 = employed
      0 = unemployed / not working
    """
    df = df.copy()
    df["is_worker"] = (df[workstatus_col] == 1).astype(int)
    return df

    
def sort_pairs(row):
    # Combine into pairs, sort by start time, unzip back
    pairs = sorted(zip(row["start_time_hour"], row["end_time_hour"]))
    starts, ends = zip(*pairs)
    row["start_time_hour"] = list(starts)
    row["end_time_hour"] = list(ends)
    return row

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


def synt_data(alternatives):
      ### SYNTETHIC POPULATION ###
    # 1. Synthetic people
    persons = pd.read_csv("Z:/sa_run_again_n/zones_sa_4_copy/30901122736_synthetic_population.csv")
    persons['STUDYING'] = persons.apply(get_studying, axis=1)
    persons['WORKSTAT'] = persons.apply(get_workstat, axis=1)
    #persons['STUDYING'] = persons['typp'].map({1: 1, 2: 1, 3: 2, 4:2}).fillna(1)    
    ''' this is typp Primary - Non Government
        Secondary - Non-Government
        Tertiary - Non University
        Preschool
        Primary - Government
        Secondary - Government
        Tertiary - University or other higher education
        Other
        Not stated
        Not applicable
        Overseas visitor
        '''
    #this is the styup staturs, 
    '''Not Stated G
        Not attending
        Full-time student
        Part-time student
        Overseas visitor
        '''
    # us stuo us Not attending not stated or oversears visitor automaticall applt STUDING TO 0
    # else if primary assing 1
    
    
    #from these catreogies i ened to make the studying variable    
        
    
    
    logging.debug(f"Persons columns: {persons.columns.tolist()}")
    persons['person_id'] = persons['abspid']
    #persons['choice_id'] = 5
    # 2. Choice set from your model
    #alternatives = pd.read_csv("choice_set.csv")
    alternatives_= add_time_dummies(alternatives, "start_of_day", "end_of_day")
    choice_id_set = alternatives_["choice_id"].tolist()

    # 3. Link data via a common key
    # Larch can construct this via "cartesian join" automatically
    choice_ids_dict = pd.DataFrame({'alt_key': choice_id_set})

# Cartesian merge – every person with every alternative
    persons_long = persons.assign(key=1).merge(choice_ids_dict.assign(key=1), on='key').drop('key', axis=1)
    a = persons_long.merge(alternatives_, left_on='alt_key', right_on='choice_id', how='left')
    person_ds = lx.Dataset.construct.from_idco(a.set_index('person_id'), alts=choice_id_set)
    #person_ds = lx.Dataset.construct.from_df(persons.set_index("person_id"))
    alt_ds = lx.Dataset.construct.from_idco(alternatives_.set_index("choice_id"),
                                            alts=choice_id_set)

    # Build the DataTree (cartesian product relationship)
    dt = lx.DataTree(
        person=person_ds,
        #alt=alt_ds,
        #relationships=(
       # "person._altid_ @ alt._altid_",
       # ),
        )
    
    
   # dt = lx.DataTree.auto(person_ds, alt_ds)
    
    print("Datasets linked.")
    return dt
    






def read_in():
    
   
    
    from stage1_activity_frequency import run_stage1
    from data_connect import Synth_Connect, get_workstat, get_studying

    # --- survey side ---
    out   = Synth_Connect().process_survey(save=False)

    # --- synthetic population ---
    synth = pd.read_csv('saved_data/zone_ids_sa2/order_40.csv')
    synth['WORKSTAT'] = synth.apply(get_workstat, axis=1)
    synth['STUDYING'] = synth.apply(get_studying, axis=1)
    #synth['vehrd']    = synth['HHVEH'].clip(upper=4)
    synth['sexp']     = synth['sexp']
    synth['nprd']     = synth['nprd']

    synth_pred, models = run_stage1(out["tours"], out["persons"], synth)
    synth_pred.to_csv('saved_data/stage1_output.csv', index=False)
    
    
    from stage2_day_duration import run_stage2

    out      = Synth_Connect().process_survey(save=False)
    synth_s1 = pd.read_csv('saved_data/stage1_output.csv')

    synth_s2, models = run_stage2(out["tours"], out["persons"], synth_s1)
    synth_s2.to_csv('saved_data/stage2_output.csv', index=False)
    
    
    
    from stage3_activity_duration import run_stage3

    out      = Synth_Connect().process_survey(save=False)
    synth_s2 = pd.read_csv('saved_data/stage2_output.csv')

    activities, models = run_stage3(out["tours"], out["persons"], synth_s2, trips=out["trips"])
    activities.to_csv('saved_data/stage3_output.csv', index=False)
    
    
    
    from stage4_schedule_placement import run_stage4

    out      = Synth_Connect().process_survey(save=False)
    synth_s3 = pd.read_csv('saved_data/stage3_output.csv')

    scheduled, models = run_stage4(
        out["trips"], out["tours"], out["persons"], synth_s3
    )
    scheduled.to_csv('saved_data/stage4_output.csv', index=False)
    
    from stage5_destination_choice import run_stage5

    sc       = Synth_Connect()
    out      = sc.process_survey(save=False)
    synth_s4 = pd.read_csv('saved_data/stage4_output.csv')
    
    
    
    import openmatrix as omx
    from read_skims import read_omx_skims, list_omx_matrices

    # first — check what matrices are in your file
    list_omx_matrices('downcasted_matrices_base.omx')
    
    #skims    = pd.read_csv('OD_skims.csv')      # needs OTAZ, DTAZ, travel_time, distance
    landuse  = sc.get_land()                    # already in your Synth_Connect
    # option A — pass the OMX path directly (stage5 handles it)
    scheduled_dest, models = run_stage5(
        out["trips"], out["persons"], synth_s4,
        skims='downcasted_matrices_base.omx',   # ← path string
        landuse=landuse,
    )
    scheduled_dest, models = run_stage5(
        out["trips"], out["persons"], synth_s4, skims, landuse
    )
    scheduled_dest.to_csv('saved_data/stage5_output.csv', index=False)
    
    from stage6_mode_choice          import run_stage6
    plan,_ = run_stage6(out["trips"], out["persons"], scheduled_dest, landuse)
    from plot_activity_plans import plot_all
    plot_all(plan, n_sample=8, save_dir='saved_data/plots')
    '''
    from tour_duration_model import (
    aggregate_to_person_day, run_tour_duration_model, predict_synthetic
    )
    out        = Synth_Connect().process_survey(save=False)
    person_day = aggregate_to_person_day(out["tours"])
    tours_pred, res, feats = run_tour_duration_model(out["tours"], person_day)

    # --- apply to synthetic population ---
    synth      = pd.read_csv('saved_data/ativities_time_synthetic_population.csv')
    synth_pred = predict_synthetic(res.x, synth, feats)
    
    '''
    #Synth_Connect().trips_to_tours(1, 1)
    #Synth_Connect().process_survey()
    #Synth_Connect.process_purpose()
    Synth_Connect().trips_to_tours()
    a,b,c, d =Synth_Connect().linked_stops_to_tours()
    df = Synth_Connect().read_in_tours()
    ##### ACTUIL OBSERVED DATASET #####
    #df = pd.read_csv('tours_dataset.csv')
    
    # that's not quite right, i need to group so that first instance of a per
    print(df.head())
    # clean
    df['duration'] = df['duration'].fillna(0)
    #df.loc[df['end_time'].isna(), 'end_time'] = df['arr_time'] + df['duration']
    # Step 2 — confirm each person's unique choice set
    df['start_time_hour'] = df['start_time'].apply(time_hour)
    df['end_time_hour'] = df['end_time'].apply(time_hour)
    person_choices = (
    df.sort_values(["person_id", "start_time_hour", "end_time_hour"])  # optional pre‑sort
      .groupby("person_id")[["start_time_hour", "end_time_hour"]]
      .agg(list)
      .reset_index()
    )
    
    daily_span = (
    df.groupby("person_id")[["start_time_hour", "end_time_hour", "n_eat_out", 'n_edu_out', 'n_social_out', 'tour_id', 'n_escort_out', 'n_shopping_out', 'n_othdiscr_out', 'n_othmaint_out', 'n_work_out' ]]
      .agg(start_of_day=("start_time_hour", "min"),
           end_of_day=("end_time_hour", "max"),
           n_eat_out = ('n_eat_out', 'sum'),
           n_othdiscr_out = ('n_othdiscr_out', 'sum'),
           n_othmaint_out = ('n_othmaint_out', 'sum'),
           n_edu_out = ('n_eat_out', 'sum'),
           n_escort_out = ('n_escort_out', 'sum'),
           n_work_out = ('n_work_out', 'sum'),
           n_social_out = ('n_social_out', 'sum'),
           n_shopping_out = ('n_shopping_out', 'sum'),
           n_tours = ('tour_id', 'max')
           
           
           
           )
      .reset_index()
    )

    print(daily_span)

    # Add helper columns for sorting each person by their first (earliest) time
    #person_choices["first_start"] = person_choices["start_time_hour"].apply(min)
    #person_choices["last_end"]   = person_choices["end_time_hour"].apply(max)

    # Now sort *people* (the rows) by earliest start, then earliest end
    person_choices = daily_span.sort_values(["start_of_day", "end_of_day"]).reset_index(drop=True)
    print(person_choices)
    # Optional: drop helper columns
    #person_choices = person_choices.drop(columns=["first_start", "first_end"])
    observed_ranges = person_choices[['start_of_day', 'end_of_day']].drop_duplicates().sort_values(['start_of_day', 'end_of_day'])
    realistic_ranges = observed_ranges.query("start_of_day < end_of_day")
    realistic_ranges = realistic_ranges[
    (realistic_ranges.end_of_day - realistic_ranges.start_of_day) >= 1  # e.g. max 2‑week span
    ]
    realistic_ranges = realistic_ranges[
    (realistic_ranges.end_of_day - realistic_ranges.start_of_day) <= 18  # e.g. max 2‑week span
    ]
    
    realistic_ranges = realistic_ranges[
    (realistic_ranges.start_of_day ) <= 21  # e.g. max start day of 21
    ]
    realistic_ranges = realistic_ranges[
    (realistic_ranges.end_of_day ) <= 23  # e.g. max start day of 21
    ]
    print(realistic_ranges)
    choice_set = realistic_ranges.reset_index(drop=True)
    choice_set.index.name = "choice_id"
    choice_set = choice_set.reset_index()
    choice_set['combo'] = choice_set['start_of_day'].astype(str) + '_' + choice_set['end_of_day'].astype(str)
    choice_dict = dict(zip(choice_set.choice_id, zip(choice_set.start_of_day, choice_set.end_of_day)))
    df = person_choices.merge(choice_set, on=['start_of_day', 'end_of_day'], how='left')
    df = df[~df['combo'].isna()]
    df = df[~df['choice_id'].isna()]
    df['duration'] = df['end_of_day'] - df['start_of_day']
    df['EM_start'] = df['start_of_day'] <7
    df['EV_start'] = df['start_of_day']>14
    df = add_time_dummies(df)
   # df = add_preriod_dummies(df)
    
    choice_id_set = list(set(df['choice_id'].astype(int)))
    choice_dict = Dict(choice_dict).freeze()
  
    
    pp = Synth_Connect().read_in_persons()
    #pp = pd.read_csv(os.path.join(os.path.dirname(__file__),'person_ss.csv'))
    zone_id = pd.read_csv(os.path.join(os.path.dirname(__file__),'seq_zone_ids.csv')).fillna(0)
    zone_id_i = zone_id[['SA2_MAIN16', 'ORDER']]
    zone_id_i = zone_id_i.rename(columns={"ORDER": "HOMEZONE"})
    pp = pp.merge(zone_id_i, left_on = 'OTAZ', right_on = 'HOMEZONE')
    skims = os.path.join(os.path.dirname(__file__),'downcasted_matrices_base.omx')
    skims = omx.open_file(skims, 'a')
    
    LARGE_VAL = 1000
    df_dataset = lx.Dataset.construct.from_idco(df.set_index('person_id'), alts=choice_id_set)
    valid_ids = set(pp['person_id'])
    df_dataset_t = df_dataset.to_dataframe().reset_index()
    df_dataset_t = df_dataset_t[df_dataset_t['person_id'].isin(valid_ids)]
    
    
    used_ids = df_dataset_t['person_id'].unique()
    pp = pp[pp['person_id'].isin(used_ids)].drop_duplicates(subset=['person_id'])
    #pp = pp.set_index('person_id')
    
    
    od_skims = lx.Dataset.construct.from_omx(skims)
    od_skims = od_skims.fillna(LARGE_VAL)
    dt = lx.DataTree(
        time=df_dataset,
        person=pp.set_index('person_id'),
        od=od_skims,
        #do=od_skims,
        relationships=(
            "time.person_id @ person.person_id",
           # 'od.HOMEZONE @ person.HOMEZONE',
           # ''
            #"tours.person_id @ person.PERSONID",
    
        ),
    )
    
    '''Consider these components
    - hhsize
    - roundtrip_auto_time_to_work
    - num_workers
    - income_in_thousands
    - work_and_school_and_worker
    - work_and_school_and_student
    - workplace_in_cbd
    - home_is_rural
    - mandatory_tour_frequency
    - is_worker
    - is_student
    - is_university
    - workplace_zone_id
    - school_zone_id
    - home_zone_id
    - roundtrip_auto_time_to_work
    - num_mand
    - num_escort_tours
    - num_non_escort_tours
    - adult
    '''
    
    '''
    SEGMENTS
    tour_purpose	time_period	start	end
    work	EA	3	5
    work	AM	6	8
    work	MD	10	13
    work	PM	15	17
    work	EV	19	23
    
    '''
    # trip production morning
    
    # trip pdouction evening
    
    m = lx.Model(dt, compute_engine="numba")
    #m.load(dt_work)
    #modelname = {'Education': 'school_location',
     #            'Work': 'work_location'}.get(purpose, 'school_location')
    #m.title = f"{modelname} Mode Choice v1"
    m.choice_co_code = 'choice_id'
    
    m.utility_co = {}

    for i in choice_id_set:
        start, end = choice_dict[i]
        duration = end - start
        start_dummy = get_start_dummy_name(start)
        end_dummy   = get_end_dummy_name(end)
        start_p = get_period(start, start='start')
        end_p = get_period(end, start='end')
        
        # Start with 0 (no ASCs)
        util_expr = 0
        util_expr += P("asc") * X("vehrd >= 0")
        # Add the terms only if they apply
        if start_dummy:
            util_expr += P(start_dummy) * X('vehrd>=0')
        if end_dummy:
            util_expr += P(end_dummy) * X('vehrd>=0')
            
            
            # Personal preferences
        if end_p:
            term_name = f"{end_p}_x_worker"
            util_expr += P(term_name)  * X("WORKSTAT == 1")

        if duration > 0:
            term_name = f'{duration}_activities_eat_and_shop'
            util_expr += P(term_name) * (X("n_eat_out")+X("n_shopping_out"))
            term_name = f'{duration}_work_and_edu'
            util_expr += P(term_name) * (X("n_work_out")+X("n_edu_out"))
            term_name = f'{duration}_escort_and_other'
            util_expr += P(term_name) * (X("n_escort_out")+X("n_othmaint_out")+X("n_othdiscr_out"))
        
        if start_p:
          #  term_name = f"{start_p}_Dist"
           # util_expr += P(term_name) * X("DIST")
            term_name = f"{start_p}_x_worker"
            util_expr += P(term_name) * X("WORKSTAT == 1")
           # util_expr += P(term_name) * X[start_p] * X("WORKSTAT == 1")
        
        if start_p:
            term_name = f"{start_p}_young_student"
            util_expr += P(term_name) * X("STUDYING == 1")
            #util_expr += P(term_name) * X[start_p] * X("STUDYING == 1")

        if end_p:
            term_name = f"{end_p}_income_high"
            util_expr += P(term_name)  * X("incp >= 5")

        
        ### activity
        if end_p:
            term_name = f"{end_p}_eat"
            util_expr += P(term_name)  * X("n_eat_out")

        if start_p:
            term_name = f"{start_p}_escort"
            util_expr += P(term_name)  * X("n_escort_out")
            term_name = f"{start_p}_escort_vehicle"
            util_expr += P(term_name)  * X("vehrd>=2") * X("n_escort_out")
        
        if start_p:
            term_name = f"{start_p}_work"
            util_expr += P(term_name)  * X("n_work_out")

        if end_p:
            term_name = f"{end_p}_edu"
            util_expr += P(term_name)  * X("n_edu_out")
        
        if end_p:
            term_name = f"{end_p}_shopping"
            util_expr += P(term_name)  * X("n_shopping_out")

        if end_p:
            term_name = f"{end_p}_othmaint"
            util_expr += P(term_name)  * X("n_othmaint_out")
        
        
        # Store the final expression
        m.utility_co[i] = util_expr
    
    
    ''' OFF FOR NOW 
    
    
    
    for i in choice_id_set:
        start_col, end_col = choice_dict[i]  # e.g. ("START_i", "END_i")
        duration = end_col - start_col
        duration_name = f"d_{duration}"
        start_name = f"early_start_y_{start_col}"
        as_c =  f"asc{start_col}"
        
        
        util_expr = (P(as_c)+
        P(start_name) * X('AGEGROUP <=4')+P(duration_name)*X('INCOMEN') ) 
      
        m.utility_co[i] =  + util_expr
        
    '''
    # terms = [f"P('{name}')*X['{name}']" for name in coefficients.keys()]

    # Join them with +
    #utility_expression_str = " + ".join(terms)
    
    #df['EV_start']
    #m.utility_ca = (P('ev')*X['EV_start'] +P('em')*X['EM_start'] )
    
    
    m.set_cap(20)
    r = m.maximize_loglike(method ="bhhh", maxiter =100)
    m.parameter_summary()
    probs = m.probability()
    from zsim_helper import monte_sample
    predicted = monte_sample(probs)
    
    
    #predicted = np.argmax(probs, axis=1)
    #print(f'set of predicted{set(predicted)}')
    #accuracy = (predicted == df['choice_id'].values).mean()
    #print(accuracy)
    #logging.info('done')
    external_data, person_data = Synth_Connect().synth_data_test_stage_2(choice_set)
    ''''''
    #external_data -> generate_alts...
    #get_data = synt_data(choice_set)
    
    #m.datatree = get_data
    m.datatree = dt = external_data
   
    probs = m.probability()
    predicted_synth = monte_sample(probs)
    joined = pd.concat(
    [
        person_data.reset_index(drop=True),
        choice_set.iloc[predicted_synth].reset_index(drop=True)
    ],
    axis=1)
    
    joined.to_csv('saved_data/ativities_time_synthetic_population.csv', index=False)
    #synthetic population predict
    #now apply to syntethic population. Note the syntetic population does dot have attibutes such as 
    
    #build choices..

if __name__ == '__main__':
    logging.info('lets run')
    read_in()
    print('done')



