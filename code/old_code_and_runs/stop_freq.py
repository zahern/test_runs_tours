import pandas as pd
# how to activate the venve one back ..
import os
import sys
import numpy as np
import xarray as xr
import jax
import logging
from addicty import Dict
import larch as lx
from functools import reduce
from operator import add
from larch import P, X
import pandas as pd
import openmatrix as omx
print(lx.__version__)
import shutil
import math
logging.basicConfig(level=logging.INFO, format="%(message)s")
# number of tours


def synthesize_data_with_frequency(choice_set):
    persons = pd.read_csv("Z:/sa_run_again_n/zones_sa_4_copy/30901122736_synthetic_population.csv")
    logging.critical('need to process persons that already hace choices...')
    return persons

def get_in_dummy_name(in_s):
    return f'n_inbound_{in_s}'
def get_out_dummy_name(out_s):
    return f'n_outbound_{out_s}'



def trips_to_tours_with_direction():
    """
    Build tours based ONLY on trip data.
    No stops.
    Adds: tour_id, main_trip_id, trip_direction (outbound/main/inbound)
    Returns: trips, tours, person
    """
    import pandas as pd
    import numpy as np
    from data_connect import Synth_Connect

    # -----------------------------------------
    # LOAD DATA
    # -----------------------------------------
    person = pd.read_excel('../2_QTS_PERSONS.xlsx')
    household = pd.read_excel('../1_QTS_HOUSEHOLDS.xlsx')
    persons = person.merge(household, on='HHID', how='left')
    person = Synth_Connect().process_hhts_person(persons)

    trips = pd.read_excel('../5_QTS_TRIPS.xlsx')

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
        'Pickup/deliver something (not work-related)':'othermaint',
        'Social Visit':'social',
        'Recreation/leisure activity':'social',
        'Eat/drink':'eat',
        'Personal business':'othmaint',
        'Just accompanying someone':'escort',
        'Shopping':'shopping'
    }

    MODES = {
        'Car driver': 'DA',
        'Walking': 'Walk',
        'Car passenger': 'DA',
        'Public Bus': 'Transit',
        'Motorcycle driver': 'DA',
        'Truck passenger': 'DA',
        'Charter/Courtesy/Other bus':'Transit',
        'Light rail':'Transit',
        'Public bus':'Transit',
        'School bus (with route number)': 'Transit',
        'E-bike or E-scooter': 'Bike',
        'Truck driver':'DA',
        'Uber / Other Ride Share':'Other',
        'Train': 'Transit',
        'Other method':'Other',
        'Motorcycle passenger':'DA',
        'Taxi':'Other',
        'Bicycle':'Bike',
        'Mobility scooter':'Other',
        'Ferry':'Transit',
        'School bus (private/chartered)': 'Transit'
    }

    trips['MAINMODE'] = trips['MAINMODE'].map(MODES)
    trips['ORIGPURP'] = trips['ORIGPURP'].map(purpose)
    trips['DESTPURP'] = trips['DESTPURP'].map(purpose)

    # Add ENDTIME column
    trips["ENDTIME"] = trips["ARRTIME"]

    # -----------------------------------------
    # STEP 1: SORT TRIPS
    # -----------------------------------------
    trips = trips.sort_values(["PERSID", "STARTIME"])

    # -----------------------------------------
    # STEP 2: DEFINE ANCHOR PLACES
    # -----------------------------------------
    anchor_places = ["My home", "work", "school"]

    # -----------------------------------------
    # STEP 3: ASSIGN tour_id FROM ANCHORS
    # -----------------------------------------
    # Previous destination
    trips["prev_dest"] = trips.groupby("PERSID")["DESTPLACE"].shift()

    # New tour starts whenever the previous destination was an anchor
    trips["new_tour_flag"] = trips["prev_dest"].isin(anchor_places).astype(int)

    # First trip of each person always starts a tour
    first_idx = trips.groupby("PERSID").head(1).index
    trips.loc[first_idx, "new_tour_flag"] = 1

    # Create cumulative tour_id
    trips["tour_id"] = trips.groupby("PERSID")["new_tour_flag"].cumsum()

    # Clean up
    trips = trips.drop(columns=["prev_dest", "new_tour_flag"])

    # -----------------------------------------
    # STEP 4: MAIN TRIP DETECTION & DIRECTION
    # -----------------------------------------
    purpose_priority = {
        'work': 1,
        'edu': 2,
        'escort': 3,
        'shopping': 4,
        'othermaint': 5,
        'othdiscr': 6,
        'eat': 7,
        'social': 8,
        'home': 9
    }

    trips["purpose_rank"] = trips["DESTPURP"].map(purpose_priority).fillna(5)

    # Main trip = lowest purpose_rank per tour
    main_trip_indices = trips.loc[
        trips.groupby(["PERSID", "tour_id"])["purpose_rank"].idxmin()
    ].index

    main_idx_map = (
        trips.loc[main_trip_indices, ['PERSID', 'tour_id', 'TRIPID']]
        .rename(columns={'TRIPID': 'main_trip_id'})
    )

    trips = trips.merge(main_idx_map, on=["PERSID", "tour_id"], how='left')

    # Assign direction
    trips["trip_direction"] = np.where(
        trips["TRIPID"] < trips["main_trip_id"], "outbound",
        np.where(trips["TRIPID"] == trips["main_trip_id"], "main", "inbound")
    )

    # -----------------------------------------
    # STEP 5: BUILD MINIMAL TOURS TABLE
    # -----------------------------------------
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

    # Add home SA2
    tours = tours.merge(
        person[['PERSID', 'HOME_SA2_2021']],
        on="PERSID",
        how="left"
    ).rename(columns={"HOME_SA2_2021": "tour_start_SA2"})

    return trips, tours, person


def linked_stops_to_tours():
    '''
    
    returns trips, tours, stops, person
    '''
    from data_connect import Synth_Connect
    
    

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
    
    purpose =   {   'escort': 'Just accompanying someone',
                    'work': 'Work (other work reason)',
                    'escort': 'Childcare or kindergarten',
                    'work':'Work (my workplace)',
                    'othdiscr':'Refuel my vehicle',
                    'home':'Go home',
                    'home': 'At Home',
                    'escort':'Pick up or drop off a passenger',
                    'edu':'Education',
                    'othermaint':'Pickup/deliver something (not work-related)',
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
    'Pickup/deliver something (not work-related)': 'othermaint',
    'Social Visit':'social',
    'Recreation/leisure activity': 'social',
    'Eat/drink': 'eat',
    'Personal business': 'othmaint',
    'Just accompanying someone': 'escort',
    'Shopping': 'shopping'
    }
    
    MODES = {
        'Car driver': 'DA',
        'Walking': 'Walk',
        'Car passenger': 'DA',
        'Public Bus': 'Transit',
        'Motorcycle driver': 'DA',
        'Truck passenger': 'DA',
        'Charter/Courtesy/Other bus':'Transit',
        'Light rail':  'Transit',
        'Public bus': 'Transit',
        'School bus (with route number)': 'Transit',
        'E-bike or E-scooter': 'Bike',
        'Truck driver': 'DA',
        'Uber / Other Ride Share': 'Other',
        'Train':   'Transit',
        'Other method': 'Other',
        'Motorcycle passenger': 'DA',
        'Taxi': 'Other',
        'Bicycle': 'Bike',
        'Mobility scooter': 'Other',
        'Ferry':  'Transit',
        'School bus (private/chartered)': 'Transit'
        }
    logging.info('reconsider modes')
    trips['MAINMODE'] = trips['MAINMODE'].map(MODES)
    missing = trips.loc[~trips['ORIGPURP'].isin(purpose.keys()), 'ORIGPURP'].unique()
    print(missing)
    trips['ORIGPURP'] = trips['ORIGPURP'].map(purpose)
    trips['DESTPURP'] = trips['DESTPURP'].map(purpose)
    

    # Copy ARRTIME into ENDTIME for convenience
    trips["ENDTIME"] = trips["ARRTIME"]

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
    'othermaint': 5,
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
    tours = tours.merge(main_mode, on=['PERSID', 'tour_id'], how='left')

    trips["tour_leg"] = "outbound"
    trips.loc[main_trip_indices, "tour_leg"] = "main"
    
   
    
    
    # Identify the main trip's destination SA2
    main_sa2 = (
        trips.loc[main_trip_indices, ['PERSID', 'tour_id', 'DESTSA2_2021']]
        .rename(columns={'DESTSA2_2021': 'tour_dest_SA2'})
    )

    tours = tours.merge(main_sa2, on=['PERSID', 'tour_id'], how='left')

    
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
    stops = stops.merge(
        trips[["TRIPID", "PERSID", "tour_id", "trip_direction"]],
        on=["TRIPID", "PERSID", "tour_id"],
        how="left"
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
    stops['DURATION'] = stops["DURATION"].fillna(0)
    
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
    return trips, tours, stops, person


def assign_choice(row_df, choice_set):
    in_ = row_df['num_inbound_stops']
    out = row_df['num_outbound_stops']    
    for i, row in choice_set.iterrows():
        if row['out'] == out and row['in'] == in_:
            return i
    return -1  # not fou

    

def read_in():
    '''
    Docstring for read_in
    '''
    choice_set = pd.read_csv('alternative_configs/stop_frequency_alternatives.csv')['alt']
    choice_full = pd.read_csv('alternative_configs/stop_frequency_alternatives.csv')
    trips, tours, stops, persons = linked_stops_to_tours()
    
    #comment out
    '''
    long = (
    persons
    .assign(key=1)
    .merge(pd.DataFrame({'alt': choice_set, 'key': 1}), on='key')
    .drop(columns='key')    
    )
    # nah can;t use the long tours
    long_tours = (
    tours
    .assign(key=1)
    .merge(pd.DataFrame({'alt': choice_set, 'key': 1}), on='key')
    .drop(columns='key')    
    )
    long_tours = long_tours.merge(choice_full, left_on='alt', right_on='alt')
    #assign choice
    long_tours['num_outbound_stops'] = long_tours['num_outbound_stops'].clip(upper=3)
    long_tours['num_inbound_stops'] = long_tours['num_inbound_stops'].clip(upper=3)
    long_tours['choice_t'] = long_tours.apply(assign_choice, axis=1)
    '''
    long_tours = tours
    long_tours['choice_t'] = long_tours.apply(lambda r: assign_choice(r, choice_full), axis=1)
    for k in range(4):     # 0,1,2,3
        col = f"n_inbound_{k}"
        if col not in long_tours:
            long_tours[col] = (long_tours['num_inbound_stops'] == k).astype(int)
    for k in range(4):
        col = f"n_outbound_{k}"
        if col not in long_tours:
            long_tours[col] = (long_tours['num_outbound_stops'] == k).astype(int)
    
    long_tours['case_id'] = long_tours['PERSID'].astype(str) + "_" + long_tours['tour_id'].astype(str)
    df_dataset = lx.Dataset.construct.from_idco(long_tours.set_index(['case_id']), alts=choice_set)
    #tour_dataset = lx.Dataset.construct.from_idco(tours.set_index('PERSID'), alts=choice_set)
    # no make the choice, 
    # if choice set ful out_bound = num_outbound, and inbound = num_inbound make the chocie accordingito the the choiceset
    
    
    #used_ids = df_dataset_t['PERSID'].unique()
    #pp = pp[pp['PERSID'].isin(used_ids)].drop_duplicates(subset=['PERSID'])
    #pp = pp.set_index('PERSID')
    
    
   # od_skims = lx.Dataset.construct.from_omx(skims)
    #od_skims = od_skims.fillna(LARGE_VAL)
    dt = lx.DataTree(
        tours = df_dataset,
        person = persons.set_index('PERSID'),
        relationships=( 
            "tours.PERSID @ person.PERSID",)
    
        
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
   
    
    
    m = lx.Model(dt, compute_engine="numba")
    #m.load(dt_work)
    #modelname = {'Education': 'school_location',
     #            'Work': 'work_location'}.get(purpose, 'school_location')
    #m.title = f"{modelname} Mode Choice v1"
    m.choice_co_code = 'choice_t'
    m.choice_ca_var
    m.utility_co = {}

    for i, row in choice_full.iterrows():
        in_n, end_n = row['out'], row['in']
        in_dummy = get_in_dummy_name(in_n)
        out_dummy = get_out_dummy_name(end_n)
        #start_dummy = get_start_dummy_name(start)
        #end_dummy   = get_end_dummy_name(end)
        #start_p = get_period(start, start='start')
        #end_p = get_period(end, start='end')
        
        # Start with 0 (no ASCs)
        util_expr = 0

        # Add the terms only if they apply
        if in_dummy:
            #duration term
            if in_n >0:
                term_name = f"{in_dummy}_duration"
                util_expr += P(term_name) * X[in_dummy] * (X("tour_start")-X("tour_end"))
                # activities term
                term_name = f"{in_dummy}_trips"
                util_expr += P(term_name) * X[in_dummy] * X("num_trips")
                #num_shopping
                term_name = f"{in_dummy}_shopping"
                util_expr += P(term_name) * X[in_dummy] * X("num_shopping_total")
                #num_social     
                term_name = f"{in_dummy}_social"
                util_expr += P(term_name) * X[in_dummy] * X("num_social_total")
                #num_other
                term_name = f"{in_dummy}_other"
                util_expr += P(term_name) * X[in_dummy] * X("num_other_total")
            #ASC 
            util_expr += P(out_dummy) * X[in_dummy]
        if out_dummy:
            #ASC
            util_expr += P(out_dummy) * X[in_dummy]
            #duration term
            term_name = f"{out_dummy}_duration"
            util_expr += P(term_name) * X[out_dummy] * (X("tour_start")-X("tour_end"))
            # activities term
            term_name = f"{out_dummy}_trips"
            util_expr += P(term_name) * X[out_dummy] * X("num_trips")
            #num_shopping
            term_name = f"{out_dummy}_shopping"
            util_expr += P(term_name) * X[out_dummy] * X("num_shopping_total")
            #num_social     
            term_name = f"{out_dummy}_social"
            util_expr += P(term_name) * X[out_dummy] * X("num_social_total")
            #num_other
            term_name = f"{out_dummy}_other"
            util_expr += P(term_name) * X[out_dummy] * X("num_other_total")
            
            
            
            # Personal preferences
        

        

        
        
        
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
    r = m.maximize_loglike(method ="bhhh", maxiter =1000)
    m.parameter_summary()
    probs = m.probability()
    predicted = np.argmax(probs, axis=1)
    print(f'set of predicted{set(predicted)}')
    #accuracy = (predicted == df['choice_id'].values).mean()
    #print(accuracy)
    logging.info('done')
    get_data = synthesize_data_with_frequency(choice_set)
    
    m.datatree = get_data
    #probs = m.probability()


    
    


stop_freqeuncy_alts = pd.read_csv('alternative_configs/stop_frequency_alternatives.csv')




import logging
import pandas as pd
import numpy as np


def build_utilities(m, X, Activity, P, config):
    """
    Build symbolic utilities compatible with larch.
    """
    for alt_key, variables in config.items():
        asc_name = f"ASC_{alt_key}"
        expr = P(asc_name)

        for var in variables:
            base_var = var
            param_name = f"{base_var}_{alt_key}" if hasattr(P, f"{base_var}_{alt_key}") else base_var
            expr += P(param_name) * X(base_var)

        alt_attr = getattr(Activity, alt_key)
        m.utility_co[alt_attr] = expr

    return m


import json
def remap_and_json(pp, var):
    pp[var] = pp[var].fillna("").replace("", "MISSING")
    #pp['WORKSTATUS'] = pp['WORKSTATUS'].fillna("").replace("", "MISSING")
    unique_values = pp[var].unique()
    mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
    mapping["MISSING"] = 0  # assign 0 for missing or incomplete entries
    pp[var] = pp[var].map(mapping)
    with open(f"category_mapping_{var}.json", "w") as f:
        json.dump(mapping, f)









    




def anneal_config(setup_freq, config=None, all_variables=None, iterations=100, T0=1.0, alpha=0.95):
    """
    Generic simulated annealing process for optimizing model configs
    regardless of how many alternatives (Wrk1, Wrk2, Sch1, etc.) exist.
    """

    # 1️⃣ Default variables if not passed
    if all_variables is None:
        all_variables = [
            "EBIKES", "MCLICENCE", "ESCOOTER",
            "RELATIONSHIP == 3", "RELATIONSHIP == 1", "RELATIONSHIP == 2",
            "WORKSTATUS == 1",
            "INCOMEN<12000", "INCOMEN>=12000",
            "HHVEH", "CARLICENCE", 'AGEGROUP <5', 
             'TRIPS'
        ]

    # 2️⃣ Default config if none provided
    if config is None:
        # Default to 5 utilities instead of hardcoding 3
        keys = ["wrl1", "wrk2", "sch1", "sch2", "wrk_and_sch"]
        config = {k: [] for k in keys}
        # A minimal seed configuration
        config["wrk1"] = ["INCOMEN<12000", "INCOMEN>=12000", "HHVEH"]

    # 3️⃣ Evaluate initial configuration
    best_config = copy.deepcopy(config)
    best_res = setup_freq(config)
    best_score = best_res["loglike"]
    current_score = best_score
    T = T0

    logging.info(f"Initial loglike = {best_score:.3f}")

    # 4️⃣ Main Annealing Loop
    for i in range(iterations):
        # Propose new candidate
        new_config = random_neighbor(config, all_variables)
        try:
            res = setup_freq(new_config)
            score = res["loglike"]
        except Exception as e:
            logging.warning(f"Setup failed for candidate: {e}")
            continue

        Δ = score - current_score
        # Accept via standard SA rule
        if Δ > 0 or math.exp(Δ / T) > random.random():
            config = new_config
            current_score = score
            if score > best_score:
                best_score = score
                best_config = copy.deepcopy(config)

        T *= alpha
        print(f"Iter {i+1}: loglike={score:.3f} best={best_score:.3f} T={T:.3f}")
    
    print("\n✅ Best Config Found:")
    for k, vars in best_config.items():
        print(f"  {k}: {vars}")
    
    return best_config


import random
import copy

def random_neighbor(config, variables):
    new_config = copy.deepcopy(config)

    # List of all config keys dynamically
    activity_keys = list(new_config.keys())

    # 1️⃣ Randomly select a variable to move
    v = random.choice(variables)

    # Remove it from all existing lists
    for k in activity_keys:
        if v in new_config[k]:
            new_config[k].remove(v)

    # 2️⃣ Randomly choose in which configs to place it (1–num_activities)
    chosen = random.sample(activity_keys, k=random.randint(1, len(activity_keys)))
    for c in chosen:
        new_config[c].append(v)

    # --------------------------------------------------
    # APPLY CONSTRAINTS (editable policy section)
    # --------------------------------------------------

   

    # Ensure every config has at least one variable
    for a in activity_keys:
        if len(new_config[a]) == 0:
            fallback = random.choice(variables)
            new_config[a].append(fallback)

    # Done 👍
    return new_config


if __name__ == '__main__':
    logging.info('cool beans bro beans')
    read_in()
    linked_stops_to_tours()
    print('stops linked to tours')
   # anneal_config(setup_freq, config = config)
    #setup_freq()