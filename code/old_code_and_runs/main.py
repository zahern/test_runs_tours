# how to activate the venve one back ..
import os
import sys
import numpy as np
import xarray as xr
import jax
import logging
from addicty import Dict
import larch as lx
## new test
#lx.set_backend("jax")
#print(lx.get_backend())
## delete me if not stable
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
###
from functools import reduce
from operator import add
from larch import P, X, PX
import pandas as pd
import openmatrix as omx
print(lx.__version__)
import shutil
from data_connect import Synth_Connect
from zsim_helper import monte_sample, monte_sample_key
from sqlalchemy import create_engine
from rich.console import Console
engine = create_engine(
    "sqlite:///tour_location.db"
)
compute_engine = 'jax'

def cdap_plan():
    lx.example
    logging.info('import from synth connect')
    #pp = Synth_Connect().read_in_persons()
    #tour = Synth_Connect().read_in_tours()
    pp = Synth_Connect().persons_to_active()
    # drop DTAZ 330
    #tour = tour[tour['DTAZ'] != 0]
    #tour = tour[tour['OTAZ'] != 0]
    #tour['main_purpose'] = tour['main_purpose'].fillna('othmaint')
    #tour['ptour_id'] =tour['person_id'].astype(str) + '_' + tour['tour_id'].astype(str)
    zone_id = pd.read_csv(os.path.join(os.path.dirname(__file__),'seq_zone_ids_.csv')).fillna(0)
    zone_id = Synth_Connect().get_land()
    zone_id_i = zone_id[['SA2_MAIN16', 'ORDER']]
    zone_id_i = zone_id_i.rename(columns={"ORDER": "HOMEZONE"})
    pp = pp.merge(zone_id_i, left_on = 'HOME_SA2_2021', right_on = 'SA2_MAIN16')
    skims = os.path.join(os.path.dirname(__file__),'downcasted_matrices_base.omx')
    skims = omx.open_file(skims, 'r')
    
    
    Activity = Dict(
        M = 1,
        N = 2,
        H = 3
    ).freeze()
    
    # if i have value 1, how do i get back M, and do this for list of values
    pp['ACTC'] = pp['ACTC'].map(Activity).fillna(2)
    LARGE_VAL = 1000
    pp_dataset = lx.Dataset.construct.from_idco(pp.set_index('person_id'), alts=Activity)
    od_skims = lx.Dataset.construct.from_omx(skims)
    od_skims = od_skims.fillna(LARGE_VAL)
    
    #tour_dataset = lx.Dataset.construct.from_idco(tour.set_index('ptour_id'), alts=Activity)
    dt = lx.DataTree(
        
       # tours=tour_dataset,
        person=pp_dataset
        )
   

    
    
    m = lx.Model(dt, compute_engine=compute_engine)
    #m.load(dt_work)
    modelname = 'CDAP_model'
    m.title = f"{modelname} Choice v1"
    print('the mode choice needs to be here.')
    m.choice_co_code = 'ACTC'
    
    # will run a script to set everything up if the example data is not already available.



    # # Load data and prep model for estimation


    

    #modelname="school_location"



    m.utility_co[Activity.M] = (
        
            +P.MAND_Worker_FT *X('WORKSTAT == 1')
            +P.MAND_Worker_PT *X('WORKSTAT == 2')
            +P.coef_UNAVAILABLE*X('WORKSTAT in [5,6,7,0]')
            +P.MAND_School_Child*X('STUDYING  in [1,2]')
           # +P.School_High*X('STUDYING == 2')
            #+ P.MAND_STUDY * X('STUDYING == 3')
            +P.MAND_School_and_Work * X('WORKSTAT in [1,2]') *X('STUDYING in [2,3,4]')
            #+P.Mand_High_income * X('incp >= 10 <= 17')
            #+P('-999')*X('WORKSTAT in [5,6,7,0]')
         
            
            
    )

    m.utility_co[Activity.N] = (
            + P.ASC_NON_MAND
            + P.NON_MAND_PT_Work *X('WORKSTAT == 2')
            + P.Non_MAND_STUDY * X('STUDYING in [3,4]')
            +P.coef_UNAVAILABLE*X('WORKSTAT in [1]')
            + P.Low_Income_Non * X('incp <= 4')
           # +P('-999')*X('WORKSTAT in [1]')
            
        #+ P("LogIncome:Other") * X("INCOMEN")
    )

    m.utility_co[Activity.H] = (
            + P.Asc_Home
            + P.Non_Work_Home* X('WORKSTAT in [5, 6, 7, 0]')
            + P.Low_Income_Home * X('incp <= 4')
           # +P.coef_UNAVAILABLE*X('WORKSTAT in [1]')
            # +P('-999')*X('WORKSTAT in [1]')
            
    )
    

   

    # nest strucuture
    

    '''
    m.availability_co_vars = {
        Activity.M: 'WORKSTAT in [1,2] or STUDYING in [1,2,3,4]',
        Activity.N: 1,
        Activity.H: 1,
        
    }
    
    '''

    m.choice_avail_summary()
    print(m.choice_avail_summary())
    m.set_cap(80) # improves optimization stability
    m.doctor(repair_ch_av = '+')
   # m.doctor(repair_ch_av = '?')
   # m.doctor()
    print(m.choice_avail_summary())
    result = m.maximize_loglike()
    m.calculate_parameter_covariance()
    report = lx.Reporter(title=m.title)
    report << "# loglike" << m.estimation_statistics()
    report << "# Parameter Summary" << m.parameter_summary()

    report.save(
        'brisbane_m_nm_h_choice.html',
        overwrite=True,
        metadata=m,
    )
    logging.info('done')
    
    ## NOW PREDICT
    persons_pred, persons_o = Synth_Connect().synth_data_test(Activity)
    
    logging.info('lets predict')
    '''
    tours_pred_ds= lx.Dataset.construct.from_idco(
    tours_pred,
    alts=Mode   # SAME dict used in estimation
    )
    pop_data = Synth_Connect().generate_alternatives(persons_ex, 'DEST_SAMPLE')
    
    data_synth = Synth_Connect().synth_data_pass(choice_id_set_zones, pop_data, filter_function = lambda df:df['STUDYING'].isin([0,5, 6,1,2,3,4]))
    # not yet this is only the person
    '''
    tree_pred = lx.DataTree(
    #base = ca_pred,
    pp = persons_pred,
    
    )
    
    
    # LETS see IF THIS WORKS
    m.datatree = tree_pred
    m.mangle()
    m.unmangle()
    checkthepreds = m.probability()
    
    predicted_synthmode = monte_sample(checkthepreds)+1
    
    
    #a = Activity.iloc[predicted_synthmode].reset_index(drop=True)
    #a = a.rename(columns={"HOMEZONE": "DTAZ"})
    Activity_rev = {v: k for k, v in Activity.items()}
    persons_o['ACT'] = np.vectorize(Activity_rev.get)(predicted_synthmode)
    persons_o['ACT_v'] = predicted_synthmode
    
    persons_o.to_csv('saved_data/cdap_synthetic_population.csv', index=False)
    
def trip_production():
    pp =  Synth_Connect().read_in_persons()
    tour = Synth_Connect().read_in_tours()
    '''
    
      obs_edu_trips = (
    tour
    .query("main_purpose == 'edu'")   # optional, but recommended
    .groupby("DTAZ")
    .size()
    .rename("OBS_EDU_TRIPS")
    )
    obs_work_trips = (
    tour
    .query("main_purpose == 'work'")   # optional, but recommended
    .groupby("DTAZ")
    .size()
    .rename("OBS_WORK_TRIPS")
    )
    zone_id = (
    zone_id
    .merge(obs_edu_trips, left_on="ORDER", right_index=True, how="left")
    .fillna({"OBS_EDU_TRIPS": 0})
    )
    zone_id = (
    zone_id
    .merge(obs_work_trips, left_on="ORDER", right_index=True, how="left")
    .fillna({"OBS_WORK_TRIPS": 0})
    )
    zone_id['HSENROLL'] = zone_id['OBS_EDU_TRIPS'] +zone_id['HSENROLL']+1
    zone_id['AGE0519'] = zone_id['OBS_EDU_TRIPS'] +zone_id['AGE0519']+1
    total_ = zone_id['HSENROLL'].sum() +  zone_id['AGE0519'].sum()
   # zone_id['HSENROLL'] = zone_id['HSENROLL'] 
    #zone_id['AGE0519'] = zone_id['AGE0519']/ to
    
    zone_id_t = zone_id['FPSEMPN'].sum()+zone_id['HEREMPN'].sum()+zone_id['AGREMPN'].sum()+zone_id['MWTEMPN'].sum() + zone_id['OTHEMP'].sum() + zone_id['RETEMPN'].sum()
    
    zone_id['FPSEMPN'] = (zone_id['FPSEMPN']+1)
    zone_id['HEREMPN'] = (zone_id['FPSEMPN']+1)
    zone_id['AGREMPN']= (zone_id['AGREMPN']+1)
    zone_id['MWTEMPN']= (zone_id['MWTEMPN']+1)
    zone_id['OTHEMP']= (zone_id['OTHEMP']+1)
    zone_id['RETEMPN'] = (zone_id['RETEMPN']+1)
    
    
    pp['cell'] = (
    pp['OTAZ'].astype(str) 
    )
    sample_counts = (
    pp.groupby('cell')
      .size()
      .rename('sample_n')
      .reset_index()
    )
    '''
    
    
    
    
    zone_id = pd.read_csv(os.path.join(os.path.dirname(__file__),'seq_zone_ids_.csv')).fillna(0)
    zone_id = Synth_Connect().get_land()
    zone_id_i = zone_id[['SA2_MAIN16', 'ORDER']]
    zone_id_i = zone_id_i.rename(columns={"ORDER": "HOMEZONE"})
    tot_samples = len(pp)
    total_pop = zone_id['TOTPOP'].sum()
    weight = total_pop/tot_samples
    '''
    obs_edu_trips = (
    tour
    .query("main_purpose == 'edu'")   # optional, but recommended
    .groupby("DTAZ")
    .size()
    .rename("OBS_EDU_TRIPS")
    )
    obs_work_trips = (
    tour
    .query("main_purpose == 'work'")   # optional, but recommended
    .groupby("DTAZ")
    .size()
    .rename("OBS_WORK_TRIPS")
    )
    # shopping
    obs_shopping_trips = (
    tour
    .query("main_purpose == 'shopping'")   # optional, but recommended
    .groupby("DTAZ")
    .size()
    .rename("OBS_SHOP_TRIPS")
    )
    #other
    
    # do this for all other putporsoses
    
    #social
    '''
    obs_trips = (
    tour
    .groupby(["DTAZ", "main_purpose"])
    .size()
    .unstack(fill_value=0)   # creates one column per purpose
    )
    #work_pp = tour[tour['main_purpose'] == 'work']
    #work_pp = work_pp.merge(pp, how = 'left', on = ['person_id', 'OTAZ', 'HHID'])
    # Rename columns to match your naming style
   # work_pp = work_pp.dropna(subset=['incp'])
    obs_trips = obs_trips.add_prefix("OBS_").add_suffix("_TOURS")*int(weight)
    
    
    zone_id = (
    zone_id
    .merge(obs_trips, left_on="ORDER", right_index=True, how="left")
    .fillna(0)
    )
    
    zone_id['OBS_nm_TOURS'] = zone_id['OBS_social_TOURS']+zone_id['OBS_escort_TOURS']+zone_id['OBS_othmaint_TOURS']+zone_id['OBS_othdiscr_TOURS']
    
    
    
    '''
    skims = os.path.join(os.path.dirname(__file__),'downcasted_matrices_base.omx')
    skims = omx.open_file(skims, 'a')
    
    T = skims         # shape (n_zones, n_zones)
    land_use = zone_id     # must contain TOTEMP

    beta_auto = -0.045

    TOTEMP = land_use["TOTEMP"].values
    TOTEMP = np.where(TOTEMP > 0, TOTEMP, 1)

    # Utility
    V = beta_auto * T + np.log(TOTEMP)

    # Stable logsum
    max_V = np.max(V, axis=1, keepdims=True)
    accessibility = max_V.flatten() + np.log(np.sum(np.exp(V - max_V), axis=1))

    # Save result
    land_use["autoPeakAccessibility"] = accessibility
    land_use.to_csv("land_use_with_accessibility.csv", index=False)

    print("Accessibility calculation complete.")
    
    
    '''
    return zone_id



    
    '''
    
    zone_id = (
    zone_id
    .merge(obs_edu_trips, left_on="ORDER", right_index=True, how="left")
    .fillna({"OBS_EDU_TRIPS": 0})
    )
    zone_id = (
    zone_id
    .merge(obs_work_trips, left_on="ORDER", right_index=True, how="left")
    .fillna({"OBS_WORK_TRIPS": 0})
    
    
    
    
    )
    zone_id['HSENROLL'] = zone_id['OBS_EDU_TRIPS'] +zone_id['HSENROLL']+1
    zone_id['AGE0519'] = zone_id['OBS_EDU_TRIPS'] +zone_id['AGE0519']+1
    # propo
    '''
    
def tour_location_joint_cnl(purpose='M', submodel=None, SAFETY=1):

    import os
    import logging
    import pandas as pd
    import numpy as np
    import larch as lx
    import openmatrix as omx

    logging.info("Starting JOINT Mode-Destination CNL estimation")

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    pp = Synth_Connect().read_in_persons()
    tour = Synth_Connect().read_in_tours()

    tour = tour[(tour['DTAZ'] != 0) &
                (tour['OTAZ'] != 0) &
                (tour['OTAZ'] != 330) &
                (tour['DTAZ'] != 330)]

    tour['main_purpose'] = tour['main_purpose'].fillna('othmaint')
    tour['ptour_id'] = tour['person_id'].astype(str) + '_' + tour['tour_id'].astype(str)

    # Purpose filter
    if submodel is None:
        if purpose == 'M':
            tour = tour[tour["main_purpose"].isin(["work", "edu"])]
        else:
            tour = tour[~tour["main_purpose"].isin(["work", "edu"])]
    else:
        tour = tour[tour["main_purpose"] == submodel]

    # --------------------------------------------------
    # 2. DEFINE MODES
    # --------------------------------------------------

    Mode = {
        1: "DA",
        2: "Other",
        3: "Walk",
        4: "Bike",
        5: "Transit"
    }

    tour['tour_mode'] = tour['tour_mode'].fillna(2)

    # --------------------------------------------------
    # 3. CREATE JOINT ALTERNATIVE SPACE (DTAZ × Mode)
    # --------------------------------------------------

    skims_path = os.path.join(os.path.dirname(__file__),
                              'downcasted_matrices_base.omx')
    skims = omx.open_file(skims_path)

    dest_ids = skims.root.lookup.ids[:]
    modes = list(Mode.keys())

    joint = pd.MultiIndex.from_product(
        [dest_ids, modes],
        names=["DTAZ", "mode"]
    ).to_frame(index=False)

    joint["alt_id"] = (
        joint["DTAZ"].astype(str) + "_" +
        joint["mode"].astype(str)
    )

    joint = joint.set_index("alt_id")

    # Observed joint choice
    tour["alt_id"] = (
        tour["DTAZ"].astype(str) + "_" +
        tour["tour_mode"].astype(int).astype(str)
    )

    # --------------------------------------------------
    # 4. LANDUSE + SKIMS
    # --------------------------------------------------

    zone_id = Synth_Connect().get_land()
    zone_id = zone_id.set_index("ORDER")

    od_skims = lx.Dataset.construct.from_omx(skims)

    # --------------------------------------------------
    # 5. BUILD LARCH DATA TREE
    # --------------------------------------------------

    joint_ds = lx.Dataset.construct.from_idca(
        joint,
        alts=joint.index
    )

    tour_ds = lx.Dataset.construct.from_idco(
        tour.set_index("ptour_id"),
        alts=joint.index
    )

    tree = lx.DataTree(
        tour=tour_ds,
        person=pp.set_index("person_id"),
        emp=zone_id,
        od=od_skims,
        joint=joint_ds,
        relationships=(
            "tour.person_id @ person.person_id",
            "joint.DTAZ @ emp.ORDER",
            "joint.DTAZ @ od.dtaz",
            "tour.OTAZ @ od.otaz",
        )
    )

    # --------------------------------------------------
    # 6. DEFINE JOINT MODEL
    # --------------------------------------------------

    m = lx.Model(tree, compute_engine=compute_engine)
    m.choice_co_code = "tour.alt_id"
    m.title = f"Joint Mode-Destination CNL ({purpose}-{submodel})"

    # --------------------------------------------------
    # 7. UTILITY SPECIFICATION
    # --------------------------------------------------

    m.utility_ca = (

        # ----------------------
        # DESTINATION COMPONENT
        # ----------------------

        + P.distance * X("od.DIST")
        + P.zone_size * X("emp.OBS_work_TOURS")

        # ----------------------
        # MODE COMPONENT
        # ----------------------

        + P.ASC_DA * X("joint.mode == 1")
        + P.ASC_Other * X("joint.mode == 2")
        + P.ASC_Walk * X("joint.mode == 3")
        + P.ASC_Bike * X("joint.mode == 4")
        + P.ASC_Transit * X("joint.mode == 5")

        + P.Time_Car * X("od.SOVTOLL_TIME__AM") * X("joint.mode in [1,2]")
        + P.Time_PT * X("od.DRV_LOC_WLK_TOTIVT__AM") * X("joint.mode == 5")

        + P.Income * X("person.incp") * X("joint.mode in [1,2,5]")
    )

    if SAFETY:
        m.utility_ca += (
            + P.Crash_Car * X("od.CRASH") * X("joint.mode in [1,2]")
            + P.Crash_PT * X("od.CRASHPT") * X("joint.mode == 5")
        )

    # --------------------------------------------------
    # 8. CROSS-NESTED STRUCTURE
    # --------------------------------------------------

    # Mode nests
    mode_nodes = {}
    for mode_id in modes:
        alts = joint[joint["mode"] == mode_id].index.tolist()
        mode_nodes[mode_id] = m.graph.new_node(
            parameter=f"Mu_mode_{mode_id}",
            children=alts
        )

    # Destination nests
    dest_nodes = {}
    for d in dest_ids:
        alts = joint[joint["DTAZ"] == d].index.tolist()
        dest_nodes[d] = m.graph.new_node(
            parameter=f"Mu_dest_{d}",
            children=alts
        )

    # Root
    m.graph.new_node(
        parameter="Mu_root",
        children=list(mode_nodes.values()) +
                 list(dest_nodes.values())
    )

    # --------------------------------------------------
    # 9. ESTIMATION
    # --------------------------------------------------

    m.set_cap(10)
    m.doctor(repair_ch_av="+")
    m.doctor(repair_nan_utility=True)

    result = m.maximize_loglike()
    m.calculate_parameter_covariance()

    print(m.parameter_summary())

    report = lx.Reporter(title=m.title)
    report << "# Parameter Summary" << m.parameter_summary()
    report.save(
        f"Joint_CNL_{purpose}_{submodel}_SAFETY{SAFETY}.html",
        overwrite=True,
        metadata=m,
    )

    logging.info("Joint CNL estimation complete")

    return m



def tour_location_joint_nested(purpose='M', submodel=None, SAFETY=1):

    import os
    import logging
    import pandas as pd
    import numpy as np
    import larch as lx
    import openmatrix as omx

    logging.info("Starting JOINT 2-Level Nested Mode-Destination Model")

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    pp = Synth_Connect().read_in_persons()
    tour = Synth_Connect().read_in_tours()

    tour = tour[(tour['DTAZ'] != 0) &
                (tour['OTAZ'] != 0) &
                (tour['OTAZ'] != 330) &
                (tour['DTAZ'] != 330)]

    tour['main_purpose'] = tour['main_purpose'].fillna('othmaint')
    tour['ptour_id'] = tour['person_id'].astype(str) + '_' + tour['tour_id'].astype(str)

    # Purpose filtering
    if submodel is None:
        if purpose == 'M':
            tour = tour[tour["main_purpose"].isin(["work", "edu"])]
        else:
            tour = tour[~tour["main_purpose"].isin(["work", "edu"])]
    else:
        tour = tour[tour["main_purpose"] == submodel]

    # --------------------------------------------------
    # 2. DEFINE MODES
    # --------------------------------------------------

    modes = [1,2,3,4,5]

    tour['tour_mode'] = tour['tour_mode'].fillna(2)

    # --------------------------------------------------
    # 3. JOINT ALTERNATIVES (Destination × Mode)
    # --------------------------------------------------

    skims_path = os.path.join(os.path.dirname(__file__),
                              'downcasted_matrices_base.omx')
    skims = omx.open_file(skims_path)

    dest_ids = skims.root.lookup.ids[:]

    joint = pd.MultiIndex.from_product(
        [dest_ids, modes],
        names=["DTAZ", "mode"]
    ).to_frame(index=False)

    joint["alt_id"] = (
        joint["DTAZ"].astype(str) + "_" +
        joint["mode"].astype(str)
    )

    joint = joint.set_index("alt_id")
    mode_map = {
    "DA": 1,
    "Other": 2,
    "Walk": 3,
    "Bike": 4,
    "Transit": 5
    }

    tour["tour_mode"] = tour["tour_mode"].map(mode_map)


    # Observed joint choice
    tour["alt_id"] = (
        tour["DTAZ"].astype(str) + "_" +
        tour["tour_mode"].astype(int).astype(str)
    )

    # --------------------------------------------------
    # 4. BUILD DATASETS
    # --------------------------------------------------

    zone_id = Synth_Connect().get_land()
    zone_id = zone_id.set_index("ORDER")

    od_skims = lx.Dataset.construct.from_omx(skims)

    
    # Cross join tours and alternatives
    tour["_tmpkey"] = 1
    joint["_tmpkey"] = 1

    idca = tour[["ptour_id", "OTAZ"]].merge(
        joint[["alt_id", "DTAZ", "mode"]],
        on="_tmpkey"
    ).drop(columns="_tmpkey")
    
    joint_ds = lx.Dataset.construct.from_idca(
        joint
    )

    tour_ds = lx.Dataset.construct.from_idco(
        tour.set_index("ptour_id"),
        alts=joint.index
    )

    tree = lx.DataTree(
        tour=tour_ds,
        person=pp.set_index("person_id"),
        emp=zone_id,
        od=od_skims,
        joint=joint_ds,
        relationships=(
            "tour.person_id @ person.person_id",
            "joint.DTAZ @ emp.ORDER",
            "joint.DTAZ @ od.dtaz",
            "tour.OTAZ @ od.otaz",
        )
    )

    # --------------------------------------------------
    # 5. DEFINE MODEL
    # --------------------------------------------------

    m = lx.Model(tree, compute_engine=compute_engine)
    m.choice_co_code = "tour.alt_id"
    m.title = f"Joint Nested Mode-Destination ({purpose}-{submodel})"

    # --------------------------------------------------
    # 6. UTILITY SPECIFICATION
    # --------------------------------------------------

    m.utility_ca = (

        # ---------------------------
        # DESTINATION COMPONENT
        # ---------------------------

        + P.distance * X("od.DIST")
        + P.zone_size * X("emp.OBS_work_TOURS")

        # ---------------------------
        # MODE COMPONENT
        # ---------------------------

        + P.ASC_DA * X("joint.mode == 1")
        + P.ASC_Other * X("joint.mode == 2")
        + P.ASC_Walk * X("joint.mode == 3")
        + P.ASC_Bike * X("joint.mode == 4")
        + P.ASC_Transit * X("joint.mode == 5")

        + P.Time_Car * X("od.SOVTOLL_TIME__AM") * X("joint.mode in [1,2]")
        + P.Time_PT * X("od.DRV_LOC_WLK_TOTIVT__AM") * X("joint.mode == 5")

        + P.Income * X("person.incp") * X("joint.mode in [1,2,5]")
    )

    if SAFETY:
        m.utility_ca += (
            + P.Crash_Car * X("od.CRASH") * X("joint.mode in [1,2]")
            + P.Crash_PT * X("od.CRASHPT") * X("joint.mode == 5")
        )

    # --------------------------------------------------
    # 7. NESTED STRUCTURE
    # --------------------------------------------------

    # Mode nest (lower level)
    mode_nodes = []
    for d in dest_ids:

        alts = joint[joint["DTAZ"] == d].index.tolist()

        node = m.graph.new_node(
            parameter="Mu_mode",
            children=alts
        )
        mode_nodes.append(node)

    # Destination nest (upper level)
    m.graph.new_node(
        parameter="Mu_dest",
        children=mode_nodes
    )

    # --------------------------------------------------
    # 8. ESTIMATION
    # --------------------------------------------------

    m.set_cap(10)
    m.doctor(repair_ch_av="+")
    m.doctor(repair_nan_utility=True)

    result = m.maximize_loglike()
    m.calculate_parameter_covariance()

    print(m.parameter_summary())

    report = lx.Reporter(title=m.title)
    report << "# Parameter Summary" << m.parameter_summary()
    report << "# loglike" << m.estimation_statistics()
    report.save(
        f"Joint_Nested_{purpose}_{submodel}_SAFETY{SAFETY}.html",
        overwrite=True,
        metadata=m,
    )

    logging.info("Joint Nested estimation complete")

    return m
    
    

''' PURPOSE CAN ME MAND: M or Non-Mand N'''
def tour_location(purpose = 'M', submodel = 'edu', OTAZ = 1, SAFETY = 1):
    
    # how to make it relative to path in file..
    #df = pd.read_csv(os.path.join(os.path.dirname(__file__), "tour_dataset.csv"))
   
    logging.info('import from synth connect')
    pp = Synth_Connect().read_in_persons()
    tour = Synth_Connect().read_in_tours()
   # tour = tour.drop(columns=['incp'])
    # drop DTAZ 330
    tour = tour[tour['DTAZ'] != 0]
    tour = tour[tour['OTAZ'] != 0]
    tour = tour[tour['OTAZ'] != 330]
    tour = tour[tour['DTAZ'] != 330]
    
    # how many tours??
    
    #tour = tour[~tour["OTAZ"].isin([22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300])]
    #tour = tour[~tour["DTAZ"].isin([22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300])]
    #tour = tour[]
    tour['main_purpose'] = tour['main_purpose'].fillna('othmaint')
    tour['ptour_id'] =tour['person_id'].astype(str) + '_' + tour['tour_id'].astype(str)
    zone_id = pd.read_csv(os.path.join(os.path.dirname(__file__),'seq_zone_ids_.csv')).fillna(0)
    zone_id = Synth_Connect().get_land()
    zone_id_i = zone_id[['SA2_MAIN16', 'ORDER']]
    zone_id_i = zone_id_i.rename(columns={"ORDER": "HOMEZONE"})
    zone_id = trip_production()
    pp = pp.merge(zone_id_i, left_on = 'HOME_SA2_2021', right_on = 'SA2_MAIN16')
    skims = os.path.join(os.path.dirname(__file__),'downcasted_matrices_base.omx')
    skims = omx.open_file(skims, 'r')

    #skims.create_mapping('ids',zone_id['ORDER'].unique() ).
    #skims.close()


    # Open the existing file in read mode

   


    
    #skims.root.lookup['ids'][:]
    Mode = Dict(
        DA = 1,
        Other = 2,
        Walk = 3,
        Bike = 4,
        Transit = 5,
    ).freeze()
    tour['tour_mode'] = tour['tour_mode'].map(Mode).fillna(2)
    #trips['mode'] = trips['mode'].map(Mode)
    

    LARGE_VAL = 1000
    tour_dataset = lx.Dataset.construct.from_idco(tour.set_index('ptour_id'), alts=Mode)
    if purpose not in tour_dataset['main_purpose']:
        logging.warning(f'{purpose} not in data')
    od_skims = lx.Dataset.construct.from_omx(skims)
    od_skims = od_skims.fillna(LARGE_VAL)
    #landuse  = pd.read_csv('Z:/lanuse/landuse_final.csv')
    dt = lx.DataTree(
        
        tour=tour_dataset,
        person=pp.set_index('person_id'),
        emp = zone_id.set_index("ORDER"),
        od=od_skims,
      
        relationships=(
            "emp.ORDER  @  tour.DTAZ",
            "tour.person_id @ person.person_id",
            "person.OTAZ @ tour.OTAZ",
            #"tours.person_id @ person.PERSONID",
            "tour.OTAZ @ od.otaz",
            "tour.DTAZ @ od.dtaz",
        
           #"pp.OTAZ @ do.dtaz",
          # "tours.DTAZ @ do.otaz"
        ),
    )
    
   
    if submodel is None:
        if purpose =='M':
            dt_work = dt.query_cases("main_purpose in ['work', 'edu']")
           # dt_work = dt_work.query_cases('DTAZ not in [22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300]')
        else :
            dt_work = dt.query_cases("main_purpose not in ['work', 'edu']")
         #   dt_work = dt_work.query_cases('DTAZ in [22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300]')
    else:
        dt_work = dt.query_cases(f"main_purpose == '{submodel}'")
       # dt_work = dt_work.query_cases('OTAZ not in [22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300]')
       # dt_work = dt_work.query_cases('DTAZ not in [22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300]')
    






    m = lx.Model(dt_work, compute_engine=compute_engine)
    #m.load(dt_work)
    
    
    modelname = {'Education': 'school_location',
                 'Work': 'work_location'}.get(purpose, 'school_location')
    m.title = f"{modelname} Main Act Location - {submodel} -- {SAFETY}"
    m.choice_co_code = 'tour.tour_mode'
    
    # will run a script to set everything up if the example data is not already available.

    

    # # Load data and prep model for estimation

         # DIST = X('log(od.DIST+1)')




    #modelname="school_location"

    if submodel is None:

        # 'work', 'edu', 'social', 'otherdiscr', 'othmaint', 'shopping', 'escort', 'eat', 'home
       # m.utility_ca = (
     #        +P.shopping *X('RETEMPN')*X('tour.main_purpose in ["shopping", "eat"]')
       #     + P.other * X('FPSEMPN')*X('tour.main_purpose in ["othmaint", "otherdiscr"]')
       # )
        
        
        if SAFETY:
            m.utility_co[Mode.DA] = (
            
                #+ P.Distance_d* X('od.DIST')
                
                
                #+ P.Shopping_Drive *X("main_purpose in ['shopping']")
                #+ P.no_vehicles *X('person.vehrd ==0')
               # + P.UNAVIAL *X('od.DIST >= 60')
                + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                +P("Income:Dr") * X("person.incp") *X("person.incp <= 17")
                + P.crash_random*X('log(od.CRASH+1)'))
                    # + P.MAIN_P_D *X('main_purpose' in ['work', 'edu'])
                    # + P.MAIN_D_SHOPPING *X('main_purpose' in ['shopping'])
                        
                    #  +  P.Licence*X('CARLICENCE')
                    # +  P.WORKER_D*X('WORKSTAT in [2,3]')
                    #  + P.TIME*X.HOV2TOLL_TIME__MD
                
            
        else:
            
            m.utility_co[Mode.DA] = (
                
                    #+ P.Distance_d* X('od.DIST')
                    
                    
                    #+ P.Shopping_Drive *X("main_purpose in ['shopping']")
                    #+ P.no_vehicles *X('person.vehrd ==0')
                # + P.UNAVIAL *X('od.DIST >= 60')
                    + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                    +P("Income:Dr") * X("person.incp") *X("person.incp <= 17")
                    #+ (P.CRASH *X.CRASH if SAFETY else 0)
                # + P.MAIN_P_D *X('main_purpose' in ['work', 'edu'])
                # + P.MAIN_D_SHOPPING *X('main_purpose' in ['shopping'])
                    
                #  +  P.Licence*X('CARLICENCE')
                # +  P.WORKER_D*X('WORKSTAT in [2,3]')
                #  + P.TIME*X.HOV2TOLL_TIME__MD
                    
            )

        if SAFETY:
                m.utility_co[Mode.Other] = (
                    + P.ASC_SR
                    #+P.DISTANCE_s*X('od.DIST')
                    + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                    + P.crash_random *X('log(od.CRASH+1)')
                #  + P.UNAVIAL *X('od.DIST >= 50')
                # + P.shopping_Other *X("main_purpose in ['shopping']")
                # + P.social_Other *X("main_purpose in ['social']")
                # + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                    #+ P.Time_OVT *X('od.DRV_LOC_WLK_WAIT__AM')
                    
                    +P("Income:OTHER") * X("person.incp") *X("person.incp <= 17")
                    
                    
                #+ P("LogIncome:Other") * X("INCOMEN")11
            )
            
        else:
            m.utility_co[Mode.Other] = (
                    + P.ASC_SR
                    #+P.DISTANCE_s*X('od.DIST')
                    + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                     +P.shared_per_trips * X('tour.outbound_stops')
                #  + P.UNAVIAL *X('od.DIST >= 50')
                # + P.shopping_Other *X("main_purpose in ['shopping']")
                # + P.social_Other *X("main_purpose in ['social']")
                # + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                    #+ P.Time_OVT *X('od.DRV_LOC_WLK_WAIT__AM')
                    
                    +P("Income:OTHER") * X("person.incp") *X("person.incp <= 17")
                    
                    
                #+ P("LogIncome:Other") * X("INCOMEN")11
            )

        m.utility_co[Mode.Walk] = (
                + P.ASC_Walk
                + P.Time_Active  * X('od.DISTWALK')
                #+ P.UNAVIAL *X('od.DIST >= 20')
                +P("Income:Active") * X("person.incp") *X("person.incp <= 17")
                +P.walk_per_trips * X('tour.outbound_stops')
               
                
            #3 +# P.Young_Walk *X('AGEGROUP<=3')
            #  P("Income:Walk") * X("person.incp") *X('person.incp <= 17')
                
            # + P("LogIncome:Walk") * X("INCOMEN")
        )

        m.utility_co[Mode.Bike] = (
                + P.ASC_Bike
                + P.Time_Active_b * (X('od.DISTBIKE'))
                + P.bike_ownership *X('person.bikes >=1')
                +P.bike_per_trips * X('tour.outbound_stops')
                +P.shopping_ifbike*X('(tour.n_shopping_out > 0)')
                + P("Income:Active") * X("person.incp") *X("person.incp <= 17")
                
            # + P.Young_Bike *X('AGEGROUP<=3')
                #+ P("LogIncome:Bike") * X("INCOMEN")
        )

        
        if SAFETY:
            m.utility_co[Mode.Transit] = (
                    + P.ASC_Transit
                    
                    #+ P.DISTANCE_T * X.DIST
                    #+ P.Time_INV_T *X('od.DRV_LOC_WLK_TOTIVT__AM')
                    #+ P.Time_OVT *X('od.DRV_LOC_WLK_WAIT__AM')
                    + P.Time_INV_T * X('od.SOVTOLL_TIME__AM')
                    + P("Income:transit") * X("person.incp") *X("person.incp <= 17")  
                    #+ P.CRASH_PT *X('CRASHPT')  
                    
                )
        else:
            m.utility_co[Mode.Transit] = (
                    + P.ASC_Transit
                    
                    #+ P.DISTANCE_T * X.DIST
                    #+ P.Time_INV_T *X('od.DRV_LOC_WLK_TOTIVT__AM')
                    #+ P.Time_OVT *X('od.DRV_LOC_WLK_WAIT__AM')
                    + P.Time_INV_T * X('od.SOVTOLL_TIME__AM')
                    + P("Income:transit") * X("person.incp") *X("person.incp <= 17")    
                    
                )
        #m.set_value('UNAVIAL', -9)
    else:
            #if submodel == 'work':
            #  m.utility_ca = (
            #  + P.worker * X('FPSEMPN')*X('person.indp in [11, 13, 12, 10]')
            # + P.Health_worker *X('HEREMPN')*X('person.indp in [17, 16]') 
           # + P.Agriculture_worker *X('AGREMPN')*X('person.indp in [1,2]')
           # + P.Manufacturing_worker *X('MWTEMPN') * X('person.indp in [3,4,5,6]')
           # + P.Other_worker *X('OTHEMP')*X('person.indp in [19, 14, 15]')
           # )
            
        if SAFETY:
            m.utility_co[Mode.DA] = (
                
                # + P.DISTANCE* X('od.DIST')
                    +P("Income:D") * X("person.incp") *X("person.incp <= 17")
                    + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                    + P.crash_random*X('log(od.CRASH+1)')
                
                    
                )

            m.utility_co[Mode.Other] = (
                    + P.ASC_SR
                    + P.Time_INV * X('SOVTOLL_TIME__AM')      
                    + P("Income:OTHER") * X("person.incp") *X("person.incp <= 17")
                    + P.crash_random *X('log(od.CRASH+1)')
                )
            
        else:
        
            m.utility_co[Mode.DA] = (
                
                # + P.DISTANCE* X('od.DIST')
                    +P("Income:D") * X("person.incp") *X("person.incp <= 17")
                    + P.Time_INV * X('od.SOVTOLL_TIME__AM')

                
                    
            )

            m.utility_co[Mode.Other] = (
                    + P.ASC_SR
                    + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                     +P.shared_per_trips * X('tour.outbound_stops')
                # + P.DISTANCE_o* X('od.DIST')*1.05
                # + P.UNAVIAL *X('DIST >60')
                #  + P.Time_INV * X('SOVTOLL_TIME__AM')
                    #+ P.Time_OVT* X('DRV_LOC_WLK_WAIT__AM')        
                    +P("Income:OTHER") * X("person.incp") *X("person.incp <= 17")
                )
            
            

        m.utility_co[Mode.Walk] = (
                + P.ASC_Walk
                + P.Time_Active  * X('DISTWALK')
                 +P.walk_per_trips * X('tour.outbound_stops')
               # + P.UNAVIAL *X('DISTWALK >= 10')
                +P("Income:Active") * X("person.incp") *X("person.incp <= 17")
                
    
        )

        m.utility_co[Mode.Bike] = (
                + P.ASC_Bike
                + P.Time_Active_b * X('DISTBIKE')
                +P.shopping_ifbike*(X('(tour.n_shopping_out> 0)'))
                
                +P.bike_per_trips * X('tour.outbound_stops')
               
               # + P.UNAVIAL *X('DISTBIKE >= 20')

                + P("Income:Active") * X("person.incp") *X("person.incp <= 17")
                
            # + P.Young_Bike *X('AGEGROUP<=3')
                #+ P("LogIncome:Bike") * X("INCOMEN")
        )

        
        if SAFETY:
             m.utility_co[Mode.Transit] = (
                + P.ASC_Transit
                
                + P.Time_INVT *X('DRV_LOC_WLK_WAIT__AM')
               #+ P.Time_OVT* X('DRV_LOC_WLK_TOTIVT__AM')
                #+ P.DISTANCE_T* X('od.DIST')
                #+ P.UNAVIAL *X('DIST >30')
                
                
                + P("Income:transit") *X("person.incp") *X("person.incp <= 17")
               
                )
        else:
            
            m.utility_co[Mode.Transit] = (
                + P.ASC_Transit
                
                + P.Time_INVT *X('DRV_LOC_WLK_WAIT__AM')
               #+ P.Time_OVT* X('DRV_LOC_WLK_TOTIVT__AM')
                #+ P.DISTANCE_T* X('od.DIST')
                #+ P.UNAVIAL *X('DIST >30')
                
               # + P.Time_INV * X('od.SOVTOLL_TIME__AM')
                + P("Income:transit") *X("person.incp") *X("person.incp <= 17")
               
                )
            

            
            
                
                # + P("LogIncome:Transit") * X('INCOMEN')

    
    # nest strucuture
    Car = m.graph.new_node(parameter='Mu:Car', children=[Mode.DA, Mode.Other], name='Car')
    NonMotor = m.graph.new_node(parameter='Mu:NonMotor', children=[Mode.Walk, Mode.Bike], name='NonMotor')
    Motor = m.graph.new_node(parameter='Mu:Motor', children=[Car, Mode.Transit], name='Motor')
    Root = m.graph.new_node(
    parameter='Mu:Root',
    children=[Motor, NonMotor],
    name='Root')
    
    m.set_value("Mu:Root", 1)
    #m.lock_value("Mu:Root")
    print('uncommenting')

    if SAFETY:
        m.mixtures.append(
        lx.mixtures.Normal("crash_random", "std_crash")
        )
        m.n_draws = 200
        m.seed = 42

    m.availability_co_vars = {
        Mode.DA: 1,
        Mode.Other: 1,
        Mode.Walk: X('DISTWALK < 10'),
        Mode.Bike: X('DISTWALK < 25'),
        Mode.Transit: 1,
    }
    #m.set_bounds("Mu:Car", 0.01, 1)
    #model.set_bounds('Mu:Car', lower=0.01, upper=1.0)
    #m.set_bounds("Mu:Motor", 0.01, 1)
    #m.set_bounds("Mu:NonMotor", 0.01, 1)
    
    #m.set_value("UNAVIAL", -99)
    #m.set_value(UNAVIAL = -999)
   
    
    m.choice_avail_summary()
    print(m.choice_avail_summary())
    #m.set_value("UNAVIAL", -9)
    m.set_cap(10) # improves optimization stability
    m.doctor(repair_ch_av = '+')
    m.doctor(repair_nan_data_co="True")
   # m.doctor(repair_ch_av = '?')
   # m.doctor()
    #m.doctor()
    print(m.choice_avail_summary())
    result = m.maximize_loglike()
    m.calculate_parameter_covariance()
    print('Mode Choice model')
   

    # Assuming `table` is your rich Table object
    console = Console()
    console.print(result)  #
    print(result)
    m.parameter_summary()

    
    report = lx.Reporter(title=m.title)
    report << "# Parameter Summary" << m.parameter_summary()
    report << "# loglike" << m.estimation_statistics()
    m.utility_functions()

    report.save(
        f'Brisbane_main_mode{purpose}-{submodel}-{SAFETY}.html',
        overwrite=True,
        metadata=m,
    )



    m.parameter_summary()
    # ## Write updated utility coefficients


    m.probability()
    logging.info('prediction test turn back on ')
   
    logsums = lx.DataArray.zeros(
        dt_work.caseids(),
        skims.root.lookup.ids,
        name="logsums",
    )
    
    
    template = dt_work._getitem("DTAZ")
    for i, dtaz in enumerate(logsums.ids):
        m.datatree = dt_work.replace_datasets(
            tour=dt_work.root_dataset.assign(
                DTAZ=xr.full_like(template, dtaz)
                ),
        )
        logsums.loc[dict(ids=dtaz)] = m.logsums()
        
    
    
    if logsums.isnull().any():
        logsums = logsums.fillna(0)
        print("NaN detected in logsums")
  
    #### THIS IS THE PREDICTIONS BIT
    
    #setup_prediction_dataset..
    #### PREDICTION START ######
    
    logging.info('get population data')
    
    
    #pop_data = pd.read_csv('Z:/sa_run_again_n/zones_sa_4/31004129921_synthetic_population.csv')
   # pop_data['DTAZ'] = -1
    #pop_data['SA1'] = 31004129921
    
    # get data
    #choice_id_set_zones = list(range(1, 330))
    logging.info('choice id set_range... update this so its not dependenetn')
    choice_id_set_zones = pd.DataFrame(index=range(1, 331))
    '''
    pop_data, persons_ex = Synth_Connect().synth_data_with_M(choice_id_set_zones)
    '''
    # this where synth_data_test_stage_3 was done,,,
    if purpose == 'N':
        pop_data, persons_ex = Synth_Connect().synth_data_with_M(choice_id_set_zones, 'N')
    else:
        pop_data, persons_ex = Synth_Connect().synth_data_with_M(choice_id_set_zones, 'M')
    
    #persons_ex['main_act'] = 
    #['work'] if persons_ex['n_work_out'], persons_ex['n_edu_out'] else 'edu', 'n_social_out', 'n_other_out'
    conditions = [
        persons_ex['n_work_out'] > 0,
        persons_ex['n_edu_out'] > 0,
        persons_ex['n_social_out'] > 0,
        persons_ex['n_othdiscr_out'] > 0,
        persons_ex['n_othmaint_out'] > 0,
        persons_ex['n_shopping_out'] > 0,
        persons_ex['n_escort_out'] > 0,
        persons_ex['n_eat_out'] > 0,
        persons_ex['n_escort_out'] >= 0,
        
    ]
    choices = ['work', 'edu', 'social', 'otherdiscr', 'othmaint', 'shopping', 'escort', 'eat', 'home']
    persons_ex['main_purpose'] = np.select(conditions, choices, default=None)
    
    
    pop_data = Synth_Connect().generate_alternatives(persons_ex, 'DEST_SAMPLE')
    #pop_data = pop_data.query_cases('STUDYING in [1,2,3,4]')
    
    
    
    
    if purpose == 'N':
        pred_tours = pop_data[pop_data["ACT"].isin(['N'])]
        
    else:
        logging.info('anyone else')
        pred_tours = pop_data[pop_data["ACT"].isin(['M'])]
    
    if submodel is not None:
        pred_tours = pred_tours[pred_tours["main_purpose"].isin([submodel])]
        pred_tours_final = pred_tours.copy()
    else:
        pred_tours_final = pred_tours.copy()
    
    
        
    pred_tours['ptour_id'] = pred_tours['person_id'].astype(str) + "_1"
    
    #generate DTAZ options..add(
    # )
    #pred_tours['DTAZ'] = 2     # unknown destination
    #pred_tours['OTAZ'] = pred_tours['OTAZ']  # must exist
    pred_tours['tour_mode'] = -1  # placeholder
    #alts = skims.root.lookup.ids  # same zone numbering as estimation
    #alts =  list(map(int, alts))
    tours_pred = pred_tours.copy()
    #add
    tours_pred = tours_pred.set_index("ptour_id")
    tours_pred_ds= lx.Dataset.construct.from_idco(
    tours_pred,
    alts=Mode   # SAME dict used in estimation
    )
    pop_data = Synth_Connect().generate_alternatives(persons_ex, 'DEST_SAMPLE')
    
    
    # not yet this is only the person
    pop_data['HOMEZONE'] = pop_data['OTAZ']
    tree_pred = lx.DataTree(
    #base = ca_pred,
    tour = tours_pred_ds,
    person = pop_data.set_index('person_id'),
    emp = zone_id.set_index("ORDER"),
    od = od_skims,
    #od_skim = od_skims,
    relationships = (
        #"base.ids @ emp.ORDER",
        #"base.ptour_id @ tours.ptour_id",
        "emp.ORDER @ tour.DTAZ",
        "tour.person_id @ person.person_id",
        "person.OTAZ @ tour.OTAZ",
        "tour.OTAZ @ od.otaz",
        "tour.DTAZ   @ od.dtaz",
    #  "person.OTAZ  @ od_skim.dtaz",
     #  "tours.DTAZ   @ od_skim.otaz",
        #"base.ids @ skims.dtaz",
    ),
    )
    
    if submodel is None:
        if purpose == 'N':
            tree_p = tree_pred.query_cases('main_purpose not in ["work", "edu"]')
            #tree_p = tree_pred.query_cases('OTAZ in [39,40,41,42]')
        else:
            tree_p = tree_pred.query_cases('main_purpose in ["edu", "work"]')
            #tree_p = tree_pred.query_cases('OTAZ in [39,40,41,42]')
    else:
        tree_p = tree_pred.query_cases(f"main_purpose == '{submodel}'")
    
    logsums_pred = lx.DataArray.zeros(tree_p.caseids(),
        skims.root.lookup.ids,
        name="logsums",
        
    )
    # LETS see IF THIS WORKS
    m.datatree = tree_p
    m.mangle()
    m.unmangle()
    checkthepreds = m.probability()
    
    predicted_synthmode = monte_sample(checkthepreds)+1
    
    
    
    #lets check this 
    print('checking predict logsums')
    OLD = 0
    if OLD:
        for dtaz in logsums_pred.ids:
            m.datatree = tree_p.replace_datasets(tour=tree_p.root_dataset.assign(DTAZ=xr.full_like(tree_p._getitem("DTAZ"), dtaz)))
            logsums_pred.loc[dict(ids=dtaz)] = m.logsums()
    else:
        
        # main way
        template = tree_p._getitem("DTAZ")
    
        for dtaz in logsums_pred.ids.values:
            if dtaz % 50 == 0:
                print('predicting dtaz.. ', dtaz)
            #print('dtaz ', dtaz)
            m.datatree = tree_p.replace_datasets(
                tour=tree_p.root_dataset.assign(
                    DTAZ=xr.full_like(template, dtaz)
                )
            )
            logsums_pred.loc[dict(ids=dtaz)] = m.logsums()
        
        
       


        


        # Run comparison
        

       
        
        # the alternative way
        '''
        template = tree_p._getitem("DTAZ")

        for dtaz in logsums_pred.ids.values:
            if dtaz % 50 == 0:
                print('predicting dtaz.. ', dtaz)
            #print('dtaz ', dtaz)
            m.datatree = tree_p.replace_datasets(
                tour=tree_p.root_dataset.assign(
                    DTAZ=xr.full_like(template, dtaz)
                )
            )
            logsums_pred.loc[dict(ids=dtaz)] = m.logsums()
        
        '''
        
    

    print('finished loop')



    ca = lx.Dataset.construct(
        {"logsum": logsums},
        caseid="ptour_id",
        alts=skims.root.lookup.ids,
    )
    

    # destination choice part

    #print('cool')


    #schools 
   
    logging.info('reload tour')
    tour = Synth_Connect().read_in_tours()
    # drop DTAZ 330
    tour = tour[tour['DTAZ'] != 0]
    tour = tour[tour['OTAZ'] != 0]
    tour = tour[tour['OTAZ'] != 330]
    tour = tour[tour['DTAZ'] != 330]
    tour['main_purpose'] = tour['main_purpose'].fillna('othmaint')
    tour['tour_mode'] = tour['tour_mode'].map(Mode).fillna(2)
    tour['ptour_id'] =tour['person_id'].astype(str) + '_' + tour['tour_id'].astype(str)
    school_t = tour.set_index('ptour_id')
    #pp if OTAZ in pp drop that column, same as DTAZ in pp
    cols_to_drop = [c for c in ["OTAZ", "DTAZ"] if c in pp.columns]
    pp = pp.drop(columns=cols_to_drop)
    
    logging.info('only work or edu')
    if submodel is None:
        if purpose == 'M':
            tour = school_t[school_t["main_purpose"].isin(["edu", "work"])]
           # tour = tour[~tour["OTAZ"].isin([22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300])]
          #  tour = tour[~tour["DTAZ"].isin([22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300])]
             #tree_p = tree_pred.query_cases('OTAZ in [39,40,41,42]')
        
        else:
            tour = school_t[~school_t["main_purpose"].isin(["edu", "work"])]
         #   tour = tour[~tour["OTAZ"].isin([22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300])]
         #   tour = tour[~tour["DTAZ"].isin([22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300])]
    else:
        tour = school_t[school_t["main_purpose"] == submodel]
       # tour =  tour[~tour["OTAZ"].isin([22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300])]
       # tour = tour[~tour["DTAZ"].isin([22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300])]
    #tour = school_t[school_t['main_purpose'] in ['edu', 'work']]
    
    #[22,55,59,68,97,118,126,127,149,163,198,211,220,224,240,254,265,286,303,16,159,206,207,242,291,299,300]
    
    # Extract observed choices (actual destination zones)
    #observed_choices = co.to_dataframe().reset_index()[['ptour_id', 'DTAZ', 'ids']]
    # bane jkeep only the choices that are ids == DTAZ
   # observed_choices = observed_choices[observed_choices['ids'] == observed_choices['DTAZ']]
    
    # this was reloaded, muight cause issues
    #"logsum":reloaded
    '''
    ca = lx.Dataset.construct(
        {"logsum":reloaded},
        caseid='ptour_id',
        alts=skims.root.lookup.ids,
    )
    '''

    
    
    '''
    co_df = co.to_dataframe().reset_index()
    logsums_df = logsums.to_dataframe().reset_index()
    co = co_df.merge(
    logsums_df,
    left_on=["ptour_id", "ids"],
    right_on=["ptour_id", "ids"],
    how="left"
    )
    
    #add zone id
 
    '''
    
    
    
    #zone_id['COLLFTE'] =zone_id['COLLFTE'] +zone_id['OBS_EDU_TRIPS']
    #zone_id['COLLPTE'] =zone_id['COLLPTE'] +zone_id['OBS_EDU_TRIPS']
    #zone_id['OBS_EDU_TRIPS']
   # zone_id['OBS_WORK_TRIPS'] = zone_id['OBS_WORK_TRIPS']*2+2
    # how to merge in in logsums.. into co
    tours_dest=lx.Dataset.construct.from_idco( tour.reset_index(),
        alts=skims.root.lookup.ids)
    tree = lx.DataTree(
        
        base=ca,
    # tour = tour_dataset.rename_axis('TOUR_ID')
        #hh=hh.set_index("HHID"),
        tour=tour,
       # base = ca,
        person=pp.set_index('person_id'),
        emp=zone_id.set_index('ORDER'),
        emp_orig = zone_id.set_index('ORDER'),
        od=od_skims,

        #do_skims = od_skims,
        relationships=(
            #'tour.ids@base.ids',
            "base.ids @ emp.ORDER",
            
            "base.ptour_id @ tour.ptour_id",
           # "emp.ORDER @  tour.DTAZ",
            "tour.person_id @ person.person_id",
           # "person.HOMEZONE @ tour.OTAZ",
            'base.ids @ od.dtaz',
            "tour.OTAZ @ od.otaz",
            #"tour.OTAZ @ emp_orig.ORDER",
            #"tour.DTAZ @ od.dtaz",

            #"person.OTAZ @ do_skims.dtaz",
            #"tour.DTAZ @ do_skims.otaz",
            #"base.ids @tour.DTAZ",
            
            
            

        ),
    )

    
    '''
    
    mm.probability()'base.ids @ emp.ORDER',
            'emp.ORDER @ tour.DTAZ',
            "base.ptour_id @ tour.ptour_id",
            "tour.person_id @ person.person_id",
            "person.HOMEZONE @ tour.OTAZ",
            "tour.OTAZ @ od.otaz",
            "tour.DTAZ @ od.dtaz",
    
     mm.utility_ca =  (+P.logsums *X('base.logsum')+P.distance * X('od.DIST') +P.UNA*X('od.DIST >= 35') +P.Crash*X('od.CRASH')
    '''


    #tree = tree.query_cases("main_purpose == 'Education'")

    mm = lx.Model(datatree=tree, compute_engine=compute_engine)
    mm.choice_co_code = "tour.DTAZ"
    mm.title = f"{purpose} Tour Destination Main{SAFETY}"
    
    '''
     mm.quantity_ca = (
        +P.EDUCATION * (X("AGE0519") * X('STUDYING in [1]')*X('AGEGROUP <= 2')
        +P.GRADE_HIGHSCHOOL *X("HSENROLL") * X('STUDYING in [2]')
        + P.GRADE_UNI *X('COLLFTE')*X ('STUDYING in [3]')
        + P.GRADE_UNI_PT *X('COLLPTE')*X('STUDYING in [4]'))
    '''
    
    ## SCHOOL MODEL SPECIFICATION
    if submodel == 'edu': 
    
        '''
        mm.quantity_ca = (
        +P.EDUCATION_S * X("emp.AGE0519") * X('person.STUDYING in [1]')
        +P.EDUCATION_H *X("emp.HSENROLL") * X('person.STUDYING in [2]')
        #+P.EDUCATION_UNI* X('emp.COLLFTE')*X ('person.STUDYING in [3]')
        
        '''
       # +P.EDUCATION_PT_UNI* X('emp.COLLPTE')*X('person.STUDYING in [4]'))
        
        '''
        mm.quantity_ca = (
        +P.EDUCATION_S * X("emp.AGE0519") * X('person.STUDYING in [1]')
        +P.EDUCATION_H *X("emp.HSENROLL") * X('person.STUDYING in [2]'))
        '''
        #mm.quantity_ca = (
       # +P.EDUCATION_early_q * (X("emp.Primary_total_priv") +X("emp.Primary"))* X('person.STUDYING in [1]')
       # +P.EDUCATION_highschool_q *(X("emp.Secondary_total_priv") +X("emp.Secondary")) * X('person.STUDYING in [2]'))
       # +P.EDUCATION_post_school_ft *X("emp.OBS_edu_TOURS") * X('person.STUDYING in [3]')
       # +P.EDUCATION_post_school_pt *X("emp.OBS_edu_TOURS") * X('person.STUDYING in [4]'))
        
        #+P.EDUCATION_UNI* X('emp.COLLFTE')*X ('person.STUDYING in [3]')
        #+P.Total_STUDY1 *X("TOTPOP") * X('STUDYING == 1')
        
        #mm.quantity_scale = P.Theta
        if SAFETY:
        
            mm.utility_ca = ( P.logsums *X('base.logsum')+ P.distance * X('log(od.DIST+1)') +P.crash_random*X('log(od.CRASH+1)')*X('tour.tour_mode in [1,2,5]') + P.crash_dist_interact *X('log(od.CRASH+1)')*X('log(od.DIST+1)')*X('tour.tour_mode in [1,2,5]')
                             +P.educational_post_school*X('emp.Education') *X('person.STUDYING in [3,4]')
                             +P.EDUCATION_early_q_enrolments * (X("log(emp.Primary_total_priv+1)") +X("log(emp.Primary+1)"))* X('person.STUDYING in [1]')+
                             +P.EDUCATION_highschool_enrolments *(X("log(emp.Secondary_total_priv+1)") +X("log(emp.Secondary+1)")) * X('person.STUDYING in [2]')
                            +P.educational_post_area*X('log(emp.education+1)') *X('person.STUDYING in [3,4]')
                            +P.educatioal_buildings*X('emp.Education')+
                             +P.educational_area*X('log(emp.education+1)')+ P.origin_distance_interact * X('log(od.DIST+1)') * (1 / (X('emp.education+1'))))
        else:
            
            mm.utility_ca = (  P.logsums *X('base.logsum')+ P.distance * X('log(od.DIST+1)') 
                             +P.educational_post_school*X('emp.Education') *X('person.STUDYING in [3,4]')
                             +P.EDUCATION_early_q_enrolments * (X("log(emp.Primary_total_priv+1)") +X("log(emp.Primary+1)"))* X('person.STUDYING in [1]')+
                             +P.EDUCATION_highschool_enrolments *(X("log(emp.Secondary_total_priv+1)") +X("log(emp.Secondary+1)")) * X('person.STUDYING in [2]')
                            +P.educational_post_area*X('log(emp.education+1)') *X('person.STUDYING in [3,4]')
                            +P.educatioal_buildings*X('emp.Education')+
                             +P.educational_area*X('log(emp.education+1)')+ P.origin_distance_interact * X('log(od.DIST+1)') * (1 / (X('emp.education+1'))))
        
       # mm.set_value("UNA", -999)
      #  +P.FT_S *X("COLLFTE")*X("AGEGROUP >= 5")
    elif submodel == 'work':
        
        """
        mm.quantity_ca = (
        
        + P.Retail_worker *X('emp.RETEMPN')*X('person.indp in [7, 8]')
        #+ P.Time * X('od.SOVTOLL_TIME__AM')*X('tour.tour_mode in [1,2,5]') 
        #+ P.Time_ACTIVE*(X('od.DIST')*X('tour.tour_mode ==3') ) 
        #+ P.Time_BIKE*(X('od.DISTBIKE')*X('tour.tour_mode ==4')) 
        + P.worker * (X('emp.FPSEMPN'))*X('person.indp in [11, 13, 12, 10]')   
        + P.Health_worker *(X('emp.HEREMPN')) *X('person.indp in [17, 16]')  
        + P.Agriculture_worker* (X('emp.AGREMPN')) *X('person.indp in [1,2]')  
        + P.Manufacturing_worker * (X('emp.MWTEMPN')) * X('person.indp in [3,4,5,6]') 
        + P.Other_worker * (X('emp.OTHEMP')) *X('person.indp in [19, 14, 15]')
        )
        
        for d in mm.graph.elementals:

            # example membership rules
            if zone_distance[d] < 5:
                alpha_near = 0.7
            else:
                alpha_near = 0.3

            alpha_far = 1 - alpha_near

            if cbd_flag[d] == 1:
                alpha_cbd = 0.8
            else:
                alpha_cbd = 0.2

            alpha_sub = 1 - alpha_cbd

            mm.graph.add_edge(Near, d, allocation=alpha_near)
            mm.graph.add_edge(Far, d, allocation=alpha_far)
            mm.graph.add_edge(CBD, d, allocation=alpha_cbd)
            mm.graph.add_edge(Suburban, d, allocation=alpha_sub)
        
        
        """
        
        
       # mm.quantity_ca = (
        
        #+P.working_quantity * X('emp.OBS_work_TOURS'))
        """
        + P.Retail_worker *X('emp.OBS_work_TOURS')*X('person.indp in [7, 8]')
        #+ P.Time * X('od.SOVTOLL_TIME__AM')*X('tour.tour_mode in [1,2,5]') 
        #+ P.Time_ACTIVE*(X('od.DIST')*X('tour.tour_mode ==3') ) 
        #+ P.Time_BIKE*(X('od.DISTBIKE')*X('tour.tour_mode ==4')) 
        + P.Finance_worker* (X('emp.OBS_work_TOURS'))*X('person.indp in [11, 13, 12, 10]')   
        + P.Health_worker *(X('emp.OBS_work_TOURS')) *X('person.indp in [17, 16]')  
        + P.Agriculture_worker* (X('emp.OBS_work_TOURS')) *X('person.indp in [1,2]')  
        + P.Manufacturing_worker * (X('emp.OBS_work_TOURS')) * X('person.indp in [3,4,5,6]') 
        + P.Other_worker * (X('emp.OBS_work_TOURS')) *X('person.indp in [19, 14, 15]')
        ( P.logsums *X('base.logsum')+ P.distance * X('log(od.DIST+1)') +P.crash_random*X('log(od.CRASH+1)')*X('tour.tour_mode in [1,2,5]') + P.crash_dist_interact *X('log(od.CRASH+1)')*X('log(od.DIST+1)')*X('tour.tour_mode in [1,2,5]')
                             +P.educational_post_school*X('emp.Education') *X('person.STUDYING in [3,4]')
                             +P.EDUCATION_early_q_enrolments * (X("log(emp.Primary_total_priv+1)") +X("log(emp.Primary+1)"))* X('person.STUDYING in [1]')+
                             +P.EDUCATION_highschool_enrolments *(X("log(emp.Secondary_total_priv+1)") +X("log(emp.Secondary+1)")) * X('person.STUDYING in [2]')
                            +P.educational_post_area*X('log(emp.education+1)') *X('person.STUDYING in [3,4]')
                            +P.educatioal_buildings*X('emp.Education')+
                             +P.educational_area*X('log(emp.education+1)')+ P.origin_distance_interact * X('log(od.DIST+1)') * (1 / (X('emp.education+1'))))
        """
        ##)
        
        
        #+P.EDUCATION_UNI* X('emp.COLLFTE')*X ('person.STUDYING in [3]')
        if SAFETY:
            
            mm.utility_ca =  (+P.logsums *X('base.logsum')+P.distance * X('log(od.DIST+1)') +P.crash_random*X('log(od.CRASH+1)')*X('tour.tour_mode in [1,2,5]') + P.crash_dist_interact *X('log(od.CRASH+1)')*X('log(od.DIST+1)')*X('tour.tour_mode in [1,2,5]')
                              +P.finance_jobs*X('log(emp.FPSEMPN+1)')*X('person.indp in [11, 13, 12, 10]') +P.health_jobs*X('log(emp.HEREMPN+1)')*X('person.indp in [17, 16]')+P.agriculture_jobs*X('log(emp.AGREMPN+1)')*X('person.indp in [1,2]')+
                              P.Manufacture_jobs*X('log(emp.MWTEMPN+1)')*X('person.indp in [3,4,5,6]')+P.other_jobs*X('log(emp.OTHEMP+1)')*X('person.indp in [19, 14, 15]') +P.retail_job*X('log(emp.RETEMPN+1)')*X('person.indp in [7, 8]') +P.industrial_area*X('log(emp.industrial+1)')
            +P.construction_area*X('log(emp.construction+1)')+P.commercial_area*X('log(emp.commercial+1)')+P.education_area*X('log(emp.education+1)')+P.retrail_area*X('log(emp.retail+1)')+ P.origin_distance_interact * X('log(od.DIST+1)') * (1 / (X('emp.education+1')+X('emp.commercial+1')+X('emp.industrial+1')+X('emp.retail+1')+X('emp.construction+1')))
            )
                              
        else:
            mm.utility_ca =  (+P.logsums *X('base.logsum')+P.distance * X('log(od.DIST+1)') 
                              +P.finance_jobs*X('log(emp.FPSEMPN+1)')*X('person.indp in [11, 13, 12, 10]') +P.health_jobs*X('log(emp.HEREMPN+1)')*X('person.indp in [17, 16]')+P.agriculture_jobs*X('log(emp.AGREMPN+1)')*X('person.indp in [1,2]')+
                              P.Manufacture_jobs*X('log(emp.MWTEMPN+1)')*X('person.indp in [3,4,5,6]')+P.other_jobs*X('log(emp.OTHEMP+1)')*X('person.indp in [19, 14, 15]') +P.retail_job*X('log(emp.RETEMPN+1)')*X('person.indp in [7, 8]') +P.industrial_area*X('log(emp.industrial+1)')
            +P.construction_area*X('log(emp.construction+1)')+P.commercial_area*X('log(emp.commercial+1)')+P.education_area*X('log(emp.education+1)')+P.retrail_area*X('log(emp.retail+1)')+ P.origin_distance_interact * X('log(od.DIST+1)') * (1 / (X('emp.education+1')+X('emp.commercial+1')+X('emp.industrial+1')+X('emp.retail+1')+X('emp.construction+1')))
            )
            
            
   
        
                                            
        
   
        
    elif purpose == 'N':
        logging.info('non work non edu location model')
        print('non mandatory work')
        
 
        
        #mm.quantity_scale = P.Theta
        
        #+ P.SHOP_SIZE * X('emp.RETEMPN') * X('tour.main_purpose == "shopping"')
        #                  + P.SOC_SIZE * X('emp.TOTPOP') * X('tour.main_purpose == "social"'
        if SAFETY:
            mm.utility_ca =  (+P.logsums *X('base.logsum')+ P.distance * X('log(od.DIST+1)') +P.crash_random*X('log(od.CRASH+1)')*X('tour.tour_mode in [1,2,5]') + P.crash_dist_interact *X('log(od.CRASH+1)')*X('log(od.DIST+1)')*X('tour.tour_mode in [1,2,5]')
                              +P.shopping*X('emp.Shopping')*X('tour.main_purpose in ["shopping"]')+P.escorting*X('emp.ESCORT')*X('tour.main_purpose in ["escort"]')
                              +P.social*X('emp.Entertainment')*X('tour.main_purpose in ["social"]') +P.other *X('emp.OTHERMAINTENANCE')*X('tour.main_purpose in ["othmaint"]')
                              +P.other_disc *X('emp.OTHERDISCRFacilities')*X('tour.main_purpose in ["othdiscr"]')
                              +P.residentail_att*X('log(emp.residential+1)')+ P.retail_att*X('log(emp.retail+1)')+P.CBD*X('emp.city')+ P.eating*X('emp.EATOUT')*X('tour.main_purpose in ["shopping"]')
                              + P.origin_distance_interact * X('log(od.DIST+1)') * (1 / (X('emp.education+1')+X('emp.commercial+1')+X('emp.retail+1')))
                              )
        else:
            
            mm.utility_ca =  (+P.logsums *X('base.logsum') + P.distance * X('log(od.DIST+1)') +P.shopping*X('emp.Shopping')*X('tour.main_purpose in ["shopping"]')+P.escorting*X('emp.ESCORT')*X('tour.main_purpose in ["escort"]')+
                                P.social*X('emp.Entertainment')*X('tour.main_purpose in ["social"]') +P.other *X('emp.OTHERMAINTENANCE')*X('tour.main_purpose in ["othmaint"]')+P.other_disc *X('emp.OTHERDISCRFacilities')*X('tour.main_purpose in ["othdiscr"]')
                                +P.residentail_att*X('log(emp.residential+1)')+ P.retail_att*X('log(emp.retail+1)')+P.CBD*X('emp.city')+ P.eating*X('emp.EATOUT')*X('tour.main_purpose in ["shopping"]')
                                 + P.origin_distance_interact * X('log(od.DIST+1)') * (1 / (X('emp.education+1')+X('emp.commercial+1')+X('emp.retail+1')))
                                
                         )
                     
       # mm.quantity_ca = (P.Quantity *X('main_purpose in ["social"]')
        #mm.set_value("UNA", -9)
        logging.info('non mandatory work location model')
    
        
        
    else:
        ## WORK MODEL SPECIFICATION
        logging.info('work location model')
        mm.quantity_ca = (
        +P.EMP * ( X("OTHEMP")+X('AGREMPN')+ X('RETEMPN'))
        +P.TOT*X("TOTPOP"))
    
    """
        m.availability_ca_var = {
        X("AGE0519>=0")
    }

    """

    
    #mm.utility_ca = ( +P.logsums *X.logsum + P.distance * X.DIST +P.Total *X.TOTPOP )
    print('check_tips')
   # mm.utility_ca =  (+P.logsums *X.logsum + P.distance_work * X.DIST *X('main_purpose in ["work"]') + P.distance_edu * X.DIST *X('main_purpose in ["edu"]') + P.distance_shopping * X.DIST *X('main_purpose in ["shopping"]') 
      #  + P.distance_other * X.DIST *X('main_purpose in ["othdiscr", "othmaint"]') +P.CRASH*X.CRASH*X('tour_mode in [1,2]') +P.CRASHPT*X.CRASHPT*X('tour_mode in [5]'))
    
  
    '''
    mm.utility_ca =  (+P.logsums *X.logsum + P.distance_work * X.DIST *X('main_purpose in ["work"]') + P.distance_edu * X.DIST *X('main_purpose in ["edu"]') 
       +P.CRASH*X.CRASH*X('tour_mode in [1,2]') +P.CRASHPT*X.CRASHPT*X('tour_mode in [5]'))                                                                     
    '''
   # mm.choice_co_code = "tour.DTAZ"
    mm.availability_ca_var = X('od.DIST < 150')
    '''
    if SAFETY:
        mm.mixtures.append(
        lx.mixtures.Normal("crash_random", "s_distance")
        )
        mm.n_draws = 200
        m.seed = 42
    else:
        mm.mixtures.append(
        lx.mixtures.Normal("distance", "s_distance")
        )
        mm.n_draws = 200
        mm.seed = 42
    '''
        
                               
   
    #mm.set_value('UNA', -9)
    #mm.d_loglike()
    print('doing')
    mm.set_cap(10) # improves optimization stability
    #m.doctor(repair_ch_av = '+')
    #m.doctor(repair_nan_data_co="True")
    
   
    mm.doctor(repair_ch_av = "+")
    #mm.doctor(repair_ch_zq="?")
   # mm.doctor(repair_ch_zq="!")
    #mm.doctor(repair_nan_data_co="True")
    #mm.doctor(repair_ch_av = "+")
               
    mm.doctor(repair_ch_zq = "-")
    
    mm.doctor(repair_nan_utility= True)
    #mm.doctor(repair_ch_av = "+")
    #mm = mm.query_cases("quantity_ca > 0", level="ca")
    #mm.doctor(repair_ch_av = "+")
   # mm.set_cap(10)
    #print(mm.choice_avail_summary())
    #mm.loglike()
    #print(mm.choice_avail_summary())
    r = mm.maximize_loglike()
    mm.calculate_parameter_covariance()
    print('did it do it')
    

    # Assuming `table` is your rich Table object
    console = Console()
    console.print(r)  #
    print(r)
    mm.parameter_summary()

    print('done')
    report = lx.Reporter(title=mm.title)
    report << "# loglike" << mm.estimation_statistics()
    report << "# Parameter Summary" << mm.parameter_summary()

    mm.utility_functions()

    report.save(
        f'Brisbane_DEST_choice{purpose}-{submodel}-{SAFETY}.html',
        overwrite=True,
        metadata=mm,
    )


    logging.info('now lets predict')
    inspect_the_prob = mm.probability() #(id, zone_id)
    # if inspect the prob is nan, majke each dimernision equal un prob.
    
    predicted = monte_sample(inspect_the_prob)
    print('check predicted')
    
    ca_pred = lx.Dataset.construct(
    {"logsum": logsums_pred},  # or recompute logsums
    caseid="ptour_id",
    alts=skims.root.lookup.ids,
    )  
    
    
    
    #external_datap, person_data = Synth_Connect().synth_data_with_M(choice_id_set_zones, purpose)
    
    #person_data = person_data[person_data["ACT"].isin(['M' if purpose == 'M' else 'N'])]
    #if submodel is not None:
    person_data = pred_tours_final
    
    #external_datap, person_data = Synth_Connect().synth_data_test_stage_2(choice_id_set_zones)
    
    #external_data['DTAZ'] = -1
    pred_tours['tour_mode'] = predicted_synthmode 
    external_data = lx.DataTree(
        base = ca_pred,
        tour = pred_tours.set_index("ptour_id"),
        person =pop_data.set_index('person_id'),
        emp = zone_id.set_index("ORDER"),
        od = od_skims,
      
       # do_skims = od_skims,
        relationships = (
            'base.ids @ emp.ORDER',
            
           # 'emp.ORDER @ tour.DTAZ',
            "base.ptour_id @ tour.ptour_id",
            "tour.person_id @ person.person_id",
            #"person.HOMEZONE @ tour.OTAZ",
            'base.ids @ od.dtaz',
            "tour.OTAZ @ od.otaz",
            
            #"tour.DTAZ @ od.dtaz",
    
           # "person.OTAZ @ do_skims.dtaz",
            #"tour.DTAZ @ do_skims.otaz",
            
            
            #'zone.ORDER @ od.dtaz',
        )
            
    )
    #mm.choice_co_code = "tour.chosen"
    
    ''''''
    #external_data -> generate_alts...
    #get_data = synt_data(choice_set)
    #main purpise is not defined
    #m.datatree = get_data
   
    mm.datatree = external_data
    mm.mangle()
    mm.unmangle()
    probs = mm.probability()
    #probs = m.probability()
    all_nan_rows = pd.DataFrame(probs).isna().all(axis=1)
    if purpose == 'N':
        predicted_synth = monte_sample_key(probs, 50)
    else:
        if submodel == 'work':
            #OTAZ =  zone_id[zone_id['OBS_WORK_TRIPS'].idxmax()]['ORDER']
            predicted_synth = monte_sample_key(probs, 50)
        
        elif submodel == 'edu':
            #OTAZ =  zone_id[zone_id['HSENROLL'].idxmax()]['ORDER']
            predicted_synth = monte_sample_key(probs, 30)
   
    predicted_synth[all_nan_rows] = OTAZ
    
    
    choice_set = zone_id_i.copy()
    a = choice_set.iloc[predicted_synth].reset_index(drop=True)
    a = a.rename(columns={"HOMEZONE": "DTAZ"})
    print('need to check this part')
    joined = pd.concat(
    [
        person_data.reset_index(drop=True).drop(columns=['DTAZ']),
        choice_set.iloc[predicted_synth].reset_index(drop=True).rename(columns={"HOMEZONE": "DTAZ"})
    ],
    axis=1
    )
    joined['tour_mode'] = predicted_synthmode
    
    #now predict mode again
    joined = joined.loc[:, ~joined.columns.duplicated()]
    if SAFETY:
        joined.to_csv(f'saved_data/activities_locations_synthetic_population_{purpose}.csv', index=False)
    else:
        joined.to_csv(f'saved_data/activities_locations_synthetic_population_{purpose}_NOS.csv', index=False)
    joined['ptour_id'] =joined['person_id'].astype(str) + f'_{purpose}' + '_1'
    joined['HOMEZONE'] = joined['OTAZ']
    joined = joined.set_index('ptour_id')
    print('check this for new joined')
    joined_a = joined.rename(columns={"incp": "incp_a"})
    tours_pred_ds= lx.Dataset.construct.from_idco(
    joined_a,
    alts=Mode   # SAME dict used in estimation
    )
    tree_pred = lx.DataTree(
    #base = ca_pred,
    tour= tours_pred_ds,
    person = joined.reset_index().set_index('person_id'),
    emp = zone_id.set_index("ORDER"),
    od = od_skims,
    #od_skim = od_skims,
    relationships = (
        #base.id
        "tour.DTAZ@ emp.ORDER",
        #"base.ptour_id @ tours.ptour_id",
        #"emp.ORDER
        "tour.person_id @ person.person_id",
        "person.HOMEZONE @ tour.OTAZ",
        "tour.OTAZ @ od.otaz",
        "tour.DTAZ   @ od.dtaz",

    #  "person.OTAZ  @ od_skim.dtaz",
     #  "tours.DTAZ   @ od_skim.otaz",
        
    ),
    )
    
    if submodel is None:
        if purpose == 'N':
            tree_p = tree_pred.query_cases('main_purpose not in ["work", "edu"]')
        else:
            tree_p = tree_pred.query_cases('main_purpose in ["edu", "work"]')
    else:
        tree_p = tree_pred.query_cases(f'main_purpose == "{submodel}"')
    
    logsums_pred = lx.DataArray.zeros(tree_p.caseids(),
        skims.root.lookup.ids,
        name="logsums",
        
    )
    # LETS see IF THIS WORKS
    m.datatree = tree_p
    m.mangle()
    m.unmangle()
    
    
    print('now lets add in the predicted mode')
    probs = m.probability() #(id, zone_id)
    predicted_mode = monte_sample_key(probs, 2) + 1
    joined['tour_mode'] = predicted_mode
    
    if SAFETY:
        joined.to_csv(f'saved_data/activities_locations_synthetic_population_{purpose}_{submodel}_withmode.csv', index=False)
    else:
        joined.to_csv(f'saved_data/activities_locations_synthetic_population_{purpose}_{submodel}_withmode_NOS.csv', index=False)

    logging.info('')
    
    


   
    
  
    
    
    
    
    
    
    #logsums = compute_logsums(mm, tree_pred)
    
    
    '''
    logsums = lx.DataArray.zeros(
        pred_tours.set_index("ptour_id").caseids(),
        skims.root.lookup.ids,
        name="logsums",
    )


    for dtaz in logsums.ids:
        mm.datatree = dt_work.replace_datasets(
            tours=dt_work.root_dataset.assign(DTAZ=xr.full_like(dt_work._getitem("DTAZ"), dtaz)),
        )
        logsums.loc[dict(ids=dtaz)] = m.logsums()
    '''
    
   
    #new_data = pd.read_csv(r'Z:\Household_Match\hhid_fid_fixed.csv')  # Replace with your dataset


def helper_plot():
    scenarios = {
        'baseline': {
            'N': 'saved_data/activities_locations_synthetic_population_N_None_withmode.csv',
            'work': 'saved_data/activities_locations_synthetic_population_M_work_withmode.csv',
            'edu': 'saved_data/activities_locations_synthetic_population_M_edu_withmode.csv'
        },
        'safety': {
            'N': 'saved_data/activities_locations_synthetic_population_N_None_withmode_NOS.csv',
            'work': 'saved_data/activities_locations_synthetic_population_M_work_withmode_NOS.csv',
            'edu': 'saved_data/activities_locations_synthetic_population_M_edu_withmode_NOS.csv'
        }
    }

    results, summary = run_sa2_destination_and_mode_analysis(
        scenarios=scenarios,
        sa2_index_path='saved_data/SA2_2021_AUST_SHP_GDA2020/sa2index.csv',
        sa2_shapefile_path='saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2016_AUST.shp',
        output_dir='saved_data/outputs',
        density_quantile=0.0
    )
    print(summary)


def run_sa2_destination_and_mode_analysis(
    scenarios,
    sa2_index_path,
    sa2_shapefile_path,
    output_dir,
    density_quantile=0.0,
    top_n_bars=20,
    top_n_mode_sa2=10
):
    """
    Runs SA2 destination density + mode split analysis for multiple scenarios.

    Parameters
    ----------
    scenarios : dict
        {
            'baseline': {
                'N': 'file.csv',
                'work': 'file.csv',
                'edu': 'file.csv'
            },
            'policy_A': {...}
        }

    Returns
    -------
    pd.DataFrame
        Tidy dataframe with scenario × activity × SA2 × mode results
    """

    import os
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Read SA2 index
    # --------------------------------------------------
    sa2_index = pd.read_csv(sa2_index_path)
    dtaz_to_sa2 = (
        sa2_index[['SA2_MAIN16', 'ORDER']]
        .assign(SA2_MAIN16=lambda x: x['SA2_MAIN16'].astype(str))
        .set_index('ORDER')['SA2_MAIN16']
        .to_dict()
    )

    valid_sa2s = set(sa2_index['SA2_MAIN16'].astype(str))

    # --------------------------------------------------
    # Read SA2 geometry
    # --------------------------------------------------
    sa2_gdf = gpd.read_file(sa2_shapefile_path)
    sa2_gdf['SA2_MAIN16'] = sa2_gdf['SA2_MAIN16'].astype(str)
    sa2_gdf = sa2_gdf[sa2_gdf['SA2_MAIN16'].isin(valid_sa2s)]

    all_results = []
    summary_rows = []

    # ==================================================
    # Loop over scenarios
    # ==================================================
    for scenario_name, activity_files in scenarios.items():

        print(f'\n=== Scenario: {scenario_name} ===')

        for act_type, filepath in activity_files.items():

            print(f'Processing {act_type}')

            # ----------------------------------------------
            # Read activity file
            # ----------------------------------------------
            df = pd.read_csv(filepath)
            df['DTAZ'] = df['DTAZ'].astype(int)

            df['SA2_MAIN16'] = df['DTAZ'].map(dtaz_to_sa2)
            df = df.dropna(subset=['SA2_MAIN16'])

            df['SA2_MAIN16'] = (
                df['SA2_MAIN16']
               #.round()
                .astype('Int64')
                .astype(str)
                .str.zfill(9)
            )

            # ----------------------------------------------
            # Destination density
            # ----------------------------------------------
            sa2_totals = (
                df
                .groupby('SA2_MAIN16', as_index=False)
                .size()
                .rename(columns={'size': 'total_trips'})
            )

            # ----------------------------------------------
            # Mode split by SA2
            # ----------------------------------------------
            sa2_mode = (
                df
                .groupby(['SA2_MAIN16', 'tour_mode'], as_index=False)
                .size()
                .rename(columns={'size': 'trip_count'})
            )

            sa2_mode = sa2_mode.merge(sa2_totals, on='SA2_MAIN16')
            sa2_mode['mode_share'] = (
                sa2_mode['trip_count'] / sa2_mode['total_trips']
            )

            sa2_mode['scenario'] = scenario_name
            sa2_mode['activity'] = act_type

            all_results.append(sa2_mode)

            # ----------------------------------------------
            # Summary statistics
            # ----------------------------------------------
            summary_rows.append({
                'scenario': scenario_name,
                'activity': act_type,
                'total_trips': len(df),
                'mean_trips_per_sa2': sa2_totals['total_trips'].mean(),
                'median_trips_per_sa2': sa2_totals['total_trips'].median(),
                'top_sa2_share': (
                    sa2_totals['total_trips'].max()
                    / sa2_totals['total_trips'].sum()
                )
            })

            # ----------------------------------------------
            # Join geometry
            # ----------------------------------------------
            sa2_plot = sa2_gdf.merge(
                sa2_totals,
                on='SA2_MAIN16',
                how='left'
            ).fillna({'total_trips': 0})

            sa2_plot = sa2_plot[sa2_plot['total_trips'] > 0]

            # ----------------------------------------------
            # Density clipping
            # ----------------------------------------------
            threshold = sa2_plot['total_trips'].quantile(density_quantile)
            dense_sa2 = sa2_plot[sa2_plot['total_trips'] >= threshold]

            minx, miny, maxx, maxy = dense_sa2.total_bounds

            # ----------------------------------------------
            # Density map
            # ----------------------------------------------
            fig, ax = plt.subplots(figsize=(12, 10))

            cmap = mcolors.LinearSegmentedColormap.from_list(
                "white_hot",
                ["white", "#d9ba3f", "#8427b9", "#e31a1c", "black"]
            )

            dense_sa2.plot(
                column='total_trips',
                cmap=cmap,
                linewidth=0.2,
                edgecolor='black',
                legend=True,
                ax=ax
            )

            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_title(
                f'SA2 Destination Density – {act_type.upper()} ({scenario_name})'
            )
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(
                f'{output_dir}/sa2_density_{scenario_name}_{act_type}.png',
                dpi=300
            )
            plt.close()

            # ----------------------------------------------
            # Top SA2 bar chart
            # ----------------------------------------------
            top_sa2 = sa2_totals.nlargest(top_n_bars, 'total_trips')

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_sa2['SA2_MAIN16'], top_sa2['total_trips'])
            ax.invert_yaxis()
            ax.set_title(
                f'Top SA2 Destinations – {act_type.upper()} ({scenario_name})'
            )

            plt.tight_layout()
            plt.savefig(
                f'{output_dir}/sa2_top_{scenario_name}_{act_type}.png',
                dpi=300
            )
            plt.close()

            # ----------------------------------------------
            # Mode split stacked bars (top SA2s)
            # ----------------------------------------------
            top_sa2_ids = top_sa2.head(top_n_mode_sa2)['SA2_MAIN16']

            pivot = (
                sa2_mode[sa2_mode['SA2_MAIN16'].isin(top_sa2_ids)]
                .pivot(index='SA2_MAIN16', columns='tour_mode', values='mode_share')
                .fillna(0)
                .loc[top_sa2_ids]
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            pivot.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')

            ax.set_ylabel('Mode Share')
            ax.set_title(
                f'Mode Split by Destination – {act_type.upper()} ({scenario_name})'
            )

            plt.tight_layout()
            plt.savefig(
                f'{output_dir}/sa2_mode_split_{scenario_name}_{act_type}.png',
                dpi=300
            )
            plt.close()

            # ----------------------------------------------
            # Dominant mode map
            # ----------------------------------------------
            dominant_mode = (
                sa2_mode
                .sort_values('trip_count', ascending=False)
                .groupby('SA2_MAIN16')
                .first()
                .reset_index()
            )

            sa2_dom = sa2_gdf.merge(
                dominant_mode,
                on='SA2_MAIN16',
                how='left'
            )

            fig, ax = plt.subplots(figsize=(12, 10))
            sa2_dom.plot(
                column='tour_mode',
                categorical=True,
                legend=True,
                linewidth=0.1,
                ax=ax
            )

            ax.set_title(
                f'Dominant Mode by Destination – {act_type.upper()} ({scenario_name})'
            )
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(
                f'{output_dir}/sa2_dominant_mode_{scenario_name}_{act_type}.png',
                dpi=300
            )
            plt.close()

    # ==================================================
    # Final outputs
    # ==================================================
    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(
        f'{output_dir}/sa2_mode_by_destination_all_scenarios.csv',
        index=False
    )
    summary_df.to_csv(
        f'{output_dir}/scenario_summary.csv',
        index=False
    )

    return results_df, summary_df





def read_in_files_and_plot():
    pd1 = pd.read_csv('saved_data/activities_locations_synthetic_population_N_None_withmode.csv')
    pd2 = pd.read_csv('saved_data/activities_locations_synthetic_population_M_work_withmode.csv')
    pd3 = pd.read_csv('saved_data/activities_locations_synthetic_population_N_edu_withmode.csv')
    combined = pd.concat([pd1, pd2, pd3], ignore_index=True)
    
    #now plot the DTAZ values..
    #plot the mode choice values
    # then look up 
    ''' saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2021_AUST_GDA2020.shp and plot the tour destinations to the sa2 codes, from the DTAZ values''' #dtaz values is order
    'DTAZ LOOKUP IS IN ved_data/SA2_2021_AUST_SHP_GDA2020/sa2indxex. ORDER REPRESENTS DTAZ ID, SA2_MAIN16 is the SA2 code link this and plot the detination counts, also clip the shap file' 
    
def compute_logsums(mm, tree_pred, verbose=True):
    """
    Compute mode-choice logsums for a prediction DataTree.
    
    mm: Larch model (already loaded)
    tree_pred: DataTree with:
        - base.ids as zone alternatives
        - tours with OTAZ and a placeholder DTAZ
        - person dataset aligned to tours
        - skims and emp datasets attached
    
    Returns
    -------
    logsums : xarray.DataArray  
        shape (ptour_id, zone_id)
    """
    
    # Number of cases and number of zones
    case_ids = list(tree_pred.caseids())
    alt_ids = tree_pred.altids()
    N = len(case_ids)
    Z = len(alt_ids)

    if verbose:
        print(f"Computing logsums for {N} persons × {Z} zones...")

    # Allocate output DataArray
    logsums = xr.DataArray(
        np.zeros((N, Z), dtype=np.float32),
        dims=["ptour_id", "ids"],
        coords={"ptour_id": case_ids, "ids": alt_ids},
        name="logsum",
    )

    # Loop over each destination zone
    for i, dtaz in enumerate(alt_ids):
        if verbose and i % 25 == 0:
            print(f"  dtaz={dtaz}  ({i+1} of {Z})")

        # Replace DTAZ in tours dataset
        tours_new = tree_pred.root_dataset.assign(
            DTAZ=xr.full_like(tree_pred._getitem("DTAZ"), dtaz)
        )

        # Temporary tree with updated tours
        tree_temp = tree_pred.replace_datasets(tour=tours_new)

        # Attach and shake the model
        mm.datatree = tree_temp
        mm.mangle()
        mm.unmangle()

        # Compute logsums for this dtaz
        ls = mm.logsums()    # shape (N,)
        logsums[:, i] = ls.astype(np.float32)

    if verbose:
        print("Logsums computed.")

    return logsums

    
def read_in_files_and_plot_density_clipped(
    density_quantile=0
    ):
    """
    density_quantile:
        0.8 keeps the top 20% most dense SA2s
    """
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt 

    # --------------------------------------------------
    # 1. Read activity CSVs
    # --------------------------------------------------
    pd1 = pd.read_csv(
        'saved_data/activities_locations_synthetic_population_N_None_withmode.csv'
    )
    pd2 = pd.read_csv(
        'saved_data/activities_locations_synthetic_population_M_work_withmode.csv'
    )
    pd3 = pd.read_csv(
        'saved_data/activities_locations_synthetic_population_M_edu_withmode.csv'
    )

    combined = pd.concat([pd1, pd2, pd3], ignore_index=True)
    combined.to_csv('saved_data/persons_with_activity.csv', index=False)
    # Ensure DTAZ is integer
    combined['DTAZ'] = combined['DTAZ'].astype(int)

    # --------------------------------------------------
    # 2. Read DTAZ → SA2 lookup (ORDER = DTAZ ID)
    # --------------------------------------------------
    sa2_index = pd.read_csv(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/sa2index.csv'
    )

    # Mapping via index position
    dtaz_to_sa2 = sa2_index[['SA2_MAIN16', 'ORDER']].set_index('ORDER')['SA2_MAIN16'].to_dict()

    combined['SA2_MAIN16'] = combined['DTAZ'].map(dtaz_to_sa2)

    # Drop invalid mappings
    combined = combined.dropna(subset=['SA2_MAIN16'])

    combined['SA2_MAIN16'] = (
        combined['SA2_MAIN16']
        .round()
        .astype('Int64')
        .astype(str)
        .str.zfill(9)
    )

    # --------------------------------------------------
    # 3. Aggregate destination density by SA2
    # --------------------------------------------------
    sa2_counts = (
        combined
        .groupby('SA2_MAIN16', as_index=False)
        .size()
        .rename(columns={'size': 'dest_count'})
    )
    
    # --------------------------------------------------
    # 4. Load SA2 shapefile
    # --------------------------------------------------
    sa2_gdf = gpd.read_file(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2016_AUST.shp'
    )

    # Ensure join keys match type
    sa2_gdf['SA2_MAIN16'] = sa2_gdf['SA2_MAIN16'].astype(str)
    
    valid_sa2s = set(sa2_gdf['SA2_MAIN16'])

    sa2_gdf = sa2_gdf[sa2_gdf['SA2_MAIN16'].isin(valid_sa2s)]
    
    sa2_counts['SA2_MAIN16'] = sa2_counts['SA2_MAIN16'].astype(str)

    # --------------------------------------------------
    # 5. Join density to geometry
    # --------------------------------------------------
    sa2_plot = sa2_gdf.merge(
        sa2_counts,
        on='SA2_MAIN16',
        how='left'
    )

    sa2_plot['dest_count'] = sa2_plot['dest_count'].fillna(0)

    # --------------------------------------------------
    # 5.5 REMOVE ZERO-COUNT SA2s COMPLETELY
    # --------------------------------------------------
    sa2_plot = sa2_plot[sa2_plot['dest_count'] >=0 ]

    # --------------------------------------------------
    # 6. Density-based clipping
    # --------------------------------------------------
    threshold = sa2_plot['dest_count'].quantile(density_quantile)

    dense_sa2 = sa2_plot[
        sa2_plot['dest_count'] >= threshold
    ]

    # --------------------------------------------------
    # 6. Density-based clipping (CORE STEP)
    # --------------------------------------------------
    threshold = sa2_plot['dest_count'].quantile(density_quantile)

    dense_sa2 = sa2_plot[
        sa2_plot['dest_count'] >= threshold
    ]

    # --------------------------------------------------
    # 7. Auto-zoom to dense region
    # --------------------------------------------------
    minx, miny, maxx, maxy = dense_sa2.total_bounds

    # --------------------------------------------------
    # 8. Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    dense_sa2.plot(
        column='dest_count',
        cmap='viridis',
        linewidth=0.2,
        edgecolor='black',
        legend=True,
        ax=ax
    )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_title(
        f'SA2 Destination Density (Top {(1-density_quantile)*100:.0f}%)',
        fontsize=14
    )
    ax.axis('off')

    plt.show()   
    plt.savefig('saved_data/sa2_destination_density_clipped.png', dpi=300)




def read_in_truns_and_plot_density_clipped( top_n_bars=20):
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # 1. File definitions by activity type
    # --------------------------------------------------
    df_all = pd.read_csv('saved_data/activities_locations_trips.csv')
    # group by DESTPURP

    
    # --------------------------------------------------
    # 2. Read SA2 index (DTAZ → SA2)
    # --------------------------------------------------
    sa2_index = pd.read_csv(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/sa2index.csv'
    )

    dtaz_to_sa2 = sa2_index[['SA2_MAIN16', 'ORDER']].set_index('ORDER')['SA2_MAIN16'].to_dict()
    
    valid_sa2s=set(sa2_index['SA2_MAIN16'].astype(str))
    
    # --------------------------------------------------
    # 3. Read SA2 shapefile
    # --------------------------------------------------
    sa2_gdf = gpd.read_file(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2016_AUST.shp'
    )
    sa2_gdf['SA2_MAIN16'] = sa2_gdf['SA2_MAIN16'].astype(str)
    sa2_gdf = sa2_gdf[sa2_gdf['SA2_MAIN16'].isin(valid_sa2s)]
    # --------------------------------------------------
    # 4. Loop over activity types
    # --------------------------------------------------
    for act_type, df in df_all.groupby("DESTPURP"):
        print(f"Processing activity type: {act_type}")

        print(f'Processing activity type: {act_type}')

        # ----------------------------------------------
        # 4.1 Read activity file
        # ----------------------------------------------
        #df = pd.read_csv(filepath)
        df['DTAZ'] = df['DTAZ'].astype(int)

        # Map DTAZ → SA2
        df['SA2_MAIN16'] = df['DTAZ'].map(dtaz_to_sa2)

        df = df.dropna(subset=['SA2_MAIN16'])

        df['SA2_MAIN16'] = (
            df['SA2_MAIN16']
            .round()
            .astype('Int64')
            .astype(str)
            .str.zfill(9)
        )

        # ----------------------------------------------
        # 4.2 Aggregate destination density
        # ----------------------------------------------
        sa2_counts = (
            df
            .groupby('SA2_MAIN16', as_index=False)
            .size()
            .rename(columns={'size': 'dest_count'})
        )

        # ----------------------------------------------
        # 4.3 Join with geometry
        # ----------------------------------------------
        sa2_plot = sa2_gdf.merge(
            sa2_counts,
            on='SA2_MAIN16',
            how='left'
        )

        sa2_plot['dest_count'] = sa2_plot['dest_count'].fillna(0)

        # Remove zero-count SA2s
        sa2_plot = sa2_plot[sa2_plot['dest_count'] > 0]

        # ----------------------------------------------
        # 4.4 Density-based clipping
        # ----------------------------------------------
        threshold = sa2_plot['dest_count'].quantile(0)

        dense_sa2 = sa2_plot[
            sa2_plot['dest_count'] >= threshold
        ]

        minx, miny, maxx, maxy = dense_sa2.total_bounds

        # ----------------------------------------------
        # 4.5 Plot SA2 density map
        # ----------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        import matplotlib.colors as mcolors

        bounds = [0, 1, 10, 50, 100, dense_sa2['dest_count'].max()]

        # colors: 0 = white, then increasingly hot reds
        colors = [
            "white",     # 0
            "yellow",    # 1–4
            'purple',
            "darkred",   # 5–9
            "black"      # 10+
        ]

        cmap = mcolors.ListedColormap(colors)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "white_then_hot",
            [
                (0.0, "white"),     # exactly 0
                (0.02, "#d9ba3f"),  # very light yellow (value ~1)
                (0.15, "#8427b9"),  # orange
                (0.5, "#e31a1c"),   # red
                (1.0, "black")     # hottest
            ]
        )
        vmax = dense_sa2['dest_count'].max()
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        #norm = mcolors.BoundaryNorm(bounds, cmap.N)
        dense_sa2.plot(
            column='dest_count',
            cmap=cmap,
            norm = norm,
            linewidth=0.2,
            edgecolor='black',
            legend=True,
            ax=ax
        )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ax.set_title(
            f'SA2 Destination Density – {act_type.upper()} '
           ,
            fontsize=14
        )
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(
            f'saved_data/sa2_density_{act_type}.png',
            dpi=300
        )
        plt.close()

        # ----------------------------------------------
        # 4.6 Bar chart (top SA2s)
        # ----------------------------------------------
        top_sa2 = (
            sa2_counts
            .sort_values('dest_count', ascending=False)
            .head(top_n_bars)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            top_sa2['SA2_MAIN16'],
            top_sa2['dest_count']
        )

        ax.invert_yaxis()
        ax.set_xlabel('Destination Count')
        ax.set_ylabel('SA2')
        ax.set_title(
            f'Top {top_n_bars} SA2s by Destination Count – {act_type.upper()}'
        )

        plt.tight_layout()
        plt.savefig(
            f'saved_data/sa2_density_bar_{act_type}.png',
            dpi=300
        )
        plt.close()

    

def read_in_files_and_plot_density_clippeda(
    density_quantile=0.0,
    top_n_bars=20, suffix = ''
):
    """
    density_quantile:
        0.8 keeps the top 20% most dense SA2s
    """

    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # 1. File definitions by activity type
    # --------------------------------------------------
    activity_files = {
        'N': f'saved_data/activities_locations_synthetic_population_N_None_withmode{suffix}.csv',
        'work': f'saved_data/activities_locations_synthetic_population_M_work_withmode{suffix}.csv',
        'edu': f'saved_data/activities_locations_synthetic_population_M_edu_withmode{suffix}.csv'
    }
    pd1 = pd.read_csv(
        f'saved_data/activities_locations_synthetic_population_N_None_withmode{suffix}.csv'
    )
    pd2 = pd.read_csv(
        f'saved_data/activities_locations_synthetic_population_M_work_withmode{suffix}.csv'
    )
    pd3 = pd.read_csv(
        f'saved_data/activities_locations_synthetic_population_M_edu_withmode{suffix}.csv'
    )

    combined = pd.concat([pd1, pd2, pd3], ignore_index=True)
    combined.to_csv(f'saved_data/persons_with_activity{suffix}.csv', index=False)
    # --------------------------------------------------
    # 2. Read SA2 index (DTAZ → SA2)
    # --------------------------------------------------
    sa2_index = pd.read_csv(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/sa2index.csv'
    )

    dtaz_to_sa2 = sa2_index[['SA2_MAIN16', 'ORDER']].set_index('ORDER')['SA2_MAIN16'].to_dict()
    
    valid_sa2s=set(sa2_index['SA2_MAIN16'].astype(str))
    
    # --------------------------------------------------
    # 3. Read SA2 shapefile
    # --------------------------------------------------
    sa2_gdf = gpd.read_file(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2016_AUST.shp'
    )
    sa2_gdf['SA2_MAIN16'] = sa2_gdf['SA2_MAIN16'].astype(str)
    sa2_gdf = sa2_gdf[sa2_gdf['SA2_MAIN16'].isin(valid_sa2s)]
    # --------------------------------------------------
    # 4. Loop over activity types
    # --------------------------------------------------
    for act_type, filepath in activity_files.items():

        print(f'Processing activity type: {act_type}')

        # ----------------------------------------------
        # 4.1 Read activity file
        # ----------------------------------------------
        df = pd.read_csv(filepath)
        df['DTAZ'] = df['DTAZ'].astype(int)

        # Map DTAZ → SA2
        df['SA2_MAIN16'] = df['DTAZ'].map(dtaz_to_sa2)

        df = df.dropna(subset=['SA2_MAIN16'])

        df['SA2_MAIN16'] = (
            df['SA2_MAIN16']
            .round()
            .astype('Int64')
            .astype(str)
            .str.zfill(9)
        )

        # ----------------------------------------------
        # 4.2 Aggregate destination density
        # ----------------------------------------------
        sa2_counts = (
            df
            .groupby('SA2_MAIN16', as_index=False)
            .size()
            .rename(columns={'size': 'dest_count'})
        )

        # ----------------------------------------------
        # 4.3 Join with geometry
        # ----------------------------------------------
        sa2_plot = sa2_gdf.merge(
            sa2_counts,
            on='SA2_MAIN16',
            how='left'
        )

        sa2_plot['dest_count'] = sa2_plot['dest_count'].fillna(0)

        # Remove zero-count SA2s
        sa2_plot = sa2_plot[sa2_plot['dest_count'] > 0]

        # ----------------------------------------------
        # 4.4 Density-based clipping
        # ----------------------------------------------
        threshold = sa2_plot['dest_count'].quantile(density_quantile)

        dense_sa2 = sa2_plot[
            sa2_plot['dest_count'] >= threshold
        ]

        minx, miny, maxx, maxy = dense_sa2.total_bounds

        # ----------------------------------------------
        # 4.5 Plot SA2 density map
        # ----------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        import matplotlib.colors as mcolors

        bounds = [0, 1, 10, 50, 100, dense_sa2['dest_count'].max()]

        # colors: 0 = white, then increasingly hot reds
        colors = [
            "white",     # 0
            "yellow",    # 1–4
            'purple',
            "darkred",   # 5–9
            "black"      # 10+
        ]

        cmap = mcolors.ListedColormap(colors)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "white_then_hot",
            [
                (0.0, "white"),     # exactly 0
                (0.02, "#d9ba3f"),  # very light yellow (value ~1)
                (0.15, "#8427b9"),  # orange
                (0.5, "#e31a1c"),   # red
                (1.0, "black")     # hottest
            ]
        )
        vmax = dense_sa2['dest_count'].max()
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        #norm = mcolors.BoundaryNorm(bounds, cmap.N)
        dense_sa2.plot(
            column='dest_count',
            cmap=cmap,
            norm = norm,
            linewidth=0.2,
            edgecolor='black',
            legend=True,
            ax=ax
        )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ax.set_title(
            f'SA2 Destination Density – {act_type.upper()} '
           ,
            fontsize=14
        )
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(
            f'saved_data/sa2_density_{act_type}{suffix}.png',
            dpi=300
        )
        plt.close()

        # ----------------------------------------------
        # 4.6 Bar chart (top SA2s)
        # ----------------------------------------------
        top_sa2 = (
            sa2_counts
            .sort_values('dest_count', ascending=False)
            .head(top_n_bars)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            top_sa2['SA2_MAIN16'],
            top_sa2['dest_count']
        )

        ax.invert_yaxis()
        ax.set_xlabel('Destination Count')
        ax.set_ylabel('SA2')
        ax.set_title(
            f'Top {top_n_bars} SA2s by Destination Count – {act_type.upper()}'
        )

        plt.tight_layout()
        plt.savefig(
            f'saved_data/sa2_density_bar_{act_type}{suffix}.png',
            dpi=300
        )
        plt.close()

    print('✅ All plots generated and saved.')



def read_in_files_plot_map_and_bars(
    density_quantile=0.8,
    top_n_bars=20
    ):
    # --------------------------------------------------
    # 1. Read activity CSVs
    # --------------------------------------------------
    pd1 = pd.read_csv(
        'saved_data/activities_locations_synthetic_population_N_None_withmode.csv'
    )
    pd2 = pd.read_csv(
        'saved_data/activities_locations_synthetic_population_M_work_withmode.csv'
    )
    pd3 = pd.read_csv(
        'saved_data/activities_locations_synthetic_population_N_edu_withmode.csv'
    )

    combined = pd.concat([pd1, pd2, pd3], ignore_index=True)
    combined['DTAZ'] = combined['DTAZ'].astype(int)

    # --------------------------------------------------
    # 2. DTAZ → SA2 lookup (index-based)
    # --------------------------------------------------
    sa2_index = pd.read_csv(
        'savved_data/SA2_2021_AUST_SHP_GDA2020/sa2index.csv'
    )
    sa2_index['ORDER'] = sa2_index['ORDER'].astype(int)

    combined = combined.merge(
    sa2_index[['ORDER', 'SA2_MAIN16']],
    left_on='DTAZ',
    right_on='ORDER',
    how='left'
    ).drop(columns=['ORDER'])

    combined = combined.dropna(subset=['SA2_MAIN16'])

    # --------------------------------------------------
    # 3. Aggregate density
    # --------------------------------------------------
    sa2_counts = (
        combined
        .groupby('SA2_MAIN16')
        .size()
        .reset_index(name='dest_count')
    )

    # --------------------------------------------------
    # 4. Load SA2 geometry
    # --------------------------------------------------
    sa2_gdf = gpd.read_file(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2021_AUST_GDA2020.shp'
    )

    sa2_gdf['SA2_MAIN16'] = sa2_gdf['SA2_MAIN16'].astype(str)
    sa2_counts['SA2_MAIN16'] = sa2_counts['SA2_MAIN16'].astype(str)

    sa2_plot = sa2_gdf.merge(
        sa2_counts,
        on='SA2_MAIN16',
        how='left'
    )

    sa2_plot['dest_count'] = sa2_plot['dest_count'].fillna(0)

    # --------------------------------------------------
    # 5. Density-based clipping
    # --------------------------------------------------
    threshold = sa2_plot['dest_count'].quantile(density_quantile)

    dense_sa2 = sa2_plot[
        sa2_plot['dest_count'] >= threshold
    ]

    # --------------------------------------------------
    # 6. Prepare bar-plot data
    # --------------------------------------------------
    bar_df = (
        dense_sa2[['SA2_MAIN16', 'dest_count']]
        .sort_values('dest_count', ascending=False)
        .head(top_n_bars)
    )

    # --------------------------------------------------
    # 7. Plot map
    # --------------------------------------------------
    fig, (ax_map, ax_bar) = plt.subplots(
        2, 1,
        figsize=(14, 14),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    dense_sa2.plot(
        column='dest_count',
        cmap='Reds',
        linewidth=0.2,
        edgecolor='black',
        legend=True,
        ax=ax_map
    )

    minx, miny, maxx, maxy = dense_sa2.total_bounds
    ax_map.set_xlim(minx, maxx)
    ax_map.set_ylim(miny, maxy)

    ax_map.set_title(
        f'SA2 Destination Density (Top {(1-density_quantile)*100:.0f}%)',
        fontsize=14
    )
    ax_map.axis('off')

    # --------------------------------------------------
    # 8. Plot bars
    # --------------------------------------------------
    ax_bar.bar(
        bar_df['SA2_MAIN16'],
        bar_df['dest_count'],
        color='steelblue'
    )

    ax_bar.set_title(
        f'Top {top_n_bars} SA2s by Destination Count'
    )
    ax_bar.set_xlabel('SA2 Code')
    ax_bar.set_ylabel('Destination Count')

    ax_bar.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


def read_in_files_and_plot_density_clippeda(
    density_quantile=0.0,
    top_n_bars=20, suffix = ''
):
    """
    density_quantile:
        0.8 keeps the top 20% most dense SA2s
    """

    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt

    # --------------------------------------------------
    # 1. File definitions by activity type
    # --------------------------------------------------
    activity_files = {
        'N': f'saved_data/activities_locations_synthetic_population_N_None_withmode{suffix}.csv',
        'work': f'saved_data/activities_locations_synthetic_population_M_work_withmode{suffix}.csv',
        'edu': f'saved_data/activities_locations_synthetic_population_M_edu_withmode{suffix}.csv'
    }
    pd1 = pd.read_csv(
        f'saved_data/activities_locations_synthetic_population_N_None_withmode{suffix}.csv'
    )
    pd2 = pd.read_csv(
        f'saved_data/activities_locations_synthetic_population_M_work_withmode{suffix}.csv'
    )
    pd3 = pd.read_csv(
        f'saved_data/activities_locations_synthetic_population_M_edu_withmode{suffix}.csv'
    )

    combined = pd.concat([pd1, pd2, pd3], ignore_index=True)
    combined.to_csv(f'saved_data/persons_with_activity{suffix}.csv', index=False)
    # --------------------------------------------------
    # 2. Read SA2 index (DTAZ → SA2)
    # --------------------------------------------------
    sa2_index = pd.read_csv(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/sa2index.csv'
    )

    dtaz_to_sa2 = sa2_index[['SA2_MAIN16', 'ORDER']].set_index('ORDER')['SA2_MAIN16'].to_dict()
    
    valid_sa2s=set(sa2_index['SA2_MAIN16'].astype(str))
    
    # --------------------------------------------------
    # 3. Read SA2 shapefile
    # --------------------------------------------------
    sa2_gdf = gpd.read_file(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2016_AUST.shp'
    )
    sa2_gdf['SA2_MAIN16'] = sa2_gdf['SA2_MAIN16'].astype(str)
    sa2_gdf = sa2_gdf[sa2_gdf['SA2_MAIN16'].isin(valid_sa2s)]
    # --------------------------------------------------
    # 4. Loop over activity types
    # --------------------------------------------------
    for act_type, filepath in activity_files.items():

        print(f'Processing activity type: {act_type}')

        # ----------------------------------------------
        # 4.1 Read activity file
        # ----------------------------------------------
        df = pd.read_csv(filepath)
        df['DTAZ'] = df['DTAZ'].astype(int)

        # Map DTAZ → SA2
        df['SA2_MAIN16'] = df['DTAZ'].map(dtaz_to_sa2)

        df = df.dropna(subset=['SA2_MAIN16'])

        df['SA2_MAIN16'] = (
            df['SA2_MAIN16']
            .round()
            .astype('Int64')
            .astype(str)
            .str.zfill(9)
        )

        # ----------------------------------------------
        # 4.2 Aggregate destination density
        # ----------------------------------------------
        sa2_counts = (
            df
            .groupby('SA2_MAIN16', as_index=False)
            .size()
            .rename(columns={'size': 'dest_count'})
        )

        # ----------------------------------------------
        # 4.3 Join with geometry
        # ----------------------------------------------
        sa2_plot = sa2_gdf.merge(
            sa2_counts,
            on='SA2_MAIN16',
            how='left'
        )

        sa2_plot['dest_count'] = sa2_plot['dest_count'].fillna(0)

        # Remove zero-count SA2s
        sa2_plot = sa2_plot[sa2_plot['dest_count'] > 0]

        # ----------------------------------------------
        # 4.4 Density-based clipping
        # ----------------------------------------------
        threshold = sa2_plot['dest_count'].quantile(density_quantile)

        dense_sa2 = sa2_plot[
            sa2_plot['dest_count'] >= threshold
        ]

        minx, miny, maxx, maxy = dense_sa2.total_bounds

        # ----------------------------------------------
        # 4.5 Plot SA2 density map
        # ----------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        import matplotlib.colors as mcolors

        bounds = [0, 1, 10, 50, 100, dense_sa2['dest_count'].max()]

        # colors: 0 = white, then increasingly hot reds
        colors = [
            "white",     # 0
            "yellow",    # 1–4
            'purple',
            "darkred",   # 5–9
            "black"      # 10+
        ]

        cmap = mcolors.ListedColormap(colors)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "white_then_hot",
            [
                (0.0, "white"),     # exactly 0
                (0.02, "#d9ba3f"),  # very light yellow (value ~1)
                (0.15, "#8427b9"),  # orange
                (0.5, "#e31a1c"),   # red
                (1.0, "black")     # hottest
            ]
        )
        vmax = dense_sa2['dest_count'].max()
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        #norm = mcolors.BoundaryNorm(bounds, cmap.N)
        dense_sa2.plot(
            column='dest_count',
            cmap=cmap,
            norm = norm,
            linewidth=0.2,
            edgecolor='black',
            legend=True,
            ax=ax
        )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ax.set_title(
            f'SA2 Destination Density – {act_type.upper()} '
           ,
            fontsize=14
        )
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(
            f'saved_data/sa2_density_{act_type}{suffix}.png',
            dpi=300
        )
        plt.close()

        # ----------------------------------------------
        # 4.6 Bar chart (top SA2s)
        # ----------------------------------------------
        top_sa2 = (
            sa2_counts
            .sort_values('dest_count', ascending=False)
            .head(top_n_bars)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            top_sa2['SA2_MAIN16'],
            top_sa2['dest_count']
        )

        ax.invert_yaxis()
        ax.set_xlabel('Destination Count')
        ax.set_ylabel('SA2')
        ax.set_title(
            f'Top {top_n_bars} SA2s by Destination Count – {act_type.upper()}'
        )

        plt.tight_layout()
        plt.savefig(
            f'saved_data/sa2_density_bar_{act_type}{suffix}.png',
            dpi=300
        )
        plt.close()

    print('✅ All plots generated and saved.')



def compare_safety_vs_nosafety(
    density_quantile=0.0,
    top_n_bars=20
):
    """
    Compare destination density between:
        suffix=''      (WITH safety)
        suffix='_NOS'  (WITHOUT safety)
    """

    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    suffix_safe = ''
    suffix_nos = '_NOS'

    activity_types = {
        'N': 'activities_locations_synthetic_population_N_None_withmode',
        'work': 'activities_locations_synthetic_population_M_work_withmode',
        'edu': 'activities_locations_synthetic_population_M_edu_withmode'
    }

    # --------------------------------------------------
    # SA2 mapping + geometry
    # --------------------------------------------------
    sa2_index = pd.read_csv(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/sa2index.csv'
    )

    dtaz_to_sa2 = (
        sa2_index[['SA2_MAIN16', 'ORDER']]
        .set_index('ORDER')['SA2_MAIN16']
        .to_dict()
    )

    sa2_gdf = gpd.read_file(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2016_AUST.shp'
    )
    sa2_gdf['SA2_MAIN16'] = sa2_gdf['SA2_MAIN16'].astype(str)

    # --------------------------------------------------
    # Loop activities
    # --------------------------------------------------
    for act_type, base_name in activity_types.items():

        print(f'Comparing: {act_type}')

        df_safe = pd.read_csv(f'saved_data/{base_name}{suffix_safe}.csv')
        df_nos = pd.read_csv(f'saved_data/{base_name}{suffix_nos}.csv')

        for df in [df_safe, df_nos]:
            df['SA2_MAIN16'] = df['DTAZ'].map(dtaz_to_sa2)
            df['SA2_MAIN16'] = (
                df['SA2_MAIN16']
                .astype('Int64')
                .astype(str)
                .str.zfill(9)
            )

        # Aggregate
        safe_counts = (
            df_safe.groupby('SA2_MAIN16')
            .size()
            .reset_index(name='safe_count')
        )

        nos_counts = (
            df_nos.groupby('SA2_MAIN16')
            .size()
            .reset_index(name='nos_count')
        )

        merged = safe_counts.merge(
            nos_counts,
            on='SA2_MAIN16',
            how='outer'
        ).fillna(0)
        

        merged['diff'] = merged['safe_count'] - merged['nos_count']
        merged['pct_change'] = np.where(
                (merged['safe_count'] + merged['nos_count']) > 0,
                merged['diff'] / ((merged['safe_count'] + merged['nos_count']) / 2) * 100,
                np.nan
            )
        
        
        total_safe = merged['safe_count'].sum()
        total_nos = merged['nos_count'].sum()

        merged['safe_share'] = merged['safe_count'] / total_safe
        merged['nos_share'] = merged['nos_count'] / total_nos

        merged['share_diff'] = merged['safe_share'] - merged['nos_share']

        sa2_plot = sa2_gdf.merge(
            merged,
            on='SA2_MAIN16',
            how='inner'
        )

        minx, miny, maxx, maxy = sa2_plot.total_bounds

        # --------------------------------------------------
        # 1️⃣ Absolute Difference Map
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        sa2_plot.plot(
            column='diff',
            cmap='RdYlGn',
            linewidth=0.2,
            edgecolor='black',
            legend=True,
            ax=ax
        )

        # ---- ADD LABELS ----
        import matplotlib.patheffects as pe

        sa2_plot['label_point'] = sa2_plot.geometry.representative_point()

        for _, row in sa2_plot.iterrows():
            if not np.isnan(row['diff']) and row['diff'] != 0:
                ax.text(
                    row['label_point'].x,
                    row['label_point'].y,
                    f"{int(row['diff'])}",   # integer formatting
                    ha='center',
                    va='center',
                    fontsize=6,
                    color='black',
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")]
                )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(
            f'Destination Count Difference (Safe − NoSafe) – {act_type}'
        )
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(
            f'saved_data/density_difference_{act_type}.png',
            dpi=300
        )
        plt.close()

        # --------------------------------------------------
        # 2️⃣ Percentage Change Map (KEY EXPLANATION)
        # --------------------------------------------------
        # --------------------------------------------------
        # --------------------------------------------------
        # 2️⃣ Percentage Change Map
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        sa2_plot.plot(
            column='pct_change',
            cmap='RdYlGn',
            linewidth=0.2,
            edgecolor='black',
            legend=True,
            ax=ax
        )

        # ---- ADD LABELS ----
        import matplotlib.patheffects as pe

        sa2_plot['label_point'] = sa2_plot.geometry.representative_point()
        threshold = 5
        for _, row in sa2_plot.iterrows():
            if not np.isnan(row['pct_change']) and abs(row['pct_change']) > threshold:
           
                ax.text(
                    row['label_point'].x,
                    row['label_point'].y,
                    f"{row['pct_change']:.1f}%",   # show as percent
                    ha='center',
                    va='center',
                    fontsize=6,
                    color='black',
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")]
                )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(f'% Change Due to Safety Skims – {act_type}')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(
            f'saved_data/density_pct_change_{act_type}.png',
            dpi=300
        )
        plt.close()
        
        
         # --------------------------------------------------
        # 23 Percentage Change Map (KEY EXPLANATION)
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))

        sa2_plot.plot(
            column='share_diff',
            cmap='RdYlGn',
            linewidth=0.2,
            edgecolor='black',
            legend=True,
            ax=ax
        )

        # ---- ADD LABELS ----
        import matplotlib.patheffects as pe
        sa2_plot['label_point'] = sa2_plot.geometry.representative_point()

        for _, row in sa2_plot.iterrows():
            if not np.isnan(row['share_diff']):
                ax.text(
                    row['label_point'].x,
                    row['label_point'].y,
                    f"{row['share_diff']:.2f}",
                    ha='center',
                    va='center',
                    fontsize=6,
                    color='black',
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")]
                )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title(f'% Change Due to Safety Skims – {act_type}')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'saved_data/sharechange_{act_type}.png', dpi=300)
        plt.close()

    print("✅ Comparison plots generated.")
    
def full_safety_comparison_framework(
    origin_zone=39,
    density_quantile=0.0
):
    """
    Full analytical comparison:
    - Destination density maps
    - Difference maps
    - % change maps
    - Top gain/loss SA2s
    - Lorenz curves
    - Spatial entropy
    - Safety skim surface map
    """

    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    suffix_safe = ''
    suffix_nos = '_NOS'
    all_active_sa2s = set()
    #active_sa2s = list(all_active_sa2s)

    activity_types = {
        'N': 'activities_locations_synthetic_population_N_None_withmode',
        'work': 'activities_locations_synthetic_population_M_work_withmode',
        'edu': 'activities_locations_synthetic_population_M_edu_withmode'
    }

    # --------------------------------------------------
    # SA2 Mapping + Geometry
    # --------------------------------------------------
    sa2_index = pd.read_csv(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/sa2index.csv'
    )

    dtaz_to_sa2 = (
        sa2_index[['SA2_MAIN16', 'ORDER']]
        .set_index('ORDER')['SA2_MAIN16']
        .to_dict()
    )

    sa2_gdf = gpd.read_file(
        'saved_data/SA2_2021_AUST_SHP_GDA2020/SA2_2016_AUST.shp'
    )
    sa2_gdf['SA2_MAIN16'] = sa2_gdf['SA2_MAIN16'].astype(str)

  
    

   
    
    
    
    
    
    
    
    
    
    
    
    for act_type, base_name in activity_types.items():

        print(f'Processing activity: {act_type}')

        df_safe = pd.read_csv(f'saved_data/{base_name}{suffix_safe}.csv')
        df_nos = pd.read_csv(f'saved_data/{base_name}{suffix_nos}.csv')

        for df in [df_safe, df_nos]:
            df['SA2_MAIN16'] = df['DTAZ'].map(dtaz_to_sa2)
            df['SA2_MAIN16'] = (
                df['SA2_MAIN16']
                .astype('Int64')
                .astype(str)
                .str.zfill(9)
            )

        safe_counts = df_safe.groupby('SA2_MAIN16').size().reset_index(name='safe')
        nos_counts = df_nos.groupby('SA2_MAIN16').size().reset_index(name='nos')
        all_active_sa2s.update(safe_counts['SA2_MAIN16'].unique())
        all_active_sa2s.update(nos_counts['SA2_MAIN16'].unique())
        merged = safe_counts.merge(nos_counts, on='SA2_MAIN16', how='outer').fillna(0)

        merged['diff'] = merged['safe'] - merged['nos']
        merged['pct_change'] = np.where(
            merged['nos'] != 0,
            merged['diff'] / merged['nos'] * 100,
            0
        )

        sa2_plot = sa2_gdf.merge(merged, on='SA2_MAIN16', how='inner')

        minx, miny, maxx, maxy = sa2_plot.total_bounds
        

        # ---------------------------------------------
        # RAW SAFE MAP
        # ---------------------------------------------
        fig, ax = plt.subplots(figsize=(10,8))
        sa2_plot.plot(column='safe', cmap='viridis',
                      edgecolor='black', linewidth=0.2,
                      legend=True, ax=ax)
        ax.set_title(f' Destinations Incoporating ECF – {act_type}')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'saved_data/map_safe_{act_type}.png', dpi=300)
        plt.close()

        # ---------------------------------------------
        # RAW NOSAFE MAP
        # ---------------------------------------------
        fig, ax = plt.subplots(figsize=(10,8))
        sa2_plot.plot(column='nos', cmap='viridis',
                      edgecolor='black', linewidth=0.2,
                      legend=True, ax=ax)
        ax.set_title(f'Destinations – {act_type}')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'saved_data/map_nosafe_{act_type}.png', dpi=300)
        plt.close()

        # ---------------------------------------------
        # DIFFERENCE MAP
        # ---------------------------------------------
        vmax = abs(sa2_plot['diff']).max()
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(10,8))
        sa2_plot.plot(column='diff', cmap='RdYlGn',
                      norm=norm, edgecolor='black',
                      linewidth=0.2, legend=True, ax=ax)
        ax.set_title(f'Difference (Safe − NoSafe) – {act_type}')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'saved_data/map_diff_{act_type}.png', dpi=300)
        plt.close()

        # ---------------------------------------------
        # % CHANGE MAP
        # ---------------------------------------------
        vmax = np.nanpercentile(abs(sa2_plot['pct_change']), 95)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(10,8))
        sa2_plot.plot(column='pct_change', cmap='RdYlGn',
                      norm=norm, edgecolor='black',
                      linewidth=0.2, legend=True, ax=ax)
        ax.set_title(f'% Change Due to Safety – {act_type}')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'saved_data/map_pct_change_{act_type}.png', dpi=300)
        plt.close()

        # ---------------------------------------------
        # TOP GAINS / LOSSES
        # ---------------------------------------------
        
        
        
        top_gain = merged.sort_values('diff', ascending=False).head(10)
        top_loss = merged.sort_values('diff').head(10)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(top_gain['SA2_MAIN16'], top_gain['diff'])
        ax.set_title(f'Top 10 Gains – {act_type}')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'saved_data/top_gains_{act_type}.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(top_loss['SA2_MAIN16'], top_loss['diff'])
        ax.set_title(f'Top 10 Losses – {act_type}')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'saved_data/top_losses_{act_type}.png', dpi=300)
        plt.close()
        
        # ---------------------------------------------
        # TOP GAINS / LOSSES — SPATIAL MAP VERSION
        # ---------------------------------------------

        # Identify top 10 gains and losses
        top_gain = merged.sort_values('diff', ascending=False).head(10)
        top_loss = merged.sort_values('diff').head(10)

        top_combined = pd.concat([top_gain, top_loss])

        # Merge with geometry
        top_map = sa2_gdf.merge(top_combined, on='SA2_MAIN16', how='inner')

        # Merge full Brisbane boundary (for grey background)
        brisbane_map = sa2_gdf.merge(merged, on='SA2_MAIN16', how='inner')

        fig, ax = plt.subplots(figsize=(10,10))

        # Plot full Brisbane lightly
        brisbane_map.plot(
            color='lightgrey',
            edgecolor='white',
            linewidth=0.3,
            ax=ax
        )

        # Center color scale on 0
        vmax = abs(top_map['pct_change']).max()
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        # Overlay top gain/loss zones
        top_map.plot(
            column='pct_change',
            cmap='RdYlGn',
            norm=norm,
            edgecolor='black',
            linewidth=0.8,
            legend=True,
            ax=ax
        )

        # Add labels (SA2 code + % change)
        top_map['label_point'] = top_map.geometry.representative_point()
        '''
        for _, row in top_map.iterrows():
            ax.text(
                row['label_point'].x,
                row['label_point'].y,
                f"{row['pct_change']:.1f}%",
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold',
                color='black'
            )
            
            '''
        import matplotlib.patheffects as pe

        for _, row in top_map.iterrows():
            ax.text(
                row['label_point'].x,
                row['label_point'].y,
                f"{row['pct_change']:.1f}%",
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold',
                color='black',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            )

        ax.set_title(f'Top 10 Gains & Losses – % Change Due to Safety ({act_type})')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'saved_data/top_gain_loss_mapm_{act_type}.png', dpi=300)
        plt.close()
        ###
        # -----------------------------------------------------
        # Merge crash (safety) values into merged dataframe
        # -----------------------------------------------------
        
        skims = os.path.join(os.path.dirname(__file__),'downcasted_matrices_base.omx')
        skims = omx.open_file(skims, 'r')
        crash_matrix = skims['CRASH'][:]   # full OD matrix

        # --------------------------------------------
        # Extract row for chosen origin
        # --------------------------------------------
        origin_index = origin_zone - 1   # if zones are 1-based
        crash_row = crash_matrix[origin_index, :]
        dest_df = pd.DataFrame({
        'DTAZ': np.arange(1, len(crash_row)+1),
        'safety': crash_row
        })
        # --------------------------------------------
        # Build dataframe of destinations
        # --------------------------------------------
        dest_df['SA2_MAIN16'] = dest_df['DTAZ'].map(dtaz_to_sa2)
        dest_df = dest_df.dropna(subset=['SA2_MAIN16'])

        dest_df['SA2_MAIN16'] = (
            dest_df['SA2_MAIN16']
            .astype('Int64')
            .astype(str)
            .str.zfill(9)
        )

        # --------------------------------------------
        # Aggregate crash to SA2 level
        # --------------------------------------------
        skim_vals = (
            dest_df
            .groupby('SA2_MAIN16')['safety']
            .mean()
            .reset_index()
        )
        
        
        
        # -----------------------------------------------------
        # Merge crash (safety) values into merged dataframe
        # -----------------------------------------------------
        merged = merged.merge(
            skim_vals,
            on='SA2_MAIN16',
            how='left'
        )

        # Avoid divide-by-zero
        merged['safety'] = merged['safety'].replace(0, np.nan)

        # -----------------------------------------------------
        # KEEP ONLY NON-ZERO DIFFERENCES
        # -----------------------------------------------------
        nonzero = merged[merged['diff'] != 0].copy()

        # -----------------------------------------------------
        # GAINS (ALL diff > 0)
        # -----------------------------------------------------
        gains = nonzero[nonzero['diff'] > 0].copy()
        gains['gain_score'] = gains['diff'] / gains['safety']

        gain_map = sa2_plot.merge(gains, on='SA2_MAIN16', how='inner')

        # -----------------------------------------------------
        # LOSSES (ALL diff < 0)
        # -----------------------------------------------------
        losses = nonzero[nonzero['diff'] < 0].copy()
        losses['loss_rate'] = abs(losses['diff']) / losses['safety']

        loss_map = sa2_plot.merge(losses, on='SA2_MAIN16', how='inner')
        
        fig, ax = plt.subplots(figsize=(10,10))

        # Background
        sa2_plot.plot(color='lightgrey', edgecolor='white', linewidth=0.3, ax=ax)

        gain_map.plot(
            column='gain_score',
            cmap='Greens',
            edgecolor='black',
            linewidth=0.4,
            legend=True,
            ax=ax
        )

        ax.set_title(f'All Gains (Diff / Crash) – {act_type}')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'saved_data/all_gains_crash_adjusted_{act_type}.png', dpi=300)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10,10))

        sa2_plot.plot(color='lightgrey', edgecolor='white', linewidth=0.3, ax=ax)

        loss_map.plot(
            column='loss_rate',
            cmap='Reds',
            edgecolor='black',
            linewidth=0.4,
            legend=True,
            ax=ax
        )

        ax.set_title(f'All Losses (|Diff| / Crash) – {act_type}')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'saved_data/all_losses_crash_rate_{act_type}.png', dpi=300)
        plt.close()
        
        
        
        
        

        # Avoid division by zero
        merged['safety'] = merged['safety'].replace(0, np.nan)

        # -----------------------------------------------------
        # GAINS (diff > 0)
        # -----------------------------------------------------
        gains = merged[merged['diff'] > 0].copy()

        gains['gain_score'] = gains['diff'] / gains['safety']

        top_gains = gains.sort_values('gain_score', ascending=False).head(10)

        gain_map = sa2_plot.merge(top_gains, on='SA2_MAIN16', how='inner')

        # -----------------------------------------------------
        # LOSSES (diff < 0)
        # -----------------------------------------------------
        losses = merged[merged['diff'] < 0].copy()

        losses['loss_rate'] = abs(losses['diff']) / losses['safety']

        top_losses = losses.sort_values('loss_rate', ascending=False).head(10)

        loss_map = sa2_plot.merge(top_losses, on='SA2_MAIN16', how='inner')
        
        
        fig, ax = plt.subplots(figsize=(10,10))

        # Background
        sa2_plot.plot(color='lightgrey', edgecolor='white', linewidth=0.3, ax=ax)

        gain_map.plot(
            column='gain_score',
            cmap='Greens',
            edgecolor='black',
            linewidth=0.8,
            legend=True,
            ax=ax
        )

        gain_map['label_point'] = gain_map.geometry.representative_point()

        for _, row in gain_map.iterrows():
            ax.text(
                row['label_point'].x,
                row['label_point'].y,
                f"{row['gain_score']:.2f}",
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold'
            )

        ax.set_title(f'Top 10 Gains (Adjusted by Crash) – {act_type}')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'saved_data/top_gains_crash_adjusted_{act_type}.png', dpi=300)
        plt.close()
        
        
        fig, ax = plt.subplots(figsize=(10,10))

        sa2_plot.plot(color='lightgrey', edgecolor='white', linewidth=0.3, ax=ax)

        loss_map.plot(
            column='loss_rate',
            cmap='Reds',
            edgecolor='black',
            linewidth=0.8,
            legend=True,
            ax=ax
        )

        loss_map['label_point'] = loss_map.geometry.representative_point()

        for _, row in loss_map.iterrows():
            ax.text(
                row['label_point'].x,
                row['label_point'].y,
                f"{row['loss_rate']:.2f}",
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold'
            )

        ax.set_title(f'Top 10 Losses (Rate Relative to Crash) – {act_type}')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'saved_data/top_losses_crash_rate_{act_type}.png', dpi=300)
        plt.close()
        

        # ---------------------------------------------
        # LORENZ CURVE
        # ---------------------------------------------
        def lorenz(values):
            sorted_vals = np.sort(values)
            cumvals = np.cumsum(sorted_vals)
            cumvals = cumvals / cumvals[-1]
            return np.insert(cumvals, 0, 0)

        lor_safe = lorenz(merged['safe'].values)
        lor_nos = lorenz(merged['nos'].values)
        x = np.linspace(0,1,len(lor_safe))

        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(x, lor_safe, label='Safe')
        ax.plot(x, lor_nos, label='NoSafe')
        ax.plot([0,1],[0,1],'k--')
        ax.set_title(f'Lorenz Curve – {act_type}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'saved_data/lorenz_{act_type}.png', dpi=300)
        plt.close()

        # ---------------------------------------------
        # SPATIAL ENTROPY
        # ---------------------------------------------
        def entropy(vals):
            p = vals / vals.sum()
            p = p[p > 0]
            return -np.sum(p * np.log(p))

        H_safe = entropy(merged['safe'].values)
        H_nos = entropy(merged['nos'].values)

        print(f'{act_type} entropy SAFE: {H_safe:.3f}')
        print(f'{act_type} entropy NOSAFE: {H_nos:.3f}')
        print('------------------------------------')

    # ==================================================
    # PART 2: SAFETY SKIM SURFACE (Origin 39)
    # ==================================================
    #od_safe = pd.read_csv(f'saved_data/od_skims.csv')
    # ==================================================
    # PART 2: SAFETY SKIM SURFACE (FROM OMX)
    # ==================================================

    skims = os.path.join(os.path.dirname(__file__),'downcasted_matrices_base.omx')
    skims = omx.open_file(skims, 'r')
    crash_matrix = skims['CRASH'][:]   # full OD matrix

    # --------------------------------------------
    # Extract row for chosen origin
    # --------------------------------------------
    origin_index = origin_zone   # if zones are 1-based
    crash_row = crash_matrix[origin_index, :]
    dest_df = pd.DataFrame({
    'DTAZ': np.arange(1, len(crash_row)+1),
    'safety': crash_row
    })
    # --------------------------------------------
    # Build dataframe of destinations
    # --------------------------------------------
    dest_df['SA2_MAIN16'] = dest_df['DTAZ'].map(dtaz_to_sa2)
    dest_df = dest_df.dropna(subset=['SA2_MAIN16'])

    dest_df['SA2_MAIN16'] = (
        dest_df['SA2_MAIN16']
        .astype('Int64')
        .astype(str)
        .str.zfill(9)
    )

    # --------------------------------------------
    # Aggregate crash to SA2 level
    # --------------------------------------------
    skim_vals = (
        dest_df
        .groupby('SA2_MAIN16')['safety']
        .mean()
        .reset_index()
    )

    # --------------------------------------------
    # KEEP ONLY SA2s THAT RECEIVED SAFE TRIPS
    # --------------------------------------------
   # active_sa2s = safe_counts['SA2_MAIN16'].unique()
    active_sa2s = list(all_active_sa2s)
    skim_vals = skim_vals[
        skim_vals['SA2_MAIN16'].isin(active_sa2s)
    ]

    # --------------------------------------------
    # Merge with geometry
    # --------------------------------------------
    sa2_plot = sa2_gdf.merge(
        skim_vals,
        on='SA2_MAIN16',
        how='inner'
    )

    # --------------------------------------------
    # Plot
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(10,8))

    sa2_plot.plot(
        column='safety',
        cmap='plasma',
        edgecolor='black',
        linewidth=0.2,
        legend=True,
        ax=ax
    )
    # Add numeric labels at centroids
    # --------------------------------------------
    sa2_plot['centroid'] = sa2_plot.geometry.centroid

    for idx, row in sa2_plot.iterrows():
        ax.text(
            row['centroid'].x,
            row['centroid'].y,
            f"{row['safety']:.2f}",   # format to 2 decimals
            ha='center',
            va='center',
            fontsize=6,
            color='black'
        )

    ax.set_title(f'ECF - Origin {origin_zone}')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('saved_data/safety_skim_surface.png', dpi=300)
    plt.close()

    print("✅ FULL SAFETY ANALYSIS COMPLETE.")

    
if __name__ == '__main__':
    logging.info('runing main')
   # compare_safety_vs_nosafety()
   # full_safety_comparison_framework()
    #tour_location_joint_nested()
    #compare_safety_vs_nosafety()
    #full_safety_comparison_framework()
    #read_in_files_and_plot_density_clippeda(suffix='_NOS')
    #print('done')
   # cdap_plan()
    #read_in_truns_and_plot_density_clipped()
    #read_in_files_and_plot_density_clippeda()
    #helper_plot()
    
    #print('done')
    
   
    #trip_production()
    
    #print('NM TRIPS')
    tour_location(purpose='N', submodel=None, OTAZ=39, SAFETY = 1 )
    tour_location(purpose='M', submodel='work', OTAZ=39,  SAFETY = 0)
    tour_location(OTAZ=39,  SAFETY = 0)
    tour_location(purpose='N', submodel=None, OTAZ=39, SAFETY = 0 )
   
    tour_location(OTAZ=39,  SAFETY = 1)
    tour_location(purpose='M', submodel='work', OTAZ=39,  SAFETY = 1)
    compare_safety_vs_nosafety()
    full_safety_comparison_framework()
    read_in_files_and_plot_density_clippeda(suffix='_NOS')
    
    helper_plot()
    read_in_files_and_plot_density_clippeda()
    print('hopefull this assings')
   
    print('done_all')
    #run_location_model()