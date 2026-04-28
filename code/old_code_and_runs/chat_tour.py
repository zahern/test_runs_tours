import numpy as np
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
from larch import P, X, PX
import pandas as pd
import openmatrix as omx
print(lx.__version__)
import shutil
from data_connect import Synth_Connect
from zsim_helper import monte_sample, monte_sample_key
from rich.console import Console

def filter_tours_by_purpose(tours, purpose, submodel=None):
    tours = tours.copy()

    if submodel is not None:
        return tours[tours["main_purpose"] == submodel]

    if purpose == "M":
        return tours[tours["main_purpose"].isin(["work", "edu"])]

    if purpose == "N":
        return tours[~tours["main_purpose"].isin(["work", "edu"])]

    return tours


def mode_utility_spec(m, Mode, purpose, safety):

    # ----- CAR -----
    m.utility_co[Mode.DA] = (
        + P.Time_INV * X("od.SOVTOLL_TIME__AM")
        + P("Income:D") * X("person.incp")
    )

    if safety:
        m.utility_co[Mode.DA] += (
            + P.crash_random * X("log(od.CRASH+1)")
        )

    # ----- WALK -----
    m.utility_co[Mode.Walk] = (
        + P.ASC_Walk
        + P.Time_Active * X("od.DISTWALK")
    )

    # ----- TRANSIT -----
    m.utility_co[Mode.Transit] = (
        + P.ASC_Transit
        + P.Time_INV_T * X("od.DRV_LOC_WLK_TOTIVT__AM")
    )

    return m


def destination_utility_spec(mm, purpose, safety):

    mm.utility_ca = (
        + P.beta_logsum * X("base.logsum")
        + P.beta_distance * X("log(od.DIST+1)")
    )

    if purpose == "edu":
        mm.utility_ca += (
            + P.education_size * X("log(emp.Education+1)")
        )

    if purpose == "work":
        mm.utility_ca += (
            + P.employment_size * X("log(emp.TOTEMP+1)")
        )

    if safety:
        mm.utility_ca += (
            + P.beta_crash * X("log(od.CRASH+1)")
        )

    return mm


def compute_logsums_fast(mm, tree_pred, verbose=True):
    """
    Compute mode-choice logsums for each possible destination zone.

    Parameters
    ----------
    mm : lx.Model
        Estimated mode choice model.
    tree_pred : lx.DataTree
        Prediction tree with:
            - tour dataset (with OTAZ + placeholder DTAZ)
            - person
            - od skims
    """

    case_ids = list(tree_pred.caseids())
    alt_ids  = list(tree_pred.altids())   # destination zones

    logsums = lx.DataArray.zeros(
        case_ids,
        alt_ids,
        name="logsum"
    )

    template = tree_pred._getitem("DTAZ")

    if verbose:
        print(f"Computing logsums for {len(case_ids)} cases × {len(alt_ids)} zones")

    for i, dtaz in enumerate(alt_ids):

        if verbose and i % 25 == 0:
            print(f"  Zone {i+1}/{len(alt_ids)}")

        # Replace DTAZ in tour dataset
        new_tours = tree_pred.root_dataset.assign(
            DTAZ=xr.full_like(template, dtaz)
        )

        # Build temporary tree
        temp_tree = tree_pred.replace_datasets(tour=new_tours)

        # Attach to model
        mm.datatree = temp_tree

        # Important: ensure graph is consistent
        mm.mangle()
        mm.unmangle()

        # Compute logsum (vector over cases)
        ls = mm.logsums()

        logsums.loc[dict(ids=dtaz)] = ls

    return logsums


def load_base_data():
    sc = Synth_Connect()
    return {
        "persons": sc.read_in_persons(),
        "tours": sc.read_in_tours(),
        "land": sc.get_land()
    }


def build_mode_model(dt, Mode, purpose, submodel, safety):

    m = lx.Model(tree, compute_engine="jax")
    m.choice_co_code = "tour.tour_mode"

    m = apply_mode_spec(m, Mode, purpose, submodel, safety)

    # ----------------------------
    # Nest structure
    # ----------------------------

    Car = m.graph.new_node(parameter="Mu:Car",
                           children=[Mode.DA, Mode.Other])

    NonMotor = m.graph.new_node(parameter="Mu:NonMotor",
                                children=[Mode.Walk, Mode.Bike])

    Motor = m.graph.new_node(parameter="Mu:Motor",
                             children=[Car, Mode.Transit])

    Root = m.graph.new_node(parameter="Mu:Root",
                            children=[Motor, NonMotor])

    m.set_value("Mu:Root", 1)

    # ----------------------------
    # Availability
    # ----------------------------

    m.availability_co_vars = {
        Mode.DA: 1,
        Mode.Other: 1,
        Mode.Walk: X("DISTWALK < 10"),
        Mode.Bike: X("DISTBIKE < 20"),
        Mode.Transit: 1,
    }
    
     #✅ Add random crash parameter if safety is on
    if safety:
        m.mixtures.append(
            lx.mixtures.Normal("crash_random", "std_crash")
        )
        m.n_draws = 200
        m.seed = 42
    

    return m



def estimate_model(model):
    model.set_cap(10)
    model.doctor(repair_ch_av="+")
    model.doctor(repair_nan_utility=True)
    model.maximize_loglike()
    model.calculate_parameter_covariance()
    return model


def build_destination_model(tree, purpose, submodel, safety):

    mm = lx.Model(tree, compute_engine="jax")
    mm.choice_ca_code = "tour.DTAZ"

    mm = apply_destination_spec(mm, purpose, submodel, safety)

    mm.availability_ca_var = X('od.DIST < 150')

    return mm


def build_mode_estimation_tree(pp, tour, zone_id, skims, Mode):

    # -------------------------------------------------
    # 1. Construct tour dataset (choice over modes)
    # -------------------------------------------------

    tour_ds = lx.Dataset.construct.from_idco(
        tour.set_index("ptour_id"),
        alts=Mode
    )

    # -------------------------------------------------
    # 2. Construct OD skim dataset
    # -------------------------------------------------

    od_skims = lx.Dataset.construct.from_omx(skims)

    # -------------------------------------------------
    # 3. Build DataTree with relationships
    # -------------------------------------------------

    dt = lx.DataTree(

        tour=tour_ds,
        person=pp.set_index("person_id"),
        emp=zone_id.set_index("ORDER"),
        od=od_skims,

        relationships=(
            "tour.person_id @ person.person_id",
            "emp.ORDER @ tour.DTAZ",
            "tour.OTAZ @ od.otaz",
            "tour.DTAZ @ od.dtaz",
        )
    )

    return dt


def build_destination_estimation_tree(
    tour,
    pp,
    zone_id,
    skims,
    logsums
):

    # -------------------------------------------------
    # 1. Base dataset (case × zone)
    # -------------------------------------------------

    ca = lx.Dataset.construct(
        {"logsum": logsums},
        caseid="ptour_id",
        alts=skims.root.lookup.ids,
    )

    # -------------------------------------------------
    # 2. Tour dataset (choice over zones)
    # -------------------------------------------------

    tour_ds = lx.Dataset.construct.from_idco(
        tour.reset_index(),
        alts=skims.root.lookup.ids
    )

    # -------------------------------------------------
    # 3. OD Skims
    # -------------------------------------------------

    od_skims = lx.Dataset.construct.from_omx(skims)

    # -------------------------------------------------
    # 4. Build DataTree
    # -------------------------------------------------

    tree = lx.DataTree(

        base=ca,
        tour=tour_ds,
        person=pp.set_index("person_id"),
        emp=zone_id.set_index("ORDER"),
        emp_orig=zone_id.set_index("ORDER"),
        od=od_skims,

        relationships=(
            "base.ids @ emp.ORDER",
            "base.ptour_id @ tour.ptour_id",
            "tour.person_id @ person.person_id",
            "base.ids @ od.dtaz",
            "tour.OTAZ @ od.otaz",
            #"tour.OTAZ @ emp_orig.ORDER",
        ),
    )

    return tree






# ===============================
# MODE SPECIFICATIONS
# ===============================

def mode_spec_work(m, Mode, safety):

    # =========================
    # DRIVE ALONE
    # =========================
    if safety:
        m.utility_co[Mode.DA] = (
            + P("Income:D") * X("person.incp") * X("person.incp <= 17")
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P.crash_random * X('log(od.CRASH+1)')
        )
    else:
        m.utility_co[Mode.DA] = (
            + P("Income:D") * X("person.incp") * X("person.incp <= 17")
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
        )

    # =========================
    # SHARED / OTHER
    # =========================
    if safety:
        m.utility_co[Mode.Other] = (
            + P.ASC_SR
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P("Income:OTHER") * X("person.incp") * X("person.incp <= 17")
            + P.crash_random * X('log(od.CRASH+1)')
        )
    else:
        m.utility_co[Mode.Other] = (
            + P.ASC_SR
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P.shared_per_trips * X('tour.outbound_stops')
            + P("Income:OTHER") * X("person.incp") * X("person.incp <= 17")
        )

    # =========================
    # WALK
    # =========================
    m.utility_co[Mode.Walk] = (
        + P.ASC_Walk
        + P.Time_Active * X('od.DISTWALK')
        + P.walk_per_trips * X('tour.outbound_stops')
        + P("Income:Active") * X("person.incp") * X("person.incp <= 17")
    )

    # =========================
    # BIKE
    # =========================
    m.utility_co[Mode.Bike] = (
        + P.ASC_Bike
        + P.Time_Active_b * X('od.DISTBIKE')
        + P.bike_per_trips * X('tour.outbound_stops')
        + P.shopping_ifbike * X('(tour.n_shopping_out > 0)')
        + P("Income:Active") * X("person.incp") * X("person.incp <= 17")
    )

    # =========================
    # TRANSIT
    # =========================
    if safety:
        m.utility_co[Mode.Transit] = (
            + P.ASC_Transit
            + P.Time_INVT * X('DRV_LOC_WLK_WAIT__AM')
            + P("Income:transit") * X("person.incp") * X("person.incp <= 17")
        )
    else:
        m.utility_co[Mode.Transit] = (
            + P.ASC_Transit
            + P.Time_INVT * X('DRV_LOC_WLK_WAIT__AM')
            + P("Income:transit") * X("person.incp") * X("person.incp <= 17")
        )

    return m


# ===============================
# MODE SPECIFICATIONS
# ===============================

def mode_spec_edu(m, Mode, safety):

    # =========================
    # DRIVE ALONE
    # =========================
    if safety:
        m.utility_co[Mode.DA] = (
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P.crash_random * X('log(od.CRASH+1)')
        )
    else:
        m.utility_co[Mode.DA] = (
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
        )

    # =========================
    # SHARED / OTHER
    # =========================
    if safety:
        m.utility_co[Mode.Other] = (
            + P.ASC_SR
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P.crash_random * X('log(od.CRASH+1)')
        )
    else:
        m.utility_co[Mode.Other] = (
            + P.ASC_SR
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P.shared_per_trips * X('tour.outbound_stops')
        )

    # =========================
    # WALK
    # =========================
    m.utility_co[Mode.Walk] = (
        + P.ASC_Walk
        + P.Time_Active * X('od.DISTWALK')
        + P.walk_per_trips * X('tour.outbound_stops')
    )

    # =========================
    # BIKE
    # =========================
    m.utility_co[Mode.Bike] = (
        + P.ASC_Bike
        + P.Time_Active_b * X('od.DISTBIKE')
        + P.bike_per_trips * X('tour.outbound_stops')
        + P.shopping_ifbike * X('(tour.n_shopping_out > 0)')
    )

    # =========================
    # TRANSIT
    # =========================
    if safety:
        m.utility_co[Mode.Transit] = (
            + P.ASC_Transit
            + P.Time_INVT * X('DRV_LOC_WLK_WAIT__AM')
        )
    else:
        m.utility_co[Mode.Transit] = (
            + P.ASC_Transit
            + P.Time_INVT * X('DRV_LOC_WLK_WAIT__AM')
        )

    return m

# ===============================
# MODE SPECIFICATIONS
# ===============================

def mode_spec_nonmandatory(m, Mode, safety):

    # =========================
    # DRIVE ALONE
    # =========================
    if safety:
        m.utility_co[Mode.DA] = (
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P("Income:Dr") * X("person.incp") * X("person.incp <= 17")
            + P.crash_random * X('log(od.CRASH+1)')
        )
    else:
        m.utility_co[Mode.DA] = (
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P("Income:Dr") * X("person.incp") * X("person.incp <= 17")
        )

    # =========================
    # SHARED RIDE / OTHER
    # =========================
    if safety:
        m.utility_co[Mode.Other] = (
            + P.ASC_SR
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P.crash_random * X('log(od.CRASH+1)')
            + P("Income:OTHER") * X("person.incp") * X("person.incp <= 17")
        )
    else:
        m.utility_co[Mode.Other] = (
            + P.ASC_SR
            + P.Time_INV * X('od.SOVTOLL_TIME__AM')
            + P.shared_per_trips * X('tour.outbound_stops')
            + P("Income:OTHER") * X("person.incp") * X("person.incp <= 17")
        )

    # =========================
    # WALK
    # =========================
    m.utility_co[Mode.Walk] = (
        + P.ASC_Walk
        + P.Time_Active * X('od.DISTWALK')
        + P.walk_per_trips * X('tour.outbound_stops')
        + P("Income:Active") * X("person.incp") * X("person.incp <= 17")
    )

    # =========================
    # BIKE
    # =========================
    m.utility_co[Mode.Bike] = (
        + P.ASC_Bike
        + P.Time_Active_b * X('od.DISTBIKE')
        + P.bike_ownership * X('person.bikes >= 1')
        + P.bike_per_trips * X('tour.outbound_stops')
        + P.shopping_ifbike * X('(tour.n_shopping_out > 0)')
        + P("Income:Active") * X("person.incp") * X("person.incp <= 17")
    )

    # =========================
    # TRANSIT
    # =========================
    if safety:
        m.utility_co[Mode.Transit] = (
            + P.ASC_Transit
            + P.Time_INV_T * X('od.SOVTOLL_TIME__AM')
            + P("Income:transit") * X("person.incp") * X("person.incp <= 17")
        )
    else:
        m.utility_co[Mode.Transit] = (
            + P.ASC_Transit
            + P.Time_INV_T * X('od.SOVTOLL_TIME__AM')
            + P("Income:transit") * X("person.incp") * X("person.incp <= 17")
        )

    return m



def apply_mode_spec(m, Mode, purpose, submodel, safety):

    if purpose == "N":
        return mode_spec_nonmandatory(m, Mode, safety)

    if submodel == "work":
        return mode_spec_work(m, Mode, safety)

    if submodel == "edu":
        return mode_spec_edu(m, Mode, safety)

    raise ValueError("Unknown mode specification")



# ===============================
# DESTINATION SPECIFICATIONS
# ===============================

def dest_spec_work(mm, safety):

    if safety:
        mm.utility_ca = (
            + P.logsums * X('base.logsum')
            + P.distance * X('log(od.DIST+1)')
            + P.crash_random * X('log(od.CRASH+1)')
            # FULL WORK TERMS FROM ORIGINAL
        )
    else:
        mm.utility_ca = (
            + P.logsums * X('base.logsum')
            + P.distance * X('log(od.DIST+1)')
            # FULL WORK TERMS
        )

    return mm


def apply_destination_spec(mm, purpose, submodel, safety):

    if purpose == "N":
        return dest_spec_nonmandatory(mm, safety)

    if submodel == "work":
        return dest_spec_work(mm, safety)

    if submodel == "edu":
        return dest_spec_edu(mm, safety)

    raise ValueError("Unknown destination specification")


def dest_spec_edu(mm, safety):

    if safety:
        mm.utility_ca = (
            + P.logsums * X('base.logsum')
            + P.distance * X('log(od.DIST+1)')
            + P.crash_random * X('log(od.CRASH+1)') * X('tour.tour_mode in [1,2,5]')
            + P.crash_dist_interact * X('log(od.CRASH+1)') * X('log(od.DIST+1)') * X('tour.tour_mode in [1,2,5]')

            + P.educational_post_school * X('emp.Education') * X('person.STUDYING in [3,4]')
            + P.EDUCATION_early_q_enrolments *
                (X("log(emp.Primary_total_priv+1)") + X("log(emp.Primary+1)"))
                * X('person.STUDYING in [1]')

            + P.EDUCATION_highschool_enrolments *
                (X("log(emp.Secondary_total_priv+1)") + X("log(emp.Secondary+1)"))
                * X('person.STUDYING in [2]')

            + P.educational_post_area * X('log(emp.education+1)') * X('person.STUDYING in [3,4]')
            + P.educatioal_buildings * X('emp.Education')
            + P.educational_area * X('log(emp.education+1)')

            + P.origin_distance_interact *
                X('log(od.DIST+1)') *
                (1 / (X('emp.education+1')))
        )
    else:
        mm.utility_ca = (
            + P.logsums * X('base.logsum')
            + P.distance * X('log(od.DIST+1)')

            + P.educational_post_school * X('emp.Education') * X('person.STUDYING in [3,4]')
            + P.EDUCATION_early_q_enrolments *
                (X("log(emp.Primary_total_priv+1)") + X("log(emp.Primary+1)"))
                * X('person.STUDYING in [1]')

            + P.EDUCATION_highschool_enrolments *
                (X("log(emp.Secondary_total_priv+1)") + X("log(emp.Secondary+1)"))
                * X('person.STUDYING in [2]')

            + P.educational_post_area * X('log(emp.education+1)') * X('person.STUDYING in [3,4]')
            + P.educatioal_buildings * X('emp.Education')
            + P.educational_area * X('log(emp.education+1)')

            + P.origin_distance_interact *
                X('log(od.DIST+1)') *
                (1 / (X('emp.education+1')))
        )

    return mm



def dest_spec_work(mm, safety):

    if safety:
        mm.utility_ca = (
            + P.logsums * X('base.logsum')
            + P.distance * X('log(od.DIST+1)')
            + P.crash_random * X('log(od.CRASH+1)') * X('tour.tour_mode in [1,2,5]')
            + P.crash_dist_interact * X('log(od.CRASH+1)') * X('log(od.DIST+1)') * X('tour.tour_mode in [1,2,5]')

            + P.finance_jobs * X('log(emp.FPSEMPN+1)') * X('person.indp in [11,13,12,10]')
            + P.health_jobs * X('log(emp.HEREMPN+1)') * X('person.indp in [17,16]')
            + P.agriculture_jobs * X('log(emp.AGREMPN+1)') * X('person.indp in [1,2]')
            + P.Manufacture_jobs * X('log(emp.MWTEMPN+1)') * X('person.indp in [3,4,5,6]')
            + P.other_jobs * X('log(emp.OTHEMP+1)') * X('person.indp in [19,14,15]')
            + P.retail_job * X('log(emp.RETEMPN+1)') * X('person.indp in [7,8]')

            + P.industrial_area * X('log(emp.industrial+1)')
            + P.construction_area * X('log(emp.construction+1)')
            + P.commercial_area * X('log(emp.commercial+1)')
            + P.education_area * X('log(emp.education+1)')
            + P.retrail_area * X('log(emp.retail+1)')

            + P.origin_distance_interact *
                X('log(od.DIST+1)') *
                (1 / (
                    X('emp.education+1')
                    + X('emp.commercial+1')
                    + X('emp.industrial+1')
                    + X('emp.retail+1')
                    + X('emp.construction+1')
                ))
        )
    else:
        mm.utility_ca = (
            + P.logsums * X('base.logsum')
            + P.distance * X('log(od.DIST+1)')

            + P.finance_jobs * X('log(emp.FPSEMPN+1)') * X('person.indp in [11,13,12,10]')
            + P.health_jobs * X('log(emp.HEREMPN+1)') * X('person.indp in [17,16]')
            + P.agriculture_jobs * X('log(emp.AGREMPN+1)') * X('person.indp in [1,2]')
            + P.Manufacture_jobs * X('log(emp.MWTEMPN+1)') * X('person.indp in [3,4,5,6]')
            + P.other_jobs * X('log(emp.OTHEMP+1)') * X('person.indp in [19,14,15]')
            + P.retail_job * X('log(emp.RETEMPN+1)') * X('person.indp in [7,8]')

            + P.industrial_area * X('log(emp.industrial+1)')
            + P.construction_area * X('log(emp.construction+1)')
            + P.commercial_area * X('log(emp.commercial+1)')
            + P.education_area * X('log(emp.education+1)')
            + P.retrail_area * X('log(emp.retail+1)')

            + P.origin_distance_interact *
                X('log(od.DIST+1)') *
                (1 / (
                    X('emp.education+1')
                    + X('emp.commercial+1')
                    + X('emp.industrial+1')
                    + X('emp.retail+1')
                    + X('emp.construction+1')
                ))
        )

    return mm

def dest_spec_nonmandatory(mm, safety):

    if safety:
        mm.utility_ca = (
            + P.logsums * X('base.logsum')
            + P.distance * X('log(od.DIST+1)')
            + P.crash_random * X('log(od.CRASH+1)') * X('tour.tour_mode in [1,2,5]')
            + P.crash_dist_interact * X('log(od.CRASH+1)') * X('log(od.DIST+1)') * X('tour.tour_mode in [1,2,5]')

            + P.shopping * X('emp.Shopping') * X('tour.main_purpose in ["shopping"]')
            + P.escorting * X('emp.ESCORT') * X('tour.main_purpose in ["escort"]')
            + P.social * X('emp.Entertainment') * X('tour.main_purpose in ["social"]')
            + P.other * X('emp.OTHERMAINTENANCE') * X('tour.main_purpose in ["othmaint"]')
            + P.other_disc * X('emp.OTHERDISCRFacilities') * X('tour.main_purpose in ["othdiscr"]')

            + P.residentail_att * X('log(emp.residential+1)')
            + P.retail_att * X('log(emp.retail+1)')
            + P.CBD * X('emp.city')
            + P.eating * X('emp.EATOUT') * X('tour.main_purpose in ["shopping"]')

            + P.origin_distance_interact *
                X('log(od.DIST+1)') *
                (1 / (X('emp.education+1') + X('emp.commercial+1') + X('emp.retail+1')))
        )
    else:
        mm.utility_ca = (
            + P.logsums * X('base.logsum')
            + P.distance * X('log(od.DIST+1)')

            + P.shopping * X('emp.Shopping') * X('tour.main_purpose in ["shopping"]')
            + P.escorting * X('emp.ESCORT') * X('tour.main_purpose in ["escort"]')
            + P.social * X('emp.Entertainment') * X('tour.main_purpose in ["social"]')
            + P.other * X('emp.OTHERMAINTENANCE') * X('tour.main_purpose in ["othmaint"]')
            + P.other_disc * X('emp.OTHERDISCRFacilities') * X('tour.main_purpose in ["othdiscr"]')

            + P.residentail_att * X('log(emp.residential+1)')
            + P.retail_att * X('log(emp.retail+1)')
            + P.CBD * X('emp.city')
            + P.eating * X('emp.EATOUT') * X('tour.main_purpose in ["shopping"]')

            + P.origin_distance_interact *
                X('log(od.DIST+1)') *
                (1 / (X('emp.education+1') + X('emp.commercial+1') + X('emp.retail+1')))
        )

    return mm


def run_scenario(purpose, submodel, safety):

    print(f"\n==============================")
    print(f"Running scenario: {purpose}-{submodel}, safety={safety}")
    print(f"==============================")

    # -------------------------------------------------
    # 1️⃣ Load Data
    # -------------------------------------------------
    base_data = load_base_data()
    persons = base_data["persons"]
    tours   = base_data["tours"]
    land    = base_data["land"]

    tours = tours[(tours["DTAZ"] != 0) & (tours["OTAZ"] != 0)& (tours["DTAZ"] != 330) & (tours["OTAZ"] != 330)].copy()

    tours["ptour_id"] = (
        tours["person_id"].astype(str) + "_" +
        tours["tour_id"].astype(str)
    )

    # ✅ Filter by purpose
    tours = filter_tours_by_purpose(tours, purpose, submodel)

    if len(tours) == 0:
        print("⚠ No tours for this segment. Skipping.")
        return

    # -------------------------------------------------
    # 2️⃣ Define Modes
    # -------------------------------------------------
    Mode = Dict(
        DA=1,
        Other=2,
        Walk=3,
        Bike=4,
        Transit=5,
    ).freeze()

    tours["tour_mode"] = tours["tour_mode"].map(Mode).fillna(2)

    # -------------------------------------------------
    # 3️⃣ Open Skims
    # -------------------------------------------------
    skims_path = os.path.join(
        os.path.dirname(__file__),
        "downcasted_matrices_base.omx"
    )
    skims = omx.open_file(skims_path)

    # -------------------------------------------------
    # 4️⃣ Estimate MODE
    # -------------------------------------------------
    dt_mode = build_mode_estimation_tree(
        persons, tours, land, skims, Mode
    )

    mode_model = build_mode_model(dt_mode, Mode, submodel, safety)
    mode_model = estimate_model(mode_model)

    # -------------------------------------------------
    # 5️⃣ Compute Logsums
    # -------------------------------------------------
    logsums = compute_logsums_fast(mode_model, dt_mode)

    # -------------------------------------------------
    # 6️⃣ Estimate DESTINATION
    # -------------------------------------------------
    dt_dest = build_destination_estimation_tree(
        tours, persons, land, skims, logsums
    )

    dest_model = build_destination_model(
        dt_dest, purpose, submodel, safety
    )

    dest_model = estimate_model(dest_model)

    # -------------------------------------------------
    # 7️⃣ Predict DESTINATION
    # -------------------------------------------------
    dest_probs = dest_model.probability()
    chosen_dest = monte_sample(dest_probs)+1

    # -------------------------------------------------
    # 8️⃣ Predict MODE conditional on chosen DTAZ
    # -------------------------------------------------
    tours_pred = tours.copy()
    tours_pred["DTAZ"] = chosen_dest

    dt_mode_pred = build_mode_estimation_tree(
        persons, tours_pred, land, skims, Mode
    )

    mode_model.datatree = dt_mode_pred
    mode_model.mangle()
    mode_model.unmangle()

    mode_probs = mode_model.probability()
    chosen_mode = monte_sample(mode_probs)+1

    # -------------------------------------------------
    # 9️⃣ Save Results
    # -------------------------------------------------
    results = tours[["ptour_id"]].copy()
    results["pred_DTAZ"] = chosen_dest
    results["pred_mode"] = chosen_mode

    suffix = "" if safety else "_NOS"

    filename = f"tour_predictions_{purpose}_{submodel}{suffix}.csv"
    results.to_csv(filename, index=False)

    print(f"✅ Saved {filename}")

if __name__ == "__main__":

    scenarios = [
        ("M", "work", 1),
        ("M", "edu", 1),
        ("N", None, 1),
        ("M", "work", 0),
        ("M", "edu", 0),
        ("N", None, 0),
    ]

    for purpose, submodel, safety in scenarios:
        run_scenario(purpose, submodel, safety)

