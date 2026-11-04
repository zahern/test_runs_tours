"""
predict_and_export_dta_tested.py
==================================
Prediction-only run (no re-estimation): loads the new_pop synthetic
population (remapped onto the ABM's standard 1-330 ORDER zone numbering),
reuses the pipeline's already-fitted stage1-5 models from saved_data/,
runs Stage 5b + Stage 6, and writes vehicle.dat directly into the
DTA_Brisbane_tested DynusT network — with OTAZ/DTAZ remapped through a
spatial zone crosswalk, since that network's own zone numbering (1-1515)
does not match the ABM's zone numbers (1-330) at all.

IMPORTANT — read before running
--------------------------------
1. stage5_models.pkl in saved_data/ is dated 31/05/2026. It does NOT
   include whatever ga_pred_run06 (job 23300124) was computing — that job
   was killed by the 24h PBS walltime limit before it could write a
   refreshed stage5_models.pkl (see runs/ga_parallel/run06/merged/
   ga_results.pkl for the raw GA search state, and note the job's log was
   spinning through the same ~20 "writing fresh flow code" hashes right
   before it died — worth investigating separately, may be why it never
   finished in time). Predictions here reflect the PRE-GA-refinement
   destination/mode choice models. Re-run this script once a fresh
   stage5_models.pkl exists if you want the GA-optimised version instead.

2. This OVERWRITES
       Z:\\test_runs_tours\\DTA\\DTA_Brisbane_tested\\vehicle.dat
   directly. The previous file is copied to vehicle.dat.bak_<timestamp>
   first as a cheap safety net (it's a 24h+ HPC artifact).

3. Population source: new_pop, loaded via NEW_POP_MAPPED=1 (NOT plain
   NEW_POP=1 — that would leave OTAZ/DTAZ as raw 9-digit SA2 codes, which
   are incompatible with both the ABM's own models and the zone
   crosswalk). NEW_POP_MAPPED remaps new_pop's population back onto the
   standard 1-330 ORDER scheme that stage1-5 models were estimated on.
   Per new_pop/todo.txt, the finer-resolution rebuild of HHTS-based
   activity models for that zone scope is NOT finished yet — this run
   applies the standard, already-fitted 330-zone models to the new_pop
   population, it does not re-estimate anything at new_pop's resolution.

4. Zone crosswalk: build_dynust_zone_crosswalk.py matches each of the
   ABM's 330 zones to its nearest DynusT zone centroid in
   DTA_Brisbane_tested. Only ~228/330 ABM zones are actually within this
   network's coverage (the rest — Sunshine Coast, Gold Coast, regional
   towns — are 9-114 km outside it); trips to/from the other ~102 zones
   are dropped rather than forced onto a distant/wrong DynusT zone. This
   script builds the crosswalk automatically if it doesn't exist yet at
   saved_data/abm_to_dynust_zone_crosswalk.csv.

Usage
-----
    python predict_and_export_dta_tested.py

Expect this to take substantially less time than a full estimation run
(no model fitting happens here — only prediction), but Stage 5 prediction
over ~2.7M persons across ~228 origin zones is still the dominant cost.
Monitor saved_data/pipeline.log for progress.
"""
import os

# Must be set BEFORE importing pipeline_logger — it reads these at import time
# to decide where the HHTS/landuse/synth population come from.
os.environ.setdefault('NEW_POP_MAPPED', '1')
os.environ.setdefault('NEW_POP', '0')       # keep HHTS + landuse on the standard 330-zone scope
os.environ.setdefault('MODE', 'predict_only')
# Full-SEQ zone scope (incl. Tweed + NSW Richmond Valley): NEW_POP=1 switches
# pipeline_logger to the SA2-direct zone universe (OTAZ IS the SA2 code) with
# the 371-zone skims/landuse, instead of the legacy 330-zone ORDER scope whose
# seq_zone_ids_.csv never carried the NSW zones ("100 SA2 codes not in zone
# mapping ... defaulted to TAZ 330"). Must be set BEFORE pipeline_logger is
# imported - it resolves scope at module level. Override with NEW_POP=0 to
# reproduce a legacy-scope export.
os.environ.setdefault('NEW_POP', '1')

import gc
import shutil
import datetime

import pandas as pd
import openmatrix as omx

import pipeline_logger as pl
from stage6_plan_finalisation import run_stage6
from spatial_assignment import get_zone_centroids_rich
from build_dynust_zone_crosswalk import build_crosswalk, build_origin_link_map

# Target network switched 2026-07: DTA_Brisbane_tested's zone files were
# stale from 2019-01; SEQ_Running_UE is the actively-maintained network
# (349 zones, full SEQ coverage). Its zone geometry comes from
# demand_/dta_349_zones.gpkg (EPSG:4326, zone_id matches network.dat) since
# it ships no GIS_Network.xls. Env overrides let a run target the old
# network without code edits.
# Paths resolved relative to this file so they work both on the HPC
# (/mnt/hpccs01/home/ahernz/test_runs_tours/...) and on the Z:-mounted
# Windows checkout - never hardcode the drive letter here.
_CODE_DIR     = os.path.dirname(os.path.abspath(__file__))
_TRT_ROOT     = os.path.normpath(os.path.join(_CODE_DIR, '..'))
DTA_DIR       = os.environ.get('DTA_DIR',
                               os.path.join(_TRT_ROOT, 'DTA', 'SEQ_Running_UE'))
DTA_ZONE_GPKG = os.environ.get('DTA_ZONE_GPKG',
                               os.path.join(_TRT_ROOT, 'demand_', 'dta_349_zones.gpkg'))
SCRATCH_DIR   = 'runs/predict_dta_tested'
CROSSWALK_CSV = os.environ.get('DTA_CROSSWALK_CSV',
                               'saved_data/abm_to_seq_running_ue_crosswalk.csv')


def ensure_crosswalk() -> None:
    if os.path.exists(CROSSWALK_CSV):
        print(f"Using existing crosswalk: {CROSSWALK_CSV}")
        return
    print("Zone crosswalk not found — building it now (see build_dynus_zone_crosswalk.py)...")
    # SEQ_Running_UE covers all of SEQ, so a generous match radius keeps
    # every ABM zone; genuinely-external zones still fall out via dist_m.
    # Zone list follows scope: NEW_POP=1 uses the SA2-keyed 371-zone
    # reference (matching OTAZ=SA2-code trips), legacy uses seq_zone_ids_.
    _abm_zones = (os.path.join(_TRT_ROOT, 'demand_', 'sa2_zones_reference_357.csv')
                  if os.environ.get('NEW_POP', '0') == '1'
                  else 'seq_zone_ids_.csv')
    # SEQ_Running_UE has no origin.dat/GIS_Network.xls - the gpkg is the ONLY
    # centroid source for it, so a missing gpkg must fail loudly here rather
    # than fall back to the legacy parser and confuse the traceback.
    _gpkg = DTA_ZONE_GPKG if os.path.exists(DTA_ZONE_GPKG) else None
    if _gpkg is None and not os.path.exists(os.path.join(DTA_DIR, 'origin.dat')):
        raise FileNotFoundError(
            f"Neither zone gpkg ({DTA_ZONE_GPKG}) nor {DTA_DIR}/origin.dat "
            "exists - cannot build zone centroids for this DTA network. "
            "Set DTA_ZONE_GPKG to the network's zone-polygon GeoPackage."
        )
    cw = build_crosswalk(DTA_DIR, _abm_zones, max_dist_m=15000.0,
                         zone_gpkg=_gpkg)
    os.makedirs(os.path.dirname(CROSSWALK_CSV), exist_ok=True)
    cw.to_csv(CROSSWALK_CSV, index=False)
    n_in = int(cw['in_network'].sum())
    print(f"  {n_in}/{len(cw)} ABM zones matched to {os.path.basename(DTA_DIR)}'s zones")


def main() -> None:
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    ensure_crosswalk()

    print("Loading landuse + HHTS (standard scope) and new_pop synthetic population...")
    sc, out, landuse = pl.load_data()
    hhts_persons = out['persons']
    hhts_tours = out.get('tours')

    synth = pl.load_synth_all_zones(pl.SYNTH_DIR, landuse=landuse)
    synth = pl.usual_locations(synth, landuse, hhts_tours=hhts_tours)
    synth = pl.cdap({'persons': hhts_persons}, synth)

    travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
    print(f"CDAP filter: {len(travellers):,}/{len(synth):,} travellers")
    del synth
    gc.collect()

    print("Stage 1 predict (activity frequency)...")
    synth_s1 = pl.predict_stage1(travellers, hhts_persons=hhts_persons, hhts_tours=hhts_tours)

    print("Stage 2 predict (day duration)...")
    synth_s2 = pl.predict_stage2(synth_s1)
    del synth_s1
    gc.collect()

    print("Stage 3 predict (activity duration)...")
    activities = pl.predict_stage3(synth_s2)
    del synth_s2
    gc.collect()

    print("Stage 4 predict (schedule placement)...")
    scheduled = pl.predict_stage4(activities)
    del activities
    gc.collect()

    print("Stage 5 predict (destination + mode choice)...")
    skims = omx.open_file(pl.PIPELINE_SKIMS_PATH, 'r')
    _out_s5 = {
        'persons': travellers,
        'trips': pd.DataFrame(),
        'tours': hhts_tours if hhts_tours is not None else pd.DataFrame(),
    }
    scheduled_dest = pl.predict_stage5(
        scheduled, out=_out_s5, landuse=landuse, skims=skims, sk=None, persons=travellers,
    )
    del scheduled, _out_s5
    gc.collect()

    print("Stage 5b predict (trip-level intermediate-stop destinations)...")
    try:
        scheduled_dest = pl.stage5b({'persons': travellers}, scheduled_dest, landuse)
    except Exception as e:
        print(f"  Stage 5b failed (non-fatal): {e}")

    print("Stage 6: validate, export, write vehicle.dat (zone-crosswalked)...")
    sa2_geo = get_zone_centroids_rich(landuse, cache_dir='saved_data/osm_cache')
    sa2_geo = sa2_geo[['ORDER', 'lat', 'lon']].copy()

    # Generation-link nodes per zone from the network's origin.dat. Without
    # this, vehicles fall back to usec=dsec=zone and DynusT rejects them
    # ("UP AND DOWNSTREAM NODES are the same").
    origin_dat = os.path.join(DTA_DIR, 'origin.dat')
    if os.path.exists(origin_dat):
        origin_link_map = build_origin_link_map(origin_dat)
        print(f"  origin.dat generation links: {len(origin_link_map)} zones")
    else:
        origin_link_map = None
        print(f"  WARNING: {origin_dat} missing — vehicle.dat will have "
              f"degenerate generation links and may be rejected by DynusT")

    run_stage6(
        plan                  = scheduled_dest,
        persons               = hhts_persons,
        landuse               = landuse,
        skims                 = skims,
        skim_lookup           = None,
        sa2_geo               = sa2_geo,
        save_dir              = SCRATCH_DIR,
        dynust_zone_crosswalk = CROSSWALK_CSV,
        repair_dwells         = False,   # stage5_models not reloaded in this script — skip repair loop
        dynus_sim_start_min   = 360,
        dynus_value_of_time   = 15.0,
        origin_link_map       = origin_link_map,
    )
    skims.close()

    src = os.path.join(SCRATCH_DIR, 'vehicle.dat')
    dst = os.path.join(DTA_DIR, 'vehicle.dat')
    if os.path.exists(dst):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = os.path.join(DTA_DIR, f'vehicle.dat.bak_{ts}')
        shutil.copy2(dst, backup)
        print(f"Backed up existing vehicle.dat -> {backup}")
    shutil.copy2(src, dst)
    print(f"vehicle.dat written -> {dst}")

    # path.dat must ship with vehicle.dat (DynusT terminates at demand
    # generation if path.dat is empty).
    src_p = os.path.join(SCRATCH_DIR, 'path.dat')
    dst_p = os.path.join(DTA_DIR, 'path.dat')
    if os.path.exists(src_p):
        shutil.copy2(src_p, dst_p)
        print(f"path.dat written -> {dst_p}")


if __name__ == '__main__':
    main()
