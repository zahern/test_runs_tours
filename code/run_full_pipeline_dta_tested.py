"""
run_full_pipeline_dta_tested.py
=================================
Single end-to-end process: build models, predict population, write vehicle.dat.

Unlike predict_and_export_dta_tested.py (which reuses the stale 31/05
stage5_models.pkl and only predicts), this script:
  1. Reuses the already-completed GA feature search
     (runs/ga_parallel/run06/merged/ga_results.pkl — the search itself
     finished 5/07; only the finalize+predict step after it was failing)
  2. Re-estimates stage 1-5 with the GA-selected features
     (apply_ga_results_to_pipeline — same call job_ga_staged.pbs makes)
  3. Predicts on the new_pop population instead of the standard population
  4. Exports through the DTA_Brisbane_tested zone crosswalk, writing
     vehicle.dat directly into that network's folder

This relies on the _logsums() fix in stage5_destination_choice.py that
resolved the runaway per-zone Larch flow-recompilation bug — confirmed via
ga_pred_run06's log to be why Stage 5 prediction never got past its first
of 304 origin-zone batches in 24h. Without that fix this script would fail
the same way.

IMPORTANT — read before running
--------------------------------
1. This OVERWRITES Z:\\test_runs_tours\\DTA\\DTA_Brisbane_tested\\vehicle.dat
   directly (previous file backed up to vehicle.dat.bak_<timestamp> first).
2. Stage 5 runs its "baseline prefit" gate by default
   (GA_STAGE5_PREFIT_BASELINE=1) — a full unperturbed Stage 5 estimate+
   predict pass BEFORE applying GA-selected features, to guard against
   validating a broken GA feature set. This roughly doubles Stage 5 cost;
   set GA_STAGE5_PREFIT_BASELINE=0 to skip it if you trust the GA result.
3. Population: new_pop, loaded via NEW_POP_MAPPED=1 (remaps new_pop's raw
   SA2 codes back onto the ABM's 1-330 ORDER scheme so OTAZ/DTAZ stay
   compatible with both the stage models and the zone crosswalk).
4. Zone crosswalk: only ~228/330 ABM zones are inside DTA_Brisbane_tested's
   coverage (see build_dynust_zone_crosswalk.py) — trips to/from the other
   ~102 (Sunshine Coast, Gold Coast, regional) are dropped rather than
   forced onto a distant/wrong DynusT zone. Built automatically if missing.

Usage
-----
    python run_full_pipeline_dta_tested.py
"""
import os

# Must be set BEFORE importing pipeline_logger — it reads these at import time.
# ZONE MIGRATION (371): run SA2-native. NEW_POP=1 loads the new_pop population
# with OTAZ/DTAZ = SA2 code directly (no ORDER indirection) and uses the 371-zone
# 507 skims. NEW_POP_MAPPED is explicitly OFF — it collapses the 371 SA2 zones
# down to the legacy 330 ORDER scheme (nearest-order), which is exactly what the
# migration retires. 17 NSW-fringe zones have no population (354/371 populated) —
# accepted; they remain valid destinations via land use.
os.environ.setdefault('NEW_POP', '1')       # 371-zone SA2-native scope
os.environ.setdefault('NEW_POP_MAPPED', '0')
os.environ.setdefault('MODE', 'ga_staged')
os.environ.setdefault('GA_ESTIMATOR', 'sa_bandit')

import gc
import pickle
import shutil
import datetime

import openmatrix as omx

import pipeline_logger as pl
from ga_hypothesis_testing import apply_ga_results_to_pipeline
from stage5_destination_choice import _skim_flat, _skim_lookup
from stage6_plan_finalisation import run_stage6
from spatial_assignment import get_zone_centroids_rich
from build_dynust_zone_crosswalk import build_crosswalk

DTA_DIR        = os.environ.get('DTA_DIR') or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'DTA', 'DTA_Brisbane_tested'))
SAVE_DIR       = 'runs/full_pipeline_dta_tested'
CROSSWALK_CSV  = 'saved_data/abm_to_dynust_zone_crosswalk.csv'
GA_RESULTS_PKL = os.environ.get('GA_RESULTS_PKL', 'runs/ga_parallel/run06/merged/ga_results.pkl')


def ensure_crosswalk() -> None:
    if os.path.exists(CROSSWALK_CSV):
        print(f"Using existing crosswalk: {CROSSWALK_CSV}")
        return
    print("Zone crosswalk not found — building it now (see build_dynust_zone_crosswalk.py)...")
    # ZONE MIGRATION (371): SA2-native scope — map the 371 SA2 zones (by sa2_code
    # + centroid) to DynusT network zones. sa2_zones_reference_357.csv has the
    # 'sa2_code' column build_crosswalk keys on, replacing the legacy 330-zone
    # seq_zone_ids_.csv (ORDER-based).
    _abm_zone_csv = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'demand_', 'sa2_zones_reference_357.csv'))
    cw = build_crosswalk(DTA_DIR, _abm_zone_csv)
    os.makedirs(os.path.dirname(CROSSWALK_CSV), exist_ok=True)
    cw.to_csv(CROSSWALK_CSV, index=False)
    n_in = int(cw['in_network'].sum())
    print(f"  {n_in}/{len(cw)} ABM zones kept (inside DTA_Brisbane_tested's coverage)")


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    ensure_crosswalk()

    print(f"Loading GA search results from {GA_RESULTS_PKL}...")
    with open(GA_RESULTS_PKL, 'rb') as f:
        all_ga = pickle.load(f)
    print(f"  Stages available: {sorted(all_ga.results.keys())}")

    print("Loading landuse + HHTS (standard scope)...")
    sc, out, landuse = pl.load_data()

    print("Loading new_pop synthetic population (NEW_POP_MAPPED=1)...")
    synth = pl.load_synth_all_zones(pl.SYNTH_DIR, landuse=landuse)
    synth = pl.usual_locations(synth, landuse, hhts_tours=out.get('tours'))
    synth = pl.cdap(out, synth)
    travellers = synth[synth['CDAP'].isin(['M', 'N'])].copy()
    print(f"  CDAP filter: {len(travellers):,}/{len(synth):,} travellers")
    del synth
    gc.collect()

    print("Building stage 1-5 models with GA-selected features + predicting new_pop population...")
    scheduled_dest, s5_models = apply_ga_results_to_pipeline(
        ga_results = all_ga,
        out        = out,
        synth      = travellers,
        landuse    = landuse,
        skims_path = pl.PIPELINE_SKIMS_PATH,
        save_dir   = SAVE_DIR,
    )
    gc.collect()

    print("Stage 5b predict (trip-level intermediate-stop destinations)...")
    try:
        scheduled_dest = pl.stage5b(out, scheduled_dest, landuse)
    except Exception as e:
        print(f"  Stage 5b failed (non-fatal): {e}")

    print("Stage 6: validate, export, write vehicle.dat (zone-crosswalked)...")
    skims = omx.open_file(pl.PIPELINE_SKIMS_PATH, 'r')
    sk = _skim_lookup(_skim_flat(skims))
    sa2_geo = get_zone_centroids_rich(landuse, cache_dir='saved_data/osm_cache')
    sa2_geo = sa2_geo[['ORDER', 'lat', 'lon']].copy()

    run_stage6(
        plan                  = scheduled_dest,
        persons               = out['persons'],
        landuse               = landuse,
        skims                 = skims,
        skim_lookup           = sk,
        sa2_geo               = sa2_geo,
        stage5_models         = s5_models,
        save_dir              = SAVE_DIR,
        dynust_zone_crosswalk = CROSSWALK_CSV,
        repair_dwells         = True,
        dynus_sim_start_min   = 360,
        dynus_value_of_time   = 15.0,
    )
    skims.close()

    src = os.path.join(SAVE_DIR, 'vehicle.dat')
    dst = os.path.join(DTA_DIR, 'vehicle.dat')
    if os.path.exists(dst):
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = os.path.join(DTA_DIR, f'vehicle.dat.bak_{ts}')
        shutil.copy2(dst, backup)
        print(f"Backed up existing vehicle.dat -> {backup}")
    shutil.copy2(src, dst)
    print(f"vehicle.dat written -> {dst}")


if __name__ == '__main__':
    main()
