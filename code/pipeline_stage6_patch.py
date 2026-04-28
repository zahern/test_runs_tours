"""
pipeline_stage6_patch.py
=========================
Drop-in replacement for the stage6() function in pipeline_logger.py.

WHAT TO CHANGE IN pipeline_logger.py
--------------------------------------
Replace the existing stage6() function (lines ~216-230) with this version.
No other changes needed — get_zone_centroids_rich is already imported.

The only addition is:
  1. Build sa2_geo from landuse (same as spatial_plots already does)
  2. Pass sa2_geo= to run_stage6 so trip rows get lat/lon pinned on them
  3. Pass dynus_sim_start_min= and dynus_value_of_time= (optional tuning)

This enables:
  - origin_lat/lon + dest_lat/lon on every motorised trip row
  - saved_data/trips_DA.csv       — drive-alone trips with coordinates
  - saved_data/trips_SR.csv       — shared-ride trips with coordinates
  - saved_data/trips_Transit.csv  — transit trips with coordinates
  - saved_data/trips_motorised.csv— all motorised trips combined
  - saved_data/vehicle.dat        — DynusT demand file (veh_id, orig, dest,
                                    dep_time_sec, veh_type, info, comp, VOT)
"""


@timer("Stage 6 — plan validation + Aimsun export")
def stage6(out, scheduled_dest, landuse,
           stage5_models=None, skims=None, skim_lookup=None):
    """
    Stage 6: validate plan, attach coordinates, export Aimsun + DynusT outputs.
    """
    # ── build zone centroid table for coordinate attachment ────────────────
    # Reuses the same helper already used by spatial_plots().
    try:
        sa2_geo = get_zone_centroids_rich(landuse, cache_dir='saved_data/osm_cache')
        # keep ORDER column (not renamed to zone_id here — run_stage6 handles both)
        sa2_geo = sa2_geo[['ORDER', 'lat', 'lon']].copy()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"sa2_geo build failed in stage6: {e}")
        sa2_geo = None

    final_plan, trips_aimsun = run_stage6(
        plan               = scheduled_dest,
        persons            = out["persons"],
        landuse            = landuse,
        stage5_models      = stage5_models,
        skims              = skims,
        skim_lookup        = skim_lookup,
        sa2_geo            = sa2_geo,       # ← NEW: enables coordinate attachment
        repair_dwells      = stage5_models is not None,
        dynus_sim_start_min= 360,           # 6:00 AM simulation start
        dynus_value_of_time= 15.0,          # $/hour VOT in vehicle.dat
    )
    return final_plan, trips_aimsun
