# estimation_sf — ActivitySim San Francisco estimation-mode benchmark

Staged 2026-09-28 for M0 (reference) vs M1 (pipeline default) vs M2 (searched)
estimation benchmarking, replacing the 33-tour prototype_mtc regress sample.

## Provenance (all public ActivitySim Consortium data)

- `data_sf/` (households/persons/land_use/readme + `survey_data/`):
  `activitysim` repo, `activitysim/examples/example_estimation/data_sf/`
  (master, Sep 2026). readme: "2000 household data file generated from
  data with random seed 1".
- `configs_estimation/` (estimation.yaml, logging.yaml, settings.yaml):
  `activitysim` repo, `activitysim/examples/example_estimation/configs/`.
- `data_sf/skims.omx` (39.8 MB, 190-zone SF skims):
  `activitysim_resources` repo, `mtc_data_sf/skims.omx`
  (sha256 per example_manifest.yaml:
  579d6007266db3b055d0f9e4814004f4d5ccfae27a36e40f4881e3662bc3d3f1 —
  verify after copying to HPC if needed).

Reference for the wiring: `activitysim/examples/example_manifest.yaml`,
entry `example_estimation_sf` ("Estimation mode San Francisco MTC dataset
with 190 zones, 2k households and 8k persons").

## Scale check (verified at staging)

- survey_tours.csv: 5,778 tours / 3,796 persons
- survey_trips.csv: 14,352 trips, real mode codes
  (DRIVEALONEFREE 4317, SHARED2FREE 1872, WALK_LOC 1828, WALK 1621, ...)
- tour_types present (tour_type/tour_category/origin/destination/start/end)

Segment counts are in the hundreds–thousands: comfortably above
STAGE5_MIN_TOURS=30, so full M1/M2 estimation can run (unlike the
33-tour mtc_ab_x sample, which skips stage 5 by design).

## Next step (HPC, from `test_runs_tours/code`)

Wire an `estimation_sf` mapping pass following the `mtc_connect.py`
pattern (map_households / map_persons / landuse / skim_df, then reuse the
mtc_compare.py M0/M1/M2 harness against `survey_*` as the observed
reference). Still TODO: nothing in this folder is referenced by code yet.
