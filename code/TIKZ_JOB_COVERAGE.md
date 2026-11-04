# TikZ workflow vs `full_stage3_suite` PBS job — coverage matrix

Reference diagram: `pipeline_model_workflow_tikz.tex`
Job wrapper:               `job_full_stage3_suite.pbs`
Estimator/predictor:       `run_stage3_model_suite.py`

This note records which box in the diagram is actually reflected by the
single-submission job, and how it is wired. All switches below default to *on*
in the PBS file; set to `0` to disable.

| TikZ block | In job? | How it is covered |
|------------|:-------:|-------------------|
| Population synthesis (veh/bike ownership) | YES | `--ownership` (RUN_OWNERSHIP=1): `predict_vehicle_bike_ownership` imputes `vehrd` (OrderedLogit) + `bikes` (ZeroInflatedOrderProbit) when missing/all-zero, propagates `vehrd/bikes/veh_bin/bike_bin` to the Stage-2 frame. |
| Accessibility computation | YES | Upstream in `stage1_activity_frequency.py --fresh` → `run_accessibility`. |
| Routine activity location (work/school) | YES | Upstream `--fresh` → `run_usual_locations`. |
| Telecommuting / WFH | partial | Present in upstream synth enrichment / Stage-1 features; not separately invoked by the suite runner. |
| Stage 0 — household coordination, joint travel | YES | `--stage0` (RUN_STAGE0=1): `cdap_households` coordinated CDAP + joint-tour frequency/participation/purpose folded into `n_*_out` counts, when synthetic persons carry a household id (HHID/hh_id/HHSIZE). |
| Stage 1 — activity frequency (ordered logit) | YES | `--fresh` → `stage1_activity_frequency.py --fresh`. |
| Stage 2 — day duration budget `B` | YES | `--fresh` → `stage2_day_duration.py`. |
| Stage 3 — duration + planning order (4 variants) | YES | `--models all`: lognormal, copula_hazard, lognormal_mdcev, copula_hazard_mdcev (+ Polaris planning-order RRM). |
| Stage 4 — schedule placement / timing (RRM) | YES | `--run-stage4` (RUN_STAGE4=1) → `STAGE4_TIMING_MODEL=rrm`. |
| Stage 5 — tour dest + mode, trip chain | YES | `--run-stage5-6` (RUN_STAGE5_6=1) → `run_stage5`. |
| Stage 5b — trip destination (intermediate stops) | YES | `--run-stage5b` (RUN_STAGE5B=1): first variant estimates `stage5b_models.pkl`, later variants reuse predict-only; Stage 6 consumes the augmented trips. |
| Stage 6 — trip mode choice (nested logit) | YES | `run_stage6` per variant. |
| Output artifacts (stage_*.csv, DTA, validation) | partial | Per-variant `stageX_output.csv` are written; DTA/dynusT export and validation report not produced by the suite wrapper. |
| GA hypothesis-testing feedback | OPTIONAL | `--ga-results` (env `GA_RESULTS_PKL`): when set, GA-optimised feature sets are used for Stage 3/4/5. Empty = defaults. |
| Micro-simulation / assignment skim loop | NOT in this job | Closed by `equilibrium_loop.py` / `run_full_pipeline_dta_tested.py`; the suite is a single forward pass. |

## Notes

- Stages 0, 5b, ownership and GA overrides were added to `run_stage3_model_suite.py`
  during this pass; py_compile passes.
- The stage chain (S0 → S1 → S2 → S3 → S4 → S5 → S5b → S6) matches the TikZ
  arrows exactly.
- Rows marked "partial" or "NOT in job" are covered by sibling scripts, not this
  one PBS file.