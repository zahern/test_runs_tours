# Suncorp Telematics EB Analysis - HPC Setup

## Overview
This directory contains the Empirical Bayes (EB) analysis pipeline for the Suncorp Stadium event-day policing deployment study. The analysis uses 2.5M hard braking events and 414K segment-phase observations to compute EB-adjusted risk scores.

## Directory Structure
```
suncorp_telematics/
├── data/                              # Input data
│   ├── hard_braking_events.csv        # 2.5M HB events (March 2026)
│   ├── segment_roads_part0_5000.csv.gz # 414K segment-phase obs
│   ├── osm_roads_suncorp.geojson      # 7,825 road segments
│   └── road_meta.csv                  # Road metadata
│   ├── 18_prepare_eb_dataset.py       # Create near_miss_hb_ecf.csv
│   ├── 19_eb_analysis.py              # SPF fitting + EB adjustment
│   ├── 20_eb_adjusted_street_risk.py  # Event/Control EB comparison
│   ├── 21_eb_timewise_risk.py         # Time-resolved (phase) EB profiling
│   ├── 22_build_dashboard.py          # Interactive HTML time-based dashboard
│   ├── 23_conceptual_forecast.py      # Conceptual forward model (future events)
│   ├── 24_holdout_validation.py       # Event-based hold-out validation
│   └── 25_road_geometry_features.py   # OSM/Overture geometry features
├── results/                           # Output files
├── figures/                           # Generated figures
├── 01_prepare_data.pbs               # Job 1: Data prep
├── 01b_geometry.pbs                  # Job 1b: Road geometry features
├── 02_eb_analysis.pbs                # Job 2: EB analysis
├── 03_eb_street_risk.pbs             # Job 3: Street risk
├── 04_eb_timewise.pbs                # Job 4: Timewise EB
├── 05_dashboard.pbs                  # Job 5: Dashboard
├── 06_concept_holdout.pbs            # Job 6: Conceptual + hold-out
├── 07_temporal.pbs                   # Job 7: Temporal random-walk dynamics
└── submit_all.sh                     # Submit all jobs

```

## Dataset Summary
- **7,825 road segments** (Suncorp study area, 38.5 km²)
- **2,504,777 hard braking events** statewide
- **201,381 HB events** matched to study area segments
  - 200,941 Candidate
  - 414 Critical-Likely
  - 26 Extreme
- **414,121 segment-phase observations** (Event/Control)

## Jobs

### Job 1: Data Preparation (01_prepare_data.pbs)
**Runtime:** 2 hours | **Resources:** 4 CPUs, 32GB RAM

Snaps hard braking events to road segments and creates `near_miss_hb_ecf.csv`.

**Outputs:**
- `data/near_miss_hb_ecf.csv` (7,825 segments with HB counts)

### Job 1b: Road Geometry Features (01b_geometry.pbs)
**Runtime:** 1 hour | **Resources:** 4 CPUs, 32GB RAM

Derives crash-prone road-geometry / network covariates from `osm_roads_suncorp.geojson`
(curvature, sinuosity, intersection density, distance-to-intersection, lane count,
speed limit, road class, one-way, link flag) and optionally enriches from an Overture
transportation extract. These features are merged into `near_miss_hb_ecf.csv` and joined
into the segment table so the SPFs in Jobs 2/3/4 can model *why* a segment is risky, not
just *that* it is. Depends on Job 1.

**Outputs:**
- `data/road_geometry_features.csv` (7,825 segments x 10 geometry features)
- `data/near_miss_hb_ecf.csv` (re-written with the geometry columns merged in)

### Job 2: EB Analysis (02_eb_analysis.pbs)
**Runtime:** 4 hours | **Resources:** 8 CPUs, 64GB RAM

Fits Safety Performance Functions (negative binomial), computes EB adjustments, and runs 5-fold cross-validation. The SPF is augmented with OSM geometry covariates (road class, speed, lanes, curvature, distance-to-intersection) so risk reflects crash-prone road form, not just observed counts.

**Outputs:**
- `results/eb_results.tex` (LaTeX tables for paper)
- `results/eb_spf_coef.csv` (SPF coefficients)
- `results/eb_predictions.csv` (EB-adjusted predictions per segment)
- `figures/eb_obs_vs_pred.png` (Observed vs predicted/EB)
- `figures/eb_weights.png` (EB weights vs exposure)
- `figures/eb_severity_dist.png` (HB severity distribution)

### Job 3: EB Street Risk (03_eb_street_risk.pbs)
**Runtime:** 3 hours | **Resources:** 8 CPUs, 64GB RAM

Computes EB-adjusted risk scores separately for Event and Control windows, identifies event-specific corridors.

**Outputs:**
- `results/eb_street_risk.csv` (EB scores per segment)
- `results/eb_street_risk.md` (Summary report with top 20 segments)
- `figures/eb_street_risk_comparison.png` (Event vs Control)
- `figures/eb_street_risk_uplift_dist.png` (Uplift distribution)

### Job 4: Time-resolved EB (04_eb_timewise.pbs)
**Runtime:** 4 hours | **Resources:** 8 CPUs, 64GB RAM

Fits six phase-resolved SPFs (Event/Control x pre/during/post), computes phase-level
and 24-hour mean EB risk per segment, and benchmarks an EB SPF with road-class
covariates against the base SPF (the "better methods" check). Depends on Job 3.

**Outputs:**
- `results/eb_timewise_segment_phase.csv` (EB risk per segment x type x phase)
- `results/eb_timewise_hourly.csv` / `eb_timewise_hourly_control.csv` (24h profiles)
- `results/eb_timewise_summary.csv`, `results/method_comparison.csv`
- `results/eb_timewise.tex` (LaTeX summary for the paper)
- `figures/eb_timewise_phase_curve.png`, `figures/eb_timewise_hourly.png`

### Job 5: Interactive Dashboard (05_dashboard.pbs)
**Runtime:** 2 hours | **Resources:** 4 CPUs, 32GB RAM

Builds a self-contained interactive HTML dashboard of the time-based risk profile
(Event/Control toggle, clock-hour slider, phase buttons, spatial map, top-20 table)
from the Job 4 outputs. Depends on Job 4.

**Outputs:**
- `figures/risk_dashboard.html` (open in any browser; no server needed)
- `figures/risk_dashboard_data.json` (embedded data)

### Job 6: Conceptual Model + Hold-out (06_concept_holdout.pbs)
**Runtime:** 4 hours | **Resources:** 8 CPUs, 64GB RAM

Fits the conceptual forward model (event-factor regression for future fixtures) and
runs the event-based hold-out validation (leave-one-event-out across 16 events and a
prospective 2025->2026 test). Depends on Job 4.

**Outputs:**
- `results/conceptual_forecast.csv`, `results/conceptual_forecast_coef.csv`,
  `results/conceptual_forecast.tex`, `figures/conceptual_forecast_profile.png`
- `results/holdout_validation.csv`, `results/holdout_validation.tex`,
  `results/holdout_validation.md`

### Job 7: Temporal Random-Walk Dynamics (07_temporal.pbs)
**Runtime:** 2 hours | **Resources:** 8 CPUs, 64GB RAM

Proper time-series layer over the static EB pipeline:
- **A.** Local-level (random walk) state-space fit of network risk per Event/Control window (Kalman filtering via `UnobservedComponents`), with no-dynamics null comparison, RW innovation variance, and next-window forecast.
- **B.** Per-segment **dynamic EB**: latent risk evolves as a random walk between fixtures; momentum coefficient g* tuned by one-step-ahead validation (g\*=0.85 Event / 0.50 Control).
- **C.** Momentum test — lag-1 autocorrelation of demeaned segment log-rates.
- **D.** Holdout (2026 fixtures): dynamic-RW EB vs static EB (RMSE, Spearman, Top-N overlap).

Outputs:
- `results/temporal_level.csv`, `results/temporal_summary.csv`, `results/temporal_autocorr.csv`
- `results/temporal_holdout.csv`, `results/temporal_dynamic_event.csv`, `results/temporal_dynamics.tex`
- `figures/temporal_level_curve.png`, `figures/temporal_momentum.png`

**Design note:** matched control windows share `ref_dt` with their paired fixture
(same-day matched windows), so cross-day post-event decay is not identifiable in
this design — every control day has zero gap by construction. Temporal dependence
is tested instead via lag-1 autocorrelation of demeaned segment log-rates across
consecutive fixtures. Include unmatched control days separated from fixtures if
post-event recovery timescales are needed.

### Job 8: Crash Transferability Validation (08_transfer_validation.pbs)
**Runtime:** 1 hour | **Resources:** 4 CPUs, 32GB RAM

Tests whether hard-braking surrogates transfer to crashes:

- **A.** SEQ-wide ranking & calibration on EB-adjusted hotspots: Spearman between
  HB-EB and crash-EB rankings (top-N overlap), 10-decile calibration of mean
  crash rate per HB-risk bin. SEQ dataset: 10,780 segments with both outcomes.
- **B.** 5-fold CV: crashes ~ geometry vs crashes ~ geometry + log(HB+1)
  (RMSE, deviance, Spearman on holdout). HB improves holdout RMSE 19% (4.60→3.73).
- **C.** Suncorp SA2 zone validation: aggregate Suncorp HB-EB per SA2 via
  point-in-polygon, correlate with SA2 crash density (507-zone model).
  Result: zone-level transfer is weak (Spearman −0.02, top-5 overlap 20%),
  indicating Suncorp's 1-year event-window signal does not predict 5-year
  crash density at coarse zone geography — segment-level transfer (A/B) holds,
  zone-level does not. Bounds deployment to segment rankings.

Outputs:
- `results/crash_transfer_ranking.csv`, `results/crash_transfer_holdout.csv`,
  `results/suncorp_zone_validation.csv`, `results/crash_transfer_validation.tex`
- `figures/crash_transfer_rank_scatter.png`, `figures/crash_transfer_calibration.png`,
  `figures/suncorp_zone_scatter.png`

## Usage

### Option 1: Submit All Jobs at Once (Recommended)
```bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
bash submit_all.sh
```

This submits all jobs with dependency chains (Job 1b waits for Job 1, Job 2 waits for Job 1b, Job 3 waits for Job 2).

### Option 2: Submit Jobs Individually
```bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics

# Job 1
qsub 01_prepare_data.pbs

# Wait for Job 1 to complete, then:
qsub 02_eb_analysis.pbs

# Wait for Job 2 to complete, then:
qsub 03_eb_street_risk.pbs
```

### Monitor Jobs
```bash
# Check job status
qstat -u $USER

# Check specific job
qstat -f <job_id>

# View live output
tail -f suncorp_prepare.o*
tail -f suncorp_eb.o*
tail -f suncorp_street_risk.o*
```

## Environment
All jobs use the **`zigenv`** conda environment:
```bash
module load Anaconda3/2024.02-1
conda activate zigenv
```

Required packages (already in zigenv):
- pandas, numpy, scipy
- geopandas, shapely
- statsmodels
- matplotlib, seaborn
- sklearn

## Expected Results

### EB Analysis (Job 2)
- SPF: AIC ≈ 41,364, overdispersion α ≈ 1.18
- Mean EB weight: ≈ 0.081 (strong shrinkage)
- CV RMSE: ≈ 94 ± 26
- 5,071 segments analyzed

### Street Risk (Job 3)
- Event SPF: AIC ≈ 54,507
- Control SPF: AIC ≈ 63,803
- 7,517 segments analyzed
- Top segments: Stanley St, Grey St, Roma St, Riverside Expy

## Outputs for Paper
Results are formatted for direct inclusion in:
- **Police paper** (`new_paper/telematics_big_Data/main.tex`)
- **TRB paper** (if integrating Suncorp case study)

## Troubleshooting

### Job fails at data prep
- Check that `hard_braking_events.csv` exists and is readable
- Verify `osm_roads_suncorp.geojson` is valid

### Job fails at EB analysis
- Ensure `near_miss_hb_ecf.csv` was created by Job 1
- Check memory usage (may need more than 64GB for large datasets)

### Missing outputs
- Check PBS `.o*` files for Python tracebacks
- Verify paths are correctly updated by `sed` commands

## Notes
- All scripts automatically convert Windows paths to HPC paths via `sed`
- Results are automatically checked at job completion
- Jobs send email notifications on abort/begin/end (PBS -m abe)
- Jobs are restartable (PBS -r y)

## Contact
Zeke Ahern - z.ahern@qut.edu.au
