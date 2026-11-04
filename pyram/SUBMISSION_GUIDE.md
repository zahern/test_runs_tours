# PYRAM Job Submission Guide

## Overview

This guide covers all ways to submit and run PYRAM pipelines, from simple local runs to full HPC submissions with various model configurations.

---

## Quick Start

### 1. Local Development / Testing

```bash
# Install in development mode
cd pyram
pip install -e .[visualizer]

# Create default config
pyram init -o config.yaml

# Edit config.yaml with your data paths
# Then run pipeline
pyram run -c config.yaml --dry-run  # See execution plan
pyram run -c config.yaml            # Run full pipeline
```

### 2. Single Model Estimation (HHTS Data)

```bash
# Estimate all models
pyram estimate -c config.yaml -m all

# Estimate specific models
pyram estimate -c config.yaml -m destination
pyram estimate -c config.yaml -m mode
pyram estimate -c config.yaml -m parking
pyram estimate -c config.yaml -m trip_dest
```

---

## HPC / PBS Submission

### Standard Full Pipeline

```bash
# Submit standard pipeline (MNL destination, NL mode, parking, DTA off)
qsub job_submit_ga_parallel_chain.pbs
```

### With Environment Variables

```bash
# Override any config via PYRAM_ prefix (double underscore = nested key)
qsub -v "
  RUN_TAG=my_experiment,
  PYRAM_DESTINATION__MODEL_TYPE=nl_distance,
  PYRAM_MODE__MIXED_LOGIT=true,
  PYRAM_DTA__ENABLE=true,
  PYRAM_DTA__ITERATIONS=3,
  STAGE_MEM=256GB,
  PRED_MEM=256GB,
  STAGE_WALLTIME=48:00:00
" job_submit_ga_parallel_chain.pbs
```

### Environment Variable Reference

| Variable | Config Path | Description |
|----------|-------------|-------------|
| `RUN_TAG` | - | Output directory suffix |
| `STAGE_NCPUS` | `hpc.stage_ncpus` | CPUs per stage job |
| `STAGE_MEM` | `hpc.stage_mem` | Memory per stage (e.g., 128GB) |
| `STAGE_WALLTIME` | `hpc.stage_walltime` | Walltime per stage |
| `PRED_NCPUS` | `hpc.pred_ncpus` | CPUs for prediction |
| `PRED_MEM` | `hpc.pred_mem` | Memory for prediction |
| `PRED_WALLTIME` | `hpc.pred_walltime` | Walltime for prediction |
| `STAGE5_DEST_NL_DISTANCE` | `destination.model_type` | Enable NL distance bands (1=on) |
| `STAGE5_DEST_NL_DISTANCE` | `destination.model_type` | Enable NL distance bands |
| `STAGE5_TRIP_DEST_ENHANCE` | `activity_generation` | Trip-level destination enhancement |
| `STAGE5_ENABLE_MXL` | `mode.mixed_logit` | Enable Mixed Logit mode |
| `STAGE5_ENABLE_GSE` | `mode.model_type=mxl_gse` | Enable GSE Mixed Logit |
| `STAGE7_PARKING` | `parking.enable` | Enable parking choice |
| `DTA_FEEDBACK_ITERS` | `dta.iterations` | DTA feedback iterations (0=off) |
| `DTA_ENGINE` | `dta.engine` | DTA engine (matsim/sumo/dynust) |
| `CONDA_ENV` | `hpc.conda_env` | Conda environment name |

---

## Model Configuration Options

### Destination Choice Models

| Model Type | Config | When to Use |
|------------|--------|-------------|
| MNL | `model_type: mnl` | Baseline, fast estimation |
| NL Distance Bands | `model_type: nl_distance` | When distance correlation matters (recommended) |
| NL SA3 | `model_type: nl_sa3` | When geographic nesting (SA3 regions) |

```yaml
destination:
  model_type: nl_distance
  distance_bands: [0, 2, 5, 10, 100]  # km
  logsum_smoothing: true
  logsum_orthogonalize: true
  crash_imputation: true
```

### Mode Choice Models

| Model Type | Config | When to Use |
|------------|--------|-------------|
| NL | `model_type: nl` | Standard nested logit (fast) |
| MXL | `model_type: mxl, mixed_logit: true` | Random coefficients needed |
| MXL-GSE | `model_type: mxl_gse, mixed_logit: true` | Gradient Score Enhanced (best fit) |

```yaml
mode:
  model_type: mxl_gse
  mixed_logit: true
  panel_mxl: true
```

---

## Hazard Framework (Competing Hazard Activity Generation)

### When to Use

| Scenario | Use Hazard Framework? |
|----------|----------------------|
| Need continuous-time activity generation | ✅ Yes |
| Activities compete for time (work vs leisure) | ✅ Yes |
| Want covariate effects on activity timing | ✅ Yes |
| Simple cross-classification sufficient | ❌ No (use cross_classification) |

### Enable in Config

```yaml
activity_generation:
  model_type: competing_hazard
  hazard_type: piecewise_constant  # or weibull, exponential
  time_bins: 96          # 15-min intervals
  max_time_hours: 24.0
  use_metacount: true
  baseline_hazard_smoothing: 1.0
```

### Enable via Environment Variables

```bash
qsub -v "
  PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard,
  PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant,
  PYRAM_ACTIVITY_GENERATION__TIME_BINS=96,
  PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true
" job_submit_ga_parallel_chain.pbs
```

### Python API Usage

```python
from pyram.models import create_all_models
from pyram.config import Config

config = Config()
config.activity_generation.model_type = "competing_hazard"
config.activity_generation.hazard_type = "piecewise_constant"
config.activity_generation.time_bins = 96

models = create_all_models(config)

# Generate activities for synthetic population
activities = models['activity_generation'].predict_activities(persons, seed=42)

# Apply planning order
activities = models['planning_order'].predict_planning_order(activities, persons)

# Apply timing
activities = models['timing'].predict_timing(activities, persons)

# Routine locations
persons = models['work_location'].predict(persons, landuse, skim_lookup, 'work')
persons = models['school_location'].predict(persons, landuse, skim_lookup, 'school')
```

---

## DTA Feedback Loop

### When to Use

| Scenario | Use DTA Feedback? |
|----------|-------------------|
| Need equilibrated skims for final results | ✅ Yes |
| Quick exploratory runs | ❌ No (adds hours) |
| Policy scenarios with congestion effects | ✅ Yes |

### Enable

```yaml
dta:
  enable: true
  engine: matsim  # or sumo, dynust
  iterations: 3
  convergence_threshold: 0.02
  walltime_hours: 48
```

### Submit with DTA

```bash
qsub -v "
  DTA_FEEDBACK_ITERS=3,
  DTA_ENGINE=matsim,
  DTA_CONVERGENCE_THRESHOLD=0.02,
  STAGE_WALLTIME=48:00:00,
  PRED_WALLTIME=48:00:00
" job_submit_ga_parallel_chain.pbs
```

---

## NL Distance Band Benchmark

### When to Use

Compare MNL vs NL distance-band for model selection:

```bash
# Run model suite with NL benchmark
RUN_STAGE5_6=1 RUN_STAGE5_NL_DEST=1 python run_stage3_model_suite.py

# Or single run with NL
qsub -v "STAGE5_DEST_NL_DISTANCE=1,RUN_STAGE5_6=1" job_submit_ga_parallel_chain.pbs
```

---

## Trip-Level Destination Enhancement

### When to Use

For intermediate stop modeling (POLARIS/ActivitySim pattern):

```yaml
# In config.yaml (enabled by default)
# STAGE5_TRIP_DEST_ENHANCE=1
```

```bash
# Enable explicitly
qsub -v "STAGE5_TRIP_DEST_ENHANCE=1" job_submit_ga_parallel_chain.pbs
```

---

## Parking Choice

### When to Use

For precise parking locations + walk access legs (MATSim/SUMO export):

```bash
# Enable (default on)
qsub -v "STAGE7_PARKING=1" job_submit_ga_parallel_chain.pbs
```

---

## Complete Examples

### 1. Minimal Local Test

```bash
cd pyram
pyram init -o test_config.yaml
# Edit test_config.yaml with local data paths
pyram run -c test_config.yaml --dry-run
pyram run -c test_config.yaml
```

### 2. Standard HPC Run

```bash
qsub job_submit_ga_parallel_chain.pbs
```

### 3. NL Distance Band + Mixed Logit + DTA

```bash
qsub -v "
  RUN_TAG=nl_mxl_dta,
  PYRAM_DESTINATION__MODEL_TYPE=nl_distance,
  PYRAM_MODE__MODEL_TYPE=mxl_gse,
  PYRAM_MODE__MIXED_LOGIT=true,
  PYRAM_DTA__ENABLE=true,
  PYRAM_DTA__ITERATIONS=3,
  STAGE_MEM=256GB,
  PRED_MEM=256GB,
  STAGE_WALLTIME=48:00:00
" job_submit_ga_parallel_chain.pbs
```

### 4. Full Hazard Framework + DTA

```bash
qsub -v "
  RUN_TAG=hazard_dta,
  PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard,
  PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant,
  PYRAM_ACTIVITY_GENERATION__TIME_BINS=96,
  PYRAM_ACTIVITY_GENERATION__USE_METACOUNT=true,
  PYRAM_DESTINATION__MODEL_TYPE=nl_distance,
  PYRAM_MODE__MODEL_TYPE=mxl_gse,
  PYRAM_MODE__MIXED_LOGIT=true,
  PYRAM_DTA__ENABLE=true,
  PYRAM_DTA__ITERATIONS=3,
  STAGE_MEM=256GB,
  PRED_MEM=256GB,
  STAGE_WALLTIME=48:00:00
" job_submit_ga_parallel_chain.pbs
```

### 5. Benchmark Suite (MNL vs NL)

```bash
# Runs each variant twice: MNL + NL distance
RUN_STAGE5_6=1 RUN_STAGE5_NL_DEST=1 python run_stage3_model_suite.py \
  --out-dir output/benchmark \
  --models "mnl_base,nl_distance"
```

---

## Monitoring Jobs

```bash
# Check job status
qstat -u $USER

# Check specific job
qstat -f JOB_ID

# View output
cat job_submit_ga_parallel_chain.pbs.oJOBID
cat job_submit_ga_parallel_chain.pbs.eJOBID
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Job rejected by all possible destinations` | Memory request too high | Reduce `STAGE_MEM`, `PRED_MEM` |
| `Stage 5 destination model compiled with empty pf` | No valid utility terms | Check skim data, landuse columns |
| `Metacount estimation failed` | Insufficient data | Check HHTS sample size, reduce features |
| `DTA feedback not converging` | Increase iterations or threshold | Increase `DTA_FEEDBACK_ITERS` or `DTA_CONVERGENCE_THRESHOLD` |
| `Visualizer import error` | Missing dependencies | `pip install pyram[visualizer]` |

---

## Output Structure

```
runs/ga_parallel/{RUN_TAG}/
├── stage0/          # CDAP household coordination
├── stage1/          # Tour frequency
├── stage2/          # Timing/duration
├── stage3/          # Destination choice
├── stage4/          # Mode choice
├── stage5/          # Trip chain + parking
├── stage6/          # Plan finalization
├── stage7/          # Parking choice (if enabled)
├── merged/          # Merged GA results
├── ga_*/            # GA-optimized feature sets
├── dta_feedback/    # DTA iterations (if enabled)
├── matsim/          # MATSim export
└── plots/           # Validation plots
```

---

## Key Files to Check

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration |
| `job_submit_ga_parallel_chain.pbs` | PBS submission script |
| `stage*_models.pkl` | Fitted models per stage |
| `stage*_output.csv` | Stage outputs |
| `manifest.json` | Run metadata |
| `stage5_models.pkl` | Destination + mode models |
| `stage6_output.csv` | Final trip chains |
| `stage7_output.csv` | With parking (if enabled) |
| `stage7_dta_equilibrated.csv` | DTA-equilibrated plans |