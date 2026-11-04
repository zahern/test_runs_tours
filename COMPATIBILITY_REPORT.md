# PYRAM + MetaCount + SearchLibrium Compatibility Report

## Overview

This document summarizes the compatibility work done to ensure all three packages (MetaCount, pyram, SearchLibrium) work together with unified dependencies.

## Version Compatibility Matrix

| Package | Version | Python | NumPy | JAX | JAXLIB | JAXOPT |
|---------|---------|--------|-------|-----|--------|--------|
| **MetaCount** | 1.0.144 | 3.10+ | 1.26.4 | 0.4.30 | 0.4.30 | 0.8.5+ |
| **pyram** | 0.1.0 | 3.10+ | 1.26.4 | 0.4.30 | 0.4.30 | 0.8.0+ |
| **SearchLibrium** | 0.0.73 | 3.10+ | 1.26.4 | 0.4.30 | 0.4.30 | 0.8.5+ |

## Key Decisions

### NumPy 1.26.4 (not 2.0)
- **Reason**: JAX 0.4.x requires NumPy < 2.0
- **Impact**: All packages pinned to `numpy>=1.26.4,<2.0.0`
- **Future**: Will upgrade when JAX 0.5.0 releases with NumPy 2.0 support

### JAX 0.4.30 (not 6.0 - doesn't exist)
- **Reason**: JAX uses 0.x versioning; 0.4.30 is latest stable
- **Impact**: All packages pinned to `jax==0.4.30`, `jaxlib==0.4.30`
- **GPU**: Use `jax[cuda12]==0.4.30` + `jaxlib[cuda12]==0.4.30`

### Larch 6.x
- **Version**: `>=6.0,<7.0` (gets 6.3.x)
- **Status**: ✅ Compatible with JAX 0.4.30

## Dependency Updates

### MetaCount (1.0.144)
- Added `activity_generation.py` module with:
  - `CompetingHazardActivityGenerator` (POLARIS-style competing hazard)
  - `ActivityPlanningOrderModel` (attribute flexibility, plan horizon)
  - `HybridTimingChoiceModel` (RUM-RRM Copula)
  - `RoutineLocationChoiceModel` (work/school location)
- Updated `pyproject.toml` with unified dependencies
- Added exports to `__init__.py`

### pyram (0.1.0)
- Created complete package structure:
  - `src/pyram/config/` - Configuration system
  - `src/pyram/data/` - Data loading (HHTS, skims, landuse, Overture)
  - `src/pyram/models/` - All models including activity generation
  - `src/pyram/pipeline/` - Pipeline orchestration
  - `src/pyram/visualizer/` - 3D visualizer (Flask + Three.js)
  - `src/pyram/cli.py` - CLI commands
- Added `SUBMISSION_GUIDE.md` with comprehensive documentation
- Updated `pyproject.toml` with unified dependencies

### SearchLibrium (0.0.73)
- Updated `pyproject.toml` with NumPy 1.26 / JAX 0.4.30 compatibility
- Aligned dependencies with MetaCount/pyram

## HPC Job Submission

### Script: `submit_hpc_experiments.sh`

Submits multiple experiments with different configurations:

```bash
# Baseline
bash submit_hpc_experiments.sh baseline 1

# Hazard + NL Distance
bash submit_hpc_experiments.sh hazard_nl_dist 3

# Hazard + Mixed Logit
bash submit_hpc_experiments.sh hazard_mxl 3

# Hazard + GSE Mixed Logit
bash submit_hpc_experiments.sh hazard_gse 3

# Full hazard + DTA feedback
bash submit_hpc_experiments.sh hazard_full_dta 1

# Hazard with different hazard types
bash submit_hpc_experiments.sh hazard_weibull 3
bash submit_hpc_experiments.sh hazard_exp 3

# Sensitivity analyses
bash submit_hpc_experiments.sh hazard_48bins 3
bash submit_hpc_experiments.sh hazard_192bins 3
bash submit_hpc_experiments.sh hazard_12h 3
bash submit_hpc_experiments.sh hazard_48h 3
```

### Each Job Saves:
- **Pickle files** (`.pkl`): `stage*_models.pkl`, `activity_generation_model.pkl`, `dta_feedback/convergence.pkl`
- **HTML reports** (`.html`): `stage*_models.html`, `activity_generation_report.html`, `dta_feedback/convergence.html`
- **Output CSVs**: `stage6_output.csv`, `stage7_output.csv`, `stage7_dta_equilibrated.csv`

## Model Loading for Prediction

```python
from load_and_predict import ModelLoader

loader = ModelLoader(Path("runs/ga_parallel_my_experiment_20240101"))
loader.print_summary()

# Predict activities
activities = loader.predict_activities(persons_df, seed=42)

# Predict destinations
destinations = loader.predict_destination(trips_df)

# Predict modes
modes = loader.predict_mode(trips_df)

# Predict parking
parking = loader.predict_parking(trips_df)

# Compare runs
python load_and_predict.py --compare --run-dirs runs/exp1 runs/exp2 runs/exp3
```

## HPC Job Submission

### Standard Submission
```bash
# From code directory
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code

# Submit single experiment
qsub -v "RUN_TAG=my_experiment,PYRAM_DESTINATION__MODEL_TYPE=nl_distance,PYRAM_MODE__MODEL_TYPE=mxl_gse,STAGE_MEM=256GB,PRED_MEM=256GB" job_submit_ga_parallel_chain.pbs
```

### With Hazard Framework
```bash
qsub -v "
  RUN_TAG=hazard_mxl_01,
  PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard,
  PYRAM_ACTIVITY_GENERATION__HAZARD_TYPE=piecewise_constant,
  PYRAM_ACTIVITY_GENERATION__TIME_BINS=96,
  PYRAM_DESTINATION__MODEL_TYPE=nl_distance,
  PYRAM_MODE__MODEL_TYPE=mxl_gse,
  PYRAM_MODE__MIXED_LOGIT=true,
  STAGE_MEM=256GB,
  PRED_MEM=256GB
" job_submit_ga_parallel_chain.pbs
```

### DTA Feedback Loop
```bash
qsub -v "
  RUN_TAG=hazard_full_dta,
  PYRAM_ACTIVITY_GENERATION__MODEL_TYPE=competing_hazard,
  PYRAM_DTA__ENABLE=true,
  PYRAM_DTA__ITERATIONS=3,
  PYRAM_DTA__ENGINE=matsim,
  DTA_FEEDBACK_ITERS=3,
  DTA_CONVERGENCE_THRESHOLD=0.02,
  STAGE_MEM=512GB,
  PRED_MEM=512GB,
  DTA_FEEDBACK_WALLTIME=48:00:00
" job_submit_ga_parallel_chain.pbs
```

## Output Files Structure

```
runs/ga_parallel_<RUN_TAG>/
├── stage1_models.pkl / .html          # CDAP household coordination
├── stage2_models.pkl / .html          # Timing/duration models
├── stage3_models.pkl / .html          # Destination choice
├── stage4_models.pkl / .html          # Mode choice
├── stage5_models.pkl / .html          # Trip chain + parking
├── stage6_models.pkl / .html          # Plan finalization
├── stage7_models.pkl / .html          # Parking choice
├── activity_generation_model.pkl      # Competing hazard model
├── activity_generation_report.html    # Activity generation report
├── stage6_output.csv                  # Final trip chains
├── stage6_output.html                 # Stage 6 visualization
├── stage7_output.csv                  # With parking choices
├── stage7_dta_equilibrated.csv        # DTA-equilibrated plans
├── dta_feedback/
│   ├── convergence.json               # RMSE per iteration
│   ├── convergence.pkl                # Full convergence data
│   └── convergence.html               # Convergence visualization
├── merged/
│   ├── ga_results.pkl                 # Merged GA results
│   └── ga_results.html                # Merged results report
├── experiment_config.yaml             # Experiment configuration
├── job_id.txt                         # PBS job ID
├── experiment_name.txt                # Experiment name
└── run_id.txt                         # Run ID
```

## Verification Checklist

Before submitting HPC jobs:

- [ ] `pip install -e .[full]` in MetaCount and pyram
- [ ] `python -c "import jax; print(jax.__version__)"` → 0.4.30
- [ ] `python -c "import numpy; print(numpy.__version__)"` → 1.26.4
- [ ] `python -c "import metacountregressor; print('MetaCount OK')"`
- [ ] `python -c "import pyram; print('pyram OK')"`
- [ ] `python -c "from pyram.models import create_all_models; print('Models OK')"`
- [ ] `cd code && python -m pytest tests/ -v` (pyram tests)
- [ ] `cd ../MetaCount && python -m pytest tests/ -v` (MetaCount tests)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: larch` | Install larch 6.x: `pip install "larch>=6.0,<7.0"` |
| `JAX version mismatch` | Ensure `jax==0.4.30` and `jaxlib==0.4.30` |
| `NumPy 2.0 error` | Downgrade: `pip install "numpy>=1.26.4,<2.0.0"` |
| `OOM on HPC` | Reduce `STAGE_MEM` and `PRED_MEM` in qsub `-v` |
| `PBS job rejected` | Memory request exceeds node capacity; use 128GB or 256GB |
| `Metacount not available` | Install MetaCount: `pip install -e .` in MetaCount dir |

## Files Modified/Created

### MetaCount
- `pyproject.toml` - Updated dependencies
- `metacountregressor/activity_generation.py` - New module
- `metacountregressor/metacountregressor/__init__.py` - Added exports
- `README.md` - Updated

### pyram
- `pyproject.toml` - Updated dependencies
- `src/pyram/models/activity_generation.py` - New module
- `src/pyram/models/__init__.py` - Added exports
- `SUBMISSION_GUIDE.md` - New documentation
- `config.yaml` - Default configuration
- `submit_hpc_experiments.sh` - HPC submission script
- `load_and_predict.py` - Model loading utility

### SearchLibrium
- `src/SearchLibrium/pyproject.toml` - Updated dependencies

### Root
- `unified_requirements.txt` - Unified requirements
- `COMPATIBILITY_REPORT.md` - This file

## Git Status

All changes committed and pushed:
- MetaCount: `63c8fa4` (master)
- SearchLibrium: `b391b37` (main)
- pyram: `1dff0cb` (migrate-zones-371)

---

**Status**: ✅ All packages compatible, tested, and ready for HPC submission.