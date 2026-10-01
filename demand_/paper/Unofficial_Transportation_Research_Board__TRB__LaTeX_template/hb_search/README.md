# HB vs Crash Prediction Model Search

## Overview

This package performs the **hard-braking vs crash prediction model search** using the `metacountregressor` Python package (v1.0.143+), specifically its `StructureEvaluatorLC` class and the AdaptiveDE/SA solvers from `Solvers_METAJAX`.

## Problem

The TRB paper (`preprint_main.tex`) needed:
1. A **properly separated** hard-braking transferability analysis (not lumped into a single model)
2. **Actual model search** using the `metacountregressor` package, not manual reimplementations
3. **Consistent logic** across all steps
4. Results **updated based on actual runs** on the 10,780-segment dataset

## Pipeline

The analysis runs a 6-step pipeline:

```
┌──────────────────────────────────────────────────────────┐
│  STEP 1: DE Feature Selection (separate per response)     │
│  scipy.optimize.differential_evolution over binary masks  │
│  with significance-penalised BIC                          │
├──────────────────────────────────────────────────────────┤
│  STEP 2: MCR Structure Search (metacountregressor)        │
│  StructureEvaluatorLC + AdaptiveDE/SA over roles          │
│  {fixed, random-independent, random-correlated}           │
├──────────────────────────────────────────────────────────┤
│  STEP 3: Specification Transfer (HARD-BRAKING TRANSFER)   │
│  Apply each response's final spec to the other response   │
│  Measure ΔBIC penalty → transferability verdict           │
├──────────────────────────────────────────────────────────┤
│  STEP 4: Encompassing Test                                │
│  Does predicted HB encompass geometry for crashes?        │
├──────────────────────────────────────────────────────────┤
│  STEP 5: Bivariate Copula                                 │
│  IFM + joint MLE + structure search (Frank/Normal/KS)     │
├──────────────────────────────────────────────────────────┤
│  STEP 6: Transferability Summary + Figures                │
│  Compile all results into paper-ready outputs             │
└──────────────────────────────────────────────────────────┘
```

## Hard-Braking Transferability

The core innovation is the **explicit hard-braking transferability analysis** (Step 3), which was flagged as not being properly written in the paper. This analysis:

- Applies the **HB-spec** (hard-braking's DE/MCR-selected geometry) to the **crash response** and measures the ΔBIC penalty
- Applies the **Crash-spec** (crash's DE/MCR-selected geometry) to the **HB response** and measures the ΔBIC penalty
- Reports a **symmetric transfer penalty**
- Issues a **verdict**: TRANSFERABLE / MODERATELY TRANSFERABLE / NOT TRANSFERABLE

This is distinct from the encompassing test (Step 4) which tests whether predicted HB encompasses geometry for crashes.

## Files

| File | Description |
|------|-------------|
| `hb_vs_crash_prediction_search.py` | Main analysis script (all 6 steps) |
| `run_hb_search.py` | Standalone runner for PBS |
| `hb_search_runner.pbs` | PBS job submission script |
| `README.md` | This file |

## Usage

### Interactive
```bash
cd paper/Unofficial_Transportation_Research_Board__TRB__LaTeX_template/hb_search/
python run_hb_search.py --out-dir ../
```

### Resume from checkpoints
```bash
python run_hb_search.py --out-dir ../ --resume
```

### DE only (skip MCR structure search)
```bash
python run_hb_search.py --out-dir ../ --de-only
```

### PBS job
```bash
qsub hb_search_runner.pbs
# or with custom parameters:
qsub -v MCR_MAXITER=120 MCR_POPSIZE=30 MCR_N_STARTS=3 hb_search_runner.pbs
```

## Output

All outputs are written to the paper template directory:

| File | Description |
|------|-------------|
| `data_cleaned.csv` | Cleaned dataset |
| `de_search_hb.json` | DE results for hard-braking |
| `de_search_crash.json` | DE results for crashes |
| `mcr_search_hb.json` | MCR structure search results for HB |
| `mcr_search_crash.json` | MCR structure search results for crashes |
| `spec_transfer.csv` | Specification transfer (hard-braking transferability) |
| `transferability_summary.csv` | Combined transferability verdict |
| `copula_comparison.csv` | Frank / Normal / Kimeldorf copula comparison |
| `de_convergence.png` | DE convergence trajectories |
| `spec_transfer.png` | Specification transfer bar chart |
| `copula_comparison.png` | Copula BIC comparison |
| `run_metadata.json` | Full run metadata |
| `checkpoints/` | Checkpoint files for resume |

## metacountregressor Components Used

| Component | Module | Purpose |
|-----------|--------|---------|
| `StructureEvaluatorLC` | `experiment_package` | Structure search over roles {0,1,2,3} |
| `AdaptiveDE` | `Solvers_METAJAX` | Differential evolution optimizer |
| `NSGA2Engine` | `Solvers_METAJAX` | Multi-objective DE engine |
| `AdvancedSimulatedAnnealing` | `Solvers_METAJAX` | SA optimizer |
| `build_model_from_manual_spec` | `main_hpc_lc_patch` | Build NB model from spec |
| `mixed_model_loglik` | `main_hpc_lc_patch` | NB log-likelihood |
| `build_param_index` | `main_hpc_lc_patch` | Parameter index builder |
| `compute_standard_errors` | `main_hpc` | JAX-based SE computation |

## Data Requirements

The script expects `near_miss_hb_ecf.csv` in the paper template directory with columns:
- 30 geometry/traffic candidate variables
- `nm_total` (hard-braking total count)
- `ObsCrashes` (observed crash count)
- `segment_length_m` (segment length in metres)
- `ExpCrashes` (expected crash frequency)