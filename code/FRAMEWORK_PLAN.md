# Activity-Based Model: Core-Model Framework Plan
## Stages 0–4 · SearchLibrium + MetaCountRegressor + Optional Packages
---

## Overview

This document outlines the planned attack for building and validating an
Activity-Based Model (ABM) pipeline using **core (non-searched) models** from
SearchLibrium and MetaCountRegressor.  The pipeline is **HHTS-validation-only**
(no synthetic population prediction), runs on **CPUs exclusively**, and provides
a side-by-side BIC-ranked comparison of every admissible model structure at each
stage.

```
HHTS Survey Data
     │
     ▼
Stage 0 — Data Loading & Preprocessing
     │
     ▼
Stage 1 — Activity Frequency (count models)
     │         └─ NB2 · ZIP · ZINB · Latent-Class NB  [MCR]
     │         └─ Ordered Logit · Ordered Probit       [SL]
     ▼
Stage 2 — Day Duration Budget (censored / selection models)
     │         └─ Tobit I · Lognormal · Mixed Lognormal  [MCR]
     │         └─ Heckman (Probit + OLS+IMR)             [statsmodels]
     ▼
Stage 3 — Activity Duration (survival / AFT models)
     │         └─ Mixed Lognormal                        [MCR]
     │         └─ AFT Lognormal · Weibull · LogLogistic  [lifelines]
     ▼
Stage 4 — Schedule Placement (discrete-choice models)
           └─ MNL · Nested Logit                         [SearchLibrium]
           └─ MNL cross-check                            [pylogit]
           └─ (optional) Mixed Logit · RRM · LC-MXL     [SearchLibrium]
     │
     ▼
Validation Report (HHTS in-sample only — no population)
```

---

## CPU-Only Setup

All packages are forced to the CPU backend **before any import**:

```python
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"   # MetaCountRegressor / JAX
os.environ["JAX_PLATFORMS"]     = "cpu"   # XLA platform selection
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide any GPU from CUDA drivers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress XLA/LLVM noise
```

SearchLibrium is **NumPy-native by default** — models instantiated with
`_jax=False` (the default) never touch JAX at all.

MetaCountRegressor requires JAX but honours `JAX_PLATFORM_NAME=cpu`, which
routes all XLA computation to the CPU device.  On Windows / HPC, setting this
environment variable **before** importing `jax` prevents the `hwloc`/XLA
GPU-binding errors documented in the project notes.

---

## Package Sources

| Package | Local editable source | PyPI name |
|---|---|---|
| SearchLibrium | `C:/Users/ahernz/source/SearchLibrium` | `SearchLibrium` |
| MetaCountRegressor | `C:/Users/ahernz/source/MetaCount/metacountregressor` | `metacountregressor` |
| lifelines | PyPI | `lifelines` |
| statsmodels | PyPI | `statsmodels` |
| pylogit | PyPI | `pylogit` |
| biogeme | PyPI | `biogeme` |
| PyMC | PyPI | `pymc` |

---

## Stage 0 — Data Loading

**Purpose:** Load and pre-process the HHTS survey.  Build tour-level, trip-level, and person-level tables.

**Models:** None (data engineering only).

**Key operations:**
- `Synth_Connect().process_survey(save=False)` → `{tours, trips, persons}`
- `add_total_activity_counts(tours)` — combine outbound + inbound activity counts
- `assign_ptype_column(tours)` — rule-based person-type assignment

**Validation contract:** No synthetic population loaded.  All downstream
estimation and validation operates on the HHTS records only.

---

## Stage 1 — Activity Frequency

**Question:** How many activities of each purpose does a person make per day?

**Outcome:** Non-negative integer count `n_{purpose}` ∈ {0, 1, 2, 3, …}

**Estimation units:** One model per *(purpose × ptype)* cell (~56 cells).

---

### Model catalogue

| # | Model | Package | Class / function | Notes |
|---|---|---|---|---|
| 1 | **Negative Binomial 2 (NB2)** | MetaCountRegressor | `ExperimentBuilder.fit_manual_model(model="nb", dispersion=1)` | Primary. Handles overdispersion (var > mean). Most appropriate for daily trip counts |
| 2 | **Poisson** | MetaCountRegressor | `fit_manual_model(model="poisson", dispersion=0)` | Nested in NB2; use if dispersion parameter not significant |
| 3 | **Zero-Inflated Poisson (ZIP)** | MetaCountRegressor | `fit_manual_model(model="poisson")` + `zi_terms=[...]` | When zero-activity rate is high (e.g., non-workers for 'work' purpose) |
| 4 | **Zero-Inflated NB (ZINB)** | MetaCountRegressor | NB + `zi_terms` | Combines overdispersion + structural zeros |
| 5 | **Latent-Class NB** | MetaCountRegressor | `make_manual_spec(latent_classes=2)` | Identifies latent traveller segments with different base rates |
| 6 | **Ordered Logit** | SearchLibrium | `OrderedLogit(_jax=False, X=..., y=..., J=...)` | Treats count as ordinal; proportional-odds assumption. Secondary cross-check |
| 7 | **Ordered Probit** | SearchLibrium | `OrderedLogit(_jax=False, distr="probit", ...)` | Probit link function alternative to logit |

**Selection criterion:** BIC (lower is better).

**Representative decision:** NB2 vs Ordered Logit comparison illuminates whether
treating counts as *ordinal categories* vs *stochastic rate outcomes* yields
meaningfully different fit and predictions.

---

## Stage 2 — Day Duration Budget

**Question:** When does a person start/end out-of-home activities, and what is their total out-of-home time budget *B* (hours)?

**Outcomes:**
- `start_of_day` — minutes from midnight (left-censored at 240 = 4 am)
- `end_of_day`   — minutes from midnight (right-censored at 1440 = midnight)
- `B`            — total out-of-home budget hours (strictly positive for travellers; 0 for non-travellers)

---

### Model catalogue

| # | Model | Package | Notes |
|---|---|---|---|
| 1 | **Tobit I (left)** | MetaCountRegressor | `model="tobit"` for `start_of_day`. Handles observations at the lower bound |
| 2 | **Tobit I (right)** | MetaCountRegressor | `model="tobit"` for `end_of_day`. Handles observations at the upper bound |
| 3 | **Lognormal regression** | MetaCountRegressor | `model="lognormal"` for budget *B*. Strictly-positive support; right-skewed |
| 4 | **Mixed Lognormal** | MetaCountRegressor | `rdm_terms=["income_per_person"]` adds random heterogeneity in income response |
| 5 | **Heckman (Heckit)** | statsmodels | Two-step: Probit participation + OLS+IMR budget. Jointly models *decision to travel* and *how long*. Correct for sample-selection bias |
| 6 | **Gaussian** | MetaCountRegressor | `model="gaussian"`. Simplest baseline; symmetric residuals. Use as lower-bound benchmark |

**Identification note (Heckman):** The selection equation requires at least one variable not in the outcome equation (exclusion restriction).  Here `n_mandatory` and `n_discretionary` serve as instruments — they determine travel participation but are less directly causal for *how long* one travels conditional on travelling.

---

## Stage 3 — Activity Duration

**Question:** How long is each individual activity (minutes)?

**Outcome:** `duration_min` > 0, per activity record.

**Estimation units:** One model per *purpose* (8 models).

---

### Model catalogue

| # | Model | Package | Family | Notes |
|---|---|---|---|---|
| 1 | **Mixed Lognormal** | MetaCountRegressor | `model="lognormal"`, `rdm_terms=["B"]` | Primary. Random parameter on budget captures individual-level heterogeneity in duration–budget relationship |
| 2 | **AFT Lognormal** | lifelines | `LogNormalAFTFitter` | Log-linear hazard; equivalent to lognormal regression with censoring support |
| 3 | **AFT Weibull** | lifelines | `WeibullAFTFitter` | Proportional hazard on log-time; monotone hazard shape. Appropriate for activities with increasing or decreasing termination rates |
| 4 | **AFT Log-Logistic** | lifelines | `LogLogisticAFTFitter` | Non-monotone (unimodal) hazard. Appropriate for activities with a peak termination rate (e.g. meals, social activities with typical end-times) |
| 5 | **AFT Gamma** | lifelines | `CoxPHFitter` (parametric) | Flexible two-parameter family; nests exponential and approaches normal for large shape |
| 6 | **Lognormal (fixed)** | MetaCountRegressor | `model="lognormal"`, no `rdm_terms` | Baseline with no random parameters for BIC comparison |

**Budget constraint:** After fitting, predicted durations are rescaled so
$\sum_k \hat{d}_k \leq B$ per person (budget feasibility).  This rescaling step
is independent of which model family is selected.

---

## Stage 4 — Schedule Placement (Time-of-Day Choice)

**Question:** In which time period does each activity occur?

**Time periods:**

| Period | Clock range |
|---|---|
| `early` | 04:00–07:00 |
| `am_peak` | 07:00–09:00 |
| `midday` | 09:00–14:00 |
| `pm_peak` | 14:00–17:00 |
| `evening` | 17:00–22:00 |
| `late` | 22:00–00:00 |

**Estimation units:** One model per *purpose* (8 models).

---

### Model catalogue

| # | Model | Package | Class / API | Notes |
|---|---|---|---|---|
| 1 | **Multinomial Logit (MNL)** | **Larch** | `lx.Model(dt, compute_engine='numba')` + `utility_co[j]` | **Primary.** IDCO format; exact gradient via numba JIT. No JAX/XLA required. IIA-constrained; 6 time-period alternatives |
| 2 | **Nested Logit** | **Larch** | `lx.Model` + `m.graph.new_node(parameter="Mu_*", children=[...])` | **Secondary.** Temporal nests `{off_peak=[early,late], peak=[am/pm_peak], shoulder=[midday,evening]}`. Larch NestingTree estimates logsum μ per nest. Faster than SL on large cells |
| 3 | **Mixed Logit (MXL)** | **Larch** | `lx.Model` + `m.mixtures[p] = lx.mixtures.Normal(...)` | **Tertiary.** Random params on time-of-day covariates. Requires JAX engine. Halton quasi-random draws (N=200) |
| 4 | **MNL (cross-check)** | SearchLibrium | `MultinomialLogit(_jax=False)` | Independent BFGS cross-validation of Larch MNL log-likelihood |
| 5 | **Nested Logit (cross-check)** | SearchLibrium | `NestedLogit(_jax=False)` | Temporal nests; same structure as Larch NL; verifies nest logsum parameters |
| 6 | **Random Regret MNL (RRM)** | SearchLibrium | `RandomRegret()` | Regret-based utility: $R_{ni} = \sum_{j \neq i} \sum_m \log(1 + e^{\beta_m(x_{njm}-x_{nim})})$. Alternative to utility maximisation |
| 7 | **Latent Class MXL** | SearchLibrium | `LatentClassMixedLogit(n_classes=2)` | EM algorithm. Identifies latent segments with different time-of-day preferences |
| 8 | **MNL (pylogit)** | pylogit | `create_choice_model(model_type="MNL")` | L-BFGS-B independent cross-validation |
| 9 | **MNL (biogeme)** | biogeme | `BIOGEME` + `MNL()` | Swissrail-style standard errors; further independent cross-check |

**Larch engine selection:**  `compute_engine='numba'` is the default on Windows
(no JAX/XLA; avoids hwloc binding errors).  Override with `LARCH_ENGINE=jax` to
use JAX autodiff (needed for Larch MXL and LatentClass).

**Nesting rationale:** The temporal nest structure groups periods by congestion
context (`off_peak`, `peak`, `shoulder`).  If the inclusive value $\lambda$ for
the `peak` nest is significantly less than 1, early-morning and late-afternoon
slots are indeed substitutes (reasonable), validating the nest.

---

## Larch ↔ SearchLibrium / MetaCountRegressor GA Bridge

### Motivation

SearchLibrium and MetaCountRegressor both include meta-heuristic search engines
(Simulated Annealing, Harmony Search, Differential Evolution) that traditionally
operate over the *model structure* space — which variables to include, which to
make random, which latent classes to create.

Larch provides **exact MLE** for discrete-choice models with analytically correct
gradients via numba or JAX autodiff.  The `LarchSABridge` class in `main_2.py`
connects these two capabilities: **Larch estimates coefficients, SA searches the
variable set.**

### Architecture

```
SA/HS search engine           Larch estimator
(SearchLibrium / MCR)         (larch5 MNL or NL)
       │                              │
  candidate variable mask ──────────► _larch_mnl_fit(active_vars)
       │                              │
  BIC ◄──────────────────────── m.loglike()  →  BIC = k·ln(n) − 2·ℓ̂
       │                              │
  accept / reject (Metropolis)        │
  toggle next variable               │
       └──────────────────────────────┘
```

### Fitness function

$$\mathcal{F}(\mathbf{s}) = \text{BIC}\!\left(\hat{\boldsymbol{\theta}}_{\text{Larch}}(\mathbf{s})\right)$$

where $\mathbf{s} \in \{0,1\}^K$ is the binary variable-selection vector
and $\hat{\boldsymbol{\theta}}_{\text{Larch}}(\mathbf{s})$ is the Larch MLE
given the active variables $\{x_k : s_k = 1\}$.

### Integration APIs

| API | How to use |
|---|---|
| `LarchSABridge.fitness(mask)` | Call directly from any SA loop |
| `LarchSABridge.run_sl_search(n_iter, temperature)` | Built-in SA using Metropolis–Hastings |
| `LarchSABridge.as_mcr_evaluator()` | Returns object with `.fitness(decision)` for MCR `run(evaluator, algo='sa')` |
| SearchLibrium `call_siman(params, fitness=bridge.fitness)` | Pass bridge as custom fitness to SL's SA engine |

### Why not use SL/MCR's internal model classes?

Larch's advantage over SearchLibrium's MNL for Stage 4 is:
- **DataTree format** handles unbalanced availability (not all alternatives available for every observation) — important for time-of-day models where some periods may be structurally unavailable
- **NestingTree** automatically ensures utility-maximisation consistency conditions (μ_k ∈ (0,1])
- **Exact gradients** via autodiff eliminate finite-difference noise in the BIC surface, giving SA cleaner signals

SearchLibrium and MCR remain the primary tools for Stages 1–3 (count, duration, Tobit) where Larch has no equivalent model families.

---

## Cross-Cutting: Model Selection Logic

At each stage, every fitted model is compared by **BIC**:

$$\text{BIC} = k \ln n - 2\hat{\ell}$$

where $k$ = number of estimated parameters, $n$ = sample size,
$\hat{\ell}$ = maximised log-likelihood.

The **best** model per cell is recorded in the validation CSV and HTML report.
A model is preferred over the baseline MNL/OL if:

$$\Delta\text{BIC} = \text{BIC}_{\text{complex}} - \text{BIC}_{\text{base}} < -10$$

(decisive evidence by Kass–Raftery scale).

---

## Optional / Future Integratable Packages

Beyond SearchLibrium, MetaCountRegressor and Larch, the following packages can be
seamlessly added to the pipeline:

### Larch (larch5) — already integrated
- **Stages:** 4 (time-of-day choice), 5 (destination choice), 6 (mode choice)
- **Install:** `pip install larch5`
- **Key classes:** `lx.Model`, `lx.LatentClass`, `lx.MixedLatentClass`
- **Windows note:** Always use `compute_engine='numba'` to avoid JAX/XLA binding errors

### lifelines (survival analysis)
- **Purpose:** Stages 3 (duration models)
- **Install:** `pip install lifelines`
- **Key classes:** `WeibullAFTFitter`, `LogNormalAFTFitter`, `LogLogisticAFTFitter`
- **Integration effort:** Low — `fit(df, duration_col, event_col)` interface

### statsmodels
- **Purpose:** Stage 2 Heckman selection, OLS benchmarks, Probit
- **Install:** `pip install statsmodels`
- **Key classes:** `Probit`, `OLS`, `Logit`
- **Integration effort:** Low — standard Python ML interface

### pylogit
- **Purpose:** Stage 4 MNL/Nested independent cross-validation
- **Install:** `pip install pylogit`
- **Integration effort:** Medium — requires long-format data prep
- **Value:** Independent optimisation backend to verify Larch / SearchLibrium coefficients

### biogeme
- **Purpose:** Stage 4 MNL / Cross-Nested / Mixed (Bregman/ML estimation)
- **Install:** `pip install biogeme`
- **Integration effort:** Medium — requires biogeme-specific model specification
- **Value:** Industry-standard tool for transport demand modelling; Swissrail-style error structure

### PyMC (Bayesian inference)
- **Purpose:** Any stage — Bayesian posteriors for coefficient uncertainty
- **Install:** `pip install pymc`
- **Integration effort:** High — requires probabilistic model specification
- **Value:** Full posterior distributions; natural hierarchical/multilevel structures
  for person-type heterogeneity; MCMC diagnostics (R-hat, ESS)

### XGBoost / LightGBM (gradient boosting)
- **Purpose:** Non-parametric benchmarks for Stages 1 and 3
- **Install:** `pip install xgboost lightgbm`
- **Integration effort:** Low — sklearn-compatible API
- **Value:** BIC-equivalent criterion: adjusted RMSE. Useful as an upper-bound
  on predictive accuracy to detect if parametric models are under-fitting

---

## Validation-Only Contract

The pipeline in `main_2.py` enforces three invariants:

1. **No synthetic population loaded** — `build_synth_population` is never called
2. **No prediction push** — `predict_activity_frequency`, `predict_day_budget`, etc.
   are never called on population records
3. **In-sample metrics only** — all RMSE, BIC, log-likelihood values are computed
   on the same HHTS data used for estimation

This design allows rapid model comparison without the computational overhead of
synthetic population generation (~329 zone files × 1,000+ persons/zone).

---

## Output Structure

```
runs/main2__{timestamp}/
  main2.log                   ← pipeline execution log
  run_manifest.json           ← environment metadata
  stage1_core_models.pkl      ← all fitted Stage 1 models
  stage1_validation.csv       ← BIC / AIC / LL per (purpose×ptype×model)
  stage2_core_models.pkl
  stage2_validation.csv
  stage3_core_models.pkl
  stage3_validation.csv
  stage4_core_models.pkl
  stage4_validation.csv
  stage4_sa_history.csv       ← LarchSABridge SA variable-selection trace
  all_stages_validation.csv   ← merged summary
  validation_report.html      ← self-contained HTML report
  timings.csv                 ← elapsed time per stage
```
