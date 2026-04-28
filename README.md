# test_runs_tours — SEQ Activity-Based Travel Model

A sequential activity-based model (ABM) pipeline for South-East Queensland (SEQ), Australia.
The pipeline generates synthetic daily activity plans (destination, mode, timing) for the SEQ
population using the QTS Household Travel Survey (HHTS) and ABS Census data as inputs.

---

## Repository Structure

```
test_runs_tours/
├── code/                        # Python pipeline source (all stages)
│   ├── main_zim.py              # Master pipeline entry point
│   ├── data_connect.py          # HHTS data loading and processing
│   ├── build_synth_population.py
│   ├── cdap.py                  # Stage 0: Coordinated Daily Activity Pattern
│   ├── stage1_activity_frequency.py
│   ├── stage2_day_duration.py
│   ├── stage3_activity_duration.py
│   ├── stage4_schedule_placement.py
│   ├── stage5_destination_choice.py  # Tour-level destination + mode (Larch NL)
│   ├── stage6_mode_choice.py         # Trip-level mode choice (Larch NL)
│   ├── paper_model_forms.py          # Paper-section HTML/LaTeX output
│   ├── pipeline_logger.py
│   └── saved_data/              # Model outputs (gitignored)
├── SA1_SHP/                     # ABS SA1 shapefiles (gitignored — download separately)
├── SA2_2021_AUST_SHP_GDA2020/   # ABS SA2 shapefiles (gitignored — download separately)
└── README.md
```

---

## Prerequisites

### 1. Download Shapefiles

Download the ABS digital boundary files from:
> https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files

You need two folders placed at the **root** of this repository (next to `code/`):
- `SA1_SHP/` — containing `SA1_2021_AUST_GDA2020.shp` (and companion files)
- `SA2_2021_AUST_SHP_GDA2020/` — containing the SA2 2021 boundary shapefile

### 2. Python Environment

Python 3.10+ is required. Create and activate a virtual environment:

```powershell
cd Z:\test_runs_tours\code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r installed_packages.txt
```

> On PBS/HPC clusters use the provided job scripts (`job_main.pbs`, etc.) which
> activate the environment automatically.

---

## Running the Full Pipeline

```powershell
cd Z:\test_runs_tours\code
python main_zim.py
```

This runs all stages in sequence and writes outputs to `code/saved_data/`.

---

## Running Individual Stages

Each stage can be run standalone for testing or re-estimation:

| Stage | File | What it does |
|-------|------|--------------|
| Synth pop | `build_synth_population.py` | Build synthetic population from ABS Census |
| CDAP | `cdap.py` | Assign daily activity pattern (Home / Non-mandatory / Mandatory) |
| Stage 1 | `stage1_activity_frequency.py` | Activity frequency per purpose |
| Stage 2 | `stage2_day_duration.py` | Daily out-of-home time budget |
| Stage 3 | `stage3_activity_duration.py` | Duration of each activity |
| Stage 4 | `stage4_schedule_placement.py` | Time-of-day placement |
| Stage 5 | `stage5_destination_choice.py` | Tour-level destination + mode (nested logit) |
| Stage 6 | `stage6_mode_choice.py` | Trip-level mode choice (nested logit per purpose) |

```powershell
python stage5_destination_choice.py
python stage6_mode_choice.py
```

---

## Outputs

All outputs are written to `code/saved_data/` (gitignored locally):

| File | Description |
|------|-------------|
| `stage5_models_coefficients.html` | Tour-level mode + destination model tables |
| `stage6_models_coefficients.html` | Trip-level mode model tables (all purposes) |
| `stage6_models_coefficients.csv` | Coefficient CSV with SE, t-stat, p-value |
| `stage6_output.csv` | Final synthetic trip list |
| `paper/sec3_tour_destination_and_mode_choice_model.html` | Paper Section 3 HTML |
| `paper/sec4_trip_level_mode_choice_model.html` | Paper Section 4 HTML |
| `paper/sec5_safety_benefits.html` | Safety deterrence summary for paper |

---

## Running on a PBS/HPC Cluster

Submit the full pipeline with:

```bash
qsub job_main.pbs
```

Stage-specific jobs:

```bash
qsub job_ga.pbs          # Genetic algorithm model selection
qsub job_ga_staged.pbs   # Staged GA run
```

Logs are written to `saw_main.eXXXXXX` / `saw_main.oXXXXXX` (gitignored).

---

## Safety Model

The pipeline embeds crash risk (OD-level police-recorded crash counts) directly into
the mode choice utility functions at both tour level (Stage 5) and trip level (Stage 6).
A negative `crash` coefficient indicates that travellers are less likely to drive on
high-crash OD corridors — providing demand-side evidence of safety-aware travel behaviour.

See `code/paper_model_forms.py → write_safety_benefits_html()` for the full safety output.

---

## Data Sources

- **QTS HHTS** — Queensland Travel Survey household/person/trip records (included as `.xlsx`)
- **ABS Census 2021** — SA1/SA2 population and dwelling data
- **OpenStreetMap** — Land use categories via `SEQ_SA2_2016_OSM_*.csv`
- **OD Crash matrix** — `OD_Crashes.csv` (police-recorded crashes aggregated to SA2 OD pairs)
- **Skim matrices** — `downcasted_matrices_base.omx` (travel time / distance by mode, gitignored)
