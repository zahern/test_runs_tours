# PYRAM: Travel Demand Modeling Pipeline

[![PyPI](https://img.shields.io/pypi/v/pyram.svg)](https://pypi.org/project/pyram/)
[![Python](https://img.shields.io/pypi/pyversions/pyram.svg)](https://pypi.org/project/pyram/)
[![License](https://img.shields.io/pypi/l/pyram.svg)](https://github.com/pyram/pyram/blob/main/LICENSE)

**PYRAM** (Python-based Activity-based Regional Model) is a complete travel demand modeling pipeline for activity-based transport simulation, inspired by POLARIS and ActivitySim.

## Features

- **Synthetic Population Generation**: Expand HHTS survey to full population with Overture Maps integration
- **Household Coordination (CDAP)**: Coordinated daily activity patterns with joint tours
- **Destination Choice**: MNL and Nested Logit with distance-band nests
- **Mode Choice**: Nested Logit, Mixed Logit (MXL), MXL with Gradient Score Enhancement (GSE)
- **Trip Chain Building**: Complete trip chains with tour-based structure
- **Parking Choice**: On-street, garage, and park-and-ride with walk access legs
- **DTA Feedback Loop**: Iterate demand model ↔ DTA (MATSim/SUMO/DynusT) for equilibrated skims
- **Microsimulation Export**: MATSim and SUMO compatible outputs

## Installation

```bash
pip install pyram
```

With GPU support (JAX):
```bash
pip install pyram[gpu]
```

Development install:
```bash
git clone https://github.com/pyram/pyram
cd pyram
pip install -e .[dev]
```

## Quick Start

### 1. Create Configuration

```bash
pyram init -o config.yaml
```

Edit `config.yaml` to set your data paths:

```yaml
data:
  hhts_trips: "data/hhts_trips.csv"
  hhts_persons: "data/hhts_persons.csv"
  hhts_households: "data/hhts_households.csv"
  skims_omx: "data/skims.omx"
  landuse_csv: "data/landuse.csv"
  zone_crosswalk: "data/zone_crosswalk.csv"
  location_table: "data/location_table.csv"
  output_dir: "output"

population:
  use_overture: true
  target_region: "QLD"

destination:
  model_type: "nl_distance"  # mnl, nl_distance, nl_sa3

mode:
  model_type: "mxl"  # nl, mxl, mxl_gse

parking:
  enable: true

dta:
  enable: true
  engine: "matsim"
  iterations: 3
```

### 2. Validate Configuration

```bash
pyram validate -c config.yaml
```

### 3. Generate HPC Environment Variables

```bash
pyram env -c config.yaml -o .env
# Then source .env before qsub
```

### 4. Run Pipeline

```bash
# Full pipeline
pyram run -c config.yaml

# Specific stages only
pyram run -c config.yaml -s 3 4 5

# Dry run to see execution plan
pyram run -c config.yaml --dry-run
```

### 5. Python API

```python
from pyram import Pipeline, Config

config = Config.from_yaml("config.yaml")
pipeline = Pipeline(config)
result = pipeline.run()

# Or run individual stages
from pyram.pipeline import run_stage3_destination
result = run_stage3_destination(config, data)
```

## Pipeline Stages

| Stage | Description | Key Models |
|-------|-------------|------------|
| 0 | Data Loading | HHTS, Skims, Landuse, Overture |
| 1 | CDAP | Household coordination, joint tours |
| 2 | Timing | Departure time, duration (RRM/TDD) |
| 3 | Destination | MNL / NL Distance Bands / NL SA3 |
| 4 | Mode | NL / MXL / MXL-GSE |
| 5 | Trip Chains | Tour-based chain assembly |
| 6 | Parking | On-street / Garage / P&R |
| 7 | DTA Feedback | MATSim / SUMO / DynusT iteration |

## Configuration

Configuration is via YAML with environment variable overrides (`PYRAM_` prefix):

```bash
PYRAM_DESTINATION__MODE_TYPE=nl_distance
PYRAM_MODE__MIXED_LOGIT=true
PYRAM_DTA__ENABLE=true
PYRAM_HPC__STAGE_NCPUS=16
```

## HPC / PBS Submission

The `pyram env` command generates environment variables for PBS:

```bash
# Generate PBS submission variables
pyram env -c config.yaml -o submit.env

# Submit with qsub
qsub -v "$(cat submit.env | tr '\n' ',')" job_submit_pipeline.pbs
```

## Model Types

### Destination Choice
- `mnl`: Multinomial Logit (baseline)
- `nl_distance`: Nested Logit with distance bands (0-2km, 2-5km, 5-10km, 10km+)
- `nl_sa3`: Nested Logit with SA3 geographic nests

### Mode Choice
- `nl`: Nested Logit (Motorized / Non-motorized nests)
- `mxl`: Mixed Logit with random coefficients
- `mxl_gse`: MXL with Gradient Score Enhancement (systematic γ·g loadings)

## Data Requirements

### Required Input Files
- `hhts_trips.csv`: Trip records with OTAZ, DTAZ, purpose, mode, times
- `hhts_persons.csv`: Person attributes (age, sex, income, license, etc.)
- `hhts_households.csv`: Household attributes (size, vehicles, income)
- `skims.omx`: Zone-to-zone skim matrices (OpenMatrix format)
- `landuse.csv`: Zone-level land use (employment by type, population, etc.)
- `zone_crosswalk.csv`: SA2 ↔ ORDER zone ID mapping
- `location_table.csv`: Parking facilities (from `pyram-build-location-table`)

### Overture Maps (Optional)
For synthetic population control totals outside survey region:
```bash
pyram-build-overture --bbox "151,-28,153,-26" --output data/overture.parquet
```

## Extending

### Custom Models
```python
from pyram.models import BaseChoiceModel

class MyCustomModel(BaseChoiceModel):
    def estimate(self, data, **kwargs):
        # Your estimation logic
        pass
    
    def predict(self, data, **kwargs):
        # Your prediction logic
        pass
```

### Custom Pipeline Stages
```python
from pyram.pipeline import Pipeline

class MyPipeline(Pipeline):
    def _run_custom_stage(self):
        # Your custom logic
        pass
```

## Citation

If you use PYRAM in research, please cite:

```bibtex
@software{pyram,
  title = {PYRAM: Travel Demand Modeling Pipeline},
  author = {PYRAM Team},
  year = {2024},
  url = {https://github.com/pyram/pyram}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by POLARIS (ANL) and ActivitySim frameworks
- Uses Larch for discrete choice estimation
- Overture Maps for global POI/building data
- MATSim/SUMO/DynusT for DTA integration