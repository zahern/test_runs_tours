# Event-Driven Activity-Based Travel Simulation

A modular, event-driven simulation framework for activity-based travel demand modeling.
Integrates with existing estimation models (Stages 0-6) and supports GPU/CPU acceleration.

## Quick Start

```python
from simulation.runner import SimulationRunner

# Create and configure simulation
runner = SimulationRunner({
    'data': {
        'landuse_path': 'path/to/landuse.csv',
        'skim_path': 'path/to/skims.omx',
        'population_path': 'path/to/population.csv'
    },
    'models': {
        'dir': 'path/to/models'
    }
})

runner.setup()
runner.load_population('path/to/population.csv')
results = runner.run(max_time=24*60)  # 24 hours
```

## Architecture Overview

```
simulation/
├── events/           # Core event system
│   └── core.py       # Event, EventQueue, Scheduler, Dispatcher
├── agents/           # Agent representations
│   └── agent.py      # Agent, PersonAttributes, Activity, Trip, Plan
├── handlers/         # Stage event handlers
│   └── stage_handlers.py  # CDAP, Frequency, Duration, Timing, Mode/Dest, Trip
├── models/
│   └── integration.py       # Model integration layer
├── data/             # Data access layer
│   └── access.py     # Land use, skims, networks, population
├── performance/      # Performance optimizations
│   └── optimization.py
├── validation/       # Debugging and validation
│   └── debugger.py
└── runner.py         # High-level runner
```

## Key Features

### Event-Driven Architecture
- **Event Queue**: Priority-based event queue with time ordering
- **Event Dispatcher**: Routes events to registered handlers with priority
- **Event Types**: 20+ event types covering all modeling stages
- **Priorities**: CRITICAL, HIGH, NORMAL, LOW, DEFERRED

### Agent-Based Modeling
- **Agents**: Persons with attributes, plans, and state machines
- **Plans**: Activities + Trips with full temporal structure
- **Activities**: 11 types (HOME, WORK, EDUCATION, SHOPPING, etc.)
- **Modes**: DRIVE_ALONE, SHARED_RIDE, TRANSIT, WALK, BIKE
- **Plans**: Status tracking, scoring, tour extraction

### Model Integration (Stages 0-6)
- **Stage 0**: CDAP (Coordinated Daily Activity Pattern)
- **Stage 1**: Activity Frequency (8 purposes)
- **Stage 2**: Activity Duration (AFT Weibull)
- **Stage 3**: Schedule Timing (AFT Weibull for day bounds)
- **Stage 4**: Schedule Placement
- **Stage 5**: Joint Mode + Destination Choice (Nested Logit)
- **Stage 6**: Plan Finalization & Trip Execution

### GPU/CPU Auto-Detection
```bash
# Automatic GPU detection
if GPU available:
    JAX_PLATFORMS=gpu
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
else:
    JAX_PLATFORMS=cpu  # Suppresses CUDA warnings
```

### Performance Optimizations
- **Vectorized model application**: Batch process agents
- **Model caching**: LRU cache for repeated predictions
- **Parallel processing**: ThreadPoolExecutor for event batches
- **Batch size**: Configurable (default 1000)

### Validation & Debugging
- **ModelValidator**: Frequency, duration, choice consistency checks
- **SimulationDebugger**: Breakpoints, watch expressions, step-through
- **ValidationReporter**: JSON/text reports with severity levels
- **SimulationAuditor**: Conservation, time monotonicity, queue checks
- **ModelInspector**: Feature importance, coefficients, marginal effects

## Running Simulations

### Exampville Benchmark
```bash
cd simulation
python -u exampville_compare.py --outdir runs/exampville_ab
```
Or via PBS:
```bash
qsub -v "CONDA_ENV=exampville,GA_ENGINE=agds" ./job_exampville_ab.pbs
```

### QLD 371-Zone Pipeline
```bash
# Stage 1-5 GA search
qsub -v "CONDA_ENV=zigenv,GA_ENGINE=agds,GA_MAX_WORKERS=1,GA_WALLTIME_SECONDS=13800,STAGE5_ESTIMATE_MU=1,STAGE_ID=1" job_ga_stage_single.pbs

# Full pipeline after GA completes
export NEW_POP=1; export NEW_POP_MAPPED=0; export MODE=ga_staged; export GA_ESTIMATOR=sa_bandit
python run_full_pipeline_dta_tested.py
```

### Run Directly (No PBS)
```python
from simulation.runner import SimulationRunner

runner = SimulationRunner({
    'data': {
        'landuse_path': '../new_pop/saved_data/zone_landuse_overture.csv',
        'skim_path': '../demand_/skims_371.omx',
        'population_path': '../new_pop/saved_data/persons.csv'
    },
    'models': {'dir': 'runs/ga_parallel/run06/merged'}
})

runner.setup()
runner.load_population('../new_pop/saved_data/persons.csv')
results = runner.run(max_time=24*60, debug=True)
```

## Key Files

| File | Purpose |
|------|---------|
| `events/core.py` | Core event system |
| `agents/agent.py` | Agent, Plan, Activity, Trip |
| `handlers/stage_handlers.py` | Stage 0-6 event handlers |
| `models/integration.py` | Model integration layer |
| `data/access.py` | Land use, skims, networks |
| `performance/optimization.py` | Vectorization, caching, parallel |
| `validation/debugger.py` | Debugging, validation, auditing |
| `runner.py` | High-level runner |

## Configuration

Environment variables:
- `CONDA_ENV`: Conda environment name (default: `zigenv`)
- `GA_ENGINE`: Search engine (`agds`, `ga`, `sa`, `portfolio`)
- `GA_MAX_WORKERS`: Parallel workers (default: 1)
- `GA_WALLTIME_SECONDS`: Wall time limit (seconds)
- `STAGE5_ESTIMATE_MU`: Estimate nest scales (0/1)
- `ZONE_SYSTEM_N_ZONES`: Zone system size (40 or 371)

## Model Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| larch | >=6.0 | Choice modeling |
| searchlibrium | >=0.0.150 | GA search |
| metacountregressor | >=1.0.150 | Count regression |
| openmatrix | - | OMX skim reading |
| jax | - | GPU acceleration (optional) |

## Performance Notes

- **Event processing**: ~10,000-50,000 events/sec on CPU
- **Memory**: ~1-2 GB for 100k agents
- **Batch size**: 1000 agents for vectorized models
- **GPU**: Use `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` for GPU runs

## Debugging

```python
from simulation.validation.debugger import SimulationDebugger

debugger = SimulationDebugger(scheduler)
debugger.add_breakpoint(EventType.MODE_CHOICE)
debugger.add_watch('agent_count', lambda ctx: len(ctx.agents))

# Step through
debugger.step()
debugger.continue_execution()
```

## License

Proprietary - QUT Transport Modeling Group