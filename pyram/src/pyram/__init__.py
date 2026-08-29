"""
pyram
=====

PYRAM travel demand modeling pipeline for activity-based transport simulation.

This package provides a complete pipeline for:
- Synthetic population generation (with Overture Maps integration)
- Activity scheduling (CDAP, joint tours)
- Destination choice (MNL, Nested Logit with distance bands)
- Mode choice (Nested Logit, Mixed Logit)
- Trip chain building
- Parking choice (on-street, garage, P&R)
- DTA feedback loop (MATSim, SUMO, DynusT)
- MATSim/SUMO export for microsimulation

Quick start:
    from pyram import Pipeline, Config
    
    config = Config.from_yaml("config.yaml")
    pipeline = Pipeline(config)
    result = pipeline.run()
    
    # Or use CLI:
    # pyram-run-pipeline --config config.yaml

Main modules:
- pyram.config: Configuration system
- pyram.data: Data loading (HHTS, skims, landuse, Overture)
- pyram.models: Discrete choice models (destination, mode, parking)
- pyram.pipeline: Main pipeline orchestration
- pyram.cli: Command-line interface
"""

__version__ = "0.1.0"
__author__ = "PYRAM Team"

# Core exports
from pyram.config import Config, ConfigManager
from pyram.pipeline import Pipeline, PipelineResult

# Model exports
from pyram.models import (
    DestinationChoiceModel,
    ModeChoiceModel,
    ParkingChoiceModel,
    TripDestinationModel,
)

# Data exports
from pyram.data import (
    load_hhts,
    load_skims,
    load_landuse,
    build_synthetic_population,
    fetch_overture_population,
)

# Pipeline exports
from pyram.pipeline import (
    run_stage1_cdap,
    run_stage2_timing,
    run_stage3_destination,
    run_stage4_mode,
    run_stage5_trip_chain,
    run_stage6_parking,
    run_stage7_dta_feedback,
)

__all__ = [
    # Core
    "Config",
    "ConfigManager", 
    "Pipeline",
    "PipelineResult",
    # Models
    "DestinationChoiceModel",
    "ModeChoiceModel", 
    "ParkingChoiceModel",
    "TripDestinationModel",
    # Data
    "load_hhts",
    "load_skims",
    "load_landuse",
    "build_synthetic_population",
    "fetch_overture_population",
    # Pipeline stages
    "run_stage1_cdap",
    "run_stage2_timing",
    "run_stage3_destination",
    "run_stage4_mode",
    "run_stage5_trip_chain",
    "run_stage6_parking",
    "run_stage7_dta_feedback",
]