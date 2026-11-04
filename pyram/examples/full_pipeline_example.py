#!/usr/bin/env python
"""
Example: Full PYRAM pipeline from Python API.

This script demonstrates how to use PYRAM programmatically
to run the complete travel demand modeling pipeline.
"""

from pathlib import Path
import tempfile

from pyram import Pipeline, Config
from pyram.config import create_default_config
from pyram.data import load_hhts, load_skims, load_landuse
from pyram.models import (
    create_all_models,
    DestinationChoiceModel,
    ModeChoiceModel,
    ParkingChoiceModel,
)


def example_basic_usage():
    """Basic usage with default config."""
    print("=== Basic Usage ===")
    
    # Create default config
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config = create_default_config(str(config_path))
        print(f"Created config at {config_path}")
        
        # Modify config for your data
        config.data.hhts_trips = "data/hhts_trips.csv"
        config.data.hhts_persons = "data/hhts_persons.csv"
        config.data.skims_omx = "data/skims.omx"
        config.data.landuse_csv = "data/landuse.csv"
        
        # Enable specific models
        config.destination.model_type = "nl_distance"  # Nested Logit with distance bands
        config.mode.model_type = "mxl"  # Mixed Logit
        config.parking.enable = True
        config.dta.enable = True
        config.dta.engine = "matsim"
        
        # Run pipeline
        pipeline = Pipeline(config)
        # result = pipeline.run()  # Uncomment to actually run
        
        print("Pipeline ready to run!")
        print(f"  Stages: CDAP, Timing, Destination, Mode, Trip Chain, Parking, DTA")


def example_model_estimation():
    """Estimate models from HHTS data."""
    print("\n=== Model Estimation ===")
    
    # Load your data
    # data = load_all_data(config)
    # For demo, we'll show the API
    
    # Create models from config
    config = Config()
    models = create_all_models(config)
    
    # Estimate destination choice
    dest_model = models["destination"]
    # dest_model.estimate(
    #     trips=data["trips"],
    #     persons=data["persons"],
    #     landuse=data["landuse"],
    #     skims=data["skim_lookup"],
    # )
    
    # Save model
    # dest_model.save("output/models/destination_model.pkl")
    
    print("Destination model estimation API:")
    print("  model.estimate(trips, persons, landuse, skims)")
    print("  model.save('path/to/model.pkl')")
    
    # Estimate mode choice
    mode_model = models["mode"]
    # mode_model.estimate(...)
    # mode_model.save("output/models/mode_model.pkl")
    
    print("\nMode model estimation API:")
    print("  model.estimate(trips, persons, landuse, skims)")
    print("  model.save('path/to/model.pkl')")


def example_prediction():
    """Run prediction on synthetic population."""
    print("\n=== Prediction ===")
    
    # Load saved models
    dest_model = DestinationChoiceModel.load("output/models/destination_model.pkl")
    mode_model = ModeChoiceModel.load("output/models/mode_model.pkl")
    parking_model = ParkingChoiceModel.load("output/models/parking_model.pkl")
    
    # Load synthetic population
    # synth_pop = pd.read_csv("output/synthetic_population.csv")
    
    # Predict destinations
    # dest_pred = dest_model.predict(
    #     trips=synth_pop,
    #     persons=persons,
    #     landuse=landuse,
    #     skims=skim_lookup,
    # )
    
    # Predict modes
    # mode_pred = mode_model.predict(
    #     trips=dest_pred,
    #     persons=persons,
    #     landuse=landuse,
    #     skims=skim_lookup,
    # )
    
    # Predict parking
    # final_plan = parking_model.predict(
    #     trips=mode_pred,
    #     persons=persons,
    #     location_table=location_table,
    #     landuse=landuse,
    #     skims=skim_lookup,
    # )
    
    # final_plan.to_csv("output/final_plan.csv", index=False)
    
    print("Prediction API:")
    print("  model.predict(trips, persons, landuse, skims)")
    print("  Returns DataFrame with predicted choices")


def example_custom_config():
    """Custom configuration for specific region."""
    print("\n=== Custom Configuration ===")
    
    # Create config for different region
    config = Config()
    
    # Australian South East Queensland example
    config.data.hhts_trips = "data/seq/hhts_trips.csv"
    config.data.hhts_persons = "data/seq/hhts_persons.csv"
    config.data.hhts_households = "data/seq/hhts_households.csv"
    config.data.skims_omx = "data/seq/skims_371.omx"
    config.data.landuse_csv = "data/seq/landuse_371.csv"
    config.data.zone_crosswalk = "data/seq/sa2_to_order.csv"
    config.data.location_table = "data/seq/location_table.csv"
    config.data.output_dir = "output/seq"
    
    config.zones.zone_id_col = "ORDER"
    config.zones.n_zones = 371
    config.zones.crs = "EPSG:28356"
    
    config.population.target_region = "QLD"
    config.population.use_overture = True
    
    # Model configuration
    config.destination.model_type = "nl_distance"
    config.destination.distance_bands = [0, 2, 5, 10, 100]
    config.destination.cbd_zones = [1, 2, 3, 4, 5]  # Brisbane CBD
    
    config.mode.model_type = "mxl"
    config.mode.mixed_logit = True
    
    config.parking.enable = True
    config.parking.cbd_zones = [1, 2, 3, 4, 5]
    
    # HPC resources
    config.hpc.stage_ncpus = 16
    config.hpc.stage_mem = "500GB"
    config.hpc.stage_walltime = "48:00:00"
    
    print("Custom SEQ config:")
    print(f"  Zones: {config.zones.n_zones}")
    print(f"  Destination: {config.destination.model_type}")
    print(f"  Mode: {config.mode.model_type}")
    print(f"  Output: {config.data.output_dir}")


def example_hpc_submission():
    """Generate environment variables for PBS submission."""
    print("\n=== HPC/PBS Submission ===")
    
    from pyram.config import ConfigManager
    
    config = Config()
    manager = ConfigManager(config)
    overrides = config.get_env_overrides()
    
    print("Environment variables for qsub:")
    for k, v in sorted(overrides.items()):
        print(f"  {k}={v}")
    
    # Or write to file
    # with open("submit.env", "w") as f:
    #     for k, v in overrides.items():
    #         f.write(f"{k}={v}\n")
    # 
    # Then: qsub -v "$(cat submit.env | tr '\n' ',')" job_submit.pbs


def example_dta_feedback():
    """Run DTA feedback loop."""
    print("\n=== DTA Feedback Loop ===")
    
    from pyram.pipeline import run_stage7_dta_feedback
    
    config = Config()
    config.dta.enable = True
    config.dta.engine = "matsim"
    config.dta.iterations = 3
    config.dta.convergence_threshold = 0.02
    
    # Load Stage 6 output (plan with parking)
    # plan = pd.read_csv("output/stage6_output.csv")
    
    # final_plan, final_skims = run_stage7_dta_feedback(
    #     plan=plan,
    #     persons=persons,
    #     landuse=landuse,
    #     skim_path="data/skims.omx",
    #     iterations=3,
    #     engine="matsim",
    #     output_dir="output/dta_feedback",
    # )
    
    print("DTA Feedback API:")
    print("  run_stage7_dta_feedback(plan, persons, landuse, skim_path, ...)")
    print("  Returns: (equilibrated_plan, final_skim_path)")


if __name__ == "__main__":
    example_basic_usage()
    example_model_estimation()
    example_prediction()
    example_custom_config()
    example_hpc_submission()
    example_dta_feedback()
    
    print("\n=== All examples completed ===")
    print("See config.yaml for all configuration options.")
    print("Run 'pyram --help' for CLI commands.")