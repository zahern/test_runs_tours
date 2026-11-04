"""
Tests for pyram package.
"""

import pytest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np


def test_config_creation():
    """Test default config creation."""
    from pyram.config import Config, create_default_config
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"
        config = create_default_config(str(config_path))
        
        assert config is not None
        assert config_path.exists()
        assert config.data.output_dir == "output"
        assert config.zones.n_zones == 371


def test_config_yaml_roundtrip():
    """Test config save/load from YAML."""
    from pyram.config import Config
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"
        
        # Create and save
        config = Config()
        config.data.output_dir = "custom_output"
        config.zones.n_zones = 500
        config.to_yaml(config_path)
        
        # Load back
        loaded = Config.from_yaml(config_path)
        assert loaded.data.output_dir == "custom_output"
        assert loaded.zones.n_zones == 500


def test_config_env_overrides():
    """Test environment variable overrides."""
    from pyram.config import Config
    import os
    import tempfile
    from pathlib import Path
    
    # Set env vars
    os.environ["PYRAM_DESTINATION__MODEL_TYPE"] = "nl_distance"
    os.environ["PYRAM_MODE__MIXED_LOGIT"] = "true"
    os.environ["PYRAM_HPC__STAGE_NCPUS"] = "16"
    
    try:
        # Need to use from_yaml to apply env overrides
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            Config().to_yaml(config_path)  # Create default config file
            config = Config.from_yaml(config_path)
        
        assert config.destination.model_type == "nl_distance"
        assert config.mode.mixed_logit is True
        assert config.hpc.stage_ncpus == 16
    finally:
        # Clean up
        for key in ["PYRAM_DESTINATION__MODEL_TYPE", "PYRAM_MODE__MIXED_LOGIT", "PYRAM_HPC__STAGE_NCPUS"]:
            os.environ.pop(key, None)


def test_data_loading():
    """Test HHTS data loading with mock data."""
    from pyram.data import load_hhts
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock HHTS files
        trips = pd.DataFrame({
            "PERSID": [1, 1, 2],
            "TRIPID": [1, 2, 1],
            "ORIGIN": [10, 20, 30],
            "DESTINATION": [20, 10, 40],
            "PURPOSE": ["work", "work", "shop"],
            "MODE": [1, 1, 3],
            "START_TIME": [480, 540, 600],
            "END_TIME": [510, 570, 630],
        })
        persons = pd.DataFrame({
            "PERSID": [1, 2],
            "HHID": [100, 100],
            "AGEP": [35, 30],
            "SEXP": [1, 2],
            "INCP": [80000, 60000],
            "VEHRD": [2, 1],
            "CARLICENCE": [1, 1],
        })
        households = pd.DataFrame({
            "HHID": [100],
            "HHSIZE": [2],
            "VEHRD": [2],
        })
        
        trips_path = Path(tmpdir) / "trips.csv"
        persons_path = Path(tmpdir) / "persons.csv"
        hh_path = Path(tmpdir) / "households.csv"
        
        trips.to_csv(trips_path, index=False)
        persons.to_csv(persons_path, index=False)
        households.to_csv(hh_path, index=False)
        
        # Load
        data = load_hhts(str(trips_path), str(persons_path), str(hh_path))
        
        assert "trips" in data
        assert "persons" in data
        assert "households" in data
        assert len(data["trips"]) == 3
        assert "person_id" in data["trips"].columns
        assert "OTAZ" in data["trips"].columns
        assert "DTAZ" in data["trips"].columns


def test_models_creation():
    """Test model factory functions."""
    from pyram.models import (
        create_destination_model,
        create_mode_model,
        create_parking_model,
        create_trip_destination_model,
    )
    from pyram.config import Config
    
    config = Config()
    
    dest_model = create_destination_model(config.destination.__dict__)
    mode_model = create_mode_model(config.mode.__dict__)
    parking_model = create_parking_model(config.parking.__dict__)
    trip_dest_model = create_trip_destination_model({})
    
    assert dest_model is not None
    assert mode_model is not None
    assert parking_model is not None
    assert trip_dest_model is not None


def test_destination_model_types():
    """Test different destination model types."""
    from pyram.models import DestinationChoiceModel
    
    for model_type in ["mnl", "nl_distance", "nl_sa3"]:
        config = {"model_type": model_type}
        model = DestinationChoiceModel(config)
        assert model.model_type == model_type
        
        if model_type == "nl_distance":
            assert model.distance_bands == [0, 2, 5, 10, 100]


def test_mode_model_types():
    """Test different mode model types."""
    from pyram.models import ModeChoiceModel
    
    for model_type in ["nl", "mxl", "mxl_gse"]:
        config = {"model_type": model_type, "mixed_logit": model_type in ["mxl", "mxl_gse"]}
        model = ModeChoiceModel(config)
        assert model.model_type == model_type
        
        if model_type in ["mxl", "mxl_gse"]:
            assert model.mixed_logit is True
        else:
            assert model.mixed_logit is False


def test_pipeline_creation():
    """Test pipeline instantiation."""
    from pyram.pipeline import Pipeline
    from pyram.config import Config
    
    config = Config()
    pipeline = Pipeline(config)
    
    assert pipeline.config == config
    assert pipeline.manager is not None


def test_pipeline_result():
    """Test PipelineResult dataclass."""
    from pyram.pipeline import PipelineResult
    
    result = PipelineResult(
        success=True,
        stages_completed=["load_data", "stage3_destination"],
        outputs={"model_path": "output/models/dest.pkl"},
    )
    
    assert result.success is True
    assert len(result.stages_completed) == 2


def test_cli_import():
    """Test CLI module imports correctly."""
    from pyram.cli import app
    
    assert app is not None
    assert app.info.name == "pyram"


def test_activity_generation_model_creation():
    """Test competing hazard activity generation model creation."""
    from pyram.models import (
        create_activity_generation_model,
        create_planning_order_model,
        create_timing_model,
        create_routine_location_model,
    )
    from pyram.config import Config
    
    config = Config()
    
    # Test activity generation model
    act_gen = create_activity_generation_model(config.activity_generation.__dict__)
    assert act_gen is not None
    assert hasattr(act_gen, 'estimate')
    assert hasattr(act_gen, 'predict_activities')
    
    # Test planning order model
    planning = create_planning_order_model()
    assert planning is not None
    
    # Test timing model
    timing = create_timing_model()
    assert timing is not None
    
    # Test routine location model
    location = create_routine_location_model()
    assert location is not None


def test_competing_hazard_basic():
    """Test basic competing hazard functionality."""
    from pyram.models.activity_generation import CompetingHazardActivityGenerator
    
    gen = CompetingHazardActivityGenerator(seed=42)
    
    # Test covariate building
    persons = pd.DataFrame({
        'person_id': ['p1', 'p2'],
        'age5p': [3, 4],
        'sexp': [1, 2],
        'WORKSTAT': [1, 0],
        'incp': [80000, 50000],
        'vehrd': [2, 1],
    })
    
    X = gen._build_covariates(persons)
    assert X.shape == (2, X.shape[1])
    
    # Test default configs
    assert 'work' in gen.configs
    assert 'home' in gen.configs
    assert gen.configs['work'].n_intervals == 96


def test_activity_generation_predict():
    """Test activity generation prediction."""
    from pyram.models.activity_generation import CompetingHazardActivityGenerator
    
    gen = CompetingHazardActivityGenerator(seed=42)
    
    # Create mock persons and activity episodes for estimation
    persons = pd.DataFrame({
        'person_id': ['p1', 'p2', 'p3'],
        'age5p': [3, 4, 2],
        'sexp': [1, 2, 1],
        'WORKSTAT': [1, 0, 1],
        'incp': [80000, 50000, 30000],
        'vehrd': [2, 1, 0],
    })
    
    # Mock activity episodes for estimation
    episodes = pd.DataFrame({
        'person_id': ['p1', 'p1', 'p2', 'p3'],
        'activity_type': ['work', 'home', 'shopping', 'work'],
        'start_time': [8.0, 17.0, 14.0, 9.0],
        'duration': [8.0, 12.0, 1.5, 8.0],
    })
    
    # Estimate model
    gen.estimate(persons, episodes)
    assert gen.fitted
    
    # Predict activities for new persons
    new_persons = pd.DataFrame({
        'person_id': ['p4', 'p5'],
        'age5p': [3, 4],
        'sexp': [1, 2],
        'WORKSTAT': [1, 0],
        'incp': [70000, 60000],
        'vehrd': [1, 2],
    })
    
    predicted = gen.predict_activities(new_persons, seed=42)
    
    assert len(predicted) > 0
    assert 'person_id' in predicted.columns
    assert 'activity_type' in predicted.columns
    assert 'start_time' in predicted.columns
    assert 'duration' in predicted.columns
    assert 'end_time' in predicted.columns


def test_planning_order_model():
    """Test activity planning order model."""
    from pyram.models.activity_generation import ActivityPlanningOrderModel
    
    planning = ActivityPlanningOrderModel(seed=42)
    
    activities = pd.DataFrame({
        'person_id': ['p1', 'p1', 'p2'],
        'activity_type': ['work', 'shopping', 'work'],
        'start_time': [8.0, 17.0, 9.0],
        'duration': [8.0, 1.5, 8.0],
    })
    
    persons = pd.DataFrame({
        'person_id': ['p1', 'p2'],
        'age5p': [3, 4],
    })
    
    # Should not raise
    result = planning.predict_planning_order(activities, persons)
    
    assert 'flexibility' in result.columns
    assert 'plan_horizon' in result.columns
    
    # Check default mappings
    work_row = result[result['activity_type'] == 'work'].iloc[0]
    assert work_row['flexibility'] == 'fixed'
    assert work_row['plan_horizon'] == 'routine'
    
    shop_row = result[result['activity_type'] == 'shopping'].iloc[0]
    assert shop_row['flexibility'] == 'semi_flexible'
    assert shop_row['plan_horizon'] == 'pre_planned'


def test_timing_model():
    """Test hybrid timing choice model."""
    from pyram.models.activity_generation import HybridTimingChoiceModel
    
    timing = HybridTimingChoiceModel(seed=42)
    
    activities = pd.DataFrame({
        'person_id': ['p1', 'p1', 'p2'],
        'activity_type': ['work', 'shopping', 'work'],
    })
    
    persons = pd.DataFrame({
        'person_id': ['p1', 'p2'],
    })
    
    result = timing.predict_timing(activities, persons)
    
    assert 'start_time' in result.columns
    assert 'duration' in result.columns
    assert 'end_time' in result.columns
    
    # Check routine activities get reasonable times
    work_rows = result[result['activity_type'] == 'work']
    assert all(work_rows['start_time'].between(6, 10))
    assert all(work_rows['duration'].between(6, 12))


def test_routine_location_model():
    """Test routine location choice model."""
    from pyram.models.activity_generation import RoutineLocationChoiceModel
    
    location = RoutineLocationChoiceModel(seed=42)
    
    persons = pd.DataFrame({
        'person_id': ['p1', 'p2', 'p3'],
        'WORKSTAT': [1, 0, 1],
        'STUDYING': [0, 1, 0],
    })
    
    landuse = pd.DataFrame({
        'zone_id': [1, 2, 3, 4, 5],
        'TOTEMP': [1000, 500, 200, 100, 50],
        'ENROLLMENT': [500, 300, 200, 100, 50],
    }, index=[1, 2, 3, 4, 5])
    
    # Test work location
    result = location.predict(persons.copy(), landuse, {}, segment='work')
    assert 'work_zone' in result.columns
    assert result.loc[result['WORKSTAT'] > 0, 'work_zone'].notna().all()
    
    # Test school location
    result = location.predict(persons.copy(), landuse, {}, segment='school')
    assert 'school_zone' in result.columns
    assert result.loc[result['STUDYING'] > 0, 'school_zone'].notna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])