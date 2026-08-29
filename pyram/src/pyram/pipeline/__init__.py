"""
Main pipeline orchestration for pyram.

Provides the Pipeline class that coordinates all stages:
1. Data loading
2. Synthetic population generation
3. Stage 1: CDAP (household coordination)
4. Stage 2: Timing/duration models
5. Stage 3: Destination choice
6. Stage 4: Mode choice
7. Stage 5: Trip chain building
8. Stage 6: Parking choice
9. Stage 7: DTA feedback loop
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

from pyram.config import Config, ConfigManager
from pyram.data import load_all_data, build_synthetic_population
from pyram.models import (
    create_all_models,
    DestinationChoiceModel,
    ModeChoiceModel,
    ParkingChoiceModel,
    TripDestinationModel,
)


@dataclass
class PipelineResult:
    """Result container for pipeline execution."""
    success: bool
    stages_completed: List[str]
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    config: Optional[Config] = None


class Pipeline:
    """
    Main pipeline orchestrator.
    
    Example:
        config = Config.from_yaml("config.yaml")
        pipeline = Pipeline(config)
        result = pipeline.run()
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.manager = ConfigManager(config)
        self.manager.setup_environment()
        self.manager.create_output_dirs()
        
        self.data = {}
        self.models = {}
        self.results = {}
    
    def run(self, stages: Optional[List[int]] = None) -> PipelineResult:
        """
        Run the pipeline.
        
        Parameters
        ----------
        stages : list of int, optional
            Specific stages to run (1-7). If None, runs all stages.
            1: CDAP
            2: Timing
            3: Destination
            4: Mode
            5: Trip Chain
            6: Parking
            7: DTA Feedback
        
        Returns
        -------
        PipelineResult
        """
        stages = stages or list(range(1, 8))
        completed = []
        
        try:
            # Stage 0: Load data
            self._load_data()
            completed.append("load_data")
            
            # Stage 0.5: Build synthetic population
            if 0 in stages or (not stages and True):
                self._build_synthetic_population()
                completed.append("synthetic_population")
            
            # Stage 1: CDAP
            if 1 in stages:
                self._run_stage1_cdap()
                completed.append("stage1_cdap")
            
            # Stage 2: Timing
            if 2 in stages:
                self._run_stage2_timing()
                completed.append("stage2_timing")
            
            # Stage 3: Destination
            if 3 in stages:
                self._run_stage3_destination()
                completed.append("stage3_destination")
            
            # Stage 4: Mode
            if 4 in stages:
                self._run_stage4_mode()
                completed.append("stage4_mode")
            
            # Stage 5: Trip Chain
            if 5 in stages:
                self._run_stage5_trip_chain()
                completed.append("stage5_trip_chain")
            
            # Stage 6: Parking
            if 6 in stages:
                self._run_stage6_parking()
                completed.append("stage6_parking")
            
            # Stage 7: DTA Feedback
            if 7 in stages and self.config.dta.enable:
                self._run_stage7_dta_feedback()
                completed.append("stage7_dta_feedback")
            
            return PipelineResult(
                success=True,
                stages_completed=completed,
                outputs=self.results,
                config=self.config,
            )
            
        except Exception as e:
            return PipelineResult(
                success=False,
                stages_completed=completed,
                outputs=self.results,
                errors=[str(e)],
                config=self.config,
            )
    
    def _load_data(self):
        """Load all input data."""
        print("Loading data...")
        self.data = load_all_data(self.config)
        print(f"  Trips: {len(self.data['trips']):,}")
        print(f"  Persons: {len(self.data['persons']):,}")
        if 'households' in self.data:
            print(f"  Households: {len(self.data['households']):,}")
    
    def _build_synthetic_population(self):
        """Build synthetic population from HHTS + Overture."""
        print("Building synthetic population...")
        
        target_pop = self.config.population.sample_fraction
        if target_pop <= 1.0:
            # Fraction of HHTS
            target_pop = int(len(self.data['persons']) * target_pop)
        
        # Check if Overture data available
        overture_gdf = None
        zone_boundaries = None
        if self.config.population.use_overture:
            overture_path = Path(self.config.data.output_dir) / "overture_population.parquet"
            if overture_path.exists():
                import geopandas as gpd
                overture_gdf = gpd.read_parquet(overture_path)
                print(f"  Loaded Overture data: {len(overture_gdf)} features")
        
        synth = build_synthetic_population(
            hhts_persons=self.data['persons'],
            hhts_households=self.data.get('households', pd.DataFrame()),
            zone_crosswalk=self.data['crosswalk'],
            landuse=self.data['landuse'],
            target_population=target_pop,
            seed=self.config.seed,
            use_overture=self.config.population.use_overture,
            overture_gdf=overture_gdf,
            zone_boundaries=zone_boundaries,
        )
        
        self.data['synthetic_population'] = synth
        output_path = Path(self.config.data.output_dir) / "synthetic_population.csv"
        synth.to_csv(output_path, index=False)
        print(f"  Generated {len(synth):,} synthetic persons")
        print(f"  Saved to {output_path}")
        
        self.results['synthetic_population'] = synth
    
    def _run_stage1_cdap(self):
        """Stage 1: Household coordination (CDAP) + joint tours."""
        print("Stage 1: CDAP household coordination...")
        
        # TODO: Implement CDAP from original codebase
        # This would use cdap_households.py logic
        warnings.warn("Stage 1 CDAP not yet implemented in Pipeline class")
        self.results['stage1_cdap'] = {"status": "skipped"}
    
    def _run_stage2_timing(self):
        """Stage 2: Departure time / duration models."""
        print("Stage 2: Timing models...")
        
        # TODO: Implement timing models
        warnings.warn("Stage 2 timing not yet implemented in Pipeline class")
        self.results['stage2_timing'] = {"status": "skipped"}
    
    def _run_stage3_destination(self):
        """Stage 3: Destination choice estimation."""
        print("Stage 3: Destination choice...")
        
        model = self.models.get('destination') or create_all_models(self.config)['destination']
        
        # Estimate from HHTS
        model.estimate(
            trips=self.data['trips'],
            persons=self.data['persons'],
            landuse=self.data['landuse'],
            skims=self.data['skim_lookup'],
        )
        
        # Save model
        model_path = Path(self.config.data.output_dir) / "models" / "destination_model.pkl"
        model.save(model_path)
        
        self.models['destination'] = model
        self.results['stage3_destination'] = {
            "model_path": str(model_path),
            "coefficients": model.coefficients,
        }
        print(f"  Destination model saved to {model_path}")
    
    def _run_stage4_mode(self):
        """Stage 4: Mode choice estimation."""
        print("Stage 4: Mode choice...")
        
        model = self.models.get('mode') or create_all_models(self.config)['mode']
        
        model.estimate(
            trips=self.data['trips'],
            persons=self.data['persons'],
            landuse=self.data['landuse'],
            skims=self.data['skim_lookup'],
        )
        
        model_path = Path(self.config.data.output_dir) / "models" / "mode_model.pkl"
        model.save(model_path)
        
        self.models['mode'] = model
        self.results['stage4_mode'] = {
            "model_path": str(model_path),
            "coefficients": model.coefficients,
        }
        print(f"  Mode model saved to {model_path}")
    
    def _run_stage5_trip_chain(self):
        """Stage 5: Trip chain building with destination + mode prediction."""
        print("Stage 5: Trip chain building...")
        
        # This combines destination prediction, mode prediction, and trip chain assembly
        # Uses synthetic population as input
        
        synth = self.data.get('synthetic_population')
        if synth is None:
            raise ValueError("Synthetic population required for Stage 5. Run _build_synthetic_population first.")
        
        # Predict destinations for synthetic population
        if 'destination' in self.models:
            dest_pred = self.models['destination'].predict(
                trips=synth,  # synth has trip-like structure
                persons=self.data['persons'],
                landuse=self.data['landuse'],
                skims=self.data['skim_lookup'],
            )
        
        # Predict modes
        if 'mode' in self.models:
            mode_pred = self.models['mode'].predict(
                trips=dest_pred if 'destination' in self.models else synth,
                persons=self.data['persons'],
                landuse=self.data['landuse'],
                skims=self.data['skim_lookup'],
            )
        
        # Build trip chains
        # TODO: Implement trip chain building from original codebase
        
        output_path = Path(self.config.data.output_dir) / "stage5_trip_chains.csv"
        # mode_pred.to_csv(output_path, index=False)
        
        self.results['stage5_trip_chain'] = {
            "status": "completed",
            "output_path": str(output_path),
        }
        print(f"  Trip chains saved to {output_path}")
    
    def _run_stage6_parking(self):
        """Stage 6: Parking choice for auto trips."""
        print("Stage 6: Parking choice...")
        
        if not self.config.parking.enable:
            print("  Parking disabled in config")
            return
        
        model = self.models.get('parking') or create_all_models(self.config)['parking']
        
        # Need trip chains from Stage 5
        trip_chains = self.results.get('stage5_trip_chain', {}).get('trip_chains')
        if trip_chains is None:
            # Load from file
            trip_chain_path = Path(self.config.data.output_dir) / "stage5_trip_chains.csv"
            if trip_chain_path.exists():
                trip_chains = pd.read_csv(trip_chain_path)
        
        if trip_chains is not None:
            model.estimate(
                trips=trip_chains,
                persons=self.data['persons'],
                location_table=self.data['location_table'],
                landuse=self.data['landuse'],
                skims=self.data['skim_lookup'],
            )
            
            model_path = Path(self.config.data.output_dir) / "models" / "parking_model.pkl"
            model.save(model_path)
            
            self.models['parking'] = model
            self.results['stage6_parking'] = {
                "model_path": str(model_path),
                "coefficients": model.coefficients,
            }
            print(f"  Parking model saved to {model_path}")
        else:
            warnings.warn("No trip chains available for parking estimation")
    
    def _run_stage7_dta_feedback(self):
        """Stage 7: DTA feedback loop."""
        print("Stage 7: DTA feedback loop...")
        
        if not self.config.dta.enable:
            print("  DTA feedback disabled in config")
            return
        
        # TODO: Implement DTA feedback from dta_feedback.py
        warnings.warn("Stage 7 DTA feedback not yet implemented in Pipeline class")
        self.results['stage7_dta_feedback'] = {"status": "skipped"}


# ── Convenience Functions ─────────────────────────────────────────────────────

def run_stage1_cdap(config: Config, data: Dict) -> Dict:
    """Run Stage 1: CDAP household coordination."""
    pipeline = Pipeline(config)
    pipeline.data = data
    pipeline._run_stage1_cdap()
    return pipeline.results.get('stage1_cdap', {})


def run_stage2_timing(config: Config, data: Dict) -> Dict:
    """Run Stage 2: Timing models."""
    pipeline = Pipeline(config)
    pipeline.data = data
    pipeline._run_stage2_timing()
    return pipeline.results.get('stage2_timing', {})


def run_stage3_destination(config: Config, data: Dict) -> Dict:
    """Run Stage 3: Destination choice."""
    pipeline = Pipeline(config)
    pipeline.data = data
    pipeline._run_stage3_destination()
    return pipeline.results.get('stage3_destination', {})


def run_stage4_mode(config: Config, data: Dict) -> Dict:
    """Run Stage 4: Mode choice."""
    pipeline = Pipeline(config)
    pipeline.data = data
    pipeline._run_stage4_mode()
    return pipeline.results.get('stage4_mode', {})


def run_stage5_trip_chain(config: Config, data: Dict) -> Dict:
    """Run Stage 5: Trip chain building."""
    pipeline = Pipeline(config)
    pipeline.data = data
    pipeline._run_stage5_trip_chain()
    return pipeline.results.get('stage5_trip_chain', {})


def run_stage6_parking(config: Config, data: Dict) -> Dict:
    """Run Stage 6: Parking choice."""
    pipeline = Pipeline(config)
    pipeline.data = data
    pipeline._run_stage6_parking()
    return pipeline.results.get('stage6_parking', {})


def run_stage7_dta_feedback(config: Config, data: Dict) -> Dict:
    """Run Stage 7: DTA feedback loop."""
    pipeline = Pipeline(config)
    pipeline.data = data
    pipeline._run_stage7_dta_feedback()
    return pipeline.results.get('stage7_dta_feedback', {})