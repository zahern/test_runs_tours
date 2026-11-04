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
from pyram.dta import DTAFeedbackLoop
from pyram.models import (
    create_all_models,
    create_cdap_model,
    create_purpose_timing_model,
    expand_daily_patterns,
    build_trip_chains,
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
            
            # Stage 0.5: Build synthetic population (required by stages 1 and 5)
            if 0 in stages or any(stage in stages for stage in (1, 5)):
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
        
        if not self.config.cdap.enable:
            print("  CDAP disabled in config")
            self.results['stage1_cdap'] = {"status": "disabled"}
            return
        
        synth = self.data.get('synthetic_population')
        if synth is None:
            raise ValueError(
                "Synthetic population required for Stage 1. "
                "Run _build_synthetic_population first."
            )
        
        cdap_config = {
            **self.config.cdap.__dict__,
            "max_acts_per_person": self.config.trip_chain.max_acts_per_person,
        }
        model = create_cdap_model(cdap_config, seed=self.config.seed)
        model.estimate(persons=self.data['persons'], trips=self.data['trips'])
        
        patterns = model.simulate(synth)
        if self.config.cdap.joint_tours:
            joint_tours = model.coordinate_households(patterns, synth)
        else:
            joint_tours = pd.DataFrame(
                columns=["tour_id", "hhid", "purpose", "participants", "n_participants"]
            )
        
        output_dir = Path(self.config.data.output_dir)
        pattern_path = output_dir / "stage1_daily_patterns.csv"
        joint_path = output_dir / "stage1_joint_tours.csv"
        patterns.to_csv(pattern_path, index=False)
        joint_tours.to_csv(joint_path, index=False)
        model.save(output_dir / "models" / "cdap_model.pkl")
        
        self.models['cdap'] = model
        self.data['daily_patterns'] = patterns
        self.data['joint_tours'] = joint_tours
        
        counts = patterns['cdap'].value_counts()
        shares = {
            f"share_{state}": float(counts.get(state, 0)) / max(len(patterns), 1)
            for state in ("H", "N", "M")
        }
        
        self.results['stage1_cdap'] = {
            "status": "completed",
            "model_path": str(output_dir / "models" / "cdap_model.pkl"),
            "pattern_path": str(pattern_path),
            "joint_tour_path": str(joint_path),
            "n_joint_tours": int(len(joint_tours)),
            **shares,
        }
        print(
            f"  Patterns: H {shares['share_H']:.1%}, N {shares['share_N']:.1%}, "
            f"M {shares['share_M']:.1%}"
        )
        print(f"  Joint tours: {len(joint_tours):,}")
        print(f"  Saved to {pattern_path}")
    
    def _run_stage2_timing(self):
        """Stage 2: Departure time / duration models."""
        print("Stage 2: Timing models...")
        
        patterns = self.data.get('daily_patterns')
        if patterns is None:
            raise ValueError(
                "Stage 2 requires daily patterns from Stage 1 "
                "(run stages 1-2 together, or enable cdap in the config)."
            )
        
        synth = self.data.get('synthetic_population')
        activities = expand_daily_patterns(
            patterns,
            persons=synth,
            joint_tours=self.data.get('joint_tours'),
            cdap_model=self.models.get('cdap'),
            seed=self.config.seed,
        )
        
        timing = create_purpose_timing_model(
            self.config.timing.__dict__, seed=self.config.seed
        )
        timing.estimate(self.data['trips'])
        schedule = timing.predict(activities)
        
        output_dir = Path(self.config.data.output_dir)
        schedule_path = output_dir / "stage2_activity_schedule.csv"
        schedule.to_csv(schedule_path, index=False)
        timing.save(output_dir / "models" / "timing_model.pkl")
        
        self.models['timing'] = timing
        self.data['activity_schedule'] = schedule
        
        self.results['stage2_timing'] = {
            "status": "completed",
            "model_path": str(output_dir / "models" / "timing_model.pkl"),
            "schedule_path": str(schedule_path),
            "n_activities": int(len(schedule)),
        }
        print(f"  Scheduled {len(schedule):,} activities")
        print(f"  Saved to {schedule_path}")
    
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
        """
        Stage 5: Trip chain building with destination + mode prediction.
        
        Builds home-based tours from the Stage 2 activity schedule, applies the
        fitted destination and mode models to those tours, and writes the final
        synthetic trip list.
        """
        print("Stage 5: Trip chain building...")
        
        schedule = self.data.get('activity_schedule')
        if schedule is None:
            schedule_path = (
                Path(self.config.data.output_dir) / "stage2_activity_schedule.csv"
            )
            if schedule_path.exists():
                schedule = pd.read_csv(schedule_path)
        if schedule is None:
            raise ValueError(
                "Stage 5 requires an activity schedule from Stage 2. "
                "Run stages 1-2 first, or provide stage2_activity_schedule.csv."
            )
        
        synth = self.data.get('synthetic_population')
        if synth is None:
            raise ValueError(
                "Synthetic population required for Stage 5. "
                "Run _build_synthetic_population first."
            )
        
        chains = build_trip_chains(schedule, persons=synth)
        landuse = self.data['landuse']
        skims = self.data.get('skim_lookup', {})
        
        destination_model = self.models.get('destination')
        if destination_model is not None and getattr(destination_model, 'fitted', False):
            outbound = chains['leg'] == 'outbound'
            predicted = destination_model.predict(
                trips=chains.loc[outbound].copy(),
                persons=synth,
                landuse=landuse,
                skims=skims,
            )
            destinations = dict(
                zip(predicted['tour_id'], predicted['DTAZ_pred'])
            )
            mapped = chains['tour_id'].map(destinations)
            chains.loc[outbound, 'DTAZ'] = (
                mapped[outbound]
                .fillna(chains.loc[outbound, 'DTAZ'])
                .astype(int)
                .to_numpy()
            )
            chains.loc[~outbound, 'OTAZ'] = (
                mapped[~outbound]
                .fillna(chains.loc[~outbound, 'OTAZ'])
                .astype(int)
                .to_numpy()
            )
            self.results.setdefault('stage5_trip_chain', {})['destination_model'] = 'applied'
        else:
            warnings.warn(
                "Destination model is not fitted; Stage 5 keeps home zones as destinations"
            )
        
        mode_model = self.models.get('mode')
        if mode_model is not None and getattr(mode_model, 'fitted', False):
            predicted = mode_model.predict(
                trips=chains.copy(),
                persons=synth,
                landuse=landuse,
                skims=skims,
            )
            chains['mode_pred'] = predicted['mode_pred'].to_numpy()
            chains['mode_code_pred'] = predicted['mode_code_pred'].to_numpy()
            chains['mode'] = chains['mode_code_pred'].fillna(1).astype(int)
            self.results.setdefault('stage5_trip_chain', {})['mode_model'] = 'applied'
        else:
            warnings.warn("Mode model is not fitted; Stage 5 leaves mode unassigned")
            chains['mode'] = np.nan
        
        output_path = Path(self.config.data.output_dir) / "stage5_trip_chains.csv"
        chains.to_csv(output_path, index=False)
        
        self.data['trip_chains'] = chains
        existing = self.results.get('stage5_trip_chain', {})
        self.results['stage5_trip_chain'] = {
            **existing,
            "status": "completed",
            "output_path": str(output_path),
            "trip_chains": chains,
            "n_trips": int(len(chains)),
            "n_tours": int(chains['tour_id'].nunique()),
        }
        print(f"  Built {chains['tour_id'].nunique():,} tours / {len(chains):,} trips")
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
                persons=self.data.get('synthetic_population', self.data['persons']),
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
        
        trip_chains = self.data.get('trip_chains')
        if trip_chains is None:
            trip_chain_path = Path(self.config.data.output_dir) / "stage5_trip_chains.csv"
            if trip_chain_path.exists():
                trip_chains = pd.read_csv(trip_chain_path)
        if trip_chains is None:
            warnings.warn("No trip chains available for DTA feedback")
            self.results['stage7_dta_feedback'] = {"status": "skipped"}
            return
        
        loop = DTAFeedbackLoop(
            config=self.config.dta,
            output_dir=Path(self.config.data.output_dir) / "dta_feedback",
        )
        outcome = loop.run(
            trip_chains,
            skim_lookup=self.data.get('skim_lookup'),
            landuse=self.data.get('landuse'),
        )
        
        if outcome.status not in ("completed", "max_iterations"):
            warnings.warn(f"DTA feedback {outcome.status}: {outcome.message}")
            self.results['stage7_dta_feedback'] = {
                "status": outcome.status,
                "engine": outcome.engine,
                "iterations": outcome.iterations,
                "message": outcome.message,
            }
            print(f"  DTA feedback not run: {outcome.message}")
            return
        
        final_plan_path = (
            Path(self.config.data.output_dir) / "stage7_final_plan.csv"
        )
        if outcome.plan is not None and len(outcome.plan):
            outcome.plan.to_csv(final_plan_path, index=False)
            self.data['trip_chains'] = outcome.plan
        if outcome.skim_lookup:
            self.data['skim_lookup'] = outcome.skim_lookup
        
        self.results['stage7_dta_feedback'] = {
            "status": outcome.status,
            "engine": outcome.engine,
            "iterations": outcome.iterations,
            "converged": outcome.converged,
            "history": outcome.history,
            "output_path": str(final_plan_path),
            "output_dir": outcome.output_dir,
        }
        print(
            f"  {outcome.engine}: {outcome.iterations} iteration(s), "
            f"{'converged' if outcome.converged else 'not converged'}"
        )
        print(f"  Final plan saved to {final_plan_path}")


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
