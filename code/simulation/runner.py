"""
Main Simulation Runner
======================
High-level entry point for running simulations.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from simulation.events.core import (
    SimulationContext, Scheduler, EventDispatcher, Event,
    EventType, EventPriority, create_simulation
)
from simulation.agents.agent import Agent, PersonAttributes, ActivityType, PersonType
from simulation.handlers.stage_handlers import (
    CDAPHandler, ActivityFrequencyHandler, ActivityDurationHandler,
    ScheduleTimingHandler, ModeDestinationHandler, TripExecutionHandler,
    create_default_dispatcher
)
from simulation.models.integration import ModelIntegrationLayer
from simulation.data.access import DataManager
from simulation.performance.optimization import AgentBatchProcessor, ModelCache
from simulation.validation.debugger import (
    SimulationDebugger, ModelValidator, ValidationReporter
)

logger = logging.getLogger(__name__)


class SimulationRunner:
    """
    High-level simulation runner.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.context = None
        self.scheduler = None
        self.dispatcher = None
        self.data_manager = None
        self.model_interface = None
        self.debugger = None
        self.validator = None
        
    def setup(self, config: Dict = None):
        """Set up simulation with configuration."""
        config = config or self.config
        
        # 1. Create context and scheduler
        self.context = SimulationContext()
        self.dispatcher = EventDispatcher()
        self.scheduler = Scheduler(self.context, None)
        self.scheduler.dispatcher = EventDispatcher()
        
        # 2. Register default handlers
        from simulation.handlers.stage_handlers import create_default_dispatcher
        self.dispatcher = create_default_dispatcher(self.context)
        
        # 2. Load data
        self.data_manager = DataManager()
        if 'data' in config:
            self.data_manager.initialize(config['data'])
        
        # 3. Load models
        from simulation.models.integration import ModelIntegrationLayer
        self.model_interface = ModelIntegrationLayer()
        if 'models' in config:
            self.load_models(config['models'])
        
        # 4. Set up debugger if needed
        if config.get('debug', False):
            from simulation.validation.debugger import SimulationDebugger
            self.debugger = SimulationDebugger(self.scheduler)
        
        # 5. Set up validator
        self.validator = ModelValidator({})
        
    def load_models(self, model_config: Dict):
        """Load models from configuration."""
        # This would load models from files/directory
        pass
    
    def create_agents_from_population(self, population_df: pd.DataFrame) -> List:
        """Create agents from synthetic population DataFrame."""
        agents = []
        for _, row in population_df.iterrows():
            # Create person attributes
            attrs = PersonAttributes(
                person_id=str(row.get('PERSON_ID', row.get('PERSONID', ''))),
                household_id=str(row.get('HHID', '')),
                age=int(row.get('AGE', 0)),
                gender=row.get('GENDER', 'M'),
                person_type=PersonType(row.get('PERSON_TYPE', 'other')),
                has_license=bool(row.get('CARLICENCE', 0)),
                vehicles=int(row.get('VEHICLES', 0)),
                household_size=int(row.get('HHSIZE', 1)),
                workers=int(row.get('WORKERS', 0)),
                income=float(row.get('INCOME', 0)),
                home_zone=str(row.get('HOME_ZONE', '')),
                work_zone=str(row.get('WORK_ZONE', '')),
                school_zone=str(row.get('SCHOOL_ZONE', '')),
            )
            
            agent = Agent(attributes=attrs)
            agents.append(agent)
        
        return agents
    
    def load_population(self, path: str) -> int:
        """Load synthetic population and create agents."""
        df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
        agents = self.create_agents_from_population(df)
        
        # Register agents with context
        for agent in agents:
            self.scheduler.context.register_agent(agent)
        
        logger.info(f"Loaded {len(agents)} agents")
        return len(agents)
    
    def run(self, max_time: float = None, max_events: int = None, 
            debug: bool = False) -> Dict[str, Any]:
        """
        Run the simulation.
        
        Args:
            max_time: Maximum simulation time (minutes)
            max_events: Maximum number of events to process
            debug: Enable debugging
            
        Returns:
            Simulation results dictionary
        """
        if not self.scheduler:
            raise RuntimeError("Simulation not set up. Call setup() first.")
        
        # Set up debugger if requested
        if debug and self.debugger:
            self.scheduler.add_callback(self._debug_callback)
        
        # Run simulation
        logger.info("Starting simulation...")
        results = self.scheduler.run(max_time=max_time, max_events=max_events)
        
        return results
    
    def _debug_callback(self, event: Event):
        """Debug callback for event processing."""
        if self.debugger and self.debugger.should_break(event):
            self.scheduler.running = False
            logger.info(f"Breakpoint hit: {event.event_type}")
    
    def run_step(self) -> Event:
        """Execute single simulation step."""
        if not self.scheduler.running:
            self.scheduler.running = True
        
        event = self.scheduler.queue.pop()
        if event:
            self.scheduler.current_time = event.time
            self.scheduler.context.current_time = event.time
            new_events = self.scheduler.dispatcher.dispatch(event, self.scheduler.context)
            for e in new_events:
                self.scheduler.schedule(e)
            return event
        return None
    
    def get_results(self) -> Dict:
        """Get simulation results."""
        return {
            'final_time': self.scheduler.current_time,
            'events_processed': self.scheduler._event_count,
            'agents': len(self.scheduler.context.agents),
            'final_queue_size': len(self.scheduler.queue),
            'stats': dict(self.scheduler.context.stats)
        }
    
    def save_results(self, output_path: str):
        """Save simulation results."""
        results = self.get_results()
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def validate(self) -> Dict:
        """Run validation on current state."""
        # Collect agent states
        agents_data = []
        for agent in self.context.agents.values():
            agents_data.append({
                'person_id': agent.agent_id,
                'person_type': agent.attributes.person_type.value,
                'age': agent.attributes.age,
                'is_worker': agent.attributes.is_worker,
                'is_student': agent.attributes.is_student,
                'auto_available': agent.attributes.auto_available,
            })
        
        validator = ModelValidator({})
        results = validator.run_all_validations(agents_data)
        
        reporter = ValidationReporter()
        for r in results:
            reporter.add_result(r)
        
        return {
            'passed': all(r.passed for r in results),
            'report': ValidationReporter().generate_report()
        }


def create_simulation_from_config(config_path: str) -> 'SimulationRunner':
    """Create simulation runner from config file."""
    import json
    with open(config) as f:
        config = json.load(f)
    runner = SimulationRunner(config)
    runner.setup()
    return runner


def create_exampville_simulation() -> 'SimulationRunner':
    """Create simulation configured for Exampville."""
    config = {
        'data': {
            'landuse_path': '../demand_/exampville_landuse.csv',
            'skim_path': '../demand_/exampville_skims.omx',
            'population_path': 'runs/exampville_ab/source/exampville_persons.csv'
        },
        'models': {
            'dir': 'runs/exampville_ab/models'
        },
        'zone_system': 40,
        'exampville': True
    }
    runner = SimulationRunner(config)
    runner.setup()
    return runner


def create_qld_simulation() -> 'SimulationRunner':
    """Create simulation for QLD 371-zone network."""
    config = {
        'data': {
            'landuse_path': '../new_pop/saved_data/zone_landuse_overture.csv',
            'skim_path': '../demand_/skims_371.omx',
            'population_path': '../new_pop/saved_data/persons.csv'
        },
        'models': {
            'dir': 'runs/ga_parallel/run06/merged'
        },
        'zone_system': 371,
        'exampville': False
    }
    runner = SimulationRunner(config)
    runner.setup()
    return runner


def create_debug_simulation(config: Dict = None) -> Tuple[object, object]:
    """Create simulation with debugging enabled."""
    from simulation.validation.debugger import SimulationDebugger
    
    runner = SimulationRunner(config)
    runner.setup()
    runner.debugger = SimulationDebugger(runner.scheduler)
    return runner, runner.debugger


def run_exampville_benchmark(config: Dict = None) -> Dict:
    """Run Exampville benchmark comparison."""
    # This would run the 3-arm comparison
    pass


def run_full_pipeline(config: Dict = None) -> Dict:
    """Run complete pipeline: GA search -> validation -> DTA export."""
    pass


def main():
    """Main entry point for running simulations from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ABM simulation')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--exampville', action='store_true', help='Run Exampville benchmark')
    parser.add_argument('--qld', action='store_true', help='Run QLD 371-zone simulation')
    parser.add_argument('--debug', action='store_true', help='Enable debugging')
    parser.add_argument('--max-time', type=float, help='Max simulation time (minutes)')
    parser.add_argument('--max-events', type=int, help='Max events to process')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    if args.exampville:
        runner = create_exampville_simulation()
    elif args.qld:
        runner = create_qld_simulation()
    else:
        runner = SimulationRunner()
    
    runner.setup()
    
    if args.debug:
        runner.debug = True
    
    results = runner.run(
        max_time=args.max_time,
        max_events=args.max_events,
        debug=args.debug
    )
    
    if args.output:
        runner.save_results(args.output)
    
    # Validate
    validation = runner.validate()
    print(f"Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
    
    return results


if __name__ == '__main__':
    main()