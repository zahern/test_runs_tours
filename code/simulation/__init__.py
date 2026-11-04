"""
Event-Driven Activity-Based Travel Simulation
==============================================
A modular, event-driven simulation framework for activity-based travel demand modeling.
Integrates with existing estimation models (Stages 0-6) and supports GPU/CPU acceleration.

Key Components:
- events/core.py: Core event system (Event, EventQueue, Scheduler, Dispatcher)
- agents/agent.py: Agent and Plan representations
- handlers/stage_handlers.py: Event handlers for each modeling stage (0-6)
- models/integration.py: Model integration layer
- data/access.py: Data access layer (landuse, skims, networks, population)
- performance/optimization.py: Performance optimization utilities
- validation/debugger.py: Debugging and validation tools
- runner.py: High-level simulation runner

Usage:
    from simulation.runner import SimulationRunner
    
    runner = SimulationRunner(config)
    runner.setup()
    runner.run(max_time=24*60)
"""

__version__ = "1.0.0"
__author__ = "ABM Simulation Team"

# Core exports
from simulation.events.core import (
    Event, EventType, EventPriority, EventQueue,
    SimulationContext, Scheduler, EventDispatcher, EventHandler,
    create_simulation
)

# Agent exports
from simulation.agents.agent import (
    Agent, PersonAttributes, Activity, Trip, Plan,
    ActivityType, ModeType, PersonType,
    create_agent, create_person_attributes
)

# Handler exports
from simulation.handlers.stage_handlers import (
    CDAPHandler, ActivityFrequencyHandler, ActivityDurationHandler,
    ScheduleTimingHandler, ModeDestinationHandler, TripExecutionHandler,
    create_default_dispatcher
)

# Model integration
from simulation.models.integration import (
    ModelIntegrationLayer, ModelRegistry, ModelWrapper,
    ModelInterface, get_model_interface
)

# Data access
from simulation.data.access import (
    DataAccessLayer, DataManager, SkimManager, NetworkManager,
    DataManager, get_data_manager
)

# Performance
from simulation.performance.optimization import (
    VectorizedModelApplier, AgentBatchProcessor, ModelCache,
    AgentBatchGenerator, parallel_agent_processing, CachedModelWrapper,
    EventBatchProcessor, PerformanceMonitor, timed, profile
)

# Validation
from simulation.validation.debugger import (
    SimulationDebugger, ModelValidator, ValidationReporter,
    SimulationAuditor, ModelInspector, SimulationProfiler,
    ModelConsistencyChecker
)

# Runner
from simulation.runner import (
    SimulationRunner, create_simulation_from_config,
    create_exampville_simulation, create_qld_simulation,
    create_debug_simulation, run_exampville_benchmark,
    run_full_pipeline, SimulationRunner
)

__all__ = [
    # Core
    'Event', 'EventType', 'EventPriority', 'EventQueue',
    'SimulationContext', 'Scheduler', 'EventDispatcher', 'EventHandler',
    'create_simulation',
    # Agents
    'Agent', 'PersonAttributes', 'Activity', 'Trip', 'Plan',
    'ActivityType', 'ModeType', 'PersonType',
    'create_agent', 'create_person_attributes',
    # Handlers
    'CDAPHandler', 'ActivityFrequencyHandler', 'ActivityDurationHandler',
    'ScheduleTimingHandler', 'ModeDestinationHandler', 'TripExecutionHandler',
    'create_default_dispatcher',
    # Models
    'ModelIntegrationLayer', 'ModelRegistry', 'ModelWrapper',
    'ModelInterface', 'get_model_interface',
    # Data
    'DataAccessLayer', 'DataManager', 'SkimManager', 'NetworkManager',
    'get_data_manager',
    # Performance
    'VectorizedModelApplier', 'AgentBatchProcessor', 'ModelCache',
    'AgentBatchGenerator', 'parallel_agent_processing', 'CachedModelWrapper',
    'EventBatchProcessor', 'PerformanceMonitor', 'timed', 'profile',
    # Validation
    'SimulationDebugger', 'ModelValidator', 'ValidationReporter',
    'SimulationAuditor', 'ModelInspector', 'SimulationProfiler',
    'ModelConsistencyChecker',
    # Runner
    'SimulationRunner', 'create_simulation_from_config',
    'create_exampville_simulation', 'create_qld_simulation',
    'create_debug_simulation', 'run_exampville_benchmark',
    'run_full_pipeline'
]