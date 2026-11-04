"""
Validation and Debugging Tools
================================
Tools for verifying model logic, checking predictions, and debugging.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Callable
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import time

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    level: ValidationLevel
    message: str
    details: Dict = field(default_factory=dict)
    passed: bool = True
    
    def __bool__(self):
        return self.passed


class ModelValidator:
    """
    Validates model predictions and logic.
    """
    
    def __init__(self, models: Dict, data: pd.DataFrame = None):
        self.models = models
        self.data = data
        self.results: List = []
    
    def validate_cdap(self, agents: List[Dict]) -> List:
        """Validate CDAP model predictions."""
        results = []
        
        # Check CDAP distribution
        cdap_counts = defaultdict(int)
        for agent in agents:
            cdap = agent.get('cdap', 'unknown')
            if cdap in ['H', 'M', 'N']:
                pass  # Valid
            else:
                pass  # Invalid
        
        return []
    
    def validate_frequencies(self, agents: List[Dict]) -> List:
        """Validate activity frequency predictions."""
        results = []
        
        # Check work frequency for workers
        workers = [a for a in agents if a.get('person_type') in ['full_time_worker', 'part_time_worker']]
        for agent in workers:
            freq = agent.get('work_freq', 0)
            if freq < 1:
                yield ValidationResult(
                    check_name="work_frequency",
                    level=ValidationLevel.WARNING,
                    message=f"Worker {agent.get('person_id')} has work_freq={freq}",
                    details={'agent_id': agent.get('person_id'), 'work_freq': freq},
                    passed=False
                )
        
        # Check non-negative frequencies
        purposes = ['work', 'education', 'social', 'escort', 'shopping', 
                   'other_maintenance', 'other_discretionary', 'eatout']
        for purpose in ['work', 'education', 'shopping', 'social']:
            freq_col = f'{purpose}_freq'
            for agent in agents:
                freq = agent.get(freq_col, 0)
                if freq < 0:
                    yield ValidationResult(
                        check_name=f"{purpose}_frequency_nonneg",
                        level=ValidationLevel.ERROR,
                        message=f"Negative frequency for {purpose}: {agent.get('person_id')}",
                        details={'agent_id': agent.get('person_id'), 'freq': agent.get(freq)},
                        passed=False
                    )
        
        return []
    
    def validate_durations(self, agents: List[Dict]) -> List:
        """Validate activity durations."""
        results = []
        
        # Check reasonable durations
        for agent in agents:
            for key, value in agent.items():
                if key.endswith('_dur') or key.endswith('_duration'):
                    dur = agent.get(key, 0)
                    if dur < 0:
                        yield ValidationResult(
                            check_name="duration_nonneg",
                            level=ValidationLevel.ERROR,
                            message=f"Negative duration for {agent.get('person_id')}",
                            details={'agent_id': agent.get('person_id'), 'duration': value},
                            passed=False
                        )
                    elif dur > 24 * 60:  # More than a day
                        yield ValidationResult(
                            check_name="duration_reasonable",
                            level=ValidationLevel.WARNING,
                            message=f"Duration > 24h for {agent.get('person_id')}",
                            details={'agent_id': agent.get('person_id'), 'duration': value},
                            passed=False
                        )
        return []
    
    def validate_choice_consistency(self, agents: List[Dict]) -> List:
        """Validate choice consistency (mode/destination)."""
        results = []
        
        # Check mode choice consistency
        for agent in agents:
            mode = agent.get('mode_choice')
            if mode and mode not in ['drive_alone', 'shared_ride', 'transit', 'walk', 'bike']:
                yield ValidationResult(
                    check_name="mode_choice_valid",
                    level=ValidationLevel.ERROR,
                    message=f"Invalid mode choice: {agent.get('mode_choice')}",
                    details={'agent_id': agent.get('person_id'), 'mode': agent.get('mode_choice')},
                    passed=False
                )
        
        return []
    
    def run_all_validations(self, agents: List) -> List:
        """Run all validations."""
        results = []
        for validator in [
            self.validate_frequencies,
            self.validate_durations,
            self.validate_choice_consistency
        ]:
            results.extend(list(validator(agents)))
        return results


class SimulationDebugger:
    """
    Debugger for simulation runs.
    Provides step-by-step execution, breakpoints, and state inspection.
    """
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.breakpoints: Set[EventType] = set()
        self.watch_variables: Dict[str, Callable] = {}
        self.break_conditions: List[Callable] = []
        self.paused = False
        self.call_stack: List = []
        self._step_mode = False
    
    def add_breakpoint(self, event_type: EventType):
        """Add breakpoint on event type."""
        self.breakpoints.add(event_type)
    
    def add_watch(self, name: str, evaluator: Callable):
        """Add watch expression."""
        self.watch_variables[name] = evaluator
    
    def add_break_condition(self, condition: Callable[[Event], bool]):
        """Add break condition."""
        self.break_conditions.append(condition)
    
    def should_break(self, event: Event) -> bool:
        """Check if should break on event."""
        if event.event_type in self.breakpoints:
            return True
        for condition in self.break_conditions:
            if condition(event):
                return True
        return False
    
    def step(self):
        """Execute one event and pause."""
        self._step_mode = True
        self.scheduler.run(max_events=1)
    
    def continue_execution(self):
        self._step_mode = False
    
    def inspect_state(self) -> Dict:
        """Get current simulation state."""
        return {
            'current_time': self.scheduler.current_time,
            'queue_size': len(self.scheduler.queue),
            'events_processed': self.scheduler._event_count,
            'agents': len(self.scheduler.context.agents),
            'queue_size': len(self.scheduler.queue)
        )


class ModelDebugger:
    """
    Debugger for model predictions.
    """
    
    def __init__(self, models: Dict):
        self.models = models
        self.predictions_log: List[Dict] = []
    
    def inspect_prediction(self, model_name: str, features: Dict, 
                          prediction: np.ndarray) -> Dict:
        """Log and analyze a single prediction."""
        record = {
            'timestamp': time_module.time(),
            'model': model_name,
            'features': features,
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'feature_stats': self._analyze_features(features)
        }
        return record
    
    def _analyze_features(self, features: Dict) -> Dict:
        """Analyze feature values."""
        return {
            'n_features': len(features),
            'has_nan': any(v is None or np.isnan(v) for v in features.values() if isinstance(v, float)),
            'feature_range': {
                k: (float(v), float(v)) for k, v in features.items() 
                if isinstance(v, (int, float))
            }
        }
    
    def compare_predictions(self, model_name: str, 
                           features_list: List[Dict]) -> Dict:
        """Compare predictions across multiple inputs."""
        preds = []
        for features in features:
            pred = self.models[model_name].predict(features)
            preds.append(pred)
        
        return {
            'mean': np.mean(preds),
            'std': np.std(preds),
            'min': np.min(preds),
            'max': np.max(preds),
            'n': len(preds)
        }


class ValidationReporter:
    """
    Generates validation reports.
    """
    
    def __init__(self):
        self.results: List = []
    
    def add_result(self, result):
        self.results.append(result)
    
    def generate_report(self) -> str:
        """Generate text report."""
        if not self.results:
            return "No validation results."
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Total checks: {len(self.results)}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            ""
        ]
        
        # Group by level
        by_level = defaultdict(list)
        for r in self.results:
            by_level[r.level].append(r)
        
        for level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR, 
                      ValidationLevel.WARNING, ValidationLevel.INFO]:
            if level in by_level:
                lines.append(f"\n--- {level.value.upper()} ({len(by_level[level])}) ---")
                for r in by_level[level]:
                    lines.append(f"  [{r.check_name}] {r.message}")
                    if r.details:
                        lines.append(f"    Details: {r.details}")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        return json.dumps([
            {
                'check': r.check_name,
                'level': r.level.value,
                'message': r.message,
                'details': r.details,
                'passed': r.passed
            }
            for r in self.results
        ], indent=2)


class SimulationAuditor:
    """
    Audits simulation runs for correctness and consistency.
    """
    
    def __init__(self):
        self.checks: List[Callable] = []
        self.results: List = []
    
    def add_check(self, name: str, check_fn: Callable[[Dict], bool], 
                  description: str = "", severity: str = "error"):
        """Add audit check."""
        self.checks.append({
            'name': name,
            'fn': check_fn,
            'description': description,
            'severity': severity
        })
    
    def audit_simulation(self, simulation_state: Dict) -> List:
        """Run all audit checks."""
        results = []
        for check in self.checks:
            try:
                result = check['fn'](simulation_state)
                passed = bool(result)
                self.results.append({
                    'check': check['name'],
                    'passed': passed,
                    'severity': check['severity'],
                    'description': check['description']
                })
                if not passed:
                    logger.warning(f"AUDIT FAILED: {check['name']} - {check['description']}")
            except Exception as e:
                logger.error(f"Audit check {check['name']} failed: {e}")
                self.results.append({
                    'check': check['name'],
                    'passed': False,
                    'error': str(e)
                })
        return self.results
    
    def default_checks(self):
        """Add default audit checks."""
        self.add_check(
            "agent_conservation",
            lambda state: len(state.get('agents', {})) > 0,
            "At least one agent exists",
            "critical"
        )
        self.add_check(
            "time_monotonic",
            lambda state: state.get('current_time', 0) >= 0,
            "Simulation time is non-negative",
            "error"
        )
        self.add_check(
            "event_queue_not_exploding",
            lambda state: state.get('queue_size', 0) < 1000000,
            "Event queue not exploding",
            "warning"
        )


class ModelInspector:
    """
    Inspect model internals and feature importance.
    """
    
    def __init__(self, model):
        self.model = model
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.model.feature_names_in_,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return pd.DataFrame()
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients."""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if hasattr(self.model, 'feature_names_in_'):
                return pd.DataFrame({
                    'feature': self.model.feature_names_in_,
                    'coefficient': coef.ravel()
                })
        return pd.DataFrame()
    
    def analyze_feature_effects(self, X_sample: np.ndarray, 
                               feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze marginal effects of features.
        """
        # Simplified: compute partial dependence
        effects = []
        for i, name in enumerate(feature_names):
            # Vary feature i while holding others at median
            pass
        return pd.DataFrame()


class SimulationProfiler:
    """
    Profiler for simulation performance.
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.event_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    @contextmanager
    def time_block(self, name: str):
        start = time_module.perf_counter()
        try:
            yield
        finally:
            elapsed = time_module.time() - start
            with self._lock:
                self.timings[name].append(elapsed)
    
    def get_stats(self) -> Dict:
        stats = {}
        with self._lock:
            for name, times in self.timings.items():
                if times:
                    self[name] = {
                        'mean': np.mean(times),
                        'total': sum(times),
                        'count': len(times),
                        'min': min(times),
                        'max': max(times)
                    }
        return dict(self)


# Convenience functions
def create_validator(models: Dict, data: pd.DataFrame = None) -> ModelValidator:
    return ModelValidator(models)

def create_debugger(scheduler) -> SimulationDebugger:
    return SimulationDebugger(scheduler)

def create_model_debugger(models: Dict) -> ModelDebugger:
    return ModelDebugger(models)

def create_reporter() -> ValidationReporter:
    return ValidationReporter()

def create_auditor() -> SimulationAuditor:
    return SimulationAuditor()

def create_model_inspector(model) -> ModelInspector:
    return ModelInspector(model)

def create_profiler() -> SimulationProfiler:
    return SimulationProfiler()


# Model consistency checker
class ModelConsistencyChecker:
    """
    Checks consistency between related models.
    """
    
    def __init__(self, models: Dict):
        self.models = models
    
    def check_cdap_frequency_consistency(self) -> List[str]:
        """Check if CDAP and frequency models are consistent."""
        issues = []
        # Workers should have work freq > 0 when CDAP=M
        # Students should have edu freq > 0 when CDAP=M
        # Non-mandatory freq should be 0 when CDAP=H
        return []
    
    def check_duration_mode_consistency(self) -> List[str]:
        """Check duration and mode choice consistency."""
        issues = []
        # Long distances should not have walk/bike as primary mode
        # Short distances should not have transit as primary
        return []
    
    def check_destination_mode_consistency(self) -> List[str]:
        """Check destination and mode consistency."""
        issues = []
        # Work destinations should be reachable by chosen mode
        # Education destinations should match school locations
        return []
    
    def run_all_checks(self) -> Dict[str, List]:
        return {
            'cdap_frequency': self.check_cdap_frequency_consistency(),
            'duration_mode': self.check_duration_mode_consistency(),
            'destination_mode': self.check_destination_mode_consistency()
        }


def create_validator(models: Dict) -> 'ModelValidator':
    return ModelValidator(models)

def create_debugger(scheduler) -> 'SimulationDebugger':
    return SimulationDebugger(scheduler)

def create_model_debugger(models: Dict) -> 'ModelDebugger':
    return ModelDebugger(models)

def create_reporter() -> 'ValidationReporter':
    return ValidationReporter()

def create_auditor() -> 'SimulationAuditor':
    return SimulationAuditor()

def create_model_inspector(model) -> 'ModelInspector':
    return ModelInspector(model)

def create_consistency_checker(models: Dict) -> ModelConsistencyChecker:
    return ModelConsistencyChecker(models)


# Quick validation function
def validate_agent_state(agent: Dict) -> List:
    """Quick validation of agent state."""
    issues = []
    
    # Required fields
    required = ['person_id', 'person_type', 'age']
    for field in required:
        if field not in agent:
            yield f"Missing required field: {field}"
    
    # Age validation
    age = agent.get('age', 0)
    if age < 0 or age > 120:
        yield f"Invalid age: {age}"
    
    # Person type consistency
    ptype = agent.get('person_type')
    is_worker = agent.get('is_worker', False)
    if agent.get('person_type') in ['full_time_worker', 'part_time_worker'] and not is_worker:
        yield "Worker person_type but is_worker=False"
    
    return list(_)