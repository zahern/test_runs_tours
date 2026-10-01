"""
Event Handlers for Each Modeling Stage
======================================
Integrates existing estimation models into event handlers.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
import numpy as np
import pandas as pd
import logging

from simulation.events.core import Event, EventType, EventPriority, EventHandler, SimulationContext, Event
from simulation.agents.agent import Agent, PersonAttributes, ActivityType, PersonType, Activity, Trip

logger = logging.getLogger(__name__)


class CDAPHandler:
    """Stage 0: Coordinated Daily Activity Pattern."""
    
    handled_event_types = {EventType.DAILY_PLAN_CREATION, EventType.PERSON_DAY_START}
    
    def __init__(self, model=None):
        self.model = model
        self.cdap_classes = ['M', 'N', 'H']  # Mandatory, Non-mandatory, Home
        
    def handle(self, event: Event, context: SimulationContext) -> List[Event]:
        new_events = []
        
        if event.event_type == EventType.PERSON_DAY_START:
            agent_id = event.agent_id
            context.agents[agent_id] = agent_id  # Ensure agent exists
            # Schedule CDAP choice
            new_events.append(Event(
                time=event.time,
                priority=EventPriority.HIGH,
                event_type=EventType.DAILY_PLAN_CREATION,
                agent_id=event.agent_id,
                data={'stage': 0}
            ))
            
        elif event.event_type == EventType.DAILY_PLAN_CREATION:
            agent_id = event.agent_id
            agent = context.agents.get(event.agent_id)
            if not agent:
                return []
            
            # Get CDAP choice from model
            cdap = self._choose_cdap(agent, context)
            
            # Schedule next stage based on CDAP
            if cdap == 'H':  # Home only
                # Schedule day end
                pass
            elif cdap == 'N':  # Non-mandatory only
                new_events.append(Event(
                    time=event.time,
                    priority=EventPriority.NORMAL,
                    event_type=EventType.ACTIVITY_SCHEDULING,
                    agent_id=event.agent_id,
                    data={'cdap': 'N', 'stage': 1}
                ))
            elif cdap == 'M':  # Mandatory + possibly non-mandatory
                new_events.append(Event(
                    time=event.time,
                    priority=EventPriority.NORMAL,
                    event_type=EventType.ACTIVITY_SCHEDULING,
                    agent_id=event.agent_id,
                    data={'cdap': 'M', 'stage': 1}
                ))
        
        return new_events
    
    def _choose_cdap(self, agent: Agent, context: SimulationContext) -> str:
        """Choose CDAP using model or fallback."""
        if self.model:
            # Use model prediction
            try:
                probs = self.model.predict_proba(agent.attributes)
                return np.random.choice(self.cdap_classes, p=probs)
            except:
                pass
        # Fallback: deterministic based on person type
        if agent.attributes.person_type in [PersonType.FULL_TIME_WORKER, PersonType.PART_TIME_WORKER]:
            return 'M'
        elif agent.attributes.person_type in [PersonType.UNIVERSITY_STUDENT, PersonType.SCHOOL_CHILD]:
            return 'M'
        return 'N'


class ActivityFrequencyHandler:
    """Stage 1: Activity Frequency Model."""
    
    handled_event_types = {EventType.ACTIVITY_SCHEDULING}
    
    PURPOSES = ['work', 'education', 'social', 'escort', 'shopping', 
                'other_maintenance', 'other_discretionary', 'eatout']
    
    def __init__(self, models: Dict = None):
        self.models = models or {}  # Dict of purpose -> model
    
    def handle(self, event: Event, context: SimulationContext) -> List[Event]:
        new_events = []
        agent = context.agents.get(event.agent_id)
        if not agent:
            return []
        
        cdap = event.data.get('cdap', 'N')
        
        # Determine which purposes to generate
        purposes = self._get_applicable_purposes(agent, cdap)
        
        # Generate frequency for each purpose
        activity_counts = {}
        for purpose in purposes:
            count = self._generate_frequency(agent, purpose)
            if count > 0:
                # Create activity generation events
                for i in range(count):
                    pass  # Will be handled by scheduling stage
        
        # Schedule duration modeling
        from simulation.events.core import Event, EventType, EventPriority
        return [
            Event(
                time=event.time,
                priority=EventPriority.NORMAL,
                event_type=EventType.ACTIVITY_SCHEDULING,  # Reuse for next stage
                agent_id=event.agent_id,
                data={
                    'stage': 2,  # Stage 2: Duration
                    'activity_counts': {}
                }
            )
        ]
    
    def _get_applicable_purposes(self, agent, cdap: str) -> List[str]:
        """Determine which purposes apply based on CDAP and person type."""
        person_type = agent.attributes.person_type
        purposes = []
        
        # Mandatory activities
        if cdap == 'M':
            if agent.attributes.is_worker:
                purposes.append('work')
            if agent.attributes.is_student:
                purposes.append('education')
        
        # Non-mandatory (for everyone with cdap != 'H')
        if cdap != 'H':
            purposes.extend(['social', 'escort', 'shopping', 'other_maintenance', 
                           'other_discretionary', 'eatout'])
        
        return purposes
    
    def _generate_frequency(self, agent, purpose: str) -> int:
        """Generate activity count for purpose."""
        model = self.model.get(purpose)
        if model:
            try:
                return int(model.predict(agent.attributes)[0])
            except:
                pass
        # Fallback: simple rules
        if purpose == 'work':
            return 1 if agent.attributes.is_worker else 0
        elif purpose == 'education':
            return 1 if agent.attributes.is_student else 0
        return np.random.poisson(0.5)


class ActivityDurationHandler:
    """Stage 2: Activity Duration Model (AFT Weibull)."""
    
    handled_event_types = {EventType.ACTIVITY_SCHEDULING}  # Reuse for stage 2
    
    def __init__(self, models: Dict = None):
        self.models = models or {}
    
    def handle(self, event: Event, context: SimulationContext) -> List[Event]:
        new_events = []
        agent_id = event.agent_id
        agent = context.agents.get(event.agent_id)
        
        if event.data.get('stage') == 2:
            # Generate durations for scheduled activities
            # This would be called after activities are scheduled
            
            # Schedule timing (Stage 3)
            from simulation.events.core import Event, EventType, EventPriority
            new_events.append(Event(
                time=event.time,
                priority=EventPriority.NORMAL,
                event_type=EventType.ACTIVITY_SCHEDULING,
                agent_id=event.agent_id,
                data={'stage': 3}  # Stage 3: Timing
            ))
        
        return new_events


class ScheduleTimingHandler:
    """Stage 3: Schedule Placement (AFT Weibull for start/end of day)."""
    
    handled_event_types = {EventType.ACTIVITY_SCHEDULING}
    
    def __init__(self, model=None):
        self.model = model
    
    def handle(self, event: Event, context: SimulationContext) -> List[Event]:
        new_events = []
        
        if event.data.get('stage') == 3:
            agent_id = event.agent_id
            agent = context.agents.get(event.agent_id)
            
            # Generate start/end of day using AFT model
            start_time, end_time = self._generate_day_bounds(agent)
            
            # Schedule mode/destination choice (Stage 5)
            from simulation.events.core import Event, EventType, EventPriority
            return [
                Event(
                    time=event.time,
                    priority=EventPriority.NORMAL,
                    event_type=EventType.MODE_CHOICE,
                    agent_id=event.agent_id,
                    data={
                        'stage': 5,
                        'day_start': start_time,
                        'day_end': end_time
                    }
                )
            ]
        
        return new_events
    
    def _generate_day_bounds(self, agent) -> Tuple[float, float]:
        """Generate start and end of day using AFT Weibull model."""
        # Default times
        start = 6 * 60  # 6:00 AM
        end = 22 * 60   # 10:00 PM
        
        if self.model:
            try:
                # Model predicts log-duration
                log_dur = self.model.predict(agent.attributes)
                duration = np.exp(log_dur[0])
                start = 6 * 60 + np.random.exponential(30)  # Some variation
                end = start + duration
            except:
                pass
        
        return max(0, min(start, 24*60)), min(24*60, max(end, start + 60))


class ModeDestinationHandler:
    """Stage 5: Joint Mode and Destination Choice."""
    
    handled_event_types = {EventType.MODE_CHOICE, EventType.DESTINATION_CHOICE}
    
    def __init__(self, mode_model=None, dest_model=None, ga_features=None):
        self.mode_model = mode_model
        self.dest_model = dest_model
        self.ga_features = ga_features or {}
    
    def handle(self, event: Event, context: SimulationContext) -> List[Event]:
        new_events = []
        
        if event.event_type == EventType.MODE_CHOICE:
            # Mode choice for each trip
            new_events.extend(self._handle_mode_choice(event, context))
        elif event.event_type == EventType.DESTINATION_CHOICE:
            new_events.extend(self._handle_destination_choice(event, context))
        
        return new_events
    
    def _handle_mode_choice(self, event: Event, context: SimulationContext) -> List[Event]:
        """Handle mode choice for each trip."""
        # This would call the mode choice model (nested logit)
        # For now, return destination choice events
        return [
            Event(
                time=event.time,
                priority=EventPriority.NORMAL,
                event_type=EventType.DESTINATION_CHOICE,
                agent_id=event.agent_id,
                data=event.data
            )
        ]
    
    def _handle_destination_choice(self, event: Event, context: SimulationContext) -> List[Event]:
        """Handle destination choice using MNL with sampling."""
        # This would call destination choice model
        # Return trip departure events
        return []


class TripExecutionHandler:
    """Stage 6: Trip Execution and Plan Finalization."""
    
    handled_event_types = {EventType.TRIP_DEPARTURE, EventType.TRIP_ARRIVAL, 
                           EventType.ACTIVITY_START, EventType.ACTIVITY_END}
    
    def handle(self, event: Event, context: SimulationContext) -> List[Event]:
        new_events = []
        
        if event.event_type == EventType.TRIP_DEPARTURE:
            # Start trip
            new_events.append(Event(
                time=event.time + event.data.get('travel_time', 30),
                priority=EventPriority.HIGH,
                event_type=EventType.TRIP_ARRIVAL,
                agent_id=event.agent_id,
                data=event.data
            ))
        
        elif event.event_type == EventType.TRIP_ARRIVAL:
            # Arrive at destination, start activity
            new_events.append(Event(
                time=event.time,
                priority=EventPriority.NORMAL,
                event_type=EventType.ACTIVITY_START,
                agent_id=event.agent_id,
                data=event.data
            ))
        
        elif event.event_type == EventType.ACTIVITY_START:
            # Schedule activity end
            duration = event.data.get('duration', 60)
            new_events.append(Event(
                time=event.time + duration,
                priority=EventPriority.NORMAL,
                event_type=EventType.ACTIVITY_END,
                agent_id=event.agent_id,
                data=event.data
            ))
        
        elif event.event_type == EventType.ACTIVITY_END:
            # Check if more activities/trips
            # If more, schedule next trip departure
            # If done, schedule day end
            pass
        
        return []


# Registry of all handlers
DEFAULT_HANDLERS = [
    ('cdap', CDAPHandler),
    ('activity_frequency', ActivityFrequencyHandler),
    ('activity_duration', ActivityDurationHandler),
    ('schedule_timing', ScheduleTimingHandler),
    ('mode_destination', ModeDestinationHandler),
    ('trip_execution', TripExecutionHandler),
]


def create_default_dispatcher(context: 'SimulationContext') -> 'EventDispatcher':
    """Create dispatcher with all default handlers."""
    from simulation.events.core import EventDispatcher
    
    dispatcher = EventDispatcher()
    
    # Register handlers
    dispatcher.register(CDAPHandler(), priority=10)
    dispatcher.register(ActivityFrequencyHandler(), priority=9)
    dispatcher.register(ActivityDurationHandler(), priority=8)
    dispatcher.register(ScheduleTimingHandler(), priority=7)
    dispatcher.register(ModeDestinationHandler(), priority=6)
    dispatcher.register(TripExecutionHandler(), priority=5)
    
    return dispatcher