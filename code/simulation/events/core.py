"""
Core Event-Driven Simulation Framework
=======================================
Event-driven simulation framework for activity-based travel demand modeling.
Inspired by BEAM/MATSim but extended for complex multi-model integration.
"""

from __future__ import annotations
import heapq
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic
from collections import defaultdict
import threading
import logging
import time as time_module

logger = logging.getLogger(__name__)

# Type aliases
Time = float  # Simulation time in minutes from midnight
AgentId = str
EventId = str
LocationId = str

T = TypeVar('T')


class EventType(Enum):
    """Types of events in the simulation."""
    # Activity lifecycle
    ACTIVITY_START = auto()
    ACTIVITY_END = auto()
    
    # Trip lifecycle
    TRIP_DEPARTURE = auto()
    TRIP_ARRIVAL = auto()
    TRIP_MODE_CHOICE = auto()
    TRIP_ROUTE_CHOICE = auto()
    
    # Planning/decision events
    DAILY_PLAN_CREATION = auto()
    ACTIVITY_SCHEDULING = auto()
    MODE_CHOICE = auto()
    DESTINATION_CHOICE = auto()
    ROUTE_CHOICE = auto()
    
    # System events
    SIMULATION_START = auto()
    SIMULATION_END = auto()
    PERSON_DAY_START = auto()
    PERSON_DAY_END = auto()
    
    # Model estimation/calibration
    MODEL_ESTIMATION = auto()
    MODEL_APPLICATION = auto()


class EventPriority(Enum):
    """Event priority levels (lower = higher priority)."""
    CRITICAL = 0    # System-critical, must execute immediately
    HIGH = 1        # Time-critical decisions
    NORMAL = 2      # Standard events
    LOW = 3         # Background/housekeeping
    DEFERRED = 4    # Can be delayed


@dataclass(order=True)
class Event:
    """
    Core event class.
    Ordered by (time, priority, sequence) for priority queue.
    """
    time: Time
    priority: EventPriority
    event_type: EventType
    agent_id: AgentId
    event_id: EventId = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    sequence: int = field(default=0, compare=False)
    created_at: float = field(default_factory=time_module.time, compare=False)
    
    def __post_init__(self):
        if not isinstance(self.priority, EventPriority):
            self.priority = EventPriority(self.priority)
        if not isinstance(self.event_type, EventType):
            self.event_type = EventType(self.event_type)
    
    def with_data(self, **kwargs) -> Event:
        """Create new event with additional data."""
        new_data = {**self.data, **kwargs}
        return Event(
            time=self.time,
            priority=self.priority,
            event_type=self.event_type,
            agent_id=self.agent_id,
            data=new_data,
            sequence=self.sequence
        )


class EventQueue:
    """
    Thread-safe priority queue for events.
    Orders by (time, priority, sequence).
    """
    
    def __init__(self):
        self._queue: List[Event] = []
        self._sequence = 0
        self._lock = threading.RLock()
        self._event_map: Dict[EventId, Event] = {}  # For lookup/cancellation
    
    def push(self, event: Event) -> EventId:
        """Add event to queue."""
        with self._lock:
            event.sequence = self._sequence
            self._sequence += 1
            heapq.heappush(self._queue, event)
            self._event_map[event.event_id] = event
            return event.event_id
    
    def pop(self) -> Optional[Event]:
        """Pop next event."""
        with self._lock:
            if not self._queue:
                return None
            event = heapq.heappop(self._queue)
            self._event_map.pop(event.event_id, None)
            return event
    
    def peek(self) -> Optional[Event]:
        """Peek at next event without removing."""
        with self._lock:
            return self._queue[0] if self._queue else None
    
    def remove(self, event_id: EventId) -> bool:
        """Remove specific event (mark as cancelled)."""
        with self._lock:
            if event_id in self._event_map:
                # Mark as cancelled by setting time to infinity
                event = self._event_map[event_id]
                event.time = float('inf')
                # Note: heapq doesn't support efficient removal, 
                # so we just let it pop and discard
                return True
            return False
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0


class EventHandler(ABC):
    """Abstract base for event handlers."""
    
    @abstractmethod
    def handle(self, event: Event, context: SimulationContext) -> List[Event]:
        """
        Handle an event, return new events to schedule.
        """
        pass
    
    @property
    @abstractmethod
    def handled_event_types(self) -> Set[EventType]:
        """Event types this handler processes."""
        pass
    
    def can_handle(self, event: Event) -> bool:
        return event.event_type in self.handled_event_types


class EventDispatcher:
    """
    Dispatches events to registered handlers.
    Supports priority ordering of handlers.
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, List[Tuple[int, EventHandler]]] = defaultdict(list)
    
    def register(self, handler: EventHandler, priority: int = 0):
        """Register handler for its event types."""
        for event_type in handler.handled_event_types:
            self._handlers[event_type].append((priority, handler))
            # Sort by priority (higher first)
            self._handlers[event_type].sort(key=lambda x: -x[0])
    
    def dispatch(self, event: Event, context: SimulationContext) -> List[Event]:
        """Dispatch event to all registered handlers."""
        new_events = []
        handlers = self._handlers.get(event.event_type, [])
        
        for _, handler in self._handlers.get(event.event_type, []):
            try:
                new_events.extend(handler.handle(event, context))
            except Exception as e:
                logger.error(f"Handler {handler.__class__.__name__} failed for {event.event_type}: {e}")
                raise
        
        return new_events


class SimulationContext:
    """
    Shared context passed to all event handlers.
    Contains simulation state, models, data access, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.current_time: Time = 0.0
        self.agents: Dict[AgentId, Agent] = {}
        self.models: Dict[str, Any] = {}  # Estimation models
        self.data_access: DataAccessLayer = None  # type: ignore
        self.rng = None  # Random number generator
        self.stats = defaultdict(int)
        self.metadata: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def get_model(self, name: str) -> Any:
        """Get model by name."""
        with self._lock:
            return self.models.get(name)
    
    def register_model(self, name: str, model: Any):
        """Register a model."""
        with self._lock:
            self.models[name] = model
    
    def get_agent(self, agent_id: AgentId) -> Optional[Agent]:
        with self._lock:
            return self.agents.get(agent_id)
    
    def register_agent(self, agent: Agent):
        with self._lock:
            self.agents[agent.agent_id] = agent
    
    def increment_stat(self, key: str, value: int = 1):
        with self._lock:
            self.stats[key] += value


class Scheduler:
    """
    Main simulation scheduler.
    Manages event queue, dispatches events, advances time.
    """
    
    def __init__(self, context: SimulationContext, dispatcher: EventDispatcher):
        self.context = context
        self.dispatcher = dispatcher
        self.queue = EventQueue()
        self.current_time: Time = 0.0
        self.running = False
        self.max_time: Optional[Time] = None
        self._event_count = 0
        self._max_events: Optional[int] = None
        self._callbacks: List[Callable[[Event], None]] = []
    
    def schedule(self, event: Event) -> EventId:
        """Schedule an event."""
        return self.queue.push(event)
    
    def schedule_at(self, event_type: EventType, agent_id: AgentId, 
                    time: Time, priority: EventPriority = EventPriority.NORMAL,
                    data: Dict = None) -> EventId:
        """Convenience method to schedule event at specific time."""
        event = Event(
            time=time,
            priority=EventPriority.NORMAL,
            event_type=event_type,
            agent_id=agent_id,
            data=data or {}
        )
        return self.schedule(event)
    
    def schedule_after(self, event_type: EventType, agent_id: AgentId,
                       delay: Time, priority: EventPriority = EventPriority.NORMAL,
                       data: Dict = None) -> EventId:
        """Schedule event after delay from current time."""
        return self.schedule_at(event_type, agent_id, self.current_time + delay, priority, data)
    
    def add_callback(self, callback: Callable[[Event], None]):
        """Add callback for each processed event."""
        self._callbacks.append(callback)
    
    def run(self, max_time: Optional[Time] = None, max_events: Optional[int] = None) -> Dict[str, Any]:
        """
        Run simulation until max_time or max_events reached.
        Returns statistics.
        """
        self.running = True
        self.max_time = max_time
        self._max_events = max_events
        self._event_count = 0
        start_time = time_module.time()
        
        logger.info(f"Starting simulation at time {self.current_time}")
        
        while self.running and not self.queue.is_empty():
            event = self.queue.pop()
            if event is None:
                break
            
            # Check time limit
            if self.max_time is not None and event.time > self.max_time:
                # Put back and stop
                self.queue.push(event)
                break
            
            # Check event limit
            if self._max_events is not None and self._event_count >= self._max_events:
                self.queue.push(event)
                break
            
            # Advance time
            self.current_time = event.time
            self.context.current_time = event.time
            
            # Execute callbacks
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Dispatch event
            new_events = self.dispatcher.dispatch(event, self.context)
            
            # Schedule new events
            for new_event in new_events:
                self.schedule(new_event)
            
            self._event_count += 1
            
            # Progress logging
            if self._event_count % 10000 == 0:
                logger.info(f"Processed {self._event_count} events, time={self.current_time:.1f}, queue={len(self.queue)}")
        
        elapsed = time_module.time() - start_time
        logger.info(f"Simulation ended: {self._event_count} events in {elapsed:.2f}s")
        
        return {
            'events_processed': self._event_count,
            'final_time': self.current_time,
            'elapsed_seconds': elapsed,
            'events_per_second': self._event_count / elapsed if elapsed > 0 else 0,
            'queue_remaining': len(self.queue),
            'stats': dict(self.context.stats)
        }
    
    def stop(self):
        """Stop simulation."""
        self.running = False
    
    def add_event_listener(self, callback: Callable[[Event], None]):
        self.add_callback(callback)


# Convenience function to create default simulation setup
def create_simulation(config: Dict[str, Any] = None) -> Tuple[SimulationContext, Scheduler]:
    """Create default simulation context and scheduler."""
    context = SimulationContext(config)
    dispatcher = EventDispatcher()
    scheduler = Scheduler(context, None)  # Will set dispatcher after
    dispatcher = EventDispatcher()
    scheduler.dispatcher = dispatcher
    return context, scheduler