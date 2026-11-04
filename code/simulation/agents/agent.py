"""
Agent and Plan Classes
======================
Core agent and plan representations for the simulation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, List, Tuple
from uuid import uuid4
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import timedelta


class ActivityType(Enum):
    """Activity types from the model."""
    HOME = "home"
    WORK = "work"
    EDUCATION = "education"
    SHOPPING = "shopping"
    ESCORT = "escort"
    SOCIAL = "social"
    RECREATION = "recreation"
    EATOUT = "eatout"
    OTHER_MAINTENANCE = "other_maintenance"
    OTHER_DISCRETIONARY = "other_discretionary"
    OTHER = "other"


class ModeType(Enum):
    """Travel mode types."""
    DRIVE_ALONE = "drive_alone"
    SHARED_RIDE = "shared_ride"
    TRANSIT = "transit"
    WALK = "walk"
    BIKE = "bike"


class PersonType(Enum):
    """Person types from CDAP model."""
    FULL_TIME_WORKER = "full_time_worker"
    PART_TIME_WORKER = "part_time_worker"
    UNIVERSITY_STUDENT = "university_student"
    SCHOOL_CHILD = "school_child"
    PRE_SCHOOL = "pre_school"
    RETIRED = "retired"
    HOME_MAKER = "home_maker"
    UNEMPLOYED = "unemployed"
    OTHER = "other"


@dataclass
class Location:
    """Location with zone and coordinates."""
    zone_id: str
    x: float = 0.0
    y: float = 0.0
    name: str = ""
    
    def __hash__(self):
        return hash(self.zone_id)


@dataclass
class Activity:
    """An activity in a plan."""
    activity_type: ActivityType
    location: Location
    start_time: float  # minutes from midnight
    end_time: float
    duration: float
    purpose: str = ""
    is_fixed: bool = False  # Cannot be moved/rescheduled
    attributes: Dict = field(default_factory=dict)
    
    @property
    def duration_minutes(self) -> float:
        return self.end_time - self.start_time


@dataclass
class Trip:
    """A trip between activities."""
    origin: Location
    destination: Location
    departure_time: float
    arrival_time: float
    mode: Optional[str] = None
    mode_type: Optional[str] = None  # ModeType
    distance: float = 0.0
    distance_km: float = 0.0
    travel_time: float = 0.0
    cost: float = 0.0
    purpose: str = ""
    tour_id: Optional[str] = None
    attributes: Dict = field(default_factory=dict)
    
    @property
    def travel_time_minutes(self) -> float:
        return self.arrival_time - self.departure_time


class PlanStatus(Enum):
    """Status of a plan."""
    EMPTY = "empty"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


@dataclass
class PersonAttributes:
    """Person socio-demographic attributes."""
    person_id: str
    household_id: str
    age: int
    gender: str  # 'M', 'F', 'O'
    person_type: PersonType
    has_license: bool = False
    vehicles: int = 0
    household_size: int = 1
    workers: int = 0
    income: float = 0.0
    has_disability: bool = False
    employment_status: str = ""
    industry: str = ""
    occupation: str = ""
    education_level: str = ""
    home_zone: str = ""
    work_zone: str = ""
    school_zone: str = ""
    home_location: Optional[object] = None
    work_location: Optional[object] = None
    school_location: Optional[object] = None
    
    # Derived/computed
    is_worker: bool = False
    is_student: bool = False
    auto_available: bool = False
    
    def __post_init__(self):
        self.is_worker = self.person_type in (
            PersonType.FULL_TIME_WORKER, PersonType.PART_TIME_WORKER
        )
        self.is_student = self.person_type in (
            PersonType.UNIVERSITY_STUDENT, PersonType.SCHOOL_CHILD, PersonType.PRE_SCHOOL
        )
        self.auto_available = self.has_license and self.vehicles > 0


@dataclass
class Plan:
    """
    A daily plan consisting of activities and trips.
    """
    person_id: str
    activities: List[object] = field(default_factory=list)  # Activity objects
    trips: List[object] = field(default_factory=list)  # Trip objects
    status: str = "empty"
    score: float = 0.0
    creation_time: float = 0.0
    last_modified: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    # Computed properties
    total_travel_time: float = 0.0
    total_distance: float = 0.0
    total_cost: float = 0.0
    n_trips: int = 0
    n_activities: int = 0
    
    def __post_init__(self):
        self._recompute()
    
    def _recompute(self):
        """Recompute derived attributes."""
        self.n_activities = len(self.activities)
        self.n_trips = len(self.trips)
        self.total_travel_time = sum(t.travel_time_minutes for t in self.trips)
        self.total_distance = sum(t.distance_km for t in self.trips)
        self.total_cost = sum(t.cost for t in self.trips)
    
    def add_activity(self, activity):
        self.activities.append(activity)
        self._recompute()
    
    def add_trip(self, trip):
        self.trips.append(trip)
        self._recompute()
    
    def get_activities_by_type(self, activity_type) -> List:
        return [a for a in self.activities if a.activity_type == activity]
    
    def get_tours(self) -> List[List]:
        """
        Extract tours (home-based tour chains).
        Returns list of tours, each tour is list of [activity, trip, activity, ...]
        """
        tours = []
        current_tour = []
        
        for item in self._flatten_plan():
            if isinstance(item, Activity):
                if item.activity_type == ActivityType.HOME:
                    if current_tour:
                        tours.append(current_tour)
                    current_tour = [item]
                else:
                    current_tour.append(item)
            elif isinstance(item, Trip):
                if current_tour:
                    current_tour.append(item)
        
        if current_tour:
            tours.append(current_tour)
        
        return tours
    
    def _flatten_plan(self) -> List:
        """Return interleaved list of activities and trips in chronological order."""
        events = []
        for a in self.activities:
            events.append((a.start_time, 'activity', a))
        for t in self.trips:
            events.append((t.departure_time, 'trip', t))
        events.sort(key=lambda x: x[0])
        return [e[2] for e in events]


@dataclass
class Agent:
    """
    Simulation agent (person).
    """
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: Any = None
    plans: List[object] = field(default_factory=list)  # Plan objects
    current_plan: Optional[object] = None  # Plan
    current_plan_index: int = 0
    
    # State tracking
    current_activity: Optional[object] = None
    current_trip: Optional[object] = None
    current_location: object = None
    current_time: float = 0.0
    
    # State machine
    state: str = "idle"  # idle, at_activity, traveling, waiting
    current_plan_index: int = 0
    current_activity_index: int = 0
    current_trip_index: int = 0
    
    # History/logging
    event_log: List = field(default_factory=list)
    completed_tours: List = field(default_factory=list)
    
    def __post_init__(self):
        self.agent_id = self.agent_id or str(uuid.uuid4())
    
    @property
    def person_id(self) -> str:
        return self.agent_id
    
    def add_plan(self, plan):
        plan.person_id = self.agent_id
        self.plans.append(plan)
        if self.current_plan is None:
            self.current_plan = plan
    
    def set_current_plan(self, plan):
        self.current_plan = plan
        self.current_plan_index = self.plans.index(plan) if plan in self.plans else 0
        self.current_activity_index = 0
        self.current_trip_index = 0
        self.current_time = 0.0
        self.state = "idle"
    
    def get_current_activity(self):
        if self.current_plan and self.current_plan.activities:
            idx = self.current_activity_index
            if 0 <= idx < len(self.current_plan.activities):
                return self.current_plan.activities[idx]
        return None
    
    def get_next_trip(self):
        if self.current_plan and self.current_trip_index < len(self.current_plan.trips):
            return self.current_plan.trips[self.current_trip_index]
        return None
    
    def advance_to_next_activity(self):
        """Move to next activity in plan."""
        if self.current_plan:
            self.current_activity_index += 1
            if self.current_activity_index < len(self.current_plan.activities):
                return self.current_plan.activities[self.current_activity_index]
        return None
    
    def advance_to_next_trip(self):
        """Move to next trip in plan."""
        if self.current_plan:
            self.current_trip_index += 1
            if self.current_trip_index < len(self.current_plan.trips):
                return self.current_plan.trips[self.current_trip_index]
        return None
    
    def is_day_complete(self) -> bool:
        """Check if day's plan is complete."""
        if not self.current_plan:
            return True
        return (self.current_activity_index >= len(self.current_plan.activities) - 1 and
                self.current_trip_index >= len(self.current_plan.trips))
    
    def log_event(self, event_type: str, data: dict = None):
        """Log event for debugging/replay."""
        self.event_log.append({
            'time': getattr(self, 'current_time', 0),
            'type': event_type,
            'data': data or {}
        })
    
    def get_current_state(self) -> dict:
        """Get current state snapshot."""
        return {
            'agent_id': self.agent_id,
            'current_time': self.current_time,
            'state': self.state,
            'current_activity': str(self.current_activity.activity_type) if self.current_activity else None,
            'current_trip_mode': self.current_trip.mode_type if self.current_trip else None,
            'plan_progress': f"{self.current_activity_index}/{len(self.current_plan.activities)}" if self.current_plan else "0/0"
        }


# Factory functions
def create_person_attributes(person_id: str, household_id: str, 
                            age: int, gender: str, person_type: PersonType,
                            **kwargs) -> PersonAttributes:
    """Create person attributes with defaults."""
    return PersonAttributes(
        person_id=person_id,
        household_id=household_id,
        age=age,
        gender=gender,
        person_type=person_type,
        **kwargs
    )


def create_agent(person_attrs: PersonAttributes) -> Agent:
    """Create agent from person attributes."""
    agent = Agent(
        agent_id=person_attrs.person_id,
        attributes=person_attrs
    )
    return agent


def create_activity(activity_type: ActivityType, location: Any, 
                   start_time: float, duration: float, **kwargs) -> object:
    """Create activity (using dynamic class to avoid import issues)."""
    from simulation.agents.agent import Activity as ActivityClass
    return ActivityClass(
        activity_type=activity_type,
        location=location,
        start_time=start_time,
        end_time=start_time + duration,
        duration=duration,
        **kwargs
    )


def create_trip(origin: object, destination: object, departure_time: float,
                travel_time: float, mode: str = None, **kwargs) -> object:
    """Create trip (using dynamic class)."""
    from simulation.agents.agent import Trip as TripClass
    return TripClass(
        origin=origin,
        destination=destination,
        departure_time=departure_time,
        arrival_time=departure_time + travel_time,
        mode=mode,
        travel_time=travel_time,
        **kwargs
    )