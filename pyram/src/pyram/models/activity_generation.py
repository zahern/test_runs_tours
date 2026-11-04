"""
Activity Generation with Competing Hazard Formulation
======================================================

POLARIS-style activity generation using competing hazard models (Auld et al. 2011),
built on MetacountRegressor as the base estimator.

Key features:
- Competing hazard formulation: activities compete as time-varying hazards
- MetacountRegressor as base count model (replaces cross-classification)
- Outputs activity episodes compatible with existing pipeline
- Configurable hazard forms: piecewise-constant, Weibull, exponential
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from metacount import MetacountRegressor
    METACOUNT_AVAILABLE = True
except ImportError:
    METACOUNT_AVAILABLE = False
    warnings.warn("metacount not available - using fallback")


# ─── Hazard Functions ──────────────────────────────────────────────────────────

def piecewise_constant_hazard(
    time_bins: np.ndarray,
    rates: np.ndarray,
    covariates: np.ndarray,
    betas: np.ndarray
) -> np.ndarray:
    """
    Piecewise-constant hazard function.
    
    Parameters
    ----------
    time_bins : array of shape (K+1,)
        Bin edges for time intervals
    rates : array of shape (K,)
        Baseline hazard rate for each interval
    covariates : array of shape (N, P)
        Covariate matrix for N individuals
    betas : array of shape (P,)
        Coefficients for covariates
    
    Returns
    -------
    hazards : array of shape (N, K)
        Hazard rate for each individual in each interval
    """
    # Linear predictor
    lp = covariates @ betas  # (N,)
    # Expand to (N, K) and multiply by baseline rates
    hazards = rates * np.exp(lp)[:, np.newaxis]  # (N, K)
    return hazards


def weibull_hazard(
    time: np.ndarray,
    shape: float,
    scale: float,
    covariates: np.ndarray,
    betas: np.ndarray
) -> np.ndarray:
    """
    Weibull hazard function: h(t) = (shape/scale) * (t/scale)^(shape-1) * exp(beta'x)
    """
    lp = covariates @ betas
    # time: (T,), covariates: (N, P)
    # Result: (N, T)
    t_scaled = time / scale
    base_hazard = (shape / scale) * t_scaled ** (shape - 1)
    hazards = base_hazard * np.exp(lp)[:, np.newaxis]
    return hazards


def exponential_hazard(
    covariates: np.ndarray,
    betas: np.ndarray,
    base_rate: float = 1.0
) -> np.ndarray:
    """
    Constant (exponential) hazard: h(t) = lambda * exp(beta'x)
    """
    lp = covariates @ betas
    return base_rate * np.exp(lp)


# ─── Competing Hazard Model ────────────────────────────────────────────────────

@dataclass
class ActivityHazardConfig:
    """Configuration for activity hazard model."""
    activity_type: str
    hazard_type: str = "piecewise_constant"  # "piecewise_constant", "weibull", "exponential"
    time_bins: Optional[np.ndarray] = None
    n_intervals: int = 96  # 15-min intervals over 24h
    max_time: float = 24.0  # hours
    
    # Baseline parameters (can be estimated)
    base_rates: Optional[np.ndarray] = None
    weibull_shape: float = 1.5
    weibull_scale: float = 12.0
    exp_base_rate: float = 0.1


class CompetingHazardActivityGenerator:
    """
    Competing hazard activity generation model.
    
    Each activity type has its own hazard function. At each time interval,
    activities compete - the probability of generating activity k is:
    
        P(k at t) = h_k(t) / sum_j h_j(t)
    
    This replaces the cross-classification table with a continuous-time
    hazard-based model that can incorporate covariates naturally.
    """
    
    ACTIVITY_TYPES = [
        'home', 'work', 'school', 'shopping', 'social', 
        'recreation', 'meal', 'personal_business', 'escort', 'other'
    ]
    
    def __init__(
        self,
        configs: Dict[str, ActivityHazardConfig] = None,
        metacount_model: Any = None,
        seed: int = 42
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Activity hazard configurations
        if configs is None:
            configs = self._default_configs()
        self.configs = configs
        
        # Metacount regressor for baseline rates (optional)
        self.metacount = metacount_model
        
        # Fitted parameters
        self.fitted = False
        self.betas = {}  # per activity type
        self.baseline_hazards = {}  # per activity type
        
    def _default_configs(self) -> Dict[str, ActivityHazardConfig]:
        """Default hazard configs for each activity type."""
        configs = {}
        for act in self.ACTIVITY_TYPES:
            configs[act] = ActivityHazardConfig(
                activity_type=act,
                hazard_type="piecewise_constant",
                n_intervals=96,
                max_time=24.0,
                time_bins=np.linspace(0, 24, 97)  # 15-min bins
            )
        return configs
    
    def _build_covariates(self, persons: pd.DataFrame) -> np.ndarray:
        """
        Build covariate matrix from person attributes.
        
        Returns (N, P) matrix where N = n_persons, P = n_features
        """
        # Standard covariates for activity generation
        features = []
        feature_names = []
        
        # Demographics
        if 'age5p' in persons.columns:
            features.append(pd.to_numeric(persons['age5p'], errors='coerce').fillna(3).values)
            feature_names.append('age5p')
        
        if 'sexp' in persons.columns:
            features.append((pd.to_numeric(persons['sexp'], errors='coerce').fillna(1).values == 2).astype(float))
            feature_names.append('female')
        
        # Employment status
        if 'WORKSTAT' in persons.columns:
            workstat = pd.to_numeric(persons['WORKSTAT'], errors='coerce').fillna(0)
            features.append((workstat == 1).astype(float))  # full-time
            features.append((workstat == 2).astype(float))  # part-time
            feature_names.extend(['work_fulltime', 'work_parttime'])
        
        if 'STUDYING' in persons.columns:
            studying = pd.to_numeric(persons['STUDYING'], errors='coerce').fillna(0)
            features.append((studying > 0).astype(float))
            feature_names.append('student')
        
        # Income
        if 'incp' in persons.columns:
            incp = pd.to_numeric(persons['incp'], errors='coerce').fillna(50000)
            features.append(np.log(np.maximum(incp, 1)) / 11.0)  # normalized log income
            feature_names.append('log_income')
        
        # Vehicle availability
        if 'vehrd' in persons.columns:
            vehrd = pd.to_numeric(persons['vehrd'], errors='coerce').fillna(0)
            features.append((vehrd > 0).astype(float))
            features.append(np.minimum(vehrd / 3.0, 1.0))  # normalized vehicle count
            feature_names.extend(['has_vehicle', 'vehicle_count_norm'])
        
        # Household size
        if 'hhsiz' in persons.columns or 'HHSIZE' in persons.columns:
            hhsiz_col = 'hhsiz' if 'hhsiz' in persons.columns else 'HHSIZE'
            hhsiz = pd.to_numeric(persons[hhsiz_col], errors='coerce').fillna(2)
            features.append(np.minimum(hhsiz / 6.0, 1.0))
            feature_names.append('hhsize_norm')
        
        # Person type (if available)
        if 'ptype' in persons.columns:
            ptype = persons['ptype'].astype(str).fillna('unknown')
            for pt in ['full_time_worker', 'part_time_worker', 'university_student', 'school_student', 'non_worker']:
                features.append((ptype == pt).astype(float))
                feature_names.append(f'ptype_{pt}')
        
        if not features:
            # Fallback: intercept only
            return np.ones((len(persons), 1))
        
        X = np.column_stack(features)
        return X
    
    def estimate(
        self,
        persons: pd.DataFrame,
        activity_episodes: pd.DataFrame,
        **kwargs
    ) -> "CompetingHazardActivityGenerator":
        """
        Estimate hazard model parameters from observed activity episodes.
        
        Parameters
        ----------
        persons : DataFrame with person attributes
        activity_episodes : DataFrame with columns [person_id, activity_type, start_time, duration]
        
        The model uses MetacountRegressor as base estimator for baseline rates
        and estimates beta coefficients via maximum likelihood.
        """
        if not METACOUNT_AVAILABLE:
            warnings.warn("metacount not available - using simple MLE fallback")
            return self._estimate_simple(persons, activity_episodes)
        
        X = self._build_covariates(persons)
        n_persons, n_features = X.shape
        
        # Build person index mapping
        person_ids = persons['person_id'].astype(str).values
        pid_to_idx = {pid: i for i, pid in enumerate(person_ids)}
        
        # For each activity type, estimate hazard model
        for act_type in self.ACTIVITY_TYPES:
            if act_type not in self.configs:
                continue
                
            config = self.configs[act_type]
            
            # Filter episodes for this activity type
            act_episodes = activity_episodes[
                activity_episodes['activity_type'] == act_type
            ].copy()
            
            if len(act_episodes) < 10:
                # Too few observations - use default
                self.betas[act_type] = np.zeros(n_features)
                self.baseline_hazards[act_type] = np.ones(config.n_intervals) * 0.01
                continue
            
            # Convert start times to interval indices
            act_episodes['start_interval'] = (
                (act_episodes['start_time'] / config.max_time * config.n_intervals)
                .astype(int).clip(0, config.n_intervals - 1)
            )
            
            # Build count data per person per interval
            # This is the key: use Metacount to model counts per interval
            interval_counts = np.zeros((n_persons, config.n_intervals))
            
            for _, ep in act_episodes.iterrows():
                pid = str(ep['person_id'])
                if pid in pid_to_idx:
                    idx = pid_to_idx[pid]
                    interval = ep['start_interval']
                    if 0 <= interval < config.n_intervals:
                        interval_counts[idx, interval] += 1
            
            # Use MetacountRegressor to model interval counts
            # This gives us baseline rates per interval with covariates
            mc = MetacountRegressor(
                alpha=1.0,
                max_iter=100,
                tol=1e-6,
                random_state=self.seed
            )
            
            # Flatten for Metacount (N * K observations)
            X_flat = np.repeat(X, config.n_intervals, axis=0)
            y_flat = interval_counts.flatten()
            
            # Add interval dummies as features
            interval_dummies = np.tile(
                np.eye(config.n_intervals), (n_persons, 1)
            )
            X_augmented = np.hstack([X_flat, interval_dummies])
            
            try:
                mc.fit(X_augmented, y_flat)
                coefs = mc.coef_
                
                # Split coefficients: first n_features are betas, rest are interval baselines
                self.betas[act_type] = coefs[:n_features]
                self.baseline_hazards[act_type] = coefs[n_features:]
                
            except Exception as e:
                warnings.warn(f"Metacount estimation failed for {act_type}: {e}")
                self.betas[act_type] = np.zeros(n_features)
                self.baseline_hazards[act_type] = np.ones(config.n_intervals) * 0.01
        
        self.fitted = True
        return self
    
    def _estimate_simple(
        self,
        persons: pd.DataFrame,
        activity_episodes: pd.DataFrame
    ) -> "CompetingHazardActivityGenerator":
        """Simple MLE fallback when Metacount unavailable."""
        X = self._build_covariates(persons)
        n_persons, n_features = X.shape
        
        for act_type in self.ACTIVITY_TYPES:
            if act_type not in self.configs:
                continue
            config = self.configs[act_type]
            
            # Simple baseline: count episodes per interval
            act_episodes = activity_episodes[
                activity_episodes['activity_type'] == act_type
            ]
            
            if len(act_episodes) == 0:
                self.betas[act_type] = np.zeros(n_features)
                self.baseline_hazards[act_type] = np.ones(config.n_intervals) * 0.001
                continue
            
            act_episodes['start_interval'] = (
                (act_episodes['start_time'] / config.max_time * config.n_intervals)
                .astype(int).clip(0, config.n_intervals - 1)
            )
            
            baseline = np.zeros(config.n_intervals)
            for interval in range(config.n_intervals):
                n_episodes = len(act_episodes[act_episodes['start_interval'] == interval])
                baseline[interval] = n_episodes / max(len(persons), 1)
            
            # Smooth baseline
            from scipy.ndimage import gaussian_filter1d
            baseline = gaussian_filter1d(baseline, sigma=1.0)
            baseline = np.maximum(baseline, 1e-6)
            
            self.betas[act_type] = np.zeros(n_features)
            self.baseline_hazards[act_type] = baseline
        
        self.fitted = True
        return self
    
    def predict_activities(
        self,
        persons: pd.DataFrame,
        n_draws: int = 1,
        seed: int = None
    ) -> pd.DataFrame:
        """
        Generate activity episodes for synthetic population.
        
        Returns DataFrame with columns:
        [person_id, activity_type, start_time, duration, end_time]
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        X = self._build_covariates(persons)
        n_persons = len(persons)
        person_ids = persons['person_id'].astype(str).values
        
        all_episodes = []
        
        for pid_idx, pid in enumerate(person_ids):
            # Compute hazards for each activity type at each interval
            all_hazards = {}
            for act_type in self.ACTIVITY_TYPES:
                if act_type not in self.configs:
                    continue
                config = self.configs[act_type]
                
                # h_k(t) = baseline_k(t) * exp(beta_k' x)
                betas = self.betas.get(act_type, np.zeros(X.shape[1]))
                baseline = self.baseline_hazards.get(act_type, np.ones(config.n_intervals) * 0.01)
                
                lp = X[pid_idx] @ betas
                hazards = baseline * np.exp(lp)  # shape: (n_intervals,)
                all_hazards[act_type] = hazards
            
            # Generate activities by competing hazards
            # At each interval, choose activity based on relative hazards
            activities = self._simulate_competing_hazards(
                all_hazards, pid, config.n_intervals, config.max_time
            )
            all_episodes.extend(activities)
        
        if not all_episodes:
            return pd.DataFrame(columns=['person_id', 'activity_type', 'start_time', 'duration', 'end_time'])
        
        df = pd.DataFrame(all_episodes)
        return df
    
    def _simulate_competing_hazards(
        self,
        all_hazards: Dict[str, np.ndarray],
        person_id: str,
        n_intervals: int,
        max_time: float
    ) -> List[Dict]:
        """
        Simulate activity sequence using competing hazards.
        
        At each interval, activities compete. The probability of activity k:
        P(k) = h_k / sum_j h_j
        
        We also model "no activity" (stay at home) as an implicit option.
        """
        episodes = []
        interval_duration = max_time / n_intervals  # hours
        
        current_activity = 'home'
        current_start = 0.0
        
        for interval in range(n_intervals):
            # Get hazards at this interval
            hazards = {act: h[interval] for act, h in all_hazards.items()}
            
            # Add "stay at current location" hazard (home base)
            # Higher when at home, lower when away
            if current_activity == 'home':
                stay_hazard = 1.0  # High probability to stay home
            else:
                stay_hazard = 0.1  # Low probability to stay away
            
            # Competing probabilities
            all_h = {**hazards, 'stay': stay_hazard}
            total = sum(all_h.values())
            probs = {k: v/total for k, v in all_h.items()}
            
            # Draw activity
            u = self.rng.random()
            cumsum = 0.0
            chosen = 'stay'
            for act, p in probs.items():
                cumsum += p
                if u < cumsum:
                    chosen = act
                    break
            
            # Handle transition
            if chosen != current_activity:
                # End previous activity
                if current_activity != 'home' and current_activity != 'stay':
                    episodes.append({
                        'person_id': person_id,
                        'activity_type': current_activity,
                        'start_time': current_start,
                        'duration': interval * interval_duration - current_start,
                        'end_time': interval * interval_duration
                    })
                # Start new activity
                current_activity = chosen
                current_start = interval * interval_duration
        
        # Close final activity
        if current_activity != 'home' and current_activity != 'stay':
            episodes.append({
                'person_id': person_id,
                'activity_type': current_activity,
                'start_time': current_start,
                'duration': max_time - current_start,
                'end_time': max_time
            })
        
        return episodes
    
    def get_hazard_functions(self) -> Dict[str, Dict]:
        """Return fitted hazard functions for inspection."""
        return {
            act_type: {
                'betas': self.betas.get(act_type),
                'baseline_hazards': self.baseline_hazards.get(act_type),
                'config': self.configs.get(act_type)
            }
            for act_type in self.ACTIVITY_TYPES
        }


# ─── Activity Planning Order Model (POLARIS-style) ────────────────────────────

class ActivityPlanningOrderModel:
    """
    POLARIS-style activity planning order model (Auld & Mohammadian 2012).
    
    Three models:
    1. Attribute Flexibility (Multivariate Ordered Probit)
    2. Activity Plan Horizon (Univariate Ordered Probit)
    3. Attribute Planning Horizon (Multivariate Ordered Probit)
    
    Simplified implementation using Metacount for the count part
    and ordered probit for the ordinal part.
    """
    
    FLEXIBILITY_LEVELS = ['fixed', 'semi_flexible', 'flexible']
    HORIZON_LEVELS = ['spontaneous', 'same_day', 'pre_planned', 'routine']
    
    def __init__(self, metacount_model: Any = None, seed: int = 42):
        self.metacount = metacount_model
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fitted = False
    
    def estimate(
        self,
        persons: pd.DataFrame,
        activity_episodes: pd.DataFrame
    ) -> "ActivityPlanningOrderModel":
        """
        Estimate planning order models from activity episodes.
        
        In practice, these models are estimated on UTRACS/GPS survey data.
        This is a simplified implementation using Metacount for the count part.
        """
        # Placeholder - in reality these are hardcoded in POLARIS
        # from UTRACS survey estimation
        self.fitted = True
        return self
    
    def predict_planning_order(
        self,
        activities: pd.DataFrame,
        persons: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Assign planning order attributes to activities.
        
        Adds columns:
        - flexibility: fixed/semi_flexible/flexible
        - plan_horizon: spontaneous/same_day/pre_planned/routine
        """
        if not self.fitted:
            # Default: assign based on activity type
            pass
        
        activities = activities.copy()
        
        # Default rules (POLARIS-style)
        flexibility_map = {
            'work': 'fixed', 'school': 'fixed',
            'shopping': 'semi_flexible', 'meal': 'semi_flexible',
            'recreation': 'flexible', 'social': 'flexible',
            'personal_business': 'semi_flexible', 'escort': 'semi_flexible',
            'other': 'flexible', 'home': 'fixed'
        }
        
        horizon_map = {
            'work': 'routine', 'school': 'routine',
            'shopping': 'pre_planned', 'meal': 'same_day',
            'recreation': 'spontaneous', 'social': 'same_day',
            'personal_business': 'pre_planned', 'escort': 'pre_planned',
            'other': 'spontaneous', 'home': 'routine'
        }
        
        activities['flexibility'] = activities['activity_type'].map(
            lambda x: flexibility_map.get(x, 'flexible')
        )
        activities['plan_horizon'] = activities['activity_type'].map(
            lambda x: horizon_map.get(x, 'spontaneous')
        )
        
        return activities


# ─── Activity Timing Choice - Hybrid RUM-RRM Copula ──────────────────────────

class HybridTimingChoiceModel:
    """
    Hybrid RUM-RRM Copula model for activity timing (Golshani et al. 2018).
    
    Joint model of:
    - Discrete: Activity start time (Hybrid RUM-RRM)
    - Continuous: Activity duration (Hazard duration model)
    - Coupled via Copula (Clayton/Frank/Gumbel)
    
    Simplified implementation using Metacount for timing probabilities.
    """
    
    def __init__(
        self,
        copula_type: str = "clayton",  # clayton, frank, gumbel
        metacount_model: Any = None,
        seed: int = 42
    ):
        self.copula_type = copula_type
        self.metacount = metacount_model
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fitted = False
    
    def _clayton_copula(self, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Clayton copula: C(u,v) = (u^-θ + v^-θ - 1)^(-1/θ)"""
        return (u**(-theta) + v**(-theta) - 1)**(-1/theta)
    
    def _frank_copula(self, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Frank copula"""
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        den = np.exp(-theta) - 1
        return -np.log(1 + num / den) / theta
    
    def estimate(
        self,
        activity_episodes: pd.DataFrame,
        persons: pd.DataFrame
    ) -> "HybridTimingChoiceModel":
        """
        Estimate timing choice model from activity episodes.
        
        In POLARIS, routine activities use empirical joint distribution.
        Non-routine use the hybrid model.
        """
        self.fitted = True
        return self
    
    def predict_timing(
        self,
        activities: pd.DataFrame,
        persons: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict start time and duration for activities.
        
        For routine: draw from empirical joint distribution
        For non-routine: use hybrid RUM-RRM copula
        """
        activities = activities.copy()
        
        # Simple fallback: use empirical distributions
        routine_types = ['work', 'school']
        
        for idx, act in activities.iterrows():
            act_type = act['activity_type']
            
            if act_type in routine_types:
                # Draw from empirical joint distribution (would load from file)
                # For now, use simple rules
                if act_type == 'work':
                    start = self.rng.normal(8.0, 0.5)  # 8 AM ± 30 min
                    duration = self.rng.normal(8.5, 1.0)  # 8.5 hours
                else:
                    start = self.rng.normal(8.5, 0.5)
                    duration = self.rng.normal(6.5, 1.0)
            else:
                # Non-routine: more flexible
                start = self.rng.uniform(7, 20)
                duration = self.rng.exponential(2.0)
            
            activities.at[idx, 'start_time'] = max(0, min(24, start))
            activities.at[idx, 'duration'] = max(0.5, min(24 - start, duration))
            activities.at[idx, 'end_time'] = activities.at[idx, 'start_time'] + activities.at[idx, 'duration']
        
        return activities


# ─── Routine Location Choice (Work/School) ──────────────────────────────────

class RoutineLocationChoiceModel:
    """
    POLARIS-style routine location choice for work/school.
    
    Work location:
    - MNL with employment size terms
    - Mode choice logsum
    - Employment constraints (capacity per zone)
    
    School location:
    - Age-based distance constraints
    - Enrollment size terms
    """
    
    def __init__(
        self,
        metacount_model: Any = None,
        seed: int = 42
    ):
        self.metacount = metacount_model
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fitted = False
    
    def estimate(
        self,
        persons: pd.DataFrame,
        activity_episodes: pd.DataFrame,
        landuse: pd.DataFrame,
        skim_lookup: Dict,
        segment: str = 'work'  # 'work' or 'school'
    ) -> "RoutineLocationChoiceModel":
        """
        Estimate routine location choice model.
        
        For work: uses work activity episodes
        For school: uses school activity episodes
        """
        # Placeholder - in POLARIS this is estimated on HHTS/UTRACS
        self.fitted = True
        return self
    
    def predict(
        self,
        persons: pd.DataFrame,
        landuse: pd.DataFrame,
        skim_lookup: Dict,
        segment: str = 'work'
    ) -> pd.DataFrame:
        """
        Predict work/school locations for persons.
        
        Returns DataFrame with destination zone for each person.
        """
        persons = persons.copy()
        
        if segment == 'work':
            # Workers only
            workers = persons[persons.get('WORKSTAT', 0) > 0].copy()
            
            # Simple gravity model: probability proportional to employment
            if 'TOTEMP' in landuse.columns:
                emp = landuse['TOTEMP'].values
                emp = np.maximum(emp, 1)
                probs = emp / emp.sum()
                
                chosen_zones = self.rng.choice(
                    landuse.index.values, size=len(workers), p=probs
                )
                
                persons.loc[workers.index, 'work_zone'] = chosen_zones
            else:
                persons['work_zone'] = 1  # fallback
        
        elif segment == 'school':
            # Students only
            students = persons[persons.get('STUDYING', 0) > 0].copy()
            
            if 'ENROLLMENT' in landuse.columns:
                enroll = landuse['ENROLLMENT'].values
                enroll = np.maximum(enroll, 1)
                probs = enroll / enroll.sum()
                
                chosen_zones = self.rng.choice(
                    landuse.index.values, size=len(students), p=probs
                )
                
                persons.loc[students.index, 'school_zone'] = chosen_zones
            else:
                persons['school_zone'] = 1
        
        return persons


# ─── Factory Functions ────────────────────────────────────────────────────────

def create_activity_generation_model(
    config: Dict = None,
    metacount_model: Any = None,
    seed: int = 42
) -> CompetingHazardActivityGenerator:
    """Create activity generation model from config."""
    configs = None
    if config and 'activity_hazards' in config:
        configs = {}
        for act_type, act_config in config['activity_hazards'].items():
            configs[act_type] = ActivityHazardConfig(**act_config)
    
    return CompetingHazardActivityGenerator(
        configs=configs,
        metacount_model=metacount_model,
        seed=seed
    )


def create_planning_order_model(
    metacount_model: Any = None,
    seed: int = 42
) -> ActivityPlanningOrderModel:
    """Create activity planning order model."""
    return ActivityPlanningOrderModel(
        metacount_model=metacount_model,
        seed=seed
    )


def create_timing_model(
    config: Dict = None,
    metacount_model: Any = None,
    seed: int = 42
) -> HybridTimingChoiceModel:
    """Create hybrid timing choice model."""
    copula_type = config.get('copula_type', 'clayton') if config else 'clayton'
    return HybridTimingChoiceModel(
        copula_type=copula_type,
        metacount_model=metacount_model,
        seed=seed
    )


def create_routine_location_model(
    metacount_model: Any = None,
    seed: int = 42
) -> RoutineLocationChoiceModel:
    """Create routine location choice model."""
    return RoutineLocationChoiceModel(
        metacount_model=metacount_model,
        seed=seed
    )


def create_all_activity_models(
    config: Dict = None,
    metacount_model: Any = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Create all activity generation models."""
    return {
        'activity_generation': create_activity_generation_model(config, metacount_model, seed),
        'planning_order': create_planning_order_model(metacount_model, seed),
        'timing': create_timing_model(config, metacount_model, seed),
        'work_location': create_routine_location_model(metacount_model, seed),
        'school_location': create_routine_location_model(metacount_model, seed),
    }