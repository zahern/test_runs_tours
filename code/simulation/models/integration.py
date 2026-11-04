"""
Model Integration Layer
=======================
Bridges existing estimation models (Stages 0-6) with the event-driven simulation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wrapper for estimation models providing unified interface.
    """
    
    def __init__(self, model: Any, model_type: str, 
                 feature_names: List[str] = None,
                 target_names: List[str] = None,
                 preprocessor: Callable = None):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.target_names = target_names or []
        self.preprocessor = preprocessor
        self._is_fitted = hasattr(model, 'predict') or hasattr(model, 'predict_proba')
    
    def predict(self, features: Dict[str, Any]) -> np.ndarray:
        """Predict using the model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        # Prepare features
        X = self._prepare_features(features)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise ValueError("Model has no predict method")
    
    def predict_log(self, features: Dict[str, Any]) -> np.ndarray:
        """Predict log-scale (for AFT models)."""
        pred = self.predict(features)
        return np.log(np.maximum(pred, 1e-6))
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Convert dict features to model input array."""
        if self.preprocessor:
            return self.preprocessor(features)
        
        # Default: extract features in order
        if self.feature_names:
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
        else:
            # Assume dict values in consistent order
            X = np.array([list(features.values())])
        return X


class ModelRegistry:
    """
    Registry for all estimation models.
    Provides unified access to all models used in simulation.
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.wrappers: Dict[str, ModelWrapper] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def register(self, name: str, model: Any, model_type: str,
                 features: List[str] = None, targets: List[str] = None,
                 preprocessor: Callable = None, metadata: Dict = None):
        """Register a model."""
        wrapper = ModelWrapper(model, model_type, 
                              feature_names=metadata.get('features') if metadata else None,
                              preprocessor=metadata.get('preprocessor') if metadata else None)
        self.wrappers[name] = wrapper
        self.models[name] = model
        self.metadata[name] = metadata or {}
        self.metadata[name].update({
            'type': model_type,
            'features': metadata.get('features', []) if metadata else [],
            'targets': metadata.get('targets', []) if metadata else []
        })
    
    def get(self, name: str) -> Any:
        """Get raw model."""
        return self.models.get(name)
    
    def get_wrapper(self, name: str):
        return self.wrappers.get(name)
    
    def predict(self, name: str, features: Dict) -> np.ndarray:
        """Predict using named model."""
        wrapper = self.wrappers.get(name)
        if not wrapper:
            raise ValueError(f"Model {name} not found")
        return wrapper.predict(features)
    
    def predict_log(self, name: str, features: Dict) -> np.ndarray:
        """Predict log-scale."""
        wrapper = self.wrappers.get(name)
        if not wrapper:
            raise ValueError(f"Model {name} not found")
        return wrapper.predict_log(features)
    
    def list_models(self) -> Dict[str, Dict]:
        """List all registered models with metadata."""
        return {name: self.metadata[name] for name in self.models}


class ModelIntegrationLayer:
    """
    High-level integration layer connecting simulation to estimation models.
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self._load_default_models()
    
    def _load_default_models(self):
        """Load models from standard locations."""
        # This would load from disk/pickle
        pass
    
    def load_from_pickle(self, path: str, name: str, 
                        model_type: str, metadata: Dict = None):
        """Load model from pickle file."""
        import pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.register(name, model, metadata=metadata)
    
    def load_all_from_directory(self, dir_path: str):
        """Load all models from directory."""
        import os
        import pickle
        for fname in os.listdir(path):
            if fname.endswith('.pkl') or fname.endswith('.pickle'):
                name = fname.replace('.pkl', '').replace('.pickle', '')
                path = os.path.join(path, fname)
                # Infer model type from filename
                model_type = 'unknown'
                if 'cdap' in name.lower():
                    model_type = 'cdap'
                elif 'freq' in name.lower():
                    model_type = 'frequency'
                elif 'dur' in name.lower():
                    model_type = 'duration'
                elif 'mode' in name.lower():
                    model_type = 'mode'
                elif 'dest' in name.lower():
                    model_type = 'destination'
                
                with open(os.path.join(path, fname), 'rb') as f:
                    model = pickle.load(f)
                
                self.register(
                    name=name,
                    model=model,
                    model_type=model_type,
                    metadata={'source': path}
                )


class ModelApplicationEngine:
    """
    Applies models to agents in the simulation context.
    """
    
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
    
    def apply_cdap(self, agent_attributes: Dict) -> str:
        """Apply CDAP model (Stage 0)."""
        wrapper = self.wrappers.get('cdap')
        if not wrapper:
            # Fallback logic
            return 'M'  # Default
        
        probs = self.predict('cdap', agent_attrs)
        classes = ['H', 'M', 'N']
        return np.random.choice(['H', 'M', 'N'], p=probs[0])
    
    def predict_activity_frequency(self, agent_attrs: Dict, purpose: str) -> int:
        """Predict activity frequency for purpose."""
        wrapper = self.wrappers.get(f'freq_{purpose}')
        if not wrapper:
            # Fallback
            if purpose == 'work':
                return 1
            elif purpose == 'education':
                return 1
            return max(0, int(np.random.poisson(0.5)))
        
        pred = self.predict(f'freq_{purpose}', agent_attrs)
        return max(0, int(np.round(pred[0])))
    
    def predict_activity_duration(self, agent_attrs: Dict, 
                                  activity_type: str) -> float:
        """Predict activity duration (log-minutes)."""
        model_name = f'dur_{activity_type}'
        wrapper = self.wrappers.get(model_name)
        if not wrapper:
            return 60.0  # Default 1 hour
        
        log_dur = self.predict_log(model_name, agent_attrs)
        return max(15, float(np.exp(log_dur[0])))
    
    def predict_day_bounds(self, agent_attrs: Dict) -> Tuple[float, float]:
        """Predict start and end of day (minutes from midnight)."""
        wrapper = self.wrappers.get('day_bounds')
        if not wrapper:
            return 6*60, 22*60  # 6AM - 10PM default
        
        log_dur = self.predict_log('day_bounds', agent_attrs)
        duration = np.exp(log_dur[0])
        start = 6*60 + np.random.exponential(30)  # Around 6 AM
        end = start + duration
        return max(0, start), min(24*60, end)
    
    def predict_mode_choice(self, agent_attrs: Dict, trip_attrs: Dict,
                           available_modes: List[str]) -> str:
        """Predict mode choice using nested logit."""
        wrapper = self.wrappers.get('mode_choice')
        if not wrapper:
            return 'drive_alone'  # Default
        
        probs = self.predict('mode_choice', {**agent_attrs, **trip_attrs})
        # Return mode with highest probability
        modes = ['drive_alone', 'shared_ride', 'transit', 'walk', 'bike']
        idx = np.argmax(probs[0])
        return modes[idx]
    
    def predict_destination(self, agent_attrs: Dict, 
                           trip_attrs: Dict, 
                           available_zones: List[str]) -> str:
        """Predict destination zone."""
        wrapper = self.wrappers.get('destination')
        if not wrapper:
            return np.random.choice(available_zones)
        
        # This would compute utilities for each zone and sample
        # Simplified: return random from available
        return np.random.choice(available_zones)
    
    def predict_trip_generation(self, agent_attrs: Dict) -> int:
        """Predict total trips for person."""
        wrapper = self.wrappers.get('trip_gen')
        if not wrapper:
            return np.random.poisson(3.5)
        return int(np.round(self.predict('trip_gen', agent_attrs)[0]))


class ModelInterface:
    """
    Unified interface for all model operations in simulation.
    """
    
    def __init__(self, model_dir: str = None):
        self.registry = ModelRegistry()
        self.engine = ModelApplicationEngine(self.registry)
        
        if model_dir:
            self.load_all_models(model_dir)
    
    def load_model(self, name: str, path: str, model_type: str, 
                   features: List[str] = None, targets: List[str] = None,
                   preprocessor=None):
        """Load single model."""
        import pickle
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        from simulation.models.integration import ModelWrapper
        wrapper = ModelWrapper(model, 'pickle', 
                              feature_names=None,
                              preprocessor=None)
        # Would need to properly register
        pass
    
    def get_predictor(self, model_name: str) -> Callable:
        """Get prediction function for model."""
        def predictor(features: Dict) -> Any:
            return self.engine.registry.predict(model_name, features)
        return predictor
    
    def get_log_predictor(self, model_name: str) -> Callable:
        """Get log-scale predictor."""
        def predictor(features: Dict) -> np.ndarray:
            return self.engine.registry.predict_log(model_name, features)
        return predictor


# Global model interface instance
_model_interface = None

def get_model_interface() -> ModelInterface:
    """Get singleton model interface."""
    global _model_interface
    if _model_interface is None:
        _model_interface = ModelInterface()
    return _model_interface


def load_models_from_pipeline(results_dir: str):
    """
    Load all models from a pipeline results directory.
    Expected structure: results_dir/stage_X/model_name.pkl
    """
    import os
    import pickle
    from pathlib import Path
    
    models = {}
    base = Path(model_dir)
    
    for stage_dir in base.glob('stage_*'):
        stage_name = stage_dir.name
        for pkl_file in stage_dir.glob('*.pkl'):
            name = f"{stage_name}_{pkl_file.stem}"
            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)
            # Would register with ModelRegistry
            pass
    
    return {}