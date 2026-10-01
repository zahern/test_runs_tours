"""
Performance Optimization Utilities
===================================
Optimizations for large-scale agent-based simulation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, Iterator
import numpy as np
import pandas as pd
from functools import lru_cache
from functools import wraps
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed
import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)


class VectorizedModelApplier:
    """
    Batch model application for performance.
    Instead of calling model.predict() for each agent, batch process.
    """
    
    def __init__(self, model, batch_size: int = 1000):
        self.model = model
        self.batch_size = batch_size
        self._cache = {}
    
    def predict_batch(self, agents: List[Dict], feature_names: List[str]) -> np.ndarray:
        """Predict for multiple agents at once."""
        # Build feature matrix
        X = np.array([[agent.get(f, 0) for f in feature_names] for agent in agents])
        
        # Predict in batches
        results = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i + self.batch_size]
            pred = self.model.predict(batch)
            results.append(pred)
        
        return np.vstack(results) if len(results) > 1 else results[0]
    
    def predict_log_batch(self, agents: List[Dict], feature_names: List[str]) -> np.ndarray:
        """Batch log-prediction."""
        X = np.array([[agent.get(f, 0) for f in feature_names] for agent in agents])
        results = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i + self.batch_size]
            pred = self.model.predict(batch)
            results.append(np.log(np.maximum(pred, 1e-6)))
        return np.vstack(results)


class AgentBatchProcessor:
    """
    Process agents in batches for model application.
    """
    
    def __init__(self, model_registry, batch_size: int = 1000):
        self.model_registry = model_registry
        self.batch_size = batch_size
        self._feature_cache = {}
    
    def apply_frequency_model(self, agents: List[Dict], purposes: List[str]) -> Dict:
        """Batch apply frequency models."""
        results = {}
        for purpose in purposes:
            model = self._get_model(f'freq_{purpose}')
            if not model:
                # Fallback
                for agent in agents:
                    agent[f'{purpose}_freq'] = self._fallback_freq(agent, purpose)
                continue
            
            # Extract features
            features = self._extract_features(agents, model)
            preds = self._predict_batch(model, agents)
            
            for i, agent in enumerate(agents):
                agent[f'{purpose}_freq'] = max(0, int(np.round(preds[i])))
        
        return agents
    
    def _get_model(self, name):
        """Get model from registry."""
        # Implementation depends on model registry
        pass
    
    def _predict_batch(self, model, agents):
        """Batch predict."""
        pass


class ModelCache:
    """
    LRU cache for model predictions.
    """
    
    def __init__(self, maxsize: int = 10000):
        self._cache = {}
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Any:
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any):
        if len(self._cache) >= self._maxsize:
            # Simple FIFO eviction
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value
    
    def get_stats(self) -> Dict:
        total = self._hits + self._misses
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        return {'hits': self._hits, 'misses': self._misses, 'hit_rate': hit_rate}


class AgentBatchGenerator:
    """
    Generate agent batches for vectorized processing.
    """
    
    def __init__(self, agents: List[Any], batch_size: int = 1000):
        self.agents = agents
        self.batch_size = batch_size
        self._index = 0
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self):
        if self._index >= len(self.agents):
            raise StopIteration
        batch = self.agents[self._index:self._index + self.batch_size]
        self._index += self.batch_size
        return self.agents[self._index - self.batch_size:self._index]


def parallel_agent_processing(agents: List, process_fn: Callable, 
                             n_workers: int = None, chunk_size: int = 100) -> List:
    """
    Process agents in parallel using process pool.
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(agents))
    
    if n_workers <= 1 or len(agents) < 100:
        # Sequential
        return [process_fn(agent) for agent in agents]
    
    # Split into chunks
    chunk_size = max(1, len(agents) // n_workers)
    chunks = [agents[i:i + chunk_size] for i in range(0, len(agents), chunk_size)]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            results.extend(future.result())
    
    return results


def _process_chunk(chunk: List) -> List:
    """Process chunk of agents (placeholder)."""
    return chunk


class VectorizedModelApplier:
    """
    Vectorized model application for batch processing.
    """
    
    def __init__(self, models: Dict, feature_cols: Dict[str, List[str]], batch_size: int = 1000):
        self.models = models
        self.feature_cols = feature_cols
        self.batch_size = batch_size
    
    def apply_frequency_models(self, agents_df: pd.DataFrame, 
                              purposes: List[str]) -> pd.DataFrame:
        """Apply frequency models vectorized."""
        df = agents_df.copy()
        
        for purpose in purposes:
            model = self._get_model(f'freq_{purpose}')
            if model is None:
                continue
            
            # Get feature columns for this model
            feature_cols = self._get_feature_cols(purpose)
            
            # Prepare feature matrix
            X = self._prepare_features(purpose)
            
            # Predict
            preds = self.model.predict(X)
            df[f'{purpose}_freq'] = np.maximum(0, np.round(preds)).astype(int)
        
        return df
    
    def _get_model(self, purpose):
        pass
    
    def _prepare_features(self, agents_df, purpose):
        pass


class AgentVectorizer:
    """
    Convert agent list to vectorized format for batch processing.
    """
    
    @staticmethod
    def to_dataframe(agents: List[Dict]) -> pd.DataFrame:
        """Convert list of agent dicts to DataFrame."""
        return pd.DataFrame(agents)
    
    @staticmethod
    def to_array(agents: List[Dict], feature_cols: List[str]) -> np.ndarray:
        """Convert to feature matrix."""
        n = len(agents)
        n_features = len(feature_cols)
        X = np.zeros((len(agents), len(feature_cols)), dtype=np.float32)
        
        for i, agent in enumerate(agents):
            for j, col in enumerate(feature_cols):
                X[i, j] = agent.get(col, 0)
        
        return X
    
    @staticmethod
    def assign_predictions(agents: List[Dict], predictions: np.ndarray, 
                          col_name: str, transform: Callable = None):
        """Assign predictions back to agents."""
        for i, agent in enumerate(agents):
            val = predictions[i]
            if callable(transform):
                val = transform(val)
            agent[col_name] = val


class CachedModelWrapper:
    """
    Wrapper that caches model predictions.
    """
    
    def __init__(self, model, cache_size: int = 10000):
        self.model = model
        self.cache = {}
        self.max_cache = 10000
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, features: Dict) -> str:
        # Create hashable key from features
        items = tuple(sorted(features.items()))
        return str(hash(items))
    
    def predict(self, features: Dict) -> Any:
        key = self._make_key(features)
        if key in self._cache:
            return self._cache[key]
        
        pred = self.model.predict(features)
        if len(self._cache) >= 10000:
            # Simple eviction
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = pred
        return pred
    
    def clear_cache(self):
        self._cache.clear()


class AsyncEventProcessor:
    """
    Asynchronous event processing for high throughput.
    """
    
    def __init__(self, scheduler, max_workers: int = 4):
        self.scheduler = scheduler
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
        self._lock = threading.Lock()
    
    def submit_event_batch(self, events: List[Event]) -> List:
        """Process batch of events asynchronously."""
        futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._process_event, e) for e in events]
            return [f.result() for f in as_completed(futures)]
    
    def _process_event(self, event: Event) -> List[Event]:
        """Process single event (placeholder)."""
        return []


class EventBatchProcessor:
    """
    Batch process events for efficiency.
    """
    
    def __init__(self, dispatcher, batch_size: int = 100):
        self.dispatcher = dispatcher
        self.batch_size = batch_size
        self._buffer = []
        self._lock = threading.Lock()
    
    def add_event(self, event: Event):
        """Add event to batch buffer."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.batch_size:
                self._flush()
    
    def _flush(self):
        """Flush buffered events."""
        events = self._buffer
        self._buffer = []
        for event in events:
            self.dispatcher.dispatch(event, context)


# Performance monitoring
class PerformanceMonitor:
    """
    Monitor simulation performance.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
    
    def record(self, metric: str, value: float):
        with self._lock:
            self.metrics[metric].append(value)
    
    def get_stats(self) -> Dict:
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        return stats
    
    def reset(self):
        with self._lock:
            self.metrics.clear()


# Decorators for performance
def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time_module.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time_module.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def cached_model_predict(model, cache_size: int = 10000):
    """Decorator to cache model predictions."""
    cache = CachedModelWrapper(model, cache_size)
    return cache.predict


# Profiling context manager
class profile:
    """Context manager for profiling code blocks."""
    
    def __init__(self, name: str):
        self.name = name
        self.start = 0
    
    def __enter__(self):
        self.start = time_module.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time_module.perf_counter() - self.start
        logger.info(f"{self.name} took {elapsed:.4f}s")