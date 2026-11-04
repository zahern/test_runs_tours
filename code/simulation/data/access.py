"""
Data Access Layer
=================
Abstracts data access for the simulation (land use, skims, networks, etc.)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataAccessLayer:
    """
    Unified data access layer for simulation.
    Provides unified access to land use, skims, networks, and synthetic population.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self._landuse_cache: Dict = {}
        self._skim_cache: Dict = {}
        self._network_cache: Dict = {}
        self._population_cache: Dict = {}
        
    def load_landuse(self, path: str) -> pd.DataFrame:
        """Load land use data."""
        path = Path(path)
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        # Ensure required columns
        required = ['SA2_MAIN16', 'TOTPOP', 'TOTEMP']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Landuse missing required column: {col}")
        
        self._landuse_cache['main'] = df
        return df
    
    def get_landuse(self, zone_ids: List[str] = None) -> pd.DataFrame:
        """Get landuse data, optionally filtered by zone IDs."""
        df = self._landuse_cache.get('main')
        if df is None:
            raise ValueError("Landuse not loaded. Call load_landuse() first.")
        
        if zone_ids:
            return df[df['SA2_MAIN16'].isin(zone_ids)]
        return df
    
    def load_skims(self, path: str, matrices: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Load skim matrices from OMX file.
        Returns dict of matrix_name -> 2D array.
        """
        try:
            import openmatrix as omx
        except ImportError:
            raise ImportError("openmatrix required for OMX files")
        
        skims = {}
        with omx.open_file(path, 'r') as omx_file:
            available = omx_file.list_matrices()
            matrices_to_load = matrices or available
            
            for mat_name in matrices_to_load:
                if mat_name in available:
                    skims[mat_name] = np.array(omx_file[mat_name])
                else:
                    logger.warning(f"Matrix {mat_name} not found in {path}")
        
        self._skim_cache['main'] = skims
        return skims
    
    def get_skim(self, origin_zone: str, dest_zone: str, 
                 matrix: str, time_period: str = 'AM') -> float:
        """Get skim value for OD pair."""
        skims = self._skim_cache.get('main', {})
        matrix_name = f"{matrix}_{mode}" if mode else matrix
        
        # Implementation depends on skim structure
        # This is a placeholder
        return 0.0
    
    def get_skim_matrix(self, matrix: str, time_period: str = 'AM') -> np.ndarray:
        """Get full skim matrix."""
        skims = self._skim_cache.get('main', {})
        key = f"{matrix}_AM"  # Assuming AM period
        return self._skim_cache.get('main', {}).get(matrix, np.array([]))
    
    def load_synthetic_population(self, path: str) -> pd.DataFrame:
        """Load synthetic population."""
        path = Path(path)
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        self._population_cache['main'] = df
        return df
    
    def get_population(self, zone_ids: List[str] = None) -> pd.DataFrame:
        """Get synthetic population, optionally filtered by zone."""
        df = self._population_cache.get('main')
        if df is None:
            raise ValueError("Population not loaded")
        
        if zone_ids:
            return df[df['HOME_ZONE'].isin(zone_ids)]
        return df
    
    def get_zone_centroids(self) -> pd.DataFrame:
        """Get zone centroids (lat/lon)."""
        # This would typically come from a separate file
        return pd.DataFrame()


class SkimManager:
    """
    High-level skim access with caching and fallback.
    """
    
    def __init__(self, data_access: DataAccessLayer):
        self.data_access = data_access
        self._skim_cache = {}
        self._matrix_cache = {}
    
    def get_travel_time(self, origin: str, dest: str, mode: str, 
                       time_period: str = 'AM') -> float:
        """Get travel time between zones."""
        # Implementation depends on skim structure
        # This is a placeholder
        return 30.0  # Default 30 minutes
    
    def get_travel_cost(self, origin: str, dest: str, mode: str,
                       time_period: str = 'AM') -> float:
        """Get travel cost."""
        return 0.0
    
    def get_distance(self, origin: str, dest: str) -> float:
        """Get distance between zones."""
        return 0.0
    
    def get_logsum(self, origin: str, dest: str, mode: str) -> float:
        """Get mode choice logsum."""
        return 0.0


class NetworkManager:
    """
    Network operations (routing, pathfinding).
    """
    
    def __init__(self, network_path: str = None):
        self.network_path = network_path
        self._network = None
        self._graph = None
    
    def load_network(self, path: str):
        """Load network from file (Shapefile, GraphML, etc.)."""
        # Implementation depends on format
        pass
    
    def shortest_path(self, origin: str, dest: str, mode: str) -> List:
        """Find shortest path."""
        return []
    
    def get_link_attributes(self, link_id: str) -> Dict:
        """Get link attributes."""
        return {}


class DataManager:
    """
    High-level data manager coordinating all data access.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.landuse = DataAccessLayer()
        self.skims = SkimManager(DataAccessLayer())
        self.network = NetworkManager()
        self._initialized = False
    
    def initialize(self, config: Dict = None):
        """Initialize all data sources."""
        config = config or self.config
        
        # Load land use
        if 'landuse_path' in config:
            self.landuse.load_landuse(config['landuse_path'])
        
        # Load skims
        if 'skim_path' in config:
            self.skims.data_access.load_skims(config['skim_path'])
        
        # Load network
        if 'network_path' in config:
            self.network.load_network(config['network_path'])
        
        # Load population
        if 'population_path' in config:
            self.landuse.load_synthetic_population(config['population_path'])
        
        self._initialized = True
    
    def get_zone_attributes(self, zone_id: str) -> Dict:
        """Get all attributes for a zone."""
        lu = self.landuse.get_landuse([zone_id])
        if lu.empty:
            return {}
        return lu.iloc[0].to_dict()
    
    def get_zone_size_term(self, zone_id: str, purpose: str) -> float:
        """Get size term for destination choice."""
        attrs = self.get_zone_attributes(zone_id)
        if not attrs:
            return 1.0
        
        # Map purpose to land use variables
        purpose_map = {
            'work': 'TOTEMP',
            'education': 'Education',
            'shopping': 'Shopping',
            'other_maintenance': 'TOTEMP',
            'other_discretionary': 'TOTEMP',
            'social': 'TOTEMP',
            'recreation': 'TOTEMP',
            'eatout': 'EATOUT',
        }
        
        col = purpose_map.get(purpose, 'TOTEMP')
        return max(1.0, float(attrs.get(col, 1)))
    
    def get_travel_time(self, origin: str, dest: str, mode: str, 
                       time_period: str = 'AM') -> float:
        """Get travel time."""
        # Would use skim matrices in real implementation
        return 30.0  # Default


# Singleton instance
_data_manager = None

def get_data_manager(config: Dict = None) -> 'DataManager':
    """Get singleton data manager."""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    if not _data_manager._initialized:
        _data_manager.initialize()
    return _data_manager