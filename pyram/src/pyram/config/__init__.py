"""
Configuration system for pyram.

Provides YAML-based configuration with environment variable overrides,
validation, and defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class DataPaths:
    """File paths for input data."""
    hhts_trips: str = "data/hhts_trips.csv"
    hhts_persons: str = "data/hhts_persons.csv"
    hhts_households: str = "data/hhts_households.csv"
    skims_omx: str = "data/skims.omx"
    landuse_csv: str = "data/landuse.csv"
    zone_crosswalk: str = "data/zone_crosswalk.csv"
    location_table: str = "data/location_table.csv"
    output_dir: str = "output"


@dataclass
class ZoneSystem:
    """Zone system configuration."""
    zone_id_col: str = "ORDER"  # SA2 code column
    n_zones: int = 371
    crs: str = "EPSG:28356"  # GDA94/MGA Zone 56


@dataclass
class SyntheticPopulation:
    """Synthetic population generation config."""
    use_overture: bool = True
    overture_year: int = 2020
    overture_themes: List[str] = field(default_factory=lambda: ["buildings", "places"])
    target_region: str = "QLD"  # Australian state
    sample_fraction: float = 1.0
    seed: int = 42


@dataclass
class CDAPConfig:
    """Household coordination (CDAP) config."""
    enable: bool = True
    estimate_from_hhts: bool = True
    joint_tours: bool = True
    coordination_type: str = "full"  # "full", "individual_only"


@dataclass
class DestinationChoiceConfig:
    """Destination choice model config."""
    model_type: str = "mnl"  # "mnl", "nl_distance", "nl_sa3"
    distance_bands: List[float] = field(default_factory=lambda: [0, 2, 5, 10, 100])
    sample_size: int = 0  # 0 = all zones
    logsum_smoothing: bool = True
    logsum_orthogonalize: bool = True
    crash_imputation: bool = True
    quantity_term: bool = False
    sign_constraints: bool = True
    cbd_zones: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])


@dataclass
class ModeChoiceConfig:
    """Mode choice model config."""
    model_type: str = "nl"  # "nl", "mxl", "mxl_gse"
    nest_structure: Dict = field(default_factory=lambda: {
        "motorized": ["DA", "SR", "Transit"],
        "non_motorized": ["Walk", "Bike"]
    })
    transit_submodes: bool = True
    access_modes: bool = True
    mixed_logit: bool = False
    panel_mxl: bool = True


@dataclass
class TimingConfig:
    """Departure time / duration model config."""
    model_type: str = "rrm"  # "rrm", "tdd", "rum"
    mandatory_periods: bool = True
    period_skims: bool = False


@dataclass
class TripChainConfig:
    """Trip chain building config."""
    max_acts_per_person: int = 6
    tour_mode_consistency: bool = True
    micro_locations: bool = False


@dataclass
class ParkingConfig:
    """Parking choice config."""
    enable: bool = True
    model_type: str = "mnl"  # "mnl", "heuristic"
    onstreet_search_time: float = 5.0  # minutes
    walk_speed_kmh: float = 5.0
    cbd_zones: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])


@dataclass
class DTAConfig:
    """DTA feedback loop config."""
    enable: bool = False
    engine: str = "matsim"  # "matsim", "sumo", "dynust"
    iterations: int = 3
    convergence_threshold: float = 0.02
    walltime_hours: int = 48
    matsim_jar: str = "/opt/matsim/matsim.jar"
    sumo_bin: str = "sumo"
    dynust_exe: str = "dynust"


@dataclass
class HPCConfig:
    """HPC/PBS job submission config."""
    conda_env: str = "zigenv"
    stage_ncpus: int = 8
    stage_mem: str = "320GB"
    stage_walltime: str = "24:00:00"
    pred_ncpus: int = 8
    pred_mem: str = "250GB"
    pred_walltime: str = "24:00:00"
    gpu: bool = False
    partition: str = "normal"


@dataclass
class ActivityGenerationConfig:
    """Activity generation model config."""
    model_type: str = "competing_hazard"  # "competing_hazard", "cross_classification"
    hazard_type: str = "piecewise_constant"  # "piecewise_constant", "weibull", "exponential"
    time_bins: int = 96  # 15-min intervals over 24h
    max_time_hours: float = 24.0
    use_metacount: bool = True
    baseline_hazard_smoothing: float = 1.0  # Gaussian filter sigma
    
    # Per-activity-type hazard configs (optional overrides)
    activity_hazards: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration class."""
    data: DataPaths = field(default_factory=DataPaths)
    zones: ZoneSystem = field(default_factory=ZoneSystem)
    population: SyntheticPopulation = field(default_factory=SyntheticPopulation)
    cdap: CDAPConfig = field(default_factory=CDAPConfig)
    destination: DestinationChoiceConfig = field(default_factory=DestinationChoiceConfig)
    mode: ModeChoiceConfig = field(default_factory=ModeChoiceConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    trip_chain: TripChainConfig = field(default_factory=TripChainConfig)
    parking: ParkingConfig = field(default_factory=ParkingConfig)
    dta: DTAConfig = field(default_factory=DTAConfig)
    hpc: HPCConfig = field(default_factory=HPCConfig)
    activity_generation: ActivityGenerationConfig = field(default_factory=ActivityGenerationConfig)
    
    # Runtime
    run_tag: str = ""
    seed: int = 42
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file with env var overrides."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        data = cls._apply_env_overrides(data)
        
        return cls._from_dict(data)
    
    @classmethod
    def _apply_env_overrides(cls, data: Dict) -> Dict:
        """Apply environment variable overrides (PYRAM_ prefix)."""
        prefix = "PYRAM_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert double underscore to dot for nested keys
                config_key = key[len(prefix):].lower().replace("__", ".")
                cls._set_nested(data, config_key, value)
        return data
    
    @classmethod
    def _set_nested(cls, data: Dict, key: str, value: str):
        """Set nested dictionary value from dot-separated key."""
        parts = key.split(".")
        d = data
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        
        # Type conversion
        try:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        except ValueError:
            pass
        
        d[parts[-1]] = value
    
    @classmethod
    def _from_dict(cls, data: Dict) -> "Config":
        """Create Config from nested dictionary."""
        # Extract sub-configs
        sub_configs = {}
        for field_name in [
            "data", "zones", "population", "cdap", "destination", 
            "mode", "timing", "trip_chain", "parking", "dta", "hpc"
        ]:
            if field_name in data:
                sub_cls = cls.__dataclass_fields__[field_name].type
                if hasattr(sub_cls, '__dataclass_fields__'):
                    sub_configs[field_name] = sub_cls(**data.pop(field_name))
        
        return cls(**data, **sub_configs)
    
    def to_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def get_env_overrides(self) -> Dict[str, str]:
        """Get environment variable overrides for PBS submission."""
        overrides = {}
        
        # Stage 5 destination
        if self.destination.model_type == "nl_distance":
            overrides["STAGE5_DEST_NL_DISTANCE"] = "1"
        if self.destination.logsum_smoothing:
            overrides["STAGE5_LOGSUM_SMOOTH"] = "1"
        if self.destination.logsum_orthogonalize:
            overrides["STAGE5_LOGSUM_ORTHOG"] = "1"
        if self.destination.crash_imputation:
            overrides["STAGE5_CRASH_ZERO_TO_MAX"] = "1"
        
        # Stage 5 trip-level
        overrides["STAGE5_TRIP_DEST_ENHANCE"] = "1"
        
        # Stage 6 mode
        if self.mode.mixed_logit:
            overrides["STAGE5_ENABLE_MXL"] = "1"
        if self.mode.model_type == "mxl_gse":
            overrides["STAGE5_ENABLE_GSE"] = "1"
        
        # Stage 7 parking
        if self.parking.enable:
            overrides["STAGE7_PARKING"] = "1"
        
        # DTA feedback
        if self.dta.enable:
            overrides["DTA_FEEDBACK_ITERS"] = str(self.dta.iterations)
            overrides["DTA_ENGINE"] = self.dta.engine
            overrides["DTA_CONVERGENCE_THRESHOLD"] = str(self.dta.convergence_threshold)
        
        # HPC resources
        overrides["STAGE_NCPUS"] = str(self.hpc.stage_ncpus)
        overrides["STAGE_MEM"] = self.hpc.stage_mem
        overrides["STAGE_WALLTIME"] = self.hpc.stage_walltime
        overrides["PRED_NCPUS"] = str(self.hpc.pred_ncpus)
        overrides["PRED_MEM"] = self.hpc.pred_mem
        overrides["PRED_WALLTIME"] = self.hpc.pred_walltime
        overrides["CONDA_ENV"] = self.hpc.conda_env
        
        return overrides


class ConfigManager:
    """Manages configuration loading, validation, and environment setup."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def setup_environment(self):
        """Apply configuration as environment variables."""
        for key, value in self.config.get_env_overrides().items():
            os.environ[key] = value
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of warnings."""
        warnings = []
        
        # Check required files exist
        required = [
            ("hhts_trips", self.config.data.hhts_trips),
            ("hhts_persons", self.config.data.hhts_persons),
            ("skims_omx", self.config.data.skims_omx),
            ("landuse_csv", self.config.data.landuse_csv),
        ]
        for name, path in required:
            if not Path(path).exists():
                warnings.append(f"Required file not found: {name} -> {path}")
        
        # Validate zone system
        if self.config.zones.n_zones <= 0:
            warnings.append("Invalid n_zones")
        
        # Validate DTA
        if self.config.dta.enable and self.config.dta.iterations <= 0:
            warnings.append("DTA enabled but iterations <= 0")
        
        return warnings
    
    def create_output_dirs(self):
        """Create output directories."""
        Path(self.config.data.output_dir).mkdir(parents=True, exist_ok=True)
        for subdir in ["models", "logs", "plots", "skims", "dta_feedback"]:
            Path(self.config.data.output_dir, subdir).mkdir(exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return Config()


def create_default_config(output_path: str = "config.yaml"):
    """Create a default configuration file."""
    config = Config()
    config.to_yaml(output_path)
    return config