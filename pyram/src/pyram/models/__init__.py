"""
Discrete choice models for pyram.

Provides:
- DestinationChoiceModel: MNL and Nested Logit with distance bands
- ModeChoiceModel: Nested Logit and Mixed Logit for mode choice
- ParkingChoiceModel: Parking choice (on-street, garage, P&R)
- TripDestinationModel: Trip-level intermediate stop destination
- CompetingHazardActivityGenerator: Competing hazard activity generation
- ActivityPlanningOrderModel: POLARIS-style planning order
- HybridTimingChoiceModel: Hybrid RUM-RRM Copula timing
- RoutineLocationChoiceModel: Work/school location choice
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd

try:
    import larch as lx
    from larch import P, X
    LARCH_AVAILABLE = True
except ImportError:
    LARCH_AVAILABLE = False
    warnings.warn("larch not installed - discrete choice models will use heuristics")

try:
    from pyram.models.activity_generation import (
        CompetingHazardActivityGenerator as _CompetingHazardActivityGenerator,
        ActivityPlanningOrderModel as _ActivityPlanningOrderModel,
        HybridTimingChoiceModel as _HybridTimingChoiceModel,
        RoutineLocationChoiceModel as _RoutineLocationChoiceModel,
        create_activity_generation_model as _create_activity_generation_model,
        create_planning_order_model as _create_planning_order_model,
        create_timing_model as _create_timing_model,
        create_routine_location_model as _create_routine_location_model,
        create_all_activity_models as _create_all_activity_models,
    )
    ACTIVITY_GEN_AVAILABLE = True
except ImportError:
    ACTIVITY_GEN_AVAILABLE = False
    warnings.warn("activity_generation module not available")

from pyram.models.stages import (
    CDAPModel,
    PurposeTimingModel,
    build_trip_chains,
    create_cdap_model,
    create_purpose_timing_model,
    expand_daily_patterns,
    normalise_purpose,
)


# ── Case-Alternative Construction Helpers ────────────────────────────────────

#: Land-use attributes carried into the destination choice alternative table.
_DEST_LANDUSE_COLUMNS = ("TOTEMP", "RETEMPN", "FPSEMPN", "HEREMPN", "TOTPOP", "is_cbd")


def _zone_attributes(landuse: pd.DataFrame) -> pd.DataFrame:
    """Return land use indexed by zone id, with one row per zone.

    Lets the alternative tables look attributes up by position, instead of the
    per-alternative ``landuse[landuse["zone_id"] == zone]`` scan that used to run
    inside the inner loop.
    """
    if "zone_id" not in landuse.columns:
        return landuse

    zones = pd.to_numeric(landuse["zone_id"], errors="coerce")
    frame = landuse.loc[zones.notna()].copy()
    frame["zone_id"] = zones.loc[zones.notna()].to_numpy()
    return frame.drop_duplicates(subset=["zone_id"]).set_index("zone_id")


def _numeric_column(values: pd.Series) -> np.ndarray:
    """Coerce a land-use column to an array, keeping non-numeric values as-is."""
    if values.dtype == bool:
        return values.to_numpy(dtype=np.int64)
    if values.dtype == object:
        numeric = pd.to_numeric(values, errors="coerce")
        if bool(numeric.notna().all()):
            return numeric.to_numpy(dtype=float)
    return values.to_numpy()


def _skim_matrices(
    skims: Optional[Dict],
    zone_ids: np.ndarray,
    fields: Tuple[str, ...],
) -> Tuple[Dict[str, np.ndarray], Dict[Any, int]]:
    """Materialise OD skims as dense zone-position matrices plus a zone index.

    ``skims`` is normally the ``{(otaz, dtaz): {matrix: value}}`` lookup built by
    :func:`pyram.data.build_skim_lookup`. Walking it once costs n_zones**2 dict
    probes; every later lookup is then a single array index rather than one dict
    probe per (trip, zone) pair.
    """
    zone_ids = np.asarray(list(zone_ids))
    n_zones = len(zone_ids)
    position = {zone: i for i, zone in enumerate(zone_ids)}
    matrices = {name: np.zeros((n_zones, n_zones), dtype=float) for name in fields}

    if not skims:
        return matrices, position

    sample = next(iter(skims.values()), None)
    if isinstance(sample, np.ndarray):
        # Already {matrix: 2-D array}; assume it shares the zone ordering.
        for name in fields:
            matrix = skims.get(name)
            if matrix is not None and np.shape(matrix) == (n_zones, n_zones):
                matrices[name] = np.asarray(matrix, dtype=float)
        return matrices, position

    for (origin, dest), values in skims.items():
        if not values:
            continue
        i = position.get(origin)
        j = position.get(dest)
        if i is None or j is None:
            continue
        for name in fields:
            if name in values:
                matrices[name][i, j] = values[name]
    return matrices, position


def _zone_positions(zone_values: np.ndarray, position: Dict[Any, int]) -> np.ndarray:
    """Map zone ids to matrix positions, -1 where the zone is unknown."""
    codes, uniques = pd.factorize(pd.Series(zone_values), sort=False)
    lookup = np.array([position.get(zone, -1) for zone in uniques], dtype=np.int64)
    positions = np.full(len(zone_values), -1, dtype=np.int64)
    known = codes >= 0
    if known.any():
        positions[known] = lookup[codes[known]]
    return positions


def _release_model_dataset(model: Any) -> None:
    """Drop a fitted model's reference to its (very large) dataset.

    larch keeps the estimation dataset on ``Model.data``. Holding it on every
    segment model pins gigabytes for the life of the process and serialises the
    whole alternative table into the saved model file, so it is released once the
    parameters have been read off. Simulation re-attaches a dataset.
    """
    if model is None:
        return
    try:
        model.data = None
    except Exception as exc:  # pragma: no cover - depends on the larch version
        warnings.warn(f"Could not release the fitted dataset: {exc}")

# ── Base Model Class ─────────────────────────────────────────────────────────

class BaseChoiceModel:
    """Base class for discrete choice models."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.coefficients = {}
        self.fitted = False
    
    def estimate(self, data: pd.DataFrame, **kwargs) -> "BaseChoiceModel":
        """Estimate model from data."""
        raise NotImplementedError
    
    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict choices for new data."""
        raise NotImplementedError
    
    def save(self, path: Union[str, Path]):
        """Save model to file."""
        import pickle
        # The fitted dataset dominates both the pickle and resident memory.
        _release_model_dataset(self.model)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "coefficients": self.coefficients,
                "config": self.config,
                "fitted": self.fitted,
                # Segment models / zone ids are needed to predict after a reload.
                "segment_models": getattr(self, "segment_models", None),
                "zone_ids": getattr(self, "zone_ids", None),
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseChoiceModel":
        """Load model from file."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(data.get("config", {}))
        obj.model = data.get("model")
        obj.coefficients = data.get("coefficients", {})
        obj.fitted = data.get("fitted", False)
        if data.get("segment_models") is not None:
            obj.segment_models = data["segment_models"]
        if data.get("zone_ids") is not None:
            obj.zone_ids = data["zone_ids"]
        return obj


# ── Destination Choice Model ─────────────────────────────────────────────────

class DestinationChoiceModel(BaseChoiceModel):
    """
    Destination choice model with support for:
    - MNL (Multinomial Logit)
    - NL Distance Bands (Nested Logit with distance-based nests)
    - NL SA3 (Nested Logit with SA3-level nests)
    
    Based on POLARIS/ActivitySim destination choice patterns.
    """
    
    MODEL_TYPES = ["mnl", "nl_distance", "nl_sa3"]
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model_type = self.config.get("model_type", "mnl")
        self.distance_bands = self.config.get("distance_bands", [0, 2, 5, 10, 100])
        self.sample_size = self.config.get("sample_size", 0)
        self.zone_ids = None
        self.segment_models = {}  # per segment (work, edu, nonmand)
    
    def _build_utility(self, ca_df: pd.DataFrame, segment: str) -> Dict[str, Any]:
        """Build utility specification for a segment."""
        if not LARCH_AVAILABLE:
            return {}
        
        # Base utility terms (same for all segments)
        utility_terms = {
            # Size term (employment/attractions)
            "c_size_term": "log_size",
            # Distance deterrence
            "c_log_dist": "log_DIST_1",
            "c_dist": "DIST",
            # Mode choice logsum (accessibility)
            "c_logsum": "logsum",
            # Crash risk
            "c_crash": "log_CRASH_1",
            # Accessibility
            "c_transit_acc": "transit_acc",
            "c_walk_acc": "walk_acc",
            "c_auto_work_acc": "auto_work_acc",
            "c_auto_shop_acc": "auto_shop_acc",
            # CBD
            "c_cbd_dest": "is_cbd",
            # Industry interactions
            "c_indp_finance": "FPSEMPN",
            "c_indp_health": "HEREMPN",
            "c_indp_retail": "RETEMPN",
        }
        
        # Segment-specific terms
        if segment == "work":
            utility_terms.update({
                "c_occp_dist": "occp_group",
                "c_edu_post": "Education",
            })
        elif segment == "edu":
            utility_terms.update({
                "c_edu_primary": "Primary",
                "c_edu_secondary": "Secondary",
            })
        elif segment == "nonmand":
            utility_terms.update({
                "c_shopping": "RETEMPN",
                "c_social": "Entertainment",
                "c_escort": "TOTPOP",
            })
        
        return utility_terms
    
    def _assign_nest_ids(self, ca_df: pd.DataFrame) -> pd.DataFrame:
        """Assign nest IDs based on distance bands."""
        if self.model_type != "nl_distance":
            return ca_df
        
        dist_km = ca_df.get("DIST", ca_df.get("dist_km", 0))
        ca_df = ca_df.copy()
        
        # Create distance band nest IDs
        conditions = []
        choices = []
        for i, (low, high) in enumerate(zip(self.distance_bands[:-1], self.distance_bands[1:])):
            if high == 100:  # last band
                cond = dist_km >= low
            else:
                cond = (dist_km >= low) & (dist_km < high)
            conditions.append(cond)
            choices.append(i)
        
        ca_df["nest_id"] = np.select(conditions, choices, default=len(self.distance_bands) - 2)
        return ca_df
    
    def estimate(
        self,
        trips: pd.DataFrame,
        persons: pd.DataFrame,
        landuse: pd.DataFrame,
        skims: Dict,
        segment_col: str = "purpose",
        **kwargs
    ) -> "DestinationChoiceModel":
        """
        Estimate destination choice model.
        
        Parameters
        ----------
        trips : HHTS trips with OTAZ, DTAZ, purpose, person_id
        persons : Person attributes
        landuse : Zone land use data
        skims : Skim matrices or lookup
        segment_col : Column to segment by (purpose, tour_purpose)
        """
        if not LARCH_AVAILABLE:
            warnings.warn("larch not available - using heuristic fallback")
            return self._estimate_heuristic(trips, persons, landuse, skims, segment_col)
        
        # Merge person attributes onto trips
        trips = trips.merge(persons, on="person_id", how="left", suffixes=("", "_p"))
        
        # Segment trips
        segments = trips[segment_col].unique()
        
        for segment in segments:
            seg_trips = trips[trips[segment_col] == segment].copy()
            
            if len(seg_trips) < 30:
                warnings.warn(f"Segment {segment}: only {len(seg_trips)} observations")
                continue
            
            # Build case-alternative dataset
            ca_df = self._build_ca_dataset(seg_trips, landuse, skims)
            
            # Assign nest IDs for NL
            ca_df = self._assign_nest_ids(ca_df)
            
            # Estimate
            model = self._estimate_segment(ca_df, segment)
            self.segment_models[segment] = model
        
        self.fitted = True
        return self
    
    def _build_ca_dataset(self, trips: pd.DataFrame, landuse: pd.DataFrame, skims: Dict) -> pd.DataFrame:
        """Build the case-alternative table for larch.

        The table is assembled by repeating the trip block over the zone axis and
        indexing dense skim matrices, instead of appending one Python dict per
        (trip, zone) pair. At 371 zones and ~50k trips the dict form holds tens of
        GB of live objects and costs O(trips * zones**2) time, because the
        land-use lookup re-scanned the whole table inside the inner loop; that is
        what exhausted memory and walltime on the cluster.
        """
        lu_frame = _zone_attributes(landuse)
        self.zone_ids = lu_frame.index.to_numpy()

        zone_ids = np.asarray(self.zone_ids)
        n_zones = len(zone_ids)
        n_cases = len(trips)
        if n_cases == 0 or n_zones == 0:
            return pd.DataFrame(
                columns=[
                    "case_id",
                    "alt_id",
                    "chosen",
                    "OTAZ",
                    "DTAZ",
                    "DIST",
                    "log_DIST_1",
                    "logsum",
                    "log_CRASH_1",
                ]
            )

        matrices, position = _skim_matrices(skims, zone_ids, ("DIST", "logsum", "CRASH"))

        otaz = trips["OTAZ"].to_numpy()
        dtaz = trips["DTAZ"].to_numpy()
        origin_pos = _zone_positions(otaz, position)
        chosen_pos = _zone_positions(dtaz, position)

        sample_size = int(self.sample_size or 0)
        if 2 <= sample_size < n_zones and (chosen_pos >= 0).all():
            case_rep, alt_rep = self._sample_alternatives(
                chosen_pos, n_zones, sample_size
            )
        else:
            if 2 <= sample_size < n_zones:
                warnings.warn(
                    "Destination alternative sampling disabled: observed destinations "
                    "fall outside the zone system, so the chosen alternative cannot be "
                    "kept for every case."
                )
            case_rep, alt_rep = self._full_alternatives(n_cases, n_zones)

        case_origin = origin_pos[case_rep]
        known_origin = case_origin >= 0
        safe_origin = np.where(known_origin, case_origin, 0)

        alt_ids = zone_ids[alt_rep]
        dist = matrices["DIST"][safe_origin, alt_rep]
        logsum = matrices["logsum"][safe_origin, alt_rep]
        crash = matrices["CRASH"][safe_origin, alt_rep]
        if not known_origin.all():
            # Origin outside the zone system: no skim information available.
            dist = np.where(known_origin, dist, 0.0)
            logsum = np.where(known_origin, logsum, 0.0)
            crash = np.where(known_origin, crash, 0.0)

        data = {
            "case_id": trips.index.to_numpy()[case_rep],
            "alt_id": alt_ids,
            "chosen": (alt_ids == dtaz[case_rep]).astype(int),
            "OTAZ": otaz[case_rep],
            "DTAZ": dtaz[case_rep],
            "DIST": dist,
            "log_DIST_1": np.log(np.maximum(dist, 1.0)),
            "logsum": logsum,
            "log_CRASH_1": np.log(np.maximum(crash, 1.0)),
        }
        for column in _DEST_LANDUSE_COLUMNS:
            if column in lu_frame.columns:
                data[column.lower()] = _numeric_column(lu_frame[column])[alt_rep]

        return pd.DataFrame(data)

    @staticmethod
    def _full_alternatives(n_cases: int, n_zones: int) -> Tuple[np.ndarray, np.ndarray]:
        """Every zone as an alternative for every case, in case-major blocks."""
        return (
            np.repeat(np.arange(n_cases), n_zones),
            np.tile(np.arange(n_zones), n_cases),
        )

    def _sample_alternatives(
        self, chosen_pos: np.ndarray, n_zones: int, sample_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw a fixed-size alternative set per case, always keeping the chosen one.

        Non-chosen alternatives are drawn uniformly, which leaves the slope
        parameters consistent without a Manski-Lerman log-correction: that
        correction is only required when alternatives are drawn with unequal
        probability (importance sampling on a size term).
        """
        n_cases = len(chosen_pos)
        sample_size = int(min(max(sample_size, 2), n_zones))
        rng = np.random.default_rng(self.config.get("seed", 42))

        # Sort in bounded blocks so the draw keys never dominate the allocation.
        block = max(1, 2_000_000 // max(n_zones, 1))
        cases: List[np.ndarray] = []
        alts: List[np.ndarray] = []
        for start in range(0, n_cases, block):
            stop = min(start + block, n_cases)
            keys = rng.random((stop - start, n_zones))
            keys[np.arange(stop - start), chosen_pos[start:stop]] = -1.0
            picked = np.argpartition(keys, sample_size - 1, axis=1)[:, :sample_size]
            cases.append(np.repeat(np.arange(start, stop), sample_size))
            alts.append(picked.ravel())
        return np.concatenate(cases), np.concatenate(alts)

    def _estimate_segment(self, ca_df: pd.DataFrame, segment: str) -> Any:
        """Estimate model for one segment."""
        ca_df = ca_df.set_index(["case_id", "alt_id"])
        dt = lx.Dataset.construct.from_idca(ca_df, crack=False)
        m = lx.Model(dt)
        m.choice_ca_var = "chosen"

        # Build utility
        utility_terms = self._build_utility(ca_df, segment)

        if self.model_type == "nl_distance":
            # Nested Logit with distance bands
            nest_names = [f"dist_{i}" for i in range(len(self.distance_bands) - 1)]
            nest_params = {name: P(name) for name in nest_names}
            m.nesting_parameters = nest_params

        # Add utility terms
        utility = None
        for coef_name, var_name in utility_terms.items():
            if var_name in ca_df.columns:
                term = P(coef_name) * X(var_name)
                utility = term if utility is None else utility + term

        if utility is not None:
            m.utility_ca = utility

        # Estimate
        m.maximize_loglike(method="slsqp", engine="numba")

        # Store coefficients
        self.coefficients[segment] = {
            name: float(m.pf.loc[name, "value"]) for name in m.pf.index
        }

        # The dataset is only needed to fit; keeping it would pin this segment's
        # whole alternative table in memory and in the saved model file.
        _release_model_dataset(m)

        return m

    def _estimate_heuristic(self, trips, persons, landuse, skims, segment_col):
        """Heuristic fallback when Larch unavailable."""
        # Simple gravity model estimation
        self.fitted = True
        return self

    def predict(
        self,
        trips: pd.DataFrame,
        persons: pd.DataFrame,
        landuse: pd.DataFrame,
        skims: Dict,
        segment_col: str = "purpose",
        **kwargs
    ) -> pd.DataFrame:
        """
        Predict destinations for trips.
        
        Returns trips with predicted DTAZ column.
        """
        if not self.fitted:
            warnings.warn("Model not fitted - using gravity fallback")
            return self._predict_gravity(trips, landuse, skims)
        
        # Similar to estimation but with simulation
        trips = trips.merge(persons, on="person_id", how="left", suffixes=("", "_p"))
        predictions = []
        
        for segment in trips[segment_col].unique():
            if segment not in self.segment_models:
                continue
            
            seg_trips = trips[trips[segment_col] == segment].copy()
            model = self.segment_models[segment]
            
            # Build CA dataset
            ca_df = self._build_ca_dataset(seg_trips, landuse, skims)
            ca_df = self._assign_nest_ids(ca_df)
            
            # Simulate
            preds = self._simulate(model, ca_df, seg_trips)
            predictions.append(preds)
        
        if predictions:
            pred_df = pd.concat(predictions).drop_duplicates(subset=["case_id"])
            pred_map = dict(zip(pred_df["case_id"], pred_df["predicted_DTAZ"]))
            trips["DTAZ_pred"] = pd.Series(trips.index.map(pred_map), index=trips.index)
            trips["DTAZ_pred"] = trips["DTAZ_pred"].fillna(trips["DTAZ"])
        else:
            trips["DTAZ_pred"] = trips["DTAZ"]
        
        return trips
    
    def _simulate(self, model, ca_df, trips) -> pd.DataFrame:
        """Simulate choices from fitted model."""
        # Use Larch simulation
        ca_df = ca_df.set_index(["case_id", "alt_id"])
        dt = lx.Dataset.construct.from_idca(ca_df, crack=False)
        model.data = dt
        
        choices = model.simulate_choices()
        _release_model_dataset(model)
        return pd.DataFrame({"case_id": choices.index, "predicted_DTAZ": choices.values})
    
    def _predict_gravity(self, trips, landuse, skims):
        """Gravity model fallback prediction."""
        trips = trips.copy()
        trips["DTAZ_pred"] = trips["DTAZ"]  # fallback to observed
        return trips


# ── Mode Choice Model ────────────────────────────────────────────────────────

class ModeChoiceModel(BaseChoiceModel):
    """
    Mode choice model with:
    - Nested Logit (NL)
    - Mixed Logit (MXL)
    - MXL with GSE (Gradient Score Enhanced)
    
    Based on POLARIS mode choice structure.
    """
    
    MODEL_TYPES = ["nl", "mxl", "mxl_gse"]
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model_type = self.config.get("model_type", "nl")
        self.nest_structure = self.config.get("nest_structure", {
            "motorized": ["DA", "SR", "Transit"],
            "non_motorized": ["Walk", "Bike"]
        })
        self.transit_submodes = self.config.get("transit_submodes", True)
        self.access_modes = self.config.get("access_modes", True)
        self.mixed_logit = self.config.get("mixed_logit", False)
        self.panel_mxl = self.config.get("panel_mxl", True)
        self.mode_codes = {"DA": 1, "SR": 2, "Walk": 3, "Bike": 4, "Transit": 5}
    
    def estimate(
        self,
        trips: pd.DataFrame,
        persons: pd.DataFrame,
        landuse: pd.DataFrame,
        skims: Dict,
        **kwargs
    ) -> "ModeChoiceModel":
        """Estimate mode choice model."""
        if not LARCH_AVAILABLE:
            warnings.warn("larch not available - using heuristic fallback")
            return self
        
        # Prepare data
        trips = trips.merge(persons, on="person_id", how="left", suffixes=("", "_p"))
        
        # Build CA dataset
        ca_df = self._build_mode_ca(trips, landuse, skims)
        
        # Estimate
        self.model = self._estimate_mode_model(ca_df)
        self.fitted = True
        return self
    
    def _build_mode_ca(self, trips, landuse, skims):
        """Build case-alternative dataset for mode choice."""
        ca_rows = []
        modes = list(self.mode_codes.keys())
        
        for idx, trip in trips.iterrows():
            otaz = trip["OTAZ"]
            dtaz = trip["DTAZ"]
            purpose = trip.get("purpose", "other")
            
            for mode in modes:
                mode_code = self.mode_codes[mode]
                
                # Availability
                avail = self._check_availability(trip, mode)
                
                # Skim values
                skim_key = (otaz, dtaz)
                skim_vals = skims.get(skim_key, {})
                
                row = {
                    "case_id": idx,
                    "alt_id": mode_code,
                    "mode": mode,
                    **{f"mode_{candidate}": int(candidate == mode) for candidate in modes},
                    "chosen": int(
                        pd.to_numeric(
                            pd.Series([trip.get("mode", trip.get("tour_mode", 0))]),
                            errors="coerce",
                        ).fillna(0).iloc[0]
                        == mode_code
                    ),
                    "avail": avail,
                    "cost": skim_vals.get("cost", 0),
                    "time": skim_vals.get(f"{mode}_time", skim_vals.get("time", 0)),
                    "dist": skim_vals.get("DIST", 0),
                    "walk_dist": skim_vals.get("DISTWALK", 0),
                    "bike_dist": skim_vals.get("DISTBIKE", 0),
                }
                
                # Add person attributes
                for attr in ["incp", "age5p", "sexp", "vehrd", "bikes", "CARLICENCE"]:
                    if attr in trip.index:
                        row[attr] = trip[attr]
                
                ca_rows.append(row)
        
        return pd.DataFrame(ca_rows)
    
    def _check_availability(self, trip, mode):
        """Check mode availability."""
        if mode == "Walk":
            return trip.get("DISTWALK", trip.get("dist", 100)) <= 10
        elif mode == "Bike":
            return trip.get("DISTBIKE", trip.get("dist", 100)) <= 25
        elif mode == "Transit":
            return trip.get("transit_avail", 1) > 0
        elif mode == "DA":
            return trip.get("CARLICENCE", 0) > 0 and trip.get("vehrd", 0) > 0
        elif mode == "SR":
            return trip.get("vehrd", 0) > 0
        return True
    
    def _estimate_mode_model(self, ca_df):
        """Estimate mode choice model."""
        ca_df = ca_df.set_index(["case_id", "alt_id"])
        dt = lx.Dataset.construct.from_idca(ca_df, crack=False)
        m = lx.Model(dt)
        m.choice_ca_var = "chosen"
        m.availability_ca_var = "avail"
        
        # Build utility
        utility = None
        # ASC terms
        for mode in ["SR", "Walk", "Bike", "Transit"]:
            term = P(f"ASC_{mode}") * X(f"mode_{mode}")
            utility = term if utility is None else utility + term
        
        # Cost, time terms
        for term_name, var_name in [
            ("c_cost", "cost"),
            ("c_time", "time"),
            ("c_dist", "dist"),
            ("c_walk_dist", "walk_dist"),
            ("c_bike_dist", "bike_dist"),
        ]:
            if var_name in ca_df.columns:
                term = P(term_name) * X(var_name)
                utility = utility + term if utility is not None else term
        
        m.utility_ca = utility
        
        # Nesting
        if self.model_type in ["nl", "mxl", "mxl_gse"]:
            motorized = P("mu_motorized")
            nonmotor = P("mu_nonmotor")
            m.nesting_parameters = {"motorized": motorized, "nonmotor": nonmotor}
        
        if self.mixed_logit:
            # Add random parameters
            pass  # MXL implementation
        
        m.maximize_loglike(method="slsqp", engine="numba")
        
        self.coefficients = {
            name: float(m.pf.loc[name, "value"]) for name in m.pf.index
        }
        
        return m
    
    def predict(self, trips, persons, landuse, skims, **kwargs) -> pd.DataFrame:
        """Predict mode choices."""
        if not self.fitted:
            return self._predict_heuristic(trips)
        
        trips = trips.merge(persons, on="person_id", how="left", suffixes=("", "_p"))
        ca_df = self._build_mode_ca(trips, landuse, skims)
        ca_df = ca_df.set_index(["case_id", "alt_id"])
        
        dt = lx.Dataset.construct.from_idca(ca_df, crack=False)
        self.model.data = dt
        
        choices = self.model.simulate_choices()
        choice_map = pd.Series(choices.values, index=choices.index)
        trips = trips.copy()
        trips["mode_code_pred"] = pd.Series(trips.index.map(choice_map), index=trips.index)
        trips["mode_pred"] = trips["mode_code_pred"].map(
            {v: k for k, v in self.mode_codes.items()}
        )
        
        return trips
    
    def _predict_heuristic(self, trips):
        """Heuristic mode prediction."""
        trips = trips.copy()
        trips["mode_pred"] = "DA"
        trips["mode_code_pred"] = 1
        return trips


# ── Parking Choice Model ─────────────────────────────────────────────────────

class ParkingChoiceModel(BaseChoiceModel):
    """
    Parking choice model for auto trips.
    
    Alternatives:
    - On-street (free/cheap, search time, walk to destination)
    - Garage (paid, guaranteed space, walk to destination)
    - Park-and-Ride (transit access)
    """
    
    PARKING_TYPES = {
        1: "onstreet",
        2: "garage", 
        3: "pnr",
    }
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model_type = self.config.get("model_type", "mnl")
        self.onstreet_search_time = self.config.get("onstreet_search_time", 5.0)
        self.walk_speed_kmh = self.config.get("walk_speed_kmh", 5.0)
        self.cbd_zones = self.config.get("cbd_zones", [1, 2, 3, 4, 5])
    
    def estimate(
        self,
        trips: pd.DataFrame,
        persons: pd.DataFrame,
        location_table: pd.DataFrame,
        landuse: pd.DataFrame,
        skims: Dict,
        **kwargs
    ) -> "ParkingChoiceModel":
        """Estimate parking choice model."""
        if not LARCH_AVAILABLE:
            warnings.warn("larch not available - using heuristic fallback")
            return self
        
        # Filter auto trips
        auto_trips = trips[
            (trips["mode"].isin([1, 2])) &  # DA, SR
            (trips["purpose"] != "home")
        ].copy()
        
        if auto_trips.empty:
            warnings.warn("No auto trips for parking estimation")
            return self
        
        # Build CA dataset
        ca_df = self._build_parking_ca(auto_trips, persons, location_table, landuse, skims)
        
        # Estimate
        self.model = self._estimate_parking_model(ca_df)
        self.fitted = True
        return self
    
    def _build_parking_ca(self, trips, persons, location_table, landuse, skims):
        """Build case-alternative dataset for parking choice."""
        ca_rows = []
        
        for idx, trip in trips.iterrows():
            pid = trip["person_id"]
            dest_zone = trip["DTAZ"]
            purpose = trip["purpose"]
            act_dur = trip.get("predicted_duration_min", 60) / 60.0
            
            # Person attributes
            person = persons[persons["person_id"] == pid].iloc[0] if pid in persons["person_id"].values else None
            
            # Available parking in destination zone
            zone_facilities = location_table[location_table["zone"] == dest_zone]
            
            if len(zone_facilities) == 0:
                # Create synthetic alternatives
                alternatives = [
                    {"type": 1, "cost": 0, "walk_dist": 0.5, "search_time": self.onstreet_search_time},
                    {"type": 2, "cost": 5, "walk_dist": 0.3, "search_time": 0},
                ]
            else:
                alternatives = []
                for _, fac in zone_facilities.iterrows():
                    alternatives.append({
                        "type": 2,  # garage
                        "cost": fac.get("avg_parking_cost", 5),
                        "walk_dist": 0.3,
                        "search_time": 0,
                    })
                # Add on-street option
                alternatives.append({
                    "type": 1,
                    "cost": 0,
                    "walk_dist": 0.5,
                    "search_time": self.onstreet_search_time,
                })
            
            for alt in alternatives:
                ca_rows.append({
                    "case_id": idx,
                    "alt_id": alt["type"],
                    "parking_type": alt["type"],
                    "chosen": 0,  # Will be set from observed if available
                    "parking_fee": alt["cost"],
                    "walk_dist_km": alt["walk_dist"],
                    "search_time_min": alt["search_time"],
                    "act_dur_hr": act_dur,
                    "dest_zone": dest_zone,
                    "is_cbd": int(dest_zone in self.cbd_zones),
                    "female": int(person.get("sexp", 0) == 2) if person is not None else 0,
                    "income": person.get("incp", 50000) if person is not None else 50000,
                })
        
        return pd.DataFrame(ca_rows)
    
    def _estimate_parking_model(self, ca_df):
        """Estimate parking MNL."""
        ca_df = ca_df.set_index(["case_id", "alt_id"])
        dt = lx.Dataset.construct.from_idca(ca_df, crack=False)
        m = lx.Model(dt)
        m.choice_ca_var = "chosen"
        
        utility = None
        for coef, var in [
            ("c_fee", "parking_fee"),
            ("c_walk", "walk_dist_km"),
            ("c_search", "search_time_min"),
            ("c_duration", "act_dur_hr"),
            ("c_cbd", "is_cbd"),
        ]:
            if var in ca_df.columns:
                term = P(coef) * X(var)
                utility = utility + term if utility is not None else term
        
        m.utility_ca = utility
        m.maximize_loglike(method="slsqp", engine="numba")
        
        self.coefficients = {
            name: float(m.pf.loc[name, "value"]) for name in m.pf.index
        }
        
        return m
    
    def predict(
        self,
        trips: pd.DataFrame,
        persons: pd.DataFrame,
        location_table: pd.DataFrame,
        landuse: pd.DataFrame,
        skims: Dict,
        **kwargs
    ) -> pd.DataFrame:
        """Predict parking choices."""
        if not self.fitted:
            return self._predict_heuristic(trips, location_table)
        
        # Similar to estimation but with simulation
        auto_mask = (trips["mode"].isin([1, 2])) & (trips["purpose"] != "home")
        auto_trips = trips[auto_mask].copy()
        
        ca_df = self._build_parking_ca(auto_trips, persons, location_table, landuse, skims)
        ca_df = ca_df.set_index(["case_id", "alt_id"])
        
        dt = lx.Dataset.construct.from_idca(ca_df, crack=False)
        self.model.data = dt
        
        choices = self.model.simulate_choices()
        choice_map = pd.Series(choices.values, index=choices.index)
        
        # Merge back on the case ids used to build the alternatives
        trips = trips.copy()
        codes = pd.Series(auto_trips.index.map(choice_map), index=auto_trips.index)
        trips.loc[auto_mask, "parking_type_code"] = codes
        trips.loc[auto_mask, "parking_type"] = codes.map(self.PARKING_TYPES)
        
        return trips
    
    def _predict_heuristic(self, trips, location_table):
        """Heuristic parking assignment."""
        trips = trips.copy()
        auto_mask = (trips["mode"].isin([1, 2])) & (trips["purpose"] != "home")
        
        for idx in trips.index[auto_mask]:
            dest_zone = trips.at[idx, "DTAZ"]
            if dest_zone in self.cbd_zones:
                trips.at[idx, "parking_type"] = "garage"
                trips.at[idx, "parking_type_code"] = 2
            else:
                trips.at[idx, "parking_type"] = "onstreet"
                trips.at[idx, "parking_type_code"] = 1
        
        return trips


# ── Trip Destination Model ───────────────────────────────────────────────────

class TripDestinationModel(BaseChoiceModel):
    """
    Trip-level intermediate stop destination choice.
    
    For non-mandatory tours with multiple stops, predicts
    stop destinations conditional on tour destination and home anchor.
    
    Based on ActivitySim trip_destination and POLARIS patterns.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.detour_max_km = self.config.get("detour_max_km", 50.0)
        self.detour_max_ratio = self.config.get("detour_max_ratio", 1.0)
        self.detour_base_km = self.config.get("detour_base_km", 3.0)
        self.size_terms = self.config.get("size_terms", {
            "work": ["TOTEMP", "FPSEMPN", "HEREMPN"],
            "shopping": ["RETEMPN", "EATOUT"],
            "social": ["TOTPOP", "Entertainment"],
        })
    
    def estimate(
        self,
        stop_trips: pd.DataFrame,
        landuse: pd.DataFrame,
        skims: Dict,
        segment_col: str = "purpose",
        **kwargs
    ) -> "TripDestinationModel":
        """Estimate trip-level destination model for intermediate stops."""
        if not LARCH_AVAILABLE:
            warnings.warn("larch not available - using heuristic fallback")
            return self
        
        self.segment_models = {}
        
        for segment in stop_trips[segment_col].unique():
            seg_trips = stop_trips[stop_trips[segment_col] == segment]
            
            if len(seg_trips) < 30:
                continue
            
            model = self._estimate_segment(seg_trips, landuse, skims, segment)
            self.segment_models[segment] = model
        
        self.fitted = True
        return self
    
    def _estimate_segment(self, trips, landuse, skims, segment):
        """Estimate MNL for one purpose segment."""
        # Build CA dataset with detour distance
        ca_df = self._build_trip_ca(trips, landuse, skims)
        
        ca_df = ca_df.set_index(["case_id", "alt_id"])
        dt = lx.Dataset.construct.from_idca(ca_df, crack=False)
        m = lx.Model(dt)
        m.choice_ca_var = "chosen"
        
        # Utility with detour, size, logsum
        utility = None
        for coef, var in [
            ("c_detour", "detour_dist_km"),
            ("c_log_size", "log_size"),
            ("c_logsum", "logsum"),
            ("c_crash", "AutoCrashRisk"),
        ]:
            if var in ca_df.columns:
                term = P(coef) * X(var)
                utility = utility + term if utility is not None else term
        
        m.utility_ca = utility
        m.maximize_loglike(method="slsqp", engine="numba")
        
        return m
    
    def _build_trip_ca(self, trips, landuse, skims):
        """Build the trip-destination CA dataset, vectorised over alternatives."""
        lu_frame = _zone_attributes(landuse)
        zone_ids = lu_frame.index.to_numpy()
        n_zones = len(zone_ids)
        n_cases = len(trips)
        if n_cases == 0 or n_zones == 0:
            return pd.DataFrame(
                columns=[
                    "case_id",
                    "alt_id",
                    "chosen",
                    "detour_dist_km",
                    "log_size",
                    "logsum",
                    "AutoCrashRisk",
                    "feasible",
                ]
            )

        matrices, position = _skim_matrices(skims, zone_ids, ("DIST", "CRASH"))
        dist = matrices["DIST"]
        crash = matrices["CRASH"]

        origin_pos = _zone_positions(trips["origin_taz"].to_numpy(), position)
        anchor_pos = _zone_positions(trips["anchor_down_taz"].to_numpy(), position)

        case_rep = np.repeat(np.arange(n_cases), n_zones)
        alt_rep = np.tile(np.arange(n_zones), n_cases)

        case_origin = origin_pos[case_rep]
        known_origin = case_origin >= 0
        safe_origin = np.where(known_origin, case_origin, 0)
        safe_anchor = np.where(anchor_pos[case_rep] >= 0, anchor_pos[case_rep], 0)

        d_os = dist[safe_origin, alt_rep]
        d_sa = dist[alt_rep, safe_anchor]
        d_oa = dist[safe_origin, safe_anchor]
        risk = crash[safe_origin, alt_rep]
        if not known_origin.all():
            d_os = np.where(known_origin, d_os, 0.0)
            risk = np.where(known_origin, risk, 0.0)

        detour = np.maximum(d_os + d_sa - d_oa, 0.0)
        max_detour = np.minimum(
            self.detour_max_km, self.detour_max_ratio * d_oa + self.detour_base_km
        )

        purposes = (
            trips["purpose"].to_numpy()
            if "purpose" in trips.columns
            else np.full(n_cases, "other", dtype=object)
        )
        purpose_codes, purpose_values = pd.factorize(pd.Series(purposes), sort=False)
        purpose_codes = np.where(purpose_codes < 0, 0, purpose_codes)
        size_table = np.vstack(
            [self._size_term(lu_frame, str(purpose)) for purpose in purpose_values]
        )
        size = size_table[purpose_codes][case_rep, alt_rep]

        data = {
            "case_id": trips.index.to_numpy()[case_rep],
            "alt_id": zone_ids[alt_rep],
            "chosen": (
                zone_ids[alt_rep] == trips["dest_taz"].to_numpy()[case_rep]
            ).astype(int),
            "detour_dist_km": detour,
            "log_size": np.log(np.maximum(size, 1.0)),
            "logsum": 0.0,
            "AutoCrashRisk": risk,
            "feasible": detour <= max_detour,
        }
        return pd.DataFrame(data)

    def _size_term(self, lu_frame: pd.DataFrame, purpose: str) -> np.ndarray:
        """Summed size term per zone for one purpose."""
        size = np.zeros(len(lu_frame), dtype=float)
        for column in self.size_terms.get(purpose, ["TOTPOP"]):
            if column in lu_frame.columns:
                values = pd.to_numeric(
                    pd.Series(_numeric_column(lu_frame[column])), errors="coerce"
                )
                size = size + values.fillna(0.0).to_numpy(dtype=float)
        return size

    def predict(
        self,
        stop_trips: pd.DataFrame,
        landuse: pd.DataFrame,
        skims: Dict,
        segment_col: str = "purpose",
        **kwargs
    ) -> pd.DataFrame:
        """Predict stop destinations."""
        if not self.fitted:
            return self._predict_gravity(stop_trips, landuse, skims)
        
        predictions = []
        for segment in stop_trips[segment_col].unique():
            if segment not in self.segment_models:
                continue
            
            seg_trips = stop_trips[stop_trips[segment_col] == segment]
            model = self.segment_models[segment]
            
            ca_df = self._build_trip_ca(seg_trips, landuse, skims)
            ca_df = ca_df.set_index(["case_id", "alt_id"])
            
            dt = lx.Dataset.construct.from_idca(ca_df, crack=False)
            model.data = dt
            
            choices = model.simulate_choices()
            predictions.append(pd.DataFrame({
                "case_id": choices.index,
                "predicted_dest_taz": choices.values,
            }))
        
        if predictions:
            pred_df = pd.concat(predictions).drop_duplicates(subset=["case_id"])
            pred_map = dict(zip(pred_df["case_id"], pred_df["predicted_dest_taz"]))
            stop_trips["dest_taz_pred"] = pd.Series(
                stop_trips.index.map(pred_map), index=stop_trips.index
            )
            stop_trips["dest_taz_pred"] = stop_trips["dest_taz_pred"].fillna(
                stop_trips["dest_taz"]
            )
        else:
            stop_trips["dest_taz_pred"] = stop_trips["dest_taz"]
        
        return stop_trips
    
    def _predict_gravity(self, trips, landuse, skims):
        """Gravity fallback."""
        trips = trips.copy()
        trips["dest_taz_pred"] = trips["dest_taz"]
        return trips


# ── Factory Functions ────────────────────────────────────────────────────────

def create_destination_model(config: Dict = None) -> DestinationChoiceModel:
    """Create destination choice model from config."""
    return DestinationChoiceModel(config)


def create_mode_model(config: Dict = None) -> ModeChoiceModel:
    """Create mode choice model from config."""
    return ModeChoiceModel(config)


def create_parking_model(config: Dict = None) -> ParkingChoiceModel:
    """Create parking choice model from config."""
    return ParkingChoiceModel(config)


def create_trip_destination_model(config: Dict = None) -> TripDestinationModel:
    """Create trip destination model from config."""
    return TripDestinationModel(config)


def create_activity_generation_model(config: Dict = None, metacount_model: Any = None, seed: int = 42):
    """Create competing hazard activity generation model."""
    if ACTIVITY_GEN_AVAILABLE:
        return _create_activity_generation_model(config, metacount_model, seed)
    raise ImportError("activity_generation module not available")


def create_planning_order_model(metacount_model: Any = None, seed: int = 42):
    """Create activity planning order model."""
    if ACTIVITY_GEN_AVAILABLE:
        return _create_planning_order_model(metacount_model, seed)
    raise ImportError("activity_generation module not available")


def create_timing_model(config: Dict = None, metacount_model: Any = None, seed: int = 42):
    """Create hybrid timing choice model."""
    if ACTIVITY_GEN_AVAILABLE:
        return _create_timing_model(config, metacount_model, seed)
    raise ImportError("activity_generation module not available")


def create_routine_location_model(metacount_model: Any = None, seed: int = 42):
    """Create routine location choice model."""
    if ACTIVITY_GEN_AVAILABLE:
        return _create_routine_location_model(metacount_model, seed)
    raise ImportError("activity_generation module not available")


def create_all_models(config: "pyram.Config", metacount_model: Any = None) -> Dict[str, BaseChoiceModel]:
    """Create all models from main config."""
    models = {
        "destination": create_destination_model(config.destination.__dict__),
        "mode": create_mode_model(config.mode.__dict__),
        "parking": create_parking_model(config.parking.__dict__),
        "trip_destination": create_trip_destination_model({}),
    }
    
    if ACTIVITY_GEN_AVAILABLE:
        mc_model = models.get("metacount")  # if available
        activity_models = _create_all_activity_models(
            config.activity_generation.__dict__ if hasattr(config, 'activity_generation') else None,
            mc_model,
            config.seed
        )
        models.update(activity_models)
    
    return models
