"""
Data loading utilities for pyram.

Provides functions to load:
- HHTS survey data (trips, persons, households)
- Skim matrices (OMX format)
- Land use data
- Zone crosswalks
- Overture Maps population data
- Synthetic population generation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

# Overture Maps integration (optional)
try:
    import overturemaps
    from overturemaps import download_overture
    OVERTURE_AVAILABLE = True
except ImportError:
    OVERTURE_AVAILABLE = False
    download_overture = None
    warnings.warn("overturemaps not available - Overture Maps integration disabled")

# Try newer overturemaps API
if OVERTURE_AVAILABLE:
    try:
        from overturemaps.core import download_overture as _download_overture
        download_overture = _download_overture
    except ImportError:
        pass
    try:
        from overturemaps.overture import download_overture as _download_overture
        download_overture = _download_overture
    except ImportError:
        pass


# ── HHTS Data Loading ────────────────────────────────────────────────────────

def load_hhts(
    trips_path: Union[str, Path],
    persons_path: Union[str, Path],
    households_path: Optional[Union[str, Path]] = None,
    trips_encoding: str = "utf-8",
    persons_encoding: str = "utf-8",
) -> Dict[str, pd.DataFrame]:
    """
    Load HHTS (Household Travel Survey) data.
    
    Parameters
    ----------
    trips_path : path to trips CSV
    persons_path : path to persons CSV  
    households_path : optional path to households CSV
    trips_encoding : encoding for trips file
    persons_encoding : encoding for persons file
    
    Returns
    -------
    dict with keys: 'trips', 'persons', 'households' (if provided)
    """
    trips = pd.read_csv(trips_path, encoding=trips_encoding)
    persons = pd.read_csv(persons_path, encoding=persons_encoding)
    
    # Standardize column names
    trips = _standardize_hhts_trips(trips)
    persons = _standardize_hhts_persons(persons)
    
    result = {"trips": trips, "persons": persons}
    
    if households_path and Path(households_path).exists():
        households = pd.read_csv(households_path, encoding=persons_encoding)
        households = _standardize_hhts_households(households)
        result["households"] = households
    
    return result


def _standardize_hhts_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize HHTS trips column names."""
    df = df.copy()
    
    # Common column mappings
    rename_map = {
        # IDs
        "PERSID": "person_id",
        "TRIPID": "trip_id",
        "TOURID": "tour_id",
        "HHID": "hhid",
        # Locations
        "ORIGIN": "OTAZ",
        "DESTINATION": "DTAZ",
        "ORIG_TAZ": "OTAZ",
        "DEST_TAZ": "DTAZ",
        "HOME_TAZ": "home_taz",
        # Purpose
        "PURPOSE": "purpose",
        "TRIP_PURPOSE": "purpose",
        "TOUR_PURPOSE": "tour_purpose",
        # Mode
        "MODE": "mode",
        "TRIP_MODE": "mode",
        "TOUR_MODE": "tour_mode",
        # Time
        "START_TIME": "STARTIME",
        "END_TIME": "ENDTIME",
        "DEPART_TIME": "STARTIME",
        "ARRIVE_TIME": "ENDTIME",
        "DURATION": "duration_min",
        "TRAVEL_TIME": "travtime",
        # Distance
        "DISTANCE": "distance_km",
        "TRIP_DIST": "distance_km",
        # Weights
        "WEIGHT": "weight",
        "PERWT": "weight",
    }
    
    # Apply renames (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}
    for old, new in rename_map.items():
        if old.lower() in cols_lower:
            df = df.rename(columns={cols_lower[old.lower()]: new})
    
    # Ensure required columns exist
    required = ["person_id", "OTAZ", "DTAZ", "purpose"]
    for col in required:
        if col not in df.columns:
            warnings.warn(f"Required column '{col}' not found in trips data")
    
    # Convert zone IDs to numeric
    for col in ["OTAZ", "DTAZ", "home_taz"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    
    # Convert times to minutes if needed
    for col in ["STARTIME", "ENDTIME"]:
        if col in df.columns:
            if df[col].max() > 2000:  # likely seconds
                df[col] = df[col] / 60.0
    
    return df


def _standardize_hhts_persons(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize HHTS persons column names."""
    df = df.copy()
    
    rename_map = {
        "PERSID": "person_id",
        "HHID": "hhid",
        "AGE": "age5p",
        "AGEP": "age5p",
        "SEX": "sexp",
        "SEXP": "sexp",
        "INCOME": "incp",
        "INCP": "incp",
        "VEHICLES": "vehrd",
        "VEHRD": "vehrd",
        "BIKES": "bikes",
        "LICENSE": "CARLICENCE",
        "CARLICENCE": "CARLICENCE",
        "WORKSTATUS": "WORKSTAT",
        "STUDYING": "STUDYING",
        "OCCUPATION": "occp",
        "INDUSTRY": "indp",
        "PERSON_TYPE": "ptype",
        "PTYPE": "ptype",
        "WEIGHT": "weight",
        "PERWT": "weight",
    }
    
    cols_lower = {c.lower(): c for c in df.columns}
    for old, new in rename_map.items():
        if old.lower() in cols_lower:
            df = df.rename(columns={cols_lower[old.lower()]: new})
    
    return df


def _standardize_hhts_households(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize HHTS households column names."""
    df = df.copy()
    
    rename_map = {
        "HHID": "hhid",
        "HHSIZE": "hhsize",
        "VEHICLES": "vehrd",
        "VEHRD": "vehrd",
        "BIKES": "bikes",
        "INCOME": "hhincome",
        "HHTYPE": "hh_type",
        "DWELLING": "dwelling_type",
        "WEIGHT": "weight",
        "HHWT": "weight",
    }
    
    cols_lower = {c.lower(): c for c in df.columns}
    for old, new in rename_map.items():
        if old.lower() in cols_lower:
            df = df.rename(columns={cols_lower[old.lower()]: new})
    
    return df


# ── Skim Matrix Loading ──────────────────────────────────────────────────────

def load_skims(omx_path: Union[str, Path]) -> Tuple[object, Dict[str, np.ndarray]]:
    """
    Load skim matrices from OMX file.
    
    Returns
    -------
    (omx_handle, matrix_dict)
    """
    import openmatrix as omx
    
    omx_path = Path(omx_path)
    if not omx_path.exists():
        raise FileNotFoundError(f"Skim file not found: {omx_path}")
    
    handle = omx.open_file(str(omx_path), "r")
    matrices = {}
    for name in handle.listMatrices():
        matrices[name] = np.array(handle[name])
    
    return handle, matrices


def build_skim_lookup(omx_path: Union[str, Path]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Build {(otaz, dtaz): {matrix: value}} lookup from OMX.
    
    Useful for fast OD queries during model estimation/prediction.
    """
    handle, matrices = load_skims(omx_path)
    
    # Get zone mapping
    mapping = dict(handle.mapping(handle.listMappings()[0]))
    zone_ids = sorted(mapping, key=lambda k: mapping[k])
    zone_ids = [int(z) for z in zone_ids]
    n = len(zone_ids)
    
    # Build lookup
    lookup = {}
    for i, o in enumerate(zone_ids):
        for j, d in enumerate(zone_ids):
            key = (o, d)
            lookup[key] = {name: float(mat[i, j]) for name, mat in matrices.items()}
    
    return lookup


# ── Land Use Data ────────────────────────────────────────────────────────────

def load_landuse(
    csv_path: Union[str, Path],
    zone_id_col: str = "ORDER",
    required_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load zone-level land use data.
    
    Expected columns (configurable):
    - zone_id_col: zone identifier (SA2 code, ORDER, etc.)
    - TOTEMP: total employment
    - TOTPOP: total population
    - RETEMPN: retail employment
    - FPSEMPN: finance/professional/scientific employment
    - HEREMPN: health/education/retail employment
    - etc.
    """
    df = pd.read_csv(csv_path)
    
    # Ensure zone ID column exists
    if zone_id_col not in df.columns:
        # Try common alternatives
        for alt in ["SA2_MAIN16", "SA2_CODE21", "zone_id", "zone", "TAZ"]:
            if alt in df.columns:
                df = df.rename(columns={alt: zone_id_col})
                break
    
    if zone_id_col not in df.columns:
        raise ValueError(f"Zone ID column '{zone_id_col}' not found in landuse data")
    
    df[zone_id_col] = pd.to_numeric(df[zone_id_col], errors="coerce").fillna(0).astype(int)
    df = df[df[zone_id_col] > 0].copy()
    
    # Fill missing numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


# ── Zone Crosswalk ───────────────────────────────────────────────────────────

def load_zone_crosswalk(
    csv_path: Union[str, Path],
    from_col: str = "SA2_MAIN16",
    to_col: str = "ORDER",
) -> pd.DataFrame:
    """
    Load zone crosswalk (e.g., SA2 to ORDER mapping).
    """
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    rename_map = {
        "SA2_MAIN16": "SA2_MAIN16",
        "SA2_CODE21": "SA2_CODE21",
        "zone_id": "zone_id",
        "ORDER": "ORDER",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    
    return df[[from_col, to_col]].dropna().astype({from_col: int, to_col: int})


# ── Overture Maps Integration ────────────────────────────────────────────────

def fetch_overture_population(
    bbox: Tuple[float, float, float, float],  # (min_lon, min_lat, max_lon, max_lat)
    year: int = 2020,
    themes: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """
    Fetch population data from Overture Maps.
    
    Parameters
    ----------
    bbox : bounding box (min_lon, min_lat, max_lon, max_lat)
    year : data year
    themes : Overture themes to download (default: buildings, places)
    output_path : optional path to save parquet
    
    Returns
    -------
    GeoDataFrame with population estimates
    """
    if not OVERTURE_AVAILABLE or download_overture is None:
        raise ImportError(
            "overturemaps package not available or download_overture function not found. "
            "Install with: pip install overturemaps"
        )
    
    if themes is None:
        themes = ["buildings", "places"]
    
    # Download Overture data
    gdf = download_overture(
        bbox=bbox,
        themes=themes,
    )
    
    # Filter for residential buildings and population-relevant places
    if "buildings" in gdf.columns:
        residential = gdf[gdf["subtype"].isin(["house", "apartments", "residential"])]
    else:
        residential = gpd.GeoDataFrame()
    
    if "places" in gdf.columns:
        pop_places = gdf[gdf["categories"].apply(
            lambda x: any(cat in str(x).lower() for cat in ["residential", "apartments", "housing"])
        )] if "categories" in gdf.columns else gpd.GeoDataFrame()
    else:
        pop_places = gpd.GeoDataFrame()
    
    # Combine and estimate population
    combined = pd.concat([residential, pop_places], ignore_index=True)
    
    if output_path:
        combined.to_parquet(output_path)
    
    return combined


def overture_to_synthetic_population(
    overture_gdf: gpd.GeoDataFrame,
    zone_boundaries: gpd.GeoDataFrame,
    zone_id_col: str = "ORDER",
    persons_per_dwelling: float = 2.5,
    workers_per_dwelling: float = 1.2,
) -> pd.DataFrame:
    """
    Convert Overture building/place data to synthetic population.
    
    Parameters
    ----------
    overture_gdf : GeoDataFrame from fetch_overture_population
    zone_boundaries : GeoDataFrame with zone polygons and zone_id_col
    zone_id_col : column name for zone ID in boundaries
    persons_per_dwelling : average persons per residential unit
    workers_per_dwelling : average workers per residential unit
    
    Returns
    -------
    Synthetic persons DataFrame with zone assignment
    """
    # Spatial join to assign zones
    if overture_gdf.crs != zone_boundaries.crs:
        overture_gdf = overture_gdf.to_crs(zone_boundaries.crs)
    
    joined = gpd.sjoin(overture_gdf, zone_boundaries[[zone_id_col, "geometry"]], how="left", predicate="within")
    
    # Count dwellings per zone
    dwelling_counts = joined.groupby(zone_id_col).size().rename("n_dwellings")
    
    # Generate synthetic persons
    persons_list = []
    for zone_id, n_dwellings in dwelling_counts.items():
        n_persons = int(n_dwellings * persons_per_dwelling)
        n_workers = int(n_dwellings * workers_per_dwelling)
        
        for i in range(n_persons):
            person = {
                "person_id": f"overture_{zone_id}_{i}",
                "zone_id": zone_id,
                "is_worker": i < n_workers,
                "age": np.random.randint(18, 80),
                "sex": np.random.choice([1, 2]),
            }
            persons_list.append(person)
    
    return pd.DataFrame(persons_list)


# ── Synthetic Population Generation ──────────────────────────────────────────

def build_synthetic_population(
    hhts_persons: pd.DataFrame,
    hhts_households: pd.DataFrame,
    zone_crosswalk: pd.DataFrame,
    landuse: pd.DataFrame,
    target_population: int,
    seed: int = 42,
    use_overture: bool = True,
    overture_gdf: Optional[gpd.GeoDataFrame] = None,
    zone_boundaries: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """
    Build synthetic population from HHTS survey expanded to target population.
    
    Can optionally augment with Overture Maps derived population.
    
    Parameters
    ----------
    hhts_persons : HHTS persons data (standardized)
    hhts_households : HHTS households data (standardized)
    zone_crosswalk : SA2 to ORDER mapping
    landuse : zone landuse data (for zone-level controls)
    target_population : total persons to generate
    seed : random seed
    use_overture : whether to use Overture data as control totals
    overture_gdf : pre-loaded Overture data
    zone_boundaries : zone polygons for spatial join
    
    Returns
    -------
    Synthetic persons DataFrame
    """
    rng = np.random.default_rng(seed)
    
    # Merge persons with households
    persons = hhts_persons.merge(hhts_households, on="hhid", how="left", suffixes=("", "_hh"))
    
    # Apply zone crosswalk
    if "home_taz" in persons.columns:
        persons["zone_id"] = persons["home_taz"]
    elif "HOME_TAZ" in persons.columns:
        persons["zone_id"] = persons["HOME_TAZ"]
    else:
        # Use crosswalk if available
        if "SA2_MAIN16" in persons.columns and "ORDER" in zone_crosswalk.columns:
            sa2_to_order = zone_crosswalk.set_index("SA2_MAIN16")["ORDER"].to_dict()
            persons["zone_id"] = persons["SA2_MAIN16"].map(sa2_to_order)
    
    persons["zone_id"] = pd.to_numeric(persons["zone_id"], errors="coerce").fillna(0).astype(int)
    
    # If using Overture, calibrate zone-level totals
    if use_overture and overture_gdf is not None and zone_boundaries is not None:
        overture_counts = overture_to_synthetic_population(
            overture_gdf, zone_boundaries
        ).groupby("zone_id").size()
        
        # Adjust expansion factors to match Overture totals
        zone_totals = persons.groupby("zone_id").size()
        expansion = overture_counts / zone_totals.replace(0, 1)
        expansion = expansion.fillna(1.0).clip(0.1, 10.0)
        
        persons["expansion_factor"] = persons["zone_id"].map(expansion).fillna(1.0)
    else:
        # Uniform expansion
        current_pop = len(persons)
        expansion_factor = target_population / current_pop
        persons["expansion_factor"] = expansion_factor
    
    # Expand persons. Repeat the row block by its expansion factor instead of
    # copying one Series per expanded person: the per-row form builds millions of
    # mixed-dtype objects, which is the second largest allocation in the pipeline
    # after the destination alternative tables.
    factors = (
        pd.to_numeric(persons["expansion_factor"], errors="coerce")
        .fillna(1.0)
        .round()
        .clip(lower=1)
        .astype(np.int64)
        .to_numpy()
    )
    total = int(factors.sum())
    if target_population > 0 and total > 2 * target_population:
        # Scale before allocating: the trim below would otherwise build a
        # population several times the size it keeps.
        factors = np.maximum(
            1, np.round(factors * (target_population / total)).astype(np.int64)
        )

    positions = np.repeat(np.arange(len(persons)), factors)
    synth = persons.iloc[positions].reset_index(drop=True)
    if len(synth):
        # within-person copy ordinal: 0, 1, ... n_copies - 1
        starts = np.repeat(np.cumsum(factors) - factors, factors)
        copies = np.arange(len(positions), dtype=np.int64) - starts
        synth["person_id"] = (
            synth["person_id"].astype(str).to_numpy() + "_exp" + copies.astype(str)
        )

    # Trim to target if overshot
    if len(synth) > target_population:
        synth = synth.sample(n=target_population, random_state=seed).reset_index(drop=True)
    
    # Ensure unique person_ids
    synth["person_id"] = synth["person_id"].astype(str)
    synth = synth.drop_duplicates(subset=["person_id"]).reset_index(drop=True)
    
    return synth


# ── Convenience Functions ────────────────────────────────────────────────────

def load_all_data(
    config: "pyram.Config",
) -> Dict[str, pd.DataFrame]:
    """
    Load all input data from configuration.
    
    Returns dict with: trips, persons, households, skims, landuse, crosswalk, location_table
    """
    data = {}
    
    # HHTS
    data.update(load_hhts(
        config.data.hhts_trips,
        config.data.hhts_persons,
        config.data.hhts_households,
    ))
    
    # Skims
    data["skims_handle"], data["skims_matrices"] = load_skims(config.data.skims_omx)
    data["skim_lookup"] = build_skim_lookup(config.data.skims_omx)
    
    # Landuse
    data["landuse"] = load_landuse(config.data.landuse_csv, config.zones.zone_id_col)
    
    # Crosswalk
    data["crosswalk"] = load_zone_crosswalk(config.data.zone_crosswalk)
    
    # Location table
    if Path(config.data.location_table).exists():
        data["location_table"] = pd.read_csv(config.data.location_table)
    else:
        data["location_table"] = pd.DataFrame()
    
    return data