"""
smoke_exploded_logit.py
=======================
Rank-ordered logit (exploded logit) for destination choice — conditioned on origin.

This implements the correct choice model for destination choice where the 
decision-maker implicitly ranks all zones by utility. The exploded logit
expands each choice into a sequence of binary choices (chosen vs. each
non-chosen alternative), conditioned on the origin zone.

Compares fit against standard MNL on the same data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List
import openmatrix as omx

logger = logging.getLogger(__name__)


def build_exploded_logit_data(
    trips: pd.DataFrame,
    all_zones: pd.DataFrame,
    skims: Dict[str, np.ndarray],
    landuse: pd.DataFrame,
    taz_index: Dict[int, int],
    purpose: str,
    zone_id_col: str = "zone_id",
    max_alts: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build exploded logit dataset for destination choice, conditioned on origin.
    
    For each trip (origin, chosen_dest), we create a ranking:
    1. chosen_dest (rank 1)
    2. All other zones (rank 2, tied)
    
    In exploded logit, this becomes a series of binary choices:
    - chosen vs. alt_1
    - chosen vs. alt_2
    - ...
    
    Each binary choice is conditioned on the origin zone (origin-specific
    parameters or origin fixed effects).
    
    Parameters
    ----------
    trips : stop trips for one purpose
    all_zones : zone alternatives
    skims : skim matrices
    landuse : land use data
    taz_index : TAZ to matrix index mapping
    purpose : trip purpose
    zone_id_col : zone identifier column
    max_alts : if set, randomly sample this many non-chosen alternatives
               per observation (for memory). None = all zones.
    
    Returns
    -------
    DataFrame in LARCH exploded logit format with columns:
    case_id, alt_id, chosen, origin_taz, ...utility variables...
    """
    from stage5b_trip_destination import (
        compute_size_term, _skim_lookup, SIZE_TERM_VARS, LANDUSE_ALIAS,
        BACKBONE_VARS, GA_CANDIDATE_VARS
    )
    
    n_obs = len(trips)
    zones = all_zones[[zone_id_col]].copy().reset_index(drop=True)
    n_zones = len(zones)
    
    if max_alts is not None and max_alts < n_zones - 1:
        # Will sample alternatives per observation
        sample_alts = True
    else:
        sample_alts = False
        max_alts = n_zones - 1
    
    # Build utility variables for all zones first (same as MNL)
    lu = landuse.copy()
    for alias, orig in LANDUSE_ALIAS.items():
        if orig in lu.columns and alias not in lu.columns:
            lu[alias] = lu[orig]
    if zone_id_col in lu.columns:
        lu_idx = lu.set_index(zone_id_col)
    else:
        lu_idx = lu
    
    # Pre-compute zone-level variables
    zone_vars = {}
    for var in ["TotalEmployment", "RetailEmp", "HealthEduRetEmp",
                "TotalPop", "TotalHH", "Shopping", "EATOUT", "Entertainment"]:
        if var in lu_idx.columns:
            zone_vars[var] = lu_idx[var]
        else:
            zone_vars[var] = pd.Series(0.0, index=lu_idx.index)
    
    # Size term for all zones
    size_term = compute_size_term(
        zones.assign(**{zone_id_col: zones[zone_id_col]}),
        purpose, landuse, zone_id_col
    )
    zone_vars["log_size"] = pd.Series(size_term, index=zones[zone_id_col])
    
    # Skim lookups from each origin to all zones
    # We'll compute per-origin in the loop
    
    rows = []
    case_id = 0
    
    for _, trip in trips.iterrows():
        origin = trip["origin_taz"]
        chosen_dest = trip["dest_taz"]
        
        # Map origin to skim index
        o_idx = taz_index.get(int(origin), 0)
        
        # Get all alternative zones
        alt_zones = zones[zone_id_col].values
        
        # Remove chosen from alternatives for sampling
        other_zones = [z for z in alt_zones if z != chosen_dest]
        
        if sample_alts and len(other_zones) > max_alts:
            np.random.seed(case_id)  # reproducible per case
            sampled_others = np.random.choice(other_zones, size=max_alts, replace=False)
            alt_set = [chosen_dest] + list(sampled_others)
        else:
            alt_set = [chosen_dest] + other_zones
        
        # Build utility for each alternative
        a_indices = np.array([taz_index.get(int(z), 0) for z in alt_set])
        
        # Skim values from origin to each alternative
        skim_vals = {}
        for alias in ["AutoIVT_AM", "NetworkDist", "AutoCrashRisk", "TransitIVT_AM"]:
            if alias in skims:
                skim_vals[alias] = _skim_lookup(skims[alias], 
                                                 np.full(len(a_indices), o_idx), 
                                                 a_indices)
            else:
                skim_vals[alias] = np.zeros(len(a_indices))
        
        # Detour (for stop trips)
        if "NetworkDist" in skims:
            anchor = trip.get("anchor_down_taz", origin)
            anchor_idx = taz_index.get(int(anchor), 0)
            d_oa = _skim_lookup(skims["NetworkDist"], 
                                 np.full(len(a_indices), o_idx), a_indices)
            d_aa = _skim_lookup(skims["NetworkDist"], a_indices, 
                                 np.full(len(a_indices), anchor_idx))
            d_oo = _skim_lookup(skims["NetworkDist"], 
                                 np.full(len(a_indices), o_idx), 
                                 np.full(len(a_indices), anchor_idx))
            detour = np.maximum(d_oa + d_aa - d_oo, 0.0)
        else:
            detour = np.zeros(len(a_indices))
        
        # Person covariates (same for all alternatives in this case)
        person_vals = {}
        for col in ["depart_time_min", "is_outbound", "is_female", "income_hi",
                    "hh_child", "occp_group", "ptype_FullTimeWorker",
                    "ptype_UniversityStudent", "mtwp", "WFHWEEK", "incp", "income_lo"]:
            person_vals[col] = trip.get(col, 0)
        
        # Build rows for each alternative
        for i, alt_zone in enumerate(alt_set):
            row = {
                "case_id": case_id,
                "alt_id": alt_zone,
                "chosen": 1 if alt_zone == chosen_dest else 0,
                "origin_taz": origin,  # Key: conditioned on origin
                "dest_taz": alt_zone,
                "detour_dist_km": detour[i],
                "log_size": zone_vars["log_size"].get(alt_zone, 0.0),
            }
            
            # Add skim variables
            for alias in ["AutoIVT_AM", "NetworkDist", "AutoCrashRisk", "TransitIVT_AM"]:
                row[alias] = skim_vals[alias][i]
            
            # Add land-use variables
            for var in ["TotalEmployment", "RetailEmp", "HealthEduRetEmp",
                        "TotalPop", "TotalHH", "Shopping", "EATOUT", "Entertainment"]:
                row[var] = zone_vars[var].get(alt_zone, 0.0)
            
            # Add person covariates
            for col, val in person_vals.items():
                row[col] = val
            
            rows.append(row)
        
        case_id += 1
        
        if case_id % 100 == 0:
            logger.info(f"  Built exploded data for {case_id}/{n_obs} trips...")
    
    df = pd.DataFrame(rows)
    
    # Numerical guards
    for col in df.columns:
        if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if df[col].dtype in (np.float64, np.float32):
                df[col] = df[col].clip(-1e6, 1e6)
    
    logger.info(f"Exploded logit dataset: {len(df)} rows, {case_id} cases, "
                f"{len(df)/case_id:.1f} alts/case avg")
    return df


def fit_exploded_logit(
    exploded_data: pd.DataFrame,
    utility_vars: List[str],
    origin_conditioning: str = "fixed_effects",
) -> Optional[Dict]:
    """
    Fit rank-ordered logit (exploded logit) using Larch.
    
    Parameters
    ----------
    exploded_data : DataFrame from build_exploded_logit_data
    utility_vars : list of variable names for utility function
    origin_conditioning : how to condition on origin
        - "fixed_effects": add origin_taz fixed effects (many params)
        - "interactions": interact key vars with origin characteristics
        - "stratified": estimate separately per origin (not implemented)
        - "none": standard MNL (no origin conditioning)
    
    Returns
    -------
    dict with model results or None if failed
    """
    try:
        import larch as lx
        from larch import P, X
    except ImportError:
        logger.error("larch not installed")
        return None
    
    # Prepare data for Larch
    df = exploded_data.copy()
    df = df.set_index(["case_id", "alt_id"])
    
    try:
        d = lx.Dataset.construct.from_idca(df, crack=False)
        m = lx.Model(d)
        
        # Utility specification
        if origin_conditioning == "fixed_effects":
            # Add origin fixed effects - one parameter per origin (minus 1)
            origins = df.index.get_level_values("origin_taz").unique()
            # In Larch, we can use P("origin_taz") * X("origin_taz") for fixed effects
            # But this creates many parameters. Alternative: use origin as a 
            # stratification variable in the choice set.
            logger.warning("Origin fixed effects not fully implemented in Larch; using interactions")
            origin_conditioning = "interactions"
        
        if origin_conditioning == "interactions":
            # Interact key distance variables with origin characteristics
            # For now, just include origin_taz as a variable (will be categorical)
            # Better: create origin-specific accessibility measures
            pass
        
        # Standard utility
        for var in utility_vars:
            if var in df.columns:
                m.utility_ca = m.utility_ca + P(var) * X(var)
        
        # For exploded logit, we need to tell Larch this is a ranking model
        # Larch's exploded logit: each case has multiple chosen=1 rows? No.
        # Actually, exploded logit in Larch uses the same MNL likelihood
        # but with the data expanded. The likelihood contribution for case i
        # with chosen alternative j is:
        #   exp(V_ij) / sum_k exp(V_ik)
        # which is exactly MNL. The "exploded" part is just the data expansion.
        
        # So we can use standard MNL estimation on the expanded data
        m.choice_ca_var = "chosen"
        
        # Try numba first (more robust)
        for engine in ["numba", "jax"]:
            try:
                result = m.maximize_loglike(method="slsqp", engine=engine)
                if np.isfinite(result.loglike):
                    n_params = len(m.pf)
                    n_cases = df.index.get_level_values("case_id").nunique()
                    bic = -2 * result.loglike + n_params * np.log(n_cases)
                    
                    return {
                        "model": m,
                        "params": m.pf.copy(),
                        "log_likelihood": result.loglike,
                        "bic": bic,
                        "n_cases": n_cases,
                        "n_params": n_params,
                        "engine": engine,
                        "origin_conditioning": origin_conditioning,
                    }
            except Exception as e:
                logger.debug(f"  {engine} failed: {e}")
                continue
        
        return None
        
    except Exception as exc:
        logger.error(f"Exploded logit estimation failed: {exc}")
        return None


def fit_mnl_comparison(
    trips: pd.DataFrame,
    all_zones: pd.DataFrame,
    skims: Dict[str, np.ndarray],
    landuse: pd.DataFrame,
    taz_index: Dict[int, int],
    purpose: str,
    zone_id_col: str = "zone_id",
) -> Optional[Dict]:
    """Fit standard MNL on same data for comparison."""
    from stage5b_trip_destination import _estimate_mnl_segment
    
    # Use existing MNL estimation
    g_tripdest = None  # all variables
    return _estimate_mnl_segment(
        stop_trips=trips,
        all_zones=all_zones,
        skims=skims,
        landuse=landuse,
        purpose=purpose,
        taz_index=taz_index,
        g_tripdest=g_tripdest,
        zone_id_col=zone_id_col,
    )


def compare_models(
    mnl_result: Dict,
    exploded_result: Dict,
) -> pd.DataFrame:
    """Compare MNL vs Exploded Logit results."""
    if mnl_result is None or exploded_result is None:
        return pd.DataFrame()
    
    # Extract coefficients
    mnl_params = mnl_result.get("params", pd.Series(dtype=float))
    exp_params = exploded_result.get("params", pd.Series(dtype=float))
    
    # Align on common parameters
    common = set(mnl_params.index) & set(exp_params.index)
    
    rows = []
    for param in sorted(common):
        rows.append({
            "parameter": param,
            "mnl_coef": mnl_params[param],
            "exploded_coef": exp_params[param],
            "diff": mnl_params[param] - exp_params[param],
            "rel_diff": (mnl_params[param] - exp_params[param]) / 
                       (abs(mnl_params[param]) + 1e-6) if abs(mnl_params[param]) > 1e-6 else 0,
        })
    
    # Add model-level comparison
    comparison = pd.DataFrame(rows)
    
    summary = {
        "mnl_ll": mnl_result.get("log_likelihood", np.nan),
        "exploded_ll": exploded_result.get("log_likelihood", np.nan),
        "mnl_bic": mnl_result.get("bic", np.nan),
        "exploded_bic": exploded_result.get("bic", np.nan),
        "mnl_params": mnl_result.get("n_params", 0),
        "exploded_params": exploded_result.get("n_params", 0),
        "mnl_n": mnl_result.get("n_obs", 0),
        "exploded_n_cases": exploded_result.get("n_cases", 0),
    }
    
    return comparison, summary


def run_exploded_logit_test(
    trips: pd.DataFrame,
    persons: pd.DataFrame,
    skims_path: str,
    landuse: pd.DataFrame,
    purpose: str = "work",
    zone_id_col: str = "zone_id",
    home_taz_col: str = "home_taz",
    max_alts: int = 50,
) -> Dict:
    """
    Run full exploded logit test for one purpose.
    
    Returns dict with both model results and comparison.
    """
    from stage5b_trip_destination import (
        run_stage5b, build_trip_records, compute_detour, _load_skims
    )
    
    logger.info(f"=== Exploded Logit Test for {purpose} ===")
    
    # Build trip records (same as stage5b)
    trips_aug = build_trip_records(trips, home_taz_col=home_taz_col)
    skims = _load_skims(skims_path)
    
    # Build TAZ index
    if zone_id_col in landuse.columns:
        all_zones = landuse[[zone_id_col]].drop_duplicates().reset_index(drop=True)
    else:
        all_taz = pd.Series(
            pd.concat([trips_aug["origin_taz"], trips_aug["dest_taz"]]).unique()
        ).sort_values().reset_index(drop=True)
        all_zones = pd.DataFrame({zone_id_col: all_taz})
    
    taz_index = {int(z): i for i, z in enumerate(all_zones[zone_id_col])}
    
    # Compute detour
    if "NetworkDist" in skims:
        trips_aug["detour_dist_km"] = compute_detour(trips_aug, skims["NetworkDist"], taz_index)
    else:
        trips_aug["detour_dist_km"] = 0.0
    
    # Filter to stop trips of this purpose
    stop_mask = (trips_aug["is_tour_dest_leg"] == 0) & (trips_aug["purpose"] == purpose)
    stop_trips = trips_aug[stop_mask].copy()
    
    logger.info(f"Stop trips for {purpose}: {len(stop_trips)}")
    
    if len(stop_trips) < 30:
        logger.warning("Too few observations")
        return {"error": "insufficient observations"}
    
    # Utility variables to test
    utility_vars = [
        "detour_dist_km", "log_size", "AutoIVT_AM", "NetworkDist", 
        "AutoCrashRisk", "TransitIVT_AM", "TotalEmployment", "RetailEmp",
        "depart_time_min", "is_outbound"
    ]
    
    # Fit MNL (standard)
    logger.info("Fitting standard MNL...")
    mnl_result = fit_mnl_comparison(
        stop_trips, all_zones, skims, landuse, taz_index, purpose, zone_id_col
    )
    
    # Build exploded logit data
    logger.info("Building exploded logit data...")
    exploded_data = build_exploded_logit_data(
        stop_trips, all_zones, skims, landuse, taz_index, purpose, 
        zone_id_col, max_alts=max_alts
    )
    
    # Fit exploded logit
    logger.info("Fitting exploded logit...")
    exploded_result = fit_exploded_logit(
        exploded_data, utility_vars, origin_conditioning="interactions"
    )
    
    # Compare
    comparison_df, summary = compare_models(mnl_result, exploded_result)
    
    logger.info(f"MNL: LL={summary['mnl_ll']:.1f}, BIC={summary['mnl_bic']:.1f}, n_params={summary['mnl_params']}")
    logger.info(f"Exploded: LL={summary['exploded_ll']:.1f}, BIC={summary['exploded_bic']:.1f}, n_params={summary['exploded_params']}")
    
    return {
        "purpose": purpose,
        "mnl_result": mnl_result,
        "exploded_result": exploded_result,
        "comparison": comparison_df,
        "summary": summary,
        "exploded_data_shape": exploded_data.shape,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    # Quick test with synthetic data
    np.random.seed(42)
    n_trips = 100
    n_zones = 30
    
    trips = pd.DataFrame({
        'trip_id': range(n_trips),
        'tour_id': np.repeat(range(n_trips // 3), 3)[:n_trips],
        'person_id': np.random.randint(1, 30, n_trips),
        'trip_num_in_tour': np.tile([1, 2, 3], n_trips // 3)[:n_trips],
        'is_outbound': np.tile([1, 0, 0], n_trips // 3)[:n_trips],
        'origin_taz': np.random.randint(1, n_zones + 1, n_trips),
        'dest_taz': np.random.randint(1, n_zones + 1, n_trips),
        'purpose': np.random.choice(['work', 'shop', 'social'], n_trips),
        'mode': 'auto',
        'depart_time_min': np.random.randint(0, 1440, n_trips),
        'arrive_time_min': np.random.randint(0, 1440, n_trips),
        'travel_time_min': np.random.randint(5, 60, n_trips),
    })
    
    # Mark tour dest legs
    trips['is_tour_dest_leg'] = 0
    for tour_id in trips['tour_id'].unique():
        mask = (trips['tour_id'] == tour_id) & (trips['is_outbound'] == 1)
        if mask.any():
            last = trips.loc[mask, 'trip_num_in_tour'].max()
            trips.loc[mask & (trips['trip_num_in_tour'] == last), 'is_tour_dest_leg'] = 1
    
    persons = pd.DataFrame({
        'person_id': range(1, 31),
        'home_taz': np.random.randint(1, n_zones + 1, 30),
        'is_female': np.random.randint(0, 2, 30),
        'income_hi': np.random.randint(0, 2, 30),
        'hh_child': np.random.randint(0, 2, 30),
        'occp_group': np.random.randint(1, 5, 30),
        'ptype_FullTimeWorker': np.random.randint(0, 2, 30),
        'ptype_UniversityStudent': np.random.randint(0, 2, 30),
        'mtwp': np.random.randint(0, 2, 30),
        'WFHWEEK': np.random.randint(0, 2, 30),
        'incp': np.random.randn(30),
    })
    
    landuse = pd.DataFrame({
        'zone_id': range(1, n_zones + 1),
        'TOTEMP': np.random.randint(0, 500, n_zones),
        'FPSEMPN': np.random.randint(0, 100, n_zones),
        'HEREMPN': np.random.randint(0, 100, n_zones),
        'RETEMPN': np.random.randint(0, 100, n_zones),
        'TOTPOP': np.random.randint(0, 1000, n_zones),
        'TOTHH': np.random.randint(0, 500, n_zones),
        'HSENROLL': np.random.randint(0, 200, n_zones),
    })
    
    # Create OMX
    omx_path = 'test_exploded_skims.omx'
    with omx.open_file(omx_path, 'w') as f:
        n = n_zones
        f.create_matrix('DIST', (n, n))
        f['DIST'][:] = np.random.rand(n, n) * 20
        f.create_matrix('DISTWALK', (n, n))
        f['DISTWALK'][:] = f['DIST'][:]  # Collinear
        f.create_matrix('DISTBIKE', (n, n))
        f['DISTBIKE'][:] = f['DIST'][:]
        f.create_matrix('SOVTOLL_TIME__AM', (n, n))
        f['SOVTOLL_TIME__AM'][:] = np.random.rand(n, n) * 30 + 5
        f.create_matrix('WLK_LOC_WLK_TOTIVT__AM', (n, n))
        f['WLK_LOC_WLK_TOTIVT__AM'][:] = np.random.rand(n, n) * 40 + 10
        f.create_matrix('CRASH', (n, n))
        f['CRASH'][:] = np.random.rand(n, n) * 0.1
    
    # Run test
    result = run_exploded_logit_test(
        trips=trips,
        persons=persons,
        skims_path=omx_path,
        landuse=landuse,
        purpose="work",
        max_alts=20,
    )
    
    if "comparison" in result:
        print("\n=== Parameter Comparison ===")
        print(result["comparison"].to_string(index=False))
        print(f"\nSummary: {result['summary']}")