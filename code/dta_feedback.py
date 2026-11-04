#!/usr/bin/env python
"""
dta_feedback.py
===============
DTA Feedback Loop: Demand Model → DTA → Updated Skims → Re-estimate
=====================================================================

Iterates between:
  1. Demand model (Stages 1-7) produces trip plans
  2. DTA (MATSim/SUMO/DynusT) simulates traffic, produces new link travel times
  3. Aggregate link times to zone-to-zone skims
  4. Re-run demand model with updated skims
  5. Repeat until convergence or max iterations

Outputs:
  - Updated OMX skim matrices at each iteration
  - Convergence metrics (skim RMSE, flow change)
  - Final equilibrated trip plans
"""

import os
import logging
import subprocess
import shutil
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import openmatrix as omx

logger = logging.getLogger(__name__)

# ── Default paths ────────────────────────────────────────────────────────────
MATSIM_JAR = os.environ.get('MATSIM_JAR', '/opt/matsim/matsim.jar')
SUMO_BIN = os.environ.get('SUMO_BIN', 'sumo')
DYNUST_EXE = os.environ.get('DYNUST_EXE', 'dynust')


def run_dta_feedback_loop(
    plan: pd.DataFrame,
    persons: pd.DataFrame,
    landuse: pd.DataFrame,
    skim_path: str,
    iterations: int = 3,
    engine: str = 'matsim',
    output_dir: str = 'saved_data',
    convergence_threshold: float = 0.02,
) -> Tuple[pd.DataFrame, str]:
    """
    Run DTA feedback loop.

    Parameters
    ----------
    plan : DataFrame
        Trip plan from Stage 7 (with parking, walk legs)
    persons : DataFrame
        Synthetic population
    landuse : DataFrame
        Zone landuse
    skim_path : str
        Path to initial OMX skim file
    iterations : int
        Maximum DTA iterations
    engine : str
        'matsim', 'sumo', or 'dynust'
    output_dir : str
        Output directory
    convergence_threshold : float
        Relative RMSE threshold for skim convergence

    Returns
    -------
    (final_plan, final_skim_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    current_plan = plan.copy()
    current_skim = skim_path

    for it in range(iterations):
        logger.info(f"DTA Feedback Iteration {it + 1}/{iterations}")

        # 1. Export plan to DTA format
        dta_input_dir = os.path.join(output_dir, f'dta_iter_{it + 1}')
        os.makedirs(dta_input_dir, exist_ok=True)

        if engine == 'matsim':
            dta_output = _run_matsim_dta(current_plan, persons, landuse, current_skim, dta_input_dir)
        elif engine == 'sumo':
            dta_output = _run_sumo_dta(current_plan, persons, landuse, current_skim, dta_input_dir)
        elif engine == 'dynust':
            dta_output = _run_dynust_dta(current_plan, persons, landuse, current_skim, dta_input_dir)
        else:
            raise ValueError(f"Unknown DTA engine: {engine}")

        # 2. Extract new skims from DTA output
        new_skim_path = os.path.join(dta_input_dir, f'skims_iter_{it + 1}.omx')
        _extract_skims_from_dta(dta_output, current_skim, new_skim_path, landuse)

        # 3. Check convergence
        if it > 0:
            rmse = _compute_skim_rmse(current_skim, new_skim_path)
            logger.info(f"  Skim RMSE vs previous: {rmse:.4f}")
            if rmse < convergence_threshold:
                logger.info(f"  Converged (RMSE < {convergence_threshold})")
                break

        # 4. Re-run demand model with new skims (Stages 5-7)
        # This is a lightweight re-estimation: only destination/mode/parking
        # with fixed upstream (tours, timing)
        logger.info("  Re-estimating destination/mode/parking with updated skims...")
        current_plan = _reestimate_with_new_skims(
            current_plan, persons, landuse, new_skim_path, output_dir
        )

        current_skim = new_skim_path

    logger.info(f"DTA feedback loop complete after {it + 1} iterations")
    return current_plan, current_skim


def _run_matsim_dta(plan: pd.DataFrame, persons: pd.DataFrame, landuse: pd.DataFrame,
                    skim_path: str, output_dir: str) -> str:
    """Run MATSim simulation and return output directory."""
    # Export plans.xml, facilities.xml, config.xml
    matsim_dir = os.path.join(output_dir, 'matsim')
    os.makedirs(matsim_dir, exist_ok=True)

    # Convert plan to MATSim format
    _export_matsim_plan(plan, persons, landuse, matsim_dir)

    # Run MATSim
    config_path = os.path.join(matsim_dir, 'config.xml')
    cmd = [
        'java', '-Xmx4g', '-cp', MATSIM_JAR,
        'org.matsim.run.Controler', config_path
    ]

    logger.info(f"Running MATSim: {' '.join(cmd[:3])} ...")
    result = subprocess.run(cmd, cwd=matsim_dir, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        logger.error(f"MATSim failed: {result.stderr}")
        raise RuntimeError(f"MATSim failed: {result.stderr}")

    logger.info("MATSim completed successfully")
    return matsim_dir


def _run_sumo_dta(plan: pd.DataFrame, persons: pd.DataFrame, landuse: pd.DataFrame,
                  skim_path: str, output_dir: str) -> str:
    """Run SUMO simulation and return output directory."""
    sumo_dir = os.path.join(output_dir, 'sumo')
    os.makedirs(sumo_dir, exist_ok=True)

    # Convert plan to SUMO routes (DUA or router)
    _export_sumo_routes(plan, persons, landuse, sumo_dir)

    # Run SUMO
    config_path = os.path.join(sumo_dir, 'sumo.sumocfg')
    cmd = [SUMO_BIN, '-c', config_path, '--output-prefix', 'iter_']

    logger.info(f"Running SUMO: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=sumo_dir, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        logger.error(f"SUMO failed: {result.stderr}")
        raise RuntimeError(f"SUMO failed: {result.stderr}")

    logger.info("SUMO completed successfully")
    return sumo_dir


def _run_dynust_dta(plan: pd.DataFrame, persons: pd.DataFrame, landuse: pd.DataFrame,
                    skim_path: str, output_dir: str) -> str:
    """Run DynusT simulation and return output directory."""
    dynust_dir = os.path.join(output_dir, 'dynust')
    os.makedirs(dynust_dir, exist_ok=True)

    # Convert plan to DynusT vehicle.dat + demand
    _export_dynust_demand(plan, persons, landuse, dynust_dir)

    # Run DynusT
    cmd = [DYNUST_EXE]

    logger.info(f"Running DynusT: {cmd[0]}")
    result = subprocess.run(cmd, cwd=dynust_dir, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        logger.error(f"DynusT failed: {result.stderr}")
        raise RuntimeError(f"DynusT failed: {result.stderr}")

    logger.info("DynusT completed successfully")
    return dynust_dir


def _export_matsim_plan(plan: pd.DataFrame, persons: pd.DataFrame, landuse: pd.DataFrame,
                        matsim_dir: str):
    """Export plan to MATSim plans.xml + facilities.xml."""
    from matsim_export import export_matsim  # reuse existing export
    export_matsim(
        plan_dir=os.path.dirname(matsim_dir),  # parent dir has stage7_output.csv
        out_dir=matsim_dir,
        landuse=landuse,
        crs='EPSG:28356',
        sample_frac=1.0,
    )


def _export_sumo_routes(plan: pd.DataFrame, persons: pd.DataFrame, landuse: pd.DataFrame,
                        sumo_dir: str):
    """Export plan to SUMO route files."""
    # Convert trips to SUMO person trips (router input)
    trips = plan[plan['row_type'] == 'trip'].copy()

    # Need network to route - use existing network
    net_path = os.environ.get('SUMO_NET', os.path.join(sumo_dir, '..', 'network.net.xml'))

    # Write person trips XML for duarouter
    trips_xml = os.path.join(sumo_dir, 'person_trips.xml')
    with open(trips_xml, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<personTrips>\n')
        for _, trip in trips.iterrows():
            f.write(f'  <personTrip id="{trip.name}" depart="{trip.STARTIME * 60:.0f}" ')
            f.write(f'fromZone="{trip.OTAZ}" toZone="{trip.DTAZ}" ')
            f.write(f'vehicleClass="{_sumo_vehicle_class(trip.mode)}"/>\n')
        f.write('</personTrips>\n')

    # Run duarouter
    cmd = [
        'duarouter',
        '--net-file', net_path,
        '--person-trip-files', trips_xml,
        '--output-file', os.path.join(sumo_dir, 'routes.rou.xml'),
        '--ignore-errors', 'true'
    ]
    subprocess.run(cmd, check=True)


def _sumo_vehicle_class(mode: int) -> str:
    """Map mode to SUMO vehicle class."""
    return {1: 'passenger', 2: 'passenger', 3: 'pedestrian', 4: 'bicycle', 5: 'passenger'}.get(mode, 'passenger')


def _export_dynust_demand(plan: pd.DataFrame, persons: pd.DataFrame, landuse: pd.DataFrame,
                          dynust_dir: str):
    """Export plan to DynusT vehicle.dat format."""
    # Use existing stage6 write_dynus_vehicle_dat
    from stage6_plan_finalisation import write_dynus_vehicle_dat
    write_dynus_vehicle_dat(plan, save_path=os.path.join(dynust_dir, 'vehicle.dat'), persons=persons)


def _extract_skims_from_dta(dta_output_dir: str, old_skim_path: str, new_skim_path: str,
                            landuse: pd.DataFrame):
    """
    Extract zone-to-zone skims from DTA output and save as OMX.

    For MATSim: read events.xml or linkstats.txt, compute zone OD travel times
    For SUMO: read edgeData.xml or summary, compute zone OD travel times
    For DynusT: read link_performance.dat, compute zone OD travel times
    """
    # This is a placeholder - actual implementation depends on DTA output format
    # For now, copy old skims with small perturbation to test pipeline
    logger.warning("Using placeholder skim extraction - copy with perturbation")

    with omx.open_file(old_skim_path, 'r') as old:
        with omx.open_file(new_skim_path, 'w') as new:
            for name in old.listMatrices():
                mat = np.array(old[name])
                # Add small random perturbation (1-5%) to simulate DTA feedback
                noise = 1.0 + np.random.normal(0, 0.02, mat.shape)
                mat = mat * noise
                mat = np.maximum(mat, 0.1)  # no negative times
                new.create_matrix(name, mat.shape, dtype=mat.dtype)
                new[name][:] = mat
            # Copy mappings
            for map_name in old.listMappings():
                mapping = dict(old.mapping(map_name))
                new.create_mapping(map_name, mapping)


def _compute_skim_rmse(skim1: str, skim2: str) -> float:
    """Compute relative RMSE between two skim matrices."""
    with omx.open_file(skim1, 'r') as f1, omx.open_file(skim2, 'r') as f2:
        rmses = []
        for name in f1.listMatrices():
            if name not in f2.listMatrices():
                continue
            m1 = np.array(f1[name]).flatten()
            m2 = np.array(f2[name]).flatten()
            mask = (m1 > 0) & (m2 > 0)
            if mask.sum() > 0:
                rel_diff = (m2[mask] - m1[mask]) / m1[mask]
                rmses.append(np.sqrt(np.mean(rel_diff ** 2)))
        return float(np.mean(rmses)) if rmses else 0.0


def _reestimate_with_new_skims(plan: pd.DataFrame, persons: pd.DataFrame, landuse: pd.DataFrame,
                               new_skim_path: str, output_dir: str) -> pd.DataFrame:
    """
    Re-run Stages 5-7 with updated skims.
    Only re-estimates destination choice, mode choice, parking choice.
    Keeps tour generation, timing, etc. fixed.
    """
    # This would call the pipeline stages 5-7 with the new skim path
    # For now, return original plan (placeholder)
    logger.warning("Re-estimation with new skims not fully implemented - returning original plan")
    return plan


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("DTA Feedback module loaded")
    print("Usage: from dta_feedback import run_dta_feedback_loop")