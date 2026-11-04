"""
DTA feedback loop for pyram.

Iterates between the demand model and a dynamic traffic assignment engine:

1. export the current trip plan for the engine,
2. run the engine (MATSim, SUMO or DynusT),
3. read the engine's congested OD travel times back into a skim lookup,
4. optionally re-run destination/mode choice with the updated skims,
5. repeat until the skim change falls below the configured threshold.

Engine contract
---------------
The loop does not invent network results. Each engine adapter must leave an
``od_skim.csv`` file in its iteration output directory with at least the
columns ``OTAZ``, ``DTAZ``, ``time_min`` (and optionally ``dist_km``). If the
engine is not installed, or it finishes without writing that file, the loop
reports the failure instead of fabricating congested times.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COMPATIBLE_TIMES = ("time_min", "TIME", "time")
_COMPATIBLE_DISTS = ("dist_km", "DIST", "distance_km")


@dataclass
class DTAFeedbackResult:
    """Outcome of a DTA feedback run."""

    status: str
    engine: str
    iterations: int = 0
    converged: bool = False
    history: List[Dict[str, float]] = field(default_factory=list)
    plan: Optional[pd.DataFrame] = None
    skim_lookup: Dict[Tuple[int, int], Dict[str, float]] = field(default_factory=dict)
    output_dir: str = ""
    message: str = ""


class DTAFeedbackLoop:
    """Demand <-> DTA feedback loop with an explicit engine contract."""

    def __init__(
        self,
        config: Optional[Any] = None,
        output_dir: Union[str, Path] = "output/dta_feedback",
        engine: Optional[str] = None,
        iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        timeout_seconds: int = 7200,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.engine = engine or getattr(config, "engine", "matsim")
        self.iterations = int(iterations or getattr(config, "iterations", 3))
        self.convergence_threshold = float(
            convergence_threshold
            if convergence_threshold is not None
            else getattr(config, "convergence_threshold", 0.02)
        )
        self.timeout_seconds = int(timeout_seconds)

        self.matsim_jar = str(
            getattr(config, "matsim_jar", "/opt/matsim/matsim.jar")
        )
        self.sumo_bin = str(getattr(config, "sumo_bin", "sumo"))
        self.dynust_exe = str(getattr(config, "dynust_exe", "dynust"))

    # ── Engine discovery ────────────────────────────────────────────────────

    def engine_available(self) -> bool:
        """Whether the configured engine executable/jar can be found."""
        if self.engine == "matsim":
            return Path(self.matsim_jar).exists()
        if self.engine == "sumo":
            return shutil.which(self.sumo_bin) is not None
        if self.engine == "dynust":
            return shutil.which(self.dynust_exe) is not None
        return False

    def _unavailable_message(self) -> str:
        if self.engine == "matsim":
            return (
                f"MATSim jar not found at {self.matsim_jar}. Set dta.matsim_jar in "
                "the configuration (or MATSIM_JAR) to the packaged MATSim jar."
            )
        if self.engine == "sumo":
            return (
                f"SUMO executable '{self.sumo_bin}' is not on PATH. Set dta.sumo_bin "
                "in the configuration to the SUMO binary."
            )
        if self.engine == "dynust":
            return (
                f"DynusT executable '{self.dynust_exe}' is not on PATH. Set "
                "dta.dynust_exe in the configuration."
            )
        return f"Unknown DTA engine '{self.engine}' (expected matsim, sumo or dynust)."

    def _command(self, iteration_dir: Path) -> List[str]:
        if self.engine == "matsim":
            return [
                "java", "-Xmx4g", "-cp", self.matsim_jar,
                "org.matsim.run.Controler", str(iteration_dir / "config.xml"),
            ]
        if self.engine == "sumo":
            return [self.sumo_bin, "-c", str(iteration_dir / "sumo.sumocfg")]
        if self.engine == "dynust":
            return [self.dynust_exe]
        raise ValueError(self._unavailable_message())

    # ── Loop ────────────────────────────────────────────────────────────────

    def run(
        self,
        plan: pd.DataFrame,
        skim_lookup: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
        landuse: Optional[pd.DataFrame] = None,
        reestimate: Optional[
            Callable[[Dict[Tuple[int, int], Dict[str, float]]], pd.DataFrame]
        ] = None,
    ) -> DTAFeedbackResult:
        """
        Run the feedback loop.

        Parameters
        ----------
        plan : trip plan from the demand model (needs OTAZ, DTAZ, depart_time)
        skim_lookup : current {(otaz, dtaz): {'DIST': ..., ...}} skims
        landuse : zone table, used to write engine facilities
        reestimate : optional callable applying updated skims to the plan
        """
        current_plan = plan.copy() if plan is not None else pd.DataFrame()
        current_skims = dict(skim_lookup or {})
        history: List[Dict[str, float]] = []

        if not self.engine_available():
            message = self._unavailable_message()
            logger.warning(message)
            return DTAFeedbackResult(
                status="engine_unavailable",
                engine=self.engine,
                plan=current_plan,
                skim_lookup=current_skims,
                output_dir=str(self.output_dir),
                message=message,
            )

        if current_plan is None or current_plan.empty:
            return DTAFeedbackResult(
                status="no_demand",
                engine=self.engine,
                plan=current_plan,
                skim_lookup=current_skims,
                output_dir=str(self.output_dir),
                message="No trip plan supplied; nothing to assign.",
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        converged = False
        completed = 0

        for iteration in range(1, self.iterations + 1):
            iteration_dir = self.output_dir / f"dta_iter_{iteration}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            self._export_plan(current_plan, iteration_dir, landuse)

            command = self._command(iteration_dir)
            logger.info("DTA iteration %d/%d: %s", iteration, self.iterations, command[0])
            try:
                completed_process = subprocess.run(
                    command,
                    cwd=str(iteration_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return DTAFeedbackResult(
                    status="engine_timeout",
                    engine=self.engine,
                    iterations=completed,
                    history=history,
                    plan=current_plan,
                    skim_lookup=current_skims,
                    output_dir=str(self.output_dir),
                    message=(
                        f"{self.engine} exceeded the {self.timeout_seconds}s "
                        "iteration timeout."
                    ),
                )
            if completed_process.returncode != 0:
                return DTAFeedbackResult(
                    status="engine_failed",
                    engine=self.engine,
                    iterations=completed,
                    history=history,
                    plan=current_plan,
                    skim_lookup=current_skims,
                    output_dir=str(self.output_dir),
                    message=(
                        f"{self.engine} exited with code "
                        f"{completed_process.returncode}: "
                        f"{(completed_process.stderr or '').strip()[:2000]}"
                    ),
                )

            od_path = iteration_dir / "od_skim.csv"
            if not od_path.exists():
                return DTAFeedbackResult(
                    status="engine_output_missing",
                    engine=self.engine,
                    iterations=completed,
                    history=history,
                    plan=current_plan,
                    skim_lookup=current_skims,
                    output_dir=str(self.output_dir),
                    message=(
                        f"{self.engine} wrote no {od_path.name} in {iteration_dir}; "
                        "the engine adapter must emit OTAZ, DTAZ, time_min."
                    ),
                )

            updated_skims = read_od_skim(od_path)
            rmse = skim_rmse(current_skims, updated_skims) if current_skims else float("nan")
            history.append(
                {"iteration": float(iteration), "skim_rmse": float(rmse)}
            )
            logger.info("  skim RMSE vs previous iteration: %.5f", rmse)

            if reestimate is not None:
                current_plan = reestimate(updated_skims)

            current_skims = updated_skims
            completed = iteration

            if (
                iteration > 1
                and np.isfinite(rmse)
                and rmse < self.convergence_threshold
            ):
                converged = True
                logger.info(
                    "  converged: RMSE %.5f < %.5f", rmse, self.convergence_threshold
                )
                break

        self._write_history(history)
        return DTAFeedbackResult(
            status="completed" if converged else "max_iterations",
            engine=self.engine,
            iterations=completed,
            converged=converged,
            history=history,
            plan=current_plan,
            skim_lookup=current_skims,
            output_dir=str(self.output_dir),
            message=(
                "Converged." if converged
                else f"Stopped after {completed} iteration(s) without meeting "
                     f"RMSE < {self.convergence_threshold}."
            ),
        )

    # ── Export ──────────────────────────────────────────────────────────────

    def _export_plan(
        self,
        plan: pd.DataFrame,
        iteration_dir: Path,
        landuse: Optional[pd.DataFrame] = None,
    ) -> None:
        """Write the plan in the engine's expected layout."""
        plan.to_csv(iteration_dir / "plan.csv", index=False)

        if landuse is not None and len(landuse):
            landuse.to_csv(iteration_dir / "zones.csv", index=False)

        if self.engine == "matsim":
            self._write_matsim_config(iteration_dir, plan, landuse)
        elif self.engine == "sumo":
            self._write_sumo_config(iteration_dir)

    def _write_matsim_config(
        self,
        iteration_dir: Path,
        plan: pd.DataFrame,
        landuse: Optional[pd.DataFrame],
    ) -> None:
        config = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">\n'
            "<config>\n"
            '  <module name="controller">\n'
            '    <param name="outputDirectory" value="./output" />\n'
            '    <param name="firstIteration" value="0" />\n'
            '    <param name="lastIteration" value="0" />\n'
            "  </module>\n"
            '  <module name="plans">\n'
            '    <param name="inputPlansFile" value="plan.csv" />\n'
            "  </module>\n"
            "</config>\n"
        )
        (iteration_dir / "config.xml").write_text(config, encoding="utf-8")

    def _write_sumo_config(self, iteration_dir: Path) -> None:
        config = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<configuration>\n"
            '  <input>\n'
            '    <net-file value="network.net.xml"/>\n'
            '    <route-files value="plan.csv"/>\n'
            "  </input>\n"
            '  <time><begin value="0"/><end value="86400"/></time>\n'
            "</configuration>\n"
        )
        (iteration_dir / "sumo.sumocfg").write_text(config, encoding="utf-8")

    def _write_history(self, history: List[Dict[str, float]]) -> None:
        if not history:
            return
        pd.DataFrame(history).to_csv(
            self.output_dir / "dta_convergence.csv", index=False
        )


def read_od_skim(path: Union[str, Path]) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Read an ``od_skim.csv`` produced by a DTA engine adapter."""
    table = pd.read_csv(path)
    required = {"OTAZ", "DTAZ"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"{path} must contain OTAZ and DTAZ columns; found {list(table.columns)}"
        )

    time_col = next((c for c in _COMPATIBLE_TIMES if c in table.columns), None)
    if time_col is None:
        raise ValueError(f"{path} must contain a travel-time column (time_min)")
    dist_col = next((c for c in _COMPATIBLE_DISTS if c in table.columns), None)

    lookup: Dict[Tuple[int, int], Dict[str, float]] = {}
    for row in table.itertuples(index=False):
        key = (int(getattr(row, "OTAZ")), int(getattr(row, "DTAZ")))
        values = {"time": float(getattr(row, time_col))}
        if dist_col is not None:
            values["DIST"] = float(getattr(row, dist_col))
        lookup[key] = values
    return lookup


def skim_rmse(
    previous: Dict[Tuple[int, int], Dict[str, float]],
    current: Dict[Tuple[int, int], Dict[str, float]],
) -> float:
    """Root-mean-square difference in travel time over the shared OD pairs."""
    shared = set(previous).intersection(current)
    if not shared:
        return float("nan")

    differences = []
    for key in shared:
        old = previous[key].get("time")
        new = current[key].get("time")
        if old is None or new is None:
            continue
        differences.append((float(new) - float(old)) ** 2)

    if not differences:
        return float("nan")
    return float(np.sqrt(np.mean(differences)))


def run_dta_feedback(
    plan: pd.DataFrame,
    skim_lookup: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
    landuse: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None,
    output_dir: Union[str, Path] = "output/dta_feedback",
    iterations: Optional[int] = None,
    engine: Optional[str] = None,
    convergence_threshold: Optional[float] = None,
    reestimate: Optional[
        Callable[[Dict[Tuple[int, int], Dict[str, float]]], pd.DataFrame]
    ] = None,
) -> DTAFeedbackResult:
    """Convenience wrapper around :class:`DTAFeedbackLoop`."""
    loop = DTAFeedbackLoop(
        config=config,
        output_dir=output_dir,
        engine=engine,
        iterations=iterations,
        convergence_threshold=convergence_threshold,
    )
    return loop.run(
        plan,
        skim_lookup=skim_lookup,
        landuse=landuse,
        reestimate=reestimate,
    )
