# MATSim scenario toolkit

Turns a finished tours-pipeline run (with SparseEA-AGDS-searched models) into a
runnable **MATSim** scenario and network-level analysis.

## Contents

| File | Role |
|---|---|
| `pt2matsim-shaded.jar` | Fat jar: MATSim 2027 + geotools + pt2matsim OSM/GTFS tools. Runs the Controler. |
| `network.xml.gz` | Multimodal (car + pt) network from `seq_osm.pbf`, **EPSG:28356**. *Population-independent.* |
| `transitSchedule.xml.gz`, `transitVehicles.xml.gz` | SEQ GTFS transit mapped onto the network. *Population-independent.* |
| `RunScenario.java` / `.class` | MATSim run driver (car on network, pt via GTFS, walk/bike/ride teleported; capacity scaled to the sample). |
| `build_base_inputs.sh` | Regenerate network + transit from OSM/GTFS (rarely needed). |
| `build_pt2matsim.sh` | Rebuild the jar from source (rarely needed). |

The network and transit are **independent of the population**, so they are built
once (staged here) and reused by every scenario run.

## Run it (HPC)

From the code dir, submit `job_matsim_scenario.pbs` pointing at a finished run:

```bash
qsub -v "RUN_DIR=runs/agds_metacount_mv_staged__12345,MATSIM_SAMPLE_FRAC=0.1,MATSIM_ITERS=20,MATSIM_HEAP=64g" job_matsim_scenario.pbs
```

The job (1) exports a **sampled** `plans.xml` from the run's trips, (2) runs the
MATSim Controler with `flowCapacityFactor = storageCapacityFactor = MATSIM_SAMPLE_FRAC`
(the standard scaling so a p% sample loads the network like the full population),
and (3) writes network-level plots. Outputs land in `RUN_DIR/matsim/`:
`plans.xml`, `output/` (full MATSim output), `analysis/` (volume + congestion maps).

### DynusT-like replanning (ReRoute-only, decaying weights, innovation tail)

DynusT is a simulation-based DTA: fixed modes, iterative route choice, flows
stabilising to equilibrium. `RunScenario` mirrors that — **ReRoute only** (no
mode switching, so plans keep their activity-specified modes), reroute weight
stepping down across iteration bands, then an innovation-free tail:

| Iterations | Active reroute weight | Behaviour |
|---|---|---|
| 0 – 30% | w0+w1+w2 (default 0.6) | broad route exploration |
| 30 – 60% | w1+w2 (default 0.3) | narrowing |
| 60 – innovFrac | w2 (default 0.1) | fine-tuning |
| innovFrac – 100% | 0 | pure plan selection (equilibration tail) |

Knobs (also overridable per submit with `-v`):

```bash
qsub -v "RUN_DIR=...,MATSIM_SAMPLE_FRAC=0.1,MATSIM_ITERS=20,MATSIM_INNOV_FRAC=0.85,MATSIM_REROUTE_SCHED=0.3,0.2,0.1" job_matsim_scenario.pbs
```

`MATSIM_INNOV_FRAC` (default 0.8; 0.8–0.9 recommended) is
`fractionOfIterationsToDisableInnovation`. Judge stabilisation by the
relative-gap verdict `matsim_analysis.py` prints from per-iteration link stats:
if it still says NOT converged, extend `MATSIM_ITERS` before touching weights —
with the tail in place, more iterations settle the gap rather than churning it.

### The full ~2M synthetic population

Prefer a **sample** (`MATSIM_SAMPLE_FRAC=0.05`–`0.10`) with a large heap and
walltime — 100% needs very large memory (events alone reach hundreds of GB) and
is rarely necessary. Set `MATSIM_SAMPLE_FRAC=1.0` only for a final full run on a
big-memory node.

## Run it (local)

```bash
python matsim_export.py --run-dir <run> --sample-frac 0.1
java -Xmx24g -cp "matsim/pt2matsim-shaded.jar:matsim" RunScenario \
  matsim/network.xml.gz <run>/matsim/plans.xml \
  matsim/transitSchedule.xml.gz matsim/transitVehicles.xml.gz \
  <run>/matsim/output 0.1 10 30
python matsim_analysis.py --output-dir <run>/matsim/output
```
(On Windows use `;` as the classpath separator instead of `:`.)
