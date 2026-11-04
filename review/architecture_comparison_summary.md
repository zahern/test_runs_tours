# Architecture Comparison: Review Document vs. Codebase

## Review Document Summary
The `review.docx` outlines a **12-system co-simulation platform** for Brisbane with preferred tools:

| System | Preferred Tool | Alternative | Role |
|--------|----------------|-------------|------|
| Co-simulation | **HELICS** | mosaik/FMI | Synchronization & communication |
| Travel behaviour | **MATSim / ActivitySim** | BEAM | Household activities/mobility |
| Traffic | **MATSim** | SUMO | Network dynamics |
| Land use | **UrbanSim** | lightweight module | Households/jobs/development |
| Buildings | **URBANopt** | CEA | Urban building stock |
| Building physics | **EnergyPlus/OpenStudio** | Modelica Buildings | High-fidelity energy |
| Distribution grid | **OpenDSS** | GridLAB-D | Feeders/transformers/DER |
| Regional grid | **PyPSA** | Calliope | Generation/storage/capacity |
| Socio-technical agents | **Mesa + new code** | custom | Adoption/investment |
| Spatial network/data | **OSMnx/OpenStreetMap** | — | Open network foundation |
| Uncertainty | **Python + new workflow** | — | Ensembles/RDM |
| Project innovation | **NEW** | — | Tipping points/intervention discovery |

---

## Codebase Implementation Audit (Z:\test_runs_tours\code)

### ✅ FULLY IMPLEMENTED & MATURE

| Review System | Code Implementation | Key Files |
|---------------|---------------------|-----------|
| **Travel behaviour** | Full 7-stage ActivitySim-inspired ABM pipeline with GA variable selection | `stage1_activity_frequency.py` ... `stage7_parking_choice.py` |
| **MATSim integration** | Export pipeline → MATSim XML; analysis of MATSim outputs | `matsim_export.py`, `matsim_analysis.py` |
| **ActivitySim alignment** | CDAP, accessibility, logsum cascade, tour/trip mode choice hierarchy | `cdap.py`, `compute_accessibility.py`, `logsum_accessibility.py` |
| **Land use (UrbanSim-role)** | Accessibility-driven reallocation on 371 SA2s; scenario engine | `run_luti_scenarios_371.py`, `build_zone_landuse_overture.py` |
| **Buildings (URBANopt-role)** | Overture-based building stock with floor area by type/vintage | `build_zone_landuse_overture.py`, `build_building_energy_371.py` |
| **Building physics (EnergyPlus-role)** | CityBES-role: EUI-based energy with DOE/ASHRAE reference schedules | `build_building_energy_371.py` |
| **Distribution grid (OpenDSS-role)** | URBANopt-anchored synthetic LV feeders per SA2 | `build_power_network_371.py` |
| **Spatial network/data** | Overture Maps buildings/places/transport; OSM zone supply | `build_location_table_overture.py`, `build_zone_supply.py` |
| **Uncertainty/Ensembles** | GA multi-restart, adaptive mutation, fitness caching, bootstrap validation | `ga_hypothesis_testing.py`, `test_pipeline_health.py` |

### 🟡 PARTIALLY IMPLEMENTED / PROXIED

| Review System | Status | Gap |
|---------------|--------|-----|
| **Traffic (MATSim/SUMO)** | MATSim export ✅; SUMO ❌; DynusT `vehicle.dat` ✅ | No SUMO; no dynamic traffic assignment loop |
| **Regional grid (PyPSA/Calliope)** | ❌ Not implemented | Only distribution exists; no transmission/generation |
| **Co-simulation (HELICS/mosaik)** | ❌ Not implemented | Sequential pipeline only; no federated time sync |
| **Socio-technical agents (Mesa)** | ❌ Not implemented | No agent-based adoption/investment models |
| **Project innovation (Tipping points)** | ❌ Not implemented | Scenario analysis exists but no automated discovery |

### 🔧 INFRASTRUCTURE STRENGTHS (Beyond Review Scope)
- **GA variable selection** across all 7 stages with collinearity repair, adaptive mutation
- **HPC orchestration** with PBS job scripts for every stage
- **Pipeline logging & reproducibility** with run-isolated output dirs, JAX/CUDA patches
- **Validation & testing** with HHTS validation, smoke tests, pipeline health checks
- **Data pipelines** for synthetic population, accessibility, skims, zone crosswalks

---

## Scale Audit: Review Vision vs. Codebase Reality

### Spatial Scale

| Dimension | Review Implies | Codebase Implements |
|-----------|----------------|---------------------|
| **Zone system** | Not specified | **371 SA2 zones** (SEQ) — migrated from legacy 330 ORDER zones |
| **Building resolution** | URBANopt (building-level) | **SA2-aggregate** (371 zones × ~13 building types × 8 vintages) |
| **Network resolution** | MATSim/SUMO (link-level) | **OMX skims** (zone-to-zone); DynusT export for link-level |
| **Power network** | OpenDSS (feeder-level) | **1 representative LV feeder per SA2** (371 feeders) |
| **Land use parcels** | UrbanSim (parcel-level) | **SA2-level** reallocation (no parcels) |

### Temporal Scale

| Dimension | Review Implies | Codebase Implements |
|-----------|----------------|---------------------|
| **Simulation horizon** | Not specified | **2025–2045** (5-year steps) for LUTI scenarios |
| **Time resolution (travel)** | MATSim (second-level) | **Minute-level** activity scheduling (1440 min/day) |
| **Time resolution (energy)** | EnergyPlus (sub-hourly) | **Hourly** (24h profiles per zone) |
| **Time resolution (power)** | OpenDSS (sub-second) | **Hourly** load profiles; no dynamic powerflow |
| **Co-simulation step** | HELICS (configurable) | **None** (sequential, no time synchronization) |

### Population / Entity Scale

| Dimension | Review Implies | Codebase Implements |
|-----------|----------------|---------------------|
| **Synthetic population** | Household-level (ActivitySim) | **~2.5M persons** (SA2-native 371 zones) |
| **Households** | Household-level (ActivitySim) | **~1M households** (from HHTS expansion) |
| **Buildings** | Building-level (URBANopt) | **~5,000 aggregate building-stock records** (371 × 13 types) |
| **Vehicles** | Agent-level (MATSim/BEAM) | **Trip-level** export to MATSim/DynusT |
| **Power customers** | Meter-level (OpenDSS) | **Aggregate per feeder** (residential + commercial + industrial) |
| **Agents (socio-technical)** | Mesa agents (thousands) | **Zero** (not implemented) |

### Computational Scale

| Dimension | Review Implies | Codebase Implements |
|-----------|----------------|---------------------|
| **GA search** | Not mentioned | **~50 stages × segments**, population 50–100, 20–50 generations |
| **Model estimation** | Standard MLE | **Larch/JAX nested logit** on 371-zone choice sets |
| **HPC usage** | Not specified | **PBS jobs**: 100s of cores, 24–48h walltime per GA chain |
| **Data volumes** | Not specified | **OMX skims**: 12GB+; synthetic pop CSV: ~500MB; profiles: 371 × 24h |
| **Co-simulation** | HELICS federation | **Sequential only** — no parallel federate execution |

### Scenario / Policy Scale

| Dimension | Review Implies | Codebase Implements |
|-----------|----------------|---------------------|
| **Scenarios** | Implied multiple | **3 scenarios** (BASE, TELE, MBUF) × 5 years × coupled/uncoupled |
| **Policy levers** | Tipping points/interventions | **WFH rate**, **congestion fee**, **land use reallocation** |
| **Uncertainty quantification** | Ensembles/RDM | **GA multi-restart ensembles** only; no formal RDM |
| **Intervention discovery** | Automated (NEW) | **Manual** scenario comparison only |

---

## Key Scale Mismatches

### 1. **Co-simulation Architecture** — FUNDAMENTAL MISMATCH
- **Review**: Federated simulators (HELICS) exchanging data at each time step
- **Codebase**: Monolithic sequential Python pipeline — stages run once, pass DataFrames
- **Impact**: Cannot capture feedback loops (e.g., congestion → land use → travel → congestion)

### 2. **Building Physics Fidelity** — 2-3 ORDERS OF MAGNITUDE
- **Review**: EnergyPlus/OpenStudio (sub-hourly, 1000s of thermal zones per building)
- **Codebase**: EUI × floor area × occupancy elasticity (hourly, 1 zone per SA2 per type)
- **Impact**: No thermal comfort, HVAC dynamics, demand response, or peak shifting fidelity

### 3. **Power System Scope** — MISSING ENTIRE LAYER
- **Review**: OpenDSS (distribution) + PyPSA (transmission/generation)
- **Codebase**: OpenDSS-input-generation only (distribution, synthetic, no solver)
- **Impact**: No capacity expansion, no renewable integration studies, no reliability metrics

### 4. **Agent Behavioural Depth** — ABSENT
- **Review**: Mesa agents for adoption/investment decisions
- **Codebase**: Zero agent framework; all behaviour in estimated discrete choice models
- **Impact**: Cannot simulate emergent adoption dynamics, social contagion, learning

### 5. **Traffic Assignment** — ONE-WAY ONLY
- **Review**: MATSim/SUMO dynamic traffic assignment (iterative)
- **Codebase**: Static OMX skims → pipeline → DynusT `vehicle.dat` (no feedback)
- **Impact**: No congestion-induced rerouting, no induced demand, no VOT sensitivity

### 6. **Land Use Resolution** — AGGREGATE VS. PARCEL
- **Review**: UrbanSim (parcel-level, hedonic prices, developer decisions)
- **Codebase**: SA2-level accessibility reallocation (gravity-style, fixed control totals)
- **Impact**: No gentrification, densification, infill, or parcel-scale policy testing

---

## What's Better in Codebase Than Review

| Capability | Review Mentions | Codebase Delivers |
|------------|-----------------|-------------------|
| **GA variable selection** | No | Full GA across 7 stages with statistical rigour |
| **Polaris-style destination choice** | No | Mode-conditional destination, planning-constrained sets |
| **JAX/Larch acceleration** | No | GPU-ready nested logit for 371-zone choice sets |
| **HPC pipeline orchestration** | No | PBS scripts, checkpointing, stability validation |
| **Overture Maps integration** | OSMnx only | Buildings + Places + Transportation (richer than OSM) |
| **Joint household CDAP** | ActivitySim standard | + Interaction Coefficients (ActivitySim+) |
| **Zero-inflated frequency models** | Standard ordered logit | Handles excess zeros in non-mandatory activities |

---

## Recommendations by Scale Gap

| Gap | Priority | Effort | Approach |
|-----|----------|--------|----------|
| **HELICS federation** | Critical | High | Wrap each stage as HELICS federate; define data exchange APIs |
| **PyPSA regional grid** | Critical | High | Build transmission network from Powerlink data; link to distribution |
| **Mesa agent framework** | High | Medium | Implement EV/solar/mode-shift agents; couple to pipeline outputs |
| **DTA feedback loop** | High | Medium | MATSim/SUMO iterative skim update (weekly cycles) |
| **EnergyPlus archetypes** | Medium | Medium | Calibrate 5–10 representative buildings; couple to CityBES aggregate |
| **Parcel land use** | Medium | High | Integrate Queensland cadastre; downscale SA2 reallocation |
| **RDM/uncertainty workflow** | Medium | Medium | SALib + scenario discovery on LUTI outputs |
| **Tipping point detection** | Low | Medium | Early-warning indicators on coupled scenario trajectories |

---

## Verdict

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Travel Behaviour Core** | ⭐⭐⭐⭐⭐ | Exceeds review; production-grade ABM |
| **Land Use / UrbanSim** | ⭐⭐⭐⭐ | Lightweight but functional at SA2 scale |
| **Buildings / Energy** | ⭐⭐⭐ | Good proxies; 2–3 orders below EnergyPlus fidelity |
| **Power Systems** | ⭐⭐⭐ | Distribution synthetic only; regional missing |
| **Co-simulation** | ⭐ | Sequential only; HELICS absent |
| **Agent / Behavioural** | ⭐ | No Mesa/agent framework |
| **Spatial Data** | ⭐⭐⭐⭐⭐ | Overture > OSM; excellent coverage |
| **Uncertainty / Innovation** | ⭐⭐ | Ensembles yes; RDM/tipping points no |

**The codebase is a production-grade travel demand pipeline with integrated land-use/energy/power *proxies*. The review document describes a *true multi-physics co-simulation platform* — the gaps are exactly the integration layer (HELICS, PyPSA, Mesa) and high-fidelity physics (EnergyPlus, parcel UrbanSim, dynamic DTA).**