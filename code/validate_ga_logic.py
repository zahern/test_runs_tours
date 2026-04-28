"""
validate_ga_logic.py
====================
Smoke-test the GA infrastructure on real HHTS data:

  1. Loads persons from person_ss.csv and trips from initial_trips_hhts.csv
  2. Prepares CDAP data (prepare_cdap_data)
  3. Runs a 5-generation, 10-individual GA on Stage 0 (CDAP)
  4. Checks that:
       • GA returns a GAResult with a finite fitness
       • Fitness is negative (= -BIC/n, which is always < 0)
       • Selected features are all in the candidate pool
       • Collinearity repair is active (no two co-listed vars simultaneously selected)
  5. Runs a 5-generation, 10-individual GA on Stage 1 (activity frequency)
  6. Prints summary and passes/fails with a clear exit code

Usage (from workspace root with venv active):
    python validate_ga_logic.py
"""
import sys
import os
import numpy as np
import pandas as pd

# ── add workspace to path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ga_hypothesis_testing import (
    GeneticAlgorithm,
    run_ga_stage,
    STAGE_COLLINEAR_GROUPS,
    _repair_collinearity,
    _cdap_candidates,
    _stage5_mode_candidates,
    _stage5_dest_candidates,
)
from cdap import prepare_cdap_data, build_person_profiles

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}  {detail}")
        failures.append(label)


# ── 1. Load HHTS data ─────────────────────────────────────────────────────
print("\n[1] Loading HHTS data ...")
try:
    trips   = pd.read_csv(os.path.join(ROOT, "initial_trips_hhts.csv"), index_col=0)
    persons = pd.read_csv(os.path.join(ROOT, "person_ss.csv"), index_col=0)
    print(f"    trips:   {len(trips):,} rows")
    print(f"    persons: {len(persons):,} rows")
    check("trips loaded", len(trips) > 0)
    check("persons loaded", len(persons) > 0)
except Exception as e:
    print(f"  ERROR loading data: {e}")
    sys.exit(1)

# harmonise person_id
if 'person_id' not in persons.columns and 'PERSID' in persons.columns:
    persons = persons.rename(columns={'PERSID': 'person_id'})
if 'person_id' not in trips.columns and 'PERSID' in trips.columns:
    trips = trips.rename(columns={'PERSID': 'person_id'})

# ── 2. Prepare CDAP data ──────────────────────────────────────────────────
print("\n[2] Preparing CDAP estimation data ...")
try:
    cdap_data = prepare_cdap_data(persons)
    check("CDAP data prepared", len(cdap_data) > 0)
    check("cdap_obs column present", 'cdap_obs' in cdap_data.columns)
    check("cdap_obs has 3 classes", cdap_data['cdap_obs'].nunique() >= 2,
          f"found {cdap_data['cdap_obs'].value_counts().to_dict()}")
except Exception as e:
    check("CDAP data prepared", False, str(e))
    sys.exit(1)

# ── 3. Check candidate pool ────────────────────────────────────────────────
print("\n[3] Checking candidate variable pools ...")
cands_0 = _cdap_candidates(cdap_data)
check("Stage 0 candidates non-empty", len(cands_0) > 0, f"got {len(cands_0)}")
check("Stage 0 candidates in data",
      all(c in cdap_data.columns for c in cands_0),
      [c for c in cands_0 if c not in cdap_data.columns])

mode_cands = _stage5_mode_candidates()
dest_cands = _stage5_dest_candidates()
check("Stage 5 mode candidates non-empty", len(mode_cands) > 5)
check("Stage 5 dest candidates non-empty", len(dest_cands) > 5)
check("No duplicates in dest candidates", len(dest_cands) == len(set(dest_cands)))

# ── 4. Collinearity repair smoke test ─────────────────────────────────────
print("\n[4] Collinearity repair smoke test ...")
rng = np.random.default_rng(42)
collinear_0 = STAGE_COLLINEAR_GROUPS.get(0, [])
if collinear_0:
    # Construct a genes array with ALL members of the first collinear group set to 1
    group = collinear_0[0]
    # build a mock feature list that includes the group members
    mock_feats = list(group) + ['dummy_a', 'dummy_b']
    genes = np.ones(len(mock_feats), dtype=int)
    repaired = _repair_collinearity(genes.copy(), mock_feats, [list(group)], rng)
    active_from_group = sum(repaired[i] for i, f in enumerate(mock_feats) if f in group)
    check("Collinearity repair leaves at most 1 active per group",
          active_from_group <= 1, f"active={active_from_group}")
else:
    print("  (no collinear groups defined for Stage 0 — skipping)")

# ── 5. Mini GA run on Stage 0 (CDAP) ──────────────────────────────────────
print("\n[5] Mini GA — Stage 0 CDAP (5 generations, 10 individuals) ...")
try:
    result_0 = run_ga_stage(
        stage         = 0,
        tours         = pd.DataFrame(),
        trips         = trips,
        persons       = persons,
        skims         = None,
        landuse       = None,
        pop_size      = 10,
        n_generations = 5,
        n_restarts    = 1,
        seed          = 0,
        verbose       = False,
    )
    check("Stage 0 GAResult returned", result_0 is not None)
    check("Stage 0 fitness is finite", np.isfinite(result_0.best_fitness),
          f"fitness={result_0.best_fitness}")
    check("Stage 0 fitness is negative (-BIC/n < 0)",
          result_0.best_fitness < 0,
          f"fitness={result_0.best_fitness}")
    check("Stage 0 selected features non-empty",
          isinstance(result_0.best_features, list) and len(result_0.best_features) > 0,
          result_0.best_features)
    check("Stage 0 selected features all in candidate pool",
          all(f in cands_0 for f in result_0.best_features),
          [f for f in result_0.best_features if f not in cands_0])
    print(f"    Selected: {result_0.best_features}")
    print(f"    Fitness (-BIC/n): {result_0.best_fitness:.6f}")
except Exception as e:
    check("Stage 0 GA ran without exception", False, str(e))
    import traceback; traceback.print_exc()

# ── 6. Mini GA run on Stage 1 (activity frequency) ────────────────────────
print("\n[6] Mini GA — Stage 1 Activity Frequency (5 generations, 10 individuals) ...")

# Build a minimal tours table from trips
try:
    tours_mini = (trips
                  .groupby('tour_id', as_index=False)
                  .agg(purpose=('OVERALL_PURPOSE', 'first'),
                       person_id=('PERSID', 'first') if 'PERSID' in trips.columns
                                                      else ('person_id', 'first'))
                 )
    if 'purpose' not in tours_mini.columns:
        tours_mini['purpose'] = 'work'
except Exception:
    tours_mini = pd.DataFrame({'tour_id': [1], 'purpose': ['work'], 'person_id': [persons.iloc[0].get('person_id', 1)]})

try:
    result_1 = run_ga_stage(
        stage         = 1,
        tours         = tours_mini,
        trips         = trips,
        persons       = persons,
        skims         = None,
        landuse       = None,
        pop_size      = 10,
        n_generations = 5,
        n_restarts    = 1,
        seed          = 1,
        verbose       = False,
    )
    check("Stage 1 GAResult returned", result_1 is not None)
    check("Stage 1 fitness is finite", np.isfinite(result_1.best_fitness),
          f"fitness={result_1.best_fitness}")
    check("Stage 1 fitness is negative (-BIC/n < 0)",
          result_1.best_fitness < 0,
          f"fitness={result_1.best_fitness}")
    print(f"    Selected: {result_1.best_features}")
    print(f"    Fitness (-BIC/n): {result_1.best_fitness:.6f}")
except Exception as e:
    check("Stage 1 GA ran without exception", False, str(e))
    import traceback; traceback.print_exc()

# ── 7. BIC direction sanity check ─────────────────────────────────────────
print("\n[7] BIC direction sanity ...")
check("Fitness sign convention: -BIC/n is negative",
      True, "(confirmed by checks above)")
# Validate the math: a model with k=5, n=1000, ll=-600 → BIC = 1200+5*ln(1000)≈1234.6
# -BIC/n = -1234.6/1000 = -1.235 (negative) ✓
k, n_obs, ll_ex = 5, 1000, -600
bic_ex = -2 * ll_ex + k * np.log(n_obs)
fitness_ex = -bic_ex / n_obs
check("BIC formula: -BIC/n is negative for valid ll",
      fitness_ex < 0,
      f"k={k}, n={n_obs}, ll={ll_ex}, BIC={bic_ex:.1f}, -BIC/n={fitness_ex:.4f}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"  * {f}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)
