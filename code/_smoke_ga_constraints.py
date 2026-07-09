#!/usr/bin/env python
"""Smoke test for the collinearity / require-one chromosome constraints.

Self-contained (no real HHTS/synth-pop data needed) - exercises
GeneticAlgorithm, SimulatedAnnealing, and BanditFeatureSearcher directly on a
synthetic feature set built from the actual '5_dest' and '5_mode'
STAGE_COLLINEAR_GROUPS / STAGE_REQUIRE_ONE_GROUPS, with a cheap random
fitness function. After each run, every chromosome the searcher ever
evaluated (via its fitness cache) is checked against both constraint types.

Run: python _smoke_ga_constraints.py
"""
from __future__ import annotations

import numpy as np

from ga_hypothesis_testing import (
    GeneticAlgorithm,
    SimulatedAnnealing,
    BanditFeatureSearcher,
    STAGE_COLLINEAR_GROUPS,
    STAGE_REQUIRE_ONE_GROUPS,
)


def _decode_key(key, n_features):
    """Cache keys vary by class: GA/Bandit use genes.tobytes(), SA uses tuple(genes)."""
    if isinstance(key, (bytes, bytearray)):
        genes = np.frombuffer(key, dtype=int)
        if genes.size != n_features:
            # int dtype width mismatch fallback - try platform default int
            genes = np.frombuffer(key, dtype=np.int64)
        return genes
    return np.asarray(key)


def _check_chromosomes(cache_keys, feature_names, collinear_groups, require_one_groups, label):
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    n_checked = 0
    violations = []
    for key in cache_keys:
        genes = _decode_key(key, len(feature_names))
        n_checked += 1
        for group in collinear_groups:
            idxs = [feat_idx[f] for f in group if f in feat_idx]
            n_active = sum(genes[i] for i in idxs)
            if n_active > 1:
                violations.append(('collinear>1', group, key))
        for group in require_one_groups:
            idxs = [feat_idx[f] for f in group if f in feat_idx]
            if idxs and sum(genes[i] for i in idxs) == 0:
                violations.append(('require_one=0', group, key))
    status = "PASS" if not violations else "FAIL"
    print(f"  [{label}] {status}: checked {n_checked} distinct chromosomes, "
          f"{len(violations)} violation(s)")
    for kind, group, key in violations[:5]:
        print(f"      {kind}: group={group}")
    return len(violations) == 0


def _fake_fitness(genes: np.ndarray) -> float:
    # Cheap deterministic-ish "fitness": reward moderate sparsity, no model fit.
    rng = np.random.default_rng(int(genes.sum()) * 7919 + hash(genes.tobytes()) % 997)
    return -abs(int(genes.sum()) - 6) - rng.random()


def run_smoke_test():
    # Stage 5 destination: collinearity-only (no require-one here anymore -
    # DEST_UEC already hardcodes c_log_dist/c_dist/c_logsum/c_size_term
    # unconditionally, outside anything the GA searches over, so forcing an
    # *enrichment* variable via require-one had no theoretical basis).
    feature_names = sorted({f for group in STAGE_COLLINEAR_GROUPS['5_dest'] for f in group}
                            | {'extra_feat_1', 'extra_feat_2', 'extra_feat_3'})
    collinear_groups = STAGE_COLLINEAR_GROUPS['5_dest']
    require_one_groups = STAGE_REQUIRE_ONE_GROUPS.get('5_dest', [])

    print(f"Feature set ({len(feature_names)}): {feature_names}")
    print(f"Collinear groups: {collinear_groups}")
    print(f"Require-one groups: {require_one_groups}\n")

    n_features = len(feature_names)
    all_pass = True

    common_kwargs = dict(
        n_features=n_features,
        fitness_fn=_fake_fitness,
        pop_size=12,
        n_generations=8,
        min_features=3,
        max_features=8,
        seed=123,
        verbose=False,
        feature_names=feature_names,
        collinear_groups=collinear_groups,
        require_one_groups=require_one_groups,
    )

    print("Running GeneticAlgorithm...")
    ga = GeneticAlgorithm(**common_kwargs)
    ga.run_with_restarts(n_restarts=1)
    all_pass &= _check_chromosomes(ga._cache.keys(), feature_names,
                                    collinear_groups, require_one_groups, "GeneticAlgorithm")

    print("Running SimulatedAnnealing...")
    sa = SimulatedAnnealing(**common_kwargs)
    sa.run_with_restarts(n_restarts=1)
    all_pass &= _check_chromosomes(sa._cache.keys(), feature_names,
                                    collinear_groups, require_one_groups, "SimulatedAnnealing")

    print("Running BanditFeatureSearcher...")
    bandit_kwargs = dict(common_kwargs)
    bandit_kwargs['pop_size'] = 6
    bandit_kwargs['n_generations'] = 10
    bandit = BanditFeatureSearcher(**bandit_kwargs)
    bandit.run_with_restarts(n_restarts=1)
    all_pass &= _check_chromosomes(bandit._cache.keys(), feature_names,
                                    collinear_groups, require_one_groups, "BanditFeatureSearcher")

    # ── Stage 6 (trip mode): union require-one across ALL impedance vars ────
    # No backbone UEC here, so this constraint is load-bearing: the model
    # must have SOME impedance measure, but not a specific forced skim.
    print("\nRunning GeneticAlgorithm on Stage 6 (impedance require-one, union) constraint...")
    s6_features = sorted({f for group in STAGE_COLLINEAR_GROUPS[6] for f in group}
                          | {f for group in STAGE_REQUIRE_ONE_GROUPS[6] for f in group}
                          | {'extra_feat_1', 'extra_feat_2'})
    s6_collinear = STAGE_COLLINEAR_GROUPS[6]
    s6_require = STAGE_REQUIRE_ONE_GROUPS[6]
    ga_s6 = GeneticAlgorithm(
        n_features=len(s6_features), fitness_fn=_fake_fitness,
        pop_size=12, n_generations=8, min_features=2, max_features=6,
        seed=7, verbose=False, feature_names=s6_features,
        collinear_groups=s6_collinear, require_one_groups=s6_require,
    )
    ga_s6.run_with_restarts(n_restarts=1)
    s6_pass = _check_chromosomes(ga_s6._cache.keys(), s6_features,
                                 s6_collinear, s6_require, "Stage 6 impedance require-one")
    # Extra check: the impedance union has >=1 active every time, AND it's not
    # always the SAME variable (i.e. this isn't secretly forcing one skim).
    feat_idx = {f: i for i, f in enumerate(s6_features)}
    impedance_idx = [feat_idx[f] for f in s6_require[0] if f in feat_idx]
    zero_violations = 0
    which_active = set()
    for key in ga_s6._cache.keys():
        genes = _decode_key(key, len(s6_features))
        active_here = [i for i in impedance_idx if genes[i] == 1]
        if not active_here:
            zero_violations += 1
        which_active.update(active_here)
    distinct_names = sorted(s6_features[i] for i in which_active)
    print(f"  [Stage 6 impedance] zero-active violations: {zero_violations}/{len(ga_s6._cache)}")
    print(f"  [Stage 6 impedance] distinct impedance vars seen active across all chromosomes: {distinct_names}")
    all_pass &= s6_pass and zero_violations == 0 and len(distinct_names) > 1

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


if __name__ == '__main__':
    ok = run_smoke_test()
    raise SystemExit(0 if ok else 1)
