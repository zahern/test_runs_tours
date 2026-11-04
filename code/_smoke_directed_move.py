#!/usr/bin/env python
"""Smoke test for BanditFeatureSearcher._directed_candidate() - the
'drop-the-weakest, try-a-substitute' analyst-like perturbation.

Run: python _smoke_directed_move.py
"""
from __future__ import annotations

import numpy as np

from ga_hypothesis_testing import BanditFeatureSearcher, STAGE_COLLINEAR_GROUPS


def _fake_fitness(genes: np.ndarray) -> float:
    return -abs(int(genes.sum()) - 4) - 0.01


def run_smoke_test() -> bool:
    feature_names = ['DIST', 'SOVTOLL_TIME__AM', 'TOTEMP', 'FPSEMPN', 'extra_1', 'extra_2']
    collinear_groups = [['DIST', 'SOVTOLL_TIME__AM'], ['TOTEMP', 'FPSEMPN']]

    bandit = BanditFeatureSearcher(
        n_features=len(feature_names), fitness_fn=_fake_fitness,
        pop_size=4, n_generations=2, min_features=2, max_features=5,
        seed=42, verbose=False, feature_names=feature_names,
        collinear_groups=collinear_groups,
    )

    all_pass = True

    # Case 1: DIST is active and known-weak; SOVTOLL_TIME__AM (its collinear
    # group-mate) is off. Directed move should drop DIST and turn on
    # SOVTOLL_TIME__AM as the substitute.
    cur_genes = np.array([1, 0, 1, 0, 1, 0])  # DIST, TOTEMP, extra_1 active
    bandit._weakness = {'DIST': 6.0}  # stuck/at-bound-tier weakness
    cand = bandit._directed_candidate(cur_genes)
    ok1 = cand[0] == 0 and cand[1] == 1
    print(f"Case 1 (drop weak DIST, substitute SOVTOLL_TIME__AM): "
          f"{'PASS' if ok1 else 'FAIL'}  genes={cand}")
    all_pass &= ok1

    # Case 2: no weakness signal at all -> should be a no-op (returns copy unchanged)
    bandit._weakness = {}
    cand2 = bandit._directed_candidate(cur_genes)
    ok2 = np.array_equal(cand2, cur_genes)
    print(f"Case 2 (no weakness signal -> no-op): {'PASS' if ok2 else 'FAIL'}  genes={cand2}")
    all_pass &= ok2

    # Case 3: weak feature has NO group-mate (extra_1) -> should just drop it,
    # no substitute available, then bounds repair may re-add something else.
    bandit._weakness = {'extra_1': 6.0}
    cand3 = bandit._directed_candidate(cur_genes)
    ok3 = cand3[4] == 0  # extra_1 dropped
    print(f"Case 3 (drop weak feature with no group-mate): "
          f"{'PASS' if ok3 else 'FAIL'}  genes={cand3}")
    all_pass &= ok3

    # Case 4: run() end-to-end for a couple generations, just confirm no crash
    # and constraints hold on the final best.
    try:
        best, hist, pop = bandit.run_with_restarts(n_restarts=0)
        feat_idx = {f: i for i, f in enumerate(feature_names)}
        violation = False
        for group in collinear_groups:
            idxs = [feat_idx[f] for f in group]
            if sum(best.genes[i] for i in idxs) > 1:
                violation = True
        ok4 = not violation
        print(f"Case 4 (end-to-end run, no collinearity violation in best): "
              f"{'PASS' if ok4 else 'FAIL'}  best.genes={best.genes}")
        all_pass &= ok4
    except Exception as e:
        print(f"Case 4 (end-to-end run): FAIL - exception {e!r}")
        all_pass = False

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


if __name__ == '__main__':
    ok = run_smoke_test()
    raise SystemExit(0 if ok else 1)
