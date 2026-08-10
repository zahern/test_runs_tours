"""
feature_search.py — compact GA feature selection for the pre-day models
=======================================================================
The stage models (1-6, CDAP, usual-location) are hypothesis-tested by the GA in
ga_hypothesis_testing.py. The pre-day numpy models (mobility_models,
wfh_telecommute) were fixed-spec. This gives them the same treatment with a small
self-contained binary-GA over their candidate feature set, scored by the model's
own BIC (which penalises useless features), so their specifications are searched
too — matching the "GA applies to every estimated model" pipeline design.

Kept deliberately small (these numpy fits are cheap): a binary-encoded GA with
tournament selection, uniform crossover, bit-flip mutation, elitism, a fitness
cache, and a minimum-feature floor. Pure numpy — testable in the base env.

Gate globally with GA_SEARCH_PREDAY=0 (default on).
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def preday_search_enabled() -> bool:
    return os.environ.get('GA_SEARCH_PREDAY', '1').strip() != '0'


def _force_min(mask: np.ndarray, min_feats: int, rng) -> np.ndarray:
    """Turn on random genes until at least min_feats are active."""
    if mask.sum() >= min_feats:
        return mask
    off = np.where(~mask)[0]
    if len(off) == 0:
        return mask
    need = int(min_feats - mask.sum())
    pick = rng.choice(off, size=min(need, len(off)), replace=False)
    mask = mask.copy()
    mask[pick] = True
    return mask


def _tournament(pop, fit, rng, k=3):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[int(np.argmax([fit[i] for i in idx]))]
    return pop[best]


def ga_feature_select(eval_fn: Callable[[np.ndarray], float], n_feats: int,
                      names: Optional[Sequence[str]] = None,
                      pop_size: int = 24, n_gen: int = 30, seed: int = 42,
                      min_feats: int = 1, mutation: float = 0.15,
                      elitism: int = 2, fixed: Optional[np.ndarray] = None):
    """Search a binary feature mask maximising eval_fn(mask) (e.g. -BIC/n).

    fixed: optional boolean array of genes forced ON (never dropped).
    Returns (best_mask, best_fitness). Falls back to the all-on mask if the
    search never beats it or n_feats is tiny."""
    if n_feats <= 1:
        return np.ones(n_feats, dtype=bool), eval_fn(np.ones(n_feats, dtype=bool))
    rng = np.random.default_rng(seed)
    fixed = (np.zeros(n_feats, bool) if fixed is None else np.asarray(fixed, bool))
    cache: dict = {}

    def ev(mask):
        mask = mask | fixed
        key = mask.tobytes()
        if key in cache:
            return cache[key], mask
        if int(mask.sum()) < min_feats:
            cache[key] = -1e18
            return -1e18, mask
        try:
            f = float(eval_fn(mask))
        except Exception as e:                       # pragma: no cover
            logger.debug(f"ga_feature_select eval failed: {e}")
            f = -1e18
        if not np.isfinite(f):
            f = -1e18
        cache[key] = f
        return f, mask

    # init population: random densities + the all-on baseline
    pop = []
    for _ in range(pop_size - 1):
        m = rng.random(n_feats) < rng.uniform(0.3, 0.7)
        pop.append(_force_min(m | fixed, min_feats, rng))
    pop.append(np.ones(n_feats, bool))
    scored = [ev(m) for m in pop]
    fit = [s for s, _ in scored]
    pop = [m for _, m in scored]

    for _g in range(n_gen):
        order = np.argsort(fit)[::-1]
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]
        new = pop[:elitism]
        while len(new) < pop_size:
            a = _tournament(pop, fit, rng)
            b = _tournament(pop, fit, rng)
            child = np.where(rng.random(n_feats) < 0.5, a, b)
            child = np.logical_xor(child, rng.random(n_feats) < mutation)
            child = _force_min(child | fixed, min_feats, rng)
            new.append(child)
        scored = [ev(m) for m in new]
        fit = [s for s, _ in scored]
        pop = [m for _, m in scored]

    best = int(np.argmax(fit))
    if names is not None:
        _sel = [n for n, b in zip(names, pop[best]) if b]
        logger.info(f"ga_feature_select: {int(pop[best].sum())}/{n_feats} features "
                    f"kept (fitness={fit[best]:.4f}): {_sel}")
    return pop[best], fit[best]


if __name__ == '__main__':
    # Self-check: recover the 3 informative features out of 10 (7 are noise) by
    # a BIC-style score = fit_quality - complexity.
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 10))
    beta = np.zeros(10); beta[[1, 4, 7]] = [1.5, -1.2, 0.9]
    y = X @ beta + rng.normal(size=n) * 0.5

    def score(mask):
        cols = np.where(mask)[0]
        Xs = np.column_stack([np.ones(n), X[:, cols]])
        coef, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        rss = float(np.sum((y - Xs @ coef) ** 2))
        k = Xs.shape[1]
        bic = n * np.log(rss / n) + k * np.log(n)
        return -bic / n

    mask, f = ga_feature_select(score, 10, names=[f'x{i}' for i in range(10)],
                                pop_size=20, n_gen=25)
    kept = set(np.where(mask)[0])
    print(f"kept features: {sorted(kept)}  (true informative = [1, 4, 7])")
    print("recovered all informative:", {1, 4, 7}.issubset(kept),
          "| dropped most noise:", len(kept - {1, 4, 7}) <= 3)
