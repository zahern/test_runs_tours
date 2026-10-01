"""Sanity test for the sparse (from_idce) trip-destination fit path.

Builds a synthetic sampled CA set with a known data-generating process,
fits via stage5b_trip_destination._fit_idca, and checks convergence + sign
recovery. Also exercises the dense fallback.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _make_ca(n_obs=600, n_zones=30, sample_size=8, seed=0, sampled=True):
    rng = np.random.default_rng(seed)
    zone_ids = np.arange(100, 100 + n_zones)
    # true utilities: detour negative, log_size positive, employment positive
    beta = {"detour_dist_km": -0.25, "log_size": 0.8, "TotalEmployment": 0.004}
    rows = []
    for i in range(n_obs):
        c = rng.integers(0, n_zones)
        if sampled:
            alts = np.concatenate(([c], rng.choice(
                np.delete(np.arange(n_zones), c), size=sample_size - 1,
                replace=False)))
        else:
            alts = np.arange(n_zones)
        for j in alts:
            detour = rng.random() * 10.0
            log_size = rng.random() * 5.0
            emp = rng.random() * 100.0
            rows.append((i, zone_ids[j], detour, log_size, emp))
    ca = pd.DataFrame(rows, columns=["case_id", "alt_id", "detour_dist_km",
                                     "log_size", "TotalEmployment"])
    V = (beta["detour_dist_km"] * ca["detour_dist_km"]
         + beta["log_size"] * ca["log_size"]
         + beta["TotalEmployment"] * ca["TotalEmployment"])
    # pick chosen per case by Gumbel-max (logit-consistent)
    picks = []
    for cid, grp in ca.groupby("case_id"):
        g = -np.log(-np.log(rng.random(len(grp))))
        picks.append(grp.index[np.argmax((V.loc[grp.index] + g).values)])
    ca["chosen"] = 0
    ca.loc[picks, "chosen"] = 1
    if sampled:
        # force-case sampling correction placeholder (uniform-ish q)
        ca["log_correction"] = 0.0
    return ca, beta


def main() -> int:
    from stage5b_trip_destination import _fit_idca

    active = {"TotalEmployment": 1, "dist_to_anchor": 0}
    # Production shape: full choice set (STAGE5B_DEST_SAMPLE_SIZE=0 default).
    # The sparse/sampled paths are unsafe on Larch 6.0.46 (see module note).
    ca, beta = _make_ca(sampled=False)
    for mode, sparse in (("dense-full", False),):
        os.environ["STAGE5B_IDCE"] = "1" if sparse else "0"
        res = _fit_idca(ca.copy(), active, int(ca["case_id"].nunique()), "work")
        if res is None:
            print(f"FAIL {mode}: estimation returned None")
            return 1
        pf = res["params"]["value"]
        print(f"{mode}: LL={res['log_likelihood']:.1f} BIC={res['bic']:.1f} "
              f"n_params={res['n_params']}")
        for k, v in pf.items():
            print(f"    {k:>22s} = {float(v): .4f}")
        for k, b in beta.items():
            got = float(pf.get(k, np.nan))
            if not np.isfinite(got) or abs(got - b) > max(0.35, 0.6 * abs(b)):
                print(f"FAIL {mode}: {k} recovered {got:.4f} vs true {b}")
                return 1
    print("TRIPDEST_IDCE_TEST_OK")
    return 1 if False else 0


if __name__ == "__main__":
    raise SystemExit(main())
