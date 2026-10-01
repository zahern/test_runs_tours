"""Isolate the zero-gradient / sparse behaviour of the trip-dest fit."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _test_tripdest_idce as t  # noqa: E402
from stage5b_trip_destination import _fit_idca  # noqa: E402


def main() -> int:
    ca, beta = t._make_ca()
    for sparse in (1, 0):
        for force_jax in (1, 0):
            os.environ["STAGE5B_IDCE"] = str(sparse)
            os.environ["LARCH_FORCE_JAX"] = str(force_jax)
            r = _fit_idca(ca.copy(), {"TotalEmployment": 1},
                          int(ca["case_id"].nunique()), "work")
            tag = f"idce={sparse} force_jax={force_jax}"
            if r is None:
                print(f"TEST {tag}: None")
                continue
            pf = r["params"]["value"]
            print(f"TEST {tag}: LL={r['log_likelihood']:.1f} "
                  f"engine={r['engine']} " +
                  " ".join(f"{k}={float(v):.4f}" for k, v in pf.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
