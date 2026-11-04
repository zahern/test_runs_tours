"""Minimal Larch sparse-vs-dense MNL test (no pipeline wrappers)."""
import numpy as np
import pandas as pd
import larch as lx
from larch import P, X


def make(N=400, J=10, U=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    beta = (0.8, -0.4)
    universe = np.arange(1000, 1000 + U)
    for i in range(N):
        alts = rng.choice(universe, size=J, replace=False)
        v = beta[0] * rng.random(J) + beta[1] * rng.random(J)
        g = -np.log(-np.log(rng.random(J)))
        c = int(np.argmax(v + g))
        for j, a in enumerate(sorted(alts)):
            rows.append((i, int(a), rng.random(), rng.random(), 1 if a == alts[c] else 0))
    return pd.DataFrame(rows, columns=["case_id", "alt_id", "x1", "x2", "chosen"])


def run(kind, engine="jax"):
    df = make()
    if kind == "sparse":
        alg = df.groupby("case_id")["alt_id"].apply(
            lambda s: np.all(np.diff(s.values) > 0) and False).any()
        ca = df.set_index(["case_id", "alt_id"])
        d = lx.Dataset.construct.from_idce(ca, crack=False)
    else:
        ca = df.set_index(["case_id", "alt_id"])
        d = lx.Dataset.construct.from_idca(ca, crack=False)
    m = lx.Model(d, compute_engine=engine)
    m.utility_ca = P("b1") * X("x1") + P("b2") * X("x2")
    m.choice_ca_var = "chosen"
    res = m.maximize_loglike(method="slsqp", quiet=True)
    pf = m.pf["value"]
    print(f"RESULT {kind}/{engine}: LL={res.loglike:.1f} "
          f"b1={float(pf.get('b1', np.nan)):.4f} b2={float(pf.get('b2', np.nan)):.4f}")
    return m


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    engine = sys.argv[2] if len(sys.argv) > 2 else "jax"
    run(kind, engine)
    print("MIN_TEST_OK")
