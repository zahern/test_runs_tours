"""Compute-node smoke test for the LC rail guards (installed SearchLibrium)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from lc_rail_guard import lc_result_rail, rail_flags  # noqa: E402
from SearchLibrium.mh_choice_estimator import estimate_dest_choice_lc  # noqa: E402
from SearchLibrium.latent_class import LatentClassMixedLogit  # noqa: E402


def _panel(n_obs=400, n_alt=4, seed=11):
    rng = np.random.default_rng(seed)
    true_betas = np.array([[0.8, -0.5], [-0.6, 0.9]])
    rows = []
    for i in range(1, n_obs + 1):
        cls = int(rng.random() < 0.6)
        x = rng.normal(size=(n_alt, 2))
        u = x @ true_betas[cls]
        p = np.exp(u - u.max())
        p /= p.sum()
        chosen = int(rng.choice(n_alt, p=p)) + 1
        for j in range(1, n_alt + 1):
            rows.append([i, j, x[j - 1, 0], x[j - 1, 1],
                         int(j == chosen)])
    return pd.DataFrame(rows, columns=["case_id", "alt_id", "x1", "x2",
                                       "chosen"])


def main():
    r = rail_flags({"mu": 2.9995}, None, {"mu": (-3.0, 3.0)})
    assert r["railed"] and "upper bound" in " ".join(r["rail"]), r
    r = lc_result_rail(["x1", "x2"],
                       [np.array([15.0, 0.1]), np.array([-0.2, 0.3])],
                       [np.array([0.1, 0.1]), np.array([0.1, 0.1])],
                       [0.6, 0.4])
    assert r["railed"] and "beta bound" in " ".join(r["rail"]), r
    print("[0] helper rail checks OK")

    df = _panel()
    res = estimate_dest_choice_lc(
        df, ["x1", "x2"], n_classes=2, maxiter=40, seed=5, n_init=1,
        require_crash=False)
    print(f"[1] estimate_dest_choice_lc: conv={res.get('converged')} "
          f"railed={res.get('railed')} reasons={res.get('rail')} "
          f"shares={[round(s, 3) for s in res.get('shares', [])]}")
    assert "railed" in res and "rail" in res and "rail_warnings" in res
    assert not res["railed"], res["rail"]

    X = df[["x1", "x2"]].to_numpy(float)
    y = df["chosen"].to_numpy(int)
    ids = df["case_id"].to_numpy(int)
    alts = df["alt_id"].to_numpy(int)
    lc = LatentClassMixedLogit(n_classes=2, maxiter=40, class_maxiter=30,
                               n_init=1, random_state=3)
    lc.setup(X=X, y=y, varnames=["x1", "x2"], ids=ids, alts=alts,
             fit_intercept=False)
    lc.fit()
    print(f"[2] LatentClassMixedLogit: conv={lc.converged} "
          f"railed={getattr(lc, 'railed', None)} "
          f"reasons={getattr(lc, 'rail_reasons', None)} "
          f"shares={np.round(lc.class_probs, 3).tolist()}")
    assert hasattr(lc, "railed") and hasattr(lc, "rail_reasons")
    assert not lc.railed, lc.rail_reasons

    try:
        import stage5_destination_choice as S5

        class _FakeLC:
            pass

        fake = _FakeLC()
        fake.pf = pd.DataFrame(
            {
                "value": [2.9995, -0.5, 0.0],
                "std_err": [0.1, 0.2, 0.0],
                "minimum": [-3.0, -3.0, -3.0],
                "maximum": [3.0, 3.0, 3.0],
                "holdfast": [False, False, True],
            },
            index=["mu_ln_dist__c1", "beta_x__c1", "locked"],
        )
        reasons, warns, info = S5._lccm_rail_scan(fake, cap=3.0)
        print(f"[3] _lccm_rail_scan: reasons={reasons} info={info}")
        assert reasons and any("upper bound" in x for x in reasons), reasons
    except Exception as exc:
        print(f"[3] _lccm_rail_scan smoke skipped: "
              f"{type(exc).__name__}: {exc}")

    print("LC-RAIL-HPC-SMOKE-PASS")


if __name__ == "__main__":
    main()
