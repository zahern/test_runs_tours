"""Regression test for lc_rail_guard (run: python _test_lc_rail_guard.py)."""
import numpy as np

from lc_rail_guard import lc_result_rail, rail_flags


def main():
    bounds = {"mu_ln_dist": (-6.0, 3.0), "beta_ln_dist": (-10.0, -1e-3)}

    r = rail_flags({"mu_ln_dist__c1": 2.9995}, None, bounds)
    assert r["railed"], r
    assert "upper bound" in " ".join(r["rail"]), r

    r = rail_flags({"mu_ln_dist": 1.2, "sigma_ln_dist": 0.4},
                   {"mu_ln_dist": 0.2, "sigma_ln_dist": 0.3}, bounds)
    assert not r["railed"], r
    assert not r["warnings"], r

    r = rail_flags({"beta_x": -18.7}, {"beta_x": 9.99e7})
    assert r["railed"], r
    assert "degenerate SE" in " ".join(r["rail"]), r

    r = rail_flags({"beta_x": 1.0}, {"beta_x": 0.0})
    assert not r["railed"] and r["warnings"], r

    names = ["log_dist", "crash"]
    clean_b = [np.array([-1.0, 0.2]), np.array([-0.5, 0.4])]
    clean_s = [np.array([0.10, 0.20]), np.array([0.10, 0.30])]
    r = lc_result_rail(names, clean_b, clean_s, [0.6, 0.4])
    assert not r["railed"] and not r["warnings"], r

    pinned_b = [np.array([15.0, 0.2]), np.array([-0.5, 0.4])]
    r = lc_result_rail(names, pinned_b, clean_s, [0.6, 0.4])
    assert r["railed"] and "beta bound" in " ".join(r["rail"]), r

    degen_s = [np.array([0.10, 9.99e7]), np.array([0.10, 0.30])]
    r = lc_result_rail(names, clean_b, degen_s, [0.6, 0.4])
    assert r["railed"] and "degenerate SE" in " ".join(r["rail"]), r

    r = lc_result_rail(names, clean_b, clean_s, [0.995, 0.005])
    assert r["railed"] and "collapsed" in " ".join(r["rail"]), r

    r = lc_result_rail(names, [clean_b[0], clean_b[0].copy()],
                       clean_s, [0.5, 0.5])
    assert r["railed"] and "identical" in " ".join(r["rail"]), r

    r = lc_result_rail(names, clean_b, clean_s, [0.995, 0.005],
                       method="grouped")
    assert not r["railed"], r

    r = lc_result_rail(names, clean_b, clean_s, [0.5, 0.5],
                       membership_coefs=np.array([[0.0, 75.0]]))
    assert r["railed"] and "gamma" in " ".join(r["rail"]), r

    print("LC-RAIL-GUARD-TESTS-PASS")


if __name__ == "__main__":
    main()
