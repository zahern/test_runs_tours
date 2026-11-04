#!/usr/bin/env python
"""
smoke_test_dest.py — destination-model smoke tests (Larch + overspecification).
================================================================================
Independent checks that the Larch destination-choice machinery WORKS, modelled on
the Exampville destination example (size/quantity term + distance utility):

  1. basic dest MNL on synthetic data estimates & converges with SENSIBLE,
     NON-ZERO coefficients (guards the "flat 0 coefficients" failure);
  2. the Larch quantity_ca size term works;
  3. larch.model.troubleshooting.overspecification FLAGS a deliberately-
     redundant (linearly-dependent) parameter — the same detection now wired into
     _estimate (_dest_overspecified_params) to lock unidentified params in-search.

Run in zigenv (needs larch):  python smoke_test_dest.py
Exits 0 on pass, 1 on failure, 0+SKIP when larch is unavailable.
"""
import sys
import numpy as np
import pandas as pd


def _skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(0)


try:
    import larch as lx
    from larch import P, X
except Exception as e:                        # pragma: no cover
    _skip(f"larch not importable ({e}) — run in zigenv")


def _synth(n_cases=600, n_alts=20, seed=0, dup=False):
    """idca frame: per (case, alt) DIST + SIZE + chosen (true b_dist≈-0.12)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cases):
        dist = rng.uniform(1, 40, n_alts)
        size = rng.uniform(1, 100, n_alts)
        u = -0.12 * dist + np.log(size) + rng.gumbel(0, 1, n_alts)
        chosen = int(np.argmax(u))
        for a in range(n_alts):
            r = {'case': c, 'alt': a, 'DIST': dist[a], 'SIZE': size[a],
                 'chosen': 1.0 if a == chosen else 0.0}
            if dup:
                r['DIST2'] = dist[a]          # exact duplicate → overspecified
            rows.append(r)
    return lx.Dataset.construct.from_idca(pd.DataFrame(rows).set_index(['case', 'alt']))


def test_basic_dest():
    print("[1] basic dest model (distance + size term)")
    m = lx.Model(_synth())
    m.utility_ca = P('b_dist') * X('DIST')
    m.quantity_ca = P('b_size') * X('SIZE')
    m.choice_ca_var = 'chosen'
    r = m.maximize_loglike(stderr=True)
    b = float(m.pf.loc['b_dist', 'value'])
    print(f"    b_dist={b:.3f} (true ~-0.12)  LL={float(r.loglike):.1f}")
    assert abs(b) > 1e-3, "b_dist collapsed to ~0 — NOT identified (flat-zero failure)"
    assert b < 0, "b_dist has the wrong sign"
    print("    PASS")


def test_overspecification():
    print("[2] overspecification detects a linearly-dependent duplicate")
    from larch.model.troubleshooting import overspecification
    m = lx.Model(_synth(dup=True))
    m.utility_ca = (P('b_dist') * X('DIST') + P('b_dist2') * X('DIST2')
                    + P('b_size_lin') * X('SIZE'))
    m.choice_ca_var = 'chosen'
    try:
        m.maximize_loglike()
    except Exception:
        pass
    try:
        _out = overspecification(m, repair='?', verbose=0)
    except TypeError:
        _out = overspecification(m)
    diag = _out[1] if isinstance(_out, tuple) and len(_out) > 1 else _out
    n = 0 if diag is None else len(diag)
    print(f"    overspecification diagnosis rows: {n}")
    assert n > 0, "overspecification did NOT flag the duplicate DIST/DIST2 pair"
    print("    PASS")


def test_helper_locks():
    print("[3] _dest_overspecified_params locks the redundant param")
    try:
        from stage5_destination_choice import _dest_overspecified_params
    except Exception as e:
        print(f"    SKIP (import: {e})"); return
    m = lx.Model(_synth(dup=True))
    m.utility_ca = (P('b_dist') * X('DIST') + P('b_dist2') * X('DIST2')
                    + P('b_size_lin') * X('SIZE'))
    m.choice_ca_var = 'chosen'
    try:
        m.maximize_loglike()
    except Exception:
        pass
    locked = _dest_overspecified_params(m, protected={'b_dist', 'b_size_lin'})
    print(f"    locked: {locked}")
    assert locked, "_dest_overspecified_params locked nothing (expected b_dist2)"
    print("    PASS")


if __name__ == '__main__':
    print("=== destination model smoke tests ===")
    _fail = 0
    for _t in (test_basic_dest, test_overspecification, test_helper_locks):
        try:
            _t()
        except AssertionError as e:
            print(f"    FAIL: {e}"); _fail += 1
        except Exception as e:
            print(f"    ERROR: {e}"); _fail += 1
    print("=" * 40)
    print("ALL SMOKE TESTS PASSED" if _fail == 0 else f"{_fail} SMOKE TEST(S) FAILED")
    sys.exit(1 if _fail else 0)
