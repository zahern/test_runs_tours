"""_fit_frame_mnl: does the fit reach the optimum, and is the verdict honest?

Builds a small sampled choice-set frame from a known MNL, then checks that
  * a normal fit reports success and a small gradient;
  * a fit truncated by ``maxiter`` is re-started and recovers the optimum
    instead of returning the truncated point;
  * the returned result carries the gradient evidence.
"""
import numpy as np
import pandas as pd

from SearchLibrium.mh_choice_estimator import ChoiceSetFrame, _fit_frame_mnl

RNG = np.random.default_rng(11)
N_CASES, N_ALTS, N_FEAT = 1200, 4, 3
BETA_TRUE = np.array([1.1, -0.7, 0.45])


def build_frame(seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for case in range(N_CASES):
        x = rng.standard_normal((N_ALTS, N_FEAT))
        v = x @ BETA_TRUE
        p = np.exp(v - v.max())
        p = p / p.sum()
        chosen = int(rng.choice(N_ALTS, p=p))
        for a in range(N_ALTS):
            rows.append({
                "case_id": case, "alt_id": a, "chosen": a == chosen,
                "log_correction": 0.0,
                **{f"f{k}": x[a, k] for k in range(N_FEAT)},
            })
    return pd.DataFrame(rows)


def fit(df, **kw):
    fr = ChoiceSetFrame.from_long(df, [f"f{k}" for k in range(N_FEAT)])
    beta, ll, res = _fit_frame_mnl(fr, np.zeros(fr.dimension), **kw)
    return fr, beta, ll, res


df = build_frame()

print("== normal fit")
fr, beta, ll, res = fit(df)
print(f"   success={res.success} evidence={res.converged_evidence} "
      f"grad={res.grad_inf:.2e} tol={res.grad_tol:.2e} ll={ll:.3f}")
print(f"   beta={np.round(beta, 3)} (true {BETA_TRUE})")
assert res.success, "a converged fit must report success"
assert res.grad_inf <= res.grad_tol, "converged fit has a large gradient"
assert np.allclose(beta, BETA_TRUE, atol=0.15), "coefficients did not recover"

print("== truncated fit (maxiter=2)")
fr2, beta2, ll2, res2 = fit(df, maxiter=2)
print(f"   success={res2.success} evidence={res2.converged_evidence} "
      f"grad={res2.grad_inf:.2e} ll={ll2:.3f}")
assert ll2 >= ll - 1e-6, "the restart returned a worse fit than the full fit"
assert res2.success, "the restarted fit should end at a stationary point"
assert res2.grad_inf <= res2.grad_tol

print("== evidence fields")
for field in ("grad_inf", "grad_tol", "fit_message", "converged_evidence"):
    assert hasattr(res, field), f"missing {field}"

print("\nFIT-FRAME-MNL CHECKS PASSED")
