"""Smoke check: the latent-class arm reports its M-step optimality.

estimate_dest_choice_lc's `converged` is the EM's own log-likelihood change
test, so a class-level solve that stopped short of its optimum was invisible in
the arm's results. The M-step now records the gradient it finished on, and this
check asserts the report survives label switching (the gradient must travel
with the betas, or class 1's evidence would be attributed to class 2).
"""
import numpy as np
import pandas as pd

from SearchLibrium.mh_choice_estimator import estimate_dest_choice_lc

rng = np.random.default_rng(7)
N_CASES, N_ALTS = 400, 12
FEATS = ["log_DIST_1", "log_CRASH_1", "aggl"]
MEMB = ["incp", "vehrd"]

rows = []
truth = {
    0: np.array([-1.4, -0.9, 0.6]),
    1: np.array([-0.5, -0.15, 0.1]),
}
for case in range(N_CASES):
    incp = float(rng.integers(1, 4))
    vehrd = float(rng.integers(0, 3))
    share1 = 1.0 / (1.0 + np.exp(-(-0.6 + 0.55 * incp - 0.3 * vehrd)))
    cls = 0 if rng.random() < share1 else 1
    beta = truth[cls]
    x = np.column_stack([
        np.log(rng.uniform(0.5, 60.0, N_ALTS)),
        rng.uniform(0.0, 1200.0, N_ALTS),
        rng.normal(0.0, 1.0, N_ALTS),
    ])
    v = x @ beta
    p = np.exp(v - v.max())
    p /= p.sum()
    chosen = int(rng.choice(N_ALTS, p=p))
    for a in range(N_ALTS):
        rec = {"case_id": f"c{case}", "alt_id": a + 1,
               "chosen": int(a == chosen), "log_correction": 1.0,
               "incp": incp, "vehrd": vehrd}
        rec.update(dict(zip(FEATS, x[a])))
        rows.append(rec)

frame = pd.DataFrame(rows)


def report(tag, **kw):
    res = estimate_dest_choice_lc(
        frame, FEATS, n_classes=2, n_init=3, perturb_sigma=0.5,
        share_floor=0.10, membership_l2=0.1, maxiter=60, seed=3,
        require_crash=False, **kw)
    assert "mstep_grad_inf" in res, sorted(res)
    g = res["mstep_grad_inf"]
    tol = res["mstep_grad_tol"]
    stat = res["mstep_stationary"]
    print(f"[{tag}] LL={res['loglike']:.3f} iters={res['iterations']} "
          f"converged={res['converged']} "
          f"shares={[round(s, 3) for s in res['shares']]}")
    print(f"[{tag}] mstep |g| = {['%.2e' % v for v in g]}  tol={tol:.2e} "
          f"stationary={stat}")
    assert len(g) == 2 and len(stat) == 2, "one entry per class"
    assert tol is not None and tol > 0
    assert all(np.isfinite(v) for v in g), g
    assert stat == [v <= tol for v in g], "flags disagree with the gradients"
    assert len(res["betas"]) == len(g), "one gradient per class beta vector"
    return res


report("membership", membership_vars=MEMB)
# The fixed-share path re-orders classes by share (label switching); the
# gradient must be permuted with its class betas or the evidence lands on the
# wrong class.
fixed = report("fixed-share")
shares = fixed["shares"]
assert shares == sorted(shares, reverse=True), shares
print("OK: M-step optimality reported per class on both paths")
