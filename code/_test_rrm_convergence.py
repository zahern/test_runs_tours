"""Does the RRM convergence fix actually change the fit?

Reproduces the Stage-4 timing problem: 5 periods (alternatives), the same six
period features ``stage4_schedule_placement._build_X`` builds, 6k observations.
Choices are sampled from a known RRM so the true coefficients are available.

Then it fits the model twice --
  old  : convergence = scipy result.success, no polish   (pre-fix behaviour)
  new  : gradient-based verdict + re-polish              (this fix)
and compares verdict, log-likelihood, gradient norm, iterations, coefficient
recovery and out-of-sample log-loss.

Run:  python code/_test_rrm_convergence.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'code'))

import SearchLibrium.rrm as RRM
from SearchLibrium import RandomRegret

PERIODS = [(420, 540), (540, 660), (660, 840), (840, 1020), (1020, 1260)]
DAY_START, DAY_SPAN = 300.0, 1140.0


def build_X(n, durations, seed=0):
    """Same six features as stage4 _build_X, shape (n, J, M)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(PERIODS), 6), dtype=float)
    for pidx, (p_start, p_end) in enumerate(PERIODS):
        center_hr = (p_start + p_end) / 120.0
        width = max(float(p_end - p_start), 1.0)
        X[:, pidx, :] = np.column_stack([
            np.full(n, center_hr),
            np.full(n, center_hr ** 2),
            np.full(n, float(pidx in (1, 3))),
            np.full(n, (p_start - DAY_START) / DAY_SPAN),
            (p_end - durations - DAY_START) / DAY_SPAN,
            durations / width,
        ])
    return X


def make_model(X, y):
    m = RandomRegret()
    m.X = X
    m.y = y
    m.nb_samples = len(X)
    m.nb_alt = X.shape[1]
    m.nb_attr = X.shape[2]
    m.attrs = [f'f{i}' for i in range(X.shape[2])]
    m.avail = np.ones((len(X), X.shape[1]), dtype=float)
    m.initialise()
    m.normalize = False
    return m


def fit(model, mode, start=None):
    """mode='old' reproduces the pre-fix verdict; 'new' is the patched path."""
    n_obs = len(model.y)
    keep_polish, keep_assess = RRM.polish_result, RRM.assess_convergence
    keep_min = RRM.minimize
    seen = []

    def spy(*a, **kw):
        res = keep_min(*a, **kw)
        seen.append(res)
        return res

    RRM.minimize = spy
    if mode == 'old':
        RRM.polish_result = lambda fun, result, **kw: result
        RRM.assess_convergence = lambda result, gtol=None, n_obs=None, grad=None: (
            bool(result.success), RRM._grad_inf_norm(result),
            str(getattr(result, 'message', '')))
    try:
        model.fit(start=start, compute_inference=False)
    finally:
        RRM.polish_result, RRM.assess_convergence = keep_polish, keep_assess
        RRM.minimize = keep_min
    return seen[-1] if seen else None


def describe(tag, model, res, X=None, y=None):
    grad = RRM._grad_inf_norm(res) if res is not None else float('nan')
    ll = float(model.loglik)
    out = {
        'tag': tag, 'converged': bool(model.converged), 'll': ll,
        'grad': grad, 'n_iter': int(getattr(res, 'nit', 0)),
        'message': str(getattr(model, 'fit_message', '')),
        'beta': np.asarray(model.coeff_est, dtype=float),
    }
    if X is not None:
        p = np.clip(model.predict_proba(X), 1e-12, 1.0)
        out['oos_ll'] = float(np.sum(np.log(p[np.arange(len(y)), y])))
    oos = f"oos LL={out['oos_ll']:11.3f}" if 'oos_ll' in out else ''
    print(f"  {tag:5s} converged={str(out['converged']):5s} "
          f"LL={ll:12.3f} |grad|inf={grad:9.2e} iters={out['n_iter']:3d} {oos}")
    print(f"        scipy: {out['message'][:80]}")
    return out


def stall_scenario(X, y, beta_true):
    """A fit that stops early, far from the optimum (`maxiter=8`).

    The pre-fix code accepted whatever the truncated BFGS call left behind and
    only flagged it converged=False; the polish retry never ran, so those
    coefficients were reported as the answer. That is the failure mode behind
    the eight Stage-4 'optimizer_stopped' timing models.
    """
    out = {}
    for mode in ('old', 'new'):
        m = make_model(X, y)
        m.maxiter = 8          # truncate the primary fit: an early stop
        res = fit(m, mode)
        out[mode] = describe(f'stall-{mode}', m, res)
    o, nw = out['old'], out['new']
    if o['grad'] > RRM.grad_tolerance(len(y)):
        assert nw['ll'] > o['ll'], 'polish did not improve a stalled fit'
        assert nw['grad'] < o['grad'], 'polish did not reduce the gradient'
        print(f"  polish recovered {nw['ll'] - o['ll']:+.4f} LL, "
              f"gradient {o['grad']:.2e} -> {nw['grad']:.2e}")
    else:
        print('  (start did not stall; recovery path not exercised)')
    return out


def main():
    n, hold = 6000, 2000
    rng = np.random.default_rng(7)
    durations = rng.uniform(10.0, 120.0, size=n + hold)
    X_all = build_X(n + hold, durations, seed=3)
    X, X_ho = X_all[:n], X_all[n:]
    beta_true = np.array([-0.02, 0.0008, 0.35, 0.90, -0.70, 0.45])

    gen = make_model(X, np.zeros(n, dtype=int))
    p_true = gen.compute_probability(beta_true)
    y = np.array([rng.choice(len(PERIODS), p=row) for row in p_true])

    gen_ho = make_model(X_ho, np.zeros(hold, dtype=int))
    p_ho = gen_ho.compute_probability(beta_true)
    y_ho = np.array([rng.choice(len(PERIODS), p=row) for row in p_ho])

    print(f'true betas: {np.round(beta_true, 4)}')
    print(f'chosen-period shares: {np.round(np.bincount(y, minlength=5) / n, 3)}\n')

    results = {}
    for mode in ('old', 'new'):
        m = make_model(X, y)
        res = fit(m, mode)
        results[mode] = describe(mode, m, res, X_ho, y_ho)

    print('\nbeta comparison')
    hdr = f"  {'true':>10s} {'old':>10s} {'new':>10s}"
    print(hdr)
    for i, b in enumerate(beta_true):
        print(f"  {b:10.4f} {results['old']['beta'][i]:10.4f} "
              f"{results['new']['beta'][i]:10.4f}")

    o, nw = results['old'], results['new']
    print('\nverdict')
    print(f"  old converged={o['converged']}  new converged={nw['converged']}")
    tol = RRM.grad_tolerance(n)
    print(f"  new graderror={nw['grad']:.2e} (tol {tol:.2e} = "
          f"{RRM.RRM_GRAD_TOL:.0e} x {n} obs)")
    improved = nw['ll'] >= o['ll'] - 1e-6
    print(f"  new LL >= old LL: {improved}  (+{nw['ll'] - o['ll']:.4f})")

    assert np.isfinite(nw['ll']), 'new fit produced a non-finite loglike'
    assert improved, 'polish reported a worse fit than the one it started from'
    assert nw['grad'] <= tol, 'new fit is still not at a stationary point'

    print('\nstalled-start scenario')
    stall_scenario(X, y, beta_true)
    print('\nRRM CONVERGENCE CHECKS PASSED')


if __name__ == '__main__':
    main()
