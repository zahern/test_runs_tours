"""Was the Stage-4 RRM 'optimizer_stopped' label a real failure or a mislabel?

Loads the eight timing models that a real pipeline run saved
(``stage4_models.pkl``), and for each one:

  * evaluates the analytic gradient of the log-likelihood at the *stored*
    coefficients -- that is where the fit actually stopped;
  * compares it with the per-observation tolerance used by the patched
    ``assess_convergence``;
  * refits from the stored coefficients with the patched ``fit`` to see what
    the polish retry does to it.

Run:  python code/_check_rrm_stage4.py runs/<run dir>/stage4_models.pkl
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, '.')

from SearchLibrium import rrm as R          # noqa: E402


def gradient_at(model, beta):
    """Analytic gradient of the (negative) log-like at ``beta``."""
    keep = model.loglik
    try:
        return np.asarray(
            model.get_loglike_gradient(np.asarray(beta, dtype=float),
                                       np.ones(len(beta)) * 1e-10)[1],
            dtype=float)
    finally:
        model.loglik = keep


def main(path):
    with open(path, 'rb') as fh:
        models = pickle.load(fh)

    print(f"{'purpose':10s} {'n':>6s} {'LL stored':>12s} {'|g|':>10s} "
          f"{'tol':>10s} {'old':>5s} | {'LL refit':>12s} {'dLL':>9s} "
          f"{'|g| refit':>10s} {'new':>5s}")
    rows = []
    for purpose, wrapper in models.items():
        rrm = getattr(wrapper, 'rrm_', None)
        if rrm is None:
            print(f'{purpose:10s}  (no rrm_)')
            continue
        n = int(len(rrm.y))
        ll0 = float(rrm.loglik)
        g0 = gradient_at(rrm, rrm.coeff_est)
        g0_inf = float(np.max(np.abs(g0))) if g0.size else float('nan')
        tol = R.grad_tolerance(n)

        r2 = pickle.loads(pickle.dumps(rrm))     # independent copy
        r2.fit(start=np.asarray(rrm.coeff_est, dtype=float),
               compute_inference=False)
        ll1 = float(r2.loglik)
        g1_inf = float(getattr(r2, 'grad_norm_inf', float('nan')))

        print(f'{purpose:10s} {n:6d} {ll0:12.3f} {g0_inf:10.2e} {tol:10.2e} '
              f'{str(bool(rrm.converged)):>5s} | {ll1:12.3f} {ll1 - ll0:+9.3f} '
              f'{g1_inf:10.2e} {str(bool(r2.converged)):>5s}')
        rows.append((purpose, n, ll0, g0_inf, tol, bool(rrm.converged),
                     ll1, bool(r2.converged)))

    if not rows:
        print('no RRM models found')
        return

    n_total = len(rows)
    at_opt = sum(1 for _p, _n, _l, g, t, _c, _l1, _c1 in rows if g <= t)
    old_ok = sum(1 for _p, _n, _l, _g, _t, c, _l1, _c1 in rows if c)
    new_ok = sum(1 for _p, _n, _l, _g, _t, _c, _l1, c1 in rows if c1)
    gain = sum(l1 - l0 for _p, _n, l0, _g, _t, _c, l1, _c1 in rows)

    print()
    print(f'  already at a stationary point (per-obs tolerance): {at_opt}/{n_total}')
    print(f'  labelled converged  old: {old_ok}/{n_total}   new: {new_ok}/{n_total}')
    print(f'  total LL change from the polish retry: {gain:+.4f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else 'runs/agds_metacount_mv_staged__paper_predict_20261029/stage4_models.pkl')
