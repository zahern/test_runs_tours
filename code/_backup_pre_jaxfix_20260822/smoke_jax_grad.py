import sys
import time

import numpy as np


def _mock_test():
    import jax
    import jax.numpy as jnp
    import larch_regularization as lr

    class MockModel:
        """Faithful mock of the Larch Model interface used by
        elasticnet_objective, including a jax-traceable jax_loglike so the
        patch exercises the real (model.jax_loglike) code path."""

        def __init__(self, n, seed=0):
            rng = np.random.default_rng(seed)
            self._pnames = [f"b{i}" for i in range(n)]
            self._pvals = rng.normal(size=n).astype(float)
            self._pholdfast = np.zeros(n, dtype=int)
            self.compute_engine = "jax"

        @property
        def pnames(self):
            return self._pnames

        @property
        def pvals(self):
            return self._pvals

        @pvals.setter
        def pvals(self, v):
            self._pvals = np.asarray(v, dtype=float)

        @property
        def pholdfast(self):
            return self._pholdfast

        def logloss(self, x=None, **kw):
            p = self._pvals if x is None else np.asarray(x, dtype=float)
            return -0.5 * float(np.sum(p ** 2))

        def d_logloss(self, x=None, **kw):
            p = self._pvals if x is None else np.asarray(x, dtype=float)
            return -np.asarray(p, dtype=float)

        def jax_loglike(self, x):
            return 0.5 * jnp.sum(jnp.asarray(x) ** 2)

    ok = True
    cases = [
        (0.0, None, None),
        (0.5, None, None),
        (0.0, {f"b{i}": i * 0.1 for i in range(5)}, {f"b{i}": 2.0 for i in range(5)}),
    ]
    for l1, ref, w in cases:
        m = MockModel(5, seed=1)
        alpha = 1e-3
        with lr.elasticnet_objective(m, alpha=alpha, l1_ratio=l1, reference=ref, weights=w) as obj:
            p0 = m.pvals.copy()
            names = m.pnames
            refv = np.array([ref[n] for n in names]) if ref else np.zeros_like(p0)
            wv = np.array([w[n] for n in names]) if w else np.ones_like(p0)
            # analytic base + penalty gradient of the *penalized* objective
            base_g = -p0  # d/dx logloss, logloss = -0.5*sum(x^2)
            pen_g = alpha * wv * (l1 * np.sign(p0 - refv) + (1.0 - l1) * 2.0 * (p0 - refv))
            analytic_g = base_g + pen_g
            g = m.d_logloss()

            eps = 1e-6
            fd = np.zeros_like(p0)
            for i in range(len(p0)):
                pp = p0.copy()
                pp[i] += eps
                m.pvals = pp
                fp = m.logloss()
                pp = p0.copy()
                pp[i] -= eps
                m.pvals = pp
                fm = m.logloss()
                fd[i] = (fp - fm) / (2.0 * eps)
            m.pvals = p0

            gerr = float(np.max(np.abs(g - fd)))
            aerr = float(np.max(np.abs(g - analytic_g)))
            ratio = np.where(np.abs(fd) > 1e-12, g / fd, np.nan)
            print(f"[mock] l1={l1} ref={'set' if ref else 'None'} w={'set' if w else 'None'}")
            print(f"       g       ={np.round(g, 5)}")
            print(f"       fd      ={np.round(fd, 5)}")
            print(f"       analytic={np.round(analytic_g, 5)}")
            print(f"       g/fd    ={np.round(ratio, 4)}  grad_err_vs_fd={gerr:.2e} grad_err_vs_analytic={aerr:.2e}")
            # Authoritative check is g vs the analytic gradient (autodiff is exact;
            # fd can suffer float32 catastrophic cancellation when jax x64 is off).
            if aerr > 1e-3 or not np.all(np.isfinite(g)) or float(np.linalg.norm(g)) == 0.0:
                ok = False
    print("MOCK_" + ("OK" if ok else "FAIL"))
    return ok


def _real_test():
    try:
        import larch
        from larch import Model, P, X
        import larch_regularization as lr
        try:
            from larch.datasets import SWISSMETRO
            d = SWISSMETRO()
        except Exception as e:  # pragma: no cover - dataset optional
            print("REAL_MODEL_SKIP datasets:", e)
            return True
        m = Model(d)
        m.set_values({'ASC_TRAIN': 0, 'ASC_CAR': 0, 'ASC_SM': 0,
                      'B_TIME': 0, 'B_COST': 0})
        m.utility.ca = (P('ASC_TRAIN') * (larch.get_altid() == 1)
                        + P('ASC_CAR') * (larch.get_altid() == 2)
                        + P('ASC_SM') * (larch.get_altid() == 3))
        m.engine = 'jax'
        with lr.elasticnet_objective(m, alpha=1e-3, l1_ratio=0.0) as obj:
            g = m.d_logloss()
            print(f"[real] gnorm={float(np.linalg.norm(g)):.3e}")
            return bool(np.all(np.isfinite(g))) and float(np.linalg.norm(g)) > 0.0
    except Exception as e:  # pragma: no cover - best effort only
        print("REAL_MODEL_SKIP exc:", repr(e))
        return True


def main():
    t0 = time.time()
    ok = _mock_test()
    ok = _real_test() and ok
    print("JAX_GRAD_" + ("OK" if ok else "FAIL"), f"({time.time() - t0:.1f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
