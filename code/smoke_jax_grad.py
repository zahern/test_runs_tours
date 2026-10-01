import os
import sys
import time

import numpy as np

# ── isolation guard ────────────────────────────────────────────────────────
# Never write into the production code/data tree: relocate CWD to a private
# scratch dir before anything else runs (see smoke_jax_models.py for the
# full rationale; the PBS wrapper redirects TMPDIR/CACHE/PYCACHE too).
import tempfile
os.chdir(tempfile.mkdtemp(prefix="smoke_jax_grad_scratch_"))


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


def _build_model(compute_engine):
    """Real nested-logit Larch model on deterministic synthetic data.

    Fully self-contained: no larch.datasets / example-file downloads, so it
    works on clusters where those extras are absent.  Sets
    compute_engine=... - elasticnet_objective branches on model.compute_engine,
    NOT model.engine.
    """
    import pandas as pd
    import xarray as xr
    import larch
    from larch import Model, P, X

    rng = np.random.default_rng(7)
    n_cases, n_alts = 800, 3
    case = np.arange(n_cases)
    alts = np.array([1, 2, 3])  # larch convention: root node is 0
    time_v = rng.gamma(2.0, 2.0, size=(n_cases, n_alts))
    price_v = rng.gamma(2.0, 1.5, size=(n_cases, n_alts))
    # choices from known coefficients + gumbel noise -> parameters identified,
    # SLSQP has a real gradient to follow and will move decisively off priors
    util_true = (-0.10 * time_v - 0.05 * price_v
                 + rng.gumbel(size=(n_cases, n_alts)))
    ch_v = np.zeros_like(time_v)
    ch_v[np.arange(n_cases), np.argmin(util_true, axis=1)] = 1.0

    ca = pd.DataFrame(
        {
            "time": time_v.ravel(),
            "price": price_v.ravel(),
            "ch": ch_v.ravel(),
        },
        index=pd.MultiIndex.from_product([case, alts],
                                         names=["caseid", "altid"]),
    )
    ds = larch.Dataset.construct.from_idca(ca)

    m = Model(ds, compute_engine=compute_engine)
    m.choice_ca_var = "ch"
    # nest all-but-one alternative so the NL rollup code path is exercised too;
    # alt codes read defensively from the initialized graph
    mu_name = None
    try:
        alts_in_graph = sorted(int(a) for a in m.graph.nodes if a != 0)
        if len(alts_in_graph) >= 2:
            nest_code = max(alts_in_graph) + 1
            m.graph.add_node(nest_code, children=tuple(alts_in_graph[1:]),
                             parent=0, parameter="Mu_Nest")
            mu_name = "Mu_Nest"
    except Exception:
        mu_name = None  # plain MNL still exercises both gradient paths
    m.utility_ca = P("B_TIME") * X("time") + P("B_PRICE") * X("price")
    vals = {"B_TIME": 0.0, "B_PRICE": 0.0}
    if mu_name:
        vals[mu_name] = 1.0
    # Model.set_values is deprecated; assign pvals by parameter order.
    m.pvals = np.array([vals.get(n, 0.0) for n in m.pnames], dtype=float)
    m.set_cap()
    return m


def _real_test():
    """End-to-end: patched gradient on compute_engine='jax' must be finite,
    non-zero, and match the numba engine's patched gradient to ~float64
    tolerance; SLSQP must actually move the parameters under maximize_loglike.
    Any REAL_MODEL_SKIP here is a test bug, not a pass."""
    try:
        import larch
    except Exception as e:
        print("REAL_MODEL_FAIL no larch:", repr(e))
        return False
    from larch_regularization import elasticnet_objective

    ok = True
    grads = {}
    starts = {}
    for eng in ("numba", "jax"):
        try:
            m = _build_model(eng)
        except Exception as e:
            print(f"REAL_MODEL_FAIL build({eng}):", repr(e))
            return False
        starts[eng] = np.asarray(m.pvals).copy()
        try:
            with elasticnet_objective(m, alpha=1e-3, l1_ratio=0.0):
                g = np.asarray(m.d_logloss(np.asarray(starts[eng])))
            grads[eng] = g
            gn = float(np.linalg.norm(g))
            finite = bool(np.all(np.isfinite(g)))
            print(f"[real:{eng}] |g|={gn:.6e} finite={finite} "
                  f"x64={bool(getattr(__import__('jax').config, 'jax_enable_x64', False))}")
            if not finite or gn == 0.0:
                print(f"REAL_MODEL_FAIL {eng}: dead/NaN gradient")
                ok = False
        except Exception as e:
            print(f"REAL_MODEL_FAIL {eng} d_logloss:", repr(e))
            return False

    if "numba" in grads and "jax" in grads:
        dn = float(np.max(np.abs(grads["numba"] - grads["jax"])))
        scale = max(float(np.max(np.abs(grads["numba"]))), 1e-12)
        rel = dn / scale
        print(f"[real] max|g_numba-g_jax|={dn:.3e} rel={rel:.3e}")
        # engines must agree now that both run mean-scale float64.  Tolerance
        # is loose enough for kernel differences (~1e-3) but catches scale
        # bugs (total-vs-mean = rel ~n_cases) and sign flips instantly.
        if not (rel < 5e-3 or dn < 1e-6):
            print("REAL_MODEL_FAIL: jax and numba gradients disagree "
                  "(scale or precision mismatch)")
            ok = False

    # movement check on the jax engine: SLSQP must leave iteration 0
    try:
        m = _build_model("jax")
        start = np.asarray(m.pvals).copy()
        with elasticnet_objective(m, alpha=1e-3, l1_ratio=0.0):
            m.maximize_loglike(quiet=True, options={"maxiter": 200})
        moved = float(np.max(np.abs(np.asarray(m.pvals) - start)))
        print(f"[real:jax] max|movement| after maximize_loglike = {moved:.6f}")
        if moved < 1e-8:
            print("REAL_MODEL_FAIL: parameters did not move on the jax engine "
                  "(zero-gradient stuck at priors)")
            ok = False
    except Exception as e:
        print("REAL_MODEL_FAIL jax maximize_loglike:", repr(e))
        ok = False

    print("REAL_MODEL_" + ("OK" if ok else "FAIL"))
    return ok


def main():
    t0 = time.time()
    ok = _mock_test()
    ok = _real_test() and ok
    print("JAX_GRAD_" + ("OK" if ok else "FAIL"), f"({time.time() - t0:.1f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
