"""Destination mixed-logit estimation, warm-started from the fitted MNL.

This is the closest analogue of the mode MXL helper
(_estimate_mode_model_mixed_logit): fitted means as start, Normal mixtures on
c_log_dist / c_crash_safety_dest, sd bounds materialised AFTER
mangle/unmangle, and the safe-objective wrapper for non-finite draw regions.
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np, cloudpickle


def patch():
    from larch.dataset import flow as f
    def g(self, n):
        if n == "_obj": raise AttributeError(n)
        o = self.__dict__.get("_obj")
        if o is None: raise AttributeError(n)
        r = getattr(o, n)
        try: r = self.transfer_dimension_attrs(r)
        except Exception: pass
        return r
    for c in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
        k = getattr(f, c, None)
        if k is not None: k.__getattr__ = g


patch()
import larch as lx
import stage5_destination_choice as S5

cache = cloudpickle.load(open(r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
if cache.get("dt_dest") is None:
    from _s5_larch_perturb import _rebuild_dt
    _dt, _sk = _rebuild_dt(cache); _sk.close(); cache["dt_dest"] = _dt
base = json.load(open(
    r"runs\_s5_larch_perturb\work_perturb_results_fixed.json"))["runs"][0]["coefs"]

mm = S5._build_dest_model(cache["dt_dest"], "work")
for n, v in base.items():
    if n in mm.pf.index and np.isfinite(float(v)):
        try: mm.set_value(n, float(v))
        except Exception: pass

for param, family in (("c_log_dist", "normal"),
                      ("c_crash_safety_dest", "normal")):
    if param not in mm.pf.index:
        continue
    sd = f"s_{param}"
    mm.mixtures.append(lx.mixtures.Normal(param, sd)
                       if family == "normal"
                       else lx.mixtures.NegLogNormal(param, sd))
    try:
        mm.mangle(); mm.unmangle()
    except Exception as e:
        print("mangle failed:", e)
    S5._pf_set(mm, sd, "value", 0.25)
    S5._pf_set(mm, sd, "minimum", 0.05)
    S5._pf_set(mm, sd, "maximum", 3.0)
mm.n_draws = int(os.environ.get("MXL_DRAWS", "32"))
mm.seed = 42
print("mixtures:", [type(x).__name__ for x in mm.mixtures],
      "| n_draws:", mm.n_draws)
t0 = time.time()
ll0 = float(mm.loglike(check_if_best=False, error_if_bad=False))
print(f"warm-start mixture LL = {ll0:.3f} ({time.time()-t0:.0f}s)")

# validate the gradient-gate fix path (jax-native gradient)
try:
    g = np.asarray(mm.jax_d_loglike(np.asarray(mm.pvals, dtype=float)), dtype=float)
    print("jax_d_loglike norm at start:", float(np.linalg.norm(g)))
except Exception as e:
    print("jax_d_loglike failed:", type(e).__name__, e)

_orig_ll, _orig_d = mm.logloss, mm.d_logloss
_start = np.asarray(mm.pvals, dtype=float).copy()
PEN = 1e6


def safe_ll(x=None, **kw):
    _x = np.asarray(x if x is not None else mm.pvals, dtype=float)
    try:
        v = float(_orig_ll(x, **kw))
    except Exception:
        v = float("inf")
    if np.isfinite(v):
        return v
    d = _x - _start
    return PEN + 0.01 * PEN * float(np.dot(d, d))


def safe_d(x=None, **kw):
    _x = np.asarray(x if x is not None else mm.pvals, dtype=float)
    try:
        gg = np.asarray(_orig_d(x, **kw), dtype=float)
    except Exception:
        gg = np.full_like(_x, np.nan)
    return gg if np.all(np.isfinite(gg)) else 0.02 * PEN * (_x - _start)


mm.logloss, mm.d_logloss = safe_ll, safe_d
t0 = time.time()
try:
    res = mm.maximize_loglike(method="slsqp", stderr=False,
                              options={"maxiter": 25, "ftol": 1e-7}, quiet=True)
    print(f"fit: success={res.get('success')} nit={res.get('nit')} "
          f"logloss={res.get('logloss')} ({time.time()-t0:.0f}s)")
finally:
    mm.logloss, mm.d_logloss = _orig_ll, _orig_d

try:
    ll1 = float(mm.loglike(check_if_best=False, error_if_bad=False))
except Exception as e:
    ll1 = float("nan")
print(f"true LL at final = {ll1:.3f} (start {ll0:.3f}, "
      f"delta {ll1-ll0:+.1f})")
pf = mm.pf
for n in ("c_log_dist", "s_c_log_dist", "c_crash_safety_dest", "s_c_crash_safety_dest"):
    if n in pf.index:
        print(f"  {n:24s} = {float(pf.loc[n,'value']):+.4f}  "
              f"[{float(pf.loc[n,'minimum'])}, {float(pf.loc[n,'maximum'])}]")
