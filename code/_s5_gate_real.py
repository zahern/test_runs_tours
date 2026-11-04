"""Gradient-norm gate on the REAL QLD work destination fit + remediation tests.

Real frame: crash-fixed cache (skims_371, real CRASH, 138,180 rows, 39-46
utility terms).  The fit is ill-conditioned (scaled information cond ~1.8e7)
and the default SLSQP stops at nit~10 with gradient ~6e7.

Each arm refits larch from the same warm start and reports
_fit_diagnostics['grad_norm'] / ['stationary'] so we can see whether
  (a) solver knobs (trust-constr / l-bfgs-b / more maxiter), or
  (b) dropping features on the near-null directions (identify_weak_features,
      at tol 1e-6 since the scaled smallest eigenvalue is ~5e-8)
produce a true stationary point.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np, cloudpickle


def patch_larch_flow():
    from larch.dataset import flow as _flow
    def _safe(self, name):
        if name == "_obj": raise AttributeError(name)
        o = self.__dict__.get("_obj")
        if o is None: raise AttributeError(name)
        r = getattr(o, name)
        try: r = self.transfer_dimension_attrs(r)
        except Exception: pass
        return r
    for c in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
        k = getattr(_flow, c, None)
        if k is not None: k.__getattr__ = _safe


patch_larch_flow()
import stage5_destination_choice as S5
from SearchLibrium.mh_choice_estimator import identify_weak_features

cache = cloudpickle.load(open(r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
if cache.get("dt_dest") is None:
    from _s5_larch_perturb import _rebuild_dt
    _dt, _sk = _rebuild_dt(cache); _sk.close(); cache["dt_dest"] = _dt
dt = cache["dt_dest"]
base = json.load(open(
    r"runs\_s5_larch_perturb\work_perturb_results_fixed.json"))["runs"][0]["coefs"]


def refit(tag, method=None, maxiter=None, cap=None, drop_coefs=None):
    old = {k: os.environ.get(k) for k in
           ("STAGE5_LARCH_METHOD", "STAGE5_LARCH_MAXITER", "STAGE5_LARCH_CAP")}
    if method is not None:
        os.environ["STAGE5_LARCH_METHOD"] = method
    if maxiter is not None:
        os.environ["STAGE5_LARCH_MAXITER"] = str(maxiter)
    if cap is not None:
        os.environ["STAGE5_LARCH_CAP"] = str(cap)
    try:
        _custom = None
        if drop_coefs:
            _custom = {k: v for k, v in S5.DEST_UEC_FLAT.items()
                       if (v[1] is None or "work" in v[1])
                       and k not in set(drop_coefs)}
        t0 = time.time()
        mm = S5._build_dest_model(dt, "work", custom_dest_vars=_custom)
        for n, v in base.items():
            if n in mm.pf.index and np.isfinite(float(v)):
                try:
                    mm.set_value(n, float(v))
                except Exception:
                    pass
        mm = S5._estimate(mm, two_phase=True)
        d = dict(mm._fit_diagnostics or {})
        print(f"{tag:28s} LL={d.get('loglike'):12.1f} "
              f"grad={d.get('grad_norm'):11.4g} "
              f"stationary={str(d.get('stationary')):5s} "
              f"converged={str(d.get('converged')):5s} "
              f"nit={d.get('nit')} success={d.get('success')} "
              f"({time.time()-t0:.0f}s, n_free={d.get('n_free')})", flush=True)
        return d
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


print("=== gate on real QLD work destination fit ===")
d0 = refit("slsqp default", method=None)

# (a) solver knobs
refit("trust-constr maxiter=5000", method="trust-constr", maxiter=5000)
refit("l-bfgs-b maxiter=5000", method="l-bfgs-b", maxiter=5000)
refit("slsqp maxiter=8000", method="slsqp", maxiter=8000)

# (b) identification-based reduction on the real frame
frame, fmap = S5._adaptive_dest_choice_frame_full(
    dt, cache["sampled_alts"], "work")
rep = identify_weak_features(frame, list(fmap), beta=base, tol=1e-6,
                             loading_threshold=0.5)
print(f"\nidentification(tol=1e-6): cond={rep.condition_number:.3g} "
      f"weak_dirs={len(rep.weak_directions)} drops={rep.suggested_drop}")
drop_coefs = [fmap[c] for c in rep.suggested_drop if c in fmap]
print("drop coefficients:", drop_coefs)
if drop_coefs:
    refit(f"reduced ({len(drop_coefs)} dropped)", method=None, drop_coefs=drop_coefs)
