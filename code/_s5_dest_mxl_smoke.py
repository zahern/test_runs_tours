"""Destination mixed-logit smoke test on the real cached QLD work frame.

Uses the same larch native-mixtures API the pipeline already uses for the
MODE model (_estimate_mode_model_mixed_logit, stage5_destination_choice.py):
  lx.mixtures.Normal(param, sd_name); mxl.mixtures.append(mix);
  mxl.n_draws = ...; mxl.seed = ...
then loglike + a short SLSQP fit, so we know the destination MXL path can be
estimated end-to-end.
"""
import os, sys, time, warnings
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
import larch as lx
import stage5_destination_choice as S5

cache = cloudpickle.load(open(r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
if cache.get("dt_dest") is None:
    from _s5_larch_perturb import _rebuild_dt
    _dt, _sk = _rebuild_dt(cache); _sk.close(); cache["dt_dest"] = _dt
dt = cache["dt_dest"]

mm = S5._build_dest_model(dt, "work")
print("pf params:", len(mm.pf))

RANDOM = ["c_log_dist", "c_crash_safety_dest"]
for param in RANDOM:
    if param not in mm.pf.index:
        print("skip (not in pf):", param)
        continue
    sd = f"s_{param}"
    mm.mixtures.append(lx.mixtures.Normal(param, sd))
    S5._pf_set(mm, sd, "value", 0.25)
    S5._pf_set(mm, sd, "minimum", 0.01)
    S5._pf_set(mm, sd, "maximum", 5.0)
    print("added Normal mixture:", param, "->", sd)

mm.n_draws = int(os.environ.get("MXL_DRAWS", "8"))
mm.seed = 42
print("n_draws:", mm.n_draws)

t0 = time.time()
try:
    ll0 = float(mm.loglike(check_if_best=False, error_if_bad=False))
    print(f"initial MXL loglike = {ll0:.3f} ({time.time()-t0:.0f}s)")
except Exception as exc:
    print(f"loglike FAILED: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

# Same guard as _estimate_mode_model_mixed_logit: SLSQP probes non-finite
# utility regions (draws x parameters); return a penalty instead of inf.
_orig_logloss = mm.logloss
_orig_d_logloss = mm.d_logloss
_start_pvals = np.asarray(mm.pvals, dtype=float).copy()
_invalid_loss = 1e6


def _safe_ll(x=None, **kw):
    _x = np.asarray(x if x is not None else mm.pvals, dtype=float)
    try:
        _v = float(_orig_logloss(x, **kw))
    except Exception:
        _v = float("inf")
    if np.isfinite(_v):
        return _v
    _d = _x - _start_pvals
    return _invalid_loss + 0.01 * _invalid_loss * float(np.dot(_d, _d))


def _safe_dll(x=None, **kw):
    _x = np.asarray(x if x is not None else mm.pvals, dtype=float)
    try:
        _g = np.asarray(_orig_d_logloss(x, **kw), dtype=float)
    except Exception:
        _g = np.full_like(_x, np.nan)
    if np.all(np.isfinite(_g)):
        return _g
    return 0.02 * _invalid_loss * (_x - _start_pvals)


mm.logloss = _safe_ll
mm.d_logloss = _safe_dll
t0 = time.time()
try:
    res = mm.maximize_loglike(
        method="slsqp", options={"maxiter": 10, "ftol": 1e-6}, quiet=True)
    print(f"short SLSQP fit done in {time.time()-t0:.0f}s "
          f"success={res.get('success')} nit={res.get('nit')} "
          f"LL={-res.get('logloss', float('nan')) * mm.total_weight():.3f}")
finally:
    mm.logloss = _orig_logloss
    mm.d_logloss = _orig_d_logloss

pf = mm.pf
for name in ("c_log_dist", "s_c_log_dist", "c_crash_safety_dest",
             "s_c_crash_safety_dest"):
    if name in pf.index:
        print(f"  {name:24s} value={float(pf.loc[name,'value']):+.4f} "
              f"holdfast={pf.loc[name].get('holdfast')}")
