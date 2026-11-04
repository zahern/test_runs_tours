"""Validate the pipeline env path: STAGE5_DEST_INTERACTIONS through
_estimate_adaptive_dest_choice (frame covariates -> interaction estimator)."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["STAGE5_DEST_FULL_SPEC"] = "1"
os.environ["STAGE5_DEST_MH_DRAWS"] = "1500"
os.environ["STAGE5_DEST_MH_BURN_IN"] = "400"
os.environ["STAGE5_DEST_INTERACTIONS"] = json.dumps({
    "case_covariates": ["STUDYING", "occp_trades", "age5p"],
    "actor_defs": {"student": "STUDYING > 0", "senior": "age5p >= 9",
                   "trades": "occp_trades > 0"},
    "interactions": {"log_DIST_1": ["student"],
                     "log_CRASH_1": ["trades"]},
})
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
import stage5_destination_choice as S5

cache = cloudpickle.load(open(r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
if cache.get("dt_dest") is None:
    from _s5_larch_perturb import _rebuild_dt
    _dt, _sk = _rebuild_dt(cache); _sk.close(); cache["dt_dest"] = _dt
base = json.load(open(
    r"runs\_s5_larch_perturb\work_perturb_results_fixed.json"))["runs"][0]["coefs"]

t0 = time.time()
coefs, meta = S5._estimate_adaptive_dest_choice(
    seg_tours=cache["seg_acdc"],
    sampled_alts=cache["sampled_alts"],
    landuse=cache["lu_with_st"],
    logsums=cache["logsums"],
    skim_df=None,
    method="mh",
    seed=42,
    dt_dest=cache["dt_dest"],
    segment="work",
    full_spec=True,
    initial_coefs=base,
)
print(f"\nwrapper finished in {time.time()-t0:.0f}s")
print("n_features:", meta["n_features"])
print("dropped:", meta.get("dropped_features"))
print("interactions:", meta.get("interactions"))
print("interaction_inference:")
for k, v in (meta.get("interaction_inference") or {}).items():
    print(f"  {k}: value={v.get('value')}, se={v.get('std_err')}, "
          f"t={v.get('tstat')}, ({v.get('actor')} x {v.get('component')})")
print("ix coefficients in dest_coef:",
      [k for k in coefs if k.startswith("ix_")])
