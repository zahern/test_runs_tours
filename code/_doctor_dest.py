"""Run larch's doctor + overspecification on the real QLD destination model
and print which parameters/groups are near-dependent (eigenvector loadings)."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)
import numpy as np
import cloudpickle


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
import larch
print("larch", larch.__version__)
import stage5_destination_choice as S5
from larch.model import troubleshooting

cache = cloudpickle.load(open(
    r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
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
try:
    print("loglike at warm start:", float(mm.loglike(check_if_best=False, error_if_bad=False)))
except Exception as e:
    print("loglike failed:", e)

print("\n=== doctor (data checks + low-variance + overspecification) ===")
mm2, problems = troubleshooting.doctor(
    mm, check_low_variance_data_co="?", check_overspec="?", verbose=3)
print("problem checks with findings:", list(problems.keys()))
for name, diag in problems.items():
    print(f"\n--- {name}: {len(diag)} issue(s) ---")
    try:
        print(diag.head(30).to_string())
    except Exception as e:
        print("  (could not render)", e)

# Raw overspec directions with loadings, at full precision
try:
    _m, diag = troubleshooting.overspecification(mm, repair="?", verbose=3)
    if diag is not None:
        print("\n=== overspecification directions (eigenvalue, parameter loadings) ===")
        for idx in diag.index.get_level_values(0).unique():
            sub = diag.xs(idx, level=0)
            ev = sub.index.get_level_values(0)[0]
            print(f"direction {idx}: eigenvalue={ev:.3e}")
            load = sub.droplevel(0)
            load = load.reindex(load.abs().sort_values(ascending=False).index)
            for pname, val in load.head(10).items():
                print(f"    {str(pname):32s} {float(val):+.4f}")
except Exception as e:
    print("overspecification failed:", type(e).__name__, e)
