"""Doctor-guided orthogonalisation probe on the real QLD destination frame.

1. Doctor/overspecification loadings (which parameter GROUPS are near-dependent).
2. For every declared orthogonalize_on pair: OLS beta and R^2 of the
   destination component on its control (the "percentage based on the other
   variable"), and whether _apply_orthogonalization_to_ca actually materialises
   the residual column (or why it skips).
"""
import json, os, sys, warnings, logging
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE); os.chdir(CODE)
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
import stage5_destination_choice as S5
from ga_constraints import get_constraints

cache = cloudpickle.load(open(
    r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
if cache.get("dt_dest") is None:
    from _s5_larch_perturb import _rebuild_dt
    _dt, _sk = _rebuild_dt(cache); _sk.close(); cache["dt_dest"] = _dt
frame, fmap = S5._adaptive_dest_choice_frame_full(
    cache["dt_dest"], cache["sampled_alts"], "work")

spec = dict(get_constraints("stage5_dest").get("orthogonalize_on", {}))
print(f"orthogonalize_on pairs declared: {len(spec)}")
UEC = S5.DEST_UEC_FLAT

print(f"\n{'coeff':24s} {'control':16s} {'y_col':22s} {'x_col':18s} "
      f"{'beta':>9s} {'R2':>7s}  materialised")
mat = {}
for coeff, base in spec.items():
    y_parsed = S5._ortho_plain_column(UEC.get(coeff, ("", None))[0]) if coeff in UEC else None
    x_parsed = S5._ortho_plain_column(UEC.get(base, ("", None))[0]) if base in UEC else None
    y_col = y_parsed[0] if y_parsed else "-"
    x_col = x_parsed[0] if x_parsed else "-"
    if y_parsed and x_parsed and y_col in frame.columns and x_col in frame.columns:
        y = np.asarray(frame[y_col], float)
        x = np.asarray(frame[x_col], float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 10 and np.var(x[m]) > 1e-12:
            b = float(np.cov(x[m], y[m])[0, 1] / np.var(x[m]))
            r2 = float(np.corrcoef(x[m], y[m])[0, 1] ** 2)
            rcol = S5._ortho_resid_col(y_col, x_col)
            mat[coeff] = (rcol, b)
            print(f"{coeff:24s} {base:16s} {y_col:22s} {x_col:18s} "
                  f"{b:9.4f} {r2:7.3f}  {rcol}")
        else:
            print(f"{coeff:24s} {base:16s} {y_col:22s} {x_col:18s} "
                  f"{'n/a':>9s} {'n/a':>7s}  no variance/mask")
    else:
        miss = []
        if coeff not in UEC: miss.append(f"{coeff} not in DEST_UEC_FLAT")
        if base not in UEC: miss.append(f"{base} not in DEST_UEC_FLAT")
        if y_parsed is None and coeff in UEC: miss.append("y expr not plain")
        if x_parsed is None and base in UEC: miss.append("x expr not plain")
        if y_col not in frame.columns and y_parsed: miss.append(f"{y_col} missing")
        if x_col not in frame.columns and x_parsed: miss.append(f"{x_col} missing")
        print(f"{coeff:24s} {base:16s} {y_col:22s} {x_col:18s} "
              f"{'SKIP':>9s} {'-':>7s}  {'; '.join(miss)}")

# Actually run the materialiser on a copy and count what it creates
before = set(frame.columns)
mat2 = S5._apply_orthogonalization_to_ca(frame.copy(), UEC, spec)
print(f"\n_apply_orthogonalization_to_ca materialised {len(mat2)} residual(s):")
for coeff, (rcol, beta) in sorted(mat2.items()):
    print(f"  {coeff:24s} -> {rcol:40s} beta={beta:+.4f}")
