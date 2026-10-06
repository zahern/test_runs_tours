"""Reference: larch LatentClass on the real frame with compat fixes.

- larch_lc_compat.apply(): n_cases metadata patch (class dataset)
- from_idca(fill_missing=0): kills NaN CA cells -> clean gradients
- membership covariates merged from seg_acdc (the flat frame's copies are
  zero-filled for this cache)
- jittered restarts (perturbation) to break the symmetric pooled optimum
"""
import os, sys, time, warnings, traceback
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("NEW_POP", "1")
os.environ.setdefault("NEW_POP_DIR", r"Z:\test_runs_tours\new_pop")
os.environ.setdefault("NEW_POP_MAPPED", "0")
os.environ.setdefault("PIPELINE_SKIMS_PATH",
                      r"Z:\test_runs_tours\demand_\skims_371.omx")
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)
import numpy as np
import pandas as pd
import cloudpickle


def patch_flow():
    from larch.dataset import flow as f

    def g(self, n):
        if n == "_obj":
            raise AttributeError(n)
        o = self.__dict__.get("_obj")
        if o is None:
            raise AttributeError(n)
        r = getattr(o, n)
        try:
            r = self.transfer_dimension_attrs(r)
        except Exception:
            pass
        return r
    for c in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
        k = getattr(f, c, None)
        if k is not None:
            k.__getattr__ = g


patch_flow()
import larch_lc_compat
print("compat:", larch_lc_compat.apply(), flush=True)
import larch as lx
from larch import P, X
import stage5_destination_choice as S5

N_CASES = int(os.environ.get("LC_N_CASES", "300"))
N_PERT = int(os.environ.get("LC_N_PERT", "2"))
SEED = int(os.environ.get("LC_SEED", "42"))
COLS = ["log_DIST_1", "log_CRASH_1", "logsum", "visit_work", "safety_x_crash"]
MEMB = ["STUDYING", "incp", "vehrd", "sexp"]
CRASH0 = "log_CRASH_1"

cache = cloudpickle.load(open(
    r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
seg = cache["seg_acdc"]
sa = cache["sampled_alts"]
if N_CASES and N_CASES < seg["ptour_id"].nunique():
    keep = set(seg[["ptour_id"]].drop_duplicates().sample(
        n=N_CASES, random_state=SEED)["ptour_id"].astype(str))
    seg = seg[seg["ptour_id"].astype(str).isin(keep)].copy()
    sa = sa[sa["ptour_id"].astype(str).isin(keep)].copy()
import openmatrix as omx
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = S5._skim_flat(skims)
S5._skim_lookup(skim_df)
dt = S5._dest_tree(seg, cache["persons"], cache["lu_with_st"], skims,
                   cache["logsums"], sampled_alts=sa, skim_df=skim_df,
                   crash_skims=skims, segment="work")
skims.close()
frame, fmap = S5._adaptive_dest_choice_frame_full(
    dt, sa, "work", case_covariates=MEMB)
# Membership covariates from seg_acdc (the flat frame's copies are zero).
_segcov = (seg[["ptour_id"] + MEMB].drop_duplicates("ptour_id").copy())
_segcov["case_id"] = _segcov["ptour_id"].astype(str)
frame = frame.merge(_segcov[["case_id"] + MEMB], on="case_id",
                    how="left", suffixes=("", "_seg"))
for _m in MEMB:
    if f"{_m}_seg" in frame.columns:
        frame[_m] = frame[f"{_m}_seg"].fillna(frame[_m])
        frame = frame.drop(columns=[f"{_m}_seg"])
print("memb variance:", {m: int(frame[m].nunique()) for m in MEMB}, flush=True)

ldf = frame[["case_id", "alt_id", "chosen", "log_correction"]
            + COLS + MEMB].copy()
ldf["case_id"] = ldf["case_id"].astype(str)
ldf["alt_id"] = pd.to_numeric(ldf["alt_id"], errors="coerce").astype(int)
_dtaz = ldf[ldf["chosen"] == 1].set_index("case_id")["alt_id"]
ldf["DTAZ"] = ldf["case_id"].map(_dtaz).astype(int)
ldf = ldf.drop(columns=["chosen"]).set_index(["case_id", "alt_id"])
dtc = lx.Dataset.construct.from_idca(ldf, fill_missing=0.0)
_co_ids = [str(c) for c in np.asarray(dtc.coords["case_id"].values).ravel()]
_case_cov = frame.drop_duplicates("case_id").set_index("case_id")
for cov in MEMB:
    dtc = dtc.assign(**{f"m_{cov}": (
        "case_id", np.asarray(_case_cov[cov].reindex(_co_ids).values, float))})
dtc = dtc.assign(group_id=("case_id",
                           np.arange(len(_co_ids)).astype(np.int64)))


def _cm(c):
    m = lx.Model()
    expr = None
    for col in COLS:
        t = P(f"c{c}_{col}") * X(col)
        expr = t if expr is None else expr + t
    m.utility_ca = expr
    m.choice_co_code = "DTAZ"
    m.availability_ca_var = "_avail_"
    return m


m1, m2 = _cm(1), _cm(2)
mk = lx.Model()
_me = P("g_intercept") * 1.0
for cov in MEMB:
    _me = _me + P(f"g_{cov}") * X(f"m_{cov}")
mk.utility_co[102] = _me
b = lx.LatentClass(mk, {101: m1, 102: m2}, datatree=dtc, groupid="group_id")
b.set_cap(20)
pnames = [str(n) for n in b.pnames]
crash_pos = pnames.index(f"c1_{CRASH0}") if f"c1_{CRASH0}" in pnames else None
print(f"params={len(pnames)} crash_pos={crash_pos}", flush=True)

_rng = np.random.default_rng(SEED)
_best = None
_restart_lls = []
_base = np.asarray(b.pvals, float).copy()
for _s in range(1 + N_PERT):
    if _s:
        jit = _rng.normal(0.0, 0.5 * np.maximum(np.abs(_base), 0.05))
        try:
            b.pvals = _base + jit
        except Exception as e:
            print("  jitter set failed:", e, flush=True)
    t0 = time.time()
    r = b.maximize_loglike(method="slsqp",
                           options={"maxiter": 300, "ftol": 1e-7},
                           quiet=True)
    ll = float(b.loglike(check_if_best=False, error_if_bad=False))
    _restart_lls.append(ll)
    print(f"  restart {_s}: ok={r.get('success')} ll={ll:.2f} "
          f"nit={r.get('nit')} ({time.time()-t0:.0f}s)", flush=True)
    if np.isfinite(ll) and (_best is None or ll > _best[0]):
        _best = (ll, np.asarray(b.pvals, float).copy())
ll_best, pv_best = _best
try:
    b.pvals = pv_best
except Exception:
    pass
print(f"BEST ll={ll_best:.2f} restarts={[round(v, 1) for v in _restart_lls]}",
      flush=True)
pf = b.pf
print("class coefs:", {str(n): round(float(pf.loc[n, 'value']), 3)
                       for n in pf.index}, flush=True)
try:
    pm = b.class_membership_model.pf
    print("membership:", {str(n): round(float(pm.loc[n, 'value']), 3)
                          for n in pm.index}, flush=True)
except Exception as e:
    print("membership read failed:", e, flush=True)
