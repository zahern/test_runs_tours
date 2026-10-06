"""Real-frame LC destination with a guaranteed class-specific CRASH component
and proper perturbation: multi-restart jitter of class betas + membership
gammas (SearchLibrium estimate_dest_choice_lc)."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("NEW_POP", "1")
os.environ.setdefault("NEW_POP_DIR", r"Z:\test_runs_tours\new_pop")
os.environ.setdefault("NEW_POP_MAPPED", "0")
os.environ.setdefault("PIPELINE_SKIMS_PATH",
                      r"Z:\test_runs_tours\demand_\skims_371.omx")
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np
import pandas as pd
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
from SearchLibrium.mh_choice_estimator import estimate_dest_choice_lc

N_CASES = int(os.environ.get("LC_N_CASES", "1200"))
cache = cloudpickle.load(open(
    r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
seg = cache["seg_acdc"]; sa = cache["sampled_alts"]
if N_CASES and N_CASES < seg["ptour_id"].nunique():
    keep = set(seg[["ptour_id"]].drop_duplicates().sample(
        n=N_CASES, random_state=42)["ptour_id"].astype(str))
    seg = seg[seg["ptour_id"].astype(str).isin(keep)].copy()
    sa = sa[sa["ptour_id"].astype(str).isin(keep)].copy()
import openmatrix as omx
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = S5._skim_flat(skims)
S5._skim_lookup(skim_df)
MEMB = ["incp", "vehrd", "STUDYING", "age5p", "sexp", "safety_sensitivity"]

# _adaptive_dest_choice_frame_full expects the flat Dataset; rebuild it
dt = S5._dest_tree(seg, cache["persons"], cache["lu_with_st"], skims,
                   cache["logsums"], sampled_alts=sa, skim_df=skim_df,
                   crash_skims=skims, segment="work")
skims.close()
frame, fmap = S5._adaptive_dest_choice_frame_full(
    dt, sa, "work", case_covariates=MEMB)
covs = [c for c in MEMB if c in frame.columns]
print(f"[lc-crash] cases={frame['case_id'].nunique()} features={len(fmap)} "
      f"membership_vars={covs}", flush=True)

t0 = time.time()
res = estimate_dest_choice_lc(
    frame, list(fmap), n_classes=2,
    membership_vars=covs,
    n_init=2, perturb_sigma=0.5, maxiter=60, share_floor=0.10, membership_l2=0.1, seed=7,
    require_crash=True)
print(f"done {time.time()-t0:.0f}s  ll={res['loglike']:.1f} bic={res['bic']:.1f} "
      f"conv={res['converged']} iters={res['iterations']}")
print("shares:", [round(s, 3) for s in res["shares"]])
print("restart LLs:", [round(v, 1) for v in res.get("restart_lls", [])],
      "span=", round(res.get("restart_ll_span", 0.0), 2))
print("class_crash:", res.get("class_crash"),
      "gap=", res.get("crash_class_gap"), "cols=", res.get("crash_cols"))
print("membership_coefs:")
for k, v in (res.get("membership_coefs") or {}).items():
    print(f"  {k:28s} {v:+.4f}")
for c in range(2):
    b = res["betas"][c]
    top = sorted(b, key=lambda n: -abs(b[n]))[:6]
    print(f"class {c}: " + ", ".join(f"{n}={b[n]:+.2f}" for n in top))
json.dump({k: v for k, v in res.items() if k != "responsibilities"},
          open(r"runs\_s5_larch_perturb\work_lc_crash.json", "w"), indent=2,
          default=str)
print("saved -> work_lc_crash.json")
