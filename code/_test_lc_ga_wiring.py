"""End-to-end wiring test: the real Stage-5 GA LC fitness path
(_stage5_dest_lc_fitness) on cached real QLD data with a small subsample."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("NEW_POP", "1")
os.environ.setdefault("NEW_POP_DIR", r"Z:\test_runs_tours\new_pop")
os.environ.setdefault("NEW_POP_MAPPED", "0")
os.environ.setdefault("PIPELINE_SKIMS_PATH",
                      r"Z:\test_runs_tours\demand_\skims_371.omx")
os.environ.setdefault("STAGE5_LC_N_INIT", "1")
os.environ.setdefault("STAGE5_LC_MAXITER", "60")
os.environ.setdefault("STAGE5_DEST_EST_CASES", "300")
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)
import numpy as np
import pandas as pd
import cloudpickle


def patch():
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


patch()
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
import stage5_destination_choice as S5
import ga_hypothesis_testing as G
import stage5_lc_genes as L
from stage5_lc_genes import extend_candidates, decode_lc_spec

_orig_fit = L.fit_lc_genome


def _fit_debug(frame, c2c, spec, **kw):
    sc, info = _orig_fit(frame, c2c, spec, **kw)
    print("[wiring-debug] score=", sc, "info=",
          json.dumps(info, default=str)[:600], flush=True)
    return sc, info


L.fit_lc_genome = _fit_debug

cache = cloudpickle.load(open(
    r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
seg = cache["seg_acdc"]
N = int(os.environ.get("STAGE5_DEST_EST_CASES", "300"))
if N < seg["ptour_id"].nunique():
    keep = set(seg[["ptour_id"]].drop_duplicates().sample(
        n=N, random_state=42)["ptour_id"].astype(str))
    seg = seg[seg["ptour_id"].astype(str).isin(keep)].copy()

import openmatrix as omx
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = S5._skim_flat(skims)
S5._skim_lookup(skim_df)
persons = cache["persons"]
lu_st = cache["lu_with_st"]
ls_cached = cache["logsums"]

BASE = ["c_log_dist", "c_crash", "c_visit_work"]
CANDS = extend_candidates(BASE)
idx = {c: i for i, c in enumerate(CANDS)}
g_lc = np.zeros(len(CANDS), dtype=int)
for nm in ("c_log_dist", "c_crash", "c_visit_work", "lc_on",
           "lc_memb:STUDYING"):
    g_lc[idx[nm]] = 1
g_pool = np.zeros(len(CANDS), dtype=int)
for nm in ("c_log_dist", "c_crash", "c_visit_work"):
    g_pool[idx[nm]] = 1

spec = decode_lc_spec(g_lc, CANDS, BASE)
print(f"[wiring] spec lc_on={spec['lc_on']} nc={spec['n_classes']} "
      f"features={spec['features']} memb={spec['membership_vars']}", flush=True)

t0 = time.time()
score = G._stage5_dest_lc_fitness(
    g_lc, CANDS, BASE, "work", seg, persons, lu_st, skims, ls_cached, seg)
dt = time.time() - t0
print(f"[wiring] LC genome score(-BIC/n)={score:.4f} ({dt:.0f}s)", flush=True)
skims.close()

ok = np.isfinite(score) and score > -1e8
print(f"[wiring] {'PASS' if ok else 'FAIL'}: real-frame LC fitness path")
json.dump({"score": float(score), "seconds": dt, "ok": bool(ok),
           "spec": {k: v for k, v in spec.items()}},
          open(r"runs\_s5_larch_perturb\lc_ga_wiring_test.json", "w"),
          indent=2, default=str)
sys.exit(0 if ok else 1)
