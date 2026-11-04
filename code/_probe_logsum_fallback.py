"""What does _dest_tree actually put in the logsum column for a tour that has
no row in the cached mode-logsum surface?

The audit in _paper_stage5_arms.py reports those tours as taking a "0.0
fallback".  The saved diagnostic frame says otherwise: for the 79 tours absent
from the surface the logsum column has a per-case mean of exactly 0 and a
per-case sd around 0.48, i.e. real within-case variation with the case mean
removed.  One of the two is wrong, so read the column straight out of
_dest_tree instead of inferring it.

Runs _dest_tree on a handful of tours (3 without a surface row, 3 with one) so
the call is cheap; _dest_tree itself does no estimation.
"""
import os, sys, time

os.environ.setdefault("NEW_POP", "1")
os.environ.setdefault("NEW_POP_MAPPED", "0")
os.environ.setdefault("NEW_POP_DIR", "../new_pop")
os.environ.setdefault("PIPELINE_SKIMS_PATH", "../demand_/skims_371.omx")
CODE = os.environ.get("PIPELINE_CODE_DIR") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
os.chdir(CODE)

import cloudpickle
import numpy as np
import openmatrix as omx
import stage5_destination_choice as S5

CACHE = os.environ.get("PAPER_CACHE", "runs/_s5_larch_perturb/work_dest_cache.pkl")
SEG = os.environ.get("PAPER_SEGMENT", "work")

cache = cloudpickle.load(open(CACHE, "rb"))
seg = cache["seg_acdc"]
sa = cache["sampled_alts"]
ls = cache["logsums"]
ls_ids = {str(c) for c in ls.coords["cases"].values}
seg_ids = [str(p) for p in seg["ptour_id"].astype(str).unique()]
missing = [c for c in seg_ids if c not in ls_ids]
kept = [c for c in seg_ids if c in ls_ids][:3]
pick = missing[:3] + kept
print(f"segment tours={len(seg_ids)} in_surface={len(ls_ids)} missing={len(missing)}")
print(f"probing {len(pick)} tours: {pick}")

seg_p = seg[seg["ptour_id"].astype(str).isin(pick)].copy()
sa_p = sa[sa["ptour_id"].astype(str).isin(pick)].copy()
print(f"reduced: seg rows={len(seg_p)} sampled rows={len(sa_p)}")

skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
try:
    skim_df = S5._skim_flat(skims)
    S5._skim_lookup(skim_df)
    t0 = time.time()
    dt = S5._dest_tree(seg_p, cache["persons"], cache["lu_with_st"], skims,
                       ls, sampled_alts=sa_p, skim_df=skim_df,
                       crash_skims=skims, segment=SEG)
    print(f"_dest_tree returned in {time.time() - t0:.0f}s; "
          f"data_vars has logsum={'logsum' in dt.data_vars}")
finally:
    try:
        skims.close()
    except Exception:
        pass

arr = np.asarray(dt["logsum"].values, dtype=float)
print(f"dt['logsum'] shape={arr.shape}")
cases = [str(c) for c in np.asarray(dt.coords["ptour_id"].values).ravel()]
for cid in pick:
    rows = [i for i, c in enumerate(cases) if c == cid]
    if not rows:
        print(f"  {cid}: absent from the Dataset")
        continue
    v = arr[rows] if arr.ndim == 2 else arr[0][rows]
    v = np.asarray(v, dtype=float).ravel()
    print(f"  {cid}: in_surface={cid in ls_ids} n={v.size} "
          f"mean={v.mean():+.4f} sd={v.std():.4f} min={v.min():+.4f} "
          f"max={v.max():+.4f} n_zero={int((np.abs(v) < 1e-12).sum())}")

# The same rows through the frame builder the arms use.
frame, fmap = S5._adaptive_dest_choice_frame_full(
    dt, sa_p, SEG, case_covariates=["STUDYING", "sexp", "incp", "vehrd"])
lc = [c for c, v in fmap.items() if str(v) == "c_logsum"]
print(f"frame logsum column(s): {lc}")
for col in lc:
    g = frame.groupby("case_id")[col]
    stat = g.agg(["mean", "std"])
    print(stat.round(4).to_string())
