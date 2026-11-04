"""Same tours, both code paths: what the logsum column looks like before and
after gating the logsum-perp-distance step to rows with a real surface value.

Loads two copies of stage5_destination_choice -- the pre-fix backup (OLD) and the
patched module (NEW) -- and runs _dest_tree plus the arms' frame builder on 3
tours that have no row on the mode-logsum surface and 3 that do.

For each tour it reports the within-case sd of the logsum column and the
within-case slope of logsum on log_DIST_1.  The slope is the quantity that was
wrong: a constant-zero fill came out of the old residualization as
-(beta_1*dw + beta_2*ldw), i.e. a mirror of distance, and should now be 0.

Run:  python _probe_fill_before_after.py <path-to-bak-file>
"""
import glob
import importlib.machinery
import importlib.util
import os
import sys
import time

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

CACHE = os.environ.get("PAPER_CACHE", "runs/_s5_larch_perturb/work_dest_cache.pkl")
SEG = os.environ.get("PAPER_SEGMENT", "work")


def load_module(path, name):
    # The backup has no .py suffix, so the loader has to be named explicitly
    # (spec_from_file_location returns None without one).
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


bak = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("stage5_destination_choice.py.bak_logsumfill_*"))[-1]
print(f"OLD module: {bak}")
OLD = load_module(bak, "_s5_old_fill")
NEW = load_module(os.path.join(CODE, "stage5_destination_choice.py"), "_s5_new_fill")
print("NEW module: stage5_destination_choice.py")

cache = cloudpickle.load(open(CACHE, "rb"))
seg, sa, ls = cache["seg_acdc"], cache["sampled_alts"], cache["logsums"]
ls_ids = {str(c) for c in ls.coords["cases"].values}
seg_ids = [str(p) for p in seg["ptour_id"].astype(str).unique()]
missing = [c for c in seg_ids if c not in ls_ids]
kept = [c for c in seg_ids if c in ls_ids][:3]
pick = missing[:3] + kept
print(f"segment tours={len(seg_ids)} in_surface={len(ls_ids)} missing={len(missing)}")
print(f"probing {len(pick)}: {pick}")

seg_p = seg[seg["ptour_id"].astype(str).isin(pick)].copy()
sa_p = sa[sa["ptour_id"].astype(str).isin(pick)].copy()


def within_case_slope(frame, ycol, xcol):
    """Per-case slope of y on x (no intercept beyond the case mean)."""
    y = frame[ycol].astype(float).values
    x = frame[xcol].astype(float).values
    g = frame.groupby("case_id", sort=False)
    dy = y - g[ycol].transform("mean").astype(float).values
    dx = x - g[xcol].transform("mean").astype(float).values
    out = {}
    for cid, idx in g.indices.items():
        den = float((dx[idx] ** 2).sum())
        out[str(cid)] = float((dx[idx] * dy[idx]).sum()) / den if den > 0 else np.nan
    return out


res = {}
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
try:
    skim_df = NEW._skim_flat(skims)
    NEW._skim_lookup(skim_df)
    for tag, mod in (("OLD", OLD), ("NEW", NEW)):
        t0 = time.time()
        dt = mod._dest_tree(seg_p, cache["persons"], cache["lu_with_st"], skims,
                            ls, sampled_alts=sa_p, skim_df=skim_df,
                            crash_skims=skims, segment=SEG)
        frame, fmap = mod._adaptive_dest_choice_frame_full(
            dt, sa_p, SEG, case_covariates=["STUDYING", "sexp", "incp", "vehrd"])
        lcol = [c for c, v in fmap.items() if str(v) == "c_logsum"][0]
        xcol = [c for c, v in fmap.items() if str(v) == "c_log_dist"][0]
        g = frame.groupby("case_id")[lcol]
        sd = g.std().to_dict()
        slope = within_case_slope(frame, lcol, xcol)
        res[tag] = {str(k): (float(sd[k]), float(slope[k])) for k in sd}
        print(f"[{tag}] _dest_tree+frame {time.time() - t0:.0f}s  "
              f"rows={len(frame)} cases={frame['case_id'].nunique()}")
finally:
    try:
        skims.close()
    except Exception:
        pass

print()
print(f"{'tour':22s} {'surface':7s} | {'OLD sd':>7s} {'NEW sd':>7s} | "
      f"{'OLD d(logsum)/d(log DIST)':>26s} {'NEW':>8s}")
for cid in pick:
    o_sd, o_sl = res["OLD"][cid]
    n_sd, n_sl = res["NEW"][cid]
    print(f"{cid:22s} {str(cid in ls_ids):7s} | {o_sd:7.4f} {n_sd:7.4f} | "
          f"{o_sl:+26.4f} {n_sl:+8.4f}")

# Whole-sample view: the 79 fallback tours in one build each way.
print()
print("sampled 6 tours above; running the full segment would need the arms job.")
