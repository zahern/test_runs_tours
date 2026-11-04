"""dt-level logsum diagnostic (COMPUTE NODE ONLY).

Confirms the flat-logsum diagnosis before changing stage5_destination_choice.py:
  1. the exact larch logsums() failure (fix 1 target attribute)
  2. OMX mapping size / origin-hit rate / fringe share (fix 2 diagnosis)
  3. whether the mode coefficient cache is populated
  4. the fallback's resulting within-case cross-zone std
"""
import os
import sys
import logging
import traceback
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")

os.environ.setdefault("NEW_POP", "1")
os.environ.setdefault("NEW_POP_DIR", "../new_pop")
os.environ.setdefault("NEW_POP_MAPPED", "0")
os.environ.setdefault("PIPELINE_SKIMS_PATH", "../demand_/skims_371.omx")
# Abort right after the logsum diagnostic (mode fit only, no destination fit).
os.environ.setdefault("STAGE5_REQUIRE_LOGSUM_VARIANCE", "1")

import numpy as np
import stage5_destination_choice as S5

_orig_fallback = S5._logsums_numpy_fallback
_orig_logsums = S5._logsums
_TAG = {"n": 0}


def _diag_mapping(skims, dt_mode, case_ids, zone_ids):
    z2i = {}
    try:
        mapping = dict(skims.mapping(skims.listMappings()[0])) if skims is not None else {}
        z2i = {int(k): int(v) for k, v in mapping.items()}
    except Exception as e:
        print(f"[diag] mapping read FAILED: {type(e).__name__}: {e}", flush=True)
    otaz = np.ones(len(case_ids), dtype=np.int64)
    try:
        tdf = dt_mode["tour"].to_dataframe()
        tdf.index = tdf.index.map(str)
        ids = [str(c) for c in case_ids]
        if "OTAZ" in tdf.columns:
            for i, cid in enumerate(ids):
                if cid in tdf.index:
                    otaz[i] = int(tdf.at[cid, "OTAZ"])
    except Exception as e:
        print(f"[diag] tour read FAILED: {type(e).__name__}: {e}", flush=True)
    hit = sum(1 for z in otaz if int(z) in z2i)
    n999 = int((otaz == 999).sum())
    uniq = np.unique(otaz)
    print(f"[diag] n_cases={len(case_ids)} n_zones={len(zone_ids)} "
          f"map_size={len(z2i)} origin_hits={hit}/{len(otaz)} "
          f"origin_999={n999} n_unique_origins={len(uniq)} "
          f"origin_sample={otaz[:8].tolist()} uniq_sample={uniq[:8].tolist()} "
          f"zone_sample={list(zone_ids)[:4]}", flush=True)
    # Are any origins resolvable at all?
    if hit == 0:
        print("[diag] *** ZERO origins resolve in the OMX mapping -> "
              "every row falls back to the constant fill -> FLAT LOGSUMS ***",
              flush=True)
    return hit, n999


def diag_fallback(mode_model, dt_mode, case_ids, zone_ids, skims=None,
                  verbose=False, coef_override=None):
    _TAG["n"] += 1
    print(f"\n[diag] ===== fallback call #{_TAG['n']} =====", flush=True)
    try:
        _diag_mapping(skims, dt_mode, case_ids, zone_ids)
    except Exception as e:
        print("[diag] mapping diag failed:", type(e).__name__, e, flush=True)
    try:
        cache = S5._coef_dict_from_model(mode_model, coef_override)
        names = sorted(cache.keys())
        print(f"[diag] coef cache size={len(cache)} sample={names[:10]}",
              flush=True)
        zero = [k for k, v in cache.items() if float(v) == 0.0]
        print(f"[diag] coef cache zeros={len(zero)} of {len(cache)}", flush=True)
    except Exception as e:
        print("[diag] coef cache read failed:", type(e).__name__, e, flush=True)
    out = _orig_fallback(mode_model, dt_mode, case_ids, zone_ids,
                         skims=skims, verbose=verbose,
                         coef_override=coef_override)
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 2 and arr.shape[1] > 1:
        within = np.nanstd(arr, axis=1)
        print(f"[diag] FALLBACK within-case std: median={np.nanmedian(within):.6g} "
              f"max={np.nanmax(within):.6g} "
              f"flat_cases={float(np.mean(within < 1e-9)):.1%} "
              f"global_std={float(np.nanstd(arr)):.6g}", flush=True)
    else:
        print("[diag] fallback returned shape", arr.shape, flush=True)
    return out


def diag_logsums(*args, **kwargs):
    mode_model = kwargs.get("mode_model", args[0] if len(args) > 0 else None)
    dt_mode = kwargs.get("dt_mode", args[1] if len(args) > 1 else None)
    if mode_model is not None and dt_mode is not None:
        print("\n[diag] ----- probing the LARCH logsums() path -----", flush=True)
        for label, build in (("datatree-kwarg", 0), ("set+logsums", 1)):
            try:
                mt = mode_model.copy()
            except Exception as e:
                print(f"[diag] copy failed: {type(e).__name__}: {e}", flush=True)
                break
            try:
                if build == 0:
                    res = np.asarray(mt.logsums(datatree=dt_mode),
                                     dtype=np.float32)
                else:
                    mt.datatree = dt_mode
                    mt.mangle()
                    mt.unmangle()
                    res = np.asarray(mt.logsums(), dtype=np.float32)
                res2 = np.asarray(res, dtype=float)
                if res2.ndim == 2 and res2.shape[1] > 1:
                    within = np.nanstd(res2, axis=1)
                    print(f"[diag] LARCH logsums() OK via {label}: "
                          f"shape={res2.shape} within-case std "
                          f"median={np.nanmedian(within):.6g} "
                          f"flat={float(np.mean(within < 1e-9)):.1%}", flush=True)
                else:
                    print(f"[diag] LARCH logsums() OK via {label}: "
                          f"shape={res2.shape}", flush=True)
                break
            except Exception as e:
                print(f"[diag] LARCH logsums() FAILED via {label}: "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc(limit=6)
    return _orig_logsums(*args, **kwargs)


S5._logsums_numpy_fallback = diag_fallback
S5._logsums = diag_logsums

import pipeline_logger as pl

OUT_DIR = os.path.abspath("runs/probe_logsum_diag")
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)
pl.OUT_DIR = OUT_DIR
pl.PIPELINE_OUT_DIR = OUT_DIR
pl.PIPELINE_RUN_DIR = OUT_DIR
print(f"[diag] probe out dir: {OUT_DIR}", flush=True)

_, out, landuse = pl.load_data()
models, skims, _ = pl.estimate_stage5(out, landuse)
if skims is not None:
    try:
        skims.close()
    except Exception:
        pass
print(f"[diag] segments returned: {sorted(models.keys()) if isinstance(models, dict) else models}",
      flush=True)
print("PROBE DONE", flush=True)
