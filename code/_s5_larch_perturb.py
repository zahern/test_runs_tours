"""Local larch destination fit + perturbation test for Stage 5.

Answers two questions without the cluster:
  1. can larch fit the Stage-5 destination models at all (finite LL,
     optimizer success), and
  2. if we PERTURB the fitted solution and re-optimise, do we get NEW
     (distinct / better) destination models, or does every start collapse to
     the same local optimum?

The estimation frame (flat lx.Dataset with sampled alternatives + mode
logsums) is built ONCE by running the real _run_segment path, then cached so
perturbation refits are cheap.  SearchLibrium MH coefficients are read back
from runs/stage5_compare/sl_20260930/stage5_models.pkl for comparison.

Usage:
  python _s5_larch_perturb.py all     --segment work --n 6
  python _s5_larch_perturb.py build   --segment work
  python _s5_larch_perturb.py perturb --segment work --n 6 --sigma 0.5
  python _s5_larch_perturb.py compare --segment work

NOTE (local box): numpy is 2.5.2 / statsmodels 0.14.2, so the MODE model fit
is degenerate (LL=-0.5, rho2=1.000 vs the cluster's -5151.24) and the mode
logsums carry almost no cross-zone signal.  Destination LLs from this script
are therefore NOT comparable to the cluster SearchLibrium LLs; coefficient
shape / convergence / perturbation diversity ARE meaningful.
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("NEW_POP", "1")
os.environ["NEW_POP_DIR"] = r"Z:\test_runs_tours\new_pop"
os.environ["NEW_POP_MAPPED"] = "0"
os.environ["PIPELINE_SKIMS_PATH"] = r"Z:\test_runs_tours\demand_\skims_371.omx"
os.environ["PIPELINE_SEED"] = "42"
os.environ["PIPELINE_ACCELERATOR"] = "cpu"
os.environ["LARCH_FORCE_JAX"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["STAGE5_CLEAR_SHARROW_CACHE"] = "1"
os.environ.setdefault("STAGE5_DEST_ESTIMATOR", "larch")
os.environ.setdefault("STAGE5_DEST_SAMPLE_SIZE", "20")
os.environ.setdefault("STAGE5_REQUIRE_DEST_ESTIMATION", "1")
os.environ.setdefault("STAGE5_DEST_ORTHOG", "1")

CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
import pandas as pd

RUN_DIR = os.path.join(CODE, "runs", "_s5_larch_perturb")
os.makedirs(RUN_DIR, exist_ok=True)

SEARCHLIBRIUM_PKL = os.path.join(
    CODE, "runs", "stage5_compare", "sl_20260930", "stage5_models.pkl")


def _cache_path(segment):
    return os.path.join(RUN_DIR, f"{segment}_dest_cache.pkl")


def _results_path(segment, tag=None):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(RUN_DIR, f"{segment}_perturb_results{suffix}.json")


def _coefs_path(segment, tag=None):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(RUN_DIR, f"{segment}_perturb_coefficients{suffix}.csv")


def _log(msg):
    print(f"[perturb] {msg}", flush=True)


def _extract_coefs(mdl):
    # _coef_cache is the plain-Python snapshot taken immediately after
    # maximize_loglike, BEFORE JAX can clobber model.pf / the stuck-drop
    # revert can leave stale zeros in _final_pf.  Prefer it; fall back to
    # _final_pf/pf only when no snapshot exists.
    cc = getattr(mdl, "_coef_cache", None)
    if isinstance(cc, dict) and cc:
        out = {}
        for name, val in cc.items():
            try:
                v = float(val)
                if np.isfinite(v):
                    out[str(name)] = v
            except (TypeError, ValueError):
                continue
        if out:
            return out
    pf = getattr(mdl, "_final_pf", None)
    if pf is None or not isinstance(pf, pd.DataFrame) or "value" not in pf:
        pf = getattr(mdl, "pf", None)
    out = {}
    if pf is None:
        return out
    for name in pf.index:
        try:
            v = float(pf.loc[name, "value"])
            if np.isfinite(v):
                out[str(name)] = v
        except Exception:
            continue
    return out


def _model_ll(mdl):
    ll = getattr(mdl, "_cached_ll", float("nan"))
    try:
        ll = float(ll)
    except Exception:
        ll = float("nan")
    if not np.isfinite(ll):
        try:
            ll = float(mdl.most_recent_estimation_result.loglike)
        except Exception:
            pass
    return ll


# ─────────────────────────────────────────────────────────────────────────────
# build: run the real segment path once, cache the estimation frame
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build(args):
    import cloudpickle
    import pipeline_logger as pl
    import openmatrix as omx
    import stage5_destination_choice as S5
    from stage5_destination_choice import (
        _skim_flat, _skim_lookup, _prep_hhts_tours, _run_segment,
    )

    captured = {}

    _orig_dest_tree = S5._dest_tree

    def _spy_dest_tree(tours, persons, landuse, skims, logsums,
                       sampled_alts=None, skim_df=None, crash_skims=None,
                       segment="", **kw):
        dt = _orig_dest_tree(
            tours, persons, landuse, skims, logsums,
            sampled_alts=sampled_alts, skim_df=skim_df,
            crash_skims=crash_skims, segment=segment, **kw)
        if "dt" not in captured and sampled_alts is not None:
            captured["dt"] = dt
            captured["seg_acdc"] = tours
            captured["sampled_alts"] = sampled_alts
            captured["lu_with_st"] = landuse
            captured["logsums"] = logsums
            _log(f"_dest_tree captured: {len(tours)} tours, "
                 f"{0 if sampled_alts is None else len(sampled_alts)} sampled rows")
        return dt

    S5._dest_tree = _spy_dest_tree

    t0 = time.time()
    _, out, landuse = pl.load_data()
    _log(f"load_data {time.time()-t0:.0f}s | tours={len(out['tours']):,} "
         f"persons={len(out['persons']):,} landuse={len(landuse):,}")

    hhts = _prep_hhts_tours(out["tours"], out["persons"])
    skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
    skim_df = _skim_flat(skims)
    _skim_lookup(skim_df)

    SEG = args.segment
    _log(f"=== running _run_segment('{SEG}') (this is the slow part) ===")
    t1 = time.time()
    res = _run_segment(
        SEG, hhts, out["persons"], landuse,
        skims, skim_df=skim_df, seed=42, verbose=True,
        crash_skims=skims,
    )
    _log(f"_run_segment finished in {time.time()-t1:.0f}s")

    try:
        skims.close()
    except Exception:
        pass

    dest_ll = res.get("dest_ll")
    dest_coef = res.get("dest_coef") or {}
    _log(f"baseline mode_ll={res.get('mode_ll')} dest_ll={dest_ll} "
         f"n_dest_coef={len(dest_coef)}")
    _log(f"dest_fit_diagnostics={res.get('dest_fit_diagnostics')}")

    # NOTE: the lx.Dataset is deliberately NOT cached - cloudpickling it
    # produced a ~4 GB file that fails to unpickle (larch Flow.__getattr__
    # recursion).  Cache the lightweight _dest_tree INPUTS instead and rebuild
    # the frame in-process (seconds), which is what perturb does.
    cache = {
        "segment": SEG,
        "dest_coef": dict(dest_coef),
        "dest_ll": float(dest_ll) if dest_ll is not None else float("nan"),
        "dest_fit_diagnostics": dict(res.get("dest_fit_diagnostics") or {}),
        "mode_ll": float(res.get("mode_ll", float("nan"))),
        "seg_acdc": captured.get("seg_acdc"),
        "persons": out["persons"],
        "lu_with_st": captured.get("lu_with_st"),
        "sampled_alts": captured.get("sampled_alts"),
        "logsums": captured.get("logsums"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = _cache_path(SEG)
    with open(path, "wb") as fh:
        cloudpickle.dump(cache, fh)
    _log(f"cache written -> {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    finite = np.isfinite(cache["dest_ll"])
    _log("VERDICT: larch destination "
         + ("PRODUCED A FINITE LL" if finite else "STILL FAILED (LL=nan)"))
    return 0 if finite else 1


# ─────────────────────────────────────────────────────────────────────────────
# perturb: refit from the baseline solution + jittered starts
# ─────────────────────────────────────────────────────────────────────────────

def _refit(dt, segment, seed_coefs, sigma, rng, abs_floor=0.0):
    import stage5_destination_choice as S5

    mm = S5._build_dest_model(dt, segment)
    pf0 = mm.pf
    n_total = len(pf0)
    holdfast = (pf0["holdfast"].fillna(False).astype(bool)
                if "holdfast" in pf0.columns
                else pd.Series(False, index=pf0.index))

    applied = 0
    for name, val in (seed_coefs or {}).items():
        if name in pf0.index and np.isfinite(float(val)):
            try:
                mm.set_value(name, float(val))
                applied += 1
            except Exception:
                pass

    jittered = []
    if sigma > 0:
        for name in pf0.index:
            if bool(holdfast.get(name, False)):
                continue
            try:
                v = float(mm.pf.loc[name, "value"])
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            _scale = max(sigma * max(abs(v), 0.05), abs_floor)
            delta = float(rng.normal(0.0, _scale))
            v2 = v + delta
            try:
                lo = float(mm.pf.loc[name, "minimum"])
            except Exception:
                lo = float("nan")
            try:
                hi = float(mm.pf.loc[name, "maximum"])
            except Exception:
                hi = float("nan")
            if np.isfinite(lo):
                v2 = max(v2, lo + 1e-6)
            if np.isfinite(hi):
                v2 = min(v2, hi - 1e-6)
            try:
                mm.set_value(name, v2)
                jittered.append(name)
            except Exception:
                pass

    t0 = time.time()
    mm = S5._estimate(mm, two_phase=True)
    ll = _model_ll(mm)
    coefs = _extract_coefs(mm)
    diag = dict(getattr(mm, "_fit_diagnostics", {}) or {})
    return {
        "ll": ll,
        "coefs": coefs,
        "diagnostics": {k: v for k, v in diag.items()
                        if isinstance(v, (int, float, str, bool, type(None)))},
        "n_params": n_total,
        "n_seeded": applied,
        "n_jittered": len(jittered),
        "seconds": round(time.time() - t0, 1),
    }


def _distinct(base, other):
    common = [k for k in base if k in other]
    if not common:
        return {"n_common": 0, "max_abs_diff": None, "l2": None}
    d = np.array([other[k] - base[k] for k in common], dtype=float)
    return {
        "n_common": len(common),
        "max_abs_diff": float(np.max(np.abs(d))),
        "l2": float(np.sqrt(np.sum(d ** 2))),
        "n_moved_gt_1e-3": int(np.sum(np.abs(d) > 1e-3)),
    }


def _patch_larch_flow():
    """Guard larch Flow.__getattr__ against a missing _obj during unpickle.

    cloudpickle round-trips of an lx.Dataset hit an infinite recursion in
    larch.dataset.flow._GenericFlow.__getattr__ (getattr(self._obj, name)
    with self._obj itself a half-initialised Flow).  Without this the cached
    frame cannot be reloaded at all.
    """
    try:
        from larch.dataset import flow as _flow

        def _safe_getattr(self, name):
            if name == "_obj":
                raise AttributeError(name)
            obj = self.__dict__.get("_obj")
            if obj is None:
                raise AttributeError(name)
            result = getattr(obj, name)
            try:
                result = self.transfer_dimension_attrs(result)
            except Exception:
                pass
            return result

        for _cls in ("_GenericFlow", "_DatasetDC", "_DataArrayDC",
                     "_DataTreeDC"):
            _c = getattr(_flow, _cls, None)
            if _c is not None:
                _c.__getattr__ = _safe_getattr
        _log("larch Flow.__getattr__ patched for cache unpickle")
    except Exception as e:
        _log(f"larch Flow patch skipped ({e!r})")


def _rebuild_dt(cache):
    import openmatrix as omx
    import stage5_destination_choice as S5
    from stage5_destination_choice import _skim_flat, _skim_lookup

    skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
    skim_df = _skim_flat(skims)
    _skim_lookup(skim_df)
    _log(f"rebuilding dest frame via _dest_tree "
         f"(alts={0 if cache['sampled_alts'] is None else len(cache['sampled_alts'])})")
    t0 = time.time()
    dt = S5._dest_tree(
        cache["seg_acdc"], cache["persons"], cache["lu_with_st"], skims,
        cache["logsums"], sampled_alts=cache["sampled_alts"],
        skim_df=skim_df, crash_skims=skims, segment=cache["segment"])
    _log(f"_dest_tree rebuilt in {time.time()-t0:.0f}s "
         f"({len(dt.coords.get('_altid_', []))} alternatives)")
    return dt, skims


def cmd_perturb(args):
    import cloudpickle

    seg = args.segment
    cache_path = _cache_path(seg)
    if not os.path.isfile(cache_path):
        _log(f"no cache at {cache_path}; run `build` first")
        return 2
    _patch_larch_flow()
    with open(cache_path, "rb") as fh:
        cache = cloudpickle.load(fh)

    dt = cache.get("dt_dest")
    if dt is not None:
        _log("using cached dt_dest (no rebuild needed)")
    else:
        try:
            dt, skims = _rebuild_dt(cache)
        except Exception:
            traceback.print_exc()
            return 2
        try:
            skims.close()
        except Exception:
            pass
    base_coefs = dict(cache.get("dest_coef") or {})
    base_ll = float(cache.get("dest_ll", float("nan")))
    _log(f"baseline cached: LL={base_ll} n_coef={len(base_coefs)}")

    results = {
        "segment": seg,
        "baseline_ll": base_ll,
        "baseline_coef": base_coefs,
        "baseline_fit_diagnostics": cache.get("dest_fit_diagnostics") or {},
        "mode_ll_local_degenerate": cache.get("mode_ll"),
        "runs": [],
    }

    # Report the free-parameter set the perturbation actually explores.
    try:
        import stage5_destination_choice as S5
        _mm_probe = S5._build_dest_model(dt, seg)
        _pf = _mm_probe.pf
        _hf = (_pf["holdfast"].fillna(False).astype(bool)
               if "holdfast" in _pf.columns
               else pd.Series(False, index=_pf.index))
        _free = [p for p in _pf.index if not bool(_hf.get(p, False))]
        _log(f"model pf: {len(_pf)} params, {len(_free)} free")
        for _p in _free:
            _log(f"  free: {_p} = {float(_pf.loc[_p, 'value']):.6f}")
        results["free_params"] = [str(p) for p in _free]
        results["pf_start"] = {str(p): float(_pf.loc[p, "value"])
                               for p in _pf.index}
    except Exception as _pe:
        _log(f"pf probe failed: {_pe!r}")

    # 0) sanity: warm-start from baseline with NO jitter - should reproduce
    _log("refit 0/N: warm start, no jitter (reproducibility check)")
    r0 = _refit(dt, seg, base_coefs, 0.0, np.random.default_rng(42))
    results["runs"].append({"kind": "warmstart_no_jitter", **r0})
    _log(f"  LL={r0['ll']:.3f}  distinct={_distinct(base_coefs, r0['coefs'])}")

    # 1..N) jittered restarts
    for i in range(1, args.n + 1):
        rng = np.random.default_rng(args.seed + i)
        _log(f"refit {i}/{args.n}: jitter sigma={args.sigma} "
             f"abs_floor={args.abs_floor} (seed={args.seed + i})")
        r = _refit(dt, seg, base_coefs, args.sigma, rng,
                   abs_floor=args.abs_floor)
        r["distinct_from_seed_baseline"] = _distinct(base_coefs, r["coefs"])
        # Apples-to-apples reference is the NO-JITTER refit of the SAME spec
        # build (cache baseline came from a different/pruned spec).
        r["distinct_from_no_jitter_refit"] = _distinct(r0["coefs"], r["coefs"])
        results["runs"].append({"kind": "jittered", "index": i,
                                "sigma": args.sigma,
                                "abs_floor": args.abs_floor, **r})
        _log(f"  LL={r['ll']:.3f}  conv={r['diagnostics'].get('converged')} "
             f"nit={r['diagnostics'].get('nit')}  "
             f"max|dcoef vs refit0|="
             f"{r['distinct_from_no_jitter_refit']['max_abs_diff']}  "
             f"moved={r['distinct_from_no_jitter_refit']['n_moved_gt_1e-3']}")

    lls = [r["ll"] for r in results["runs"] if np.isfinite(r["ll"])]
    ref_ll = r0["ll"]
    best_ll = max(lls) if lls else float("nan")
    new_models = [r for r in results["runs"]
                  if r.get("kind") == "jittered"
                  and r.get("distinct_from_no_jitter_refit", {}).get("n_moved_gt_1e-3", 0) > 0]
    better = [r for r in results["runs"]
              if np.isfinite(r["ll"]) and np.isfinite(ref_ll)
              and r["ll"] > ref_ll + 0.5]
    results["summary"] = {
        "n_refits": len(results["runs"]),
        "n_finite_ll": len(lls),
        "best_ll": best_ll,
        "no_jitter_refit_ll": ref_ll,
        "cached_pruned_baseline_ll": base_ll,
        "improvement_over_no_jitter_refit": (best_ll - ref_ll)
        if (np.isfinite(best_ll) and np.isfinite(ref_ll)) else None,
        "n_new_distinct_models": len(new_models),
        "n_better_than_refit": len(better),
        "perturbation_produces_new_models": bool(len(new_models) > 0),
        "ll_span_across_refits": (max(lls) - min(lls)) if lls else None,
    }
    with open(_results_path(seg, getattr(args, "tag", None)), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    rows = []
    for r in results["runs"]:
        row = {"kind": r["kind"], "index": r.get("index"),
               "ll": r["ll"], "n_jittered": r.get("n_jittered"),
               "seconds": r.get("seconds")}
        row.update({k: v for k, v in r["diagnostics"].items()})
        if "distinct_from_no_jitter_refit" in r:
            row.update(r["distinct_from_no_jitter_refit"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        _coefs_path(seg, getattr(args, "tag", None)), index=False)

    _log(f"summary: {json.dumps(results['summary'], indent=2)}")
    _log(f"results -> {_results_path(seg, getattr(args, 'tag', None))}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# compare: local larch vs SearchLibrium MH (sl_20260930)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_compare(args):
    import cloudpickle
    import stage5_destination_choice as S5

    seg = args.segment
    _tag = getattr(args, "tag", None)
    res_path = _results_path(seg, _tag)
    if not os.path.isfile(res_path):
        _log(f"no perturb results at {res_path}; run `perturb` first")
        return 2
    with open(res_path, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    base_coefs = results.get("baseline_coef") or {}
    finite_runs = [r for r in results["runs"] if np.isfinite(r.get("ll", float("nan")))]
    best_run = max(finite_runs, key=lambda r: r["ll"]) if finite_runs else None
    larch_coefs = (best_run or {}).get("coefs") or base_coefs

    sl_coefs, sl_ll, sl_meta = {}, None, {}
    # Primary source: the coefficient CSV report (plain pandas, version-proof).
    sl_csv = os.path.join(os.path.dirname(SEARCHLIBRIUM_PKL),
                          "stage5_models_coefficients.csv")
    try:
        _cdf = pd.read_csv(sl_csv)
        _d = _cdf[(_cdf["segment"] == seg) &
                  (_cdf["model_type"].isin(["dest", "destination"]))]
        sl_coefs = {str(r.parameter): float(r.value) for r in _d.itertuples()}
        _lls = pd.to_numeric(_d["loglike"], errors="coerce").dropna().unique()
        sl_ll = float(_lls[0]) if len(_lls) else None
        _log(f"SearchLibrium report CSV: n_coef={len(sl_coefs)} LL={sl_ll}")
    except Exception as e:
        _log(f"SearchLibrium CSV read failed ({e!r}) - trying pkl")
        try:
            models = S5.load_models(SEARCHLIBRIUM_PKL)
            res = models.get(seg) or {}
            sl_coefs = dict(res.get("dest_coef") or {})
            sl_ll = res.get("dest_estimator_ll")
            sl_meta = dict(res.get("dest_estimator_metadata") or {})
            _log(f"SearchLibrium pkl loaded: n_coef={len(sl_coefs)} LL={sl_ll} "
                 f"ess_min={(sl_meta.get('diagnostics') or {}).get('ess_min')}")
        except Exception as e2:
            _log(f"SearchLibrium pkl load failed ({e2!r}) - larch-only comparison")

    common = sorted(set(larch_coefs) & set(sl_coefs))
    rows = []
    for name in common:
        rows.append({
            "param": name,
            "larch": larch_coefs[name],
            "searchlibrium_mh": sl_coefs[name],
            "diff": larch_coefs[name] - sl_coefs[name],
        })
    cmp_df = pd.DataFrame(rows)
    _suffix = f"_{_tag}" if _tag else ""
    cmp_path = os.path.join(RUN_DIR,
                            f"{seg}_larch_vs_searchlibrium{_suffix}.csv")
    cmp_df.to_csv(cmp_path, index=False)

    corr = None
    if len(cmp_df) > 2:
        corr = float(cmp_df["larch"].corr(cmp_df["searchlibrium_mh"]))
    sign_agree = None
    if len(cmp_df):
        sign_agree = float(np.mean(np.sign(cmp_df["larch"]) ==
                                   np.sign(cmp_df["searchlibrium_mh"])))

    summary = {
        "segment": seg,
        "larch_baseline_ll": results.get("baseline_ll"),
        "larch_best_ll": (best_run or {}).get("ll"),
        "larch_best_kind": (best_run or {}).get("kind"),
        "searchlibrium_ll": sl_ll,
        "searchlibrium_ess_min": (sl_meta.get("diagnostics") or {}).get("ess_min"),
        "n_common_params": len(common),
        "coef_pearson_r": corr,
        "sign_agreement": sign_agree,
        "max_abs_coef_diff": float(cmp_df["diff"].abs().max()) if len(cmp_df) else None,
        "note": ("Local larch LL uses degenerate local mode logsums "
                 "(numpy 2.x); SearchLibrium LL is from the cluster zigenv. "
                 "Compare coefficient shape/signs, not LL levels."),
    }
    out = os.path.join(RUN_DIR,
                       f"{seg}_comparison_summary{_suffix}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _log(f"comparison: {json.dumps(summary, indent=2)}")
    _log(f"coefficient table -> {cmp_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["all", "build", "perturb", "compare"])
    ap.add_argument("--segment", default="work")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--abs-floor", type=float, default=0.0,
                    help="minimum jitter std (absolute) so near-zero starts "
                         "are actually perturbed")
    ap.add_argument("--tag", default=None,
                    help="suffix for output artifacts (avoids clobbering)")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    if args.stage == "build":
        return cmd_build(args)
    if args.stage == "perturb":
        return cmd_perturb(args)
    if args.stage == "compare":
        return cmd_compare(args)
    rc = cmd_build(args)
    if rc != 0:
        return rc
    rc = cmd_perturb(args)
    if rc != 0:
        return rc
    return cmd_compare(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
