"""
_run_ga_shard.py
================
Run ONE micro-job shard: a single (stage, segment) search plus its
model-family variant shootout.

Usage (PBS calls this; see job_submit_ga_sharded.pbs):
    python -u _run_ga_shard.py --stage-id 4 --segment work \
        --run-root runs/ga_parallel --run-tag qld

What it does:
  1. Sets GA_SHARD_SEGMENT so the stage search only runs this segment
     (filter hooks in ga_hypothesis_testing._shard_filter).
  2. Redirects OUT_DIR + GA_CHECKPOINT_DIR to
     <run_root>/<run_tag>/stage<N>/shards/<segment>/ so sibling shards
     never clobber each other's ga_results.pkl / checkpoints.
  3. Runs the normal single-stage search (same runner, same budgets).
  4. Runs the variant refits (ga_variant_refits) for stages 4/5 unless
     GA_SHARD_VARIANTS=0 / --no-variants.
  5. Saves convergence (plots + ga_reports CSVs), variant leaderboard
     (variants/leaderboard.{csv,html} + winner.json + winner specs),
     fitted variant models, and shard_info.json.

`merge_ga_shards.py` later unions every shard dir into merged/ga_results.pkl.
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

import compat.numpy_shim  # noqa: F401  (NumPy 2.0 shim, first import)

import warnings
warnings.filterwarnings("ignore",
                        message=r"JAX plugin .* version .* is installed.*",
                        category=RuntimeWarning)

import pipeline_logger as pl
from pipeline_logger import (
    load_data,
    _save_ga_checkpoint,
    _write_stage_checkpoint_report,
    _format_ga_checkpoint_summary,
    _ga_save_stage_coef_html_from_best,
    logger,
)
from ga_hypothesis_testing import AllStagesResult
import _run_ga_single_stage as _single


def _parse_args():
    p = argparse.ArgumentParser(description="Run one GA shard (stage, segment)")
    p.add_argument("--stage-id", type=int, required=True,
                   choices=[0, 1, 2, 3, 4, 5, 51])
    p.add_argument("--segment", type=str, required=True,
                   help="Shard segment key, e.g. work/edu/nonmand/start/tdd... "
                        "Must be a member of ga_shard.STAGE_SEGMENTS[stage].")
    p.add_argument("--run-root", default=os.path.join("runs", "ga_parallel"))
    p.add_argument("--run-tag", required=True)
    p.add_argument("--no-variants", action="store_true",
                   help="Skip the model-family variant shootout.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    stage_id = int(args.stage_id)
    segment = str(args.segment).strip().lower()
    t0 = time.time()

    import ga_shard as _shard
    valid = _shard.STAGE_SEGMENTS.get(stage_id, ["all"])
    if segment not in valid:
        print(f"FATAL: segment '{segment}' not valid for stage {stage_id} "
              f"(valid: {valid})")
        return 2

    stage_dir = os.path.join(args.run_root, args.run_tag, f"stage{stage_id}")
    shard_dir = os.path.join(stage_dir, "shards", segment)
    os.makedirs(os.path.join(shard_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(shard_dir, "ga_reports"), exist_ok=True)
    os.makedirs(os.path.join(shard_dir, "variants"), exist_ok=True)

    os.environ["GA_SHARD_SEGMENT"] = segment
    os.environ.setdefault("GA_CHECKPOINT_DIR", shard_dir)
    # Isolate this shard's checkpoint from siblings sharing the stage dir.
    if os.environ.get("GA_CHECKPOINT_DIR") == stage_dir:
        os.environ["GA_CHECKPOINT_DIR"] = shard_dir
    pl.OUT_DIR = shard_dir
    os.environ["PIPELINE_OUT_DIR"] = shard_dir

    logger.info("=" * 70)
    logger.info(f"SHARD start: stage={stage_id} segment={segment} tag={args.run_tag}")
    logger.info(f"shard_dir={shard_dir}")
    logger.info("=" * 70)

    all_ga = AllStagesResult()
    try:
        _, out, landuse = load_data()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Shard stage {stage_id}/{segment}: load_data failed: {exc}")
        logger.error(traceback.format_exc())
        return 1

    runner, effective_sid = _single._stage_runner(stage_id)
    if runner is None:  # stage 3 no-op under STAGE23_JOINT=1
        logger.info(f"Shard stage {stage_id}/{segment}: no-op (joint mode)")
        _write_info(shard_dir, args, segment, "noop", t0, {})
        return 0

    try:
        res = runner(out, landuse)
    except Exception as exc:  # noqa: BLE001
        _save_ga_checkpoint(all_ga, stage_id=stage_id, failed=True)
        logger.error(f"Shard stage {stage_id}/{segment} search failed: {exc}")
        logger.error(traceback.format_exc())
        try:
            partial, n_seg = _single._salvage_partial_result(stage_id)
        except Exception:
            partial, n_seg = None, 0
        if partial is None:
            _write_info(shard_dir, args, segment, "failed", t0,
                        {"error": str(exc)})
            return 1
        res = partial
        logger.warning(f"Shard salvaged partial ({n_seg} completed unit(s)))")

    # Keep only this shard's segment(s) in case the filter missed anything.
    res = _keep_shard_segments(res, stage_id, segment)
    all_ga.results[stage_id] = res
    _save_ga_checkpoint(all_ga, stage_id=stage_id, failed=False)

    try:
        import pandas as pd
        hist = getattr(res, "history", []) or []
        hist_df = pd.DataFrame(hist) if hist else None
        _write_stage_checkpoint_report(
            stage_tag=f"ga_stage{stage_id}_{segment}",
            title=f"GA Stage {stage_id} / {segment} - shard",
            summary=_format_ga_checkpoint_summary(res),
            table=hist_df)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Shard report skipped: {exc}")

    try:
        _ga_save_stage_coef_html_from_best(effective_sid, res, out, landuse=landuse)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Shard coef/HTML export failed (result still saved): {exc}")

    try:
        all_ga.export_reports(os.path.join(shard_dir, "ga_reports"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Shard export_reports failed: {exc}")

    try:
        all_ga.plot_all(save_path=os.path.join(shard_dir, "plots", "ga_all_stages.png"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Shard convergence plot skipped: {exc}")

    # Per-stage convergence plot (ga_stage<id>.png). The driver historically
    # wrote only ga_all_stages.png, so salvaged/partial shards (walltime-guard
    # or OOM kills: empty result history) ended with no per-stage plot for
    # the paper. Render from the result; if it carries no history, fall back
    # to the OOM-proof on-disk partial-history CSVs. Never raises.
    try:
        from pipeline_logger import _ga_plot_stage_result as _plot_stage_res
        _stage_png = os.path.join(shard_dir, "plots", f"ga_stage{stage_id}.png")
        _plot_stage_res(res, _stage_png)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Shard per-stage plot skipped: {exc}")
    try:
        _stage_png = os.path.join(shard_dir, "plots", f"ga_stage{stage_id}.png")
        if not os.path.isfile(_stage_png):
            import glob as _glob
            _partials = sorted(_glob.glob(os.path.join(shard_dir, "ga_history_partial_*.csv")))
            _drew = 0
            if _partials:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import pandas as _pd
                _fig, _ax = plt.subplots(figsize=(9, 4.5))
                for _pp in _partials:
                    try:
                        _df = _pd.read_csv(_pp)
                        if "best_so_far" in _df.columns and "generation" in _df.columns \
                           and len(_df) > 1:
                            _bs = pd.to_numeric(_df["best_so_far"], errors="coerce").to_numpy()
                            _gg = pd.to_numeric(_df["generation"], errors="coerce").to_numpy()
                            _ok = np.isfinite(_bs) & np.isfinite(_gg)
                            if _ok.sum() > 1:
                                _neg = bool(np.nanmedian(_bs[_ok]) < 0)
                                _ax.plot(_gg[_ok], -_bs[_ok] if _neg else _bs[_ok],
                                         marker="o", ms=3,
                                         label=os.path.basename(_pp)[:44])
                                _drew += 1
                    except Exception:
                        continue
                if _drew:
                    _ax.set_xlabel("generation")
                    _ax.set_ylabel("BIC/n (lower = better)")
                    _ax.set_title(f"GA convergence (partial salvage) — stage {stage_id}/{segment}")
                    _ax.legend(fontsize=7)
                    _ax.grid(alpha=0.3)
                    _fig.tight_layout()
                    _fig.savefig(_stage_png, dpi=120)
                    logger.info(f"Shard per-stage salvage plot → {_stage_png}")
                plt.close(_fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Shard per-stage salvage plot skipped: {exc}")

    # ---- Model-family variant shootout (stages 4/5) ---------------------
    variant_rows = []
    do_variants = (not args.no_variants
                   and os.environ.get("GA_SHARD_VARIANTS", "1").strip() != "0"
                   and stage_id in (4, 5))
    if do_variants:
        try:
            variant_rows = _run_variants(stage_id, segment, res, out, landuse,
                                         os.path.join(shard_dir, "variants"))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Shard variant shootout failed (search result kept): {exc}")
            logger.error(traceback.format_exc(limit=5))

    # ---- Trip-level (Stage 5b) estimation models -------------------------
    # The search above covers tour-level tour mode/destination (stage 5).
    # Trip-level stop-destination + trip-mode models (stage 5b) are estimated
    # here so the shard also archives trip models with terms
    # (stage5b/stage5b_models.pkl + HTML reports). Env-gated
    # (GA_SHARD_STAGE5B=1) as it roughly doubles shard cost; failures never
    # break the shard (tour results are already saved above). Runs over all
    # trip purposes (per-purpose models inside), not just this segment.
    if stage_id == 5 and os.environ.get("GA_SHARD_STAGE5B", "0").strip() == "1":
        try:
            from stage5b_trip_destination import run_stage5b_pipeline as _run_s5b
            _s5b_trips = out.get("trips") if isinstance(out, dict) else None
            _s5b_persons = out.get("persons") if isinstance(out, dict) else None
            _s5b_skims_path = (getattr(pl, "PIPELINE_SKIMS_PATH", None)
                               or os.environ.get("PIPELINE_SKIMS_PATH", ""))
            _s5b_dir = os.path.join(shard_dir, "stage5b")
            os.makedirs(_s5b_dir, exist_ok=True)
            if _s5b_trips is None or _s5b_persons is None or not _s5b_skims_path:
                logger.warning("Shard stage5b skipped: missing trips/persons/skims_path")
            else:
                _s5b_res = _run_s5b(trips=_s5b_trips, persons=_s5b_persons,
                                    skims_path=_s5b_skims_path, landuse=landuse,
                                    output_dir=_s5b_dir, validate=True)
                logger.info(f"Shard stage5b done: "
                            f"{len(_s5b_res.get('dest_models', {}))} dest + "
                            f"{len(_s5b_res.get('mode_models', {}))} mode trip models -> {_s5b_dir}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Shard stage5b failed (tour results kept): {exc}")

    _write_info(shard_dir, args, segment, "done", t0,
                {"best_fitness": float(getattr(res, "best_fitness", float("nan"))),
                 "n_variant_rows": len(variant_rows)})
    logger.info(f"SHARD done: stage={stage_id} segment={segment} "
                f"elapsed={(time.time()-t0)/60:.1f}min -> {shard_dir}")
    return 0


def _keep_shard_segments(res, stage_id: int, segment: str):
    """Drop sibling segments if the filter let any through (safety net)."""
    try:
        seg_results = (res.metadata or {}).get("segment_results")
        if not seg_results or stage_id in (0,):
            return res
        import ga_shard as _shard
        aliases = _shard.SEGMENT_ALIASES.get(segment, [segment])
        keep = {}
        for label, sub in seg_results.items():
            ll = str(label).lower()
            if any(a in ll for a in aliases) or segment == "all":
                keep[label] = sub
        if keep and len(keep) != len(seg_results):
            res.metadata["segment_results"] = keep
            logger.info(f"Shard kept {len(keep)}/{len(seg_results)} segment result(s)")
    except Exception:
        pass
    return res


def _run_variants(stage_id, segment, res, out, landuse, variants_dir):
    import ga_variant_refits as _vr
    feats = res.best_features
    if stage_id == 4:
        key = segment if isinstance(feats, dict) and segment in feats else None
        if key is None and isinstance(feats, dict) and feats:
            # label-keyed fallback (e.g. 'Work' vs 'work')
            want = segment.lower()
            for k in feats:
                if str(k).lower() == want or want in str(k).lower():
                    key = k
                    break
        purp_feats = list((feats.get(key) or []) if key else [])
        logger.info(f"Shard S4/{segment}: variant shootout on {len(purp_feats)} feats")
        return _vr.run_s4_variants(segment, purp_feats, out.get("trips"),
                                   out.get("tours"), out.get("persons"),
                                   variants_dir)
    if stage_id == 5:
        import ga_shard as _shard
        import ga_variant_refits as _vr
        fmap = feats if isinstance(feats, dict) else {}
        mode_vars = list(fmap.get("mode", []))
        dest_vars = list(fmap.get("dest", []))
        meta = {}
        try:
            for _lbl, _sub in ((res.metadata or {}).get("segment_results", {}) or {}).items():
                md = getattr(_sub, "metadata", {}) or {}
                if md.get("distribution_policy") or "gse_selected" in md:
                    meta.update(md)
        except Exception:
            pass
        rows = []
        skims = None
        try:
            tours, persons, skims = _s5_search_frames(out)
            rows += _vr.run_s5mode_variants(
                segment, [1] * len(mode_vars), mode_vars, tours, persons,
                landuse, skims, variants_dir,
                distribution_policy=meta.get("distribution_policy"),
                search_metadata=meta)
            # Dest context (logsums/size/acdc) rebuilt from the winning mode
            # spec, mirroring the worker prep; guarded — failure yields
            # non-converged rows, never a crash.
            try:
                ctx = _s5_dest_context(segment, mode_vars, tours, persons,
                                       landuse, skims)
                rows += _vr.run_s5dest_variants(
                    segment, [1] * len(dest_vars), dest_vars, ctx["seg_tours"],
                    persons, ctx["lu_st"], skims, ctx["ls"], ctx["acdc"],
                    variants_dir, warm_start=ctx.get("warm_start"))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Shard S5/{segment}: dest context rebuild failed "
                               f"({exc!r}) — dest variants recorded as not-run")
                for v in _shard.s5_dest_variants():
                    rows.append({"stage": 5, "segment": segment, "slot": "s5dest",
                                 "variant": v,
                                 "group": _shard.variant_group("s5dest", v),
                                 "primary_group": _shard.PRIMARY_GROUP["s5dest"],
                                 "converged": 0, "ll": "", "k": "", "n": "",
                                 "bic_per_obs": "", "winner_in_group": 0,
                                 "overall_pick": 0,
                                 "note": f"context rebuild failed: {exc!r}"})
            # Single combined write (mode + dest rows together — the variant
            # functions deliberately do not write, so neither family clobbers
            # the other).
            try:
                _shard.write_leaderboard(rows, variants_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Shard S5/{segment}: leaderboard write failed ({exc!r})")
            finally:
                try:
                    if skims is not None:
                        skims.close()
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Shard S5/{segment}: frames/variant setup failed ({exc!r})")
            try:
                if skims is not None:
                    skims.close()
            except Exception:
                pass
        return rows
    return []


def _s5_search_frames(out):
    """Prepped (tours, persons, skims-handle) the S5 fitness fns expect."""
    import pandas as pd
    from stage5_destination_choice import (
        _prep_hhts_tours, _ensure_person_behavioural_cols)
    persons = _ensure_person_behavioural_cols(out["persons"])
    tours_src = out.get("tours")
    _tours_input = tours_src if tours_src is not None else out["trips"]
    tours = _prep_hhts_tours(_tours_input, persons)
    if "ptype" in tours.columns:
        _dummies = pd.get_dummies(tours["ptype"], prefix="ptype")
        for _c in _dummies.columns:
            if _c not in tours.columns:
                tours[_c] = _dummies[_c].astype(int)
    if "is_female" not in tours.columns and "sexp" in tours.columns:
        tours["is_female"] = (tours["sexp"] == 2).astype(int)
    skims_path = getattr(pl, "PIPELINE_SKIMS_PATH", None) or \
        os.environ.get("PIPELINE_SKIMS_PATH", "")
    import openmatrix as omx
    skims = omx.open_file(skims_path, "r") if skims_path else None
    return tours, persons, skims


def _s5_dest_context(seg, mode_vars, tours, persons, landuse, skims):
    """Rebuild the worker's dest context for variant scoring.

    Mirrors _ga_stage5_segment_worker prep: mode tree -> estimate best mode
    spec -> logsums -> size term -> acdc terms -> starter warm start.
    Returns dict(seg_tours, lu_st, ls, acdc, warm_start).
    """
    from stage5_destination_choice import (
        _mode_tree, _build_mode_model, _estimate, _logsums,
        _attach_size_term, _attach_acdc_terms, _skim_flat, _filter_seg,
        estimate_dest_starter, dest_starter_terms)
    from ga_hypothesis_testing import _ga_sample_dest_alts
    try:
        skim_df = _skim_flat(skims)
    except Exception:
        skim_df = pd.DataFrame()
    seg_tours = _filter_seg(tours, seg)
    dt_m = _mode_tree(seg_tours, persons, landuse, skims)
    mode_mdl = _build_mode_model(dt_m, seg, custom_mode_vars=list(mode_vars),
                                 avail_vars_hint=list(seg_tours.columns))
    mode_mdl = _estimate(mode_mdl)
    ls = _logsums(mode_mdl, dt_m, skims=skims, verbose=False)
    lu_st = _attach_size_term(landuse, seg)
    lu_st, acdc = _attach_acdc_terms(lu_st, seg_tours, skim_df, seg)
    warm_start = {}
    try:
        samp = _ga_sample_dest_alts(acdc, lu_st, ls)
        from stage5_destination_choice import _dest_tree
        dt_d = _dest_tree(acdc, persons, lu_st, skims, ls, sampled_alts=samp)
        st_res = estimate_dest_starter(dt_d, seg)
        if st_res.get("converged"):
            warm_start = dict(st_res.get("coefs", {}))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Shard S5/{seg}: starter warm start unavailable ({exc!r})")
    return {"seg_tours": seg_tours, "lu_st": lu_st, "ls": ls, "acdc": acdc,
            "warm_start": warm_start or None}


def _write_info(shard_dir, args, segment, status, t0, extra):
    try:
        info = {"stage": args.stage_id, "segment": segment,
                "run_root": args.run_root, "run_tag": args.run_tag,
                "status": status, "elapsed_min": (time.time() - t0) / 60.0,
                "shard_dir": shard_dir}
        info.update(extra or {})
        with open(os.path.join(shard_dir, "shard_info.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
