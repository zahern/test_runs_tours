"""
_run_predict_phase.py
=====================
Phase-sharded replacement for the monolithic ``pipeline_logger.py ga_staged``
prediction run.

One phase per PBS job, chained with ``afterok`` so each job's peak memory is
its own working set (plus one checkpoint load and one checkpoint write)
instead of the whole synth-load + estimate + predict + plan pipeline in a
single 190 GB process.

Phases (env ``PREDICT_PHASE``)::

    prep      load HHTS+landuse, GA pkl, ALL-zones synth, enrich, WFH,
              mobility, CDAP, tour sequence -> checkpoints
              ``_phases/travellers.pkl`` + ``_phases/synth_full.pkl``
    s1        Stage 1 activity-frequency prediction
    s23       Stages 2+3 day/activity duration prediction
    s4        Stage 4 schedule placement
    s5        Stage 5 mode + destination (GA specs, baseline prefit, warm
              start) -> ``stage5_output`` checkpoint
    finalize  Stage 5b + Stage 6 plan + validation summary (plan checkpoint)
    plots     spatial/plan plots (small job; no synth load)

The phase bodies mirror ``ga_hypothesis_testing.apply_ga_results_to_pipeline``
and ``pipeline_logger.GA_STAGED`` argument-for-argument; only the checkpoint
boundaries are new.

Checkpoints live in ``<out_dir>/_phases/`` and are written as parquet
(compressed; pickle fallback) so phases reload less than a full pickle dict
would cost. The all-zones synth frame is never persisted: only ``travellers``
is needed downstream, and ``spatial_plots`` ignores its ``synth`` argument.

Env::

    PREDICT_PHASE      required (prep|s1|s23|s4|s5|finalize|plots)
    PIPELINE_RUN_DIR   required output dir (exported as PIPELINE_OUT_DIR)
    GA_STAGED_GA_DIR   merged shard dir containing ga_results.pkl
    GA_STAGED_REUSE_ESTIMATOR=1 expected (no GA is run here)
    plus the usual PIPELINE_* / NEW_POP* / GA_* knobs read by pipeline_logger.
"""

from __future__ import annotations

import gc
import logging
import os
import pickle
import sys
import time

import compat.numpy_shim  # noqa: F401  (NumPy 2.0 shim, first import)

import pandas as pd

PHASE = os.environ.get("PREDICT_PHASE", "").strip().lower()
OUT_DIR = os.environ.get("PIPELINE_RUN_DIR", "").strip() or os.environ.get(
    "PIPELINE_OUT_DIR", ""
).strip()
PHASES = ("prep", "s1", "s23", "s4", "s5", "finalize", "plots")

if PHASE not in PHASES:
    raise SystemExit(
        f"PREDICT_PHASE must be one of {PHASES}; got {PHASE!r}"
    )
if not OUT_DIR:
    raise SystemExit("PIPELINE_RUN_DIR (or PIPELINE_OUT_DIR) is required")

os.makedirs(OUT_DIR, exist_ok=True)
os.environ["PIPELINE_OUT_DIR"] = OUT_DIR
CKPT_DIR = os.path.join(OUT_DIR, "_phases")
os.makedirs(CKPT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUT_DIR, f"predict_{PHASE}.log"), mode="w"),
    ],
)
logger = logging.getLogger("pipeline")

import pipeline_logger as pl  # noqa: E402

pl.OUT_DIR = OUT_DIR  # load_data() writes accessibility plots under OUT_DIR

from ga_hypothesis_testing import (  # noqa: E402
    AllStagesResult,
    _ga_save_stage_coef_csv,
    _save_model_bundle,
)


def _ckpt(name: str) -> str:
    return os.path.join(CKPT_DIR, name)


def _save_pkl(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"  checkpoint -> {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def _load_pkl(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"checkpoint missing: {path} (previous phase not run?)"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _save_df(base: str, df: pd.DataFrame) -> None:
    """Checkpoint a DataFrame as parquet (compressed, ~3-5x smaller than
    pickle and no per-object overhead), falling back to pickle when pyarrow
    is unavailable."""
    try:
        path = base + ".parquet"
        tmp = path + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  parquet checkpoint failed ({exc}); using pickle")
        path = base + ".pkl"
        _save_pkl(path, df)
        return
    print(f"  checkpoint -> {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def _load_df(base: str) -> pd.DataFrame:
    pq = base + ".parquet"
    pk = base + ".pkl"
    if os.path.isfile(pq):
        return pd.read_parquet(pq)
    if os.path.isfile(pk):
        return _load_pkl(pk)
    raise FileNotFoundError(
        f"checkpoint missing: {pq} / {pk} (previous phase not run?)"
    )


def _load_ga() -> AllStagesResult:
    ga_dir = os.environ.get("GA_STAGED_GA_DIR", "").strip()
    if not ga_dir:
        raise SystemExit("GA_STAGED_GA_DIR is required")
    pkl = os.path.join(ga_dir, "ga_results.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(f"ga_results.pkl not found: {pkl}")
    with open(pkl, "rb") as fh:
        ga = pickle.load(fh)
    if not isinstance(ga, AllStagesResult):
        raise TypeError(f"not an AllStagesResult: {pkl}")
    print(f"  ga_results.pkl <- {pkl} (stages {sorted(ga.results)})")
    return ga


def _write_selected_features_summary(ga: AllStagesResult) -> None:
    try:
        import json as _json

        summary = {}
        for sid, res in sorted(ga.results.items()):
            if res is None:
                continue
            bf = getattr(res, "best_features", None)
            ff = getattr(res, "best_fitness", float("nan"))
            fm = getattr(res, "fitness_metric", "")
            if isinstance(bf, dict):
                summary[f"stage{sid}"] = {
                    "features": {
                        str(k): list(v) if isinstance(v, (list, tuple, set)) else v
                        for k, v in bf.items()
                    },
                    "best_fitness": float(ff),
                    "fitness_metric": str(fm),
                }
            elif isinstance(bf, (list, tuple)):
                summary[f"stage{sid}"] = {
                    "features": list(bf),
                    "best_fitness": float(ff),
                    "fitness_metric": str(fm),
                }
        path = os.path.join(OUT_DIR, "selected_features_summary.json")
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(summary, fh, indent=2, default=str)
        print(f"  Final selected features -> {path}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  selected_features_summary.json write failed: {exc}")


def _stage5_free_vars(model) -> int:
    """Count free non-constant parameters (mirrors apply_ga_results_to_pipeline)."""
    if model is None:
        return 0
    pf = getattr(model, "_final_pf", None) or getattr(model, "pf", None)
    if pf is None or not hasattr(pf, "index"):
        return 0
    try:
        holdfast = (
            pf["holdfast"].fillna(False).astype(bool)
            if "holdfast" in pf.columns
            else pd.Series(False, index=pf.index)
        )
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for pname, is_held in holdfast.items():
        if bool(is_held):
            continue
        nm = str(pname)
        if nm.startswith("ASC_") or nm.startswith("c_ASC"):
            continue
        if nm in ("log_correction", "c_size_term") or nm.startswith("Mu_"):
            continue
        n += 1
    return n


def _invalid_stage5_segments(models) -> list:
    expected = ("work", "edu", "nonmand")
    bad = []
    if not isinstance(models, dict):
        return list(expected)
    for seg in expected:
        seg_res = models.get(seg)
        if not isinstance(seg_res, dict):
            bad.append(seg)
            continue
        for mtype in ("mode_model", "dest_model"):
            mdl = seg_res.get(mtype)
            if mdl is None:
                bad.append(seg)
                break
            if _stage5_free_vars(mdl) < 1:
                logger.warning(
                    f"GA Stage 5 segment '{seg}' {mtype} has no free "
                    "non-constant variables (degenerate fit) - "
                    "will retry with baseline spec."
                )
                bad.append(seg)
                break
    return bad


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_prep() -> None:
    ga = _load_ga()
    _write_selected_features_summary(ga)
    _, out, landuse = pl.load_data()

    synth = pl.load_synth_all_zones(pl.SYNTH_DIR, landuse=landuse)
    synth = pl.usual_locations(synth, landuse, hhts_tours=out.get("tours"))
    synth = pl.wfh_assign(synth, out)
    synth = pl.mobility_assign(synth, out)
    _s0_res = ga.results.get(0)
    synth = pl.cdap(out, synth,
                    feature_override=getattr(_s0_res, "best_features", None))
    synth = pl.tour_sequence_assign(synth, out)
    travellers = synth[synth["CDAP"].isin(["M", "N"])].copy()
    print(f"  CDAP filter: {len(travellers):,}/{len(synth):,} persons are travellers")

    # Only the travellers frame is checkpointed: spatial_plots(plan, synth, ...)
    # never reads its synth argument (verified in pipeline_logger.spatial_plots),
    # so persisting the full enriched synth would be pure disk/memory waste.
    _save_df(_ckpt("travellers"), travellers)
    del synth
    gc.collect()


def phase_s1() -> None:
    ga = _load_ga()
    _, out, landuse = pl.load_data()
    travellers = _load_df(_ckpt("travellers"))
    from stage1_activity_frequency import run_stage1

    s1_res = ga.results.get(1)
    print("\n[1/5] Stage 1   activity frequency prediction...")
    synth_s1, s1_models = run_stage1(
        out["tours"], out["persons"], travellers,
        feature_override=s1_res.best_features if s1_res else None,
    )
    del travellers
    synth_s1.to_csv(os.path.join(OUT_DIR, "stage1_output.csv"), index=False)
    _save_df(_ckpt("stage1_output"), synth_s1)
    _save_model_bundle(s1_models, OUT_DIR, os.path.join(OUT_DIR, "models"),
                       "stage1_models.pkl")
    _ga_save_stage_coef_csv(s1_models, "stage1", OUT_DIR)
    del synth_s1
    gc.collect()


def phase_s23() -> None:
    ga = _load_ga()
    if ga.results.get(23) is not None:
        logger.warning(
            "ga_results contains a joint Stage 2+3 model (stage 23); the phased "
            "driver runs the independent Stage 2/3 path instead."
        )
    _, out, landuse = pl.load_data()
    synth_s1 = _load_df(_ckpt("stage1_output"))
    from stage2_day_duration import run_stage2
    from stage3_activity_duration import run_stage3

    s2_res = ga.results.get(2)
    s3_res = ga.results.get(3)
    print("\n[2/5] Stage 2   day duration prediction...")
    synth_s2, s2_models = run_stage2(
        out["tours"], out["persons"], synth_s1,
        feature_override=s2_res.best_features if s2_res else None,
    )
    del synth_s1
    synth_s2.to_csv(os.path.join(OUT_DIR, "stage2_output.csv"), index=False)
    _save_df(_ckpt("stage2_output"), synth_s2)
    _save_model_bundle(s2_models, OUT_DIR, os.path.join(OUT_DIR, "models"),
                       "stage2_models.pkl")
    _ga_save_stage_coef_csv(s2_models, "stage2", OUT_DIR)

    print("\n[3/5] Stage 3   activity duration prediction...")
    activities, s3_models = run_stage3(
        out["tours"], out["persons"], synth_s2, trips=out["trips"],
        feature_override=s3_res.best_features if s3_res else None,
    )
    del synth_s2
    activities.to_csv(os.path.join(OUT_DIR, "stage3_output.csv"), index=False)
    _save_df(_ckpt("stage3_output"), activities)
    _save_model_bundle(s3_models, OUT_DIR, os.path.join(OUT_DIR, "models"),
                       "stage3_models.pkl")
    _ga_save_stage_coef_csv(s3_models, "stage3", OUT_DIR)
    del activities
    gc.collect()


def phase_s4() -> None:
    ga = _load_ga()
    _, out, landuse = pl.load_data()
    activities = _load_df(_ckpt("stage3_output"))
    from stage4_schedule_placement import run_stage4

    s4_res = ga.results.get(4)
    print("\n[4/5] Stage 4   schedule placement prediction...")
    scheduled, s4_models = run_stage4(
        out["trips"], out["tours"], out["persons"], activities,
        feature_override=s4_res.best_features if s4_res else None,
    )
    del activities
    scheduled.to_csv(os.path.join(OUT_DIR, "stage4_output.csv"), index=False)
    _save_df(_ckpt("stage4_output"), scheduled)
    _save_model_bundle(s4_models, OUT_DIR, os.path.join(OUT_DIR, "models"),
                       "stage4_models.pkl")
    _ga_save_stage_coef_csv(s4_models, "stage4", OUT_DIR)
    del scheduled
    gc.collect()


def phase_s5() -> None:
    ga = _load_ga()
    _, out, landuse = pl.load_data()
    travellers = _load_df(_ckpt("travellers"))
    scheduled = _load_df(_ckpt("stage4_output"))
    from stage5_destination_choice import run_stage5

    prev_larch = os.environ.get("GA_LARCH_ENGINE")
    prev_strict = os.environ.get("STAGE5_STRICT_ESTIMATION")
    if os.name != "nt":
        os.environ["GA_LARCH_ENGINE"] = "jax"
    os.environ["STAGE5_STRICT_ESTIMATION"] = "1"

    try:
        s5_res = ga.results.get(5)
        bf = getattr(s5_res, "best_features", None) if s5_res else None
        if isinstance(bf, dict):
            s5_mode_feats = bf.get("mode") or bf.get("work_mode")
            s5_dest_feats = bf.get("dest") or bf.get("work_dest")
        else:
            s5_mode_feats = None
            s5_dest_feats = None

        baseline_warm = {}
        if os.environ.get("GA_STAGE5_PREFIT_BASELINE", "1").strip() != "0":
            logger.info("Stage 5 prefit: estimating baseline model (no GA overrides).")
            _, base_models = run_stage5(
                out["trips"], travellers, scheduled,
                skims=pl.PIPELINE_SKIMS_PATH,
                landuse=landuse,
                tours=out["tours"],
                seed=42,
                mode_feature_override=None,
                dest_feature_override=None,
                save_path=os.path.join(OUT_DIR, "stage5_models_prefit.pkl"),
            )
            base_bad = _invalid_stage5_segments(base_models)
            if base_bad:
                raise RuntimeError(
                    "Stage 5 baseline prefit failed (incomplete models for "
                    f"segments: {base_bad})."
                )
            for seg_res in base_models.values():
                if not isinstance(seg_res, dict):
                    continue
                mc = seg_res.get("mode_coef")
                dc = seg_res.get("dest_coef")
                if isinstance(mc, dict):
                    baseline_warm.update(mc)
                if isinstance(dc, dict):
                    baseline_warm.update(dc)
            logger.info(
                f"Stage 5 prefit passed; warm-start with {len(baseline_warm)} coefficients."
            )

        print("\n[5/5] Stage 5   destination + mode choice (zone-by-zone)...")
        scheduled_dest, s5_models = run_stage5(
            out["trips"], travellers, scheduled,
            skims=pl.PIPELINE_SKIMS_PATH,
            landuse=landuse,
            tours=out["tours"],
            seed=42,
            mode_feature_override=s5_mode_feats,
            dest_feature_override=s5_dest_feats,
            coef_warm_start=baseline_warm,
            save_path=os.path.join(OUT_DIR, "stage5_models.pkl"),
        )
        bad = _invalid_stage5_segments(s5_models)
        if bad and os.environ.get("GA_STAGE5_FALLBACK_TO_MAIN", "0").strip() != "0":
            logger.warning(
                "Stage 5 GA models incomplete; retrying baseline spec "
                f"(segments: {bad})."
            )
            scheduled_dest, s5_models = run_stage5(
                out["trips"], travellers, scheduled,
                skims=pl.PIPELINE_SKIMS_PATH,
                landuse=landuse,
                tours=out["tours"],
                seed=42,
                mode_feature_override=None,
                dest_feature_override=None,
                save_path=os.path.join(OUT_DIR, "stage5_models.pkl"),
            )
            bad = _invalid_stage5_segments(s5_models)
        if bad:
            raise RuntimeError(
                f"Stage 5 models incomplete for segments {bad}; "
                "set GA_STAGE5_FALLBACK_TO_MAIN=1 to auto-retry baseline."
            )
    finally:
        if prev_larch is None:
            os.environ.pop("GA_LARCH_ENGINE", None)
        else:
            os.environ["GA_LARCH_ENGINE"] = prev_larch
        if prev_strict is None:
            os.environ.pop("STAGE5_STRICT_ESTIMATION", None)
        else:
            os.environ["STAGE5_STRICT_ESTIMATION"] = prev_strict

    scheduled_dest.to_csv(os.path.join(OUT_DIR, "stage5_output.csv"), index=False)
    scheduled_dest.to_csv(os.path.join(OUT_DIR, "stage6_output.csv"), index=False)
    _save_df(_ckpt("stage5_output"), scheduled_dest)
    _save_model_bundle(s5_models, OUT_DIR, os.path.join(OUT_DIR, "models"),
                       "stage5_models.pkl")

    try:
        from stage5_destination_choice import _save_coefficients as _s5_save_coeff

        _s5_save_coeff(s5_models, os.path.join(OUT_DIR, "stage5_models_coefficients.csv"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  Stage 5 coefficient save failed: {exc}")
    _ga_save_stage_coef_csv(s5_models, "stage5", OUT_DIR)


def phase_finalize() -> None:
    ga = _load_ga()
    _, out, landuse = pl.load_data()
    scheduled_dest = _load_df(_ckpt("stage5_output"))
    s5_models = _load_pkl(os.path.join(OUT_DIR, "stage5_models.pkl"))

    # Stage-51 trip-destination GA winners -> conditional stop-destination MNL.
    _g_tripdest = None
    try:
        from stage5b_trip_destination import GA_CANDIDATE_VARS
        _s51 = ga.results.get(51)
        _feats = list(getattr(_s51, "best_features", {}).get("dest", [])) if _s51 else []
        if _feats:
            _g_tripdest = [1 if v in _feats else 0 for v in GA_CANDIDATE_VARS]
            print(f"  Stage 51 trip-dest features -> {_feats}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Stage 51 trip-dest gene vector unavailable: {exc}")

    pl._ga_post_estimation_report(s5_models, out, OUT_DIR)

    import openmatrix as omx
    from stage5_destination_choice import _skim_flat, _skim_lookup

    skims = omx.open_file(pl.PIPELINE_SKIMS_PATH, "r")
    sk = _skim_lookup(_skim_flat(skims))
    try:
        try:
            scheduled_dest = pl.stage5b(out, scheduled_dest, landuse,
                                        g_tripdest=_g_tripdest)
        except Exception as exc:  # noqa: BLE001
            import traceback as _tb

            logger.warning(f"Stage 5b failed (non-fatal): {exc!r}\n{_tb.format_exc(limit=8)}")
        plan, _ = pl.stage6(out, scheduled_dest, landuse,
                            stage5_models=s5_models, skims=skims, skim_lookup=sk)
        pl.plot(plan)
        _save_df(_ckpt("plan"), plan)
    finally:
        try:
            skims.close()
        except Exception:  # noqa: BLE001
            pass

    # -- validation summary (mirrors GA_STAGED) ----------------------------
    print("\n" + "=" * 60)
    print("GA STAGED - VALIDATION SUMMARY")
    print("=" * 60)
    val_rows = []
    try:
        trips_out = (
            plan[plan.get("row_type", pd.Series()) == "trip"]
            if "row_type" in plan.columns
            else plan
        )
        if "mode_name" in trips_out.columns:
            mode_share = trips_out["mode_name"].value_counts(normalize=True) * 100
            print("\n  Mode share (predicted):")
            for mode, share in mode_share.sort_values(ascending=False).items():
                print(f"    {mode:<12} {share:5.1f}%")
                val_rows.append({"metric": f"mode_share_{mode}",
                                 "value": round(share, 2), "unit": "%"})
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  Mode share validation failed: {exc}")

    try:
        from stage6_plan_finalisation import compute_zone_validation_metrics

        zv = compute_zone_validation_metrics(plan, verbose=True)
        for k, v in zv.items():
            if k != "catchall_by_mode" and isinstance(v, (int, float)):
                val_rows.append({"metric": f"zone_{k}", "value": v, "unit": ""})
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  Zone validation failed: {exc}")

    try:
        n_trips = int((plan["row_type"] == "trip").sum()) if "row_type" in plan.columns else len(plan)
        n_acts = int((plan["row_type"] == "activity").sum()) if "row_type" in plan.columns else 0
        n_pax = plan["person_id"].nunique() if "person_id" in plan.columns else 0
        trips_per = n_trips / max(n_pax, 1)
        print(f"\n  Trips: {n_trips:,}  Activities: {n_acts:,}  Persons: {n_pax:,}")
        print(f"  Avg trips/person: {trips_per:.2f}")
        val_rows += [
            {"metric": "n_trips", "value": n_trips, "unit": "trips"},
            {"metric": "n_activities", "value": n_acts, "unit": "activities"},
            {"metric": "n_persons", "value": n_pax, "unit": "persons"},
            {"metric": "trips_per_person", "value": round(trips_per, 3),
             "unit": "trips/person"},
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  Trip/activity summary failed: {exc}")

    try:
        tt_col = next((c for c in ["travtime", "tt_avg"] if c in plan.columns), None)
        if tt_col:
            trip_rows = plan[plan["row_type"] == "trip"] if "row_type" in plan.columns else plan
            tt = pd.to_numeric(trip_rows[tt_col], errors="coerce").dropna()
            if tt.median() > 200:
                tt = tt / 60.0
            print(f"\n  Travel time (trips): mean={tt.mean():.1f}min  "
                  f"median={tt.median():.1f}min  p90={tt.quantile(0.9):.1f}min")
            val_rows += [
                {"metric": "travtime_mean_min", "value": round(float(tt.mean()), 2),
                 "unit": "min"},
                {"metric": "travtime_median_min", "value": round(float(tt.median()), 2),
                 "unit": "min"},
                {"metric": "travtime_p90_min",
                 "value": round(float(tt.quantile(0.9)), 2), "unit": "min"},
            ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"  Travel time summary failed: {exc}")

    if val_rows:
        val_path = os.path.join(OUT_DIR, "ga_staged_validation.csv")
        pd.DataFrame(val_rows).to_csv(val_path, index=False)
        print(f"\n  Validation saved -> {val_path}")

    print("=" * 60)
    logger.info("GA staged predict phase complete.")


def phase_plots() -> None:
    """Spatial/plan plots in their own small job.

    ``spatial_plots(plan, synth, landuse)`` never reads its ``synth`` argument
    (verified in ``pipeline_logger.spatial_plots``), so this phase only needs
    the checkpointed plan + landuse — the full synth frame is never loaded.
    """
    _, out, landuse = pl.load_data()
    plan = _load_df(_ckpt("plan"))
    pl.spatial_plots(plan, None, landuse)
    del plan
    gc.collect()


PHASE_FUNCS = {
    "prep": phase_prep,
    "s1": phase_s1,
    "s23": phase_s23,
    "s4": phase_s4,
    "s5": phase_s5,
    "finalize": phase_finalize,
    "plots": phase_plots,
}


def main() -> int:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info(f"PREDICT PHASE: {PHASE}   out_dir={OUT_DIR}")
    logger.info("=" * 70)
    PHASE_FUNCS[PHASE]()
    logger.info(f"PHASE {PHASE} done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
