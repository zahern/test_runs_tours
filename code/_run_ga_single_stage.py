"""
_run_ga_single_stage.py
Run exactly one GA stage (1-5) and save a one-stage ga_results.pkl artifact.

Designed for PBS fan-out where stages run in parallel jobs and are merged later.

Special behaviour when STAGE23_JOINT=1:
  - stage-id 2 → runs ga_stage23_joint (joint budget-constrained Stage 2+3)
  - stage-id 3 → skips gracefully (work already done by the stage-2 job)
"""

# NumPy 2.0 compatibility shim - MUST BE FIRST IMPORT
import compat.numpy_shim

# Suppress the JAX plugin version-mismatch RuntimeWarning before any JAX import.
# This fires when jax_cuda13_plugin and jaxlib have a minor version skew; the
# warning is cosmetic (the plugin is simply not used) and floods .o log files.
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"JAX plugin .* version .* is installed.*not compatible",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"JAX plugin .* not compatible with the installed jaxlib",
    category=RuntimeWarning,
    module=r"jaxlib.*",
)

import argparse
import os
import traceback

_STAGE23_JOINT: bool = os.environ.get("STAGE23_JOINT", "0").strip() == "1"

import pipeline_logger as pl
from pipeline_logger import (
    load_data,
    ga_stage0,
    ga_stage1,
    ga_stage2,
    ga_stage3,
    ga_stage4,
    ga_stage5,
    ga_stage51,
    ga_stage23_joint,
    _save_ga_checkpoint,
    _write_stage_checkpoint_report,
    _format_ga_checkpoint_summary,
    _ga_save_stage_coef_html_from_best,
    logger,
)
from ga_hypothesis_testing import AllStagesResult


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one GA stage and persist ga_results.pkl")
    p.add_argument("--stage-id", type=int, required=True,
                   choices=[0, 1, 2, 3, 4, 5, 51])
    p.add_argument("--run-root", default=os.path.join("runs", "ga_parallel"))
    p.add_argument("--run-tag", required=True)
    return p.parse_args()


def _stage_runner(stage_id: int):
    """Return (runner_fn, effective_sid) for the given stage.

    When STAGE23_JOINT=1:
      stage 2 → ga_stage23_joint; effective_sid=23 so HTML uses the joint code path
      stage 3 → None (no-op; joint work is in the stage-2 job)
    """
    if _STAGE23_JOINT:
        if stage_id == 2:
            return lambda out, landuse: ga_stage23_joint(out), 23
        if stage_id == 3:
            return None, 3  # skip

    mapping = {
        0: lambda out, landuse: ga_stage0(out),
        1: lambda out, landuse: ga_stage1(out, landuse),
        2: lambda out, landuse: ga_stage2(out),
        3: lambda out, landuse: ga_stage3(out),
        4: lambda out, landuse: ga_stage4(out),
        5: lambda out, landuse: ga_stage5(out, landuse),
        51: lambda out, landuse: ga_stage51(out, landuse),
    }
    return mapping[stage_id], stage_id


def _ckpt_candidates(stage_id: int):
    """Per-segment checkpoint files for a stage, in priority search order."""
    ck = os.environ.get("GA_CHECKPOINT_DIR", ".")
    names = {
        1: ["ga_stage1_purposes.pkl"],
        2: ["ga_stage2_targets.pkl"],
        3: ["ga_stage3_purposes.pkl"],
        5: ["ga_stage5_segments.pkl"],
    }.get(stage_id, [])
    return [os.path.join(p, n) for p in (ck, ".") for n in names]


def _load_segment_ckpt(stage_id: int):
    """Load the first existing per-segment checkpoint for this stage."""
    import pickle
    for path in _ckpt_candidates(stage_id):
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    return pickle.load(fh), path
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"salvage: could not load {path}: {exc}")
    return None, None


def _salvage_best_npz(stage_id: int):
    """Fallback salvage from best_*.npz when no per-segment checkpoint exists.

    The walltime guard can trip before the first Stage-5 segment finishes, in
    which case ga_stage5_segments.pkl is empty/absent but best_<slug>.npz
    files hold the winning genes so far. Reassemble those into a partial
    GAResult so the shard exits 0 and merge keeps the best-so-far features
    instead of reporting the stage MISSING. Returns (GAResult, n_files) or
    (None, 0).
    """
    import glob
    import numpy as np
    from ga_hypothesis_testing import GAResult, Individual
    ck = os.environ.get("GA_CHECKPOINT_DIR", ".")
    paths = sorted(glob.glob(os.path.join(ck, "best_*.npz")))
    if not paths:
        return None, 0
    mode_vars, dest_vars = [], []
    best_fit = -np.inf
    n_used = 0
    for path in paths:
        try:
            z = np.load(path, allow_pickle=True)
            genes = np.asarray(z["genes"]).ravel()
            feats = list(np.asarray(z["feat_names"], dtype=object).ravel())
            fit = float(z["fitness"]) if "fitness" in z else -np.inf
            label = str(z["label"]) if "label" in z else os.path.basename(path)
        except Exception:
            continue
        if not np.isfinite(fit) or fit <= -1e8 or not feats:
            continue
        sel = [str(f) for f, g in zip(feats, genes) if int(g) == 1]
        if not sel:
            continue
        n_used += 1
        best_fit = max(best_fit, fit)
        if "dest" in label.lower():
            for v in sel:
                if v not in dest_vars:
                    dest_vars.append(v)
        else:
            for v in sel:
                if v not in mode_vars:
                    mode_vars.append(v)
    if n_used == 0:
        return None, 0
    # Stage 51 labels carry purpose-specific dest genes in the same files;
    # for stage 5 the GAResult shape is {'mode': [...], 'dest': [...]}.
    res = GAResult(
        stage=stage_id, best_features={"mode": mode_vars, "dest": dest_vars},
        best_fitness=best_fit,
        best_individual=Individual(genes=np.zeros(0, dtype=np.int8)),
        history=[], all_candidates=[], feature_names=[],
        fitness_metric="salvaged-partial (best-so-far genes only)",
        n_restarts=0, cache_hits=0,
    )
    res.metadata["segment_results"] = {}
    return res, n_used


def _salvage_partial_result(stage_id: int):
    """
    Build a usable partial GAResult from whatever per-segment checkpoints
    exist, so a walltime/OOM-killed stage still hands the prediction chain a
    best-so-far feature set (completed segments keep their searched features;
    unfinished segments fall back to the baseline spec downstream).
    Returns (GAResult, n_segments) or (None, 0).
    """
    import numpy as np
    from ga_hypothesis_testing import GAResult, Individual, _combine_segment_feature_results

    ck, path = _load_segment_ckpt(stage_id)
    if not ck:
        # No completed-segment checkpoint: fall back to best-so-far genes so
        # a walltime-killed Stage-5 search (0/N completed) still salvages.
        try:
            return _salvage_best_npz(stage_id)
        except Exception:
            return None, 0

    try:
        if stage_id == 5:
            # ck: {seg: {mode_seg_result, dest_seg_result, best_mode_vars,
            #            best_dest_vars, ...}}
            if not ck:
                return None, 0
            mode_lists, dest_lists, seg_results = [], [], {}
            best_fit = -np.inf
            # Only finite, non-sentinel fitnesses may set best_fit. Without this
            # filter a segment carrying the -1e9 failure sentinel propagates into
            # the salvaged best_fitness, and pipeline_logger's prediction path
            # treats <=-1e8 as "invalid GA result" and DISCARDS the (valid) mode
            # feature overrides too. Mirrors the filter in _salvage_best_npz.
            for seg, entry in ck.items():
                m_res = entry.get("mode_seg_result")
                d_res = entry.get("dest_seg_result")
                if m_res is not None:
                    seg_results[f"{seg}   Mode"] = m_res
                    _mf = float(getattr(m_res, "best_fitness", -np.inf))
                    if np.isfinite(_mf) and _mf > -1e8:
                        best_fit = max(best_fit, _mf)
                if d_res is not None:
                    seg_results[f"{seg}   Dest"] = d_res
                    _df = float(getattr(d_res, "best_fitness", -np.inf))
                    if np.isfinite(_df) and _df > -1e8:
                        best_fit = max(best_fit, _df)
                if entry.get("best_mode_vars") is not None:
                    mode_lists.append(tuple(entry["best_mode_vars"]))
                if entry.get("best_dest_vars") is not None:
                    dest_lists.append(tuple(entry["best_dest_vars"]))
            if not seg_results:
                try:
                    return _salvage_best_npz(stage_id)
                except Exception:
                    return None, 0
            # Every segment carried the sentinel (or none finished) -- publishing
            # best_fitness=-inf here would surface as the -1e9 sentinel downstream.
            if not np.isfinite(best_fit) or best_fit <= -1e8:
                try:
                    _npz_res = _salvage_best_npz(stage_id)
                    if _npz_res is not None:
                        return _npz_res
                except Exception:
                    pass
                print("  WARNING: salvaged stage-5 checkpoint carried only "
                      f"failure-sentinel fitnesses (best={best_fit}); leaving "
                      "stage MISSING rather than publishing a -1e9 best_fitness",
                      flush=True)
                return None, 0

            def _majority(lists):
                """Most common variant across segments; ties broken by the
                LONGER feature set (more information preserved)."""
                if not lists:
                    return []
                from collections import Counter
                best = max(Counter(lists).items(),
                           key=lambda kv: (kv[1], len(kv[0])))
                return list(best[0])

            mode_vars = _majority(mode_lists)
            dest_vars = _majority(dest_lists)
            res = GAResult(
                stage=5, best_features={"mode": mode_vars, "dest": dest_vars},
                best_fitness=best_fit,
                best_individual=Individual(genes=np.zeros(0, dtype=np.int8)),
                history=[], all_candidates=[], feature_names=[],
                fitness_metric="salvaged-partial (per-segment best)",
                n_restarts=0, cache_hits=0,
            )
            res.metadata["segment_results"] = seg_results
            return res, len(ck)

        # stages 1 / 2 / 3: {label: {'seg_result': GAResult, 'feats': [...]}}
        rev_maps = {
            1: "stage1", 3: "stage3",
        }
        feature_map, segment_results, n = {}, {}, 0
        best_fit = -np.inf
        for label, entry in ck.items():
            seg_result = entry.get("seg_result")
            feats = entry.get("feats")
            if seg_result is None or feats is None:
                continue
            segment_results[label] = seg_result
            n += 1
            best_fit = max(best_fit, float(getattr(seg_result, "best_fitness", -np.inf)))
            # recover the feature-map key from the label suffix
            # (labels look like 'Stage3   Work' / 'Stage2   Start-of-Day')
            suffix = str(label).split(maxsplit=1)[-1].strip()
            key = _label_to_key(stage_id, suffix)
            feature_map[key] = list(feats)
        if not feature_map:
            return None, 0
        combined = _combine_segment_feature_results(
            stage=stage_id,
            segment_results=segment_results,
            feature_map=feature_map,
            fitness_metric="salvaged-partial (completed segments only)",
            n_restarts=0,
        )
        combined.best_fitness = best_fit
        return combined, n
    except Exception as exc:  # noqa: BLE001
        logger.error(f"salvage: failed to build partial result: {exc}", exc_info=True)
        return None, 0


_LABEL_KEY_MAPS = {
    1: {"Work": "work", "Education": "edu", "Start-of-Day": "start",
        "End-of-Day": "end", "Day Duration (B)": "budget"},
    2: {"Start-of-Day": "start", "End-of-Day": "end", "Day Duration (B)": "budget"},
    3: {"Work": "work", "Education": "edu", "Escort": "escort",
        "Shopping": "shopping", "Social": "social", "Other Maintenance": "othmaint",
        "Other Discretionary": "othdiscr", "Eat": "eat", "Non-Mandatory": "nonmand",
        "Nonmand": "nonmand"},
}


def _label_to_key(stage_id: int, suffix: str) -> str:
    suffix_l = suffix.strip().lower()
    for lbl, key in _LABEL_KEY_MAPS.get(stage_id, {}).items():
        if lbl.lower() == suffix_l:
            return key
    return suffix_l.replace(" ", "_")


def main() -> int:
    args = _parse_args()
    stage_id = int(args.stage_id)

    runner, effective_sid = _stage_runner(stage_id)

    # Stage 3 is a no-op when STAGE23_JOINT=1 (joint work done by stage-2 job).
    if runner is None:
        logger.info(
            f"STAGE23_JOINT=1: stage {stage_id} skipped "
            f"(joint Stage 2+3 was estimated in the stage-2 job)"
        )
        return 0

    stage_dir = os.path.join(args.run_root, args.run_tag, f"stage{stage_id}")
    os.makedirs(os.path.join(stage_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(stage_dir, "ga_reports"), exist_ok=True)

    # Per-segment GA checkpoints (ga_stageN_purposes/targets/segments.pkl) and
    # best-so-far npz files land here, so a walltime/OOM kill leaves everything
    # the salvage path needs inside this stage directory.
    os.environ.setdefault("GA_CHECKPOINT_DIR", stage_dir)

    # Point pipeline logger outputs to this stage-specific directory.
    pl.OUT_DIR = stage_dir
    os.environ["PIPELINE_OUT_DIR"] = stage_dir

    logger.info("=" * 70)
    logger.info(
        f"Single-stage GA runner starting: stage={stage_id}"
        + (f" (effective_sid={effective_sid}, STAGE23_JOINT=1)" if _STAGE23_JOINT else "")
    )
    logger.info(f"OUT_DIR={stage_dir}")
    logger.info("=" * 70)

    all_ga = AllStagesResult()

    try:
        _, out, landuse = load_data()
        res = runner(out, landuse)

        all_ga.results[stage_id] = res
        _save_ga_checkpoint(all_ga, stage_id=stage_id, failed=False)

        hist = getattr(res, "history", []) or []
        hist_df = None
        if hist:
            import pandas as pd
            hist_df = pd.DataFrame(hist)

        summary = _format_ga_checkpoint_summary(res)
        _write_stage_checkpoint_report(
            stage_tag=f"ga_stage{stage_id}",
            title=f"GA Stage {stage_id} - Single Stage Parallel Job",
            summary=summary,
            table=hist_df,
        )
        # Use effective_sid so the HTML writer picks the right code path.
        # This is non-fatal — log loudly but keep exit code 0 so the PBS
        # depend=afterok chain sees the stage as succeeded.
        try:
            _ga_save_stage_coef_html_from_best(effective_sid, res, out, landuse=landuse)
        except Exception as _coef_exc:
            logger.error(
                f"Stage {stage_id} coef/HTML export FAILED (GA result still saved): "
                f"{_coef_exc}",
                exc_info=True,
            )

        try:
            all_ga.export_reports(os.path.join(stage_dir, "ga_reports"))
        except Exception as _rep_exc:
            logger.error(f"Stage {stage_id} export_reports failed: {_rep_exc}", exc_info=True)

        logger.info(f"Stage {stage_id} complete. Artifact: {os.path.join(stage_dir, 'ga_results.pkl')}")
        return 0
    except Exception as exc:
        _save_ga_checkpoint(all_ga, stage_id=stage_id, failed=True)
        logger.error(f"Stage {stage_id} failed: {exc}")
        logger.error(traceback.format_exc())

        # ── Salvage: hand the prediction chain a best-so-far result ────────
        # Per-segment checkpoints (and the incremental best_*.npz files) hold
        # completed segments / winning genes. Reassemble them into a partial
        # GAResult, persist it as the stage's ga_results.pkl, and exit 0 so
        # the afterany dependency chain (merge → prediction) still proceeds.
        try:
            partial, n_seg = _salvage_partial_result(stage_id)
        except Exception as _salv_exc:
            logger.error(f"salvage errored: {_salv_exc}")
            partial, n_seg = None, 0
        # Honesty gate (2026-10-08, mirrors _run_ga_shard.py): a 0-segment
        # salvage must not be blessed as success with a live ga_results.pkl.
        if partial is not None and n_seg > 0:
            all_ga.results[stage_id] = partial
            _save_ga_checkpoint(all_ga, stage_id=stage_id, failed=False)
            marker = os.path.join(stage_dir, "PARTIAL")
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(
                    f"stage={stage_id}\nsegments_completed={n_seg}\n"
                    f"fitness_metric={getattr(partial, 'fitness_metric', '')}\n"
                    f"best_fitness={getattr(partial, 'best_fitness', float('nan'))}\n"
                )
            logger.warning(
                f"Stage {stage_id} SALVAGED a partial result "
                f"({n_seg} completed segment(s)) → ga_results.pkl; "
                f"marker: {marker}. Downstream prediction will proceed with "
                "best-so-far features (unfinished segments use baseline spec)."
            )
            # ── Partial plots/reports: without this, killed/failed stages
            # leave plots/ and ga_reports/ empty (acc_smoke stages 0/1/4/5) and
            # the paper gets no convergence figures for them. Same writers as
            # the success path, strictly non-fatal — a plotting failure must
            # never turn a salvaged stage back into a failed one.
            try:
                all_ga.plot_all(save_path=os.path.join(
                    stage_dir, "plots", "ga_all_stages.png"))
                logger.info(f"Stage {stage_id} salvage plot → plots/ga_all_stages.png")
            except Exception as _pp_exc:
                logger.warning(f"Stage {stage_id} salvage plots skipped: {_pp_exc}")
            try:
                all_ga.export_reports(os.path.join(stage_dir, "ga_reports"))
            except Exception as _pr_exc:
                logger.warning(f"Stage {stage_id} salvage export_reports skipped: {_pr_exc}")
            try:
                _ga_save_stage_coef_html_from_best(
                    effective_sid, partial, out, landuse=landuse)
            except Exception as _pc_exc:
                logger.warning(f"Stage {stage_id} salvage coef/HTML export skipped: {_pc_exc}")
            return 0

        logger.error(
            f"Stage {stage_id} failed with nothing to salvage — downstream "
            "merge will run in non-strict fallback (baseline specs).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
