"""
_run_ga_single_stage.py
Run exactly one GA stage (1-5) and save a one-stage ga_results.pkl artifact.

Designed for PBS fan-out where stages run in parallel jobs and are merged later.

Special behaviour when STAGE23_JOINT=1:
  - stage-id 2 → runs ga_stage23_joint (joint budget-constrained Stage 2+3)
  - stage-id 3 → skips gracefully (work already done by the stage-2 job)
"""

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
    p.add_argument("--stage-id", type=int, required=True, choices=[0, 1, 2, 3, 4, 5])
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
    }
    return mapping[stage_id], stage_id


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
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
