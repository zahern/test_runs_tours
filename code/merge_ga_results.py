"""
merge_ga_results.py
Merge one-stage GA artifacts into a single AllStagesResult for ga_staged reuse.
"""

import argparse
import os
import pickle
import sys

from ga_hypothesis_testing import AllStagesResult


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge stage ga_results.pkl files")
    p.add_argument("--run-root", default=os.path.join("runs", "ga_parallel"))
    p.add_argument("--run-tag", required=True)
    p.add_argument("--stage-ids", default="1,2,3,4,5", help="Comma-separated stage ids")
    p.add_argument("--strict", action="store_true", help="Fail if any requested stage missing")
    return p.parse_args()


def _load_stage_result(path: str, stage_id: int):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, AllStagesResult):
        raise TypeError(f"Not an AllStagesResult: {path}")

    if stage_id in obj.results:
        return obj.results[stage_id]

    if len(obj.results) == 1:
        only_stage = next(iter(obj.results.keys()))
        return obj.results[only_stage]

    raise KeyError(f"Could not resolve stage {stage_id} in {path}")


def _stage_failed(base: str, stage_id: int) -> bool:
    """True when the stage's own progress file records failure.

    Failed runs quarantine their pkls (ga_results.FAILED.pkl), but older runs
    may still hold a live-named partial — the flag keeps merge from blessing it.
    Missing/unreadable progress file means "unknown", never "failed".
    """
    import json
    try:
        with open(os.path.join(base, f"stage{stage_id}", "ga_progress.json")) as f:
            return bool(json.load(f).get("failed", False))
    except Exception:
        return False


def main() -> int:
    args = _parse_args()
    stage_ids = [int(s.strip()) for s in args.stage_ids.split(",") if s.strip()]

    merged = AllStagesResult()
    base = os.path.join(args.run_root, args.run_tag)

    missing = []
    for sid in stage_ids:
        pkl_path = os.path.join(base, f"stage{sid}", "ga_results.pkl")
        loaded = False
        # A failed flag means any live-named artifact is untrustworthy
        # (pre-quarantine runs) — skip straight to salvage/reporting.
        if _stage_failed(base, sid):
            print(f"Stage {sid} marked FAILED (ga_progress.json) — "
                  f"live artifact ignored")
        elif os.path.isfile(pkl_path):
            try:
                merged.results[sid] = _load_stage_result(pkl_path, sid)
                print(f"Loaded stage {sid}: {pkl_path}")
                loaded = True
            except Exception as e:
                print(f"Stage {sid} artifact unresolvable ({e}) — "
                      f"trying checkpoint")
        if not loaded:
            # Salvaged stages persist the identical AllStagesResult in the
            # checkpoint file — prefer it before declaring the stage missing.
            ckpt_path = os.path.join(base, f"stage{sid}", "ga_results_checkpoint.pkl")
            if os.path.isfile(ckpt_path):
                try:
                    merged.results[sid] = _load_stage_result(ckpt_path, sid)
                    print(f"Loaded stage {sid} (SALVAGED checkpoint): {ckpt_path}")
                    loaded = True
                except Exception as e:
                    print(f"Stage {sid} checkpoint unresolvable ({e})")
        if not loaded:
            missing.append((sid, pkl_path))

    if missing:
        print("Missing/unusable stages: "
              + ", ".join(f"{sid} ({p})" for sid, p in missing))
    if missing and args.strict:
        return 2

    merged_dir = os.path.join(base, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    out_pkl = os.path.join(merged_dir, "ga_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(merged, f)

    merged.export_reports(os.path.join(merged_dir, "ga_reports"))

    # Generate convergence plots from the merged results (including salvaged)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pipeline_logger import regenerate_ga_plots_from_pkl
        regenerate_ga_plots_from_pkl(out_pkl, merged_dir)
        print(f"Generated GA plots: {merged_dir}/plots/")
    except Exception as _e:
        print(f"Plot generation skipped: {_e}")

    done = sorted(int(k) for k in merged.results.keys())
    print(f"Merged stages: {done}")
    print(f"Merged artifact: {out_pkl}")

    if not done:
        print("No stages merged.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
