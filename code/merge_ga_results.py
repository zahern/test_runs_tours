"""
merge_ga_results.py
Merge one-stage GA artifacts into a single AllStagesResult for ga_staged reuse.
"""

import argparse
import os
import pickle

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


def main() -> int:
    args = _parse_args()
    stage_ids = [int(s.strip()) for s in args.stage_ids.split(",") if s.strip()]

    merged = AllStagesResult()
    base = os.path.join(args.run_root, args.run_tag)

    missing = []
    for sid in stage_ids:
        pkl_path = os.path.join(base, f"stage{sid}", "ga_results.pkl")
        if not os.path.isfile(pkl_path):
            missing.append((sid, pkl_path))
            continue
        merged.results[sid] = _load_stage_result(pkl_path, sid)
        print(f"Loaded stage {sid}: {pkl_path}")

    if missing and args.strict:
        for sid, p in missing:
            print(f"Missing stage {sid}: {p}")
        return 2

    merged_dir = os.path.join(base, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    out_pkl = os.path.join(merged_dir, "ga_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(merged, f)

    merged.export_reports(os.path.join(merged_dir, "ga_reports"))

    done = sorted(int(k) for k in merged.results.keys())
    print(f"Merged stages: {done}")
    print(f"Merged artifact: {out_pkl}")

    if not done:
        print("No stages merged.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
