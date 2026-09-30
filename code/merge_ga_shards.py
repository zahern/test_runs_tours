"""
merge_ga_shards.py
==================
Union per-shard GA artifacts into the legacy merged artifact.

Reads every shard
    <run_root>/<run_tag>/stage<N>/shards/<segment>/ga_results.pkl
(plus legacy unsharded stage<N>/ga_results.pkl when present), unions the
per-segment results per stage, picks variant winners from the shard
leaderboards, and writes:

    <run_root>/<run_tag>/merged/
        ga_results.pkl            AllStagesResult (legacy format: prediction
                                  consumes this via GA_STAGED_GA_DIR=.../merged)
        ga_reports/               history CSVs + best-feature CSVs per stage
        plots/                    convergence plots per stage
        shard_leaderboard.csv/html  ALL variant rows, winners marked
        variant_winners.json      overall_pick per (stage, segment, slot)
        shard_feature_maps.json   winning feature spec per shard segment
        merge_info.json           which shards/legacy files fed the merge

Conflict rule: the same segment from two sources -> higher best_fitness wins.
Empty shards (filter matched nothing, fitness <= -1e8) are ignored unless
nothing better exists. Missing segments are reported; --strict fails on them.

Stage-5 note: each shard's best_features is {mode:[...], dest:[...]} for ITS
segment. The merged best_features unions both lists (sorted, deduped) so
legacy `.get('mode')` consumers keep working; the per-segment truth lives in
metadata['segment_results'] + shard_feature_maps.json, which is what
prediction refits from.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ga_hypothesis_testing import (
    AllStagesResult,
    _combine_segment_feature_results,
)
import ga_shard as _shard


def _parse_args():
    p = argparse.ArgumentParser(description="Merge GA shard artifacts")
    p.add_argument("--run-root", default=os.path.join("runs", "ga_parallel"))
    p.add_argument("--run-tag", required=True)
    p.add_argument("--stage-ids", default="0,1,2,3,4,5,51",
                   help="Comma- or underscore-separated stage ids "
                        "(underscore form travels safely via qsub -v)")
    p.add_argument("--strict", action="store_true",
                   help="Fail if any expected segment is missing")
    return p.parse_args()


def _load_allstages(path: str):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, AllStagesResult):
        raise TypeError(f"Not an AllStagesResult: {path}")
    return obj


def _is_empty(res) -> bool:
    try:
        if not getattr(res, "best_features", None):
            return True
        if float(getattr(res, "best_fitness", -1e9)) <= -1e8:
            segs = (getattr(res, "metadata", {}) or {}).get("segment_results", {})
            if not segs:
                return True
        return False
    except Exception:
        return True


def _collect_stage(base: str, sid: int):
    """Return list of (source_label, GAResult-for-sid, shard_dir-or-None)."""
    found = []
    stage_dir = os.path.join(base, f"stage{sid}")
    # sharded sources first (higher precedence on conflict handled by fitness)
    for pkl in sorted(glob.glob(os.path.join(stage_dir, "shards", "*",
                                              "ga_results.pkl"))):
        shard_dir = os.path.dirname(pkl)
        try:
            allst = _load_allstages(pkl)
            res = allst.results.get(sid)
            if res is None and len(allst.results) == 1:
                res = next(iter(allst.results.values()))
            if res is not None and not _is_empty(res):
                found.append((f"shard:{os.path.basename(shard_dir)}", res,
                              shard_dir))
            else:
                print(f"  stage {sid}: ignoring empty shard {shard_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"  stage {sid}: unreadable shard {pkl} ({exc})")
    # legacy unsharded source
    legacy = os.path.join(stage_dir, "ga_results.pkl")
    if os.path.isfile(legacy):
        try:
            allst = _load_allstages(legacy)
            res = allst.results.get(sid)
            if res is None and len(allst.results) == 1:
                res = next(iter(allst.results.values()))
            if res is not None and not _is_empty(res):
                found.append(("legacy", res, None))
        except Exception as exc:  # noqa: BLE001
            print(f"  stage {sid}: unreadable legacy {legacy} ({exc})")
    return found


def _union_stage(sid: int, sources: list) -> tuple:
    """Union segment_results + feature maps across shard sources."""
    seg_results: dict = {}
    seg_source: dict = {}
    feature_maps: dict = {}   # per-segment winning spec (stage5: seg->{mode,dest})
    for label, res, _shard_dir in sources:
        segs = (getattr(res, "metadata", {}) or {}).get("segment_results", {}) or {}
        if segs:
            for seg_label, sub in segs.items():
                prev = seg_results.get(seg_label)
                if prev is None or float(getattr(sub, "best_fitness", -1e9)) > \
                        float(getattr(prev, "best_fitness", -1e9)):
                    seg_results[seg_label] = sub
                    seg_source[seg_label] = label
        fm = getattr(res, "best_features", None)
        if isinstance(fm, dict):
            # stage5 shards share {mode,dest} keys per segment: stash per source,
            # union at the end (see below).
            feature_maps[label] = {k: list(v) for k, v in fm.items() if v}
        elif isinstance(fm, list) and fm:
            feature_maps[label] = list(fm)
    return seg_results, seg_source, feature_maps


def _merged_features(sid: int, feature_maps: dict, seg_results: dict):
    if sid == 5:
        modes, dests = [], []
        for fm in feature_maps.values():
            if isinstance(fm, dict):
                modes += list(fm.get("mode", []))
                dests += list(fm.get("dest", []))
        return {"mode": sorted(set(modes)), "dest": sorted(set(dests))}
    merged = {}
    for fm in feature_maps.values():
        if isinstance(fm, dict):
            for k, v in fm.items():
                merged.setdefault(k, [])
                for x in (v or []):
                    if x not in merged[k]:
                        merged[k].append(x)
        elif isinstance(fm, list):
            merged.setdefault("all", [])
            for x in fm:
                if x not in merged["all"]:
                    merged["all"].append(x)
    return merged


def main() -> int:
    args = _parse_args()
    stage_ids = [int(s.strip()) for s in str(args.stage_ids).replace("_", ",").split(",") if s.strip()]
    base = os.path.join(args.run_root, args.run_tag)

    merged = AllStagesResult()
    all_variant_rows = []
    variant_winners = {}
    shard_feature_maps = {}
    merge_info = {"run_root": args.run_root, "run_tag": args.run_tag,
                  "sources": {}, "missing": {}}

    for sid in stage_ids:
        sources = _collect_stage(base, sid)
        merge_info["sources"][str(sid)] = [label for label, _, _ in sources]
        if not sources:
            merge_info["missing"][str(sid)] = "no usable shard/legacy artifact"
            print(f"Stage {sid}: MISSING (no usable artifact)")
            continue
        seg_results, seg_source, feature_maps = _union_stage(sid, sources)
        feats = _merged_features(sid, feature_maps, seg_results)
        # Stage 5 must produce BOTH a mode and a destination spec. _union_stage
        # drops empty per-source values (feature_maps keeps only `if v`), and
        # _merged_features unions whatever survives -- so if no shard ever
        # yielded a dest spec the merge silently emits an EMPTY `dest=` alongside
        # a perfectly plausible best_fitness and writes a valid-looking report.
        # Observed in runs/ga_parallel/qld_shard1/merged/ga_reports/
        # ga_stage5_best_model.txt (fitness_metric=mean(-BIC/n),
        # best_fitness=-2.8391605167, `dest=` bare). Fail loudly instead.
        if sid == 5 and isinstance(feats, dict):
            _empty = [k for k in ("mode", "dest") if not feats.get(k)]
            if _empty:
                merge_info.setdefault("invalid_spec", {})[str(sid)] = (
                    f"empty {_empty} in merged stage-5 feature spec "
                    f"(from sources {merge_info['sources'][str(sid)]})")
                print(f"ERROR: stage 5 merged spec is missing {', '.join(_empty)} "
                      f"-- the merged best-model report will contain an empty "
                      f"feature set and must not be treated as a valid result.")
                combined_unusable = True
            else:
                combined_unusable = False
        else:
            combined_unusable = False
        fitness_metric = sources[0][1].fitness_metric \
            if getattr(sources[0][1], "fitness_metric", "") else "mean(-BIC/n)"
        n_restarts = max(int(getattr(r, "n_restarts", 0) or 0) for _, r, _ in sources)
        combined = _combine_segment_feature_results(
            stage=sid, segment_results=seg_results, feature_map=feats,
            fitness_metric=fitness_metric, n_restarts=n_restarts)
        combined.metadata["shard_sources"] = seg_source
        combined.metadata["shard_feature_maps"] = feature_maps
        if combined_unusable:
            # Keep the merged artifact (so the operator can inspect it) but flag
            # it so nothing downstream mistakes it for a converged stage-5 spec.
            combined.metadata["invalid_spec"] = True
            combined.fitness_metric = f"{fitness_metric} [INVALID SPEC]"
        merged.results[sid] = combined
        print(f"Stage {sid}: merged {len(seg_results)} segment(s) "
              f"from {merge_info['sources'][str(sid)]}")

        # variant leaderboards + winner specs from each shard dir
        for label, _res, shard_dir in sources:
            if not shard_dir:
                continue
            seg_key = os.path.basename(shard_dir)
            for lb in glob.glob(os.path.join(shard_dir, "variants",
                                              "leaderboard.csv")):
                for row in _shard.read_leaderboards([lb]):
                    all_variant_rows.append(row)
            wj = os.path.join(shard_dir, "variants", "winner.json")
            if os.path.isfile(wj):
                try:
                    with open(wj, encoding="utf-8") as fh:
                        for k, v in json.load(fh).items():
                            variant_winners[f"{sid}/{seg_key}/{k}"] = v
                except Exception:
                    pass
            for spec in glob.glob(os.path.join(shard_dir, "variants",
                                               "winner_spec_*.json")):
                try:
                    with open(spec, encoding="utf-8") as fh:
                        shard_feature_maps[f"{sid}/{seg_key}/"
                                           f"{os.path.basename(spec)}"] = json.load(fh)
                except Exception:
                    pass

    # expected-segment audit
    for sid in stage_ids:
        if sid == 0:
            continue
        expected = set(_shard.STAGE_SEGMENTS.get(sid, []))
        got = set()
        res = merged.results.get(sid)
        if res is not None:
            segs = (getattr(res, "metadata", {}) or {}).get("segment_results", {}) or {}
            lbls = " ".join(segs.keys()).lower()
            for seg in expected:
                aliases = _shard.SEGMENT_ALIASES.get(seg, [seg])
                if any(a in lbls for a in aliases):
                    got.add(seg)
        missing = sorted(expected - got)
        if missing:
            merge_info["missing"][str(sid)] = missing
            print(f"Stage {sid}: missing segments {missing}")

    merged_dir = os.path.join(base, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    out_pkl = os.path.join(merged_dir, "ga_results.pkl")
    with open(out_pkl, "wb") as fh:
        pickle.dump(merged, fh)
    merged.export_reports(os.path.join(merged_dir, "ga_reports"))

    if all_variant_rows:
        _shard.write_leaderboard(
            [{k: r.get(k, "") for k in _shard.LEADERBOARD_FIELDS}
             for r in all_variant_rows],
            os.path.join(merged_dir, "variant_board"))
        import shutil
        for ext in ("leaderboard.csv", "leaderboard.html"):
            src = os.path.join(merged_dir, "variant_board", ext)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(merged_dir, f"shard_{ext}"))
        print(f"Variant board: {len(all_variant_rows)} rows -> "
              f"{merged_dir}/shard_leaderboard.csv")

    with open(os.path.join(merged_dir, "variant_winners.json"), "w",
              encoding="utf-8") as fh:
        json.dump(variant_winners, fh, indent=2)
    with open(os.path.join(merged_dir, "shard_feature_maps.json"), "w",
              encoding="utf-8") as fh:
        json.dump(shard_feature_maps, fh, indent=2)
    with open(os.path.join(merged_dir, "merge_info.json"), "w",
              encoding="utf-8") as fh:
        json.dump(merge_info, fh, indent=2)

    try:
        from pipeline_logger import regenerate_ga_plots_from_pkl
        regenerate_ga_plots_from_pkl(out_pkl, merged_dir)
        print(f"GA plots -> {merged_dir}/plots/")
    except Exception as exc:  # noqa: BLE001
        print(f"Plot generation skipped: {exc}")

    done = sorted(int(k) for k in merged.results.keys())
    print(f"Merged stages: {done}")
    print(f"Merged artifact: {out_pkl}")
    if not done:
        return 3
    if merge_info["missing"] and args.strict:
        print("STRICT: missing segments/stages -> failing")
        return 2
    if merge_info.get("invalid_spec"):
        _specs = ", ".join(f"stage {k} ({v})" for k, v in merge_info["invalid_spec"].items())
        print(f"INVALID SPEC: {_specs}")
        if args.strict:
            print("STRICT: invalid merged spec -> failing")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
