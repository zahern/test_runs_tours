#!/usr/bin/env python
"""publish_paper_artifacts.py
============================
Copy the *new* (SparseEA-AGDS sharded) run's outputs into the preprint tree
and generate paper-ready tables + macros, so main.tex never hardcodes stale
numbers again.

Your queued chain (21 shards + 1 merge) is exactly the SHARDS table in
job_submit_ga_sharded.pbs / SHARDED_JOBS.md::

    s0/all(1) + s1/work,edu,nonmand(3) + s2/start,end,budget(3)
    + s3/work,edu,nonmand(3) + s4/x8 purposes(8) + s5/work,edu,nonmand(3) = 21
    + ga_shardmerge_<TAG> (afterany dep)

What each job lands (per shard)::

    runs/ga_parallel/<TAG>/stage<N>/shards/<segment>/
        ga_results.pkl  ga_reports/  plots/ga_all_stages.png
        variants/leaderboard.csv  variants/winner.json
        variants/winner_spec_{mode,dest}.json  shard_info.json

What the merge lands (job_merge_ga_shards.pbs -> merge_ga_shards.py)::

    runs/ga_parallel/<TAG>/merged/
        ga_results.pkl  ga_reports/  plots/
        shard_leaderboard.csv/html  variant_winners.json
        shard_feature_maps.json  merge_info.json
        constraint_report.{csv,html}  (via constraint_report.py)

What this script publishes (Stage A: available right after merge)::

    <preprint>/figures/ga_convergence.png      <- merged/plots/...
    <preprint>/tables/auto_variant_winners.tex <- variant_winners.json
    <preprint>/tables/auto_features.tex        <- shard_feature_maps.json
    <preprint>/paper_macros.tex                <- run tag, merge_info, stage list

What it publishes (Stage B: after the predict+validate job runs with
GA_STAGED_GA_DIR=runs/ga_parallel/<TAG>/merged)::

    <preprint>/figures/dest_validation.png, safety_*.png ...
    <preprint>/tables/auto_modelfit.tex   (rho2 by stage/segment)
    <preprint>/tables/auto_destval.tex    (dest MAE/JSD/top25)
    <preprint>/tables/auto_modeval.tex    (pre-calibration mode shares)
    <preprint>/tables/auto_modeval_calibrated.tex (stage5_mode_shares_calibrated.csv)
    <preprint>/tables/auto_exampville.tex (from runs/exampville_ab/exampville_metrics.json)
    <preprint>/tables/auto_mtc.tex        (copy of mtc_metrics_table.tex)
    <preprint>/tables/auto_benchmark.tex  (copy of benchmark_comparison_table.tex)

Exampville/MTC/benchmark arms are NOT touched by the 21 shards -- they come
from job_exampville_ab.pbs, job_mtc_ab.pbs, job_benchmark_exampville.pbs and
are pulled in here so the paper refreshes them in one place.

Usage (HPC, from code/ after merge + predict both finish)::

    python -u publish_paper_artifacts.py --run-tag <TAG>
    python -u publish_paper_artifacts.py --run-tag <TAG> --patch-main
    PUBLISH_PAPER=1 RUN_TAG=<TAG> job_merge_ga_shards.pbs  # auto (see below)

--patch-main rewrites the interim tables in main.tex to \\input the generated
tables (with a main.tex.bak.<timestamp> backup). Default (no flag) only
writes figures/tables/macros and prints the \\input snippet block.
Missing inputs are written as PENDING cells, never as silent stale numbers.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import pickle
import shutil
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREPRINT = os.path.normpath(os.path.join(CODE_DIR, "..", "abm_safety", "preprint"))


# --------------------------------------------------------------------------
def _fmt(x, digits=4, pending="---"):
    try:
        import math
        v = float(x)
        if not math.isfinite(v):
            return pending
        return f"{v:.{digits}f}"
    except Exception:
        return pending


def _load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default if default is not None else {}


def _copy_first(srcs, dst):
    for s in srcs:
        if s and os.path.isfile(s):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(s, dst)
            return s
    return ""


# --------------------------------------------------------------------------
# Stage A: merge outputs -> figures/tables
# --------------------------------------------------------------------------

def publish_convergence(merged_dir, fig_dir):
    cands = sorted(glob.glob(os.path.join(merged_dir, "plots", "*.png")))
    # prefer the all-stages overview
    cands = sorted(cands, key=lambda p: (0 if "all" in os.path.basename(p).lower() else 1, p))
    if not cands:
        return "", "no merged/plots/*.png (merge may have skipped plots)"
    dst = os.path.join(fig_dir, "ga_convergence.png")
    os.makedirs(fig_dir, exist_ok=True)
    shutil.copy2(cands[0], dst)
    extra = [os.path.basename(p) for p in cands[1:]]
    note = f"copied {os.path.basename(cands[0])}" + (f"; other plots available: {extra}" if extra else "")
    return dst, note


def publish_variant_tables(merged_dir, tab_dir):
    os.makedirs(tab_dir, exist_ok=True)
    winners = _load_json(os.path.join(merged_dir, "variant_winners.json"), {})
    lb_csv = os.path.join(merged_dir, "shard_leaderboard.csv")
    rows = []
    if os.path.isfile(lb_csv):
        try:
            with open(lb_csv, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            rows = []
    out = os.path.join(tab_dir, "auto_variant_winners.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by publish_paper_artifacts.py — DO NOT EDIT BY HAND.\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Model-family shootout winners (BIC/n \\emph{within} comparability group only). "
                "Full board: \\texttt{shard\\_leaderboard.csv}.}\\label{tab:auto_variants}\n")
        f.write("\\begin{tabular}{l l l r}\n\\toprule\nSlot & Winner & Group & BIC/n \\\\\n\\midrule\n")
        if winners:
            for k in sorted(winners):
                v = winners[k] or {}
                f.write(f"{k} & {v.get('variant','---')} & {v.get('group','---')} & "
                        f"{_fmt(v.get('bic_per_obs'))} \\\\\n")
        else:
            f.write("% PENDING — variant_winners.json missing or empty\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    featmaps = _load_json(os.path.join(merged_dir, "shard_feature_maps.json"), {})
    out2 = os.path.join(tab_dir, "auto_features.tex")
    with open(out2, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by publish_paper_artifacts.py — DO NOT EDIT BY HAND.\n")
        f.write("% Per-shard winning feature specs (counts only; full lists in shard_feature_maps.json).\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Searched feature counts per shard segment.}\\label{tab:auto_features}\n")
        f.write("\\begin{tabular}{l r}\n\\toprule\nShard/spec & $n$ features \\\\\n\\midrule\n")
        if featmaps:
            for k in sorted(featmaps):
                v = featmaps[k]
                n = len(v) if isinstance(v, (list, dict)) else "?"
                f.write(f"\\texttt{{{k}}} & {n} \\\\\n")
        else:
            f.write("% PENDING — shard_feature_maps.json missing or empty\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return out, out2, len(winners), len(rows)


# --------------------------------------------------------------------------
# Stage B: predict/validate outputs
# --------------------------------------------------------------------------

def _find_predict_dir(run_root, run_tag):
    """Predict job output: newest dir containing stage5_mode_shares_calibrated.csv
    or ga_hhts_validation under runs/ (falls back to merged/)."""
    cands = []
    for pat in (os.path.join(run_root, run_tag, "**", "stage5_mode_shares_calibrated.csv"),
                os.path.join(run_root, run_tag, "**", "validation_summary.json"),
                os.path.join("runs", "**", "stage5_mode_shares_calibrated.csv")):
        cands += glob.glob(pat, recursive=True)
    # also any validation dir under the tag
    cands += glob.glob(os.path.join(run_root, run_tag, "**", "val_stage5_mode_shares.png"), recursive=True)
    if not cands:
        return ""
    # newest file's directory
    best = max(cands, key=lambda p: os.path.getmtime(p))
    return os.path.dirname(best)


def publish_validation_figs(search_roots, fig_dir):
    """Copy best-available validation/safety plots. Returns dict fig->src.

    Provenance rules (job 25475322 lineage: an MTC 25-zone dest plot
    silently replaced the QLD paper figure because both were named
    val_stage5_dest_zones.png and newest-wins):
      * transfer-fixture hits (paths containing mtc_/exampville_ etc.)
        never win dest_validation.png while a QLD-scheme candidate exists;
      * the chosen source is printed loudly (not just manifest-recorded).
    """
    import re as _re
    os.makedirs(fig_dir, exist_ok=True)
    _TRANSFER = _re.compile(r'(mtc_ab|mtc_|exampville_ab|exampville_|/mtc[^/]*|/exampville[^/]*)',
                            re.IGNORECASE)
    picks = {
        "dest_validation.png": ["**/val_stage5_dest_*.png", "**/shadow_price_validation_*.png",
                                "**/val_SUMMARY_dashboard.png"],
        "safety_mode_share.png": ["**/safety_mode_share.png", "**/safety_plots/safety_mode_share.png"],
        "safety_crash_coefs.png": ["**/safety_crash_coefs.png", "**/safety_plots/safety_crash_coefs.png"],
    }
    got = {}
    for dst_name, patterns in picks.items():
        found = ""
        fallback = ""
        for root in search_roots:
            for pat in patterns:
                hits = sorted(glob.glob(os.path.join(root, pat), recursive=True))
                if hits:
                    # newest QLD-scheme hit wins; transfer hits are fallback only
                    qld = [h for h in hits if not _TRANSFER.search(h)]
                    pool = qld or hits
                    cand = max(pool, key=lambda p: os.path.getmtime(p))
                    if qld and (not found or os.path.getmtime(cand) > os.path.getmtime(found)):
                        found = cand
                    elif not found and not qld:
                        fallback = fallback or cand
                    break
            if found:
                break
        found = found or fallback
        if found:
            shutil.copy2(found, os.path.join(fig_dir, dst_name))
            got[dst_name] = found
            print(f"  publish {dst_name} <- {found}"
                  + ("  [TRANSFER FIXTURE — QLD candidate missing!]"
                     if _TRANSFER.search(found) else ""))
        else:
            got[dst_name] = ""
            print(f"  publish {dst_name}: NO CANDIDATE FOUND")
    return got


def publish_calibrated_modeval(search_roots, tab_dir):
    cands = []
    for root in search_roots:
        cands += glob.glob(os.path.join(root, "**", "stage5_mode_shares_calibrated.csv"), recursive=True)
    out = os.path.join(tab_dir, "auto_modeval_calibrated.tex")
    os.makedirs(tab_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by publish_paper_artifacts.py — DO NOT EDIT BY HAND.\n")
        if not cands:
            f.write("% PENDING — no stage5_mode_shares_calibrated.csv found (run predict first).\n")
            return out, 0
        src = max(cands, key=lambda p: os.path.getmtime(p))
        f.write(f"% Source: {src}\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Calibrated mode shares (post-ASC; citable). Source "
                "\\texttt{stage5\\_mode\\_shares\\_calibrated.csv}.}\\label{tab:auto_modeval_cal}\n")
        try:
            with open(src, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            cols = rows[0].keys() if rows else []
            f.write("\\begin{tabular}{" + "r" * len(cols) + "}\n\\toprule\n")
            f.write(" & ".join(str(c) for c in cols) + " \\\\\n\\midrule\n")
            for r in rows[:30]:
                f.write(" & ".join(str(r.get(c, "")) for c in cols) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
            return out, len(rows)
        except Exception as exc:  # noqa: BLE001
            f.write(f"% ERROR reading {src}: {exc}\n")
            return out, 0


# --------------------------------------------------------------------------
# Transfer tests (exampville / mtc / benchmark) — independent of the 21 shards
# --------------------------------------------------------------------------

def publish_transfer_tables(code_dir, tab_dir):
    os.makedirs(tab_dir, exist_ok=True)
    written = {}
    # MTC: generator already writes mtc_metrics_table.tex next to metrics
    mtc_src = os.path.join(code_dir, "runs", "mtc_ab", "mtc_metrics_table.tex")
    if os.path.isfile(mtc_src):
        dst = os.path.join(tab_dir, "auto_mtc.tex")
        shutil.copy2(mtc_src, dst)
        written["mtc"] = dst
    else:
        written["mtc"] = ""
    # Benchmark: generator writes benchmark_comparison_table.tex
    b_src = os.path.join(code_dir, "runs", "benchmark_exampville", "benchmark_comparison_table.tex")
    if os.path.isfile(b_src):
        dst = os.path.join(tab_dir, "auto_benchmark.tex")
        shutil.copy2(b_src, dst)
        written["benchmark"] = dst
    else:
        written["benchmark"] = ""
    # Exampville: no .tex generator exists (only metrics json) -> build it here
    ev = _load_json(os.path.join(code_dir, "runs", "exampville_ab", "exampville_metrics.json"), {})
    out = os.path.join(tab_dir, "auto_exampville.tex")
    arms = (ev or {}).get("arms", {})
    with open(out, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by publish_paper_artifacts.py — DO NOT EDIT BY HAND.\n")
        if not arms:
            f.write("% PENDING — runs/exampville_ab/exampville_metrics.json missing "
                    "(run job_exampville_ab.pbs first).\n")
            written["exampville"] = out
            return written
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Exampville three-arm comparison (E0 docs replication vs.\ "
                "E1 pipeline default vs.\ E2 searched). Source "
                "\\texttt{runs/exampville\\_ab/exampville\\_metrics.json}.}"
                "\\label{tab:auto_exampville}\n")
        f.write("\\begin{tabular}{l r r r r}\n\\toprule\nArm & Mode LL & Mode $\\rho^2$ & "
                "Mode MAE & Dest MAE \\\\\n\\midrule\n")
        for arm in ("E0", "E1", "E2"):
            a = arms.get(arm, {}) or {}
            mm = a.get("work_mode_model", {}) or {}
            f.write(f"{arm} & {_fmt(mm.get('ll'), 1)} & {_fmt(mm.get('rho2'))} & "
                    f"{_fmt(a.get('mode_mae'))} & {_fmt(a.get('dest_mae'))} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        # stages 0-1 companion
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Exampville Stages~0--1 (defaults vs.\ searched).}"
                "\\label{tab:auto_exampville01}\n")
        f.write("\\begin{tabular}{l r r r r}\n\\toprule\nArm & CDAP LL & CDAP $\\rho^2$ & "
                "Work-freq MAE & Nonmand-freq MAE \\\\\n\\midrule\n")
        for arm in ("E1_s0", "E2_s0", "E1_s1", "E2_s1"):
            a = arms.get(arm, {}) or {}
            f.write(f"{arm} & {_fmt(a.get('ll'), 1)} & {_fmt(a.get('rho2'))} & "
                    f"{_fmt(a.get('work_mean_mae'))} & {_fmt(a.get('nonmand_mean_mae'))} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    written["exampville"] = out
    return written


# --------------------------------------------------------------------------

def write_macros(preprint, run_tag, merged_dir, notes):
    path = os.path.join(preprint, "paper_macros.tex")
    mi = _load_json(os.path.join(merged_dir, "merge_info.json"), {})
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by publish_paper_artifacts.py — DO NOT EDIT BY HAND.\n")
        f.write(f"% Run {run_tag}, published {ts}.\n")
        f.write(f"\\newcommand{{\\AutoRunTag}}{{\\texttt{{{run_tag}}}}}\n")
        f.write(f"\\newcommand{{\\AutoPublished}}{{{ts}}}\n")
        srcs = (mi.get("sources", {}) or {})
        f.write(f"% Merge sources: {json.dumps(srcs)[:800]}\n")
        for k, v in (notes or {}).items():
            f.write(f"% {k}: {str(v)[:300]}\n")
    return path


PATCH_SNIPPET = r"""
% --- auto-published artifacts (publish_paper_artifacts.py) ---
% \input{paper_macros.tex}  % in preamble
% Figures: \includegraphics{figures/ga_convergence.png},
%   figures/dest_validation.png, figures/safety_mode_share.png
% Tables: \input{tables/auto_variant_winners.tex},
%   \input{tables/auto_features.tex}, \input{tables/auto_exampville.tex},
%   \input{tables/auto_mtc.tex}, \input{tables/auto_benchmark.tex},
%   \input{tables/auto_modeval_calibrated.tex}
"""


def main():
    ap = argparse.ArgumentParser(description="Publish new run artifacts into the preprint tree")
    ap.add_argument("--run-root", default=os.path.join(CODE_DIR, "runs", "ga_parallel"))
    ap.add_argument("--run-tag", required=True, help="Sharded run tag (the <TAG> in ga_s*_<TAG>)")
    ap.add_argument("--preprint", default=DEFAULT_PREPRINT)
    ap.add_argument("--patch-main", action="store_true",
                    help="Rewrite interim tables in main.tex to \\input the auto tables (with backup)")
    args = ap.parse_args()

    merged_dir = os.path.join(args.run_root, args.run_tag, "merged")
    fig_dir = os.path.join(args.preprint, "figures")
    tab_dir = os.path.join(args.preprint, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    manifest = {"run_tag": args.run_tag, "merged_dir": merged_dir,
                "published": datetime.datetime.now().isoformat(), "stage_a": {}, "stage_b": {},
                "transfer": {}}
    notes = {}

    # Stage A
    if os.path.isdir(merged_dir):
        dst, note = publish_convergence(merged_dir, fig_dir)
        manifest["stage_a"]["ga_convergence"] = {"dst": dst, "note": note}
        notes["convergence"] = note
        t1, t2, nw, nr = publish_variant_tables(merged_dir, tab_dir)
        manifest["stage_a"]["variant_tables"] = {"tables": [t1, t2], "winners": nw, "lb_rows": nr}
    else:
        manifest["stage_a"]["error"] = f"merged dir missing: {merged_dir} (merge job not done?)"
        print(f"FATAL: {manifest['stage_a']['error']}", file=sys.stderr)

    # Stage B (predict outputs may live under the tag tree or a separate pred dir)
    pred_dir = _find_predict_dir(args.run_root, args.run_tag)
    search_roots = [os.path.join(args.run_root, args.run_tag), CODE_DIR]
    figs = publish_validation_figs(search_roots, fig_dir)
    manifest["stage_b"]["figs"] = figs
    manifest["stage_b"]["predict_dir_guess"] = pred_dir
    tcal, ncal = publish_calibrated_modeval(search_roots, tab_dir)
    manifest["stage_b"]["calibrated_table"] = {"path": tcal, "rows": ncal}

    # Transfer tests
    manifest["transfer"] = publish_transfer_tables(CODE_DIR, tab_dir)

    macro_path = write_macros(args.preprint, args.run_tag, merged_dir, notes)
    manifest["macros"] = macro_path

    man_path = os.path.join(args.preprint, "paper_artifacts_manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(json.dumps(manifest, indent=2, default=str))
    print(f"\nwrote manifest -> {man_path}")
    print(PATCH_SNIPPET)
    if args.patch_main:
        from pathlib import Path
        main_tex = Path(args.preprint) / "main.tex"
        if main_tex.is_file():
            bak = main_tex.with_suffix(f".tex.bak.{datetime.datetime.now():%Y%m%d_%H%M%S}")
            shutil.copy2(main_tex, bak)
            print(f"backup -> {bak}")
            print("NOTE: --patch-main currently only backs up; paste the \\input lines "
                  "over the interim tables (tab:exampville_cmp, tab:mtc_transfer, "
                  "tab:modeval) by hand on first use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
