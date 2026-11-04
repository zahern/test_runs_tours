#!/usr/bin/env python
"""Generate the Exampville and prototype-MTC table ROWS from run metrics JSON.

Why rows and not whole tables: the captions, labels and provenance blocks in
main.tex are prose that a generator would only reproduce verbatim.  The rows
are the part that gets copied by hand and drifts, so those are generated and
\\input from inside the existing table environments.

Sources (all paths relative to the pipeline code dir):
  runs/exampville_ab_27074291.aqua/exampville_metrics.json  -> Table 12 (E0/E1/E2 Stage 5)
  runs/exampville_ab/exampville_metrics.json                -> Table 13 (CDAP, 13-feature default)
  runs/exampville_s1fix_20261029/exampville_metrics.json    -> Table 13 (Stage 1 frequency)
  runs/mtc_ab/mtc_metrics.json                              -> prototype-MTC table

Usage:
  python tables/gen_paper_tables.py [--code-dir PATH] [--out-dir PATH]

Every emitted number is formatted from the artifact; nothing is typed by hand.
Missing values print as n.r. rather than being invented.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Published Larch example 203 destination statistics.  These are documentation
# values, not pipeline output, so they are the one hard-coded pair in this file.
DOCS_203 = {"ll": -25157.72, "rho2": 0.109}


def num(x, nd=2, commas=True, signed=False):
    """Format a float for LaTeX, or 'n.r.' when absent."""
    if x is None:
        return "n.r."
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n.r."
    if v != v:  # NaN
        return "n.r."
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    if commas:
        head, _, tail = s.partition(".")
        neg = head.startswith("-")
        digits = head.lstrip("-")
        digits = f"{int(digits):,}".replace(",", "{,}")
        s = ("-" if neg else "") + digits + (("." + tail) if tail else "")
    return f"${s}$"


def tex_num(x, nd=2, signed=False):
    return num(x, nd=nd, commas=True, signed=signed)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def k_from_arm(arm):
    """Feature count for a CDAP arm, preferring the explicit bookkeeping.

    ``len(features)`` and ``n_features`` disagree in the 25 Oct retry; the
    explicit n_features is the count actually used for BIC, so it wins.
    """
    k = arm.get("n_features")
    if k is not None:
        return int(k)
    feats = arm.get("features")
    return len(feats) if feats else None


def table12_rows(retry):
    """Exampville Stage-5 comparison (work tours)."""
    a = retry["arms"]
    e0, e1, e2 = a["E0"], a["E1"], a["E2"]
    docs = e0.get("docs_reference", {})

    def mode_mae(arm):
        return tex_num(arm.get("mode_mae"), nd=4)

    # rho2 to 3 dp, MAE to 4 dp: the precision the paper actually quotes, so a
    # regeneration reproduces the printed digits instead of silently rounding
    # 0.027 -> 0.03 or 0.0106 -> 0.01.
    rows = [
        "E0 docs replication & "
        f"{tex_num(e0.get('ll'))} & {tex_num(docs.get('rho2'), nd=3)} (docs) & "
        f"{tex_num(DOCS_203['ll'])} (203) & {tex_num(DOCS_203['rho2'], nd=3)} (203) & "
        f"{mode_mae(e0)} & {tex_num(e0.get('dest_mae'), nd=4)} \\\\",
        "E1 pipeline default & "
        f"{tex_num(e1['work_mode_model']['ll'])} & {tex_num(e1['work_mode_model']['rho2'], nd=3)} & "
        f"{tex_num(e1['work_dest_model']['ll'])} & {tex_num(e1['work_dest_model']['rho2'], nd=3)} & "
        f"{mode_mae(e1)} & {tex_num(e1.get('dest_mae'), nd=4)} \\\\",
        "E2 searched & "
        f"{tex_num(e2['work_mode_model']['ll'])} & {tex_num(e2['work_mode_model']['rho2'], nd=3)} & "
        f"{tex_num(e2['work_dest_model']['ll'])} & {tex_num(e2['work_dest_model']['rho2'], nd=3)} & "
        f"{mode_mae(e2)} & {tex_num(e2.get('dest_mae'), nd=4)} \\\\",
    ]
    return "\n".join(rows) + "\n"


def table13_rows(ab, s1):
    """Exampville Stages 0--1: CDAP with BIC evidence, then frequency recovery.

    Stage 1 comes from the corrected re-run (job 27308422), not from the
    Oct-25 retry: the retry's work-frequency metric averaged every column
    whose name contained the purpose string, which swept employment totals in
    with the per-person ``n_work_out`` count and produced a mean of 2853.6
    tours per person.  That run records work_mean_mae = 2852.94; this one
    records 0.0093.
    """
    a = ab["arms"]
    e1, e2 = a["E1_s0"], a["E2_s0"]
    # CDAP sample size lives in the run's SOURCE.json. The metrics artifact
    # records only a cluster path, so fall back to the local run directory.
    n = None
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (ab.get("root", "") + "/SOURCE.json",
                 os.path.join(here, "..", "..", "..", "code", "runs",
                              "exampville_ab", "SOURCE.json")):
        try:
            n = load(cand).get("n_persons")
            if n:
                break
        except Exception:
            continue

    ll1, ll2 = float(e1["ll"]), float(e2["ll"])
    k1, k2 = k_from_arm(e1), k_from_arm(e2)
    dll = ll2 - ll1
    if n:
        import math
        dbic = (k2 - k1) * math.log(n) - 2.0 * dll
        dbic_n = dbic / n
    else:
        dbic_n = None

    # Stage 1. The searched arm still records the failure sentinel, so its
    # figures are a fallback specification, not a search outcome.
    s1a = s1["arms"]
    s1e1, s1e2 = s1a["E1_s1"], s1a["E2_s1"]
    dagger = ""
    if float(s1e2.get("best_fitness") or 0) <= -1e8:
        dagger = "$^{\\dagger}$"

    rows = [
        f"E1 defaults & {k1} & {tex_num(ll1)} & --- & --- & --- & --- \\\\",
        f"E2 searched & {k2}  & {tex_num(ll2)} & {tex_num(dll, signed=True)} & "
        f"{tex_num(dbic_n, nd=4, signed=True)} & --- & --- \\\\",
        "\\midrule",
        f"E1 defaults & --- & --- & --- & --- & "
        f"{tex_num(s1e1.get('work_mean_mae'), nd=4)} & "
        f"{tex_num(s1e1.get('nonmand_mean_mae'), nd=4)} \\\\",
        f"E2 searched{dagger} & --- & --- & --- & --- & "
        f"{tex_num(s1e2.get('work_mean_mae'), nd=4)} & "
        f"{tex_num(s1e2.get('nonmand_mean_mae'), nd=4)} \\\\",
    ]
    return "\n".join(rows) + "\n"


def mtc_rows(mtc):
    """Prototype-MTC transfer: observed vs predicted universes kept apart."""
    a = mtc["arms"]
    m0, m1, m2 = a["M0"], a["M1"], a["M2"]

    def row(label, arm, ref=False):
        if ref:
            return (f"{label} & {num(arm.get('n_tours'), nd=0)} & --- & --- & --- & --- \\\\")
        return (
            f"{label} & {num(mtc.get('n_tours'), nd=0)} & {num(arm.get('n_pred'), nd=0)} & "
            f"{tex_num(arm.get('dest_mae'), nd=4)} & {tex_num(arm.get('dest_top10'), nd=4)} & "
            f"{tex_num(arm.get('trip_len_mae'), nd=4)} \\\\"
        )

    failed = float(m2.get("best_fitness", 0) or 0) <= -1e8
    rows = [
        row("M0 ActivitySim regress (reference)", m0, ref=True),
        row("M1 pipeline default", m1),
        ("M2 searched (search failed) & "
         f"{num(mtc.get('n_tours'), nd=0)} & {num(m2.get('n_pred'), nd=0)} & "
         "$\\equiv$M1 & $\\equiv$M1 & $\\equiv$M1 \\\\")
        if failed else row("M2 searched", m2),
    ]
    return "\n".join(rows) + "\n"


def tabular(colspec, head, body):
    """Wrap rows in a complete tabular.

    The whole environment is emitted rather than bare rows because a row
    ending in ``\\\\`` at the end of an ``\\input`` file leaves the alignment
    open, and the following ``\\bottomrule`` then raises 'Misplaced \\noalign'.
    """
    return (
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{head} \\\\\n"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def write(path, header, body):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("% Auto-generated by tables/gen_paper_tables.py --- DO NOT EDIT.\n")
        fh.write(f"% {header}\n")
        fh.write(body)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", default=os.path.join(here, "..", "..", "..", "code"))
    ap.add_argument("--out-dir", default=here)
    args = ap.parse_args()

    code = os.path.abspath(args.code_dir)
    out = os.path.abspath(args.out_dir)

    retry_p = os.path.join(code, "runs", "exampville_ab_27074291.aqua", "exampville_metrics.json")
    s5_p = os.path.join(code, "runs", "exampville_s5fix_20261030", "exampville_metrics.json")
    ab_p = os.path.join(code, "runs", "exampville_ab", "exampville_metrics.json")
    s1_p = os.path.join(code, "runs", "exampville_s1fix_20261029", "exampville_metrics.json")
    mtc_p = os.path.join(code, "runs", "mtc_ab", "mtc_metrics.json")

    for p in (retry_p, s5_p, ab_p, s1_p, mtc_p):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")

    retry, ab, mtc = load(retry_p), load(ab_p), load(mtc_p)
    s1 = load(s1_p)
    s5 = load(s5_p)

    write(os.path.join(out, "auto_exampville_cmp_rows.tex"),
          "source: runs/exampville_s5fix_20261030/exampville_metrics.json (job 27338231) "
          "-- first run to complete Stage 5 end-to-end after the OMX zone-mapping "
          "and matrix-listing fixes; supersedes runs/exampville_ab_27074291.aqua",
          tabular("l r r r r r r",
                  "Arm & Mode LL & Mode $\\rho^2$ & Dest LL & Dest $\\rho^2$ & Mode MAE & Dest MAE",
                  table12_rows(s5)))
    write(os.path.join(out, "auto_exampville_cmp01_rows.tex"),
          "source: Stage 0 runs/exampville_ab/exampville_metrics.json; "
          "Stage 1 runs/exampville_s1fix_20261029/exampville_metrics.json (job 27308422)",                  tabular("l r r r r r r",
                  "& \\multicolumn{4}{c}{Stage 0 (CDAP)} & "
                  "\\multicolumn{2}{c}{Stage 1 (frequency)} \\\\"
                  "\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}"
                  "\nArm & $k$ & CDAP LL & $\\Delta$LL & $\\Delta$BIC/$n$ & "
                  "Work MAE & Nonmand MAE",
                  table13_rows(ab, s1)))
    write(os.path.join(out, "auto_mtc_rows.tex"),
          "source: runs/mtc_ab/mtc_metrics.json",
          tabular("l r r r r r",
                  "Arm & $n_{\\mathrm{obs}}$ & $n_{\\mathrm{pred}}$ & Dest MAE & Dest top-10 & Trip-len MAE",
                  mtc_rows(mtc)))

    # Provenance echoes so a human can check the generator against the paper.
    print("wrote auto_exampville_cmp_rows.tex / auto_exampville_cmp01_rows.tex / auto_mtc_rows.tex")
    print(f"  Table12 E1 work-mode LL  = {s5['arms']['E1']['work_mode_model']['ll']}")
    print(f"  Table12 E2 work-mode LL  = {s5['arms']['E2']['work_mode_model']['ll']}")
    print(f"  Table13 E1/E2 CDAP LL    = {ab['arms']['E1_s0']['ll']} / {ab['arms']['E2_s0']['ll']}")
    print(f"  Table13 E1/E2 work MAE   = {s1['arms']['E1_s1'].get('work_mean_mae')} / "
          f"{s1['arms']['E2_s1'].get('work_mean_mae')}")
    print(f"  MTC    M1 n_pred         = {mtc['arms']['M1'].get('n_pred')}")


if __name__ == "__main__":
    main()
