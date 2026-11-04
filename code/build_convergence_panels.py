"""Rebuild the QLD appendix convergence panels from each shard's own history.

Why this replaces the old panels
--------------------------------
The old figures were copied from ``plots/ga_stageN.png``.  Those plots switched
to a "proposed candidates" band whenever the search recorded proposal ranges,
and in that mode they *replaced* the population mean with a band of every
proposed (mostly rejected) chromosome.  On several stages the proposals sat far
above the incumbent, so the panel showed a flat best-so-far line under a wide
band that never narrowed --- visually indistinguishable from a search that
never converged, even where the history shows a clean plateau.

This builder plots what the search actually recorded, per generation:

  top     best-so-far BIC/n (bold), population mean (dashed), and the
          population spread (mean -- worst, clipped)
  bottom  population diversity

and adds an explicit verdict computed from the incumbent trajectory:
converged / plateau / budget-limited / smoke (not a convergence result).

Outputs (under abm_safety/preprint):
  figures/appendix/conv_s<stage>_<segment>.png
  convergence_audit.csv
  convergence_verdicts.json

Usage (on the HPC login node, from code/):
  TSP_ALLOW_HEAD_PY=1 conda run -n zigenv python build_convergence_panels.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(HERE, "runs", "ga_parallel")
PRE = os.path.abspath(os.path.join(HERE, "..", "abm_safety", "preprint"))
FIGDIR = os.path.join(PRE, "figures", "appendix")

PRIMARY = "qld_conv1"
# Order matters: first tag with a usable history wins for each segment.
FALLBACKS = [
    "qld_shard4_retry_20260924",
    "qld_s4fix_20261102",
    "qld_shard2",
    "qld_shard1",
]
TAGS = [PRIMARY] + [t for t in FALLBACKS if t != PRIMARY]

EXPECTED = {
    0: ["all"],
    1: ["work", "edu", "nonmand"],
    2: ["start", "end", "budget"],
    3: ["work", "edu", "nonmand"],
    4: ["work", "edu", "escort", "shopping", "othmaint", "othdiscr",
        "eat", "social"],
}

STAGE_NAMES = {
    0: "CDAP",
    1: "Activity frequency",
    2: "Day budget",
    3: "Activity duration",
    4: "Schedule placement",
}
SEG_LABEL = {
    "all": "all persons",
    "nonmand": "non-mandatory (combined)",
    "othmaint": "other maintenance",
    "othdiscr": "other discretionary",
    "eat": "eat out",
    "start": "start of day",
    "end": "end of day",
    "budget": "day budget",
}

COL_BEST = "#3d3a9e"
COL_GEN = "#9c99e8"
COL_MEAN = "#666666"
COL_FILL = "#c98a3a"


def _read_csv(path: str) -> List[Dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _float(v, default=float("nan")) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _history_path(sid: int, seg: str, sdir: str) -> Optional[str]:
    rep = os.path.join(sdir, "ga_reports")
    cands = [os.path.join(rep, f"ga_stage{sid}_segment_history.csv"),
             os.path.join(rep, f"ga_stage{sid}_history.csv")]
    for p in cands:
        rows = _read_csv(p)
        if rows:
            return p
    return None


def find_shard(sid: int, seg: str) -> Optional[Tuple[str, str, str]]:
    """Return (tag, shard_dir, history_path) for the first usable artifact."""
    for tag in TAGS:
        sdir = os.path.join(RUN_ROOT, tag, f"stage{sid}", "shards", seg)
        if not os.path.isdir(sdir):
            continue
        hp = _history_path(sid, seg, sdir)
        if hp:
            return tag, sdir, hp
    return None


def _select_rows(rows: List[Dict[str, str]], seg: str) -> List[Dict[str, str]]:
    """Keep the rows for one segment when the history is pooled."""
    if not rows or "segment" not in rows[0]:
        return rows
    names = {r.get("segment", "") for r in rows}
    if len(names) <= 1:
        return rows
    # Exact-ish match on the segment token; fall back to all rows (pooled).
    want = seg.lower()
    aliases = {
        "edu": ("edu", "education", "school"),
        "nonmand": ("non-mandatory", "nonmand", "non mandatory"),
        "othmaint": ("oth-maint", "othmaint", "other maint"),
        "othdiscr": ("oth-discr", "othdiscr", "other discr"),
        "eat": ("eat",),
    }.get(want, (want,))
    picked = [r for r in rows
              if any(a in str(r.get("segment", "")).lower() for a in aliases)]
    return picked or rows


def load_history(sid: int, seg: str, path: str) -> List[Dict[str, float]]:
    rows = _select_rows(_read_csv(path), seg)
    out: List[Dict[str, float]] = []
    for r in rows:
        g = _float(r.get("generation"))
        if not math.isfinite(g):
            continue
        out.append({
            "generation": int(g),
            "best": _float(r.get("best_fitness")),
            "best_so_far": _float(r.get("best_so_far", r.get("best_fitness"))),
            "mean": _float(r.get("mean_fitness")),
            "worst": _float(r.get("worst_fitness")),
            "diversity": _float(r.get("diversity")),
            "stagnation": _float(r.get("stagnation")),
        })
    out.sort(key=lambda d: d["generation"])
    # Collapse duplicate generation rows (restart overlays) to the best incumbent.
    dedup: Dict[int, Dict[str, float]] = {}
    for r in out:
        g = int(r["generation"])
        if g not in dedup or r["best_so_far"] > dedup[g]["best_so_far"]:
            dedup[g] = r
    return [dedup[g] for g in sorted(dedup)]


def diagnose(hist: List[Dict[str, float]]) -> Dict[str, object]:
    if not hist:
        return {"verdict": "no history retained", "converged": False,
                "n_generations": 0, "last_improvement": None,
                "flat_generations": 0, "bic_start": None, "bic_final": None}
    best = np.array([h["best_so_far"] for h in hist], dtype=float)
    bic = -best
    n = len(hist)
    # A strict improvement is a decrease in BIC/n greater than rounding noise.
    improves = [i for i in range(1, n) if bic[i] < bic[i - 1] - 1e-9]
    last = improves[-1] if improves else 0
    flat = n - 1 - last
    # The engine's own stagnation counter is the stop criterion.  It is not the
    # same as ``flat`` when a restart resets the counter, so prefer it; a run
    # that ended at its generation cap has a counter below the patience and is
    # not certified converged even if its incumbent happens to be flat.
    final_stag = int(hist[-1].get("stagnation", 0) or 0)
    if n <= 3:
        verdict = (f"NOT CONVERGED --- smoke budget: {n} generations "
                   f"(search stopped before any plateau can be assessed)")
        converged = False
    elif final_stag >= 45:
        verdict = (f"Converged under the search's stagnation rule: last "
                   f"improvement at generation {last}, then {flat} flat "
                   f"generations (stagnation counter {final_stag}; {n} total)")
        converged = True
    elif flat >= 5:
        verdict = (f"Budget-limited plateau, not a stagnation stop: {n} "
                   f"generations, last improvement at {last}, {flat} flat, "
                   f"stagnation counter {final_stag}")
        converged = False
    else:
        verdict = (f"NOT CONVERGED: still improving at the final generation "
                   f"({n} generations, last improvement at {last})")
        converged = False
    return {
        "verdict": verdict,
        "converged": bool(converged),
        "n_generations": n,
        "last_improvement": int(last),
        "flat_generations": int(flat),
        "final_stagnation": final_stag,
        "bic_start": float(bic[0]),
        "bic_final": float(bic[-1]),
        "n_improvements": len(improves),
    }


def make_panel(sid: int, seg: str, hist: List[Dict[str, float]],
               diag: Dict[str, object], out_path: str) -> None:
    g = np.array([h["generation"] for h in hist], dtype=float)
    # The history writes the generation winner into "best" after the incumbent
    # update, so best == best_so_far on every row; plotting both would hide the
    # second line exactly under the first.
    bsf = -np.array([h["best_so_far"] for h in hist], dtype=float)
    mean = -np.array([h["mean"] for h in hist], dtype=float)
    worst = -np.array([h["worst"] for h in hist], dtype=float)
    div = np.array([h["diversity"] for h in hist], dtype=float)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 4.4), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.05], "hspace": 0.08},
        facecolor="white")

    # Population spread: mean to worst, clipped by the y-limit below.  The
    # incumbent and mean set the view; the worst tail is allowed to clip rather
    # than compress the convergence signal into a sliver.
    finite = np.isfinite(mean) & np.isfinite(worst)
    if finite.any():
        ax.fill_between(g[finite], mean[finite], worst[finite], color=COL_FILL,
                        alpha=0.10, linewidth=0, label="population spread (mean--worst)")

    marker = "o" if len(g) <= 5 else None
    ax.plot(g, mean, color=COL_MEAN, lw=1.15, ls="--", marker=marker,
            ms=4, label="population mean")
    ax.plot(g, bsf, color=COL_BEST, lw=2.4, marker=marker, ms=4.5,
            label="best-so-far")

    core = np.concatenate([bsf[np.isfinite(bsf)], mean[np.isfinite(mean)]])
    if core.size:
        lo, hi = float(np.min(core)), float(np.max(core))
        pad = max(0.08 * (hi - lo), 0.01)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_ylabel("BIC / n   (lower is better)", fontsize=9.5)
    ax.grid(alpha=0.22, lw=0.6)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right", ncol=2)
    title = f"Stage {sid} \u00b7 {STAGE_NAMES[sid]} \u00b7 {SEG_LABEL.get(seg, seg)}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=21)
    colour = "#1b7f3a" if diag["converged"] else "#b3261e"
    ax.text(0.5, 1.012, str(diag["verdict"]), transform=ax.transAxes,
            fontsize=7.6, color=colour, va="bottom", ha="center")

    if finite.any():
        ax2.plot(g, div, color="#4f6d7a", lw=1.2)
    ax2.set_ylabel("diversity", fontsize=8.5)
    ax2.set_xlabel("Generation", fontsize=9)
    ax2.grid(alpha=0.22, lw=0.6)
    ax2.tick_params(labelsize=8)
    ax.tick_params(labelsize=8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    os.makedirs(FIGDIR, exist_ok=True)
    audit: List[Dict[str, object]] = []
    verdicts: Dict[str, Dict[str, object]] = {}
    for sid, segments in EXPECTED.items():
        for seg in segments:
            found = find_shard(sid, seg)
            key = f"s{sid}_{seg}"
            if not found:
                print(f"[missing] Stage {sid}/{seg}: no usable history")
                continue
            tag, sdir, hp = found
            hist = load_history(sid, seg, hp)
            if not hist:
                print(f"[missing] Stage {sid}/{seg}: empty history in {hp}")
                continue
            diag = diagnose(hist)
            diag.update({"stage": sid, "segment": seg, "run_tag": tag,
                         "history": os.path.relpath(hp, HERE)})
            out = os.path.join(FIGDIR, f"conv_s{sid}_{seg}.png")
            make_panel(sid, seg, hist, diag, out)
            verdicts[key] = diag
            audit.append(diag)
            print(f"[ok] s{sid}/{seg:<9} {tag:<28} gens={diag['n_generations']:>4} "
                  f"last={diag['last_improvement']:>4} flat={diag['flat_generations']:>4} "
                  f"{'CONVERGED' if diag['converged'] else 'NOT CONVERGED'}")

    csv_path = os.path.join(PRE, "convergence_audit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        cols = ["stage", "segment", "run_tag", "history", "n_generations",
                "n_improvements", "last_improvement", "flat_generations",
                "final_stagnation", "bic_start", "bic_final", "converged",
                "verdict"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(audit)
    with open(os.path.join(PRE, "convergence_verdicts.json"), "w",
              encoding="utf-8") as fh:
        json.dump(verdicts, fh, indent=2)
    print(f"\nwrote {len(audit)} panels, {csv_path}, convergence_verdicts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
