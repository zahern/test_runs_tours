"""
ga_shard.py
===========
Micro-job (shard) spec for the cooperative GA search.

A shard = one (stage, segment) search that can run as its own small PBS job.
Shards cooperate through files, never through shared memory:

  <run_root>/<run_tag>/stage<N>/shards/<shard_id>/
      ga_results.pkl          per-shard AllStagesResult (1 segment inside)
      ga_reports/             per-shard history CSVs + best-model txt
      plots/                  per-shard convergence plot
      variants/               model-family leaderboard for this segment
          leaderboard.csv     one row per refit variant (LL, k, n, BIC/n, ...)
          leaderboard.html    same, human-readable
          best_model_<variant>.pkl   fitted model objects per variant
          winner.json         winning variant + why (comparability group)

`merge_ga_shards.py` unions every shard into the legacy
`merged/ga_results.pkl` that prediction (`GA_STAGED_GA_DIR=.../merged`)
already consumes, and picks the winning model family per segment from
the leaderboard (best BIC/n *within each comparability group*).

Shard axes (matches the searches already in the code):

  stage 0 : [all]            CDAP, single search (no split needed)
  stage 1 : work edu nonmand (`stage1_activity_frequency.PURPOSES` pooled to 3)
  stage 2 : start end budget (`start_of_day / end_of_day / B`)
  stage 3 : work edu nonmand (pooled; see `_run_ga_stage3`)
    stage 4 : work edu nonmand (all six non-mandatory purposes pooled)
    stage 5 : work edu nonmand (mode+dest searched together per segment)
    stage 51: work edu escort shopping othmaint othdiscr eat social

Model-variant arms (refits on the searched feature set, NOT new searches):

  stage 4 : rrm (default) + tdd (joint 21-alt TDDChoiceModel).
            Optional: mnl, ordered, hybrid via GA_S4_VARIANTS.
  stage 5 mode : nl (nested Motor/NonMotor, default) + mnl (flat).
            Optional: mxl, gse via GA_S5_MODE_VARIANTS.
  stage 5 dest : mnl (default) + nl_dist (distance-band NL).
            Optional diagnostic: exploded (rank-ordered/exploded logit,
            likelihood differs -> flagged non-comparable).
            `ordered` is intentionally NOT offered for destination choice:
            zone alternatives are unordered, so an ordered logit has no
            valid dependent variable there. Ordered logit IS offered for
            stage 4 (periods are ordinal) and already used in stage 1.

Comparability rule (this is what "gets the best model" means):
  BIC/n may only decide between variants that share the same dependent
  variable and choice set. Each variant declares a comparability group;
  the winner is picked per group. Cross-group rows are reported side by
  side but never auto-compared. The merge records the winner of the
  PRIMARY group (the one prediction uses) plus every group's winner.
"""

from __future__ import annotations

import csv
import json
import os


# ---------------------------------------------------------------------------
# Shard axes
# ---------------------------------------------------------------------------

#: Search segments per stage. Keys must match the segment keys used in
#: `ga_hypothesis_testing` feature maps / checkpoint labels (lower-cased
#: matching is applied, so 'Work'/'work' both work).
STAGE_SEGMENTS = {
    0: ["all"],
    1: ["work", "edu", "nonmand"],
    2: ["start", "end", "budget"],
    3: ["work", "edu", "nonmand"],
    4: ["work", "edu", "nonmand"],
    5: ["work", "edu", "nonmand"],
    51: ["work", "edu", "escort", "shopping", "othmaint", "othdiscr",
         "eat", "social"],
}

#: Suffix fragments that identify a segment inside a checkpoint label such as
#: 'Stage4   Work' or 'Work (tour)   Mode'. Used to map labels -> shard keys.
SEGMENT_ALIASES = {
    "work": ["work"],
    "edu": ["edu", "school"],
    "nonmand": ["non-mandatory", "nonmand", "nonmandary",
                # stage-51 trip labels carry the raw purpose names
                "social", "shopping", "escort", "eat", "othmaint",
                "othdiscr"],
    "start": ["start-of-day", "start"],
    "end": ["end-of-day", "end"],
    "budget": ["day duration (b)", "budget"],
    "escort": ["escort"],
    "shopping": ["shopping"],
    "social": ["social"],
    "othmaint": ["oth-maint", "othmaint", "other maintenance"],
    "othdiscr": ["oth-discr", "othdiscr", "other discretionary"],
    "eat": ["eat out", "eat"],
    "all": ["all"],
}


def _env_list(name: str, default: str) -> list:
    raw = os.environ.get(name, default)
    return [t.strip().lower() for t in str(raw).replace(";", ",").split(",")
            if t.strip()]


def shard_segments(stage: int) -> list:
    """Segments this job should run (GA_SHARD_SEGMENT filter, default all)."""
    filt = _env_list("GA_SHARD_SEGMENT", "all")
    if filt == ["all"] or not filt:
        return list(STAGE_SEGMENTS.get(stage, ["all"]))
    valid = STAGE_SEGMENTS.get(stage, ["all"])
    return [s for s in valid if s in filt]


def shard_keep(stage: int, segment_key: str) -> bool:
    """True when a (stage, segment) belongs in this job's shard."""
    filt = _env_list("GA_SHARD_SEGMENT", "all")
    if filt == ["all"] or not filt:
        return True
    return segment_key.strip().lower() in filt


# ---------------------------------------------------------------------------
# Model-variant arms
# ---------------------------------------------------------------------------

S4_VARIANTS_DEFAULT = ["rrm", "tdd"]
S4_VARIANTS_ALL = ["rrm", "tdd", "mnl", "ordered", "hybrid", "mxl"]
S5_MODE_VARIANTS_DEFAULT = ["nl", "mnl"]
S5_MODE_VARIANTS_ALL = ["nl", "mnl", "mxl", "gse"]
S5_DEST_VARIANTS_DEFAULT = ["mnl", "nl_dist"]
S5_DEST_VARIANTS_ALL = ["mnl", "nl_dist", "exploded"]


def s4_variants() -> list:
    """Stage-4 refit variants for this shard (GA_S4_VARIANTS, default rrm+tdd)."""
    asked = _env_list("GA_S4_VARIANTS", ",".join(S4_VARIANTS_DEFAULT))
    if asked == ["all"]:
        return list(S4_VARIANTS_ALL)
    return [v for v in asked if v in S4_VARIANTS_ALL] or list(S4_VARIANTS_DEFAULT)


def s5_mode_variants() -> list:
    asked = _env_list("GA_S5_MODE_VARIANTS", ",".join(S5_MODE_VARIANTS_DEFAULT))
    if asked == ["all"]:
        return list(S5_MODE_VARIANTS_ALL)
    return [v for v in asked if v in S5_MODE_VARIANTS_ALL] or list(S5_MODE_VARIANTS_DEFAULT)


def s5_dest_variants() -> list:
    asked = _env_list("GA_S5_DEST_VARIANTS", ",".join(S5_DEST_VARIANTS_DEFAULT))
    if asked == ["all"]:
        return list(S5_DEST_VARIANTS_ALL)
    return [v for v in asked if v in S5_DEST_VARIANTS_ALL] or list(S5_DEST_VARIANTS_DEFAULT)


#: Comparability group per (stage, sub-model, variant). Winners are only
#: picked *within* a group. Groups sharing a dependent variable + choice
#: set are directly BIC-comparable; the rest are diagnostic side-by-side.
VARIANT_GROUPS = {
    ("s4", "rrm"): "s4-period",
    ("s4", "mnl"): "s4-period",
    ("s4", "ordered"): "s4-period",
    ("s4", "hybrid"): "s4-period",
    ("s4", "mxl"): "s4-period-mxl",
    ("s4", "tdd"): "s4-joint",
    ("s5mode", "nl"): "s5mode",
    ("s5mode", "mnl"): "s5mode",
    ("s5mode", "mxl"): "s5mode-mxl",
    ("s5mode", "gse"): "s5mode-mxl",
    ("s5dest", "mnl"): "s5dest",
    ("s5dest", "nl_dist"): "s5dest",
    ("s5dest", "exploded"): "s5dest-diag",
}

#: The group whose winner feeds prediction for each model slot.
PRIMARY_GROUP = {
    "s4": "s4-period",
    "s5mode": "s5mode",
    "s5dest": "s5dest",
}

VARIANT_DESCRIPTIONS = {
    ("s4", "rrm"): "SearchLibrium random-regret period choice (6 periods)",
    ("s4", "tdd"): "Joint (start,end,duration) TDD choice (21 alts; own choice set)",
    ("s4", "mnl"): "Plain MNL period choice",
    ("s4", "ordered"): "Ordered logit over periods (periods are ordinal)",
    ("s4", "hybrid"): "Hybrid RUM-RRM regret choice",
    ("s5mode", "nl"): "Nested logit Motor/NonMotor (default structure)",
    ("s5mode", "mnl"): "Flat MNL (no nests)",
    ("s5mode", "mxl"): "Mixed logit, random coefficients (needs converged base)",
    ("s5mode", "gse"): "GSE mixed logit (systematic gamma*g loadings + mixtures)",
    ("s5dest", "mnl"): "Sampled-choice MNL destination (default)",
    ("s5dest", "nl_dist"): "Distance-band nested-logit destination benchmark",
    ("s5dest", "exploded"): "Exploded/rank-ordered logit DIAGNOSTIC (different "
                            "likelihood: reported, never auto-compared)",
}


def variant_group(slot: str, variant: str) -> str:
    return VARIANT_GROUPS.get((slot, variant), f"{slot}-other")


def bic_per_obs(ll: float, k: int, n: int) -> float:
    """BIC/n (lower = better). Returns +inf when inputs are invalid."""
    import math
    try:
        if not math.isfinite(ll) or n is None or n <= 0 or k is None or k < 0:
            return float("inf")
        return (-2.0 * float(ll) + float(k) * math.log(float(n))) / float(n)
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Leaderboard IO
# ---------------------------------------------------------------------------

LEADERBOARD_FIELDS = [
    "stage", "segment", "slot", "variant", "group", "primary_group",
    "converged", "ll", "k", "n", "bic_per_obs", "winner_in_group",
    "overall_pick", "note",
]


def pick_winners(rows: list) -> list:
    """Mark winner_in_group (best BIC/n per group) + overall_pick.

    overall_pick = the winner of the PRIMARY group for each
    (stage, segment, slot). Rows that did not converge can never win.
    """
    out = [dict(r) for r in rows]
    groups: dict = {}
    for i, r in enumerate(out):
        if r.get("converged") and _finite(r.get("bic_per_obs")):
            groups.setdefault(
                (r.get("stage"), r.get("segment"), r.get("group")), []).append(i)
    group_winner = {}
    for key, idxs in groups.items():
        best = min(idxs, key=lambda i: float(out[i]["bic_per_obs"]))
        group_winner[key] = best
        out[best]["winner_in_group"] = 1
    for r in out:
        r.setdefault("winner_in_group", 0)
        prim = r.get("primary_group")
        key = (r.get("stage"), r.get("segment"), r.get("group"))
        r["overall_pick"] = 1 if (prim == r.get("group")
                                  and group_winner.get(key) is not None
                                  and out[group_winner[key]] is r) else 0
    return out


def _finite(x) -> bool:
    try:
        import math
        return math.isfinite(float(x))
    except Exception:
        return False


def write_leaderboard(rows: list, out_dir: str) -> str:
    """Write leaderboard.csv + leaderboard.html + winner.json. Returns csv path."""
    os.makedirs(out_dir, exist_ok=True)
    rows = pick_winners(rows)
    csv_path = os.path.join(out_dir, "leaderboard.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEADERBOARD_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in LEADERBOARD_FIELDS})
    html_path = os.path.join(out_dir, "leaderboard.html")
    _write_leaderboard_html(rows, html_path)
    winners = {}
    for r in rows:
        if r.get("overall_pick"):
            winners[f"{r.get('stage')}/{r.get('segment')}/{r.get('slot')}"] = {
                "variant": r.get("variant"),
                "group": r.get("group"),
                "bic_per_obs": r.get("bic_per_obs"),
            }
    with open(os.path.join(out_dir, "winner.json"), "w", encoding="utf-8") as fh:
        json.dump(winners, fh, indent=2)
    return csv_path


def _write_leaderboard_html(rows: list, path: str) -> None:
    def _fmt(v):
        try:
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                return str(v)
            return f"{f:.4f}"
        except Exception:
            return str(v)
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>"
             "<title>Variant leaderboard</title>"
             "<style>body{font-family:Arial,sans-serif;margin:20px;font-size:13px}"
             "table{border-collapse:collapse;width:100%}"
             "th{background:#2c3e50;color:white;padding:6px 10px;text-align:left}"
             "td{border:1px solid #ddd;padding:5px 10px}"
             ".win{background:#e8f8e8;font-weight:bold}</style></head><body>"
             "<h1>Model-variant leaderboard (BIC/n, lower = better)</h1>"
             "<p>Winners are picked <b>within</b> each comparability group only. "
             "Cross-group BIC/n values are shown side by side, never auto-compared.</p>"
             "<table><tr>" + "".join(f"<th>{c}</th>" for c in LEADERBOARD_FIELDS)
             + "</tr>"]
    for r in sorted(rows, key=lambda d: (str(d.get("stage")), str(d.get("segment")),
                                         str(d.get("slot")), str(d.get("group")))):
        cls = " class='win'" if r.get("overall_pick") or r.get("winner_in_group") else ""
        parts.append("<tr%s>" % cls + "".join(
            f"<td>{_fmt(r.get(c, ''))}</td>" for c in LEADERBOARD_FIELDS) + "</tr>")
    parts.append("</table></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def read_leaderboards(glob_paths: list) -> list:
    """Read many leaderboard.csv files into one row list."""
    import glob as _glob
    rows = []
    for pat in glob_paths:
        for path in _glob.glob(pat):
            try:
                with open(path, newline="", encoding="utf-8") as fh:
                    rows.extend(list(csv.DictReader(fh)))
            except Exception:
                continue
    return rows
