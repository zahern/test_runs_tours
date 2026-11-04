#!/usr/bin/env python
"""post_selection_inference.py -- PS-I / PS-III / PS-IV for the paper.

Runs the search-aware uncertainty passes defined in the paper
(sec: post-selection inference) over the published Stage-5 specification:

  PS-I   sample splitting      -- search on half D1, report overlap with the
                                  published spec; a second search on D2 gives
                                  the split-sample view. (Conservatively
                                  implemented: two independent half-frames;
                                  the D1-selected spec is *not* force-refit on
                                  D2, so the split-sample t-ratios are read
                                  from the half's own leaderboard.)
  PS-III search-aware bootstrap -- resample persons within segment, re-run the
                                  search B times; report the selection
                                  frequency q_j and the empirical |t| critical
                                  value against the nominal z.
  PS-IV  subsample stability    -- repeat the search on person subsamples at
                                  rho in {0.5, 0.7, 0.9}; report the Jaccard
                                  overlap of the selected gene set with the
                                  published spec.

The search is restricted per replication through ``GA_PS_SUBSET_FILE`` read by
``_run_ga_shard.py`` (a CSV of person_ids); nothing else about the search is
changed, so ordinary runs are unaffected.

Because the replication searches are themselves GA runs, this script is a
*driver*: it prepares the resampled person-id files, shells out to
``_run_ga_shard.py`` once per replication, and merges the per-replication
winning gene specs. It runs on a compute node only (never the login node).

Layout (all under ``--ps-root``)::

    manifest.json                      segments, B, id-file paths
    published_features.json            {segment: [feature, ...]}
    ids/PS-I/<seg>/{d1,d2}.csv
    ids/PS-IV/<seg>/rho0.5.csv ...
    ids/PS-III/<seg>/boot_<b>.csv
    runs/<tag>/stage5/shards/<seg>/...   per-replication GA output
    out/<proto>_<seg>.json             per-replication feature sets
    tables/ps_diagnostics.csv

Usage::

    python post_selection_inference.py --prepare --run-tag TAG
    python post_selection_inference.py --protocol III --segment work --B 20
    python post_selection_inference.py --merge
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

SEGMENT_PURPOSES = {
    "work": {"work"},
    "education": {"edu"},
    "nonmandatory": {"social", "shopping", "escort", "othmaint", "eat",
                     "othdiscr", "home"},
}
SEG_KEYS = ("work", "education", "nonmandatory")
PS_RHO = (0.5, 0.7, 0.9)
NOMINAL_Z = 1.959964  # two-sided 5%


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _seg_of_person(tours):
    """Majority segment bucket per person from tour purposes.

    Handles both the synthetic token frame (``purpose`` in
    work/edu/social/...) and the HHTS `tours` frame, whose label column is
    ``main_purpose`` / ``primary_purpose`` with free-text values like
    'Direct Work Commute' or 'Shopping'.  Seg_of=0 (observed 21/10: the
    prepare silently found no ``purpose`` column and returned {}) starves
    every downstream replication.
    """
    if tours is None or len(tours) == 0:
        return {}
    t = tours.copy()
    if "person_id" not in t.columns:
        if "PERSID" in t.columns:
            t = t.rename(columns={"PERSID": "person_id"})
        else:
            return {}
    t["person_id"] = t["person_id"].astype(str)
    pcol = next((c for c in ("purpose", "main_purpose", "primary_purpose")
                 if c in t.columns), None)
    if pcol is None:
        return {}
    rev = {p: seg for seg, ps in SEGMENT_PURPOSES.items() for p in ps}
    purp = t[pcol].astype(str).str.strip().str.lower()
    t["seg"] = purp.map(rev)
    # Free-text HHTS labels: keyword buckets for the unmapped methods.
    _unmapped = t["seg"].isna() & (purp != "nan")
    t.loc[_unmapped & purp.str.contains("work", na=False), "seg"] = "work"
    t.loc[_unmapped & purp.str.contains(
        r"edu|school|childcare|kindergarten|class", na=False, regex=True),
        "seg"] = "education"
    # every other purpose text (shopping, recreation, pickup, personal
    # business, social, other) is nonmandatory travel.
    t.loc[t["seg"].isna() & (purp != "nan"), "seg"] = "nonmandatory"
    t = t.dropna(subset=["seg"])
    if t.empty:
        return {}
    mode = (t.groupby(["person_id", "seg"]).size()
              .reset_index(name="n")
              .sort_values("n", ascending=False)
              .drop_duplicates("person_id"))
    return dict(zip(mode["person_id"], mode["seg"]))


def _write_ids(ids, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"person_id": sorted(set(map(str, ids)))}).to_csv(path, index=False)


def _feature_set_from_specs(shard_dir):
    """Union of feature names from <shard_dir>/variants/winner_spec_*.json."""
    feats = set()
    for spec in glob.glob(os.path.join(shard_dir, "variants", "winner_spec_*.json")):
        obj = _json(spec, {})
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, (list, tuple)):
                    feats.update(map(str, v))
        elif isinstance(obj, list):
            feats.update(map(str, obj))
    return feats


def _feature_set_from_run(ps_root, tag, seg):
    base = os.path.join(ps_root, "runs", tag, "stage5", "shards", seg)
    return _feature_set_from_specs(base)


def _published_features(run_root, run_tag):
    """{segment: set(features)} from the published merged feature maps.

    The merged map is keyed like ``5/<purpose>/winner_spec_{mode,dest}.json``
    (e.g. ``5/edu/winner_spec_mode.json``) and each value carries the chosen
    gene list under a ``*_vars`` field (``mode_vars`` / ``dest_vars``).  The
    purpose token is mapped onto the three paper segments.  Structure/family
    lists (``structures_scored`` / ``families_scored``) are NOT features and are
    excluded -- the previous implementation wrongly swept every list in.
    """
    _KEY2SEG = {"work": "work", "edu": "education", "education": "education",
                "nonmand": "nonmandatory", "nonmandatory": "nonmandatory",
                "social": "nonmandatory", "shopping": "nonmandatory",
                "escort": "nonmandatory", "othmaint": "nonmandatory",
                "eat": "nonmandatory", "othdiscr": "nonmandatory",
                "home": "nonmandatory"}
    merged = _json(os.path.join(run_root, run_tag, "merged",
                                "shard_feature_maps.json"), {})
    out = {}
    for key, spec in merged.items():
        ks = str(key).replace("\\", "/")
        if not (ks.startswith("5/") or "/5/" in ks or "stage5" in ks.lower()):
            continue
        seg = None
        if isinstance(spec, dict) and spec.get("segment"):
            seg = _KEY2SEG.get(str(spec["segment"]).strip().lower())
        if seg is None and "/" in ks:
            seg = _KEY2SEG.get(ks.split("/")[1].lower())
        if seg is None:
            continue
        feats = set()
        if isinstance(spec, dict):
            for fname, val in spec.items():
                if isinstance(val, (list, tuple)) and str(fname).endswith("_vars"):
                    feats.update(map(str, val))
        elif isinstance(spec, list):
            feats.update(map(str, spec))
        out.setdefault(seg, set()).update(feats)
    return out


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def cmd_prepare(a):
    from pipeline_logger import load_data
    _, out, _ = load_data()
    tours = out.get("tours")
    if tours is None:
        tours = out.get("trips")
    seg_of = _seg_of_person(tours)
    persons = out.get("persons")
    pids = [str(p) for p in pd.unique(persons["person_id"])] \
        if persons is not None and "person_id" in persons.columns else list(seg_of)

    # The GA search runs on the estimation frame (out['tours']); the subset
    # files MUST use those person_ids, because _run_ga_shard.py filters
    # persons/trips/tours by them.  out['persons'] can be a different
    # population (NEW_POP synthetic ids), in which case the intersection is
    # empty and the old code silently wrote 0-person splits (observed 16/10:
    # "prepare work: 0 persons" -> every replication would search an empty
    # frame).
    overlap = len(set(pids) & set(seg_of)) if seg_of else 0
    if seg_of and overlap == 0 and pids:
        print(f"  [prepare] persons frame ({len(pids):,} ids) does not overlap "
              f"the tour frame ({len(seg_of):,} ids) — using tour person_ids "
              f"(the GA search frame)")
        pids = list(seg_of)
    if not seg_of and pids and persons is not None and "ptype" in persons.columns:
        _ptmap = persons.drop_duplicates("person_id")[["person_id", "ptype"]]
        _PT_SEG = {"FullTimeWorker": "work", "PartTimeWorker": "work",
                   "WorkingStudent": "work", "SchoolChild": "education",
                   "UniversityStudent": "education", "Preschool": "education",
                   "NonWorker": "nonmandatory", "Retired": "nonmandatory"}
        seg_of = {str(p): _PT_SEG[str(pt)] for p, pt in
                  zip(_ptmap["person_id"], _ptmap["ptype"])
                  if str(pt) in _PT_SEG}
        print(f"  [prepare] tours frame empty — segmented {len(seg_of):,} "
              f"persons by ptype (work/education/nonmandatory)")

    by_seg = {s: [p for p in pids if seg_of.get(p) == s] for s in SEG_KEYS}
    _tot = sum(len(v) for v in by_seg.values())
    if _tot == 0:
        _nt = 0 if tours is None else len(tours)
        _np = 0 if persons is None else len(persons)
        raise SystemExit(
            "FATAL: post-selection prepare produced 0 persons across all "
            "segments — refusing to write empty subsets (they would restrict "
            "every replication search to nothing). "
            f"tours={_nt} persons={_np} seg_of={len(seg_of)} pids={len(pids)}")
    rng = np.random.default_rng(20260116 + int(a.stage_id))
    ids_root = os.path.join(a.ps_root, "ids")
    manifest = {"stage_id": int(a.stage_id), "B": int(a.B), "segments": {}}

    for seg in SEG_KEYS:
        pool = np.array(by_seg[seg])
        rng.shuffle(pool)
        half = len(pool) // 2
        d1, d2 = pool[:half], pool[half:]
        _write_ids(d1, os.path.join(ids_root, "PS-I", seg, "d1.csv"))
        _write_ids(d2, os.path.join(ids_root, "PS-I", seg, "d2.csv"))
        seg_manifest = {"n_persons": int(len(pool)),
                        "d1": os.path.join(ids_root, "PS-I", seg, "d1.csv"),
                        "d2": os.path.join(ids_root, "PS-I", seg, "d2.csv"),
                        "PS-IV": {}, "PS-III": []}
        for rho in PS_RHO:
            k = max(1, int(round(len(pool) * rho)))
            sub = pool[:k]
            p = os.path.join(ids_root, "PS-IV", seg, f"rho{rho}.csv")
            _write_ids(sub, p)
            seg_manifest["PS-IV"][str(rho)] = p
        for b in range(int(a.B)):
            brng = np.random.default_rng(90210 + b)
            draw = brng.choice(pool, size=len(pool), replace=True) if len(pool) else pool
            p = os.path.join(ids_root, "PS-III", seg, f"boot_{b:03d}.csv")
            _write_ids(draw, p)
            seg_manifest["PS-III"].append(p)
        manifest["segments"][seg] = seg_manifest
        print(f"prepare {seg}: {len(pool)} persons, D1={len(d1)} D2={len(d2)}, "
              f"B={manifest['B']}")

    pub = {k: sorted(v) for k, v in _published_features(a.run_root, a.run_tag).items()}
    _dump(pub, os.path.join(a.ps_root, "published_features.json"))
    _dump(manifest, os.path.join(a.ps_root, "manifest.json"))
    print(f"prepared -> {a.ps_root}  (published features for {len(pub)} segments)")


# ---------------------------------------------------------------------------
# one (protocol, segment)
# ---------------------------------------------------------------------------

def _id_files_for(a, proto, seg):
    man = _json(os.path.join(a.ps_root, "manifest.json"), {})
    sm = (man.get("segments") or {}).get(seg, {})
    if proto == "I":
        return [("d1", sm.get("d1")), ("d2", sm.get("d2"))]
    if proto == "IV":
        return [(f"rho{r}", p) for r, p in sorted((sm.get("PS-IV") or {}).items())]
    return [(f"boot{i:03d}", p) for i, p in enumerate(sm.get("PS-III") or [])]


# Paper segment name -> shard stage-5 key (ga_shard.STAGE_SEGMENTS[5] =
# ['work', 'edu', 'nonmand']).  22/10 bug: the driver passed paper names into
# _run_ga_shard.py, so every non-"work" replication died with
# "FATAL: segment 'education' not valid for stage 5" while its array slot
# still exited 0 — empty feature sets for 6/9 slots.
_SEG2SHARD = {"work": "work", "education": "edu", "nonmandatory": "nonmand"}


def cmd_protocol(a):
    files = [(lbl, p) for lbl, p in _id_files_for(a, a.protocol, a.segment) if p]
    if not files:
        print(f"no id files for {a.protocol}/{a.segment} -- run --prepare first")
        return 2
    shard_seg = _SEG2SHARD.get(str(a.segment).strip().lower(), a.segment)
    out = {}
    n_failed = 0
    for lbl, idfile in files:
        tag = f"ps_{a.protocol.lower()}_{a.segment}_{lbl}"
        env = dict(os.environ)
        env["GA_PS_SUBSET_FILE"] = idfile
        env["GA_BUDGET"] = a.budget
        env.setdefault("PIPELINE_RUN_ROOT", a.run_root)
        cmd = [sys.executable, "-u", "_run_ga_shard.py",
               "--stage-id", str(a.stage_id), "--segment", shard_seg,
               "--run-root", os.path.join(a.ps_root, "runs"), "--run-tag", tag]
        print(f"[{a.protocol}/{a.segment}/{lbl}] {' '.join(cmd)}  (subset={idfile})")
        rc = 1
        try:
            rc = subprocess.run(cmd, env=env, check=False,
                                cwd=os.path.dirname(os.path.abspath(__file__))).returncode
        except Exception as exc:  # noqa: BLE001
            print(f"  search failed: {exc}")
        feats = _feature_set_from_run(a.ps_root, tag, shard_seg)
        if rc != 0:
            n_failed += 1
            print(f"  {lbl}: shard rc={rc} (treated as failed replication)")
        elif not feats:
            n_failed += 1
            print(f"  {lbl}: succeeded but selected 0 features")
        out[lbl] = sorted(feats)
        print(f"  {lbl}: {len(feats)} features")
    _dump(out, os.path.join(a.ps_root, "out", f"{a.protocol}_{a.segment}.json"))
    if n_failed and n_failed == len(files):
        print(f"FATAL: all {len(files)} replications failed for "
              f"{a.protocol}/{a.segment} — not recording a usable result")
        return 2
    return 0 if n_failed < len(files) else 2


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def cmd_merge(a):
    pub = _json(os.path.join(a.ps_root, "published_features.json"), {})
    rows = []
    for proto in ("I", "III", "IV"):
        for seg in SEG_KEYS:
            p = os.path.join(a.ps_root, "out", f"{proto}_{seg}.json")
            if not os.path.isfile(p):
                continue
            res = _json(p, {})
            gstar = set(pub.get(seg, []))
            sets = [set(v) for v in res.values()]
            if not sets:
                continue
            if proto == "III":
                terms = sorted(set().union(*sets) | gstar)
                for term in terms:
                    q = float(np.mean([1.0 if term in s else 0.0 for s in sets]))
                    rows.append({"protocol": proto, "segment": seg, "term": term,
                                 "metric": "selection_freq_q", "value": q,
                                 "n_reps": len(sets)})
                rows.append({"protocol": proto, "segment": seg, "term": "*",
                             "metric": "mean_jaccard_vs_published",
                             "value": float(np.mean([_jaccard(s, gstar) for s in sets])),
                             "n_reps": len(sets)})
            else:
                jac = float(np.mean([_jaccard(s, gstar) for s in sets]))
                rows.append({"protocol": proto, "segment": seg, "term": "*",
                             "metric": "mean_jaccard_vs_published", "value": jac,
                             "n_reps": len(sets)})
                if proto == "I" and len(sets) == 2:
                    rows.append({"protocol": proto, "segment": seg, "term": "*",
                                 "metric": "d1_d2_jaccard",
                                 "value": _jaccard(sets[0], sets[1]),
                                 "n_reps": 2})
    if not rows:
        print("merge: no per-protocol outputs found")
        return 2
    tab_dir = os.path.join(a.ps_root, "tables")
    os.makedirs(tab_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(tab_dir, "ps_diagnostics.csv")
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(df)} rows)")
    tex = os.path.join(tab_dir, "auto_ps.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by post_selection_inference.py -- DO NOT EDIT.\n")
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write("\\caption{Post-selection diagnostics (PS-I/III/IV).}\\label{tab:auto_ps}\n")
        f.write("\\begin{tabular}{l l l r}\\toprule\n")
        f.write("Protocol & Segment & Metric & Value \\\\ \\midrule\n")
        for _, r in df.iterrows():
            f.write(f"{r['protocol']} & {r['segment']} & "
                    f"{str(r['metric']).replace('_', chr(92)+'_')} & "
                    f"{r['value']:.3f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(f"wrote {tex}")
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--ps-root", default="runs/ps_inference")
    ap.add_argument("--stage-id", type=int, default=5)
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--budget", default="smoke")
    ap.add_argument("--protocol", choices=["I", "III", "IV"])
    ap.add_argument("--segment", choices=list(SEG_KEYS))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    if a.prepare:
        cmd_prepare(a)
        return 0
    if a.merge:
        return cmd_merge(a)
    if a.protocol and a.segment:
        return cmd_protocol(a)
    print("nothing to do: pass --prepare, --merge, or --protocol with --segment")
    return 2


if __name__ == "__main__":
    sys.exit(main())
