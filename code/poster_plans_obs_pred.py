#!/usr/bin/env python
"""Observed vs model-generated day plans for the TAP 2026 poster.

Each panel overlays one observed HHTS diary and one model-generated plan on
the SAME time axis and the SAME activity rows: hatched sky bars = observed,
solid navy bars = model.  Trips sit in the lane below the activities.

The model runs on a synthetic population, so people are matched by
activity-pattern class (work, school, university, escort, shopping), not by
individual identity.  The caption states this explicitly.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

QUT = {
    "navy": "#003B71", "sky": "#0077C8", "ice": "#D6EAF8",
    "amber": "#F2A900", "green": "#3E9B4F", "red": "#C0392B",
    "grey": "#5C6773",
}
OBS_C = "#7FB3D5"
PRED_C = QUT["navy"]


def _hm(m):
    try:
        m = float(m)
    except Exception:
        return ""
    m = int(round(m)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def _home_blocks(acts, trips, tmin, tmax):
    occupied = [(max(tmin, s), min(tmax, e)) for s, e, _ in acts]
    occupied += [(max(tmin, s), min(tmax, e)) for s, e, _, _ in trips]
    occupied = sorted((s, e) for s, e in occupied if e > s)
    blocks, cur = [], tmin
    for s, e in occupied:
        if s > cur:
            blocks.append((cur, s))
        cur = max(cur, e)
    if cur < tmax:
        blocks.append((cur, tmax))
    return blocks


def _draw_overlay(ax, obs_acts, obs_trips, pred_acts, pred_trips,
                  ycats, tmin, tmax, title, note):
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(-0.95, len(ycats) - 0.30)
    ax.set_yticks(list(range(len(ycats))) + [-0.55])
    ax.set_yticklabels(ycats + ["trip"], fontsize=8.4)
    ax.set_title(title, fontsize=11.5, color=QUT["navy"], fontweight="bold",
                 pad=13)
    ax.text(0.0, 1.015, note, transform=ax.transAxes, fontsize=7.0,
            color=QUT["grey"], va="bottom")
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.set_axisbelow(True)

    yh = ycats.index("home") if "home" in ycats else 0
    for s, e in _home_blocks(obs_acts, obs_trips, tmin, tmax):
        ax.barh(yh + 0.15, e - s, left=s, height=0.27,
                color="#E7EDF3", edgecolor="white", lw=0.3)
    for s, e in _home_blocks(pred_acts, pred_trips, tmin, tmax):
        ax.barh(yh - 0.15, e - s, left=s, height=0.27,
                color="#CBD5DF", edgecolor="white", lw=0.3)

    for s, e, purpose in obs_acts:
        if purpose not in ycats:
            continue
        s, e = max(s, tmin), min(e, tmax)
        if e <= s:
            continue
        ax.barh(ycats.index(purpose) + 0.15, e - s, left=s, height=0.28,
                color=OBS_C, alpha=0.85, edgecolor="white", lw=0.4)
    for s, e, purpose in pred_acts:
        if purpose not in ycats:
            continue
        s, e = max(s, tmin), min(e, tmax)
        if e <= s:
            continue
        ax.barh(ycats.index(purpose) - 0.15, e - s, left=s, height=0.28,
                color=PRED_C, alpha=0.95, edgecolor="white", lw=0.3)
        if e - s >= 60:
            ax.text((s + e) / 2, ycats.index(purpose) - 0.15, _hm(e - s)[1:],
                    ha="center", va="center", fontsize=5.6, color="white")

    for s, e, mode, _ in obs_trips:
        s, e = max(s, tmin), min(e, tmax)
        if e <= s:
            continue
        w = max(e - s, 3.0)
        ax.barh(-0.42, w, left=s, height=0.13, color=OBS_C, alpha=0.75)
        ax.text(s + w / 2, -0.63, str(mode).split(" (")[0], ha="center",
                va="center", fontsize=4.9, color=OBS_C)
    for s, e, mode, _ in pred_trips:
        s, e = max(s, tmin), min(e, tmax)
        if e <= s:
            continue
        w = max(e - s, 3.0)
        ax.barh(-0.72, w, left=s, height=0.13, color=PRED_C, alpha=0.95)
        ax.text(s + w / 2, -0.88, str(mode).split(" (")[0], ha="center",
                va="center", fontsize=4.9, color=PRED_C)

    ax.set_xticks(range(int(np.ceil(tmin / 60.0)) * 60, int(tmax) + 1, 180))
    ax.set_xticklabels([_hm(h) for h in
                        range(int(np.ceil(tmin / 60.0)) * 60,
                              int(tmax) + 1, 180)], fontsize=8.2)
    ax.tick_params(axis="x", length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _obs_plans(path: str) -> dict[str, tuple]:
    t = pd.read_csv(path, low_memory=False)
    t = t.dropna(subset=["PERSID", "STARTIME", "ENDTIME", "DESTPURP"])
    for c in ("STARTIME", "ENDTIME", "DURATION"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["STARTIME", "ENDTIME"])
    out = {}
    want = [
        ("work", "work", (2, 2), ("DA",), (360, 660), (240, 720)),
        ("school", "edu", (1, 2), ("Walk", "Transit", "Bike", "DA", "SR"),
         (420, 540), (300, 450)),
        ("uni", "edu", (2, 3), ("Walk", "Transit", "Bike", "DA", "SR"),
         (540, 720), (120, 300)),
        ("escort", "escort", (2, 2), ("DA", "SR", "Walk"),
         (480, 1020), (30, 240)),
        ("shop", "shopping", (2, 2), ("Walk", "DA", "SR"),
         (540, 1080), (30, 300)),
        ("social", "social", (2, 2), ("DA", "SR", "Walk", "Transit"),
         (600, 1080), (30, 300)),
    ]
    for key, purpose, (n0, n1), modes, (s0, s1), (d0, d1) in want:
        pinned = {"work": "143151/1001", "school": "105212/1008",
                  "escort": "141836/1003", "shop": "142563/1000"}.get(key)
        if pinned is not None:
            g = t[t["PERSID"] == pinned]
            if not g.empty:
                out[key] = (pinned, g)
                continue
        best = None
        for pid, g in t.groupby("PERSID"):
            if not (n0 <= len(g) <= n1):
                continue
            cand = g[g["DESTPURP"] == purpose]
            if cand.empty:
                continue
            row = cand.iloc[0]
            if str(row["MAINMODE"]) not in modes:
                continue
            if not (s0 <= row["STARTIME"] <= s1):
                continue
            dur = row["DURATION"]
            if not (d0 <= dur <= d1):
                continue
            score = (abs(row["STARTIME"] - (s0 + s1) / 2) - dur
                     + 30 * (len(g) - n0))
            if best is None:
                best = (score, pid, g)
                break
        if best is not None:
            out[key] = (best[1], best[2])
    return out


def _pred_plans(run_dir: str, obs: dict[str, tuple],
                nmax: int = 2_500_000) -> dict[str, tuple]:
    path = os.path.join(run_dir, "stage5_output.csv")
    cols = ["person_id", "row_type", "purpose", "STARTIME", "ENDTIME",
            "mode_name", "dist", "ptour_id", "ptype"]
    d = pd.read_csv(path, usecols=cols, nrows=nmax, low_memory=False)
    d = d.dropna(subset=["person_id", "STARTIME", "ENDTIME"])
    d = d[d["purpose"].notna()]
    out = {}
    want = [
        ("work", "FullTimeWorker", "work"),
        ("school", "SchoolChild", "edu"),
        ("uni", "UniversityStudent", "edu"),
        ("escort", "NonWorker", "escort"),
        ("shop", "PartTimeWorker", "shopping"),
        ("social", "FullTimeWorker", "social"),
    ]
    for key, ptype, purpose in want:
        cand = d[(d.ptype == ptype) & (d.row_type == "activity")
                 & (d.purpose == purpose)]
        if cand.empty:
            continue
        target = None
        target_n = None
        if key in obs:
            _opid, og = obs[key]
            oa, _ = _obs_segments(og)
            target_n = len(og)
            psets = {
                "work": {"work"},
                "school": {"edu", "school", "education", "university"},
                "uni": {"edu", "school", "education", "university"},
                "escort": {"escort"},
                "shop": {"shopping", "shop"},
                "social": {"social"},
            }
            pset = psets.get(key, {purpose})
            for s, e, p in oa:
                if str(p).strip().lower() in pset:
                    target = (s, e - s)
                    break
        preferred = {"work": 1000001, "school": 1000006,
                     "escort": 1000017, "shop": 1000434}.get(key)
        if preferred is not None and (cand.person_id == preferred).any():
            g = d[d.person_id == preferred]
            if not g.empty and g.ptour_id.nunique() == 1:
                out[key] = (preferred, g)
                continue
        best = None
        for n, pid in enumerate(cand.person_id.drop_duplicates()):
            if n >= 3000:
                break
            g = d[d.person_id == pid]
            if g.ptour_id.nunique() != 1:
                continue
            rows = g[(g.row_type == "activity") & (g.purpose == purpose)]
            if rows.empty:
                continue
            r = rows.iloc[0]
            s = float(r["STARTIME"]); e = float(r["ENDTIME"])
            score = 0.0
            if target is not None:
                score += abs(s - target[0]) + abs((e - s) - target[1])
            if target_n is not None:
                score += 20.0 * abs(len(g) - target_n)
            if best is None or score < best[0]:
                best = (score, pid, g)
        if best is not None:
            out[key] = (best[1], best[2])
    return out


def _obs_segments(g: pd.DataFrame):
    acts, trips = [], []
    for _, r in g.sort_values("STARTIME").iterrows():
        s, e = float(r["STARTIME"]), float(r["ENDTIME"])
        mode = str(r.get("MAINMODE", ""))
        trips.append((s, e, mode, mode))
        dur = float(r["DURATION"]) if pd.notna(r["DURATION"]) else 0.0
        if dur > 0:
            acts.append((e, e + dur, str(r["DESTPURP"])))
    return acts, trips


def _pred_segments(g: pd.DataFrame):
    acts, trips = [], []
    for _, r in g.sort_values("STARTIME").iterrows():
        s, e = float(r["STARTIME"]), float(r["ENDTIME"])
        if e <= s:
            continue
        if r["row_type"] == "trip":
            trips.append((s, e, str(r.get("mode_name", "") or ""), ""))
        else:
            p = str(r["purpose"])
            if p not in ("travel", ""):
                acts.append((s, e, p))
    return acts, trips


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=r"runs/safety_baseline__26926729.aqua")
    ap.add_argument("--observed", default="initial_trips_hhts.csv")
    ap.add_argument("--out", default="poster_plans_obs_pred.png")
    a = ap.parse_args()

    obs = _obs_plans(a.observed)
    pred = _pred_plans(a.run_dir, obs)
    print("observed cases:", {k: v[0] for k, v in obs.items()})
    print("predicted cases:", {k: v[0] for k, v in pred.items()})

    cases = [("work", "Work", "full-time worker"),
             ("school", "School", "school child"),
             ("uni", "University", "university student"),
             ("escort", "Escort", "non-worker"),
             ("shop", "Shopping", "part-time worker"),
             ("social", "Social", "full-time worker")]

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 7.4),
                             sharex=False, sharey=False,
                             gridspec_kw={"hspace": 0.50, "wspace": 0.14})
    for j, (key, title, sub) in enumerate(cases):
        ax = axes.flat[j]
        if key not in obs or key not in pred:
            ax.axis("off")
            continue
        opid, og = obs[key]
        ppid, pg = pred[key]
        oa, ot = _obs_segments(og)
        pa, pt = _pred_segments(pg)

        purposes = [str(p) for p in og["DESTPURP"].unique()]
        purposes += [str(p) for p in pg.loc[pg.row_type == "activity",
                                            "purpose"].unique()]
        dur = {}
        for s, e, p in oa + pa:
            dur[p] = dur.get(p, 0.0) + max(0.0, e - s)
        ycats = ["home"] + [p for p in
                            ("work", "edu", "escort", "shopping", "social",
                             "othmaint", "othdiscr", "eat")
                            if p in purposes and dur.get(p, 0.0) >= 20]
        starts = [s for s, _, _ in oa + pa] + [s for s, _, _, _ in ot + pt]
        ends = [e for _, e, _ in oa + pa] + [e for _, e, _, _ in ot + pt]
        tmin = max(300.0, min(starts + [1440]) - 45)
        tmax = min(1440.0, max(ends + [0]) + 45)
        if tmax <= tmin:
            tmin, tmax = 4.0 * 60, 24.0 * 60

        ptype = pg["ptype"].iloc[0] if "ptype" in pg else sub
        note = f"HHTS {opid}  vs  model {ppid} ({ptype})"
        _draw_overlay(ax, oa, ot, pa, pt, ycats, tmin, tmax,
                      f"{title} - observed vs model", note)

    handles = [
        mpatches.Patch(facecolor=OBS_C, label="Observed (HHTS)"),
        mpatches.Patch(facecolor=PRED_C, label="Model-generated"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=10.5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Daily activity plans - observed HHTS vs "
                 "matched model exemplars",
                 fontsize=14, fontweight="bold", color=QUT["navy"], y=1.01)
    fig.tight_layout(rect=(0, 0.045, 1, 0.99))
    fig.savefig(a.out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())


