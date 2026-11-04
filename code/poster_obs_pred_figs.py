#!/usr/bin/env python
"""Observed-vs-predicted figures for the TAP 2026 poster.

Inputs (a completed prediction run dir):
  stages5_output.csv        -- prediction output (trips with DTAZ, purpose, ptour_id)
  _hhts_tours.pkl           -- HHTS observed tours (tour_destination = ORDER zone)
  plots/validation/stage5_mode_shares_calibrated.csv -- obs vs pred mode shares

Outputs (default: abm_safety/preprint/figures/)
  poster_mode_obs_pred.png        grouped bars obs vs pred per segment
  poster_dest_obs_pred_<seg>.png  1:1 scatter of per-zone destination shares
  poster_dist_obs_pred_<seg>.png  observed vs predicted tour-distance bands
  poster_obs_pred_summary.csv     headline numbers used in the poster captions
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUT = {
    "navy": "#003B71", "sky": "#0077C8", "ice": "#D6EAF8",
    "amber": "#F2A900", "green": "#3E9B4F", "red": "#C0392B",
    "grey": "#5C6773",
}
OBS_C = QUT["sky"]
PRED_C = QUT["red"]
SEG_TITLES = {"work": "Work", "edu": "Education", "nonmand": "Non-mandatory"}

# purpose buckets (mirrors stage5_destination_choice._filter_seg)
_WORK_P = ("work", "Work", "Work (my workplace)", "Work (other work reason)",
           "Direct Work Commute", "Work Related", "workFullTime", "workPartTime",
           "work_ftw", "work_ptw", "work_ft", "work_pt", "WORK", "WorkFull",
           "WorkPart", "mandatory_work", "at_work")
_EDU_P = ("edu", "Education", "Childcare or Kindergarten",
          "Childcare or kindergarten", "school", "School", "University",
          "university", "college", "College", "EDU", "SCHOOL", "primary_school",
          "secondary_school", "tertiary", "Tertiary", "mandatory_school",
          "at_school")


def _seg_of(vals: pd.Series) -> pd.Series:
    v = vals.fillna("").astype(str).str.strip()
    work = v.isin(_WORK_P)
    edu = v.isin(_EDU_P)
    out = pd.Series("", index=vals.index)
    out[work] = "work"
    out[edu] = "edu"
    out[(~work) & (~edu) & (v != "")] = "nonmand"
    return out


def _order_map() -> dict[int, int]:
    import json
    geo = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "saved_data", "sa2_371.geojson")
    g = json.load(open(geo, encoding="utf-8"))
    return {int(f["properties"]["SA2_MAIN16"]): int(f["properties"]["ORDER"])
            for f in g["features"]}


def _pred_dest_shares(run_dir: str, chunks: int = 4_000_000) -> dict[str, pd.Series]:
    """Per-segment predicted destination shares at population scale."""
    sa2_to_order = _order_map()

    path = os.path.join(run_dir, "stage5_output.csv")
    parts: dict[str, dict] = {"work": {}, "edu": {}, "nonmand": {}}
    n_trips = 0
    for i, chunk in enumerate(pd.read_csv(
            path, usecols=["row_type", "ptour_id", "purpose", "DTAZ"],
            dtype={"row_type": "string", "ptour_id": "string",
                   "purpose": "string", "DTAZ": "Int64"},
            chunksize=chunks, low_memory=True)):
        if i == 0:
            print(f"row types (chunk 0): "
                  f"{chunk['row_type'].value_counts(dropna=False).to_dict()}")
        a = chunk.loc[chunk["row_type"] == "activity"].dropna(
            subset=["DTAZ", "ptour_id"])
        if a.empty:
            continue
        a = a.copy()
        a["seg"] = _seg_of(a["purpose"])
        a = a[a["seg"] != ""]
        if a.empty:
            continue
        sa2 = pd.to_numeric(a["DTAZ"], errors="coerce").astype("Int64")
        a["ORDER"] = sa2.map(sa2_to_order)
        a = a.dropna(subset=["ORDER"])
        a = a[(a["ORDER"] >= 1) & (a["ORDER"] <= 371)]
        if a.empty:
            continue
        # one destination per tour: first activity of that segment
        d = a.groupby(["seg", "ptour_id"])["ORDER"].first().reset_index()
        n_trips += len(d)
        for seg in parts:
            z = d.loc[d["seg"] == seg, "ORDER"].astype(int)
            if len(z):
                parts[seg] = _add_counts(parts[seg], z.value_counts().astype(float))
    print(f"tour destinations used: {n_trips:,}")
    return {seg: (pd.Series(s).sort_index() / sum(s)) if s
            else pd.Series(dtype=float)
            for seg, s in parts.items()}


def _add_counts(acc: dict, new: pd.Series) -> dict:
    for zone, n in new.items():
        acc[zone] = acc.get(zone, 0.0) + n
    return acc


def _obs_dest_shares(run_dir: str) -> dict[str, pd.Series]:
    tours = pd.read_pickle(os.path.join(run_dir, "_hhts_tours.pkl"))
    seg = _seg_of(tours["main_purpose"])
    d = ppt(tours["tour_destination"])
    out = {}
    for s in ("work", "edu", "nonmand"):
        m = (seg == s) & d.notna() & (d >= 1) & (d <= 371)
        if not m.any():
            out[s] = pd.Series(dtype=float)
            continue
        cnt = tours.loc[m].groupby(d[m].astype(int)).size().astype(float)
        out[s] = cnt / cnt.sum()
    return out


def ppt(d: pd.Series) -> pd.Series:
    return pd.to_numeric(d, errors="coerce")


def _iso(v: float) -> str:
    return f"{v:.1f}"


def scatter_1to1(obs: pd.Series, pred: pd.Series, seg: str, path: str,
                 title: str) -> dict:
    zones = sorted(set(obs.index) | set(pred.index))
    o = obs.reindex(zones).fillna(0.0).to_numpy() * 100
    p = pred.reindex(zones).fillna(0.0).to_numpy() * 100
    keep = (o > 0) & (p > 0)          # log-only points
    lo, hi = 1e-3, max(o.max(), p.max()) * 1.6
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([lo, hi], [lo, hi], color=QUT["grey"], lw=1.2,
            label="1 : 1 (perfect match)")
    pmask = (o > 0) & (p > 0)
    n_off = int((~pmask).sum())
    if n_off:
        deck = np.full(n_off, lo)
        # zones predicted (or observed) at zero sit on the floor edges
        ax.scatter(o[~pmask], deck, marker="v", s=14, color=QUT["grey"],
                   alpha=0.35, lw=0)
        ax.scatter(deck, p[~pmask], marker="<", s=14, color=QUT["grey"],
                   alpha=0.35, lw=0)
        ax.annotate(f"{n_off} non-overlapping zones\n(parked on axes)",
                    (lo, lo), xytext=(6, 34), textcoords="offset points",
                    fontsize=7.5, color=QUT["grey"])
    ax.scatter(o[pmask], p[pmask], s=26, color=seg_color(seg),
               edgecolor="white", lw=0.4, alpha=0.9, label=f"{title} zones")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Observed share (%)", fontsize=9)
    ax.set_ylabel("Predicted share (%)", fontsize=9)
    ax.set_title(title, fontsize=10.5, color=QUT["navy"], fontweight="bold")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    o_log, p_log = np.log(o[pmask]), np.log(p[pmask])
    from scipy.stats import spearmanr
    if pmask.sum() > 2:
        r = float(np.corrcoef(o_log, p_log)[0, 1])
        rho = float(spearmanr(o_log, p_log).statistic)
    else:
        r = rho = float("nan")
    ax.text(0.04, 0.96, f"$r={r:.2f}$  (log)\nSpearman $\\rho={rho:.2f}$",
            transform=ax.transAxes, va="top", fontsize=9.5,
            color=QUT["navy"], fontweight="bold",
            bbox=dict(fc="white", ec=QUT["grey"], boxstyle="round,pad=0.3"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"seg": seg, "r_log": r, "rho": rho,
            "n_zones_nonzero": int(pmask.sum())}


def seg_color(seg: str) -> str:
    return {"work": QUT["navy"], "edu": QUT["sky"],
            "nonmand": QUT["green"]}[seg]


def mode_panel(cal_csv: str, segs, path: str) -> dict:
    df = pd.read_csv(cal_csv)
    order = ["DA", "SR", "Walk", "Bike", "Transit"]
    df = df[df["mode"].isin(order) & df["segment"].isin(list(segs))]
    make = {o: k for k, o in enumerate(order)}
    df["_o"] = df["mode"].map(make)
    df = df.sort_values(["segment", "_o"])
    segs_use = [s for s in segs if s in set(df["segment"])]
    fig, axes = plt.subplots(1, len(segs_use),
                             figsize=(3.4 * len(segs_use), 3.6), sharey=True)
    if len(segs_use) == 1:
        axes = [axes]
    info = {}
    for ax, seg in zip(axes, segs_use):
        s = df[df["segment"] == seg]
        x = np.arange(len(s))
        ax.bar(x - 0.2, s["observed_pct"], 0.4, label="Observed (HHTS)",
               color=OBS_C, alpha=0.9)
        ax.bar(x + 0.2, s["predicted_pct"], 0.4, label="Predicted",
               color=PRED_C, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(s["mode"], rotation=25, ha="right", fontsize=8.5)
        ax.set_title(SEG_TITLES[seg], fontsize=10.5, color=QUT["navy"],
                     fontweight="bold")
        gap = s["delta_pp"].abs().max()
        info[seg] = {"max_gap_pp": float(gap)}
        ax.set_xlabel(f"max$|\\Delta|$ = {gap:.1f} pp", fontsize=8.5,
                      color=QUT["grey"])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Share (%)", fontsize=9.5)
    axes[0].legend(fontsize=8.2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=
        r"Z:\test_runs_tours\code\runs\safety_baseline__26958837.aqua")
    ap.add_argument("--out-dir", default=
        r"Z:\test_runs_tours\abm_safety\preprint\figures")
    ap.add_argument("--segs", default="work,nonmand,edu")
    ap.add_argument("--skip-dest", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    segs = [s.strip() for s in a.segs.split(",") if s.strip()]

    cal_path = os.path.join(a.run_dir, "plots", "validation",
                            "stage5_mode_shares_calibrated.csv")
    rows = []
    if os.path.exists(cal_path):
        info = mode_panel(cal_path, segs,
                          os.path.join(a.out_dir, "poster_mode_obs_pred.png"))
        for k, v in info.items():
            rows.append({"fig": "mode", "seg": k, **v})
        print("mode panel -> poster_mode_obs_pred.png")
    if not a.skip_dest:
        obs = _obs_dest_shares(a.run_dir)
        pred = _pred_dest_shares(a.run_dir)
        for seg in segs:
            if seg not in obs or seg not in pred:
                continue
            rows.append(scatter_1to1(
                obs[seg], pred[seg], seg,
                os.path.join(a.out_dir, f"poster_dest_obs_pred_{seg}.png"),
                SEG_TITLES[seg]))

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(a.out_dir, "poster_obs_pred_summary.csv"), index=False)
        print(pd.DataFrame(rows).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
