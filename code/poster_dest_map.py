#!/usr/bin/env python
"""Destination-choice map overlay for the TAP 2026 poster.

Observed HHTS tour destinations vs model-generated destinations, drawn as
zone-level shares on the 371 SA2 polygons (observed | predicted | difference).

Usage:
    python poster_dest_map.py --run-dir runs/safety_baseline__26958837.aqua \
        --out abm_safety/preprint/figures/poster_dest_map.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

QUT = {"navy": "#003B71", "sky": "#0077C8", "amber": "#F2A900",
       "grey": "#5C6773", "red": "#C0392B"}

_WORK = ("work", "Work", "Work (my workplace)", "Work (other work reason)",
         "Direct Work Commute", "Work Related", "workFullTime", "workPartTime",
         "work_ftw", "work_ptw", "work_ft", "work_pt", "WORK", "WorkFull",
         "WorkPart", "mandatory_work", "at_work")
_EDU = ("edu", "Education", "Childcare or Kindergarten",
        "Childcare or kindergarten", "school", "School", "University",
        "university", "college", "College", "EDU", "SCHOOL", "primary_school",
        "secondary_school", "tertiary", "Tertiary", "mandatory_school",
        "at_school")


def _seg(v: pd.Series) -> pd.Series:
    v = v.fillna("").astype(str).str.strip()
    w, e = v.isin(_WORK), v.isin(_EDU)
    out = pd.Series("", index=v.index)
    out[w] = "work"
    out[e] = "edu"
    out[(~w) & (~e) & (v != "")] = "nonmand"
    return out


def _order_map(geo_path: str) -> dict[int, int]:
    g = json.load(open(geo_path, encoding="utf-8"))
    return {int(f["properties"]["SA2_MAIN16"]): int(f["properties"]["ORDER"])
            for f in g["features"]}


def _pred_shares(run_dir: str, seg: str, sa2_to_order: dict[int, int],
                 nmax: int | None = None) -> pd.Series:
    path = os.path.join(run_dir, "stage5_output.csv")
    cols = ["row_type", "purpose", "DTAZ", "ptour_id"]
    acc: dict[int, float] = {}
    for ch in pd.read_csv(path, usecols=cols, chunksize=2_000_000,
                          low_memory=False, nrows=nmax):
        a = ch[ch["row_type"] == "activity"].dropna(subset=["DTAZ", "ptour_id"])
        if a.empty:
            continue
        a = a.copy()
        a["seg"] = _seg(a["purpose"])
        a = a[a["seg"] == seg]
        if a.empty:
            continue
        a["ORDER"] = pd.to_numeric(a["DTAZ"], errors="coerce").astype("Int64").map(sa2_to_order)
        a = a.dropna(subset=["ORDER"])
        a = a[(a["ORDER"] >= 1) & (a["ORDER"] <= 371)]
        if a.empty:
            continue
        # one destination per tour: first activity of that purpose
        d = a.groupby("ptour_id")["ORDER"].first().astype(int)
        for z, n in d.value_counts().items():
            acc[z] = acc.get(z, 0.0) + float(n)
    s = pd.Series(acc, dtype=float).sort_index()
    return s / s.sum() if len(s) else s


def _obs_shares(run_dir: str, seg: str) -> pd.Series:
    t = pd.read_pickle(os.path.join(run_dir, "_hhts_tours.pkl"))
    s = _seg(t["main_purpose"])
    d = pd.to_numeric(t["tour_destination"], errors="coerce")
    m = (s == seg) & d.notna() & (d >= 1) & (d <= 371)
    if not m.any():
        return pd.Series(dtype=float)
    cnt = t.loc[m].groupby(d[m].astype(int)).size().astype(float)
    return cnt / cnt.sum()


def _polys(path: str) -> dict[int, list[np.ndarray]]:
    g = json.load(open(path, encoding="utf-8"))
    out: dict[int, list[np.ndarray]] = {}
    for f in g["features"]:
        z = int(f["properties"]["ORDER"])
        geom = f["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiPolygon" \
            else [geom["coordinates"]]
        rings = []
        for poly in parts:
            for ring in poly[:1]:
                rings.append(np.asarray(ring, dtype=float))
        out[z] = rings
    return out


def _bounds(polys) -> tuple[float, float, float, float]:
    xs = np.concatenate([r[:, 0] for rings in polys.values() for r in rings])
    ys = np.concatenate([r[:, 1] for rings in polys.values() for r in rings])
    dx, dy = 0.03 * (xs.max() - xs.min()), 0.03 * (ys.max() - ys.min())
    return xs.min() - dx, xs.max() + dx, ys.min() - dy, ys.max() + dy


def _draw(ax, polys, values: pd.Series, norm, cmap, title, bounds):
    verts, vals = [], []
    for z, rings in polys.items():
        v = float(values.get(z, 0.0))
        for r in rings:
            verts.append(r)
            vals.append(v)
    pc = PolyCollection(verts, array=np.asarray(vals), cmap=cmap, norm=norm,
                        edgecolors="white", linewidths=0.15)
    ax.add_collection(pc)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=QUT["navy"], fontweight="bold")
    return pc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default=r"runs/safety_baseline__26958837.aqua")
    ap.add_argument("--geo", default=r"saved_data/sa2_371.geojson")
    ap.add_argument("--out", default="poster_dest_map.png")
    ap.add_argument("--segs", default="work,nonmand")
    a = ap.parse_args()
    segs = [s.strip() for s in a.segs.split(",") if s.strip()]
    polys = _polys(a.geo)
    bounds = _bounds(polys)
    sa2_to_order = _order_map(a.geo)

    cache = os.path.splitext(a.out)[0] + "_shares.pkl"
    if os.path.exists(cache):
        shares = pd.read_pickle(cache)
        print("loaded cached shares from", cache)
    else:
        shares = {}
        for seg in segs:
            shares[(seg, "obs")] = _obs_shares(a.run_dir, seg)
            shares[(seg, "pred")] = _pred_shares(a.run_dir, seg, sa2_to_order)
        pd.to_pickle(shares, cache)
        print("wrote share cache", cache)

    rows = []
    fig, axes = plt.subplots(len(segs), 3, figsize=(13.5, 5.0 * len(segs)))
    if len(segs) == 1:
        axes = axes[None, :]
    for i, seg in enumerate(segs):
        obs = shares[(seg, "obs")]
        pred = shares[(seg, "pred")]
        zones = sorted(set(obs.index) | set(pred.index))
        o = obs.reindex(zones).fillna(0.0) * 100
        p = pred.reindex(zones).fillna(0.0) * 100
        k = (o > 0) & (p > 0)
        r = float(np.corrcoef(np.log(o[k]), np.log(p[k]))[0, 1]) if k.sum() > 2 else np.nan
        rho = float(spearmanr(np.log(o[k]), np.log(p[k])).statistic) if k.sum() > 2 else np.nan
        top25 = set(obs.nlargest(25).index)
        overlap = len(top25 & set(pred.nlargest(25).index)) / 25.0
        rows.append({"segment": seg, "r_log": r, "rho": rho,
                     "top25_overlap": overlap,
                     "mae_share": float((o - p).abs().mean() / 100.0),
                     "n_zones_both": int(k.sum())})
        print(seg, "obs top:", {int(z): round(float(v), 4)
                                for z, v in obs.nlargest(8).items()})
        print(seg, "pred top:", {int(z): round(float(v), 4)
                                 for z, v in pred.nlargest(8).items()})
        vmax = max(o.max(), p.max(), 1e-6)
        norm = LogNorm(vmin=max(1e-3, min(o[o > 0].min(), p[p > 0].min())),
                       vmax=vmax)
        cmap = "viridis"
        pc = _draw(axes[i, 0], polys, obs * 100, norm, cmap,
                   f"{seg.title()} — observed (HHTS)", bounds)
        _draw(axes[i, 1], polys, pred * 100, norm, cmap,
              f"{seg.title()} — predicted (model)", bounds)
        cb = fig.colorbar(pc, ax=axes[i, 0], orientation="horizontal",
                          fraction=0.035, pad=0.01, shrink=0.85)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("share of tours (%)", fontsize=7)

        diff = (p - o)
        lim = float(np.nanpercentile(np.abs(diff), 98)) or 1.0
        dnorm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        pc2 = _draw(axes[i, 2], polys, diff, dnorm, "RdBu_r",
                    f"{seg.title()} — predicted − observed (pp)", bounds)
        cb2 = fig.colorbar(pc2, ax=axes[i, 2], orientation="horizontal",
                           fraction=0.035, pad=0.01, shrink=0.85)
        cb2.ax.tick_params(labelsize=7)
        cb2.set_label("percentage points", fontsize=7)

    fig.suptitle("Destination choice — predicted vs observed tour destinations "
                 "(371 SA2 zones)", fontsize=13, fontweight="bold",
                 color=QUT["navy"], y=1.00)
    fig.tight_layout()
    fig.savefig(a.out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    out_csv = os.path.splitext(a.out)[0] + "_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("wrote", a.out, "and", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
