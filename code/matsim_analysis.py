"""
matsim_analysis.py
==================
Network-level analysis plots for a completed MATSim run (the output/ directory
produced by the Controler). Goes beyond MATSim's default per-mode histograms:

  1. network_volume_car.png     link volumes (car), colored + width-scaled
  2. network_volume_total.png   all vehicles incl. transit on the road network
  3. network_congestion.png     peak-hour volume / capacity (V/C) hot spots
  4. mode_departure_profile.png departures per hour by mode
  5. trip_distance_by_mode.png  trip distance distribution by mode
  6. summary.json               headline network stats

Reads MATSim's compressed outputs directly (.zst via the `zstd` CLI, or .gz),
so no extra Python deps beyond matplotlib/pandas/numpy are required.

CLI:
    python matsim_analysis.py --output-dir <run>/output [--out <run>/output/analysis]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


# --------------------------------------------------------------------------- #
#  compressed-file readers                                                     #
# --------------------------------------------------------------------------- #
def _open_binary(path):
    """Return a binary file-like stream for a plain/.gz/.zst file."""
    if path.endswith(".zst"):
        if shutil.which("zstd") is None:
            raise RuntimeError("need the `zstd` CLI to read .zst outputs")
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        return p.stdout
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _find(output_dir, *basenames):
    """Find the first existing file matching any basename (with/without .zst/.gz)."""
    for base in basenames:
        for ext in ("", ".zst", ".gz"):
            p = os.path.join(output_dir, base + ext)
            if os.path.exists(p):
                return p
    return None


# --------------------------------------------------------------------------- #
#  network geometry                                                            #
# --------------------------------------------------------------------------- #
def read_network(path):
    """Stream-parse a MATSim network → (nodes {id:(x,y)}, links list of dicts)."""
    nodes = {}
    links = []
    stream = _open_binary(path)
    for event, elem in ET.iterparse(stream, events=("end",)):
        tag = elem.tag
        if tag == "node":
            nid = elem.get("id")
            nodes[nid] = (float(elem.get("x")), float(elem.get("y")))
            elem.clear()
        elif tag == "link":
            links.append({
                "id": elem.get("id"),
                "from": elem.get("from"),
                "to": elem.get("to"),
                "capacity": float(elem.get("capacity", "0") or 0),
                "freespeed": float(elem.get("freespeed", "0") or 0),
                "length": float(elem.get("length", "0") or 0),
                "modes": elem.get("modes", ""),
            })
            elem.clear()
    return nodes, links


# --------------------------------------------------------------------------- #
#  link volumes from events                                                    #
# --------------------------------------------------------------------------- #
def link_volumes_from_events(path, n_hours=30):
    """Stream events → per-link hourly volumes for total & car (non-transit).

    Returns (total[linkId]->np.array(n_hours), car[linkId]->np.array(n_hours)).
    Transit vehicles are identified from TransitDriverStarts events.
    """
    total = defaultdict(lambda: np.zeros(n_hours, dtype=np.int32))
    car = defaultdict(lambda: np.zeros(n_hours, dtype=np.int32))
    transit_veh = set()
    stream = _open_binary(path)
    for _evt, elem in ET.iterparse(stream, events=("end",)):
        if elem.tag != "event":
            continue
        etype = elem.get("type")
        if etype == "TransitDriverStarts":
            v = elem.get("vehicleId")
            if v:
                transit_veh.add(v)
        elif etype == "left link":
            link = elem.get("link")
            veh = elem.get("vehicle")
            t = float(elem.get("time", "0") or 0)
            h = min(int(t // 3600), n_hours - 1)
            total[link][h] += 1
            if veh not in transit_veh:
                car[link][h] += 1
        elem.clear()
    return total, car


# --------------------------------------------------------------------------- #
#  plotting                                                                    #
# --------------------------------------------------------------------------- #
def _link_segments(nodes, links, only=None):
    segs, vals_idx = [], []
    for i, lk in enumerate(links):
        a, b = nodes.get(lk["from"]), nodes.get(lk["to"])
        if a is None or b is None:
            continue
        if only is not None and lk["id"] not in only:
            continue
        segs.append([a, b])
        vals_idx.append(i)
    return segs, vals_idx


def plot_volume_map(nodes, links, vol_daily, title, path, cmap="turbo"):
    loaded = {lid for lid, v in vol_daily.items() if v > 0}
    # faint basemap of all links
    base_segs, _ = _link_segments(nodes, links)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=130)
    if base_segs:
        ax.add_collection(LineCollection(base_segs, colors="#e8e8e8",
                                         linewidths=0.15, zorder=1))
    segs, idx = _link_segments(nodes, links, only=loaded)
    vals = np.array([vol_daily[links[i]["id"]] for i in idx], dtype=float)
    if len(vals):
        order = np.argsort(vals)
        segs = [segs[k] for k in order]
        vals = vals[order]
        lw = 0.3 + 3.5 * (vals / vals.max()) ** 0.5
        lc = LineCollection(segs, array=np.log10(vals + 1), cmap=cmap,
                            linewidths=lw, zorder=2)
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label("daily link volume (log10 vehicles)")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Easting (EPSG:28356, m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return len(loaded)


def plot_congestion_map(nodes, links, hourly_car, path):
    link_by_id = {lk["id"]: lk for lk in links}
    vc = {}
    for lid, arr in hourly_car.items():
        lk = link_by_id.get(lid)
        if not lk or lk["capacity"] <= 0:
            continue
        vc[lid] = float(arr.max()) / lk["capacity"]     # peak-hour V/C
    base_segs, _ = _link_segments(nodes, links)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=130)
    if base_segs:
        ax.add_collection(LineCollection(base_segs, colors="#ececec",
                                         linewidths=0.15, zorder=1))
    segs, idx = _link_segments(nodes, links, only=set(vc))
    vals = np.clip(np.array([vc[links[i]["id"]] for i in idx]), 0, 1.5)
    if len(vals):
        order = np.argsort(vals)
        segs = [segs[k] for k in order]
        vals = vals[order]
        lw = 0.3 + 3.0 * np.clip(vals, 0, 1)
        lc = LineCollection(segs, array=vals, cmap="RdYlGn_r",
                            linewidths=lw, zorder=2)
        lc.set_clim(0, 1.0)
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label("peak-hour volume / capacity")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("Peak-hour congestion (V/C) — car", fontsize=13)
    ax.set_xlabel("Easting (EPSG:28356, m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return len(vc)


def plot_trip_stats(output_dir, out_dir):
    """Departure-time profile and distance distribution by mode from trips."""
    trips_path = _find(output_dir, "output_trips.csv")
    if trips_path is None:
        return None
    df = pd.read_csv(_open_binary(trips_path), sep=";")
    mode_col = "main_mode" if "main_mode" in df.columns else (
        "modes" if "modes" in df.columns else None)
    dep_col = "dep_time" if "dep_time" in df.columns else None
    dist_col = next((c for c in ("traveled_distance", "euclidean_distance")
                     if c in df.columns), None)

    def _hms_to_h(s):
        try:
            h, m, sec = str(s).split(":")
            return int(h) + int(m) / 60 + float(sec) / 3600
        except Exception:
            try:
                return float(s) / 3600
            except Exception:
                return np.nan

    # departure profile
    if dep_col and mode_col:
        df["_h"] = df[dep_col].map(_hms_to_h)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
        bins = np.arange(0, 31, 1)
        modes = [m for m in df[mode_col].dropna().unique()]
        bottom = np.zeros(len(bins) - 1)
        for m in sorted(modes):
            h, _ = np.histogram(df.loc[df[mode_col] == m, "_h"].dropna(), bins=bins)
            ax.bar(bins[:-1], h, bottom=bottom, width=1.0, label=str(m), align="edge")
            bottom += h
        ax.set_xlabel("hour of day")
        ax.set_ylabel("departures")
        ax.set_title("Trip departures by hour and mode")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "mode_departure_profile.png"))
        plt.close(fig)

    # distance distribution
    if dist_col and mode_col:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
        for m in sorted(df[mode_col].dropna().unique()):
            d = pd.to_numeric(df.loc[df[mode_col] == m, dist_col],
                              errors="coerce").dropna() / 1000.0
            if len(d) > 1:
                ds = np.sort(d.values)
                ax.plot(ds, np.linspace(0, 1, len(ds)), label=str(m), lw=1.8)
        ax.set_xlabel("trip distance (km)")
        ax.set_ylabel("cumulative share")
        ax.set_xlim(0, np.nanpercentile(pd.to_numeric(df[dist_col],
                    errors="coerce") / 1000.0, 98))
        ax.set_title("Trip distance distribution by mode (CDF)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "trip_distance_by_mode.png"))
        plt.close(fig)
    return df


# --------------------------------------------------------------------------- #
#  orchestrator                                                                #
# --------------------------------------------------------------------------- #
def analyze(output_dir, out_dir=None):
    out_dir = out_dir or os.path.join(output_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    net_path = _find(output_dir, "output_network.xml")
    events_path = _find(output_dir, "output_events.xml")
    if net_path is None:
        raise FileNotFoundError(f"no output_network.xml* under {output_dir}")

    print(f"[analysis] reading network {net_path}")
    nodes, links = read_network(net_path)
    print(f"[analysis] network: {len(nodes)} nodes, {len(links)} links")

    summary = {"n_nodes": len(nodes), "n_links": len(links)}

    if events_path:
        print(f"[analysis] parsing events {events_path} (this can take a while)...")
        total_h, car_h = link_volumes_from_events(events_path)
        total_daily = {k: int(v.sum()) for k, v in total_h.items()}
        car_daily = {k: int(v.sum()) for k, v in car_h.items()}
        n_car = plot_volume_map(nodes, links, car_daily,
                                "Network link volumes — car",
                                os.path.join(out_dir, "network_volume_car.png"))
        n_tot = plot_volume_map(nodes, links, total_daily,
                                "Network link volumes — all vehicles (incl. transit)",
                                os.path.join(out_dir, "network_volume_total.png"))
        n_vc = plot_congestion_map(nodes, links, car_h,
                                   os.path.join(out_dir, "network_congestion.png"))
        summary.update({
            "loaded_links_car": n_car,
            "loaded_links_total": n_tot,
            "links_with_vc": n_vc,
            "car_link_traversals": int(sum(car_daily.values())),
            "total_link_traversals": int(sum(total_daily.values())),
            "max_car_link_daily_volume": int(max(car_daily.values()) if car_daily else 0),
        })
        print(f"[analysis] volume maps: {n_car} car-loaded links, {n_tot} total-loaded")
    else:
        print("[analysis] no events file — skipping network volume maps")

    trips = plot_trip_stats(output_dir, out_dir)
    if trips is not None:
        summary["n_trips"] = int(len(trips))
        mcol = "main_mode" if "main_mode" in trips.columns else None
        if mcol:
            summary["mode_split"] = {str(k): round(v, 4) for k, v in
                                     trips[mcol].value_counts(normalize=True).items()}

    # ── DTA convergence: relative volume gap between the last two iterations ─
    # (TRB DTA Primer ch.2: convergence is a core quality dimension; a
    # practical proxy for the relative gap is the iteration-to-iteration
    # change in link volumes — small changes ⇒ no user has a strong incentive
    # to keep switching routes.)
    gap_info = relative_gap_from_linkstats(output_dir)
    if gap_info:
        summary.update(gap_info)
        rg = gap_info.get("relative_gap_volume_last2")
        if rg is not None:
            verdict = ("converged" if rg < 0.05 else
                       "acceptable" if rg < 0.15 else "NOT converged — "
                       "consider more iterations")
            print(f"[analysis] DTA convergence: relative volume gap = {rg:.3f} "
                  f"({verdict})")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analysis] wrote plots + summary.json to {out_dir}")
    return summary


def relative_gap_from_linkstats(output_dir, top_n: int = 2) -> dict | None:
    """
    Compare daily link volumes between the two most recent per-iteration
    linkstats files (written every iteration by RunScenario) and return
        {'relative_gap_volume_last2': Σ|v_t − v_{t−1}| / Σ v_t,
         'linkstats_iterations': [N-1, N]}
    None when fewer than two linkstats files exist.
    """
    import glob
    import re as _re
    import gzip as _gzip
    from io import BytesIO
    import pandas as _pd

    files = glob.glob(os.path.join(output_dir, "ITERS", "it.*", "*.linkstats.txt.gz"))
    if not files:
        files = glob.glob(os.path.join(output_dir, "ITERS", "it.*", "*.linkstats.csv.gz"))
    if not files:
        return None
    iters = []
    for f in files:
        m = _re.search(r"it\.(\d+)", f)
        if m:
            iters.append((int(m.group(1)), f))
    iters.sort()
    if len(iters) < top_n:
        return None
    (_, f_prev), (_, f_last) = iters[-2], iters[-1]

    def _daily_volume(path):
        opener = _gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            df = _pd.read_csv(fh, sep="\t")
        col = "HRS0-24avg" if "HRS0-24avg" in df.columns else df.columns[-4]
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        s.index = df["LINK"].astype(str)
        return s

    v_prev = _daily_volume(f_prev)
    v_last = _daily_volume(f_last)
    joined = v_prev.to_frame("prev").join(v_last.to_frame("last"), how="outer").fillna(0.0)
    denom = float(joined["last"].abs().sum())
    if denom <= 0:
        return None
    gap = float((joined["last"] - joined["prev"]).abs().sum() / denom)
    return {"relative_gap_volume_last2": round(gap, 4),
            "linkstats_iterations": [iters[-2][0], iters[-1][0]]}


def main():
    ap = argparse.ArgumentParser(description="Network-level analysis of a MATSim run.")
    ap.add_argument("--output-dir", required=True, help="MATSim output/ directory")
    ap.add_argument("--out", default=None, help="output dir for plots (default <output-dir>/analysis)")
    args = ap.parse_args()
    s = analyze(args.output_dir, args.out)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
