#!/usr/bin/env python
"""
plot_luti_figures_371.py — paper-figure analogues for SEQ (arXiv:2608.24817).
================================================================================
Fig 2: SA2 change vs BASE 2045, population & job density (TELE | MBUF).
Fig 3: household travel trajectories 2025-2045 (VMT, VHT, trips, cost).
Fig 4: normalised occupancy schedules by building type vs DOE reference.
Fig 6: SA2 % change in site energy vs BASE 2045.
Fig 7: (a) hourly occupancy change by sector; (b) SF-vs-MF occupant-hours
       change by distance-to-CBD band (paper accessibility mechanism).

No geopandas: SA2 polygons parsed from sa2_371.geojson with json and drawn as
matplotlib patches. All inputs from run_luti_scenarios_371.py +
build_building_energy_371.py. Non-fatal per-figure (plots never kill a run).

Run:  python plot_luti_figures_371.py
Env:  LUTI_OUTDIR, ENERGY_OUTDIR, SA2_GEOJSON, ZONE_IDS_371
"""
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("lutifig")

_HERE = os.path.dirname(os.path.abspath(__file__))
LUTI = os.environ.get("LUTI_OUTDIR", os.path.join(_HERE, "saved_data", "luti"))
ENERGY = os.environ.get("ENERGY_OUTDIR", os.path.join(_HERE, "saved_data", "energy"))
GEOJSON = os.environ.get("SA2_GEOJSON",
                         os.path.join(_HERE, "saved_data", "sa2_371.geojson"))
ZONE_IDS = os.environ.get("ZONE_IDS_371",
                          os.path.join(_HERE, "saved_data", "seq_zone_ids_371.csv"))
FIGDIR = os.path.join(LUTI, "figures")
CBD_LON, CBD_LAT = 153.0251, -27.4679

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


def _polys():
    """SA2_MAIN16 -> list of (N,2) lon/lat rings (outer rings only)."""
    with open(GEOJSON) as f:
        gj = json.load(f)
    out = {}
    for feat in gj.get("features", []):
        p = feat.get("properties", {})
        code = None
        for k in ("SA2_MAIN16", "sa2_code", "SA2_CODE", "ORDER"):
            if k in p and p[k] is not None:
                try:
                    code = int(float(p[k]))
                    break
                except (ValueError, TypeError):
                    continue
        if code is None:
            continue
        g = feat.get("geometry", {})
        rings = []
        if g.get("type") == "Polygon":
            rings = [np.asarray(g["coordinates"][0], dtype=float)]
        elif g.get("type") == "MultiPolygon":
            rings = [np.asarray(poly[0], dtype=float) for poly in g["coordinates"]]
        if rings:
            out[code] = rings
    return out


def _choropleth(ax, polys, values, title, vmin=None, vmax=None, cmap="RdBu_r",
                label="% change vs BASE"):
    import matplotlib as _mpl
    codes = [c for c in values.index if c in polys]
    vals = np.array([values[c] for c in codes], dtype=float)
    vmin = np.nanpercentile(vals, 2) if vmin is None else vmin
    vmax = np.nanpercentile(vals, 98) if vmax is None else vmax
    norm = _mpl.colors.Normalize(vmin=min(vmin, -0.1), vmax=max(vmax, 0.1))
    patches, cols = [], []
    for c, v in zip(codes, vals):
        for ring in polys[c]:
            patches.append(MplPolygon(ring, closed=True))
            cols.append(v)
    pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="none")
    pc.set_array(np.array(cols))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    plt.colorbar(pc, ax=ax, shrink=0.8).set_label(label, fontsize=8)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")


def fig2_density_maps(polys):
    try:
        base = pd.read_csv(os.path.join(LUTI, "landuse_371_BASE_2045.csv"))
        zid = pd.read_csv(ZONE_IDS)
        area = dict(zip(pd.to_numeric(zid["SA2_MAIN16"], errors="coerce").astype(int).values,
                        pd.to_numeric(zid["area_sqkm"], errors="coerce").fillna(10.0).values))
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, scen, col, lab in [
                (axes[0, 0], "TELE", "pop", "TELE pop density"),
                (axes[0, 1], "MBUF", "pop", "MBUF pop density"),
                (axes[1, 0], "TELE", "emp", "TELE job density"),
                (axes[1, 1], "MBUF", "emp", "MBUF job density")]:
            sc = pd.read_csv(os.path.join(LUTI, f"landuse_371_{scen}_2045.csv"))
            m = sc.set_index("SA2_MAIN16")[col] / base.set_index("SA2_MAIN16")[col].clip(lower=1e-9)
            a = sc["SA2_MAIN16"].map(area).clip(lower=0.1)
            pct = ((sc.set_index("SA2_MAIN16")[col] / a.values)
                   / (base.set_index("SA2_MAIN16")[col] / a.values).clip(lower=1e-9) - 1.0) * 100.0
            _choropleth(ax, polys, pct, f"{lab} — %Δ vs BASE, 2045")
        fig.suptitle("SA2-level change relative to BASE, 2045 (paper Fig 2 analogue)",
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(FIGDIR, "fig2_density_maps.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"[LUTI] plot -> {p}")
    except Exception as e:  # noqa: BLE001
        log.warning("fig2 failed (non-fatal): %s", e)


def fig3_trajectories():
    try:
        t = pd.read_csv(os.path.join(LUTI, "scenario_travel.csv"))
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, col, lab in [(axes[0, 0], "vmt", "VMT (km/yr)"),
                             (axes[0, 1], "vht", "VHT (h/yr)"),
                             (axes[1, 0], "trips", "trips/yr"),
                             (axes[1, 1], "cost_aud", "cost (A$/yr)")]:
            for scen, ls in (("BASE", "-"), ("TELE", "--"), ("MBUF", ":")):
                s = t[t["scenario"] == scen].sort_values("year")
                ax.plot(s["year"], s[col] / 1e9, marker="o", linestyle=ls, label=scen)
            ax.set_title(lab)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Household travel over the forecast horizon (paper Fig 3 analogue)",
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(FIGDIR, "fig3_travel_trajectories.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"[LUTI] plot -> {p}")
    except Exception as e:  # noqa: BLE001
        log.warning("fig3 failed (non-fatal): %s", e)


def fig4_schedules():
    try:
        ref = pd.read_csv(os.path.join(ENERGY, "doe_reference_schedules.csv"), index_col=0)
        show = ["single_family", "multifamily", "office_large", "retail_standalone",
                "school_primary", "hospital", "restaurant_full", "warehouse"]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
        for ax, b in zip(axes.flat, show):
            for scen, ls in (("BASE", "-"), ("TELE", "--"), ("MBUF", ":")):
                p = os.path.join(ENERGY, f"occupancy_371_{scen}.csv")
                if os.path.exists(p):
                    occ = pd.read_csv(p, index_col=0)
                    if b in occ.columns:
                        ax.plot(occ.index, occ[b], linestyle=ls, label=scen, linewidth=1.2)
            if b in ref.columns:
                ax.plot(ref.index, ref[b], color="k", linestyle="-.", linewidth=1.0,
                        label="DOE ref")
            ax.set_title(b, fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 23)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle("Normalised daily occupancy by building type, 2045 (paper Fig 4 analogue)",
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(FIGDIR, "fig4_occupancy_schedules.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"[LUTI] plot -> {p}")
    except Exception as e:  # noqa: BLE001
        log.warning("fig4 failed (non-fatal): %s", e)


def fig6_energy_maps(polys):
    try:
        base = pd.read_csv(os.path.join(ENERGY, "building_energy_371_BASE.csv"))
        btot = base.drop(columns=["SA2_MAIN16"]).sum(axis=1)
        bidx = base["SA2_MAIN16"].astype(int).values
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, scen in zip(axes, ("TELE", "MBUF")):
            p = os.path.join(ENERGY, f"building_energy_371_{scen}.csv")
            if not os.path.exists(p):
                ax.set_title(f"{scen} — missing run")
                continue
            sc = pd.read_csv(p)
            stot = sc.drop(columns=["SA2_MAIN16"]).sum(axis=1)
            pct = pd.Series((stot.values / btot.clip(lower=1e-9).values - 1.0) * 100.0,
                            index=sc["SA2_MAIN16"].astype(int).values)
            _choropleth(ax, polys, pct, f"{scen} site energy — %Δ vs BASE",
                        label="%Δ site energy")
        fig.suptitle("SA2 percent change in total site energy, 2045 (paper Fig 6 analogue)",
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(FIGDIR, "fig6_energy_maps.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"[LUTI] plot -> {p}")
    except Exception as e:  # noqa: BLE001
        log.warning("fig6 failed (non-fatal): %s", e)


def fig7_occupancy_shift():
    try:
        from build_building_energy_371 import compute_occupancy
        lu = pd.read_csv(os.environ.get(
            "LANDUSE_371", os.path.join(_HERE, "saved_data", "zone_landuse_overture.csv")))
        lu["SA2_MAIN16"] = pd.to_numeric(lu["SA2_MAIN16"], errors="coerce").astype("Int64")
        lu = lu.dropna(subset=["SA2_MAIN16"]).copy()
        zids = lu["SA2_MAIN16"].astype(int).values
        trips = pd.read_csv(os.environ.get(
            "TRIPS_371", os.path.join(_HERE, "saved_data", "trips_processed.csv")),
            low_memory=False, usecols=["DESTSA2_2021", "DESTPURP", "DURATION",
                                       "TRIPWGT23", "ARRTIME"])
        occ_b, _ = compute_occupancy(trips, lu, zids, scenario="BASE")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # (a) hourly change by sector
        for scen, ls in (("TELE", "--"), ("MBUF", ":")):
            occ_s, _ = compute_occupancy(trips, lu, zids, scenario=scen)
            d = (occ_s.sum(axis=0) - occ_b.sum(axis=0))
            axes[0].bar(d.index, d.values, alpha=0.5, label=scen)
        axes[0].set_title("(a) Δ daily occupant-hours by building type")
        axes[0].tick_params(axis="x", rotation=60, labelsize=7)
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        # (b) SF vs MF change by distance-to-CBD band
        zid = pd.read_csv(ZONE_IDS)
        lon = pd.to_numeric(zid["longitude"], errors="coerce").fillna(CBD_LON).values
        lat = pd.to_numeric(zid["latitude"], errors="coerce").fillna(CBD_LAT).values
        dist = 2 * 6371 * np.arcsin(np.sqrt(
            np.sin(np.radians(lat - CBD_LAT) / 2) ** 2 + np.cos(np.radians(CBD_LAT))
            * np.cos(np.radians(lat)) * np.sin(np.radians(lon - CBD_LON) / 2) ** 2))
        zc = pd.to_numeric(zid["SA2_MAIN16"], errors="coerce").astype(int).values
        bands = [(0, 5), (5, 15), (15, 30), (30, 1000)]
        x = np.arange(len(bands))
        wdt = 0.2
        for k, scen in enumerate(("TELE", "MBUF")):
            occ_s, _ = compute_occupancy(trips, lu, zids, scenario=scen)
            for j, (bcol, lbl) in enumerate((("single_family", "SF"), ("multifamily", "MF"))):
                vals = []
                dmap = dict(zip(zc, dist))
                for lo, hi in bands:
                    sel = [z for z in occ_s.index
                           if z in dmap and lo <= dmap[z] < hi]
                    vals.append(float((occ_s.loc[sel, bcol] - occ_b.loc[sel, bcol]).sum()))
                axes[1].bar(x + (k * 2 + j - 1.5) * wdt, vals, width=wdt,
                            label=f"{scen} {lbl}")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(["0-5km", "5-15km", "15-30km", "30km+"])
        axes[1].set_title("(b) Δ occupant-hours SF vs MF by CBD distance")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.suptitle("Occupancy shifts under TELE/MBUF (paper Fig 7 analogue)",
                     fontweight="bold")
        fig.tight_layout()
        p = os.path.join(FIGDIR, "fig7_occupancy_shift.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print(f"[LUTI] plot -> {p}")
    except Exception as e:  # noqa: BLE001
        log.warning("fig7 failed (non-fatal): %s", e)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    polys = _polys() if os.path.exists(GEOJSON) else {}
    log.info("polygons for %d SA2 zones", len(polys))
    fig3_trajectories()
    fig4_schedules()
    if polys:
        fig2_density_maps(polys)
        fig6_energy_maps(polys)
    fig7_occupancy_shift()
    print("Figures complete.")


if __name__ == "__main__":
    main()
