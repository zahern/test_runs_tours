#!/usr/bin/env python
"""
plot_energy_dashboard_371.py — "what is happening" visualisation for power + energy.
================================================================================
Answers "visualise the current power consumption right now" with:

  1. power_now.png — BASE snapshot figure: 24-h system load curve (aggregated
     zone profiles), top-12 zones stacked by sector, system sector split, KPIs.
  2. power_now_map.png — peak-MW choropleth of the 371 SA2s (CBD + top-5 labels).
  3. energy_dashboard_371.html — interactive Leaflet map: metric dropdown
     (peak MW, daily energy, kW per capita, residential share, TELE/MBUF %Δ),
     hover tooltips with zone names, legend. Self-contained except the Leaflet
     CDN (needs internet in the browser; static PNGs cover offline use).

Inputs degrade gracefully: zone_power_371.csv + profiles/ (power NOW),
building_energy_371_*.csv + energy_totals_*.json (scenarios), and — once the
power job completes — opendss_results_371.csv (loading/voltage/violations get
extra map layers + a violations panel automatically).

Pure pandas/numpy/matplotlib + stdlib json (no geopandas/folium needed).

Run (login node is fine, <2 min):  python plot_energy_dashboard_371.py
Env:  POWER_OUTDIR, ENERGY_OUTDIR, SA2_GEOJSON, ZONE_IDS_371, VIZ_OUTDIR
Outputs: saved_data/power/viz/{power_now.png, power_now_map.png,
         energy_dashboard_371.html}
"""
import glob
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("powerviz")

_HERE = os.path.dirname(os.path.abspath(__file__))
POWER = os.environ.get("POWER_OUTDIR", os.path.join(_HERE, "saved_data", "power"))
ENERGY = os.environ.get("ENERGY_OUTDIR", os.path.join(_HERE, "saved_data", "energy"))
GEOJSON = os.environ.get("SA2_GEOJSON",
                         os.path.join(_HERE, "saved_data", "sa2_371.geojson"))
ZONE_IDS = os.environ.get("ZONE_IDS_371",
                          os.path.join(_HERE, "saved_data", "seq_zone_ids_371.csv"))
VIZ = os.environ.get("VIZ_OUTDIR", os.path.join(POWER, "viz"))
CBD_LON, CBD_LAT = 153.0251, -27.4679

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


def _load_zone_power():
    p = os.path.join(POWER, "zone_power_371.csv")
    if not os.path.exists(p):
        raise SystemExit(f"FATAL: {p} missing — run build_power_network_371.py first")
    zp = pd.read_csv(p)
    names = {}
    if os.path.exists(ZONE_IDS):
        zid = pd.read_csv(ZONE_IDS, usecols=lambda c: c in ("SA2_MAIN16", "SA2_NAME16"))
        if {"SA2_MAIN16", "SA2_NAME16"} <= set(zid.columns):
            names = dict(zip(pd.to_numeric(zid["SA2_MAIN16"], errors="coerce").astype(int).values,
                             zid["SA2_NAME16"].astype(str).values))
    zp["zone_name"] = pd.to_numeric(zp["SA2_MAIN16"], errors="coerce").astype(int).map(names).fillna(
        "SA2 " + zp["SA2_MAIN16"].astype(str))
    return zp


def _system_curve(zp):
    """24-h system kW: aggregate zone profiles; fallback to shaped peak."""
    profs = sorted(glob.glob(os.path.join(POWER, "profiles", "zone_*_load.csv")))
    if profs:
        curve = np.zeros(24)
        for fp in profs:
            try:
                d = pd.read_csv(fp)
                if "kw" in d.columns:
                    v = pd.to_numeric(d["kw"], errors="coerce").fillna(0.0).values[:24]
                    if len(v) == 24:
                        curve += v
            except Exception:  # noqa: BLE001 — one bad profile never kills the viz
                continue
        if curve.sum() > 0:
            return curve
    # fallback: standard SEQ shape scaled to system peak
    hrs = np.arange(24)
    shape = (0.55 + 0.45 * np.exp(-((hrs - 18.5) ** 2) / 8.0)
             + 0.15 * np.exp(-((hrs - 7.5) ** 2) / 6.0))
    return zp["peak_kw"].sum() * shape / shape.max()


def plot_power_now(zp):
    curve = _system_curve(zp)
    peak_mw = float(zp["peak_kw"].sum() / 1000.0)
    day_gwh = float(curve.sum() / 1000.0 / 1000.0)
    top = zp.nlargest(12, "peak_kw").iloc[::-1]
    sectors = {c: float(zp[c].sum() / 1000.0)
               for c in ("res_kw", "com_kw", "ind_kw", "ev_kw") if c in zp.columns}

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    fig.suptitle(f"SEQ 371-zone power — right now (BASE): {peak_mw:,.0f} MW system peak | "
                 f"{day_gwh:.1f} GWh/day", fontweight="bold", fontsize=13)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(range(24), curve / 1000.0, color="#B3541E", linewidth=2.5)
    ax.fill_between(range(24), curve / 1000.0, alpha=0.2, color="#B3541E")
    pk = int(np.argmax(curve))
    ax.annotate(f"peak {curve[pk] / 1000:,.0f} MW @ {pk}:00",
                xy=(pk, curve[pk] / 1000.0), xytext=(pk + 2, curve[pk] / 1000.0 * 1.02),
                arrowprops=dict(arrowstyle="->", color="k"), fontsize=10)
    ax.set_xlim(0, 23)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("system load (MW)")
    ax.grid(alpha=0.3)
    ax.set_title("24-h system load curve (aggregated zone profiles)", fontsize=11)

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for c, col in (("res_kw", "#2E86AB"), ("com_kw", "#F18F01"),
                   ("ind_kw", "#C73E1D"), ("ev_kw", "#6A994E")):
        if c in top.columns:
            v = top[c].values / 1000.0
            ax.barh(y, v, left=left, color=col, label=c.replace("_kw", ""))
            left += v
    ax.set_yticks(y)
    ax.set_yticklabels(top["zone_name"].values, fontsize=7)
    ax.set_xlabel("zone peak (MW)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Top-12 zones by peak, stacked by sector", fontsize=11)

    ax = fig.add_subplot(gs[1, 1])
    ax.pie(list(sectors.values()), labels=[f"{k}\n{v:,.0f} MW" for k, v in sectors.items()],
           autopct="%1.0f%%", colors=["#2E86AB", "#F18F01", "#C73E1D", "#6A994E"],
           textprops={"fontsize": 9})
    ax.set_title("System peak by sector", fontsize=11)

    fig.tight_layout()
    p = os.path.join(VIZ, "power_now.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"[PowerViz] plot -> {p}  (peak {peak_mw:,.0f} MW, {day_gwh:.1f} GWh/day)")


def _polys():
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


def plot_power_map(zp):
    import matplotlib as _mpl
    polys = _polys()
    vals = zp.set_index(zp["SA2_MAIN16"].astype(int))["peak_kw"] / 1000.0
    codes = [c for c in vals.index if c in polys]
    v = np.array([vals[c] for c in codes])
    norm = _mpl.colors.LogNorm(vmin=max(v.min(), 0.5), vmax=v.max())
    patches = [MplPolygon(ring, closed=True) for c in codes for ring in polys[c]]
    pc = PatchCollection(patches, cmap="YlOrRd", norm=norm, edgecolor="none")
    pc.set_array(np.repeat(v, [len(polys[c]) for c in codes]))

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    plt.colorbar(pc, ax=ax, shrink=0.8).set_label("zone peak (MW, log scale)")
    ax.plot(CBD_LON, CBD_LAT, "b*", markersize=14, label="Brisbane CBD")
    for _, r in zp.nlargest(5, "peak_kw").iterrows():
        ax.annotate(r["zone_name"], xy=(r["longitude"], r["latitude"]), fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.set_title("Current peak demand by SA2 (MW) — 371 SEQ zones", fontweight="bold")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(VIZ, "power_now_map.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"[PowerViz] plot -> {p}")


def _simplify_rings(polys, max_pts=600000):
    """Round + stride-decimate rings so the HTML stays light."""
    total = sum(len(r) for rings in polys.values() for r in rings)
    stride = max(1, int(total // max_pts) + 1)
    simp = {}
    for code, rings in polys.items():
        out = []
        for r in rings:
            s = r[::stride]
            if len(s) >= 4:
                out.append(np.round(s, 4).tolist())
        if out:
            simp[code] = out
    return simp


def build_dashboard(zp):
    polys = _simplify_rings(_polys())
    z = zp.copy()
    z["code"] = z["SA2_MAIN16"].astype(int)
    z["peak_mw"] = (z["peak_kw"] / 1000.0).round(2)
    z["energy_mwh_day"] = (z["energy_kwh_day"] / 1000.0).round(1)
    if "n_customers" in z.columns:
        tothh = pd.to_numeric(z["n_customers"], errors="coerce").fillna(0.0)
    else:
        tothh = pd.Series(0.0, index=z.index)
    z["kw_per_customer"] = (z["peak_kw"] / tothh.clip(lower=1)).round(2)
    z["res_share"] = (z["res_kw"] / z["peak_kw"].clip(lower=1e-9)).round(3)
    metrics = {
        "peak_mw": "Peak demand (MW)",
        "energy_mwh_day": "Daily energy (MWh)",
        "kw_per_customer": "kW per customer",
        "res_share": "Residential share",
    }
    # scenario %Δ layers from the CityBES-role energy outputs
    scen_layers = {}
    try:
        b = pd.read_csv(os.path.join(ENERGY, "building_energy_371_BASE.csv"))
        btot = b.drop(columns=["SA2_MAIN16"]).sum(axis=1)
        bidx = b["SA2_MAIN16"].astype(int).values
        for scen in ("TELE", "MBUF"):
            p = os.path.join(ENERGY, f"building_energy_371_{scen}.csv")
            if os.path.exists(p):
                s = pd.read_csv(p)
                stot = s.drop(columns=["SA2_MAIN16"]).sum(axis=1)
                pct = (stot.values / btot.clip(lower=1e-9).values - 1.0) * 100.0
                scen_layers[scen] = dict(zip(s["SA2_MAIN16"].astype(int).values,
                                             np.round(pct, 2).tolist()))
                metrics[f"energy_d_{scen.lower()}"] = f"Site-energy %Δ vs BASE ({scen})"
        _ = bidx
    except Exception as e:  # noqa: BLE001
        log.warning("scenario energy layers skipped: %s", e)
    # OpenDSS layers appear automatically once the power job completes
    op = os.path.join(POWER, "opendss_results_371.csv")
    if os.path.exists(op):
        try:
            o = pd.read_csv(op)
            oidx = o["SA2_MAIN16"].astype(int).values
            for col, lab in (("xfmr_loading_pct", "Transformer loading (%)"),
                             ("min_pu_voltage", "Min voltage (pu)"),
                             ("n_violations", "Violation count"),
                             ("upgrade_cost_aud", "Upgrade cost (A$)"),
                             ("headroom_frac", "Headroom fraction")):
                if col in o.columns:
                    metrics[f"grid_{col}"] = lab
            scen_layers["_grid_idx"] = oidx.tolist()
            scen_layers["_grid"] = o.to_dict(orient="records")
        except Exception as e:  # noqa: BLE001
            log.warning("grid layers skipped: %s", e)

    feats = []
    for _, r in z.iterrows():
        c = int(r["code"])
        if c not in polys:
            continue
        props = {"name": str(r["zone_name"]), "code": c,
                 "peak_mw": float(r["peak_mw"]),
                 "energy_mwh_day": float(r["energy_mwh_day"]),
                 "kw_per_customer": float(r["kw_per_customer"]),
                 "res_share": float(r["res_share"])}
        for scen, mp in scen_layers.items():
            if scen.startswith("_"):
                continue
            if c in mp:
                props[f"energy_d_{scen.lower()}"] = float(mp[c])
        if "_grid" in scen_layers:
            rec = next((g for g in scen_layers["_grid"]
                        if int(g.get("SA2_MAIN16", -1)) == c), None)
            if rec:
                for col in ("xfmr_loading_pct", "min_pu_voltage", "n_violations",
                            "upgrade_cost_aud", "headroom_frac"):
                    if col in rec and rec[col] == rec[col]:
                        props[f"grid_{col}"] = float(rec[col])
        for ring in polys[c]:
            feats.append({"type": "Feature", "properties": props,
                          "geometry": {"type": "Polygon", "coordinates": [ring]}})
    gj = json.dumps({"type": "FeatureCollection", "features": feats})
    metrics_js = json.dumps(metrics)

    html = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>SEQ 371 power & energy dashboard</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>body{margin:0;font-family:sans-serif}#map{height:92vh}
#bar{padding:8px;background:#f5f5f5;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#bar select{font-size:14px}#info{font-size:13px;color:#333}</style></head><body>
<div id="bar"><b>SEQ 371 power & energy</b><label>Metric:
<select id="metric"></select></label>
<span id="info">Hover a zone. Grid-violation layers appear once opendss_results_371.csv lands.</span></div>
<div id="map"></div><script>
const METRICS = @@METRICS@@;
const DATA = @@GEOJSON@@;
const map = L.map('map').setView([-27.5, 153.05], 9);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
 {attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
const sel = document.getElementById('metric');
Object.entries(METRICS).forEach(([k,v])=>{const o=document.createElement('option');
 o.value=k;o.textContent=v;sel.appendChild(o);});
function vals(m){return DATA.features.map(f=>f.properties[m]).filter(v=>v!==undefined&&v!==null);}
function colorFor(v,lo,hi){const t=Math.max(0,Math.min(1,(v-lo)/Math.max(hi-lo,1e-9)));
 const r=Math.round(255*t),b=Math.round(255*(1-t));return `rgb(${r},80,${b})`;}
let layer=null;
function draw(){
 const m=sel.value;const vs=vals(m).sort((a,b)=>a-b);
 const lo=vs[Math.floor(vs.length*0.05)]??0, hi=vs[Math.floor(vs.length*0.95)]??1;
 if(layer)map.removeLayer(layer);
 layer=L.geoJSON(DATA,{style:f=>({fillColor:colorFor(f.properties[m]??lo,lo,hi),
  weight:0.4,color:'#666',fillOpacity:0.75}),
  onEachFeature:(f,l)=>{const p=f.properties;
   l.bindTooltip(`<b>${p.name}</b> (${p.code})<br>${METRICS[m]}: ${p[m]??'n/a'}<br>`+
   `Peak: ${p.peak_mw} MW<br>Energy: ${p.energy_mwh_day} MWh/day`);}}).addTo(map);
 document.getElementById('info').textContent =
  `${METRICS[m]} — p5 ${lo.toFixed?lo.toFixed(2):lo} / p95 ${hi.toFixed?hi.toFixed(2):hi}`;
}
sel.onchange=draw;draw();
</script></body></html>"""
    html = html.replace("@@METRICS@@", metrics_js).replace("@@GEOJSON@@", gj)
    p = os.path.join(VIZ, "energy_dashboard_371.html")
    with open(p, "w") as f:
        f.write(html)
    print(f"[PowerViz] dashboard -> {p} ({os.path.getsize(p) / 1e6:.1f} MB, "
          f"{len(feats)} polygons, {len(metrics)} metrics)")


def main():
    os.makedirs(VIZ, exist_ok=True)
    zp = _load_zone_power()
    print(f"[PowerViz] {len(zp)} zones | system peak "
          f"{zp['peak_kw'].sum() / 1000:,.0f} MW | "
          f"{zp['energy_kwh_day'].sum() / 1e6:.1f} GWh/day")
    plot_power_now(zp)
    try:
        plot_power_map(zp)
    except Exception as e:  # noqa: BLE001
        log.warning("power map failed (non-fatal): %s", e)
    try:
        build_dashboard(zp)
    except Exception as e:  # noqa: BLE001
        log.warning("dashboard failed (non-fatal): %s", e)
    print("Visualisation complete.")


if __name__ == "__main__":
    main()
