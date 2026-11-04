#!/usr/bin/env python
"""
build_seq_visualizer_371.py — ANTARES-style visualizer for the SEQ 371 framework.
================================================================================
POLARIS ANTARES (polaris.taps.anl.gov/polaris/visualizer) is a Windows-only C++
desktop app, so there is no web source to reuse. This builder generates an
equivalent web visualizer customised for our framework, with our 371 SA2 zones
embedded:

  left   LAYER SELECTOR — toggle + per-layer colour-scheme picker + click to
         inspect (Zones, Substations, Feeders, Violations, Buildings)
  centre MAP CANVAS — Leaflet map of SEQ with the 371 zones + electric network
  right  OBJECT INFORMATION — click any zone/substation/feeder for metrics,
         24-h load curve, building-stock and energy bars
  bottom TIME CONTROLS (play/pause/speed/hour slider animating hourly load)
         + DATA PLOTS (system curve, scenario KPI chips)

Energy info, building info and building draws: per-zone floorspace + site
energy by building type (CityBES-role outputs), hourly kW from the 371 zone
profiles, and grid loading/violations/headroom from the OpenDSS step.

Pure pandas/numpy + stdlib json (no geopandas/folium). Leaflet via CDN in the
browser; everything else is embedded in the single output HTML.

Run (login node, ~1 min):  python build_seq_visualizer_371.py
Env:  POWER_OUTDIR, ENERGY_OUTDIR, SA2_GEOJSON, ZONE_IDS_371, LANDUSE_371
Output: saved_data/power/viz/seq_visualizer_371.html
"""
import glob
import json
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("seqviz")

_HERE = os.path.dirname(os.path.abspath(__file__))
POWER = os.environ.get("POWER_OUTDIR", os.path.join(_HERE, "saved_data", "power"))
ENERGY = os.environ.get("ENERGY_OUTDIR", os.path.join(_HERE, "saved_data", "energy"))
LUTI = os.path.join(_HERE, "saved_data", "luti")
GEOJSON = os.environ.get("SA2_GEOJSON",
                         os.path.join(_HERE, "saved_data", "sa2_371.geojson"))
ELEC_GEOJSON = os.path.join(POWER, "sa2_371_electric.geojson")
ZONE_IDS = os.environ.get("ZONE_IDS_371",
                          os.path.join(_HERE, "saved_data", "seq_zone_ids_371.csv"))
LANDUSE = os.environ.get("LANDUSE_371",
                         os.path.join(_HERE, "saved_data", "zone_landuse_overture.csv"))
VIZ = os.path.join(POWER, "viz")


def _names():
    if os.path.exists(ZONE_IDS):
        zid = pd.read_csv(ZONE_IDS, usecols=lambda c: c in ("SA2_MAIN16", "SA2_NAME16"))
        if {"SA2_MAIN16", "SA2_NAME16"} <= set(zid.columns):
            return dict(zip(pd.to_numeric(zid["SA2_MAIN16"],
                                          errors="coerce").astype(int).values,
                            zid["SA2_NAME16"].astype(str).values))
    return {}


def _rings(gj_path):
    with open(gj_path) as f:
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
        if code is None and "id" in p and isinstance(p["id"], str):
            try:
                code = int(p["id"].split("_")[-1])
            except ValueError:
                code = None
        g = feat.get("geometry", {})
        rings = []
        if g.get("type") == "Polygon":
            rings = [np.asarray(g["coordinates"][0], dtype=float)]
        elif g.get("type") == "MultiPolygon":
            rings = [np.asarray(poly[0], dtype=float) for poly in g["coordinates"]]
        if rings:
            out.setdefault(code, []).extend(rings)
    return out, gj.get("features", [])


def _simplify(polys, max_pts=500000):
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


def main():
    os.makedirs(VIZ, exist_ok=True)
    names = _names()
    zp = pd.read_csv(os.path.join(POWER, "zone_power_371.csv"))
    zp["code"] = pd.to_numeric(zp["SA2_MAIN16"], errors="coerce").astype(int)
    zp["name"] = zp["code"].map(names).fillna("SA2 " + zp["SA2_MAIN16"].astype(str))

    # hourly kW per zone from profiles (time-animation layer)
    hourly, sys_curve = {}, np.zeros(24)
    for fp in sorted(glob.glob(os.path.join(POWER, "profiles", "zone_*_load.csv"))):
        try:
            sa2 = int(os.path.basename(fp).split("_")[1])
        except (ValueError, IndexError):
            continue
        try:
            v = pd.to_numeric(pd.read_csv(fp)["kw"], errors="coerce").fillna(0.0).values[:24]
            if len(v) == 24:
                hourly[sa2] = np.round(v.astype(float), 1).tolist()
                sys_curve += v
        except Exception:  # noqa: BLE001
            continue

    # grid metrics (appear automatically once the power job completes)
    grid = {}
    op = os.path.join(POWER, "opendss_results_371.csv")
    if os.path.exists(op):
        o = pd.read_csv(op)
        for _, r in o.iterrows():
            try:
                grid[int(r["SA2_MAIN16"])] = {
                    "loading": float(r.get("line_loading_pct", 0)),
                    "xloading": float(r.get("xfmr_loading_pct", 0)),
                    "minV": float(r.get("min_pu_voltage", 1)),
                    "maxV": float(r.get("max_pu_voltage", 1)),
                    "viol": int(r.get("n_violations", 0)),
                    "cost": float(r.get("upgrade_cost_aud", 0)),
                    "headroom": float(r.get("headroom_frac", 0)),
                }
            except (ValueError, TypeError):
                continue

    # building stock + energy by type (CityBES-role outputs)
    stock, energy_bt = {}, {}
    try:
        lu = pd.read_csv(LANDUSE)
        lu["SA2_MAIN16"] = pd.to_numeric(lu["SA2_MAIN16"], errors="coerce").astype("Int64")
        lu = lu.dropna(subset=["SA2_MAIN16"]).copy()
        import build_building_energy_371 as _bbe
        st = _bbe.compute_stock(lu)
        for sa2, row in zip(lu["SA2_MAIN16"].astype(int).values, st.values):
            stock[int(sa2)] = {t: round(float(v), 1)
                               for t, v in zip(st.columns, row) if v > 1.0}
    except Exception as e:  # noqa: BLE001
        log.warning("building stock skipped: %s", e)
    be_path = os.path.join(ENERGY, "building_energy_371_BASE.csv")
    if os.path.exists(be_path):
        be = pd.read_csv(be_path)
        cols = [c for c in be.columns if c != "SA2_MAIN16"]
        for _, r in be.iterrows():
            try:
                energy_bt[int(r["SA2_MAIN16"])] = {
                    c: round(float(r[c]) / 1000.0, 2) for c in cols if r[c] > 0.5}
            except (ValueError, TypeError):
                continue

    # zone table
    zones = {}
    for _, r in zp.iterrows():
        c = int(r["code"])
        zones[c] = {
            "name": str(r["name"]), "lon": round(float(r["longitude"]), 4),
            "lat": round(float(r["latitude"]), 4),
            "peak_mw": round(float(r["peak_kw"]) / 1000.0, 2),
            "energy_mwh": round(float(r["energy_kwh_day"]) / 1000.0, 1),
            "pv_mw": round(float(r.get("pv_kw", 0)) / 1000.0, 2),
            "res_share": round(float(r.get("res_kw", 0)) / max(float(r["peak_kw"]), 1e-9), 3),
            "n_xfmr": int(r.get("n_transformers", 0)),
            "xfmr_mva": round(float(r.get("xfmr_mva", 0)), 2),
            "feeder_km": round(float(r.get("feeder_km", 0)), 2),
            "hourly": hourly.get(c, [0.0] * 24),
            "grid": grid.get(c, {}),
            "stock": stock.get(c, {}),
            "energy_bt": energy_bt.get(c, {}),
        }

    # electric network: substations (points) + feeders (lines)
    subs, feeders = [], []
    if os.path.exists(ELEC_GEOJSON):
        try:
            _, feats = _rings(ELEC_GEOJSON)
            for f in feats:
                pr, gm = f.get("properties", {}), f.get("geometry", {})
                t = pr.get("type", "")
                if t == "Substation" and gm.get("type") == "Point":
                    sa2 = None
                    try:
                        sa2 = int(str(pr.get("id", "sub_0")).split("_")[1])
                    except (ValueError, IndexError):
                        pass
                    subs.append({"id": str(pr.get("id")), "name": str(pr.get("name", "")),
                                 "lon": gm["coordinates"][0], "lat": gm["coordinates"][1],
                                 "mva": pr.get("mva", 0), "sa2": sa2})
                elif t == "ElectricalConnector" and gm.get("type") == "LineString":
                    sa2 = None
                    try:
                        sa2 = int(str(pr.get("id", "line_0_0")).split("_")[1])
                    except (ValueError, IndexError):
                        pass
                    feeders.append({"id": str(pr.get("id")),
                                    "coords": np.round(np.asarray(
                                        gm["coordinates"], dtype=float), 4).tolist(),
                                    "sa2": sa2,
                                    "loading": grid.get(sa2, {}).get("loading", 0)})
        except Exception as e:  # noqa: BLE001
            log.warning("electric network skipped: %s", e)

    # scenario KPI chips (LUTI + energy rollups, when present)
    kpis = []
    tp = os.path.join(LUTI, "scenario_travel.csv")
    if os.path.exists(tp):
        t = pd.read_csv(tp)
        b45 = t[(t["scenario"] == "BASE") & (t["year"] == 2045)]
        if len(b45):
            b45 = b45.iloc[0]
            for s in ("TELE", "MBUF"):
                r = t[(t["scenario"] == s) & (t["year"] == 2045)]
                if len(r):
                    r = r.iloc[0]
                    kpis.append(
                        {"label": f"{s} VMT", "value": f"{(r['vmt'] / b45['vmt'] - 1) * 100:+.1f}%"})
                    kpis.append(
                        {"label": f"{s} VHT", "value": f"{(r['vht'] / b45['vht'] - 1) * 100:+.1f}%"})
    for s in ("TELE", "MBUF"):
        jp = os.path.join(ENERGY, f"energy_totals_{s}.json")
        if os.path.exists(jp):
            try:
                d = json.load(open(jp))
                kpis.append({"label": f"{s} energy", "value": f"{d.get('delta_pct', 0):+.2f}%"})
            except Exception:  # noqa: BLE001
                continue

    polys, _ = _rings(GEOJSON)
    data = {"zones": zones, "geom": _simplify({c: r for c, r in polys.items() if c in zones}),
            "subs": subs, "feeders": feeders,
            "sys_curve": np.round(sys_curve.astype(float), 1).tolist(), "kpis": kpis}
    html = APP_TEMPLATE.replace("@@DATA@@", json.dumps(data))
    out = os.path.join(VIZ, "seq_visualizer_371.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"[SEQViz] {len(zones)} zones, {len(subs)} substations, {len(feeders)} feeders, "
          f"{len(kpis)} KPI chips -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


APP_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>SEQ 371 co-simulation visualizer (ANTARES-style)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html,body{margin:0;height:100%;font-family:sans-serif;font-size:13px}
#app{display:grid;grid-template-columns:250px 1fr 320px;grid-template-rows:44px 1fr 190px;
 height:100vh;grid-template-areas:"head head head" "left map right" "left plots right"}
#head{grid-area:head;background:#1b2a4a;color:#fff;display:flex;align-items:center;gap:14px;padding:0 12px}
#head .chip{background:#33456b;border-radius:10px;padding:2px 10px;font-size:12px}
#left{grid-area:left;background:#f4f5f7;overflow-y:auto;padding:8px;border-right:1px solid #ccc}
#right{grid-area:right;background:#fbfbfb;overflow-y:auto;padding:10px;border-left:1px solid #ccc}
#map{grid-area:map}
#plots{grid-area:plots;background:#fff;border-top:1px solid #ccc;display:flex;gap:10px;padding:6px 10px;align-items:center}
.lyr{border:1px solid #ddd;background:#fff;border-radius:6px;margin-bottom:8px;padding:6px}
.lyr.sel{border-color:#1b2a4a;box-shadow:0 0 4px #1b2a4a}
.lyr select{width:100%;margin-top:4px;font-size:12px}
.lbl{font-size:11px;color:#555}
#timebar{display:flex;gap:8px;align-items:center;margin-top:6px}
#plots canvas{border:1px solid #ddd}
table.info{border-collapse:collapse;width:100%;font-size:12px}
table.info td{border-bottom:1px solid #eee;padding:2px 4px}
.bar{display:flex;align-items:center;gap:6px;font-size:11px;margin:1px 0}
.bar .fill{height:10px;background:#2E86AB;border-radius:2px}
h3{margin:6px 0 4px;font-size:13px}
</style></head><body>
<div id="app">
<div id="head"><b>SEQ 371 co-simulation visualizer</b><span class="lbl" style="color:#cdd6ea">
ANTARES-style: layers + object info + time + plots (zones embedded)</span><span id="kpis"></span></div>
<div id="left"><h3>Layer selector</h3><div id="layers"></div>
<h3>Zone colour metric</h3><select id="metric"></select>
<h3>Time controls</h3>
<div id="timebar"><button id="play">▶</button>
<select id="speed"><option value="900">1x</option><option value="250" selected>4x</option>
<option value="80">12x</option></select>
<input id="hour" type="range" min="0" max="23" value="18" style="flex:1"/>
<b id="hourlab">18:00</b></div>
<div class="lbl">Playing animates zones by hourly load (profiles).</div></div>
<div id="map"></div>
<div id="right"><h3>Object information</h3><div id="info">Click a zone, substation or feeder.</div></div>
<div id="plots"><canvas id="sysplot" width="520" height="170"></canvas>
<div class="lbl">System 24-h curve (MW). Cursor follows the time slider.<br/>
<span id="plotkpi"></span></div></div>
</div>
<script>
const DATA = @@DATA@@;
const SCHEMES = {
 YlOrRd:[[255,255,204],[254,204,92],[253,141,60],[240,59,32],[189,0,38]],
 Blues:[[239,243,255],[158,202,225],[49,130,189],[8,81,156]],
 Greens:[[237,248,233],[161,217,155],[35,139,69],[0,68,27]],
 Viridis:[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]],
 RdYlGn:[[215,25,28],[253,174,97],[255,255,191],[166,217,106],[26,150,65]]};
const METRICS = {peak_mw:{label:'Peak demand (MW)',get:z=>z.peak_mw,scheme:'YlOrRd',log:true},
 energy_mwh:{label:'Daily energy (MWh)',get:z=>z.energy_mwh,scheme:'YlOrRd',log:true},
 res_share:{label:'Residential share',get:z=>z.res_share,scheme:'Blues'},
 loading:{label:'Feeder loading (%)',get:z=>z.grid.loading||0,scheme:'YlOrRd'},
 headroom:{label:'Headroom fraction',get:z=>z.grid.headroom??0,scheme:'RdYlGn'},
 viol:{label:'Violations',get:z=>z.grid.viol||0,scheme:'YlOrRd'},
 cost:{label:'Upgrade cost (A$)',get:z=>z.grid.cost||0,scheme:'YlOrRd',log:true}};
const state = {metric:'peak_mw',schemes:{},hour:18,playing:false,timer:null,selLayer:'zones',animOn:false};
function schemeFor(layer){return state.schemes[layer]||(layer==='zones'?METRICS[state.metric].scheme:'YlOrRd');}
function interp(stops,t){t=Math.max(0,Math.min(1,t));const x=t*(stops.length-1),i=Math.min(Math.floor(x),stops.length-2),f=x-i;
 const a=stops[i],b=stops[i+1];return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;}
const map = L.map('map').setView([-27.55,153.05],9);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap contributors'}).addTo(map);
function zoneVals(){return Object.values(DATA.zones).map(z=>METRICS[state.metric].get(z)).filter(v=>v!==undefined&&v!==null&&isFinite(v));}
function scale(){const vs=zoneVals().sort((a,b)=>a-b);if(!vs.length)return{vmin:0,vmax:1};
 let lo=vs[Math.floor(vs.length*0.05)],hi=vs[Math.floor(vs.length*0.95)];if(hi<=lo)hi=lo+1;
 if(METRICS[state.metric].log){lo=Math.max(lo,0.01);}return{vmin:lo,vmax:hi};}
function colorFor(v){const{vmin,vmax}=scale();const m=METRICS[state.metric];let t=(v-vmin)/Math.max(vmax-vmin,1e-9);
 if(m.log){t=(Math.log(Math.max(v,1e-2))-Math.log(Math.max(vmin,1e-2)))/Math.max(Math.log(Math.max(vmax,1e-2))-Math.log(Math.max(vmin,1e-2)),1e-9);}
 return interp(SCHEMES[schemeFor('zones')],t);}
let zoneLayer=null,subLayer=null,feedLayer=null,violLayer=null,bldLayer=null;
function hourlyFrac(z,h){const arr=z.hourly||[];const mx=Math.max(...arr,1e-9);return (arr[h]??0)/mx;}
function zoneColor(code){const z=DATA.zones[code];const h=+document.getElementById('hour').value;
 if(state.animOn){return interp(SCHEMES[schemeFor('zones')],hourlyFrac(z,h));}
 return colorFor(METRICS[state.metric].get(z)??0);}
function drawZones(){if(zoneLayer)map.removeLayer(zoneLayer);
 const feats=[];for(const[code,rings]of Object.entries(DATA.geom)){const z=DATA.zones[code];if(!z)continue;
  for(const ring of rings)feats.push({type:'Feature',properties:{code:+code},geometry:{type:'Polygon',coordinates:[ring]}});}
 zoneLayer=L.geoJSON({type:'FeatureCollection',features:feats},{style:f=>({fillColor:zoneColor(f.properties.code),weight:0.4,color:'#666',fillOpacity:0.78}),
  onEachFeature:(f,l)=>{l.on('click',()=>showZone(f.properties.code));}}).addTo(map);}
function loadColor(pct){return interp(SCHEMES[schemeFor('feeders')],Math.max(0,Math.min(1,pct/120)));}
function drawNetwork(){[subLayer,feedLayer].forEach(l=>{if(l)map.removeLayer(l);});
 feedLayer=L.geoJSON(DATA.feeders.map(f=>({type:'Feature',properties:f,geometry:{type:'LineString',coordinates:f.coords}})),
  {style:f=>({color:loadColor(f.properties.loading||0),weight:2.2,opacity:0.9}),
   onEachFeature:(f,l)=>{l.on('click',()=>showFeeder(f.properties));}}).addTo(map);
 subLayer=L.geoJSON(DATA.subs.map(s=>({type:'Feature',properties:s,geometry:{type:'Point',coordinates:[s.lon,s.lat]}})),
  {pointToLayer:(f,ll)=>L.circleMarker(ll,{radius:3+Math.sqrt(f.properties.mva||1),color:'#1b2a4a',weight:2,fillColor:'#ffd60a',fillOpacity:0.95}),
   onEachFeature:(f,l)=>{l.on('click',()=>showSub(f.properties));}}).addTo(map);}
function drawViols(){if(violLayer)map.removeLayer(violLayer);
 const feats=[];for(const[code,rings]of Object.entries(DATA.geom)){const z=DATA.zones[code];
  if(!z||!(z.grid.viol>0))continue;
  for(const ring of rings)feats.push({type:'Feature',properties:{code:+code},geometry:{type:'Polygon',coordinates:[ring]}});}
 violLayer=L.geoJSON({type:'FeatureCollection',features:feats},{style:{color:'#d00000',weight:2,fill:false},
  onEachFeature:(f,l)=>{l.on('click',()=>showZone(f.properties.code));}}).addTo(map);}
function drawBld(){if(bldLayer)map.removeLayer(bldLayer);
 const feats=Object.entries(DATA.zones).filter(([c,z])=>Object.keys(z.stock||{}).length)
  .map(([c,z])=>({type:'Feature',properties:{code:+c},
   geometry:{type:'Point',coordinates:[z.lon,z.lat]}}));
 bldLayer=L.geoJSON({type:'FeatureCollection',features:feats},
  {pointToLayer:(f,ll)=>{const z=DATA.zones[f.properties.code];
   const m2=Object.values(z.stock).reduce((a,b)=>a+b,0);
   return L.circleMarker(ll,{radius:2+6*Math.sqrt(m2/1e6),color:'#5a3d2b',weight:1.5,fillColor:'#c08552',fillOpacity:0.55});},
   onEachFeature:(f,l)=>{l.on('click',()=>showZone(f.properties.code));}}).addTo(map);}
function sparkSVG(arr,color){const w=260,h=70,mx=Math.max(...arr,1e-9);
 const pts=arr.map((v,i)=>`${(i/23*w).toFixed(1)},${(h-4-(v/mx)*(h-10)).toFixed(1)}`).join(' ');
 return `<svg width="${w}" height="${h}"><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"/>`+
 `<circle cx="${(state.hour/23*w).toFixed(1)}" cy="${(h-4-((arr[state.hour]??0)/mx)*(h-10)).toFixed(1)}" r="3.5" fill="#d00000"/></svg>`;}
function bars(obj,color){const es=Object.entries(obj).sort((a,b)=>b[1]-a[1]).slice(0,6);
 const mx=Math.max(...es.map(e=>e[1]),1e-9);
 return es.map(([k,v])=>`<div class="bar"><span style="width:110px">${k}</span><span class="fill" style="width:${Math.max(3,v/mx*120)}px;background:${color}"></span><span>${v.toLocaleString()}</span></div>`).join('');}
function showZone(code){const z=DATA.zones[code];if(!z)return;const g=z.grid||{};
 document.getElementById('info').innerHTML=`<b>${z.name}</b> (${code})<br/>`+
 `<table class="info"><tr><td>Peak</td><td><b>${z.peak_mw} MW</b></td></tr>`+
 `<tr><td>Daily energy</td><td>${z.energy_mwh} MWh</td></tr>`+
 `<tr><td>Rooftop PV</td><td>${z.pv_mw} MW</td></tr>`+
 `<tr><td>Transformers</td><td>${z.n_xfmr} × (${z.xfmr_mva} MVA)</td></tr>`+
 `<tr><td>Feeder</td><td>${z.feeder_km} km</td></tr>`+
 (g.loading!==undefined?`<tr><td>Feeder loading</td><td>${g.loading}%</td></tr>`:'')+
 (g.minV!==undefined?`<tr><td>Voltage</td><td>${g.minV}–${g.maxV} pu</td></tr>`:'')+
 (g.viol!==undefined?`<tr><td>Violations</td><td><b>${g.viol}</b></td></tr>`:'')+
 (g.headroom!==undefined?`<tr><td>Headroom</td><td>${(g.headroom*100).toFixed(1)}%</td></tr>`:'')+
 (g.cost?`<tr><td>Upgrade cost</td><td>A$${Math.round(g.cost).toLocaleString()}</td></tr>`:'')+
 `</table><h3>24-h load (MW)</h3>`+sparkSVG(z.hourly.map(v=>v/1000),'#B3541E')+
 `<h3>Building floorspace (m²)</h3>`+(Object.keys(z.stock).length?bars(z.stock,'#6A994E'):'<span class="lbl">n/a</span>')+
 `<h3>Site energy by type (MWh/day)</h3>`+(Object.keys(z.energy_bt).length?bars(z.energy_bt,'#F18F01'):'<span class="lbl">n/a</span>');}
function showSub(s){const z=DATA.zones[s.sa2]||{};
 document.getElementById('info').innerHTML=`<b>Substation ${s.id}</b><br/>${s.name}<br/>`+
 `<table class="info"><tr><td>Capacity</td><td>${s.mva} MVA</td></tr>`+
 `<tr><td>Zone</td><td>${z.name||('SA2 '+s.sa2)}</td></tr>`+
 (s.sa2&&DATA.zones[s.sa2]?`<tr><td>Zone peak</td><td>${DATA.zones[s.sa2].peak_mw} MW</td></tr>`:'')+`</table>`;}
function showFeeder(f){const z=DATA.zones[f.sa2]||{};
 document.getElementById('info').innerHTML=`<b>Feeder ${f.id}</b><br/>`+
 `<table class="info"><tr><td>Loading</td><td><b>${(f.loading||0).toFixed(1)}%</b></td></tr>`+
 `<tr><td>Zone</td><td>${z.name||('SA2 '+f.sa2)}</td></tr></table>`+
 `<span class="lbl">Colour scale: 0% (yellow) → 120%+ (red).</span>`;}
const LAYERS=[{id:'zones',label:'SA2 zones (metric choropleth)'},{id:'subs',label:'Substations'},
 {id:'feeders',label:'Feeders (by loading)'},{id:'viols',label:'Violation outlines'},{id:'bld',label:'Buildings (floorspace)'}];
const on={zones:true,subs:true,feeders:true,viols:true,bld:false};
function drawAll(){on.zones?drawZones():zoneLayer&&map.removeLayer(zoneLayer);
 on.subs||on.feeders?(on.subs&&on.feeders?drawNetwork():0):[subLayer,feedLayer].forEach(l=>{if(l)map.removeLayer(l);});
 if(on.subs&&!on.feeders){drawNetwork();map.removeLayer(feedLayer);}
 if(on.feeders&&!on.subs){drawNetwork();map.removeLayer(subLayer);}
 on.viols?drawViols():violLayer&&map.removeLayer(violLayer);
 on.bld?drawBld():bldLayer&&map.removeLayer(bldLayer);}
function buildLayerPanel(){const d=document.getElementById('layers');d.innerHTML='';
 LAYERS.forEach(L=>{const div=document.createElement('div');div.className='lyr'+(state.selLayer===L.id?' sel':'');
  div.innerHTML=`<label><input type="checkbox" ${on[L.id]?'checked':''}/> <b>${L.label}</b></label><br/>`+
  `<span class="lbl">Colours:</span><select data-l="${L.id}">`+
  Object.keys(SCHEMES).map(s=>`<option ${schemeFor(L.id)===s?'selected':''}>${s}</option>`).join('')+`</select>`;
  div.querySelector('input').onchange=e=>{on[L.id]=e.target.checked;drawAll();};
  div.querySelector('select').onchange=e=>{state.schemes[L.id]=e.target.value;drawAll();};
  div.onclick=()=>{state.selLayer=L.id;buildLayerPanel();};
  d.appendChild(div);});}
function buildMetric(){const s=document.getElementById('metric');s.innerHTML='';
 Object.entries(METRICS).forEach(([k,m])=>{const o=document.createElement('option');o.value=k;o.textContent=m.label;s.appendChild(o);});
 s.value=state.metric;s.onchange=()=>{state.metric=s.value;state.animOn=false;drawZones();};}
function drawSysPlot(){const c=document.getElementById('sysplot'),x=c.getContext('2d');
 const a=DATA.sys_curve,W=c.width,H=c.height,mx=Math.max(...a,1e-9);
 x.clearRect(0,0,W,H);x.strokeStyle='#ddd';for(let g=0;g<=4;g++){x.beginPath();x.moveTo(0,H/4*g);x.lineTo(W,H/4*g);x.stroke();}
 x.strokeStyle='#B3541E';x.lineWidth=2;x.beginPath();
 a.forEach((v,i)=>{const px=i/23*W,py=H-8-(v/mx)*(H-20);i?x.lineTo(px,py):x.moveTo(px,py);});x.stroke();
 const hx=state.hour/23*W;x.strokeStyle='#d00000';x.beginPath();x.moveTo(hx,0);x.lineTo(hx,H);x.stroke();
 document.getElementById('plotkpi').textContent=
  `${(DATA.sys_curve[state.hour]/1000).toFixed(0)} MW @ ${state.hour}:00 | `+
  `daily ${(a.reduce((s,v)=>s+v,0)/1e6).toFixed(1)} GWh`;}
function setHour(h){state.hour=h;document.getElementById('hour').value=h;
 document.getElementById('hourlab').textContent=h+':00';drawSysPlot();
 if(state.animOn)drawZones();}
document.getElementById('hour').oninput=e=>{state.animOn=true;setHour(+e.target.value);};
document.getElementById('play').onclick=()=>{state.playing=!state.playing;state.animOn=true;
 document.getElementById('play').textContent=state.playing?'⏸':'▶';
 if(state.playing){const step=()=>{if(!state.playing)return;
   setHour((state.hour+1)%24);state.timer=setTimeout(step,+document.getElementById('speed').value);};step();}
 else clearTimeout(state.timer);};
document.getElementById('speed').onchange=()=>{};
document.getElementById('kpis').innerHTML=(DATA.kpis||[]).map(k=>`<span class="chip">${k.label}: <b>${k.value}</b></span>`).join('');
buildLayerPanel();buildMetric();drawAll();drawSysPlot();
</script></body></html>"""


if __name__ == "__main__":
    main()
