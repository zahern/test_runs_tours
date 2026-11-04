"""
22_build_dashboard.py
Build a self-contained interactive HTML dashboard of the time-based risk
profile from the outputs of 21_eb_timewise_risk.py.

The dashboard opens directly in a browser (no server required). It loads
Plotly.js from a CDN for charts and renders the road network as a coordinate
scatter (lon/lat) coloured by risk, so no map-tile token is needed.

Controls
    - Event / Control toggle
    - Hourly mode: a 0-23 clock-hour slider (+ Animate button) over the
      24-hour mean EB risk profile
    - Phase mode: pre / during / post buttons over the phase-resolved EB risk
    - A 24h mean-profile line chart and a Top-20 highest-risk-segment table

Outputs
    figures/risk_dashboard.html
    figures/risk_dashboard_data.json   (the embedded data, for reuse)

Usage
    python 22_build_dashboard.py
"""
import os
import json
import numpy as np
import pandas as pd

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
os.makedirs(FIG, exist_ok=True)

HOURLY = os.path.join(RES, "eb_timewise_hourly.csv")
HOURLY_CTRL = os.path.join(RES, "eb_timewise_hourly_control.csv")
SEG_PHASE = os.path.join(RES, "eb_timewise_segment_phase.csv")
GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
OUT_HTML = os.path.join(FIG, "risk_dashboard.html")
OUT_JSON = os.path.join(FIG, "risk_dashboard_data.json")


def segment_centroids():
    """Return dict osmid -> (lon, lat, name) from the OSM geojson."""
    import geopandas as gpd
    gdf = gpd.read_file(GEOJSON)
    out = {}
    for _, r in gdf.iterrows():
        geom = r["geometry"]
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        name = r.get("name", "") or ""
        out[str(r["osmid"])] = [round(c.x, 5), round(c.y, 5), str(name)]
    return out


def main():
    print("=" * 80)
    print("22_build_dashboard.py - interactive time-based risk dashboard")
    print("=" * 80)

    print("\n1. Loading timewise outputs...")
    hourly = pd.read_csv(HOURLY)
    hourly_ctrl = pd.read_csv(HOURLY_CTRL)
    seg_phase = pd.read_csv(SEG_PHASE)
    print(f"   hourly: {hourly.shape}, hourly_ctrl: {hourly_ctrl.shape}, "
          f"seg_phase: {seg_phase.shape}")

    print("\n2. Computing per-segment risk arrays...")
    osmids = hourly["road_osmid"].astype(str).tolist()
    hourly_arr = hourly[[f"h{h:02d}" for h in range(24)]].values.astype(float)

    def phase_array(tt, ph):
        s = seg_phase[(seg_phase["type"] == tt) & (seg_phase["phase"] == ph)]
        m = s.set_index("road_osmid")["eb"]
        return [float(m.get(int(o), 0.0)) for o in osmids]

    phase_event = {p: phase_array("Event", p) for p in ["pre", "during", "post"]}
    phase_control = {p: phase_array("Control", p) for p in ["pre", "during", "post"]}

    print("\n3. Loading segment geometries (centroids)...")
    cent = segment_centroids()
    lon, lat, names = [], [], []
    for o in osmids:
        if o in cent:
            x, y, n = cent[o]
        else:
            x, y, n = 153.0, -27.47, ""
        lon.append(x); lat.append(y); names.append(n)

    hourly_ctrl_arr = hourly_ctrl[[f"h{h:02d}" for h in range(24)]].values.astype(float)
    data = {
        "osmids": osmids,
        "lon": lon, "lat": lat, "names": names,
        "hourly_event": hourly_arr.tolist(),
        "hourly_control": hourly_ctrl_arr.tolist(),
        "phase_event": phase_event,
        "phase_control": phase_control,
        "mean_profile_event": hourly_arr.mean(axis=0).tolist(),
        "mean_profile_control": hourly_ctrl_arr.mean(axis=0).tolist(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(data, f)
    print(f"   Wrote {OUT_JSON}")

    html = _html_template(data)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"   Wrote {OUT_HTML} ({len(html):,} bytes)")


def _html_template(data):
    payload = json.dumps(data)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Suncorp Telematics - Time-Based Risk Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#0f1620;color:#e6edf3}
  header{padding:14px 20px;background:#16202c;border-bottom:1px solid #243244}
  header h1{margin:0;font-size:18px}
  header small{color:#8aa0b6}
  .controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:12px 20px;background:#121b26;border-bottom:1px solid #243244}
  .controls label{font-size:13px;color:#9fb3c8}
  select,button{background:#1d2b3a;color:#e6edf3;border:1px solid #34495e;border-radius:6px;padding:6px 10px;font-size:13px}
  button{cursor:pointer}
  .grid{display:grid;grid-template-columns:1.4fr 1fr;gap:14px;padding:14px 20px}
  .card{background:#121b26;border:1px solid #243244;border-radius:10px;padding:10px}
  .card h2{margin:6px 10px;font-size:14px;color:#cfe0f0}
  #map{height:520px}
  #profile{height:300px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th,td{padding:4px 8px;border-bottom:1px solid #233143;text-align:left}
  th{color:#9fb3c8}
  .slider{display:flex;align-items:center;gap:10px}
  input[type=range]{width:240px}
</style>
</head>
<body>
<header>
  <h1>Suncorp Stadium &mdash; Time-Based Near-Miss Risk Dashboard</h1>
  <small>Empirical-Bayes adjusted risky-driving signal (braking + speeding) &middot; 7,517 segments &middot; 16 event windows</small>
</header>

<div class="controls">
  <label>Dataset
    <select id="dtype"><option value="event">Event</option><option value="control">Control</option></select>
  </label>
  <label>Mode
    <select id="mode"><option value="hourly">Hourly (clock)</option><option value="phase">Phase (pre/during/post)</option></select>
  </label>
  <label>Phase
    <select id="phase" disabled><option value="pre">Pre-arrival</option><option value="during">During</option><option value="post">Post-dispersal</option></select>
  </label>
  <div class="slider" id="sliderwrap">
    <span>Hour</span>
    <input type="range" id="hour" min="0" max="23" value="19" step="1"/>
    <span id="hourlbl">19:00</span>
  </div>
  <button id="anim">&#9654; Animate</button>
</div>

<div class="grid">
  <div class="card"><h2>Spatial risk profile</h2><div id="map"></div></div>
  <div class="card"><h2>24-hour mean risk profile</h2><div id="profile"></div>
    <h2 style="margin-top:8px">Top 20 highest-risk segments (selected view)</h2>
    <div style="max-height:220px;overflow:auto"><table id="top"><thead>
      <tr><th>#</th><th>Segment</th><th>Risk</th></tr></thead><tbody></tbody></table></div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;
let animTimer=null;

function vals(){
  const dt=document.getElementById('dtype').value;
  const mode=document.getElementById('mode').value;
  if(mode==='phase'){
    const ph=document.getElementById('phase').value;
    return (dt==='event'?DATA.phase_event:DATA.phase_control)[ph];
  }
  const h=+document.getElementById('hour').value;
  return DATA[dt==='event'?'hourly_event':'hourly_control'].map(r=>r[h]);
}

function redraw(){
  const v=vals();
  const txt=DATA.names.map((n,i)=>'Seg '+(DATA.osmids[i])+(n?(' ('+n+')'):''));
  const mapTrace={type:'scattergl',mode:'markers',x:DATA.lon,y:DATA.lat,
    text:txt,hoverinfo:'text',
    marker:{size:6,color:v,colorscale:'Inferno',reversescale:false,
      showscale:true,colorbar:{title:'EB risk'}}};
  Plotly.react('map',[mapTrace],{margin:{t:6,b:6,l:6,r:6},
    xaxis:{title:'Longitude',zeroline:false},yaxis:{title:'Latitude',zeroline:false},
    paper_bgcolor:'#121b26',plot_bgcolor:'#121b26',font:{color:'#cfe0f0'}},
    {displayModeBar:false});

  const mode=document.getElementById('mode').value;
  const dt=document.getElementById('dtype').value;
  const xs = mode==='phase'?['Pre','During','Post']:
    Array.from({length:24},(_,i)=>i+':00');
  const meanProf = dt==='event'?DATA.mean_profile_event:DATA.mean_profile_control;
  const phaseSet = dt==='event'?DATA.phase_event:DATA.phase_control;
  const profileTrace={type:'scatter',mode:'lines+markers',x:xs,
    y: mode==='phase'? ['pre','during','post'].map(p=>
        (phaseSet[p]).reduce((a,b)=>a+b,0)/phaseSet[p].length):
      meanProf, line:{color:'#ff7a59',width:2}};
  Plotly.react('profile',[profileTrace],{margin:{t:6,b:30,l:40,r:10},
    xaxis:{title: mode==='phase'?'Phase':'Clock hour'},yaxis:{title:'Mean EB risk'},
    paper_bgcolor:'#121b26',plot_bgcolor:'#121b26',font:{color:'#cfe0f0'}},
    {displayModeBar:false});

  const order=v.map((val,i)=>[i,val]).sort((a,b)=>b[1]-a[1]).slice(0,20);
  const tb=document.querySelector('#top tbody');tb.innerHTML='';
  order.forEach(([i,val],k)=>{
    const tr=document.createElement('tr');
    tr.innerHTML='<td>'+(k+1)+'</td><td>'+(DATA.osmids[i])+(DATA.names[i]?(' '+(DATA.names[i])):'')+'</td><td>'+val.toFixed(2)+'</td>';
    tb.appendChild(tr);
  });
}

function syncLabels(){
  const h=+document.getElementById('hour').value;
  document.getElementById('hourlbl').textContent=String(h).padStart(2,'0')+':00';
}
document.getElementById('dtype').onchange=redraw;
document.getElementById('phase').onchange=redraw;
document.getElementById('mode').onchange=function(){
  const m=this.value;
  document.getElementById('phase').disabled=(m!=='phase');
  document.getElementById('sliderwrap').style.opacity=(m==='phase'?0.35:1);
  document.getElementById('hour').disabled=(m==='phase');
  redraw();
};
document.getElementById('hour').oninput=function(){syncLabels();redraw();};
document.getElementById('anim').onclick=function(){
  if(animTimer){clearInterval(animTimer);animTimer=null;this.innerHTML='\\u25B6 Animate';return;}
  this.innerHTML='\\u23F8 Pause';
  animTimer=setInterval(()=>{
    let h=(+document.getElementById('hour').value+1)%24;
    document.getElementById('hour').value=h;syncLabels();redraw();
  },700);
};
syncLabels();redraw();
</script>
</body>
</html>
"""
    return html.replace("__PAYLOAD__", payload)


if __name__ == "__main__":
    main()
