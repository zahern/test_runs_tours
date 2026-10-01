"""
50_outcome_visualizer_3d.py - ANTARES-pattern 3D outcome visualizer.

The POLARIS ANTARES visualizer is a Windows-only C++ app bound to a POLARIS
build and cannot be reused; this script replicates its *pattern* as a
self-contained web page driven by OUR outputs (pure predicted speeding, no
risk weighting):

  3D canvas      - deck.gl: network links coloured/extruded by predicted
                   finable speeding per hour, police posts as 3D columns,
                   officer tokens at staffed posts, stadium marker.
  Layer selector - toggles for links / posts / officers / labels.
  Object info    - click a link (segment stats) or post (served speeding,
                   staffed or not).
  Time controls  - play/pause + slider over the 7 event-relative hourly bins
                   (KO-3h..KO+4h) with phase label; officer tokens hold the
                   season-mean static roster (the MIP optimum is near-static).
  Data plots     - coverage bar (staffed vs random vs none) + hourly profile.

Opens directly in a browser (CDN: deck.gl, maplibre-gl, plotly - same pattern
as 22_build_dashboard.py). Reference fixture = highest-attendance calendar
fixture.

Inputs: data/osm_roads_suncorp.geojson,
        results/segment_speeding_pred_fixture.csv.gz (reference fixture rows),
        results/segment_speeding_pred_post_hourly.csv,
        results/speeding_only_posts.csv (fallback: post_selection_mip.csv),
        results/speeding_only_deployment.csv, results/speeding_only_audit.json
Outputs: figures/outcome_3d.html, figures/outcome_3d_data.json

Usage: python 50_outcome_visualizer_3d.py [--ref-fixture ""] [--top-links 7825]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
PRED_FIX = os.path.join(RES, "segment_speeding_pred_fixture.csv.gz")
POST_H = os.path.join(RES, "segment_speeding_pred_post_hourly.csv")
DEPLOY = os.path.join(RES, "speeding_only_deployment.csv")
AUDIT = os.path.join(RES, "speeding_only_audit.json")
for _p in (os.path.join(RES, "speeding_only_posts.csv"),
           os.path.join(RES, "post_selection_mip.csv")):
    if os.path.exists(_p):
        POSTS = _p
        break
else:
    POSTS = None

HBINS = [-3, -2, -1, 0, 1, 2, 3]
PHASE_OF = {-3: "pre", -2: "pre", -1: "pre", 0: "during", 1: "during",
            2: "post", 3: "post"}
STADIUM = {"lon": 153.0093, "lat": -27.4647}


def _color(t):
    # green -> yellow -> red ramp
    t = float(np.clip(t, 0, 1))
    if t < 0.5:
        k = t / 0.5
        return [int(11 + k * (255 - 11)), int(110 + k * (200 - 110)), 79]
    k = (t - 0.5) / 0.5
    return [255, int(200 - k * (200 - 40)), int(79 - k * 79)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-fixture", type=str, default="")
    ap.add_argument("--top-links", type=int, default=7825)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    for f in (PRED_FIX, POST_H, POSTS):
        if f is None or not os.path.exists(f):
            raise SystemExit(f"FATAL: missing {f} (run 47+48 first)")
    print("=" * 80)
    print("50_outcome_visualizer_3d.py - 3D outcome visualizer (ANTARES pattern)")
    print("=" * 80)

    import geopandas as gpd
    hz = pd.read_csv(PRED_FIX)
    if args.ref_fixture:
        ref = args.ref_fixture
    else:
        ref = hz.loc[hz["h_total"].groupby(hz["fixture"]).transform("sum")
                     .idxmax(), "fixture"]
    hz = hz[hz["fixture"] == ref].copy()
    print(f"   reference fixture: {ref} ({len(hz):,} segments)")
    ph = pd.read_csv(POST_H)  # post, hbin, h_pred (train-mean hourly)
    posts = pd.read_csv(POSTS).reset_index(drop=True)
    audit = json.load(open(AUDIT)) if os.path.exists(AUDIT) else {}
    P = int(audit.get("P", 5))
    J = len(posts)

    # staffed roster: top-P posts by train-mean served speeding (season mean)
    post_tot = ph.groupby("post")["h_pred"].sum()
    order = np.argsort([-float(post_tot.get(j, 0.0)) for j in range(J)])
    staffed = set(order[:min(P, J)].tolist())

    # per-link hourly predictions: rebuild from phase shares is unnecessary -
    # use h_total split by post-hourly profile shape per link is overkill;
    # honest simplification: link hourly = h_total * network hour-shares.
    hh = ph.groupby("hbin")["h_pred"].sum()
    hshare = {int(h): float(hh.get(h, 0)) / max(float(hh.sum()), 1e-12)
              for h in HBINS}
    g = gpd.read_file(GEOJSON)
    g["osmid"] = pd.to_numeric(g["osmid"], errors="coerce")
    g = g.dropna(subset=["osmid"])
    pred = dict(zip(hz["road_osmid"].astype(np.int64), hz["h_total"]))
    names = dict(zip(g["osmid"].astype(np.int64),
                     g.get("name", pd.Series("", index=g.index))
                     .fillna("").astype(str)))
    links = []
    for _, r in g.iterrows():
        o = int(r["osmid"])
        tot = float(pred.get(o, 0.0))
        geom = r.geometry
        coords = []
        if geom is not None and not geom.is_empty:
            parts = getattr(geom, "geoms", [geom])
            for part in parts:
                coords.extend([[round(float(x), 5), round(float(y), 5)]
                               for x, y in part.coords])
        if not coords:
            continue
        links.append({"id": o, "name": names.get(o, ""),
                      "path": coords, "h_total": tot,
                      "h": [tot * hshare[h] for h in HBINS]})
    links = sorted(links, key=lambda d: -d["h_total"])[:int(args.top_links)]
    mx = max([d["h_total"] for d in links] + [1e-9])
    for d in links:
        d["t"] = d["h_total"] / mx
        d["color"] = _color(d["t"])
    print(f"   links embedded: {len(links):,} (max {mx:.1f} events)")

    post_feat = []
    for j in range(J):
        r = posts.iloc[j]
        served = float(post_tot.get(j, 0.0))
        post_feat.append({"id": int(j), "lon": float(r["lon"]),
                          "lat": float(r["lat"]),
                          "served": served,
                          "staffed": bool(j in staffed),
                          "h": [float(ph[(ph["post"] == j)
                                         & (ph["hbin"] == h)]["h_pred"].sum())
                                for h in HBINS]})
    officers = [{"id": int(n), "post": int(j),
                 "lon": float(posts.iloc[j]["lon"]),
                 "lat": float(posts.iloc[j]["lat"])}
                for n, j in enumerate(sorted(staffed))]
    cov = []
    if os.path.exists(DEPLOY):
        cov = pd.read_csv(DEPLOY).to_dict(orient="records")
    hourly_net = [float(hh.get(h, 0.0)) for h in HBINS]

    data = {"fixture": ref, "P": P, "J": J, "stadium": STADIUM,
            "hbins": HBINS, "phases": [PHASE_OF[h] for h in HBINS],
            "links": links, "posts": post_feat, "officers": officers,
            "coverage": cov, "hourly_net": hourly_net}
    with open(os.path.join(FIG, "outcome_3d_data.json"), "w") as f:
        json.dump(data, f)
    print(f"   data json: {os.path.getsize(os.path.join(FIG, 'outcome_3d_data.json')) / 1e6:.1f} MB")

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Suncorp deployment outcome - 3D visualizer</title>
<script src="https://unpkg.com/deck.gl@9.0.0/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{margin:0;font-family:sans-serif;display:flex;height:100vh}
#map{flex:3;position:relative}
#side{flex:1;min-width:300px;padding:10px;overflow-y:auto;background:#111;color:#eee}
.layer{display:block;margin:4px 0}
#info{background:#222;padding:8px;border-radius:6px;min-height:80px;margin-top:8px;font-size:13px}
#controls{position:absolute;bottom:12px;left:12px;background:rgba(17,17,17,.85);color:#eee;padding:10px;border-radius:8px;z-index:5}
button{margin:0 4px}
</style></head><body>
<div id="map"></div>
<div id="side">
<h3>Deployment outcome (pure speeding)</h3>
<div id="layers">
<label class="layer"><input type="checkbox" id="l-links" checked/> Links (predicted speeding)</label>
<label class="layer"><input type="checkbox" id="l-posts" checked/> Police posts</label>
<label class="layer"><input type="checkbox" id="l-off" checked/> Officers</label>
<label class="layer"><input type="checkbox" id="l-lab"/> Post labels</label>
</div>
<div id="info">Click a link or post for details.</div>
<div id="covplot"></div>
<div id="hourplot"></div>
</div>
<div id="controls">
<button id="play">&#9654;</button>
<input type="range" id="time" min="0" max="6" value="0" step="1"/>
<span id="tlabel"></span>
</div>
<script>
const D = DATAJSON;
let hour = 0, playing = false, timer = null;
const deckgl = new deck.DeckGL({
  container: 'map',
  mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  initialViewState: {longitude: D.stadium.lon, latitude: D.stadium.lat, zoom: 13.5, pitch: 50},
  controller: true,
  getTooltip: ({object}) => object && object.tip
});
function linkColor(d){ const c = d.color; return [c[0], c[1], c[2], 200]; }
function render(){
  const h = D.hbins[hour];
  document.getElementById('tlabel').textContent =
    'KO' + (h>=0?'+':'') + h + 'h (' + D.phases[hour] + ')  fixture: ' + D.fixture;
  const showL = document.getElementById('l-links').checked;
  const showP = document.getElementById('l-posts').checked;
  const showO = document.getElementById('l-off').checked;
  const showT = document.getElementById('l-lab').checked;
  const layers = [];
  if (showL) layers.push(new deck.PathLayer({
    id:'links', data: D.links, getPath: d=>d.path,
    getColor: d=>linkColor(d), getWidth: d=> 2 + 60*d.h[hour]/Math.max(...D.hourly_net,1e-9),
    widthMinPixels: 1, pickable: true,
    onClick: (info)=>{ if(info.object){ const d=info.object;
      document.getElementById('info').innerHTML =
      '<b>Segment '+d.id+'</b> '+(d.name||'(unnamed)')+'<br/>Predicted speeding this hour: '+d.h[hour].toFixed(2)+'<br/>Fixture total: '+d.h_total.toFixed(1); } }
  }));
  if (showP) layers.push(new deck.ColumnLayer({
    id:'posts', data: D.posts, getPosition: d=>[d.lon,d.lat],
    getElevation: d=> 30 + 400*d.h[hour], radius: 40,
    getFillColor: d=> d.staffed ? [11,110,79,220] : [120,120,120,160],
    extruded: true, pickable: true,
    onClick: (info)=>{ if(info.object){ const d=info.object;
      document.getElementById('info').innerHTML =
      '<b>Post '+d.id+'</b> '+(d.staffed?'STAFFED':'not staffed')+'<br/>Served speeding this hour: '+d.h[hour].toFixed(2); } }
  }));
  if (showO) layers.push(new deck.ScatterplotLayer({
    id:'officers', data: D.officers, getPosition: d=>[d.lon,d.lat],
    getRadius: 60, getFillColor: [0,180,255,230], pickable: true,
    onClick: (info)=>{ if(info.object){ const d=info.object;
      document.getElementById('info').innerHTML = '<b>Officer '+d.id+'</b> holding post '+d.post+' (season-mean static roster)'; } }
  }));
  if (showT) layers.push(new deck.TextLayer({
    id:'labels', data: D.posts.filter(d=>d.staffed),
    getPosition: d=>[d.lon,d.lat], getText: d=>'P'+d.id, size: 14
  }));
  layers.push(new deck.ScatterplotLayer({id:'stad', data:[D.stadium],
    getPosition: d=>[d.lon,d.lat], getRadius: 120, getFillColor:[255,215,0,220]}));
  deckgl.setProps({layers});
}
['l-links','l-posts','l-off','l-lab'].forEach(id=>document.getElementById(id).onchange=render);
document.getElementById('time').oninput = (e)=>{ hour=+e.target.value; render(); };
document.getElementById('play').onclick = ()=>{
  playing=!playing; document.getElementById('play').innerHTML = playing?'&#10074;&#10074;':'&#9654;';
  if(playing){ timer=setInterval(()=>{ hour=(hour+1)%7; document.getElementById('time').value=hour; render(); },1200); }
  else clearInterval(timer);
};
Plotly.newPlot('covplot', [{x: D.coverage.map(c=>c.scenario), y: D.coverage.map(c=>c.coverage),
  type:'bar', marker:{color:['#0B6E4F','#888','#5b8fb0']}}],
  {title:'Share of predicted speeding detected', paper_bgcolor:'#111', plot_bgcolor:'#111', font:{color:'#eee'}, height:260, margin:{t:40}});
Plotly.newPlot('hourplot', [{x: D.hbins.map(h=>'KO'+(h>=0?'+':'')+h+'h'), y: D.hourly_net, type:'scatter', mode:'lines+markers'}],
  {title:'Network predicted speeding by hour', paper_bgcolor:'#111', plot_bgcolor:'#111', font:{color:'#eee'}, height:240, margin:{t:40}});
render();
</script></body></html>"""
    html = html.replace("DATAJSON", json.dumps(data))
    with open(os.path.join(FIG, "outcome_3d.html"), "w") as f:
        f.write(html)
    print("   wrote outcome_3d.html")
    print("DONE! 3D outcome visualizer complete.")


if __name__ == "__main__":
    main()
