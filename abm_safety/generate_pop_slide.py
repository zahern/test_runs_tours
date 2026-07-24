# -*- coding: utf-8 -*-
"""Generate interactive SA2 polygon population map using geopandas + actual shapefile."""
import json, csv, os, glob, struct, base64, math
import geopandas as gpd

DEMAND_DIR = r"Z:\test_runs_tours\demand_"
NEW_POP_DIR = r"Z:\test_runs_tours\new_pop\saved_data\zone_ids_sa2"
REF_CSV = os.path.join(DEMAND_DIR, "sa2_zones_reference_357.csv")
DECK_HTML = r"Z:\test_runs_tours\abm_safety\model_review_deck.html"
EXPANSION_CSV = os.path.join(DEMAND_DIR, "planned_expansion_38_zones.csv")
SA2_SHP = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
CDAP_CSV = r"Z:\test_runs_tours\code\runs\ga_parallel\sa_bandit__guided03\cdap_output.csv"
PTYPES = ["FullTimeWorker","PartTimeWorker","UniversityStudent","WorkingStudent",
          "SchoolChild","Preschool","Retired","NonWorker"]

# ── Load shapefile ──
print("Loading SA2 shapefile...")
sa2 = gpd.read_file(SA2_SHP)
sa2 = sa2.rename(columns={"SA2_CODE21": "code", "SA2_NAME21": "name"})
sa2["code"] = sa2["code"].astype(str)
if sa2.crs and sa2.crs.to_epsg() != 4326:
    sa2 = sa2.to_crs(epsg=4326)
print(f"  {len(sa2)} shapes")

# ── Load zone reference ──
ref = {}
with open(REF_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ref[row["sa2_code"]] = {
            "name": row["sa2_name"], "sa3": row["sa3_name"],
            "lat": float(row["centroid_lat"]), "lon": float(row["centroid_lon"]),
        }

# ── Expansion zones ──
exp_codes = set()
if os.path.exists(EXPANSION_CSV):
    with open(EXPANSION_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            exp_codes.add(row["sa2_code"])

# ── Count population ──
zone_pop, zone_ptype = {}, {}

# Read ptype from CDAP output (the raw zone CSVs don't have WORKSTAT/STUDYING/ptype)
print(f"Reading CDAP output for ptype data: {CDAP_CSV}")
cdap_exists = os.path.exists(CDAP_CSV)
if cdap_exists:
    with open(CDAP_CSV) as f:
        for row in csv.DictReader(f):
            zc = str(row.get("SA2_MAIN16", "") or row.get("OTAZ", ""))
            pt = row.get("ptype", "")
            if not zc: continue
            zone_pop[zc] = zone_pop.get(zc, 0) + 1
            if pt:
                if zc not in zone_ptype:
                    zone_ptype[zc] = {}
                zone_ptype[zc][pt] = zone_ptype[zc].get(pt, 0) + 1
else:
    # Fallback: read zone CSVs for total pop only (no ptype)
    print("CDAP output not found, using raw zone CSVs (no ptype breakdown)")
    for zf in glob.glob(os.path.join(NEW_POP_DIR, "zone_*.csv")):
        code = os.path.basename(zf).replace("zone_", "").replace(".csv", "")
        with open(zf) as f:
            zone_pop[code] = max(sum(1 for _ in f) - 1, 0)
total_pop = sum(zone_pop.values())
print(f"  Pop: {total_pop:,} across {len(zone_pop)} zones")
if cdap_exists and zone_ptype:
    pt_totals = {}
    for c, pt in zone_ptype.items():
        for p, cnt in pt.items():
            pt_totals[p] = pt_totals.get(p, 0) + cnt
    for pt in PTYPES:
        print(f"    {pt}: {pt_totals.get(pt, 0):,}")

# ── Simplify polygons ──
def simplify_ring(pts, tol=0.0008):
    if len(pts) <= 2: return pts
    x1, y1 = pts[0]; x2, y2 = pts[-1]
    dx, dy = x2 - x1, y2 - y1
    ls = max(dx*dx + dy*dy, 1e-12)
    md, mi = 0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        t = ((px - x1)*dx + (py - y1)*dy) / ls
        if t < 0: d = math.hypot(px - x1, py - y1)
        elif t > 1: d = math.hypot(px - x2, py - y2)
        else: d = math.hypot(px - (x1 + t*dx), py - (y1 + t*dy))
        if d > md: md, mi = d, i
    if md <= tol: return [pts[0], pts[-1]]
    l = simplify_ring(pts[:mi+1], tol)
    r = simplify_ring(pts[mi:], tol)
    return l[:-1] + r

sa2_idx = {r["code"]: i for i, r in sa2.iterrows()}
all_pts = []; offsets = {}
for code in ref:
    if code not in sa2_idx: continue
    geom = sa2.iloc[sa2_idx[code]].geometry
    if geom is None or geom.is_empty: continue
    start = len(all_pts)
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for poly in polys:
        coords = [(c[0], c[1]) for c in poly.exterior.coords]
        coords = simplify_ring(coords, 0.0008)
        all_pts.append(-len(coords))
        px, py = coords[0]; all_pts.extend([px, py])
        for i in range(1, len(coords)):
            all_pts.extend([coords[i][0] - px, coords[i][1] - py])
            px, py = coords[i]
    offsets[code] = [start, len(all_pts) - start]

print(f"  {len(offsets)} zones with polygon data")

# ── Pack binary ──
buf = bytearray()
for v in all_pts: buf.extend(struct.pack("<f", v))
poly_b64 = base64.b64encode(bytes(buf)).decode()
print(f"  Polygon data: {len(buf):,} bytes -> {len(poly_b64):,} b64 chars")

# ── Build zone data JSON ──
zd = {}
for code in ref:
    pop = zone_pop.get(code, 0)
    pt = zone_ptype.get(code, {})
    st = "a"
    if code in exp_codes: st = "p"
    elif pop == 0: st = "n"
    zd[code] = {
        "n": ref[code]["name"], "s": ref[code]["sa3"],
        "t": st, "p": pop,
        "o": offsets.get(code, [0, 0]),
        "pt": {k: pt.get(k, 0) for k in PTYPES if pt.get(k, 0) > 0},
    }

# Ptype options
pt_totals = {}
for c, pt in zone_ptype.items():
    for p, cnt in pt.items():
        pt_totals[p] = pt_totals.get(p, 0) + cnt
pt_opts = "".join(
    f'<option value="{p}">{p} ({pt_totals.get(p, 0):,})</option>'
    for p in PTYPES
)

# ── Build slide HTML ──
slide = f"""\
      <section class="slide" id="synth-pop-slide" style="padding:1.8% 3%;">
        <h2 class="slide-kicker">Population &amp; Geography</h2>
        <h1 class="slide-title">Synthetic population: {total_pop:,} persons across {len(ref)} SA2 zones</h1>
        <div class="body-area" style="gap:0.4vh; margin-top:0.1vh;">
          <div style="display:flex; gap:0.5vw; align-items:center; flex-wrap:wrap;">
            <span style="font-size:0.64rem; color:var(--text-muted);">Color:</span>
            <button class="pbtn active" data-m="pop" onclick="PM.set(this,'pop')">Population</button>
            <button class="pbtn" data-m="status" onclick="PM.set(this,'status')">Status</button>
            <button class="pbtn" data-m="sa3" onclick="PM.set(this,'sa3')">SA3 District</button>
            <span style="width:1px;align-self:stretch;background:var(--border);margin:0 0.3em;"></span>
            <span style="font-size:0.64rem; color:var(--text-muted);">Ptype:</span>
            <select id="psel" onchange="PM.setPtype(this.value)" style="padding:3px 8px;border:1px solid var(--border);border-radius:4px;font-size:0.58rem;background:var(--surface);color:var(--text);max-width:155px;">
              <option value="total">All persons ({total_pop:,})</option>
              {pt_opts}
            </select>
            <span style="margin-left:auto;font-size:0.5rem;color:var(--text-faint);">Hover for details &middot; Scroll to zoom</span>
          </div>
          <div style="flex:1; position:relative; border-radius:8px; overflow:hidden; border:1px solid var(--border); min-height:0; background:#eef1f5;">
            <canvas id="pmap" style="width:100%;height:100%;display:block;"></canvas>
            <div id="ptip" style="position:absolute;pointer-events:none;background:rgba(30,39,97,0.93);color:#fff;padding:6px 10px;border-radius:5px;font-size:0.56rem;display:none;white-space:nowrap;z-index:10;line-height:1.4;"></div>
          </div>
          <div style="display:flex; gap:0.8vw; font-size:0.56rem; color:var(--text-muted); justify-content:center; margin-top:1px;">
            <span id="pleg"></span>
          </div>
        </div>
      </section>

<style>
.pbtn{{padding:4px 12px;border:1px solid var(--border);border-radius:16px;background:var(--surface);color:var(--text);cursor:pointer;font-size:0.6rem;transition:all .2s;}}
.pbtn.active,.pbtn:hover{{background:var(--primary);color:#fff;border-color:var(--primary);}}
</style>

<script>
(function(){{
  var ZD={json.dumps(zd, separators=(",",":"), ensure_ascii=False)};
  var PB64="{poly_b64}";
  var ZIDS=Object.keys(ZD);
  var SA3C={{}};
  var sl=[...new Set(ZIDS.map(function(k){{return ZD[k].s;}}))].sort();
  sl.forEach(function(s,i){{SA3C[s]='hsl('+Math.round(i*360/sl.length)+',60%,55%)';}});
  var SC={{a:'#2E3B8E',p:'#C85A14',n:'#888'}};
  var cnv,ctx,mode='pop',curPt='total',v={{cx:153.05,cy:-27.55,zm:120}},dr=false,ds=null,hz=null,PBUF=null;

  function dp(){{if(PBUF)return PBUF;var raw=Uint8Array.from(atob(PB64),function(c){{return c.charCodeAt(0);}});PBUF=new Float32Array(raw.buffer);return PBUF;}}
  function gr(off){{var b=dp(),rn=-b[off[0]],pts=[],i=off[0]+1;for(var ri=0;ri<rn;ri++){{var x=b[i],y=b[i+1];i+=2;if(ri===0)pts.push([x,y]);else{{var l=pts[pts.length-1];pts.push([l[0]+x,l[1]+y]);}}}}return pts;}}
  function l2(x,y){{var w=cnv.width/(window.devicePixelRatio||1),h=cnv.height/(window.devicePixelRatio||1);return{{x:(x-v.cx)*v.zm+w/2,y:-(y-v.cy)*v.zm+h/2}};}}
  function gp(z){{return curPt==='total'?z.p:((z.pt||{{}})[curPt]||0);}}
  var ref_zz={{}};ZIDS.forEach(function(id){{ref_zz[id]={{lon:ZD[id].lon||153,lat:ZD[id].lat||-27}};}});

  function fZ(mx,my){{
    for(var i=0;i<ZIDS.length;i++){{var id=ZIDS[i],z=ZD[id],o=z.o;if(!o||o[1]===0)continue;
      var ring=gr(o),inside=false;
      for(var j=0,k=ring.length-1;j<ring.length;k=j++){{
        var a=l2(ring[j][0],ring[j][1]),b=l2(ring[k][0],ring[k][1]);
        if((a.y>my)!==(b.y>my)&&mx<(b.x-a.x)*(my-a.y)/(b.y-a.y)+a.x)inside=!inside;
      }}
      if(inside)return id;
    }}
    return null;
  }}
  function hc(t){{t=Math.max(0,Math.min(1,t));if(t<0.33){{var s=t/0.33;return[Math.round(36+150*s),Math.round(80+100*s),Math.round(160-30*s)];}}if(t<0.66){{var s=(t-0.33)/0.33;return[Math.round(186+30*s),Math.round(180-80*s),Math.round(130-100*s)];}}var s=(t-0.66)/0.34;return[Math.round(216+20*s),Math.round(100-60*s),Math.round(30+10*s)];}}
  var _mxPc=null,_mxPp=null;
  function gmp(){{if(_mxPp===curPt)return _mxPc||1;_mxPp=curPt;_mxPc=0;ZIDS.forEach(function(id){{_mxPc=Math.max(_mxPc,gp(ZD[id]));}});return _mxPc||1;}}
  function gc(z,m){{if(m==='pop'){{var v=Math.min(gp(z),gmp())/Math.max(1,gmp());var c=hc(v);return'rgb('+c[0]+','+c[1]+','+c[2]+')';}}else if(m==='status')return SC[z.t]||'#999';else return SA3C[z.s]||'#888';}}
  function draw(){{
    if(!ctx)return;_mxPc=null;_mxPp=null;
    var w=cnv.width/(window.devicePixelRatio||1),h=cnv.height/(window.devicePixelRatio||1);
    ctx.clearRect(0,0,w,h);
    var order=ZIDS.slice().sort(function(a,b){{return gp(ZD[b])-gp(ZD[a]);}});
    order.forEach(function(id){{
      var z=ZD[id],o=z.o;if(!o||o[1]===0)return;
      ctx.beginPath();var ring=gr(o),f=l2(ring[0][0],ring[0][1]);
      ctx.moveTo(f.x,f.y);for(var i=1;i<ring.length;i++){{var p=l2(ring[i][0],ring[i][1]);ctx.lineTo(p.x,p.y);}}
      ctx.closePath();
      var ih=(id===hz);
      ctx.fillStyle=gc(z,mode);ctx.globalAlpha=ih?0.92:0.72;ctx.fill();ctx.globalAlpha=1;
      ctx.strokeStyle=ih?'#fff':'rgba(255,255,255,0.3)';ctx.lineWidth=ih?1.2:0.4;ctx.stroke();
      if(v.zm>250&&gp(z)>5000){{var c=l2(ref_zz[id].lon,ref_zz[id].lat);ctx.fillStyle='#222';ctx.font='bold 7px system-ui';ctx.textAlign='center';ctx.fillText(z.n.substring(0,12),c.x,c.y);}}
    }});
  }}
  function bl(m){{var el=document.getElementById('pleg');if(m==='pop')el.innerHTML='<span style=\"color:#246e82;\">Low</span> &rarr; <span style=\"color:#ec2828;\">High '+(curPt==='total'?'pop':'count')+'</span>';else if(m==='status')el.innerHTML='<span style=\"display:inline-block;width:8px;height:8px;border-radius:50%;background:#2E3B8E;margin-right:2px;\"></span> Active &nbsp; <span style=\"display:inline-block;width:8px;height:8px;border-radius:50%;background:#C85A14;margin-right:2px;\"></span> Planned &nbsp; <span style=\"display:inline-block;width:8px;height:8px;border-radius:50%;background:#888;margin-right:2px;\"></span> No data';else el.innerHTML='Color = SA3 district';}}
  window.PM=window.PM||{{}};window.PM.set=function(btn,m){{mode=m;document.querySelectorAll('.pbtn').forEach(function(b){{b.classList.remove('active');}});btn.classList.add('active');bl(m);draw();}};
  window.PM.setPtype=function(pt){{curPt=pt;draw();}};

  function init(){{
    cnv=document.getElementById('pmap');if(!cnv)return;ctx=cnv.getContext('2d');
    function rs(){{var r=cnv.parentElement.getBoundingClientRect();cnv.width=r.width*(window.devicePixelRatio||1);cnv.height=r.height*(window.devicePixelRatio||1);ctx.setTransform(1,0,0,1,0,0);ctx.scale(window.devicePixelRatio||1,window.devicePixelRatio||1);cnv.style.width=r.width+'px';cnv.style.height=r.height+'px';draw();}}
    rs();window.addEventListener('resize',rs);
    cnv.addEventListener('mousedown',function(e){{dr=true;ds={{x:e.offsetX,y:e.offsetY,cx:v.cx,cy:v.cy}};}});
    cnv.addEventListener('mousemove',function(e){{if(dr){{v.cx=ds.cx-(e.offsetX-ds.x)/v.zm;v.cy=ds.cy+(e.offsetY-ds.y)/v.zm;draw();return;}}var tip=document.getElementById('ptip'),zid=fZ(e.offsetX,e.offsetY);if(zid!==hz){{hz=zid;if(zid&&ZD[zid]){{var z=ZD[zid],pop=gp(z),st=(z.t==='a'?'Active':z.t==='p'?'Planned':'No data');tip.style.display='block';tip.style.left=(e.offsetX+14)+'px';tip.style.top=(e.offsetY-6)+'px';tip.innerHTML='<b>'+z.n+'</b> ('+zid+')<br>SA3: '+z.s+'<br>'+(curPt==='total'?'Population':'Count')+': '+pop.toLocaleString()+'<br>Status: '+st;}}else tip.style.display='none';draw();}}}});
    cnv.addEventListener('mouseup',function(){{dr=false;}});
    cnv.addEventListener('mouseleave',function(){{dr=false;hz=null;document.getElementById('ptip').style.display='none';draw();}});
    cnv.addEventListener('wheel',function(e){{e.preventDefault();v.zm*=e.deltaY<0?1.12:0.88;v.zm=Math.max(20,Math.min(600,v.zm));draw();}});
    bl('pop');
  }}
  var obs=new MutationObserver(function(){{var el=document.getElementById('synth-pop-slide');if(el&&el.offsetParent!==null&&!cnv)setTimeout(init,100);}});
  var deckEl=document.getElementById('deck');if(deckEl)obs.observe(deckEl,{{attributes:true,attributeFilter:['style']}});
  setTimeout(function(){{var el=document.getElementById('synth-pop-slide');if(el&&el.offsetParent!==null)init();}},600);
}})();
</script>"""

# ── Inject into deck ──
with open(DECK_HTML, encoding="utf-8") as f:
    html = f.read()

os_tag = html.find('<section class="slide" id="sa2-density-slide"')
if os_tag < 0:
    os_tag = html.find('<section class="slide" id="synth-pop-slide"')
oe = html.find("</section>", os_tag)
if os_tag >= 0 and oe >= 0:
    oe += len("</section>")
    trail = html[oe:oe + 500]
    if "<script>" in trail[:500]:
        sc_end = html.find("</script>", oe) + len("</script>")
        oe = sc_end
    html = html[:os_tag] + slide + html[oe:]
    print(f"Replaced slide at offset {os_tag}")

with open(DECK_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Wrote {len(html):,} bytes. Done.")
