"""
49_matsim_export.py - offline MATSim scenario export of the Suncorp study area.

Why: the ANTARES 3D visualizer is a Windows-only C++ app tied to a POLARIS
build and cannot be reused; a MATSim run on the cluster is infeasible
(Maven needs internet for dependencies). What IS feasible offline is a
complete, VIA-ready MATSim scenario built from our own data, plus a
self-contained web 3D outcome visualizer (script 50, ANTARES-pattern:
3D canvas + layer selector + time controls + object info + data plots).

This script writes (pure Python, offline):
  results/matsim/network.xml      directed links from the OSM study-area
      network: length from projected geometry (m), freespeed from posted
      speed (m/s), capacity = lanes x CAP_PER_LANE veh/h, permlanes, car-only.
      oneway=yes -> single directed link, else both directions (Polaris-style
      bidirectional links with per-direction lanes/speeds).
  results/matsim/population.xml   illustrative car demand: background agents
      (random node OD, spread over the day) + event surge (stadium-bound
      pre-kickoff, outbound post) for a reference fixture scaled from
      fixture attendance x EVENT_SHARE. Demand levels are illustrative and
      labelled as such; link capacities carry the real geometry.
  results/matsim/facilities.xml   police posts as activity facilities
      (actType="police_post") so VIA can display the deployment.
  results/matsim/README_VIA.md    how to open in VIA / OTFVis and what to look at.

Inputs: data/osm_roads_suncorp.geojson, data/road_geometry_features.csv,
        data/fixture_calendar.csv, results/speeding_only_posts.csv
        (falls back to results/post_selection_mip.csv).
Outputs: results/matsim/* (see above).

Usage: python 49_matsim_export.py [--n-bg 2000] [--ref-fixture "NRL Broncos-Dolphins"]
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import pandas as pd

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
OUT = os.path.join(RES, "matsim")

GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
FEATS = os.path.join(DATA, "road_geometry_features.csv")
CAL = os.path.join(DATA, "fixture_calendar.csv")
for _p in (os.path.join(RES, "speeding_only_posts.csv"),
           os.path.join(RES, "post_selection_mip.csv")):
    if os.path.exists(_p):
        POSTS = _p
        break
else:
    POSTS = None

CAP_PER_LANE = 800.0   # veh/h/lane, urban arterial default (documented assumption)
FREESPEED_DEF = 13.9   # m/s (50 km/h) where no limit is known
EVENT_SHARE = 0.02     # modelled agents per attendee (illustrative demand scale)
STADIUM_LON, STADIUM_LAT = 153.0093, -27.4647


def _parse_speed(v):
    try:
        return float(str(v).split()[0])
    except (ValueError, TypeError, AttributeError):
        return np.nan


def _parse_lanes(v):
    try:
        return float(str(v).split()[0].split(";")[0])
    except (ValueError, TypeError, AttributeError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bg", type=int, default=2000)
    ap.add_argument("--ref-fixture", type=str, default="")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(int(args.seed))
    os.makedirs(OUT, exist_ok=True)
    print("=" * 80)
    print("49_matsim_export.py - offline MATSim scenario export")
    print("=" * 80)

    import geopandas as gpd
    g = gpd.read_file(GEOJSON)
    g["osmid"] = pd.to_numeric(g["osmid"], errors="coerce")
    g = g.dropna(subset=["osmid"]).copy()
    g["osmid"] = g["osmid"].astype(np.int64)
    feat = pd.read_csv(FEATS) if os.path.exists(FEATS) else pd.DataFrame()
    if len(feat):
        g = g.merge(feat[["road_osmid", "lanes", "maxspeed_kmh",
                          "length_m"]].rename(columns={"road_osmid": "osmid"}),
                    on="osmid", how="left", suffixes=("", "_f"))
        for c in ["lanes", "maxspeed_kmh"]:
            fc = c + "_f"
            if fc in g.columns:
                g[c] = g[fc].combine_first(g[c])
    gu = g.to_crs(28356)

    # ---- nodes: snap endpoints on a 0.1 m grid (Polaris-style rounding) -----
    def key(pt):
        return (round(float(pt[0]), 1), round(float(pt[1]), 1))

    nodes, node_id_of, links = {}, {}, []
    for _, r in gu.iterrows():
        geom = r.geometry
        coords = list(geom.coords) if geom is not None else []
        if len(coords) < 2:
            continue
        a, b = key(coords[0]), key(coords[-1])
        for p in (a, b):
            if p not in node_id_of:
                node_id_of[p] = f"n{len(node_id_of)}"
                # back-project node to lon/lat via segment centroid offset is
                # unnecessary: MATSim accepts any CRS as long as consistent.
                nodes[p] = p
        length = float(geom.length)
        lanes = _parse_lanes(r.get("lanes"))
        lanes = float(np.clip(lanes, 1, 6)) if np.isfinite(lanes) else 1.0
        vmax = _parse_speed(r.get("maxspeed"))
        if not np.isfinite(vmax):
            vmax = 50.0
        freespeed = max(float(vmax) / 3.6, 2.0)
        oneway = str(r.get("oneway", "")).lower() in ("yes", "true", "1")
        dirs = [(a, b)] if oneway else [(a, b), (b, a)]
        for k, (u, v) in enumerate(dirs):
            links.append({
                "id": f"{int(r['osmid'])}_{k}",
                "from": node_id_of[u], "to": node_id_of[v],
                "length": round(length, 1), "freespeed": round(freespeed, 2),
                "capacity": round(lanes * CAP_PER_LANE, 1),
                "permlanes": lanes, "oneway": "true" if oneway else "false",
                "modes": "car", "origid": str(int(r["osmid"])),
            })
    print(f"   nodes={len(nodes):,} links={len(links):,}")

    net = ET.Element("network", {"name": "suncorp_study_area"})
    ET.SubElement(net, "nodes")
    nn = net.find("nodes")
    for p, nid in node_id_of.items():
        ET.SubElement(nn, "node", {"id": nid, "x": str(p[0]), "y": str(p[1])})
    ET.SubElement(net, "links")
    ll = net.find("links")
    for L in links:
        ET.SubElement(ll, "link", {k: str(v) for k, v in L.items()})
    with open(os.path.join(OUT, "network.xml"), "w") as f:
        f.write(minidom.parseString(
            ET.tostring(net, encoding="unicode")).toprettyxml(indent="  "))
    print("   wrote network.xml")

    # ---- population: background + event surge --------------------------------
    cal = pd.read_csv(CAL)
    ref = cal[cal["label"] == args.ref_fixture]
    ref = ref.iloc[0] if len(ref) else cal.loc[cal["attendance"].idxmax()]
    n_event = int(float(ref["attendance"]) * EVENT_SHARE)
    node_ids = list(node_id_of.values())
    # stadium node: nearest network node to the stadium (planar CRS).
    st = gpd.GeoSeries([gpd.points_from_xy([STADIUM_LON],
                                           [STADIUM_LAT])[0]],
                       crs=4326).to_crs(28356)
    sx, sy = float(st.x.iloc[0]), float(st.y.iloc[0])
    d2 = np.array([[(p[0] - sx) ** 2 + (p[1] - sy) ** 2] for p in node_id_of])
    stad_node = node_ids[int(np.argmin(d2))]

    pop = ET.Element("population", {"name": "suncorp_event"})
    pid = 0

    def add_person(dep_node, arr_node, dep_s):
        nonlocal pid
        p = ET.SubElement(pop, "person", {"id": f"p{pid}"})
        pid += 1
        plan = ET.SubElement(p, "plan", {"selected": "yes"})
        ET.SubElement(plan, "activity",
                      {"type": "home", "link": dep_node, "end_time": str(dep_s)})
        ET.SubElement(plan, "leg", {"mode": "car"})
        ET.SubElement(plan, "activity", {"type": "stadium", "link": arr_node})

    # background across the day
    for _ in range(int(args.n_bg)):
        a, b = rng.choice(node_ids, size=2, replace=False)
        add_person(a, b, f"{int(rng.integers(6 * 3600, 22 * 3600))}")
    # event surge: inbound KO-3h..KO-0.5h, outbound KO+1.5h..KO+4h
    ko = pd.to_datetime(ref["kickoff_iso"])
    ko_s = int((ko - ko.normalize()).total_seconds())
    for _ in range(n_event // 2):
        a = rng.choice(node_ids)
        add_person(a, stad_node,
                   f"{ko_s + int(rng.integers(-3 * 3600, -1800))}")
    for _ in range(n_event - n_event // 2):
        b = rng.choice(node_ids)
        add_person(stad_node, b,
                   f"{ko_s + int(rng.integers(5400, 4 * 3600))}")
    with open(os.path.join(OUT, "population.xml"), "w") as f:
        f.write(minidom.parseString(
            ET.tostring(pop, encoding="unicode")).toprettyxml(indent="  "))
    print(f"   wrote population.xml ({pid:,} agents; ref fixture "
          f"'{ref['label']}', attendance {float(ref['attendance']):,.0f})")

    # ---- facilities: police posts ---------------------------------------------
    fac = ET.Element("facilities", {"name": "suncorp_police_posts"})
    if POSTS is not None:
        posts = pd.read_csv(POSTS)
        for i, r in posts.reset_index(drop=True).iterrows():
            f = ET.SubElement(fac, "facility",
                              {"id": f"post{i}",
                               "x": str(r["lon"]), "y": str(r["lat"])})
            ET.SubElement(f, "activity", {"type": "police_post"})
        print(f"   wrote facilities.xml ({len(posts)} posts from "
              f"{os.path.basename(POSTS)})")
    else:
        print("   WARN: no posts file; facilities.xml left empty")
    with open(os.path.join(OUT, "facilities.xml"), "w") as f:
        f.write(minidom.parseString(
            ET.tostring(fac, encoding="unicode")).toprettyxml(indent="  "))
    with open(os.path.join(OUT, "README_VIA.md"), "w") as f:
        f.write(
            "# Suncorp MATSim scenario - VIA quick start\n\n"
            "Files: `network.xml` (study-area road network, EPSG:28356 planar "
            "metres), `population.xml` (illustrative background + event-surge "
            f"car demand, ref fixture '{ref['label']}'), `facilities.xml` "
            "(police posts as `police_post` activities).\n\n"
            "Open in VIA: load `network.xml`, add `population.xml` as demand "
            "and `facilities.xml` as facilities. Run the MATSim simulation "
            "first to produce `output_events.xml.gz` for the animation tab; "
            "colour links by entered/left volumes to see the arrival/dispersal "
            "peaks the deployment roster targets.\n\n"
            "Assumptions: capacity = lanes x 800 veh/h; freespeed = posted "
            "limit; demand scale is illustrative (see script header).\n")
    print("DONE! MATSim export complete.")


if __name__ == "__main__":
    main()
