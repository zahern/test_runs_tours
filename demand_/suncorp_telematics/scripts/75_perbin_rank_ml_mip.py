#!/usr/bin/env python
"""Per-bin site ranking + ML (speeding + hard-braking) value surface -> MIP.

1) Snap observed hard-braking events (2.5M rows, lat/lon+timestamp) to the
   300 m-separated candidate pool within 75 m, in each fixture's game-day
   window  -> observed HB counts per (fixture, site, rel_bin).
2) ML model: sklearn HistGradientBoostingRegressor(loss='poisson') per site-bin:
   features = rel_bin, log exposure exp, flow_uplift, road geometry (curvature,
   speed, lanes), AADT, dist_stadium, road class + fixture type encoding.
   Target = observed HB counts; trained on first 4 fixtures, holdout = last 2.
   Speeding margin: existing per-bin predictions from the exact parquet.
   Combined value = pred_speeding + w_hb * pred_hb  (w_hb scales HB to the
   speeding event scale using totals across train fixtures).
3) Per-bin ranking churn: top-5 per rel_bin, overlapturnover across bins.
4) Oracle per-bin mobility ceiling on the COMBINED surface vs static set.
5) Per-bin MIP: 28 bin layers, 5 officers, <=2 relocations/fixture;
   strategies: combined_static vs combined_binwise_optimal.

Outputs:
   results_seg/game_day_hb_sitebin_obs_parquet      (snapped HB surface)
   results_seg/game_day_hb_ml_holdout_metrics.json
   results_seg/game_day_value_surface_perbin.csv
   results_seg/game_day_perbin_rank_churn.csv
   results_seg/game_day_perbin_ml_mip_value.csv
"""
import json, math
import numpy as np
import pandas as pd

ROOT = r"Z:\test_runs_tours\demand_\suncorp_telematics"

# ---------- candidate pool (identical construction to script 74) ----------
sites75 = pd.read_csv(fr"{ROOT}\results_seg\game_day_top20_per_game.csv")
cand = (sites75.groupby(["road_osmid", "road_name", "mid_lat", "mid_lon"], as_index=False)
               .agg(volume=("pred_speeding", "max"))
               .sort_values("volume", ascending=False).reset_index(drop=True))
coord = sites75.drop_duplicates("road_osmid").set_index("road_osmid")
def hav(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2 * R * math.asin(math.sqrt(h))
keep = []
for i, row in cand.iterrows():
    la, lo = coord.loc[row.road_osmid, ["mid_lat", "mid_lon"]]
    if all(hav((la, lo), (coord.loc[cand.road_osmid.iloc[k], ["mid_lat", "mid_lon"]])
               if False else
               (coord.loc[cand.road_osmid.iloc[k], "mid_lat"],
                coord.loc[cand.road_osmid.iloc[k], "mid_lon"])) > 300.0 for k in keep):
        keep.append(i)
    else:
        pass
CAND = cand.loc[keep].reset_index(drop=True)
OSM = list(CAND.road_osmid.values); S = len(OSM)
SITE_LL = {r.road_osmid: (r.mid_lat, r.mid_lon) for r in CAND.itertuples()}
print("candidate pool:", S, "sites")

fixtures = sorted(df.fixture.unique()) if False else sorted(
    pd.read_parquet(fr"{ROOT}\results_seg\game_day_pred_15min.parquet").fixture.unique())
PHASES = {
    fixtures[0]: (15, 21), fixtures[1]: (16, 22), fixtures[2]: (3, 10),
    fixtures[3]: (3, 8), fixtures[4]: (3, 12), fixtures[5]: (15, 21)}
def phase_of(f, rb):
    a, b = PHASES[f]
    return "pre" if rb < a else ("during" if rb <= b else "post")

# ---------- 1) snap HB events ----------
from shapely.geometry import Point, LineString
gj = json.load(open(fr"{ROOT}\data\osm_roads_suncorp.geojson"))
cand_osm = set(OSM)
polylines = {f["properties"]["osmid"]: LineString(f["geometry"]["coordinates"])
             for f in gj["features"] if f["properties"]["osmid"] in cand_osm}
print("polylines matched:", len(polylines))
ROAD_LL_WGS = {o: ([(x, y) for x, y in list(pl.coords)])
               for o, pl in polylines.items()}

hb = pd.read_csv(fr"{ROOT}\data\hard_braking_events.csv",
                 usecols=["EventStart", "SnappedLatitude", "SnappedLongitude"],
                 parse_dates=["EventStart"])
cal = pd.read_csv(fr"{ROOT}\data\fixture_calendar.csv")
print("calendar cols:", cal.columns.tolist())
# find ko time per fixture ('fixture' strings have @ <time>)
fix_when = {}
for fx in fixtures:
    dt = pd.to_datetime(fx.split(" @ ")[-1], format="%Y-%m-%d %H:%M")
    fix_when[fx] = dt
    print(fx, "->", dt, dt.weekday())
