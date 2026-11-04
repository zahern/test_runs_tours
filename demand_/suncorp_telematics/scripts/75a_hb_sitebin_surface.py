#!/usr/bin/env python
"""75a: observed hard-braking counts per (fixture, candidate site, rel_bin).

Snaps 2.5M raw HB events (SnappedLat/Lon + EventStart) to the 300 m-separated
candidate pool (75 m buffer), inside each game-day window KO-3h..KO+4h,
binned into 28 x 15-min rel_bins (KO at rel_bin 12, matching 59/60 pipeline).

Output: results_seg/game_day_hb_sitebin_obs.parquet
        columns: fixture, road_osmid, rel_bin, obs_hb
"""
import json, math
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString

ROOT = r"Z:\test_runs_tours\demand_\suncorp_telematics"

sites75 = pd.read_csv(fr"{ROOT}\results_seg\game_day_top20_per_game.csv")
cand = (sites75.groupby(["road_osmid", "road_name", "mid_lat", "mid_lon"], as_index=False)
               .agg(volume=("pred_speeding", "max"))
               .sort_values("volume", ascending=False).reset_index(drop=True))
coord = sites75.drop_duplicates("road_osmid").set_index("road_osmid")
def hav(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))
keep = []
for i, row in cand.iterrows():
    la, lo = coord.loc[row.road_osmid, ["mid_lat", "mid_lon"]]
    if all(hav((la, lo),
               (coord.loc[cand.road_osmid.iloc[k], "mid_lat"],
                coord.loc[cand.road_osmid.iloc[k], "mid_lon"])) > 300.0 for k in keep):
        keep.append(i)
CAND = cand.loc[keep].reset_index(drop=True)
OSM = list(CAND.road_osmid.values)
print("candidate pool:", len(OSM), "sites:", [CAND.set_index('road_osmid').road_name[o] for o in OSM])

gj = json.load(open(fr"{ROOT}\data\osm_roads_suncorp.geojson"))
plx = {f["properties"]["osmid"]: LineString(f["geometry"]["coordinates"])
       for f in gj["features"] if f["properties"]["osmid"] in set(OSM)}
print("polylines matched:", len(plx), "of", len(OSM))

hbv = pd.read_csv(fr"{ROOT}\data\hard_braking_events.csv",
                  usecols=["EventStart", "SnappedLatitude", "SnappedLongitude"],
                  parse_dates=["EventStart"])
print("HB events:", len(hbv))

# fixtures: KO datetimes parsed from names
fx_list = sorted(pd.read_parquet(fr"{ROOT}\results_seg\game_day_pred_15min.parquet").fixture.unique())
ko = {fx: pd.to_datetime(fx.split(" @ ")[-1], format="%Y-%m-%d %H:%M") for fx in fx_list}
for fx in fx_list:
    print(fx, ko[fx], "weekday:", ko[fx].weekday())

pts = shapely.points(hbv.SnappedLongitude.values, hbv.SnappedLatitude.values)
rows = []
for fx, k in ko.items():
    t0, t1 = k - pd.Timedelta(hours=3), k + pd.Timedelta(hours=4)
    w = hbv[(hbv.EventStart >= t0) & (hbv.EventStart < t1)]
    print(fx, "window events:", len(w))
    if not len(w):
        continue
    idx = np.where((hbv.EventStart.values >= t0.to_datetime64()) &
                   (hbv.EventStart.values < t1.to_datetime64()))[0]
    pw = pts[idx]
    rb = ((hbv.EventStart.values[idx] - t0.to_datetime64())
          / np.timedelta64(15, "m")).astype(int).clip(0, 27)
    for osmid, pl in plx.items():
        d = shapely.distance(pw, pl)
        m = d < 75.0
        if m.sum():
            sub = pd.DataFrame({"rel_bin": rb[m]})
            sub["road_osmid"] = osmid
            sub["fixture"] = fx
            rows.append(sub.groupby(["fixture", "road_osmid", "rel_bin"]).size()
                        .rename("obs_hb").reset_index())
obs = pd.concat(rows, ignore_index=True)
# fill zero cells for model training
full = [(fx, o, b) for fx in fx_list for o in plx for b in range(28)]
obs = (obs.set_index(["fixture", "road_osmid", "rel_bin"])
          .reindex(pd.MultiIndex.from_tuples(full, names=["fixture", "road_osmid", "rel_bin"]))
          .fillna(0.0).reset_index())
obs.to_parquet(fr"{ROOT}\results_seg\game_day_hb_sitebin_obs.parquet")
print("HB surface rows:", len(obs), "total HB events in windows:", obs.obs_hb.sum())
print(obs.groupby("fixture").obs_hb.sum())
