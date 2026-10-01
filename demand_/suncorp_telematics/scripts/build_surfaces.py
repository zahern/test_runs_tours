"""
Faithful reconstruction of the per-fixture space-time risk surfaces of the
2026-08-28 12:50 deployment run (validated: P=5 coverage == 0.6398).

Assignment matches the run that produced the paper's numbers: every in-study-area
event is mapped to its NEAREST candidate post (Voronoi tessellation of the study
area; no distance cap), then binned into the day's 96 x 15-min buckets and sliced
to the 28-bucket window from the top of the hour of (kickoff - 3h).

Outputs results/surfaces.npz and prints the snap-distance distribution so the
catchment can be described honestly in the paper.
"""
import json, numpy as np, pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree

BBOX = dict(lon_min=152.98, lon_max=153.05, lat_min=-27.49, lat_max=-27.44)
ORDINARY = ("NonEvent Wed 18Mar", "2026-03-18T16:00")

posts = pd.read_csv("results/post_selection_mip.csv")
plat, plon = posts.lat.to_numpy(), posts.lon.to_numpy()
prank = posts["rank"].to_numpy(); node_id = posts["node_id"].to_numpy()
J = len(plat)
tree = cKDTree(np.column_stack([plat, plon]))

cal = pd.read_csv("data/fixture_calendar.csv")
labels = list(cal["label"]); kickoffs = list(cal["kickoff_iso"])
crowds = list(cal["attendance"]); types = list(cal["fixture_type"])
fixtures = list(zip(labels, kickoffs, crowds, types))
need = set(pd.Timestamp(k).date() for k in kickoffs)
need.add(pd.Timestamp(ORDINARY[1]).date())

acc = defaultdict(lambda: np.zeros((J, 96)))
snap = []  # sampled snap distances (m) for reporting
for ch in pd.read_csv("data/hard_braking_events.csv",
                      usecols=["EventStart", "SnappedLatitude", "SnappedLongitude", "SpeedLoss_kmh"],
                      parse_dates=["EventStart"], chunksize=500_000):
    ch = ch.rename(columns={"EventStart": "ts", "SnappedLatitude": "lat", "SnappedLongitude": "lon"})
    ch = ch[(ch.lon >= BBOX["lon_min"]) & (ch.lon <= BBOX["lon_max"]) &
            (ch.lat >= BBOX["lat_min"]) & (ch.lat <= BBOX["lat_max"])]
    ch = ch[ch.ts.dt.date.isin(need)]
    if ch.empty:
        continue
    coords = np.column_stack([ch.lat.to_numpy(float), ch.lon.to_numpy(float)])
    v = np.isfinite(coords).all(1); ch = ch[v]; coords = coords[v]
    _, idx = tree.query(coords)  # nearest post, no cap
    dlat = np.radians(coords[:, 0] - plat[idx]); dlon = np.radians(coords[:, 1] - plon[idx])
    a = np.sin(dlat/2)**2 + np.cos(np.radians(coords[:, 0]))*np.cos(np.radians(plat[idx]))*np.sin(dlon/2)**2
    dm = 2*6371000.0*np.arcsin(np.sqrt(a))
    snap.append(dm)
    b = (ch.ts.dt.hour*4 + ch.ts.dt.minute//15).to_numpy()
    w = np.where(ch.SpeedLoss_kmh.to_numpy(float) > 60, 0.5, 1.0)
    dser = ch.ts.dt.date.to_numpy()
    for dd in np.unique(dser):
        mm = dser == dd
        np.add.at(acc[dd], (idx[mm], b[mm]), w[mm])

def window(kickoff):
    A = acc.get(pd.Timestamp(kickoff).date())
    sb = (pd.Timestamp(kickoff) - pd.Timedelta(hours=3)).hour * 4
    return A[:, sb:sb+28] if A is not None else np.zeros((J, 28))

acc_fx = np.stack([window(k) for k in kickoffs])
r_event = acc_fx.mean(axis=0)
r_ord = window(ORDINARY[1])

snap = np.concatenate(snap)
pct = {p: float(np.percentile(snap, p)) for p in (50, 75, 90, 95)}
within500 = float((snap <= 500).mean())
print(f"snap dist (m): median {pct[50]:.0f}  p90 {pct[90]:.0f}  p95 {pct[95]:.0f}  "
      f"within 500 m {100*within500:.1f}%")

tot = r_event.sum(axis=1); top5 = np.argsort(-tot)[:5]
cov5 = r_event[top5].sum() / r_event.sum()
print(f"[{len(kickoffs)} real fixtures] r_event.sum = {r_event.sum():.0f}  "
      f"top-5 static coverage = {cov5:.4f}")
print("top-5 posts:", [f"P{int(prank[j])}" for j in top5])

np.savez_compressed("results/surfaces.npz",
                    acc_fx=acc_fx, r_event=r_event, r_ord=r_ord,
                    labels=np.array(labels), kickoffs=np.array(kickoffs),
                    crowds=np.array(crowds, float), types=np.array(types),
                    prank=prank, node_id=node_id, plat=plat, plon=plon,
                    snap_median=pct[50], snap_p90=pct[90], within500=within500)
print("saved results/surfaces.npz")
