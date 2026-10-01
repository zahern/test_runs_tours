"""
52_build_surfaces.py
========================================================================
Build the per-fixture space-time risk surfaces from the REAL extracted
events (results/event_points_full.csv.gz) and the fixture calendar
(data/fixture_calendar.csv).

Risk = hard-braking events (brake==1) assigned to their nearest candidate
post (Voronoi tessellation of the study area), binned into the 28 x 15-min
buckets of each fixture's 7-hour window (kick-off -3h .. +4h).

Outputs results/surfaces.npz consumed by the deployment analyses
(officer tokens, K choice, phased posts, weight sensitivity, per-game).
Also an event cache results/events_cache.npz for the K/weight analyses.
"""
import argparse, os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree

BBOX = dict(lon_min=152.98, lon_max=153.05, lat_min=-27.49, lat_max=-27.44)
ORDINARY_LABEL = "NonEvent baseline"


def haversine_m(a1, o1, a2, o2):
    R = 6371000.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = p2 - p1, np.radians(o2 - o1)
    x = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="results/event_points_full.csv.gz")
    ap.add_argument("--calendar", default="data/fixture_calendar.csv")
    ap.add_argument("--posts", default="results/post_selection_mip.csv")
    ap.add_argument("--out", default="results/surfaces.npz")
    ap.add_argument("--cache", default="results/events_cache.npz")
    ap.add_argument("--ordinary", default=None,
                    help="ISO datetime of an ordinary (non-event) contrast window")
    args = ap.parse_args()

    posts = pd.read_csv(args.posts)
    plat, plon = posts.lat.to_numpy(), posts.lon.to_numpy()
    prank = posts["rank"].to_numpy(); node_id = posts["node_id"].to_numpy()
    J = len(plat)
    tree = cKDTree(np.column_stack([plat, plon]))

    cal = pd.read_csv(args.calendar)
    labels = list(cal["label"]); kickoffs = list(cal["kickoff_iso"])
    crowds = [float(c) if str(c) != "" and not pd.isna(c) else np.nan for c in cal["attendance"]]
    types = list(cal["fixture_type"])
    ordinary = args.ordinary
    ko_ts = [pd.Timestamp(k) for k in kickoffs]
    if ordinary:
        ko_ts_ord = pd.Timestamp(ordinary)

    ev = pd.read_csv(args.events, usecols=["dt", "SnappedLongitude", "SnappedLatitude", "brake"])
    ev = ev[ev.brake == 1].copy()
    ev["dt"] = pd.to_datetime(ev["dt"], errors="coerce")
    ev = ev.dropna(subset=["dt"])
    ev = ev[(ev.SnappedLongitude >= BBOX["lon_min"]) & (ev.SnappedLongitude <= BBOX["lon_max"]) &
            (ev.SnappedLatitude >= BBOX["lat_min"]) & (ev.SnappedLatitude <= BBOX["lat_max"])]
    coords = np.column_stack([ev.SnappedLatitude.to_numpy(float), ev.SnappedLongitude.to_numpy(float)])
    _, idx = tree.query(coords)
    dm = haversine_m(coords[:, 0], coords[:, 1], plat[idx], plon[idx])
    ev = ev.assign(post=idx, snap_m=dm)
    print(f"[52] {len(ev)} brake events; snap median {np.median(dm):.0f} m")

    def surface(kt):
        start = kt - pd.Timedelta(hours=3)
        m = (ev.dt >= start) & (ev.dt < start + pd.Timedelta(hours=7))
        sub = ev[m]
        b = ((sub.dt - start) / pd.Timedelta(minutes=15)).astype(int).clip(0, 27).to_numpy()
        acc = np.zeros((J, 28))
        np.add.at(acc, (sub.post.to_numpy(), b), 1.0)
        return acc

    acc_fx = np.stack([surface(kt) for kt in ko_ts]) if ko_ts else np.zeros((0, J, 28))
    r_event = acc_fx.mean(axis=0) if len(acc_fx) else np.zeros((J, 28))
    r_ord = surface(ko_ts_ord) if ordinary else np.zeros((J, 28))

    per = acc_fx.sum(axis=(1, 2))
    print(f"[52] {len(acc_fx)} fixtures; events/fixture min {per.min():.0f} "
          f"med {np.median(per):.0f} max {per.max():.0f}")
    tot = r_event.sum(axis=1); top5 = np.argsort(-tot)[:5]
    print(f"[52] mean-surface top-5 static coverage = "
          f"{r_event[top5].sum()/max(r_event.sum(),1):.4f}; "
          f"posts {[f'P{int(prank[j])}' for j in top5]}")

    np.savez_compressed(args.out, acc_fx=acc_fx, r_event=r_event, r_ord=r_ord,
                        labels=np.array(labels), kickoffs=np.array(kickoffs),
                        crowds=np.array(crowds, float), types=np.array(types),
                        prank=prank, node_id=node_id, plat=plat, plon=plon,
                        snap_median=float(np.median(dm)),
                        within500=float((dm <= 500).mean()))
    print(f"[52] saved {args.out}")

    # event cache for K/weight analyses: in-window events with per-fixture bucket
    fx_list, bk_list, la_list, lo_list = [], [], [], []
    all_ko = ko_ts + ([ko_ts_ord] if ordinary else [])
    all_lab = labels + ([ORDINARY_LABEL] if ordinary else [])
    for f, kt in enumerate(all_ko):
        start = kt - pd.Timedelta(hours=3)
        m = (ev.dt >= start) & (ev.dt < start + pd.Timedelta(hours=7))
        sub = ev[m]
        b = ((sub.dt - start) / pd.Timedelta(minutes=15)).astype(int)
        inwin = (b >= 0) & (b < 28)
        fx_list.append(np.full(len(sub), f, np.int16))
        bk_list.append(np.where(inwin, b, -1).astype(np.int16))
        la_list.append(sub.SnappedLatitude.to_numpy(float))
        lo_list.append(sub.SnappedLongitude.to_numpy(float))
    if fx_list:
        np.savez_compressed(args.cache,
                            fx=np.concatenate(fx_list), bucket=np.concatenate(bk_list),
                            lat=np.concatenate(la_list), lon=np.concatenate(lo_list),
                            labels=np.array(all_lab),
                            is_ordinary=np.array([l == ORDINARY_LABEL for l in all_lab]))
        print(f"[52] saved {args.cache}")


if __name__ == "__main__":
    main()
