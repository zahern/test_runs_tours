"""
One pass over the 448 MB event file -> compact in-memory cache of every
study-area event on the 42 fixture dates (+ the ordinary control day):
saves (fixture_index, bucket-of-window, lat, lon) so that ANY later analysis
(different post sets, K sweeps, per-phase selection) can re-snap in memory with
no further disk scan.  A bucket value of -1 means the event fell outside that
fixture's 28-bucket window (kept for the total-study-area denominator).
"""
import json, numpy as np, pandas as pd

BBOX = dict(lon_min=152.98, lon_max=153.05, lat_min=-27.49, lat_max=-27.44)
ORDINARY = ("NonEvent Wed 18Mar", "2026-03-18T16:00")
cal = pd.read_csv("data/fixture_calendar.csv")
kickoffs = list(cal["kickoff_iso"]) + [ORDINARY[1]]
labels = list(cal["label"]) + [ORDINARY[0]]
ko_ts = [pd.Timestamp(k) for k in kickoffs]
date_of = [t.date() for t in ko_ts]
start_bucket = [(t - pd.Timedelta(hours=3)).hour * 4 for t in ko_ts]  # day-bucket of window start
need = set(date_of)

rows_fx, rows_b, rows_lat, rows_lon = [], [], [], []
for ch in pd.read_csv("data/hard_braking_events.csv",
                      usecols=["EventStart", "SnappedLatitude", "SnappedLongitude"],
                      parse_dates=["EventStart"], chunksize=500_000):
    ch = ch.rename(columns={"EventStart": "ts", "SnappedLatitude": "lat", "SnappedLongitude": "lon"})
    ch = ch[(ch.lon >= BBOX["lon_min"]) & (ch.lon <= BBOX["lon_max"]) &
            (ch.lat >= BBOX["lat_min"]) & (ch.lat <= BBOX["lat_max"])]
    d = ch.ts.dt.date
    ch = ch[d.isin(need)]
    if ch.empty:
        continue
    ch = ch[np.isfinite(ch.lat) & np.isfinite(ch.lon)]
    dd = ch.ts.dt.date.to_numpy()
    daybucket = (ch.ts.dt.hour*4 + ch.ts.dt.minute//15).to_numpy()
    lat = ch.lat.to_numpy(float); lon = ch.lon.to_numpy(float)
    for f, (dtf, sb) in enumerate(zip(date_of, start_bucket)):
        m = dd == dtf
        if not m.any():
            continue
        b = daybucket[m] - sb  # bucket within this fixture's window
        inwin = (b >= 0) & (b < 28)
        bb = np.where(inwin, b, -1)
        rows_fx.append(np.full(m.sum(), f, np.int16))
        rows_b.append(bb.astype(np.int16))
        rows_lat.append(lat[m]); rows_lon.append(lon[m])

fx = np.concatenate(rows_fx); bk = np.concatenate(rows_b)
la = np.concatenate(rows_lat); lo = np.concatenate(rows_lon)
np.savez_compressed("results/events_cache.npz",
                    fx=fx, bucket=bk, lat=la, lon=lo,
                    labels=np.array(labels), kickoffs=np.array(kickoffs),
                    is_ordinary=np.array([l == ORDINARY[0] for l in labels]))
print(f"cached {len(fx)} study-area events across {len(labels)} windows "
      f"({(bk>=0).sum()} inside 28-bucket windows)")
