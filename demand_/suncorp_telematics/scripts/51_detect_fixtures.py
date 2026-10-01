"""
51_detect_fixtures.py
========================================================================
Build the fixture calendar for the deployment from the extracted real
events (results/event_points_full.csv.gz).

A Suncorp fixture leaves a braking surge in a 7-hour evening/afternoon
window near the stadium.  We score every calendar day by the peak count
of brake events in a sliding 7-hour window inside the stadium's 1 km
catchment, and keep days whose peak clearly exceeds the rolling weekday
baseline -- this discovers fixtures across ALL codes (NRL Broncos &
Dolphins, Super Rugby Reds, A-League Roar), not just the Broncos games in
fixtures_all.csv.

Known crowds/opponents from data/fixtures_all.csv (Broncos) are merged in
by date; discovered days without a match are labelled by their peak hour
and left with crowd = NaN (to be filled if a schedule is available).

Output: data/fixture_calendar.csv
    columns: label, kickoff_iso, attendance, fixture_type
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STAD_LAT, STAD_LON = -27.4647, 153.0093
BBOX = dict(lon_min=152.98, lon_max=153.05, lat_min=-27.49, lat_max=-27.44)


def haversine_m(a1, o1, a2, o2):
    R = 6371000.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = p2 - p1, np.radians(o2 - o1)
    x = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="results/event_points_full.csv.gz")
    ap.add_argument("--fixtures", default=None,
                    help="Broncos fixtures_all.csv for crowd/opponent lookup")
    ap.add_argument("--out", default="data/fixture_calendar.csv")
    ap.add_argument("--catchment-m", type=float, default=1000.0)
    ap.add_argument("--min-surge", type=float, default=1.6,
                    help="keep days whose peak-window brake count exceeds this "
                         "multiple of the day-of-week baseline")
    ap.add_argument("--top-n", type=int, default=0,
                    help="if >0, keep the N highest-surge days instead of a threshold")
    ap.add_argument("--date-min", default="2025-01-01")
    ap.add_argument("--date-max", default="2026-03-31")
    ap.add_argument("--mode", choices=["known", "discover"], default="known",
                    help="'known' = use every real Suncorp home fixture in the "
                         "calendar (reliable); 'discover' = telemetry surge detection "
                         "(unreliable, needs the full event set)")
    args = ap.parse_args()

    # ---- reliable path: real fixtures from the calendar --------------------
    if args.mode == "known":
        if not (args.fixtures and os.path.exists(args.fixtures)):
            raise SystemExit("--mode known needs --fixtures data/fixtures_all.csv")
        fx = pd.read_csv(args.fixtures)
        fx = fx[(fx.venue == "Suncorp") & (fx.home_away.astype(str).str.startswith("H"))]
        fx = fx[(fx.date >= args.date_min) & (fx.date <= args.date_max)].copy()
        fx = fx.sort_values("date")
        rows = []
        for _, r in fx.iterrows():
            ko = pd.to_datetime(r.kickoff) if pd.notna(r.kickoff) else pd.Timestamp(r.date) + pd.Timedelta(hours=19)
            comp = str(r.comp)
            ftype = "SuperRugby" if "Super" in comp else ("ALeague" if "League" in comp else "NRL")
            rows.append(dict(label=f"{r.date} vs {r.opponent}",
                             kickoff_iso=ko.isoformat(),
                             attendance=("" if pd.isna(r.crowd) else int(r.crowd)),
                             fixture_type=ftype))
        out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"[51] mode=known: wrote {len(out)} real Suncorp home fixtures "
              f"({args.date_min}..{args.date_max}) -> {args.out}")
        return

    ev = pd.read_csv(args.events, usecols=["dt", "SnappedLongitude", "SnappedLatitude", "brake"])
    ev = ev[ev.brake == 1].copy()
    ev["dt"] = pd.to_datetime(ev["dt"], errors="coerce")
    ev = ev.dropna(subset=["dt"])
    d = haversine_m(ev.SnappedLatitude.to_numpy(), ev.SnappedLongitude.to_numpy(),
                    STAD_LAT, STAD_LON)
    ev = ev[d <= args.catchment_m]
    ev["date"] = ev.dt.dt.date
    ev["hour"] = ev.dt.dt.hour + ev.dt.dt.minute/60.0
    ev["dow"] = ev.dt.dt.dayofweek

    # peak 7-hour window brake count per day, and its kickoff = window centre-ish
    recs = []
    for dt, g in ev.groupby("date"):
        h = g.hour.to_numpy()
        best = 0; best_start = 16.0
        for s in np.arange(9.0, 18.0, 0.5):   # candidate window starts (kick-off-3h)
            c = int(((h >= s) & (h < s + 7)).sum())
            if c > best:
                best = c; best_start = s
        recs.append(dict(date=dt, dow=g.dow.iloc[0], peak=best, win_start=best_start))
    days = pd.DataFrame(recs)
    # baseline = median peak by day-of-week (ordinary traffic)
    base = days.groupby("dow").peak.median().rename("baseline")
    days = days.merge(base, on="dow")
    days["surge"] = days.peak / days.baseline.clip(lower=1)

    if args.top_n > 0:
        keep = days.sort_values("surge", ascending=False).head(args.top_n)
    else:
        keep = days[days.surge >= args.min_surge]
    keep = keep.sort_values("date")
    print(f"[51] scored {len(days)} days; {len(keep)} flagged as fixtures "
          f"(surge>={args.min_surge} or top-{args.top_n})")

    # merge known Broncos crowds/opponents by date
    known = {}
    if args.fixtures and os.path.exists(args.fixtures):
        fx = pd.read_csv(args.fixtures)
        fx = fx[(fx.venue == "Suncorp") & (fx.home_away.astype(str).str.startswith("H"))]
        for _, r in fx.iterrows():
            known[str(pd.to_datetime(r.date).date())] = (r.opponent, r.crowd,
                                                         "NRL" if "Round" in str(r.comp) or "Final" in str(r.comp) else "NRL")

    rows = []
    for i, r in enumerate(keep.itertuples(index=False), 1):
        ds = str(r.date)
        kickoff = pd.Timestamp(r.date) + pd.Timedelta(hours=float(r.win_start) + 3.0)
        opp, crowd, ftype = known.get(ds, (None, np.nan, "NRL"))
        label = f"{ds} vs {opp}" if opp else f"Fixture {ds}"
        rows.append(dict(label=label, kickoff_iso=kickoff.isoformat(),
                         attendance=("" if pd.isna(crowd) else int(crowd)),
                         fixture_type=ftype))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    n_known = sum(1 for r in rows if r["attendance"] != "")
    print(f"[51] wrote {len(out)} fixtures -> {args.out} ({n_known} with known crowd)")


if __name__ == "__main__":
    main()
