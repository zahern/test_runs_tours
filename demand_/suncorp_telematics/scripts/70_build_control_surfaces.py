#!/usr/bin/env python3
"""70_build_control_surfaces.py - observed surfaces for the event vs
non-event MIP comparison.

Builds three segment x rel_bin surfaces from the 15-minute panel
(results/segment_15min_panel_2026-*.parquet, target trips_mean_gt_10):

  event_obs   : the six held-out fixture windows, observed counts
  control_d1  : nearest clean same-weekday non-event evening, observed
  control_obs : mean of up to N clean same-weekday non-event evenings

Clean = same weekday, within +/-21 days, not any fixture date, not Easter
2026, and with a window row count at least half the candidate median (drops
partially-covered days). Both arms use identical extraction so the MIP sees
the same kind of surface; the shipped predicted surface is a third arm.

Surfaces carry the columns 60_game_day_mip.py needs: fixture, bin (timestamp),
road_osmid, rel_bin, pred_speeding (here the observed count / mean count).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TZ = "Australia/Brisbane"
REL_BINS = 28
HOLIDAYS = {"2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06"}


def kickoff(fixture: str) -> pd.Timestamp:
    m = re.search(r"@\s*(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2})",
                  str(fixture))
    if not m:
        raise ValueError(f"cannot parse kickoff from {fixture!r}")
    return pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}", tz=TZ)


def load_panel(months: list[str], target: str) -> pd.DataFrame:
    cols = ["road_osmid", "bin", target]
    frames = []
    for mon in months:
        p = ROOT / "results" / f"segment_15min_panel_{mon}.parquet"
        if not p.exists():
            raise SystemExit(f"missing panel {p}")
        frames.append(pd.read_parquet(p, columns=cols))
    df = pd.concat(frames, ignore_index=True)
    df["bin"] = pd.to_datetime(df["bin"], utc=True).dt.tz_convert(TZ)
    df["road_osmid"] = df["road_osmid"].astype("int64")
    df["y"] = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
    return df


def window(panel: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame | None:
    end = start + pd.Timedelta(minutes=15 * REL_BINS)
    sub = panel[(panel["bin"] >= start) & (panel["bin"] < end)]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["rel_bin"] = ((sub["bin"] - start)
                      / pd.Timedelta(minutes=15)).round().astype(int)
    out = (sub.groupby(["road_osmid", "rel_bin"], as_index=False)
           ["y"].sum())
    out.attrs["n_rows"] = int(len(sub))
    return out


def window_start(date: pd.Timestamp, clock: pd.Timestamp) -> pd.Timestamp:
    naive = pd.Timestamp(f"{date.date()} {clock.strftime('%H:%M')}")
    return naive.tz_localize(TZ) - pd.Timedelta(hours=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="2026-02,2026-03,2026-04")
    ap.add_argument("--pred", default=str(ROOT / "results"
                    / "game_day_pred_15min.parquet"))
    ap.add_argument("--target", default="trips_mean_gt_10")
    ap.add_argument("--calendar", default=str(ROOT / "data"
                    / "fixture_calendar.csv"))
    ap.add_argument("--obs-cells", default=str(ROOT / "results"
                    / "game_day_pred_vs_actual_cells.csv.gz"))
    ap.add_argument("--out-event", default=str(ROOT / "results"
                    / "obs_event_surface_gt10.parquet"))
    ap.add_argument("--out-d1", default=str(ROOT / "results"
                    / "obs_control_surface_d1_gt10.parquet"))
    ap.add_argument("--out-control", default=str(ROOT / "results"
                    / "obs_control_surface_gt10.parquet"))
    ap.add_argument("--n-controls", type=int, default=3)
    ap.add_argument("--window-days", type=int, default=21)
    args = ap.parse_args()

    pred_fx = (pd.read_parquet(args.pred, columns=["fixture"])["fixture"]
               .drop_duplicates().tolist())
    cal = pd.read_csv(args.calendar)
    fixture_dates = set(pd.to_datetime(cal["kickoff_iso"]).dt.date)
    panel = load_panel([m.strip() for m in args.months.split(",")],
                       args.target)
    obs_cells = None
    if Path(args.obs_cells).exists():
        oc = pd.read_csv(args.obs_cells)
        obs_cells = oc.groupby("fixture")["observed"].sum()

    frames = {"event": [], "d1": [], "control": []}
    rows = []
    for fx in pred_fx:
        ko = kickoff(fx)
        kdate = ko.tz_localize(None).normalize()
        clock = ko.tz_localize(None)
        start = window_start(kdate, clock)
        ev = window(panel, start)
        if ev is None:
            raise SystemExit(f"no event-window rows for {fx}")
        ev = ev.copy()
        ev["fixture"] = fx
        ev["bin"] = start + pd.to_timedelta(ev["rel_bin"] * 15, unit="min")
        ev["pred_speeding"] = ev["y"].astype(float)
        frames["event"].append(ev[["fixture", "bin", "road_osmid", "rel_bin",
                                   "pred_speeding"]])

        cands = []
        for k in range(args.window_days * 2 + 1):
            d = kdate + pd.Timedelta(days=k - args.window_days)
            if d.weekday() != kdate.weekday():
                continue
            if d.date() in fixture_dates or str(d.date()) in HOLIDAYS:
                continue
            w = window(panel, window_start(d, clock))
            if w is not None:
                cands.append((d, w, int(w.attrs.get("n_rows", 0))))
        if not cands:
            raise SystemExit(f"no control candidates for {fx}")
        med_rows = float(np.median([c[2] for c in cands]))
        cands = [c for c in cands if c[2] >= 0.5 * med_rows]
        cands.sort(key=lambda c: (abs((c[0] - kdate).days), c[0]))
        chosen = cands[:max(1, args.n_controls)]

        def mean_surface(sel):
            cat = pd.concat([c[1] for c in sel], ignore_index=True)
            agg = (cat.groupby(["road_osmid", "rel_bin"], as_index=False)
                   ["y"].sum())
            agg["y"] /= float(len(sel))
            return agg

        d1 = mean_surface(chosen[:1])
        cm = mean_surface(chosen)
        for tag, surf, label in (
                ("d1", d1, "CTRL1: " + fx),
                ("control", cm, "CTRL: " + fx)):
            s = surf.copy()
            s["fixture"] = label
            s["bin"] = start + pd.to_timedelta(s["rel_bin"] * 15, unit="min")
            s["pred_speeding"] = s["y"].astype(float)
            frames[tag].append(s[["fixture", "bin", "road_osmid", "rel_bin",
                                  "pred_speeding"]])
        rows.append({
            "fixture": fx,
            "event_total": float(ev["pred_speeding"].sum()),
            "event_total_from_obs_cells": (
                float(obs_cells.get(fx, np.nan)) if obs_cells is not None
                else None),
            "n_controls": len(chosen),
            "control_dates": "|".join(str(c[0].date()) for c in chosen),
            "control_total_mean": float(cm["y"].sum()),
            "control_total_d1": float(d1["y"].sum()),
        })

    for tag, path in (("event", args.out_event), ("d1", args.out_d1),
                      ("control", args.out_control)):
        out = pd.concat(frames[tag], ignore_index=True)
        out.to_parquet(path, index=False)
        print(f"wrote {path}: {len(out):,} rows, "
              f"{out['fixture'].nunique()} fixtures, "
              f"{out['road_osmid'].nunique()} segments, "
              f"mass {out['pred_speeding'].sum():.0f}")

    map_df = pd.DataFrame(rows)
    map_path = Path(args.out_control).with_name("control_dates_gt10.csv")
    map_df.to_csv(map_path, index=False)
    print(map_df.to_string(index=False))
    print(f"wrote {map_path}")
    summary = {"months": args.months, "target": args.target,
               "n_controls": args.n_controls,
               "window_days": args.window_days,
               "fixtures": rows}
    with open(Path(args.out_control).with_name("control_build.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
