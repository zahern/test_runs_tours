"""
58b_build_braking_panel.py - segment x 15-minute HARD-BRAKING counts.

Companion to 58_build_15min_panel.py (the speeding panel). Produces the braking
margin the copula model needs on the SAME key as the speeding panel:

  road_osmid, bin (15-min, Australia/Brisbane)

    braking_events      hard-braking events in the bin
    braking_high_conf    events with Confidence != 'Low'
    braking_severe       events with PeakDecel_g >= --severe-g (default 0.50)
    braking_severity     sum of PeakDecel_g over the bin
    braking_vehicles     distinct VehicleID braking in the bin

Source: data/hard_braking_events_full.csv when present (the 55_rebuild_hb_full.py
rebuild over raw 2025-01-01..2026-03-31), else the legacy provider file
data/hard_braking_events.csv. The legacy file claims EventStart dates out to
Dec 2026 although raw trips end 2026-03-31 (18_prepare_eb_dataset.py), so rows
outside --date-min/--date-max are dropped rather than trusted.

Snapping mirrors 58 exactly (gpd.sjoin_nearest to segment geometry, EPSG:28356,
--snap-m 30 m) so a braking bin and a speeding bin describe the same stretch of
road. Compass timestamps are UTC; the speeding panel stores Brisbane bins (58
finalize). We convert EventStart UTC -> Australia/Brisbane and floor to 15 min
so the two panels share bin labels; --validate-panel checks that overlap and the
diurnal alignment against the speeding panel.

Usage:
  python scripts/58b_build_braking_panel.py --build --limit-chunks 5   # pilot
  python scripts/58b_build_braking_panel.py --build --validate-panel   # full
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RES = PROJ / "results"
FIG = PROJ / "figures"
GEOJSON = DATA / "osm_roads_suncorp.geojson"
_HB_FULL = DATA / "hard_braking_events_full.csv"
_HB_LEGACY = DATA / "hard_braking_events.csv"
PANEL = RES / "segment_15min_panel.parquet"
TZ = "Australia/Brisbane"
DEFAULT_OUT = RES / "segment_15min_panel_braking.parquet"
USECOLS = ["VehicleID", "EventStart", "PeakDecel_g", "Category", "Confidence",
           "SnappedLongitude", "SnappedLatitude"]
DATE_MIN, DATE_MAX = "2025-01-01", "2026-04-02"


def _resolve_events(explicit):
    if explicit:
        return Path(explicit)
    return _HB_FULL if _HB_FULL.exists() else _HB_LEGACY


def _load_roads():
    """Road segments in EPSG:28356 keyed by road_osmid - identical to 58."""
    import geopandas as gpd

    roads = gpd.read_file(GEOJSON).to_crs(28356)
    roads["osmid"] = pd.to_numeric(roads["osmid"], errors="coerce")
    roads = roads.dropna(subset=["osmid"])
    roads["road_osmid"] = roads["osmid"].astype("int64")
    return roads[["road_osmid", "geometry"]]


def _snap(chunk, roads, snap_m):
    """Snap a chunk of events to the nearest segment; drop tie matches.

    Same recipe as 58: project to EPSG:28356, sjoin_nearest within snap_m,
    then drop any point whose index matched more than one road (ambiguous).
    """
    import geopandas as gpd

    chunk = chunk.dropna(subset=["SnappedLongitude", "SnappedLatitude"])
    if chunk.empty:
        return chunk
    pts = gpd.GeoDataFrame(
        chunk, geometry=gpd.points_from_xy(chunk["SnappedLongitude"],
                                           chunk["SnappedLatitude"]),
        crs=4326).to_crs(28356)
    joined = gpd.sjoin_nearest(pts, roads, how="inner",
                               max_distance=float(snap_m), distance_col="dist_m")
    return joined.loc[~joined.index.duplicated(keep=False)]


def run_build(events_file, out_path, snap_m=30.0, date_min=DATE_MIN,
              date_max=DATE_MAX, severe_g=0.50, chunk_size=400_000,
              limit_chunks=0):
    """Build the braking panel from events_file.

    Roads are snapped per chunk (the spatial join is the memory driver) but the
    snapped rows are accumulated and grouped once, so every published column
    -- including the distinct-vehicle count -- is exact, not a chunk sum.
    """
    roads = _load_roads()
    start = pd.Timestamp(date_min, tz=TZ)
    end = pd.Timestamp(date_max, tz=TZ) + pd.Timedelta(days=1)
    rows = []
    n_raw = n_inwin = n_matched = 0
    cat_counts, conf_counts = Counter(), Counter()
    for part, chunk in enumerate(pd.read_csv(
            events_file, chunksize=chunk_size, usecols=USECOLS,
            dtype={"VehicleID": "string", "Category": "string",
                   "Confidence": "string"})):
        if limit_chunks and part >= limit_chunks:
            break
        n_raw += len(chunk)
        cat_counts.update(chunk["Category"].astype("string").fillna("NA"))
        conf_counts.update(chunk["Confidence"].astype("string").fillna("NA"))
        dt = pd.to_datetime(chunk["EventStart"], errors="coerce", utc=True)
        keep = dt.notna()
        if not keep.any():
            continue
        chunk, dt = chunk.loc[keep].copy(), dt[keep]
        dt_b = dt.dt.tz_convert(TZ)
        inwin = (dt_b >= start) & (dt_b < end)
        chunk, dt_b = chunk.loc[inwin].copy(), dt_b[inwin]
        if chunk.empty:
            continue
        n_inwin += len(chunk)
        chunk["bin"] = dt_b.dt.floor("15min")
        joined = _snap(chunk, roads, snap_m)
        if joined.empty:
            continue
        n_matched += len(joined)
        joined = joined.drop_duplicates(["VehicleID", "EventStart", "road_osmid"])
        # HIGH-CONF and SEVERE are computed before PeakDecel is coerced to float.
        joined["_hi"] = (joined["Confidence"].astype("string").str.lower()
                         .fillna("na").ne("low").astype("int64"))
        joined["_sev"] = (pd.to_numeric(joined["PeakDecel_g"], errors="coerce")
                          >= float(severe_g)).astype("int64")
        joined["PeakDecel_g"] = pd.to_numeric(joined["PeakDecel_g"],
                                              errors="coerce").fillna(0.0)
        rows.append(joined[["road_osmid", "bin", "VehicleID", "_hi", "_sev",
                            "PeakDecel_g"]])
        if (part + 1) % 10 == 0:
            print(f"   [build] chunk {part + 1}: {n_matched:,} snapped, "
                  f"{n_inwin:,} in window", flush=True)
    if not rows:
        raise SystemExit("no braking events matched the window/roads")
    snapped = pd.concat(rows, ignore_index=True)
    panel = snapped.groupby(["road_osmid", "bin"]).agg(
        braking_events=("VehicleID", "size"),
        braking_vehicles=("VehicleID", "nunique"),
        braking_high_conf=("_hi", "sum"),
        braking_severe=("_sev", "sum"),
        braking_severity=("PeakDecel_g", "sum")).reset_index()
    panel["road_osmid"] = panel["road_osmid"].astype("int64")
    for c in ("braking_events", "braking_vehicles", "braking_high_conf",
              "braking_severe"):
        panel[c] = panel[c].astype("int64")
    panel["braking_severity"] = panel["braking_severity"].astype(float)
    panel = panel.sort_values(["road_osmid", "bin"]).reset_index(drop=True)
    panel.to_parquet(out_path, index=False)
    monthly = (panel.assign(month=panel["bin"].dt.strftime("%Y-%m"))
               .groupby("month")["braking_events"].sum().to_dict())
    audit = {
        "events_file": str(events_file), "out": str(out_path),
        "rows": int(len(panel)),
        "segments": int(panel["road_osmid"].nunique()),
        "bins": int(panel["bin"].nunique()),
        "braking_events": int(panel["braking_events"].sum()),
        "braking_vehicles": int(panel["braking_vehicles"].sum()),
        "braking_high_conf": int(panel["braking_high_conf"].sum()),
        "braking_severe": int(panel["braking_severe"].sum()),
        "braking_severity": float(panel["braking_severity"].sum()),
        "events_raw": int(n_raw), "events_in_window": int(n_inwin),
        "events_matched": int(n_matched),
        "match_pct_of_window": round(100 * n_matched / max(n_inwin, 1), 2),
        "date_min": date_min, "date_max": date_max, "snap_m": float(snap_m),
        "severe_g": float(severe_g), "tz": TZ,
        "category_counts": dict(cat_counts), "confidence_counts": dict(conf_counts),
        "monthly_events": monthly,
    }
    return panel, audit


def run_validate(braking_path, panel_path=PANEL, fig_dir=None):
    """Check the braking panel lands on the speeding panel's key + diurnal.

    Reads only road_osmid/bin/vehicle_count from the 446 MB panel so this stays
    cheap. Keys are matched with a hash merge (never a Python set over the 64M
    panel rows, which would not fit). A correct tz/floor convention shows
    braking bins landing on panel bins and an hourly profile correlated with the
    panel's vehicle_count.
    """
    import pyarrow.parquet as pq

    br = pd.read_parquet(braking_path)
    br["_bin"] = pd.to_datetime(br["bin"], utc=True)
    br["road_osmid"] = br["road_osmid"].astype("int64")
    br = br.reset_index(drop=True)
    if not Path(panel_path).exists():
        return {"ok": False, "reason": f"panel missing: {panel_path}"}
    pan = pq.read_table(panel_path,
                        columns=["road_osmid", "bin", "vehicle_count"]).to_pandas()
    pan["_bin"] = pd.to_datetime(pan["bin"], utc=True)
    pan["road_osmid"] = pan["road_osmid"].astype("int64")
    pan_keys = pan[["road_osmid", "_bin"]].drop_duplicates().assign(_on=1)
    left = br[["road_osmid", "_bin"]].copy()
    left["_i"] = np.arange(len(left))
    merged = left.merge(pan_keys, on=["road_osmid", "_bin"],
                        how="left").sort_values("_i")
    on_panel = merged["_on"].notna().to_numpy()
    hourly_b = (br.assign(h=br["_bin"].dt.tz_convert(TZ).dt.hour)
                .groupby("h")["braking_events"].sum())
    hourly_p = (pan.assign(h=pan["_bin"].dt.tz_convert(TZ).dt.hour)
                .groupby("h")["vehicle_count"].sum())
    common = sorted(set(hourly_b.index) & set(hourly_p.index))
    r = float(hourly_b.reindex(common).corr(hourly_p.reindex(common))) \
        if len(common) > 2 else float("nan")
    result = {
        "ok": True, "panel": str(panel_path),
        "braking_segments": int(br["road_osmid"].nunique()),
        "panel_segments": int(pan["road_osmid"].nunique()),
        "segment_overlap": int(len(set(br["road_osmid"]) & set(pan["road_osmid"]))),
        "braking_rows": int(len(br)),
        "braking_rows_on_panel_key": int(on_panel.sum()),
        "braking_row_hit_pct": round(100 * float(on_panel.mean()), 2),
        "braking_events_on_panel_key": int(br.loc[on_panel, "braking_events"].sum()),
        "diurnal_pearson_r": (None if not np.isfinite(r) else round(r, 4)),
    }
    if fig_dir is not None and len(common) > 2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
            ax[0].plot(common, hourly_b.reindex(common).values, "o-",
                       label="hard-braking events")
            ax[0].set_ylabel("braking events")
            ax[0].set_title("Diurnal alignment vs speeding panel "
                            f"(r = {result['diurnal_pearson_r']})")
            ax[0].legend()
            ax[1].plot(common, hourly_p.reindex(common).values, "s-",
                       color="tab:orange", label="panel vehicle_count")
            ax[1].set_ylabel("vehicle_count")
            ax[1].set_xlabel("hour of day (Australia/Brisbane)")
            ax[1].legend()
            fig.tight_layout()
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(fig_dir) / "braking_panel_alignment.png", dpi=120)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - figure is advisory
            print(f"[58b] WARN: alignment figure skipped ({exc})")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--validate-panel", action="store_true")
    ap.add_argument("--events", default=None, help="hard-braking CSV "
                    "(default: full rebuild if present, else legacy)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--snap-m", type=float, default=30.0)
    ap.add_argument("--severe-g", type=float, default=0.50)
    ap.add_argument("--date-min", default=DATE_MIN)
    ap.add_argument("--date-max", default=DATE_MAX)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--fig-dir", default=str(FIG))
    args = ap.parse_args()
    if not (args.build or args.validate_panel):
        ap.error("choose --build and/or --validate-panel")

    if args.build:
        events = _resolve_events(args.events)
        if not events.exists():
            raise SystemExit(f"FATAL: braking source not found: {events}")
        print(f"[58b] source: {events}", flush=True)
        panel, audit = run_build(
            events, Path(args.out), snap_m=args.snap_m,
            date_min=args.date_min, date_max=args.date_max,
            severe_g=args.severe_g, limit_chunks=args.limit_chunks)
        apath = Path(args.out).with_name(Path(args.out).stem + ".audit.json")
        apath.write_text(json.dumps(audit, indent=2, default=str))
        print(f"[58b] wrote {args.out} ({len(panel):,} rows)")
        print(f"[58b] audit: {json.dumps(audit, default=str)}")

    if args.validate_panel:
        out = Path(args.out)
        if not out.exists():
            raise SystemExit(f"FATAL: {out} missing; run --build first")
        res = run_validate(out, Path(args.panel), fig_dir=args.fig_dir)
        print(f"[58b] validate: {json.dumps(res, default=str)}")
        if res.get("ok") and res.get("braking_row_hit_pct", 0.0) < 90.0:
            print("[58b] WARN: <90% of braking bins land on a panel bin - "
                  "check the tz/floor convention against 58")


if __name__ == "__main__":
    main()
