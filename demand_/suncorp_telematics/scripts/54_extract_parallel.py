"""
54_extract_parallel.py
========================================================================
Parallel raw-event extraction: reads the 5,000 raw trip files DIRECTLY
(no 120 GB copy needed) and explodes them across all CPU cores, writing
one merged results/event_points_full.csv.gz.

~2.7 h single-threaded  ->  ~10-20 min on a multi-core box.

Usage:
    python scripts/54_extract_parallel.py --raw-dir "<path to raw_near_miss_2026>" [--workers N]
    python scripts/54_extract_parallel.py --raw-dir "<path>" --all-points --build-panel \
        [--panel-out results/segment_hour_panel.parquet] [--snap-m 30] [--workers N]

--all-points keeps every in-bbox point (plus VehicleID/TripID) instead of only
kinematic events; --build-panel then snaps those points to OSM segments,
derives posted limits, and aggregates to segment-hour rows with vehicle-weighted
mean speeds and strict (>) margin exceedance counts (5/10/15/20 km/h).
"""
import argparse
import glob
import gzip
import json
import os
import time
import zlib
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

BBOX = dict(lon_min=152.98, lon_max=153.05, lat_min=-27.49, lat_max=-27.44)
TH_BRAKE, TH_ACCEL, TH_CORNER = -0.30, 0.30, 0.35
TH_BRAKE_M, TH_ACCEL_M, TH_CORNER_M = -0.20, 0.20, 0.30
SPEED_EXTREME = 110.0
OUT_COLS = ["dt", "SnappedLongitude", "SnappedLatitude", "Speed", "XAcc", "YAcc",
            "brake", "accel", "corner", "extreme", "mod", "VehicleType"]
USECOLS = ["StartDate", "SnappedPath", "TimestampPath", "SpeedPath_kph",
           "XAccPath", "YAccPath", "VehicleType"]
DATE_MIN, DATE_MAX = "2025-01-01", "2026-03-31"


def parse_trip(sp, tp, spd, xa, ya, vtype):
    if not isinstance(sp, str) or not isinstance(tp, str):
        return None
    pts = sp.split(",")
    ts = tp.split(",")
    xs = xa.split(",") if isinstance(xa, str) else []
    ys = ya.split(",") if isinstance(ya, str) else []
    ss = spd.split(",") if isinstance(spd, str) else []
    n = min(len(pts), len(ts))
    out = []
    for i in range(n):
        ll = pts[i].split()
        if len(ll) != 2:
            continue
        try:
            lon = float(ll[0])
            lat = float(ll[1])
        except ValueError:
            continue
        if not (BBOX["lon_min"] <= lon <= BBOX["lon_max"] and
                BBOX["lat_min"] <= lat <= BBOX["lat_max"]):
            continue

        def g(arr):
            if i < len(arr) and arr[i] not in ("null", "", "None"):
                try:
                    return float(arr[i])
                except ValueError:
                    return np.nan
            return np.nan
        out.append((ts[i], lon, lat, g(ss), g(xs), g(ys), vtype))
    return out or None


_SHARD_BASE = None  # set in main via initializer
_ALL_POINTS = False


def _init(base, all_points=False):
    global _SHARD_BASE, _ALL_POINTS
    _SHARD_BASE = base
    _ALL_POINTS = all_points


def is_valid_gzip(path):
    try:
        with open(path, "rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                return False
        with gzip.open(path, "rb") as stream:
            while stream.read(1 << 20):
                pass
        return True
    except (gzip.BadGzipFile, EOFError, zlib.error):
        return False


def read_points(f):
    usecols = USECOLS + (["VehicleID", "TripID"] if _ALL_POINTS else [])
    string_columns = ["SnappedPath", "TimestampPath", "SpeedPath_kph",
                      "XAccPath", "YAccPath"]
    if _ALL_POINTS:
        string_columns += ["VehicleID", "TripID"]
    df = pd.read_csv(f, usecols=usecols, dtype=dict.fromkeys(string_columns, "string"))
    if _ALL_POINTS and df[["VehicleID", "TripID"]].isna().any().any():
        raise ValueError(f"Missing vehicle/trip identifiers in {os.path.basename(f)}")
    if not _ALL_POINTS:
        df = df[(df.StartDate >= DATE_MIN) & (df.StartDate <= DATE_MAX)]
    rows = []
    for t in df.itertuples(index=False):
        if _ALL_POINTS:
            paths = (t.SnappedPath, t.TimestampPath, t.SpeedPath_kph)
            if not all(isinstance(value, str) for value in paths):
                raise ValueError(f"Missing trajectory path in {os.path.basename(f)}")
            if len({len(value.split(',')) for value in paths}) != 1:
                raise ValueError(f"Misaligned trajectory paths in {os.path.basename(f)}")
        pr = parse_trip(t.SnappedPath, t.TimestampPath, t.SpeedPath_kph,
                        t.XAccPath, t.YAccPath, t.VehicleType)
        if pr:
            rows.extend([row + (t.VehicleID, t.TripID) for row in pr]
                        if _ALL_POINTS else pr)
    columns = ["dt", "SnappedLongitude", "SnappedLatitude",
               "Speed", "XAcc", "YAcc", "VehicleType"]
    if _ALL_POINTS:
        columns += ["VehicleID", "TripID"]
    p = pd.DataFrame(rows, columns=columns)
    xa = p["XAcc"].fillna(0.0)
    ya = p["YAcc"].fillna(0.0)
    sp = p["Speed"].fillna(0.0)
    p["brake"] = (xa <= TH_BRAKE).astype(int)
    p["accel"] = (xa >= TH_ACCEL).astype(int)
    p["corner"] = (ya.abs() >= TH_CORNER).astype(int)
    p["extreme"] = (sp >= SPEED_EXTREME).astype(int)
    p["mod"] = ((xa <= TH_BRAKE_M) | (xa >= TH_ACCEL_M) | (ya.abs() >= TH_CORNER_M)).astype(int)
    m = p[["brake", "accel", "corner", "extreme", "mod"]].sum(axis=1) > 0
    return p[OUT_COLS + ["VehicleID", "TripID"]] if _ALL_POINTS else p.loc[m, OUT_COLS]


def _record_error(f, error_type):
    path = f"{_SHARD_BASE}.pid{os.getpid()}.errors.jsonl"
    record = {"source": os.path.basename(f), "error_type": error_type}
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def process_file(f):
    if not is_valid_gzip(f):
        message = "not a valid gzip stream"
        print(f"  !! skipping corrupt file {os.path.basename(f)}: {message}", flush=True)
        _record_error(f, message)
        return -1
    try:
        ev = read_points(f)
    except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
        print(f"  !! skipping corrupt file {os.path.basename(f)}: {exc}", flush=True)
        _record_error(f, type(exc).__name__)
        return -1
    if ev.empty:
        return 0
    out_shard = f"{_SHARD_BASE}.pid{os.getpid()}.gz"
    with gzip.open(out_shard, "at", newline="") as fo:
        ev.to_csv(fo, header=False, index=False)
    return len(ev)


def summarize_segment_hours(points, margins=(5, 10, 15, 20)):
    margins = tuple(sorted(set(float(value) for value in margins)))
    if not margins or any(not np.isfinite(value) or value < 0 for value in margins):
        raise ValueError("Margins must be finite non-negative km/h values")
    keys = ["road_osmid", "hour"]
    work = points.copy()
    required = ["road_osmid", "dt", "VehicleID", "TripID", "Speed", "posted_limit_kmh"]
    missing = set(required) - set(work.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if work[["road_osmid", "VehicleID", "TripID"]].isna().any().any():
        raise ValueError("Segment, vehicle and trip identifiers must not be missing")
    work["dt"] = pd.to_datetime(work["dt"], format="mixed", errors="raise")
    if work["dt"].isna().any():
        raise ValueError("Timestamps must not be missing")
    if work["dt"].dt.tz is None:
        work["dt"] = work["dt"].dt.tz_localize("Australia/Brisbane")
    else:
        work["dt"] = work["dt"].dt.tz_convert("Australia/Brisbane")
    work["hour"] = work["dt"].dt.floor("h")
    for column in ["Speed", "posted_limit_kmh"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    identity = ["road_osmid", "VehicleID", "TripID", "dt"]
    duplicates = work.groupby(identity)[["Speed", "posted_limit_kmh"]].nunique(dropna=False)
    if (duplicates > 1).any().any():
        raise ValueError("Conflicting observations for the same trip timestamp")
    work = work.drop_duplicates(identity)
    if (work.groupby(keys)["posted_limit_kmh"].nunique() > 1).any():
        raise ValueError("Multiple posted limits within a segment-hour; split the segment")
    work["valid_speed"] = np.isfinite(work["Speed"]) & (work["Speed"] >= 0)
    work["valid_limit"] = np.isfinite(work["posted_limit_kmh"]) & (work["posted_limit_kmh"] > 0)
    work["usable_speed"] = work["Speed"].where(work["valid_speed"])
    work["exceedance"] = (work["Speed"] - work["posted_limit_kmh"]).where(
        work["valid_speed"] & work["valid_limit"])
    vehicle = work.groupby(keys + ["VehicleID"]).agg(
        mean_speed_kmh=("usable_speed", "mean"),
        valid_speed_points=("usable_speed", "count"))
    trip = work.groupby(keys + ["VehicleID", "TripID"]).agg(
        mean_speed_kmh=("usable_speed", "mean"),
        mean_exceedance_kmh=("exceedance", "mean"),
        max_exceedance_kmh=("exceedance", "max"),
        valid_speed_points=("usable_speed", "count"),
        judged_points=("exceedance", "count"))
    summary = vehicle.groupby(level=keys).agg(
        vehicle_count=("mean_speed_kmh", "size"),
        vehicles_with_speed=("mean_speed_kmh", "count"),
        mean_vehicle_speed_kmh=("mean_speed_kmh", "mean"))
    summary["trip_count"] = trip.groupby(level=keys).size()
    summary["trips_with_speed"] = trip["mean_speed_kmh"].notna().groupby(level=keys).sum()
    summary["trips_with_judged_speed"] = (
        trip["mean_exceedance_kmh"].notna().groupby(level=keys).sum())
    summary["trips_without_judged_speed"] = (
        summary["trip_count"] - summary["trips_with_judged_speed"])
    for margin in margins:
        label = f"{margin:g}".replace(".", "p")
        summary[f"trips_mean_gt_{label}"] = (
            (trip["mean_exceedance_kmh"] > margin).groupby(level=keys).sum())
        summary[f"trips_any_point_gt_{label}"] = (
            (trip["max_exceedance_kmh"] > margin).groupby(level=keys).sum())
    return summary.reset_index(), vehicle.reset_index(), trip.reset_index()


def build_panel(traffic_csv, panel_out, snap_m=30.0, default_limit=50.0):
    import geopandas as gpd

    project_root = Path(__file__).resolve().parents[1]
    geojson_path = project_root / "data" / "osm_roads_suncorp.geojson"
    roads = gpd.read_file(geojson_path).to_crs(28356)
    roads["osmid"] = pd.to_numeric(roads["osmid"], errors="coerce")
    roads = roads.dropna(subset=["osmid"])
    roads["road_osmid"] = roads["osmid"].astype("int64")
    project_root = Path(__file__).resolve().parents[1]
    feats_path = project_root / "data" / "road_geometry_features.csv"
    limit_map = pd.read_csv(feats_path).set_index("road_osmid")["maxspeed_kmh"]

    matched_parts = []
    for chunk in pd.read_csv(traffic_csv, chunksize=500_000,
                             dtype={"VehicleID": "string", "TripID": "string"},
                             parse_dates=["dt"]):
        pts = gpd.GeoDataFrame(
            chunk, geometry=gpd.points_from_xy(chunk.SnappedLongitude, chunk.SnappedLatitude),
            crs=4326).to_crs(28356)
        joined = gpd.sjoin_nearest(pts, roads[["road_osmid", "geometry"]],
                                   how="inner", max_distance=snap_m, distance_col="dist_m")
        ambiguous = joined.index.duplicated(keep=False)
        joined = joined.loc[~ambiguous]
        joined["posted_limit_kmh"] = (joined["road_osmid"].map(limit_map)
                                      .fillna(default_limit))
        matched_parts.append(joined[["road_osmid", "dt", "VehicleID", "TripID",
                                     "Speed", "posted_limit_kmh"]])
    if not matched_parts:
        raise SystemExit("no matched points")
    points = pd.concat(matched_parts, ignore_index=True)
    summary, _, _ = summarize_segment_hours(points)
    out = Path(panel_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out, index=False)
    audit = {
        "traffic_points": int(len(points)),
        "panel_rows": int(len(summary)),
        "segments": int(summary.road_osmid.nunique()),
        "hours": int(summary.hour.nunique()),
        "snap_m": snap_m,
        "default_limit": default_limit,
    }
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, default=str))
    return summary, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--all-points", action="store_true",
                    help="retain all study-area points and vehicle/trip IDs for traffic counts")
    ap.add_argument("--build-panel", action="store_true",
                    help="after extraction, snap points to segments and aggregate the "
                         "segment-hour panel (requires --all-points)")
    ap.add_argument("--panel-out", default="results/segment_hour_panel.parquet")
    ap.add_argument("--snap-m", type=float, default=30.0)
    ap.add_argument("--default-limit", type=float, default=50.0)
    args = ap.parse_args()
    if args.build_panel and not args.all_points:
        ap.error("--build-panel requires --all-points")
    if args.out is None:
        args.out = ("results/traffic_points_full.csv.gz" if args.all_points
                    else "results/event_points_full.csv.gz")
    if args.all_points and os.path.basename(args.out) == "event_points_full.csv.gz":
        ap.error("--all-points requires a separate traffic output, not event_points_full.csv.gz")

    marker = Path(args.out + ".complete.json")
    marker.unlink(missing_ok=True)
    if args.workers < 1 or not np.isfinite(args.snap_m) or args.snap_m <= 0:
        ap.error("workers and snap distance must be positive")
    files = sorted(glob.glob(os.path.join(args.raw_dir, "trips-*.csv.gz")))
    if not files:
        raise SystemExit(f"no trips-*.csv.gz in {args.raw_dir}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    W = args.workers
    print(f"[54] {len(files)} files across {W} workers")

    # clean any prior pid-shards
    for s in glob.glob(args.out + ".pid*.gz"):
        os.remove(s)

    for path in glob.glob(args.out + ".pid*.errors.jsonl"):
        os.remove(path)
    t0 = time.time()
    total = 0
    done = 0
    skipped = 0
    with Pool(W, initializer=_init, initargs=(args.out, args.all_points)) as pool:
        for n in pool.imap_unordered(process_file, files, chunksize=8):
            skipped += int(n < 0)
            total += max(n, 0)
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(files)} files | {total:,} events | "
                      f"{el:.0f}s | ETA {el/done*(len(files)-done):.0f}s", flush=True)
    print(f"[54] extracted {total:,} events in {time.time()-t0:.0f}s; merging...")

    if total == 0:
        raise SystemExit("No usable points; completion marker not written")
    shards = sorted(glob.glob(args.out + ".pid*.gz"))
    temporary_output = args.out + ".partial"
    merged_rows = 0
    with gzip.open(temporary_output, "wt", newline="") as fo:
        output_columns = OUT_COLS + (["VehicleID", "TripID"] if args.all_points else [])
        fo.write(",".join(output_columns) + "\n")
        for s in shards:
            with gzip.open(s, "rt") as fi:
                for line in fi:
                    fo.write(line)
                    merged_rows += 1
    if merged_rows != total:
        raise RuntimeError("Merged row count differs from extracted row count")
    os.replace(temporary_output, args.out)
    for s in shards:
        os.remove(s)
    print(f"[54] extracted -> {args.out} ({total:,} points from {len(shards)} shards)")

    if args.build_panel:
        print(f"[54] building segment-hour panel -> {args.panel_out}")
        _, audit = build_panel(args.out, args.panel_out,
                               snap_m=args.snap_m, default_limit=args.default_limit)
        print(f"[54] panel: {audit['panel_rows']:,} rows, "
              f"{audit['segments']:,} segments, {audit['hours']:,} hours")

    errors = []
    for path in sorted(glob.glob(args.out + ".pid*.errors.jsonl")):
        with open(path, encoding="utf-8") as stream:
            errors.extend(json.loads(line) for line in stream)
    if len(errors) != skipped:
        raise RuntimeError("Skipped-file audit count mismatch")
    status = {"status": "partial" if skipped else "complete",
              "input_files": len(files), "skipped_files": skipped,
              "point_rows": total, "errors": errors,
              "output": str(Path(args.out).resolve()),
              "panel_output": str(Path(args.panel_out).resolve()) if args.build_panel else None}
    temporary_marker = marker.with_suffix(".tmp")
    temporary_marker.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temporary_marker, marker)
    print(f"[54] {status['status']}: {marker}", flush=True)


if __name__ == "__main__":
    main()
