"""
58_build_15min_panel.py - segment x 15-minute speeding counts from Compass points.

Consumes the all-points extraction (results/traffic_points_full.csv.gz; produced
by 54_extract_parallel.py --all-points) and produces the panel the police paper
needs: per segment per 15-minute bin

  vehicle_count            distinct Compass vehicles in the bin
  trip_count               distinct (vehicle, trip) in the bin
  mean_vehicle_speed_kmh   equal-vehicle mean of per-trip mean speeds
  trips_mean_gt_{m}        trips whose trip-mean exceedance > m km/h
                           (strict >, m in 5/10/15/20)
  trips_any_point_gt_{m}   trips with any point exceedance > m

Exceedance = GPS speed - posted limit (OSM maxspeed, fallback --default-limit).
Compass is a ~5% sample of activity; counts are sample counts, penetration
scaling happens downstream (see 59/60).

Design: ONE streaming pass over the traffic file. Each chunk is snapped to the
nearest OSM segment (30 m) and reduced to per (segment, 15-min bin, vehicle,
trip) partial sums; partials are partitioned by month. A second, cheap step
finalizes each month into results/segment_15min_panel_YYYY-MM.parquet. The pass
is resumable: it skips when _15min_tmp/_pass_done.json exists, and finalized
months are skipped individually.

Usage:
  python scripts/58_build_15min_panel.py --pass --limit-chunks 40   # pilot
  python scripts/58_build_15min_panel.py --all                      # full build
  python scripts/58_build_15min_panel.py --merge                    # merge + plots
  python scripts/58_build_15min_panel.py --patch-target             # add
      # speed_limit + speeding_trips to an existing panel in place
"""
import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RES = PROJ / "results"
FIG = PROJ / "figures"
TRAFFIC = RES / "traffic_points_full.csv.gz"
GEOJSON = DATA / "osm_roads_suncorp.geojson"
FEATS = DATA / "road_geometry_features.csv"
TZ = "Australia/Brisbane"
MARGINS = (5, 10, 15, 20)
TMP_ROOT = RES / "_15min_tmp"
DONE_MARKER = TMP_ROOT / "_pass_done.json"
ROAD_META = DATA / "road_meta.csv"
# Posted limits at or above this are treated as credible. Below it the field is
# not a real road speed: living/service streets in this extract carry posted
# limits of 5, 10 and 20 km/h, so a segment-average of ~20 km/h on a laneway
# is trivially "over" its 10 km/h limit and 63% of those segment-bins qualify.
# Limits <= 20 carry 31.3% of the unfiltered target, and the single largest
# target segment was "Fish Lane", a living_street posted at 10 km/h. Restricting
# to >= 40 keeps 98.7% of segment-bins and 68.7% of the target while dropping
# the over-limit bin frequency to a plausible 2.9%.
MIN_TRUSTED_LIMIT = 40


def load_speed_limits():
    """Posted limit per road_osmid, as float (NaN where unknown).

    Read from road_meta.csv rather than the geojson so the limit used here is
    the same one the target was validated against. Returns an empty Series if
    the file is absent, which makes the speeding target come out all-zero
    rather than silently wrong.
    """
    if not ROAD_META.exists():
        print(f"[58] WARN: {ROAD_META} missing; speeding_trips will be 0")
        return pd.Series(dtype=float)
    md = pd.read_csv(ROAD_META, usecols=["osmid", "speed_limit"])
    md["osmid"] = pd.to_numeric(md["osmid"], errors="coerce")
    md["speed_limit"] = pd.to_numeric(md["speed_limit"], errors="coerce")
    md = md.dropna(subset=["osmid"]).drop_duplicates("osmid")
    return pd.Series(md["speed_limit"].to_numpy(),
                     index=md["osmid"].astype("int64").to_numpy())


def add_speeding_target(summary, limits, min_limit=MIN_TRUSTED_LIMIT):
    """Attach speed_limit and the speeding_trips target.

        speeding_trips = vehicle_count  if mean speed > posted limit
                                           and posted limit >= min_limit
                         0               otherwise

    WHY THIS AND NOT trips_mean_gt_10. That column, which the model was fitted
    on, counts trips whose mean speed is BELOW 10 km/h -- a congestion measure
    that was reported as "speeding" throughout. It falls with traffic volume
    (a binomial MLE over 350,946 trials put the flow elasticity at -0.954),
    so a flow-projection offset pushed prediction onto the busiest arterials,
    which structurally never register slow trips. The result was a negative
    correlation between predicted and observed on all six holdout games
    (r = -0.06 to -0.14). This target rises with speed instead, and a flow
    offset is at least directionally sensible for it.
    """
    if len(limits) == 0:
        summary["speed_limit"] = np.nan
        summary["speeding_trips"] = 0
        return summary
    summary = summary.merge(
        limits.rename("speed_limit").reset_index()
        .rename(columns={"index": "road_osmid"}),
        on="road_osmid", how="left")
    _sp = pd.to_numeric(summary["mean_vehicle_speed_kmh"], errors="coerce")
    _lim = pd.to_numeric(summary["speed_limit"], errors="coerce")
    trusted = _lim >= float(min_limit)
    summary["speeding_trips"] = np.where(
        trusted & (_sp > _lim),
        pd.to_numeric(summary["vehicle_count"], errors="coerce"), 0)
    summary["speeding_trips"] = pd.to_numeric(
        summary["speeding_trips"], errors="coerce").fillna(0).astype("int64")
    return summary


def patch_target(path, limits, min_limit=MIN_TRUSTED_LIMIT,
                 batch_size=1_000_000, force=False):
    """Add speed_limit + speeding_trips to an existing panel parquet in place.

    WHY THIS EXISTS. finalize_month() derives speeding_trips from quantities
    the panel ALREADY carries: mean_vehicle_speed_kmh and vehicle_count, plus
    the posted limit from road_meta.csv. A panel built before the target
    existed therefore does not need the 20 h re-stream of the 7.3 GB traffic
    extract -- the same derivation can be applied to the rows on disk. This
    is the cheap, auditable path; the streaming rebuild remains the
    authoritative one if the underlying pass ever changes.

    Streamed row-group by row-group so a 64M-row merged panel never has to
    be resident. Idempotent: a file that already carries both columns is
    skipped unless force=True. Written to a sibling .patchtmp then swapped,
    so a kill mid-write cannot truncate the original.

    Returns an audit dict with the totals and the two invariants the rebuild
    also asserts: target <= vehicle_count, and no target on a segment whose
    posted limit is below the trusted floor.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    need = ("road_osmid", "mean_vehicle_speed_kmh", "vehicle_count")
    missing = [c for c in need if c not in names]
    if missing:
        return {"path": str(path), "ok": False,
                "reason": f"missing source columns {missing}"}
    if "speeding_trips" in names and "speed_limit" in names and not force:
        return {"path": str(path), "ok": True, "skipped": "columns present"}
    tmp = path.with_suffix(path.suffix + ".patchtmp")
    writer = None
    rows = total = over = untrusted_hit = 0
    try:
        for batch in pf.iter_batches(batch_size=int(batch_size)):
            tbl = pa.Table.from_batches([batch])
            oid = pd.to_numeric(tbl.column("road_osmid").to_pandas(),
                                errors="coerce")
            sp = pd.to_numeric(
                tbl.column("mean_vehicle_speed_kmh").to_pandas(),
                errors="coerce")
            vc = pd.to_numeric(tbl.column("vehicle_count").to_pandas(),
                               errors="coerce")
            lim = pd.to_numeric(oid.map(limits), errors="coerce")
            trusted = lim >= float(min_limit)
            st = (vc.where(trusted & (sp > lim), 0)
                  .fillna(0).astype("int64"))
            arrays = [tbl.column(c) for c in names]
            arrays.append(pa.array(lim.to_numpy(dtype=float)))
            arrays.append(pa.array(st.to_numpy()))
            out = pa.Table.from_arrays(
                arrays, names=names + ["speed_limit", "speeding_trips"])
            if writer is None:
                writer = pq.ParquetWriter(tmp, out.schema,
                                          compression="snappy")
            writer.write_table(out)
            rows += len(st)
            total += int(st.sum())
            over += int((st > vc.fillna(0)).sum())
            untrusted_hit += int(((~trusted) & (st > 0)).sum())
        if writer is not None:
            writer.close()
            writer = None
        # Windows: an open ParquetFile handle blocks os.replace with
        # PermissionError (WinError 5). Drop the reader (and collect) before
        # swapping, or the patch can never land on the destination.
        del pf
        gc.collect()
        tmp.replace(path)
    except Exception:
        if writer is not None:
            writer.close()
        try:
            del pf
        except UnboundLocalError:
            pass
        tmp.unlink(missing_ok=True)
        raise
    return {"path": str(path), "ok": True, "rows": int(rows),
            "speeding_trips": int(total), "over_vehicle_count": int(over),
            "untrusted_limit_hits": int(untrusted_hit),
            "min_trusted_limit": float(min_limit)}


def _load_roads():
    import geopandas as gpd

    roads = gpd.read_file(GEOJSON).to_crs(28356)
    roads["osmid"] = pd.to_numeric(roads["osmid"], errors="coerce")
    roads = roads.dropna(subset=["osmid"])
    roads["road_osmid"] = roads["osmid"].astype("int64")
    feats = pd.read_csv(FEATS)
    feats["road_osmid"] = pd.to_numeric(feats["road_osmid"], errors="coerce")
    feats = feats.dropna(subset=["road_osmid"])
    limit_map = feats.groupby("road_osmid")["maxspeed_kmh"].first()
    return roads, limit_map


def run_pass(traffic_file, tmp_root, snap_m=30.0, default_limit=50.0,
             chunk_size=500_000, limit_chunks=0):
    import geopandas as gpd

    roads, limit_map = _load_roads()
    n_points = n_matched = n_partials = 0
    month_dirs = {}
    for part, chunk in enumerate(pd.read_csv(
            traffic_file, chunksize=chunk_size,
            usecols=["dt", "SnappedLongitude", "SnappedLatitude", "Speed",
                     "VehicleID", "TripID"],
            dtype={"VehicleID": "string", "TripID": "string"})):
        if limit_chunks and part >= limit_chunks:
            break
        dt = pd.to_datetime(chunk["dt"], format="mixed", errors="coerce")
        chunk = chunk.loc[dt.notna()].copy()
        dt = dt[dt.notna()]
        chunk["dt"] = dt
        chunk["month"] = (dt.dt.tz_localize("UTC").dt.tz_convert(TZ)
                          .dt.strftime("%Y-%m"))
        n_points += len(chunk)
        chunk = chunk.dropna(subset=["SnappedLongitude", "SnappedLatitude"])
        pts = gpd.GeoDataFrame(
            chunk, geometry=gpd.points_from_xy(chunk["SnappedLongitude"],
                                               chunk["SnappedLatitude"]),
            crs=4326).to_crs(28356)
        joined = gpd.sjoin_nearest(pts, roads[["road_osmid", "geometry"]],
                                   how="inner", max_distance=snap_m,
                                   distance_col="dist_m")
        joined = joined.loc[~joined.index.duplicated(keep=False)]
        if joined.empty:
            continue
        joined["posted_limit_kmh"] = (joined["road_osmid"].map(limit_map)
                                      .fillna(default_limit))
        joined["Speed"] = pd.to_numeric(joined["Speed"], errors="coerce")
        joined["bin"] = pd.to_datetime(joined["dt"]).dt.floor("15min")
        joined = joined.dropna(subset=["bin"])
        valid = np.isfinite(joined["Speed"]) & (joined["Speed"] >= 0)
        limit_ok = np.isfinite(joined["posted_limit_kmh"]) & (joined["posted_limit_kmh"] > 0)
        joined["usable_speed"] = joined["Speed"].where(valid)
        joined["exceedance"] = (joined["Speed"] - joined["posted_limit_kmh"]).where(
            valid & limit_ok)
        joined = joined.drop_duplicates(["road_osmid", "VehicleID", "TripID", "dt"])
        trip = joined.groupby(["month", "road_osmid", "bin", "VehicleID", "TripID"]).agg(
            speed_sum=("usable_speed", "sum"), speed_n=("usable_speed", "count"),
            exceed_sum=("exceedance", "sum"), exceed_n=("exceedance", "count"),
            exceed_max=("exceedance", "max")).reset_index()
        n_matched += len(joined)
        for month, group in trip.groupby("month"):
            month_dir = month_dirs.get(month)
            if month_dir is None:
                month_dir = tmp_root / month
                month_dir.mkdir(parents=True, exist_ok=True)
                month_dirs[month] = month_dir
            group.drop(columns=["month"]).to_parquet(
                month_dir / f"trips_{n_partials:06d}.parquet", index=False)
            n_partials += 1
        if (part + 1) % 20 == 0:
            print(f"   [pass] chunk {part + 1}: {n_matched:,} matched points, "
                  f"{n_partials} partials", flush=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / "_pass_done.json").write_text(json.dumps(
        {"chunks": part + 1, "points": int(n_points),
         "matched": int(n_matched), "partials": int(n_partials)}))
    return {"chunks": part + 1, "points": int(n_points),
            "matched": int(n_matched), "partials": int(n_partials)}


def finalize_month(month, tmp_root, out_path, limits=None,
                   min_limit=MIN_TRUSTED_LIMIT):
    month_dir = tmp_root / month
    parts = sorted(month_dir.glob("trips_*.parquet"))
    if not parts:
        return None
    trips = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    trips = trips.groupby(["road_osmid", "bin", "VehicleID", "TripID"]).agg(
        speed_sum=("speed_sum", "sum"), speed_n=("speed_n", "sum"),
        exceed_sum=("exceed_sum", "sum"), exceed_n=("exceed_n", "sum"),
        exceed_max=("exceed_max", "max")).reset_index()
    trips["trip_speed"] = trips["speed_sum"] / trips["speed_n"].clip(lower=1)
    trips["trip_exceed"] = (trips["exceed_sum"]
                            / trips["exceed_n"].clip(lower=1)).where(
        trips["exceed_n"] > 0)
    vehicle = trips.groupby(["road_osmid", "bin", "VehicleID"]).agg(
        veh_speed=("trip_speed", "mean"))
    summary = vehicle.groupby(level=["road_osmid", "bin"]).agg(
        vehicle_count=("veh_speed", "size"),
        mean_vehicle_speed_kmh=("veh_speed", "mean"))
    summary = summary.join(trips.groupby(["road_osmid", "bin"]).agg(
        trip_count=("TripID", "size"),
        trips_with_judged_speed=("trip_exceed", "count")))
    summary["trips_without_judged_speed"] = (summary["trip_count"]
                                             - summary["trips_with_judged_speed"])
    hits_mean = trips.assign(
        **{f"m{m}": (trips["trip_exceed"] > m).astype(float) for m in MARGINS})
    hits_any = trips.assign(
        **{f"a{m}": (trips["exceed_max"] > m).astype(float) for m in MARGINS})
    mean_sum = hits_mean.groupby(["road_osmid", "bin"])[
        [f"m{m}" for m in MARGINS]].sum()
    any_sum = hits_any.groupby(["road_osmid", "bin"])[
        [f"a{m}" for m in MARGINS]].sum()
    for margin in MARGINS:
        summary[f"trips_mean_gt_{margin}"] = mean_sum[f"m{margin}"]
        summary[f"trips_any_point_gt_{margin}"] = any_sum[f"a{margin}"]
    summary = summary.reset_index()
    # The actual speeding target the model is fitted on. Added before the int64
    # sweep so speed_limit can stay float (unknown limits must remain NaN, not
    # become a fake 0).
    summary = add_speeding_target(summary, limits, min_limit)
    summary["bin"] = pd.to_datetime(summary["bin"], utc=True).dt.tz_convert(TZ)
    for col in summary.columns:
        if col not in ("road_osmid", "bin", "mean_vehicle_speed_kmh",
                       "speed_limit"):
            summary[col] = summary[col].astype("int64")
    summary = summary.sort_values(["road_osmid", "bin"]).reset_index(drop=True)
    summary.to_parquet(out_path, index=False)
    shutil.rmtree(month_dir)
    audit = {"month": month, "trips": int(len(trips)), "rows": int(len(summary)),
             "segments": int(summary["road_osmid"].nunique()),
             "bins": int(summary["bin"].nunique()),
             "speeding_gt5": int(summary["trips_mean_gt_5"].sum()),
             "speeding_trips": int(summary["speeding_trips"].sum()),
             "speeding_bins": int((summary["speeding_trips"] > 0).sum()),
             "min_trusted_limit": float(min_limit),
             "segments_unlimited": int(summary["speed_limit"].isna().sum())}
    return summary, audit


def plot_summary(panel, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    frame = panel.copy()
    frame["dt"] = pd.to_datetime(frame["bin"])
    frame["dow"] = frame["dt"].dt.dayofweek
    frame["tod"] = frame["dt"].dt.hour + frame["dt"].dt.minute / 60.0
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    pivot = (frame.groupby(["dow", frame["tod"].round(0)])["trips_mean_gt_5"]
             .sum().unstack(fill_value=0))
    sns.heatmap(pivot, ax=axes[0, 0], cmap="viridis", cbar_kws={"label": "trips"})
    axes[0, 0].set_title("Speeding (>5 km/h over) by day-of-week x hour")
    axes[0, 0].set_xlabel("hour of day")
    axes[0, 0].set_ylabel("day of week")
    monthly = frame.groupby(frame["dt"].dt.strftime("%Y-%m")).agg(
        speeding=("trips_mean_gt_5", "sum"), vehicles=("vehicle_count", "sum"))
    monthly.plot(ax=axes[0, 1])
    axes[0, 1].set_title("Monthly totals (Compass sample)")
    axes[0, 1].set_xlabel("")
    top = (frame.groupby("road_osmid")["trips_mean_gt_5"].sum()
           .nlargest(20).sort_values())
    top.plot.barh(ax=axes[1, 0])
    axes[1, 0].set_title("Top-20 segments by observed speeding trips")
    axes[1, 0].set_xlabel("trips")
    axes[1, 1].hist(frame["mean_vehicle_speed_kmh"].dropna(), bins=60)
    axes[1, 1].set_title("Mean vehicle speed distribution")
    axes[1, 1].set_xlabel("km/h")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", default=str(TRAFFIC))
    ap.add_argument("--pass", dest="run_pass", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--patch-target", action="store_true",
                    help="add speed_limit + speeding_trips to the existing"
                         " monthly and merged panels in place, derived from"
                         " mean_vehicle_speed_kmh and road_meta.csv limits"
                         " exactly as finalize does. No re-stream; idempotent")
    ap.add_argument("--patch-force", action="store_true",
                    help="rewrite the two target columns even if present")
    ap.add_argument("--months", default=None, help="comma-separated YYYY-MM")
    ap.add_argument("--snap-m", type=float, default=30.0)
    ap.add_argument("--default-limit", type=float, default=50.0)
    ap.add_argument("--min-trusted-limit", type=float,
                    default=MIN_TRUSTED_LIMIT,
                    help="posted limits below this are not treated as real "
                         "road speeds; segments under it get speeding_trips=0 "
                         "(living/service streets here carry 5-20 km/h limits, "
                         "which makes a mean-speed comparison meaningless)")
    ap.add_argument("--limit-chunks", type=int, default=0, help="pilot: first N chunks")
    ap.add_argument("--out-dir", default=str(RES))
    ap.add_argument("--fig-dir", default=str(FIG))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    tmp_root = out_dir / "_15min_tmp"
    marker = tmp_root / "_pass_done.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.patch_target:
        limits = load_speed_limits()
        if len(limits) == 0:
            raise SystemExit(
                f"cannot derive speed_limit: {ROAD_META} missing or empty")
        print(f"[58] patch-target: {len(limits):,} posted limits "
              f"(trusted >= {MIN_TRUSTED_LIMIT:g} km/h)")
        parts = sorted(out_dir.glob("segment_15min_panel_????-??.parquet"))
        merged = out_dir / "segment_15min_panel.parquet"
        targets = parts + ([merged] if merged.exists() else [])
        if not targets:
            raise SystemExit("no panel parquet files to patch")
        audits = []
        for p in targets:
            a = patch_target(p, limits, force=args.patch_force)
            print(f"[58] {p.name}: {json.dumps(a, default=str)}")
            audits.append(a)
        patched = [a for a in audits
                   if a.get("ok") and not a.get("skipped")]
        problems = [a for a in patched
                    if a["over_vehicle_count"] or a["untrusted_limit_hits"]]
        if problems:
            raise SystemExit(
                "FATAL: patched target violates its invariants: "
                + json.dumps(problems, default=str))
        if patched and sum(a["speeding_trips"] for a in patched) <= 0:
            raise SystemExit("FATAL: the patched target is empty")
        ma = next((a for a in patched
                   if Path(a["path"]) == merged), None)
        if ma is not None:
            apath = out_dir / "segment_15min_panel.audit.json"
            try:
                audit = (json.loads(apath.read_text())
                         if apath.exists() else {})
                audit["speeding_trips"] = int(ma["speeding_trips"])
                audit["speed_limit_segments"] = int(len(limits))
                audit["target_patched"] = True
                apath.write_text(json.dumps(audit, indent=2, default=str))
            except Exception as exc:  # noqa: BLE001 - audit is advisory
                print(f"[58] WARN: could not update audit json ({exc})")
        print(f"[58] patch-target done: {len(patched)} file(s) patched, "
              f"{len(audits) - len(patched)} skipped")
        return

    if args.merge:
        parts = sorted(out_dir.glob("segment_15min_panel_????-??.parquet"))
        if not parts:
            raise SystemExit("no monthly parts to merge")
        panel = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        panel = panel.sort_values(["road_osmid", "bin"]).reset_index(drop=True)
        out = out_dir / "segment_15min_panel.parquet"
        panel.to_parquet(out, index=False)
        audit = {"rows": int(len(panel)),
                 "segments": int(panel["road_osmid"].nunique()),
                 "bins": int(panel["bin"].nunique()),
                 "months": [p.stem.split("_")[-1] for p in parts],
                 "speeding_gt5": int(panel["trips_mean_gt_5"].sum()),
                 "speeding_trips": int(panel["speeding_trips"].sum())
                 if "speeding_trips" in panel.columns else None,
                 "vehicles": int(panel["vehicle_count"].sum())}
        if "speeding_trips" not in panel.columns:
            print("[58] WARN: monthly parts predate the speeding_trips "
                  "target; re-run --finalize (delete the monthly parquets) "
                  "before fitting on speeding_trips")
        (out_dir / "segment_15min_panel.audit.json").write_text(
            json.dumps(audit, indent=2, default=str))
        plot_summary(panel, fig_dir / "segment_15min_panel_summary.png")
        print(f"[58] merged {len(parts)} months -> {out} ({len(panel):,} rows)")
        print(f"[58] audit: {json.dumps(audit, default=str)}")
        return

    if args.all and not marker.exists():
        args.run_pass = True
    if args.run_pass:
        if marker.exists() and not args.limit_chunks:
            print("[58] pass already complete; skipping (delete marker to rerun)")
        else:
            print("[58] streaming pass over traffic points...", flush=True)
            stats = run_pass(args.traffic, tmp_root, snap_m=args.snap_m,
                             default_limit=args.default_limit,
                             limit_chunks=args.limit_chunks)
            print(f"[58] pass done: {json.dumps(stats)}")

    do_finalize = args.finalize or args.all
    if do_finalize:
        if args.months:
            months = [m.strip() for m in args.months.split(",")]
        else:
            months = sorted(d.name for d in tmp_root.glob("????-??")
                            if d.is_dir())
        limits = load_speed_limits()
        print(f"[58] posted limits for {len(limits):,} segments; "
              f"speeding_trips requires limit >= "
              f"{args.min_trusted_limit:g} km/h")
        for month in months:
            out_path = out_dir / f"segment_15min_panel_{month}.parquet"
            if out_path.exists():
                print(f"[58] {month}: finalized, skip")
                continue
            print(f"[58] {month}: finalizing...", flush=True)
            result = finalize_month(month, tmp_root, out_path, limits=limits,
                                    min_limit=args.min_trusted_limit)
            if result is None:
                print(f"[58] {month}: no partials")
                continue
            _, audit = result
            (out_dir / f"segment_15min_panel_{month}.audit.json").write_text(
                json.dumps(audit, indent=2, default=str))
            print(f"[58] {month}: {audit['rows']:,} rows, "
                  f"{audit['segments']:,} segments, "
                  f"{audit['speeding_gt5']:,} trips>5km/h, "
                  f"{audit['speeding_trips']:,} SPEEDING trips")
        return
    if not args.run_pass:
        ap.error("choose --pass, --finalize, --all or --merge")


if __name__ == "__main__":
    main()
