#!/usr/bin/env python3
"""
Phase 0 — Re-aggregate the FULL-YEAR hard-braking telematics onto SEQ segments
==============================================================================
The paper's segment model (near_miss_hb_ecf.csv) was built from the *March-2026*
hard-braking extract (~2.5M events, 44 days). The full-year extract
(hard_braking_events_year.csv, ~5.1 GB, Jan-2025 onward, ~11x more events) is
available but unused. This script snaps every year event to its nearest SEQ road
segment and rebuilds the HB count columns on the FULL YEAR, keeping every
geometry / crash column of the existing modelling table intact.

It also emits a segment x month panel so the downstream search can do a genuine
TEMPORAL transfer test (train on early months, predict later months).

Inputs
  --events  hard_braking_events_year.csv   (cols incl. EventStart, Category,
            Snapped/Raw Longitude/Latitude, VehicleType)
  --gpkg    SEQ_Expected_Crashes_1000.gpkg  (layer 'roads', EPSG:28355,
            canonical segment_id + LineString geometry)
  --base    near_miss_hb_ecf.csv            (assembled modelling table whose
            geometry + ObsCrashes/ExpCrashes columns we inherit)

Outputs (--out-dir, default = paper dir)
  near_miss_hb_ecf_year.csv     drop-in replacement for near_miss_hb_ecf.csv,
                                HB count columns recomputed on the full year
  near_miss_hb_panel_year.csv   segment x year-month HB counts (temporal transfer)
  hb_year_provenance.csv        events by tier / month / vehicle type + match rate
  hb_year_provenance.tex        paper-ready provenance table

Usage
  python aggregate_hb_year.py                          # full run (HPC)
  python aggregate_hb_year.py --smoke 2000000          # local smoke test (2M rows)
  qsub aggregate_hb_year.pbs
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMAND_DIR = os.path.dirname(os.path.dirname(PAPER_DIR))  # .../demand_

DEF_EVENTS = os.path.join(PAPER_DIR, "hard_braking_events_year.csv")
DEF_GPKG_CANDIDATES = [
    os.path.join(DEMAND_DIR, "crash_info_zonal", "SEQ_Expected_Crashes_1000.gpkg"),
    r"Z:\test_runs_tours\crash_info_zonal\SEQ_Expected_Crashes_1000.gpkg",
    "/mnt/hpccs01/home/ahernz/test_runs_tours/crash_info_zonal/SEQ_Expected_Crashes_1000.gpkg",
]
DEF_BASE = os.path.join(PAPER_DIR, "near_miss_hb_ecf.csv")

# Category label -> short tier key
TIER = {
    "Hard-Braking Candidate": "candidate",
    "Critical-Likely":        "critical",
    "Extreme":                "extreme",
}
SEG_CRS = 28355   # GPKG projected CRS (metres)


def _first_existing(cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return cands[0]


def load_segments(gpkg):
    import geopandas as gpd
    print(f"  Reading segments: {gpkg}")
    seg = gpd.read_file(gpkg, layer="roads")
    if seg.crs is None or seg.crs.to_epsg() != SEG_CRS:
        seg = seg.to_crs(SEG_CRS)
    seg["segment_id"] = seg["segment_id"].astype(str).str.strip()
    seg = seg[["segment_id", "geometry"]].reset_index(drop=True)
    print(f"  Segments: {len(seg):,} | CRS: {seg.crs.to_epsg()}")
    return seg


def process_chunk(chunk, seg_gdf, max_dist):
    """Snap one chunk of events to nearest segment; return matched rows with
    segment_id, ym (year-month), tier, vehicletype."""
    import geopandas as gpd
    # coordinates: prefer Snapped (already on road network), fall back to Raw
    lon = pd.to_numeric(chunk.get("SnappedLongitude"), errors="coerce")
    lat = pd.to_numeric(chunk.get("SnappedLatitude"), errors="coerce")
    rlon = pd.to_numeric(chunk.get("RawLongitude"), errors="coerce")
    rlat = pd.to_numeric(chunk.get("RawLatitude"), errors="coerce")
    lon = lon.fillna(rlon)
    lat = lat.fillna(rlat)
    ok = lon.notna() & lat.notna()
    if not ok.any():
        return None
    sub = chunk[ok].copy()
    pts = gpd.GeoDataFrame(
        sub, geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs=4326
    ).to_crs(SEG_CRS)
    joined = gpd.sjoin_nearest(
        pts[["geometry"]], seg_gdf, how="left",
        max_distance=max_dist, distance_col="snap_dist",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    sub = sub.assign(segment_id=joined["segment_id"].values,
                     snap_dist=joined["snap_dist"].values)
    sub = sub[sub["segment_id"].notna()].copy()
    if len(sub) == 0:
        return None
    dt = pd.to_datetime(sub["EventStart"], errors="coerce")
    sub = sub[dt.notna()].copy()
    sub["ym"] = dt[dt.notna()].dt.strftime("%Y-%m").values
    sub["tier"] = sub["Category"].map(TIER).fillna("candidate")
    sub["vehicletype"] = sub.get("VehicleType", "NA").astype(str)
    return sub[["segment_id", "ym", "tier", "vehicletype"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=DEF_EVENTS)
    ap.add_argument("--gpkg", default=_first_existing(DEF_GPKG_CANDIDATES))
    ap.add_argument("--base", default=DEF_BASE)
    ap.add_argument("--out-dir", default=PAPER_DIR)
    ap.add_argument("--max-dist", type=float, default=500.0,
                    help="max snap distance (m); events beyond are dropped. "
                         "500 m matches the convention used to build the "
                         "original month-based segment table (~54-57%% match).")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, stop after roughly this many event rows")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("  Phase 0: full-year HB re-aggregation onto SEQ segments")
    print(f"  events : {args.events}")
    print(f"  gpkg   : {args.gpkg}")
    print(f"  base   : {args.base}")
    print(f"  max_dist={args.max_dist} m  chunksize={args.chunksize:,}"
          + (f"  SMOKE={args.smoke:,}" if args.smoke else ""))
    print("=" * 70)

    seg_gdf = load_segments(args.gpkg)

    usecols = ["EventStart", "Category", "VehicleType",
               "SnappedLongitude", "SnappedLatitude",
               "RawLongitude", "RawLatitude"]
    # accumulators: MultiIndex Series counts
    seg_tier = None      # (segment_id, tier) -> count
    seg_ym = None        # (segment_id, ym)   -> count
    veh_counts = pd.Series(dtype="int64")
    ym_counts = pd.Series(dtype="int64")
    tier_counts = pd.Series(dtype="int64")
    n_read = n_matched = 0

    reader = pd.read_csv(args.events, usecols=usecols, chunksize=args.chunksize,
                         low_memory=False)
    for ci, chunk in enumerate(reader):
        n_read += len(chunk)
        m = process_chunk(chunk, seg_gdf, args.max_dist)
        if m is not None and len(m):
            n_matched += len(m)
            st = m.groupby(["segment_id", "tier"]).size()
            sy = m.groupby(["segment_id", "ym"]).size()
            seg_tier = st if seg_tier is None else seg_tier.add(st, fill_value=0)
            seg_ym = sy if seg_ym is None else seg_ym.add(sy, fill_value=0)
            veh_counts = veh_counts.add(m["vehicletype"].value_counts(), fill_value=0)
            ym_counts = ym_counts.add(m["ym"].value_counts(), fill_value=0)
            tier_counts = tier_counts.add(m["tier"].value_counts(), fill_value=0)
        print(f"  chunk {ci:>3d} | read {n_read:>12,} | matched {n_matched:>12,} "
              f"| {time.time()-t0:6.0f}s", flush=True)
        if args.smoke and n_read >= args.smoke:
            print("  [smoke] stopping early")
            break

    if seg_tier is None:
        print("  ERROR: no events matched — check coords / max_dist")
        sys.exit(1)

    # ---- per-segment tier totals -------------------------------------------
    tier_wide = seg_tier.unstack(fill_value=0)
    for t in ("candidate", "critical", "extreme"):
        if t not in tier_wide.columns:
            tier_wide[t] = 0
    tier_wide = tier_wide.rename(columns={
        "candidate": "hb_candidate", "critical": "hb_critical",
        "extreme": "hb_extreme"})
    tier_wide["nm_total"] = tier_wide[["hb_candidate", "hb_critical",
                                       "hb_extreme"]].sum(axis=1)
    tier_wide["nm_braking"] = tier_wide["nm_total"]
    tier_wide["HB_Total"] = tier_wide["nm_total"]
    tier_wide = tier_wide.reset_index()
    tier_wide["segment_id"] = tier_wide["segment_id"].astype(str).str.strip()

    # ---- merge onto base modelling table (inherit geometry + crashes) ------
    base = pd.read_csv(args.base, dtype={"segment_id": str}, low_memory=False)
    base["segment_id"] = base["segment_id"].astype(str).str.strip()
    hb_cols = ["nm_total", "nm_braking", "hb_candidate", "hb_critical",
               "hb_extreme", "HB_Total", "HB_Critical-Likely", "HB_Extreme",
               "HB_Hard-Braking Candidate"]
    drop = [c for c in hb_cols if c in base.columns]
    merged = base.drop(columns=drop).merge(tier_wide, on="segment_id", how="left")
    fillcols = ["nm_total", "nm_braking", "hb_candidate", "hb_critical",
                "hb_extreme", "HB_Total"]
    for c in fillcols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

    n_orphan = int((merged["nm_total"] == 0).sum())
    out_ecf = os.path.join(args.out_dir, "near_miss_hb_ecf_year.csv")
    merged.to_csv(out_ecf, index=False)

    # ---- segment x month panel (temporal transfer) -------------------------
    panel = seg_ym.reset_index()
    panel.columns = ["segment_id", "ym", "hb_count"]
    panel["segment_id"] = panel["segment_id"].astype(str).str.strip()
    out_panel = os.path.join(args.out_dir, "near_miss_hb_panel_year.csv")
    panel.to_csv(out_panel, index=False)

    # ---- provenance --------------------------------------------------------
    match_rate = 100.0 * n_matched / max(n_read, 1)
    prov_rows = [("events_read", n_read), ("events_matched", n_matched),
                 ("match_rate_pct", round(match_rate, 2)),
                 ("segments_with_hb", int((merged["nm_total"] > 0).sum())),
                 ("segments_zero_hb", n_orphan),
                 ("hb_total_year", int(merged["nm_total"].sum()))]
    for k, v in tier_counts.sort_index().items():
        prov_rows.append((f"tier_{k}", int(v)))
    for k, v in veh_counts.sort_index().items():
        prov_rows.append((f"veh_{k}", int(v)))
    prov = pd.DataFrame(prov_rows, columns=["metric", "value"])
    prov.to_csv(os.path.join(args.out_dir, "hb_year_provenance.csv"), index=False)

    ym_tbl = ym_counts.sort_index()
    with open(os.path.join(args.out_dir, "hb_year_provenance.tex"), "w") as f:
        f.write("% auto-generated by aggregate_hb_year.py\n")
        f.write("\\begin{tabular}{lr}\n\\hline\n")
        f.write("Telematics coverage (full year) & Hard-braking events \\\\\n\\hline\n")
        for k, v in ym_tbl.items():
            f.write(f"{k} & {int(v):,} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Total matched & {n_matched:,} \\\\\n")
        f.write(f"Match rate & {match_rate:.1f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    print("\n" + "=" * 70)
    print(f"  DONE in {time.time()-t0:.0f}s")
    print(f"  events read/matched : {n_read:,} / {n_matched:,} ({match_rate:.1f}%)")
    print(f"  HB total (year)     : {int(merged['nm_total'].sum()):,}")
    print(f"  months covered      : {len(ym_tbl)}  ({ym_tbl.index.min()}..{ym_tbl.index.max()})")
    print(f"  segments with HB    : {int((merged['nm_total']>0).sum()):,} / {len(merged):,}")
    print(f"  -> {out_ecf}")
    print(f"  -> {out_panel}")
    print("=" * 70)


if __name__ == "__main__":
    main()
