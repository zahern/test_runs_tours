"""
matsim_export.py
================
Convert the tours-pipeline final activity plan into a runnable **MATSim**
scenario (population ``plans.xml`` + ``facilities.xml`` + minimal ``config.xml``).

The staged ABM (Stage 0..6, see FRAMEWORK_PLAN.md) emits zone-based
Aimsun/DynusT trip legs (``stage6_output.csv`` / ``trips_for_aimsun.csv`` with
``OTAZ`` / ``DTAZ``).  MATSim instead needs a population of persons, each with a
day *plan* of XY-coordinate activities separated by mode legs.  This module
reconstructs those plans and writes standard MATSim XML.

Design notes
------------
* Self-contained: only the Python stdlib + numpy/pandas + (optional) pyproj.
  No MATSim Java dependency.
* Coordinates: zone centroids (lat/lon, WGS84) reprojected to **GDA94 / MGA
  Zone 56 (EPSG:28356)** - the standard projected CRS for South-East
  Queensland.  If ``pyproj`` is unavailable the exporter degrades to raw
  lat/lon (EPSG:4326) and stamps the config CRS accordingly, so a run never
  hard-fails on reprojection.
* Reuses the pipeline's own zone-centroid utilities
  (``spatial_assignment.get_zone_centroids_rich`` /
  ``build_sa2_centroids_from_landuse``) when a landuse table is supplied,
  otherwise reads ``SA1_SHP/sa2_centroids.csv`` directly.

CLI
---
    python matsim_export.py --run-dir runs/<run> --out runs/<run>/matsim
    python matsim_export.py --run-dir runs/<run> --crs EPSG:28356
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Target projected CRS for SE Queensland (GDA94 / MGA Zone 56).
DEFAULT_CRS = "EPSG:28356"

# Pipeline mode_name codes -> MATSim network modes.
MODE_MAP = {
    "DA": "car",        # drive alone
    "Other": "ride",    # shared ride / HOV
    "SR": "ride",
    "Transit": "pt",
    "PT": "pt",
    "Walk": "walk",
    "Bike": "bike",
    "Cycle": "bike",
}
DEFAULT_MODE = "car"

# Nominal typical durations (seconds) per activity type for the scoring config.
_TYPICAL_DURATION = {
    "home": 12 * 3600,
    "work": 8 * 3600,
    "education": 6 * 3600,
    "shop": 1 * 3600,
    "shopping": 1 * 3600,
    "social": 2 * 3600,
    "recreation": 2 * 3600,
    "escort": 30 * 60,
    "personal": 45 * 60,
    "other": 1 * 3600,
}
_DEFAULT_TYPICAL_DURATION = 3600


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #
def _minutes_to_hms(minutes) -> str | None:
    """Minutes-from-midnight -> 'HH:MM:SS'.  None/NaN -> None."""
    if minutes is None:
        return None
    try:
        m = float(minutes)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(m) or m < 0:
        return None
    total = int(round(m * 60.0))              # seconds from midnight
    h, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"{h:02d}:{mm:02d}:{ss:02d}"


def _first_col(df: pd.DataFrame, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _act_type(purpose) -> str:
    if purpose is None or (isinstance(purpose, float) and not np.isfinite(purpose)):
        return "other"
    return str(purpose).strip().lower() or "other"


def _mode_of(mode_name) -> str:
    if mode_name is None:
        return DEFAULT_MODE
    return MODE_MAP.get(str(mode_name).strip(), DEFAULT_MODE)


# --------------------------------------------------------------------------- #
#  zone -> coordinate table                                                    #
# --------------------------------------------------------------------------- #
def _load_centroid_frame(landuse=None) -> pd.DataFrame:
    """Return a DataFrame with columns ['zone_id', 'lat', 'lon']."""
    sa2_geo = None
    if landuse is not None:
        try:
            from spatial_assignment import get_zone_centroids_rich
            sa2_geo = get_zone_centroids_rich(landuse, cache_dir="saved_data/osm_cache")
        except Exception as e:                                    # noqa: BLE001
            logger.warning("get_zone_centroids_rich failed (%s); trying landuse fallback", e)
            try:
                from spatial_assignment import build_sa2_centroids_from_landuse
                sa2_geo = build_sa2_centroids_from_landuse(landuse)
            except Exception as e2:                              # noqa: BLE001
                logger.warning("build_sa2_centroids_from_landuse failed (%s)", e2)
                sa2_geo = None

    if sa2_geo is None:
        # direct CSV fallback (SA2, lat, lon) shipped with the pipeline
        for path in ("SA1_SHP/sa2_centroids.csv", "saved_data/sa2_centroids.csv"):
            if os.path.exists(path):
                sa2_geo = pd.read_csv(path)
                break
    if sa2_geo is None:
        raise FileNotFoundError(
            "No zone centroids available: pass landuse, or provide "
            "SA1_SHP/sa2_centroids.csv"
        )

    sa2_geo = sa2_geo.copy()
    zone_col = _first_col(sa2_geo, "zone_id", "ORDER", "DTAZ", "OTAZ",
                          "SA2_MAIN16", "SA2", "zone")
    lat_col = _first_col(sa2_geo, "lat", "latitude", "y_lat", "centroid_lat")
    lon_col = _first_col(sa2_geo, "lon", "lng", "longitude", "x_lon", "centroid_lon")
    if zone_col is None or lat_col is None or lon_col is None:
        raise ValueError(
            f"Centroid table needs a zone id + lat + lon; got columns {list(sa2_geo.columns)}"
        )
    out = sa2_geo[[zone_col, lat_col, lon_col]].rename(
        columns={zone_col: "zone_id", lat_col: "lat", lon_col: "lon"})
    out = out.dropna(subset=["zone_id", "lat", "lon"])
    return out


def make_transformer(crs: str = DEFAULT_CRS):
    """Return ``(transform, effective_crs)`` where ``transform(lon, lat) -> (x, y)``
    reprojects WGS84 geographic coordinates to ``crs``.

    If ``crs`` is geographic, or pyproj is unavailable, ``transform`` is the
    identity (x=lon, y=lat) and ``effective_crs`` becomes ``EPSG:4326``.
    """
    if crs.upper() in ("EPSG:4326", "WGS84"):
        return (lambda lon, lat: (lon, lat)), "EPSG:4326"
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        def _t(lon, lat):
            # accepts scalars or numpy arrays; returns same shape
            return tr.transform(lon, lat)                        # x=easting, y=northing
        return _t, crs
    except Exception as e:                                       # noqa: BLE001
        logger.warning(
            "pyproj reprojection to %s unavailable (%s); emitting lat/lon (EPSG:4326)",
            crs, e)
        return (lambda lon, lat: (lon, lat)), "EPSG:4326"


def load_zone_xy(landuse=None, transform=None, crs: str = DEFAULT_CRS):
    """Build ``{zone_id: (x, y)}`` for facilities using zone centroids.

    Best-effort: returns ``{}`` if no centroid source is available.  ``transform``
    (from :func:`make_transformer`) reprojects lat/lon; if omitted one is built
    from ``crs``.
    """
    if transform is None:
        transform, _ = make_transformer(crs)
    try:
        frame = _load_centroid_frame(landuse)
    except Exception as e:                                       # noqa: BLE001
        logger.warning("zone centroids unavailable (%s); facilities omitted", e)
        return {}

    zone_xy = {}
    for zid, lat, lon in zip(frame["zone_id"].to_numpy(),
                             frame["lat"].astype(float).to_numpy(),
                             frame["lon"].astype(float).to_numpy()):
        x, y = transform(lon, lat)
        try:
            key = int(zid)
        except (TypeError, ValueError):
            key = str(zid)
        zone_xy[key] = (float(x), float(y))
    return zone_xy


def _zone_key(z):
    try:
        return int(z)
    except (TypeError, ValueError):
        return str(z)


# --------------------------------------------------------------------------- #
#  plan reconstruction                                                         #
# --------------------------------------------------------------------------- #
def _is_geographic(x_series, y_series):
    """Heuristic: are these x/y columns actually lon/lat (degrees)?"""
    try:
        return (x_series.abs().max() <= 180.0) and (y_series.abs().max() <= 90.0)
    except Exception:                                            # noqa: BLE001
        return False


def _sample_persons(df, pid_col, sample_frac, max_persons, seed):
    """Return df restricted to a random subset of persons (deterministic)."""
    pids = df[pid_col].dropna().unique()
    n = len(pids)
    keep = n
    if sample_frac is not None and 0 < sample_frac < 1:
        keep = max(1, int(round(n * sample_frac)))
    if max_persons is not None:
        keep = min(keep, max_persons)
    if keep >= n:
        return df, n, n
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(pids, size=keep, replace=False))
    return df[df[pid_col].isin(chosen)], n, keep


def build_plans(plan_df: pd.DataFrame, zone_xy: dict, transform=None,
                max_persons: int | None = None, sample_frac: float | None = None,
                seed: int = 42):
    """Reconstruct MATSim day plans from the pipeline's trip-leg table.

    Coordinates come from (in priority order):
      1. embedded per-trip ``origin_lat/lon`` + ``dest_lat/lon`` (reprojected),
      2. embedded ``origin_x/y`` + ``dest_x/y`` (reprojected if geographic, else
         used as-is),
      3. zone-centroid lookup via ``zone_xy``.

    Returns ``(persons, stats)``; each person has
    ``len(acts) == len(legs) + 1``.
    """
    if transform is None:
        transform = lambda lon, lat: (lon, lat)                  # noqa: E731
    df = plan_df.copy()
    if "row_type" in df.columns:
        df = df[df["row_type"] == "trip"].copy()

    pid_col = _first_col(df, "person_id", "PERSID", "pid")
    o_col = _first_col(df, "OTAZ", "otaz", "o_zone", "origin_zone")
    d_col = _first_col(df, "DTAZ", "dtaz", "d_zone", "dest_zone")
    dep_col = _first_col(df, "departure_time", "STARTIME", "start_time", "dep_time")
    mode_col = _first_col(df, "mode_name", "mode")
    purp_col = _first_col(df, "purpose", "act_purpose", "activity")
    if not all((pid_col, dep_col)) or not (o_col and d_col):
        raise ValueError(
            "trip table needs person id, OTAZ, DTAZ and a departure time; "
            f"got columns {list(df.columns)}"
        )

    df[dep_col] = pd.to_numeric(df[dep_col], errors="coerce")

    # ---- decide the coordinate source ----------------------------------
    olat = _first_col(df, "origin_lat", "olat", "o_lat")
    olon = _first_col(df, "origin_lon", "olon", "o_lon")
    dlat = _first_col(df, "dest_lat", "dlat", "d_lat")
    dlon = _first_col(df, "dest_lon", "dlon", "d_lon")
    ox_c = _first_col(df, "origin_x", "ox")
    oy_c = _first_col(df, "origin_y", "oy")
    dx_c = _first_col(df, "dest_x", "dx")
    dy_c = _first_col(df, "dest_y", "dy")

    coord_source = "zone_centroid"
    if olat and olon and dlat and dlon:
        coord_source = "embedded_latlon"
        ox_arr, oy_arr = transform(df[olon].to_numpy(float), df[olat].to_numpy(float))
        dx_arr, dy_arr = transform(df[dlon].to_numpy(float), df[dlat].to_numpy(float))
        df["_ox"], df["_oy"], df["_dx"], df["_dy"] = ox_arr, oy_arr, dx_arr, dy_arr
    elif ox_c and oy_c and dx_c and dy_c:
        if _is_geographic(df[ox_c], df[oy_c]):                   # x/y are lon/lat
            coord_source = "embedded_xy_geographic"
            ox_arr, oy_arr = transform(df[ox_c].to_numpy(float), df[oy_c].to_numpy(float))
            dx_arr, dy_arr = transform(df[dx_c].to_numpy(float), df[dy_c].to_numpy(float))
            df["_ox"], df["_oy"], df["_dx"], df["_dy"] = ox_arr, oy_arr, dx_arr, dy_arr
        else:                                                    # already projected
            coord_source = "embedded_xy_projected"
            df["_ox"], df["_oy"] = df[ox_c].astype(float), df[oy_c].astype(float)
            df["_dx"], df["_dy"] = df[dx_c].astype(float), df[dy_c].astype(float)
    else:
        df[o_col] = pd.to_numeric(df[o_col], errors="coerce")
        df[d_col] = pd.to_numeric(df[d_col], errors="coerce")
        df = df[(df[o_col] > 0) & (df[d_col] > 0)]

    embedded = coord_source != "zone_centroid"

    # ---- population sampling (fraction and/or hard cap) ----------------
    df, n_total, n_kept = _sample_persons(df, pid_col, sample_frac, max_persons, seed)
    max_persons = None   # already applied at the person-set level

    persons = []
    stats = {
        "coord_source": coord_source,
        "sample_frac": sample_frac,
        "n_persons_total": n_total,
        "n_persons_in": int(df[pid_col].nunique()),
        "n_persons_out": 0,
        "n_persons_skipped_no_coords": 0,
        "n_activities": 0,
        "n_legs": 0,
        "mode_counts": {},
        "activity_counts": {},
        "missing_zones": set(),
    }

    for pid, grp in df.groupby(pid_col, sort=False):
        if max_persons is not None and stats["n_persons_out"] >= max_persons:
            break
        grp = grp.sort_values(dep_col)
        rows = list(grp.itertuples(index=False))
        col_index = {name: i for i, name in enumerate(grp.columns)}

        def _cell(r, col):
            return r[col_index[col]] if col in col_index else None

        def _o_xy(r):
            if embedded:
                return float(_cell(r, "_ox")), float(_cell(r, "_oy"))
            z = _zone_key(_cell(r, o_col))
            return zone_xy.get(z), z

        def _d_xy(r):
            if embedded:
                return float(_cell(r, "_dx")), float(_cell(r, "_dy"))
            z = _zone_key(_cell(r, d_col))
            return zone_xy.get(z), z

        acts, legs, ok = [], [], True

        # first activity = origin of the first trip (day anchor, 'home')
        if embedded:
            x0, y0 = _o_xy(rows[0])
        else:
            xy, z = _o_xy(rows[0])
            if xy is None:
                stats["missing_zones"].add(z); ok = False
            else:
                x0, y0 = xy
        if ok:
            acts.append({"type": "home", "x": x0, "y": y0,
                         "end_time": _minutes_to_hms(_cell(rows[0], dep_col))})

        for i, r in enumerate(rows):
            if not ok:
                break
            if embedded:
                xd, yd = _d_xy(r)
            else:
                xy, z = _d_xy(r)
                if xy is None:
                    stats["missing_zones"].add(z); ok = False; break
                xd, yd = xy
            legs.append({"mode": _mode_of(_cell(r, mode_col)) if mode_col else DEFAULT_MODE})
            atype = _act_type(_cell(r, purp_col)) if purp_col else "other"
            end_time = _minutes_to_hms(_cell(rows[i + 1], dep_col)) if i + 1 < len(rows) else None
            acts.append({"type": atype, "x": xd, "y": yd, "end_time": end_time})

        if not ok or len(acts) < 2:
            stats["n_persons_skipped_no_coords"] += 1
            continue

        acts[-1]["end_time"] = None                              # open-ended final activity
        persons.append({"id": str(pid), "acts": acts, "legs": legs})
        stats["n_persons_out"] += 1
        stats["n_activities"] += len(acts)
        stats["n_legs"] += len(legs)
        for lg in legs:
            stats["mode_counts"][lg["mode"]] = stats["mode_counts"].get(lg["mode"], 0) + 1
        for a in acts:
            stats["activity_counts"][a["type"]] = stats["activity_counts"].get(a["type"], 0) + 1

    stats["missing_zones"] = sorted(str(z) for z in stats["missing_zones"])[:50]
    return persons, stats


# --------------------------------------------------------------------------- #
#  XML writers                                                                 #
# --------------------------------------------------------------------------- #
def write_plans_xml(persons, path: str):
    """Write a MATSim population_v6 plans file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<!DOCTYPE population SYSTEM '
                '"http://www.matsim.org/files/dtd/population_v6.dtd">\n')
        f.write("<population>\n")
        for p in persons:
            f.write(f'  <person id={quoteattr(p["id"])}>\n')
            f.write('    <plan selected="yes">\n')
            acts, legs = p["acts"], p["legs"]
            for i, a in enumerate(acts):
                attrs = (f'type={quoteattr(a["type"])} '
                         f'x="{a["x"]:.2f}" y="{a["y"]:.2f}"')
                if a.get("end_time"):
                    attrs += f' end_time="{a["end_time"]}"'
                f.write(f"      <activity {attrs} />\n")
                if i < len(legs):
                    f.write(f'      <leg mode={quoteattr(legs[i]["mode"])} />\n')
            f.write("    </plan>\n")
            f.write("  </person>\n")
        f.write("</population>\n")


def write_facilities_xml(zone_xy, path: str):
    """Write one MATSim facility per zone at its centroid."""
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<!DOCTYPE facilities SYSTEM '
                '"http://www.matsim.org/files/dtd/facilities_v1.dtd">\n')
        f.write('<facilities name="tours-pipeline zone centroids">\n')
        for zid, (x, y) in sorted(zone_xy.items(), key=lambda kv: str(kv[0])):
            f.write(f'  <facility id={quoteattr(f"zone_{zid}")} '
                    f'x="{x:.2f}" y="{y:.2f}" />\n')
        f.write("</facilities>\n")


def write_config_xml(path: str, plans_rel: str, facilities_rel: str,
                     crs: str, activity_types):
    """Write a minimal, runnable MATSim config_v2 file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<!DOCTYPE config SYSTEM '
                '"http://www.matsim.org/files/dtd/config_v2.dtd">\n')
        f.write("<config>\n")
        f.write('  <module name="global">\n')
        f.write(f'    <param name="coordinateSystem" value={quoteattr(crs)} />\n')
        f.write("  </module>\n")
        f.write('  <module name="plans">\n')
        f.write(f'    <param name="inputPlansFile" value={quoteattr(plans_rel)} />\n')
        f.write("  </module>\n")
        if facilities_rel:
            f.write('  <module name="facilities">\n')
            f.write(f'    <param name="inputFacilitiesFile" value={quoteattr(facilities_rel)} />\n')
            f.write("  </module>\n")
        f.write('  <module name="controler">\n')
        f.write('    <param name="outputDirectory" value="./output" />\n')
        f.write('    <param name="firstIteration" value="0" />\n')
        f.write('    <param name="lastIteration" value="0" />\n')
        f.write("  </module>\n")
        f.write('  <module name="planCalcScore">\n')
        f.write('    <parameterset type="scoringParameters">\n')
        for atype in sorted(activity_types):
            dur = _TYPICAL_DURATION.get(atype, _DEFAULT_TYPICAL_DURATION)
            f.write('      <parameterset type="activityParams">\n')
            f.write(f'        <param name="activityType" value={quoteattr(atype)} />\n')
            f.write(f'        <param name="typicalDuration" value="{_minutes_to_hms(dur / 60.0)}" />\n')
            f.write("      </parameterset>\n")
        f.write("    </parameterset>\n")
        f.write("  </module>\n")
        f.write("</config>\n")


# --------------------------------------------------------------------------- #
#  orchestrator                                                                #
# --------------------------------------------------------------------------- #
def _find_plan_csv(run_dir: str):
    """Locate the final plan / trip CSV under ``run_dir``."""
    candidates = [
        "trips_for_aimsun.csv",
        "stage6_output.csv",
        os.path.join("saved_data", "trips_for_aimsun.csv"),
        os.path.join("saved_data", "stage6_output.csv"),
    ]
    for rel in candidates:
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            return p
    # recursive fallback (shallow) for suite/variant layouts
    for root, _dirs, files in os.walk(run_dir):
        for name in ("trips_for_aimsun.csv", "stage6_output.csv"):
            if name in files:
                return os.path.join(root, name)
    return None


def export_matsim(run_dir: str, out_dir: str | None = None, landuse=None,
                  crs: str = DEFAULT_CRS, plan_csv: str | None = None,
                  max_persons: int | None = None, sample_frac: float | None = None,
                  seed: int = 42):
    """Convert a finished run's activity plan into a MATSim scenario.

    Writes ``plans.xml``, ``facilities.xml`` (when zone centroids are available),
    ``config.xml`` and ``export_manifest.json`` into ``out_dir`` (default
    ``<run_dir>/matsim``).  Returns the manifest dict.
    """
    if plan_csv is None:
        plan_csv = _find_plan_csv(run_dir)
    if plan_csv is None or not os.path.exists(plan_csv):
        raise FileNotFoundError(
            f"No final plan CSV (trips_for_aimsun.csv / stage6_output.csv) under {run_dir}")

    out_dir = out_dir or os.path.join(run_dir, "matsim")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("MATSim export: reading %s", plan_csv)
    plan_df = pd.read_csv(plan_csv)
    transform, effective_crs = make_transformer(crs)
    zone_xy = load_zone_xy(landuse, transform=transform)          # {} if unavailable
    persons, stats = build_plans(plan_df, zone_xy, transform=transform,
                                 max_persons=max_persons, sample_frac=sample_frac,
                                 seed=seed)

    plans_path = os.path.join(out_dir, "plans.xml")
    cfg_path = os.path.join(out_dir, "config.xml")
    write_plans_xml(persons, plans_path)
    fac_rel = None
    if zone_xy:
        fac_path = os.path.join(out_dir, "facilities.xml")
        write_facilities_xml(zone_xy, fac_path)
        fac_rel = "facilities.xml"
    write_config_xml(cfg_path, "plans.xml", fac_rel,
                     effective_crs, set(stats["activity_counts"]))

    manifest = {
        "source_csv": os.path.abspath(plan_csv),
        "crs": effective_crs,
        "requested_crs": crs,
        "n_zones": len(zone_xy),
        "plans_xml": os.path.abspath(plans_path),
        "facilities_xml": os.path.abspath(os.path.join(out_dir, "facilities.xml")) if zone_xy else None,
        "config_xml": os.path.abspath(cfg_path),
        **{k: v for k, v in stats.items()},
    }
    with open(os.path.join(out_dir, "export_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "MATSim export: %d persons, %d activities, %d legs -> %s (crs=%s)",
        stats["n_persons_out"], stats["n_activities"], stats["n_legs"],
        out_dir, effective_crs)
    return manifest


def _parse_args():
    p = argparse.ArgumentParser(
        description="Export a tours-pipeline run to a MATSim scenario.")
    p.add_argument("--run-dir", required=True,
                   help="Run directory containing the final plan CSV.")
    p.add_argument("--out", default=None,
                   help="Output directory (default <run-dir>/matsim).")
    p.add_argument("--crs", default=DEFAULT_CRS,
                   help="Target CRS (default EPSG:28356 = GDA94/MGA Zone 56).")
    p.add_argument("--plan-csv", default=None,
                   help="Explicit plan CSV path (overrides auto-discovery).")
    p.add_argument("--max-persons", type=int, default=None,
                   help="Cap the number of persons written (for quick checks).")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="Randomly keep this fraction of persons (0-1), e.g. 0.1 for a 10%% sample.")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    manifest = export_matsim(args.run_dir, out_dir=args.out, crs=args.crs,
                             plan_csv=args.plan_csv, max_persons=args.max_persons,
                             sample_frac=args.sample_frac, seed=args.seed)
    print(json.dumps({k: manifest[k] for k in
                      ("crs", "n_persons_out", "n_activities", "n_legs", "plans_xml")
                      if k in manifest}, indent=2))


if __name__ == "__main__":
    main()
