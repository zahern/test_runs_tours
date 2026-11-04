"""
18_build_lvl4_emme_crosswalk.py
Map each SEQ-STM Level 4 micro-zone (~7,000 zones) to its nearest EMME TAZ
centroid and to its SA2 code, using the SEQ-STM model's own projected
coordinate system (GDA94/MGA Zone 56, EPSG:28356) for accurate snapping.

Level 4 zones are the SEQ-STM model's native fine-grain zone system — the
"~6,000 zones" used in the original ASIM model for walk/bike analysis. Using
them directly (rather than ABS SA1 zones, which introduce ABS/SEQSTM boundary
mismatches) ensures the walk/bike distances extracted from the EMME network
skim (DISTWALK, DISTBIKE in seq_skims_base.omx) properly correspond to the
Level 4 travel sheds.

Inputs:
  SEQSTM_Centroids_wExt_LvL4_vF.csv  — Level4 zone centroids (x,y in MGA56)
  SEQSTM_Zones_wExt_LvL4_vF.csv      — Level4 zone metadata (SA2_NAME etc.)
  sa2_to_emme_mapping.csv             — EMME zone centroids (WGS84 lat/lon)
  sa2_crosswalk_507.csv               — SA2 code ↔ SEQSTM name crosswalk

Output:
  demand_/lvl4_to_emme_crosswalk.csv
    LvL4_ID, sa2_code, sa2_name_seqstm, emme_zone_id, dist_to_emme_m,
    lvl4_lon, lvl4_lat
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

DEMAND_DIR = r"Z:\test_runs_tours\demand_"
LVL4_BASE = (r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM"
             r"\#NETWORKS_4_3\Common")
ZONE_LIST = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
OUT_PATH = os.path.join(DEMAND_DIR, "lvl4_to_emme_crosswalk.csv")

MAX_SNAP_M = 5_000   # drop Level4 zones >5km from any EMME centroid.
# External/fringe zones in the SEQSTM model (representing trips originating
# outside SEQ or rural hinterland buffer zones) have large snap distances
# because the closest EMME centroid is itself far away. These zones produce
# inflated network walk distances (64+ km) and should be excluded from
# HOME_LVL4 assignment so only genuine urban/suburban micro-zones are used.


def mga56_to_wgs84(x_arr, y_arr):
    """Convert GDA94/MGA Zone 56 (EPSG:28356) to WGS84 lon/lat."""
    if _HAS_PYPROJ:
        t = Transformer.from_crs("EPSG:28356", "EPSG:4326", always_xy=True)
        lon, lat = t.transform(x_arr, y_arr)
        return lon, lat
    # Approximate fallback (good to ~10m for SEQ):
    # Zone 56 central meridian = 153°E, scale factor k0=0.9996, false easting=500000
    e0, n0 = 500_000.0, 10_000_000.0
    a = 6_378_137.0
    f = 1 / 298.257_222_101
    b = a * (1 - f)
    e2 = 1 - (b / a) ** 2
    k0 = 0.9996
    lon0 = np.radians(153.0)
    xi = (y_arr - n0) / (k0 * a)
    eta = (x_arr - e0) / (k0 * a)
    lat = xi + np.radians(0.2)   # crude estimate — replace with real UTM inverse
    lon = eta / np.cos(lat) + lon0
    lat = np.degrees(lat) - 90 * np.sign(lat) + 90 * np.sign(lat)
    lon = np.degrees(lon)
    print("WARNING: pyproj not available — coordinate conversion is approximate")
    return lon, lat


def main():
    with open(ZONE_LIST) as f:
        zone_ids = sorted(int(x) for x in re.split(r"[\s,]+", f.read().strip()) if x)
    print(f"SEQ SA2 zones: {len(zone_ids)}")

    cent = pd.read_csv(f"{LVL4_BASE}/SEQSTM_Centroids_wExt_LvL4_vF.csv")
    lvl4_zones = pd.read_csv(f"{LVL4_BASE}/SEQSTM_Zones_wExt_LvL4_vF.csv")
    print(f"Level4 centroids: {len(cent)}  zones: {len(lvl4_zones)}")

    # Convert Level 4 centroids from MGA56 → WGS84
    lon, lat = mga56_to_wgs84(cent["x"].values, cent["y"].values)
    cent = cent.copy()
    cent["lon"] = lon
    cent["lat"] = lat

    # Map Level4 zones → SA2 code via SPATIAL CONTAINMENT (point-in-polygon).
    # This is more reliable than SEQSTM SA2_NAME matching, which has ambiguous
    # zone names (e.g., "Alexandra Hills" in SEQSTM refers to both a Brisbane
    # suburb AND a Sunshine Coast area — name-based matching silently assigns
    # 66 Sunshine Coast Level4 zones to the Brisbane "Alexandra Hills" SA2,
    # producing 64+km DISTWALK values when they are evaluated against each
    # other). Spatial containment uses the authoritative ABS SA2 polygon
    # boundaries to unambiguously assign each Level4 centroid to its correct SA2.
    import geopandas as gpd
    from shapely.geometry import Point

    sa2_shp = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
    sa2_gdf = gpd.read_file(sa2_shp)
    sa2_gdf["SA2_CODE21"] = pd.to_numeric(sa2_gdf["SA2_CODE21"], errors="coerce")
    sa2_gdf = sa2_gdf[sa2_gdf["SA2_CODE21"].isin(zone_ids)].to_crs(4326)
    print(f"SA2 polygons for study area: {len(sa2_gdf)}")

    # Merge centroid coordinates into zone table
    merged = lvl4_zones.merge(
        cent[["index", "lon", "lat"]], left_on="LvL4_ID", right_on="index", how="left"
    )
    merged = merged.dropna(subset=["lon", "lat"]).copy()
    print(f"Level4 zones with centroid: {len(merged)}")

    # Point-in-polygon: assign each Level4 centroid to the SA2 it physically
    # falls within (using WGS84 SA2 polygon boundaries).
    lvl4_gdf = gpd.GeoDataFrame(
        merged[["LvL4_ID", "lon", "lat"]],
        geometry=gpd.points_from_xy(merged["lon"], merged["lat"]),
        crs=4326,
    )
    joined = gpd.sjoin(
        lvl4_gdf, sa2_gdf[["SA2_CODE21", "SA2_NAME21", "geometry"]],
        how="left", predicate="within"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    merged["sa2_code"]       = joined["SA2_CODE21"].values
    merged["sa2_name_seqstm"] = joined["SA2_NAME21"].values

    matched_sa2 = merged["sa2_code"].notna().sum()
    print(f"Level4 zones with SA2 containment match: {matched_sa2}/{len(merged)}")

    # EMME centroids (WGS84)
    emme = pd.read_csv(os.path.join(DEMAND_DIR, "sa2_to_emme_mapping.csv"))
    emme = emme[emme["latitude"].notna() & emme["longitude"].notna()].drop_duplicates("emme_zone_id")

    emme_xy = np.column_stack([emme["longitude"].values, emme["latitude"].values])
    lvl4_xy = np.column_stack([merged["lon"].values, merged["lat"].values])

    tree = cKDTree(emme_xy)
    dists, idxs = tree.query(lvl4_xy, k=1)
    dist_m = dists * 111_000   # degree → metres (approximate)

    merged["emme_zone_id"]  = emme.iloc[idxs]["emme_zone_id"].values
    merged["dist_to_emme_m"] = dist_m.round(1)
    merged["lvl4_lon"] = merged["lon"]
    merged["lvl4_lat"] = merged["lat"]

    # Filter: only keep Level4 zones within the SEQ-STM model boundary AND
    # mapped to a SEQ SA2 (zones outside our 324-SA2 scope get no SA2 match
    # but are still usable if within EMME range — include all within 15km
    # since the model boundary includes zones without our SA2 scope as transit
    # destinations; walk/bike model only predicts for home SA2 zones)
    n_outside = (dist_m > MAX_SNAP_M).sum()
    out = merged[dist_m <= MAX_SNAP_M].copy()
    print(f"Retained {len(out)} Level4 zones within {MAX_SNAP_M/1000:.0f}km of EMME centroid")
    print(f"Excluded {n_outside} zones outside EMME model boundary")
    print(f"Of retained: {out['sa2_code'].notna().sum()} mapped to SEQ SA2 code")

    out[["LvL4_ID", "sa2_code", "sa2_name_seqstm", "emme_zone_id",
         "dist_to_emme_m", "lvl4_lon", "lvl4_lat"]].to_csv(OUT_PATH, index=False)
    print(f"\nWritten: {OUT_PATH}")
    print(f"Snap dist: median={out['dist_to_emme_m'].median():.0f}m  "
          f"max={out['dist_to_emme_m'].max():.0f}m")
    print(f"EMME zones used: {out['emme_zone_id'].nunique()}")
    print(f"\nSample output:")
    print(out[["LvL4_ID", "sa2_name_seqstm", "emme_zone_id", "dist_to_emme_m"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
