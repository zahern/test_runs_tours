"""
16_build_sa1_emme_crosswalk.py
Map each SA1 zone in the SEQ+Tweed study area to its nearest EMME zone(s)
using centroid-to-centroid KDTree matching. This is the foundation for
extracting fine-resolution walk/bike skims from the 2,062-zone EMME source
at SA1 origin level instead of losing within-SA2 variation by averaging to SA2.

Why SA1: SA1 zones (13,241 in study area, median 21 per SA2) are small
enough that walk distances within a single SA1 are roughly constant (~500m
diameter), making SA1 the right granularity for walk/bike mode modelling.

Output: demand_/sa1_to_emme_crosswalk_seqtweed.csv
  Columns: SA1_CODE21, SA2_CODE21, emme_zone_id, emme_lat, emme_lon,
           dist_to_emme_m (snap distance for QA)
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

DEMAND_DIR = r"Z:\test_runs_tours\demand_"
SA1_SHP = r"Z:\test_runs_tours\SA1_SHP\SA1_2021_AUST_GDA2020.shp"
SA1_CSV = r"Z:\test_runs_tours\SA1_SHP\sa1_2021.csv"
EMME_MAPPING = os.path.join(DEMAND_DIR, "sa2_to_emme_mapping.csv")
ZONE_LIST = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
OUT_PATH = os.path.join(DEMAND_DIR, "sa1_to_emme_crosswalk_seqtweed.csv")


def main():
    import re
    with open(ZONE_LIST) as f:
        zone_ids = sorted(int(x) for x in re.split(r"[\s,]+", f.read().strip()) if x)
    zone_set = set(str(z) for z in zone_ids)
    print(f"Study area SA2 zones: {len(zone_set)}")

    # SA1→SA2 lookup (fast, from CSV)
    sa1_lu = pd.read_csv(SA1_CSV, dtype=str)
    sa1_study = sa1_lu[sa1_lu["SA2_CODE21"].isin(zone_set)].copy()
    print(f"SA1 zones in study area: {len(sa1_study)}")

    # SA1 centroids from shapefile (projected → WGS84)
    print("Loading SA1 shapefile for centroids...")
    gdf = gpd.read_file(SA1_SHP)
    gdf = gdf[gdf["SA1_CODE21"].isin(set(sa1_study["SA1_CODE21"]))].to_crs(4326)
    gdf["sa1_lat"] = gdf.geometry.centroid.y
    gdf["sa1_lon"] = gdf.geometry.centroid.x
    sa1_centroids = gdf[["SA1_CODE21", "sa1_lat", "sa1_lon"]].copy()
    print(f"SA1 centroid rows: {len(sa1_centroids)}")

    # EMME zone centroids
    emme = pd.read_csv(EMME_MAPPING)
    emme = emme[emme["latitude"].notna() & emme["longitude"].notna()].copy()
    emme_coords = emme[["emme_zone_id", "longitude", "latitude"]].drop_duplicates("emme_zone_id")
    print(f"EMME zones with centroids: {len(emme_coords)}")

    # KDTree: snap each SA1 centroid to nearest EMME zone
    emme_xy = np.column_stack([emme_coords["longitude"].values, emme_coords["latitude"].values])
    tree = cKDTree(emme_xy)

    sa1_xy = np.column_stack([
        sa1_centroids["sa1_lon"].values,
        sa1_centroids["sa1_lat"].values,
    ])
    dists, idxs = tree.query(sa1_xy, k=1)

    # 1 degree ≈ 111km; convert degree distance to metres
    dist_m = dists * 111_000
    matched_emme = emme_coords.iloc[idxs]["emme_zone_id"].values
    matched_lat  = emme_coords.iloc[idxs]["latitude"].values
    matched_lon  = emme_coords.iloc[idxs]["longitude"].values

    out = sa1_centroids.copy().reset_index(drop=True)
    out["emme_zone_id"]  = matched_emme
    out["emme_lat"]      = matched_lat
    out["emme_lon"]      = matched_lon
    out["dist_to_emme_m"] = dist_m.round(1)

    # Merge SA2 code
    out = out.merge(sa1_study[["SA1_CODE21", "SA2_CODE21"]], on="SA1_CODE21", how="left")

    # Filter to SA1 zones within the SEQ-STM model boundary: snap distance
    # ≤ 15km from the nearest EMME centroid. Zones further away are outside
    # the SEQ-STM network (e.g. SA1 zones in Cairns or Outback QLD that are
    # in the study-area SA2 zone list but outside SEQ-STM's coverage) and
    # would get meaningless EMME walk/bike distances snapped from 50+ km away.
    # Those zones fall back to the SA2-averaged skim in the walk/bike builder.
    MAX_SNAP_M = 15_000
    n_outside = int((dist_m > MAX_SNAP_M).sum())
    out = out[out["dist_to_emme_m"] <= MAX_SNAP_M].copy()
    print(f"\nFiltered to SA1 within {MAX_SNAP_M/1000:.0f}km of SEQ-STM model boundary:")
    print(f"  Retained: {len(out)} SA1 zones")
    print(f"  Excluded (outside SEQ-STM model): {n_outside} SA1 zones "
          f"(will use SA2-averaged walk/bike skim in stage5)")

    out.to_csv(OUT_PATH, index=False)
    print(f"\nWritten: {OUT_PATH}  ({len(out)} SA1 zones)")
    print(f"Snap distance: median={out['dist_to_emme_m'].median():.0f}m  "
          f"max={out['dist_to_emme_m'].max():.0f}m")
    print("\nSample:")
    print(out[["SA1_CODE21", "SA2_CODE21", "emme_zone_id", "dist_to_emme_m"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
