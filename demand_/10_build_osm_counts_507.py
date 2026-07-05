"""
10_build_osm_counts_507.py
Rebuild OSM POI tag-value counts for all 507 SA2 zones (superset of the
existing 329-zone Z:\\test_runs_tours\\code\\SEQ_SA2_2016_OSM_counts.csv).

No builder script for that legacy file was found in the codebase, so this
reimplements it from scratch: for each SA2 zone polygon (from the ABS SA2
shapefile already used by demand_'s 01_zone_crosswalk.py), runs a single
Overpass polygon query across a broad set of OSM tag keys
(amenity/shop/leisure/tourism/office/craft/healthcare/landuse), then tallies
a count column per distinct tag VALUE seen (regardless of which key it came
from) — matching the legacy file's flattened-by-value column convention
(e.g. 'school', 'cafe', 'residential' are each a single count column).

Network-bound and slow (507 Overpass polygon queries) — results are cached
to disk per zone (saved_data/osm_cache_507/<sa2_code>.json) so the script is
safely resumable/restartable.

Output: Z:\\test_runs_tours\\new_pop\\saved_data\\SA2_2016_OSM_counts_507.csv,
one row per SA2 zone, one column per OSM tag value + SA2_MAIN16 + zone
metadata columns mirroring the legacy schema where available.
"""
import json
import os
import time
import requests
import geopandas as gpd
import pandas as pd

SHP_PATH = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
ZONE_LIST_507 = r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt"
OUT_DIR = r"Z:\test_runs_tours\new_pop\saved_data"
CACHE_DIR = os.path.join(OUT_DIR, "osm_cache_507")
OUT_PATH = os.path.join(OUT_DIR, "SA2_2016_OSM_counts_507.csv")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TAG_KEYS = ["amenity", "shop", "leisure", "tourism", "office", "craft", "healthcare", "landuse"]
HEADERS = {"User-Agent": "synth-pop-paper-research/1.0 (contact: zekeahern96@gmail.com)"}


def _overpass_query(query, retries=4, wait=5.0):
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            print(f"    HTTP {resp.status_code}, retry {attempt + 1}/{retries}")
        except Exception as e:
            print(f"    error: {e}, retry {attempt + 1}/{retries}")
        time.sleep(wait * (attempt + 1))
    return None


def _polygon_query(coords, tag_keys):
    # coords: list of (lat, lon) outer-ring points (Overpass poly filter)
    poly_str = " ".join(f"{lat:.6f} {lon:.6f}" for lat, lon in coords)
    union_parts = []
    for key in tag_keys:
        for el in ("node", "way"):
            union_parts.append(f'{el}["{key}"](poly:"{poly_str}");')
    return "[out:json][timeout:90];(" + "".join(union_parts) + ");out tags center;"


def _zone_coords(geom, max_points=300):
    """Outer-ring lat/lon points for the (possibly multi-)polygon, downsampled."""
    if geom.geom_type == "Polygon":
        polys = [geom]
    else:
        polys = list(geom.geoms)
    # Use the largest sub-polygon's exterior ring (good enough for a bbox-like
    # poly filter on mostly-contiguous SA2 areas; avoids overpass query-length
    # limits for zones with many small offshore/island fragments).
    largest = max(polys, key=lambda p: p.area)
    coords = list(largest.exterior.coords)
    if len(coords) > max_points:
        step = len(coords) // max_points
        coords = coords[::step]
    return [(lat, lon) for lon, lat in coords]  # shapely is (x=lon, y=lat)


def fetch_zone_counts(sa2_code, geom):
    cache_path = os.path.join(CACHE_DIR, f"{sa2_code}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    coords = _zone_coords(geom)
    query = _polygon_query(coords, TAG_KEYS)
    data = _overpass_query(query)

    tags_seen = []
    if data is not None:
        for el in data.get("elements", []):
            tags_seen.append(el.get("tags", {}))

    with open(cache_path, "w") as f:
        json.dump(tags_seen, f)
    return tags_seen


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(ZONE_LIST_507) as f:
        zone_ids_507 = sorted(int(x) for x in f.read().split())
    print(f"Target SA2 zones: {len(zone_ids_507)}")

    gdf = gpd.read_file(SHP_PATH)
    gdf["SA2_CODE21"] = pd.to_numeric(gdf["SA2_CODE21"], errors="coerce")
    gdf = gdf.dropna(subset=["SA2_CODE21"])
    gdf["SA2_CODE21"] = gdf["SA2_CODE21"].astype(int)
    gdf = gdf[gdf["SA2_CODE21"].isin(zone_ids_507)].set_index("SA2_CODE21")
    missing_geom = [z for z in zone_ids_507 if z not in gdf.index]
    if missing_geom:
        print(f"WARNING: {len(missing_geom)} zones have no shapefile geometry, skipped: {missing_geom}")

    all_counts = {}
    n_done = 0
    for sa2_code in zone_ids_507:
        if sa2_code not in gdf.index:
            all_counts[sa2_code] = {}
            continue
        cache_path = os.path.join(CACHE_DIR, f"{sa2_code}.json")
        already = os.path.exists(cache_path)
        tags_seen = fetch_zone_counts(sa2_code, gdf.loc[sa2_code, "geometry"])

        counts = {}
        for tags in tags_seen:
            for key in TAG_KEYS:
                if key in tags:
                    val = tags[key]
                    if val and val != "yes":
                        counts[val] = counts.get(val, 0) + 1
        all_counts[sa2_code] = counts

        n_done += 1
        tag_label = "(cached)" if already else "(fetched)"
        print(f"  [{n_done}/{len(zone_ids_507)}] SA2 {sa2_code}: {len(tags_seen)} elements, "
              f"{len(counts)} distinct values {tag_label}")

    df = pd.DataFrame.from_dict(all_counts, orient="index").fillna(0).astype(int)
    df.index.name = "SA2_MAIN16"
    df = df.reset_index()
    df = df.reindex(columns=["SA2_MAIN16"] + sorted(c for c in df.columns if c != "SA2_MAIN16"))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nWritten: {OUT_PATH}  ({len(df)} zones, {len(df.columns) - 1} OSM tag-value columns)")


if __name__ == "__main__":
    main()
