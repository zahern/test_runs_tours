"""
18_prepare_eb_dataset.py
Prepare near_miss_hb_ecf.csv for Empirical Bayes analysis by aggregating
hard braking events and segment roads data to road-segment level.

Creates near_miss_hb_ecf.csv with columns:
  - road_osmid: OSM road segment ID
  - road_name: street name
  - highway: road type
  - speed_limit: posted limit
  - length_km: segment length
  - dist_stadium_km: distance to Suncorp Stadium
  - hb_candidate: count of hard-braking candidates (Low confidence)
  - hb_critical: count of critical-likely events
  - hb_extreme: count of extreme events
  - total_hb: total hard braking events
  - speeding: speeding observations
  - exposure_trips: exposure (trip counts)
  - event_braking: braking during event windows
  - control_braking: braking during control windows

Usage:
  python 18_prepare_eb_dataset.py
"""
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
ROOT = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics/data"

ROADS_GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
HB_EVENTS = os.path.join(ROOT, "hard_braking_events.csv")
SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
ROAD_META = os.path.join(DATA, "road_meta.csv")

OUTPUT = os.path.join(DATA, "near_miss_hb_ecf.csv")


def snap_hb_to_roads(hb_df, roads_gdf, max_dist_m=50):
    """Snap hard braking events to nearest road segments."""
    print("Building spatial index...")
    
    # Get road midpoints
    mid_lon = np.array([np.mean([c[0] for c in geom.coords]) for geom in roads_gdf["geometry"]])
    mid_lat = np.array([np.mean([c[1] for c in geom.coords]) for geom in roads_gdf["geometry"]])
    
    tree = cKDTree(np.c_[mid_lon, mid_lat])
    
    # Filter HB events with valid coordinates
    valid = hb_df["SnappedLatitude"].notna() & hb_df["SnappedLongitude"].notna()
    hb_valid = hb_df[valid].copy()
    
    print(f"Snapping {len(hb_valid):,} hard braking events to {len(roads_gdf):,} roads...")
    
    # Query nearest road
    points = np.c_[hb_valid["SnappedLongitude"].values, hb_valid["SnappedLatitude"].values]
    dists, indices = tree.query(points, k=1)
    
    # Convert to meters (approximate)
    dlat = np.abs(mid_lat[indices] - points[:, 1])
    dlon = np.abs(mid_lon[indices] - points[:, 0])
    dist_m = np.sqrt((dlon * 105000)**2 + (dlat * 111000)**2)
    
    # Keep only within max_dist_m
    within_range = dist_m <= max_dist_m
    hb_valid = hb_valid[within_range].copy()
    road_indices = indices[within_range]
    
    hb_valid["road_osmid"] = roads_gdf.iloc[road_indices]["osmid"].values
    hb_valid["snap_dist_m"] = dist_m[within_range]
    
    print(f"Matched {len(hb_valid):,} events within {max_dist_m}m ({100*len(hb_valid)/len(hb_df):.1f}%)")
    
    return hb_valid


def main():
    print("=" * 80)
    print("18_prepare_eb_dataset.py - Create near_miss_hb_ecf.csv")
    print("=" * 80)
    
    # Load road metadata
    print("\n1. Loading road metadata...")
    if os.path.exists(ROAD_META):
        roads_meta = pd.read_csv(ROAD_META)
        print(f"   Loaded {len(roads_meta):,} roads from road_meta.csv")
    else:
        print("   Warning: road_meta.csv not found, will compute from GeoJSON")
        roads_meta = None
    
    # Load road geometry
    print("\n2. Loading road geometry...")
    roads_gdf = gpd.read_file(ROADS_GEOJSON)
    roads_gdf = roads_gdf.set_crs(epsg=4326) if roads_gdf.crs is None else roads_gdf.to_crs(epsg=4326)
    print(f"   Loaded {len(roads_gdf):,} road segments")
    
    # Load hard braking events
    print("\n3. Loading hard braking events...")
    hb_df = pd.read_csv(HB_EVENTS)
    print(f"   Loaded {len(hb_df):,} hard braking events")
    print(f"   Categories: {dict(hb_df['Category'].value_counts())}")
    
    # Snap HB events to roads
    print("\n4. Snapping hard braking events to roads...")
    hb_snapped = snap_hb_to_roads(hb_df, roads_gdf)
    
    # Aggregate by road_osmid and category
    print("\n5. Aggregating by road segment and category...")
    hb_agg = hb_snapped.groupby(['road_osmid', 'Category']).size().unstack(fill_value=0)
    hb_agg = hb_agg.reset_index()
    
    # Rename columns
    col_map = {
        'Hard-Braking Candidate': 'hb_candidate',
        'Critical-Likely': 'hb_critical',
        'Extreme': 'hb_extreme'
    }
    hb_agg = hb_agg.rename(columns=col_map)
    
    # Ensure all columns exist
    for col in ['hb_candidate', 'hb_critical', 'hb_extreme']:
        if col not in hb_agg.columns:
            hb_agg[col] = 0
    
    hb_agg['total_hb'] = hb_agg['hb_candidate'] + hb_agg['hb_critical'] + hb_agg['hb_extreme']
    
    print(f"   Aggregated to {len(hb_agg):,} road segments with HB events")
    print(f"   Total HB: {hb_agg['total_hb'].sum():,} (candidate: {hb_agg['hb_candidate'].sum():,}, "
          f"critical: {hb_agg['hb_critical'].sum():,}, extreme: {hb_agg['hb_extreme'].sum():,})")
    
    # Load segment roads data for exposure
    print("\n6. Loading segment roads data for exposure...")
    seg_df = pd.read_csv(SEGMENT_ROADS)
    print(f"   Loaded {len(seg_df):,} segment-phase observations")
    
    # Aggregate by road_osmid (sum across all events/phases)
    seg_agg = seg_df.groupby('road_osmid').agg({
        'braking': 'sum',
        'speeding': 'sum',
        'trips': 'sum'
    }).reset_index()
    
    # Split by event/control type
    event_agg = seg_df[seg_df['type'] == 'Event'].groupby('road_osmid')['braking'].sum().reset_index()
    event_agg = event_agg.rename(columns={'braking': 'event_braking'})
    
    control_agg = seg_df[seg_df['type'] == 'Control'].groupby('road_osmid')['braking'].sum().reset_index()
    control_agg = control_agg.rename(columns={'braking': 'control_braking'})
    
    seg_agg = seg_agg.merge(event_agg, on='road_osmid', how='left')
    seg_agg = seg_agg.merge(control_agg, on='road_osmid', how='left')
    seg_agg['event_braking'] = seg_agg['event_braking'].fillna(0)
    seg_agg['control_braking'] = seg_agg['control_braking'].fillna(0)
    
    print(f"   Aggregated to {len(seg_agg):,} road segments with exposure data")
    
    # Merge all data
    print("\n7. Merging datasets...")
    
    # Start with road metadata
    if roads_meta is not None:
        result = roads_meta.copy()
    else:
        # Extract metadata from GeoJSON
        result = roads_gdf[['osmid', 'name', 'highway']].copy()
        result = result.rename(columns={'osmid': 'road_osmid', 'name': 'road_name'})
    
    # Merge HB data
    result = result.merge(hb_agg, left_on='osmid' if 'osmid' in result.columns else 'road_osmid', 
                         right_on='road_osmid', how='left')
    
    # Merge segment roads data
    result = result.merge(seg_agg, on='road_osmid', how='left')
    
    # Fill missing values
    for col in ['hb_candidate', 'hb_critical', 'hb_extreme', 'total_hb', 
                'braking', 'speeding', 'trips', 'event_braking', 'control_braking']:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)
    
    # Rename for consistency
    if 'osmid' in result.columns and 'road_osmid' not in result.columns:
        result = result.rename(columns={'osmid': 'road_osmid'})
    if 'name' in result.columns and 'road_name' not in result.columns:
        result = result.rename(columns={'name': 'road_name'})
    
    result['road_name'] = result['road_name'].fillna('').astype(str)
    result['exposure_trips'] = result['trips']
    
    # Select final columns
    final_cols = ['road_osmid', 'road_name', 'highway', 'speed_limit', 'length_km', 
                  'dist_stadium_km', 'hb_candidate', 'hb_critical', 'hb_extreme', 
                  'total_hb', 'speeding', 'exposure_trips', 'event_braking', 'control_braking']
    
    # Keep only columns that exist
    final_cols = [c for c in final_cols if c in result.columns]
    result = result[final_cols]
    
    print(f"\n8. Final dataset: {len(result):,} road segments")
    print(f"   Segments with HB events: {(result['total_hb'] > 0).sum():,}")
    print(f"   Total HB events: {result['total_hb'].sum():,}")
    print(f"     - Candidate: {result['hb_candidate'].sum():,}")
    print(f"     - Critical: {result['hb_critical'].sum():,}")
    print(f"     - Extreme: {result['hb_extreme'].sum():,}")
    
    # Save
    print(f"\n9. Writing to {OUTPUT}...")
    result.to_csv(OUTPUT, index=False)
    
    print("\n" + "=" * 80)
    print("Dataset statistics:")
    print("=" * 80)
    print(result[['total_hb', 'hb_candidate', 'hb_critical', 'hb_extreme', 
                  'speeding', 'exposure_trips']].describe())
    
    print("\n✓ Done! Created near_miss_hb_ecf.csv")


if __name__ == "__main__":
    main()
