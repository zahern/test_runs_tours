"""
01_zone_crosswalk.py
Build authoritative SA2->EMME crosswalk covering all 507 SA2 zones.
"""
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from difflib import SequenceMatcher

OUT_DIR = r"Z:\test_runs_tours\demand_"
os.makedirs(OUT_DIR, exist_ok=True)

# --- a) Load 507 SA2 zone IDs ---
with open(r"Z:\Paper_Plan\synth_pop\zone_ids_sa_only_507.txt") as f:
    content = f.read()
zone_ids_507 = [int(x) for x in re.split(r'[\s,]+', content.strip()) if x]
print(f"Loaded {len(zone_ids_507)} SA2 zone IDs from zone_ids file")

# --- b) Load SA2 shapefile ---
shp_path = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
gdf_all = gpd.read_file(shp_path)
print(f"Shapefile columns: {gdf_all.columns.tolist()}")
print(f"Total SA2 rows: {len(gdf_all)}")

# Filter to Queensland (STE_CODE21 = '3')
gdf_qld = gdf_all[gdf_all['STE_CODE21'] == '3'].copy()
print(f"Queensland SA2 count: {len(gdf_qld)}")

# Reproject to WGS84 for lat/lon centroids
gdf_qld = gdf_qld.to_crs(epsg=4326)
gdf_qld['sa2_lat'] = gdf_qld.geometry.centroid.y
gdf_qld['sa2_lon'] = gdf_qld.geometry.centroid.x
gdf_qld['SA2_CODE21'] = gdf_qld['SA2_CODE21'].astype(str).str.strip()
gdf_qld['SA2_NAME21'] = gdf_qld['SA2_NAME21'].astype(str).str.strip()
gdf_qld['AREASQKM21'] = pd.to_numeric(gdf_qld['AREASQKM21'], errors='coerce')

# Build SA2_CODE -> name, lat, lon lookup
shp_lookup = gdf_qld.set_index('SA2_CODE21')[['SA2_NAME21', 'sa2_lat', 'sa2_lon', 'AREASQKM21']].to_dict('index')

# --- c) Load zones_reference.csv ---
zones_ref = pd.read_csv(
    r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM\#Model_Masters\scripts\post_demand\zones_reference.csv"
)
print(f"zones_reference shape: {zones_ref.shape}")
print(f"zones_reference columns: {zones_ref.columns.tolist()}")
# Columns: emme_zone_id, sa2_name, latitude, longitude, centroid_x, centroid_y
zones_ref['sa2_name'] = zones_ref['sa2_name'].astype(str).str.strip()
zones_ref_valid = zones_ref[zones_ref['sa2_name'].notna() & (zones_ref['sa2_name'] != 'nan')].copy()
print(f"zones_reference valid sa2_name rows: {len(zones_ref_valid)}")

# --- d) Load SEQSTM_Zones CSV ---
seqstm_zones = pd.read_csv(
    r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM\#NETWORKS_4_3\Common\SEQSTM_Zones_wExt_LvL4_vF.csv",
    encoding='latin1'
)
print(f"SEQSTM_Zones shape: {seqstm_zones.shape}")
seqstm_unique_sa2 = seqstm_zones['SA2_NAME'].dropna().unique()
print(f"SEQSTM unique SA2_NAME values: {len(seqstm_unique_sa2)}")

# --- e) Build SA2_NAME->SA2_CODE21 lookup from shapefile (case-insensitive) ---
name_to_code = {}
for code, row in shp_lookup.items():
    name_key = row['SA2_NAME21'].lower().strip()
    name_to_code[name_key] = code

# Build sa2_name -> list of emme_zone_ids from zones_reference
sa2_name_to_emme = {}
for _, row in zones_ref_valid.iterrows():
    name = row['sa2_name'].strip()
    eid = int(row['emme_zone_id'])
    sa2_name_to_emme.setdefault(name, []).append(eid)

print(f"Unique SA2 names in zones_reference: {len(sa2_name_to_emme)}")

# SEQSTM SA2 name set for fuzzy matching
seqstm_sa2_names = sorted(set(seqstm_zones['SA2_NAME'].dropna().str.strip().tolist()))
zones_ref_names = sorted(sa2_name_to_emme.keys())

def fuzzy_match(name, candidates, threshold=0.75):
    """Return best fuzzy match above threshold."""
    best_score = 0
    best_match = None
    name_l = name.lower().strip()
    for c in candidates:
        score = SequenceMatcher(None, name_l, c.lower().strip()).ratio()
        if score > best_score:
            best_score = score
            best_match = c
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score

# --- f) For each 507 SA2 code, build crosswalk ---
records = []
match_counts = {'exact': 0, 'fuzzy': 0, 'none': 0}

for sa2_code in zone_ids_507:
    code_str = str(sa2_code)

    # Get name from shapefile
    shp_row = shp_lookup.get(code_str)
    if shp_row:
        sa2_name = shp_row['SA2_NAME21']
        sa2_lat = shp_row['sa2_lat']
        sa2_lon = shp_row['sa2_lon']
        area_sqkm = shp_row['AREASQKM21']
    else:
        sa2_name = f"UNKNOWN_{code_str}"
        sa2_lat = np.nan
        sa2_lon = np.nan
        area_sqkm = np.nan

    # Try exact match in zones_reference sa2_name
    emme_ids = sa2_name_to_emme.get(sa2_name, None)
    match_type = None
    sa2_name_seqstm = None

    if emme_ids:
        match_type = 'exact'
        sa2_name_seqstm = sa2_name
    else:
        # Try case-insensitive exact
        sa2_name_lower = sa2_name.lower().strip()
        for ref_name, ids in sa2_name_to_emme.items():
            if ref_name.lower().strip() == sa2_name_lower:
                emme_ids = ids
                match_type = 'exact_ci'
                sa2_name_seqstm = ref_name
                break

    if not emme_ids:
        # Fuzzy match
        best, score = fuzzy_match(sa2_name, zones_ref_names)
        if best:
            emme_ids = sa2_name_to_emme.get(best, [])
            match_type = f'fuzzy_{score:.2f}'
            sa2_name_seqstm = best
        else:
            match_type = 'none'
            sa2_name_seqstm = None
            emme_ids = []

    if match_type.startswith('exact'):
        match_counts['exact'] += 1
    elif match_type.startswith('fuzzy'):
        match_counts['fuzzy'] += 1
    else:
        match_counts['none'] += 1

    records.append({
        'sa2_code': sa2_code,
        'sa2_name': sa2_name,
        'sa2_lat': sa2_lat,
        'sa2_lon': sa2_lon,
        'area_sqkm': area_sqkm,
        'sa2_name_seqstm': sa2_name_seqstm,
        'match_type': match_type,
        'n_emme_zones': len(emme_ids),
        'emme_zone_ids': ','.join(str(e) for e in sorted(emme_ids))
    })

crosswalk = pd.DataFrame(records)

# --- g) For unmatched SA2 zones, find nearest matched SA2 by lat/lon ---
matched = crosswalk[crosswalk['n_emme_zones'] > 0].copy()
unmatched = crosswalk[crosswalk['n_emme_zones'] == 0].copy()
print(f"\nBefore nearest-neighbor imputation: {len(matched)} matched, {len(unmatched)} unmatched")

if len(unmatched) > 0 and len(matched) > 0:
    matched_coords = matched[['sa2_lat', 'sa2_lon']].values.astype(float)
    for idx, row in unmatched.iterrows():
        if np.isnan(row['sa2_lat']) or np.isnan(row['sa2_lon']):
            crosswalk.at[idx, 'proxy_sa2_code'] = np.nan
            continue
        dists = np.sqrt(
            (matched_coords[:, 0] - row['sa2_lat'])**2 +
            (matched_coords[:, 1] - row['sa2_lon'])**2
        )
        nearest_idx = np.argmin(dists)
        proxy_code = matched.iloc[nearest_idx]['sa2_code']
        proxy_emme = matched.iloc[nearest_idx]['emme_zone_ids']
        crosswalk.at[idx, 'proxy_sa2_code'] = proxy_code
        crosswalk.at[idx, 'proxy_emme_zone_ids'] = proxy_emme
        crosswalk.at[idx, 'match_type'] = 'proxy_nearest'

# Ensure proxy columns exist
for col in ['proxy_sa2_code', 'proxy_emme_zone_ids']:
    if col not in crosswalk.columns:
        crosswalk[col] = np.nan

# --- h) Save sa2_crosswalk_507.csv ---
out_csv = os.path.join(OUT_DIR, 'sa2_crosswalk_507.csv')
crosswalk.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv} ({len(crosswalk)} rows)")

# --- i) Save expanded CSV (one row per sa2_code, emme_zone_id) ---
expanded_rows = []
for _, row in crosswalk.iterrows():
    if row['n_emme_zones'] > 0 and pd.notna(row['emme_zone_ids']) and row['emme_zone_ids']:
        for eid in str(row['emme_zone_ids']).split(','):
            if eid.strip():
                expanded_rows.append({
                    'sa2_code': int(row['sa2_code']),
                    'sa2_name': row['sa2_name'],
                    'emme_zone_id': int(eid.strip())
                })

expanded = pd.DataFrame(expanded_rows)
out_exp = os.path.join(OUT_DIR, 'sa2_crosswalk_507_expanded.csv')
expanded.to_csv(out_exp, index=False)
print(f"Saved {out_exp} ({len(expanded)} rows)")

# --- j) Print summary ---
print("\n=== CROSSWALK SUMMARY ===")
print(f"Total SA2 zones: {len(crosswalk)}")
print(f"Exact matches: {match_counts['exact']}")
print(f"Fuzzy matches: {match_counts['fuzzy']}")
print(f"No match (proxy): {match_counts['none']}")
print(f"Total EMME zones covered: {expanded['emme_zone_id'].nunique()}")
print(f"\nMatch type breakdown:")
print(crosswalk['match_type'].value_counts())
print(f"\nSample matched:")
print(crosswalk[crosswalk['n_emme_zones']>0].head(5)[['sa2_code','sa2_name','match_type','n_emme_zones']].to_string())
print(f"\nSample unmatched:")
print(crosswalk[crosswalk['n_emme_zones']==0].head(5)[['sa2_code','sa2_name','match_type']].to_string())
