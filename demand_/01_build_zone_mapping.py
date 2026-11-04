"""
Script 01: Build SA2 <-> EMME zone crosswalk
SA2_CODE -> EMME_zone_id(s) -> SA2 polygon geometry
"""
import os
import sys
import pandas as pd
import geopandas as gpd
import difflib

# ── Paths ──────────────────────────────────────────────────────────────────────
SEQSTM_BASE   = r"z:\activity_sim_resources\new_asim\SEQ-STM-ASIM\SEQ-STM-ASIM"
ZONES_REF_CSV = SEQSTM_BASE + r"\#Model_Masters\scripts\post_demand\zones_reference.csv"
SEQSTM_ZONES  = SEQSTM_BASE + r"\#NETWORKS_4_3\Common\SEQSTM_Zones_wExt_LvL4_vF.csv"
SA2_SHP       = r"Z:\test_runs_tours\SA2_2021_AUST_SHP_GDA2020\SA2_2021_AUST_GDA2020.shp"
OUTPUT_DIR    = r"Z:\test_runs_tours\demand_"

# ── 1. Load zones_reference ─────────────────────────────────────────────────
print("=" * 60)
print("Script 01: Build Zone Mapping")
print("=" * 60)

print("\n[1] Loading zones_reference.csv ...")
zr = pd.read_csv(ZONES_REF_CSV)
print(f"    Loaded {len(zr)} zones. Columns: {list(zr.columns)}")

# Only keep zones that have an SA2 name assigned
zr_sa2 = zr[zr['sa2_name'].notna()].copy()
print(f"    Zones with SA2 name: {len(zr_sa2)}")

# ── 2. Load SEQSTM zones CSV for SA2_NAME ───────────────────────────────────
print("\n[2] Loading SEQSTM zones CSV ...")
sz = pd.read_csv(SEQSTM_ZONES, usecols=['LvL4_ID', 'SA2_NAME', 'Area'])
sz.rename(columns={'LvL4_ID': 'emme_zone_id', 'SA2_NAME': 'sa2_name_seqstm', 'Area': 'area_sqkm'}, inplace=True)
sz = sz[sz['sa2_name_seqstm'].notna()].copy()
print(f"    Loaded {len(sz)} SEQSTM zones with SA2 name")

# Merge zones_reference with SEQSTM zones on zone ID
zr_full = zr.merge(sz, on='emme_zone_id', how='left')
print(f"    After merge: {len(zr_full)} rows")

# ── 3. Load SA2 shapefile ─────────────────────────────────────────────────────
print("\n[3] Loading SA2 shapefile ...")
sa2_shp = gpd.read_file(SA2_SHP)
print(f"    Loaded {len(sa2_shp)} SA2 polygons")
print(f"    Columns: {list(sa2_shp.columns)}")
print(sa2_shp[['SA2_CODE21', 'SA2_NAME21', 'STE_NAME21']].head(3).to_string())

# Filter to Queensland (QLD) and nearby states to keep manageable
# SA2 shapefile has all Australia – filter to QLD
qld_sa2 = sa2_shp[sa2_shp['STE_NAME21'] == 'Queensland'].copy()
print(f"    QLD SA2 zones: {len(qld_sa2)}")

# ── 4. Match SEQSTM sa2_name -> SA2_NAME21 ───────────────────────────────────
print("\n[4] Matching SEQSTM SA2 names to ABS SA2 names ...")

# Build lookup: lower-stripped ABS name -> SA2_CODE21
abs_name_to_code = {}
abs_name_clean_to_code = {}
for _, row in sa2_shp.iterrows():
    name = str(row['SA2_NAME21']).strip()
    abs_name_to_code[name.lower()] = (row['SA2_CODE21'], name)

# Unique SEQSTM SA2 names from the combined data
# Use zr_full which has sa2_name (from zones_reference) and sa2_name_seqstm (from SEQSTM CSV)
# Prefer sa2_name_seqstm when available
zr_full['sa2_name_for_match'] = zr_full['sa2_name_seqstm'].fillna(zr_full['sa2_name'])

unique_seqstm_names = zr_full['sa2_name_for_match'].dropna().unique()
print(f"    Unique SEQSTM SA2 names: {len(unique_seqstm_names)}")

# Match: exact first, then fuzzy
all_abs_names_lower = list(abs_name_to_code.keys())
name_match_results = {}  # seqstm_name -> (sa2_code, abs_name, match_type)

exact = 0
fuzzy = 0
no_match = 0

for seqstm_name in unique_seqstm_names:
    key = seqstm_name.strip().lower()
    if key in abs_name_to_code:
        code, abs_name = abs_name_to_code[key]
        name_match_results[seqstm_name] = (code, abs_name, 'exact')
        exact += 1
    else:
        # Try fuzzy match
        close = difflib.get_close_matches(key, all_abs_names_lower, n=1, cutoff=0.7)
        if close:
            code, abs_name = abs_name_to_code[close[0]]
            name_match_results[seqstm_name] = (code, abs_name, f'fuzzy:{close[0]}')
            fuzzy += 1
        else:
            name_match_results[seqstm_name] = (None, None, 'no_match')
            no_match += 1
            print(f"    NO MATCH: '{seqstm_name}'")

total_names = len(unique_seqstm_names)
print(f"\n    Match results for {total_names} unique SEQSTM SA2 names:")
print(f"      Exact matches:  {exact}  ({100*exact/total_names:.1f}%)")
print(f"      Fuzzy matches:  {fuzzy}  ({100*fuzzy/total_names:.1f}%)")
print(f"      No match:       {no_match} ({100*no_match/total_names:.1f}%)")

# ── 5. Build output dataframe ─────────────────────────────────────────────────
print("\n[5] Building sa2_to_emme_mapping.csv ...")

rows = []
for _, row in zr_full.iterrows():
    name = row['sa2_name_for_match']
    if pd.isna(name):
        continue
    result = name_match_results.get(name, (None, None, 'no_match'))
    sa2_code, abs_name, match_type = result
    rows.append({
        'emme_zone_id': row['emme_zone_id'],
        'sa2_code': sa2_code,
        'sa2_name_seqstm': name,
        'sa2_name_abs': abs_name,
        'match_type': match_type,
        'longitude': row.get('longitude'),
        'latitude': row.get('latitude'),
    })

mapping_df = pd.DataFrame(rows)
print(f"    Total rows: {len(mapping_df)}")

# Count EMME zones per SA2
n_per_sa2 = mapping_df.groupby('sa2_code')['emme_zone_id'].count().rename('n_emme_zones_in_sa2')
mapping_df = mapping_df.merge(n_per_sa2, on='sa2_code', how='left')

out_mapping = os.path.join(OUTPUT_DIR, 'sa2_to_emme_mapping.csv')
mapping_df.to_csv(out_mapping, index=False)
print(f"    Saved: {out_mapping}")

# ── 6. Build sa2_zones_reference.csv ─────────────────────────────────────────
print("\n[6] Building sa2_zones_reference.csv ...")

# Aggregate by SA2
grp = mapping_df.groupby('sa2_code').agg(
    sa2_name=('sa2_name_seqstm', 'first'),
    sa2_name_abs=('sa2_name_abs', 'first'),
    n_emme_zones=('emme_zone_id', 'count'),
    centroid_lon=('longitude', 'mean'),
    centroid_lat=('latitude', 'mean'),
).reset_index()

# Add area from SEQSTM zones
area_by_sa2 = sz.groupby('sa2_name_seqstm')['area_sqkm'].sum().reset_index()
area_by_sa2.rename(columns={'sa2_name_seqstm': 'sa2_name'}, inplace=True)
grp = grp.merge(area_by_sa2, on='sa2_name', how='left')

# Add SA2 polygon area from shapefile
shp_area = sa2_shp[['SA2_CODE21', 'AREASQKM21']].copy()
shp_area.columns = ['sa2_code', 'area_sqkm_abs']
grp = grp.merge(shp_area, on='sa2_code', how='left')
# Use ABS area if SEQSTM area is missing
grp['area_sqkm'] = grp['area_sqkm'].fillna(grp['area_sqkm_abs'])
grp.drop(columns=['area_sqkm_abs'], inplace=True)

out_ref = os.path.join(OUTPUT_DIR, 'sa2_zones_reference.csv')
grp.to_csv(out_ref, index=False)
print(f"    SA2 zones in reference: {len(grp)}")
print(f"    Saved: {out_ref}")

# ── 7. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SCRIPT 01 COMPLETE")
print(f"  EMME zones mapped: {mapping_df['sa2_code'].notna().sum()} / {len(mapping_df)}")
print(f"  Unique SA2 codes matched: {mapping_df['sa2_code'].nunique()}")
print(f"  Output: {out_mapping}")
print(f"  Output: {out_ref}")
print("=" * 60)
