"""
fix_xy.py
Transforms xy.dat from DynusT custom CRS → GDA2020 lat/lon → MGA56 (EPSG:7856)
to match zone.dat, and saves corrected xy.dat to prepare_network/new/
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

INPUT_XY  = r"Z:\test_runs_tours\prepare_network\xy.dat"
OUTPUT_DIR = r"Z:\test_runs_tours\prepare_network\new"
OUTPUT_XY  = os.path.join(OUTPUT_DIR, "xy.dat")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CRS = 7856  # GDA2020 / MGA Zone 56

# -------------------------------------------------------
# 1. AFFINE PARAMETERS (exact copy of dynus.py get_affine)
# -------------------------------------------------------
def get_affine():
    lat_lon = np.array([
        [-27.57639,   152.699837],
        [-27.263422,  153.079553],
        [-28.1682445, 153.543538],
    ])
    x_y = np.array([
        [264800, 372600],
        [399690, 498300],
        [563910, 135885],
    ])
    lat, lon = lat_lon[:, 0], lat_lon[:, 1]
    A = np.zeros((6, 6))
    b = np.zeros(6)
    for i in range(3):
        A[i,   0] = lon[i]; A[i,   1] = lat[i]; A[i,   2] = 1; b[i]   = x_y[i, 0]
        A[i+3, 3] = lon[i]; A[i+3, 4] = lat[i]; A[i+3, 5] = 1; b[i+3] = x_y[i, 1]
    return np.linalg.solve(A, b)

def xy_to_lonlat(x, y, params):
    a, b_, c, d, e, f = params
    det = a*e - b_*d
    lon = (e*(x - c) - b_*(y - f)) / det
    lat = (a*(y - f) - d*(x - c)) / det
    return lon, lat

# -------------------------------------------------------
# 2. LOAD xy.dat
# -------------------------------------------------------
print(f"Loading {INPUT_XY}...")
df = pd.read_csv(INPUT_XY, header=None, names=['NodeID', 'X', 'Y'], sep=r'\s+', engine='python')
print(f"  Loaded {len(df)} nodes")
print(f"  X range: {df['X'].min():.1f} - {df['X'].max():.1f}")
print(f"  Y range: {df['Y'].min():.1f} - {df['Y'].max():.1f}")

# -------------------------------------------------------
# 3. AFFINE TRANSFORM → lat/lon (GDA2020, EPSG:7844)
# -------------------------------------------------------
print("Applying affine transform (DynusT -> GDA2020 lat/lon)...")
params = get_affine()
lons, lats = zip(*[xy_to_lonlat(row.X, row.Y, params) for _, row in df.iterrows()])
df['lon'] = lons
df['lat'] = lats

# Sanity check — SE Queensland bounds
lon_ok = df['lon'].between(150, 155)
lat_ok = df['lat'].between(-30, -25)
pct_valid = (lon_ok & lat_ok).mean() * 100
print(f"  Nodes within SE Qld bounds (150-155 E, 25-30 S): {pct_valid:.1f}%")
print(f"  Lon range: {df['lon'].min():.4f} - {df['lon'].max():.4f}")
print(f"  Lat range: {df['lat'].min():.4f} - {df['lat'].max():.4f}")

# -------------------------------------------------------
# 4. REPROJECT → MGA56 (EPSG:7856)
# -------------------------------------------------------
print(f"Reprojecting to EPSG:{TARGET_CRS} (MGA Zone 56)...")
gdf = gpd.GeoDataFrame(
    df[['NodeID', 'X', 'Y', 'lon', 'lat']],
    geometry=[Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])],
    crs=7844  # GDA2020 geographic
)
gdf_m = gdf.to_crs(TARGET_CRS)

# Extract reprojected coordinates from geometry (NOT the original X/Y columns)
gdf_m['E'] = gdf_m.geometry.x   # Easting in MGA56
gdf_m['N'] = gdf_m.geometry.y   # Northing in MGA56

# Sanity check
print(f"  Easting  range: {gdf_m['E'].min():.1f} - {gdf_m['E'].max():.1f}")
print(f"  Northing range: {gdf_m['N'].min():.1f} - {gdf_m['N'].max():.1f}")
# Expected: E ≈ 380,000–700,000; N ≈ 6,850,000–7,100,000 for SE Qld

# -------------------------------------------------------
# 5. WRITE OUTPUT xy.dat
# -------------------------------------------------------
print(f"Writing {OUTPUT_XY}...")
with open(OUTPUT_XY, 'w') as f:
    for _, row in gdf_m.iterrows():
        # Match original format: padded NodeID, then E, N with 3 decimal places
        f.write(f"{int(row['NodeID']):>8}   {row['E']:>15.3f}   {row['N']:>15.3f}\n")

print(f"  Done. {len(gdf_m)} nodes written.")

# Verify output
with open(OUTPUT_XY) as f:
    lines = f.readlines()
print(f"  Output lines: {len(lines)}")
print(f"  First 5 lines:\n{''.join(lines[:5])}")
print(f"  Last  5 lines:\n{''.join(lines[-5:])}")

# -------------------------------------------------------
# 6. ALSO COPY zone.dat TO new/ (unchanged — already MGA56)
# -------------------------------------------------------
import shutil
src_zone = r"Z:\test_runs_tours\prepare_network\zone.dat"
dst_zone  = os.path.join(OUTPUT_DIR, "zone.dat")
shutil.copy2(src_zone, dst_zone)
print(f"\nCopied zone.dat → {dst_zone} (unchanged, already MGA56)")

# -------------------------------------------------------
# 7. CROSS-CHECK: do xy nodes fall within zone.dat bounds?
# -------------------------------------------------------
zone_e_min, zone_e_max = 468000, 511000
zone_n_min, zone_n_max = 6968000, 7035000
# Use a slightly wider check to account for external/fringe nodes
e_check = gdf_m['E'].between(zone_e_min - 100000, zone_e_max + 100000)
n_check = gdf_m['N'].between(zone_n_min - 100000, zone_n_max + 100000)
pct_in = (e_check & n_check).mean() * 100
print(f"\nCross-check: {pct_in:.1f}% of nodes fall within extended zone.dat bounds")
print(f"  (zone.dat covers E {zone_e_min}–{zone_e_max}, N {zone_n_min}–{zone_n_max})")

print("\n=== COMPLETE ===")
print(f"Output: {OUTPUT_XY}")
