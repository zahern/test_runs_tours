''' ----------------------------------------------------------- '''
''' DYNUS                                                     '''
''' ----------------------------------------------------------- '''

'''
Info:
https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/main-structure-and-greater-capital-city-statistical-areas/statistical-area-level-1

Shapefile from:
https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files#post-release-changes

GDA2020 / MGA2020 (latest releases)
Projection: GDA2020 / MGA Zone XX
EPSG codes: 7844–7855 (one per MGA zone)
GEOGCS["GDA2020",DATUM["GDA2020",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433],AUTHORITY["EPSG",7844]]

Each SA1 has polygon boundaries
SA1 codes (e.g., SA1_CODE21)

EPSG: 7844 coordinate system: latitude, longitude (geographic). Units: degree
EPSG:3857 coordinate system: metres east, north
EPSG:4326: lat, lon
DynusT does not impose a particular map projection or datum; it works with the spatial coordinate system of the network you feed it.

'''

# Coordinates already in 7844
def xy_in_zone(sa1: gpd.GeoDataFrame, x:float, y:float):
# {
    point = Point(x, y)  # x/y in GDA2020
    sa1_code = sa1[sa1.contains(point)]['SA1_CODE20'].values
    print("Point is in SA1:", sa1_code[0] if sa1_code.size > 0 else "None")
# }

def in_zone(sa1: gpd.GeoDataFrame, lon:float, lat:float, crs_input: str = "EPSG:4326"):
# {
    point = Point(lon, lat)

    # Wrap in GeoSeries only if transformation is needed
    if crs_input != sa1.crs.to_string():
        point = gpd.GeoSeries([point], crs=crs_input).to_crs(sa1.crs).iloc[0]

    contained_sa1 = sa1[sa1.contains(point)]
    if not contained_sa1.empty:
        code = contained_sa1.iloc[0]['SA1_CODE21']  # or your SA1 code column
        print("Point is in SA1:", code)
        return code
    else:
        print("Point is not in any SA1 polygon.")
        return "None"
# }

def load_shapefile(file):
# {
    return gpd.read_file(file)
# }

 # Inverse affine function
def xy_to_lonlat(x, y, params):
    a, b_, c, d, e, f = params
    det = a*e - b_*d
    lon = (e*(x - c) - b_*(y - f)) / det
    lat = (a*(y - f) - d*(x - c)) / det
    return lon, lat

# Affine function
def lonlat_to_xy(lon, lat, params):
    a, b_, c, d, e, f = params
    x = a * lon + b_ * lat + c
    y = d * lon + e * lat + f
    return x, y

def get_affine():
# {
    lat_lon = np.array([
        [-27.57639, 152.699837], # Warrego Highway / Wanora
        [-27.263422, 153.079553], #[-27.263237, 153.079803], # Redcliff (left corner)
        [-28.1682445, 153.543538],  # Tweed Heads (roundabout)
        [-27.368126, 153.178835], #[-27.368118, 153.178814],  # Port of Brisbane (right - top)
        [-27.371916, 153.185859], #[-27.371912, 153.185786], # Port of Brisbane (right - bottom)
        [-26.498462, 153.09246], # Emu Mountain Road (roundabout)
        [-26.387296, 153.09831], # Noosa
        # #[-27.259823, 153.10386], # Redcliff
        #[-27.193936, 153.111887], # Redcliff
        # #[-26.6847, 153.1331], # Coloundra roundabout bottom
    ])

    x_y = np.array([
        [264800, 372600],
        [399690, 498300],
        [563910, 135885],
        [435116, 456200],
        [437494, 454656],
        [404483, 803971],
        [406720, 85054],
        #[408290, 498930],
        #[411310, 525910],
        #[414390, 682100],
    ])

    # Split lat/lon
    lat, lon = lat_lon[:,0], lat_lon[:,1]

    # Prepare A matrix for affine
    A = np.zeros((6,6))
    b = np.zeros(6)

    # X equations
    for i in range(3):
        A[i, 0] = lon[i]
        A[i, 1] = lat[i]
        A[i, 2] = 1
        b[i] = x_y[i,0]

    # Y equations
    for i in range(3):
        A[i+3, 3] = lon[i]
        A[i+3, 4] = lat[i]
        A[i+3, 5] = 1
        b[i+3] = x_y[i,1]

    params = np.linalg.solve(A, b) # Solve
    return params

    # Test:
    #x, y = 399790, 498270
    #lon, lat = xy_to_lonlat(x, y, params)
    #print ("lon=",lon," lat=",lat) # lat_lon = [-27.263237, 153.079803],
# }

def load_nodes(file, params):
# {
    df = pd.read_csv(file, header=None, names=['NodeID', 'X', 'Y'], delim_whitespace=True)

    # To map (x,y) points to google (lon, lat)
    '''plt.figure()
    plt.scatter(df['X'], df['Y'], c='blue', marker='o')
    plt.axis('equal')  # Keep aspect ratio 1:1
    plt.show()'''

    # Convert (x,y) to (lon,lat)
    df[['lon', 'lat']] = df.apply(
        lambda row: xy_to_lonlat(row['X'], row['Y'], params), axis=1, result_type='expand')
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip( df['lon'],  df['lat'])], crs=7844)

    return gdf
# }

def simplify_boundary(poly, tolerance):
# {
    coords = list(poly.exterior.coords)

    # Remove last point if it is the same as the first (closing point)
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    # If fewer points than requested, just return original
    if len(coords) <= 50:
        sampled = coords
    else:
        sampled = random.sample(coords, 50)

    # Sort them in original order to avoid self-crossing polygon
    # Find index of each chosen point in the original list
    idx_map = {tuple(pt): i for i, pt in enumerate(coords)}
    sampled_sorted = sorted(sampled, key=lambda pt: idx_map[tuple(pt)])

    # Close polygon by repeating first point
    sampled_sorted.append(sampled_sorted[0])

    return Polygon(sampled_sorted)


    #simp = poly.simplify(50, preserve_topology=True)

    '''resample_spacing = 100 # Approximate distance between points after resampling (meters)

    # Resample boundary
    coords = np.array(simp.exterior.coords)
    deltas = np.diff(coords, axis=0)
    segment_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    cumdist = np.insert(segment_lengths.cumsum(), 0, 0)

    # New distances along boundary
    new_distances = np.arange(0, cumdist[-1], resample_spacing)

    # Interpolate X and Y
    new_x = np.interp(new_distances, cumdist, coords[:, 0])
    new_y = np.interp(new_distances, cumdist, coords[:, 1])

    resampled_coords = list(zip(new_x, new_y))

    # Ensure polygon is closed
    if resampled_coords[0] != resampled_coords[-1]:
        resampled_coords.append(resampled_coords[0])

    return Polygon(resampled_coords)'''


    # binary search for simplification tolerance
    '''low, high = 0, 250  # adjust high if you need more
    for _ in range(10):  # 20 iterations is plenty
        mid = (low + high) / 2
        p = poly.simplify(mid, preserve_topology=True)
        if len(p.exterior.coords) > tolerance:
            low = mid
        else:
            high = mid
    new_polygon = poly.simplify(high, preserve_topology=True)
    return new_polygon'''
# }

'''
7844 => (lon, lat)
7856 => (x, y) metres
LOAD ZONES (7844) 
EXTRACT CONVEX HULL OF ZONE
LOAD NODES (?) => (lon, lat) USING AFFINE
(ZONES, NODES) => 7856

'''
def translate():
# {
    TARGET_CRS = 7856

    # -----------------------------
    # 1. LOAD SA ZONES AND THEIR POLYGONS
    # -----------------------------

    # SA1 => 61845 zones, 18 columns, EPSG code 7844
    # SA2 => 2473 zones
    # make sure we get the zone files
    sa_shp_file = "SA2_2021_AUST_GDA2020.shp"
    gdf_zone = load_shapefile(sa_shp_file)
    if gdf_zone.crs is None: gdf_zone.set_crs(7844, inplace=True)
    gdf_zone_m = gdf_zone.to_crs(TARGET_CRS)# Reproject to MGA Zone 56 (meters)

    # -----------------------------
    # 2. PROCESS ZONE BOUNDARY
    # -----------------------------

    valid_zone = gdf_zone_m[
        gdf_zone_m.geometry.notnull() &
        (~gdf_zone_m.geometry.is_empty) &
        (gdf_zone_m.geometry.type.isin(['Polygon', 'MultiPolygon']))
        ]

    zone_to_points = {}
    simplified_geoms = {}
    for idx, row in valid_zone.iterrows():
    # {
        zone_code = row['SA2_CODE21']  # The zone ID column
        geom = row.geometry  # Define shortcut

        # CONVEX HULL OPTION
        hull = geom.convex_hull
        pts = list(hull.exterior.coords) # Extract boundary points
        zone_to_points[zone_code] = pts  # Store points (i.e., ordered list)
        simplified_geoms[zone_code] = hull # Store geometry object

        # SECOND OPTION
        '''polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            simp = simplify_boundary(poly, 50)
            simplified_geoms[zone_code] = simp
            pts = list(simp.exterior.coords)
            zone_to_points[zone_code] = pts
            for pt in pts: unique_points.add(pt)
    
            #fig, ax = plt.subplots(figsize=(7, 7))
            #x, y = poly.exterior.xy
            #xs, ys = hull.exterior.xy
            #ax.plot(x, y, color="gray", linewidth=1, label="Original")
            #ax.plot(xs, ys, color="red", linewidth=2, label="Simplified")
            #ax.legend()
            #ax.set_aspect("equal", "box")
            #plt.show()
        '''
    # }

    # INDEXING AND UNQIUE POINTS
    point_to_idx = {} # Dictionary of points to indices
    idx_counter = 1
    zone_to_point_idx = {}
    for zone_code, pts in zone_to_points.items():
    # {
        zone_ids = []
        for pt in pts:
        # {
            if pt not in point_to_idx: # Not in the dictionary
            # {
                point_to_idx[pt] = idx_counter # Add the point and give it an index
                idx_counter += 1
            # }
            zone_ids.append(point_to_idx[pt])
        # }
        zone_to_point_idx[zone_code] = zone_ids
    # }
    unique_points_list = list(point_to_idx.keys())

    # -----------------------------
    # 3. LOAD DYNUS NODES
    # -----------------------------

    # 21306 nodes
    #xy = "C:\\Users\\rober\\PycharmProjects\\searchlogit-0.3.57\\searchlogit-0.3.57\\searchlogit\\xy_orig.dat"
    xy = "xy_orig.dat"
    params = get_affine()
    gdf_nodes = load_nodes(xy, params)
    gdf_nodes_m = gdf_nodes.to_crs(TARGET_CRS) # Reproject

    # -----------------------------
    # 4. PLOT FOR DEBUGGING
    # -----------------------------

    gdf_simplified_m = gpd.GeoDataFrame(
        {
            "SA2_CODE21": list(simplified_geoms.keys()),
            "geometry": list(simplified_geoms.values())
        }, crs=TARGET_CRS
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_zone_m.plot(ax=ax, edgecolor="black", facecolor="lightgray")
    gdf_nodes_m.plot(ax=ax, color="red", markersize=5)
    gdf_simplified_m.plot(ax=ax, edgecolor="blue", facecolor="none", linewidth=1.5)
    #plt.show()

    # ------------------------------------------------------------
    # 5. SPATIAL JOIN (NEAREST ZONE)
    # ------------------------------------------------------------
    # Only keep columns 'SA1_CODE21' and 'geometry' from gdf_sa1
    # "left" join => Keep all rows from gdf_nodes
    nearest_zone = gpd.sjoin_nearest(
        gdf_nodes_m,
        gdf_zone_m[['SA2_CODE21', 'geometry']],
        how='left',
        distance_col='dist',
        #max_distance=0.001 # Add this to see how many points without a zone
    )
    ###DEBUG: nans = nearest_zone.isna().sum().sum() print("Nans= ",nans)

    # -----------------------------
    # 6. BUILD DICTIONARIES
    # -----------------------------
    # Make a dictionary mapping nearest zones to node IDs
    nodes_in_zone = nearest_zone.groupby('SA2_CODE21')['NodeID'].apply(list).to_dict()
    zone_to_idx = {}  # Assign contiguous integer to each zone
    node_zone = {}    # Assign zone ID to each node

    for idx, (zone_code, node_list) in enumerate(nodes_in_zone.items(), start=1):
        zone_to_idx[zone_code] = idx
        for node_id in node_list:
            node_zone[node_id] = idx

    nb_zones = len(zone_to_idx)

    # -----------------------------
    # 7. WRITE XY, ZONE_ZONE AND ZONE_MAPPING DATA
    # -----------------------------
    with open("xy.dat", "w") as file:
    # {
        for idx, row in gdf_nodes_m.iterrows():
            file.write(f"{row['NodeID']}  {row.X}  {row.Y}\n")
    # }
    with open("node_zone.dat", "w") as file:
    # {
        for idx, row in nearest_zone.iterrows():
            node_id = row['NodeID']
            file.write(f"{node_id} {node_zone[node_id]}\n")
    # }
    with open("zone_mapping.dat", "w") as file:
    # {
        for zone, idx in zone_to_idx.items():
            file.write(f"{zone}: {idx}\n")
    # }

    # -----------------------------
    # 8. WRITE ZONE FILE
    # -----------------------------

    import pyproj
    transformer = pyproj.Transformer.from_crs(7856, 7844, always_xy=True)
    with open("zone.dat", "w") as file:
    # {
        file.write(" This file defines zone regions\n")
        file.write(" number of feature points, number of zones\n")
        file.write(f"     {len(unique_points_list)}       {nb_zones}\n")
        file.write("node #, x-coordinate, y-coordinate")

        for i, (x,y) in enumerate(unique_points_list, start=1):
            #X,Y = transformer.transform(x,y)
            #X_,Y_ = lonlat_to_xy(x,y,params)
            file.write(f"\n      {i}        {x},        {y}")

        file.write("\nzone #, number of nodes, node #'s")

        for zone_code, z_idx in zone_to_idx.items():
            point_ids = zone_to_point_idx[zone_code]
            file.write(f"\n      {z_idx}       {len(point_ids)}:")
            for pid in point_ids:
                file.write(f"       {pid},")
    # }

