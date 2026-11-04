"""
25_road_geometry_features.py
Derive crash-prone road-geometry / network features from the OSM extract and
(optionally) an Overture transportation release, then merge them into the
EB dataset so the SPFs in 19/20/21 can use them as covariates.

Why this exists
---------------
Every SPF in the pipeline is currently fit on exposure only:
    total_hb ~ np.log(exposure_trips + 1)        # 19_eb_analysis.py
    total_risk ~ np.log(trips + 1)               # 20_eb_adjusted_street_risk.py
    Y ~ np.log(trips + 1)  (+ C(highway))        # 21_eb_timewise_risk.py
The model answers "which segment is hot" but never "why". This script adds the
road-inventory / geometry covariates that the safety literature (HSM SPFs) and
crash-mechanism intuition say drive risk:
    curvature / sinuosity        - sharp bends
    intersection density / node degree / dist-to-intersection
    lane count, maxspeed, oneway, road class, link flag
    connectivity (betweenness-ish funnel indicator)

Outputs
-------
  data/road_geometry_features.csv   one row per road_osmid + all features
  data/near_miss_hb_ecf.csv        re-written WITH the new columns merged in
                                   (so 19/20/21 can reference them directly)

Usage
-----
  python 25_road_geometry_features.py

Optional Overture enrichment (Polaris-style network build)
-------------------------------------------------------
Download the Overture transportation theme for the study-area bbox, e.g.
    overturemaps download -f geoparquet -t transportation \
        --bbox <minlon>,<minlat>,<maxlon>,<maxlat> -o overture_transport.parquet
then point OVERTURE_PARQUET at that file. Following the Polaris supply-build
pattern (OSM -> GMNS -> Polaris: bidirectional links, per-direction lanes and
speeds, KNN imputation of missing lanes/speeds, minimum one lane per active
direction, link length from projected geometry, snapped node topology), the
Overture join extracts what OSM tags do poorly:
    width_rules        -> ov_width_m (edge-to-edge carriageway width, m)
    speed_limits       -> ov_speed_fwd / ov_speed_bwd (km/h, mph converted)
    road_surface       -> paved / unpaved flags
    road_flags         -> ov_is_link / is_tunnel / is_bridge / is_toll flags
    class / subclass   -> ov_class / ov_subclass (+ ov_class_rank)
    routes             -> has_ov_ref (numbered shield present)
    connectors         -> n_ov_connectors, ov_mid_connector (routing decision
                          points strictly inside the segment); connector-shared
                          endpoints refine the intersection-distance feature.
Missing lanes/speeds are KNN-imputed within road class (Polaris-style;
class-median fallback), with a minimum-one-lane rule on trafficked roads.
If the file is absent the script runs on OSM alone.

Requires: geopandas, shapely, numpy, pandas, scipy (all in zigenv).
"""
import os
import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

PROJ = os.environ.get(
    "PROJ",
    "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics",
)
DATA = os.path.join(PROJ, "data")

ROADS_GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
NEAR_MISS = os.path.join(DATA, "near_miss_hb_ecf.csv")
OUT_FEATURES = os.path.join(DATA, "road_geometry_features.csv")

# Optional Overture transportation extract (segment + connector).
OVERTURE_PARQUET = os.path.join(DATA, "overture_transportation.parquet")

# Brisbane is covered by GDA2020 / MGA zone 56 (EPSG:7855). Project to planar
# for accurate length, curvature and distance computations.
PLANAR_CRS = "EPSG:7855"
NODE_TOL_M = 15.0          # endpoints within this distance are the same node
ANGLE_TOL_DEG = 25.0       # link/segment considered "straight" below this turn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_line(geom):
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        return unary_union(geom)
    return None


def _parse_speed(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if not s:
        return np.nan
    try:
        return float(s.split()[0])          # "50", "50 mph" -> 50
    except ValueError:
        return np.nan


def _parse_lanes(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return np.nan
    try:
        return float(s.split()[0])          # "2", "2;1" -> 2
    except ValueError:
        return np.nan


def _turn_angle_deg(p0, p1, p2):
    """Absolute exterior turn angle (deg) at p1 between segments p0-p1 and p1-p2."""
    v1 = (p1[0] - p0[0], p1[1] - p0[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _curvature(geom_planar):
    """Total absolute turn (deg) per km of length. 0 for straight segments."""
    line = _to_line(geom_planar)
    if line is None or len(line.coords) < 3:
        return 0.0, 0.0
    coords = list(line.coords)
    total_turn = 0.0
    for i in range(1, len(coords) - 1):
        total_turn += _turn_angle_deg(coords[i - 1], coords[i], coords[i + 1])
    length_m = line.length
    if length_m < 1e-6:
        return 0.0, 0.0
    # sinuosity = path length / straight-line distance between endpoints
    sl_dist = math.hypot(coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
    sinuosity = length_m / sl_dist if sl_dist > 1e-6 else 1.0
    return total_turn / (length_m / 1000.0), sinuosity   # deg/km, unitless


ROAD_CLASS_RANK = {
    "motorway": 8, "trunk": 7, "primary": 6, "secondary": 5, "tertiary": 4,
    "unclassified": 3, "residential": 2, "service": 1, "living_street": 1,
    "track": 0, "path": 0, "footway": 0, "cycleway": 0, "pedestrian": 0,
}


def _class_rank(hw):
    if hw is None:
        return 0
    return ROAD_CLASS_RANK.get(str(hw).split(":")[0].strip().lower(), 0)


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [x for x in list(v)
                if not (isinstance(x, float) and np.isnan(x))]
    return [v]


def _rule_value(rule, key="value"):
    if isinstance(rule, dict):
        return rule.get(key)
    return rule


def _to_kmh(value, unit):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    u = str(unit or "km/h").lower()
    if "mph" in u or "mi" in u:
        return v * 1.60934
    return v


def _parse_overture_segment(sub):
    """Aggregate one OSM segment's intersecting Overture rows (Polaris-style).

    Returns width (m), directional speed limits (km/h), surface/flag dummies,
    finer class rank, route-shield presence, and connector counts. Overture
    struct columns arrive as dicts/lists; every accessor is defensive so a
    schema change degrades to NaN/0 rather than crashing the build.
    """
    widths, sp_fwd, sp_bwd = [], [], []
    flags, surfaces, classes, subs, refs = [], [], [], [], []
    n_conn, mid_conn = [], []
    for _, r in sub.iterrows():
        get = r.get if hasattr(r, "get") else lambda k, d=None: r[k]
        for rule in _as_list(get("width_rules")):
            w = _rule_value(rule)
            try:
                w = float(w)
                if np.isfinite(w):
                    widths.append(w)
            except (TypeError, ValueError):
                pass
        for rule in _as_list(get("speed_limits")):
            if not isinstance(rule, dict):
                continue
            mx = rule.get("max_speed") or {}
            v = _to_kmh(mx.get("value"), mx.get("unit"))
            if not np.isfinite(v):
                continue
            when = rule.get("when") or {}
            head = str(when.get("heading") or "").lower()
            if "backward" in head or head in ("back", "bwd"):
                sp_bwd.append(v)
            else:  # forward / both / unspecified
                sp_fwd.append(v)
        for rule in _as_list(get("road_surface")):
            s = str(_rule_value(rule) or "").lower()
            if s:
                surfaces.append(s)
        for rule in _as_list(get("road_flags")):
            if isinstance(rule, dict):
                vals = rule.get("values", rule.get("value", []))
            else:
                vals = rule
            for f in _as_list(vals):
                flags.append(str(f).lower())
        cl = get("class")
        if cl is not None and str(cl).strip():
            classes.append(str(cl).strip().lower())
        sc = get("subclass")
        if sc is not None and str(sc).strip():
            subs.append(str(sc).strip().lower())
        for rt in _as_list(get("routes")):
            ref = (rt.get("ref") if isinstance(rt, dict) else rt)
            if ref is not None and str(ref).strip():
                refs.append(str(ref).strip())
        conns = _as_list(get("connectors"))
        n_conn.append(len(conns))
        mid = 0
        for cn in conns:
            try:
                at = float(cn.get("at")) if isinstance(cn, dict) else np.nan
                if 0.0 < at < 1.0:
                    mid += 1
            except (TypeError, ValueError):
                pass
        mid_conn.append(mid)
    fl = set(flags)
    su = set(surfaces)
    rec = {
        "ov_width_m": float(np.median(widths)) if widths else np.nan,
        "ov_speed_fwd": float(np.max(sp_fwd)) if sp_fwd else np.nan,
        "ov_speed_bwd": float(np.max(sp_bwd)) if sp_bwd else np.nan,
        "paved": float(any("pav" in s for s in su)) if su else np.nan,
        "unpaved": float(any("unpav" in s or s in ("gravel", "dirt")
                             for s in su)) if su else 0.0,
        "ov_is_link": float(any("link" in f for f in fl)),
        "is_tunnel": float(any("tunnel" in f for f in fl)),
        "is_bridge": float(any("bridge" in f for f in fl)),
        "is_toll": float(any("toll" in f for f in fl)),
        "ov_class": max(set(classes), key=classes.count) if classes else None,
        "ov_subclass": max(set(subs), key=subs.count) if subs else None,
        "has_ov_ref": float(len(refs) > 0),
        "n_ov_connectors": int(np.max(n_conn)) if n_conn else 0,
        "mid_conn": int(np.max(mid_conn)) if mid_conn else 0,
    }
    rec["ov_class_rank"] = _class_rank(rec["ov_class"])
    return rec


def _knn_impute_lanes_speed(feat):
    """Polaris-style imputation: KNN within road class for lanes/speed.

    Neighbours are same-class segments in (log length, curvature) space
    (k=5); class-median then global-median fallbacks; minimum-one-lane rule
    on trafficked roads (Polaris sets min 1 lane per active direction).
    Operates on a copy; adds ov_speed-derived maxspeed fill first.
    """
    out = feat.copy()
    # Overture directional speeds back-fill OSM maxspeed where OSM is silent.
    if "ov_speed_fwd" in out.columns:
        fill = pd.to_numeric(out["ov_speed_fwd"], errors="coerce")
        out["maxspeed_kmh"] = pd.to_numeric(out["maxspeed_kmh"],
                                            errors="coerce").fillna(fill)
    for col in ["lanes", "maxspeed_kmh"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    try:
        from sklearn.neighbors import KNeighborsRegressor
        _has_knn = True
    except Exception:
        _has_knn = False
    X = pd.DataFrame({
        "cls": pd.to_numeric(out.get("road_class_rank"),
                             errors="coerce").fillna(0),
        "loglen": np.log(pd.to_numeric(out.get("length_m"),
                                       errors="coerce").fillna(100.0) + 1.0),
        "curv": pd.to_numeric(out.get("curvature_deg_km"),
                              errors="coerce").fillna(0.0),
    }).to_numpy(dtype=float)
    for col in ["lanes", "maxspeed_kmh"]:
        y = out[col].to_numpy(dtype=float)
        miss = ~np.isfinite(y)
        if not miss.any():
            continue
        done = False
        if _has_knn and np.isfinite(y).sum() >= 6:
            try:
                knn = KNeighborsRegressor(n_neighbors=min(5, int(np.isfinite(y).sum())),
                                          weights="distance")
                knn.fit(X[np.isfinite(y)], y[np.isfinite(y)])
                y[miss] = knn.predict(X[miss])
                done = True
            except Exception:
                done = False
        if not done:
            cls = pd.to_numeric(out.get("road_class_rank"),
                                errors="coerce").fillna(-1)
            y[miss] = cls[miss].map(
                pd.Series(y).groupby(cls.values).median()).fillna(
                    float(np.nanmedian(y)) if np.isfinite(y).any() else np.nan)
        out[col] = y
    # minimum-one-lane rule on roads carrying trips/geometry (Polaris).
    ln = pd.to_numeric(out["lanes"], errors="coerce")
    out["lanes"] = ln.fillna(1.0).clip(lower=1.0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("25_road_geometry_features.py - derive crash-prone geometry features")
    print("=" * 80)

    # ---- 1. Load OSM roads, project to planar ------------------------------
    print("\n1. Loading OSM roads and projecting to", PLANAR_CRS, "...")
    g = gpd.read_file(ROADS_GEOJSON)
    g = g.to_crs(PLANAR_CRS)
    g = g.reset_index(drop=True)
    print(f"   {len(g):,} segments")

    # ---- 2. Per-segment geometry / inventory features ----------------------
    print("\n2. Computing curvature, length, inventory covariates...")
    rows = []
    endpoints = []          # (node_key, seg_idx) for network topology
    for idx, r in g.iterrows():
        line = _to_line(r.geometry)
        length_m = line.length if line is not None else 0.0
        curv_deg_km, sinu = _curvature(line) if line is not None else (0.0, 1.0)

        hw = r.get("highway")
        is_link = bool(hw) and str(hw).lower().endswith("link")
        oneway = r.get("oneway")
        oneway_flag = 1 if str(oneway).lower() in ("yes", "true", "1") else 0

        rows.append({
            "road_osmid": r.get("osmid"),
            "length_m": length_m,
            "curvature_deg_km": curv_deg_km,
            "sinuosity": sinu,
            "maxspeed_kmh": _parse_speed(r.get("maxspeed")),
            "lanes": _parse_lanes(r.get("lanes")),
            "oneway": oneway_flag,
            "is_link": int(is_link),
            "road_class_rank": _class_rank(hw),
            "highway": hw,
        })

        # collect endpoints for topology
        if line is not None and len(line.coords) >= 2:
            a = line.coords[0]
            b = line.coords[-1]
            endpoints.append((round(a[0], 1), round(a[1], 1), idx))
            endpoints.append((round(b[0], 1), round(b[1], 1), idx))

    feat = pd.DataFrame(rows)
    print(f"   curvature range: {feat['curvature_deg_km'].min():.1f}.."
          f"{feat['curvature_deg_km'].max():.1f} deg/km")

    # ---- 3. Network topology: node degree & intersection proximity ---------
    print("\n3. Building network topology (nodes, degree, intersection dist)...")
    ep = pd.DataFrame(endpoints, columns=["x", "y", "seg_idx"])
    # snap endpoints to a node grid within NODE_TOL_M
    tree = cKDTree(ep[["x", "y"]].values)
    pairs = tree.query_pairs(r=NODE_TOL_M)
    parent = {i: i for i in range(len(ep))}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in pairs:
        union(a, b)
    ep["node"] = ep.index.map(find)

    # node degree = how many segment-endpoints touch each node
    node_deg = ep.groupby("node")["seg_idx"].nunique()
    seg_deg = ep.groupby("seg_idx")["node"].apply(
        lambda ns: max(node_deg[n] for n in ns))
    feat["node_degree"] = seg_deg.reindex(feat.index).fillna(1).astype(int)

    # distance to nearest intersection (node with degree >= 3)
    inter_nodes = node_deg[node_deg >= 3].index
    if len(inter_nodes) > 0:
        inter_pts = ep[ep["node"].isin(inter_nodes)][["x", "y"]].drop_duplicates()
        itree = cKDTree(inter_pts.values)
        seg_pts = g.geometry.apply(
            lambda gm: _to_line(gm).centroid.coords[0] if _to_line(gm) else None)
        seg_xy = np.array([p for p in seg_pts if p])
        _, dist = itree.query(seg_xy, k=1)
        dist_to_inter = np.full(len(feat), np.nan)
        valid = [i for i, p in enumerate(seg_pts) if p]
        for k, i in enumerate(valid):
            dist_to_inter[i] = dist[k]
        feat["dist_to_intersection_m"] = dist_to_inter
    else:
        feat["dist_to_intersection_m"] = np.nan
    print(f"   intersections (deg>=3): {len(inter_nodes):,}")

    # ---- 4. Optional Overture enrichment (Polaris-style) ----------------------
    ov_new = {}
    if os.path.exists(OVERTURE_PARQUET):
        print("\n4. Enriching from Overture transportation extract (Polaris-style)...")
        try:
            ov = gpd.read_parquet(OVERTURE_PARQUET)
            ov = ov[ov.geometry.notna()].to_crs(PLANAR_CRS)
            if "subtype" in ov.columns:
                ov = ov[ov["subtype"].isin(
                    ["road", "link", "service", "connector"])]
            ov = ov.reset_index(drop=True)
            gjoin = gpd.sjoin(g, ov, how="left", predicate="intersects")
            agg = {}
            for seg_idx, grp in gjoin.groupby("index_left"):
                sub = ov.loc[grp["index_right"].dropna().astype(int).unique()]
                if len(sub) == 0:
                    continue
                rec = _parse_overture_segment(sub)
                # connector-shared endpoints -> true intersection topology:
                # a segment whose endpoint connector is shared with 2+ other
                # Overture segments sits at a real junction.
                rec["ov_mid_connector"] = int(rec.pop("mid_conn", 0) > 0)
                agg[seg_idx] = rec
            if agg:
                ovdf = pd.DataFrame.from_dict(agg, orient="index")
                for c in ovdf.columns:
                    feat[c] = ovdf[c].reindex(feat.index)
                ov_new = {c: feat[c].notna().sum() for c in ovdf.columns}
            print(f"   joined Overture for {len(agg):,} segments: {ov_new}")
        except Exception as e:
            print(f"   Overture join failed ({e}); continuing with OSM only")
    else:
        print("\n4. No Overture file found -> using OSM-derived features only.")
        print(f"   (optional) place Overture parquet at: {OVERTURE_PARQUET}")
        print("   (fetch: overturemaps download -f geoparquet -t transportation "
              "--bbox 152.98,-27.49,153.05,-27.44 -o overture_transportation.parquet)")

    # ---- 4b. Polaris-style KNN imputation of lanes/speed ---------------------
    # Polaris imputes missing lanes/speeds (knn over neighbouring links);
    # here neighbours are same-class segments in (class, length, curvature)
    # space, with a minimum-one-lane rule on trafficked roads.
    feat = _knn_impute_lanes_speed(feat)
    print("   lanes/speed after KNN imputation: "
          f"lanes NaN={int(feat['lanes'].isna().sum())}, "
          f"maxspeed NaN={int(feat['maxspeed_kmh'].isna().sum())}")

    # ---- 5. Save features, then merge into the EB dataset ------------------
    print("\n5. Saving features and merging into near_miss_hb_ecf.csv...")
    feat_out = feat.drop(columns=["highway"], errors="ignore")
    feat_out.to_csv(OUT_FEATURES, index=False)
    print(f"   wrote {OUT_FEATURES} ({len(feat_out.columns)} columns)")

    if os.path.exists(NEAR_MISS):
        nm = pd.read_csv(NEAR_MISS)
        # Idempotent: only merge geometry columns not already present, so
        # re-running 25 (or replaying the pipeline) never clobbers the table.
        geo_new = [c for c in feat_out.columns
                   if c != "road_osmid" and c not in nm.columns]
        if geo_new:
            merge_cols = ["road_osmid"] + geo_new
            nm = nm.merge(feat_out[merge_cols], on="road_osmid", how="left")
            nm.to_csv(NEAR_MISS, index=False)
            print(f"   merged {len(geo_new)} new feature columns into {NEAR_MISS}")
            print(f"   columns now: {len(nm.columns)}")
        else:
            print(f"   near_miss already has geometry features "
                  f"({len([c for c in feat_out.columns if c!='road_osmid'])}); "
                  f"skipping merge (idempotent)")

    # ---- 6. Suggested SPF formulas for 19 / 20 / 21 ------------------------
    print("\n6. Suggested SPF covariate blocks (add to 19/20/21):")
    print("    base  : ~ np.log(exposure_trips + 1)")
    print("    +inv  : + C(highway) + maxspeed_kmh + lanes + oneway")
    print("    +geo  : + curvature_deg_km + sinuosity + dist_to_intersection_m + node_degree")
    print("    +over : + ov_width_m + ov_speed_fwd + paved + is_tunnel + is_bridge")
    print("            + is_toll + n_ov_connectors + ov_mid_connector + ov_class_rank")
    print("            (Overture columns; script 47 searches them automatically)")
    print("    Use formula='total_hb ~ np.log(exposure_trips+1) + C(highway)"
          " + maxspeed_kmh + lanes + curvature_deg_km + dist_to_intersection_m'")

    print("\n" + "=" * 80)
    print("DONE! Road geometry features ready for the EB SPFs.")
    print("=" * 80)


if __name__ == "__main__":
    main()
