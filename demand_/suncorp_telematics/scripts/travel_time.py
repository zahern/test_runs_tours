"""
travel_time.py - network-based travel times for deployment MIPs.

Replaces straight-line haversine / uniform-speed travel assumptions with
shortest-path times over the OSM road graph using per-link speeds, so the
MIP only permits relocations (and captures) that are actually feasible in
time. Only numpy/scipy are required; geopandas is imported lazily and only
by the geojson convenience loader.

Speed model per directed edge (documented, no hidden tuning):
    v_edge = posted_maxspeed_kmh * congest_factor,
falling back to a highway-class default (also scaled by congest_factor)
where the posted limit is missing. congest_factor < 1 models
event-congested driving; the default 0.5 halves free speeds.

Public API:
    post_travel_matrix(...)   post-to-post minutes + bucket offsets
    segment_post_times(...)   segment-to-post minutes (capture feasibility)
    bucket_offsets(...)       minutes -> MIP bucket offsets + feasible mask
"""
import heapq

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except Exception:  # noqa: BLE001
    _HAS_KDTREE = False

# Fallback free speeds by OSM highway class (km/h) where posted limits miss.
HIGHWAY_SPEEDS = {
    "motorway": 80.0, "trunk": 70.0, "primary": 60.0, "secondary": 60.0,
    "tertiary": 50.0, "unclassified": 50.0, "residential": 50.0,
    "service": 30.0, "living_street": 20.0,
}
for _lnk in ("motorway_link", "trunk_link", "primary_link", "secondary_link",
             "tertiary_link"):
    HIGHWAY_SPEEDS[_lnk] = 50.0
DEFAULT_SPEED_KMH = 40.0


def edge_speed_kmh(maxspeed_kmh=None, highway=None,
                   default_speed_kmh=DEFAULT_SPEED_KMH, congest_factor=0.5):
    """Per-edge free speed scaled for event congestion (km/h, floor 5)."""
    try:
        v = float(maxspeed_kmh)
    except (TypeError, ValueError):
        v = float("nan")
    if not np.isfinite(v) or v <= 0:
        v = HIGHWAY_SPEEDS.get(str(highway or "").strip().lower(),
                               float(default_speed_kmh))
    return max(5.0, float(v) * float(congest_factor))


def build_time_graph(node_xy, edges):
    """Adjacency map for Dijkstra.

    node_xy: (N,2) array of node coordinates (any planar CRS, metres).
    edges: iterable of (u, v, minutes) undirected links.
    Returns dict {u: [(v, minutes), ...]}.
    """
    adj = {i: [] for i in range(len(node_xy))}
    for u, v, mins in edges:
        u, v = int(u), int(v)
        w = float(mins)
        if not np.isfinite(w) or w < 0:
            continue
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def _dijkstra(adj, source):
    dist = {source: 0.0}
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")) - 1e-12:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _snap(pxy, node_xy):
    if not _HAS_KDTREE:
        raise ImportError("scipy.spatial.cKDTree required for snapping")
    _, ii = cKDTree(np.asarray(node_xy, dtype=float)).query(
        np.asarray(pxy, dtype=float))
    return np.asarray(ii, dtype=int)


def travel_matrix(post_xy, node_xy, adj, fallback_speed_kmh=12.0):
    """All-pairs shortest-path minutes between posts.

    Posts snap to their nearest graph node; disconnected pairs fall back to
    straight-line distance at fallback_speed_kmh (recorded in the audit by
    the caller via the returned fallback mask).
    Returns (T_min (J,J), fallback_mask (J,J bool)).
    """
    post_xy = np.asarray(post_xy, dtype=float)
    J = len(post_xy)
    nodes = _snap(post_xy, node_xy)
    T = np.full((J, J), np.inf)
    for i in range(J):
        d = _dijkstra(adj, int(nodes[i]))
        for j in range(J):
            T[i, j] = d.get(int(nodes[j]), float("inf"))
    T = np.maximum(T, 0.0)
    fb = ~np.isfinite(T)
    if fb.any():
        dh = np.sqrt(((post_xy[:, None, :] - post_xy[None, :, :]) ** 2).sum(-1))
        T[fb] = dh[fb] / 1000.0 / max(float(fallback_speed_kmh), 1e-6) * 60.0
    np.fill_diagonal(T, 0.0)
    fb[np.diag_indices(J)] = False
    return T, fb


def segment_post_times(seg_xy, post_xy, node_xy, adj,
                       fallback_speed_kmh=12.0):
    """Minutes from each segment to each post (capture-feasibility check).

    Returns (T_min (n_seg, J), fallback_mask). An officer stationed at post
    j can only capture segment i's events within a bucket of length B if
    T_min[i, j] <= B (+ setup); callers enforce this per bucket.
    """
    seg_xy = np.asarray(seg_xy, dtype=float)
    post_xy = np.asarray(post_xy, dtype=float)
    sn = _snap(seg_xy, node_xy)
    pn = _snap(post_xy, node_xy)
    # Dijkstra from each post node (J is small); segment time = dist at its node.
    T = np.full((len(seg_xy), len(post_xy)), np.inf)
    for j in range(len(post_xy)):
        d = _dijkstra(adj, int(pn[j]))
        for i in range(len(seg_xy)):
            T[i, j] = d.get(int(sn[i]), float("inf"))
    fb = ~np.isfinite(T)
    if fb.any():
        dh = np.sqrt(((seg_xy[:, None, :] - post_xy[None, :, :]) ** 2).sum(-1))
        T[fb] = dh[fb] / 1000.0 / max(float(fallback_speed_kmh), 1e-6) * 60.0
    return np.maximum(T, 0.0), fb


def bucket_offsets(T_min, setup_min=5.0, bucket_min=15.0, max_move_min=30.0):
    """Minutes matrix -> integer MIP bucket offsets + feasibility mask.

    Same convention as the paper (eq:tau): hold = 1 bucket, genuine moves
    cost >= 2 buckets; moves needing more than max_move_min are infeasible.
    Returns (tau (J,J) int, feasible (J,J) bool).
    """
    T = np.asarray(T_min, dtype=float)
    J = T.shape[0]
    tau = np.full((J, J), 10 ** 6, dtype=int)
    feasible = np.zeros((J, J), dtype=bool)
    for i in range(J):
        for j in range(J):
            if i == j:
                tau[i, j] = 1
                feasible[i, j] = True
            elif np.isfinite(T[i, j]) and T[i, j] <= float(max_move_min):
                tau[i, j] = max(
                    2, int(np.ceil((float(setup_min) + T[i, j])
                                   / float(bucket_min))))
                feasible[i, j] = True
    return tau, feasible


def graph_from_segments(seg_ids, x0, y0, x1, y1, length_m,
                        maxspeed_kmh=None, highway=None,
                        default_speed_kmh=DEFAULT_SPEED_KMH,
                        congest_factor=0.5, snap_m=1.0):
    """Build a time-weighted undirected graph from segment endpoints.

    (x0,y0)/(x1,y1): endpoint coordinates in a metric CRS; length_m: link
    lengths; maxspeed_kmh/highway optional per-link (lists/Series or None).
    Endpoints snap to a snap_m metre grid (same convention as script 28).
    Returns (node_xy (N,2), adj, osmid_per_link not tracked).
    """
    node_id = {}
    node_xy = []

    def _node(px, py):
        key = (int(round(px / snap_m)), int(round(py / snap_m)))
        if key not in node_id:
            node_id[key] = len(node_xy)
            node_xy.append((px, py))
        return node_id[key]

    edges = []
    n = len(seg_ids)
    for k in range(n):
        u = _node(float(x0[k]), float(y0[k]))
        v = _node(float(x1[k]), float(y1[k]))
        if u == v:
            continue
        L = float(length_m[k]) if np.isfinite(float(length_m[k])) else 0.0
        if L <= 0:
            continue
        ms = (maxspeed_kmh[k] if maxspeed_kmh is not None else None)
        hw = (highway[k] if highway is not None else None)
        vkmh = edge_speed_kmh(ms, hw, default_speed_kmh, congest_factor)
        edges.append((u, v, L / 1000.0 / vkmh * 60.0))
    return np.asarray(node_xy, dtype=float), build_time_graph(
        np.asarray(node_xy, dtype=float), edges)


def graph_from_gdf(gu, speed_col="maxspeed_kmh", highway_col="highway",
                   default_speed_kmh=DEFAULT_SPEED_KMH, congest_factor=0.5,
                   snap_m=1.0):
    """Time graph from a projected (metric CRS) segment GeoDataFrame.

    Uses LineString endpoint coordinates; non-LineString rows are skipped.
    Returns (node_xy, adj, info dict with node/edge counts).
    """
    xs0, ys0, xs1, ys1, lens, spds, hws = [], [], [], [], [], [], []
    for _, r in gu.iterrows():
        geom = r.geometry
        if geom is None or getattr(geom, "geom_type", "") != "LineString":
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        (ax, ay), (bx, by) = coords[0], coords[-1]
        try:
            L = float(geom.length)
        except Exception:  # noqa: BLE001
            continue
        xs0.append(ax)
        ys0.append(ay)
        xs1.append(bx)
        ys1.append(by)
        lens.append(L)
        spds.append(r.get(speed_col, None) if hasattr(r, "get") else None)
        hws.append(r.get(highway_col, None) if hasattr(r, "get") else None)
    node_xy, adj = graph_from_segments(
        range(len(xs0)), np.asarray(xs0), np.asarray(ys0),
        np.asarray(xs1), np.asarray(ys1), np.asarray(lens),
        maxspeed_kmh=spds or None, highway=hws or None,
        default_speed_kmh=default_speed_kmh,
        congest_factor=congest_factor, snap_m=snap_m)
    info = {"n_nodes": int(len(node_xy)),
            "n_links": int(sum(len(v) for v in adj.values()) // 2),
            "congest_factor": float(congest_factor)}
    return node_xy, adj, info


def lonlat_to_utm(lon, lat, epsg=28356):
    """Project lon/lat vectors to a metric CRS (requires geopandas)."""
    import geopandas as gpd
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(np.asarray(lon, dtype=float),
                                    np.asarray(lat, dtype=float)),
        crs=4326).to_crs(epsg)
    return np.column_stack([pts.geometry.x.values, pts.geometry.y.values])
