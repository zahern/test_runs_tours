"""
21_build_osm_walk_bike_skims.py
================================
Build a REAL OSM pedestrian-network walk distance skim over the full 371-zone
SA2 system, and derive a distinct bike distance as walk x a cycling detour
factor. Replaces the EMME-derived DISTWALK/DISTBIKE (which were identical to
each other and only a 0.9 scalar off auto DIST) with genuine street-network
walk distances, and gives walk != bike for the first time.

Why this design
---------------
* Active-mode distance only matters at SHORT range: walk trips are viable to
  ~2-3 km, bike to ~10 km. Beyond that the OD pair is infeasible for active
  modes regardless of exact distance, so routing every one of the 371^2 =
  137,641 pairs over a metropolitan pedestrian graph (the study area spans
  ~176 km x 313 km, Toowoomba->coast, Tweed->Sunshine Coast) is wasted work.
  We route only pairs within CUTOFF_KM great-circle of each origin and fill the
  rest with great_circle * NONDIRECT_FILL (flagged non-viable for active modes).
* Walk vs bike DISTANCE differs only where the permitted network differs.
  Rather than route a second (bicycle) network, we apply a calibrated cycling
  detour factor to the pedestrian distance (BIKE_DETOUR, default 1.10): cyclists
  route around pedestrian-only shortcuts / plazas onto the road network, so bike
  distance is modestly longer than the pedestrian shortest path. This is the
  "walk x detour factor" approach; swap in a real bicycle-network route here if a
  bike graph becomes available.

Inputs
------
  ../demand_/sa2_zones_reference_357.csv   371 SA2 zones: sa2_code,
                                           centroid_lon, centroid_lat,
                                           in_emme_network
Output
------
  ../demand_/walk_bike_osm_sa2_371.omx     371x371 OMX (h5py-written, matching
                                           the demand_ OMX convention read by
                                           read_skims._omx_to_dataframe_h5py):
                                             attrs: OMX_VERSION, SHAPE
                                             data/DISTWALK  float32 (km)
                                             data/DISTBIKE  float32 (km)
                                             lookup/SA2_CODE  |S20 (zone ids,
                                               sorted ascending == ORDER order
                                               of saved_data/seq_zone_ids_371.csv)
  ../demand_/walk_bike_osm_sa2_371_coverage.csv   per-zone routing coverage /
                                                  quality flags.

Run
---
  Needs OSM egress (osmnx/Overpass) + a machine with RAM for ~CUTOFF_KM-radius
  pedestrian graphs. Heavy: expect hours for the full 371 origins. Resumable via
  the on-disk graph cache (GRAPH_CACHE_DIR) and the partial-matrix checkpoint.

  python 21_build_osm_walk_bike_skims.py \
      [--cutoff-km 20] [--bike-detour 1.10] [--fill-nondirect 1.30] \
      [--out ../demand_/walk_bike_osm_sa2_371.omx] [--limit N]
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

DEMAND_DIR = os.path.dirname(os.path.abspath(__file__))
ZONE_REF = os.path.join(DEMAND_DIR, "sa2_zones_reference_357.csv")
OUT_OMX_DEFAULT = os.path.join(DEMAND_DIR, "walk_bike_osm_sa2_371.omx")
GRAPH_CACHE_DIR = os.path.join(DEMAND_DIR, "_osm_walk_graph_cache")

EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km); vectorised over the destination arrays."""
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + math.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_zones():
    ref = pd.read_csv(ZONE_REF)
    ref = ref.rename(columns={"centroid_lon": "lon", "centroid_lat": "lat"})
    # Sort by SA2 code ascending == the OMX column order == ORDER in
    # saved_data/seq_zone_ids_371.csv, so this skim aligns with the 507 skims
    # and the migration reference without any re-indexing downstream.
    ref = ref.sort_values("sa2_code").reset_index(drop=True)
    need = {"sa2_code", "lon", "lat"}
    missing = need - set(ref.columns)
    if missing:
        raise ValueError(f"{ZONE_REF} missing columns: {missing}")
    return ref[["sa2_code", "lon", "lat", "in_emme_network"]].copy()


def build(args):
    # osmnx / networkx imported lazily so --help and dry inspection work without
    # the geo stack, and so the (heavy) import cost is only paid on a real run.
    try:
        import osmnx as ox
        import networkx as nx
    except Exception as e:  # pragma: no cover - env-dependent
        print(f"FATAL: osmnx/networkx required for the routing build: {e!r}")
        sys.exit(2)

    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(GRAPH_CACHE_DIR, "_ox_http_cache")

    zones = load_zones()
    n = len(zones)
    if args.limit:
        print(f"[limit] routing only the first {args.limit} of {n} origins "
              f"(smoke/partial run)")
    lat = zones["lat"].to_numpy()
    lon = zones["lon"].to_numpy()
    codes = zones["sa2_code"].to_numpy(dtype=np.int64)

    cutoff_m = args.cutoff_km * 1000.0
    # Distance matrices in km; np.nan == not yet routed.
    walk = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(walk, 0.0)

    coverage = []  # per-origin routing diagnostics

    n_origins = args.limit or n
    for i in range(n_origins):
        # Destinations worth routing: within cutoff great-circle of this origin.
        gc = haversine_km(lat[i], lon[i], lat, lon)
        in_range = np.where(gc <= args.cutoff_km)[0]

        graph_ok = False
        n_routed = 0
        try:
            # ~CUTOFF-radius pedestrian graph around the origin centroid.
            G = ox.graph_from_point((lat[i], lon[i]), dist=cutoff_m,
                                    network_type="walk", simplify=True)
            # osmnx tags each edge with 'length' (metres) on download.
            orig_node = ox.distance.nearest_nodes(G, lon[i], lat[i])
            # Single-source shortest path lengths (metres) capped at cutoff.
            lengths = nx.single_source_dijkstra_path_length(
                G, orig_node, cutoff=cutoff_m, weight="length")
            for j in in_range:
                if j == i:
                    continue
                dnode = ox.distance.nearest_nodes(G, lon[j], lat[j])
                d_m = lengths.get(dnode)
                if d_m is not None and np.isfinite(d_m):
                    walk[i, j] = d_m / 1000.0
                    n_routed += 1
            graph_ok = True
        except Exception as e:
            print(f"  [origin {i} sa2={codes[i]}] graph/route failed: {e!r}")

        # Fill any in-range pair the graph could not reach, plus every
        # out-of-range pair, with great_circle * NONDIRECT_FILL.
        unrouted = np.isnan(walk[i])
        walk[i, unrouted] = gc[unrouted] * args.fill_nondirect
        walk[i, i] = 0.0

        coverage.append({
            "sa2_code": int(codes[i]),
            "in_emme_network": bool(zones["in_emme_network"].iloc[i]),
            "graph_ok": graph_ok,
            "n_in_range": int(len(in_range) - 1),
            "n_routed": int(n_routed),
            "route_share": round(n_routed / max(len(in_range) - 1, 1), 3),
        })
        if (i + 1) % 10 == 0 or i == n_origins - 1:
            print(f"  routed {i + 1}/{n_origins} origins "
                  f"(last: sa2={codes[i]} routed={n_routed}/{len(in_range) - 1})")

    # Symmetrise: walk distance is direction-agnostic; where only one direction
    # was routed (origin i not in the --limit slice), fill from the transpose.
    both = ~np.isnan(walk) & ~np.isnan(walk.T)
    walk_sym = np.where(both, np.fmin(walk, walk.T), np.where(np.isnan(walk), walk.T, walk))
    walk = np.where(np.isnan(walk_sym), gc_fallback(lat, lon, args.fill_nondirect), walk_sym)

    bike = walk * args.bike_detour

    write_omx(args.out, codes, walk.astype(np.float32), bike.astype(np.float32))
    cov = pd.DataFrame(coverage)
    cov_path = os.path.splitext(args.out)[0] + "_coverage.csv"
    cov.to_csv(cov_path, index=False)

    print(f"\nWrote {args.out}")
    print(f"  DISTWALK mean={np.nanmean(walk):.3f} km  DISTBIKE mean={np.nanmean(bike):.3f} km")
    print(f"  walk!=bike: max_abs_diff={np.nanmax(np.abs(bike - walk)):.4f} km "
          f"(detour factor {args.bike_detour})")
    print(f"  coverage -> {cov_path}  "
          f"(mean routed-share {cov['route_share'].mean():.2f}, "
          f"{int((cov['route_share'] < 0.5).sum())} zones <50% routed)")


def gc_fallback(lat, lon, fill):
    """Full great-circle*fill matrix, used only where symmetrisation left NaN."""
    n = len(lat)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        M[i] = haversine_km(lat[i], lon[i], lat, lon) * fill
    return M


def write_omx(path, codes, walk, bike):
    """Write a demand_-convention OMX (h5py) matching read_skims' h5py reader."""
    import h5py
    n = len(codes)
    with h5py.File(path, "w") as f:
        f.attrs["OMX_VERSION"] = np.bytes_("0.2")
        f.attrs["SHAPE"] = np.array([n, n], dtype=np.int64)
        d = f.create_group("data")
        d.create_dataset("DISTWALK", data=walk, compression="gzip", compression_opts=4)
        d.create_dataset("DISTBIKE", data=bike, compression="gzip", compression_opts=4)
        lk = f.create_group("lookup")
        lk.create_dataset("SA2_CODE",
                          data=np.array([str(c).encode() for c in codes], dtype="S20"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff-km", type=float,
                    default=float(os.environ.get("WB_CUTOFF_KM", "20")),
                    help="Only route OD pairs within this great-circle radius; "
                         "beyond it active modes are infeasible and the pair is "
                         "filled with great_circle*fill (default 20).")
    ap.add_argument("--bike-detour", type=float,
                    default=float(os.environ.get("WB_BIKE_DETOUR", "1.10")),
                    help="DISTBIKE = DISTWALK * this (default 1.10).")
    ap.add_argument("--fill-nondirect", type=float,
                    default=float(os.environ.get("WB_FILL_NONDIRECT", "1.30")),
                    help="Great-circle multiplier for unrouted/out-of-range "
                         "pairs (default 1.30).")
    ap.add_argument("--out", default=os.environ.get("WB_OUT", OUT_OMX_DEFAULT))
    ap.add_argument("--limit", type=int, default=0,
                    help="Route only the first N origins (smoke test); the rest "
                         "are filled via symmetry/great-circle.")
    args = ap.parse_args()
    if not os.path.exists(ZONE_REF):
        print(f"FATAL: zone reference not found: {ZONE_REF}")
        sys.exit(2)
    build(args)


if __name__ == "__main__":
    main()
