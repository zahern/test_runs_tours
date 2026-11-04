"""
48b_extent_mip.py - P4 siting solved separately per planning extent around
Suncorp Stadium, on PURE predicted speeding (no weights, no decay).

For each extent (radius in metres; 0 = full study area):
  1. Keep road segments whose centerline midpoint lies within the radius.
  2. Score one candidate at each segment midpoint by predicted speeding per
      km (eq:score, S_j = h_j): mean predicted finable-speeding events per
      segment from script-47 fixture-hourly output (fallback: script-43
      EB finable per segment), divided by segment length, min-max normalised.
  3. Solve the exact P4 binary MIP (separation delta + cardinality K).
  4. Write per-extent posts CSV/TeX, audit JSON, and a siting map.

Realism guards (so small extents stay feasible, not forced):
  - K auto-scales with candidate count: K = max(3, round(K_full *
    n_extent / n_full)) unless --k is given explicitly.
  - If the MIP is infeasible at K (separation too tight for the area),
    K steps down to the largest feasible value and the audit records it.
  - Extents with fewer than 4 candidates are skipped, not forced.

Inputs:
  results/segment_speeding_pred_fixture_hourly.csv.gz (preferred) or
  results/finable_speeding_segments.csv (fallback)
  data/osm_roads_suncorp.geojson, data/near_miss_hb_ecf.csv,
  data/road_geometry_features.csv (segment lengths)
Outputs (per --tag):
  results/extent_posts_<tag>.csv/.tex, results/extent_mip_<tag>.json,
  figures/extent_posts_<tag>.png

Usage:
  python 48b_extent_mip.py --radius-m 2000 --tag inner_2km [--k 0]
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import geopandas as gpd

PROJ = os.environ.get(
    "SUNCORP_PROJ", "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics")
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
SCREEN = os.path.join(DATA, "near_miss_hb_ecf.csv")
FEATS = os.path.join(DATA, "road_geometry_features.csv")
FIXTURE_HOURLY = os.path.join(RES, "segment_speeding_pred_fixture_hourly.csv.gz")
FINABLE_SEG = os.path.join(RES, "finable_speeding_segments.csv")

STADIUM_LAT, STADIUM_LON = -27.4647, 153.0093

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "p4legacy", os.path.join(_HERE, "28_post_selection_mip.py"))
p4legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4legacy)


def load_segment_predictions():
    """Mean predicted speeding per segment; (series, source_note)."""
    if os.path.exists(FIXTURE_HOURLY):
        fh = pd.read_csv(FIXTURE_HOURLY, usecols=["road_osmid", "h_pred"])
        s = fh.groupby("road_osmid")["h_pred"].mean()
        return s, ("47 fixture-hourly mean h_pred "
                   f"({len(fh):,} segment-hours)")
    if os.path.exists(FINABLE_SEG):
        sc = pd.read_csv(FINABLE_SEG, usecols=["road_osmid", "eb_finable"])
        s = sc.groupby("road_osmid")["eb_finable"].mean()
        return s, ("43 EB finable per segment "
                   "(47 hourly not yet available)")
    raise SystemExit("FATAL: no segment predictions (run 43 or 47 first)")


def midpoint_candidates(segs_wgs, segs_utm):
    """Return one exact centerline midpoint candidate per road segment."""
    ids = pd.to_numeric(segs_wgs["osmid"], errors="coerce").to_numpy(float)
    valid = (np.isfinite(ids)
             & segs_utm.geometry.geom_type.eq("LineString").to_numpy())
    ids = ids[valid].astype(np.int64)
    utm = segs_utm.loc[valid].reset_index(drop=True)
    unique = ~pd.Series(ids).duplicated().to_numpy()
    ids = ids[unique]
    utm = utm.loc[unique].reset_index(drop=True)
    mid_utm = utm.geometry.interpolate(0.5, normalized=True)
    mid_wgs = gpd.GeoSeries(mid_utm, crs=segs_utm.crs).to_crs(4326)
    xy = np.column_stack([mid_utm.x.to_numpy(float),
                          mid_utm.y.to_numpy(float)])
    return (ids, xy, mid_wgs.y.to_numpy(float),
            mid_wgs.x.to_numpy(float), utm.geometry.length.to_numpy(float))


def solve_p4_exact(S, conf_xy_and_cliques, K, delta_m, time_limit):
    """Exact max-weight independent set with cardinality K.

    Returns (sel_idx, mip_obj, greedy_obj, status). Raises SystemExit only
    if nothing feasible at any K>=1 (caller steps K down first).
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    import networkx as nx
    from scipy.spatial import cKDTree
    cand_xy = conf_xy_and_cliques["xy"]
    tree = cKDTree(cand_xy)
    pairs = tree.query_pairs(delta_m, output_type="ndarray")
    conf = nx.Graph()
    conf.add_nodes_from(range(len(cand_xy)))
    conf.add_edges_from(map(tuple, pairs))
    cliques = [c for c in nx.find_cliques(conf) if len(c) >= 2]
    greedy = p4legacy.greedy_select(S, conf, K)
    greedy_obj = float(S[greedy].sum()) if greedy else 0.0
    n = len(S)
    ri, ci, dv, lb, ub = [], [], [], [], []

    def add_row(entries, lo, hi):
        r = len(lb)
        for jx, val in entries:
            ri.append(r)
            ci.append(jx)
            dv.append(val)
        lb.append(lo)
        ub.append(hi)

    for (i, j) in conf.edges():
        add_row([(i, 1.0), (j, 1.0)], -np.inf, 1.0)
    for cl in cliques:
        if 2 < len(cl) <= 10:
            add_row([(j, 1.0) for j in cl], -np.inf, 1.0)
    add_row([(j, 1.0) for j in range(n)], float(K), float(K))
    add_row([(j, float(S[j])) for j in range(n)],
            max(greedy_obj - 1e-9, 0.0), np.inf)
    A = csr_matrix((dv, (ri, ci)), shape=(len(lb), n))
    res = milp(c=-S, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    status = int(res.status)
    if res.x is None or status != 0:
        return None, greedy_obj, greedy_obj, status
    z = np.round(res.x).astype(int)
    sel = [int(i) for i in np.where(z == 1)[0]]
    return sel, float(S[sel].sum()), greedy_obj, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-m", type=float, default=0.0,
                    help="planning extent radius around the stadium (m); "
                         "0 = full study area")
    ap.add_argument("--tag", type=str, default="full",
                    help="output filename tag for this extent")
    ap.add_argument("--k", type=int, default=0,
                    help="post budget; 0 = auto-scale from candidate count")
    ap.add_argument("--k-full", type=int, default=14,
                    help="reference budget at full extent (manuscript K=14)")
    ap.add_argument("--delta-m", type=float, default=800.0)
    ap.add_argument("--min-len-m", type=float, default=100.0)
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    audit = {"tag": args.tag, "radius_m": float(args.radius_m),
             "pure_speeding": True, "risk_weight": 0.0}
    print("=" * 80)
    print(f"48b_extent_mip.py - pure-speeding P4 at extent '{args.tag}' "
          f"(radius={args.radius_m:.0f} m)")
    print("=" * 80)

    segs_wgs = gpd.read_file(GEOJSON)
    segs_wgs["osmid"] = pd.to_numeric(segs_wgs["osmid"], errors="coerce")
    segs_wgs = segs_wgs.dropna(subset=["osmid"])
    segs_utm = segs_wgs.to_crs(28356)
    mid = segs_utm.geometry.interpolate(0.5, normalized=True)
    _st = gpd.GeoSeries([gpd.points_from_xy([STADIUM_LON], [STADIUM_LAT])[0]],
                        crs=4326).to_crs(28356)
    sx, sy = float(_st.x.iloc[0]), float(_st.y.iloc[0])
    d_stad_seg = np.sqrt((mid.x.values - sx) ** 2 + (mid.y.values - sy) ** 2)
    if args.radius_m and args.radius_m > 0:
        keep = d_stad_seg <= float(args.radius_m)
        segs_wgs = segs_wgs[keep].copy()
        segs_utm = segs_utm[keep].copy()
        print(f"   extent filter: {int(keep.sum())}/{len(keep)} segments "
              f"within {args.radius_m:.0f} m")
    else:
        print(f"   full extent: {len(segs_wgs)} segments (no radius filter)")
    if len(segs_wgs) < 10:
        raise SystemExit("FATAL: too few segments in extent for siting")

    ecf = pd.read_csv(SCREEN)
    # Full-extent midpoint count for K auto-scaling (cheap: no MIP).
    _all_wgs = gpd.read_file(GEOJSON)
    _all_wgs["osmid"] = pd.to_numeric(_all_wgs["osmid"], errors="coerce")
    _all_wgs = _all_wgs.dropna(subset=["osmid"])
    _all_utm = _all_wgs.to_crs(28356)
    _all_ids, _, _, _, _ = midpoint_candidates(_all_wgs, _all_utm)
    n_full = int(len(_all_ids))
    cand, cand_xy, clat, clon, cand_len_m = midpoint_candidates(
        segs_wgs, segs_utm)
    n_ext = int(len(cand))
    print(f"   candidates (one per segment midpoint): full={n_full}, "
          f"extent={n_ext}")
    if n_ext < 4:
        raise SystemExit("FATAL: fewer than 4 midpoint candidates in extent "
                         "(area too small for a separated post network)")
    audit.update({"n_candidates_full": n_full, "n_candidates_extent": n_ext})

    K = int(args.k)
    if K <= 0:
        K = max(3, int(round(int(args.k_full) * n_ext / max(n_full, 1))))
        audit["k_rule"] = (f"auto: max(3, round({args.k_full} * "
                           f"{n_ext}/{n_full}))")
    else:
        audit["k_rule"] = "explicit --k"
    K = min(K, n_ext)
    print(f"   post budget K={K}")

    # Pure-speeding benefit per segment midpoint (eq:score, zero weights).
    pred, pred_note = load_segment_predictions()
    audit["benefit_source"] = pred_note
    try:
        feats = pd.read_csv(FEATS, usecols=["road_osmid", "length_m"])
        len_map = dict(zip(feats["road_osmid"].astype(np.int64),
                           feats["length_m"].astype(float)))
    except Exception:  # noqa: BLE001
        len_map = {}
    length_m = np.array([len_map.get(int(s), fallback)
                         for s, fallback in zip(cand, cand_len_m)], float)
    length_m = np.where(length_m > 0, length_m, cand_len_m)
    per_km = (pred.reindex(cand).fillna(0.0).to_numpy(float)
              / np.maximum(length_m / 1000.0, 0.05))
    S_raw = per_km.copy()
    S = p4legacy.minmax(S_raw)
    print(f"   benefit: pure predicted speeding/km "
          f"(max {S_raw.max():.1f} events/km at best midpoint)")

    # Exact MIP with feasibility fallback (step K down, never force).
    K_used, k_adjusted = K, False
    sel = obj = gob = status = None
    while K_used >= 1:
        sel, obj, gob, status = solve_p4_exact(
            S, {"xy": cand_xy}, K_used, float(args.delta_m),
            float(args.time_limit))
        if sel is not None:
            break
        K_used -= 1
        k_adjusted = True
    if sel is None:
        raise SystemExit("FATAL: MIP infeasible even at K=1")
    if k_adjusted:
        print(f"   K adjusted {K}->{K_used} for separation feasibility")
    audit.update({"K_requested": K, "K_used": int(K_used),
                  "k_adjusted": bool(k_adjusted), "mip_status": status,
                  "mip_objective": round(float(obj), 4),
                  "greedy_objective": round(float(gob), 4)})
    print(f"   MIP OPTIMAL: objective={obj:.4f} (greedy {gob:.4f})")

    d_j = p4legacy.haversine_m(clat, clon, STADIUM_LAT, STADIUM_LON)
    out = pd.DataFrame({
        "road_osmid": cand[sel], "node_id": cand[sel],
        "lat": clat[sel], "lon": clon[sel],
        "post_geometry": "segment_midpoint",
        "b_speeding": S[sel], "per_km": per_km[sel],
        "dist_stadium_m": d_j[sel],
    }).sort_values("b_speeding", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.to_csv(os.path.join(RES, f"extent_posts_{args.tag}.csv"), index=False)
    audit["posts"] = out.round(4).to_dict(orient="records")

    rows = "\n".join(
        f"{int(r['rank'])} & ({r['lon']:.4f}, {r['lat']:.4f}) & "
        f"{r['b_speeding']:.3f} & {r['per_km']:.1f} & "
        f"{r['dist_stadium_m']/1000:.2f} \\\\"
        for _, r in out.iterrows())
    tex = (
        "% extent_posts - auto-generated by 48b_extent_mip.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\begin{small}\n"
        "\\begin{tabular}{rlrrr}\n\\toprule\n"
           "Rank & Segment midpoint (lon, lat) & $b_j$ & Speed/km & $d_{stad}$ (km) \\\\\n\\midrule\n"
        + rows +
        "\n\\bottomrule\n\\end{tabular}\n\\end{small}\n"
        "\\captionsetup{margin={0.5in,0in}}\n"
        # Tags carry underscores (e.g. "precinct_1km"); inside
        # \textcolor{...}{...} they are TEXT mode, so an unescaped "_"
        # makes TeX emit "Missing $ inserted" and then derails math mode
        # for the rest of the document. Escape for the caption only -- the
        # label and the filenames must keep the raw tag.
        "\\caption{\\textcolor{Highlight}{Pure-speeding P4 posts at extent "
        f"{str(args.tag).replace('_', chr(92) + '_')}}} "
        f"(radius {args.radius_m:.0f} m, $K={int(K_used)}$, "
        "$\\delta=0.8$~km; exact MIP, zero weights).}\n"
        f"\\label{{tab:extent{args.tag}}}\n\\end{{table}}\n")
    with open(os.path.join(RES, f"extent_posts_{args.tag}.tex"), "w") as f:
        f.write(tex)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    segs_wgs.plot(ax=ax, color="#cccccc", linewidth=0.4)
    ax.scatter(clon[sel], clat[sel], marker="*", s=160, color="#C8102E",
               zorder=5, label=f"selected posts (K={int(K_used)})")
    ax.scatter([STADIUM_LON], [STADIUM_LAT], marker="X", s=90,
               color="black", zorder=6, label="Suncorp Stadium")
    if args.radius_m and args.radius_m > 0:
        ax.add_patch(plt.Circle((STADIUM_LON, STADIUM_LAT),
                                float(args.radius_m) / 111320.0,
                                color="black", fill=False, ls="--", lw=1,
                                label=f"extent radius {args.radius_m:.0f} m"))
    ax.set_title(f"Pure-speeding P4 posts - extent '{args.tag}'",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"extent_posts_{args.tag}.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RES, f"extent_mip_{args.tag}.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"   wrote extent_posts_{args.tag}.csv/.tex, "
          f"extent_mip_{args.tag}.json, extent_posts_{args.tag}.png")
    print("=" * 80)
    print(f"DONE in {time.time()-t0:.1f}s | extent={args.tag} "
          f"K={int(K_used)} obj={obj:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

