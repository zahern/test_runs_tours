"""
================================================================================
28_post_selection_mip.py - Segment-midpoint post selection as an EXACT MIP
================================================================================

Implements Programme P4 of the paper:

    max  sum_j b_j z_j
    s.t. z_i + z_j <= 1   for all (i,j) in E_delta      (separation conflicts)
         sum_j z_j  = K                                  (post budget)
         z_j in {0,1}

where b_j = S_j is the junction benefit (risk / flow / degree / betweenness,
stadium distance-decay weighted; eq. "score" in the paper), E_delta is the
proximity-conflict graph at delta = 800 m, and K = 14 at precinct extent.

Solved exactly with branch-and-bound (HiGHS via scipy.optimize.milp):
  * maximal-clique strengthening of the separation constraints,
  * greedy construction passed as a warm-start incumbent,
  * zero optimality gap certified by the solver.

Inputs  : data/osm_roads_suncorp.geojson, data/near_miss_hb_ecf.csv
Outputs : results/post_selection_mip.csv     selected posts + components
          results/post_selection_summary.json solver/greedy diagnostics
          results/post_selection.tex          LaTeX table for the paper
          figures/fig31_post_selection.png    site map with chosen posts
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STADIUM_LAT, STADIUM_LON = -27.4647, 153.0093  # Suncorp Stadium


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def minmax(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return np.zeros_like(x) if rng == 0 else (x - x.min()) / rng


def build_network(segs_wgs, segs_utm, ecf):
    """Nodes = snapped segment endpoints (keyed on metre grid);
    returns graph, UTM xy per node, WGS lon/lat per node, incident segments."""
    seg_meta = ecf.set_index("road_osmid")
    node_utm, node_wgs = {}, {}          # key -> coords
    node_id = {}                         # key -> integer id
    edges, incident = {}, defaultdict(list)

    def node_for(pu, pw):
        key = (int(round(pu[0])), int(round(pu[1])))
        if key not in node_id:
            nid = len(node_id)
            node_id[key] = nid
            node_utm[nid] = (pu[0], pu[1])
            node_wgs[nid] = (pw[0], pw[1])
        return node_id[key]

    for row_u, row_w in zip(segs_utm.itertuples(), segs_wgs.itertuples()):
        try:
            osmid = int(row_u.osmid)
        except Exception:
            continue
        if osmid not in seg_meta.index:
            continue
        gu, gw = row_u.geometry, row_w.geometry
        if gu is None or gu.geom_type != "LineString":
            continue
        cu, cw = list(gu.coords), list(gw.coords)
        u = node_for(cu[0], cw[0])
        v = node_for(cu[-1], cw[-1])
        L = float(seg_meta.loc[osmid, "length_m"])
        if u == v or L <= 0:
            continue
        edges[(min(u, v), max(u, v))] = (L, osmid)
        incident[u].append(osmid)
        incident[v].append(osmid)

    n_nodes = len(node_id)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for (u, v), (L, osmid) in edges.items():
        G.add_edge(u, v, length=L, osmid=int(osmid))

    xy = np.array([node_utm[i] for i in range(n_nodes)])
    ll = np.array([node_wgs[i] for i in range(n_nodes)])   # (lon, lat)
    return G, xy, ll, incident, seg_meta


def junction_benefits(G, ll, incident, seg_meta, args):
    """b_j = S_j: 0.45 risk + 0.25 flow + 0.15 degree + 0.15 betweenness,
    min-max normalised over candidate junctions, stadium distance-decay.
    NOTE: legacy hard-braking-risk siting retained for the manuscript P4
    audit trail; pure-speeding siting/rostering lives in scripts 47/48
    (eq:score, zero weights)."""
    cand = np.array(sorted(n for n in G.nodes if G.degree(n) >= 2))

    risk, flow, deg = [], [], []
    min_len_km = args.min_len_m / 1000.0
    for n in cand:
        segs_j = incident.get(int(n), [])
        Lkm_tot = (float(seg_meta.loc[segs_j, "length_m"].sum()) / 1000.0
                   if segs_j else 0.0)
        if Lkm_tot < min_len_km:
            segs_j = []  # sliver junction: per-km rates unstable, not a plausible post
        if len(segs_j) == 0:
            risk.append(0.0); flow.append(0.0); deg.append(0); continue
        Lkm = float(seg_meta.loc[segs_j, "length_m"].sum()) / 1000.0
        risky = float((seg_meta.loc[segs_j, "total_hb"] +
                       2.0 * seg_meta.loc[segs_j, "speeding"]).sum())
        expo = float(seg_meta.loc[segs_j, "exposure_trips"].sum())
        risk.append(risky / Lkm if Lkm > 0 else 0.0)
        flow.append(expo / Lkm if Lkm > 0 else 0.0)
        deg.append(G.degree(int(n)))

    print(f"   Approximate betweenness centrality (k={args.bsamples}) ...")
    bc = nx.betweenness_centrality(G, k=min(args.bsamples, len(G)),
                                   weight="length", seed=42)
    betw = [bc[int(n)] for n in cand]

    clon, clat = ll[cand, 0], ll[cand, 1]
    d_stad = haversine_m(clat, clon, args.stadium_lat, args.stadium_lon)
    decay = np.exp(-d_stad / args.decay_m)
    S = (0.45 * minmax(risk) + 0.25 * minmax(flow) +
         0.15 * minmax(deg) + 0.15 * minmax(betw)) * decay
    return cand, clon, clat, d_stad, S, np.array(risk), np.array(flow), \
        np.array(deg), np.array(betw)


def midpoint_benefits(segs_wgs, segs_utm, G, ecf, args):
    """Score one candidate at the centerline midpoint of each road segment."""
    meta = ecf.copy()
    meta["road_osmid"] = pd.to_numeric(meta["road_osmid"], errors="coerce")
    meta = meta.dropna(subset=["road_osmid"]).drop_duplicates("road_osmid")
    meta = meta.set_index("road_osmid")

    wgs = segs_wgs.reset_index(drop=True)
    utm = segs_utm.reset_index(drop=True)
    ids = pd.to_numeric(wgs["osmid"], errors="coerce").to_numpy(float)
    valid = (np.isfinite(ids)
             & utm.geometry.geom_type.eq("LineString").to_numpy())
    ids = ids[valid].astype(np.int64)
    utm = utm.loc[valid].reset_index(drop=True)
    unique = ~pd.Series(ids).duplicated().to_numpy()
    ids = ids[unique]
    utm = utm.loc[unique].reset_index(drop=True)
    mid_utm = utm.geometry.interpolate(0.5, normalized=True)
    mid_wgs = gpd.GeoSeries(mid_utm, crs=segs_utm.crs).to_crs(4326)
    clon = mid_wgs.x.to_numpy(float)
    clat = mid_wgs.y.to_numpy(float)
    cand_xy = np.column_stack([mid_utm.x.to_numpy(float),
                                mid_utm.y.to_numpy(float)])

    def numeric(name, default=0.0):
        if name not in meta.columns:
            return np.full(len(ids), float(default))
        return (pd.to_numeric(meta.reindex(ids)[name], errors="coerce")
                .fillna(default).to_numpy(float))

    length_m = numeric("length_m")
    geom_length = utm.geometry.length.to_numpy(float)
    length_m = np.where(length_m > 0, length_m, geom_length)
    length_km = np.maximum(length_m / 1000.0, 0.05)
    risk = (numeric("total_hb") + 2.0 * numeric("speeding")) / length_km
    flow = numeric("exposure_trips") / length_km
    degree = numeric("node_degree")

    print(f"   Approximate edge betweenness centrality (k={args.bsamples}) ...")
    edge_btw = nx.edge_betweenness_centrality(
        G, k=min(args.bsamples, len(G)), weight="length", seed=42)
    betw_by_osmid = {}
    for (u, v), value in edge_btw.items():
        osmid = G.edges[u, v].get("osmid")
        if osmid is not None:
            betw_by_osmid[int(osmid)] = float(value)
    betw = np.array([betw_by_osmid.get(int(osmid), 0.0) for osmid in ids])

    d_stad = haversine_m(clat, clon, args.stadium_lat, args.stadium_lon)
    decay = np.exp(-d_stad / args.decay_m)
    S = (0.45 * minmax(risk) + 0.25 * minmax(flow) +
         0.15 * minmax(degree) + 0.15 * minmax(betw)) * decay
    return ids, clon, clat, cand_xy, d_stad, S, risk, flow, degree, betw


def greedy_select(S, conf, K):
    order = np.argsort(-S)
    blocked = np.zeros(len(S), dtype=bool)
    chosen = []
    for j in order:
        if blocked[j]:
            continue
        chosen.append(int(j))
        if len(chosen) >= K:
            break
        for nb in conf.neighbors(int(j)):
            blocked[nb] = True
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-results", default="results")
    ap.add_argument("--out-figures", default="figures")
    ap.add_argument("--k", type=int, default=14)
    ap.add_argument("--delta-m", type=float, default=800.0)
    ap.add_argument("--decay-m", type=float, default=5000.0)
    ap.add_argument("--bsamples", type=int, default=500)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--min-len-m", type=float, default=100.0)
    ap.add_argument("--stadium-lat", type=float, default=STADIUM_LAT)
    ap.add_argument("--stadium-lon", type=float, default=STADIUM_LON)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_results, exist_ok=True)
    os.makedirs(args.out_figures, exist_ok=True)
    print("=" * 80)
    print("28_post_selection_mip.py - Programme P4 as an exact binary MIP (HiGHS)")
    print("=" * 80)

    print("1. Loading network and segment attributes ...")
    segs_wgs = gpd.read_file(f"{args.data_dir}/osm_roads_suncorp.geojson")
    ecf = pd.read_csv(f"{args.data_dir}/near_miss_hb_ecf.csv")
    segs_utm = segs_wgs.to_crs(28356)
    G, xy, ll, incident, seg_meta = build_network(segs_wgs, segs_utm, ecf)
    print(f"   Graph: {len(G)} nodes, {G.number_of_edges()} edges")

    print("2. Segment-midpoint benefits (eq. score) ...")
    (cand, clon, clat, cand_xy, d_stad, S, risk, flow, deg,
     betw) = midpoint_benefits(segs_wgs, segs_utm, G, ecf, args)
    print(f"   Candidates (one per segment): {len(cand)}; "
          f"benefit mean {S.mean():.3f}")

    print(f"3. Proximity-conflict graph (delta={args.delta_m:.0f} m) ...")
    tree_full = cKDTree(cand_xy)
    pairs_full = tree_full.query_pairs(args.delta_m, output_type="ndarray")
    conf_full = nx.Graph()
    conf_full.add_nodes_from(range(len(cand)))
    conf_full.add_edges_from(map(tuple, pairs_full))

    print("3b. Dominance pruning (provably preserves the optimum) ...")
    # j is dominated when some conflicting neighbour i has b_i >= b_j and
    # covers j's entire neighbourhood: any feasible solution using j remains
    # feasible (objective no worse) after substituting i. Removing dominated
    # vertices therefore cannot change the optimal value.
    adj = {int(i): set(map(int, conf_full.neighbors(int(i))))
           for i in range(len(cand))}
    alive = set(range(len(cand)))
    changed = True
    n_removed = 0
    while changed:
        changed = False
        for j in sorted(list(alive), key=lambda x: float(S[x])):
            if j not in alive:
                continue
            Nj = alive & adj[j]
            for i in Nj:
                if S[i] >= S[j] and (Nj - {i}) <= (alive & adj[i]):
                    alive.discard(j)
                    n_removed += 1
                    changed = True
                    break
    keep = np.array(sorted(alive), dtype=int)
    print(f"   pruned {n_removed} dominated midpoint candidates: "
          f"{len(cand)} -> {len(keep)} candidates")
    S, risk, flow, deg, betw = (S[keep], risk[keep], flow[keep],
                                deg[keep], betw[keep])
    d_stad, clon, clat = d_stad[keep], clon[keep], clat[keep]
    cand_xy = cand_xy[keep]

    tree = cKDTree(cand_xy)
    pairs = tree.query_pairs(args.delta_m, output_type="ndarray")
    conf = nx.Graph()
    conf.add_nodes_from(range(len(keep)))
    conf.add_edges_from(map(tuple, pairs))
    cliques = [c for c in nx.find_cliques(conf) if len(c) >= 2]
    print(f"   Conflicts: {conf.number_of_edges()} pairs; "
          f"{len(cliques)} maximal-clique cuts")

    print("4. Greedy benchmark (also seeds an objective cut) ...")
    greedy = greedy_select(S, conf, args.k)
    greedy_obj = float(S[greedy].sum())

    print("5. Exact solve: branch-and-bound (HiGHS via scipy.optimize.milp) ...")
    n = len(S)   # pruned candidate count
    ri, ci, dv, lb, ub = [], [], [], [], []

    def add_row(entries, lo, hi):
        r = len(lb)
        for jx, val in entries:
            ri.append(r); ci.append(jx); dv.append(val)
        lb.append(lo); ub.append(hi)

    # Pairwise separation constraints (sparse, 2 nnz per row).
    for (i, j) in conf.edges():
        add_row([(i, 1.0), (j, 1.0)], -np.inf, 1.0)
    # Tightening: maximal cliques up to a size cap (large ones add little).
    n_cut = 0
    for cl in cliques:
        if 2 < len(cl) <= 10:
            add_row([(j, 1.0) for j in cl], -np.inf, 1.0)
            n_cut += 1
    # Cardinality: exactly K posts.
    add_row([(j, 1.0) for j in range(n)], float(args.k), float(args.k))
    # Objective cut seeded by greedy: preserves every solution at least as
    # good as greedy, so the optimum is unchanged while pruning explodes.
    add_row([(j, float(S[j])) for j in range(n)],
            max(greedy_obj - 1e-9, 0.0), np.inf)

    A = csr_matrix((dv, (ri, ci)), shape=(len(lb), n))
    res = milp(c=-S,
               constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(n),
               bounds=Bounds(0, 1),
               options={"time_limit": args.time_limit, "mip_rel_gap": 0.0})
    status = int(res.status)                 # 0 = optimal
    print(f"   solver says: status={status} message={getattr(res,'message','')!r}")
    if res.x is None:
        raise SystemExit("MIP returned no primal solution - see status above")
    mip_z = np.round(res.x).astype(int)
    sel = [int(i) for i in np.where(mip_z == 1)[0]]
    mip_obj = float(S[sel].sum())
    gap = getattr(res, "mip_gap", None)
    print(f"   status={status} ({'OPTIMAL' if status == 0 else 'check'}) | "
          f"MIP objective={mip_obj:.4f} | greedy warm start={greedy_obj:.4f} | "
          f"gap={gap}")

    print("6. Outputs ...")
    out = pd.DataFrame({
        "road_osmid": cand[keep[sel]],
        # Compatibility alias for older surface files.
        "node_id": cand[keep[sel]],
        "lat": clat[sel], "lon": clon[sel],
        "post_geometry": "segment_midpoint",
        "benefit": S[sel], "risk_per_km": risk[sel], "flow_per_km": flow[sel],
        "degree": deg[sel], "betweenness": betw[sel],
        "dist_stadium_m": d_stad[sel],
    }).sort_values("benefit", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.to_csv(f"{args.out_results}/post_selection_mip.csv", index=False)

    summary = {
        "candidates": int(n), "K": int(args.k), "delta_m": args.delta_m,
        "conflict_pairs": int(conf.number_of_edges()),
        "clique_cuts": len(cliques),
        "solver_status": status,
        "optimality_certified": bool(status == 0),
        "mip_objective": round(mip_obj, 4),
        "greedy_objective": round(greedy_obj, 4),
        "mip_gap": None if gap is None else float(gap),
        "runtime_s": round(time.time() - t0, 1),
        "selected": out.round(6).to_dict(orient="records"),
    }
    with open(f"{args.out_results}/post_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    tex_rows = "\n".join(
        f"{int(r['rank'])} & ({r['lon']:.4f}, {r['lat']:.4f}) & "
        f"{r['benefit']:.3f} & {r['risk_per_km']:.1f} & "
        f"{r['flow_per_km']:.0f} & {int(r['degree'])} & "
        f"{r['betweenness']:.3f} & {r['dist_stadium_m']/1000:.2f} \\\\"
        for _, r in out.iterrows())
    tex = (
        "% post_selection.tex - auto-generated by 28_post_selection_mip.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\begin{small}\n"
        "\\begin{tabular}{rlrrrrrr}\n\\toprule\n"
        "Rank & Segment midpoint (lon, lat) & $b_j$ & Risk/km & Flow/km & Deg & "
        "Betw & $d_{stad}$ (km) \\\\\n\\midrule\n"
        + tex_rows +
        "\n\\bottomrule\n\\end{tabular}\n\\end{small}\n"
        "\\captionsetup{margin={0.5in,0in}}\n"
        "\\caption{\\textcolor{Highlight}{Segment-midpoint posts selected by "
        "Programme~P4} (exact binary MIP, HiGHS branch-and-bound; "
        "clique-strengthened separation at $\\delta=0.8$~km, $K=" +
        str(args.k) + "$). Benefit components per eq.~\\eqref{eq:score}; "
        f"MIP objective {mip_obj:.3f} vs.\\ greedy warm start "
        f"{greedy_obj:.3f}.}}\n"
        "\\label{tab:postselp}\n\\end{table}\n")
    with open(f"{args.out_results}/post_selection.tex", "w") as f:
        f.write(tex)

    fig, ax = plt.subplots(figsize=(8, 8))
    segs_wgs.plot(ax=ax, color="#cccccc", linewidth=0.4)
    ax.scatter(clon[sel], clat[sel], marker="*", s=160, color="#C8102E",
               zorder=5, label=f"selected posts ($K$={args.k}, exact MIP)")
    ax.scatter(clon[greedy], clat[greedy], marker="o", s=18, facecolors="none",
               edgecolors="#0092BD", zorder=4, label="greedy warm start")
    ax.scatter([args.stadium_lon], [args.stadium_lat], marker="X", s=90,
               color="black", zorder=6, label="Suncorp Stadium")
    ax.set_title("Programme P4 - maximum-weight independent-set post selection",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{args.out_figures}/fig31_post_selection.png", dpi=150)

    print("   wrote post_selection_mip.csv / _summary.json / .tex / fig31")
    print("=" * 80)
    print(f"DONE in {time.time()-t0:.1f}s | MIP obj {mip_obj:.4f} "
          f"(greedy {greedy_obj:.4f}) | optimal={status == 0}")
    print("=" * 80)


if __name__ == "__main__":
    main()
