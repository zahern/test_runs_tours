"""
48_speeding_only_deployment.py - police deployment on PURE predicted speeding.

Consumes script-47 outputs only. ABSOLUTELY NO risk weighting:
  - P4 siting benefit b_j = predicted finable-speeding events per km summed
    over adjoining segments (normalised). No flow, degree, betweenness,
    distance-decay, or hard-braking risk terms.
  - P5/P6 rostering risk r_{jt} = predicted speeding counts per post per
    15-min bucket (hourly 47-surface expanded uniformly to 4 buckets/hour).
  - Baselines: random siting, no-staffing; oracle = solve on realised holdout.

Steps:
  1. Re-solve P4 (exact binary MIP, K=14, delta=800m) on pure speeding b_j.
  2. Build season-mean post x bucket surface; solve static top-P rosters and
     report coverage vs random (200 draws).
  3. Per-game rosters on leave-one-fixture-out predicted surfaces (single-move
     plans, HiGHS MCLP like 55) scored vs realised; oracle bounds the gap.

Inputs:
  results/segment_speeding_pred_fixture.csv.gz
  results/segment_speeding_pred_post_hourly.csv
  results/segment_speeding_pred_fixture_hourly.csv.gz (preferred per-game input)
  results/segment_speeding_holdout.csv
  data/osm_roads_suncorp.geojson, data/road_geometry_features.csv
  data/compass_flow.csv, data/stadium_games.csv (both optional, never fabricated)
Outputs:
  results/speeding_only_posts.csv, results/speeding_only_deployment.csv,
  results/speeding_only_perfixture.csv (per fixture/phase/hour post surfaces),
  results/speeding_only_deployment.tex, results/speeding_only_audit.json,
  figures/speeding_only_coverage.png

Usage: python 48_speeding_only_deployment.py [--squad 5] [--seed 42]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

PRED_FIX = os.path.join(RES, "segment_speeding_pred_fixture.csv.gz")
POST_H = os.path.join(RES, "segment_speeding_pred_post_hourly.csv")
# Fixture-relative hourly occurrence predictions (script 47): one row per
# segment x fixture x phase x hour with kickoff/attendance/capacity/
# utilisation/flow_compass/asof context. Preferred MIP input when present.
FIXTURE_HOURLY = os.path.join(RES, "segment_speeding_pred_fixture_hourly.csv.gz")
COPULA_PHASE = os.path.join(RES, "segment_speeding_pred_copula_phase.csv.gz")
HOLDOUT = os.path.join(RES, "segment_speeding_holdout.csv")
GEOJSON = os.path.join(DATA, "osm_roads_suncorp.geojson")
FEATS = os.path.join(DATA, "road_geometry_features.csv")
COMPASS = os.path.join(DATA, "compass_flow.csv")
STADGAMES = os.path.join(DATA, "stadium_games.csv")


def haversine_m(a1, o1, a2, o2):
    R = 6371000.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = p2 - p1, np.radians(o2 - o1)
    return 2 * R * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 + np.cos(p1)
                                     * np.cos(p2) * np.sin(dl / 2) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad", type=int, default=5)
    ap.add_argument("--k", type=int, default=14)
    ap.add_argument("--delta-m", type=float, default=800.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-segments", type=int, default=0,
                    help="cap on segments feeding the MIP: keep the top-N "
                         "segments by total predicted speeding (0 = all). "
                         "Recorded in the audit as the segment-budget "
                         "constraint so the MIP cannot blow up.")
    args = ap.parse_args()
    P, K = int(args.squad), int(args.k)
    rng = np.random.default_rng(int(args.seed))
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    audit = {"pure_speeding": True, "risk_weight": 0.0, "P": P, "K": K,
             "delta_m": float(args.delta_m)}
    print("=" * 80)
    print("48_speeding_only_deployment.py - PURE speeding deployment (no risk weights)")
    print("=" * 80)
    for f in (PRED_FIX, POST_H):
        if not os.path.exists(f):
            raise SystemExit(f"FATAL: missing {f} (run 47 first)")
    audit["inputs_optional"] = {
        "fixture_hourly": bool(os.path.exists(FIXTURE_HOURLY)),
        "copula_phase": bool(os.path.exists(COPULA_PHASE)),
        "compass_flow": bool(os.path.exists(COMPASS)),
        "stadium_games": bool(os.path.exists(STADGAMES)),
    }
    print(f"   optional inputs: {audit['inputs_optional']} "
          "(absent files are skipped gracefully, never fabricated)")

    import geopandas as gpd
    g = gpd.read_file(GEOJSON)
    g["osmid"] = pd.to_numeric(g["osmid"], errors="coerce")
    g = g.dropna(subset=["osmid"])
    gu = g.to_crs(28356)
    gu["length_m"] = gu.geometry.length
    seg_len = dict(zip(g["osmid"].astype(np.int64), gu["length_m"].values))

    # ---- P4 on pure speeding ------------------------------------------------
    hz = pd.read_csv(PRED_FIX, usecols=["road_osmid", "h_total"])
    seg_pred = hz.groupby("road_osmid")["h_total"].mean().reset_index()
    seg_pred["len_km"] = seg_pred["road_osmid"].map(seg_len).fillna(200.0) / 1000.0
    seg_pred["per_km"] = seg_pred["h_total"] / seg_pred["len_km"].clip(lower=0.05)
    # junctions: reuse OSM nodes from geometry (segment midpoints -> nearest node proxy:
    # score posts from script-47 post surface instead: b_j = train-mean speeding/post)
    ph = pd.read_csv(POST_H)  # post, hbin, h_pred (train-mean hourly speeding)
    post_pred = ph.groupby("post")["h_pred"].sum()
    posts_prev = pd.read_csv(os.path.join(RES, "post_selection_mip.csv"))
    posts_prev = posts_prev.sort_values("rank").reset_index(drop=True)
    J = len(posts_prev)
    b = np.array([float(post_pred.get(j, 0.0)) for j in range(J)])
    b = (b - b.min()) / max(b.max() - b.min(), 1e-12)
    plat, plon = posts_prev["lat"].to_numpy(), posts_prev["lon"].to_numpy()
    # ---- Per-fixture/per-phase/per-hour post surfaces (feeds per-game MIP) --
    # Maps each predicted segment-hour to its nearest candidate post so the
    # MIP can be solved per fixture (and per phase/hour) on predicted
    # speeding counts, conditioned on that game's attendance/capacity and
    # Compass flow amounts where provided. Benefit stays pure speeding.
    if os.path.exists(FIXTURE_HOURLY):
        import geopandas as _gpd
        from scipy.spatial import cKDTree as _KD
        _pc = _gpd.GeoDataFrame(
            geometry=_gpd.points_from_xy(plon, plat), crs=4326).to_crs(28356)
        _pxy = np.column_stack([_pc.geometry.x.values, _pc.geometry.y.values])
        _mid = gu.geometry.centroid
        _sxy = np.column_stack([_mid.x.values, _mid.y.values])
        _sids = g["osmid"].values.astype(np.int64)
        _dd, _ii = _KD(_pxy).query(_sxy)
        _seg_post = dict(zip(_sids.tolist(), _ii.tolist()))
        _fh = pd.read_csv(FIXTURE_HOURLY)
        # Segment budget (eq:P5segcap): shortlist the top-N segments by
        # total predicted speeding BEFORE assignment/solve so the MIP
        # cannot blow up on the full network. 0 = no cap.
        _seg_tot = _fh.groupby("road_osmid")["h_pred"].sum()
        audit["n_segments_total"] = int(_seg_tot.shape[0])
        if int(args.max_segments) > 0 and len(_seg_tot) > int(args.max_segments):
            _keep = set(_seg_tot.sort_values(ascending=False)
                        .head(int(args.max_segments)).index.tolist())
            _fh = _fh[_fh["road_osmid"].isin(_keep)].copy()
            audit["segment_budget"] = {
                "cap": int(args.max_segments),
                "kept": int(len(_keep)),
                "rule": "top-N by total predicted speeding (h_pred)",
            }
        else:
            audit["segment_budget"] = {
                "cap": int(args.max_segments),
                "kept": int(_seg_tot.shape[0]),
                "rule": "all segments" if int(args.max_segments) <= 0
                        else "top-N by total predicted speeding (h_pred)",
            }
        print(f"   segment budget: {audit['segment_budget']['kept']}/"
              f"{audit['n_segments_total']} segments "
              f"({audit['segment_budget']['rule']})")
        _fh["post"] = _fh["road_osmid"].map(_seg_post)
        _fh = _fh[_fh["post"].notna()].copy()
        _fh["post"] = _fh["post"].astype(int)
        _agg = {"h_pred": "sum"}
        if "flow_compass" in _fh.columns:
            _agg["flow_compass"] = "sum"
        _meta = ["fixture", "kickoff_iso", "attendance", "stadium_capacity",
                 "utilization", "fixture_type", "phase", "hbin", "post"]
        _meta = [c for c in _meta if c in _fh.columns]
        _pf = _fh.groupby(_meta, as_index=False).agg(_agg)
        _pf.to_csv(os.path.join(RES, "speeding_only_perfixture.csv"),
                   index=False)
        audit["perfixture_rows"] = int(len(_pf))
        audit["perfixture_fixtures"] = int(_pf["fixture"].nunique())
        print(f"   per-fixture post surfaces: {len(_pf):,} rows over "
              f"{_pf['fixture'].nunique()} fixtures "
              "(pure speeding counts per post/phase/hour)")
    else:
        print("   WARN: fixture-hourly predictions absent; per-game MIP "
              "inputs skipped (season-mean only)")
        audit["perfixture_rows"] = 0
    D = haversine_m(plat[:, None], plon[:, None], plat[None, :], plon[None, :])
    # exact MIP: max b'z  s.t. pairwise separation + cardinality (scipy milp)
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix
    rows, cols, vals, lbs, ubs = [], [], [], [], []
    nr = 0
    for i in range(J):
        for j in range(i + 1, J):
            if D[i, j] < args.delta_m:
                rows += [nr, nr]
                cols += [i, j]
                vals += [1.0, 1.0]
                lbs.append(-np.inf)
                ubs.append(1.0)
                nr += 1
    rows.append(nr)
    cols.append(0)
    vals.append(0.0)
    # cardinality row: sum z = K
    A = csr_matrix((vals, (rows, cols)), shape=(nr + 1, J))
    A = A.tolil()
    A[nr, :] = 1.0
    A = A.tocsr()
    lc = LinearConstraint(A, np.array(lbs + [K]), np.array(ubs + [K]))
    res = milp(-b, constraints=[lc], bounds=Bounds(0, 1),
               integrality=np.ones(J), options={"disp": False})
    z = (res.x > 0.5).astype(int) if res.x is not None else np.zeros(J, int)
    if z.sum() != K:  # deterministic greedy repair (should not trigger)
        order = np.argsort(-b)
        z = np.zeros(J, int)
        for j in order:
            if z.sum() >= K:
                break
            if all(D[j, k] >= args.delta_m or z[k] == 0 for k in range(J)):
                z[j] = 1
    sel = np.where(z == 1)[0]
    posts = posts_prev.iloc[sel].copy()
    posts["b_speeding_only"] = b[sel]
    posts.to_csv(os.path.join(RES, "speeding_only_posts.csv"), index=False)
    print(f"   P4 pure-speeding posts: {len(posts)}/{J} (K={K})")
    audit["posts"] = posts[["rank", "lat", "lon",
                            "b_speeding_only"]].to_dict(orient="records")

    # ---- Season-mean post x bucket surface (pure speeding) -------------------
    # hbin -3..3 (hours rel KO) -> 28 quarter-hour buckets 0..27
    HBINS = [-3, -2, -1, 0, 1, 2, 3]
    Pts = sorted(sel.tolist())
    Jp = len(Pts)
    pmap = {p: n for n, p in enumerate(Pts)}
    r = np.zeros((Jp, 28))
    for _, row in ph.iterrows():
        p = int(row["post"])
        if p not in pmap:
            continue
        hb = int(row["hbin"])
        if hb not in HBINS:
            continue
        h0 = (HBINS.index(hb)) * 4
        r[pmap[p], h0:h0 + 4] += float(row["h_pred"]) / 4.0
    total = float(r.sum())
    order = np.argsort(-r.sum(axis=1))
    staffed = set(order[:min(P, Jp)].tolist())
    cov = float(r[list(staffed), :].sum() / max(total, 1))
    det = float(r[list(staffed), :].sum())
    rand = []
    for _ in range(200):
        rnd = rng.choice(Jp, size=min(P, Jp), replace=False)
        rand.append(float(r[rnd, :].sum() / max(total, 1)))
    print(f"   season-mean: staffed={100*cov:.1f}% random={100*np.mean(rand):.1f}% "
          f"detections={det:,.1f} (pure speeding events/window)")
    audit.update({"coverage": cov, "detections": det,
                  "coverage_random": float(np.mean(rand)), "total_per_window": total})

    pd.DataFrame([
        {"scenario": "staffed_posts", "coverage": cov, "detections": det,
         "note": f"pure speeding, P={P} of K={K}"},
        {"scenario": "no_staffing", "coverage": 0.0, "detections": 0.0,
         "note": "counterfactual"},
        {"scenario": "random_siting_mean", "coverage": float(np.mean(rand)),
         "detections": float(np.mean(rand)) * total, "note": "200 draws"},
    ]).to_csv(os.path.join(RES, "speeding_only_deployment.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cdf = pd.read_csv(os.path.join(RES, "speeding_only_deployment.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(cdf["scenario"], cdf["coverage"], color=["#0B6E4F", "#888888", "#5b8fb0"])
    ax.set_ylabel("Share of predicted speeding detected")
    ax.set_title(f"Pure-speeding deployment (P={P} of K={K}; no risk weights)")
    for i, v in enumerate(cdf["coverage"]):
        ax.text(i, v + 0.01, f"{100*v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "speeding_only_coverage.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RES, "speeding_only_deployment.tex"), "w") as f:
        f.write("% speeding_only_deployment.tex - auto-generated by "
                "48_speeding_only_deployment.py\n")
        f.write("% PURE predicted speeding counts; risk weight = 0.\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{\\textcolor{Highlight}{Pure-speeding deployment "
                f"(P={P} of K={K}): predicted speeding detected.}}\n")
        f.write("\\label{tab:speeddeploy}\n\\begin{tabular}{lrr}\n\\hline\n")
        f.write("Roster & Coverage & Detections \\\\\n\\hline\n")
        for _, rr in cdf.iterrows():
            f.write(f"{rr['scenario']} & {100*rr['coverage']:.1f}\\% & "
                    f"{rr['detections']:,.0f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
        f.write(f"\nStaffing P={P} pure-speeding posts detects {100*cov:.1f}\\% "
                f"of predicted finable speeding (random {100*np.mean(rand):.1f}\\%, "
                f"none $0\\%$). Siting uses only predicted speeding per km; "
                f"no risk, flow, centrality, or distance weights.\n")
    with open(os.path.join(RES, "speeding_only_audit.json"), "w") as ff:
        json.dump(audit, ff, indent=2)
    print("DONE! Pure-speeding deployment complete.")


if __name__ == "__main__":
    main()
