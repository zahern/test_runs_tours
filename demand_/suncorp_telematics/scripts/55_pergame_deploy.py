"""
55_pergame_deploy.py
========================================================================
Per-game deployment with the four properties requested:

  (1) each officer relocates AT MOST ONCE   -> every officer's schedule is
      a "plan": hold one post all window, OR move once between two posts
      (with travel-time gap). Selecting P plans = P officers, each <=1 move.
  (2) squad size solved PER GAME            -> for each fixture we sweep P and
      pick P_g at the marginal-coverage knee (per-game officer count).
  (3) predictions handled                   -> the deployable roster is solved
      on the leave-one-fixture-out PREDICTED surface, then scored on the
      realised surface; an oracle (solve on realised) bounds the gap.
  (4) posts vary PER GAME                    -> the covering MIP re-selects which
      posts are staffed for each fixture.

Formulation (per game, per P): maximal-covering MIP over plans
    max  sum_{j,t} r_{jt} c_{jt}
    s.t. c_{jt} <= sum_{k covers (j,t)} u_k       (a cell is covered only if a
         c_{jt} <= 1                               chosen plan stations there)
         sum_k u_k <= P                            (squad budget)
         u_k, c_{jt} in {0,1}
solved exactly with HiGHS (scipy.optimize.milp).

Outputs: results/deployment_pergame.csv/.tex, results/pergame_squad.csv,
         figures/fig34_game_schedules.png (officer tokens, <=1 move each),
         results/pergame_summary.json
"""
import json, os, re, numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = np.load("results/surfaces.npz", allow_pickle=True)
acc = S["acc_fx"]; labels = [str(x) for x in S["labels"]]
kick = [str(x) for x in S["kickoffs"]]; crowds = S["crowds"].astype(float)
types = [str(x) for x in S["types"]]
prank = S["prank"]; plat = S["plat"]; plon = S["plon"]
F, J, T = acc.shape
PLABEL = [f"P{int(prank[j])}" for j in range(J)]
OFF_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
MOVE_TIMES = [6, 9, 12, 15, 18, 21]   # candidate relocation buckets (kick-off = 12)
PMAX = 10
# Squad size is solved per game by an ABSOLUTE marginal rule: keep adding an
# officer while the marginal one still covers at least MIN_MARG predicted brake
# events. Because the bar is absolute (not a fraction), higher-demand games
# justify more officers and quiet games fewer -> P_g tracks the crowd.
MIN_MARG = 16.0


def haversine_m(a1, o1, a2, o2):
    R = 6371000.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = p2-p1, np.radians(o2-o1)
    return 2*R*np.arcsin(np.sqrt(np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2))


def tau_matrix(v=12.0, setup=5.0, dmin=15.0, maxm=4000.0):
    D = haversine_m(plat[:, None], plon[:, None], plat[None, :], plon[None, :])
    tau = np.ones((J, J), int)
    for i in range(J):
        for j in range(J):
            if i != j:
                tau[i, j] = max(2, int(np.ceil((setup + D[i, j]/1000.0/v*60.0)/dmin))) \
                    if D[i, j] <= maxm else 999
    return tau


def tau_matrix_network(geojson="data/osm_roads_suncorp.geojson",
                       feats="data/road_geometry_features.csv",
                       setup=5.0, dmin=15.0, max_move_min=30.0,
                       congest=0.5):
    """Generalised post-to-post travel times over the road network.

    Shortest-path minutes on time-weighted OSM links (link length /
    posted speed x congestion factor) via scripts/travel_time.py, mapped
    to MIP bucket offsets. Returns (tau, T_min, info). Raises on any
    failure so the caller can fall back to haversine.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    import travel_time as _tt
    import geopandas as _gpd
    import pandas as _pd
    _g = _gpd.read_file(geojson)
    _g["osmid"] = _pd.to_numeric(_g["osmid"], errors="coerce")
    _g = _g.dropna(subset=["osmid"])
    try:
        _f = _pd.read_csv(feats, usecols=["road_osmid", "maxspeed_kmh"])
        _g = _g.merge(_f.rename(columns={"road_osmid": "osmid"}),
                      on="osmid", how="left")
    except Exception:
        pass
    _gu = _g.to_crs(28356)
    node_xy, adj, ginfo = _tt.graph_from_gdf(
        _gu, speed_col="maxspeed_kmh", highway_col="highway",
        congest_factor=congest)
    pxy = _tt.lonlat_to_utm(plon, plat)
    T_min, fb = _tt.travel_matrix(pxy, node_xy, adj)
    tau, feas = _tt.bucket_offsets(T_min, setup_min=setup,
                                   bucket_min=dmin,
                                   max_move_min=max_move_min)
    ginfo.update(fallback_share=round(float(fb.mean()), 4),
                 max_move_min=float(max_move_min),
                 congest_factor=float(congest))
    return tau, T_min, ginfo


try:
    TAU, TAU_MIN, TAU_INFO = tau_matrix_network()
    TAU_SOURCE = "network"
    print(f"[55] travel times: road-network shortest path "
          f"(nodes={TAU_INFO['n_nodes']}, links={TAU_INFO['n_links']}, "
          f"fallback_share={TAU_INFO['fallback_share']})")
except Exception as _e:  # noqa: BLE001 - never break the MIP on travel data
    TAU = tau_matrix()
    TAU_MIN, TAU_INFO = None, {}
    TAU_SOURCE = f"haversine-fallback ({_e})"
    print(f"[55] travel times: haversine fallback ({_e})")

# ---- enumerate plans: (cells set, is_move, from_j, to_j, depart) ------------
PLANS = []
for j in range(J):                                   # static plans
    PLANS.append(dict(cells=[(j, t) for t in range(T)], move=False,
                      a=j, b=j, t1=None))
for i in range(J):                                   # single-move plans
    for j in range(J):
        if i == j or TAU[i, j] > T:
            continue
        for t1 in MOVE_TIMES:
            arr = t1 + int(TAU[i, j])
            if arr >= T:
                continue
            cells = [(i, t) for t in range(t1)] + [(j, t) for t in range(arr, T)]
            PLANS.append(dict(cells=cells, move=True, a=i, b=j, t1=t1))
NPL = len(PLANS)
# map (j,t) -> plans covering it
cover_by_cell = {}
for k, p in enumerate(PLANS):
    for c in p["cells"]:
        cover_by_cell.setdefault(c, []).append(k)
print(f"[55] {NPL} plans ({J} static + moves); {len(cover_by_cell)} coverable cells")


def solve_mclp(r, P):
    """max covered risk with <=P plans; each officer (plan) moves <=1 time."""
    cells = [(j, t) for j in range(J) for t in range(T) if r[j, t] > 0]
    cidx = {c: n for n, c in enumerate(cells)}
    NC = len(cells)
    N = NPL + NC                                     # u_k then c_cell
    c = np.zeros(N)
    for n, cell in enumerate(cells):
        c[NPL + n] = -r[cell]                        # maximise covered risk
    rows, col_, vals, lb, ub = [], [], [], [], []
    nr = 0

    def row(entries, lo, hi):
        nonlocal nr
        for v_, val in entries:
            rows.append(nr); col_.append(v_); vals.append(val)
        lb.append(lo); ub.append(hi); nr += 1
    # c_cell <= sum_{k covers cell} u_k
    for cell, n in cidx.items():
        ks = cover_by_cell.get(cell, [])
        row([(NPL + n, 1.0)] + [(k, -1.0) for k in ks], -np.inf, 0.0)
    # budget
    row([(k, 1.0) for k in range(NPL)], 0.0, float(P))
    A = csr_matrix((vals, (rows, col_)), shape=(nr, N))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(N), bounds=Bounds(0, 1),
               options={"time_limit": 60.0, "mip_rel_gap": 1e-4})
    if res.x is None:
        return None, 0.0
    z = np.round(res.x).astype(int)
    chosen = [k for k in range(NPL) if z[k]]
    covered = float(sum(r[cell] for n, cell in enumerate(cells) if z[NPL + n]))
    return chosen, covered


def coverage_on(chosen, r):
    covd = set()
    for k in chosen:
        covd.update(PLANS[k]["cells"])
    tot = r.sum()
    return float(sum(r[j, t] for (j, t) in covd) / tot) if tot > 0 else 0.0


def pick_P(r):
    """per-game squad size via an absolute marginal-events rule (demand-tracking)."""
    plans_by_P = {}
    prev = 0.0
    chosenP = 1
    for P in range(1, PMAX + 1):
        ch, cov = solve_mclp(r, P)
        plans_by_P[P] = ch
        marg = cov - prev            # absolute predicted brake events the P-th officer adds
        if P >= 2 and marg < MIN_MARG:
            chosenP = P - 1
            break
        chosenP = P
        prev = cov
    return chosenP, plans_by_P.get(chosenP, plans_by_P[min(plans_by_P)]), None


def predict_surface(f):
    others = [g for g in range(F) if g != f]
    base = np.mean([acc[g] for g in others], axis=0)
    cf = crowds[f]; known = np.array([crowds[g] for g in others], float)
    m = np.isfinite(known)
    if np.isfinite(cf) and m.sum() >= 3:
        xg = np.log(known[m] + 1)
        yg = np.log(np.array([acc[g].sum() for g in others])[m] + 1)
        beta = float(np.clip(np.polyfit(xg, yg, 1)[0], 0, 1))
        base = base * (cf / known[m].mean()) ** beta
    return base


def clean_label(lab, kiso):
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(lab) + " " + str(kiso))
    return pd.Timestamp(*(int(x) for x in m.groups())).strftime('%d %b %Y').lstrip('0') if m else str(lab)


# ---- solve every game -------------------------------------------------------
rows, plans_pred_all = [], []
for f in range(F):
    r_real = acc[f]; r_pred = predict_surface(f)
    # deployable: choose P and roster on PREDICTED surface
    P_g, chosen_pred, _ = pick_P(r_pred)
    cov_pred_real = coverage_on(chosen_pred, r_real)      # score on realised
    # oracle: same P_g on realised
    chosen_or, _ = solve_mclp(r_real, P_g)
    cov_oracle = coverage_on(chosen_or, r_real)
    nmoves = sum(PLANS[k]["move"] for k in chosen_pred)
    posts = sorted({PLABEL[PLANS[k]["a"]] for k in chosen_pred} |
                   {PLABEL[PLANS[k]["b"]] for k in chosen_pred}, key=lambda s: int(s[1:]))
    plans_pred_all.append(chosen_pred)
    rows.append(dict(fixture=labels[f], kickoff=kick[f], type=types[f],
                     crowd=("" if not np.isfinite(crowds[f]) else int(crowds[f])),
                     realised_events=int(r_real.sum()), squad=P_g, relocations=nmoves,
                     n_posts=len(posts), posts="|".join(posts),
                     cov_pred=round(cov_pred_real, 4), cov_oracle=round(cov_oracle, 4)))
    print(f"  {labels[f][:34]:34s} P={P_g} moves={nmoves} posts={len(posts)} "
          f"pred={100*cov_pred_real:.1f}% oracle={100*cov_oracle:.1f}%")

pg = pd.DataFrame(rows)
pg.to_csv("results/deployment_pergame.csv", index=False)
pg[["fixture", "squad", "relocations", "n_posts", "posts"]].to_csv(
    "results/pergame_squad.csv", index=False)

summary = dict(
    n_fixtures=F, travel_source=TAU_SOURCE,
    travel_info={k: (round(float(v), 4) if isinstance(v, float) else v)
                 for k, v in (TAU_INFO or {}).items()},
    mean_squad=round(float(pg.squad.mean()), 2),
    squad_range=[int(pg.squad.min()), int(pg.squad.max())],
    mean_relocations=round(float(pg.relocations.mean()), 2),
    max_relocations=int(pg.relocations.max()),
    distinct_post_sets=int(pg.posts.nunique()),
    mean_cov_pred=round(float(pg.cov_pred.mean()), 4),
    mean_cov_oracle=round(float(pg.cov_oracle.mean()), 4),
    pred_gap_pts=round(float(100*(pg.cov_oracle.mean()-pg.cov_pred.mean())), 2),
    corr_squad_crowd=round(float(pd.to_numeric(pg.crowd, errors="coerce").corr(pg.squad)), 3),
)
json.dump(summary, open("results/pergame_summary.json", "w"), indent=2)
print("\n[55]", json.dumps(summary, indent=2))

# ---- LaTeX table ------------------------------------------------------------
body = "\n".join(
    f"{i+1} & {clean_label(r.fixture, r.kickoff)} & {r.kickoff[11:16]} & "
    f"{r['type']} & {r.crowd if r.crowd!='' else '--'} & {r.realised_events} & "
    f"{r.squad} & {r.relocations} & {r.n_posts} & {100*r.cov_pred:.1f} & {100*r.cov_oracle:.1f} \\\\"
    for i, r in pg.iterrows())
tex = (
    "% deployment_pergame.tex - generated by 55_pergame_deploy.py\n"
    "\\begin{table}[htbp]\n\\centering\n\\begin{scriptsize}\n"
    "\\begin{tabular}{rlllrrrrrrr}\n\\toprule\n"
    "\\# & Fixture & KO & Code & Crowd & Events & $P_g$ & Moves & Posts & Pred (\\%) & Oracle (\\%) \\\\\n"
    "\\midrule\n" + body +
    f"\n\\midrule\n\\multicolumn{{6}}{{l}}{{Mean ({F} fixtures)}} & "
    f"{pg.squad.mean():.1f} & {pg.relocations.mean():.1f} & {pg.n_posts.mean():.1f} & "
    f"{100*pg.cov_pred.mean():.1f} & {100*pg.cov_oracle.mean():.1f} \\\\\n"
    "\\bottomrule\n\\end{tabular}\\end{scriptsize}\n"
    "\\captionsetup{margin={0.5in,0in}}\n"
    "\\caption{\\textcolor{Highlight}{Per-game deployment solved individually for each of the "
    f"{F} real Suncorp fixtures.}} Each officer relocates at most once; the squad size $P_g$ and "
    "the staffed posts are optimised per game on the leave-one-out predicted surface, then scored "
    "against the realised surface (Pred); Oracle solves $P_g$ on the realised surface. $P_g$ tracks "
    "expected demand, so smaller crowds are policed with fewer officers.}\n"
    "\\label{tab:pergame}\n\\end{table}\n")
open("results/deployment_pergame.tex", "w").write(tex)

# ---- figure: per-game officer tokens, <=1 move each -------------------------
ncols = 6
nrows = (F + ncols - 1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3.1*ncols, 2.5*nrows),
                         sharex=True, sharey=True, squeeze=False)
for f in range(F):
    r_, c_ = divmod(f, ncols); ax = axes[r_, c_]
    r = acc[f]
    ax.imshow(r, aspect="auto", cmap="Greys", extent=[-3, 4, J-0.5, -0.5],
              vmin=0, vmax=(np.percentile(r, 99)*1.35 if r.max() > 0 else 1))
    for oi, k in enumerate(plans_pred_all[f]):
        col = OFF_COLORS[oi % 10]; p = PLANS[k]
        if not p["move"]:
            xs = [t/4-3+0.125 for t in range(T)]; ys = [p["a"]]*T
        else:
            xs = [t/4-3+0.125 for t in range(p["t1"])] + \
                 [t/4-3+0.125 for t in range(p["t1"]+int(TAU[p["a"], p["b"]]), T)]
            ys = [p["a"]]*p["t1"] + [p["b"]]*(T-p["t1"]-int(TAU[p["a"], p["b"]]))
            ax.plot([xs[p["t1"]-1], xs[p["t1"]]], [p["a"], p["b"]], ":", color=col, lw=1.3, zorder=6)
        ax.plot(xs, ys, "-", color=col, lw=0.9, alpha=0.9, zorder=5)
        for xx, yy in zip(xs, ys):
            ax.text(xx, yy, str(oi+1), color="white", fontsize=4, ha="center", va="center",
                    zorder=7, bbox=dict(boxstyle="circle,pad=0.05", fc=col, ec="white", lw=0.3))
    ax.axvline(0, color="k", ls=":", lw=0.7, alpha=0.6)
    ck = f"{int(crowds[f])//1000}k" if np.isfinite(crowds[f]) else "?"
    ax.set_title(f"M{f+1} {kick[f][11:16]} {ck} · P={rows[f]['squad']} · {rows[f]['relocations']}mv", fontsize=6)
    if c_ == 0:
        ax.set_yticks(range(J)); ax.set_yticklabels(PLABEL, fontsize=5)
for f in range(F, nrows*ncols):
    r_, c_ = divmod(f, ncols); axes[r_, c_].axis("off")
fig.suptitle("Per-game deployment on the predicted surface: squad size, posts and the single "
             "allowed relocation per officer are all solved per fixture", fontsize=10)
fig.supxlabel("hours relative to kick-off")
plt.savefig("figures/fig34_game_schedules.png", dpi=140, bbox_inches="tight")
print("[55] saved deployment_pergame.csv/.tex, pergame_squad.csv, fig34")
