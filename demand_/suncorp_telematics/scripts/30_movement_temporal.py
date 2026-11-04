"""
Movement- and temporal-explicit deployment analysis for the paper.

Reads results/surfaces.npz (validated reconstruction of the 12:50 run) and:
  1. Re-solves Programme P5 per fixture with FULL relocation-arc export.
  2. Re-solves the aggregate event surface at P=5..8 to locate where timed
     movement first pays (the movement-value curve).
  3. Emits movement tokens (from-post -> to-post, depart/arrive bucket) to CSV.
  4. Builds three figures:
       fig32_mip_schedule.png  aggregate schedule with movement arcs (P shown)
       fig34_game_schedules.png per-EVENT schedules with movement arcs (42 panels)
       fig35_temporal_sites.png how the staffed post-set & movement vary with
                                kick-off time and crowd across all 42 fixtures
  5. Writes results/deployment_movement.csv and results/movement_summary.json.

Everything is scored on the exact same surfaces that produce the paper's
0.6398 headline, so figures and numbers stay mutually consistent.
"""
import json, numpy as np, pandas as pd
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

S = np.load("results/surfaces.npz", allow_pickle=True)
acc_fx = S["acc_fx"]; r_event = S["r_event"]; r_ord = S["r_ord"]
labels = [str(x) for x in S["labels"]]; kickoffs = [str(x) for x in S["kickoffs"]]
crowds = S["crowds"].astype(float); types = list(S["types"])
prank = S["prank"]; plat = S["plat"]; plon = S["plon"]
F, J, T = acc_fx.shape
PLABEL = [f"P{int(prank[j])}" for j in range(J)]


def haversine_m(a1, o1, a2, o2):
    R = 6371000.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = p2 - p1, np.radians(o2 - o1)
    x = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(x))


def tau_matrix(plat, plon, dmin=15.0, v=12.0, setup=5.0, maxm=4000.0):
    D = haversine_m(plat[:, None], plon[:, None], plat[None, :], plon[None, :])
    tau = np.full((J, J), 10**6, int)
    for i in range(J):
        for j in range(J):
            if i == j:
                tau[i, j] = 1
            elif D[i, j] <= maxm:
                tau[i, j] = max(2, int(np.ceil((setup + D[i, j]/1000.0/v*60.0)/dmin)))
    return tau, D


TAU, D = tau_matrix(plat, plon)


def solve_p5_arcs(r, P, tau, lam=0.25, teleport=False, tl=120.0):
    """Exact relocation MIP; returns (x[J,T] bool, coverage, moves list)."""
    if teleport:
        tau = np.ones_like(tau)
        arcs = [(i, j) for i in range(J) for j in range(J)]
    else:
        arcs = [(i, j) for i in range(J) for j in range(J) if tau[i, j] <= T]
    NX = J*T
    N = NX + len(arcs)*T
    c = np.zeros(N)
    for j in range(J):
        for t in range(T):
            c[j*T+t] = -r[j, t]
    for a, (i, j) in enumerate(arcs):
        if i != j:
            for t in range(T):
                c[NX+a*T+t] = lam
    rows, cols, vals, lb, ub = [], [], [], [], []
    nr = 0

    def row(entries, lo, hi):
        nonlocal nr
        for var, val in entries:
            rows.append(nr); cols.append(var); vals.append(val)
        lb.append(lo); ub.append(hi); nr += 1

    def xv(j, t): return j*T + t
    def yv(a, t): return NX + a*T + t
    out_arcs, in_arcs = defaultdict(list), defaultdict(list)
    for a, (i, j) in enumerate(arcs):
        out_arcs[i].append((a, j)); in_arcs[j].append((a, i))
    row([(xv(j, 0), 1.0) for j in range(J)], P, P)
    for j in range(J):
        for t in range(1, T):
            ent = [(xv(j, t), -1.0)]
            ent += [(yv(a, t-tau[i, j]), 1.0) for a, i in in_arcs[j] if 0 <= t-tau[i, j] < T]
            row(ent, 0.0, 0.0)
    for j in range(J):
        for t in range(T-1):
            row([(xv(j, t), 1.0)] + [(yv(a, t), -1.0) for a, _ in out_arcs[j]], 0.0, 0.0)
    A = csr_matrix((vals, (rows, cols)), shape=(nr, N))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(N), bounds=Bounds(0, 1),
               options={"time_limit": tl, "mip_rel_gap": 0.0})
    z = np.round(res.x).astype(int)
    x = z[:NX].reshape(J, T).astype(bool)
    moves = []
    for a, (i, j) in enumerate(arcs):
        if i == j:
            continue
        for t in range(T):
            if z[NX+a*T+t]:
                moves.append(dict(from_post=PLABEL[i], to_post=PLABEL[j], from_j=i, to_j=j,
                                  depart=t, arrive=t+tau[i, j], travel=int(tau[i, j])))
    cov = float(r[x].sum()/r.sum())
    return x, cov, moves


def coverage(x, r): return float(r[x].sum()/r.sum())


# ----------------------------------------------------------------------------
# 1. aggregate schedule + squad-size movement curve
# ----------------------------------------------------------------------------
P = 5
x5, cov5, mv5 = solve_p5_arcs(r_event, P, TAU, lam=0.25)
print(f"aggregate P={P}: coverage={cov5:.4f} relocations={len(mv5)}")

sweep = []
for p in range(1, 11):
    xp, cp, mvp = solve_p5_arcs(r_event, p, TAU, lam=0.25)
    xt, ct, _ = solve_p5_arcs(r_event, p, TAU, lam=0.0, teleport=True)
    tp = np.argsort(-r_event.sum(1))[:p]
    sx = np.zeros((J, T), bool); sx[tp] = True
    sweep.append(dict(P=p, coverage=round(cp, 4), static=round(coverage(sx, r_event), 4),
                      teleport=round(ct, 4), relocations=len(mvp),
                      move_pts=round(100*(ct-coverage(sx, r_event)), 2)))
    print(f"  P={p}: mip={cp:.3f} static={coverage(sx,r_event):.3f} teleport={ct:.3f} moves={len(mvp)}")

# ----------------------------------------------------------------------------
# 2. per-fixture solve with arcs (predicted-surface roster would need MetaCount;
#    here we score the ORACLE per-fixture schedule so movement reflects each
#    game's realised space-time surface -- this is what "per event" shows).
# ----------------------------------------------------------------------------
per_fx = []
mv_rows = []
x_fx = np.zeros((F, J, T), bool)
for f in range(F):
    xf, covf, mvf = solve_p5_arcs(acc_fx[f], P, TAU, lam=0.25)
    x_fx[f] = xf
    posts_used = sorted({PLABEL[j] for j in range(J) if xf[j].any()}, key=lambda s: int(s[1:]))
    per_fx.append(dict(fixture=labels[f], kickoff=kickoffs[f], crowd=(int(crowds[f]) if np.isfinite(crowds[f]) else -1),
                       type=types[f], coverage=round(covf, 4), relocations=len(mvf),
                       n_posts=len(posts_used), posts="|".join(posts_used)))
    for m in mvf:
        mv_rows.append(dict(fixture=labels[f], kickoff=kickoffs[f], crowd=(int(crowds[f]) if np.isfinite(crowds[f]) else -1), **m))

pf = pd.DataFrame(per_fx)
pd.DataFrame(mv_rows).to_csv("results/deployment_movement.csv", index=False)
pf.to_csv("results/deployment_perfixture_sites.csv", index=False)
print(f"per-fixture: {int((pf.relocations>0).sum())}/{F} games with movement; "
      f"{pf.relocations.sum()} total relocations")
print(f"distinct staffed-post sets: {pf.posts.nunique()}")

# core set = posts staffed in >=50% of games
core = [p for p in PLABEL if pf.posts.str.contains(rf'\b{p}\b').mean() >= 0.5]
print("core posts:", core)

# ----------------------------------------------------------------------------
# helpers to draw a schedule with movement arcs
# ----------------------------------------------------------------------------
def draw_schedule(ax, r, x, moves, title, show_labels=True):
    im = ax.imshow(r, aspect="auto", cmap="viridis", extent=[-3, 4, J-0.5, -0.5])
    for j in range(J):
        for t in range(T):
            if x[j, t]:
                ax.add_patch(plt.Rectangle((t/4-3, j-0.5), 0.25, 1, fill=False,
                                           edgecolor="white", lw=1.3))
    for m in moves:
        x0 = m["depart"]/4 - 3 + 0.12; y0 = m["from_j"]
        x1 = m["arrive"]/4 - 3; y1 = m["to_j"]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=11, lw=1.6, color="#FF3B30",
                     connectionstyle="arc3,rad=0.15", zorder=6))
    ax.axvline(0, color="w", ls=":", lw=0.8, alpha=0.7)  # kick-off
    if show_labels:
        ax.set_yticks(range(J)); ax.set_yticklabels(PLABEL, fontsize=7)
    ax.set_title(title, fontsize=8)
    return im


# ---- fig32: aggregate at P=5 (static) with P=7 inset showing movement --------
fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
x7, cov7, mv7 = solve_p5_arcs(r_event, 7, TAU, lam=0.25)
im0 = draw_schedule(axs[0], r_event, x5, mv5,
                    f"Mean event-day surface — P=5 optimum\ncoverage {100*cov5:.1f}%, {len(mv5)} relocations (static)")
im1 = draw_schedule(axs[1], r_event, x7, mv7,
                    f"Same surface — P=7 optimum\ncoverage {100*cov7:.1f}%, {len(mv7)} timed relocation(s)")
for ax in axs:
    ax.set_xlabel("hours relative to kick-off")
axs[0].set_ylabel("candidate post")
fig.colorbar(im1, ax=axs, label="mean risky-driving per 15-min bucket", fraction=0.035, pad=0.02)
fig.suptitle("Optimal space–time deployment: once EB-sited, movement pays only as the squad grows (red arrows = timed relocations)",
             fontsize=10)
plt.savefig("figures/fig32_mip_schedule.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- fig34: per-EVENT schedules with movement arcs (42 panels) --------------
ncols = 6
nrows = (F + ncols - 1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.4*nrows),
                         sharex=True, sharey=True, squeeze=False)
for f in range(F):
    r_, c_ = divmod(f, ncols)
    ax = axes[r_, c_]
    mvf = [m for m in mv_rows if m["fixture"] == labels[f]]
    ko = kickoffs[f][11:16]
    tag = f"M{f+1} {ko} {(str(int(crowds[f])//1000)+"k") if np.isfinite(crowds[f]) else "?"}"
    if per_fx[f]["relocations"] > 0:
        tag += f" ·{per_fx[f]['relocations']}mv"
    draw_schedule(ax, acc_fx[f], x_fx[f], mvf, tag, show_labels=(c_ == 0))
for f in range(F, nrows*ncols):
    r_, c_ = divmod(f, ncols)
    axes[r_, c_].axis("off")
fig.suptitle("Per-fixture optimal deployment for every event (white = on-station bucket; red arrow = timed relocation). "
             "Core roster is stable; movement appears in off-peak-timed and higher-demand games.",
             fontsize=10)
fig.supxlabel("hours relative to kick-off")
plt.savefig("figures/fig34_game_schedules.png", dpi=140, bbox_inches="tight")
plt.close()

# ---- fig35: temporal / per-event variation vs kick-off and crowd ------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
# (a) staffing matrix: post (row) x fixture ordered by kickoff hour
ko_hour = np.array([pd.Timestamp(k).hour for k in kickoffs])
order = np.argsort(ko_hour + crowds/1e7)  # sort by kickoff then crowd
M = np.zeros((J, F))
for f in range(F):
    for j in range(J):
        M[j, f] = x_fx[f][j].sum()  # buckets staffed
ax = axes[0]
im = ax.imshow(M[:, order], aspect="auto", cmap="magma")
ax.set_yticks(range(J)); ax.set_yticklabels(PLABEL, fontsize=7)
ax.set_xlabel("fixtures ordered by kick-off hour →")
ax.set_title("(a) Staffed buckets per post across all 42 fixtures")
fig.colorbar(im, ax=ax, label="buckets on station", fraction=0.046)
# (b) n posts staffed vs crowd, coloured by kickoff hour
ax = axes[1]
sc = ax.scatter(crowds, pf.n_posts, c=ko_hour, cmap="twilight", s=45, edgecolor="k", lw=0.4)
ax.set_xlabel("crowd (attendance)"); ax.set_ylabel("distinct posts staffed")
ax.set_title("(b) Deployment footprint vs crowd size")
ax.grid(alpha=.3)
fig.colorbar(sc, ax=ax, label="kick-off hour")
# (c) relocations vs kickoff hour
ax = axes[2]
jit = ko_hour + np.random.default_rng(0).normal(0, 0.08, F)
ax.scatter(jit, pf.relocations, c=crowds, cmap="viridis", s=45, edgecolor="k", lw=0.4)
ax.set_xlabel("kick-off hour"); ax.set_ylabel("timed relocations in optimal roster")
ax.set_title("(c) When movement is worth it")
ax.grid(alpha=.3)
fig.suptitle("How the optimal deployment shifts with match day, kick-off time and event size",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/fig35_temporal_sites.png", dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------
summary = dict(
    P=P, coverage_event=round(cov5, 4), relocations_event=len(mv5),
    games_with_movement=int((pf.relocations > 0).sum()), total_relocations=int(pf.relocations.sum()),
    distinct_site_sets=int(pf.posts.nunique()), core_posts=core,
    n_core=len(core), sweep=sweep,
    snap_median_m=float(S["snap_median"]), within_500m=float(S["within500"]),
    corr_nposts_crowd=float(np.corrcoef(crowds, pf.n_posts)[0, 1]),
)
json.dump(summary, open("results/movement_summary.json", "w"), indent=2)
print("\nsaved figures fig32/fig34/fig35 + movement CSVs + summary")
print(json.dumps({k: summary[k] for k in
     ("games_with_movement", "total_relocations", "distinct_site_sets", "core_posts",
      "corr_nposts_crowd", "snap_median_m", "within_500m")}, indent=2))
