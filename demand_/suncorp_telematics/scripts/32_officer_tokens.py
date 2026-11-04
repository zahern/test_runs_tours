"""
Officer-token deployment figures.

Replaces the arrow overlay with a per-OFFICER token: each of the P officers is a
numbered/coloured marker that we thread through the exact P5 solution (flow
decomposition of the relocation arcs), so a reader sees officer k hold, then
move, then hold -- an explicit space-time trajectory per unit.

Builds:
  figures/fig32_mip_schedule.png   aggregate P=5 (static) beside P=7 (a move),
                                   officers drawn as tokens + trajectory lines
  figures/fig34_game_schedules.png every fixture with officer-token trajectories
"""
import json, numpy as np, pandas as pd
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = np.load("results/surfaces.npz", allow_pickle=True)
acc_fx = S["acc_fx"]; r_event = S["r_event"]
labels = [str(x) for x in S["labels"]]; kickoffs = [str(x) for x in S["kickoffs"]]
crowds = S["crowds"].astype(float)
prank = S["prank"]; plat = S["plat"]; plon = S["plon"]
F, J, T = acc_fx.shape
PLABEL = [f"P{int(prank[j])}" for j in range(J)]
OFF_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def haversine_m(a1, o1, a2, o2):
    R = 6371000.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = p2-p1, np.radians(o2-o1)
    x = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(x))


def tau_matrix(plat, plon, dmin=15.0, v=12.0, setup=5.0, maxm=4000.0):
    Dm = haversine_m(plat[:, None], plon[:, None], plat[None, :], plon[None, :])
    tau = np.full((J, J), 10**6, int)
    for i in range(J):
        for j in range(J):
            if i == j:
                tau[i, j] = 1
            elif Dm[i, j] <= maxm:
                tau[i, j] = max(2, int(np.ceil((setup+Dm[i, j]/1000.0/v*60.0)/dmin)))
    return tau


TAU = tau_matrix(plat, plon)


def solve_p5_full(r, P, tau, lam=0.25, tl=120.0):
    """Return x[J,T] bool and the chosen arcs y as dict (i,j,t)->1 (holds incl.)."""
    arcs = [(i, j) for i in range(J) for j in range(J) if tau[i, j] <= T]
    aidx = {a: k for k, a in enumerate(arcs)}
    NX = J*T
    N = NX + len(arcs)*T
    c = np.zeros(N)
    for j in range(J):
        for t in range(T):
            c[j*T+t] = -r[j, t]
    for (i, j), k in aidx.items():
        if i != j:
            for t in range(T):
                c[NX+k*T+t] = lam
    rows, cols, vals, lb, ub = [], [], [], [], []
    nr = 0

    def row(entries, lo, hi):
        nonlocal nr
        for var, val in entries:
            rows.append(nr); cols.append(var); vals.append(val)
        lb.append(lo); ub.append(hi); nr += 1

    def xv(j, t): return j*T+t
    def yv(k, t): return NX+k*T+t
    out_arcs, in_arcs = defaultdict(list), defaultdict(list)
    for (i, j), k in aidx.items():
        out_arcs[i].append((k, j)); in_arcs[j].append((k, i))
    row([(xv(j, 0), 1.0) for j in range(J)], P, P)
    for j in range(J):
        for t in range(1, T):
            ent = [(xv(j, t), -1.0)] + [(yv(k, t-tau[i, j]), 1.0)
                                        for k, i in in_arcs[j] if 0 <= t-tau[i, j] < T]
            row(ent, 0.0, 0.0)
    for j in range(J):
        for t in range(T-1):
            row([(xv(j, t), 1.0)] + [(yv(k, t), -1.0) for k, _ in out_arcs[j]], 0.0, 0.0)
    A = csr_matrix((vals, (rows, cols)), shape=(nr, N))
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(N),
               bounds=Bounds(0, 1), options={"time_limit": tl, "mip_rel_gap": 0.0})
    z = np.round(res.x).astype(int)
    x = z[:NX].reshape(J, T).astype(bool)
    yy = {}
    for (i, j), k in aidx.items():
        for t in range(T):
            if z[NX+k*T+t]:
                yy[(i, j, t)] = 1
    cov = float(r[x].sum()/r.sum())
    return x, yy, cov, tau


def thread_officers(x, yy, tau):
    """Flow-decompose into P trajectories: list per officer of
    (kind, post, t) where kind in {'on','transit_start'}; returns officers list
    of dicts: {'cells':[(t,post)...], 'moves':[(t0,p0,t1,p1)...]}."""
    P = int(x[:, 0].sum())
    start_posts = [j for j in range(J) if x[j, 0]]
    officers = [{"cells": [], "moves": [], "cur": j} for j in start_posts]
    post_to_off = {j: k for k, j in enumerate(start_posts)}
    arrivals = defaultdict(list)  # bucket -> list of (officer, post)
    for t in range(T):
        # settle arrivals scheduled for t
        for (off, post) in arrivals.get(t, []):
            post_to_off[post] = off
            officers[off]["cur"] = post
        # record on-station cells
        for j in range(J):
            if x[j, t] and post_to_off.get(j) is not None:
                off = post_to_off[j]
                officers[off]["cells"].append((t, j))
        if t >= T-1:
            continue
        # process departures at t
        new_post_to_off = {}
        moved_from = set()
        for j in list(post_to_off.keys()):
            off = post_to_off[j]
            if off is None or not x[j, t]:
                continue
            # find outgoing arc from j at t
            dest = None
            for k in range(J):
                if (j, k, t) in yy:
                    dest = k
                    break
            if dest is None:
                continue
            if dest == j:  # hold
                new_post_to_off[j] = off
            else:
                a = t + int(tau[j, dest])
                officers[off]["moves"].append((t, j, a, dest))
                arrivals[a].append((off, dest))
                moved_from.add(j)
        post_to_off = new_post_to_off
    return officers


def draw_tokens(ax, r, officers, title, show_labels=True, fs=7):
    vmax = np.percentile(r, 99) if r.max() > 0 else 1
    im = ax.imshow(r, aspect="auto", cmap="Greys", extent=[-3, 4, J-0.5, -0.5],
                   vmin=0, vmax=vmax*1.35)  # lighten so coloured tokens pop
    for k, off in enumerate(officers):
        col = OFF_COLORS[k % 10]
        xs = [t/4-3+0.125 for (t, j) in off["cells"]]
        ys = [j for (t, j) in off["cells"]]
        # trajectory line (includes transit jumps)
        ax.plot(xs, ys, "-", color=col, lw=1.0, alpha=0.9, zorder=5)
        for (t, j) in off["cells"]:
            ax.text(t/4-3+0.125, j, str(k+1), color="white", fontsize=fs,
                    ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="circle,pad=0.08", fc=col, ec="white", lw=0.5))
        for (t0, p0, t1, p1) in off["moves"]:
            ax.plot([t0/4-3+0.125, t1/4-3+0.125], [p0, p1], ":", color=col,
                    lw=1.4, zorder=6)
    ax.axvline(0, color="k", ls=":", lw=0.8, alpha=0.6)
    if show_labels:
        ax.set_yticks(range(J)); ax.set_yticklabels(PLABEL, fontsize=7)
    else:
        ax.set_yticks(range(J)); ax.set_yticklabels([])
    ax.set_title(title, fontsize=8)
    return im


# ---- aggregate fig32 --------------------------------------------------------
x5, y5, cov5, _ = solve_p5_full(r_event, 5, TAU)
o5 = thread_officers(x5, y5, TAU)
x7, y7, cov7, _ = solve_p5_full(r_event, 7, TAU)
o7 = thread_officers(x7, y7, TAU)
nmv5 = sum(len(o["moves"]) for o in o5)
nmv7 = sum(len(o["moves"]) for o in o7)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
draw_tokens(axs[0], r_event, o5,
            f"Mean event-day surface — 5 officers\ncoverage {100*cov5:.1f}%, {nmv5} relocations (static)")
draw_tokens(axs[1], r_event, o7,
            f"Same surface — 7 officers\ncoverage {100*cov7:.1f}%, {nmv7} timed relocation(s)", show_labels=False)
for ax in axs:
    ax.set_xlabel("hours relative to kick-off")
axs[0].set_ylabel("candidate post")
fig.suptitle("Optimal space–time deployment — each numbered token is one officer's trajectory "
             "(dotted = a timed relocation)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/fig32_mip_schedule.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"fig32: P5 cov {cov5:.4f} moves {nmv5}; P7 cov {cov7:.4f} moves {nmv7}")

# ---- per-event fig34 --------------------------------------------------------
ncols = 6
nrows = (F + ncols - 1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3.1*ncols, 2.5*nrows),
                         sharex=True, sharey=True, squeeze=False)
movecount = []
for f in range(F):
    xf, yf, covf, _ = solve_p5_full(acc_fx[f], 5, TAU)
    off = thread_officers(xf, yf, TAU)
    nm = sum(len(o["moves"]) for o in off)
    movecount.append(nm)
    r_, c_ = divmod(f, ncols)
    ko = kickoffs[f][11:16]
    ck = f"{int(crowds[f])//1000}k" if np.isfinite(crowds[f]) else "?"
    tag = f"M{f+1} {ko} {ck} · {nm}mv"
    draw_tokens(axes[r_, c_], acc_fx[f], off, tag, show_labels=(c_ == 0), fs=5)
for f in range(F, nrows*ncols):
    r_, c_ = divmod(f, ncols)
    axes[r_, c_].axis("off")
fig.suptitle("Per-fixture optimal deployment for every event — numbered tokens are the 5 officers; "
             "dotted segments are timed relocations", fontsize=10)
fig.supxlabel("hours relative to kick-off")
plt.savefig("figures/fig34_game_schedules.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"fig34: {sum(1 for m in movecount if m>0)}/{F} games move; {sum(movecount)} total moves")
