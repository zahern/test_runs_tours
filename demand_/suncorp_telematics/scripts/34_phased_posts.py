"""
Do the candidate posts change with time?  Phase-resolved Programme P4.

Infrastructure terms (flow, degree, betweenness, stadium decay) are time-
invariant; only the risk term is phase-specific.  We rebuild per-junction risk
separately for the arrival (pre), occupancy (during) and dispersal (post)
phases from the event cache, re-score eq. (score), and re-select a delta-
separated K-set per phase.  Comparing the three phase-optimal pools shows how
much siting is genuinely temporal, and how large a stable core the fixed
operational pool must contain.

Outputs results/phased_posts.json + figures/fig38_phased_posts.png
"""
import json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = np.load("results/benefits_cache.npz")
cand, clon, clat = B["cand"], B["clon"], B["clat"]
flow, deg, betw, dst = B["flow"], B["deg"], B["betw"], B["dst"]
xyc = B["xy"]
decay = np.exp(-dst / 5000.0)
Nc = len(cand)


def minmax(x):
    x = np.asarray(x, float); r = x.max() - x.min()
    return np.zeros_like(x) if r == 0 else (x - x.min()) / r


C = np.load("results/events_cache.npz", allow_pickle=True)
cfx, cbk, laE, loE, iso = C["fx"], C["bucket"], C["lat"], C["lon"], C["is_ordinary"]
real = ~np.array([bool(x) for x in iso])[cfx]
inwin = (cbk >= 0) & real
tk = cKDTree(np.column_stack([clat, clon]))
_, jnear = tk.query(np.column_stack([laE[inwin], loE[inwin]]))
bk = cbk[inwin]

phase_of = np.where(bk < 10, 0, np.where(bk < 18, 1, 2))  # pre/during/post
PHASES = ["pre (arrival)", "during (occupancy)", "post (dispersal)"]
risk_phase = np.zeros((3, Nc))
for p in range(3):
    m = phase_of == p
    np.add.at(risk_phase[p], jnear[m], 1.0)

pairs = cKDTree(xyc).query_pairs(800.0, output_type="ndarray")
conf = nx.Graph(); conf.add_nodes_from(range(Nc)); conf.add_edges_from(map(tuple, pairs))


def greedy(S, K=14):
    order = np.argsort(-S); blocked = np.zeros(Nc, bool); chosen = []
    for j in order:
        if blocked[j]:
            continue
        chosen.append(int(j))
        if len(chosen) >= K:
            break
        for nb in conf.neighbors(int(j)):
            blocked[nb] = True
    return set(chosen)


def score(riskv):
    return (minmax(riskv) + minmax(flow)) * decay


sets = [greedy(score(risk_phase[p])) for p in range(3)]
union = set().union(*sets)
core = sets[0] & sets[1] & sets[2]


def jac(a, b): return len(a & b) / len(a | b)


overlap = {f"{PHASES[i]} vs {PHASES[j]}": round(jac(sets[i], sets[j]), 3)
           for i in range(3) for j in range(i+1, 3)}
print("phase-pool Jaccard overlaps:", overlap)
print(f"core (all 3 phases): {len(core)} posts; union: {len(union)} posts")

summary = dict(phase_overlap=overlap, n_core=len(core), n_union=len(union),
               core_frac=round(len(core)/14, 3),
               selected={PHASES[p]: sorted(map(int, sets[p])) for p in range(3)})
json.dump(summary, open("results/phased_posts.json", "w"), indent=2)

# map figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
STAD = (153.0093, -27.4647)
for p in range(3):
    ax = axes[p]
    ax.scatter(clon, clat, s=3, color="#dddddd")
    sel = np.array(sorted(sets[p]))
    incore = np.array([j in core for j in sel])
    ax.scatter(clon[sel[~incore]], clat[sel[~incore]], s=120, marker="*",
               color="#C8102E", edgecolor="k", zorder=5, label="phase-specific site")
    ax.scatter(clon[sel[incore]], clat[sel[incore]], s=120, marker="*",
               color="#0092BD", edgecolor="k", zorder=6, label="stable-core site")
    ax.scatter(*STAD, marker="X", s=120, color="black", zorder=7, label="Suncorp Stadium")
    ax.set_title(f"P4 optimum — {PHASES[p]}", fontsize=10)
    ax.set_xlabel("lon")
    if p == 0:
        ax.set_ylabel("lat"); ax.legend(loc="lower left", fontsize=7)
fig.suptitle(f"Phase-resolved candidate-post selection: {len(core)}/14 posts are stable across all phases "
             f"(blue), the rest migrate with the arrival→dispersal peak (red)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("figures/fig38_phased_posts.png", dpi=150, bbox_inches="tight")
print("saved phased_posts.json + fig38")
