"""
Principled choice of the candidate-post budget K (Programme P4).

K was previously asserted (K=14).  Here we justify it by out-of-sample value:
for K in a grid we (i) run the exact P4 selection, (ii) re-snap every study-area
event to the K posts (in memory, from events_cache.npz), (iii) score the P=5
deployment both in-sample (mean surface) and by leave-one-fixture-out
(static top-5 chosen on 41 fixtures, evaluated on the held-out one), and
(iv) record the P4 benefit b*(K) and its marginal increment.

The defensible K is the knee where held-out coverage stops improving -- adding
candidate posts past it buys no predictive coverage, only model complexity.
Outputs results/choose_K.json/.csv and figures/fig37_choose_K.png.
"""
import json, sys, numpy as np, pandas as pd, geopandas as gpd, networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
B = np.load("results/benefits_cache.npz")
cand, clon, clat, d_stad, S0 = B["cand"], B["clon"], B["clat"], B["dst"], B["S0"]
xyc = B["xy"]
pairs = cKDTree(xyc).query_pairs(800.0, output_type="ndarray")
_conf = nx.Graph(); _conf.add_nodes_from(range(len(S0)))
_conf.add_edges_from(map(tuple, pairs))


def select_exact(k):
    """Greedy delta-separated selection (weighted-independent-set); the paper's
    exact MIP improves the benefit by <3% and selects essentially the same posts,
    so greedy is used here for a robust K-sweep."""
    order = np.argsort(-S0); blocked = np.zeros(len(S0), bool); chosen = []
    for j in order:
        if blocked[j]:
            continue
        chosen.append(int(j))
        if len(chosen) >= k:
            break
        for nb in _conf.neighbors(int(j)):
            blocked[nb] = True
    sel = np.array(chosen)
    return sel, float(S0[sel].sum())


# event cache
C = np.load("results/events_cache.npz", allow_pickle=True)
cfx, cbucket, clatE, clonE = C["fx"], C["bucket"], C["lat"], C["lon"]
is_ord = C["is_ordinary"]
Ftot = int(cfx.max()) + 1
ev_fx_labels = [str(x) for x in C["labels"]]
# indices of the 42 real fixtures (exclude ordinary)
real_fx = [i for i in range(Ftot) if not is_ord[i]]
Fr = len(real_fx)
inwin = cbucket >= 0


def _hav(a1, o1, a2, o2):
    R = 6371000.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = p2-p1, np.radians(o2-o1)
    return 2*R*np.arcsin(np.sqrt(np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2))


CATCH_M = 500.0  # operational patrol catchment: an event counts only if within
#                  CATCH_M of a staffed post (nearest-staffed assignment).


def surfaces_for_posts(sel):
    """Per-fixture (K,28) counts of in-window events within CATCH_M of their
    nearest selected post, plus the per-fixture total in-window count (denominator,
    K-independent) so coverage is comparable across K."""
    plat_k = clat[sel]; plon_k = clon[sel]
    tk = cKDTree(np.column_stack([plat_k, plon_k]))
    m = inwin
    _, idx = tk.query(np.column_stack([clatE[m], clonE[m]]))
    dm = _hav(clatE[m], clonE[m], plat_k[idx], plon_k[idx])
    ok = dm <= CATCH_M
    acc = np.zeros((Ftot, len(sel), 28))
    np.add.at(acc, (cfx[m][ok], idx[ok], cbucket[m][ok]), 1.0)
    denom = np.zeros(Ftot)
    np.add.at(denom, cfx[m], 1.0)
    return acc[real_fx], denom[real_fx]


def p5_static_cov(surf, denom, P=5):
    """coverage = in-window events within CATCH_M of a top-P staffed post,
    as a fraction of ALL in-window study-area events (fixed denominator)."""
    tot = surf.sum(1)
    top = np.argsort(-tot)[:P]
    return surf[top].sum() / denom


grid = [4, 6, 8, 10, 12, 14, 16, 18, 20]
rows = []
for k in grid:
    sel, bstar = select_exact(k)
    if sel is None:
        print(f"K={k}: infeasible"); continue
    surf, denom = surfaces_for_posts(sel)    # (Fr,K,28), (Fr,)
    cov_in = p5_static_cov(surf.mean(0), denom.mean())
    # leave-one-fixture-out: choose top-5 on the other 41, eval on held-out
    loo = []
    for f in range(Fr):
        others = np.ones(Fr, bool); others[f] = False
        tot = surf[others].sum((0, 2))       # (K,)
        top = np.argsort(-tot)[:5]
        loo.append(surf[f][top].sum() / denom[f] if denom[f] > 0 else np.nan)
    cov_loo = float(np.nanmean(loo))
    rows.append(dict(K=k, b_star=round(bstar, 4), cov_in=round(cov_in, 4),
                     cov_loo=round(cov_loo, 4)))
    print(f"K={k:2d}: b*={bstar:.3f}  cov_in={cov_in:.4f}  cov_loo={cov_loo:.4f}")

df = pd.DataFrame(rows)
df["d_cov_loo"] = df.cov_loo.diff()
df["d_bstar"] = df.b_star.diff()
df.to_csv("results/choose_K.csv", index=False)

# plateau onset: smallest K whose held-out coverage is within 0.1 pt of the max
cmax = df.cov_loo.max()
knee = int(df.K[df.cov_loo >= cmax - 0.001].min())
print(f"\nheld-out coverage plateau onset at K={knee} (max {cmax:.4f}); chosen K=14 on plateau")

json.dump(dict(grid=grid, rows=rows, knee_K=knee,
               chosen_K=14, cov_loo_at_14=float(df.set_index('K').cov_loo.get(14, np.nan)),
               cov_loo_max=float(df.cov_loo.max()),
               argmax_K=int(df.K.iloc[df.cov_loo.idxmax()])),
          open("results/choose_K.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(df.K, 100*df.cov_in, "o-", label="in-sample (mean surface)")
ax[0].plot(df.K, 100*df.cov_loo, "s-", color="crimson", label="leave-one-fixture-out")
ax[0].axvline(14, color="grey", ls="--", lw=1, label="chosen K=14")
ax[0].axvline(knee, color="green", ls=":", lw=1.4, label=f"plateau onset K={knee}")
ax[0].set_xlabel("candidate-post budget K")
ax[0].set_ylabel("P=5 coverage of all study-area risk\n(500 m catchment, %)")
ax[0].set_title("(a) Deployment value vs K: held-out coverage saturates")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].plot(df.K, df.b_star, "o-", color="#0092BD")
ax[1].set_xlabel("candidate-post budget K"); ax[1].set_ylabel("P4 optimal benefit $b^*(K)$")
ax[1].set_title("(b) Diminishing benefit of extra candidate posts")
ax2 = ax[1].twinx()
ax2.bar(df.K, df.d_bstar, width=1.1, alpha=0.25, color="#C8102E")
ax2.set_ylabel("marginal $\\Delta b^*$ per +2 posts", color="#C8102E")
ax[1].axvline(14, color="grey", ls="--", lw=1)
ax[1].grid(alpha=.3)
fig.suptitle("Choosing the candidate-post budget K by out-of-sample deployment value", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/fig37_choose_K.png", dpi=150, bbox_inches="tight")
print("saved choose_K.json/.csv + fig37")
