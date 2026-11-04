"""fig35: per-event variation in the optimal deployment, honestly labelled.
A stable core with game-specific, low-value relocations."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = np.load("results/surfaces.npz", allow_pickle=True)
acc_fx = S["acc_fx"]; prank = S["prank"]
kickoffs = [str(x) for x in S["kickoffs"]]; crowds = S["crowds"].astype(float)
F, J, T = acc_fx.shape
PLABEL = [f"P{int(prank[j])}" for j in range(J)]
pf = pd.read_csv("results/deployment_perfixture_sites.csv")
ko_hour = np.array([pd.Timestamp(k).hour for k in kickoffs])

# staffed-buckets matrix (post x fixture), fixtures ordered by kick-off hour
from scipy.optimize import milp  # noqa (ensure env ok)
# rebuild staffed buckets per fixture from schedules movement csv is arc-level;
# recompute quickly: static top-5 vs actual n_posts already in pf. For the matrix
# use realised-surface top staffing intensity: buckets each post is in top-5 by col.
M = np.zeros((J, F))
for f in range(F):
    r = acc_fx[f]
    for t in range(T):
        top = np.argsort(-r[:, t])[:5]
        M[top, f] += 1  # bucket where post is among 5 busiest -> proxy staffing

order = np.argsort(ko_hour + crowds/1e7)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
im = ax.imshow(M[:, order], aspect="auto", cmap="magma")
ax.set_yticks(range(J)); ax.set_yticklabels(PLABEL, fontsize=7)
ax.set_xlabel("42 fixtures, ordered by kick-off hour →")
ax.set_title("(a) Busiest-5 posts per bucket across fixtures\n(bright = consistently top-tier)")
fig.colorbar(im, ax=ax, label="buckets in busiest-5", fraction=0.046)

ax = axes[1]
sc = ax.scatter(crowds, pf.n_posts, c=ko_hour, cmap="twilight", s=45, edgecolor="k", lw=0.4)
ax.set_xlabel("crowd (attendance)"); ax.set_ylabel("distinct posts touched in optimal roster")
ax.set_title("(b) Deployment footprint vs crowd\n(footprint 5–10; weakly tied to size)")
ax.grid(alpha=.3); fig.colorbar(sc, ax=ax, label="kick-off hour")

ax = axes[2]
jit = ko_hour + np.random.default_rng(0).normal(0, 0.08, F)
sc2 = ax.scatter(jit, pf.relocations, c=crowds, cmap="viridis", s=45, edgecolor="k", lw=0.4)
ax.set_xlabel("kick-off hour"); ax.set_ylabel("timed relocations in optimal roster")
ax.set_title("(c) Relocations per fixture\n(present in 39/42 games; worth +0.5 pt mean)")
ax.grid(alpha=.3); fig.colorbar(sc2, ax=ax, label="crowd")

fig.suptitle("Per-event variation in the optimal deployment: a stable core plus game-specific, "
             "low-value timed relocations", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/fig35_temporal_sites.png", dpi=150, bbox_inches="tight")
print("saved fig35")
