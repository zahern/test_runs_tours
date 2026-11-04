#!/usr/bin/env python
"""Poster panel from validate_safety_arms.py holdout metrics."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUT = {"navy": "#003B71", "sky": "#0077C8", "amber": "#F2A900",
       "green": "#3E9B4F", "red": "#C0392B", "grey": "#5C6773"}

CSV = r"Z:\test_runs_tours\code\runs\safety_validation_full\safety_validation_metrics.csv"
OUT = r"Z:\test_runs_tours\abm_safety\preprint\figures\poster_safety_validation.png"

df = pd.read_csv(CSV)
df = df[df.arm.isin(["M_S", "M_0_ablation"])].copy()
segs = ["edu", "nonmand"]
labels = {"edu": "Education", "nonmand": "Non-mandatory"}
colors = {"M_S": QUT["navy"], "M_0_ablation": QUT["red"]}
arm_labels = {"M_S": "With safety (M$_S$)", "M_0_ablation": "Without safety (M$_0$)"}


def val(seg, arm, col):
    r = df[(df.segment == seg) & (df.arm == arm)]
    return float(r.iloc[0][col]) if not r.empty else np.nan


fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
x = np.arange(len(segs))
w = 0.32

# --- A: destination-share divergence (lower is better) ---
ax = axes[0]
m0 = [val(s, "M_0_ablation", "dest_js") for s in segs]
ms = [val(s, "M_S", "dest_js") for s in segs]
ax.bar(x - w / 2, m0, w, color=colors["M_0_ablation"], alpha=0.9,
       label=arm_labels["M_0_ablation"])
ax.bar(x + w / 2, ms, w, color=colors["M_S"], alpha=0.95,
       label=arm_labels["M_S"])
ax.axhline(0, color=QUT["grey"], ls="--", lw=1.1)
ax.set_xticks(x); ax.set_xticklabels([labels[s] for s in segs], fontsize=9)
ax.set_ylabel("Jensen–Shannon divergence", fontsize=9)
ax.set_title("Destination-share divergence\n(lower = closer to observed)",
             fontsize=10.5, color=QUT["navy"], fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
for i, (a, b) in enumerate(zip(m0, ms)):
    ax.text(i - w / 2, a + 0.012, f"{a:.2f}", ha="center", fontsize=7.5)
    ax.text(i + w / 2, b + 0.012, f"{b:.2f}", ha="center", fontsize=7.5)

# --- B: top-25 overlap (higher is better) ---
ax = axes[1]
m0 = [val(s, "M_0_ablation", "dest_top25") for s in segs]
ms = [val(s, "M_S", "dest_top25") for s in segs]
ax.bar(x - w / 2, m0, w, color=colors["M_0_ablation"], alpha=0.9)
ax.bar(x + w / 2, ms, w, color=colors["M_S"], alpha=0.95)
ax.axhline(1.0, color=QUT["grey"], ls="--", lw=1.1)
ax.set_ylim(0, 1.08)
ax.set_xticks(x); ax.set_xticklabels([labels[s] for s in segs], fontsize=9)
ax.set_ylabel("Top-25 overlap", fontsize=9)
ax.set_title("Top-25 destination overlap\n(higher = closer to observed)",
             fontsize=10.5, color=QUT["navy"], fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for i, (a, b) in enumerate(zip(m0, ms)):
    ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7.5)
    ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7.5)

# --- C: chosen-destination crash exposure (lower is better) ---
ax = axes[2]
x3 = np.arange(len(segs))
w3 = 0.24
obs = [val(s, "M_S", "crash_obs_mean") for s in segs]
m0 = [val(s, "M_0_ablation", "crash_exp_mean") for s in segs]
ms = [val(s, "M_S", "crash_exp_mean") for s in segs]
ax.bar(x3 - w3, obs, w3, color=QUT["grey"], label="Observed HHTS")
ax.bar(x3, m0, w3, color=colors["M_0_ablation"], alpha=0.9,
       label=arm_labels["M_0_ablation"])
ax.bar(x3 + w3, ms, w3, color=colors["M_S"], alpha=0.95,
       label=arm_labels["M_S"])
ax.set_xticks(x3); ax.set_xticklabels([labels[s] for s in segs], fontsize=9)
ax.set_ylabel("Mean OD crash exposure (log scale)", fontsize=9)
ax.set_yscale("log")
ax.set_title("Chosen-destination crash exposure\n(lower = closer to observed)",
             fontsize=10.5, color=QUT["navy"], fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7.0, frameon=False, loc="upper left")
for i, (o, a, b) in enumerate(zip(obs, m0, ms)):
    ax.text(i - w3, o + 3, f"{o:.0f}", ha="center", fontsize=7.0)
    ax.text(i, a + 3, f"{a:.0f}", ha="center", fontsize=7.0)
    ax.text(i + w3, b + 3, f"{b:.0f}", ha="center", fontsize=7.0)

fig.suptitle("Reality check: adding fitted safety terms moves destinations "
             "closer to observed HHTS", fontsize=13.5, fontweight="bold",
             color=QUT["navy"])
fig.text(0.5, 0.005,
         "30% person-level HHTS holdout: education n=1,019 tours; non-mandatory n=3,441 tours. "
         "Mean exposure is heavy-tailed; log scale. "
         "M$_0$ = same fitted model with crash coefficients set to zero.",
         ha="center", fontsize=7.6, color=QUT["grey"])
fig.tight_layout(rect=(0, 0.035, 1, 0.95))
fig.savefig(OUT, dpi=220, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)
