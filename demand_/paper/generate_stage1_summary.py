"""
generate_stage1_summary.py
Extract Stage 1 (activity frequency) results from the most recent completed
bandit run and produce paper-ready coefficient tables and convergence plots.

These results can be included in the ABM paper even before the full
stages 2-6 pipeline completes on HPC.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = r"Z:\test_runs_tours\code\runs\sa_bandit_20260622_113611"
OUT_DIR = r"Z:\test_runs_tours\demand_\paper\population_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

print("Extracting Stage 1 activity frequency model results...")

# ── Load coefficients ──────────────────────────────────────────────────────
coef = pd.read_csv(os.path.join(RUN_DIR, "stage1_coefficients.csv"))
print(f"  Coefficient rows: {len(coef)}")
print(f"  Segments: {coef['segment'].unique().tolist()}")

# Filter to significant coefficients (p < 0.05)
sig = coef[pd.to_numeric(coef["p_value"], errors="coerce") < 0.05].copy()
print(f"  Significant (p<0.05): {len(sig)}/{len(coef)}")

# ── Figure: Coefficient forest plot for work segment ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("Stage 1 Activity Frequency Model - Selected Coefficients\n"
             "(Bandit SA-search, HHTS calibration)", fontsize=11, fontweight="bold")

segments = ["work", "edu", "nonmand"]
seg_labels = {"work": "Work Tours", "edu": "Education Tours", "nonmand": "Non-Mandatory Tours"}
colors = {"work": "#E74C3C", "edu": "#2ECC71", "nonmand": "#3498DB"}

for ax, seg in zip(axes, segments):
    seg_data = coef[coef["segment"].str.startswith(seg)].copy()
    if len(seg_data) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue

    # Take most common sub-segment or first group
    if "/" in seg_data["segment"].iloc[0]:
        # Has sub-segment (e.g. work/FullTimeWorker)
        best_subseg = seg_data.groupby("segment")["p_value"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean()).idxmin()
        seg_data = seg_data[seg_data["segment"] == best_subseg]

    seg_data = seg_data.copy()
    seg_data["value"] = pd.to_numeric(seg_data["value"], errors="coerce")
    seg_data["std_err"] = pd.to_numeric(seg_data["std_err"], errors="coerce")
    seg_data["p_value"] = pd.to_numeric(seg_data["p_value"], errors="coerce")
    seg_data = seg_data.dropna(subset=["value", "std_err"]).sort_values("value")

    y = range(len(seg_data))
    sig_mask = seg_data["p_value"] < 0.05
    ax.barh(list(y), seg_data["value"].values,
            xerr=seg_data["std_err"].values,
            color=[colors[seg] if s else "#BDC3C7" for s in sig_mask],
            edgecolor="white", capsize=3, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([p.replace("ptype_","").replace("_"," ") for p in seg_data["param"]],
                       fontsize=8)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"{seg_labels.get(seg, seg)}\n({seg_data['segment'].iloc[0]})", fontsize=10)
    if seg == "work":
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=colors[seg], label="p<0.05"),
                            Patch(color="#BDC3C7", label="p≥0.05")], fontsize=8, framealpha=0.7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "stage1_coefficients.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> stage1_coefficients.png")

# ── Best feature set summary ───────────────────────────────────────────────
best_feats = pd.read_csv(os.path.join(RUN_DIR, "ga_reports", "ga_stage1_best_features.csv"))
feat_summary = best_feats.groupby("group")["feature"].apply(list).reset_index()
feat_summary.to_csv(os.path.join(OUT_DIR, "stage1_best_features.csv"), index=False)
print("  -> stage1_best_features.csv")

# ── GA convergence from history ────────────────────────────────────────────
hist_path = os.path.join(RUN_DIR, "ga_reports", "ga_stage1_segment_history.csv")
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)
    print(f"  History rows: {len(hist)}, columns: {hist.columns.tolist()[:8]}")
    if "generation" in hist.columns and "best_so_far" in hist.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        for seg, grp in hist.groupby("segment"):
            ax.plot(grp["generation"], -grp["best_so_far"],
                    label=seg.split("/")[-1] if "/" in str(seg) else str(seg),
                    alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best BIC/n")
        ax.set_title("Stage 1 SA-Bandit Convergence — Activity Frequency", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.7, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "stage1_convergence.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  -> stage1_convergence.png")

print(f"\nAll Stage 1 outputs -> {OUT_DIR}")
print(f"\nTop Stage 1 features by segment:")
print(feat_summary.to_string(index=False))
