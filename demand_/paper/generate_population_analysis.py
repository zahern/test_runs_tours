"""
generate_population_analysis.py
Produce paper-ready figures and tables from the generated synthetic population.

Runs locally (no HPC/JAX/larch needed) — uses only pandas/numpy/matplotlib
and the skim/landuse data already built.

Outputs → demand_/paper/population_analysis/
  population_demographics.png    — ptype + age + income + vehicle ownership
  zone_safety_exposure.png       — crash rate by SA2 zone (choropleth-style)
  safety_by_ptype.png            — ECF exposure distribution by person type
  mode_availability.png          — walk/bike availability at Level4 resolution
  population_summary.csv         — key headline statistics
  zone_exposure_table.csv        — per-zone population × crash-rate metrics
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

warnings.filterwarnings("ignore")

DEMAND_DIR   = r"Z:\test_runs_tours\demand_"
NEW_POP_DIR  = r"Z:\test_runs_tours\new_pop"
OUT_DIR = os.path.join(DEMAND_DIR, "paper", "population_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
PTYPE_COLORS = {
    "FullTimeWorker":  "#E74C3C",
    "PartTimeWorker":  "#E67E22",
    "JointWorkStudy":  "#F1C40F",
    "School":          "#2ECC71",
    "Student":         "#27AE60",
    "NonWorker":       "#3498DB",
    "Preschool":       "#9B59B6",
    "Retired":         "#95A5A6",
}

STYLE = dict(fontsize=10, fontweight="bold")

print("Loading synthetic population…")
persons = pd.read_csv(os.path.join(NEW_POP_DIR, "saved_data", "synth_persons.csv"),
                      low_memory=False)
print(f"  {len(persons):,} persons | {persons['OTAZ'].nunique()} zones")

# ── Figure 1: DEMOGRAPHICS ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("SEQ Synthetic Population Demographics (324-zone SA2 Scope)",
             fontsize=13, fontweight="bold")

# 1a — Person type distribution
ax = axes[0, 0]
ptypes = persons["ptype"].value_counts()
labels = [p.replace("Worker","Worker\n").replace("Study","Study\n") for p in ptypes.index]
colors = [PTYPE_COLORS.get(p, "#BDC3C7") for p in ptypes.index]
bars = ax.barh(range(len(ptypes)), ptypes.values / 1e6, color=colors, edgecolor="white")
ax.set_yticks(range(len(ptypes)))
ax.set_yticklabels([p.replace("Worker", " Worker").replace("Study", " Study")
                    for p in ptypes.index], fontsize=9)
ax.set_xlabel("Persons (millions)")
ax.set_title("Person Type", **STYLE)
for i, (v, b) in enumerate(zip(ptypes.values, bars)):
    ax.text(b.get_width() + 0.002, i, f"{v/len(persons)*100:.1f}%", va="center", fontsize=8)

# 1b — Age distribution (age5p 1-18)
ax = axes[0, 1]
age_vals = pd.to_numeric(persons["age5p"], errors="coerce").dropna()
age_labels = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
              "40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"]
age_counts = age_vals.value_counts().sort_index()
ax.bar(age_counts.index - 0.5, age_counts.values / 1e3, color="#3498DB", edgecolor="white")
ax.set_xticks(range(1, 19, 2))
ax.set_xticklabels(age_labels[::2], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Age Group")
ax.set_ylabel("Persons (000s)")
ax.set_title("Age Distribution", **STYLE)

# 1c — Vehicle ownership
ax = axes[1, 0]
vehrd = pd.to_numeric(persons["vehrd"], errors="coerce").fillna(0).clip(0, 4).astype(int)
veh_counts = vehrd.value_counts().sort_index()
veh_labels = {0: "0\n(no car)", 1: "1", 2: "2", 3: "3", 4: "4+"}
ax.bar(range(len(veh_counts)), veh_counts.values / 1e3,
       color=["#E74C3C", "#E67E22", "#2ECC71", "#3498DB", "#9B59B6"][:len(veh_counts)],
       edgecolor="white")
ax.set_xticks(range(len(veh_counts)))
ax.set_xticklabels([veh_labels.get(k, str(k)) for k in veh_counts.index])
ax.set_xlabel("Household vehicles")
ax.set_ylabel("Persons (000s)")
ax.set_title("Vehicle Ownership", **STYLE)
for i, v in enumerate(veh_counts.values):
    ax.text(i, v / 1e3 + 1, f"{v/len(persons)*100:.1f}%", ha="center", fontsize=8)

# 1d — Income distribution (incp 1-19)
ax = axes[1, 1]
incp = pd.to_numeric(persons["incp"], errors="coerce").dropna()
incp_counts = incp.value_counts().sort_index()
incp_labels = {1:"<$150",2:"$150-",3:"$250-",4:"$350-",5:"$450-",6:"$550-",
               7:"$650-",8:"$800-",9:"$1000-",10:"$1250-",11:"$1500-",12:"$1750-",
               13:"$2000-",14:"$2500-",15:"$3000-",16:"$3500-",17:"$4000+",18:"N/A",19:"N/A"}
ax.bar(incp_counts.index, incp_counts.values / 1e3,
       color="#3498DB", edgecolor="white", alpha=0.8)
ax.axvline(6, color="#E74C3C", linestyle="--", alpha=0.6, label="$600/wk")
ax.axvline(11, color="#E67E22", linestyle="--", alpha=0.6, label="$1500/wk")
ax.set_xlabel("ABS income code (weekly)")
ax.set_ylabel("Persons (000s)")
ax.set_title("Personal Income Distribution", **STYLE)
ax.legend(fontsize=8, framealpha=0.7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "population_demographics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> population_demographics.png")

# ── Figure 2: ZONE-LEVEL SAFETY EXPOSURE ───────────────────────────────────
print("Computing zone-level crash exposure…")
land = pd.read_csv(os.path.join(NEW_POP_DIR, "saved_data", "zone_landuse_507.csv"))
pop_by_zone = persons.groupby("OTAZ").size().reset_index(name="population")
pop_by_zone.columns = ["SA2_MAIN16", "population"]

# Load crash OD matrix, compute origin-zone average crash rate
with h5py.File(os.path.join(DEMAND_DIR, "crash_matrices_507.omx"), "r") as f:
    zone_ids = [int(z.decode()) for z in f["lookup"]["SA2_CODE"][...]]
    crash_rate = np.array(f["data"]["crash_rate_total"], dtype=np.float32)

# Mean crash rate along ALL routes FROM each origin zone (row mean, excluding diagonal)
np.fill_diagonal(crash_rate, np.nan)
mean_crash_by_origin = np.nanmean(crash_rate, axis=1)
crash_df = pd.DataFrame({"SA2_MAIN16": zone_ids, "mean_outbound_crash_rate": mean_crash_by_origin})
crash_df = crash_df.merge(pop_by_zone, on="SA2_MAIN16", how="left")
crash_df["population"] = crash_df["population"].fillna(0)

# Add near-miss total (324-zone scope — align to crash_df by SA2 code)
try:
    with h5py.File(os.path.join(DEMAND_DIR, "nearmiss_ecf_skim_507.omx"), "r") as f:
        nm_zone_ids = [int(z.decode()) for z in f["lookup"]["SA2_CODE"][...]]
        nm_rate = np.array(f["data"]["nm_per_km_total"], dtype=np.float32)
    np.fill_diagonal(nm_rate, np.nan)
    nm_mean = np.nanmean(nm_rate, axis=1)
    nm_df = pd.DataFrame({"SA2_MAIN16": nm_zone_ids, "mean_nm_per_km": nm_mean})
    crash_df = crash_df.merge(nm_df, on="SA2_MAIN16", how="left")
except Exception as e:
    print(f"  [warn] near-miss skim: {e}")
    crash_df["mean_nm_per_km"] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Zone-Level Safety Exposure — SEQ Synthetic Population", fontsize=12, fontweight="bold")

# 2a — Crash rate distribution across zones
ax = axes[0]
valid = crash_df["mean_outbound_crash_rate"].dropna()
ax.hist(valid, bins=40, color="#E74C3C", edgecolor="white", alpha=0.8)
ax.axvline(valid.mean(), color="black", linestyle="--", label=f"Mean: {valid.mean():.3f}")
ax.set_xlabel("Mean outbound crash rate (crashes/km)")
ax.set_ylabel("Number of SA2 zones")
ax.set_title("Outbound Crash Rate Distribution", **STYLE)
ax.legend(fontsize=9)

# 2b — Population exposure by crash rate quartile
ax = axes[1]
rate = crash_df["mean_outbound_crash_rate"].fillna(0)
q1, q3 = rate.quantile(0.25), rate.quantile(0.75)
crash_df["crash_band"] = pd.cut(rate, bins=[-np.inf, q1, q3, np.inf],
                                labels=["Low (<Q1)", "Mid (Q1-Q3)", "High (>Q3)"])
crash_df["crash_quartile"] = crash_df["crash_band"]
pop_by_q = crash_df.groupby("crash_band", observed=True)["population"].sum()
colors = ["#2ECC71", "#F1C40F", "#E74C3C"]
bars = ax.bar(range(len(pop_by_q)), pop_by_q.values / 1e6, color=colors[:len(pop_by_q)], edgecolor="white")
ax.set_xticks(range(len(pop_by_q)))
ax.set_xticklabels(pop_by_q.index.tolist())
ax.set_xlabel("Crash exposure band")
ax.set_ylabel("Population (millions)")
ax.set_title("Population by Crash Exposure", **STYLE)
for i, (v, b) in enumerate(zip(pop_by_q.values, bars)):
    ax.text(i, b.get_height() + 0.01, f"{v/1e6:.2f}M", ha="center", fontsize=9)

# 2c — Crash rate vs population scatter
ax = axes[2]
plot_df = crash_df[crash_df["population"] > 100].copy()
sc = ax.scatter(plot_df["mean_outbound_crash_rate"], plot_df["population"] / 1e3,
                alpha=0.6, c=plot_df["mean_outbound_crash_rate"],
                cmap="RdYlGn_r", s=30, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Crash rate (crashes/km)")
ax.set_xlabel("Mean outbound crash rate")
ax.set_ylabel("Zone population (000s)")
ax.set_title("Crash Rate vs Zone Population", **STYLE)
corr = plot_df["mean_outbound_crash_rate"].corr(plot_df["population"])
ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=9,
        va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "zone_safety_exposure.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> zone_safety_exposure.png")

# ── Figure 3: SAFETY EXPOSURE BY PERSON TYPE ───────────────────────────────
print("Computing safety exposure by person type…")
# Merge crash rate onto persons via home zone
persons_exp = persons.merge(crash_df[["SA2_MAIN16", "mean_outbound_crash_rate", "mean_nm_per_km"]],
                            left_on="OTAZ", right_on="SA2_MAIN16", how="left")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Safety Exposure by Person Type", fontsize=12, fontweight="bold")

# 3a — Boxplot: crash rate by ptype
ax = axes[0]
ptypes_ord = ["FullTimeWorker", "PartTimeWorker", "JointWorkStudy",
              "School", "Student", "NonWorker", "Preschool", "Retired"]
data = [persons_exp[persons_exp["ptype"] == p]["mean_outbound_crash_rate"].dropna().values
        for p in ptypes_ord]
labels_short = ["FT-Work","PT-Work","Work+Study","School","Student","NonWork","Preschool","Retired"]
bp = ax.boxplot(data, labels=labels_short, patch_artist=True, vert=True,
                medianprops=dict(color="black", linewidth=2))
for patch, p in zip(bp["boxes"], ptypes_ord):
    patch.set_facecolor(PTYPE_COLORS.get(p, "#BDC3C7"))
    patch.set_alpha(0.8)
ax.set_xticklabels(labels_short, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Mean outbound crash rate (crashes/km)")
ax.set_title("Crash Exposure by Person Type", **STYLE)
ax.grid(axis="y", alpha=0.3)

# 3b — Mean crash rate comparison
ax = axes[1]
mean_by_ptype = persons_exp.groupby("ptype")["mean_outbound_crash_rate"].mean().reindex(ptypes_ord)
mean_by_ptype_nm = persons_exp.groupby("ptype")["mean_nm_per_km"].mean().reindex(ptypes_ord)
x = np.arange(len(ptypes_ord))
w = 0.4
bars1 = ax.bar(x - w/2, mean_by_ptype.values, width=w, label="Crash rate",
               color=[PTYPE_COLORS.get(p, "#BDC3C7") for p in ptypes_ord],
               alpha=0.9, edgecolor="white")
ax2 = ax.twinx()
ax2.bar(x + w/2, mean_by_ptype_nm.values, width=w, label="Near-miss rate",
        color="lightblue", alpha=0.7, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(labels_short, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Mean crash rate", color="#E74C3C")
ax2.set_ylabel("Mean near-miss/km rate", color="steelblue")
ax.set_title("Mean Safety Exposure by Person Type", **STYLE)
lines1, _ = ax.get_legend_handles_labels()
lines2, _ = ax2.get_legend_handles_labels()
ax.legend(lines1 + [plt.Rectangle((0,0),1,1,color="lightblue")],
          ["Crash rate", "Near-miss rate"], fontsize=9, loc="upper right")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "safety_by_ptype.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> safety_by_ptype.png")

# ── Figure 4: WALK/BIKE AVAILABILITY ───────────────────────────────────────
print("Computing walk/bike availability distribution…")
# Load Level4 walk/bike skim and check distribution of distances for short trips
WALK_MAX_KM = 1.5
BIKE_MAX_KM = 8.0
try:
    with h5py.File(os.path.join(DEMAND_DIR, "walk_bike_lvl4_to_sa2_skims.h5"), "r") as f:
        lvl4_ids  = f["lvl4_ids"][...]
        sa2_codes = f["sa2_codes"][...]
        distwalk  = np.array(f["DISTWALK"], dtype=np.float32)
        distbike  = np.array(f["DISTBIKE"],  dtype=np.float32)

    # For each Level4 origin: fraction of destination SA2s reachable by walk and bike
    walk_reachable = (distwalk > 0) & (distwalk <= WALK_MAX_KM)
    bike_reachable = (distbike > 0) & (distbike <= BIKE_MAX_KM)
    pct_walk = walk_reachable.mean(axis=1) * 100
    pct_bike = bike_reachable.mean(axis=1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Active Mode Destination Availability at Level 4 Micro-Zone Resolution\n"
                 f"(Walk ≤{WALK_MAX_KM}km, Bike ≤{BIKE_MAX_KM}km)", fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.hist(pct_walk[pct_walk > 0], bins=50, color="#2ECC71", edgecolor="white", alpha=0.8)
    ax.set_xlabel("% of SEQ SA2 destinations reachable by walk")
    ax.set_ylabel("Number of Level4 micro-zones")
    ax.set_title("Walk Destination Availability", **STYLE)
    ax.axvline(np.median(pct_walk[pct_walk > 0]), color="darkgreen", linestyle="--",
               label=f"Median: {np.median(pct_walk[pct_walk > 0]):.1f}%")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(pct_bike[pct_bike > 0], bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.set_xlabel("% of SEQ SA2 destinations reachable by bike")
    ax.set_ylabel("Number of Level4 micro-zones")
    ax.set_title("Bike Destination Availability", **STYLE)
    ax.axvline(np.median(pct_bike[pct_bike > 0]), color="darkblue", linestyle="--",
               label=f"Median: {np.median(pct_bike[pct_bike > 0]):.1f}%")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mode_availability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> mode_availability.png")
except Exception as e:
    print(f"  [skip] mode_availability: {e}")

# ── Table: POPULATION SUMMARY ───────────────────────────────────────────────
print("Writing summary tables…")
summary = {
    "total_persons":       len(persons),
    "total_zones":         persons["OTAZ"].nunique(),
    "ptype_FullTimeWorker": (persons["ptype"] == "FullTimeWorker").sum(),
    "ptype_PartTimeWorker": (persons["ptype"] == "PartTimeWorker").sum(),
    "ptype_School":         (persons["ptype"] == "School").sum(),
    "ptype_NonWorker":      (persons["ptype"] == "NonWorker").sum(),
    "ptype_other":          persons["ptype"].isin(["JointWorkStudy","Student","Preschool","Retired"]).sum(),
    "pct_zero_vehicle":     (pd.to_numeric(persons["vehrd"], errors="coerce").fillna(0) == 0).mean() * 100,
    "mean_vehicles_per_person": pd.to_numeric(persons["vehrd"], errors="coerce").fillna(0).mean(),
    "pct_full_licence":     (pd.to_numeric(persons.get("CARLICENCE", pd.Series(0)), errors="coerce").fillna(0) > 0).mean() * 100,
    "zones_with_pop":       (pop_by_zone["population"] > 0).sum(),
    "zones_no_pop":         (pop_by_zone["population"] == 0).sum(),
    "mean_pop_per_zone":    pop_by_zone["population"].mean(),
    "median_zone_crash_rate": crash_df["mean_outbound_crash_rate"].median(),
    "pct_pop_high_crash": (crash_df[crash_df["crash_band"] == "High (>Q3)"]["population"].sum() /
                           crash_df["population"].sum() * 100) if "crash_band" in crash_df.columns else None,
}
pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(
    os.path.join(OUT_DIR, "population_summary.csv"))

# Zone exposure table
crash_df[["SA2_MAIN16", "population", "mean_outbound_crash_rate", "mean_nm_per_km",
           "crash_quartile"]].to_csv(os.path.join(OUT_DIR, "zone_exposure_table.csv"), index=False)

print("\n=== SUMMARY ===")
for k, v in list(summary.items())[:10]:
    print(f"  {k}: {v:,.1f}" if isinstance(v, float) else f"  {k}: {v:,}")
print(f"\nAll outputs written to: {OUT_DIR}")
