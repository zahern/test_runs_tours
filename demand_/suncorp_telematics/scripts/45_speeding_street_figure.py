"""
45_speeding_street_figure.py - Fig3 (fig:strisk) remodel on SPEEDING-ONLY.

Review note: fig:strisk plotted braking+speeding EB "risk" (script 20).
The enforcement estimand is speeding: expected finable-speeding
exceedances (>=10 kph over posted limit) per segment per DAY,
event vs matched control. Rebuild both panels from the speeding-only
fix-job outputs (no refit):

  Top:    per-day EB speeding, event vs control scatter + 1:1 line.
  Bottom: per-day EB uplift (event - control) distribution.

Per-day denominators: 16 event days, 14 control days (ref_ids in
segment_roads_part0_5000.csv.gz).

Inputs:
  results/speeding_timewise_segment_phase.csv
  data/segment_roads_part0_5000.csv.gz  (road names)

Outputs:
  figures/speeding_street_eb_event_control.png
  figures/speeding_street_uplift_dist.png

Usage:
  python 45_speeding_street_figure.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJ = r"Z:\test_runs_tours\demand_\suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

SEG_PHASE = os.path.join(RES, "speeding_timewise_segment_phase.csv")
SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
FIG_SCATTER = os.path.join(FIG, "speeding_street_eb_event_control.png")
FIG_UPLIFT = os.path.join(FIG, "speeding_street_uplift_dist.png")

N_EVENT_DAYS = 16
N_CONTROL_DAYS = 14

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def main():
    print("=" * 80)
    print("45_speeding_street_figure.py - speeding-only Fig3")
    print("=" * 80)
    seg = pd.read_csv(SEG_PHASE)
    ev = seg[seg["type"] == "Event"].groupby("road_osmid")["eb"].sum()
    ct = seg[seg["type"] == "Control"].groupby("road_osmid")["eb"].sum()
    both = pd.DataFrame({"eb_event": ev, "eb_control": ct})
    both["ev_day"] = both["eb_event"] / N_EVENT_DAYS
    both["ct_day"] = both["eb_control"] / N_CONTROL_DAYS
    both["uplift_day"] = both["ev_day"].fillna(0) - both["ct_day"].fillna(0)
    print(f"   segments: event={ev.count():,} control={ct.count():,} "
          f"both={(~both[['eb_event', 'eb_control']].isna()).all(axis=1).sum():,}")
    print(f"   total EB/day: event={both['ev_day'].sum():,.1f} "
          f"control={both['ct_day'].sum():,.1f}")

    names = pd.read_csv(SEGMENT_ROADS, usecols=["road_osmid", "road_name"])
    names = names.groupby("road_osmid")["road_name"].first()
    ranked = both.join(names).sort_values("ev_day", ascending=False)
    print("   top 10 by expected speeding/day (event):")
    for i, (oid, r) in enumerate(ranked.head(10).iterrows(), 1):
        nm = str(r["road_name"]) if pd.notna(r["road_name"]) else "(unnamed)"
        print(f"   {i:>2} {oid:<12} {nm[:28]:28s} {r['ev_day']:8.1f}/day "
              f"uplift {r['uplift_day']:+8.1f}/day")
    ranked.head(20).to_csv(os.path.join(RES, "speeding_street_top20.csv"))
    both.to_csv(os.path.join(RES, "speeding_street_perday.csv"))

    plot = both.dropna(subset=["ev_day", "ct_day"]).copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot["ct_day"], plot["ev_day"], alpha=0.4, s=20)
    lim = max(plot["ev_day"].max(), plot["ct_day"].max())
    ax.plot([0, lim], [0, lim], "r--", lw=2, label="1:1 line")
    ax.set_xlabel("Control expected speeding / day (EB, >=10 kph over limit)")
    ax.set_ylabel("Event expected speeding / day (EB, >=10 kph over limit)")
    ax.set_title("Expected speeding per day: event vs matched control (7,517 segments)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_SCATTER, dpi=150)
    plt.close(fig)
    print(f"   Saved {FIG_SCATTER}")

    fig, ax = plt.subplots(figsize=(8, 5))
    up = both["uplift_day"].replace([np.inf, -np.inf], np.nan).dropna()
    up = up[up != 0]
    ax.hist(up, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero uplift")
    ax.set_xlabel("EB uplift per day (event - control, speeding events)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of per-day speeding uplift (positive = needs incremental staffing)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_UPLIFT, dpi=150)
    plt.close(fig)
    print(f"   Saved {FIG_UPLIFT}")
    print("DONE!")


if __name__ == "__main__":
    main()
