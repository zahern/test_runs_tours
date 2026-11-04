"""
44_speeding_relative_hours.py - Fig6-bottom remodel on RELATIVE hours.

Review note on main.tex: the hourly speeding panel (fig:timewise, bottom)
plotted mean speeding by CLOCK hour of day. Fixtures kick off 14:00-20:00,
so clock hours conflate pre-arrival of a late game with dispersal of an
early one. Rebuild the curve on hours RELATIVE to kickoff (KO-3h .. KO+4h,
hourly bins -3..+3), which is the paper's deployment frame: every fixture
contributes to every bin by construction, so there are no unobserved hours.

Method mirrors 42_speeding_timewise_fix.build_hourly (uniform rate within
each phase window) but with kickoff-invariant relative bins:
  pre    [-3.0,-0.5): bins -3:1.0, -2:1.0, -1:0.5  (duration 2.5h)
  during [-0.5,+1.5): bins -1:0.5,  0:1.0, +1:0.5  (duration 2.0h)
  post   [+1.5,+4.0): bins +1:0.5, +2:1.0, +3:1.0  (duration 2.5h)
Mean per contributing window per bin; Event windows only (as before).

Inputs (no refit -- reuses fix-job outputs):
  results/speeding_timewise_segment_phase.csv  (per segment x type x phase EB)
  data/segment_roads_part0_5000.csv.gz         (contributing-window counts)

Outputs:
  results/speeding_timewise_hourly_relative.csv
  figures/speeding_timewise_hourly.png  (OVERWRITTEN with relative-hour axis)

Usage:
  python 44_speeding_relative_hours.py
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
OUT_CSV = os.path.join(RES, "speeding_timewise_hourly_relative.csv")
FIG_HOURLY = os.path.join(FIG, "speeding_timewise_hourly.png")

# Relative-hour bins [k, k+1) and per-phase overlap weights (uniform rate).
BINS = [-3, -2, -1, 0, 1, 2, 3]
OVERLAP = {
    "pre": {-3: 1.0, -2: 1.0, -1: 0.5},
    "during": {-1: 0.5, 0: 1.0, 1: 0.5},
    "post": {1: 0.5, 2: 1.0, 3: 1.0},
}
DURATION = {"pre": 2.5, "during": 2.0, "post": 2.5}

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def main():
    print("=" * 80)
    print("44_speeding_relative_hours.py - relative-hour Fig6-bottom")
    print("=" * 80)
    seg = pd.read_csv(SEG_PHASE)
    seg = seg[seg["type"] == "Event"].copy()
    print(f"   segment-phase rows (Event): {len(seg):,}")

    print("   counting contributing windows per (segment, phase)...")
    st = pd.read_csv(SEGMENT_ROADS, usecols=["type", "phase", "road_osmid", "trips"])
    st = st[(st["type"] == "Event") & (st["trips"] > 0)]
    nwin = st.groupby(["road_osmid", "phase"]).size().rename("n_win").reset_index()
    print(f"   contributing (segment, phase) windows: {len(nwin):,}")

    m = seg.merge(nwin, on=["road_osmid", "phase"], how="left")
    m["n_win"] = m["n_win"].fillna(0).astype(int)
    m = m[m["n_win"] > 0].copy()
    print(f"   matched: {len(m):,} rows")

    mass = {k: 0.0 for k in BINS}
    denom = {k: 0.0 for k in BINS}
    for _, r in m.iterrows():
        per_win = float(r["eb"]) / int(r["n_win"])
        dur = DURATION[r["phase"]]
        for k, ov in OVERLAP[r["phase"]].items():
            frac = ov / dur
            mass[k] += per_win * frac * int(r["n_win"])
            denom[k] += frac * int(r["n_win"])
    mean = {k: (mass[k] / denom[k] if denom[k] > 0 else np.nan) for k in BINS}
    out = pd.DataFrame([{
        "rel_hour": k,
        "mean_eb_speeding": mean[k],
        "eb_mass": mass[k],
        "contrib_windows": denom[k],
    } for k in BINS])
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"   Wrote {OUT_CSV}")

    # sanity: total mass conserved
    tot_in = float(m["eb"].sum())
    tot_out = float(sum(mass.values()))
    print(f"   EB mass in={tot_in:,.1f} out={tot_out:,.1f} "
          f"(ratio {tot_out / tot_in:.4f})")
    assert abs(tot_out / tot_in - 1.0) < 1e-9, "mass not conserved"

    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.array(BINS)
    ys = np.array([mean[k] for k in BINS])
    ax.plot(xs, ys, "o-", lw=2, color="firebrick")
    ax.axvline(0, color="k", ls="--", lw=1, label="Kick-off")
    ax.axvspan(-3, -0.5, color="steelblue", alpha=0.08, label="Pre-arrival")
    ax.axvspan(-0.5, 1.5, color="grey", alpha=0.10, label="Occupancy")
    ax.axvspan(1.5, 4, color="darkorange", alpha=0.08, label="Dispersal")
    ax.set_xlabel("Hours relative to kick-off (KO-3h .. KO+4h; every fixture contributes to every bin)")
    ax.set_ylabel("Mean EB speeding events per segment per contributing window")
    ax.set_title("Hourly speeding profile around kick-off (speeding-only; 16 fixtures aligned on KO)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k:+d}" for k in xs])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_HOURLY, dpi=150)
    plt.close(fig)
    print(f"   Saved {FIG_HOURLY}")
    print("DONE!")


if __name__ == "__main__":
    main()
