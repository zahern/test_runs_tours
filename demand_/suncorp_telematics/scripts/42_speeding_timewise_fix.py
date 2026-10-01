"""
42_speeding_timewise_fix.py - Fig5 remodel on SPEEDING-ONLY + hour-zero fix.

Why this exists
--------------
Referee/user review of main.tex found two load-bearing defects in Fig5
(fig:timewise, scripts/21_eb_timewise_risk.py):

  1. Wrong estimand. The paper's enforcement point is *speeding events*
     (counts of threshold exceedances per segment x window). The published
     Fig5 instead plots generic "risk" with Y = braking + speeding
     (21_eb_timewise_risk.py line 180), so arrival/occupancy/dispersal
     shares and the hourly curve mix hard-braking behaviour with speeding.
  2. Hour-zero artefact. build_hourly() divides accumulated risk by
     max(counts[h], 1). Kickoffs span 14:00-20:00 so phases cover ~11:00-24:00;
     clock hours 01-10 have counts[h] == 0 (no contributing window) and are
     plotted as exact 0.0 (see figures/eb_timewise_hourly.png, e.g. hour 4 = 0).
     A zero there reads as "no risk at 4am" when the truth is "no observation
     at 4am". Unobserved hours must be NaN / omitted, never 0.

What this script does
---------------------
  1. Y = speeding only (per-segment threshold-exceedance counts from the
     telematics speeding stream; one row = one segment x fixture-phase window).
  2. Six phase-resolved NB random-parameters SPFs (Event/Control x
     pre/during/post) via metacountregressor, same spec family as script 21.
  3. EB shrinkage per (segment, event-day, type, phase).
  4. Hourly curve built ONLY over observed clock hours: hours with zero
     contributing windows are NaN and excluded from the plot and the mean
     (no max(counts,1) fill-with-zero).
  5. Emits replacement Fig5 panels + summary CSV + LaTeX snippet.

Outputs
-------
  results/speeding_timewise_segment_phase.csv
  results/speeding_timewise_hourly.csv        (NaN = unobserved hour)
  results/speeding_timewise_summary.csv
  results/speeding_timewise_fix.tex
  results/speeding_timewise_audit.json
  figures/speeding_timewise_phase_curve.png   (replacement Fig5-top)
  figures/speeding_timewise_hourly.png        (replacement Fig5-bottom)

Usage:
  python 42_speeding_timewise_fix.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import prepare_metacount_frame, fit_metacount_nb, predict_metacount

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
FEATURES = os.path.join(DATA, "road_geometry_features.csv")
OUT_SEG_PHASE = os.path.join(RES, "speeding_timewise_segment_phase.csv")
OUT_HOURLY = os.path.join(RES, "speeding_timewise_hourly.csv")
OUT_SUMMARY = os.path.join(RES, "speeding_timewise_summary.csv")
OUT_TEX = os.path.join(RES, "speeding_timewise_fix.tex")
OUT_JSON = os.path.join(RES, "speeding_timewise_audit.json")
FIG_PHASE = os.path.join(FIG, "speeding_timewise_phase_curve.png")
FIG_HOURLY = os.path.join(FIG, "speeding_timewise_hourly.png")

PHASES = ["pre", "during", "post"]
TYPES = ["Event", "Control"]
RDM_GEO = ["lanes_z:normal", "dist_to_intersection_m_z:normal"]
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def _mc_terms_for(mc):
    fixed = ["logexp"]
    for c in mc.columns:
        if c.startswith("hw_"):
            fixed.append(c)
    for c in ["maxspeed_kmh_z", "curvature_deg_km_z"]:
        if c in mc.columns and c not in fixed:
            fixed.append(c)
    rdm = [r for r in RDM_GEO if r.split(":")[0] in mc.columns]
    return fixed, rdm


def eb_adjust(y, expected, alpha):
    w = 1.0 / (1.0 + alpha * np.asarray(expected, dtype=float))
    return w * np.asarray(expected, dtype=float) + (1.0 - w) * np.asarray(y, dtype=float), w


def phase_clock_window(kickoff_h, phase):
    if phase == "pre":
        return (kickoff_h - 3.0, kickoff_h - 0.5)
    if phase == "during":
        return (kickoff_h - 0.5, kickoff_h + 1.5)
    return (kickoff_h + 1.5, kickoff_h + 4.0)


def main():
    audit = {"estimand": "speeding-only threshold exceedances per segment x window"}
    print("=" * 80)
    print("42_speeding_timewise_fix.py - speeding-only Fig5 + hour-zero fix")
    print("=" * 80)

    df = pd.read_csv(SEGMENT_ROADS)
    # FIX 1: speeding-only estimand (was braking + speeding in script 21).
    assert "speeding" in df.columns, "speeding column missing from segment table"
    df["Y"] = df["speeding"].astype(float)
    audit["n_rows"] = int(len(df))
    audit["total_speeding"] = float(df["Y"].sum())
    audit["total_braking_ignored"] = float(df["braking"].sum()) if "braking" in df.columns else 0.0
    print(f"   rows={len(df):,} speeding={audit['total_speeding']:,.0f} "
          f"(braking {audit['total_braking_ignored']:,.0f} excluded by design)")

    if os.path.exists(FEATURES):
        feat = pd.read_csv(FEATURES).drop(columns=["highway"], errors="ignore")
        df = df.merge(feat, on="road_osmid", how="left")
        for c in ["maxspeed_kmh", "lanes", "dist_to_intersection_m"]:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].median())
        for c in ["curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
                  "dist_to_intersection_m", "node_degree", "length_m"]:
            if c in df.columns:
                sd = df[c].std()
                df[c + "_z"] = (df[c] - df[c].mean()) / (sd if sd and sd > 0 else 1.0)
    if "highway" in df.columns:
        df["highway"] = df["highway"].fillna("unknown").astype(str)
    df["ref_dt"] = pd.to_datetime(df["ref_dt"])
    df["kickoff_h"] = df["ref_dt"].dt.hour + df["ref_dt"].dt.minute / 60.0

    # Phase-resolved speeding SPFs.
    spfs = {}
    for t in TYPES:
        for ph in PHASES:
            sub = df[(df["type"] == t) & (df["phase"] == ph) & (df["trips"] > 0)].copy()
            if len(sub) < 50:
                continue
            mc, _ = prepare_metacount_frame(sub, exposure_col="trips", highway_col="highway")
            fixed, rdm = _mc_terms_for(mc)
            info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                                    fixed_terms=fixed, rdm_terms=rdm,
                                    model="nb", R=150, de_maxiter=16,
                                    de_popsize=10, verbose=False)
            spfs[(t, ph)] = (info, float(info["alpha"]))
            print(f"   {t:7s}/{ph:6s}: n={len(sub):>6,} AIC={info['aic']:>9.0f} "
                  f"alpha={float(info['alpha']):.3f}")

    rows = []
    calib = {}  # stratum calibration factors (HSM-style calibration-in-the-large)
    for (t, ph), (info, alpha) in spfs.items():
        sub = df[(df["type"] == t) & (df["phase"] == ph) & (df["trips"] > 0)].copy()
        mc_sub, _ = prepare_metacount_frame(sub, exposure_col="trips", highway_col="highway")
        sub["expected"] = predict_metacount(mc_sub, info)
        # FIX 3: stratum calibration factor C = sum(Y)/sum(expected) so the SPF
        # level matches observed totals by construction (was 2-5x low: coarse DE
        # + 85% zeros). C is reported, not hidden; EB shrinks toward C*expected.
        ysum, esum = float(sub["Y"].sum()), float(sub["expected"].sum())
        C = ysum / esum if esum > 0 else 1.0
        calib[f"{t}/{ph}"] = C
        print(f"   {t:7s}/{ph:6s}: calibration C={C:.3f}")
        sub["expected"] = sub["expected"] * C
        sub["eb"], sub["w"] = eb_adjust(sub["Y"].values, sub["expected"].values, alpha)
        g = sub.groupby("road_osmid").agg(
            Y=("Y", "sum"), exposure=("trips", "sum"),
            expected=("expected", "sum"), eb=("eb", "sum"),
            w=("w", "mean")).reset_index()
        g["type"] = t
        g["phase"] = ph
        rows.append(g)
    seg_phase = pd.concat(rows, ignore_index=True)
    seg_phase.to_csv(OUT_SEG_PHASE, index=False)

    # FIX 2: hourly with NaN for unobserved hours (was 0 via max(counts,1)).
    def build_hourly(sub_df, eb_phase_df):
        sub = sub_df.merge(eb_phase_df[["road_osmid", "phase", "eb"]],
                           on=["road_osmid", "phase"], how="left").dropna(subset=["eb"])
        hourly = {r: np.zeros(24) for r in sub["road_osmid"].unique()}
        counts = np.zeros(24)
        for _, r in sub.iterrows():
            a, b = phase_clock_window(r["kickoff_h"], r["phase"])
            hrs = np.arange(int(np.ceil(a)), int(np.floor(b)) + 1)
            if len(hrs) == 0:
                continue
            share = r["eb"] / len(hrs)
            for h in hrs:
                hourly[r["road_osmid"]][h % 24] += share
                counts[h % 24] += 1
        out = pd.DataFrame(
            [{"road_osmid": s, **{f"h{h:02d}": hourly[s][h] for h in range(24)}}
             for s in hourly])
        observed = counts > 0
        audit["unobserved_hours"] = [int(h) for h in range(24) if not observed[h]]
        for h in range(24):
            # Mean over CONTRIBUTING windows only; NaN where nothing observed.
            out[f"h{h:02d}"] = out[f"h{h:02d}"] / counts[h] if observed[h] else np.nan
        return out, observed, counts

    hourly_df, observed, counts = build_hourly(
        df[df["type"] == "Event"], seg_phase[seg_phase["type"] == "Event"])
    hourly_df.to_csv(OUT_HOURLY, index=False)
    print(f"   unobserved clock hours (NaN, not 0): {audit['unobserved_hours']}")

    summary = seg_phase.groupby(["type", "phase"]).agg(
        segments=("road_osmid", "nunique"),
        total_speeding=("Y", "sum"),
        total_expected=("expected", "sum"),
        total_eb=("eb", "sum"),
        mean_weight=("w", "mean")).reset_index()
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))

    # Replacement Fig5-top: phase curve in SPEEDING counts.
    ev = summary[summary["type"] == "Event"].set_index("phase").reindex(PHASES)
    ct = summary[summary["type"] == "Control"].set_index("phase").reindex(PHASES)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(PHASES))
    ax.plot(x, ev["total_eb"].values, "o-", label="Event", lw=2)
    ax.plot(x, ct["total_eb"].values, "s--", label="Control", lw=2)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in PHASES])
    ax.set_xlabel("Event window phase (pre: KO-3h..KO-0.5h; during: KO-0.5h..KO+1.5h; post: KO+1.5h..KO+4h)")
    ax.set_ylabel("Total EB-adjusted SPEEDING events (threshold exceedances)")
    ax.set_title("Phase-resolved speeding: Event vs Control (speeding-only EB)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PHASE, dpi=150)
    plt.close(fig)

    # Replacement Fig5-bottom: hourly speeding, observed hours only.
    mean_hour = hourly_df[[f"h{h:02d}" for h in range(24)]].mean().values
    obs_h = np.array([h for h in range(24) if observed[h]])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(obs_h, mean_hour[observed], "o-", lw=2, color="firebrick")
    ax.set_xlabel("Clock hour (only hours covered by a fixture window; unobserved hours omitted, not zero)")
    ax.set_ylabel("Mean EB speeding events per segment per contributing window")
    ax.set_title("Hourly speeding profile (speeding-only; averaged over contributing event windows)")
    ax.set_xticks(obs_h)
    ax.set_xticklabels([str(int(h)) for h in obs_h])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_HOURLY, dpi=150)
    plt.close(fig)

    with open(OUT_TEX, "w") as f:
        f.write("% speeding_timewise_fix.tex - auto-generated by 42_speeding_timewise_fix.py\n")
        f.write("% Speeding-only phase shares: pre/during/post EB totals below.\n")
        f.write("% Expected = calibrated SPF (C*lambda); C per stratum below.\n")
        for _, r in summary.iterrows():
            f.write(f"% {r['type']}/{r['phase']}: speeding={r['total_speeding']:,.0f} "
                    f"EB={r['total_eb']:,.0f} w={r['mean_weight']:.3f} "
                    f"C={calib.get(r['type'] + '/' + r['phase'], 1.0):.3f}\n")
    audit["calibration"] = calib
    audit["outputs"] = [OUT_SEG_PHASE, OUT_HOURLY, OUT_SUMMARY, OUT_TEX, FIG_PHASE, FIG_HOURLY]
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print("DONE! Speeding-only Fig5 + hour-zero fix complete.")


if __name__ == "__main__":
    main()
