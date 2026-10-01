"""
21_eb_timewise_risk.py
Time-resolved Empirical Bayes risk profiling for the Suncorp telematics study.

This extends the existing EB pipeline (19/20) by *preserving the time
dimension* that the aggregate scripts discard. The segment table already
carries a full time structure:

    segment_roads_part0_5000.csv.gz
      type   : Event | Control
      ref_id : event-day identifier (1..16, spanning Mar-2025..Mar-2026)
      phase  : pre | during | post   (7h window centred on kickoff)
      ref_dt : kickoff datetime of the event/control day
      road_osmid, braking, speeding, trips

What this script produces
-------------------------
1. Phase-resolved EB estimates. Instead of a single SPF per (Event/Control)
   type, we fit a separate Safety Performance Function for every
   (type x phase) stratum (6 SPFs). This is the "timewise" improvement:
   risk is shrunk toward a phase-specific expectation, not a single global
   mean, so the pre/during/post surge is modelled explicitly.
2. A 24-hour mean risk profile per segment, built by mapping each
   (event, phase) onto absolute clock hours relative to kickoff and
   averaging over all 16 event windows ("profiling all the data").
3. A "better methods" comparison: the plain phase EB is benchmarked against
   (a) raw counts and (b) an EB SPF that also includes road class
   (highway) as a covariate, evaluated out-of-sample on the 2026 hold-out
   events.

Outputs
-------
  results/eb_timewise_segment_phase.csv   per (segment,type,phase) EB risk
  results/eb_timewise_hourly.csv          per segment x 24h mean EB risk
  results/eb_timewise_summary.csv         phase-level aggregates
  results/method_comparison.csv           raw vs EB vs EB+class RMSE
  figures/eb_timewise_phase_curve.png     phase risk curve (event vs control)
  figures/eb_timewise_hourly.png          24h mean risk profile
  results/eb_timewise.tex                 LaTeX summary for the paper

Usage
  python 21_eb_timewise_risk.py
"""
import os
import sys
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import prepare_metacount_frame, fit_metacount_nb, predict_metacount

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
os.makedirs(FIG, exist_ok=True)

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
FEATURES = os.path.join(DATA, "road_geometry_features.csv")
OUT_SEG_PHASE = os.path.join(RES, "eb_timewise_segment_phase.csv")
OUT_HOURLY = os.path.join(RES, "eb_timewise_hourly.csv")
OUT_HOURLY_CTRL = os.path.join(RES, "eb_timewise_hourly_control.csv")
OUT_SUMMARY = os.path.join(RES, "eb_timewise_summary.csv")
OUT_METHOD = os.path.join(RES, "method_comparison.csv")
OUT_TEX = os.path.join(RES, "eb_timewise.tex")

PHASES = ["pre", "during", "post"]
TYPES = ["Event", "Control"]
HOLD_OUT_EVENTS = [14, 15, 16]          # March-2026 Suncorp fixtures (prospective test)
# MetaCount NB random-parameters specs (replaces old FORMULA_AUG SPF strings).
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


def fit_spf(df, formula=None, y_col="Y", R=150):
    """Fit NB random-parameters count model via metacountregressor.

    Returns (fit_info, alpha). `formula` arg kept for call compat, ignored.
    """
    mc, _ = prepare_metacount_frame(df, exposure_col="trips", highway_col="highway")
    fixed, rdm = _mc_terms_for(mc)
    info = fit_metacount_nb(mc, y_col=y_col, id_col="road_osmid",
                            fixed_terms=fixed, rdm_terms=rdm,
                            model="nb", R=R, de_maxiter=8,
                            de_popsize=8, verbose=False)
    return info, float(info["alpha"])


def fit_spf_variant(mc, kind, R=60):
    """Exposure-only / class-only / geo variants for method comparison.

    Caller passes SEGMENT-AGGREGATED mc (n~7.5k). Row-level EB_geo RP fits
    exceeded 120GB twice (OOM kills 25270105/25277095): never fit RP on
    ~340k window rows. Light DE settings + caller gc between variants.
    """
    if kind == "EB_base":
        info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                                fixed_terms=["logexp"], rdm_terms=[],
                                model="nb", R=R, de_maxiter=5,
                                de_popsize=5, verbose=False)
    elif kind == "EB_class":
        fixed = ["logexp"] + [c for c in mc.columns if c.startswith("hw_")]
        info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                                fixed_terms=fixed, rdm_terms=[],
                                model="nb", R=R, de_maxiter=5,
                                de_popsize=5, verbose=False)
    else:
        fixed, rdm = _mc_terms_for(mc)
        info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                                fixed_terms=fixed, rdm_terms=rdm,
                                model="nb", R=R, de_maxiter=5,
                                de_popsize=5, verbose=False)
    return info


def eb_adjust(y, expected, alpha):
    """Empirical Bayes adjustment.

    EB = w * expected + (1 - w) * y, where w = 1/(1 + alpha * expected).
    Low-exposure segments (small expected) get high weight on the model
    prediction; high-exposure segments retain their own observations.
    """
    w = 1.0 / (1.0 + alpha * expected)
    return w * expected + (1.0 - w) * y, w


def phase_clock_window(kickoff_h, phase):
    """Return [start,end) absolute hours (decimal) covered by a phase."""
    if phase == "pre":
        return (kickoff_h - 3.0, kickoff_h - 0.5)
    if phase == "during":
        return (kickoff_h - 0.5, kickoff_h + 1.5)
    return (kickoff_h + 1.5, kickoff_h + 4.0)


def main():
    print("=" * 80)
    print("21_eb_timewise_risk.py - time-resolved Empirical Bayes")
    print("=" * 80)

    print("\n1. Loading segment table (time-resolved)...")
    df = pd.read_csv(SEGMENT_ROADS)
    print(f"   {len(df):,} rows | {df['road_osmid'].nunique():,} segments | "
          f"{df['ref_id'].nunique()} event-days")

    # Merge OSM geometry / road-inventory features so the SPF can model crash
    # form (curvature, intersection proximity, lanes, speed, class) directly.
    if os.path.exists(FEATURES):
        feat = pd.read_csv(FEATURES).drop(columns=["highway"], errors="ignore")
        df = df.merge(feat, on="road_osmid", how="left")
        for c in ["curvature_deg_km", "sinuosity", "is_link", "oneway",
                  "road_class_rank", "node_degree", "length_m"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)
        for c in ["maxspeed_kmh", "lanes", "dist_to_intersection_m"]:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].median())
        # Standardise numeric geometry covariates (mixed scales -> NB alpha diverges)
        for c in ["curvature_deg_km", "sinuosity", "maxspeed_kmh", "lanes",
                  "dist_to_intersection_m", "node_degree", "length_m"]:
            if c in df.columns:
                sd = df[c].std()
                df[c + "_z"] = (df[c] - df[c].mean()) / (sd if sd and sd > 0 else 1.0)
        print(f"   merged {feat.shape[1]-1} geometry features")
    if "highway" in df.columns:
        df["highway"] = df["highway"].fillna("unknown").astype(str)

    df["Y"] = df["braking"] + df["speeding"]          # risky telematics observation
    df["ref_dt"] = pd.to_datetime(df["ref_dt"])
    df["kickoff_h"] = df["ref_dt"].dt.hour + df["ref_dt"].dt.minute / 60.0

    # ---- 2. Phase-resolved NB random-parameters models (type x phase) ----
    print("\n2. Fitting 6 phase-resolved MetaCount NB-RP models (type x phase)...")
    spfs = {}
    for t in TYPES:
        for ph in PHASES:
            sub = df[(df["type"] == t) & (df["phase"] == ph) & (df["trips"] > 0)].copy()
            if len(sub) < 50:
                continue
            info, alpha = fit_spf(sub, R=150)
            # strip heavy JAX objects: downstream only needs beta/parsed/terms
            info.pop("fit", None)
            info.pop("builder", None)
            gc.collect()
            spfs[(t, ph)] = (info, alpha)
            b0 = info["parsed"]["mle"].get("__INTERCEPT__", float("nan"))
            print(f"   {t:7s}/{ph:6s}: n={len(sub):>6,} AIC={info['aic']:>9.0f} "
                  f"alpha={alpha:.3f} beta0={b0:.3f} [{info['spec_label']}]")

    # ---- 3. EB adjustment per (segment, event-day, type, phase) -------------
    print("\n3. Computing phase-resolved EB estimates...")
    rows = []
    for (t, ph), (info, alpha) in spfs.items():
        sub = df[(df["type"] == t) & (df["phase"] == ph) & (df["trips"] > 0)].copy()
        mc_sub, _ = prepare_metacount_frame(sub, exposure_col="trips", highway_col="highway")
        sub["expected"] = predict_metacount(mc_sub, info)
        sub["eb"], sub["w"] = eb_adjust(sub["Y"].values, sub["expected"].values, alpha)
        g = sub.groupby("road_osmid").agg(
            Y=("Y", "sum"),
            exposure=("trips", "sum"),
            expected=("expected", "sum"),
            eb=("eb", "sum"),
            w=("w", "mean"),
        ).reset_index()
        g["type"] = t
        g["phase"] = ph
        rows.append(g)
    seg_phase = pd.concat(rows, ignore_index=True)
    seg_phase.to_csv(OUT_SEG_PHASE, index=False)
    print(f"   Wrote {OUT_SEG_PHASE} ({len(seg_phase):,} rows)")

    # ---- 4. 24-hour mean risk profile per segment ---------------------------
    print("\n4. Building 24-hour mean risk profile (Event & Control)...")

    def build_hourly(sub_df, eb_phase_df):
        sub = sub_df.merge(eb_phase_df[["road_osmid", "phase", "eb"]],
                           on=["road_osmid", "phase"], how="left").dropna(subset=["eb"])
        hourly = {r: np.zeros(24) for r in sub["road_osmid"].unique()}
        counts = np.zeros(24)
        for _, r in sub.iterrows():
            s, ph, ko, val = r["road_osmid"], r["phase"], r["kickoff_h"], r["eb"]
            a, b = phase_clock_window(ko, ph)
            hrs = np.arange(int(np.ceil(a)), int(np.floor(b)) + 1)
            if len(hrs) == 0:
                continue
            share = val / len(hrs)
            for h in hrs:
                hh = h % 24
                hourly[s][hh] += share
                counts[hh] += 1
        out = pd.DataFrame(
            [{"road_osmid": s, **{f"h{h:02d}": hourly[s][h] for h in range(24)}}
             for s in hourly])
        for h in range(24):
            out[f"h{h:02d}"] = out[f"h{h:02d}"] / max(counts[h], 1)
        return out

    hourly_df = build_hourly(df[df["type"] == "Event"],
                             seg_phase[seg_phase["type"] == "Event"])
    hourly_df.to_csv(OUT_HOURLY, index=False)
    print(f"   Wrote {OUT_HOURLY} ({len(hourly_df):,} segments x 24h)")

    hourly_ctrl = build_hourly(df[df["type"] == "Control"],
                               seg_phase[seg_phase["type"] == "Control"])
    hourly_ctrl.to_csv(OUT_HOURLY_CTRL, index=False)
    print(f"   Wrote {OUT_HOURLY_CTRL} ({len(hourly_ctrl):,} segments x 24h)")

    # ---- 5. Phase-level summary ---------------------------------------------
    print("\n5. Phase-level summary...")
    summary = seg_phase.groupby(["type", "phase"]).agg(
        segments=("road_osmid", "nunique"),
        total_observed=("Y", "sum"),
        total_expected=("expected", "sum"),
        total_eb=("eb", "sum"),
        mean_weight=("w", "mean"),
    ).reset_index()
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))

    # ---- 6. "Better methods" comparison (segment-aggregated training) -----
    # OOM FIX (kills 25270105/25277095 at EB_geo, 53GB/123GB): the ranking
    # target is segment-level, so train variants on segment totals (n~7.5k)
    # instead of window rows (n~340k). Test predictions stay row-level.
    print("\n6. Method comparison on 2026 hold-out events "
          f"{HOLD_OUT_EVENTS}...")
    train = df[~df["ref_id"].isin(HOLD_OUT_EVENTS)].copy()
    test = df[df["ref_id"].isin(HOLD_OUT_EVENTS)].copy()
    _agg = {"Y": ("Y", "sum"), "trips": ("trips", "sum")}
    if "highway" in train.columns:
        _agg["highway"] = ("highway", "first")
    for _c in ["maxspeed_kmh_z", "curvature_deg_km_z", "lanes_z",
               "dist_to_intersection_m_z"]:
        if _c in train.columns:
            _agg[_c] = (_c, "first")
    train_seg = train.groupby("road_osmid", as_index=False).agg(**_agg)
    print(f"   aggregated train: {len(train):,} windows -> {len(train_seg):,} segments")
    mc_tr, _ = prepare_metacount_frame(
        train_seg, exposure_col="trips",
        highway_col="highway" if "highway" in train_seg.columns else "__none__")
    mc_te, _ = prepare_metacount_frame(test, exposure_col="trips", highway_col="highway")
    # align test dummies to train
    for c in mc_tr.columns:
        if c not in mc_te.columns:
            mc_te[c] = 0.0
    methods = {}
    for tag in ["EB_base", "EB_class", "EB_geo"]:
        info = fit_spf_variant(mc_tr, tag, R=60)
        alpha = float(info["alpha"])
        test = test.copy()
        mc_te_aligned = mc_te.copy()
        test["expected"] = predict_metacount(mc_te_aligned, info)
        test["eb"], _ = eb_adjust(test["Y"].values, test["expected"].values, alpha)
        rmse_raw = np.sqrt(((test["Y"] - test["expected"]) ** 2).mean())
        rmse_eb = np.sqrt(((test["Y"] - test["eb"]) ** 2).mean())

        # ---- rank-based hot-spot metrics (fairer than count-RMSE, which
        # penalises EB precisely for shrinking noisy segments) ---------------
        # Aggregate the per-window holdout rows to segment level, then compare
        # each method's ranking against what actually happened.
        seg = test.groupby("road_osmid").agg(
            obs=("Y", "sum"), expected=("expected", "sum"),
            eb=("eb", "sum")).reset_index()

        def top_overlap(score_col, n):
            top_obs = set(seg.nlargest(n, "obs")["road_osmid"])
            top_meth = set(seg.nlargest(n, score_col)["road_osmid"])
            return len(top_obs & top_meth) / float(n)

        methods[tag] = {
            "rmse_raw": rmse_raw, "rmse_eb": rmse_eb,
            "alpha": alpha, "aic": info["aic"],
            "spearman_exp": spearmanr(seg["expected"], seg["obs"]).statistic,
            "spearman_eb": spearmanr(seg["eb"], seg["obs"]).statistic,
            "top20_overlap": top_overlap("eb", 20),
            "top50_overlap": top_overlap("eb", 50),
            "top100_overlap": top_overlap("eb", 100),
            "top50_overlap_exp": top_overlap("expected", 50),
        }
        print(f"   {tag:9s}: RMSE raw={rmse_raw:.3f}  RMSE EB={rmse_eb:.3f}  "
              f"Spearman(EB)={methods[tag]['spearman_eb']:.3f}  "
              f"Top50={methods[tag]['top50_overlap']:.2f}  AIC={info['aic']:.0f} [{info['spec_label']}]")
        del info
        gc.collect()
    pd.DataFrame(
        [{"method": k, **v} for k, v in methods.items()]
    ).to_csv(OUT_METHOD, index=False)

    # ---- 7. Figures ---------------------------------------------------------
    print("\n7. Figures...")
    ev_curve = summary[summary["type"] == "Event"].set_index("phase").reindex(PHASES)
    ct_curve = summary[summary["type"] == "Control"].set_index("phase").reindex(PHASES)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(PHASES))
    ax.plot(x, ev_curve["total_eb"].values, "o-", label="Event", lw=2)
    ax.plot(x, ct_curve["total_eb"].values, "s--", label="Control", lw=2)
    ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in PHASES])
    ax.set_xlabel("Event window phase"); ax.set_ylabel("Total EB-adjusted risk")
    ax.set_title("Phase-resolved risk: Event vs Control")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "eb_timewise_phase_curve.png"), dpi=150)
    plt.close(fig)

    mean_hour = hourly_df[[f"h{h:02d}" for h in range(24)]].mean().values
    # Filter to active hours only: exclude hours with negligible risk
    # (fixtures cluster in afternoon/evening; early-morning hours have no data)
    active_mask = mean_hour > 0.005 * mean_hour.max()
    active_hours = np.array([h for h in range(24) if active_mask[h]])
    active_risks = mean_hour[active_mask]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(active_hours, active_risks, "o-", lw=2, color="firebrick")
    ax.set_xlabel("Clock hour"); ax.set_ylabel("Mean EB risk per segment")
    ax.set_title("24-hour mean risk profile (averaged over 16 event windows)")
    ax.set_xticks(active_hours); ax.set_xticklabels([str(int(h)) for h in active_hours])
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "eb_timewise_hourly.png"), dpi=150)
    plt.close(fig)
    print("   Saved phase_curve + hourly figures")

    # ---- 8. LaTeX summary ---------------------------------------------------
    L = []  # collected LaTeX lines (plain strings; no f-strings w/ backslashes)
    L.append("% eb_timewise.tex - auto-generated by 21_eb_timewise_risk.py")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{Phase-resolved Empirical Bayes risk "
             "(type $\\times$ phase strata).}")
    L.append("\\label{tab:eb_timewise}")
    L.append("\\begin{tabular}{llrrrr}")
    L.append("\\hline")
    L.append("Type & Phase & Segments & Observed & Expected & EB \\\\")
    L.append("\\hline")
    for _, r in summary.iterrows():
        L.append("{0} & {1} & {2:,} & {3:,.0f} & {4:,.0f} & {5:,.0f} \\\\".format(
            r["type"], r["phase"], int(r["segments"]), r["total_observed"],
            r["total_expected"], r["total_eb"]))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{Out-of-sample method comparison on 2026 hold-out "
             "events (IDs " + str(HOLD_OUT_EVENTS) + "). Overlap = share of "
             "observed top-50 segments recovered by the method's EB ranking.}")
    L.append("\\label{tab:method_cmp}")
    L.append("\\begin{tabular}{lrrrrr}")
    L.append("\\hline")
    L.append("Method & RMSE (raw) & RMSE (EB) & Spearman (EB) & "
             "Top-50 overlap & AIC \\\\")
    L.append("\\hline")
    for k, v in methods.items():
        L.append("{0} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.2f} & {5:.0f} \\\\".format(
            k, v["rmse_raw"], v["rmse_eb"], v["spearman_eb"],
            v["top50_overlap"], v["aic"]))
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    print("   Wrote " + OUT_TEX)

    print("\n" + "=" * 80)
    print("DONE! Time-resolved EB complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
