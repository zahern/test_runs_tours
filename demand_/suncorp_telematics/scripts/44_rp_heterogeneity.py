"""
44_rp_heterogeneity.py - unobserved heterogeneity, shown properly (no zero-inflation).

Scientific point
----------------
The paper claims unobserved heterogeneity in driver behaviour (and site
response), but the published RP fits cannot evidence it: weak optimiser
(DE 8x8, R=150, single seed), no coef/RP-SD artefacts saved, no RP-vs-fixed
test, and means-only scoring that drops the Jensen term. Job-12's collapse
(alpha~0 with VMR 20-166) is itself a symptom of massive slope
heterogeneity scored at the means -- not a refutation of it.

Modelling constraint (analyst-imposed, non-negotiable)
------------------------------------------------------
No zero-inflated or hurdle components anywhere. All modelled rows have
trips > 0 (at-risk exposure), so zeros are sampling zeros, not structural
"safe states". Heterogeneity is shown with continuous random parameters
(slopes that vary across units -- a regime that cannot be misread as a
safe-state mass) plus the NB gamma-mixing dispersion alpha.

Part A -- site heterogeneity (segment x window rows; speeding-only)
-------------------------------------------------------------------
Per (type x phase) stratum, fit with metacountregressor:
  * NB-RP (lanes_z, dist_to_intersection_m_z normal random parameters),
    STRONG settings: R=300 Halton draws, DE maxiter=15/popsize=10,
    seeds {0,1}, best-AIC kept. Full coef tables SAVED (means, SDs, SEs,
    z, p) -- the RP SD rows with significance ARE the heterogeneity result.
  * NB-fixed (same fixed terms) for the nested comparison.
Report per stratum: alpha (SE), RP SDs with p-values, LR = 2(l_RP-l_fixed)
with df = #RP variances (naive chi-square; boundary caveat noted since
SDs are tested at 0), AIC_RP vs AIC_fixed, and the Jensen factor
sum(integrated predictions)/sum(means-only predictions) quantifying what
job-12 scoring discarded.
Predictions use the INTEGRATED fitted mean fit["predictions"], never
means-only scoring. EB = w*lam_int + (1-w)*y with the fitted alpha.
ABORT unless every stratum EB total is within 5% of observed.

Part B -- driver-panel scoping (read-only, sampled)
----------------------------------------------------
Segment-aggregated tables cannot identify driver effects (every row pools
all drivers traversing the segment). The raw trip files DO carry VehicleID
(data/raw_near_miss_2026/trips-*.csv.gz header confirmed), so a
driver-indexed panel is feasible -- but note the existing extraction
(results/event_points_full.csv.gz) DROPPED VehicleID (OUT_COLS has
VehicleType only), so driver work needs re-extraction with the ID retained.
This part measures (on a small sample): raw volume, VehicleID coverage,
distinct vehicles, trips/vehicle distribution, and extrapolates the
full-panel build cost. No heavy compute; findings to
results/driver_panel_scope.json.

Outputs
-------
  results/rp_heterogeneity_coefs.csv          all strata, means/SDs/SEs/z/p
  results/rp_heterogeneity_summary.csv        alpha, RP SDs, LR, AICs, Jensen, EB/obs
  results/rp_heterogeneity_segment_phase.csv  EB surface (integrated-RP based)
  results/rp_heterogeneity_hourly.csv         NaN = unobserved hour
  results/rp_heterogeneity.tex                paper heterogeneity table
  results/rp_heterogeneity_audit.json
  results/driver_panel_scope.json
  figures/rp_heterogeneity_hourly.png         broken-line hourly (speeding-only)

Usage:
  python 44_rp_heterogeneity.py
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import prepare_metacount_frame, fit_metacount_nb, predict_metacount

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
RAW_DIR = os.path.join(DATA, "raw_near_miss_2026")

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
FEATURES = os.path.join(DATA, "road_geometry_features.csv")

OUT_COEF = os.path.join(RES, "rp_heterogeneity_coefs.csv")
OUT_SUMMARY = os.path.join(RES, "rp_heterogeneity_summary.csv")
OUT_SEG_PHASE = os.path.join(RES, "rp_heterogeneity_segment_phase.csv")
OUT_HOURLY = os.path.join(RES, "rp_heterogeneity_hourly.csv")
OUT_TEX = os.path.join(RES, "rp_heterogeneity.tex")
OUT_JSON = os.path.join(RES, "rp_heterogeneity_audit.json")
OUT_SCOPE = os.path.join(RES, "driver_panel_scope.json")
FIG_HOURLY = os.path.join(FIG, "rp_heterogeneity_hourly.png")

PHASES = ["pre", "during", "post"]
TYPES = ["Event", "Control"]
RDM_GEO = ["lanes_z:normal", "dist_to_intersection_m_z:normal"]
R_RP, DE_MAXITER, DE_POPSIZE, SEEDS = 300, 15, 10, (0, 1)
EB_TOL = 0.05

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


def eb_adjust(y, lam, alpha):
    w = 1.0 / (1.0 + alpha * np.asarray(lam, dtype=float))
    return w * np.asarray(lam, dtype=float) + (1.0 - w) * np.asarray(y, dtype=float), w


def phase_clock_window(kickoff_h, phase):
    if phase == "pre":
        return (kickoff_h - 3.0, kickoff_h - 0.5)
    if phase == "during":
        return (kickoff_h - 0.5, kickoff_h + 1.5)
    return (kickoff_h + 1.5, kickoff_h + 4.0)


def sd_pvalue(row, parsed):
    """Two-sided p for an RP SD row: match sd(name) to its z-value."""
    return float(parsed["p"].get(row, np.nan))


def main():
    audit = {"no_zero_inflation": "continuous RPs only: slopes vary across "
             "units; no safe-state mass point. All rows trips>0."}
    print("=" * 80)
    print("44_rp_heterogeneity.py - unobserved heterogeneity, shown properly")
    print("=" * 80)

    # ================= Part B first (cheap): driver-panel scoping =========
    print("\n[Scope] Driver-panel feasibility from raw trip files...")
    scope = {"raw_dir": RAW_DIR}
    files = sorted(glob.glob(os.path.join(RAW_DIR, "trips-*.csv.gz")))
    scope["n_shards"] = len(files)
    scope["bytes_total"] = int(sum(os.path.getsize(f) for f in files)) if files else 0
    print(f"   shards={scope['n_shards']} total_GB={scope['bytes_total']/1e9:.1f}")
    sampled_vehicles, sampled_trips, null_ids, per_veh = set(), 0, 0, {}
    for f in files[:3]:
        try:
            df = pd.read_csv(f, usecols=["VehicleID", "TripID", "VehicleType",
                                         "points_in_area"],
                             nrows=20000)
        except Exception as e:  # noqa: BLE001 - record and continue sampling
            scope.setdefault("read_errors", []).append(f"{os.path.basename(f)}: {e}")
            continue
        sampled_trips += len(df)
        null_ids += int(df["VehicleID"].isna().sum())
        for vid in df["VehicleID"].dropna().astype(str):
            sampled_vehicles.add(vid)
            per_veh[vid] = per_veh.get(vid, 0) + 1
    scope["sampled_trips"] = int(sampled_trips)
    scope["vehicleid_null_share"] = float(null_ids / max(sampled_trips, 1))
    scope["distinct_vehicles_sample"] = len(sampled_vehicles)
    if per_veh:
        arr = np.array(sorted(per_veh.values()), dtype=float)
        scope["trips_per_vehicle_p50"] = float(np.median(arr))
        scope["trips_per_vehicle_mean"] = float(arr.mean())
    # Does the existing extracted event file retain VehicleID?
    scope["event_points_exists"] = bool(glob.glob(
        os.path.join(RES, "event_points_full.csv.gz")))
    scope["event_points_retains_vehicleid"] = False  # OUT_COLS: VehicleType only
    scope["conclusion"] = ("Driver panel feasible (VehicleID in raw headers) "
                           "but requires re-extraction WITH VehicleID retained; "
                           "current event_points file dropped it. Next: per-driver "
                           "rates x exposure panel -> grouped RP with driver random "
                           "effects (or two-way driver x site).")
    with open(OUT_SCOPE, "w") as f:
        json.dump(scope, f, indent=2)
    print(f"   distinct_vehicles(sample)={scope['distinct_vehicles_sample']:,} "
          f"null_id_share={scope['vehicleid_null_share']:.4f}")
    print(f"   Wrote {OUT_SCOPE}")
    audit["scope"] = scope

    # ================= Part A: site heterogeneity via strong RP ===========
    print("\n[Het] Loading segment table (speeding-only, at-risk rows)...")
    df = pd.read_csv(SEGMENT_ROADS)
    df = df[df["trips"] > 0].copy()
    df["Y"] = df["speeding"].astype(float)
    audit["n_rows_at_risk"] = int(len(df))
    audit["total_speeding"] = float(df["Y"].sum())
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
    df["highway"] = df.get("highway", "unknown").fillna("unknown").astype(str)
    df["ref_dt"] = pd.to_datetime(df["ref_dt"])
    df["kickoff_h"] = df["ref_dt"].dt.hour + df["ref_dt"].dt.minute / 60.0

    coef_rows, sum_rows, seg_rows = [], [], []
    for t in TYPES:
        for ph in PHASES:
            sub = df[(df["type"] == t) & (df["phase"] == ph)].copy()
            if len(sub) < 50:
                raise SystemExit(f"FATAL: stratum {t}/{ph} n={len(sub)}")
            mc, _ = prepare_metacount_frame(sub, exposure_col="trips",
                                            highway_col="highway")
            fixed, rdm = _mc_terms_for(mc)
            # --- RP: multi-seed, best AIC ---
            best, tries = None, []
            for s in SEEDS:
                try:
                    info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                                            fixed_terms=fixed, rdm_terms=rdm,
                                            model="nb", R=R_RP,
                                            de_maxiter=DE_MAXITER,
                                            de_popsize=DE_POPSIZE, seed=s,
                                            verbose=False)
                    tries.append(f"seed{s}:AIC={info['aic']:.0f}")
                    if best is None or info["aic"] < best["aic"]:
                        best = info
                except Exception as e:  # noqa: BLE001
                    tries.append(f"seed{s}:FAIL({type(e).__name__})")
            if best is None:
                raise SystemExit(f"FATAL: RP failed all seeds {t}/{ph}: {tries}")
            # --- nested fixed NB for the LR heterogeneity test ---
            fix = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                                   fixed_terms=fixed, rdm_terms=[],
                                   model="nb", R=R_RP, de_maxiter=DE_MAXITER,
                                   de_popsize=DE_POPSIZE, seed=SEEDS[0],
                                   verbose=False)
            lr = 2.0 * (best["loglik"] - fix["loglik"])
            lr_df = max(len(rdm), 1)
            lr_p = float(chi2.sf(max(lr, 0.0), lr_df))  # naive; boundary caveat
            # --- integrated vs means-only predictions (Jensen diagnostic) --
            lam_int = np.asarray(best["expected"], dtype=float)  # integrated
            lam_mu = np.asarray(predict_metacount(mc, best), dtype=float)
            jensen = float(lam_int.sum() / max(lam_mu.sum(), 1e-9))
            alpha = float(best["alpha"])
            sub = sub.copy()
            sub["expected"], sub["w"] = lam_int, 1.0 / (1.0 + alpha * lam_int)
            sub["eb"] = sub["w"] * lam_int + (1.0 - sub["w"]) * sub["Y"].values
            obs, ebt = float(sub["Y"].sum()), float(sub["eb"].sum())
            if abs(ebt - obs) / max(obs, 1) > EB_TOL:
                raise SystemExit(
                    f"FATAL: stratum {t}/{ph} EB {ebt:,.0f} vs obs {obs:,.0f}")
            parsed = best["parsed"]
            rp_sds = {}
            for r_ in rdm:
                base = r_.split(":")[0]
                for key in (f"sd({base})", f"SD({base})", f"sd_{base}"):
                    if key in parsed["parte"]:
                        rp_sds[base] = {
                            "sd": float(parsed["parte"][key]),
                            "p": sd_pvalue(key, parsed)}
                        break
            print(f"   {t:7s}/{ph:6s}: n={len(sub):>6,} "
                  f"AIC_RP={best['aic']:>9.0f} AIC_F={fix['aic']:>9.0f} "
                  f"LR={lr:.1f}(df{lr_df},p={lr_p:.3g}) alpha={alpha:.3f} "
                  f"Jensen={jensen:.2f}x obs={obs:,.0f} EB={ebt:,.0f} "
                  f"seeds={tries}")
            audit[f"{t}/{ph}"] = {
                "spec": best["spec_label"], "seeds": tries,
                "aic_rp": best["aic"], "aic_fixed": fix["aic"],
                "lr_chi2": float(lr), "lr_df": int(lr_df), "lr_p": lr_p,
                "lr_boundary_caveat": "SDs tested at 0; naive chi2 is "
                "conservative vs the chi-bar mixture",
                "alpha": alpha, "rp_sds": rp_sds, "jensen_factor": jensen,
                "obs": obs, "eb": ebt,
                "wbar": float(pd.Series(1.0 / (1.0 + alpha * lam_int)).mean())}
            for _, r in best["coef"].iterrows():
                coef_rows.append({"type": t, "phase": ph,
                                  "variable": r["variable"],
                                  "coef": r["coef_mle"],
                                  "std_err": r["std_err"],
                                  "z": r["z_value"], "p": r["p_value"]})
            sum_rows.append({"type": t, "phase": ph,
                             "segments": int(sub["road_osmid"].nunique()),
                             "total_speeding": obs,
                             "total_expected_integrated": float(lam_int.sum()),
                             "total_expected_meansonly": float(lam_mu.sum()),
                             "jensen_factor": jensen,
                             "total_eb": ebt,
                             "mean_weight": float(
                                 pd.Series(1.0 / (1.0 + alpha * lam_int)).mean()),
                             "alpha": alpha,
                             "aic_rp": best["aic"], "aic_fixed": fix["aic"],
                             "lr_chi2": float(lr), "lr_p": lr_p})
            g = sub.groupby("road_osmid").agg(
                Y=("Y", "sum"), exposure=("trips", "sum"),
                expected=("expected", "sum"), eb=("eb", "sum"),
                w=("w", "mean")).reset_index()
            g["type"] = t
            g["phase"] = ph
            seg_rows.append(g)

    pd.DataFrame(coef_rows).to_csv(OUT_COEF, index=False)
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    seg_phase = pd.concat(seg_rows, ignore_index=True)
    seg_phase.to_csv(OUT_SEG_PHASE, index=False)

    # ---- hourly (NaN unobserved; line broken at gaps) ----
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
        for h in range(24):
            out[f"h{h:02d}"] = out[f"h{h:02d}"] / counts[h] if observed[h] else np.nan
        return out, observed

    hourly_df, observed = build_hourly(
        df[df["type"] == "Event"], seg_phase[seg_phase["type"] == "Event"])
    hourly_df.to_csv(OUT_HOURLY, index=False)
    audit["observed_hours"] = [int(h) for h in range(24) if observed[h]]
    audit["unobserved_hours"] = [int(h) for h in range(24) if not observed[h]]

    mean_hour = hourly_df[[f"h{h:02d}" for h in range(24)]].mean().values
    xx = np.arange(24)
    fig, ax = plt.subplots(figsize=(9, 4))
    cur = []
    for h in list(range(24)) + [-1]:
        if h >= 0 and observed[h]:
            cur.append(h)
        elif cur:
            ax.plot(xx[cur], mean_hour[cur], marker="o", lw=2, color="firebrick")
            cur = []
    ax.set_xlabel("Clock hour (gaps = no observation, never zero)")
    ax.set_ylabel("Mean EB speeding events per segment per contributing window")
    ax.set_title("Hourly speeding profile (integrated-RP EB; contributing windows only)")
    ax.set_xticks(audit["observed_hours"])
    ax.set_xticklabels([str(h) for h in audit["observed_hours"]])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_HOURLY, dpi=150)
    plt.close(fig)

    # ---- paper heterogeneity table ----
    L = ["% rp_heterogeneity.tex - auto-generated by 44_rp_heterogeneity.py",
         "\\begin{table}[htbp]", "\\centering",
         "\\caption{\\textcolor{Highlight}{Unobserved heterogeneity in speeding "
         "risk (NB random-parameters vs fixed NB, speeding-only).} "
         "RP SD rows test slope variation across sites; LR tests the joint "
         "RP variances (naive $\\chi^2$, boundary-conservative). Jensen = "
         "integrated-mean / means-only totals: what scoring at RP means discards.}",
         "\\label{tab:hetero}",
         "\\begin{tabular}{llrrrrr}",
         "\\hline",
         "Type & Phase & $\\alpha$ & LR $\\chi^2$ (p) & AIC RP/Fixed & Jensen & EB/Obs \\\\",
         "\\hline"]
    for r in sum_rows:
        L.append("{0} & {1} & {2:.3f} & {3:.1f} ({4:.2g}) & {5:.0f}/{6:.0f} & "
                 "{7:.2f}x & {8:.3f} \\\\".format(
                     r["type"], r["phase"], r["alpha"], r["lr_chi2"], r["lr_p"],
                     r["aic_rp"], r["aic_fixed"], r["jensen_factor"],
                     r["total_eb"] / max(r["total_speeding"], 1)))
    L += ["\\hline", "\\end{tabular}", "\\end{table}"]
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    audit["outputs"] = [OUT_COEF, OUT_SUMMARY, OUT_SEG_PHASE, OUT_HOURLY,
                        OUT_TEX, OUT_SCOPE, FIG_HOURLY]
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print("DONE! Heterogeneity fits + driver-panel scope complete.")


if __name__ == "__main__":
    main()
