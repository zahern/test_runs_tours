"""
46_speeding_hazard.py - per-segment speed-exceedance hazard for eq:score.

Estimand (paper eq:hazard / eq:score): expected finable-speeding
exceedances (>=10 kph over posted limit) per segment for a fixture g
in phase p,

    h_i(g,p) = lambda_i * (A_g / Abar)^eta * s_p,

  lambda_i  segment baseline from a metacountregressor NB
            random-parameters count model on event-window speeding
            totals ~ road features (exposure, highway class, lanes,
            posted speed, curvature, intersection distance).
  eta       crowd elasticity from log-log OLS of fixture speeding
            totals on published attendance (16 NRL fixtures).
  s_p       event phase shares (Table 5 EB shares).
  Abar      mean attendance of the 16 fitted fixtures.

Two cheap stages (no 100k-row RP fit: OOM lessons from 25270105/25277095).
"Most expected" speeding segments for any event size = top h_i.

Inputs:
  data/segment_roads_part0_5000.csv.gz
  data/road_geometry_features.csv
  data/fixture_calendar.csv                 (41 fixtures + attendance)
  results/speeding_timewise_segment_phase.csv (phase shares)

Outputs:
  results/speeding_hazard_base.csv     (osmid, lambda_base)
  results/speeding_hazard_fixture.csv.gz (fixture x segment expected totals)
  results/speeding_hazard.tex          (eta, R2, reference top-10)
  results/speeding_hazard_audit.json

Usage:
  python 46_speeding_hazard.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import prepare_metacount_frame, fit_metacount_nb

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")

SEGMENT_ROADS = os.path.join(DATA, "segment_roads_part0_5000.csv.gz")
FEATURES = os.path.join(DATA, "road_geometry_features.csv")
CAL = os.path.join(DATA, "fixture_calendar.csv")
SEG_PHASE = os.path.join(RES, "speeding_timewise_segment_phase.csv")
OUT_BASE = os.path.join(RES, "speeding_hazard_base.csv")
OUT_FIX = os.path.join(RES, "speeding_hazard_fixture.csv.gz")
OUT_TEX = os.path.join(RES, "speeding_hazard.tex")
OUT_JSON = os.path.join(RES, "speeding_hazard_audit.json")


def main():
    audit = {}
    print("=" * 80)
    print("46_speeding_hazard.py - speed-exceedance hazard (eq:hazard)")
    print("=" * 80)

    # ---- Stage 0: inputs -------------------------------------------------
    seg = pd.read_csv(SEGMENT_ROADS, usecols=["type", "ref_id", "phase", "road_osmid",
                                              "highway", "speeding", "trips", "ref_dt"])
    seg["ref_dt"] = pd.to_datetime(seg["ref_dt"])
    cal = pd.read_csv(CAL, parse_dates=["kickoff_iso"])
    att = dict(zip(cal["kickoff_iso"], cal["attendance"]))
    ko_of = seg[seg["type"] == "Event"].groupby("ref_id")["ref_dt"].first()
    KoAtt = {int(r): float(att.get(k, np.nan)) for r, k in ko_of.items()}
    assert all(np.isfinite(v) for v in KoAtt.values()), \
        f"unmatched event kickoffs: {[r for r, v in KoAtt.items() if not np.isfinite(v)]}"
    print(f"   events={len(KoAtt)} attendance range "
          f"{min(KoAtt.values()):,.0f}-{max(KoAtt.values()):,.0f}")
    audit["attendance"] = {"n": len(KoAtt), "min": min(KoAtt.values()),
                           "max": max(KoAtt.values())}

    ev = seg[seg["type"] == "Event"].copy()
    ev["att"] = ev["ref_id"].map(KoAtt)

    # ---- Stage A: segment baseline via metacountregressor NB-RP ----------
    print("\n[Stage A] segment baseline NB-RP (metacountregressor)...")
    agg = ev.groupby("road_osmid").agg(Y=("speeding", "sum"),
                                       trips=("trips", "sum"),
                                       highway=("highway", "first")).reset_index()
    agg = agg[agg["trips"] > 0].copy()
    if os.path.exists(FEATURES):
        feat = pd.read_csv(FEATURES).drop(columns=["highway"], errors="ignore")
        agg = agg.merge(feat, on="road_osmid", how="left")
        for c in ["maxspeed_kmh", "lanes", "dist_to_intersection_m"]:
            if c in agg.columns:
                agg[c] = agg[c].fillna(agg[c].median())
        for c in ["curvature_deg_km", "maxspeed_kmh", "lanes",
                  "dist_to_intersection_m"]:
            if c in agg.columns:
                sd = agg[c].std()
                agg[c + "_z"] = (agg[c] - agg[c].mean()) / (sd if sd and sd > 0 else 1.0)
    agg["highway"] = agg["highway"].fillna("unknown").astype(str)
    mc, _ = prepare_metacount_frame(agg, exposure_col="trips", highway_col="highway")
    fixed = ["logexp"] + [c for c in mc.columns if c.startswith("hw_")]
    for c in ["maxspeed_kmh_z", "curvature_deg_km_z"]:
        if c in mc.columns:
            fixed.append(c)
    rdm = [f"{c}:normal" for c in ["lanes_z", "dist_to_intersection_m_z"]
           if c in mc.columns]
    info = fit_metacount_nb(mc, y_col="Y", id_col="road_osmid",
                            fixed_terms=fixed, rdm_terms=rdm,
                            model="nb", R=150, de_maxiter=8,
                            de_popsize=8, verbose=True)
    agg["lambda_base"] = info["expected"]
    print(f"   AIC={info['aic']:.0f} alpha={info['alpha']:.3f} "
          f"[{info['spec_label']}] n={len(agg):,}")
    audit["stageA"] = {"aic": info["aic"], "bic": info["bic"],
                       "alpha": info["alpha"], "n_seg": len(agg),
                       "spec": info["spec_label"]}
    agg[["road_osmid", "lambda_base"]].to_csv(OUT_BASE, index=False)

    # ---- Stage B: crowd elasticity (16 fixtures) --------------------------
    print("\n[Stage B] crowd elasticity (log-log OLS, 16 fixtures)...")
    fx = ev.groupby("ref_id").agg(speed=("speeding", "sum"),
                                  att=("att", "first")).reset_index()
    fx["log_speed"] = np.log(fx["speed"] + 1)
    fx["log_att"] = np.log(fx["att"] + 1)
    ols = smf.ols("log_speed ~ log_att", data=fx).fit()
    eta = float(ols.params["log_att"])
    print(ols.summary())
    audit["stageB"] = {"eta": eta, "se": float(ols.bse["log_att"]),
                       "p": float(ols.pvalues["log_att"]),
                       "rsquared": float(ols.rsquared), "n": len(fx)}
    abar = float(np.mean(list(KoAtt.values())))

    # ---- Phase shares (Table 5 EB) ----------------------------------------
    sp = pd.read_csv(SEG_PHASE)
    sp = sp[sp["type"] == "Event"]
    tot = sp.groupby("phase")["eb"].sum()
    shares = (tot / tot.sum()).to_dict()
    print("   phase shares:", {k: round(float(v), 4) for k, v in shares.items()})
    audit["phase_shares"] = {k: float(v) for k, v in shares.items()}

    # ---- Hazard table for the 41 calendar fixtures -------------------------
    print("\n[Hazard] fixture x segment expectations...")
    base = agg[["road_osmid", "lambda_base"]]
    rows = []
    for _, f in cal.iterrows():
        scale = (float(f["attendance"]) / abar) ** eta
        h = base.copy()
        h["fixture"] = f["label"]
        h["attendance"] = float(f["attendance"])
        h["h_total"] = h["lambda_base"] * scale
        for ph, s in shares.items():
            h[f"h_{ph}"] = h["h_total"] * float(s)
        rows.append(h)
    hz = pd.concat(rows, ignore_index=True)
    hz.to_csv(OUT_FIX, index=False)
    print(f"   Wrote {OUT_FIX} ({len(hz):,} rows)")
    top = hz[hz["fixture"] == hz.iloc[0]["fixture"]].nlargest(10, "h_total")
    audit["top10_reference"] = [
        {"osmid": int(r["road_osmid"]), "h": float(r["h_total"])}
        for _, r in top.iterrows()]
    print("   top 10 @ reference fixture:")
    for d in audit["top10_reference"]:
        print(f"     {d['osmid']:<12} {d['h']:8.1f}/fixture")

    with open(OUT_TEX, "w") as f:
        f.write("% speeding_hazard.tex - auto-generated by 46_speeding_hazard.py\n")
        f.write("\\begin{equation}\n")
        f.write(f"\\log h_i(g,p) = \\hat\\beta_0 + \\hat\\beta_1\\log(E_i+1) "
                f"+ \\mathbf{{x}}_i'\\hat{{\\boldsymbol{{\\beta}}}} + "
                f"\\hat\\gamma_p + {eta:+.3f}\\,\\log(A_g+1) + b_i\n")
        f.write("\\end{equation}\n")
        f.write(f"% crowd elasticity eta={eta:.3f} (SE {ols.bse['log_att']:.3f}, "
                f"p={ols.pvalues['log_att']:.3f}, R2={ols.rsquared:.3f}, n=16); "
                f"NB-RP AIC={info['aic']:.0f} alpha={info['alpha']:.3f}\n")
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print("DONE! Speed-exceedance hazard complete.")


if __name__ == "__main__":
    main()
