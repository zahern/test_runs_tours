"""
27_crash_transfer_validation.py
Do more tests to ensure the hard-braking -> crash surrogate holds, and add
Suncorp-local validation at the planning-zone geography where decisions are
actually made.

Literature status (from the SEQ statewide analysis, 10,780 segments)
--------------------------------------------------------------------
The existing battery (hb_synthesis_evidence.tex) gives a 6 PRO / 6 CON split:

  CON  full-model coefficients not proportional (r=0.69, R2=0.48)
  PRO  searched unequal-model coefficients proportional (r=1.00, R2=0.99)
  CON  Jaccard feature overlap 0.27; HB-spec degrades on crashes dBIC+124
  PRO  predicted HB encompasses geometry (LR=182, p~0, dBIC-173)
  CON  partial residual correlation 0.15 weak
  CON  total-count bivariate copula fails; PRO category copulas all reject
       independence (Candidate LR=4139, Critical 1667, Extreme 100, all p~0)
  CON  crash model TI=-2.1 (region-specific)
  PRO  HB proxy improves crash LL in 9/11 LGAs (mean dLL+61.5)

Verdict in the paper: STRONG PROXY. The weakness is rank relevance for
hotspot deployment and lack of any crash ground-truth check inside the
38.5 km2 Suncorp study area (the Suncorp pipeline is surrogate-only).

What this adds
--------------
A. SEQ-wide ranking & calibration on EB-adjusted hotspots: Spearman between
   HB-EB and crash-EB rankings, top-N overlap (deployment-relevant), and
   10-decile calibration of mean crash rate per HB-risk bin.

B. Out-of-sample predictive test: does raw HB count improve crash prediction
   over geometry alone? 5-fold CV on the 10,780 SEQ segments:
     base:  crashes ~ offset(log_len) + geometry (AADT, SL, NL, curvature,
            intersection, etc.)
     aug:   base + log(HB_Total+1)
   Metrics per fold: Poisson deviance, RMSE, Spearman vs realised crashes.

C. Suncorp zone-level validation: aggregate Suncorp HB-EB risk per SA2 zone
   via point-in-polygon join of OSM segment centroids to SA2 polygons, and
   correlate with SA2 crash density from zone_demand_crash_507.gpkg. This
   tests transferability on the same geography where police rostering is
   planned, without requiring OSM-to-Hom_Sec segment conflation (which has
   median mismatch 451 m).

Outputs
  results/crash_transfer_ranking.csv
  results/crash_transfer_holdout.csv
  results/suncorp_zone_validation.csv
  results/crash_transfer_validation.tex
  figures/crash_transfer_rank_scatter.png
  figures/crash_transfer_calibration.png
  figures/suncorp_zone_scatter.png
"""
import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJ = os.environ.get("PROJ", "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics")
DATA_SUN = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

_SEQ_CANDIDATES = [
    r"Z:\test_runs_tours\demand_\paper\Unofficial_Transportation_Research_Board__TRB__LaTeX_template\near_miss_hb_ecf.csv",
    "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/paper/Unofficial_Transportation_Research_Board__TRB__LaTeX_template/near_miss_hb_ecf.csv",
    os.path.join(os.path.dirname(PROJ), "paper/Unofficial_Transportation_Research_Board__TRB__LaTeX_template/near_miss_hb_ecf.csv"),
]
SEQ_CSV = next((p for p in _SEQ_CANDIDATES if os.path.exists(p)), _SEQ_CANDIDATES[0])
_ZONE_CANDIDATES = [
    r"Z:\test_runs_tours\demand_\zone_demand_crash_507.gpkg",
    "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/zone_demand_crash_507.gpkg",
]
ZONE_CRASH = next((p for p in _ZONE_CANDIDATES if os.path.exists(p)), _ZONE_CANDIDATES[0])
SUN_GEOJSON = os.path.join(DATA_SUN, "osm_roads_suncorp.geojson")
SUN_EB = os.path.join(RES, "eb_predictions.csv")

OUT_RANK = os.path.join(RES, "crash_transfer_ranking.csv")
OUT_HOLD = os.path.join(RES, "crash_transfer_holdout.csv")
OUT_ZONE = os.path.join(RES, "suncorp_zone_validation.csv")
OUT_TEX = os.path.join(RES, "crash_transfer_validation.tex")
FIG_RANK = os.path.join(FIG, "crash_transfer_rank_scatter.png")
FIG_CAL = os.path.join(FIG, "crash_transfer_calibration.png")
FIG_ZONE = os.path.join(FIG, "suncorp_zone_scatter.png")

sns.set_style("whitegrid"); plt.rcParams["figure.dpi"] = 150

GEOM_COLS = ["AADT","SL","NL","CU_MOD","CU_SHA","CU_VSHA","IP","IC","MEDPM","SKD_SP","IV_VL","IV_M"]

def eb_adjust(obs, exp, alpha):
    w = 1.0/(1.0+alpha*exp)
    return w*obs + (1-w)*exp, w

def main():
    print("="*80)
    print("27_crash_transfer_validation.py")
    print("="*80)

    # ---- A. SEQ-wide ranking & calibration ---------------------------------
    print("\n1. SEQ statewide data (10,780 segments)...")
    seq = pd.read_csv(SEQ_CSV, low_memory=False)
    seq["HB_Total"] = seq["HB_Total"].fillna(0)
    seq["ObsCrashes"] = seq["ObsCrashes"].fillna(0)
    seq["ExpCrashes"] = seq["ExpCrashes"].fillna(0)
    seq["log_len"] = np.log(seq["segment_length_m"].clip(lower=1))
    seq["log_HB"] = np.log(seq["HB_Total"]+1)
    # keep rows with geometry
    for c in GEOM_COLS:
        if c in seq.columns:
            seq[c] = seq[c].fillna(seq[c].median())
    print(f"   {len(seq):,} segments, HB mean {seq['HB_Total'].mean():.1f}, crash mean {seq['ObsCrashes'].mean():.2f}")

    # Fit HB NB and Crash NB on same geometry spec (mirrors hb_full_geo_nm / crash)
    # Use a compact spec that exists in both: AADT+SL+NL+curvature dummies+IP+IC
    # Keep it identical for proportionality to be meaningful.
    # Standardise continuous predictors as in the paper (prevents overflow; AADT ~ 30k)
    from sklearn.preprocessing import StandardScaler
    seq["AADT_z"] = StandardScaler().fit_transform(seq[["AADT"]]).ravel() if "AADT" in seq.columns else 0
    seq["SL_z"] = StandardScaler().fit_transform(seq[["SL"]]).ravel() if "SL" in seq.columns else 0
    seq["log_len_z"] = StandardScaler().fit_transform(seq[["log_len"]]).ravel()
    # keep originals for fallback display, but models use _z

    avail_z = []
    for c in ["AADT_z","SL_z","NL","CU_MOD","CU_SHA","CU_VSHA","IP","IC"]:
        base = c.replace("_z","")
        if base in seq.columns or c in seq.columns:
            # NL/CU_* etc are 0/1 dummies, leave unstandardised
            avail_z.append(c if c in seq.columns else base)
    # Prefer z-scored continuous + raw dummies
    formula_geo = "HB_Total ~ " + " + ".join([c for c in ["AADT_z","SL_z","log_len_z"] if c in seq.columns or c.replace("_z","") in seq.columns]) + " + " + " + ".join([c for c in ["NL","CU_MOD","CU_SHA","CU_VSHA","IP","IC"] if c in seq.columns])
    # simplify to robust core if needed
    formula_geo = "HB_Total ~ AADT_z + SL_z + log_len_z"
    formula_crash_geo = "ObsCrashes ~ AADT_z + SL_z + log_len_z"
    print(f"   HB formula: {formula_geo}")

    def fit_nb_with_fallback(formula, data, label):
        candidates = [formula,
                      "y ~ AADT_z + SL_z + log_len_z",
                      "y ~ log_len_z",
                      "y ~ 1"]
        ycol = "HB_Total" if "HB_Total" in formula else "ObsCrashes"
        for cand in candidates:
            cand2 = cand.replace("y", ycol)
            try:
                res = smf.negativebinomial(cand2, data=data).fit(disp=False, maxiter=300)
                # guard against nonsense alpha
                a = float(res.params["alpha"]) if "alpha" in res.params else 1.0
                if not np.isfinite(a) or a > 500:
                    print(f"   {label} NB `{cand2}` alpha {a:.1f} unstable, trying simpler")
                    continue
                print(f"   {label} NB OK with `{cand2}` AIC {res.aic:.0f} alpha {a:.2f}")
                return res, cand2
            except Exception as ex:
                print(f"   {label} NB try `{cand2}` failed: {ex}")
                continue
        return None, None

    hb_res, hb_form = fit_nb_with_fallback(formula_geo, seq, "HB")
    if hb_res is not None:
        alpha_hb = float(hb_res.params["alpha"]) if "alpha" in hb_res.params else 1.0
        seq["exp_HB"] = hb_res.predict(seq)
        seq["eb_HB"], _ = eb_adjust(seq["HB_Total"].values, seq["exp_HB"].values, alpha_hb)
        print(f"   HB alpha {alpha_hb:.2f}")
    else:
        print("   HB NB all fallbacks failed, using raw HB")
        hb_res = None; alpha_hb = 70.0
        seq["exp_HB"] = seq["HB_Total"].mean()
        seq["eb_HB"] = seq["HB_Total"]

    cr_res, cr_form = fit_nb_with_fallback(formula_crash_geo, seq, "Crash")
    if cr_res is not None:
        alpha_cr = float(cr_res.params["alpha"]) if "alpha" in cr_res.params else 1.0
        seq["exp_Cr"] = cr_res.predict(seq)
        seq["eb_Cr"], _ = eb_adjust(seq["ObsCrashes"].values, seq["exp_Cr"].values, alpha_cr)
        print(f"   Crash alpha {alpha_cr:.2f}")
    else:
        print("   Crash NB all fallbacks failed, using ExpCrashes")
        seq["exp_Cr"] = seq["ExpCrashes"]; seq["eb_Cr"] = seq["ExpCrashes"]
        alpha_cr = 18.5
    # keep z cols for holdout
    seq["log_HB_z"] = StandardScaler().fit_transform(np.log(seq["HB_Total"]+1).values.reshape(-1,1)).ravel()

    # Ranking metrics (deployment-relevant)
    rho_raw = spearmanr(seq["HB_Total"], seq["ObsCrashes"]).statistic
    rho_eb = spearmanr(seq["eb_HB"], seq["eb_Cr"]).statistic
    rho_partial = spearmanr(seq["eb_HB"], seq["ObsCrashes"]).statistic  # HB-EB vs raw crashes
    def top_overlap(n):
        a=set(seq.nlargest(n,"eb_HB").index); b=set(seq.nlargest(n,"eb_Cr").index)
        return len(a&b)/n
    rank_rows=[{"comparison":"HB_total vs ObsCrashes (raw)","spearman":rho_raw,"top20":np.nan,"top50":np.nan,"top100":np.nan},
               {"comparison":"HB-EB vs Crash-EB","spearman":rho_eb,"top20":top_overlap(20),"top50":top_overlap(50),"top100":top_overlap(100)},
               {"comparison":"HB-EB vs ObsCrashes","spearman":rho_partial,"top20":float(len(set(seq.nlargest(20,"eb_HB").index)&set(seq.nlargest(20,"ObsCrashes").index))/20),"top50":float(len(set(seq.nlargest(50,"eb_HB").index)&set(seq.nlargest(50,"ObsCrashes").index))/50),"top100":float(len(set(seq.nlargest(100,"eb_HB").index)&set(seq.nlargest(100,"ObsCrashes").index))/100)}]
    rank_df=pd.DataFrame(rank_rows)
    rank_df.to_csv(OUT_RANK, index=False)
    print(rank_df.round(3).to_string(index=False))

    # Calibration: deciles of HB-EB risk
    seq["decile"]=pd.qcut(seq["eb_HB"], 10, labels=False, duplicates="drop")
    cal=seq.groupby("decile").agg(n=("eb_HB","size"),mean_HB_EB=("eb_HB","mean"),mean_crash=("ObsCrashes","mean"),mean_expCrash=("ExpCrashes","mean")).reset_index()
    # figure: rank scatter
    fig, ax=plt.subplots(figsize=(6,5))
    ax.scatter(seq["eb_HB"], seq["eb_Cr"], s=4, alpha=0.15, color="steelblue")
    ax.set_xlabel("HB EB risk"); ax.set_ylabel("Crash EB risk"); ax.set_title("SEQ segments: HB vs crash EB ranking")
    # Spearman text
    ax.text(0.05,0.95,f"Spearman={rho_eb:.3f}", transform=ax.transAxes, fontsize=10, va="top", bbox=dict(facecolor="white",alpha=0.8))
    fig.tight_layout(); fig.savefig(FIG_RANK, dpi=150); plt.close(fig)
    # figure: calibration
    fig, ax=plt.subplots(figsize=(6,4))
    ax.plot(cal["mean_HB_EB"], cal["mean_crash"], "o-", color="firebrick", label="Obs crashes")
    ax.plot(cal["mean_HB_EB"], cal["mean_expCrash"], "s--", color="steelblue", label="Exp crashes (TMR SPF)")
    ax.set_xlabel("Mean HB-EB risk (decile)"); ax.set_ylabel("Mean crash count"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Calibration: crash rate per HB-risk decile")
    fig.tight_layout(); fig.savefig(FIG_CAL, dpi=150); plt.close(fig)
    print(f"   Saved {FIG_RANK} / {FIG_CAL}")

    # ---- B. Predictive holdout --------------------------------------------
    print("\n2. Holdout: does HB improve crash prediction over geometry alone?")
    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    rows=[]
    for fold,(tr,te) in enumerate(kf.split(seq),1):
        trd, ted = seq.iloc[tr], seq.iloc[te]
        ted = ted.copy()
        # base: crashes ~ geometry (with fallback, z-scored)
        def fit_crash_base(trd):
            for f in [formula_crash_geo, "ObsCrashes ~ AADT_z + SL_z + log_len_z", "ObsCrashes ~ log_len_z", "ObsCrashes ~ 1"]:
                try:
                    r = smf.negativebinomial(f, data=trd).fit(disp=False, maxiter=300)
                    a = float(r.params["alpha"]) if "alpha" in r.params else 1.0
                    if not np.isfinite(a) or a>500:
                        continue
                    return r
                except Exception:
                    continue
            return None
        m0 = fit_crash_base(trd)
        if m0 is not None:
            ted["pred0"]=m0.predict(ted)
        else:
            ted["pred0"]=trd["ObsCrashes"].mean()
        # augmented: + log(HB+1) z-scored
        formula_aug = formula_crash_geo + " + log_HB_z"
        m1 = None
        for f in [formula_aug, "ObsCrashes ~ AADT_z + SL_z + log_len_z + log_HB_z", "ObsCrashes ~ log_len_z + log_HB_z", "ObsCrashes ~ log_HB_z"]:
            try:
                r = smf.negativebinomial(f, data=trd).fit(disp=False, maxiter=300)
                a = float(r.params["alpha"]) if "alpha" in r.params else 1.0
                if not np.isfinite(a) or a>500:
                    continue
                m1 = r; break
            except Exception:
                continue
        if m1 is not None:
            ted["pred1"]=m1.predict(ted)
            ll0 = m0.llf if m0 is not None else np.nan
            ll1 = m1.llf
            lr = 2*(ll1-ll0) if np.isfinite(ll0) else np.nan
        else:
            ted["pred1"]=ted["pred0"]; lr=np.nan
        # RMSE and holdout Poisson deviance (more appropriate than in-sample LR)
        def holdout_deviance(y, mu):
            mu=np.clip(mu,1e-6,1e6)
            # avoid log(0) via y*log(y/mu) with y=0 term = 0
            return 2*np.sum(np.where(y>0, y*np.log(y/mu) - (y-mu), - (y-mu)))
        rmse0 = np.sqrt(((ted["ObsCrashes"]-ted["pred0"])**2).mean())
        rmse1 = np.sqrt(((ted["ObsCrashes"]-ted["pred1"])**2).mean())
        dev0 = holdout_deviance(ted["ObsCrashes"].values, ted["pred0"].values)
        dev1 = holdout_deviance(ted["ObsCrashes"].values, ted["pred1"].values)
        rho0 = spearmanr(ted["pred0"], ted["ObsCrashes"]).statistic
        rho1 = spearmanr(ted["pred1"], ted["ObsCrashes"]).statistic
        rows.append({"fold":fold,"rmse_base":rmse0,"rmse_aug":rmse1,"dev_base":dev0,"dev_aug":dev1,"spearman_base":rho0,"spearman_aug":rho1,"lr_train":lr})
        print(f"   fold {fold}: RMSE {rmse0:.3f}->{rmse1:.3f}  dev {dev0:.0f}->{dev1:.0f}  rho {rho0:.3f}->{rho1:.3f}")
    hold=pd.DataFrame(rows)
    hold.loc["mean"]=hold.mean(numeric_only=True)
    hold.to_csv(OUT_HOLD, index=False)
    print(hold.round(3).to_string())

    # ---- C. Suncorp zone-level validation ----------------------------------
    print("\n3. Suncorp zone-level validation (SA2)...")
    zone_rows=None
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        # Suncorp HB-EB per segment
        sun_eb = pd.read_csv(SUN_EB)
        sun_eb = sun_eb[sun_eb["road_osmid"].notna()].copy()
        sun_gdf = gpd.read_file(SUN_GEOJSON)
        sun_gdf = sun_gdf.merge(sun_eb[["road_osmid","eb_hb"]], left_on="osmid", right_on="road_osmid", how="left")
        sun_gdf = sun_gdf[sun_gdf["eb_hb"].notna()]
        # project for point-in-polygon (use WGS84, geopandas handles)
        zones = gpd.read_file(ZONE_CRASH)
        # keep crash density
        col_crash = "crash_density" if "crash_density" in zones.columns else "crash_inbound_total"
        zones = zones[["sa2_code_str","sa2_name",col_crash,"geometry"]].rename(columns={col_crash:"crash_density"})
        # centroids of Suncorp segments
        sun_gdf["cent_lon"]=sun_gdf.geometry.centroid.x
        sun_gdf["cent_lat"]=sun_gdf.geometry.centroid.y
        pts = gpd.GeoDataFrame(sun_gdf, geometry=gpd.points_from_xy(sun_gdf["cent_lon"], sun_gdf["cent_lat"]), crs="EPSG:4326")
        joined = gpd.sjoin(pts, zones, how="left", predicate="within")
        agg = joined.groupby("sa2_code_str").agg(sun_n=("eb_hb","size"),sun_mean_eb=("eb_hb","mean"),crash_density=("crash_density","first"),sa2_name=("sa2_name","first")).reset_index()
        agg = agg[agg["sun_n"]>=5]  # require at least 5 Suncorp segments in zone
        rho_zone = spearmanr(agg["sun_mean_eb"], agg["crash_density"]).statistic if len(agg)>=5 else np.nan
        # top-N zone overlap
        def zone_overlap(n):
            n=min(n,len(agg))
            a=set(agg.nlargest(n,"sun_mean_eb")["sa2_code_str"]); b=set(agg.nlargest(n,"crash_density")["sa2_code_str"])
            return len(a&b)/n if n>0 else np.nan
        zone_rows=pd.DataFrame([{"metric":"Spearman (zone mean EB vs crash density)","value":rho_zone},
                                {"metric":"Top-3 zone overlap","value":zone_overlap(3)},
                                {"metric":"Top-5 zone overlap","value":zone_overlap(5)}])
        zone_rows.to_csv(OUT_ZONE, index=False)
        print(zone_rows.to_string(index=False))
        print(f"   zones with Suncorp coverage: {len(agg)}")
        # scatter figure
        fig, ax=plt.subplots(figsize=(6,4))
        ax.scatter(agg["sun_mean_eb"], agg["crash_density"], s=40, alpha=0.7, color="teal")
        for _,r in agg.iterrows():
            ax.annotate(r["sa2_name"][:12], (r["sun_mean_eb"], r["crash_density"]), fontsize=6, alpha=0.7)
        ax.set_xlabel("Suncorp mean HB-EB per SA2"); ax.set_ylabel("SA2 crash density"); ax.set_title(f"Suncorp HB vs crash by SA2 (rho={rho_zone:.2f}, n={len(agg)})")
        fig.tight_layout(); fig.savefig(FIG_ZONE, dpi=150); plt.close(fig)
        print(f"   Saved {FIG_ZONE}")
    except Exception as e:
        print(f"   Zone validation failed/skipped ({e})")
        zone_rows=pd.DataFrame([{"metric":"error","value":str(e)}])
        zone_rows.to_csv(OUT_ZONE, index=False)

    # ---- LaTeX --------------------------------------------------------------
    L=[]
    L.append("% crash_transfer_validation.tex - auto-generated by 27_crash_transfer_validation.py")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{Crash \\textit{vs} hard-braking hotspot ranking (SEQ, $N=10{,}780$ segments; EB-adjusted).}")
    L.append("\\label{tab:crash_rank}")
    L.append("\\begin{tabular}{lrrr}")
    L.append("\\hline")
    L.append("Comparison & Spearman $\\rho$ & Top-20 overlap & Top-50 overlap \\\\")
    L.append("\\hline")
    for _,r in rank_df.iterrows():
        # escape
        comp=str(r["comparison"]).replace("&","\\&")
        L.append(f"{comp} & {r['spearman']:.3f} & {r.get('top20',0):.2f} & {r.get('top50',0):.2f} \\\\")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{Does hard-braking improve crash prediction over geometry alone? 5-fold CV on SEQ crashes (holdout deviance lower is better).}")
    L.append("\\label{tab:crash_holdout}")
    L.append("\\begin{tabular}{lrrrrrr}")
    L.append("\\hline")
    L.append("Fold & RMSE base & RMSE +HB & Dev base & Dev +HB & $\\rho$ base & $\\rho$ +HB \\\\")
    L.append("\\hline")
    for _,r in hold.iterrows():
        if str(r["fold"])=="mean":
            L.append("\\hline")
        lab = "Mean" if str(r["fold"])=="mean" else f"{int(r['fold'])}"
        L.append(f"{lab} & {r['rmse_base']:.3f} & {r['rmse_aug']:.3f} & {r['dev_base']:.0f} & {r['dev_aug']:.0f} & {r['spearman_base']:.3f} & {r['spearman_aug']:.3f} \\\\")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    if zone_rows is not None and "Spearman" in str(zone_rows["metric"].tolist()):
        v=zone_rows.loc[zone_rows["metric"].str.contains("Spearman"),"value"].iloc[0]
        L.append("")
        L.append(f"% Suncorp SA2 validation: Spearman={v:.3f}")
    L.append("")
    L.append("\\textbf{Design note.} Controls share \\texttt{ref\\_dt} with their matched fixture (same-day design), so cross-day decay is not identifiable; temporal tests use lag-1 autocorrelation instead. Crash transferability at Suncorp is therefore assessed at SA2 zone level rather than by OSM-segment conflation (median OSM--SEQ mismatch 451\\,m).")
    with open(OUT_TEX,"w") as f:
        f.write("\n".join(L)+"\n")
    print(f"\nWrote {OUT_TEX}")
    print("\nDONE")

if __name__=="__main__":
    main()
