"""
Direct Link: Near-Miss Events -> Crash Occurrence
===================================================
Uses metacountregressor's native JAX pipeline for:
  - Model building (build_model_from_manual_spec)
  - MLE optimization (scipy + JAX gradients)
  - Standard errors (jax.hessian via compute_standard_errors)
  - Summary output (print_summary)
  - Metaheuristic search (harmony_search)

Data: Z:/test_runs_tours/demand_/near_miss_ecf.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── metacountregressor imports ──────────────────────────────────
_MCR_PATH = r"C:\Users\ahernz\AppData\Local\Programs\Python\Python39\Lib\site-packages\metacountregressor"
if _MCR_PATH not in sys.path:
    sys.path.insert(0, _MCR_PATH)

import metacountregressor as _mcr
from main_hpc import compute_standard_errors
from main_hpc_lc_patch import (
    ModelSpec, build_param_index,
    build_model_from_manual_spec, mixed_model_loglik,
    print_summary,
)
import jax
import jax.numpy as jnp
from scipy.optimize import minimize as scipy_min
from scipy import stats as scipy_stats
from dataclasses import replace
from sklearn.preprocessing import StandardScaler

jax.config.update("jax_enable_x64", True)

# ── Paths ────────────────────────────────────────────────────────
DATA_PATH = r"Z:\test_runs_tours\demand_\near_miss_ecf.csv"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# 0. VARIABLE NAMES — human-readable
# ======================================================================
VAR_DICT = {
    "AADT":       "Annual Avg Daily Traffic",
    "SL":         "Speed Limit (km/h)",
    "NL":         "Number of Lanes",
    "SKD_SP":     "Skid Resistance",
    "CU_ST":      "Curvature: Straight",
    "CU_MOD":     "Curvature: Moderate",
    "CU_SHA":     "Curvature: Sharp",
    "CU_VSHA":    "Curvature: Very Sharp",
    "DEL_P":      "Delineation Posts Present",
    "IC":         "Intersection Conflict Present",
    "IP":         "Intersection Present",
    "MEDPM":      "Median Present",
    "MEDSFBar":   "Median Safety Barrier",
    "PA_COM1":    "Parking Access Commercial",
    "IV_VL":      "Interaction Volume: Very Low",
    "IV_L":       "Interaction Volume: Low",
    "IV_M":       "Interaction Volume: Medium",
    "IV_100":     "Interaction Volume: 100+",
    "PSPS1T24":   "Posted Speed: 1-24%",
    "PSPSNIL":    "Posted Speed: nil%",
    "PSDSG24":    "Speed Diff: >24%",
    "PSDS1T24":   "Speed Diff: 1-24%",
    "PSDSNIL":    "Speed Diff: nil%",
    "RSPSDL5":    "Roadside Sev: Pole <=5m",
    "RSDSDL1":    "Roadside Sev: Ditch <=1m",
    "RSDSDG10":   "Roadside Sev: Ditch >10m",
    "RSDSO":      "Roadside Sev: Other",
    "RSPSDL1":    "Roadside Sev: Pole <=1m",
    "RSDSDL5":    "Roadside Sev: Ditch <=5m",
    "RSPSDL10":   "Roadside Sev: Pole <=10m",
    "RSDSDL10":   "Roadside Sev: Ditch <=10m",
    "RSPSDG10":   "Roadside Sev: Pole >10m",
    "nm_braking":      "Near-Miss: Hard Braking Count",
    "nm_total":        "Near-Miss: Total Count",
    "nm_per_km":       "Near-Miss: Per km",
    "avg_accel_all":   "Avg Acceleration (G)",
    "avg_speed_all":   "Avg Speed (km/h)",
    "avg_iri_all":     "Avg IRI (Roughness)",
    "ExpCrashes":      "Expected Crashes (ECF)",
    "ObsCrashes":      "Observed Crashes",
    "segment_length_m":"Segment Length (m)",
}

def vname(col):
    return VAR_DICT.get(col, col)


# ======================================================================
# 1. LOAD AND CLEAN DATA
# ======================================================================
def load_and_clean():
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # ── Feature groups ──────────────────────────────────────────
    GEO = [
        "AADT", "SL", "NL", "SKD_SP",
        "CU_ST", "CU_MOD", "CU_SHA", "CU_VSHA",
        "DEL_P", "IC", "IP", "MEDPM", "MEDSFBar", "PA_COM1",
        "PSPS1T24", "PSPSNIL", "PSDSG24", "PSDS1T24", "PSDSNIL",
        "IV_VL", "IV_L", "IV_M", "IV_100",
        "RSPSDL5", "RSDSDL1", "RSDSDG10", "RSDSO",
        "RSPSDL1", "RSDSDL5", "RSPSDL10", "RSDSDL10", "RSPSDG10",
    ]
    GEO = [c for c in GEO if c in df.columns and df[c].std() > 1e-10]

    NM = [
        "nm_braking", "nm_total", "nm_per_km",
        "avg_accel_all", "avg_speed_all", "avg_iri_all",
    ]
    NM = [c for c in NM if c in df.columns]

    # ── Clean ────────────────────────────────────────────────────
    df = df.dropna(subset=["ObsCrashes"])
    df = df[df["ObsCrashes"] >= 0]
    for c in GEO:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)
    for c in NM:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(0)
    if "ExpCrashes" in df.columns:
        df["ExpCrashes"] = df["ExpCrashes"].fillna(0).clip(lower=0)

    # Derived
    df["seg_len_km"] = (df.get("segment_length_m", 1000.0) / 1000.0).clip(lower=0.001)
    df["log_len"]   = np.log(df["seg_len_km"])
    df["log_ecf"]   = np.log(np.maximum(df["ExpCrashes"], 1e-6))
    df["log_aadt"]  = np.log(np.maximum(df["AADT"], 1.0))

    print(f"  Clean rows:        {len(df):,}")
    print(f"  Has crashes:       {(df['ObsCrashes']>0).sum():,} ({df['ObsCrashes'].gt(0).mean()*100:.1f}%)")
    print(f"  Geometry features: {len(GEO)}")
    print(f"  Near-miss features:{len(NM)}")

    return df, GEO, NM


# ======================================================================
# 2. MODEL FITTING  (metacountregressor native pipeline)
# ======================================================================
def fit_model(df, feature_cols, y_col, offset_col,
              dispersion=0, label="Model", verbose=True):
    """
    Fit a Poisson/Negative-Binomial count regression using
    metacountregressor's native build_model_from_manual_spec +
    JAX-based mixed_model_loglik + jax.hessian for standard errors.

    Returns dict with ll, aic, bic, coef_table, result, etc.
    """

    # Prepare dataframe
    df_m = df.dropna(subset=feature_cols + [y_col, offset_col]).reset_index(drop=True)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_m[feature_cols].values.astype(float))
    for i, f in enumerate(feature_cols):
        df_m[f"__f{i}__"] = X_scaled[:, i]
    feat_std = [f"__f{i}__" for i in range(len(feature_cols))]

    y_raw = df_m[y_col].values.astype(float)
    df_m["_id"] = np.arange(len(df_m))
    df_m["_offset"] = df_m[offset_col].values.astype(float)

    # Build model spec via metacountregressor
    manual_spec = {
        "fixed_terms":     feat_std,
        "rdm_terms":       [],
        "rdm_cor_terms":   [],
        "grouped_terms":   [],
        "hetro_in_means":  [],
        "transformations": ["no"] * len(feat_std),
        "dispersion":      dispersion,
        "latent_classes":  1,
    }

    data, spec = build_model_from_manual_spec(
        df_m, manual_spec, id_col="_id", y_col=y_col, offset_col="_offset"
    )
    model_type = "nb" if dispersion else "poisson"
    spec = replace(spec, model=model_type)
    param_idx = build_param_index(spec)
    K = param_idx["total_params"]

    # ── Optimize via JAX + scipy L-BFGS-B ─────────────────────
    init = np.zeros(K)
    init[0] = np.log(max(y_raw.mean(), 0.01))
    if spec.model == "nb" and K > len(feat_std) + 1:
        init[-1] = 0.1

    def objective(params):
        return float(mixed_model_loglik(jnp.array(params), data, spec, indivi=False))

    res = scipy_min(objective, init, method="L-BFGS-B", options={"maxiter": 3000})
    if not res.success and verbose:
        print(f"  [{label}] warning: {res.message}")

    params = res.x
    n = len(df_m)
    k = K
    ll = -objective(params)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    # ── Standard errors via jax.hessian (metacountregressor's method) ──
    try:
        se = np.asarray(compute_standard_errors(jnp.array(params), objective))
        se = np.where(se > 10, np.nan, se)   # cap numerical noise
    except Exception:
        se = np.full(K, np.nan)

    zvals = np.where(np.isfinite(se) & (se > 1e-12), params / se, np.nan)
    pvals = np.where(np.isfinite(zvals),
                     2 * (1 - scipy_stats.norm.cdf(np.abs(zvals))), np.nan)

    # ── Build readable coefficient table ────────────────────────
    names = ["(Intercept)"] + [vname(f) for f in feature_cols]
    if spec.model == "nb" and K > len(names):
        names.append("ln(overdispersion)")

    coef_table = pd.DataFrame({
        "Variable": names[:K],
        "Estimate": params[:K],
        "Std.Err":  se[:K],
        "z":        zvals[:K],
        "p":        pvals[:K],
        "IRR":      np.exp(params[:K]),
        "Sig": ["" if np.isnan(p) or p >= 0.10 else
                "*" if p >= 0.05 else "**" if p >= 0.01 else "***"
                for p in pvals[:K]],
    })

    out = {
        "label": label, "ll": ll, "aic": aic, "bic": bic,
        "k": k, "n": n, "converged": res.success,
        "coef_table": coef_table, "result": res,
        "params": params, "se": se, "spec": spec, "data": data,
        "param_idx": param_idx, "objective": objective,
    }

    if verbose:
        model_str = "NB" if dispersion else "Poisson"
        print(f"\n{'='*65}")
        print(f"  {label}  ({model_str})")
        print(f"{'='*65}")
        print(f"  LL={ll:.2f}  AIC={aic:.2f}  BIC={bic:.2f}  k={k}  n={n}")
        print(f"\n  {'Variable':<42s} {'Est':>8s} {'IRR':>8s} {'z':>8s} {'p':>7s}  Sig")
        print(f"  {'-'*85}")
        for _, row in coef_table.iterrows():
            z_str = f"{row['z']:8.2f}" if np.isfinite(row['z']) else f"{'':>8s}"
            p_str = f"{row['p']:7.4f}" if np.isfinite(row['p']) else f"{'':>7s}"
            print(f"  {row['Variable']:<42s} {row['Estimate']:+8.4f} "
                  f"{row['IRR']:8.4f} {z_str} {p_str}  {row['Sig']}")

    return out


# ======================================================================
# 3. HARMONY SEARCH — find best subset of features
# ======================================================================
def harmony_search(df, candidates, y_col, offset_col,
                   max_iter=100, dispersion=1, verbose=True):
    """
    Custom Harmony Search over feature subsets.
    Evaluates each subset using the NB model (dispersion=1) via metacountregressor.
    """
    print(f"\n{'='*70}")
    print(f"  HARMONY SEARCH: Best Feature Subset")
    print(f"{'='*70}")
    print(f"  Candidates: {len(candidates)}  |  Max iter: {max_iter}  |  Model: NB")

    df_s = df.dropna(subset=candidates + [y_col, offset_col]).reset_index(drop=True)
    all_f = list(candidates)
    Kf = len(all_f)

    HMS = 20; HMCR = 0.85; PAR = 0.30
    np.random.seed(42)

    def rand_mask(mins=2, maxs=min(8, Kf)):
        s = np.random.randint(mins, maxs + 1)
        m = np.zeros(Kf, dtype=int)
        m[np.random.choice(Kf, s, replace=False)] = 1
        return m

    def evaluate(mask):
        sel = [all_f[i] for i in range(Kf) if mask[i] == 1]
        if len(sel) < 2:
            return 1e12
        o = fit_model(df_s, sel, y_col, offset_col,
                      dispersion=dispersion, label="HS", verbose=False)
        return o["bic"] if o else 1e12

    mem_m = [rand_mask() for _ in range(HMS)]
    mem_s = [evaluate(m) for m in mem_m]
    best_s = min(mem_s); best_m = mem_m[mem_s.index(best_s)].copy()
    history = []

    for it in range(max_iter):
        new = np.zeros(Kf, dtype=int)
        for i in range(Kf):
            if np.random.random() < HMCR:
                new[i] = mem_m[np.random.randint(HMS)][i]
                if np.random.random() < PAR:
                    new[i] = 1 - new[i]
            else:
                new[i] = np.random.randint(0, 2)
        if new.sum() < 2:
            new[np.random.choice(Kf, 1)] = 1

        ns = evaluate(new)
        wi = np.argmax(mem_s)
        if ns < mem_s[wi]:
            mem_m[wi] = new.copy(); mem_s[wi] = ns
            if ns < best_s:
                best_s = ns; best_m = new.copy()
                if verbose and it % 15 == 0:
                    sel = [vname(all_f[i]) for i in range(Kf) if best_m[i] == 1]
                    print(f"  Iter {it:4d}: BIC={best_s:.2f},  features={len(sel)}")
        history.append(best_s)

    best_feats = [all_f[i] for i in range(Kf) if best_m[i] == 1]

    print(f"\n  {'─'*60}")
    print(f"  BEST HARMONY SEARCH RESULT")
    print(f"  {'─'*60}")
    print(f"  BIC: {best_s:.2f}  |  {len(best_feats)} features")
    for f in best_feats:
        print(f"    • {vname(f)}")

    final = fit_model(df_s, best_feats, y_col, offset_col,
                      dispersion=dispersion, label="HS-BEST", verbose=True)
    return final, history, best_feats


# ======================================================================
# 4. MAIN
# ======================================================================
def main():
    print("\n" + "=" * 70)
    print("  DIRECT LINK: Near-Miss Events -> Crash Occurrence")
    print("  metacountregressor + JAX pipeline")
    print("=" * 70)

    df, GEO, NM = load_and_clean()
    results = {}

    # ══════════════════════════════════════════════════════════════
    # MODEL A: NEAR-MISS ONLY  (NB)
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "#" * 70)
    print("#  MODEL A: NEAR-MISS -> Observed Crashes (NB)")
    print("#  Can near-miss metrics alone predict crashes?")
    print("#" * 70)

    rA = fit_model(df, NM, "ObsCrashes", "log_len", dispersion=1,
                   label="A: Near-Miss Only (NB)")
    if rA: results["A_NM_NB"] = rA

    # ══════════════════════════════════════════════════════════════
    # MODEL B: ROAD GEOMETRY ONLY  (NB)
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "#" * 70)
    print("#  MODEL B: ROAD GEOMETRY -> Observed Crashes (NB)")
    print("#  Conventional approach: predict crashes from geometry alone.")
    print("#" * 70)

    rB = fit_model(df, GEO, "ObsCrashes", "log_len", dispersion=1,
                   label="B: Geometry Only (NB)")
    if rB: results["B_Geo_NB"] = rB

    # ══════════════════════════════════════════════════════════════
    # MODEL C: GEOMETRY + NEAR-MISS  (NB)
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "#" * 70)
    print("#  MODEL C: GEOMETRY + NEAR-MISS -> Observed Crashes (NB)")
    print("#  Does adding near-miss data improve the geometry model?")
    print("#" * 70)

    rC = fit_model(df, GEO + NM, "ObsCrashes", "log_len", dispersion=1,
                   label="C: Geometry + Near-Miss (NB)")
    if rC: results["C_Geo_NM_NB"] = rC

    # ══════════════════════════════════════════════════════════════
    # MODEL D: ECF BENCHMARK  (how good is the official SPF?)
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "#" * 70)
    print("#  MODEL D: ECF BENCHMARK")
    print("#  How well does the official SPF (ExpCrashes offset) predict?")
    print("#" * 70)

    # Manual ECF-offset Poisson via scipy
    dfD = df.dropna(subset=["ObsCrashes", "log_ecf"])
    yD = dfD["ObsCrashes"].values.astype(float)
    oD = dfD["log_ecf"].values

    def ecf_obj(p):
        lam = np.exp(p[0] + oD)
        from scipy.special import gammaln
        return -np.sum(yD * np.log(np.maximum(lam, 1e-15)) - lam - gammaln(yD + 1))

    rD = scipy_min(ecf_obj, [np.log(max(yD.mean(), 0.01))], method="L-BFGS-B")
    nD = len(dfD)
    llD = -rD.fun; kD = 1
    bicD = kD * np.log(nD) - 2 * llD; aicD = 2 * kD - 2 * llD

    results["D_ECF"] = {
        "label": "D: ECF Benchmark (offset-only)", "ll": llD,
        "aic": aicD, "bic": bicD, "k": kD, "n": nD, "converged": rD.success,
    }
    print(f"\n  ECF Benchmark: LL={llD:.2f}  AIC={aicD:.2f}  "
          f"BIC={bicD:.2f}  conv={rD.success}")

    # ══════════════════════════════════════════════════════════════
    # MODEL E: HARMONY SEARCH  (Geo + NM)
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "#" * 70)
    print("#  MODEL E: HARMONY SEARCH (Geo + Near-Miss)")
    print("#  Metaheuristic search for optimal feature subset.")
    print("#" * 70)

    rE, hs_hist, hs_feats = harmony_search(
        df, GEO + NM, "ObsCrashes", "log_len",
        max_iter=100, dispersion=1, verbose=True
    )
    if rE:
        results["E_HS"] = rE
        results["E_HS_features"] = hs_feats

    # ══════════════════════════════════════════════════════════════
    # 5. MODEL COMPARISON
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)

    rows = []
    for k, r in results.items():
        if isinstance(r, dict) and "label" in r and "bic" in r:
            rows.append({
                "Model": r["label"], "LL": r.get("ll"), "AIC": r.get("aic"),
                "BIC": r.get("bic"), "k": r.get("k"), "n": r.get("n"),
                "Converged": r.get("converged"),
            })

    comp = pd.DataFrame(rows).sort_values("BIC")
    best_bic = comp["BIC"].min()
    comp["delta_BIC"] = comp["BIC"] - best_bic
    comp["Rank"] = range(1, len(comp) + 1)

    print(f"\n  {'Rank':<5s} {'Model':<42s} {'BIC':>10s} {'AIC':>10s} "
          f"{'LL':>10s} {'k':>5s} {'dBIC':>10s}")
    print(f"  {'-'*100}")
    for _, row in comp.iterrows():
        print(f"  {row['Rank']:<5d} {row['Model']:<42s} {row['BIC']:10.2f} "
              f"{row['AIC']:10.2f} {row['LL']:10.2f} {row['k']:5d} "
              f"{row['delta_BIC']:10.2f}")

    # ══════════════════════════════════════════════════════════════
    # 6. INTERPRETATION
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    ecf_bic = comp[comp["Model"].str.contains("ECF")]["BIC"].values
    nm_bic  = comp[comp["Model"] == "A: Near-Miss Only (NB)"]["BIC"].values
    geo_bic = comp[comp["Model"] == "B: Geometry Only (NB)"]["BIC"].values
    combo_bic = comp[comp["Model"] == "C: Geometry + Near-Miss (NB)"]["BIC"].values

    if len(ecf_bic) > 0:
        print(f"\n  1. ECF (Expected Crash Frequency) Benchmark BIC: {ecf_bic[0]:.2f}")
        print("     This represents the official SPF model's predictive power.")

    if len(nm_bic) > 0 and len(ecf_bic) > 0:
        delta = nm_bic[0] - ecf_bic[0]
        print(f"\n  2. Near-Miss Only BIC: {nm_bic[0]:.2f}  (delta vs ECF: {delta:+.2f})")
        print(f"     Near-misses alone are {'COMPARABLE to' if delta < 100 else 'WEAKER than'} ECF.")

    if len(geo_bic) > 0 and len(ecf_bic) > 0:
        delta = geo_bic[0] - ecf_bic[0]
        print(f"\n  3. Road Geometry Only BIC: {geo_bic[0]:.2f}  (delta vs ECF: {delta:+.2f})")

    if len(combo_bic) > 0 and len(geo_bic) > 0:
        delta = combo_bic[0] - geo_bic[0]
        print(f"\n  4. Adding Near-Miss to Geometry:")
        print(f"     Geometry BIC:          {geo_bic[0]:.2f}")
        print(f"     Geometry + Near-Miss:  {combo_bic[0]:.2f}")
        print(f"     Improvement: {delta:+.2f} BIC units")
        if delta < -7:
            print("     -> Near-miss data STRONGLY improves crash prediction!")
        elif delta < -2:
            print("     -> Near-miss data provides MODEST improvement.")
        elif delta < 0:
            print("     -> Near-miss data provides WEAK improvement.")
        else:
            print("     -> Near-miss data does NOT improve the geometry model.")

    # Significant near-miss predictors from Model C
    if rC and rC["coef_table"] is not None:
        nm_terms = rC["coef_table"][rC["coef_table"]["Variable"].str.contains("Near-Miss", na=False)]
        nm_sig = nm_terms[nm_terms["p"] < 0.10]
        if len(nm_sig) > 0:
            print(f"\n  5. Significant Near-Miss Predictors (p < 0.10):")
            for _, row in nm_sig.iterrows():
                d = "increase" if row["IRR"] > 1 else "decrease"
                print(f"     • {row['Variable']}: IRR={row['IRR']:.4f} "
                      f"({d}s crash risk, p={row['p']:.4f})")

    # Harmony search features
    if "E_HS" in results:
        print(f"\n  6. Harmony Search Best Model BIC: {results['E_HS']['bic']:.2f}")
        print(f"     Selected {len(hs_feats)} features from {len(GEO+NM)} candidates.")

    comp.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(f"\n\n  Results saved to model_comparison.csv")
    print(f"  Output: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
