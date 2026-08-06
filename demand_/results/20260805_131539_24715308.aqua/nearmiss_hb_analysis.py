"""
Hard-Braking Category Proxy Variable Analysis
==============================================
Tests whether hard-braking category counts (Candidate / Critical-Likely / Extreme)
can serve as proxy/leading indicators for crash frequency at the road-segment level.

Econometric Framework:
  1. Differential Evolution — find best geometry features for HB vs crashes
  2. MCR Structure Search — roles: fixed / random-indep / random-corr
  3. Full models (all geometry baseline)
  4. Coefficient Proportionality Test
  5. Specification Transfer
  6. Encompassing Test
  7. Bivariate latent risk correlation
  8. Bivariate Copula (IFM + Joint MLE + Copula Structure Search)
  9. Random Parameters NB
  A. Multivariate HB-category analysis (Critical/Emergency/Extreme jointly with Crashes)

Data: near_miss_hb_ecf.csv (aggregated from SEQ_Expected_Crashes_with_HardBraking.gpkg)

RESUME SUPPORT: Each step saves a checkpoint CSV. On re-run, completed steps
are detected and skipped. Delete the relevant checkpoint to force re-computation.
"""

import os, sys, warnings, argparse, json
import numpy as np
import pandas as pd
import scipy.special as sp_special

warnings.filterwarnings("ignore")
os.environ["JAX_LOG_COMPILES"] = "0"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metacountregressor as _mcr
from metacountregressor.main_hpc import compute_standard_errors
from metacountregressor.main_hpc_lc_patch import (
    build_model_from_manual_spec, mixed_model_loglik,
    build_param_index,
)
import jax, jax.numpy as jnp
from scipy.optimize import minimize as scipy_min, differential_evolution
from scipy import stats as scipy_stats
from dataclasses import replace
from sklearn.preprocessing import StandardScaler

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# ===================================================================
# CONFIGURATION
# ===================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "near_miss_hb_ecf.csv"),
    r"Z:\test_runs_tours\demand_\paper\Unofficial_Transportation_Research_Board__TRB__LaTeX_template\near_miss_hb_ecf.csv",
    r"Z:\test_runs_tours\demand_\near_miss_hb_ecf.csv",
    os.path.join(os.path.dirname(_SCRIPT_DIR), "near_miss_hb_ecf.csv"),
    "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/paper/Unofficial_Transportation_Research_Board__TRB__LaTeX_template/near_miss_hb_ecf.csv",
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===================================================================
# HUMAN-READABLE VARIABLE NAMES
# ===================================================================
VLABEL = {
    "AADT": "Annual Avg Daily Traffic (AADT)",
    "SL": "Speed Limit (km/h)",
    "NL": "Number of Lanes",
    "SKD_SP": "Skid Resistance Present",
    "CU_ST": "Curvature: Straight",
    "CU_MOD": "Curvature: Moderate",
    "CU_SHA": "Curvature: Sharp",
    "CU_VSHA": "Curvature: Very Sharp",
    "DEL_P": "Delineation Posts Present",
    "IC": "Intersection Conflict Present",
    "IP": "Intersection Present",
    "MEDPM": "Median Present",
    "MEDSFBar": "Median Safety Barrier",
    "PA_COM1": "Parking Access (Commercial)",
    "IV_VL": "Interaction Volume: Very Low",
    "IV_L": "Interaction Volume: Low",
    "IV_M": "Interaction Volume: Medium",
    "IV_100": "Interaction Volume: 100+ veh/day",
    "PSPS1T24": "Posted Speed Compliance: 1--24\\%",
    "PSPSNIL": "Posted Speed Compliance: nil\\%",
    "PSDSG24": "Speed Differential: >24\\%",
    "PSDS1T24": "Speed Differential: 1--24\\%",
    "PSDSNIL": "Speed Differential: nil\\%",
    "RSPSDL5": "Roadside Severity: Pole $\\leq$5m",
    "RSDSDL1": "Roadside Severity: Ditch $\\leq$1m",
    "RSDSDG10": "Roadside Severity: Ditch >10m",
    "RSDSO": "Roadside Severity: Other",
    "RSPSDL1": "Roadside Severity: Pole $\\leq$1m",
    "RSDSDL5": "Roadside Severity: Ditch $\\leq$5m",
    "RSPSDL10": "Roadside Severity: Pole $\\leq$10m",
    "RSDSDL10": "Roadside Severity: Ditch $\\leq$10m",
    "RSPSDG10": "Roadside Severity: Pole >10m",
    "nm_total": "Hard-Braking: Total Count",
    "nm_braking": "Hard-Braking: Candidate Count",
    "hb_critical": "Hard-Braking: Critical-Likely",
    "hb_extreme": "Hard-Braking: Extreme",
    "hb_candidate": "Hard-Braking: Candidate",
    "ExpCrashes": "Expected Crash Frequency (ECF)",
    "ObsCrashes": "Observed Crash Count",
    "pred_nm": "Predicted Hard-Braking Count",
}
_V = lambda c: VLABEL.get(c, c)

# ===================================================================
# CHECKPOINT SYSTEM
# ===================================================================
class Checkpoint:
    """Simple JSON checkpoint dict. When a step is completed, the
    results dict is written to {step_name}.json. On re-run, if the
    checkpoint exists and is valid, it is loaded instead of re-running."""
    def __init__(self, name):
        self.path = os.path.join(CHECKPOINT_DIR, f"{name}.json")
        self.name = name

    def exists(self):
        return os.path.exists(self.path)

    def save(self, data):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        serialisable = {}
        for k, v in data.items():
            try:
                json.dumps({k: v if not isinstance(v, (np.ndarray, np.generic, pd.DataFrame)) else None})
                serialisable[k] = v
            except (TypeError, OverflowError):
                pass
        with open(self.path, "w") as f:
            json.dump({"step": self.name, "completed": True,
                        "stored_keys": list(serialisable.keys())}, f)

    def mark_done(self):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"step": self.name, "completed": True}, f)

    def is_done(self):
        if not self.exists():
            return False
        try:
            with open(self.path) as f:
                d = json.load(f)
            return d.get("completed", False)
        except Exception:
            return False

    def clear(self):
        if self.exists():
            os.remove(self.path)

    def load(self):
        with open(self.path) as f:
            return json.load(f)


# ===================================================================
# 1. DATA LOADING
# ===================================================================
def load_data():
    print("="*70)
    print("  LOADING DATA: near_miss_hb_ecf.csv")
    print("="*70)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    GEO = [c for c in [
        "AADT","SL","NL","SKD_SP","CU_MOD","CU_SHA","CU_VSHA",
        "DEL_P","IC","IP","MEDPM","MEDSFBar","PA_COM1",
        "PSPS1T24","PSPSNIL","PSDSG24","PSDS1T24","PSDSNIL",
        "IV_VL","IV_L","IV_M","IV_100",
        "RSPSDL5","RSDSDL1","RSDSDG10","RSDSO",
        "RSPSDL1","RSDSDL5","RSPSDL10","RSDSDL10","RSPSDG10",
    ] if c in df.columns and df[c].std() > 1e-10]

    df = df.dropna(subset=["ObsCrashes"])
    df = df[df["ObsCrashes"] >= 0]
    for c in GEO:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

    nm_response_cols = [c for c in ["nm_braking", "nm_total", "hb_critical", "hb_extreme"]
                        if c in df.columns]
    for c in nm_response_cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(0)

    df["seg_len_km"] = (df.get("segment_length_m", 1000)/1000).clip(lower=0.001)
    df["log_len"] = np.log(df["seg_len_km"])
    df["log_ecf"] = np.log(np.maximum(df["ExpCrashes"], 1e-6))

    print(f"  Segments: {len(df):,}  |  Geometry features: {len(GEO)}")
    print(f"  NM responses: {nm_response_cols}")
    print(f"  Has crashes: {df['ObsCrashes'].gt(0).sum():,} ({df['ObsCrashes'].gt(0).mean()*100:.1f}%)")
    print(f"  Has HB events: {df['nm_total'].gt(0).sum():,} ({df['nm_total'].gt(0).mean()*100:.1f}%)")
    for c in ['hb_critical', 'hb_extreme', 'hb_candidate']:
        if c in df.columns:
            print(f"    {c}: sum={df[c].sum():.0f}, nonzero={df[c].gt(0).sum()}")

    return df, GEO, nm_response_cols


# ===================================================================
# 2. MODEL FITTER (same as original)
# ===================================================================
def _scipy_hessian_se(obj_fn, params, eps=1e-4):
    K = len(params)
    H = np.zeros((K, K))
    f0 = obj_fn(params)
    for i in range(K):
        ei = np.zeros(K); ei[i] = eps
        fp = obj_fn(params + ei); fm = obj_fn(params - ei)
        H[i, i] = (fp - 2*f0 + fm) / eps**2
        for j in range(i + 1, K):
            ej = np.zeros(K); ej[j] = eps
            fpp = obj_fn(params + ei + ej)
            fpm = obj_fn(params + ei - ej)
            fmp = obj_fn(params - ei + ej)
            fmm = obj_fn(params - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
    try:
        cov = np.linalg.lstsq(H, np.eye(K), rcond=None)[0]
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        se = np.full(K, np.nan)
    return se

def fit_model(df, features, y_col, offset_col, dispersion=1, label="", verbose=False):
    df_m = df.dropna(subset=features + [y_col, offset_col]).reset_index(drop=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_m[features].values.astype(float))
    for i in range(len(features)):
        df_m[f"__f{i}__"] = Xs[:, i]
    fstd = [f"__f{i}__" for i in range(len(features))]

    y_raw = df_m[y_col].values.astype(float)
    df_m["_id"] = np.arange(len(df_m))
    df_m["_off"] = df_m[offset_col].values.astype(float)

    manual_spec = {
        "fixed_terms": fstd, "rdm_terms": [], "rdm_cor_terms": [],
        "grouped_terms": [], "hetro_in_means": [],
        "transformations": ["no"]*len(fstd),
        "dispersion": dispersion, "latent_classes": 1,
    }

    data, spec = build_model_from_manual_spec(
        df_m, manual_spec, id_col="_id", y_col=y_col, offset_col="_off"
    )
    spec = replace(spec, model="nb" if dispersion else "poisson")
    pi = build_param_index(spec); K = pi["total_params"]

    init = np.zeros(K)
    init[0] = np.log(max(y_raw.mean(), 0.01))
    if spec.model == "nb" and K > len(features) + 1:
        init[-1] = 0.1

    def obj(p):
        return float(mixed_model_loglik(jnp.array(p), data, spec, indivi=False))

    res = scipy_min(obj, init, method="L-BFGS-B", options={"maxiter": 5000})
    params = res.x; n = len(df_m); k = K
    ll = -obj(params); aic = 2*k - 2*ll; bic = k*np.log(n) - 2*ll

    se = None
    try:
        se_jax = np.asarray(compute_standard_errors(jnp.array(params), obj))
        if np.isfinite(se_jax).any() and np.nanmax(np.abs(se_jax)) < 50:
            se = se_jax
    except Exception:
        pass
    if se is None:
        se = _scipy_hessian_se(obj, params)
    se = np.where(np.abs(se) > 50, np.nan, se)

    z = np.where(np.isfinite(se) & (se > 1e-12), params/se, np.nan)
    p = np.where(np.isfinite(z), 2*(1-scipy_stats.norm.cdf(np.abs(z))), np.nan)

    names = ["(Intercept)"] + [_V(f) for f in features]
    if spec.model == "nb" and K > len(names):
        names.append("ln(overdispersion)")

    coef_tbl = pd.DataFrame({
        "Variable": names[:K], "Estimate": params[:K], "Std.Err": se[:K],
        "z": z[:K], "p": p[:K], "IRR": np.exp(params[:K]),
    })

    return {"label": label, "ll": ll, "aic": aic, "bic": bic, "k": k, "n": n,
            "converged": res.success, "coef_table": coef_tbl, "params": params,
            "se": se, "features": features, "y_col": y_col, "scaler": scaler,
            "alpha": float(np.exp(params[-1])) if spec.model == "nb" and K > len(features)+1 else np.nan}


# ===================================================================
# 3. DIFFERENTIAL EVOLUTION FEATURE SEARCH (from original, with resume)
# ===================================================================
def differential_evolution_feature_search(
    df, candidates, y_col, offset_col, maxiter=60, popsize=15,
    label="DE", plot_path=None, p_threshold=0.10,
    penalty_per_insig=10.0, seed=42,
):
    print(f"\n  DIFFERENTIAL EVOLUTION [{label}] — y = {y_col}")
    print(f"  {len(candidates)} candidates, maxiter={maxiter}, popsize={popsize}")
    print(f"  p-threshold={p_threshold}, penalty={penalty_per_insig}/insig")

    dfs = df.dropna(subset=candidates + [y_col, offset_col]).reset_index(drop=True)
    af, Kf = list(candidates), len(candidates)

    def _score(x):
        mask = (np.asarray(x) > 0.5).astype(int)
        if mask.sum() < 2:
            return 1e12
        sel = [af[i] for i in range(Kf) if mask[i] == 1]
        o = fit_model(dfs, sel, y_col, offset_col, label="DE", verbose=False)
        if o is None:
            return 1e12
        coefs = o["coef_table"]
        feature_rows = coefs.iloc[1:]
        if coefs["Variable"].iloc[-1].startswith("ln("):
            feature_rows = feature_rows.iloc[:-1]
        n_insig = sum(1 for p_val in feature_rows["p"]
                      if np.isnan(p_val) or p_val > p_threshold)
        return o["bic"] + n_insig * penalty_per_insig

    history = []
    def _callback(xk, convergence=None):
        history.append(_score(xk))

    print(f"  Launching scipy.differential_evolution (seed={seed}) ...")
    res = differential_evolution(
        _score, bounds=[(0.0, 1.0)] * Kf,
        maxiter=maxiter, popsize=popsize, seed=seed,
        mutation=(0.4, 1.4), recombination=0.7,
        tol=0.0, polish=False, callback=_callback,
    )
    history.append(_score(res.x))
    best_so_far = np.minimum.accumulate(history)

    best_mask = (np.asarray(res.x) > 0.5).astype(int)
    bf = [af[i] for i in range(Kf) if best_mask[i] == 1]
    if len(bf) < 2:
        bf = list(np.random.choice(af, max(2, min(8, len(af))), replace=False))

    final = fit_model(dfs, bf, y_col, offset_col, label=f"DE-{label}", verbose=False)

    coefs_f = final["coef_table"]
    feature_rows_f = coefs_f.iloc[1:]
    if coefs_f["Variable"].iloc[-1].startswith("ln("):
        feature_rows_f = feature_rows_f.iloc[:-1]
    n_insig_f = sum(1 for p_val in feature_rows_f["p"]
                    if np.isnan(p_val) or p_val > p_threshold)
    penalized_bic_f = final["bic"] + n_insig_f * penalty_per_insig

    print(f"\n  BEST [{label}]: raw_BIC={final['bic']:.2f}  "
          f"penalised_BIC={penalized_bic_f:.2f}  "
          f"insig={n_insig_f}/{len(bf)}  |  {len(bf)} features:")
    for f in bf:
        pv = feature_rows_f[coefs_f["Variable"] == _V(f)]["p"].values
        p_str = f"p={pv[0]:.4f}" if len(pv) and np.isfinite(pv[0]) else "p=nan"
        sig = "***" if (len(pv) and (not np.isnan(pv[0])) and pv[0] < p_threshold) else ""
        print(f"    {sig:3s} {_V(f):<50s} {p_str}")

    if plot_path:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(best_so_far) + 1), best_so_far, "darkorange", linewidth=1.2, alpha=0.8)
        ax.axhline(y=min(best_so_far), color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Penalised BIC (best so far)")
        ax.set_title(f"Differential Evolution Convergence — {label}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    return final, bf, best_so_far.tolist()


# ===================================================================
# 4. MODEL TO LATEX
# ===================================================================
def model_to_latex(model, filename, caption, label):
    tbl = model["coef_table"].copy()
    latex_rows = []
    for _, row in tbl.iterrows():
        var = row["Variable"].replace("_", "\\_").replace("\\%", "%").replace("%", "\\%")
        est = f"{row['Estimate']:+.4f}"
        irr = f"{row['IRR']:.4f}"
        zs = f"{row['z']:.2f}" if np.isfinite(row['z']) else "--"
        ps = f"{row['p']:.4f}" if np.isfinite(row['p']) else "--"
        sig = "***" if not np.isnan(row['p']) and row['p'] < 0.01 else \
              "**" if not np.isnan(row['p']) and row['p'] < 0.05 else \
              "*" if not np.isnan(row['p']) and row['p'] < 0.10 else ""
        latex_rows.append(f"    {var} & {est} & {irr} & {zs} & {ps} & {sig} \\\\")

    tex = f"""% Auto-generated by HB proxy variable analysis
% Model: {model['label']}
% LL = {model['ll']:.2f}, AIC = {model['aic']:.2f}, BIC = {model['bic']:.2f}
% k = {model['k']}, n = {model['n']:,}
\\begin{{table}}[!ht]
  \\caption{{{caption}}}\\label{{{label}}}
  \\begin{{center}}
      \\begin{{tabular}}{{lrrrrc}}
        \\toprule
        Variable & Estimate & IRR & z & p & Sig \\\\
        \\midrule
"""
    tex += "\n".join(latex_rows)
    tex += f"""
        \\bottomrule
      \\end{{tabular}}
  \\end{{center}}
  \\footnotesize{{Note: $N$ = {model['n']:,} segments. Negative Binomial (NB) regression.
  All predictors standardized. IRR = Incidence Rate Ratio.}}
\\end{{table}}
"""
    with open(filename, "w") as f:
        f.write(tex)
    print(f"  LaTeX table saved: {filename}")


# ===================================================================
# 5. COEFFICIENT PROPORTIONALITY
# ===================================================================
def coefficient_proportionality_test(model_hb, model_crash, out_dir):
    print("\n" + "="*70)
    print("  COEFFICIENT PROPORTIONALITY TEST")
    print("="*70)

    feats_hb = model_hb["features"]
    feats_cr = model_crash["features"]
    common = sorted(set(feats_hb) & set(feats_cr))

    tbl_hb = model_hb["coef_table"]; tbl_cr = model_crash["coef_table"]
    beta_hb = {}; beta_cr = {}
    for f in common:
        vn = _V(f)
        rn = tbl_hb[tbl_hb["Variable"] == vn]
        rc = tbl_cr[tbl_cr["Variable"] == vn]
        if len(rn) and len(rc):
            beta_hb[f] = rn["Estimate"].values[0]
            beta_cr[f] = rc["Estimate"].values[0]

    common_f = list(beta_hb.keys())
    b_nm = np.array([beta_hb[f] for f in common_f])
    b_cr = np.array([beta_cr[f] for f in common_f])

    k_hat = np.sum(b_nm * b_cr) / np.sum(b_nm**2) if np.sum(b_nm**2) > 0 else 0
    residuals = b_cr - k_hat * b_nm
    rss = np.sum(residuals**2)
    corr = np.corrcoef(b_nm, b_cr)[0, 1] if len(b_nm) > 1 else 0
    tss = np.sum(b_cr**2)
    R2_prop = 1 - rss / tss if tss > 0 else 0

    print(f"  Proportionality factor k = {k_hat:.4f}")
    print(f"  Correlation r = {corr:.4f}")
    print(f"  Proportionality R^2 = {R2_prop:.4f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(b_nm, b_cr, alpha=0.6, s=40)
    x_line = np.linspace(b_nm.min(), b_nm.max(), 100)
    ax.plot(x_line, k_hat * x_line, "r--", alpha=0.5, label=f"k={k_hat:.3f}")
    ax.plot(x_line, x_line, "gray", alpha=0.3, linestyle=":", label="1:1 line")
    ax.set_xlabel("beta (Hard-Braking model)")
    ax.set_ylabel("beta (Crash model)")
    ax.set_title(f"Coefficient Proportionality\nr = {corr:.3f}, R^2 = {R2_prop:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "hb_coefficient_proportionality.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")

    return {"k": k_hat, "corr": corr, "R2_prop": R2_prop}


# ===================================================================
# 6. SPECIFICATION TRANSFER
# ===================================================================
def specification_transfer(df, hb_features, crash_features, out_dir):
    print("\n" + "="*70)
    print("  SPECIFICATION TRANSFER TEST")
    print("="*70)

    results = {}
    for label, feats, y in [
        ("HB-spec -> HB (native)", hb_features, "nm_total"),
        ("HB-spec -> Crashes", hb_features, "ObsCrashes"),
        ("Crash-spec -> HB", crash_features, "nm_total"),
        ("Crash-spec -> Crashes (native)", crash_features, "ObsCrashes"),
    ]:
        r = fit_model(df, feats, y, "log_len", label=label, verbose=False)
        results[label] = r

    print(f"  {'Model':<40s} {'BIC':>10s} {'AIC':>10s} {'LL':>10s} {'k':>4s}")
    print(f"  {'-'*77}")
    for k in results:
        m = results[k]
        print(f"  {m['label']:<40s} {m['bic']:10.2f} {m['aic']:10.2f} {m['ll']:10.2f} {m['k']:4d}")

    nm_native = results["HB-spec -> HB (native)"]["bic"]
    cr_native = results["Crash-spec -> Crashes (native)"]["bic"]
    nm_to_cr = results["HB-spec -> Crashes"]["bic"]
    cr_to_nm = results["Crash-spec -> HB"]["bic"]

    penalty_nm = cr_to_nm - nm_native
    penalty_cr = nm_to_cr - cr_native

    print(f"\n  Transfer penalty (Crash-spec on HB): {penalty_nm:+.2f} BIC")
    print(f"  Transfer penalty (HB-spec on Crashes): {penalty_cr:+.2f} BIC")
    print(f"  Symmetric penalty: {(penalty_nm+penalty_cr)/2:+.2f} BIC")
    return results


# ===================================================================
# 7. ENCOMPASSING TEST
# ===================================================================
def encompassing_test(df, hb_feats, crash_feats, out_dir):
    print("\n" + "="*70)
    print("  ENCOMPASSING TEST")
    print("="*70)

    hb_model = fit_model(df, hb_feats, "nm_total", "log_len", label="Step1: HB ~ Geo", verbose=False)
    print(f"  HB model: LL={hb_model['ll']:.2f}, BIC={hb_model['bic']:.2f}")

    all_feats = sorted(set(hb_feats) | set(crash_feats))
    df_e = df.dropna(subset=all_feats + ["nm_total", "ObsCrashes", "log_len"]).copy()
    params_nm = hb_model["params"]
    n_geo = len(hb_feats)

    scaler_en = StandardScaler()
    X_r = scaler_en.fit_transform(df_e[hb_feats].values.astype(float))
    intercept_nm = params_nm[0]
    beta_nm = params_nm[1:n_geo+1]
    eta_nm = intercept_nm + X_r @ beta_nm + df_e["log_len"].values
    df_e["pred_nm"] = np.exp(np.clip(eta_nm, -20, 20))
    df_e["pred_nm"] = np.nan_to_num(df_e["pred_nm"], nan=0.0, posinf=1e6, neginf=0.0)
    df_e["pred_nm"] = df_e["pred_nm"].fillna(0)

    crash_geo = fit_model(df_e, crash_feats, "ObsCrashes", "log_len",
                          label="Restricted: Crashes ~ Geo", verbose=False)
    crash_nm = fit_model(df_e, crash_feats + ["pred_nm"], "ObsCrashes", "log_len",
                         label="Unrestricted: Crashes ~ Geo + predHB", verbose=False)

    ll_r = crash_geo["ll"]; ll_u = crash_nm["ll"]
    k_r = crash_geo["k"]; k_u = crash_nm["k"]
    lr_stat = 2 * (ll_u - ll_r); df_lr = k_u - k_r
    p_lr = 1 - scipy_stats.chi2.cdf(lr_stat, df_lr)

    print(f"  Restricted (Geo only): LL={ll_r:.2f}, BIC={crash_geo['bic']:.2f}")
    print(f"  Unrestricted (+predHB): LL={ll_u:.2f}, BIC={crash_nm['bic']:.2f}")
    print(f"  LR stat = {lr_stat:.2f}  |  df = {df_lr}  |  p = {p_lr:.6f}")
    print(f"  delta BIC = {crash_nm['bic'] - crash_geo['bic']:+.2f}")

    return {"lr_stat": lr_stat, "df": df_lr, "p": p_lr,
            "bic_r": crash_geo["bic"], "bic_u": crash_nm["bic"]}


# ===================================================================
# 8. BIVARIATE LATENT RISK
# ===================================================================
def bivariate_latent_risk(df, geo_feats, nm_col, out_dir):
    print("\n" + "="*70)
    print("  BIVARIATE LATENT RISK ANALYSIS")
    print("="*70)

    df_b = df.dropna(subset=geo_feats + [nm_col, "ObsCrashes", "log_len"]).copy()
    y_nm = df_b[nm_col].values / df_b["seg_len_km"].values
    y_cr = df_b["ObsCrashes"].values / df_b["seg_len_km"].values

    from scipy.stats import spearmanr, pearsonr
    rho_s, p_s = spearmanr(y_nm, y_cr)
    r_p, p_p = pearsonr(np.log1p(y_nm), np.log1p(y_cr))

    from sklearn.linear_model import LinearRegression
    X_g = df_b[geo_feats].values
    resid_nm = np.log1p(y_nm) - LinearRegression().fit(X_g, np.log1p(y_nm)).predict(X_g)
    resid_cr = np.log1p(y_cr) - LinearRegression().fit(X_g, np.log1p(y_cr)).predict(X_g)
    r_partial, p_partial = pearsonr(resid_nm, resid_cr)

    print(f"  Raw Spearman rho = {rho_s:.4f} (p = {p_s:.2e})")
    print(f"  Partial (after geo) r = {r_partial:.4f} (p = {p_partial:.2e})")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(resid_nm, resid_cr, gridsize=50, cmap="Blues", mincnt=1)
    ax.set_xlabel("HB residual (log1p per-km)")
    ax.set_ylabel("Crash residual (log1p per-km)")
    ax.set_title(f"Residual Correlation After Geometry\nPartial r = {r_partial:.4f}")
    plt.colorbar(ax.collections[0], ax=ax, label="Count")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "hb_bivariate_residuals.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {"spearman_r": rho_s, "partial_r": r_partial, "partial_p": p_partial}


# ===================================================================
# 9. BIVARIATE COPULA FUNCTIONS
# ===================================================================
def _nb_pmf(y, mu, alpha):
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return scipy_stats.nbinom.pmf(y, r, p)

def _nb_cdf(y, mu, alpha):
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return scipy_stats.nbinom.cdf(y, r, p)

def _frank_cop(u, v, theta):
    eps = 1e-300
    denom = np.expm1(-theta)
    if abs(denom) < eps:
        return u * v
    return -np.log1p(np.expm1(-theta * u) * np.expm1(-theta * v) / denom) / theta

def _normal_cop(u, v, rho):
    if abs(rho) < 1e-8:
        return u * v
    from scipy.stats import norm, multivariate_normal
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    pts = np.column_stack([norm.ppf(u), norm.ppf(v)])
    return multivariate_normal.cdf(pts, mean=[0, 0], cov=[[1, rho], [rho, 1]])

def _clayton_cop(u, v, theta):
    if theta < 1e-6:
        return u * v
    return np.maximum(u ** (-theta) + v ** (-theta) - 1.0, 1e-300) ** (-1.0 / theta)

_COPULA_FNS    = {"frank": _frank_cop, "normal": _normal_cop, "kimeldorf": _clayton_cop}
_COPULA_BOUNDS = {"frank": (0.01, 30.0), "normal": (-0.99, 0.99), "kimeldorf": (0.001, 20.0)}
_COPULA_INIT   = {"frank": 1.0,          "normal": 0.3,           "kimeldorf": 0.5}

def _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, theta):
    p = (cop_fn(u1c, u2c, theta) - cop_fn(u1c, u2p, theta)
       - cop_fn(u1p, u2c, theta) + cop_fn(u1p, u2p, theta))
    return np.maximum(p, 1e-300)

def _rebuild_mu(df_sub, model):
    feats  = model["features"]
    params = model["params"]
    X_std  = model["scaler"].transform(df_sub[feats].values.astype(float))
    eta    = params[0] + X_std @ params[1:len(feats) + 1] + df_sub["log_len"].values
    mu     = np.exp(np.clip(eta, -20, 20))
    alpha  = float(np.exp(params[-1]))
    return mu, alpha


def fit_bivariate_copula(df, nm_col, nm_model, crash_model, copula_name="kimeldorf"):
    all_feats = list(set(nm_model["features"]) | set(crash_model["features"]))
    df_c = df.dropna(subset=all_feats + [nm_col, "ObsCrashes", "log_len"]).reset_index(drop=True)
    n = len(df_c)

    mu1, alpha1 = _rebuild_mu(df_c, nm_model)
    mu2, alpha2 = _rebuild_mu(df_c, crash_model)
    y1 = df_c[nm_col].values.astype(int)
    y2 = df_c["ObsCrashes"].values.astype(int)

    eps = 1e-10
    u1c = np.clip(_nb_cdf(y1, mu1, alpha1), eps, 1 - eps)
    u1p = np.where(y1 > 0, np.clip(_nb_cdf(y1 - 1, mu1, alpha1), 0.0, 1 - eps), 0.0)
    u2c = np.clip(_nb_cdf(y2, mu2, alpha2), eps, 1 - eps)
    u2p = np.where(y2 > 0, np.clip(_nb_cdf(y2 - 1, mu2, alpha2), 0.0, 1 - eps), 0.0)

    indep_pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, lambda u, v, t: u * v, 0.0)
    ll_indep = float(np.sum(np.log(np.maximum(indep_pmf, eps))))

    cop_fn = _COPULA_FNS[copula_name]
    lo, hi = _COPULA_BOUNDS[copula_name]
    t0     = _COPULA_INIT[copula_name]

    def neg_ll(t):
        pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, float(t[0]))
        return -float(np.sum(np.log(pmf)))

    res = scipy_min(neg_ll, [t0], method="L-BFGS-B",
                    bounds=[(lo, hi)], options={"maxiter": 500, "ftol": 1e-9})
    theta_opt = float(res.x[0])
    ll_joint  = float(np.sum(np.log(_joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, theta_opt))))

    k_indep  = nm_model["k"] + crash_model["k"]
    k_copula = k_indep + 1
    bic_indep  = k_indep  * np.log(n) - 2 * ll_indep
    bic_copula = k_copula * np.log(n) - 2 * ll_joint

    lr_stat = 2.0 * (ll_joint - ll_indep)
    p_lr    = float(1 - scipy_stats.chi2.cdf(lr_stat, df=1))

    return {
        "copula": copula_name, "theta": theta_opt, "n": n,
        "ll_joint": ll_joint, "ll_indep": ll_indep,
        "bic_joint": bic_copula, "bic_indep": bic_indep,
        "k_copula": k_copula, "k_indep": k_indep,
        "lr_stat": lr_stat, "p_lr": p_lr,
        "df_c": df_c, "y1": y1, "y2": y2,
        "mu1": mu1, "alpha1": alpha1, "mu2": mu2, "alpha2": alpha2,
        "u1c": u1c, "u1p": u1p, "u2c": u2c, "u2p": u2p,
    }


def conditional_crash_prediction(fit_r, max_y2=60):
    cop_fn   = _COPULA_FNS[fit_r["copula"]]
    theta    = fit_r["theta"]
    mu1, alpha1 = fit_r["mu1"], fit_r["alpha1"]
    mu2, alpha2 = fit_r["mu2"], fit_r["alpha2"]
    u1c, u1p = fit_r["u1c"], fit_r["u1p"]
    n        = len(fit_r["y1"])
    eps      = 1e-10

    p_y1 = np.maximum(_nb_pmf(fit_r["y1"], mu1, alpha1), eps)
    E_y2 = np.zeros(n)

    for k in range(max_y2 + 1):
        k_arr  = np.full(n, k)
        u2c_k  = np.clip(_nb_cdf(k_arr, mu2, alpha2), eps, 1 - eps)
        u2p_k  = np.where(k > 0, np.clip(_nb_cdf(k_arr - 1, mu2, alpha2), 0.0, 1 - eps), 0.0)
        joint  = _joint_pmf_vec(u1c, u1p, u2c_k, u2p_k, cop_fn, theta)
        E_y2  += k * np.maximum(joint / p_y1, 0.0)
    return E_y2


def bivariate_copula_analysis(df, nm_col, nm_model, crash_model, out_dir):
    print("\n" + "=" * 70)
    print("  BIVARIATE COPULA ANALYSIS — Frank / Normal / Kimeldorf")
    print("=" * 70)

    fits = {}
    for cn in ("frank", "normal", "kimeldorf"):
        print(f"    {cn:12s} ...", end=" ", flush=True)
        r = fit_bivariate_copula(df, nm_col, nm_model, crash_model, copula_name=cn)
        fits[cn] = r
        print(f"theta={r['theta']:.4f}  LL={r['ll_joint']:.1f}  BIC={r['bic_joint']:.1f}  LR={r['lr_stat']:.1f}  p={r['p_lr']:.2e}")

    r0 = fits["frank"]
    ll_indep = r0["ll_indep"]
    k_indep = r0["k_indep"]
    n = r0["n"]
    bic_indep = k_indep * np.log(n) - 2 * ll_indep
    best_cn = min(fits, key=lambda k: fits[k]["bic_joint"])

    rows = [{"copula": "independence", "theta": np.nan, "n": n, "k": k_indep,
             "ll": ll_indep, "bic": bic_indep, "delta_bic": 0.0,
             "lr_stat": np.nan, "p_lr": np.nan}]
    for cn in ("frank", "normal", "kimeldorf"):
        r = fits[cn]
        rows.append({"copula": cn, "theta": r["theta"], "n": r["n"], "k": r["k_copula"],
                     "ll": r["ll_joint"], "bic": r["bic_joint"],
                     "delta_bic": r["bic_joint"] - bic_indep,
                     "lr_stat": r["lr_stat"], "p_lr": r["p_lr"]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "hb_copula_comparison.csv"), index=False)

    # Conditional predictions from best copula
    best_r = fits[best_cn]
    print(f"\n  Best copula: {best_cn} (theta={best_r['theta']:.4f})")
    E_cond = conditional_crash_prediction(best_r, max_y2=60)
    E_marg = best_r["mu2"]
    y2_obs = best_r["y2"]

    r_marg,  _  = scipy_stats.pearsonr(E_marg, y2_obs)
    r_cond,  _  = scipy_stats.pearsonr(E_cond, y2_obs)
    mae_marg = float(np.mean(np.abs(E_marg - y2_obs)))
    mae_cond = float(np.mean(np.abs(E_cond - y2_obs)))
    rmse_marg = float(np.sqrt(np.mean((E_marg - y2_obs) ** 2)))
    rmse_cond = float(np.sqrt(np.mean((E_cond - y2_obs) ** 2)))

    print(f"  CRASH PREDICTION: Marginal r={r_marg:.4f}  Copula r={r_cond:.4f}  delta={r_cond-r_marg:+.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    ax.hexbin(E_marg, y2_obs, gridsize=40, cmap="Blues", mincnt=1)
    lim = max(float(np.percentile(E_marg, 99)), float(np.percentile(y2_obs, 99)))
    ax.plot([0, lim], [0, lim], "r--", alpha=0.6)
    ax.set_xlabel("Marginal E[crashes | X]"); ax.set_ylabel("Observed crashes")
    ax.set_title(f"Marginal NB\nr = {r_marg:.3f}, MAE = {mae_marg:.3f}")

    ax = axes[1]
    ax.hexbin(E_cond, y2_obs, gridsize=40, cmap="Greens", mincnt=1)
    ax.plot([0, lim], [0, lim], "r--", alpha=0.6)
    ax.set_xlabel(f"Copula-conditional E[crashes | HB, X]\n({best_cn}, theta = {best_r['theta']:.3f})")
    ax.set_ylabel("Observed crashes")
    ax.set_title(f"Copula-Conditional ({best_cn})\nr = {r_cond:.3f}, MAE = {mae_cond:.3f}")

    ax = axes[2]
    improvement = np.abs(E_marg - y2_obs) - np.abs(E_cond - y2_obs)
    ax.hist(improvement, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("|err_marginal| - |err_copula|  (positive = copula better)")
    ax.set_ylabel("Segment count")
    ax.set_title("Per-Segment Prediction Improvement")

    fig.suptitle(f"Bivariate Copula Crash Prediction — {best_cn}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hb_copula_prediction_comparison.png"), dpi=150)
    plt.close(fig)

    return {
        "fits": fits, "best_copula": best_cn, "bic_indep": bic_indep,
        "best_lr_stat": best_r["lr_stat"], "best_lr_p": best_r["p_lr"],
        "best_theta": best_r["theta"],
        "pred": {"r_marg": r_marg, "r_cond": r_cond, "mae_marg": mae_marg,
                 "mae_cond": mae_cond, "rmse_marg": rmse_marg, "rmse_cond": rmse_cond},
    }


# ===================================================================
# 10. MCR STRUCTURE SEARCH (from original)
# ===================================================================
def mcr_model_search(df, y_col, feats_subset, offset_col,
                     label="MCR-DE", n_starts=2, max_iter=500,
                     R=5, max_latent_classes=1, out_dir=None,
                     N_max=1500, max_time_per_start=1800,
                     solver="de", pop_size=30):
    try:
        from metacountregressor.Solvers_METAJAX import AdaptiveDE, NSGA2Engine, AdvancedSimulatedAnnealing
    except ImportError:
        print(f"  mcr_model_search: Cannot import Solvers_METAJAX; skipping.")
        return None
    try:
        from metacountregressor.experiment_package import StructureEvaluatorLC
    except ImportError:
        print(f"  mcr_model_search: Cannot import StructureEvaluatorLC; skipping.")
        return None

    def _fixed_generate_neighbor(self, solution, T=None, max_attempts=20, min_active=2):
        for _ in range(max_attempts):
            neighbor = solution.copy()
            if T is not None and self.T0 is not None:
                mut_rate = self.mutation_rate * (T / self.T0)
            else:
                mut_rate = self.mutation_rate
            mut_rate = float(np.clip(mut_rate, 0.0, 1.0))
            n_changes = np.random.randint(self.min_changes, self.max_changes + 1)
            indices = list(np.random.choice(self.dim, size=n_changes, replace=False))
            D = self.dim_core
            if getattr(self.evaluator, "max_latent_classes", 1) > 1:
                mask_start = 2 * D + 2; mask_end = 3 * D + 1
                mask_indices = list(range(mask_start, min(mask_end + 1, self.dim)))
                if mask_indices and not any(i in indices for i in mask_indices):
                    indices[np.random.randint(len(indices))] = int(np.random.choice(mask_indices))
            indices = np.asarray(indices, dtype=int)
            changed = False
            for idx in indices:
                if np.random.rand() < mut_rate:
                    if idx < D:
                        neighbor[idx] = self.sample_allowed_role(idx)
                        changed = True
                    elif idx < 2 * D:
                        neighbor[idx] = np.random.randint(0, 6)
                        changed = True
            active = np.sum(neighbor[:D] != 0)
            if active < min_active:
                zeros = np.where(neighbor[:D] == 0)[0]
                if len(zeros):
                    for j in np.random.choice(zeros, size=min(min_active - active, len(zeros)), replace=False):
                        neighbor[j] = self.sample_allowed_role(j, force_active=True)
            neighbor = self.repair(neighbor)
            if changed and not self.is_same(neighbor, solution):
                return neighbor
        return solution

    AdvancedSimulatedAnnealing.generate_neighbor = _fixed_generate_neighbor

    print(f"\n  MCR STRUCTURE SEARCH [{label}] — y = {y_col}")
    print(f"  {len(feats_subset)} features, n_starts={n_starts}, max_iter={max_iter}, N_max={N_max}")

    df_m = df.dropna(subset=feats_subset + [y_col, offset_col]).reset_index(drop=True).copy()
    if len(df_m) > N_max:
        df_m = df_m.sample(n=N_max, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {N_max} rows")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_m[feats_subset].values.astype(float))
    std_cols = [f"__x{i}__" for i in range(len(feats_subset))]
    for i, c in enumerate(std_cols):
        df_m[c] = Xs[:, i]
    df_m["_id"] = np.arange(len(df_m))

    allowed_roles = {c: [0, 1, 2, 3] for c in std_cols}
    allowed_dists = {c: ["normal"] for c in std_cols}

    try:
        evaluator = StructureEvaluatorLC(
            df_m, "_id", y_col, std_cols,
            allowed_roles, allowed_dists,
            mode="single", offset_col=offset_col,
            R=R, max_latent_classes=max_latent_classes,
        )
    except Exception as e:
        print(f"  StructureEvaluatorLC init failed: {e}")
        return None

    D = len(std_cols)
    dim = 3 * D + 2 if max_latent_classes > 1 else 2 * D + 2
    best_sol = None; best_bic = 1e12

    for start in range(n_starts):
        try:
            import gc; jax.clear_caches(); gc.collect()
        except Exception:
            pass

        np.random.seed(42 + start * 13)
        try:
            if solver == "de":
                operator = AdaptiveDE(F=0.5, CR=0.7)
                ng = NSGA2Engine(
                    evaluator=evaluator, operator=operator,
                    dimension=dim, pop_size=pop_size,
                    max_iter=max_iter, n_jobs=1, save_history=True,
                )
                archive, scores = ng.optimize()
                archive = np.atleast_2d(archive)
                scores  = np.atleast_1d(scores)
            else:
                sa = AdvancedSimulatedAnnealing(
                    evaluator=evaluator, dimension=dim,
                    max_iter=max_iter, max_time=max_time_per_start,
                    alpha=0.97, mutation_rate=0.35,
                    min_changes=1, max_changes=3,
                )
                archive, scores = sa.optimize()
        except Exception as e:
            print(f"  [Start {start+1}] solver failed: {e}")
            continue

        if len(scores) > 0:
            idx = int(np.argmin(scores))
            if float(scores[idx]) < best_bic:
                best_bic = float(scores[idx])
                best_sol = archive[idx].copy()
        print(f"  [Start {start+1}/{n_starts}] best BIC = {best_bic:.2f}")

    if best_sol is None:
        print(f"  MCR search produced no valid solution for [{label}]")
        return None

    best_spec = evaluator.build_spec(best_sol)
    if best_spec is None:
        print(f"  MCR: build_spec returned None [{label}]")
        return None

    col_map = {f"__x{i}__": feats_subset[i] for i in range(len(feats_subset))}

    def remap_list(lst):
        out = []
        for item in lst:
            s = str(item)
            if ":" in s:
                head, tail = s.split(":", 1)
                out.append(f"{col_map.get(head, head)}:{tail}")
            else:
                out.append(col_map.get(s, s))
        return out

    fixed_orig = [col_map.get(f, f) for f in best_spec.get("fixed_terms", [])]
    rdm_i_orig = remap_list(best_spec.get("rdm_terms", []))
    rdm_c_orig = remap_list(best_spec.get("rdm_cor_terms", []))
    lc = best_spec.get("latent_classes", 1)
    nb = bool(best_spec.get("dispersion", 1))

    print(f"\n  BEST MCR [{label}]: BIC = {best_bic:.2f}")
    print(f"  Fixed params:     {[_V(f) for f in fixed_orig]}")
    print(f"  Rnd-Independent:  {[_V(r.split(':')[0]) for r in rdm_i_orig]}")
    print(f"  Rnd-Correlated:   {[_V(r.split(':')[0]) for r in rdm_c_orig]}")

    if out_dir:
        pd.DataFrame([{
            "label": label, "bic": best_bic,
            "fixed_terms": ";".join(fixed_orig),
            "rdm_ind_terms": ";".join(rdm_i_orig),
            "rdm_cor_terms": ";".join(rdm_c_orig),
            "latent_classes": lc, "nb": int(nb),
        }]).to_csv(os.path.join(out_dir, f"hb_mcr_search_{label}.csv"), index=False)

    return {
        "spec": best_spec, "bic": best_bic,
        "fixed_orig": fixed_orig, "rdm_i_orig": rdm_i_orig,
        "rdm_c_orig": rdm_c_orig, "lc": lc, "nb": nb,
    }


# ===================================================================
# 11. HB CATEGORY MULTIVARIATE ANALYSIS (NEW)
# ===================================================================
def hb_category_multivariate_analysis(df, geo_feats, out_dir):
    """Jointly model HB categories (candidate, critical, extreme) with crashes
    using a multivariate copula framework. Produces category-level insights."""
    print("\n" + "="*70)
    print("  HB CATEGORY MULTIVARIATE ANALYSIS")
    print("="*70)

    hb_cats = [c for c in ['hb_candidate', 'hb_critical', 'hb_extreme'] if c in df.columns]
    print(f"  HB categories available: {hb_cats}")

    # Fit individual models per HB category
    cat_models = {}
    for cat in hb_cats:
        print(f"\n  Fitting {cat} ~ Geo ...")
        m = fit_model(df, geo_feats, cat, "log_len", label=f"Cat: {cat}", verbose=False)
        if m is not None:
            cat_models[cat] = m
            print(f"    LL={m['ll']:.2f}, BIC={m['bic']:.2f}, k={m['k']}")

    # Bivariate copula between each HB category and crashes
    crash_model = fit_model(df, geo_feats, "ObsCrashes", "log_len", label="Crash (multivar)", verbose=False)
    print(f"\n  Crash model: LL={crash_model['ll']:.2f}, BIC={crash_model['bic']:.2f}")

    cat_copula_results = {}
    for cat, cat_m in cat_models.items():
        print(f"\n  Bivariate copula: {cat} ~ Crashes (Kimeldorf)...")
        r = fit_bivariate_copula(df, cat, cat_m, crash_model, copula_name="kimeldorf")
        cat_copula_results[cat] = r
        print(f"    theta={r['theta']:.4f}  LR={r['lr_stat']:.1f}  p={r['p_lr']:.2e}")

    # Save comparison
    rows = []
    for cat, r in cat_copula_results.items():
        rows.append({"hb_category": cat, "theta": r["theta"],
                     "ll_joint": r["ll_joint"], "ll_indep": r["ll_indep"],
                     "bic_joint": r["bic_joint"], "bic_indep": r["bic_indep"],
                     "lr_stat": r["lr_stat"], "p_lr": r["p_lr"]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "hb_category_copula.csv"), index=False)
    print(f"  Saved: hb_category_copula.csv")

    return cat_models, cat_copula_results


# ===================================================================
# MAIN
# ===================================================================
def _fit_intercept_nb(df, y_col):
    """Estimate a constant-only NB model on df; return its total log-likelihood."""
    d = df.dropna(subset=[y_col, "log_len"]).reset_index(drop=True)
    y = d[y_col].values.astype(float)
    d["_id"] = np.arange(len(d))
    d["_off"] = d["log_len"].values
    manual_spec = {
        "fixed_terms": [], "rdm_terms": [], "rdm_cor_terms": [],
        "grouped_terms": [], "hetro_in_means": [], "transformations": [],
        "dispersion": 1, "latent_classes": 1,
    }
    data, spec = build_model_from_manual_spec(
        d, manual_spec, id_col="_id", y_col=y_col, offset_col="_off")
    spec = replace(spec, model="nb")
    pi = build_param_index(spec)
    init = np.zeros(pi["total_params"])
    init[0] = np.log(max(y.mean(), 0.01))
    init[-1] = 0.1
    def obj(p):
        return float(mixed_model_loglik(jnp.array(p), data, spec, indivi=False))
    res = scipy_min(obj, init, method="L-BFGS-B", options={"maxiter": 3000})
    return -float(obj(res.x)), pi["total_params"]


def _apply_model_ll(df_sub, model):
    """Log-likelihood of df_sub under a fitted source model (transfer evaluation).

    Standardizes features with the source model's scaler and evaluates the NB
    negative log-likelihood using the source model's parameters — the core of
    any transferability index.
    """
    feats = model["features"]
    d = df_sub.dropna(subset=feats + ["log_len"]).reset_index(drop=True)
    if len(d) == 0:
        return np.nan
    Xs = model["scaler"].transform(d[feats].values.astype(float))
    for i in range(len(feats)):
        d[f"__f{i}__"] = Xs[:, i]
    fstd = [f"__f{i}__" for i in range(len(feats))]
    d["_id"] = np.arange(len(d))
    d["_off"] = d["log_len"].values
    manual_spec = {
        "fixed_terms": fstd, "rdm_terms": [], "rdm_cor_terms": [],
        "grouped_terms": [], "hetro_in_means": [],
        "transformations": ["no"] * len(fstd),
        "dispersion": 1, "latent_classes": 1,
    }
    data, spec = build_model_from_manual_spec(
        d, manual_spec, id_col="_id", y_col=model["y_col"], offset_col="_off")
    spec = replace(spec, model="nb")
    ll = -float(mixed_model_loglik(jnp.array(model["params"]), data, spec, indivi=False))
    return ll


def _pred_mean(df_sub, model):
    mu, _ = _rebuild_mu(df_sub.dropna(subset=model["features"] + ["log_len"]).reset_index(drop=True), model)
    return mu


def _mae_rmse(obs, pred):
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    m = np.isfinite(pred) & np.isfinite(obs) & (pred >= 0)
    if m.sum() == 0:
        return np.nan, np.nan
    e = obs[m] - pred[m]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def transferability_analysis(df, hb_final_feats, crash_final_feats, nm_col, out_dir):
    """STEP 11: Formal transferability of the crash model across space and of
    the HB proxy as a transferred crash surrogate.

    Methods (per left-out LGA):
      A. Spatial leave-one-LGA-out transfer: fit crash model on all-but-one LGA,
         apply to the held-out LGA. Koppelman-Wilmot-style Transferability Index
         TI = (LL_transferred - LL_const) / (LL_refit - LL_const).
      B. Cross-model HB -> crash proxy transfer: fit geometry->HB proxy on train,
         extrapolate HB to held-out LGA, feed predicted HB load into the crash
         model, and test whether it beats the geometry-only transfer.
      C. Log-likelihood transfer test and prediction error (MAE/RMSE) transfer.
      D. Spatial coefficient stability: distribution of each crash coefficient
         across the 11 leave-one-LGA-out training fits.
    """
    lgas = sorted(df["lga"].dropna().unique())
    roi_cols = ["lga", nm_col, "ObsCrashes", "log_len"] + sorted(
        set(crash_final_feats) | set(hb_final_feats))
    duse = df[roi_cols].dropna(subset=["lga", "ObsCrashes", "log_len"]).reset_index(drop=True)
    duse = duse[duse["lga"].isin(lgas)]

    rows = []
    train_fits = []       # per-LGA source crash model (leaves lga out)
    hb_proxy_fits = []    # per-LGA geometry->HB proxy model
    for lga in lgas:
        tr = duse[duse["lga"] != lga].dropna(
            subset=crash_final_feats + hb_final_feats + ["ObsCrashes", "log_len"])
        te = duse[duse["lga"] == lga].dropna(
            subset=crash_final_feats + hb_final_feats + ["ObsCrashes", "log_len"])
        if len(tr) < 50 or len(te) < 20:
            continue

        # intercept-only (baseline) on the held-out target
        ll_const, k_const = _fit_intercept_nb(te, "ObsCrashes")

        # -- A/C: geometry-only crash model transferred
        m_src = fit_model(tr, crash_final_feats, "ObsCrashes", "log_len",
                          label=f"Src-crash (excl {lga})")
        ll_xfer = _apply_model_ll(te, m_src)

        # refit on the target alone (best-case local transfer)
        m_refit = fit_model(te, crash_final_feats, "ObsCrashes", "log_len",
                            label=f"Refit {lga}")
        ll_refit = m_refit["ll"]

        # TI (Koppelman-Wilmot): 1 => transferred fits like local refit; 0 => no better than constant
        denom = (ll_refit - ll_const)
        ti = (ll_xfer - ll_const) / denom if abs(denom) > 1e-9 else np.nan

        # prediction errors on target
        pred_src = _pred_mean(te, m_src)
        pred_ref = _pred_mean(te, m_refit)
        ma_xfer, rm_xfer = _mae_rmse(te["ObsCrashes"].values, pred_src)
        ma_ref,  rm_ref  = _mae_rmse(te["ObsCrashes"].values, pred_ref)
        rmse_ratio = rm_ref / rm_xfer if rm_xfer > 0 else np.nan  # <1 => transfer as good as local

        # -- B: geometry->HB proxy on train, extrapolate to target, then crash model with HB load
        m_hb = fit_model(tr, hb_final_feats, nm_col, "log_len", label=f"HB-proxy (excl {lga})")
        hb_pred_tr = _pred_mean(tr, m_hb)
        hb_pred_te = _pred_mean(te, m_hb)
        tr_aug = tr.copy();  tr_aug["HBp"] = np.log1p(hb_pred_tr)
        te_aug_now = te.copy(); te_aug_now["HBp"] = np.log1p(hb_pred_te)
        m_hbc = fit_model(tr_aug, crash_final_feats + ["HBp"], "ObsCrashes", "log_len",
                          label=f"Crash+HBproxy (excl {lga})")
        ll_hb = _apply_model_ll(te_aug_now, m_hbc)

        rows.append({
            "LGA": lga, "n_train": len(tr), "n_test": len(te),
            "LL_const": ll_const, "LL_transferred": ll_xfer, "LL_refit": ll_refit,
            "TI": float(ti),
            "LL_crashHBproxy": ll_hb,
            "HBproxy_dLL": ll_hb - ll_xfer,          # >0 => proxy transfer beats geometry alone
            "MAE_transferred": ma_xfer, "MAE_refit": ma_ref,
            "RMSE_transferred": rm_xfer, "RMSE_refit": rm_ref,
            "RMSE_ratio_ref_vs_xfer": rmse_ratio,
        })
        train_fits.append((lga, m_src))
        hb_proxy_fits.append((lga, m_hb))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "hb_transferability_lga.csv"), index=False)

    # ---- D: spatial coefficient stability across leave-one-LGA-out crash fits
    if train_fits:
        stab_rows = []
        coef_names = [r["Variable"] for r in train_fits[0][1]["coef_table"].to_dict("records")]
        for m_src_name, m_src in train_fits:
            tbl = m_src["coef_table"].set_index("Variable")
            row = {"LGA": m_src_name}
            for v in coef_names:
                if v in tbl.index:
                    row[v] = tbl.loc[v, "Estimate"]
            stab_rows.append(row)
        stab = pd.DataFrame(stab_rows).set_index("LGA")
        stab_summary = pd.DataFrame({
            "Variable": stab.columns,
            "mean": stab.mean(), "std": stab.std(ddof=1),
            "min": stab.min(), "max": stab.max(),
            "abs_mean_over_std": (stab.mean().abs() / stab.std(ddof=1).replace(0, np.nan)),
        })
        stab_summary.to_csv(os.path.join(out_dir, "hb_transferability_coef_stability.csv"), index=False)
        stab.to_csv(os.path.join(out_dir, "hb_transferability_coef_by_lga.csv"))
    else:
        stab_summary = pd.DataFrame()

    # ---- summary metrics
    summary = pd.DataFrame([{
        "metric": "TI_mean",
        "value": res["TI"].mean() if len(res) else np.nan,
        "note": "Koppelman-Wilmot TI across leave-one-LGA-out folds (1=perfect transfer)",
    }, {
        "metric": "TI_min",
        "value": res["TI"].min() if len(res) else np.nan,
        "note": "worst-transferring region",
    }, {
        "metric": "HBproxy_dLL_mean",
        "value": res["HBproxy_dLL"].mean() if len(res) else np.nan,
        "note": "mean LL gain of crash+HB-proxy transfer vs geometry-only (>0 is evidence)",
    }, {
        "metric": "HBproxy_dLL_positive_folds",
        "value": int((res["HBproxy_dLL"] > 0).sum()) if len(res) else np.nan,
        "note": "number of folds where HB proxy improved crash transfer",
    }, {
        "metric": "RMSE_ratio_ref_vs_xfer_mean",
        "value": res["RMSE_ratio_ref_vs_xfer"].mean() if len(res) else np.nan,
        "note": "mean RMSE(refit)/RMSE(transferred); <1 means transfer matches local refit",
    }, {
        "metric": "coef_median_abs_mean_over_std",
        "value": stab_summary["abs_mean_over_std"].median() if len(stab_summary) else np.nan,
        "note": "median |mean|/SD of coefficients across folds; high => stable transfer",
    }])
    summary.to_csv(os.path.join(out_dir, "hb_transferability_summary.csv"), index=False)

    # ---- LaTeX table for the paper
    lines = ["\\begin{table}[htbp]\n\\centering\n\\caption{Spatial Transferability of the Crash Model and HB Proxy "
             "(leave-one-LGA-out)}\\label{tab:transfer}\\small\n\\begin{tabular}{lrrrrrr}\n\\hline\n"
             "LGA & n\\_test & LL\\_trans & LL\\_refit & TI & HB-proxy $\\Delta$LL & RMSE\\_ratio\\\\\n\\hline"]
    if len(res):
        for r in res.to_dict("records"):
            lines.append(f"{r['LGA']} & {int(r['n_test'])} & {r['LL_transferred']:.1f} & "
                         f"{r['LL_refit']:.1f} & {r['TI']:.3f} & {r['HBproxy_dLL']:+.1f} & "
                         f"{(r['RMSE_ratio_ref_vs_xfer'] if np.isfinite(r['RMSE_ratio_ref_vs_xfer']) else float('nan')):.3f}\\\\")
    lines.append("\\hline\n\\end{tabular}\n\\end{table}")
    with open(os.path.join(out_dir, "hb_transferability.tex"), "w") as fh:
        fh.write("\n".join(lines))

    print(f"  Transferability: mean TI={res['TI'].mean():.3f} (min {res['TI'].min():.3f})" if len(res) else "  no folds")
    if len(res):
        print(f"  HB-proxy transfer LL gain: mean={res['HBproxy_dLL'].mean():+.1f}, "
              f"{int((res['HBproxy_dLL']>0).sum())}/{len(res)} folds positive")
    return res, stab_summary


def main():
    print("\n" + "="*70)
    print("  HARD-BRAKING CATEGORY PROXY VARIABLE ANALYSIS")
    print("  Can Hard-Braking Events (by Severity) Serve as Crash Proxies?")
    print("="*70)
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Output dir:  {OUTPUT_DIR}")

    cp_main = Checkpoint("step_00_data")
    cp_de_nm  = Checkpoint("step_01a_de_nm")
    cp_de_cr  = Checkpoint("step_01b_de_cr")
    cp_de   = Checkpoint("step_01_de_search")
    cp_mcr  = Checkpoint("step_02_mcr_search")
    cp_transfer = Checkpoint("step_05_transfer")
    cp_enc  = Checkpoint("step_06_encompassing")
    cp_biv  = Checkpoint("step_07_bivariate")
    cp_cop  = Checkpoint("step_08_copula")
    cp_cats = Checkpoint("step_10_hb_categories")
    cp_transf = Checkpoint("step_11_transferability")

    # ── DATA ──
    if not cp_main.is_done():
        df, GEO, NM_RESP = load_data()
        nm_col = "nm_total"
        cp_main.mark_done()
    else:
        print("\n  STEP 00 already complete (cached)")
        df, GEO, NM_RESP = load_data()
        nm_col = "nm_total"

    # ── STEP 1: DE FEATURE SEARCH ──
    # Sub-checkpoint per response so partial progress survives walltime kills.
    nm_features_csv = os.path.join(OUTPUT_DIR, "hb_de_features_nm.csv")
    cr_features_csv = os.path.join(OUTPUT_DIR, "hb_de_features_crash.csv")

    if not cp_de.is_done():
        print("\n\n" + "#"*70)
        print("#  STEP 1: DIFFERENTIAL EVOLUTION — Best geometry for HB vs crashes")
        print("#"*70)

        # --- 1a: HB response DE ---
        if not cp_de_nm.is_done():
            de_nm, de_nm_feats, de_nm_hist = differential_evolution_feature_search(
                df, GEO, nm_col, "log_len",
                label="HB_response", p_threshold=0.10, penalty_per_insig=10.0,
                maxiter=25, popsize=6, seed=42,
                plot_path=os.path.join(OUTPUT_DIR, "hb_de_convergence_nm.png"))

            pd.DataFrame({"feature": de_nm_feats, "response": "HB", "step": "DE"}).to_csv(
                nm_features_csv, index=False)
            model_to_latex(de_nm, os.path.join(OUTPUT_DIR, "hb_de_best_nm.tex"),
                           "DE Best Model: Hard-Braking Response", "tab:hb_de_nm")
            cp_de_nm.mark_done()
            print("  STEP 1a (HB DE) COMPLETE")
        else:
            print("\n  STEP 1a (HB DE) already complete (resume)")
            de_nm_feats = pd.read_csv(nm_features_csv)["feature"].tolist()
            de_nm = fit_model(df, de_nm_feats, nm_col, "log_len", label="DE-HB (resumed)")
            de_nm_hist = []  # history not saved; skip combined plot

        # --- 1b: Crash response DE ---
        if not cp_de_cr.is_done():
            print("\n  --- Starting Crash DE ---")
            de_crash, de_crash_feats, de_crash_hist = differential_evolution_feature_search(
                df, GEO, "ObsCrashes", "log_len",
                label="Crash_response", p_threshold=0.10, penalty_per_insig=10.0,
                maxiter=25, popsize=6, seed=43,
                plot_path=os.path.join(OUTPUT_DIR, "hb_de_convergence_crash.png"))

            pd.DataFrame({"feature": de_crash_feats, "response": "Crash", "step": "DE"}).to_csv(
                cr_features_csv, index=False)
            model_to_latex(de_crash, os.path.join(OUTPUT_DIR, "hb_de_best_crash.tex"),
                           "DE Best Model: Crash Response", "tab:hb_de_crash")
            cp_de_cr.mark_done()
            print("  STEP 1b (Crash DE) COMPLETE")
        else:
            print("\n  STEP 1b (Crash DE) already complete (resume)")
            de_crash_feats = pd.read_csv(cr_features_csv)["feature"].tolist()
            de_crash = fit_model(df, de_crash_feats, "ObsCrashes", "log_len", label="DE-Crash (resumed)")
            de_crash_hist = []  # history not saved; skip combined plot

        # Combined convergence plot (only if both histories present)
        if len(de_nm_hist) and len(de_crash_hist):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(range(1, len(de_nm_hist)+1), de_nm_hist, "darkorange", linewidth=0.8)
            ax1.set_title("DE: Hard-Braking Response"); ax1.set_xlabel("Generation"); ax1.grid(True, alpha=0.3)
            ax2.plot(range(1, len(de_crash_hist)+1), de_crash_hist, "darkorange", linewidth=0.8)
            ax2.set_title("DE: Crash Response"); ax2.set_xlabel("Generation"); ax2.grid(True, alpha=0.3)
            fig.suptitle("Differential Evolution Convergence (NB, significance-penalised)")
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "hb_de_convergence_combined.png"), dpi=150)
            plt.close(fig)
            cp_de.mark_done()
            print("  STEP 1 COMPLETE")
    else:
        print("\n  STEP 1 already complete (resume)")
        de_nm_feats = pd.read_csv(nm_features_csv)["feature"].tolist()
        de_crash_feats = pd.read_csv(cr_features_csv)["feature"].tolist()
        de_nm = fit_model(df, de_nm_feats, nm_col, "log_len", label="DE-HB (resumed)")
        de_crash = fit_model(df, de_crash_feats, "ObsCrashes", "log_len", label="DE-Crash (resumed)")

    # ── STEP 2: MCR STRUCTURE SEARCH ──
    if not cp_mcr.is_done():
        print("\n\n" + "#"*70)
        print("#  STEP 2: MCR STRUCTURE SEARCH")
        print("#"*70)

        mcr_nm = mcr_model_search(df, nm_col, de_nm_feats, "log_len",
                                  label="HB", n_starts=2, max_iter=500, R=5,
                                  max_latent_classes=1, out_dir=OUTPUT_DIR,
                                  N_max=1500, solver="de", pop_size=20)
        mcr_crash = mcr_model_search(df, "ObsCrashes", de_crash_feats, "log_len",
                                     label="Crash", n_starts=2, max_iter=500, R=5,
                                     max_latent_classes=1, out_dir=OUTPUT_DIR,
                                     N_max=1500, solver="de", pop_size=20)

        def _mcr_final_feats(mcr_result, fallback_feats):
            if mcr_result is None:
                return list(fallback_feats)
            feats = sorted(set(
                mcr_result.get("fixed_orig", [])
                + [r.split(":")[0] for r in mcr_result.get("rdm_i_orig", [])]
                + [r.split(":")[0] for r in mcr_result.get("rdm_c_orig", [])]
            ))
            return feats if len(feats) >= 2 else list(fallback_feats)

        hb_final_feats    = _mcr_final_feats(mcr_nm, de_nm_feats)
        crash_final_feats = _mcr_final_feats(mcr_crash, de_crash_feats)

        pd.DataFrame({"feature": hb_final_feats, "response": "HB"}).to_csv(
            os.path.join(OUTPUT_DIR, "hb_final_features_nm.csv"), index=False)
        pd.DataFrame({"feature": crash_final_feats, "response": "Crash"}).to_csv(
            os.path.join(OUTPUT_DIR, "hb_final_features_crash.csv"), index=False)

        cp_mcr.mark_done()
    else:
        print("\n  STEP 2 already complete (resume)")
        hb_final_feats    = pd.read_csv(os.path.join(OUTPUT_DIR, "hb_final_features_nm.csv"))["feature"].tolist()
        crash_final_feats = pd.read_csv(os.path.join(OUTPUT_DIR, "hb_final_features_crash.csv"))["feature"].tolist()

    # Fit final models
    spec_hb_model    = fit_model(df, hb_final_feats, nm_col, "log_len",
                                 label="Final: HB (searched spec)", verbose=False)
    spec_crash_model = fit_model(df, crash_final_feats, "ObsCrashes", "log_len",
                                 label="Final: Crash (searched spec)", verbose=False)

    overlap = set(hb_final_feats) & set(crash_final_feats)
    jaccard = len(overlap) / len(set(hb_final_feats) | set(crash_final_feats))
    print(f"\n  HB final spec ({len(hb_final_feats)}): {[_V(f) for f in hb_final_feats]}")
    print(f"  Crash final spec ({len(crash_final_feats)}): {[_V(f) for f in crash_final_feats]}")
    print(f"  Feature overlap: {len(overlap)} shared (Jaccard = {jaccard:.4f})")
    print(f"  HB model: LL={spec_hb_model['ll']:.2f}, BIC={spec_hb_model['bic']:.2f}")
    print(f"  Crash model: LL={spec_crash_model['ll']:.2f}, BIC={spec_crash_model['bic']:.2f}")

    model_to_latex(spec_hb_model, os.path.join(OUTPUT_DIR, "hb_final_spec_nm.tex"),
                   "MCR-Searched Final Model: Hard-Braking Response", "tab:hb_final_nm")
    model_to_latex(spec_crash_model, os.path.join(OUTPUT_DIR, "hb_final_spec_crash.tex"),
                   "MCR-Searched Final Model: Crash Response", "tab:hb_final_crash")

    # ── STEP 3: FULL MODELS ──
    print("\n\n" + "#"*70)
    print("#  STEP 3: Full models (all geometry baseline)")
    print("#"*70)
    full_hb = fit_model(df, GEO, nm_col, "log_len", label="Full: Geo -> HB", verbose=False)
    full_crash = fit_model(df, GEO, "ObsCrashes", "log_len", label="Full: Geo -> Crashes", verbose=False)
    print(f"  HB full model:    LL={full_hb['ll']:.2f}, BIC={full_hb['bic']:.2f}, k={full_hb['k']}")
    print(f"  Crash full model: LL={full_crash['ll']:.2f}, BIC={full_crash['bic']:.2f}, k={full_crash['k']}")

    # ── STEP 4: COEFFICIENT PROPORTIONALITY ──
    prop = coefficient_proportionality_test(full_hb, full_crash, OUTPUT_DIR)

    # ── STEP 5: SPECIFICATION TRANSFER ──
    if not cp_transfer.is_done():
        transfer = specification_transfer(df, hb_final_feats, crash_final_feats, OUTPUT_DIR)
        cp_transfer.mark_done()
    else:
        print("\n  STEP 5 already complete (resume)")
        transfer = specification_transfer(df, hb_final_feats, crash_final_feats, OUTPUT_DIR)

    # ── STEP 6: ENCOMPASSING TEST ──
    if not cp_enc.is_done():
        enc = encompassing_test(df, hb_final_feats, crash_final_feats, OUTPUT_DIR)
        cp_enc.mark_done()
    else:
        print("\n  STEP 6 already complete (resume)")
        enc = encompassing_test(df, hb_final_feats, crash_final_feats, OUTPUT_DIR)

    # ── STEP 7: BIVARIATE LATENT RISK ──
    if not cp_biv.is_done():
        geo_control_feats = sorted(set(hb_final_feats) | set(crash_final_feats))
        biv = bivariate_latent_risk(df, geo_control_feats, nm_col, OUTPUT_DIR)
        cp_biv.mark_done()
    else:
        print("\n  STEP 7 already complete (resume)")
        geo_control_feats = sorted(set(hb_final_feats) | set(crash_final_feats))
        biv = bivariate_latent_risk(df, geo_control_feats, nm_col, OUTPUT_DIR)

    # ── STEP 8: BIVARIATE COPULA ──
    if not cp_cop.is_done():
        cop = bivariate_copula_analysis(df, nm_col, spec_hb_model, spec_crash_model, OUTPUT_DIR)
        cp_cop.mark_done()
    else:
        print("\n  STEP 8 already complete (resume)")
        cop = bivariate_copula_analysis(df, nm_col, spec_hb_model, spec_crash_model, OUTPUT_DIR)

    # ── STEP 10: HB CATEGORY MULTIVARIATE ──
    if not cp_cats.is_done():
        cat_models, cat_copula = hb_category_multivariate_analysis(df, hb_final_feats, OUTPUT_DIR)
        cp_cats.mark_done()
    else:
        print("\n  STEP 10 already complete (resume)")
        cat_models, cat_copula = hb_category_multivariate_analysis(df, hb_final_feats, OUTPUT_DIR)

    # ── STEP 11: SPATIAL TRANSFERABILITY ──
    if not cp_transf.is_done():
        transf, transf_stab = transferability_analysis(
            df, hb_final_feats, crash_final_feats, nm_col, OUTPUT_DIR)
        cp_transf.mark_done()
    else:
        print("\n  STEP 11 already complete (resume)")
        transf = pd.read_csv(os.path.join(OUTPUT_DIR, "hb_transferability_lga.csv"))
        transf_stab = pd.read_csv(os.path.join(OUTPUT_DIR, "hb_transferability_coef_stability.csv"))

    # ── SYNTHESIS ──
    print("\n\n" + "="*70)
    print("  SYNTHESIS: Is Hard-Braking a Valid Crash Proxy?")
    print("="*70)

    evidence = []
    if prop["corr"] > 0.7 and prop["R2_prop"] > 0.5:
        evidence.append(("PRO", f"Coefficients proportional: r={prop['corr']:.3f}, R2={prop['R2_prop']:.3f}"))
    else:
        evidence.append(("CON", f"Coefficients NOT proportional: r={prop['corr']:.3f}, R2={prop['R2_prop']:.3f}"))

    if jaccard > 0.5:
        evidence.append(("PRO", f"High feature overlap: Jaccard={jaccard:.3f}"))
    else:
        evidence.append(("CON", f"Low feature overlap: Jaccard={jaccard:.3f}"))

    penalty_cr = transfer["HB-spec -> Crashes"]["bic"] - transfer["Crash-spec -> Crashes (native)"]["bic"]
    if abs(penalty_cr) < 20:
        evidence.append(("PRO", f"HB spec transfers to crashes: dBIC={penalty_cr:+.1f}"))
    else:
        evidence.append(("CON", f"HB spec degrades on crashes: dBIC={penalty_cr:+.1f}"))

    if enc["p"] < 0.05:
        evidence.append(("PRO", f"Predicted HB encompasses geometry: LR={enc['lr_stat']:.1f}, p={enc['p']:.4f}"))
    else:
        evidence.append(("CON", f"HB does NOT encompass geometry: LR={enc['lr_stat']:.1f}, p={enc['p']:.4f}"))

    if biv["partial_r"] > 0.2 and biv["partial_p"] < 1e-10:
        evidence.append(("PRO", f"Shared latent risk: partial r={biv['partial_r']:.3f}"))
    else:
        evidence.append(("CON", f"Weak shared latent risk: partial r={biv['partial_r']:.3f}"))

    best_cop = cop["best_copula"]
    cop_fit = cop["fits"][best_cop]
    if cop_fit["p_lr"] < 0.01 and cop_fit["bic_joint"] - cop["bic_indep"] < 0:
        evidence.append(("PRO", f"Bivariate copula rejects independence: theta={cop_fit['theta']:.3f}, LR={cop_fit['lr_stat']:.1f}, p={cop_fit['p_lr']:.2e}"))
    else:
        evidence.append(("CON", f"Bivariate copula does NOT improve independence: LR={cop_fit['lr_stat']:.1f}, p={cop_fit['p_lr']:.2e}"))

    # HB category evidence
    for cat, r in cat_copula.items():
        if r["p_lr"] < 0.05 and r["bic_joint"] - r["bic_indep"] < 0:
            evidence.append(("PRO", f"HB category '{cat}' copula rejects independence: LR={r['lr_stat']:.1f}, p={r['p_lr']:.2e}"))
        else:
            evidence.append(("CON", f"HB category '{cat}' not significant: LR={r['lr_stat']:.1f}, p={r['p_lr']:.2e}"))

    # Transferability evidence (STEP 11)
    if len(transf):
        t_mean = transf["TI"].mean()
        if t_mean > 0.7:
            evidence.append(("PRO", f"Spatial transferability: mean TI={t_mean:.3f} (crash model transfers across LGAs)"))
        elif t_mean > 0.4:
            evidence.append(("CON", f"Moderate spatial transferability: mean TI={t_mean:.3f}"))
        else:
            evidence.append(("CON", f"Poor spatial transferability: mean TI={t_mean:.3f} (crash model is region-specific)"))
        dll = transf["HBproxy_dLL"].mean()
        npos = int((transf["HBproxy_dLL"] > 0).sum())
        if dll > 0 and npos >= len(transf) * 0.6:
            evidence.append(("PRO", f"HB proxy transfers crash risk: mean dLL={dll:+.1f} ({npos}/{len(transf)} folds)"))
        else:
            evidence.append(("CON", f"HB proxy does not transfer crash risk: mean dLL={dll:+.1f} ({npos}/{len(transf)} folds)"))

    pro = sum(1 for e in evidence if e[0] == "PRO")
    con = sum(1 for e in evidence if e[0] == "CON")

    print(f"\n  {'Evidence':<80s} {'Verdict':>10s}")
    print(f"  {'-'*90}")
    for tag, msg in evidence:
        print(f"  {msg:<80s} {'[+]' if tag=='PRO' else '[-]'}")

    print(f"\n  Score: {pro} PRO / {con} CON")
    if pro >= 4:
        ver = "STRONG PROXY: HB can serve as a crash surrogate"
    elif pro >= 2:
        ver = "PARTIAL PROXY: HB provides complementary information"
    else:
        ver = "WEAK PROXY: HB and crashes have different structural drivers"
    print(f"  VERDICT: {ver}")

    pd.DataFrame([{"Tag": t, "Evidence": m} for t, m in evidence]).to_csv(
        os.path.join(OUTPUT_DIR, "hb_proxy_evidence.csv"), index=False)

    print(f"\n  All outputs in: {OUTPUT_DIR}")
    print("="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HB proxy variable analysis")
    ap.add_argument("--data",    default=None, help="Path to near_miss_hb_ecf.csv")
    ap.add_argument("--out-dir", default=None, dest="out_dir", help="Output directory")
    args = ap.parse_args()
    if args.data:
        DATA_PATH = args.data
    if args.out_dir:
        OUTPUT_DIR = args.out_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    main()
