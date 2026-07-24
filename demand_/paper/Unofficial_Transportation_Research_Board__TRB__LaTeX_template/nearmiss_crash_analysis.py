"""
Proxy Variable Analysis: Can Near-Miss Events Serve as a Crash Surrogate?
==========================================================================
Tests whether near-miss metrics can serve as a valid proxy/leading indicator
for crash frequency at the road-segment level.

Econometric Framework:
  1. Dual Harmony Search — find best geometry features for NM vs crashes
  2. Coefficient Proportionality Test — are β_crash ∝ β_nm?
  3. Specification Transfer — does NM-optimal spec predict crashes?
  4. Encompassing Test — does predicted NM encompass geometry?
  5. Bivariate latent risk correlation
  6. Non-nested Vuong-style test

Data: Z:/test_runs_tours/demand_/near_miss_ecf.csv
Python: C:/Users/ahernz/AppData/Local/Temp/opencode/nm_venv/Scripts/python.exe
"""

import os, sys, warnings, argparse
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_CANDIDATES = [
    r"Z:\test_runs_tours\demand_\near_miss_ecf.csv",                             # Windows mapped drive
    os.path.join(os.path.dirname(_SCRIPT_DIR), "near_miss_ecf.csv"),             # parent of script dir
    os.path.expanduser("~/test_runs_tours/demand_/near_miss_ecf.csv"),           # cluster home
    "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/near_miss_ecf.csv",        # explicit cluster path
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "nm_total": "Near-Miss: Total Count",
    "nm_braking": "Near-Miss: Hard Braking Count",
    "nm_per_km": "Near-Miss: Per-km Rate",
    "avg_accel_all": "Mean Acceleration (G)",
    "avg_speed_all": "Mean Speed (km/h)",
    "avg_iri_all": "Mean IRI (Roughness)",
    "ExpCrashes": "Expected Crash Frequency (ECF)",
    "ObsCrashes": "Observed Crash Count",
    "pred_nm": "Predicted Near-Miss Count",
}
_V = lambda c: VLABEL.get(c, c)


# ===================================================================
# 1. DATA
# ===================================================================
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # CU_ST (straight) omitted: the four curvature dummies are mutually exclusive
    # and exhaustive (CU_ST + CU_MOD + CU_SHA + CU_VSHA = 1 on every row),
    # causing perfect multicollinearity with the intercept. CU_MOD/CU_SHA/CU_VSHA
    # are interpreted relative to straight alignment as the reference category.
    GEO = [c for c in [
        "AADT","SL","NL","SKD_SP","CU_MOD","CU_SHA","CU_VSHA",
        "DEL_P","IC","IP","MEDPM","MEDSFBar","PA_COM1",
        "PSPS1T24","PSPSNIL","PSDSG24","PSDS1T24","PSDSNIL",
        "IV_VL","IV_L","IV_M","IV_100",
        "RSPSDL5","RSDSDL1","RSDSDG10","RSDSO",
        "RSPSDL1","RSDSDL5","RSPSDL10","RSDSDL10","RSPSDG10",
    ] if c in df.columns and df[c].std() > 1e-10]

    df = df.dropna(subset=["ObsCrashes"]); df = df[df["ObsCrashes"] >= 0]
    for c in GEO:
        if df[c].isna().any(): df[c] = df[c].fillna(0)

    nm_response_cols = [c for c in ["nm_braking", "nm_total"] if c in df.columns]
    for c in nm_response_cols:
        if c in df.columns and df[c].isna().any(): df[c] = df[c].fillna(0)

    df["seg_len_km"] = (df.get("segment_length_m", 1000)/1000).clip(lower=0.001)
    df["log_len"] = np.log(df["seg_len_km"])
    df["log_ecf"] = np.log(np.maximum(df["ExpCrashes"], 1e-6))

    print(f"  Segments: {len(df):,}  |  Geometry features: {len(GEO)}  |  "
          f"NM responses: {len(nm_response_cols)}")
    print(f"  Has crashes: {df['ObsCrashes'].gt(0).sum():,} ({df['ObsCrashes'].gt(0).mean()*100:.1f}%)")
    return df, GEO, nm_response_cols


# ===================================================================
# 2a. NUMERICAL HESSIAN SE (fallback when JAX autodiff is unreliable)
# ===================================================================
def _scipy_hessian_se(obj_fn, params, eps=1e-4):
    """Compute SEs via central-difference Hessian of the negative log-likelihood.

    obj_fn minimises -LL, so its Hessian H ≈ Fisher information.
    SE = sqrt(diag(H⁻¹)).  Uses lstsq instead of inv for numerical stability.
    Cost: O(k²) function evaluations — acceptable for k ≤ 35.
    """
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
        # Use least-squares pseudo-inverse for robustness against near-singularity
        cov = np.linalg.lstsq(H, np.eye(K), rcond=None)[0]
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        se = np.full(K, np.nan)
    return se


# ===================================================================
# 2. MODEL FITTER
# ===================================================================
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

    # Try JAX-based SEs first; fall back to scipy numerical Hessian
    se = None
    try:
        se_jax = np.asarray(compute_standard_errors(jnp.array(params), obj))
        if np.isfinite(se_jax).any() and np.nanmax(np.abs(se_jax)) < 50:
            se = se_jax
    except Exception:
        pass
    if se is None:
        se = _scipy_hessian_se(obj, params)
    se = np.where(np.abs(se) > 50, np.nan, se)  # flag truly degenerate SEs

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
            "se": se, "features": features, "y_col": y_col, "scaler": scaler}


# ===================================================================
# 3. SIMULATED ANNEALING — parsimonious feature selection
# ===================================================================
def simulated_annealing_search(df, candidates, y_col, offset_col, max_iter=1200,
                               label="SA", plot_path=None, p_threshold=0.10,
                               penalty_per_insig=10.0, T_start=8.0, T_end=0.01,
                               cooling_rate=0.995, n_restarts=3):
    """Simulated Annealing with significance penalty for parsimonious models.

    Algorithm:
      1. Start with a random feature subset at temperature T_start
      2. Each iteration: propose flipping one random feature on/off
      3. Always accept if penalised_BIC improves
      4. Accept worse states with probability exp(-delta / T)
      5. Cool: T = T * cooling_rate
      6. Track the best solution seen (elitism)

    The penalised BIC = raw_BIC + n_insig * penalty_per_insig,
    where n_insig counts included coefficients with p > p_threshold.
    """
    print(f"\n  SIMULATED ANNEALING [{label}] — y = {y_col}")
    print(f"  {len(candidates)} candidates, {max_iter} iters x {n_restarts} restarts, NB, "
          f"p-threshold={p_threshold}, penalty={penalty_per_insig}/insig")
    print(f"  T_start={T_start}, T_end={T_end}, cooling={cooling_rate}")

    dfs = df.dropna(subset=candidates + [y_col, offset_col]).reset_index(drop=True)
    af, Kf = list(candidates), len(candidates)

    def random_mask(lo=3, hi=min(12, Kf), rng=None):
        rng = rng or np.random
        s = rng.randint(lo, hi+1)
        m = np.zeros(Kf, int); m[rng.choice(Kf, s, replace=False)] = 1
        return m

    def _propose(mask, rng):
        move = rng.randint(3)  # 0=flip, 1=swap, 2=double-flip
        new = mask.copy()
        on  = np.where(new == 1)[0]
        off = np.where(new == 0)[0]
        if move == 0 or len(on) < 2 or len(off) < 1:
            # flip one feature
            idx = rng.randint(Kf)
            new[idx] = 1 - new[idx]
        elif move == 1:
            # swap one on↔off (preserves feature count, explores landscape)
            new[rng.choice(on)]  = 0
            new[rng.choice(off)] = 1
        else:
            # double flip: toggle two random features
            for idx in rng.choice(Kf, 2, replace=False):
                new[idx] = 1 - new[idx]
        if new.sum() < 2:
            new[rng.randint(Kf)] = 1  # enforce minimum
        return new

    def score(mask):
        sel = [af[i] for i in range(Kf) if mask[i] == 1]
        if len(sel) < 2:
            return 1e12
        o = fit_model(dfs, sel, y_col, offset_col, label="SA", verbose=False)
        if o is None:
            return 1e12
        coefs = o["coef_table"]
        feature_rows = coefs.iloc[1:]
        if coefs["Variable"].iloc[-1].startswith("ln("):
            feature_rows = feature_rows.iloc[:-1]
        # NaN SE → treat as insignificant so penalty fires; avoids ill-identified models
        n_insig = sum(1 for p in feature_rows["p"]
                      if np.isnan(p) or p > p_threshold)
        return o["bic"] + n_insig * penalty_per_insig

    # --- Multi-restart: run n_restarts independent chains, keep global best ---
    global_best_mask  = None
    global_best_score = 1e12
    global_best_history = []

    for restart in range(n_restarts):
        rng = np.random.RandomState(42 + restart * 17)
        current_mask  = random_mask(rng=rng)
        current_score = score(current_mask)
        best_mask_r   = current_mask.copy()
        best_score_r  = current_score
        run_history   = [current_score]
        T = T_start

        print(f"\n  [Restart {restart+1}/{n_restarts}] init score={current_score:.2f}  "
              f"T={T:.4f}  feats={current_mask.sum()}")

        for it in range(2, max_iter + 1):
            new_mask  = _propose(current_mask, rng)
            new_score = score(new_mask)
            delta     = new_score - current_score

            if delta <= 0:
                accept = True
            else:
                prob   = np.exp(-delta / max(T, 1e-9))
                accept = rng.random() < prob

            if accept:
                current_mask  = new_mask
                current_score = new_score
                if current_score < best_score_r:
                    best_score_r = current_score
                    best_mask_r  = current_mask.copy()

            T = max(T * cooling_rate, T_end)
            run_history.append(best_score_r)

            if it % 100 == 0 or it == max_iter:
                print(f"  Iter {it:4d}: best={best_score_r:.2f}  T={T:.4f}  "
                      f"feats={best_mask_r.sum()}")

        if best_score_r < global_best_score:
            global_best_score   = best_score_r
            global_best_mask    = best_mask_r.copy()
        global_best_history.extend(run_history)

    best_mask = global_best_mask
    best_history = global_best_history

    # --- Final model ---
    bf = [af[i] for i in range(Kf) if best_mask[i]==1]
    final = fit_model(dfs, bf, y_col, offset_col, label=f"SA-{label}", verbose=False)
    coefs_f = final["coef_table"]
    feature_rows_f = coefs_f.iloc[1:]
    if coefs_f["Variable"].iloc[-1].startswith("ln("):
        feature_rows_f = feature_rows_f.iloc[:-1]
    n_insig_f = sum(1 for p in feature_rows_f["p"]
                    if np.isnan(p) or p > p_threshold)
    penalized_bic_f = final["bic"] + n_insig_f * penalty_per_insig

    print(f"\n  BEST [{label}]: raw_BIC={final['bic']:.2f}  "
          f"penalised_BIC={penalized_bic_f:.2f}  "
          f"insig={n_insig_f}/{len(bf)}  |  {len(bf)} features:")
    for f in bf:
        pv = feature_rows_f[coefs_f["Variable"] == _V(f)]["p"].values
        p_str = f"p={pv[0]:.4f}" if len(pv) and np.isfinite(pv[0]) else "p=nan"
        sig = "***" if (len(pv) and (not np.isnan(pv[0])) and pv[0] < p_threshold) else ""
        print(f"    {sig:3s} {_V(f):<50s} {p_str}")

    # Convergence plot
    if plot_path:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(best_history)+1), best_history, "b-", linewidth=1, alpha=0.7)
        ax.axhline(y=min(best_history), color="red", linestyle="--", alpha=0.5,
                   label=f"Best = {min(best_history):.2f}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Penalised BIC (best so far)")
        ax.set_title(f"Simulated Annealing Convergence — {label}\n"
                     f"(penalty = {penalty_per_insig} per p > {p_threshold}, "
                     f"cooling rate = {cooling_rate})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Convergence plot saved: {plot_path}")

    return final, bf, best_history


def differential_evolution_feature_search(
    df, candidates, y_col, offset_col, maxiter=60, popsize=15,
    label="DE", plot_path=None, p_threshold=0.10,
    penalty_per_insig=10.0, seed=42,
):
    """Differential Evolution for binary feature selection with significance penalty.

    Each candidate feature gets one continuous gene in [0, 1]; a gene > 0.5
    includes that feature.  scipy's ``differential_evolution`` minimises
    penalised_BIC = raw_BIC + n_insig * penalty_per_insig.
    """
    print(f"\n  DIFFERENTIAL EVOLUTION [{label}] — y = {y_col}")
    print(f"  {len(candidates)} candidates, maxiter={maxiter}, popsize={popsize}, "
          f"p-threshold={p_threshold}, penalty={penalty_per_insig}/insig")

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
        n_insig = sum(1 for p in feature_rows["p"]
                      if np.isnan(p) or p > p_threshold)
        return o["bic"] + n_insig * penalty_per_insig

    history = []

    def _callback(xk, convergence=None):
        history.append(_score(xk))

    print(f"  Launching scipy.differential_evolution (seed={seed}) ...")
    res = differential_evolution(
        _score, bounds=[(0.0, 1.0)] * Kf,
        maxiter=maxiter, popsize=popsize, seed=seed,
        mutation=(0.4, 1.4), recombination=0.7,
        # tol=0 disables scipy's population-variance early-stopping check
        # (default tol=0.01), which was firing after 1-2 generations here
        # -- the binary-via-threshold feature-selection landscape has large
        # flat regions (many masks collapse to the same score), so the
        # population looked "converged" almost immediately and the
        # convergence plot only ever got 1-2 points. This forces the full
        # maxiter generations to run every time.
        tol=0.0,
        polish=False, callback=_callback,
    )
    history.append(_score(res.x))
    best_so_far = np.minimum.accumulate(history)

    best_mask = (np.asarray(res.x) > 0.5).astype(int)
    bf = [af[i] for i in range(Kf) if best_mask[i] == 1]

    if len(bf) < 2:
        print(f"  DE [{label}] returned <2 features; falling back to random mask.")
        bf = list(np.random.choice(af, max(2, min(8, len(af))), replace=False))

    # Final model fit
    final = fit_model(dfs, bf, y_col, offset_col, label=f"DE-{label}", verbose=False)
    coefs_f = final["coef_table"]
    feature_rows_f = coefs_f.iloc[1:]
    if coefs_f["Variable"].iloc[-1].startswith("ln("):
        feature_rows_f = feature_rows_f.iloc[:-1]
    n_insig_f = sum(1 for p in feature_rows_f["p"]
                    if np.isnan(p) or p > p_threshold)
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
        ax.axhline(y=min(best_so_far), color="red", linestyle="--", alpha=0.5,
                   label=f"Best = {min(best_so_far):.2f}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Penalised BIC (best so far)")
        ax.set_title(f"Differential Evolution Convergence — {label}\n"
                     f"(penalty={penalty_per_insig} per p > {p_threshold})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Convergence plot saved: {plot_path}")

    return final, bf, best_so_far.tolist()


def model_to_latex(model, filename, caption, label):
    """Export coefficient table as LaTeX .tex fragment."""
    tbl = model["coef_table"].copy()
    # Format columns
    latex_rows = []
    for _, row in tbl.iterrows():
        var = row["Variable"].replace("_", "\\_")
        var = var.replace("\\%", "%").replace("%", "\\%")  # avoid double-escaping labels already containing \%
        est = f"{row['Estimate']:+.4f}"
        irr = f"{row['IRR']:.4f}"
        zs = f"{row['z']:.2f}" if np.isfinite(row['z']) else "--"
        ps = f"{row['p']:.4f}" if np.isfinite(row['p']) else "--"
        sig = "***" if not np.isnan(row['p']) and row['p'] < 0.01 else \
              "**" if not np.isnan(row['p']) and row['p'] < 0.05 else \
              "*" if not np.isnan(row['p']) and row['p'] < 0.10 else ""
        latex_rows.append(f"    {var} & {est} & {irr} & {zs} & {ps} & {sig} \\\\")

    tex = f"""% Auto-generated by proxy variable analysis
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
  Dispersion $\\ln(\\alpha)$ = {model['params'][-1]:.4f}.
  All predictors standardized. IRR = Incidence Rate Ratio.}}
\\end{{table}}
"""
    with open(filename, "w") as f:
        f.write(tex)
    print(f"  LaTeX table saved: {filename}")


# ===================================================================
# 4. COEFFICIENT PROPORTIONALITY TEST
# ===================================================================
def coefficient_proportionality_test(model_nm, model_crash, out_dir):
    print("\n" + "="*70)
    print("  COEFFICIENT PROPORTIONALITY TEST")
    print("  H0: beta_crash = k * beta_nm (same structure, different scale)")
    print("="*70)

    feats_nm = model_nm["features"]
    feats_cr = model_crash["features"]
    common = sorted(set(feats_nm) & set(feats_cr))

    tbl_nm = model_nm["coef_table"]; tbl_cr = model_crash["coef_table"]
    beta_nm = {}; beta_cr = {}
    for f in common:
        vn = _V(f)
        rn = tbl_nm[tbl_nm["Variable"] == vn]
        rc = tbl_cr[tbl_cr["Variable"] == vn]
        if len(rn) and len(rc):
            beta_nm[f] = rn["Estimate"].values[0]
            beta_cr[f] = rc["Estimate"].values[0]

    common_f = list(beta_nm.keys())
    b_nm = np.array([beta_nm[f] for f in common_f])
    b_cr = np.array([beta_cr[f] for f in common_f])

    k_hat = np.sum(b_nm * b_cr) / np.sum(b_nm**2)
    residuals = b_cr - k_hat * b_nm
    rss = np.sum(residuals**2)
    corr = np.corrcoef(b_nm, b_cr)[0, 1]
    tss = np.sum(b_cr**2)
    R2_prop = 1 - rss / tss if tss > 0 else 0

    print(f"\n  Proportionality factor k = {k_hat:.4f}")
    print(f"  Correlation of coefficient vectors: r = {corr:.4f}")
    print(f"  Proportionality R^2 = {R2_prop:.4f}")

    # Scatter plot of coefficient vectors
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(b_nm, b_cr, alpha=0.6, s=40)
    # OLS through origin
    x_line = np.linspace(b_nm.min(), b_nm.max(), 100)
    ax.plot(x_line, k_hat * x_line, "r--", alpha=0.5, label=f"k={k_hat:.3f}")
    # 1:1 line for reference
    ax.plot(x_line, x_line, "gray", alpha=0.3, linestyle=":", label="1:1 line")
    for i, f in enumerate(common_f[:10]):  # label top 10
        ax.annotate(f, (b_nm[i], b_cr[i]), fontsize=6, alpha=0.7)
    ax.set_xlabel("beta (Near-Miss model)")
    ax.set_ylabel("beta (Crash model)")
    ax.set_title(f"Coefficient Proportionality\nr = {corr:.3f}, R^2 = {R2_prop:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "coefficient_proportionality.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Coefficients plot saved: {plot_path}")

    return {"k": k_hat, "corr": corr, "R2_prop": R2_prop}


# ===================================================================
# 5. SPECIFICATION TRANSFER
# ===================================================================
def specification_transfer(df, nm_features, crash_features, out_dir):
    print("\n" + "="*70)
    print("  SPECIFICATION TRANSFER TEST")
    print("="*70)

    results = {}
    for label, feats, y in [
        ("NM-spec -> NM (native)", nm_features, "nm_total"),
        ("NM-spec -> Crashes", nm_features, "ObsCrashes"),
        ("Crash-spec -> NM", crash_features, "nm_total"),
        ("Crash-spec -> Crashes (native)", crash_features, "ObsCrashes"),
    ]:
        r = fit_model(df, feats, y, "log_len", label=label, verbose=False)
        results[label] = r

    print(f"\n  {'Model':<40s} {'BIC':>10s} {'AIC':>10s} {'LL':>10s} {'k':>4s}")
    print(f"  {'-'*77}")
    for k in ["NM-spec -> NM (native)", "NM-spec -> Crashes",
              "Crash-spec -> NM", "Crash-spec -> Crashes (native)"]:
        m = results[k]
        print(f"  {m['label']:<40s} {m['bic']:10.2f} {m['aic']:10.2f} "
              f"{m['ll']:10.2f} {m['k']:4d}")

    nm_native = results["NM-spec -> NM (native)"]["bic"]
    cr_native = results["Crash-spec -> Crashes (native)"]["bic"]
    nm_to_cr = results["NM-spec -> Crashes"]["bic"]
    cr_to_nm = results["Crash-spec -> NM"]["bic"]

    penalty_nm = cr_to_nm - nm_native
    penalty_cr = nm_to_cr - cr_native

    print(f"\n  Transfer penalty (Crash-spec on NM): {penalty_nm:+.2f} BIC")
    print(f"  Transfer penalty (NM-spec on Crashes): {penalty_cr:+.2f} BIC")
    print(f"  Symmetric penalty: {(penalty_nm+penalty_cr)/2:+.2f} BIC")
    print(f"  -> Specifications are {'TRANSFERABLE' if abs(penalty_cr)<50 and abs(penalty_nm)<50 else 'NOT transferable'}")

    return results


# ===================================================================
# 6. ENCOMPASSING TEST
# ===================================================================
def encompassing_test(df, nm_feats, crash_feats, out_dir):
    """Does the NM model's predicted value encompass the Crash model's own
    geometry?  nm_feats and crash_feats should be each response's own final
    (searched) spec, not a single shared feature list for both — the two
    responses aren't assumed to need the same geometry.
    """
    print("\n" + "="*70)
    print("  ENCOMPASSING TEST: Does predicted NM encompass geometry?")
    print("="*70)

    nm_model = fit_model(df, nm_feats, "nm_total", "log_len",
                         label="Step1: NM ~ Geo", verbose=False)
    print(f"  NM model: LL={nm_model['ll']:.2f}, BIC={nm_model['bic']:.2f}")

    # Predict NM from fitted model
    all_feats = sorted(set(nm_feats) | set(crash_feats))
    df_e = df.dropna(subset=all_feats + ["nm_total", "ObsCrashes", "log_len"]).copy()
    params_nm = nm_model["params"]
    n_geo = len(nm_feats)

    # Re-standardize to get predictions
    scaler_en = StandardScaler()
    X_r = scaler_en.fit_transform(df_e[nm_feats].values.astype(float))
    intercept_nm = params_nm[0]
    beta_nm = params_nm[1:n_geo+1]
    eta_nm = intercept_nm + X_r @ beta_nm + df_e["log_len"].values
    df_e["pred_nm"] = np.exp(np.clip(eta_nm, -20, 20))
    # Ensure no NaN/Inf
    df_e["pred_nm"] = np.nan_to_num(df_e["pred_nm"], nan=0.0, posinf=1e6, neginf=0.0)
    if df_e["pred_nm"].isna().any():
        print("  WARNING: pred_nm still has NaN, filling with 0")
        df_e["pred_nm"] = df_e["pred_nm"].fillna(0)

    # Restricted: crashes ~ geometry (Crash's own final spec)
    crash_geo = fit_model(df_e, crash_feats, "ObsCrashes", "log_len",
                          label="Restricted: Crashes ~ Geo", verbose=False)
    print(f"  Restricted (Geo only): LL={crash_geo['ll']:.2f}, BIC={crash_geo['bic']:.2f}")

    # Unrestricted: crashes ~ geometry + predicted NM
    crash_nm = fit_model(df_e, crash_feats + ["pred_nm"], "ObsCrashes", "log_len",
                         label="Unrestricted: Crashes ~ Geo + predNM", verbose=False)
    print(f"  Unrestricted (+predNM): LL={crash_nm['ll']:.2f}, BIC={crash_nm['bic']:.2f}")

    ll_r = crash_geo["ll"]; ll_u = crash_nm["ll"]
    k_r = crash_geo["k"]; k_u = crash_nm["k"]
    lr_stat = 2 * (ll_u - ll_r); df_lr = k_u - k_r
    p_lr = 1 - scipy_stats.chi2.cdf(lr_stat, df_lr)

    print(f"\n  LR stat = {lr_stat:.2f}  |  df = {df_lr}  |  p = {p_lr:.6f}")
    print(f"  delta BIC = {crash_nm['bic'] - crash_geo['bic']:+.2f}")

    pred_nm_row = crash_nm["coef_table"][
        crash_nm["coef_table"]["Variable"] == "Predicted Near-Miss Count"
    ]
    if len(pred_nm_row) > 0:
        pv = pred_nm_row["p"].values[0]
        irr = pred_nm_row["IRR"].values[0]
        print(f"  pred_NM: IRR={irr:.4f}, p={pv:.4f}")
        print(f"  -> {'NM ENCOMPASSES geometry' if (not np.isnan(pv)) and pv < 0.05 and lr_stat > 3.84 else 'NM does NOT encompass geometry'}")

    return {"lr_stat": lr_stat, "df": df_lr, "p": p_lr,
            "bic_r": crash_geo["bic"], "bic_u": crash_nm["bic"]}


# ===================================================================
# 7. BIVARIATE LATENT RISK
# ===================================================================
def bivariate_latent_risk(df, geo_feats, nm_col, out_dir):
    """geo_feats should be the union of both responses' final (searched)
    feature sets — the best-known drivers of either response — rather than
    the full unselected candidate pool, so the partial correlation controls
    for the geometry that's actually believed to matter.
    """
    print("\n" + "="*70)
    print("  BIVARIATE LATENT RISK ANALYSIS")
    print("="*70)

    df_b = df.dropna(subset=geo_feats + [nm_col, "ObsCrashes", "log_len"]).copy()
    y_nm = df_b[nm_col].values / df_b["seg_len_km"].values
    y_cr = df_b["ObsCrashes"].values / df_b["seg_len_km"].values

    from scipy.stats import spearmanr, pearsonr
    rho_s, p_s = spearmanr(y_nm, y_cr)
    r_p, p_p = pearsonr(np.log1p(y_nm), np.log1p(y_cr))

    # Partial correlation (remove geometry)
    from sklearn.linear_model import LinearRegression
    X_g = df_b[geo_feats].values
    resid_nm = np.log1p(y_nm) - LinearRegression().fit(X_g, np.log1p(y_nm)).predict(X_g)
    resid_cr = np.log1p(y_cr) - LinearRegression().fit(X_g, np.log1p(y_cr)).predict(X_g)
    r_partial, p_partial = pearsonr(resid_nm, resid_cr)

    print(f"  Raw Spearman rho = {rho_s:.4f} (p = {p_s:.2e})")
    print(f"  Partial (after geo) r = {r_partial:.4f} (p = {p_partial:.2e})")
    print(f"  -> {'STRONG shared latent risk' if r_partial > 0.3 else 'MODERATE shared risk' if r_partial > 0.15 else 'WEAK shared risk'}")

    # Residual scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hexbin(resid_nm, resid_cr, gridsize=50, cmap="Blues", mincnt=1)
    ax.set_xlabel("NM residual (log1p per-km)")
    ax.set_ylabel("Crash residual (log1p per-km)")
    ax.set_title(f"Residual Correlation After Geometry\nPartial r = {r_partial:.4f}")
    cb = plt.colorbar(ax.collections[0], ax=ax, label="Count")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "bivariate_residuals.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")

    return {"spearman_r": rho_s, "partial_r": r_partial, "partial_p": p_partial}


# ===================================================================
# 8. BIVARIATE COPULA MODELS
# ===================================================================

def _nb_pmf(y, mu, alpha):
    """Vectorized NB PMF: P(Y = y | mu, alpha)."""
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return scipy_stats.nbinom.pmf(y, r, p)

def _nb_cdf(y, mu, alpha):
    """Vectorized NB CDF: P(Y <= y | mu, alpha)."""
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return scipy_stats.nbinom.cdf(y, r, p)

def _frank_cop(u, v, theta):
    """Frank copula C(u,v;theta). theta in (0, inf) for positive dependence."""
    eps = 1e-300
    denom = np.expm1(-theta)
    if abs(denom) < eps:
        return u * v
    return -np.log1p(np.expm1(-theta * u) * np.expm1(-theta * v) / denom) / theta

def _normal_cop(u, v, rho):
    """Gaussian copula C(u,v;rho). rho=0 → independence (u*v)."""
    if abs(rho) < 1e-8:
        return u * v
    from scipy.stats import norm, multivariate_normal
    eps = 1e-10
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    pts = np.column_stack([norm.ppf(u), norm.ppf(v)])
    return multivariate_normal.cdf(pts, mean=[0, 0], cov=[[1, rho], [rho, 1]])

def _clayton_cop(u, v, theta):
    """Clayton (Kimeldorf-Sampson) copula C(u,v;theta). theta > 0 -> positive dep."""
    if theta < 1e-6:
        return u * v
    return np.maximum(u ** (-theta) + v ** (-theta) - 1.0, 1e-300) ** (-1.0 / theta)

_COPULA_FNS    = {"frank": _frank_cop, "normal": _normal_cop, "kimeldorf": _clayton_cop}
_COPULA_BOUNDS = {"frank": (0.01, 30.0), "normal": (-0.99, 0.99), "kimeldorf": (0.001, 20.0)}
_COPULA_INIT   = {"frank": 1.0,          "normal": 0.3,           "kimeldorf": 0.5}

def _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, theta):
    """
    Bivariate discrete joint PMF via copula rectangle formula:
      P(Y1=y1, Y2=y2) = C(F1(y1),F2(y2)) - C(F1(y1-1),F2(y2))
                      - C(F1(y1),F2(y2-1)) + C(F1(y1-1),F2(y2-1))
    All inputs are length-n arrays.
    """
    p = (cop_fn(u1c, u2c, theta) - cop_fn(u1c, u2p, theta)
       - cop_fn(u1p, u2c, theta) + cop_fn(u1p, u2p, theta))
    return np.maximum(p, 1e-300)

def _rebuild_mu(df_sub, model):
    """Reconstruct predicted mean (mu) and alpha for rows in df_sub."""
    feats  = model["features"]
    params = model["params"]
    X_std  = model["scaler"].transform(df_sub[feats].values.astype(float))
    eta    = params[0] + X_std @ params[1:len(feats) + 1] + df_sub["log_len"].values
    mu     = np.exp(np.clip(eta, -20, 20))
    alpha  = float(np.exp(params[-1]))
    return mu, alpha


def fit_bivariate_copula(df, nm_col, nm_model, crash_model, copula_name="kimeldorf"):
    """
    Two-step Inference-Functions-for-Margins (IFM) bivariate NB copula.
    Marginal NB parameters are held fixed; only the copula parameter theta is
    optimised over the joint log-likelihood.

    Returns a dict with: theta, ll_joint, ll_indep, bic_joint, bic_indep,
    lr_stat, p_lr, and the aligned data arrays for downstream analysis.
    """
    all_feats = list(set(nm_model["features"]) | set(crash_model["features"]))
    df_c = df.dropna(
        subset=all_feats + [nm_col, "ObsCrashes", "log_len"]
    ).reset_index(drop=True)
    n = len(df_c)

    mu1, alpha1 = _rebuild_mu(df_c, nm_model)
    mu2, alpha2 = _rebuild_mu(df_c, crash_model)
    y1 = df_c[nm_col].values.astype(int)
    y2 = df_c["ObsCrashes"].values.astype(int)

    eps = 1e-10
    u1c = np.clip(_nb_cdf(y1,                  mu1, alpha1), eps, 1 - eps)
    u1p = np.where(y1 > 0,
                   np.clip(_nb_cdf(y1 - 1, mu1, alpha1), 0.0, 1 - eps), 0.0)
    u2c = np.clip(_nb_cdf(y2,                  mu2, alpha2), eps, 1 - eps)
    u2p = np.where(y2 > 0,
                   np.clip(_nb_cdf(y2 - 1, mu2, alpha2), 0.0, 1 - eps), 0.0)

    # Independence baseline via copula rectangle formula at theta=0
    # (NOT via _nb_pmf directly) to guarantee numerical consistency:
    # the rectangle formula recovers independence exactly at theta=0 for
    # all copulas (C(u,v;0)=u*v), so the copula-optimised LL can NEVER be
    # worse than independence.
    indep_pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p,
                                lambda u, v, t: u * v,  # theta=0 → C(u,v) = u*v
                                0.0)
    ll_indep = float(np.sum(np.log(np.maximum(indep_pmf, eps))))

    cop_fn = _COPULA_FNS[copula_name]
    lo, hi = _COPULA_BOUNDS[copula_name]
    t0     = _COPULA_INIT[copula_name]

    def neg_ll(t):
        pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, float(t[0]))
        return -float(np.sum(np.log(pmf)))

    res       = scipy_min(neg_ll, [t0], method="L-BFGS-B",
                          bounds=[(lo, hi)], options={"maxiter": 500, "ftol": 1e-9})
    theta_opt = float(res.x[0])
    ll_joint  = float(np.sum(np.log(
        _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, theta_opt)
    )))

    # BIC: independence uses k_nm + k_crash params; copula adds 1 (theta)
    k_indep  = nm_model["k"] + crash_model["k"]
    k_copula = k_indep + 1
    bic_indep  = k_indep  * np.log(n) - 2 * ll_indep
    bic_copula = k_copula * np.log(n) - 2 * ll_joint

    lr_stat = 2.0 * (ll_joint - ll_indep)
    p_lr    = float(1 - scipy_stats.chi2.cdf(lr_stat, df=1))

    return {
        "copula": copula_name, "theta": theta_opt, "n": n,
        "ll_joint": ll_joint,  "ll_indep": ll_indep,
        "bic_joint": bic_copula, "bic_indep": bic_indep,
        "k_copula": k_copula,  "k_indep": k_indep,
        "lr_stat": lr_stat, "p_lr": p_lr,
        "df_c": df_c, "y1": y1, "y2": y2,
        "mu1": mu1, "alpha1": alpha1,
        "mu2": mu2, "alpha2": alpha2,
        "u1c": u1c, "u1p": u1p, "u2c": u2c, "u2p": u2p,
    }


# ===================================================================
# 8b. JOINT COPULA — full MLE (all params simultaneously)
# ===================================================================

def fit_joint_copula_full(df, nm_col, nm_model, crash_model, copula_name="kimeldorf"):
    """Full joint MLE of bivariate NB copula model.

    Unlike IFM (two-step), β_NM, α_NM, β_crash, α_crash AND θ_copula are all
    optimised jointly from the bivariate rectangle-formula log-likelihood.
    Initialised from separately-fitted IFM marginal estimates.

    Parameter layout:
      [0 .. K1]        intercept + K1 betas for NM
      [K1+1]           log_alpha_NM
      [K1+2 .. K1+K2+2] intercept + K2 betas for Crash
      [K1+K2+3]        log_alpha_Crash
      [K1+K2+4]        theta_copula
    """
    all_feats_nm = nm_model["features"]
    all_feats_cr = crash_model["features"]
    all_feats    = list(set(all_feats_nm) | set(all_feats_cr))

    df_c = df.dropna(
        subset=all_feats + [nm_col, "ObsCrashes", "log_len"]
    ).reset_index(drop=True)
    n = len(df_c)

    y1     = df_c[nm_col].values.astype(int)
    y2     = df_c["ObsCrashes"].values.astype(int)
    offset = df_c["log_len"].values

    K1, K2 = len(all_feats_nm), len(all_feats_cr)
    n_nm, n_cr = 1 + K1, 1 + K2

    sc_nm = StandardScaler()
    Xnm   = sc_nm.fit_transform(df_c[all_feats_nm].values.astype(float))
    Xnm_f = np.column_stack([np.ones(n), Xnm])

    sc_cr = StandardScaler()
    Xcr   = sc_cr.fit_transform(df_c[all_feats_cr].values.astype(float))
    Xcr_f = np.column_stack([np.ones(n), Xcr])

    cop_fn     = _COPULA_FNS[copula_name]
    lo_t, hi_t = _COPULA_BOUNDS[copula_name]

    i_la1   = n_nm                # K1+1
    i_b2    = n_nm + 1            # K1+2
    i_la2   = i_b2 + n_cr        # K1+K2+3
    i_theta = i_la2 + 1          # K1+K2+4
    K_total = i_theta + 1        # K1+K2+5

    eps = 1e-10

    def neg_joint_ll(params):
        b1    = params[:n_nm]
        la1   = params[i_la1]
        b2    = params[i_b2:i_b2 + n_cr]
        la2   = params[i_la2]
        theta = params[i_theta]

        a1  = np.exp(np.clip(la1, -10, 10))
        a2  = np.exp(np.clip(la2, -10, 10))
        mu1 = np.exp(np.clip(Xnm_f @ b1 + offset, -20, 20))
        mu2 = np.exp(np.clip(Xcr_f @ b2 + offset, -20, 20))

        u1c = np.clip(_nb_cdf(y1,     mu1, a1), eps, 1 - eps)
        u1p = np.where(y1 > 0, np.clip(_nb_cdf(y1 - 1, mu1, a1), 0, 1 - eps), 0.0)
        u2c = np.clip(_nb_cdf(y2,     mu2, a2), eps, 1 - eps)
        u2p = np.where(y2 > 0, np.clip(_nb_cdf(y2 - 1, mu2, a2), 0, 1 - eps), 0.0)

        pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, float(theta))
        return -float(np.sum(np.log(np.maximum(pmf, 1e-300))))

    # Initialise from IFM marginal models on the same df_c subset
    nm_r = fit_model(df_c, all_feats_nm, nm_col,      "log_len", label="_jnm")
    cr_r = fit_model(df_c, all_feats_cr, "ObsCrashes", "log_len", label="_jcr")

    init = np.zeros(K_total)
    if nm_r is not None:
        init[:n_nm]           = nm_r["params"][:n_nm]
        init[i_la1]           = nm_r["params"][-1]
    else:
        init[0] = np.log(max(float(y1.mean()), 0.01))
    if cr_r is not None:
        init[i_b2:i_b2+n_cr] = cr_r["params"][:n_cr]
        init[i_la2]           = cr_r["params"][-1]
    else:
        init[i_b2] = np.log(max(float(y2.mean()), 0.01))

    try:
        _ifm = fit_bivariate_copula(df_c, nm_col,
                                    nm_r or nm_model, cr_r or crash_model,
                                    copula_name=copula_name)
        init[i_theta] = _ifm["theta"]
    except Exception:
        init[i_theta] = _COPULA_INIT[copula_name]

    bounds_list           = [(None, None)] * K_total
    bounds_list[i_theta]  = (lo_t, hi_t)

    res        = scipy_min(neg_joint_ll, init, method="L-BFGS-B",
                           bounds=bounds_list,
                           options={"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-7})
    params_opt = res.x
    ll_joint   = -neg_joint_ll(params_opt)
    theta_opt  = float(params_opt[i_theta])

    # Decode the jointly-optimised marginals so this result is self-sufficient
    # for conditional_crash_prediction() — same field names fit_bivariate_copula
    # (IFM) produces, but built from the joint (not two-step) MLE.
    b1_opt, la1_opt = params_opt[:n_nm], params_opt[i_la1]
    b2_opt, la2_opt = params_opt[i_b2:i_b2 + n_cr], params_opt[i_la2]
    alpha1_opt = float(np.exp(np.clip(la1_opt, -10, 10)))
    alpha2_opt = float(np.exp(np.clip(la2_opt, -10, 10)))
    mu1_opt    = np.exp(np.clip(Xnm_f @ b1_opt + offset, -20, 20))
    mu2_opt    = np.exp(np.clip(Xcr_f @ b2_opt + offset, -20, 20))
    u1c_opt = np.clip(_nb_cdf(y1, mu1_opt, alpha1_opt), eps, 1 - eps)
    u1p_opt = np.where(y1 > 0, np.clip(_nb_cdf(y1 - 1, mu1_opt, alpha1_opt), 0, 1 - eps), 0.0)
    u2c_opt = np.clip(_nb_cdf(y2, mu2_opt, alpha2_opt), eps, 1 - eps)
    u2p_opt = np.where(y2 > 0, np.clip(_nb_cdf(y2 - 1, mu2_opt, alpha2_opt), 0, 1 - eps), 0.0)

    # Independence baseline via the SAME rectangle formula with
    # C(u,v;0)=u*v — numerically consistent with the copula LL path.
    # This guarantees ll_joint >= ll_indep (LR >= 0) because the
    # copula optimiser can always return to theta=0.
    indep_pmf = _joint_pmf_vec(u1c_opt, u1p_opt, u2c_opt, u2p_opt,
                                lambda u, v, t: u * v, 0.0)
    ll_indep = float(np.sum(np.log(np.maximum(indep_pmf, eps))))
    k_indep  = k_nm_r + k_cr_r
    bic_indep = k_indep  * np.log(n) - 2 * ll_indep
    bic_joint = K_total  * np.log(n) - 2 * ll_joint
    lr_stat   = 2.0 * (ll_joint - ll_indep)
    df_lr     = K_total - k_indep
    p_lr      = float(1 - scipy_stats.chi2.cdf(lr_stat, df=max(df_lr, 1)))

    return {
        "copula": copula_name, "theta": theta_opt, "n": n,
        "ll_joint": ll_joint, "ll_indep": ll_marg,
        "bic_joint": bic_joint, "bic_indep": bic_indep,
        "delta_bic": bic_joint - bic_indep,
        "k_joint": K_total, "k_indep": k_indep,
        "lr_stat": lr_stat, "p_lr": p_lr, "converged": res.success,
        "params": params_opt,
        "mu1": mu1_opt, "alpha1": alpha1_opt, "mu2": mu2_opt, "alpha2": alpha2_opt,
        "u1c": u1c_opt, "u1p": u1p_opt, "u2c": u2c_opt, "u2p": u2p_opt,
        "y1": y1, "y2": y2, "df_c": df_c,
    }


def joint_copula_search(df, nm_col, nm_model, crash_model, out_dir):
    """Search over Frank / Normal / Kimeldorf with full joint MLE.

    Complements IFM analysis: jointly optimises all β and θ simultaneously.
    Saves copula_joint_comparison.csv.
    """
    print("\n  Joint MLE copula optimisation (all params simultaneously)...")
    joint_fits = {}
    for cn in ("frank", "normal", "kimeldorf"):
        print(f"    {cn:12s} ...", end=" ", flush=True)
        try:
            r = fit_joint_copula_full(df, nm_col, nm_model, crash_model, copula_name=cn)
            joint_fits[cn] = r
            print(f"θ={r['theta']:.4f}  LL={r['ll_joint']:.1f}  "
                  f"BIC={r['bic_joint']:.1f}  LR={r['lr_stat']:.1f}  "
                  f"p={r['p_lr']:.2e}  conv={r['converged']}")
        except Exception as e:
            print(f"FAILED: {e}")
            joint_fits[cn] = None

    valid = {k: v for k, v in joint_fits.items() if v is not None}
    if not valid:
        print("  All joint fits failed.")
        return {}

    best_jcn = min(valid, key=lambda k: valid[k]["bic_joint"])
    r0       = next(iter(valid.values()))
    bic_i    = r0["bic_indep"]

    print(f"\n  {'Copula':<14} {'θ':>8} {'LL':>12} {'BIC':>10} {'ΔBIC':>8} "
          f"{'LR':>8} {'p-LR':>12}")
    print(f"  {'-'*78}")
    print(f"  {'Independence':<14} {'---':>8} {r0['ll_indep']:>12.1f} "
          f"{bic_i:>10.1f} {'0':>8} {'---':>8} {'---':>12}")
    for cn, r in valid.items():
        tag = "  <-- BEST" if cn == best_jcn else ""
        print(f"  {cn:<14} {r['theta']:>8.4f} {r['ll_joint']:>12.1f} "
              f"{r['bic_joint']:>10.1f} {r['delta_bic']:>+8.1f} "
              f"{r['lr_stat']:>8.1f} {r['p_lr']:>12.2e}{tag}")

    rows = [{"method": "joint_MLE", "copula": cn,
             "theta": r["theta"], "ll": r["ll_joint"],
             "bic": r["bic_joint"], "delta_bic": r["delta_bic"],
             "lr_stat": r["lr_stat"], "p_lr": r["p_lr"],
             "k_joint": r["k_joint"], "k_indep": r["k_indep"],
             "converged": int(r["converged"])}
            for cn, r in valid.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "copula_joint_comparison.csv"), index=False
    )
    print(f"  Saved: copula_joint_comparison.csv")

    return {"fits": valid, "best_copula": best_jcn,
            "best_bic": valid[best_jcn]["bic_joint"],
            "best_theta": valid[best_jcn]["theta"],
            "bic_indep": bic_i}


def conditional_crash_prediction(fit_r, max_y2=60):
    """
    Compute E[Y_crash | Y_NM = y1, X] from the fitted bivariate copula model.

    P(Y2=k | Y1=y1) = P(Y1=y1, Y2=k) / P(Y1=y1)   (copula rectangle formula)
    E[Y2 | Y1=y1] = sum_{k=0}^{max_y2} k * P(Y2=k | Y1=y1)

    The normalising constant P(Y1=y1) = NB-PMF(y1 | mu1, alpha1).
    """
    cop_fn   = _COPULA_FNS[fit_r["copula"]]
    theta    = fit_r["theta"]
    mu1, alpha1 = fit_r["mu1"], fit_r["alpha1"]
    mu2, alpha2 = fit_r["mu2"], fit_r["alpha2"]
    u1c, u1p = fit_r["u1c"], fit_r["u1p"]
    n        = len(fit_r["y1"])
    eps      = 1e-10

    p_y1     = np.maximum(_nb_pmf(fit_r["y1"], mu1, alpha1), eps)
    E_y2     = np.zeros(n)

    for k in range(max_y2 + 1):
        k_arr  = np.full(n, k)
        u2c_k  = np.clip(_nb_cdf(k_arr,     mu2, alpha2), eps, 1 - eps)
        u2p_k  = np.where(k > 0,
                          np.clip(_nb_cdf(k_arr - 1, mu2, alpha2), 0.0, 1 - eps),
                          0.0)
        joint  = _joint_pmf_vec(u1c, u1p, u2c_k, u2p_k, cop_fn, theta)
        E_y2  += k * np.maximum(joint / p_y1, 0.0)

    return E_y2


def bivariate_copula_analysis(df, nm_col, nm_model, crash_model, out_dir):
    """
    Full bivariate copula analysis pipeline:
      1. Fit Frank / Normal / Clayton copulas and compare BIC
      2. Compute conditional E[crashes | NM, X] from the best copula
      3. Compare copula-conditional vs marginal crash predictions (r, MAE, RMSE)
      4. Produce comparison plot and CSV outputs
    """
    print("\n" + "=" * 70)
    print("  BIVARIATE COPULA ANALYSIS")
    print("  Frank / Normal / Clayton — coupled NB models + conditional prediction")
    print("=" * 70)

    # ── 1. Fit all three copulas ──────────────────────────────────────────
    print("\n  Fitting copula models (IFM two-step)...")
    fits = {}
    for cn in ("frank", "normal", "kimeldorf"):
        print(f"    {cn:12s} ...", end=" ", flush=True)
        r = fit_bivariate_copula(df, nm_col, nm_model, crash_model, copula_name=cn)
        fits[cn] = r
        print(f"θ={r['theta']:.4f}  LL={r['ll_joint']:.1f}  "
              f"BIC={r['bic_joint']:.1f}  LR={r['lr_stat']:.1f}  p={r['p_lr']:.2e}")

    # ── 2. Comparison table ───────────────────────────────────────────────
    r0        = fits["frank"]
    ll_indep  = r0["ll_indep"]
    k_indep   = r0["k_indep"]
    n         = r0["n"]
    bic_indep = k_indep * np.log(n) - 2 * ll_indep
    best_cn   = min(fits, key=lambda k: fits[k]["bic_joint"])

    print(f"\n  {'Model':<14} {'θ':>8} {'LL_joint':>12} {'BIC':>10} "
          f"{'ΔBIC':>8} {'LR':>8} {'p-LR':>12}")
    print(f"  {'-' * 80}")
    print(f"  {'Independence':<14} {'---':>8} {ll_indep:>12.1f} "
          f"{bic_indep:>10.1f} {'0':>8} {'---':>8} {'---':>12}")
    for cn in ("frank", "normal", "kimeldorf"):
        r   = fits[cn]
        tag = "  <-- BEST" if cn == best_cn else ""
        print(f"  {cn:<14} {r['theta']:>8.4f} {r['ll_joint']:>12.1f} "
              f"{r['bic_joint']:>10.1f} {r['bic_joint'] - bic_indep:>+8.1f} "
              f"{r['lr_stat']:>8.1f} {r['p_lr']:>12.2e}{tag}")

    # Save CSV
    rows = [{"copula": "independence", "theta": np.nan, "n": n, "k": k_indep,
             "ll": ll_indep, "bic": bic_indep, "delta_bic": 0.0,
             "lr_stat": np.nan, "p_lr": np.nan}]
    for cn in ("frank", "normal", "kimeldorf"):
        r = fits[cn]
        rows.append({"copula": cn, "theta": r["theta"], "n": r["n"], "k": r["k_copula"],
                     "ll": r["ll_joint"], "bic": r["bic_joint"],
                     "delta_bic": r["bic_joint"] - bic_indep,
                     "lr_stat": r["lr_stat"], "p_lr": r["p_lr"]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "copula_comparison.csv"), index=False)

    # ── 3. Conditional crash prediction from the best copula ─────────────
    best_r = fits[best_cn]
    print(f"\n  Computing E[crashes | NM, X] under {best_cn} copula "
          f"(θ={best_r['theta']:.4f}, max_y2=60)...")
    E_cond = conditional_crash_prediction(best_r, max_y2=60)
    E_marg = best_r["mu2"]
    y2_obs = best_r["y2"]

    # ── 4. Prediction comparison metrics ─────────────────────────────────
    r_marg,  _  = scipy_stats.pearsonr(E_marg, y2_obs)
    r_cond,  _  = scipy_stats.pearsonr(E_cond, y2_obs)
    rho_marg, _ = scipy_stats.spearmanr(E_marg, y2_obs)
    rho_cond, _ = scipy_stats.spearmanr(E_cond, y2_obs)
    mae_marg    = float(np.mean(np.abs(E_marg - y2_obs)))
    mae_cond    = float(np.mean(np.abs(E_cond - y2_obs)))
    rmse_marg   = float(np.sqrt(np.mean((E_marg - y2_obs) ** 2)))
    rmse_cond   = float(np.sqrt(np.mean((E_cond - y2_obs) ** 2)))

    print(f"\n  CRASH PREDICTION COMPARISON (n={n:,})")
    print(f"  {'Metric':<30} {'Marginal NB':>14} {'Copula-Conditional':>20} {'Δ':>8}")
    print(f"  {'-' * 78}")
    print(f"  {'Pearson r':<30} {r_marg:>14.4f} {r_cond:>20.4f} {r_cond - r_marg:>+8.4f}")
    print(f"  {'Spearman ρ':<30} {rho_marg:>14.4f} {rho_cond:>20.4f} {rho_cond - rho_marg:>+8.4f}")
    print(f"  {'MAE':<30} {mae_marg:>14.4f} {mae_cond:>20.4f} {mae_cond - mae_marg:>+8.4f}")
    print(f"  {'RMSE':<30} {rmse_marg:>14.4f} {rmse_cond:>20.4f} {rmse_cond - rmse_marg:>+8.4f}")

    # Save predictions
    out_pred = best_r["df_c"].copy()
    out_pred["mu_crash_marginal"]          = E_marg
    out_pred["mu_crash_copula_conditional"] = E_cond
    out_pred["nm_observed"]                 = best_r["y1"]
    out_pred.to_csv(os.path.join(out_dir, "copula_conditional_predictions.csv"), index=False)

    # ── 5. Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: marginal NB predictions
    ax = axes[0]
    ax.hexbin(E_marg, y2_obs, gridsize=40, cmap="Blues", mincnt=1)
    lim = max(float(np.percentile(E_marg, 99)), float(np.percentile(y2_obs, 99)))
    ax.plot([0, lim], [0, lim], "r--", alpha=0.6, label="1:1")
    ax.set_xlabel("Marginal E[crashes | X]")
    ax.set_ylabel("Observed crashes")
    ax.set_title(f"Marginal NB\nr = {r_marg:.3f},  MAE = {mae_marg:.3f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel B: copula-conditional predictions
    ax = axes[1]
    ax.hexbin(E_cond, y2_obs, gridsize=40, cmap="Greens", mincnt=1)
    ax.plot([0, lim], [0, lim], "r--", alpha=0.6, label="1:1")
    ax.set_xlabel(f"Copula-conditional E[crashes | NM, X]\n"
                  f"({best_cn}, θ = {best_r['theta']:.3f})")
    ax.set_ylabel("Observed crashes")
    ax.set_title(f"Copula-Conditional ({best_cn})\nr = {r_cond:.3f},  MAE = {mae_cond:.3f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel C: per-segment error improvement
    ax = axes[2]
    improvement = np.abs(E_marg - y2_obs) - np.abs(E_cond - y2_obs)
    ax.hist(improvement, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="No change")
    ax.axvline(float(improvement.mean()), color="limegreen", linestyle="--",
               label=f"Mean = {improvement.mean():+.3f}")
    ax.set_xlabel("|err_marginal| − |err_copula|  (positive = copula better)")
    ax.set_ylabel("Segment count")
    ax.set_title("Per-Segment Prediction Improvement")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Bivariate Copula Crash Prediction — {best_cn} copula  "
        f"(θ = {best_r['theta']:.3f},  LR = {best_r['lr_stat']:.1f},  "
        f"p = {best_r['p_lr']:.2e})",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "copula_prediction_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: copula_prediction_comparison.png")
    print(f"  Saved: copula_comparison.csv")
    print(f"  Saved: copula_conditional_predictions.csv")

    # ── 6. Link validity summary ──────────────────────────────────────────
    print(f"\n  LINK VALIDITY ASSESSMENT")
    best_lr_p = best_r["p_lr"]
    bic_gain  = bic_indep - best_r["bic_joint"]
    if best_lr_p < 0.001 and bic_gain > 10:
        verdict = ("STRONG — copula captures significant latent NM-crash dependence "
                   f"(LR={best_r['lr_stat']:.1f}, ΔBIC={bic_gain:.1f})")
    elif best_lr_p < 0.05:
        verdict = ("MODERATE — dependence detected but BIC gain is modest "
                   f"(LR={best_r['lr_stat']:.1f}, ΔBIC={bic_gain:.1f})")
    else:
        verdict = f"WEAK — copula LR test not significant (p={best_lr_p:.4f})"
    print(f"  Best copula: {best_cn}  θ={best_r['theta']:.4f}")
    print(f"  {verdict}")

    return {
        "fits": fits, "best_copula": best_cn,
        "bic_indep": bic_indep,
        "best_lr_stat": best_r["lr_stat"], "best_lr_p": best_r["p_lr"],
        "best_theta": best_r["theta"],
        "pred": {"r_marg": r_marg, "r_cond": r_cond,
                 "mae_marg": mae_marg, "mae_cond": mae_cond,
                 "rmse_marg": rmse_marg, "rmse_cond": rmse_cond},
    }


# ===================================================================
# 8c. COPULA STRUCTURE SEARCH — joint feature selection for NM & Crash,
#     scored against the FULL JOINT-MLE copula BIC (fit_joint_copula_full:
#     beta_NM, alpha_NM, beta_crash, alpha_crash AND theta all optimised
#     together), not the two-step IFM approximation and not each marginal
#     in isolation.
#
#     Steps 7/7b fit a copula theta on top of whichever NM/Crash feature
#     sets were already chosen (by Step 1's marginal-only SA). That answers
#     "given these two independently-chosen specs, is there leftover
#     dependence?" but never asks whether a DIFFERENT pair of specs would
#     make the two responses fit together better as a coupled system. This
#     search answers that: it perturbs the NM and Crash feature sets
#     directly against the joint-MLE copula BIC, so the two specs can trade
#     off against each other instead of being frozen from separate marginal
#     searches — and because it's the full joint MLE, the dependence
#     parameter itself feeds back into the marginal estimates too, not just
#     the reverse. Solved with scipy's differential_evolution rather than a
#     hand-rolled simulated-annealing sweep.
# ===================================================================

def differential_evolution_copula_search(
    df, candidates, nm_col, offset_col, copula_name="kimeldorf",
    init_nm_feats=None, init_crash_feats=None,
    maxiter=60, popsize=15, n_runs=3, seed=42, plot_path=None,
):
    """Jointly search NM and Crash feature subsets to minimise the FULL
    joint-MLE copula BIC (fit_joint_copula_full), using scipy's
    differential_evolution solver.

    Each candidate feature gets one continuous gene per response in [0, 1]
    (2*len(candidates) genes total); a gene > 0.5 includes that feature in
    that response's spec. This is a continuous relaxation of the underlying
    binary feature-selection problem, which is what lets DE (a
    continuous-domain solver) operate on it directly.

    Every candidate solution triggers two NB refits plus a full joint-MLE
    copula fit (itself two more NB refits for its own initialisation), so
    this is markedly more expensive per-evaluation than the old IFM-scored
    SA search — keep maxiter/popsize modest. Runs n_runs independent DE
    searches (different seeds) and keeps the global best; run 0 seeds its
    initial population from (init_nm_feats, init_crash_feats) when given
    (e.g. the MCR-searched final specs) so the search refines from a good
    starting point instead of from scratch.
    """
    print(f"\n  DE JOINT-MODEL COPULA STRUCTURE SEARCH — joint (NM, Crash) "
          f"feature selection")
    print(f"  {len(candidates)} candidates, maxiter={maxiter}, popsize={popsize}, "
          f"{n_runs} runs, copula={copula_name} (scored on full joint-MLE BIC)")

    dfs = df.dropna(subset=candidates + [nm_col, "ObsCrashes", offset_col]).reset_index(drop=True)
    af, Kf = list(candidates), len(candidates)
    n_genes = 2 * Kf

    def _decode(x):
        x = np.asarray(x)
        nm_mask = (x[:Kf] > 0.5).astype(int)
        cr_mask = (x[Kf:] > 0.5).astype(int)
        return nm_mask, cr_mask

    def _score(x):
        nm_mask, cr_mask = _decode(x)
        nm_sel = [af[i] for i in range(Kf) if nm_mask[i] == 1]
        cr_sel = [af[i] for i in range(Kf) if cr_mask[i] == 1]
        if len(nm_sel) < 2 or len(cr_sel) < 2:
            return 1e12, None
        try:
            nm_m  = fit_model(dfs, nm_sel, nm_col, offset_col, label="_dej_nm", verbose=False)
            cr_m  = fit_model(dfs, cr_sel, "ObsCrashes", offset_col, label="_dej_cr", verbose=False)
            fit_r = fit_joint_copula_full(dfs, nm_col, nm_m, cr_m, copula_name=copula_name)
            return fit_r["bic_joint"], (nm_m, cr_m, fit_r)
        except Exception:
            return 1e12, None

    def _objective(x):
        return _score(x)[0]

    seed_vec = None
    if init_nm_feats or init_crash_feats:
        seed_vec = np.full(n_genes, 0.1)
        for f in (init_nm_feats or []):
            if f in af: seed_vec[af.index(f)] = 0.9
        for f in (init_crash_feats or []):
            if f in af: seed_vec[Kf + af.index(f)] = 0.9

    global_best_score   = 1e12
    global_best_payload = None
    global_best_history = []

    for run in range(n_runs):
        run_seed = seed + run * 97
        history  = []

        def _callback(xk, convergence=None):
            history.append(_objective(xk))

        init_pop = "latinhypercube"
        if run == 0 and seed_vec is not None:
            rng0 = np.random.RandomState(run_seed)
            pop  = rng0.uniform(0.0, 1.0, size=(popsize * n_genes, n_genes))
            pop[0] = seed_vec
            init_pop = pop

        print(f"\n  [Run {run+1}/{n_runs}] seed={run_seed}  genes={n_genes}")
        res = differential_evolution(
            _objective, bounds=[(0.0, 1.0)] * n_genes,
            maxiter=maxiter, popsize=popsize, seed=run_seed,
            mutation=(0.4, 1.4), recombination=0.7,
            # See differential_evolution_feature_search() for why tol=0 is
            # needed -- without it this also stopped after 1-2 generations.
            tol=0.0,
            init=init_pop, polish=False, callback=_callback,
        )
        final_score, final_payload = _score(res.x)
        history.append(final_score)
        best_so_far = np.minimum.accumulate(history)

        print(f"  [Run {run+1}] best joint BIC={best_so_far[-1]:.2f}  nfev={res.nfev}")

        if best_so_far[-1] < global_best_score and final_payload is not None:
            global_best_score   = float(best_so_far[-1])
            global_best_payload = final_payload
        global_best_history.extend(best_so_far.tolist())

    if global_best_payload is None:
        print("  DE joint-model copula search failed to find a valid solution.")
        return None

    nm_model_best, crash_model_best, fit_best = global_best_payload
    nm_feats_best    = nm_model_best["features"]
    crash_feats_best = crash_model_best["features"]

    print(f"\n  BEST DE JOINT-MODEL STRUCTURE: joint BIC={global_best_score:.2f}  "
          f"theta={fit_best['theta']:.4f}  copula={copula_name}")
    print(f"  NM features    ({len(nm_feats_best)}): {[_V(f) for f in nm_feats_best]}")
    print(f"  Crash features ({len(crash_feats_best)}): {[_V(f) for f in crash_feats_best]}")

    # ── Final model coefficient tables ────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  FINAL JOINT MODEL — NM marginal ({copula_name} copula)")
    print(f"  {'─' * 60}")
    print(nm_model_best["coef_table"].to_string(index=False))
    print(f"\n  NM: LL={nm_model_best['ll']:.2f}  BIC={nm_model_best['bic']:.2f}  "
          f"k={nm_model_best['k']}  dispersion(alpha)={nm_model_best['alpha']:.4f}")

    print(f"\n  {'─' * 60}")
    print(f"  FINAL JOINT MODEL — Crash marginal ({copula_name} copula)")
    print(f"  {'─' * 60}")
    print(crash_model_best["coef_table"].to_string(index=False))
    print(f"\n  Crash: LL={crash_model_best['ll']:.2f}  BIC={crash_model_best['bic']:.2f}  "
          f"k={crash_model_best['k']}  dispersion(alpha)={crash_model_best['alpha']:.4f}")
    print(f"  Copula theta={fit_best['theta']:.4f}  "
          f"Joint-BIC={fit_best['bic_joint']:.2f}  "
          f"Indep-BIC={fit_best['bic_indep']:.2f}  "
          f"dBIC={fit_best['bic_indep'] - fit_best['bic_joint']:+.1f}")
    print(f"  {'─' * 60}\n")

    if plot_path:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(global_best_history) + 1), global_best_history,
                "darkorange", linewidth=1.2, alpha=0.8)
        ax.axhline(y=min(global_best_history), color="red", linestyle="--", alpha=0.5,
                   label=f"Best = {min(global_best_history):.2f}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Joint-MLE copula BIC (best so far)")
        ax.set_title("Differential Evolution Joint-Model Search Convergence\n"
                     f"(popsize={popsize}, {n_runs} runs, {copula_name} copula)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Convergence plot saved: {plot_path}")

    return {
        "nm_features": nm_feats_best, "crash_features": crash_feats_best,
        "nm_model": nm_model_best, "crash_model": crash_model_best,
        "fit": fit_best, "bic_joint": global_best_score,
        "history": global_best_history,
    }


# ===================================================================
# 8b. RANDOM PARAMETERS NB
# ===================================================================
def fit_random_params_nb(df, fixed_feats, rdm_feats, y_col, offset_col, label="RP-NB"):
    """Estimate a random-parameters NB where rdm_feats have normally distributed
    coefficients across segments (mean + std deviation estimated).

    Uses build_model_from_manual_spec with rdm_terms populated.
    Compares LL against fixed-parameters version via LR test.
    """
    print(f"\n  Fitting RP-NB [{label}] — {len(rdm_feats)} random, {len(fixed_feats)} fixed")
    df_m = df.dropna(subset=fixed_feats + rdm_feats + [y_col, offset_col]).reset_index(drop=True)

    all_feats  = fixed_feats + [f for f in rdm_feats if f not in fixed_feats]
    scaler     = StandardScaler()
    Xs         = scaler.fit_transform(df_m[all_feats].values.astype(float))
    for i, f in enumerate(all_feats):
        df_m[f"__f{i}__"] = Xs[:, i]

    n_fixed = len(fixed_feats)
    n_rdm   = len([f for f in rdm_feats if f not in fixed_feats]) + \
              len([f for f in rdm_feats if f in fixed_feats])
    fstd_fixed = [f"__f{i}__" for i in range(len(all_feats)) if all_feats[i] in fixed_feats]
    fstd_rdm   = [f"__f{i}__" for i in range(len(all_feats)) if all_feats[i] in rdm_feats]

    df_m["_id"]  = np.arange(len(df_m))
    df_m["_off"] = df_m[offset_col].values.astype(float)

    # parse_manual_spec expects rdm_terms as "varname:distribution" strings
    manual_spec = {
        "fixed_terms": fstd_fixed,
        "rdm_terms": [f"{c}:normal" for c in fstd_rdm],
        "rdm_cor_terms": [],
        "grouped_terms": [], "hetro_in_means": [],
        "transformations": ["no"] * len(all_feats),
        "dispersion": 1, "latent_classes": 1,
    }

    try:
        data, spec = build_model_from_manual_spec(
            df_m, manual_spec, id_col="_id", y_col=y_col, offset_col="_off"
        )
        spec = replace(spec, model="nb")
        pi = build_param_index(spec); K = pi["total_params"]

        init = np.zeros(K)
        init[0] = np.log(max(df_m[y_col].mean(), 0.01))
        # small positive init for std-dev parameters
        if "rdm_stds" in pi:
            for idx in pi["rdm_stds"]:
                init[idx] = 0.1

        def obj(p):
            return float(mixed_model_loglik(jnp.array(p), data, spec, indivi=False))

        res    = scipy_min(obj, init, method="L-BFGS-B", options={"maxiter": 8000})
        params = res.x; n = len(df_m)
        ll     = -obj(params); aic = 2*K - 2*ll; bic = K*np.log(n) - 2*ll
        se     = _scipy_hessian_se(obj, params)
        se     = np.where(np.abs(se) > 50, np.nan, se)
        z      = np.where(np.isfinite(se) & (se > 1e-12), params/se, np.nan)
        p_vals = np.where(np.isfinite(z), 2*(1-scipy_stats.norm.cdf(np.abs(z))), np.nan)

        print(f"  RP-NB: LL={ll:.2f}, AIC={aic:.2f}, BIC={bic:.2f}, k={K}, "
              f"converged={res.success}")
        return {
            "label": label, "ll": ll, "aic": aic, "bic": bic, "k": K, "n": n,
            "converged": res.success, "params": params, "se": se,
            "z": z, "p": p_vals, "fixed_feats": fixed_feats, "rdm_feats": rdm_feats,
        }
    except Exception as e:
        import traceback
        print(f"  RP-NB FAILED: {e}")
        traceback.print_exc()
        return None


def random_params_analysis(df, nm_col, sel_nm_feats, sel_crash_feats, full_nm, full_crash, out_dir,
                           mcr_nm=None, mcr_crash=None):
    """Run random-parameters NB for both responses.

    Preferred strategy: use the fixed/random-independent/random-correlated
    role split already found by the MCR structure search (mcr_nm/mcr_crash),
    since that role assignment was actually searched rather than guessed.
    Falls back to the old heuristic (treat the 3-4 largest-|coef| features
    as random) only when no MCR result is available for that response.
    Compare LL via LR test (df = n_rdm, chi²).
    """
    print("\n" + "="*70)
    print("  RANDOM PARAMETERS NB ANALYSIS")
    print("="*70)

    results = {}
    for y_col, sel_feats, full_model, tag, mcr_result in [
        (nm_col,      sel_nm_feats,    full_nm,    "NM",    mcr_nm),
        ("ObsCrashes", sel_crash_feats, full_crash, "Crash", mcr_crash),
    ]:
        rdm = fixed = None
        if mcr_result is not None:
            rdm_candidates = sorted(set(
                [r.split(":")[0] for r in mcr_result.get("rdm_i_orig", [])]
                + [r.split(":")[0] for r in mcr_result.get("rdm_c_orig", [])]
            ))
            fixed_candidates = [f for f in mcr_result.get("fixed_orig", []) if f not in rdm_candidates]
            if rdm_candidates and fixed_candidates:
                rdm, fixed = rdm_candidates, fixed_candidates
                print(f"  [{tag}] Using MCR-searched roles: "
                      f"{len(fixed)} fixed, {len(rdm)} random")

        if rdm is None:
            # Fallback: MCR result unavailable/degenerate for this response —
            # pick top-k features by |coef| from the full model as random candidates
            # (exclude intercept and dispersion row).
            print(f"  [{tag}] MCR roles unavailable; falling back to top-4-|coef| heuristic")
            feat_rows = full_model["coef_table"].copy()
            if feat_rows["Variable"].iloc[-1].startswith("ln("):
                feat_rows = feat_rows.iloc[:-1]
            feat_rows  = feat_rows[feat_rows["Variable"].isin([_V(f) for f in sel_feats])]
            feat_rows  = feat_rows.reindex(feat_rows["Estimate"].abs().sort_values(ascending=False).index)
            top_feats_labels = feat_rows["Variable"].head(4).tolist()
            rdm = [f for f in sel_feats if _V(f) in top_feats_labels]
            fixed = [f for f in sel_feats if f not in rdm]

        if len(rdm) < 1 or len(fixed) < 1:
            print(f"  [{tag}] Insufficient features for RP-NB, skipping.")
            continue

        rp = fit_random_params_nb(df, fixed, rdm, y_col, "log_len", label=f"RP-NB-{tag}")
        if rp is None:
            results[tag] = None
            continue

        # LR test vs fixed-parameters baseline model
        ll_fixed = full_model["ll"]
        k_fixed  = full_model["k"]
        lr_stat  = 2 * (rp["ll"] - ll_fixed)
        df_lr    = rp["k"] - k_fixed  # extra std-dev params
        p_lr     = scipy_stats.chi2.sf(lr_stat, df=max(df_lr, 1))
        delta_bic = rp["bic"] - full_model["bic"]

        print(f"\n  [{tag}] LR test (RP-NB vs fixed): LR={lr_stat:.2f}, "
              f"df={df_lr}, p={p_lr:.4f}")
        print(f"  [{tag}] ΔBIC (RP-NB vs fixed): {delta_bic:+.2f}")
        if p_lr < 0.05 and delta_bic < 0:
            print(f"  [{tag}] ✓ Random parameters significantly improve fit (heterogeneity confirmed)")
        else:
            print(f"  [{tag}] ✗ Random parameters do NOT significantly improve fit")

        results[tag] = {**rp, "lr_stat": lr_stat, "df_lr": df_lr, "p_lr": p_lr,
                        "delta_bic": delta_bic, "rdm_feats": rdm}

    # Save comparison table
    rows = []
    for tag, r in results.items():
        if r:
            rows.append({"Model": f"RP-NB ({tag})", "LL": r["ll"], "BIC": r["bic"],
                         "k": r["k"], "LR_vs_fixed": r["lr_stat"],
                         "p_lr": r["p_lr"], "delta_bic": r["delta_bic"],
                         "rdm_feats": ";".join(r["rdm_feats"])})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "rp_nb_comparison.csv"), index=False)
        print(f"\n  Saved: rp_nb_comparison.csv")

    return results


# ===================================================================
# 9b. MCR STRUCTURE SEARCH — fixed / random-indep / random-corr roles
# ===================================================================

def mcr_model_search(df, y_col, feats_subset, offset_col,
                     label="MCR-DE", n_starts=2, max_iter=500,
                     R=5, max_latent_classes=1, out_dir=None,
                     N_max=1500, max_time_per_start=1800,
                     solver="de", pop_size=30):
    """Search over parameter roles (fixed / random-indep / random-corr) for a
    given feature set using metacountregressor StructureEvaluatorLC.

    solver='de' uses NSGA2Engine + AdaptiveDE (differential evolution).
    solver='sa' uses AdvancedSimulatedAnnealing (legacy).
    
    Data is subsampled to N_max rows for tractable JAX compilation cost.
    Each start is capped at max_time_per_start seconds.
    """
    try:
        from metacountregressor.experiment_package import StructureEvaluatorLC
        from metacountregressor import Solvers_METAJAX as _sol
    except ImportError as e:
        print(f"  mcr_model_search: import failed ({e}); skipping.")
        return None

    # Re-patch generate_neighbor for SA path (when used)
    def _fixed_generate_neighbor(self, solution, T=None, max_attempts=20, min_active=2):
        for _ in range(max_attempts):
            neighbor = solution.copy()
            if T is not None and self.T0 is not None:
                mut_rate = self.mutation_rate * (T / self.T0)
            else:
                mut_rate = self.mutation_rate
            mut_rate = float(np.clip(mut_rate, 0.0, 1.0))
            n_changes = np.random.randint(self.min_changes, self.max_changes + 1)
            indices   = list(np.random.choice(self.dim, size=n_changes, replace=False))
            D = self.dim_core
            if getattr(self.evaluator, "max_latent_classes", 1) > 1:
                mask_start   = 2 * D + 2
                mask_end     = 3 * D + 1
                mask_indices = list(range(mask_start, min(mask_end + 1, self.dim)))
                if mask_indices and not any(i in indices for i in mask_indices):
                    indices[np.random.randint(len(indices))] = int(np.random.choice(mask_indices))
                if mask_indices and np.random.rand() < 0.8:
                    extra = int(np.random.choice(mask_indices))
                    if extra not in indices:
                        indices.append(extra)
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
                    else:
                        tail = idx - 2 * D
                        if tail == 0:
                            neighbor[idx] = 1 - int(neighbor[idx])
                        elif tail == 1:
                            max_code = getattr(self.evaluator, "max_latent_classes", 1) - 1
                            neighbor[idx] = int(np.clip(int(neighbor[idx]) + np.random.choice([-1, 1]), 0, max_code))
                        else:
                            old = int(neighbor[idx])
                            neighbor[idx] = int(np.random.choice([v for v in (0, 1, 2) if v != old]))
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
        neighbor = solution.copy()
        idx = np.random.randint(0, self.dim_core)
        neighbor[idx] = self.sample_allowed_role(idx)
        return neighbor

    _sol.AdvancedSimulatedAnnealing.generate_neighbor = _fixed_generate_neighbor

    print(f"\n  MCR STRUCTURE SEARCH [{label}] — y = {y_col}  solver={solver.upper()}")
    print(f"  {len(feats_subset)} features, roles={{0,1,2,3}}, "
          f"R={R}, max_iter={max_iter}×{n_starts}, max_LC={max_latent_classes}, "
          f"N_max={N_max}, time_cap={max_time_per_start}s")

    df_m = df.dropna(subset=feats_subset + [y_col, offset_col]).reset_index(drop=True).copy()

    if len(df_m) > N_max:
        df_m = df_m.sample(n=N_max, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {N_max} rows (from {len(df.dropna(subset=feats_subset + [y_col, offset_col]))})")

    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(df_m[feats_subset].values.astype(float))
    std_cols = [f"__x{i}__" for i in range(len(feats_subset))]
    for i, c in enumerate(std_cols):
        df_m[c] = Xs[:, i]
    df_m["_id"] = np.arange(len(df_m))

    allowed_roles = {c: [0, 1, 2, 3] for c in std_cols}
    allowed_dists = {c: ["normal"]   for c in std_cols}

    try:
        evaluator = StructureEvaluatorLC(
            df_m, "_id", y_col, std_cols,
            allowed_roles, allowed_dists,
            mode="single",
            offset_col=offset_col,
            R=R,
            max_latent_classes=max_latent_classes,
        )
    except Exception as e:
        print(f"  StructureEvaluatorLC init failed: {e}")
        return None

    D   = len(std_cols)
    dim = 3 * D + 2 if max_latent_classes > 1 else 2 * D + 2

    best_sol = None
    best_bic = 1e12

    for start in range(n_starts):
        try:
            import gc
            jax.clear_caches()
            gc.collect()
        except Exception:
            pass

        np.random.seed(42 + start * 13)
        try:
            if solver == "de":
                operator = _sol.AdaptiveDE(F=0.5, CR=0.7)
                ng = _sol.NSGA2Engine(
                    evaluator=evaluator,
                    operator=operator,
                    dimension=dim,
                    pop_size=pop_size,
                    max_iter=max_iter,
                    n_jobs=1,
                    save_history=True,
                )
                archive, scores = ng.optimize()
                # Single-objective NSGA2Engine.optimize() returns a single
                # (best_individual, best_score) pair, not population-sized
                # arrays like AdvancedSimulatedAnnealing.optimize() does.
                # Normalise shapes so the argmin/indexing below works for both.
                archive = np.atleast_2d(archive)
                scores  = np.atleast_1d(scores)
            else:
                AdvSA = _sol.AdvancedSimulatedAnnealing
                sa = AdvSA(
                    evaluator=evaluator,
                    dimension=dim,
                    max_iter=max_iter,
                    max_time=max_time_per_start,
                    alpha=0.97,
                    mutation_rate=0.35,
                    min_changes=1,
                    max_changes=3,
                )
                archive, scores = sa.optimize()
        except Exception as e:
            print(f"  [Start {start+1}] solver failed: {e}")
            import traceback; traceback.print_exc()
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
        print(f"  MCR: build_spec returned None for best solution [{label}]")
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
    lc         = best_spec.get("latent_classes", 1)
    nb         = bool(best_spec.get("dispersion", 1))

    print(f"\n  BEST MCR [{label}]: BIC = {best_bic:.2f}")
    print(f"  Fixed params:     {[_V(f) for f in fixed_orig]}")
    rdm_labels = [_V(r.split(':')[0]) for r in rdm_i_orig]
    cor_labels = [_V(r.split(':')[0]) for r in rdm_c_orig]
    print(f"  Rnd-Independent:  {rdm_labels}")
    print(f"  Rnd-Correlated:   {cor_labels}")
    print(f"  Latent classes:   {lc}")
    print(f"  NB dispersion:    {nb}")

    if out_dir:
        pd.DataFrame([{
            "label":          label,
            "bic":            best_bic,
            "fixed_terms":    ";".join(fixed_orig),
            "rdm_ind_terms":  ";".join(rdm_i_orig),
            "rdm_cor_terms":  ";".join(rdm_c_orig),
            "latent_classes": lc,
            "nb":             int(nb),
        }]).to_csv(os.path.join(out_dir, f"mcr_search_{label}.csv"), index=False)
        print(f"  Saved: mcr_search_{label}.csv")

    return {
        "spec": best_spec, "bic": best_bic,
        "fixed_orig": fixed_orig, "rdm_i_orig": rdm_i_orig,
        "rdm_c_orig": rdm_c_orig, "lc": lc, "nb": nb,
    }


# ===================================================================
# 10. MAIN
# ===================================================================
def main():
    print("\n" + "="*70)
    print("  PROXY VARIABLE ANALYSIS")
    print("  Can Near-Miss Events Serve as a Crash Surrogate?")
    print("="*70)

    df, GEO, NM_RESP = load_data()
    nm_col = "nm_total"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # STEP 1: DIFFERENTIAL EVOLUTION (coarse feature pre-filter)
    # ============================================================
    # Narrows the 30-feature GEO pool down to a tractable candidate set for
    # each response before the more expensive MCR role search below.
    print("\n\n" + "#"*70)
    print("#  STEP 1: DIFFERENTIAL EVOLUTION — Best geometry for NM vs crashes")
    print("#  (coarse pre-filter; STEP 2 below does the real role search)")
    print("#"*70)

    hs_nm, hs_nm_feats, hs_nm_hist = differential_evolution_feature_search(
        df, GEO, nm_col, "log_len",
        label="NM_response", p_threshold=0.10, penalty_per_insig=10.0,
        maxiter=40, popsize=10, seed=42,
        plot_path=os.path.join(OUTPUT_DIR, "de_convergence_nm.png")
    )
    hs_crash, hs_crash_feats, hs_crash_hist = differential_evolution_feature_search(
        df, GEO, "ObsCrashes", "log_len",
        label="Crash_response", p_threshold=0.10, penalty_per_insig=10.0,
        maxiter=40, popsize=10, seed=43,
        plot_path=os.path.join(OUTPUT_DIR, "de_convergence_crash.png")
    )

    # Overall convergence plot (both on one)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, len(hs_nm_hist)+1), hs_nm_hist, "darkorange", linewidth=0.8, alpha=0.8)
    ax1.axhline(y=min(hs_nm_hist), color="red", linestyle="--", alpha=0.5)
    ax1.set_title("DE: NM Response\n(penalised BIC best so far)")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Penalised BIC"); ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(hs_crash_hist)+1), hs_crash_hist, "darkorange", linewidth=0.8, alpha=0.8)
    ax2.axhline(y=min(hs_crash_hist), color="red", linestyle="--", alpha=0.5)
    ax2.set_title("DE: Crash Response\n(penalised BIC best so far)")
    ax2.set_xlabel("Generation"); ax2.set_ylabel("Penalised BIC"); ax2.grid(True, alpha=0.3)

    fig.suptitle("Differential Evolution Convergence (NB, popsize=10×40 gen, significance-penalised)", fontsize=13)
    fig.tight_layout()
    de_plot = os.path.join(OUTPUT_DIR, "de_convergence_combined.png")
    fig.savefig(de_plot, dpi=150)
    plt.close(fig)

    # Save the DE models as LaTeX tables
    model_to_latex(hs_nm,
                   os.path.join(OUTPUT_DIR, "de_best_nm.tex"),
                   "Differential Evolution Best Model: Near-Miss Response",
                   "tab:de_nm")
    model_to_latex(hs_crash,
                   os.path.join(OUTPUT_DIR, "de_best_crash.tex"),
                   "Differential Evolution Best Model: Crash Response",
                   "tab:de_crash")

    # ============================================================
    # STEP 2: MCR STRUCTURE SEARCH — roles: fixed / rdm-ind / rdm-corr
    # ============================================================
    # This is the rigorous search (searches parameter ROLES, not just
    # in/out), so it now runs before — not after — the tests that need a
    # "best spec": proportionality, transfer, encompassing, bivariate risk,
    # copula, and RP-NB all consume its output below instead of Step 1's
    # cruder fixed-effects-only selection.
    print("\n\n" + "#"*70)
    print("#  STEP 2: MCR STRUCTURE SEARCH — roles: fixed / rdm-ind / rdm-corr")
    print("#  Uses metacountregressor StructureEvaluatorLC + NSGA2Engine DE")
    print("#"*70)

    mcr_nm = mcr_model_search(
        df, nm_col, hs_nm_feats, "log_len",
        label="NM", n_starts=2, max_iter=500, R=5,
        max_latent_classes=1, out_dir=OUTPUT_DIR,
        N_max=1500, solver="de", pop_size=20,
    )
    mcr_crash = mcr_model_search(
        df, "ObsCrashes", hs_crash_feats, "log_len",
        label="Crash", n_starts=2, max_iter=500, R=5,
        max_latent_classes=1, out_dir=OUTPUT_DIR,
        N_max=1500, solver="de", pop_size=20,
    )

    def _mcr_final_feats(mcr_result, fallback_feats):
        """Union of MCR's fixed + random-independent + random-correlated
        variable names, falling back to the Step 1 feature set when the
        MCR search failed or returned a degenerate (<2 feature) result."""
        if mcr_result is None:
            return list(fallback_feats)
        feats = sorted(set(
            mcr_result.get("fixed_orig", [])
            + [r.split(":")[0] for r in mcr_result.get("rdm_i_orig", [])]
            + [r.split(":")[0] for r in mcr_result.get("rdm_c_orig", [])]
        ))
        return feats if len(feats) >= 2 else list(fallback_feats)

    nm_final_feats    = _mcr_final_feats(mcr_nm, hs_nm_feats)
    crash_final_feats = _mcr_final_feats(mcr_crash, hs_crash_feats)
    print(f"\n  Final NM spec    ({len(nm_final_feats)} vars): {[_V(f) for f in nm_final_feats]}")
    print(f"  Final Crash spec ({len(crash_final_feats)} vars): {[_V(f) for f in crash_final_feats]}")

    overlap = set(nm_final_feats) & set(crash_final_feats)
    jaccard = len(overlap) / len(set(nm_final_feats) | set(crash_final_feats))
    print(f"\n  Feature overlap (final specs): {len(overlap)}/"
          f"{len(set(nm_final_feats) | set(crash_final_feats))} (Jaccard = {jaccard:.4f})")

    # Fit the two response models on their final (searched) feature sets —
    # everything from here on uses these, not the Step 1 pre-filter models.
    spec_nm_model    = fit_model(df, nm_final_feats, nm_col, "log_len",
                                 label="Final: NM (searched spec)", verbose=False)
    spec_crash_model = fit_model(df, crash_final_feats, "ObsCrashes", "log_len",
                                 label="Final: Crash (searched spec)", verbose=False)
    print(f"  Final NM model:    LL={spec_nm_model['ll']:.2f}, BIC={spec_nm_model['bic']:.2f}, "
          f"k={spec_nm_model['k']}")

    # Export the final (DE pre-filter + MCR-condensed) specs to LaTeX. These
    # are the models the rest of the pipeline (Steps 5-10) actually builds
    # on, so they -- not the Step-1 DE pre-filter -- are what the paper's
    # "best model" tables should show.
    model_to_latex(spec_nm_model,
                   os.path.join(OUTPUT_DIR, "final_spec_nm.tex"),
                   "MCR-Searched Final Model: Near-Miss Response",
                   "tab:final_nm")
    model_to_latex(spec_crash_model,
                   os.path.join(OUTPUT_DIR, "final_spec_crash.tex"),
                   "MCR-Searched Final Model: Crash Response",
                   "tab:final_crash")
    print(f"  Final Crash model: LL={spec_crash_model['ll']:.2f}, BIC={spec_crash_model['bic']:.2f}, "
          f"k={spec_crash_model['k']}")

    # ============================================================
    # STEP 3: FULL MODELS (all geometry) — internal baseline only, not
    # exported to LaTeX. Kept on the FULL 30-feature GEO set (not the
    # searched final specs) specifically for the proportionality test
    # below: that test compares coefficients on shared variables, which
    # needs both models fit on a common variable set to be an
    # apples-to-apples comparison. The full 30-variable coefficient tables
    # themselves are not shown in the paper (only their BIC/LL are cited),
    # so no model_to_latex() call here.
    # ============================================================
    print("\n\n" + "#"*70)
    print("#  STEP 3: Full models — same geometry, two responses (baseline)")
    print("#"*70)

    full_nm = fit_model(df, GEO, nm_col, "log_len",
                        label="Full: Geometry -> Near-Miss", verbose=False)
    full_crash = fit_model(df, GEO, "ObsCrashes", "log_len",
                           label="Full: Geometry -> Crashes", verbose=False)
    print(f"  NM full model:   LL={full_nm['ll']:.2f}, BIC={full_nm['bic']:.2f}, k={full_nm['k']}")
    print(f"  Crash full model: LL={full_crash['ll']:.2f}, BIC={full_crash['bic']:.2f}, k={full_crash['k']}")

    # ============================================================
    # STEP 4: COEFFICIENT PROPORTIONALITY (uses the full-geometry baseline)
    # ============================================================
    prop = coefficient_proportionality_test(full_nm, full_crash, OUTPUT_DIR)

    # ============================================================
    # STEP 5: SPECIFICATION TRANSFER (uses MCR-searched final specs)
    # ============================================================
    transfer = specification_transfer(df, nm_final_feats, crash_final_feats, OUTPUT_DIR)

    # ============================================================
    # STEP 6: ENCOMPASSING TEST (uses MCR-searched final specs)
    # ============================================================
    enc = encompassing_test(df, nm_final_feats, crash_final_feats, OUTPUT_DIR)

    # ============================================================
    # STEP 7: BIVARIATE LATENT RISK (controls for the union of both
    # final specs, rather than the full unselected 30-feature pool)
    # ============================================================
    geo_control_feats = sorted(set(nm_final_feats) | set(crash_final_feats))
    biv = bivariate_latent_risk(df, geo_control_feats, nm_col, OUTPUT_DIR)

    # ============================================================
    # STEP 8: BIVARIATE COPULA MODELS (on the MCR-searched final specs)
    # ============================================================
    print("\n\n" + "#"*70)
    print("#  STEP 8: BIVARIATE COPULA — Frank / Normal / Clayton")
    print("#  Coupled NB models + E[crashes | NM, X] conditional prediction")
    print("#"*70)

    cop = bivariate_copula_analysis(df, nm_col, spec_nm_model, spec_crash_model, OUTPUT_DIR)

    # Joint MLE copula search (all params simultaneously)
    print("\n\n" + "#"*70)
    print("#  STEP 8b: JOINT COPULA MLE — Frank / Normal / Kimeldorf")
    print("#  All β, α, θ optimised simultaneously (vs IFM two-step above)")
    print("#"*70)

    joint_cop = joint_copula_search(df, nm_col, spec_nm_model, spec_crash_model, OUTPUT_DIR)

    # ============================================================
    # STEP 9: RANDOM PARAMETERS NB (uses MCR's searched roles directly,
    # instead of guessing which features are random by |coef| size)
    # ============================================================
    print("\n\n" + "#"*70)
    print("#  STEP 9: RANDOM PARAMETERS NB — heterogeneity in coefficients")
    print("#"*70)

    rp_results = random_params_analysis(
        df, nm_col, nm_final_feats, crash_final_feats, spec_nm_model, spec_crash_model,
        OUTPUT_DIR, mcr_nm=mcr_nm, mcr_crash=mcr_crash,
    )

    # ============================================================
    # STEP 10: DE JOINT-MODEL COPULA STRUCTURE SEARCH — the definitive
    # joint model. Steps 8/8b fit a copula theta on top of feature sets
    # chosen by separately optimising each marginal (Step 8b's "joint MLE"
    # still uses the Step-2 MCR feature sets, not a search of its own).
    # This step instead perturbs the NM and Crash feature sets DIRECTLY
    # against the FULL joint-MLE copula BIC (fit_joint_copula_full: all of
    # beta_NM, alpha_NM, beta_crash, alpha_crash, theta optimised together),
    # so the two specs can trade off against each other and against the
    # dependence parameter — answering "what's the best COUPLED model"
    # rather than "given these two independently-best marginals, is there
    # leftover dependence".
    # ============================================================
    print("\n\n" + "#"*70)
    print("#  STEP 10: DE JOINT-MODEL COPULA STRUCTURE SEARCH")
    print("#  Searches both specs together against the full joint-MLE copula BIC")
    print("#"*70)

    cs_output_csv = os.path.join(OUTPUT_DIR, "copula_search_best.csv")
    cs_convergence_plot = os.path.join(OUTPUT_DIR, "copula_search_convergence.png")

    # ── Resume support: skip if previous run already saved best csv ──
    copula_search = None
    if os.path.exists(cs_output_csv):
        print(f"  STEP 10 already completed; loading from {cs_output_csv}")
        try:
            cs_df = pd.read_csv(cs_output_csv)
            if len(cs_df) > 0:
                row = cs_df.iloc[0]
                # Reconstruct a minimal dict so evidence logic works downstream
                copula_search = {
                    "nm_features":    row["nm_features"].split(";"),
                    "crash_features": row["crash_features"].split(";"),
                    "fit": {
                        "theta":    row["theta"],
                        "bic_joint": row["bic_joint"],
                        "bic_indep": row["bic_indep"],
                        "lr_stat":  row.get("lr_stat", float("nan")),
                        "p_lr":     row.get("p_lr", float("nan")),
                    },
                }
                print(f"  Loaded: theta={row['theta']:.4f}, "
                      f"BIC={row['bic_joint']:.2f}, "
                      f"dBIC={row['bic_indep'] - row['bic_joint']:+.1f}")
        except Exception as e:
            print(f"  Failed to load {cs_output_csv}: {e}")
            copula_search = None

    if copula_search is None:
        copula_search_candidates = sorted(
            set(hs_nm_feats) | set(hs_crash_feats) | set(nm_final_feats) | set(crash_final_feats)
        )
        print(f"  {len(copula_search_candidates)} candidates. "
              f"maxiter=40, popsize=10, n_runs=2 "
              f"(note: runs ~2-4 hours; increase walltime if needed)")
        copula_search = differential_evolution_copula_search(
            df, copula_search_candidates, nm_col, "log_len",
            copula_name=cop["best_copula"],
            init_nm_feats=nm_final_feats, init_crash_feats=crash_final_feats,
            maxiter=40, popsize=10, n_runs=2,
            plot_path=cs_convergence_plot,
        )

    if copula_search is not None:
        cs_fit = copula_search["fit"]
        cs_bic_indep = cs_fit["bic_indep"]
        cs_bic_gain  = cs_bic_indep - cs_fit["bic_joint"]
        E_cond_cs = conditional_crash_prediction(cs_fit, max_y2=60)
        r_cond_cs, _ = scipy_stats.pearsonr(E_cond_cs, cs_fit["y2"])
        r_marg_cs, _ = scipy_stats.pearsonr(cs_fit["mu2"], cs_fit["y2"])
        print(f"\n  Joint-searched copula: theta={cs_fit['theta']:.4f}  "
              f"LR={cs_fit['lr_stat']:.1f}  p={cs_fit['p_lr']:.2e}  "
              f"dBIC vs independence={cs_bic_gain:+.1f}")
        print(f"  Conditional-prediction r: marginal={r_marg_cs:.4f}  "
              f"copula-conditional={r_cond_cs:.4f}  delta={r_cond_cs - r_marg_cs:+.4f}")
        pd.DataFrame([{
            "nm_features": ";".join(copula_search["nm_features"]),
            "crash_features": ";".join(copula_search["crash_features"]),
            "copula": cop["best_copula"], "theta": cs_fit["theta"],
            "bic_joint": cs_fit["bic_joint"], "bic_indep": cs_bic_indep,
            "delta_bic": cs_bic_gain, "lr_stat": cs_fit["lr_stat"], "p_lr": cs_fit["p_lr"],
            "r_marginal": r_marg_cs, "r_conditional": r_cond_cs,
        }]).to_csv(os.path.join(OUTPUT_DIR, "copula_search_best.csv"), index=False)
        print(f"  Saved: copula_search_best.csv")

    # ============================================================
    # SYNTHESIS
    # ============================================================
    print("\n\n" + "="*70)
    print("  SYNTHESIS: Is Near-Miss a Valid Crash Proxy?")
    print("="*70)

    evidence = []
    if prop["corr"] > 0.7 and prop["R2_prop"] > 0.5:
        evidence.append(("PRO", f"Coefficients proportional: r={prop['corr']:.3f}, R2={prop['R2_prop']:.3f}"))
    else:
        evidence.append(("CON", f"Coefficients NOT proportional: r={prop['corr']:.3f}, R2={prop['R2_prop']:.3f}"))

    if jaccard > 0.5:
        evidence.append(("PRO", f"High feature overlap: Jaccard={jaccard:.3f}"))
    else:
        evidence.append(("CON", f"Low feature overlap: Jaccard={jaccard:.3f} — NM and crashes have different drivers"))

    penalty_cr = transfer["NM-spec -> Crashes"]["bic"] - transfer["Crash-spec -> Crashes (native)"]["bic"]
    if abs(penalty_cr) < 20:
        evidence.append(("PRO", f"NM spec transfers to crashes: dBIC={penalty_cr:+.1f}"))
    else:
        evidence.append(("CON", f"NM spec degrades on crashes: dBIC={penalty_cr:+.1f}"))

    if enc["p"] < 0.05:
        evidence.append(("PRO", f"Predicted NM encompasses geometry: LR={enc['lr_stat']:.1f}, p={enc['p']:.4f}"))
    else:
        evidence.append(("CON", f"NM does NOT encompass geometry: LR={enc['lr_stat']:.1f}, p={enc['p']:.4f}"))

    if biv["partial_r"] > 0.2 and biv["partial_p"] < 1e-10:
        evidence.append(("PRO", f"Shared latent risk: partial r={biv['partial_r']:.3f}"))
    else:
        evidence.append(("CON", f"Weak shared latent risk: partial r={biv['partial_r']:.3f}"))

    # Copula evidence: prefer the joint-searched result (Step 10) — it's the
    # best-found coupled model, not just theta bolted onto separately-chosen
    # marginals — falling back to the IFM result (Step 8) if the search failed.
    if copula_search is not None:
        cop_label = "Bivariate copula (joint-searched spec)"
        cop_fit   = copula_search["fit"]
        cop_bic_gain  = cop_fit["bic_indep"] - cop_fit["bic_joint"]
        cop_r_delta   = r_cond_cs - r_marg_cs
        cop_theta, cop_lr, cop_p = cop_fit["theta"], cop_fit["lr_stat"], cop_fit["p_lr"]
    else:
        cop_label = "Bivariate copula (IFM on searched marginals)"
        best_cop  = cop["best_copula"]
        cop_fit   = cop["fits"][best_cop]
        cop_bic_gain = cop["bic_indep"] - cop_fit["bic_joint"]
        cop_r_delta  = cop["pred"]["r_cond"] - cop["pred"]["r_marg"]
        cop_theta, cop_lr, cop_p = cop_fit["theta"], cop_fit["lr_stat"], cop_fit["p_lr"]

    if cop_p < 0.01 and cop_bic_gain > 0:
        evidence.append(("PRO", (
            f"{cop_label} rejects independence: "
            f"θ={cop_theta:.3f}, LR={cop_lr:.1f}, "
            f"p={cop_p:.2e}, ΔBIC={cop_bic_gain:.1f}; "
            f"conditional prediction Δr={cop_r_delta:+.4f}"
        )))
    else:
        evidence.append(("CON", (
            f"{cop_label} does NOT significantly improve independence: "
            f"LR={cop_lr:.1f}, p={cop_p:.2e}"
        )))

    pro = sum(1 for e in evidence if e[0] == "PRO")
    con = sum(1 for e in evidence if e[0] == "CON")

    print(f"\n  {'Evidence':<70s} {'Verdict':>10s}")
    print(f"  {'-'*80}")
    for tag, msg in evidence:
        print(f"  {msg:<70s} {'[+]' if tag=='PRO' else '[-]'}")

    print(f"\n  Score: {pro} PRO / {con} CON")
    if pro >= 4:
        ver = "STRONG PROXY: NM can serve as a crash surrogate"
    elif pro >= 2:
        ver = "PARTIAL PROXY: NM provides complementary information"
    else:
        ver = "WEAK PROXY: NM and crashes have different structural drivers"
    print(f"\n  VERDICT: {ver}")

    # Save evidence
    pd.DataFrame([{"Tag": t, "Evidence": m} for t, m in evidence]).to_csv(
        os.path.join(OUTPUT_DIR, "proxy_evidence.csv"), index=False
    )
    print(f"\n  All outputs in: {OUTPUT_DIR}")
    print(f"  DE plots:     de_convergence_nm.png, de_convergence_crash.png, "
          f"de_convergence_combined.png")
    print(f"  MCR search:   mcr_search_NM.csv, mcr_search_Crash.csv")
    print(f"  Model tables: de_best_nm.tex, de_best_crash.tex, "
          f"final_spec_nm.tex, final_spec_crash.tex")
    print(f"  Diagnostics:  coefficient_proportionality.png, bivariate_residuals.png")
    print(f"  Copula (IFM): copula_comparison.csv, copula_conditional_predictions.csv, "
          f"copula_prediction_comparison.png")
    print(f"  Copula (Joint MLE): copula_joint_comparison.csv")
    print(f"  Copula (structure search): copula_search_convergence.png, copula_search_best.csv")
    print(f"  RP-NB:        rp_nb_comparison.csv")
    print(f"  Summary:      proxy_evidence.csv")
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Near-miss proxy variable analysis")
    ap.add_argument("--data",    default=None, help="Path to near_miss_ecf.csv")
    ap.add_argument("--out-dir", default=None, dest="out_dir",
                    help="Directory for output files (default: script directory)")
    args = ap.parse_args()
    if args.data:
        DATA_PATH = args.data          # override module-level default
    if args.out_dir:
        OUTPUT_DIR = args.out_dir      # override module-level default
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
