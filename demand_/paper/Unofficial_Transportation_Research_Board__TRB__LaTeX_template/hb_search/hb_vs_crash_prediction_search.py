#!/usr/bin/env python3
"""
HB vs Crash Prediction Model Search
=====================================
Properly searches for the hard-braking vs crash prediction model using
the metacountregressor package (StructureEvaluatorLC + AdaptiveDE/SA),
with explicit transferability analysis between the two response models.

Pipeline:
  1. LOAD DATA          read near_miss_hb_ecf.csv (10,780 segments)
  2. DE FEATURE SEARCH  differential evolution over inclusion masks
     (separate per response: hard-braking and crashes)
  3. MCR STRUCTURE SEARCH  StructureEvaluatorLC + AdaptiveDE over
     parameter roles {fixed, random-independent, random-correlated}
  4. SPECIFICATION TRANSFER  apply each response's final spec to the
     other response; measure BIC penalty (transferability)
  5. ENCOMPASSING TEST  predicted HB encompasses geometry for crashes?
  6. BIVARIATE COPULA   IFM + joint MLE + structure search
  7. TRANSFERABILITY SUMMARY  hard-braking transferability verdict

Outputs (in --out-dir, default = script dir):
  de_search_hb.csv / de_search_crash.csv     DE convergence + selected specs
  mcr_search_hb.csv / mcr_search_crash.csv   MCR structure search results
  spec_transfer.csv                          cross-response BIC penalties
  copula_comparison.csv                      Frank / Normal / Kimeldorf
  transferability_summary.csv                hard-braking transferability verdict
  hb_vs_crash_results.tex                    paper-ready LaTeX snippets
  hb_vs_crash_*.png                          figures for the paper

Usage:
  python hb_vs_crash_prediction_search.py [--out-dir DIR] [--n-jobs N]
  python hb_vs_crash_prediction_search.py --resume [--out-dir DIR]

PBS submission:
  qsub hb_vs_crash_pbs.pbs
"""

import os, sys, re, warnings, argparse, json, time
from dataclasses import replace, dataclass
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.optimize import minimize as scipy_min, differential_evolution
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import metacountregressor as _mcr
from metacountregressor.main_hpc import compute_standard_errors
from metacountregressor.main_hpc_lc_patch import (
    build_model_from_manual_spec, mixed_model_loglik, build_param_index,
)
from metacountregressor.experiment_package import (
    StructureEvaluatorLC, ExperimentBuilder, FitResult,
)
from metacountregressor.Solvers_METAJAX import (
    AdaptiveDE, NSGA2Engine, AdvancedSimulatedAnnealing,
)

warnings.filterwarnings("ignore")
os.environ["JAX_LOG_COMPILES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

# ===================================================================
# CONFIGURATION
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "near_miss_hb_ecf.csv"),
    r"Z:\test_runs_tours\demand_\paper\Unofficial_Transportation_Research_Board__TRB__LaTeX_template\near_miss_hb_ecf.csv",
    r"Z:\test_runs_tours\demand_\near_miss_hb_ecf.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if os.path.exists(p)), DATA_CANDIDATES[0])

GEO = [
    "AADT", "SL", "NL", "SKD_SP", "CU_MOD", "CU_SHA", "CU_VSHA",
    "DEL_P", "IC", "IP", "MEDPM", "MEDSFBar", "PA_COM1",
    "PSPS1T24", "PSPSNIL", "PSDSG24", "PSDS1T24", "PSDSNIL",
    "IV_VL", "IV_L", "IV_M", "IV_100",
    "RSPSDL5", "RSDSDL1", "RSDSDG10", "RSDSO",
    "RSPSDL1", "RSDSDL5", "RSPSDL10", "RSDSDL10", "RSPSDG10",
]

RESPONSES = {
    "HB": {"col": "nm_total", "label": "Hard-Braking Total Count"},
    "CRASH": {"col": "ObsCrashes", "label": "Observed Crash Count"},
}

# ===================================================================
# CHECKPOINT / RESUME SYSTEM
# ===================================================================
class CheckpointManager:
    """JSON-based checkpoint system for resume support."""
    def __init__(self, out_dir):
        self.ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def done(self, step):
        p = os.path.join(self.ckpt_dir, f"{step}.done")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f).get("completed", False)
            except Exception:
                return False
        return False

    def mark(self, step, meta=None):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        d = {"step": step, "completed": True, "ts": time.time()}
        if meta:
            d.update(meta)
        with open(os.path.join(self.ckpt_dir, f"{step}.done"), "w") as f:
            json.dump(d, f)

    def results(self, step):
        p = os.path.join(self.ckpt_dir, f"{step}.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    def save_results(self, step, results):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        serialisable = {}
        for k, v in results.items():
            try:
                json.dumps(v)
                serialisable[k] = v
            except (TypeError, ValueError):
                serialisable[k] = str(v) if v is not None else None
        with open(os.path.join(self.ckpt_dir, f"{step}.json"), "w") as f:
            json.dump({"step": step, **serialisable}, f, default=str)


# ===================================================================
# DATA LOADING
# ===================================================================
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    geo = [c for c in GEO if c in df.columns and df[c].std() > 1e-10]
    df = df.dropna(subset=["ObsCrashes"]).copy()
    df = df[df["ObsCrashes"] >= 0]
    for c in geo:
        if df[c].isna().any():
            df[c] = 0.0
    df["seg_len_km"] = (
        pd.to_numeric(df.get("segment_length_m", 0), errors="coerce")
        .fillna(0) / 1000.0
    ).clip(lower=0.001)
    df["log_len"] = np.log(df["seg_len_km"])
    for c in ["nm_total", "nm_braking", "hb_critical", "hb_extreme",
              "hb_candidate", "ObsCrashes"]:
        if c in df.columns and df[c].isna().any():
            df[c] = 0.0
    for c in geo:
        if df[c].std() > 1e-10:
            df[c] = (df[c] - df[c].mean()) / df[c].std()
    print(f"  Segments: {len(df):,} | Geometry features: {len(geo)}")
    print(f"  Crashes: {int(df['ObsCrashes'].sum()):,} | HB: {int(df['nm_total'].sum()):,}")
    return df, geo


# ===================================================================
# MODEL FITTER (metacountregressor-powered)
# ===================================================================
def fit_nb_model(df, features, y_col, offset_col="log_len", label=""):
    """Fit a fixed-effects NB model using metacountregressor's
    build_model_from_manual_spec + mixed_model_loglik."""
    df_m = df.dropna(subset=features + [y_col, offset_col]).reset_index(drop=True)
    df_m["_id"] = np.arange(len(df_m))
    df_m["_off"] = df_m[offset_col].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_m[features].values.astype(float))
    for i in range(len(features)):
        df_m[f"__f{i}__"] = Xs[:, i]
    fstd = [f"__f{i}__" for i in range(len(features))]
    col2f = {feat: f"__f{i}__" for i, feat in enumerate(features)}

    manual_spec = {
        "fixed_terms": fstd, "rdm_terms": [], "rdm_cor_terms": [],
        "grouped_terms": [], "hetro_in_means": [],
        "transformations": ["no"] * len(fstd),
        "dispersion": 1, "latent_classes": 1,
    }
    data, spec = build_model_from_manual_spec(
        df_m, manual_spec, id_col="_id", y_col=y_col, offset_col="_off",
    )
    spec = replace(spec, model="nb")
    pi = build_param_index(spec)
    K = pi["total_params"]
    init = np.zeros(K)
    init[pi["fixed"][0]] = np.log(max(df_m[y_col].mean(), 0.01))

    def obj(p):
        return float(mixed_model_loglik(jnp.array(p), data, spec, indivi=False))

    res = scipy_min(obj, init, method="L-BFGS-B", options={"maxiter": 5000})
    params = res.x
    n = len(df_m)
    ll = -obj(params)
    aic = 2 * K - 2 * ll
    bic = K * np.log(n) - 2 * ll

    se = None
    try:
        se_jax = np.asarray(compute_standard_errors(jnp.array(params), obj))
        if np.isfinite(se_jax).any() and np.nanmax(np.abs(se_jax)) < 50:
            se = se_jax
    except Exception:
        pass
    if se is None:
        try:
            H = np.zeros((K, K))
            eps = 1e-4
            f0 = obj(params)
            for i in range(K):
                ei = np.zeros(K); ei[i] = eps
                H[i, i] = (obj(params + ei) - 2 * f0 + obj(params - ei)) / eps**2
                for j in range(i + 1, K):
                    ej = np.zeros(K); ej[j] = eps
                    H[i, j] = (obj(params + ei + ej) - obj(params + ei - ej)
                               - obj(params - ei + ej) + obj(params - ei - ej)) / (4 * eps**2)
            cov = np.linalg.lstsq(H, np.eye(K), rcond=None)[0]
            se = np.sqrt(np.maximum(np.diag(cov), 0))
        except Exception:
            se = np.full(K, np.nan)

    se = np.where(np.abs(se) > 50, np.nan, se)
    z = np.where(np.isfinite(se) & (se > 1e-12), params / se, np.nan)
    p_vals = np.where(np.isfinite(z), 2 * (1 - sps.norm.cdf(np.abs(z))), np.nan)

    names = ["(Intercept)"] + features
    if spec.model == "nb" and K > len(names):
        names.append("ln(overdispersion)")

    coef_tbl = pd.DataFrame({
        "Variable": names[:K], "Estimate": params[:K],
        "Std.Err": se[:K], "z": z[:K], "p": p_vals[:K],
        "IRR": np.exp(params[:K]),
    })

    return {
        "label": label, "ll": ll, "aic": aic, "bic": bic, "k": K, "n": n,
        "converged": res.success, "coef_table": coef_tbl, "params": params,
        "se": se, "features": features, "y_col": y_col,
        "scaler": scaler, "spec": spec, "pi": pi,
        "alpha": float(np.exp(params[-1])) if spec.model == "nb" and "dispersion" in pi else np.nan,
    }


# ===================================================================
# 1. DIFFERENTIAL EVOLUTION FEATURE SEARCH
# ===================================================================
def de_feature_search(df, candidates, y_col, offset_col, label="DE",
                      maxiter=60, popsize=15, seed=42,
                      p_threshold=0.10, penalty=10.0):
    """Differential evolution over binary feature masks with significance penalty."""
    print(f"\n{'='*70}")
    print(f"  DIFFERENTIAL EVOLUTION [{label}] — y = {y_col}")
    print(f"  {len(candidates)} candidates, maxiter={maxiter}, popsize={popsize}")
    print(f"  p-threshold={p_threshold}, penalty={penalty}/insig")
    print(f"{'='*70}")

    dfs = df.dropna(subset=candidates + [y_col, offset_col]).reset_index(drop=True)
    af, Kf = list(candidates), len(candidates)

    def _score(x):
        mask = (np.asarray(x) > 0.5).astype(int)
        if mask.sum() < 2:
            return 1e12
        sel = [af[i] for i in range(Kf) if mask[i] == 1]
        o = fit_nb_model(dfs, sel, y_col, offset_col, label="DE")
        if o is None:
            return 1e12
        coefs = o["coef_table"]
        feature_rows = coefs.iloc[1:]
        if coefs["Variable"].iloc[-1].startswith("ln("):
            feature_rows = feature_rows.iloc[:-1]
        n_insig = sum(1 for p in feature_rows["p"]
                      if np.isnan(p) or p > p_threshold)
        return o["bic"] + n_insig * penalty

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

    final = fit_nb_model(dfs, bf, y_col, offset_col, label=f"DE-{label}")
    coefs_f = final["coef_table"]
    feature_rows_f = coefs_f.iloc[1:]
    if coefs_f["Variable"].iloc[-1].startswith("ln("):
        feature_rows_f = feature_rows_f.iloc[:-1]
    n_insig_f = sum(1 for p in feature_rows_f["p"]
                    if np.isnan(p) or p > p_threshold)
    pb = final["bic"] + n_insig_f * penalty

    print(f"\n  BEST [{label}]: raw_BIC={final['bic']:.2f}  "
          f"penalised_BIC={pb:.2f}  insig={n_insig_f}/{len(bf)}  "
          f"| {len(bf)} features")
    for f in bf:
        pv = feature_rows_f[coefs_f["Variable"] == f]["p"].values
        p_str = f"p={pv[0]:.4f}" if len(pv) and np.isfinite(pv[0]) else "p=nan"
        sig = "***" if (len(pv) and (not np.isnan(pv[0])) and pv[0] < p_threshold) else ""
        print(f"    {sig:3s} {f:<50s} {p_str}")

    return final, bf, best_so_far.tolist()


# ===================================================================
# 2. MCR STRUCTURE SEARCH (using metacountregressor StructureEvaluatorLC)
# ===================================================================
def mcr_structure_search(df, y_col, feats_subset, offset_col,
                         label="MCR-GA", n_starts=1, max_iter=500,
                         R=5, N_max=1500, solver="de", pop_size=30):
    """Single-start structure search (no-restarts policy; n_starts must stay 1)."""
    print(f"\n{'='*70}")
    print(f"  MCR STRUCTURE SEARCH [{label}] — y = {y_col}")
    print(f"  {len(feats_subset)} features, n_starts={n_starts}, "
          f"max_iter={max_iter}, N_max={N_max}")
    print(f"{'='*70}")

    try:
        from metacountregressor.Solvers_METAJAX import AdaptiveDE, NSGA2Engine, AdvancedSimulatedAnnealing
        from metacountregressor.experiment_package import StructureEvaluatorLC
    except ImportError as e:
        print(f"  MCR search: Cannot import metacountregressor components: {e}")
        return None

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

    evaluator = StructureEvaluatorLC(
        df_m, "_id", y_col, std_cols,
        allowed_roles, allowed_dists,
        mode="single", offset_col=offset_col,
        R=R, max_latent_classes=1,
    )

    D = len(std_cols)
    dim = 2 * D + 2  # roles + dists + dispersion bit

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
                operator = AdaptiveDE(F=0.5, CR=0.7)
                ng = NSGA2Engine(
                    evaluator=evaluator, operator=operator,
                    dimension=dim, pop_size=pop_size,
                    max_iter=max_iter, n_jobs=1, save_history=True,
                )
                archive, scores = ng.optimize()
                archive = np.atleast_2d(archive)
                scores = np.atleast_1d(scores)
            else:
                sa = AdvancedSimulatedAnnealing(
                    evaluator=evaluator, dimension=dim,
                    max_iter=max_iter, max_time=1800,
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

    return {
        "spec": best_spec, "bic": best_bic,
        "fixed_orig": fixed_orig, "rdm_i_orig": rdm_i_orig,
        "rdm_c_orig": rdm_c_orig, "lc": lc, "nb": nb,
        "evaluator": evaluator,
    }


def _V(c):
    """Variable label lookup."""
    labels = {
        "AADT": "Annual Avg Daily Traffic", "SL": "Speed Limit (km/h)",
        "NL": "Number of Lanes", "SKD_SP": "Skid Resistance Present",
        "CU_MOD": "Curvature: Moderate", "CU_SHA": "Curvature: Sharp",
        "CU_VSHA": "Curvature: Very Sharp", "DEL_P": "Delineation Posts",
        "IC": "Intersection Conflict", "IP": "Intersection Present",
        "MEDPM": "Median Present", "MEDSFBar": "Median Safety Barrier",
        "PA_COM1": "Parking Access (Commercial)",
        "IV_VL": "Interaction Volume: Very Low",
        "IV_L": "Interaction Volume: Low",
        "IV_M": "Interaction Volume: Medium",
        "IV_100": "Interaction Volume: 100+ veh/day",
        "PSPS1T24": "Posted Speed Compliance: 1-24%",
        "PSPSNIL": "Posted Speed Compliance: nil%",
        "PSDSG24": "Speed Differential: >24%",
        "PSDS1T24": "Speed Differential: 1-24%",
        "PSDSNIL": "Speed Differential: nil%",
        "RSPSDL5": "Roadside Severity: Pole ≤5m",
        "RSDSDL1": "Roadside Severity: Ditch ≤1m",
        "RSDSDG10": "Roadside Severity: Ditch >10m",
        "RSDSO": "Roadside Severity: Other",
        "RSPSDL1": "Roadside Severity: Pole ≤1m",
        "RSDSDL5": "Roadside Severity: Ditch ≤5m",
        "RSPSDL10": "Roadside Severity: Pole ≤10m",
        "RSDSDL10": "Roadside Severity: Ditch ≤10m",
        "RSPSDG10": "Roadside Severity: Pole >10m",
    }
    return labels.get(c, c)


# ===================================================================
# 3. SPECIFICATION TRANSFER (HARD-BRAKING TRANSFERABILITY)
# ===================================================================
def specification_transfer(df, hb_features, crash_features, out_dir):
    """
    HARD-BRAKING TRANSFERABILITY: apply each response's final spec to
    the other response. This is the core transferability analysis the
    user flagged as not being properly written.

    The key question: does the hard-braking model's selected geometry
    transfer to crash prediction, and vice versa?
    """
    print(f"\n{'='*70}")
    print(f"  SPECIFICATION TRANSFER — HARD-BRAKING TRANSFERABILITY")
    print(f"{'='*70}")

    results = {}
    for label, feats, y in [
        ("HB-spec → HB (native)", hb_features, "nm_total"),
        ("HB-spec → Crashes", hb_features, "ObsCrashes"),
        ("Crash-spec → HB", crash_features, "nm_total"),
        ("Crash-spec → Crashes (native)", crash_features, "ObsCrashes"),
    ]:
        r = fit_nb_model(df, feats, y, "log_len", label=label)
        results[label] = r

    print(f"\n  {'Model':<40s} {'BIC':>10s} {'AIC':>10s} {'LL':>10s} {'k':>4s}")
    print(f"  {'-'*77}")
    for k in results:
        m = results[k]
        print(f"  {m['label']:<40s} {m['bic']:10.2f} {m['aic']:10.2f} "
              f"{m['ll']:10.2f} {m['k']:4d}")

    hb_native = results["HB-spec → HB (native)"]["bic"]
    cr_native = results["Crash-spec → Crashes (native)"]["bic"]
    hb_to_cr = results["HB-spec → Crashes"]["bic"]
    cr_to_hb = results["Crash-spec → HB"]["bic"]

    penalty_hb = hb_to_cr - hb_native   # cost of using HB spec for crashes
    penalty_cr = cr_to_hb - cr_native   # cost of using crash spec for HB
    penalty_sym = (penalty_hb + penalty_cr) / 2

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  HARD-BRAKING TRANSFERABILITY ANALYSIS                 ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  HB-native BIC:            {hb_native:>12.2f}                    ║")
    print(f"  ║  Crash-native BIC:         {cr_native:>12.2f}                    ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  HB-spec → Crashes:      {hb_to_cr:>12.2f}  (ΔBIC {penalty_hb:+.1f})  ║")
    print(f"  ║  Crash-spec → HB:        {cr_to_hb:>12.2f}  (ΔBIC {penalty_cr:+.1f})  ║")
    print(f"  ║  Symmetric penalty:      {penalty_sym:>12.2f}                    ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    if abs(penalty_cr) < 50:
        verdict = "TRANSFERABLE"
    elif abs(penalty_cr) < 150:
        verdict = "MODERATELY TRANSFERABLE"
    else:
        verdict = "NOT TRANSFERABLE"

    print(f"\n  Hard-braking transferability verdict: {verdict}")
    print(f"  (HB-spec on Crashes ΔBIC = {penalty_hb:+.1f})")
    print(f"  (Crash-spec on HB ΔBIC = {penalty_cr:+.1f})")

    # Save CSV
    rows = []
    for k in results:
        m = results[k]
        rows.append({
            "model": k, "bic": m["bic"], "aic": m["aic"],
            "ll": m["ll"], "k": m["k"], "n": m["n"],
            "features": ";".join(m["features"]),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "spec_transfer.csv"), index=False)

    return {
        "results": results,
        "hb_native_bic": hb_native, "cr_native_bic": cr_native,
        "hb_to_cr_bic": hb_to_cr, "cr_to_hb_bic": cr_to_hb,
        "penalty_hb": penalty_hb, "penalty_cr": penalty_cr,
        "penalty_sym": penalty_sym, "verdict": verdict,
    }


# ===================================================================
# 4. ENCOMPASSING TEST
# ===================================================================
def encompassing_test(df, hb_feats, crash_feats, out_dir):
    """Does predicted HB encompass geometry for crashes?"""
    print(f"\n{'='*70}")
    print(f"  ENCOMPASSING TEST: Does predicted HB encompass geometry?")
    print(f"{'='*70}")

    hb_model = fit_nb_model(df, hb_feats, "nm_total", "log_len", label="Step1: HB ~ Geo")
    print(f"  HB model: LL={hb_model['ll']:.2f}, BIC={hb_model['bic']:.2f}")

    all_feats = sorted(set(hb_feats) | set(crash_feats))
    df_e = df.dropna(subset=all_feats + ["nm_total", "ObsCrashes", "log_len"]).copy()
    params_nm = hb_model["params"]
    n_geo = len(hb_feats)

    scaler_en = StandardScaler()
    X_r = scaler_en.fit_transform(df_e[hb_feats].values.astype(float))
    intercept_nm = params_nm[0]
    beta_nm = params_nm[1:n_geo + 1]
    eta_nm = intercept_nm + X_r @ beta_nm + df_e["log_len"].values
    df_e["pred_hb"] = np.exp(np.clip(eta_nm, -20, 20))
    df_e["pred_hb"] = np.nan_to_num(df_e["pred_hb"], nan=0.0, posinf=1e6, neginf=0.0)

    crash_geo = fit_nb_model(df_e, crash_feats, "ObsCrashes", "log_len",
                             label="Restricted: Crashes ~ Geo")
    crash_nm = fit_nb_model(df_e, crash_feats + ["pred_hb"], "ObsCrashes", "log_len",
                            label="Unrestricted: Crashes ~ Geo + predHB")

    ll_r = crash_geo["ll"]; ll_u = crash_nm["ll"]
    k_r = crash_geo["k"]; k_u = crash_nm["k"]
    lr_stat = 2 * (ll_u - ll_r); df_lr = k_u - k_r
    p_lr = 1 - sps.chi2.cdf(lr_stat, df_lr)

    print(f"  Restricted (Geo only): LL={ll_r:.2f}, BIC={crash_geo['bic']:.2f}")
    print(f"  Unrestricted (+predHB): LL={ll_u:.2f}, BIC={crash_nm['bic']:.2f}")
    print(f"  LR stat = {lr_stat:.2f} | df = {df_lr} | p = {p_lr:.6f}")
    print(f"  ΔBIC = {crash_nm['bic'] - crash_geo['bic']:+.2f}")

    return {
        "lr_stat": lr_stat, "df": df_lr, "p": p_lr,
        "bic_r": crash_geo["bic"], "bic_u": crash_nm["bic"],
        "ll_r": ll_r, "ll_u": ll_u,
    }


# ===================================================================
# 5. BIVARIATE COPULA
# ===================================================================
def _nb_pmf(y, mu, alpha):
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return sps.nbinom.pmf(y, r, p)

def _nb_cdf(y, mu, alpha):
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return sps.nbinom.cdf(y, r, p)

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

_COPULA_FNS = {"frank": _frank_cop, "normal": _normal_cop, "kimeldorf": _clayton_cop}
_COPULA_BOUNDS = {"frank": (0.01, 30.0), "normal": (-0.99, 0.99), "kimeldorf": (0.001, 20.0)}
_COPULA_INIT = {"frank": 1.0, "normal": 0.3, "kimeldorf": 0.5}

def _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, theta):
    p = (cop_fn(u1c, u2c, theta) - cop_fn(u1c, u2p, theta)
         - cop_fn(u1p, u2c, theta) + cop_fn(u1p, u2p, theta))
    return np.maximum(p, 1e-300)

def _rebuild_mu(df_sub, model):
    feats = model["features"]
    params = model["params"]
    X_std = model["scaler"].transform(df_sub[feats].values.astype(float))
    eta = params[0] + X_std @ params[1:len(feats) + 1] + df_sub["log_len"].values
    mu = np.exp(np.clip(eta, -20, 20))
    # MCR's NB2 dispersion is softplus(param), not exp(param). Using exp here
    # mis-scaled the copula margins in the previous generation of results.
    alpha = float(np.logaddexp(0.0, params[-1]))
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
    t0 = _COPULA_INIT[copula_name]

    def neg_ll(t):
        pmf = _joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, float(t[0]))
        return -float(np.sum(np.log(pmf)))

    res = scipy_min(neg_ll, [t0], method="L-BFGS-B",
                    bounds=[(lo, hi)], options={"maxiter": 500, "ftol": 1e-9})
    theta_opt = float(res.x[0])
    ll_joint = float(np.sum(np.log(_joint_pmf_vec(u1c, u1p, u2c, u2p, cop_fn, theta_opt))))

    k_indep = nm_model["k"] + crash_model["k"]
    k_copula = k_indep + 1
    bic_indep = k_indep * np.log(n) - 2 * ll_indep
    bic_copula = k_copula * np.log(n) - 2 * ll_joint
    lr_stat = 2.0 * (ll_joint - ll_indep)
    p_lr = float(1 - sps.chi2.cdf(lr_stat, df=1))

    return {
        "copula": copula_name, "theta": theta_opt, "n": n,
        "ll_joint": ll_joint, "ll_indep": ll_indep,
        "bic_joint": bic_copula, "bic_indep": bic_indep,
        "k_copula": k_copula, "k_indep": k_indep,
        "lr_stat": lr_stat, "p_lr": p_lr,
    }

def bivariate_copula_analysis(df, nm_model, crash_model, out_dir):
    print(f"\n{'='*70}")
    print(f"  BIVARIATE COPULA ANALYSIS — Frank / Normal / Kimeldorf")
    print(f"{'='*70}")

    fits = {}
    for cn in ("frank", "normal", "kimeldorf"):
        print(f"    {cn:12s} ...", end=" ", flush=True)
        r = fit_bivariate_copula(df, "nm_total", nm_model, crash_model, copula_name=cn)
        fits[cn] = r
        print(f"θ={r['theta']:.4f}  LL={r['ll_joint']:.1f}  BIC={r['bic_joint']:.1f}  LR={r['lr_stat']:.1f}  p={r['p_lr']:.2e}")

    r0 = fits["frank"]
    ll_indep = r0["ll_indep"]
    k_indep = r0["k_indep"]
    n = r0["n"]
    bic_indep = k_indep * np.log(n) - 2 * ll_indep
    best_cn = min(fits, key=lambda k: fits[k]["bic_joint"])

    print(f"\n  {'Model':<14} {'θ':>8} {'LL_joint':>12} {'BIC':>10} {'ΔBIC':>8} {'LR':>8} {'p-LR':>12}")
    print(f"  {'-'*80}")
    print(f"  {'Independence':<14} {'---':>8} {ll_indep:>12.1f} {bic_indep:>10.1f} {'0':>8} {'---':>8} {'---':>12}")
    for cn in ("frank", "normal", "kimeldorf"):
        r = fits[cn]
        tag = "  <-- BEST" if cn == best_cn else ""
        print(f"  {cn:<14} {r['theta']:>8.4f} {r['ll_joint']:>12.1f} "
              f"{r['bic_joint']:>10.1f} {r['bic_joint'] - bic_indep:>+8.1f} "
              f"{r['lr_stat']:>8.1f} {r['p_lr']:>12.2e}{tag}")

    rows = [{"copula": "independence", "theta": np.nan, "n": n, "k": k_indep,
             "ll": ll_indep, "bic": bic_indep, "delta_bic": 0.0,
             "lr_stat": np.nan, "p_lr": np.nan}]
    for cn in ("frank", "normal", "kimeldorf"):
        r = fits[cn]
        rows.append({"copula": cn, "theta": r["theta"], "n": r["n"],
                     "k": r["k_copula"], "ll": r["ll_joint"],
                     "bic": r["bic_joint"], "delta_bic": r["bic_joint"] - bic_indep,
                     "lr_stat": r["lr_stat"], "p_lr": r["p_lr"]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "copula_comparison.csv"), index=False)

    return {"fits": fits, "best_copula": best_cn, "bic_indep": bic_indep}


# ===================================================================
# 6. TRANSFERABILITY SUMMARY (paper-ready)
# ===================================================================
def transferability_summary(st, encomp, copula, out_dir):
    """Compile the hard-braking transferability summary for the paper."""
    print(f"\n{'='*70}")
    print(f"  TRANSFERABILITY SUMMARY — HARD-BRAKING vs CRASH PREDICTION")
    print(f"{'='*70}")

    rows = []
    rows.append({"metric": "HB-native BIC", "value": f"{st['hb_native_bic']:.2f}"})
    rows.append({"metric": "Crash-native BIC", "value": f"{st['cr_native_bic']:.2f}"})
    rows.append({"metric": "HB-spec → Crashes BIC", "value": f"{st['hb_to_cr_bic']:.2f}"})
    rows.append({"metric": "HB-spec → Crashes ΔBIC", "value": f"{st['penalty_hb']:+.1f}"})
    rows.append({"metric": "Crash-spec → HB BIC", "value": f"{st['cr_to_hb_bic']:.2f}"})
    rows.append({"metric": "Crash-spec → HB ΔBIC", "value": f"{st['penalty_cr']:+.1f}"})
    rows.append({"metric": "Symmetric transfer penalty", "value": f"{st['penalty_sym']:+.1f}"})
    rows.append({"metric": "Transferability verdict", "value": st["verdict"]})
    rows.append({"metric": "Encompassing LR", "value": f"{encomp['lr_stat']:.1f}"})
    rows.append({"metric": "Encompassing p", "value": f"{encomp['p']:.6f}"})
    rows.append({"metric": "Encompassing ΔBIC", "value": f"{encomp['bic_u'] - encomp['bic_r']:+.1f}"})
    rows.append({"metric": "Best copula", "value": copula["best_copula"]})
    rows.append({"metric": "Copula independence BIC", "value": f"{copula['bic_indep']:.1f}"})

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "transferability_summary.csv"), index=False)

    # Print summary
    print(f"\n  TRANSFERABILITY SUMMARY:")
    print(f"  {'-'*60}")
    for _, row in pd.DataFrame(rows).iterrows():
        print(f"  {row['metric']:<35s} {row['value']:>20s}")

    return rows


# ===================================================================
# 7. FIGURES
# ===================================================================
def make_figures(de_hb, de_cr, st, encomp, copula, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # DE convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(de_hb["history"], "b-", linewidth=0.8, alpha=0.7)
    axes[0].set_title("DE Convergence — Hard-Braking")
    axes[0].set_xlabel("Generation"); axes[0].set_ylabel("Penalised BIC")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(de_cr["history"], "r-", linewidth=0.8, alpha=0.7)
    axes[1].set_title("DE Convergence — Crashes")
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("Penalised BIC")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "de_convergence.png"), dpi=150)
    plt.close(fig)

    # Specification transfer bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    models = ["HB→HB\n(native)", "HB→Cr", "Crash→HB", "Crash→Crash\n(native)"]
    bic_vals = [
        st["results"]["HB-spec → HB (native)"]["bic"],
        st["results"]["HB-spec → Crashes"]["bic"],
        st["results"]["Crash-spec → HB"]["bic"],
        st["results"]["Crash-spec → Crashes (native)"]["bic"],
    ]
    colors = ["#4477AA", "#CC6677", "#CC6677", "#4477AA"]
    bars = ax.bar(models, bic_vals, color=colors, alpha=0.8)
    for bar, v in zip(bars, bic_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("BIC"); ax.set_title("Specification Transfer — BIC by Model")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spec_transfer.png"), dpi=150)
    plt.close(fig)

    # Copula comparison
    fits = copula["fits"]
    if fits:
        fig, ax = plt.subplots(figsize=(8, 4))
        copulas = list(fits.keys()) + ["Independence"]
        bics = [fits[c]["bic_joint"] for c in copulas[:-1]] + [copula["bic_indep"]]
        colors_c = ["#4477AA", "#CC6677", "#88CCEE", "#999999"]
        bars = ax.bar(copulas, bics, color=colors_c, alpha=0.8)
        for bar, v in zip(bars, bics):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Joint BIC"); ax.set_title("Bivariate Copula — Joint BIC")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "copula_comparison.png"), dpi=150)
        plt.close(fig)

    print(f"\n  Figures saved to {out_dir}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HB vs Crash Prediction Model Search using metacountregressor")
    parser.add_argument("--out-dir", default=SCRIPT_DIR)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--de-only", action="store_true",
                        help="Skip MCR structure search (DE only)")
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--mcr-n-starts", type=int, default=1,
                        help="MCR single start (no-restarts policy; must stay 1)")
    parser.add_argument("--mcr-max-iter", type=int, default=500)
    parser.add_argument("--mcr-nmax", type=int, default=1500,
                        help="MCR structure-search subsample cap. Set >= n "
                             "(e.g. 12000) for a full-sample search on HPC.")
    parser.add_argument("--solver", default="de", choices=["de", "sa"])
    parser.add_argument("--data", default=None,
                        help="Override modelling table (e.g. the full-year "
                             "near_miss_hb_ecf_year.csv).")
    parser.add_argument("--panel", default=None,
                        help="segment x month HB panel for temporal transfer "
                             "(default: near_miss_hb_panel_year.csv beside data).")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip held-out prediction + master table.")
    args = parser.parse_args()

    # ---- data source selection --------------------------------------------
    global DATA_PATH
    if args.data:
        DATA_PATH = os.path.abspath(args.data)
    panel = None
    panel_path = args.panel
    if panel_path is None:
        cand = os.path.join(os.path.dirname(DATA_PATH), "near_miss_hb_panel_year.csv")
        panel_path = cand if os.path.exists(cand) else None
    if panel_path and os.path.exists(panel_path):
        panel = pd.read_csv(panel_path, dtype={"segment_id": str})
        print(f"  Temporal panel: {panel_path} ({len(panel):,} rows)")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = CheckpointManager(out_dir)

    print("=" * 70)
    print("  HB vs CRASH PREDICTION MODEL SEARCH")
    print("  Using metacountregressor (StructureEvaluatorLC + AdaptiveDE)")
    print("=" * 70)
    print(f"  Data: {DATA_PATH}")
    print(f"  Output: {out_dir}")
    print(f"  Resume: {args.resume}")

    # Load data
    df, geo = load_data()
    df.to_csv(os.path.join(out_dir, "data_cleaned.csv"), index=False)

    # --- STEP 1: DE Feature Search (both responses) ---
    if not args.resume or not ckpt.done("de_search"):
        print(f"\n{'#'*70}")
        print(f"  STEP 1: DIFFERENTIAL EVOLUTION FEATURE SEARCH")
        print(f"{'#'*70}")

        hb_de, hb_feats, hb_hist = de_feature_search(
            df, geo, "nm_total", "log_len", label="HB",
            maxiter=args.maxiter, popsize=args.popsize)
        cr_de, cr_feats, cr_hist = de_feature_search(
            df, geo, "ObsCrashes", "log_len", label="CRASH",
            maxiter=args.maxiter, popsize=args.popsize)

        ckpt.save_results("de_search_hb", {
            "features": hb_feats, "bic": hb_de["bic"], "ll": hb_de["ll"],
            "history": hb_hist, "n_features": len(hb_feats)})
        ckpt.save_results("de_search_crash", {
            "features": cr_feats, "bic": cr_de["bic"], "ll": cr_de["ll"],
            "history": cr_hist, "n_features": len(cr_feats)})
        ckpt.mark("de_search")
    else:
        print("\n  [RESUME] Loading DE search results...")
        hb_res = ckpt.results("de_search_hb")
        cr_res = ckpt.results("de_search_crash")
        # Re-fit with saved features
        hb_de = fit_nb_model(df, hb_res["features"], "nm_total", "log_len", label="DE-HB")
        cr_de = fit_nb_model(df, cr_res["features"], "ObsCrashes", "log_len", label="DE-CRASH")
        hb_feats = hb_res["features"]
        cr_feats = cr_res["features"]
        hb_hist = hb_res.get("history", [])
        cr_hist = cr_res.get("history", [])

    print(f"\n  HB DE features ({len(hb_feats)}): {hb_feats}")
    print(f"  Crash DE features ({len(cr_feats)}): {cr_feats}")

    # --- STEP 2: MCR Structure Search ---
    if not args.de_only and (not args.resume or not ckpt.done("mcr_search")):
        print(f"\n{'#'*70}")
        print(f"  STEP 2: MCR STRUCTURE SEARCH (metacountregressor)")
        print(f"{'#'*70}")

        hb_mcr = mcr_structure_search(
            df, "nm_total", hb_feats, "log_len", label="HB",
            n_starts=args.mcr_n_starts, max_iter=args.mcr_max_iter,
            N_max=args.mcr_nmax, solver=args.solver)
        cr_mcr = mcr_structure_search(
            df, "ObsCrashes", cr_feats, "log_len", label="CRASH",
            n_starts=args.mcr_n_starts, max_iter=args.mcr_max_iter,
            N_max=args.mcr_nmax, solver=args.solver)

        if hb_mcr:
            ckpt.save_results("mcr_search_hb", {
                "bic": hb_mcr["bic"], "fixed": hb_mcr["fixed_orig"],
                "rdm_i": hb_mcr["rdm_i_orig"], "rdm_c": hb_mcr["rdm_c_orig"],
                "lc": hb_mcr["lc"], "nb": int(hb_mcr["nb"])})
        if cr_mcr:
            ckpt.save_results("mcr_search_crash", {
                "bic": cr_mcr["bic"], "fixed": cr_mcr["fixed_orig"],
                "rdm_i": cr_mcr["rdm_i_orig"], "rdm_c": cr_mcr["rdm_c_orig"],
                "lc": cr_mcr["lc"], "nb": int(cr_mcr["nb"])})
        ckpt.mark("mcr_search")
    elif args.de_only:
        hb_mcr = None
        cr_mcr = None
        print("\n  [--de-only] Skipping MCR structure search.")
    else:
        print("\n  [RESUME] Loading MCR search results...")
        hb_mcr_res = ckpt.results("mcr_search_hb")
        cr_mcr_res = ckpt.results("mcr_search_crash")
        hb_mcr = hb_mcr_res
        cr_mcr = cr_mcr_res

    # --- STEP 3: Specification Transfer (HARD-BRAKING TRANSFERABILITY) ---
    if not args.resume or not ckpt.done("spec_transfer"):
        print(f"\n{'#'*70}")
        print(f"  STEP 3: SPECIFICATION TRANSFER (HARD-BRAKING TRANSFERABILITY)")
        print(f"{'#'*70}")

        # Use DE-selected features for transfer
        st = specification_transfer(df, hb_feats, cr_feats, out_dir)
        ckpt.save_results("spec_transfer", st)
        ckpt.mark("spec_transfer")
    else:
        print("\n  [RESUME] Loading specification transfer results...")
        st = ckpt.results("spec_transfer")
        # Re-fit models
        st["results"] = {}
        for label, feats, y in [
            ("HB-spec → HB (native)", hb_feats, "nm_total"),
            ("HB-spec → Crashes", hb_feats, "ObsCrashes"),
            ("Crash-spec → HB", cr_feats, "nm_total"),
            ("Crash-spec → Crashes (native)", cr_feats, "ObsCrashes"),
        ]:
            st["results"][label] = fit_nb_model(df, feats, y, "log_len", label=label)

    # --- STEP 4: Encompassing Test ---
    if not args.resume or not ckpt.done("encompassing"):
        print(f"\n{'#'*70}")
        print(f"  STEP 4: ENCOMPASSING TEST")
        print(f"{'#'*70}")
        encomp = encompassing_test(df, hb_feats, cr_feats, out_dir)
        ckpt.save_results("encompassing", encomp)
        ckpt.mark("encompassing")
    else:
        print("\n  [RESUME] Loading encompassing test results...")
        encomp = ckpt.results("encompassing")

    # --- STEP 5: Bivariate Copula ---
    if not args.resume or not ckpt.done("copula"):
        print(f"\n{'#'*70}")
        print(f"  STEP 5: BIVARIATE COPULA ANALYSIS")
        print(f"{'#'*70}")
        copula = bivariate_copula_analysis(df, hb_de, cr_de, out_dir)
        ckpt.save_results("copula", copula)
        ckpt.mark("copula")
    else:
        print("\n  [RESUME] Loading copula results...")
        copula = ckpt.results("copula")

    # --- STEP 6: Held-out prediction + master best-models table ---
    holdout = None
    master = None
    if not args.skip_validation:
        import finalize
        holdout = finalize.run_holdout(df, panel, hb_feats, cr_feats, out_dir,
                                       kfolds=args.kfolds)
        ckpt.save_results("holdout", holdout)
        master, _ = finalize.build_master_table(
            df, hb_de, cr_de, hb_feats, cr_feats, st, encomp, copula,
            holdout, hb_mcr, cr_mcr, out_dir)

    # --- STEP 7: Transferability Summary ---
    rows = transferability_summary(st, encomp, copula, out_dir)

    # --- STEP 7: Figures ---
    make_figures(
        {"history": hb_hist}, {"history": cr_hist}, st, encomp, copula, out_dir)

    # --- Save run metadata ---
    meta = {
        "data_path": DATA_PATH,
        "n_segments": len(df),
        "n_geo_features": len(geo),
        "hb_features": hb_feats,
        "crash_features": cr_feats,
        "hb_de_bic": hb_de["bic"],
        "crash_de_bic": cr_de["bic"],
        "hb_mcr_bic": hb_mcr["bic"] if hb_mcr else None,
        "crash_mcr_bic": cr_mcr["bic"] if cr_mcr else None,
        "transferability_verdict": st["verdict"],
        "encompassing_lr": encomp["lr_stat"],
        "encompassing_p": encomp["p"],
        "best_copula": copula["best_copula"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  DONE. All outputs in: {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()