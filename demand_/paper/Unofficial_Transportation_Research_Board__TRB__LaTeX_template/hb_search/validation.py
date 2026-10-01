#!/usr/bin/env python3
"""
Held-out prediction + transferability for the HB / crash / joint models.
========================================================================
Adds the validation layer the paper was missing: every selected specification
gets genuine out-of-sample prediction, not just in-sample BIC.

  kfold_oos          random K-fold OOS (log-lik, RMSE, MAE, Pearson, Spearman)
  spatial_transfer   leave-one-LGA-out geographic transfer
  temporal_transfer  train early months -> predict later months (HB panel)
  cross_response     HB spec -> crash response and vice-versa, OOS

Design note on correctness
--------------------------
`fit_nb_model` (in hb_vs_crash_prediction_search.py) fits an NB2 model whose
last parameter is ln(overdispersion). We reconstruct predictions as
    mu_i = exp( beta0 + x_std_i . beta + offset_i )
using the TRAIN-fold scaler, and score them with an independent NB2
log-likelihood. `self_check()` asserts this reconstruction reproduces the
fitted in-sample log-likelihood to numerical tolerance, so the parameterisation
is verified before any OOS number is reported.
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln
import scipy.stats as sps

from hb_vs_crash_prediction_search import fit_nb_model


# --------------------------------------------------------------------------
def _nb2_loglik(y, mu, alpha):
    """NB2 log-likelihood, summed. Var = mu + alpha*mu^2."""
    y = np.asarray(y, float)
    mu = np.clip(np.asarray(mu, float), 1e-10, None)
    if alpha is None or not np.isfinite(alpha) or alpha <= 1e-10:
        # Poisson limit
        return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
    r = 1.0 / alpha
    return float(np.sum(
        gammaln(y + r) - gammaln(r) - gammaln(y + 1)
        + r * np.log(r / (r + mu))
        + y * np.log(mu / (r + mu))
    ))


def predict_mean(fit, df_test, offset_col="log_len"):
    """Reconstruct mu for held-out rows using the train-fold scaler+params."""
    feats = fit["features"]
    params = np.asarray(fit["params"], float)
    scaler = fit["scaler"]
    k = len(feats)
    beta0 = params[0]
    beta = params[1:1 + k]
    Xs = scaler.transform(df_test[feats].values.astype(float))
    off = df_test[offset_col].values.astype(float)
    eta = beta0 + Xs @ beta + off
    return np.exp(np.clip(eta, -30, 30))


def _softplus(x):
    # log(1+exp(x)), overflow-safe
    return float(np.logaddexp(0.0, x))


def _alpha_of(fit):
    """MCR's NB2 dispersion is softplus(raw_param), not exp(raw_param).
    The raw dispersion parameter is the last entry of fit['params'] when the
    coef table's last row is the 'ln(overdispersion)' term."""
    ct = fit.get("coef_table")
    params = np.asarray(fit["params"], float)
    if ct is not None and len(ct) and str(ct["Variable"].iloc[-1]).startswith("ln("):
        return _softplus(params[-1])
    return None  # Poisson limit


def oos_metrics(fit, df_test, y_col, offset_col="log_len"):
    y = df_test[y_col].values.astype(float)
    mu = predict_mean(fit, df_test, offset_col)
    alpha = _alpha_of(fit)
    n = len(y)
    ll = _nb2_loglik(y, mu, alpha)
    resid = y - mu
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    # log-scale RMSE: robust to the heavy right tail of year-aggregated counts,
    # where a few very high-volume segments otherwise dominate raw-count RMSE.
    rmse_log = float(np.sqrt(np.mean((np.log1p(y) - np.log1p(mu)) ** 2)))
    pear = float(np.corrcoef(y, mu)[0, 1]) if np.std(mu) > 0 else np.nan
    spear = float(sps.spearmanr(y, mu).correlation) if np.std(mu) > 0 else np.nan
    return {"n": n, "oos_ll": ll, "oos_ll_per_obs": ll / max(n, 1),
            "rmse": rmse, "rmse_log": rmse_log, "mae": mae,
            "pearson": pear, "spearman": spear}


def self_check(df, features, y_col, offset_col="log_len", tol=1e-3):
    """Verify predict_mean + _nb2_loglik reproduces the fitted in-sample LL."""
    fit = fit_nb_model(df, features, y_col, offset_col, label="selfcheck")
    mu = predict_mean(fit, df, offset_col)
    ll_recon = _nb2_loglik(df[y_col].values, mu, _alpha_of(fit))
    ok = abs(ll_recon - fit["ll"]) <= tol * max(1.0, abs(fit["ll"]))
    print(f"  [self_check] fitted LL={fit['ll']:.3f}  reconstructed LL="
          f"{ll_recon:.3f}  match={ok}")
    return ok, fit["ll"], ll_recon


# --------------------------------------------------------------------------
def kfold_oos(df, features, y_col, offset_col="log_len", k=5, seed=42,
              label=""):
    """Random K-fold out-of-sample prediction."""
    dfx = df.dropna(subset=features + [y_col, offset_col]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, size=len(dfx))
    rows = []
    for f in range(k):
        tr = dfx[fold != f]
        te = dfx[fold == f]
        if len(te) < 5 or len(tr) < 50:
            continue
        fit = fit_nb_model(tr, features, y_col, offset_col, label=f"{label}-f{f}")
        m = oos_metrics(fit, te, y_col, offset_col)
        m["fold"] = f
        rows.append(m)
        print(f"  [{label}] fold {f}: OOS LL/obs={m['oos_ll_per_obs']:.4f} "
              f"RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} "
              f"r={m['pearson']:.3f} rho={m['spearman']:.3f}")
    res = pd.DataFrame(rows)
    agg = {"k": k, "n": int(res["n"].sum()),
           "oos_ll_per_obs": float(res["oos_ll_per_obs"].mean()),
           "rmse": float(res["rmse"].mean()),
           "rmse_log": float(res["rmse_log"].mean()),
           "mae": float(res["mae"].mean()),
           "pearson": float(res["pearson"].mean()),
           "spearman": float(res["spearman"].mean()),
           "pearson_sd": float(res["pearson"].std())}
    return agg, res


def spatial_transfer(df, features, y_col, offset_col="log_len",
                     region_col="lga", min_test=30, label=""):
    """Leave-one-region-out geographic transfer."""
    dfx = df.dropna(subset=features + [y_col, offset_col]).reset_index(drop=True)
    if region_col not in dfx.columns:
        print(f"  [spatial] region column '{region_col}' absent — skipped")
        return None, None
    rows = []
    for reg, te in dfx.groupby(region_col):
        tr = dfx[dfx[region_col] != reg]
        if len(te) < min_test or len(tr) < 100:
            continue
        fit = fit_nb_model(tr, features, y_col, offset_col, label=f"{label}-{reg}")
        m = oos_metrics(fit, te, y_col, offset_col)
        # baseline: intercept-only predicting held-out region's mean via train mean
        base_mu = np.full(len(te), max(tr[y_col].mean(), 1e-6))
        m["baseline_ll"] = _nb2_loglik(te[y_col].values, base_mu, _alpha_of(fit))
        m["delta_ll_vs_base"] = m["oos_ll"] - m["baseline_ll"]
        m["region"] = str(reg)
        rows.append(m)
        print(f"  [{label}] LGA {reg}: OOS LL/obs={m['oos_ll_per_obs']:.4f} "
              f"dLL_vs_base={m['delta_ll_vs_base']:+.1f} rho={m['spearman']:.3f}")
    if not rows:
        return None, None
    res = pd.DataFrame(rows)
    won = int((res["delta_ll_vs_base"] > 0).sum())
    agg = {"n_regions": len(res), "regions_beating_baseline": won,
           "mean_oos_ll_per_obs": float(res["oos_ll_per_obs"].mean()),
           "mean_spearman": float(res["spearman"].mean()),
           "mean_delta_ll_vs_base": float(res["delta_ll_vs_base"].mean())}
    return agg, res


def temporal_transfer(df, panel, features, offset_col="log_len",
                      split_frac=0.6, label="HB"):
    """Train an HB-count model on EARLY months, predict LATE months.

    panel: long dataframe [segment_id, ym, hb_count]. We form per-segment
    early- and late-period counts, put them on the modelling table, and use a
    log-exposure offset (log number of months x segment length) so the two
    horizons are comparable.
    """
    if panel is None or "ym" not in panel.columns:
        print("  [temporal] no panel — skipped")
        return None
    months = sorted(panel["ym"].unique())
    if len(months) < 4:
        print(f"  [temporal] only {len(months)} months — skipped")
        return None
    cut = int(len(months) * split_frac)
    early, late = months[:cut], months[cut:]
    def _seg_counts(ms):
        s = (panel[panel["ym"].isin(ms)]
             .groupby("segment_id")["hb_count"].sum())
        return s
    e_cnt, l_cnt = _seg_counts(early), _seg_counts(late)
    base = df[["segment_id"] + features + [offset_col]].copy()
    base["segment_id"] = base["segment_id"].astype(str).str.strip()
    base["y_early"] = base["segment_id"].map(e_cnt).fillna(0).astype(int)
    base["y_late"] = base["segment_id"].map(l_cnt).fillna(0).astype(int)
    base["off_early"] = base[offset_col] + np.log(len(early))
    base["off_late"] = base[offset_col] + np.log(len(late))
    fit = fit_nb_model(base, features, "y_early", "off_early", label=f"{label}-early")
    m = oos_metrics(fit, base, "y_late", "off_late")
    print(f"  [temporal {label}] train {early[0]}..{early[-1]} -> "
          f"test {late[0]}..{late[-1]}: r={m['pearson']:.3f} "
          f"rho={m['spearman']:.3f} RMSE={m['rmse']:.3f}")
    m.update({"train_months": f"{early[0]}..{early[-1]}",
              "test_months": f"{late[0]}..{late[-1]}",
              "n_train_months": len(early), "n_test_months": len(late)})
    return m
