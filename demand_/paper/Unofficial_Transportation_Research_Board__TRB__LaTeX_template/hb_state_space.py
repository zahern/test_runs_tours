"""
hb_state_space.py
=================
State-space (dynamic Poisson/NB) model for estimating crash risk from
hard-braking events with temporal components.

Model specification:
  y_{s,t} ~ NB(lambda_{s,t}, alpha)
  log(lambda_{s,t}) = alpha_s + gamma_{d,h} + theta_{s,t} + beta' x_s
  theta_{s,t} = theta_{s,t-1} + eta_t,  eta_t ~ N(0, sigma2_theta)

Pipeline:
  1. Load raw HB events; parse timestamps; extract hour, DOW, date.
  2. Aggregate to daily counts per spatial grid cell (lat/lon bins).
  3. Load segment-level covariates from near_miss_hb_ecf.csv.
  4. Fit NB2 via scipy MLE with DOW fixed effects + segment covariates.
  5. Apply Kalman filter to log-residuals for latent crash-risk trend.
  6. Generate outputs.

Outputs (in --out-dir):
  hb_state_space.tex            paper-ready table block
  hb_state_space_trace.png      latent crash-risk over time
  hb_state_space_temporal.png   hour-of-day / day-of-week risk patterns
  hb_state_space_prediction.csv per-cell-period crash-risk predictions
  hb_state_space_summary.csv    summary statistics

Usage:
  python hb_state_space.py
  python hb_state_space.py --events hard_braking_events_year.csv
  python hb_state_space.py --grid-res 0.005
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import nbinom as _nbinom
from scipy.optimize import minimize as _minimize
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DIR = os.path.dirname(os.path.abspath(__file__))
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================================
# 1. DATA LOADING
# =====================================================================

def load_events(path, nrows=None):
    """Load hard-braking events, parse timestamps, extract temporal keys."""
    print(f"  loading events from {os.path.basename(path)} ...")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df["EventStart"] = pd.to_datetime(df["EventStart"], errors="coerce")
    df = df.dropna(subset=["EventStart", "SnappedLongitude", "SnappedLatitude"])
    df["date"] = df["EventStart"].dt.date
    df["hour"] = df["EventStart"].dt.hour
    df["dow"] = df["EventStart"].dt.dayofweek
    df["week"] = df["EventStart"].dt.isocalendar().week.astype(int)
    print(f"    {len(df):,} events, "
          f"{df['VehicleID'].nunique():,} vehicles, "
          f"{df['date'].nunique()} dates")
    return df


def load_segments(path):
    """Load segment-level geometry and HB aggregates."""
    print(f"  loading segments from {os.path.basename(path)} ...")
    seg = pd.read_csv(path, low_memory=False)
    num_cols = ["AADT", "SL", "NL", "CU_MOD", "CU_SHA", "CU_VSHA",
                "DEL_P", "IC", "IP", "MEDPM", "MEDSFBar", "PA_COM1",
                "PSPS1T24", "PSPSNIL", "PSDSG24", "PSDS1T24", "PSDSNIL",
                "SKD_SP", "segment_length_m",
                "IV_VL", "IV_L", "IV_M", "IV_100",
                "RSPSDL5", "RSDSDL1", "RSDSDG10", "RSDSO",
                "RSPSDL1", "RSDSDL5", "RSPSDL10", "RSDSDL10", "RSPSDG10",
                "ObsCrashes", "HB_Total", "HB_Critical-Likely", "HB_Extreme"]
    for c in num_cols:
        if c in seg.columns:
            seg[c] = pd.to_numeric(seg[c], errors="coerce").fillna(0.0)
    seg["log_len"] = np.log(seg["segment_length_m"].clip(lower=0.1) / 1000.0)
    print(f"    {len(seg):,} segments, "
          f"{seg['HB_Total'].sum():,} total HB events")
    return seg


# =====================================================================
# 2. SPATIAL GRID AGGREGATION
# =====================================================================

def assign_grid_cells(events, grid_res=0.01):
    """Bin events into lat/lon grid cells; returns grid_id per event."""
    events = events.copy()
    events["grid_lat"] = (events["SnappedLatitude"] / grid_res).round().astype(int)
    events["grid_lon"] = (events["SnappedLongitude"] / grid_res).round().astype(int)
    events["grid_id"] = (events["grid_lat"].astype(str) + "_"
                         + events["grid_lon"].astype(str))
    return events


def build_panel(events, grid_res=0.01):
    """Build panel: grid_id x date with HB counts and severity splits."""
    events = assign_grid_cells(events, grid_res)
    panel = (events.groupby(["grid_id", "date"])
             .agg(
                 hb_count=("VehicleID", "size"),
                 n_critical=("Category", lambda s: (s == "Critical-Likely").sum()),
                 n_extreme=("Category", lambda s: (s == "Extreme").sum()),
                 mean_decel=("PeakDecel_g", "mean"),
                 mean_jerk=("PeakJerk_g_s", "mean"),
                 mean_speedloss=("SpeedLoss_kmh", "mean"),
                 grid_lat=("SnappedLatitude", "mean"),
                 grid_lon=("SnappedLongitude", "mean"),
             )
             .reset_index())
    panel["date"] = pd.to_datetime(panel["date"])
    panel["dow"] = panel["date"].dt.dayofweek
    panel["week"] = panel["date"].dt.isocalendar().week.astype(int)
    panel["log_hb"] = np.log(panel["hb_count"].clip(lower=0.5))
    panel["log_hb_c"] = panel["log_hb"] - panel["log_hb"].mean()
    print(f"  panel: {panel['grid_id'].nunique()} cells x "
          f"{panel['date'].nunique()} dates = {len(panel):,} obs")
    return panel


def build_hourly(events, grid_res=0.01):
    """Build panel: grid_id x hour with HB counts (hour-floored timestamps).

    Same grid assignment as build_panel; hour bins span the full observation
    window (Jan 2025 - Mar 2026 on the year file). Returns hourly network
    aggregates ready for the Kalman filter: one row per clock-hour with
    mean_log_hb (mean centred log-count across active cells), total_hb and
    n_cells.
    """
    events = assign_grid_cells(events, grid_res)
    events["hour_ts"] = events["EventStart"].dt.floor("h")
    g = events.groupby(["grid_id", "hour_ts"])
    cell_hour = g.agg(hb_count=("VehicleID", "size")).reset_index()
    cell_hour["log_hb"] = np.log(cell_hour["hb_count"].clip(lower=0.5))
    cell_hour["log_hb_c"] = cell_hour["log_hb"] - cell_hour["log_hb"].mean()
    hourly = (cell_hour.groupby("hour_ts")
              .agg(mean_log_hb=("log_hb_c", "mean"),
                   total_hb=("hb_count", "sum"),
                   n_cells=("grid_id", "nunique"))
              .reset_index()
              .sort_values("hour_ts")
              .reset_index(drop=True))
    print(f"  hourly: {hourly['hour_ts'].nunique()} clock-hours, "
          f"{len(cell_hour):,} cell-hours")
    return hourly


# =====================================================================
# 3. NB2 MLE (scipy-based, no statsmodels dependency)
# =====================================================================

def _nb2_loglik(params, y, X):
    """Negative log-likelihood for NB2 GLM.

    params = [beta_0, ..., beta_{p-1}, log_alpha]
    mu = exp(X @ beta)
    alpha = dispersion (NB2 variance = mu + alpha * mu^2)
    """
    beta = params[:-1]
    log_alpha = params[-1]
    alpha = np.exp(log_alpha)
    eta = X @ beta
    mu = np.exp(eta)
    mu = np.clip(mu, 1e-10, 1e10)
    # NB2: r = 1/alpha, p = r/(r+mu)
    r = 1.0 / max(alpha, 1e-12)
    p = r / (r + mu)
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    ll = _nbinom.logpmf(y, r, p)
    return -np.sum(ll)


def _nb2_hessian_diag(params, y, X):
    """Approximate diagonal of Hessian via finite differences for SE."""
    n = len(params)
    hess_diag = np.full(n, np.nan)
    f0 = _nb2_loglik(params, y, X)
    eps = 1e-5 * (1.0 + np.abs(params))
    for i in range(n):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps[i]
        p_minus[i] -= eps[i]
        f_plus = _nb2_loglik(p_plus, y, X)
        f_minus = _nb2_loglik(p_minus, y, X)
        hess_diag[i] = (f_plus - 2 * f0 + f_minus) / (eps[i] ** 2)
    return hess_diag


def fit_nb_mle(y, X, label=""):
    """Fit NB2 via maximum likelihood. Returns dict with results."""
    n, p = X.shape
    # Poisson start
    try:
        log_y = np.log(y + 0.5)
        beta0 = np.linalg.lstsq(X, log_y, rcond=None)[0]
    except Exception:
        beta0 = np.zeros(p)

    # initial alpha guess
    mu0 = np.exp(X @ beta0)
    alpha0 = max(float(np.var(y - mu0) / max(np.mean(mu0 ** 2), 1e-9)), 0.01)
    x0 = np.concatenate([beta0, [np.log(alpha0)]])

    # L-BFGS-B (gradient-based, faster for larger parameter sets)
    try:
        res = _minimize(_nb2_loglik, x0, args=(y, X), method="L-BFGS-B",
                        options={"maxiter": 5000, "ftol": 1e-10})
    except Exception:
        res = _minimize(_nb2_loglik, x0, args=(y, X), method="Nelder-Mead",
                        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})

    beta_hat = res.x[:-1]
    alpha_hat = np.exp(res.x[-1])
    mu_hat = np.exp(X @ beta_hat)

    # Hessian-based SE
    hess_diag = _nb2_hessian_diag(res.x, y, X)
    se_beta = np.where(hess_diag > 0, 1.0 / np.sqrt(hess_diag), np.nan)
    se_beta = se_beta[:-1]  # drop alpha SE

    # p-values (Wald z-test)
    z = beta_hat / np.where(se_beta > 0, se_beta, np.nan)
    pvals = 2.0 * sps.norm.sf(np.abs(z))

    ll = -res.fun
    k = p + 1  # beta + alpha
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    # deviance
    dev = 2.0 * np.sum(y * np.log(np.clip(y, 1e-10, None) / np.clip(mu_hat, 1e-10, None))
                        - (y + alpha_hat) * np.log(
                            (y + alpha_hat) / (mu_hat + alpha_hat)))

    return {
        "params": beta_hat,
        "se": se_beta,
        "pvals": pvals,
        "alpha": alpha_hat,
        "llf": ll,
        "aic": aic,
        "bic": bic,
        "deviance": dev,
        "fittedvalues": mu_hat,
        "nobs": n,
        "df_model": p - 1,
        "converged": res.success,
        "x_names": [f"x{i}" for i in range(p)],
    }


def fit_nb_with_dow(panel):
    """Fit NB2 of hb_count ~ DOW fixed effects. Returns result dict."""
    dows = pd.get_dummies(panel["dow"], prefix="dow", drop_first=True,
                          dtype=float)
    X = sm_add_constant(dows.values)
    y = panel["hb_count"].values.astype(float)
    res = fit_nb_mle(y, X, label="temporal")
    # name the DOW dummies
    names = ["Intercept"] + [f"DOW={DOW_NAMES[i+1]}" for i in range(6)]
    res["x_names"] = names
    return res, X


def sm_add_constant(X):
    """Add intercept column (like statsmodels)."""
    n = X.shape[0] if hasattr(X, "shape") else len(X)
    X = np.asarray(X, float)
    return np.column_stack([np.ones(n), X])


# =====================================================================
# 4. KALMAN FILTER FOR LATENT CRASH-RISK TREND
# =====================================================================

class KalmanFilter1D:
    """Scalar Kalman filter for a random-walk latent state.

    State:  theta_t = theta_{t-1} + eta_t,   eta_t ~ N(0, Q)
    Obs:    z_t = theta_t + eps_t,            eps_t ~ N(0, R)
    """

    def __init__(self, Q_init=0.1, R_init=1.0):
        self.Q = Q_init
        self.R = R_init

    def _filter(self, z, Q, R):
        """Forward Kalman filter. Returns (theta_filtered, P_filtered)."""
        n = len(z)
        theta = np.empty(n)
        P = np.empty(n)
        theta[0] = z[0]
        P[0] = R + Q
        for t in range(1, n):
            theta_pred = theta[t - 1]
            P_pred = P[t - 1] + Q
            K = P_pred / (P_pred + R)
            theta[t] = theta_pred + K * (z[t] - theta_pred)
            P[t] = (1.0 - K) * P_pred
        return theta, P

    def loglik(self, z, Q, R):
        """Negative log-likelihood of observations."""
        n = len(z)
        ll = 0.0
        theta_prev = z[0]
        P_prev = R + Q
        S0 = R + Q
        ll += -0.5 * (np.log(2 * np.pi * S0) + (z[0] - 0) ** 2 / S0)
        for t in range(1, n):
            theta_pred = theta_prev
            P_pred = P_prev + Q
            S = P_pred + R
            innov = z[t] - theta_pred
            ll += -0.5 * (np.log(2 * np.pi * S) + innov ** 2 / S)
            K = P_pred / S
            theta_new = theta_pred + K * innov
            P_new = (1.0 - K) * P_pred
            theta_prev = theta_new
            P_prev = P_new
        return -ll

    def fit_mle(self, z):
        """Estimate Q, R by MLE via grid search + Nelder-Mead."""
        best_ll = np.inf
        best_Q, best_R = self.Q, self.R
        for Q_try in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
            for R_try in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
                try:
                    ll = self.loglik(z, Q_try, R_try)
                    if ll < best_ll:
                        best_ll, best_Q, best_R = ll, Q_try, R_try
                except Exception:
                    continue

        def negll(params):
            Q_t, R_t = np.exp(params)
            try:
                return self.loglik(z, Q_t, R_t)
            except Exception:
                return 1e12

        try:
            res = _minimize(negll,
                            x0=[np.log(best_Q), np.log(best_R)],
                            method="Nelder-Mead",
                            options={"maxiter": 1000, "xatol": 1e-5, "fatol": 1e-5})
            self.Q, self.R = np.exp(res.x)
        except Exception:
            self.Q, self.R = best_Q, best_R

        return self

    def smooth(self, z):
        """Forward filter + Rauch-Tung-Striebel backward smoother."""
        n = len(z)
        theta_f, P_f = self._filter(z, self.Q, self.R)
        theta_s = np.empty(n)
        theta_s[-1] = theta_f[-1]
        for t in range(n - 2, -1, -1):
            P_pred = P_f[t] + self.Q
            A = P_f[t] / max(P_pred, 1e-12)
            theta_s[t] = theta_f[t] + A * (theta_s[t + 1] - theta_f[t])
        return theta_s


def apply_kalman(panel):
    """Apply Kalman filter to daily mean log-counts across grid cells.

    Returns (daily_df, Q, R).
    """
    daily = (panel.groupby("date")
             .agg(
                 mean_log_hb=("log_hb_c", "mean"),
                 total_hb=("hb_count", "sum"),
                 n_cells=("grid_id", "nunique"),
             )
             .reset_index()
             .sort_values("date")
             .reset_index(drop=True))

    z = daily["mean_log_hb"].values.astype(float)
    mask = np.isfinite(z)
    if mask.sum() < 3:
        print("    WARNING: too few valid observations for Kalman filter")
        daily["latent_risk"] = 0.0
        daily["latent_se"] = np.nan
        return daily, 0.1, 1.0

    z[~mask] = np.interp(np.where(~mask)[0], np.where(mask)[0], z[mask])

    kf = KalmanFilter1D()
    kf.fit_mle(z)
    theta_smooth = kf.smooth(z)

    daily["latent_risk"] = theta_smooth
    _, P_f = kf._filter(z, kf.Q, kf.R)
    daily["latent_se"] = np.sqrt(np.maximum(P_f, 0.0))

    print(f"    Kalman: Q={kf.Q:.4f}, R={kf.R:.4f}, "
          f"range=[{theta_smooth.min():.3f}, {theta_smooth.max():.3f}]")
    return daily, kf.Q, kf.R


def apply_kalman_hourly(hourly):
    """Apply Kalman filter to hourly mean log-counts across grid cells.

    The hourly series carries a strong calendar (day-of-week x hour-of-day)
    seasonal that would drive the observation variance to zero, so it is
    removed first and the filter runs on the deseasonalised residual:
    latent risk is then the deviation from the calendar norm. Returns
    (hourly_df, Q, R) with latent_risk/latent_se plus seasonal columns.
    """
    hod = hourly["hour_ts"].dt.hour
    dow = hourly["hour_ts"].dt.dayofweek
    seasonal = hourly.groupby([dow, hod])["mean_log_hb"].transform("mean")
    hourly["seasonal"] = seasonal.values
    hourly["seasonal_amp"] = float(seasonal.max() - seasonal.min())
    z = (hourly["mean_log_hb"] - hourly["seasonal"]).values.astype(float)
    mask = np.isfinite(z)
    if mask.sum() < 3:
        print("    WARNING: too few valid observations for hourly Kalman filter")
        hourly["latent_risk"] = 0.0
        hourly["latent_se"] = np.nan
        return hourly, 0.1, 1.0

    z[~mask] = np.interp(np.where(~mask)[0], np.where(mask)[0], z[mask])

    kf = KalmanFilter1D()
    kf.fit_mle(z)
    theta_smooth = kf.smooth(z)

    hourly["latent_risk"] = theta_smooth
    _, P_f = kf._filter(z, kf.Q, kf.R)
    hourly["latent_se"] = np.sqrt(np.maximum(P_f, 0.0))

    print(f"    Hourly Kalman: Q={kf.Q:.4f}, R={kf.R:.4f}, "
          f"range=[{theta_smooth.min():.3f}, {theta_smooth.max():.3f}]")
    return hourly, kf.Q, kf.R


# =====================================================================
# 5. CROSS-SECTIONAL SEGMENT NB2
# =====================================================================

SEG_FEATS = ["AADT", "SL", "NL", "SKD_SP", "CU_MOD", "CU_SHA", "CU_VSHA",
             "DEL_P", "IC", "IP", "MEDPM", "MEDSFBar", "PA_COM1",
             "IV_VL", "IV_L", "IV_M", "IV_100",
             "PSPS1T24", "PSPSNIL", "PSDSG24", "PSDS1T24", "PSDSNIL",
             "RSPSDL5", "RSDSDL1", "RSDSDG10", "RSDSO",
             "RSPSDL1", "RSDSDL5", "RSPSDL10", "RSDSDL10", "RSPSDG10"]

# Mapping from LaTeX variable names to CSV column names
# (derived from nearmiss_hb_analysis.py VLABEL)
VAR_MAP = {
    "Annual Avg Daily Traffic (AADT)": "AADT",
    "Speed Limit (km/h)": "SL",
    "Number of Lanes": "NL",
    "Skid Resistance Present": "SKD_SP",
    "Curvature: Moderate": "CU_MOD",
    "Curvature: Sharp": "CU_SHA",
    "Curvature: Very Sharp": "CU_VSHA",
    "Delineation Posts Present": "DEL_P",
    "Intersection Conflict Present": "IC",
    "Intersection Present": "IP",
    "Median Present": "MEDPM",
    "Median Safety Barrier": "MEDSFBar",
    "Parking Access (Commercial)": "PA_COM1",
    "Interaction Volume: Very Low": "IV_VL",
    "Interaction Volume: Low": "IV_L",
    "Interaction Volume: Medium": "IV_M",
    "Posted Speed Compliance: 1--24\\%": "PSPS1T24",
    "Posted Speed Compliance: nil\\%": "PSPSNIL",
    "Speed Differential: >24\\%": "PSDSG24",
    "Speed Differential: 1--24\\%": "PSDS1T24",
    "Speed Differential: nil\\%": "PSDSNIL",
    "Roadside Severity: Pole $\\leq$5m": "RSPSDL5",
    "Roadside Severity: Ditch $\\leq$1m": "RSDSDL1",
    "Roadside Severity: Ditch >10m": "RSDSDG10",
    "Roadside Severity: Other": "RSDSO",
    "Roadside Severity: Pole $\\leq$1m": "RSPSDL1",
    "Roadside Severity: Ditch $\\leq$5m": "RSDSDL5",
    "Roadside Severity: Pole $\\leq$10m": "RSPSDL10",
    "Roadside Severity: Ditch $\\leq$10m": "RSDSDL10",
    "Roadside Severity: Pole >10m": "RSPSDG10",
}


def parse_de_best_vars(tex_path):
    """Parse a DE best model .tex file and return the list of CSV column names."""
    import re
    if not os.path.exists(tex_path):
        print(f"  WARNING: {tex_path} not found, using all SEG_FEATS")
        return SEG_FEATS
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Extract variable names from lines like:  Annual Avg Daily Traffic (AADT) & +0.7242
    vars_found = []
    for line in text.split("\n"):
        m = re.match(r"\s+(.+?)\s*&\s*[+-]?\d", line)
        if m:
            name = m.group(1).strip()
            if name == "(Intercept)" or name.startswith("ln("):
                continue
            # clean LaTeX escapes
            name = name.replace("\\$", "").replace("\\textbackslash", "")
            csv_col = VAR_MAP.get(name)
            if csv_col:
                vars_found.append(csv_col)
            else:
                print(f"  WARNING: unmapped variable '{name}' in {tex_path}")
    if vars_found:
        print(f"  parsed {len(vars_found)} variables from {os.path.basename(tex_path)}")
        return vars_found
    print(f"  WARNING: no variables parsed from {tex_path}, using all SEG_FEATS")
    return SEG_FEATS


def fit_segment_nb(seg, var_list=None):
    """Fit NB2 GLM of HB_Total ~ segment geometry covariates."""
    if var_list is None:
        var_list = SEG_FEATS
    available = [c for c in var_list if c in seg.columns]
    if len(available) < 2:
        return None, available

    mask = seg[available + ["HB_Total", "log_len"]].notna().all(axis=1)
    seg_clean = seg.loc[mask].copy()
    if len(seg_clean) < len(available) + 10:
        print(f"  WARNING: only {len(seg_clean)} rows after dropping NaN")
        return None, available

    raw = seg_clean[available].values.astype(float)
    sd_x = raw.std(axis=0)
    keep = sd_x > 1e-12
    available = [c for c, k in zip(available, keep) if k]
    raw = raw[:, keep]
    mu_x = raw.mean(axis=0)
    sd_x = raw.std(axis=0)
    sd_x[sd_x < 1e-12] = 1.0
    X_std = (raw - mu_x) / sd_x
    X = sm_add_constant(X_std)
    y = seg_clean["HB_Total"].values.astype(float)
    off = seg_clean["log_len"].values.astype(float)

    # adjust loglik for offset: mu = exp(X@beta + offset)
    def nb_negll_offset(params, y, X, off):
        beta = params[:-1]
        log_alpha = params[-1]
        alpha = np.exp(log_alpha)
        eta = X @ beta + off
        mu = np.exp(eta)
        mu = np.clip(mu, 1e-10, 1e10)
        r = 1.0 / max(alpha, 1e-12)
        p = r / (r + mu)
        p = np.clip(p, 1e-10, 1.0 - 1e-10)
        ll = _nbinom.logpmf(y, r, p)
        return -np.sum(ll)

    # Poisson start
    try:
        beta0 = np.linalg.lstsq(X, np.log(y + 0.5) - off, rcond=None)[0]
    except Exception:
        beta0 = np.zeros(X.shape[1])
    mu0 = np.exp(X @ beta0 + off)
    alpha0 = max(float(np.var(y - mu0) / max(np.mean(mu0 ** 2), 1e-9)), 0.01)
    x0 = np.concatenate([beta0, [np.log(alpha0)]])

    # try L-BFGS-B first (faster for larger parameter sets), fall back to NM
    try:
        res = _minimize(nb_negll_offset, x0, args=(y, X, off),
                        method="L-BFGS-B",
                        options={"maxiter": 5000, "ftol": 1e-10})
    except Exception:
        res = _minimize(nb_negll_offset, x0, args=(y, X, off),
                        method="Nelder-Mead",
                        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})

    beta_hat = res.x[:-1]
    alpha_hat = np.exp(res.x[-1])
    mu_hat = np.exp(X @ beta_hat + off)

    # Hessian SE
    n = len(y)
    p = len(beta_hat)
    hess_diag = np.full(p + 1, np.nan)
    f0 = res.fun
    eps = 1e-5 * (1.0 + np.abs(res.x))
    for i in range(p + 1):
        pp = res.x.copy(); pp[i] += eps[i]
        pm = res.x.copy(); pm[i] -= eps[i]
        hess_diag[i] = (nb_negll_offset(pp, y, X, off)
                         - 2 * f0
                         + nb_negll_offset(pm, y, X, off)) / (eps[i] ** 2)
    se_beta = np.where(hess_diag[:-1] > 0, 1.0 / np.sqrt(hess_diag[:-1]), np.nan)
    z = beta_hat / np.where(se_beta > 0, se_beta, np.nan)
    pvals = 2.0 * sps.norm.sf(np.abs(z))

    ll = -res.fun
    k = p + 1
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    dev = 2.0 * np.sum(
        y * np.log(np.clip(y, 1e-10, None) / np.clip(mu_hat, 1e-10, None))
        - (y + alpha_hat) * np.log((y + alpha_hat) / (mu_hat + alpha_hat)))

    names = ["Intercept"] + available
    return {
        "params": beta_hat, "se": se_beta, "pvals": pvals,
        "alpha": alpha_hat, "llf": ll, "aic": aic, "bic": bic,
        "deviance": dev, "fittedvalues": mu_hat, "nobs": n,
        "df_model": p, "converged": res.success, "x_names": names,
    }, available


# =====================================================================
# 6. OUTPUT GENERATION
# =====================================================================

def _tex_escape(s):
    s = str(s)
    for k, v in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(k, v)
    return s


def write_latex(nb_temporal, nb_seg, seg_feats, daily, Q, R, panel, path):
    """Write paper-ready LaTeX table block."""
    lines = [
        "% Auto-generated by hb_state_space.py - table block only",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{State-space crash-risk model: NB2 GLM with latent "
        r"random-walk component.}\label{tab:state_space}",
        r"\begin{tabular}{lrr}\hline",
        r"\multicolumn{3}{l}{\textbf{Panel A: Temporal model (daily HB "
        r"counts)}}\\[-2pt]",
        r"\multicolumn{3}{l}{\textit{Response: daily HB event count across "
        r"spatial grid cells}}\\\hline",
        r"Parameter & Estimate & Std.\ error\\\hline",
    ]

    if nb_temporal is not None:
        for i in range(len(nb_temporal["params"])):
            name = nb_temporal["x_names"][i]
            b = nb_temporal["params"][i]
            se = nb_temporal["se"][i]
            pv = nb_temporal["pvals"][i]
            star = ("***" if pv < 0.001 else
                    "**" if pv < 0.01 else
                    "*" if pv < 0.05 else "")
            se_s = f"{se:.4f}" if np.isfinite(se) else "---"
            lines.append(f"{_tex_escape(name)} & {b:.4f} & {se_s}{star}\\\\")

    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{l}{\textbf{Panel B: Segment cross-sectional "
                 r"model (NB2)}}\\[-2pt]")
    lines.append(r"\multicolumn{3}{l}{\textit{Response: total HB events per "
                 r"segment, offset log(segment length), DE-selected covariates}}"
                 r"\\\hline")
    lines.append(r"Parameter & Estimate & Std.\ error\\\hline")

    if nb_seg is not None:
        for i in range(len(nb_seg["params"])):
            name = nb_seg["x_names"][i]
            b = nb_seg["params"][i]
            se = nb_seg["se"][i]
            pv = nb_seg["pvals"][i]
            star = ("***" if pv < 0.001 else
                    "**" if pv < 0.01 else
                    "*" if pv < 0.05 else "")
            se_s = f"{se:.4f}" if np.isfinite(se) else "---"
            lines.append(f"{_tex_escape(name)} & {b:.4f} & {se_s}{star}\\\\")

    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{l}{\textbf{Panel C: State-space "
                 r"components}}\\\hline")
    lines.append(f"State variance $\\sigma^2_{{\\theta}}$ & {Q:.4f} & ---\\\\")
    lines.append(f"Obs. variance $\\sigma^2_{{\\varepsilon}}$ & {R:.4f} & ---\\\\")
    n_dates = daily["date"].nunique()
    n_cells = panel["grid_id"].nunique()
    lines.append(f"Grid cells & {n_cells:,} & ---\\\\")
    lines.append(f"Observation days & {n_dates} & ---\\\\")
    lines.append(f"Total panel obs. & {len(panel):,} & ---\\\\")
    if nb_temporal is not None:
        lines.append(f"Deviance (temporal) & {nb_temporal['deviance']:.0f} & ---\\\\")
        lines.append(f"AIC (temporal) & {nb_temporal['aic']:.0f} & ---\\\\")
    if nb_seg is not None:
        lines.append(f"Deviance (segment) & {nb_seg['deviance']:.0f} & ---\\\\")
        lines.append(f"AIC (segment) & {nb_seg['aic']:.0f} & ---\\\\")

    lines += [
        r"\hline\end{tabular}",
        (r"\emph{Panel A: NB2 MLE of daily HB counts across spatial grid cells "
         r"with day-of-week fixed effects. Panel B: cross-sectional NB2 of total "
         r"HB events per segment on DE-selected covariates, offset log(segment "
         r"length). Panel C: Kalman-filter parameters for the latent crash-risk "
         r"random walk $\theta_t$. Significance: $^{*}p<0.05$, "
         r"$^{**}p<0.01$, $^{***}p<0.001$. Asymptotic standard errors.}"),
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  latex written: {os.path.basename(path)}")


def plot_trace(daily, path):
    """Plot latent crash-risk time series with uncertainty band."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    dates = pd.to_datetime(daily["date"])
    risk = daily["latent_risk"].values
    se = daily["latent_se"].values

    ax = axes[0]
    ax.fill_between(dates, risk - 1.96 * se, risk + 1.96 * se,
                    alpha=0.25, color="#4477AA", label=r"$\pm$ 95\% CI")
    ax.plot(dates, risk, lw=1.4, color="#4477AA",
            label=r"Latent risk $\hat{\theta}_t$")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_ylabel(r"Latent log-risk $\hat{\theta}_t$")
    ax.set_title("Estimated latent crash risk from hard-braking events "
                 "(Kalman filter)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.bar(dates, daily["total_hb"], width=0.8, color="#CC6677", alpha=0.7)
    ax2.set_ylabel("Daily HB count")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  trace plot written: {os.path.basename(path)}")


def plot_trace_hourly(hourly, path):
    """Plot hourly latent crash-risk time series with uncertainty band."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ts = pd.to_datetime(hourly["hour_ts"])
    risk = hourly["latent_risk"].values
    se = hourly["latent_se"].values

    ax = axes[0]
    ax.fill_between(ts, risk - 1.96 * se, risk + 1.96 * se,
                    alpha=0.25, color="#4477AA", label=r"$\pm$ 95\% CI")
    ax.plot(ts, risk, lw=0.6, color="#4477AA",
            label=r"Latent risk $\hat{\theta}_t$")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_ylabel(r"Latent log-risk $\hat{\theta}_t$ (deseasonalised)")
    ax.set_title("Estimated hourly latent crash risk from hard-braking events "
                 "(Kalman filter on calendar-deseasonalised counts)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.bar(ts, hourly["total_hb"], width=0.02, color="#CC6677", alpha=0.7)
    ax2.set_ylabel("Hourly HB count")
    ax2.set_xlabel("Hour")
    ax2.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  hourly trace plot written: {os.path.basename(path)}")


def plot_temporal(events, path):
    """Plot hour-of-day and day-of-week risk patterns."""
    fig = plt.figure(figsize=(10, 4.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    hod = events.groupby("hour").size()
    hod = hod / hod.max()
    ax1.bar(hod.index, hod.values, color="#4477AA", alpha=0.8)
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Relative HB rate")
    ax1.set_title("(a) Time-of-day pattern")
    ax1.set_xticks(range(0, 24, 3))
    ax1.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)], fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    ax2 = fig.add_subplot(gs[0, 1])
    dow = events.groupby("dow").size()
    dow = dow / dow.max()
    ax2.bar(dow.index, dow.values, color="#CC6677", alpha=0.8)
    ax2.set_xlabel("Day of week")
    ax2.set_ylabel("Relative HB rate")
    ax2.set_title("(b) Day-of-week pattern")
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(DOW_NAMES, fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Temporal patterns in hard-braking events", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  temporal plot written: {os.path.basename(path)}")


def write_prediction(panel, daily, path):
    """Write per-cell-period crash-risk predictions."""
    pred = panel[["grid_id", "date", "hb_count", "n_critical", "n_extreme",
                   "mean_decel", "grid_lat", "grid_lon"]].copy()
    daily_map = daily.set_index("date")["latent_risk"].to_dict()
    pred["latent_risk"] = pred["date"].map(daily_map)
    pred["predicted_count"] = np.exp(
        pred["latent_risk"] + panel["log_hb_c"].mean() + panel["log_hb"].mean())
    pred["residual"] = pred["hb_count"] - pred["predicted_count"]
    pred = pred.sort_values(["date", "grid_id"]).reset_index(drop=True)
    pred.to_csv(path, index=False)
    print(f"  prediction CSV written: {os.path.basename(path)} "
          f"({len(pred):,} rows)")


def write_summary(nb_temporal, nb_seg, daily, Q, R, panel, events, path,
                  hourly=None, Q_h=None, R_h=None):
    """Write summary statistics."""
    rows = [
        {"metric": "n_events_total", "value": len(events)},
        {"metric": "n_vehicles", "value": events["VehicleID"].nunique()},
        {"metric": "n_dates", "value": panel["date"].nunique()},
        {"metric": "n_grid_cells", "value": panel["grid_id"].nunique()},
        {"metric": "n_panel_obs", "value": len(panel)},
        {"metric": "mean_daily_hb",
         "value": panel.groupby("date")["hb_count"].sum().mean()},
        {"metric": "median_daily_hb",
         "value": panel.groupby("date")["hb_count"].sum().median()},
        {"metric": "std_daily_hb",
         "value": panel.groupby("date")["hb_count"].sum().std()},
        {"metric": "kalman_Q", "value": Q},
        {"metric": "kalman_R", "value": R},
        {"metric": "latent_range_min", "value": daily["latent_risk"].min()},
        {"metric": "latent_range_max", "value": daily["latent_risk"].max()},
        {"metric": "latent_std", "value": daily["latent_risk"].std()},
    ]
    if nb_temporal is not None:
        rows += [
            {"metric": "temporal_aic", "value": nb_temporal["aic"]},
            {"metric": "temporal_deviance", "value": nb_temporal["deviance"]},
            {"metric": "temporal_loglik", "value": nb_temporal["llf"]},
            {"metric": "temporal_nobs", "value": nb_temporal["nobs"]},
            {"metric": "temporal_converged", "value": nb_temporal["converged"]},
        ]
    if nb_seg is not None:
        rows += [
            {"metric": "segment_aic", "value": nb_seg["aic"]},
            {"metric": "segment_deviance", "value": nb_seg["deviance"]},
            {"metric": "segment_loglik", "value": nb_seg["llf"]},
            {"metric": "segment_nobs", "value": nb_seg["nobs"]},
            {"metric": "segment_converged", "value": nb_seg["converged"]},
        ]
    # temporal pattern stats
    hod = events.groupby("hour").size()
    rows.append({"metric": "peak_hour", "value": int(hod.idxmax())})
    rows.append({"metric": "trough_hour", "value": int(hod.idxmin())})
    dow = events.groupby("dow").size()
    rows.append({"metric": "peak_dow", "value": int(dow.idxmax())})
    rows.append({"metric": "trough_dow", "value": int(dow.idxmin())})
    # latent risk interpretation
    lat_min = daily["latent_risk"].min()
    lat_max = daily["latent_risk"].max()
    rows.append({"metric": "latent_pct_elevation_at_peak",
                 "value": round(100 * (np.exp(lat_max) - 1), 2)})
    rows.append({"metric": "latent_pct_depression_at_trough",
                 "value": round(100 * (np.exp(lat_min) - 1), 2)})
    if hourly is not None:
        rows += [
            {"metric": "n_hours", "value": len(hourly)},
            {"metric": "hour_min", "value": str(hourly["hour_ts"].min())},
            {"metric": "hour_max", "value": str(hourly["hour_ts"].max())},
            {"metric": "kalman_Q_h", "value": Q_h},
            {"metric": "kalman_R_h", "value": R_h},
            {"metric": "latent_range_min_h",
             "value": hourly["latent_risk"].min()},
            {"metric": "latent_range_max_h",
             "value": hourly["latent_risk"].max()},
            {"metric": "latent_std_h", "value": hourly["latent_risk"].std()},
            {"metric": "mean_hourly_hb", "value": hourly["total_hb"].mean()},
            {"metric": "seasonal_amp_h",
             "value": float(hourly["seasonal"].max() - hourly["seasonal"].min())},
        ]
    # severity
    for cat in ["Hard-Braking Candidate", "Critical-Likely", "Extreme"]:
        n = int((events["Category"] == cat).sum())
        tag = cat.lower().replace("-", "_").replace(" ", "_")
        rows.append({"metric": f"n_{tag}", "value": n})

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  summary CSV written: {os.path.basename(path)}")
    return df


# =====================================================================
# 7. MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="State-space crash-risk model from hard-braking events")
    ap.add_argument("--events",
                    default=os.path.join(DIR, "hard_braking_events.csv"),
                    help="raw HB events CSV")
    ap.add_argument("--segments",
                    default=os.path.join(DIR, "near_miss_hb_ecf.csv"),
                    help="segment-level geometry and HB aggregates")
    ap.add_argument("--grid-res", type=float, default=0.01,
                    help="spatial grid resolution in degrees (default 0.01)")
    ap.add_argument("--max-events", type=int, default=None,
                    help="cap number of events loaded (for memory)")
    ap.add_argument("--out-dir", default=DIR)
    ap.add_argument("--de-best-hb",
                    default=os.path.join(DIR, "hb_de_best_nm.tex"),
                    help="DE best HB model .tex file (for variable selection)")
    ap.add_argument("--de-best-crash",
                    default=os.path.join(DIR, "hb_de_best_crash.tex"),
                    help="DE best crash model .tex file (for variable selection)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("STATE-SPACE CRASH-RISK MODEL")
    print("=" * 60)

    # --- load data --------------------------------------------------
    events = load_events(args.events, nrows=args.max_events)
    segments = load_segments(args.segments)

    # --- build panel ------------------------------------------------
    panel = build_panel(events, grid_res=args.grid_res)

    # --- fit NB GLM with temporal FE --------------------------------
    print("\nfitting NB2 MLE (daily counts, DOW FE) ...")
    nb_temporal, X_temp = fit_nb_with_dow(panel)
    print(f"  alpha={nb_temporal['alpha']:.4f}, AIC={nb_temporal['aic']:.0f}, "
          f"deviance={nb_temporal['deviance']:.0f}, "
          f"converged={nb_temporal['converged']}")

    # --- Kalman filter ----------------------------------------------
    print("\napplying Kalman filter to daily log-counts ...")
    daily, Q, R = apply_kalman(panel)

    # --- Hourly Kalman filter (full-window latent risk at hour level) --
    print("\napplying Kalman filter to hourly log-counts ...")
    hourly = build_hourly(events, grid_res=args.grid_res)
    hourly, Q_h, R_h = apply_kalman_hourly(hourly)

    # --- segment cross-sectional model ------------------------------
    print("\nfitting segment NB2 MLE (HB_Total ~ DE-selected geometry) ...")
    hb_vars = parse_de_best_vars(args.de_best_hb)
    nb_seg, seg_feats = fit_segment_nb(segments, var_list=hb_vars)
    if nb_seg is not None:
        print(f"  AIC={nb_seg['aic']:.0f}, deviance={nb_seg['deviance']:.0f}, "
              f"features={len(seg_feats)}, converged={nb_seg['converged']}")

    # --- outputs ----------------------------------------------------
    print("\ngenerating outputs ...")
    write_latex(nb_temporal, nb_seg, seg_feats, daily, Q, R, panel,
                os.path.join(args.out_dir, "hb_state_space.tex"))
    plot_trace(daily, os.path.join(args.out_dir, "hb_state_space_trace.png"))
    plot_trace_hourly(hourly,
                      os.path.join(args.out_dir, "hb_state_space_trace_hourly.png"))
    plot_temporal(events,
                  os.path.join(args.out_dir, "hb_state_space_temporal.png"))
    write_prediction(panel, daily,
                     os.path.join(args.out_dir, "hb_state_space_prediction.csv"))
    summary = write_summary(nb_temporal, nb_seg, daily, Q, R, panel, events,
                            os.path.join(args.out_dir, "hb_state_space_summary.csv"),
                            hourly=hourly, Q_h=Q_h, R_h=R_h)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("\ndone.")


if __name__ == "__main__":
    main()
