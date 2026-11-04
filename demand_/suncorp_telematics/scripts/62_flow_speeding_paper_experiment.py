"""Paper experiment for conditional speeding counts and shared flow risk.

The primary outcome is ``trips_mean_gt_10``.  Its primary exposure is the
observed number of trips with a judged mean speed, entered as the fixed offset
``log(trips_with_judged_speed)``.  A ``log(trip_count)`` offset is fitted as a
declared sensitivity analysis.  The shared model is a penalized NB2 MAP model
with one latent segment effect entering both the flow and speeding equations.

This script deliberately contains no hurdle or zero-inflated model path.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, polygamma

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ = SCRIPT_DIR.parent
DATA = PROJ / "data"
RESULTS = PROJ / "results"
DEFAULT_PANEL = RESULTS / "segment_15min_panel.parquet"
DEFAULT_CALENDAR = DATA / "fixture_calendar.csv"
DEFAULT_OUT = RESULTS / "paper_flow_speeding"
TARGET = "trips_mean_gt_10"
PRIMARY_EXPOSURE = "trips_with_judged_speed"
SENSITIVITY_EXPOSURE = "trip_count"
TZ = "Australia/Brisbane"
WINDOW_HOURS = (-3.0, 4.0)
BINS_PER_HOUR = 4
READ_COLUMNS = [
    "road_osmid", "bin", "vehicle_count", "trip_count",
    "trips_with_judged_speed", TARGET,
]
NUMERIC_FEATURES = [
    "in_event", "log_att", "rel_hour", "rel_hour_sq", "rel_sin",
    "rel_cos", "pre_game", "post_game", "tod_sin", "tod_cos",
    "dow_sin", "dow_cos", "weekend", "dist_stad", "event_x_dist",
    "rel_x_dist", "att_x_dist", "ev_lanes", "lanes_z", "maxspeed_z",
    "curvature_z", "dist_inter_z", "node_degree_z", "log_length",
    "lanes_known",
]


def _load_game_helpers():
    path = SCRIPT_DIR / "59_game_day_search_predict.py"
    spec = importlib.util.spec_from_file_location("game_day_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load feature helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text or "missing"


def _timestamp_ns(value: pd.Timestamp) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(TZ)
    return int(stamp.tz_convert("UTC").value)


def load_calendar(path: Path) -> pd.DataFrame:
    calendar = pd.read_csv(path)
    calendar["ko"] = pd.to_datetime(calendar["kickoff_iso"]).dt.tz_localize(TZ)
    calendar["attendance"] = pd.to_numeric(
        calendar["attendance"], errors="coerce"
    ).fillna(0.0)
    calendar["label"] = calendar["label"].astype(str)
    return calendar.sort_values("ko").reset_index(drop=True)


def discover_panel_parts(panel: Path) -> list[Path]:
    if panel.name != "segment_15min_panel.parquet":
        return [panel]
    monthly = sorted(panel.parent.glob("segment_15min_panel_????-??.parquet"))
    return monthly or [panel]


def _fixture_bounds(calendar: pd.DataFrame) -> list[tuple[str, int, int, pd.Timestamp]]:
    bounds = []
    for row in calendar.itertuples(index=False):
        start = row.ko + pd.Timedelta(hours=WINDOW_HOURS[0])
        end = row.ko + pd.Timedelta(hours=WINDOW_HOURS[1])
        bounds.append((row.label, _timestamp_ns(start), _timestamp_ns(end), start))
    return bounds


def load_event_and_background(
    panel: Path,
    calendar: pd.DataFrame,
    background_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    bounds = _fixture_bounds(calendar)
    event_parts: list[pd.DataFrame] = []
    background_parts: list[pd.DataFrame] = []
    parts = discover_panel_parts(panel)
    per_part_background = max(0, int(math.ceil(background_rows / max(len(parts), 1))))

    for part_no, part in enumerate(parts, start=1):
        print(f"   reading panel part {part_no}/{len(parts)}: {part.name}")
        frame = pd.read_parquet(part, columns=READ_COLUMNS)
        frame["bin"] = pd.to_datetime(frame["bin"], utc=True)
        bin_ns = frame["bin"].astype("int64").to_numpy()
        covered = np.zeros(len(frame), dtype=bool)

        for label, start_ns, end_ns, start in bounds:
            mask = (bin_ns >= start_ns) & (bin_ns < end_ns)
            if not mask.any():
                continue
            covered |= mask
            selected = frame.loc[mask].copy()
            selected["fixture"] = label
            selected["rel_bin"] = (
                (selected["bin"] - start) / pd.Timedelta(minutes=15)
            ).round().astype(int).clip(0, 27)
            meta = calendar.loc[calendar["label"].eq(label)].iloc[0]
            selected["attendance"] = float(meta["attendance"])
            selected["fixture_type"] = str(meta.get("fixture_type", "") or "")
            selected["ko_hour"] = float(meta["ko"].hour) + float(meta["ko"].minute) / 60.0
            event_parts.append(selected)

        if per_part_background and (~covered).any():
            candidates = np.flatnonzero(~covered)
            n = min(per_part_background, len(candidates))
            picked = rng.choice(candidates, size=n, replace=False)
            background = frame.iloc[np.sort(picked)].copy()
            background["fixture"] = "__background__"
            background["rel_bin"] = -1
            background["attendance"] = 0.0
            background["fixture_type"] = ""
            background["ko_hour"] = 0.0
            background_parts.append(background)
        del frame
        gc.collect()

    if not event_parts:
        raise RuntimeError("no fixture-window rows were found in the panel")
    events = pd.concat(event_parts, ignore_index=True)
    background = (
        pd.concat(background_parts, ignore_index=True)
        if background_parts else pd.DataFrame(columns=READ_COLUMNS)
    )
    if len(background) > background_rows:
        background = background.sample(background_rows, random_state=seed)
    return events, background, [p.name for p in parts]


def _prepare_feature_frame(frame: pd.DataFrame, helpers, context: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = helpers.add_calendar(frame)
    is_event = frame["fixture"].ne("__background__")
    frame["in_event"] = is_event.astype(float)
    frame["rel_hour"] = 0.0
    frame.loc[is_event, "rel_hour"] = (
        frame.loc[is_event, "rel_bin"].to_numpy(float) / BINS_PER_HOUR
        + WINDOW_HOURS[0]
    )
    frame["log_att"] = np.log1p(pd.to_numeric(frame["attendance"], errors="coerce").fillna(0.0))
    frame["rel_hour_sq"] = 0.0
    frame["rel_sin"] = 0.0
    frame["rel_cos"] = 1.0
    frame["pre_game"] = 0.0
    frame["post_game"] = 0.0
    if is_event.any():
        event = frame.loc[is_event].copy()
        event = helpers.add_game_temporal(event)
        frame.loc[is_event, ["rel_hour_sq", "rel_sin", "rel_cos", "pre_game", "post_game"]] = event[
            ["rel_hour_sq", "rel_sin", "rel_cos", "pre_game", "post_game"]
        ].to_numpy()
    frame = helpers.add_static(frame, context)
    frame["event_x_dist"] = frame["in_event"] * frame["dist_stad"]
    frame = helpers.add_game_wide(frame, crowd_med=1e18)
    return frame


def build_design(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    design = pd.DataFrame(index=frame.index)
    for column in NUMERIC_FEATURES:
        values = frame[column] if column in frame.columns else pd.Series(0.0, index=frame.index)
        design[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)

    highway = frame.get("highway", pd.Series("unclassified", index=frame.index))
    highway = highway.astype("string").fillna("unclassified").astype(str)
    for value in sorted(set(highway)):
        if value == "unclassified":
            continue
        design[f"highway__{_safe_name(value)}"] = (highway == value).astype(float)

    fixture_type = frame.get("fixture_type", pd.Series("", index=frame.index))
    fixture_type = fixture_type.astype("string").fillna("").astype(str)
    for value in sorted(set(fixture_type)):
        if not value:
            continue
        design[f"fixture_type__{_safe_name(value)}"] = (fixture_type == value).astype(float)

    design = design.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    design.insert(0, "Intercept", 1.0)
    return design, list(design.columns)


def _sample_training(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    event = frame.loc[frame["fixture"].ne("__background__")]
    background = frame.loc[frame["fixture"].eq("__background__")]
    event_n = min(len(event), max(1, int(round(maximum * 0.9))))
    bg_n = min(len(background), maximum - event_n)
    event_sample = event.sample(event_n, random_state=seed) if event_n else event.iloc[[]]
    bg_sample = background.sample(bg_n, random_state=seed + 1) if bg_n else background.iloc[[]]
    selected = pd.concat([event_sample, bg_sample], ignore_index=True)
    if len(selected) < maximum:
        selected_sources = set(selected["_source_index"].tolist())
        remaining = frame.loc[~frame["_source_index"].isin(selected_sources)]
        if len(remaining):
            selected = pd.concat(
                [selected, remaining.sample(min(maximum - len(selected), len(remaining)), random_state=seed + 2)],
                ignore_index=True,
            )
    return selected.reset_index(drop=True)


def _nb2_parts(y: np.ndarray, eta: np.ndarray, log_alpha: float):
    y = np.asarray(y, dtype=float)
    eta = np.clip(np.asarray(eta, dtype=float), -30.0, 30.0)
    alpha = float(np.exp(np.clip(log_alpha, -12.0, 5.0)))
    mu = np.exp(eta)
    inv_alpha = 1.0 / alpha
    log_one = np.log1p(alpha * mu)
    ll = (
        gammaln(y + inv_alpha) - gammaln(inv_alpha) - gammaln(y + 1.0)
        + y * (np.log(alpha) + np.log(mu))
        - (y + inv_alpha) * log_one
    )
    denominator = 1.0 + alpha * mu
    score_eta = (y - mu) / denominator
    score_log_alpha = (
        (digamma(inv_alpha) - digamma(y + inv_alpha)) / alpha
        + y + log_one / alpha
        - (alpha * y + 1.0) * mu / denominator
    )
    return ll, score_eta, score_log_alpha, mu, alpha


def _nb2_initial(y: np.ndarray, offset: np.ndarray, p: int) -> np.ndarray:
    exposure = np.exp(np.clip(offset, -30.0, 30.0))
    intercept = np.log(max(float(np.sum(y)) / max(float(np.sum(exposure)), 1e-12), 1e-9))
    theta = np.zeros(p + 1, dtype=float)
    theta[0] = intercept
    theta[-1] = np.log(0.5)
    return theta


def fit_nb2(X: np.ndarray, y: np.ndarray, offset: np.ndarray, maxiter: int) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    p = X.shape[1]

    def objective(theta, with_gradient=True):
        eta = X @ theta[:p] + offset
        ll, score_eta, score_alpha, _, _ = _nb2_parts(y, eta, theta[-1])
        value = -float(np.sum(ll))
        if not with_gradient:
            return value
        gradient = np.empty_like(theta)
        gradient[:p] = -(X.T @ score_eta)
        gradient[-1] = -float(np.sum(score_alpha))
        return value, gradient

    bounds = [(-15.0, 15.0)] * p + [(-12.0, 5.0)]
    result = minimize(
        lambda theta: objective(theta, True),
        _nb2_initial(y, offset, p),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-10, "gtol": 1e-6, "maxls": 40},
    )
    theta = np.asarray(result.x, dtype=float)
    value = float(objective(theta, False))
    gradient_norm = float(np.max(np.abs(np.asarray(result.jac, dtype=float))))
    return {
        "beta": theta[:p],
        "log_alpha": float(theta[-1]),
        "alpha": float(np.exp(theta[-1])),
        "loglik": -value,
        "n_params": int(p + 1),
        "converged": bool(result.success or gradient_norm / max(len(y), 1) < 1e-5),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "gradient_norm": gradient_norm,
        "gradient_norm_per_observation": gradient_norm / max(len(y), 1),
        "bse": _nb2_bse(X, y, offset, theta),
    }


def _nb2_bse(X, y, offset, theta) -> np.ndarray:
    """Observed-information standard errors for the fixed-effects NB2 fit."""
    p = X.shape[1]
    beta = theta[:p]
    log_alpha = theta[-1]
    ll, score_eta, score_alpha, mu, alpha = _nb2_parts(y, X @ beta + offset, log_alpha)
    del ll, score_eta, score_alpha
    denominator = 1.0 + alpha * mu
    h_eta = -mu * (1.0 + alpha * y) / denominator ** 2
    h_cross = -(y - mu) * alpha * mu / denominator ** 2
    q = 1.0 / alpha
    h_alpha = (
        -q * (digamma(q) - digamma(y + q)
              + q * (polygamma(1, q) - polygamma(1, y + q)))
        - q * np.log1p(alpha * mu) + q * mu / denominator
        - alpha * mu * (y - mu) / denominator ** 2
    )
    hessian = np.zeros((p + 1, p + 1), dtype=float)
    hessian[:p, :p] = (X.T * h_eta) @ X
    cross = X.T @ h_cross
    hessian[:p, -1] = cross
    hessian[-1, :p] = cross
    hessian[-1, -1] = float(np.sum(h_alpha))
    try:
        covariance = np.linalg.pinv(-hessian, rcond=1e-10)
        diagonal = np.clip(np.diag(covariance), 0.0, np.inf)
        return np.sqrt(diagonal)
    except Exception:
        return np.full(p + 1, np.nan)


def predict_nb2(model: dict, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.asarray(X) @ model["beta"] + offset, -30.0, 30.0))


def fit_shared_latent(
    X: np.ndarray,
    y_flow: np.ndarray,
    y_speed: np.ndarray,
    speed_offset: np.ndarray,
    segment_codes: np.ndarray,
    flow_start: dict,
    speed_start: dict,
    maxiter: int,
) -> dict:
    X = np.asarray(X, dtype=float)
    y_flow = np.asarray(y_flow, dtype=float)
    y_speed = np.asarray(y_speed, dtype=float)
    speed_offset = np.asarray(speed_offset, dtype=float)
    segment_codes = np.asarray(segment_codes, dtype=np.int64)
    p = X.shape[1]
    n_segments = int(segment_codes.max()) + 1 if len(segment_codes) else 0
    flow_pred_start = predict_nb2(flow_start, X, np.zeros(len(X)))
    residual = np.log((y_flow + 1.0) / (flow_pred_start + 1.0))
    initial_u = np.zeros(n_segments, dtype=float)
    for code in range(n_segments):
        values = residual[segment_codes == code]
        initial_u[code] = float(np.clip(np.mean(values) if len(values) else 0.0, -2.0, 2.0))
    initial_u -= float(np.mean(initial_u)) if len(initial_u) else 0.0
    theta0 = np.concatenate([
        flow_start["beta"], speed_start["beta"],
        [flow_start["log_alpha"], speed_start["log_alpha"], 0.5],
        initial_u,
    ])

    def unpack(theta):
        return (
            theta[:p], theta[p:2 * p], theta[2 * p], theta[2 * p + 1],
            theta[2 * p + 2], theta[2 * p + 3:],
        )

    def objective(theta, with_gradient=True):
        beta_flow, beta_speed, log_af, log_as, loading, latent = unpack(theta)
        eta_flow = X @ beta_flow + latent[segment_codes]
        eta_speed = X @ beta_speed + speed_offset + loading * latent[segment_codes]
        ll_flow, ge_flow, ga_flow, _, _ = _nb2_parts(y_flow, eta_flow, log_af)
        ll_speed, ge_speed, ga_speed, _, _ = _nb2_parts(y_speed, eta_speed, log_as)
        prior = -0.5 * float(np.sum(latent ** 2))
        value = -float(np.sum(ll_flow) + np.sum(ll_speed) + prior)
        if not with_gradient:
            return value
        gradient = np.empty_like(theta)
        gradient[:p] = -(X.T @ ge_flow)
        gradient[p:2 * p] = -(X.T @ ge_speed)
        gradient[2 * p] = -float(np.sum(ga_flow))
        gradient[2 * p + 1] = -float(np.sum(ga_speed))
        gradient[2 * p + 2] = -float(np.sum(ge_speed * latent[segment_codes]))
        gradient[2 * p + 3:] = -(
            np.bincount(
                segment_codes,
                weights=ge_flow + loading * ge_speed,
                minlength=n_segments,
            ) - latent
        )
        return value, gradient

    bounds = (
        [(-15.0, 15.0)] * (2 * p)
        + [(-12.0, 5.0), (-12.0, 5.0), (-5.0, 5.0)]
        + [(-10.0, 10.0)] * n_segments
    )
    result = minimize(
        lambda theta: objective(theta, True),
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-9, "gtol": 1e-5, "maxls": 40, "maxcor": 20},
    )
    theta = np.asarray(result.x, dtype=float)
    beta_flow, beta_speed, log_af, log_as, loading, latent = unpack(theta)
    gradient_norm = float(np.max(np.abs(np.asarray(result.jac, dtype=float))))
    return {
        "beta_flow": beta_flow,
        "beta_speed": beta_speed,
        "log_alpha_flow": float(log_af),
        "log_alpha_speed": float(log_as),
        "alpha_flow": float(np.exp(log_af)),
        "alpha_speed": float(np.exp(log_as)),
        "latent_sd": 1.0,
        "loading": float(loading),
        "latent": latent,
        "penalized_loglik": -float(objective(theta, False)),
        "n_params": int(2 * p + 3 + n_segments),
        "converged": bool(result.success or gradient_norm / max(len(y_flow), 1) < 1e-4),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "gradient_norm": gradient_norm,
        "gradient_norm_per_observation": gradient_norm / max(len(y_flow), 1),
        "stationarity_tolerance_per_observation": 1e-4,
    }


def predict_shared(model: dict, X: np.ndarray, offset: np.ndarray, codes: np.ndarray):
    latent = np.asarray(model["latent"])
    safe_codes = np.asarray(codes, dtype=np.int64)
    valid = (safe_codes >= 0) & (safe_codes < len(latent))
    latent_values = np.zeros(len(safe_codes), dtype=float)
    latent_values[valid] = latent[safe_codes[valid]]
    flow = np.exp(np.clip(np.asarray(X) @ model["beta_flow"] + latent_values, -30.0, 30.0))
    speed = np.exp(np.clip(
        np.asarray(X) @ model["beta_speed"] + offset
        + model["loading"] * latent_values, -30.0, 30.0
    ))
    return flow, speed


def spearman(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if mask.sum() < 10:
        return float("nan")
    a = pd.Series(observed[mask]).rank().to_numpy(float).copy()
    b = pd.Series(predicted[mask]).rank().to_numpy(float).copy()
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / max(np.sqrt(np.dot(a, a) * np.dot(b, b)), 1e-12))


def top_overlap(observed, predicted) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    k = max(10, int(round(0.05 * len(observed))))
    if len(observed) < k or k <= 0:
        return float("nan")
    left = set(np.argsort(-observed, kind="stable")[:k].tolist())
    right = set(np.argsort(-predicted, kind="stable")[:k].tolist())
    return float(len(left & right) / max(len(left | right), 1))


def metric_rows(observed, predicted, aggregation: str, model: str, outcome: str, group=None):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if group is not None:
        grouped = pd.DataFrame({"group": group, "observed": observed, "predicted": predicted})
        grouped = grouped.groupby("group", as_index=False)[["observed", "predicted"]].sum()
        observed = grouped["observed"].to_numpy(float)
        predicted = grouped["predicted"].to_numpy(float)
    error = predicted - observed
    slope, intercept = (np.polyfit(predicted, observed, 1)
                         if len(observed) >= 10 and np.var(predicted) > 0 else (np.nan, np.nan))
    return {
        "model": model,
        "outcome": outcome,
        "aggregation": aggregation,
        "n": int(len(observed)),
        "observed_sum": float(np.sum(observed)),
        "predicted_sum": float(np.sum(predicted)),
        "spearman": spearman(observed, predicted),
        "rmse": float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan,
        "mae": float(np.mean(np.abs(error))) if len(error) else np.nan,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "top_5pct_jaccard": top_overlap(observed, predicted),
    }


def coefficient_rows(
    model_name: str,
    terms: list[str],
    beta: np.ndarray,
    bse: np.ndarray | None,
    alpha: float | None,
    equation: str = "speeding",
):
    rows = []
    for i, term in enumerate(terms):
        estimate = float(beta[i])
        se = float(bse[i]) if bse is not None and i < len(bse) else np.nan
        z_value = estimate / se if np.isfinite(se) and se > 0 else np.nan
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0)) if np.isfinite(z_value) else np.nan
        rows.append({"model": model_name, "equation": equation, "term": term,
                     "estimate": estimate, "std_error": se, "z_value": z_value,
                     "p_value": p_value, "alpha": alpha})
    return rows


def write_tex(path: Path, comparison: pd.DataFrame, audit: dict) -> None:
    lines = [
        "% Generated by 62_flow_speeding_paper_experiment.py",
        "\\begin{tabular}{llrrrr}",
        "Model & Outcome & Spearman & RMSE & MAE & Top-5\\% Jaccard\\\\",
        "\\hline",
    ]
    for row in comparison.itertuples(index=False):
        if row.aggregation != "row":
            continue
        lines.append(
            f"{row.model} & {row.outcome} & {row.spearman:.3f} & "
            f"{row.rmse:.3f} & {row.mae:.3f} & {row.top_5pct_jaccard:.3f}\\\\"
        )
    lines += ["\\end{tabular}", "", "% Primary offset: log(trips_with_judged_speed), fixed coefficient 1.",
              f"% Held-out fixtures: {', '.join(audit['holdout_fixtures'])}"]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--holdout-fixtures", type=int, default=6)
    parser.add_argument("--background-rows", type=int, default=250000)
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="0 keeps all training rows; pilot uses a smaller cap")
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--shared-maxiter", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pilot:
        args.background_rows = min(args.background_rows, 10000)
        args.max_train_rows = args.max_train_rows or 30000
        args.maxiter = min(args.maxiter, 80)
        args.shared_maxiter = min(args.shared_maxiter, 60)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("[62] loading calendar and panel windows")
    calendar = load_calendar(Path(args.calendar))
    if args.holdout_fixtures < 1 or args.holdout_fixtures >= len(calendar):
        raise SystemExit("--holdout-fixtures must leave at least one training fixture")
    holdout = set(calendar["label"].tail(args.holdout_fixtures).tolist())
    helpers = _load_game_helpers()
    context = helpers.static_context()
    events, background, input_parts = load_event_and_background(
        Path(args.panel), calendar, int(args.background_rows), int(args.seed)
    )
    frame = pd.concat([events, background], ignore_index=True)
    del events, background
    frame = _prepare_feature_frame(frame, helpers, context)
    design, terms = build_design(frame)
    frame = frame.reset_index(drop=True)
    frame["_source_index"] = np.arange(len(frame), dtype=np.int64)
    design = design.reset_index(drop=True)
    is_test = frame["fixture"].isin(holdout).to_numpy()
    train_frame = _sample_training(frame.loc[~is_test].copy(), int(args.max_train_rows), int(args.seed))
    train_index = train_frame["_source_index"].to_numpy(dtype=np.int64)
    train_frame = train_frame.reset_index(drop=True)
    train_design_frame = design.loc[train_index].reset_index(drop=True)
    variable_columns = [
        column for column in train_design_frame.columns
        if column == "Intercept" or float(train_design_frame[column].std()) > 1e-12
    ]
    design = design[variable_columns]
    design_scales = {}
    for column in variable_columns:
        if column == "Intercept":
            continue
        center = float(train_design_frame[column].mean())
        scale = float(train_design_frame[column].std())
        scale = scale if np.isfinite(scale) and scale > 1e-8 else 1.0
        design[column] = (design[column] - center) / scale
        design_scales[column] = {"center": center, "scale": scale}
    terms = list(design.columns)
    train_design = design.loc[train_index].reset_index(drop=True).to_numpy(float)
    test_frame = frame.loc[is_test].reset_index(drop=True)
    test_design = design.loc[np.flatnonzero(is_test)].reset_index(drop=True).to_numpy(float)

    train_y = pd.to_numeric(train_frame[TARGET], errors="coerce").fillna(0.0).to_numpy(float)
    test_y = pd.to_numeric(test_frame[TARGET], errors="coerce").fillna(0.0).to_numpy(float)
    train_flow = pd.to_numeric(train_frame["trip_count"], errors="coerce").fillna(0.0).to_numpy(float)
    test_flow = pd.to_numeric(test_frame["trip_count"], errors="coerce").fillna(0.0).to_numpy(float)
    train_primary_offset = np.log(np.maximum(
        pd.to_numeric(train_frame[PRIMARY_EXPOSURE], errors="coerce").fillna(0.0).to_numpy(float), 1e-6
    ))
    test_primary_offset = np.log(np.maximum(
        pd.to_numeric(test_frame[PRIMARY_EXPOSURE], errors="coerce").fillna(0.0).to_numpy(float), 1e-6
    ))
    train_sensitivity_offset = np.log(np.maximum(
        pd.to_numeric(train_frame[SENSITIVITY_EXPOSURE], errors="coerce").fillna(0.0).to_numpy(float), 1e-6
    ))
    test_sensitivity_offset = np.log(np.maximum(
        pd.to_numeric(test_frame[SENSITIVITY_EXPOSURE], errors="coerce").fillna(0.0).to_numpy(float), 1e-6
    ))
    segment_values = pd.concat([train_frame["road_osmid"], test_frame["road_osmid"]]).astype(str)
    segment_codes_all, segment_levels = pd.factorize(segment_values, sort=True)
    train_codes = segment_codes_all[:len(train_frame)]
    test_codes = segment_codes_all[len(train_frame):]

    print(f"   training rows={len(train_frame):,}; held-out rows={len(test_frame):,}")
    print(f"   training fixtures={train_frame.loc[train_frame.fixture != '__background__', 'fixture'].nunique()}")
    print(f"   held-out fixtures={sorted(holdout)}")
    print(f"   primary offset={PRIMARY_EXPOSURE}; sensitivity={SENSITIVITY_EXPOSURE}")

    flow_fit = fit_nb2(train_design, train_flow, np.zeros(len(train_frame)), int(args.maxiter))
    flow_train_pred = predict_nb2(flow_fit, train_design, np.zeros(len(train_frame)))
    flow_test_pred = predict_nb2(flow_fit, test_design, np.zeros(len(test_frame)))
    primary_fit = fit_nb2(train_design, train_y, train_primary_offset, int(args.maxiter))
    primary_pred = predict_nb2(primary_fit, test_design, test_primary_offset)
    sensitivity_fit = fit_nb2(train_design, train_y, train_sensitivity_offset, int(args.maxiter))
    sensitivity_pred = predict_nb2(sensitivity_fit, test_design, test_sensitivity_offset)
    shared_fit = fit_shared_latent(
        train_design, train_flow, train_y, train_primary_offset, train_codes,
        flow_fit, primary_fit, int(args.shared_maxiter)
    )
    shared_flow_train, shared_speed_train = predict_shared(
        shared_fit, train_design, train_primary_offset, train_codes
    )
    shared_flow_test, shared_speed_test = predict_shared(
        shared_fit, test_design, test_primary_offset, test_codes
    )

    predictions = test_frame[["fixture", "road_osmid", "bin", "rel_bin"]].copy()
    predictions["observed_speeding"] = test_y
    predictions["observed_trip_count"] = test_flow
    predictions["trips_with_judged_speed"] = np.exp(test_primary_offset).clip(0.0, None)
    predictions["pred_speeding_judged_offset"] = primary_pred
    predictions["pred_speeding_trip_offset"] = sensitivity_pred
    predictions["pred_flow_nb2"] = flow_test_pred
    predictions["pred_flow_shared_latent"] = shared_flow_test
    predictions["pred_speeding_shared_latent"] = shared_speed_test
    predictions.to_parquet(out / "predictions.parquet", index=False)

    comparison_rows = []
    for model_name, outcome, observed, predicted in [
        ("flow_nb2", "trip_count", test_flow, flow_test_pred),
        ("conditional_nb2_judged_offset", "speeding", test_y, primary_pred),
        ("conditional_nb2_trip_offset", "speeding", test_y, sensitivity_pred),
        ("shared_latent_nb2", "trip_count", test_flow, shared_flow_test),
        ("shared_latent_nb2", "speeding", test_y, shared_speed_test),
    ]:
        comparison_rows.append(metric_rows(observed, predicted, "row", model_name, outcome))
        comparison_rows.append(metric_rows(
            observed, predicted, "segment", model_name, outcome,
            group=test_frame["road_osmid"].to_numpy(),
        ))
    comparison = pd.DataFrame(comparison_rows)
    fit_metadata = {
        ("flow_nb2", "trip_count"): {
            "train_loglik": flow_fit["loglik"], "fit_alpha": flow_fit["alpha"],
            "fit_converged": flow_fit["converged"], "fit_iterations": flow_fit["iterations"],
            "fit_criterion": "NB2", "fit_aic": -2.0 * flow_fit["loglik"] + 2.0 * flow_fit["n_params"],
            "fit_bic": -2.0 * flow_fit["loglik"] + flow_fit["n_params"] * np.log(max(len(train_frame), 1)),
        },
        ("conditional_nb2_judged_offset", "speeding"): {
            "train_loglik": primary_fit["loglik"], "fit_alpha": primary_fit["alpha"],
            "fit_converged": primary_fit["converged"], "fit_iterations": primary_fit["iterations"],
            "fit_criterion": "NB2", "fit_aic": -2.0 * primary_fit["loglik"] + 2.0 * primary_fit["n_params"],
            "fit_bic": -2.0 * primary_fit["loglik"] + primary_fit["n_params"] * np.log(max(len(train_frame), 1)),
        },
        ("conditional_nb2_trip_offset", "speeding"): {
            "train_loglik": sensitivity_fit["loglik"], "fit_alpha": sensitivity_fit["alpha"],
            "fit_converged": sensitivity_fit["converged"], "fit_iterations": sensitivity_fit["iterations"],
            "fit_criterion": "NB2", "fit_aic": -2.0 * sensitivity_fit["loglik"] + 2.0 * sensitivity_fit["n_params"],
            "fit_bic": -2.0 * sensitivity_fit["loglik"] + sensitivity_fit["n_params"] * np.log(max(len(train_frame), 1)),
        },
        ("shared_latent_nb2", "trip_count"): {
            "train_loglik": shared_fit["penalized_loglik"], "fit_alpha": shared_fit["alpha_flow"],
            "fit_converged": shared_fit["converged"], "fit_iterations": shared_fit["iterations"],
            "fit_criterion": "MAP_pseudo", "fit_aic": -2.0 * shared_fit["penalized_loglik"] + 2.0 * shared_fit["n_params"],
            "fit_bic": -2.0 * shared_fit["penalized_loglik"] + shared_fit["n_params"] * np.log(max(len(train_frame), 1)),
            "latent_loading": shared_fit["loading"],
        },
        ("shared_latent_nb2", "speeding"): {
            "train_loglik": shared_fit["penalized_loglik"], "fit_alpha": shared_fit["alpha_speed"],
            "fit_converged": shared_fit["converged"], "fit_iterations": shared_fit["iterations"],
            "fit_criterion": "MAP_pseudo", "fit_aic": -2.0 * shared_fit["penalized_loglik"] + 2.0 * shared_fit["n_params"],
            "fit_bic": -2.0 * shared_fit["penalized_loglik"] + shared_fit["n_params"] * np.log(max(len(train_frame), 1)),
            "latent_loading": shared_fit["loading"],
        },
    }
    metadata_columns = ["train_loglik", "fit_alpha", "fit_converged", "fit_iterations",
                        "fit_criterion", "fit_aic", "fit_bic", "latent_loading"]
    for column in metadata_columns:
        comparison[column] = [
            fit_metadata.get((row.model, row.outcome), {}).get(column, np.nan)
            for row in comparison.itertuples(index=False)
        ]
    comparison.to_csv(out / "model_comparison.csv", index=False)

    coefficients = []
    coefficients.extend(coefficient_rows(
        "flow_nb2", terms, flow_fit["beta"], flow_fit["bse"], flow_fit["alpha"], equation="flow"
    ))
    coefficients.extend(coefficient_rows(
        "conditional_nb2_judged_offset", terms, primary_fit["beta"], primary_fit["bse"], primary_fit["alpha"]
    ))
    coefficients.extend(coefficient_rows(
        "conditional_nb2_trip_offset", terms, sensitivity_fit["beta"], sensitivity_fit["bse"], sensitivity_fit["alpha"]
    ))
    for equation, beta, alpha in [
        ("flow", shared_fit["beta_flow"], shared_fit["alpha_flow"]),
        ("speeding", shared_fit["beta_speed"], shared_fit["alpha_speed"]),
    ]:
        for i, term in enumerate(terms):
            coefficients.append({"model": "shared_latent_nb2", "equation": equation,
                                 "term": term, "estimate": float(beta[i]),
                                 "std_error": np.nan, "z_value": np.nan,
                                 "p_value": np.nan, "alpha": alpha})
    coefficients += [
        {"model": "shared_latent_nb2", "equation": "shared", "term": "latent_loading",
         "estimate": shared_fit["loading"], "std_error": np.nan, "z_value": np.nan,
         "p_value": np.nan, "alpha": np.nan},
        {"model": "shared_latent_nb2", "equation": "shared", "term": "latent_sd",
         "estimate": shared_fit["latent_sd"], "std_error": np.nan, "z_value": np.nan,
         "p_value": np.nan, "alpha": np.nan},
    ]
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)

    audit = {
        "script": "62_flow_speeding_paper_experiment.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_panel_parts": input_parts,
        "calendar": str(args.calendar),
        "target": TARGET,
        "primary_exposure": PRIMARY_EXPOSURE,
        "sensitivity_exposure": SENSITIVITY_EXPOSURE,
        "offset_definition": "log(max(exposure, 1e-6)); coefficient fixed at one",
        "holdout_fixtures": sorted(holdout),
        "n_train_rows": int(len(train_frame)),
        "n_test_rows": int(len(test_frame)),
        "n_train_segments": int(train_frame["road_osmid"].nunique()),
        "n_test_segments": int(test_frame["road_osmid"].nunique()),
        "feature_terms": terms,
        "design_standardized": True,
        "design_scales": design_scales,
        "excluded_families": ["hurdle", "zero_inflated"],
        "conditional_judged": {k: v for k, v in primary_fit.items() if k not in {"beta", "bse"}},
        "conditional_trip_offset": {k: v for k, v in sensitivity_fit.items() if k not in {"beta", "bse"}},
        "flow_nb2": {k: v for k, v in flow_fit.items() if k not in {"beta", "bse"}},
        "shared_latent": {k: v for k, v in shared_fit.items() if k not in {"beta_flow", "beta_speed", "latent"}},
        "runtime_seconds": float(time.time() - started),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="ascii")
    write_tex(out / "model_comparison.tex", comparison, audit)
    print(f"[62] wrote paper artifacts to {out}")
    print(comparison.loc[comparison["aggregation"].eq("row")].to_string(index=False))


if __name__ == "__main__":
    main()