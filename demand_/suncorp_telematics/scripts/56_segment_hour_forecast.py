"""
56_segment_hour_forecast.py - all-hours per-segment flow + speeding forecasts.

Two-stage MetaCount design (scales to 27M segment-hour rows):

  Stage A (segment baselines, ~7.5k rows):
      total vehicles / total speeding(>5km/h) per segment over TRAIN hours,
      NB random-parameters via metacountregressor (geometry + highway +
      log observed-hours exposure). Gives hourly flow_base[i] and a
      model-smoothed speeding propensity[i].
  Stage B (network time model, ~11k hourly rows):
      network totals per clock hour with calendar cyclicals (hour-of-day,
      day-of-week, month), trend and lag1/lag24 network flows, NB fixed
      effects via metacountregressor. Gives 1-step network forecasts and
      recursive multi-hour forecasts from the current network state.
  Stage C (combine):
      pred_flow(i,t) = flow_base[i] * net_flow(t)/mean_net_flow_train
      pred_spd(i,t)  = pred_flow(i,t) * propensity[i] *
                       (net_spd_rate(t)/train_spd_rate)

Targets use the panel from script 54 --build-panel:
  vehicle_count          = distinct Compass vehicles per segment-hour
  trips_mean_gt_5        = trips whose mean speed exceeds posted+5 km/h
  (strict >, equal-vehicle weighting; see summarize_segment_hours).

Split is strictly chronological (first 80% of clock hours train, last 20%
test); nothing from the future enters any feature. The panel only contains
observed segment-hours, so zeros-vs-unobserved is NOT resolved here:
metrics are conditional on observed rows and this is stated in the audit.

Usage:
  Train + validate (pilot, minutes):
    python scripts/56_segment_hour_forecast.py --pilot
  Full run (~25 min):
    python scripts/56_segment_hour_forecast.py
  Predict next N hours from a cutoff using observed history:
    python scripts/56_segment_hour_forecast.py --predict --cutoff "2026-03-20 18:00" --n-hours 6
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacount_nb import (  # noqa: E402
    fit_metacount_nb,
    predict_metacount,
    prepare_metacount_frame,
)

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RES = PROJ / "results"
FIG = PROJ / "figures"
PANEL = RES / "segment_hour_panel.parquet"
GEOM = DATA / "road_geometry_features.csv"
ECF = DATA / "near_miss_hb_ecf.csv"
SPD_COL = "trips_mean_gt_5"
FLOW_COL = "vehicle_count"
TZ = "Australia/Brisbane"
SHRINK_K = 50.0


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 10:
        return float("nan")
    ra = np.array(pd.Series(a).rank(), dtype=float)
    rb = np.array(pd.Series(b).rank(), dtype=float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / max(denom, 1e-12))


def _jaccard_topk(obs, pred, k):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    k = int(min(k, len(obs)))
    if k <= 0:
        return float("nan")
    top_obs = set(np.argsort(-obs)[:k].tolist())
    top_pred = set(np.argsort(-pred)[:k].tolist())
    return float(len(top_obs & top_pred) / max(len(top_obs | top_pred), 1))


def _calibration(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    if len(obs) < 10 or np.var(pred) <= 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(pred, obs, 1)
    return float(slope), float(intercept)


def _calendar(frame):
    out = frame.copy()
    hod = out["hour"].dt.hour + out["hour"].dt.minute / 60.0
    dow = out["hour"].dt.dayofweek
    month = out["hour"].dt.month
    out["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    out["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["weekend"] = (dow >= 5).astype(float)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out


def _standardise(train, test, cols):
    params = {}
    for col in cols:
        mean = float(train[col].mean())
        std = float(train[col].std())
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        params[col] = (mean, std)
        train[col + "_z"] = (train[col] - mean) / std
        test[col + "_z"] = (test[col] - mean) / std
    return params


def _apply_standardise(frame, params):
    for col, (mean, std) in params.items():
        frame[col + "_z"] = (frame[col] - mean) / std
    return frame


def split_hours(panel_path, test_frac=0.2, max_hours=None):
    import pyarrow.parquet as pq

    hours = pq.read_table(str(panel_path), columns=["hour"]).to_pandas()["hour"]
    hours = pd.to_datetime(hours, utc=True).dt.tz_convert(TZ)
    uniq = np.sort(hours.unique())
    if max_hours is not None:
        uniq = uniq[-int(max_hours):]
    cut = uniq[int(len(uniq) * (1.0 - test_frac))]
    return pd.Timestamp(cut), uniq.min(), uniq.max(), len(uniq)


def aggregate_panel(panel_path, train_end, max_segments=None, min_hour=None):
    import pyarrow.parquet as pq

    seg_flow, seg_spd, seg_n = {}, {}, {}
    hour_flow, hour_spd, hour_nseg = {}, {}, {}
    parquet = pq.ParquetFile(str(panel_path))
    keep = None
    for batch in parquet.iter_batches(batch_size=500_000,
                                      columns=["road_osmid", "hour", FLOW_COL, SPD_COL]):
        df = batch.to_pandas()
        df["hour"] = pd.to_datetime(df["hour"], utc=True).dt.tz_convert(TZ)
        if min_hour is not None:
            df = df[df["hour"] >= min_hour]
            if df.empty:
                continue
        train_mask = df["hour"] <= train_end
        for is_train, seg_f, seg_s, seg_c in (
                (True, seg_flow, seg_spd, seg_n), (False, None, None, None)):
            part = df[train_mask] if is_train else df[~train_mask]
            if part.empty:
                continue
            grouped = part.groupby("road_osmid").agg(
                flow=(FLOW_COL, "sum"), spd=(SPD_COL, "sum"), n=("hour", "size"))
            store = (seg_flow, seg_spd, seg_n) if is_train else None
            if store is not None:
                for seg, row in grouped.iterrows():
                    seg = int(seg)
                    seg_flow[seg] = seg_flow.get(seg, 0) + int(row["flow"])
                    seg_spd[seg] = seg_spd.get(seg, 0) + int(row["spd"])
                    seg_n[seg] = seg_n.get(seg, 0) + int(row["n"])
        grouped_h = df.groupby("hour").agg(
            flow=(FLOW_COL, "sum"), spd=(SPD_COL, "sum"), nseg=("road_osmid", "nunique"))
        for hour, row in grouped_h.iterrows():
            key = pd.Timestamp(hour).tz_convert(TZ)
            hour_flow[key] = hour_flow.get(key, 0) + int(row["flow"])
            hour_spd[key] = hour_spd.get(key, 0) + int(row["spd"])
            hour_nseg[key] = hour_nseg.get(key, 0) + int(row["nseg"])
    seg = pd.DataFrame({"road_osmid": list(seg_flow.keys()),
                        "train_flow": [seg_flow[k] for k in seg_flow],
                        "train_spd": [seg_spd.get(k, 0) for k in seg_flow],
                        "train_hours": [seg_n.get(k, 0) for k in seg_flow]})
    if max_segments is not None:
        seg = seg.nlargest(int(max_segments), "train_flow")
        keep = set(seg["road_osmid"].tolist())
    hourly = pd.DataFrame({"hour": list(hour_flow.keys()),
                           "net_flow": [hour_flow[k] for k in hour_flow],
                           "net_spd": [hour_spd.get(k, 0) for k in hour_flow],
                           "n_seg": [hour_nseg.get(k, 0) for k in hour_flow]})
    hourly = hourly.sort_values("hour").reset_index(drop=True)
    return seg, hourly, keep


def build_hourly_features(hourly, train_end):
    full_idx = pd.date_range(hourly["hour"].min(), hourly["hour"].max(),
                             freq="h", tz=TZ)
    full = pd.DataFrame({"hour": full_idx})
    full = full.merge(hourly, on="hour", how="left")
    for col in ("net_flow", "net_spd", "n_seg"):
        full[col] = full[col].fillna(0).astype(float)
    full["trend_days"] = ((full["hour"] - full["hour"].min()).dt.total_seconds()
                          / 86400.0)
    full = _calendar(full)
    full["lag1_flow"] = full["net_flow"].shift(1)
    full["lag24_flow"] = full["net_flow"].shift(24)
    full["lag1_spd"] = full["net_spd"].shift(1)
    full["lag24_spd"] = full["net_spd"].shift(24)
    full = full.dropna().reset_index(drop=True)
    full["split"] = np.where(full["hour"] <= train_end, "train", "test")
    return full


def fit_stage_a(seg, rdm_scale="full"):
    seg = seg.copy()
    seg["n_hours_obs"] = seg["train_hours"].clip(lower=1)
    geom = pd.read_csv(GEOM)
    ecf = pd.read_csv(ECF, usecols=["road_osmid", "highway", "dist_stadium_km"])
    seg = seg.merge(geom, on="road_osmid", how="left")
    seg = seg.merge(ecf, on="road_osmid", how="left")
    for col in ("maxspeed_kmh", "lanes", "curvature_deg_km", "dist_to_intersection_m",
                "sinuosity", "node_degree", "length_m", "dist_stadium_km"):
        if col in seg.columns:
            seg[col] = seg[col].fillna(float(seg[col].median()))
    seg["highway"] = seg["highway"].fillna("unknown").astype(str)
    for col in ("maxspeed_kmh", "lanes", "curvature_deg_km", "dist_to_intersection_m"):
        std = float(seg[col].std())
        seg[col + "_z"] = ((seg[col] - float(seg[col].mean()))
                           / (std if std and std > 0 else 1.0))
    mc, dummies = prepare_metacount_frame(seg, exposure_col="n_hours_obs",
                                          highway_col="highway")
    fixed = ["logexp"] + dummies + ["maxspeed_kmh_z", "curvature_deg_km_z"]
    rdm = ["lanes_z:normal", "dist_to_intersection_m_z:normal"]
    if rdm_scale == "pilot":
        flow_info = fit_metacount_nb(mc, y_col="train_flow", id_col="road_osmid",
                                     fixed_terms=fixed, rdm_terms=[],
                                     model="nb", R=50, de_maxiter=4,
                                     de_popsize=6, verbose=True)
        spd_info = fit_metacount_nb(mc, y_col="train_spd", id_col="road_osmid",
                                    fixed_terms=fixed, rdm_terms=[],
                                    model="nb", R=50, de_maxiter=4,
                                    de_popsize=6, verbose=True)
    else:
        flow_info = fit_metacount_nb(mc, y_col="train_flow", id_col="road_osmid",
                                     fixed_terms=fixed, rdm_terms=rdm,
                                     model="nb", R=200, de_maxiter=8,
                                     de_popsize=8, verbose=True)
        spd_info = fit_metacount_nb(mc, y_col="train_spd", id_col="road_osmid",
                                    fixed_terms=fixed, rdm_terms=rdm,
                                    model="nb", R=200, de_maxiter=8,
                                    de_popsize=8, verbose=True)
    seg["flow_base_total"] = np.maximum(np.asarray(flow_info["expected"]), 1e-9)
    seg["spd_base_total"] = np.maximum(np.asarray(spd_info["expected"]), 1e-9)
    seg["flow_base"] = seg["flow_base_total"] / seg["n_hours_obs"]
    seg["spd_base"] = seg["spd_base_total"] / seg["n_hours_obs"]
    return seg, flow_info, spd_info, fixed, rdm


def fit_stage_b(full):
    train = full[full["split"] == "train"].copy()
    test = full[full["split"] == "test"].copy()
    _standardise(train, test, ["trend_days", "lag1_flow", "lag24_flow"])
    _standardise(train, test, ["lag1_spd", "lag24_spd"])
    train["hour_id"] = np.arange(len(train))
    test["hour_id"] = np.arange(len(test)) + len(train)
    flow_fixed = ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "weekend",
                  "month_sin", "month_cos", "trend_days_z", "lag1_flow_z",
                  "lag24_flow_z"]
    spd_fixed = ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "weekend",
                 "month_sin", "month_cos", "trend_days_z", "lag1_spd_z",
                 "lag24_spd_z"]
    flow_info = fit_metacount_nb(train, y_col="net_flow", id_col="hour_id",
                                 fixed_terms=flow_fixed, rdm_terms=[],
                                 model="nb", R=100, de_maxiter=8,
                                 de_popsize=8, verbose=True)
    spd_info = fit_metacount_nb(train, y_col="net_spd", id_col="hour_id",
                                fixed_terms=spd_fixed, rdm_terms=[],
                                model="nb", R=100, de_maxiter=8,
                                de_popsize=8, verbose=True)
    return train, test, flow_info, spd_info, flow_fixed, spd_fixed


def _cap_lin(value, cap):
    hi = float(np.log(max(cap, 1e-9))) if cap else 20.0
    return min(max(float(value), -20.0), hi)


def recursive_forecast(history, flow_bundle, spd_bundle, n_hours,
                       cap_flow=None, cap_spd=None):
    last = history.sort_values("hour").tail(48).copy()
    flow_rows, spd_rows = [], []
    flow_hist = last[["hour", "net_flow"]].copy()
    spd_hist = last[["hour", "net_spd"]].copy()
    for step in range(1, int(n_hours) + 1):
        nxt = flow_hist["hour"].max() + pd.Timedelta(hours=1)
        frame = pd.DataFrame({"hour": [nxt]})
        frame = _calendar(frame)
        frame["trend_days"] = float(
            (nxt - history["hour"].min()).total_seconds() / 86400.0)
        lag1_f = float(flow_hist.set_index("hour").reindex([nxt - pd.Timedelta(hours=1)])[
            "net_flow"].fillna(flow_bundle["lag_fallback"]).iloc[0])
        lag24_f = float(flow_hist.set_index("hour").reindex([nxt - pd.Timedelta(hours=24)])[
            "net_flow"].fillna(flow_bundle["lag_fallback"]).iloc[0])
        lag1_s = float(spd_hist.set_index("hour").reindex([nxt - pd.Timedelta(hours=1)])[
            "net_spd"].fillna(spd_bundle["lag_fallback"]).iloc[0])
        lag24_s = float(spd_hist.set_index("hour").reindex([nxt - pd.Timedelta(hours=24)])[
            "net_spd"].fillna(spd_bundle["lag_fallback"]).iloc[0])
        frame["lag1_flow_z"] = (lag1_f - flow_bundle["means"]["lag1_flow"]) / flow_bundle[
            "stds"]["lag1_flow"]
        frame["lag24_flow_z"] = (lag24_f - flow_bundle["means"]["lag24_flow"]) / flow_bundle[
            "stds"]["lag24_flow"]
        frame["trend_days_z"] = (frame["trend_days"] - flow_bundle["means"]["trend_days"]
                                 ) / flow_bundle["stds"]["trend_days"]
        lin_f = flow_bundle["intercept"]
        for col in flow_bundle["coefs"]:
            lin_f += flow_bundle["coefs"][col] * float(frame[col].iloc[0])
        pred_f = float(np.exp(_cap_lin(lin_f, cap_flow)))
        if cap_flow:
            pred_f = min(pred_f, float(cap_flow))
        frame["lag1_spd_z"] = (lag1_s - spd_bundle["means"]["lag1_spd"]) / spd_bundle[
            "stds"]["lag1_spd"]
        frame["lag24_spd_z"] = (lag24_s - spd_bundle["means"]["lag24_spd"]) / spd_bundle[
            "stds"]["lag24_spd"]
        frame["trend_days_z"] = (frame["trend_days"] - spd_bundle["means"]["trend_days"]
                                 ) / spd_bundle["stds"]["trend_days"]
        lin_s = spd_bundle["intercept"]
        for col in spd_bundle["coefs"]:
            lin_s += spd_bundle["coefs"][col] * float(frame[col].iloc[0])
        pred_s = float(np.exp(_cap_lin(lin_s, cap_spd)))
        pred_s = min(pred_s, float(cap_spd) if cap_spd else pred_s, pred_f)
        flow_rows.append((nxt, pred_f))
        spd_rows.append((nxt, pred_s))
        flow_hist = pd.concat([flow_hist, pd.DataFrame({"hour": [nxt], "net_flow": [pred_f]})],
                              ignore_index=True)
        spd_hist = pd.concat([spd_hist, pd.DataFrame({"hour": [nxt], "net_spd": [pred_s]})],
                             ignore_index=True)
    flow_fc = pd.DataFrame(flow_rows, columns=["hour", "pred_net_flow"])
    spd_fc = pd.DataFrame(spd_rows, columns=["hour", "pred_net_spd"])
    return flow_fc.merge(spd_fc, on="hour")


def _bundle(fit_info, fixed, means_cols):
    beta = dict(fit_info.get("beta") or {})
    coefs = {col: float(beta.get(col, 0.0)) for col in fixed}
    return {"intercept": float(beta.get("__INTERCEPT__", 0.0)), "coefs": coefs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--out-dir", default=str(RES))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-segments", type=int, default=None)
    ap.add_argument("--max-hours", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--n-hours", type=int, default=6)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (FIG).mkdir(parents=True, exist_ok=True)

    if args.predict:
        bundle_path = Path(args.bundle or (out_dir / "segment_hour_forecast_bundle.json"))
        bundle = json.loads(bundle_path.read_text())
        if not args.cutoff:
            raise SystemExit("--predict requires --cutoff 'YYYY-MM-DD HH:MM'")
        cutoff = pd.Timestamp(args.cutoff)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(TZ)
        else:
            cutoff = cutoff.tz_convert(TZ)
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(str(args.panel))
        hour_flow, hour_spd = {}, {}
        for batch in parquet.iter_batches(batch_size=500_000,
                                          columns=["hour", FLOW_COL, SPD_COL]):
            df = batch.to_pandas()
            df["hour"] = pd.to_datetime(df["hour"], utc=True).dt.tz_convert(TZ)
            df = df[df["hour"] <= cutoff]
            if df.empty:
                continue
            grouped = df.groupby("hour").agg(flow=(FLOW_COL, "sum"), spd=(SPD_COL, "sum"))
            for hour, row in grouped.iterrows():
                key = pd.Timestamp(hour).tz_convert(TZ)
                hour_flow[key] = hour_flow.get(key, 0) + int(row["flow"])
                hour_spd[key] = hour_spd.get(key, 0) + int(row["spd"])
        history = pd.DataFrame({"hour": list(hour_flow.keys()),
                                "net_flow": [hour_flow[k] for k in hour_flow],
                                "net_spd": [hour_spd.get(k, 0) for k in hour_flow]})
        history = history.sort_values("hour").reset_index(drop=True)
        if len(history) < 48:
            raise SystemExit(f"only {len(history)} history hours before cutoff; need >= 48")
        forecast = recursive_forecast(history, bundle["flow_time"], bundle["spd_time"],
                                      args.n_hours,
                                      cap_flow=bundle.get("cap_flow"),
                                      cap_spd=bundle.get("cap_spd"))
        recal_f = bundle.get("recal_flow") or {"slope": 1.0, "intercept": 0.0}
        recal_s = bundle.get("recal_spd") or {"slope": 1.0, "intercept": 0.0}
        forecast["pred_net_flow"] = np.maximum(
            recal_f["slope"] * forecast["pred_net_flow"] + recal_f["intercept"], 1e-9)
        forecast["pred_net_spd"] = np.maximum(
            recal_s["slope"] * forecast["pred_net_spd"] + recal_s["intercept"], 1e-9)
        base = pd.DataFrame(bundle["segments"])
        rows = []
        for _, fc in forecast.iterrows():
            flow_factor = (float(fc["pred_net_flow"]) / bundle["mean_net_flow_train"]
                           * bundle["calib_flow"])
            rate = float(fc["pred_net_spd"]) / max(float(fc["pred_net_flow"]), 1e-9)
            rate_factor = rate / bundle["train_spd_rate"] * bundle["calib_spd"]
            for _, seg in base.iterrows():
                pred_flow = float(seg["flow_base"]) * flow_factor
                share = min(max(float(seg["propensity"]) * rate_factor, 0.0), 1.0)
                pred_spd = pred_flow * share
                rows.append((int(seg["road_osmid"]), fc["hour"], pred_flow, pred_spd))
        pred = pd.DataFrame(rows, columns=["road_osmid", "hour", "pred_vehicles",
                                           "pred_speeding_gt5"])
        out_path = Path(args.out or (out_dir / "segment_hour_next_hours.csv"))
        pred.to_csv(out_path, index=False)
        print(f"[56] wrote {len(pred):,} segment-hour predictions -> {out_path}")
        return

    max_hours = 720 if args.pilot else args.max_hours
    max_segments = 300 if args.pilot else args.max_segments
    rdm_scale = "pilot" if args.pilot else "full"
    tag = "pilot" if args.pilot else "full"

    print("[56/1] splitting hours (chronological)...")
    train_end, hour_min, hour_max, n_hours = split_hours(args.panel, args.test_frac,
                                                         max_hours)
    print(f"   hours={n_hours} range={hour_min}..{hour_max} train_end={train_end}")
    print("[56/2] aggregating panel (chunked)...")
    seg, hourly, keep = aggregate_panel(args.panel, train_end, max_segments, hour_min)
    print(f"   segments={len(seg)} hourly_rows={len(hourly)}")
    audit = {"tag": tag, "train_end": str(train_end), "n_hours": int(n_hours),
             "n_segments": int(len(seg)), "test_frac": float(args.test_frac)}

    print("[56/3] Stage A: segment baselines via metacountregressor...")
    seg, flow_a, spd_a, fixed_a, rdm_a = fit_stage_a(seg, rdm_scale)
    audit["stage_a"] = {"flow": {"aic": flow_a["aic"], "bic": flow_a["bic"],
                                 "alpha": flow_a["alpha"], "spec": flow_a["spec_label"]},
                        "spd": {"aic": spd_a["aic"], "bic": spd_a["bic"],
                                "alpha": spd_a["alpha"], "spec": spd_a["spec_label"]}}
    train_spd_rate = float(seg["train_spd"].sum() / max(seg["train_flow"].sum(), 1))
    raw_prop = ((seg["spd_base"] / seg["flow_base"].clip(lower=1e-9))
                / max(train_spd_rate, 1e-12))
    shrink_w = seg["train_flow"] / (seg["train_flow"] + SHRINK_K)
    seg["propensity"] = (shrink_w * raw_prop + (1.0 - shrink_w) * 1.0).clip(lower=0.0,
                                                                            upper=10.0)
    audit["train_spd_rate"] = train_spd_rate

    print("[56/4] Stage B: network time model via metacountregressor...")
    full = build_hourly_features(hourly, train_end)
    train, test, flow_b, spd_b, flow_fixed, spd_fixed = fit_stage_b(full)
    audit["stage_b"] = {"flow": {"aic": flow_b["aic"], "alpha": flow_b["alpha"]},
                        "spd": {"aic": spd_b["aic"], "alpha": spd_b["alpha"]}}
    train["pred_flow"] = predict_metacount(train, flow_b)
    test["pred_flow"] = predict_metacount(test, flow_b)
    train["pred_spd"] = predict_metacount(train, spd_b)
    test["pred_spd"] = predict_metacount(test, spd_b)
    calib_flow = float(train["net_flow"].sum() / max(train["pred_flow"].sum(), 1e-9))
    calib_spd = float(train["net_spd"].sum() / max(train["pred_spd"].sum(), 1e-9))
    slope_flow, icept_flow = _calibration(train["net_flow"], train["pred_flow"])
    slope_spd, icept_spd = _calibration(train["net_spd"], train["pred_spd"])
    if not np.isfinite(slope_flow):
        slope_flow, icept_flow = 1.0, 0.0
    if not np.isfinite(slope_spd):
        slope_spd, icept_spd = 1.0, 0.0
    train["pred_flow"] = np.maximum(slope_flow * train["pred_flow"] + icept_flow, 1e-9)
    train["pred_spd"] = np.maximum(slope_spd * train["pred_spd"] + icept_spd, 1e-9)
    test["pred_flow"] = np.maximum(slope_flow * test["pred_flow"] + icept_flow, 1e-9)
    test["pred_spd"] = np.maximum(slope_spd * test["pred_spd"] + icept_spd, 1e-9)
    audit["calib_flow"] = calib_flow
    audit["calib_spd"] = calib_spd
    audit["recal_flow"] = {"slope": slope_flow, "intercept": icept_flow}
    audit["recal_spd"] = {"slope": slope_spd, "intercept": icept_spd}

    mean_flow_train = float(train["net_flow"].mean())
    metrics = {
        "hourly_flow_1step": {
            "spearman": _spearman(test["net_flow"], test["pred_flow"]),
            "rmse": float(np.sqrt(np.mean((test["net_flow"] - test["pred_flow"]) ** 2))),
            "mae": float(np.mean(np.abs(test["net_flow"] - test["pred_flow"]))),
            "slope": _calibration(test["net_flow"], test["pred_flow"])[0],
        },
        "hourly_spd_1step": {
            "spearman": _spearman(test["net_spd"], test["pred_spd"]),
            "rmse": float(np.sqrt(np.mean((test["net_spd"] - test["pred_spd"]) ** 2))),
            "mae": float(np.mean(np.abs(test["net_spd"] - test["pred_spd"]))),
            "slope": _calibration(test["net_spd"], test["pred_spd"])[0],
        },
    }
    metrics["segment_rate_rank"] = {"note": "baseline rate vs train rate (in-sample rank)"}

    print("[56/5] granular test sample (chunked, observed rows only)...")
    import pyarrow.parquet as pq

    flow_base_map = seg.set_index("road_osmid")["flow_base"].to_dict()
    prop_map = seg.set_index("road_osmid")["propensity"].to_dict()
    test_hour_flow = test.set_index("hour")["pred_flow"].to_dict()
    test_hour_spd = test.set_index("hour")["pred_spd"].to_dict()
    obs_f, pred_f, obs_s, pred_s = [], [], [], []
    stride = 7
    parquet = pq.ParquetFile(str(args.panel))
    for batch in parquet.iter_batches(batch_size=500_000,
                                      columns=["road_osmid", "hour", FLOW_COL, SPD_COL]):
        df = batch.to_pandas()
        df["hour"] = pd.to_datetime(df["hour"], utc=True).dt.tz_convert(TZ)
        df = df[(df["hour"] > train_end) & (df["hour"] >= hour_min)]
        if keep is not None:
            df = df[df["road_osmid"].isin(keep)]
        if df.empty:
            continue
        df = df.iloc[::stride]
        df["flow_factor"] = (df["hour"].map(test_hour_flow).fillna(mean_flow_train)
                             / mean_flow_train * calib_flow)
        df["pf"] = df["road_osmid"].map(flow_base_map).fillna(0.0) * df["flow_factor"]
        rate = (df["hour"].map(test_hour_spd).fillna(0.0)
                / df["hour"].map(test_hour_flow).fillna(1.0).clip(lower=1e-9))
        share = (df["road_osmid"].map(prop_map).fillna(1.0)
                 * (rate / max(train_spd_rate, 1e-12) * calib_spd)).clip(lower=0.0,
                                                                         upper=1.0)
        df["ps"] = df["pf"] * share
        obs_f.extend(df[FLOW_COL].tolist())
        pred_f.extend(df["pf"].tolist())
        obs_s.extend(df[SPD_COL].tolist())
        pred_s.extend(df["ps"].tolist())
        if len(obs_f) > 400_000:
            break
    metrics["segment_hour_flow"] = {
        "n_sample": len(obs_f),
        "spearman": _spearman(obs_f, pred_f),
        "rmse": float(np.sqrt(np.mean((np.asarray(obs_f) - np.asarray(pred_f)) ** 2))),
        "mae": float(np.mean(np.abs(np.asarray(obs_f) - np.asarray(pred_f)))),
        "slope": _calibration(obs_f, pred_f)[0],
        "top100_jaccard": _jaccard_topk(obs_f, pred_f, 100),
    }
    metrics["segment_hour_spd"] = {
        "n_sample": len(obs_s),
        "spearman": _spearman(obs_s, pred_s),
        "rmse": float(np.sqrt(np.mean((np.asarray(obs_s) - np.asarray(pred_s)) ** 2))),
        "mae": float(np.mean(np.abs(np.asarray(obs_s) - np.asarray(pred_s)))),
        "slope": _calibration(obs_s, pred_s)[0],
        "top100_jaccard": _jaccard_topk(obs_s, pred_s, 100),
    }

    print("[56/6] rolling next-24h origins...")
    origins = test["hour"].iloc[[-72, -48, -24]].tolist() if len(test) >= 96 else [
        test["hour"].iloc[-1]]
    horizon_rmse_f, horizon_rmse_s = {}, {}
    flow_means = {c.replace("_z", ""): float(train[c.replace("_z", "")].mean())
                  for c in flow_fixed if c.endswith("_z")}
    flow_stds = {c.replace("_z", ""): float(train[c.replace("_z", "")].std())
                 for c in flow_fixed if c.endswith("_z")}
    spd_means = {c.replace("_z", ""): float(train[c.replace("_z", "")].mean())
                 for c in spd_fixed if c.endswith("_z")}
    spd_stds = {c.replace("_z", ""): float(train[c.replace("_z", "")].std())
                for c in spd_fixed if c.endswith("_z")}
    flow_bundle = _bundle(flow_b, flow_fixed, None)
    flow_bundle.update({"means": flow_means, "stds": flow_stds,
                        "lag_fallback": float(train["net_flow"].mean())})
    spd_bundle = _bundle(spd_b, spd_fixed, None)
    spd_bundle.update({"means": spd_means, "stds": spd_stds,
                       "lag_fallback": float(train["net_spd"].mean())})
    cap_flow = float(train["net_flow"].max() * 3.0)
    cap_spd = float(train["net_spd"].max() * 3.0)
    for origin in origins:
        hist = full[full["hour"] <= origin][["hour", "net_flow", "net_spd"]].copy()
        if len(hist) < 48:
            continue
        fc = recursive_forecast(hist, flow_bundle, spd_bundle, 24,
                                cap_flow=cap_flow, cap_spd=cap_spd)
        fc["pred_net_flow"] = np.maximum(slope_flow * fc["pred_net_flow"] + icept_flow,
                                         1e-9)
        fc["pred_net_spd"] = np.maximum(slope_spd * fc["pred_net_spd"] + icept_spd, 1e-9)
        actual = full[(full["hour"] > origin)].head(24)[["hour", "net_flow", "net_spd"]]
        merged = actual.merge(fc, on="hour")
        if merged.empty:
            continue
        for _, row in merged.iterrows():
            horizon = int((row["hour"] - origin).total_seconds() // 3600)
            horizon_rmse_f.setdefault(horizon, []).append(
                (row["net_flow"] - row["pred_net_flow"]) ** 2)
            horizon_rmse_s.setdefault(horizon, []).append(
                (row["net_spd"] - row["pred_net_spd"]) ** 2)
    metrics["rolling_24h"] = {
        "flow_rmse_h1": float(np.sqrt(np.mean(horizon_rmse_f.get(1, [np.nan])))),
        "flow_rmse_h6": float(np.sqrt(np.mean(horizon_rmse_f.get(6, [np.nan])))),
        "flow_rmse_h24": float(np.sqrt(np.mean(horizon_rmse_f.get(24, [np.nan])))),
        "spd_rmse_h1": float(np.sqrt(np.mean(horizon_rmse_s.get(1, [np.nan])))),
        "spd_rmse_h6": float(np.sqrt(np.mean(horizon_rmse_s.get(6, [np.nan])))),
        "spd_rmse_h24": float(np.sqrt(np.mean(horizon_rmse_s.get(24, [np.nan])))),
    }

    print("[56/7] writing outputs...")
    seg_out = seg[["road_osmid", "train_flow", "train_spd", "train_hours",
                   "flow_base", "spd_base", "propensity"]].copy()
    seg_out.to_csv(out_dir / f"segment_hour_forecast_baseline_{tag}.csv", index=False)
    train_out = train[["hour", "net_flow", "pred_flow", "net_spd", "pred_spd"]].copy()
    train_out["split"] = "train"
    test_out = test[["hour", "net_flow", "pred_flow", "net_spd", "pred_spd"]].copy()
    test_out["split"] = "test"
    pd.concat([train_out, test_out], ignore_index=True).to_csv(
        out_dir / f"segment_hour_forecast_hourly_{tag}.csv", index=False)
    (out_dir / f"segment_hour_forecast_metrics_{tag}.json").write_text(
        json.dumps({"metrics": metrics, "audit": audit}, indent=2, default=str))
    bundle = {
        "segments": seg_out.to_dict(orient="records"),
        "flow_time": flow_bundle,
        "spd_time": spd_bundle,
        "mean_net_flow_train": mean_flow_train,
        "mean_net_spd_train": float(train["net_spd"].mean()),
        "train_spd_rate": train_spd_rate,
        "calib_flow": calib_flow,
        "calib_spd": calib_spd,
        "recal_flow": {"slope": float(slope_flow), "intercept": float(icept_flow)},
        "recal_spd": {"slope": float(slope_spd), "intercept": float(icept_spd)},
        "cap_flow": cap_flow,
        "cap_spd": cap_spd,
        "train_end": str(train_end),
    }
    (out_dir / f"segment_hour_forecast_bundle_{tag}.json").write_text(json.dumps(bundle))
    with open(out_dir / f"segment_hour_forecast_{tag}.tex", "w") as tex:
        tex.write("% segment-hour forecast validation (" + tag + ")\n")
        for key, val in metrics.items():
            tex.write(f"% {key}: {val}\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample = test.tail(24 * 14)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(sample["hour"], sample["net_flow"], label="observed")
        axes[0].plot(sample["hour"], sample["pred_flow"], label="predicted")
        axes[0].set_ylabel("vehicles/hour (network)")
        axes[0].legend()
        axes[1].plot(sample["hour"], sample["net_spd"], label="observed")
        axes[1].plot(sample["hour"], sample["pred_spd"], label="predicted")
        axes[1].set_ylabel("speeding trips/hour (network)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIG / f"segment_hour_forecast_timeseries_{tag}.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - plots must not kill the run
        print(f"   WARN: plot failed ({exc})")
    print(f"[56] DONE ({tag}). metrics: {json.dumps(metrics, default=str)}")


if __name__ == "__main__":
    main()
