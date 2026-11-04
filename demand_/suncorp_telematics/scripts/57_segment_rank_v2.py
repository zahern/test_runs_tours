"""
57_segment_rank_v2.py - segment-hour ranking refinement (Stage C) + top segments.

Goal: rank WHERE speeding is likely to occur (top predicted segments), not
hotspot detection. Stage C refines the script-56 two-stage predictions with
a directly fitted segment-hour count model using richer features:

  own history   : previous-observed flow/speed/speeding + hour-gap + trailing
                  means (no clock-grid imputation; gaps are explicit features)
  network flows : neighbour-segment lagged means from OSM endpoint adjacency
                  (sparse matvec per hour-chunk; strictly causal, t-1 only)
  speed state   : lagged mean vehicle speed (congestion -> speeding link)
  events        : fixture window flag + attendance + event x stadium-distance
  context       : hard-braking history rate, posted limit, calendar cyclicals
  anchor        : script-56 combined prediction (log scale)

Targets: vehicle_count, trips_mean_gt_5 (strict >, equal-vehicle weighting).

Compass is ~5% market share (--penetration): fitting and RANKING use
Compass counts only; penetration scales absolute network estimates in
reporting outputs (ranking is invariant under uniform penetration).

Split reuses the script-56 bundle train_end (chronological; no refit of A/B).

Live Compass feed: NOT available; wiring this to a live feed (rebuild lags
from the feed, call --predict per cycle) is noted as future work in the audit.

Usage:
  python scripts/57_segment_rank_v2.py --pilot
  python scripts/57_segment_rank_v2.py --score-full   # + full-panel v2 pred parquet
  python scripts/57_segment_rank_v2.py --predict --cutoff "2026-03-20 18:00" --n-hours 6
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
from metacount_nb import fit_metacount_nb  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RES = PROJ / "results"
FIG = PROJ / "figures"
PANEL = RES / "segment_hour_panel.parquet"
BUNDLE56 = RES / "segment_hour_forecast_bundle_full.json"
HOURLY56 = RES / "segment_hour_forecast_hourly_full.csv"
GEOJSON = DATA / "osm_roads_suncorp.geojson"
ECF = DATA / "near_miss_hb_ecf.csv"
FIXTURES = DATA / "fixture_calendar.csv"
FLOW_COL = "vehicle_count"
SPD_COL = "trips_mean_gt_5"
SPEED_COL = "mean_vehicle_speed_kmh"
TZ = "Australia/Brisbane"
EPS = 1e-9


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
    top_obs = set(np.argsort(-obs, kind="stable")[:k].tolist())
    top_pred = set(np.argsort(-pred, kind="stable")[:k].tolist())
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


def build_adjacency(geojson_path):
    import geopandas as gpd

    roads = gpd.read_file(geojson_path)
    roads["osmid"] = pd.to_numeric(roads["osmid"], errors="coerce")
    roads = roads.dropna(subset=["osmid"])
    segs = roads["osmid"].astype(np.int64).to_numpy()
    point_map = {}
    for seg, geom in zip(segs, roads.geometry.to_numpy()):
        if geom.geom_type != "LineString":
            continue
        coords = list(geom.coords)
        for end in (coords[0], coords[-1]):
            key = (round(float(end[0]), 5), round(float(end[1]), 5))
            point_map.setdefault(key, set()).add(int(seg))
    adjacency = {}
    for seg in segs:
        adjacency[int(seg)] = set()
    for members in point_map.values():
        for seg in members:
            adjacency[seg] |= (members - {seg})
    n_isolated = sum(1 for v in adjacency.values() if not v)
    return adjacency, n_isolated


def load_fixtures(path):
    cal = pd.read_csv(path)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"]).dt.tz_localize(TZ)
    cal["attendance"] = pd.to_numeric(cal["attendance"], errors="coerce").fillna(0.0)
    return cal[["ko", "attendance"]].copy()


def fixture_features(hours, fixtures):
    col = pd.to_datetime(hours, utc=True)
    if not isinstance(col, pd.Series):
        col = pd.Series(col)
    if col.dt.tz is None:
        col = col.dt.tz_localize(TZ)
    else:
        col = col.dt.tz_convert(TZ)
    hours = pd.DatetimeIndex(col)
    in_event = np.zeros(len(hours), dtype=float)
    att = np.zeros(len(hours), dtype=float)
    for _, row in fixtures.iterrows():
        start = row["ko"] - pd.Timedelta(hours=3)
        end = row["ko"] + pd.Timedelta(hours=4)
        mask = (hours >= start) & (hours < end)
        in_event[mask] = 1.0
        att[mask] = np.maximum(att[mask], float(row["attendance"]))
    return in_event, np.log1p(att)


def build_lag_frame(panel_path, min_hour=None):
    import pyarrow.parquet as pq

    cols = ["road_osmid", "hour", FLOW_COL, SPEED_COL, SPD_COL,
            "trips_with_judged_speed", "trip_count"]
    parts = []
    parquet = pq.ParquetFile(str(panel_path))
    for batch in parquet.iter_batches(batch_size=500_000, columns=cols):
        df = batch.to_pandas()
        df["hour"] = pd.to_datetime(df["hour"], utc=True).dt.tz_convert(TZ)
        if min_hour is not None:
            df = df[df["hour"] >= min_hour]
            if df.empty:
                continue
        parts.append(df)
    frame = pd.concat(parts, ignore_index=True)
    frame["road_osmid"] = frame["road_osmid"].astype(np.int64)
    frame["flow"] = frame[FLOW_COL].astype(np.int64)
    frame["spd"] = frame[SPD_COL].astype(np.int64)
    frame["trips"] = frame["trip_count"].astype(float).clip(lower=1.0)
    frame["judged_share"] = frame["trips_with_judged_speed"].astype(float) / frame["trips"]
    global_speed = float(frame[SPEED_COL].median())
    seg_med = frame.groupby("road_osmid")[SPEED_COL].transform("median")
    frame["speed"] = frame[SPEED_COL].fillna(seg_med).fillna(global_speed).astype(float)
    frame = frame.sort_values(["road_osmid", "hour"]).reset_index(drop=True)
    grouped = frame.groupby("road_osmid", sort=False)
    frame["prev_hour"] = grouped["hour"].shift(1)
    frame["gap_hours"] = ((frame["hour"] - frame["prev_hour"]).dt.total_seconds()
                          / 3600.0)
    frame["prev_flow"] = grouped["flow"].shift(1)
    frame["prev_spd"] = grouped["spd"].shift(1)
    frame["prev_speed"] = grouped["speed"].shift(1)
    frame["prev_judged"] = grouped["judged_share"].shift(1)
    frame["trail5_flow"] = grouped["flow"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    frame["trail20_spd"] = grouped["spd"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    frame["has_hist"] = frame["prev_hour"].notna().astype(float)
    frame["prev_flow"] = frame["prev_flow"].fillna(0.0)
    frame["prev_spd"] = frame["prev_spd"].fillna(0.0)
    frame["prev_speed"] = frame["prev_speed"].fillna(global_speed)
    frame["prev_judged"] = frame["prev_judged"].fillna(float(frame["judged_share"].mean()))
    frame["gap_hours"] = frame["gap_hours"].fillna(999.0)
    frame["trail5_flow"] = frame["trail5_flow"].fillna(0.0)
    frame["trail20_spd"] = frame["trail20_spd"].fillna(0.0)
    return frame


def neighbour_means(frame, adjacency, value_cols=("flow", "speed")):
    from scipy import sparse

    segs = np.sort(frame["road_osmid"].unique())
    seg_index = {int(s): i for i, s in enumerate(segs)}
    hour_min = frame["hour"].min().floor("h")
    hour_max = frame["hour"].max().ceil("h")
    hours = pd.date_range(hour_min, hour_max, freq="h", tz=TZ)
    hour_index = {pd.Timestamp(h): j for j, h in enumerate(hours)}
    n_seg, n_hour = len(segs), len(hours)
    row = frame["road_osmid"].map(seg_index).to_numpy()
    col = frame["hour"].map(hour_index).to_numpy()
    neighbours = [sorted(adjacency.get(int(s), set())) for s in segs]
    degrees = np.array([len(v) for v in neighbours], dtype=float)
    data_r, data_c = [], []
    for i, nbrs in enumerate(neighbours):
        for nbr in nbrs:
            if nbr in seg_index:
                data_r.append(i)
                data_c.append(seg_index[nbr])
    adj = sparse.csr_matrix((np.ones(len(data_r)), (data_r, data_c)),
                            shape=(n_seg, n_seg))
    out = {}
    mask = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(n_seg, n_hour))
    hour_global = {}
    for col_name in value_cols:
        vals = frame[col_name].astype(np.float32).to_numpy()
        mat = sparse.csr_matrix((vals, (row, col)), shape=(n_seg, n_hour))
        nbr_sum, nbr_cnt, nbr_mean = [], [], []
        chunk = 1000
        for start in range(0, n_hour, chunk):
            sl = slice(start, min(start + chunk, n_hour))
            s_sum = np.asarray((adj @ mat[:, sl]).todense())
            s_cnt = np.asarray((adj @ mask[:, sl]).todense())
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(s_cnt > 0, s_sum / np.maximum(s_cnt, 1), np.nan)
            nbr_sum.append(s_sum)
            nbr_cnt.append(s_cnt)
            nbr_mean.append(mean)
        nbr_sum = np.hstack(nbr_sum)
        nbr_cnt = np.hstack(nbr_cnt)
        nbr_mean = np.hstack(nbr_mean)
        glob = np.asarray(mat.sum(axis=0)).ravel() / np.maximum(
            np.asarray(mask.sum(axis=0)).ravel(), 1)
        hour_global[col_name] = glob
        finite = np.isfinite(nbr_mean)
        col_idx = np.broadcast_to(np.arange(n_hour)[None, :], (n_seg, n_hour))
        nbr_mean[~finite] = np.take(glob, col_idx[~finite])
        out[col_name] = nbr_mean
    return out, seg_index, hour_index, degrees, hour_global


def attach_neighbour(frame, nbr_mats, seg_index, hour_index, hours_global):
    row = frame["road_osmid"].map(seg_index).to_numpy()
    col = frame["hour"].map(hour_index).to_numpy()
    n_hour = len(hour_index)
    prev_col = np.clip(col - 1, 0, n_hour - 1)
    flat_prev = row * n_hour + prev_col
    has_prev = col > 0
    result = {}
    for key, mat in nbr_mats.items():
        vals = mat.ravel()[flat_prev].astype(float)
        fallback = np.array([hours_global[key][c] for c in prev_col])
        vals[~has_prev] = fallback[~has_prev]
        result[key] = np.where(np.isfinite(vals), vals, fallback)
    return result


FLOW_FEATS = ["log_v1", "log_prev_flow", "log_gap", "has_hist", "log_trail5",
              "log_prev_spd", "log_trail20", "prev_speed", "prev_judged",
              "log_nbr_flow", "nbr_speed", "log_degree", "log_hb",
              "dist_stad", "maxspeed", "in_event", "log_att", "event_x_dist",
              "hod_sin", "hod_cos", "dow_sin", "dow_cos", "weekend"]
SPD_FEATS = list(FLOW_FEATS)


def build_features(rows, params=None):
    feats = pd.DataFrame(index=rows.index)
    feats["log_v1"] = np.log(rows["v1"].clip(lower=EPS))
    feats["log_prev_flow"] = np.log1p(rows["prev_flow"])
    feats["log_gap"] = np.log1p(rows["gap_hours"])
    feats["has_hist"] = rows["has_hist"]
    feats["log_trail5"] = np.log1p(rows["trail5_flow"])
    feats["log_prev_spd"] = np.log1p(rows["prev_spd"])
    feats["log_trail20"] = np.log1p(rows["trail20_spd"])
    feats["prev_speed"] = rows["prev_speed"]
    feats["prev_judged"] = rows["prev_judged"]
    feats["log_nbr_flow"] = np.log1p(rows["nbr_flow"].clip(lower=0.0))
    feats["nbr_speed"] = rows["nbr_speed"]
    feats["log_degree"] = np.log1p(rows["degree"])
    feats["log_hb"] = np.log1p(rows["hb_rate"])
    feats["dist_stad"] = rows["dist_stad"]
    feats["maxspeed"] = rows["maxspeed"]
    feats["in_event"] = rows["in_event"]
    feats["log_att"] = rows["log_att"]
    feats["event_x_dist"] = rows["in_event"] * rows["dist_stad_raw"]
    hod = rows["hour"].dt.hour + rows["hour"].dt.minute / 60.0
    dow = rows["hour"].dt.dayofweek
    feats["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    feats["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    feats["weekend"] = (dow >= 5).astype(float)
    if params is None:
        params = {}
        for col in FLOW_FEATS:
            mean = float(feats[col].mean())
            std = float(feats[col].std())
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            params[col] = (mean, std)
    for col in FLOW_FEATS:
        mean, std = params[col]
        feats[col] = (feats[col] - mean) / std
    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feats, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--score-full", action="store_true")
    ap.add_argument("--penetration", type=float, default=0.05)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--n-hours", type=int, default=6)
    ap.add_argument("--segments", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bundle56", default=str(BUNDLE56))
    ap.add_argument("--hourly56", default=str(HOURLY56))
    ap.add_argument("--out-dir", default=str(RES))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tag = "pilot" if args.pilot else "full"
    if args.pilot and args.bundle56 == str(BUNDLE56):
        pilot_bundle = RES / "segment_hour_forecast_bundle_pilot.json"
        pilot_hourly = RES / "segment_hour_forecast_hourly_pilot.csv"
        if pilot_bundle.exists() and pilot_hourly.exists():
            args.bundle56 = str(pilot_bundle)
            args.hourly56 = str(pilot_hourly)
    penetration = float(args.penetration)
    if not 0 < penetration <= 1:
        ap.error("--penetration must be in (0, 1]")

    bundle56 = json.loads(Path(args.bundle56).read_text())
    hourly56 = pd.read_csv(args.hourly56, parse_dates=["hour"])
    hourly56["hour"] = pd.to_datetime(hourly56["hour"], utc=True).dt.tz_convert(TZ)
    train_end = pd.Timestamp(bundle56["train_end"]).tz_convert(TZ)
    seg_base = pd.DataFrame(bundle56["segments"]).set_index("road_osmid")
    flow_base = seg_base["flow_base"].to_dict()
    propensity = seg_base["propensity"].to_dict()
    mean_flow_train = float(bundle56["mean_net_flow_train"])
    train_spd_rate = float(bundle56["train_spd_rate"])
    calib_flow = float(bundle56["calib_flow"])
    calib_spd = float(bundle56["calib_spd"])
    hour_pred_flow = dict(zip(hourly56["hour"], hourly56["pred_flow"]))
    hour_pred_spd = dict(zip(hourly56["hour"], hourly56["pred_spd"]))

    ecf = pd.read_csv(ECF)
    ecf["road_osmid"] = pd.to_numeric(ecf["road_osmid"], errors="coerce")
    ecf = ecf.dropna(subset=["road_osmid"])
    ecf["road_osmid"] = ecf["road_osmid"].astype(np.int64)
    static = ecf.groupby("road_osmid").agg(
        total_hb=("total_hb", "sum"), exposure_trips=("exposure_trips", "sum"),
        dist_stadium_km=("dist_stadium_km", "first"),
        road_name=("road_name", "first"), highway=("highway", "first"))
    static["hb_rate"] = (static["total_hb"]
                         / static["exposure_trips"].clip(lower=1.0)).fillna(0.0)
    hb_rate = static["hb_rate"].to_dict()
    dist_stad = static["dist_stadium_km"].to_dict()
    med_dist = float(np.median(list(dist_stad.values())))
    names = static["road_name"].fillna("").to_dict()
    highways = static["highway"].fillna("unknown").to_dict()
    geom = pd.read_csv(DATA / "road_geometry_features.csv")
    geom["road_osmid"] = pd.to_numeric(geom["road_osmid"], errors="coerce")
    geom = geom.dropna(subset=["road_osmid"])
    geom["road_osmid"] = geom["road_osmid"].astype(np.int64)
    maxspeed_map = geom.groupby("road_osmid")["maxspeed_kmh"].first().to_dict()
    med_speed_lim = float(np.median(list(maxspeed_map.values())))
    fixtures = load_fixtures(FIXTURES)
    adjacency, n_isolated = build_adjacency(GEOJSON)

    def enrich(rows):
        rows = rows.copy()
        rows["flow_base_seg"] = rows["road_osmid"].map(flow_base).fillna(
            float(np.mean(list(flow_base.values()))))
        rows["prop_seg"] = rows["road_osmid"].map(propensity).fillna(1.0).clip(0.0, 10.0)
        rows["hour_pf"] = rows["hour"].map(hour_pred_flow).fillna(mean_flow_train)
        rows["hour_ps"] = rows["hour"].map(hour_pred_spd).fillna(0.0)
        rows["v1_flow"] = (rows["flow_base_seg"] * rows["hour_pf"] / mean_flow_train
                           * calib_flow)
        rate = rows["hour_ps"] / rows["hour_pf"].clip(lower=EPS)
        share = (rows["prop_seg"] * rate / max(train_spd_rate, 1e-12)
                 * calib_spd).clip(0.0, 1.0)
        rows["v1_spd"] = rows["v1_flow"] * share
        rows["degree"] = rows["road_osmid"].map(
            {s: float(len(v)) for s, v in adjacency.items()}).fillna(0.0)
        rows["hb_rate"] = rows["road_osmid"].map(hb_rate).fillna(0.0)
        rows["dist_stad_raw"] = rows["road_osmid"].map(dist_stad).fillna(med_dist)
        rows["dist_stad"] = rows["dist_stad_raw"]
        rows["maxspeed"] = rows["road_osmid"].map(maxspeed_map).fillna(med_speed_lim)
        in_event, log_att = fixture_features(rows["hour"], fixtures)
        rows["in_event"] = in_event
        rows["log_att"] = log_att
        return rows

    if args.predict:
        raise SystemExit("predict mode wired after validation; see audit future work")

    min_hour = None
    if args.pilot:
        import pyarrow.parquet as pq

        hours = pd.to_datetime(
            pq.read_table(str(PANEL), columns=["hour"]).to_pandas()["hour"],
            utc=True).dt.tz_convert(TZ)
        min_hour = pd.Timestamp(np.sort(hours.unique())[-720])

    print("[57/1] building lag frame...")
    frame = build_lag_frame(PANEL, min_hour)
    print(f"   rows={len(frame):,} segments={frame['road_osmid'].nunique():,}")
    print("[57/2] neighbour means (causal t-1, sparse matvec)...")
    nbr_mats, seg_index, hour_index, degrees, hour_global = neighbour_means(
        frame, adjacency)
    attached = attach_neighbour(frame, nbr_mats, seg_index, hour_index, hour_global)
    frame["nbr_flow"] = attached["flow"]
    frame["nbr_speed"] = attached["speed"]
    del nbr_mats, attached
    frame = enrich(frame)
    audit = {"tag": tag, "n_isolated_segments": int(n_isolated),
             "penetration": penetration, "train_end": str(train_end)}
    audit["future_work"] = [
        "live Compass feed adapter: rebuild own-history lags + neighbour "
        "state from the feed each cycle, call predict for next N hours",
        "public-holiday features (holidays package unavailable offline)",
        "time-varying penetration calibration against independent counts",
    ]

    is_train = frame["hour"] <= train_end
    train_idx = frame.index[is_train].to_numpy()
    test_idx = frame.index[~is_train].to_numpy()
    print(f"   train={len(train_idx):,} test={len(test_idx):,}")
    rng = np.random.default_rng(7)
    spd_pos = train_idx[frame.loc[train_idx, "spd"].to_numpy() > 0]
    spd_zero = train_idx[frame.loc[train_idx, "spd"].to_numpy() == 0]
    if args.pilot:
        n_pos, n_total = 10000, 25000
        r_draws, de_iter, de_pop = 30, 4, 6
    else:
        n_pos, n_total = 150000, 300000
        r_draws, de_iter, de_pop = 100, 8, 8
    take_pos = spd_pos[rng.choice(len(spd_pos), min(n_pos, len(spd_pos)), replace=False)]
    take_zero = spd_zero[rng.choice(len(spd_zero), min(n_total - len(take_pos),
                                                       len(spd_zero)), replace=False)]
    sample_idx = np.concatenate([take_pos, take_zero])
    rng.shuffle(sample_idx)
    sample = frame.loc[sample_idx].copy()
    sample["row_id"] = np.arange(len(sample))
    sample["v1"] = sample["v1_flow"]
    print("[57/3] fitting Stage C (metacountregressor, stratified train sample)...")
    flow_feats, flow_params = build_features(sample, None)
    sample_flow = pd.concat([sample[["row_id", "flow", "spd"]].reset_index(drop=True),
                             flow_feats.reset_index(drop=True)], axis=1)
    flow_info = fit_metacount_nb(sample_flow, y_col="flow", id_col="row_id",
                                 fixed_terms=FLOW_FEATS, rdm_terms=[],
                                 model="nb", R=r_draws, de_maxiter=de_iter,
                                 de_popsize=de_pop, verbose=True)
    sample["v1"] = sample["v1_spd"]
    spd_feats, spd_params = build_features(sample, None)
    sample_spd = pd.concat([sample[["row_id", "flow", "spd"]].reset_index(drop=True),
                            spd_feats.reset_index(drop=True)], axis=1)
    spd_info = fit_metacount_nb(sample_spd, y_col="spd", id_col="row_id",
                                fixed_terms=SPD_FEATS, rdm_terms=[],
                                model="nb", R=r_draws, de_maxiter=de_iter,
                                de_popsize=de_pop, verbose=True)
    audit["stage_c"] = {
        "n_sample": int(len(sample)),
        "flow": {"aic": flow_info["aic"], "alpha": flow_info["alpha"],
                 "spec": flow_info["spec_label"]},
        "spd": {"aic": spd_info["aic"], "alpha": spd_info["alpha"],
                "spec": spd_info["spec_label"]},
    }

    def linear_predict(feats, info):
        beta = info.get("beta") or {}
        lin = np.full(len(feats), float(beta.get("__INTERCEPT__", 0.0)))
        for col in feats.columns:
            lin = lin + feats[col].to_numpy() * float(beta.get(col, 0.0))
        return np.maximum(np.exp(lin), EPS)

    print("[57/4] scoring (chunked; test rows feed metrics)...")
    order = np.sort(frame.index.to_numpy())
    seg_obs_f, seg_obs_s = {}, {}
    seg_v1f, seg_v1s, seg_v2f, seg_v2s, seg_n = {}, {}, {}, {}, {}
    gran_o_f, gran_p1_f, gran_p2_f = [], [], []
    gran_o_s, gran_p1_s, gran_p2_s = [], [], []
    gran_full = False
    pred_parts = []
    chunk = 500_000
    for start in range(0, len(order), chunk):
        sl = order[start:start + chunk]
        rows = frame.loc[sl].copy()
        rows["v1"] = rows["v1_flow"]
        feats_f, _ = build_features(rows, flow_params)
        rows["v1"] = rows["v1_spd"]
        feats_s, _ = build_features(rows, spd_params)
        rows["p2_flow"] = linear_predict(feats_f[FLOW_FEATS], flow_info)
        rows["p2_spd"] = np.minimum(linear_predict(feats_s[SPD_FEATS], spd_info),
                                    rows["p2_flow"])
        if args.score_full:
            pred_parts.append(rows[["road_osmid", "hour", "p2_flow", "p2_spd"]])
        trows = rows[rows["hour"] > train_end]
        if trows.empty:
            continue
        for seg, grp in trows.groupby("road_osmid"):
            seg = int(seg)
            seg_obs_f[seg] = seg_obs_f.get(seg, 0) + int(grp["flow"].sum())
            seg_obs_s[seg] = seg_obs_s.get(seg, 0) + int(grp["spd"].sum())
            seg_v1f[seg] = seg_v1f.get(seg, 0.0) + float(grp["v1_flow"].sum())
            seg_v1s[seg] = seg_v1s.get(seg, 0.0) + float(grp["v1_spd"].sum())
            seg_v2f[seg] = seg_v2f.get(seg, 0.0) + float(grp["p2_flow"].sum())
            seg_v2s[seg] = seg_v2s.get(seg, 0.0) + float(grp["p2_spd"].sum())
            seg_n[seg] = seg_n.get(seg, 0) + len(grp)
        if not gran_full:
            take = trows.iloc[::7]
            gran_o_f.extend(take["flow"].tolist())
            gran_p1_f.extend(take["v1_flow"].tolist())
            gran_p2_f.extend(take["p2_flow"].tolist())
            gran_o_s.extend(take["spd"].tolist())
            gran_p1_s.extend(take["v1_spd"].tolist())
            gran_p2_s.extend(take["p2_spd"].tolist())
            gran_full = len(gran_o_f) > 400_000
    if args.score_full:
        scored = pd.concat(pred_parts, ignore_index=True)
        scored.to_parquet(out_dir / f"segment_hour_pred_v2_{tag}.parquet", index=False)
        audit["score_full_rows"] = int(len(scored))
        del pred_parts, scored

    metrics: dict = {}

    def seg_metric(obs_d, pred_d, label):
        common = sorted(set(obs_d) & set(pred_d) & set(seg_n))
        obs_rate = np.array([obs_d[s] / max(seg_n[s], 1) for s in common])
        pred_rate = np.array([pred_d.get(s, 0.0) / max(seg_n[s], 1) for s in common])
        return {label + "_spearman": _spearman(obs_rate, pred_rate),
                label + "_top20": _jaccard_topk(obs_rate, pred_rate, 20),
                label + "_top50": _jaccard_topk(obs_rate, pred_rate, 50),
                label + "_top100": _jaccard_topk(obs_rate, pred_rate, 100),
                label + "_top500": _jaccard_topk(obs_rate, pred_rate, 500),
                label + "_n_seg": len(common)}

    metrics["segment_flow_v1"] = seg_metric(seg_obs_f, seg_v1f, "rate")
    metrics["segment_flow_v2"] = seg_metric(seg_obs_f, seg_v2f, "rate")
    metrics["segment_spd_v1"] = seg_metric(seg_obs_s, seg_v1s, "rate")
    metrics["segment_spd_v2"] = seg_metric(seg_obs_s, seg_v2s, "rate")
    metrics["segment_hour_flow_v1"] = {"spearman": _spearman(gran_o_f, gran_p1_f)}
    metrics["segment_hour_flow_v2"] = {"spearman": _spearman(gran_o_f, gran_p2_f)}
    metrics["segment_hour_spd_v1"] = {"spearman": _spearman(gran_o_s, gran_p1_s)}
    metrics["segment_hour_spd_v2"] = {"spearman": _spearman(gran_o_s, gran_p2_s)}
    metrics["n_granular"] = len(gran_o_f)

    print("[57/5] top predicted segments (weekly rate, penetration-scaled)...")
    weekly_hours = 168.0
    top_rows = []
    for seg in seg_n:
        rate_v2 = seg_v2s.get(seg, 0.0) / max(seg_n[seg], 1)
        rate_v1 = seg_v1s.get(seg, 0.0) / max(seg_n[seg], 1)
        rate_obs = seg_obs_s.get(seg, 0) / max(seg_n[seg], 1)
        compass_week = rate_v2 * weekly_hours
        top_rows.append({
            "road_osmid": seg,
            "road_name": names.get(seg, ""),
            "highway": highways.get(seg, "unknown"),
            "test_hours_obs": seg_n[seg],
            "obs_spd_rate": rate_obs,
            "v1_spd_rate": rate_v1,
            "v2_spd_rate": rate_v2,
            "pred_compass_week": compass_week,
            "pred_network_week": compass_week / penetration,
        })
    top = pd.DataFrame(top_rows).sort_values("v2_spd_rate", ascending=False)
    top.to_csv(out_dir / f"top_segments_speeding_{tag}.csv", index=False)

    coef_rows = []
    for feat in FLOW_FEATS:
        coef_rows.append({"feature": feat,
                          "flow_coef": float((flow_info.get("beta") or {}).get(feat, 0.0)),
                          "spd_coef": float((spd_info.get("beta") or {}).get(feat, 0.0))})
    pd.DataFrame(coef_rows).to_csv(out_dir / f"segment_rank_v2_coefs_{tag}.csv",
                                   index=False)
    (out_dir / f"segment_rank_v2_metrics_{tag}.json").write_text(
        json.dumps({"metrics": metrics, "audit": audit}, indent=2, default=str))
    bundle2 = {"flow_params": {k: list(v) for k, v in flow_params.items()},
               "spd_params": {k: list(v) for k, v in spd_params.items()},
               "flow_beta": {k: float(v) for k, v in (flow_info.get("beta") or {}).items()},
               "spd_beta": {k: float(v) for k, v in (spd_info.get("beta") or {}).items()},
               "features": FLOW_FEATS, "penetration": penetration,
               "train_end": str(train_end), "bundle56": str(args.bundle56)}
    (out_dir / f"segment_rank_v2_bundle_{tag}.json").write_text(json.dumps(bundle2))
    with open(out_dir / f"segment_rank_v2_{tag}.tex", "w") as tex:
        tex.write("% segment rank v2 validation (" + tag + ")\n")
        for key, val in metrics.items():
            tex.write(f"% {key}: {val}\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        common = sorted(set(seg_obs_s) & set(seg_v2s) & set(seg_n))
        obs_r = np.array([seg_obs_s[s] / max(seg_n[s], 1) for s in common])
        v1_r = np.array([seg_v1s.get(s, 0.0) / max(seg_n[s], 1) for s in common])
        v2_r = np.array([seg_v2s.get(s, 0.0) / max(seg_n[s], 1) for s in common])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        axes[0].scatter(obs_r, v1_r, s=4, alpha=0.4)
        axes[0].set_title("v1 segment speeding rate")
        axes[1].scatter(obs_r, v2_r, s=4, alpha=0.4, color="#0B6E4F")
        axes[1].set_title("v2 segment speeding rate")
        for ax in axes:
            ax.set_xlabel("observed test rate")
            ax.set_ylabel("predicted test rate")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"segment_rank_v2_rates_{tag}.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - plots must not kill the run
        print(f"   WARN: plot failed ({exc})")
    print(f"[57] DONE ({tag}). metrics: {json.dumps(metrics, default=str)}")


if __name__ == "__main__":
    main()
