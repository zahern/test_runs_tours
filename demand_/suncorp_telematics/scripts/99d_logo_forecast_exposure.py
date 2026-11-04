"""
99d_logo_forecast_exposure.py - LOGO with a real exposure forecast.

Same EB score family as 99b, but the expected-event score now uses
exposure predicted from results/segment_hour_pred_v2_full.parquet
(road_osmid x hour -> p2_flow), scaled by a single calibration constant
fitted on the TRAINING fixtures only:

    exposure_fcst(bin) = 0.25 * p2_flow(site, floor_hour(bin)) * c
    c = sum(training veh) / sum(training 0.25*p2_flow),  rows with exp>0

Scores (LOGO):
  s_cnt_fcst   : s_rate_add x exposure_fcst            <- deployable
  s_cnt_add    : s_rate_add x training-mean exposure   (99b comparison)
  s_rate_add   : rate only
  s_expo_fcst  : exposure_fcst only (traffic volume, no event data)
  s_expo       : training-mean exposure only

Usage:
  python scripts/99d_logo_forecast_exposure.py --marker sp
  python scripts/99d_logo_forecast_exposure.py --marker hb
    (hb needs results/segment_15min_panel_braking_full.parquet, see 99c)
Outputs -> results/obs_site_bin_diag/logo/logo_fcst_<marker>_<gran>.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(__file__).resolve().parents[1]
DIAG = PROJ / "results" / "obs_site_bin_diag"
OUT = DIAG / "logo"
P2 = PROJ / "results" / "segment_hour_pred_v2_full.parquet"
SP_PANEL = PROJ / "results" / "segment_15min_panel.parquet"
HB_PANEL = PROJ / "results" / "segment_15min_panel_braking_full.parquet"
CAL = PROJ / "data" / "fixture_calendar.csv"
SEG = "road_osmid"
TOPK = 25
MIN_VEH_EVAL = 10
MIN_EVAL = 25
SCORES = ["s_rate_add", "s_rate_inter", "s_cnt_add", "s_cnt_fcst",
          "s_cnt_fcst_inter", "s_expo", "s_expo_fcst"]
TZ = "Australia/Brisbane"


def _load_99b():
    spec = importlib.util.spec_from_file_location(
        "b99b", PROJ / "scripts" / "99b_phase_surface_logo_validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def game_windows_all():
    """window bins for every fixture in the calendar (start >= 2025-01-01)."""
    cal = pd.read_csv(CAL)
    cal["fixture_id"] = cal["label"] + "|" + cal["kickoff_iso"].str[:10]
    rows = []
    for _, f in cal.iterrows():
        ko = pd.Timestamp(f["kickoff_iso"]).tz_localize(TZ)
        if ko < pd.Timestamp("2025-01-04", tz=TZ):
            continue
        for r in range(28):
            rows.append({"fixture_id": f["fixture_id"], "rel_bin": r,
                         "bin": ko - pd.Timedelta(hours=3)
                                + pd.Timedelta(minutes=15 * r)})
    return pd.DataFrame(rows)


def tz_str(s):
    return pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%S%z")


def observed_veh(bins, num_col=None):
    """observed rows from the speeding panel for game-window bins (duckdb)."""
    import duckdb
    con = duckdb.connect()
    con.register("wb", bins)
    numsel = f", {num_col} AS num" if num_col else ""
    df = con.execute(f"""
        SELECT road_osmid, strftime(bin,'%Y-%m-%dT%H:%M:%S%z') AS bin_str,
               vehicle_count{numsel}
        FROM read_parquet(?)
        WHERE bin IN (SELECT bin::TIMESTAMPTZ_NS FROM wb)
    """, [str(SP_PANEL)]).df()
    df["bin"] = tz_str(df.pop("bin_str"))
    return df.rename(columns={"vehicle_count": "veh"})


def load_cells(marker):
    """per-bin rows: observed events + observed veh, game windows only."""
    win = game_windows_all()
    ub = win[["bin"]].drop_duplicates()

    if marker == "hb":
        ev = observed_veh(ub)
        hb = pd.read_parquet(HB_PANEL, columns=["road_osmid", "bin",
                                                "braking_events"])
        hb = hb[hb["bin"].isin(ub["bin"])]
        ev = ev.merge(hb.rename(columns={"braking_events": "num"}),
                      on=[SEG, "bin"], how="left")
        ev["num"] = ev["num"].fillna(0).astype("int64")
    else:
        ev = observed_veh(ub, num_col="trips_any_point_gt_5")

    ev = ev.merge(win[["fixture_id", "rel_bin", "bin"]], on="bin", how="inner")
    ev["hour"] = ev["bin"].dt.floor("1h")

    # forecast exposure: p2_flow for the (site,hour) grid we need
    import duckdb
    con = duckdb.connect()
    lo, hi = ev["hour"].min(), ev["hour"].max()
    p2 = con.execute("""
        SELECT road_osmid, strftime(hour,'%Y-%m-%dT%H:%M:%S%z') AS hour_str,
               p2_flow
        FROM read_parquet(?)
        WHERE hour >= ? AND hour <= ?
    """, [str(P2), str(lo.tz_convert(None)), str(hi.tz_convert(None))]).df()
    p2["hour"] = tz_str(p2.pop("hour_str"))
    ev = ev.merge(p2, on=[SEG, "hour"], how="left")
    ev["exp_fcst"] = 0.25 * ev["p2_flow"].fillna(0.0)
    ev = ev[ev["veh"] >= MIN_VEH_EVAL].copy()
    return ev


def fit_and_logo(b99b, ev, gran_name, marker):
    GRAN = b99b.GRANS[gran_name]
    ev = ev.copy()
    ev[gran_name] = [GRAN(r) for r in ev["rel_bin"]]
    g = ev.groupby([SEG, "fixture_id", gran_name]).agg(
        num=("num", "sum"), veh=("veh", "sum"),
        exp_fcst=("exp_fcst", "sum")).reset_index()
    g["exp_fcst"] = g["exp_fcst"].fillna(0.0)

    rows = []
    for fid, ev_f in g.groupby("fixture_id"):
        tr = g[g["fixture_id"] != fid]
        if tr["num"].sum() < 100:
            continue
        model = b99b.fit_scores(tr, gran_name)
        cal_mask = tr["exp_fcst"] > 0
        c = float(tr.loc[cal_mask, "veh"].sum()
                  / max(tr.loc[cal_mask, "exp_fcst"].sum(), 1e-9))
        by = ev_f[[SEG, gran_name, "num", "veh", "exp_fcst"]].copy()
        by["rate_obs"] = by["num"] / by["veh"].clip(lower=1)
        preds = b99b.predict(model, by, gran_name)
        e = by["exp_fcst"].values * c
        preds["s_cnt_fcst"] = preds["s_rate_add"] * e
        preds["s_cnt_fcst_inter"] = preds["s_rate_inter"] * e
        preds["s_expo_fcst"] = e
        evv = pd.concat([by, preds], axis=1)
        for gval, sub in evv.groupby(gran_name):
            for r in _evaluate(sub, SCORES):
                rows.append({"fold": fid, gran_name: gval, **r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"logo_fcst_{marker}_{gran_name}.csv", index=False)
    agg = res.groupby([gran_name, "score"]).agg(
        spearman=("spearman", "mean"),
        topk_jaccard=("topk_jaccard", "mean"),
        capture=(f"capture@{TOPK}", "mean"),
        folds=("fold", pd.Series.nunique)).reset_index()
    agg.to_csv(OUT / f"logo_fcst_agg_{marker}_{gran_name}.csv", index=False)
    return agg


def _evaluate(ev, score_list, k=TOPK):
    mask = (ev["veh"] >= MIN_VEH_EVAL).values
    rs = ev["rate_obs"].values
    if int(mask.sum()) < MIN_EVAL:
        return []
    rows = []
    for score in score_list:
        s = ev[score].values
        ok = mask & np.isfinite(s) & np.isfinite(rs)
        if ok.sum() < MIN_EVAL:
            continue
        sub, s2, r2 = ev[ok], s[ok], rs[ok]
        rho = float(stats.spearmanr(s2, r2).statistic)
        top = sub.index[np.argsort(-s2)][:k]
        obs_top = sub.index[np.argsort(-r2)][:k]
        jac = len(set(top) & set(obs_top)) / len(set(top) | set(obs_top))
        cap = float(sub.loc[top, "num"].sum()) / max(float(sub["num"].sum()), 1.0)
        rows.append({"score": score, "spearman": rho, "topk_jaccard": jac,
                     f"capture@{k}": cap, "n_eval": int(ok.sum()),
                     "events": float(sub["num"].sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", default="sp", choices=["sp", "hb"])
    args = ap.parse_args()
    b99b = _load_99b()
    ev = load_cells(args.marker)
    print(f"loaded {len(ev):,} bin rows, "
          f"{ev['fixture_id'].nunique()} fixtures", flush=True)
    for gran_name in ["phase", "block"]:
        agg = fit_and_logo(b99b, ev, gran_name, args.marker)
        print(f"\n=== LOGO + forecast exposure {args.marker.upper()} "
              f"@ {gran_name} ===")
        print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
