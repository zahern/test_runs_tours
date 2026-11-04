"""67_game_effect_did.py - mechanically identified game effect (DiD).

Does a game day shift the finable-speeding RATE (per probe vehicle) at
segment x 15-min resolution, relative to the SAME segment at the SAME
time-of-day slot on matched non-game days? No model is fitted here: this is
the yardstick every structural claim (game_lift, sd_lift) must beat at the
resolution the MIP deploys on.

Design (per fixture; operational day op = (t - 6h).date so late-night slots
belong to the game night):
  game  : rows on the fixture's op date, 28 slots from ko-3h to ko+3h45
  base  : rows with the same time-of-day slot, same op weekday, within
          +-21 days, excluding EVERY calendar fixture op date and the
          fixture's own op date
  cell  : (road_osmid, mid), mid = hour*4 + minute//15
          rate_g = (y_g + .5) / (v_g + 1)
          rate_b = (sum y_b + .5) / (sum v_b + 1)   pooled over base days
          dlog_rate  = log(rate_g) - log(rate_b)            [propensity]
          dlog_count = log(y_g + .5) - log(y_b/nd + .5)     [counts]
Pools use ratio-of-sums (Poisson-appropriate). Cross-fixture t-stats use the
per-fixture pooled log-ratios (cluster = fixture).

Holdout validation (the 6 shipped fixtures):
  - train DiD curve vs holdout per-fixture effects
  - cell-level corr: dlog_rate vs sd_lift/pred_sd_nogame (SD relative lift),
    dlog_count vs game_lift/pred_nogame (regression relative lift)
  - rank value: top-20-cell Jaccard vs observed for (A) shipped pred_speeding
    and (B) pred_speeding * exp(train DiD class x rel multiplier)

Outputs: results/game_effect_did.json, figures/game_effect_did.png
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "data"
RES = PROJ / "results"
FIGS = PROJ / "figures"
CAL = DATA / "fixture_calendar.csv"
FEATS = DATA / "road_geometry_features.csv"
ECF = DATA / "near_miss_hb_ecf.csv"
CELLS = RES / "game_day_pred_vs_actual_cells.csv.gz"
PQ = RES / "game_day_pred_15min.parquet"
TZ = "Australia/Brisbane"
REL = 28
PRE_BINS = 12
WINDOW_D = 21
OP_SHIFT_H = 6
EPS = 1e-12
CLASSES = ["motorway", "primary", "secondary", "tertiary", "residential",
           "motorway_link", "primary_link", "secondary_link", "tertiary_link",
           "trunk", "trunk_link", "living_street", "service", "unclassified",
           "na"]


def fid_of(label, ko):
    return str(label) + " @ " + ko.strftime("%Y-%m-%d %H:%M")


def jacc(a, b):
    a, b = set(a), set(b)
    return float(len(a & b) / max(len(a | b), 1))


def corr(a, b, method="pearson"):
    sa = pd.Series(np.asarray(a, float))
    sb = pd.Series(np.asarray(b, float))
    if len(sa) < 10 or sa.nunique() < 2 or sb.nunique() < 2:
        return float("nan")
    return float(sa.corr(sb, method=method))


def date_int(ts):
    v = ts.dt if isinstance(ts, pd.Series) else ts
    return np.asarray(v.year * 10000 + v.month * 100 + v.day, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="trips_mean_gt_10")
    ap.add_argument("--min-base-days", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: only first N calendar fixtures")
    args = ap.parse_args()
    t0 = time.time()

    cal = pd.read_csv(CAL)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"]).dt.tz_localize(TZ)
    cal = cal.sort_values("ko").reset_index(drop=True)
    cal["fid"] = [fid_of(l, k) for l, k in zip(cal["label"], cal["ko"])]
    cal["op_ts"] = cal["ko"] - pd.Timedelta(hours=OP_SHIFT_H)
    cal["op_date"] = cal["op_ts"].dt.date
    hold = set(pd.read_csv(CELLS, usecols=["fixture"])["fixture"].astype(str))
    cal["is_hold"] = cal["fid"].isin(hold)
    if args.limit:
        cal = cal.iloc[:args.limit].reset_index(drop=True)
    fix_op_ints = set(date_int(cal["op_ts"]))
    print(f"[67] fixtures={len(cal)}  holdout={int(cal['is_hold'].sum())}  "
          f"target={args.target}", flush=True)

    feats = pd.read_csv(FEATS, usecols=lambda c: c in ("road_osmid", "highway"))
    ecf = pd.read_csv(ECF, usecols=lambda c: c in ("road_osmid", "highway",
                                                   "dist_stadium_km"))
    if "highway" not in ecf.columns and "highway" not in feats.columns:
        raise SystemExit("no highway column in FEATS or ECF")
    for d in (feats, ecf):
        d["road_osmid"] = pd.to_numeric(d["road_osmid"], errors="coerce")
        d.dropna(subset=["road_osmid"], inplace=True)
        d["road_osmid"] = d["road_osmid"].astype(np.int64)
    ctx = (ecf.groupby("road_osmid").first()
           .combine_first(feats.groupby("road_osmid").first())
           .reset_index()[["road_osmid", "highway", "dist_stadium_km"]])
    ctx["highway"] = (ctx["highway"].astype(str).str.strip().str.lower()
                      .replace({"": "na", "nan": "na"}))
    ctx["dist_stadium_km"] = pd.to_numeric(ctx["dist_stadium_km"],
                                           errors="coerce").fillna(99.0)
    cls_map = dict(zip(ctx["road_osmid"], ctx["highway"]))
    dist_map = dict(zip(ctx["road_osmid"], ctx["dist_stadium_km"]))

    parts = sorted(RES.glob("segment_15min_panel_????-??.parquet"))

    def month_of(d):
        return f"{d.year:04d}-{d.month:02d}"

    cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get_month(key):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        cands = [p for p in parts if p.stem.endswith(key)]
        if not cands:
            cache[key] = None
            return None
        df = pd.read_parquet(cands[0], columns=["road_osmid", "bin",
                                                args.target, "vehicle_count"])
        opt = df["bin"] - pd.Timedelta(hours=OP_SHIFT_H)
        df["_opi"] = date_int(opt)
        df["_wd"] = opt.dt.weekday.to_numpy().astype(np.int8)
        df["_mid"] = (df["bin"].dt.hour * 4
                      + df["bin"].dt.minute // 15).to_numpy().astype(np.int16)
        cache[key] = df
        while len(cache) > 8:
            cache.popitem(last=False)
        return df

    per_fx = {}
    cls_accum: dict[tuple, np.ndarray] = {}
    cell_hold = []
    COLS = ["road_osmid", args.target, "vehicle_count", "bin", "_mid", "_opi"]
    n = len(cal)
    for i, row in enumerate(cal.itertuples()):
        ko = row.ko
        ko_i = ko.year * 10000 + ko.month * 100 + ko.day
        wd0 = int(row.op_ts.weekday())
        ko_mid = ko.hour * 4 + ko.minute // 15
        start_mid = (ko_mid - PRE_BINS) % 96
        mids = np.array([(start_mid + r) % 96 for r in range(REL)],
                        dtype=np.int16)
        d0 = row.op_ts.normalize() - pd.Timedelta(days=WINDOW_D)
        d1 = row.op_ts.normalize() + pd.Timedelta(days=WINDOW_D)
        i0, i1 = date_int(pd.Series([d0]))[0], date_int(pd.Series([d1]))[0]
        base_ints = np.array(sorted({
            date_int(pd.Series([dd]))[0]
            for dd in pd.date_range(d0, d1, freq="D")
            if int(dd.weekday()) == wd0
            and int(date_int(pd.Series([dd]))[0]) not in fix_op_ints}),
            dtype=np.int64)
        keys = {month_of(d0), month_of(row.op_ts), month_of(d1)}
        chunks = []
        for k in sorted(keys):
            fr = get_month(k)
            if fr is None:
                continue
            opi = fr["_opi"].to_numpy()
            inwin = (opi >= i0) & (opi <= i1)
            if not inwin.any():
                continue
            slot = np.isin(fr["_mid"].to_numpy(), mids)
            base = np.isin(opi, base_ints, assume_unique=False)
            sel = inwin & slot & ((opi == ko_i) | base)
            if sel.any():
                chunks.append(fr.loc[sel, COLS])
        if not chunks:
            print(f"   {row.fid}: no rows, skip", flush=True)
            continue
        sub = pd.concat(chunks, ignore_index=True)
        sub["is_g"] = (sub["_opi"] == ko_i).to_numpy()
        y = pd.to_numeric(sub[args.target], errors="coerce").fillna(0.0)
        v = pd.to_numeric(sub["vehicle_count"], errors="coerce").fillna(0.0)
        sub["_y"] = y.to_numpy()
        sub["_v"] = v.to_numpy()
        gg = sub[sub["is_g"]].groupby(["road_osmid", "_mid"])[
            ["_y", "_v", "bin"]].agg({"_y": "sum", "_v": "sum",
                                      "bin": "first"})
        gg.columns = ["y_g", "v_g", "bin"]
        bsub = sub[~sub["is_g"]]
        bb = bsub.groupby(["road_osmid", "_mid"])[["_y", "_v"]].sum()
        bb.columns = ["y_b", "v_b"]
        nd = (bsub.groupby(["road_osmid", "_mid"])["_opi"]
              .nunique().rename("nd"))
        m = gg.join(bb).join(nd)
        m = m[m["nd"].fillna(0) >= args.min_base_days].copy()
        if not len(m):
            print(f"   {row.fid}: no cells, skip", flush=True)
            continue
        m["rel"] = ((m.index.get_level_values("_mid") - start_mid) % 96)
        m["rate_g"] = (m["y_g"] + 0.5) / (m["v_g"] + 1.0)
        m["rate_b"] = (m["y_b"] + 0.5) / (m["v_b"] + 1.0)
        m["dlog_rate"] = np.log(m["rate_g"]) - np.log(m["rate_b"])
        m["dlog_cnt"] = (np.log(m["y_g"] + 0.5)
                         - np.log(m["y_b"] / m["nd"] + 0.5))
        rm = m.index.get_level_values("road_osmid")
        m["cls"] = pd.Series(rm.map(cls_map)).fillna("na").to_numpy()
        m["dist"] = pd.Series(rm.map(dist_map)).fillna(99.0).to_numpy()
        rates = np.full(REL, np.nan)
        cnts = np.full(REL, np.nan)
        g2 = m.groupby("rel")[["y_g", "v_g", "y_b", "v_b", "nd"]].sum()
        ri = g2.index.to_numpy()
        rates[ri] = (np.log((g2["y_g"] + 0.5) / (g2["v_g"] + 1.0))
                     - np.log((g2["y_b"] + 0.5) / (g2["v_b"] + 1.0)))
        cnts[ri] = (np.log(g2["y_g"] + 0.5)
                    - np.log(g2["y_b"] / g2["nd"].clip(lower=1) + 0.5))
        tot_r = float(np.log((m["y_g"].sum() + 0.5) / (m["v_g"].sum() + 1.0))
                      - np.log((m["y_b"].sum() + 0.5) / (m["v_b"].sum() + 1.0)))
        tot_c = float(np.log(m["y_g"].sum() / len(m) + 0.5)
                      - np.log(m["y_b"].sum() / max(m["nd"].sum(), 1) + 0.5))
        per_fx[row.fid] = {
            "rate": rates.tolist(), "cnt": cnts.tolist(),
            "tot_rate": tot_r, "tot_cnt": tot_c,
            "att": float(row.attendance) if np.isfinite(row.attendance) else None,
            "type": str(row.fixture_type), "hold": bool(row.is_hold),
            "n_cells": int(len(m)),
        }
        if not row.is_hold:
            g3 = m.groupby(["cls", "rel"])[
                ["y_g", "v_g", "y_b", "v_b", "nd"]].sum()
            for (cl, rl), v5 in g3.iterrows():
                acc = cls_accum.setdefault((cl, int(rl)), np.zeros(5))
                acc += v5.to_numpy(float)
        else:
            m2 = m.reset_index()
            m2["fixture"] = row.fid
            cell_hold.append(
                m2[["fixture", "road_osmid", "rel", "cls", "dist",
                    "y_g", "v_g", "y_b", "v_b", "nd", "dlog_rate",
                    "dlog_cnt", "bin", "_mid"]])
        print(f"[67] {i+1:2d}/{n} {row.fid[:38]:38s} cells={len(m):6d} "
              f"tot_rate={tot_r:+.3f} tot_cnt={tot_c:+.3f} "
              f"({time.time()-t0:5.0f}s)", flush=True)

    tr = [v for v in per_fx.values() if not v["hold"]]
    ho = [v for v in per_fx.values() if v["hold"]]
    Rtr = np.array([v["rate"] for v in tr], float)
    tr_rate = np.nanmean(Rtr, axis=0)
    tr_rate_sd = np.nanstd(Rtr, axis=0, ddof=1) / np.sqrt(len(tr))
    t_rate = tr_rate / np.where(tr_rate_sd > 0, tr_rate_sd, np.nan)
    Ctr = np.array([v["cnt"] for v in tr], float)
    tr_cnt = np.nanmean(Ctr, axis=0)
    tr_cnt_sd = np.nanstd(Ctr, axis=0, ddof=1) / np.sqrt(len(tr))
    # train class x rel multiplier (ratio-of-sums, rate scale)
    cls_mult, cls_curve = {}, {}
    for (cl, rl), v5 in cls_accum.items():
        yg, vg, yb, vb, nd = v5
        eff = float(np.log((yg + 0.5) / (vg + 1.0))
                    - np.log((yb + 0.5) / (vb + 1.0)))
        cls_mult[f"{cl}|{rl}"] = float(np.exp(eff))
        cls_curve.setdefault(cl, {})[rl] = eff
    out = {
        "target": args.target,
        "n_train": len(tr), "n_hold": len(ho),
        "train_rate": tr_rate.tolist(), "train_rate_se": tr_rate_sd.tolist(),
        "train_rate_t": t_rate.tolist(), "train_cnt": tr_cnt.tolist(),
        "train_cnt_se": tr_cnt_sd.tolist(),
        "cls_mult": cls_mult,
        "per_fixture": per_fx,
    }
    print("[67] train pooled rate effect by rel:",
          np.array2string(np.round(tr_rate, 3), separator=","), flush=True)
    print("[67] train pooled rate t:",
          np.array2string(np.round(t_rate, 2), separator=","), flush=True)

    val = {}
    hcells = pd.concat(cell_hold, ignore_index=True) if cell_hold else None
    if hcells is not None and len(ho):
        pq = pd.read_parquet(PQ, columns=["fixture", "road_osmid", "bin",
                                          "pred_speeding", "pred_nogame",
                                          "game_lift", "pred_sd_nogame",
                                          "sd_lift"])
        pq["fixture"] = pq["fixture"].astype(str)
        pq["bin"] = pd.to_datetime(pq["bin"])
        if pq["bin"].dt.tz is None:
            pq["bin"] = pq["bin"].dt.tz_localize(TZ)
        hcells["bin"] = pd.to_datetime(hcells["bin"])
        if hcells["bin"].dt.tz is None:
            hcells["bin"] = hcells["bin"].dt.tz_localize(TZ)
        j = hcells.merge(pq, on=["fixture", "road_osmid", "bin"], how="inner")
        j["sd_rel"] = (pd.to_numeric(j["sd_lift"], errors="coerce")
                       / pd.to_numeric(j["pred_sd_nogame"], errors="coerce")
                       .clip(lower=EPS))
        j["gl_rel"] = (pd.to_numeric(j["game_lift"], errors="coerce")
                       / pd.to_numeric(j["pred_nogame"], errors="coerce")
                       .clip(lower=EPS))
        j["mult"] = [cls_mult.get(f"{cl}|{int(rl)}", 1.0)
                     for cl, rl in zip(j["cls"], j["rel"])]
        val["n_join"] = int(len(j))
        val["corr_dlograte_vs_sd_rel"] = corr(j["dlog_rate"], j["sd_rel"])
        val["corr_dlograte_vs_sd_rel_spearman"] = corr(
            j["dlog_rate"], j["sd_rel"], "spearman")
        val["corr_dlogcnt_vs_gl_rel"] = corr(j["dlog_cnt"], j["gl_rel"])
        val["corr_dlogcnt_vs_gl_rel_spearman"] = corr(
            j["dlog_cnt"], j["gl_rel"], "spearman")
        per = {}
        for f, g in j.groupby("fixture"):
            per[f] = {
                "n": int(len(g)),
                "corr_rate_sd": corr(g["dlog_rate"], g["sd_rel"]),
                "corr_cnt_gl": corr(g["dlog_cnt"], g["gl_rel"]),
            }
        val["per_fixture"] = per
        # top-20 observed cells per fixture: shipped vs DiD-adjusted
        ja, jb = [], []
        static_sets = []
        for f, g in j.groupby("fixture"):
            if len(g) < 50:
                continue
            k = 20
            top_obs = set(g["y_g"].to_numpy().argsort()[::-1][:k])
            top_a = set(g["pred_speeding"].to_numpy().argsort()[::-1][:k])
            B = (pd.to_numeric(g["pred_speeding"], errors="coerce").fillna(0.0)
                 * g["mult"].to_numpy(float))
            top_b = set(B.to_numpy().argsort()[::-1][:k])
            ja.append(jacc(top_obs, top_a))
            jb.append(jacc(top_obs, top_b))
            static_sets.append(top_a)
        val["top20_jaccard_shipped"] = float(np.nanmean(ja)) if ja else None
        val["top20_jaccard_did_adj"] = float(np.nanmean(jb)) if jb else None
        val["top20_per_fixture_shipped"] = [round(float(x), 3) for x in ja]
        val["top20_per_fixture_did_adj"] = [round(float(x), 3) for x in jb]
        if len(static_sets) > 1:
            val["shipped_top20_static_jaccard"] = round(
                float(np.mean([jacc(static_sets[0], s)
                               for s in static_sets[1:]])), 3)
    out["holdout_validation"] = val
    print("[67] holdout validation:", json.dumps(
        {k: v for k, v in val.items() if k != "per_fixture"}, indent=2),
        flush=True)

    with open(RES / "game_effect_did.json", "w") as fh:
        json.dump(out, fh, indent=2)

    # ---------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(REL) - PRE_BINS
    a = ax[0, 0]
    a.axhline(0, color="0.6", lw=0.8)
    a.fill_between(x, tr_rate - 1.96 * tr_rate_sd, tr_rate + 1.96 * tr_rate_sd,
                   color="C0", alpha=0.18)
    a.plot(x, tr_rate, color="C0", lw=2, label="train pooled (rate log-ratio)")
    for v in ho:
        a.plot(x, np.array(v["rate"], float), color="C1", lw=0.9, alpha=0.7)
    if ho:
        a.plot(x, np.nanmean(np.array([v["rate"] for v in ho], float), axis=0),
               color="C1", lw=2, label="holdout mean")
    a.axvline(0, color="0.4", ls="--", lw=0.8)
    a.set_title("DiD game effect on exceeding RATE (per probe veh)")
    a.set_xlabel("rel bin (0 = kickoff, 1 bin = 15 min)")
    a.legend(fontsize=8)
    b = ax[0, 1]
    xs = np.arange(len(tr))
    b.bar(xs, [v["tot_rate"] for v in tr], color="C0", label="train")
    b.bar(np.arange(len(tr), len(tr) + len(ho)),
          [v["tot_rate"] for v in ho], color="C1", label="holdout")
    b.axhline(0, color="0.5", lw=0.8)
    b.set_title("whole-window pooled effect per fixture")
    b.legend(fontsize=8)
    c = ax[1, 0]
    for cl in CLASSES:
        cur = cls_curve.get(cl)
        if not cur or len(cur) < 10:
            continue
        rl = np.array(sorted(cur))
        c.plot(rl - PRE_BINS, [cur[r] for r in rl], lw=1.2, label=cl)
    c.axhline(0, color="0.6", lw=0.8)
    c.axvline(0, color="0.4", ls="--", lw=0.8)
    c.set_title("train class curves (rate effect)")
    c.legend(fontsize=7)
    d = ax[1, 1]
    if hcells is not None and val.get("n_join"):
        jj2 = j.sample(min(40000, len(j)), random_state=0)
        d.scatter(jj2["sd_rel"], jj2["dlog_rate"], s=4, alpha=0.2, color="C2")
        d.set_title("holdout cells: dlog_rate vs SD relative lift "
                    f"(r={val.get('corr_dlograte_vs_sd_rel'):.3f})")
        d.set_xlabel("sd_lift / pred_sd_nogame")
        d.set_ylabel("dlog rate (obs vs matched baseline)")
    fig.tight_layout()
    fig.savefig(FIGS / "game_effect_did.png", dpi=130)
    print(f"[67] done in {time.time()-t0:.0f}s -> "
          "results/game_effect_did.json, figures/game_effect_did.png",
          flush=True)


if __name__ == "__main__":
    main()
