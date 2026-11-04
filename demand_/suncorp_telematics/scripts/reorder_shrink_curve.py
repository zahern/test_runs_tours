#!/usr/bin/env python
"""Shrinkage curve for the kickoff x flow reordering cross.

The band+cross arm passed the reordering gate (pairwise surface Spearman
0.9896 < 0.99) but failed calibration (slope 1.461 > 1.35). The obvious fix is
to shrink the two crossed coefficients; the question is whether ANY shrink
factor satisfies both gates. Answering it by refitting is ~90 minutes per
point, so this script scores the whole curve from the ALREADY FITTED surface.

Why that is exact enough to decide with: the tilt enters the linear predictor
additively as

    beta_s * sin(2*pi*ko/24) * flow_z  +  beta_c * cos(2*pi*ko/24) * flow_z

so replacing (beta_s, beta_c) by s*(beta_s, beta_c) multiplies the predicted
rate by exp((s-1) * c_fs) where c_fs is the per-(fixture, segment) value of
that sum. Every other coefficient is held at its fitted value. That is the
first-order effect of re-fitting with the covariates scaled by s; it ignores
the small movement of the other 25 terms, which is exactly the price of not
paying 90 minutes per point. The s=1 row is a free check on the whole
reconstruction: it must reproduce the recorded run's calibration slope and
gate statistics.

Reported per s: pooled calibration slope (obs on pred, the same fit the
handoff gate uses), pairwise surface Spearman across fixtures, top-20 and
top-10 site Jaccard, and the per-fixture Pearson mean.

It then scores the OTHER channel, an amplitude map p -> m(p/m)^k, which is
monotone and therefore leaves every rank (and so both rank-based gate
statistics) exactly unchanged. That separates "the surface reorders wrongly"
from "the surface is mis-scaled", which the coefficient shrink cannot.

Read the s=1.0 row first: it must reproduce the recorded run exactly
(slope 1.4613, rho 0.9896, jaccard_top20 0.795, jaccard_top10 0.649). If it
does not, the observed reconstruction is wrong and no row below it means
anything -- which is how the wrong target column was caught.

Usage:
    python scripts/reorder_shrink_curve.py [--run-dir DIR] [--scales ...]
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results_tests" / "t9_ab_band_cross"
PANEL = ROOT / "results" / "segment_15min_panel.parquet"
# The A/B arms run with --target trips_mean_gt_10 (see run_t9_ab_grain.sh
# COMMON), and the metrics join obs_light[args.target] as "observed". Using
# TARGET_DEFAULT (speeding_trips) here silently scored a different response:
# the row count still matched 155,584, so only the slope gave it away.
TARGET = "trips_mean_gt_10"

HANDOFF_SPEARMAN_MAX = 0.99
HANDOFF_JACCARD_MAX = 0.80
CAL_BAND = (0.75, 1.35)


def load_preds(run: Path) -> pd.DataFrame:
    p = run / "game_day_pred_15min.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    d = ds.dataset(p)
    # `fixture` already carries the kickoff label ("NRL ... @ 2026-03-15
    # 17:15"); the run keeps no separate fixture_label column on disk.
    cols = ["fixture", "road_osmid", "bin", "pred_speeding", "exp"]
    df = d.to_table(columns=cols).to_pandas()
    # pred_flow was overwritten with log1p(exposure) before saving (see
    # keys["exp"] = fw["logexp"]), so `exp` is already on the log scale --
    # it is a monotone stand-in for segment flow, which is all flow_z needs.
    # Keep the timezone: the panel's bin is timestamp[ns, Australia/Brisbane]
    # and Arrow refuses to compare a tz-aware filter value against a tz-naive
    # column (and vice versa). Both frames stay on the panel's clock.
    df["bin"] = pd.to_datetime(df["bin"])
    return df


def flow_z(df: pd.DataFrame) -> pd.Series:
    """Standardised log flow per segment, as add_exposure builds it.

    The real column standardises over the frame being processed, not over the
    panel; this uses all bins of all future fixtures, so it differs by an
    affine transform. Only the tilt's amplitude is affected, never its
    direction, and the s=1 anchor row is unchanged either way.
    """
    per_seg = df.groupby("road_osmid")["exp"].mean()
    z = (per_seg - per_seg.mean()) / max(float(per_seg.std()), 1e-12)
    return z.clip(-3.0, 3.0)


def ko_hours(df: pd.DataFrame) -> pd.Series:
    """Kickoff hour per fixture, parsed from the fixture label.

    The label is the only place the kickoff survives in the prediction
    parquet, and it is the same quantity add_kick_flow_crosses uses
    (grid["ko_hour"], from the calendar row). A label without a parseable
    HH:MM fails the run rather than silently tilting at midnight.
    """
    out = {}
    for fx in df["fixture"].unique():
        tail = str(fx).rsplit("@", 1)[-1].strip()
        parts = tail.split()[-1].split(":")
        if len(parts) < 2:
            raise SystemExit(f"cannot parse kickoff from fixture {fx!r}")
        out[fx] = float(parts[0]) + float(parts[1]) / 60.0
    return pd.Series(out)


def cross_coefs(run: Path) -> tuple[float, float]:
    c = pd.read_csv(run / "game_day_model_coefs.csv")
    c = c.set_index("variable")["coefficient"]
    miss = [v for v in ("ko_sin_x_flow", "ko_cos_x_flow") if v not in c.index]
    if miss:
        raise SystemExit(f"{run}/game_day_model_coefs.csv lacks {miss}")
    return float(c["ko_sin_x_flow"]), float(c["ko_cos_x_flow"])


def observed(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Panel target on the future fixtures' bins, keyed as the run joins it.

    59 builds future_events from the panel for the future windows and inner-
    joins it to the prediction grid on (fixture, road_osmid, bin). The set of
    (fixture, bin) pairs is already known from the predictions, so the panel
    read is filtered to those 112 timestamps and the whole 64M-row panel is
    never materialised.
    """
    if not PANEL.exists():
        raise SystemExit(f"missing panel {PANEL}")
    keys = df[["fixture", "road_osmid", "bin"]].drop_duplicates()
    bins = sorted(keys["bin"].unique())
    tab = ds.dataset(PANEL).to_table(
        columns=["road_osmid", "bin", target],
        filter=ds.field("bin").isin(bins)).to_pandas()
    if not len(tab):
        raise SystemExit("panel filter returned no rows; bin dtype mismatch?")
    tab["bin"] = pd.to_datetime(tab["bin"])
    obs = (tab.groupby(["road_osmid", "bin"])[target].sum()
           .rename("observed").reset_index())
    out = keys.merge(obs, on=["road_osmid", "bin"], how="inner")
    return out


def _pairwise_spearman(tot: pd.DataFrame) -> tuple[float, dict]:
    cols = list(tot.columns)
    vals = {}
    for a, b in combinations(cols, 2):
        ra, rb = tot[a].rank(), tot[b].rank()
        vals[f"{a[:26]}|{b[:26]}"] = (
            float(np.corrcoef(ra, rb)[0, 1])
            if ra.std() > 0 and rb.std() > 0 else float("nan"))
    v = [x for x in vals.values() if np.isfinite(x)]
    return (float(np.mean(v)) if v else float("nan")), vals


def _jaccard(tot: pd.DataFrame, n: int) -> float:
    tops = [set(tot[c].nlargest(n).index) for c in tot.columns]
    vals = [len(a & b) / max(len(a | b), 1)
            for a, b in combinations(tops, 2)]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--scales", default="1,0.85,0.7,0.55,0.4,0.25,0.1,0",
                    help="comma-separated multipliers on the crossed coefs")
    ap.add_argument("--csv", default="")
    ap.add_argument("--target", default=TARGET,
                    help="panel column the run used as its response")
    args = ap.parse_args()
    run = Path(args.run_dir)
    scales = [float(x) for x in args.scales.split(",")]

    df = load_preds(run)
    print(f"predictions: {len(df):,} rows, {df['fixture'].nunique()} fixtures")
    fz = flow_z(df)
    kh = ko_hours(df)
    bs, bc = cross_coefs(run)
    print(f"crossed coefs: sin={bs:+.4f} cos={bc:+.4f}")

    ksi = np.sin(2 * np.pi * kh / 24.0)
    kco = np.cos(2 * np.pi * kh / 24.0)
    per_fx = (bs * ksi + bc * kco)          # (fixture) -> tilt slope in flow_z
    df["tilt"] = (df["fixture"].map(per_fx).to_numpy(float)
                  * df["road_osmid"].map(fz).to_numpy(float))
    print(f"tilt (log-scale, full strength): sd={df['tilt'].std():.3f} "
          f"p1={df['tilt'].quantile(.01):+.3f} p99={df['tilt'].quantile(.99):+.3f}")

    obs = observed(df, args.target)
    print(f"observed rows: {len(obs):,} (run reported 155,584 merged)")

    base = df[["fixture", "road_osmid", "bin", "pred_speeding"]].copy()
    base["observed"] = np.nan
    j = base.merge(obs[["fixture", "road_osmid", "bin", "observed"]],
                   on=["fixture", "road_osmid", "bin"], how="left",
                   suffixes=("", "_o"))
    base["observed"] = j["observed_o"].to_numpy(float)
    print(f"merged rows: {int(np.isfinite(base['observed']).sum()):,}")

    rows = []
    for s in scales:
        pred = df["pred_speeding"].to_numpy(float) * np.exp(
            (s - 1.0) * df["tilt"].to_numpy(float))
        d = pd.DataFrame({"fixture": df["fixture"], "road_osmid": df["road_osmid"],
                          "pred": pred, "observed": base["observed"].to_numpy()})
        tot = d.pivot_table(index="road_osmid", columns="fixture",
                            values="pred", aggfunc="sum").fillna(0.0)
        rho, _ = _pairwise_spearman(tot)
        o = d["observed"].to_numpy(float)
        p = d["pred"].to_numpy(float)
        msk = np.isfinite(o) & np.isfinite(p)
        slope, icept = (np.polyfit(p[msk], o[msk], 1)
                        if msk.sum() > 10 and np.var(p[msk]) > 0
                        else (np.nan, np.nan))
        seg = d[msk].groupby(["fixture", "road_osmid"]).agg(
            o=("observed", "sum"), p=("pred", "sum"))
        pr = [np.corrcoef(g["o"], g["p"])[0, 1] for _, g in seg.groupby(level=0)
              if g["p"].std() > 0 and g["o"].std() > 0]
        rows.append({
            "scale": s,
            "calib_slope": round(float(slope), 4),
            "surface_spearman": round(float(rho), 4),
            "jaccard_top20": round(_jaccard(tot, 20), 3),
            "jaccard_top10": round(_jaccard(tot, 10), 3),
            "pearson_mean": round(float(np.mean(pr)), 4) if pr else None,
            "calib_ok": bool(CAL_BAND[0] <= slope <= CAL_BAND[1]),
            "spearman_ok": bool(rho <= HANDOFF_SPEARMAN_MAX),
        })
        print(f"   s={s:<5} slope={slope:5.3f} rho={rho:.4f} "
              f"jac20={rows[-1]['jaccard_top20']:.3f}")

    out = pd.DataFrame(rows)
    out["both_ok"] = out["calib_ok"] & out["spearman_ok"]
    txt = out.to_string(index=False)
    print("\n" + txt)
    print(f"\nscales satisfying BOTH gates: "
          f"{out.loc[out['both_ok'], 'scale'].tolist() or 'none'}")
    csv = Path(args.csv) if args.csv else run / "reorder_shrink_curve.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")

    # ---------------------------------------------------------------- amplitude
    # The channel that does NOT trade against the ranking. The map
    #
    #     p -> m * (p / m) ** k          (m = pooled mean rate, k > 0)
    #
    # is strictly increasing in p for every k, so all ranks -- and therefore
    # both rank-based gate statistics -- are preserved EXACTLY. It stays
    # positive for every p > 0 (the shipped surface is floored at 1e-9), so
    # no boundary ties are introduced. Expanding the deviations by k raises
    # var(pred) relative to cov(obs, pred), which divides the calibration
    # slope by roughly k.
    #
    # The affine form m + k(p-m) is the obvious alternative and is WRONG here:
    # for p below m(1 - 1/k) it goes negative and has to be clipped to zero,
    # which piles a large mass of distinct rows onto one value. That tie mass
    # lowers the surface Spearman too (0.9896 -> 0.9375 at k=1.46), so the
    # gates appear to pass while the ranking is simply being destroyed -- a
    # fake pass of exactly the kind this gate was tightened to catch.
    #
    # Reported so the SIZE of the required correction is on the record: a
    # calibration fix that needs k=4 is not a calibration fix, it is a
    # different model. Measured post-hoc on the holdout -- shipping it means
    # fitting k on TRAINING rows inside 59, not here.
    m = float(np.exp(np.mean(np.log(np.maximum(
        df["pred_speeding"].to_numpy(float), 1e-12)))))
    print(f"\n[amplitude] p -> m(p/m)^k, m(geometric)={m:.5f}, "
          "full tilt strength:")
    arows = []
    for k in [1.0, 1.04, 1.08, 1.12, 1.2, 1.35, 1.46]:
        pred = m * (np.maximum(df["pred_speeding"].to_numpy(float), 1e-12)
                    / m) ** k
        d = pd.DataFrame({"fixture": df["fixture"],
                          "road_osmid": df["road_osmid"],
                          "pred": pred,
                          "observed": base["observed"].to_numpy(float)})
        tot = d.pivot_table(index="road_osmid", columns="fixture",
                            values="pred", aggfunc="sum").fillna(0.0)
        rho, _ = _pairwise_spearman(tot)
        o = d["observed"].to_numpy(float)
        p = d["pred"].to_numpy(float)
        msk = np.isfinite(o) & np.isfinite(p)
        slope = float(np.polyfit(p[msk], o[msk], 1)[0]) if msk.sum() > 10 else np.nan
        arows.append({"k": k, "calib_slope": round(slope, 4),
                      "surface_spearman": round(float(rho), 4),
                      "jaccard_top20": round(_jaccard(tot, 20), 3),
                      "calib_ok": bool(CAL_BAND[0] <= slope <= CAL_BAND[1]),
                      "spearman_ok": bool(rho <= HANDOFF_SPEARMAN_MAX)})
        print(f"   k={k:<5} slope={slope:5.3f} rho={rho:.4f} "
              f"jac20={arows[-1]['jaccard_top20']:.3f}")
    amp = pd.DataFrame(arows)
    amp["both_ok"] = amp["calib_ok"] & amp["spearman_ok"]
    print("\n" + amp.to_string(index=False))
    print(f"\nk values satisfying BOTH gates: "
          f"{amp.loc[amp['both_ok'], 'k'].tolist() or 'none'}")
    amp.to_csv(run / "reorder_amplitude_curve.csv", index=False)
    print(f"wrote {run / 'reorder_amplitude_curve.csv'}")


if __name__ == "__main__":
    main()
