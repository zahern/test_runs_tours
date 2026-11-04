#!/usr/bin/env python
"""Where does the 3-4x holdout under-prediction come from?

The shipped count surface is Y ~ NB2(mu) with eta = a + 1.0 * logexp + Xb:
the exposure is a PINNED offset, so

    mu = exp(a + Xb) * exp

and any holdout level error is either the offset column arriving on a
different scale than it was fitted on, or eta being evaluated somewhere the
fit never went. Those are different bugs with different fixes, so this
script separates them multiplicatively instead of eyeballing one total:

    obs/pred = [obs/E] / [pred/E]        (a rate ratio at a shared exposure)

and then splits that rate ratio into

    pred/E  vs  train  Y/E     -> what the model does to eta on holdout rows
    obs/E   vs  train  Y/E     -> what the world did, that the model missed

It also checks the exposure column itself, because the artifacts store it in
two different parameterisations and mixing them up invents a fake gap: the
15-min parquet's `exp` column is actually log1p(exposure) (59 writes
keys["exp"] = fw["logexp"]), while the cell probe's `exp` is the rate scale.

Inputs are the band+cross+NB arm's artifacts; nothing is refitted.

    python scripts/diagnose_exposure_transfer.py [arm_dir]

Writes <arm_dir>/exposure_transfer.csv and prints the decomposition.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_tail = sys.argv[1:] or ["results_tests/t9_ab_band_cross_nb"]
ARM = Path(_tail[0])
ARM = ARM if ARM.is_absolute() else ROOT / ARM
TZ = "Australia/Brisbane"


def load_calendar(frame: pd.DataFrame, col: str = "bin") -> pd.DataFrame:
    t = pd.to_datetime(frame[col], utc=True).dt.tz_convert(TZ)
    frame = frame.copy()
    frame[col] = t
    frame["hour"] = t.dt.hour
    frame["dow"] = t.dt.dayofweek
    frame["quarter"] = t.dt.hour * 4 + (t.dt.minute // 15)
    return frame


def rates(y: float, e: float) -> float:
    return float(y) / max(float(e), 1e-12)


def main() -> None:
    # ---- train rows: the cell probe is the final design frame, post-refit,
    # post-calibration, so its exposure is exactly what the model was fitted
    # against. Its `exp` column is the rate scale (logexp is stored beside it).
    probe = load_calendar(pd.read_csv(ARM / "game_day_cell_probe.csv.gz"))
    if "in_event" in probe.columns:
        probe = probe[probe["in_event"].astype(float) > 0.5]
    assert np.allclose(np.log1p(probe["exp"]), probe["logexp"], atol=1e-6), \
        "probe exp is not exp(logexp-1): the train exposure parameterisation " \
        "changed, recheck which column is log-scale before comparing arms"

    # ---- holdout rows: grid carries the exposure, the cells file carries the
    # observation. Join on (fixture, segment, bin) to get both plus calendar.
    grid = load_calendar(pd.read_parquet(
        ARM / "game_day_pred_15min.parquet",
        columns=["fixture", "road_osmid", "bin", "exp", "pred_flow",
                 "flow_uplift", "pred_speeding_count", "pred_speeding"]))
    obs = load_calendar(pd.read_csv(
        ARM / "game_day_pred_vs_actual_cells.csv.gz",
        usecols=["fixture", "road_osmid", "bin", "observed"]))
    n_cells = len(obs)
    hold = grid.merge(obs.drop(columns=["hour", "dow", "quarter"]),
                      on=["fixture", "road_osmid", "bin"], how="inner")
    assert len(hold) == n_cells, (
        f"exposure grid covers {len(hold):,} of {n_cells:,} holdout cells; "
        "the parquet is not the same row set the metrics scored")
    # 59 writes keys["exp"] = fw["logexp"], i.e. log1p of the offset. Verify
    # rather than assume: the log-scale column must track log1p(pred_flow),
    # the rate-scale one must not.
    _log_corr = float(np.corrcoef(hold["exp"], np.log1p(hold["pred_flow"]))[0, 1])
    _lvl_corr = float(np.corrcoef(hold["exp"], hold["pred_flow"])[0, 1])
    assert _log_corr > _lvl_corr, (
        "the grid `exp` column is NOT log-scale here; this arm's artifacts use "
        "a different parameterisation and expm1() below would be wrong")
    hold["E"] = np.expm1(hold["exp"])
    hold["cal_mult"] = hold["E"] / hold["pred_flow"].clip(lower=1e-12)

    # Restrict holdout to the hours the probe actually covers, so the rate
    # comparison is not confounded by 3am bins.
    hours = sorted(probe["hour"].unique())
    hold = hold[hold["hour"].isin(hours)]

    # ---- 1. does the offset arrive on the same scale? ----------------------
    tr_e = float(probe["exp"].sum())
    tr_n = len(probe)
    ho_e = float(hold["E"].sum())
    ho_n = len(hold)
    print("== exposure transfer (pinned offset, coef 1.0) ==")
    print(f"   train  rows={tr_n:,}  E/row={tr_e / tr_n:.4f}  "
          f"logexp={np.log1p(probe['exp']).mean():.4f}  "
          f"sd(logexp)={np.log1p(probe['exp']).std():.4f}")
    print(f"   holdout rows={ho_n:,}  E/row={ho_e / ho_n:.4f}  "
          f"logexp={hold['exp'].mean():.4f}  sd={hold['exp'].std():.4f}")
    print(f"   exposure ratio holdout/train = {(ho_e / ho_n) / (tr_e / tr_n):.3f}"
          f"  (delta logexp = {hold['exp'].mean() - np.log1p(probe['exp']).mean():+.4f})")
    print(f"   future calibration multiplier E/pred_flow: "
          f"mean={hold['cal_mult'].mean():.3f} "
          f"p10/50/90={hold['cal_mult'].quantile(.1):.2f}/"
          f"{hold['cal_mult'].median():.2f}/{hold['cal_mult'].quantile(.9):.2f}")

    # Matched-segment version: same road_osmid on both sides removes the
    # segment mix as an explanation for any residual gap.
    a = probe.groupby("road_osmid").agg(n_t=("Y", "size"), E_t=("exp", "sum"),
                                        Y_t=("Y", "sum"))
    b = hold.groupby("road_osmid").agg(n_h=("observed", "size"),
                                       E_h=("E", "sum"),
                                       Y_h=("observed", "sum"),
                                       P_h=("pred_speeding_count", "sum"))
    m = a.join(b, how="inner")
    m = m[(m["n_t"] >= 2) & (m["n_h"] >= 2)]
    r_exp = (m["E_h"].sum() / m["n_h"].sum()) / (m["E_t"].sum() / m["n_t"].sum())
    print(f"   matched segments={len(m):,} ({m['Y_t'].sum():.0f} train obs / "
          f"{m['Y_h'].sum():.0f} holdout obs): E/row ratio={r_exp:.3f}")

    # ---- 2. the rate decomposition ----------------------------------------
    tr_rate = rates(probe["Y"].sum(), tr_e)
    ho_obs = rates(hold["observed"].sum(), ho_e)
    ho_prd = rates(hold["pred_speeding_count"].sum(), ho_e)
    print("\n== rate per unit exposure (shared exposure basis) ==")
    print(f"   train  Y/E    = {tr_rate:.5f}")
    print(f"   holdout obs/E = {ho_obs:.5f}   ({ho_obs / tr_rate:.2f}x train)")
    print(f"   holdout pred/E= {ho_prd:.5f}   ({ho_prd / tr_rate:.2f}x train)")
    print(f"   => obs/pred = {ho_obs / ho_prd:.2f}x, "
          f"of which model level {ho_prd / tr_rate:.2f}x and "
          f"world {ho_obs / tr_rate:.2f}x")

    # ---- 3. where in eta? --------------------------------------------------
    # If the model's rate deficit is a flat multiplicative shift it is the
    # intercept / a lost offset; if it tracks a covariate it is that term.
    rows = []

    def bucket(frame, by, pred_col, obs_col, E_col, label):
        g = frame.groupby(by, observed=True)
        out = pd.DataFrame({
            "obs": g[obs_col].sum(), "pred": g[pred_col].sum(),
            "E": g[E_col].sum(), "n": g[obs_col].size()})
        out["obs_rate"] = out["obs"] / out["E"]
        out["pred_rate"] = out["pred"] / out["E"]
        out["deficit"] = out["obs_rate"] / out["pred_rate"].replace(0, np.nan)
        out["segment"] = label
        out["bucket"] = out.index.astype(str)
        return out.reset_index(drop=True)

    hold["E_q"] = pd.qcut(hold["E"], 8, duplicates="drop").astype(str)
    rows.append(bucket(hold, "hour", "pred_speeding_count", "observed", "E",
                       "holdout by hour"))
    rows.append(bucket(hold, "E_q", "pred_speeding_count", "observed", "E",
                       "holdout by exposure"))
    rows.append(bucket(hold, "fixture", "pred_speeding_count", "observed", "E",
                       "holdout by fixture"))
    tab = pd.concat(rows, ignore_index=True)
    tab["train_rate"] = tr_rate
    tab["pred_vs_train"] = tab["pred_rate"] / tr_rate
    tab["obs_vs_train"] = tab["obs_rate"] / tr_rate

    for seg in ("holdout by hour", "holdout by exposure", "holdout by fixture"):
        t = tab[tab["segment"] == seg]
        print(f"\n-- {seg}")
        print(t[["bucket", "n", "E", "obs_rate", "pred_rate", "obs_vs_train",
                 "pred_vs_train", "deficit"]].round(4).to_string(index=False))

    # ---- 4. why: the offset is not actually pinned ------------------------
    # logexp is pinned at 1.0, but the design also carries log_flow_z, a
    # z-scored version of the same segment flow. Its free coefficient eats
    # part of the elasticity back, so the surface's true response to flow is
    # 1 + c/b, not 1.
    coefs = pd.read_csv(ARM / "game_day_model_coefs.csv").set_index("variable")
    le = probe["logexp"].to_numpy(float)
    net = 1.0
    terms = {}
    for v in ("log_flow_z", "aadt_z"):
        if v not in probe.columns or v not in coefs.index:
            continue
        z = pd.to_numeric(probe[v], errors="coerce").to_numpy(float)
        ok = np.isfinite(z) & np.isfinite(le)
        if int(ok.sum()) < 100 or float(np.std(z[ok])) == 0.0:
            continue
        b = float(np.polyfit(z[ok], le[ok], 1)[0])
        c = float(coefs.loc[v, "coefficient"])
        if abs(b) < 0.10:
            continue            # no overlap with the offset; c/b is noise
        net += c / b
        terms[v] = {"slope_on_logexp": round(b, 4), "coef": round(c, 4),
                    "elasticity_removed": round(-c / b, 4)}
    print("\n== is the offset actually pinned? ==")
    for k, v in terms.items():
        print(f"   {k}: slope on logexp={v['slope_on_logexp']:.3f}, "
              f"free coef={v['coef']:+.3f} -> removes "
              f"{v['elasticity_removed']:.3f} of elasticity")
    print(f"   NET exposure elasticity = {net:.3f} (nominal 1.0)")
    amp = None
    for sib in (ARM.parent / "t9_ab_band_cross_amp"
                / "game_day_holdout_metrics.json",):
        if sib.exists():
            amp = json.loads(sib.read_text())["audit"].get("surface_amp", {}) \
                .get("in_sample_slope")
    if amp:
        print(f"   same defect seen as an in-sample calibration slope: "
              f"{float(amp):.4f} vs 1/{net:.3f} = {1.0 / net:.4f}. A surface "
              f"compressed by 1/e is exactly a slope of 1/e, so the "
              f"'amplitude problem' measured earlier IS this coefficient.")

    # ---- 5. shape: model gradient vs the world's --------------------------
    g = tab[tab["segment"] == "holdout by exposure"].sort_values("bucket")
    grad_model = g["pred_rate"].iloc[-1] / g["pred_rate"].iloc[0]
    grad_world = g["obs_rate"].iloc[-1] / g["obs_rate"].iloc[0]
    print("\n== exposure gradient (top octile rate / bottom octile rate) ==")
    print(f"   model pred/E: {grad_model:.3f}")
    print(f"   world  obs/E: {grad_world:.3f}")
    print(f"   model is {grad_world / max(grad_model, 1e-9):.1f}x too steep")

    tab.to_csv(ARM / "exposure_transfer.csv", index=False)
    summary = {
        "train_rows": tr_n, "holdout_rows": ho_n,
        "exposure_per_row_train": tr_e / tr_n,
        "exposure_per_row_holdout": ho_e / ho_n,
        "exposure_ratio": (ho_e / ho_n) / (tr_e / tr_n),
        "matched_segment_exposure_ratio": float(r_exp),
        "delta_logexp": float(hold["exp"].mean()
                              - np.log1p(probe["exp"]).mean()),
        "cal_mult_mean": float(hold["cal_mult"].mean()),
        "train_rate": tr_rate, "holdout_obs_rate": ho_obs,
        "holdout_pred_rate": ho_prd,
        "model_level_vs_train": ho_prd / tr_rate,
        "world_vs_train": ho_obs / tr_rate,
        "deficit": ho_obs / ho_prd,
        "net_exposure_elasticity": net,
        "elasticity_terms": terms,
        "in_sample_calibration_slope": amp,
        "gradient_model": float(grad_model),
        "gradient_world": float(grad_world),
    }
    (ARM / "exposure_transfer.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nwrote {ARM / 'exposure_transfer.csv'}")


if __name__ == "__main__":
    main()
