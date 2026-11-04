#!/usr/bin/env python
"""Reconstruct the holdout surface of a pinned-offset arm with the offset restored.

WHY THIS EXISTS
---------------
metacount_nb.predict_metacount_sim rebuilt the model with
``build_model_from_manual_spec(...)`` but never passed ``offset_col``.
``fit()`` removes a pinned exposure from fixed/rdm/grouped/hetero terms (it is
passed separately to build_model_from_manual_spec), so every --exposure offset
arm predicted exp(Xb) -- the exposure-NORMALISED rate -- while the fit itself
was correct (identical AIC). The error is a pure multiplicative factor per row:

    pred_shipped = pred_correct * exp(-logexp)     (keys["exp"] = fw["logexp"])

and 59 stores that logexp in the shipped grid, so an already-shipped arm can be
corrected exactly, without refitting:

    pred_correct = pred_shipped * exp(parquet["exp"])

The factor is exp(logexp) = 1 + (rate-scale exposure), NOT expm1(logexp): the
offset enters eta additively in log space. Multiplying by expm1 instead leaves
the surface ~24% low at the mean, so the distinction matters.

Verified by scripts/check_offset_predict.py, which fits a panel with a known
offset: with the offset passed the draw-averaged prediction reproduces the
fitted mean (ratio 0.9997 ≒ 1); withheld, it lands at 0.147.

This script is a diagnostic for arms produced BEFORE the fix. A run made after
it already ships the corrected numbers, and this becomes a consistency check.

    python scripts/reconstruct_offset_surface.py [arm_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARM = Path(sys.argv[1] if len(sys.argv) > 1 else "results_tests/t9_ab_band_cross_pindrop")
ARM = ARM if ARM.is_absolute() else ROOT / ARM


def slope_int(obs, pred) -> tuple[float, float]:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred) & (pred > 0)
    if ok.sum() < 10:
        return float("nan"), float("nan")
    s, i = np.polyfit(pred[ok], obs[ok], 1)
    return float(s), float(i)


def main() -> None:
    js = ARM / "game_day_holdout_metrics.json"
    audit = json.loads(js.read_text())["audit"] if js.exists() else {}
    mode = ((audit.get("refit") or {}).get("exposure_mode")
            or (audit.get("refit") or {}).get("offset_col"))
    if mode != "offset-pinned":
        print(f"NOTE: {ARM.name} is not an offset-pinned arm (exposure={mode!r}); "
              "there is no dropped offset to restore.")

    grid = pd.read_parquet(
        ARM / "game_day_pred_15min.parquet",
        columns=["fixture", "road_osmid", "bin", "exp", "pred_speeding"])
    obs = pd.read_csv(ARM / "game_day_pred_vs_actual_cells.csv.gz",
                      usecols=["fixture", "road_osmid", "bin", "observed"])
    for d in (grid, obs):
        d["bin"] = pd.to_datetime(d["bin"], utc=True).astype("int64")
    m = grid.merge(obs, on=["fixture", "road_osmid", "bin"], how="inner")
    # parquet["exp"] is logexp; the restore factor is exp(logexp).
    m["F"] = np.exp(m["exp"].to_numpy(float))
    m["E_rate"] = np.expm1(m["exp"].to_numpy(float))
    m = m[m["F"] > 0].copy()
    m["pred_fix"] = m["pred_speeding"] * m["F"]

    probe = pd.read_csv(ARM / "game_day_cell_probe.csv.gz")
    if "in_event" in probe.columns:
        probe = probe[probe["in_event"].astype(float) > 0.5]
    ycol = "Y" if "Y" in probe.columns else "trips_mean_gt_10"
    tr_rate = float(probe[ycol].sum() / probe["exp"].sum())

    y = m["observed"].to_numpy(float)
    p0 = m["pred_speeding"].to_numpy(float)
    p1 = m["pred_fix"].to_numpy(float)
    F = m["F"].to_numpy(float)
    keep = y > 0
    harm = float(y[keep].sum() / np.sum(y[keep] / F[keep]))
    s0, _ = slope_int(y, p0)
    s1, _ = slope_int(y, p1)

    print(f"== {ARM.name}: {len(m):,} holdout cells, obs={y.sum():,.0f} ==")
    print(f"   restore factor: mean={F.mean():.4f}  observed-weighted "
          f"harmonic mean={harm:.3f}  (rate-scale exposure mean="
          f"{m['E_rate'].mean():.4f})")
    print(f"   train rate Y/E={tr_rate:.6f}")
    print("")
    print(f"   {'':<20}{'as shipped':>13}{'offset restored':>17}")
    print(f"   {'pred total':<20}{p0.sum():>13,.0f}{p1.sum():>17,.0f}")
    print(f"   {'obs/pred':<20}{y.sum() / p0.sum():>13.3f}{y.sum() / p1.sum():>17.3f}")
    print(f"   {'pred/E over Y/E':<20}"
          f"{(p0.sum() / m['E_rate'].sum()) / tr_rate:>13.3f}"
          f"{(p1.sum() / m['E_rate'].sum()) / tr_rate:>17.3f}")
    print(f"   {'calib slope':<20}{s0:>13.4f}{s1:>17.4f}")

    # Same cut points on both sides, so the octile axis is comparable.
    m["oct"] = pd.qcut(m["E_rate"], 8, labels=False, duplicates="drop")
    rows = []
    for o, g in m.groupby("oct"):
        rows.append({"octile": int(o) + 1, "n": len(g),
                     "E_share": g["E_rate"].sum() / m["E_rate"].sum(),
                     "obs_share": g["observed"].sum() / y.sum(),
                     "deficit_shipped": g["observed"].sum() / g["pred_speeding"].sum(),
                     "deficit_fixed": g["observed"].sum() / g["pred_fix"].sum()})
    t = pd.DataFrame(rows)
    print("\n   per exposure octile (obs/pred):")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = ARM / "offset_reconstruction.csv"
    t.to_csv(out, index=False)
    print(f"\n   -> {out}")


if __name__ == "__main__":
    main()
