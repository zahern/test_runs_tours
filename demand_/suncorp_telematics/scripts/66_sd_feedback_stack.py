"""Stage 1 of the SD-feedback job: close the loop ONLY if it earns its place.

THE QUESTION
------------
The shipped surface pred_speeding predicts next-game finable speeding counts
directly. The SD (speed-density) twin produces its own game-lift sd_lift. The
old calibration (calibrate_sd_lift in 59) fitted and applied a scale on the
same fixture-level points it was judged on -- n~35, errors-in-variables, and
a statistically zero slope (b=-0.104+-0.342, p=0.76) -- so the pipeline
refuses it and the loop stays open.

This job asks the honest, cell-level version with out-of-sample discipline.
With base = the no-game baseline and

    S0 = base + game_lift                      (shipped direct surface)
    S2 = base + lam*game_lift + (1-lam)*sd_lift

lam is chosen on five holdout games and scored on the sixth
(leave-one-fixture-out over the 6 holdout fixtures), so the weight is
estimated on data it is not judged on. The scoring rule is the Poisson
deviance (the count-appropriate rule); the blend is accepted only if the
paired per-game deviance change vs S0 has mean < 0 with |mean/se| >= 2 --
the same guard style as the SD calibration it replaces.

WHAT IS MODIFIED
----------------
Nothing that ships. On accept, a NEW file game_day_pred_15min_sdfb.parquet
(copy of the shipped parquet + pred_speeding_sdfb at the accepted lam) is
written for a stage-60 MIP arm to consume. The shipped parquet, search
checkpoint and figures are never touched. On fail, the report says why and
stage 2 (feature-arm re-search) is not submitted.

HUMAN OVERRIDE
--------------
--override [--override-lam L] writes the SD-stacked surface regardless of
the guard, for a decision-maker who elects to deploy it against the
holdout evidence. decision becomes "override"; the report records the
guard verdict, the contrary evidence (full-SD-swap paired test), the
lambda and the date. The guard's own numbers are never rewritten.

INPUTS / OUTPUTS
----------------
in : results/game_day_pred_vs_actual_cells.csv.gz  (holdout cells: observed)
     results/game_day_pred_15min.parquet           (base/lift columns)
out: results/sd_feedback_stack_report.json
     figures/sd_feedback_stack.png
     results/game_day_pred_15min_sdfb.parquet      (accept only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/"
            "suncorp_telematics")
RES = PROJ / "results"
FIGS = PROJ / "figures"
LAM_GRID = np.round(np.linspace(0.0, 1.0, 21), 3)
EPS = 1e-12
KEY = ["fixture", "road_osmid", "bin"]
CARRY = ["observed", "pred_speeding", "pred_speeding_count", "pred_sd_game",
         "game_lift"]
PQ_WANT = ("fixture", "road_osmid", "bin", "pred_speeding", "pred_nogame",
           "game_lift", "pred_sd_game", "pred_sd_nogame", "sd_lift")


def poisson_dev(y, mu):
    """Cell-wise Poisson deviance 2[y*ln(y/mu) - (y - mu)]; y=0 -> 2*mu.

    The count-appropriate scoring rule. Used for BOTH the lam selection and
    the paired accept/reject test, so the decision is made on the same
    likelihood family the count model is fitted under.
    """
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0.0,
                        2.0 * (y * np.log(y / mu) - (y - mu)),
                        2.0 * mu)
    return term


def _spearman(a, b):
    sa = pd.Series(np.asarray(a, dtype=float))
    sb = pd.Series(np.asarray(b, dtype=float))
    if sa.nunique() < 2 or sb.nunique() < 2:
        return float("nan")
    return float(sa.corr(sb, method="spearman"))


def top_quartile_share(y, mu):
    """Share of observed events caught by the top predicted quartile of
    cells (25% = chance); monotone in the ranking, so invariant to lam
    except through ties/ordering."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if len(y) < 8 or y.sum() <= 0:
        return float("nan")
    idx = np.argsort(-mu, kind="stable")
    parts = np.array_split(y[idx], 4)
    return float(parts[0].sum() / max(y.sum(), 1e-9))


def score(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return {"dev_mean": float(np.mean(poisson_dev(y, mu))),
            "dev_total": float(np.sum(poisson_dev(y, mu))),
            "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
            "spearman": _spearman(y, mu),
            "top_quartile_obs_share": top_quartile_share(y, mu),
            "pred_total": float(np.sum(mu))}


def _parquet_cols(path):
    import pyarrow.parquet as papq
    return set(papq.ParquetFile(str(path)).schema.names)


def load_merged(cells_path, parquet_path):
    """Holdout cells joined to the parquet's lift columns on the cell key."""
    cells = pd.read_csv(cells_path)
    missing = [c for c in ("fixture", "road_osmid", "bin", "observed",
                           "pred_speeding") if c not in cells.columns]
    if missing:
        raise SystemExit(f"[66] FATAL: cells csv lacks {missing}")
    cells = cells[[c for c in KEY + CARRY if c in cells.columns]].copy()
    for c in CARRY:
        if c in cells.columns and c != "fixture":
            cells[c] = pd.to_numeric(cells[c], errors="coerce")
    cells["fixture"] = cells["fixture"].astype(str)
    cells["road_osmid"] = pd.to_numeric(cells["road_osmid"], errors="coerce")
    cells["bin"] = pd.to_datetime(cells["bin"], utc=True, errors="coerce")
    cells = cells.dropna(subset=KEY + ["observed"]).reset_index(drop=True)

    have = _parquet_cols(parquet_path)
    fut = pd.read_parquet(str(parquet_path),
                          columns=[c for c in PQ_WANT if c in have])
    for c in PQ_WANT:
        if c in fut.columns and c not in KEY:
            fut[c] = pd.to_numeric(fut[c], errors="coerce")
    fut["fixture"] = fut["fixture"].astype(str)
    fut["road_osmid"] = pd.to_numeric(fut["road_osmid"], errors="coerce")
    fut["bin"] = pd.to_datetime(fut["bin"], utc=True, errors="coerce")
    fut = (fut.dropna(subset=KEY).drop_duplicates(subset=KEY)
              .reset_index(drop=True))
    merged = cells.merge(fut, on=KEY, how="left", suffixes=("", "_pq"),
                         validate="many_to_one")
    return merged, have


def derive(merged, have, warnings):
    """base / game_lift / sd_lift with explicit fallbacks; every fallback is
    recorded in the report so a silent re-definition cannot change the
    decision unnoticed."""
    s0 = np.clip(pd.to_numeric(merged["pred_speeding"],
                               errors="coerce").to_numpy(float), EPS, None)
    if "pred_nogame" in merged.columns and \
            np.isfinite(pd.to_numeric(merged["pred_nogame"],
                                      errors="coerce")).any():
        base = np.clip(pd.to_numeric(merged["pred_nogame"],
                                     errors="coerce").to_numpy(float),
                       EPS, None)
    elif "game_lift" in merged.columns and \
            np.isfinite(pd.to_numeric(merged["game_lift"],
                                      errors="coerce")).any():
        base = np.clip(s0 - pd.to_numeric(merged["game_lift"],
                                          errors="coerce").to_numpy(float),
                       EPS, None)
        warnings.append("pred_nogame absent; base := pred_speeding - "
                        "game_lift")
    else:
        raise SystemExit("[66] FATAL: neither pred_nogame nor game_lift "
                         "available; cannot build the baseline")
    gl = s0 - base
    if "sd_lift" in merged.columns and \
            np.isfinite(pd.to_numeric(merged["sd_lift"],
                                      errors="coerce")).any():
        sl = pd.to_numeric(merged["sd_lift"], errors="coerce").to_numpy(float)
    elif "pred_sd_game" in merged.columns and "pred_sd_nogame" in merged.columns:
        sl = (pd.to_numeric(merged["pred_sd_game"], errors="coerce")
              - pd.to_numeric(merged["pred_sd_nogame"], errors="coerce")
              ).to_numpy(float)
        warnings.append("sd_lift absent; derived as pred_sd_game - "
                        "pred_sd_nogame")
    else:
        raise SystemExit("[66] FATAL: no sd_lift channel in the parquet; "
                         "stage-1 needs the SD twin")
    bad = ~(np.isfinite(sl) & np.isfinite(base) & np.isfinite(s0))
    if bad.any():
        warnings.append(f"dropped {int(bad.sum())} cells with non-finite "
                        "base/sd_lift")
    keep = ~bad
    return keep, s0, base, gl, sl


def mu_lam(base, gl, sl, lam):
    return np.clip(base + lam * gl + (1.0 - lam) * sl, EPS, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells",
                    default=str(RES / "game_day_pred_vs_actual_cells.csv.gz"))
    ap.add_argument("--parquet",
                    default=str(RES / "game_day_pred_15min.parquet"))
    ap.add_argument("--out-dir", default=str(RES))
    ap.add_argument("--fig-dir", default=str(FIGS))
    ap.add_argument("--override", action="store_true",
                    help="human override: write the SD-stacked surface "
                         "regardless of the guard; the guard verdict and "
                         "contrary evidence are recorded in the report")
    ap.add_argument("--override-lam", type=float, default=0.0,
                    help="blend weight written by --override "
                         "(1 = shipped game_lift, 0 = pure SD swap)")
    args = ap.parse_args()
    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    cells_path, parquet_path = Path(args.cells), Path(args.parquet)
    for p in (cells_path, parquet_path):
        if not p.exists():
            raise SystemExit(f"[66] FATAL: missing input {p}")
    warnings = []

    merged, have = load_merged(cells_path, parquet_path)
    keep, s0, base, gl, sl = derive(merged, have, warnings)
    y = pd.to_numeric(merged["observed"], errors="coerce").to_numpy(float)
    s0, base, gl, sl, y = (a[keep] for a in (s0, base, gl, sl, y))
    fx = merged.loc[keep, "fixture"].to_numpy()
    games = sorted(set(fx))
    if len(y) < 1000 or len(games) < 3:
        raise SystemExit(f"[66] FATAL: too little holdout data "
                         f"(cells={len(y)}, games={len(games)})")

    s1 = mu_lam(base, gl, sl, 0.0)
    twin_gap = float(np.nanmax(np.abs(
        s1 - np.clip(pd.to_numeric(merged.loc[keep, "pred_sd_game"],
                                   errors="coerce").to_numpy(float),
                     EPS, None)))) if "pred_sd_game" in merged.columns else \
        float("nan")
    corr_pool = float(pd.Series(gl).corr(pd.Series(sl), method="pearson")) \
        if np.std(gl) > 0 and np.std(sl) > 0 else float("nan")
    corr_fx = {}
    for g in games:
        m = fx == g
        corr_fx[g] = (float(pd.Series(gl[m]).corr(pd.Series(sl[m])))
                      if np.std(gl[m]) > 0 and np.std(sl[m]) > 0
                      else float("nan"))

    print(f"[66] cells={len(y):,}  games={len(games)}  "
          f"nonzero_obs={int((y > 0).sum()):,}")
    print(f"[66] corr(game_lift, sd_lift) pooled={corr_pool:+.3f}")
    print(f"[66] max |base+sd_lift - pred_sd_game| = {twin_gap:.4g} "
          "(baseline-convention gap between regression and SD twin)")

    sweep = []
    for lam in LAM_GRID:
        st = score(y, mu_lam(base, gl, sl, float(lam)))
        st["lambda"] = float(lam)
        sweep.append(st)
    lam_full = float(min(sweep, key=lambda r: r["dev_mean"])["lambda"])
    print(f"[66] full-sample argmin dev lam*={lam_full:.2f} "
          f"(dev {min(r['dev_mean'] for r in sweep):.6f} vs "
          f"S0 {score(y, s0)['dev_mean']:.6f})")

    idx = {g: np.where(fx == g)[0] for g in games}
    allix = np.arange(len(y))
    lofo = []
    for g in games:
        te, tr = idx[g], np.setdiff1d(allix, idx[g])
        dev_tr = [float(np.mean(poisson_dev(y[tr],
                                            mu_lam(base, gl, sl, float(lam))[tr])))
                  for lam in LAM_GRID]
        lam_g = float(LAM_GRID[int(np.argmin(dev_tr))])
        d_lam = float(np.mean(poisson_dev(y[te],
                                          mu_lam(base, gl, sl, lam_g)[te])))
        d_s0 = float(np.mean(poisson_dev(y[te], s0[te])))
        d_s1 = float(np.mean(poisson_dev(y[te], s1[te])))
        lofo.append({"fixture": g, "lambda": lam_g,
                     "dev_lambda": d_lam, "dev_s0": d_s0, "dev_s1": d_s1,
                     "delta_lambda_vs_s0": d_lam - d_s0,
                     "delta_s1_vs_s0": d_s1 - d_s0})
        print(f"[66] LOFO {g[:34]:34s} lam={lam_g:.2f} "
              f"d_dev={d_lam - d_s0:+.6f} (vs S0)")

    dd = np.array([r["delta_lambda_vs_s0"] for r in lofo], dtype=float)
    mean_d = float(dd.mean())
    se = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else \
        float("nan")
    tstat = mean_d / se if se and np.isfinite(se) and se > 0 else float("nan")
    accept = bool(mean_d < 0 and np.isfinite(tstat) and abs(tstat) >= 2.0)
    override = bool(getattr(args, "override", False))
    lam_o = float(args.override_lam)
    if not 0.0 <= lam_o <= 1.0:
        raise SystemExit("[66] FATAL: --override-lam must be in [0, 1]")
    decision = "accept" if accept else ("override" if override
                                        else "no_gain")
    print(f"[66] paired delta_lam_vs_s0: mean={mean_d:+.6f} se={se:.6f} "
          f"t={tstat:+.2f}  -> {'ACCEPT' if accept else 'NO demonstrated gain'}")

    report = {"decision": decision,
              "guard": {"mean_delta": mean_d, "se": se, "t": tstat,
                        "rule": "accept iff mean<0 and |t|>=2",
                        "note": "vacuous at lam*=1 (S2==S0): the "
                                "informative test is the full SD swap "
                                "(lam=0) vs shipped (lam=1)"},
              "lambda_star_full": lam_full,
              "lambda_lofo": {r["fixture"]: r["lambda"] for r in lofo},
              "lofo": lofo, "sweep": sweep,
              "n_cells": int(len(y)), "n_games": int(len(games)),
              "corr_game_lift_sd_lift_pooled": corr_pool,
              "corr_game_lift_sd_lift_per_game": corr_fx,
              "twin_baseline_gap_max": twin_gap,
              "s0_full": score(y, s0), "s1_full": score(y, s1),
              "warnings": warnings, "override": None}
    if override:
        _d = np.array([r["delta_s1_vs_s0"] for r in lofo], dtype=float)
        _se = float(_d.std(ddof=1) / np.sqrt(len(_d))) if len(_d) > 1 \
            else float("nan")
        report["override"] = {
            "by": "human directive (ahernz, 2026-10-16)",
            "lambda": lam_o,
            "reason": "decision-maker elected to deploy the SD-feedback "
                      "surface despite the no_gain guard",
            "contrary_evidence": {
                "full_sd_swap_vs_shipped": {
                    "mean_delta_dev": float(_d.mean()),
                    "se": _se,
                    "t": float(_d.mean() / _se)
                    if np.isfinite(_se) and _se > 0 else float("nan"),
                    "fixtures_worse": int((_d > 0).sum()),
                    "n_fixtures": int(len(_d))},
                "s0_dev_mean": score(y, s0)["dev_mean"],
                "s1_dev_mean": score(y, s1)["dev_mean"],
                "diagnostics": "results/sd_feedback_diagnostics.json"}}
    (out_dir / "sd_feedback_stack_report.json").write_text(
        json.dumps(report, indent=2, default=float))
    print(f"[66] report -> sd_feedback_stack_report.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
        ax = axes[0]
        lam_sweep = [r["lambda"] for r in sweep]
        dev_sweep = [r["dev_mean"] for r in sweep]
        ax.plot(lam_sweep, dev_sweep, "-o", ms=3, lw=1.4, color="#1f77b4",
                label="mean Poisson deviance")
        ax.axhline(score(y, s0)["dev_mean"], ls="--", lw=1.0,
                   color="#d62728", label="shipped (lam=1 = S0)")
        ax.axvline(lam_full, ls=":", lw=1.0, color="#2ca02c",
                   label=f"full-sample argmin lam*={lam_full:.2f}")
        ax.set_xlabel("lambda  (1 = shipped game_lift, 0 = pure SD-lift swap)")
        ax.set_ylabel("mean Poisson deviance (all holdout cells)")
        ax.set_title("Blend weight sweep", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[1]
        gnames = [r["fixture"][:22] for r in lofo]
        deltas = [r["delta_lambda_vs_s0"] for r in lofo]
        cols = ["#2ca02c" if d < 0 else "#d62728" for d in deltas]
        ax.bar(range(len(deltas)), deltas, color=cols)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(deltas)))
        ax.set_xticklabels(gnames, rotation=35, ha="right", fontsize=6.5)
        ax.set_ylabel("held-out deviance change vs S0")
        ax.set_title(f"Per-game out-of-sample check "
                     f"(mean={mean_d:+.4f}, t={tstat:+.2f}; "
                     f"{'accept' if accept else 'no gain'})", fontsize=9)
        fig.suptitle("SD-feedback stacking: does the SD twin earn a weight "
                     "out of sample?", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(str(fig_dir / "sd_feedback_stack.png"), dpi=130)
        plt.close(fig)
        print("[66] figure -> sd_feedback_stack.png")
    except Exception as exc:  # noqa: BLE001
        print(f"[66] WARN: figure failed ({exc})")

    if accept or override:
        fut = pd.read_parquet(str(parquet_path))
        fcol = {c: pd.to_numeric(fut[c], errors="coerce").to_numpy(float)
                for c in ("pred_speeding", "pred_nogame", "game_lift",
                          "sd_lift") if c in fut.columns}
        if "pred_nogame" in fcol:
            fbase = np.clip(fcol["pred_nogame"], EPS, None)
        else:
            fbase = np.clip(fcol["pred_speeding"] - fcol["game_lift"],
                            EPS, None)
        if "sd_lift" in fcol:
            fsl = fcol["sd_lift"]
        else:
            fsl = (pd.to_numeric(fut["pred_sd_game"], errors="coerce")
                   - pd.to_numeric(fut["pred_sd_nogame"], errors="coerce")
                   ).to_numpy(float)
        fgl = np.clip(fcol["pred_speeding"], EPS, None) - fbase
        lam_star = float(min(sweep, key=lambda r: r["dev_mean"])["lambda"]) \
            if accept else lam_o
        fut["pred_speeding_sdfb"] = np.clip(
            fbase + lam_star * fgl + (1.0 - lam_star) * fsl, EPS, None)
        dest = out_dir / "game_day_pred_15min_sdfb.parquet"
        fut.to_parquet(str(dest), index=False)
        if override:
            print(f"[66] OVERRIDE: {dest.name} written with "
                  f"pred_speeding_sdfb at lam={lam_star:.2f} despite the "
                  f"no_gain guard; contrary evidence recorded in the report "
                  f"and sd_feedback_diagnostics.json")
        else:
            print(f"[66] ACCEPT: {dest.name} written with pred_speeding_sdfb "
                  f"(lam*={lam_star:.2f})")
    else:
        print("[66] NO GAIN: loop stays open; stage 2 is not justified; "
              "no sdfb parquet written")
    print("[66] done")


if __name__ == "__main__":
    main()
