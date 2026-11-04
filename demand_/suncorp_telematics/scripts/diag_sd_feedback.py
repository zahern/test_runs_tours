"""Why does the SD-feedback arm hurt? Cell-level diagnostics.

Reads the same inputs as 66 (holdout cells csv + shipped parquet) and
quantifies the mechanism behind the no_gain verdict:

  - the guard's accept test was vacuous (lam*=1.0 everywhere -> the blend
    S2(lam*) is identical to S0, so mean delta = 0 and t = NaN); the
    informative test is the full SD swap (lam=0) vs shipped (lam=1)
  - scale and sign structure of sd_lift vs game_lift
  - correlation of each lift with observed counts, pooled and per fixture
  - the wrong-sign mechanism: cells where sign(sd_lift) opposes the
    residual (observed - base) are pushed away from the data
  - whether sd_lift carries ANY complementary signal (where game_lift ~ 0)
  - cross-fixture rank instability of the SD surface vs the shipped one
    (top-20 Jaccard per fixture): a deployment surface that re-ranks
    streets every game is churn, not signal

Writes results/sd_feedback_diagnostics.json and prints the story.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/"
            "suncorp_telematics")
RES = PROJ / "results"
EPS = 1e-12
KEY = ["fixture", "road_osmid", "bin"]


def poisson_dev(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0.0,
                        2.0 * (y * np.log(y / mu) - (y - mu)),
                        2.0 * mu)
    return term


def spearman(a, b):
    sa = pd.Series(np.asarray(a, dtype=float))
    sb = pd.Series(np.asarray(b, dtype=float))
    if sa.nunique() < 2 or sb.nunique() < 2:
        return float("nan")
    return float(sa.corr(sb, method="spearman"))


def clip(x):
    return np.clip(np.asarray(x, dtype=float), EPS, None)


def main():
    cells = pd.read_csv(RES / "game_day_pred_vs_actual_cells.csv.gz")
    fut = pd.read_parquet(RES / "game_day_pred_15min.parquet")
    cells["fixture"] = cells["fixture"].astype(str)
    cells["road_osmid"] = pd.to_numeric(cells["road_osmid"], errors="coerce")
    cells["bin"] = pd.to_datetime(cells["bin"], utc=True, errors="coerce")
    fut["fixture"] = fut["fixture"].astype(str)
    fut["road_osmid"] = pd.to_numeric(fut["road_osmid"], errors="coerce")
    fut["bin"] = pd.to_datetime(fut["bin"], utc=True, errors="coerce")
    fut = fut.drop_duplicates(subset=KEY)
    m = cells.merge(fut, on=KEY, how="left", validate="many_to_one",
                    suffixes=("", "_pq"))

    y = pd.to_numeric(m["observed"], errors="coerce").to_numpy(float)
    s0 = clip(pd.to_numeric(m["pred_speeding"], errors="coerce"))
    base = clip(pd.to_numeric(m["pred_nogame"], errors="coerce"))
    gl = s0 - base
    if "sd_lift" in m.columns:
        sl = pd.to_numeric(m["sd_lift"], errors="coerce").to_numpy(float)
        sl_src = "sd_lift"
    else:
        sl = (pd.to_numeric(m["pred_sd_game"], errors="coerce")
              - pd.to_numeric(m["pred_sd_nogame"], errors="coerce")
              ).to_numpy(float)
        sl_src = "pred_sd_game - pred_sd_nogame"
    ok = np.isfinite(y) & np.isfinite(s0) & np.isfinite(base) & np.isfinite(sl)
    y, s0, base, gl, sl = (a[ok] for a in (y, s0, base, gl, sl))
    fx = m.loc[ok, "fixture"].to_numpy()
    games = sorted(set(fx))
    print(f"[diag] cells={len(y):,} games={len(games)} sd_lift source={sl_src}")

    rep = json.loads((RES / "sd_feedback_stack_report.json").read_text())
    lam_star = rep["lambda_star_full"]
    lofo = rep["lofo"]
    d_s1 = np.array([r["delta_s1_vs_s0"] for r in lofo], dtype=float)
    mean_d = float(d_s1.mean())
    se = float(d_s1.std(ddof=1) / np.sqrt(len(d_s1)))
    tstat = mean_d / se if se > 0 else float("nan")
    print(f"[diag] guard accept test was vacuous: lam*={lam_star:.2f} -> "
          f"S2(lam*)==S0, mean_delta=0, t=NaN")
    print(f"[diag] informative test (full SD swap vs shipped): "
          f"mean_delta={mean_d:+.6f} se={se:.6f} t={tstat:+.2f} "
          f"-> SD arm worse on {int((d_s1 > 0).sum())}/{len(d_s1)} fixtures")

    def _desc(a, name):
        q = np.percentile(a, [0, 25, 50, 75, 100])
        return {"name": name, "mean": float(a.mean()), "sd": float(a.std()),
                "pct_negative": float(100 * (a < 0).mean()),
                "pct_zero": float(100 * (a == 0).mean()),
                "p0": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                "p75": float(q[3]), "p100": float(q[4])}

    desc_gl = _desc(gl, "game_lift")
    desc_sl = _desc(sl, "sd_lift")
    print(f"[diag] game_lift: mean={desc_gl['mean']:+.4f} sd={desc_gl['sd']:.4f} "
          f"neg={desc_gl['pct_negative']:.1f}% zero={desc_gl['pct_zero']:.1f}%")
    print(f"[diag] sd_lift  : mean={desc_sl['mean']:+.4f} sd={desc_sl['sd']:.4f} "
          f"neg={desc_sl['pct_negative']:.1f}% zero={desc_sl['pct_zero']:.1f}%")

    def _corr_block(a, name):
        out = {"pooled_pearson": float(pd.Series(a).corr(pd.Series(y))),
               "pooled_spearman": spearman(a, y), "per_fixture_spearman": {}}
        for g in games:
            mm = fx == g
            out["per_fixture_spearman"][g] = spearman(a[mm], y[mm])
        vals = [v for v in out["per_fixture_spearman"].values()
                if np.isfinite(v)]
        out["per_fixture_min"] = float(np.min(vals)) if vals else None
        out["per_fixture_max"] = float(np.max(vals)) if vals else None
        return out

    corr_gl = _corr_block(gl, "game_lift")
    corr_sl = _corr_block(sl, "sd_lift")
    print(f"[diag] corr with observed: game_lift pooled r="
          f"{corr_gl['pooled_pearson']:+.4f} rho={corr_gl['pooled_spearman']:+.4f} "
          f"(per-game rho {corr_gl['per_fixture_min']:+.3f}.."
          f"{corr_gl['per_fixture_max']:+.3f})")
    print(f"[diag] corr with observed: sd_lift   pooled r="
          f"{corr_sl['pooled_pearson']:+.4f} rho={corr_sl['pooled_spearman']:+.4f} "
          f"(per-game rho {corr_sl['per_fixture_min']:+.3f}.."
          f"{corr_sl['per_fixture_max']:+.3f})")

    r = y - base
    s1 = clip(base + sl)
    dd = poisson_dev(y, s1) - poisson_dev(y, s0)
    nz = sl != 0
    helpful = nz & (np.sign(sl) == np.sign(r))
    harmful = nz & (np.sign(sl) == -np.sign(r))
    big = nz & (np.abs(sl) >= np.percentile(np.abs(sl[nz]), 90))
    mech = {
        "n_cells": int(len(y)),
        "n_sd_nonzero": int(nz.sum()),
        "pct_helpful_sign": float(100 * helpful.sum() / max(nz.sum(), 1)),
        "pct_harmful_sign": float(100 * harmful.sum() / max(nz.sum(), 1)),
        "mean_ddev_helpful": float(dd[helpful].mean()) if helpful.any() else None,
        "mean_ddev_harmful": float(dd[harmful].mean()) if harmful.any() else None,
        "mean_abs_sd_lift_helpful": float(np.abs(sl[helpful]).mean())
        if helpful.any() else None,
        "mean_abs_sd_lift_harmful": float(np.abs(sl[harmful]).mean())
        if harmful.any() else None,
        "pct_harmful_in_top_decile_abs_sd": float(100 * (harmful & big).sum()
                                                  / max(big.sum(), 1)),
        "mean_ddev_all": float(dd.mean()),
        "pct_cells_worse": float(100 * (dd > 0).mean()),
        "corr_ddev_vs_sd_lift": float(pd.Series(dd).corr(pd.Series(sl))),
        "corr_ddev_vs_residual": float(pd.Series(dd).corr(pd.Series(r))),
    }
    print(f"[diag] wrong-sign mechanism: sd_lift nonzero in "
          f"{mech['n_sd_nonzero']:,} cells; sign helps in "
          f"{mech['pct_helpful_sign']:.1f}%, hurts in "
          f"{mech['pct_harmful_sign']:.1f}%")
    print(f"[diag] mean per-cell dDev: helpful {mech['mean_ddev_helpful']:+.5f} "
          f"vs harmful {mech['mean_ddev_harmful']:+.5f}; "
          f"{mech['pct_harmful_in_top_decile_abs_sd']:.0f}% of top-decile "
          f"|sd_lift| cells are harmed")
    print(f"[diag] overall: mean dDev={mech['mean_ddev_all']:+.5f}, "
          f"{mech['pct_cells_worse']:.1f}% of cells worse under full SD swap")

    gl0 = np.abs(gl) < 0.01
    sl0 = np.abs(sl) < 0.01
    comp = {
        "n_game_lift_near_zero": int(gl0.sum()),
        "corr_sd_lift_obs_where_game_lift_zero":
            spearman(sl[gl0], y[gl0]) if gl0.sum() > 50 else None,
        "n_sd_lift_near_zero": int(sl0.sum()),
        "corr_game_lift_obs_where_sd_lift_zero":
            spearman(gl[sl0], y[sl0]) if sl0.sum() > 50 else None,
    }
    print(f"[diag] complementarity: where game_lift~0 (n={comp['n_game_lift_near_zero']:,}), "
          f"corr(sd_lift, obs)={comp['corr_sd_lift_obs_where_game_lift_zero']}; "
          f"where sd_lift~0 (n={comp['n_sd_lift_near_zero']:,}), "
          f"corr(game_lift, obs)={comp['corr_game_lift_obs_where_sd_lift_zero']}")

    sd_rank = pd.to_numeric(fut["pred_sd_game"], errors="coerce")
    sp_rank = pd.to_numeric(fut["pred_speeding"], errors="coerce")
    fut2 = fut.assign(_sd=sd_rank, _sp=sp_rank)
    ref = None
    jac_sd, jac_sp = {}, {}
    for g in games:
        sub = fut2[fut2["fixture"].astype(str) == g]
        t_sd = set(sub.nlargest(20, "_sd")["road_osmid"])
        t_sp = set(sub.nlargest(20, "_sp")["road_osmid"])
        if ref is None:
            ref = (t_sd, t_sp)
            jac_sd[g], jac_sp[g] = 1.0, 1.0
        else:
            jac_sd[g] = len(t_sd & ref[0]) / len(t_sd | ref[0])
            jac_sp[g] = len(t_sp & ref[1]) / len(t_sp | ref[1])
    stab = {"top20_jaccard_sd_game_vs_first": jac_sd,
            "top20_jaccard_pred_speeding_vs_first": jac_sp}
    print(f"[diag] top-20 Jaccard vs first fixture: pred_speeding "
          f"{min(jac_sp.values()):.2f}..{max(jac_sp.values()):.2f}  vs  "
          f"pred_sd_game {min(jac_sd.values()):.2f}..{max(jac_sd.values()):.2f}")

    out = {"sd_lift_source": sl_src,
           "guard": {"lambda_star_full": lam_star,
                     "accept_test": "vacuous (S2(lam*)==S0)",
                     "informative_test_full_sd_swap": {
                         "mean_delta": mean_d, "se": se, "t": tstat,
                         "fixtures_worse": int((d_s1 > 0).sum()),
                         "n_fixtures": len(d_s1)}},
           "scale": {"game_lift": desc_gl, "sd_lift": desc_sl},
           "corr_with_observed": {"game_lift": corr_gl, "sd_lift": corr_sl},
           "wrong_sign_mechanism": mech,
           "complementarity": comp,
           "rank_stability_top20": stab}
    dest = RES / "sd_feedback_diagnostics.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"[diag] -> {dest.name}")


if __name__ == "__main__":
    main()
