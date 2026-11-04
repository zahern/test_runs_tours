"""
68_copula_aggression.py - joint speeding x hard-braking copula + aggression score.

Stage 2 of the copula extension (see copula_hard_braking_spec.md). Joins the
speeding panel (script 58) and the hard-braking panel (script 58b) on
(road_osmid, 15-min bin) and models the two margins jointly:

  * margin S: speeding_trips per cell, an empirical-Bayes exposure rate
  * margin B: braking_events per cell, the same construction
  * copula:   Gaussian / Clayton / Gumbel / Frank fitted to the marginal PITs
              by maximum likelihood, selected by AIC

The aggression score is the per-cell joint-exceedance probability
P(Y_S > a, Y_B > b) = 1 - F_S(a) - F_B(b) + C(F_S(a), F_B(b)) under the fitted
copula. The thresholds (a, b) are NOT hand-picked: we sweep them over training
quantiles and pick the operating point q that maximises the average precision
(AP) of the joint-event ranking on TRAIN, then report the held-out TEST AP at
that q plus the full sweep. Deployment needs no threshold at all -- rank by the
score and take the top-K sites (K = your budget).

Gates (recorded in the report, never silently passed):
  G1  braking margin beats an intercept-only rate on held-out deviance
  G2  the chosen copula beats independence (LRT) and the dependence is positive
  G3  joint score AP beats each margin alone at the same q

Inputs:  results/segment_15min_panel.parquet          (speeding panel, 58)
         results/segment_15min_panel_braking.parquet  (braking panel, 58b)
         results/game_day_pred_15min.parquet          (defines the 6 holdouts)
         data/fixture_calendar.csv                    (all 41 fixtures)
Outputs: results/copula_aggression_report.json
         results/segment_aggression.parquet
         figures/copula_aggression.png

Usage:
  python scripts/68_copula_aggression.py --smoke          # tiny, fast check
  python scripts/68_copula_aggression.py                  # full
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results"
FIG = PROJ / "figures"
PANEL_S = RES / "segment_15min_panel.parquet"
PANEL_B = RES / "segment_15min_panel_braking.parquet"
PRED = RES / "game_day_pred_15min.parquet"
CAL = PROJ / "data" / "fixture_calendar.csv"
TZ = "Australia/Brisbane"
EPS = 1e-6
K_GRID = (20.0, 100.0, 500.0, 2000.0, 10000.0)
Q_GRID = tuple(np.round(np.linspace(0.50, 0.98, 13), 3))
FAMILIES = ("gaussian", "clayton", "gumbel", "frank")


# --------------------------------------------------------------------------- #
# load + join
# --------------------------------------------------------------------------- #
def load_cells(panel_s=PANEL_S, panel_b=PANEL_B):
    s = pd.read_parquet(panel_s, columns=["road_osmid", "bin",
                                          "vehicle_count", "speeding_trips"])
    b = pd.read_parquet(panel_b, columns=["road_osmid", "bin", "braking_events"])
    cells = s.merge(b, on=["road_osmid", "bin"], how="left")
    cells["braking_events"] = (cells["braking_events"].fillna(0)
                               .astype("int64"))
    cells = cells[cells["vehicle_count"] > 0].reset_index(drop=True)
    cells["exposure"] = cells["vehicle_count"].astype("float64")
    return cells


def holdout_kickoffs(pred_path=PRED):
    p = pd.read_parquet(pred_path, columns=["fixture"])
    out = []
    for fx in p["fixture"].astype(str).unique():
        if "@" in fx:
            out.append(pd.Timestamp(fx.split("@")[-1].strip()))
    return sorted(set(out))


def fixture_windows(cal_path=CAL, holdouts=None, pre_h=3.0, post_h=4.0, in_h=3.0):
    cal = pd.read_csv(cal_path)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"])
    holds = set(pd.Timestamp(k) for k in (holdouts or []))
    cal["is_test"] = cal["ko"].isin(holds)
    wins = []
    for _, r in cal.iterrows():
        wins.append({"label": r["label"], "ko": r["ko"], "test": bool(r["is_test"]),
                     "start": r["ko"] - pd.Timedelta(hours=pre_h),
                     "mid": r["ko"] + pd.Timedelta(hours=in_h),
                     "end": r["ko"] + pd.Timedelta(hours=post_h)})
    return cal, wins


def mark_cells(cells, wins):
    """Flag each cell train/test, game/non-game, its fixture and pre/in/post phase.

    Fixture labels are recorded for ALL windows (not only the holdouts) so the
    dependence-by-fixture analysis has training games to look at too.
    """
    bins = cells["bin"].dt.tz_localize(None)
    test = np.zeros(len(cells), bool)
    game = np.zeros(len(cells), bool)
    fixture = np.array([""] * len(cells), dtype=object)
    phase = np.array([""] * len(cells), dtype=object)
    for w in wins:
        s = w["start"].tz_localize(None)
        e = w["end"].tz_localize(None)
        ko = w["ko"].tz_localize(None)
        mid = w["mid"].tz_localize(None)
        m = (bins >= s) & (bins < e)
        game |= m
        fixture[m] = w["label"]
        phase[m & (bins < ko)] = "pre"
        phase[m & (bins >= ko) & (bins < mid)] = "in"
        phase[m & (bins >= mid)] = "post"
        if w["test"]:
            test |= m
    cells = cells.copy()
    cells["is_game"] = game
    cells["is_test"] = test
    cells["fixture"] = fixture
    cells["phase"] = phase
    return cells


# --------------------------------------------------------------------------- #
# empirical-Bayes exposure-rate margin
# --------------------------------------------------------------------------- #
def _poi_dev(y, mu):
    mu = np.maximum(mu, 1e-9)
    return float(2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
                            - (y - mu)))


def fit_eb_margin(df, target, exposure="exposure", k_grid=K_GRID):
    """Gamma-Poisson EB rate per segment; prior weight k chosen by train deviance.

    Returns (rate_by_segment Series, global_rate, chosen_k, seg_events, seg_exp)
    so callers can also compute the held-out deviance of a plain intercept model.
    """
    g = df.groupby("road_osmid").agg(ev=(target, "sum"), ex=(exposure, "sum"))
    lam = float(df[target].sum() / max(df[exposure].sum(), 1.0))
    y = g["ev"].to_numpy(float)
    ex = g["ex"].to_numpy(float)
    best, best_k = None, k_grid[0]
    for k in k_grid:
        rate = (y + lam * k) / (ex + k)
        dev = _poi_dev(y, ex * rate)
        if best is None or dev < best:
            best, best_k, best_rate = dev, k, rate
    return pd.Series(best_rate, index=g.index, name="rate"), lam, float(best_k), g


def cell_mu(cells, rate, game_mult, default_rate):
    r = (rate.reindex(cells["road_osmid"]).fillna(default_rate).to_numpy(float))
    mu = cells["exposure"].to_numpy(float) * r
    mu = np.where(cells["is_game"].to_numpy(), mu * game_mult, mu)
    return np.maximum(mu, 1e-9)


def game_multiplier(df, target, exposure="exposure"):
    g = df[df["is_game"]]
    if len(g) == 0 or df.loc[df["is_game"], exposure].sum() <= 0:
        return 1.0
    r_g = g[target].sum() / max(g[exposure].sum(), 1.0)
    r_a = df[target].sum() / max(df[exposure].sum(), 1.0)
    return float(np.clip(r_g / max(r_a, 1e-12), 0.1, 10.0))


def nb_alpha(y, mu):
    mu = np.maximum(mu, 1e-12)
    num = float(np.sum((y - mu) ** 2 - mu))
    den = float(np.sum(mu ** 2))
    return max(num / max(den, 1e-12), 1e-6)


def nb_cdf(y, mu, alpha):
    r = 1.0 / alpha
    p = r / (r + np.maximum(mu, 1e-12))
    return stats.nbinom.cdf(y, r, p)


# --------------------------------------------------------------------------- #
# copulas
# --------------------------------------------------------------------------- #
def cop_ll(family, param, u, v):
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    if family == "gaussian":
        r = float(np.clip(param, -0.999, 0.999))
        z1, z2 = stats.norm.ppf(u), stats.norm.ppf(v)
        ll = (-0.5 * np.log1p(-r * r)
              - (r * r * (z1 * z1 + z2 * z2) - 2 * r * z1 * z2)
              / (2 * (1 - r * r)))
        return float(np.sum(ll))
    if family == "clayton":
        t = float(param)
        A = np.maximum(u ** (-t) + v ** (-t) - 1.0, EPS)
        return float(np.sum(np.log1p(t) - (1 + t) * (np.log(u) + np.log(v))
                            - (2 + 1 / t) * np.log(A)))
    if family == "frank":
        t = float(param)
        num = -t * np.expm1(-t) * np.exp(-t * (u + v))
        den = (np.expm1(-t) + np.expm1(-t * u) * np.expm1(-t * v)) ** 2
        return float(np.sum(np.log(np.maximum(num / den, EPS))))
    if family == "gumbel":
        t = float(param)
        x, y = -np.log(u), -np.log(v)
        a = x ** t + y ** t
        dens = np.exp(-a ** (1 / t)) * (x * y) ** (t - 1) \
            * a ** (-2 + 1 / t) * (a ** (1 / t) + t - 1) / (u * v)
        return float(np.sum(np.log(np.maximum(dens, EPS))))
    raise ValueError(family)


def cop_cdf(family, param, u, v):
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    if family == "gaussian":
        z1, z2 = stats.norm.ppf(u), stats.norm.ppf(v)
        r = float(np.clip(param, -0.999, 0.999))
        mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]])
        return mvn.cdf(np.column_stack([z1, z2]))
    if family == "clayton":
        t = float(param)
        return np.maximum(u ** (-t) + v ** (-t) - 1.0, EPS) ** (-1 / t)
    if family == "frank":
        t = float(param)
        return -1 / t * np.log1p(np.expm1(-t * u) * np.expm1(-t * v) / np.expm1(-t))
    if family == "gumbel":
        t = float(param)
        x, y = -np.log(u), -np.log(v)
        return np.exp(-(x ** t + y ** t) ** (1 / t))
    raise ValueError(family)


def fit_copula(u, v):
    """Fit all families by MLE on PIT uniforms; return dicts + AIC choice."""
    z1, z2 = stats.norm.ppf(np.clip(u, EPS, 1 - EPS)), stats.norm.ppf(
        np.clip(v, EPS, 1 - EPS))
    rho = float(np.clip(np.corrcoef(z1, z2)[0, 1], -0.999, 0.999))

    def neg(fam):
        return lambda t: -cop_ll(fam, t, u, v)

    fits = {"gaussian": (1, rho, cop_ll("gaussian", rho, u, v))}
    for fam, lo, hi in (("clayton", 1e-3, 20.0),
                        ("gumbel", 1.0001, 20.0),
                        ("frank", -25.0, 25.0)):
        try:
            res = optimize.minimize_scalar(
                lambda t: -cop_ll(fam, t, u, v), bounds=(lo, hi), method="bounded")
            th = float(res.x)
            fits[fam] = (1, th, cop_ll(fam, th, u, v))
        except Exception:  # noqa: BLE001 - a family failing to fit is not fatal
            continue
    # independence reference loglik is 0 for all the u/v densities above
    table = []
    for fam, (k, par, ll) in fits.items():
        table.append({"family": fam, "param": float(par), "loglik": float(ll),
                      "k": int(k), "aic": float(-2 * ll + 2 * k)})
    table.sort(key=lambda r: r["aic"])
    best = table[0]
    indep_ll = 0.0
    return {
        "table": table, "best_family": best["family"],
        "best_param": best["param"], "best_loglik": best["loglik"],
        "independence_loglik": indep_ll,
        "lr_stat": float(2 * (best["loglik"] - indep_ll)),
        "gaussian_rho": rho,
    }


def joint_exceedance(family, param, a, b, mu_s, mu_b, alpha_s, alpha_b):
    fs = nb_cdf(a, mu_s, alpha_s)
    fb = nb_cdf(b, mu_b, alpha_b)
    c = cop_cdf(family, param, fs, fb)
    return np.clip(1.0 - fs - fb + c, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def average_precision(y, score):
    y = np.asarray(y, bool)
    n = int(y.sum())
    if n == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score, float))
    y = y[order]
    precision = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((precision * y).sum() / n)


def marginal_cdf(mu, alpha, thr):
    return nb_cdf(thr, mu, alpha)


def dependence_by_group(u, v, groups, min_n=2000):
    """Gaussian-copula rho (normal-score correlation) within each group label.

    This is the 'does the dependence change?' readout: grouped by phase
    (pre/in/post) or by fixture, rho is refitted on that subset only.
    """
    z1 = stats.norm.ppf(np.clip(u, EPS, 1 - EPS))
    z2 = stats.norm.ppf(np.clip(v, EPS, 1 - EPS))
    frame = pd.DataFrame({"z1": z1, "z2": z2, "g": np.asarray(groups)})
    out = {}
    for key, sub in frame.groupby("g"):
        if len(sub) >= int(min_n) and sub["z1"].std() > 0 and sub["z2"].std() > 0:
            out[str(key)] = {"rho": float(np.corrcoef(sub["z1"], sub["z2"])[0, 1]),
                             "n": int(len(sub))}
    return out


def lead_lag(cells_game, lags=(-2, -1, 0, 1, 2), step_min=15):
    """Cross-correlation corr(y_S(t), y_B(t+L)); L>0 means speeding leads braking.

    Answers the temporal-order question directly: if the peak sits at a positive
    lag, a segment's speeding precedes its hard braking, which is the signature
    of aggressive driving rather than independent co-occurrence.
    """
    if len(cells_game) == 0:
        return {}
    base = cells_game[["road_osmid", "bin", "speeding_trips",
                       "braking_events"]].copy()
    base["bin"] = base["bin"].dt.tz_localize(None)
    out = {}
    for lag in lags:
        sh = base[["road_osmid", "bin", "braking_events"]].copy()
        sh["bin"] = sh["bin"] + pd.Timedelta(minutes=int(lag) * step_min)
        m = base.merge(sh, on=["road_osmid", "bin"], suffixes=("", "_lead"))
        if len(m) > 100 and m["speeding_trips"].std() > 0 \
                and m["braking_events_lead"].std() > 0:
            out[int(lag)] = float(np.corrcoef(m["speeding_trips"],
                                              m["braking_events_lead"])[0, 1])
    return out


def _rho_summary(rho_by_group):
    vals = [v["rho"] for v in rho_by_group.values()]
    if not vals:
        return None
    return {"n_groups": len(vals), "mean": float(np.mean(vals)),
            "std": float(np.std(vals)), "min": float(np.min(vals)),
            "max": float(np.max(vals))}


def run(args):
    print("[68] loading cells...", flush=True)
    cells = load_cells()
    holdouts = holdout_kickoffs()
    _, wins = fixture_windows(holdouts=holdouts)
    cells = mark_cells(cells, wins)
    n_test_fixtures = len(set(cells.loc[cells["is_test"], "fixture"]) - {""})
    print(f"[68] cells={len(cells):,} game={int(cells['is_game'].sum()):,} "
          f"test={int(cells['is_test'].sum()):,} test_fixtures={n_test_fixtures}",
          flush=True)
    if n_test_fixtures == 0:
        raise SystemExit("test set empty: the holdout fixtures in "
                         f"{PRED.name} did not match {CAL.name}; check kickoffs")

    train = cells[~cells["is_test"]].reset_index(drop=True)
    test = cells[cells["is_test"]].reset_index(drop=True)
    if args.smoke:
        train = train.sample(min(len(train), 400_000), random_state=0).reset_index(drop=True)
        test = test.sample(min(len(test), 200_000), random_state=0).reset_index(drop=True)
    yS_tr = train["speeding_trips"].to_numpy(float)
    yB_tr = train["braking_events"].to_numpy(float)

    print("[68] fitting margins (EB exposure rates)...", flush=True)
    rateS, lamS, kS, gS = fit_eb_margin(train, "speeding_trips")
    rateB, lamB, kB, gB = fit_eb_margin(train, "braking_events")
    gmS = game_multiplier(train, "speeding_trips")
    gmB = game_multiplier(train, "braking_events")
    muS_tr = cell_mu(train, rateS, gmS, lamS)
    muB_tr = cell_mu(train, rateB, gmB, lamB)
    aS = nb_alpha(yS_tr, muS_tr)
    aB = nb_alpha(yB_tr, muB_tr)
    print(f"[68] kS={kS:g} kB={kB:g} gmS={gmS:.3f} gmB={gmB:.3f} "
          f"alphaS={aS:.4g} alphaB={aB:.4g}", flush=True)

    # G1: braking margin vs intercept-only on TEST
    yB_te = test["braking_events"].to_numpy(float)
    muB_te = cell_mu(test, rateB, gmB, lamB)
    expo_te = test["exposure"].to_numpy(float)
    dev_model = _poi_dev(yB_te, muB_te)
    dev_intercept = _poi_dev(yB_te, expo_te * lamB)
    g1 = {"braking_dev_model": dev_model, "braking_dev_intercept": dev_intercept,
          "beats_intercept": bool(dev_model < dev_intercept)}

    # copula on a train subsample
    rng = np.random.default_rng(0)
    n_cop = min(len(train), args.copula_n)
    idx = rng.choice(len(train), size=n_cop, replace=False)
    uS = nb_pit_rand(yS_tr[idx], muS_tr[idx], aS)
    uB = nb_pit_rand(yB_tr[idx], muB_tr[idx], aB)
    print(f"[68] fitting copulas on {n_cop:,} train cells...", flush=True)
    cop = fit_copula(uS, uB)
    print(f"[68] best copula={cop['best_family']} param={cop['best_param']:.4g} "
          f"rho={cop['gaussian_rho']:.4f} lr={cop['lr_stat']:.1f}", flush=True)
    g2 = {"best_family": cop["best_family"], "param": cop["best_param"],
          "lr_stat": cop["lr_stat"],
          "lr_pvalue": float(1 - stats.chi2.cdf(cop["lr_stat"], 1)),
          "gaussian_rho": cop["gaussian_rho"],
          "positive_dependence": bool(cop["gaussian_rho"] > 0)}

    # threshold sweep on a TRAIN subsample -> choose q*, then report TEST at q*
    print("[68] threshold sweep (train subsample)...", flush=True)
    qs_tr_S = np.quantile(yS_tr, Q_GRID)
    qs_tr_B = np.quantile(yB_tr, Q_GRID)
    swyS, swyB = yS_tr[idx], yB_tr[idx]
    swmuS, swmuB = muS_tr[idx], muB_tr[idx]
    sweep = []
    for q, a, b in zip(Q_GRID, qs_tr_S, qs_tr_B):
        a, b = float(a), float(b)
        if b < 1.0:            # braking-high must require >=1 event
            continue
        sc = joint_exceedance(cop["best_family"], cop["best_param"], a, b,
                              swmuS, swmuB, aS, aB)
        ap = average_precision((swyS > a) & (swyB > b), sc)
        sweep.append({"q": float(q), "a": a, "b": b, "ap_train": ap})
    if not sweep:
        raise SystemExit("no valid threshold (braking-high empty); braking too sparse")
    best_row = max(sweep, key=lambda r: (r["ap_train"] if np.isfinite(r["ap_train"]) else -1))
    q_star, a_star, b_star = best_row["q"], best_row["a"], best_row["b"]
    print(f"[68] q*={q_star} a*={a_star} b*={b_star} "
          f"train_AP={best_row['ap_train']:.4f}", flush=True)

    yS_te = test["speeding_trips"].to_numpy(float)
    muS_te = cell_mu(test, rateS, gmS, lamS)
    sc_joint = joint_exceedance(cop["best_family"], cop["best_param"], a_star, b_star,
                                muS_te, muB_te, aS, aB)
    sc_S = 1.0 - marginal_cdf(muS_te, aS, a_star)
    sc_B = 1.0 - marginal_cdf(muB_te, aB, b_star)
    lab = (yS_te > a_star) & (yB_te > b_star)
    ap_joint = average_precision(lab, sc_joint)
    ap_S = average_precision(lab, sc_S)
    ap_B = average_precision(lab, sc_B)
    g3 = {"test_joint_events": int(lab.sum()),
          "ap_joint": ap_joint, "ap_speeding_only": ap_S, "ap_braking_only": ap_B,
          "joint_beats_each": bool(np.isfinite(ap_joint) and ap_joint > max(
              ap_S if np.isfinite(ap_S) else -1, ap_B if np.isfinite(ap_B) else -1))}
    print(f"[68] TEST joint_events={g3['test_joint_events']} "
          f"AP joint/S/B = {ap_joint:.4f}/{ap_S:.4f}/{ap_B:.4f}", flush=True)

    # --- system dynamics: dependence by phase / fixture + lead-lag ---
    print("[68] dynamics: phase/fixture dependence + lead-lag...", flush=True)
    dynamics = {}
    gm_tr = train[train["is_game"]].reset_index(drop=True)
    if len(gm_tr):
        gmuS = cell_mu(gm_tr, rateS, gmS, lamS)
        gmuB = cell_mu(gm_tr, rateB, gmB, lamB)
        guS = nb_pit_rand(gm_tr["speeding_trips"].to_numpy(float), gmuS, aS)
        guB = nb_pit_rand(gm_tr["braking_events"].to_numpy(float), gmuB, aB)
        fx_rho = dependence_by_group(guS, guB, gm_tr["fixture"].to_numpy())
        dynamics = {"phase_rho": dependence_by_group(guS, guB, gm_tr["phase"].to_numpy()),
                    "fixture_rho_summary_train": _rho_summary(fx_rho),
                    "lead_lag": lead_lag(gm_tr)}
    gm_te = test[test["is_game"]].reset_index(drop=True)
    if len(gm_te):
        tuS = nb_pit_rand(gm_te["speeding_trips"].to_numpy(float),
                          cell_mu(gm_te, rateS, gmS, lamS), aS)
        tuB = nb_pit_rand(gm_te["braking_events"].to_numpy(float),
                          cell_mu(gm_te, rateB, gmB, lamB), aB)
        fx_te = dependence_by_group(tuS, tuB, gm_te["fixture"].to_numpy())
        dynamics["fixture_rho_test"] = fx_te
        dynamics["fixture_rho_summary_test"] = _rho_summary(fx_te)
    dynamics["overall_rho"] = cop["gaussian_rho"]
    print(f"[68] phase rho="
          f"{ {k: round(v['rho'], 3) for k, v in dynamics.get('phase_rho', {}).items()} } "
          f"lead_lag={dynamics.get('lead_lag')}", flush=True)

    # per-segment aggression surface (test fixtures)
    agg = (test.assign(score=sc_joint)
           .groupby(["fixture", "road_osmid"], as_index=False)
           .agg(aggression=("score", "sum"),
                aggression_max=("score", "max"),
                speeding=("speeding_trips", "sum"),
                braking=("braking_events", "sum"),
                exposure=("exposure", "sum")))
    agg = agg[agg["fixture"] != ""].sort_values(
        ["fixture", "aggression"], ascending=[True, False])
    agg.to_parquet(RES / "segment_aggression.parquet", index=False)

    report = {
        "generated": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "smoke": bool(args.smoke),
        "cells": int(len(cells)), "train_cells": int(len(train)),
        "test_cells": int(len(test)),
        "test_fixtures": n_test_fixtures,
        "margins": {"speeding": {"global_rate": lamS, "k": kS, "game_mult": gmS,
                                 "alpha": aS},
                    "braking": {"global_rate": lamB, "k": kB, "game_mult": gmB,
                                "alpha": aB}},
        "copula": cop, "gate_G1_margin": g1, "gate_G2_copula": g2,
        "threshold_selection": {"q_star": q_star, "a_star": a_star, "b_star": b_star,
                                "train_ap": best_row["ap_train"], "sweep": sweep},
        "gate_G3_honest": g3,
        "dynamics": dynamics,
    }
    (RES / "copula_aggression_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print("[68] wrote results/copula_aggression_report.json "
          "and results/segment_aggression.parquet", flush=True)
    plot(cop, sweep, uS, uB, agg, g3, FIG / "copula_aggression.png")
    plot_dynamics(dynamics, FIG / "copula_dynamics.png")
    return report


def nb_pit_rand(y, mu, alpha):
    r = 1.0 / alpha
    p = r / (r + np.maximum(mu, 1e-12))
    F = stats.nbinom.cdf(y, r, p)
    F1 = np.where(y > 0, stats.nbinom.cdf(y - 1, r, p), 0.0)
    u = F1 + np.random.default_rng(0).random(len(y)) * (F - F1)
    return np.clip(u, EPS, 1 - EPS)


def plot(cop, sweep, uS, uB, agg, g3, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        ax[0].plot(uS, uB, ".", ms=1, alpha=0.05)
        ax[0].set_title(f"Marginal PITs\ncopula={cop['best_family']} "
                        f"(rho={cop['gaussian_rho']:.3f})")
        ax[0].set_xlabel("F_S(speeding)"), ax[0].set_ylabel("F_B(braking)")
        ax[1].plot([r["q"] for r in sweep], [r["ap_train"] for r in sweep], "o-")
        ax[1].set_title("Threshold sweep (train AP)")
        ax[1].set_xlabel("q (quantile)"), ax[1].set_ylabel("average precision")
        top = agg[agg["fixture"] == agg["fixture"].iloc[0]].nlargest(20, "aggression")
        ax[2].barh(range(len(top)), top["aggression"][::-1])
        ax[2].set_yticks(range(len(top)))
        ax[2].set_yticklabels([str(o) for o in top["road_osmid"][::-1]], fontsize=6)
        ax[2].set_title(f"Top aggression segments\n(test AP={g3['ap_joint']:.3f})")
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"[68] wrote {out_png}", flush=True)
    except Exception as exc:  # noqa: BLE001 - figure is advisory
        print(f"[68] WARN: figure skipped ({exc})")


def plot_dynamics(dynamics, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        phase_rho = dynamics.get("phase_rho", {})
        if phase_rho:
            order = [p for p in ("pre", "in", "post") if p in phase_rho]
            ax[0].bar(order, [phase_rho[p]["rho"] for p in order],
                      color=["#888", "#d62728", "#1f77b4"][:len(order)])
            ax[0].axhline(dynamics.get("overall_rho", 0.0), ls="--", c="k", lw=1)
            ax[0].set_title("Speeding-braking dependence by phase")
            ax[0].set_ylabel("Gaussian copula rho")
        ll = dynamics.get("lead_lag", {})
        if ll:
            lags = sorted(ll)
            ax[1].plot(lags, [ll[l] for l in lags], "o-")
            ax[1].axvline(0, ls=":", c="k", lw=1)
            ax[1].set_title("Lead-lag: corr(speeding(t), braking(t+L))\n"
                            "L>0 => speeding leads braking")
            ax[1].set_xlabel("lag L (15-min bins)")
            ax[1].set_ylabel("correlation")
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"[68] wrote {out_png}", flush=True)
    except Exception as exc:  # noqa: BLE001 - figure is advisory
        print(f"[68] WARN: dynamics figure skipped ({exc})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--copula-n", type=int, default=1_500_000)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
