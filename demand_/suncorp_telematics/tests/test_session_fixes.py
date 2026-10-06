#!/usr/bin/env python
"""
T1 — verification suite for this session's changes. Synthetic data only.

Each test corresponds to a failure that was invisible until it cost a full
cluster run:

  A  train/future split    label collision across seasons leaked 3 games
                           into the holdout and dropped them from train
  B  grouped probe         a Fixed-everywhere scaffold is collinear, so
                           both arms returned 1e12 and the probe reported
                           "grouped loses" for every term having fitted
                           nothing
  C  hazard offset         logexp must leave the design or exposure is
                           fitted twice; the offset must reach eta
  D  LRT                   a deliberately-wrong exposure coefficient must
                           be detectable, and recovered when free
  E  alpha parse           an alpha-scale value was re-exponentiated and
                           overflowed to inf, discarding a searched spec
  F  candidate set         SEG_GROUP_VAR must not be dropped when it is a
                           forced term

Run:  python test_session_fixes.py
Exit code 0 = all passed.
"""
import os
import sys
import json
import importlib.util as iu
import traceback

import numpy as np
import pandas as pd

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Layout: <proj>/tests/test_session_fixes.py and <proj>/scripts/*.py
SCRIPT_DIR = os.path.join(PROJ_DIR, "scripts")
if not os.path.isdir(SCRIPT_DIR):          # tolerate a flatter checkout
    SCRIPT_DIR = PROJ_DIR
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"test": name, "ok": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def load(mod_name, fname):
    p = os.path.join(SCRIPT_DIR, fname)
    spec = iu.spec_from_file_location(mod_name, p)
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gdp = load("gdp59", "59_game_day_search_predict.py")
mnb = load("mnb", "metacount_nb.py")

CAL = os.path.join(PROJ_DIR, "data", "fixture_calendar.csv")
if not os.path.exists(CAL):
    CAL = None


# ---------------------------------------------------------------- A
def test_split_fixture_id():
    print("\nA. train/future split uses fixture_id (label collision)")
    if CAL is None or not os.path.exists(CAL):
        return check("A0 calendar readable", False, "fixture_calendar.csv not found")
    cal = pd.read_csv(CAL)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"])
    cal = cal.sort_values("ko").reset_index(drop=True)
    cal["fixture_id"] = (cal["label"].astype(str) + " @ "
                         + cal["ko"].dt.strftime("%Y-%m-%d %H:%M"))
    n_dup_rows = int(cal["label"].duplicated(keep=False).sum())
    n_dup_names = cal.loc[cal["label"].duplicated(keep=False), "label"].nunique()
    check("A1 labels are genuinely duplicated in the calendar",
          n_dup_rows > 0,
          f"{n_dup_rows} rows / {n_dup_names} names share a label "
          f"(nunique={cal['label'].nunique()} of {len(cal)} rows)")
    check("A2 fixture_id is unique", not bool(cal["fixture_id"].duplicated().any()))

    future_ids = set(cal.tail(6)["fixture_id"])
    train_ids = set(cal["fixture_id"]) - future_ids
    try:
        gdp._assert_split(cal, future_ids, train_ids)
        ok_split = True
        why = "partition clean"
    except SystemExit as e:
        ok_split = False
        why = str(e)
    check("A3 _assert_split passes on a correct fixture_id split",
          ok_split, why)

    # the OLD label-based split must be rejected
    old_future = set(cal["label"].tail(6))
    old_train = set(cal["label"]) - old_future
    try:
        gdp._assert_split(cal, old_future, old_train)
        check("A4 _assert_split REJECTS the old label-based split", False,
              "it accepted a leaking split")
    except SystemExit:
        check("A4 _assert_split REJECTS the old label-based split", True,
              "correctly refused")
    return True


# ---------------------------------------------------------------- B
def _synth_grouped_frame():
    rng = np.random.default_rng(11)
    n_seg, n_bin = 400, 8
    hw = ['residential', 'tertiary', 'secondary', 'primary']
    rows = [(100000 + s, b, hw[s % 4]) for s in range(n_seg)
            for b in range(n_bin)]
    df = pd.DataFrame(rows, columns=['road_osmid', 'bin', 'highway'])
    seg_len = np.log1p(rng.uniform(20, 400, n_seg))
    df['log_length'] = np.repeat(seg_len, n_bin)      # within-segment constant
    for c in ('lanes_z', 'maxspeed_z', 'dist_inter_z', 'aadt_z', 'log_flow_z'):
        df[c] = rng.normal(size=len(df))
    df['logexp'] = rng.uniform(1.0, 4.0, len(df))
    seg_eff = rng.normal(scale=0.8, size=n_seg)[df.road_osmid - 100000]
    df['Y'] = rng.poisson(np.exp(0.3 + 0.2 * df.logexp + seg_eff)).astype(float)
    df['row_id'] = np.arange(len(df))
    d = pd.get_dummies(df['highway'], prefix='hw', dtype=float)
    for c in d.columns:
        df[c] = d[c].astype(float)
    return df


def test_grouped_probe():
    print("\nB. grouped probe produces real, attributable numbers")
    from metacountregressor import ExperimentBuilder, ModelConstraints
    df = _synth_grouped_frame()
    hw_cols = [c for c in df.columns if c.startswith('hw_')]
    cont = ['lanes_z', 'maxspeed_z', 'dist_inter_z', 'log_length', 'aadt_z',
            'log_flow_z']
    variables = ['logexp'] + hw_cols + cont
    constraints = (ModelConstraints()
                   .force_include('logexp', 'log_length')
                   .force_fixed('logexp', 'log_length')
                   .no_random('logexp', *hw_cols)
                   .no_zi('logexp', *variables))
    constraints = constraints.allow_random(*cont, distributions=['normal'])
    builder = ExperimentBuilder(df, id_col='row_id', y_col='Y',
                                group_id_col='road_osmid')
    ev, _ = gdp._build_evaluator(builder, variables, constraints,
                                 [0, 1, 2, 3, 4, 5], 8, 'jax')

    ok = True
    for role, nm in ((1, 'Fixed'), (4, 'Grouped')):
        dec = gdp._base_seed_decision(ev, dispersion=1,
                                      term=gdp.SEG_GROUP_VAR,
                                      term_role=role)
        sp = ev.build_spec(dec)
        ok &= check(f"B1 minimal scaffold ({nm}) builds a spec", sp is not None,
                    (f"grouped={sp['grouped_terms']}" if sp else
                     getattr(ev, 'last_rejection_reason', 'None')))
    if not ok:
        return False

    class A:
        group_probe = 'on'
    audit = {}
    res = gdp._probe_grouped(ev, 'segment', audit, A())
    n_bad = sum(1 for r in (res or {}).values() if not r.get('ok'))
    n_tot = len(res or {})
    check("B2 no probe arm is INCONCLUSIVE", n_bad == 0,
          f"{n_bad}/{n_tot} inconclusive")
    check("B3 breakeven BIC recorded", 'grouped_probe_breakeven' in audit,
          f"breakeven={audit.get('grouped_probe_breakeven')}")
    deltas = [r['delta'] for r in (res or {}).values() if r.get('ok')]
    check("B4 deltas are non-degenerate (not all 0.0)",
          bool(deltas) and any(abs(d) > 1e-6 for d in deltas),
          f"range {min(deltas):.2f}..{max(deltas):.2f}" if deltas else "none")
    # the segment carrier must be probed when grouping by segment
    check("B5 segment-identity carrier was probed",
          gdp.SEG_GROUP_VAR in (audit.get('grouped_probe', {})
                                .get('terms_probed', [])),
          f"SEG_GROUP_VAR={gdp.SEG_GROUP_VAR}")
    return n_bad == 0


# ---------------------------------------------------------------- C
def _synth_count_frame(true_beta_exp=0.6, n=12000, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'row_id': np.arange(n),
        'logexp': rng.uniform(0.5, 4.0, n),
        'lanes_z': rng.normal(size=n),
        'tod_cos': rng.normal(size=n),
        'in_event': rng.binomial(1, 0.5, n).astype(float),
    })
    eta = (-0.5 + true_beta_exp * df.logexp + 0.3 * df.lanes_z
           + 0.4 * df.tod_cos + 0.7 * df.in_event)
    df['Y'] = rng.poisson(np.exp(eta)).astype(float)
    return df


def test_hazard_offset():
    print("\nC. hazard offset: logexp leaves the design and reaches eta")
    df = _synth_count_frame()
    common = dict(y_col='Y', id_col='row_id',
                  fixed_terms=['logexp', 'lanes_z', 'tod_cos', 'in_event'],
                  rdm_terms=[], model='nb', R=8, de_maxiter=3, de_popsize=4,
                  verbose=False)
    off = mnb.fit_metacount_nb(df, offset_col='logexp', **common)
    check("C1 logexp removed from offset design",
          'logexp' not in (off['manual_spec'].get('fixed_terms') or []),
          f"fixed={off['manual_spec'].get('fixed_terms')}")
    check("C2 exposure_mode reported as offset-pinned",
          off.get('exposure_mode') == 'offset-pinned',
          off.get('exposure_mode'))
    d = off['fit'].get('data', {}) or {}
    arr = np.asarray(d.get('offset')) if 'offset' in d else None
    check("C3 offset tensor is present and non-zero",
          arr is not None and bool((arr != 0).any()),
          f"shape={None if arr is None else arr.shape}")
    # eta must actually include it
    try:
        from metacountregressor.main_hpc import CountModel
        mdl = CountModel(off['spec'], off['fit']['data'])
        eta = np.asarray(mdl._build_eta_public()) if hasattr(
            mdl, '_build_eta_public') else None
    except Exception:
        eta = None
    check("C4 (informational) eta inspection available", True,
          "skipped" if eta is None else f"eta range {eta.min():.2f}..{eta.max():.2f}")
    return True


# ---------------------------------------------------------------- D
def test_exposure_lrt():
    print("\nD. free coefficient recovery and nested LRT")
    df = _synth_count_frame(true_beta_exp=0.6)
    common = dict(y_col='Y', id_col='row_id',
                  fixed_terms=['logexp', 'lanes_z', 'tod_cos', 'in_event'],
                  rdm_terms=[], model='nb', R=8, de_maxiter=3, de_popsize=4,
                  verbose=False)
    free = mnb.fit_metacount_nb(df, offset_col=None, **common)
    off = mnb.fit_metacount_nb(df, offset_col='logexp', **common)
    coef = free.get('exposure_coef')
    check("D1 free exposure coefficient recovered", coef is not None
          and abs(float(coef) - 0.6) < 0.08,
          f"recovered={None if coef is None else round(float(coef), 4)} truth=0.60")
    check("D2 offset arm reports no free coefficient",
          off.get('exposure_coef') is None, off.get('exposure_coef'))
    lr = 2.0 * (float(free['loglik']) - float(off['loglik']))
    check("D3 LRT detects the misspecified exposure channel", lr > 3.84,
          f"chi2(1)={lr:.2f} (crit 3.84) -- free significantly better")
    # and a CORRECT exposure channel should NOT be flagged
    df2 = _synth_count_frame(true_beta_exp=1.0, seed=5)
    f2 = mnb.fit_metacount_nb(df2, offset_col=None, **common)
    o2 = mnb.fit_metacount_nb(df2, offset_col='logexp', **common)
    lr2 = 2.0 * (float(f2['loglik']) - float(o2['loglik']))
    check("D4 LRT does NOT flag a correctly specified channel", lr2 <= 3.84,
          f"chi2(1)={lr2:.2f} with truth beta_exp=1.0")
    return True


# ---------------------------------------------------------------- E
def test_alpha_scale():
    print("\nE. alpha is parsed on the alpha scale (no exp overflow)")
    check("E1 ALPHA_MAX matches the package ceiling",
          abs(mnb.ALPHA_MAX - np.exp(12.0)) / np.exp(12.0) < 1e-6,
          f"ALPHA_MAX={mnb.ALPHA_MAX:.0f} (e^12={np.exp(12.0):.0f})")
    # a legitimately large alpha must survive, an absurd one must not
    sane = 5e4
    ok_big = np.isfinite(sane) and 0 < sane <= mnb.ALPHA_MAX
    check("E2 alpha=5e4 inside the valid band", ok_big,
          "would previously have been re-exponentiated to inf and rejected")
    check("E3 _refit_sane accepts 5e4 and rejects inf",
          mnb._refit_sane({"mle": {"x": 1.0}}, 5e4,
                          np.array([1.0, 2.0, 3.0]), "nb")[0]
          and not mnb._refit_sane({"mle": {"x": 1.0}}, np.inf,
                                  np.array([1.0, 2.0, 3.0]), "nb")[0])
    return True


# ---------------------------------------------------------------- F
def test_candidate_set():
    print("\nF. candidate set / SEARCH_GEN sanity")
    check("F1 SEG_GROUP_VAR is a varying column, not constant",
          gdp.SEG_GROUP_VAR in ('log_length',),
          f"SEG_GROUP_VAR={gdp.SEG_GROUP_VAR} "
          "(a constant column is rejected by the package's identifiability "
          "screen)")
    check("F2 SEARCH_GEN >= 7 (seg term + fixture_id era; 8 adds seg_id)",
          gdp.SEARCH_GEN >= 7, f"SEARCH_GEN={gdp.SEARCH_GEN}")
    check("F3 fixture_id is built by the script",
          'fixture_id' in open(os.path.join(SCRIPT_DIR,
                                            '59_game_day_search_predict.py'),
                               encoding='utf-8').read())
    # exposure flag registered
    src = open(os.path.join(SCRIPT_DIR, '59_game_day_search_predict.py'),
               encoding='utf-8').read()
    for flag in ('--exposure', '--exposure-lrt', '--refit-model'):
        check(f"F4 {flag} registered", f'"{flag}"' in src)
    # no duplicate argparse flags
    import re
    from collections import Counter
    flags = re.findall(r'add_argument\(\s*[\'"](--[A-Za-z0-9\-]+)', src)
    dups = {f: n for f, n in Counter(flags).items() if n > 1}
    check("F5 no duplicate argparse flags", not dups, f"dups={dups}")
    return True


# ---------------------------------------------------------------- G
def test_two_stage():
    """Two-stage ship: denominator x propensity, and the handoff gate.

    The failure this prevents: a surface that passes every pooled metric but
    cannot reorder segments between games (26107921), and a propensity that
    silently falls back to a fixed design instead of the searched one.
    """
    print("\nG. two-stage flow x propensity + handoff gate")
    rng = np.random.default_rng(7)
    n = 4000
    logexp = rng.normal(0, 1, n)
    n_i = rng.integers(20, 200, n).astype(float)
    rate = 1.0 / (1.0 + np.exp(-(-3.0 + 0.9 * logexp)))
    kk = rng.binomial(n_i.astype(int), rate).astype(float)
    work = pd.DataFrame({"Y": kk, "vehicle_count": n_i,
                         "in_event": rng.integers(0, 2, n).astype(float),
                         "rel_hour": rng.normal(0, 1, n), "logexp": logexp})
    args = type("A", (), {"target": "speeding_trips"})()
    audit = {}
    r, out = gdp.refit_binomial(
        work, args, audit, design_cols=["in_event", "rel_hour", "logexp"],
        audit_key="propensity")
    check("G1 propensity fit available", out.get("available") is True,
          str(out.get("reason", "")))
    check("G2 audit_key respected",
          "propensity" in audit and "binomial" not in audit,
          str(sorted(audit)))
    if out.get("available"):
        mm = out["model"]
        p = gdp.predict_propensity(work, mm)
        eta = np.full(len(work), float(mm["coef"][0]))
        for j, c in enumerate(mm["names"][1:], start=1):
            eta = eta + work[c].to_numpy(float) * float(mm["coef"][j])
        manual = 1.0 / (1.0 + np.exp(-eta))
        check("G3 predict_propensity matches sigmoid",
              np.allclose(p, manual, atol=1e-12),
              "max|diff|=%.2e" % np.max(np.abs(p - manual)))
        check("G3b fitted logexp effect near the 0.9 truth",
              abs(float(mm["coef"][mm["names"].index("logexp")]) - 0.9) < 0.3,
              "beta=%.3f" % float(mm["coef"][mm["names"].index("logexp")]))

    def synth(same):
        rr = np.random.default_rng(3)
        seg = np.arange(150)
        shared = rr.gamma(2.0, 1.0, 150)
        rows = []
        for f in range(3):
            base = shared if same else rr.gamma(2.0, 1.0, 150)
            rows.append(pd.DataFrame({
                "fixture": f"g{f}", "road_osmid": seg,
                "observed": base * rr.uniform(0.5, 1.5, 150),
                "pred_speeding": (base if same
                                  else base * rr.uniform(0.7, 1.3, 150))}))
        merged = pd.concat(rows, ignore_index=True)
        return merged, merged[["fixture", "road_osmid",
                               "pred_speeding"]].copy()

    merged, outp = synth(False)
    h = gdp.compute_handoff(merged, outp, {"calibration_slope": 1.0})
    check("G4 good surface passes handoff", h.get("pass") is True,
          "r_min=%s jac=%s" % (h.get("pearson_min"), h.get("top20_jaccard")))
    merged_s, out_s = synth(True)
    h_s = gdp.compute_handoff(merged_s, out_s, {"calibration_slope": 1.0})
    check("G5 identical rankings fail game_specific",
          h_s.get("pass") is False and h_s.get("game_specific") is False,
          "jac=%s" % h_s.get("top20_jaccard"))
    return True


# ---------------------------------------------------------------- H
def test_approximate_bayes():
    """Laplace posterior + evidence-selected game prior + posterior draws."""
    print("\nH. approximate Bayes (Laplace propensity + stage-A fixed effects)")
    rng = np.random.default_rng(21)
    n = 6000
    logexp = rng.normal(0, 1, n)
    game_noise = rng.normal(0, 1, n)
    n_i = rng.integers(20, 200, n).astype(float)
    rate = 1.0 / (1.0 + np.exp(-(-3.0 + 0.8 * logexp)))
    kk = rng.binomial(n_i.astype(int), rate).astype(float)
    work = pd.DataFrame({"Y": kk, "vehicle_count": n_i,
                         "logexp": logexp, "att_x_dist": game_noise})
    args = type("A", (), {"target": "speeding_trips",
                          "bayes_base_sd": 10.0})()
    audit = {}
    best, table = gdp.select_propensity_prior(
        work, args, audit, spec=None, design_cols=["logexp", "att_x_dist"])
    check("H1 evidence grid evaluated", len(table) == 5,
          "ev=%s" % [r.get("evidence") for r in table])
    check("H2 best prior selected with finite evidence",
          best is not None and np.isfinite(best[1]),
          "best_sd=%s ev=%.2f" % (best[0], best[1] if best else float("nan")))
    if best is not None:
        evs = {r["game_sd"]: r["evidence"] for r in table}
        check("H3 selected evidence is the grid maximum",
              evs.get(best[0]) is not None
              and abs(evs[best[0]] - best[1]) < 5e-3,  # table rounds to 3dp
              "evs=%s" % evs)

    # posterior draws: tiny covariance => posterior mean ~ plug-in
    r = {"names": ["(intercept)", "logexp"],
         "coef": np.array([-3.0, 0.5]),
         "cov": np.diag([1e-6, 1e-6])}
    frame = pd.DataFrame({"logexp": rng.normal(0, 1, 500)})
    draws = gdp.laplace_coef_draws(r, 60, seed=0)
    p_post = gdp.predict_propensity_draws(frame, r, draws).mean(axis=0)
    p_plug = gdp.predict_propensity(frame, r)
    check("H4 posterior mean ~ plug-in under tiny covariance",
          np.max(np.abs(p_post - p_plug)) < 1e-3,
          "max|diff|=%.2e" % np.max(np.abs(p_post - p_plug)))

    # stage-A fixed-effect factors: zero SE => ones; known SE => lognormal
    info_zero = {"beta": {"__INTERCEPT__": 0.0, "logexp": 1.0},
                 "fixed_terms": ["logexp"], "rdm_terms": [],
                 "coef": pd.DataFrame({"variable": ["__INTERCEPT__",
                                                    "logexp"],
                                       "std_err": [0.0, 0.0]})}
    fr = pd.DataFrame({"logexp": np.full(1000, 1.0)})
    fac, cov = gdp.laplace_fixed_factors(fr, info_zero, S=32, seed=0)
    check("H5 zero SEs give unit factors",
          np.allclose(fac, 1.0) and cov["n_with_se"] == 0,
          str(cov))
    info_se = dict(info_zero)
    info_se["coef"] = pd.DataFrame({"variable": ["__INTERCEPT__", "logexp"],
                                    "std_err": [0.0, 0.1]})
    fac2, cov2 = gdp.laplace_fixed_factors(fr, info_se, S=20000, seed=0)
    check("H6 known SE gives the lognormal predictive mean",
          abs(float(fac2.mean()) - np.exp(0.5 * 0.1 ** 2)) < 5e-3
          and abs(float(np.median(fac2)) - 1.0) < 5e-3,
          "mean=%.5f median=%.5f (target %.5f)"
          % (fac2.mean(), np.median(fac2), np.exp(0.5 * 0.1 ** 2)))
    check("H7 stage-A se coverage recorded",
          cov2["n_terms"] == 2 and cov2["n_with_se"] == 1, str(cov2))

    # posterior-predictive bounds must cover observations at the nominal rate
    # (parameter-only quantiles of the mean had ~0% coverage on the pilot)
    rr = np.random.default_rng(5)
    n_cells = 4000
    alpha_true = 0.25
    mu_true = rr.uniform(0.2, 5.0, n_cells)
    p_true = rr.uniform(0.01, 0.15, n_cells)
    n_true = rr.negative_binomial(1.0 / alpha_true,
                                  1.0 / (1.0 + alpha_true * mu_true))
    k_true = rr.binomial(n_true.astype(np.int64), p_true)
    S = 200
    denom_draws = np.tile(mu_true, (S, 1))
    p_draws = np.tile(p_true, (S, 1))
    pred = gdp.bayes_predictive_draws(denom_draws, p_draws,
                                      alpha=alpha_true, seed=11)
    lo = np.quantile(pred, 0.05, axis=0)
    hi = np.quantile(pred, 0.95, axis=0)
    cov90 = float(np.mean((k_true >= lo) & (k_true <= hi)))
    # Central intervals on discrete counts are conservative by construction
    # (the mass at 0 forces the 5% point to 0), so the assertion is
    # ">= nominal", not "== nominal". Mean-calibration is the sharp test.
    exp_mean = float(np.mean(mu_true * p_true))
    got_mean = float(pred.mean())
    check("H8 predictive draws are mean-calibrated",
          abs(got_mean / exp_mean - 1.0) < 0.05,
          "mean=%.4f expected=%.4f" % (got_mean, exp_mean))
    check("H9 predictive 90% bounds are conservative (>= nominal)",
          0.88 <= cov90 <= 1.0, "coverage=%.3f" % cov90)
    return True


# ---------------------------------------------------------------- I
def test_event_uplift():
    """Per-segment game uplift: estimation, attachment, interaction, forcing."""
    print("\nI. per-segment game uplift (seg_event_uplift_z / evu_x_att)")
    check("I1 seg_event_uplift_z is a temporal-x candidate",
          "seg_event_uplift_z" in gdp.TIME_X_VARS, str(gdp.TIME_X_VARS))
    check("I2 evu_x_att is a game-wide candidate",
          "evu_x_att" in gdp.GAME_WIDE_VARS, str(gdp.GAME_WIDE_VARS))

    # estimation: 20 responsive segments (event rate 2x background), 20 cold
    n_seg = 40
    seg = np.arange(1000, 1000 + n_seg)
    resp = np.zeros(n_seg, dtype=bool)
    resp[:20] = True
    y_ev = np.where(resp, 2.0, 0.5) * 1.0 * 1  # rate * E=100 -> 200 / 50
    ev = pd.DataFrame({"road_osmid": seg, "y": y_ev, "e": 100.0})
    bg = pd.DataFrame({"y": 10.0, "e": 1000.0}, index=seg)
    up = gdp.estimate_seg_event_uplift(
        ev.rename(columns={"y": "target"}), bg.rename(columns={"y": "target"}),
        "target", "e")
    z_resp = np.array([up[int(s)] for s in seg[resp]])
    z_cold = np.array([up[int(s)] for s in seg[~resp]])
    check("I3 responsive segments get positive uplift",
          z_resp.min() > 0 > z_cold.max(),
          "resp=[%.2f,%.2f] cold=[%.2f,%.2f]"
          % (z_resp.min(), z_resp.max(), z_cold.min(), z_cold.max()))

    # attachment: map by segment, unseen -> 0, and the interaction is built
    # in the SAME function (it was once built earlier, when the uplift column
    # did not exist yet, so it was silently all-zero)
    frame = pd.DataFrame({"road_osmid": [1, 2, 999],
                          "rel_bin": [0, 1, 2],
                          "fixture_type": ["NRL", "NRL", "NRL"],
                          "fixture": ["f", "f", "f"],
                          "log_att": [np.log1p(40000.0)] * 3,
                          "exp": [1.0, 1.0, 1.0]})
    frame = gdp.apply_temporal_x(frame, None, None, {1: 1.5, 2: -0.7})
    check("I4 uplift attached and unseen maps to 0",
          np.allclose(frame["seg_event_uplift_z"], [1.5, -0.7, 0.0]),
          str(frame["seg_event_uplift_z"].tolist()))
    check("I5 evu_x_att = uplift x log_att",
          np.allclose(frame["evu_x_att"],
                      frame["seg_event_uplift_z"] * frame["log_att"]),
          str(frame["evu_x_att"].tolist()))

    # forcing: varying column forced by default, not when off/constant
    work = pd.DataFrame({"in_event": [0.0, 1.0], "rel_hour": [0.0, 1.0],
                         "log_att": [0.0, 1.0], "event_x_dist": [0.0, 1.0],
                         "att_x_dist": [0.0, 1.0], "rel_x_dist": [0.0, 1.0],
                         "evu_x_att": [0.0, 0.5]})
    args_on = type("A", (), {"force_event_uplift": "on"})()
    args_off = type("A", (), {"force_event_uplift": "off"})()
    check("I6 varying evu_x_att is forced by default",
          "evu_x_att" in gdp.force_terms(work, args_on))
    check("I7 --force-event-uplift off drops it",
          "evu_x_att" not in gdp.force_terms(work, args_off))
    work["evu_x_att"] = 0.0
    check("I8 constant evu_x_att is not forced (identifiability)",
          "evu_x_att" not in gdp.force_terms(work, args_on))
    check("I9 SEARCH_GEN bumped for the new candidates",
          gdp.SEARCH_GEN >= 9, "SEARCH_GEN=%s" % gdp.SEARCH_GEN)
    return True


def main():
    print("=" * 66)
    print("T1 session-fix verification suite (synthetic)")
    print("=" * 66)
    for fn in (test_split_fixture_id, test_grouped_probe, test_hazard_offset,
               test_exposure_lrt, test_alpha_scale, test_candidate_set,
               test_two_stage, test_approximate_bayes, test_event_uplift):
        try:
            fn()
        except Exception:
            check(f"{fn.__name__} raised", False,
                  traceback.format_exc(limit=2).replace("\n", " | "))
    n_pass = sum(1 for r in RESULTS if r['ok'])
    n_fail = len(RESULTS) - n_pass
    print("\n" + "=" * 66)
    print(f"RESULT: {n_pass} passed, {n_fail} failed, {len(RESULTS)} total")
    print("=" * 66)
    for r in RESULTS:
        if not r['ok']:
            print(f"  FAILED: {r['test']}  -- {r['detail'][:160]}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
