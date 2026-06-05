"""Comprehensive smoke test for SearchLibrium 0.0.86 — all base models.

Tests: MultinomialLogit, MixedLogit, NestedLogit, MultinomialProbit,
       BinaryProbit, HeckmanTwoStep, OrderedLogit, RandomRegret,
       LatentClassMixedLogit, MDCEVModel

Run:
    $env:PYTHONIOENCODING='utf-8'; $env:JAX_PLATFORMS='cpu'
    z:\\test_runs_tours\\code\\.venv\\Scripts\\python.exe -u `
        z:\\test_runs_tours\\code\\_test_searchlibrium.py

All tests should PASS.
"""

import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── helpers ──────────────────────────────────────────────────────────────────

_results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    _results.append((status, name, detail))
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{suffix}")


# ── shared synthetic data ────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

N_OBS   = 300   # number of individuals
N_ALT   = 3     # number of choice alternatives
N_VARS  = 2     # number of generic (ASC-free) attributes

# Long-format choice data: N_OBS * N_ALT rows
# Each row: [obs_id, alt_id, x1, x2, choice]
def _make_long_data(n_obs=N_OBS, n_alt=N_ALT, n_vars=N_VARS, seed=42):
    rng = np.random.default_rng(seed)
    betas_true = np.array([1.2, -0.8])          # true coefficients
    rows = []
    for i in range(1, n_obs + 1):
        X_i = rng.standard_normal((n_alt, n_vars))
        V_i = X_i @ betas_true + rng.gumbel(size=n_alt)
        chosen = int(np.argmax(V_i)) + 1
        for j in range(1, n_alt + 1):
            rows.append([i, j, X_i[j - 1, 0], X_i[j - 1, 1], int(j == chosen)])
    df = pd.DataFrame(rows, columns=["id", "alt", "x1", "x2", "choice"])
    return df


def _mnl_arrays(df):
    """Return (X, y, ids, alts) arrays for MNL/RRM/nested setup."""
    X = df[["x1", "x2"]].values.astype(float)
    y = df["choice"].values.astype(int)
    ids = df["id"].values.astype(int)
    alts = df["alt"].values.astype(int)
    return X, y, ids, alts


LONG_DF = _make_long_data()
X_mnl, y_mnl, ids_mnl, alts_mnl = _mnl_arrays(LONG_DF)
VARNAMES = ["x1", "x2"]

# ── Section 0: import ────────────────────────────────────────────────────────
print("=" * 62)
print("SearchLibrium 0.0.86 — comprehensive model smoke test")
print("=" * 62)

print("\n[0] Package import")
try:
    import SearchLibrium as SL
    check("SearchLibrium imports OK", True, f"v{SL.__version__}")
except Exception as e:
    check("SearchLibrium imports OK", False, str(e))
    sys.exit(1)

from SearchLibrium import (
    MultinomialLogit,
    MixedLogit,
    NestedLogit,
    MultinomialProbit,
    BinaryProbit,
    HeckmanTwoStep,
    OrderedLogit,
    RandomRegret,
    LatentClassMixedLogit,
    MDCEVModel,
)

# ── Section 1: MultinomialLogit ──────────────────────────────────────────────
print("\n[1] MultinomialLogit")

try:
    mnl = MultinomialLogit()
    mnl.setup(
        X=X_mnl, y=y_mnl,
        varnames=VARNAMES,
        alts=alts_mnl,
        ids=ids_mnl,
        isvars=[],
        fit_intercept=False,
        maxiter=200,
    )
    mnl.fit()
    check("MNL fit completes", True)
    check("MNL coeff_est shape", len(mnl.coeff_est) == len(VARNAMES),
          f"len={len(mnl.coeff_est)}")
    check("MNL loglik finite", np.isfinite(mnl.loglik), f"ll={mnl.loglik:.2f}")
    check("MNL coeff sign (x1>0)", float(mnl.coeff_est[0]) > 0,
          f"b1={mnl.coeff_est[0]:.3f}")
except Exception as e:
    check("MNL fit completes", False, str(e))
    check("MNL coeff_est shape", False, "skipped")
    check("MNL loglik finite", False, "skipped")
    check("MNL coeff sign (x1>0)", False, "skipped")

# ── Section 2: MixedLogit ────────────────────────────────────────────────────
print("\n[2] MixedLogit")

try:
    mxl = MixedLogit()
    mxl.setup(
        X=X_mnl, y=y_mnl,
        varnames=VARNAMES,
        alts=alts_mnl,
        ids=ids_mnl,
        isvars=[],
        randvars={"x1": "n"},       # x1 random normal
        fit_intercept=False,
        n_draws=50,
        maxiter=100,
    )
    mxl.fit()
    check("MXL fit completes", True)
    check("MXL coeff_est non-empty", len(mxl.coeff_est) > 0,
          f"len={len(mxl.coeff_est)}")
    check("MXL loglik finite", np.isfinite(mxl.loglik), f"ll={mxl.loglik:.2f}")
except Exception as e:
    check("MXL fit completes", False, str(e))
    check("MXL coeff_est non-empty", False, "skipped")
    check("MXL loglik finite", False, "skipped")

# ── Section 3: NestedLogit ───────────────────────────────────────────────────
print("\n[3] NestedLogit")

try:
    # Two nests: {1,2} and {3}
    nl = NestedLogit()
    nl.setup(
        X=X_mnl, X_nest=None, y=y_mnl,
        varnames=VARNAMES,
        isvars=[],
        alts=alts_mnl,
        ids=ids_mnl,
        nests={1: [0, 1], 2: [2]},
        fit_intercept=False,
    )
    nl.fit()
    check("NL fit completes", True)
    check("NL coeff_est non-empty", len(nl.coeff_est) > 0,
          f"len={len(nl.coeff_est)}")
    check("NL loglik finite", np.isfinite(nl.loglik), f"ll={nl.loglik:.2f}")
except Exception as e:
    check("NL fit completes", False, str(e))
    check("NL coeff_est non-empty", False, "skipped")
    check("NL loglik finite", False, "skipped")

# ── Section 4: MultinomialProbit ─────────────────────────────────────────────
print("\n[4] MultinomialProbit")

try:
    # MNP is slow — use very small dataset and few iterations
    df_small = _make_long_data(n_obs=60, n_alt=3, seed=1)
    X_s, y_s, ids_s, alts_s = _mnl_arrays(df_small)
    mnp = MultinomialProbit()
    mnp.setup(
        X=X_s, y=y_s,
        varnames=VARNAMES,
        alts=alts_s,
        ids=ids_s,
        isvars=[],
        fit_intercept=False,
        maxiter=30,
    )
    mnp.fit()
    check("MNP fit completes", True)
    check("MNP coeff_est non-empty", len(mnp.coeff_est) > 0,
          f"len={len(mnp.coeff_est)}")
    check("MNP loglik finite", np.isfinite(mnp.loglik), f"ll={mnp.loglik:.2f}")
except Exception as e:
    check("MNP fit completes", False, str(e))
    check("MNP coeff_est non-empty", False, "skipped")
    check("MNP loglik finite", False, "skipped")

# ── Section 5: BinaryProbit ──────────────────────────────────────────────────
print("\n[5] BinaryProbit")

try:
    rng5 = np.random.default_rng(5)
    n5 = 400
    X5 = rng5.standard_normal((n5, 2))
    y5 = (X5 @ np.array([1.5, -1.0]) + rng5.standard_normal(n5) > 0).astype(int)
    X5_df = pd.DataFrame(X5, columns=["z1", "z2"])

    bp = BinaryProbit()
    bp.setup(X=X5_df, y=y5, varnames=["z1", "z2"], fit_intercept=True)
    bp.fit(disp=False)
    check("BinaryProbit fit completes", True)
    check("BinaryProbit coeff_est shape", len(bp.coeff_est) == 3,
          f"len={len(bp.coeff_est)}")   # 2 vars + intercept
    check("BinaryProbit loglik finite", np.isfinite(bp.loglik), f"ll={bp.loglik:.2f}")
    check("BinaryProbit z1 coeff sign > 0", float(bp.coeff_est[0]) > 0,
          f"b_z1={bp.coeff_est[0]:.3f}")
    preds = bp.predict_proba(X5_df)
    check("BinaryProbit predict_proba shape", len(preds) == n5,
          f"len={len(preds)}")
    check("BinaryProbit probs in [0,1]",
          bool(np.all(preds >= 0) and np.all(preds <= 1)))
except Exception as e:
    check("BinaryProbit fit completes", False, str(e))
    for lbl in ["coeff shape", "loglik", "z1 sign", "predict shape", "probs range"]:
        check(f"BinaryProbit {lbl}", False, "skipped")

# ── Section 6: HeckmanTwoStep ────────────────────────────────────────────────
print("\n[6] HeckmanTwoStep")

try:
    rng6 = np.random.default_rng(6)
    n6 = 400
    Z6 = rng6.standard_normal((n6, 2))               # selection equation vars
    X6 = rng6.standard_normal((n6, 2))               # outcome vars
    sel_index = Z6 @ np.array([1.0, 0.5]) + rng6.standard_normal(n6)
    sel_y = (sel_index > 0).astype(int)              # 1 = selected
    out_y = X6 @ np.array([2.0, -1.0]) + rng6.standard_normal(n6)
    # Only selected obs enter outcome equation
    sel_mask = sel_y == 1
    X6_sel = pd.DataFrame(X6[sel_mask], columns=["w1", "w2"])
    out_y_sel = out_y[sel_mask]
    Z6_df = pd.DataFrame(Z6, columns=["z1", "z2"])

    ht = HeckmanTwoStep()
    ht.setup(
        selection_X=Z6_df, selection_y=sel_y,
        outcome_X=pd.DataFrame(X6, columns=["w1", "w2"]), outcome_y=out_y,
        selection_varnames=["z1", "z2"],
        outcome_varnames=["w1", "w2"],
        fit_intercept=True,
    )
    ht.fit(disp=False)
    check("HeckmanTwoStep fit completes", True)
    sf = ht.summary_frame()
    check("HeckmanTwoStep summary_frame is DataFrame", isinstance(sf, pd.DataFrame),
          f"shape={sf.shape}")
    check("HeckmanTwoStep outcome coeff non-empty",
          hasattr(ht, "outcome_result") and len(ht.outcome_result.params) > 0)
except Exception as e:
    check("HeckmanTwoStep fit completes", False, str(e))
    check("HeckmanTwoStep summary_frame is DataFrame", False, "skipped")
    check("HeckmanTwoStep outcome coeff non-empty", False, "skipped")

# ── Section 7: OrderedLogit ──────────────────────────────────────────────────
print("\n[7] OrderedLogit")

try:
    rng7 = np.random.default_rng(7)
    n7 = 400
    X7 = rng7.standard_normal((n7, 2))
    latent7 = X7 @ np.array([1.0, -0.5]) + rng7.standard_normal(n7)
    thresholds = [-0.8, 0.8]
    y7 = np.digitize(latent7, thresholds).astype(int)  # {0, 1, 2}

    ol = OrderedLogit(
        varnames=["v1", "v2"],
        X=pd.DataFrame(X7, columns=["v1", "v2"]),
        y=y7,
        J=3,
        fit_intercept=False,
    )
    ol.fit(method="bfgs")
    check("OrderedLogit fit completes", True)
    check("OrderedLogit params non-empty", ol.params is not None and len(ol.params) > 0,
          f"len={len(ol.params)}")
    check("OrderedLogit loglik finite",
          ol.loglik is not None and np.isfinite(ol.loglik),
          f"ll={ol.loglik:.2f}" if ol.loglik is not None else "None")
    _y_lat7 = ol.compute_latent(X7, ol.get_beta(ol.params))
    _thr7 = ol.get_thresholds(ol.params)
    preds7 = ol.predict_category(_y_lat7, _thr7)
    check("OrderedLogit predict_category shape", len(preds7) == n7,
          f"len={len(preds7)}")
    check("OrderedLogit predictions in {0,1,2}",
          bool(np.all(np.isin(preds7, [0, 1, 2]))))
except Exception as e:
    check("OrderedLogit fit completes", False, str(e))
    for lbl in ["params", "loglik", "predict shape", "predict range"]:
        check(f"OrderedLogit {lbl}", False, "skipped")

# ── Section 8: RandomRegret ──────────────────────────────────────────────────
print("\n[8] RandomRegret")

try:
    rrm = RandomRegret()
    rrm.setup(
        X=X_mnl, y=y_mnl,
        varnames=VARNAMES,
        alts=alts_mnl,
        ids=ids_mnl,
        isvars=[],
        maxiter=100,
    )
    rrm.fit()
    check("RandomRegret fit completes", True)
    check("RandomRegret coeff_est non-empty", len(rrm.coeff_est) > 0,
          f"len={len(rrm.coeff_est)}")
    check("RandomRegret loglik finite", np.isfinite(rrm.loglik),
          f"ll={rrm.loglik:.2f}")
except Exception as e:
    check("RandomRegret fit completes", False, str(e))
    check("RandomRegret coeff_est non-empty", False, "skipped")
    check("RandomRegret loglik finite", False, "skipped")

# ── Section 9: LatentClassMixedLogit ─────────────────────────────────────────
print("\n[9] LatentClassMixedLogit")

try:
    df_lc = _make_long_data(n_obs=200, seed=9)
    X_lc, y_lc, ids_lc, alts_lc = _mnl_arrays(df_lc)
    lc = LatentClassMixedLogit(n_classes=2, maxiter=50, class_maxiter=30)
    lc.setup(
        X=X_lc, y=y_lc,
        varnames=VARNAMES,
        ids=ids_lc,
        alts=alts_lc,
        fit_intercept=False,
    )
    lc.fit()
    check("LCML fit completes", True)
    check("LCML coeff_est non-empty", len(lc.coeff_est) > 0,
          f"len={len(lc.coeff_est)}")
    check("LCML loglik finite", np.isfinite(lc.loglik), f"ll={lc.loglik:.2f}")
except Exception as e:
    check("LCML fit completes", False, str(e))
    check("LCML coeff_est non-empty", False, "skipped")
    check("LCML loglik finite", False, "skipped")

# ── Section 10: MDCEVModel ───────────────────────────────────────────────────
print("\n[10] MDCEVModel")

PURPOSES = ["work", "edu", "social", "escort", "shopping", "othmaint", "othdiscr", "eat"]
_rng10 = np.random.default_rng(42)

def _gen_durations(n):
    specs = {
        "work":     (0.55, 420,  90),
        "edu":      (0.25, 360,  80),
        "social":   (0.35, 100,  50),
        "escort":   (0.30,  30,  20),
        "shopping": (0.40,  45,  25),
        "othmaint": (0.35,  30,  20),
        "othdiscr": (0.40,  60,  35),
        "eat":      (0.50,  35,  20),
    }
    data = np.zeros((n, len(PURPOSES)), dtype=float)
    for j, p in enumerate(PURPOSES):
        rate, mu, sd = specs[p]
        active = _rng10.random(n) < rate
        log_mu = np.log(mu) - 0.5 * np.log(1 + (sd / mu) ** 2)
        log_sd = np.sqrt(np.log(1 + (sd / mu) ** 2))
        data[active, j] = np.maximum(_rng10.lognormal(log_mu, log_sd, active.sum()), 1.0)
    zero_rows = data.sum(axis=1) == 0.0
    if zero_rows.any():
        data[zero_rows, PURPOSES.index("eat")] = 20.0
    return data

outside_idx = PURPOSES.index("work")
raw_dur = _gen_durations(200)

try:
    mdcev = MDCEVModel(outside_good=outside_idx)
    mdcev.fit(raw_dur, labels=PURPOSES)
    check("MDCEV fit() completes", True)

    fr = mdcev.fit_result_
    check("MDCEV labels stored", fr.labels == PURPOSES)
    check("MDCEV alpha in bounds",
          bool(np.all(fr.alpha >= mdcev.alpha_floor - 1e-9) and
               np.all(fr.alpha <= mdcev.alpha_cap + 1e-9)),
          f"min={fr.alpha.min():.3f} max={fr.alpha.max():.3f}")
    check("MDCEV gamma all positive",
          bool(np.all(fr.gamma > 0)), f"min={fr.gamma.min():.4f}")

    # predict
    test_budgets = np.array([300.0, 480.0, 720.0])
    preds = mdcev.predict(test_budgets)
    check("MDCEV predict shape", preds.shape == (3, len(PURPOSES)), str(preds.shape))
    rel_err = float(np.max(np.abs(preds.sum(axis=1) - test_budgets) / test_budgets))
    check("MDCEV budget constraint (rel err < 0.1%)", rel_err < 0.001,
          f"max_rel_err={rel_err:.2e}")
    check("MDCEV all allocations non-negative", bool(np.all(preds >= -1e-9)),
          f"min={preds.min():.4f}")

    # simulate
    sims = mdcev.simulate(np.array([480.0]), n_draws=30, random_state=7)
    check("MDCEV simulate shape", sims.shape == (30, 1, len(PURPOSES)), str(sims.shape))
    sim_err = float(np.max(np.abs(sims[:, 0, :].sum(axis=1) - 480.0) / 480.0))
    check("MDCEV simulate budget constraint (< 1%)", sim_err < 0.01,
          f"max_rel_err={sim_err:.2e}")

    # summary
    df_s = mdcev.summary()
    check("MDCEV summary() is DataFrame", isinstance(df_s, pd.DataFrame))
    check("MDCEV summary rows = n_purposes", len(df_s) == len(PURPOSES),
          f"rows={len(df_s)}")

    # fit_mle API
    check("MDCEV fit_mle method exists", callable(getattr(MDCEVModel, "fit_mle", None)))
    import inspect as _insp
    _sig = str(_insp.signature(MDCEVModel.fit_mle))
    check("MDCEV fit_mle accepts maxiter", "maxiter" in _sig)
    check("MDCEV fit_mle accepts l2_penalty", "l2_penalty" in _sig)

    # unfitted predict raises
    try:
        MDCEVModel().predict(np.array([300.0]))
        check("MDCEV unfitted predict raises RuntimeError", False, "no error raised")
    except RuntimeError:
        check("MDCEV unfitted predict raises RuntimeError", True)
    except Exception as e:
        check("MDCEV unfitted predict raises RuntimeError", False, type(e).__name__)

except Exception as e:
    check("MDCEV fit() completes", False, str(e))
    for lbl in ["labels","alpha","gamma","predict shape","budget constraint",
                "non-negative","simulate shape","sim budget","summary","fit_mle exists",
                "fit_mle maxiter","fit_mle l2","unfitted raises"]:
        check(f"MDCEV {lbl}", False, "skipped")

# ── final report ─────────────────────────────────────────────────────────────
n_pass = sum(1 for s, _, _ in _results if s == "PASS")
n_fail = sum(1 for s, _, _ in _results if s == "FAIL")
print(f"\n{'=' * 62}")
print(f"Results: {n_pass}/{n_pass + n_fail} PASS")
if n_fail:
    print("\nFailed tests:")
    for s, name, detail in _results:
        if s == "FAIL":
            suf = f"  [{detail}]" if detail else ""
            print(f"  [FAIL] {name}{suf}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
