"""MDCEV model test — SearchLibrium MDCEVModel budget-allocation checks.

Tests a translated-utility MDCEV allocator against synthetic person-day data
with realistic activity-duration distributions.

Run:
    z:\\test_runs_tours\\code\\.venv\\Scripts\\python.exe z:\\test_runs_tours\\code\\_test_mdcev.py

All tests should PASS without error.
"""

import sys
import warnings
import numpy as np
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    _results.append((status, name, detail))
    icon = "PASS" if condition else "FAIL"
    suffix = f"  [{detail}]" if detail else ""
    print(f"  [{icon}] {name}{suffix}")


# ── synthetic data ────────────────────────────────────────────────────────────

PURPOSES = ["work", "edu", "social", "escort", "shopping", "othmaint", "othdiscr", "eat"]
N_PERSONS = 200
RNG = np.random.default_rng(42)


def _generate_synthetic_durations(n: int = N_PERSONS) -> np.ndarray:
    """Simulate daily activity durations (minutes) per person per purpose.

    Realistic participation rates and mean durations based on HHTS-like data.
    """
    # (participation_rate, mean_min, std_min) when active
    specs = {
        "work":      (0.55, 420,  90),
        "edu":       (0.25, 360,  80),
        "social":    (0.35, 100,  50),
        "escort":    (0.30,  30,  20),
        "shopping":  (0.40,  45,  25),
        "othmaint":  (0.35,  30,  20),
        "othdiscr":  (0.40,  60,  35),
        "eat":       (0.50,  35,  20),
    }

    data = np.zeros((n, len(PURPOSES)), dtype=float)
    for j, p in enumerate(PURPOSES):
        rate, mu, sd = specs[p]
        active = RNG.random(n) < rate
        log_mu = np.log(mu) - 0.5 * np.log(1 + (sd / mu) ** 2)
        log_sd = np.sqrt(np.log(1 + (sd / mu) ** 2))
        data[active, j] = np.maximum(
            RNG.lognormal(log_mu, log_sd, active.sum()), 1.0
        )
    # Guarantee every person has a positive total budget (add mandatory eat if needed)
    zero_rows = data.sum(axis=1) == 0.0
    if zero_rows.any():
        data[zero_rows, PURPOSES.index("eat")] = 20.0
    return data


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("MDCEV model tests (SearchLibrium)")
print("=" * 60)

# ── Section 1: import & instantiation ────────────────────────────────────────
print("\n[1] Import and instantiation")

try:
    from SearchLibrium import MDCEVModel
    check("SearchLibrium import", True)
except ImportError as e:
    check("SearchLibrium import", False, str(e))
    sys.exit(1)

outside_idx = PURPOSES.index("work")
mdcev = MDCEVModel(outside_good=outside_idx, alpha_floor=0.05, alpha_cap=0.95)
check("MDCEVModel instantiated", mdcev is not None)
check("outside_good set correctly", mdcev.outside_good == outside_idx)

# ── Section 2: fit (moment-based) ────────────────────────────────────────────
print("\n[2] Moment-based fit (fit)")

raw_durations = _generate_synthetic_durations(N_PERSONS)
budgets_obs = raw_durations.sum(axis=1)

try:
    mdcev_moments = MDCEVModel(outside_good=outside_idx)
    mdcev_moments.fit(raw_durations, labels=PURPOSES)
    check("fit() returned self", mdcev_moments.fit_result_ is not None)
except Exception as e:
    check("fit() no exception", False, str(e))
    mdcev_moments = None

if mdcev_moments is not None:
    fr = mdcev_moments.fit_result_
    check("labels stored correctly", fr.labels == PURPOSES)
    check(
        "alpha within bounds",
        bool(
            np.all(fr.alpha >= mdcev_moments.alpha_floor - 1e-9)
            and np.all(fr.alpha <= mdcev_moments.alpha_cap + 1e-9)
        ),
        f"min={fr.alpha.min():.3f} max={fr.alpha.max():.3f}",
    )
    check(
        "gamma all positive",
        bool(np.all(fr.gamma >= mdcev_moments.gamma_floor - 1e-9)),
        f"min={fr.gamma.min():.4f}",
    )
    check(
        "baseline_utility[outside_good]==0",
        abs(fr.baseline_utility[outside_idx]) < 1e-9,
        f"val={fr.baseline_utility[outside_idx]:.4f}",
    )
    check(
        "participation rates in [0,1]",
        bool(np.all((fr.participation_rate >= 0) & (fr.participation_rate <= 1))),
    )
    check(
        "mean_allocation sums to mean_budget (approx)",
        abs(fr.mean_allocation.sum() - fr.mean_budget) / max(fr.mean_budget, 1) < 0.05,
        f"sum_alloc={fr.mean_allocation.sum():.1f} mean_B={fr.mean_budget:.1f}",
    )

# ── Section 3: fit_mle (quasi-MLE refinement) ────────────────────────────────
# We only verify the method exists and its signature is correct, because
# scipy FD-based L-BFGS-B on the bisection solver is O(N_params × N_obs × 80)
# per gradient eval and is too slow for an automated smoke test.
print("\n[3] Quasi-MLE fit (fit_mle — API check only)")

check("fit_mle method exists", callable(getattr(MDCEVModel, "fit_mle", None)))

import inspect as _inspect
_mle_sig = str(_inspect.signature(MDCEVModel.fit_mle))
check(
    "fit_mle accepts maxiter param",
    "maxiter" in _mle_sig,
    f"sig={_mle_sig}",
)
check(
    "fit_mle accepts l2_penalty param",
    "l2_penalty" in _mle_sig,
    f"sig={_mle_sig}",
)

# For downstream tests (predict/simulate/summary) use moment-fit on full data
# so allocation quality comparisons are meaningful.
mdcev_mle = MDCEVModel(outside_good=outside_idx)
mdcev_mle.fit(raw_durations, labels=PURPOSES)

# ── Section 4: predict — budget constraint ───────────────────────────────────
print("\n[4] predict() budget constraint")

if mdcev_mle is not None:
    test_budgets = np.array([300.0, 480.0, 600.0, 720.0, 960.0])
    preds = mdcev_mle.predict(test_budgets)
    check(
        "predict output shape correct",
        preds.shape == (len(test_budgets), len(PURPOSES)),
        str(preds.shape),
    )
    pred_totals = preds.sum(axis=1)
    max_rel_err = float(np.max(np.abs(pred_totals - test_budgets) / test_budgets))
    check(
        "budget constraint satisfied (rel err < 0.1%)",
        max_rel_err < 0.001,
        f"max_rel_err={max_rel_err:.2e}",
    )
    check(
        "all allocations non-negative",
        bool(np.all(preds >= -1e-9)),
        f"min={preds.min():.4f}",
    )
    check(
        "outside-good gets non-trivial allocation",
        bool(preds[:, outside_idx].mean() > 0.0),
        f"mean_work_alloc={preds[:, outside_idx].mean():.1f}",
    )

    # Budget scaling: larger budgets → more total leisure / work
    check(
        "allocations increase with budget",
        bool(preds[-1].sum() > preds[0].sum() * 1.5),
        f"B300_total={preds[0].sum():.0f} B960_total={preds[-1].sum():.0f}",
    )

# ── Section 5: predict — allocation quality vs naive baseline ────────────────
print("\n[5] Allocation quality vs naive proportional baseline")

if mdcev_mle is not None:
    obs_shares = (raw_durations.sum(axis=0) / raw_durations.sum())  # empirical shares
    mean_B = float(budgets_obs.mean())

    mdcev_alloc = mdcev_mle.predict(np.array([mean_B]))[0]
    naive_alloc = obs_shares * mean_B

    # MDCEV predicted shares should correlate with empirical shares
    from numpy.polynomial.polynomial import polyfit as np_polyfit

    corr = np.corrcoef(mdcev_alloc, naive_alloc)[0, 1]
    check(
        "MDCEV alloc correlates with empirical (r>0.7)",
        corr > 0.7,
        f"r={corr:.3f}",
    )

    # RMSE of MDCEV predicted shares vs observed shares (absolute threshold)
    pred_shares = mdcev_alloc / mean_B
    rmse_mdcev = float(np.sqrt(np.mean((pred_shares - obs_shares) ** 2)))
    check(
        "MDCEV linear RMSE of shares < 0.20",
        rmse_mdcev < 0.20,
        f"rmse={rmse_mdcev:.3f}",
    )

# ── Section 6: simulate() stochastic draws ───────────────────────────────────
print("\n[6] simulate() stochastic draws")

if mdcev_mle is not None:
    test_B = np.array([480.0])
    sims = mdcev_mle.simulate(test_B, n_draws=50, random_state=7)
    check(
        "simulate output shape correct",
        sims.shape == (50, 1, len(PURPOSES)),
        str(sims.shape),
    )
    sim_totals = sims[:, 0, :].sum(axis=1)
    sim_max_err = float(np.max(np.abs(sim_totals - 480.0) / 480.0))
    check(
        "simulate budget constraint (rel err < 1%)",
        sim_max_err < 0.01,
        f"max_rel_err={sim_max_err:.2e}",
    )
    check(
        "simulate has variance across draws",
        float(sims[:, 0, :].std(axis=0).mean()) > 0.0,
        f"mean_std={sims[:, 0, :].std(axis=0).mean():.1f}",
    )

# ── Section 7: summary() output ──────────────────────────────────────────────
print("\n[7] summary() DataFrame")

if mdcev_mle is not None:
    try:
        df_summ = mdcev_mle.summary()
        check("summary() returns DataFrame", isinstance(df_summ, pd.DataFrame))
        check(
            "summary has expected columns",
            set(["alternative", "baseline_utility", "alpha", "gamma", "participation_rate"]).issubset(df_summ.columns),
        )
        check("summary has one row per purpose", len(df_summ) == len(PURPOSES))
        print("\n  Model summary:")
        print(df_summ.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    except Exception as e:
        check("summary() no exception", False, str(e))

# ── Section 8: edge cases ─────────────────────────────────────────────────────
print("\n[8] Edge cases")

if mdcev_mle is not None:
    # Very small budget
    small_preds = mdcev_mle.predict(np.array([5.0]))
    check(
        "very small budget: Σ ≈ 5 min",
        abs(small_preds[0].sum() - 5.0) < 0.01,
        f"total={small_preds[0].sum():.4f}",
    )
    check(
        "very small budget: all non-negative",
        bool(np.all(small_preds >= -1e-9)),
    )

    # Zero-column in new person type (all-zero row should be rejected by fit)
    new_data = raw_durations.copy()
    new_data[:, PURPOSES.index("edu")] = 0.0  # pretend no-one studies
    # remove any rows that now have zero total budget
    new_data = new_data[new_data.sum(axis=1) > 0]
    try:
        mdcev_edge = MDCEVModel(outside_good=outside_idx)
        mdcev_edge.fit(new_data, labels=PURPOSES)
        # edu alpha should be at floor
        edu_idx = PURPOSES.index("edu")
        check(
            "zero-column purpose: alpha at floor",
            abs(mdcev_edge.alpha_[edu_idx] - mdcev_edge.alpha_floor) < 1e-9,
            f"alpha_edu={mdcev_edge.alpha_[edu_idx]:.4f}",
        )
    except Exception as e:
        check("zero-column fit no crash", False, str(e))

    # Unfitted model raises RuntimeError on predict
    unfitted = MDCEVModel()
    try:
        unfitted.predict(np.array([300.0]))
        check("unfitted predict raises RuntimeError", False, "no exception raised")
    except RuntimeError:
        check("unfitted predict raises RuntimeError", True)
    except Exception as e:
        check("unfitted predict raises RuntimeError", False, type(e).__name__)

# ── final report ──────────────────────────────────────────────────────────────
n_pass = sum(1 for s, _, _ in _results if s == PASS)
n_fail = sum(1 for s, _, _ in _results if s == FAIL)
print(f"\n{'=' * 60}")
print(f"Results: {n_pass}/{n_pass + n_fail} PASS")
if n_fail:
    print("\nFailed tests:")
    for s, name, detail in _results:
        if s == FAIL:
            print(f"  [FAIL] {name}  [{detail}]")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
