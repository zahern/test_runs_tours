"""
Smoke tests for random-parameter and correlated-random-parameter models.

Sections
--------
1. MetaCount mixed NB2   — independent random coefficients
2. MetaCount mixed lognormal — independent random coefficients
3. MetaCount correlated random params — rdm_cor_terms spec
4. SearchLibrium MixedLogit — independent random normal coefficients
5. SearchLibrium MixedLogit — correlated random parameters (correlated_vars)
6. SA search stage-1 analogue — ordered logit fitness, verify progressive search
7. SA search stage-4 analogue — SearchLibrium MNL fitness, verify progressive search

(LCML omitted — too expensive to run in smoke context.)

Run:
    $env:JAX_PLATFORMS='cpu'; $env:PYTHONIOENCODING='utf-8'
    z:\\test_runs_tours\\code\\.venv\\Scripts\\python.exe -u `
        z:\\test_runs_tours\\code\\_test_random_params.py
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── result tracking ──────────────────────────────────────────────────────────
_results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    _results.append((status, name, detail))
    suf = f"  [{detail}]" if detail else ""
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{suf}")


# ── shared synthetic data ────────────────────────────────────────────────────
RNG = np.random.default_rng(99)
N = 250

df_base = pd.DataFrame({
    'obs_id':     np.arange(N),
    'age':        RNG.uniform(20, 65, N),
    'incp':       RNG.standard_normal(N),
    'vehrd':      RNG.integers(0, 3, N).astype(float),
    'nprd':       RNG.integers(1, 5, N).astype(float),
    'n_work_out': RNG.poisson(1.5, N),
    'duration':   np.maximum(RNG.lognormal(4.5, 0.7, N), 5.0),
})

VARS = ['age', 'incp', 'vehrd', 'nprd']

# ── Section 1: MetaCount mixed NB2 (independent random params) ──────────────
print("\n" + "=" * 62)
print("Section 1: MetaCount mixed NB2 — independent random params")
print("=" * 62)

try:
    from ga_external_estimators import _METACOUNT_OK, _metacount_fit_mixed
    if not _METACOUNT_OK:
        check("MetaCount available", False, "not installed — skip sections 1-3")
        _mc_skip = True
    else:
        _mc_skip = False
        score = _metacount_fit_mixed(
            df_base[['obs_id', 'n_work_out'] + VARS],
            id_col='obs_id', y_col='n_work_out',
            selected_vars=VARS, random_vars=['age', 'incp'],
            family='nb2', R=50,
        )
        check("MC mixed NB2 runs", True)
        check("MC mixed NB2 finite score", np.isfinite(score) and score > -1e8,
              f"score={score:.4f}")
except Exception as e:
    _mc_skip = True
    check("MC mixed NB2 runs", False, str(e))
    check("MC mixed NB2 finite score", False, "skipped")

# ── Section 2: MetaCount mixed lognormal ─────────────────────────────────────
print("\n" + "=" * 62)
print("Section 2: MetaCount mixed lognormal — independent random params")
print("=" * 62)

if _mc_skip:
    check("MC mixed lognormal runs", False, "MetaCount unavailable — skip")
    check("MC mixed lognormal finite score", False, "skipped")
else:
    try:
        score_ln = _metacount_fit_mixed(
            df_base[['obs_id', 'duration'] + VARS],
            id_col='obs_id', y_col='duration',
            selected_vars=VARS, random_vars=['age', 'nprd'],
            family='duration', R=50,
        )
        check("MC mixed lognormal runs", True)
        check("MC mixed lognormal finite score",
              np.isfinite(score_ln) and score_ln > -1e8,
              f"score={score_ln:.4f}")
    except Exception as e:
        check("MC mixed lognormal runs", False, str(e))
        check("MC mixed lognormal finite score", False, "skipped")

# ── Section 3: MetaCount correlated random params (rdm_cor_terms) ───────────
print("\n" + "=" * 62)
print("Section 3: MetaCount correlated random params (rdm_cor_terms)")
print("=" * 62)

if _mc_skip:
    check("MC correlated random NB2 runs", False, "MetaCount unavailable — skip")
    check("MC correlated random NB2 finite score", False, "skipped")
else:
    try:
        from ga_external_estimators import (
            _create_metacount_builder, _fit_metacount_manual_model,
            _result_vector, _metacount_start_key, _METACOUNT_START_VALUES,
        )
        _cor_df  = df_base[['obs_id', 'n_work_out', 'age', 'incp']].copy()
        _cor_vars = ['age', 'incp']
        _cor_spec = {
            'fixed_terms':      [],
            'rdm_terms':        [],
            'rdm_cor_terms':    [f'{v}:normal' for v in _cor_vars],
            'grouped_terms':    [],
            'hetro_in_means':   [],
            'zi_terms':         [],
            'membership_terms': [],
            'dispersion':       1,
            'latent_classes':   1,
        }
        _builder = _create_metacount_builder(_cor_df, id_col='obs_id',
                                              y_col='n_work_out', family='nb2')
        _fitted  = _fit_metacount_manual_model(_builder, _cor_spec, R=50, family='nb2')
        check("MC correlated random NB2 runs", _fitted is not None)
        if _fitted is not None:
            if isinstance(_fitted, dict):
                _bic = (_fitted.get('summary') or {}).get('bic')
            else:
                _bic = getattr(_fitted, 'bic', None) or getattr(_fitted, 'BIC', None)
            check("MC correlated random NB2 has BIC",
                  _bic is not None and np.isfinite(float(_bic)),
                  f"bic={_bic}")
        else:
            check("MC correlated random NB2 has BIC", False, "fitted=None")
    except Exception as e:
        check("MC correlated random NB2 runs", False, str(e))
        check("MC correlated random NB2 has BIC", False, "skipped")

# ── shared long-format MNL data ──────────────────────────────────────────────
N_OBS, N_ALT = 200, 3


def _make_long(n_obs=N_OBS, n_alt=N_ALT, seed=42):
    rng = np.random.default_rng(seed)
    betas = np.array([1.2, -0.8])
    rows = []
    for i in range(1, n_obs + 1):
        X_i = rng.standard_normal((n_alt, 2))
        V_i = X_i @ betas + rng.gumbel(size=n_alt)
        chosen = int(np.argmax(V_i)) + 1
        for j in range(1, n_alt + 1):
            rows.append([i, j, X_i[j-1, 0], X_i[j-1, 1], int(j == chosen)])
    return pd.DataFrame(rows, columns=["id", "alt", "x1", "x2", "choice"])


LONG = _make_long()
X_L    = LONG[["x1", "x2"]].values.astype(float)
y_L    = LONG["choice"].values.astype(int)
ids_L  = LONG["id"].values.astype(int)
alts_L = LONG["alt"].values.astype(int)
VNAMES = ["x1", "x2"]

_sl_skip = False
try:
    from SearchLibrium import MixedLogit as _MXL
except ImportError:
    _sl_skip = True

# ── Section 4: SearchLibrium MixedLogit — independent random params ──────────
print("\n" + "=" * 62)
print("Section 4: SearchLibrium MixedLogit — independent random params")
print("=" * 62)

if _sl_skip:
    check("SL MixedLogit available", False, "SearchLibrium not installed — skip 4-5")
    check("SL MixedLogit fits", False, "skipped")
else:
    try:
        mxl = _MXL()
        mxl.setup(
            X=X_L, y=y_L, varnames=VNAMES,
            alts=alts_L, ids=ids_L, isvars=[],
            randvars={"x1": "n", "x2": "n"},
            fit_intercept=False, n_draws=100, maxiter=80,
        )
        mxl.fit()
        check("SL MixedLogit fits", True)
        check("SL MixedLogit loglik finite", np.isfinite(mxl.loglik),
              f"ll={mxl.loglik:.2f}")
        check("SL MixedLogit >= 4 params (2 means + 2 stds)",
              len(mxl.coeff_est) >= 4, f"n={len(mxl.coeff_est)}")
    except Exception as e:
        check("SL MixedLogit fits", False, str(e))
        check("SL MixedLogit loglik finite", False, "skipped")
        check("SL MixedLogit >= 4 params", False, "skipped")

# ── Section 5: SearchLibrium MixedLogit — correlated random params ───────────
print("\n" + "=" * 62)
print("Section 5: SearchLibrium MixedLogit — correlated random params")
print("=" * 62)

if _sl_skip:
    check("SL correlated MixedLogit fits", False, "SearchLibrium not installed — skip")
    check("SL correlated MixedLogit loglik finite", False, "skipped")
else:
    try:
        mxl_cor = _MXL()
        mxl_cor.setup(
            X=X_L, y=y_L, varnames=VNAMES,
            alts=alts_L, ids=ids_L, isvars=[],
            randvars={"x1": "n", "x2": "n"},
            correlated_vars=["x1", "x2"],   # full Cholesky covariance
            fit_intercept=False, n_draws=100, maxiter=80,
        )
        mxl_cor.fit()
        check("SL correlated MixedLogit fits", True)
        check("SL correlated MixedLogit loglik finite", np.isfinite(mxl_cor.loglik),
              f"ll={mxl_cor.loglik:.2f}")
        # correlated: 2 means + 3 Cholesky elements (lower-triangle 2×2)
        check("SL correlated MixedLogit >= 5 params",
              len(mxl_cor.coeff_est) >= 5, f"n={len(mxl_cor.coeff_est)}")
    except Exception as e:
        check("SL correlated MixedLogit fits", False, str(e))
        check("SL correlated MixedLogit loglik finite", False, "skipped")
        check("SL correlated MixedLogit >= 5 params", False, "skipped")

# ── Section 6: SA progressive search — stage-1 analogue ─────────────────────
print("\n" + "=" * 62)
print("Section 6: SA progressive search — stage-1 ordered-logit fitness")
print("=" * 62)

try:
    from ga_hypothesis_testing import SimulatedAnnealing, _stage1_fitness_aggregate

    sa_data = pd.DataFrame({
        'ptype':      np.random.choice(['FT worker', 'PT worker'], N),
        'n_work_out': np.random.poisson(1.5, N),
        'age':   RNG.uniform(20, 65, N),
        'incp':  RNG.standard_normal(N),
        'vehrd': RNG.integers(0, 3, N).astype(float),
        'nprd':  RNG.integers(1, 5, N).astype(float),
    })
    feat_names = ['age', 'incp', 'vehrd', 'nprd']
    ptypes     = sa_data['ptype'].unique().tolist()

    def _s1_fit(genes):
        return _stage1_fitness_aggregate(genes, sa_data, feat_names, ['work'], ptypes)

    sa1 = SimulatedAnnealing(
        n_features=len(feat_names), fitness_fn=_s1_fit,
        pop_size=10, n_generations=8, min_features=1,
        seed=7, verbose=True, fitness_metric='mean(-BIC/n)',
    )
    best1, hist1, _ = sa1.run_with_restarts(n_restarts=1)
    check("SA stage-1 runs without error", True)
    check("SA stage-1 history length correct", len(hist1) >= 8, f"len={len(hist1)}")
    check("SA stage-1 best fitness finite", np.isfinite(best1.fitness),
          f"fit={best1.fitness:.4f}")
    # Progressive: best_so_far should be monotonically non-decreasing
    bsf = [h['best_so_far'] for h in hist1]
    is_progressive = all(bsf[i] <= bsf[i+1] + 1e-10 for i in range(len(bsf)-1))
    check("SA stage-1 best_so_far non-decreasing (progressive)", is_progressive,
          f"bsf[0]={bsf[0]:.4f} bsf[-1]={bsf[-1]:.4f}")
    check("SA stage-1 selected >= 1 feature", int(best1.genes.sum()) >= 1,
          f"n={int(best1.genes.sum())}")
except Exception as e:
    import traceback as _tb
    check("SA stage-1 runs without error", False, str(e))
    for lbl in ["history length", "fitness", "progressive", "features"]:
        check(f"SA stage-1 {lbl}", False, "skipped")
    print(_tb.format_exc())

# ── Section 7: SA progressive search — stage-4 analogue ─────────────────────
print("\n" + "=" * 62)
print("Section 7: SA progressive search — stage-4 SearchLibrium MNL fitness")
print("=" * 62)

try:
    from ga_hypothesis_testing import SimulatedAnnealing as _SA4
    from ga_external_estimators import _SEARCHLIB_OK, stage4_fitness_searchlibrium

    if not _SEARCHLIB_OK:
        check("SA stage-4 (SL) available", False, "SearchLibrium not installed — skip")
        check("SA stage-4 progressive", False, "skipped")
    else:
        n4   = 150
        rng4 = np.random.default_rng(4)
        timing_df = pd.DataFrame({
            'purpose': np.random.choice(['work', 'edu'], n4),
            'period':  rng4.integers(0, 6, n4),
            'age':     rng4.uniform(20, 65, n4),
            'incp':    rng4.standard_normal(n4),
            'case_id': np.arange(n4),
        })
        feat4 = ['age', 'incp']

        def _s4_fit(genes):
            return stage4_fitness_searchlibrium(
                genes, timing_df, feat4, ['work'], n_alts=6)

        sa4 = _SA4(
            n_features=len(feat4), fitness_fn=_s4_fit,
            pop_size=8, n_generations=6, min_features=1,
            seed=11, verbose=True, fitness_metric='mean(-BIC/n)',
        )
        best4, hist4, _ = sa4.run_with_restarts(n_restarts=1)
        check("SA stage-4 runs without error", True)
        check("SA stage-4 history length correct", len(hist4) >= 6, f"len={len(hist4)}")
        check("SA stage-4 best fitness finite",
              np.isfinite(best4.fitness) and best4.fitness > -1e8,
              f"fit={best4.fitness:.4f}")
        bsf4 = [h['best_so_far'] for h in hist4]
        is_prog4 = all(bsf4[i] <= bsf4[i+1] + 1e-10 for i in range(len(bsf4)-1))
        check("SA stage-4 best_so_far non-decreasing (progressive)", is_prog4,
              f"bsf[0]={bsf4[0]:.4f} bsf[-1]={bsf4[-1]:.4f}")
except Exception as e:
    import traceback as _tb
    check("SA stage-4 runs without error", False, str(e))
    for lbl in ["history length", "fitness", "progressive"]:
        check(f"SA stage-4 {lbl}", False, "skipped")
    print(_tb.format_exc())

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
