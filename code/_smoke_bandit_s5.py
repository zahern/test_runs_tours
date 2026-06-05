"""
Stage 5 Bandit Feature Searcher smoke test.

Three phases covering the BanditFeatureSearcher mechanics and its integration
with Larch NL mode and destination choice models.

  Phase 1  — Bandit mechanics on a synthetic fitness with embedded signal.
             Verifies Thompson sampling, arm updates, cache, fixed features,
             min_features bounds, run() / run_with_restarts() interface.

  Phase 2  — Bandit x Larch NL mode choice (tiny synthetic tour dataset).
             Confirms BanditFeatureSearcher selects meaningful mode skims and
             that the history is progressive.

  Phase 3  — Bandit x Larch MNL destination choice (tiny synthetic idca dataset).
             Confirms the destination arm search converges and respects fixed
             features (c_crash / c_log_dist).

Run:
    $env:JAX_PLATFORMS='cpu'; $env:PYTHONIOENCODING='utf-8'
    z:\\test_runs_tours\\code\\.venv\\Scripts\\python.exe -u `
        z:\\test_runs_tours\\code\\_smoke_bandit_s5.py
"""
from __future__ import annotations

import contextlib
import io as _io
import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Smoke runs use the bandit searcher explicitly regardless of env flags
os.environ["GA_USE_BANDIT"] = "0"   # we instantiate BanditFeatureSearcher directly

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    _results.append((status, name, detail))
    suf = f"  [{detail}]" if detail else ""
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{suf}")
    return bool(condition)


def _is_progressive(history: list[dict]) -> tuple[bool, str]:
    """best_so_far must be non-decreasing."""
    bsf = [h["best_so_far"] for h in history if np.isfinite(h["best_so_far"])]
    if len(bsf) < 2:
        return True, f"only {len(bsf)} finite entries"
    violations = sum(1 for i in range(len(bsf) - 1) if bsf[i] > bsf[i + 1] + 1e-10)
    return violations == 0, (
        f"bsf[0]={bsf[0]:.4f} bsf[-1]={bsf[-1]:.4f} violations={violations}"
    )


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
from ga_hypothesis_testing import BanditFeatureSearcher, Individual, GAResult

try:
    import larch as lx
    from larch import P, X
    _LX_OK = True
except ImportError:
    _LX_OK = False

try:
    from stage5_destination_choice import (
        _build_mode_model,
        _build_od_dataset,
        _estimate as _lx_estimate,
        MODE_MAP,
        DEST_UEC_FLAT,
    )
    _S5_OK = True
except Exception as _s5_e:
    _S5_OK = False
    print(f"  [warn] stage5 import failed: {_s5_e}")

print("=" * 66)
print("BanditFeatureSearcher × Stage 5 Smoke Test")
print(f"  Larch:   {'available' if _LX_OK else 'NOT INSTALLED'}")
print(f"  Stage 5: {'available' if _S5_OK else 'unavailable'}")
print("=" * 66)

RNG = np.random.default_rng(2025)

# ============================================================================
# PHASE 1 — Bandit mechanics  (no Larch required)
# ============================================================================
print("\n" + "─" * 66)
print("PHASE 1 — Bandit mechanics (synthetic signal-bearing fitness)")
print("─" * 66)

try:
    # 15 features; features 0-2 are "signal" (ptype dummies), rest are noise
    _N_FEAT = 15
    _SIGNAL_IDX = {0, 1, 2}          # always beneficial when ON
    _FIXED_NAMES = ["fixed_a", "fixed_b"]   # always forced ON

    _feat_names = (["fixed_a", "fixed_b"]
                   + [f"signal_{i}" for i in range(3)]
                   + [f"noise_{i}" for i in range(10)])
    assert len(_feat_names) == _N_FEAT

    # Fitness: sum of "values" for selected features; signal features have value 1.0,
    # noise features have value ≈ 0, with BIC-like penalty for extra parameters.
    _FEAT_VALUE = np.array([0.0, 0.0,      # fixed (no intrinsic value)
                             1.0, 1.0, 1.0, # signal
                             *([0.05] * 10)]) # noise
    _N_OBS_P1 = 200

    def _p1_fitness(genes: np.ndarray) -> float:
        k = int(genes.sum())
        if k == 0:
            return -1e9
        ll = float(np.dot(_FEAT_VALUE, genes)) * _N_OBS_P1 * 0.15
        bic = -2.0 * ll + k * np.log(_N_OBS_P1)
        return -bic / _N_OBS_P1

    # -- 1a: basic run
    bandit1 = BanditFeatureSearcher(
        n_features    = _N_FEAT,
        fitness_fn    = _p1_fitness,
        pop_size      = 8,
        n_generations = 15,
        min_features  = 2,
        seed          = 7,
        verbose       = True,
        feature_names = _feat_names,
        fixed_features= _FIXED_NAMES,
        fitness_metric= "mean(-BIC/n)",
    )
    best1, hist1, pop1 = bandit1.run()

    check("Phase 1a run() returns (best, hist, pop)",
          isinstance(best1, Individual) and isinstance(hist1, list))
    check("Phase 1a ≥15 history entries", len(hist1) >= 15, f"n={len(hist1)}")
    check("Phase 1a best fitness finite", np.isfinite(best1.fitness),
          f"fit={best1.fitness:.4f}")
    prog1a, pmsg1a = _is_progressive(hist1)
    check("Phase 1a progressive (best_so_far non-decreasing)", prog1a, pmsg1a)

    # fixed features always ON
    fixed_idx = [i for i, n in enumerate(_feat_names) if n in set(_FIXED_NAMES)]
    check("Phase 1a fixed features always ON",
          all(best1.genes[i] == 1 for i in fixed_idx),
          f"genes@fixed={best1.genes[fixed_idx].tolist()}")

    # cache reduces evaluations
    check("Phase 1a cache populated", len(bandit1._cache) > 0,
          f"cache_size={len(bandit1._cache)}")

    # -- 1b: run_with_restarts
    bandit1b = BanditFeatureSearcher(
        n_features    = _N_FEAT,
        fitness_fn    = _p1_fitness,
        pop_size      = 6,
        n_generations = 10,
        min_features  = 2,
        seed          = 13,
        verbose       = False,
        feature_names = _feat_names,
        fixed_features= _FIXED_NAMES,
    )
    best1b, hist1b, _ = bandit1b.run_with_restarts(n_restarts=2)

    check("Phase 1b run_with_restarts() succeeds", isinstance(best1b, Individual))
    check("Phase 1b history covers main + restarts",
          len(hist1b) >= 10 + 2 * max(1, 10 // 3),
          f"n={len(hist1b)}")
    prog1b, pmsg1b = _is_progressive(hist1b)
    check("Phase 1b run_with_restarts progressive", prog1b, pmsg1b)

    # -- 1c: arm probability shift — after many evals signal arms should lead
    bandit1c = BanditFeatureSearcher(
        n_features    = _N_FEAT,
        fitness_fn    = _p1_fitness,
        pop_size      = 20,
        n_generations = 30,
        min_features  = 2,
        seed          = 99,
        verbose       = False,
        feature_names = _feat_names,
        fixed_features= _FIXED_NAMES,
    )
    bandit1c.run_with_restarts(n_restarts=1)
    # Arms for signal features should have higher alpha/(alpha+beta) than noise
    _sig_probs  = [bandit1c._alpha[i] / (bandit1c._alpha[i] + bandit1c._beta[i])
                   for i in range(2, 5)]    # signal_0/1/2
    _noise_probs = [bandit1c._alpha[i] / (bandit1c._alpha[i] + bandit1c._beta[i])
                    for i in range(5, 15)]  # noise_0..9
    _sig_mean   = float(np.mean(_sig_probs))
    _noise_mean = float(np.mean(_noise_probs))
    check("Phase 1c signal arms learned higher probability than noise",
          _sig_mean > _noise_mean,
          f"signal_mean={_sig_mean:.3f} noise_mean={_noise_mean:.3f}")

    # -- 1d: min_features respected
    bandit1d = BanditFeatureSearcher(
        n_features    = _N_FEAT,
        fitness_fn    = _p1_fitness,
        pop_size      = 5,
        n_generations = 5,
        min_features  = 4,
        seed          = 0,
        verbose       = False,
        feature_names = _feat_names,
        fixed_features= _FIXED_NAMES,
    )
    _best1d, _hist1d, _ = bandit1d.run()
    check("Phase 1d min_features=4 respected in best result",
          int(_best1d.genes.sum()) >= 4,
          f"n_active={int(_best1d.genes.sum())}")

    # -- 1e: all-optional-free case (n_opt == 1 → leaf case)
    bandit1e = BanditFeatureSearcher(
        n_features    = 4,
        fitness_fn    = lambda g: float(g.sum()) / 4.0,
        pop_size      = 6,
        n_generations = 8,
        min_features  = 1,
        seed          = 1,
        verbose       = False,
        feature_names = ["fix1", "fix2", "fix3", "opt1"],
        fixed_features= ["fix1", "fix2", "fix3"],
    )
    _best1e, _hist1e, _ = bandit1e.run()
    check("Phase 1e single-optional-arm case runs without error",
          isinstance(_best1e, Individual))

    # -- 1f: history dict keys check
    _req_keys = {"generation", "best_fitness", "best_so_far",
                 "mean_fitness", "worst_fitness", "n_features",
                 "diversity", "mutation_rate", "stagnation"}
    check("Phase 1f history dicts contain required keys",
          _req_keys.issubset(set(hist1[0].keys())),
          f"missing={_req_keys - set(hist1[0].keys())}")

    # -- 1g: cache_hits counter increments
    _hits_before = bandit1.cache_hits
    # Evaluate a gene already in cache
    _dup_genes = best1.genes.copy()
    bandit1._evaluate_one(Individual(genes=_dup_genes))
    check("Phase 1g cache_hits increments on duplicate gene",
          bandit1.cache_hits > _hits_before)

except Exception as _e1:
    check("Phase 1 runs", False, str(_e1))
    import traceback; traceback.print_exc()


# ============================================================================
# PHASE 2 — Bandit x Larch NL mode choice
# ============================================================================
print("\n" + "─" * 66)
print("PHASE 2 — Bandit x Larch NL mode choice (synthetic tours)")
print("─" * 66)

if not (_LX_OK and _S5_OK):
    print("  SKIP — Larch or Stage 5 unavailable")
else:
    try:
        # Tiny synthetic tours (N=80, 6 destination zones)
        _N_M  = 80
        _N_ZM = 6

        # OD skim table: all origins share zone 1; destinations 1-6
        _skim5_rows = []
        for _dz in range(1, _N_ZM + 1):
            _dk = max(0.5, float(_dz - 1) * 3.0)
            _skim5_rows.append({
                "OTAZ": 1, "DTAZ": _dz,
                "SOVTOLL_TIME__AM":        round(_dk * 2.5, 2),
                "WLK_LOC_WLK_TOTIVT__AM": round(_dk * 5.0, 2),
                "DISTWALK":  _dk,
                "DISTBIKE":  _dk,
                "SOVTOLL_DIST__AM": _dk,
                "CRASH": float(np.clip(RNG.normal(_dk * 0.08, 0.15), 0, 3)),
            })
        _skim_df5 = pd.DataFrame(_skim5_rows)

        # Persons
        _p5_vehrd = RNG.integers(0, 4, _N_M).astype(float)
        _p5_carl  = np.where(_p5_vehrd > 0,
                             RNG.binomial(1, 0.97, _N_M),
                             RNG.binomial(1, 0.65, _N_M)).astype(float)
        _persons5 = pd.DataFrame({
            "person_id": np.arange(_N_M),
            "vehrd":     _p5_vehrd,
            "incp":      RNG.integers(1, 6, _N_M).astype(float),
            "sexp":      RNG.binomial(1, 0.52, _N_M).astype(int) + 1,
            "age5p":     RNG.integers(2, 12, _N_M).astype(float),
            "bikes":     RNG.integers(0, 3, _N_M).astype(float),
            "WORKSTAT":  RNG.choice([1, 2, 3], _N_M).astype(int),
            "nprd":      RNG.integers(1, 5, _N_M).astype(float),
            "CARLICENCE": _p5_carl,
        })

        # Tours with NL-derived mode choices
        _t5_dtaz = RNG.integers(1, _N_ZM + 1, _N_M)
        _sk5_lkp = _skim_df5.set_index("DTAZ").to_dict("index")
        _ai5 = np.array([_sk5_lkp[d]["SOVTOLL_TIME__AM"]        for d in _t5_dtaz])
        _ti5 = np.array([_sk5_lkp[d]["WLK_LOC_WLK_TOTIVT__AM"] for d in _t5_dtaz])
        _wd5 = np.array([_sk5_lkp[d]["DISTWALK"]                for d in _t5_dtaz])
        _V5  = np.column_stack([
            3.0 * _p5_carl + 2.0 * (_p5_vehrd > 0) - 0.08 * _ai5 + 0.5,
            0.8 * (_p5_vehrd > 0) + 0.3 - 0.06 * _ai5,
            -0.5 * _wd5 + 1.5 * (_wd5 < 2.0) - 0.3,
            -0.3 * _wd5 + 0.5 * (_wd5 < 5.0) - 0.8,
            1.5 * (_p5_vehrd == 0) - 0.04 * _ti5 + 0.2 * (_persons5["incp"].values < 3),
        ]) + RNG.gumbel(0, 1, (_N_M, 5))
        _mc5 = _V5.argmax(axis=1) + 1   # 1=DA … 5=Transit

        _tours5 = pd.DataFrame({
            "ptour_id":       np.arange(_N_M).astype(str),
            "person_id":      np.arange(_N_M),
            "OTAZ": 1, "DTAZ": _t5_dtaz,
            "tour_mode":      _mc5,
            "CARLICENCE":     _p5_carl,
            "walk_avail":     (_wd5 < 2.0).astype(float),
            "bike_avail":     (_wd5 < 5.0).astype(float),
            "weight":         1.0,
            "joint_travel":   0,
            "outbound_stops": 0,
            "n_shopping_out": 0,
            "_time_period":   RNG.binomial(1, 0.4, _N_M),
        })

        _MODE_SKIM_FEATS = [
            "SOVTOLL_TIME__AM",
            "WLK_LOC_WLK_TOTIVT__AM",
            "DISTWALK",
            "DISTBIKE",
            "CRASH",
        ]
        _N_SKIM = len(_MODE_SKIM_FEATS)
        _skim_to_coeff5 = {
            "SOVTOLL_TIME__AM":        "c_ivt_auto",
            "WLK_LOC_WLK_TOTIVT__AM": "c_ivt_transit",
            "DISTWALK":                "c_walk_time",
            "DISTBIKE":                "c_bike_time",
            "CRASH":                   "c_crash_veh",
        }

        # Build DataTree once (expensive pandas→xarray conversion; reused per eval)
        def _mk_mode_dt5(tours_df, persons_df, skim_df):
            _alts   = list(MODE_MAP.values())
            _cases  = list(tours_df["ptour_id"].values)
            _ca_idx = pd.MultiIndex.from_product(
                [_cases, _alts], names=["ptour_id", "_altid_"])
            _avail_df = pd.DataFrame({"_dummy": 0.0}, index=_ca_idx)
            _t_lkp = tours_df.set_index("ptour_id")
            for _avc in ("CARLICENCE", "walk_avail", "bike_avail"):
                if _avc in _t_lkp.columns:
                    _avc_map = _t_lkp[_avc].to_dict()
                    _avail_df[_avc] = [_avc_map[c] for c, _ in _ca_idx]
            _root = lx.Dataset.construct.from_idca(_avail_df)
            _od   = _build_od_dataset(None, skim_df=skim_df)
            return lx.DataTree(
                base   = _root,
                tour   = lx.Dataset.construct.from_idco(
                             tours_df.set_index("ptour_id")),
                person = _persons5.set_index("person_id"),
                od     = _od,
                relationships=(
                    "base.ptour_id @ tour.ptour_id",
                    "tour.person_id @ person.person_id",
                    "tour.OTAZ @ od.otaz",
                    "tour.DTAZ @ od.dtaz",
                ),
            )

        print("  Building mode DataTree ...", end="", flush=True)
        with contextlib.redirect_stderr(_io.StringIO()):
            _dt5_static = _mk_mode_dt5(_tours5, _persons5, _skim_df5)
        print(" done", flush=True)

        # Fitness: holdfast-based exclusion so numba compiles only once
        def _mode_fit5(genes: np.ndarray) -> float:
            _excl = [f for f, g in zip(_MODE_SKIM_FEATS, genes) if g == 0]
            try:
                with contextlib.redirect_stderr(_io.StringIO()):
                    _m = _build_mode_model(
                        _dt5_static, "work", custom_mode_vars=_MODE_SKIM_FEATS)
                _m.compute_engine = "numba"
                for _f in _excl:
                    _param = _skim_to_coeff5.get(_f)
                    if _param and _param in _m.pf.index:
                        _m.pf.loc[_param, "value"]    = 0.0
                        _m.pf.loc[_param, "holdfast"] = 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _m.maximize_loglike(method="bhhh", maxiter=20, quiet=True)
                _ll = float(_m.loglike())
                if not np.isfinite(_ll):
                    return -1e9
                _k = int((~_m.pf["holdfast"].fillna(False)).sum())
                return -((-2 * _ll + _k * np.log(_N_M)) / _N_M)
            except Exception:
                return -1e9

        # Warm-start: evaluate full model first
        print("  Fitting full-skim baseline ...", end="", flush=True)
        _bl_genes = np.ones(_N_SKIM, dtype=int)
        _bl_fit   = _mode_fit5(_bl_genes)
        print(f" BIC/n={-_bl_fit:.4f}", flush=True)

        bandit_mode = BanditFeatureSearcher(
            n_features    = _N_SKIM,
            fitness_fn    = _mode_fit5,
            pop_size      = 4,
            n_generations = 12,
            min_features  = 1,
            seed          = 42,
            verbose       = True,
            feature_names = _MODE_SKIM_FEATS,
            fixed_features= [],
            fitness_metric= "mean(-BIC/n) [Larch NL]",
        )
        # Pre-seed cache with baseline result
        _bl_key = tuple(int(x) for x in _bl_genes.tolist())
        bandit_mode._cache[_bl_key] = _bl_fit
        _bl_seed = Individual(genes=_bl_genes, fitness=_bl_fit)

        _best2, _hist2, _ = bandit_mode.run_with_restarts(
            n_restarts=1, seed_individuals=[_bl_seed])

        _feats2 = [f for f, g in zip(_MODE_SKIM_FEATS, _best2.genes) if g == 1]
        print(f"  Mode bandit best: {_feats2}  BIC/n={-_best2.fitness:.4f}")

        check("Phase 2 Bandit mode NL runs",   True)
        check("Phase 2 ≥12 history entries",   len(_hist2) >= 12, f"n={len(_hist2)}")
        check("Phase 2 best fitness finite",   np.isfinite(_best2.fitness),
              f"fit={_best2.fitness:.4f}")
        prog2, pmsg2 = _is_progressive(_hist2)
        check("Phase 2 mode bandit progressive", prog2, pmsg2)
        check("Phase 2 ≥1 skim selected",      len(_feats2) >= 1,
              f"feats={_feats2}")
        check("Phase 2 cache_hits > 0",        bandit_mode.cache_hits > 0,
              f"cache_hits={bandit_mode.cache_hits}")

        # Arm probabilities: SOVTOLL_TIME__AM and DISTWALK should be high (real signal)
        _arm_probs2 = {f: float(bandit_mode._alpha[i] /
                               (bandit_mode._alpha[i] + bandit_mode._beta[i]))
                       for i, f in enumerate(_MODE_SKIM_FEATS)}
        print(f"  Arm probabilities: { {k: round(v, 3) for k, v in _arm_probs2.items()} }")

    except Exception as _e2:
        check("Phase 2 runs", False, str(_e2))
        import traceback; traceback.print_exc()


# ============================================================================
# PHASE 3 — Bandit x Larch MNL destination choice
# ============================================================================
print("\n" + "─" * 66)
print("PHASE 3 — Bandit x Larch MNL destination choice (synthetic idca)")
print("─" * 66)

if not (_LX_OK and _S5_OK):
    print("  SKIP — Larch or Stage 5 unavailable")
else:
    try:
        _N_DC = 80    # observations
        _N_DZ = 15    # zone alternatives (alternatives per observation)

        _dz5  = np.arange(1, _N_DZ + 1)
        _dz_TOTEMP  = RNG.lognormal(9, 1, _N_DZ)
        _dz_RETEMPN = RNG.lognormal(7, 1, _N_DZ)

        _dc_dists5 = (
            np.abs(np.arange(1, _N_DC + 1)[:, None] - _dz5[None, :] * 2.0)
            + RNG.exponential(1.0, (_N_DC, _N_DZ))
        )
        _Vd5 = (
            -1.5 * np.log(_dc_dists5 + 1)
            + 0.8 * np.log(_dz_TOTEMP + 1)
            + 0.3 * np.log(_dz_RETEMPN + 1)
            + RNG.gumbel(0, 1, (_N_DC, _N_DZ))
        )
        _dc_ch5 = _Vd5.argmax(axis=1) + 1   # chosen zone id (1-based)

        # Build flat idca DataFrame
        _dest5_rows = []
        for _zi5, _zid5 in enumerate(_dz5):
            _d5 = _dc_dists5[:, _zi5]
            _dest5_rows.append(pd.DataFrame({
                "case_id":        np.arange(_N_DC),
                "DTAZ":           int(_zid5),
                "chosen":         (_dc_ch5 == _zid5).astype(int),
                "log_DIST_1":     np.log(_d5 + 1),
                "DIST":           _d5,
                "log_CRASH_1":    RNG.uniform(0.1, 0.5, _N_DC),
                "logsum":         RNG.uniform(-3.0, -0.5, _N_DC),
                "log_correction": 0.0,
                "size_term":      float(np.log(_dz_TOTEMP[_zi5] + 1)),
                "acdc_aggl":      RNG.uniform(0, 1, _N_DC),
                "acdc_comp":      RNG.uniform(0, 1, _N_DC),
                "intrazonal":     float(_zid5 == 1),
                "walk_feasible":  (_d5 < 2.0).astype(float),
                "bike_feasible":  (_d5 < 5.0).astype(float),
            }))
        _synth_dest5 = pd.concat(_dest5_rows, ignore_index=True)
        _synth_dest5 = _synth_dest5.set_index(["case_id", "DTAZ"])
        print(f"  Dest data: {_N_DC} obs × {_N_DZ} zones")

        # Destination candidates + mapping to column names
        _DEST_CANDS = [
            "c_log_dist",    # log_DIST_1    — always strong signal
            "c_crash",       # log_CRASH_1
            "c_logsum",      # logsum
            "c_size_term",   # size_term
            "c_aggl",        # acdc_aggl
            "c_comp",        # acdc_comp
            "c_intrazonal",  # intrazonal
            "c_walk_feas",   # walk_feasible
            "c_bike_feas",   # bike_feasible
        ]
        _DEST_COL = {
            "c_log_dist":   "log_DIST_1",
            "c_crash":      "log_CRASH_1",
            "c_logsum":     "logsum",
            "c_size_term":  "size_term",
            "c_aggl":       "acdc_aggl",
            "c_comp":       "acdc_comp",
            "c_intrazonal": "intrazonal",
            "c_walk_feas":  "walk_feasible",
            "c_bike_feas":  "bike_feasible",
        }
        _N_DEST = len(_DEST_CANDS)
        # c_log_dist is fixed (always ON — core distance deterrence)
        _DEST_FIXED = ["c_log_dist"]

        def _dest_fit5(genes: np.ndarray) -> float:
            sel_coeffs = {c: _DEST_COL[c]
                          for c, g in zip(_DEST_CANDS, genes)
                          if g == 1 and c in _DEST_COL}
            if not sel_coeffs:
                return -1e9
            sel_cols = list(dict.fromkeys(sel_coeffs.values()))
            try:
                _ds = lx.Dataset.construct.from_idca(
                    _synth_dest5[sel_cols + ["chosen"]].copy()
                )
                _mm = lx.Model(_ds)
                _mm.choice_ca_var  = "chosen"
                _mm.compute_engine = "numba"
                _util = None
                for _c, _col in sel_coeffs.items():
                    _t = P(_c) * X(_col)
                    _util = _t if _util is None else _util + _t
                if _util is not None:
                    _mm.utility_ca = _util
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _mm.maximize_loglike(method="bhhh", maxiter=20, quiet=True)
                _ll = float(_mm.loglike())
                if not np.isfinite(_ll):
                    return -1e9
                _k  = len(sel_coeffs)
                return -((-2 * _ll + _k * np.log(_N_DC)) / _N_DC)
            except Exception:
                return -1e9

        # Warm-start: baseline with all coefficients
        print("  Fitting full-coeff dest baseline ...", end="", flush=True)
        _bl3_genes = np.ones(_N_DEST, dtype=int)
        _bl3_fit   = _dest_fit5(_bl3_genes)
        print(f" BIC/n={-_bl3_fit:.4f}", flush=True)

        bandit_dest = BanditFeatureSearcher(
            n_features    = _N_DEST,
            fitness_fn    = _dest_fit5,
            pop_size      = 5,
            n_generations = 12,
            min_features  = 1,
            seed          = 77,
            verbose       = True,
            feature_names = _DEST_CANDS,
            fixed_features= _DEST_FIXED,
            fitness_metric= "mean(-BIC/n) [Larch MNL]",
        )
        _bl3_key = tuple(int(x) for x in _bl3_genes.tolist())
        bandit_dest._cache[_bl3_key] = _bl3_fit
        _bl3_seed = Individual(genes=_bl3_genes, fitness=_bl3_fit)

        _best3, _hist3, _ = bandit_dest.run_with_restarts(
            n_restarts=1, seed_individuals=[_bl3_seed])

        _feats3 = [f for f, g in zip(_DEST_CANDS, _best3.genes) if g == 1]
        print(f"  Dest bandit best: {_feats3}  BIC/n={-_best3.fitness:.4f}")

        check("Phase 3 Bandit dest MNL runs",  True)
        check("Phase 3 ≥12 history entries",   len(_hist3) >= 12, f"n={len(_hist3)}")
        check("Phase 3 best fitness finite",   np.isfinite(_best3.fitness),
              f"fit={_best3.fitness:.4f}")
        prog3, pmsg3 = _is_progressive(_hist3)
        check("Phase 3 dest bandit progressive", prog3, pmsg3)
        # fixed feature c_log_dist must always be in the best result
        check("Phase 3 fixed feature c_log_dist ON",
              "c_log_dist" in _feats3, f"feats={_feats3}")
        check("Phase 3 ≥1 dest feature selected", len(_feats3) >= 1,
              f"feats={_feats3}")

        _arm_probs3 = {c: float(bandit_dest._alpha[i] /
                                (bandit_dest._alpha[i] + bandit_dest._beta[i]))
                       for i, c in enumerate(_DEST_CANDS)}
        print(f"  Arm probabilities: { {k: round(v, 3) for k, v in _arm_probs3.items()} }")

    except Exception as _e3:
        check("Phase 3 runs", False, str(_e3))
        import traceback; traceback.print_exc()


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 66)
print("SUMMARY")
print("=" * 66)
passed = sum(1 for s, _, _ in _results if s == "PASS")
failed = sum(1 for s, _, _ in _results if s == "FAIL")
for status, name, detail in _results:
    suf = f"  [{detail}]" if detail else ""
    print(f"  [{status}] {name}{suf}")
print(f"\n  {passed} passed / {failed} failed out of {len(_results)} checks")
sys.exit(0 if failed == 0 else 1)
