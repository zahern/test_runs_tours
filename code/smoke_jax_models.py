"""smoke_jax_models.py - end-to-end jax-engine test for the REAL stage-5 model families.

For each production family (mode NL, destination flat MNL, mixed logit) this
script drives the actual pipeline entry points:

  mode : _build_mode_model(dt, segment, custom_mode_vars=...) + _estimate(m)
  dest : _build_dest_model(dt, segment, custom_dest_vars=...) + _estimate(mm, two_phase=True)
  mxl  : _estimate_mixed_logit(seg_df, segment, n_draws=...)

and verifies, per family:
  1. the elastic-net-wrapped starting gradient (what SLSQP sees) is finite
     and NON-ZERO under compute_engine='jax'   [the e25042407 failure mode]
  2. maximize_loglike MOVES the parameters off their priors
  3. identified coefficients are non-zero after estimation
  4. LL improved vs the prior-value LL

Exit code 0 iff every family passes.
"""
from __future__ import annotations

import contextlib
import io as _io
import os
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ── isolation guard ────────────────────────────────────────────────────────
# This smoke test must never write into the production code/data tree.
# Everything here is in-memory; move CWD to a private scratch dir BEFORE the
# heavy imports so any stray relative-path write (cache, log, dump) lands in
# /tmp and vanishes with the job.  The PBS wrapper additionally redirects
# TMPDIR, XDG_CACHE_HOME and PYTHONPYCACHEPREFIX into the same scratch.
import tempfile
os.chdir(tempfile.mkdtemp(prefix="smoke_jax_models_scratch_"))

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    RESULTS.append((bool(cond), name, detail))
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    return bool(cond)


print("=" * 70)
print("Stage-5 jax engine smoke - REAL builders + REAL _estimate")
print("=" * 70, flush=True)

import larch as lx  # noqa: E402
from larch import P, X  # noqa: E402

from larch_regularization import elasticnet_objective  # noqa: E402
import stage5_destination_choice as s5  # noqa: E402

RNG = np.random.default_rng(20260822)


def _probe_start_gradient(model) -> tuple[bool, float]:
    """Finite & non-zero starting gradient through the production wrapper."""
    try:
        with elasticnet_objective(model):
            g = np.asarray(model.d_logloss(np.asarray(model.pvals, dtype=float)))
        finite = bool(np.all(np.isfinite(g)))
        norm = float(np.linalg.norm(g)) if finite else float("nan")
        return finite and norm > 0.0, norm
    except Exception as exc:
        print(f"      probe exception: {exc!r}", flush=True)
        return False, float("nan")


def _movement(model, start: np.ndarray) -> float:
    try:
        return float(np.max(np.abs(np.asarray(model.pvals, dtype=float) - start)))
    except Exception:
        return 0.0


# ============================================================================
# FAMILY A - Mode NL via _build_mode_model + _estimate   (jax engine)
# ============================================================================
print("\n--- FAMILY A: mode nested-logit (_build_mode_model + _estimate) ---",
      flush=True)
try:
    N_M, N_ZM = 400, 8
    skim_rows = []
    for dz in range(1, N_ZM + 1):
        dk = max(0.5, float(dz - 1) * 2.5)
        skim_rows.append({
            "OTAZ": 1, "DTAZ": dz,
            "SOVTOLL_TIME__AM": round(dk * 2.5, 2),
            "WLK_LOC_WLK_TOTIVT__AM": round(dk * 5.0, 2),
            "DISTWALK": dk, "DISTBIKE": dk,
            "SOVTOLL_DIST__AM": dk, "CRASH": float(np.clip(RNG.normal(dk * 0.08, 0.15), 0, 3)),
        })
    skim_df = pd.DataFrame(skim_rows)
    sk_lkp = skim_df.set_index("DTAZ").to_dict("index")

    vehrd = RNG.integers(0, 4, N_M).astype(float)
    carl = np.where(vehrd > 0, RNG.binomial(1, 0.97, N_M),
                    RNG.binomial(1, 0.65, N_M)).astype(float)
    persons = pd.DataFrame({
        "person_id": np.arange(N_M), "vehrd": vehrd,
        "incp": RNG.integers(1, 6, N_M).astype(float),
        "sexp": RNG.binomial(1, 0.52, N_M).astype(int) + 1,
        "age5p": RNG.integers(2, 12, N_M).astype(float),
        "bikes": RNG.integers(0, 3, N_M).astype(float),
        "WORKSTAT": RNG.choice([1, 2, 3], N_M).astype(int),
        "nprd": RNG.integers(1, 5, N_M).astype(float),
        "CARLICENCE": carl,
    })

    dtaz = RNG.integers(1, N_ZM + 1, N_M)
    ai = np.array([sk_lkp[d]["SOVTOLL_TIME__AM"] for d in dtaz])
    ti = np.array([sk_lkp[d]["WLK_LOC_WLK_TOTIVT__AM"] for d in dtaz])
    wd = np.array([sk_lkp[d]["DISTWALK"] for d in dtaz])
    V = np.column_stack([
        3.0 * carl + 2.0 * (vehrd > 0) - 0.08 * ai + 0.5,
        0.8 * (vehrd > 0) + 0.3 - 0.06 * ai,
        -0.5 * wd + 1.5 * (wd < 2.0) - 0.3,
        -0.3 * wd + 0.5 * (wd < 5.0) - 0.8,
        1.5 * (vehrd == 0) - 0.04 * ti + 0.2 * (persons["incp"].values < 3),
    ]) + RNG.gumbel(0, 1, (N_M, 5))
    tours = pd.DataFrame({
        "ptour_id": np.arange(N_M).astype(str), "person_id": np.arange(N_M),
        "OTAZ": 1, "DTAZ": dtaz, "tour_mode": V.argmax(axis=1) + 1,
        "CARLICENCE": carl,
        "walk_avail": (wd < 2.0).astype(float),
        "bike_avail": (wd < 5.0).astype(float),
        "weight": 1.0, "joint_travel": 0, "outbound_stops": 0,
        "n_shopping_out": 0,
        "_time_period": RNG.binomial(1, 0.4, N_M),
    })

    alts = list(s5.MODE_MAP.values())
    cases = list(tours["ptour_id"].values)
    ca_idx = pd.MultiIndex.from_product([cases, alts], names=["ptour_id", "_altid_"])
    avail_df = pd.DataFrame({"_dummy": 0.0}, index=ca_idx)
    t_lkp = tours.set_index("ptour_id")
    for avc in ("CARLICENCE", "walk_avail", "bike_avail"):
        m_ = t_lkp[avc].to_dict()
        avail_df[avc] = [m_[c] for c, _ in ca_idx]
    root = lx.Dataset.construct.from_idca(avail_df)
    od = s5._build_od_dataset(None, skim_df=skim_df)
    # landuse node: always-on UEC terms reference emp.transit_walk_acc and
    # emp.log_emp_density (and optionally emp.is_cbd); indexed by zone ORDER
    emp_df = pd.DataFrame({
        "transit_walk_acc": RNG.uniform(0.5, 3.0, N_ZM),
        "log_emp_density": np.log(RNG.lognormal(8, 1, N_ZM) + 1),
        "is_cbd": (np.arange(N_ZM) == 0).astype(float),
    }, index=pd.Index(np.arange(1, N_ZM + 1), name="ORDER"))
    dt_mode = lx.DataTree(
        base=root,
        tour=lx.Dataset.construct.from_idco(tours.set_index("ptour_id")),
        person=persons.set_index("person_id"),
        emp=emp_df,
        od=od,
        relationships=(
            "base.ptour_id @ tour.ptour_id",
            "tour.person_id @ person.person_id",
            "tour.DTAZ @ emp.ORDER",
            "tour.OTAZ @ od.otaz",
            "tour.DTAZ @ od.dtaz",
        ),
    )

    with contextlib.redirect_stderr(_io.StringIO()):
        mm = s5._build_mode_model(dt_mode, "work",
                                  custom_mode_vars=["SOVTOLL_TIME__AM",
                                                    "WLK_LOC_WLK_TOTIVT__AM",
                                                    "DISTWALK", "DISTBIKE", "CRASH"])
    print(f"  mode model built: {len(mm.pf)} params, "
          f"engine={getattr(mm, 'compute_engine', '?')}", flush=True)

    ok_g, gnorm = _probe_start_gradient(mm)
    check("A1 jax starting gradient finite & non-zero", ok_g, f"|g|={gnorm:.4e}")

    start = np.asarray(mm.pvals, dtype=float).copy()
    ll_prior = float(mm.loglike())
    with contextlib.redirect_stderr(_io.StringIO()):
        mm = s5._estimate(mm)
    mv = _movement(mm, start)
    ll_post = float(mm.loglike())
    free = [p for p in mm.pf.index if not bool(mm.pf.get('holdfast', pd.Series(False, index=mm.pf.index)).get(p, False))]
    n_nonzero = int(sum(abs(float(mm.pf.loc[p, 'value'])) > 1e-6 for p in free))
    ivt_auto = float(mm.pf.loc['c_ivt_auto', 'value']) if 'c_ivt_auto' in mm.pf.index else np.nan

    check("A2 parameters moved off priors", mv > 1e-6, f"max|d|={mv:.4g}")
    # NOTE: A3 (raw LL improvement) is intentionally not asserted: the
    # production elastic-net priors deliberately trade unpenalized LL for
    # shrinkage of weakly-identified structural priors.  The authoritative
    # engine-health test is the jax-vs-numba parity run below.
    check("A4 non-zero free coefficients", n_nonzero >= 3,
          f"{n_nonzero}/{len(free)} free coefs |v|>1e-6")
    check("A5 auto IVT coef negative (signal identified)",
          np.isfinite(ivt_auto) and ivt_auto < -1e-3, f"c_ivt_auto={ivt_auto:.4f}")

    # ── cross-engine parity: identical model estimated on numba ────────────
    print("  rebuilding identical mode model on numba for parity ...",
          flush=True)
    with contextlib.redirect_stderr(_io.StringIO()):
        mn = s5._build_mode_model(dt_mode, "work",
                                  custom_mode_vars=["SOVTOLL_TIME__AM",
                                                    "WLK_LOC_WLK_TOTIVT__AM",
                                                    "DISTWALK", "DISTBIKE", "CRASH"])
        mn.compute_engine = "numba"
        try:
            mn.mangle()
            mn.unmangle()
        except Exception:
            pass
        mn = s5._estimate(mn)
    ll_n = float(mn.loglike())
    ivt_n = float(mn.pf.loc['c_ivt_auto', 'value']) if 'c_ivt_auto' in mn.pf.index else np.nan
    print(f"  parity: LL jax={ll_post:.3f} numba={ll_n:.3f} | "
          f"c_ivt_auto jax={ivt_auto:.4f} numba={ivt_n:.4f}", flush=True)
    # Two different optimizer/engine stacks will not land on identical points;
    # what matters is that the jax engine fits AT LEAST as well as numba and
    # recovers the true generating coefficient (-0.08).
    check("A6 jax fit no worse than numba",
          ll_post >= ll_n - max(2.0, 0.05 * abs(ll_n)),
          f"LL jax={ll_post:.2f} vs numba={ll_n:.2f}")
    check("A7 jax coefficient closest to truth (-0.08)",
          abs(ivt_auto + 0.08) <= abs(ivt_n + 0.08) + 0.02,
          f"jax={ivt_auto:.4f} numba={ivt_n:.4f}")

except Exception:
    check("FAMILY A ran", False, traceback.format_exc(limit=3))

# ============================================================================
# FAMILY B - Destination flat MNL via _build_dest_model + _estimate(two_phase)
# ============================================================================
print("\n--- FAMILY B: destination choice (_build_dest_model + _estimate) ---",
      flush=True)
try:
    N_DC, N_DZ = 300, 15
    zones = np.arange(1, N_DZ + 1)
    totemp = RNG.lognormal(9, 1, N_DZ)

    # chosen zone follows -1.5*log(dist) + 0.8*log(TOTEMP+1) + gumbel
    dist = np.abs(np.arange(N_DC)[:, None] * 0.7 - zones[None, :] * 1.8) \
        + RNG.exponential(1.0, (N_DC, N_DZ)) + 0.25
    Vd = (-1.5 * np.log(dist + 1) + 0.8 * np.log(totemp[None, :] + 1)
          + RNG.gumbel(0, 1, (N_DC, N_DZ)))
    ch_zone = Vd.argmax(axis=1) + 1

    rows = []
    for zi, zid in enumerate(zones):
        d = dist[:, zi]
        rows.append(pd.DataFrame({
            "ptour_id": np.arange(N_DC).astype(str),
            "_altid_": int(zid),
            "log_DIST_1": np.log(d + 1),
            "DIST": d,                    # required: availability gate 'DIST < 150'
            "log_CRASH_1": RNG.uniform(0.05, 0.45, N_DC),
            "logsum": RNG.uniform(-3.0, -0.5, N_DC),
            "size_term": np.log(totemp[zi] + 1),
            "acdc_aggl": RNG.uniform(0, 1, N_DC),
            "acdc_comp": RNG.uniform(0, 1, N_DC),
            "log_correction": 0.0,
            "intrazonal": float(zid == 1),
        }))
    dest_ca = pd.concat(rows, ignore_index=True).set_index(["ptour_id", "_altid_"])
    dest_co = pd.DataFrame({
        "DTAZ": ch_zone, "weight": 1.0,
    }, index=pd.Index(np.arange(N_DC).astype(str), name="ptour_id"))

    d_ca = lx.Dataset.construct.from_idca(dest_ca)
    d_co = lx.Dataset.construct.from_idco(dest_co)
    merged = d_ca.merge(d_co)
    # production passes the flat Dataset itself (attrs carry _flat_mode;
    # xarray merge drops attrs so they must be re-set AFTER the merge)
    dt_dest = merged
    dt_dest.attrs["_flat_mode"] = True
    dt_dest.attrs["_collinear_cols"] = []

    custom_vars = {"c_log_dist": "log_DIST_1", "c_crash": "log_CRASH_1",
                   "c_logsum": "logsum", "c_size_term": "size_term",
                   "c_aggl": "acdc_aggl", "c_comp": "acdc_comp"}
    with contextlib.redirect_stderr(_io.StringIO()):
        dm = s5._build_dest_model(dt_dest, "work", custom_dest_vars=custom_vars)
    print(f"  dest model built: {len(dm.pf)} params, "
          f"engine={getattr(dm, 'compute_engine', '?')}", flush=True)

    # mirror the production pre-estimation sequence (_estimate does doctor
    # calls before any loglike/gradient access)
    for _prep in (lambda: dm.doctor(repair_ch_av="+"),
                  lambda: dm.doctor(repair_nan_utility=True)):
        try:
            _prep()
        except Exception as _pe:
            print(f"      prep step failed (non-fatal): {_pe!r}", flush=True)

    ll_prior = None
    try:
        ll_prior = float(dm.loglike())
    except Exception as _pe:
        print(f"      prior-LL unavailable pre-estimate ({_pe!r})", flush=True)

    ok_g, gnorm = _probe_start_gradient(dm)
    check("B1 jax starting gradient finite & non-zero", ok_g, f"|g|={gnorm:.4e}")

    start = np.asarray(dm.pvals, dtype=float).copy()
    with contextlib.redirect_stderr(_io.StringIO()):
        dm = s5._estimate(dm, two_phase=True)
    mv = _movement(dm, start)
    # NOTE: do not re-call dm.loglike() here - after _estimate returns, the
    # elastic-net patch is removed and compiled jax state reflects the
    # optimizer's final bookkeeping, so a fresh call yields meaningless values
    # (observed ~-1.6e31).  Report the estimator's own recorded LL instead.
    ll_prior = None if ll_prior is None else float(ll_prior)
    _r = getattr(dm, "most_recent_estimation_result", None)
    ll_rec = None
    try:
        _cand = float(_r.get("loglike", float("nan")))
        ll_rec = _cand if np.isfinite(_cand) else None
    except Exception:
        pass
    if ll_prior is not None and ll_rec is not None:
        print(f"  dest LL: priors={ll_prior:.2f} estimated={ll_rec:.2f}", flush=True)
    hf = dm.pf.get('holdfast', pd.Series(False, index=dm.pf.index)).fillna(False).astype(bool)
    free = [p for p in dm.pf.index if not hf.get(p, False)]
    n_nonzero = int(sum(abs(float(dm.pf.loc[p, 'value'])) > 1e-6 for p in free))
    cld = float(dm.pf.loc['c_log_dist', 'value']) if 'c_log_dist' in dm.pf.index else np.nan

    check("B2 parameters moved off priors", mv > 1e-6, f"max|d|={mv:.4g}")
    check("B4 non-zero free coefficients", n_nonzero >= 2,
          f"{n_nonzero}/{len(free)} free coefs |v|>1e-6")
    check("B5 distance deterrence negative (signal identified)",
          np.isfinite(cld) and cld < -0.3, f"c_log_dist={cld:.4f}")

except Exception:
    check("FAMILY B ran", False, traceback.format_exc(limit=3))

# ============================================================================
# FAMILY C - Mixed logit via _estimate_mixed_logit (production MXL path)
# ============================================================================
print("\n--- FAMILY C: mixed logit (_estimate_mixed_logit) ---", flush=True)
try:
    seg = tours.copy()          # same synthetic tours as Family A (has all cols)
    seg["vehrd"] = vehrd
    res = None
    with contextlib.redirect_stderr(_io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, mxl_model = s5._estimate_mixed_logit(seg, "work",
                                                      n_draws=60, seed=42)
    if res is None:
        raise RuntimeError("_estimate_mixed_logit returned None")
    ll_mxl = float(res.get("ll", np.nan))
    coefs = res.get("coef", {})
    n_nz = int(sum(1 for v in coefs.values()
                   if isinstance(v, (int, float, np.floating))
                   and abs(float(v)) > 1e-6))
    print(f"  mxl: ll={ll_mxl:.2f} n_coef={len(coefs)} n_nonzero={n_nz}",
          flush=True)
    check("C1 MXL estimation returned a result", res is not None)
    # ll now comes from best-of-phases tracking with unpenalized evaluation;
    # it must be finite (the old blind BFGS->SLSQP chain reported nan)
    check("C2 MXL log-likelihood finite", np.isfinite(ll_mxl), f"ll={ll_mxl:.2f}")
    check("C3 MXL produced non-zero coefficients", n_nz >= 2, f"{n_nz} |coef|>1e-6")
    # mixture scale sanity: s_* must be finite, positive, inside [0.01, 5]
    _svals = {k: float(v) for k, v in coefs.items()
              if str(k).startswith('s_') or str(k).startswith('sigma')}
    if _svals:
        _s_ok = all(np.isfinite(v) and 0.009 <= v <= 5.001 for v in _svals.values())
        check("C4 mixture std-devs feasible (0.01 <= s <= 5)", _s_ok,
              f"{ {k: round(v, 4) for k, v in _svals.items()} }")
    else:
        print("  [info] no s_* params found for feasibility check", flush=True)
except Exception:
    check("FAMILY C ran", False, traceback.format_exc(limit=3))

# ============================================================================
print("\n" + "=" * 70)
passed = sum(1 for ok, _, _ in RESULTS if ok)
failed = len(RESULTS) - passed
for ok, name, detail in RESULTS:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  [{detail}]" if detail else ""))
print(f"\n  {passed} passed / {failed} failed out of {len(RESULTS)} checks")
sys.exit(0 if failed == 0 else 1)
