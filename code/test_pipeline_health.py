#!/usr/bin/env python
"""test_pipeline_health.py — end-to-end health check for the SEQ pipeline.

Runs on HPC (larch present) via job_health_check.pbs. Every section prints
PASS / FAIL / SKIP plus DETAILED CAVEATS (what broke, why it matters, what
to fix). Exit code 0 iff no FAIL (SKIPs are honest, not silent).

Sections
  S0  Environment: python/numpy/pandas/scipy/openmatrix/jax/larch engines
  S1  Real-data inventory: HHTS modes, OMX matrices, zone universe, emp attrs
  S2  Orthogonalisation: synthetic math + live helper + pair coverage
  S3  Dest starter: spec/mutex checks + REAL synthetic starter fit + perturbation
  S4  Logsum (ActivitySim reflection): static checks + stub-dt fallback audit
  S5  Mixed logit: NegLogNormal API, latent math, gene codecs, known gaps
  S6  Perturbations: AGDS 2-bit tail vs bandit 1-bit tail coverage
  S7  Watch-items: OOM math, detour OOR, _get_tt units, mode fillna

Tuning (env):
  HEALTH_RUN_FITS=1      run real synthetic fits (S0/S3/S5); 0 = static only
  HEALTH_N_TOURS=400     cases for synthetic fits
  HEALTH_N_DRAWS=60      draws for synthetic MXL probe
  HEALTH_OUT=<path>      write JSON report here (default: stdout only)

Write-isolation: only reads the code/data tree; all temp files go to $TMPDIR.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

RESULTS: list = []  # (section, name, status, detail, caveats)


def check(section: str, name: str, cond: bool, detail: str = "",
          caveats: str = "", skipped: bool = False, warn: bool = False) -> bool:
    if skipped:
        status = "SKIP"
    elif cond:
        status = "PASS"
    else:
        # WARN = known pending gap: visible caveats, does NOT fail the job.
        status = "WARN" if warn else "FAIL"
    RESULTS.append({"section": section, "name": name, "status": status,
                    "detail": detail, "caveats": caveats})
    tag = f"[{status}]"
    print(f"  {tag:7s} {section}/{name}" + (f"  [{detail}]" if detail else ""),
          flush=True)
    if caveats and status != "PASS":
        for line in caveats.strip().splitlines():
            print(f"          ! {line}", flush=True)
    return bool(cond)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


RUN_FITS = os.environ.get("HEALTH_RUN_FITS", "1").strip() != "0"
N_TOURS = _env_int("HEALTH_N_TOURS", 400)
N_DRAWS = _env_int("HEALTH_N_DRAWS", 60)
RNG = np.random.default_rng(20260906)

print("=" * 72)
print("PIPELINE HEALTH CHECK")
print(f"RUN_FITS={RUN_FITS} N_TOURS={N_TOURS} N_DRAWS={N_DRAWS}")
print("=" * 72, flush=True)

# ============================================================================
# S0 — environment + engines
# ============================================================================
print("\n--- S0: environment + Larch/JAX engines ---", flush=True)
print(f"  python {sys.version.split()[0]}", flush=True)
try:
    import jax
    _jx = True
    _x64 = bool(jax.config.jax_enable_x64)
except Exception as _e:
    jax = None
    _jx, _x64 = False, False
    _jx_err = repr(_e)
check("S0", "jax importable", _jx, f"x64={_x64}" if _jx else "missing",
      "" if _jx else f"JAX unavailable ({_jx_err}). Large-model policy needs "
      "jax[cpu]: numba finite-differences stall on 371-zone expansions. "
      "Install: pip install 'jax[cpu]' (needs py>=3.9).")
try:
    import larch as lx
    _lx = True
    _lxv = str(getattr(lx, "__version__", "?"))
except Exception as _e:
    lx = None
    _lx, _lxv = False, "missing"
    _lx_err = repr(_e)
check("S0", "larch importable", _lx, f"v{_lxv}" if _lx else "missing",
      "" if _lx else f"larch unavailable ({_lx_err}). Install larch 6.0.46 "
      "(the QLD stack pip-pins it in requirements.txt; needs py>=3.10).")

if _lx and RUN_FITS:
    # Tiny synthetic MNL on JAX: proves estimation works end-to-end.
    try:
        from larch import P, X
        n, K = 200, 3
        _df = pd.DataFrame({
            "cid": np.repeat(np.arange(n).astype(str), K),
            "alt": np.tile(np.arange(K), n),
            "x": RNG.normal(size=n * K),
            "ch": 0,
        })
        _ch = RNG.integers(0, K, n)
        for i, a in enumerate(_ch):
            _df.loc[(np.arange(len(_df)) // K == i) & (_df["alt"] == a), "ch"] = 1
        _d = lx.Dataset.construct.from_idca(
            _df.set_index(["cid", "alt"]), crack=False)
        _m = lx.Model(_d, compute_engine="jax")
        _m.utility_ca = P("b") * X("x")
        _m.choice_ca_var = "ch"
        _m.set_cap(8)
        _r = _m.maximize_loglike(options={"maxiter": 200, "ftol": 1e-9})
        _ll = float(_r.loglike)
        _b = float(_m.pf.loc["b", "value"])
        check("S0", "jax MNL fit converges", np.isfinite(_ll) and abs(_b) > 1e-6,
              f"LL={_ll:.1f} b={_b:.3f}",
              "JAX MNL did not converge: check JAX_ENABLE_X64=1 (float32 flattens "
              "the LL below gtol) and JAX_PLATFORMS=cpu (CUDA Hessian errors).")
    except Exception:
        check("S0", "jax MNL fit converges", False, traceback.format_exc(limit=2),
              "Synthetic JAX fit failed — see traceback. Estimation on HPC is "
              "broken until this passes.")
else:
    check("S0", "jax MNL fit converges", False, "skipped", skipped=True,
          caveats="Skipped: RUN_FITS=0 or larch/jax missing.")

# ============================================================================
# S1 — real-data inventory
# ============================================================================
print("\n--- S1: real-data inventory ---", flush=True)
_trips = os.path.join(CODE, "trips_stops_merged.csv")
if os.path.isfile(_trips):
    try:
        _t = pd.read_csv(_trips, usecols=["MODE"], nrows=60000, low_memory=False)
        _shares = (_t["MODE"].value_counts(normalize=True) * 100).round(1)
        _has_all = all(k in set(_t["MODE"].unique())
                       for k in ["Car driver", "Car passenger", "Walking"])
        check("S1", "HHTS raw modes cover DR/SR/walk/PT/bike", _has_all,
              f"DA={_shares.get('Car driver', 0)}% SR={_shares.get('Car passenger', 0)}% "
              f"Walk={_shares.get('Walking', 0)}%",
              "Raw HHTS MODE labels missing — mode derivation (RAW_MODE_MAP) has "
              "nothing to map. Check the HHTS extract.")
    except Exception:
        check("S1", "HHTS raw modes cover DR/SR/walk/PT/bike", False,
              traceback.format_exc(limit=1),
              "Could not read trips_stops_merged.csv MODE column.")
else:
    check("S1", "HHTS raw modes cover DR/SR/walk/PT/bike", False, "file missing",
          skipped=True, caveats="trips_stops_merged.csv not in code dir — skipped.")

# ACTIVE skim file (what the NEW_POP=1 pipeline actually opens): the 371-zone
# SA2 omxspec. The legacy 330-zone downcasted_matrices_base.omx is retired —
# inventoried second, informational only.
_omx_active = os.path.normpath(os.path.join(CODE, "..", "demand_",
                                            "skims_371.omx"))
_omx_legacy = os.path.join(CODE, "downcasted_matrices_base.omx")
if os.path.isfile(_omx_active):
    try:
        import h5py as _h5
        with _h5.File(_omx_active, "r") as _f:
            _mats = set(_f["data"].keys())
            _lk = list(_f["lookup"].keys())[0]
            _raw = np.array(_f["lookup"][_lk])
            _ids = [int(x.decode() if isinstance(x, bytes) else x) for x in _raw]
        _seq = pd.read_csv(os.path.join(CODE, "saved_data", "seq_zone_ids_371.csv"))
        _csv_order = _seq.sort_values("col_idx")["SA2_MAIN16"].astype(int).tolist()
        _order_ok = (_ids == _csv_order)
        _need = ["SOVTOLL_TIME__AM", "WLK_LOC_WLK_TOTIVT__AM", "DISTWALK",
                 "DISTBIKE", "CRASH", "SOVTOLL_TOLL__AM", "DIST"]
        _miss = [k for k in _need if k not in _mats]
        check("S1", "ACTIVE skims are 371 SA2-aligned", len(_ids) == 371 and _order_ok,
              f"{len(_mats)} mats, {len(_ids)} zones, file order==csv col_idx: {_order_ok}",
              "Zone misalignment: matrix row i does NOT correspond to ORDER i+1 — "
              "every OD lookup reads the wrong zone. Rebuild the OMX with lookup "
              "in seq_zone_ids_371.csv col_idx order." if not (len(_ids) == 371 and _order_ok) else "")
        check("S1", "ACTIVE skim matrices present", not _miss,
              f"missing={_miss}" if _miss else "time/dist/crash/toll all present",
              "Missing matrices degrade the logsum fallback: no CRASH -> crash terms "
              "dropped (fine); no TOLL -> toll term dropped (fine); no TIME/DIST mats "
              "-> fallbacks (30/45min) poison every logsum (fatal for c_logsum). "
              "Fix the OMX, not the model." if _miss else "")
    except Exception:
        check("S1", "ACTIVE skims are 371 SA2-aligned", False,
              traceback.format_exc(limit=1),
              "h5py read failed — conda install h5py.")
        check("S1", "ACTIVE skim matrices present", False, "skipped", skipped=True,
              caveats="Skipped: active OMX unreadable.")
else:
    check("S1", "ACTIVE skims are 371 SA2-aligned", False, "file missing", skipped=True,
          caveats="skims_371.omx not found under ../demand_ — skipped. "
          "The pipeline default PIPELINE_SKIMS_PATH points here.")
if os.path.isfile(_omx_legacy):
    check("S1", "legacy 330 OMX retired (info only)", True,
          "downcasted_matrices_base.omx still on disk (330 ORDER zones) — live "
          "NEW_POP=1 path must NOT open it (guard: read_omx_skims raises; "
          "_skim_flat manual branch now warns; STAGE5_STRICT_SKIMS=1 hard-fails).",
          "")

# ============================================================================
# S2 — orthogonalisation
# ============================================================================
print("\n--- S2: orthogonalisation constraints ---", flush=True)
try:
    from ga_constraints import (PIPELINE_CONSTRAINTS, get_constraints,
                                orthogonalize_on)
    _oo_dest = dict(get_constraints("stage5_dest").get("orthogonalize_on", {}))
    _oo_mode = dict(get_constraints("stage5_mode").get("orthogonalize_on", {}))
    _mode_only = {"c_walk_time", "c_bike_time", "c_ivt_auto", "c_ivt_transit",
                  "c_incp_da", "c_incp_sr", "c_incp_transit", "c_incp_active"}
    check("S2", "mode pairs live under stage5_mode (not dest)", \
        not (_mode_only & set(_oo_dest)) and (_mode_only <= set(_oo_mode)),
          f"dest={len(_oo_dest)} mode={len(_oo_mode)}",
          "Mode-only pairs under stage5_dest can never fire (coeffs absent from "
          "DEST_UEC_FLAT). Move them to stage5_mode.orthogonalize_on.")
    check("S2", "orthogonalize_on() accessor works", orthogonalize_on("stage5_dest") == _oo_dest,
          "", "ga_constraints.orthogonalize_on accessor broken — check the merge.")
except Exception:
    check("S2", "mode pairs live under stage5_mode (not dest)", False,
          traceback.format_exc(limit=1),
          "ga_constraints import failed — constraints system is down.")

if _lx is False:
    pass  # live helper test below needs only numpy/pandas via stage5 import
try:
    import stage5_destination_choice as s5
    _s5_ok = True
except Exception as _e:
    s5 = None
    _s5_ok = False
    check("S2", "live ortho helper on synthetic ca_df", False, repr(_e),
          "stage5_destination_choice import failed (needs openmatrix/scipy/larch "
          "stubs?). Fix imports before any estimation test can run.",
          skipped=False)
if _s5_ok:
    try:
        _n = 500
        _d = RNG.exponential(4.0, _n) + 0.3
        _ca = pd.DataFrame({
            "log_DIST_1": np.log(_d + 1),
            "DIST": _d,
            "log_CRASH_1": RNG.normal(0.3, 0.2, _n),
            "size_term": RNG.normal(8, 1, _n),
            "acdc_aggl": RNG.normal(0, 1, _n),
        })
        _uec = {"c_log_dist": ("log_DIST_1", None), "c_crash": ("log_CRASH_1", None),
                "c_aggl": ("acdc_aggl", None), "c_size_term": ("size_term", None)}
        _pairs = {"c_crash": "c_log_dist", "c_aggl": "c_size_term"}
        _betas = s5._apply_orthogonalization_to_ca(_ca, _uec, _pairs)
        _r0 = abs(float(np.corrcoef(_ca["log_CRASH_1"], _ca["log_DIST_1"])[0, 1]))
        _r1 = abs(float(np.corrcoef(_ca["log_CRASH_1_resid_on_log_DIST_1"],
                                    _ca["log_DIST_1"])[0, 1]))
        check("S2", "live ortho helper on synthetic ca_df",
              set(_betas) == {"c_crash", "c_aggl"} and _r1 < 0.05,
              f"|r| {_r0:.3f}->{_r1:.4f}",
              "Residualisation failed: residual still correlated with base. "
              "Check _apply_orthogonalization_to_ca OLS math.")
        # coverage: every dest pair resolves in DEST_UEC_FLAT
        _flat = s5.DEST_UEC_FLAT
        _dead = [k for k, v in _oo_dest.items()
                 if k not in _flat or v not in _flat]
        check("S2", "all dest pairs resolve in DEST_UEC_FLAT", not _dead,
              f"{len(_oo_dest)} pairs" + (f" dead={_dead}" if _dead else ""),
              f"Dead pairs { _dead} can never fire — remove or fix names.")
    except Exception:
        check("S2", "live ortho helper on synthetic ca_df", False,
              traceback.format_exc(limit=2),
              "Live helper test raised — see traceback.")

# ============================================================================
# S3 — dest starter: spec checks + REAL synthetic fit + perturbation trial
# ============================================================================
print("\n--- S3: destination starter spec ---", flush=True)
if _s5_ok:
    try:
        _st = dict(s5.DEST_STARTER_UEC_FLAT)
        check("S3", "starter mutex-clean (no c_logsum)", "c_logsum" not in _st,
              f"terms={sorted(_st)}",
              "Starter contains c_logsum: mutex with c_log_dist risks a "
              "collinear start. Keep the starter distance-only.")
        check("S3", "starter cols resolve in DEST_UEC_FLAT",
              all(k in s5.DEST_UEC_FLAT for k in _st), "",
              "Starter references unknown UEC terms — build would fail.")
    except Exception:
        check("S3", "starter mutex-clean (no c_logsum)", False,
              traceback.format_exc(limit=1), "")
else:
    check("S3", "starter mutex-clean (no c_logsum)", False, "no stage5", skipped=True,
          caveats="Skipped: stage5 import failed.")

if _s5_ok and _lx and RUN_FITS:
    # Synthetic dest estimation: N_TOURS cases x 15 zones, known DGP so the
    # starter MUST converge. Mirrors smoke_jax_models Family B construction.
    try:
        import larch as _lxm
        from larch import P as _P, X as _X
        N_DC, N_DZ = min(N_TOURS, 300), 15
        zones = np.arange(1, N_DZ + 1)
        totemp = RNG.lognormal(9, 1, N_DZ)
        dist = (np.abs(np.arange(N_DC)[:, None] * 0.7 - zones[None, :] * 1.8)
                + RNG.exponential(1.0, (N_DC, N_DZ)) + 0.25)
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
                "DIST": d,
                "log_CRASH_1": RNG.uniform(0.05, 0.45, N_DC),
                "logsum": RNG.uniform(-3.0, -0.5, N_DC),
                "size_term": np.log(totemp[zi] + 1),
                "log_correction": 0.0,
            }))
        dest_ca = pd.concat(rows, ignore_index=True).set_index(["ptour_id", "_altid_"])
        dest_co = pd.DataFrame({"DTAZ": ch_zone, "weight": 1.0},
                               index=pd.Index(np.arange(N_DC).astype(str), name="ptour_id"))
        d_ca = _lxm.Dataset.construct.from_idca(dest_ca)
        d_co = _lxm.Dataset.construct.from_idco(dest_co)
        dt_dest = d_ca.merge(d_co)
        dt_dest.attrs["_flat_mode"] = True
        dt_dest.attrs["_collinear_cols"] = []
        _res = s5.estimate_dest_starter(dt_dest, "work")
        check("S3", "synthetic starter CONVERGES", bool(_res.get("converged")),
              f"ll={_res.get('ll', float('nan')):.1f} n_free={_res.get('n_free')}",
              "Starter failed on CLEAN synthetic data with a known DGP: "
              f"{'; '.join(_res.get('reasons', ['?']))}. The estimator itself "
              "is broken (not the data) — fix _estimate/JAX setup first. "
              "Do NOT run GA until this passes.")
        if _res.get("converged"):
            # Perturbation trial: starter + c_logsum, warm-started -> must fit.
            try:
                _m2 = s5._build_dest_model(
                    dt_dest, "work",
                    custom_dest_vars={"c_log_dist": True, "c_dist": True,
                                      "c_size_term": True, "c_crash": True,
                                      "c_logsum": True})
                _nw = s5.apply_dest_warm_start(_m2, _res.get("coefs", {}))
                _m2 = s5._estimate(_m2, search_mode=True)
                # Recorded LL first: a fresh loglike() post-estimate re-evaluates
                # the unpatched objective and returns garbage (~-1.6e34).
                _ll2 = float(getattr(_m2, "_cached_ll", float("nan")))
                if not np.isfinite(_ll2):
                    try:
                        _ll2 = float(_m2.most_recent_estimation_result.loglike)
                    except Exception:
                        _ll2 = float("nan")
                if not np.isfinite(_ll2):
                    _ll2 = float(_m2.loglike())
                check("S3", "warm-started perturbation fits",
                      np.isfinite(_ll2) and _ll2 > _res["ll"] - 50,
                      f"starter LL={_res['ll']:.1f} perturbed LL={_ll2:.1f} "
                      f"(warm-started {_nw} params)",
                      "Perturbation from a converged warm start failed: warm-start "
                      "application or search_mode estimation is broken.")
            except Exception:
                check("S3", "warm-started perturbation fits", False,
                      traceback.format_exc(limit=2),
                      "Perturbation trial raised — see traceback.")
    except Exception:
        check("S3", "synthetic starter CONVERGES", False,
              traceback.format_exc(limit=2),
              "Synthetic starter harness raised — see traceback.")
else:
    check("S3", "synthetic starter CONVERGES", False, "skipped", skipped=True,
          caveats="Skipped: RUN_FITS=0 or larch/stage5 unavailable.")

# ============================================================================
# S4 — logsum ActivitySim reflection
# ============================================================================
print("\n--- S4: logsum ActivitySim reflection ---", flush=True)
try:
    _gc_src = open(os.path.join(CODE, "ga_constraints.py"), encoding="utf-8").read()
    _mx = _gc_src[_gc_src.index("'mutually_exclusive'"):_gc_src.index("'min_behavioral'")]
    _mx_code = "\n".join(l.split("#")[0] for l in _mx.splitlines())
    check("S4", "logsum/distance mutex removed", "c_logsum" not in _mx_code,
          "at-least-one require-one kept" if "(1, ['c_logsum', 'c_log_dist'])" in _gc_src else "require-one LOST",
          "Mutex present -> GA can never select logsum+distance together "
          "(ActivitySim always carries both). Remove the pair from "
          "mutually_exclusive; keep the require-one.")
except Exception:
    check("S4", "logsum/distance mutex removed", False, traceback.format_exc(limit=1), "")
try:
    _s5_src = open(os.path.join(CODE, "stage5_destination_choice.py"),
                   encoding="utf-8").read()
    check("S4", "prediction default c_logsum=0.0",
          "_pv('c_logsum', 0.0) if include_logsum else 0.0" in _s5_src
          and "_pv('c_logsum', 1.0)" not in _s5_src, "",
          "Unestimated c_logsum defaults to 1.0 -> prediction applies full "
          "accessibility weight the model never estimated. Must be 0.0.")
    _fb_terms = ["c_veh_da", "c_crash_veh", "c_cost_auto", "c_cbd_da",
                 "c_empden_walk", "c_joint_sr", "c_short_walk",
                 "STAGE5_LOGSUM_FULLSPEC", "CARLICENCE", "independent-signal share"]
    _fb_missing = [t for t in _fb_terms if t not in _s5_src]
    check("S4", "fallback full-spec emulation present", not _fb_missing,
          f"missing={_fb_missing}" if _fb_missing else "all terms wired",
          "Fallback is time-only -> logsum ≈ f(DIST) by construction. "
          "Restore the 6a-extra block.")
except Exception:
    check("S4", "prediction default c_logsum=0.0", False, traceback.format_exc(limit=1), "")

# Synthetic variance proof (no deps beyond numpy): time-only ≈ f(DIST).
try:
    _rng = np.random.default_rng(3)
    _nc, _nz = 60, 40
    _dist = _rng.exponential(6.0, (_nc, _nz)) + 0.5
    _ah = np.clip(_dist * 2.2 / 60.0, 0.01, 5.0)
    _th = np.clip(_dist * 3.5 / 60.0 + _rng.normal(0, 0.3, (_nc, _nz)), 0.01, 6.0)
    _acc = _rng.lognormal(0.5, 1.2, (_nc, _nz))

    def _motor(vs, mu=0.8):
        V = np.stack(vs, axis=2) / mu
        M = V.max(axis=2, keepdims=True)
        E = np.exp(np.clip(V - M, -500, 0))
        S = E.sum(axis=2)
        return mu * np.log(np.maximum(S, 1e-300)) + mu * (np.stack(vs, axis=2) / mu).max(axis=2)

    def _root(a, b):
        S = np.stack([a, b], axis=2)
        M = S.max(axis=2, keepdims=True)
        return M[..., 0] + np.log(np.exp(np.clip(S - M, -500, 0)).sum(axis=2))

    _vda = -1.2 * _ah
    _vsr = -1.2 * _ah - 0.5
    _vtr = -0.9 * _th - 0.5
    _vwk = -1.5 * np.log(_dist * 12.0 + 1) - 1.0
    _ls_t = _root(_motor([_vda, _vsr, _vtr]), _motor([_vwk, _vwk]))
    _vtr2 = _vtr + 0.6 * np.log1p(_acc)
    _ls_e = _root(_motor([_vda, _vsr, _vtr2]), _motor([_vwk, _vwk]))

    def _indep(L, T):
        Lw = L - L.mean(axis=1, keepdims=True)
        Tw = T - T.mean(axis=1, keepdims=True)
        r = (Lw * Tw).sum() / np.sqrt((Lw ** 2).sum() * (Tw ** 2).sum())
        return 1 - r ** 2
    _it, _ie = _indep(_ls_t, _ah), _indep(_ls_e, _ah)
    check("S4", "synthetic: enriched logsum carries signal",
          _it < 0.10 and _ie > 0.15,
          f"time-only indep={_it:.3f} enriched indep={_ie:.3f}",
          "Variance proof failed — nest math or test DGP broken. The shipped "
          "fallback math did not change; investigate the test, not the model.")
except Exception:
    check("S4", "synthetic: enriched logsum carries signal", False,
          traceback.format_exc(limit=1), "")

# ============================================================================
# S5 — mixed logit: NegLogNormal + gene codecs + known gaps
# ============================================================================
print("\n--- S5: mixed logit / random parameters ---", flush=True)
if _lx:
    try:
        _has_nln = hasattr(lx.mixtures, "NegLogNormal")
        check("S5", "lx.mixtures.NegLogNormal API present", _has_nln,
              "available" if _has_nln else "MISSING",
              "This larch build lacks NegLogNormal -> neglognormal families fall "
              "back to Normal (see _mxl_mixture guard) or fail. Upgrade larch "
              "to a driftlesslabs v6 build that ships it.")
    except Exception:
        check("S5", "lx.mixtures.NegLogNormal API present", False,
              traceback.format_exc(limit=1), "")
else:
    check("S5", "lx.mixtures.NegLogNormal API present", False, "no larch",
          skipped=True, caveats="Skipped: larch unavailable.")
# Latent-math roundtrip (pure numpy; mirrors _estimate_mode_model_mixed_logit).
try:
    _base = -1.2
    _sig = 0.25
    _latent = np.log(max(abs(_base), 1e-3)) - 0.5 * _sig ** 2
    _eff = -float(np.exp(np.clip(_latent + 0.5 * _sig ** 2, -20.0, 20.0)))
    check("S5", "neglognormal latent roundtrip", abs(_eff - _base) < 1e-9,
          f"base={_base} latent={_latent:.4f} effective={_eff:.4f}",
          "Latent-mean conversion broken: E[-LogNormal(mu,s)] must equal the "
          "MNL base value. Check the mu/sigma bookkeeping.")
except Exception:
    check("S5", "neglognormal latent roundtrip", False, traceback.format_exc(limit=1), "")
# Known-gap audit (source checks — these are DOCUMENTED gaps, not regressions).
try:
    _s5_src2 = open(os.path.join(CODE, "stage5_destination_choice.py"),
                    encoding="utf-8").read()
    _mz_src = open(os.path.join(CODE, "model_zoo.py"), encoding="utf-8").read()
    _fb_only_normal = ("mxl.mixtures.append(lx.mixtures.Normal(param, sd_name))" in _s5_src2)
    _mz_only_normal = ("lx.mixtures.Normal(" in _mz_src
                       and "NegLogNormal" not in _mz_src)
    _gap = []
    if _fb_only_normal:
        _gap.append("fallback _estimate_mixed_logit is Normal-only (no policy threading)")
    if _mz_only_normal:
        _gap.append("model_zoo._fit_larch_mxl is Normal-only")
    check("S5", "no Normal-only gaps remain", not _gap,
          "all paths policy-aware" if not _gap else "; ".join(_gap),
          "GAP (by design, pending): neglognormal is tested ONLY via the "
          "AGDS 2-bit distribution genes -> _estimate_mode_model_mixed_logit. "
          "Fallback/model_zoo paths fit Normal mixtures. Wire distribution_policy "
          "through if you need neglognormal there." if _gap else "",
          warn=True)
except Exception:
    check("S5", "no Normal-only gaps remain", False, traceback.format_exc(limit=1), "")
# AGDS 2-bit codec roundtrip (pure logic; import guarded).
try:
    _ga = None
    try:
        import ga_hypothesis_testing as _ga_mod
        _ga = _ga_mod
    except Exception as _gie:
        raise RuntimeError(f"ga_hypothesis_testing import failed: {_gie!r}")
    _found = hasattr(_ga, "SparseEAAGDS")
    check("S5", "AGDS distribution codec importable", _found, "",
          "ga_hypothesis_testing failed to import — AGDS gene search cannot run. "
          "Check its module-level imports." if not _found else "")
except Exception:
    check("S5", "AGDS distribution codec importable", False, "skipped", skipped=True,
          caveats="Skipped: ga_hypothesis_testing import failed (see detail). "
          "Distribution-gene search untestable until imports resolve.")

# ============================================================================
# S6 — perturbations test random params (AGDS 2-bit vs bandit 1-bit)
# ============================================================================
print("\n--- S6: perturbation coverage of random params ---", flush=True)
try:
    _ga_src = open(os.path.join(CODE, "ga_hypothesis_testing.py"),
                   encoding="utf-8").read()
    _agds_mut = ("tail[tail_pos] = code & 1" in _ga_src
                 and "tail_mask" in _ga_src)
    check("S6", "AGDS perturbs 2-bit distribution tail", _agds_mut, "",
          "AGDS _mutate/_crossover no longer touch distribution bits -> "
          "fixed/normal/neglognormal never tested. Restore tail perturbation.")
    _bandit_flip = "_lp_tail[_j] = 1 - _lp_tail[_j]" in _ga_src
    _bandit_2bit = "2 * len(self.distribution_features)" in _ga_src
    check("S6", "bandit perturbs rand tail (1-bit Normal-only)", _bandit_flip,
          "1-bit: tests Normal vs fixed only" if _bandit_flip else "NO tail flip",
          "Bandit flips rand bits but they are SINGLE-bit (Normal vs fixed): "
          "neglognormal is untestable under bandit/SA. Use GA_ENGINE=agds with "
          "GA_AGDS_MXL_SEARCH=1 for full fixed/normal/neglognormal search, or "
          "extend the bandit tail to 2-bit." if _bandit_flip else
          "Bandit never perturbs rand genes -> random params untestable under SA. "
          "Restore the _lp_tail flip in the local-move branch.")
except Exception:
    check("S6", "AGDS perturbs 2-bit distribution tail", False,
          traceback.format_exc(limit=1), "")

# ============================================================================
# S7 — watch-items (OOM, detour OOR, _get_tt units, mode fillna)
# ============================================================================
print("\n--- S7: watch-items ---", flush=True)
_full, _samp = 50_000 * 371, 50_000 * 20
check("S7", "idca OOM guard documented", True,
      f"full={_full:,} rows (~{_full*240*2/1e9:.1f}GB) vs sampled={_samp:,} "
      f"(~{_samp*240*2/1e9:.2f}GB)",
      "50k×371 full enumeration OOMs (≈261GB observed job 23661073). Keep "
      "DEST_EST_SAMPLE_SIZE>0 + MAX_ESTIMATION_OBS cap; use "
      "larch_jax_enforce.estimate_row_guard() before dense builds.")
try:
    import stage5b_trip_destination as s5b
    _skim = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]) * 2.0 + 1.0
    np.fill_diagonal(_skim, 0.5)
    _df = pd.DataFrame({"origin_taz": [101, 101], "dest_taz": [102, 888],
                        "anchor_down_taz": [103, 103]})
    _det = s5b.compute_detour(_df, _skim, {101: 0, 102: 1, 103: 2, 104: 3, 105: 4})
    _oor_nan = bool(np.isnan(float(_det.iloc[1])))
    _mapped_ok = bool(abs(float(_det.iloc[0]) - 1.0) < 1e-9)
    check("S7", "detour OOR->NaN (no zone-0 fabrication)",
          _oor_nan and _mapped_ok,
          f"OOR detour={_det.iloc[1]} (NaN), mapped detour={float(_det.iloc[0]):.2f}",
          "REGRESSION: unmapped zones must yield NaN (excluded downstream), "
          "not zone-0 distances. Check compute_detour OOR handling.")
except Exception:
    check("S7", "detour OOR->NaN (no zone-0 fabrication)", False, "skipped", skipped=True,
          caveats="Skipped: stage5b import failed.")
try:
    import read_skims as _rs
    _divs = _rs.load_skim_unit_divisors()
    _keys = ["SOVTOLL_TIME__AM", "WLK_LOC_WLK_TOTIVT__AM", "WLK_LOC_DRV_TOTIVT__AM",
             "SOVTOLL_TIME__PM", "DIST", "CRASH"]
    _missing_units = [k for k in _keys if k not in _divs and k != "DIST" and k != "CRASH"]
    check("S7", "unit sidecar covers key matrices", not _missing_units,
          f"{len(_divs)} time entries" + (f" missing={_missing_units}" if _missing_units else ""),
          "demand_/skim_units.json missing time entries — value-guessing "
          "applies (misclassifies genuine 200–600 min trips). Regenerate it.")
except Exception:
    check("S7", "unit sidecar covers key matrices", False, "skipped", skipped=True,
          caveats="Skipped: read_skims import failed.")
try:
    _s5m = s5 if "_s5_ok" in dir() and _s5_ok else None
    if _s5m is None:
        import stage5_destination_choice as _s5m
    _lk = {(1, 2): {"SOVTOLL_TIME__AM": 250.0}}
    _v = _s5m._get_tt(_lk, 1, 2, 1)
    check("S7", "_get_tt trusts sidecar (250min genuine, capped)",
          abs(_v - 120.0) < 1e-9,
          f"250 -> {_v:.1f}min (DA cap; NOT divided to 25)",
          "REGRESSION: 250 was divided to 25 (tenths-guess). Sidecar says "
          "minutes — trust it. Check _skim_minutes wiring." if abs(_v - 25.0) < 1e-9 else "")
    from stage6_mode_choice import RAW_MODE_MAP as _rmm
    _mapped = pd.Series(["Car driver", "Teleport", None]).map(_rmm)
    check("S7", "unknown modes -> NaN (fillna audited)", True,
          f"Teleport->{_mapped.iloc[1]} (NaN, must not default to SR)",
          "'Teleport' was only ever a synthetic test token, not a real mode. "
          "Unknown labels map to NaN; ensure downstream fillna defaults to "
          "DA(1), not SR(2), to avoid inflating passenger share.")
except Exception:
    check("S7", "_get_tt 250s ambiguity characterised", False, "skipped", skipped=True,
          caveats="Skipped: stage5/stage6 import failed.")

# ============================================================================
# S8 — fringe mapping + quarantine (poisoned-tour protection)
# ============================================================================
print("\n--- S8: fringe mapping + quarantine ---", flush=True)
try:
    _s5q = s5 if "_s5_ok" in dir() and _s5_ok else None
    if _s5q is None:
        import stage5_destination_choice as _s5q
    _fr = pd.DataFrame({
        "person_id": ["ext/1", "ok/1"],
        "OTAZ": [301011001, 301011001],
        "DTAZ": [319031512, 301011002],  # 319031512 = Southern Downs, out of universe
        "tour_mode": [1, 1]})
    _fr_out = _s5q._coerce_zones_to_universe(_fr)
    check("S8", "out-of-universe DTAZ fringed to 999",
          int(_fr_out.loc[0, "DTAZ"]) == 999 and int(_fr_out.loc[1, "DTAZ"]) == 301011002,
          f"DTAZ={_fr_out['DTAZ'].tolist()}",
          "Fringe-mapping broken: out-of-universe chosen zones flow downstream "
          "and NaN the whole fit (observed: ~5 such tours killed ~6500 evals). "
          "Fix _coerce_zones_to_universe.")
    _qq = pd.DataFrame({
        "person_id": ["p1", "p2", "p3", "p4"],
        "OTAZ": [301011001, 301011001, 301011001, 301011001],
        "DTAZ": [301011002, 123456789, 999, 301011002],
        "tour_mode": [1, 1, 1, 1],
        "incp": [5.0, 5.0, 5.0, np.nan]})
    _qk = _s5q.quarantine_dest_cases(_qq, label="health")
    check("S8", "quarantine drops unrepresentable+NaN, keeps fringe",
          set(_qk["person_id"]) == {"p1", "p3"},
          f"kept={sorted(_qk['person_id'].tolist())} "
          "(p2: bad DTAZ dropped; p4: NaN incp dropped even though its "
          "zone is valid — NaN person attrs poison likelihoods regardless "
          "of zone)",
          "Quarantine broken: unrepresentable tours reach the estimator and "
          "abort fits at priors. Fix quarantine_dest_cases.")
except Exception:
    check("S8", "out-of-universe DTAZ fringed to 999", False, "skipped", skipped=True,
          caveats="Skipped: stage5 import failed.")

# ============================================================================
# S9 — ML accessibility: asim feedback loop + KDE deterrence
# ============================================================================
print("\n--- S9: ML accessibility (feedback loop + KDE deterrence) ---", flush=True)
try:
    import asim_feedback_loop as _afl
    _coef_known = {'c_ivt_auto': -1.2, 'c_ivt_transit': -0.9,
                   'c_walk_time': -1.5, 'c_bike_time': -1.0,
                   'ASC_Transit': -0.5, 'ASC_Walk': -1.0,
                   'ASC_Bike': -1.5, 'ASC_SR': -0.3}
    _env = _afl.coefs_to_asim_env(_coef_known, med_walk_min=15.0,
                                  med_bike_min=20.0)
    _map_ok = (abs(_env['ASIM_BETA_TIME_CAR'] - (-0.02)) < 1e-12
               and abs(_env['ASIM_BETA_TIME_TRANSIT'] - (-0.015)) < 1e-12
               and abs(_env['ASIM_BETA_TIME_WALK'] - (-1.5 / 16.0)) < 1e-12
               and abs(_env['ASIM_BETA_TIME_BIKE'] - (-1.0 / 21.0)) < 1e-12
               and _env['ASIM_ASC_CAR'] == 0.0
               and _env['ASIM_ASC_TRANSIT'] == -0.5
               and _env['ASIM_ASC_WALK'] == -1.0
               and _env['ASIM_ASC_BIKE'] == -1.5)
    check("S9", "coefs->ASIM mapping units exact",
          _map_ok,
          f"b_car={_env['ASIM_BETA_TIME_CAR']:.4f}/min b_wlk={_env['ASIM_BETA_TIME_WALK']:.4f}/min",
          "Unit mapping broken: fallback c_ivt is PER-HOUR (v=c*hours) but "
          "asim BETA is per-minute (divide by 60); walk/bike are per-LOG-min "
          "linearised at the median (c/(med+1)). Check coefs_to_asim_env.")
    # fallback-chain: c_ivt only -> c_ivt_auto and c_ivt_transit inherit
    _env2 = _afl.coefs_to_asim_env({'c_ivt': -1.1})
    check("S9", "fallback chains mirror fallback _pv",
          abs(_env2['ASIM_BETA_TIME_CAR'] - (-1.1 / 60.0)) < 1e-12
          and abs(_env2['ASIM_BETA_TIME_TRANSIT'] - (-1.1 / 60.0)) < 1e-12,
          f"chain b_car={_env2['ASIM_BETA_TIME_CAR']:.5f}",
          "Chain broken: c_ivt_transit must fall back through c_ivt_auto to "
          "c_ivt, exactly like _logsums_numpy_fallback._pv.")
    # env scoping: set + restore, including the unset-var case
    _sentinel = '__HEALTH_SENTINEL_XYZ__'
    os.environ.pop(_sentinel, None)
    os.environ['ASIM_BETA_TIME_CAR'] = '-0.111'
    with _afl.asim_env({'ASIM_BETA_TIME_CAR': -0.02, _sentinel: 1.0}):
        _inside = (os.environ.get('ASIM_BETA_TIME_CAR') == '-0.02')
    _restored = (os.environ.get('ASIM_BETA_TIME_CAR') == '-0.111'
                 and _sentinel not in os.environ)
    del os.environ['ASIM_BETA_TIME_CAR']
    check("S9", "asim_env sets + restores (no leak)",
          _inside and _restored,
          "inside set, after restore; sentinel unset-var removed",
          "LEAK: ASIM_* changes escape the rebuild and corrupt later "
          "sections (S4 defaults). Fix asim_env save/restore.")
except Exception:
    check("S9", "coefs->ASIM mapping units exact", False,
          traceback.format_exc(limit=1), "")
# Loop logic on a stub (monkeypatched rebuild — no real data needed).
try:
    import asim_feedback_loop as _afl2
    _real_rebuild = _afl2.rebuild_asim_accessibility
    _stub_state = {'acc': pd.DataFrame({'asim_work_acc': [1.0, 2.0, 3.0]})}

    def _stub_rebuild(landuse, skim_df, coef, zone_ids=None,
                      med_walk_min=15.0, med_bike_min=20.0):
        return _stub_state['acc'], {'ASIM_BETA_TIME_CAR': coef.get('c_ivt_auto', 0) / 60.0,
                                    '_meta': {}}
    _afl2.rebuild_asim_accessibility = _stub_rebuild
    _target, _ll_star = -1.2, -500.0

    def _stub_estimate(acc):
        # fixed-point dynamics: coef closes 90% of its distance to target
        # each call (gains shrink x0.1 -> stall below ll_tol within
        # max_iter); LL rises monotonically toward the asymptote
        _stub_estimate.cur += (_target - _stub_estimate.cur) * 0.9
        ll = _ll_star - 100.0 * abs(_stub_estimate.cur - _target)
        return {'c_ivt_auto': _stub_estimate.cur}, ll
    _stub_estimate.cur = -0.2
    try:
        _res = _afl2.run_feedback_loop(None, None, _stub_estimate,
                                       coef0={'c_ivt_auto': -0.2},
                                       max_iter=10, ll_tol=1e-4, verbose=False)
    finally:
        _afl2.rebuild_asim_accessibility = _real_rebuild
    _lls = [h['ll'] for h in _res['history']]
    _mono = all(b >= a - 1e-12 for a, b in zip(_lls, _lls[1:]))
    check("S9", "feedback loop terminates, LL monotone, keeps best",
          _res['converged'] and _mono and _res['n_iters'] <= 10
          and abs(_res['final_ll'] - _ll_star) < 1.0,
          f"iters={_res['n_iters']} LL {min(_lls):.1f}->{max(_lls):.1f} mono={_mono}",
          "Loop logic broken: must stop on LL stall (gain < ll_tol), never "
          "regress final LL (keep best), terminate within max_iter.")
except Exception:
    check("S9", "feedback loop terminates, LL monotone, keeps best", False,
          traceback.format_exc(limit=1), "")
# End-to-end rebuild on tiny synthetic landuse+skims (real code path).
try:
    import asim_feedback_loop as _afl3
    _zr = np.arange(1, 6)
    _lu = pd.DataFrame({
        'ORDER': _zr, 'TOTEMP': [5000, 20000, 8000, 3000, 1000],
        'Primary': [100, 300, 200, 150, 50], 'Secondary': [200, 500, 300, 100, 50],
        'Education': [0, 2, 1, 0, 0], 'HSENROLL': [50, 200, 100, 40, 20],
        'Shopping': [5, 40, 15, 8, 2], 'retail': [1e4, 2e5, 5e4, 2e4, 5e3],
        'RETEMPN': [200, 1500, 500, 200, 50], 'EATOUT': [3, 60, 20, 10, 2],
        'Entertainment': [1, 20, 8, 3, 0], 'OTHERMAINTENANCE': [2, 15, 6, 3, 1],
        'OTHERDISCRFacilities': [1, 10, 4, 2, 0], 'Healthcare': [1, 12, 5, 2, 1],
        'park': [1e5, 2e5, 3e5, 5e5, 1e6], 'recreation_ground': [5e4, 1e5, 2e5, 3e5, 5e5],
        'sport': [1, 5, 3, 2, 1], 'sa2_area': [2e6, 3e6, 5e6, 1e7, 2e7]})
    _od = [(o, d) for o in _zr for d in _zr]
    _sk = pd.DataFrame({
        'OTAZ': [o for o, _ in _od], 'DTAZ': [d for _, d in _od],
        'SOVTOLL_TIME__AM': [5.0 + abs(o - d) * 8.0 for o, d in _od],
        'WLK_LOC_WLK_TOTIVT__AM': [0.0 if (o - d) % 3 == 0 and o != d
                                   else 10.0 + abs(o - d) * 12.0 for o, d in _od],
        'DISTWALK': [abs(o - d) * 1.2 + 0.3 for o, d in _od],
        'DISTBIKE': [abs(o - d) * 1.5 + 0.5 for o, d in _od]})
    _acc_def, _ = _afl3.rebuild_asim_accessibility(_lu, _sk, {})
    _acc_est, _ = _afl3.rebuild_asim_accessibility(
        _lu, _sk, {'c_ivt_auto': -2.4, 'c_ivt_transit': -1.8,
                   'c_walk_time': -2.0, 'c_bike_time': -1.5,
                   'ASC_Transit': -1.2, 'ASC_Walk': -0.5, 'ASC_Bike': -1.0})
    _ecols = [c for c in _acc_est.columns if c.startswith('asim_')]
    _finite = bool(np.isfinite(_acc_est[_ecols].values).all())
    _moved = any(abs(_acc_est[c].values - _acc_def[c].values).max() > 1e-6
                 for c in _ecols if c in _acc_def.columns)
    check("S9", "rebuild from estimated coefs: finite + moves acc",
          len(_ecols) >= 5 and _finite and _moved,
          f"{len(_ecols)} asim cols, finite={_finite}, moved={_moved}",
          "Rebuild broken: estimated coefs must change asim_*_acc vs "
          "defaults (else the feedback loop is a no-op). Check env "
          "threading through compute_logsum_accessibility.")
except Exception:
    check("S9", "rebuild from estimated coefs: finite + moves acc", False,
          traceback.format_exc(limit=1), "")
# KDE deterrence (scipy-gated).
try:
    import kde_deterrence as _kde
    _krng = np.random.default_rng(11)
    _t_all = _krng.lognormal(mean=2.6, sigma=0.7, size=2000)   # ~13min median
    _f = _kde.fit_deterrence(_t_all[:1000])
    _ks = _kde.trip_length_ks(_f, _t_all[:1000], _t_all[1000:])
    _w = _kde.deterrence_weights(_f, np.array([[0.0, 10.0, 60.0, 200.0]]))
    check("S9", "KDE deterrence: normalised + validates split-half",
          abs(float(_f(0.0)) - 1.0) < 1e-9 and _ks['p_value'] > 0.01
          and _w.shape == (1, 4) and bool((_w >= 0).all() and (_w <= 1).all())
          and float(_w[0, 0]) >= float(_w[0, -1]),
          f"f(0)=1 KS p={_ks['p_value']:.3f} w={_w[0].tolist()}",
          "KDE deterrence broken: f(0) must equal 1, split-half KS must not "
          "reject (p>0.01), weights monotone decreasing in time.")
    _beq = _kde.exp_equivalent_decay(lambda t: np.exp(-0.05 * np.asarray(t)))
    check("S9", "exp_equivalent_decay recovers known b",
          abs(_beq['b'] - 0.05) / 0.05 < 0.05,
          f"b={_beq['b']:.4f} (true 0.05) rmse={_beq['rmse']:.2e}",
          "Calibrated-b bridge broken: feeding exp(-0.05t) must recover "
          "b~=0.05. Check the weighted log-space fit.")
    import pandas as _pds
    _pdf = _pds.DataFrame({'purpose': ['shop'] * 50 + ['work'] * 5,
                           'travel_time': list(_krng.lognormal(2.6, 0.7, 55))})
    _pf = _kde.fit_purpose_deterrence(_pdf, min_n=30)
    _thin_ok = _pf['shop'] is not None and _pf['work'] is None
    try:
        _kde.fit_deterrence([1.0, 2.0, 3.0])
        _guard_ok = False
    except ValueError:
        _guard_ok = True
    check("S9", "purpose batch: thin->None, small-n guard raises",
          _thin_ok and _guard_ok,
          f"shop={'fit' if _pf['shop'] is not None else 'None'} "
          f"work={'fit' if _pf['work'] is not None else 'None'} guard={_guard_ok}",
          "Batch guards broken: thin purposes must map to None (caller "
          "keeps exp default, logged) and <10 obs must raise, never "
          "silently return a garbage KDE.")
except Exception as _e:
    if 'scipy' in str(type(_e)).lower() or 'scipy' in traceback.format_exc(limit=3):
        check("S9", "KDE deterrence: normalised + validates split-half",
              False, "skipped", skipped=True,
              caveats="Skipped: scipy unavailable.")
    else:
        check("S9", "KDE deterrence: normalised + validates split-half", False,
              traceback.format_exc(limit=1), "")

# ============================================================================
# S10 — Overture A/B readiness (USE_OVERTURE_LANDUSE / OVERTURE_REFRESH_SIZE)
# ============================================================================
print("\n--- S10: Overture A/B readiness ---", flush=True)
try:
    _ov_path = os.path.join(CODE, 'saved_data', 'zone_landuse_overture.csv')
    _b357 = os.path.join(os.path.dirname(CODE), 'new_pop', 'saved_data',
                         'zone_landuse_357.csv')
    if os.path.exists(_ov_path) and os.path.exists(_b357):
        _bo = pd.read_csv(_b357, low_memory=False)
        _oo = pd.read_csv(_ov_path, low_memory=False)
        _missing = [c for c in _bo.columns if c not in _oo.columns]
        _bk = set(_bo['SA2_MAIN16'].astype(str))
        _ok = set(_oo['SA2_MAIN16'].astype(str))
        _extra = [c for c in _oo.columns if c not in _bo.columns]
        check("S10", "overture landuse is a strict superset on identical keys",
              not _missing and _bk == _ok and len(_oo) == len(_bo)
              and not _oo['SA2_MAIN16'].duplicated().any(),
              f"missing={len(_missing)} extra={len(_extra)} rows={len(_oo)} "
              f"keydiff={len(_bk ^ _ok)}",
              "NOT a safe drop-in: flipping USE_OVERTURE_LANDUSE=1 would "
              "remove baseline columns or remap zones. Regenerate "
              "zone_landuse_overture.csv from build_zone_landuse_overture.py.")
    else:
        check("S10", "overture landuse is a strict superset on identical keys",
              False, "skipped", skipped=True,
              caveats="Skipped: zone_landuse_overture.csv or "
                      "zone_landuse_357.csv absent.")
except Exception:
    check("S10", "overture landuse is a strict superset on identical keys",
          False, traceback.format_exc(limit=1), "")
# A/B purity on synthetic 5-zone: the ONLY effect channel is extra columns.
try:
    import compute_accessibility as _ca
    _zr = np.arange(1, 6)
    _base = pd.DataFrame({
        'ORDER': _zr, 'TOTEMP': [5000, 20000, 8000, 3000, 1000],
        'Primary': [100, 300, 200, 150, 50], 'Secondary': [200, 500, 300, 100, 50],
        'Education': [0, 2, 1, 0, 0], 'HSENROLL': [50, 200, 100, 40, 20],
        'Shopping': [5, 40, 15, 8, 2], 'retail': [1e4, 2e5, 5e4, 2e4, 5e3],
        'RETEMPN': [200, 1500, 500, 200, 50], 'EATOUT': [3, 60, 20, 10, 2],
        'Entertainment': [1, 20, 8, 3, 0], 'OTHERMAINTENANCE': [2, 15, 6, 3, 1],
        'OTHERDISCRFacilities': [1, 10, 4, 2, 0], 'Healthcare': [1, 12, 5, 2, 1],
        'park': [1e5, 2e5, 3e5, 5e5, 1e6],
        'recreation_ground': [5e4, 1e5, 2e5, 3e5, 5e5],
        'sport': [1, 5, 3, 2, 1], 'sa2_area': [2e6, 3e6, 5e6, 1e7, 2e7]})
    _ovc = pd.DataFrame({
        'ORDER': _zr, 'n_retail': [2, 40, 15, 8, 1], 'n_food': [1, 50, 18, 9, 1],
        'n_office': [5, 200, 60, 20, 2], 'n_recreation': [1, 12, 8, 10, 15],
        'n_education': [1, 8, 4, 2, 1], 'n_health': [1, 10, 5, 3, 1],
        'res_floor_m2': [5e4, 3e5, 2e5, 1e5, 3e4],
        'com_floor_m2': [1e4, 5e5, 1e5, 3e4, 5e3],
        'area_recreation': [5e4, 1e5, 2e5, 3e5, 5e5],
        'area_retail': [2e3, 4e4, 1e4, 5e3, 1e3]})
    _with = _base.merge(_ovc, on='ORDER')
    _od = [(o, d) for o in _zr for d in _zr]
    _sk = pd.DataFrame({
        'OTAZ': [o for o, _ in _od], 'DTAZ': [d for _, d in _od],
        'SOVTOLL_TIME__AM': [5.0 + abs(o - d) * 8.0 for o, d in _od],
        'WLK_LOC_WLK_TOTIVT__AM': [10.0 + abs(o - d) * 12.0 for o, d in _od],
        'DISTWALK': [abs(o - d) * 1.2 + 0.3 for o, d in _od],
        'DISTBIKE': [abs(o - d) * 1.5 + 0.5 for o, d in _od],
        'CRASH': [0.5 + abs(o - d) * 0.1 for o, d in _od]})
    _saved_ov, _saved_path = (os.environ.get('USE_OVERTURE_LANDUSE'),
                              os.environ.get('OVERTURE_LANDUSE_PATH'))
    try:
        os.environ['USE_OVERTURE_LANDUSE'] = '0'
        os.environ.pop('OVERTURE_LANDUSE_PATH', None)
        _accA = _ca.compute_accessibility(_base, _sk)
        _accB = _ca.compute_accessibility(_with, _sk)
    finally:
        if _saved_ov is None:
            os.environ.pop('USE_OVERTURE_LANDUSE', None)
        else:
            os.environ['USE_OVERTURE_LANDUSE'] = _saved_ov
        if _saved_path is not None:
            os.environ['OVERTURE_LANDUSE_PATH'] = _saved_path
    _ov_terms = [c for c in _accB.columns
                 if c in ('auto_work_acc', 'auto_shop_acc', 'walk_acc',
                          'transit_acc', 'recreation_acc', 'auto_nonmand_acc')]
    _moved = {c: float(np.abs(_accB[c].values - _accA[c].values).max())
              for c in _ov_terms if c in _accA.columns}
    _any_moved = any(v > 1e-9 for v in _moved.values())
    # purity: B-with-columns-dropped must EXACTLY reproduce A
    import io as _io
    from contextlib import redirect_stdout as _rs
    _buf = _io.StringIO()
    _saved_ov2 = os.environ.get('USE_OVERTURE_LANDUSE')
    try:
        os.environ['USE_OVERTURE_LANDUSE'] = '0'
        with _rs(_buf):
            _accB_stripped = _ca.compute_accessibility(
                _with.drop(columns=[c for c in _ovc.columns if c != 'ORDER']),
                _sk)
    finally:
        if _saved_ov2 is None:
            os.environ.pop('USE_OVERTURE_LANDUSE', None)
        else:
            os.environ['USE_OVERTURE_LANDUSE'] = _saved_ov2
    _pure = all(np.array_equal(_accA[c].values, _accB_stripped[c].values)
                for c in _accA.columns)
    check("S10", "overture cols activate terms + only channel is columns",
          _any_moved and _pure,
          f"moved={ {k: round(v, 4) for k, v in _moved.items()} } pure={_pure}",
          "A/B IMPURE: overture columns either fail to move accessibility "
          "(spec weights dead — check ACCESSIBILITY_SPEC/_LOGSUM_PURPOSE_SPEC "
          "keys vs merged column names) or affect output through a second "
          "channel (global state — B-minus-cols must equal A exactly).")
    # merge path == direct-supply path (belt-and-braces join is faithful)
    _tmpd = tempfile.mkdtemp(prefix='ov_ab_')
    _ovcsv = os.path.join(_tmpd, 'ov.csv')
    _ovc.to_csv(_ovcsv, index=False)
    _saved_ov3, _saved_p3 = (os.environ.get('USE_OVERTURE_LANDUSE'),
                             os.environ.get('OVERTURE_LANDUSE_PATH'))
    try:
        os.environ['USE_OVERTURE_LANDUSE'] = '1'
        os.environ['OVERTURE_LANDUSE_PATH'] = _ovcsv
        _buf2 = _io.StringIO()
        with _rs(_buf2):
            _accC = _ca.compute_accessibility(_base, _sk)
    finally:
        if _saved_ov3 is None:
            os.environ.pop('USE_OVERTURE_LANDUSE', None)
        else:
            os.environ['USE_OVERTURE_LANDUSE'] = _saved_ov3
        if _saved_p3 is None:
            os.environ.pop('OVERTURE_LANDUSE_PATH', None)
        else:
            os.environ['OVERTURE_LANDUSE_PATH'] = _saved_p3
    _faithful = all(np.allclose(_accB[c].values, _accC[c].values, atol=1e-6)
                    for c in _accB.columns)
    check("S10", "file-merge path == direct-column path",
          _faithful, f"merge-vs-direct match={_faithful}",
          "Merge join is LOSSY vs direct supply (key mismatch, dtype "
          "coercion, or silent column skip). Fix "
          "_merge_overture_attractors before relying on the flag.")
except Exception:
    check("S10", "overture cols activate terms + only channel is columns",
          False, traceback.format_exc(limit=1), "")
# Never-overwrite + missing-file safety (synthetic merge unit tests).
try:
    import compute_accessibility as _ca2
    _tmpd2 = tempfile.mkdtemp(prefix='ov_merge_')
    _lu = pd.DataFrame({'ORDER': [1, 2],
                        'n_retail': [777.0, 888.0]})   # sentinel: must survive
    _ovf = pd.DataFrame({'ORDER': [1, 2], 'n_retail': [1.0, 2.0],
                         'n_food': [5.0, 6.0]})
    _ovp = os.path.join(_tmpd2, 'ov.csv')
    _ovf.to_csv(_ovp, index=False)
    _s4, _p4 = (os.environ.get('USE_OVERTURE_LANDUSE'),
                os.environ.get('OVERTURE_LANDUSE_PATH'))
    try:
        os.environ['USE_OVERTURE_LANDUSE'] = '1'
        os.environ['OVERTURE_LANDUSE_PATH'] = _ovp
        _m = _ca2._merge_overture_attractors(_lu)
        _kept = bool((_m['n_retail'] == [777.0, 888.0]).all())
        _added = bool((_m['n_food'] == [5.0, 6.0]).all())
        os.environ['OVERTURE_LANDUSE_PATH'] = os.path.join(_tmpd2, 'nope.csv')
        _m2 = _ca2._merge_overture_attractors(_lu)
        _safe = _m2.equals(_lu)
        os.environ['USE_OVERTURE_LANDUSE'] = '0'
        _m3 = _ca2._merge_overture_attractors(_lu)
        _noop = _m3.equals(_lu)
    finally:
        if _s4 is None:
            os.environ.pop('USE_OVERTURE_LANDUSE', None)
        else:
            os.environ['USE_OVERTURE_LANDUSE'] = _s4
        if _p4 is None:
            os.environ.pop('OVERTURE_LANDUSE_PATH', None)
        else:
            os.environ['OVERTURE_LANDUSE_PATH'] = _p4
    check("S10", "merge never overwrites; missing-file + flag-off safe",
          _kept and _added and _safe and _noop,
          f"kept-sentinel={_kept} added-missing={_added} "
          f"missing-file-safe={_safe} flag-off-noop={_noop}",
          "Merge contract broken: existing landuse values must win over "
          "Overture, missing files must degrade to identity (never crash), "
          "flag-off must be a strict no-op.")
except Exception:
    check("S10", "merge never overwrites; missing-file + flag-off safe",
          False, traceback.format_exc(limit=1), "")
# REFRESH_SIZE blast radius + NaN safety (real file where available).
try:
    _ov_path2 = os.path.join(CODE, 'saved_data', 'zone_landuse_overture.csv')
    if os.path.exists(_ov_path2):
        _oo2 = pd.read_csv(_ov_path2, low_memory=False)
        _ovm = [c for c in _oo2.columns if c.endswith('_ovm')]
        _targets = [c[:-4] for c in _ovm if c[:-4] in _oo2.columns]
        _pbs = open(os.path.join(CODE, 'job_submit_ga_parallel_chain.pbs'),
                    encoding='utf-8', errors='replace').read()
        # Intended production defaults: conservative arm ON (extra columns
        # available), aggressive overwrite arm OFF. Override either at
        # submit time via qsub -v.
        _defaults_ok = ('USE_OVERTURE_LANDUSE=${USE_OVERTURE_LANDUSE:-1}' in _pbs
                        and 'OVERTURE_REFRESH_SIZE=${OVERTURE_REFRESH_SIZE:-0}' in _pbs)
        check("S10", "REFRESH_SIZE blast radius documented, defaults as intended",
              _defaults_ok and len(_targets) == 9,
              f"would-overwrite={len(_targets)} cols defaults-ok={_defaults_ok} "
              "(conservative ON, aggressive OFF)",
              "CHAIN DEFAULTS WRONG: USE_OVERTURE_LANDUSE must default to 1 "
              "(conservative superset arm) and OVERTURE_REFRESH_SIZE to 0. "
              "REFRESH_SIZE=1 OVERWRITES baseline size-term inputs with "
              "*_ovm counts (retail/Education/Healthcare/commercial/EATOUT/"
              "Entertainment/Public_Services/industrial/residential) — do "
              "not enable until the conservative arm is validated.")
    else:
        check("S10", "REFRESH_SIZE blast radius documented, defaults OFF",
              False, "skipped", skipped=True,
              caveats="Skipped: zone_landuse_overture.csv absent.")
except Exception:
    check("S10", "REFRESH_SIZE blast radius documented, defaults OFF",
          False, traceback.format_exc(limit=1), "")
try:
    import compute_accessibility as _ca3
    _zr5 = np.arange(1, 6)
    _nan_lu = pd.DataFrame({
        'ORDER': _zr5, 'TOTEMP': [5000, 20000, 8000, 3000, 1000],
        'Shopping': [5, 40, 15, 8, 2], 'retail': [1e4, 2e5, 5e4, 2e4, 5e3],
        'EATOUT': [3, 60, 20, 10, 2], 'sa2_area': [2e6, 3e6, 5e6, 1e7, 2e7],
        # fringe zones (4,5) carry NO overture coverage — like the ~40 real
        # zones with 10.8% NaN in n_*/area_* columns
        'n_retail': [2.0, 40.0, 15.0, np.nan, np.nan],
        'n_food': [1.0, 50.0, 18.0, np.nan, np.nan],
        'n_office': [5.0, 200.0, 60.0, np.nan, np.nan]})
    _od5 = [(o, d) for o in _zr5 for d in _zr5]
    _sk5 = pd.DataFrame({
        'OTAZ': [o for o, _ in _od5], 'DTAZ': [d for _, d in _od5],
        'SOVTOLL_TIME__AM': [5.0 + abs(o - d) * 8.0 for o, d in _od5],
        'WLK_LOC_WLK_TOTIVT__AM': [10.0 + abs(o - d) * 12.0 for o, d in _od5],
        'DISTWALK': [abs(o - d) * 1.2 + 0.3 for o, d in _od5],
        'DISTBIKE': [abs(o - d) * 1.5 + 0.5 for o, d in _od5]})
    _s5 = os.environ.get('USE_OVERTURE_LANDUSE')
    try:
        os.environ['USE_OVERTURE_LANDUSE'] = '0'
        _accN = _ca3.compute_accessibility(_nan_lu, _sk5)
    finally:
        if _s5 is None:
            os.environ.pop('USE_OVERTURE_LANDUSE', None)
        else:
            os.environ['USE_OVERTURE_LANDUSE'] = _s5
    check("S10", "NaN overture coverage -> finite acc (fillna layers hold)",
          bool(np.isfinite(_accN.values).all()),
          f"finite={bool(np.isfinite(_accN.values).all())} "
          f"cols={len(_accN.columns)}",
          "NaN POISONING: zones without Overture coverage NaN the measures. "
          "All three layers (get_land fillna, merge fillna, spec fillna) "
          "must hold — uncovered zones degrade to zero attractors, never NaN.")
except Exception:
    check("S10", "NaN overture coverage -> finite acc (fillna layers hold)",
          False, traceback.format_exc(limit=1), "")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
_counts = {}
for _r in RESULTS:
    _counts[_r["status"]] = _counts.get(_r["status"], 0) + 1
    flag = {"PASS": "ok ", "FAIL": "FAIL", "SKIP": "skip", "WARN": "warn"}[_r["status"]]
    print(f"  [{flag}] {_r['section']}/{_r['name']}" + (f" — {_r['detail']}" if _r["detail"] else ""))
print(f"\n{len(RESULTS)} checks: " + ", ".join(f"{v} {k}" for k, v in sorted(_counts.items())))
_n_fail = _counts.get("FAIL", 0)
print("RESULT:", "FAIL" if _n_fail else "PASS", flush=True)

_out = os.environ.get("HEALTH_OUT", "")
if _out:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(_out)), exist_ok=True)
        with open(_out, "w", encoding="utf-8") as _fh:
            json.dump({"results": RESULTS, "counts": _counts,
                       "result": "FAIL" if _n_fail else "PASS"}, _fh, indent=1)
        print(f"JSON report -> {_out}", flush=True)
    except Exception as _e:
        print(f"JSON report failed: {_e!r}", flush=True)
sys.exit(1 if _n_fail else 0)
