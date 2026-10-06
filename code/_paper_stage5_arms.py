"""Four-arm Stage-5 destination comparison for the paper.

Arms (same sampled cases/alternatives, same utility spec; the latent arms
share the same membership spec over BEHAVIORAL person types):

  A  larch MNL (pooled) ......... _build_dest_model + _estimate
  B  SearchLibrium MNL (pooled) .. ChoiceSetFrame + _fit_frame_mnl (exact)
  C  larch latent class .......... lx.LatentClass (larch_lc_compat) with
                                   jittered SLSQP restarts
  D  SearchLibrium latent class .. estimate_dest_choice_lc (EM + perturbations)

Arm C resolves the three larch 6.0.46 defects found by instrumenting the
reflow path: (1) larch_lc_compat patches _GenericFlow.n_cases so the
folded-then-dissolved class dataset reports n_cases from the group dim;
(2) from_idca(fill_missing=0) replaces NaN unsampled cells so the JAX
backward pass stops producing NaN gradients (0 * NaN); (3) alternative
codes must not include 0 (NestingTree root_id=0 absorbs it).  The manual
EM remains as a fallback (PAPER_LC_ENGINE=manual).  All arms fit identical
rows; membership covariates are variance-checked (zero-variance vars are
dropped, and flat-frame copies are replaced from the tour frame).

Env:
  PAPER_SEGMENT   purpose segment (default "work")
  PAPER_N_CASES   cases to use, 0 = all (default 300; full on cluster)
  PAPER_UTILITY   comma list of utility COEF names
                  (default "c_log_dist,c_crash,c_logsum,c_visit_work,c_aggl")
  PAPER_MEMB      comma list of person-type membership columns
                  (default "STUDYING,age5p,sexp,safety_sensitivity")
  PAPER_N_PERT    perturbation restarts for arms C/D (default 2)
  PAPER_LC_MAXITER      EM/SLSQP cap for arms C/D (default 150)
  PAPER_LC_PERTURB_SIGMA  jitter scale (default 0.5)
  PAPER_SEED      sampling/restart seed (default 42)
  PAPER_OUT       output dir (default runs/stage5_paper_arms/<stamp>)

Writes arms.json + arms.md (comparison table) into PAPER_OUT.
"""
import json, os, sys, time, warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("NEW_POP", "1")
os.environ.setdefault("NEW_POP_DIR", r"Z:\test_runs_tours\new_pop")
os.environ.setdefault("NEW_POP_MAPPED", "0")
os.environ.setdefault("PIPELINE_SKIMS_PATH",
                      r"Z:\test_runs_tours\demand_\skims_371.omx")
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)
import numpy as np
import pandas as pd
import cloudpickle


def patch():
    from larch.dataset import flow as f

    def g(self, n):
        if n == "_obj":
            raise AttributeError(n)
        o = self.__dict__.get("_obj")
        if o is None:
            raise AttributeError(n)
        r = getattr(o, n)
        try:
            r = self.transfer_dimension_attrs(r)
        except Exception:
            pass
        return r
    for c in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
        k = getattr(f, c, None)
        if k is not None:
            k.__getattr__ = g


patch()
import larch as lx
from larch import P, X
import larch_lc_compat
larch_lc_compat.apply()
import stage5_destination_choice as S5
from SearchLibrium.mh_choice_estimator import (
    ChoiceSetFrame, _fit_frame_mnl, estimate_dest_choice_lc)

SEG = os.environ.get("PAPER_SEGMENT", "work")
N_CASES = int(os.environ.get("PAPER_N_CASES", "300"))
COEFS = [c.strip() for c in os.environ.get(
    "PAPER_UTILITY",
    "c_log_dist,c_crash,c_logsum,c_visit_work,c_crash_safety_dest").split(
        ",") if c.strip()]
MEMB_WANT = [c.strip() for c in os.environ.get(
    "PAPER_MEMB",
    "STUDYING,age5p,sexp,safety_sensitivity").split(",") if c.strip()]
N_PERT = int(os.environ.get("PAPER_N_PERT", "2"))
LC_MAXITER = int(os.environ.get("PAPER_LC_MAXITER", "150"))
PERT_SIGMA = float(os.environ.get("PAPER_LC_PERTURB_SIGMA", "0.5"))
SEED = int(os.environ.get("PAPER_SEED", "42"))
STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT = os.environ.get("PAPER_OUT",
                     os.path.join("runs", "stage5_paper_arms", STAMP))
os.makedirs(OUT, exist_ok=True)

ROWS = []


def record(**kw):
    ROWS.append(kw)
    ll = kw.get("ll")
    lls = f"{ll:.1f}" if isinstance(ll, float) and np.isfinite(ll) else str(ll)
    print(f"[{kw['arm']}] {kw['label']}: ll={lls} k={kw.get('k')} "
          f"conv={kw.get('converged')} secs={kw.get('seconds')} "
          f"{kw.get('note', '')}", flush=True)


def bic(ll, k, n):
    return float(-2.0 * ll + k * np.log(max(n, 2)))


# ── data (one sampled frame for every arm) ─────────────────────────────────
cache = cloudpickle.load(open(
    r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
seg = cache["seg_acdc"]
sa = cache["sampled_alts"]
if N_CASES and N_CASES < seg["ptour_id"].nunique():
    keep = set(seg[["ptour_id"]].drop_duplicates().sample(
        n=N_CASES, random_state=SEED)["ptour_id"].astype(str))
    seg = seg[seg["ptour_id"].astype(str).isin(keep)].copy()
    sa = sa[sa["ptour_id"].astype(str).isin(keep)].copy()
import openmatrix as omx
skims = omx.open_file(os.environ["PIPELINE_SKIMS_PATH"], "r")
skim_df = S5._skim_flat(skims)
S5._skim_lookup(skim_df)
dt = S5._dest_tree(seg, cache["persons"], cache["lu_with_st"], skims,
                   cache["logsums"], sampled_alts=sa, skim_df=skim_df,
                   crash_skims=skims, segment=SEG)
skims.close()
frame, fmap = S5._adaptive_dest_choice_frame_full(
    dt, sa, SEG, case_covariates=MEMB_WANT)
coef_to_col = {str(v): str(k) for k, v in fmap.items()}
missing = [c for c in COEFS if c not in coef_to_col]
if missing:
    raise KeyError(f"paper utility coefs missing from frame: {missing} "
                   f"(have e.g. {sorted(coef_to_col)[:8]}...)")
COLS = [coef_to_col[c] for c in COEFS]
MEMB = [m for m in MEMB_WANT if m in frame.columns]
if not MEMB:
    raise KeyError(f"no membership columns in frame (wanted {MEMB_WANT})")
# The flat frame's case covariates can be zero-filled for a cached segment
# (seen on the work cache: STUDYING/incp/vehrd/sexp arrive as constants);
# prefer the tour frame's own columns when the frame copies are constant.
for _m in list(MEMB):
    try:
        if _m in seg.columns and frame[_m].nunique() <= 1:
            _map = dict(zip(seg["ptour_id"].astype(str), seg[_m]))
            frame[_m] = (frame["case_id"].astype(str).map(_map)
                         .fillna(frame[_m]).astype(float))
    except Exception:
        pass
_flat = [m for m in MEMB if frame[m].nunique() <= 1]
if _flat:
    print(f"[data] dropping zero-variance membership vars: {_flat}", flush=True)
    MEMB = [m for m in MEMB if m not in _flat]
if not MEMB:
    raise ValueError("no membership covariate has variance in the frame")
N = int(frame["case_id"].nunique())
print(f"[data] segment={SEG} cases={N} utility coefs={COEFS}")
print(f"[data] utility cols={COLS} membership={MEMB}", flush=True)
CRASH_COLS = [c for c in COLS if "crash" in c.lower()]
CRASH0 = CRASH_COLS[0] if CRASH_COLS else None

# ── Arm A: larch MNL ───────────────────────────────────────────────────────
try:
    t0 = time.time()
    # Dict (not list): the list path expands to the always-include backbone,
    # the dict path filters DEST_UEC_FLAT exactly, so arm A estimates the same
    # free parameters as arm B on the same rows.  c_size_term is held at 1.0
    # by the builder (same Manski-Lerman/size offset the SearchLibrium frame
    # folds in), so it adds no free parameter.
    _a_spec = {c: True for c in COEFS}
    # The builder holds c_size_term at 1.0 (same offset the SearchLibrium
    # frame folds in); without it arm A's likelihood omits the size term
    # and is not comparable with arm B's.
    _a_spec["c_size_term"] = True
    mdl = S5._build_dest_model(dt, SEG, custom_dest_vars=_a_spec)
    mdl = S5._estimate(mdl, search_mode=True)
    try:
        ll_a = float(mdl.loglike())
    except Exception:
        ll_a = float("nan")
    try:
        g_a = float(np.linalg.norm(
            np.nan_to_num(np.asarray(mdl.d_loglike(), float))))
    except Exception:
        g_a = float("nan")
    try:
        _pf_names = [str(n) for n in mdl.pf.index]
        k_a = int(sum(1 for n in _pf_names if n in set(COEFS)))
        _pf_note = ("" if k_a == len(COEFS)
                    else f" pf_mismatch={_pf_names}")
        _size_held = ("c_size_term" in mdl.pf.index
                      and abs(float(mdl.pf.loc["c_size_term", "value"]) - 1.0)
                      < 1e-9)
        _pf_note += "" if _size_held else " SIZE_NOT_HELD"
    except Exception:
        k_a = len(COEFS)
        _pf_note = ""
    coefs_a = {}
    try:
        coefs_a = {str(n): float(mdl.pf.loc[n, "value"])
                   for n in mdl.pf.index}
    except Exception:
        pass
    record(arm="A", label="larch MNL (pooled)", ll=ll_a, k=k_a, n=N,
           bic=bic(ll_a, k_a, N) if np.isfinite(ll_a) else float("nan"),
           converged=bool(np.isfinite(ll_a) and np.isfinite(g_a)
                           and g_a < 1e-2),
           seconds=round(time.time() - t0, 1),
           note=f"grad_norm={g_a:.2g} pf_rows={len(coefs_a)}{_pf_note}",
           coefs=coefs_a)
except Exception as e:
    record(arm="A", label="larch MNL (pooled)", ll=float("nan"), k=None, n=N,
           bic=float("nan"), converged=False, seconds=None,
           note=f"FAILED {type(e).__name__}: {e}")

# ── Arm B: SearchLibrium exact MNL ─────────────────────────────────────────
b_b = None
try:
    t0 = time.time()
    fr = ChoiceSetFrame.from_long(frame, COLS)
    b_b, ll_b, res_b = _fit_frame_mnl(fr, np.zeros(fr.dimension))
    k_b, n_b = fr.dimension, fr.n_cases
    record(arm="B", label="SearchLibrium MNL (pooled)", ll=float(ll_b),
           k=k_b, n=n_b, bic=bic(ll_b, k_b, n_b),
           converged=bool(getattr(res_b, "success", False)),
           seconds=round(time.time() - t0, 1),
           note=f"scipy_success={getattr(res_b, 'success', '?')}",
           coefs={c: float(v) for c, v in zip(COLS, b_b)})
except Exception as e:
    record(arm="B", label="SearchLibrium MNL (pooled)", ll=float("nan"),
           k=None, n=N, bic=float("nan"), converged=False, seconds=None,
           note=f"FAILED {type(e).__name__}: {e}")

# ── Arm C: larch latent class ──────────────────────────────────────────────
def _run_arm_c_latentclass():
    """lx.LatentClass arm (larch_lc_compat + fill_missing=0 + jitter restarts).

    Resolutions of the three defects found by instrumenting larch 6.0.46:
      1. larch_lc_compat.apply() patches _GenericFlow.n_cases so the class
         membership dataset (folded then dissolved) reports n_cases from the
         group dim instead of KeyError(None)/'ingroup'.
      2. from_idca(fill_missing=0.0) fills unsampled (case, alt) cells with 0
         instead of NaN: availability still masks them, but the JAX backward
         pass no longer multiplies NaN CA cells by zero gradients (which made
         every class-beta gradient NaN and aborted every optimizer).
      3. Alternative codes must not include 0 (NestingTree(root_id=0) absorbs
         it); real zone ids are safe, synthetic tests must use 1-based alts.
    """
    t0 = time.time()
    ldf = frame[["case_id", "alt_id", "chosen", "log_correction"]
                + COLS + MEMB].copy()
    ldf["case_id"] = ldf["case_id"].astype(str)
    ldf["alt_id"] = pd.to_numeric(ldf["alt_id"], errors="coerce").astype(int)
    if 0 in set(ldf["alt_id"].unique()):
        raise ValueError("LatentClass graphs cannot contain a 0-coded "
                         "alternative (root_id=0 collision)")
    _dtaz = ldf[ldf["chosen"] == 1].set_index("case_id")["alt_id"]
    ldf["DTAZ"] = ldf["case_id"].map(_dtaz).astype(int)
    if ldf["DTAZ"].isna().any():
        raise ValueError("cases without a chosen alternative in the frame")
    ldf = ldf.drop(columns=["chosen"]).set_index(["case_id", "alt_id"])
    dtc = lx.Dataset.construct.from_idca(ldf, fill_missing=0.0)
    _co_ids = [str(c) for c in
               np.asarray(dtc.coords["case_id"].values).ravel()]
    _case_cov = frame.drop_duplicates("case_id").set_index("case_id")
    for cov in MEMB:
        dtc = dtc.assign(**{f"m_{cov}": (
            "case_id", np.asarray(
                _case_cov[cov].reindex(_co_ids).values, float))})
    dtc = dtc.assign(group_id=("case_id",
                               np.arange(len(_co_ids)).astype(np.int64)))

    def _cm(c):
        m = lx.Model()
        expr = None
        for col in COLS:
            t = P(f"c{c}_{col}") * X(col)
            expr = t if expr is None else expr + t
        m.utility_ca = expr
        m.choice_co_code = "DTAZ"
        m.availability_ca_var = "_avail_"
        return m

    m1, m2 = _cm(1), _cm(2)
    mk = lx.Model()
    _me = P("g_intercept") * 1.0
    for cov in MEMB:
        _me = _me + P(f"g_{cov}") * X(f"m_{cov}")
    mk.utility_co[102] = _me
    b = lx.LatentClass(mk, {101: m1, 102: m2}, datatree=dtc,
                       groupid="group_id")
    b.set_cap(20)
    _rng = np.random.default_rng(SEED)
    _base = np.asarray(b.pvals, float).copy()
    _restart_lls = []
    _best = None
    for _s in range(1 + N_PERT):
        if _s:
            b.pvals = (_base + _rng.normal(
                0.0, PERT_SIGMA * np.maximum(np.abs(_base), 0.05)))
        r = b.maximize_loglike(method="slsqp",
                               options={"maxiter": LC_MAXITER,
                                        "ftol": 1e-7},
                               quiet=True)
        ll = float(b.loglike(check_if_best=False, error_if_bad=False))
        _restart_lls.append(ll)
        if np.isfinite(ll) and (_best is None or ll > _best[0]):
            _best = (ll, np.asarray(b.pvals, float).copy())
    ll_c, pv_best = _best
    b.pvals = pv_best
    pf = b.pf
    _betas = {str(n): float(pf.loc[n, "value"]) for n in pf.index}
    _crash_c = [_betas.get(f"c{c}_{CRASH0}") for c in (1, 2)]
    _gap_c = (max(_crash_c) - min(_crash_c)
              if all(v is not None for v in _crash_c) else None)
    pm = b.class_membership_model.pf
    _g = {str(n): float(pm.loc[n, "value"]) for n in pm.index}
    _u = np.full(len(_co_ids), _g.get("g_intercept", 0.0))
    for cov in MEMB:
        _u = _u + _g.get(f"g_{cov}", 0.0) * np.asarray(
            _case_cov[cov].reindex(_co_ids).values, float)
    _p2 = 1.0 / (1.0 + np.exp(-np.clip(_u, -50, 50)))
    _shares_c = [float(1.0 - _p2.mean()), float(_p2.mean())]
    k_c = 2 * len(COLS) + (len(MEMB) + 1)
    record(arm="C", label="larch latent class (lx.LatentClass)",
           ll=float(ll_c), k=k_c, n=N,
           bic=bic(ll_c, k_c, N) if np.isfinite(ll_c) else float("nan"),
           converged=bool(np.isfinite(ll_c)),
           seconds=round(time.time() - t0, 1),
           note=f"restarts={[round(v, 1) for v in _restart_lls]} "
                f"shares={[round(s, 3) for s in _shares_c]} "
                f"crash={[round(v, 2) for v in _crash_c]} gap={_gap_c}",
           shares=_shares_c, class_crash=_crash_c, crash_gap=_gap_c,
           membership=_g)


_arm_c_done = False
if os.environ.get("PAPER_LC_ENGINE", "latentclass").strip().lower() != "manual":
    try:
        _run_arm_c_latentclass()
        _arm_c_done = True
    except Exception as e:
        import traceback
        print(f"[C] lx.LatentClass failed ({type(e).__name__}: {e}) - "
              f"falling back to manual EM", flush=True)
        traceback.print_exc(limit=6)

try:
    if _arm_c_done:
        raise RuntimeError("__arm_c_skip__")
    t0 = time.time()
    ldf = frame[["case_id", "alt_id", "chosen", "log_correction"]
                + COLS + MEMB].copy()
    ldf["case_id"] = ldf["case_id"].astype(str)
    ldf["alt_id"] = pd.to_numeric(ldf["alt_id"], errors="coerce").astype(int)
    _dtaz = ldf[ldf["chosen"] == 1].set_index("case_id")["alt_id"]
    ldf["DTAZ"] = ldf["case_id"].map(_dtaz).astype(int)
    if ldf["DTAZ"].isna().any():
        raise ValueError("cases without a chosen alternative in the frame")
    ldf = ldf.drop(columns=["chosen"]).set_index(["case_id", "alt_id"])
    dtc = lx.Dataset.construct.from_idca(ldf)
    dtc.attrs["_flat_mode"] = True
    _co_ids = [str(c) for c in
               np.asarray(dtc.coords["case_id"].values).ravel()]
    _case_cov = frame.drop_duplicates("case_id").set_index("case_id")
    for cov in MEMB:
        dtc = dtc.assign(**{f"m_{cov}": (
            "case_id", np.asarray(
                _case_cov[cov].reindex(_co_ids).values, float))})
    dtc = dtc.assign(m_intercept=("case_id", np.ones(len(_co_ids))))
    # Integer group codes (NOT the string case ids): sharrow compiles the
    # groupid relationship with numba, and fixed-width unicode group keys hit
    # "Unknown attribute 'item'" in nopython mode.  Integers group identically.
    dtc = dtc.assign(group_id=("case_id",
                               np.arange(len(_co_ids)).astype(np.int64)))

    # ── Manual EM: larch's LatentClass+groupid path is broken in this stack
    # (numba TypingError on string keys; cascade/KeyError failures on int
    # keys — reproduced even on synthetic data), so the E/M steps run here
    # explicitly.  Every choice-probability evaluation and every class M-step
    # is a larch MNL fit (weight_co_var = responsibilities); only the
    # person-type membership M-step is numpy/scipy (fractional logit).
    from scipy.optimize import minimize as _sci_min

    _NC = len(_co_ids)
    _alt_ids = [int(a) for a in
                np.asarray(dtc.coords["alt_id"].values).ravel()]
    _chose = (frame[frame["chosen"] == 1].set_index("case_id")["alt_id"]
              .astype(int).to_dict())
    _ROW = np.arange(_NC)
    _COL = np.array([_alt_ids.index(int(_chose[c])) for c in _co_ids])
    _Xa = np.column_stack(
        [np.ones(_NC)] + [np.asarray(_case_cov[m].reindex(_co_ids).values,
                                     float) for m in MEMB])
    _MSTEP_MAXITER = int(os.environ.get("PAPER_MSTEP_MAXITER", "200"))

    def _build_m(w, x0):
        d = dtc.assign(w=("case_id", np.asarray(w, float)))
        m = S5._lx_model(d, compute_engine="jax")
        expr = None
        for col in COLS:
            t = P(f"b_{col}") * X(col)
            expr = t if expr is None else expr + t
        expr = expr + P("off") * X("log_correction")
        m.utility_ca = expr
        m.choice_co_code = "DTAZ"
        # Sparse sampled frame: from_idca densifies to the union alt axis
        # with NaN utilities off-sample; _avail_ marks the sampled (case,
        # alt) rows available.  availability=1 would expose the NaN cells.
        m.availability_ca_var = "_avail_"
        m.weight_co_var = "w"
        m.lock_value("off", 1.0)
        if x0 is not None:
            try:
                if isinstance(x0, dict):
                    _pv = np.asarray(m.pvals, float).copy()
                    _pn = [str(n) for n in m.pnames]
                    for _i, _n in enumerate(_pn):
                        if _n in x0:
                            _pv[_i] = float(x0[_n])
                    m.pvals = _pv
                else:
                    _x0 = np.asarray(x0, float)
                    if _x0.shape == np.asarray(m.pvals, float).shape:
                        m.pvals = _x0
            except Exception:
                pass
        return m

    def _fit_m(m, maxiter):
        try:
            m.maximize_loglike(method="SLSQP",
                               options={"maxiter": int(maxiter),
                                        "ftol": 1e-9},
                               quiet=True)
        except Exception:
            pass
        try:
            return np.asarray(m.pvals, float).copy()
        except Exception:
            return None

    def _chosen_ll(m):
        pr = np.asarray(m.probability(), float)
        return np.log(np.clip(pr[_ROW, _COL], 1e-300, None))

    def _fit_gamma(p1, l2=0.1):
        _p = np.asarray(p1, float)
        _Xl = _Xa

        def _nll(g):
            u = _Xl @ np.asarray(g, float)
            lp2 = -np.logaddexp(0.0, -u)
            lp1 = -np.logaddexp(0.0, u)
            return float(-(_p * lp2 + (1.0 - _p) * lp1).sum()
                         + 0.5 * float(l2) * float(np.dot(g[1:], g[1:])))
        r = _sci_min(_nll, np.zeros(_Xl.shape[1]), method="L-BFGS-B",
                     options={"maxiter": 200})
        return np.asarray(r.x, float)

    _rng = np.random.default_rng(SEED)
    # pooled start: arm B's exact solution mapped by parameter name (same
    # model, same rows — EM still has to find the latent improvement itself).
    # Falls back to a cold larch fit when arm B failed.
    _x0_pool = None
    if b_b is not None:
        try:
            _x0_pool = {f"b_{c}": float(v) for c, v in zip(COLS, b_b)}
            _x0_pool["off"] = 1.0
        except Exception:
            _x0_pool = None
    _mp0 = _build_m(np.ones(_NC), _x0_pool)
    _bp = _fit_m(_mp0, _MSTEP_MAXITER)
    if _bp is None:
        raise RuntimeError("arm C pooled larch M-step failed")
    _pnames = [str(n) for n in _mp0.pnames]
    _crash_pos = (_pnames.index(f"b_{CRASH0}") if CRASH0
                  and f"b_{CRASH0}" in _pnames else None)
    _ll_pool_c = float(np.sum(_chosen_ll(_mp0)))
    if abs(_ll_pool_c - float(_mp0.loglike())) > 1e-3:
        raise RuntimeError(
            f"arm C probability() misaligned: {_ll_pool_c:.3f} vs "
            f"{float(_mp0.loglike()):.3f}")
    _FLOOR = 0.10
    _best = None
    _restart_lls = []
    for _s in range(1 + N_PERT):
        betas = []
        for _c in range(2):
            _b = np.asarray(_bp, float).copy()
            if _s or _c:
                _b = _b + _rng.normal(0.0, PERT_SIGMA, _b.shape)
            if _crash_pos is not None:
                _b[_crash_pos] += [0.0, -1.0][_c]
            betas.append(_b)
        gammas = np.zeros(_Xa.shape[1])
        models = [_build_m(np.full(_NC, 0.5), betas[_c]) for _c in (0, 1)]
        prev, conv, it, cur = -np.inf, False, 0, -np.inf
        for it in range(1, LC_MAXITER + 1):
            ll_nc = np.column_stack([_chosen_ll(models[_c])
                                     for _c in (0, 1)])
            _u = _Xa @ gammas
            _lse = np.logaddexp(0.0, _u)
            prior = np.column_stack([-_lse, _u - _lse])
            llw = ll_nc + prior
            _m2 = llw.max(axis=1, keepdims=True)
            p = np.exp(llw - _m2)
            p /= p.sum(axis=1, keepdims=True)
            p = np.clip(p, _FLOOR / 2.0, None)
            p /= p.sum(axis=1, keepdims=True)
            cur = float(_m2[:, 0].sum()
                        + np.log(np.exp(llw - _m2).sum(axis=1)).sum())
            if np.isfinite(prev) and abs(cur - prev) < 1e-6 * (1 + abs(prev)):
                conv = True
                break
            prev = cur
            for _c in range(2):
                models[_c] = _build_m(p[:, _c], betas[_c])
                _nb = _fit_m(models[_c], _MSTEP_MAXITER)
                if _nb is not None and _nb.shape == betas[_c].shape:
                    betas[_c] = _nb
            gammas = _fit_gamma(p[:, 1])
            shares = p.mean(axis=0)
            if float(shares.min()) < 0.5 * _FLOOR:
                _coll = int(np.argmin(shares))
                betas[_coll] = (np.asarray(_bp, float)
                                + _rng.normal(0.0, PERT_SIGMA,
                                              np.asarray(_bp, float).shape))
                if _crash_pos is not None:
                    betas[_coll][_crash_pos] += [0.0, -1.0][_coll]
        ll_nc = np.column_stack([_chosen_ll(models[_c]) for _c in (0, 1)])
        _u = _Xa @ gammas
        _lse = np.logaddexp(0.0, _u)
        prior = np.column_stack([-_lse, _u - _lse])
        llw = ll_nc + prior
        _m2 = llw.max(axis=1, keepdims=True)
        p = np.exp(llw - _m2)
        p /= p.sum(axis=1, keepdims=True)
        fit_ll = float(_m2[:, 0].sum()
                       + np.log(np.exp(llw - _m2).sum(axis=1)).sum())
        # polish: full M-steps to recover GEM slack from capped M-steps
        for _pr in range(3):
            for _c in range(2):
                models[_c] = _build_m(p[:, _c], betas[_c])
                _nb = _fit_m(models[_c], max(_MSTEP_MAXITER, 500))
                if _nb is not None and _nb.shape == betas[_c].shape:
                    betas[_c] = _nb
            gammas = _fit_gamma(p[:, 1])
            ll_nc = np.column_stack([_chosen_ll(models[_c])
                                     for _c in (0, 1)])
            _u = _Xa @ gammas
            _lse = np.logaddexp(0.0, _u)
            prior = np.column_stack([-_lse, _u - _lse])
            llw = ll_nc + prior
            _m2 = llw.max(axis=1, keepdims=True)
            _pp = np.exp(llw - _m2)
            _pp /= _pp.sum(axis=1, keepdims=True)
            _nll = float(_m2[:, 0].sum()
                         + np.log(np.exp(llw - _m2).sum(axis=1)).sum())
            if _nll > fit_ll + 1e-9:
                fit_ll, p = _nll, _pp
            else:
                break
        _restart_lls.append(fit_ll)
        if _best is None or fit_ll > _best[0]:
            _best = (fit_ll, betas, gammas, p, conv, it)
    if _ll_pool_c > _best[0]:
        # Nesting guarantee: the LC nests the pooled MNL (equal class betas,
        # zero gammas), so a worse EM mode is never reported above it.
        _best = (_ll_pool_c,
                 [np.asarray(_bp, float).copy(),
                  np.asarray(_bp, float).copy()],
                 np.zeros(_Xa.shape[1]), np.full((_NC, 2), 0.5), True, 0)
        _degen = " DEGENERATE-POOLED-SELECTED"
    else:
        _degen = ""
    ll_c, _betas_c, _gammas_c, _p_c, ok_c, _it_c = _best
    msg_c = (f"em iters={_it_c} pooled_ref={_ll_pool_c:.1f} "
             f"restarts={len(_restart_lls)}{_degen}")
    k_c = 2 * len(COLS) + (len(MEMB) + 1)
    _g = {"g_intercept": float(_gammas_c[0])}
    _g.update({f"g_{m}": float(v) for m, v in zip(MEMB, _gammas_c[1:])})
    _u_c = _Xa @ _gammas_c
    _p2_c = 1.0 / (1.0 + np.exp(-np.clip(_u_c, -50, 50)))
    _shares_c = [float(1.0 - _p2_c.mean()), float(_p2_c.mean())]
    _crash_c = [(float(_betas_c[c][_crash_pos]) if _crash_pos is not None
                 else None) for c in (0, 1)]
    _gap_c = (max(_crash_c) - min(_crash_c)
              if _crash_c and all(v is not None for v in _crash_c) else None)
    record(arm="C", label="larch latent class (2, manual EM)", ll=float(ll_c),
           k=k_c, n=N,
           bic=bic(ll_c, k_c, N) if np.isfinite(ll_c) else float("nan"),
           converged=bool(ok_c and np.isfinite(ll_c)),
           seconds=round(time.time() - t0, 1),
           note=f"em conv={ok_c} restarts={[round(v, 1) for v in _restart_lls]} "
                f"shares={[round(s, 3) for s in _shares_c]} "
                f"crash={[round(v, 2) for v in _crash_c]} gap={_gap_c} :: {msg_c}",
           shares=_shares_c, class_crash=_crash_c, crash_gap=_gap_c,
           membership=_g)
except Exception as e:
    if "__arm_c_skip__" in str(e):
        pass
    else:
        import traceback
        record(arm="C", label="larch latent class (2, manual EM)",
               ll=float("nan"), k=None,
               n=N, bic=float("nan"), converged=False, seconds=None,
               note=f"FAILED {type(e).__name__}: {e}\n"
                    + traceback.format_exc(limit=6))

# ── Arm D: SearchLibrium latent class ──────────────────────────────────────
try:
    t0 = time.time()
    res_d = estimate_dest_choice_lc(
        frame, COLS, n_classes=2, membership_vars=MEMB,
        n_init=1 + N_PERT, perturb_sigma=PERT_SIGMA,
        share_floor=0.10, membership_l2=0.1, maxiter=LC_MAXITER, seed=SEED,
        require_crash=True)
    ll_d = float(res_d["loglike"])
    k_d = 2 * len(COLS) + (len(MEMB) + 1)
    record(arm="D", label="SearchLibrium latent class (2)", ll=ll_d, k=k_d,
           n=int(res_d["n_cases"]),
           bic=bic(ll_d, k_d, int(res_d["n_cases"])),
           converged=bool(res_d.get("converged")),
           seconds=round(time.time() - t0, 1),
           note=f"iters={res_d.get('iterations')} shares={res_d.get('shares')} "
                f"crash_gap={res_d.get('crash_class_gap')} "
                f"restarts={[round(v, 1) for v in res_d.get('restart_lls', [])]}",
           shares=res_d.get("shares"), class_crash=res_d.get("class_crash"),
           crash_gap=res_d.get("crash_class_gap"),
           membership=res_d.get("membership_coefs"))
except Exception as e:
    import traceback
    record(arm="D", label="SearchLibrium latent class (2)", ll=float("nan"),
           k=None, n=N, bic=float("nan"), converged=False, seconds=None,
           note=f"FAILED {type(e).__name__}: {e}\n"
                + traceback.format_exc(limit=6))

# ── table ──────────────────────────────────────────────────────────────────
def _md_table(rows):
    head = ("| arm | model | LL | k | n | BIC | BIC/n | converged | secs | "
            "shares | crash gap | note |")
    sep = ("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | "
           "--- | --- | --- |")
    out = [head, sep]
    for r in rows:
        ll = r.get("ll")
        b = r.get("bic")
        n = r.get("n") or 0
        out.append(
            f"| {r['arm']} | {r['label']} | "
            f"{(f'{ll:.1f}' if isinstance(ll, float) and np.isfinite(ll) else ll)} | "
            f"{r.get('k')} | {n} | "
            f"{(f'{b:.1f}' if isinstance(b, float) and np.isfinite(b) else b)} | "
            f"{(f'{b / max(n, 1):.4f}' if isinstance(b, float) and np.isfinite(b) and n else '')} | "
            f"{r.get('converged')} | {r.get('seconds')} | "
            f"{r.get('shares', '')} | {r.get('crash_gap', '')} | "
            f"{str(r.get('note', ''))[:160]} |")
    return "\n".join(out)


table = _md_table(ROWS)
print("\n" + table + "\n", flush=True)
meta = {"segment": SEG, "n_cases": N_CASES, "utility_coefs": COEFS,
        "utility_cols": COLS, "membership": MEMB, "n_pert": N_PERT,
        "lc_maxiter": LC_MAXITER, "seed": SEED, "stamp": STAMP}
json.dump({"meta": meta, "rows": ROWS},
          open(os.path.join(OUT, "arms.json"), "w"), indent=2, default=str)
open(os.path.join(OUT, "arms.md"), "w").write(
    f"# Stage-5 destination: four estimation arms ({SEG}, {STAMP} UTC)\n\n"
    f"Utility coefs: {COEFS}\n\nMembership (behavioral person types): {MEMB}\n\n"
    + table + "\n")
print(f"saved -> {OUT}/arms.json + arms.md")
