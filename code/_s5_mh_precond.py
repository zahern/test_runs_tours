"""Adapt larch's curvature machinery to SearchLibrium MH.

Larch converges because it optimizes with analytic gradients and the
Hessian/BHHH (outer product of per-case scores) as curvature:
  * larch/optimize.py:85-98      jax_d_loglike / jax_d2_loglike / invhess
  * larch/model/optimization.py  minimize(..., jac=d_logloss, bounds=pbounds)
  * optimization.py:500-540      BHHH: pvals += solve(BHHH, gradient)

SearchLibrium's RWMH starts from an IDENTITY proposal covariance and lets it
adapt from the chain - so a low-acceptance chain collapses its own proposal
(the 0.0008-acceptance run).  Adaptation tested here:
  1. warm start at the larch mode (the estimator's `initial=` argument),
  2. precondition the proposal with the inverse Hessian / OPG at that mode
     (Roberts-Rosenthal 2.38^2/d scaling is already in `scale`),
  3. keep a prior N(0, prior_scale^2) as before.

Runs baseline MH vs preconditioned MH on the SAME crash-fixed full frame and
prints LL / acceptance / ESS for each.
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"

CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)

import numpy as np
import cloudpickle
from scipy.special import logsumexp


def patch_larch_flow():
    from larch.dataset import flow as _flow

    def _safe_getattr(self, name):
        if name == "_obj":
            raise AttributeError(name)
        obj = self.__dict__.get("_obj")
        if obj is None:
            raise AttributeError(name)
        result = getattr(obj, name)
        try:
            result = self.transfer_dimension_attrs(result)
        except Exception:
            pass
        return result

    for _cls in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
        _c = getattr(_flow, _cls, None)
        if _c is not None:
            _c.__getattr__ = _safe_getattr


patch_larch_flow()
import stage5_destination_choice as S5
from SearchLibrium.mh_choice_estimator import (
    ChoiceSetFrame, GaussianRandomWalkProposal, _mh_estimate,
    estimate_dest_choice)

SEG = "work"
CACHE = os.path.join(CODE, "runs", "_s5_larch_perturb", f"{SEG}_dest_cache.pkl")
LARCH = os.path.join(CODE, "runs", "_s5_larch_perturb",
                     f"{SEG}_perturb_results_fixed.json")
OUT = os.path.join(CODE, "runs", "_s5_larch_perturb",
                   f"{SEG}_mh_precond.json")

DRAWS = int(os.environ.get("DRAWS", "8000"))
BURN = int(os.environ.get("BURN", "1600"))

with open(CACHE, "rb") as fh:
    cache = cloudpickle.load(fh)
if cache.get("dt_dest") is None:
    from _s5_larch_perturb import _rebuild_dt
    _dt, _sk = _rebuild_dt(cache)
    try:
        _sk.close()
    except Exception:
        pass
    cache["dt_dest"] = _dt

frame, fmap = S5._adaptive_dest_choice_frame_full(
    cache["dt_dest"], cache["sampled_alts"], SEG)
cols = list(fmap)
csf = ChoiceSetFrame.from_long(frame, cols)
d = len(cols)
print(f"frame {len(frame)} rows x {d} features")

larch_all = json.load(open(LARCH))["runs"][0]["coefs"]
beta_larch = np.array([float(larch_all.get(fmap[c], 0.0)) for c in cols])
print(f"larch LL on frame: {csf.loglike(beta_larch):.3f}")


def hessian_and_opg(beta):
    """Exact Hessian and BHHH/OPG for the sampled MNL at beta.

    Mirrors larch's jax_d2_loglike (exact curvature) and the BHHH matrix
    used by fit_bhhh/propose_direction (outer product of case scores).
    """
    x = np.asarray(csf._packed()[0])            # (n, k, d)
    mask = np.asarray(csf._packed()[1])         # (n, k)
    off = np.asarray(csf._packed()[2])          # (n, k)
    util = np.einsum("nap,p->na", x, beta) + off
    util = np.where(mask, util, -np.inf)
    lse = logsumexp(util, axis=1)
    p = np.where(mask, np.exp(util - lse[:, None]), 0.0)   # (n, k)
    ex = np.einsum("na,nap->np", p, x)                       # (n, d)
    # Per-case observed information, looped to avoid an (n,k,d,d) temporary.
    info = np.zeros((d, d))
    for n in range(x.shape[0]):
        xn = x[n, mask[n]]
        pn = p[n, mask[n]]
        exn = pn @ xn
        info += (xn * pn[:, None]).T @ xn - np.outer(exn, exn)
    scores = np.asarray(csf._packed()[3]) - ex               # chosen - E[x]
    opg = scores.T @ scores
    return info, opg


prior = 5.0
t0 = time.time()
info, opg = hessian_and_opg(beta_larch)
# Laplace/preconditioning covariance of the POSTERIOR, not the bare
# likelihood: larch's inverse Hessian is taken at the penalized optimum, and
# here the N(0, prior^2) prior is what identifies the near-null directions.
# pinv(info) alone has condition 1e25 and produces wild proposals -> the
# 0.0007 acceptance seen in the first attempt.
prior_prec = np.eye(d) / (prior ** 2)
ihess = np.linalg.inv(info + prior_prec)
iopg = np.linalg.inv(opg + prior_prec)
print(f"curvature built in {time.time()-t0:.0f}s; "
      f"cond(info)={np.linalg.cond(info):.2e} "
      f"cond(post)={np.linalg.cond(info + prior_prec):.2e}")


def run(tag, initial, cov, draws=DRAWS, burn=BURN, scale=None):
    prop = GaussianRandomWalkProposal(
        d, scale=scale, burn_in=burn, seed=7, initial_cov=cov)
    t = time.time()
    est = _mh_estimate(
        csf, seed=42, draws=draws, burn_in=burn,
        initial=initial, prior_scale=prior, proposal=prop)
    dd = est.diagnostics
    ll = est.loglike
    print(f"  {tag:14s} LL={ll:12.3f} acc={est.acceptance_rate:.4f} "
          f"ess_min={dd.get('ess_min'):8.2f} ess_med={np.median(dd.get('ess')):8.1f} "
          f"({time.time()-t:.0f}s)")
    return est


print(f"\nsame frame, {DRAWS} draws/{BURN} burn:")
base = run("baseline", np.zeros(d), np.eye(d))
warm = run("warm-start", beta_larch, np.eye(d))
precond = run("precond ihess", beta_larch, ihess)
precond_opg = run("precond opg", beta_larch, iopg)

# MALA-style: preconditioned gradient proposal needs a custom loop; the
# inverse-Hessian RWM above is the direct larch-BHHH analogue.  Record the
# posterior-mean coefficient gap for each arm.
out = {"segment": SEG, "n_features": d, "draws": DRAWS, "burn_in": BURN,
       "larch_ll": float(csf.loglike(beta_larch)),
       "cond_info": float(np.linalg.cond(info))}
for tag, est in (("baseline", base), ("warm", warm),
                 ("precond_ihess", precond), ("precond_opg", precond_opg)):
    coef = np.asarray(est.coef)
    gap = coef - beta_larch
    out[tag] = {
        "ll": float(est.loglike),
        "acceptance": float(est.acceptance_rate),
        "ess_min": float(est.diagnostics.get("ess_min")),
        "max_abs_coef_gap_vs_larch": float(np.max(np.abs(gap))),
        "l2_coef_gap_vs_larch": float(np.sqrt(np.sum(gap ** 2))),
    }
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"\nsummary -> {OUT}")
print(json.dumps(out, indent=2))
