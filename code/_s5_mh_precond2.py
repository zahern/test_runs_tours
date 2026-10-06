"""Decisive preconditioned-MH test, adapted from larch's optimization design.

Larch: fixed curvature at the optimum + line-search / trust scaling
(scipy SLSQP with jac, or BHHH `solve(BHHH, grad)` with a halving line search)
- it never re-estimates curvature from a short walk.

Here: RWMH whose covariance is FIXED at the Laplace posterior covariance
(inverse of likelihood information + prior precision, i.e. larch's
`jax_param_cov` at the penalized optimum), with only a scalar Robbins-Monro
step scale adapted to the 0.234 optimal-acceptance target during burn-in.
Post-burn-in acceptance and ESS are reported separately (the stock
estimator's acceptance includes the burn-in and hid this).
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
    ChoiceSetFrame, effective_sample_size)

SEG = "work"
CACHE = os.path.join(CODE, "runs", "_s5_larch_perturb", f"{SEG}_dest_cache.pkl")
LARCH = os.path.join(CODE, "runs", "_s5_larch_perturb",
                     f"{SEG}_perturb_results_fixed.json")
OUT = os.path.join(CODE, "runs", "_s5_larch_perturb", f"{SEG}_mh_fixedcov.json")

DRAWS = int(os.environ.get("DRAWS", "20000"))
BURN = int(os.environ.get("BURN", "5000"))
PRIOR = 5.0

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
larch_all = json.load(open(LARCH))["runs"][0]["coefs"]
beta0 = np.array([float(larch_all.get(fmap[c], 0.0)) for c in cols])
print(f"frame {len(frame)} x {d}; larch LL = {csf.loglike(beta0):.3f}")

# ── exact likelihood information at beta0 (observed Hessian) ───────────────
x = np.asarray(csf._packed()[0])
mask = np.asarray(csf._packed()[1])
off = np.asarray(csf._packed()[2])
util = np.einsum("nap,p->na", x, beta0) + off
util = np.where(mask, util, -np.inf)
lse = logsumexp(util, axis=1)
p = np.where(mask, np.exp(util - lse[:, None]), 0.0)
ex = np.einsum("na,nap->np", p, x)
info = np.zeros((d, d))
for n in range(x.shape[0]):
    xn = x[n, mask[n]]
    pn = p[n, mask[n]]
    exn = pn @ xn
    info += (xn * pn[:, None]).T @ xn - np.outer(exn, exn)
post_prec = info + np.eye(d) / (PRIOR ** 2)
cov = np.linalg.inv(post_prec)
print(f"cond(info)={np.linalg.cond(info):.2e} "
      f"cond(post)={np.linalg.cond(post_prec):.2e}")

# Cholesky of the FIXED proposal covariance (2.38^2/d optimal scaling applied
# via the scalar step, not baked into cov).
L = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-12 * np.eye(d))
target_accept = 0.234


def log_post(beta):
    return float(csf.loglike(beta)) - 0.5 * float(np.sum((beta / PRIOR) ** 2))


def run(tag):
    rng = np.random.default_rng(123)
    beta = beta0.copy()
    cur = log_post(beta)
    log_eps = 0.0
    retained = []
    acc_burn = acc_keep = 0
    n_burn = 0
    n_keep = 0
    t0 = time.time()
    total = BURN + DRAWS
    for it in range(total):
        z = rng.standard_normal(d)
        cand = beta + np.exp(log_eps) * (L @ z)
        cand_score = log_post(cand)
        if np.log(rng.random()) < min(0.0, cand_score - cur):
            beta, cur = cand, cand_score
            accepted = 1
        else:
            accepted = 0
        if it < BURN:
            acc_burn += accepted
            n_burn += 1
            # Robbins-Monro log-scale adaptation during burn-in only
            gamma = 1.0 / np.sqrt(it + 10.0)
            log_eps += gamma * (accepted - target_accept)
        else:
            acc_keep += accepted
            n_keep += 1
            retained.append(beta.copy())
    samples = np.asarray(retained)
    ess = np.array([effective_sample_size(samples[:, j])
                    for j in range(d)])
    post_mean = samples.mean(axis=0)
    out = {
        "tag": tag,
        "accept_burn_in": acc_burn / max(n_burn, 1),
        "accept_retained": acc_keep / max(n_keep, 1),
        "posterior_mean_ll": float(csf.loglike(post_mean)),
        "log_post_at_mean": float(log_post(post_mean)),
        "log_post_at_mode": float(log_post(beta0)),
        "ess_min": float(np.min(ess)),
        "ess_median": float(np.median(ess)),
        "ess_max": float(np.max(ess)),
        "max_abs_coef_gap_vs_larch": float(np.max(np.abs(post_mean - beta0))),
        "l2_coef_gap_vs_larch": float(np.sqrt(np.sum((post_mean - beta0) ** 2))),
        "seconds": round(time.time() - t0, 1),
    }
    return out, samples


result, samples = run("fixed_laplace_cov")
print(json.dumps(result, indent=2))
np.save(os.path.join(CODE, "runs", "_s5_larch_perturb",
                     f"{SEG}_mh_fixedcov_samples.npy"), samples)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)
print(f"summary -> {OUT}")
