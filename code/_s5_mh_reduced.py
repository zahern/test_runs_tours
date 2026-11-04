"""Apply larch's overspecification lesson: drop the collinear distance recodes
and re-test both estimators.

Kept distance terms:  log_DIST_1 (log decay) + dist_decay (exponential kernel).
Dropped: dist_decay_slow, dist_sq, dist_0_2, dist_2_5, dist_5_10  (all encode
the same decay and load on the near-null singular directions).

Runs:
  1. larch refit on the reduced spec (custom_dest_vars), gradient of the
     reduced-frame likelihood at its solution -> should be ~0 if identified;
  2. fixed-Laplace-covariance preconditioned MH (RM step adaptation) on the
     same reduced frame -> acceptance / ESS / gradient at the mean.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np, cloudpickle
from scipy.special import logsumexp


def patch_larch_flow():
    from larch.dataset import flow as _flow
    def _safe(self, name):
        if name == "_obj": raise AttributeError(name)
        o = self.__dict__.get("_obj")
        if o is None: raise AttributeError(name)
        r = getattr(o, name)
        try: r = self.transfer_dimension_attrs(r)
        except Exception: pass
        return r
    for c in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
        k = getattr(_flow, c, None)
        if k is not None: k.__getattr__ = _safe


patch_larch_flow()
import stage5_destination_choice as S5
from SearchLibrium.mh_choice_estimator import ChoiceSetFrame, effective_sample_size

SEG = "work"
PRIOR = 5.0
DROPPED_COLS = {"dist_decay_slow", "dist_sq", "dist_0_2", "dist_2_5", "dist_5_10"}

cache = cloudpickle.load(open(rf"runs\_s5_larch_perturb\{SEG}_dest_cache.pkl", "rb"))
if cache.get("dt_dest") is None:
    from _s5_larch_perturb import _rebuild_dt
    _dt, _sk = _rebuild_dt(cache); _sk.close(); cache["dt_dest"] = _dt
dt = cache["dt_dest"]

# ── reduced frame ──────────────────────────────────────────────────────────
frame, fmap = S5._adaptive_dest_choice_frame_full(dt, cache["sampled_alts"], SEG)
keep_cols = [c for c in frame.columns
             if c not in DROPPED_COLS or c in ("case_id", "alt_id", "chosen",
                                               "log_correction")]
keep_cols = [c for c in keep_cols if c in frame.columns]
red = frame[keep_cols].copy()
fmap_red = {c: fmap[c] for c in fmap if c in red.columns}
csf = ChoiceSetFrame.from_long(red, list(fmap_red))
cols = list(fmap_red)
d = len(cols)
print(f"reduced frame {len(red)} x {d}")


def log_post(b):
    return float(csf.loglike(b)) - 0.5 * float(np.sum((b / PRIOR) ** 2))


def grad_norm(b):
    _, g = csf.loglike_and_gradient(b)
    return float(np.linalg.norm(g - b / (PRIOR ** 2)))


# ── Laplace covariance on the reduced frame at the larch point ────────────
larch_all = json.load(open(
    rf"runs\_s5_larch_perturb\{SEG}_perturb_results_fixed.json"))["runs"][0]["coefs"]
beta0 = np.array([float(larch_all.get(fmap_red[c], 0.0)) for c in cols])
print(f"larch point: LL={csf.loglike(beta0):.1f} |grad_post|={grad_norm(beta0):.4g}")
x = np.asarray(csf._packed()[0]); mask = np.asarray(csf._packed()[1])
off = np.asarray(csf._packed()[2])
util = np.einsum("nap,p->na", x, beta0) + off
util = np.where(mask, util, -np.inf)
lse = logsumexp(util, axis=1)
p = np.where(mask, np.exp(util - lse[:, None]), 0.0)
ex = np.einsum("na,nap->np", p, x)
info = np.zeros((d, d))
for n in range(x.shape[0]):
    xn = x[n, mask[n]]; pn = p[n, mask[n]]; exn = pn @ xn
    info += (xn * pn[:, None]).T @ xn - np.outer(exn, exn)
post_prec = info + np.eye(d) / (PRIOR ** 2)
cov = np.linalg.inv(post_prec)
print(f"reduced cond(info)={np.linalg.cond(info):.3e}")

# ── 2. preconditioned MH on the reduced frame ─────────────────────────────
L = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-12 * np.eye(d))
rng = np.random.default_rng(123)
beta = beta0.copy(); cur = log_post(beta)
log_eps = 0.0
retained = []
acc_b = acc_k = nb = nk = 0
BURN, DRAWS = int(os.environ.get("BURN", "5000")), int(os.environ.get("DRAWS", "20000"))
t0 = time.time()
for it in range(BURN + DRAWS):
    cand = beta + np.exp(log_eps) * (L @ rng.standard_normal(d))
    cs = log_post(cand)
    acc = int(np.log(rng.random()) < min(0.0, cs - cur))
    if acc:
        beta, cur = cand, cs
    if it < BURN:
        acc_b += acc; nb += 1
        log_eps += (1.0 / np.sqrt(it + 10.0)) * (acc - 0.234)
    else:
        acc_k += acc; nk += 1
        retained.append(beta.copy())
samples = np.asarray(retained)
mean = samples.mean(0)
ess = np.array([effective_sample_size(samples[:, j]) for j in range(d)])
print(f"\npreconditioned MH (reduced): accept_burn={acc_b/max(nb,1):.3f} "
      f"accept_kept={acc_k/max(nk,1):.3f} ({time.time()-t0:.0f}s)")
print(f"  posterior mean LL={csf.loglike(mean):.1f} "
      f"|grad_post|={grad_norm(mean):.4g} "
      f"ess_min={ess.min():.1f} ess_med={np.median(ess):.1f}")

# ── 1. larch refit on the reduced spec ────────────────────────────────────
t1 = time.time()
try:
    _keep = {k: v for k, v in S5.DEST_UEC_FLAT.items()
             if (v[1] is None or SEG in v[1])
             and v[0] not in DROPPED_COLS and k != "c_size_term"}
    mm = S5._build_dest_model(dt, SEG, custom_dest_vars=_keep)
    for n, v in larch_all.items():
        if n in mm.pf.index and np.isfinite(float(v)):
            try: mm.set_value(n, float(v))
            except Exception: pass
    mm = S5._estimate(mm, two_phase=True)
    cc = dict(getattr(mm, "_coef_cache", {}) or {})
    bref = np.array([float(cc.get(fmap_red[c], 0.0)) for c in cols])
    print(f"\nlarch reduced refit ({time.time()-t1:.0f}s): "
          f"LL={csf.loglike(bref):.1f} |grad_post|={grad_norm(bref):.4g}")
except Exception as e:
    print(f"larch reduced refit failed: {e!r}")

out = {
    "dropped_columns": sorted(DROPPED_COLS),
    "n_features": d,
    "mh": {"accept_burn": acc_b / max(nb, 1), "accept_kept": acc_k / max(nk, 1),
           "ll_mean": float(csf.loglike(mean)), "grad_norm_mean": grad_norm(mean),
           "ess_min": float(ess.min()), "ess_median": float(np.median(ess))},
}
json.dump(out, open(rf"runs\_s5_larch_perturb\{SEG}_reduced_basis.json", "w"),
          indent=2)
print(f"\nsaved -> {SEG}_reduced_basis.json")
