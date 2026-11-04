"""Test the new SearchLibrium destination latent-class / grouped estimator."""
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
import pandas as pd
import cloudpickle
from SearchLibrium.mh_choice_estimator import estimate_dest_choice_lc

# ── A. synthetic 2-class recovery with non-zero sampling offsets ──────────
rng = np.random.default_rng(11)
N, J, K = 600, 5, 3
true_betas = np.array([[0.8, -0.5, 0.3],
                       [-0.6, 0.2, 0.9]])
true_shares = np.array([0.6, 0.4])
rows = []
for n in range(N):
    cls = int(rng.choice(2, p=true_shares))
    x = rng.normal(size=(J, K))
    q = rng.uniform(0.02, 1.0, size=J)          # sampling probabilities
    off = -np.log(q)                             # Manski-Lerman correction
    u = x @ true_betas[cls] + off
    p = np.exp(u - u.max()); p /= p.sum()
    ch = int(rng.choice(J, p=p))
    for a in range(J):
        rows.append({"case_id": n, "alt_id": a, "chosen": int(a == ch),
                     "log_correction": float(off[a]),
                     "x1": x[a, 0], "x2": x[a, 1], "x3": x[a, 2]})
df = pd.DataFrame(rows)
t0 = time.time()
res = estimate_dest_choice_lc(df, ["x1", "x2", "x3"], n_classes=2,
                              maxiter=60, seed=3, n_init=2,
                              require_crash=False)
print(f"[synthetic] method={res['method']} conv={res['converged']} "
      f"iters={res['iterations']} ll={res['loglike']:.1f} "
      f"bic={res['bic']:.1f} ({time.time()-t0:.0f}s)")
print("  shares:", [round(s, 3) for s in res["shares"]],
      "| true:", list(true_shares))
for c, beta in enumerate(res["betas"]):
    print(f"  class {c}: " +
          ", ".join(f"{k}={v:+.2f}(se {res['std_err'][c][k]:.2f})"
                    for k, v in beta.items()))

# ── B. observed-group mode on the same synthetic frame ────────────────────
df["grp"] = (df["case_id"] % 2).map({0: "even", 1: "odd"})
res_g = estimate_dest_choice_lc(df, ["x1", "x2", "x3"], group_col="grp",
                                maxiter=1, require_crash=False)
print(f"\n[grouped] groups={res_g['group_labels']} "
      f"shares={[round(s,2) for s in res_g['shares']]} ll={res_g['loglike']:.1f}")

# ── C. real QLD work destination frame, 2 latent classes ──────────────────
cache = cloudpickle.load(open(r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
if cache.get("dt_dest") is None:
    def _patch():
        from larch.dataset import flow as f
        def g(self, n):
            if n == "_obj": raise AttributeError(n)
            o = self.__dict__.get("_obj")
            if o is None: raise AttributeError(n)
            r = getattr(o, n)
            try: r = self.transfer_dimension_attrs(r)
            except Exception: pass
            return r
        for c in ("_GenericFlow", "_DatasetDC", "_DataArrayDC", "_DataTreeDC"):
            k = getattr(f, c, None)
            if k is not None: k.__getattr__ = g
    _patch()
    from _s5_larch_perturb import _rebuild_dt
    _dt, _sk = _rebuild_dt(cache); _sk.close(); cache["dt_dest"] = _dt
import stage5_destination_choice as S5
frame, fmap = S5._adaptive_dest_choice_frame_full(
    cache["dt_dest"], cache["sampled_alts"], "work")
base = json.load(open(
    r"runs\_s5_larch_perturb\work_perturb_results_fixed.json"))["runs"][0]["coefs"]
init = np.array([float(base.get(fmap[c], 0.0)) for c in frame.columns
                 if c in fmap])
t0 = time.time()
res_r = estimate_dest_choice_lc(frame, list(fmap), n_classes=2, maxiter=8,
                                initial=init, seed=7, n_init=1)
print(f"\n[real QLD work] method={res_r['method']} conv={res_r['converged']} "
      f"iters={res_r['iterations']} ll={res_r['loglike']:.1f} "
      f"bic={res_r['bic']:.1f} ({time.time()-t0:.0f}s)")
print("  shares:", [round(s, 3) for s in res_r["shares"]])
names = res_r["feature_names"]
for c in range(2):
    b = res_r["betas"][c]
    moved = sorted(names, key=lambda n: -abs(b[n] - res_r["betas"][1 - c][n]))[:6]
    print(f"  class {c} top-diverging components: " +
          ", ".join(f"{n}={b[n]:+.2f}" for n in moved))
json.dump({k: v for k, v in res_r.items() if k != "responsibilities"},
          open(r"runs\_s5_larch_perturb\work_lc_dest.json", "w"), indent=2)
print("saved -> work_lc_dest.json")
