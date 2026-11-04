"""Test the behavioural-interaction feature (component x people-type)."""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE); os.chdir(CODE)
import numpy as np, pandas as pd

from SearchLibrium.mh_choice_estimator import (
    add_interaction_columns, estimate_dest_choice_interactions)

# ── A. synthetic: distance effect only for students, crash only for trades ─
rng = np.random.default_rng(5)
N, J = 700, 4
rows = []
TRUE = {"dist": -0.30, "student_x_dist": -0.90, "crash": -0.10,
        "trades_x_crash": -0.80, "size": 0.40}
for n in range(N):
    student = int(rng.random() < 0.35)
    trades = int(rng.random() < 0.30)
    dist = rng.uniform(1, 60, size=J)
    crash = rng.uniform(0, 300, size=J)
    size = rng.normal(size=J)
    logd = np.log1p(dist)
    u = (TRUE["dist"] * logd + TRUE["student_x_dist"] * student * logd
         + TRUE["crash"] * np.log1p(crash)
         + TRUE["trades_x_crash"] * trades * np.log1p(crash)
         + TRUE["size"] * size)
    p = np.exp(u - u.max()); p /= p.sum()
    ch = int(rng.choice(J, p=p))
    for a in range(J):
        rows.append({"case_id": n, "alt_id": a, "chosen": int(a == ch),
                     "log_correction": 0.0,
                     "log_DIST_1": logd[a], "log_CRASH_1": np.log1p(crash[a]),
                     "size_term": size[a],
                     "STUDYING": student, "occp_trades": trades})
df = pd.DataFrame(rows)

aug, new_feats, mapping = add_interaction_columns(
    df, {"log_DIST_1": ["student"], "log_CRASH_1": ["trades"]},
    actor_defs={"student": "STUDYING > 0", "trades": "occp_trades > 0"})
print("[synthetic] new features:", new_feats, "| mapping:", mapping)
print("  true student_x_dist = -0.90 | trades_x_crash = -0.80")

t0 = time.time()
est = estimate_dest_choice_interactions(
    df,
    base_features=["log_DIST_1", "log_CRASH_1", "size_term"],
    interactions={"log_DIST_1": ["student"], "log_CRASH_1": ["trades"]},
    actor_defs={"student": "STUDYING > 0", "trades": "occp_trades > 0"},
    method="mh", draws=6000, burn_in=1500, seed=11, precondition=True)
print(f"  fit {time.time()-t0:.0f}s  ll={est.loglike:.1f} "
      f"accept_ret={est.diagnostics.get('acceptance_retained'):.3f} "
      f"ess_min={est.diagnostics.get('ess_min'):.1f}")
for name, val in est.coef_dict.items():
    se = est.std_err_dict.get(name)
    t = val / se if se else float("nan")
    print(f"    {name:36s} {val:+.3f}  se={se:.3f}  t={t:+.2f}")
print("  interactions meta:", est.metadata.get("interactions"))

# ── B. real QLD work frame: student x distance, trades x crash ────────────
cache = __import__("cloudpickle").load(
    open(r"runs\_s5_larch_perturb\work_dest_cache.pkl", "rb"))
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
    cache["dt_dest"], cache["sampled_alts"], "work",
    case_covariates=["STUDYING", "occp_trades", "age5p"])
print("\n[real] frame cols contain covariates:",
      [c for c in ("STUDYING", "occp_trades", "age5p") if c in frame.columns])
base = json.load(open(
    r"runs\_s5_larch_perturb\work_perturb_results_fixed.json"))["runs"][0]["coefs"]
init = {c: float(base.get(fmap[c], 0.0)) for c in fmap}
t0 = time.time()
est_r = estimate_dest_choice_interactions(
    frame,
    base_features=list(fmap),
    interactions={"log_DIST_1": ["student", "senior"],
                  "log_CRASH_1": ["trades"]},
    actor_defs={"student": "STUDYING > 0", "senior": "age5p >= 9",
                "trades": "occp_trades > 0"},
    method="mh", draws=4000, burn_in=1000, seed=5,
    initial=init, precondition=True, identify=True)
print(f"  fit {time.time()-t0:.0f}s ll={est_r.loglike:.1f} "
      f"accept_ret={est_r.diagnostics.get('acceptance_retained'):.3f} "
      f"ess_min={est_r.diagnostics.get('ess_min'):.1f}")
print("  dropped by identification:", est_r.metadata.get("dropped_features"))
for col, pair in (est_r.metadata.get("interactions") or {}).items():
    val = est_r.coef_dict.get(col)
    se = est_r.std_err_dict.get(col)
    if val is None:
        print(f"    {col:40s} (not estimated - dropped/weak) "
              f"({pair['actor']} x {pair['component']})")
        continue
    t = val / se if se else float("nan")
    print(f"    {col:40s} {val:+.3f}  se={round(se,3) if se else se}  "
          f"t={t:+.2f}   ({pair['actor']} x {pair['component']})")
json.dump({k: v for k, v in est_r.metadata.items()
           if k not in ("identification",)},
          open(r"runs\_s5_larch_perturb\work_dest_interactions.json", "w"),
          indent=2, default=str)
print("saved -> work_dest_interactions.json")
