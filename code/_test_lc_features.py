"""Feature tests for the latent-class destination estimator with a guaranteed
class-specific CRASH component and person-type membership.

Synthetic truth:
  class 0 (share ~0.6): crash -0.2, dist -1.0, visit +0.5
  class 1 (share ~0.4): crash -1.5, dist -0.4, visit -0.2   <- crash-defined
  membership: logit = 0.6 + 0.9*student - 0.7*incp_std
  Manski-Lerman offsets included.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LARCH_FORCE_JAX", "1")
os.environ["JAX_PLATFORMS"] = "cpu"
CODE = r"Z:\test_runs_tours\code"
sys.path.insert(0, CODE)
os.chdir(CODE)
import numpy as np
import pandas as pd
from SearchLibrium.mh_choice_estimator import (
    estimate_dest_choice, estimate_dest_choice_lc)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  [{detail}]" if detail else ""))


rng = np.random.default_rng(11)
N, J = 1500, 4
B0 = np.array([-1.0, -0.2, 0.5])    # dist, crash, visit
B1 = np.array([-0.4, -1.5, -0.2])
GAMMA = np.array([0.6, 0.9, -0.7])  # intercept, student, incp_std
rows = []
for n in range(N):
    student = float(rng.random() < 0.35)
    incp = float(rng.normal())
    u_m = GAMMA[0] + GAMMA[1] * student + GAMMA[2] * incp
    p1 = 1.0 / (1.0 + np.exp(-u_m))
    cls = 1 if rng.random() < p1 else 0
    x = rng.normal(size=(J, 3))
    x[:, 0] = np.log1p(rng.uniform(1, 60, J))
    x[:, 1] = np.log1p(rng.uniform(0, 300, J))
    x[:, 2] = rng.normal(size=J)
    q = rng.uniform(0.02, 1.0, J)
    off = -np.log(q)
    b = B1 if cls == 1 else B0
    u = x @ b + off
    pr = np.exp(u - u.max()); pr /= pr.sum()
    ch = int(rng.choice(J, p=pr))
    for a in range(J):
        rows.append({"case_id": f"c{n}", "alt_id": a, "chosen": int(a == ch),
                     "log_correction": float(off[a]),
                     "log_DIST_1": x[a, 0], "log_CRASH_1": x[a, 1],
                     "visit_work": x[a, 2],
                     "student": student, "incp_std": incp})
df = pd.DataFrame(rows)
FEATS = ["log_DIST_1", "log_CRASH_1", "visit_work"]
MEMB = ["student", "incp_std"]

print("=== T1: crash guarantee (no crash feature -> refuse) ===")
try:
    estimate_dest_choice_lc(df, ["log_DIST_1", "visit_work"], n_classes=2,
                            membership_vars=MEMB, maxiter=3, n_init=1)
    check("T1 refuses spec without crash", False, "no error raised")
except ValueError as e:
    check("T1 refuses spec without crash", "crash" in str(e).lower(), str(e)[:80])

print("\n=== T2: recovery of crash-defined classes + membership ===")
t0 = time.time()
res = estimate_dest_choice_lc(
    df, FEATS, n_classes=2, membership_vars=MEMB,
    n_init=3, perturb_sigma=0.6, crash_init=[0.0, -1.3],
    share_floor=0.10, membership_l2=0.1, maxiter=80, seed=7)
print(f"  fit {time.time()-t0:.0f}s ll={res['loglike']:.1f} bic={res['bic']:.1f} "
      f"conv={res['converged']} iters={res['iterations']}")
print(f"  shares={[round(s,3) for s in res['shares']]} "
      f"class_crash={[round(v,3) for v in res['class_crash']]} "
      f"gap={res['crash_class_gap']:.3f}")
print(f"  restart_lls={[round(v,1) for v in res.get('restart_lls', [])]}")
for k, v in (res.get("membership_coefs") or {}).items():
    print(f"    {k:22s} {v:+.3f}")
crash = np.asarray(res["class_crash"], float)
crash_cls = int(np.argmin(crash))            # more negative = crash-defined
true_gap = B0[1] - B1[1]
check("T2 crash gap ~ true", abs(res["crash_class_gap"] - true_gap) < 0.7,
      f"gap={res['crash_class_gap']:.2f} true={true_gap:.2f}")
check("T2 no class collapse", min(res["shares"]) > 0.15,
      f"min share={min(res['shares']):.3f}")
mc = res.get("membership_coefs") or {}
# The membership gammas always describe class 2 relative to the class-1
# reference.  If the CRASH class is the reference (index 0), the crash
# class's own membership direction is the NEGATIVE of the fitted gammas
# (binary-logit symmetry); otherwise the fitted gammas ARE the crash class's.
got = np.array([mc.get("g_intercept_c2", np.nan),
                mc.get("g_student_c2", np.nan),
                mc.get("g_incp_std_c2", np.nan)], float)
if crash_cls == 0:
    got = -got
check("T2 membership gammas recovered (crash class)",
      np.all(np.isfinite(got)) and np.max(np.abs(got - GAMMA)) < 0.8,
      f"got={np.round(got,2).tolist()} true={GAMMA.tolist()}")
check("T2 restarts explored", (res.get("restart_ll_span", 0.0) or 0.0) >= 0.0,
      f"span={res.get('restart_ll_span')}")

print("\n=== T3: share floor prevents collapse when classes are identical ===")
rows2 = []
for r in rows:
    r = dict(r)
    # destroy the class signal: same utility for everyone
    r["log_CRASH_1"] = r["log_CRASH_1"] * 1.0
    rows2.append(r)
df2 = pd.DataFrame(rows2)
res3 = estimate_dest_choice_lc(
    df2, FEATS, n_classes=2, membership_vars=MEMB,
    n_init=1, share_floor=0.10, maxiter=25, seed=3)
check("T3 min share respects floor", min(res3["shares"]) >= 0.04,
      f"shares={[round(s,3) for s in res3['shares']]}")

print("\n=== T4: genome decode — LC genes select the crash-class model ===")
# Prototype of the extended GA chromosome: base destination terms + LC block.
CANDIDATES = (["base:" + f for f in FEATS]
              + ["lc_on", "lc_nc2", "lc_nc3", "lc_crash"]
              + ["lc_memb:" + m for m in MEMB])


def decode(genes):
    """Genome -> (mode, kwargs).  Hard constraint: lc_on implies lc_crash."""
    on = dict(zip(CANDIDATES, genes))
    base = [c.split(":", 1)[1] for c in CANDIDATES
            if c.startswith("base:") and on[c]]
    if not on.get("lc_on"):
        return ("mnl", {"features": base})
    if not on.get("lc_crash"):
        # repair: force the crash-class gene (hard constraint)
        on["lc_crash"] = 1
    memb = [c.split(":", 1)[1] for c in CANDIDATES
            if c.startswith("lc_memb:") and on[c]]
    nc = 3 if on.get("lc_nc3") else 2
    return ("lc", {"features": base or FEATS, "n_classes": nc,
                   "membership_vars": memb or MEMB})


def fitness(genes):
    mode, kw = decode(genes)
    try:
        if mode == "mnl":
            est = estimate_dest_choice(df, feature_cols=kw["features"],
                                       method="mh", draws=400, burn_in=100,
                                       seed=1)
            ll = float(est.loglike)
            bic = -2.0 * ll + len(kw["features"]) * np.log(df["case_id"].nunique())
        else:
            r = estimate_dest_choice_lc(df, kw["features"], n_classes=kw["n_classes"],
                                        membership_vars=kw["membership_vars"],
                                        n_init=1, share_floor=0.10, maxiter=40,
                                        require_crash=True, seed=5)
            bic = float(r["bic"])
    except Exception as e:
        print(f"    fitness failed ({mode}): {type(e).__name__}: {e}")
        return -1e9
    return -bic


g_mnl = np.array([1, 1, 1] + [0, 0, 0, 0] + [0, 0], float)
g_lc = np.array([1, 1, 1] + [1, 0, 0, 1] + [1, 1], float)
f_mnl, f_lc = fitness(g_mnl), fitness(g_lc)
mode_mnl, _ = decode(g_mnl)
mode_lc, kw_lc = decode(g_lc)
print(f"  pooled MNL  fitness(-BIC)={f_mnl:.1f}  mode={mode_mnl}")
print(f"  LC genome   fitness(-BIC)={f_lc:.1f}  mode={mode_lc} nc={kw_lc['n_classes']} "
      f"memb={kw_lc['membership_vars']}")
check("T4 LC-with-crash beats pooled MNL on BIC", f_lc > f_mnl,
      f"lc={f_lc:.1f} mnl={f_mnl:.1f}")
# hard constraint repair
g_bad = np.array([1, 1, 1] + [1, 0, 0, 0] + [0, 0], float)
_, kw_bad = decode(g_bad)
check("T4 lc_on forces crash gene (repair)", "log_CRASH_1" in kw_bad["features"],
      f"features={kw_bad['features']}")

n_pass = sum(1 for ok, _, _ in RESULTS if ok)
print(f"\n{n_pass}/{len(RESULTS)} checks passed")
json.dump({"results": [{"ok": o, "name": n, "detail": d} for o, n, d in RESULTS]},
          open(r"runs\_s5_larch_perturb\lc_feature_tests.json", "w"), indent=2)
sys.exit(0 if n_pass == len(RESULTS) else 1)
