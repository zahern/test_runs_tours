"""Live MH-vs-MLE test on real HHTS Stage-4 timing data (escort purpose).

Builds a plain 6-period MNL from observed timing covariates
(day_position, time_consumed, duration_min, time_remaining) with
alternative-specific coefficients, fits it by SLSQP MLE (scipy) and by
the pipeline's RWMH sampler (mh_choice_estimator, adaptive Gaussian
proposal), and compares. Proves the MH machinery + proposal distribution
on real estimation data.

Writes mh_test_s4escort/mh_stage4_report.{txt,json}.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

import compat.numpy_shim  # noqa: F401
import warnings
warnings.filterwarnings("ignore")

import pipeline_logger as pl
from stage4_schedule_placement import prepare_timing_data
from mh_choice_estimator import estimate_mnl_mh

OUT = "mh_test_s4escort"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
t0 = time.time()

print("== MH stage-4 live test: loading survey ==", flush=True)
_, out, landuse = pl.load_data()
trips = out.get("trips")
tours = out.get("tours")
persons = out.get("persons")
print(f"trips={len(trips):,} tours={len(tours) if tours is not None else 0:,} "
      f"persons={len(persons):,}", flush=True)

data = prepare_timing_data(trips, tours, persons)
sub = data[data["purpose"] == "escort"].copy()
print(f"escort rows={len(sub):,}", flush=True)
if len(sub) > 1500:
    sub = sub.sample(1500, random_state=42).reset_index(drop=True)
    print("subsampled to 1500 rows for the test", flush=True)

FEATS = ["day_position", "time_consumed", "duration_min", "time_remaining"]
for c in FEATS:
    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)
# standardise (MH mixes better; MLE invariant up to scaling)
means = sub[FEATS].mean()
stds = sub[FEATS].std().replace(0, 1.0)
Z = ((sub[FEATS] - means) / stds).to_numpy()
periods = np.sort(sub["period"].unique())
P = len(periods)
pidx = {p: i for i, p in enumerate(periods)}
y = sub["period"].map(pidx).to_numpy(dtype=int)
print(f"periods={list(periods)}", flush=True)

# Alt-specific design: X_n[j] = x_n in block j (reference alt = last)
F = len(FEATS)
D = F * (P - 1)
Xlist, yidx = [], []
for n in range(len(sub)):
    Xn = np.zeros((P, D))
    for j in range(P - 1):
        Xn[j, j * F:(j + 1) * F] = Z[n]
    Xlist.append(Xn)
    yidx.append(int(y[n]))
print(f"design: {len(Xlist)} cases x {P} alts x {D} params", flush=True)

# ---- SLSQP MLE reference ----
from scipy.optimize import minimize


def nll(b):
    ll = 0.0
    for Xn, j in zip(Xlist, yidx):
        v = Xn @ b
        v -= v.max()
        e = np.exp(v)
        ll += v[j] - np.log(e.sum())
    return -ll - 0.5 * 1e-4 * float(b @ b) / len(Xlist)


res = minimize(nll, np.zeros(D), method="L-BFGS-B",
               options={"maxiter": 500})
print(f"SLSQP success={res.success} fun={res.fun:.1f}", flush=True)
mle = res.x

# ---- MH posterior ----
fit = estimate_mnl_mh(Xlist, yidx, None, prior_sd=5.0,
                      n_burn=2000, n_draw=3000, thin=3,
                      seed=42, verbose=True)
pm = fit["posterior_mean"]
psd = np.sqrt(np.clip(np.diag(fit["posterior_cov"]), 0, None))
print("accept_rate=", round(fit["accept_rate"], 3), flush=True)

names = [f"{FEATS[f]}_p{j}" for j in range(P - 1) for f in range(F)]
report = {"n_cases": len(sub), "n_alts": P, "n_params": D,
          "slsqp_fun": float(res.fun), "slsqp_success": bool(res.success),
          "mh_accept_rate": float(fit["accept_rate"]),
          "mh_ess_median": float(np.median(np.asarray(fit["ess"], float))),
          "params": []}
print(f"{'param':28s} {'mle':>9s} {'mh_mean':>9s} {'mh_sd':>8s} {'|diff|':>7s}")
maxd = 0.0
for nm, a, b, s in zip(names, mle, pm, psd):
    d = abs(float(a) - float(b))
    maxd = max(maxd, d)
    report["params"].append({"param": nm, "mle": float(a),
                             "mh_mean": float(b), "mh_sd": float(s),
                             "abs_diff": float(d)})
    print(f"{nm:28s} {a:9.3f} {b:9.3f} {s:8.3f} {d:7.3f}")
print(f"max|mle - mh_mean| = {maxd:.3f}  elapsed={time.time()-t0:.0f}s",
      flush=True)
report["max_abs_diff"] = float(maxd)
report["elapsed_s"] = time.time() - t0
report["verdict"] = ("PASS" if maxd < 0.35 and
                     0.10 < fit["accept_rate"] < 0.70 else "CHECK")
with open(os.path.join(OUT, "mh_stage4_report.json"), "w") as fh:
    json.dump(report, fh, indent=2)
print("VERDICT:", report["verdict"], flush=True)
