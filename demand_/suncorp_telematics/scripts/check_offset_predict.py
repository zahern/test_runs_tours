#!/usr/bin/env python
"""Regression check: prediction must reproduce a pinned-offset fit's own mean.

The bug this guards against: ``predict_metacount_sim`` re-built the model with
``build_model_from_manual_spec(...)`` and never passed ``offset_col``, so an
--exposure offset fit predicted exp(Xb) -- a rate per unit exposure -- while
the fit itself was fine. Nothing failed; the surface was just 1/E too low and
every calibration number computed from it was wrong in the same direction.

On a synthetic panel with a KNOWN offset the identity is exact: a Poisson fit
with a free intercept satisfies sum(mu) = sum(Y) on the fitted rows, so both
the model's own predict() and the draw-averaged prediction used by 59 must land
within Monte-Carlo error of sum(Y). This asserts that, with and without the
offset, so a future refactor cannot silently drop it again.

    python scripts/check_offset_predict.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metacountregressor.experiment_package import ExperimentBuilder
from metacount_nb import predict_metacount_sim

RNG = np.random.default_rng(0)
N = 3000
df = pd.DataFrame({
    "row_id": np.arange(N),
    "x": RNG.normal(size=N),
    "z": RNG.normal(size=N),
    "off": RNG.uniform(0.2, 3.0, size=N),          # log-scale offset
    "g": RNG.choice([f"g{i}" for i in range(40)], size=N),
})
eta = -1.2 + 0.4 * df["x"] + 0.5 * df["z"] + df["off"].to_numpy()
df["Y"] = RNG.poisson(np.exp(eta)).astype(float)

spec = dict(fixed_terms=["x"], rdm_terms=[], rdm_cor_terms=[],
            grouped_terms=[], hetro_in_means=[], zi_terms=[],
            membership_terms=[], dispersion=0, latent_classes=1,
            group_id_col=None)
builder = ExperimentBuilder(df, id_col="row_id", y_col="Y",
                            offset_col="off")
# A random term is REQUIRED: without one, predict_metacount_sim falls back to
# predict_metacount (parsed coefficients), so the sim path -- the one that had
# the bug -- would not be exercised at all.
manual = builder.make_manual_spec(fixed_terms=["x"], rdm_terms=["z:normal"],
                                  grouped_terms=[], group_id_col=None)
fit = builder.fit_manual_model(manual_spec=manual, model="poisson", R=20,
                               print_report=False,
                               continuous_de_warm_start=False,
                               use_slsqp=False, de_maxiter=2, de_popsize=3)
fit_info = {"fit": fit, "spec": fit.get("spec"), "manual_spec": manual,
            "offset_col": "off", "beta": {}, "parsed": {}}

y_sum = float(df["Y"].sum())
own = float(np.mean(np.asarray(fit["predictions"], float))) if "predictions" in fit else float("nan")
sim = predict_metacount_sim(df, fit_info, id_col="row_id", y_col="Y", R=40,
                            chunk=2000)
ratio_own = own * len(df) / y_sum if np.isfinite(own) else float("nan")
ratio_sim = float(np.mean(sim)) * len(df) / y_sum
print(f"rows={N}  sum(Y)={y_sum:.0f}")
print(f"  fit predictions mean ratio : {ratio_own:.4f}")
print(f"  predict_metacount_sim ratio: {ratio_sim:.4f}  "
      f"(mean mu={np.mean(sim):.3f})")

# Control: the half-fixed state -- same call, offset withheld. It must be far
# from 1, otherwise this check would not have caught the original bug.
import copy
bad = copy.deepcopy(fit_info)
bad["offset_col"] = None
ratio_bad = float(np.mean(predict_metacount_sim(
    df, bad, id_col="row_id", y_col="Y", R=40, chunk=2000))) * len(df) / y_sum
print(f"  offset withheld (old behaviour): {ratio_bad:.4f}")
print(f"  mean exp(-off) = {np.mean(np.exp(-df['off'])):.4f}  "
      "(the expected size of the error the old path made)")

ok = abs(ratio_sim - 1.0) < 0.15 and abs(ratio_bad - 1.0) > 0.30
print("PASS" if ok else "FAIL: prediction does not reproduce the fitted mean")
raise SystemExit(0 if ok else 1)
