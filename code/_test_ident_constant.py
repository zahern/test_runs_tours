"""Regression test: identify_weak_features must catch case-constant columns.

The real failure this covers: the paper's Stage-5 work-segment frame carries a
``logsum`` column that is constant within every case (the mode logsum surface
was flat across destination zones).  A conditional logit cannot identify such
a column -- it cancels from the probability -- and _fit_frame_mnl parks its
coefficient at the ridge (3.6e-16).  The estimator's own identification check
reported NO weak direction, because the scaled-information test divides each
column by sqrt(diag(info)) and that diagonal is the column's own numerical
residue: the dead column acquires a unit "self-correlation" and looks perfect.

Run:  python _test_ident_constant.py            (synthetic checks only)
      FRAME_PARQUET=<path> python ...          (also re-check a real frame)
"""
import glob
import os

import numpy as np
import pandas as pd

from SearchLibrium.mh_choice_estimator import ChoiceSetFrame, identify_weak_features

rng = np.random.default_rng(7)
N_CASES, N_ALTS = 60, 5

rows = []
for c in range(N_CASES):
    case_const = rng.normal(0.0, 2.0)          # varies ACROSS cases only
    for a in range(N_ALTS):
        x1 = rng.normal()
        rows.append(dict(case_id=f"c{c}", alt_id=a, chosen=False,
                         x1=x1, x2=rng.normal(), x1_dup=2.0 * x1,
                         case_const=case_const, zero=0.0,
                         log_correction=0.0))
df = pd.DataFrame(rows)
df.loc[df.groupby("case_id").head(1).index, "chosen"] = True

COLS = ["x1", "x2", "x1_dup", "case_const", "zero"]
rep = identify_weak_features(df, COLS)
print("suggested_drop  :", rep.suggested_drop)
print("constant_features:", rep.constant_features)

# 1. the case-constant column is reported, and named as such
assert "case_const" in rep.constant_features, rep.constant_features
assert "case_const" in rep.suggested_drop, rep.suggested_drop
# 2. a column with no variance at all is equally unidentified
assert "zero" in rep.constant_features, rep.constant_features
# 3. columns that vary within case are NOT called constant
for good in ("x1", "x2", "x1_dup"):
    assert good not in rep.constant_features, (good, rep.constant_features)
# 4. the exactly collinear duplicate is still caught by the eigen pass
assert ("x1" in rep.suggested_drop) or ("x1_dup" in rep.suggested_drop), \
    rep.suggested_drop
# 5. the dead direction is recorded with its loading
dirs = [d for d in rep.weak_directions if "case_const" in d["loadings"]]
assert dirs and dirs[0]["eigenvalue"] == 0.0, dirs

# 6. protected features are never proposed for dropping
rep_p = identify_weak_features(df, COLS, protected=("case_const",))
assert "case_const" not in rep_p.suggested_drop, rep_p.suggested_drop
print("synthetic checks: PASS")

# ── optional: the real Stage-5 work frame, if one has been saved ───────────
path = os.environ.get("FRAME_PARQUET", "")
if not path:
    hits = sorted(glob.glob(os.path.join(
        "runs", "_s5_logsum_diag", "*", "work_dest_frame.parquet")))
    path = hits[-1] if hits else ""
if path and os.path.isfile(path):
    frame = pd.read_parquet(path)
    cols = ["log_DIST_1", "log_CRASH_1", "logsum", "visit_work", "safety_x_crash"]
    fr = ChoiceSetFrame.from_long(frame, cols)
    b, _, _ = __import__(
        "SearchLibrium.mh_choice_estimator", fromlist=["_fit_frame_mnl"]
    )._fit_frame_mnl(fr, np.zeros(fr.dimension))
    rep_real = identify_weak_features(fr, beta=b)
    print(f"real frame {path}: case-constant={rep_real.constant_features} "
          f"drops={rep_real.suggested_drop}")
    assert "logsum" in rep_real.constant_features, rep_real.constant_features
    print("real-frame check: PASS")
else:
    print("real-frame check: skipped (no saved work frame)")
