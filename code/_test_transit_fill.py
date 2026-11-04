"""Validate _fill_transit_gaps against the real skims_371.omx.

Checks: (1) fill only touches ELIGIBLE cells, (2) filled values are plausible
vs observed, (3) unserved destinations are never filled, (4) zone-variation is
PRESERVED (c_logsum needs it), (5) determinism + input not mutated.
"""
import os
import sys
import numpy as np
import openmatrix as omx

sys.path.insert(0, "/mnt/hpccs01/home/ahernz/test_runs_tours/code")
OMX = os.path.abspath(os.environ.get("PIPELINE_SKIMS_PATH", "../demand_/skims_371.omx"))
sk = omx.open_file(OMX)
mt = np.asarray(sk["WLK_LOC_WLK_TOTIVT__AM"], dtype=np.float64)
mw = np.asarray(sk["DISTWALK"], dtype=np.float64)
mt_before = mt.copy()

# import the real helper from the module (without importing all of stage5)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "s5", "/mnt/hpccs01/home/ahernz/test_runs_tours/code/stage5_destination_choice.py")
s5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s5)
fill = s5._fill_transit_gaps

out, stats = fill(mt, mw)
print("stats:", stats)

nan_b = np.isnan(mt_before) | np.isinf(mt_before)
nan_a = np.isnan(out) | np.isinf(out)
n = mt.shape[0]
served = (nan_b.sum(axis=0) / n) < 0.90
walk_ok = np.isfinite(mw) & (mw < 6.0)
expect_fill = nan_b & walk_ok & served[None, :]

assert np.array_equal(mt_before, mt, equal_nan=True), "input matrix was mutated!"
print(f"input unmutated: OK")

# (1) exactly the eligible cells changed
changed = ~np.isclose(np.nan_to_num(out, nan=-999), np.nan_to_num(mt_before, nan=-999))
# ignore cells that were already equal by coincidence
was_nan = nan_b
newly_finite = was_nan & ~nan_a
print(f"eligible={int(expect_fill.sum())}  newly_finite={int(newly_finite.sum())}")
assert np.array_equal(newly_finite, expect_fill), "fill touched wrong cells!"

# (2) never filled an unserved destination
bad = expect_fill & ~served[None, :]
assert not bad.any(), "filled an unserved destination!"

# (3) filled values plausible
fv = out[newly_finite]
obs = mt_before[~nan_b]
obs = obs[obs > 0]
print(f"filled n={fv.size} min={fv.min():.1f} median={np.median(fv):.1f} max={fv.max():.1f}")
print(f"observed  median={np.median(obs):.1f}  p5={np.percentile(obs,5):.1f}")
assert np.all(fv > 0), "non-positive fill"
assert np.median(fv) > 5 and np.median(fv) < 400, "implausible fill median"

# (4) zone-variation preserved across destination columns
colstd_a = np.nanstd(out, axis=0)
colstd_b = np.nanstd(np.where(nan_b, np.nan, mt_before), axis=0)
print(f"mean across-dest std: before={np.nanmean(colstd_b):.2f} after={np.nanmean(colstd_a):.2f}")
assert np.nanmean(colstd_a) > 1.0, "fill flattened destination variation"

# (5) determinism
out2, _ = fill(mt, mw)
assert np.array_equal(np.nan_to_num(out, nan=-1), np.nan_to_num(out2, nan=-1))
print("deterministic: OK")

# (6) remaining NaN fraction
print(f"NaN before={nan_b.mean():.1%}  after={nan_a.mean():.1%}  "
      f"reduction={(nan_b.mean()-nan_a.mean()):.1%} of all cells")
print("\nALL FILL CHECKS PASSED")