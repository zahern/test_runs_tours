"""Diagnose WHY WLK_LOC_WLK_TOTIVT__AM is ~55% NaN in skims_371.omx.

Answers: how sparse, in what PATTERN (diagonal? specific destination zones?
whole origin rows?), and whether walk access (WLK_LOC_WLK_* / DISTWALK) can
predict transit availability for the missing cells.
"""
import os
import sys
import numpy as np

OMX = os.environ.get("PIPELINE_SKIMS_PATH", "../demand_/skims_371.omx")
OMX = os.path.abspath(OMX)
print(f"skims: {OMX}")
if not os.path.exists(OMX):
    sys.exit(f"MISSING: {OMX}")

import openmatrix as omx  # noqa: E402

sk = omx.open_file(OMX)
names = list(sk.listMatrices())
mapping = dict(sk.mapping(sk.listMappings()[0]))
order = sorted(mapping, key=lambda k: mapping[k])
n = len(order)
print(f"zones={n} matrices={len(names)}")

WALKCOLS = [m for m in names if "WLK" in m and ("WALK" in m or "LOC" in m)]
print(f"walk-ish columns ({len(WALKCOLS)}): {WALKCOLS[:12]}")


def grab(name):
    a = np.asarray(sk[name], dtype=np.float64)
    return a


TR = "WLK_LOC_WLK_TOTIVT__AM"
print(f"\n=== {TR} ===")
if TR not in names:
    print("ABSENT from OMX!")
    sys.exit(0)
t = grab(TR)
print(f"shape={t.shape}  nan={int(np.isnan(t).sum())} "
      f"({np.isnan(t).mean():.1%})  inf={int(np.isinf(t).sum())}  "
      f"zeros={int((t == 0).sum())}  min={np.nanmin(t):.2f} "
      f"max={np.nanmax(t):.2f}")

nanmask = np.isnan(t)
print("\n--- NaN pattern ---")
row_nan = nanmask.sum(axis=1)
col_nan = nanmask.sum(axis=0)
print(f"rows fully NaN      : {int((row_nan == n).sum())}")
print(f"cols fully NaN      : {int((col_nan == n).sum())}")
print(f"rows with 0 NaN     : {int((row_nan == 0).sum())}")
print(f"NaN on diagonal     : {int(np.diag(nanmask).sum())} of {n}")
off = nanmask.copy()
np.fill_diagonal(off, False)
print(f"NaN off-diagonal    : {int(off.sum())} "
      f"({off.sum() / (n * n - n):.1%} of off-diag)")
print(f"rows with any NaN   : {int((row_nan > 0).sum())}")
print(f"cols with any NaN   : {int((col_nan > 0).sum())}")

# Is the NaN set a contiguous block (banded/intrazonal) or scattered?
print("\n--- per-destination-column NaN rate (top 10 worst) ---")
rates = col_nan / n
worst = np.argsort(-rates)[:10]
for j in worst:
    z = order[j]
    print(f"  col {j:>4} zone {z:>10}  NaN {col_nan[j]:>4}/{n} ({rates[j]:.1%})")

# Transit vs walk: can walk availability predict transit availability?
print("\n--- coupling with walk ---")
wcol = None
for cand in ("WLK_LOC_WLK_TOTIVT__AM", "DRV_LOC_WLK_TOTIVT__AM"):
    if cand in names and cand != TR:
        wcol = cand
        break
dw = grab("DISTWALK") if "DISTWALK" in names else None
acc = None
for cand in ("WLK_LOC_WLK_WACE__AM", "WLK_LOC_WLK_FAR__AM"):
    if cand in names:
        acc = grab(cand)
        break

# For cells where transit is NaN, what do the walk-side columns say?
if dw is not None:
    dn = dw[~np.isnan(dw)]
    dn_nan = dw[np.isnan(t)]
    print(f"DISTWALK  overall  : n={dn.size} median={np.median(dn):.2f} "
          f"min={dn.min():.2f} max={dn.max():.2f}")
    print(f"DISTWALK @NaN cell: n={dn_nan.size} "
          f"median={np.median(dn_nan):.2f} "
          f"min={np.nanmin(dn_nan):.2f} max={np.nanmax(dn_nan):.2f}")
    # walkable fraction (walk < 6km, the availability mask used in the model)
    walkable = np.nan_to_num(dw, nan=1e9) < 6.0
    print(f"P(transit NaN | walkable<6km)     = "
          f"{np.nanmean(nanmask[walkable]):.1%}  (n={walkable.sum()})")
    print(f"P(transit NaN | NOT walkable)    = "
          f"{np.nanmean(nanmask[~walkable]):.1%}  (n={(~walkable).sum()})")

if acc is not None:
    an = acc[~np.isnan(acc)]
    an_nan = acc[np.isnan(t)]
    print(f"WACE      overall  : median={np.median(an):.3f}")
    print(f"WACE @NaN cell    : median={np.nanmedian(an_nan):.3f} "
          f"n={an_nan.size}")

print("\n--- sample NaN destinations (row 0) ---")
r0 = np.flatnonzero(nanmask[0])[:12]
print("  dest cols with NaN:", r0.tolist())
print("  their zone ids    :", [order[j] for j in r0])
ok0 = np.flatnonzero(~nanmask[0])[:12]
print("  dest cols WITHOUT NaN:", ok0.tolist())
print("  their zone ids       :", [order[j] for j in ok0])