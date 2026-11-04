"""Is transit-NaN a hard network/distance CUTOFF (imputable) or scattered
missingness (not imputable)? Bins DISTWALK by decile and reports P(NaN),
plus the transit-time distribution among OBSERVED cells for reference."""
import os
import numpy as np
import openmatrix as omx

OMX = os.path.abspath(os.environ.get("PIPELINE_SKIMS_PATH", "../demand_/skims_371.omx"))
sk = omx.open_file(OMX)
t = np.asarray(sk["WLK_LOC_WLK_TOTIVT__AM"], dtype=np.float64)
dw = np.asarray(sk["DISTWALK"], dtype=np.float64)
n = t.shape[0]
np.fill_diagonal(dw, np.nan)
np.fill_diagonal(t, np.nan)          # intrazonal: 0 = no service, not evidence

nan = np.isnan(t)
d = dw[~np.isnan(dw)]
y = nan[~np.isnan(dw)]

print("P(transit NaN) by DISTWALK decile:")
edges = np.nanpercentile(d, np.arange(0, 101, 10))
print(f"  total cells {d.size}, P(NaN)={y.mean():.1%}\n")
for i in range(10):
    lo, hi = edges[i], edges[i + 1]
    m = (d >= lo) & (d <= hi if i == 9 else d < hi)
    if m.sum():
        print(f"  {lo:7.1f}-{hi:7.1f} km  n={m.sum():>7}  "
              f"P(NaN)={y[m].mean():6.1%}")

print("\nTransit time among OBSERVED off-diagonal cells:")
obs = t[~np.isnan(t)]
obs = obs[obs > 0]
print(f"  n={obs.size}  min={obs.min():.1f}  p5={np.percentile(obs,5):.1f}  "
      f"median={np.median(obs):.1f}  p95={np.percentile(obs,95):.1f}  "
      f"max={obs.max():.1f}")

print("\nWalk-access columns that could proxy transit availability:")
for c in ("WLK_LOC_WLK_WACE__AM", "WLK_LOC_WLK_FAR__AM",
          "WLK_LOC_WLK_BOARDS__AM", "DRV_LOC_WLK_TOTIVT__AM"):
    try:
        a = np.asarray(sk[c], dtype=np.float64)
    except Exception:
        continue
    ann = np.isnan(a)
    if ann.mean() > 0.5:
        print(f"  {c:<24} too sparse ({ann.mean():.0%} NaN) - unusable")
        continue
    # does the proxy separate observed from missing transit cells?
    common = ~np.isnan(dw) & ~ann
    x = a[common]
    ym = nan[common]
    sep = abs(x[~ym].mean() - x[ym].mean()) / (x.std() + 1e-9)
    print(f"  {c:<24} coverage={1-ann.mean():5.1%}  "
          f"mean(observed)={x[~ym].mean():.2f}  mean(NaN)={x[ym].mean():.2f}  "
          f"separation={sep:.2f} sd")