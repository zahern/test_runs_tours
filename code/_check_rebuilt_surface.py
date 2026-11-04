"""_check_rebuilt_surface.py -- characterise the rebuilt work logsum surface.

The Oct-2 cache carried a logsum that was constant across every zone of every
case, so c_logsum could not be identified.  A rebuild makes the column live,
but "live" is not the same as "informative": the surface is standardised by its
global standard deviation in _run_segment, and if a handful of fringe zones
carry almost all of that spread, the fitted c_logsum is driven by those zones
rather than by ordinary accessibility differences.

This script reads the cache (no estimation) and reports:

  * across-zone spread per case, on the full zone universe and on the SA2
    universe destination choice actually samples from;
  * how much of the standardising SD comes from the fringe zone;
  * the within-case share of variance after standardisation, i.e. what fraction
    of the column a conditional logit can actually use;
  * the case coverage of the surface against the segment's tours.

Usage:  python _check_rebuilt_surface.py [cache.pkl]
"""
import sys

import numpy as np

CACHE = (sys.argv[1] if len(sys.argv) > 1
         else "runs/_s5_larch_perturb/work_dest_cache.pkl")

import cloudpickle  # noqa: E402

print(f"[surface] {CACHE}", flush=True)
cache = cloudpickle.load(open(CACHE, "rb"))
ls = cache["logsums"]
v = np.asarray(ls.values, dtype=float)
zids = np.asarray([int(z) for z in ls.coords["zones"].values])
cids = [str(c) for c in ls.coords["cases"].values]
seg_ids = {str(p) for p in cache["seg_acdc"]["ptour_id"].astype(str).unique()}

print(f"[surface] built={cache.get('timestamp', '?')} segment={cache.get('segment')!r}")
print(f"[surface] shape={v.shape}  cases={len(cids)}  zones={len(zids)}")
print(f"[surface] value range {np.nanmin(v):.6g} .. {np.nanmax(v):.6g}")
cov = len(seg_ids & set(cids)) / max(len(seg_ids), 1)
print(f"[surface] case coverage of segment: {cov:.4%} "
      f"({len(seg_ids - set(cids))} tours without a logsum row)")

rowsd = np.nanstd(v, axis=1)
print(f"\n[all zones] across-zone sd per case: min={np.nanmin(rowsd):.6g} "
      f"median={np.nanmedian(rowsd):.6g} max={np.nanmax(rowsd):.6g}")
print(f"[all zones] flat cases: {int((rowsd <= 0).sum())}/{rowsd.size}")

# SA2 universe: the 371 destination zones (9-digit SA2 codes); anything else
# is the fringe/external composite (999), which carries near-infinite
# impedance in the skims and therefore an absurdly low logsum.
sa2 = zids > 1000
print(f"\n[SA2] {int(sa2.sum())} zones; fringe/other {int((~sa2).sum())} "
      f"(ids {sorted(set(zids[~sa2].tolist()))[:5]})")
vs = v[:, sa2]
rsd_sa2 = np.nanstd(vs, axis=1)
print(f"[SA2] across-zone sd per case: min={np.nanmin(rsd_sa2):.6g} "
      f"median={np.nanmedian(rsd_sa2):.6g} max={np.nanmax(rsd_sa2):.6g}")
print(f"[SA2] flat cases: {int((rsd_sa2 <= 0).sum())}/{rsd_sa2.size}")

# How much of the standardising SD is the fringe zone?
all_sd = float(np.nanstd(v))
sa2_sd = float(np.nanstd(vs))
print(f"\n[scale] sd over all zones={all_sd:.4g}  sd over SA2 only={sa2_sd:.4g}  "
      f"ratio={all_sd / sa2_sd if sa2_sd > 0 else float('nan'):.4g}")

# Within-case share: what a conditional logit can use after case differencing.
within = np.nanstd(v - np.nanmean(v, axis=1, keepdims=True), axis=1)
print(f"[within ] per-case within sd: min={np.nanmin(within):.6g} "
      f"median={np.nanmedian(within):.6g} max={np.nanmax(within):.6g}")
tot = float(np.nanstd(v))
print(f"[within ] mean within sd / total sd = "
      f"{float(np.nanmean(within)) / tot if tot > 0 else float('nan'):.4g}")

# Per-case concentration: is the spread carried by one zone in each case?
_mx = np.nanmax(vs, axis=1)
_mn = np.nanmin(vs, axis=1)
print(f"\n[shape] SA2 spread per case (max-min): median="
      f"{np.nanmedian(_mx - _mn):.4g}  p90={np.nanpercentile(_mx - _mn, 90):.4g}")
argmax = np.nanargmax(vs, axis=1)
top = np.bincount(argmax, minlength=vs.shape[1])
top3 = np.argsort(-top)[:3]
print("[shape] zones most often the per-case maximum: "
      + ", ".join(f"zone {int(zids[sa2][i])} ({100.0 * top[i] / vs.shape[0]:.1f}%)"
                  for i in top3))
