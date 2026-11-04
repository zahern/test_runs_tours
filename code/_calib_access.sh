#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PIPELINE_SKIMS_PATH=${PIPELINE_SKIMS_PATH:-../demand_/skims_371.omx}
python - <<'PY'
import os, numpy as np, openmatrix as omx
sk = omx.open_file(os.path.abspath(os.environ["PIPELINE_SKIMS_PATH"]))
n = set(sk.listMatrices())
print("WLK_LOC_DRV_* present:", sorted(x for x in n if "WLK_LOC_DRV" in x))
# use only matrices that actually exist
walk = np.asarray(sk["WLK_LOC_WLK_TOTIVT__AM"], dtype=np.float64) / 60.0
pr   = np.asarray(sk["WLK_LOC_DRV_TOTIVT__AM"], dtype=np.float64) / 60.0
np.fill_diagonal(walk, np.nan); np.fill_diagonal(pr, np.nan)
both = np.isfinite(walk) & (walk > 0) & (walk < 2.5) & np.isfinite(pr) & (pr > 0) & (pr < 2.5)
w, p = walk[both], pr[both]
print(f"\ngenuine walk-vs-PnR cells: {w.size}")
print(f"  walk_h median={np.median(w):.2f}  pr_h median={np.median(p):.2f}")
print(f"  PnR faster on {np.mean(p < w):.1%} of these pairs")

CAP = 2.5
def share(beta, asc_p, park, catch):
    uw = -beta*np.nan_to_num(w, nan=CAP, posinf=CAP)
    up = (asc_p - beta*np.nan_to_num(p, nan=CAP, posinf=CAP) - park - catch*0.5)
    m = np.maximum(uw, up)
    return float((np.exp(np.clip(up-m,-500,0)) /
                  (np.exp(np.clip(uw-m,-500,0)) + np.exp(np.clip(up-m,-500,0)))).mean())

print("\n PnR share among the genuine walk-vs-PnR choice set:")
print(f"  {'beta':>5} {'ascP':>6} {'park':>5} {'catch':>6}  {'PnR':>7}")
for beta, asc_p, park, catch in [
        (1.5, -0.8, 0.0, 0.0),
        (0.6, -0.35, 1.10, 0.45),
        (0.6, 0.0, 1.6, 0.6),
        (0.5, 0.0, 2.2, 0.8),
        (0.5, 0.3, 2.6, 1.0)]:
    print(f"  {beta:5.2f} {asc_p:6.2f} {park:5.2f} {catch:6.2f}  {share(beta,asc_p,park,catch):6.1%}")
PY