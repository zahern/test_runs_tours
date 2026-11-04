#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
# The 3-way access softmax + observed-access override are gated under
# STAGE5_PT_MODES=1 (default runs keep the legacy binary walk/drive logit).
export STAGE5_PT_MODES=1
echo "=== compile ==="
python -m py_compile stage5_destination_choice.py ga_hypothesis_testing.py \
  ga_variant_refits.py ga_shard.py && echo "COMPILE OK" || { echo FAILED; exit 1; }
python -u - <<'PY'
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/mnt/hpccs01/home/ahernz/test_runs_tours/code")
import stage5_destination_choice as s5

rng=np.random.default_rng(7)
N=40
def mk(n):
    return pd.DataFrame({
      "tour_id":[f"t{i}" for i in range(n)],
      "CARLICENCE":rng.integers(0,2,n),
      "vehrd":rng.integers(0,3,n),
      "bikes":rng.integers(0,2,n),
      "DISTWALK":rng.uniform(0.2,12,n),
      "WLK_LOC_WLK_TOTIVT__AM":rng.uniform(30,200,n),
      "WLK_LOC_DRV_TOTIVT__AM":rng.uniform(20,160,n),
      "WLK_LOC_DRV_DTIM__AM":rng.uniform(2,25,n),
      "WLK_LOC_DRV_WAIT__AM":rng.uniform(1,30,n),
      "WLK_LOC_DRV_BOARDS__AM":rng.uniform(0.2,4,n),
      "WLK_LOC_DRV_DDIST__AM":rng.uniform(1,30,n),
    })

print("== A. no observed map (pure simulation) ==")
r=mk(N); out=s5._predict_transit_access(r,None,np.random.default_rng(1))
print(f"   len(out)={len(out)}  len(rows)={len(r)}  ALIGNED={len(out)==len(r)}")
print(f"   codes seen={sorted(set(out.tolist()))} (valid 1/2/3)")

print("== B. ALL observed ==")
obs={f"t{i}": (1 if i%3==0 else 2 if i%3==1 else 3) for i in range(N)}
out=s5._predict_transit_access(mk(N),None,np.random.default_rng(2),observed_access=obs)
exp=np.array([obs[f"t{i}"] for i in range(N)])
print(f"   len={len(out)} ALIGNED={len(out)==N}  matches_observed={np.array_equal(out,exp)}")

print("== C. MIXED observed + unknown  (the bug case) ==")
mixed={f"t{i}": 2 for i in range(0,N,2)}          # even rows observed=Drive
out=s5._predict_transit_access(mk(N),None,np.random.default_rng(3),observed_access=mixed)
evens=np.array([mixed[f"t{i}"] for i in range(0,N,2)])
print(f"   len(out)={len(out)}  len(rows)={N}  ALIGNED={len(out)==N}")
print(f"   all observed(even) rows honoured = {np.array_equal(out[0::2],evens)}")
print(f"   simulated(odd) rows in valid set  = {set(out[1::2].tolist()) <= {1,2,3}}")

print("== D. unknown tour_id in map -> 0 must NOT count as observed ==")
z={f"t{i}":0 for i in range(N)}                  # 0 == unknown sentinel
out=s5._predict_transit_access(mk(N),None,np.random.default_rng(4),observed_access=z)
print(f"   len={len(out)} ALIGNED={len(out)==N}  any zeros kept={bool((out==0).any())} (must be False)")
print(f"   codes={sorted(set(out.tolist()))}")

print("== E. idempotence: observed rows never drift across seeds ==")
o1=s5._predict_transit_access(mk(N),None,np.random.default_rng(9),observed_access=mixed)
o2=s5._predict_transit_access(mk(N),None,np.random.default_rng(123),observed_access=mixed)
print(f"   observed positions identical across seeds = {np.array_equal(o1[0::2],o2[0::2])}")
print("PT TESTS DONE")
PY