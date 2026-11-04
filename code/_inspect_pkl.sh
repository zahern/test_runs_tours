#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import pickle, os
p="runs/ga_parallel/cdap_s5_refit_20260928/stage0/shards/all/ga_results.pkl"
print("pkl mtime:", os.path.getmtime(p))
with open(p,"rb") as fh: res=pickle.load(fh)
print("type:", type(res).__name__)
bf=getattr(res,"best_fitness",None); print("best_fitness:", bf)
feats=getattr(res,"best_features",None)
print("best_features type:", type(feats).__name__,
      "len:", (len(feats) if feats is not None else None))
if isinstance(feats,dict): print("  keys:", list(feats)[:6])
elif isinstance(feats,list): print("  vals:", feats[:6])
md=getattr(res,"metadata",{}) or {}
print("fitness_metric:", getattr(res,"fitness_metric",None))
sr=md.get("segment_results",{})
print("segment_results keys:", list(sr) if isinstance(sr,dict) else type(sr).__name__)
for k,v in (sr.items() if isinstance(sr,dict) else []):
    print(f"   seg[{k}] best_fitness={getattr(v,'best_fitness',None)} "
          f"nfeat={len(getattr(v,'best_features',[]) or [])}")
print("md keys:", list(md)[:15])
PY