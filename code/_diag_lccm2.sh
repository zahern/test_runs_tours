#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import larch.model.latent_class as lcl
import inspect
# Check the graph property
print("=== graph property ===")
print(inspect.getsource(lcl.LatentClass.graph.fget) if hasattr(lcl.LatentClass.graph, 'fget') else "no fget")
print()
# Check the classmodel setter
print("=== classmodel dataset setter ===")
import larch.model.basemodel as bm
print(inspect.getsource(bm.BaseModel.dataset.fset) if hasattr(bm.BaseModel.dataset, 'fset') else "no fset")
PY