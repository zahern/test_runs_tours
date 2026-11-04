#!/usr/bin/env bash
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import larch.model.basemodel as bm
import inspect
print(inspect.getsource(bm.BaseModel.initialize_graph))
PY