#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import larch.model.latent_class as lcl
import inspect
print(inspect.getsourcefile(lcl))
# Check the reflow_data_arrays method
print(inspect.getsource(lcl.LatentClass.reflow_data_arrays)[:3000])
PY