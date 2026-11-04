#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python -u - <<'PY'
import sys
sys.path.insert(0,"/mnt/hpccs01/home/ahernz/test_runs_tours/code")
import larch
from larch.model.latent_class import LatentClass
print("LatentClass has pf property :", isinstance(getattr(LatentClass,'pf',None), property))
print("LatentClass has pf attr     :", hasattr(LatentClass,'pf'))
import larch.model.numbamodel as nm
print("NumbaModel pf property      :", isinstance(getattr(nm.NumbaModel,'pf',None), property))
PY
echo
echo "=== is stage5 work 26665306 still running (uses OLD code, no ll/k)? ==="
qstat -x -u ahernz 2>/dev/null | grep -E "26665306|26665307|26665309" || echo "(none in qstat)"