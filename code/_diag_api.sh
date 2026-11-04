#!/usr/bin/env bash
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
python - <<'PY'
import openmatrix as o
print("openmatrix attrs:", [a for a in dir(o) if not a.startswith("_")])
PY
echo "--- how the pipeline reads omx ---"
grep -nE "import openmatrix|omx\.(load|from_file|loadSkim|openSkim)|_OMX|def read_omx" read_skims.py | head -12