#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import pandas as pd
from data_connect import Synth_Connect
sc = Synth_Connect()
out = sc.process_survey(save=False)
tours = out['tours']
print(tours['primary_purpose'].value_counts())
PY