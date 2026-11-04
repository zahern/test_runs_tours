#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
python - <<'PY'
import glob, os, re
# Find raw HHTS source files and any transit/access fields
print("=== raw survey/hhts files present ===")
cands = []
for pat in ("**/*.dta", "**/*.csv", "**/*.sav", "**/*.parquet"):
    for p in glob.glob(pat, recursive=True):
        b = os.path.basename(p).lower()
        if any(k in b for k in ("hhts","survey","trip","tour","pers","household","cdap")):
            cands.append(p)
for p in sorted(set(cands))[:25]:
    print("  ", p)
print(f"  (total {len(set(cands))})")
PY
echo
echo "=== where do tours get built from raw survey? ==="
grep -rnE "def process_survey|def load_data|read_stata|read_csv|ptype|primary_purpose" data_connect.py pipeline_logger.py 2>/dev/null | head -15