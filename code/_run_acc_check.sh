#!/bin/bash
# Accessibility power-measure check: writes a SEPARATE check file, never touches
# the production accessibility_371.csv.
cd ~/test_runs_tours/code || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu
module load Anaconda3/2024.02-1 >/dev/null 2>&1 || true
conda activate zigenv >/dev/null 2>&1 || true
python -u run_accessibility_371.py --out saved_data/accessibility_371_powercheck.csv --plot-dir saved_data/plots/acc_power_check 2>&1 | tail -40
