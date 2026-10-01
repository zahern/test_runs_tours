#!/bin/bash
# Cluster-side pre-submit assessment for the T4 relaunch.
# Run on aquarius02 via: bash cluster_check.sh
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1

echo "=== log files for recent T4/T5 jobs ==="
ls -la t4mip.o* t5*.o* 2>/dev/null | tail -12
echo
echo "--- wallclock / exit of the previous T4 ---"
for L in t4mip.o26075201; do
  [ -f "$L" ] && grep -E 'Start Execution|End Execution|Exit Status|Resource usage' "$L" | head -8
done
echo
echo "--- previous T4 output sizes (how far it actually got) ---"
wc -l results_tests/t4_armA_base/*.csv 2>/dev/null | tail -4
wc -l results_tests/t4_armB_calib/*.csv 2>/dev/null | tail -4
ls -la results/game_day_pred_15min.parquet 2>/dev/null
echo
echo "=== interpreter + package the job will use ==="
module purge 2>/dev/null
module load Anaconda3/2024.02-1 2>/dev/null
conda activate zigenv 2>/dev/null
python -c 'import sys; print("python", sys.version.split()[0])'
python -c 'from importlib.metadata import version; print("metacountregressor", version("metacountregressor"))'
python -c 'import scipy, numpy, pandas; print("scipy", scipy.__version__, "numpy", numpy.__version__, "pandas", pandas.__version__)'
python -c 'from scipy.optimize import milp; print("scipy.optimize.milp OK")'
echo
echo "=== does the cluster copy of the behavioural gate pass? ==="
python -u "$P/check_grouped_draws.py" 2>&1 | tail -8
echo
echo "=== preflight on the cluster ==="
python -u preflight.py 2>&1 | tail -6
echo
echo "=== queue depth for my jobs ==="
qstat -u ahernz 2>&1 | wc -l
echo "=== walltime the previous T4 requested vs likely need ==="
grep -E '^#PBS -l walltime' t4_mip_uclass.pbs
