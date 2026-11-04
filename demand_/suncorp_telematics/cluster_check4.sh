#!/bin/bash
# Verify the fix under the interpreter the JOB uses (zigenv 3.11), not the
# base 3.12 that a bare `python3` gives.
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1
module purge 2>/dev/null
module load Anaconda3/2024.02-1 2>/dev/null
conda activate zigenv 2>/dev/null
export JAX_PLATFORMS=cpu

echo "=== interpreter ==="
which python
python -c 'import sys; print("python", sys.version.split()[0])'
python -c 'import metacountregressor.main_hpc as m; print("module:", m.__file__)'
echo
echo "=== behavioural gate under the job interpreter ==="
python -u "$P/check_grouped_draws.py" 2>&1 | grep -vE 'Jax plugin|jax_plugins|Traceback|File "|  |^$'
echo "gate rc=${PIPESTATUS[0]}"
echo
echo "=== the line itself ==="
grep -n -A1 '_g_draws = int(G)' /home/ahernz/.local/lib/python3.11/site-packages/metacountregressor/main_hpc.py
echo
echo "=== both interpreters see the same file? ==="
for py in python3 "$(which python)"; do
  echo -n "  $py -> "
  $py -c 'import metacountregressor.main_hpc as m; print(m.__file__)' 2>/dev/null || echo "(import failed)"
done
