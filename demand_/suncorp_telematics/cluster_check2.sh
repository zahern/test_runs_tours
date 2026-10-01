#!/bin/bash
# Can the cluster obtain the grouped-draws fix? Route matters:
#   PyPI reachable -> pip upgrade is enough
#   PyPI blocked   -> the single fixed file has to be staged onto the share
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1
module purge 2>/dev/null
module load Anaconda3/2024.02-1 2>/dev/null
conda activate zigenv 2>/dev/null

echo "=== pip index reachable? (10s timeout) ==="
timeout 20 python -m pip index versions metacountregressor 2>&1 | head -4
echo "pip index rc=$?"
echo
echo "=== pip config / index url ==="
python -m pip config list 2>&1 | head -6
echo "PIP_INDEX_URL=${PIP_INDEX_URL:-unset}"
echo
echo "=== where does the installed package live, and can we write it? ==="
MP=$(python -c 'import metacountregressor.main_hpc as m; print(m.__file__)' 2>/dev/null)
echo "module: $MP"
D=$(dirname "$MP")
[ -w "$D" ] && echo "WRITABLE: $D" || echo "NOT writable: $D"
echo
echo "=== how long did stage 59 take last time? (per-game timing) ==="
ls -la --time-style=+%H:%M results_tests/t4_armA_base/ 2>/dev/null | head -8
echo
echo "=== peak memory hint from the previous T4 log ==="
grep -iE 'memory|maxresident|elapsed' t4mip.o26075201 2>/dev/null | head -6
echo
echo "=== is there a staging dir for a local install? ==="
mkdir -p "$P/vendor" && echo "writable: $P/vendor" || echo "cannot create $P/vendor"
