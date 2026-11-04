#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== TIMING: patch file vs job start ==="
echo "larch_patches.py mtime:"
ls -la --time-style=full-iso larch_patches.py
echo
echo "stage5_destination_choice.py lines 70-85:"
sed -n '70,85p' stage5_destination_choice.py
echo
echo "job start times (Started line from .o files):"
for j in 26637303 26637305 26637306; do
  o=$(ls ga_s5*lccm3_fixed.o$j 2>/dev/null | head -1)
  [ -n "$o" ] && echo "-- $j: $(grep -m1 'Job id\|Started\|PBS_JOBID' $o) / $(grep -m1 'Finished:' $o)"
done
echo
echo "=================================================="
echo "=== LIVE TEST: does the patch fix the crash? ==="
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
python -u - <<'PY'
import sys, logging
sys.path.insert(0, "/mnt/hpccs01/home/ahernz/test_runs_tours/code")
logging.basicConfig(level=logging.INFO, format="%(message)s")

import larch_patches
larch_patches.apply()

from larch.model.jaxmodel import Model as JaxModel
fset = JaxModel.dataset.fset
print("RESULT: JaxModel.dataset.fset patched =", getattr(fset, "_graph_tolerant", False))

# Directly exercise the exact failing call path on a graphless model.
m = JaxModel()
print("fresh model _graph =", m._graph)
try:
    m.dataset = None
    print("RESULT: graphless dataset assign OK (patched)")
except Exception as e:
    print("RESULT: graphless dataset assign RAISED", type(e).__name__, e)
PY