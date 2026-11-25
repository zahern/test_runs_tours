#!/bin/bash
# P2LC refit after the LC constraint / unpickle / LC-scoring fixes (4e6bb98).
set -euo pipefail
cd "$(dirname "$0")"
P=${PROJECT:-FPAXJW8067}
J_LC=$(qsub -P "$P" -N p0p2_P2LC -l walltime=12:00:00 \
       -v "ARM=P2,OUT_TAG=P2LC,STAGE5_MODE_STRUCTURE=lccm" job_p0p2_arm.pbs)
echo "P2LC -> $J_LC"
for SEG in work edu nonmand; do
  J=$(qsub -P "$P" -N "sc_LC_${SEG}" -W "depend=afterok:${J_LC}" \
        -v "SEG=${SEG},TAG=P2LC,STAGE5_MODE_STRUCTURE=lccm" job_safety_contrast.pbs)
  echo "contrast P2LC/${SEG} -> $J"
done
