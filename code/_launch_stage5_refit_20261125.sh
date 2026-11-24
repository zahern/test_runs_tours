#!/bin/bash
# Stage-5 refit after the 2026-11-25 fixes: shared fit/score inputs with an
# LL-reproduction check, pooled in-vehicle time, monotone non-mandatory
# distance basis, road crash on road-borne modes, safety latent-class arm.
#   P0, P2           : fit + score (work edu nonmand)
#   P2LC             : P2 specs with the safety latent-class mode model
#   safety contrasts : M_S vs M_0 per segment, after the arm they score
# Run from the code dir on an aqua login node:  bash _launch_stage5_refit_20261125.sh
set -euo pipefail
cd "$(dirname "$0")"
P=${PROJECT:-FPAXJW8067}

J_P0=$(qsub -P "$P" -N p0p2_P0 -v "ARM=P0" job_p0p2_arm.pbs)
J_P2=$(qsub -P "$P" -N p0p2_P2 -v "ARM=P2" job_p0p2_arm.pbs)
J_LC=$(qsub -P "$P" -N p0p2_P2LC -l walltime=12:00:00 \
       -v "ARM=P2,OUT_TAG=P2LC,STAGE5_MODE_STRUCTURE=lccm" job_p0p2_arm.pbs)
echo "P0   -> $J_P0"
echo "P2   -> $J_P2"
echo "P2LC -> $J_LC"

for SEG in work edu nonmand; do
  J=$(qsub -P "$P" -N "sc_P2_${SEG}" -W "depend=afterok:${J_P2}" \
        -v "SEG=${SEG},TAG=P2" job_safety_contrast.pbs)
  echo "contrast P2/${SEG}   -> $J (after $J_P2)"
  J=$(qsub -P "$P" -N "sc_LC_${SEG}" -W "depend=afterok:${J_LC}" \
        -v "SEG=${SEG},TAG=P2LC,STAGE5_MODE_STRUCTURE=lccm" job_safety_contrast.pbs)
  echo "contrast P2LC/${SEG} -> $J (after $J_LC)"
done
qstat -u "$(whoami)" | tail -15
