#!/bin/bash
# Is pinning c_log_dist at 0 harmless? Fit P0 and P2 non-mandatory under four
# distance bases and compare LL/BIC and, above all, the route coefficient:
#   pin    : current behaviour (log distance held at its sign bound 0)
#   free   : c_log_dist sign left to the data (STAGE5_LOGDIST_FREE=1)
#   iz     : + own-zone constant c_intrazonal (STAGE5_DEST_INTRAZONAL=1)
#   iz_free: both
# Plus the P2LC re-score (fits fine; scorer previously applied 0/122 values)
# and its LC safety contrasts.
set -euo pipefail
cd "$(dirname "$0")"
P=${PROJECT:-FPAXJW8067}
for ARM in P0 P2; do
  for V in pin free iz iz_free; do
    case $V in
      pin)     EXTRA="" ;;
      free)    EXTRA=",STAGE5_LOGDIST_FREE=1" ;;
      iz)      EXTRA=",STAGE5_DEST_INTRAZONAL=1" ;;
      iz_free) EXTRA=",STAGE5_LOGDIST_FREE=1,STAGE5_DEST_INTRAZONAL=1" ;;
    esac
    J=$(qsub -P "$P" -N "nm_${ARM}_${V}" -v "ARM=${ARM},SEGS=nonmand,OUT_TAG=basis_${ARM}_${V}${EXTRA}" job_p0p2_arm.pbs)
    echo "basis ${ARM}/${V} = $J"
  done
done
J=$(qsub -P "$P" -N p2lc_rescore -l walltime=06:00:00 \
     -v "ARM=P2,OUT_TAG=P2LC,STAGE5_MODE_STRUCTURE=lccm,SCORE_ONLY=1" job_p0p2_arm.pbs)
echo "P2LC rescore = $J"
for SEG in work edu nonmand; do
  C=$(qsub -P "$P" -N "sc_LC_${SEG}" -W "depend=afterok:${J}" \
        -v "SEG=${SEG},TAG=P2LC,STAGE5_MODE_STRUCTURE=lccm" job_safety_contrast.pbs)
  echo "contrast P2LC/${SEG} = $C"
done
