#!/bin/bash
# Verify the fit/score mode-LL reproduction fixes (2026-11-25):
#   * _run_segment saves the LIVE parameter vector (the one mode_ll is
#     measured at), not the stale _final_pf snapshot  -> needs a refit (P0r/P2r)
#   * the LC scorer keeps elemental alternatives only and compares unweighted
#     -> existing P2LC bundles, re-score only
# The scorer logs "LL reproduction: ... rel_gap=..."; expect ~0 everywhere.
set -euo pipefail
cd "$(dirname "$0")"
P=${PROJECT:-FPAXJW8067}
S=runs/p0p2_routechoice/specs
for ARM in P0 P2; do
  for SEG in work edu nonmand; do
    [ -f "$S/prune_${ARM}_${SEG}.json" ] && cp -f "$S/prune_${ARM}_${SEG}.json" "$S/prune_${ARM}r_${SEG}.json"
  done
  J=$(qsub -P "$P" -N "repro_${ARM}" -v "ARM=${ARM},OUT_TAG=${ARM}r" job_p0p2_arm.pbs)
  echo "refit ${ARM} -> ${ARM}r = $J"
done
J=$(qsub -P "$P" -N p2lc_rescore2 -l walltime=06:00:00 \
     -v "ARM=P2,OUT_TAG=P2LC,STAGE5_MODE_STRUCTURE=lccm,SCORE_ONLY=1" job_p0p2_arm.pbs)
echo "P2LC rescore = $J"
