#!/bin/bash -l
# 17_submit_chain.sh - chained submit: 17a (search-only, checkpoints) -> 17b
# (resume + hourly predictions + copula + pure-speeding MIP deploy) -> 18
# (extent-varying pure-speeding P4: precinct/inner/district/full).
#
# 17b only starts if 17a exits 0 AND leaves segment_search47_best.json, so a
# walltime kill in the search never silently feeds the MIP stale predictions.
# 18 (array of 4 extents) only starts after 17b succeeds.
# Usage: bash 17_submit_chain.sh [--no-extents]
set -u

submit_17a=$(qsub 17a_segment_search.pbs)
echo "submitted 17a: $submit_17a"
# qsub echoes "<jobid>.<server>"; strip to the numeric id for -W depend.
job17a=$(echo "$submit_17a" | cut -d. -f1)
submit_17b=$(qsub -W depend=afterok:"$job17a" 17b_segment_predict_deploy.pbs)
echo "submitted 17b (afterok $job17a): $submit_17b"
echo "chain: 17a=$submit_17a -> 17b=$submit_17b"

if [ "${1:-}" != "--no-extents" ]; then
  job17b=$(echo "$submit_17b" | cut -d. -f1)
  submit_18=$(qsub -W depend=afterok:"$job17b" 18_extent_mip.pbs)
  echo "submitted 18 (afterok $job17b): $submit_18"
  echo "chain: 17a=$submit_17a -> 17b=$submit_17b -> 18=$submit_18"
fi
