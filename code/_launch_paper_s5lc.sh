#!/bin/bash
# Launch the paper Stage-5 re-estimation with the new LC-via-larch / transit /
# grouped-reporting code, reusing stages 0-4 from qld_conv1.
#
#   Stage 5  (mode + destination) : nl;mnl;lccm  -> exercises build_latent_class
#   Stage 51 (trip destination)   : per-purpose, freshly searched
#   Stages 0-4                    : seeded (copied) from qld_conv1 (already fresh)
#   -> merge unions 0_1_2_3_4_5_51 -> Stage-A paper publish -> predict (Stage B)
#
# Run from the code dir on an aqua login node:  bash _launch_paper_s5lc.sh
set -euo pipefail
cd "$(dirname "$0")"

TAG=${TAG:-paper_s5lc_20261029}
ROOT=${ROOT:-runs/ga_parallel}
SRC=${SRC:-$ROOT/qld_conv1}
PROJECT=${PROJECT:-FPAXJW8067}
DST="$ROOT/$TAG"

echo "=== seed stages 0-4 from $SRC into $DST ==="
mkdir -p "$DST"
for s in stage0 stage1 stage2 stage3 stage4; do
  if [ -d "$SRC/$s" ]; then
    if [ -d "$DST/$s" ]; then
      echo "  $s already present - skip"
    else
      cp -r "$SRC/$s" "$DST/$s"
      echo "  seeded $s ($(find "$DST/$s" -name ga_results.pkl | wc -l) shard pkls)"
    fi
  else
    echo "  WARNING: $SRC/$s missing"
  fi
done

echo "=== submit Stage 5 + 51 (reuse 0-4), merge full set, chain predict ==="
qsub -P "$PROJECT" -j oe \
  -v "RUN_TAG=$TAG,STAGES=5_51,MERGE_STAGE_IDS=0_1_2_3_4_5_51,PUBLISH_PAPER=1,CHAIN_PREDICT=1,GA_S5_MODE_VARIANTS=nl;mnl;lccm" \
  job_submit_ga_sharded.pbs

echo
echo "Submitter queued. It will submit the Stage 5/51 shards, then the merge"
echo "(afterany, STAGE_IDS=0_1_2_3_4_5_51, PUBLISH_PAPER=1), then predict"
echo "(afterok merge, PUBLISH_PAPER=1). Watch: qstat -u \$(whoami)"
