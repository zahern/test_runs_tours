#!/bin/bash
# submit_now.sh - submit the 61 / 61b chains on the cluster login node.
#
# Run ON the cluster (aqua), not locally:
#   ssh ahernz@aqua.qut.edu.au
#   cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
#   bash submit_now.sh
#
# Optional approximate-Bayes arm (ships the posterior mean and reports the
# plug-in comparison on the same holdout):
#   BAYES=on bash submit_now.sh
#
# The panel already carries speeding_trips / speed_limit (58 --patch-target),
# so p58_rebuild_panel.pbs is NOT part of this chain.
set -u
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1

echo "== preflight =="
python -u preflight.py || { echo "FAIL: preflight refused; not submitting"; exit 1; }

echo
echo "== queue check (avoid duplicate submissions) =="
if qstat -u ahernz | grep -qE "suncorp_segtime_mip|suncorp_segtime_seg"; then
  echo "61/61b already queued or running; not resubmitting"
  qstat -u ahernz | grep -E "suncorp_segtime_mip|suncorp_segtime_seg"
  exit 0
fi

echo
echo "== submit =="
BAYES_ARG=()
[ "${BAYES:-off}" = "on" ] && BAYES_ARG=(-v BAYES=on)
J61=$(qsub "${BAYES_ARG[@]}" 61_segment_time_mip.pbs)
echo "61  -> $J61"
J61B=$(qsub "${BAYES_ARG[@]}" 61b_segment_grouped_mip.pbs)
echo "61b -> $J61B"

echo
qstat -u ahernz | grep -E "suncorp_segtime_mip|suncorp_segtime_seg" || true
echo
echo "logs: $P/suncorp_segtime_mip.o<jobid>, $P/suncorp_segtime_seg.o<jobid>"
echo "watch: qstat -u ahernz"
echo "gate: 61 refuses the MIP unless audit.handoff.pass in"
echo "      results/game_day_holdout_metrics.json (override ALLOW_WEAK_SURFACE=1)"
