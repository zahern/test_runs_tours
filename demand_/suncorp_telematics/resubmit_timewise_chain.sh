#!/bin/bash
# Resubmit timewise (04, patched OOM fix) -> concept/holdout (06)
cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
echo "submitting 04_eb_timewise.pbs ..."
J4=$(qsub 04_eb_timewise.pbs)
echo "J4=$J4"
echo "submitting 06 (afterok $J4) ..."
J6=$(qsub -W depend=afterok:$J4 06_concept_holdout.pbs)
echo "J6=$J6"
echo "CHAIN $J4 -> $J6"
qstat -u ahernz 2>&1 | grep -E "suncorp|Job ID" | head -10
