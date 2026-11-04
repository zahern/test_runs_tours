#!/bin/bash
# Resubmit MetaCount EB chain: 02 -> 03 -> 04 -> 06
cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
echo "submitting 02_eb_analysis.pbs ..."
J2=$(qsub 02_eb_analysis.pbs)
echo "J2=$J2"
echo "submitting 03 (afterok $J2) ..."
J3=$(qsub -W depend=afterok:$J2 03_eb_street_risk.pbs)
echo "J3=$J3"
echo "submitting 04 (afterok $J3) ..."
J4=$(qsub -W depend=afterok:$J3 04_eb_timewise.pbs)
echo "J4=$J4"
echo "submitting 06 (afterok $J4) ..."
J6=$(qsub -W depend=afterok:$J4 06_concept_holdout.pbs)
echo "J6=$J6"
echo "CHAIN $J2 -> $J3 -> $J4 -> $J6"
qstat -u ahernz | head -30
