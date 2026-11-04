#!/bin/bash
# go.sh - submit the full Suncorp Telematics EB pipeline on the HPC login node.
# Run this ON the cluster (e.g. after: ssh ahernz@aqua.qut.edu.au), NOT locally.
#
#   cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
#   bash go.sh
#
# It submits the whole dependency chain via submit_all.sh:
#   1 -> 1b (geometry) -> 2 -> 3 -> 4 -> 5 / 6
#
# If the data is already prepared (near_miss_hb_ecf.csv exists) and you only
# want to (re)run the geometry feature job + downstream EB jobs, set:
#   FAST=1 bash go.sh
# which submits 01b then 02..06 directly (01 is skipped).
set -e

PROJ_DIR=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$PROJ_DIR"

if [ "${FAST:-0}" = "1" ]; then
    echo "FAST mode: skipping Job 1, submitting 01b + 02..06"
    JOB1b=$(qsub 01b_geometry.pbs)
    JOB2=$(qsub -W depend=afterok:$JOB1b 02_eb_analysis.pbs)
    JOB3=$(qsub -W depend=afterok:$JOB2 03_eb_street_risk.pbs)
    JOB4=$(qsub -W depend=afterok:$JOB3 04_eb_timewise.pbs)
    JOB5=$(qsub -W depend=afterok:$JOB4 05_dashboard.pbs)
    JOB6=$(qsub -W depend=afterok:$JOB4 06_concept_holdout.pbs)
    echo "Submitted: 1b=$JOB1b 2=$JOB2 3=$JOB3 4=$JOB4 5=$JOB5 6=$JOB6"
else
    echo "Full chain mode: submitting via submit_all.sh"
    bash submit_all.sh
fi
