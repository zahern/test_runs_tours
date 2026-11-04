#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Submit all Suncorp Telematics EB Analysis jobs to HPC
# ══════════════════════════════════════════════════════════════════════════════

PROJ_DIR=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics

cd "$PROJ_DIR"

echo "════════════════════════════════════════════════════════════════════════════"
echo "Submitting Suncorp Telematics EB Analysis Jobs"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Job 1: Prepare EB dataset
echo "Submitting Job 1: Data preparation..."
JOB1=$(qsub 01_prepare_data.pbs)
echo "  Job ID: $JOB1"
echo ""

# Job 1b: Road geometry features (depends on Job 1)
echo "Submitting Job 1b: Road geometry features (depends on $JOB1)..."
JOB1b=$(qsub -W depend=afterok:$JOB1 01b_geometry.pbs)
echo "  Job ID: $JOB1b"
echo ""

# Job 2: EB analysis (depends on Job 1b)
echo "Submitting Job 2: EB analysis (depends on $JOB1b)..."
JOB2=$(qsub -W depend=afterok:$JOB1b 02_eb_analysis.pbs)
echo "  Job ID: $JOB2"
echo ""

# Job 3: EB street risk (depends on Job 2)
echo "Submitting Job 3: EB street risk analysis (depends on $JOB2)..."
JOB3=$(qsub -W depend=afterok:$JOB2 03_eb_street_risk.pbs)
echo "  Job ID: $JOB3"
echo ""

# Job 4: Time-resolved EB risk profiling (depends on Job 3 outputs being present;
#        it only requires segment_roads which exists, but we chain after Job 3)
echo "Submitting Job 4: Time-resolved EB profiling (depends on $JOB3)..."
JOB4=$(qsub -W depend=afterok:$JOB3 04_eb_timewise.pbs)
echo "  Job ID: $JOB4"
echo ""

# Job 5: Interactive dashboard (depends on Job 4)
echo "Submitting Job 5: Time-based risk dashboard (depends on $JOB4)..."
JOB5=$(qsub -W depend=afterok:$JOB4 05_dashboard.pbs)
echo "  Job ID: $JOB5"
echo ""

# Job 6: Conceptual forward model + hold-out validation (depends on Job 4)
echo "Submitting Job 6: Conceptual model + hold-out validation (depends on $JOB4)..."
JOB6=$(qsub -W depend=afterok:$JOB4 06_concept_holdout.pbs)
echo "  Job ID: $JOB6"
echo ""

# Job 7: Temporal dynamics - random walk / time-series risk (depends on Job 1b)
echo "Submitting Job 7: Temporal random-walk modelling (depends on $JOB1b)..."
JOB7=$(qsub -W depend=afterok:$JOB1b 07_temporal.pbs)
echo "  Job ID: $JOB7"
echo ""

# Job 8: Crash transferability validation (depends on Job 2)
echo "Submitting Job 8: Crash transferability validation (depends on $JOB2)..."
JOB8=$(qsub -W depend=afterok:$JOB2 08_transfer_validation.pbs)
echo "  Job ID: $JOB8"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "All jobs submitted!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Job chain:"
echo "  1.  $JOB1  - Data preparation (2h, 4 CPUs, 32GB)"
echo "  1b. $JOB1b - Road geometry features (1h, 4 CPUs, 32GB) [after Job 1]"
echo "  2.  $JOB2  - EB analysis (4h, 8 CPUs, 64GB) [after Job 1b]"
echo "  3.  $JOB3  - Street risk (3h, 8 CPUs, 64GB) [after Job 2]"
echo "  4.  $JOB4  - Timewise EB (4h, 8 CPUs, 64GB) [after Job 3]"
echo "  5.  $JOB5  - Dashboard (2h, 4 CPUs, 32GB) [after Job 4]"
echo "  6.  $JOB6  - Conceptual + hold-out (4h, 8 CPUs, 64GB) [after Job 4]"
echo "  7.  $JOB7  - Temporal RW dynamics (2h, 8 CPUs, 64GB) [after Job 1b]"
echo "  8.  $JOB8  - Crash transferability (1h, 4 CPUs, 32GB) [after Job 2]"
echo ""
echo "Monitor with:"
echo "  qstat -u $USER"
echo ""
echo "Check logs in:"
echo "  $PROJ_DIR/"
echo ""
