#!/bin/bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
qsub 16_speeding_hazard.pbs
qstat -u ahernz | head -30
