#!/bin/bash
cd ~/test_runs_tours/code || exit 1
rm -f _submit_s5check.sh _poll_s5check.sh _poll_all.sh _resubmit_batch.sh
ls _submit_s5check.sh _poll_s5check.sh _poll_all.sh _resubmit_batch.sh 2>/dev/null || echo SCRATCH_CLEAN
qstat -u ahernz 2>/dev/null | grep -E '25395528|25395625|25395626|25395627' || echo QUEUE_STATUS_UNAVAILABLE
