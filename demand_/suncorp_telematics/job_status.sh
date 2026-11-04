#!/bin/bash
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1

echo "=== my t4 job ==="
qstat 26095208 2>&1 | head -4
echo
echo "=== queue state counts (all users) ==="
qstat 2>/dev/null | tail -n +4 | awk '{print $NF}' | sort | uniq -c
echo
echo "=== total jobs queued vs running ==="
qstat 2>/dev/null | tail -n +4 | awk '{print $NF}' | sort | uniq -c | sed 's/^/  /'
echo
echo "=== my running jobs (jobid state elapsed) ==="
qstat -u ahernz 2>/dev/null | tail -n +4 | sed 's/  */ /g' | cut -d' ' -f1,5,6,7 | head -25
echo
echo "=== is anything ahead of 26095208 in cpu_batch_exec? ==="
qstat 2>/dev/null | tail -n +4 | awk '$NF=="Q" && $5=="cpu_batch_exec" {print "  Q:", $1, $2}'
echo
echo "=== sched trace for my job (why is it waiting) ==="
tracejob 26095208 2>&1 | head -30
echo
echo "=== checkq accounting ==="
checkq 2>&1 | head -12
