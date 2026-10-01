#!/bin/bash
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1

echo "=== state of every job in the queue (S is column 5) ==="
qstat 2>/dev/null | tail -n +4 | awk '{print $5}' | sort | uniq -c | sed 's/^/  /'
echo
echo "=== queued (Q) or held (H) jobs, with core/time requests ==="
qstat 2>/dev/null | tail -n +4 | awk '$5=="Q" || $5=="H" {printf "  %-18s %-22s %-2s time=%s\n", $1, $2, $5, $4}'
echo
echo "=== my job in full ==="
qstat -f 26095208 2>&1 | grep -Ei 'job_state|queue|Resource_List|error|comment|Execution|Time_Start|submission' | head -20
echo
echo "=== how many 8-core jobs of mine are running right now ==="
qstat -u ahernz 2>/dev/null | tail -n +4 | awk '$7==8 {print "  ", $1, $2, "state=" $11, "elapsed=" $12}'
echo
echo "=== node availability hint: how many nodes, and are any idle? ==="
qstat -Q 2>&1 | head -5
echo
echo "=== recent starts by my account (is the queue actually turning over?) ==="
ls -lt --time-style=+%d_%H:%M *.o* 2>/dev/null | head -8
