#!/bin/bash
echo '=== all ga_/qld/mtc jobs ==='
qstat -u ahernz | grep -iE 'ga_|qld|mtc'
echo
for j in 25422308 25422309 25422310 25422313 25434284; do
  echo "-- job $j"
  qstat -f "$j" 2>/dev/null | grep -iE 'job_state|comment =|hold_types|Submit_arguments' | head -4
done