#!/bin/bash
cd ~/test_runs_tours/code || exit 1
if pgrep -f run_luti_scenarios_371.py >/dev/null 2>&1; then
  echo ALREADY_RUNNING
else
  rm -f /tmp/luti_run.log
  nohup python3 run_luti_scenarios_371.py > /tmp/luti_run.log 2>&1 &
  echo STARTED_PID_$!
fi
sleep 5
tail -5 /tmp/luti_run.log 2>/dev/null || echo NO_LOG_YET
ls -la saved_data/luti/scenario_travel.csv saved_data/luti/table1_analogue.csv 2>/dev/null || true
