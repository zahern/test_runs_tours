#!/usr/bin/env bash
sleep 150
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== stage0 merge job 26665329 ==="
o=$(ls ga_merge_s0.o26665329 2>/dev/null)
e=$(ls ga_merge_s0.e26665329 2>/dev/null)
[ -n "$o" ] && { grep -E "Exit_status|Finished:" $o | tail -2; echo "--- stdout tail ---"; tail -20 $o; } || echo "(no stdout yet)"
[ -n "$e" ] && [ -s "$e" ] && { echo "--- stderr tail ---"; tail -10 $e; }
echo
echo "=== merged stage0 summary (should be REAL now) ==="
cat runs/ga_parallel/cdap_s5_refit_20260928/stage0/merged/ga_reports/ga_summary.csv 2>/dev/null | head -3 || echo "(no merged summary)"
echo
echo "=== queue state ==="
qstat -u ahernz 2>/dev/null | grep -c "qld_conv1\|ga_merge_s0"