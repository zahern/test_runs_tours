#!/bin/bash
# Full post-run inspection of T4 job 26107921.
export PATH=/opt/pbs/bin:$PATH
P=/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics
cd "$P" || exit 1
L=t4mip.o26107921

echo "=== 1. job state / exit ==="
qstat 26107921 2>&1 | head -4
acct -j 26107921 2>/dev/null | head -4 || qstat -x 26107921 2>&1 | head -6
echo
echo "=== 2. did the plotting fault recur? ==="
grep -c 'plotting failed' "$L" 2>/dev/null && echo "  ^ occurrences of 'plotting failed'"
grep -nE 'WARN' "$L" 2>/dev/null | head -10 || echo "  no WARN lines"
echo
echo "=== 3. part boundaries + exit codes ==="
grep -nE '### PART|exit:|Exit_status|exits:' "$L" 2>/dev/null | head -20
echo
echo "=== 4. the MIP input that was actually used ==="
grep -nE 'predictions:|MIP input' "$L" 2>/dev/null | head -5
echo
echo "=== 5. squad selection + mobility A/B ==="
grep -nE 'mobility A/B|mobility summary|stop_reason|coverage=' "$L" 2>/dev/null | head -24
echo
echo "=== 6. proof figure (pred vs actual) ==="
grep -nE '\[value\]|pred-vs-actual' "$L" 2>/dev/null | head -12
echo
echo "=== 7. figures on disk ==="
for d in figures_tests/t4_armA_base figures_tests/t4_armB_calib figures_tests/t4_mip; do
  echo "--- $d ---"
  ls -la --time-style=+%H:%M "$d" 2>/dev/null | tail -n +4 | awk '{printf "   %-42s %8s  %s\n", $7, $5, $6}'
done
echo
echo "=== 8. MIP tables ==="
for f in game_day_mip_sites.csv game_day_mip_mobility_ab.csv game_day_mip_curves.csv; do
  p=results_tests/t4_mip/$f
  if [ -s "$p" ]; then echo "   $f  ($(wc -l < "$p") lines)"; else echo "   $f  MISSING"; fi
done
echo
echo "=== 9. proof CSVs ==="
for d in results_tests/t4_armA_base results_tests/t4_armB_calib; do
  for f in game_day_pred_vs_actual.csv game_day_top20_per_game.csv; do
    p=$d/$f
    [ -s "$p" ] && echo "   $d/$f  ($(wc -l < "$p") lines)" || echo "   $d/$f  MISSING"
  done
done
