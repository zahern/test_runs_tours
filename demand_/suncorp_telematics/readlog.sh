#!/bin/bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics || exit 1
echo "=== log files for 26107921 ==="
ls -la --time-style=+%H:%M t4mip.*26107921* 2>/dev/null || echo "(none matched)"
ls -la --time-style=+%H:%M t4mip.oSPOOL 2>/dev/null || echo "(no spool)"
echo
for f in t4mip.o26107921 t4mip.oSPOOL; do
  [ -f "$f" ] || continue
  echo "################ $f ################"
  echo "--- plotting fault present? ---"
  if grep -q 'plotting failed' "$f"; then echo "  YES, recurred:"; grep -n 'plotting failed' "$f" | head -5; else echo "  NO -- the surfaces tick fix held"; fi
  echo "--- any WARN ---"
  grep -n 'WARN' "$f" | head -10 || echo "  (none)"
  echo "--- part boundaries / exits ---"
  grep -nE '### PART|exit:|exits:' "$f" | head -20
  echo "--- MIP input used ---"
  grep -nE 'predictions:|MIP input' "$f" | head -5
  echo "--- proof numbers ---"
  grep -nE '\[value\]|pred-vs-actual ->' "$f" | head -14
  echo "--- last 12 lines ---"
  tail -12 "$f"
  echo
done
