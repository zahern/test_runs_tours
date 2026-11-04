#!/bin/bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics || exit 1
echo "=== authoritative stat + md5 (cluster side) ==="
for d in figures_tests/t4_armA_base figures_tests/t4_armB_calib; do
  echo "--- $d ---"
  for f in game_day_surfaces.png game_day_pred_vs_actual.png game_day_coefs.png game_day_calibration.png; do
    p="$d/$f"
    if [ -f "$p" ]; then
      printf "   %-34s %9s  %s  md5=%s\n" "$f" "$(stat -c%s "$p")" \
        "$(stat -c%y "$p" | cut -c12-16)" "$(md5sum "$p" | cut -c1-12)"
    else
      printf "   %-34s MISSING\n" "$f"
    fi
  done
done
echo
echo "=== png integrity (header + IHDR size, not just byte count) ==="
python3 - <<'PY'
import struct, glob
for p in sorted(glob.glob("figures_tests/t4_arm*/game_day_*.png")):
    with open(p, "rb") as fh:
        head = fh.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        print("   BAD HEADER", p); continue
    w, h = struct.unpack(">II", head[16:24])
    print("   %-62s %dx%d" % (p, w, h))
PY
