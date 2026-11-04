#!/usr/bin/env bash
sleep 240
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
o=ga_s5work_qld_conv1.o26665306
e=ga_s5work_qld_conv1.e26665306
echo "=== state ==="
qstat -x -u ahernz 2>/dev/null | grep -E "26665306" || echo "(finished/left queue)"
echo
echo "=== stdout lines: $(wc -l < $o 2>/dev/null || echo 0) ==="
echo "--- larch_patches (proves graph-tolerant LC fix active) ---"
grep -h "larch_patches" $o 2>/dev/null | head -10 || echo "(none)"
echo "--- LC / latent class evidence ---"
grep -hiE "latent.class|LC gene|lc_genes|lccm" $o 2>/dev/null | head -10 || echo "(none yet)"
echo "--- node_slot_arrays crash? ---"
grep -c "node_slot_arrays" $e $o 2>/dev/null | grep -v ":0" || echo "NO CRASH (good)"
echo "--- stderr tail ---"
tail -12 $e 2>/dev/null || echo "(none)"
echo
echo "=== stage0 job ==="
qstat -x -u ahernz 2>/dev/null | grep 26666095 || echo "(left queue)"
ls -la ga_s0all_qld_conv1.* 2>/dev/null | head -4 || echo "(no stage0 files yet)"