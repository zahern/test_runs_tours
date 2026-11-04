#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
o=ga_s5work_qld_conv1.o26665306
e=ga_s5work_qld_conv1.e26665306
echo "=== job 26665306 (stage5 work) running? ==="
qstat -x -u ahernz 2>/dev/null | grep 26665306 || echo "(not in qstat)"
echo
echo "=== stdout size / first lines ==="
[ -f $o ] && { wc -l $o; sed -n '1,25p' $o; } || echo "(no stdout yet)"
echo
echo "=== did larch_patches apply? (proves LC fix is live) ==="
grep -h "larch_patches" $o 2>/dev/null | head -10 || echo "(no patch lines yet)"
echo
echo "=== LC genes active? ==="
grep -hiE "lc_gene|latent.?class|STAGE5_LC_GENES" $o 2>/dev/null | head -8 || echo "(none yet)"
echo
echo "=== stderr tail ==="
[ -f $e ] && tail -15 $e || echo "(no stderr yet)"