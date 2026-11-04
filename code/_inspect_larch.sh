#!/usr/bin/env bash
SP=/mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv/lib/python3.12/site-packages/larch
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== existing larch patch infra in code dir ==="
ls -la larch_patches.py _install_patched_searchlibrium.sh 2>&1 | sed 's/^/  /'
echo
echo "=== latent_class.py reflow_data_arrays + unmangle + pbounds ==="
sed -n '380,435p' $SP/model/latent_class.py
echo
echo "=== graph property / initialize_graph (jaxmodel + basemodel) ==="
grep -n "def graph\|_graph\b\|def initialize_graph" $SP/model/jaxmodel.py $SP/model/basemodel.py $SP/model/numbamodel.py | head -40