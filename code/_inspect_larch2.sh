#!/usr/bin/env bash
SP=/mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv/lib/python3.12/site-packages/larch
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== basemodel initialize_graph 874-955 ==="
sed -n '874,955p' $SP/model/basemodel.py
echo
echo "=== numbamodel _rebuild_fixed_arrays 1145-1165 ==="
sed -n '1145,1165p' $SP/model/numbamodel.py
echo
echo "=== jaxmodel dataset setter ~240-260 ==="
sed -n '238,262p' $SP/model/jaxmodel.py