#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
source /mnt/hpccs01/home/ahernz/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/hpccs01/home/ahernz/miniconda3/envs/zigenv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
echo "=== initial_trips_hhts.csv columns (the raw survey trips) ==="
head -1 initial_trips_hhts.csv | tr ',' '\n' | nl | head -80
echo
echo "=== any transit/access-like column in that header? ==="
head -1 initial_trips_hhts.csv | tr ',' '\n' | grep -inE "transit|access|bus|train|pt|mode|walk|cycle|bike|public" | head -30