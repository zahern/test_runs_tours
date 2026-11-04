#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== which access skims does _transit_submode_features read? ==="
sed -n '/def _transit_submode_features/,/return features/p' stage5_destination_choice.py | grep -nE "TOTIVT|DTIM|DDIST|features\['" | head -30