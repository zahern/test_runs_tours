#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== _mode_tree head (how alts are declared) ==="
sed -n '4041,4070p' stage5_destination_choice.py
echo
echo "=== alt-name literals inside _mode_tree (4041-4300) ==="
awk 'NR>=4041 && NR<=4300' stage5_destination_choice.py | grep -nE "DA|SR|Walk|Bike|Transit|choice|altern" | head -25