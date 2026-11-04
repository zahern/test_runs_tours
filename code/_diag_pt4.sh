#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== MODE_TT_COL / MODE_MAP / hard-coded mode-name sets ==="
grep -nE "MODE_MAP|MODE_NAMES|TRANSIT_SUBMODE|TRANSIT_ACCESS|5: 'Transit'|5\]" stage5_destination_choice.py | head -40