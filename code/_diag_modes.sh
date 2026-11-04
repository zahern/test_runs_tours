#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== MODE constants / id maps ==="
grep -nE 'MODE_TT_COL|MODE_LABEL|MODE_NAMES|MODE_ID|MODE_MAP|CHOICE_MODES|ALT_MODES' stage5_destination_choice.py | head -20
echo
echo "--- MODE_TT_COL block ---"
sed -n '/^MODE_TT_COL/,/^}/p' stage5_destination_choice.py
echo
echo "=== tour-level structures (two-leg / access-egress) ==="
grep -nE 'access_leg|egress_leg|two_leg|outbound|return_leg|purp_tour|TOUR_TYPES|tour_type' stage5_destination_choice.py | head -20
echo
echo "=== how a tour's mode is chosen (alt set) ==="
grep -nE "alt_choice_co|altids|_ALTS|choice_set|mode_choices" stage5_destination_choice.py | head -20