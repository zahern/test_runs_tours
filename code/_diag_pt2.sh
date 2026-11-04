#!/usr/bin/env bash
cd /mnt/hpccs01/home/ahernz/test_runs_tours/code
echo "=== where is 'class Mode' defined? ==="
grep -rn "class Mode" *.py 2>/dev/null | head
echo
echo "=== choice-set / DataChoice construction ==="
grep -nE "DataChoice|dc\.add|catdef|make_choice|set_altids|choice_co_vars|add_group|new_nodes" stage5_destination_choice.py | head -20
echo
echo "=== the DataTree dc node build (_mode_tree) ==="
grep -nE "def _mode_tree" stage5_destination_choice.py