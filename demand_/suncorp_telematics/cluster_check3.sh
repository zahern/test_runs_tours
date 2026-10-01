#!/bin/bash
# Inspect the installed main_hpc.py around the grouped-draws allocation, so a
# minimal patch can be applied instead of swapping in a whole different build
# (installed 1.0.153 is 7171 lines, the source tree is 6067 -- they are NOT
# interchangeable).
MP=/home/ahernz/.local/lib/python3.11/site-packages/metacountregressor/main_hpc.py
echo "=== version metadata ==="
cat /home/ahernz/.local/lib/python3.11/site-packages/metacountregressor/_version.py 2>/dev/null
echo
echo "=== every draws_g assignment / generate call ==="
grep -n 'draws_g' "$MP" | head -20
echo
echo "=== context around the generation call ==="
LN=$(grep -n 'draws_g = generate_halton_normal' "$MP" | head -1 | cut -d: -f1)
echo "line: ${LN:-<not found>}"
if [ -n "$LN" ]; then
  echo "--- $((LN-8)) .. $((LN+6)) ---"
  sed -n "$((LN-8)),$((LN+6))p" "$MP" | cat -n | sed "s/^/    /"
fi
echo
echo "=== is G / group_ids available at that point? ==="
grep -n 'group_codes is not None\|G = \|G=\|n_groups' "$MP" | head -12
echo
echo "=== is there already a group-aware version anywhere (double fix)? ==="
grep -n '_g_draws\|per group' "$MP" | head
echo "(empty above = not yet fixed)"
