#!/bin/bash
# Fix the over-broad pylogit patch: only Iterable belongs in collections.abc;
# OrderedDict must stay in collections (Python 3.12).
SP="$HOME/miniconda3/envs/zigenv/lib/python3.12/site-packages/pylogit"
grep -rl "from collections.abc import OrderedDict" "$SP" 2>/dev/null | while read -r f; do
  sed -i 's/from collections.abc import OrderedDict/from collections import OrderedDict/' "$f"
  echo "reverted: $f"
done
echo "== final import lines =="
grep -rn "from collections" "$SP" 2>/dev/null
