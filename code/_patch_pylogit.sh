#!/bin/bash
# Patch pylogit 1.0.1 for Python 3.12: collections.Iterable -> collections.abc
SP="$HOME/miniconda3/envs/zigenv/lib/python3.12/site-packages/pylogit"
echo "== occurrences =="
grep -rn "from collections import" "$SP" 2>/dev/null | head -10
echo "== patching =="
grep -rl "from collections import" "$SP" 2>/dev/null | while read -r f; do
  sed -i 's/from collections import /from collections.abc import /' "$f"
  echo "patched: $f"
done
echo "== verify import =="
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate zigenv
python -c "import pylogit, lifelines; print('pylogit', pylogit.__version__, 'lifelines', lifelines.__version__)"
