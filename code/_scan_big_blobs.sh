#!/bin/bash
# List blobs > LIMIT bytes in the commits not yet on origin/master.
cd "$(dirname "$0")/.." || exit 1
LIMIT=${LIMIT:-95000000}
echo "range: $(git rev-list --count origin/master..HEAD) commits"
git rev-list --objects origin/master..HEAD \
  | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk -v L="$LIMIT" '$1=="blob" && $2>L {printf "%d MB\t%s\n", $2/1048576, $3}' \
  | sort -rn
echo "SCAN_DONE"
