#!/bin/bash
# Make the not-yet-pushed commits GitHub-safe by dropping blobs > LIMIT bytes.
#
# GitHub rejects any blob over 100 MB, and the unpushed range of
# migrate-zones-371 carries DTA run outputs up to 1.3 GB, so the branch can
# never be pushed as-is. Those commits exist only locally (nothing beyond
# origin/master is on the remote), so rewriting just that range is safe.
#
# Done in a BARE clone so the working tree (with its uncommitted edits) is never
# touched; the original tip is kept as backup/pre-strip-<stamp>. The cleaned
# branch is fetched back as refs/heads/push-clean. Applying it to the working
# branch is a separate, explicit step (see the echo at the end).
set -euo pipefail
REPO=${REPO:-$HOME/test_runs_tours}
LIMIT=${LIMIT:-95000000}
BARE=${BARE:-$HOME/tmp_push_strip.git}
cd "$REPO"
BASE=$(git rev-parse origin/master)
TIP=$(git rev-parse HEAD)
STAMP=$(date +%Y%m%d_%H%M%S)
git branch -f "backup/pre-strip-$STAMP" "$TIP"
echo "base=$BASE tip=$TIP backup=backup/pre-strip-$STAMP"

rm -rf "$BARE"
git clone --bare --quiet "$REPO" "$BARE"
cd "$BARE"
git branch -f push-clean "$TIP"

git rev-list --objects "$BASE..push-clean" \
  | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk -v L="$LIMIT" '$1=="blob" && $2>L { sub(/^[^ ]+ [^ ]+ /, ""); print }' \
  | sort -u > big_paths.txt
N=$(wc -l < big_paths.txt)
echo "$N path(s) carry blobs > $LIMIT bytes:"
sed 's/^/  /' big_paths.txt
cp big_paths.txt "$REPO/code/_big_paths.txt"

if [ "$N" -gt 0 ]; then
  export FILTER_BRANCH_SQUELCH_WARNING=1
  git filter-branch -f --index-filter \
    "xargs -a '$BARE/big_paths.txt' -d '\n' git --literal-pathspecs rm --cached --ignore-unmatch -q --" \
    -- "$BASE..push-clean"
fi

LEFT=$(git rev-list --objects "$BASE..push-clean" \
  | git cat-file --batch-check='%(objecttype) %(objectsize)' \
  | awk -v L="$LIMIT" '$1=="blob" && $2>L' | wc -l)
echo "remaining blobs > $LIMIT: $LEFT"
[ "$LEFT" -eq 0 ] || { echo "FAIL: oversized blobs remain"; exit 1; }

NEW=$(git rev-parse push-clean)
cd "$REPO"
# Copy back only the objects the repo lacks (the rewritten commits and trees;
# every surviving blob is already here). A plain fetch re-packed the whole
# history and pack-objects was OOM-killed (job 28994946).
git --git-dir="$BARE" rev-list --objects --no-object-names "$BASE..push-clean"   | git cat-file --batch-check='%(objectname) %(objecttype)'   | awk '$2=="missing"{print $1}' > "$BARE/missing.txt"
echo "objects to copy back: $(wc -l < "$BARE/missing.txt")"
git --git-dir="$BARE" pack-objects --stdout --window=0 --depth=0 < "$BARE/missing.txt"   | git index-pack --stdin > /dev/null
git update-ref refs/heads/push-clean "$NEW"
git rev-list --objects "$BASE..push-clean" > /dev/null   # connectivity check
echo "push-clean=$NEW (commits: $(git rev-list --count "$BASE..push-clean"))"
echo "Apply with: git reset --mixed push-clean   (moves migrate-zones-371, keeps working files)"
echo STRIP_OK
