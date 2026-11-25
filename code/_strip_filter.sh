#!/bin/bash
# filter-branch --index-filter body for _strip_big_blobs_for_push.sh.
# Untracks every index entry whose BLOB ID is in $BIG_BLOBS, whatever its path,
# and logs the removed paths to $REMOVED_PATHS.
git -c core.quotepath=off ls-files -s \
  | awk 'NR==FNR { big[$1]; next } ($2 in big) { sub(/^[^\t]*\t/, ""); print }' \
        "$BIG_BLOBS" - \
  | tee -a "$REMOVED_PATHS" \
  | xargs -r -d '\n' git --literal-pathspecs rm --cached -q --
