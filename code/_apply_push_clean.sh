#!/bin/bash
# Point the current branch at the stripped history from job_strip_for_push.pbs
# and ignore the stripped paths. Files stay on disk (reset --mixed only moves
# the branch and the index). Refuses to run if the branch moved after the
# strip, so a commit made in the meantime is never dropped.
set -euo pipefail
cd "$(dirname "$0")/.."
BK=$(git for-each-ref --sort=-creatordate --format='%(refname:short)' \
       'refs/heads/backup/pre-strip-*' | head -1)
[ -n "$BK" ] || { echo "FAIL: no backup/pre-strip-* branch"; exit 1; }
git rev-parse --verify -q push-clean >/dev/null || { echo "FAIL: no push-clean"; exit 1; }
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "$BK")" ]; then
  echo "FAIL: HEAD moved since the strip ($BK) - re-run job_strip_for_push.pbs"
  exit 1
fi
echo "backup kept: $BK"
git reset --mixed -q push-clean
{
  echo ""
  echo "# Generated data (>95 MB) stripped from the unpushed history before the"
  echo "# 2026-11 push; the files stay on disk, they just are not tracked."
  sed 's#^#/#' code/_big_paths.txt | while IFS= read -r l; do
    grep -qxF -- "$l" .gitignore || echo "$l"
  done
} >> .gitignore
git add .gitignore
git commit -q -F - <<'MSG'
Ignore oversized generated data stripped from history before push

Run outputs, synthetic populations, OSM extracts and telematics results
over 95 MB (GitHub rejects >100 MB). They remain on disk, untracked.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
echo "branch $(git rev-parse --abbrev-ref HEAD) -> $(git rev-parse --short HEAD), $(git rev-list --count origin/master..HEAD) commits ahead of origin/master"
echo APPLY_OK
