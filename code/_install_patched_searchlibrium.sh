#!/bin/bash
# Install the patched SearchLibrium modules into zigenv so every job gets the
# new destination features (_fit_frame_mnl, estimate_dest_choice_lc,
# identify_weak_features, interactions, corrected mixture LL) AND the latent
# class rail guards (bound/SE/collapse detection in latent_class.py).
SP="$HOME/miniconda3/envs/zigenv/lib/python3.12/site-packages/SearchLibrium"
SRC_DIR="/home/ahernz/test_runs_tours/code/_patches/SearchLibrium"
FILES=(lc_rail_guard.py latent_class.py mh_choice_estimator.py)

for f in "${FILES[@]}"; do
  SRC="$SRC_DIR/$f"
  DST="$SP/$f"
  if [[ ! -f "$SRC" ]]; then
    echo "FATAL: patched source missing: $SRC"
    exit 1
  fi
  if [[ -f "$DST" ]] && ! cmp -s "$SRC" "$DST"; then
    if [[ ! -f "$DST.qld_orig" ]]; then
      cp -p "$DST" "$DST.qld_orig"
      echo "backed up original -> $DST.qld_orig"
    fi
  fi
  cp -p "$SRC" "$DST"
  echo "installed: $DST"
done

find "$SP" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "installed patched modules:"
ls -la "$SP/mh_choice_estimator.py" "$SP/latent_class.py" "$SP/lc_rail_guard.py"
grep -c "def _fit_frame_mnl" "$SP/mh_choice_estimator.py" || true
grep -c "def estimate_dest_choice_lc" "$SP/mh_choice_estimator.py" || true
grep -c "def _compute_rail_flags" "$SP/latent_class.py" || true
grep -c "def lc_result_rail" "$SP/lc_rail_guard.py" || true
echo "INSTALL DONE"
