#!/usr/bin/env python
"""
Standalone local validation of the mode model DE pre-fit.

Run this on your local machine BEFORE submitting to the HPC.  It:
  1. Loads real data (same path as the full pipeline)
  2. Builds the Larch nested-logit mode model for each requested segment
  3. Runs _prefit_mode_model (Steps 1-3: Mu reset, ASC log-share, DE search)
     WITHOUT running the full maximize_loglike - so it only takes minutes,
     not hours
  4. Prints the LL improvement from DE so you can judge whether the pre-fit
     is finding a meaningfully better starting basin
  5. Saves the found parameters to  runs/mode_prefit_warmstart.json
     which the HPC run can load as a warm-start (pipeline_logger already
     passes coef_warm_start to run_stage5)

Usage
-----
# With DE (default):
    python _test_mode_prefit.py

# With DE disabled (just Mu-reset + log-share, fast):
    $env:STAGE5_MODE_DE_PREFIT="0"; python _test_mode_prefit.py

# Specific segments only:
    python _test_mode_prefit.py work edu

# Use more DE budget (slower, more thorough):
    $env:STAGE5_PREFIT_MAXITER="20"; $env:STAGE5_PREFIT_POPSIZE="6"
    python _test_mode_prefit.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ── Engine preference: Numba on Windows (avoids XLA hwloc errors) ─────────
if sys.platform == 'win32' or os.name == 'nt':
    os.environ.setdefault('GA_LARCH_ENGINE', 'numba')

os.environ.setdefault('STAGE5_MODE_DE_PREFIT', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# Pull budget overrides from env or use small defaults for local validation
_MAXITER  = int(os.environ.get('STAGE5_PREFIT_MAXITER', '10'))
_POPSIZE  = int(os.environ.get('STAGE5_PREFIT_POPSIZE', '4'))

# ── Patch differential_evolution call to respect env overrides ────────────
# (the main function uses hardcoded defaults; override by monkeypatching)
import scipy.optimize as _so
_orig_de = _so.differential_evolution
def _patched_de(func, bounds, maxiter=None, popsize=None, **kw):
    return _orig_de(func, bounds,
                    maxiter=maxiter if maxiter is not None else _MAXITER,
                    popsize=popsize if popsize is not None else _POPSIZE,
                    **kw)
_so.differential_evolution = _patched_de

from data_connect import Synth_Connect
from stage5_destination_choice import (
    _build_mode_model,
    _mode_tree,
    _prefit_mode_model,
    MODE_MAP,
)

SEGMENTS = sys.argv[1:] if len(sys.argv) > 1 else ['work', 'edu', 'nonmand']
WARMSTART_PATH = os.path.join('runs', 'mode_prefit_warmstart.json')
os.makedirs('runs', exist_ok=True)


def _segment_filter(tours: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Mirror the segment-filter logic used in _run_segment."""
    col = next((c for c in ['segment', 'tour_type', 'purpose'] if c in tours.columns), None)
    if col is None:
        return tours
    seg_map = {
        'work':    ['work', 'Work', 'HBW', 'hbw'],
        'edu':     ['edu', 'school', 'School', 'HBE', 'hbe'],
        'nonmand': ['nonmand', 'other', 'Other', 'NHBO', 'nhbo', 'HBO', 'hbo'],
    }
    allowed = seg_map.get(segment, [segment])
    return tours[tours[col].isin(allowed)].copy()


def _run_prefit_for_segment(segment, tours, persons, landuse, skims):
    seg = _segment_filter(tours, segment)
    if len(seg) == 0:
        print(f"  [{segment}] No tours - skipping")
        return {}

    # Ensure ptour_id exists and is unique
    if 'ptour_id' not in seg.columns:
        seg['ptour_id'] = (seg.get('person_id', pd.Series(range(len(seg)))).astype(str)
                           + '_' + np.arange(len(seg)).astype(str))
    if seg['ptour_id'].duplicated().any():
        seg = seg.copy()
        seg['ptour_id'] = (seg['person_id'].astype(str) + '_'
                           + np.arange(len(seg)).astype(str))

    # Ensure tour_mode exists for log-share calibration
    if 'tour_mode' not in seg.columns:
        print(f"  [{segment}] WARNING: tour_mode not found - ASC log-share step will be skipped")

    t0 = time.time()
    print(f"\n[{segment}] Building mode DataTree + model ({len(seg)} tours) ...")
    dt_m     = _mode_tree(seg, persons, landuse, skims)
    mode_mdl = _build_mode_model(dt_m, segment)

    try:
        mode_mdl.doctor(repair_ch_av="+")
    except Exception:
        pass

    # Snapshot initial LL (before any pre-fit)
    try:
        ll_before_prefit = float(mode_mdl.loglike())
    except Exception as e:
        print(f"  [{segment}] Could not compute initial LL: {e}")
        ll_before_prefit = float('nan')

    # Snapshot starting parameter values
    pf_before = mode_mdl.pf[['value']].copy()

    # ── Run the full pre-fit (Steps 1, 2, 3) ──────────────────────────────
    de_on = os.environ.get('STAGE5_MODE_DE_PREFIT', '1').strip() not in ('0', 'false', 'False', 'no')
    print(f"  [{segment}] Running pre-fit "
          f"({'Mu-reset + log-share + DE' if de_on else 'Mu-reset + log-share only'}) ...")
    mode_mdl = _prefit_mode_model(mode_mdl, seg, persons, skims, segment)

    # Snapshot LL after pre-fit
    try:
        ll_after_prefit = float(mode_mdl.loglike())
    except Exception:
        ll_after_prefit = float('nan')

    elapsed = time.time() - t0

    # ── Print results ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [{segment}] Pre-fit results  ({elapsed:.1f}s)")
    print(f"  LL before pre-fit : {ll_before_prefit:>10.2f}")
    print(f"  LL after  pre-fit : {ll_after_prefit:>10.2f}  "
          f"({'↑' if ll_after_prefit > ll_before_prefit else '─'}"
          f"{ll_after_prefit - ll_before_prefit:+.2f})")

    # Show parameter changes
    key_params = ['ASC_SR', 'ASC_Walk', 'ASC_Bike', 'ASC_Transit',
                  'Mu_Motor', 'Mu_NonMotor']
    print(f"\n  {'Parameter':<20} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    changed = {}
    for p in key_params:
        if p not in mode_mdl.pf.index:
            continue
        v_after  = float(mode_mdl.pf.loc[p, 'value'])
        v_before = float(pf_before.loc[p, 'value']) if p in pf_before.index else float('nan')
        delta = v_after - v_before
        print(f"  {p:<20} {v_before:>8.4f} {v_after:>8.4f} {delta:>+8.4f}")
        if abs(delta) > 0.001:
            changed[p] = v_after
    print(f"{'─'*60}")

    # Check for Mu at boundary (warning)
    for mu_p in ('Mu_Motor', 'Mu_NonMotor'):
        if mu_p not in mode_mdl.pf.index:
            continue
        v = float(mode_mdl.pf.loc[mu_p, 'value'])
        if v >= 0.93:
            print(f"  ⚠  [{segment}] {mu_p}={v:.4f} still near upper bound after pre-fit "
                  "(post-estimation Mu holdfast will apply during HPC run)")

    # Log observed vs predicted mode shares after pre-fit
    if 'tour_mode' in seg.columns:
        obs_counts = seg['tour_mode'].value_counts()
        obs_total  = float(obs_counts.sum())
        obs_shares = {int(k): float(v) / obs_total for k, v in obs_counts.items()}
        mode_names = {MODE_MAP[m]: m for m in MODE_MAP}
        print(f"\n  Mode shares after pre-fit ASC calibration:")
        print(f"  {'Mode':<12} {'Observed':>10}")
        for code in sorted(obs_shares):
            print(f"  {mode_names.get(code, str(code)):<12} {obs_shares[code]:>10.3f}")

    # Return the full pf for warm-start export
    result = {}
    for p in mode_mdl.pf.index:
        is_hf = bool(mode_mdl.pf.loc[p].get('holdfast', False))
        if not is_hf:
            v = float(mode_mdl.pf.loc[p, 'value'])
            if np.isfinite(v):
                result[f"{segment}_{p}"] = v  # prefix with segment name
    return result


def main():
    print("=" * 60)
    print("Mode model DE pre-fit validation")
    print(f"Segments : {SEGMENTS}")
    print(f"DE       : {'ON' if os.environ.get('STAGE5_MODE_DE_PREFIT','1') != '0' else 'OFF'}")
    print(f"popsize  : {_POPSIZE}  maxiter: {_MAXITER}")
    print("=" * 60)

    print("\nLoading data ...")
    sc     = Synth_Connect()
    out    = sc.process_survey(save=False)
    land   = sc.get_land()
    tours  = out['tours']
    persons = out['persons']

    import larch
    skims = larch.OMX('downcasted_matrices_base.omx', mode='r')

    all_params = {}
    for seg_name in SEGMENTS:
        try:
            params = _run_prefit_for_segment(seg_name, tours, persons, land, skims)
            all_params.update(params)
        except Exception as e:
            print(f"\n[{seg_name}] FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    # ── Save warm-start JSON ───────────────────────────────────────────────
    if all_params:
        with open(WARMSTART_PATH, 'w') as f:
            json.dump(all_params, f, indent=2)
        print(f"\n{'='*60}")
        print(f"Warm-start values saved → {WARMSTART_PATH}")
        print(f"  {len(all_params)} parameters across {len(SEGMENTS)} segments")
        print("  Pass this file to run_stage5 via coef_warm_start= or set the")
        print("  GA_WARMSTART_PATH env var to auto-load on HPC run.")
        print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
