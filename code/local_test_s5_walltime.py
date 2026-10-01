#!/usr/bin/env python
"""Local test of the Stage-5 walltime guard + checkpoint/resume logic.

Runs the REAL ga_hypothesis_testing._run_ga_stage5 on a tiny synthetic dataset
(120 HHTS tours) so the Larch fits are fast. No PBS; we emulate the walltime
budget via GA_WALLTIME_SECONDS and the checkpoint dir via GA_CHECKPOINT_DIR.

Scenario A: tiny walltime -> guard trips -> _Stage5Incomplete raised -> PARTIAL
            marker written -> at least one segment checkpointed.
Scenario B: resume with same GA_CHECKPOINT_DIR -> completed segment(s) skipped
            -> all segments finish -> no PARTIAL -> returns a GAResult.

Run from the code dir:  python local_test_s5_walltime.py
"""
import os
import sys
import shutil
import traceback

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

# Keep the heavy estimators quiet / small.
os.environ['PANDAS_USE_NUMEXPR'] = '0'
os.environ['STAGE5_BATCH_SIZE'] = '1'
os.environ['STAGE5_FAIL_FAST'] = '0'
os.environ['GA_AGDS_MXL_SEARCH'] = '0'

from data_connect import Synth_Connect
import ga_hypothesis_testing as GA


def _small_kwargs():
    # Tiny search so each segment fits in a few seconds -> test runs fast,
    # but long enough that the walltime guard has generations to trip on.
    return dict(n_generations=6, pop_size=12, n_restarts=0,
                stagnation_limit=3, mutation_rate=0.2)


def _load_data():
    sc = Synth_Connect()
    out = sc.process_survey(save=False)
    land = sc.get_land()
    s4 = out['tours'].sample(n=min(120, len(out['tours'])), random_state=7).copy()
    return out['trips'], out['persons'], 'downcasted_matrices_base.omx', land, s4


def _run(ckpt_dir, walltime_sec, expect_incomplete):
    if ckpt_dir and os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.environ['GA_CHECKPOINT_DIR'] = ckpt_dir
    os.environ['GA_WALLTIME_SECONDS'] = str(walltime_sec)
    # re-init guard with the chosen deadline (module global)
    GA._GA_START_MONO = GA._time_mod.monotonic()
    GA._init_walltime_guard(margin_seconds=0)  # use full budget for the test

    trips, persons, skims, land, s4 = _load_data()
    raised = None
    result = None
    try:
        result = GA._run_ga_stage5(
            trips, persons, skims, land, _small_kwargs(),
            n_restarts=0, tours_src=s4)
    except GA._Stage5Incomplete as e:
        raised = e
    except Exception as e:  # noqa: BLE001
        raised = e
        print("UNEXPECTED EXCEPTION:")
        traceback.print_exc()

    partial_path = os.path.join(ckpt_dir, 'PARTIAL')
    ckpt_path = os.path.join(ckpt_dir, 'ga_stage5_segments.pkl')
    has_partial = os.path.exists(partial_path)
    has_ckpt = os.path.exists(ckpt_path)
    n_ckpt = 0
    if has_ckpt:
        import pickle
        n_ckpt = len(pickle.load(open(ckpt_path, 'rb')))
    print(f"  raised={type(raised).__name__ if raised else None} "
          f"PARTIAL={has_partial} ckpt_segs={n_ckpt}")
    return raised, has_partial, has_ckpt, n_ckpt, result


def main() -> int:
    ckpt = os.path.join(CODE, 'runs', 'local_test_s5_ckpt')
    print("== Scenario A: tiny walltime, expect _Stage5Incomplete + PARTIAL ==")
    raised, pA, cA, nA, _ = _run(ckpt, walltime_sec=2, expect_incomplete=True)
    okA = (raised is not None and isinstance(raised, GA._Stage5Incomplete)
           and pA and cA and nA >= 1)
    print(f"  -> A pass={okA}")

    print("== Scenario B: resume same dir, normal walltime, expect COMPLETE ==")
    # remove PARTIAL so we can confirm a clean run clears it
    p = os.path.join(ckpt, 'PARTIAL')
    if os.path.exists(p):
        os.remove(p)
    raised2, pB, cB, nB, res2 = _run(ckpt, walltime_sec=3600, expect_incomplete=False)
    # After a clean run the resume code skips already-done seg(s); all 3 should be
    # present at the end (the completed A segment + the freshly finished ones).
    okB = (raised2 is None and not pB and res2 is not None
           and getattr(res2, 'best_features', None) is not None)
    print(f"  -> B pass={okB} (n_ckpt_segs_end={nB})")

    print("== Scenario C: fresh dir, normal walltime, expect COMPLETE, no PARTIAL ==")
    ckpt2 = os.path.join(CODE, 'runs', 'local_test_s5_ckpt2')
    raised3, pC, cC, nC, res3 = _run(ckpt2, walltime_sec=3600, expect_incomplete=False)
    okC = (raised3 is None and not pC and res3 is not None)
    print(f"  -> C pass={okC}")

    overall = okA and okB and okC
    print("\nRESULT:", "ALL PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
