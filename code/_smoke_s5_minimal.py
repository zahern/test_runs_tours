#!/usr/bin/env python
"""Minimal Stage 5 smoke with fail-fast traceback."""
import os
import sys

from data_connect import Synth_Connect
from stage5_destination_choice import run_stage5


os.environ['PANDAS_USE_NUMEXPR'] = '0'
os.environ['STAGE5_BATCH_SIZE'] = '1'
os.environ['STAGE5_FAIL_FAST'] = '1'


def main() -> int:
    sc = Synth_Connect()
    out = sc.process_survey(save=False)
    land = sc.get_land()

    # Keep this tiny so the first failing batch appears quickly.
    synth_s4 = out['tours'].sample(n=min(120, len(out['tours'])), random_state=7).copy()

    print(f"SMOKE_MIN: trips={len(out['trips'])}, persons={len(out['persons'])}, s4={len(synth_s4)}")

    try:
        run_stage5(
            out['trips'],
            out['persons'],
            synth_s4,
            'downcasted_matrices_base.omx',
            land,
            tours=out['tours'],
            seed=7,
            save_path='runs/stage5_minimal_models.pkl',
            segments=['work'],
            verbose=False,
        )
    except Exception as exc:
        print(f"SMOKE_MIN_FAIL: {type(exc).__name__}: {exc}")
        return 1

    print("SMOKE_MIN_OK")
    return 0


if __name__ == '__main__':
    sys.exit(main())
