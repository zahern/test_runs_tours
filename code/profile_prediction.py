"""
profile_prediction.py
======================
Function-level profiling of the full build+predict pipeline (mode models +
destination choice + activity prediction) for the 371-zone system.

Two tiers of profiling exist; use them together:

  1. COARSE (already produced by every run, zero effort):
       <OUT_DIR>/timings.csv / timings.html
     Per-stage wall time from pipeline_logger's @timer decorators. The row
     "Predict Stage 5 — destination + mode (one origin zone at a time)" is the
     destination-prediction cost; "Stage 6 …" is mode/plan finalisation.

  2. FUNCTION-LEVEL (this script):
     Runs run_full_pipeline_dta_tested.main() under cProfile and writes:
       - profile_prediction.pstats   (open in snakeviz / pstats for drill-down)
       - a printed top-N table by cumulative AND self (tottime), with a
         project-only view (stage5/stage6/ga_/read_skims) so the destination
         and prediction hotspots are obvious.

     For a lower-overhead alternative on the REAL long run, prefer sampling with
     py-spy against the live PBS job (see job_profile_prediction.pbs header) —
     cProfile adds noticeable overhead on a multi-hour run.

Run:
    python profile_prediction.py            # full build+predict, profiled
    SNAKEVIZ afterwards:  snakeviz profile_prediction.pstats
"""
import cProfile
import io
import os
import pstats
import sys

PSTATS_OUT = os.environ.get('PROFILE_OUT', 'profile_prediction.pstats')
TOP_N = int(os.environ.get('PROFILE_TOP_N', '40'))
# Modules whose time we most care about (destination + prediction + mode).
PROJECT_HINTS = ('stage5_destination_choice', 'stage6_plan_finalisation',
                 'stage6_mode_choice', 'stage5b_trip_destination',
                 'ga_hypothesis_testing', 'read_skims', 'pipeline_logger',
                 'compute_accessibility')


def _print_report(prof: cProfile.Profile) -> None:
    for sort_key, title in (('cumulative', 'TOP BY CUMULATIVE TIME (incl. callees)'),
                            ('tottime', 'TOP BY SELF TIME (tottime, excl. callees)')):
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats(sort_key).print_stats(TOP_N)
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print(s.getvalue())

    # Project-only view: filter to our modules so the real hotspots stand out.
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats('tottime').print_stats(
        '|'.join(h.replace('.', r'\.') for h in PROJECT_HINTS))
    print(f"\n{'=' * 78}\nPROJECT-ONLY (stage5/stage6/ga_/read_skims), by self time\n{'=' * 78}")
    print(s.getvalue())


def main() -> None:
    # Importing sets the pipeline's env defaults (NEW_POP=1 etc.) at import time.
    import run_full_pipeline_dta_tested as rp

    prof = cProfile.Profile()
    print(">>> profiling run_full_pipeline_dta_tested.main() under cProfile ...")
    prof.enable()
    try:
        rp.main()
    finally:
        prof.disable()
        prof.dump_stats(PSTATS_OUT)
        print(f"\n>>> wrote {PSTATS_OUT} "
              f"({os.path.getsize(PSTATS_OUT) / 1024:.0f} KB) — "
              f"open with:  snakeviz {PSTATS_OUT}")
        _print_report(prof)


if __name__ == '__main__':
    sys.exit(main())
