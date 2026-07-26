"""
profile_report.py
=================
Print a readable hotspot report from one or more cProfile .pstats files — top-N
by cumulative and self time, plus a project-only view (stage5/stage6/ga_
hypothesis search/read_skims) so the real hotspots stand out. Works for any
.pstats (prediction or SA/GA estimation).

Pass MULTIPLE .pstats to combine them into one whole-chain view — e.g. after a
GA_PROFILE=1 run of the parallel chain, aggregate every stage:
    python profile_report.py profile_estimation_stage*.pstats
A single trailing integer is treated as top_n:
    python profile_report.py profile_estimation_stage5_prof5.pstats 60
"""
import io
import pstats
import sys

PROJECT_HINTS = ('stage5_destination_choice', 'stage6_plan_finalisation',
                 'stage6_mode_choice', 'stage5b_trip_destination',
                 'ga_hypothesis_testing', 'ga_external_estimators',
                 'read_skims', 'larch', 'compute_accessibility')


def report(pstats_paths, top_n: int = 40) -> None:
    paths = [pstats_paths] if isinstance(pstats_paths, str) else list(pstats_paths)
    scope = paths[0] if len(paths) == 1 else f"{len(paths)} combined .pstats"
    print(f"Profiling report over: {scope}")
    for sort_key, title in (('cumulative', 'TOP BY CUMULATIVE TIME (incl. callees)'),
                            ('tottime', 'TOP BY SELF TIME (tottime, excl. callees)')):
        s = io.StringIO()
        # pstats.Stats(*paths) merges multiple profiles into one aggregate.
        pstats.Stats(*paths, stream=s).sort_stats(sort_key).print_stats(top_n)
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{s.getvalue()}")

    s = io.StringIO()
    pstats.Stats(*paths, stream=s).sort_stats('tottime').print_stats(
        '|'.join(h.replace('.', r'\.') for h in PROJECT_HINTS))
    print(f"\n{'=' * 78}\nPROJECT-ONLY (stage5/6, ga_ search, larch, read_skims) "
          f"by self time\n{'=' * 78}\n{s.getvalue()}")


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    # A trailing bare integer is top_n; everything else is a .pstats path.
    top_n = 40
    if len(args) > 1 and args[-1].isdigit():
        top_n = int(args[-1])
        args = args[:-1]
    report(args, top_n)
