"""
profile_report.py
=================
Print a readable hotspot report from a cProfile .pstats file — top-N by
cumulative and self time, plus a project-only view (stage5/stage6/ga_ hypothesis
search/read_skims) so the real hotspots stand out. Works for any .pstats
(prediction or SA/GA estimation).

Usage:
    python profile_report.py profile_estimation_stage5.pstats [top_n]
"""
import io
import pstats
import sys

PROJECT_HINTS = ('stage5_destination_choice', 'stage6_plan_finalisation',
                 'stage6_mode_choice', 'stage5b_trip_destination',
                 'ga_hypothesis_testing', 'ga_external_estimators',
                 'read_skims', 'larch', 'compute_accessibility')


def report(pstats_path: str, top_n: int = 40) -> None:
    for sort_key, title in (('cumulative', 'TOP BY CUMULATIVE TIME (incl. callees)'),
                            ('tottime', 'TOP BY SELF TIME (tottime, excl. callees)')):
        s = io.StringIO()
        pstats.Stats(pstats_path, stream=s).sort_stats(sort_key).print_stats(top_n)
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{s.getvalue()}")

    s = io.StringIO()
    pstats.Stats(pstats_path, stream=s).sort_stats('tottime').print_stats(
        '|'.join(h.replace('.', r'\.') for h in PROJECT_HINTS))
    print(f"\n{'=' * 78}\nPROJECT-ONLY (stage5/6, ga_ search, larch, read_skims) "
          f"by self time\n{'=' * 78}\n{s.getvalue()}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    report(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
