"""
run_crash_nearmiss.py
Master runner: executes scripts 10 -> 13 in order, reports results.
"""
import subprocess
import sys
import os
import time

SCRIPTS_DIR = r"Z:\test_runs_tours\demand_"
OUT_DIR = r"Z:\test_runs_tours\demand_"

scripts = [
    '10_crash_length_skim_507.py',
    '11_crash_length_skim_emme.py',
    '12_nearmiss_zonal_counts.py',
    '13_nearmiss_skim_matrix.py',
]

expected_outputs = [
    'crash_length_skim_507.omx',
    'crash_length_skim_507_summary.csv',
    'crash_length_skim_emme2062.omx',
    'nearmiss_zonal_counts.csv',
    'nearmiss_zonal_summary.csv',
    'nearmiss_temporal_summary.csv',
    'nearmiss_hourly_profile.csv',
    'nearmiss_skim_507.omx',
]

print("=" * 70)
print("CRASH + NEAR-MISS PIPELINE MASTER RUNNER")
print("=" * 70)
total_start = time.time()
results = {}

for script in scripts:
    script_path = os.path.join(SCRIPTS_DIR, script)
    print(f"\n{'='*70}")
    print(f"RUNNING: {script}")
    print(f"{'='*70}")
    t0 = time.time()

    proc = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
    )

    elapsed = time.time() - t0
    exit_code = proc.returncode
    status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
    results[script] = {'status': status, 'elapsed': elapsed, 'exit_code': exit_code}

    print(f"\n  [{status}] {script} completed in {elapsed:.1f}s")

    if exit_code != 0:
        print(f"  ERROR: Script returned exit code {exit_code}")
        print("  Stopping pipeline.")
        break

# Final report
print()
print("=" * 70)
print("PIPELINE RESULTS")
print("=" * 70)
total_elapsed = time.time() - total_start

for script, r in results.items():
    print(f"  {r['status']:20s}  {r['elapsed']:6.1f}s  {script}")

print()
print(f"  Total elapsed: {total_elapsed:.1f}s")

print()
print("OUTPUT FILES:")
for fname in expected_outputs:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  OK  {fname:45s} {size_kb:10.1f} KB")
    else:
        print(f"  MISSING  {fname}")

# Print temporal summary if it exists
temporal_path = os.path.join(OUT_DIR, 'nearmiss_temporal_summary.csv')
if os.path.exists(temporal_path):
    print()
    print("NEAR-MISS TEMPORAL SUMMARY:")
    import csv
    with open(temporal_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    header = list(rows[0].keys()) if rows else []
    # Print header
    print("  " + "  ".join(f"{h:>20}" for h in header))
    for r in rows:
        print("  " + "  ".join(f"{r[h]:>20}" for h in header))

# Print top-10 zonal summary
summary_path = os.path.join(OUT_DIR, 'nearmiss_zonal_summary.csv')
if os.path.exists(summary_path):
    print()
    print("TOP 10 SA2 ZONES BY NEAR-MISS COUNT:")
    import pandas as pd
    df = pd.read_csv(summary_path)
    top10 = df.sort_values('total_events', ascending=False).head(10)
    cols = ['sa2_code', 'sa2_name', 'total_events', 'am_total', 'op_total', 'pm_total', 'ev_total',
            'total_car', 'total_hcv']
    cols = [c for c in cols if c in top10.columns]
    print(top10[cols].to_string(index=False))

print()
print("=" * 70)
all_ok = all(r['exit_code'] == 0 for r in results.values())
if all_ok and len(results) == len(scripts):
    print("ALL SCRIPTS COMPLETED SUCCESSFULLY")
else:
    print("PIPELINE INCOMPLETE OR ERRORS OCCURRED - CHECK OUTPUT ABOVE")
print("=" * 70)
