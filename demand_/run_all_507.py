"""
run_all_507.py
Master runner for the SEQ demand integration pipeline (507 SA2 zones).
Runs scripts 01-07 in order.
"""
import subprocess
import sys
import os
import time

SCRIPT_DIR = r"Z:\test_runs_tours\demand_"
PYTHON = sys.executable

scripts = [
    "01_zone_crosswalk.py",
    "02_synth_pop_demand.py",
    "03_hhts_od_507.py",
    "04_aggregate_skims.py",
    "05_crash_matrices.py",
    "06_build_integrated_omx.py",
    "07_zone_polygons_crash.py",
]

def run_script(script_name):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        capture_output=False,
        text=True,
        cwd=SCRIPT_DIR
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed (exit code {result.returncode}) after {elapsed:.1f}s")
        return False
    print(f"\n[OK] {script_name} completed in {elapsed:.1f}s")
    return True

if __name__ == '__main__':
    print("SEQ Travel Demand Integration Pipeline -- 507 SA2 Zones")
    print(f"Python: {PYTHON}")
    print(f"Script dir: {SCRIPT_DIR}")

    failed = []
    for script in scripts:
        ok = run_script(script)
        if not ok:
            failed.append(script)
            print(f"ERROR in {script} -- continuing with remaining scripts")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    if failed:
        print(f"Failed scripts: {failed}")
    else:
        print("All scripts completed successfully!")

    print("\nOutputs in Z:\\test_runs_tours\\demand_\\:")
    outputs = [
        "sa2_crosswalk_507.csv",
        "sa2_crosswalk_507_expanded.csv",
        "synth_pop_od_507.csv",
        "synth_pop_summary_507.csv",
        "hhts_od_507.csv",
        "hhts_od_by_mode_507.csv",
        "sa2_skims_507.omx",
        "crash_matrices_507.omx",
        "crash_od_507.csv",
        "demand_integrated_sa2.omx",
        "zone_demand_crash_507.gpkg",
        "zone_summary_507.csv",
    ]
    for out in outputs:
        fpath = os.path.join(SCRIPT_DIR, out)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            if size > 1024**3:
                size_str = f"{size/1024**3:.2f} GB"
            elif size > 1024**2:
                size_str = f"{size/1024**2:.1f} MB"
            else:
                size_str = f"{size/1024:.1f} KB"
            print(f"  [OK] {out} ({size_str})")
        else:
            print(f"  [MISSING] {out}")

    print("\nTo launch dashboard: python 08_demand_dashboard_final.py")
