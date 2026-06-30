"""
Master runner: executes scripts 01-05 in sequence, reports errors and summary.
Script 06 (dashboard) is started separately as it runs a web server.
"""
import subprocess
import sys
import os
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = r"Z:\test_runs_tours\demand_"

SCRIPTS = [
    ("01_build_zone_mapping.py",   "Build SA2 <-> EMME zone crosswalk"),
    ("02_build_hhts_od.py",        "Extract HHTS OD at SA2 level"),
    ("03_build_synth_pop_od.py",   "Build synth pop OD matrices"),
    ("04_build_demand_omx.py",     "Create demand OMX file"),
    ("05_build_zone_polygons.py",  "Create zone demand GeoPackage"),
]

LOG_FILE = os.path.join(OUTPUT_DIR, 'run_all_demand.log')

results = {}

def run_script(name, description):
    script_path = os.path.join(SCRIPT_DIR, name)
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}")
    print(f"  {description}")
    print(f"{'='*70}")

    t0 = time.time()
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,  # Let output flow to stdout
        text=True,
        timeout=7200,  # 2 hour timeout
        env=env,
    )
    elapsed = time.time() - t0

    status = 'SUCCESS' if result.returncode == 0 else 'FAILED'
    results[name] = {
        'status': status,
        'returncode': result.returncode,
        'elapsed_s': round(elapsed, 1),
    }

    print(f"\n  --> {status} in {elapsed:.1f}s (return code: {result.returncode})")
    return result.returncode == 0

# ── Run all scripts ─────────────────────────────────────────────────────────
print("=" * 70)
print("SEQ Travel Demand Integration Pipeline")
print("Master runner: 01_build_zone_mapping -> 05_build_zone_polygons")
print("=" * 70)

all_success = True
for script_name, description in SCRIPTS:
    try:
        ok = run_script(script_name, description)
    except subprocess.TimeoutExpired:
        print(f"  --> TIMEOUT after 2 hours")
        results[script_name] = {'status': 'TIMEOUT', 'returncode': -1, 'elapsed_s': 7200}
        ok = False
    except Exception as e:
        print(f"  --> ERROR: {e}")
        results[script_name] = {'status': f'ERROR: {e}', 'returncode': -1, 'elapsed_s': 0}
        ok = False

    if not ok:
        all_success = False
        # Continue to next script unless critical failure
        # Scripts 02-05 depend on 01; 03 depends on 02; 04 depends on 03; 05 depends on 03+04
        if script_name == '01_build_zone_mapping.py':
            print("  WARNING: Zone mapping failed - downstream scripts may fail too")
        elif script_name == '03_build_synth_pop_od.py':
            print("  WARNING: Synth OD failed - scripts 04/05 may fail too")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print(f"\n{'Script':<40} {'Status':<12} {'Time (s)'}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<40} {r['status']:<12} {r['elapsed_s']}")

# Collect output file stats
print("\n" + "-" * 70)
print("OUTPUT FILES:")
output_files = [
    ('sa2_to_emme_mapping.csv',      'Zone crosswalk (EMME -> SA2)'),
    ('sa2_zones_reference.csv',      'SA2 zone reference table'),
    ('hhts_od_by_sa2_mode.csv',      'HHTS OD by SA2 and mode'),
    ('hhts_od_sa2_pivot.csv',        'HHTS OD pivot table'),
    ('synth_pop_od_sa2.csv',         'Synth pop OD (long format)'),
    ('synth_pop_od_summary.csv',     'Synth pop OD summary by mode'),
    ('synth_pop_demand.omx',         'OMX demand + aggregated skims'),
    ('zone_demand_polygons.gpkg',    'SA2 demand GeoPackage'),
    ('zone_demand_sa2.csv',          'SA2 demand attributes CSV'),
]

for fname, desc in output_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        fsize = os.path.getsize(fpath)
        if fsize > 1024*1024:
            size_str = f"{fsize/(1024*1024):.1f} MB"
        elif fsize > 1024:
            size_str = f"{fsize/1024:.1f} KB"
        else:
            size_str = f"{fsize} B"
        print(f"  [OK] {fname:<40} {size_str:<12} {desc}")
    else:
        print(f"  [MISSING] {fname:<40}            {desc}")

# Save log
log_data = {
    'results': results,
    'all_success': all_success,
}
with open(LOG_FILE, 'w') as f:
    json.dump(log_data, f, indent=2)
print(f"\nLog saved: {LOG_FILE}")

# Print key stats from output CSVs if available
print("\n" + "-" * 70)
print("KEY STATISTICS:")
try:
    import pandas as pd
    mapping_csv = os.path.join(OUTPUT_DIR, 'sa2_to_emme_mapping.csv')
    if os.path.exists(mapping_csv):
        m = pd.read_csv(mapping_csv)
        match_rate = m['sa2_code'].notna().mean() * 100
        print(f"  Zone mapping: {m['sa2_code'].notna().sum()} / {len(m)} EMME zones mapped "
              f"({match_rate:.1f}%), {m['sa2_code'].nunique()} unique SA2s")
except Exception as e:
    print(f"  Could not read mapping stats: {e}")

try:
    hhts_csv = os.path.join(OUTPUT_DIR, 'hhts_od_by_sa2_mode.csv')
    if os.path.exists(hhts_csv):
        h = pd.read_csv(hhts_csv)
        print(f"  HHTS OD: {len(h):,} rows, {h['total_weighted_trips'].sum():,.0f} total weighted trips")
except Exception as e:
    print(f"  Could not read HHTS stats: {e}")

try:
    synth_csv = os.path.join(OUTPUT_DIR, 'synth_pop_od_sa2.csv')
    if os.path.exists(synth_csv):
        s = pd.read_csv(synth_csv)
        print(f"  Synth OD: {len(s):,} rows, {s['demand'].sum():,.0f} total demand")
except Exception as e:
    print(f"  Could not read synth OD stats: {e}")

try:
    omx_file = os.path.join(OUTPUT_DIR, 'synth_pop_demand.omx')
    if os.path.exists(omx_file):
        import h5py
        with h5py.File(omx_file, 'r') as f:
            matrices = list(f['data'].keys())
            shape = f.attrs.get('SHAPE', ['?', '?'])
        fsize_mb = os.path.getsize(omx_file) / (1024*1024)
        print(f"  OMX: {fsize_mb:.1f} MB, shape {shape[0]}x{shape[1]}, "
              f"{len(matrices)} matrices: {matrices}")
except Exception as e:
    print(f"  Could not read OMX stats: {e}")

try:
    gpkg_file = os.path.join(OUTPUT_DIR, 'zone_demand_polygons.gpkg')
    if os.path.exists(gpkg_file):
        import geopandas as gpd
        gdf = gpd.read_file(gpkg_file, layer='sa2_demand')
        fsize_mb = os.path.getsize(gpkg_file) / (1024*1024)
        print(f"  GeoPackage: {fsize_mb:.1f} MB, {len(gdf)} SA2 features")
except Exception as e:
    print(f"  Could not read GeoPackage stats: {e}")

print("\n" + "=" * 70)
status_line = "ALL SCRIPTS SUCCEEDED" if all_success else "SOME SCRIPTS FAILED (see above)"
print(f"FINAL STATUS: {status_line}")
print("=" * 70)
print(f"\nTo launch the dashboard: python {os.path.join(SCRIPT_DIR, '06_demand_dashboard_sa2.py')}")
