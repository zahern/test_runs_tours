#!/usr/bin/env python
"""Run the full MTC benchmark (M0/M1/M2 stage estimation) locally on Windows.

Uses the venv in this folder (Z:/test_runs_tours/code/.venv) and saves every
artifact under a local results dir (default ~/mtc_benchmark/mtc_local).

What it runs (mtc_compare.py, start to finish):
  M0  ActivitySim regress reference aggregates (no fitting)
  M1  Pipeline default Stage-5 spec: mode + destination estimation (Larch)
  M2  GA-searched Stage-5 spec (pop 12 x 10 gens, 1 restart) + refit
Outputs in <outdir>: mtc_metrics.json, mtc_metrics_table.tex (paper table),
  stage5_models_M*.pkl, mtc_tours/persons/landuse.csv, SOURCE.json, run_local.log.

Usage (from this folder):
  .\\.venv\\Scripts\\python.exe .\\run_mtc_benchmark_local.py [--outdir PATH]

The MTC source files (~5 MB) download once from GitHub into <outdir>\\source
(validated + retried by mtc_connect); reruns reuse them. Existing valid files
are pre-seeded from runs/mtc_ab/source to skip the download.
"""
import os
import shutil
import subprocess
import sys
import traceback

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(CODE_DIR, ".venv", "Scripts", "python.exe")

# ── House numerical environment (mirrors job_mtc_ab.pbs; set before imports) ─
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PIPELINE_ACCELERATOR", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")
os.environ.setdefault("GA_ENGINE", "agds")
os.environ.setdefault("GA_MAX_WORKERS", "1")
os.environ.setdefault("GA_PARALLEL_SEGMENTS", "2")
os.environ.setdefault("GA_LARCH_ENGINE", "numba")  # stable on Windows; avoids JAX stalls
# Numba compiles sharrow flows with cache locks. The default cache lives in
# __pycache__ next to the sources (network drive here): a SIGKILLed run can
# leave a stale SMB byte-range lock that parks the next run forever at 0 CPU
# (Executive wait inside msvcrt.locking). Keep the cache on local disk.
os.environ.setdefault("NUMBA_CACHE_DIR",
                      os.path.join(os.path.expanduser("~"), "mtc_benchmark",
                                   ".numba_cache"))
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")  # never block on a figure window
# MTC fixture is tiny by design (33 tours: work=9, edu=8, nonmand=16) while
# _run_segment requires STAGE5_MIN_TOURS (default 30). Lower the gate so all
# three segments estimate; strict mode stays on so real failures still raise.
os.environ.setdefault("STAGE5_MIN_TOURS", "5")
# MTC prototype has 25 TAZ zones; the skim reader defaults to the 371-zone
# QLD system and refuses anything else. Declare the alternate zone count.
os.environ.setdefault("ZONE_SYSTEM_N_ZONES", "25")

MTC_FILES = ("households.csv", "persons.csv", "land_use.csv", "skims.omx",
             "regress_final_trips.csv")
SEED_SOURCE = os.path.join(CODE_DIR, "runs", "mtc_ab", "source")


def parse_args(argv):
    outdir = os.path.join(os.path.expanduser("~"), "mtc_benchmark", "mtc_local")
    i = 0
    while i < len(argv):
        if argv[i] == "--outdir" and i + 1 < len(argv):
            outdir = argv[i + 1]
            i += 2
        else:
            i += 1
    return os.path.abspath(outdir)


def check_venv():
    if not os.path.isfile(VENV_PY):
        sys.exit(f"venv python not found: {VENV_PY}")
    if sys.executable.lower() != VENV_PY.lower():
        print(f"NOTE: not running under the venv ({sys.executable}).")
        print(f"Relaunch with: {VENV_PY} {os.path.abspath(__file__)}")
        print("Continuing anyway in 5 s ...", flush=True)
        import time
        time.sleep(5)


def report_estimators():
    try:
        from importlib.metadata import version as _v
        print(f"metacountregressor (dist): {_v('metacountregressor')}", flush=True)
        print(f"SearchLibrium (dist): {_v('SearchLibrium')}", flush=True)
    except Exception as e:
        print(f"estimator version lookup failed: {e}", flush=True)
    try:
        import larch
        print(f"larch: {getattr(larch, '__version__', '?')}", flush=True)
    except Exception as e:
        sys.exit(f"FATAL: larch import failed: {e!r}")


def seed_source(outdir):
    """Pre-seed <outdir>/source from the shared Z: source dir (validated later
    by mtc_connect.download_mtc_data, which refetches anything stale)."""
    dest = os.path.join(outdir, "source")
    os.makedirs(dest, exist_ok=True)
    for name in MTC_FILES:
        d = os.path.join(dest, name)
        s = os.path.join(SEED_SOURCE, name)
        if not os.path.exists(d) and os.path.exists(s) and os.path.getsize(s) > 0:
            try:
                shutil.copy2(s, d)
                print(f"seeded {name} from shared source", flush=True)
            except Exception as e:
                print(f"seed copy failed for {name}: {e}", flush=True)


def main():
    # Windows console is cp1252: pipeline prints (→, ×, …) would kill the
    # tee loop with UnicodeEncodeError. Replace unencodable chars on screen
    # (the log file stays full UTF-8).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    outdir = parse_args(sys.argv[1:])
    os.chdir(CODE_DIR)
    sys.path.insert(0, CODE_DIR)
    check_venv()
    report_estimators()
    seed_source(outdir)

    log_path = os.path.join(outdir, "run_local.log")
    os.makedirs(outdir, exist_ok=True)

    cmd = [sys.executable, "-u", os.path.join(CODE_DIR, "mtc_compare.py"),
           "--outdir", outdir]
    print(f"running: {' '.join(cmd)}", flush=True)
    print(f"logging to: {log_path}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        rc = proc.wait()
    print(f"mtc_compare exit code: {rc}", flush=True)
    for f in ("mtc_metrics.json", "mtc_metrics_table.tex",
              "stage5_models_M1.pkl", "stage5_models_M2.pkl"):
        p = os.path.join(outdir, f)
        print(f"  {'OK ' if os.path.exists(p) else 'MISSING'} {p}", flush=True)
    if rc != 0:
        print("ARMS FAILED — see run_local.log and mtc_metrics.json "
              "(per-arm ok/error/trace).", flush=True)
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
