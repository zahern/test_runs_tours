#!/usr/bin/env python
"""Compute-node verification of the 2026-10 DTIM/DDIST/XWAIT + stage-5 edits.

Run ONLY via job_verify_edits.pbs (never on the head node):
    qsub job_verify_edits.pbs

Checks, in order (no pipeline runs, no OMX writes):
  1. py_compile of the edited modules           (import-free parse check)
  2. bash -n of the edited PBS scripts          (shell parse check)
  3. pure helpers: _jvm_memory_bytes, checkpoint round-trip, RSS reader
  4. JVM heap plumbing: r5py starts with exactly JVM_MEMORY (this starts a
     JVM on the compute node — that is the point of the test)
  5. PT-flag wiring inside job_submit_ga_parallel_chain.pbs
  6. DDIST presence in the builder + verify block of job_transit_dtim.pbs

Exit 0 iff every check passes; prints RESULT: ALL CHECKS PASSED at the end.
"""
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ── 1. syntax ──────────────────────────────────────────────────────────────
r = subprocess.run(
    [sys.executable, "-m", "py_compile",
     "build_transit_dtim_xwait.py", "stage5_destination_choice.py"],
    cwd=str(HERE), capture_output=True, text=True)
check("py_compile edited modules", r.returncode == 0,
      (r.stderr or r.stdout or "").strip()[-300:])

# ── 2. bash -n ─────────────────────────────────────────────────────────────
for pbs in ("job_transit_dtim.pbs", "job_submit_ga_parallel_chain.pbs"):
    r = subprocess.run(["bash", "-n", pbs], cwd=str(HERE),
                       capture_output=True, text=True)
    check(f"bash -n {pbs}", r.returncode == 0, (r.stderr or "").strip()[-200:])

# ── 3. pure helpers (module import is r5py-free; JVM only starts in step 4) ─
os.environ["JVM_MEMORY"] = "3G"
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
import build_transit_dtim_xwait as b

check("_jvm_memory_bytes 3G", b._jvm_memory_bytes("3G") == 3 * 1024 ** 3)
check("_jvm_memory_bytes 12000M", b._jvm_memory_bytes("12000M") == 12000 * 1024 ** 2)
check("_jvm_memory_bytes raw bytes", b._jvm_memory_bytes("1073741824") == 2 ** 30)
check("JAVA_TOOL_OPTIONS mirrors JVM_MEMORY",
      os.environ.get("JAVA_TOOL_OPTIONS") == "-Xmx3G",
      os.environ.get("JAVA_TOOL_OPTIONS", "(unset)"))

_rss = b._cur_rss_gb()
check("_cur_rss_gb sane", 0 < _rss < 1024, f"{_rss:.2f} GB")

with tempfile.TemporaryDirectory() as td:
    stem = os.path.join(td, "batch_1_6_xd")
    df = pd.DataFrame({"o_id": [1, 2], "d_id": [3, 4], "xwait": [0.0, 7.5],
                       "dtim": [12.0, 0.0], "ddist": [5.2, 0.0]})
    p = b._ckpt_write(df, stem)
    back = b._ckpt_read(stem)
    ok = (back is not None
          and list(back.columns) == ["o_id", "d_id", "xwait", "dtim", "ddist"]
          and float(back["ddist"].iloc[0]) == 5.2)
    check(f"checkpoint round-trip ({b._CKPT_FMT})", ok, str(p))

# ── 4. r5py honours the injected --max-memory (starts a 3G JVM here) ───────
try:
    b._pin_r5py_jvm_memory()
    import r5py.util.memory_footprint as mf
    got = int(mf.MAX_JVM_MEMORY)
    check("r5py heap == JVM_MEMORY", got == 3 * 1024 ** 3, f"{got / 1024 ** 3:.2f} GiB")
except Exception as e:  # pragma: no cover - reported, never raised
    check("r5py heap pin", False, repr(e))

# ── 5. PT wiring in the GA chain submit script ─────────────────────────────
txt = (HERE / "job_submit_ga_parallel_chain.pbs").read_text()
check("chain declares STAGE5_PT_MODES",
      "STAGE5_PT_MODES=${STAGE5_PT_MODES:-0}" in txt)
check("chain declares STAGE5B_TRANSIT_ACCESS",
      "STAGE5B_TRANSIT_ACCESS=${STAGE5B_TRANSIT_ACCESS:-0}" in txt)
n_fwd = txt.count("STAGE5_PT_MODES=${STAGE5_PT_MODES}")
n_fwd_b = txt.count("STAGE5B_TRANSIT_ACCESS=${STAGE5B_TRANSIT_ACCESS}")
check("chain forwards both in stage+predict qsub", n_fwd == 2 and n_fwd_b == 2,
      f"PT_MODES={n_fwd}, TRANSIT_ACCESS={n_fwd_b}")
n_exp = txt.count("export STAGE5_PT_MODES=")
n_exp_b = txt.count("export STAGE5B_TRANSIT_ACCESS=")
check("chain exports both in RESUME_ENV", n_exp == 1 and n_exp_b == 1,
      f"PT_MODES={n_exp}, TRANSIT_ACCESS={n_exp_b}")

# ── 6. DDIST in builder + verify block ─────────────────────────────────────
btxt = (HERE / "build_transit_dtim_xwait.py").read_text()
check("builder writes WLK_LOC_DRV_DDIST__AM",
      "f['WLK_LOC_DRV_DDIST__AM']" in btxt)
check("builder guards strided smoke publish",
      "ALLOW_PARTIAL_PUBLISH" in btxt and "sys.exit(3)" in btxt)
ptxt = (HERE / "job_transit_dtim.pbs").read_text()
check("pbs verifies DDIST", "'WLK_LOC_DRV_DDIST__AM'" in ptxt)

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED — {FAILS}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
