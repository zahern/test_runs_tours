#!/usr/bin/env python
"""
launch_constraint_gui.py  —  one-click launcher for the constraint GUI
======================================================================
Double-click target (via the .bat or the built .exe). It:

  1. locates a Python interpreter (uses the running one, or finds one when
     frozen into an .exe),
  2. makes sure `streamlit` is installed (installs it on first run),
  3. makes sure `constraint_schema.json` exists (builds a static one if not),
  4. starts `streamlit run constraint_gui.py`, which opens the editor in the
     browser.

Designed to work both as a plain script and when frozen by PyInstaller
(build_gui_exe.bat) — a frozen exe cannot run `sys.executable -m streamlit`
(sys.executable is the exe), so it locates a real interpreter first.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys


def app_dir() -> str:
    """Directory that holds constraint_gui.py (next to this launcher / exe)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def find_python() -> str | None:
    """A real Python interpreter to run Streamlit with."""
    if not getattr(sys, 'frozen', False):
        return sys.executable
    # Frozen: sys.executable is the exe — find a real interpreter.
    for cand in ('python', 'python3', 'py'):
        p = shutil.which(cand)
        if p:
            return p
    for guess in (
        os.path.expandvars(r'%LOCALAPPDATA%\Programs\Python\Python312\python.exe'),
        os.path.expandvars(r'%USERPROFILE%\miniconda3\python.exe'),
        os.path.expandvars(r'%USERPROFILE%\anaconda3\python.exe'),
        r'C:\Python312\python.exe',
    ):
        if os.path.exists(guess):
            return guess
    return None


def _has_module(py: str, mod: str) -> bool:
    try:
        subprocess.run([py, '-c', f'import {mod}'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def main() -> int:
    d = app_dir()
    gui = os.path.join(d, 'constraint_gui.py')
    if not os.path.exists(gui):
        print(f"constraint_gui.py not found next to the launcher ({d}).")
        input("Press Enter to close...")
        return 1

    py = find_python()
    if not py:
        print("Could not find a Python interpreter on this machine.\n"
              "Install Python (python.org) or run from your repo's environment.")
        input("Press Enter to close...")
        return 1

    if not _has_module(py, 'streamlit'):
        print("Installing streamlit (first run only)...")
        try:
            subprocess.run([py, '-m', 'pip', 'install', 'streamlit'], check=True)
        except Exception as exc:
            print(f"Failed to install streamlit automatically: {exc}\n"
                  f"Install it manually:  {py} -m pip install streamlit")
            input("Press Enter to close...")
            return 1

    schema = os.path.join(d, 'constraint_schema.json')
    if not os.path.exists(schema):
        print("Building constraint_schema.json (static)...")
        try:
            subprocess.run([py, os.path.join(d, 'build_constraint_schema.py'),
                            '--no-data'], cwd=d, check=False)
        except Exception:
            pass  # GUI shows a friendly message if the schema is still missing

    print("Starting the Constraint Editor in your browser (Ctrl+C here to stop)...")
    env = dict(os.environ)
    env.setdefault('PYTHONUTF8', '1')
    return subprocess.call(
        [py, '-m', 'streamlit', 'run', gui,
         '--browser.gatherUsageStats', 'false'],
        cwd=d, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
