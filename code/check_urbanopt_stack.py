#!/usr/bin/env python
"""
check_urbanopt_stack.py — full URBANopt/OpenDSS/DISCO readiness check.
===============================================================================
The 371 pipeline (build_power_network_371.py -> run_opendss_371.py ->
power_accessibility.py) runs offline with zero extra installs: the OpenDSS
runner falls back to the labelled proxy engine when OpenDSSDirect is absent.

This checker reports what is missing for the TRUE URBANopt path per
docs.urbanopt.net (workflows + installation pages) and prints the exact
install commands. Read-only: installs nothing.

Checks:
  Ruby `uo` CLI              (urbanopt-cli; install/installation.html)
  Python OpenDSSDirect.py    (pip install OpenDSSDirect.py)
  DiTTo + urbanopt-ditto-reader (opendss workflow converter)
  DISCO + PyDSS              (installation/disco.html; needs Ollama-free HPC python)
  RNM-US (optional synthetic networks; workflows/rnm/rnm.html)

Run:  python check_urbanopt_stack.py
"""
import shutil
import subprocess

DOCS = "https://docs.urbanopt.net"


def _have(cmd):
    return shutil.which(cmd) is not None


def _py_pkg(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def main():
    print("=" * 64)
    print("URBANopt stack readiness (true-OpenDSS path)")
    print("=" * 64)
    rows = [
        ("Ruby `uo` CLI (urbanopt-cli)", _have("uo"),
         "Install Ruby + `gem install urbanopt-cli`; see "
         f"{DOCS}/installation/installation.html"),
        ("Python OpenDSSDirect.py", _py_pkg("opendssdirect"),
         "pip install OpenDSSDirect.py  (true powerflow engine)"),
        ("DiTTo (ditto)", _py_pkg("ditto"),
         "pip install NREL-ditto  (distribution format conversion)"),
        ("urbanopt-ditto-reader", _py_pkg("urbanopt_ditto_reader"),
         "pip install urbanopt-ditto-reader  (URBANopt GeoJSON->OpenDSS)"),
        ("DISCO", _py_pkg("disco"),
         f"See {DOCS}/installation/disco.html (upgrade-cost analysis)"),
        ("PyDSS", _py_pkg("pydss"),
         "pip install pyDSS  (DISCO control layer)"),
        ("RNM-US (optional)", _have("rnm-us") or _py_pkg("rnm"),
         f"See {DOCS}/workflows/rnm/rnm.html (synthetic networks)"),
    ]
    missing = 0
    for label, ok, hint in rows:
        print(f"  [{'OK' if ok else '--'}] {label}")
        if not ok:
            missing += 1
            print(f"       -> {hint}")
    print("-" * 64)
    if missing == 0:
        print("Full stack present: run_opendss_371.py will use engine=opendssdirect.")
    else:
        print(f"{missing} component(s) missing: the pipeline still runs offline "
              "with engine=proxy (labelled, same schema).")
        print("HPC note: `uo` (Ruby) is usually unavailable on compute nodes — "
              "keep the Ruby CLI on a login/dev node and run the Python "
              "build/run/accessibility chain under qsub (see job_power_371.pbs).")


if __name__ == "__main__":
    main()
