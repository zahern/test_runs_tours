"""Stage-5 NaN-poisoning diagnostic (HPC, dependency-light).

Replicates the data joins that feed the stage-5 destination warm-start and
reports exactly where NaNs come from — without running any estimation:

  1. Skim inventory: matrices present, NaN rates per matrix
     (esp. WLK_LOC_WLK_TOTIVT__AM, DIST, DISTWALK/DISTBIKE fallback rates).
  2. Zone key-space overlap: skim mapping vs landuse ORDER vs observed
     tour OTAZ/DTAZ (a mismatch here = 93.8% NaN utilities downstream).
  3. Crash source resolution: what _resolve_default_crash_source finds and
     whether the fixed OMX loader accepts its shape.
  4. cost_auto smoke check: exercises the fixed Series.median path.

Needs only numpy/pandas/openmatrix. Run via job_diag_stage5_nan.pbs.

Usage (on a compute node / login shell with the project env):
    python -u diag_stage5_nan.py [--out diag_stage5_nan_report.txt]
"""
import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
SKIMS = os.environ.get(
    "PIPELINE_SKIMS_PATH",
    os.path.join(CODE_DIR, "..", "demand_", "skims_371.omx"))
LANDUSE_CANDS = [
    os.path.join(CODE_DIR, "zone_order_and_landuse.csv"),
    os.path.join(CODE_DIR, "zone_landuse_ovm.csv"),
]
TOURS_CANDS = [
    os.path.join(CODE_DIR, "tours_dataset.csv"),
    os.path.join(CODE_DIR, "trips_stops_merged.csv"),
]
CRASH_CANDS = [
    os.path.join(CODE_DIR, "downcasted_matrices_base.omx"),
    os.path.join(CODE_DIR, "OD_Crashes.csv"),
    os.path.join(CODE_DIR, "SA2_OD_Crash_Matrix.xlsx"),
]


def log(lines, msg):
    print(msg, flush=True)
    lines.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diag_stage5_nan_report.txt")
    args = ap.parse_args()
    lines = []
    log(lines, "=== stage-5 NaN diagnostic ===")
    log(lines, f"code_dir : {CODE_DIR}")
    log(lines, f"skims    : {SKIMS} (exists={os.path.isfile(SKIMS)})")

    # ── 1. skim inventory ──────────────────────────────────────────────
    try:
        import openmatrix as omx
        cf = omx.open_file(SKIMS, mode="r")
        try:
            mats = cf.listMatrices()
            log(lines, f"skim matrices: {len(mats)}")
            try:
                mappings = cf.listMappings()
                mp = dict(cf.mapping(mappings[0])) if mappings else {}
                skim_zones = sorted(int(z) for z in mp.keys())
                log(lines, f"skim mapping '{mappings[0] if mappings else None}': "
                           f"{len(skim_zones)} zones "
                           f"[{min(skim_zones) if skim_zones else '?'}.."
                           f"{max(skim_zones) if skim_zones else '?'}]")
            except Exception as e:
                skim_zones = []
                log(lines, f"skim mapping unreadable: {e!r}")
            for name in ("DIST", "DISTWALK", "DISTBIKE", "CRASH",
                         "WLK_LOC_WLK_TOTIVT__AM", "SOVTOLL_TIME__AM"):
                if name not in mats:
                    log(lines, f"  {name}: MISSING from OMX")
                    continue
                a = np.asarray(cf[name], dtype=float)
                finite = np.isfinite(a)
                log(lines, f"  {name}: shape={a.shape} "
                           f"nan_frac={1 - finite.mean():.4g} "
                           f"min={np.nanmin(a):.3g} max={np.nanmax(a):.3g}")
                if name == "DISTWALK":
                    fb = float(((a == 30.0)).mean())
                    log(lines, f"    DISTWALK==30 fallback frac={fb:.4g}")
        finally:
            cf.close()
    except Exception:
        log(lines, "skim inventory FAILED:\n" + traceback.format_exc())
        skim_zones = []

    # ── 2. zone key-space overlap ──────────────────────────────────────
    lu_zones = []
    for p in LANDUSE_CANDS:
        if os.path.isfile(p):
            try:
                lu = pd.read_csv(p, nrows=5)
                key = next((c for c in ("ORDER", "zone_id", "DTAZ", "SA2")
                            if c in lu.columns), None)
                if key:
                    full = pd.read_csv(p, usecols=[key])
                    lu_zones = sorted(pd.to_numeric(
                        full[key], errors="coerce").dropna().astype(int).unique().tolist())
                    log(lines, f"landuse {os.path.basename(p)}: key={key} "
                               f"n={len(lu_zones)}")
                    break
            except Exception as e:
                log(lines, f"landuse {p} read failed: {e!r}")
    if not lu_zones:
        log(lines, "landuse: no readable zone key found")
    for p in TOURS_CANDS:
        if os.path.isfile(p):
            try:
                t = pd.read_csv(p, usecols=lambda c: c in ("OTAZ", "DTAZ"))
                ot = sorted(pd.to_numeric(t["OTAZ"], errors="coerce")
                            .dropna().astype(int).unique().tolist())
                dt = sorted(pd.to_numeric(t["DTAZ"], errors="coerce")
                            .dropna().astype(int).unique().tolist())
                log(lines, f"tours {os.path.basename(p)}: n={len(t)} "
                           f"OTAZ uniq={len(ot)} DTAZ uniq={len(dt)}")
                for label, ref, refname in ((ot, skim_zones, "skim"),
                                            (dt, skim_zones, "skim"),
                                            (ot, lu_zones, "landuse"),
                                            (dt, lu_zones, "landuse")):
                    if ref:
                        miss = [z for z in ref if z not in
                                (skim_zones if refname == "skim" else lu_zones)]
                        # tour zones absent from reference:
                        absent = [z for z in label if z not in ref]
                        log(lines, f"  tour-vs-{refname}: {len(absent)} tour zones "
                                   f"absent from {refname} "
                                   f"e.g. {absent[:8]}")
                        if "999" in str(absent[:3]) or 999 in absent[:3]:
                            log(lines, "  NOTE: fringe zone 999 present — "
                                       "must exist in skim mapping or joins go NaN")
                break
            except Exception:
                log(lines, "tours read FAILED:\n" + traceback.format_exc())

    # ── 3. crash source ────────────────────────────────────────────────
    for p in CRASH_CANDS:
        log(lines, f"crash candidate {os.path.basename(p)}: "
                   f"exists={os.path.isfile(p)}")
    try:
        sys.path.insert(0, CODE_DIR)
        import stage5_destination_choice as s5
        src = s5._resolve_default_crash_source()
        log(lines, f"_resolve_default_crash_source -> {src}")
        if src:
            try:
                df = s5._read_crash_od_table(
                    src, zone_ids=(skim_zones or None))
                log(lines, f"crash table OK: {len(df)} OD rows, "
                           f"CRASH max={df['CRASH'].max():.4g}")
            except Exception as e:
                log(lines, f"crash table FAILED: {e!r} "
                           f"(this is the 'All arrays must be...' path)")
    except Exception:
        log(lines, "crash check FAILED:\n" + traceback.format_exc())

    # ── 4. cost_auto median path ───────────────────────────────────────
    try:
        vals = [1, 2, 3, None, 8, 8]
        med = float(pd.Series(pd.to_numeric(vals, errors="coerce")).median())
        log(lines, f"cost_auto median path OK: median({vals})={med}")
    except Exception as e:
        log(lines, f"cost_auto median path FAILED: {e!r}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
