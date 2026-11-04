#!/usr/bin/env python
"""
verify_logsum_accessibility.py — PT propagation check for fitted logsum accessibility.
================================================================================
Verifies that the unified logsum-accessibility chain (Stage-5 fitted mode
models -> compute_logsum_accessibility -> Stage 1 frequency + Stage 5
c_logsum) actually propagates the PT access alternatives (PT_Walk=6,
PT_Drive=7, PT_Bike=8) when STAGE5_PT_MODES=1:

  1. environment + mode-map preflight (STAGE5_PT_MODES=1, MODE_CODES 6/7/8);
  2. fitted models carry the PT coefficients (ASC_PT_*, c_ivt_pt_*);
  3. per-purpose surface diagnostics (coverage, finiteness, flatness) from
     compute_logsum_accessibility(...).attrs['diagnostics'];
  4. paired PT-on/off nest invariant per purpose (adding alternatives can
     only raise the inclusive value; acc_pt >= acc_off), via
     compute_logsum_accessibility(..., verify_pt_monotonicity=True);
  5. descriptive diff against a pre-PT baseline CSV (if present) — the
     refit changes all coefficients, so deltas are reported, not asserted;
     an IDENTICAL surface still warns (likely PT terms not engaged).

Run (login node OK, a few minutes; requires the pinned SearchLibrium env):
  STAGE5_PT_MODES=1 PIPELINE_SKIMS_PATH=../demand_/skims_371.omx \
      python verify_logsum_accessibility.py \
      --models saved_data/stage5_models.pkl

--models must contain fitted model objects (save_models output). A
merge_ga_shards.py ga_results.pkl carries feature names only and will fail
the _find_mode_models scan.

Exit code: 0 = all invariants pass; 1 = breach/flat/missing-PT-coefs.
Outputs: saved_data/logsum_acc_371_pt.csv (--out; disable with --no-write).
"""
import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("verify_logsum")

# The page's whole purpose is the PT surface: default the flag ON before any
# stage5 import (the flag is read at module import time).
os.environ.setdefault("STAGE5_PT_MODES", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
LANDUSE = os.environ.get(
    "LANDUSE_371", os.path.join(_HERE, "saved_data", "zone_landuse_overture.csv"))
SKIMS = os.environ.get("SKIMS_371", os.environ.get(
    "PIPELINE_SKIMS_PATH", os.path.join(_HERE, "..", "demand_", "skims_371.omx")))
OUT = os.environ.get(
    "LOGSUM_OUT_PT", os.path.join(_HERE, "saved_data", "logsum_acc_371_pt.csv"))
BASELINE = os.environ.get(
    "LOGSUM_BASELINE", os.path.join(_HERE, "saved_data", "logsum_acc_371.csv"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.environ.get("MODELS_371"),
                    help="pickle with fitted PT stage-5 models (required)")
    ap.add_argument("--landuse", default=LANDUSE)
    ap.add_argument("--skims", default=SKIMS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--no-write", action="store_true",
                    help="skip writing the output CSV")
    ap.add_argument("--skip-monotonicity", action="store_true",
                    help="skip the paired PT-on/off invariant (halves cost)")
    args = ap.parse_args()

    if not args.models or not os.path.exists(args.models):
        raise SystemExit(
            "FATAL: --models pickle with fitted PT stage-5 models required "
            "(e.g. saved_data/stage5_models.pkl). Feature-name GA results "
            "pickles are not a valid source.")
    if not os.path.exists(args.landuse):
        raise SystemExit(f"FATAL: landuse not found: {args.landuse}")

    # ── preflight: env + mode map (import AFTER env set) ─────────────────
    import stage5_destination_choice as sdc
    if not getattr(sdc, "STAGE5_PT_MODES", False):
        raise SystemExit(
            "FATAL: STAGE5_PT_MODES is OFF in this interpreter — export "
            "STAGE5_PT_MODES=1 before python starts (the flag is read at "
            "stage5_destination_choice import time).")
    mode_codes = list(getattr(sdc, "MODE_CODES", []))
    if not all(c in mode_codes for c in (6, 7, 8)):
        raise SystemExit(
            f"FATAL: MODE_CODES={mode_codes} lacks 6/7/8 (PT modes) — the "
            "PT mode map did not activate; check the env and module import.")
    log.info("mode_codes=%s  STAGE5_PT_MODES=1  skims=%s", mode_codes, args.skims)

    # ── fitted models ────────────────────────────────────────────────────
    from build_fitted_logsum_371 import _find_mode_models
    with open(args.models, "rb") as f:
        blob = pickle.load(f)
    models = _find_mode_models(blob)
    if not models:
        raise SystemExit(
            f"FATAL: no {{segment: {{'mode_model': ...}}}} mapping found in "
            f"{args.models}.")
    log.info("fitted mode models for segments: %s", sorted(models))

    # ── PT coefficients present in the fit? ──────────────────────────────
    _pt_keys = ("ASC_PT_Walk", "ASC_PT_Drive", "ASC_PT_Bike",
                "c_ivt_pt_drive", "c_ivt_pt_bike")
    seg_pt = {}
    for seg, sd in models.items():
        m = sd.get("mode_model")
        if m is None:
            seg_pt[seg] = None
            continue
        try:
            cd = sdc._coef_dict_from_model(m)
        except Exception as e:
            log.warning("segment '%s': coefficient read failed (%s)", seg, e)
            seg_pt[seg] = None
            continue
        present = [k for k in _pt_keys if k in cd]
        seg_pt[seg] = present
    n_seg_pt = sum(1 for v in seg_pt.values() if v)
    for seg, present in sorted(seg_pt.items(), key=lambda kv: str(kv[0])):
        if present:
            log.info("  [%s] PT coefs: %s", seg, ", ".join(present))
        else:
            log.warning("  [%s] NO PT coefficients — was this model refit "
                        "with STAGE5_PT_MODES=1?", seg)
    if n_seg_pt == 0:
        raise SystemExit(
            "FATAL: none of the fitted segments carry PT coefficients "
            "(ASC_PT_*/c_ivt_pt_*) — run the refit with STAGE5_PT_MODES=1 "
            "before verifying propagation.")

    # ── compute (with paired PT-on/off invariant) ────────────────────────
    from logsum_accessibility import compute_logsum_accessibility
    lu = pd.read_csv(args.landuse)
    if "ORDER" not in lu.columns:
        if "SA2_MAIN16" not in lu.columns:
            raise SystemExit(f"FATAL: {args.landuse} lacks ORDER/SA2_MAIN16")
        lu["ORDER"] = pd.to_numeric(lu["SA2_MAIN16"], errors="coerce").astype("Int64")

    acc = compute_logsum_accessibility(
        models, lu, args.skims,
        verify_pt_monotonicity=not args.skip_monotonicity)
    acc_cols = [c for c in acc.columns if str(c).endswith("_acc")]
    if not acc_cols:
        raise SystemExit("FATAL: compute_logsum_accessibility returned no "
                         "*_acc columns.")

    diag = dict(acc.attrs.get("diagnostics", {}) or {})
    flat_cols, pt_viol_cols = [], []
    for c in acc_cols:
        purp = str(c)[len("logsum_"):-len("_acc")]
        d = diag.get(purp, {})
        if d.get("flat_frac", 1.0) > 0.5 or d.get("std", 0.0) < 1e-9:
            flat_cols.append(c)
        if d.get("pt_monotonicity", {}).get("n_violations"):
            pt_viol_cols.append(c)

    # ── baseline comparison (descriptive only) ───────────────────────────
    base_note = "baseline missing — skipped"
    if args.baseline and os.path.exists(args.baseline):
        try:
            b = pd.read_csv(args.baseline)
            key_b = "zone_id" if "zone_id" in b.columns else b.columns[0]
            key_a = "ORDER" if "ORDER" in acc.columns else acc.columns[0]
            b = b.rename(columns={key_b: "zone_id"})
            a = acc.rename(columns={key_a: "zone_id"})
            b["zone_id"] = pd.to_numeric(b["zone_id"], errors="coerce").astype("Int64")
            a["zone_id"] = pd.to_numeric(a["zone_id"], errors="coerce").astype("Int64")
            j = a.merge(b, on="zone_id", how="inner", suffixes=("_pt", "_base"))
            common = [c for c in acc_cols if f"{c}_base" in j.columns]
            rows = []
            _identical_all = True
            for c in common:
                d = (pd.to_numeric(j[f"{c}_pt"], errors="coerce")
                     - pd.to_numeric(j[f"{c}_base"], errors="coerce")).dropna()
                if d.empty:
                    continue
                _mad = float(np.abs(d).max())
                _identical_all = _identical_all and _mad < 1e-12
                rows.append({
                    "col": c, "n": int(d.size),
                    "corr": float(j[f"{c}_pt"].corr(j[f"{c}_base"])),
                    "med_delta": float(d.median()),
                    "p05": float(d.quantile(0.05)),
                    "p95": float(d.quantile(0.95)),
                    "max_abs": _mad,
                })
            if rows:
                log.info("baseline diff vs %s (%d zones):", args.baseline, len(j))
                for r in rows:
                    log.info("  %-24s n=%d corr=%.4f med=%+.4g [p05=%+.4g, "
                             "p95=%+.4g] max|d|=%.4g",
                             r["col"], r["n"], r["corr"], r["med_delta"],
                             r["p05"], r["p95"], r["max_abs"])
                if _identical_all:
                    log.warning(
                        "baseline surface is IDENTICAL to the PT surface — "
                        "the PT alternatives are not moving the logsum "
                        "(or the baseline already came from a PT fit).")
                base_note = f"compared {len(rows)} columns vs {args.baseline}"
            else:
                base_note = "baseline has no overlapping *_acc columns — skipped"
        except Exception as e:
            base_note = f"baseline comparison failed: {e}"
            log.warning("%s", base_note)
    else:
        log.info("baseline %s not found — comparison skipped (first PT run)",
                 args.baseline)

    # ── write + verdict ──────────────────────────────────────────────────
    if not args.no_write:
        out = acc[[("ORDER" if "ORDER" in acc.columns else acc.columns[0])]
                  + acc_cols].copy()
        out = out.rename(columns={out.columns[0]: "zone_id"})
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.to_csv(args.out, index=False)
        log.info("wrote %d zones x %d columns -> %s",
                 len(out), len(acc_cols), args.out)

    log.info("summary: %d purposes computed; flat=%d %s; "
             "PT-monotonicity breaches=%d %s; %s",
             len(acc_cols), len(flat_cols), flat_cols or "",
             len(pt_viol_cols), pt_viol_cols or "", base_note)
    if flat_cols or pt_viol_cols or n_seg_pt == 0:
        log.error("VERDICT: FAIL — accessibility propagation is not clean.")
        return 1
    log.info("VERDICT: PASS — PT propagation through logsum accessibility "
             "is consistent (flat=0, monotonicity intact, PT coefs present "
             "in %d/%d segments).", n_seg_pt, len(models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
