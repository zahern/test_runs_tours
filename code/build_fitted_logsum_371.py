#!/usr/bin/env python
"""
build_fitted_logsum_371.py — fitted stage-5 logsum accessibility (371 SA2).
================================================================================
Paper anchor: the upstream-logsum cascade (ActivitySim design) — downstream
expected utility feeds CDAP / frequency / timing / duration choices.

Unlike the skim-based asim_* logsums (no models needed, always in
accessibility.csv), the fitted logsum_*_acc measures need estimated Stage-5
mode models. This builder computes them ONCE from an existing stage-5 fit and
writes saved_data/logsum_acc_371.csv (zone_id + logsum_*_acc), which the GA
_join_zone_accessibility picks up automatically for stages 0/2/3/4 — no
estimation reorder, no per-job stage-5 cost (cf. UNIFIED_LOGSUM_ACC=1, which
re-estimates stage 5 inside the prediction flow instead).

Models source (--models): a pickle holding EITHER a {segment: {'mode_model':
...}} mapping as returned by run_stage5(), OR a GA GAResult/AllStagesResult
pickle, which is scanned recursively for such a mapping. Fails loudly when no
usable mode models are found (run a stage-5 estimation first).

Run (needs fitted stage-5 models + skims; login node OK, a few minutes):
  python build_fitted_logsum_371.py --models runs/ga_parallel/<tag>/merged/ga_results.pkl
Env: MODELS_371, LANDUSE_371, SKIMS_371, LOGSUM_OUT
Output: saved_data/logsum_acc_371.csv
"""
import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("logsum371")

_HERE = os.path.dirname(os.path.abspath(__file__))
LANDUSE = os.environ.get("LANDUSE_371",
                         os.path.join(_HERE, "saved_data", "zone_landuse_overture.csv"))
SKIMS = os.environ.get("SKIMS_371", os.environ.get(
    "PIPELINE_SKIMS_PATH", os.path.join(_HERE, "..", "demand_", "skims_371.omx")))
OUT = os.environ.get("LOGSUM_OUT",
                     os.path.join(_HERE, "saved_data", "logsum_acc_371.csv"))


def _find_mode_models(obj, _depth=0):
    """Recursively scan a pickled object for {segment: {'mode_model': ...}}."""
    if _depth > 4 or obj is None:
        return None
    if isinstance(obj, dict):
        if any(isinstance(v, dict) and 'mode_model' in v for v in obj.values()):
            return {k: v for k, v in obj.items()
                    if isinstance(v, dict) and 'mode_model' in v}
        for v in obj.values():
            found = _find_mode_models(v, _depth + 1)
            if found:
                return found
        # objects with __dict__ (GAResult etc.)
        d = getattr(obj, '__dict__', None)
        if isinstance(d, dict):
            # metadata often holds fitted models under various keys
            for v in d.values():
                found = _find_mode_models(v, _depth + 1)
                if found:
                    return found
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_mode_models(v, _depth + 1)
            if found:
                return found
    return _find_mode_models(getattr(obj, '__dict__', None), _depth + 1) \
        if hasattr(obj, '__dict__') else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.environ.get("MODELS_371"),
                    help="pickle with fitted stage-5 models (required)")
    ap.add_argument("--landuse", default=LANDUSE)
    ap.add_argument("--skims", default=SKIMS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not args.models or not os.path.exists(args.models):
        raise SystemExit(
            "FATAL: --models pickle with fitted stage-5 models required "
            "(e.g. runs/ga_parallel/<tag>/merged/ga_results.pkl). "
            "Run a stage-5 estimation first.")

    with open(args.models, "rb") as f:
        blob = pickle.load(f)
    models = _find_mode_models(blob)
    if not models:
        raise SystemExit(
            "FATAL: no {segment: {'mode_model': ...}} mapping found in "
            f"{args.models}. Store run_stage5() output or check metadata keys.")
    log.info("mode models for segments: %s", sorted(models))

    lu = pd.read_csv(args.landuse)
    if "SA2_MAIN16" not in lu.columns:
        raise SystemExit(f"FATAL: {args.landuse} lacks SA2_MAIN16")
    # 371 convention: ORDER carries SA2 codes (same key compute_accessibility uses)
    if "ORDER" not in lu.columns:
        lu["ORDER"] = pd.to_numeric(lu["SA2_MAIN16"], errors="coerce").astype("Int64")
    zid = pd.to_numeric(lu["ORDER"], errors="coerce").dropna().astype(int).values

    from logsum_accessibility import compute_logsum_accessibility
    acc = compute_logsum_accessibility(models, lu, args.skims)
    acc_cols = [c for c in acc.columns if str(c).endswith("_acc")]
    if not acc_cols:
        raise SystemExit("FATAL: compute_logsum_accessibility returned no *_acc columns")
    key = "ORDER" if "ORDER" in acc.columns else acc.columns[0]
    out = acc[[key] + acc_cols].copy()
    out = out.rename(columns={key: "zone_id"})
    out["zone_id"] = pd.to_numeric(out["zone_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["zone_id"]).drop_duplicates("zone_id")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    log.info("wrote %d zones x %d fitted logsum columns -> %s",
             len(out), len(acc_cols), args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
