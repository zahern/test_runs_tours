#!/usr/bin/env python
"""Fit one Stage-5 arm on HHTS for the P0/P2 smoke.

P0 = pipeline default (published ActivitySim-equivalent) specification, no
search, no safety terms.
P2 = the searched feature set for the segment, with crash terms retained.

The fitted bundle is saved in the same ``stage5_models.pkl`` format used by
``_score_stage5_arm.py`` so the two arms can be scored on the identical HHTS
holdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Route-choice person-type logsum: both arms use the SAME structural inclusive
# value so the arm comparison isolates specification authority.
os.environ.setdefault(
    "STAGE5_ROUTE_TYPE_TABLE",
    os.path.join("runs", "route_choice_371", "route_logsum_type.parquet"))
os.environ.setdefault("STAGE5_LEGACY_DEST_CRASH", "0")


def _load_spec(path: str, key: str) -> list:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    values = blob.get(key) or []
    return [str(v) for v in values]


def _load_dest_spec(path: str):
    """Return either an explicit ``dest_uec`` dict or a ``dest_vars`` list."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    if isinstance(blob.get("dest_uec"), dict):
        # JSON preserves the two-tuple as a list; _build_dest_utility accepts
        # both, but normalise the shape for readability and equality checks.
        return {
            str(k): tuple(v) if isinstance(v, list) else v
            for k, v in blob["dest_uec"].items()
        }
    values = blob.get("dest_vars") or []
    return [str(v) for v in values]


def _n_cases(model):
    """Number of cases a larch model was estimated on (None if unavailable)."""
    for attr in ("n_cases",):
        try:
            v = getattr(model, attr)
            return int(v() if callable(v) else v)
        except Exception:
            pass
    try:
        return int(model.dataset.sizes[model.dataset.dc.CASEID])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="work")
    ap.add_argument("--arm", required=True, choices=["P0", "P2"])
    ap.add_argument("--mode-spec", default="",
                    help="JSON with a mode_vars list (P2 only)")
    ap.add_argument("--dest-spec", default="",
                    help="JSON with a dest_vars list (P2 only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.30,
                    help="holdout fraction excluded from estimation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="random",
                    choices=["random", "temporal"])
    ap.add_argument("--skims", default=os.environ.get(
        "PIPELINE_SKIMS_PATH", "../demand_/skims_371.omx"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.arm == "P0":
        os.environ["SAFETY_EXPERIMENT_MODE"] = "nosafety"
        os.environ.pop("STAGE5_FORCE_CRASH", None)
        # An explicit empty mode list is meaningful: it selects only the
        # _mode_spec_custom always-on backbone (ActivitySim-equivalent P0).
        mode_vars = _load_spec(a.mode_spec, "mode_vars") if a.mode_spec else None
        dest_vars = _load_dest_spec(a.dest_spec)
    else:
        os.environ["SAFETY_EXPERIMENT_MODE"] = "baseline"
        # The route-logsum design does NOT force a destination crash term;
        # P2's destination safety travels through c_route_logsum_type.
        os.environ["STAGE5_FORCE_CRASH"] = "0"
        os.environ.pop("STAGE5_FORCE_CRASH", None)
        mode_vars = _load_spec(a.mode_spec, "mode_vars") if a.mode_spec else None
        dest_vars = _load_dest_spec(a.dest_spec)

    # A P2 arm without a destination spec silently falls back to the FULL
    # utility (empty list means "no restriction" in the flat UEC builder), which
    # fits the whole backbone with priors and looks like a valid run.  Refuse it.
    if a.arm == "P2" and not dest_vars:
        raise SystemExit(
            f"P2 arm requires a non-empty destination spec; got {a.dest_spec!r}")

    from stage5_destination_choice import _run_segment, save_models
    from stage5_arm_inputs import load_arm_inputs

    # Inputs come from the shared builder so the scorer reconstructs exactly
    # this data, crash surface and walk/bike overlay (see stage5_arm_inputs).
    # Estimate on the 70% person split; the scorer holds out the other 30%.
    print(f"[{a.arm}] loading HHTS + landuse (shared input builder)", flush=True)
    inp = load_arm_inputs(a.segment, a.skims, a.frac, a.seed, a.split)
    seg_tours, seg_persons = inp["est_tours"], inp["est_persons"]
    landuse, skims, close = inp["landuse"], inp["skims"], inp["close"]
    split_used = inp["split"]
    print(f"[{a.arm}] route person types: "
          f"{seg_persons['person_type'].value_counts().to_dict()}", flush=True)
    print(f"[{a.arm}] split={split_used} holdout={inp['n_holdout_persons']:,} "
          f"persons", flush=True)
    print(f"[{a.arm}] {a.segment}: {len(seg_tours)} HHTS tours | "
          f"mode_vars={len(mode_vars or [])} dest_vars={len(dest_vars or [])}",
          flush=True)
    try:
        result = _run_segment(
            a.segment, seg_tours, seg_persons, landuse, skims,
            skim_df=inp["skim_df"], seed=42, verbose=True,
            mode_feature_override=mode_vars,
            dest_feature_override=dest_vars,
        )
    finally:
        try:
            close()
        except Exception:
            pass

    path = os.path.join(a.out, "stage5_models.pkl")
    save_models({a.segment: result}, path)

    mode_diag = dict(result.get("mode_fit_diagnostics") or {})
    dest_diag = dict(result.get("dest_fit_diagnostics") or {})
    mode_coef = dict(result.get("mode_coef") or {})
    dest_coef = dict(result.get("dest_coef") or {})

    # The route-logsum design replaces the single mode logsum with the
    # person-type inclusive value.  Accept either structural term; requiring
    # c_logsum specifically rejected every valid route-logsum fit.
    required_dest = ["c_size_term"]
    missing_required = [k for k in required_dest
                        if k not in dest_coef
                        or abs(float(dest_coef.get(k, 0.0))) < 1e-10]
    _lv_terms = ("c_route_logsum_type", "c_logsum")
    _lv_ok = any(k in dest_coef
                 and abs(float(dest_coef.get(k, 0.0))) > 1e-10
                 for k in _lv_terms)
    required_dest.append("c_route_logsum_type|c_logsum")
    if not _lv_ok:
        missing_required.append("c_route_logsum_type|c_logsum")
    _distance_bases = ("c_log_dist", "c_dist", "c_dist_sq", "c_dist_0_2",
                       "c_dist_2_5", "c_dist_5_10", "c_dist_decay")
    if not any(k in dest_coef
               and abs(float(dest_coef.get(k, 0.0))) > 1e-10
               for k in _distance_bases):
        missing_required.append("distance_basis")
    active_dest = int(sum(abs(float(v)) > 1e-10
                         for v in dest_coef.values()))
    active_mode = int(sum(abs(float(v)) > 1e-10
                         for v in mode_coef.values()))

    meta = {
        "arm": a.arm,
        "segment": a.segment,
        "mode_vars": mode_vars,
        "dest_vars": (list(dest_vars)
                      if isinstance(dest_vars, (list, tuple))
                      else sorted(dest_vars or {})),
        "split": split_used,
        "frac": a.frac,
        "seed": a.seed,
        "n_holdout_persons": inp["n_holdout_persons"],
        "n_estimation_tours": int(len(seg_tours)),
        # What this fit was built on; the scorer rebuilds and checks it.
        "inputs": inp["record"],
        # Cases the fitted mode model actually saw (after quarantine), so the
        # scorer can check it rebuilt the SAME estimation sample before
        # comparing log-likelihoods.
        "mode_n_cases": _n_cases(result.get("mode_model")),
        "dest_sample_size": os.environ.get("STAGE5_DEST_SAMPLE_SIZE", ""),
        "choice_set": ("universal"
                       if str(os.environ.get("STAGE5_DEST_SAMPLE_SIZE", "20"))
                       in ("0", "") else "sampled"),
        "mode_ll": result.get("mode_ll"),
        "dest_ll": result.get("dest_ll"),
        "mode_converged": bool(mode_diag.get("converged")
                               or mode_diag.get("success")),
        "dest_converged": bool(dest_diag.get("converged")
                               or dest_diag.get("success")),
        "mode_polish_ll": (mode_diag.get("polish") or {}).get("ll_after"),
        "dest_polish_ll": (dest_diag.get("polish") or {}).get("ll_after"),
        "n_mode_coef": len(mode_coef),
        "n_dest_coef": len(dest_coef),
        "n_mode_active": active_mode,
        "n_dest_active": active_dest,
        "required_dest": required_dest,
        "missing_required_dest": missing_required,
        "valid": not missing_required,
        "models_path": os.path.abspath(path),
    }
    with open(os.path.join(a.out, "fit_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(json.dumps(meta, indent=2, default=str), flush=True)
    print(f"[{a.arm}] wrote {path}", flush=True)
    if missing_required:
        print(f"[{a.arm}] INVALID: missing/zero required destination "
              f"coefficients {missing_required}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
