#!/usr/bin/env python
"""Score one fitted Stage-5 bundle on the 30% HHTS holdout.

This is the small, reusable arm scorer needed by the blocked P0/P2 table.
It applies an arbitrary Stage-5 model bundle through the same
``_predict_segment`` path used by ``validate_safety_arms.py`` and writes the
Stage-5 metrics that ``tab:p0p2`` needs: mode log-loss / rho2, destination
log-loss / MAE / JSD / top-25, mode-share MAE, exposure and trip length.

Usage:
    python _score_stage5_arm.py \
        --models runs/p0_work/stage5_models.pkl \
        --segment work --arm P0 \
        --run-dir runs/safety_baseline__27333960.aqua \
        --out runs/p0p2_smoke/score_p0_work.json

The same script can be pointed at the published P2 bundle; the two JSON files
are then compared by ``_compare_p0p2_smoke.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault(
    "STAGE5_ROUTE_TYPE_TABLE",
    os.path.join("runs", "route_choice_371", "route_logsum_type.parquet"))
os.environ.setdefault("STAGE5_LEGACY_DEST_CRASH", "0")

from stage5_destination_choice import (  # noqa: E402
    _filter_seg, _open_skims, _prep_hhts_tours, _read_crash_od_table,
    _skim_flat, load_models,
)
from validate_safety_arms import (  # noqa: E402
    _crash_matrix, _dist_matrix, _holdout_tours, _metrics, _predict_segment,
)


def _mode_ll(pred) -> float:
    """Log-likelihood of the observed mode, on the fit's own scale.

    Larch's fitted ``mode_ll`` is the WEIGHTED log-likelihood because the mode
    model carries ``weight_co_var='weight'``.  Comparing it with an unweighted
    sum produced a spurious 8-12% "reproduction" gap on every arm.  Use the
    tour weights returned by ``_predict_segment`` so the comparison is
    like-for-like.
    """
    mp = np.asarray(pred["mode_probs"], dtype=float)
    obs = np.asarray(pred["tour_mode"], dtype=int)
    idx = np.clip(obs - 1, 0, mp.shape[1] - 1)
    logp = np.log(np.clip(mp[np.arange(len(obs)), idx], 1e-12, 1.0))
    w = pred.get("weight")
    # LatentClass likelihoods are unweighted in larch; weighting the
    # reproduction for them would compare two different objectives.
    if not pred.get("mode_weighted", True):
        w = None
    if w is not None and np.asarray(w).size == len(logp):
        w = np.asarray(w, dtype=float)
        return float(np.sum(w * logp))
    return float(np.sum(logp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--skims", default=None,
                    help="default: the skims recorded in the fit's fit_meta.json")
    ap.add_argument("--crash-matrix", default=None,
                    help="default: the crash matrix recorded in fit_meta.json")
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repro-tol", type=float, default=0.005,
                    help="max relative gap between the fitted mode LL and the "
                         "scorer's reproduction on the estimation tours")
    ap.add_argument("--no-strict-repro", action="store_true",
                    help="record a failed LL reproduction instead of exiting")
    a = ap.parse_args()

    # Rebuild the exact construction the bundle was fitted on (stage5_arm_inputs).
    from stage5_arm_inputs import compare_records, fit_input_record, load_arm_inputs
    meta_path = os.path.join(os.path.dirname(os.path.abspath(a.models)),
                             "fit_meta.json")
    fit_meta = json.load(open(meta_path, encoding="utf-8"))         if os.path.exists(meta_path) else {}
    rec = fit_meta.get("inputs") or {}
    skims_path = a.skims or rec.get("skims") or os.path.join(
        "..", "demand_", "skims_371.omx")
    crash_matrix = a.crash_matrix or rec.get("crash_matrix") or "crash_rate_total"
    frac = a.frac if a.frac is not None else float(rec.get("frac", fit_meta.get("frac", 0.30)))
    seed = a.seed if a.seed is not None else int(rec.get("seed", fit_meta.get("seed", 42)))
    split = rec.get("split", fit_meta.get("split", "random"))
    os.environ["STAGE5_CRASH_MATRIX"] = crash_matrix

    models = load_models(a.models)
    if a.segment not in models:
        raise SystemExit(f"{a.models} has no '{a.segment}' entry: {sorted(models)}")

    inp = load_arm_inputs(a.segment, skims_path, frac, seed, split, crash_matrix)
    _close = inp["close"]
    try:
        diffs = compare_records(
            rec, fit_input_record(a.segment, skims_path, frac, seed, split,
                                  crash_matrix))
        for d in diffs:
            print(f"INPUT DIFF vs fit: {d}", flush=True)

        landuse, skims, skim = inp["landuse"], inp["skims"], inp["skim_df"]

        # Reproduction guard: the estimation tours must give back the fitted
        # mode log-likelihood, otherwise the holdout numbers describe a
        # different model than the one that was estimated.
        repro = {"mode_ll_fit": fit_meta.get("mode_ll")}
        if fit_meta.get("mode_ll") is not None:
            pred_est = _predict_segment(
                a.segment, inp["est_tours"], inp["est_persons"], models,
                landuse, skims, skim, zero_crash=False, crash_cols={})
            ll_est = _mode_ll(pred_est)
            ll_fit = float(fit_meta["mode_ll"])
            n_est = int(len(np.asarray(pred_est["tour_mode"])))
            n_fit = fit_meta.get("mode_n_cases")
            # Totals are only comparable on the SAME sample: check the case
            # count first, then compare log-likelihood per case.
            same_n = n_fit is not None and int(n_fit) == n_est
            gap = (abs(ll_est - ll_fit) / max(abs(ll_fit), 1e-9)
                   if same_n else float("nan"))
            repro.update({"mode_ll_reproduced": ll_est,
                          "mode_n_cases_fit": n_fit,
                          "mode_n_cases_scored": n_est,
                          "mode_ll_per_case_fit": (ll_fit / int(n_fit)
                                                   if n_fit else None),
                          "mode_ll_per_case_scored": ll_est / max(n_est, 1),
                          "mode_ll_rel_gap": gap,
                          "mode_ll_repro_ok": bool(same_n and gap <= a.repro_tol)})
            print(f"LL reproduction: n fit={n_fit} scored={n_est} | "
                  f"fitted={ll_fit:.1f} scored={ll_est:.1f} rel_gap={gap:.4f}"
                  + ("" if same_n else "  (sample differs - cannot verify)"),
                  flush=True)
            if not repro["mode_ll_repro_ok"] and not a.no_strict_repro:
                raise SystemExit(
                    f"mode LL not reproduced (rel gap {gap:.4f} > "
                    f"{a.repro_tol}); scoring inputs differ from the fit. "
                    f"Input diffs: {diffs or 'none recorded'}")

        pred = _predict_segment(
            a.segment, inp["ho_tours"], inp["ho_persons"], models, landuse,
            skims, skim, zero_crash=False, crash_cols={})
        metrics = _metrics(pred, _crash_matrix(skim), _dist_matrix(skim))
        split = inp["split"]
        n_alts = int(np.asarray(pred["dest_probs"]).shape[1])
        metrics.update({
            "arm": a.arm,
            "segment": a.segment,
            "models": os.path.abspath(a.models),
            "split": split,
            "frac": frac,
            "seed": seed,
            "input_diffs": diffs,
            **repro,
            "mode_rho2_uniform5": (
                1.0 - metrics["mode_logloss"] / math.log(5.0)
                if metrics["mode_logloss"] == metrics["mode_logloss"] else None),
            "dest_rho2_uniform": (
                1.0 - metrics["dest_logloss"] / math.log(max(n_alts, 2))
                if metrics["dest_logloss"] == metrics["dest_logloss"] else None),
        })
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        print(json.dumps(metrics, indent=2, default=str))
        print(f"\nwrote {a.out}")
        return 0
    finally:
        try:
            _close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
