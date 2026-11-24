"""Single source of truth for Stage-5 arm inputs, shared by fit AND score.

``_fit_stage5_arm.py`` and ``_score_stage5_arm.py`` previously built their
inputs independently and drifted apart:

  * data    : the fit read fresh HHTS via ``pipeline_logger.load_data()``; the
              scorer read an old pickle snapshot (safety_baseline__27333960).
  * crash   : the fit used ``rate_skim`` (crash-rate surface + the
              ``safety_crash_measure`` identification transforms); the scorer
              merged the raw crash OMX and skipped the transforms.

The same fitted bundle therefore scored 2,332.6 vs the fitted 2,139.6 mode
log-likelihood on the *identical* P0-edu estimation tours, predicted walk at
0.011 vs 0.096 observed, and gave 2.1 km vs 27.6 km non-mandatory trip
lengths depending on which script scored it. Every Stage-5 consumer now builds
its inputs here, so a model is always scored on exactly the construction it
was fitted on. ``fit_input_record`` is persisted in ``fit_meta.json`` and the
scorer checks it, plus reproduces the fitted mode log-likelihood on the
estimation side before trusting any holdout number.
"""
from __future__ import annotations

import os

ENV_KEYS = (
    "STAGE5_ROUTE_TYPE_TABLE", "STAGE5_CRASH_MATRIX", "STAGE5_LEGACY_DEST_CRASH",
    "STAGE5_DEST_SAMPLE_SIZE", "STAGE5_PT_MODES", "STAGE5B_TRANSIT_ACCESS",
    "UNIFIED_LOGSUM_ACC", "REAL_WALK_BIKE_OMX", "PIPELINE_SKIMS_PATH",
    "SAFETY_CRASH_PLACEBO",
    # build flags that change the utility itself
    "STAGE5_MODE_ORTHOG", "STAGE5_DEST_ORTHOG", "STAGE5_MODE_STRUCTURE",
    "STAGE5_POOLED_IVT", "STAGE5_NONMAND_DIST_BASIS", "STAGE5_LCCM_MEMBERSHIP_VARS",
    "STAGE5_LCCM_CLASS_CRASH", "STAGE5_MODE_N_CLASSES", "STAGE5_ESTIMATE_MU",
    "STAGE5_MU_FIXED_VALUE",
)


def _split_sides(tours_raw, persons_raw, segment, ho_pids):
    from stage5_destination_choice import (_filter_seg, _prep_hhts_tours,
                                           attach_person_type)
    ho = {str(p) for p in ho_pids}
    t_ho = tours_raw["person_id"].astype(str).isin(ho)
    p_ho = persons_raw["person_id"].astype(str).isin(ho)
    sides = {}
    for name, tm, pm in (("est", ~t_ho, ~p_ho), ("ho", t_ho, p_ho)):
        persons = attach_person_type(persons_raw[pm].copy())
        tours = _prep_hhts_tours(tours_raw[tm].copy(), persons)
        sides[name] = (_filter_seg(tours, segment), persons)
    return sides


def load_arm_inputs(segment: str, skims_path: str, frac: float = 0.30,
                    seed: int = 42, split: str = "random",
                    crash_matrix: str | None = None) -> dict:
    """Build the estimation and holdout Stage-5 inputs for one segment.

    Returns est/ho (segment tours, persons) pairs plus the shared landuse,
    open skim handle, the canonical flattened skim frame (OSM walk/bike
    overlay + crash-rate surface + identification transforms) and a closer.
    """
    import pipeline_logger as pl
    from refit_dest_crash_rate import rate_skim
    from stage5_destination_choice import _open_skims, _skim_flat
    from validate_safety_arms import _holdout_tours

    crash_matrix = crash_matrix or os.environ.get(
        "STAGE5_CRASH_MATRIX", "crash_rate_total")
    _sc, out, landuse = pl.load_data()
    tours_raw, persons_raw = out["tours"], out["persons"]
    ho_pids, split_used = _holdout_tours(tours_raw, persons_raw, frac, seed, split)
    sides = _split_sides(tours_raw, persons_raw, segment, ho_pids)

    skims, close = _open_skims(skims_path)
    skim_df = rate_skim(_skim_flat(skims), crash_matrix)
    return {
        "est_tours": sides["est"][0], "est_persons": sides["est"][1],
        "ho_tours": sides["ho"][0], "ho_persons": sides["ho"][1],
        "landuse": landuse, "skims": skims, "skim_df": skim_df,
        "close": close, "split": split_used, "n_holdout_persons": len(ho_pids),
        "record": fit_input_record(segment, skims_path, frac, seed, split,
                                   crash_matrix),
    }


def fit_input_record(segment, skims_path, frac, seed, split, crash_matrix) -> dict:
    """What a fit was built on; stored in fit_meta.json and checked at scoring."""
    return {
        "data_source": "pipeline_logger.load_data",
        "segment": segment,
        "skims": os.path.abspath(skims_path),
        "frac": float(frac), "seed": int(seed), "split": split,
        "crash_matrix": crash_matrix,
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
    }


def compare_records(fit_rec: dict | None, score_rec: dict) -> list[str]:
    """Human-readable differences between the fit and score input records."""
    if not fit_rec:
        return ["fit_meta.json has no 'inputs' record (fitted before the "
                "shared input builder) - cannot verify the construction"]
    diffs = []
    for k in ("data_source", "segment", "skims", "frac", "seed", "split",
              "crash_matrix"):
        if fit_rec.get(k) != score_rec.get(k):
            diffs.append(f"{k}: fit={fit_rec.get(k)!r} score={score_rec.get(k)!r}")
    fe, se = fit_rec.get("env", {}), score_rec.get("env", {})
    for k in ENV_KEYS:
        if fe.get(k) != se.get(k):
            diffs.append(f"env {k}: fit={fe.get(k)!r} score={se.get(k)!r}")
    return diffs
