"""
run_stage3_model_suite.py
=========================
Estimate every configured Stage 3 duration variant and apply each one to the
synthetic Stage 2 population.  Each variant is isolated in its own directory,
so model fitting never overwrites another model's artifacts.

Examples
--------
Local / interactive:
    python run_stage3_model_suite.py --models all --out-dir saved_data/stage3_suite

PBS:
    qsub -v STAGE3_MODEL_SUITE=all,RUN_STAGE4=1 job_stage3_model_suite.pbs

Model names
-----------
    lognormal              BudgetLognormal duration marginal
    copula_hazard           CopulaLinkedHazardDurationModel
    lognormal_mdcev         lognormal + MDCEV person-day allocation
    copula_hazard_mdcev     copula-hazard + MDCEV allocation
    all                     all four variants

The duration-model estimation path also fits the existing SearchLibrium
planning-order RRM when enough sequence data is available.  With --run-stage4,
the existing SearchLibrium timing RRM is estimated once and applied to every
Stage 3 variant.
With --run-stage5-6, each scheduled variant continues through the existing
Stage 5 destination/mode/trip-chain estimator and Stage 6 mode predictor.
With --run-stage5b, each variant also runs the Stage 5b trip-level
intermediate-stop destination choice (first variant estimates the per-purpose
MNLs, later variants reuse them predict-only).
With --ownership (default on), household vehicle/bike ownership is predicted
for the synthetic population before Stage 3/5 prediction.
With --stage0 (default on), Stage 0 household coordination / joint travel is
applied to the synthetic population before Stage 3 prediction (coordinated
CDAP + joint-tour purposes folded into the Stage-1-style purpose counts).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stage3_activity_duration import (
    estimate_duration_models,
    predict_activity_durations,
    save_models as save_stage3_models,
)


LOGGER = logging.getLogger("stage3_model_suite")
ALL_MODELS = (
    "lognormal",
    "copula_hazard",
    "lognormal_mdcev",
    "copula_hazard_mdcev",
)


def _model_spec(token: str) -> tuple[str, bool, str]:
    """Normalize a command-line model token to base model and MDCEV flag."""
    raw = token.strip().lower().replace("-", "_")
    aliases = {
        "ln": "lognormal",
        "copula": "copula_hazard",
        "hazard": "copula_hazard",
        "copula_linked_hazard": "copula_hazard",
        "ln_mdcev": "lognormal_mdcev",
        "copula_mdcev": "copula_hazard_mdcev",
    }
    raw = aliases.get(raw, raw)
    if raw.endswith("+mdcev"):
        raw = raw[:-6] + "_mdcev"
    if raw not in ALL_MODELS:
        raise ValueError(
            f"Unknown Stage 3 model {token!r}. Choose one of: "
            f"{', '.join(ALL_MODELS)}, all"
        )
    use_mdcev = raw.endswith("_mdcev")
    base = raw[:-6] if use_mdcev else raw
    return base, use_mdcev, raw


def parse_model_suite(value: str) -> list[tuple[str, bool, str]]:
    """Parse a comma-separated model list, preserving user order."""
    tokens = [part.strip() for part in value.split(",") if part.strip()]
    if not tokens:
        tokens = ["all"]
    expanded: list[str] = []
    for token in tokens:
        if token.lower() == "all":
            expanded.extend(ALL_MODELS)
        else:
            expanded.append(token)

    specs = []
    seen = set()
    for token in expanded:
        spec = _model_spec(token)
        if spec[2] not in seen:
            specs.append(spec)
            seen.add(spec[2])
    return specs


def _json_value(value):
    """Convert numpy values and paths into JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _fit_summary(model_name: str, models: dict) -> list[dict]:
    """Flatten purpose-level fit information for CSV and manifest output."""
    rows = []
    for purpose, entry in models.items():
        if not isinstance(entry, dict) or "model" not in entry:
            continue
        model = entry["model"]
        rows.append({
            "model_variant": model_name,
            "purpose": purpose,
            "estimator": entry.get("estimator", type(model).__name__),
            "n_obs": entry.get("n_obs", getattr(model, "n_", np.nan)),
            "sigma": entry.get("sigma", getattr(model, "sigma_", np.nan)),
            "loglik": entry.get("ll", getattr(model, "ll_", np.nan)),
            "hazard_loglik": getattr(model, "hazard_ll_", np.nan),
            "copula_rho": getattr(model, "rho_", np.nan),
            "rmse": entry.get("rmse", getattr(model, "rmse_", np.nan)),
            "calibration_factor": entry.get("calibration_factor", np.nan),
            "converged": getattr(model, "converged_", np.nan),
        })
    if "_planning_order" in models:
        planning = models["_planning_order"]
        rows.append({
            "model_variant": model_name,
            "purpose": "_planning_order",
            "estimator": getattr(planning, "estimator", type(planning).__name__),
            "n_obs": getattr(planning, "n_obs", np.nan),
            "sigma": np.nan,
            "loglik": getattr(getattr(planning, "rrm", None), "loglik", np.nan),
            "hazard_loglik": np.nan,
            "copula_rho": np.nan,
            "rmse": np.nan,
            "calibration_factor": np.nan,
            "converged": getattr(getattr(planning, "rrm", None), "converged", np.nan),
        })
    if "_mdcev" in models:
        mdcev = models["_mdcev"]
        rows.append({
            "model_variant": model_name,
            "purpose": "_mdcev",
            "estimator": mdcev.get("method", "mdcev"),
            "n_obs": mdcev.get("n_obs", np.nan),
            "sigma": np.nan,
            "loglik": np.nan,
            "hazard_loglik": np.nan,
            "copula_rho": np.nan,
            "rmse": np.nan,
            "calibration_factor": np.nan,
            "converged": True,
        })
    return rows


def _run_fresh_inputs(zone: str | None) -> None:
    """Build Stage 1 and Stage 2 inputs when explicitly requested."""
    stage1_cmd = [sys.executable, "stage1_activity_frequency.py", "--fresh"]
    if zone:
        stage1_cmd.append(zone)
    LOGGER.info("Fresh mode: running %s", " ".join(stage1_cmd))
    subprocess.run(stage1_cmd, check=True)
    subprocess.run([sys.executable, "stage2_day_duration.py"], check=True)


def _load_ga_overrides(path: str) -> dict[int, object]:
    """Load a merged AllStagesResult pickle and return per-stage GAResult objects.

    Stage keys of interest: 3 (duration), 4 (timing), 5 (mode+dest).  Returns an
    empty dict if the path is empty/missing so the suite runs with default features.
    """
    if not path:
        return {}
    ga_path = Path(path)
    if not ga_path.is_file():
        LOGGER.warning("GA results not found at %s — running with default features", path)
        return {}
    import pickle
    from ga_hypothesis_testing import AllStagesResult
    with open(ga_path, "rb") as fh:
        ga = pickle.load(fh)
    if not isinstance(ga, AllStagesResult):
        LOGGER.warning("GA results at %s is not an AllStagesResult — ignoring", path)
        return {}
    LOGGER.info("Loaded GA results: stages=%s", sorted(ga.results.keys()))
    return ga.results


def _best_features(result, default=None):
    """Extract best_features from a GAResult object, or return default."""
    if result is None:
        return default
    bf = getattr(result, "best_features", None)
    return bf if bf is not None else default


def run_suite(args: argparse.Namespace) -> dict:
    specs = parse_model_suite(args.models)
    if args.run_stage5_6:
        args.run_stage4 = True
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_csv = Path(args.input).resolve()

    if args.fresh:
        _run_fresh_inputs(args.zone)
    if not input_csv.is_file():
        raise FileNotFoundError(
            f"Stage 2 input not found: {input_csv}. Run stage2_day_duration.py "
            "first or use --fresh."
        )

    ga_results = _load_ga_overrides(args.ga_results)
    s3_feats = _best_features(ga_results.get(3))
    s4_feats = _best_features(ga_results.get(4))
    s5_feats = _best_features(ga_results.get(5))
    s51_feats = _best_features(ga_results.get(51))
    s51_dest_feats = (
        s51_feats.get("dest")
        if isinstance(s51_feats, dict)
        else None
    )
    if ga_results:
        LOGGER.info(
            "GA feature overrides: stage3=%s stage4=%s stage5(mode)=%s "
            "stage5(dest)=%s stage51(trip-dest)=%s",
            list(s3_feats.keys()) if isinstance(s3_feats, dict) else s3_feats,
            list(s4_feats.keys()) if isinstance(s4_feats, dict) else s4_feats,
            (s5_feats or {}).get("mode"),
            (s5_feats or {}).get("dest"),
            s51_dest_feats,
        )

    LOGGER.info("Loading HHTS survey data once for %d model variants", len(specs))
    from data_connect import Synth_Connect
    survey_connector = Synth_Connect()
    survey = survey_connector.process_survey(save=False)
    synth_s2 = pd.read_csv(input_csv)
    if "person_id" not in synth_s2.columns:
        raise ValueError(f"Stage 2 input must contain person_id: {input_csv}")
    synth_persons = synth_s2.drop_duplicates("person_id").copy()
    LOGGER.info(
        "Loaded HHTS persons=%d tours=%d trips=%d; synthetic Stage 2 rows=%d",
        len(survey["persons"]), len(survey["tours"]),
        len(survey["trips"]), len(synth_s2),
    )

# ── Long-run (TikZ): household vehicle/bike ownership prediction ─────────
    # The synthetic population built upstream may lack vehrd/bikes/veh_bin/
    # bike_bin. Estimate an ordered-logit (vehrd) + zero-inflated ordered-probit
    # (bikes) on the HHTS survey we already hold, then predict for the whole
    # synth so Stage 3/5/6 see complete mode-availability cells.
    if args.ownership:
        try:
            # process_survey already loaded above; reuse its persons table.
            from build_synth_population import (
                predict_vehicle_bike_ownership,
                _ownership_feature_cols,
            )
            _hh = survey["persons"]
            if _ownership_feature_cols(_hh) and (
                "vehrd" not in synth_persons.columns
                or "bikes" not in synth_persons.columns
                or synth_persons["vehrd"].fillna(0).eq(0).all()
                or synth_persons["bikes"].fillna(0).eq(0).all()
            ):
                synth_persons = predict_vehicle_bike_ownership(
                    synth_persons, _hh, seed=int(args.seed),
                )
                # Propagate the completed person-level columns back onto the
                # full Stage 2 activity frame used for Stage 3/4 prediction.
                # Ownership is per person, so map through person_id and only
                # replace columns that were truly missing or all-zero.
                _own_map = synth_persons[["person_id"]].copy()
                for c in ("vehrd", "bikes", "veh_bin", "bike_bin"):
                    if c in synth_persons.columns:
                        _own_map[c] = synth_persons[c].values
                for c in _own_map.columns.drop("person_id"):
                    _should_replace = (
                        c not in synth_s2.columns
                        or synth_s2[c].fillna(0).eq(0).all()
                    )
                    if not _should_replace:
                        continue
                    _vals = synth_s2["person_id"].map(
                        _own_map.set_index("person_id")[c]
                    )
                    if c in synth_s2.columns:
                        synth_s2[c] = _vals
                    else:
                        synth_s2[c] = _vals
                LOGGER.info(
                    "Ownership prediction applied to %d person rows "
                    "(vehrd/bikes or veh_bin/bike_bin populated)", len(synth_persons)
                )
        except Exception as exc:
            LOGGER.warning("Ownership prediction skipped: %s", exc)

    # ── Stage 0 (TikZ): household coordination / joint travel ───────────────
    # When the synthetic people carry a household id (HHID or HHSIZE keyed by
    # child/person composition), run the ActivitySim-style household CDAP:
    # coordinated DAP, joint-tour frequency/participation/purpose, folded back
    # into the Stage-1-style purpose counts (n_*_out) so later stages see joint
    # (+ escort/childcare) travel. Non-fatal: skipped if no household id exists.
    if args.stage0:
        try:
            from cdap_households import (
                HouseholdCdapConfig,
                coordinate_household_cdap,
                joint_tour_frequency,
                joint_tour_participation,
                joint_tour_purpose,
                reconcile_joint_tours,
            )
            _hh_id = next(
                (c for c in ("HHID", "hh_id", "HHSIZE") if c in synth_persons.columns),
                None,
            )
            if _hh_id is None:
                LOGGER.warning("Stage 0 skipped: no household id column on synthetic persons")
            elif "person_id" not in synth_persons.columns:
                LOGGER.warning("Stage 0 skipped: no person_id column")
            else:
                cfg = HouseholdCdapConfig(
                    hh_id_col=_hh_id,
                    vehicles_col="vehrd" if "vehrd" in synth_persons.columns else "vehicles",
                )
                _coord = coordinate_household_cdap(synth_persons, cfg)
                _jf = joint_tour_frequency(_coord, cfg)
                _part = joint_tour_participation(_coord, _jf, cfg)
                if len(_part):
                    _part = joint_tour_purpose(_part, _coord, cfg)
                synth_persons = _coord
                # Fold joint-tour purposes into the purpose counts on `synth_s2`
                # (the Stage-2 row frame consumed downstream) person-by-person.
                if len(_part):
                    _jm = _part[["person_id"]].copy()
                    _jm["n_joint_purposes"] = _part["purpose"]
                    _jmap = _jm.groupby("person_id")["n_joint_purposes"].apply(list)
                    for i, pid in enumerate(synth_s2["person_id"]):
                        _pl = _jmap.get(pid)
                        if _pl:
                            for p in _pl:
                                _col = f"n_{str(p).lower()}_out"
                                if _col in synth_s2.columns:
                                    synth_s2.at[i, _col] = synth_s2.at[i, _col] + 1
                    LOGGER.info(
                        "Stage 0 household CDAP: coordinated %d people, "
                        "%d joint tours folded into purpose counts",
                        len(_coord), len(_part),
                    )
        except Exception as exc:
            LOGGER.warning("Stage 0 (household CDAP) skipped: %s", exc)

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_stage2": str(input_csv),
        "out_dir": str(out_dir),
        "models_requested": [name for _, _, name in specs],
        "run_stage4": bool(args.run_stage4),
        "run_stage5_6": bool(args.run_stage5_6),
        "run_stage5b": bool(args.run_stage5b),
        "ownership": bool(args.ownership),
        "stage0": bool(args.stage0),
        "sample": bool(args.sample),
        "seed": int(args.seed),
        "status": "running",
        "variants": {},
    }
    _write_json(out_dir / "manifest.json", manifest)
    summary_rows: list[dict] = []

    for model_index, (base_model, use_mdcev, model_name) in enumerate(specs):
        variant_dir = out_dir / model_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        model_path = variant_dir / "stage3_models.pkl"
        activity_path = variant_dir / "stage3_output.csv"
        LOGGER.info(
            "[%d/%d] Estimating %s (base=%s, use_mdcev=%s)",
            model_index + 1, len(specs), model_name, base_model, use_mdcev,
        )
        models = estimate_duration_models(
            survey["tours"],
            survey["persons"],
            trips=survey["trips"],
            min_sample=args.min_sample,
            use_mdcev=use_mdcev,
            duration_model=base_model,
            feature_override=s3_feats,
        )
        if not models:
            raise RuntimeError(f"No duration models were fitted for {model_name}")
        mdcev_fitted = "_mdcev" in models
        if use_mdcev and not mdcev_fitted:
            LOGGER.warning(
                "[%s] MDCEV was requested but unavailable or did not fit; "
                "the variant contains the base duration models only",
                model_name,
            )
        save_stage3_models(models, str(model_path))

        activities = predict_activity_durations(
            synth_s2.copy(),
            models,
            sample=args.sample,
            seed=args.seed + model_index,
            use_mdcev=use_mdcev,
        )
        activities.to_csv(activity_path, index=False)
        rows = _fit_summary(model_name, models)
        summary_rows.extend(rows)
        variant_record = {
            "status": "complete",
            "base_model": base_model,
            "use_mdcev": use_mdcev,
            "mdcev_fitted": mdcev_fitted,
            "model_pickle": str(model_path),
            "activity_csv": str(activity_path),
            "n_model_entries": len(models),
            "n_activity_rows": len(activities),
            "n_persons": int(activities["person_id"].nunique())
            if "person_id" in activities.columns else 0,
        }
        manifest["variants"][model_name] = variant_record
        _write_json(out_dir / "manifest.json", manifest)
        LOGGER.info("[%s] Prediction saved to %s", model_name, activity_path)

    summary_path = out_dir / "model_fit_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    if args.run_stage4:
        LOGGER.info("Estimating Stage 4 timing models once with STAGE4_TIMING_MODEL=%s",
                    os.environ.get("STAGE4_TIMING_MODEL", "rrm"))
        from stage4_schedule_placement import estimate_timing_models, predict_schedule
        timing_models = estimate_timing_models(
            survey["trips"], survey["tours"], survey["persons"],
            min_sample=args.timing_min_sample,
            feature_override=s4_feats,
        )
        timing_path = out_dir / "stage4_timing_models.pkl"
        from stage4_schedule_placement import save_models as save_stage4_models
        save_stage4_models(timing_models, str(timing_path))
        manifest["stage4_timing_models"] = str(timing_path)
        for _, _, model_name in specs:
            activity_path = out_dir / model_name / "stage3_output.csv"
            activities = pd.read_csv(activity_path)
            scheduled = predict_schedule(
                activities, timing_models, sample=args.sample,
                seed=args.seed,
            )
            scheduled_path = out_dir / model_name / "stage4_output.csv"
            scheduled.to_csv(scheduled_path, index=False)
            manifest["variants"][model_name]["scheduled_csv"] = str(scheduled_path)
            LOGGER.info("[%s] Schedule saved to %s", model_name, scheduled_path)
        _write_json(out_dir / "manifest.json", manifest)

    if args.run_stage5_6:
        skims_path = Path(args.skims).resolve()
        if not skims_path.is_file():
            raise FileNotFoundError(f"Stage 5/6 skim file not found: {skims_path}")
        LOGGER.info("Loading land use once for Stage 5/6 continuation")
        landuse = survey_connector.get_land()
        from stage5_destination_choice import run_stage5
        from stage6_mode_choice import run_stage6
        if args.run_stage5b:
            from stage5b_trip_destination import run_stage5b
            from stage5b_trip_destination import trip_destination_feature_mask
            # Stage 5b estimation on the full predicted population builds a dense
            # obs x zones design matrix, so the estimator reuses a single fit
            # (saved from the first variant) for predict-only runs.
            stage5b_shared = out_dir / "stage5b_models.pkl"
            stage5b_estimated = False
            stage5b_hhts_predicted = False
            stage5b_feature_mask = trip_destination_feature_mask(s51_dest_feats)
        s5_mode_feats = (s5_feats or {}).get("mode") if isinstance(s5_feats, dict) else s5_feats
        s5_dest_feats = (s5_feats or {}).get("dest") if isinstance(s5_feats, dict) else None
        for model_index, (_, _, model_name) in enumerate(specs):
            variant_dir = out_dir / model_name
            scheduled_path = variant_dir / "stage4_output.csv"
            scheduled = pd.read_csv(scheduled_path)
            
            # ── Benchmark: distance-band NL destination vs MNL ──
            # If RUN_STAGE5_NL_DEST=1, run each variant twice:
            #   1. MNL (default)
            #   2. NL distance-band (STAGE5_DEST_NL_DISTANCE=1)
            nl_variants = ['mnl']
            if args.run_stage5_nl_dest:
                nl_variants.append('nl_distance')
            
            for nl_variant in nl_variants:
                variant_suffix = f"_{nl_variant}" if nl_variant != 'mnl' else ""
                variant_dir_nl = out_dir / f"{model_name}{variant_suffix}"
                variant_dir_nl.mkdir(parents=True, exist_ok=True)
                
                scheduled_path_nl = variant_dir_nl / "stage4_output.csv"
                if not scheduled_path_nl.exists():
                    # Copy stage4 output to NL variant dir
                    import shutil
                    shutil.copy2(scheduled_path, scheduled_path_nl)
                scheduled_nl = pd.read_csv(scheduled_path_nl)
                
                LOGGER.info("[%s] Running Stage 5 destination/mode/trip-chain estimation [%s]", 
                           model_name, nl_variant)
                stage5_path = variant_dir_nl / "stage5_models.pkl"
                previous_pipeline_out = os.environ.get("PIPELINE_OUT_DIR")
                os.environ["PIPELINE_OUT_DIR"] = str(variant_dir_nl)
                
                # Set NL distance band env var
                previous_nl_dest = os.environ.get("STAGE5_DEST_NL_DISTANCE")
                if nl_variant == 'nl_distance':
                    os.environ["STAGE5_DEST_NL_DISTANCE"] = "1"
                else:
                    os.environ.pop("STAGE5_DEST_NL_DISTANCE", None)
                
                try:
                    stage5_output, _ = run_stage5(
                        survey["trips"],
                        synth_persons,
                        scheduled_nl,
                        skims=str(skims_path),
                        landuse=landuse,
                        tours=survey["tours"],
                        seed=args.seed + model_index,
                        save_path=str(stage5_path),
                        hhts_persons=survey["persons"],
                        mode_feature_override=s5_mode_feats,
                        dest_feature_override=s5_dest_feats,
                    )
                finally:
                    if previous_pipeline_out is None:
                        os.environ.pop("PIPELINE_OUT_DIR", None)
                    else:
                        os.environ["PIPELINE_OUT_DIR"] = previous_pipeline_out
                    if previous_nl_dest is not None:
                        os.environ["STAGE5_DEST_NL_DISTANCE"] = previous_nl_dest
                    else:
                        os.environ.pop("STAGE5_DEST_NL_DISTANCE", None)
                
                stage5_csv = variant_dir_nl / "stage5_output.csv"
                stage5_output.to_csv(stage5_csv, index=False)
                
                # Store NL variant info in manifest
                if 'nl_variants' not in manifest["variants"][model_name]:
                    manifest["variants"][model_name]['nl_variants'] = {}
                manifest["variants"][model_name]['nl_variants'][nl_variant] = {
                    "stage5_models": str(stage5_path),
                    "stage5_csv": str(stage5_csv),
                    "n_stage5_rows": len(stage5_output),
                }
                _write_json(out_dir / "manifest.json", manifest)
            
            # Use MNL variant for Stage 6 continuation
            stage6_input = stage5_output
            LOGGER.info("[%s] Running Stage 6 mode-choice estimation", model_name)
            stage6_path = variant_dir / "stage6_models.pkl"
            if args.run_stage5b:
                # ── Stage 5b (TikZ): trip-level intermediate-stop destination ──
                stage5b_reuse = str(stage5b_shared) if stage5b_shared.is_file() else None
                LOGGER.info(
                    "[%s] Running Stage 5b trip stop-destination%s",
                    model_name,
                    " (reusing fitted models)" if stage5b_reuse else "",
                )
                try:
                    if not stage5b_estimated:
                        # Calibrate on observed HHTS trips first. The synthetic
                        # Stage 5 output is prediction input only.
                        run_stage5b(
                            survey["trips"],
                            survey["persons"],
                            skims_path=str(skims_path),
                            landuse=landuse,
                            output_dir=str(variant_dir),
                            g_tripdest=stage5b_feature_mask,
                            predict=False,
                            validate=True,
                            save_models=True,
                        )
                        _s5b_first = variant_dir / "stage5b_models.pkl"
                        if not _s5b_first.is_file():
                            raise RuntimeError(
                                "Stage 5b observed calibration did not produce "
                                f"a model bundle: {_s5b_first}"
                            )
                        stage5b_shared.write_bytes(_s5b_first.read_bytes())
                        stage5b_estimated = True
                        LOGGER.info("Stage 5b observed models shared at %s", stage5b_shared)

                    stage5b_reuse = str(stage5b_shared)
                    if not stage5b_hhts_predicted:
                        # Reapply the shared observed calibration bundle to
                        # HHTS so paper validation is separate from the
                        # synthetic population output used by Stage 6.
                        hhts_prediction_dir = out_dir / "stage5b_hhts_prediction"
                        hhts_res = run_stage5b(
                            survey["trips"],
                            survey["persons"],
                            skims_path=str(skims_path),
                            landuse=landuse,
                            output_dir=str(hhts_prediction_dir),
                            reuse_models_path=stage5b_reuse,
                            save_models=False,
                            g_tripdest=stage5b_feature_mask,
                            predict=True,
                            validate=True,
                        )
                        manifest["stage5b_hhts_prediction"] = {
                            "source": "observed_hhts_in_sample",
                            "models": str(stage5b_shared),
                            "output_dir": str(hhts_prediction_dir),
                            "trips_csv": str(hhts_prediction_dir / "stage5b_trips.csv"),
                            "validation_csv": str(hhts_prediction_dir / "stage5b_validation.csv"),
                            "n_trips_rows": len(hhts_res.get("trips", pd.DataFrame())),
                            "n_validation_rows": len(hhts_res.get("validation", pd.DataFrame())),
                        }
                        _write_json(out_dir / "manifest.json", manifest)
                        stage5b_hhts_predicted = True
                    s5b_res = run_stage5b(
                        stage5_output,
                        synth_persons,
                        skims_path=str(skims_path),
                        landuse=landuse,
                        output_dir=str(variant_dir),
                        reuse_models_path=stage5b_reuse,
                        save_models=False,
                        g_tripdest=stage5b_feature_mask,
                        predict=True,
                        validate=False,
                    )
                    stage6_input = s5b_res["trips"]
                except Exception as exc:
                    LOGGER.warning("[%s] Stage 5b failed (%s) — Stage 6 continues on Stage 5 output", model_name, exc)
            stage6_output, _ = run_stage6(
                survey["trips"],
                survey["persons"],
                stage6_input,
                landuse=landuse,
                sample=args.sample,
                seed=args.seed + model_index,
                save_path=str(stage6_path),
            )
            stage6_csv = variant_dir / "stage6_output.csv"
            stage6_output.to_csv(stage6_csv, index=False)
            manifest["variants"][model_name].update({
                "stage5_models": str(stage5_path),
                "stage5_csv": str(stage5_csv),
                "stage6_models": str(stage6_path),
                "stage6_csv": str(stage6_csv),
                "n_stage6_rows": len(stage6_output),
            })
            _write_json(out_dir / "manifest.json", manifest)

    manifest["summary_csv"] = str(summary_path)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "complete"
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=os.environ.get("STAGE3_MODEL_SUITE", "all"),
        help="Comma-separated variants or all (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        default=os.path.join(os.environ.get("PIPELINE_OUT_DIR", "saved_data"),
                             "stage2_output.csv"),
        help="Stage 2 synthetic CSV",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("STAGE3_SUITE_OUT", "saved_data/stage3_model_suite"),
        help="Suite artifact directory",
    )
    parser.add_argument("--min-sample", type=int, default=int(os.environ.get("STAGE3_MIN_SAMPLE", "50")))
    parser.add_argument("--timing-min-sample", type=int, default=int(os.environ.get("STAGE4_MIN_SAMPLE", "30")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("STAGE3_SEED", "42")))
    parser.add_argument("--no-sample", dest="sample", action="store_false", help="Use model means instead of draws")
    parser.set_defaults(sample=True)
    parser.add_argument("--run-stage4", action="store_true", default=os.environ.get("RUN_STAGE4", "0") == "1",
                        help="Estimate timing RRM/MNL and write stage4_output.csv for every variant")
    parser.add_argument("--run-stage5-6", action="store_true",
                        default=os.environ.get("RUN_STAGE5_6", "0") == "1",
                        help="Continue every variant through Stage 5 and Stage 6")
    parser.add_argument("--run-stage5b", action="store_true",
                        default=os.environ.get("RUN_STAGE5B", "0") == "1",
                        help="Continue every variant through Stage 5b trip-level "
                             "stop-destination choice (implies --run-stage5-6)")
    parser.add_argument("--run-stage5-nl-dest", action="store_true",
                        default=os.environ.get("RUN_STAGE5_NL_DEST", "0") == "1",
                        help="Run distance-band Nested Logit destination benchmark "
                             "(STAGE5_DEST_NL_DISTANCE=1) alongside MNL for comparison")
    parser.add_argument("--ownership", action="store_true",
                        default=os.environ.get("RUN_OWNERSHIP", "1") == "1",
                        help="Predict household vehicle/bike ownership for the "
                             "synthetic population before Stage 3/5 prediction")
    parser.add_argument("--stage0", action="store_true",
                        default=os.environ.get("RUN_STAGE0", "1") == "1",
                        help="Run Stage 0 household coordination / joint travel "
                             "(coordinated CDAP + joint-tour purposes folded into "
                             "purpose counts) before Stage 3 prediction")
    parser.add_argument(
        "--skims",
        default=os.environ.get("SKIMS_PATH", "downcasted_matrices_base.omx"),
        help="OMX skims required by Stage 5/6",
    )
    parser.add_argument("--fresh", action="store_true", help="Run Stage 1 and Stage 2 before the suite")
    parser.add_argument("--zone", default=None, help="Optional zone argument passed to Stage 1 with --fresh")
    parser.add_argument(
        "--ga-results",
        default=os.environ.get("GA_RESULTS_PKL", ""),
        help=("Path to a merged AllStagesResult pickle. When given, the GA-optimised "
              "feature sets are used for Stage 3 (duration), Stage 4 (timing) and "
              "Stage 5 (mode/dest) and Stage 51 (trip destination) instead of "
              "the defaults."),
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = build_parser().parse_args()
    if args.run_stage5b and not args.run_stage5_6:
        args.run_stage5_6 = True
    try:
        manifest = run_suite(args)
    except Exception:
        LOGGER.exception("Stage 3 model suite failed")
        return 1
    LOGGER.info("Stage 3 model suite complete: %s", manifest["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
