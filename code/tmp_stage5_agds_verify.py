import os
import json
import numpy as np
import pandas as pd

from data_connect import Synth_Connect
from ga_hypothesis_testing import run_ga_stage
from stage5_destination_choice import run_stage5


def _fit_stats(gr):
    hist = pd.DataFrame(gr.history or [])
    if hist.empty:
        return {
            "gens": 0,
            "best_start": np.nan,
            "best_end": np.nan,
            "improvement": np.nan,
        }
    best_start = float(hist.iloc[0].get("best_so_far", hist.iloc[0].get("best_fitness", np.nan)))
    best_end = float(hist.iloc[-1].get("best_so_far", hist.iloc[-1].get("best_fitness", np.nan)))
    return {
        "gens": int(len(hist)),
        "best_start": best_start,
        "best_end": best_end,
        "improvement": float(best_end - best_start),
    }


def _coef_summary(seg_models):
    out = {}
    for seg, sres in (seg_models or {}).items():
        seg_out = {}
        for mtype, key in (("mode", "mode_model"), ("dest", "dest_model")):
            mdl = sres.get(key)
            if mdl is None:
                seg_out[mtype] = {"missing": True}
                continue
            pf = getattr(mdl, "_final_pf", None)
            if pf is None or not isinstance(pf, pd.DataFrame):
                pf = getattr(mdl, "pf", pd.DataFrame())
            if not isinstance(pf, pd.DataFrame) or pf.empty:
                seg_out[mtype] = {"missing": True}
                continue

            vals = pd.to_numeric(pf.get("value", pd.Series(dtype=float)), errors="coerce")
            hold = pf.get("holdfast", pd.Series(False, index=pf.index)).fillna(False).astype(bool)
            free_mask = ~hold
            free_vals = vals[free_mask]
            n_free = int(free_mask.sum())
            n_zeroish = int((free_vals.abs() < 1e-8).sum()) if n_free else 0

            if "t_stat" in pf.columns:
                t = pd.to_numeric(pf["t_stat"], errors="coerce")[free_mask]
            elif "tstat" in pf.columns:
                t = pd.to_numeric(pf["tstat"], errors="coerce")[free_mask]
            else:
                t = pd.Series(dtype=float)
            n_sig = int((t.abs() >= 1.96).sum()) if len(t) else 0
            n_t = int(t.notna().sum()) if len(t) else 0

            seg_out[mtype] = {
                "missing": False,
                "n_params": int(len(pf)),
                "n_free": n_free,
                "n_free_zeroish": n_zeroish,
                "free_zeroish_ratio": float(n_zeroish / n_free) if n_free else np.nan,
                "n_sig": n_sig,
                "n_t": n_t,
                "sig_ratio": float(n_sig / n_t) if n_t else np.nan,
            }
        out[seg] = seg_out
    return out


def _load_processed_survey(sc):
    base_dir = os.path.dirname(__file__)
    saved_dir = os.path.join(base_dir, "saved_data")
    persons = pd.read_excel(os.path.join(base_dir, "..", "2_QTS_PERSONS.xlsx"))
    household = pd.read_excel(os.path.join(base_dir, "..", "1_QTS_HOUSEHOLDS.xlsx"))
    persons = sc.process_hhts_person(persons.merge(household, on="HHID", how="left"))
    if "person_id" not in persons.columns and "PERSID" in persons.columns:
        persons = persons.rename(columns={"PERSID": "person_id"})
    sa1_to_sa2 = pd.read_csv(os.path.join(base_dir, "..", "SA1_SHP", "sa1_2021.csv"))
    sa1_to_sa2 = sa1_to_sa2.set_index("SA1_CODE21")["SA2_CODE21"].astype(int).to_dict()
    persons["HOME_SA2_2021"] = persons["HOME_SA1_2021"].map(sa1_to_sa2)
    persons["HOME_SA1"] = pd.to_numeric(
        persons["HOME_SA1_2021"], errors="coerce").fillna(0).astype(np.int64)

    frames = {
        "tours": pd.read_csv(os.path.join(saved_dir, "tours_processed.csv")),
        "trips": pd.read_csv(os.path.join(saved_dir, "trips_processed.csv")),
        "persons": persons,
    }
    for frame in frames.values():
        frame.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    return frames


def main():
    os.environ.setdefault("GA_ENGINE_STAGE5_MODE", "agds")
    os.environ.setdefault("GA_ENGINE_STAGE5_DEST", "agds")
    os.environ.setdefault("GA_MAX_WORKERS", "1")
    os.environ.setdefault("GA_EARLY_STOP", "1")
    os.environ.setdefault("GA_EARLY_STOP_PATIENCE", "8")
    os.environ.setdefault("GA_EARLY_STOP_MIN_GENS", "8")
    os.environ.setdefault("STAGE5_STRICT_ESTIMATION", "1")

    out_dir = os.path.join("runs", "agds_stage5_verify")
    os.makedirs(out_dir, exist_ok=True)

    sc = Synth_Connect(new_pop=True)
    out = _load_processed_survey(sc)
    land = sc.get_land()

    ga = run_ga_stage(
        stage=5,
        tours=out["tours"],
        trips=out["trips"],
        persons=out["persons"],
        skims="downcasted_matrices_base.omx",
        landuse=land,
        pop_size=14,
        n_generations=14,
        mutation_rate=0.10,
        n_restarts=1,
        verbose=True,
        stage_overrides={5: {"pop_size": 14, "n_generations": 14, "n_restarts": 1}},
    )

    seg_results = (ga.metadata or {}).get("segment_results", {})
    seg_stats = {k: _fit_stats(v) for k, v in seg_results.items()}

    # Use a small-but-real prediction set for injection/path validation.
    synth_s4 = out["tours"].sample(n=min(2000, len(out["tours"])), random_state=42).copy()

    best_mode = (ga.best_features or {}).get("mode", None)
    best_dest = (ga.best_features or {}).get("dest", None)

    plan, models = run_stage5(
        trips=out["trips"],
        persons=out["persons"],
        synth_s4=synth_s4,
        skims="downcasted_matrices_base.omx",
        landuse=land,
        tours=out["tours"],
        seed=42,
        save_path=os.path.join(out_dir, "stage5_models.pkl"),
        segments=["work", "edu", "nonmand"],
        verbose=False,
        mode_feature_override=best_mode,
        dest_feature_override=best_dest,
        hhts_persons=out["persons"],
    )

    coef_stats = _coef_summary(models)

    pred_rows = int(len(plan)) if plan is not None else 0
    pred_dests = int(plan["DTAZ"].nunique()) if plan is not None and "DTAZ" in plan.columns else 0
    pred_modes = (
        plan["pred_mode"].value_counts(dropna=False).to_dict()
        if plan is not None and "pred_mode" in plan.columns else {}
    )

    logsum_scales = {}
    for seg, sres in (models or {}).items():
        logsum_scales[seg] = {
            "logsum_mean": float(sres.get("logsum_mean", np.nan)) if sres else np.nan,
            "logsum_std": float(sres.get("logsum_std", np.nan)) if sres else np.nan,
        }

    report = {
        "engine": {
            "stage5_mode": os.environ.get("GA_ENGINE_STAGE5_MODE"),
            "stage5_dest": os.environ.get("GA_ENGINE_STAGE5_DEST"),
        },
        "ga_best_features": ga.best_features,
        "ga_best_fitness": float(ga.best_fitness),
        "ga_segment_stats": seg_stats,
        "coef_stats": coef_stats,
        "prediction": {
            "rows": pred_rows,
            "unique_dests": pred_dests,
            "mode_counts": pred_modes,
        },
        "logsum_scales": logsum_scales,
        "outputs": {
            "models_pkl": os.path.join(out_dir, "stage5_models.pkl"),
            "plan_csv": os.path.join(out_dir, "stage5_plan_sample.csv"),
            "report_json": os.path.join(out_dir, "report.json"),
        },
    }

    if plan is not None:
        plan.to_csv(os.path.join(out_dir, "stage5_plan_sample.csv"), index=False)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
