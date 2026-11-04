"""
ga_variant_refits.py
====================
Model-family shootout inside one shard.

After a shard's feature SEARCH converges on a winning spec, this module
refits that SAME spec under each competing model family and scores every
fit with BIC/n on the SAME data, so families are comparable:

  stage 4 (per purpose) : rrm + tdd by default; mnl / ordered / hybrid opt-in
                          (GA_S4_VARIANTS / GA_SHARD_VARIANTS).
  stage 5 mode (per seg): nl (nested) + mnl (flat) by default; mxl / gse opt-in
                          (GA_S5_MODE_VARIANTS).
  stage 5 dest (per seg): mnl + nl_dist by default; exploded diagnostic opt-in
                          (GA_S5_DEST_VARIANTS).

Every variant is guarded: a failed fit is a leaderboard row with
converged=0, never a crash. Outputs per shard (under variants/):

  leaderboard.csv / leaderboard.html / winner.json   (see ga_shard.py)
  best_model_<variant>.pkl                          fitted objects (S4; S5
      stores the winning SPEC + scores — the full refit happens in
      merge/predict from the winner spec, same as the legacy GA path)

Comparability: winners are picked WITHIN groups only (ga_shard.pick_winners).
`tdd` (21 joint alts) and `exploded` (rank-expanded likelihood) live in
their own groups and are reported side by side, never auto-compared
against period-MNL / zone-MNL BIC/n.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback

import numpy as np
import pandas as pd

import ga_shard as _shard


def _row(stage, segment, slot, variant, converged=False, ll=float("nan"),
         k=None, n=None, bic=None, note=""):
    group = _shard.variant_group(slot, variant)
    prim = _shard.PRIMARY_GROUP.get(slot, group)
    if bic is None:
        bic = _shard.bic_per_obs(ll, k if k is not None else -1, n or 0)
    return {
        "stage": stage, "segment": segment, "slot": slot, "variant": variant,
        "group": group, "primary_group": prim,
        "converged": 1 if converged else 0,
        "ll": float(ll) if ll == ll else "",
        "k": k if k is not None else "",
        "n": n if n is not None else "",
        "bic_per_obs": float(bic) if bic == bic else "",
        "winner_in_group": 0, "overall_pick": 0, "note": note,
    }


def _save_model(obj, path: str) -> str:
    try:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        return path
    except Exception as exc:  # noqa: BLE001
        return f"<unsaved: {exc}>"


# ---------------------------------------------------------------------------
# Stage 4 variants
# ---------------------------------------------------------------------------

def run_s4_variants(purpose: str, feats: list, trips, tours, persons,
                    out_dir: str, variants=None) -> list:
    """Refit one purpose's searched spec under each S4 family. Returns rows."""
    from stage4_schedule_placement import (
        prepare_timing_data, MNLTimingModel, RRMTimeModel,
        TDDChoiceModel, _fit_rrm_timing_model, _fit_regret_timing_model,
        _timing_person_features,
    )
    os.makedirs(out_dir, exist_ok=True)
    variants = variants or _shard.s4_variants()
    rows = []

    try:
        data = prepare_timing_data(trips, tours, persons)
    except Exception as exc:  # noqa: BLE001
        return [_row(4, purpose, "s4", v, note=f"timing data prep failed: {exc}")
                for v in variants]

    _nonmand = {'escort', 'shopping', 'social', 'othmaint', 'othdiscr', 'eat'}
    subset = (data[data["purpose"].isin(_nonmand)].copy()
              if purpose == 'nonmand'
              else data[data["purpose"] == purpose].copy())
    allowed = MNLTimingModel.PTYPE_ALLOWLIST.get(purpose)
    if allowed is not None and "ptype" in subset.columns:
        subset = subset[subset["ptype"].isin(allowed)].copy()
    if len(subset) < 30 or ("period" in subset.columns
                            and subset["period"].nunique() < 2):
        return [_row(4, purpose, "s4", v,
                     note=f"insufficient data (n={len(subset)})")
                for v in variants]
    # Searched person features, resolved exactly as the MNL/TDD paths do.
    try:
        person_feats = _timing_person_features(subset, purpose, list(feats or []))
    except Exception:
        person_feats = [f for f in (feats or []) if f in subset.columns]

    def _feat_frame(cols):
        cols = [c for c in cols if c in subset.columns]
        X = pd.DataFrame(index=subset.index)
        for c in cols:
            X[c] = pd.to_numeric(subset[c], errors="coerce").fillna(0.0)
        return X

    # -- rrm (fixed 6-attr regret design; searched feats do not enter) ------
    if "rrm" in variants:
        try:
            m = _fit_rrm_timing_model(subset, purpose)
            if m is not None and m.rrm_ is not None:
                k = len(RRMTimeModel.FEATURE_NAMES)
                rows.append(_row(4, purpose, "s4", "rrm", converged=True,
                                 ll=float(m.ll_), k=k, n=int(m.n_obs_),
                                 note="fixed regret design; searched feats n/a"))
                _save_model(m, os.path.join(out_dir, "best_model_rrm.pkl"))
            else:
                rows.append(_row(4, purpose, "s4", "rrm",
                                 note="RRM fit unavailable/failed"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(4, purpose, "s4", "rrm",
                             note=f"RRM raised: {exc!r}"))

    # -- tdd (joint start/end/duration; searched feats enter) ---------------
    if "tdd" in variants:
        try:
            t = TDDChoiceModel()
            t.purpose_ = purpose
            t.feature_override_ = {purpose: list(person_feats)}
            t.fit(subset)
            if getattr(t, "fit_status_", "") == "fit":
                k = len(getattr(t, "param_names_", []) or [])
                rows.append(_row(4, purpose, "s4", "tdd", converged=True,
                                 ll=float(t.ll_final_), k=k,
                                 n=int(t.n_obs_), note="21 joint alts; own group"))
                _save_model(t, os.path.join(out_dir, "best_model_tdd.pkl"))
            else:
                rows.append(_row(4, purpose, "s4", "tdd",
                                 note=f"TDD status={getattr(t,'fit_status_','?')}"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(4, purpose, "s4", "tdd",
                             note=f"TDD raised: {exc!r}"))

    # -- mnl (plain period MNL on the searched spec) ------------------------
    if "mnl" in variants:
        try:
            m = MNLTimingModel()
            m.purpose_ = purpose
            m.ptypes_ = sorted(subset["ptype"].dropna().unique()) \
                if "ptype" in subset.columns else []
            m.feature_override_ = list(person_feats)
            X = m._build_X(subset)
            y = pd.to_numeric(subset["period"], errors="coerce").dropna().astype(int)
            X = X[y.index] if hasattr(y, "index") else X
            m.fit(np.asarray(X, dtype=float), np.asarray(y))
            ll = float(getattr(m, "ll_", float("nan")))
            k = int(np.asarray(getattr(m, "coef_", [[0]])).size)
            if np.isfinite(ll):
                rows.append(_row(4, purpose, "s4", "mnl", converged=True,
                                 ll=ll, k=k, n=int(len(y)),
                                 note="searched spec"))
                _save_model(m, os.path.join(out_dir, "best_model_mnl.pkl"))
            else:
                rows.append(_row(4, purpose, "s4", "mnl",
                                 note="MNL LL non-finite"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(4, purpose, "s4", "mnl",
                             note=f"MNL raised: {exc!r}"))

    # -- ordered (periods are ordinal: legit ordered logit) -----------------
    if "ordered" in variants:
        try:
            from model_zoo import ModelZoo
            X = _feat_frame(list(person_feats) or ["day_position"])
            if X.shape[1] == 0:
                raise RuntimeError("no numeric features for ordered logit")
            df = X.copy()
            df["period"] = pd.to_numeric(subset["period"],
                                         errors="coerce").fillna(0).astype(int).values
            zoo = ModelZoo(df, feat_cols=list(X.columns), choice_col="period")
            res = zoo.fit_one("sl_ordered")
            ll = float(getattr(res, "loglik", float("nan"))) if res else float("nan")
            k = int(len(getattr(res, "params", []) or [])) if res else 0
            bic = getattr(res, "bic", None)
            bic_n = (float(bic) / len(df)) if bic not in (None, "") \
                and np.isfinite(float(bic)) else None
            if res is not None and np.isfinite(ll):
                rows.append(_row(4, purpose, "s4", "ordered", converged=True,
                                 ll=ll, k=k or len(X.columns), n=int(len(df)),
                                 bic=bic_n, note="SearchLibrium ordered logit"))
                _save_model(res, os.path.join(out_dir, "best_model_ordered.pkl"))
            else:
                rows.append(_row(4, purpose, "s4", "ordered",
                                 note="ordered logit failed/non-finite"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(4, purpose, "s4", "ordered",
                             note=f"ordered raised: {exc!r}"))

    # -- mxl (opt-in random-parameter timing refit) ------------------------
    if "mxl" in variants:
        try:
            from ga_external_estimators import stage4_final_mixed_logit
            n_alts = int(pd.to_numeric(subset["period"], errors="coerce").max()) + 1
            draws = int(os.environ.get("GA_STAGE4_MXL_DRAWS", "100"))
            result = stage4_final_mixed_logit(
                data, list(person_feats), [purpose], n_alts=n_alts,
                n_draws=max(10, draws))
            fit = result.get(purpose)
            if fit and fit.get("converged") and np.isfinite(float(fit.get("bic", np.nan))):
                rv = fit.get("randvars") or {}
                n_rand = len(rv)
                k = max(0, len(person_feats) - n_rand) + 2 * n_rand
                rows.append(_row(4, purpose, "s4", "mxl", converged=True,
                                 ll=float(fit["ll"]), k=k, n=int(fit["n_obs"]),
                                 bic=float(fit["bic"]) / max(int(fit["n_obs"]), 1),
                                 note=f"random params={n_rand}; draws={draws}"))
            else:
                rows.append(_row(4, purpose, "s4", "mxl",
                                 note="MXL fit unavailable/failed"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(4, purpose, "s4", "mxl",
                             note=f"MXL raised: {exc!r}"))

    # -- hybrid (regret engine; reports BIC directly) -----------------------
    if "hybrid" in variants:
        try:
            m = _fit_regret_timing_model(subset, purpose, "hybrid")
            if m is not None and np.isfinite(float(getattr(m, "bic_", float("nan")))):
                n = int(getattr(m, "n_obs_", len(subset)))
                rows.append(_row(4, purpose, "s4", "hybrid", converged=True,
                                 bic=float(m.bic_) / max(n, 1), n=n,
                                 note=f"rule={getattr(m,'rule_','?')}"))
                _save_model(m, os.path.join(out_dir, "best_model_hybrid.pkl"))
            else:
                rows.append(_row(4, purpose, "s4", "hybrid",
                                 note="hybrid unavailable/non-finite BIC"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(4, purpose, "s4", "hybrid",
                             note=f"hybrid raised: {exc!r}"))

    _shard.write_leaderboard(rows, out_dir)
    return rows


# ---------------------------------------------------------------------------
# Stage 5 mode variants (nl vs mnl; mxl/gse recorded from search metadata)
# ---------------------------------------------------------------------------

def run_s5mode_variants(seg: str, mode_genes, mode_cands: list, tours, persons,
                        landuse, skims, out_dir: str, variants=None,
                        distribution_policy=None, search_metadata=None) -> list:
    """Score the winning mode spec under each mode family. Returns rows."""
    from ga_hypothesis_testing import _stage5_mode_only_fitness
    os.makedirs(out_dir, exist_ok=True)
    variants = variants or _shard.s5_mode_variants()
    rows = []
    genes = np.asarray(list(mode_genes), dtype=int)
    n = int(genes.sum())

    # The feature SEARCH leaves large JAX compilation caches resident; each
    # variant refit then rebuilds the mode datatree + a fresh model on top. On
    # a 64GB stage-5 shard that tipped edu into cgroup OOM (job 26463013) once
    # 'lccm' (a 2-class model) was added as a 3rd, heavier variant. Free the
    # caches before the phase and between variants so memory can't accumulate.
    def _free_jax():
        try:
            import ga_external_estimators as _gee
            _gee._maybe_clear_jax_caches()
        except Exception:
            pass
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
        import gc as _gc
        _gc.collect()

    struct_of = {"nl": "nested", "mnl": "flat", "lccm": "lccm"}
    _free_jax()
    for variant in [v for v in ("nl", "mnl", "lccm") if v in variants]:
        try:
            fit = _stage5_mode_only_fitness(
                genes, list(mode_cands), seg, tours, persons, landuse, skims,
                distribution_policy=None, structure=struct_of[variant])
            if fit is not None and np.isfinite(float(fit)) and float(fit) > -1e8:
                # Real ll/k/n recorded by the refit itself. `fit` is the GA
                # SEARCH objective (selected-feature k, GA_COMPLEXITY_PENALTY
                # scaling, minus a significance penalty) — not a BIC — so
                # writing it as bic_per_obs made a 2-class lccm look
                # artificially cheap against nl/mnl. The leaderboard now uses
                # the genuine BIC computed from the fitted ll and the fitted
                # free-parameter count, and keeps the search value in `note`.
                from ga_hypothesis_testing import get_fit_info
                _info = get_fit_info("s5mode", seg, struct_of[variant]) or {}
                if _info.get("ll") is not None and _info.get("k") is not None:
                    rows.append(_row(5, seg, "s5mode", variant, converged=True,
                                     ll=_info["ll"], k=_info["k"],
                                     n=_info.get("n", n),
                                     note=f"structure={struct_of[variant]}; "
                                          f"search_obj={float(fit):.6f}"))
                else:
                    rows.append(_row(5, seg, "s5mode", variant, converged=True,
                                     bic=-float(fit), n=n,
                                     note=f"search fitness -> BIC/n; "
                                          f"structure={struct_of[variant]} "
                                          f"(ll/k unavailable)"))
            else:
                # Log WHY: this branch was silent, so a failing family (notably
                # lccm) only ever surfaced as "family refit failed (see log)"
                # with nothing in the log. Distinguish None vs non-finite vs
                # sentinel so the cause is readable without re-running.
                _why = ("returned None"
                        if fit is None else
                        f"non-finite fitness ({fit!r})"
                        if not np.isfinite(float(fit)) else
                        f"sentinel fitness ({fit!r} <= -1e8)")
                sys.stderr.write(
                    f"[stage5-refit] FAILED segment={seg} variant={variant} "
                    f"structure={struct_of[variant]} reason={_why}\n")
                sys.stderr.flush()
                rows.append(_row(5, seg, "s5mode", variant,
                                 note=f"family refit failed ({_why})"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(5, seg, "s5mode", variant,
                             note=f"{variant} raised: {exc!r}"))
        finally:
            _free_jax()   # release this variant's caches before the next

    # mxl / gse: recorded from the search's own GSE arm when it ran (the arm
    # already refits winner +/-GSE with identical k-counting and picks by BIC/n).
    meta = search_metadata or {}
    if "mxl" in variants and "mxl" not in [r["variant"] for r in rows]:
        rows.append(_row(5, seg, "s5mode", "mxl",
                         note="MXL needs a converged MNL base; enable the "
                              "standalone MXL refit only after nl/mnl converge"))
    if "gse" in variants:
        if "gse_selected" in meta:
            try:
                bpo = float(meta.get("gse_bic", "nan"))
                rows.append(_row(
                    5, seg, "s5mode", "gse",
                    converged=bool(meta.get("gse_selected")) or np.isfinite(bpo),
                    bic=(bpo if np.isfinite(bpo) else None), n=n,
                    note="from search GSE arm "
                         f"(selected={bool(meta.get('gse_selected'))})"))
            except Exception as exc:  # noqa: BLE001
                rows.append(_row(5, seg, "s5mode", "gse",
                                 note=f"GSE metadata unreadable: {exc!r}"))
        else:
            rows.append(_row(5, seg, "s5mode", "gse",
                             note="GSE arm did not run "
                                  "(needs GA_SEARCH_GSE=1 + mixtures)"))

    # Persist the winning SPEC (genes -> vars + structure); the full refit
    # happens in merge/predict from this spec, same as the legacy GA path.
    # NOTE: leaderboard writing is left to the caller (_run_ga_shard writes
    # mode + dest rows together) so one family never overwrites the other.
    try:
        spec = {"segment": seg,
                "mode_vars": [v for v, g in zip(mode_cands, genes) if int(g) == 1],
                "structures_scored": [v for v in ("nl", "mnl", "lccm") if v in variants]}
        with open(os.path.join(out_dir, "winner_spec_mode.json"), "w",
                  encoding="utf-8") as fh:
            import json
            json.dump(spec, fh, indent=2)
    except Exception:
        pass
    return rows  # caller writes the combined leaderboard (mode + dest)


# ---------------------------------------------------------------------------
# Stage 5 dest variants (mnl vs nl_dist; exploded diagnostic opt-in)
# ---------------------------------------------------------------------------

def run_s5dest_variants(seg: str, dest_genes, dest_cands: list, seg_tours,
                        persons, lu_st_cached, skims, ls_cached, seg_acdc,
                        out_dir: str, variants=None,
                        warm_start=None, starter_uec=None) -> list:
    """Score the winning dest spec under each dest family. Returns rows."""
    from ga_hypothesis_testing import _stage5_dest_only_fitness, get_fit_info
    os.makedirs(out_dir, exist_ok=True)
    variants = variants or _shard.s5_dest_variants()
    rows = []
    genes = np.asarray(list(dest_genes), dtype=int)

    env_map = {"mnl": None, "nl_dist": "1"}
    for variant in [v for v in ("mnl", "nl_dist") if v in variants]:
        prev = os.environ.get("STAGE5_DEST_NL_DISTANCE")
        try:
            if env_map[variant] is None:
                os.environ.pop("STAGE5_DEST_NL_DISTANCE", None)
            else:
                os.environ["STAGE5_DEST_NL_DISTANCE"] = env_map[variant]
            fit = _stage5_dest_only_fitness(
                genes, list(dest_cands), seg, seg_tours, persons,
                lu_st_cached, skims, ls_cached, seg_acdc,
                warm_start=warm_start, custom_uec=None)
            if fit is not None and np.isfinite(float(fit)) and float(fit) > -1e8:
                # Genuine BIC from the fitted ll/k (see run_s5mode_variants).
                # nl_dist and mnl differ in model form, so the search objective
                # alone cannot adjudicate between them.
                _dinfo = get_fit_info("s5dest", seg, "dest") or {}
                if _dinfo.get("ll") is not None and _dinfo.get("k") is not None:
                    rows.append(_row(5, seg, "s5dest", variant, converged=True,
                                     ll=_dinfo["ll"], k=_dinfo["k"],
                                     n=_dinfo.get("n", int(genes.sum())),
                                     note=("distance-band NL" if variant == "nl_dist"
                                           else "sampled MNL")
                                          + f"; search_obj={float(fit):.6f}"))
                else:
                    rows.append(_row(5, seg, "s5dest", variant, converged=True,
                                     bic=-float(fit), n=int(genes.sum()),
                                     note="search fitness -> BIC/n; "
                                          + ("distance-band NL" if variant == "nl_dist"
                                             else "sampled MNL")
                                          + " (ll/k unavailable)"))
            else:
                rows.append(_row(5, seg, "s5dest", variant,
                                 note="family refit failed (see log)"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(5, seg, "s5dest", variant,
                             note=f"{variant} raised: {exc!r}"))
        finally:
            if prev is None:
                os.environ.pop("STAGE5_DEST_NL_DISTANCE", None)
            else:
                os.environ["STAGE5_DEST_NL_DISTANCE"] = prev

    if "exploded" in variants:
        rows.append(_row(5, seg, "s5dest", "exploded",
                         note="exploded/rank-ordered diagnostic runs in the "
                              "stage5b trip-dest path (smoke_exploded_logit); "
                              "its rank-expanded likelihood is reported there, "
                              "never auto-compared here"))

    try:
        spec = {"segment": seg,
                "dest_vars": [v for v, g in zip(dest_cands, genes) if int(g) == 1],
                "families_scored": [v for v in ("mnl", "nl_dist") if v in variants]}
        with open(os.path.join(out_dir, "winner_spec_dest.json"), "w",
                  encoding="utf-8") as fh:
            import json
            json.dump(spec, fh, indent=2)
    except Exception:
        pass
    return rows  # caller writes the combined leaderboard (mode + dest)


def run_exploded_diagnostic(seg: str, trips, landuse, skims_path: str,
                            out_dir: str, purpose: str = "work",
                            max_tours: int = 300, max_alts: int = 20) -> dict:
    """Best-effort exploded-logit diagnostic on a capped sample.

    Guarded: any failure returns {'converged': False, 'note': ...} and the
    leaderboard keeps a non-comparable diagnostic row. Never raises.
    """
    try:
        from smoke_exploded_logit import (build_exploded_logit_data,
                                          fit_exploded_logit)
        from stage5b_trip_destination import _load_skims
    except Exception as exc:  # noqa: BLE001
        return {"converged": False, "note": f"imports unavailable: {exc!r}"}
    try:
        os.makedirs(out_dir, exist_ok=True)
        sample = trips.head(max_tours).copy()
        skims = _load_skims(skims_path)
        zone_col = next((c for c in ("zone_id", "DTAZ", "SA2") if c in
                         landuse.columns), landuse.columns[0])
        zones = landuse[[zone_col]].drop_duplicates().reset_index(drop=True)
        taz_index = {int(z): i for i, z in enumerate(zones[zone_col])}
        data = build_exploded_logit_data(
            sample, zones, skims, landuse, taz_index, purpose,
            zone_id_col=zone_col, max_alts=max_alts)
        uvars = [c for c in ("detour_dist_km", "log_size", "AutoIVT_AM",
                             "NetworkDist", "AutoCrashRisk", "TransitIVT_AM")
                 if c in data.columns]
        res = fit_exploded_logit(data, uvars, origin_conditioning="interactions")
        if not res:
            return {"converged": False, "note": "exploded fit returned None"}
        with open(os.path.join(out_dir, f"exploded_{seg}.pkl"), "wb") as fh:
            pickle.dump({"params": res.get("params"),
                         "summary": {k: res.get(k) for k in
                                     ("bic", "n_cases", "n_params", "engine")}}, fh)
        return {"converged": True, "bic": float(res.get("bic", float("nan"))),
                "n": int(res.get("n_cases", 0)), "k": int(res.get("n_params", 0)),
                "note": "diagnostic only: rank-expanded likelihood, "
                        "not comparable to zone-MNL BIC/n"}
    except Exception as exc:  # noqa: BLE001
        return {"converged": False,
                "note": f"exploded diagnostic failed: {exc!r}\n"
                        f"{traceback.format_exc(limit=3)}"}
