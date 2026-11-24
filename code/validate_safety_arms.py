#!/usr/bin/env python
"""Holdout comparison of the safety model (M_S) against a crash-term ablation
(M_0) using observed HHTS choices.

This is the missing "does safety capture reality better?" check.  It is NOT a
refit of M_0: M_0 here is the same fitted specification with every crash
coefficient set to zero.  That isolates the behavioural effect of the crash
utility on the same coefficients.  A paper-grade M_0 (crash terms removed and
refitted) should be run separately; label this run accordingly.

Outputs
-------
  <out>/safety_validation_metrics.csv
  <out>/safety_validation_metrics.json
  <out>/safety_validation_panel.png

Usage (on an HPC node, not a login node)
---------------------------------------
  python validate_safety_arms.py \
      --run-dir runs/safety_baseline__26958837.aqua \
      --out runs/safety_validation --split random --bootstrap 100
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, ks_2samp, wasserstein_distance

from safety_crash_measure import apply as apply_crash_measures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage5_destination_choice import (
    load_models, _prep_hhts_tours, _filter_seg, _mode_tree,
    _build_mode_model, _dest_tree, _build_dest_model, _logsums,
    _attach_size_term, _ensure_destination_clusters, _skim_flat, _open_skims,
    _bridge_tours_zones_auto,
)

QUT = {"navy": "#003B71", "sky": "#0077C8", "amber": "#F2A900",
       "green": "#3E9B4F", "red": "#C0392B", "grey": "#5C6773"}
MODE_NAMES = {1: "DA", 2: "SR", 3: "Walk", 4: "Bike", 5: "Transit"}


# ---------------------------------------------------------------------------
# helpers copied from validate_hhts.py (the fitted-coefficient rebuild path)
# ---------------------------------------------------------------------------
def _free_vars_and_coefs(mdl):
    try:
        pf = getattr(mdl, "_final_pf", None)
        if pf is None:
            pf = mdl.pf
        hold = (pf["holdfast"].fillna(False).astype(bool)
                if "holdfast" in pf.columns
                else pd.Series(False, index=pf.index))
        free = [str(p) for p in pf.index if not bool(hold.get(p, False))]
        structural = ("ASC_", "Mu_")
        exact = ("c_size_term", "log_correction")
        variables = [p for p in free
                     if not p.startswith(structural) and p not in exact]
        values = {}
        for p in pf.index:
            try:
                v = float(pf.loc[p].get("value", float("nan")))
                if np.isfinite(v):
                    values[str(p)] = v
            except Exception:
                pass
        return variables, values
    except Exception:
        return [], {}


def _utility_param_names(mdl):
    """Parameter names literally present in the fitted utility expression.

    A cloudpickled Stage-5 model can have an empty ``pf`` after load while its
    ``_coef_cache`` still carries the *default starting priors* for every
    candidate term. Falling back to that cache would silently add terms that
    were never in the fitted specification (P0: 44 priors for a 4-term model),
    which collapses destination predictions. Parse the utility expression
    instead, so prediction applies exactly the fitted terms.
    """
    try:
        # Mode models put their utility in utility_co (one linear function per
        # alternative); destination models use utility_ca.
        expr = " ".join(str(getattr(mdl, _a, "") or "")
                        for _a in ("utility_ca", "utility_co"))
    except Exception:
        return []
    return list(dict.fromkeys(re.findall(r"P\.([A-Za-z_]\w*)", expr)))


def _term_values(mdl, coefs: dict) -> dict:
    """Values for only the terms in the fitted utility expression."""
    terms = _utility_param_names(mdl)
    if not terms:
        return dict(coefs or {})
    return {k: float(v) for k, v in (coefs or {}).items() if k in set(terms)}


def _holdfast_names(mdl):
    """Parameter names production fixed (holdfast) on this model."""
    try:
        pf = getattr(mdl, "_final_pf", None)
        if pf is None:
            pf = mdl.pf
        hold = (pf["holdfast"].fillna(False).astype(bool)
                if "holdfast" in pf.columns
                else pd.Series(False, index=pf.index))
        return {str(p) for p in pf.index if bool(hold.get(p, False))}
    except Exception:
        return set()


def _apply_coefs(mdl, coefs: dict, holdfast=None) -> int:
    """Apply fitted values AND re-lock what production held fast.

    Setting values alone silently frees every holdfasted term (39 of 46 on
    the work destination model, most at exactly 0.0), turning unidentified
    columns into free parameters: measured |grad|inf 386 against 2.9e-4 for
    the same fit with the flags carried, and no stationary point reached.
    """
    n = 0
    for k, v in (coefs or {}).items():
        try:
            mdl.set_value(str(k), float(v))
            n += 1
        except Exception:
            pass
    for name in sorted(set(holdfast or ())):
        try:
            from stage5_destination_choice import _pf_set
            _pf_set(mdl, str(name), "holdfast", True)
        except Exception:
            try:
                mdl.lock_value(str(name), float((coefs or {}).get(name, 0.0)))
            except Exception:
                pass
    return n


def _without_crash(coefs: dict) -> dict:
    return {k: (0.0 if "crash" in str(k).lower() else v)
            for k, v in (coefs or {}).items()}


def _js(a, b) -> float:
    try:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if a.sum() <= 0 or b.sum() <= 0:
            return float("nan")
        return float(jensenshannon(a / a.sum(), b / b.sum()))
    except Exception:
        return float("nan")


def _top25(a: pd.Series, b: pd.Series) -> float:
    ta = set(a.sort_values(ascending=False).head(25).index)
    tb = set(b.sort_values(ascending=False).head(25).index)
    return float(len(ta & tb) / 25.0) if ta else float("nan")


def _safe_corr(o: pd.Series, p: pd.Series):
    k = (o > 0) & (p > 0)
    if int(k.sum()) < 3:
        return float("nan"), float("nan")
    lo, lp = np.log(o[k].values), np.log(p[k].values)
    r = float(np.corrcoef(lo, lp)[0, 1])
    rho = float(spearmanr(lo, lp).statistic)
    return r, rho


# ---------------------------------------------------------------------------
# holdout construction
# ---------------------------------------------------------------------------
def _holdout_tours(tours, persons, frac, seed, split):
    rng = np.random.default_rng(seed)
    pids = pd.unique(persons["person_id"].dropna().astype(str))
    n_ho = max(1, int(round(len(pids) * frac)))
    if split == "temporal":
        cols = ("TRAVYEAR", "TRAVMONTH", "TRAVDATE")
        if all(c in persons.columns for c in cols):
            y = pd.to_numeric(persons["TRAVYEAR"], errors="coerce")
            m = pd.to_numeric(persons["TRAVMONTH"], errors="coerce")
            d = pd.to_numeric(persons["TRAVDATE"], errors="coerce")
            key = (y * 10000 + m * 100 + d).groupby(
                persons["person_id"].astype(str)).max()
            key = key.dropna().sort_values()
            if len(key) >= n_ho:
                ho = set(key.index[-n_ho:])
                return ho, "temporal"
        # fall back
    ho = set(rng.choice(pids, size=n_ho, replace=False).tolist())
    return ho, "random(fallback)" if split == "temporal" else "random"


# ---------------------------------------------------------------------------
# one segment, one arm
# ---------------------------------------------------------------------------
def _od_skim_vars(coefs: dict) -> list:
    """Skim-matrix names that the given coefficients' utility terms reference.

    _mode_spec_custom() gates an `od.` term on the SKIM name (e.g. "CRASH"),
    not on the coefficient name: it keeps the term only when the matrix name is
    in the caller's variable list.  Passing the fitted free-parameter names
    alone (c_crash_veh, ...) therefore silently DROPPED log(od.CRASH+1) from the
    rebuilt mode tree, so zeroing the mode crash coefficient changed nothing.
    """
    out = []
    try:
        from stage5_destination_choice import MODE_UEC
    except Exception:
        return out
    for k in (coefs or {}):
        spec = MODE_UEC.get(str(k))
        expr = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
        if not isinstance(expr, str):
            continue
        i = expr.find("od.")
        if i < 0:
            continue
        rest = expr[i + 3:]
        j = 0
        while j < len(rest) and (rest[j].isalnum() or rest[j] == "_"):
            j += 1
        name = rest[:j]
        if name and name not in out:
            out.append(name)
    return out


def _predict_segment(seg, seg_df, seg_persons, models, landuse, skims,
                     skim_df, zero_crash: bool, crash_cols: dict):
    res = models.get(seg) or {}
    mode_fit = res.get("mode_model")
    dest_fit = res.get("dest_model")
    if mode_fit is None or dest_fit is None:
        raise RuntimeError(f"{seg}: fitted mode/dest model missing")

    # HHTS tours are ORDER-coded while the 371-zone skims are SA2-coded.
    # _mode_tree/_dest_tree bridge internally, but the returned `otaz`/`dtaz`
    # and the destination labels must use the skimmed keyspace too, otherwise
    # crash and distance exposure maps miss every OD key.
    seg_df = _bridge_tours_zones_auto(
        seg_df, skims, skim_df, f"_predict_segment[{seg}]")

    mv, mc = _free_vars_and_coefs(mode_fit)
    if not mv or not mc:
        _all_mc = dict(res.get("mode_coef") or {})
        mc = _term_values(mode_fit, _all_mc) or _all_mc
        mv = [k for k in mc
              if not str(k).startswith(("ASC_", "Mu_"))
              and str(k) not in ("c_size_term", "log_correction")]
    if zero_crash:
        mc = _without_crash(mc)
    dt_m = _mode_tree(seg_df, seg_persons, landuse, skims, skim_df=skim_df)
    # Prefer the exact specification saved at estimation time.  Re-deriving it
    # from coefficient names is lossy: gated terms are dropped and generated
    # terms are missed, so the rebuilt model no longer reproduces the fit.
    _saved_mode_uec = res.get("mode_uec")
    if isinstance(_saved_mode_uec, dict) and _saved_mode_uec:
        mode_spec = {
            str(k): (tuple(v) if isinstance(v, (list, tuple)) else v)
            for k, v in _saved_mode_uec.items()
        }
        if zero_crash:
            mode_spec = {k: v for k, v in mode_spec.items()
                         if "crash" not in str(k).lower()}
    else:
        # Mode utilities use the gated candidate-list path: _mode_spec_custom
        # knows how to map feature names to the mode UEC and always-on backbone.
        # Building a literal MODE_UEC dict here would drop generated coefficients
        # (e.g. c_age_active_gen) that are not keys of MODE_UEC.
        mode_vars = list(mv) + [v for v in _od_skim_vars(mc) if v not in mv]
        mode_spec = mode_vars
    # Keep a flat name list for the diagnostic payload regardless of which
    # branch built the specification.
    mode_vars = (list(mode_spec) if isinstance(mode_spec, dict)
                 else list(mode_spec))
    # Rebuild exactly as _run_segment builds at estimation: the availability
    # hint spans tour AND person columns (licence/vehicles gate DA), the same
    # in-vehicle-time encoding the bundle was fitted with, and the same
    # chosen-but-unavailable repair. Any difference makes this a different
    # model, which _score_stage5_arm's LL-reproduction guard now catches.
    _ivt_prev = os.environ.get("STAGE5_POOLED_IVT")
    if not isinstance(_saved_mode_uec, dict) or not _saved_mode_uec:
        os.environ["STAGE5_POOLED_IVT"] = "1" if "c_ivt" in (mc or {}) else "0"
    try:
        mm = _build_mode_model(
            dt_m, seg, custom_mode_vars=mode_spec,
            avail_vars_hint=list(seg_df.columns) + list(seg_persons.columns))
    finally:
        if _ivt_prev is None:
            os.environ.pop("STAGE5_POOLED_IVT", None)
        else:
            os.environ["STAGE5_POOLED_IVT"] = _ivt_prev
    try:
        mm.doctor(repair_ch_av="+")
    except Exception:
        pass
    n_mode_applied = _apply_coefs(mm, mc, holdfast=_holdfast_names(mode_fit))
    try:
        mode_crash_param = bool("c_crash_veh" in mm.pf.index)
    except Exception:
        mode_crash_param = False
    mp = np.asarray(mm.probability(), dtype=float)
    mp = np.nan_to_num(mp, nan=0.0, posinf=0.0, neginf=0.0)
    rs = mp.sum(axis=1, keepdims=True)
    mp = mp / np.where(rs > 0, rs, 1.0)

    # The mode logsum must be evaluated over the REAL destination zone IDs.
    # Without an explicit override, _logsums can fall back to Larch's internal
    # positional alt codes (0..J-1); _dest_tree then uses those as `_altid_`,
    # so every predicted destination share is mislabelled and the crash and
    # distance exposure maps miss (all-NaN metrics in the what-if pass).
    _logsum_zones = None
    try:
        _zone_set = set()
        if skim_df is not None and "DTAZ" in skim_df.columns:
            _zone_set.update(
                int(z) for z in pd.to_numeric(
                    skim_df["DTAZ"], errors="coerce").dropna().unique()
                if int(z) > 0
            )
        # Keep any observed destination (including the 999 fringe sentinel)
        # in the alternative set even when the OD skim omits it.
        if "DTAZ" in seg_df.columns:
            _zone_set.update(
                int(z) for z in pd.to_numeric(
                    seg_df["DTAZ"], errors="coerce").dropna().unique()
                if int(z) > 0
            )
        _logsum_zones = sorted(_zone_set) if len(_zone_set) >= 10 else None
    except Exception:
        _logsum_zones = None
    ls = _logsums(mm, dt_m, skims=skims, verbose=False,
                  zone_ids_override=_logsum_zones)
    # _run_segment standardises the logsum surface (mean 0, unit within-case
    # std) before destination estimation and stores those moments in the model
    # bundle.  Prediction must use the same transform; feeding the raw
    # logsum back in changes c_logsum's scale completely (P0: raw range
    # -827..297 against a fitted coefficient of +0.57).
    _ls_mean = float(res.get("logsum_mean", 0.0) or 0.0)
    _ls_std = float(res.get("logsum_std", 1.0) or 1.0)
    if np.isfinite(_ls_std) and _ls_std > 1e-12:
        ls = (ls - _ls_mean) / _ls_std

    # _mode_tree quarantines infeasible cases (e.g. DA chosen by a zero-vehicle
    # tour).  Align the tour frame and the mode probabilities to the retained
    # cases before building the destination tree.
    def _case_ids(obj):
        try:
            return [str(c) for c in obj.coords["cases"].values]
        except Exception:
            pass
        try:
            return [str(c) for c in obj["case_id"].values]
        except Exception:
            return None

    mode_ids = _case_ids(ls) or _case_ids(dt_m)
    if mode_ids:
        mp_df = pd.DataFrame(mp, index=mode_ids)
        keep = seg_df["ptour_id"].astype(str).isin(mode_ids)
        seg_df = seg_df.loc[keep].copy()
        seg_df["_cid"] = seg_df["ptour_id"].astype(str)
        seg_df = seg_df.set_index("_cid").loc[mode_ids].reset_index(drop=True)
        mp = mp_df.reindex(mode_ids).to_numpy()

    # The pipeline attaches the size term before building the destination tree
    # (stage5: lu_with_st = _ensure_destination_clusters(_attach_size_term(...))).
    # get_land() returns raw landuse with no 'size_term' column, and _dest_tree
    # zero-fills missing landuse columns, so skipping this step silently drops
    # c_size_term from the rebuilt utility and the model loses all attraction:
    # Brisbane City (TOTEMP 143k, size-term rank 1 of 371) was predicted 0.65
    # work tours against 81 observed, and predicted zone counts correlated
    # NEGATIVELY with employment (r = -0.19) while observed counts were +0.31.
    lu_seg = _ensure_destination_clusters(_attach_size_term(landuse, seg))
    dt_d = _dest_tree(seg_df, seg_persons, lu_seg, skims, ls,
                      sampled_alts=None, skim_df=skim_df)
    dv, dc = _free_vars_and_coefs(dest_fit)
    if not dv or not dc:
        _all_dc = dict(res.get("dest_coef") or {})
        dc = _term_values(dest_fit, _all_dc) or _all_dc
        dv = [k for k in dc
              if not str(k).startswith(("ASC_", "Mu_"))
              and str(k) not in ("c_size_term", "log_correction")]
    if zero_crash:
        dc = _without_crash(dc)
    # Prefer the exact destination specification saved with the bundle.
    _saved_dest_uec = res.get("dest_uec")
    if isinstance(_saved_dest_uec, dict) and _saved_dest_uec:
        dv = {
            str(k): (tuple(v) if isinstance(v, (list, tuple)) else v)
            for k, v in _saved_dest_uec.items()
        }
        if zero_crash:
            dv = {k: v for k, v in dv.items()
                  if "crash" not in str(k).lower()}
    else:
        # Same exactness rule as the mode model: pass a UEC dict containing only
        # the fitted terms, never a list that re-adds always-on candidate defaults.
        _dest_terms = _utility_param_names(dest_fit)
        if _dest_terms:
            from stage5_destination_choice import DEST_UEC_FLAT
            _exact_dest = {k: DEST_UEC_FLAT[k]
                           for k in _dest_terms if k in DEST_UEC_FLAT}
            if _exact_dest:
                dv = _exact_dest
    dm = _build_dest_model(dt_d, seg, custom_dest_vars=dv)
    _apply_coefs(dm, dc, holdfast=_holdfast_names(dest_fit))
    dp = np.asarray(dm.probability(), dtype=float)
    dp = np.nan_to_num(dp, nan=0.0, posinf=0.0, neginf=0.0)
    rs = dp.sum(axis=1, keepdims=True)
    dp = dp / np.where(rs > 0, rs, 1.0)

    dest_ids = _case_ids(dt_d)
    if dest_ids and len(dest_ids) != len(seg_df):
        mp_df = pd.DataFrame(mp, index=seg_df["ptour_id"].astype(str))
        seg_df = seg_df.set_index(seg_df["ptour_id"].astype(str)).loc[dest_ids].reset_index(drop=True)
        mp = mp_df.reindex(dest_ids).fillna(0.0).to_numpy()

    # Alternative labels must be ZONE IDs, not Larch's internal 0..J-1 codes.
    # The flat _dest_tree dataset stores the real zone id in the `_altid_`
    # coordinate; `dm.alternatives` can return positional codes depending on
    # how the model was constructed, which silently mislabels every predicted
    # share and makes crash/distance exposure maps miss (all-NaN metrics).
    def _unique_int_labels(values):
        out, seen = [], set()
        for value in np.asarray(values).ravel():
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                continue
            if ivalue not in seen:
                seen.add(ivalue)
                out.append(ivalue)
        return out

    zone_candidates = []
    for _source in (
        lambda: dt_d.coords["_altid_"].values,
        lambda: dt_d.coords["zones"].values,
        lambda: ls.coords["zones"].values,
        lambda: dm.alternatives,
    ):
        try:
            zone_candidates.append(_unique_int_labels(_source()))
        except Exception:
            continue

    # Both Larch's internal alternative codes (0..J-1) and the real SA2 labels
    # can have length J, so length alone cannot choose the right one. Score
    # each candidate by overlap with the OD skim's own zone IDs and take the
    # best; this is what makes crash/distance exposure maps key correctly.
    _ref_ids = set()
    try:
        if skim_df is not None:
            for _col in ("OTAZ", "DTAZ"):
                if _col in skim_df.columns:
                    _ref_ids.update(
                        int(z) for z in pd.to_numeric(
                            skim_df[_col], errors="coerce").dropna().unique()
                        if int(z) > 0
                    )
    except Exception:
        _ref_ids = set()

    _matching = [c for c in zone_candidates if len(c) == dp.shape[1]]
    if _matching:
        zones = max(_matching, key=lambda c: len(set(c) & _ref_ids))
        if _ref_ids and not (set(zones) & _ref_ids):
            print(f"  WARNING [{seg}]: {dp.shape[1]} destination probabilities "
                  "have no zone labels overlapping the OD skim; exposure maps "
                  "will be empty")
    else:
        print(f"  WARNING [{seg}]: could not align {dp.shape[1]} destination "
              "probabilities to zone labels; falling back to positional codes")
        zones = list(range(dp.shape[1]))

    return {
        "mode_probs": mp,
        "dest_probs": dp,
        "zones": np.asarray(zones, dtype=int),
        "person_id": seg_df["person_id"].astype(str).to_numpy(),
        "weight": (pd.to_numeric(seg_df["weight"], errors="coerce")
                   .fillna(0.0).to_numpy()
                   if "weight" in seg_df.columns
                   else np.ones(len(seg_df), dtype=float)),
        "tour_mode": pd.to_numeric(seg_df["tour_mode"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "otaz": pd.to_numeric(seg_df["OTAZ"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "dtaz": pd.to_numeric(seg_df["DTAZ"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "mode_coefs_applied": int(n_mode_applied),
        "mode_crash_coef_present": bool("c_crash_veh" in mc),
        "mode_crash_param_present": bool(mode_crash_param),
        "mode_vars_passed": " ".join(str(v) for v in mode_vars),
    }


def _crash_matrix(skim_df):
    if skim_df is None or "CRASH" not in skim_df.columns:
        return {}
    d = skim_df[["OTAZ", "DTAZ", "CRASH"]].dropna()
    return {(int(o), int(dz)): float(c)
            for o, dz, c in d.itertuples(index=False, name=None)}


def _expected_exposure(pred, crash_map):
    """Per-tour expected crash rate at the chosen destination."""
    dp = np.asarray(pred["dest_probs"], dtype=float)
    zones = np.asarray(pred["zones"], dtype=int)
    otaz = np.asarray(pred["otaz"], dtype=int)
    out = np.full(len(otaz), np.nan)
    for i, o in enumerate(otaz):
        vals = np.array([crash_map.get((int(o), int(z)), np.nan)
                         for z in zones], dtype=float)
        ok = np.isfinite(vals)
        if ok.any():
            p = np.where(ok, dp[i], 0.0)
            s = p.sum()
            if s > 0:
                out[i] = float(np.sum(p * np.nan_to_num(vals)) / s)
    return out


def _ensure_crash_in_skim(skim_df):
    """Make od.CRASH real for BOTH trees, not just the destination tree.

    _build_od_dataset() (the mode tree's OD source) creates a CRASH *placeholder*
    of zeros when the flat skims carry no crash column; _dest_tree separately
    self-loads the crash OD table. So without this merge, log(od.CRASH+1) is 0
    for every OD pair in the mode model, zeroing c_crash_veh cannot change mode
    probabilities, and the M_0 ablation silently stops testing the mode-side
    crash coefficient.  That is only one of two ways the mode side loses the
    crash term: _mode_spec_custom() also gates `od.` terms on the SKIM name
    ("CRASH") rather than the coefficient name, so passing free-parameter names
    alone drops log(od.CRASH+1) from the rebuilt tree (fixed by _od_skim_vars).
    """
    diag = {"merged": False, "source": None,
            "nonzero_share": float("nan"), "max": float("nan")}
    v = None
    if "CRASH" in skim_df.columns:
        v = pd.to_numeric(skim_df["CRASH"], errors="coerce").fillna(0.0)
        diag["nonzero_share"] = float((v > 0).mean())
        diag["max"] = float(v.max()) if len(v) else float("nan")
    if v is not None and diag["nonzero_share"] > 0:
        return skim_df, diag

    try:
        from stage5_destination_choice import (_read_crash_od_table,
                                               _resolve_default_crash_source)
        src = _resolve_default_crash_source()
        if not src:
            diag["error"] = "no default crash source found"
            print("WARNING: no crash OD source found - mode crash term stays zero")
            return skim_df, diag
        tbl = _read_crash_od_table(src)
        if tbl is None or len(tbl) == 0:
            diag["error"] = "crash table empty: " + str(src)
            print("WARNING: crash table empty (" + str(src) + ")")
            return skim_df, diag
        if "CRASH" in skim_df.columns:
            skim_df = skim_df.drop(columns=["CRASH"])
        skim_df = skim_df.merge(tbl, on=["OTAZ", "DTAZ"], how="left")
        skim_df["CRASH"] = pd.to_numeric(
            skim_df["CRASH"], errors="coerce").fillna(0.0)
        c = skim_df["CRASH"]
        diag.update(merged=True, source=str(src),
                    nonzero_share=float((c > 0).mean()),
                    max=float(c.max()) if len(c) else float("nan"))
        print("crash skim merged from " + str(src) + ": nonzero="
              + format(diag["nonzero_share"], ".1%")
              + " max=" + format(diag["max"], ".4g"))
    except Exception as exc:
        diag["error"] = type(exc).__name__ + ": " + str(exc)
        print("WARNING: could not merge crash skims: " + diag["error"])
    return skim_df, diag


def repair_missing_dist(skim_df, verbose=True):
    """Fill the missing car distances in the 371-zone skim.

    Nine zones have DIST == 0 from every origin (including their own diagonal)
    while carrying positive walk and bike distances, and about 1.4\\% of HHTS
    tours do use them.  The destination utility is linear in DIST with a
    negative coefficient, so a zero is the *best possible* distance: those nine
    zones absorbed about a third of predicted work destination mass.  They must
    stay in the choice set (observed tours choose them), so this repairs the
    missing record rather than marking them unavailable.

    Rule, identical in spirit to the crash-matrix fill: replace the zero with
    the origin's mean positive distance, i.e. the distance a tour from that
    origin would face at an average destination.  Nothing is fitted.  Origins
    that are themselves car-less fall back to the global mean positive distance.
    """
    if skim_df is None or "DIST" not in skim_df.columns:
        return skim_df
    if os.environ.get("SAFETY_REPAIR_MISSING_DIST", "1").strip() == "0":
        return skim_df
    d = pd.to_numeric(skim_df["DIST"], errors="coerce")
    bad = ~np.isfinite(d) | (d <= 0)
    if not bool(bad.any()):
        return skim_df
    pos = ~bad
    if not bool(pos.any()):
        return skim_df
    per_origin = d[pos].groupby(skim_df.loc[pos, "OTAZ"]).mean()
    global_mean = float(d[pos].mean())
    out = skim_df.copy()
    fill = out.loc[bad, "OTAZ"].map(per_origin)
    out.loc[bad, "DIST"] = fill.fillna(global_mean).to_numpy()
    if verbose:
        n_bad = bad.groupby(out["DTAZ"]).sum()
        n_all = out.groupby("DTAZ").size()
        n_dest = int((n_bad.reindex(n_all.index).fillna(0) == n_all).sum())
        print(f"repaired {int(bad.sum()):,} zero-distance OD pairs "
              f"({n_dest} zones with no car-network distance) -> "
              f"origin mean {float(fill.mean()):.1f} km "
              f"[{float(fill.min()):.1f}-{float(fill.max()):.1f}]",
              flush=True)
    return out


def _dist_matrix(skim_df):
    if skim_df is None or "DIST" not in skim_df.columns:
        return {}
    d = skim_df[["OTAZ", "DTAZ", "DIST"]].dropna()
    return {(int(o), int(dz)): float(c)
            for o, dz, c in d.itertuples(index=False, name=None)}


def _metrics(pred, crash_map, dist_map, active_codes=(3, 4)):
    mp = pred["mode_probs"]
    dp = pred["dest_probs"]
    zones = pred["zones"]
    obs_mode = pred["tour_mode"]
    obs_dest = pred["dtaz"]
    otaz = pred["otaz"]
    n = len(obs_mode)

    # ---- mode ----
    obs_idx = np.clip(obs_mode - 1, 0, 4)
    p_obs = mp[np.arange(n), obs_idx]
    p_obs = np.clip(p_obs, 1e-12, 1.0)
    onehot = np.zeros_like(mp)
    onehot[np.arange(n), obs_idx] = 1.0
    mode_logloss = float(-np.mean(np.log(p_obs)))
    mode_brier = float(np.mean(np.sum((mp - onehot) ** 2, axis=1)))
    mode_acc = float(np.mean(np.argmax(mp, axis=1) == obs_idx))
    obs_share = np.bincount(obs_mode, minlength=6)[1:6] / max(n, 1)
    pred_share = mp.mean(axis=0)
    mode_mae_pp = float(np.mean(np.abs(obs_share - pred_share)) * 100)
    active_obs = float(np.mean(np.isin(obs_mode, active_codes)))
    active_pred = float(np.sum(mp[:, [c - 1 for c in active_codes]]) / max(n, 1))

    # ---- destination shares ----
    obs_counts = pd.Series(obs_dest).value_counts(normalize=True).sort_index()
    pred_counts = pd.Series(dp.sum(axis=0), index=zones).sort_index()
    pred_counts = pred_counts / max(pred_counts.sum(), 1e-12)
    zunion = sorted(set(obs_counts.index) | set(pred_counts.index))
    ov = obs_counts.reindex(zunion).fillna(0.0).values
    pv = pred_counts.reindex(zunion).fillna(0.0).values
    dest_mae = float(np.mean(np.abs(ov - pv)))
    dest_js = _js(ov, pv)
    dest_top25 = _top25(obs_counts, pred_counts)
    r_log, rho = _safe_corr(obs_counts, pred_counts)
    # probability of the observed zone (if present in the alternatives)
    z_to_col = {int(z): i for i, z in enumerate(zones)}
    p_obs_dest = []
    for d in obs_dest:
        j = z_to_col.get(int(d))
        p_obs_dest.append(dp[len(p_obs_dest), j] if j is not None else 1e-12)
    dest_logloss = float(-np.mean(np.log(np.clip(p_obs_dest, 1e-12, 1.0))))

    # ---- crash exposure of chosen destination ----
    def _crash(o, d):
        return crash_map.get((int(o), int(d)), np.nan)
    obs_crash = np.array([_crash(o, d) for o, d in zip(otaz, obs_dest)], float)
    exp_crash = np.full(n, np.nan)
    arg_crash = np.full(n, np.nan)
    for i, o in enumerate(otaz):
        vals = np.array([crash_map.get((int(o), int(z)), np.nan) for z in zones], float)
        ok = np.isfinite(vals)
        if ok.any():
            p = np.where(ok, dp[i], 0.0)
            s = p.sum()
            if s > 0:
                exp_crash[i] = float(np.sum(p * np.nan_to_num(vals)) / s)
            j = int(np.argmax(dp[i]))
            arg_crash[i] = vals[j] if np.isfinite(vals[j]) else np.nan
    if n and not np.isfinite(exp_crash).any():
        print("  WARNING metrics: crash exposure is all-NaN "
              f"(crash_map={len(crash_map)}, zones={np.asarray(zones)[:3]}, "
              f"otaz={np.asarray(otaz)[:3]})")
    # ---- distance ----
    obs_dist = np.array([dist_map.get((int(o), int(d)), np.nan)
                         for o, d in zip(otaz, obs_dest)], float)
    exp_dist = np.full(n, np.nan)
    for i, o in enumerate(otaz):
        vals = np.array([dist_map.get((int(o), int(z)), np.nan) for z in zones], float)
        ok = np.isfinite(vals)
        if ok.any():
            p = np.where(ok, dp[i], 0.0)
            s = p.sum()
            if s > 0:
                exp_dist[i] = float(np.sum(p * np.nan_to_num(vals)) / s)

    def _mean(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if len(a) else float("nan")

    def _p90(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 90)) if len(a) else float("nan")

    def _ks(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            return float("nan"), float("nan")
        return float(ks_2samp(a, b).statistic), float(wasserstein_distance(a, b))

    dks, dwass = _ks(obs_dist, exp_dist)
    # McFadden rho2 against the observed-share null (the standard definition),
    # not against a uniform distribution.  The uniform versions remain in the
    # output as diagnostics only.
    _mode_ll0 = float(np.sum(obs_share * np.log(np.clip(obs_share, 1e-12, 1.0))))
    _mode_rho2 = (1.0 - mode_logloss / (-_mode_ll0)
                  if _mode_ll0 < 0 else float("nan"))
    _dest_p = obs_counts.values if hasattr(obs_counts, "values") else np.asarray(obs_counts)
    _dest_p = np.asarray(_dest_p, float)
    _dest_p = _dest_p[_dest_p > 0]
    _dest_ll0 = float(np.sum(_dest_p * np.log(np.clip(_dest_p, 1e-12, 1.0))))
    _dest_rho2 = (1.0 - dest_logloss / (-_dest_ll0)
                  if _dest_ll0 < 0 else float("nan"))
    return {
        "n": n,
        "mode_logloss": mode_logloss,
        "mode_rho2": _mode_rho2,
        "mode_brier": mode_brier,
        "mode_accuracy": mode_acc,
        "mode_mae_pp": mode_mae_pp,
        "active_obs": active_obs,
        "active_pred": active_pred,
        "dest_logloss": dest_logloss,
        "dest_rho2": _dest_rho2,
        "dest_mae": dest_mae,
        "dest_js": dest_js,
        "dest_top25": dest_top25,
        "dest_r_log": r_log,
        "dest_rho": rho,
        "crash_obs_mean": _mean(obs_crash),
        "crash_obs_p90": _p90(obs_crash),
        "crash_exp_mean": _mean(exp_crash),
        "crash_exp_p90": _p90(exp_crash),
        "crash_argmax_mean": _mean(arg_crash),
        "dist_obs_mean": _mean(obs_dist),
        "dist_exp_mean": _mean(exp_dist),
        "dist_ks": dks,
        "dist_wass": dwass,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="M_S run directory (contains stage5_models.pkl)")
    ap.add_argument("--m0-run-dir", default="",
                    help="optional M_0 refit run directory; if omitted, use a "
                         "crash-coefficient ablation as a diagnostic")
    ap.add_argument("--out", default="runs/safety_validation")
    ap.add_argument("--split", default="random", choices=["random", "temporal"])
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skims", default=os.environ.get("PIPELINE_SKIMS_PATH", "../demand_/skims_371.omx"))
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="person-level bootstrap replications for key metrics (0=off)")
    ap.add_argument("--segments", default="work,edu,nonmand")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    segs = [s.strip() for s in a.segments.split(",") if s.strip()]

    tours = pd.read_pickle(os.path.join(a.run_dir, "_hhts_tours.pkl"))
    persons = pd.read_pickle(os.path.join(a.run_dir, "_hhts_persons.pkl"))
    tours = _prep_hhts_tours(tours, persons)
    models = load_models(os.path.join(a.run_dir, "stage5_models.pkl"))
    # Optional rate-scale mode crash coefficients. The base mode model is on
    # crash counts; when the validation scores a crash-rate surface the mode
    # coefficient must be re-expressed on that scale or the mode shares move
    # for a units reason. JSON: {"work": -0.83, ...}.
    _mcj = os.environ.get("SAFETY_MODE_CRASH_JSON", "").strip()
    if _mcj and os.path.exists(_mcj):
        with open(_mcj, "r", encoding="utf-8") as _fh:
            _betas = json.load(_fh)
        for _seg, _b in _betas.items():
            if _seg in models:
                _old = models[_seg].get("mode_coef", {}).get("c_crash_veh")
                models[_seg].setdefault("mode_coef", {})["c_crash_veh"] = float(_b)
                print(f"mode c_crash_veh [{_seg}]: {_old} -> {_b}")
    # Optional rebuilt destination models (e.g. refit on a crash RATE):
    # swap them in over the baseline so the same holdout scoring applies.
    _dm_path = os.environ.get("SAFETY_DEST_MODELS", "").strip()
    _swap_mode = os.environ.get("SAFETY_SWAP_MODE", "0").strip() == "1"
    if _dm_path and os.path.exists(_dm_path):
        import cloudpickle as _cp
        with open(_dm_path, "rb") as _fh:
            _new = _cp.load(_fh)
        for _seg, _obj in _new.items():
            if _seg in models:
                models[_seg]["dest_model"] = _obj["dest_model"]
                models[_seg]["dest_coef"] = dict(_obj.get("dest_coef") or {})
                if _swap_mode and _obj.get("mode_model") is not None:
                    models[_seg]["mode_model"] = _obj["mode_model"]
                    models[_seg]["mode_coef"] = dict(
                        _obj.get("mode_coef") or {})
                    print(f"mode + dest model for {_seg}: replaced from {_dm_path}")
                else:
                    print(f"dest model for {_seg}: replaced from {_dm_path}")
    # Optional coefficient override for calibration diagnostics: replace the
    # estimated destination crash coefficient with a fixed value so the
    # holdout exposure can be traced as a function of it. Estimation is not
    # re-run; this answers "which response is consistent with the holdout".
    _beta_ovr = os.environ.get("SAFETY_DEST_CRASH_BETA", "").strip()
    if _beta_ovr:
        for _seg in models:
            if "dest_coef" in models[_seg]:
                models[_seg]["dest_coef"] = dict(models[_seg]["dest_coef"])
                models[_seg]["dest_coef"]["c_crash"] = float(_beta_ovr)
        print(f"dest c_crash overridden to {float(_beta_ovr):.3f} "
              f"(calibration diagnostic)")
    # Optional proper M_0 destination models (crash terms removed and refit).
    _dm0_path = os.environ.get("SAFETY_DEST_MODELS_M0", "").strip()
    _m0_dest_only = False
    if _dm0_path and os.path.exists(_dm0_path):
        import cloudpickle as _cp0
        # Keep the baseline mode models; only the destination model changes.
        models0 = {_seg: dict(_obj) for _seg, _obj in models.items()}
        with open(_dm0_path, "rb") as _fh0:
            _new0 = _cp0.load(_fh0)
        for _seg, _obj in _new0.items():
            if _seg in models0:
                models0[_seg]["dest_model"] = _obj["dest_model"]
                models0[_seg]["dest_coef"] = dict(_obj.get("dest_coef") or {})
                if _swap_mode and _obj.get("mode_model") is not None:
                    models0[_seg]["mode_model"] = _obj["mode_model"]
                    models0[_seg]["mode_coef"] = dict(
                        _obj.get("mode_coef") or {})
                    print(f"M_0 mode + dest model for {_seg}: replaced from {_dm0_path}")
                else:
                    print(f"M_0 dest model for {_seg}: replaced from {_dm0_path}")
        _m0_dest_only = True
    else:
        models0 = (load_models(os.path.join(a.m0_run_dir, "stage5_models.pkl"))
                   if a.m0_run_dir else None)
    m0_arm = ("M_0_refit_dest" if _m0_dest_only
              else ("M_0_refit" if models0 is not None else "M_0_ablation"))

    skims, close_sk = _open_skims(a.skims)
    skim_df = _skim_flat(skims)
    # Repair the nine car-less zones before anything reads DIST: leaving them
    # at zero makes them the cheapest alternatives in a log(1+DIST) utility.
    skim_df = repair_missing_dist(skim_df)
    skim_df, crash_diag = _ensure_crash_in_skim(skim_df)
    skim_df = apply_crash_measures(skim_df)
    # Optional alternative crash measure (e.g. SAFETY_CRASH_MATRIX=crash_rate_total).
    # c_crash reads log(CRASH+1); a rate keeps exposure separate from distance,
    # which a raw OD count cannot do (corr(log DIST, log count) = +0.18).
    _cm = os.environ.get('SAFETY_CRASH_MATRIX', '').strip()
    if _cm:
        from stage5_destination_choice import _read_crash_od_table
        os.environ['STAGE5_CRASH_MATRIX'] = _cm
        _omx = os.environ.get('SAFETY_CRASH_OMX',
                             os.path.join('..', 'demand_', 'crash_matrices_371.omx'))
        _tbl = _read_crash_od_table(_omx)
        skim_df = skim_df.drop(columns=['CRASH'], errors='ignore').merge(
            _tbl, on=['OTAZ', 'DTAZ'], how='left')
        skim_df['CRASH'] = pd.to_numeric(skim_df['CRASH'], errors='coerce').fillna(0.0)
        print(f"crash measure: {_cm} (mean {skim_df.CRASH.mean():.2f}, "
              f"max {skim_df.CRASH.max():.1f})")
    tours = _bridge_tours_zones_auto(tours, skims, skim_df, "validate")
    try:
        from data_connect import Synth_Connect
        landuse = Synth_Connect().get_land()
    except Exception as e:
        raise RuntimeError(f"landuse unavailable: {e}")

    ho_pids, split_name = _holdout_tours(tours, persons, a.frac, a.seed, a.split)
    ho_mask = tours["person_id"].astype(str).isin(ho_pids)
    ho_tours = tours[ho_mask].copy()
    ho_persons = persons[persons["person_id"].astype(str).isin(ho_pids)].copy()
    print(f"holdout: {split_name} persons={len(ho_pids):,} tours={len(ho_tours):,}")

    # Population-propagation check: the per-person safety sensitivity that
    # drives c_crash_safety[_dest] must vary across people, not sit at 0.5.
    try:
        from stage5_destination_choice import _safety_sensitivity_score
        _ss = _safety_sensitivity_score(ho_persons)
        _desc = _ss.describe(percentiles=[.05, .25, .5, .75, .95])
        _desc["share_at_0.5"] = float((_ss.sub(0.5).abs() < 1e-9).mean())
        _desc.to_csv(os.path.join(a.out, "safety_sensitivity_distribution.csv"))
        print("safety_sensitivity:", {k: round(float(v), 3) for k, v in _desc.items()})
    except Exception as e:
        print(f"safety_sensitivity diagnostic skipped: {type(e).__name__}: {e}")

    crash_map = _crash_matrix(skim_df)
    dist_map = _dist_matrix(skim_df)
    rows = []
    act_rows = []
    eq_rows = []
    cache = {}
    _dump_equity = os.environ.get("SAFETY_DUMP_EQUITY", "").strip() == "1"
    for seg in segs:
        seg_all = _filter_seg(ho_tours, seg)
        if seg_all.empty:
            continue
        seg_pids = set(seg_all["person_id"].astype(str))
        seg_persons = ho_persons[ho_persons["person_id"].astype(str).isin(seg_pids)]
        try:
            pred_s = _predict_segment(seg, seg_all, seg_persons, models,
                                      landuse, skims, skim_df,
                                      zero_crash=False, crash_cols={})
            if models0 is not None:
                # A destination-only M_0 refit keeps the baseline mode model;
                # zero its crash coefficient too so the M_0 arm is crash-free.
                _zero_mode = bool(
                    _m0_dest_only
                    and os.environ.get("SAFETY_M0_ZERO_MODE", "1").strip() != "0")
                pred_0 = _predict_segment(seg, seg_all, seg_persons, models0,
                                          landuse, skims, skim_df,
                                          zero_crash=_zero_mode, crash_cols={})
            else:
                pred_0 = _predict_segment(seg, seg_all, seg_persons, models,
                                          landuse, skims, skim_df,
                                          zero_crash=True, crash_cols={})
            ms = _metrics(pred_s, crash_map, dist_map)
            m0 = _metrics(pred_0, crash_map, dist_map)
            _ps = np.asarray(pred_s["mode_probs"], dtype=float)
            _p0 = np.asarray(pred_0["mode_probs"], dtype=float)
            if _ps.shape == _p0.shape:
                dp = float(np.abs(_ps - _p0).max())
                share_l1 = float(np.abs(_ps.mean(axis=0) - _p0.mean(axis=0)).sum())
            else:
                dp = float("nan")
                share_l1 = float("nan")
            for _mm in (ms, m0):
                _mm["mode_ablation_maxdp"] = dp
                _mm["mode_share_l1"] = share_l1
                _mm["mode_coefs_applied"] = pred_s.get("mode_coefs_applied")
                _mm["mode_crash_coef_present"] = pred_s.get("mode_crash_coef_present")
                _mm["mode_crash_param_present"] = pred_s.get("mode_crash_param_present")
                _mm["mode_vars_passed"] = pred_s.get("mode_vars_passed")
            print("  mode ablation: max|dp|=" + format(dp, ".3e")
                  + " shareL1=" + format(share_l1, ".3e")
                  + " crash_coef_present=" + str(pred_s.get("mode_crash_coef_present"))
                  + " crash_param_in_tree=" + str(pred_s.get("mode_crash_param_present")))
            if not np.isfinite(dp) or dp < 1e-9:
                print("  WARNING [" + seg + "]: mode probabilities identical between "
                      "arms (max|dp|=" + format(dp, ".2e") + ") - the mode-side "
                      "crash coefficient was NOT exercised")
            cache[seg] = (pred_s, pred_0)
            if _dump_equity:
                eq_rows.append(pd.DataFrame({
                    "segment": seg,
                    "person_id": pred_s["person_id"],
                    "otaz": pred_s["otaz"],
                    "dtaz": pred_s["dtaz"],
                    "exposure_ms": _expected_exposure(pred_s, crash_map),
                    "exposure_m0": _expected_exposure(pred_0, crash_map),
                }))
            # ---- predicted vs actual ACTIVITY counts (not just fit stats) ----
            # A fit statistic says the likelihood moved; this says whether the
            # model puts the right NUMBER of activities in the right zones.
            try:
                _zones = np.asarray(pred_s["zones"])
                _obs_counts = pd.Series(pred_s["dtaz"]).value_counts()
                _tab = pd.DataFrame({"zone": _zones})
                _tab["obs"] = _tab["zone"].map(_obs_counts).fillna(0.0)
                _tab["pred_M_S"] = np.asarray(pred_s["dest_probs"]).sum(axis=0)
                _tab["pred_M_0"] = np.asarray(pred_0["dest_probs"]).sum(axis=0)
                _tab.insert(0, "segment", seg)
                _tab.to_csv(os.path.join(a.out, "activity_counts_" + seg + ".csv"),
                            index=False)

                def _cpc(exp_df, obs_df):
                    idx = exp_df.index.union(obs_df.index)
                    cidx = exp_df.columns.union(obs_df.columns)
                    e = exp_df.reindex(index=idx, columns=cidx).fillna(0.0).values
                    o = obs_df.reindex(index=idx, columns=cidx).fillna(0.0).values
                    return float(np.minimum(e, o).sum() / max(o.sum(), 1e-12))

                _od_s = pd.DataFrame(np.asarray(pred_s["dest_probs"]),
                                     columns=_zones).groupby(
                                         np.asarray(pred_s["otaz"])).sum()
                _od_0 = pd.DataFrame(np.asarray(pred_0["dest_probs"]),
                                     columns=np.asarray(pred_0["zones"])).groupby(
                                         np.asarray(pred_0["otaz"])).sum()
                _od_obs = pd.crosstab(np.asarray(pred_s["otaz"]),
                                      np.asarray(pred_s["dtaz"]))
                for _arm, _col, _od in (("M_S", "pred_M_S", _od_s),
                                        (m0_arm, "pred_M_0", _od_0)):
                    _p = _tab[_col].values
                    _o = _tab["obs"].values
                    _ss_res = float(np.sum((_p - _o) ** 2))
                    _ss_tot = float(np.sum((_o - _o.mean()) ** 2))
                    act_rows.append({
                        "segment": seg, "arm": _arm, "n_tours": int(np.sum(_o)),
                        "obs_activities": float(_o.sum()),
                        "pred_activities": float(_p.sum()),
                        "zones_with_activity": int((_o > 0).sum()),
                        "zone_r": float(np.corrcoef(_p, _o)[0, 1]),
                        "zone_r2": 1.0 - _ss_res / max(_ss_tot, 1e-12),
                        "zone_rmse": float(np.sqrt(_ss_res / max(len(_p), 1))),
                        "zone_mae": float(np.mean(np.abs(_p - _o))),
                        "cpc": _cpc(_od, _od_obs),
                        "top10_share_obs": float(
                            np.sort(_o)[-10:].sum() / max(_o.sum(), 1e-12)),
                        "top10_share_pred": float(
                            np.sort(_p)[-10:].sum() / max(_p.sum(), 1e-12)),
                    })
            except Exception as _e:
                print("  activity-count block failed:", type(_e).__name__, _e)
            for arm, mm in (("M_S", ms), (m0_arm, m0)):
                rows.append({"arm": arm, "segment": seg, **mm})
            print(f"{seg}: n={ms['n']} modeMAE={ms['mode_mae_pp']:.2f}/"
                  f"{m0['mode_mae_pp']:.2f} destJS={ms['dest_js']:.3f}/"
                  f"{m0['dest_js']:.3f} crashExp={ms['crash_exp_mean']:.3f}/"
                  f"{m0['crash_exp_mean']:.3f} obs={ms['crash_obs_mean']:.3f}")
        except Exception as e:
            print(f"{seg}: FAILED {type(e).__name__}: {e}")
            rows.append({"arm": "ERROR", "segment": seg, "error": str(e)[:300]})

    df = pd.DataFrame(rows)
    csv = os.path.join(a.out, "safety_validation_metrics.csv")
    df.to_csv(csv, index=False)
    with open(os.path.join(a.out, "safety_validation_metrics.json"), "w") as f:
        json.dump({"run_dir": a.run_dir, "split": split_name,
                   "frac": a.frac, "seed": a.seed,
                   "crash_skim_diag": crash_diag,
                   "rows": rows}, f, indent=2)
    print("wrote", csv)
    if eq_rows:
        _eq = pd.concat(eq_rows, ignore_index=True)
        _eqp = os.path.join(a.out, "equity_tours.csv")
        _eq.to_csv(_eqp, index=False)
        print("wrote", _eqp, f"({len(_eq):,} tours)")
    if act_rows:
        _adf = pd.DataFrame(act_rows)
        _ap = os.path.join(a.out, "activity_fit_summary.csv")
        _adf.to_csv(_ap, index=False)
        print("wrote", _ap)
        print(_adf.to_string(index=False))

    # ---- poster panel: observed vs M_0 vs M_S ----
    try:
        ok = df[df.arm.isin(["M_S", "M_0_ablation", "M_0_refit"])]
        m0_arm = "M_0_refit" if "M_0_refit" in set(df.arm) else "M_0_ablation"
        if not ok.empty:
            metrics = [
                ("dest_js", "Destination-share divergence\n(JS; 0 = observed)"),
                ("dest_top25", "Top-25 destination overlap\n(1 = observed)"),
                ("crash_exp_mean", "Chosen-destination crash exposure\n(mean; observed bar)"),
            ]
            segs_ok = [s for s in segs if s in set(ok.segment)]
            fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.8))
            if len(metrics) == 1:
                axes = [axes]
            for ax, (key, title) in zip(axes, metrics):
                x = np.arange(len(segs_ok))
                w = 0.26
                m0 = [ok[(ok.segment == s) & (ok.arm == m0_arm)].iloc[0][key]
                      if not ok[(ok.segment == s) & (ok.arm == m0_arm)].empty else np.nan
                      for s in segs_ok]
                ms = [ok[(ok.segment == s) & (ok.arm == "M_S")].iloc[0][key]
                      if not ok[(ok.segment == s) & (ok.arm == "M_S")].empty else np.nan
                      for s in segs_ok]
                if key == "crash_exp_mean":
                    obs = [ok[(ok.segment == s)].iloc[0]["crash_obs_mean"] for s in segs_ok]
                    ax.bar(x - w, obs, w, color=QUT["grey"], label="Observed HHTS")
                    ax.bar(x, m0, w, color=QUT["red"], alpha=0.85,
                           label=("M$_0$ (refit)" if m0_arm == "M_0_refit" else "M$_0$ (crash ablation)"))
                    ax.bar(x + w, ms, w, color=QUT["navy"], alpha=0.95, label="M$_S$")
                else:
                    ax.bar(x - w / 2, m0, w, color=QUT["red"], alpha=0.85,
                           label=("M$_0$ (refit)" if m0_arm == "M_0_refit" else "M$_0$ (crash ablation)"))
                    ax.bar(x + w / 2, ms, w, color=QUT["navy"], alpha=0.95, label="M$_S$")
                    target = 0.0 if key == "dest_js" else 1.0
                    ax.axhline(target, color=QUT["grey"], ls="--", lw=1.2,
                               label="Observed target")
                ax.set_xticks(x)
                ax.set_xticklabels([s.replace("nonmand", "non-mand.") for s in segs_ok])
                ax.set_title(title, fontsize=9.5, color=QUT["navy"], fontweight="bold")
                ax.grid(axis="y", alpha=0.3)
                ax.legend(fontsize=7.2, frameon=False)
            fig.suptitle("Does adding safety move the model closer to observed HHTS?",
                         fontsize=13, fontweight="bold", color=QUT["navy"])
            fig.tight_layout()
            panel = os.path.join(a.out, "safety_validation_panel.png")
            fig.savefig(panel, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print("wrote", panel)
    except Exception as e:
        print("panel failed:", e)

    if close_sk:
        try:
            skims.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
