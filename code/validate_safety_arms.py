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
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, ks_2samp, wasserstein_distance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage5_destination_choice import (
    load_models, _prep_hhts_tours, _filter_seg, _mode_tree,
    _build_mode_model, _dest_tree, _build_dest_model, _logsums,
    _attach_size_term, _skim_flat, _open_skims, _bridge_tours_zones_auto,
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


def _apply_coefs(mdl, coefs: dict) -> int:
    n = 0
    for k, v in (coefs or {}).items():
        try:
            mdl.set_value(str(k), float(v))
            n += 1
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
def _predict_segment(seg, seg_df, seg_persons, models, landuse, skims,
                     skim_df, zero_crash: bool, crash_cols: dict):
    res = models.get(seg) or {}
    mode_fit = res.get("mode_model")
    dest_fit = res.get("dest_model")
    if mode_fit is None or dest_fit is None:
        raise RuntimeError(f"{seg}: fitted mode/dest model missing")

    mv, mc = _free_vars_and_coefs(mode_fit)
    if not mv or not mc:
        mc = dict(res.get("mode_coef") or {})
        mv = [k for k in mc
              if not str(k).startswith(("ASC_", "Mu_"))
              and str(k) not in ("c_size_term", "log_correction")]
    if zero_crash:
        mc = _without_crash(mc)
    dt_m = _mode_tree(seg_df, seg_persons, landuse, skims, skim_df=skim_df)
    mm = _build_mode_model(dt_m, seg, custom_mode_vars=mv,
                           avail_vars_hint=list(seg_df.columns))
    _apply_coefs(mm, mc)
    mp = np.asarray(mm.probability(), dtype=float)
    mp = np.nan_to_num(mp, nan=0.0, posinf=0.0, neginf=0.0)
    rs = mp.sum(axis=1, keepdims=True)
    mp = mp / np.where(rs > 0, rs, 1.0)

    ls = _logsums(mm, dt_m, skims=skims, verbose=False)

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

    dt_d = _dest_tree(seg_df, seg_persons, landuse, skims, ls,
                      sampled_alts=None, skim_df=skim_df)
    dv, dc = _free_vars_and_coefs(dest_fit)
    if not dv or not dc:
        dc = dict(res.get("dest_coef") or {})
        dv = [k for k in dc
              if not str(k).startswith(("ASC_", "Mu_"))
              and str(k) not in ("c_size_term", "log_correction")]
    if zero_crash:
        dc = _without_crash(dc)
    dm = _build_dest_model(dt_d, seg, custom_dest_vars=dv)
    _apply_coefs(dm, dc)
    dp = np.asarray(dm.probability(), dtype=float)
    dp = np.nan_to_num(dp, nan=0.0, posinf=0.0, neginf=0.0)
    rs = dp.sum(axis=1, keepdims=True)
    dp = dp / np.where(rs > 0, rs, 1.0)

    dest_ids = _case_ids(dt_d)
    if dest_ids and len(dest_ids) != len(seg_df):
        mp_df = pd.DataFrame(mp, index=seg_df["ptour_id"].astype(str))
        seg_df = seg_df.set_index(seg_df["ptour_id"].astype(str)).loc[dest_ids].reset_index(drop=True)
        mp = mp_df.reindex(dest_ids).fillna(0.0).to_numpy()

    try:
        zones = [int(z) for z in list(dm.alternatives)]
    except Exception:
        zones = [int(z) for z in list(ls.coords["zones"].values)]
    if len(zones) != dp.shape[1]:
        zones = list(range(dp.shape[1]))

    return {
        "mode_probs": mp,
        "dest_probs": dp,
        "zones": np.asarray(zones, dtype=int),
        "tour_mode": pd.to_numeric(seg_df["tour_mode"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "otaz": pd.to_numeric(seg_df["OTAZ"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "dtaz": pd.to_numeric(seg_df["DTAZ"], errors="coerce").fillna(0).astype(int).to_numpy(),
    }


def _crash_matrix(skim_df):
    if skim_df is None or "CRASH" not in skim_df.columns:
        return {}
    d = skim_df[["OTAZ", "DTAZ", "CRASH"]].dropna()
    return {(int(o), int(dz)): float(c)
            for o, dz, c in d.itertuples(index=False, name=None)}


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
    return {
        "n": n,
        "mode_logloss": mode_logloss,
        "mode_brier": mode_brier,
        "mode_accuracy": mode_acc,
        "mode_mae_pp": mode_mae_pp,
        "active_obs": active_obs,
        "active_pred": active_pred,
        "dest_logloss": dest_logloss,
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
    models0 = (load_models(os.path.join(a.m0_run_dir, "stage5_models.pkl"))
               if a.m0_run_dir else None)
    m0_arm = "M_0_refit" if models0 is not None else "M_0_ablation"

    skims, close_sk = _open_skims(a.skims)
    skim_df = _skim_flat(skims)
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
    cache = {}
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
                pred_0 = _predict_segment(seg, seg_all, seg_persons, models0,
                                          landuse, skims, skim_df,
                                          zero_crash=False, crash_cols={})
            else:
                pred_0 = _predict_segment(seg, seg_all, seg_persons, models,
                                          landuse, skims, skim_df,
                                          zero_crash=True, crash_cols={})
            ms = _metrics(pred_s, crash_map, dist_map)
            m0 = _metrics(pred_0, crash_map, dist_map)
            cache[seg] = (pred_s, pred_0)
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
                   "rows": rows}, f, indent=2)
    print("wrote", csv)

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
