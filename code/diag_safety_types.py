#!/usr/bin/env python
"""Does the HHTS reveal speed-focused / crash-focused / blended travellers?

The route-choice person types (fastest / balanced / safest shares per type)
are calibrated, not estimated: lambda is set from safety_sensitivity x income
tertiles. The HHTS has no route traces, so any revealed safety preference has
to show up in WHERE people go and HOW, given exposure. This script measures
that signal per person and asks whether it is a stable trait.

Per tour, at the chosen OD and relative to what distance predicts from that
origin (same [1, d, log1p d] regression used by safety_sensitivity):
  crash residual  r_c : log1p(crash)       - fitted   (<0: safer than expected)
  speed residual  r_t : log(car time)      - fitted   (<0: faster than expected)

Tests
  1. signal     : ICC(1) of r_c, r_t across persons vs a permutation null that
                  shuffles tours across persons; split-half reliability.
  2. trade-off  : tour-level corr(r_c, r_t). Speed vs safety is only a real
                  choice if fast ODs are the risky ones (corr < 0).
  3. clusters   : Gaussian mixture on reliability-shrunk person means
                  (+ mode shares), K by BIC; centroids labelled speed /
                  crash / blend.
  4. prediction : can demographics predict the cluster (5-fold CV vs majority
                  baseline) and the continuous crash / speed traits (CV R^2)?

Writes runs/safety_types_diag/{summary.json,persons.csv,report.txt}.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PIPELINE_SKIMS_PATH", "../demand_/skims_371_routechoice.omx")

OUT = os.path.join(HERE, "runs", "safety_types_diag")
# The HHTS is a one-day diary: 74% of persons have 1 tour, 252 have >= 4.
MIN_TOURS_SIGNAL = 2      # persons with >= this many tours enter the ICC
MIN_TOURS_CLUSTER = 2
N_PERM = 200
SEED = 7

# stage5 MODE_NAMES: 1 DA, 2 SR, 3 Walk, 4 Bike, 5 Transit
MODE_NAMES = {1: "DA", 2: "SR", 3: "WALK", 4: "BIKE", 5: "TRANSIT"}
CAR_MODES = {"DA", "SR"}
ACTIVE_MODES = {"WALK", "BIKE"}


def _origin_residual(df, ycol):
    """Residual of ycol on [1, DIST, log1p DIST], fitted separately per origin."""
    out = np.full(len(df), np.nan)
    d_all = df["DIST"].to_numpy(float)
    y_all = df[ycol].to_numpy(float)
    for _, idx in df.groupby("OTAZ").indices.items():
        idx = np.asarray(idx)
        d, y = d_all[idx], y_all[idx]
        ok = np.isfinite(d) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        X = np.column_stack([np.ones(ok.sum()), d[ok], np.log1p(d[ok])])
        b, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        out[idx[ok]] = y[ok] - X @ b
    return out


def _od_residuals(skim_df):
    tcol = next((c for c in ("SOVTOLL_TIME__AM", "SOV_TIME__AM", "SOVTOLL_TIME",
                             "SOV_TIME") if c in skim_df.columns), None)
    if tcol is None:
        raise SystemExit(f"no car time column in skim_df: {list(skim_df.columns)[:40]}")
    df = skim_df[["OTAZ", "DTAZ", "DIST", "CRASH", tcol]].copy()
    for c in ("OTAZ", "DTAZ", "DIST", "CRASH", tcol):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["y_c"] = np.log1p(df["CRASH"].clip(lower=0))
    df["y_t"] = np.log(df[tcol].where(df[tcol] > 0))
    df["r_c"] = _origin_residual(df, "y_c")
    df["r_t"] = _origin_residual(df, "y_t")
    print(f"OD residuals: time column {tcol}; "
          f"{np.isfinite(df.r_c).mean():.1%} crash, {np.isfinite(df.r_t).mean():.1%} time finite")
    return df[["OTAZ", "DTAZ", "DIST", "r_c", "r_t"]]


def _icc1(values, groups):
    """One-way random-effects ICC(1) with unequal group sizes."""
    s = pd.DataFrame({"v": values, "g": groups})
    k = s.groupby("g")["v"].size()
    n, a = len(s), len(k)
    if a < 3:
        return float("nan")
    gm = s["v"].mean()
    means = s.groupby("g")["v"].mean()
    ssb = float((k * (means - gm) ** 2).sum())
    ssw = float(((s["v"] - s["g"].map(means)) ** 2).sum())
    msb, msw = ssb / (a - 1), ssw / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def _signal_test(t, col, rng):
    t = t.dropna(subset=[col])
    keep = t.groupby("person_id")[col].transform("size") >= MIN_TOURS_SIGNAL
    t = t[keep]
    obs = _icc1(t[col].to_numpy(), t["person_id"].to_numpy())
    vals = t[col].to_numpy()
    grp = t["person_id"].to_numpy()
    # Null: shuffle residuals only among tours from the SAME origin zone, so a
    # person's home geography cannot masquerade as a preference trait.
    blocks = [np.asarray(ix) for ix in t.groupby("OTAZ").indices.values()]

    def _perm():
        v = vals.copy()
        for ix in blocks:
            if len(ix) > 1:
                v[ix] = vals[rng.permutation(ix)]
        return v
    null = np.array([_icc1(_perm(), grp) for _ in range(N_PERM)])
    # split-half: random half of each person's tours vs the other half
    halves = []
    for _, v in t.groupby("person_id")[col]:
        v = rng.permutation(v.to_numpy())
        h = len(v) // 2
        halves.append((v[:h].mean(), v[h:2 * h].mean()))
    h = np.array(halves)
    r = float(np.corrcoef(h[:, 0], h[:, 1])[0, 1]) if len(h) > 5 else float("nan")
    sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")
    return {
        "n_persons": int(t["person_id"].nunique()), "n_tours": int(len(t)),
        "icc": obs, "icc_null_mean": float(null.mean()),
        "icc_null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((np.sum(null >= obs) + 1) / (N_PERM + 1)),
        "split_half_r": r, "spearman_brown": sb,
    }


def _shrunk_means(t, col):
    """Empirical-Bayes shrinkage of person means toward the grand mean."""
    t = t.dropna(subset=[col])
    g = t.groupby("person_id")[col]
    m, n, v = g.mean(), g.size(), g.var(ddof=1)
    sw2 = float(np.nanmean(v[n > 1]))                  # within-person variance
    sb2 = max(float(m[n >= MIN_TOURS_SIGNAL].var()) - sw2 / float(n[n >= MIN_TOURS_SIGNAL].mean()), 1e-12)
    gm = float(t[col].mean())
    w = n / (n + sw2 / sb2)
    return gm + w * (m - gm), n, {"sigma_within2": sw2, "sigma_between2": sb2,
                                  "mean_weight": float(w.mean())}


DEMOG = ["AGEGROUP", "sexp", "incp", "vehrd", "HHVEH", "CARLICENCE", "MCLICENCE",
         "STUDYING", "WORKSTAT", "WFHDAY", "bikes", "EBIKES", "ESCOOTER",
         "hh_1person", "hh_couple", "hh_small_group", "hh_large_group",
         "ind_professional", "ind_financial", "ind_it", "ind_health",
         "ind_education", "ind_public_admin", "ind_construction",
         "ind_manufacturing", "ind_blue_collar", "ind_retail", "ind_hospitality",
         "ind_transport", "ind_arts_other", "ind_not_working"]


def _demographics(persons):
    """Person attributes the model could use to assign a type (no weights,
    dates, zone ids or diary-derived counts)."""
    cols = [c for c in DEMOG if c in persons.columns
            and pd.to_numeric(persons[c], errors="coerce").notna().mean() >= 0.9
            and pd.to_numeric(persons[c], errors="coerce").nunique() > 1]
    X = persons[cols].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median()), cols


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)
    import pipeline_logger as pl
    from refit_dest_crash_rate import rate_skim
    from stage5_destination_choice import (_open_skims, _skim_flat,
                                           _prep_hhts_tours, attach_person_type)
    from validate_safety_arms import _bridge_tours_zones_auto

    _sc, out, _landuse = pl.load_data()
    persons = attach_person_type(out["persons"].copy())
    tours = _prep_hhts_tours(out["tours"].copy(), persons)
    skims, close = _open_skims(os.environ["PIPELINE_SKIMS_PATH"])
    try:
        skim_df = rate_skim(_skim_flat(skims),
                            os.environ.get("STAGE5_CRASH_MATRIX", "crash_rate_total"))
        tours = _bridge_tours_zones_auto(tours, skims, skim_df, "safety_types")
    finally:
        try:
            close()
        except Exception:
            pass
    od = _od_residuals(skim_df)

    mode_col = next((c for c in ("tour_mode", "mode", "TOUR_MODE", "main_mode")
                     if c in tours.columns), None)
    t = tours[["person_id", "OTAZ", "DTAZ"] + ([mode_col] if mode_col else [])].copy()
    t["person_id"] = t["person_id"].astype(str)
    for c in ("OTAZ", "DTAZ"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.merge(od, on=["OTAZ", "DTAZ"], how="left")
    mode = (pd.to_numeric(t[mode_col], errors="coerce").map(MODE_NAMES).fillna("UNKNOWN")
            if mode_col else pd.Series("UNKNOWN", index=t.index))
    t["is_car"] = mode.isin(CAR_MODES)
    t["is_active"] = mode.isin(ACTIVE_MODES)
    t["is_transit"] = mode.eq("TRANSIT")
    print(f"tours {len(t):,} | persons {t.person_id.nunique():,} | mode col {mode_col} "
          f"| car {t.is_car.mean():.1%} active {t.is_active.mean():.1%} "
          f"transit {t.is_transit.mean():.1%} | OD matched {t.r_c.notna().mean():.1%}")
    car = t[t["is_car"]]

    summary = {"n_tours": int(len(t)), "n_persons": int(t.person_id.nunique()),
               "mode_col": mode_col}

    # 1. signal
    summary["signal"] = {
        "crash_car": _signal_test(car, "r_c", rng),
        "speed_car": _signal_test(car, "r_t", rng),
        "crash_all_modes": _signal_test(t, "r_c", rng),
    }
    # 2. trade-off
    ok = car[["r_c", "r_t"]].dropna()
    od_ok = od[["r_c", "r_t"]].dropna()
    summary["tradeoff"] = {
        "corr_chosen_car_tours": float(ok.corr().iloc[0, 1]),
        "corr_all_od_pairs": float(od_ok.corr().iloc[0, 1]),
    }

    # 3. clusters on shrunk person traits + mode shares
    crash_m, n_c, sh_c = _shrunk_means(car, "r_c")
    speed_m, _n_t, sh_t = _shrunk_means(car, "r_t")
    summary["shrinkage"] = {"crash": sh_c, "speed": sh_t}
    shares = t.groupby("person_id")[["is_car", "is_active", "is_transit"]].mean()
    P = pd.DataFrame({"crash_trait": crash_m, "speed_trait": speed_m,
                      "n_car_tours": n_c}).join(shares, how="left")
    P = P[P["n_car_tours"] >= MIN_TOURS_CLUSTER].dropna()
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    # Continuous revealed traits only: the 0/0.5/1 mode shares made the mixture
    # collapse onto spikes (BIC kept falling to K=5 on degenerate components).
    feats = ["crash_trait", "speed_trait"]
    Z = StandardScaler().fit_transform(P[feats])
    bics = {}
    models = {}
    for k in range(1, 6):
        gm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(Z)
        bics[k] = float(gm.bic(Z))
        models[k] = gm
    k_best = min(bics, key=bics.get)
    gm = models[k_best]
    P["cluster"] = gm.predict(Z)
    P["cluster_conf"] = gm.predict_proba(Z).max(axis=1)
    prof = P.groupby("cluster")[feats + ["n_car_tours"]].mean()
    prof["share"] = P["cluster"].value_counts(normalize=True).sort_index()
    sd_c, sd_t = P["crash_trait"].std(), P["speed_trait"].std()

    def _label(row):
        c = row["crash_trait"] / sd_c
        s = row["speed_trait"] / sd_t
        if c < -0.5 and s > -0.25:
            return "crash-focused"
        if s < -0.5 and c > -0.25:
            return "speed-focused"
        if c < -0.5 and s < -0.5:
            return "safe-and-fast"
        return "blend"
    prof["label"] = prof.apply(_label, axis=1)
    summary["clusters"] = {"bic": bics, "k_best": int(k_best),
                           "mean_assignment_confidence": float(P["cluster_conf"].mean()),
                           "profile": prof.round(4).reset_index().to_dict(orient="records")}

    # 4. prediction from demographics
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    pp = persons.copy()
    pp["person_id"] = pp["person_id"].astype(str)
    pp = pp.drop_duplicates("person_id").set_index("person_id")
    X_all, demo_cols = _demographics(pp)
    Xp = X_all.reindex(P.index)
    pred = {"demographic_columns": demo_cols}
    if k_best > 1 and P["cluster"].value_counts().min() >= 5:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        acc = cross_val_score(clf, Xp, P["cluster"], cv=cv, scoring="accuracy")
        f1 = cross_val_score(clf, Xp, P["cluster"], cv=cv, scoring="f1_macro")
        pred["cluster_cv_accuracy"] = float(acc.mean())
        pred["cluster_cv_macro_f1"] = float(f1.mean())
        pred["majority_baseline"] = float(P["cluster"].value_counts(normalize=True).max())
    prof_cols = [c for c in ("AGEGROUP", "sexp", "incp", "vehrd", "STUDYING",
                             "WORKSTAT", "hh_large_group") if c in Xp.columns]
    pred["cluster_demographic_profile"] = (
        Xp[prof_cols].groupby(P["cluster"]).mean().round(3)
        .reset_index().to_dict(orient="records"))
    for trait in ("crash_trait", "speed_trait"):
        r2 = cross_val_score(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))),
                             Xp, P[trait], cv=KFold(5, shuffle=True, random_state=SEED),
                             scoring="r2")
        pred[f"{trait}_cv_r2"] = float(r2.mean())
    # does the a-priori route person type line up with the revealed trait?
    if "person_type" in pp.columns:
        pt = pp["person_type"].reindex(P.index)
        pred["crash_trait_by_person_type"] = (
            P.groupby(pt)["crash_trait"].mean().round(4).to_dict())
    summary["prediction"] = pred

    P.join(pp.reindex(P.index)[[c for c in ("person_type", "safety_sensitivity")
                                if c in pp.columns]]).to_csv(os.path.join(OUT, "persons.csv"))
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
