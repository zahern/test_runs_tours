"""
safety_plan_impact.py — paper-ready safety impact on travel plans + coverages.

Compares the safety-integrated run (M_S, crash skims ON) against the
no-safety run (M_0, crash skims OFF) and quantifies what the safety
component actually does to simulated travel, in numbers and figures
suitable for the paper.

Inputs (per arm): stage6_output.csv (preferred) or trips_for_aimsun.csv,
    plus stage5_models_coefficients.csv for coefficient annotation.
    Auto-detects the newest safety_baseline__* / safety_nosafety__* dirs
    under PIPELINE_RUN_ROOT (default: runs), with CLI override.

Outputs (all paper-ready, no HTML):
    tables/01_mode_share_by_arm.csv        mode shares per arm + delta (overall + by purpose)
    tables/02_demand_aggregates.csv        trips, VKT, active-km, mean dist/time, trips/person
    tables/03_active_by_distance.csv       active share by distance band per arm + delta
    tables/04_population_coverage.csv      trips / active share by ptype per arm + delta
    tables/05_spatial_coverage.csv         per-origin-zone active share per arm + delta
    tables/06_dta_coverage.csv             motorised trips by time period per arm (+in/out network if crosswalk)
    tables/07_crash_exposure.csv           crash-exposure distribution per arm, overall + by mode
    tables/08_tour_stats.csv               tours/person, trips/tour, activity participation per arm
    tables/headline_numbers.csv            one-row-per-metric headline deltas for the paper text
    figures/fig01_mode_shift.png           grouped mode shares + delta panel
    figures/fig02_demand_deltas.png        headline demand deltas (VKT, active-km, time)
    figures/fig03_active_by_distance.png   active share curves by distance band per purpose
    figures/fig04_population.png           delta active pp by ptype (population coverage)
    figures/fig05_spatial_topzones.png     top zones most/least affected (spatial coverage)
    figures/fig06_crash_exposure.png       crash-exposure histogram + high-crash share by mode
    figures/fig07_dta_coverage.png         motorised trips by time period per arm (DTA coverage)
    figures/fig08_destination_shift.png    destination-zone count scatter + mean-dist shift
    tables_for_paper.tex                   booktabs LaTeX snippet (mode shares, aggregates, population)

Single-arm mode: if only the baseline exists (nosafety not run yet), all
baseline tables/figures are still produced; delta columns are left empty
and figures carry a "M_0 not run yet" note instead of failing.

Usage:
    python safety_plan_impact.py [--baseline DIR] [--nosafety DIR]
        [--out runs/safety_plan_impact] [--crosswalk path/to/crosswalk.csv]
        [--chunksize 500000]

Column contract (both schemas supported):
    stage6_output.csv : person_id,row_type,purpose,OTAZ,DTAZ,mode,mode_name,
        dist,travtime,crash,ptype,time_period,generalised_cost
    trips_for_aimsun.csv : person_id,trip_id,OTAZ,DTAZ,mode,mode_name,purpose,
        time_period,travtime,dist,crash,generalised_cost (+departure_time)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

MODE_ORDER = ["DA", "SR", "Walk", "Bike", "Transit"]
MODE_COLORS = {
    "DA": "#C0392B",
    "SR": "#E67E22",
    "Walk": "#27AE60",
    "Bike": "#2980B9",
    "Transit": "#8E44AD",
}
MODE_TO_INT = {"DA": 1, "SR": 2, "Walk": 3, "Bike": 4, "Transit": 5}
INT_TO_MODE = {v: k for k, v in MODE_TO_INT.items()}
MOTORISED = {"DA", "SR", "Transit"}
ACTIVE = {"Walk", "Bike"}
PURPOSES = ["work", "edu", "social", "escort", "shopping", "othmaint", "othdiscr", "eat"]
DIST_BINS = [0, 2, 5, 10, 20, 50, 999]
DIST_LABELS = ["<2km", "2-5km", "5-10km", "10-20km", "20-50km", ">50km"]
TIME_ORDER = ["early", "am_peak", "midday", "pm_peak", "evening", "late"]

USECOLS = [
    "person_id", "row_type", "purpose", "OTAZ", "DTAZ", "mode", "mode_name",
    "dist", "travtime", "crash", "ptype", "time_period", "generalised_cost",
    "departure_time", "STARTIME",
]

PAPER_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


# ---------------------------------------------------------------------------
# run discovery + loading
# ---------------------------------------------------------------------------

def _latest_run_dir(prefix: str, base: str) -> str | None:
    cands = sorted(
        (p for p in glob.glob(os.path.join(base, prefix + "__*")) if os.path.isdir(p)),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if cands:
        return cands[0]
    legacy = os.path.join(base, prefix)
    return legacy if os.path.isdir(legacy) else None


def _find_trips_file(run_dir: str) -> str | None:
    for fn in ("stage6_output.csv", "trips_for_aimsun.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.isfile(p) and os.path.getsize(p) > 1000:
            return p
    return None


def _norm_mode(df: pd.DataFrame) -> pd.Series:
    """Canonical mode-name series (DA/SR/Walk/Bike/Transit)."""
    if "mode_name" in df.columns:
        s = df["mode_name"].astype(str).str.strip()
        # normalise legacy 'DA' variants
        s = s.replace({"Drive Alone": "DA", "Shared Ride": "SR", "drive_alone": "DA"})
        s = s.where(s.isin(MODE_ORDER), np.nan)
        if s.notna().mean() > 0.5:
            return s
    if "mode" in df.columns:
        m = pd.to_numeric(df["mode"], errors="coerce")
        return m.map(INT_TO_MODE)
    return pd.Series(np.nan, index=df.index)


def load_trips(run_dir: str, chunksize: int) -> pd.DataFrame:
    """Memory-safe trip-table load: keeps trip rows + canonical columns only."""
    src = _find_trips_file(run_dir)
    if src is None:
        return pd.DataFrame()
    header = pd.read_csv(src, nrows=0)
    usecols = [c for c in USECOLS if c in header.columns]
    parts = []
    for chunk in pd.read_csv(src, usecols=usecols, chunksize=chunksize,
                             low_memory=True, on_bad_lines="skip"):
        if "row_type" in chunk.columns:
            chunk = chunk[chunk["row_type"].astype(str).str.lower() == "trip"]
        else:
            # trips_for_aimsun.csv is trips-only (no row_type)
            pass
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["mode_name"] = _norm_mode(chunk)
        chunk = chunk[chunk["mode_name"].isin(MODE_ORDER)]
        for c in ("dist", "travtime", "crash", "OTAZ", "DTAZ"):
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        if "purpose" in chunk.columns:
            chunk["purpose"] = chunk["purpose"].astype(str).str.strip().str.lower()
        if "ptype" not in chunk.columns:
            chunk["ptype"] = "Unknown"
        else:
            chunk["ptype"] = chunk["ptype"].astype(str).str.strip().replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})
        if "time_period" not in chunk.columns:
            chunk["time_period"] = "unknown"
        parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["is_motor"] = out["mode_name"].isin(MOTORISED)
    out["is_active"] = out["mode_name"].isin(ACTIVE)
    return out


def load_crash_coefs(run_dir: str) -> pd.DataFrame:
    for fn in ("stage5_models_coefficients.csv", "stage5_tour_coefficients.csv"):
        p = os.path.join(run_dir, fn)
        if os.path.isfile(p):
            try:
                df = pd.read_csv(p)
                idc = next(
                    (c for c in df.columns
                     if c.lower() in ("parameter", "coef_name", "coefficient_name", "name", "variable")),
                    df.columns[0],
                )
                mask = df[idc].astype(str).str.lower().str.contains("crash", na=False)
                return df.loc[mask].copy()
            except Exception:
                return pd.DataFrame()
    # merged-shard layout: runs/.../stage5/shards/<seg>/stage5_models_coefficients.csv
    rows = []
    for p in glob.glob(os.path.join(run_dir, "stage5", "shards", "*", "stage5_models_coefficients.csv")):
        try:
            df = pd.read_csv(p)
            idc = next(
                (c for c in df.columns
                 if c.lower() in ("parameter", "coef_name", "coefficient_name", "name", "variable")),
                df.columns[0],
            )
            rows.append(df.loc[df[idc].astype(str).str.lower().str.contains("crash", na=False)])
        except Exception:
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def mode_shares(t: pd.DataFrame) -> pd.Series:
    if t.empty:
        return pd.Series(dtype=float)
    return t["mode_name"].value_counts(normalize=True).reindex(MODE_ORDER).fillna(0) * 100


def _num(t: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column as Series; all-NaN Series when the column is absent."""
    if col in t.columns:
        return pd.to_numeric(t[col], errors="coerce")
    return pd.Series(np.nan, index=t.index)


def demand_aggregates(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    d = _num(t, "dist").fillna(0).clip(lower=0)
    tt = _num(t, "travtime")
    if tt.median() > 200:  # seconds -> minutes guard
        tt = tt / 60.0
    n_pax = t["person_id"].nunique() if "person_id" in t.columns else np.nan
    mot = t["is_motor"] if "is_motor" in t.columns else t["mode_name"].isin(MOTORISED)
    act = t["is_active"] if "is_active" in t.columns else t["mode_name"].isin(ACTIVE)
    return {
        "n_trips": int(len(t)),
        "n_persons": int(n_pax) if pd.notna(n_pax) else np.nan,
        "trips_per_person": float(len(t) / n_pax) if n_pax else np.nan,
        "vkt_motor_km": float(d[mot].sum()),
        "active_km": float(d[act].sum()),
        "mean_dist_km": float(d.mean()),
        "p90_dist_km": float(d.quantile(0.9)),
        "mean_time_min": float(tt.mean()),
        "active_share_pct": float(act.mean() * 100),
        "motor_share_pct": float(mot.mean() * 100),
    }


def active_by_distance(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty or "dist" not in t.columns:
        return pd.DataFrame()
    d = t[["dist", "is_active", "purpose"]].copy()
    d["dist"] = pd.to_numeric(d["dist"], errors="coerce")
    d = d.dropna(subset=["dist"])
    d["band"] = pd.cut(d["dist"], bins=DIST_BINS, labels=DIST_LABELS, right=False)
    rows = []
    for purp, g in d.groupby("purpose"):
        s = g.groupby("band", observed=True)["is_active"].mean() * 100
        for band in DIST_LABELS:
            rows.append({"purpose": purp, "band": band,
                         "active_pct": float(s.get(band, np.nan)),
                         "n_trips": int((g["band"] == band).sum())})
    return pd.DataFrame(rows)


def population_table(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty:
        return pd.DataFrame()
    g = t.groupby("ptype")
    out = pd.DataFrame({
        "n_trips": g.size(),
        "active_pct": g["is_active"].mean() * 100,
        "motor_pct": g["is_motor"].mean() * 100,
        "mean_dist_km": _num(t, "dist").groupby(t["ptype"]).mean(),
        "mean_crash": _num(t, "crash").groupby(t["ptype"]).mean(),
    }).reset_index().sort_values("n_trips", ascending=False)
    return out


def spatial_table(tb: pd.DataFrame, tn: pd.DataFrame | None) -> pd.DataFrame:
    """Per-origin-zone active share per arm + delta; per-dest counts per arm."""
    if tb.empty:
        return pd.DataFrame()
    b = tb.assign(active=tb["is_active"].astype(float)).groupby("OTAZ").agg(
        n_trips_b=("active", "size"), active_b=("active", "mean"))
    b["active_b"] *= 100
    db = tb.groupby("DTAZ").size().rename("dest_n_b")
    if tn is not None and not tn.empty:
        n = tn.assign(active=tn["is_active"].astype(float)).groupby("OTAZ").agg(
            n_trips_n=("active", "size"), active_n=("active", "mean"))
        n["active_n"] *= 100
        dn = tn.groupby("DTAZ").size().rename("dest_n_n")
        out = b.join(n, how="outer").fillna(0)
        out["d_active_pp"] = out["active_n"] - out["active_b"]
        dest = pd.DataFrame({"dest_n_b": db, "dest_n_n": dn}).fillna(0)
        dest["d_dest_n"] = dest["dest_n_n"] - dest["dest_n_b"]
        out = out.join(dest, how="left")
    else:
        out = b
        out["d_active_pp"] = np.nan
    return out.reset_index().sort_values("d_active_pp", ascending=True)


def crash_exposure_table(t: pd.DataFrame, hi_thresh: float | None = None) -> tuple[pd.DataFrame, float]:
    if t.empty or "crash" not in t.columns:
        return pd.DataFrame(), np.nan
    c = pd.to_numeric(t["crash"], errors="coerce").dropna().clip(lower=0)
    if c.empty:
        return pd.DataFrame(), np.nan
    if hi_thresh is None or not np.isfinite(hi_thresh):
        hi_thresh = float(c.quantile(0.75))
    rows = [{
        "group": "all",
        "mean": float(c.mean()), "p50": float(c.median()), "p90": float(c.quantile(0.9)),
        "share_high_crash_pct": float((c > hi_thresh).mean() * 100),
    }]
    for m in MODE_ORDER:
        cm = pd.to_numeric(t.loc[t["mode_name"] == m, "crash"], errors="coerce").dropna().clip(lower=0)
        if len(cm):
            rows.append({
                "group": m,
                "mean": float(cm.mean()), "p50": float(cm.median()), "p90": float(cm.quantile(0.9)),
                "share_high_crash_pct": float((cm > hi_thresh).mean() * 100),
            })
    return pd.DataFrame(rows), hi_thresh


def dta_table(t: pd.DataFrame, crosswalk: pd.DataFrame | None) -> pd.DataFrame:
    if t.empty:
        return pd.DataFrame()
    d = t[t["is_motor"]].copy()
    if crosswalk is not None and not crosswalk.empty and "ORDER" in crosswalk.columns:
        in_net = set(crosswalk.loc[crosswalk.get("in_network", True).fillna(True).astype(bool), "ORDER"].astype(int))
        d["in_network"] = (pd.to_numeric(d["OTAZ"], errors="coerce").isin(in_net)
                           & pd.to_numeric(d["DTAZ"], errors="coerce").isin(in_net))
        g = d.groupby(["time_period", "in_network"]).size().unstack(fill_value=0)
    else:
        g = d.groupby("time_period").size().to_frame("motor_trips")
    g = g.reset_index()
    tp = pd.CategoricalDtype(categories=TIME_ORDER, ordered=True)
    if "time_period" in g.columns:
        g["time_period"] = g["time_period"].astype(str)
        g = g.sort_values("time_period",
                          key=lambda s: s.map({v: i for i, v in enumerate(TIME_ORDER)}).fillna(99))
    return g


# ---------------------------------------------------------------------------
# figures (paper style)
# ---------------------------------------------------------------------------

def _paper_style():
    plt.rcParams.update(PAPER_RC)


def _note_no_counterfactual(ax):
    ax.text(0.5, 0.5, "M_0 (no-safety) not run yet — single-arm audit",
            transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#7f8c8d")


def fig_mode_shift(tb, tn, path):
    _paper_style()
    sb, sn = mode_shares(tb), mode_shares(tn) if tn is not None else pd.Series(dtype=float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    fig.suptitle("Mode share: safety-integrated (M_S) vs no-safety (M_0)", fontweight="bold", fontsize=10)
    x = np.arange(len(MODE_ORDER))
    bv = [float(sb.get(m, 0)) for m in MODE_ORDER]
    cols = [MODE_COLORS[m] for m in MODE_ORDER]
    if tn is None or tn.empty:
        a1.bar(x, bv, color=cols, alpha=0.85)
        for xp, v in zip(x, bv):
            if v > 0.5:
                a1.text(xp, v + 0.3, f"{v:.1f}%", ha="center", fontsize=7)
        _note_no_counterfactual(a2)
    else:
        nv = [float(sn.get(m, 0)) for m in MODE_ORDER]
        w = 0.35
        a1.bar(x - w / 2, bv, w, color=cols, alpha=0.9, label="M_S safety ON")
        a1.bar(x + w / 2, nv, w, color=cols, alpha=0.45, hatch="//", edgecolor="grey", label="M_0 safety OFF")
        for xp, v in zip(x - w / 2, bv):
            if v > 0.5:
                a1.text(xp, v + 0.3, f"{v:.1f}%", ha="center", fontsize=7)
        a1.legend(fontsize=7)
        deltas = [n - b for b, n in zip(bv, nv)]
        a2.barh(MODE_ORDER, deltas,
                color=["#27AE60" if d < 0 else "#C0392B" for d in deltas], alpha=0.85)
        a2.axvline(0, color="black", lw=0.8)
        a2.set_xlabel("Δ share, M_0 − M_S (pp)")
        for i, d in enumerate(deltas):
            if abs(d) > 0.05:
                a2.text(d + (0.05 if d >= 0 else -0.05), i, f"{d:+.2f}pp",
                        va="center", ha="left" if d >= 0 else "right", fontsize=8)
    a1.set_xticks(x)
    a1.set_xticklabels(MODE_ORDER, rotation=15, ha="right")
    a1.set_ylabel("Share (%)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_demand_deltas(agg_b, agg_n, path):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    keys = [("vkt_motor_km", "Motor VKT (km)"), ("active_km", "Active km (walk+bike)"),
            ("n_trips", "Trips"), ("mean_time_min", "Mean time (min)")]
    if not agg_b or not agg_n:
        vals = [agg_b.get(k, np.nan) for k, _ in keys]
        ax.bar([l for _, l in keys], vals, color="#2E86AB", alpha=0.85)
        ax.set_title("Demand aggregates — M_S only (M_0 not run yet)")
        ax.set_ylabel("Absolute value (log scale)")
        ax.set_yscale("log")
    else:
        pct = []
        for k, _ in keys:
            b, n = agg_b.get(k, np.nan), agg_n.get(k, np.nan)
            if pd.notna(b) and pd.notna(n) and b:
                pct.append((n - b) / b * 100)
            else:
                pct.append(np.nan)
        ax.bar([l for _, l in keys], pct, color=["#27AE60" if v < 0 else "#C0392B" for v in pct], alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Demand change when safety skims are removed (M_0 − M_S, %)")
        ax.set_ylabel("Change (%)")
        for i, v in enumerate(pct):
            if np.isfinite(v):
                ax.text(i, v + (0.2 if v >= 0 else -0.4), f"{v:+.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_active_by_distance(ab, an, path):
    _paper_style()
    purps = ["work", "edu", "nonmand"] if False else sorted(
        set(ab["purpose"].tolist()) if not ab.empty else [])
    # collapse to the three estimation segments where possible
    seg_map = {"work": "Work", "edu": "Education", "nonmand": "Non-mandatory",
               "social": "Non-mandatory", "escort": "Non-mandatory",
               "shopping": "Non-mandatory", "othmaint": "Non-mandatory",
               "othdiscr": "Non-mandatory", "eat": "Non-mandatory"}
    for df in (ab, an):
        if not df.empty:
            df["seg"] = df["purpose"].map(seg_map).fillna(df["purpose"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    fig.suptitle("Active-mode share (walk+bike) by distance band", fontweight="bold", fontsize=10)
    if ab.empty and (an is None or an.empty):
        fig.text(0.5, 0.5, "no distance data in trip table", ha="center", fontsize=10, color="#7f8c8d")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return
    def _seg_curve(df, seg):
        if df.empty:
            return [np.nan] * len(DIST_LABELS)
        g = df[df["seg"] == seg].groupby("band")[["active_pct"]].mean()
        return [float(g.loc[b, "active_pct"]) if b in g.index else np.nan for b in DIST_LABELS]

    for ax, seg in zip(axes, ["Work", "Education", "Non-mandatory"]):
        ax.plot(range(len(DIST_LABELS)), _seg_curve(ab, seg),
                marker="o", color="#27AE60", label="M_S safety ON", lw=1.8)
        if an is not None and not an.empty:
            ax.plot(range(len(DIST_LABELS)), _seg_curve(an, seg),
                    marker="s", ls="--", color="#C0392B", label="M_0 safety OFF", lw=1.5)
        ax.set_xticks(range(len(DIST_LABELS)))
        ax.set_xticklabels(DIST_LABELS, rotation=30, ha="right", fontsize=7)
        ax.set_title(seg, fontsize=9)
        ax.set_ylabel("Active share (%)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_population(pb, pn, path):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if pb.empty:
        ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
    elif pn is None or pn.empty:
        s = pb.set_index("ptype")["active_pct"].sort_values(ascending=False).head(8)
        ax.barh(s.index.astype(str), s.values, color="#2E86AB", alpha=0.85)
        ax.set_xlabel("Active share, M_S (%)")
        ax.set_title("Population coverage — active share by person type (M_S only)")
    else:
        m = pd.merge(pb[["ptype", "active_pct"]].rename(columns={"active_pct": "b"}),
                      pn[["ptype", "active_pct"]].rename(columns={"active_pct": "n"}),
                      on="ptype", how="inner")
        m["d"] = m["n"] - m["b"]
        m = m.sort_values("d").tail(10)
        ax.barh(m["ptype"].astype(str), m["d"],
                color=["#27AE60" if v < 0 else "#C0392B" for v in m["d"]], alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Δ active share, M_0 − M_S (pp)")
        ax.set_title("Population coverage — who gains/loses active travel without safety")
        for i, v in enumerate(m["d"].values):
            ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}pp",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_spatial(spatial, path):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if spatial.empty or spatial["d_active_pp"].isna().all():
        if not spatial.empty:
            s = spatial.sort_values("active_b", ascending=False).head(15)
            ax.barh(s["OTAZ"].astype(str), s["active_b"], color="#2E86AB", alpha=0.85)
            ax.set_xlabel("Active share, M_S (%)")
            ax.set_title("Spatial coverage — top-15 active origin zones (M_S only)")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
    else:
        d = spatial.dropna(subset=["d_active_pp"]).sort_values("d_active_pp")
        top = pd.concat([d.head(8), d.tail(8)])
        ax.barh(top["OTAZ"].astype(str), top["d_active_pp"],
                color=["#27AE60" if v < 0 else "#C0392B" for v in top["d_active_pp"]], alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Δ active share, M_0 − M_S (pp)")
        ax.set_ylabel("Origin zone (OTAZ)")
        ax.set_title("Spatial coverage — zones most/least affected by safety removal")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_exposure(tb, tn, hi_thresh, path):
    _paper_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    fig.suptitle("Crash-exposure of simulated trips", fontweight="bold", fontsize=10)
    for run, col, ls, lbl in [(tb, "#2E86AB", "-", "M_S safety ON"),
                              (tn, "#C0392B", "--", "M_0 safety OFF")]:
        if run is None or run.empty or "crash" not in run.columns:
            continue
        v = pd.to_numeric(run["crash"], errors="coerce").dropna().clip(lower=0)
        v = v[v < v.quantile(0.99)]
        if len(v) > 50:
            a1.hist(np.log1p(v), bins=40, density=True, alpha=0.45, color=col, label=lbl)
    a1.set_xlabel("log(1 + OD crash frequency)")
    a1.set_ylabel("Density")
    if a1.get_legend_handles_labels()[0]:
        a1.legend(fontsize=7)
    elif tb.empty:
        a1.text(0.5, 0.5, "no crash data in trip table", ha="center",
                transform=a1.transAxes, fontsize=9, color="#7f8c8d")
    # high-crash share by mode (baseline arm)
    if not tb.empty and np.isfinite(hi_thresh):
        rows = []
        for m in MODE_ORDER:
            cm = pd.to_numeric(tb.loc[tb["mode_name"] == m, "crash"], errors="coerce").dropna()
            if len(cm):
                rows.append((m, float((cm > hi_thresh).mean() * 100)))
        if rows:
            ms, vs = zip(*rows)
            a2.bar(ms, vs, color=[MODE_COLORS[m] for m in ms], alpha=0.85)
            a2.set_ylabel(f"% trips above high-crash threshold ({hi_thresh:.1f})")
            a2.set_title("Mode coverage of crash exposure (M_S)")
            plt.setp(a2.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_dta(db, dn, path):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    has_split = db is not None and not db.empty and "in_network" in str(db.columns.tolist())
    if (db is None or db.empty) and (dn is None or dn.empty):
        ax.text(0.5, 0.5, "no motorised trips", ha="center", transform=ax.transAxes)
    elif dn is None or dn.empty:
        vals = db.set_index("time_period").reindex(TIME_ORDER) if "time_period" in db.columns else db
        col = "motor_trips" if "motor_trips" in vals.columns else vals.columns[0]
        ax.bar(TIME_ORDER, [vals.loc[t, col] if t in vals.index else 0 for t in TIME_ORDER],
               color="#2E86AB", alpha=0.85)
        ax.set_title("DTA coverage — motorised trips by time period (M_S only)")
        ax.set_ylabel("Trips")
    else:
        def _vec(df):
            v = df.set_index("time_period").reindex(TIME_ORDER)
            col = "motor_trips" if "motor_trips" in v.columns else v.columns[0]
            return [v.loc[t, col] if t in v.index else 0 for t in TIME_ORDER]
        bv, nv = _vec(db), _vec(dn)
        x = np.arange(len(TIME_ORDER))
        w = 0.35
        ax.bar(x - w / 2, bv, w, color="#2E86AB", alpha=0.9, label="M_S safety ON")
        ax.bar(x + w / 2, nv, w, color="#C0392B", alpha=0.6, hatch="//", edgecolor="grey", label="M_0 safety OFF")
        ax.set_xticks(x)
        ax.set_xticklabels(TIME_ORDER, rotation=15, ha="right")
        ax.set_title("DTA coverage — motorised demand by time period")
        ax.set_ylabel("Trips")
        ax.legend(fontsize=7)
    _ = has_split  # crosswalk in/out split is tabulated in CSV; figure stays period-based
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_destination(tb, tn, path):
    _paper_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    fig.suptitle("Destination patterns", fontweight="bold", fontsize=10)
    cb = tb.groupby("DTAZ").size() if not tb.empty else pd.Series(dtype=float)
    if tn is None or tn.empty:
        top = cb.sort_values(ascending=False).head(20)
        a1.bar(top.index.astype(str), top.values, color="#2E86AB", alpha=0.85)
        a1.set_xlabel("Destination zone (top-20, M_S)")
        a1.set_ylabel("Trips")
        _note_no_counterfactual(a2)
    else:
        cn = tn.groupby("DTAZ").size()
        j = pd.DataFrame({"b": cb, "n": cn}).fillna(0)
        a1.scatter(j["b"], j["n"], s=8, alpha=0.5, color="#2E86AB")
        mx = max(j["b"].max(), j["n"].max(), 1)
        a1.plot([0, mx], [0, mx], "k--", lw=0.8)
        a1.set_xlabel("Trips per dest zone, M_S")
        a1.set_ylabel("Trips per dest zone, M_0")
        a1.set_title("Zone destination counts")
        # mean dist by purpose
        rows = []
        if "dist" in tb.columns and "dist" in tn.columns:
            for p in sorted(set(tb["purpose"].dropna().tolist()) & set(tn["purpose"].dropna().tolist())):
                db = _num(tb[tb["purpose"] == p], "dist").mean()
                dn_ = _num(tn[tn["purpose"] == p], "dist").mean()
                rows.append((p, dn_ - db))
        if rows:
            ps, ds = zip(*rows)
            a2.barh(list(ps), list(ds),
                    color=["#27AE60" if v < 0 else "#C0392B" for v in ds], alpha=0.85)
            a2.axvline(0, color="black", lw=0.8)
            a2.set_xlabel("Δ mean dist, M_0 − M_S (km)")
            a2.set_title("Distance shift by purpose")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# latex
# ---------------------------------------------------------------------------

def write_latex(out_tex, ms_rows, agg_b, agg_n, pop_delta) -> None:
    def _f(x, nd=2):
        return f"{x:.{nd}f}" if pd.notna(x) and np.isfinite(x) else "--"
    lines = ["% Auto-generated by safety_plan_impact.py — paste into the paper",
             "\\begin{table}[htbp]\\centering\\caption{Safety impact on mode shares (M\\_S vs M\\_0).}\\label{tab:safety-modes}",
             "\\begin{tabular}{lccc}\\toprule",
             "Mode & M\\_S (\\%) & M\\_0 (\\%) & $\\Delta$ (pp) \\\\ \\midrule"]
    for r in ms_rows:
        lines.append(f"{r['mode']} & {_f(r['base'])} & {_f(r['nosa'])} & {_f(r['delta'], 2)} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}", "",
              "\\begin{table}[htbp]\\centering\\caption{Safety impact on demand aggregates.}\\label{tab:safety-demand}",
              "\\begin{tabular}{lccc}\\toprule",
              "Metric & M\\_S & M\\_0 & $\\Delta$ (\\%) \\\\ \\midrule"]
    for key, lbl in [("n_trips", "Trips"), ("vkt_motor_km", "Motor VKT (km)"),
                     ("active_km", "Active km"), ("mean_time_min", "Mean time (min)"),
                     ("active_share_pct", "Active share (\\%)")]:
        b, n = agg_b.get(key, np.nan), agg_n.get(key, np.nan)
        d = (n - b) / b * 100 if pd.notna(b) and b else np.nan
        lines.append(f"{lbl} & {_f(b, 1)} & {_f(n, 1)} & {_f(d, 1)} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    if pop_delta:
        lines += ["",
                  "\\begin{table}[htbp]\\centering\\caption{Population coverage: active-share change (pp, M\\_0 $-$ M\\_S) by person type.}\\label{tab:safety-pop}",
                  "\\begin{tabular}{lc}\\toprule",
                  "Person type & $\\Delta$ active (pp) \\\\ \\midrule"]
        for pt, d in pop_delta:
            lines.append(f"{pt} & {_f(d)} \\\\")
        lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Paper-ready safety impact on travel plans + coverages")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--nosafety", default=None)
    ap.add_argument("--out", default="runs/safety_plan_impact")
    ap.add_argument("--crosswalk", default=None)
    ap.add_argument("--chunksize", type=int, default=500_000)
    a = ap.parse_args()

    run_root = os.environ.get("PIPELINE_RUN_ROOT", "runs")
    base_dir = a.baseline or _latest_run_dir("safety_baseline", run_root)
    nosa_dir = a.nosafety or _latest_run_dir("safety_nosafety", run_root)
    # fall back: newest run with a stage6 output (single-arm audit of whatever exists)
    if base_dir is None or _find_trips_file(base_dir) is None:
        cands = [d for d in glob.glob(os.path.join(run_root, "*"))
                 if os.path.isdir(d) and _find_trips_file(d)]
        cands += [d for d in glob.glob(os.path.join(run_root, "ga_parallel", "*"))
                  if os.path.isdir(d) and _find_trips_file(d)]
        base_dir = max(cands, key=os.path.getmtime) if cands else base_dir

    print("=" * 70)
    print("SAFETY PLAN IMPACT — travel plans + coverages (paper-ready)")
    print(f"  M_S baseline : {base_dir}")
    print(f"  M_0 nosafety : {nosa_dir}")
    print(f"  output       : {a.out}")
    print("=" * 70)
    if base_dir is None or _find_trips_file(base_dir) is None:
        print("ERROR: no run with stage6_output.csv / trips_for_aimsun.csv found.")
        return 2

    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)

    print(f"Loading M_S trips from {base_dir} ...")
    tb = load_trips(base_dir, a.chunksize)
    print(f"  M_S: {len(tb):,} trips")
    tn = None
    if nosa_dir and os.path.isdir(nosa_dir) and _find_trips_file(nosa_dir):
        print(f"Loading M_0 trips from {nosa_dir} ...")
        tn = load_trips(nosa_dir, a.chunksize)
        print(f"  M_0: {len(tn):,} trips")
    else:
        print("  M_0: not found — single-arm audit (deltas left empty)")
    two_arm = tn is not None and not tn.empty

    crosswalk = None
    if a.crosswalk and os.path.isfile(a.crosswalk):
        try:
            crosswalk = pd.read_csv(a.crosswalk)
            print(f"  crosswalk: {a.crosswalk} ({len(crosswalk):,} zones)")
        except Exception as e:
            print(f"  crosswalk load failed ({e}); DTA table without in/out split")

    T, F = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")

    # ---- 01 mode shares (overall + by purpose) ----
    sb, sn = mode_shares(tb), mode_shares(tn) if two_arm else pd.Series(dtype=float)
    ms_rows, ms_purp = [], []
    for m in MODE_ORDER:
        b, n = float(sb.get(m, 0)), float(sn.get(m, np.nan)) if two_arm else np.nan
        ms_rows.append({"mode": m, "base": b, "nosa": n,
                        "delta": (n - b) if two_arm else np.nan})
    pd.DataFrame(ms_rows).to_csv(os.path.join(T, "01_mode_share_by_arm.csv"), index=False)
    for p in sorted(tb["purpose"].dropna().unique().tolist()):
        gb = tb[tb["purpose"] == p]["mode_name"].value_counts(normalize=True) * 100
        row = {"purpose": p}
        for m in MODE_ORDER:
            row[f"base_{m}"] = float(gb.get(m, 0))
            if two_arm:
                gn = tn[tn["purpose"] == p]["mode_name"].value_counts(normalize=True) * 100
                row[f"nosa_{m}"] = float(gn.get(m, 0))
                row[f"d_{m}_pp"] = row[f"nosa_{m}"] - row[f"base_{m}"]
        ms_purp.append(row)
    if ms_purp:
        pd.DataFrame(ms_purp).to_csv(os.path.join(T, "01b_mode_share_by_purpose.csv"), index=False)

    # ---- 02 demand aggregates ----
    agg_b, agg_n = demand_aggregates(tb), demand_aggregates(tn) if two_arm else {}
    d2 = []
    for k in ["n_trips", "n_persons", "trips_per_person", "vkt_motor_km", "active_km",
              "mean_dist_km", "p90_dist_km", "mean_time_min", "active_share_pct", "motor_share_pct"]:
        b, n = agg_b.get(k, np.nan), agg_n.get(k, np.nan) if two_arm else np.nan
        d2.append({"metric": k, "base": b, "nosa": n,
                   "delta": (n - b) if two_arm and pd.notna(n) and pd.notna(b) else np.nan,
                   "delta_pct": ((n - b) / b * 100) if two_arm and pd.notna(n) and pd.notna(b) and b else np.nan})
    pd.DataFrame(d2).to_csv(os.path.join(T, "02_demand_aggregates.csv"), index=False)

    # ---- 03 active by distance ----
    ab = active_by_distance(tb)
    an = active_by_distance(tn) if two_arm else pd.DataFrame()
    if not ab.empty:
        m = pd.merge(ab, an, on=["purpose", "band"], how="left", suffixes=("_b", "_n")) if two_arm else ab.rename(
            columns={"active_pct": "active_pct_b", "n_trips": "n_trips_b"})
        if two_arm:
            m["d_active_pp"] = m["active_pct_n"] - m["active_pct_b"]
        m.to_csv(os.path.join(T, "03_active_by_distance.csv"), index=False)

    # ---- 04 population ----
    pb = population_table(tb)
    pn = population_table(tn) if two_arm else pd.DataFrame()
    if not pb.empty:
        if two_arm and not pn.empty:
            m = pd.merge(pb, pn, on="ptype", how="outer", suffixes=("_b", "_n"))
            m["d_active_pp"] = m["active_pct_n"] - m["active_pct_b"]
            m["d_trips"] = m["n_trips_n"] - m["n_trips_b"]
        else:
            m = pb
        m.to_csv(os.path.join(T, "04_population_coverage.csv"), index=False)

    # ---- 05 spatial ----
    sp = spatial_table(tb, tn if two_arm else None)
    if not sp.empty:
        sp.to_csv(os.path.join(T, "05_spatial_coverage.csv"), index=False)
        top = sp.dropna(subset=["d_active_pp"]).sort_values("d_active_pp")
        if not top.empty:
            pd.concat([top.head(20), top.tail(20)]).to_csv(
                os.path.join(T, "05b_spatial_top_zones.csv"), index=False)

    # ---- 06 DTA ----
    db = dta_table(tb, crosswalk)
    dn_tbl = dta_table(tn, crosswalk) if two_arm else pd.DataFrame()
    if not db.empty:
        db.assign(arm="base").to_csv(os.path.join(T, "06_dta_coverage_base.csv"), index=False)
    if two_arm and not dn_tbl.empty:
        dn_tbl.assign(arm="nosa").to_csv(os.path.join(T, "06_dta_coverage_nosa.csv"), index=False)
        both = pd.merge(db, dn_tbl, on=["time_period"], how="outer", suffixes=("_b", "_n"))
        both.to_csv(os.path.join(T, "06_dta_coverage.csv"), index=False)

    # ---- 07 crash exposure ----
    exp_b, hi = crash_exposure_table(tb)
    exp_b.assign(arm="base").to_csv(os.path.join(T, "07_crash_exposure_base.csv"), index=False) if not exp_b.empty else None
    if two_arm:
        exp_n, _ = crash_exposure_table(tn, hi)
        if not exp_n.empty:
            exp_n.assign(arm="nosa").to_csv(os.path.join(T, "07_crash_exposure_nosa.csv"), index=False)

    # ---- 08 tour stats (from stage6 row_type mix when available) ----
    t8 = []
    for lbl, run_dir, trips in [("base", base_dir, tb), ("nosa", nosa_dir, tn)]:
        if trips is None:
            continue
        src = _find_trips_file(run_dir) if run_dir and os.path.isdir(run_dir) else None
        n_acts = np.nan
        if src and os.path.basename(src) == "stage6_output.csv":
            try:
                n_acts = int(pd.read_csv(src, usecols=["row_type"], chunksize=1_000_000).__iter__().__next__()["row_type"].eq("activity").sum())
                # full count via chunks
                n_acts = 0
                for ch in pd.read_csv(src, usecols=["row_type"], chunksize=1_000_000, low_memory=True):
                    n_acts += int((ch["row_type"].astype(str) == "activity").sum())
            except Exception:
                n_acts = np.nan
        n_pax = trips["person_id"].nunique() if "person_id" in trips.columns else np.nan
        t8.append({"arm": lbl, "n_trips": len(trips), "n_activities": n_acts,
                   "n_persons": int(n_pax) if pd.notna(n_pax) else np.nan,
                   "trips_per_person": float(len(trips) / n_pax) if n_pax else np.nan})
    if t8:
        pd.DataFrame(t8).to_csv(os.path.join(T, "08_tour_stats.csv"), index=False)

    # ---- headline numbers ----
    hl = [
        ("active_share_base", agg_b.get("active_share_pct", np.nan)),
        ("active_share_nosa", agg_n.get("active_share_pct", np.nan) if two_arm else np.nan),
        ("active_share_d_pp", (agg_n.get("active_share_pct", np.nan) - agg_b.get("active_share_pct", np.nan)) if two_arm else np.nan),
        ("motor_vkt_base", agg_b.get("vkt_motor_km", np.nan)),
        ("motor_vkt_nosa", agg_n.get("vkt_motor_km", np.nan) if two_arm else np.nan),
        ("active_km_base", agg_b.get("active_km", np.nan)),
        ("active_km_nosa", agg_n.get("active_km", np.nan) if two_arm else np.nan),
        ("mean_dist_base", agg_b.get("mean_dist_km", np.nan)),
        ("mean_dist_nosa", agg_n.get("mean_dist_km", np.nan) if two_arm else np.nan),
        ("hi_crash_thresh", hi if "hi" in dir() else np.nan),
    ]
    for r in ms_rows:
        hl.append((f"mode_{r['mode']}_base", r["base"]))
        hl.append((f"mode_{r['mode']}_delta_pp", r["delta"]))
    pd.DataFrame(hl, columns=["metric", "value"]).to_csv(os.path.join(T, "headline_numbers.csv"), index=False)

    # ---- figures ----
    fig_mode_shift(tb, tn if two_arm else None, os.path.join(F, "fig01_mode_shift.png"))
    fig_demand_deltas(agg_b, agg_n if two_arm else {}, os.path.join(F, "fig02_demand_deltas.png"))
    fig_active_by_distance(ab, an if two_arm else pd.DataFrame(), os.path.join(F, "fig03_active_by_distance.png"))
    fig_population(pb, pn if two_arm else pd.DataFrame(), os.path.join(F, "fig04_population.png"))
    fig_spatial(sp, os.path.join(F, "fig05_spatial_topzones.png"))
    fig_exposure(tb, tn if two_arm else None, hi if "hi" in dir() else np.nan, os.path.join(F, "fig06_crash_exposure.png"))
    fig_dta(db, dn_tbl if two_arm else pd.DataFrame(), os.path.join(F, "fig07_dta_coverage.png"))
    fig_destination(tb, tn if two_arm else None, os.path.join(F, "fig08_destination_shift.png"))

    # ---- latex ----
    pop_delta = []
    if two_arm and not pb.empty and not pn.empty:
        m = pd.merge(pb[["ptype", "active_pct"]], pn[["ptype", "active_pct"]],
                      on="ptype", suffixes=("_b", "_n"))
        pop_delta = [(r["ptype"], r["active_pct_n"] - r["active_pct_b"]) for _, r in m.iterrows()]
    write_latex(os.path.join(a.out, "tables_for_paper.tex"), ms_rows, agg_b, agg_n if two_arm else {}, pop_delta)

    # ---- stdout headline ----
    print("\nHeadline (M_S safety ON" + (" vs M_0 safety OFF)" if two_arm else ", M_0 not run yet)"))
    for m in MODE_ORDER:
        b = sb.get(m, 0)
        d = (sn.get(m, np.nan) - b) if two_arm else np.nan
        print(f"  {m:<8} M_S={b:5.2f}%   delta={d:+.2f}pp" if two_arm else f"  {m:<8} M_S={b:5.2f}%")
    print(f"  active share M_S={agg_b.get('active_share_pct', np.nan):.2f}%"
          + (f"   delta={agg_n.get('active_share_pct', np.nan) - agg_b.get('active_share_pct', np.nan):+.2f}pp" if two_arm else ""))
    _bvkt, _nvkt = agg_b.get("vkt_motor_km", np.nan), (agg_n.get("vkt_motor_km", np.nan) if two_arm else np.nan)
    _vkt_str = ""
    if two_arm and pd.notna(_bvkt) and pd.notna(_nvkt) and _bvkt:
        _vkt_str = f"   delta={(_nvkt - _bvkt) / _bvkt * 100:+.1f}%"
    print(f"  motor VKT    M_S={_bvkt:,.0f} km" + _vkt_str)
    print(f"\nDONE — tables: {T}  figures: {F}  latex: {os.path.join(a.out, 'tables_for_paper.tex')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
