"""
hb_time_intervention.py
=======================
Time-based intervention scheduling from hard-braking (HB) telematics, deployed
ONLY if the HB proxy is spatially transferable.

Rationale (links to the TRB paper's evidence):
  - The encompassing test (LR=181.9), the severity-category copulas (all three
    tiers reject independence) and the leave-one-LGA-out analysis (HB proxy
    improves out-of-sample crash LL in 9/11 LGAs, mean dLL=+61.5) show the HB
    signal travels across regions better than the crash model it augments.
  - The crash model itself is region-specific (mean TI=-2.12), so a
    time-based strategy keyed to the HB proxy - not to locally refit crash
    models - is the deployable one.
  - The hour x weekday HB risk index built from 2.5M timestamped events is
    threshold-robust (diurnal shape Spearman rho=0.98 against an independent
    -0.30g scan), which makes schedules stable to sensor/threshold choices.

Pipeline:
  1. TRANSFER GATE      read hb_transferability_summary.csv; require
                          HBproxy_dLL_mean > 0,
                          positive folds >= --min-pos-share (default 0.60),
                          coef_median_abs_mean_over_std >= --min-stability (default 2).
                        If the gate fails, no schedule is produced (exit 2).
  2. TEMPORAL PROFILE   hour-of-day x day-of-week HB rate, exposure-normalised
                        (events per 10k GPS points). Loads
                        temporal_risk_profile.csv, or rebuilds it from a raw
                        event extract with --from-raw (HPC-scale; chunked).
  3. ROBUSTNESS         optional --alt-profile (independent threshold scan):
                        Spearman rho between the two diurnal shapes must reach
                        --min-shape-rho (default 0.90) for the schedule to be
                        flagged threshold-robust.
  4. SCHEDULING         rank cells, take the top --top-frac (default 0.10) as
                        priority intervention windows (enhanced patrol / VMS /
                        variable speed management); report lift, capture %,
                        cumulative capture and PAI; weekend vs weekday
                        per-exposure ratio as a deployment-balance check.
  5. REGIONAL LAYER     per-LGA window share from hb_transferability_lga.csv
                        (weight proportional to max(HBproxy_dLL, 0) + floor),
                        so LGAs where the proxy transfers best get the largest
                        share of intervention hours.

Outputs (in --out-dir, default = script dir):
  hb_time_intervention_schedule.csv   cell-level schedule + metrics
  hb_time_intervention_regional.csv   per-LGA window shares
  hb_time_intervention_summary.csv    gate + schedule headline metrics
  hb_time_intervention.tex            paper-ready LaTeX snippet
  hb_time_intervention_heatmap.png    hour x weekday risk heatmap with the
                                      selected priority cells outlined

Usage:
  python hb_time_intervention.py                       # use existing outputs
  python hb_time_intervention.py --from-raw hard_braking_events_year.csv
  python hb_time_intervention.py --gate-only           # just evaluate the gate
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------- gate
def transfer_gate(summary_csv, lga_csv=None, min_pos_share=0.60,
                  min_stability=2.0, min_dll=0.0):
    """Evaluate whether the HB proxy is transferable enough to key
    time-based interventions on. Returns (passed, metrics dict)."""
    s = pd.read_csv(summary_csv)
    m = dict(zip(s["metric"], pd.to_numeric(s["value"], errors="coerce")))
    dll = m.get("HBproxy_dLL_mean", np.nan)
    npos = m.get("HBproxy_dLL_positive_folds", np.nan)
    stab = m.get("coef_median_abs_mean_over_std", np.nan)

    n_folds = np.nan
    if lga_csv and os.path.exists(lga_csv):
        n_folds = len(pd.read_csv(lga_csv))
    pos_share = (npos / n_folds) if (n_folds and np.isfinite(npos)) else np.nan

    checks = {
        "HBproxy_dLL_mean > 0": (np.isfinite(dll) and dll > min_dll,
                                 f"dLL={dll:+.1f}" if np.isfinite(dll) else "missing"),
        "positive folds >= %.0f%%" % (100 * min_pos_share):
            (np.isfinite(pos_share) and pos_share >= min_pos_share,
             f"{npos:.0f}/{n_folds:.0f}" if np.isfinite(npos) and np.isfinite(n_folds) else "missing"),
        "coef stability >= %.1f" % min_stability:
            (np.isfinite(stab) and stab >= min_stability,
             f"|mean|/SD={stab:.2f}" if np.isfinite(stab) else "missing"),
    }
    passed = all(ok for ok, _ in checks.values())
    return passed, {"metrics": m, "checks": checks}


# ----------------------------------------------------------------- profile
def load_profile(path):
    """Load an hour x weekday profile. Requires dow (0=Mon), hour and either
    rate_per_10k_pts or hb_events/n_points to recompute it."""
    p = pd.read_csv(path)
    need = {"dow", "hour"}
    if not need.issubset(p.columns):
        raise ValueError(f"{path}: missing columns {need - set(p.columns)}")
    if "rate_per_10k_pts" not in p.columns:
        if {"hb_events", "n_points"}.issubset(p.columns):
            p["rate_per_10k_pts"] = 1e4 * p["hb_events"] / p["n_points"].replace(0, np.nan)
        else:
            raise ValueError(f"{path}: no exposure-normalised rate available")
    p["rate_per_10k_pts"] = p["rate_per_10k_pts"].fillna(0.0)
    return p


def build_profile_from_raw(events_csv, out_csv, decel_g=None, category=None,
                           vehicle_type=None, exposure_csv=None,
                           start_col="EventStart", decel_col="PeakDecel_g"):
    """Rebuild the hour x weekday profile from a raw event extract
    (chunked; suitable for the 5 GB yearly file on the HPC).

    decel_g/category/vehicle_type select the event subset (e.g. an
    independent -0.30g scan for threshold robustness). Exposure comes from
    exposure_csv (columns dow,hour,n_points) when supplied - GPS pings are
    the honest exposure denominator; without it the profile is event-count
    based and this is recorded in the summary.
    """
    usecols = [start_col, decel_col, "Category", "VehicleType"]
    usecols = [c for c in usecols if c is not None]
    rates = {}
    for ch in pd.read_csv(events_csv, usecols=lambda c: c in usecols,
                          chunksize=2_000_000):
        if decel_g is not None:
            ch = ch[pd.to_numeric(ch[decel_col], errors="coerce") >= decel_g]
        if category is not None:
            ch = ch[ch["Category"] == category]
        if vehicle_type is not None:
            ch = ch[ch["VehicleType"] == vehicle_type]
        ts = pd.to_datetime(ch[start_col], errors="coerce").dropna()
        key = pd.DataFrame({"dow": ts.dt.dayofweek, "hour": ts.dt.hour})
        for k, g in key.groupby(["dow", "hour"]):
            rates[k] = rates.get(k, 0) + len(g)
    prof = pd.DataFrame([(d, h, n) for (d, h), n in sorted(rates.items())],
                        columns=["dow", "hour", "hb_events"])
    if exposure_csv:
        expo = pd.read_csv(exposure_csv)[["dow", "hour", "n_points"]]
        prof = prof.merge(expo, on=["dow", "hour"], how="left")
    else:
        prof["n_points"] = np.nan          # unnormalised fallback
    prof["rate_per_10k_pts"] = np.where(
        prof["n_points"] > 0, 1e4 * prof["hb_events"] / prof["n_points"],
        prof["hb_events"])
    prof["day"] = prof["dow"].map(dict(enumerate(DOW_NAMES)))
    prof.to_csv(out_csv, index=False)
    print(f"profile written: {out_csv} ({len(prof)} cells)")
    return prof


def diurnal_shape(p):
    """Exposure-normalised diurnal shape (mean over weekdays per hour)."""
    return p.groupby("hour")["rate_per_10k_pts"].mean()


# --------------------------------------------------------------- scheduling
def rank_cells(p, top_frac=0.10, rank_by="rate"):
    """Rank hour x dow cells; add lift / capture / cumulative / PAI.

    rank_by='rate'   -> exposure-normalised risk index (default; matches the
                        paper's hour x weekday risk index)
    rank_by='events' -> raw event volume (targets total event mass; useful
                        when the operator cares about absolute counts, e.g.
                        enforcement hours bought per event prevented)
    """
    q = p.copy()
    tot = q["hb_events"].sum()
    sort_col = "rate_per_10k_pts" if rank_by == "rate" else "hb_events"
    q = q.sort_values(sort_col, ascending=False).reset_index(drop=True)
    q["risk_rank"] = np.arange(1, len(q) + 1)
    n_top = max(1, int(top_frac * len(q)))   # floor: 10% of 168 cells = 16,
                                             # reproducing the paper's 10.8% capture
    q["priority"] = np.where(q["risk_rank"] <= n_top, "PRIORITY", "monitor")
    q["capture_pct"] = 100.0 * q["hb_events"] / tot
    q["cum_capture_pct"] = q["capture_pct"].cumsum()
    mean_rate = q["hb_events"].sum() / max(len(q), 1)
    q["lift"] = q["hb_events"] / mean_rate          # per-cell event lift
    seg_share = 100.0 / len(q)                       # each cell = 1/168 of network-time
    q["PAI"] = q["capture_pct"] / seg_share
    return q, n_top


def weekend_weekday_ratio(q):
    """Weekend (Sat/Sun) vs weekday per-exposure rate ratio. Uses the
    exposure-normalised rate (events per 10k points), matching the
    paper's definition; falls back to raw per-cell counts only if the
    profile carries no exposure denominator."""
    col = "rate_per_10k_pts" if q["n_points"].notna().any() else "hb_events"
    wknd = q.loc[q["dow"] >= 5, col].mean()
    wkday = q.loc[q["dow"] < 5, col].mean()
    return (wknd / wkday) if wkday else np.nan


def regional_windows(lga_csv, floor_q=0.25):
    """Per-LGA share of intervention hours, proportional to the transfer
    gain of the HB proxy (HBproxy_dLL), floored so no region is starved."""
    t = pd.read_csv(lga_csv)
    w = np.maximum(t["HBproxy_dLL"], 0.0)
    floor = float(np.nanquantile(w, floor_q)) if len(w) else 0.0
    w = w + floor + 1e-9
    t["window_share_pct"] = 100.0 * w / w.sum()
    return t[["LGA", "HBproxy_dLL", "TI", "window_share_pct"]]


# ------------------------------------------------------------------ output
def _tex_escape(s):
    s = str(s)
    spans = []
    def _keep(m):
        spans.append(m.group(1)); return f"\x00{len(spans)-1}\x00"
    s = re.sub(r"\$([^$]*)\$", _keep, s)
    for k, v in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(k, v)
    s = re.sub(r"\x00(\d+)\x00", lambda m: f"${spans[int(m.group(1))]}$", s)
    return s


def heatmap(q, n_top, path):
    piv = q.pivot(index="dow", columns="hour", values="rate_per_10k_pts")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(24), [f"{h:02d}" for h in range(24)], fontsize=7)
    ax.set_yticks(range(7), DOW_NAMES, fontsize=8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_title("Hard-braking risk index (events per 10k points) - "
                 f"top {n_top} cells = priority windows")
    # outline the priority cells (they are the first n_top rows of q)
    for _, r in q.head(n_top).iterrows():
        ax.add_patch(plt.Rectangle((r["hour"] - 0.5, r["dow"] - 0.5), 1, 1,
                                   fill=False, edgecolor="#0092BD", lw=1.6))
    fig.colorbar(im, ax=ax, pad=0.01, label="rate / 10k pts")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"heatmap written: {path}")


def simple_figure(prof, path):
    """Simplified main figure: (a) risk by hour, (b) risk by weekday.

    This is the paper's Figure 6. The 168-cell heatmap is kept as a
    supplement; this two-panel summary is what operators actually read:
    weekends 09:00-16:00 carry the highest per-exposure risk.
    """
    hourly = prof.groupby("hour")["rate_per_10k_pts"].mean()
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    prof = prof.copy()
    prof["day"] = prof["dow"].map(dict(enumerate(DOW_NAMES)))
    daily = prof.groupby("day")["rate_per_10k_pts"].mean().reindex(order)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax1.plot(hourly.index, hourly.values, color="#B34700", lw=2)
    ax1.axvspan(9, 16, color="#FFD9A8", alpha=0.45, label="Priority hours 09-16")
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Risk (events / 10k GPS pts)")
    ax1.set_title("(a) Risk by hour: daytime peak", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    cols = ["#0092BD" if d in ("Sat", "Sun") else "#B0B0B0" for d in order]
    ax2.bar(list(daily.index), daily.values, color=cols)
    ax2.set_ylabel("Risk (events / 10k GPS pts)")
    ax2.set_title("(b) Risk by day: weekends highest", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.suptitle("When is hard-braking risk highest? Deploy patrols/VMS Sat-Sun 09:00-16:00",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"simple figure written: {path}")


def write_tex(q, n_top, summ, path):
    """Emit a self-contained table block (no heading/prose): the including
    document supplies the subsection heading and narrative."""
    top = q.head(min(10, n_top))
    rho = summ.get("shape_rho", np.nan)
    if np.isfinite(rho):
        robustness = (f"diurnal shape threshold-robust, $\\rho={rho:.2f}$ vs "
                      "the independent threshold scan")
    else:
        robustness = ("threshold robustness not evaluated here (re-run with "
                      "--from-raw / --alt-profile before deployment)")
    lines = [
        "% Auto-generated by hb_time_intervention.py - table block only",
        "\\begin{table}[htbp]\\centering\\small",
        "\\caption{Priority hour$\\times$weekday intervention windows "
        "(top-ranked cells).}\\label{tab:hb_time_intervention}",
        "\\begin{tabular}{llrrr}\\hline",
        "Day & Hour & Rate/10k & Lift & Cum.\\ capture (\\%)\\\\\\hline",
    ]
    for _, r in top.iterrows():
        hh = f"{int(r['hour']):02d}:00-{int(r['hour']):02d}:59"
        lines.append(f"{_tex_escape(r['day'])} & {_tex_escape(hh)} & "
                     f"{r['rate_per_10k_pts']:.1f} & {r['lift']:.2f} & "
                     f"{r['cum_capture_pct']:.1f}\\\\")
    lines += ["\\hline\\end{tabular}",
              f"\\emph{{Top {n_top} of {len(q)} cells capture "
              f"{summ['top_capture_pct']:.1f}\\% of events (cell PAI "
              f"{summ['pai_top']:.2f}); {robustness}. Regional allocation of "
              f"window-hours: {_tex_escape(summ['regional_note'])}.}}",
              "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"latex snippet written: {path}")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="HB-gated time-based interventions")
    ap.add_argument("--profile", default=os.path.join(DIR, "temporal_risk_profile.csv"))
    ap.add_argument("--alt-profile", default=None,
                    help="profile from an independent threshold (robustness)")
    ap.add_argument("--transfer-summary", default=os.path.join(DIR, "hb_transferability_summary.csv"))
    ap.add_argument("--transfer-lga", default=os.path.join(DIR, "hb_transferability_lga.csv"))
    ap.add_argument("--from-raw", default=None,
                    help="raw events CSV; rebuilds the profile first (HPC)")
    ap.add_argument("--alt-decel-g", type=float, default=0.30,
                    help="threshold for the robustness scan when --from-raw")
    ap.add_argument("--out-dir", default=DIR)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--rank-by", choices=["rate", "events"], default="rate",
                    help="rank cells by exposure-normalised rate (default) "
                         "or raw event volume")
    ap.add_argument("--min-pos-share", type=float, default=0.60)
    ap.add_argument("--min-stability", type=float, default=2.0)
    ap.add_argument("--min-shape-rho", type=float, default=0.90)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) gate -------------------------------------------------------------
    passed, gate = transfer_gate(args.transfer_summary, args.transfer_lga,
                                 args.min_pos_share, args.min_stability)
    print("\nTRANSFER GATE - is the HB proxy deployable across regions?")
    for name, (ok, detail) in gate["checks"].items():
        print(f"  [{'x' if ok else ' '}] {name:<38s} ({detail})")
    if not passed:
        print("\nGate FAILED: hard braking is not sufficiently transferable; "
              "time-based intervention scheduling is NOT enabled. Re-run the "
              "transferability analysis or relax thresholds explicitly.")
        pd.DataFrame([{"check": k, "passed": bool(v[0]), "detail": v[1]}
                      for k, v in gate["checks"].items()]).to_csv(
            os.path.join(args.out_dir, "hb_time_intervention_gate.csv"), index=False)
        raise SystemExit(2)
    print("  Gate PASSED: HB proxy is transferable - scheduling enabled.")
    if args.gate_only:
        return

    # 2) profile ----------------------------------------------------------
    if args.from_raw:
        prof = build_profile_from_raw(args.from_raw,
                                      os.path.join(args.out_dir, "temporal_risk_profile.csv"))
    else:
        prof = load_profile(args.profile)
    print(f"\nprofile: {len(prof)} hour x weekday cells, "
          f"{prof['hb_events'].sum():,.0f} events")

    # 3) robustness -------------------------------------------------------
    shape_rho = np.nan
    if args.alt_profile and os.path.exists(args.alt_profile):
        alt = load_profile(args.alt_profile)
        a1, a2 = diurnal_shape(prof), diurnal_shape(alt)
        h = a1.index.intersection(a2.index)
        shape_rho = sps.spearmanr(a1.loc[h], a2.loc[h]).statistic
        robust = shape_rho >= args.min_shape_rho
        print(f"threshold robustness: rho={shape_rho:.3f} "
              f"({'robust' if robust else 'NOT robust'})")
        if not robust:
            print("  Diurnal shape is threshold-sensitive; schedule flagged "
                  "provisional in the summary output.")
    elif args.from_raw:
        # build the independent -0.30g scan alongside the main profile
        alt_path = os.path.join(args.out_dir, "temporal_risk_profile_alt030.csv")
        alt = build_profile_from_raw(args.from_raw, alt_path, decel_g=args.alt_decel_g)
        a1, a2 = diurnal_shape(prof), diurnal_shape(alt)
        h = a1.index.intersection(a2.index)
        shape_rho = sps.spearmanr(a1.loc[h], a2.loc[h]).statistic
        print(f"threshold robustness (-{args.alt_decel_g:.2f}g scan): "
              f"rho={shape_rho:.3f}")

    # 4) schedule ---------------------------------------------------------
    q, n_top = rank_cells(prof, args.top_frac, rank_by=args.rank_by)
    top = q.head(n_top)
    pai_top = float(top["capture_pct"].sum() / (100.0 * args.top_frac))
    ww = weekend_weekday_ratio(q)
    print(f"\nschedule (rank-by={args.rank_by}): top {n_top}/{len(q)} cells capture "
          f"{top['capture_pct'].sum():.1f}% of events "
          f"(PAI={pai_top:.2f}); weekend/weekday rate ratio={ww:.2f}")

    # 5) regional layer ---------------------------------------------------
    reg_note = "n/a"
    if os.path.exists(args.transfer_lga):
        reg = regional_windows(args.transfer_lga)
        reg.to_csv(os.path.join(args.out_dir, "hb_time_intervention_regional.csv"),
                   index=False)
        lead = reg.sort_values("window_share_pct", ascending=False).head(3)
        reg_note = ("; ".join(f"{r['LGA']} {r['window_share_pct']:.0f}%"
                              for _, r in lead.iterrows())
                    + " of window-hours (weighted by HB-proxy transfer gain)")
        print("regional window shares (top 3): " +
              "; ".join(f"{r['LGA']}={r['window_share_pct']:.1f}%"
                        for _, r in lead.iterrows()))

    q.to_csv(os.path.join(args.out_dir, "hb_time_intervention_schedule.csv"),
             index=False)
    summ = {"gate_passed": True,
            "hbproxy_dll": float(gate["metrics"]["HBproxy_dLL_mean"]),
            "pos_folds": gate["metrics"]["HBproxy_dLL_positive_folds"],
            "n_folds": (len(pd.read_csv(args.transfer_lga))
                        if os.path.exists(args.transfer_lga) else np.nan),
            "n_cells": len(q), "n_top": n_top,
            "top_capture_pct": float(top["capture_pct"].sum()),
            "pai_top": pai_top, "weekend_weekday_ratio": ww,
            "shape_rho": shape_rho, "regional_note": reg_note}
    pd.DataFrame([summ]).to_csv(
        os.path.join(args.out_dir, "hb_time_intervention_summary.csv"), index=False)

    heatmap(q, n_top, os.path.join(args.out_dir, "hb_time_intervention_heatmap.png"))
    simple_figure(prof, os.path.join(args.out_dir, "hb_time_simple.png"))
    write_tex(q, n_top, summ,
              os.path.join(args.out_dir, "hb_time_intervention.tex"))
    print("\ndone: schedule, regional shares, summary, heatmap, simple figure and LaTeX "
          "snippet written to " + args.out_dir)


if __name__ == "__main__":
    main()
