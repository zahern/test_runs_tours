"""
44_duration_survival.py - survival (hazard) view of finable-speeding timing.

Companion to 43: where 43 models HOW MANY and HOW FAR OVER, this models
WHEN - inter-arrival durations between consecutive finable exceedances on
the same segment. Nonparametric Kaplan-Meier + Weibull hazard shape
(c<1 bursty/decreasing hazard; c>1 regular; c=1 memoryless) plus exact
piecewise-exponential rates per fixture phase (pre/during/post) and a
rank comparison of in-window vs background gaps.

Inputs: results/finable_exceedances.csv.gz (road_osmid, dt, exceedance,
        fixture, phase) from job 13.
Outputs: results/duration_survival.tex, results/duration_survival_audit.json,
         figures/survival_km.png

Usage: python 44_duration_survival.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJ = "/mnt/hpccs01/home/ahernz/test_runs_tours/demand_/suncorp_telematics"
DATA = os.path.join(PROJ, "data")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")

EXC = os.path.join(RES, "finable_exceedances.csv.gz")
CAL = os.path.join(DATA, "fixture_calendar.csv")
OUT_TEX = os.path.join(RES, "duration_survival.tex")
OUT_JSON = os.path.join(RES, "duration_survival_audit.json")
FIG_KM = os.path.join(FIG, "survival_km.png")

sns.set_style("whitegrid")


def km_curve(t):
    """Kaplan-Meier step curve for uncensored gap sample t (minutes)."""
    t = np.sort(np.asarray(t, dtype=float))
    n = len(t)
    tt = np.concatenate([[0.0], t])
    ss = np.concatenate([[1.0], 1.0 - np.arange(1, n + 1) / n])
    return tt, ss


def weibull_mle(t):
    from scipy.stats import weibull_min
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    if len(t) < 50:
        return None
    c, _, s = weibull_min.fit(t, floc=0)
    return {"c": float(c), "scale_min": float(s), "n": int(len(t)),
            "median_min": float(np.median(t))}


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    audit = {}
    print("=" * 80)
    print("44_duration_survival.py - inter-arrival survival of finable speeding")
    print("=" * 80)

    print("\n[1/4] Loading exceedance times...")
    ex = pd.read_csv(EXC, parse_dates=["dt"])
    ex = ex.sort_values(["road_osmid", "dt"]).reset_index(drop=True)
    ex["gap_min"] = ex.groupby("road_osmid")["dt"].diff().dt.total_seconds() / 60.0
    gaps = ex.dropna(subset=["gap_min"]).copy()
    gaps["in_window"] = gaps["phase"].fillna("") != ""
    print(f"   exceedances={len(ex):,} gaps={len(gaps):,} "
          f"in-window={int(gaps['in_window'].sum()):,}")
    audit.update({"n_exceed": int(len(ex)), "n_gaps": int(len(gaps)),
                  "n_in_window": int(gaps["in_window"].sum())})

    print("\n[2/4] KM + Weibull hazard shape...")
    res = {}
    gw = gaps.loc[gaps["in_window"], "gap_min"].values
    gb = gaps.loc[~gaps["in_window"], "gap_min"].values
    res["in_window"] = weibull_mle(gw)
    res["background"] = weibull_mle(gb)
    for k, v in res.items():
        print(f"   {k}: {v}")
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(gw, gb, alternative="two-sided")
        res["mwu_p"] = float(p)
        print(f"   rank gap in-window vs background: p={p:.3g}")
    except Exception as e:  # noqa: BLE001
        res["mwu_p"] = None
        print(f"   rank test skipped ({e})")
    audit["weibull"] = res

    print("\n[3/4] Piecewise-exponential phase rates...")
    cal = pd.read_csv(CAL)
    cal["ko"] = pd.to_datetime(cal["kickoff_iso"])
    durs = {"pre": 2.5, "during": 2.0, "post": 2.5}  # hours per fixture
    NF = len(cal)
    ph = ex[ex["phase"].fillna("") != ""].groupby("phase").size()
    rates = {}
    for p, h in durs.items():
        n = int(ph.get(p, 0))
        lam = n / (NF * h)  # events per hour across network
        rates[p] = {"n": n, "per_hour": lam,
                    "mean_gap_min": 60.0 / lam if lam > 0 else float("inf")}
        print(f"   {p}: n={n:,} rate={lam:.1f}/h mean gap={rates[p]['mean_gap_min']:.1f} min")
    audit["phase_rates"] = rates

    print("\n[4/4] Figure + LaTeX...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, col, gg in (("in-window", "#0B6E4F", gw), ("background", "#888888", gb)):
        tt, ss = km_curve(gg)
        ax.step(tt, ss, where="post", label=f"{k} (n={len(gg):,})", color=col, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Inter-arrival gap (minutes, log scale)")
    ax.set_ylabel("Survival P(gap > t)")
    med_w = np.median(gw) if len(gw) else float("nan")
    ax.axvline(med_w, color="#0B6E4F", ls="--", lw=1)
    ax.set_title("Finable-speeding inter-arrivals: in-window vs background")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_KM, dpi=150)
    plt.close(fig)

    def wrow(v):
        if not v:
            return "-- & -- & --"
        return f"{v['c']:.2f} & {v['scale_min']:.0f} & {v['median_min']:.0f}"

    L = []
    L.append("% duration_survival.tex - auto-generated by 44_duration_survival.py")
    L.append("\\begin{table}[htbp]")
    L.append("\\centering")
    L.append("\\caption{\\textcolor{Highlight}{Inter-arrival survival of finable speeding.}"
             " Weibull shape $c<1$ = bursty (decreasing hazard); phase rates are exact counts.}")
    L.append("\\label{tab:survival}")
    L.append("\\begin{tabular}{lrrr}")
    L.append("\\hline")
    L.append("Sample & Weibull $c$ & Scale (min) & Median gap (min) \\\\")
    L.append("\\hline")
    L.append("In fixture window & " + wrow(res["in_window"]) + " \\\\")
    L.append("Background & " + wrow(res["background"]) + " \\\\")
    L.append("\\hline")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    L.append("")
    L.append("Piecewise-exponential network rates: " + "; ".join(
        f"{p} {rates[p]['per_hour']:.1f}/h" for p in ["pre", "during", "post"]) + ".")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(audit, f, indent=2)
    print("DONE! Survival complete.")


if __name__ == "__main__":
    main()
