#!/usr/bin/env python
"""
make_figures.py — poster figures for the SEQ 371 hybrid ABM+SD model.
================================================================================
Generates (into ./figs, PDF + PNG):
  fig_sd_stocks        — SD stock trajectories 2025-2045 (speed factor,
                         EV share, fleet cost multiplier) per scenario
  fig_sd_flow          — hybrid ABM<->SD loop with the I1-I4 interfaces
  fig_ga_status        — GA search status per tag/stage/segment from
                         _final_matrix.csv (converged generation vs budget,
                         coloured by outcome)
  fig_acc_propagation  — accessibility outcome: behavioural (asim_*) vs
                         gravity (auto_*) zone accessibility, 371 SA2s

Pure local run (needs matplotlib/pandas only):
  python make_figures.py
Inputs (read-only):
  <code>/saved_data/luti/scenario_travel.csv
  <code>/_final_matrix.csv
  <code>/saved_data/accessibility_371.csv
Outputs also include figs/sd_stocks.csv (provenance copy).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

sys.path.insert(0, CODE)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})

_SCEN_COLORS = {"BASE": "#212121", "TELE": "#1565c0", "MBUF": "#ef6c00"}
_RES_COLORS = {
    "VALID": "#2e7d32", "UNCONVERGED": "#ef6c00", "FAILED": "#c62828",
    "NO_PKL": "#9e9e9e", "NO_GA_EVID": "#bdbdbd", "NO_DIR": "#e0e0e0",
}


def _save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(FIGS, f"{name}.{ext}")
        fig.savefig(p)
        print(f"  wrote {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig_sd_stocks():
    """SD layer outputs: congestion speed factor, EV Bass diffusion, cost."""
    import sd_hybrid_371 as sd

    travel = pd.read_csv(os.path.join(CODE, "saved_data", "luti",
                                      "scenario_travel.csv"))
    stocks = sd.run_sd(travel)
    stocks.to_csv(os.path.join(FIGS, "sd_stocks.csv"), index=False)
    sd.interface_table().to_csv(os.path.join(FIGS, "sd_interfaces.csv"),
                                index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    panels = [
        ("speed_factor", "Network speed factor\n(vs base 2025)", "%.3f",
         lambda v: v),
        ("ev_share", "EV share of fleet (%)", "%.1f%%",
         lambda v: 100.0 * v),
        ("cost_mult", "Fleet fuel cost multiplier\n(vs base 2025)", "%.3f",
         lambda v: v),
    ]
    _offsets = {
        "speed_factor": {"TELE": (5, 3), "BASE": (5, -13), "MBUF": (5, 3)},
        "ev_share": {"TELE": (5, 3), "BASE": (-42, -12), "MBUF": (5, -12)},
        "cost_mult": {"TELE": (-46, -3), "BASE": (6, -3), "MBUF": (6, 8)},
    }
    for ax, (col, title, fmt, tf) in zip(axes, panels):
        for scen, g in stocks.groupby("scenario"):
            g = g.sort_values("year")
            ax.plot(g["year"], tf(g[col]), marker="o", ms=4,
                    color=_SCEN_COLORS.get(scen, None), label=scen)
            _last = g.iloc[-1]
            ax.annotate(fmt % tf(_last[col]),
                        (int(_last["year"]), tf(_last[col])),
                        textcoords="offset points",
                        xytext=_offsets[col].get(scen, (4, 2)),
                        fontsize=9, color=_SCEN_COLORS.get(scen, None))
        ax.set_title(title)
        ax.set_xlabel("year")
        ax.set_xlim(2023, 2050)
        ax.margins(y=0.13)
        ax.grid(alpha=0.3)
        if col == "ev_share":
            ax.set_ylim(0, 105)
    axes[0].set_ylabel("factor / share")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Hybrid SD layer — stocks driven by ABM VMT (slow 5-yr "
                 "horizon), feeding back into next-horizon travel costs",
                 y=1.06, fontsize=12)
    _save(fig, "fig_sd_stocks")


# ─────────────────────────────────────────────────────────────────────────────
def fig_sd_flow():
    """Hybrid ABM<->SD loop, standalone flow diagram for the poster."""
    import sd_hybrid_371 as sd

    p = sd.draw_flow_diagram(os.path.join(FIGS, "fig_sd_flow.png"), title=False)
    if p:
        print(f"  wrote {p}")


# ─────────────────────────────────────────────────────────────────────────────
def fig_ga_status():
    """GA search status per tag:stage:segment (converged gen vs budget)."""
    fm = pd.read_csv(os.path.join(CODE, "_final_matrix.csv"),
                     encoding="utf-8-sig")
    fm = fm.rename(columns={c: c.strip() for c in fm.columns})
    for col in ("MaxGen", "Need", "LastStag"):
        fm[col] = pd.to_numeric(fm[col], errors="coerce")
    d = fm[fm["Model"] != "NO"].copy()          # drop NO_DIR placeholder rows
    d = d.dropna(subset=["MaxGen", "LastStag"])
    d = d[d["MaxGen"] > 0]
    if d.empty:
        print("  fig_ga_status: no completed rows with MaxGen>0 - skipping")
        return

    _stage_order = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "51": 6}
    d["_so"] = d["Stage"].astype(str).map(_stage_order).fillna(9)
    d["label"] = (d["Tag"].astype(str) + "  S" + d["Stage"].astype(str)
                  + ":" + d["Seg"].astype(str))
    d = d.sort_values(["_so", "Seg"]).reset_index(drop=True)

    left = d[d["Tag"].astype(str) == "qld_conv1"].reset_index(drop=True)
    right = d[d["Tag"].astype(str) != "qld_conv1"].reset_index(drop=True)
    left_a = left[left["_so"] <= 3].reset_index(drop=True)
    left_b = left[left["_so"] > 3].reset_index(drop=True)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_RES_COLORS.get(k, "#607d8b"), edgecolor="none",
                     label=f"{k} ({v})")
               for k, v in d["Model"].astype(str).value_counts().items()]
    handles += [Patch(facecolor="#eeeeee", edgecolor="#bdbdbd", label="budget"),
                plt.Line2D([], [], color="#2e7d32", ls="--",
                           label="Need (per fit)")]

    panels = [
        (left_a, "qld_conv1 stages 0-3", None),
        (left_b, "qld_conv1 stages 4-51", None),
        (right, "qld_fix1 + qld_full40", 50.0),
    ]
    nrow = max(len(p[0]) for p in panels)
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 0.26 * nrow + 1.9), sharey=False)

    def _panel(ax, sub, title, xmax=None):
        y = np.arange(len(sub))
        ax.barh(y, sub["MaxGen"], color="#eeeeee", edgecolor="#bdbdbd",
                height=0.70, zorder=1)
        colors = [_RES_COLORS.get(str(r), "#607d8b") for r in sub["Model"]]
        ax.barh(y, sub["LastStag"], color=colors, height=0.70, zorder=2)
        for yi, nd in zip(y, pd.to_numeric(sub["Need"], errors="coerce")):
            if pd.notna(nd):
                ax.plot([nd, nd], [yi - 0.35, yi + 0.35], color="#2e7d32",
                        ls="--", lw=1.1, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("generation")
        ax.set_title(f"{title} ({len(sub)} fits)", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        if xmax:
            ax.set_xlim(0, xmax)

    for ax, (sub, title, xmax) in zip(axes, panels):
        _panel(ax, sub, title, xmax)
    axes[-1].legend(handles=handles, frameon=False, fontsize=9,
                    loc="lower right")
    fig.suptitle("GA search status (pre-PT stage-5 runs): bar = budget, "
                 "fill = converged generation, colour = merit outcome",
                 y=0.99, fontsize=12)
    _save(fig, "fig_ga_status")


# ─────────────────────────────────────────────────────────────────────────────
def fig_acc_propagation():
    """Accessibility outcome: behavioural (asim_*) vs gravity (auto_*)."""
    acc = pd.read_csv(os.path.join(CODE, "saved_data", "accessibility_371.csv"))
    pairs = [
        ("work", "asim_work_acc", "auto_work_acc"),
        ("edu", "asim_edu_acc", "auto_edu_acc"),
        ("shop", "asim_shop_acc", "auto_shop_acc"),
        ("nonmand", "asim_nonmand_acc", "auto_nonmand_acc"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))

    x = pd.to_numeric(acc["auto_work_acc"], errors="coerce")
    yy = pd.to_numeric(acc["asim_work_acc"], errors="coerce")
    m = x.notna() & yy.notna()
    r_p = float(np.corrcoef(x[m], yy[m])[0, 1])
    ax1.scatter(x[m], yy[m], s=14, alpha=0.55, color="#1565c0",
                edgecolor="none")
    ax1.set_xlabel("gravity accessibility  auto_work_acc")
    ax1.set_ylabel("behavioural logsum  asim_work_acc")
    ax1.set_title(f"Work accessibility, 371 SA2 zones (Pearson r = {r_p:.2f})")
    ax1.grid(alpha=0.3)

    names, rhos = [], []
    for name, a_col, g_col in pairs:
        a = pd.to_numeric(acc[a_col], errors="coerce")
        g = pd.to_numeric(acc[g_col], errors="coerce")
        mm = a.notna() & g.notna()
        rho = float(np.corrcoef(a[mm].rank(), g[mm].rank())[0, 1])
        names.append(f"{name}\n({a_col})")
        rhos.append(rho)
    ax2.bar(names, rhos, color="#2e7d32", alpha=0.85)
    for i, v in enumerate(rhos):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Spearman rank correlation")
    ax2.set_title("Behavioural vs gravity accessibility\n"
                  "(unified logsum vs attractor-decay proxy)")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Accessibility outcome propagation across the survey zone "
                 "system", y=1.04, fontsize=12)
    _save(fig, "fig_acc_propagation")


def main():
    print("Poster figures ->", FIGS)
    fig_sd_stocks()
    fig_sd_flow()
    fig_ga_status()
    fig_acc_propagation()
    print("done")


if __name__ == "__main__":
    main()
