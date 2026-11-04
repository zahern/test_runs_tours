"""One-off: regenerate game_day_pred_vs_actual.png from the holdout cells
CSV with the rank-lift design (hexbin + cumulative capture curve).
Disposable -- not part of the pipeline.
Usage: python regen_pred_vs_actual_fig.py <cells.csv.gz> <out.png>
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _rank(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def main():
    cells_path, out_png = sys.argv[1], sys.argv[2]
    df = pd.read_csv(cells_path)
    df["observed"] = pd.to_numeric(df["observed"], errors="coerce").fillna(0.0)
    df["pred_speeding"] = pd.to_numeric(df["pred_speeding"],
                                         errors="coerce").fillna(0.0)
    fx_order = [str(f) for f in df["fixture"].astype(str).unique()]
    panels = [("ALL GAMES", df)]
    panels += [(f, df[df["fixture"].astype(str) == f]) for f in fx_order]
    ncol = 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow),
                             squeeze=False)
    for i, (label, frame) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        o = pd.to_numeric(frame["observed"], errors="coerce").fillna(0.0)
        o = o.to_numpy(float)
        p = pd.to_numeric(frame["pred_speeding"], errors="coerce").fillna(0.0)
        p = p.to_numpy(float)
        m = np.isfinite(o) & np.isfinite(p)
        o, p = o[m], p[m]
        if len(o) < 8:
            ax.axis("off")
            continue
        plim = max(float(np.percentile(p, 99)), 1.0) * 1.05
        olim = max(float(o.max()), 1.0) * 1.05
        hb = ax.hexbin(p, o, gridsize=42, extent=(0, plim, 0, olim),
                       mincnt=1, cmap="viridis", norm=LogNorm())
        ax.plot([0, olim], [0, olim], ls="--", lw=1.0, color="#bbbbbb",
                zorder=3, label="1:1")
        ax.set_xlim(0, plim)
        ax.set_ylim(0, olim)
        ax.set_xlabel("PREDICTED finable speeding (count per cell)",
                      fontsize=7.5)
        ax.set_ylabel("OBSERVED finable speeding (count)", fontsize=7.5)
        cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("segment x bin cells", fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
        tot = max(float(o.sum()), 1e-9)
        ys = np.cumsum(o[np.argsort(-p, kind="stable")]) / tot
        xs = (np.arange(len(o)) + 1) / float(len(o))
        i25 = max(int(round(0.25 * len(ys))) - 1, 0)
        q1 = float(ys[i25])
        ax2 = ax.twinx()
        ax2.plot(xs, ys, color="#1f77b4", lw=1.8, alpha=0.95, zorder=4)
        ax2.plot([0.25], [q1], "o", color="#1f77b4", ms=5, zorder=5)
        ax2.annotate(f"Q1 -> {100 * q1:.0f}%", (0.25, q1),
                     textcoords="offset points", xytext=(6, -11),
                     fontsize=6.5, color="#1f77b4")
        ax2.set_ylim(0.0, 1.02)
        ax2.set_ylabel("share of observed captured", fontsize=6.5)
        ax2.tick_params(labelsize=5.5, colors="#1f77b4")
        rho = _rank(o, p)
        ax.set_title(
            f"{label[:44]}\nrho={rho:.2f}  Q1 holds {100 * q1:.1f}% of "
            f"observed  (cells={len(o):,}, {int((o > 0).sum()):,} nonzero)",
            fontsize=7.5)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=6, loc="upper left")
    fig.suptitle(
        "Full-coverage proof: predicted COUNT vs observed COUNT, every "
        "segment x 15-min cell in the holdout\n"
        "(shade = cell count, log scale; dashed = 1:1; blue curve = "
        "observed-event share captured as cells are ranked by prediction; "
        "Q1 = top quartile, 25% = chance)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
