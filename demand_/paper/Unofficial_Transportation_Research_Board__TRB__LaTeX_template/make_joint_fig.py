"""Build joint_hb_crash.png from joint_hb_crash.csv + joint_linkage.csv.

Local-only figure script (matplotlib, no jax/mcr needed). Re-run any time
the joint job refreshes its CSVs. See run_joint_hb_crash.pbs for the HPC job.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    jd = pd.read_csv(os.path.join(HERE, "joint_hb_crash.csv"))
    lk = pd.read_csv(os.path.join(HERE, "joint_linkage.csv")).iloc[0]

    order = ["Frank", "Gaussian", "Kimeldorf--Sampson", "Joe",
             "Gaussian-Joe mixture"]
    jd["model"] = pd.Categorical(jd["model"], categories=order, ordered=True)
    jd = jd.sort_values("model").reset_index(drop=True)
    short = ["Frank", "Gaussian", "Kimeldorf--\nSampson", "Joe", "Gauss--Joe\nmixture"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # A: dBIC vs independence (symlog: single families are millions-scale)
    colors = ["#c0392b"] * 4 + ["#1e8449"]
    bars = ax1.bar(short, jd["dBIC"].values, color=colors)
    ax1.axhline(0, color="k", linewidth=0.8)
    ax1.set_yscale("symlog", linthresh=20000)
    ax1.set_ylabel(r"$\Delta$BIC vs.\ independence (symlog)")
    ax1.set_title("(a) Joint HB--crash copula fit by family")
    for b, v, n in zip(bars, jd["dBIC"].values, jd["nclip"].fillna(0).values):
        lab = f"{v:+,.0f}"
        if np.isfinite(n) and n > 0:
            lab += f"\n({int(n):,} floored)"
        ax1.text(b.get_x() + b.get_width() / 2, v + np.sign(v or 1) * 40000,
                 lab, ha="center", va="bottom" if v > 0 else "top", fontsize=8)

    # B: mixture anatomy + residual dependence
    ax2.axis("off")
    txt = (f"Best linkage: {lk['best_copula']}\n"
           f"theta: {lk['theta']}\n"
           f"dBIC = {lk['dBIC']:+,.1f}   LR = {lk['lr']:,.1f}   p = {lk['lr_p']}\n"
           f"residual Spearman rho = {lk['resid_spearman']:.3f} "
           f"(p = {lk['resid_p']:.2g}, n = {int(lk['n']):,})\n\n"
           "Reading: single families collapse under rectangle\n"
           "underflow (thousands of floored evals); only the\n"
           "Gaussian--Joe mixture (w = 0.91 on Gaussian) survives.\n"
           "Rectangle LRs provisional pending the likelihood-\n"
           "identity check (run_copula_identity.pbs).")
    ax2.text(0.05, 0.95, txt, transform=ax2.transAxes, va="top", fontsize=10,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#aaa"))
    ax2.set_title("(b) Linkage verdict")

    fig.suptitle("Joint HB--crash dependence (best-spec margins, N=10,780, "
                 "VMT-years offset)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "joint_hb_crash.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
