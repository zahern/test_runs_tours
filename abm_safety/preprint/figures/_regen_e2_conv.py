"""Rebuild the E2 (GA-searched) Stage-5 convergence figure from the run artifacts.

Source: runs/exampville_s5fix_20261030 (cluster job 27338231)
        checkpoints/convergence_trace.csv   timestamped incumbent improvements
        exampville_compare.out              gen-0 incumbent, "New best ... Gen N"
                                           lines, and every-5th-gen summaries

What the artifacts do and do not contain
----------------------------------------
* `GA_CHECKPOINT_EVERY=5` fired only at generation 0 for every Stage-5
  searcher, so `ga_history_partial_Stage5_*.csv` hold a header row only and
  `plots/ga_partial_Stage5_*.png` are blank gen-0 frames. The in-RAM history
  was never persisted at a later generation.
* The trace records each accepted incumbent in time order with its fitness but
  **no generation column**; the log supplies the generation for each improvement.
* The search's generation counter is per searcher pass and restarts from 0
  whenever the population is reseeded (main run -> declared restart -> the
  re-search pass that reuses the fitness cache). The trace records those passes
  back to back, so the printed generations are not monotone and no offset that
  would put them on one axis is recorded anywhere in the run directory.

The x-axis is therefore the chronological order of accepted incumbents (0 = the
gen-0 population's best, then each strict improvement), which needs no
reconstruction, and every marker is annotated with the generation the search
itself printed for it. Duplicate re-logged ties are collapsed.

Run:  python _regen_e2_conv.py
"""
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "..", "..", "code", "runs",
                   "exampville_s5fix_20261030")
LOG = os.path.join(RUN, "exampville_compare.out")
TRACE = os.path.join(RUN, "checkpoints", "convergence_trace.csv")

# (trace/log label, panel title, gen-0 incumbent BIC/n from the log)
COMPONENTS = [
    ("Stage5 Work (tour) Mode", "Work · tour mode", 1.2728),
    ("Stage5 Non-Mandatory (tour) Mode", "Non-mandatory · tour mode", 1.0635),
    ("Stage5 Work (tour) Dest", "Work · tour destination", 5.8345),
    ("Stage5 Non-Mandatory (tour) Dest", "Non-mandatory · tour destination", 6.1495),
]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "figure.dpi": 150})


def trace_levels():
    """{label: [distinct BIC/n levels in time order]} (ties collapsed)."""
    out = {}
    with open(TRACE, newline="") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            if not label.startswith("Stage5"):
                continue
            bic = -float(row["fitness"])
            seq = out.setdefault(label, [])
            if not seq or abs(seq[-1] - bic) > 5e-7:
                seq.append(bic)
    return out


def logged_generations():
    """{label: [(gen, BIC/n as printed, 4 d.p.)]} in log order."""
    pat = re.compile(r"Gen\s+(\d+) \|.*New best \[(Stage5 [^\]]+)\].*?BIC/n=([\d.]+)")
    out = {}
    for line in open(LOG, errors="ignore"):
        m = pat.search(line)
        if m:
            out.setdefault(m.group(2), []).append((int(m.group(1)),
                                                   round(float(m.group(3)), 4)))
    return out


def main():
    levels = trace_levels()
    gens = logged_generations()

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    rows_out = []
    for ax, (label, title, base) in zip(axes.ravel(), COMPONENTS):
        seq = levels.get(label, [])
        vals = [base] + [v for v in seq if v < base]
        x = list(range(len(vals)))

        glist = gens.get(label, [])
        used = set()
        annotations = []
        for i, v in enumerate(vals):
            if i == 0:
                annotations.append("g0")
                continue
            g = None
            for gi, (gen, gval) in enumerate(glist):
                if gi not in used and abs(round(v, 4) - gval) < 1e-9:
                    g, used = gen, used | {gi}
                    break
            annotations.append(f"g{g}" if g is not None else "?")

        ax.plot(x, vals, "-o", color="#534AB7", lw=1.9, ms=5.5, zorder=4,
                label="accepted incumbent")
        ax.axhline(base, color="#999999", ls=":", lw=1.2, zorder=2,
                   label="gen-0 population best")
        for xi, v, a in zip(x, vals, annotations):
            ax.annotate(a, (xi, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color="#444444")

        first, last = vals[0], vals[-1]
        span = first - last
        pad = max(abs(span) * 0.55, 4e-4)
        ax.set_ylim(last - pad * 0.45, first + pad)
        ax.set_xlim(-0.5, len(vals) - 0.5)
        # A ~1e-3 range makes matplotlib print an offset ("+1.063") that
        # collides with the panel title; spell the values out instead.
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_formatter(
            mticker.FormatStrFormatter("%.4f" if abs(first) < 3 else "%.3f"))
        ax.set_xticks(x)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.3, axis="y")

        if span > 0:
            note = (f"{first:.4f} → {last:.4f}   "
                    f"(−{span:.4f}, {100 * span / first:.2f}%)")
        else:
            note = (f"{first:.4f} — no strict improvement after gen 0 "
                    f"(all later incumbents tie to 6 d.p.)")
        ax.text(0.98, 0.06, note, transform=ax.transAxes, ha="right",
                fontsize=7.8, color="#2b2b2b")
        rows_out.append((label, first, last, span, len(vals) - 1))

    for ax in axes[1]:
        ax.set_xlabel("accepted incumbent (chronological, 0 = gen-0 best)")
    for ax in axes[:, 0]:
        ax.set_ylabel("BIC/n   (lower = better)")

    axes[0][0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("E2 GA-searched Stage-5 specifications, Exampville — "
                 "job 27338231 (pop 12, 10+10 generations, 1 restart per segment)",
                 fontsize=10.5)
    fig.text(0.5, 0.005,
             "markers labelled with the generation the search printed for that "
             "incumbent; the counter restarts from 0 when the population is "
             "reseeding, so labels are not monotone",
             ha="center", fontsize=7.6, color="#555555", style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(HERE, "e2_stage5_convergence.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print("wrote", out)
    for label, first, last, span, n in rows_out:
        print(f"  {label:38s} {first:.4f} -> {last:.4f}  "
              f"(-{span:.4f})  {n} strict improvements")


if __name__ == "__main__":
    main()
