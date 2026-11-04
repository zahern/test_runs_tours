"""Regenerate the two Stage-5 GA figures from the job stdout that produced them.

Sources (QUT HPC tree):
  code/ga_s5edu_qld_shard2.o25559092   edu destination, 30 gens, RUN_TAG=qld_shard2
  code/ga_s5edu_qld_shard3.o25679062   edu destination, 30 gens, RUN_TAG=qld_shard3
  code/ga_s5edu_qld_full40.o26520868   edu destination, 40 gens + 1x13-gen restart, RUN_TAG=qld_full40
  code/ga_s5nonmand_qld_shard3.o25679063  non-mandatory mode + dest, OOM-killed (exit 137)

The GA logs a compact summary line every fifth generation

    Gen   5 | best_BIC/n=1.7894 mean_BIC/n=1.7931 n_feat=5 div=0.25 ...

and a "New best" line at every generation that improves the incumbent.  The
previous version of these figures kept only the last surviving summary line,
which is why the non-mandatory mode panel showed a single dot.  This script
parses both channels: the summary lines give mean BIC/n, the new-best lines
give the generation at which the best-so-far level changed.

Run:  python _regen_ga_figs.py
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = "//hpc-fs.qut.edu.au/ahernz/test_runs_tours/code"
HERE = os.path.dirname(os.path.abspath(__file__))

SUMMARY = re.compile(
    r"Gen\s+(\d+) \| best_BIC/n=([\d.]+) mean_BIC/n=([\d.]+) n_feat=(\d+)")
NEWBEST = re.compile(r"Gen\s+(\d+) \|.*New best.*?BIC/n=([\d.]+)")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})


def sections(path):
    """Split a GA stdout into its per-model main runs (mode first, dest second)."""
    lines = open(path, errors="ignore").read().splitlines()
    starts = [i for i, l in enumerate(lines)
              if "Main run:" in l and "generations" in l]
    spans = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(lines)
        summaries = [(int(m.group(1)), float(m.group(2)), float(m.group(3)),
                      int(m.group(4)))
                     for m in (SUMMARY.search(l) for l in lines[s:e]) if m]
        newbest = [(int(m.group(1)), float(m.group(2)))
                   for l in lines[s:e] for m in [NEWBEST.search(l)] if m]
        spans.append({"summaries": summaries, "newbest": newbest, "text": lines[s]})
    return spans


def best_so_far(summaries, newbest, n_gens):
    """Step series of best-so-far BIC/n over generations 0..n_gens-1."""
    base = summaries[0][1] if summaries else None
    if base is None:
        return [], []
    levels = {0: base}
    for g, v in newbest + [(g, b) for g, b, _m, _f in summaries]:
        levels[g] = min(levels.get(g, v), v)
    gens, vals = [], []
    cur = base
    for g in range(n_gens):
        cur = min(cur, levels.get(g, cur))
        gens.append(g)
        vals.append(cur)
    return gens, vals


def fig_champion():
    runs = [
        ("ga_s5edu_qld_shard2.o25559092", 1, 30, "A",
         "A (qld\\_shard2, 30 gens)", "-", "tab:blue", "o"),
        ("ga_s5edu_qld_shard3.o25679062", 1, 30, "B",
         "B (qld\\_shard3, 30 gens, rerun, bit-identical to A)", "--", "tab:orange", "s"),
        ("ga_s5edu_qld_full40.o26520868", 1, 53, "C",
         "C (qld\\_full40, 40 gens + 13-gen restart)", "-.", "tab:green", "D"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    finals = {}
    for path, idx, n_gens, tag, label, style, colour, marker in runs:
        sec = sections(os.path.join(CODE, path))[idx]
        gens, vals = best_so_far(sec["summaries"], sec["newbest"], n_gens)
        # thin markers so the dense step series stays legible
        ax.plot(gens, vals, style, color=colour, lw=1.8, label=label,
                marker=marker, markevery=4, markersize=4)
        finals[tag] = vals[-1]
        # A and B are bit-identical: label the pair once instead of overprinting
        label_end = {"A": f"A $\\equiv$ B: {vals[-1]:.4f}", "B": None}
        text = label_end.get(tag, f"{tag}: {vals[-1]:.4f}")
        if text is None:
            continue
        ax.annotate(text, (gens[-1], vals[-1]),
                    textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=9, color=colour)
    ax.set_xlabel("generation")
    ax.set_ylabel("best BIC/n so far (lower = better)")
    ax.set_title("Stage-5 education destination: best BIC/n per independent search run")
    ax.set_xlim(-1, 62)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "champ_stage_5_edu_destination.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, finals)


def fig_salvaged():
    secs = sections(os.path.join(CODE, "ga_s5nonmand_qld_shard3.o25679063"))
    mode = secs[0]["summaries"]
    dest = secs[1]["summaries"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("non-mandatory qld_shard3: salvaged GA summaries "
                 "(job OOM-killed, exit 137)")

    for col, (data, name) in enumerate([(mode, "mode"), (dest, "destination")]):
        g = [r[0] for r in data]
        best = [r[1] for r in data]
        mean = [r[2] for r in data]
        nfeat = [r[3] for r in data]

        ax = axes[0][col]
        ax.plot(g, best, "-o", color="tab:blue", lw=1.8, ms=5, label="best so far")
        ax.plot(g, mean, "-s", color="tab:orange", lw=1.4, ms=4, label="mean")
        ax.set_title(f"non-mandatory {name} search (every 5th generation)")
        ax.set_ylabel("BIC/n (lower = better)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax2 = axes[1][col]
        ax2.plot(g, nfeat, "-o", color="tab:green", lw=1.8, ms=5)
        ax2.set_title(f"non-mandatory {name}: features in the incumbent")
        ax2.set_xlabel("generation")
        ax2.set_ylabel("n features (best)")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "ga_nonmand_salvaged.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_champion()
    fig_salvaged()
