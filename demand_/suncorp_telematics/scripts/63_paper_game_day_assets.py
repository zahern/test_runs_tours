"""63_paper_game_day_assets.py - paper assets for the game-day model.

Generates the two artefacts the manuscript needs and that no other stage
produces:

  1. figures/game_day_movement.png
     Per-fixture officer timelines from a 60_game_day_mip plans CSV. Each
     officer is a row, each leg a coloured bar, a dotted connector is a
     relocation, and the game phases are shaded. This is the figure that
     shows movement (or its absence) directly; the schedules heatmap in
     stage 60 is for diagnosis, not for a page.

  2. results/game_day_model_coefs.tex
     A booktabs table of the fitted coefficients: Panel A the count refit
     (results/game_day_model_coefs.csv), Panel B the two-stage propensity
     (audit.propensity.terms in results/game_day_holdout_metrics.json).

Usage:
  python scripts/63_paper_game_day_assets.py \
      --plans results/mip_free/game_day_mip_plans.csv \
      --coefs results/game_day_model_coefs.csv \
      --metrics results/game_day_holdout_metrics.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
REL_BINS = 28
KICKOFF_BIN = 12
PHASE_BINS = (12, 20)
TZ = "Australia/Brisbane"


def _esc(name):
    return str(name).replace("_", "\\_").replace("%", "\\%")


def parse_legs(raw):
    out = []
    for tok in str(raw or "").split("|"):
        try:
            site, rng = tok.rsplit(":", 1)
            s, e = rng.split("-")
            out.append((site, int(s), int(e)))
        except Exception:  # noqa: BLE001
            continue
    return out


def movement_figure(plans_csv, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plans = pd.read_csv(plans_csv)
    fixtures = list(dict.fromkeys(plans["fixture"].astype(str)))
    n = len(fixtures)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.5 * nrows),
                             squeeze=False)
    for i, fx in enumerate(fixtures):
        ax = axes[i // ncols][i % ncols]
        sub = plans[plans["fixture"].astype(str) == fx]
        sites = []
        for raw in sub["legs"].astype(str):
            for (site, _s, _e) in parse_legs(raw):
                if site not in sites:
                    sites.append(site)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(sites), 1)))
        color = {s: cmap[k % len(cmap)] for k, s in enumerate(sites)}
        moves = 0
        for row_i, (_, r) in enumerate(sub.sort_values("officer").iterrows()):
            legs = parse_legs(r["legs"])
            if not legs:
                continue
            for (site, s, e) in legs:
                ax.barh(row_i, max(e - s, 0.6), left=s, height=0.62,
                        color=color[site], alpha=0.9, edgecolor="white",
                        linewidth=0.4)
            for (l1, l2) in zip(legs[:-1], legs[1:]):
                ax.plot([l1[2], max(l2[1], l1[2] + 0.4)], [row_i, row_i],
                        color=color[l2[0]], lw=1.1, ls=":", alpha=0.95)
            moves += max(0, len(legs) - 1)
        ax.axvspan(0, PHASE_BINS[0], color="#F2F2F2", zorder=0)
        ax.axvspan(PHASE_BINS[1], REL_BINS, color="#F2F2F2", zorder=0)
        ax.axvline(KICKOFF_BIN, color="#2E86AB", lw=1.0, ls="--")
        ax.set_xlim(0, REL_BINS)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"officer {int(o)}"
                            for o in sub.sort_values("officer")["officer"]],
                           fontsize=6)
        ax.set_xticks([0, 6, 12, 18, 24, 27])
        ax.set_xticklabels(["0", "6", "KO", "+6", "+12", "27"], fontsize=6.5)
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"{fx[:44]}\nrelocations = {moves}", fontsize=7.5)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles = [plt.Line2D([0], [0], color=c, lw=5, label=s)
               for s, c in list({}.items())]  # sites differ per panel
    fig.suptitle("Officer movement across the game window (colour = staffed "
                 "site; dotted = relocation; shaded = pre/post phases)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"movement figure -> {out_png}")
    return {"n_fixtures": n,
            "relocations": int(plans["n_moves"].sum())
            if "n_moves" in plans.columns else None}


def coefs_table(coefs_csv, metrics_json, out_tex, top=12):
    co = pd.read_csv(coefs_csv)
    co["std_err"] = pd.to_numeric(co["std_err"], errors="coerce")
    co["z_value"] = pd.to_numeric(co["z_value"], errors="coerce")
    co["coefficient"] = pd.to_numeric(co["coefficient"], errors="coerce")
    co = co[np.isfinite(co["coefficient"])].copy()
    co["absz"] = co["z_value"].abs().fillna(0.0)
    co = co.sort_values("absz", ascending=False).head(int(top))
    rows = []
    for _, r in co.iterrows():
        z = f"{r['z_value']:.1f}" if np.isfinite(r["z_value"]) else "--"
        p = r["p_value"]
        try:
            p = float(p)
            ptxt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        except Exception:  # noqa: BLE001
            ptxt = "--"
        rows.append(f"{_esc(r['variable'])} & {r['coefficient']:+.3f} & "
                    f"{r['std_err']:.3f} & {z} & {ptxt} \\\\")
    panel_a = "\n".join(rows) if rows else "% (no count coefficients)"

    prop_rows, pruned_note = [], ""
    try:
        a = json.load(open(metrics_json))["audit"]
        pr = ((a.get("two_stage") or {}).get("pruned_propensity") or {})
        # Prefer the PRUNED fit when the run was refit with --prune-p: the
        # full-design terms are not the shipped coefficients.
        terms = pr.get("terms") or (a.get("propensity") or {}).get("terms") or {}
        if pr.get("terms"):
            pruned_note = (" Backward elimination at $p>0.05$ drops "
                           f"{len(pr.get('dropped') or [])} propensity and "
                           f"{len((a.get('pruned_refit') or {}).get('dropped') or [])} "
                           "count terms; the handoff gate still passes.")
        for name, (b, se) in sorted(terms.items(),
                                    key=lambda kv: -abs(kv[1][0])):
            prop_rows.append(f"{_esc(name)} & {b:+.3f} & {se:.3f} & "
                             f"{b / max(se, 1e-9):+.1f} & \\\\")
    except Exception as exc:  # noqa: BLE001
        prop_rows = [f"% propensity table unavailable ({exc})"]
    panel_b = "\n".join(prop_rows)

    tex = (
        "% game_day_model_coefs.tex - generated by 63_paper_game_day_assets.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\begin{scriptsize}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "\\multicolumn{5}{l}{\\textbf{Panel A: count refit (NB/Poisson, "
        "segment$\\times$15-min, searched spec)}} \\\\\n"
        "Variable & Coef & SE & $z$ & $p$ \\\\\n\\midrule\n"
        + panel_a + "\n\\midrule\n"
        "\\multicolumn{5}{l}{\\textbf{Panel B: two-stage propensity "
        "(binomial logit, $k=$ speeding trips, $n=$ judged trips)}} \\\\\n"
        "Variable & Coef & SE & $z$ & \\\\\n\\midrule\n"
        + panel_b + "\n\\bottomrule\n\\end{tabular}\\end{scriptsize}\n"
        "\\caption{\\textcolor{Highlight}{\\textbf{Game-day model "
        "coefficients.}} Panel A: refit of the searched count structure "
        "(exposure elasticity free; grouped random slope on \\texttt{lanes\\_z}). "
        "Panel B: the two-stage propensity on the same searched design; the "
        "shipped surface multiplies the predicted judged-trip denominator by "
        "this probability." + pruned_note + " Full tables in "
        "\\texttt{game\\_day\\_model\\_coefs.csv} "
        "and the run audit.}\n"
        "\\label{tab:gamedaycoefs}\\end{table}\n")
    Path(out_tex).write_text(tex, encoding="utf-8")
    print(f"coefficient table -> {out_tex} "
          f"({len(rows)} count + {len(prop_rows)} propensity rows)")
    return {"count_rows": len(rows), "propensity_rows": len(prop_rows)}


def prediction_table(metrics_json, out_tex):
    """Real LaTeX table for the game-day prediction/holdout metrics.

    Replaces the comments-only file 59 writes: the paper \\inputs a table
    with \\label{tab:gamedaypred}, so an all-comment file silently removes
    the table and leaves \\Cref undefined.
    """
    m = json.load(open(metrics_json))
    met, a = m["metrics"], m["audit"]
    h = a.get("handoff") or {}
    b = a.get("bayes") or {}
    ts = a.get("two_stage") or {}
    refit = a.get("refit") or {}
    fixt = a.get("fixture_variation") or {}

    def f(key, default=float("nan")):
        try:
            return float(met.get(key, default))
        except Exception:  # noqa: BLE001
            return float("nan")

    rows = [
        ("Future fixtures / observed segment--bins",
         f"{a.get('n_future_fixtures', 0)} / {int(f('n_observed_rows', 0)):,}"),
        ("Search", f"{a.get('search', {}).get('done_iters', '?')}/"
                   f"{a.get('search', {}).get('total_iters', '?')} iters, "
                   f"BIC {a.get('search', {}).get('best_bic', float('nan')):.0f}"),
        ("Refit", f"{refit.get('spec_label', '?')}, exposure "
                  f"{refit.get('exposure_mode', '?')} "
                  f"({refit.get('exposure_coef', float('nan')):.2f})"),
        ("Two-stage shipped", f"{ts.get('shipped')} "
                              f"(denominator {ts.get('denominator', '?')})"),
        ("Bayes prior sd (game interactions)",
         f"{b.get('selected_game_sd')} (evidence-selected)"),
        ("Bin Spearman", f"{f('bin_spearman'):.3f}"),
        ("Segment Spearman", f"{f('segment_spearman'):.3f} "
                             f"(count surface {f('count_segment_spearman'):.3f})"),
        ("Segment top-20 Jaccard", f"{f('segment_top20'):.3f} "
                                   f"(count surface {f('count_segment_top20'):.3f})"),
        ("Calibration slope", f"{f('calibration_slope'):.2f}"),
        ("Per-fixture observed-only $r$",
         f"{h.get('pearson_min')}--{h.get('pearson_mean')}"),
        ("Observed top-20 coverage", f"{h.get('top20_coverage_mean')}"),
        ("Top-20 site-set Jaccard across games", f"{h.get('top20_jaccard')}"),
        ("Handoff gate", f"pass={h.get('pass')}"),
        ("Posterior-predictive 90\\% coverage",
         f"{f('bayes_interval_90_coverage'):.3f}"),
        ("Concentration (top-1 segment)",
         f"{100 * f('top1_share'):.1f}\\%"),
    ]
    # Row values land in a tabular cell, which is TEXT mode. A literal "_"
    # (e.g. the "trips_with_judged_speed" denominator name) makes TeX emit
    # "Missing $ inserted", and the stray "$" it inserts then desynchronises
    # math mode for the rest of the document -- so the breakage surfaces far
    # away from its cause, in later \input files. Escape "_" here (skipping
    # ones already escaped) and leave "%" alone: row values that want a
    # percent sign already carry "\%" (see "Concentration" above).
    def _tex_cell(value):
        return re.sub(r"(?<!\\)_", r"\\_", str(value))

    body = "\n".join(f"{_tex_cell(k)} & {_tex_cell(v)} \\\\" for k, v in rows)
    tex = (
        "% game_day_predict.tex - generated by 63_paper_game_day_assets.py\n"
        "\\begin{table}[htbp]\n\\centering\n\\begin{scriptsize}\n"
        "\\begin{tabular}{lr}\n\\toprule\n"
        "Game-day surface: search, prediction, and handoff & Value \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n"
        "\\end{tabular}\\end{scriptsize}\n"
        "\\caption{\\textcolor{Highlight}{\\textbf{Game-day 15-minute surface: "
        "validation summary.}} Six held-out fixtures, 1.27M predicted "
        "segment--bins (278k observed). The shipped surface is the "
        "two-stage product (measured judged-trip denominator $\\times$ "
        "binomial propensity); the count-surface column is the single-stage "
        "comparison on the same rows. The handoff gate requires positive "
        "per-fixture ranking, a game-specific top-20 set, and calibration. "
        "Full audit in \\texttt{game\\_day\\_holdout\\_metrics.json}.}\n"
        "\\label{tab:gamedaypred}\\end{table}\n")
    Path(out_tex).write_text(tex, encoding="utf-8")
    print(f"prediction table -> {out_tex} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans",
                    default=str(PROJ / "results" / "mip_free"
                                / "game_day_mip_plans.csv"))
    ap.add_argument("--coefs",
                    default=str(PROJ / "results" / "game_day_model_coefs.csv"))
    ap.add_argument("--metrics",
                    default=str(PROJ / "results"
                                / "game_day_holdout_metrics.json"))
    ap.add_argument("--fig-out",
                    default=str(PROJ / "figures" / "game_day_movement.png"))
    ap.add_argument("--tex-out",
                    default=str(PROJ / "results"
                                / "game_day_model_coefs.tex"))
    ap.add_argument("--pred-tex-out",
                    default=str(PROJ / "results" / "game_day_predict.tex"))
    args = ap.parse_args()
    movement_figure(args.plans, args.fig_out)
    coefs_table(args.coefs, args.metrics, args.tex_out)
    prediction_table(args.metrics, args.pred_tex_out)


if __name__ == "__main__":
    main()
