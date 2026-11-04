#!/usr/bin/env python
"""Render the per-cell game-day response as a table and a small figure.

Consumes results_tests/t9_cell_response/game_day_cell_response.csv, which 59
writes from the FITTED model (report_cell_response): for each road-class x
distance-band cell it holds the ratio of predicted flow with the crowd doubled
to predicted flow at the observed crowd (and, from the next run on, the ratio
with the crowd tier forced to 1).

Why a counterfactual and not the coefficient table: the grouped rows of
game_day_model_coefs.csv are the POPULATION mean and SD of a random
coefficient drawn per cell (beta_g ~ N(mean, sd)), on the design's
standardised scale, so they cannot be exponentiated cell by cell. Asking the
fitted model to predict the same rows twice makes every scale cancel.

The decomposition panel splits each cell's log multiplier into the part the
grouped block supplies (the cell-varying crowd slope) and the parts the fixed
interactions supply (att_x_dist, att_x_length). That split is the point of the
table: it says whether the cells differ because of the hierarchy or because
the fixed distance/length terms happen to be evaluated at different
covariates.

Usage:
    python scripts/report_cell_response.py [--csv PATH] [--coefs PATH]
                                          [--fig PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results_tests/t9_cell_response/game_day_cell_response.csv"
DEFAULT_FIG = (ROOT / "figures_tests/t9_cell_response/"
               "game_day_cell_response.png")
# The refit that produced the probe lives with the A/B cell arm; fall back to
# any game_day_model_coefs.csv next to the probe.
COEF_CANDIDATES = [
    ROOT / "results_tests/t9_ab_cell/game_day_model_coefs.csv",
    ROOT / "results/game_day_model_coefs.csv",
]

BAND_LABELS = {
    "d0_1km": "0-1 km",
    "d1_2.5km": "1-2.5 km",
    "d2.5_5km": "2.5-5 km",
    "d5km_plus": "5+ km",
    # band grain collapses road class away and merges 2.5 km upwards
    "d2.5plus": "2.5+ km",
}
CLASS_ORDER = ["local", "secondary", "arterial"]
CLASS_COLOURS = {"local": "#1f6feb", "secondary": "#d1603d",
                 "arterial": "#2e8b57"}
LN2 = float(np.log(2.0))


def prettify(cell: str) -> str:
    # cell-grain keys are "class|band"; the band grain is a bare band name.
    if "|" not in str(cell):
        return BAND_LABELS.get(str(cell), str(cell))
    cls, _, band = str(cell).partition("|")
    return f"{cls} - {BAND_LABELS.get(band, band)}"


def road_class(cell: str) -> str:
    return str(cell).partition("|")[0] if "|" in str(cell) else ""


def load_probe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("cell", "mult_att_x2"):
        if col not in df.columns:
            raise SystemExit(f"{path} is missing column {col!r}")
    df = df.copy()
    df["label"] = df["cell"].map(prettify)
    df["road_class"] = df["cell"].map(road_class)
    # A x2 crowd step in a log-link model: the elasticity is the log ratio
    # over the log crowd step, so it reads per 1% of attendance.
    df["crowd_elasticity"] = np.log(
        df["mult_att_x2"].clip(lower=1e-12)) / LN2
    return df.sort_values(["mean_dist_km", "cell"]).reset_index(drop=True)


def load_coefs(path: Path | None) -> pd.Series | None:
    if path is None or not Path(path).exists():
        return None
    raw = pd.read_csv(path)
    key = "variable" if "variable" in raw.columns else raw.columns[0]
    val = "coefficient" if "coefficient" in raw.columns else raw.columns[1]
    return raw.set_index(key)[val]


def decompose(df: pd.DataFrame, coef: pd.Series | None) -> pd.DataFrame:
    """Split ln(multiplier) into grouped vs fixed-interaction parts."""
    out = df.copy()
    tot = np.log(out["mult_att_x2"].clip(lower=1e-12))
    out["ln_total"] = tot
    if coef is None:
        out["ln_grouped"] = np.nan
        out["ln_dist"] = np.nan
        out["ln_length"] = np.nan
        out["ln_other"] = np.nan
        return out
    def c(name):
        return float(coef.get(name, 0.0))
    # Grouped block: population mean of the per-cell log_att slope.
    out["ln_grouped"] = LN2 * c("group_mean(log_att)")
    # Fixed crossed terms, at each cell's own mean covariates. lanes_z and
    # maxspeed_z enter at their (approx zero) standardised means; whatever
    # that leaves lands in "other" rather than being attributed here.
    out["ln_dist"] = LN2 * c("att_x_dist") * out["mean_dist_km"]
    out["ln_length"] = LN2 * c("att_x_length") * out["mean_log_length"]
    out["ln_other"] = (out["ln_total"] - out["ln_grouped"]
                       - out["ln_dist"] - out["ln_length"])
    return out


def markdown_table(df: pd.DataFrame, coef: pd.Series | None) -> str:
    has_bg = "mult_big_game" in df.columns
    head = ("| cell | events | mean dist (km) | crowd x2 -> flow | "
            "elasticity per +1% crowd |")
    sep = "|---|---:|---:|---:|---:|"
    if has_bg:
        head += " big game -> flow |"
        sep += "---:|"
    rows = []
    for _, r in df.iterrows():
        line = (f"| {r['label']} | {int(r['events'])} | "
                f"{r['mean_dist_km']:.2f} | x{r['mult_att_x2']:.3f} | "
                f"{r['crowd_elasticity']:+.3f} |")
        if has_bg:
            line += f" x{r['mult_big_game']:.3f} |"
        rows.append(line)
    body = "\n".join([head, sep] + rows)
    lines = [body]
    if coef is not None:
        gm = float(coef.get("group_mean(log_att)", np.nan))
        gs = float(coef.get("group_sd(log_att)", np.nan))
        tot_span = float(df["ln_total"].max() - df["ln_total"].min())
        dist_span = float(df["ln_dist"].max() - df["ln_dist"].min())
        lines.append(
            "\nGrouped crowd slope: population mean "
            f"{gm:.4f} per log-attendance, between-cell SD {gs:.4f} "
            f"(+-{(np.exp(gs * LN2) - 1) * 100:.1f}% in the x2 multiplier at "
            "1 SD). Across these cells the multiplier spans "
            f"x{df['mult_att_x2'].min():.3f} to x{df['mult_att_x2'].max():.3f} "
            f"({tot_span:.3f} in ln); the fixed att_x_dist term spans "
            f"{dist_span:.3f} on its own"
            + (" -- wider than the total, so the remaining terms offset part "
               "of it" if dist_span > tot_span else "") +
            f", and the remainder ({tot_span - dist_span:+.3f}) is the drawn "
            "cell deviation plus the length / lanes / speed terms.")
        if "group_mean(big_game)" in coef.index:
            bm = float(coef.get("group_mean(big_game)"))
            bs = float(coef.get("group_sd(big_game)"))
            lines.append(
                f"\nBig-game tier: population mean {bm:.4f} "
                f"(x{np.exp(bm):.3f}), between-cell SD {bs:.4f} "
                f"(x{np.exp(bm - bs):.3f} to x{np.exp(bm + bs):.3f} at +-1 SD). "
                "Unlike the crowd slope this term DOES vary materially by "
                "cell; it has no crossed terms, so its multiplier is "
                "exp(beta_g) directly.")
    return "\n".join(lines)


def figure(df: pd.DataFrame, out: Path) -> Path:
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150
    has_dec = df["ln_grouped"].notna().any()
    fig, axes = plt.subplots(
        1, 2 if has_dec else 1, figsize=(10.2 if has_dec else 5.6,
                                        5.0 if has_dec else 4.3),
        gridspec_kw={"width_ratios": [1.0, 1.15]} if has_dec else None,
        squeeze=False)
    ax = axes[0][0]

    # Left: the response itself, against distance from the stadium.
    ax.axhline(1.0, color="0.35", lw=1.0, ls="--", zorder=1)
    classes = [c for c in CLASS_ORDER if (df["road_class"] == c).any()]
    if not classes:
        # Band grain: one group per distance band, so there is no class split
        # to colour by and a legend would be empty.
        ax.scatter(df["mean_dist_km"], df["mult_att_x2"],
                   s=18 + 6 * np.sqrt(df["events"].to_numpy(float)),
                   color="#1f6feb", edgecolor="white", linewidth=0.7,
                   zorder=3)
    for cls in classes:
        sub = df[df["road_class"] == cls]
        ax.scatter(sub["mean_dist_km"], sub["mult_att_x2"],
                   s=18 + 6 * np.sqrt(sub["events"].to_numpy(float)),
                   color=CLASS_COLOURS.get(cls, "0.4"), label=cls,
                   edgecolor="white", linewidth=0.7, zorder=3)
    # Distance is the only ordered axis here, and the grouped block does not
    # vary with it, so no fitted guide line is drawn: every cell's own
    # att_x_length value moves its level, and a single curve through the
    # points would be a model of my own making rather than the fit's.
    # Event counts are in the right-hand labels; three cells share almost the
    # same mean distance, so annotating here only produced collisions.
    ax.set_xlabel("mean distance to stadium (km)", fontsize=9)
    ax.set_ylabel("flow multiplier, crowd x2", fontsize=9)
    ax.set_title("Crowd response by group", fontsize=10)
    ax.tick_params(labelsize=8.5)
    ax.margins(y=0.14)
    if classes:
        ax.legend(fontsize=7.5, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.15), ncol=3,
                  title="road class", title_fontsize=7.5)

    if has_dec:
        ax2 = axes[0][1]
        order = df.sort_values("mean_dist_km", ascending=False)
        y = np.arange(len(order))
        parts = (("ln_grouped", "grouped crowd slope", "#1f6feb"),
                 ("ln_dist", "fixed att_x_dist", "#d1603d"),
                 ("ln_length", "fixed att_x_length", "#8a6fbf"),
                 ("ln_other", "unreconciled (draw, lanes, speed)", "0.75"))
        left_pos = np.zeros(len(order))
        left_neg = np.zeros(len(order))
        for col, label, colour in parts:
            vals = order[col].to_numpy(float)
            base = np.where(vals >= 0, left_pos, left_neg)
            ax2.barh(y, vals, left=base, height=0.62, color=colour,
                     label=label, edgecolor="white", linewidth=0.5,
                     zorder=3)
            left_pos = left_pos + np.where(vals >= 0, vals, 0.0)
            left_neg = left_neg + np.where(vals < 0, vals, 0.0)
        ax2.scatter(order["ln_total"], y, color="black", s=16, zorder=4,
                    label="total ln multiplier")
        ax2.axvline(0.0, color="0.35", lw=1.0, zorder=2)
        ax2.set_yticks(y)
        ax2.set_yticklabels(
            [f"{r['label']}  (n={int(r['events'])})"
             for _, r in order.iterrows()], fontsize=8.5)
        ax2.set_xlabel("contribution to ln(multiplier)", fontsize=9)
        ax2.set_title("What moves the multiplier by group", fontsize=10)
        ax2.tick_params(labelsize=8.5)
        ax2.legend(fontsize=6.8, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.16), ncol=2)
        ax2.grid(axis="y", lw=0.4, alpha=0.4)

    fig.suptitle("Fitted per-cell game-day response (count surface, cell grain)",
                 fontsize=10.5)
    fig.text(0.5, 0.012,
             "Draw-averaged counterfactual predictions from the fitted model "
             "on training game rows, exposure held fixed; marker area grows "
             "with training events in the cell.",
             ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--coefs", default=None,
                    help="game_day_model_coefs.csv (default: the cell-arm fit)")
    ap.add_argument("--fig", default=str(DEFAULT_FIG))
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/run_t9_cell_response.sh")
    df = load_probe(path)
    coef_path = Path(args.coefs) if args.coefs else next(
        (p for p in COEF_CANDIDATES if p.exists()), None)
    coef = load_coefs(coef_path)
    if coef is None:
        print("note: no coefficient table found; decomposition is omitted.")
    else:
        print(f"coefficients: {coef_path}")
    df = decompose(df, coef)

    table = markdown_table(df, coef)
    print()
    print(table)
    md = path.with_suffix(".md")
    md.write_text(
        "# Fitted per-cell game-day response\n\n"
        "Cell grain: road class x distance-to-stadium band. Multipliers are "
        "ratios of draw-averaged predicted flow from the fitted count "
        "surface, so they are on the scale the shipped surface uses.\n\n"
        + table + "\n\nCrowd is doubled with the exposure (flow) proxy held at "
        "its observed value, so this is the DIRECT crowd channel of the "
        "count surface, not the exposure-mediated one.\n",
        encoding="utf-8")
    print(f"\nwrote {md}")
    print(f"wrote {figure(df, Path(args.fig))}")


if __name__ == "__main__":
    main()
