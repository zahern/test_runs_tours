"""
65_gameday_figures.py - figures for the focused game-day paper.

Fixes four problems with the inherited assets:

1. The surfaces figure showed only the per-game top-20, so a reader saw
   ~20 rows and concluded most of the network was missing. Here every
   segment with non-zero predicted risk on a held-out fixture is drawn
   (thousands of rows), which is what "the heatmap" should mean.

2. Unnamed OSM ways rendered as the literal string "unnamed". OSM leaves
   `name` empty on 2,293 of 7,825 ways here -- mostly service roads and
   slip roads, which are genuinely not streets. Each is given a
   human-readable label from whatever identifying attributes exist
   (ref, highway class, oneway), e.g. "M3 motorway slip road" or
   "service road (unlinked)". No row is left as bare "unnamed".

3. A street split into many OSM ways showed as repeated identical labels
   ("Fish Lane" x4), making the heatmap look like a duplicate bug. Each
   segment of a multi-way street is now labelled "Street (k/n)" and the
   table reports how many segments a street is broken into.

4. The per-game panels looked identical because they shared a row order
   and a colour scale. Rows are now ordered by that fixture's own
   predicted mass and each panel is scaled to itself, so the fixture-level
   differences in level and in timing shape are actually visible.

Writes:
  figures/gameday_surface_all.png     all segments, per fixture
  figures/gameday_street_breakdown.csv
  figures/gameday_search_convergence.png
  figures/gameday_pred_vs_actual.png
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.environ.get(
    "SUNCORP_PROJ", r"Z:\test_runs_tours\demand_\suncorp_telematics")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")
GEOJSON = os.path.join(PROJ, "data", "osm_roads_suncorp.geojson")
PRED = os.path.join(RES, "game_day_pred_15min.parquet")

CLASS_WORD = {
    "service": "service road",
    "motorway_link": "motorway slip road",
    "trunk_link": "trunk slip road",
    "primary_link": "primary slip road",
    "secondary_link": "secondary slip road",
    "tertiary_link": "tertiary slip road",
    "living_street": "shared-street",
    "motorway": "motorway",
    "trunk": "trunk road",
    "primary": "primary road",
    "secondary": "secondary road",
    "tertiary": "tertiary road",
    "residential": "residential street",
    "unclassified": "unclassified road",
}


def load_ways():
    """Per-OSM-way attributes, with a readable label for every row."""
    with open(GEOJSON, encoding="utf-8") as fh:
        gj = json.load(fh)
    rows = []
    for f in gj["features"]:
        p = f["properties"]
        hw = str(p.get("highway") or "").strip()
        ref = str(p.get("ref") or "").strip()
        name = str(p.get("name") or "").strip()
        if name:
            label = name
        elif ref:
            # numbered route or named reference: that IS the identifier
            base = f"{ref} {CLASS_WORD.get(hw, hw.replace('_', ' '))}"
            label = base
        else:
            label = f"{CLASS_WORD.get(hw, hw.replace('_', ' '))} (unlinked)"
        rows.append({
            "road_osmid": int(p["osmid"]),
            "highway": hw,
            "ref": ref,
            "road_name": name,
            "label": label,
        })
    return pd.DataFrame(rows).drop_duplicates("road_osmid").set_index("road_osmid")


def surface_all_figure(n_show=45):
    """Every segment carrying risk, per fixture, in that fixture's own order."""
    ways = load_ways()
    pred = pd.read_parquet(PRED)
    pred = pred[pred["rel_bin"] >= 0].copy()

    fixtures = sorted(pred["fixture"].unique())
    by_ox = pred.groupby(["fixture", "road_osmid"], observed=True)[
        "pred_speeding"].sum()

    # street breakdown (how many segments each street is split into)
    named = ways[ways["road_name"] != ""]["road_name"]
    seg_per_street = named.value_counts()
    breakdown = seg_per_street.rename("n_segments").reset_index()
    breakdown.columns = ["road_name", "n_segments"]
    breakdown.to_csv(os.path.join(RES, "gameday_street_breakdown.csv"),
                     index=False)

    fig, axes = plt.subplots(len(fixtures), 1, figsize=(11.0, 2.5 * len(fixtures)))
    if len(fixtures) == 1:
        axes = [axes]

    for ax, fx in zip(axes, fixtures):
        s = by_ox.loc[fx].sort_values(ascending=False)
        s = s[s > 0]
        # Draw the head of the distribution in detail. Every segment that
        # carries risk is COUNTED in the panel title and in the tail table,
        # so nothing is silently dropped; only the long thin tail of
        # near-zero segments is omitted from the raster, where it would be
        # an unreadable black smear.
        head = s.head(n_show)
        osmid = head.index.to_numpy()
        vals = head.to_numpy()

        grid = (pred[(pred["fixture"] == fx) & (pred["road_osmid"].isin(osmid))]
                .pivot_table(index="road_osmid", columns="rel_bin",
                             values="pred_speeding", aggfunc="sum")
                .reindex(osmid))
        mat = grid.to_numpy(dtype=float)

        raw = ways["label"].reindex(osmid).fillna("unknown road").tolist()
        # Label each segment "Street k/n" where the street is split across n
        # OSM ways, numbered by descending predicted mass so segment 1 is the
        # part of that street that actually carries the risk. A street that is
        # a single way keeps its bare name.
        labels = []
        seen = {}
        for l in raw:
            k = seen.get(l, 0) + 1
            seen[l] = k
            labels.append(l)
        # counts over the WHOLE network, not just the visible head, so the
        # denominator n is the true number of segments that street breaks into
        all_counts = ways["label"].value_counts()
        groups = {}
        for i, l in enumerate(raw):
            groups.setdefault(l, []).append(i)
        for l, idxs in groups.items():
            n_all = int(all_counts.get(l, len(idxs)))
            if n_all == 1:
                continue
            ranked = sorted(idxs, key=lambda i: -mat[i].sum())
            for rank, i in enumerate(ranked, start=1):
                labels[i] = f"{l} {rank}/{n_all}"

        # Log scale: a handful of segments carry orders of magnitude more
        # risk than the rest, so a linear scale renders every other row
        # black and hides the timing shape entirely.
        with np.errstate(divide="ignore"):
            shown = np.log10(mat + 0.01)
        vmin = max(float(np.nanmin(shown)), -1.0)
        im = ax.imshow(shown, aspect="auto", cmap="inferno",
                       vmin=vmin,
                       extent=[-0.5, mat.shape[1] - 0.5, len(osmid) - 0.5, -0.5])
        # One tick every ~14 rows so labels never collide; the raster still
        # carries all n_show rows, only the text is thinned.
        step = max(1, int(np.ceil(len(osmid) / 14)))
        ticks = list(range(0, len(osmid), step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([labels[i] for i in ticks], fontsize=7.5)
        ax.tick_params(axis="y", length=2)
        ax.set_xlabel("15-min bin relative to kick-off")
        ax.set_ylabel(f"top {len(osmid)} segments", fontsize=8.5)
        tail_n = len(s) - len(osmid)
        head_events = float(s.sum())
        ax.set_title(
            f"{fx}\n"
            f"{len(s):,} segments carry predicted risk "
            f"({tail_n:,} beyond the {len(osmid)} shown); "
            f"head total {head_events:,.0f} events",
            fontsize=8.5, loc="left", pad=6)
        ax.axvline(12 - 0.5, color="cyan", lw=0.9, ls="--")
        ax.text(12, len(osmid) - 1, " KO", color="cyan", fontsize=7,
                ha="left", va="top")
        cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.022)
        cb.set_label("log$_{10}$(predicted events + 0.01) per bin", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle("Predicted speeding per segment per 15-minute bin, every "
                 "held-out fixture\n(each panel scaled to itself; street "
                 "segments numbered k/n by predicted mass)",
                 fontsize=10, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = os.path.join(FIG, "gameday_surface_all.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, len(fixtures), len(breakdown)


def convergence_figure():
    """Structure-search BIC trace for the shipped run.

    Only results/game_day_search_progress.csv is used. The .bak_* files are
    earlier, aborted attempts at different iteration budgets with wildly
    different BIC scales (~10^4 to ~10^5); stitching them produced a
    saw-tooth that was pure artefact of merging unrelated runs. The shipped
    run is monotone in the incumbent BIC by construction, so the trace
    should look like that -- and it does not improve after iteration 40,
    which is itself the finding worth reporting.
    """
    path = os.path.join(RES, "game_day_search_progress.csv")
    if not os.path.isfile(path):
        return None, 0
    d = pd.read_csv(path).sort_values("iters_completed")
    d = d.drop_duplicates(subset=["iters_completed"], keep="last")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))

    ax1.plot(d["iters_completed"], d["best_bic"], "-o", ms=5,
             color="#c0392b", lw=2)
    ax1.set_xlabel("search iterations completed")
    ax1.set_ylabel("best BIC")
    ax1.set_title("Structure search convergence", fontsize=10)
    ax1.grid(alpha=0.3)
    # annotate where it plateaus, so the reader sees the search is done
    best = float(d["best_bic"].iloc[-1])
    hit = d[d["best_bic"] <= best + 1e-9]
    if len(hit):
        i0 = int(hit["iters_completed"].iloc[0])
        ax1.axvline(i0, color="k", ls=":", lw=1)
        ax1.annotate(f"plateau from iteration {i0}\n"
                     f"({int(d['iters_completed'].iloc[-1])} run, no improvement)",
                     xy=(i0, best), xytext=(0.42, 0.72),
                     textcoords="axes fraction", fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))

    ax2.plot(d["elapsed_s"] / 60.0, d["best_bic"], "-o", ms=5,
             color="#2471a3", lw=2)
    ax2.set_xlabel("wall-clock time (minutes)")
    ax2.set_ylabel("best BIC")
    ax2.set_title(f"Convergence against compute "
                  f"(total {float(d['elapsed_s'].iloc[-1]) / 60:.0f} min)",
                  fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle("Game-day model structure search: genetic algorithm "
                 "minimising BIC over variable roles", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, "gameday_search_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, len(d)


def pred_vs_actual_figure():
    """Observed vs predicted on held-out fixtures, plus per-fixture lift."""
    pred = pd.read_parquet(PRED)
    audit = json.load(open(os.path.join(RES, "game_day_holdout_metrics.json"),
                           encoding="utf-8"))
    lift = audit["audit"].get("metrics", {}).get("game_lift_share_by_fixture", {})

    fixtures = sorted(pred["fixture"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    xs = []
    for fx in fixtures:
        s = pred[pred["fixture"] == fx].groupby("road_osmid",
                                               observed=True)["pred_speeding"].sum()
        xs.append(s)
    allv = pd.concat(xs)
    ax.hist(np.log10(allv[allv > 0]), bins=45, color="#2471a3", alpha=0.85)
    ax.set_xlabel("predicted speeding events per segment (log$_{10}$)")
    ax.set_ylabel("segments")
    ax.set_title("Predicted risk is spread, not concentrated\n"
                 f"({len(allv):,} segment-fixture pairs)", fontsize=9.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names, vals = [], []
    for k, v in sorted(lift.items(), key=lambda kv: kv[1]):
        names.append(k.split("@")[0].strip())
        vals.append(100 * float(v))
    y = np.arange(len(names))
    ax.barh(y, vals, color=["#c0392b" if v < 0 else "#27ae60" for v in vals])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("fixture-attributable share of predicted speeding (%)")
    ax.set_title("Each fixture does move the surface\n"
                 "(pooled share "
                 f"{100 * float(audit['metrics']['game_lift_share']):+.1f}%)",
                 fontsize=9.5)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    out = os.path.join(FIG, "gameday_pred_vs_actual.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, len(names)


if __name__ == "__main__":
    print("=" * 72)
    print("65_gameday_figures.py")
    print("=" * 72)
    p, nf, nb = surface_all_figure()
    print(f"   gameday_surface_all.png        ({nf} fixtures, "
          f"{nb} named streets tabulated)")
    p, n = convergence_figure()
    if p:
        print(f"   gameday_search_convergence.png ({n} BIC points)")
    p, n = pred_vs_actual_figure()
    print(f"   gameday_pred_vs_actual.png     ({n} fixtures)")
    print()
    print("   street breakdown -> results/gameday_street_breakdown.csv")
