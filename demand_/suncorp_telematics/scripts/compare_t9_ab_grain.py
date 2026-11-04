"""Compare two game-day surfaces that differ only in grouping grain.

Reads results_tests/t9_ab_<grain>/game_day_holdout_metrics.json for each grain
and reports the handoff quantities side by side: cross-game ranking overlap
(Jaccard), pairwise surface Spearman, between-fixture variance share and
calibration slope. Also reports how far the two GRAINS disagree with each
other, which bounds how much the parameterisation is doing.

Usage: python scripts/compare_t9_ab_grain.py [grain_a grain_b]
"""
import itertools
import json
import os
import sys


def load(g):
    p = f"results_tests/t9_ab_{g}/game_day_holdout_metrics.json"
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    a, m = j.get("audit") or {}, j.get("metrics") or {}
    r = a.get("refit") or {}
    fv = a.get("fixture_variation") or {}
    h = a.get("handoff") or {}
    fg = a.get("force_grouped_game") or {}
    cg = fg.get("cell_grain") or {}
    return {
        "grain": g,
        "spec": r.get("spec_label"),
        "degraded": r.get("degraded"),
        "group_id_col": fg.get("group_id_col"),
        "n_groups": cg.get("n_cells"),
        "n_params": cg.get("n_params"),
        "n_grouped": len((r.get("fitted") or {}).get("grouped_terms") or []),
        "alpha": r.get("alpha"),
        "bic": r.get("bic"),
        "jaccard10": (fv.get("site_overlap_jaccard") or {}).get("top10"),
        "jaccard20": (fv.get("site_overlap_jaccard") or {}).get("top20"),
        "pair_spearman": fv.get("pairwise_surface_spearman_mean"),
        "var_fixtures_pct": fv.get("variance_between_fixtures_pct"),
        "var_segments_pct": fv.get("variance_between_segments_pct"),
        "unique_argmax": fv.get("unique_argmax_sites"),
        "seg_top20": m.get("segment_top20"),
        "seg_spearman": m.get("segment_spearman"),
        "cal_slope": m.get("calibration_slope"),
        "cal_intercept": m.get("calibration_intercept"),
        "pearson_mean": h.get("pearson_mean"),
        "coverage_mean": h.get("top20_coverage_mean"),
        "game_specific": h.get("game_specific"),
        "calibrated": h.get("calibrated"),
        "handoff": h.get("pass"),
        "handoff_failures": h.get("failures"),
        "top_per_fixture": fv.get("top_segments_per_fixture") or {},
    }


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def main():
    gains = sys.argv[1:3] or ["cell", "fixture"]
    rows = [r for r in (load(g) for g in gains) if r]
    print("=" * 74)
    print("T9 A/B: game-day surface grain")
    print("=" * 74)
    if len(rows) != 2:
        print(f"need both arms; found {[r['grain'] for r in rows]}")
        return 1
    print(f"{'':34s}" + "".join(f"{r['grain']:>16s}" for r in rows))
    keys = [
        ("spec shipped", "spec"), ("degraded", "degraded"),
        ("group id col", "group_id_col"), ("n groups", "n_groups"),
        ("grouped params", "n_params"), ("grouped terms", "n_grouped"),
        ("alpha", "alpha"), ("BIC", "bic"),
        ("top-20 Jaccard", "jaccard20"), ("top-10 Jaccard", "jaccard10"),
        ("pairwise surface Spearman", "pair_spearman"),
        ("variance between fixtures %", "var_fixtures_pct"),
        ("variance between segments %", "var_segments_pct"),
        ("unique argmax sites", "unique_argmax"),
        ("holdout Pearson mean", "pearson_mean"),
        ("top-20 coverage mean", "coverage_mean"),
        ("segment top-20 overlap", "seg_top20"),
        ("segment Spearman", "seg_spearman"),
        ("calibration slope", "cal_slope"),
        ("calibration intercept", "cal_intercept"),
        ("game specific", "game_specific"), ("calibrated", "calibrated"),
        ("handoff pass", "handoff"),
    ]
    for label, k in keys:
        print(f"{label:34s}" + "".join(f"{fmt(r.get(k)):>16s}" for r in rows))
    print()
    for r in rows:
        print(f"-- {r['grain']} top sites per fixture --")
        for fx, v in r["top_per_fixture"].items():
            print(f"   {fx}: {v[:8]}")
        for f in r.get("handoff_failures") or []:
            print(f"   handoff FAIL: {f}")
    # How far apart are the two grains themselves? A per-fixture Jaccard near 1
    # means the parameterisation barely matters; a low value means the choice
    # of grain is a modelling decision, not a detail.
    a_tp, b_tp = rows[0]["top_per_fixture"], rows[1]["top_per_fixture"]
    js = []
    for fx in sorted(set(a_tp) & set(b_tp)):
        A, B = set(a_tp[fx][:20]), set(b_tp[fx][:20])
        if A and B:
            js.append(len(A & B) / len(A | B))
    if js:
        print(f"\ncross-grain top-20 Jaccard per fixture: "
              f"{[round(x, 3) for x in js]} (mean {sum(js) / len(js):.3f})")
    # Within-grain: how many fixture pairs are literally the same top-20?
    for r in rows:
        tp = r["top_per_fixture"]
        ks = sorted(tp)
        pairs = [(len(set(tp[a][:20]) & set(tp[b][:20])) /
                  len(set(tp[a][:20]) | set(tp[b][:20])))
                 for a, b in itertools.combinations(ks, 2)]
        if pairs:
            print(f"{r['grain']:>8s} within-grain top-20 Jaccard: "
                  f"min {min(pairs):.3f} mean {sum(pairs) / len(pairs):.3f} "
                  f"max {max(pairs):.3f} over {len(pairs)} pairs")
    out = "results_tests/t9_ab_compare.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
