"""
59d_compare_seeds.py
========================================================================
Compare the shipped seed-7 game-day search against the fresh seed-8/9
confirmation chains (pbs_59_multiseed.pbs): do independent SA runs land
near the same BIC with overlapping specs?

Reads (missing files are reported, not fatal):
  results/game_day_search_best.json                    (seed 7, shipped)
  results_multiseed/seed8/game_day_search_best.json
  results_multiseed/seed9/game_day_search_best.json
plus each out-dir's game_day_holdout_metrics.json when present.

Prints: BIC table, fixed/random/grouped term overlap (Jaccard), full
decision-vector agreement, iteration counts, and a plain-language
verdict guide. Runs anywhere in seconds (no HPC needed):

  python scripts/59d_compare_seeds.py
"""
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = {
    "seed7 (shipped)": os.path.join(PROJ, "results",
                                    "game_day_search_best.json"),
    "seed8": os.path.join(PROJ, "results_multiseed", "seed8",
                          "game_day_search_best.json"),
    "seed9": os.path.join(PROJ, "results_multiseed", "seed9",
                          "game_day_search_best.json"),
}
HOLDOUT = {
    "seed7 (shipped)": os.path.join(PROJ, "results",
                                    "game_day_holdout_metrics.json"),
    "seed8": os.path.join(PROJ, "results_multiseed", "seed8",
                          "game_day_holdout_metrics.json"),
    "seed9": os.path.join(PROJ, "results_multiseed", "seed9",
                          "game_day_holdout_metrics.json"),
}


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    specs = {}
    for label, path in CANDIDATES.items():
        if os.path.exists(path):
            with open(path) as f:
                specs[label] = json.load(f)
            print(f"[59d] loaded {label}: {path}")
        else:
            print(f"[59d] MISSING (seed not run yet?): {path}")
    if not specs:
        print("[59d] nothing to compare.")
        return
    print()

    labels = list(specs.keys())
    print("=== BIC (lower is better) ===")
    base = None
    for lab in labels:
        bic = specs[lab].get("best_bic")
        it = specs[lab].get("iters_completed", specs[lab].get("iters"))
        algo = specs[lab].get("algo")
        seed = specs[lab].get("seed")
        flag = ""
        if base is None:
            base = bic
        elif bic is not None and base is not None:
            flag = f"  (delta vs first: {bic - base:+.2f})"
        print(f"  {lab:16s} BIC={bic:.2f}  iters={it}  algo={algo}  "
              f"seed={seed}{flag}")
    print()

    print("=== spec overlap (Jaccard; 1.0 = identical) ===")
    ref = labels[0]
    for lab in labels[1:]:
        for key in ("fixed_terms", "rdm_terms", "rdm_cor_terms",
                    "grouped_terms"):
            a = specs[ref].get("best_spec", {}).get(key, [])
            b = specs[lab].get(key, specs[lab].get("best_spec", {})
                               .get(key, []))
            # tolerate flat layout (some writers hoist terms to top level)
            print(f"  {key:15s} {ref} vs {lab}: {jaccard(a, b):.3f} "
                  f"(|A|={len(set(a))}, |B|={len(set(b))})")
    print()

    print("=== full decision-vector agreement ===")
    for lab in labels[1:]:
        da = specs[ref].get("best_decision")
        db = specs[lab].get("best_decision")
        if da is None or db is None or len(da) != len(db):
            print(f"  {ref} vs {lab}: not comparable "
                  f"(len {None if da is None else len(da)} vs "
                  f"{None if db is None else len(db)})")
            continue
        agree = sum(1 for x, y in zip(da, db) if x == y) / len(da)
        print(f"  {ref} vs {lab}: {agree:.3f} of {len(da)} genes agree")
    print()

    print("=== holdout skill (if metrics present) ===")
    for lab, path in HOLDOUT.items():
        if not os.path.exists(path):
            print(f"  {lab:16s} metrics missing")
            continue
        with open(path) as f:
            m = json.load(f).get("metrics", {})
        print(f"  {lab:16s} bin_spearman={m.get('bin_spearman')}  "
              f"seg_spearman={m.get('segment_spearman')}  "
              f"rmse={m.get('rmse')}  "
              f"cal_slope={m.get('calibration_slope')}")
    print()
    print("=== how to read this ===")
    print("  CONFIRMED: seed8/9 BICs within ~10-50 of seed 7 AND fixed-term")
    print("  Jaccard >= ~0.7 (same core terms: logexp, log_length, in_event,")
    print("  crowd/time interactions). The flat-tail optimum is then a genuine")
    print("  basin, not a one-chain fluke.")
    print("  DIVERGED: a seed lands >100 BIC lower (seed 7 was a local")
    print("  optimum -> adopt the better spec), or similar BIC with disjoint")
    print("  terms (multimodal surface -> report the ensemble, not one spec).")


if __name__ == "__main__":
    main()
