"""Score game-day run directories against the handoff gate.

Reads ``game_day_holdout_metrics.json`` from each directory and prints every
gate quantity beside the threshold it was judged on, plus the holdout level
(observed / predicted) that no gate in this file currently tests.

The thresholds are NOT redefined here -- they are read from each run's own
``audit.handoff.thresholds``, which 59 writes from its module constants. A
verdict replayed against today's constants would silently disagree with the
verdict the run actually reached, which is the whole point of comparing a
pre-fix surface with a post-fix one.

Usage: python scripts/score_handoff_gates.py <run_dir> [<run_dir> ...]
"""
import json
import os
import sys

import pandas as pd


def load(d):
    p = os.path.join(d, "game_day_holdout_metrics.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    audit = j.get("audit") or {}
    h = audit.get("handoff") or {}
    amp = audit.get("surface_amp") or {}
    fv = audit.get("fixture_variation") or {}
    level = None
    pva = os.path.join(d, "game_day_pred_vs_actual.csv")
    if os.path.exists(pva):
        t = pd.read_csv(pva)
        row = t[t["fixture"].astype(str).str.upper() == "ALL GAMES"]
        if len(row):
            obs = float(row["obs_total"].iloc[0])
            pred = float(row["pred_total"].iloc[0])
            level = {"obs": obs, "pred": pred,
                     "obs_over_pred": obs / max(pred, 1e-9)}
    return {"dir": d, "metrics": j.get("metrics") or {}, "handoff": h,
            "amp": amp, "fv": fv, "level": level}


def _fmt(v, nd=4):
    if v is None:
        return "     -"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        raise SystemExit(2)
    runs = [r for r in (load(d) for d in dirs) if r]
    if not runs:
        raise SystemExit("no run directories with game_day_holdout_metrics.json")

    h0 = runs[0]["handoff"]
    th = h0.get("thresholds") or {}
    jac_max = th.get("top20_jaccard_max")
    rho_max = th.get("surface_spearman_max")
    band = th.get("calibration_band") or [None, None]
    print(f"thresholds (from {runs[0]['dir']}): "
          f"jaccard<={jac_max} spearman<={rho_max} slope in {band}")

    rows = []
    for r in runs:
        h = r["handoff"]
        slope = h.get("calibration_slope")
        lag = ", ".join(sorted(r["dir"].rstrip("/\\").split("/")[-1]
                               .split("\\")[-1]
                               for _ in [0])[:1])
        rows.append({
            "run": lag,
            "pass": h.get("pass"),
            "pearson_min>0": h.get("pearson_min"),
            "jaccard": h.get("top20_jaccard"),
            "spearman": h.get("pairwise_surface_spearman_mean"),
            "slope": slope,
            "calibrated": h.get("calibrated"),
            "btw_fix_var_%": h.get("variance_between_fixtures_pct"),
            "argmax_sites": h.get("unique_argmax_sites"),
            "in_slope": r["amp"].get("in_sample_slope"),
            "in_level": r["amp"].get("in_sample_level_ratio"),
            "hold_obs/pred": (r["level"]["obs_over_pred"]
                              if r["level"] else None),
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200,
                           "display.max_columns", 40):
        print()
        print(df.to_string(index=False))
    print()
    for r in runs:
        h = r["handoff"]
        fails = h.get("failures") or []
        print(f"{r['dir']}: pass={h.get('pass')}"
              + (f"  failures={fails}" if fails else "  (all gates clear)"))
        lv = r["level"]
        if lv:
            _mult = lv["pred"] / max(lv["obs"], 1e-9)
            _word = "over" if _mult > 1 else "UNDER"
            print(f"   holdout level: obs={lv['obs']:.0f} "
                  f"pred={lv['pred']:.1f} obs/pred={lv['obs_over_pred']:.3f} "
                  f"(pred is {_mult:.2f}x the count: {_word}-predicts) "
                  "-- NOT TESTED BY ANY GATE ABOVE")


if __name__ == "__main__":
    main()
