"""Read E1/E2 Stage-5 metrics from several Exampville run dirs and tabulate
the E2-minus-E1 mode gain, so seed replicates can be compared directly.

Usage:  python _exv_rep_summary.py runs/exampville_s5fix_20261030 runs/exampville_s5rep1_20261030 ...
"""
import json
import os
import sys

KEYS = [
    ("work_mode_model", "work mode"),
    ("nonmand_mode_model", "nonmand mode"),
    ("work_dest_model", "work dest"),
    ("nonmand_dest_model", "nonmand dest"),
]


def load(path):
    p = path if path.endswith(".json") else os.path.join(path, "exampville_metrics.json")
    try:
        with open(p) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def main(dirs):
    rows = []
    for d in dirs:
        m = load(d)
        tag = os.path.basename(os.path.normpath(d))
        if m is None:
            rows.append((tag, None))
            continue
        e1 = m.get("arms", {}).get("E1", {})
        e2 = m.get("arms", {}).get("E2", {})
        rows.append((tag, {"E1": e1, "E2": e2, "wall": m.get("walltime_min")}))

    for tag, rec in rows:
        print(f"\n=== {tag}")
        if rec is None:
            print("   (no exampville_metrics.json yet)")
            continue
        e1, e2 = rec["E1"], rec["E2"]
        print(f"   ok: E1={e1.get('ok')} E2={e2.get('ok')}   walltime={rec['wall']} min")
        print(f"   E2 features: mode={e2.get('n_mode_features')} dest={e2.get('n_dest_features')}"
              f"   best_fitness={e2.get('best_fitness')}")
        print(f"   {'segment':14s} {'E1 LL':>12s} {'E2 LL':>12s} {'dLL':>10s}"
              f" {'E1 rho2':>9s} {'E2 rho2':>9s} {'drho2':>9s}")
        for key, name in KEYS:
            a, b = e1.get(key) or {}, e2.get(key) or {}
            if not a or not b:
                continue
            all1, all2 = a.get("ll"), b.get("ll")
            r1, r2 = a.get("rho2"), b.get("rho2")
            print(f"   {name:14s} {all1:12.4f} {all2:12.4f} {all2 - all1:+10.2f}"
                  f" {r1 if r1 is None else f'{r1:9.5f}'} {r2 if r2 is None else f'{r2:9.5f}'}"
                  f" {'' if (r1 is None or r2 is None) else f'{r2 - r1:+9.5f}'}")

    done = [(t, r) for t, r in rows if r]
    if len(done) > 1:
        print("\n=== mode LL gain (E2 - E1) across runs")
        for key, name in KEYS[:2]:
            vals = []
            for tag, rec in done:
                a = (rec["E1"].get(key) or {}).get("ll")
                b = (rec["E2"].get(key) or {}).get("ll")
                if a is not None and b is not None:
                    vals.append((tag, b - a))
            if not vals:
                continue
            print(f"  {name}: " + "  ".join(f"{v:+.2f}" for _t, v in vals)
                  + f"   (all positive: {all(v > 0 for _t, v in vals)})")


if __name__ == "__main__":
    main(sys.argv[1:])
