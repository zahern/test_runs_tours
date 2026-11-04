"""Paired before/after for the logsum-fill fix, across frames.

Reads arms.json files produced by _paper_stage5_arms.py and reports, per frame,
the mirror (pre-fix) -> fixed (constant fill) difference in c_logsum for arm B
and for each latent class of arm D, so the shift is a WITHIN-frame difference and
the spread across frames is the error bar.

    python _eb_fillfix.py frame1=DIR_MIRROR,DIR_FIXED frame2=DIR_MIRROR,DIR_FIXED

Each DIR is a run directory holding arms.json.  A frame's two runs must share the
cache (same sampled alternatives); the tool checks that and refuses a mismatched
pair, because then the difference would mix the fix with a different sample.
"""
import json
import os
import sys

import numpy as np


def load(run_dir):
    # Accept a run directory (holding arms.json) or a path straight to the file.
    path = (run_dir if os.path.isfile(run_dir)
            else os.path.join(run_dir, "arms.json"))
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}")
    d = json.load(open(path))
    rows = {r["arm"]: r for r in d["rows"]}
    return d["meta"], rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def arm_b(rows):
    r = rows.get("B")
    if not r:
        return None
    c = r.get("coefs") or {}
    se = r.get("std_err") or r.get("se") or {}
    return {"ll": num(r.get("ll")), "n": r.get("n"),
            "logsum": num(c.get("logsum")), "se": num(se.get("logsum")),
            "dist": num(c.get("log_DIST_1")), "crash": num(c.get("log_CRASH_1")),
            "visit": num(c.get("visit_work")),
            "safety": num(c.get("safety_x_crash")),
            "converged": r.get("converged")}


def arm_d(rows):
    r = rows.get("D")
    if not r:
        return None
    betas = r.get("class_betas") or []
    ses = r.get("class_std_err") or []
    return {"ll": num(r.get("ll")), "n": r.get("n"),
            "shares": [num(v) for v in (r.get("shares") or [])],
            "logsum": [num(b.get("logsum")) for b in betas],
            "se": [num(s.get("logsum")) for s in ses],
            "crash": [num(b.get("log_CRASH_1")) for b in betas],
            "crash_gap": num(r.get("crash_gap")),
            "converged": r.get("converged"),
            "note": (r.get("note") or "")[:120]}


def z(delta, se_a, se_b):
    d = abs(delta)
    s = np.hypot(se_a, se_b)
    return delta / s if s and np.isfinite(s) else float("nan")


FRAMES = []
for arg in sys.argv[1:]:
    if "=" not in arg:
        raise SystemExit(f"bad argument {arg!r}; want label=MIRROR_DIR,FIXED_DIR")
    label, pair = arg.split("=", 1)
    mir, fix = pair.split(",", 1)
    m_meta, m_rows = load(mir)
    f_meta, f_rows = load(fix)
    if m_meta.get("cache") != f_meta.get("cache"):
        raise SystemExit(
            f"frame {label}: cache mismatch\n  mirror {m_meta.get('cache')}\n"
            f"  fixed  {f_meta.get('cache')}\n"
            "the two runs must share sampled alternatives for a paired diff")
    FRAMES.append({"label": label, "mirror_dir": mir, "fixed_dir": fix,
                   "meta": f_meta, "mirror_cache": m_meta.get("cache"),
                   "b": (arm_b(m_rows), arm_b(f_rows)),
                   "d": (arm_d(m_rows), arm_d(f_rows)),
                   "n_mirror": (m_rows.get("B") or {}).get("n"),
                   "n_fixed": (f_rows.get("B") or {}).get("n")})

print("=" * 78)
print("FRAMES")
for f in FRAMES:
    m = f["meta"]
    print(f"  {f['label']}: cache={m.get('cache')} sample_seed={m.get('sample_seed')}"
          f" restart_seed={m.get('seed')} n={f['n_fixed']}")
    print(f"    mirror: {f['mirror_dir']}")
    print(f"    fixed : {f['fixed_dir']}")

print()
print("=" * 78)
print("ARM B (pooled MNL) - deterministic on a fixed frame")
hdr = f"{'frame':<10}{'handling':<9}{'LL':>12}{'c_logsum':>11}{'SE':>8}{'n':>7}"
print(hdr)
for f in FRAMES:
    for name, b in zip(("mirror", "fixed"), f["b"]):
        if b is None:
            print(f"{f['label']:<10}{name:<9}{'-- arm B absent --':>12}")
            continue
        print(f"{f['label']:<10}{name:<9}{b['ll']:>12.4f}{b['logsum']:>11.4f}"
              f"{b['se']:>8.4f}{str(b['n']):>7}")

print()
print("ARM D (latent class, K=2)")
print(f"{'frame':<10}{'handling':<9}{'LL':>12}{'shares':>18}"
      f"{'cls0 logsum':>13}{'cls1 logsum':>13}{'gap':>7}")
for f in FRAMES:
    for name, d in zip(("mirror", "fixed"), f["d"]):
        if d is None:
            print(f"{f['label']:<10}{name:<9}{'-- arm D absent --':>12}")
            continue
        sh = "/".join(f"{v:.4f}" for v in d["shares"]) or "?"
        l0 = d["logsum"][0] if d["logsum"] else float("nan")
        l1 = d["logsum"][1] if len(d["logsum"]) > 1 else float("nan")
        print(f"{f['label']:<10}{name:<9}{d['ll']:>12.4f}{sh:>18}"
              f"{l0:>13.4f}{l1:>13.4f}{d['crash_gap']:>7.3f}")

print()
print("=" * 78)
print("PAIRED SHIFT  fixed - mirror, within frame")
print(f"{'frame':<10}{'arm':<10}{'delta':>12}{'z':>8}{'dLL':>11}")
deltas = {"B": [], "D0": [], "D1": []}
for f in FRAMES:
    bm, bf = f["b"]
    if bm and bf:
        dl = bf["logsum"] - bm["logsum"]
        deltas["B"].append(dl)
        print(f"{f['label']:<10}{'B':<10}{dl:>+12.4f}"
              f"{z(dl, bm['se'], bf['se']):>8.2f}{bf['ll'] - bm['ll']:>+11.4f}")
    dm, df = f["d"]
    if dm and df and len(dm["logsum"]) == len(df["logsum"]) == 2:
        for i, key in enumerate(("D0", "D1")):
            dl = df["logsum"][i] - dm["logsum"][i]
            deltas[key].append(dl)
            print(f"{f['label']:<10}{key:<10}{dl:>+12.4f}"
                  f"{z(dl, dm['se'][i], df['se'][i]):>8.2f}"
                  f"{df['ll'] - dm['ll']:>+11.4f}")

print()
print("SUMMARY (n frames = %d)" % len(FRAMES))
for key, ds in deltas.items():
    if not ds:
        continue
    ds = np.asarray(ds, dtype=float)
    mean = float(ds.mean())
    sd = float(ds.std(ddof=1)) if ds.size > 1 else float("nan")
    se = sd / np.sqrt(ds.size) if ds.size > 1 else float("nan")
    spread = float(ds.max() - ds.min())
    print(f"  {key}: deltas={np.round(ds, 4).tolist()} mean={mean:+.4f} "
          f"sd={sd:.4f} range={spread:.4f} se={se:.4f}")
