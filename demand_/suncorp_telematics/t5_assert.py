"""
T5 assertions, from the OUTPUT ARTEFACTS rather than the PBS log.

Same reason as T4: job 26077240 fitted all five arms and then died at
`ls -t .../t5shape.o*` -> "FAIL: no output file", so zero assertions ran
and the job exited 1 with no verdict. Five arms x 22 min of cluster time
produced nothing checkable.

Reads results_tests/t5_*/game_day_holdout_metrics.json, which each arm
writes itself, and checks the things that were actually wrong last time:
  - the exposure coefficient is recorded (base arm was missing it)
  - the exposure basis actually produced columns
  - ZI role 6 is REACHABLE, not merely requested
  - calibration / per-class uplift / binomial are not silently empty
  - and prints the LRT-vs-magnitude and holdout tables for judgement
"""
import json
import os
import sys

FAIL, OK = [], []


def chk(cond, label, detail=""):
    (OK if cond else FAIL).append(label)
    print("  %-5s %-54s %s"
          % ("[ok]" if cond else "[FAIL]", label, detail))


ARMS = [("base", "t5_base", "none"), ("hinge", "t5_hinge", "hinge"),
        ("hw", "t5_hw", "linear+hw"),
        ("hinge+hw", "t5_hingehw", "hinge+hw"),
        ("zi", "t5_zi", "none")]

print("=" * 84)
print("T5 assertions, from artefacts")
print("=" * 84)

rows = []
root = sys.argv[1] if len(sys.argv) > 1 else "."
for arm, dirname, basis in ARMS:
    d = os.path.join(root, dirname)
    p = os.path.join(d, "game_day_holdout_metrics.json")
    if not os.path.exists(p):
        chk(False, "%s metrics json exists" % dirname, d)
        rows.append(None)
        continue
    with open(p) as fh:
        rows.append(json.load(fh))

print()
print("per-arm summary:")
hdr = ("%-9s %9s %9s %8s %-16s %-11s %-8s %-8s"
       % ("arm", "beta", "dev%", "chi2", "calibration", "shape", "zi", "sp"))
print(hdr)
print("-" * len(hdr))
sp0 = None
for (arm, dirname, basis), r in zip(ARMS, rows):
    if r is None:
        print("%-9s %s" % (arm, "  (missing)"))
        continue
    a = r.get("audit") or {}
    rep = a.get("exposure_coef_report") or {}
    lrt = a.get("exposure_lrt") or {}
    bi = a.get("binomial") or {}
    sp = (r.get("metrics") or {}).get("bin_spearman")
    if arm == "base" and sp is not None:
        sp0 = sp
    dev = rep.get("rel_deviation_pct")
    print("%-9s %9s %9s %8s %-11s %-11s %-8s %-6s"
          % (arm,
             ("%.4f" % rep["beta"]) if rep.get("available") else "NONE",
             ("%.2f" % dev) if dev is not None else "-",
             ("%.2f" % lrt["lr_stat"]) if "lr_stat" in lrt else "-",
             (a.get("exposure_calibration") or "-")[:11],
             (a.get("exposure_basis") or "-")[:11],
             ("%d/%d" % (len(a.get("zi_reachable") or []),
                         len(a.get("zi_candidates") or [])))
             if a.get("zi_requested") else "-",
             ("%.5f" % sp) if sp is not None else "-"))

print()
print("the checks that were missing last time:")
for (arm, dirname, basis), r in zip(ARMS, rows):
    if r is None:
        continue
    a = r.get("audit") or {}
    rep = a.get("exposure_coef_report") or {}
    chk(rep.get("available"),
        "%-9s records the exposure coefficient" % arm,
        rep.get("reason", "") if not rep.get("available")
        else "beta=%.4f dev=%.2f%% -> %s"
             % (rep["beta"], rep.get("rel_deviation_pct", float("nan")),
                rep.get("verdict", "n/a")))
    if basis != "none":
        cols = a.get("exposure_shape_cols") or []
        chk(len(cols) > 0, "%-9s exposure shape produced columns" % arm,
            "%d cols" % len(cols))
    if a.get("zi_requested"):
        reach = a.get("zi_reachable") or []
        cand = a.get("zi_candidates") or []
        chk(len(reach) > 0,
            "%-9s ZI role 6 is REACHABLE (not merely requested)" % arm,
            "%d/%d" % (len(reach), len(cand)))
    cal = a.get("exposure_calibration")
    chk(cal in ("on", "off"),
        "%-9s exposure calibration is a definite state" % arm, repr(cal))
    bi = a.get("binomial")
    if bi is not None:
        chk(bi.get("available"), "%-9s binomial propensity available" % arm,
            bi.get("reason", "available") if not bi.get("available")
            else "%d trials" % bi.get("n_trials", 0))

print()
print("holdout vs base:")
for (arm, dirname, basis), r in zip(ARMS, rows):
    if r is None:
        continue
    sp = (r.get("metrics") or {}).get("bin_spearman")
    if sp is None or sp0 is None:
        continue
    d = 0.0 if arm == "base" else 100 * (sp - sp0) / sp0
    print("  %-9s %.5f  %+.1f%%" % (arm, sp, d))

print()
print("=" * 84)
print("passed: %d   failed: %d" % (len(OK), len(FAIL)))
for f in FAIL:
    print("  FAILED: %s" % f)
print("=" * 84)
sys.exit(1 if FAIL else 0)
