"""Report the nest-scale (Mu) and sign-bound status of the E1/E2 Stage-5 models.

Reads stage5_models_<ARM>_coefficients.csv (unpickling the live models needs the
cluster's larch build). For each segment: every Mu value, whether it is holdfast
(fixed) or free, the free-parameter count, and any time coefficient sitting on
its sign bound with a |t| too small to identify it.

This is what makes two arms comparable. A Mu on the 0.10 floor / 1.00 cap, or a
time coefficient parked on the <= 0 bound at -0.001, is a censored,
unidentified value: an LL gap measured against it is not a specification gain.

Usage:  python _exv_mu_check.py runs/exampville_s5signgate2_20261102 [more dirs]
"""
import os
import sys

import numpy as np
import pandas as pd

MUS = ("Mu_Motor", "Mu_NonMotor", "Mu_Transit", "Mu_Car")
FLOOR = 0.10
CAP = 1.0
TOL = 1e-3
BOUND_EPS = 2e-3
TMIN = 1.645


def _signs(stage):
    try:
        from ga_constraints import sign_constraints as _sc
        return _sc(stage) or {}
    except Exception:
        return {}


def sign_censored(g, stage):
    """Coefficients parked on their declared sign bound that the sample did not
    locate (|t| below the pruning cut).

    Two sources, so an older run dir without the flag column still reports
    honestly: the `sign_censored` column written by the sign gate, and the
    bound + |t| pattern re-derived from the table itself.
    """
    signs = _signs(stage)
    flagged = set()
    if "sign_censored" in g.columns:
        flagged = {str(p) for p, f in zip(g["parameter"], g["sign_censored"])
                   if str(f).strip().lower() in ("true", "1")}
    out, seen = [], set()
    for _, r in g.iterrows():
        p = r["parameter"]
        v = float(r["value"])
        t = float(r["tstat"]) if pd.notna(r["tstat"]) else float("nan")
        if p in flagged:
            out.append((p, v, t))
            seen.add(p)
            continue
        if not signs.get(p) or bool(r["holdfast"]):
            continue
        if abs(v) <= BOUND_EPS and (not np.isfinite(t) or abs(t) < TMIN):
            out.append((p, v, t))
            seen.add(p)
    return out


def report(run_dir, arm):
    path = os.path.join(run_dir, f"stage5_models_{arm}_coefficients.csv")
    if not os.path.exists(path):
        print(f"  {arm}: no coefficients CSV yet")
        return
    df = pd.read_csv(path)
    for mtype, stage in (("mode", "stage5_mode"), ("dest", "stage5_dest")):
        sub = df[df["model_type"] == mtype]
        for seg, g in sub.groupby("segment", sort=False):
            n_free = int((~g["holdfast"].astype(bool)).sum())
            ll = float(g["loglike"].iloc[0])
            parts = []
            if mtype == "mode":
                for _, r in g[g["parameter"].isin(MUS)].iterrows():
                    val = float(r["value"])
                    fixed = bool(r["holdfast"])
                    tag = "fixed" if fixed else "FREE"
                    note = ""
                    if not fixed:
                        if val <= FLOOR + TOL:
                            note = "  <-- AT FLOOR"
                        elif val >= CAP - TOL:
                            note = "  <-- AT CAP"
                    parts.append(f"{r['parameter']}={val:.4f} ({tag}){note}")
            cens = sign_censored(g, stage)
            if cens:
                parts.append("SIGN-CENSORED: " + ", ".join(
                    f"{p}={v:+.4f} (|t|={t:+.2f})" for p, v, t in cens))
            print(f"  {arm} [{mtype} {seg:8s}] LL={ll:12.4f} params={len(g):3d} "
                  f"free={n_free:3d} | " + ("; ".join(parts) if parts else "-"))


def main(dirs):
    for d in dirs:
        print(f"\n=== {os.path.basename(os.path.normpath(d))}")
        for arm in ("E1", "E2"):
            report(d, arm)


if __name__ == "__main__":
    main(sys.argv[1:])
