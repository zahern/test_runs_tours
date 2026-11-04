#!/usr/bin/env python
"""
power_accessibility.py — power network as an accessibility component (371 SA2).
================================================================================
Takes the power-network outputs (zone_power_371.csv + opendss_results_371.csv)
and builds zone-level accessibility measures that plug into the existing
compute_accessibility.py pipeline (same log1p scale, same attach pattern):

  power_reliability_acc  log1p of violation-free served load
                         (zones whose feeder has no overload/voltage flags
                         score high; mirrors crash_acc as an OD-exposure idea
                         but intrinsic: reliable power attracts activity)
  power_headroom_acc     log1p of spare transformer capacity (kVA headroom)
                         — capacity for growth / new development
  power_cost_acc         log1p of (1 / (1 + upgrade_cost)) scaled — low-cost
                         zones are more accessible for electrification
  ev_hosting_acc         log1p of EV hosting capacity
                         (headroom kW + existing n_ev_charging network)

All coefficients env-tunable. Non-fatal: missing OpenDSS results -> builds
load-only measures from zone_power_371.csv so the chain never breaks.

Use:
  from power_accessibility import compute_power_accessibility
  pow_acc = compute_power_accessibility(save_path='saved_data/power/power_accessibility_371.csv')
  # then merge with transport accessibility on zone_id:
  from power_accessibility import attach_to_accessibility
  full = attach_to_accessibility(pd.read_csv('saved_data/accessibility_371.csv'), pow_acc)
"""
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("power_acc")

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("POWER_OUTDIR", os.path.join(_HERE, "saved_data", "power"))
ZONE_POWER = os.path.join(OUTDIR, "zone_power_371.csv")
OPENDSS_RESULTS = os.path.join(OUTDIR, "opendss_results_371.csv")

POWER_ACCESSIBILITY_SPEC = {
    "power_reliability_acc": "violation-free served load",
    "power_headroom_acc": "spare transformer headroom (growth capacity)",
    "power_cost_acc": "low network upgrade cost (electrification readiness)",
    "ev_hosting_acc": "EV charging hosting capacity",
}


def _w(name, default):
    return float(os.environ.get(name, default))


def compute_power_accessibility(zone_power_path=ZONE_POWER,
                                opendss_path=OPENDSS_RESULTS,
                                save_path=None):
    """Build the four power accessibility columns, indexed by zone_id."""
    if not os.path.exists(zone_power_path):
        raise SystemExit(f"FATAL: {zone_power_path} missing — "
                         "run build_power_network_371.py first")
    zp = pd.read_csv(zone_power_path)
    zid = "SA2_MAIN16" if "SA2_MAIN16" in zp.columns else zp.columns[0]

    if os.path.exists(opendss_path):
        op = pd.read_csv(opendss_path)
        df = zp.merge(op, on=zid, how="left", suffixes=("", "_od"))
    else:
        log.warning("OpenDSS results missing — load-only power accessibility")
        df = zp.copy()
        df["n_violations"] = 0.0
        df["upgrade_cost_aud"] = 0.0
        df["headroom_frac"] = 0.3

    def _col(name, default):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    peak = _col("peak_kw", 0.0).clip(lower=0)
    nviol = _col("n_violations", 0.0).clip(lower=0)
    head = _col("headroom_frac", 0.3)
    cost = _col("upgrade_cost_aud", 0.0).clip(lower=0)
    nev = _col("n_ev_charging", 0.0).clip(lower=0)

    # reliability: served load discounted per violation (env-tunable penalty)
    pen = _w("POWER_VIOLATION_PENALTY", 0.5)
    served = peak * np.exp(-pen * nviol)
    reliability = np.log1p(np.maximum(served, 0.0) / _w("POWER_SCALE_KW", 1000.0))

    # headroom: spare kVA (negative headroom -> ~0 contribution)
    headroom_kw = np.maximum(head, 0.0) * peak
    headroom = np.log1p(headroom_kw / _w("POWER_SCALE_KW", 1000.0))

    # cost: low upgrade cost -> high accessibility (inverse scale)
    cost_acc = np.log1p(1.0 / (1.0 + cost / _w("POWER_COST_SCALE_AUD", 500000.0)))

    # EV hosting: existing chargers + headroom convertible to 11 kW ports
    ports = nev + np.maximum(head, 0.0) * peak / 11.0
    ev_host = np.log1p(np.maximum(ports, 0.0) / _w("POWER_EV_SCALE", 10.0))

    zone_ids = pd.to_numeric(df[zid], errors="coerce").dropna().astype(int).values
    # NOTE: np.asarray() is load-bearing — these are Series on df's RangeIndex
    # and acc is indexed by SA2 code; passing Series would silently reindex to
    # all-NaN (the 25375547 bug: min=nan mean=nan max=nan on every measure).
    acc = pd.DataFrame({
        "power_reliability_acc": np.asarray(reliability, dtype=float),
        "power_headroom_acc": np.asarray(headroom, dtype=float),
        "power_cost_acc": np.asarray(cost_acc, dtype=float),
        "ev_hosting_acc": np.asarray(ev_host, dtype=float),
    }, index=zone_ids)
    acc.index.name = "zone_id"
    for c in acc.columns:
        v = acc[c].values
        print(f"  {c:22s}: min={v.min():.3f} mean={v.mean():.3f} max={v.max():.3f}")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        acc.to_csv(save_path)
        print(f"[PowerAccessibility] saved -> {save_path}")
    return acc


def attach_to_accessibility(access_df, power_df):
    """Left-join power measures onto a transport accessibility table."""
    a = access_df.copy()
    key = "zone_id" if "zone_id" in a.columns else a.index.name
    p = power_df.reset_index() if power_df.index.name == "zone_id" else power_df.copy()
    if key and key in a.columns and key in p.columns:
        a = a.merge(p, on=key, how="left")
    else:
        a = a.join(power_df, how="left")
    for c in POWER_ACCESSIBILITY_SPEC:
        if c in a.columns:
            med = a[c].median()
            a[c] = a[c].fillna(med if not pd.isna(med) else 0.0)
    print(f"[PowerAccessibility] attached {len(POWER_ACCESSIBILITY_SPEC)} power measures")
    return a


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "power_accessibility_371.csv"))
    ap.add_argument("--merge-with", default=None,
                    help="optional transport accessibility CSV to merge into")
    ap.add_argument("--merged-out", default=os.path.join(
        OUTDIR, "accessibility_371_with_power.csv"))
    args = ap.parse_args()
    acc = compute_power_accessibility(save_path=args.out)
    if args.merge_with and os.path.exists(args.merge_with):
        base = pd.read_csv(args.merge_with)
        full = attach_to_accessibility(base, acc)
        full.to_csv(args.merged_out, index=False)
        print(f"Wrote {args.merged_out}")


if __name__ == "__main__":
    main()
