#!/usr/bin/env python
"""
sd_hybrid_371.py — system-dynamics hybrid layer for the SEQ 371 ABM.
================================================================================
Paper anchor: Nguyen, Howick & Megiddo (2024), "A framework for
conceptualising hybrid system dynamics and agent-based simulation models",
EJOR 315(3):1153-1166 — hybrid SD+ABM couplings need explicit aggregation /
disaggregation interfaces (what crosses, which direction, at what resolution,
how feedback closes), not ad-hoc loops.

This module is that SD layer. The ABM (tour pipeline / LUTI scenarios) stays
the agent-level engine; here three aggregate STOCKS evolve on the slow
5-year horizon with delays, driven BY the ABM and feeding back INTO it:

  congestion   ABM VMT ──aggregate──▶ v/c ──BPR+MSA delay──▶ speed factor ──▶
               next-horizon travel times (cf. DTA feedback, but stock-based)
  ev_share     Bass diffusion ──▶ fleet fuel cost/km ──▶ generalised auto cost
               (MBUF/telecommute scenarios shift adoption; adoption shifts cost)
  (car ownership / crowding reserved as extensions — see interface table)

Interface table ( Nguyen et al.'s "what crosses the interface" ) is emitted
as CSV alongside the stocks trajectory, for direct paper inclusion.

Pure pandas/numpy. Deterministic. Offline.

Run:  python sd_hybrid_371.py [--travel saved_data/luti/scenario_travel.csv]
Env:  SD_BASS_P, SD_BASS_Q, SD_BPR_ALPHA, SD_BPR_BETA, SD_MSA_LAMBDA,
      EV_COST_RATIO, LUTI_OUTDIR
Outputs: saved_data/luti/sd_stocks.csv, saved_data/luti/sd_interfaces.csv
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("sdhybrid")

_HERE = os.path.dirname(os.path.abspath(__file__))
LUTI = os.environ.get("LUTI_OUTDIR", os.path.join(_HERE, "saved_data", "luti"))

P = float(os.environ.get("SD_BASS_P", 0.003))   # Bass innovation (SEQ placeholder)
Q = float(os.environ.get("SD_BASS_Q", 0.38))    # Bass imitation (SEQ placeholder)
ALPHA = float(os.environ.get("SD_BPR_ALPHA", 1.0))
BETA = float(os.environ.get("SD_BPR_BETA", 2.0))
LAMBDA = float(os.environ.get("SD_MSA_LAMBDA", 0.5))  # MSA smoothing (delay)
EV_COST_RATIO = float(os.environ.get("EV_COST_RATIO", 0.3))  # elec vs petrol /km
EV_SHARE_2025 = 0.02


def interface_table():
    """Nguyen et al. hybrid-interface spec: what crosses ABM<->SD, how."""
    rows = [
        {"from_model": "ABM (tour pipeline)", "to_model": "SD congestion stock",
         "variable": "annual VMT", "aggregation": "sum over agents x 365",
         "resolution": "5-year horizon step", "direction": "ABM->SD",
         "feedback_closed": "yes (via speed factor)"},
        {"from_model": "SD congestion stock", "to_model": "ABM (travel)",
         "variable": "network speed factor", "aggregation": "BPR(v/c) + MSA delay",
         "resolution": "5-year horizon step", "direction": "SD->ABM",
         "feedback_closed": "yes (scales VHT / travel times)"},
        {"from_model": "SD EV stock", "to_model": "ABM (cost)",
         "variable": "fleet fuel cost multiplier", "aggregation": "Bass share x cost ratio",
         "resolution": "5-year horizon step", "direction": "SD->ABM",
         "feedback_closed": "yes (generalised auto cost)"},
        {"from_model": "ABM (scenarios)", "to_model": "SD EV stock",
         "variable": "policy shifter (TELE/MBUF)", "aggregation": "scenario multiplier on q",
         "resolution": "scenario constant", "direction": "ABM->SD",
         "feedback_closed": "yes (adoption responds to policy)"},
        {"from_model": "ABM (synth pop)", "to_model": "SD (reserved)",
         "variable": "car ownership / PT crowding", "aggregation": "TBD",
         "resolution": "-", "direction": "-",
         "feedback_closed": "no (documented extension)"},
    ]
    return pd.DataFrame(rows)


def _basscup(t, p=P, q=Q):
    """Bass cumulative adoption at (continuous) time t (years since 2025)."""
    e = np.exp(-(p + q) * t)
    return (1.0 - e) / (1.0 + (q / max(p, 1e-9)) * e)


def sd_factors(scen, year, vmt, cap, sf_raw_prev=None):
    """Per-step SD factors, sharing formulas with run_sd (single source).

    scen/year/vmt/cap: scenario, horizon year, annual VMT, network capacity.
    sf_raw_prev: smoothed raw speed factor from the previous step (None on the
    first step of a trajectory). Returns (speed_factor, cost_mult, ev_share,
    sf_raw_new), all relative to BASE-2025 (first step of any trajectory is
    exactly 1.0) so observed base-year times/costs are never rescaled.
    """
    t = float(year) - 2025.0
    q = Q * (1.10 if scen == "TELE" else 0.95 if scen == "MBUF" else 1.0)
    vc = float(vmt) / max(cap, 1e-9)
    raw = 1.0 / (1.0 + ALPHA * vc ** BETA)
    base_raw = 1.0 / (1.0 + ALPHA * 0.7 ** BETA)
    if sf_raw_prev is None:
        sf_raw_new, sf_rel = raw, 1.0
    else:
        sf_raw_new = LAMBDA * raw + (1.0 - LAMBDA) * sf_raw_prev
        sf_rel = sf_raw_new / base_raw
    ev = EV_SHARE_2025 + (1.0 - EV_SHARE_2025) * _basscup(t, Q=q)
    ev = float(np.clip(ev, 0.0, 0.95))
    cm = 1.0 - ev * (1.0 - EV_COST_RATIO)
    cm0 = 1.0 - EV_SHARE_2025 * (1.0 - EV_COST_RATIO)
    return round(sf_rel, 4), round(cm / cm0, 4), round(ev, 4), sf_raw_new


def run_sd(travel):
    """Advance SD stocks along the scenario_travel trajectory.

    travel: DataFrame with scenario/year/vmt/vht/trips/cost_aud (annual).
    Returns stocks DataFrame (scenario/year/speed_factor/ev_share/cost_mult).
    """
    out = []
    # network capacity proxy: BASE 2025 VMT at v/c = 0.7 (documented assumption)
    base25 = travel[(travel["scenario"] == "BASE") & (travel["year"] == 2025)]
    cap = float(base25["vmt"].iloc[0] / 0.7) if len(base25) else 1e12
    for scen, g in travel.groupby("scenario"):
        g = g.sort_values("year").reset_index(drop=True)
        sf_raw_prev = None
        for _, r in g.iterrows():
            sf, cm, ev, sf_raw_prev = sd_factors(
                scen, int(r["year"]), float(r["vmt"]), cap, sf_raw_prev)
            out.append({"scenario": scen, "year": int(r["year"]),
                        "vc_ratio": round(float(r["vmt"]) / cap, 3),
                        "speed_factor": sf, "ev_share": ev, "cost_mult": cm})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--travel", default=os.path.join(LUTI, "scenario_travel.csv"))
    args = ap.parse_args()
    if not os.path.exists(args.travel):
        raise SystemExit(f"FATAL: {args.travel} missing — run run_luti_scenarios_371.py first")
    stocks = run_sd(pd.read_csv(args.travel))
    os.makedirs(LUTI, exist_ok=True)
    stocks.to_csv(os.path.join(LUTI, "sd_stocks.csv"), index=False)
    interface_table().to_csv(os.path.join(LUTI, "sd_interfaces.csv"), index=False)
    for scen in ("TELE", "MBUF"):
        s45 = stocks[(stocks["scenario"] == scen) & (stocks["year"] == 2045)]
        b45 = stocks[(stocks["scenario"] == "BASE") & (stocks["year"] == 2045)]
        if len(s45) and len(b45):
            log.info("%s 2045 vs BASE: speed_factor %.3f vs %.3f | ev_share %.1f%% | cost_mult %.3f",
                     scen, s45["speed_factor"].iloc[0], b45["speed_factor"].iloc[0],
                     100 * s45["ev_share"].iloc[0], s45["cost_mult"].iloc[0])
    print("Wrote sd_stocks.csv + sd_interfaces.csv")


if __name__ == "__main__":
    main()
